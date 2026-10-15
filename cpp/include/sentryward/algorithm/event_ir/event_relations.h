#pragma once

#include <algorithm>
#include <array>
#include <tuple>
#include <span>
#include <unordered_map>
#include "sentryward/algorithm/event_ir/event_families.h"
#include "sentryward/algorithm/diagnostics/hb_index_census.h"

namespace sentryward::algorithm {

// Borrowed base+delta storage. No concatenation or ownership changes on reads.
template<class T> class RelationSpan {
 public:
  RelationSpan(std::span<const T> base={},std::span<const T> delta={}) : base_(base),delta_(delta) {}
  std::size_t size() const {return base_.size()+delta_.size();}
  const T& operator[](std::size_t i) const {return i<base_.size()?base_[i]:delta_[i-base_.size()];}
  struct Iterator {
    const T* a=nullptr;const T* b=nullptr;std::size_t n=0,i=0;
    using value_type=T;using difference_type=std::ptrdiff_t;using iterator_category=std::forward_iterator_tag;
    const T& operator*() const {return i<n?a[i]:b[i-n];}
    Iterator& operator++(){++i;return *this;}
    Iterator operator++(int){auto old=*this;++*this;return old;}
    bool operator==(const Iterator&) const=default;
  };
  Iterator begin() const{return {base_.data(),delta_.data(),base_.size(),0};}
  Iterator end() const{return {base_.data(),delta_.data(),base_.size(),size()};}
  bool sameStorage(const RelationSpan& other) const {return base_.data()==other.base_.data() && delta_.data()==other.delta_.data();}
 private: std::span<const T> base_,delta_;
};

// Zipped, fan-in and fan-out edges retain their distinct member relations.
class EventRelations final {
 public:
  struct Relation {
    std::size_t source, target, first_source, first_target, count;
    bool source_step, target_step;
  };
  template<class Edges>
  EventRelations(const EventRecords& events, const Edges& edges, bool reverse = false)
      : mutable_layout_(std::make_shared<EventFamilies>(events)), layout_(mutable_layout_),
        edge_count_(edges.size()), reverse_(reverse) {
    HbIndexCensus("relations.begin",this,layout_->families().size());
    std::size_t columns=0,logical=0;
    {
    std::unordered_map<std::uint64_t,std::size_t> tails;
    std::size_t emitted=0;
    const auto collect=[&](Relation r) {
      ++emitted;
      const auto key=(std::uint64_t(r.source)<<32)|r.target;
      if(auto found=tails.find(key);found!=tails.end()) {
        auto& previous=relations_[found->second];
        if(join(previous,r))return;
        if(join(r,previous)){previous=r;return;}
        found->second=relations_.size();
      } else tails.emplace(key,relations_.size());
      relations_.push_back(r);
    };
    const auto add = [&](const auto& edge, const auto& delta, auto, auto, std::size_t count) {
      ++columns;logical+=count;
      const auto first_from = reverse ? edge.to : edge.from;
      const auto first_to = reverse ? edge.from : edge.to;
      const auto ds = delta[reverse ? 1 : 0], dt = delta[reverse ? 0 : 1];
      emitColumn(first_from.value(),ds,first_to.value(),dt,count,collect);
    };
    if constexpr (requires { edges.forEachAffineColumn(add); }) edges.forEachAffineColumn(add);
    else for (const auto& edge : edges) add(edge, std::array<std::int64_t, 2>{}, 0, 1, 1);
    HbIndexCensus("relations.emitted",this,emitted);
    }
    HbIndexCensus("relations.columns",this,columns,logical);
    HbIndexCensus("relations.residual_singletons",this,residual_singletons_);
    HbIndexCensus("relations.residual_groups",this,residual_groups_,residual_max_);
    HbIndexVector("relations.raw",this,relations_);
    std::sort(relations_.begin(), relations_.end(), [](const auto& a, const auto& b) {
      return std::tie(a.source, a.target, a.first_source, a.first_target) <
             std::tie(b.source, b.target, b.first_source, b.first_target);
    });
    std::size_t retained=0;
    for (const auto relation : relations_) {
      if (retained) {
        auto& previous = relations_[retained-1];
        if(join(previous,relation))continue;
      }
      relations_[retained++]=relation;
    }
    relations_.resize(retained);
    HbIndexVector("relations.final",this,relations_);
    incoming_.resize(layout_->families().size()); outgoing_.resize(incoming_.size());
    for (std::size_t i = 0; i < relations_.size(); ++i) {
      outgoing_[relations_[i].source].push_back(i);
      incoming_[relations_[i].target].push_back(i);
    }
    HbIndexRows("relations.incoming",this,incoming_);
    HbIndexRows("relations.outgoing",this,outgoing_);
  }
  const std::shared_ptr<const EventFamilies>& layout() const { return layout_; }
  template<class Edges>
  EventRelations(std::shared_ptr<const EventRelations> base,std::size_t events,const Edges& edges)
      :base_(std::move(base)),mutable_layout_(std::make_shared<EventFamilies>(*base_->layout())),
       layout_(mutable_layout_),edge_count_(base_->edge_count_),reverse_(false) {
    if(base_->base_)throw std::logic_error("relation base must be a root snapshot");
    append(events,edges);
  }
  RelationSpan<Relation> relations() const { return {base_?std::span<const Relation>(base_->relations_):std::span<const Relation>{},relations_}; }
  RelationSpan<std::size_t> incoming(std::size_t family) const { return {base_ && family<base_->incoming_.size()?std::span<const std::size_t>(base_->incoming_[family]):std::span<const std::size_t>{},family<incoming_.size()?std::span<const std::size_t>(incoming_[family]):std::span<const std::size_t>{}}; }
  RelationSpan<std::size_t> outgoing(std::size_t family) const { return {base_ && family<base_->outgoing_.size()?std::span<const std::size_t>(base_->outgoing_[family]):std::span<const std::size_t>{},family<outgoing_.size()?std::span<const std::size_t>(outgoing_[family]):std::span<const std::size_t>{}}; }
  const std::vector<Relation>& ownedRelations() const {return relations_;}
  // Keep the immutable base relations; inspect only appended HB edges.
  template<class Edges>
  void append(std::size_t event_count,const Edges& edges) {
    if(edges.size()<edge_count_)throw std::logic_error("relation prefix was removed");
    if(mutable_layout_.use_count()>2) {
      mutable_layout_=std::make_shared<EventFamilies>(*mutable_layout_);layout_=mutable_layout_;
    }
    mutable_layout_->appendSingletons(event_count);
    incoming_.resize(layout_->families().size());outgoing_.resize(incoming_.size());
    const auto emit=[&](Relation relation) {
      const auto sf=relation.source,tf=relation.target;
      if(!relations_.empty()) {
        auto& last=relations_.back();
        if(join(last,relation))return;
      }
      const auto id=(base_?base_->relations_.size():0)+relations_.size();
      relations_.push_back(relation);
      outgoing_[sf].push_back(id);incoming_[tf].push_back(id);
    };
    const auto column=[&](const auto& edge,const auto& delta,auto,auto,std::size_t count) {
      emitColumn((reverse_?edge.to:edge.from).value(),delta[reverse_?1:0],
          (reverse_?edge.from:edge.to).value(),delta[reverse_?0:1],count,emit);
    };
    if constexpr(requires{edges.forEachAffineColumnAfter(edge_count_,column);})edges.forEachAffineColumnAfter(edge_count_,column);
    else for(auto i=edge_count_;i<edges.size();++i)column(edges[i],std::array<std::int64_t,2>{},i,1,1);
    edge_count_=edges.size();
  }
  std::pair<std::vector<std::size_t>, std::vector<std::size_t>> potentialParallelFamilies() const {
    if(base_)throw std::logic_error("parallel-edge validation requires a root relation table");
    std::vector<std::size_t> sources, targets;
    const auto overlap = [](std::size_t a, bool sa, std::size_t na,
                            std::size_t b, bool sb, std::size_t nb) {
      return a < b + (sb ? nb : 1) && b < a + (sa ? na : 1);
    };
    for (std::size_t begin = 0; begin < relations_.size();) {
      const auto& first = relations_[begin];
      auto end = begin + 1;
      while (end < relations_.size() && relations_[end].source == first.source &&
             relations_[end].target == first.target) ++end;
      bool duplicate = false;
      for (auto i = begin; i < end && !duplicate; ++i) {
        const auto& a = relations_[i];
        duplicate = a.count > 1 && !a.source_step && !a.target_step;
        for (auto j = i + 1; j < end && !duplicate; ++j) {
          const auto& b = relations_[j];
          if (b.first_source >= a.first_source + (a.source_step ? a.count : 1)) break;
          duplicate = overlap(a.first_source, a.source_step, a.count, b.first_source, b.source_step, b.count) &&
                      overlap(a.first_target, a.target_step, a.count, b.first_target, b.target_step, b.count);
        }
      }
      if (duplicate) { sources.push_back(first.source); targets.push_back(first.target); }
      begin = end;
    }
    const auto unique = [](auto& values) { std::sort(values.begin(), values.end()); values.erase(std::unique(values.begin(), values.end()), values.end()); };
    unique(sources); unique(targets);
    return {std::move(sources), std::move(targets)};
  }
 private:
  static bool join(Relation& a,const Relation& b) {
    if(a.source!=b.source || a.target!=b.target)return false;
    const auto ds=std::int64_t(b.first_source)-std::int64_t(a.first_source);
    const auto dt=std::int64_t(b.first_target)-std::int64_t(a.first_target);
    const bool ss=ds==std::int64_t(a.count),ts=dt==std::int64_t(a.count);
    if(!(ss||ds==0) || !(ts||dt==0) || !(ss||ts) ||
       (a.count!=1 && (a.source_step!=ss || a.target_step!=ts)) ||
       (b.count!=1 && (b.source_step!=ss || b.target_step!=ts)))return false;
    a.source_step=ss;a.target_step=ts;a.count+=b.count;return true;
  }
  // Join endpoint progressions in their common edge-iterator coordinate.
  // Storage column interleaving is partitioned by residues, not by members.
  template<class Emit> void emitColumn(std::int64_t from,std::int64_t ds,
      std::int64_t to,std::int64_t dt,std::size_t count,Emit emit) {
    if(!count)return;
    if(ds<0 && dt<=0) {from+=ds*std::int64_t(count-1);to+=dt*std::int64_t(count-1);ds=-ds;dt=-dt;}
    if(dt<0 && ds==0) {to+=dt*std::int64_t(count-1);dt=-dt;}
    if(ds>=0 && dt>=0) {
      layout_->intersect({std::uint64_t(from),std::uint64_t(ds),count},[&](auto sf,auto s) {
        layout_->intersect({std::uint64_t(to+dt*std::int64_t(s.left)),std::uint64_t(dt)*s.left_step,s.count},[&](auto tf,auto t) {
          const auto sm=s.right+s.right_step*t.left;
          const auto ss=s.right_step*t.left_step,ts=t.right_step;
          if(t.count==1 || (ss<=1 && ts<=1)) {
            emit(Relation{sf,tf,sm,t.right,t.count,ss==1,ts==1});return;
          }
          // The existing scheduler represents contiguous member runs. Genuine
          // strided holes (not storage interleaving) retain the exact old path.
          residual_singletons_+=t.count;++residual_groups_;residual_max_=std::max<std::size_t>(residual_max_,t.count);
          for(std::size_t i=0;i<t.count;++i)emit(Relation{sf,tf,sm+ss*i,t.right+ts*i,1,false,false});
        });
      });
      return;
    }
    residual_singletons_+=count;++residual_groups_;residual_max_=std::max(residual_max_,count);
    for(std::size_t i=0;i<count;++i) {
      const auto [sf,sm]=layout_->locate(EventId(from+ds*std::int64_t(i)));
      const auto [tf,tm]=layout_->locate(EventId(to+dt*std::int64_t(i)));
      emit(Relation{sf,tf,sm,tm,1,false,false});
    }
  }
  std::size_t residual_singletons_=0;
  std::size_t residual_groups_=0,residual_max_=0;
  std::shared_ptr<const EventRelations> base_;
  std::shared_ptr<EventFamilies> mutable_layout_;
  std::shared_ptr<const EventFamilies> layout_;
  std::size_t edge_count_;
  bool reverse_;
  std::vector<Relation> relations_;
  std::vector<std::vector<std::size_t>> incoming_, outgoing_;
};

}  // namespace sentryward::algorithm
