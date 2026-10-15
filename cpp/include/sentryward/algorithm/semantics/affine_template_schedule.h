#pragma once
#include <map>
#include <set>
#include "sentryward/algorithm/base/affine_index.h"
#include "sentryward/algorithm/event_ir/event_families.h"

namespace sentryward::algorithm {

// Kahn scheduling over exact progression domains. Unlike interval-only
// readiness, a relation to every second/fourth member remains one relation.
// Work is proportional to stored relations and their intersections with ready
// domains. Domain refinements cost O(number of resulting residue patterns),
// not O(number of represented events). No reachability queries are needed.
class AffineTemplateSchedule final {
 public:
  struct Batch {std::size_t family;AffineDomain members;EventId first;std::uint64_t stride,count;};
 private:
  // Both domains are stored ascending. reversed retains the zipped mapping
  // from the first source member to the last target member (not all-to-all).
  struct Relation {std::size_t source,target;AffineDomain from,to;bool reversed=false;};
  struct Run {AffineDomain domain;std::size_t pending=0;bool done=false;};
  using ReadyKey=std::tuple<bool,std::uint64_t,std::uint64_t,std::size_t>;
  struct Family {std::vector<Run> runs;std::vector<std::size_t> outgoing;bool urgent=false;std::optional<ReadyKey> ready;std::size_t remaining=0;};
  using Queue=std::set<ReadyKey>;
  std::shared_ptr<const EventFamilies> layout_;
  std::vector<Family> families_;
  std::vector<Relation> relations_;
  Queue normal_,urgent_;
  std::size_t executed_=0,refinements_=0;
  template<class Update> void change(std::size_t family,AffineDomain domain,Update update) {
    auto& runs=families_[family].runs;
    std::vector<Run> next;next.reserve(runs.size()+2);
    for(const auto& run:runs) {
      auto m=IntersectAffine(run.domain,domain);
      if(!m.count){next.push_back(run);continue;}
      for(auto remainder:SubtractAffine(run.domain,domain))next.push_back({remainder,run.pending,run.done});
      Run changed{{run.domain.first+run.domain.step*m.left,
          m.count>1?run.domain.step*m.left_step:1,m.count},run.pending,run.done};
      update(changed);next.push_back(changed);
    }
    if(next.size()>runs.size())refinements_+=next.size()-runs.size();
    // Equal readiness is bookkeeping, not a new semantic thread family. In
    // particular, after fan-in updates restore a uniform state, retain one
    // run instead of leaving singleton boundaries from old pending counts.
    if(!next.empty() && std::all_of(next.begin(),next.end(),[&](const Run& run) {
        return run.pending==next.front().pending && run.done==next.front().done;
      })) {
      next.front().domain={0,1,layout_->families()[family].count};next.resize(1);
    }
    runs=std::move(next);
  }
  void refresh(std::size_t family) {
    auto& f=families_[family];auto& queue=f.urgent?urgent_:normal_;
    if(f.ready)queue.erase(*f.ready);f.ready.reset();
    const auto& shape=layout_->families()[family];
    for(const auto& run:f.runs)if(!run.pending && !run.done) {
      auto first=shape.first.value()+shape.stride*run.domain.first;
      // Scheduling priority is not an added dependency. Prefer an entire
      // ready family, then a large ready domain, before consuming a partial
      // prefix that another ready producer could make whole.
      const ReadyKey candidate{run.domain.count!=f.remaining,UINT64_MAX-run.domain.count,first,family};
      if(!f.ready || candidate<*f.ready)f.ready=candidate;
    }
    if(f.ready)queue.insert(*f.ready);
  }
 public:
  template<class Edges> AffineTemplateSchedule(const EventRecords& events,const Edges& edges)
      :layout_(std::make_shared<EventFamilies>(events)) {
    for(std::size_t i=0;i<layout_->families().size();++i) {
      const auto& f=layout_->families()[i];
      families_.push_back({{{{0,1,f.count},0,false}}, {},
          events.at(f.first.value()).scheduling_class==EventSchedulingClass::kUrgentDerived,{},f.count});
    }
    edges.forEachAffineColumn([&](const auto& first,const auto& delta,auto,auto,std::size_t count) {
      if(!count)return;
      const auto endpoint=[](__int128 value) {
        if(value<0 || value>UINT32_MAX)throw std::invalid_argument("schedule endpoint progression overflows");
        return std::uint64_t(value);
      };
      auto source_first=std::uint64_t(first.from.value()),target_first=std::uint64_t(first.to.value());
      auto source_step=delta[0],target_step=delta[1];
      if(source_step<0) {
        source_first=endpoint(__int128(source_first)+__int128(source_step)*(count-1));
        target_first=endpoint(__int128(target_first)+__int128(target_step)*(count-1));
        if(source_step==INT64_MIN || target_step==INT64_MIN)throw std::invalid_argument("schedule stride overflows");
        source_step=-source_step;target_step=-target_step;
      }
      std::size_t covered=0;
      layout_->intersect({source_first,std::uint64_t(source_step),count},[&](auto sf,auto sm) {
        if(!source_step){sm.count=count;sm.left_step=1;sm.right_step=0;}
        const auto target_start=endpoint(__int128(target_first)+__int128(target_step)*sm.left);
        const __int128 signed_step=__int128(target_step)*sm.left_step;
        const bool reversed=signed_step<0;
        const auto target_end=endpoint(__int128(target_start)+signed_step*(sm.count-1));
        const AffineDomain target{reversed?target_end:target_start,
            std::uint64_t(reversed?-signed_step:signed_step),sm.count};
        layout_->intersect(target,[&](auto tf,auto tm) {
          if(!target.step){tm.count=sm.count;tm.left_step=1;tm.right_step=0;}
          const auto first_source_index=reversed?sm.count-1-(tm.left+tm.left_step*(tm.count-1)):tm.left;
          Relation relation{sf,tf,{sm.right+sm.right_step*first_source_index,sm.right_step*tm.left_step,tm.count},
              {tm.right,tm.right_step,tm.count},reversed};
          families_[sf].outgoing.push_back(relations_.size());relations_.push_back(relation);
          change(tf,relation.to,[&](Run& run){run.pending+=relation.to.step?1:relation.to.count;});
          covered+=tm.count;
        });
      });
      if(covered!=count)throw std::logic_error("missing schedule edge endpoint");
    });
    for(std::size_t i=0;i<families_.size();++i)refresh(i);
  }
  std::size_t familyCount() const{return families_.size();}
  std::size_t relationCount() const{return relations_.size();}
  std::size_t executedCount() const{return executed_;}
  std::size_t domainRefinements() const{return refinements_;}
  const std::shared_ptr<const EventFamilies>& layout() const{return layout_;}
  std::optional<Batch> next(bool urgent) const {
    const auto& queue=urgent?urgent_:normal_;if(queue.empty())return {};
    const auto [partial,inverse_count,id,family]=*queue.begin();const auto& f=layout_->families()[family];
    (void)partial;(void)inverse_count;
    for(const auto& run:families_[family].runs)
      if(!run.pending && !run.done && f.first.value()+f.stride*run.domain.first==id)
        return Batch{family,run.domain,EventId(id),f.stride*run.domain.step,run.domain.count};
    throw std::logic_error("missing ready domain");
  }
  void finish(const Batch& batch) {
    change(batch.family,batch.members,[](Run& run) {
      if(run.pending || run.done)throw std::logic_error("event domain not ready");run.done=true;
    });
    executed_+=batch.count;families_[batch.family].remaining-=batch.count;refresh(batch.family);
    for(auto id:families_[batch.family].outgoing) {
      const auto& r=relations_[id];auto m=IntersectAffine(r.from,batch.members);if(!m.count)continue;
      if(!r.from.step){m.count=r.from.count;m.left_step=1;}
      const auto first_target_index=r.reversed?r.from.count-1-(m.left+m.left_step*(m.count-1)):m.left;
      const AffineDomain target{r.to.first+r.to.step*first_target_index,r.to.step*m.left_step,m.count};
      change(r.target,target,[&](Run& run) {
        const auto decrement=target.step?1:target.count;
        if(run.pending<decrement)throw std::logic_error("domain dependency underflow");run.pending-=decrement;
      });refresh(r.target);
    }
  }
};
} // namespace sentryward::algorithm
