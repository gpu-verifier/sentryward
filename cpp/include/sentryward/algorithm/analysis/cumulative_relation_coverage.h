#pragma once
#include <map>
#include <memory>
#include <set>
#include <span>
#include <tuple>
#include <unordered_map>
#include "sentryward/algorithm/analysis/member_relation.h"

namespace sentryward::algorithm {
// Indexed cumulative coverage. Products partition the source coordinate into
// runs with shared column maps; translated diagonals retain a separate 1-D map.
// Never subtract a diagonal from a product or materialize its member rows.
class CumulativeRelationCoverage {
 public:
  struct Counters {
    std::size_t arrivals=0,changed=0,touched=0,splits=0,row_copies=0,copied_columns=0;
    std::size_t emitted=0,duplicate_arrivals=0,earlier=0,peak_rows=0,peak_columns=0;
    std::size_t query_segments=0,mixed_candidates=0;
    std::size_t singleton_arrivals=0,singleton_emissions=0;
    std::size_t requested_labels=0,state_groups=0,peak_groups=0;
    std::size_t single_row_updates=0,single_row_noops=0,row_update_maps=0;
    std::size_t state_normalizations=0,drain_groupings=0;
    std::size_t last_arrival_hits=0,last_arrival_misses=0;
  };
  struct Work {std::size_t position,end;MemberRelation relation;std::uint64_t labels=1;};
 private:
  struct State {
    struct Group {
      std::size_t known,sent;std::uint64_t labels;
      bool operator==(const Group&) const = default;
    };
    std::vector<Group> groups;
    bool operator==(const State&) const = default;
    std::uint64_t pending() const {
      std::uint64_t mask=0;for(const auto& g:groups)if(g.known<g.sent)mask|=g.labels;return mask;
    }
    bool useful(std::size_t position,std::uint64_t labels,Counters& stats) const {
      for(const auto& g:groups){++stats.state_groups;if(g.known<=position)labels&=~g.labels;if(!labels)return false;}
      return labels!=0;
    }
    void normalize(std::span<Group> entries,Counters& stats) {
      if(entries.size()>1) {
        ++stats.state_normalizations;
        std::sort(entries.begin(),entries.end(),[](const auto& a,const auto& b){return std::pair{a.known,a.sent}<std::pair{b.known,b.sent};});
      }
      groups.clear();
      for(const auto& g:entries)if(g.labels) {
        if(!groups.empty() && groups.back().known==g.known && groups.back().sent==g.sent)groups.back().labels|=g.labels;
        else groups.push_back(g);
      }
    }
    bool advance(std::size_t position,std::uint64_t labels,Counters& stats) {
      // The common one-history state needs neither a temporary group array nor
      // sorting. New labels / partially advanced histories still use the same
      // general state transition below; they must not borrow another label's
      // earlier position or sent frontier.
      if(groups.empty()) {
        groups.push_back({position,SIZE_MAX,labels});
        stats.peak_groups=std::max(stats.peak_groups,std::size_t(1));return true;
      }
      if(groups.size()==1) {
        auto& g=groups.front();++stats.state_groups;
        if(!(labels&~g.labels) && position>=g.known)return false;
        if(labels==g.labels && position<g.known) {
          g.known=position;++stats.earlier;return true;
        }
        if(position==g.known && g.sent==SIZE_MAX) {
          g.labels|=labels;return true;
        }
      }
      std::array<Group,128> next;std::size_t n=0;bool changed=false;
      auto missing=labels;
      for(const auto& g:groups) {
        ++stats.state_groups;const auto selected=g.labels&labels;missing&=~g.labels;
        if(!selected || position>=g.known){next[n++]=g;continue;}
        if(g.labels&~labels)next[n++]={g.known,g.sent,g.labels&~labels};
        next[n++]={position,g.sent,selected};changed=true;++stats.earlier;
      }
      if(missing){next[n++]={position,SIZE_MAX,missing};changed=true;}
      if(changed)normalize(std::span(next.data(),n),stats);
      stats.peak_groups=std::max(stats.peak_groups,groups.size());return changed;
    }
    template<class Visit> void drain(std::uint64_t active,Counters& stats,Visit visit) {
      if(groups.size()==1) {
        auto& g=groups.front();const auto selected=g.labels&active;
        if(!selected || g.known>=g.sent)return;
        if(selected==g.labels) {
          visit(g.known,g.sent,selected);g.sent=g.known;return;
        }
      }
      std::array<Group,128> next;std::size_t n=0;
      for(const auto& g:groups) {
        const auto selected=g.labels&active;
        if(!selected || g.known>=g.sent){next[n++]=g;continue;}
        visit(g.known,g.sent,selected);
        if(g.labels&~active)next[n++]={g.known,g.sent,g.labels&~active};
        next[n++]={g.known,g.known,selected};
      }
      normalize(std::span(next.data(),n),stats);
    }
    bool covers(std::uint64_t selector) const {
      for(const auto& g:groups)selector&=~g.labels;return selector==0;
    }
  };
  struct Columns {
    std::map<std::size_t,State> values{{0,{}}};
    std::set<std::size_t> dirty;
    mutable std::optional<MemberSet> covered_cache;
    mutable std::uint64_t cache_selector=0;
    bool useful(std::size_t position,const MemberSet& mask,Counters& stats,std::uint64_t labels) const {
      bool result=false;
      mask.forEachRun([&](auto a,auto b) {
        if(result)return;
        for(auto it=std::prev(values.upper_bound(a));it!=values.end() && it->first<b;++it) {
          ++stats.touched;if(it->second.useful(position,labels,stats)){result=true;break;}
        }
      });
      return result;
    }
    auto split(std::size_t x,std::size_t extent,Counters& stats) {
      if(x==extent)return values.end();
      auto it=std::prev(values.upper_bound(x));
      if(it->first==x)return it;
      // std::map insertion preserves the source State. Copy its group vector
      // into the new cell once, not via a second owning temporary.
      it=values.emplace_hint(std::next(it),x,it->second);++stats.splits;
      if(it->second.pending())dirty.insert(x);
      return it;
    }
    bool advance(std::size_t position,const MemberSet& mask,Counters& stats,std::uint64_t labels) {
      bool changed=false;
      mask.forEachRun([&](auto a,auto b) {
        // Do not split unchanged coverage on redundant arrivals.
        bool useful=false;
        for(auto it=std::prev(values.upper_bound(a));it!=values.end() && it->first<b;++it) {
          ++stats.touched;if(it->second.useful(position,labels,stats)){useful=true;break;}
        }
        if(!useful)return;
        auto end=split(b,mask.extent(),stats),it=split(a,mask.extent(),stats);
        for(;it!=end;++it) {
          ++stats.touched;auto& state=it->second;
          if(state.advance(position,labels,stats)){dirty.insert(it->first);changed=true;}
        }
        // Local coalescing only: no full-map normalization per arrival.
        it=values.lower_bound(a);if(it!=values.begin())--it;
        while(it!=values.end()) {
          auto next=std::next(it);if(next==values.end() || next->first>b)break;
          if(next->second==it->second){dirty.erase(next->first);values.erase(next);}
          else it=next;
        }
      });
      if(changed)covered_cache.reset();
      stats.peak_columns=std::max(stats.peak_columns,values.size());
      return changed;
    }
    template<class Visit> void drain(std::size_t extent,std::uint64_t active,Counters& stats,Visit visit) {
      for(auto ticket=dirty.begin();ticket!=dirty.end();) {
        const auto a=*ticket;
        auto it=values.find(a);if(it==values.end()){ticket=dirty.erase(ticket);continue;}
        auto& state=it->second;
        auto next=std::next(it);const auto b=next==values.end()?extent:next->first;
        state.drain(active,stats,[&](auto p,auto e,auto labels){visit(a,b,p,e,labels);});
        if(!state.pending())ticket=dirty.erase(ticket);else ++ticket;
      }
    }
    MemberSet covered(std::size_t extent,std::uint64_t selector) const {
      if(covered_cache && cache_selector==selector)return *covered_cache;
      MemberSet result(extent);
      for(auto it=values.begin();it!=values.end();++it)if(it->second.covers(selector)) {
        auto next=std::next(it);result.addRange(it->first,next==values.end()?extent:next->first);
      }
      covered_cache=result;cache_selector=selector;
      return result;
    }
  };
  std::size_t left_extent_=0,right_extent_=0;
  bool initialized_=false;
  std::map<std::size_t,std::shared_ptr<Columns>> rows_;
  std::map<std::int64_t,Columns> diagonals_;
  std::set<std::size_t> dirty_rows_;
  std::set<std::int64_t> dirty_diagonals_;
  mutable std::shared_ptr<CumulativeRelationCoverage> transpose_;
  struct Arrival {MemberRelation relation;std::size_t position;std::uint64_t labels;};
  std::optional<Arrival> last_arrival_;
  auto splitRow(std::size_t x,Counters& stats) {
    if(x==left_extent_)return rows_.end();
    auto it=std::prev(rows_.upper_bound(x));if(it->first==x)return it;
    auto row=it->second;
    it=rows_.emplace_hint(std::next(it),x,std::move(row));++stats.splits;
    if(!it->second->dirty.empty())dirty_rows_.insert(x);
    return it;
  }
  void initialize(const MemberRelation& r) {
    if(!initialized_){left_extent_=r.left.extent();right_extent_=r.right.extent();initialized_=true;}
    if(left_extent_!=r.left.extent() || right_extent_!=r.right.extent())
      throw std::invalid_argument("cumulative relation extent mismatch");
  }
 public:
  std::size_t leftExtent() const {return left_extent_;}
  bool initialized() const {return initialized_;}
  bool pending() const {return !dirty_rows_.empty() || !dirty_diagonals_.empty();}
  bool advance(std::size_t position,const MemberRelation& r,Counters& stats,std::uint64_t labels=1) {
    ++stats.arrivals;stats.requested_labels+=std::popcount(labels);
    if(r.empty() || !labels)return false;initialize(r);
    if(r.left.count()==1)++stats.singleton_arrivals;
    // An advance certifies this exact relation at its incoming position.
    // Known coverage is monotone; draining changes only sent frontiers, so a
    // later arrival of the same relation/covered labels cannot add work.
    // Keep one bounded entry, never references to COW column owners.
    const bool same=last_arrival_ && last_arrival_->relation.diagonal==r.diagonal &&
        last_arrival_->relation.shift==r.shift && last_arrival_->relation.left==r.left &&
        last_arrival_->relation.right==r.right;
    if(same && position>=last_arrival_->position && !(labels&~last_arrival_->labels)) {
      ++stats.last_arrival_hits;++stats.duplicate_arrivals;return false;
    }
    ++stats.last_arrival_misses;
    bool changed=false;
    if(r.diagonal) {
      auto& row=diagonals_[r.shift];changed=row.advance(position,r.left,stats,labels);
      if(changed)dirty_diagonals_.insert(r.shift);
    } else {
      if(rows_.empty())rows_.emplace(0,std::make_shared<Columns>());
      std::size_t a=0,b=0,runs=0;
      r.left.forEachRun([&](auto begin,auto end){a=begin;b=end;++runs;});
      auto single=std::prev(rows_.upper_bound(a));
      auto next=std::next(single);
      if(runs==1 && b<=(next==rows_.end()?left_extent_:next->first)) {
        ++stats.single_row_updates;
        // Query the existing state BEFORE splitting. A redundant interior
        // arrival must not allocate rows only to coalesce them again.
        if(!single->second->useful(position,r.right,stats,labels))++stats.single_row_noops;
        else {
          splitRow(b,stats);single=splitRow(a,stats);
          // Splitting may have introduced a sibling, and copying the whole
          // index also shares rows. Both cases require copy-on-write.
          if(single->second.use_count()!=1) {
            ++stats.row_copies;stats.copied_columns+=single->second->values.size();
            single->second=std::make_shared<Columns>(*single->second);
          }
          changed=single->second->advance(position,r.right,stats,labels);
          if(changed)dirty_rows_.insert(a);
          // Only one old row was modified. New boundaries separate different
          // owners, so no row regrouping is needed on this transition.
        }
      } else {
      // All pieces sharing an immutable column map undergo this identical
      // update once. Copy only when a live sibling still owns the old map.
      struct Update {std::shared_ptr<Columns> row;bool useful;bool applied=false;};
      std::unordered_map<const Columns*,Update> updates;
      std::vector<std::shared_ptr<Columns>> retained;
      auto lookup=[&](const std::shared_ptr<Columns>& row)->Update& {
        const auto* key=row.get();
        if(auto found=updates.find(key);found!=updates.end())return found->second;
        const bool useful=row->useful(position,r.right,stats,labels);
        if(updates.empty())++stats.row_update_maps;
        return updates.emplace(key,Update{row,useful}).first->second;
      };
      r.left.forEachRun([&](auto a,auto b) {
        bool novel=false;
        for(auto row=std::prev(rows_.upper_bound(a));row!=rows_.end() && row->first<b;++row)
          if(lookup(row->second).useful){novel=true;break;}
        if(!novel)return;
        auto end=splitRow(b,stats),it=splitRow(a,stats);
        for(;it!=end;++it) {
          ++stats.touched;auto& update=lookup(it->second);
          if(update.useful && !update.applied) {
            // The update record and this row are the two local owners.
            // Boundaries are already split: any other owner needs the old
            // map. Retain it so the pointer-key cannot be reused this call.
            if(update.row.use_count()>2) {
              ++stats.row_copies;stats.copied_columns+=update.row->values.size();
              retained.push_back(update.row);
              update.row=std::make_shared<Columns>(*update.row);
            }
            update.row->advance(position,r.right,stats,labels);
            update.applied=true;
          }
          if(update.useful){it->second=update.row;dirty_rows_.insert(it->first);changed=true;}
        }
        // Merge rows only by shared identity, never scan or compare full maps.
        it=rows_.lower_bound(a);if(it!=rows_.begin())--it;
        while(it!=rows_.end()) {
          auto next=std::next(it);if(next==rows_.end() || next->first>b)break;
          if(it->second==next->second){
            if(!it->second->dirty.empty())dirty_rows_.insert(it->first);
            dirty_rows_.erase(next->first);rows_.erase(next);
          }
          else it=next;
        }
      });
      }
    }
    if(changed){++stats.changed;transpose_.reset();}else ++stats.duplicate_arrivals;
    if(same) {
      // Retain a certificate for the union only at the weaker position.
      // A newly earlier arrival must not lend its position to older labels.
      if(position==last_arrival_->position)last_arrival_->labels|=labels;
      else {last_arrival_->position=position;last_arrival_->labels=labels;}
    } else if(last_arrival_) {
      // Extents are fixed for this index: reuse dense MemberSet storage rather
      // than allocate two new vectors on each cache miss.
      last_arrival_->relation=r;
      last_arrival_->position=position;last_arrival_->labels=labels;
    } else last_arrival_.emplace(Arrival{r,position,labels});
    stats.peak_rows=std::max(stats.peak_rows,rows_.size()+diagonals_.size());
    return changed;
  }
  void drain(Counters& stats,std::uint64_t active,std::vector<Work>& out) {
    out.clear();
    struct Piece {std::size_t a,b,position,end;std::uint64_t labels;};
    if(dirty_rows_.size()==1) {
      auto it=rows_.find(*dirty_rows_.begin());
      if(it!=rows_.end()) {
        auto next=std::next(it);const auto b=next==rows_.end()?left_extent_:next->first;
        it->second->drain(right_extent_,active,stats,[&](auto x,auto y,auto p,auto e,auto labels) {
          out.push_back({p,e,MemberRelation::Rectangle(MemberRange(left_extent_,it->first,b),
              MemberRange(right_extent_,x,y)),labels});
        });
      }
    } else {
    std::unordered_map<const Columns*,std::vector<Piece>> updates;
    for(auto a:dirty_rows_) {
      auto it=rows_.find(a);if(it==rows_.end())continue;
      auto next=std::next(it);const auto b=next==rows_.end()?left_extent_:next->first;
      auto inserted=updates.try_emplace(it->second.get());
      auto u=inserted.first;
      if(inserted.second)it->second->drain(right_extent_,active,stats,[&](auto x,auto y,auto p,auto e,auto labels){u->second.push_back({x,y,p,e,labels});});
      for(const auto& d:u->second)out.push_back({d.position,d.end,MemberRelation::Rectangle(
          MemberRange(left_extent_,a,b),MemberRange(right_extent_,d.a,d.b)),d.labels});
    }
    }
    std::erase_if(dirty_rows_,[&](auto a){auto it=rows_.find(a);return it==rows_.end() || it->second->dirty.empty();});
    for(auto shift:dirty_diagonals_)diagonals_.at(shift).drain(left_extent_,active,stats,[&](auto a,auto b,auto p,auto e,auto labels) {
      out.push_back({p,e,MemberRelation::Diagonal(MemberRange(left_extent_,a,b),right_extent_,shift),labels});
    });
    std::erase_if(dirty_diagonals_,[&](auto shift){return diagonals_.at(shift).dirty.empty();});
    // Storage runs are not propagation units. Recombine equal column/state
    // pieces into MemberSets before returning to the graph traversal.
    for(bool by_right:{true,false}) {
      if(out.size()<2)break;
      ++stats.drain_groupings;
      std::map<std::tuple<std::size_t,std::size_t,bool,std::int64_t,std::size_t,std::uint64_t>,std::vector<std::size_t>> groups;
      std::vector<Work> combined;
      for(auto& w:out) {
        const auto& key=by_right?w.relation.right:w.relation.left;
        auto& bucket=groups[{w.position,w.end,w.relation.diagonal,w.relation.shift,w.relation.diagonal?0:key.hash(),w.labels}];
        bool found=false;
        for(auto i:bucket) {
          auto& old=combined[i];
          if(w.relation.diagonal || (by_right?old.relation.right:old.relation.left)==key) {
            old.relation.left.unite(w.relation.left);old.relation.right.unite(w.relation.right);found=true;break;
          }
        }
        if(!found){bucket.push_back(combined.size());combined.push_back(std::move(w));}
      }
      out=std::move(combined);
    }
    stats.emitted+=out.size();for(const auto& w:out)if(w.relation.left.count()==1)++stats.singleton_emissions;
  }
  // Convenience for isolated callers. Traversal reuses a query-owned buffer.
  std::vector<Work> drain(Counters& stats,std::uint64_t active=UINT64_MAX) {
    std::vector<Work> out;drain(stats,active,out);return out;
  }
  // Visit the current union, not historical insertions. Used by the lazy
  // transpose index and the deliberately separate legacy A002 adapter.
  template<class Visit> void visit(Visit fn,std::uint64_t selector=1) const {
    for(auto it=rows_.begin();it!=rows_.end();++it) {
      auto next=std::next(it);auto right=it->second->covered(right_extent_,selector);
      if(!right.empty())fn(MemberRelation::Rectangle(MemberRange(left_extent_,it->first,
          next==rows_.end()?left_extent_:next->first),std::move(right)));
    }
    for(const auto& [shift,row]:diagonals_) {
      auto left=row.covered(left_extent_,selector);if(!left.empty())fn(MemberRelation::Diagonal(std::move(left),right_extent_,shift));
    }
  }
  // Transpose labelled coverage by stored cells/groups, never by labels or
  // concrete members. Shared by all request selectors in this target bucket.
  const CumulativeRelationCoverage& transposed() const {
    if(transpose_)return *transpose_;
    auto result=std::make_shared<CumulativeRelationCoverage>();Counters stats;
    for(auto row=rows_.begin();row!=rows_.end();++row) {
      auto nr=std::next(row);const auto end=nr==rows_.end()?left_extent_:nr->first;
      for(auto col=row->second->values.begin();col!=row->second->values.end();++col) {
        auto nc=std::next(col);const auto right_end=nc==row->second->values.end()?right_extent_:nc->first;
        std::uint64_t labels=0;for(const auto& g:col->second.groups)labels|=g.labels;
        if(labels)result->advance(0,MemberRelation::Rectangle(MemberRange(right_extent_,col->first,right_end),
            MemberRange(left_extent_,row->first,end)),stats,labels);
      }
    }
    for(const auto& [shift,row]:diagonals_)for(auto it=row.values.begin();it!=row.values.end();++it) {
      auto next=std::next(it);std::uint64_t labels=0;for(const auto& g:it->second.groups)labels|=g.labels;
      if(labels)result->advance(0,MemberRelation::Diagonal(MemberRange(left_extent_,it->first,
          next==row.values.end()?left_extent_:next->first),right_extent_,shift).transpose(),stats,labels);
    }
    transpose_=std::move(result);return *transpose_;
  }
  std::size_t partitions() const {
    std::size_t n=0;for(const auto& [a,row]:rows_)n+=row->values.size();
    for(const auto& [shift,row]:diagonals_)n+=row.values.size();return n;
  }
  std::optional<std::pair<std::size_t,std::size_t>> uncovered(
      const MemberRelation& query,Counters* stats=nullptr,MemberSet* complete_rows=nullptr,std::uint64_t selector=1) const {
    if(query.empty())return {};
    // A query may use an unpadded public source extent after batch completion.
    auto q=query;
    if(initialized_ && q.left.extent()!=left_extent_)q.left=q.left.shifted(0,left_extent_);
    if(initialized_ && q.right.extent()!=right_extent_)q.right=q.right.shifted(0,right_extent_);
    if(!initialized_)return q.witness();
    if(q.diagonal) {
      MemberSet covered(left_extent_);
      if(auto it=diagonals_.find(q.shift);it!=diagonals_.end())covered=it->second.covered(left_extent_,selector);
      for(auto it=rows_.begin();it!=rows_.end();++it) {
        auto next=std::next(it);auto l=MemberRange(left_extent_,it->first,next==rows_.end()?left_extent_:next->first);
        l=l.intersect(q.left);if(l.empty())continue;
        if(stats)++stats->query_segments;
        covered.unite(l.intersect(it->second->covered(right_extent_,selector).shifted(-q.shift,left_extent_)));
      }
      if(complete_rows)complete_rows->unite(q.left.intersect(covered).shifted(0,complete_rows->extent()));
      auto missing=q.left.subtract(covered);if(missing.empty())return {};
      return std::pair{missing.first(),std::size_t(std::int64_t(missing.first())+q.shift)};
    }
    std::map<std::int64_t,MemberSet> diagonals;
    for(const auto& [shift,row]:diagonals_)diagonals.emplace(shift,row.covered(left_extent_,selector));
    std::optional<std::pair<std::size_t,std::size_t>> witness;
    const auto segment=[&](const MemberSet& domain,const MemberSet& covered) {
      if(domain.empty() || (witness && !complete_rows))return;
      if(stats)++stats->query_segments;
      auto missing=q.right.subtract(covered);
      if(missing.empty()){if(complete_rows)complete_rows->unite(domain.shifted(0,complete_rows->extent()));return;}
      const auto pivot=missing.first();MemberSet candidates(left_extent_);
      for(const auto& [shift,left]:diagonals) {
        auto i=__int128(pivot)-shift;
        if(i>=0 && i<left_extent_ && domain.contains(std::size_t(i)) && left.contains(std::size_t(i)))candidates.set(std::size_t(i));
      }
      auto outside=domain.subtract(candidates);
      if(!outside.empty() && !witness)witness=std::pair{outside.first(),pivot};
      if(witness && !complete_rows)return;
      // At most one candidate per distinct shift. Not a loop over all rows.
      candidates.forEach([&](auto i) {
        if(witness && !complete_rows)return;
        if(stats)++stats->mixed_candidates;
        MemberSet diagonal_right(right_extent_);
        for(const auto& [shift,left]:diagonals)if(left.contains(i)) {
          auto j=__int128(i)+shift;if(j>=0 && j<right_extent_)diagonal_right.set(std::size_t(j));
        }
        auto rest=missing.subtract(diagonal_right);
        if(rest.empty()){if(complete_rows)complete_rows->set(i);}
        else if(!witness)witness=std::pair{i,rest.first()};
      });
    };
    if(rows_.empty())segment(q.left,MemberSet(right_extent_));
    else q.left.forEachRun([&](auto a,auto b) {
      if(witness && !complete_rows)return;
      auto it=std::prev(rows_.upper_bound(a));
      for(;it!=rows_.end() && it->first<b;++it) {
        auto next=std::next(it);auto end=next==rows_.end()?left_extent_:next->first;
        segment(MemberRange(left_extent_,std::max(a,it->first),std::min(b,end)),it->second->covered(right_extent_,selector));
        if(witness && !complete_rows)break;
      }
    });
    return witness;
  }
};
} // namespace sentryward::algorithm
