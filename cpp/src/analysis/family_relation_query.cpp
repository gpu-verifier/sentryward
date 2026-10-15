#include "sentryward/algorithm/diagnostics/development_trace.h"
#include "sentryward/algorithm/analysis/family_relation_query.h"
#include "sentryward/algorithm/diagnostics/performance_profile.h"
#include <deque>
#include <unordered_map>
#include <type_traits>
#include <map>
#include <cstdlib>

namespace sentryward::algorithm {
FamilyRelationQuery::FamilyRelationQuery(const HappensBeforeGraph& graph)
    :graph_(graph),topology_owner_(graph.relationStorage()),topology_(*topology_owner_),chains_(graph,topology_) {
  scratch_slots_.resize(chains_.chainCount(),SIZE_MAX);
  const auto* census=sentryward::debug::environment("SENTRYWARD_HB_INDEX_CENSUS");
  visit_census_=census && std::string_view(census)!="0";
  if(visit_census_)for(auto& visits:last_visit_batch_)visits.resize(topology_.relations().size());
  for(const auto& f:families().families())
    bounds_.push_back(graph.traceOrder()?graph.traceOrder()->rankBounds(f.first,f.stride,f.count):
        std::pair<std::size_t,std::size_t>{0,SIZE_MAX});
  if(graph.traceOrder()) {
    ProfileScope profile("memory.family.alternate_order");
    alternate_=std::make_unique<AlternateTopologicalOrder>(graph.events(),topology_owner_);
  }
}
void FamilyRelationQuery::resetScratch() {
  for(std::size_t i=0;i<used_scratch_;++i) {
    auto& slot=scratch_[i];scratch_slots_[slot.chain]=SIZE_MAX;
    const auto clear=[](auto& values) {
      // Reuse ordinary query-sized buffers; retain at most 4 KiB per field.
      // An irregular batch must not permanently enlarge every pool slot.
      using Vector=std::decay_t<decltype(values)>;
      if(values.capacity()>4096/sizeof(typename Vector::value_type))Vector{}.swap(values);
      else values.clear();
    };
    clear(slot.targets);slot.frontier.clear();slot.queued=false;
  }
  used_scratch_=0;
}
FamilyRelationQuery::ChainScratch& FamilyRelationQuery::scratch(std::size_t chain) {
  auto& index=scratch_slots_.at(chain);
  if(index==SIZE_MAX) {
    index=used_scratch_++;
    if(index==scratch_.size())scratch_.emplace_back();
    scratch_[index].chain=chain;
  }
  return scratch_[index];
}
void FamilyRelationQuery::checkOrder(std::size_t source,std::vector<FamilyRelationRequest>& requests) {
  std::vector<FamilyRelationBatch> batch;batch.push_back({source,std::move(requests)});
  checkOrderBatch(batch);requests=std::move(batch.front().requests);
}
std::vector<std::uint8_t> FamilyRelationQuery::allReachGroups(
    std::span<const HbAllReachGroup> groups,bool strict) {
  ProfileScope profile("oracle.relation.all_groups");
  struct Pending {
    FamilyRelationRequest request;
    std::vector<std::pair<std::size_t,std::size_t>> outputs;
  };
  std::map<std::size_t,std::vector<Pending>> by_source;
  std::vector<std::uint8_t> answers;
  for(const auto& group:groups) {
    std::map<std::size_t,MemberSet> sources,targets;
    std::map<std::size_t,std::vector<std::pair<std::size_t,std::size_t>>> owners;
    const auto insert=[&](auto& sets,EventId event) {
      if(!graph_.contains(event))throw std::invalid_argument("unknown HB endpoint");
      const auto [family,member]=families().locate(event);
      auto [it,fresh]=sets.try_emplace(family,families().families()[family].count);
      it->second.set(member);return std::pair{family,member};
    };
    for(auto event:group.sources)insert(sources,event);
    for(auto event:group.targets) {
      const auto [family,member]=insert(targets,event);
      owners[family].emplace_back(answers.size(),member);answers.push_back(1);
    }
    // Product of stored family partitions, never of their members.
    for(const auto& [source,left]:sources)for(const auto& [target,right]:targets) {
      by_source[source].push_back({{target,{MemberRelation::Rectangle(left,right)}},owners.at(target)});
      ++profile.nodes;
    }
  }
  std::vector<FamilyRelationBatch> batch;
  std::vector<std::vector<Pending>*> metadata;
  const auto flush=[&] {
    if(batch.empty())return;
    checkOrderBatch(batch,true,strict);
    for(std::size_t b=0;b<batch.size();++b)for(std::size_t q=0;q<batch[b].requests.size();++q) {
      const auto& request=batch[b].requests[q];
      const auto failed=request.uncoveredTargets();
      for(const auto& [output,member]:(*metadata[b])[q].outputs)
        if(failed.contains(member))answers[output]=0;
    }
    batch.clear();metadata.clear();
  };
  std::size_t width=0;
  for(auto& [source,pending]:by_source) {
    const auto next_width=MemberSet::storageWords(families().families()[source].count);
    if(batch.size()==64 || (!batch.empty() && width!=next_width))flush();
    width=next_width;
    FamilyRelationBatch item{source,{}};item.requests.reserve(pending.size());
    for(auto& p:pending)item.requests.push_back(std::move(p.request));
    metadata.push_back(&pending);batch.push_back(std::move(item));
  }
  flush();return answers;
}
void FamilyRelationQuery::checkOrderBatch(std::vector<FamilyRelationBatch>& batch,bool forward_only,bool strict) {
  if(batch.empty())return;
  if(batch.size()>64)throw std::invalid_argument("family HB source-label chunk exceeds 64");
  if(visit_census_ && ++batch_serial_==0) {
    for(auto& visits:last_visit_batch_)std::fill(visits.begin(),visits.end(),0);
    ++batch_serial_;
  }
  std::size_t extent=0;
  for(const auto& query:batch) {
    const auto count=families().families().at(query.source).count;
    extent=std::max(extent,count);
    for(const auto& target:query.requests)for(const auto& relation:target.requested)
      if(relation.left.extent()!=count || relation.right.extent()!=families().families().at(target.target).count)
        throw std::invalid_argument("family HB relation domain mismatch");
  }
  // Source labels keep their own event family/member meaning. Pad only the
  // bitset's storage domain so singleton and multi-member sources share a
  // traversal. No new member becomes active; restore public extents afterward.
  for(auto& query:batch)if(families().families()[query.source].count!=extent)
    for(auto& request:query.requests)for(auto& relation:request.requested)
      relation.left=relation.left.shifted(0,extent);
  // One proof bucket per target family in this batch. Source labels are
  // selectors, not independent copies of the same proof relation storage.
  std::unordered_map<std::size_t,std::shared_ptr<CumulativeRelationCoverage>> proof_buckets;
  std::set<std::tuple<const CumulativeRelationCoverage*,std::uint64_t,std::uint64_t,std::size_t>> imported;
  for(std::size_t s=0;s<batch.size();++s)for(auto& request:batch[s].requests) {
    auto& bucket=proof_buckets[request.target];if(!bucket)bucket=std::make_shared<CumulativeRelationCoverage>();
    const auto label=std::uint64_t(1)<<s;
    if(request.coverage && imported.emplace(request.coverage.get(),request.source_label,label,request.target).second) {
      CumulativeRelationCoverage::Counters counts;
      request.visitProofs([&](const auto& relation) {
        auto padded=relation;padded.left=padded.left.shifted(0,extent);bucket->advance(0,padded,counts,label);
      });
    }
    request.coverage=bucket;request.source_label=label;
  }
  // A chain proves ordering. Opposite legal topological orders only prove
  // NON-ordering; keep those pairs as findings/warnings, without searching.
  {
  ProfileScope profile("memory.family.hb_precheck");
  ProfileScope rank_members("memory.family.hb_rank_members");
  for(std::size_t s=0;s<batch.size();++s) {
    const auto source=batch[s].source;
    for(std::size_t q=0;q<batch[s].requests.size();++q) {
      auto& request=batch[s].requests[q];const auto target=request.target;
      request.strict_self=strict && source==target;
      if(chains_.chainOf(source)==chains_.chainOf(target) &&
          (!forward_only || chains_.chainPosition(source,false)<=chains_.chainPosition(target,false))) {
        // Chain construction requires full identity relations with identical
        // member-coordinate maps; different members are still independent.
        const auto count=families().families()[source].count;
        request.prove(MemberRelation::Diagonal(MemberRange(extent,0,count),count));
      }
      if(request.complete || !alternate_)continue;
      bool unordered=false;
      const auto a=alternate_->familyBounds(source),b=alternate_->familyBounds(target);
      if((bounds_[source].second<bounds_[target].first && a.first>b.second) ||
         (bounds_[target].second<bounds_[source].first && b.first>a.second))unordered=true;
      if(source==target) {
        auto [it,inserted]=opposite_members_.try_emplace(source,false);
        if(inserted) {
          const auto& f=families().families()[source];
          bool forward=true,backward=true;
          auto first=graph_.traceOrder()->rank(f.first),second=alternate_->memberRank(source,0);
          for(std::size_t member=1;member<f.count;++member) {
            ++rank_members.nodes;++stats_.rank_member_checks;
            const auto x=graph_.traceOrder()->rank(EventId(f.first.value()+f.stride*member));
            const auto y=alternate_->memberRank(source,member);
            forward&=first<x && second>y;backward&=first>x && second<y;
            first=x;second=y;
          }
          it->second=forward || backward;
        }
        unordered=it->second;
      }
      if(unordered){request.search_disabled=true;++profile.pruned;}
      ++profile.nodes;
    }
  }
  }
  propagate(batch,false,extent);if(!forward_only)propagate(batch,true,extent);
  for(std::size_t s=0;s<batch.size();++s)for(std::size_t q=0;q<batch[s].requests.size();++q) {
    batch[s].requests[q].check();
  }
  for(auto& query:batch) {
    const auto count=families().families()[query.source].count;
    if(count!=extent)for(auto& request:query.requests) {
      for(auto& relation:request.requested)relation.left=relation.left.shifted(0,count);
      // Coverage retains its padded storage domain; query subsets are embedded
      // without activating padded members.
    }
  }
}
void FamilyRelationQuery::propagate(std::vector<FamilyRelationBatch>& batch,bool reverse,std::size_t source_extent) {
  ProfileScope profile("memory.family.hb_relation");
  // Counters remain visible in an interrupted propagation. nodes: repeated
  // visits within this batch/direction; edges: first visit in a new batch to
  // a previously visited relation/direction; pruned: first-ever visits.
  ProfileScope reuse("memory.family.hb_visit_reuse");
  ProfileScope pruning("memory.family.hb_demand_pruning");
  resetScratch();
  const auto initial_enumerations=stats_.member_enumerations;
  std::array<std::size_t,64> remaining_by_source{};
  std::uint64_t active_labels=0;
  struct Demand {
    std::size_t lo=SIZE_MAX,hi=0,lo2=SIZE_MAX,hi2=0;
    MemberSet members;
    explicit Demand(std::size_t extent):members(extent){}
  };
  std::vector<Demand> demand;
  demand.reserve(batch.size());
  for(std::size_t s=0;s<batch.size();++s)demand.emplace_back(source_extent);
  std::size_t remaining=0,lo=SIZE_MAX,hi=0,lo2=SIZE_MAX,hi2=0;
  for(std::size_t s=0;s<batch.size();++s) {
    const auto source=batch[s].source;
    lo=std::min(lo,bounds_[source].first);hi=std::max(hi,bounds_[source].second);
    if(alternate_) {const auto b=alternate_->familyBounds(source);lo2=std::min(lo2,b.first);hi2=std::max(hi2,b.second);}
    for(std::size_t i=0;i<batch[s].requests.size();++i)if(batch[s].requests[i].pending()) {
      const auto family=batch[s].requests[i].target;
      // A direction inconsistent with either topological order cannot reach
      // this target. The other direction remains checked normally.
      const auto before=[&](auto x,auto y){return reverse?x.second>=y.first:x.first<=y.second;};
      if(!before(bounds_[source],bounds_[family]) ||
         (alternate_ && !before(alternate_->familyBounds(source),alternate_->familyBounds(family))))continue;
      scratch(chains_.chainOf(family)).targets.push_back({chains_.chainPosition(family,reverse),s,i});++remaining;
      ++remaining_by_source[s];active_labels|=std::uint64_t(1)<<s;
      auto& d=demand[s];
      d.lo=std::min(d.lo,bounds_[family].first);d.hi=std::max(d.hi,bounds_[family].second);
      if(alternate_) {const auto b=alternate_->familyBounds(family);d.lo2=std::min(d.lo2,b.first);d.hi2=std::max(d.hi2,b.second);}
      for(const auto& relation:batch[s].requests[i].requested)d.members.unite(relation.left);
      lo=std::min(lo,bounds_[family].first);hi=std::max(hi,bounds_[family].second);
      if(alternate_) {const auto b=alternate_->familyBounds(family);lo2=std::min(lo2,b.first);hi2=std::max(hi2,b.second);}
    }
  }
  if(!remaining)return;
  // Compile the <=64 horizons to sorted threshold/mask tables. Each lookup
  // takes O(log labels), not a loop over labels for every graph edge.
  using Horizon=std::vector<std::pair<std::size_t,std::uint64_t>>;
  const auto horizons=[&](bool second) {
    Horizon out;
    for(std::size_t s=0;s<batch.size();++s)if(active_labels&(std::uint64_t(1)<<s)) {
      const auto& d=demand[s];
      out.emplace_back(second?(reverse?d.lo2:d.hi2):(reverse?d.lo:d.hi),std::uint64_t(1)<<s);
    }
    std::sort(out.begin(),out.end());
    if(reverse)for(std::size_t i=1;i<out.size();++i)out[i].second|=out[i-1].second;
    else for(std::size_t i=out.size();i>1;--i)out[i-2].second|=out[i-1].second;
    return out;
  };
  const auto horizon=horizons(false),horizon2=horizons(true);
  const auto allowed=[&](const Horizon& h,std::pair<std::size_t,std::size_t> b) -> std::uint64_t {
    if(reverse) {
      const auto it=std::upper_bound(h.begin(),h.end(),b.second,[](auto rank,const auto& entry){return rank<entry.first;});
      return it==h.begin()?0:std::prev(it)->second;
    }
    const auto it=std::lower_bound(h.begin(),h.end(),b.first,[](const auto& entry,auto rank){return entry.first<rank;});
    return it==h.end()?0:it->second;
  };
  // Necessary conditions only; rank order never establishes reachability.
  const auto admit=[&](std::size_t family,std::uint64_t labels) {
    const auto original=labels;
    labels&=active_labels&allowed(horizon,bounds_[family]);
    if(labels && alternate_)labels&=allowed(horizon2,alternate_->familyBounds(family));
    const auto removed=std::popcount(original)-std::popcount(labels);
    stats_.demand_pruned_labels+=removed;pruning.pruned+=removed;
    if(!labels || bounds_[family].second<lo || bounds_[family].first>hi)return std::uint64_t(0);
    if(alternate_) {const auto b=alternate_->familyBounds(family);if(b.second<lo2 || b.first>hi2)return std::uint64_t(0);}
    return labels;
  };
  for(std::size_t i=0;i<used_scratch_;++i) {
    auto& items=scratch_[i].targets;
    std::sort(items.begin(),items.end(),[](const auto& a,const auto& b){return a.position<b.position;});
  }
  ++stats_.traversals;
  std::deque<std::size_t> queue;
  RelationCoverageFrontier::Counters frontier_stats;
  ProfileScope frontier_counts("memory.family.hb_frontier_accept");
  ProfileScope frontier_work("memory.family.hb_frontier_work");
  ProfileScope union_counts("memory.family.hb_union_coverage");
  ProfileScope union_work("memory.family.hb_union_sweep");
  ProfileScope cumulative("memory.family.hb_cumulative_updates");
  ProfileScope partitions("memory.family.hb_cumulative_partitions");
  ProfileScope copies("memory.family.hb_cumulative_copies");
  ProfileScope singleton_shapes("memory.family.hb_cumulative_singletons");
  ProfileScope label_groups("memory.family.hb_label_groups");
  ProfileScope target_updates("memory.family.hb_target_shared_updates");
  ProfileScope simple_updates("memory.family.hb_simple_row_updates");
  ProfileScope state_repacking("memory.family.hb_state_repacking");
  ProfileScope arrival_cache("memory.family.hb_last_arrival_cache");
  CumulativeRelationCoverage::Counters target_update_counts;
  RelationCoverageCounters target_stats;
  const auto publish=[&] {
    frontier_counts.nodes=frontier_stats.arrivals;
    frontier_counts.edges=frontier_stats.changed_arrivals;
    frontier_counts.pruned=frontier_stats.covered_pieces;
    frontier_work.nodes=frontier_stats.emitted;
    frontier_work.edges=frontier_stats.earlier_cells;
    frontier_work.pruned=frontier_stats.stale_tickets;
    union_counts.nodes=frontier_stats.coverage.checks+target_stats.checks;
    union_counts.edges=frontier_stats.coverage.proofs+target_stats.proofs;
    union_counts.pruned=frontier_stats.coverage.diagonal_candidates+target_stats.diagonal_candidates;
    union_work.nodes=frontier_stats.coverage.endpoints+target_stats.endpoints;
    union_work.edges=frontier_stats.coverage.segments+target_stats.segments;
    const auto& c=frontier_stats.cumulative;
    cumulative.nodes=c.arrivals;cumulative.edges=c.changed;cumulative.pruned=c.duplicate_arrivals;
    partitions.nodes=c.peak_rows;partitions.edges=c.peak_columns;partitions.pruned=c.splits;
    copies.nodes=c.row_copies;copies.edges=c.copied_columns;copies.pruned=c.touched;
    singleton_shapes.nodes=c.singleton_arrivals;singleton_shapes.edges=c.singleton_emissions;
    label_groups.nodes=c.requested_labels;label_groups.edges=c.state_groups;label_groups.pruned=c.peak_groups;
    target_updates.nodes=target_update_counts.arrivals;target_updates.edges=target_update_counts.requested_labels;
    target_updates.pruned=target_update_counts.changed;
    simple_updates.nodes=c.single_row_updates;simple_updates.edges=c.single_row_noops;simple_updates.pruned=c.row_update_maps;
    state_repacking.nodes=c.state_normalizations;state_repacking.edges=c.drain_groupings;
    arrival_cache.nodes=c.last_arrival_hits;arrival_cache.edges=c.last_arrival_misses;
  };
  const auto add=[&](std::size_t family,MemberRelation relation,std::uint64_t labels) {
    // Seeds and edges use the same admission above exactly once. No target
    // completion/label retirement occurs between admission and this enqueue.
    if(!labels || relation.empty())return;
    const auto chain=chains_.chainOf(family),position=chains_.chainPosition(family,reverse);
    auto& slot=scratch(chain);
    slot.frontier.accept(position,std::move(relation),labels,frontier_stats);
    if(!slot.queued && !slot.frontier.emptyPending()) {
      slot.queued=true;queue.push_back(chain);
    }
  };
  for(std::size_t s=0;s<batch.size();++s) {
    const auto source=batch[s].source,count=families().families()[source].count;
    const auto labels=admit(source,std::uint64_t(1)<<s);
    if(labels)add(source,MemberRelation::Diagonal(demand[s].members,count),labels);
  }
  // Keep ownership outside scratch_: add() may resize/move its chain slots.
  // Reuse capacity across queue pops, including the common singleton drain.
  std::vector<RelationCoverageFrontier::Work> work;
  while(!queue.empty() && remaining) {
    const auto chain=queue.front();queue.pop_front();
    scratch(chain).queued=false;
    scratch(chain).frontier.drain(active_labels,frontier_stats,work);
    // Consume compressed proofs; partial overlap is permitted, full coverage
    // is suppressed. The target array is immutable; its storage
    // stays valid when scratch_ moves its vector-owning slots during add().
    const std::span<const Target> targets(scratch(chain).targets);
    for(auto& item:work) {
      for(auto target=std::lower_bound(targets.begin(),targets.end(),item.position,
          [](const auto& t,auto p){return t.position<p;});
          target!=targets.end() && target->position<item.end;) {
        auto end=target;std::uint64_t labels=0;
        while(end!=targets.end() && end->position==target->position) {
          const auto bit=std::uint64_t(1)<<end->source;
          if((item.labels&bit) && batch[end->source].requests[end->index].pending())labels|=bit;
          ++end;
        }
        // A chain position identifies one target family. All its requests
        // reference the same bucket, so proof insertion is shared by labels.
        auto& bucket=batch[target->source].requests[target->index].coverage;
        const bool changed=labels && bucket->advance(0,item.relation,target_update_counts,labels);
        for(auto owner=target;owner!=end;++owner) {
          if(!(labels&(std::uint64_t(1)<<owner->source)))continue;
          auto& request=batch[owner->source].requests[owner->index];
          if(!request.pending())continue;
          if(changed)request.proofChanged(&target_stats);
          if(request.complete) {
            --remaining;
            if(!--remaining_by_source[owner->source])active_labels&=~(std::uint64_t(1)<<owner->source);
          }
        }
        target=end;
      }
      item.labels&=active_labels;
      if(!item.labels){++pruning.edges;continue;}
      auto& current=item.relation;++profile.nodes;
      const auto& exits=chains_.chainExits(chain,reverse);
      auto exit=std::lower_bound(exits.begin(),exits.end(),item.position,
          [](const auto& e,auto p){return e.first<p;});
      for(;exit!=exits.end() && exit->first<item.end;++exit) {
        item.labels&=active_labels;
        if(!item.labels)break;
        auto edge=topology_.relations()[exit->second];
        if(reverse) {
          std::swap(edge.source,edge.target);std::swap(edge.first_source,edge.first_target);
          std::swap(edge.source_step,edge.target_step);
        }
        const auto labels=admit(edge.target,item.labels);
        if(!labels)continue;
        ++stats_.relation_visits;++profile.edges;
        if(visit_census_) {
          auto& previous=last_visit_batch_[reverse][exit->second];
          if(previous==batch_serial_)++reuse.nodes;
          else if(previous)++reuse.edges;
          else ++reuse.pruned;
          previous=batch_serial_;
        }
        if((stats_.relation_visits&65535)==0){publish();ProfileSnapshot();}
        const auto source_range=MemberRange(current.right.extent(),edge.first_source,
            edge.first_source+(edge.source_step?edge.count:1));
        auto selected=current.right.intersect(source_range);
        if(selected.empty())continue;
        auto left=current.diagonal?current.left.intersect(selected.shifted(-current.shift,current.left.extent())):current.left;
        const auto extent=families().families()[edge.target].count;
        if(edge.source_step && edge.target_step) {
          const auto shift=std::int64_t(edge.first_target)-std::int64_t(edge.first_source);
          add(edge.target,current.diagonal?MemberRelation::Diagonal(std::move(left),extent,current.shift+shift):
              MemberRelation::Rectangle(std::move(left),selected.shifted(shift,extent)),labels);
        } else {
          add(edge.target,MemberRelation::Rectangle(std::move(left),MemberRange(extent,edge.first_target,
              edge.first_target+(edge.target_step?edge.count:1))),labels);
        }
        if(!remaining)break;
      }
      if(!remaining)break;
    }
  }
  publish();
  ProfileScope final_partitions("memory.family.hb_cumulative_final_partitions");
  for(std::size_t i=0;i<used_scratch_;++i) {
    const auto count=scratch_[i].frontier.cellCount();
    final_partitions.nodes+=count;final_partitions.edges=std::max<std::uint64_t>(final_partitions.edges,count);
  }
  stats_.member_enumerations+=frontier_stats.member_enumerations;
  stats_.history_candidates+=frontier_stats.candidates;
  stats_.frontier_pieces+=frontier_stats.new_cells+frontier_stats.earlier_cells;
  stats_.exact_hits+=frontier_stats.covered_pieces;
  stats_.subtraction_pieces+=frontier_stats.splits;
  profile.pruned=stats_.member_enumerations-initial_enumerations;
  ProfileScope candidates("memory.family.hb_frontier_candidates");
  candidates.nodes=frontier_stats.candidates;
  candidates.edges=frontier_stats.splits;
  candidates.pruned=frontier_stats.member_enumerations;
}
} // namespace sentryward::algorithm
