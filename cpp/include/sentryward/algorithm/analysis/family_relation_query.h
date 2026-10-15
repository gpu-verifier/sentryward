#pragma once
#include "sentryward/algorithm/analysis/happens_before_graph.h"
#include "sentryward/algorithm/analysis/hb_reachability.h"
#include "sentryward/algorithm/analysis/member_relation.h"
#include "sentryward/algorithm/event_ir/event_relations.h"
#include "sentryward/algorithm/analysis/chain_frontier.h"
#include "sentryward/algorithm/analysis/alternate_topological_order.h"
#include "sentryward/algorithm/analysis/relation_frontier.h"

namespace sentryward::algorithm {
struct FamilyRelationRequest {
  std::size_t target;
  std::vector<MemberRelation> requested;
  std::shared_ptr<CumulativeRelationCoverage> coverage;
  std::uint64_t source_label=1;
  bool complete=false,search_disabled=false,strict_self=false;
  std::size_t additions=0,next_check=1;
  bool pending() const {return !complete && !search_disabled;}
  std::optional<std::pair<std::size_t,std::size_t>> witness(
      std::size_t domain,RelationCoverageCounters* stats=nullptr,bool honor_strict=true) const {
    const auto& q=requested.at(domain);
    if(honor_strict && strict_self && (!q.diagonal || !q.shift)) {
      const auto self=q.left.intersect(q.right.shifted(0,q.left.extent()));
      if(!self.empty())return std::pair{self.first(),self.first()};
    }
    if(complete)return {};
    CumulativeRelationCoverage::Counters counts;
    auto result=coverage?coverage->uncovered(q,&counts,nullptr,source_label):q.witness();
    if(stats){++stats->checks;stats->segments+=counts.query_segments;stats->diagonal_candidates+=counts.mixed_candidates;}
    return result;
  }
  bool check(RelationCoverageCounters* stats=nullptr) {
    for(std::size_t d=0;d<requested.size();++d)if(witness(d,stats,false))return false;
    return complete=true;
  }
  void prove(const MemberRelation& relation,RelationCoverageCounters* stats=nullptr) {
    if(complete)return;
    if(!coverage)coverage=std::make_shared<CumulativeRelationCoverage>();
    CumulativeRelationCoverage::Counters counts;
    if(coverage->advance(0,relation,counts,source_label))finishAddition(stats);
  }
  // Called after a single labelled insertion shared by all target owners.
  void proofChanged(RelationCoverageCounters* stats=nullptr) {if(!complete)finishAddition(stats);}
  template<class Visit> void visitProofs(Visit visit) const {if(coverage)coverage->visit(visit,source_label);}
  MemberSet uncoveredTargets(const MemberSet* source_subset=nullptr) const {
    const auto extent=requested.empty()?0:requested.front().right.extent();
    MemberSet failed(extent);
    // The target bucket, not each request, owns the lazy labelled transpose.
    for(const auto& q:requested) {
      auto restricted=q;
      if(source_subset)restricted.left=restricted.left.intersect(*source_subset);
      if(restricted.diagonal)restricted.right=restricted.right.intersect(
          restricted.left.shifted(restricted.shift,extent));
      if(restricted.left.empty())continue;
      if(!complete) {
        MemberSet good(extent);
        if(coverage)coverage->transposed().uncovered(restricted.transpose(),nullptr,&good,source_label);
        failed.unite(restricted.right.subtract(good));
      }
      if(strict_self && (!q.diagonal || !q.shift))failed.unite(restricted.right.intersect(restricted.left.shifted(0,extent)));
    }
    return failed;
  }
  bool containsUnordered(std::size_t i,std::size_t j) const {
    bool requested_pair=false;for(const auto& q:requested)requested_pair|=q.contains(i,j);
    if(!requested_pair)return false;
    if(strict_self && i==j)return true;
    const auto n=requested.empty()?0:requested.front().left.extent();
    const auto m=requested.empty()?0:requested.front().right.extent();
    return !coverage || bool(coverage->uncovered(MemberRelation::Rectangle(MemberRange(n,i,i+1),MemberRange(m,j,j+1)),nullptr,nullptr,source_label));
  }
  // A002's existing full-obligation adapter only. Ordinary race/lifecycle
  // consumers must use witness/target coverage and never call this adapter.
  std::vector<MemberRelation> duplicateResidual(std::size_t domain,std::size_t& expanded) const {
    std::vector<MemberRelation> out{requested.at(domain)};
    visitProofs([&](const MemberRelation& p) {
      auto adjusted=p;adjusted.left=adjusted.left.shifted(0,requested.at(domain).left.extent());
      RemoveRelation(out,adjusted,expanded);
    });
    return out;
  }
 private:
  void finishAddition(RelationCoverageCounters* stats) {
    if(++additions>=next_check){check(stats);next_check=additions>SIZE_MAX/2?SIZE_MAX:2*additions;}
  }
};
struct FamilyRelationBatch {
  std::size_t source;
  std::vector<FamilyRelationRequest> requests;
};
struct FamilyRelationQueryStats {
  std::size_t traversals=0,relation_visits=0,member_enumerations=0,frontier_pieces=0;
  std::size_t exact_hits=0,history_candidates=0,subtractions=0,subtraction_pieces=0,pending_hits=0;
  std::size_t demand_pruned_labels=0,pending_interval_merges=0;
  std::size_t rank_member_checks=0;
};
class FamilyRelationQuery {
 public:
  explicit FamilyRelationQuery(const HappensBeforeGraph& graph);
  const EventFamilies& families() const { return *topology_.layout(); }
  // Accumulate exact forward/backward proofs against immutable request domains.
  void checkOrder(std::size_t source,std::vector<FamilyRelationRequest>& requests);
  void checkOrderBatch(std::vector<FamilyRelationBatch>& batch,bool forward_only=false,bool strict=false);
  std::vector<std::uint8_t> allReachGroups(std::span<const HbAllReachGroup> groups,bool strict=false);
  const FamilyRelationQueryStats& stats() const { return stats_; }
 private:
  struct Target {std::size_t position,source,index;};
  struct ChainScratch {
    std::size_t chain=SIZE_MAX;
    std::vector<Target> targets;
    RelationCoverageFrontier frontier;
    bool queued=false;
  };
  // Chain IDs are dense and immutable. Initialize their direct index ONCE;
  // each traversal resets only touched slots, never the whole graph.
  std::vector<std::size_t> scratch_slots_;
  std::vector<ChainScratch> scratch_;
  std::size_t used_scratch_=0;
  void resetScratch();
  ChainScratch& scratch(std::size_t chain);
  void propagate(std::vector<FamilyRelationBatch>& batch,bool reverse,std::size_t source_extent);
  const HappensBeforeGraph& graph_;
  std::shared_ptr<const EventRelations> topology_owner_;
  const EventRelations& topology_;
  ChainQueryIndex chains_;
  std::vector<std::pair<std::size_t,std::size_t>> bounds_;
  std::unique_ptr<AlternateTopologicalOrder> alternate_;
  std::unordered_map<std::size_t,bool> opposite_members_;
  FamilyRelationQueryStats stats_;
  // Diagnostic only: relation IDs, not logical events or member pairs. Separate
  // directions so a reverse traversal is not counted as duplicate forward work.
  bool visit_census_=false;
  std::uint32_t batch_serial_=0;
  std::array<std::vector<std::uint32_t>,2> last_visit_batch_;
};
} // namespace sentryward::algorithm
