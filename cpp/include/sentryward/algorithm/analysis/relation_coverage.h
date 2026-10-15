#pragma once
#include <span>
#include "sentryward/algorithm/analysis/cumulative_relation_coverage.h"

namespace sentryward::algorithm {
struct RelationCoverageCounters {
  std::size_t checks=0,proofs=0,endpoints=0,segments=0,diagonal_candidates=0;
};
// Compatibility entry point for standalone relation tests. Production
// frontiers and requests retain their cumulative index across insertions.
inline std::optional<std::pair<std::size_t,std::size_t>> UncoveredRelation(
    const MemberRelation& query,std::span<const MemberRelation* const> proofs,
    RelationCoverageCounters* stats=nullptr,MemberSet* complete_rows=nullptr) {
  CumulativeRelationCoverage index;CumulativeRelationCoverage::Counters counts;
  for(const auto* p:proofs)index.advance(0,*p,counts);
  auto result=index.uncovered(query,&counts,complete_rows);
  if(stats){++stats->checks;stats->proofs+=proofs.size();stats->segments+=counts.query_segments;stats->diagonal_candidates+=counts.mixed_candidates;}
  return result;
}
} // namespace sentryward::algorithm
