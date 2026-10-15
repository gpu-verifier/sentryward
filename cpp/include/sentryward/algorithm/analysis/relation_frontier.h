#pragma once
#include <unordered_map>
#include "sentryward/algorithm/analysis/relation_coverage.h"
#include "sentryward/algorithm/analysis/cumulative_relation_coverage.h"

namespace sentryward::algorithm {
class RelationCoverageFrontier final {
 public:
  using Work=CumulativeRelationCoverage::Work;
  struct Counters {
    std::size_t arrivals=0,changed_arrivals=0,new_cells=0,earlier_cells=0;
    std::size_t covered_pieces=0,candidates=0,splits=0,emitted=0,stale_tickets=0;
    std::size_t member_enumerations=0;
    RelationCoverageCounters coverage;
    CumulativeRelationCoverage::Counters cumulative;
  };
  bool emptyPending() const {return pending_labels_==0;}
  std::size_t cellCount() const {
    return coverage_.partitions();
  }
  bool accept(std::size_t position,MemberRelation relation,std::uint64_t labels,Counters& stats) {
    ++stats.arrivals;
    const bool changed=coverage_.advance(position,relation,stats.cumulative,labels);
    if(changed)pending_labels_|=labels;
    if(changed)++stats.changed_arrivals;else ++stats.covered_pieces;
    stats.splits=stats.cumulative.splits;stats.candidates=stats.cumulative.touched;
    stats.earlier_cells=stats.cumulative.earlier;
    return changed;
  }
  void drain(std::uint64_t active,Counters& stats,std::vector<Work>& work) {
    coverage_.drain(stats.cumulative,active,work);
    pending_labels_&=~active;
    // Retired labels never re-activate within a query. Keep unsent state for
    // callers that explicitly activate them in a subsequent drain.
    stats.emitted+=work.size();
  }
  std::vector<Work> drain(std::uint64_t active,Counters& stats) {
    std::vector<Work> work;drain(active,stats,work);return work;
  }
  void clear(){*this=RelationCoverageFrontier{};}
 private:
  CumulativeRelationCoverage coverage_;
  std::uint64_t pending_labels_=0;
};
} // namespace sentryward::algorithm
