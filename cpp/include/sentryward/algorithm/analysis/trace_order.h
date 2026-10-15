#pragma once

#include <cstddef>
#include <map>
#include <limits>
#include <stdexcept>
#include <optional>
#include <vector>

#include "sentryward/algorithm/event_ir/dag.h"
#include "sentryward/algorithm/semantics/reference_trace.h"

namespace sentryward::algorithm {

// A fixed linear extension, validated against every original edge. Synthetic
// completion events occupy their actual trace entries, not their appended IDs.
// The graph builder must reject every edge which violates this order.
class TraceOrder final {
 public:
  TraceOrder(const EventDag& graph, const ReferenceTrace& trace);
  [[nodiscard]] std::size_t rank(EventId event) const noexcept {
    return event.value() < base_count_ ? *ranks_.get(event) : derived_ranks_[event.value() - base_count_];
  }
  [[nodiscard]] std::pair<std::size_t,std::size_t> rankBounds(
      EventId first,std::uint32_t stride,std::size_t count) const {
    if(!count)throw std::invalid_argument("empty trace rank domain");
    std::size_t low=std::numeric_limits<std::size_t>::max(),high=0;
    for(std::size_t i=0;i<count;) {
      const auto event=EventId(first.value()+stride*i);
      const auto run=event.value()<base_count_?ranks_.prefix(event,stride,count-i):
        FamilyTraceIndex::Prefix{rank(event),1,0};
      const auto last=std::size_t(std::int64_t(run.rank)+run.step*std::int64_t(run.count-1));
      low=std::min(low,std::min(run.rank,last));high=std::max(high,std::max(run.rank,last));
      i+=run.count;
    }
    return {low,high};
  }
  [[nodiscard]] std::size_t completionCount() const noexcept {
    return completion_ranks_.size();
  }
  [[nodiscard]] bool appendCompletion(EventId event, PhaseKey phase);
 private:
  FamilyTraceIndex ranks_;
  std::size_t base_count_ = 0;
  std::vector<std::size_t> derived_ranks_;
  std::map<PhaseKey, std::size_t> completion_ranks_;
};

}  // namespace sentryward::algorithm
