#pragma once

#include <algorithm>
#include <memory>
#include <optional>
#include <vector>

#include "sentryward/algorithm/semantics/template_schedule.h"

namespace sentryward::algorithm {

// Opposite ready-family preference; members in a ready batch remain independent.
class AlternateTopologicalOrder final {
 public:
  template<class Edges>
  AlternateTopologicalOrder(const EventRecords& events, const Edges& edges,std::shared_ptr<const EventRelations> topology={})
      :AlternateTopologicalOrder(events,topology?std::move(topology):std::make_shared<const EventRelations>(events,edges)) {}
  AlternateTopologicalOrder(const EventRecords& events,std::shared_ptr<const EventRelations> topology) {
    TemplateSchedule ready(events, std::move(topology));
    layout_ = ready.layout();
    runs_.resize(ready.familyCount());
    bounds_.resize(ready.familyCount(), {SIZE_MAX, 0});
    std::size_t rank = 0;
    while (true) {
      auto batch = ready.next(false, true);
      if (!batch) batch = ready.next(true, true);
      if (!batch) break;
      // All members are ready, so their descending order is a legal extension.
      runs_[batch->family].push_back({batch->begin, batch->begin + batch->count,
                                      rank + batch->count - 1});
      auto& bounds = bounds_[batch->family];
      bounds.first = std::min(bounds.first, rank);
      bounds.second = std::max(bounds.second, rank + batch->count - 1);
      rank += batch->count;
      ready.finish(*batch);
    }
    if (rank != events.size()) throw std::invalid_argument("topological order requires an acyclic graph");
    count_ = rank;
    for (auto& runs : runs_)
      std::sort(runs.begin(), runs.end(), [](auto a, auto b) { return a.begin < b.begin; });
  }
  [[nodiscard]] std::size_t rank(EventId event) const {
    const auto [family, member] = layout_->locate(event);
    return memberRank(family, member);
  }
  [[nodiscard]] std::size_t memberRank(std::size_t family, std::size_t member) const {
    const auto& runs = runs_.at(family);
    const auto& run = runs.size() == 1 ? runs.front() :
        *std::prev(std::upper_bound(runs.begin(), runs.end(), member,
            [](auto value, const Run& r) { return value < r.begin; }));
    return run.rank - (member - run.begin);
  }
  [[nodiscard]] auto familyBounds(std::size_t family) const { return bounds_.at(family); }
  [[nodiscard]] std::size_t size() const { return count_; }
  [[nodiscard]] std::size_t runCount() const {
    std::size_t result = 0;
    for (const auto& runs : runs_) result += runs.size();
    return result;
  }
 private:
  struct Run { std::size_t begin, end, rank; };
  std::shared_ptr<const EventFamilies> layout_;
  std::vector<std::vector<Run>> runs_;
  std::vector<std::pair<std::size_t,std::size_t>> bounds_;
  std::size_t count_ = 0;
};
}  // namespace sentryward::algorithm
