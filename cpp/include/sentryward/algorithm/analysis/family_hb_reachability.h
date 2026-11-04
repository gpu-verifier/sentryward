#pragma once
#include "sentryward/algorithm/analysis/hb_reachability.h"

namespace sentryward::algorithm {
class FamilyHbReachability : public HbReachability {
 public:
  bool strictlyReaches(EventId from, EventId to) { return from != to && reaches(from, to); }
  virtual void RefreshAppendOnly() = 0;
  virtual std::vector<std::uint8_t> allReachMany(std::span<const EventId> sources,
      std::span<const EventId> targets) = 0;
  virtual std::vector<std::optional<EventId>> findAncestors(std::span<const EventId> sources,
      std::span<const EventId> targets) = 0;
};
// Whether a family index applies to the graph as it is now: the budget allows
// one and the events are family-compressed. Throws for zero batch sources.
[[nodiscard]] bool FamilyHbReachabilityApplies(
    const HappensBeforeGraph& graph, const HbReachabilityOptions& options);
// Requires FamilyHbReachabilityApplies.
[[nodiscard]] std::unique_ptr<FamilyHbReachability> MakeFamilyHbReachability(
    const HappensBeforeGraph& graph, const HbReachabilityOptions& options, bool reverse = false);
// Null unless FamilyHbReachabilityApplies.
[[nodiscard]] std::unique_ptr<FamilyHbReachability> TryFamilyHbReachability(
    const HappensBeforeGraph& graph, const HbReachabilityOptions& options, bool reverse = false);
}
