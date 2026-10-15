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
[[nodiscard]] std::unique_ptr<FamilyHbReachability> TryFamilyHbReachability(
    const HappensBeforeGraph& graph, const HbReachabilityOptions& options, bool reverse = false);
}
