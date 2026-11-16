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
  // Concurrent queries on an unchanging graph: after PrepareConcurrentQueries (single-threaded, the
  // graph unchanged since), any number of threads may call the *Concurrent forms at once. They give
  // allReachMany's and findAncestors's answers without reading or filling the pair cache or the
  // statistics; each adds its chain edge visits to `edge_visits`.
  virtual void PrepareConcurrentQueries() = 0;
  [[nodiscard]] virtual std::vector<std::uint8_t> allReachManyConcurrent(std::span<const EventId> sources,
      std::span<const EventId> targets, std::size_t& edge_visits) const = 0;
  [[nodiscard]] virtual std::vector<std::optional<EventId>> findAncestorsConcurrent(
      std::span<const EventId> sources, std::span<const EventId> targets, std::size_t& edge_visits) const = 0;
  // Afterwards, single-threaded: adds `calls` concurrent queries and their edge visits to stats().
  virtual void RecordConcurrentQueries(std::size_t calls, std::size_t edge_visits) = 0;
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
