#pragma once

#include <cstddef>
#include <limits>
#include <list>
#include <optional>
#include <span>
#include <vector>

#include "sentryward/algorithm/analysis/happens_before_graph.h"
#include "sentryward/algorithm/analysis/shared_reachability_index.h"
#include "sentryward/algorithm/analysis/family_hb_reachability.h"
#include "sentryward/algorithm/base/dynamic_bitset.h"

namespace sentryward::algorithm {

struct ReachabilityOptions final {
  std::size_t dense_memory_limit_bytes = 64 * 1024 * 1024;
  std::size_t cache_memory_limit_bytes = 16 * 1024 * 1024;
  // Built lazily per search direction; small/dense graphs keep the existing
  // representation. The opt-in also provides a reproducible baseline switch.
  bool use_shared_states = false;
  // Trace-order validation and constant-time insertion checks in WellSync.
  bool use_trace_order = true;
  // Point-query intervals and phase/set-query historical cutoffs.
  bool use_interval_pruning = true;
  std::size_t shared_state_min_vertices = 100000;
  // WellSync query workers; phases and graph mutations remain ordered.
  // One preserves the serial path. CLI defaults to eight for the AWS host.
  std::size_t worker_threads = 1;
};

class ReachabilityOracle final {
 public:
  explicit ReachabilityOracle(
      const HappensBeforeGraph& graph,
      ReachabilityOptions options = ReachabilityOptions{});

  [[nodiscard]] bool Reaches(EventId from, EventId to);
  [[nodiscard]] std::vector<std::uint8_t> ReachesBatch(std::span<const HbQuery> queries);
  [[nodiscard]] bool AllReach(std::span<const EventId> from, EventId to);
  // Shared propagation; each answer still refers to an independent target.
  [[nodiscard]] std::vector<std::uint8_t> AllReachMany(
      std::span<const EventId> from, std::span<const EventId> targets);
  [[nodiscard]] bool ReachesAll(EventId from, std::span<const EventId> targets);
  [[nodiscard]] bool ReachesAny(EventId from,
                                std::span<const EventId> targets);
  [[nodiscard]] std::optional<EventId> FindDescendant(
      EventId from, std::span<const EventId> candidates);
  // Returns any reflexive ancestor from candidates.
  [[nodiscard]] std::optional<EventId> FindAncestor(
      std::span<const EventId> candidates, EventId target);
  // One actual reflexive ancestor per target, sharing the candidate traversal.
  [[nodiscard]] std::vector<std::optional<EventId>> FindAncestors(
      std::span<const EventId> candidates, std::span<const EventId> targets);

  void NotifyGraphMutation() noexcept;
  [[nodiscard]] bool Freeze();
  [[nodiscard]] bool frozen() const noexcept;
  [[nodiscard]] bool usesDenseIndex() const noexcept;
  [[nodiscard]] std::size_t cachedBitsetBytes() const noexcept;
  // Allocated lazy indexes, including empty base states left by mutation.
  // These statistics do not trigger index construction or synchronization.
  [[nodiscard]] std::size_t sharedStateCount(bool reverse) const noexcept;
  [[nodiscard]] std::size_t sharedIndexBytes() const noexcept;

 private:
  static constexpr std::size_t kUnknownVersion =
      std::numeric_limits<std::size_t>::max();

  void refreshForGraphVersion();
  FamilyHbReachability* familyQueries(bool reverse = false);
  [[nodiscard]] bool hasOrder() const noexcept;
  [[nodiscard]] std::size_t rank(EventId event) const noexcept;
  [[nodiscard]] const DynamicBitset* cached(EventId root, bool reverse,
                                             std::span<const EventId> targets);
  [[nodiscard]] std::optional<EventId> search(
      EventId root, bool reverse, std::span<const EventId> targets,
      bool require_all);
  void cacheComplete(EventId root, bool reverse);
  [[nodiscard]] std::optional<EventId> searchShared(
      EventId root, bool reverse, bool require_all, std::size_t remaining);

  struct CacheEntry final {
    EventId root;
    bool reverse;
    DynamicBitset vertices;
    // Membership is complete only inside this interval. Unbounded searches
    // use [0, max]. Never interpret a bounded miss as global non-reachability.
    std::size_t lower;
    std::size_t upper;
  };

  const HappensBeforeGraph& graph_;
  ReachabilityOptions options_;
  std::size_t observed_version_ = kUnknownVersion;
  bool frozen_ = false;
  std::size_t cached_bytes_ = 0;
  std::list<CacheEntry> cache_;
  DynamicBitset scratch_;
  DynamicBitset query_targets_;
  std::vector<EventId> worklist_;
  std::vector<EventId> query_candidates_;
  std::vector<std::optional<EventId>> batch_witnesses_;
  std::vector<EventId> batch_worklist_;
  DynamicBitset batch_targets_;
  std::vector<EventId> batch_target_ids_;
  std::vector<std::size_t> topological_ranks_;
  std::optional<std::vector<DynamicBitset>> ancestor_sets_;
  std::optional<SharedReachabilityIndex> forward_shared_;
  std::optional<SharedReachabilityIndex> reverse_shared_;
  std::unique_ptr<FamilyHbReachability> family_queries_;
  std::unique_ptr<FamilyHbReachability> reverse_family_queries_;
  bool family_queries_checked_ = false;
  bool reverse_family_queries_checked_ = false;
  DynamicBitset shared_visited_;
  std::vector<SharedReachabilityIndex::StateId> shared_worklist_;
  bool scratch_contains_shared_search_ = false;
  bool bounded_search_ = false;
  std::size_t search_lower_ = 0;
  std::size_t search_upper_ = std::numeric_limits<std::size_t>::max();
};

}  // namespace sentryward::algorithm
