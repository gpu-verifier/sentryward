#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "sentryward/algorithm/analysis/happens_before_graph.h"

namespace sentryward::algorithm {
class FamilyRelationQuery;

struct HbQuery final {
  EventId from;
  EventId to;
};

struct HbAllReachGroup final {
  std::span<const EventId> sources;
  std::span<const EventId> targets;
};

enum class HbAlgorithm { kBatchedBitset, kDfs, kAuto };

struct HbReachabilityOptions final {
  HbAlgorithm algorithm = HbAlgorithm::kAuto;
  std::size_t index_budget_bytes = 64U * 1024U * 1024U;
  std::size_t max_batch_sources = 1024;
  std::size_t cached_pairs = 262144;
};

struct HbReachabilityStats final {
  std::size_t queries = 0;
  std::size_t filtered_queries = 0;
  std::size_t batches = 0;
  std::size_t edge_visits = 0;
  std::size_t peak_index_bytes = 0;
  std::size_t dfs_queries = 0;
  std::size_t cache_hits = 0;
};

// Exact reflexive reachability on one immutable graph. Invalid IDs throw.
class HbReachability {
 public:
  virtual ~HbReachability() = default;
  [[nodiscard]] virtual const HappensBeforeGraph& graph() const noexcept = 0;
  [[nodiscard]] virtual std::string_view name() const noexcept = 0;
  [[nodiscard]] virtual bool reaches(EventId from, EventId to) = 0;
  // Answers preserve input order. Implementations may share work across queries.
  [[nodiscard]] virtual std::vector<std::uint8_t> reachesBatch(
      std::span<const HbQuery> queries);
  // One answer per target: every source reaches it (reflexive).
  // Empty sources are vacuously true. No Cartesian query list is required.
  [[nodiscard]] virtual std::vector<std::uint8_t> allReachMany(
      std::span<const EventId> sources, std::span<const EventId> targets);
  [[nodiscard]] virtual HbReachabilityStats stats() const noexcept = 0;
};

[[nodiscard]] std::unique_ptr<HbReachability> MakeHbReachability(
    const HappensBeforeGraph& graph, const HbReachabilityOptions& options = {});

using HbReachabilityFactory = std::function<std::unique_ptr<HbReachability>(
    const HappensBeforeGraph&, const HbReachabilityOptions&)>;

// One lazy index/cache for an immutable graph. The graph must outlive this object.
class StaticHbQueries final {
 public:
  explicit StaticHbQueries(const HappensBeforeGraph& graph,
      HbReachabilityOptions options = {}, HbReachabilityFactory factory = {});
  [[nodiscard]] const HappensBeforeGraph& graph() const noexcept { return graph_; }
  [[nodiscard]] HbReachability& get();
  [[nodiscard]] FamilyRelationQuery& relations();
  [[nodiscard]] HbReachabilityStats stats() const noexcept;
  [[nodiscard]] std::string_view name() const noexcept;
 private:
  const HappensBeforeGraph& graph_;
  std::size_t version_;
  HbReachabilityOptions options_;
  HbReachabilityFactory factory_;
  std::unique_ptr<HbReachability> index_;
  std::shared_ptr<FamilyRelationQuery> relations_;
};

// Counters are interval-local; peak_index_bytes remains the shared index high-water mark.
[[nodiscard]] HbReachabilityStats HbStatsSince(
    const HbReachabilityStats& current, const HbReachabilityStats& baseline) noexcept;

}  // namespace sentryward::algorithm
