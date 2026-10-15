#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "sentryward/algorithm/analysis/happens_before_graph.h"
#include "sentryward/algorithm/analysis/phase_database.h"

namespace sentryward::algorithm {

struct WaitReachabilityQuery final {
  EventId issue;
  EventId initialization;
  BarrierLifetimeId barrier;
  std::optional<std::uint32_t> previous_phase;
  std::optional<std::uint32_t> next_phase;
};

struct WaitReachabilityAnswer final {
  EventId issue;
  // Disengaged means unavailable: the caller must use its ordinary oracle.
  std::optional<bool> initialization;
  std::optional<bool> lower;
  bool upper_indexed = false;
  // With upper_indexed=true, no witness means an exact negative answer.
  std::optional<EventId> upper_witness;
};

struct WaitReachabilityBatchOptions final {
  // The caller has certified every completed phase and all consecutive phase
  // separations. Metadata alone is not a proof of that ordering.
  bool certified_phase_ordering = false;
  std::size_t scratch_limit_bytes = std::size_t{512} * 1024 * 1024;
};

// Immutable answers concerning original events. Current WellSync wait-edge
// insertions preserve these relations: each completion->issue shortcut is
// inserted only after all incoming phase effects already reach the issue.
// Unrelated later mutations require rebuilding the batch.
class WaitReachabilityBatch final {
 public:
  [[nodiscard]] static WaitReachabilityBatch Build(
      const HappensBeforeGraph& graph, std::span<const PhaseInfo> phases,
      std::span<const WaitReachabilityQuery> queries,
      WaitReachabilityBatchOptions options = {});

  [[nodiscard]] const WaitReachabilityAnswer* find(EventId issue) const noexcept;

 private:
  std::vector<WaitReachabilityAnswer> answers_;
};

}  // namespace sentryward::algorithm
