#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "sentryward/algorithm/analysis/phase_database.h"

namespace sentryward::algorithm {

// A lower bound over all subsets satisfying the force-in/out constraints.
// It does not assert that an actual graph ideal omitting this event exists.
struct AlgebraicMissingEventBound final {
  EventId omitted_effect;
  std::int64_t transaction_lower_bound = 0;

  friend bool operator==(const AlgebraicMissingEventBound&,
                         const AlgebraicMissingEventBound&) = default;
};

struct AlgebraicPhaseBounds final {
  std::int64_t transaction_lower_bound = 0;
  std::vector<AlgebraicMissingEventBound> proper_prefix_lower_bounds;

  friend bool operator==(const AlgebraicPhaseBounds&,
                         const AlgebraicPhaseBounds&) = default;
};

// Prove a completed phase's transaction bounds without inspecting graph edges.
// The caller validates event identities, barrier membership, and arrival balance.
// This helper additionally requires zero transaction balance and no arrival
// increments. A missing result means inconclusive; it is never a rejection.
// Returned proper-prefix bounds are conservative, not exact minima or ideals.
[[nodiscard]] std::optional<AlgebraicPhaseBounds>
TryBoundPhaseWithoutDependencies(std::span<const EventId> events,
                                std::span<const PhaseEffectWeights> effects,
                                bool reject_negative_tx);

}  // namespace sentryward::algorithm
