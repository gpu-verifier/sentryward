#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <vector>

#include "sentryward/algorithm/base/result.h"
#include "sentryward/algorithm/event_ir/program.h"
#include "sentryward/algorithm/semantics/reference_trace.h"

namespace sentryward::algorithm {

// The signed weights consumed by the ideal solver for one barrier effect.
struct PhaseEffectWeights final {
  BarrierLifetimeId barrier;
  std::uint32_t arrival_weight = 0;
  std::int64_t transaction_weight = 0;
  std::uint32_t arrival_increment = 0;

  friend bool operator==(const PhaseEffectWeights&,
                         const PhaseEffectWeights&) = default;
};

enum class PhaseDatabaseErrorCode {
  kInvalidGraph,
  kUnknownBarrier,
  kUnknownEvent,
  kNonEffectAssignment,
  kMissingEffectAssignment,
  kEffectBarrierMismatch,
  kDuplicatePhaseCompletion,
  kNoncontiguousPhaseCompletion,
  kEffectOutsideReferencePhase,
  kMissingWaitTarget,
  kNonWaitTarget,
  kWaitBarrierMismatch,
  kInconsistentWaitTarget,
  kArithmeticOverflow,
};

struct PhaseDatabaseError final {
  PhaseDatabaseErrorCode code = PhaseDatabaseErrorCode::kInvalidGraph;
  std::optional<EventId> event;
  std::optional<PhaseKey> phase;
};

struct PhaseInfo final {
  BarrierLifetimeId barrier;
  std::uint32_t phase = 0;
  std::vector<EventId> effects = {};
  std::vector<EventId> positive_arrivals = {};
  std::vector<EventId> waits_issued_for_phase = {};
  std::vector<EventId> waits_done_for_phase = {};
  TraceIndex completion_index = 0;
  bool completed = false;
  bool named = false;

  [[nodiscard]] PhaseKey key() const noexcept {
    return PhaseKey{barrier, phase};
  }

  friend bool operator==(const PhaseInfo&, const PhaseInfo&) = default;
};

// Extracts signed phase weights from an mbarrier counter effect.
[[nodiscard]] Result<std::optional<PhaseEffectWeights>, PhaseDatabaseError>
GetPhaseEffectWeights(const Event& event);

class PhaseDatabase final {
 public:
  [[nodiscard]] static Result<PhaseDatabase, PhaseDatabaseError> Build(
      const LoweredProgram& program, const ReferenceTrace& trace);

  [[nodiscard]] const PhaseInfo* find(PhaseKey phase) const noexcept;
  [[nodiscard]] std::span<const PhaseInfo> phases() const noexcept;
  [[nodiscard]] std::span<const PhaseKey> completedInTraceOrder()
      const noexcept;
  [[nodiscard]] std::span<const PhaseKey> openPhases() const noexcept;
  [[nodiscard]] std::optional<std::uint32_t> expectedArrivals(
      BarrierLifetimeId barrier) const noexcept;
  [[nodiscard]] std::optional<std::uint32_t> expectedArrivals(PhaseKey phase) const noexcept;

 private:
  std::vector<PhaseInfo> phases_;
  std::map<PhaseKey, std::size_t> phase_indices_;
  std::vector<PhaseKey> completed_in_trace_order_;
  std::vector<PhaseKey> open_phases_;
  std::map<BarrierLifetimeId, std::uint32_t> expected_arrivals_;
  std::map<PhaseKey, std::uint32_t> named_counts_;
};

}  // namespace sentryward::algorithm
