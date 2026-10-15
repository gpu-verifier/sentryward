#pragma once

#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "sentryward/algorithm/base/result.h"
#include "sentryward/algorithm/event_ir/program.h"
#include "sentryward/algorithm/semantics/reference_trace.h"

namespace sentryward::algorithm {

enum class SimulationErrorCode {
  kMalformedProgram,
  kDeadlock,
  kUseBeforeInitialization,
  kDuplicateInitialization,
  kZeroExpectedArrivals,
  kArrivalUnderflow,
  kTransactionOverflow,
  kTransactionUnderflow,
  kPhaseOverflow,
  kMissingSuccessfulWaitHandoff,
  kMissingStateToken,
  kStateBarrierMismatch,
  kStaleStateWait,
  kInvalidParity,
  kStaleParityWait,
  kWaitDoneBeforeIssue,
  kWaitTargetIncomplete,
  kCollectiveMismatch,
  kUnexpectedUrgentEvent,
  kArrivalOverflow,
  kNamedArrivalCountMismatch,
};

struct ReferenceDeadlock final {
  std::vector<EventId> executed;
  std::vector<EventId> blocked;
};

struct SemanticViolation final {
  SimulationErrorCode code;
  std::optional<EventId> event;
  std::optional<BarrierLifetimeId> barrier;
  std::optional<SyncId> sync_id;
  std::optional<StateId> state;
  std::string message;
};

using SimulationWitness = std::variant<ReferenceDeadlock, SemanticViolation>;

struct SimulationError final {
  SimulationErrorCode code;
  SimulationWitness witness;
  ReferenceTrace prefix;
};

struct ReplayResult final {
  struct Stall final {
    EventId first_unexecuted;

    friend bool operator==(const Stall&, const Stall&) = default;
  };

  ReferenceTrace prefix;
  std::vector<EventId> requested_order;
  std::vector<EventId> unexecuted;
  bool requested_order_preserved = true;
  std::optional<Stall> stall;
};

struct SimulationStatistics final {
  std::size_t event_families = 0;
  std::size_t dependency_relations = 0;
  std::size_t template_batches = 0;
  std::size_t template_events = 0;
  std::size_t arrival_batches = 0;
  std::size_t batched_arrivals = 0;
};

struct SimulationOptions final {
  // Legacy strict mode rejects negative transaction counts.
  bool reject_negative_tx = false;
  bool batch_arrivals = true;
  bool batch_templates = true;
  SimulationStatistics* statistics = nullptr;
};

class ReferenceSimulator final {
 public:
  [[nodiscard]] Result<ReferenceTrace, SimulationError> Run(
      const LoweredProgram& program,
      const SimulationOptions& options = {}) const;

  [[nodiscard]] Result<ReplayResult, SimulationError> ReplayPrefix(
      const LoweredProgram& program, std::span<const EventId> requested_order,
      const SimulationOptions& options = {}) const;
  [[nodiscard]] Result<ReplayResult, SimulationError> ReplayPrefix(
      const LoweredProgram& program, const TraceEventIds& requested_order,
      const SimulationOptions& options = {}) const {
    const std::vector<EventId> concrete(requested_order.begin(), requested_order.end());
    return ReplayPrefix(program, std::span<const EventId>(concrete), options);
  }
};

}  // namespace sentryward::algorithm
