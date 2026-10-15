#include "sentryward/algorithm/analysis/phase_database.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <utility>
#include <variant>

namespace sentryward::algorithm {
namespace {

std::vector<EventId> InTraceOrder(const EventIdFamilies& families, const ReferenceTrace& trace) {
  std::vector<std::pair<TraceIndex, EventId>> ordered;
  ordered.reserve(families.size());
  for (auto event : families)
    if (auto index = trace.indexOf(event)) ordered.emplace_back(*index, event);
  std::sort(ordered.begin(), ordered.end());
  std::vector<EventId> result;
  result.reserve(ordered.size());
  for (auto [index, event] : ordered) { (void)index; result.push_back(event); }
  return result;
}

[[nodiscard]] PhaseDatabaseError Error(
    PhaseDatabaseErrorCode code, std::optional<EventId> event = std::nullopt,
    std::optional<PhaseKey> phase = std::nullopt) {
  return PhaseDatabaseError{code, event, phase};
}

[[nodiscard]] Result<std::int64_t, PhaseDatabaseError> PositiveWeight(
    std::uint64_t value, EventId event) {
  if (value >
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    return Result<std::int64_t, PhaseDatabaseError>::Failure(
        Error(PhaseDatabaseErrorCode::kArithmeticOverflow, event));
  }
  return Result<std::int64_t, PhaseDatabaseError>::Success(
      static_cast<std::int64_t>(value));
}

[[nodiscard]] Result<std::int64_t, PhaseDatabaseError> NegativeWeight(
    std::uint64_t value, EventId event) {
  constexpr std::uint64_t kMinimumMagnitude =
      std::uint64_t{1} << std::numeric_limits<std::int64_t>::digits;
  if (value > kMinimumMagnitude) {
    return Result<std::int64_t, PhaseDatabaseError>::Failure(
        Error(PhaseDatabaseErrorCode::kArithmeticOverflow, event));
  }
  if (value == kMinimumMagnitude) {
    return Result<std::int64_t, PhaseDatabaseError>::Success(
        std::numeric_limits<std::int64_t>::min());
  }
  return Result<std::int64_t, PhaseDatabaseError>::Success(
      -static_cast<std::int64_t>(value));
}

[[nodiscard]] bool IsWaitIssue(const Event& event) {
  if (const auto* named = std::get_if<NamedBarrierArriveEvent>(&event.payload))
    return named->blocking;
  return std::holds_alternative<WaitStateIssueEvent>(event.payload) ||
         std::holds_alternative<WaitParityIssueEvent>(event.payload);
}

[[nodiscard]] bool IsWaitDone(const Event& event) {
  if (std::holds_alternative<NamedBarrierReturnEvent>(event.payload)) return true;
  return std::holds_alternative<WaitStateDoneEvent>(event.payload) ||
         std::holds_alternative<WaitParityDoneEvent>(event.payload);
}

[[nodiscard]] std::optional<BarrierLifetimeId> WaitBarrier(const Event& event) {
  if (const auto* p = std::get_if<NamedBarrierArriveEvent>(&event.payload)) return p->barrier;
  if (const auto* p = std::get_if<NamedBarrierReturnEvent>(&event.payload)) return p->barrier;
  if (const auto* payload = std::get_if<WaitStateIssueEvent>(&event.payload)) {
    return payload->barrier;
  }
  if (const auto* payload = std::get_if<WaitStateDoneEvent>(&event.payload)) {
    return payload->barrier;
  }
  if (const auto* payload = std::get_if<WaitParityIssueEvent>(&event.payload)) {
    return payload->barrier;
  }
  if (const auto* payload = std::get_if<WaitParityDoneEvent>(&event.payload)) {
    return payload->barrier;
  }
  return std::nullopt;
}

}  // namespace

Result<std::optional<PhaseEffectWeights>, PhaseDatabaseError>
GetPhaseEffectWeights(const Event& event) {
  if (const auto* increment =
          std::get_if<MBarrierAddArrivalsEvent>(&event.payload)) {
    return Result<std::optional<PhaseEffectWeights>,
                  PhaseDatabaseError>::Success(PhaseEffectWeights{
        increment->barrier, 0, 0, increment->arrivals});
  }
  if (const auto* named = std::get_if<NamedBarrierArriveEvent>(&event.payload))
    return Result<std::optional<PhaseEffectWeights>, PhaseDatabaseError>::Success(
        PhaseEffectWeights{named->barrier, 1, 0});
  if (const auto* update = std::get_if<MBarrierUpdateEvent>(&event.payload)) {
    auto transaction = PositiveWeight(update->transaction_increment, event.id);
    if (!transaction) {
      return Result<std::optional<PhaseEffectWeights>,
                    PhaseDatabaseError>::Failure(transaction.error());
    }
    return Result<std::optional<PhaseEffectWeights>,
                  PhaseDatabaseError>::Success(PhaseEffectWeights{
        update->barrier, update->arrival_decrement, transaction.value()});
  }

  if (const auto* completion =
          std::get_if<MBarrierCompleteTxEvent>(&event.payload)) {
    auto transaction = NegativeWeight(completion->bytes, event.id);
    if (!transaction) {
      return Result<std::optional<PhaseEffectWeights>,
                    PhaseDatabaseError>::Failure(transaction.error());
    }
    return Result<std::optional<PhaseEffectWeights>,
                  PhaseDatabaseError>::Success(PhaseEffectWeights{
        completion->barrier, 0, transaction.value()});
  }

  if (const auto* execution = std::get_if<AsyncMemoryExecEvent>(&event.payload);
      execution != nullptr && execution->completion.has_value()) {
    auto transaction = NegativeWeight(execution->completion->bytes, event.id);
    if (!transaction) {
      return Result<std::optional<PhaseEffectWeights>,
                    PhaseDatabaseError>::Failure(transaction.error());
    }
    return Result<std::optional<PhaseEffectWeights>,
                  PhaseDatabaseError>::Success(PhaseEffectWeights{
        execution->completion->barrier, 0, transaction.value()});
  }

  if (const auto* execution =
          std::get_if<GenericAsyncEffectEvent>(&event.payload);
      execution != nullptr && execution->completion) {
    if (const auto* arrival =
            std::get_if<ArriveEffect>(&*execution->completion)) {
      return Result<std::optional<PhaseEffectWeights>,
                    PhaseDatabaseError>::Success(PhaseEffectWeights{
          arrival->barrier, arrival->arrivals, 0});
    }
    const auto& completion = std::get<CompleteTxEffect>(*execution->completion);
    auto transaction = NegativeWeight(completion.bytes, event.id);
    if (!transaction) {
      return Result<std::optional<PhaseEffectWeights>,
                    PhaseDatabaseError>::Failure(transaction.error());
    }
    return Result<std::optional<PhaseEffectWeights>,
                  PhaseDatabaseError>::Success(PhaseEffectWeights{
        completion.barrier, 0, transaction.value()});
  }

  return Result<std::optional<PhaseEffectWeights>, PhaseDatabaseError>::Success(
      std::nullopt);
}

Result<PhaseDatabase, PhaseDatabaseError> PhaseDatabase::Build(
    const LoweredProgram& program, const ReferenceTrace& trace) {
  const EventDag& graph = program.graph();
  if (!graph.isAcyclic()) {
    return Result<PhaseDatabase, PhaseDatabaseError>::Failure(
        Error(PhaseDatabaseErrorCode::kInvalidGraph));
  }

  PhaseDatabase database;
  for (const auto& [barrier, initialization] :
       program.barrierInitializations().entries()) {
    database.expected_arrivals_.emplace(barrier,
                                        initialization.expected_arrivals);
  }

  std::map<BarrierLifetimeId, std::uint32_t> next_completion;
  std::set<BarrierLifetimeId> named_barriers;
  const auto& sync = program.synchronizationEvents();
  std::vector<EventId> named(sync.named_arrivals.begin(), sync.named_arrivals.end());
  std::sort(named.begin(), named.end());
  for (const auto id : named) {
    const auto event = graph.event(id);
    const auto* named = std::get_if<NamedBarrierArriveEvent>(&event.payload);
    if (!named) continue;
    named_barriers.insert(named->barrier);
    database.expected_arrivals_.try_emplace(named->barrier, named->expected_arrivals);
    const auto phase = trace.effectPhase(event.id);
    if (!phase) continue;
    const auto [it, inserted] = database.named_counts_.emplace(
        PhaseKey{named->barrier, *phase}, named->expected_arrivals);
    if (!inserted && it->second != named->expected_arrivals)
      return Result<PhaseDatabase, PhaseDatabaseError>::Failure(
          Error(PhaseDatabaseErrorCode::kEffectBarrierMismatch, event.id));
  }
  std::set<PhaseKey> seen_completions;
  const auto& entries = trace.entries();
  for (const auto& run : entries.runs()) {
    const auto index = run.begin;
    const auto* completion =
        std::get_if<BarrierPhaseCompletionTraceEntry>(&run.first);
    if (completion == nullptr) {
      continue;
    }
    const PhaseKey key = completion->phase;
    if (!database.expected_arrivals_.contains(key.barrier)) {
      return Result<PhaseDatabase, PhaseDatabaseError>::Failure(
          Error(PhaseDatabaseErrorCode::kUnknownBarrier, std::nullopt, key));
    }
    if (!seen_completions.insert(key).second) {
      return Result<PhaseDatabase, PhaseDatabaseError>::Failure(
          Error(PhaseDatabaseErrorCode::kDuplicatePhaseCompletion, std::nullopt,
                key));
    }
    std::uint32_t& expected = next_completion[key.barrier];
    if (key.phase != expected) {
      return Result<PhaseDatabase, PhaseDatabaseError>::Failure(
          Error(PhaseDatabaseErrorCode::kNoncontiguousPhaseCompletion,
                std::nullopt, key));
    }
    if (!trace.completionIndex(key).has_value() ||
        *trace.completionIndex(key) != index) {
      return Result<PhaseDatabase, PhaseDatabaseError>::Failure(
          Error(PhaseDatabaseErrorCode::kDuplicatePhaseCompletion, std::nullopt,
                key));
    }

    database.phase_indices_.emplace(key, database.phases_.size());
    database.phases_.push_back(PhaseInfo{.barrier = key.barrier,
                                         .phase = key.phase,
                                         .completion_index = index,
                                         .completed = true,
                                         .named = named_barriers.contains(key.barrier)});
    database.completed_in_trace_order_.push_back(key);
    if (expected == std::numeric_limits<std::uint32_t>::max()) {
      return Result<PhaseDatabase, PhaseDatabaseError>::Failure(Error(
          PhaseDatabaseErrorCode::kArithmeticOverflow, std::nullopt, key));
    }
    ++expected;
  }

  // Initialization creates phase zero, and every completion immediately
  // creates its successor. Thus each initialized barrier has exactly one final
  // open phase, including a phase with no counter effects.
  for (const auto& [barrier, arrivals] : database.expected_arrivals_) {
    static_cast<void>(arrivals);
    const PhaseKey key{barrier, next_completion[barrier]};
    database.phase_indices_.emplace(key, database.phases_.size());
    database.phases_.push_back(
        PhaseInfo{.barrier = barrier, .phase = key.phase, .completed = false,
                  .named = named_barriers.contains(barrier)});
    database.open_phases_.push_back(key);
  }

  std::set<EventId> assigned_effects;
  for (const auto& run : trace.normalEvents().runs()) {
    const auto event_id = run.first;
    bool valid = graph.contains(event_id);
    if (valid && run.delta > 0)
      valid = run.count - 1 <= (graph.eventCount() - 1 - event_id.value()) / std::uint64_t(run.delta);
    if (valid && run.delta < 0)
      valid = run.count - 1 <= event_id.value() / (std::uint64_t(-(run.delta + 1)) + 1);
    if (!valid) {
      return Result<PhaseDatabase, PhaseDatabaseError>::Failure(
          Error(PhaseDatabaseErrorCode::kUnknownEvent, event_id));
    }
  }
  for (EventId event_id : InTraceOrder(sync.effects, trace)) {
    const Event& event = graph.event(event_id);
    auto effect_result = GetPhaseEffectWeights(event);
    if (!effect_result) {
      return Result<PhaseDatabase, PhaseDatabaseError>::Failure(
          effect_result.error());
    }
    if (!effect_result.value().has_value()) {
      continue;
    }
    const PhaseEffectWeights effect = *effect_result.value();
    const auto assigned_phase = trace.effectPhase(event_id);
    if (!assigned_phase.has_value()) {
      return Result<PhaseDatabase, PhaseDatabaseError>::Failure(
          Error(PhaseDatabaseErrorCode::kMissingEffectAssignment, event_id));
    }
    const PhaseKey key{effect.barrier, *assigned_phase};
    const auto phase_it = database.phase_indices_.find(key);
    if (phase_it == database.phase_indices_.end()) {
      return Result<PhaseDatabase, PhaseDatabaseError>::Failure(Error(
          PhaseDatabaseErrorCode::kEffectOutsideReferencePhase, event_id, key));
    }
    PhaseInfo& phase = database.phases_[phase_it->second];
    phase.named = named_barriers.contains(phase.barrier);
    phase.effects.push_back(event_id);
    if (effect.arrival_weight > 0) {
      phase.positive_arrivals.push_back(event_id);
    }
    assigned_effects.insert(event_id);
  }

  // Check both directions: every graph counter effect has a trace assignment,
  // and every trace assignment denotes a counter effect in the graph.
  for (const auto id : sync.effects) {
    const auto event = graph.event(id);
    auto effect_result = GetPhaseEffectWeights(event);
    if (!effect_result) {
      return Result<PhaseDatabase, PhaseDatabaseError>::Failure(
          effect_result.error());
    }
    if (effect_result.value().has_value() &&
        !assigned_effects.contains(event.id)) {
      return Result<PhaseDatabase, PhaseDatabaseError>::Failure(
          Error(PhaseDatabaseErrorCode::kMissingEffectAssignment, event.id));
    }
  }
  for (const auto& [event_id, phase_number] : trace.effectPhases()) {
    static_cast<void>(phase_number);
    if (!graph.contains(event_id)) {
      return Result<PhaseDatabase, PhaseDatabaseError>::Failure(
          Error(PhaseDatabaseErrorCode::kUnknownEvent, event_id));
    }
    auto effect_result = GetPhaseEffectWeights(graph.event(event_id));
    if (!effect_result) {
      return Result<PhaseDatabase, PhaseDatabaseError>::Failure(
          effect_result.error());
    }
    if (!effect_result.value().has_value()) {
      return Result<PhaseDatabase, PhaseDatabaseError>::Failure(
          Error(PhaseDatabaseErrorCode::kNonEffectAssignment, event_id));
    }
  }

  std::map<InstId, PhaseKey> issue_target_by_origin;
  for (const auto& [event_id, barrier] : trace.initialWaits()) {
    if (!graph.contains(event_id)) {
      return Result<PhaseDatabase, PhaseDatabaseError>::Failure(
          Error(PhaseDatabaseErrorCode::kUnknownEvent, event_id));
    }
    const Event& event = graph.event(event_id);
    const auto* issue = std::get_if<WaitParityIssueEvent>(&event.payload);
    const auto* done = std::get_if<WaitParityDoneEvent>(&event.payload);
    if ((!issue && !done) || trace.waitTarget(event_id).has_value()) {
      return Result<PhaseDatabase, PhaseDatabaseError>::Failure(
          Error(PhaseDatabaseErrorCode::kNonWaitTarget, event_id));
    }
    if (*WaitBarrier(event) != barrier ||
        (issue ? issue->parity : done->parity) != 1) {
      return Result<PhaseDatabase, PhaseDatabaseError>::Failure(
          Error(PhaseDatabaseErrorCode::kWaitBarrierMismatch, event_id));
    }
    const auto milestones = event.origin
        ? program.origins().milestones(*event.origin) : std::nullopt;
    if (!milestones || trace.initialWaitBarrier(milestones->issue) != barrier ||
        trace.initialWaitBarrier(milestones->done) != barrier) {
      return Result<PhaseDatabase, PhaseDatabaseError>::Failure(
          Error(PhaseDatabaseErrorCode::kInconsistentWaitTarget, event_id));
    }
  }
  for (EventId event_id : InTraceOrder(sync.wait_issues, trace)) {
    const Event& event = graph.event(event_id);
    if (!IsWaitIssue(event) || trace.initialWaitBarrier(event_id).has_value()) {
      continue;
    }
    const auto target = trace.waitTarget(event_id);
    if (!target.has_value()) {
      return Result<PhaseDatabase, PhaseDatabaseError>::Failure(
          Error(PhaseDatabaseErrorCode::kMissingWaitTarget, event_id));
    }
    if (*WaitBarrier(event) != target->barrier) {
      return Result<PhaseDatabase, PhaseDatabaseError>::Failure(Error(
          PhaseDatabaseErrorCode::kWaitBarrierMismatch, event_id, *target));
    }
    const auto phase_it = database.phase_indices_.find(*target);
    if (phase_it == database.phase_indices_.end()) {
      return Result<PhaseDatabase, PhaseDatabaseError>::Failure(
          Error(PhaseDatabaseErrorCode::kEffectOutsideReferencePhase, event_id,
                *target));
    }
    database.phases_[phase_it->second].waits_issued_for_phase.push_back(
        event_id);
    if (event.origin.has_value()) {
      issue_target_by_origin.emplace(*event.origin, *target);
    }
  }

  for (EventId event_id : InTraceOrder(sync.wait_done, trace)) {
    const Event& event = graph.event(event_id);
    if (!IsWaitDone(event) || trace.initialWaitBarrier(event_id).has_value()) {
      continue;
    }
    std::optional<PhaseKey> target = trace.waitTarget(event_id);
    if (!target.has_value() && event.origin.has_value()) {
      const auto issue_it = issue_target_by_origin.find(*event.origin);
      if (issue_it != issue_target_by_origin.end()) {
        target = issue_it->second;
      }
    }
    if (!target.has_value()) {
      return Result<PhaseDatabase, PhaseDatabaseError>::Failure(
          Error(PhaseDatabaseErrorCode::kMissingWaitTarget, event_id));
    }
    if (*WaitBarrier(event) != target->barrier) {
      return Result<PhaseDatabase, PhaseDatabaseError>::Failure(Error(
          PhaseDatabaseErrorCode::kWaitBarrierMismatch, event_id, *target));
    }
    if (event.origin.has_value()) {
      const auto issue_it = issue_target_by_origin.find(*event.origin);
      if (issue_it != issue_target_by_origin.end() &&
          issue_it->second != *target) {
        return Result<PhaseDatabase, PhaseDatabaseError>::Failure(
            Error(PhaseDatabaseErrorCode::kInconsistentWaitTarget, event_id,
                  *target));
      }
    }
    const auto phase_it = database.phase_indices_.find(*target);
    if (phase_it == database.phase_indices_.end()) {
      return Result<PhaseDatabase, PhaseDatabaseError>::Failure(
          Error(PhaseDatabaseErrorCode::kEffectOutsideReferencePhase, event_id,
                *target));
    }
    database.phases_[phase_it->second].waits_done_for_phase.push_back(event_id);
  }

  for (const auto& [event_id, target] : trace.waitTargets()) {
    static_cast<void>(target);
    if (!graph.contains(event_id)) {
      return Result<PhaseDatabase, PhaseDatabaseError>::Failure(
          Error(PhaseDatabaseErrorCode::kUnknownEvent, event_id));
    }
    const Event& event = graph.event(event_id);
    if (!IsWaitIssue(event) && !IsWaitDone(event)) {
      return Result<PhaseDatabase, PhaseDatabaseError>::Failure(
          Error(PhaseDatabaseErrorCode::kNonWaitTarget, event_id));
    }
  }

  return Result<PhaseDatabase, PhaseDatabaseError>::Success(
      std::move(database));
}

const PhaseInfo* PhaseDatabase::find(PhaseKey phase) const noexcept {
  const auto it = phase_indices_.find(phase);
  return it == phase_indices_.end() ? nullptr : &phases_[it->second];
}

std::span<const PhaseInfo> PhaseDatabase::phases() const noexcept {
  return phases_;
}

std::span<const PhaseKey> PhaseDatabase::completedInTraceOrder()
    const noexcept {
  return completed_in_trace_order_;
}

std::span<const PhaseKey> PhaseDatabase::openPhases() const noexcept {
  return open_phases_;
}

std::optional<std::uint32_t> PhaseDatabase::expectedArrivals(PhaseKey phase) const noexcept {
  if (auto it = named_counts_.find(phase); it != named_counts_.end()) return it->second;
  return expectedArrivals(phase.barrier);
}

std::optional<std::uint32_t> PhaseDatabase::expectedArrivals(
    BarrierLifetimeId barrier) const noexcept {
  const auto it = expected_arrivals_.find(barrier);
  return it == expected_arrivals_.end()
             ? std::nullopt
             : std::optional<std::uint32_t>(it->second);
}

}  // namespace sentryward::algorithm
