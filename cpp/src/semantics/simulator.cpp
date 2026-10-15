#include "sentryward/algorithm/semantics/simulator.h"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <map>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "sentryward/algorithm/event_ir/event.h"
#include "sentryward/algorithm/diagnostics/memory_profile.h"
#include "sentryward/algorithm/diagnostics/performance_profile.h"
#include "sentryward/algorithm/semantics/barrier_state.h"
#include "sentryward/algorithm/semantics/collective_state.h"
#include "sentryward/algorithm/semantics/template_schedule.h"

namespace sentryward::algorithm {

class ReferenceTraceRecorder final {
 public:
  static void ReserveEvents(ReferenceTrace& trace, std::size_t count) {
    trace.reserveEvents(count);
  }
  static void SetFamilies(ReferenceTrace& trace, std::shared_ptr<const EventFamilies> families) {
    trace.effect_phases_.reset(families);
    trace.wait_targets_.reset(families);
    trace.initial_waits_.reset(families);
    if (families->families().size() * 2 < families->eventCount())
      trace.family_indices_.reset(std::move(families));
    else trace.reserveEvents(families->eventCount());
  }

  static void AppendNormal(ReferenceTrace& trace, EventId event) {
    trace.appendNormal(event);
  }
  static void AppendNormalBatch(ReferenceTrace& trace, EventId first,
                                std::uint32_t stride, std::size_t count) {
    trace.appendNormalBatch(first, stride, count);
  }

  static void AppendBarrierCompletion(ReferenceTrace& trace, PhaseKey phase) {
    trace.appendBarrierCompletion(phase);
  }

  static void AppendCollectiveCompletion(ReferenceTrace& trace, SyncId sync_id,
                                         EventId event) {
    trace.appendCollectiveCompletion(sync_id, event);
  }

  static void DiscardLastNormal(ReferenceTrace& trace, EventId event) {
    trace.discardLastNormal(event);
  }

  static void RecordEffectPhase(ReferenceTrace& trace, EventId event,
                                std::uint32_t phase) {
    trace.recordEffectPhase(event, phase);
  }
  static void RecordEffectBatch(ReferenceTrace& trace, EventId first, std::size_t count, std::uint32_t phase) {
    trace.effect_phases_.insert(first, count, phase);
  }
  static void CopyWaitBatch(ReferenceTrace& trace, EventId first, EventId rest, std::size_t count) {
    if (const auto phase = trace.waitTarget(first)) trace.wait_targets_.insert(rest, count, *phase);
    if (const auto barrier = trace.initialWaitBarrier(first)) trace.initial_waits_.insert(rest, count, *barrier);
  }
  static void RecordWaitBatch(ReferenceTrace& trace, EventId first, std::size_t count, PhaseKey phase) {
    trace.wait_targets_.insert(first, count, phase);
  }

  static void RecordWaitTarget(ReferenceTrace& trace, EventId event,
                               PhaseKey phase) {
    trace.recordWaitTarget(event, phase);
  }

  static void RecordInitialWait(ReferenceTrace& trace, EventId event,
                                BarrierLifetimeId barrier) {
    trace.recordInitialWait(event, barrier);
  }

  static void RecordStateToken(ReferenceTrace& trace, StateId state,
                               StateToken token) {
    trace.recordStateToken(state, token);
  }
};

namespace {

template <typename... Visitors>
struct Overloaded : Visitors... {
  using Visitors::operator()...;
};

template <typename... Visitors>
Overloaded(Visitors...) -> Overloaded<Visitors...>;

SimulationError semanticError(
    SimulationErrorCode code, std::string message,
    std::optional<EventId> event = std::nullopt,
    std::optional<BarrierLifetimeId> barrier = std::nullopt,
    std::optional<SyncId> sync_id = std::nullopt,
    std::optional<StateId> state = std::nullopt) {
  SemanticViolation violation{code,    event, barrier,
                              sync_id, state, std::move(message)};
  return SimulationError{code, std::move(violation), {}};
}

SimulationErrorCode mapBarrierError(BarrierRuntimeErrorCode code) {
  switch (code) {
    case BarrierRuntimeErrorCode::kNamedArrivalCountMismatch:
      return SimulationErrorCode::kNamedArrivalCountMismatch;
    case BarrierRuntimeErrorCode::kAlreadyInitialized:
      return SimulationErrorCode::kDuplicateInitialization;
    case BarrierRuntimeErrorCode::kZeroExpectedArrivals:
      return SimulationErrorCode::kZeroExpectedArrivals;
    case BarrierRuntimeErrorCode::kNotInitialized:
      return SimulationErrorCode::kUseBeforeInitialization;
    case BarrierRuntimeErrorCode::kArrivalUnderflow:
      return SimulationErrorCode::kArrivalUnderflow;
    case BarrierRuntimeErrorCode::kArrivalOverflow:
      return SimulationErrorCode::kArrivalOverflow;
    case BarrierRuntimeErrorCode::kTransactionOverflow:
      return SimulationErrorCode::kTransactionOverflow;
    case BarrierRuntimeErrorCode::kTransactionUnderflow:
      return SimulationErrorCode::kTransactionUnderflow;
    case BarrierRuntimeErrorCode::kPhaseOverflow:
      return SimulationErrorCode::kPhaseOverflow;
    case BarrierRuntimeErrorCode::kStateBarrierMismatch:
      return SimulationErrorCode::kStateBarrierMismatch;
    case BarrierRuntimeErrorCode::kStaleState:
      return SimulationErrorCode::kStaleStateWait;
    case BarrierRuntimeErrorCode::kInvalidParity:
      return SimulationErrorCode::kInvalidParity;
    case BarrierRuntimeErrorCode::kStaleParity:
      return SimulationErrorCode::kStaleParityWait;
    case BarrierRuntimeErrorCode::kWaitForIncompletePhase:
      return SimulationErrorCode::kWaitTargetIncomplete;
    case BarrierRuntimeErrorCode::kMissingSuccessfulWaitHandoff:
      return SimulationErrorCode::kMissingSuccessfulWaitHandoff;
  }
  return SimulationErrorCode::kMalformedProgram;
}

std::string barrierErrorMessage(BarrierRuntimeErrorCode code) {
  switch (code) {
    case BarrierRuntimeErrorCode::kNamedArrivalCountMismatch:
      return "named barrier participants specify different arrival counts in the same phase";
    case BarrierRuntimeErrorCode::kAlreadyInitialized:
      return "barrier lifetime was initialized more than once";
    case BarrierRuntimeErrorCode::kZeroExpectedArrivals:
      return "barrier initialization has zero expected arrivals";
    case BarrierRuntimeErrorCode::kNotInitialized:
      return "barrier effect or wait occurred before initialization";
    case BarrierRuntimeErrorCode::kArrivalUnderflow:
      return "barrier arrival count underflow";
    case BarrierRuntimeErrorCode::kArrivalOverflow:
      return "barrier arrival count overflow";
    case BarrierRuntimeErrorCode::kTransactionOverflow:
      return "barrier transaction count overflow";
    case BarrierRuntimeErrorCode::kTransactionUnderflow:
      return "barrier transaction count violates its configured lower bound";
    case BarrierRuntimeErrorCode::kPhaseOverflow:
      return "barrier phase number overflow";
    case BarrierRuntimeErrorCode::kStateBarrierMismatch:
      return "state token belongs to a different barrier lifetime";
    case BarrierRuntimeErrorCode::kStaleState:
      return "state token is not from the current or previous phase";
    case BarrierRuntimeErrorCode::kInvalidParity:
      return "wait parity must be zero or one";
    case BarrierRuntimeErrorCode::kStaleParity:
      return "parity does not denote a current or previous phase";
    case BarrierRuntimeErrorCode::kWaitForIncompletePhase:
      return "wait returned before its target phase completed";
    case BarrierRuntimeErrorCode::kMissingSuccessfulWaitHandoff:
      return "next-phase arrival occurred before a successful prior-phase wait";
  }
  return "unknown barrier runtime error";
}

struct SimulatorState final {
  const LoweredProgram& program;
  bool reject_negative_tx;
  ReferenceTrace trace;
  std::vector<bool> executed;
  std::vector<std::size_t> remaining_predecessors;
  std::size_t executed_count = 0;
  std::map<BarrierLifetimeId, BarrierRuntime> barriers;
  std::map<SyncId, CollectiveRuntime> collectives;
  std::map<InstId, EventId> unindexed_wait_issues;
  std::set<EventId> normal_ready;
  std::set<EventId> urgent_ready;
  std::map<PhaseKey, std::vector<EventId>> parked_phase_waits;
  std::map<SyncId, std::vector<EventId>> parked_collective_waits;
  std::map<PhaseKey, std::vector<TemplateSchedule::Batch>> parked_family_waits;
  std::map<SyncId, std::vector<TemplateSchedule::Batch>> parked_family_collectives;
  std::optional<TemplateSchedule> schedule;

  explicit SimulatorState(const LoweredProgram& input,
                          const SimulationOptions& options, bool compact = false)
      : program(input),
        reject_negative_tx(options.reject_negative_tx),
        executed(compact ? 0 : input.graph().eventCount(), false),
        remaining_predecessors(compact ? 0 : input.graph().eventCount(), 0) {
    if (!compact) ReferenceTraceRecorder::SetFamilies(trace, std::make_shared<EventFamilies>(input.graph().events()));
    if (compact) {
      schedule.emplace(input.graph());
      ReferenceTraceRecorder::SetFamilies(trace, schedule->layout());
      if (options.statistics) {
        options.statistics->event_families = schedule->familyCount();
        options.statistics->dependency_relations = schedule->relationCount();
      }
    } else for (const Event& event : input.graph().events()) {
      remaining_predecessors[event.id.value()] =
          input.graph().predecessors(event.id).size();
      if (remaining_predecessors[event.id.value()] == 0) {
        if (event.scheduling_class == EventSchedulingClass::kNormal) {
          normal_ready.insert(event.id);
        } else if (event.isUrgentDerived()) {
          urgent_ready.insert(event.id);
        }
      }
    }
    for (const auto& [sync_id, collective] : input.collectives().entries()) {
      std::vector<EventId> issues;
      issues.reserve(collective.participants.size());
      for (const CollectiveParticipant& participant : collective.participants) {
        issues.push_back(participant.milestones.issue);
      }
      collectives.emplace(sync_id, CollectiveRuntime(sync_id, issues));
    }
  }
};

template <class Value>
void ReportTreeNodeCount(MemoryReport& report, const char* label,
                         std::uint64_t count) {
#ifdef __GLIBCXX__
  constexpr auto bytes = sizeof(std::_Rb_tree_node<Value>);
  constexpr bool estimated = false;
#else
  constexpr auto bytes = sizeof(Value) + 4 * sizeof(void*);
  constexpr bool estimated = true;
#endif
  report.Add(label, count * bytes, count * bytes, count, count, estimated,
             bytes);
  report.Metric(std::string(label) + ".payload_bytes", count * sizeof(Value));
}

void ReportSimulatorMemory(const SimulatorState& state, const char* stage) {
  if (!MemoryProfileEnabled()) return;
  MemoryCheckpoint(stage);
  MemoryReport report(stage, "simulator_temporaries_excluding_trace", &state);
  // The separately emitted trace census includes its inline object header.
  report.Object("object_excluding_trace", sizeof(state) - sizeof(state.trace));
  report.Add("executed_bits", (state.executed.size() + CHAR_BIT - 1) / CHAR_BIT,
             (state.executed.capacity() + CHAR_BIT - 1) / CHAR_BIT,
             state.executed.size(), state.executed.capacity() != 0);
  report.Vector("remaining_predecessors", state.remaining_predecessors);
  report.Tree("barriers", state.barriers);
  report.Tree("collectives", state.collectives);
  report.Tree("normal_ready", state.normal_ready);
  report.Tree("urgent_ready", state.urgent_ready);
  report.Tree("parked_phase_waits", state.parked_phase_waits);
  report.Tree("parked_collective_waits", state.parked_collective_waits);
  std::uint64_t successful_waits = 0;
  for (const auto& [barrier, runtime] : state.barriers) {
    static_cast<void>(barrier);
    successful_waits += runtime.successfulWaitCount();
  }
  ReportTreeNodeCount<std::uint32_t>(report, "barriers.successful_waits",
                                    successful_waits);
  std::uint64_t expected = 0, arrived = 0;
  for (const auto& [sync, runtime] : state.collectives) {
    static_cast<void>(sync);
    expected += runtime.expectedCount();
    arrived += runtime.arrivalCount();
  }
  ReportTreeNodeCount<EventId>(report, "collectives.expected", expected);
  ReportTreeNodeCount<EventId>(report, "collectives.arrived", arrived);
  for (const auto& [phase, events] : state.parked_phase_waits) {
    static_cast<void>(phase);
    report.Vector("parked_phase_waits.events", events);
  }
  for (const auto& [sync, events] : state.parked_collective_waits) {
    static_cast<void>(sync);
    report.Vector("parked_collective_waits.events", events);
  }
  report.Metric("executed_count", state.executed_count);
  report.Metric("total_event_count", state.program.graph().eventCount());
  report.Emit();
  ReportTraceMemory(state.trace, stage);
}

void ReportSimulationError(const SimulationError& error, const char* stage) {
  if (!MemoryProfileEnabled()) return;
  std::string detail = "code=" + std::to_string(static_cast<int>(error.code));
  if (const auto* violation = std::get_if<SemanticViolation>(&error.witness)) {
    detail += " message=" + violation->message;
    if (violation->event) {
      detail += " event=" + std::to_string(violation->event->value());
    }
  }
  MemoryCheckpoint(stage, detail.c_str());
}

std::optional<SimulationError> barrierFailure(
    const BarrierRuntimeError& error, EventId event,
    std::optional<StateId> state = std::nullopt) {
  const SimulationErrorCode code = mapBarrierError(error.code);
  return semanticError(code, barrierErrorMessage(error.code), event,
                       error.barrier, std::nullopt, state);
}

BarrierRuntime* findBarrier(SimulatorState& state, BarrierLifetimeId barrier) {
  const auto found = state.barriers.find(barrier);
  return found == state.barriers.end() ? nullptr : &found->second;
}

const BarrierRuntime* findBarrier(const SimulatorState& state,
                                  BarrierLifetimeId barrier) {
  const auto found = state.barriers.find(barrier);
  return found == state.barriers.end() ? nullptr : &found->second;
}

template <typename Key>
void wakeWaiters(SimulatorState& state,
                 std::map<Key, std::vector<EventId>>& parked, const Key& key) {
  if (state.schedule) {
    auto& groups = [&]() -> auto& {
      if constexpr (std::is_same_v<Key, PhaseKey>) return state.parked_family_waits;
      else return state.parked_family_collectives;
    }();
    if (auto family = groups.find(key); family != groups.end()) {
      for (const auto& batch : family->second) state.schedule->wake(batch);
      groups.erase(family);
    }
  }
  const auto found = parked.find(key);
  if (found == parked.end()) return;
  if (state.schedule) {
    for (auto id : found->second) state.schedule->wake(id);
  } else state.normal_ready.insert(found->second.begin(), found->second.end());
  parked.erase(found);
}

void appendTransition(SimulatorState& state, EventId event,
                      const BarrierTransition& transition,
                      std::optional<StateId> produced_state) {
  ReferenceTraceRecorder::RecordEffectPhase(state.trace, event,
                                            transition.effect_phase);
  if (produced_state.has_value() && transition.produced_state.has_value()) {
    ReferenceTraceRecorder::RecordStateToken(state.trace, *produced_state,
                                             *transition.produced_state);
  }
  if (transition.completed_phase.has_value()) {
    ReferenceTraceRecorder::AppendBarrierCompletion(
        state.trace, *transition.completed_phase);
    wakeWaiters(state, state.parked_phase_waits, *transition.completed_phase);
  }
}

std::optional<EventId> issueEvent(const SimulatorState& state, const Event& event) {
  if (!event.origin) return {};
  const auto milestones = state.program.origins().milestones(*event.origin);
  if (milestones) return milestones->issue;
  const auto found = state.unindexed_wait_issues.find(*event.origin);
  return found == state.unindexed_wait_issues.end() ? std::nullopt : std::optional(found->second);
}

std::optional<PhaseKey> issuedTarget(const SimulatorState& state, const Event& event) {
  const auto issue = issueEvent(state, event);
  return issue ? state.trace.waitTarget(*issue) : std::nullopt;
}

std::optional<BarrierLifetimeId> issuedInitial(const SimulatorState& state, const Event& event) {
  const auto issue = issueEvent(state, event);
  return issue ? state.trace.initialWaitBarrier(*issue) : std::nullopt;
}

std::optional<SimulationError> executeNormal(SimulatorState& state,
                                             const Event& event) {
  ReferenceTraceRecorder::AppendNormal(state.trace, event.id);
  if (event.origin && !state.program.origins().milestones(*event.origin) &&
      (std::holds_alternative<WaitStateIssueEvent>(event.payload) ||
       std::holds_alternative<WaitParityIssueEvent>(event.payload) ||
       std::holds_alternative<NamedBarrierArriveEvent>(event.payload)))
    state.unindexed_wait_issues.emplace(*event.origin, event.id);

  return std::visit(
      Overloaded{
          [&](const NamedBarrierArriveEvent& payload)
              -> std::optional<SimulationError> {
            auto [it, inserted] = state.barriers.try_emplace(payload.barrier, payload.barrier);
            auto result = it->second.NamedArrive(payload.expected_arrivals);
            if (!result) return barrierFailure(result.error(), event.id);
            if (payload.blocking) {
              const PhaseKey target{payload.barrier, result.value().effect_phase};
              ReferenceTraceRecorder::RecordWaitTarget(state.trace, event.id, target);
            }
            appendTransition(state, event.id, result.value(), std::nullopt);
            return std::nullopt;
          },
          [&](const NamedBarrierReturnEvent& payload)
              -> std::optional<SimulationError> {
            const auto target = issuedTarget(state, event).value();
            const auto* barrier = findBarrier(state, payload.barrier);
            if (!barrier || !barrier->isPhaseCompleted(target.phase))
              return semanticError(SimulationErrorCode::kWaitTargetIncomplete,
                  "named barrier phase has not completed", event.id, payload.barrier);
            ReferenceTraceRecorder::RecordWaitTarget(state.trace, event.id, target);
            return std::nullopt;
          },
          [&](const MBarrierInitEvent& payload)
              -> std::optional<SimulationError> {
            auto [it, inserted] =
                state.barriers.try_emplace(payload.barrier, payload.barrier,
                                          state.reject_negative_tx);
            (void)inserted;
            auto result = it->second.Initialize(payload.expected_arrivals);
            if (!result) {
              return barrierFailure(result.error(), event.id);
            }
            return std::nullopt;
          },
          [&](const MBarrierUpdateEvent& payload)
              -> std::optional<SimulationError> {
            BarrierRuntime* barrier = findBarrier(state, payload.barrier);
            if (barrier == nullptr) {
              return semanticError(
                  SimulationErrorCode::kUseBeforeInitialization,
                  "barrier update occurred before initialization", event.id,
                  payload.barrier);
            }
            auto result = barrier->ApplyUpdate(
                payload.arrival_decrement, payload.transaction_increment,
                payload.produced_state.has_value());
            if (!result) {
              return barrierFailure(result.error(), event.id,
                                    payload.produced_state);
            }
            appendTransition(state, event.id, result.value(),
                             payload.produced_state);
            return std::nullopt;
          },
          [&](const MBarrierCompleteTxEvent& payload)
              -> std::optional<SimulationError> {
            BarrierRuntime* barrier = findBarrier(state, payload.barrier);
            if (barrier == nullptr) {
              return semanticError(
                  SimulationErrorCode::kUseBeforeInitialization,
                  "complete_tx occurred before barrier initialization",
                  event.id, payload.barrier);
            }
            auto result = barrier->CompleteTx(payload.bytes);
            if (!result) {
              return barrierFailure(result.error(), event.id);
            }
            appendTransition(state, event.id, result.value(), std::nullopt);
            return std::nullopt;
          },
          [&](const MBarrierAddArrivalsEvent& payload)
              -> std::optional<SimulationError> {
            BarrierRuntime* barrier = findBarrier(state, payload.barrier);
            if (barrier == nullptr) {
              return semanticError(
                  SimulationErrorCode::kUseBeforeInitialization,
                  "arrival increment occurred before initialization", event.id,
                  payload.barrier);
            }
            auto result = barrier->AddArrivals(payload.arrivals);
            if (!result) return barrierFailure(result.error(), event.id);
            appendTransition(state, event.id, result.value(), std::nullopt);
            return std::nullopt;
          },
          [&](const AsyncMemoryExecEvent& payload)
              -> std::optional<SimulationError> {
            if (!payload.completion.has_value()) {
              return std::nullopt;
            }
            BarrierRuntime* barrier =
                findBarrier(state, payload.completion->barrier);
            if (barrier == nullptr) {
              return semanticError(
                  SimulationErrorCode::kUseBeforeInitialization,
                  "async completion affected an uninitialized barrier",
                  event.id, payload.completion->barrier);
            }
            auto result = barrier->CompleteTx(payload.completion->bytes);
            if (!result) {
              return barrierFailure(result.error(), event.id);
            }
            appendTransition(state, event.id, result.value(), std::nullopt);
            return std::nullopt;
          },
          [&](const GenericAsyncIssueEvent& payload)
              -> std::optional<SimulationError> {
            for (BarrierLifetimeId target : payload.barriers) {
              if (findBarrier(state, target) == nullptr) {
                return semanticError(
                    SimulationErrorCode::kUseBeforeInitialization,
                    "async request references an uninitialized barrier",
                    event.id, target);
              }
            }
            return std::nullopt;
          },
          [&](const GenericAsyncEffectEvent& payload)
              -> std::optional<SimulationError> {
            if (!payload.completion) return std::nullopt;
            return std::visit(
                [&](const auto& action) -> std::optional<SimulationError> {
                  BarrierRuntime* barrier = findBarrier(state, action.barrier);
                  if (barrier == nullptr) {
                    return semanticError(
                        SimulationErrorCode::kUseBeforeInitialization,
                        "async effect references an uninitialized barrier",
                        event.id, action.barrier);
                  }
                  auto result = [&]() {
                    if constexpr (std::is_same_v<std::decay_t<decltype(action)>,
                                                 ArriveEffect>) {
                      return barrier->ApplyUpdate(action.arrivals, 0, false);
                    } else {
                      return barrier->CompleteTx(action.bytes);
                    }
                  }();
                  if (!result) return barrierFailure(result.error(), event.id);
                  appendTransition(state, event.id, result.value(),
                                   std::nullopt);
                  return std::nullopt;
                },
                *payload.completion);
          },
          [&](const WaitStateIssueEvent& payload)
              -> std::optional<SimulationError> {
            if (!event.origin.has_value()) {
              return semanticError(SimulationErrorCode::kMalformedProgram,
                                   "wait issue has no source origin", event.id,
                                   payload.barrier, std::nullopt,
                                   payload.state);
            }
            const auto token = state.trace.stateToken(payload.state);
            if (!token.has_value()) {
              return semanticError(
                  SimulationErrorCode::kMissingStateToken,
                  "state wait referenced a state that has not been produced",
                  event.id, payload.barrier, std::nullopt, payload.state);
            }
            BarrierRuntime* barrier = findBarrier(state, payload.barrier);
            if (barrier == nullptr) {
              return semanticError(
                  SimulationErrorCode::kUseBeforeInitialization,
                  "state wait occurred before barrier initialization", event.id,
                  payload.barrier, std::nullopt, payload.state);
            }
            auto target = barrier->SelectStateWaitTarget(*token);
            if (!target) {
              return barrierFailure(target.error(), event.id, payload.state);
            }
            if (issuedTarget(state, event)) {
              return semanticError(SimulationErrorCode::kMalformedProgram,
                                   "wait origin was issued more than once",
                                   event.id, payload.barrier);
            }
            ReferenceTraceRecorder::RecordWaitTarget(state.trace, event.id,
                                                     target.value());
            return std::nullopt;
          },
          [&](const WaitParityIssueEvent& payload)
              -> std::optional<SimulationError> {
            if (!event.origin.has_value()) {
              return semanticError(SimulationErrorCode::kMalformedProgram,
                                   "wait issue has no source origin", event.id,
                                   payload.barrier);
            }
            BarrierRuntime* barrier = findBarrier(state, payload.barrier);
            if (barrier == nullptr) {
              return semanticError(
                  SimulationErrorCode::kUseBeforeInitialization,
                  "parity wait occurred before barrier initialization",
                  event.id, payload.barrier);
            }
            auto target = barrier->SelectParityWaitTarget(payload.parity);
            if (!target) {
              return barrierFailure(target.error(), event.id);
            }
            if (issuedTarget(state, event) || issuedInitial(state, event)) {
              return semanticError(SimulationErrorCode::kMalformedProgram,
                                   "wait origin was issued more than once",
                                   event.id, payload.barrier);
            }
            if (target.value()) {
              ReferenceTraceRecorder::RecordWaitTarget(state.trace, event.id,
                                                       *target.value());
            } else {
              ReferenceTraceRecorder::RecordInitialWait(state.trace, event.id,
                                                        payload.barrier);
            }
            return std::nullopt;
          },
          [&](const WaitStateDoneEvent& payload)
              -> std::optional<SimulationError> {
            if (!event.origin.has_value() ||
                !issuedTarget(state, event)) {
              return semanticError(
                  SimulationErrorCode::kWaitDoneBeforeIssue,
                  "state wait completed without a matching issued wait",
                  event.id, payload.barrier, std::nullopt, payload.state);
            }
            const PhaseKey target = issuedTarget(state, event).value();
            BarrierRuntime* barrier = findBarrier(state, payload.barrier);
            if (barrier == nullptr ||
                !barrier->isPhaseCompleted(target.phase)) {
              return semanticError(SimulationErrorCode::kWaitTargetIncomplete,
                                   "state wait target has not completed",
                                   event.id, payload.barrier, std::nullopt,
                                   payload.state);
            }
            auto recorded = barrier->RecordSuccessfulWait(target.phase);
            if (!recorded) {
              return barrierFailure(recorded.error(), event.id, payload.state);
            }
            ReferenceTraceRecorder::RecordWaitTarget(state.trace, event.id,
                                                     target);
            return std::nullopt;
          },
          [&](const WaitParityDoneEvent& payload)
              -> std::optional<SimulationError> {
            if (event.origin.has_value() &&
                issuedInitial(state, event)) {
              if (issuedInitial(state, event) != payload.barrier ||
                  payload.parity != 1) {
                return semanticError(SimulationErrorCode::kMalformedProgram,
                                     "initial parity wait has a mismatched result",
                                     event.id, payload.barrier);
              }
              ReferenceTraceRecorder::RecordInitialWait(state.trace, event.id,
                                                        payload.barrier);
              return std::nullopt;
            }
            if (!event.origin.has_value() ||
                !issuedTarget(state, event)) {
              return semanticError(
                  SimulationErrorCode::kWaitDoneBeforeIssue,
                  "parity wait completed without a matching issued wait",
                  event.id, payload.barrier);
            }
            const PhaseKey target = issuedTarget(state, event).value();
            BarrierRuntime* barrier = findBarrier(state, payload.barrier);
            if (barrier == nullptr ||
                !barrier->isPhaseCompleted(target.phase)) {
              return semanticError(SimulationErrorCode::kWaitTargetIncomplete,
                                   "parity wait target has not completed",
                                   event.id, payload.barrier);
            }
            auto recorded = barrier->RecordSuccessfulWait(target.phase);
            if (!recorded) {
              return barrierFailure(recorded.error(), event.id);
            }
            ReferenceTraceRecorder::RecordWaitTarget(state.trace, event.id,
                                                     target);
            return std::nullopt;
          },
          [&](const CollectiveIssueEvent& payload)
              -> std::optional<SimulationError> {
            const auto found = state.collectives.find(payload.sync_id);
            if (found == state.collectives.end()) {
              return semanticError(
                  SimulationErrorCode::kCollectiveMismatch,
                  "collective issue has no matching collective table entry",
                  event.id, std::nullopt, payload.sync_id);
            }
            auto arrived = found->second.Arrive(event.id);
            if (!arrived) {
              return semanticError(SimulationErrorCode::kCollectiveMismatch,
                                   "invalid collective participant arrival",
                                   event.id, std::nullopt, payload.sync_id);
            }
            return std::nullopt;
          },
          [&](const CollectiveDoneEvent& payload)
              -> std::optional<SimulationError> {
            const auto found = state.collectives.find(payload.sync_id);
            if (found == state.collectives.end() ||
                !found->second.completed()) {
              return semanticError(
                  SimulationErrorCode::kCollectiveMismatch,
                  "collective participant returned before completion", event.id,
                  std::nullopt, payload.sync_id);
            }
            return std::nullopt;
          },
          [&](const CollectiveCompleteEvent& payload)
              -> std::optional<SimulationError> {
            return semanticError(
                SimulationErrorCode::kUnexpectedUrgentEvent,
                "collective completion was scheduled as a normal event",
                event.id, std::nullopt, payload.sync_id);
          },
          [&](const MBarrierPhaseCompleteEvent& payload)
              -> std::optional<SimulationError> {
            return semanticError(
                SimulationErrorCode::kUnexpectedUrgentEvent,
                "explicit barrier completion is not part of a lowered program",
                event.id, payload.barrier);
          },
          [&](const auto&) -> std::optional<SimulationError> {
            return std::nullopt;
          }},
      event.payload);
}

bool normalSemanticallyReady(const SimulatorState& state, const Event& event) {
  return std::visit(
      Overloaded{[&](const NamedBarrierReturnEvent& payload) {
                   const auto target = issuedTarget(state, event);
                   const auto* barrier = findBarrier(state, payload.barrier);
                   return !target || !barrier || barrier->isPhaseCompleted(target->phase);
                 },
                 [&](const WaitStateDoneEvent& payload) {
                   if (!event.origin.has_value()) {
                     return true;
                   }
                   const auto target = issuedTarget(state, event);
                   if (!target) {
                     return true;
                   }
                   const BarrierRuntime* barrier =
                       findBarrier(state, payload.barrier);
                   return barrier == nullptr ||
                          barrier->isPhaseCompleted(target->phase);
                 },
                 [&](const WaitParityDoneEvent& payload) {
                   if (!event.origin.has_value()) {
                     return true;
                   }
                   const auto target = issuedTarget(state, event);
                   if (!target) {
                     return true;
                   }
                   const BarrierRuntime* barrier =
                       findBarrier(state, payload.barrier);
                   return barrier == nullptr ||
                          barrier->isPhaseCompleted(target->phase);
                 },
                 [&](const CollectiveDoneEvent& payload) {
                   const auto found = state.collectives.find(payload.sync_id);
                   return found == state.collectives.end() ||
                          found->second.completed();
                 },
                 [&](const auto&) { return true; }},
      event.payload);
}

bool urgentSemanticallyReady(const SimulatorState& state, const Event& event) {
  const auto* completion = std::get_if<CollectiveCompleteEvent>(&event.payload);
  if (completion == nullptr) {
    return true;
  }
  const auto found = state.collectives.find(completion->sync_id);
  return found == state.collectives.end() || found->second.readyToComplete();
}

void markExecuted(SimulatorState& state, EventId event) {
  if (state.schedule) {
    state.schedule->finish(state.schedule->one(event));
    ++state.executed_count;
    return;
  }
  state.executed[event.value()] = true;
  ++state.executed_count;
  state.normal_ready.erase(event);
  state.urgent_ready.erase(event);
  for (EventId successor : state.program.graph().successors(event)) {
    if (--state.remaining_predecessors[successor.value()] == 0) {
      const Event& next = state.program.graph().event(successor);
      if (next.scheduling_class == EventSchedulingClass::kNormal) {
        state.normal_ready.insert(successor);
      } else if (next.isUrgentDerived()) {
        state.urgent_ready.insert(successor);
      }
    }
  }
}

bool executeArrivalBatch(SimulatorState& state, const Event& first,
                         SimulationStatistics* statistics) {
  if (state.normal_ready.size() < 2 || *state.normal_ready.begin() != first.id)
    return false;
  const auto* update = std::get_if<MBarrierUpdateEvent>(&first.payload);
  if (!update || !update->arrival_decrement || update->transaction_increment ||
      update->produced_state || !first.actor) return false;
  auto* barrier = findBarrier(state, update->barrier);
  if (!barrier || !barrier->initialized()) return false;
  const auto limit = barrier->pendingArrivals() / update->arrival_decrement;
  if (limit < 2) return false;
  std::vector<EventId> members;
  std::set<ActorId> actors;
  std::uint64_t first_successor = std::uint64_t{UINT32_MAX} + 1;
  for (auto id : state.normal_ready) {
    if (members.size() == limit || id.value() >= first_successor) break;
    const auto event = state.program.graph().event(id);
    const auto* payload = std::get_if<MBarrierUpdateEvent>(&event.payload);
    if (!payload || *payload != *update || !event.actor || actors.contains(*event.actor)) break;
    bool urgent = false;
    auto next = first_successor;
    for (auto successor : state.program.graph().successors(id)) {
      if (state.program.graph().event(successor).isUrgentDerived()) { urgent = true; break; }
      next = std::min(next, std::uint64_t(successor.value()));
    }
    if (urgent) break;
    members.push_back(id);
    actors.insert(*event.actor);
    first_successor = next;
  }
  if (members.size() < 2) return false;
  // The batch ends before a phase boundary or an intervening ready successor.
  auto next_barrier = *barrier;
  auto transition = next_barrier.ApplyUpdate(
      static_cast<std::uint32_t>(members.size() * update->arrival_decrement), 0);
  if (!transition) return false;
  *barrier = std::move(next_barrier);
  ProfileScope profile("simulator.arrival_batch");
  profile.nodes = members.size();
  if (statistics) {
    ++statistics->arrival_batches;
    statistics->batched_arrivals += members.size();
  }
  for (auto id : members) {
    ReferenceTraceRecorder::AppendNormal(state.trace, id);
    ReferenceTraceRecorder::RecordEffectPhase(state.trace, id, transition.value().effect_phase);
    markExecuted(state, id);
  }
  if (transition.value().completed_phase) {
    ReferenceTraceRecorder::AppendBarrierCompletion(state.trace, *transition.value().completed_phase);
    wakeWaiters(state, state.parked_phase_waits, *transition.value().completed_phase);
  }
  return true;
}

bool executeCollectiveBatch(SimulatorState& state, const Event& first, SimulationStatistics* statistics) {
  const auto* issue = std::get_if<CollectiveIssueEvent>(&first.payload);
  if (!state.schedule || !issue) return false;
  auto batch = state.schedule->next(false);
  auto runtime = state.collectives.find(issue->sync_id);
  if (!batch || runtime == state.collectives.end()) return false;
  batch->count = runtime->second.batchPrefix(batch->first, batch->stride, batch->count);
  if (batch->count < 2 || !state.schedule->collectiveBatchSafe(*batch, issue->sync_id)) return false;
  if (!runtime->second.ArriveBatch(batch->first, batch->stride, batch->count)) return false;
  ReferenceTraceRecorder::AppendNormalBatch(state.trace, batch->first, batch->stride, batch->count);
  state.schedule->finish(*batch);
  state.executed_count += batch->count;
  if (statistics) { ++statistics->template_batches; statistics->template_events += batch->count; }
  return true;
}

bool executeTemplateUpdate(SimulatorState& state, const Event& first,
                           SimulationStatistics* statistics) {
  if (!state.schedule) return false;
  auto batch = state.schedule->next(false);
  if (!batch || batch->count < 2 || state.schedule->hasUrgentSuccessor(*batch)) return false;
  std::optional<BarrierLifetimeId> target;
  std::uint32_t arrivals = 0;
  std::uint64_t bytes = 0;
  bool complete_tx = false;
  const auto* named = std::get_if<NamedBarrierArriveEvent>(&first.payload);
  if (named && named->blocking && (!first.origin || !state.program.origins().milestones(*first.origin))) return false;
  if (named) { target = named->barrier; arrivals = 1; }
  std::visit(Overloaded{
    [&](const MBarrierUpdateEvent& value) {
      if (value.produced_state) return;
      target = value.barrier; arrivals = value.arrival_decrement; bytes = value.transaction_increment;
    },
    [&](const MBarrierCompleteTxEvent& value) {
      target = value.barrier; bytes = value.bytes; complete_tx = true;
    },
    [&](const AsyncMemoryExecEvent& value) {
      if (!value.completion) return;
      target = value.completion->barrier; bytes = value.completion->bytes; complete_tx = true;
    },
    [&](const GenericAsyncEffectEvent& value) {
      if (!value.completion) return;
      std::visit([&](const auto& action) {
        target = action.barrier;
        if constexpr (requires { action.bytes; }) { bytes = action.bytes; complete_tx = true; }
        else arrivals = action.arrivals;
      }, *value.completion);
    },
    [](const auto&) {}
  }, first.payload);
  if (!target) return false;
  if (named) state.barriers.try_emplace(*target, *target);
  auto* barrier = findBarrier(state, *target);
  if (!barrier || (!named && !barrier->initialized())) return false;
  if (named && (!barrier->initialized() || barrier->pendingArrivals() == barrier->expectedArrivals()))
    batch->count = std::min(batch->count, std::size_t(named->expected_arrivals));
  else if (arrivals) batch->count = std::min(batch->count, std::size_t(barrier->pendingArrivals() / arrivals));
  if (bytes) batch->count = std::min(batch->count, std::size_t(INT64_MAX / bytes));
  if (complete_tx && bytes && barrier->pendingArrivals() == 0 && barrier->transactionCount() > 0)
    batch->count = std::min(batch->count, std::size_t(barrier->transactionCount() / bytes));
  if (batch->count < 2) return false;
  auto next = *barrier;
  auto result = named ? next.NamedArrive(named->expected_arrivals, batch->count)
      : complete_tx ? next.CompleteTx(bytes * batch->count)
      : next.ApplyUpdate(static_cast<std::uint32_t>(arrivals * batch->count), bytes * batch->count);
  if (!result) return false;
  *barrier = std::move(next);
  ReferenceTraceRecorder::AppendNormalBatch(state.trace, batch->first, batch->stride, batch->count);
  ReferenceTraceRecorder::RecordEffectBatch(state.trace, batch->first, batch->count, result.value().effect_phase);
  if (named && named->blocking)
    ReferenceTraceRecorder::RecordWaitBatch(state.trace, batch->first, batch->count, {*target, result.value().effect_phase});
  state.schedule->finish(*batch);
  state.executed_count += batch->count;
  if (result.value().completed_phase) {
    ReferenceTraceRecorder::AppendBarrierCompletion(state.trace, *result.value().completed_phase);
    wakeWaiters(state, state.parked_phase_waits, *result.value().completed_phase);
  }
  if (statistics) {
    ++statistics->template_batches; statistics->template_events += batch->count;
    if (arrivals) { ++statistics->arrival_batches; statistics->batched_arrivals += batch->count; }
  }
  return true;
}

std::optional<SimulationError> executeUrgent(SimulatorState& state,
                                             const Event& event) {
  const auto* completion = std::get_if<CollectiveCompleteEvent>(&event.payload);
  if (completion == nullptr) {
    return semanticError(SimulationErrorCode::kUnexpectedUrgentEvent,
                         "lowered graph contains an unsupported urgent event",
                         event.id);
  }
  const auto found = state.collectives.find(completion->sync_id);
  if (found == state.collectives.end()) {
    return semanticError(SimulationErrorCode::kCollectiveMismatch,
                         "collective completion has no runtime state", event.id,
                         std::nullopt, completion->sync_id);
  }
  auto completed = found->second.Complete(event.id);
  if (!completed) {
    return semanticError(SimulationErrorCode::kCollectiveMismatch,
                         "collective completed before all participants arrived",
                         event.id, std::nullopt, completion->sync_id);
  }
  ReferenceTraceRecorder::AppendCollectiveCompletion(
      state.trace, completion->sync_id, event.id);
  wakeWaiters(state, state.parked_collective_waits, completion->sync_id);
  return std::nullopt;
}

bool parkWaiter(SimulatorState& state, const Event& event) {
  return std::visit(Overloaded{
      [&](const NamedBarrierReturnEvent&) {
        const auto target = issuedTarget(state, event).value();
        state.parked_phase_waits[target].push_back(event.id);
        return true;
      },
      [&](const WaitStateDoneEvent& payload) {
        const auto target = issuedTarget(state, event).value();
        state.parked_phase_waits[{payload.barrier, target.phase}].push_back(event.id);
        return true;
      },
      [&](const WaitParityDoneEvent& payload) {
        const auto target = issuedTarget(state, event).value();
        state.parked_phase_waits[{payload.barrier, target.phase}].push_back(event.id);
        return true;
      },
      [&](const CollectiveDoneEvent& payload) {
        state.parked_collective_waits[payload.sync_id].push_back(event.id);
        return true;
      },
      [&](const auto&) { return false; }}, event.payload);
}

void restrictWaitBatch(const SimulatorState& state, TemplateSchedule::Batch& batch) {
  const auto issue = state.schedule->correspondingIssue(batch);
  if (!issue) { batch.count = 1; return; }
  const auto phase_count = state.trace.waitTargets().uniformPrefix(issue->first, issue->count);
  const auto initial_count = state.trace.initialWaits().uniformPrefix(issue->first, issue->count);
  batch.count = std::max(std::size_t{1}, std::max(phase_count, initial_count));
}

std::optional<EventId> smallestReady(SimulatorState& state,
                                     EventSchedulingClass scheduling_class) {
  if (state.schedule) {
    const bool urgent = scheduling_class == EventSchedulingClass::kUrgentDerived;
    while (auto batch = state.schedule->next(urgent)) {
      const auto event = state.program.graph().event(batch->first);
      if (urgent ? urgentSemanticallyReady(state, event) : normalSemanticallyReady(state, event))
        return event.id;
      if (urgent) return {};
      if (const auto* collective = std::get_if<CollectiveDoneEvent>(&event.payload)) {
        state.parked_family_collectives[collective->sync_id].push_back(*batch);
        state.schedule->park(*batch);
      } else if (const auto phase = issuedTarget(state, event)) {
        restrictWaitBatch(state, *batch);
        state.parked_family_waits[*phase].push_back(*batch);
        state.schedule->park(*batch);
      } else if (parkWaiter(state, event)) state.schedule->park(state.schedule->one(event.id));
      else return {};
    }
    return {};
  }
  auto& ready = scheduling_class == EventSchedulingClass::kUrgentDerived
                    ? state.urgent_ready
                    : state.normal_ready;
  for (auto position = ready.begin(); position != ready.end();) {
    const Event& event = state.program.graph().event(*position);
    const bool semantic_ready =
        scheduling_class == EventSchedulingClass::kUrgentDerived
            ? urgentSemanticallyReady(state, event)
            : normalSemanticallyReady(state, event);
    if (semantic_ready) return event.id;
    if (scheduling_class == EventSchedulingClass::kNormal && parkWaiter(state, event)) {
      position = ready.erase(position);
    } else {
      ++position;
    }
  }
  return std::nullopt;
}

SimulationError deadlockError(const SimulatorState& state) {
  ReportSimulatorMemory(state, "sync.simulator.deadlock.before_witness");
  ReferenceDeadlock witness;
  witness.executed.assign(state.trace.executedEvents().begin(),
                          state.trace.executedEvents().end());
  witness.blocked.reserve(state.program.graph().eventCount() - state.executed_count);
  for (const Event& event : state.program.graph().events()) {
    if (!(state.schedule ? state.schedule->executed(event.id) : state.executed[event.id.value()])) {
      witness.blocked.push_back(event.id);
    }
  }
  MemoryCheckpoint("sync.simulator.deadlock.before_trace_copy");
  SimulationError error{SimulationErrorCode::kDeadlock, std::move(witness),
                        state.trace};
  ReportTraceMemory(error.prefix, "sync.simulator.deadlock.after_trace_copy");
  return error;
}

}  // namespace

Result<ReferenceTrace, SimulationError> ReferenceSimulator::Run(
    const LoweredProgram& program, const SimulationOptions& options) const {
  if (!program.graph().isAcyclic()) {
    return Result<ReferenceTrace, SimulationError>::Failure(
        semanticError(SimulationErrorCode::kMalformedProgram,
                      "lowered event graph is cyclic"));
  }

  const bool compact = options.batch_templates &&
      program.graph().events().storedBindingCount() < program.graph().eventCount();
  SimulatorState state(program, options, compact);
  ReportSimulatorMemory(state, "sync.simulator.state_initialized");
  while (state.executed_count < program.graph().eventCount()) {
    while (true) {
      const std::optional<EventId> urgent =
          smallestReady(state, EventSchedulingClass::kUrgentDerived);
      if (!urgent.has_value()) {
        break;
      }
      const Event& event = program.graph().event(*urgent);
      if (auto error = executeUrgent(state, event); error.has_value()) {
        ReportSimulationError(*error, "sync.simulator.urgent_error");
        ReportSimulatorMemory(state, "sync.simulator.urgent_error.before_trace_copy");
        error->prefix = state.trace;
        ReportTraceMemory(error->prefix, "sync.simulator.urgent_error.after_trace_copy");
        return Result<ReferenceTrace, SimulationError>::Failure(
            std::move(*error));
      }
      markExecuted(state, *urgent);
    }

    if (state.executed_count == program.graph().eventCount()) {
      break;
    }

    const std::optional<EventId> normal =
        smallestReady(state, EventSchedulingClass::kNormal);
    if (!normal.has_value()) {
      return Result<ReferenceTrace, SimulationError>::Failure(
          deadlockError(state));
    }
    const Event& event = program.graph().event(*normal);
    if (executeCollectiveBatch(state, event, options.statistics)) continue;
    if (options.batch_arrivals && executeTemplateUpdate(state, event, options.statistics)) continue;
    auto batch = state.schedule ? state.schedule->next(false) : std::nullopt;
    const bool stateless = std::visit(Overloaded{
      [](const OperationIssueEvent&) { return true; },
      [](const OperationCompleteEvent&) { return true; },
      [](const ImmediateMemoryIssueEvent&) { return true; },
      [](const ImmediateMemoryExecEvent&) { return true; },
      [](const AsyncMemoryIssueEvent&) { return true; },
      [](const AsyncMemoryExecEvent& value) { return !value.completion; },
      [](const GenericAsyncIssueEvent&) { return true; },
      [](const GenericAsyncEffectEvent& value) { return !value.completion; },
      [](const GenericAsyncReadsCompleteEvent&) { return true; },
      [](const GenericAsyncCompleteEvent&) { return true; },
      [](const AwaitIssueEvent&) { return true; },
      [](const AwaitDoneEvent&) { return true; },
      [](const CollectiveDoneEvent&) { return true; },
      [](const WaitParityIssueEvent&) { return true; },
      [](const WaitStateIssueEvent&) { return true; },
      [](const WaitParityDoneEvent&) { return true; },
      [](const WaitStateDoneEvent&) { return true; },
      [](const NamedBarrierReturnEvent&) { return true; },
      [](const auto&) { return false; }
    }, event.payload);
    if (batch && (!stateless || state.schedule->hasUrgentSuccessor(*batch))) batch->count = 1;
    if (batch && batch->count > 1 &&
        (std::holds_alternative<WaitParityIssueEvent>(event.payload) ||
         std::holds_alternative<WaitStateIssueEvent>(event.payload)) &&
        (!event.origin || !state.program.origins().milestones(*event.origin))) batch->count = 1;
    if (batch && batch->count > 1 &&
        (std::holds_alternative<WaitParityDoneEvent>(event.payload) ||
         std::holds_alternative<WaitStateDoneEvent>(event.payload) ||
         std::holds_alternative<NamedBarrierReturnEvent>(event.payload))) restrictWaitBatch(state, *batch);
    if (options.batch_arrivals && executeArrivalBatch(state, event, options.statistics))
      continue;
    if (auto error = executeNormal(state, event); error.has_value()) {
      ReferenceTraceRecorder::DiscardLastNormal(state.trace, *normal);
      ReportSimulationError(*error, "sync.simulator.normal_error");
      ReportSimulatorMemory(state, "sync.simulator.normal_error.before_trace_copy");
      error->prefix = state.trace;
      ReportTraceMemory(error->prefix, "sync.simulator.normal_error.after_trace_copy");
      return Result<ReferenceTrace, SimulationError>::Failure(
          std::move(*error));
    }
    if (batch && batch->count > 1) {
      ReferenceTraceRecorder::AppendNormalBatch(state.trace,
          EventId(batch->first.value() + batch->stride), batch->stride, batch->count - 1);
      ReferenceTraceRecorder::CopyWaitBatch(state.trace, batch->first,
          EventId(batch->first.value() + batch->stride), batch->count - 1);
      state.schedule->finish(*batch);
      state.executed_count += batch->count;
      if (options.statistics) {
        ++options.statistics->template_batches;
        options.statistics->template_events += batch->count;
      }
    } else markExecuted(state, *normal);
  }

  ReportSimulatorMemory(state, "sync.simulator.success.before_return");
  return Result<ReferenceTrace, SimulationError>::Success(
      std::move(state.trace));
}

Result<ReplayResult, SimulationError> ReferenceSimulator::ReplayPrefix(
    const LoweredProgram& program, std::span<const EventId> requested_order,
    const SimulationOptions& options) const {
  if (!program.graph().isAcyclic()) {
    return Result<ReplayResult, SimulationError>::Failure(
        semanticError(SimulationErrorCode::kMalformedProgram,
                      "lowered event graph is cyclic"));
  }

  constexpr std::size_t absent = std::numeric_limits<std::size_t>::max();
  std::vector<std::size_t> positions(program.graph().eventCount(), absent);
  for (std::size_t position = 0; position < requested_order.size(); ++position) {
    const EventId event = requested_order[position];
    if (!program.graph().contains(event) || positions[event.value()] != absent ||
        program.graph().event(event).isUrgentDerived()) {
      return Result<ReplayResult, SimulationError>::Failure(semanticError(
          SimulationErrorCode::kMalformedProgram,
          "candidate replay requires unique normal events", event));
    }
    positions[event.value()] = position;
  }

  SimulatorState state(program, options);
  ReportSimulatorMemory(state, "sync.replay.state_initialized");
  std::set<std::size_t> requested_ready;
  for (EventId event : state.normal_ready) {
    if (positions[event.value()] != absent) {
      requested_ready.insert(positions[event.value()]);
    }
  }
  auto mark_replayed = [&](EventId event) {
    markExecuted(state, event);
    requested_ready.erase(positions[event.value()]);
    for (EventId successor : program.graph().successors(event)) {
      if (state.remaining_predecessors[successor.value()] == 0 &&
          positions[successor.value()] != absent) {
        requested_ready.insert(positions[successor.value()]);
      }
    }
  };
  std::size_t remaining = requested_order.size();
  while (true) {
    while (true) {
      const std::optional<EventId> urgent =
          smallestReady(state, EventSchedulingClass::kUrgentDerived);
      if (!urgent.has_value()) {
        break;
      }
      const Event& event = program.graph().event(*urgent);
      if (auto error = executeUrgent(state, event); error.has_value()) {
        ReportSimulationError(*error, "sync.replay.urgent_error");
        ReportSimulatorMemory(state, "sync.replay.urgent_error.before_trace_copy");
        error->prefix = state.trace;
        ReportTraceMemory(error->prefix, "sync.replay.urgent_error.after_trace_copy");
        return Result<ReplayResult, SimulationError>::Failure(
            std::move(*error));
      }
      mark_replayed(*urgent);
    }

    if (remaining == 0) {
      break;
    }

    std::optional<EventId> next;
    for (std::size_t position : requested_ready) {
      const EventId candidate = requested_order[position];
      const Event& event = program.graph().event(candidate);
      if (!normalSemanticallyReady(state, event)) {
        continue;
      }
      next = candidate;
      break;
    }
    if (!next.has_value()) {
      break;
    }

    const Event& event = program.graph().event(*next);
    if (auto error = executeNormal(state, event); error.has_value()) {
      ReferenceTraceRecorder::DiscardLastNormal(state.trace, *next);
      ReportSimulationError(*error, "sync.replay.normal_error");
      ReportSimulatorMemory(state, "sync.replay.normal_error.before_trace_copy");
      error->prefix = state.trace;
      ReportTraceMemory(error->prefix, "sync.replay.normal_error.after_trace_copy");
      return Result<ReplayResult, SimulationError>::Failure(std::move(*error));
    }
    mark_replayed(*next);
    --remaining;
  }

  ReportSimulatorMemory(state, "sync.replay.before_trace_move");
  ReplayResult replay;
  replay.prefix = std::move(state.trace);
  replay.requested_order.assign(requested_order.begin(), requested_order.end());
  for (EventId event : requested_order) {
    if (!state.executed[event.value()]) {
      replay.unexecuted.push_back(event);
    }
  }
  if (!replay.unexecuted.empty()) {
    replay.stall = ReplayResult::Stall{replay.unexecuted.front()};
  }

  std::size_t previous_position = 0;
  bool has_previous = false;
  for (EventId event : replay.prefix.executedEvents()) {
    const std::size_t position = positions[event.value()];
    if (position == absent) {
      continue;
    }
    if (has_previous && position < previous_position) {
      replay.requested_order_preserved = false;
      break;
    }
    previous_position = position;
    has_previous = true;
  }

  return Result<ReplayResult, SimulationError>::Success(std::move(replay));
}

}  // namespace sentryward::algorithm
