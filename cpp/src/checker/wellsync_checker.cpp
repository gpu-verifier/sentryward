#include "sentryward/algorithm/analysis/parallel_reachability.h"
#include "sentryward/algorithm/diagnostics/performance_profile.h"
#include "sentryward/algorithm/diagnostics/ir_size_profile.h"
#include "sentryward/algorithm/diagnostics/memory_profile.h"
#include "sentryward/algorithm/diagnostics/scoped_deadline.h"
#include "sentryward/algorithm/checker/wellsync_checker.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <queue>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "sentryward/algorithm/analysis/happens_before_graph.h"
#include "sentryward/algorithm/analysis/lifecycle_checker.h"
#include "sentryward/algorithm/analysis/phase_boundary.h"
#include "sentryward/algorithm/analysis/phase_database.h"
#include "sentryward/algorithm/analysis/reachability.h"
#include "sentryward/algorithm/analysis/wait_checker.h"
#include "sentryward/algorithm/event_ir/lowering.h"
#include "sentryward/algorithm/semantics/simulator.h"
#include "sentryward/algorithm/source_ir/validator.h"

namespace sentryward::algorithm {
namespace {

[[nodiscard]] CheckResult InvalidFromDiagnostics(
    std::vector<Diagnostic> diagnostics) {
  return CheckResult::Invalid(
      InvalidInput{std::move(diagnostics), "source validation failed"});
}

[[nodiscard]] CheckResult UnsupportedFromDiagnostics(
    std::vector<Diagnostic> diagnostics) {
  return CheckResult::Unsupported(UnsupportedInput{
      UnsupportedReason::kSourceFeature, std::move(diagnostics),
      "the source program uses a feature outside the v1 model"});
}

template <typename Error>
[[nodiscard]] CheckResult InvalidFrom(Error error, std::string message) {
  return CheckResult::Invalid(
      InvalidInput{std::move(error), std::move(message)});
}

template <typename Witness>
[[nodiscard]] CheckResult RejectedFrom(Witness witness,
                                       RejectionContext context) {
  return CheckResult::Rejected(RejectionWitness{std::move(witness)},
                               std::move(context));
}

[[nodiscard]] RejectionProvenance BuildProvenance(
    const SourceProgram& source, const LoweredProgram& lowered,
    const HappensBeforeGraph* proof = nullptr) {
  MemoryCheckpoint("sync.provenance.begin");
  std::map<InstId, SourceLocation> locations;
  for (const SourceInstruction& instruction : source.instructions) {
    locations.emplace(instruction.id, instruction.location);
  }

  if (MemoryProfileEnabled()) {
    MemoryReport memory("sync.provenance.locations", "source_location_index", &locations);
    memory.Object("map_header", sizeof(locations));
    memory.Tree("locations", locations);
    for (const auto& [id, location] : locations) {
      (void)id;
      memory.String("location.file", location.file);
    }
    memory.Emit();
  }
  RejectionProvenance provenance;
  const EventRecords& events =
      proof == nullptr ? lowered.graph().events() : proof->events();
  for (const Event& event : events) {
    if (event.origin.has_value()) {
      const auto location = locations.find(*event.origin);
      if (location != locations.end()) {
        provenance.events.emplace(
            event.id, SourceEventProvenance{*event.origin, location->second});
        continue;
      }
    }

    SyntheticEventProvenance synthetic;
    if (const auto* completion =
            std::get_if<CollectiveCompleteEvent>(&event.payload)) {
      synthetic.kind = SyntheticEventKind::kCollectiveCompletion;
      synthetic.sync_id = completion->sync_id;
    } else if (const auto* completion =
                   std::get_if<MBarrierPhaseCompleteEvent>(&event.payload)) {
      synthetic.kind = SyntheticEventKind::kMBarrierPhaseCompletion;
      synthetic.phase = PhaseKey{completion->barrier, completion->phase};
    }
    provenance.events.emplace(event.id, std::move(synthetic));
  }
  return provenance;
}

[[nodiscard]] bool CandidateProjectionPreserved(
    const ReferenceTrace& trace, std::span<const EventId> candidate) {
  std::unordered_map<std::uint32_t, std::size_t> positions;
  positions.reserve(candidate.size());
  for (std::size_t index = 0; index < candidate.size(); ++index) {
    positions.emplace(candidate[index].value(), index);
  }
  std::size_t previous = 0;
  bool has_previous = false;
  for (EventId event : trace.executedEvents()) {
    const auto found = positions.find(event.value());
    if (found == positions.end()) {
      continue;
    }
    const std::size_t position = found->second;
    if (has_previous && position < previous) {
      return false;
    }
    previous = position;
    has_previous = true;
  }
  return true;
}

[[nodiscard]] CounterexampleExecution ReplayCandidate(
    const LoweredProgram& lowered, std::span<const EventId> candidate,
    bool reject_negative_tx) {
  CounterexampleExecution execution;
  execution.candidate_order.assign(candidate.begin(), candidate.end());
  for (EventId event : candidate) {
    if (lowered.graph().contains(event) &&
        !lowered.graph().event(event).isUrgentDerived()) {
      execution.replayed_candidate_order.push_back(event);
    }
  }

  auto replay = ReferenceSimulator{}.ReplayPrefix(
      lowered, execution.replayed_candidate_order,
      SimulationOptions{.reject_negative_tx = reject_negative_tx});
  if (replay) {
    ReplayResult result = std::move(replay).value();
    execution.executable_prefix = std::move(result.prefix);
    execution.unexecuted_candidate = std::move(result.unexecuted);
    execution.candidate_order_preserved = result.requested_order_preserved;
    execution.stall = result.stall;
    return execution;
  }

  SimulationError error = std::move(replay).error();
  execution.executable_prefix = std::move(error.prefix);
  execution.stop = std::move(error.witness);
  execution.candidate_order_preserved = CandidateProjectionPreserved(
      execution.executable_prefix, execution.replayed_candidate_order);
  std::unordered_set<std::uint32_t> executed;
  executed.reserve(execution.executable_prefix.executedEvents().size());
  for (EventId event : execution.executable_prefix.executedEvents()) {
    executed.insert(event.value());
  }
  for (EventId event : execution.replayed_candidate_order) {
    if (!executed.contains(event.value())) {
      execution.unexecuted_candidate.push_back(event);
    }
  }
  return execution;
}

[[nodiscard]] RejectionContext BuildRejectionContext(
    const SourceProgram& source, const LoweredProgram& lowered,
    const HappensBeforeGraph* proof = nullptr,
    std::span<const EventId> candidate = {}, bool reject_negative_tx = false) {
  RejectionContext context{BuildProvenance(source, lowered, proof),
                           std::nullopt};
  if (!candidate.empty()) {
    context.counterexample = ReplayCandidate(lowered, candidate,
                                             reject_negative_tx);
  }
  return context;
}

[[nodiscard]] RejectionContext BuildSimulationContext(
    const SourceProgram& source, const LoweredProgram& lowered,
    const SimulationError& error, bool build_counterexample) {
  if (!build_counterexample) {
    return RejectionContext{BuildProvenance(source, lowered), std::nullopt};
  }
  CounterexampleExecution execution;
  MemoryCheckpoint("sync.failure.before_trace_copy");
  execution.executable_prefix = error.prefix;
  ReportTraceMemory(execution.executable_prefix, "sync.failure.copied_trace");
  execution.stop = error.witness;
  execution.replayed_candidate_order.assign(
      error.prefix.executedEvents().begin(),
      error.prefix.executedEvents().end());
  execution.candidate_order = execution.replayed_candidate_order;
  MemoryCheckpoint("sync.failure.after_order_copies");
  if (const auto* violation = std::get_if<SemanticViolation>(&error.witness);
      violation != nullptr && violation->event.has_value()) {
    execution.candidate_order.push_back(*violation->event);
    execution.replayed_candidate_order.push_back(*violation->event);
    execution.unexecuted_candidate.push_back(*violation->event);
  }
  return RejectionContext{BuildProvenance(source, lowered),
                          std::move(execution)};
}

[[nodiscard]] bool SimulationErrorIsInvalid(SimulationErrorCode code) {
  switch (code) {
    case SimulationErrorCode::kMalformedProgram:
    case SimulationErrorCode::kDuplicateInitialization:
    case SimulationErrorCode::kZeroExpectedArrivals:
    case SimulationErrorCode::kTransactionOverflow:
    case SimulationErrorCode::kPhaseOverflow:
    case SimulationErrorCode::kMissingStateToken:
    case SimulationErrorCode::kInvalidParity:
    case SimulationErrorCode::kWaitDoneBeforeIssue:
    case SimulationErrorCode::kCollectiveMismatch:
    case SimulationErrorCode::kUnexpectedUrgentEvent:
      return true;
    case SimulationErrorCode::kDeadlock:
    case SimulationErrorCode::kNamedArrivalCountMismatch:
    case SimulationErrorCode::kUseBeforeInitialization:
    case SimulationErrorCode::kArrivalUnderflow:
    case SimulationErrorCode::kArrivalOverflow:
    case SimulationErrorCode::kTransactionUnderflow:
    case SimulationErrorCode::kMissingSuccessfulWaitHandoff:
    case SimulationErrorCode::kStateBarrierMismatch:
    case SimulationErrorCode::kStaleStateWait:
    case SimulationErrorCode::kStaleParityWait:
    case SimulationErrorCode::kWaitTargetIncomplete:
      return false;
  }
  return true;
}

[[nodiscard]] bool PhaseBoundaryErrorIsInvalid(PhaseBoundaryErrorCode code) {
  switch (code) {
    case PhaseBoundaryErrorCode::kUnknownPhase:
    case PhaseBoundaryErrorCode::kWrongPhaseKind:
    case PhaseBoundaryErrorCode::kInvalidGraph:
    case PhaseBoundaryErrorCode::kInvalidEffect:
    case PhaseBoundaryErrorCode::kArrivalArithmeticOverflow:
    case PhaseBoundaryErrorCode::kTransactionArithmeticOverflow:
    case PhaseBoundaryErrorCode::kIdealSolverFailure:
      return true;
    case PhaseBoundaryErrorCode::kArrivalBalanceMismatch:
    case PhaseBoundaryErrorCode::kTransactionBalanceMismatch:
    case PhaseBoundaryErrorCode::kNegativeTransactionPrefix:
    case PhaseBoundaryErrorCode::kNegativeArrivalPrefix:
    case PhaseBoundaryErrorCode::kArrivalPrefixOverflow:
    case PhaseBoundaryErrorCode::kPrematurePhaseCompletion:
    case PhaseBoundaryErrorCode::kOpenPhaseArrivalOverflow:
    case PhaseBoundaryErrorCode::kOpenPhaseNotStrictlyPositive:
      return false;
  }
  return true;
}

[[nodiscard]] bool WaitErrorIsInvalid(WaitCheckErrorCode code) {
  switch (code) {
    case WaitCheckErrorCode::kUnknownEvent:
    case WaitCheckErrorCode::kNotWaitIssue:
    case WaitCheckErrorCode::kMissingWaitTarget:
    case WaitCheckErrorCode::kMissingWaitOrigin:
    case WaitCheckErrorCode::kMissingWaitMilestones:
    case WaitCheckErrorCode::kMismatchedWaitDone:
    case WaitCheckErrorCode::kMissingInitialization:
    case WaitCheckErrorCode::kUnknownTargetPhase:
    case WaitCheckErrorCode::kMissingTargetCompletion:
    case WaitCheckErrorCode::kIssueNotOrderedBeforeDone:
    case WaitCheckErrorCode::kMissingPreviousPhase:
    case WaitCheckErrorCode::kMissingPreviousPhaseCompletion:
    case WaitCheckErrorCode::kPhaseNumberOverflow:
    case WaitCheckErrorCode::kMissingNextPhase:
    case WaitCheckErrorCode::kMissingStateDefinition:
    case WaitCheckErrorCode::kInvalidStateDefinition:
    case WaitCheckErrorCode::kMissingStateToken:
    case WaitCheckErrorCode::kHappensBeforeGraphMutationFailed:
      return true;
    case WaitCheckErrorCode::kInitializationNotOrdered:
    case WaitCheckErrorCode::kTargetPhaseIncomplete:
    case WaitCheckErrorCode::kTargetCompletionNotOrderedBeforeDone:
    case WaitCheckErrorCode::kPreviousPhaseNotOrderedBeforeIssue:
    case WaitCheckErrorCode::kIssueNotOrderedBeforeNextPhase:
    case WaitCheckErrorCode::kStateBarrierMismatch:
    case WaitCheckErrorCode::kStatePhaseMismatch:
    case WaitCheckErrorCode::kStateDefinitionNotOrderedBeforeIssue:
    case WaitCheckErrorCode::kParityMismatch:
      return false;
  }
  return true;
}

[[nodiscard]] bool LifecycleErrorIsInvalid(LifecycleCheckErrorCode code) {
  switch (code) {
    case LifecycleCheckErrorCode::kMissingPreviousPhase:
    case LifecycleCheckErrorCode::kInvalidPositiveArrival:
      return true;
    case LifecycleCheckErrorCode::kPreviousPhaseIncomplete:
    case LifecycleCheckErrorCode::kMissingSuccessfulWaitHandoff:
      return false;
  }
  return true;
}

using ExtraEdge = std::pair<EventId, EventId>;

[[nodiscard]] std::vector<EventId> ScheduleWithExtraEdges(
    const HappensBeforeGraph& graph, std::vector<ExtraEdge> requested_edges,
    bool build_schedule) {
  if (!build_schedule) {
    return {};
  }

  std::vector<std::size_t> indegree(graph.eventCount(), 0);
  for (const Event& event : graph.events()) {
    indegree[event.id.value()] = graph.predecessors(event.id).size();
  }
  std::set<ExtraEdge> extra_edges;
  for (const auto& [before, after] : requested_edges) {
    if (!graph.contains(before) || !graph.contains(after)) {
      return {};
    }
    const bool already_present =
        std::find(graph.successors(before).begin(),
                  graph.successors(before).end(),
                  after) != graph.successors(before).end();
    if (!already_present && extra_edges.emplace(before, after).second) {
      ++indegree[after.value()];
    }
  }

  auto greater_id = [](EventId left, EventId right) {
    return left.value() > right.value();
  };
  std::priority_queue<EventId, std::vector<EventId>, decltype(greater_id)>
      ready(greater_id);
  for (const Event& event : graph.events()) {
    if (indegree[event.id.value()] == 0) {
      ready.push(event.id);
    }
  }

  std::vector<EventId> schedule;
  schedule.reserve(graph.eventCount());
  while (!ready.empty()) {
    const EventId event = ready.top();
    ready.pop();
    schedule.push_back(event);
    for (EventId successor : graph.successors(event)) {
      if (--indegree[successor.value()] == 0) {
        ready.push(successor);
      }
    }
    for (const auto& [before, after] : extra_edges) {
      if (event == before && --indegree[after.value()] == 0) {
        ready.push(after);
      }
    }
  }
  if (schedule.size() != graph.eventCount()) {
    return {};
  }
  return schedule;
}

[[nodiscard]] std::vector<EventId> WaitCounterexampleSchedule(
    const WaitCheckError& error, const HappensBeforeGraph& proof,
    ReachabilityOracle& reachability, bool build_schedule) {
  std::vector<ExtraEdge> edges;
  if (const auto* witness =
          std::get_if<WaitReachabilityWitness>(&error.witness)) {
    switch (error.code) {
      case WaitCheckErrorCode::kInitializationNotOrdered:
      case WaitCheckErrorCode::kPreviousPhaseNotOrderedBeforeIssue:
        for (EventId required : witness->required_events) {
          if (!reachability.Reaches(required, witness->issue)) {
            edges.emplace_back(witness->issue, required);
            break;
          }
        }
        break;
      case WaitCheckErrorCode::kIssueNotOrderedBeforeNextPhase:
        for (EventId required : witness->required_events) {
          if (!reachability.Reaches(witness->issue, required)) {
            edges.emplace_back(required, witness->issue);
          }
        }
        break;
      case WaitCheckErrorCode::kIssueNotOrderedBeforeDone:
        if (!witness->required_events.empty()) {
          edges.emplace_back(witness->required_events.front(), witness->issue);
        }
        break;
      default:
        break;
    }
  } else if (const auto* witness =
                 std::get_if<WaitStateWitness>(&error.witness);
             witness != nullptr && witness->definition.has_value() &&
             error.code ==
                 WaitCheckErrorCode::kStateDefinitionNotOrderedBeforeIssue) {
    edges.emplace_back(witness->issue, *witness->definition);
  }
  if (edges.empty()) {
    return {};
  }
  return ScheduleWithExtraEdges(proof, std::move(edges), build_schedule);
}

[[nodiscard]] std::vector<EventId> LifecycleCounterexampleSchedule(
    const LifecycleCheckError& error, const HappensBeforeGraph& proof,
    ReachabilityOracle& reachability, bool build_schedule) {
  std::vector<ExtraEdge> edges;
  if (error.code == LifecycleCheckErrorCode::kMissingSuccessfulWaitHandoff) {
    for (EventId wait_done : error.witness.candidate_waits) {
      if (!reachability.Reaches(wait_done, error.witness.positive_arrival)) {
        edges.emplace_back(error.witness.positive_arrival, wait_done);
      }
    }
  }
  if (edges.empty()) {
    return {};
  }
  return ScheduleWithExtraEdges(proof, std::move(edges), build_schedule);
}

[[nodiscard]] std::optional<TraceIndex> TracePosition(
    const HappensBeforeGraph& proof, const ReferenceTrace& trace,
    EventId event) {
  if (!proof.contains(event)) {
    return std::nullopt;
  }
  const auto snapshot=proof.event(event);
  if (const auto* completion = std::get_if<MBarrierPhaseCompleteEvent>(
          &snapshot.payload)) {
    return trace.completionIndex(
        PhaseKey{completion->barrier, completion->phase});
  }
  return trace.indexOf(event);
}

[[nodiscard]] std::optional<DerivedEdgeTraceOrderError>
CheckDerivedEdgeTraceOrder(const HappensBeforeGraph& proof,
                           const ReferenceTrace& trace) {
  for (const DerivedEdgeJustification& edge : proof.justifications()) {
    const std::optional<TraceIndex> from =
        TracePosition(proof, trace, edge.from);
    const std::optional<TraceIndex> to = TracePosition(proof, trace, edge.to);
    if (!from.has_value() || !to.has_value() || *from >= *to) {
      return DerivedEdgeTraceOrderError{edge.from, edge.to, from, to};
    }
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<CheckResult> CheckInitializationOrder(
    PhaseKey key, const PhaseInfo& phase, const SourceProgram& source,
    const LoweredProgram& program,
    HappensBeforeGraphBuilder& proof, ReachabilityOracle& reachability,
    const CheckerOptions& options,
    std::optional<EventId> initialized_completion) {
  ProfileScope init_profile("phase.initialization_order");
  if (phase.named) return std::nullopt;
  const std::optional<BarrierInitialization> initialization =
      program.barrierInitializations().find(key.barrier);
  if (!initialization.has_value()) {
    PhaseDatabaseError error{PhaseDatabaseErrorCode::kUnknownBarrier,
                             std::nullopt, key};
    return InvalidFrom(error, "phase has no barrier initialization");
  }

  std::vector<EventId> initialization_targets;
  initialization_targets.reserve(phase.effects.size());
  for (EventId effect : phase.effects) {
    const Event& event = proof.happensBeforeGraph().event(effect);
    if (const auto* endpoint = std::get_if<GenericAsyncEffectEvent>(&event.payload);
        endpoint && endpoint->checks_issue_initialization && event.origin) {
      initialization_targets.push_back(
          program.origins().findMilestones(*event.origin)->issue);
    }
  }
  std::sort(initialization_targets.begin(), initialization_targets.end());
  initialization_targets.erase(
      std::unique(initialization_targets.begin(), initialization_targets.end()),
      initialization_targets.end());
  if (!initialized_completion) {
    initialization_targets.insert(initialization_targets.end(),
                                  phase.effects.begin(), phase.effects.end());
  }
  if (initialized_completion) {
    init_profile.pruned = phase.effects.size();
    ProfileScope reuse_profile("phase.initialization_endpoint_reuse");
    // Completion order does not imply issue order: check the forward relation.
    initialization_targets.erase(std::remove_if(
        initialization_targets.begin(), initialization_targets.end(),
        [&](EventId issue) {
          ++reuse_profile.nodes;
          if (!reachability.Reaches(*initialized_completion, issue)) return false;
          ++reuse_profile.pruned;
          ++init_profile.pruned;
          return true;
        }), initialization_targets.end());
  }
  init_profile.nodes = initialization_targets.size();
  ProfileScope direct_profile(key.phase == 0 ? "phase.initialization_phase_zero"
                                            : "phase.initialization_uncovered_issues");
  direct_profile.nodes = initialization_targets.size();
  if (!initialization_targets.empty() &&
      !reachability.ReachesAll(initialization->event, initialization_targets)) {
    for (EventId target : initialization_targets) {
      if (!reachability.Reaches(initialization->event, target)) {
        InitializationOrderWitness witness{
            key, target, initialization->event,
            ScheduleWithExtraEdges(proof.happensBeforeGraph(),
                                   {{target, initialization->event}},
                                   options.build_counterexample_schedule)};
        return RejectedFrom(
            witness,
            BuildRejectionContext(source, program, &proof.happensBeforeGraph(),
                                  witness.counterexample_schedule));
      }
    }
  }

  return std::nullopt;
}

[[nodiscard]] std::optional<CheckResult> CheckPhaseStart(
    PhaseKey key, const PhaseInfo& phase, const SourceProgram& source,
    const LoweredProgram& program, const PhaseDatabase& phases,
    const std::map<BarrierLifetimeId, std::uint32_t>& initialized_phases,
    HappensBeforeGraphBuilder& proof, ReachabilityOracle& reachability,
    const CheckerOptions& options) {
  ProfileScope start_profile("phase.start");
  if (key.phase == 0) {
    return CheckInitializationOrder(key, phase, source, program, proof,
                                    reachability, options, std::nullopt);
  }

  const PhaseKey previous{key.barrier, key.phase - 1};
  const PhaseInfo* previous_info = phases.find(previous);
  if (previous_info == nullptr || !previous_info->completed) {
    PhaseDatabaseError error{
        PhaseDatabaseErrorCode::kEffectOutsideReferencePhase, std::nullopt,
        previous};
    return InvalidFrom(error, "a nonzero phase has no completed predecessor");
  }
  const std::optional<EventId> previous_completion =
      proof.happensBeforeGraph().phaseCompletion(previous);
  if (!previous_completion.has_value()) {
    PhaseDatabaseError error{PhaseDatabaseErrorCode::kMissingEffectAssignment,
                             std::nullopt, previous};
    return InvalidFrom(error,
                       "the previous phase has no phase completion node");
  }

  ProfileScope separation_profile("phase.separation_queries");
  const bool ordered_by_completion =
      reachability.ReachesAll(*previous_completion, phase.effects);
  const auto ordered = ordered_by_completion
                           ? std::vector<std::uint8_t>(phase.effects.size(), 1)
                           : reachability.AllReachMany(previous_info->effects,
                                                       phase.effects);
  for (std::size_t i = 0; i < phase.effects.size(); ++i) {
    const EventId effect = phase.effects[i];
    std::vector<EventId> unordered;
    if (!ordered[i]) {
      for (EventId previous_effect : previous_info->effects) {
        if (!reachability.Reaches(previous_effect, effect)) {
          unordered.push_back(previous_effect);
        }
      }
    }
    if (!unordered.empty()) {
      PreviousPhaseSeparationWitness witness{
          key, effect, previous, unordered,
          ScheduleWithExtraEdges(proof.happensBeforeGraph(),
                                 {{effect, unordered.front()}},
                                 options.build_counterexample_schedule)};
      return RejectedFrom(
          witness,
          BuildRejectionContext(source, program, &proof.happensBeforeGraph(),
                                  witness.counterexample_schedule));
    }
  }

  separation_profile.Stop();
  // Only completed, certified phases of this lifetime establish the induction.
  const auto initialized = initialized_phases.find(key.barrier);
  const bool effects_initialized =
      initialized != initialized_phases.end() &&
      initialized->second == previous.phase && !previous_info->effects.empty();
  if (auto failure = CheckInitializationOrder(
          key, phase, source, program, proof, reachability, options,
          effects_initialized ? previous_completion : std::nullopt)) {
    return failure;
  }

  ProfileScope previous_edges_profile("phase.previous_completion_edges");
  std::vector<std::pair<EventId, EventId>> previous_edges;
  previous_edges.reserve(phase.effects.size());
  for (auto effect : phase.effects) previous_edges.emplace_back(*previous_completion, effect);
  {
    auto added =
        proof.AddCertifiedEdges(previous_edges,
                               HappensBeforeEdgeReason::kCertifiedPreviousPhase,
                               CertificateRef{previous});
    if (!added) {
      if (added.error().code == HappensBeforeGraphMutationErrorCode::kCycle) {
        return RejectedFrom(added.error(),
                            BuildRejectionContext(source, program,
                                                  &proof.happensBeforeGraph()));
      }
      return InvalidFrom(added.error(),
                         "failed to add a certified previous-phase edge");
    }
  }
  reachability.NotifyGraphMutation();
  return std::nullopt;
}

[[nodiscard]] std::optional<CheckResult> AddCompletedPhase(
    const PhaseInfo& phase, const SourceProgram& source,
    const LoweredProgram& program, HappensBeforeGraphBuilder& proof,
    ReachabilityOracle& reachability, bool memory_steps = false) {
  const PhaseKey key = phase.key();
  auto completion = ProfileCall("phase.add_completion_node", [&] { return proof.AddPhaseCompletion(key); });
  if (!completion) {
    return InvalidFrom(completion.error(),
                       "failed to create a phase-completion event");
  }
  if (memory_steps) MemoryCheckpoint("sync.first_phase.completion_node.ready");
  reachability.NotifyGraphMutation();

  ProfileScope effect_edges_profile("phase.effect_to_completion_edges");
  std::vector<std::pair<EventId, EventId>> effect_edges;
  effect_edges.reserve(phase.effects.size());
  for (auto effect : phase.effects) effect_edges.emplace_back(effect, completion.value());
  {
    auto added = proof.AddCertifiedEdges(
        effect_edges,
        HappensBeforeEdgeReason::kCertifiedPhaseCompletion,
        CertificateRef{key});
    if (!added) {
      if (added.error().code == HappensBeforeGraphMutationErrorCode::kCycle) {
        return RejectedFrom(added.error(),
                            BuildRejectionContext(source, program,
                                                  &proof.happensBeforeGraph()));
      }
      return InvalidFrom(added.error(),
                         "failed to add a phase-completion edge");
    }
  }

  effect_edges_profile.Stop();
  ProfileScope wait_edges_profile("phase.completion_to_wait_edges");
  std::vector<std::pair<EventId, EventId>> wait_edges;
  wait_edges.reserve(phase.waits_done_for_phase.size());
  for (auto wait : phase.waits_done_for_phase) wait_edges.emplace_back(completion.value(), wait);
  {
    auto added = proof.AddCertifiedEdges(wait_edges,
                                        HappensBeforeEdgeReason::kWaitRelease,
                                        CertificateRef{key});
    if (!added) {
      if (added.error().code == HappensBeforeGraphMutationErrorCode::kCycle) {
        return RejectedFrom(added.error(),
                            BuildRejectionContext(source, program,
                                                  &proof.happensBeforeGraph()));
      }
      return InvalidFrom(added.error(), "failed to add a wait-release edge");
    }
  }
  reachability.NotifyGraphMutation();
  return std::nullopt;
}

}  // namespace

CheckResult WellSyncChecker::Run(const SourceProgram& source,
                                 const CheckerOptions& options) const {
  return RunImpl(source,options,nullptr);
}
CheckResult WellSyncChecker::RunPrepared(const SourceProgram& source,
    const LoweredProgram& prepared,const CheckerOptions& options) const {
  return RunImpl(source,options,&prepared);
}
namespace {

// The execution-order proof graph orders a lane's return from shfl/elect/vote/
// match/redux after every lane's issue. The wait, phase and lifecycle checks
// need that: it is what places the waits of non-elected lanes before the
// elected lane's arrive or commit. The rendezvous publishes no memory effects
// (#251), so the certificate handed to memory analysis is rebuilt on the
// memory-order lowering with the same phase completions and derived edges.
// Both lowerings emit identical events; only the collectives' completion ->
// done edges differ.
std::optional<CheckResult> UseMemoryOrderGraph(const SourceProgram& source,
    PassCertificate& pass, const CheckerOptions& options) {
  bool has_data_collective = false;
  pass.happensBeforeGraph.events().forEachAffineColumn(
      [&](const Event& event, const auto&, std::size_t, std::size_t, std::size_t) {
        if (const auto* complete = std::get_if<CollectiveCompleteEvent>(&event.payload))
          has_data_collective |= complete->kind == CollectiveKind::kWarpData;
      });
  if (!has_data_collective) return std::nullopt;

  ProfileScope profile("sync.memory_order_graph");
  const auto invalid = [](std::string message) {
    return CheckResult::Invalid(InvalidInput{std::vector<Diagnostic>{},
        "memory-order proof graph construction failed: " + std::move(message)});
  };
  // Phase-completion events are appended by the builder; replaying them in ID
  // order reproduces the IDs the derived edges refer to.
  std::vector<std::pair<EventId, PhaseKey>> completions;
  completions.reserve(pass.happensBeforeGraph.phaseCompletions().size());
  for (const auto& [phase, id] : pass.happensBeforeGraph.phaseCompletions())
    completions.emplace_back(id, phase);
  std::sort(completions.begin(), completions.end(),
            [](const auto& left, const auto& right) { return left.first < right.first; });
  const std::size_t event_count = pass.happensBeforeGraph.eventCount();
  // The derived edges live in pass.justifications; release the execution-order
  // graph before building its replacement to keep the peak footprint flat.
  pass.happensBeforeGraph = HappensBeforeGraph{};

  auto lowered = ProfileCall("sync.memory_order_lower", [&] {
    return Lowerer{}.Run(source, true, CollectiveOrdering::kMemory); });
  if (!lowered) return invalid(lowered.error().message);
  std::optional<HappensBeforeGraphBuilder> builder;
  try {
    builder.emplace(lowered.value(),
                    options.reachability.use_trace_order ? &pass.reference : nullptr);
  } catch (const std::exception& error) {
    return invalid(error.what());
  }
  for (const auto& [id, phase] : completions) {
    const auto added = builder->AddPhaseCompletion(phase);
    if (!added || added.value() != id) return invalid("phase completion mismatch");
  }
  std::vector<std::pair<EventId, EventId>> batch;
  std::optional<DerivedEdgeJustification> shared;
  const auto flush = [&]() -> bool {
    if (batch.empty()) return true;
    const auto added = builder->AddCertifiedEdges(batch, shared->reason, shared->certificate);
    batch.clear();
    return static_cast<bool>(added);
  };
  for (std::size_t index = 0; index < pass.justifications.size(); ++index) {
    const DerivedEdgeJustification edge = pass.justifications[index];
    if (shared && (shared->reason != edge.reason || !(shared->certificate == edge.certificate)) &&
        !flush()) return invalid("derived edge rejected");
    shared = edge;
    batch.emplace_back(edge.from, edge.to);
  }
  if (!flush()) return invalid("derived edge rejected");
  const HappensBeforeGraph& frozen = builder->Freeze();
  if (frozen.eventCount() != event_count) return invalid("event count mismatch");
  if (!frozen.isAcyclic()) return invalid("cyclic graph");
  pass.justifications = frozen.justifications();
  pass.happensBeforeGraph = std::move(*builder).TakeFrozen();
  ReportRecordGraphSize("hb.memory_order", pass.happensBeforeGraph.events(), pass.happensBeforeGraph.edges());
  return std::nullopt;
}

}  // namespace

CheckResult WellSyncChecker::RunImpl(const SourceProgram& source,
    const CheckerOptions& options,const LoweredProgram* prepared) const {
  ScopedWellSyncDeadline deadline(options.timeout_seconds);
  ProfileScope sync_profile("sync.total");
  // The reachability indexes of the execution-order pass are released on return.
  CheckResult result = RunExecutionOrder(source, options, prepared);
  if (PassCertificate* pass = result.certificate()) {
    if (auto failure = UseMemoryOrderGraph(source, *pass, options)) return std::move(*failure);
  }
  return result;
}

CheckResult WellSyncChecker::RunExecutionOrder(const SourceProgram& source,
    const CheckerOptions& options,const LoweredProgram* prepared) const {
  std::optional<LoweredProgram> owned;
  if(!prepared) {
  SourceValidator validator;
  std::vector<Diagnostic> diagnostics = ProfileCall("sync.validate_source", [&] { return validator.Validate(source); });
  if (!diagnostics.empty()) {
    const bool has_invalid = std::any_of(
        diagnostics.begin(), diagnostics.end(),
        [](const Diagnostic& diagnostic) {
          return diagnostic.classification == DiagnosticClass::kInvalidInput;
        });
    return has_invalid ? InvalidFromDiagnostics(std::move(diagnostics))
                       : UnsupportedFromDiagnostics(std::move(diagnostics));
  }

  Lowerer lowerer;
  auto lowered_result = ProfileCall("sync.lower", [&] {
    return lowerer.Run(source, true, CollectiveOrdering::kExecution); });
  if (!lowered_result) {
    const Diagnostic error = lowered_result.error();
    if (error.classification == DiagnosticClass::kUnsupported) {
      return UnsupportedFromDiagnostics({error});
    }
    return InvalidFrom(error, "source lowering failed");
  }
  owned.emplace(std::move(lowered_result).value());
  prepared=&*owned;
  }
  const LoweredProgram& lowered=*prepared;

  ReferenceSimulator simulator;
  auto trace_result =
      ProfileCall("sync.reference_simulation", [&] { return simulator.Run(lowered, SimulationOptions{
          .reject_negative_tx = options.reject_negative_tx}); });
  if (!trace_result) {
    MemoryCheckpoint("sync.reference_result", "failure");
    SimulationError error = std::move(trace_result).error();
    ReportTraceMemory(error.prefix, "sync.reference_error_prefix");
    if (SimulationErrorIsInvalid(error.code)) {
      return InvalidFrom(std::move(error), "reference simulation failed");
    }
    RejectionContext context = BuildSimulationContext(
        source, lowered, error, options.build_counterexample_schedule);
    return std::visit(
        [context = std::move(context)](auto witness) mutable {
          return RejectedFrom(std::move(witness), std::move(context));
        },
        std::move(error.witness));
  }
  ReferenceTrace trace = std::move(trace_result).value();
  MemoryCheckpoint("sync.reference_result", "success");
  ReportTraceMemory(trace, "sync.reference_trace");
  ReportIrSize("trace.ready", {{"executed_events",trace.executedEvents().size()},
      {"executed_event_runs",trace.executedEvents().runs().size()},
      {"normal_events",trace.normalEvents().size()},
      {"normal_event_runs",trace.normalEvents().runs().size()},
      {"entries",trace.entries().size()}, {"entry_runs",trace.entries().runs().size()}});

  MemoryCheckpoint("sync.phase_database.begin");
  auto phases_result = ProfileCall("sync.phase_database", [&] { return PhaseDatabase::Build(lowered, trace); });
  if (!phases_result) {
    return InvalidFrom(std::move(phases_result).error(),
                       "reference phase extraction failed");
  }
  PhaseDatabase phases = std::move(phases_result).value();
  MemoryCheckpoint("sync.phase_database.ready");

  std::optional<HappensBeforeGraphBuilder> graph_builder_storage;
  try {
    ProfileScope graph_profile("sync.build_hb_graph");
    MemoryCheckpoint("sync.build_hb_graph.begin");
    graph_builder_storage.emplace(lowered, options.reachability.use_trace_order ? &trace : nullptr);
    MemoryCheckpoint("sync.build_hb_graph.ready");
  } catch (const std::invalid_argument& error) {
    return CheckResult::Invalid(
        InvalidInput{std::vector<Diagnostic>{},
                     std::string("happens-before graph construction failed: ") +
                         error.what()});
  }
  HappensBeforeGraphBuilder& proof = *graph_builder_storage;
  ReportRecordGraphSize("hb.initial", proof.happensBeforeGraph().events(), proof.happensBeforeGraph().edges());
  ReachabilityOracle reachability(proof.happensBeforeGraph(), options.reachability);
  ParallelReachability parallel(proof.happensBeforeGraph(), options.reachability,
                                options.reachability.worker_threads);
  PhaseBoundaryChecker boundary_checker;
  std::map<BarrierLifetimeId, std::uint32_t> initialized_phases;

  std::vector<PhaseCertificate> completed_certificates;
  completed_certificates.reserve(phases.completedInTraceOrder().size());
  for (PhaseKey key : phases.completedInTraceOrder()) {
    ProfileScope phase_profile("sync.completed_phase");
    const PhaseInfo* phase = phases.find(key);
    if (phase) phase_profile.Phase(key.barrier.value(), key.phase, phase->effects.size());
    if (phase == nullptr || !phase->completed) {
      PhaseDatabaseError error{
          PhaseDatabaseErrorCode::kEffectOutsideReferencePhase, std::nullopt,
          key};
      return InvalidFrom(error, "completed phase metadata is inconsistent");
    }
    const bool memory_steps = MemoryProfileEnabled() && completed_certificates.empty();
    if (memory_steps) MemoryCheckpoint("sync.first_phase.initialization.begin");
    if (auto failure = CheckPhaseStart(key, *phase, source, lowered, phases,
                                       initialized_phases,
                                       proof, reachability, options);
        failure.has_value()) {
      return std::move(*failure);
    }

    if (memory_steps) MemoryCheckpoint("sync.first_phase.initialization.ready");

    auto certificate = boundary_checker.CheckCompletedPhase(
        proof.happensBeforeGraph(), phases, key,
        options.build_counterexample_schedule, options.reject_negative_tx,
        options.retain_successful_ideals);
    if (memory_steps) MemoryCheckpoint("sync.first_phase.boundary.returned",
                                      certificate ? "success" : "failure");
    if (!certificate) {
      PhaseBoundaryError error = std::move(certificate).error();
      if (!options.reject_negative_tx && error.code ==
              PhaseBoundaryErrorCode::kNegativeTransactionPrefix) {
        return CheckResult::Unsupported(UnsupportedInput{
            UnsupportedReason::kSignedTransactionCertificateUnsupported, {},
            "signed transaction certificate is inconclusive for barrier " +
                std::to_string(key.barrier.value()) + " phase " +
                std::to_string(key.phase) +
                ": a required nonnegative transaction bound failed; "
                "negative counts alone are not a kernel error"});
      }
      if (PhaseBoundaryErrorIsInvalid(error.code)) {
        return InvalidFrom(std::move(error),
                           "completed-phase certification failed");
      }
      std::span<const EventId> candidate;
      if (auto* ideal = std::get_if<PhaseIdealWitness>(&error.witness)) {
        if (options.build_counterexample_schedule) {
          candidate = ideal->topological_schedule;
        } else {
          ideal->topological_schedule.clear();
        }
      }
      RejectionContext context = BuildRejectionContext(
          source, lowered, &proof.happensBeforeGraph(), candidate,
          options.reject_negative_tx);
      return RejectedFrom(std::move(error), std::move(context));
    }
    completed_certificates.push_back(std::move(certificate).value());

    if (auto failure =
            AddCompletedPhase(*phase, source, lowered, proof, reachability, memory_steps);
        failure.has_value()) {
      return std::move(*failure);
    }
    if (!phase->named) initialized_phases[key.barrier] = key.phase;
  }

  std::vector<OpenPhaseCertificate> open_certificates;
  open_certificates.reserve(phases.openPhases().size());
  for (PhaseKey key : phases.openPhases()) {
    ProfileScope phase_profile("sync.open_phase");
    const PhaseInfo* phase = phases.find(key);
    if (phase) phase_profile.Phase(key.barrier.value(), key.phase, phase->effects.size());
    if (phase == nullptr || phase->completed) {
      PhaseDatabaseError error{
          PhaseDatabaseErrorCode::kEffectOutsideReferencePhase, std::nullopt,
          key};
      return InvalidFrom(error, "open phase metadata is inconsistent");
    }
    if (auto failure = CheckPhaseStart(key, *phase, source, lowered, phases,
                                       initialized_phases,
                                       proof, reachability, options);
        failure.has_value()) {
      return std::move(*failure);
    }

    auto certificate = boundary_checker.CheckOpenPhase(
        proof.happensBeforeGraph(), phases, key,
        options.build_counterexample_schedule, options.reject_negative_tx,
        options.retain_successful_ideals);
    if (!certificate) {
      PhaseBoundaryError error = std::move(certificate).error();
      if (!options.reject_negative_tx && error.code ==
              PhaseBoundaryErrorCode::kNegativeTransactionPrefix) {
        return CheckResult::Unsupported(UnsupportedInput{
            UnsupportedReason::kSignedTransactionCertificateUnsupported, {},
            "signed transaction certificate is inconclusive for barrier " +
                std::to_string(key.barrier.value()) + " phase " +
                std::to_string(key.phase) +
                ": a required nonnegative transaction bound failed; "
                "negative counts alone are not a kernel error"});
      }
      if (PhaseBoundaryErrorIsInvalid(error.code)) {
        return InvalidFrom(std::move(error), "open-phase certification failed");
      }
      std::span<const EventId> candidate;
      if (auto* ideal = std::get_if<PhaseIdealWitness>(&error.witness)) {
        if (options.build_counterexample_schedule) {
          candidate = ideal->topological_schedule;
        } else {
          ideal->topological_schedule.clear();
        }
      }
      RejectionContext context = BuildRejectionContext(
          source, lowered, &proof.happensBeforeGraph(), candidate,
          options.reject_negative_tx);
      return RejectedFrom(std::move(error), std::move(context));
    }
    open_certificates.push_back(std::move(certificate).value());
  }

  WaitChecker wait_checker;
  auto wait_result =
      ProfileCall("sync.waits", [&] {
        return wait_checker.CheckAll(
            lowered, trace, phases, proof, reachability, &parallel,
            WaitReachabilityBatchOptions{
                .certified_phase_ordering = options.use_batched_wait_reachability});
      });
  if (!wait_result) {
    WaitCheckError error = std::move(wait_result).error();
    if (WaitErrorIsInvalid(error.code)) {
      return InvalidFrom(std::move(error), "wait certification failed");
    }
    const std::vector<EventId> candidate = WaitCounterexampleSchedule(
        error, proof.happensBeforeGraph(), reachability,
        options.build_counterexample_schedule);
    return RejectedFrom(
        std::move(error),
        BuildRejectionContext(source, lowered, &proof.happensBeforeGraph(),
                              candidate, options.reject_negative_tx));
  }
  std::vector<WaitCertificate> wait_certificates =
      std::move(wait_result).value();

  ReportRecordGraphSize("hb.after_waits", proof.happensBeforeGraph().events(), proof.happensBeforeGraph().edges());
  LifecycleChecker lifecycle_checker;
  auto lifecycle_result =
      ProfileCall("sync.barrier_handoffs", [&] { return lifecycle_checker.Check(phases, proof.happensBeforeGraph(), reachability); });
  if (!lifecycle_result) {
    LifecycleCheckError error = std::move(lifecycle_result).error();
    if (LifecycleErrorIsInvalid(error.code)) {
      return InvalidFrom(std::move(error), "lifecycle certification failed");
    }
    const std::vector<EventId> candidate = LifecycleCounterexampleSchedule(
        error, proof.happensBeforeGraph(), reachability,
        options.build_counterexample_schedule);
    return RejectedFrom(
        std::move(error),
        BuildRejectionContext(source, lowered, &proof.happensBeforeGraph(),
                              candidate, options.reject_negative_tx));
  }
  LifecycleCertificate lifecycle = std::move(lifecycle_result).value();

  if (const auto inconsistency =
          ProfileCall("sync.validate_derived_trace_order", [&] { return CheckDerivedEdgeTraceOrder(proof.happensBeforeGraph(), trace); });
      inconsistency.has_value()) {
    return InvalidFrom(
        *inconsistency,
        "a derived happens-before edge contradicts the reference trace");
  }

  ProfileScope freeze_profile("sync.freeze");
  const HappensBeforeGraph& frozen = proof.Freeze();
  if (!frozen.isAcyclic()) {
    return RejectedFrom(
        FinalHappensBeforeCycleWitness{},
        BuildRejectionContext(source, lowered, &proof.happensBeforeGraph()));
  }
  bool has_frozen_reachability = false;
  if (options.use_bitset_reachability_after_freeze) {
    has_frozen_reachability = reachability.Freeze();
    if (!has_frozen_reachability) {
      return RejectedFrom(
          FinalHappensBeforeCycleWitness{},
          BuildRejectionContext(source, lowered, &proof.happensBeforeGraph()));
    }
  }

  freeze_profile.Stop();
  ProfileScope pass_profile("sync.transfer_certificate");
  auto justifications = frozen.justifications();
  PassCertificate pass{
      std::move(trace),
      std::move(completed_certificates),
      std::move(open_certificates),
      std::move(wait_certificates),
      std::move(lifecycle),
      std::move(proof).TakeFrozen(),
      std::move(justifications),
      has_frozen_reachability};
  ReportRecordGraphSize("hb.certified", pass.happensBeforeGraph.events(), pass.happensBeforeGraph.edges());
  return CheckResult::Passed(std::move(pass));
}

}  // namespace sentryward::algorithm
