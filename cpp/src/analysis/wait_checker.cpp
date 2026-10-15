#include "sentryward/algorithm/analysis/wait_checker.h"

#include <algorithm>
#include <limits>

#include "sentryward/algorithm/analysis/parallel_reachability.h"
#include "sentryward/algorithm/diagnostics/performance_profile.h"
#include <optional>
#include <utility>
#include <variant>

namespace sentryward::algorithm {
namespace {

template <typename Witness>
[[nodiscard]] WaitCheckError Error(WaitCheckErrorCode code,
                                   Witness witness) {
  return WaitCheckError{code, WaitCheckWitness{std::move(witness)}};
}

[[nodiscard]] WaitEventWitness EventWitness(
    EventId issue, std::optional<EventId> related = std::nullopt,
    std::optional<PhaseKey> phase = std::nullopt) {
  return WaitEventWitness{issue, related, phase};
}

[[nodiscard]] std::optional<BarrierLifetimeId> WaitBarrier(
    const Event& event) {
  if (const auto* named = std::get_if<NamedBarrierArriveEvent>(&event.payload);
      named && named->blocking) return named->barrier;
  if (const auto* wait = std::get_if<WaitStateIssueEvent>(&event.payload)) {
    return wait->barrier;
  }
  if (const auto* wait = std::get_if<WaitParityIssueEvent>(&event.payload)) {
    return wait->barrier;
  }
  return std::nullopt;
}

[[nodiscard]] bool IsMatchingDone(const Event& issue, const Event& done) {
  if (issue.origin != done.origin) {
    return false;
  }
  if (const auto* named = std::get_if<NamedBarrierArriveEvent>(&issue.payload)) {
    const auto* result = std::get_if<NamedBarrierReturnEvent>(&done.payload);
    return named->blocking && result && named->barrier == result->barrier;
  }
  if (const auto* state = std::get_if<WaitStateIssueEvent>(&issue.payload)) {
    const auto* result = std::get_if<WaitStateDoneEvent>(&done.payload);
    return result != nullptr && result->barrier == state->barrier &&
           result->state == state->state;
  }
  if (const auto* parity =
          std::get_if<WaitParityIssueEvent>(&issue.payload)) {
    const auto* result = std::get_if<WaitParityDoneEvent>(&done.payload);
    return result != nullptr && result->barrier == parity->barrier &&
           result->parity == parity->parity;
  }
  return false;
}

[[nodiscard]] WaitCheckError ReachabilityError(
    WaitCheckErrorCode code, EventId issue, PhaseKey phase,
    std::vector<EventId> required_events) {
  return Error(code, WaitReachabilityWitness{
                         issue, phase, std::move(required_events)});
}

// These relations normally have an edge installed by lowering or phase
// certification. Inspect the smaller adjacency row without allocating search
// state; hand-built graphs with only an indirect path still use the oracle.
[[nodiscard]] bool ReachesWithDirectEdge(
    const HappensBeforeGraph& graph, ReachabilityOracle& reachability,
    EventId from, EventId to) {
  if (!graph.contains(from) || !graph.contains(to)) return false;
  if (from == to) return true;
  const auto successors = graph.successors(from);
  const auto predecessors = graph.predecessors(to);
  const bool direct = successors.size() <= predecessors.size()
      ? std::find(successors.begin(), successors.end(), to) != successors.end()
      : std::find(predecessors.begin(), predecessors.end(), from) != predecessors.end();
  if (direct) return true;
  ProfileScope fallback("wait.direct_edge_fallback");
  return reachability.Reaches(from, to);
}

// A null writer means speculative validation against an immutable batch graph.
// Successful certificates can be committed after the batch; failures must be
// retried with Check after earlier certificates have been committed.
Result<WaitCertificate, WaitCheckError> CheckImpl(
    EventId issue_id, const LoweredProgram& program,
    const ReferenceTrace& trace, const PhaseDatabase& phases,
    const HappensBeforeGraph& graph, ReachabilityOracle& reachability,
    HappensBeforeGraphBuilder* writer,
    const WaitReachabilityAnswer* batch = nullptr) {
  const EventDag& base = program.graph();
  if (!base.contains(issue_id) || !graph.contains(issue_id)) {
    return Result<WaitCertificate, WaitCheckError>::Failure(
        Error(WaitCheckErrorCode::kUnknownEvent, EventWitness(issue_id)));
  }

  const Event& issue = base.event(issue_id);
  const std::optional<BarrierLifetimeId> barrier = WaitBarrier(issue);
  if (!barrier.has_value()) {
    return Result<WaitCertificate, WaitCheckError>::Failure(
        Error(WaitCheckErrorCode::kNotWaitIssue, EventWitness(issue_id)));
  }

  const std::optional<PhaseKey> target = trace.waitTarget(issue_id);
  const auto initial_barrier = trace.initialWaitBarrier(issue_id);
  if ((!target.has_value() && initial_barrier != barrier) ||
      (target.has_value() && target->barrier != *barrier) ||
      (target.has_value() && initial_barrier.has_value())) {
    return Result<WaitCertificate, WaitCheckError>::Failure(
        Error(WaitCheckErrorCode::kMissingWaitTarget,
              EventWitness(issue_id, std::nullopt, target)));
  }
  if (!issue.origin.has_value()) {
    return Result<WaitCertificate, WaitCheckError>::Failure(
        Error(WaitCheckErrorCode::kMissingWaitOrigin,
              EventWitness(issue_id, std::nullopt, target)));
  }

  const std::optional<Milestones> milestones =
      program.origins().milestones(*issue.origin);
  if (!milestones.has_value() || milestones->issue != issue_id) {
    return Result<WaitCertificate, WaitCheckError>::Failure(
        Error(WaitCheckErrorCode::kMissingWaitMilestones,
              EventWitness(issue_id, std::nullopt, target)));
  }
  const EventId done_id = milestones->done;
  if (!base.contains(done_id) || !graph.contains(done_id) ||
      !IsMatchingDone(issue, base.event(done_id))) {
    return Result<WaitCertificate, WaitCheckError>::Failure(
        Error(WaitCheckErrorCode::kMismatchedWaitDone,
              EventWitness(issue_id, done_id, target)));
  }

  if (std::holds_alternative<NamedBarrierArriveEvent>(issue.payload)) {
    const auto* phase = target ? phases.find(*target) : nullptr;
    const auto completion = target ? graph.phaseCompletion(*target) : std::nullopt;
    if (!phase || !phase->completed || !completion ||
        trace.effectPhase(issue_id) != target->phase)
      return Result<WaitCertificate, WaitCheckError>::Failure(
          Error(WaitCheckErrorCode::kTargetPhaseIncomplete, EventWitness(issue_id, done_id, target)));
    if (!ReachesWithDirectEdge(graph, reachability, *completion, done_id) ||
        !ReachesWithDirectEdge(graph, reachability, issue_id, done_id))
      return Result<WaitCertificate, WaitCheckError>::Failure(
          Error(WaitCheckErrorCode::kTargetCompletionNotOrderedBeforeDone, EventWitness(issue_id, done_id, target)));
    return Result<WaitCertificate, WaitCheckError>::Success(WaitCertificate{
        issue_id, done_id, target, std::nullopt, completion, std::nullopt, std::nullopt, NamedWaitCertificate{}});
  }
  const std::optional<BarrierInitialization> initialization =
      program.barrierInitializations().find(*barrier);
  if (!initialization.has_value() ||
      !graph.contains(initialization->event)) {
    return Result<WaitCertificate, WaitCheckError>::Failure(
        Error(WaitCheckErrorCode::kMissingInitialization,
              EventWitness(issue_id, std::nullopt, target)));
  }
  const bool initialized = ProfileCall("wait.initialization", [&] {
    if (batch && batch->initialization.has_value()) {
      return *batch->initialization;
    }
    ProfileScope fallback("wait.initialization_fallback");
    return reachability.Reaches(initialization->event, issue_id);
  });
  if (!initialized) {
    return Result<WaitCertificate, WaitCheckError>::Failure(
        Error(WaitCheckErrorCode::kInitializationNotOrdered,
              EventWitness(issue_id, initialization->event, target)));
  }

  if (initial_barrier.has_value()) {
    const auto* parity_wait = std::get_if<WaitParityIssueEvent>(&issue.payload);
    if (!parity_wait || parity_wait->parity != 1 ||
        trace.initialWaitBarrier(done_id) != initial_barrier) {
      return Result<WaitCertificate, WaitCheckError>::Failure(
          Error(WaitCheckErrorCode::kMismatchedWaitDone,
                EventWitness(issue_id, done_id)));
    }
    if (!ProfileCall("wait.issue_done", [&] {
          return ReachesWithDirectEdge(graph, reachability, issue_id, done_id);
        })) {
      return Result<WaitCertificate, WaitCheckError>::Failure(
          Error(WaitCheckErrorCode::kIssueNotOrderedBeforeDone,
                EventWitness(issue_id, done_id)));
    }
    const PhaseKey first{*barrier, 0};
    const PhaseInfo* first_info = phases.find(first);
    if (first_info == nullptr) {
      return Result<WaitCertificate, WaitCheckError>::Failure(
          Error(WaitCheckErrorCode::kMissingNextPhase,
                EventWitness(issue_id, std::nullopt, first)));
    }
    std::optional<UpperFreshnessCertificate> upper;
    if (first_info->completed) {
      const auto reachable_effect = ProfileCall("wait.upper", [&] {
        if (batch && batch->upper_indexed) return batch->upper_witness;
        ProfileScope fallback("wait.upper_fallback");
        return reachability.FindDescendant(issue_id, first_info->effects);
      });
      if (!reachable_effect) {
        return Result<WaitCertificate, WaitCheckError>::Failure(
            ReachabilityError(WaitCheckErrorCode::kIssueNotOrderedBeforeNextPhase,
                              issue_id, first, first_info->effects));
      }
      upper = UpperFreshnessCertificate{first, *reachable_effect};
    }
    return Result<WaitCertificate, WaitCheckError>::Success(WaitCertificate{
        issue_id, done_id, std::nullopt, initialization->event, std::nullopt,
        std::nullopt, std::move(upper), ParityWaitCertificate{1}});
  }

  const PhaseInfo* target_info = phases.find(*target);
  if (target_info == nullptr) {
    return Result<WaitCertificate, WaitCheckError>::Failure(
        Error(WaitCheckErrorCode::kUnknownTargetPhase,
              EventWitness(issue_id, std::nullopt, target)));
  }
  if (!target_info->completed) {
    return Result<WaitCertificate, WaitCheckError>::Failure(
        Error(WaitCheckErrorCode::kTargetPhaseIncomplete,
              EventWitness(issue_id, std::nullopt, target)));
  }

  const std::optional<EventId> target_completion =
      graph.phaseCompletion(*target);
  if (!target_completion.has_value()) {
    return Result<WaitCertificate, WaitCheckError>::Failure(
        Error(WaitCheckErrorCode::kMissingTargetCompletion,
              EventWitness(issue_id, std::nullopt, target)));
  }
  if (!ProfileCall("wait.issue_done", [&] {
        return ReachesWithDirectEdge(graph, reachability, issue_id, done_id);
      })) {
    return Result<WaitCertificate, WaitCheckError>::Failure(
        ReachabilityError(WaitCheckErrorCode::kIssueNotOrderedBeforeDone,
                          issue_id, *target, {done_id}));
  }
  if (!ProfileCall("wait.completion_done", [&] {
        return ReachesWithDirectEdge(graph, reachability, *target_completion, done_id);
      })) {
    return Result<WaitCertificate, WaitCheckError>::Failure(
        ReachabilityError(
            WaitCheckErrorCode::kTargetCompletionNotOrderedBeforeDone,
            issue_id, *target, {*target_completion}));
  }

  std::optional<LowerFreshnessCertificate> lower;
  if (target->phase > 0) {
    const PhaseKey previous{target->barrier, target->phase - 1};
    const PhaseInfo* previous_info = phases.find(previous);
    if (previous_info == nullptr || !previous_info->completed) {
      return Result<WaitCertificate, WaitCheckError>::Failure(
          Error(WaitCheckErrorCode::kMissingPreviousPhase,
                EventWitness(issue_id, std::nullopt, previous)));
    }
    const bool lower_ordered = ProfileCall("wait.lower", [&] {
      if (batch && batch->lower.has_value()) return *batch->lower;
      ProfileScope fallback("wait.lower_fallback");
      return reachability.AllReach(previous_info->effects, issue_id);
    });
    if (!lower_ordered) {
      return Result<WaitCertificate, WaitCheckError>::Failure(
          ReachabilityError(
              WaitCheckErrorCode::kPreviousPhaseNotOrderedBeforeIssue,
              issue_id, previous, previous_info->effects));
    }
    const std::optional<EventId> previous_completion =
        graph.phaseCompletion(previous);
    if (!previous_completion.has_value()) {
      return Result<WaitCertificate, WaitCheckError>::Failure(
          Error(WaitCheckErrorCode::kMissingPreviousPhaseCompletion,
                EventWitness(issue_id, std::nullopt, previous)));
    }
    if (writer) {
      const auto added = writer->AddCertifiedEdge(
          *previous_completion, issue_id,
          HappensBeforeEdgeReason::kCertifiedPreviousPhase, CertificateRef{previous});
      if (!added) {
        return Result<WaitCertificate, WaitCheckError>::Failure(
            Error(WaitCheckErrorCode::kHappensBeforeGraphMutationFailed, added.error()));
      }
      reachability.NotifyGraphMutation();
    }
    lower = LowerFreshnessCertificate{
        previous, previous_info->effects, *previous_completion};
  }

  if (target->phase == std::numeric_limits<std::uint32_t>::max()) {
    return Result<WaitCertificate, WaitCheckError>::Failure(
        Error(WaitCheckErrorCode::kPhaseNumberOverflow,
              EventWitness(issue_id, std::nullopt, target)));
  }

  std::optional<UpperFreshnessCertificate> upper;
  const PhaseKey next{target->barrier, target->phase + 1};
  const PhaseInfo* next_info = phases.find(next);
  if (next_info == nullptr) {
    return Result<WaitCertificate, WaitCheckError>::Failure(
        Error(WaitCheckErrorCode::kMissingNextPhase,
              EventWitness(issue_id, std::nullopt, next)));
  }
  if (next_info->completed) {
    const auto reachable_effect = ProfileCall("wait.upper", [&] {
      if (batch && batch->upper_indexed) return batch->upper_witness;
      ProfileScope fallback("wait.upper_fallback");
      return reachability.FindDescendant(issue_id, next_info->effects);
    });
    if (!reachable_effect.has_value()) {
      return Result<WaitCertificate, WaitCheckError>::Failure(
          ReachabilityError(
              WaitCheckErrorCode::kIssueNotOrderedBeforeNextPhase, issue_id,
              next, next_info->effects));
    }
    upper = UpperFreshnessCertificate{next, *reachable_effect};
  }

  WaitKindCertificate kind;
  if (const auto* state_wait =
          std::get_if<WaitStateIssueEvent>(&issue.payload)) {
    const std::optional<EventId> definition =
        program.stateDefinitions().definition(state_wait->state);
    if (!definition.has_value() || !base.contains(*definition)) {
      return Result<WaitCertificate, WaitCheckError>::Failure(
          Error(WaitCheckErrorCode::kMissingStateDefinition,
                WaitStateWitness{issue_id, state_wait->state, *target,
                                 definition, std::nullopt}));
    }
    const auto definition_event=base.event(*definition);
    const auto* update =
        std::get_if<MBarrierUpdateEvent>(&definition_event.payload);
    if (update == nullptr || update->produced_state != state_wait->state) {
      return Result<WaitCertificate, WaitCheckError>::Failure(
          Error(WaitCheckErrorCode::kInvalidStateDefinition,
                WaitStateWitness{issue_id, state_wait->state, *target,
                                 definition, std::nullopt}));
    }
    if (update->barrier != state_wait->barrier) {
      return Result<WaitCertificate, WaitCheckError>::Failure(
          Error(WaitCheckErrorCode::kStateBarrierMismatch,
                WaitStateWitness{issue_id, state_wait->state, *target,
                                 definition, std::nullopt}));
    }
    const std::optional<StateToken> token =
        trace.stateToken(state_wait->state);
    if (!token.has_value()) {
      return Result<WaitCertificate, WaitCheckError>::Failure(
          Error(WaitCheckErrorCode::kMissingStateToken,
                WaitStateWitness{issue_id, state_wait->state, *target,
                                 definition, token}));
    }
    if (token->barrier != target->barrier) {
      return Result<WaitCertificate, WaitCheckError>::Failure(
          Error(WaitCheckErrorCode::kStateBarrierMismatch,
                WaitStateWitness{issue_id, state_wait->state, *target,
                                 definition, token}));
    }
    if (token->phase != target->phase) {
      return Result<WaitCertificate, WaitCheckError>::Failure(
          Error(WaitCheckErrorCode::kStatePhaseMismatch,
                WaitStateWitness{issue_id, state_wait->state, *target,
                                 definition, token}));
    }
    if (!ProfileCall("wait.state_definition", [&] {
          return ReachesWithDirectEdge(graph, reachability, *definition, issue_id);
        })) {
      return Result<WaitCertificate, WaitCheckError>::Failure(
          Error(WaitCheckErrorCode::kStateDefinitionNotOrderedBeforeIssue,
                WaitStateWitness{issue_id, state_wait->state, *target,
                                 definition, token}));
    }
    kind = StateWaitCertificate{state_wait->state, *definition};
  } else {
    const auto& parity_wait = std::get<WaitParityIssueEvent>(issue.payload);
    const std::uint8_t expected =
        static_cast<std::uint8_t>(target->phase % 2U);
    if (parity_wait.parity != expected) {
      return Result<WaitCertificate, WaitCheckError>::Failure(
          Error(WaitCheckErrorCode::kParityMismatch,
                WaitParityWitness{issue_id, *target, parity_wait.parity,
                                  expected}));
    }
    kind = ParityWaitCertificate{parity_wait.parity};
  }

  return Result<WaitCertificate, WaitCheckError>::Success(WaitCertificate{
      issue_id, done_id, *target, initialization->event, *target_completion,
      std::move(lower), std::move(upper), std::move(kind)});
}

}  // namespace

Result<WaitCertificate, WaitCheckError> WaitChecker::Check(
    EventId issue, const LoweredProgram& program, const ReferenceTrace& trace,
    const PhaseDatabase& phases, HappensBeforeGraphBuilder& proof,
    ReachabilityOracle& reachability) const {
  return CheckImpl(issue, program, trace, phases, proof.happensBeforeGraph(),
                   reachability, &proof);
}

Result<std::vector<WaitCertificate>, WaitCheckError> WaitChecker::CheckAll(
    const LoweredProgram& program, const ReferenceTrace& trace,
    const PhaseDatabase& phases, HappensBeforeGraphBuilder& proof,
    ReachabilityOracle& reachability, ParallelReachability* parallel,
    WaitReachabilityBatchOptions batch_options) const {
  std::vector<WaitCertificate> certificates;
  const auto& wait_families = program.synchronizationEvents().wait_issues;
  std::vector<EventId> issues(wait_families.begin(), wait_families.end());
  std::sort(issues.begin(), issues.end());
  if ((parallel && parallel->enabled()) || batch_options.certified_phase_ordering) {
    WaitReachabilityBatch batch;
    if (batch_options.certified_phase_ordering) {
      ProfileScope query_profile("wait.batch_queries");
      std::vector<WaitReachabilityQuery> queries;
      queries.reserve(issues.size());
      for (EventId issue : issues) {
        const auto barrier = *WaitBarrier(program.graph().event(issue));
        const auto initialization = program.barrierInitializations().find(barrier);
        const auto target = trace.waitTarget(issue);
        const auto initial = trace.initialWaitBarrier(issue);
        if (!initialization ||
            (target && (target->barrier != barrier || initial)) ||
            (!target && initial != barrier)) {
          continue;  // CheckImpl retains the precise metadata error ordering.
        }
        WaitReachabilityQuery query{
            .issue = issue, .initialization = initialization->event,
            .barrier = barrier, .previous_phase = std::nullopt,
            .next_phase = std::nullopt};
        if (target && target->phase > 0) {
          query.previous_phase = target->phase - 1;
        }
        if (!target || target->phase != std::numeric_limits<std::uint32_t>::max()) {
          const auto next = target ? target->phase + 1 : 0;
          const PhaseInfo* info = phases.find(PhaseKey{barrier, next});
          if (info && info->completed) query.next_phase = next;
        }
        queries.push_back(query);
      }
      query_profile.Stop();
      batch = WaitReachabilityBatch::Build(
          proof.happensBeforeGraph(), phases.phases(), queries, batch_options);
    }
    certificates.reserve(issues.size());
    // Bounded batches limit speculative work and preserve useful cache locality.
    constexpr std::size_t kBatchSize = 256;
    for (std::size_t begin = 0; begin < issues.size(); begin += kBatchSize) {
      const auto count = std::min(kBatchSize, issues.size() - begin);
      using Checked = Result<WaitCertificate, WaitCheckError>;
      std::vector<std::optional<Checked>> checked(count);
      const auto check = [&](std::size_t i, ReachabilityOracle& local) {
        ProfileScope profile("wait.check");
        checked[i] = CheckImpl(issues[begin + i], program, trace, phases,
                              proof.happensBeforeGraph(), local, nullptr,
                              batch.find(issues[begin + i]));
      };
      if (parallel && parallel->enabled()) {
        parallel->For(count, check);
      } else {
        for (std::size_t i = 0; i < count; ++i) check(i, reachability);
      }
      ProfileScope commit_profile("wait.commit");
      std::vector<std::pair<EventId, EventId>> pending_edges;
      std::optional<PhaseKey> pending_phase;
      const auto commit = [&]() -> std::optional<WaitCheckError> {
        if (pending_edges.empty()) return {};
        auto added = proof.AddCertifiedEdges(pending_edges,
            HappensBeforeEdgeReason::kCertifiedPreviousPhase, CertificateRef{*pending_phase});
        if (!added) return Error(WaitCheckErrorCode::kHappensBeforeGraphMutationFailed, added.error());
        reachability.NotifyGraphMutation();
        pending_edges.clear();
        return {};
      };
      for (std::size_t i = 0; i < count; ++i) {
        auto result = std::move(*checked[i]);
        if (!result) {
          if (auto error = commit())
            return Result<std::vector<WaitCertificate>, WaitCheckError>::Failure(std::move(*error));
          // Earlier commits (or this wait's own lower-freshness edge) can make
          // a speculative miss reachable. Preserve the serial error ordering.
          ProfileScope retry_profile("wait.serial_retry");
          result = Check(issues[begin + i], program, trace, phases, proof, reachability);
        } else if (const auto& lower = result.value().lower_freshness) {
          if (pending_phase && *pending_phase != lower->previous_phase)
            if (auto error = commit())
              return Result<std::vector<WaitCertificate>, WaitCheckError>::Failure(std::move(*error));
          pending_phase = lower->previous_phase;
          pending_edges.emplace_back(lower->completion, result.value().issue);
        }
        if (!result) {
          return Result<std::vector<WaitCertificate>, WaitCheckError>::Failure(
              std::move(result).error());
        }
        certificates.push_back(std::move(result).value());
      }
      if (auto error = commit())
        return Result<std::vector<WaitCertificate>, WaitCheckError>::Failure(std::move(*error));
    }
    return Result<std::vector<WaitCertificate>, WaitCheckError>::Success(
        std::move(certificates));
  }
  for (const auto issue : issues) {
    auto checked = Check(issue, program, trace, phases, proof,
                         reachability);
    if (!checked) {
      return Result<std::vector<WaitCertificate>, WaitCheckError>::Failure(
          checked.error());
    }
    certificates.push_back(std::move(checked).value());
  }
  return Result<std::vector<WaitCertificate>, WaitCheckError>::Success(
      std::move(certificates));
}

}  // namespace sentryward::algorithm
