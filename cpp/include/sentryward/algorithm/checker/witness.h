#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <variant>
#include <vector>

#include "sentryward/algorithm/analysis/lifecycle_checker.h"
#include "sentryward/algorithm/analysis/phase_boundary.h"
#include "sentryward/algorithm/analysis/phase_database.h"
#include "sentryward/algorithm/analysis/happens_before_graph.h"
#include "sentryward/algorithm/analysis/wait_checker.h"
#include "sentryward/algorithm/base/source_location.h"
#include "sentryward/algorithm/semantics/simulator.h"

namespace sentryward::algorithm {

struct SourceEventProvenance final {
  InstId instruction;
  SourceLocation location;

  friend bool operator==(const SourceEventProvenance&,
                         const SourceEventProvenance&) = default;
};

enum class SyntheticEventKind {
  kCollectiveCompletion,
  kMBarrierPhaseCompletion,
  kInternalDerived,
};

struct SyntheticEventProvenance final {
  SyntheticEventKind kind = SyntheticEventKind::kInternalDerived;
  std::optional<SyncId> sync_id;
  std::optional<PhaseKey> phase;

  friend bool operator==(const SyntheticEventProvenance&,
                         const SyntheticEventProvenance&) = default;
};

using EventProvenance =
    std::variant<SourceEventProvenance, SyntheticEventProvenance>;

// The longest event list the rejection report prints (source_json.cpp RejectionReport): it names the
// first kWitnessEventListLimit events of every event list, and every single event, of the witness
// and of the counterexample (ForEachReportedEvent below).
inline constexpr std::size_t kWitnessEventListLimit = 32;

// The provenance of the events of a rejected graph that the report names (wellsync_checker.cpp
// resolves exactly ForEachReportedEvent's events), and how many events the graph has: its events
// are the ids 0 .. total - 1 and every one of them has a provenance. A source event keeps its source
// instruction in the checked program, whose location the report reads from that program; a
// synthetic event keeps its identity (a collective's or an mbarrier phase's completion, or an
// internal derived event). O(named events) entries.
struct RejectionProvenance final {
  std::unordered_map<std::uint32_t, InstId> sources;
  std::unordered_map<std::uint32_t, SyntheticEventProvenance> synthetic;
  // The number of events with a provenance (every event of the graph).
  std::size_t total = 0;

  // Records `event` as a source event of instruction `instruction`, or as a synthetic one; an event
  // recorded before keeps its first provenance. O(1) amortized.
  void addSource(EventId event, InstId instruction) {
    if (!resolved(event)) sources.emplace(event.value(), instruction);
  }
  void addSynthetic(EventId event, SyntheticEventProvenance provenance) {
    if (!resolved(event)) synthetic.emplace(event.value(), std::move(provenance));
  }
  [[nodiscard]] bool resolved(EventId event) const {
    return sources.contains(event.value()) || synthetic.contains(event.value());
  }
  // Whether the event has a provenance (an event of the graph).
  [[nodiscard]] bool contains(EventId event) const noexcept { return event.value() < total; }
  // The event's source instruction, when it is a source event. An event of the graph that was not
  // resolved is one the report was not expected to name: a logic error, never a silent default.
  [[nodiscard]] std::optional<InstId> source(EventId event) const {
    if (!contains(event)) return std::nullopt;
    if (const auto found = sources.find(event.value()); found != sources.end()) return found->second;
    if (!synthetic.contains(event.value()))
      throw std::logic_error("rejection provenance: a named event was not resolved");
    return std::nullopt;
  }
  // A synthetic event's provenance (for an event of the graph without a source instruction).
  [[nodiscard]] SyntheticEventProvenance syntheticOf(EventId event) const {
    const auto found = synthetic.find(event.value());
    if (found == synthetic.end()) throw std::logic_error("rejection provenance: a named event was not resolved");
    return found->second;
  }
};

struct CounterexampleExecution final {
  std::vector<EventId> candidate_order;
  std::vector<EventId> replayed_candidate_order;
  ReferenceTrace executable_prefix;
  std::vector<EventId> unexecuted_candidate;
  bool candidate_order_preserved = true;
  std::optional<ReplayResult::Stall> stall;
  std::optional<SimulationWitness> stop;
};

struct RejectionContext final {
  RejectionProvenance provenance;
  std::optional<CounterexampleExecution> counterexample;
};

struct InitializationOrderWitness final {
  PhaseKey phase;
  EventId effect;
  EventId initialization;
  std::vector<EventId> counterexample_schedule;

  friend bool operator==(const InitializationOrderWitness&,
                         const InitializationOrderWitness&) = default;
};

struct PreviousPhaseSeparationWitness final {
  PhaseKey phase;
  EventId effect;
  PhaseKey previous_phase;
  std::vector<EventId> unordered_previous_effects;
  std::vector<EventId> counterexample_schedule;

  friend bool operator==(const PreviousPhaseSeparationWitness&,
                         const PreviousPhaseSeparationWitness&) = default;
};

struct FinalHappensBeforeCycleWitness final {
  std::vector<EventId> partial_topological_order;

  friend bool operator==(const FinalHappensBeforeCycleWitness&,
                         const FinalHappensBeforeCycleWitness&) = default;
};

// A thread never arrives at `phase` of a CTA-wide barrier (ExitGate), yet the
// phase completion reaches the thread's exit: the barrier waits for the exit,
// which waits for the barrier.
struct ExitGateWitness final {
  PhaseKey phase;
  EventId exit;
  EventId completion;

  friend bool operator==(const ExitGateWitness&,
                         const ExitGateWitness&) = default;
};

using RejectionWitness =
    std::variant<ReferenceDeadlock, SemanticViolation,
                 InitializationOrderWitness, PreviousPhaseSeparationWitness,
                 PhaseBoundaryError, WaitCheckError, LifecycleCheckError,
                 HappensBeforeGraphMutationError, FinalHappensBeforeCycleWitness,
                 ExitGateWitness>;

// Every event the rejection report names, in no particular order (an event may be visited twice):
// each single event of the witness and of the counterexample, and the first kWitnessEventListLimit
// events of each of their event lists. It mirrors source_json.cpp RejectionReport field by field;
// the report fails (RejectionProvenance::source) if it names an event this did not visit.
// O(named events).
namespace reported_events {
template <typename Fn> void One(Fn& fn, EventId event) { fn(event); }
template <typename Fn> void One(Fn& fn, const std::optional<EventId>& event) { if (event) fn(*event); }
template <typename Fn, typename Range> void Many(Fn& fn, const Range& events) {
  std::size_t count = 0;
  for (const auto event : events) {
    if (count++ == kWitnessEventListLimit) break;
    fn(EventId(event));
  }
}
// (Declared first: the variant visitors below call overloads defined after them.)
template <typename Fn> void Visit(Fn& fn, const ReferenceDeadlock& w);
template <typename Fn> void Visit(Fn& fn, const SemanticViolation& w);
template <typename Fn> void Visit(Fn& fn, const InitializationOrderWitness& w);
template <typename Fn> void Visit(Fn& fn, const PreviousPhaseSeparationWitness& w);
template <typename Fn> void Visit(Fn& fn, const std::monostate& w);
template <typename Fn> void Visit(Fn& fn, const PhaseBalanceWitness& w);
template <typename Fn> void Visit(Fn& fn, const PhaseIdealWitness& w);
template <typename Fn> void Visit(Fn& fn, const PhaseSolverWitness& w);
template <typename Fn> void Visit(Fn& fn, const WaitEventWitness& w);
template <typename Fn> void Visit(Fn& fn, const WaitReachabilityWitness& w);
template <typename Fn> void Visit(Fn& fn, const WaitStateWitness& w);
template <typename Fn> void Visit(Fn& fn, const WaitParityWitness& w);
template <typename Fn> void Visit(Fn& fn, const PhaseBoundaryError& w);
template <typename Fn> void Visit(Fn& fn, const WaitCheckError& w);
template <typename Fn> void Visit(Fn& fn, const LifecycleCheckError& w);
template <typename Fn> void Visit(Fn& fn, const HappensBeforeGraphMutationError& w);
template <typename Fn> void Visit(Fn& fn, const FinalHappensBeforeCycleWitness& w);
template <typename Fn> void Visit(Fn& fn, const ExitGateWitness& w);
template <typename Fn> void Visit(Fn& fn, const ReferenceDeadlock& w) { Many(fn, w.executed); Many(fn, w.blocked); }
template <typename Fn> void Visit(Fn& fn, const SemanticViolation& w) { One(fn, w.event); }
template <typename Fn> void Visit(Fn& fn, const InitializationOrderWitness& w) {
  One(fn, w.effect); One(fn, w.initialization); Many(fn, w.counterexample_schedule);
}
template <typename Fn> void Visit(Fn& fn, const PreviousPhaseSeparationWitness& w) {
  One(fn, w.effect); Many(fn, w.unordered_previous_effects); Many(fn, w.counterexample_schedule);
}
template <typename Fn> void Visit(Fn&, const std::monostate&) {}
template <typename Fn> void Visit(Fn&, const PhaseBalanceWitness&) {}
template <typename Fn> void Visit(Fn& fn, const PhaseIdealWitness& w) { One(fn, w.omitted_effect); Many(fn, w.topological_schedule); }
template <typename Fn> void Visit(Fn&, const PhaseSolverWitness&) {}
template <typename Fn> void Visit(Fn& fn, const WaitEventWitness& w) { One(fn, w.issue); One(fn, w.related_event); }
template <typename Fn> void Visit(Fn& fn, const WaitReachabilityWitness& w) { One(fn, w.issue); Many(fn, w.required_events); }
template <typename Fn> void Visit(Fn& fn, const WaitStateWitness& w) { One(fn, w.issue); One(fn, w.definition); }
template <typename Fn> void Visit(Fn& fn, const WaitParityWitness& w) { One(fn, w.issue); }
template <typename Fn> void Visit(Fn& fn, const PhaseBoundaryError& w) {
  std::visit([&](const auto& item) { Visit(fn, item); }, w.witness);
}
template <typename Fn> void Visit(Fn& fn, const WaitCheckError& w) {
  std::visit([&](const auto& item) { Visit(fn, item); }, w.witness);
}
template <typename Fn> void Visit(Fn& fn, const LifecycleCheckError& w) {
  One(fn, w.witness.positive_arrival); Many(fn, w.witness.candidate_waits);
}
template <typename Fn> void Visit(Fn& fn, const HappensBeforeGraphMutationError& w) { One(fn, w.from); One(fn, w.to); }
template <typename Fn> void Visit(Fn& fn, const FinalHappensBeforeCycleWitness& w) { Many(fn, w.partial_topological_order); }
template <typename Fn> void Visit(Fn& fn, const ExitGateWitness& w) { One(fn, w.exit); One(fn, w.completion); }
} // namespace reported_events
template <typename Fn>
void ForEachReportedEvent(const RejectionWitness& witness, const CounterexampleExecution* execution, Fn fn) {
  std::visit([&](const auto& item) { reported_events::Visit(fn, item); }, witness);
  if (!execution) return;
  reported_events::Many(fn, execution->candidate_order);
  reported_events::Many(fn, execution->replayed_candidate_order);
  reported_events::Many(fn, execution->executable_prefix.executedEvents());
  reported_events::Many(fn, execution->unexecuted_candidate);
  if (execution->stall) fn(execution->stall->first_unexecuted);
  if (execution->stop) std::visit([&](const auto& item) { reported_events::Visit(fn, item); }, *execution->stop);
}

} // namespace sentryward::algorithm
