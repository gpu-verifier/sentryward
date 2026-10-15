#pragma once

#include <map>
#include <optional>
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

struct RejectionProvenance final {
  std::map<EventId, EventProvenance> events;

  [[nodiscard]] const EventProvenance* find(EventId event) const noexcept {
    const auto found = events.find(event);
    return found == events.end() ? nullptr : &found->second;
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

using RejectionWitness =
    std::variant<ReferenceDeadlock, SemanticViolation,
                 InitializationOrderWitness, PreviousPhaseSeparationWitness,
                 PhaseBoundaryError, WaitCheckError, LifecycleCheckError,
                 HappensBeforeGraphMutationError, FinalHappensBeforeCycleWitness>;

} // namespace sentryward::algorithm
