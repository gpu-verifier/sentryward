#pragma once

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include "sentryward/algorithm/analysis/phase_algebra.h"
#include "sentryward/algorithm/analysis/phase_database.h"
#include "sentryward/algorithm/base/dynamic_bitset.h"
#include "sentryward/algorithm/base/result.h"
#include "sentryward/algorithm/event_ir/dag.h"
#include "sentryward/algorithm/solver/minimum_weight_ideal.h"

namespace sentryward::algorithm {

class HappensBeforeGraph;

struct MissingEventBound final {
  EventId omitted_effect;
  bool feasible = false;
  std::int64_t minimum_transaction = 0;
  DynamicBitset ideal;

  friend bool operator==(const MissingEventBound&,
                         const MissingEventBound&) = default;
};

struct RemainingCountBound final {
  std::optional<EventId> omitted_effect;
  bool feasible = false;
  std::int64_t minimum_remaining = 0;
  DynamicBitset ideal;
};

struct ArrivalIncrementBounds final {
  std::uint64_t increment_total = 0;
  std::int64_t minimum_pending = 0;
  std::int64_t maximum_pending = 0;
  // Pending arrivals plus transaction bytes: zero iff both counters are zero.
  std::vector<RemainingCountBound> remaining_bounds;
};

enum class TransactionBoundDomain {
  kAllPrefixes,
  kZeroPendingPrefixes,
  kNoZeroPendingPrefix,
};

struct PhaseCertificate final {
  PhaseKey phase;
  std::uint64_t arrival_total = 0;
  std::int64_t transaction_total = 0;
  std::int64_t minimum_prefix_tx = 0;
  DynamicBitset minimum_prefix_ideal;
  std::vector<MissingEventBound> proper_prefix_bounds;
  std::optional<ArrivalIncrementBounds> arrival_increments = std::nullopt;
  TransactionBoundDomain transaction_bound_domain =
      TransactionBoundDomain::kAllPrefixes;
  // False permits a dependency-free proof instead of exact proper-prefix
  // minima. Exact ideal payloads are omitted in either compact proof form.
  bool successful_ideals_retained = true;
  // When present, proper_prefix_bounds is empty: these are conservative lower
  // bounds, not claims of exact query feasibility or minimizing ideals.
  // minimum_prefix_tx remains exact (zero for a balanced completed phase).
  std::optional<AlgebraicPhaseBounds> algebraic_bounds = std::nullopt;
};

struct OpenPhaseCertificate final {
  PhaseKey phase;
  std::uint64_t arrival_total = 0;
  std::int64_t transaction_total = 0;
  std::int64_t minimum_prefix_tx = 0;
  DynamicBitset minimum_prefix_ideal;
  std::vector<MissingEventBound> proper_prefix_bounds;
  std::optional<ArrivalIncrementBounds> arrival_increments = std::nullopt;
  TransactionBoundDomain transaction_bound_domain =
      TransactionBoundDomain::kAllPrefixes;
  // False means successful ideal payloads are omitted; scalar bounds remain.
  bool successful_ideals_retained = true;
};

struct PhaseBalanceWitness final {
  PhaseKey phase;
  std::uint64_t expected_arrivals = 0;
  std::uint64_t actual_arrivals = 0;
  std::int64_t transaction_total = 0;
  std::uint64_t arrival_increment_total = 0;
};

struct PhaseIdealWitness final {
  PhaseKey phase;
  DynamicBitset ideal;
  std::vector<EventId> topological_schedule;
  std::uint64_t arrival_total = 0;
  std::int64_t transaction_total = 0;
  std::optional<EventId> omitted_effect;
  std::uint64_t arrival_increment_total = 0;
};

struct PhaseSolverWitness final {
  PhaseKey phase;
  IdealStatus status = IdealStatus::kInvalidInput;
};

using PhaseBoundaryWitness =
    std::variant<std::monostate, PhaseBalanceWitness, PhaseIdealWitness,
                 PhaseSolverWitness>;

enum class PhaseBoundaryErrorCode {
  kUnknownPhase,
  kWrongPhaseKind,
  kInvalidGraph,
  kInvalidEffect,
  kArrivalArithmeticOverflow,
  kTransactionArithmeticOverflow,
  kArrivalBalanceMismatch,
  kTransactionBalanceMismatch,
  kNegativeTransactionPrefix,
  kPrematurePhaseCompletion,
  kOpenPhaseArrivalOverflow,
  kOpenPhaseNotStrictlyPositive,
  kIdealSolverFailure,
  kNegativeArrivalPrefix,
  kArrivalPrefixOverflow,
};

struct PhaseBoundaryError final {
  PhaseBoundaryErrorCode code = PhaseBoundaryErrorCode::kUnknownPhase;
  PhaseBoundaryWitness witness;
};

class PhaseBoundaryChecker final {
 public:
  [[nodiscard]] Result<PhaseCertificate, PhaseBoundaryError>
  CheckCompletedPhase(const EventDag& graph, const PhaseDatabase& phases,
                      PhaseKey phase, bool build_witness_schedule = true,
                      bool reject_negative_tx = false,
                      bool retain_successful_ideals = true) const;

  [[nodiscard]] Result<PhaseCertificate, PhaseBoundaryError>
  CheckCompletedPhase(const HappensBeforeGraph& graph, const PhaseDatabase& phases,
                      PhaseKey phase, bool build_witness_schedule = true,
                      bool reject_negative_tx = false,
                      bool retain_successful_ideals = true) const;

  [[nodiscard]] Result<OpenPhaseCertificate, PhaseBoundaryError> CheckOpenPhase(
      const EventDag& graph, const PhaseDatabase& phases, PhaseKey phase,
      bool build_witness_schedule = true,
      bool reject_negative_tx = false,
      bool retain_successful_ideals = true) const;

  [[nodiscard]] Result<OpenPhaseCertificate, PhaseBoundaryError> CheckOpenPhase(
      const HappensBeforeGraph& graph, const PhaseDatabase& phases, PhaseKey phase,
      bool build_witness_schedule = true,
      bool reject_negative_tx = false,
      bool retain_successful_ideals = true) const;

 private:
  MinimumWeightIdealSolver ideal_solver_;
};

}  // namespace sentryward::algorithm
