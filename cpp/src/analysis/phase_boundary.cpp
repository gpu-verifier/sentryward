#include "sentryward/algorithm/diagnostics/performance_profile.h"
#include "sentryward/algorithm/analysis/phase_boundary.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <unordered_map>
#include <utility>

#include "sentryward/algorithm/analysis/happens_before_graph.h"
#include "sentryward/algorithm/base/checked_int.h"

namespace sentryward::algorithm {
namespace {

struct PreparedPhase final {
  std::vector<std::int64_t> weights;
  // Only phase effects are queried here; indexing every graph vertex made
  // even a one-effect phase pay O(|V|) initialization and allocation costs.
  std::unordered_map<std::uint32_t, IdealVertex> local_index;
  DirectedGraph projected;
  std::vector<PhaseEffectWeights> effects;
  std::vector<EventId> nonarrival_effects;
  std::uint64_t arrival_total = 0;
  std::uint64_t arrival_increment_total = 0;
  std::int64_t transaction_total = 0;
};

struct SignedAccumulator final {
  std::uint64_t positive = 0;
  std::uint64_t negative = 0;
};

[[nodiscard]] PhaseBoundaryError SimpleError(PhaseBoundaryErrorCode code) {
  return PhaseBoundaryError{code, std::monostate{}};
}

[[nodiscard]] bool Accumulate(SignedAccumulator& total, std::int64_t value) {
  const bool is_negative = value < 0;
  const std::uint64_t magnitude =
      is_negative ? std::uint64_t{0} - static_cast<std::uint64_t>(value)
                  : static_cast<std::uint64_t>(value);
  std::uint64_t& destination = is_negative ? total.negative : total.positive;
  auto sum = CheckedAdd(destination, magnitude);
  if (!sum) {
    return false;
  }
  destination = sum.value();
  return true;
}

[[nodiscard]] std::optional<std::int64_t> Value(
    const SignedAccumulator& total) {
  if (total.positive >= total.negative) {
    const std::uint64_t magnitude = total.positive - total.negative;
    if (magnitude >
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      return std::nullopt;
    }
    return static_cast<std::int64_t>(magnitude);
  }

  const std::uint64_t magnitude = total.negative - total.positive;
  constexpr std::uint64_t kMinimumMagnitude =
      std::uint64_t{1} << std::numeric_limits<std::int64_t>::digits;
  if (magnitude > kMinimumMagnitude) {
    return std::nullopt;
  }
  if (magnitude == kMinimumMagnitude) {
    return std::numeric_limits<std::int64_t>::min();
  }
  return -static_cast<std::int64_t>(magnitude);
}

constexpr IdealVertex kNotRetained = std::numeric_limits<IdealVertex>::max();

class FrontierPool final {
 public:
  using Id = std::uint32_t;
  static constexpr std::size_t kMaximumFrontiers = 65536;
  static constexpr std::size_t kMaximumEntries = 4 * 1024 * 1024;

  FrontierPool() { frontiers_.emplace_back(); }

  [[nodiscard]] const std::vector<IdealVertex>& get(Id id) const {
    return frontiers_[id];
  }

  [[nodiscard]] std::optional<Id> intern(std::vector<IdealVertex> values) {
    if (values.empty()) return 0;
    std::size_t hash = values.size();
    for (IdealVertex value : values)
      hash ^= value + std::size_t{0x9e3779b9} + (hash << 6) + (hash >> 2);
    const auto [first, last] = fingerprints_.equal_range(hash);
    for (auto found = first; found != last; ++found)
      if (frontiers_[found->second] == values) return found->second;
    if (frontiers_.size() >= kMaximumFrontiers ||
        values.capacity() > kMaximumEntries - entries_) return std::nullopt;
    const Id id = static_cast<Id>(frontiers_.size());
    entries_ += values.capacity();
    frontiers_.push_back(std::move(values));
    fingerprints_.emplace(hash, id);
    return id;
  }

  [[nodiscard]] std::optional<Id> merge(Id left, Id right) {
    if (left == right || right == 0) return left;
    if (left == 0) return right;
    const auto& first = get(left);
    const auto& second = get(right);
    std::vector<IdealVertex> combined;
    combined.reserve(first.size() + second.size());
    std::set_union(first.begin(), first.end(), second.begin(), second.end(),
                   std::back_inserter(combined));
    if (combined == first) return left;
    if (combined == second) return right;
    return intern(std::move(combined));
  }

 private:
  std::vector<std::vector<IdealVertex>> frontiers_;
  std::unordered_multimap<std::size_t, Id> fingerprints_;
  std::size_t entries_ = 0;
};

template <typename Graph>
[[nodiscard]] bool ProjectMemoized(const Graph& graph, const PhaseInfo& phase,
                                  DirectedGraph& projected) {
  ProfileScope profile("boundary.project_memoized");
  if (phase.effects.size() <= 1) return true;
  if (phase.effects.size() >= FrontierPool::kMaximumFrontiers) return false;
  FrontierPool pool;
  auto frontier_of = ProfileCall("boundary.project_workspace", [&] {
    return std::vector<FrontierPool::Id>(graph.eventCount(), kNotRetained);
  });
  for (IdealVertex local = 0; local < phase.effects.size(); ++local) {
    const auto singleton = pool.intern({local});
    if (!singleton) return false;
    frontier_of[phase.effects[local].value()] = *singleton;
  }
  struct Frame final {
    EventId event;
    FrontierPool::Id frontier = 0;
    std::size_t next_predecessor = 0;
  };
  std::vector<Frame> pending;
  const auto resolve = [&](EventId root) -> bool {
    if (frontier_of[root.value()] != kNotRetained) return true;
    pending.push_back({root});
    while (!pending.empty()) {
      Frame& frame = pending.back();
      const auto predecessors = graph.predecessors(frame.event);
      if (frame.next_predecessor == predecessors.size()) {
        ++profile.nodes;
        frontier_of[frame.event.value()] = frame.frontier;
        pending.pop_back();
        continue;
      }
      const EventId predecessor = predecessors[frame.next_predecessor];
      if (frontier_of[predecessor.value()] == kNotRetained) {
        pending.push_back({predecessor});
        continue;
      }
      const auto merged = pool.merge(frame.frontier, frontier_of[predecessor.value()]);
      ++profile.edges;
      if (!merged) return false;
      frame.frontier = *merged;
      ++frame.next_predecessor;
    }
    return true;
  };
  for (IdealVertex local = 0; local < phase.effects.size(); ++local) {
    FrontierPool::Id incoming = 0;
    for (EventId predecessor : graph.predecessors(phase.effects[local])) {
      if (!resolve(predecessor)) return false;
      const auto merged = pool.merge(incoming, frontier_of[predecessor.value()]);
      ++profile.edges;
      if (!merged) return false;
      incoming = *merged;
    }
    for (IdealVertex predecessor : pool.get(incoming))
      projected.edges.push_back({predecessor, local});
  }
  return true;
}

template <typename Graph>
void ProjectByTraversal(const Graph& graph, const PhaseInfo& phase,
                        const std::unordered_map<std::uint32_t, IdealVertex>& local_index,
                        DirectedGraph& projected) {
  ProfileScope profile("boundary.project_fallback");
  std::vector<IdealVertex> visited(graph.eventCount(), kNotRetained);
  std::vector<EventId> pending;
  for (IdealVertex local = 0; local < phase.effects.size(); ++local) {
    pending.push_back(phase.effects[local]);
    while (!pending.empty()) {
      const EventId current = pending.back();
      pending.pop_back();
      for (EventId predecessor : graph.predecessors(current)) {
        if (visited[predecessor.value()] == local) continue;
        visited[predecessor.value()] = local;
        const auto retained = local_index.find(predecessor.value());
        if (retained == local_index.end())
          pending.push_back(predecessor);
        else
          projected.edges.push_back({retained->second, local});
      }
    }
  }
}

template <typename Graph>
[[nodiscard]] Result<PreparedPhase, PhaseBoundaryError> Prepare(
    const Graph& graph, const PhaseInfo& phase) {
  ProfileScope profile("boundary.prepare");
  profile.nodes = phase.effects.size();
  PreparedPhase prepared;
  prepared.weights.reserve(phase.effects.size());
  prepared.local_index.reserve(phase.effects.size());
  prepared.effects.reserve(phase.effects.size());

  std::uint64_t arrival_total = 0;
  SignedAccumulator transaction_total;
  for (EventId event_id : phase.effects) {
    if (!graph.contains(event_id)) {
      return Result<PreparedPhase, PhaseBoundaryError>::Failure(
          SimpleError(PhaseBoundaryErrorCode::kInvalidEffect));
    }
    auto decoded = GetPhaseEffectWeights(graph.event(event_id));
    if (!decoded || !decoded.value().has_value() ||
        decoded.value()->barrier != phase.barrier) {
      const PhaseBoundaryErrorCode code =
          !decoded && decoded.error().code ==
                          PhaseDatabaseErrorCode::kArithmeticOverflow
              ? PhaseBoundaryErrorCode::kTransactionArithmeticOverflow
              : PhaseBoundaryErrorCode::kInvalidEffect;
      return Result<PreparedPhase, PhaseBoundaryError>::Failure(
          SimpleError(code));
    }
    const PhaseEffectWeights effect = *decoded.value();
    const auto [local, inserted] = prepared.local_index.emplace(
        event_id.value(), static_cast<IdealVertex>(prepared.weights.size()));
    (void)local;
    if (!inserted) {
      return Result<PreparedPhase, PhaseBoundaryError>::Failure(
          SimpleError(PhaseBoundaryErrorCode::kInvalidEffect));
    }
    prepared.weights.push_back(effect.transaction_weight);
    prepared.effects.push_back(effect);
    auto next_arrival = CheckedAdd(
        arrival_total, static_cast<std::uint64_t>(effect.arrival_weight));
    if (!next_arrival) {
      return Result<PreparedPhase, PhaseBoundaryError>::Failure(
          SimpleError(PhaseBoundaryErrorCode::kArrivalArithmeticOverflow));
    }
    arrival_total = next_arrival.value();
    auto next_increment =
        CheckedAdd(prepared.arrival_increment_total,
                   static_cast<std::uint64_t>(effect.arrival_increment));
    if (!next_increment) {
      return Result<PreparedPhase, PhaseBoundaryError>::Failure(
          SimpleError(PhaseBoundaryErrorCode::kArrivalArithmeticOverflow));
    }
    prepared.arrival_increment_total = next_increment.value();
    if (!Accumulate(transaction_total, effect.transaction_weight)) {
      return Result<PreparedPhase, PhaseBoundaryError>::Failure(
          SimpleError(PhaseBoundaryErrorCode::kTransactionArithmeticOverflow));
    }
    if (effect.arrival_weight == 0) {
      prepared.nonarrival_effects.push_back(event_id);
    }
  }

  constexpr auto kMaximumTransaction =
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  if (transaction_total.positive > kMaximumTransaction ||
      transaction_total.negative > kMaximumTransaction) {
    return Result<PreparedPhase, PhaseBoundaryError>::Failure(
        SimpleError(PhaseBoundaryErrorCode::kTransactionArithmeticOverflow));
  }

  const std::optional<std::int64_t> transaction_value =
      Value(transaction_total);
  if (!transaction_value.has_value()) {
    return Result<PreparedPhase, PhaseBoundaryError>::Failure(
        SimpleError(PhaseBoundaryErrorCode::kTransactionArithmeticOverflow));
  }
  prepared.arrival_total = arrival_total;
  prepared.transaction_total = *transaction_value;
  for (EventId arrival : phase.positive_arrivals) {
    if (!graph.contains(arrival) ||
        !prepared.local_index.contains(arrival.value())) {
      return Result<PreparedPhase, PhaseBoundaryError>::Failure(
          SimpleError(PhaseBoundaryErrorCode::kInvalidEffect));
    }
  }
  prepared.projected.vertex_count = phase.effects.size();
  return Result<PreparedPhase, PhaseBoundaryError>::Success(
      std::move(prepared));
}

// Projection is deliberately lazy: arithmetic certificates and some open
// phases do not depend on the graph relation at all.
template <typename Graph>
void PrepareProjection(const Graph& graph, const PhaseInfo& phase,
                       PreparedPhase& prepared) {
  ProfileScope profile("boundary.project");
  profile.nodes = phase.effects.size();
  if (!ProjectMemoized(graph, phase, prepared.projected)) {
    prepared.projected.edges.clear();
    ProjectByTraversal(graph, phase, prepared.local_index, prepared.projected);
  }
  profile.edges = prepared.projected.edges.size();
}

template <typename Graph>
[[nodiscard]] DynamicBitset LiftIdeal(
    const Graph& graph, const PhaseInfo& phase, const DynamicBitset& local,
    bool retain = true) {
  if (!retain) return {};
  DynamicBitset global(graph.eventCount());
  std::vector<EventId> pending;
  for (std::size_t index = 0; index < phase.effects.size(); ++index) {
    if (local.test(index)) {
      global.set(phase.effects[index].value());
      pending.push_back(phase.effects[index]);
    }
  }
  while (!pending.empty()) {
    const EventId current = pending.back();
    pending.pop_back();
    for (EventId predecessor : graph.predecessors(current)) {
      if (!global.test(predecessor.value())) {
        global.set(predecessor.value());
        pending.push_back(predecessor);
      }
    }
  }
  return global;
}

template <typename Graph>
[[nodiscard]] Result<PhaseIdealWitness, PhaseBoundaryError> MakeIdealWitness(
    const Graph& graph, const PhaseInfo& phase,
    const PreparedPhase& prepared, const IdealResult& ideal,
    std::optional<EventId> omitted_effect, bool build_schedule) {
  std::uint64_t arrival_total = 0;
  std::uint64_t increment_total = 0;
  SignedAccumulator transaction_total;
  for (std::size_t index = 0; index < phase.effects.size(); ++index) {
    if (ideal.ideal.test(index)) {
      auto next_arrival = CheckedAdd(
          arrival_total,
          static_cast<std::uint64_t>(prepared.effects[index].arrival_weight));
      if (!next_arrival) {
        return Result<PhaseIdealWitness, PhaseBoundaryError>::Failure(
            SimpleError(PhaseBoundaryErrorCode::kArrivalArithmeticOverflow));
      }
      arrival_total = next_arrival.value();
      const auto next_increment = CheckedAdd(
          increment_total, static_cast<std::uint64_t>(
                               prepared.effects[index].arrival_increment));
      if (!next_increment) {
        return Result<PhaseIdealWitness, PhaseBoundaryError>::Failure(
            SimpleError(PhaseBoundaryErrorCode::kArrivalArithmeticOverflow));
      }
      increment_total = next_increment.value();
      if (!Accumulate(transaction_total,
                      prepared.effects[index].transaction_weight)) {
        return Result<PhaseIdealWitness, PhaseBoundaryError>::Failure(
            SimpleError(
                PhaseBoundaryErrorCode::kTransactionArithmeticOverflow));
      }
    }
  }
  const std::optional<std::int64_t> transaction_value =
      Value(transaction_total);
  if (!transaction_value.has_value()) {
    return Result<PhaseIdealWitness, PhaseBoundaryError>::Failure(
        SimpleError(PhaseBoundaryErrorCode::kTransactionArithmeticOverflow));
  }

  DynamicBitset global_ideal = LiftIdeal(graph, phase, ideal.ideal);
  std::vector<EventId> schedule;
  if (build_schedule) {
    const auto topological = graph.topologicalOrder();
    if (!topological.has_value()) {
      return Result<PhaseIdealWitness, PhaseBoundaryError>::Failure(
          SimpleError(PhaseBoundaryErrorCode::kInvalidGraph));
    }
    schedule.reserve(global_ideal.count());
    for (EventId event : *topological) {
      if (global_ideal.test(event.value())) {
        schedule.push_back(event);
      }
    }
  }

  return Result<PhaseIdealWitness, PhaseBoundaryError>::Success(
      PhaseIdealWitness{phase.key(), std::move(global_ideal), std::move(schedule),
                        arrival_total, *transaction_value, omitted_effect,
                        increment_total});
}

[[nodiscard]] PhaseBoundaryError SolverError(PhaseKey phase,
                                             IdealStatus status) {
  return PhaseBoundaryError{PhaseBoundaryErrorCode::kIdealSolverFailure,
                            PhaseSolverWitness{phase, status}};
}

[[nodiscard]] IdealQuery ProperPrefixQuery(const PreparedPhase& prepared,
                                           const PhaseInfo& phase,
                                           EventId omitted) {
  IdealQuery query{DynamicBitset(prepared.weights.size()),
                   DynamicBitset(prepared.weights.size())};
  for (EventId arrival : phase.positive_arrivals) {
    query.force_in.set(prepared.local_index.at(arrival.value()));
  }
  query.force_out.set(prepared.local_index.at(omitted.value()));
  return query;
}

[[nodiscard]] IdealQuery TransactionBoundQuery(const PreparedPhase& prepared,
                                              const PhaseInfo& phase,
                                              bool reject_negative_tx) {
  IdealQuery query{DynamicBitset(prepared.weights.size()),
                   DynamicBitset(prepared.weights.size())};
  if (!reject_negative_tx) {
    for (EventId arrival : phase.positive_arrivals) {
      query.force_in.set(prepared.local_index.at(arrival.value()));
    }
  }
  return query;
}

template <typename Certificate, typename Graph>
[[nodiscard]] Result<Certificate, PhaseBoundaryError> CheckProperPrefixes(
    const MinimumWeightIdealSolver& solver, const Graph& graph,
    const PhaseInfo& phase,
    const PreparedPhase& prepared, Certificate certificate,
    bool build_witness_schedule) {
  for (EventId omitted : prepared.nonarrival_effects) {
    const IdealResult result =
        solver.Solve(prepared.projected, prepared.weights,
                     ProperPrefixQuery(prepared, phase, omitted));
    if (result.status == IdealStatus::kInfeasible) {
      certificate.proper_prefix_bounds.push_back(
          MissingEventBound{omitted, false, 0,
              LiftIdeal(graph, phase, result.ideal, certificate.successful_ideals_retained)});
      continue;
    }
    if (!result.ok() || !result.feasible) {
      return Result<Certificate, PhaseBoundaryError>::Failure(
          SolverError(phase.key(), result.status));
    }
    if (result.minimum_weight <= 0) {
      auto witness = MakeIdealWitness(graph, phase, prepared, result, omitted,
                                      build_witness_schedule);
      if (!witness) {
        return Result<Certificate, PhaseBoundaryError>::Failure(
            witness.error());
      }
      return Result<Certificate, PhaseBoundaryError>::Failure(
          PhaseBoundaryError{PhaseBoundaryErrorCode::kPrematurePhaseCompletion,
                             std::move(witness).value()});
    }
    certificate.proper_prefix_bounds.push_back(
        MissingEventBound{omitted, true, result.minimum_weight,
            LiftIdeal(graph, phase, result.ideal, certificate.successful_ideals_retained)});
  }
  return Result<Certificate, PhaseBoundaryError>::Success(
      std::move(certificate));
}

template <typename Certificate, typename Graph>
[[nodiscard]] Result<Certificate, PhaseBoundaryError> CheckIncrementPhase(
    const MinimumWeightIdealSolver& solver, const Graph& graph,
    const PhaseInfo& phase, const PreparedPhase& prepared,
    std::uint32_t expected, bool completed, bool build_schedule,
    bool retain_successful_ideals) {
  using Output = Result<Certificate, PhaseBoundaryError>;
  const DirectedGraph& solver_graph = prepared.projected;
  const auto adjusted_expected = CheckedAdd(
      static_cast<std::uint64_t>(expected), prepared.arrival_increment_total);
  if (!adjusted_expected)
    return Output::Failure(
        SimpleError(PhaseBoundaryErrorCode::kArrivalArithmeticOverflow));
  const auto balanceError = [&](PhaseBoundaryErrorCode code) {
    return Output::Failure(PhaseBoundaryError{
        code, PhaseBalanceWitness{phase.key(), expected, prepared.arrival_total,
                                  prepared.transaction_total,
                                  prepared.arrival_increment_total}});
  };
  if (completed && prepared.arrival_total != adjusted_expected.value()) {
    return balanceError(PhaseBoundaryErrorCode::kArrivalBalanceMismatch);
  }
  if (completed && prepared.transaction_total != 0) {
    return balanceError(PhaseBoundaryErrorCode::kTransactionBalanceMismatch);
  }

  const auto idealError =
      [&](PhaseBoundaryErrorCode code, const IdealResult& ideal,
          std::optional<EventId> omitted = std::nullopt) -> Output {
    auto witness = MakeIdealWitness(graph, phase, prepared, ideal, omitted,
                                    build_schedule);
    if (!witness) return Output::Failure(witness.error());
    return Output::Failure(
        PhaseBoundaryError{code, std::move(witness).value()});
  };
  const IdealResult tx = solver.Solve(solver_graph, prepared.weights);
  if (!tx.ok() || !tx.feasible)
    return Output::Failure(SolverError(phase.key(), tx.status));
  if (tx.minimum_weight < 0)
    return idealError(PhaseBoundaryErrorCode::kNegativeTransactionPrefix, tx);

  std::vector<std::int64_t> pending(prepared.weights.size(), 0),
      negative_pending(prepared.weights.size(), 0);
  std::vector<std::int64_t> remaining = prepared.weights;
  for (std::size_t index = 0; index < phase.effects.size(); ++index) {
    const auto& effect = prepared.effects[index];
    const auto event = index;
    pending[event] = static_cast<std::int64_t>(effect.arrival_increment) -
                     effect.arrival_weight;
    negative_pending[event] = -pending[event];
    const auto combined = CheckedAdd(remaining[event], pending[event]);
    if (!combined)
      return Output::Failure(
          SimpleError(PhaseBoundaryErrorCode::kTransactionArithmeticOverflow));
    remaining[event] = combined.value();
  }
  const IdealResult lowest = solver.Solve(solver_graph, pending);
  const IdealResult highest = solver.Solve(solver_graph, negative_pending);
  if (!lowest.ok() || !lowest.feasible)
    return Output::Failure(SolverError(phase.key(), lowest.status));
  if (!highest.ok() || !highest.feasible)
    return Output::Failure(SolverError(phase.key(), highest.status));
  const auto minimum_pending =
      CheckedAdd(static_cast<std::int64_t>(expected), lowest.minimum_weight);
  const auto maximum_pending = CheckedSubtract(
      static_cast<std::int64_t>(expected), highest.minimum_weight);
  if (!minimum_pending || !maximum_pending)
    return Output::Failure(
        SimpleError(PhaseBoundaryErrorCode::kArrivalArithmeticOverflow));
  if (minimum_pending.value() < 0)
    return idealError(PhaseBoundaryErrorCode::kNegativeArrivalPrefix, lowest);
  if (maximum_pending.value() > std::numeric_limits<std::uint32_t>::max()) {
    return idealError(PhaseBoundaryErrorCode::kArrivalPrefixOverflow, highest);
  }

  Certificate certificate{phase.key(),
                          prepared.arrival_total,
                          prepared.transaction_total,
                          tx.minimum_weight,
                          LiftIdeal(graph, phase, tx.ideal, retain_successful_ideals),
                          {}};
  certificate.successful_ideals_retained = retain_successful_ideals;
  certificate.arrival_increments =
      ArrivalIncrementBounds{prepared.arrival_increment_total,
                             minimum_pending.value(),
                             maximum_pending.value(),
                             {}};
  const auto checkRemaining =
      [&](std::optional<EventId> omitted) -> std::optional<PhaseBoundaryError> {
    IdealQuery query{DynamicBitset(prepared.weights.size()),
                     DynamicBitset(prepared.weights.size())};
    if (omitted) query.force_out.set(prepared.local_index.at(omitted->value()));
    const IdealResult result = solver.Solve(solver_graph, remaining, query);
    if (result.status == IdealStatus::kInfeasible) {
      certificate.arrival_increments->remaining_bounds.push_back(
          {omitted, false, 0,
           LiftIdeal(graph, phase, result.ideal, retain_successful_ideals)});
      return std::nullopt;
    }
    if (!result.ok() || !result.feasible)
      return SolverError(phase.key(), result.status);
    const auto minimum_remaining =
        CheckedAdd(static_cast<std::int64_t>(expected), result.minimum_weight);
    if (!minimum_remaining)
      return SimpleError(
          PhaseBoundaryErrorCode::kTransactionArithmeticOverflow);
    // Separate nonnegativity proofs make this single objective exact.
    if (minimum_remaining.value() <= 0) {
      auto failure = idealError(
          completed ? PhaseBoundaryErrorCode::kPrematurePhaseCompletion
                    : PhaseBoundaryErrorCode::kOpenPhaseNotStrictlyPositive,
          result, omitted);
      return failure.error();
    }
    certificate.arrival_increments->remaining_bounds.push_back(
        {omitted, true, minimum_remaining.value(),
         LiftIdeal(graph, phase, result.ideal, retain_successful_ideals)});
    return std::nullopt;
  };
  if (completed) {
    for (EventId omitted : phase.effects) {
      if (auto error = checkRemaining(omitted))
        return Output::Failure(std::move(*error));
    }
  } else if (auto error = checkRemaining(std::nullopt)) {
    return Output::Failure(std::move(*error));
  }
  return Output::Success(std::move(certificate));
}

template <typename Graph>
Result<PhaseCertificate, PhaseBoundaryError>
CheckCompletedPhase(const MinimumWeightIdealSolver& solver, const Graph& graph,
                    const PhaseDatabase& phases, PhaseKey key,
                    bool build_witness_schedule, bool reject_negative_tx,
                    bool retain_successful_ideals) {
  ProfileScope profile("phase.boundary");
  const PhaseInfo* phase = phases.find(key);
  if (phase == nullptr) {
    return Result<PhaseCertificate, PhaseBoundaryError>::Failure(
        SimpleError(PhaseBoundaryErrorCode::kUnknownPhase));
  }
  if (!phase->completed) {
    return Result<PhaseCertificate, PhaseBoundaryError>::Failure(
        SimpleError(PhaseBoundaryErrorCode::kWrongPhaseKind));
  }
  if (!graph.isAcyclic()) {
    return Result<PhaseCertificate, PhaseBoundaryError>::Failure(
        SimpleError(PhaseBoundaryErrorCode::kInvalidGraph));
  }

  auto prepared_result = Prepare(graph, *phase);
  if (!prepared_result) {
    return Result<PhaseCertificate, PhaseBoundaryError>::Failure(
        prepared_result.error());
  }
  PreparedPhase prepared = std::move(prepared_result).value();
  const auto expected = phases.expectedArrivals(key);
  if (!expected.has_value()) {
    return Result<PhaseCertificate, PhaseBoundaryError>::Failure(
        SimpleError(PhaseBoundaryErrorCode::kUnknownPhase));
  }

  if (prepared.arrival_increment_total != 0) {
    PrepareProjection(graph, *phase, prepared);
    return CheckIncrementPhase<PhaseCertificate>(solver, graph, *phase,
                                                 prepared, *expected, true,
                                                 build_witness_schedule,
                                                 retain_successful_ideals);
  }

  if (prepared.arrival_total != *expected) {
    return Result<PhaseCertificate, PhaseBoundaryError>::Failure(
        PhaseBoundaryError{
            PhaseBoundaryErrorCode::kArrivalBalanceMismatch,
            PhaseBalanceWitness{key, *expected, prepared.arrival_total,
                                prepared.transaction_total}});
  }
  if (prepared.transaction_total != 0) {
    return Result<PhaseCertificate, PhaseBoundaryError>::Failure(
        PhaseBoundaryError{
            PhaseBoundaryErrorCode::kTransactionBalanceMismatch,
            PhaseBalanceWitness{key, *expected, prepared.arrival_total,
                                prepared.transaction_total}});
  }

  if (prepared.nonarrival_effects.empty() &&
      std::all_of(prepared.weights.begin(), prepared.weights.end(),
                  [](auto weight) { return weight == 0; })) {
    ProfileScope arrival_profile("boundary.arrival_only");
    PhaseCertificate certificate{key, prepared.arrival_total, 0, 0, {}, {}};
    certificate.successful_ideals_retained = retain_successful_ideals;
    certificate.transaction_bound_domain = reject_negative_tx
        ? TransactionBoundDomain::kAllPrefixes
        : TransactionBoundDomain::kZeroPendingPrefixes;
    if (retain_successful_ideals) {
      DynamicBitset selected(phase->effects.size());
      if (!reject_negative_tx)
        for (std::size_t i = 0; i < phase->effects.size(); ++i) selected.set(i);
      certificate.minimum_prefix_ideal = LiftIdeal(graph, *phase, selected);
    } else {
      certificate.algebraic_bounds = AlgebraicPhaseBounds{0, {}};
    }
    return Result<PhaseCertificate, PhaseBoundaryError>::Success(std::move(certificate));
  }

  if (!retain_successful_ideals) {
    ProfileScope algebra_profile("boundary.algebraic");
    algebra_profile.nodes = phase->effects.size();
    auto bounds = TryBoundPhaseWithoutDependencies(
        phase->effects, prepared.effects, reject_negative_tx);
    if (bounds) {
      algebra_profile.pruned = 1;
      algebra_profile.edges = bounds->proper_prefix_lower_bounds.size();
      ProfileScope certified_profile(prepared.nonarrival_effects.empty()
          ? "boundary.algebraic_arrival_only"
          : "boundary.algebraic_transaction");
      certified_profile.nodes = phase->effects.size();
      // The relaxed bound is nonnegative, while the full phase is a feasible
      // ideal of weight zero. Thus the transaction minimum is exactly zero.
      // Omission bounds do NOT establish exact feasibility/minima; preserve
      // their distinct algebraic proof form instead of manufacturing ideals.
      PhaseCertificate certificate{key, prepared.arrival_total,
                                   prepared.transaction_total, 0, {}, {}};
      certificate.successful_ideals_retained = false;
      certificate.transaction_bound_domain = reject_negative_tx
          ? TransactionBoundDomain::kAllPrefixes
          : TransactionBoundDomain::kZeroPendingPrefixes;
      certificate.algebraic_bounds = std::move(bounds);
      return Result<PhaseCertificate, PhaseBoundaryError>::Success(
          std::move(certificate));
    }
  }

  PrepareProjection(graph, *phase, prepared);
  const IdealResult minimum =
      solver.Solve(prepared.projected, prepared.weights,
                          TransactionBoundQuery(prepared, *phase,
                                                reject_negative_tx));
  if (!minimum.ok() || !minimum.feasible) {
    return Result<PhaseCertificate, PhaseBoundaryError>::Failure(
        SolverError(key, minimum.status));
  }
  if (minimum.minimum_weight < 0) {
    auto witness = MakeIdealWitness(graph, *phase, prepared, minimum,
                                    std::nullopt, build_witness_schedule);
    if (!witness) {
      return Result<PhaseCertificate, PhaseBoundaryError>::Failure(
          witness.error());
    }
    return Result<PhaseCertificate, PhaseBoundaryError>::Failure(
        PhaseBoundaryError{PhaseBoundaryErrorCode::kNegativeTransactionPrefix,
                           std::move(witness).value()});
  }

  PhaseCertificate certificate{key,
                               prepared.arrival_total,
                               prepared.transaction_total,
                               minimum.minimum_weight,
                               LiftIdeal(graph, *phase, minimum.ideal, retain_successful_ideals),
                               {}};
  certificate.successful_ideals_retained = retain_successful_ideals;
  certificate.transaction_bound_domain = reject_negative_tx
      ? TransactionBoundDomain::kAllPrefixes
      : TransactionBoundDomain::kZeroPendingPrefixes;
  return CheckProperPrefixes(solver, graph, *phase,
                             prepared, std::move(certificate),
                             build_witness_schedule);
}

template <typename Graph>
Result<OpenPhaseCertificate, PhaseBoundaryError>
CheckOpenPhase(const MinimumWeightIdealSolver& solver, const Graph& graph,
               const PhaseDatabase& phases, PhaseKey key,
               bool build_witness_schedule, bool reject_negative_tx,
               bool retain_successful_ideals) {
  ProfileScope profile("phase.boundary");
  const PhaseInfo* phase = phases.find(key);
  if (phase == nullptr) {
    return Result<OpenPhaseCertificate, PhaseBoundaryError>::Failure(
        SimpleError(PhaseBoundaryErrorCode::kUnknownPhase));
  }
  if (phase->completed) {
    return Result<OpenPhaseCertificate, PhaseBoundaryError>::Failure(
        SimpleError(PhaseBoundaryErrorCode::kWrongPhaseKind));
  }
  if (!graph.isAcyclic()) {
    return Result<OpenPhaseCertificate, PhaseBoundaryError>::Failure(
        SimpleError(PhaseBoundaryErrorCode::kInvalidGraph));
  }

  auto prepared_result = Prepare(graph, *phase);
  if (!prepared_result) {
    return Result<OpenPhaseCertificate, PhaseBoundaryError>::Failure(
        prepared_result.error());
  }
  PreparedPhase prepared = std::move(prepared_result).value();
  const auto expected = phases.expectedArrivals(key);
  if (!expected.has_value()) {
    return Result<OpenPhaseCertificate, PhaseBoundaryError>::Failure(
        SimpleError(PhaseBoundaryErrorCode::kUnknownPhase));
  }

  if (prepared.arrival_increment_total != 0) {
    PrepareProjection(graph, *phase, prepared);
    return CheckIncrementPhase<OpenPhaseCertificate>(
        solver, graph, *phase, prepared, *expected, false,
        build_witness_schedule, retain_successful_ideals);
  }

  if (!reject_negative_tx && prepared.arrival_total < *expected) {
    OpenPhaseCertificate certificate{key, prepared.arrival_total,
                                     prepared.transaction_total, 0,
                                     retain_successful_ideals ? DynamicBitset(graph.eventCount())
                                                              : DynamicBitset{}, {}};
    certificate.successful_ideals_retained = retain_successful_ideals;
    certificate.transaction_bound_domain =
        TransactionBoundDomain::kNoZeroPendingPrefix;
    return Result<OpenPhaseCertificate, PhaseBoundaryError>::Success(
        std::move(certificate));
  }

  if (std::all_of(prepared.weights.begin(), prepared.weights.end(),
                  [](auto weight) { return weight == 0; })) {
    ProfileScope arrival_profile("boundary.open_arrival_only");
    if (prepared.arrival_total >= *expected)
      return Result<OpenPhaseCertificate, PhaseBoundaryError>::Failure(
          PhaseBoundaryError{
              prepared.arrival_total > *expected
                  ? PhaseBoundaryErrorCode::kOpenPhaseArrivalOverflow
                  : PhaseBoundaryErrorCode::kOpenPhaseNotStrictlyPositive,
              PhaseBalanceWitness{key, *expected, prepared.arrival_total, 0}});
    OpenPhaseCertificate certificate{key, prepared.arrival_total, 0, 0,
        retain_successful_ideals ? DynamicBitset(graph.eventCount()) : DynamicBitset{}, {}};
    certificate.successful_ideals_retained = retain_successful_ideals;
    return Result<OpenPhaseCertificate, PhaseBoundaryError>::Success(std::move(certificate));
  }

  PrepareProjection(graph, *phase, prepared);
  const IdealResult minimum =
      solver.Solve(prepared.projected, prepared.weights,
                          TransactionBoundQuery(prepared, *phase,
                                                reject_negative_tx));
  if (!minimum.ok() || !minimum.feasible) {
    return Result<OpenPhaseCertificate, PhaseBoundaryError>::Failure(
        SolverError(key, minimum.status));
  }
  if (minimum.minimum_weight < 0) {
    auto witness = MakeIdealWitness(graph, *phase, prepared, minimum,
                                    std::nullopt, build_witness_schedule);
    if (!witness) {
      return Result<OpenPhaseCertificate, PhaseBoundaryError>::Failure(
          witness.error());
    }
    return Result<OpenPhaseCertificate, PhaseBoundaryError>::Failure(
        PhaseBoundaryError{PhaseBoundaryErrorCode::kNegativeTransactionPrefix,
                           std::move(witness).value()});
  }

  if (prepared.arrival_total > *expected) {
    return Result<OpenPhaseCertificate, PhaseBoundaryError>::Failure(
        PhaseBoundaryError{
            PhaseBoundaryErrorCode::kOpenPhaseArrivalOverflow,
            PhaseBalanceWitness{key, *expected, prepared.arrival_total,
                                prepared.transaction_total}});
  }

  OpenPhaseCertificate certificate{key,
                                   prepared.arrival_total,
                                   prepared.transaction_total,
                                   minimum.minimum_weight,
                                   LiftIdeal(graph, *phase, minimum.ideal, retain_successful_ideals),
                                   {}};
  certificate.successful_ideals_retained = retain_successful_ideals;
  certificate.transaction_bound_domain = reject_negative_tx
      ? TransactionBoundDomain::kAllPrefixes
      : TransactionBoundDomain::kZeroPendingPrefixes;
  if (prepared.arrival_total < *expected) {
    return Result<OpenPhaseCertificate, PhaseBoundaryError>::Success(
        std::move(certificate));
  }

  if (prepared.transaction_total <= 0) {
    return Result<OpenPhaseCertificate, PhaseBoundaryError>::Failure(
        PhaseBoundaryError{
            PhaseBoundaryErrorCode::kOpenPhaseNotStrictlyPositive,
            PhaseBalanceWitness{key, *expected, prepared.arrival_total,
                                prepared.transaction_total}});
  }
  return CheckProperPrefixes(solver, graph, *phase,
                             prepared, std::move(certificate),
                             build_witness_schedule);
}

}  // namespace

Result<PhaseCertificate, PhaseBoundaryError>
PhaseBoundaryChecker::CheckCompletedPhase(
    const EventDag& graph, const PhaseDatabase& phases, PhaseKey key,
    bool build_witness_schedule, bool reject_negative_tx,
    bool retain_successful_ideals) const {
  return algorithm::CheckCompletedPhase(ideal_solver_, graph, phases, key,
                                        build_witness_schedule, reject_negative_tx,
                                        retain_successful_ideals);
}

Result<PhaseCertificate, PhaseBoundaryError>
PhaseBoundaryChecker::CheckCompletedPhase(
    const HappensBeforeGraph& graph, const PhaseDatabase& phases, PhaseKey key,
    bool build_witness_schedule, bool reject_negative_tx,
    bool retain_successful_ideals) const {
  return algorithm::CheckCompletedPhase(ideal_solver_, graph, phases, key,
                                        build_witness_schedule, reject_negative_tx,
                                        retain_successful_ideals);
}

Result<OpenPhaseCertificate, PhaseBoundaryError>
PhaseBoundaryChecker::CheckOpenPhase(
    const EventDag& graph, const PhaseDatabase& phases, PhaseKey key,
    bool build_witness_schedule, bool reject_negative_tx,
    bool retain_successful_ideals) const {
  return algorithm::CheckOpenPhase(ideal_solver_, graph, phases, key,
                                   build_witness_schedule, reject_negative_tx,
                                   retain_successful_ideals);
}

Result<OpenPhaseCertificate, PhaseBoundaryError>
PhaseBoundaryChecker::CheckOpenPhase(
    const HappensBeforeGraph& graph, const PhaseDatabase& phases, PhaseKey key,
    bool build_witness_schedule, bool reject_negative_tx,
    bool retain_successful_ideals) const {
  return algorithm::CheckOpenPhase(ideal_solver_, graph, phases, key,
                                   build_witness_schedule, reject_negative_tx,
                                   retain_successful_ideals);
}

}  // namespace sentryward::algorithm
