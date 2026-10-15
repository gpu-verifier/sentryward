#include "sentryward/algorithm/diagnostics/performance_profile.h"
#include "sentryward/algorithm/solver/minimum_weight_ideal.h"

#include <algorithm>
#include <limits>
#include <utility>

#include "sentryward/algorithm/solver/dinic.h"

namespace sentryward::algorithm {
namespace {

[[nodiscard]] IdealResult ResultWithStatus(IdealStatus status,
                                           std::size_t vertex_count) {
  return IdealResult{status, false, 0, DynamicBitset(vertex_count)};
}

[[nodiscard]] bool HasExpectedSize(const DynamicBitset& bits,
                                   std::size_t vertex_count) {
  return bits.empty() || bits.size() == vertex_count;
}

[[nodiscard]] bool IsSet(const DynamicBitset& bits, std::size_t index) {
  return !bits.empty() && bits.test(index);
}

[[nodiscard]] std::uint64_t Magnitude(std::int64_t value) {
  if (value >= 0) {
    return static_cast<std::uint64_t>(value);
  }
  return std::uint64_t{0} - static_cast<std::uint64_t>(value);
}

[[nodiscard]] bool CheckedAdd(std::uint64_t lhs,
                              std::uint64_t rhs,
                              std::uint64_t* result) {
  if (lhs > std::numeric_limits<std::uint64_t>::max() - rhs) {
    return false;
  }
  *result = lhs + rhs;
  return true;
}

struct Predecessors {
  std::vector<std::size_t> offsets;
  std::vector<IdealVertex> vertices;

  [[nodiscard]] std::span<const IdealVertex> Of(IdealVertex vertex) const {
    return std::span<const IdealVertex>(vertices).subspan(
        offsets[vertex], offsets[vertex + 1] - offsets[vertex]);
  }
};

[[nodiscard]] bool BuildDag(const DirectedGraph& graph,
                            Predecessors* predecessors) {
  const std::size_t vertex_count = graph.vertex_count;
  predecessors->offsets.assign(vertex_count + 1, 0);
  std::vector<std::size_t> outdegree(vertex_count, 0);
  for (const DirectedEdge& edge : graph.edges) {
    if (edge.from >= vertex_count || edge.to >= vertex_count) {
      return false;
    }
    ++predecessors->offsets[edge.to + 1];
    ++outdegree[edge.from];
  }
  for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
    predecessors->offsets[vertex + 1] += predecessors->offsets[vertex];
  }
  predecessors->vertices.resize(graph.edges.size());
  {
    std::vector<std::size_t> next = predecessors->offsets;
    for (const DirectedEdge& edge : graph.edges) {
      predecessors->vertices[next[edge.to]++] = edge.from;
    }
  }

  std::vector<IdealVertex> ready;
  for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
    if (outdegree[vertex] == 0) {
      ready.push_back(static_cast<IdealVertex>(vertex));
    }
  }
  for (std::size_t index = 0; index < ready.size(); ++index) {
    for (const IdealVertex predecessor : predecessors->Of(ready[index])) {
      if (--outdegree[predecessor] == 0) {
        ready.push_back(predecessor);
      }
    }
  }
  return ready.size() == vertex_count;
}

}  // namespace

IdealResult MinimumWeightIdealSolver::Solve(
    const DirectedGraph& graph,
    std::span<const std::int64_t> weights,
    const IdealQuery& query) const {
  ProfileScope profile("ideal.total");
  const std::size_t vertex_count = graph.vertex_count;
  constexpr std::size_t kMaxVertices =
      static_cast<std::size_t>(std::numeric_limits<FlowNode>::max()) - 2;
  if (vertex_count > kMaxVertices || weights.size() != vertex_count ||
      !HasExpectedSize(query.force_in, vertex_count) ||
      !HasExpectedSize(query.force_out, vertex_count)) {
    return ResultWithStatus(IdealStatus::kInvalidInput, vertex_count);
  }

  Predecessors predecessors;
  if (!ProfileCall("ideal.build_dag", [&] { return BuildDag(graph, &predecessors); })) {
    return ResultWithStatus(IdealStatus::kInvalidInput, vertex_count);
  }

  if (std::all_of(weights.begin(), weights.end(), [](auto weight) { return weight == 0; })) {
    ProfileScope zero_profile("ideal.zero_weight_closure");
    DynamicBitset ideal(vertex_count);
    std::vector<IdealVertex> pending;
    for (std::size_t vertex = 0; vertex < vertex_count; ++vertex)
      if (IsSet(query.force_in, vertex)) {
        ideal.set(vertex);
        pending.push_back(static_cast<IdealVertex>(vertex));
      }
    for (std::size_t i = 0; i < pending.size(); ++i) {
      const auto vertex = pending[i];
      if (IsSet(query.force_out, vertex))
        return ResultWithStatus(IdealStatus::kInfeasible, vertex_count);
      for (auto predecessor : predecessors.Of(vertex)) {
        if (!ideal.test(predecessor)) {
          ideal.set(predecessor);
          pending.push_back(predecessor);
        }
        ++zero_profile.edges;
      }
      ++zero_profile.nodes;
    }
    return IdealResult{IdealStatus::kOk, true, 0, std::move(ideal)};
  }

  ProfileScope network_profile("ideal.build_network");
  std::uint64_t absolute_sum = 0;
  std::uint64_t negative_sum = 0;
  for (const std::int64_t weight : weights) {
    const std::uint64_t magnitude = Magnitude(weight);
    if (!CheckedAdd(absolute_sum, magnitude, &absolute_sum)) {
      return ResultWithStatus(IdealStatus::kArithmeticOverflow,
                              vertex_count);
    }
    if (weight < 0 && !CheckedAdd(negative_sum, magnitude, &negative_sum)) {
      return ResultWithStatus(IdealStatus::kArithmeticOverflow,
                              vertex_count);
    }
  }
  if (absolute_sum == std::numeric_limits<std::uint64_t>::max()) {
    return ResultWithStatus(IdealStatus::kArithmeticOverflow, vertex_count);
  }
  const std::uint64_t infinity = absolute_sum + 1;

  constexpr FlowNode kRemoved = std::numeric_limits<FlowNode>::max();
  std::vector<FlowNode> projected(vertex_count, kRemoved);
  std::vector<IdealVertex> retained;
  for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
    if (weights[vertex] != 0 || IsSet(query.force_in, vertex) ||
        IsSet(query.force_out, vertex)) {
      projected[vertex] = static_cast<FlowNode>(retained.size());
      retained.push_back(static_cast<IdealVertex>(vertex));
    }
  }

  const FlowNode source = static_cast<FlowNode>(retained.size());
  const FlowNode sink = source + 1;
  Dinic flow(retained.size() + 2);

  for (const IdealVertex vertex : retained) {
    FlowStatus status = FlowStatus::kOk;
    if (weights[vertex] < 0) {
      status = flow.AddEdge(source, projected[vertex],
                            Magnitude(weights[vertex]));
    } else if (weights[vertex] > 0) {
      status = flow.AddEdge(projected[vertex], sink,
                            Magnitude(weights[vertex]));
    }
    if (status != FlowStatus::kOk) {
      return ResultWithStatus(IdealStatus::kFlowFailure, vertex_count);
    }
  }

  std::vector<FlowNode> seen(vertex_count, kRemoved);
  std::vector<IdealVertex> pending;
  for (const IdealVertex vertex : retained) {
    const FlowNode target = projected[vertex];
    pending.push_back(vertex);
    while (!pending.empty()) {
      const IdealVertex current = pending.back();
      pending.pop_back();
      for (const IdealVertex predecessor : predecessors.Of(current)) {
        if (seen[predecessor] == target) {
          continue;
        }
        seen[predecessor] = target;
        if (projected[predecessor] == kRemoved) {
          pending.push_back(predecessor);
        } else if (flow.AddEdge(target, projected[predecessor], infinity) !=
                   FlowStatus::kOk) {
          return ResultWithStatus(IdealStatus::kFlowFailure, vertex_count);
        }
      }
    }
  }

  for (const IdealVertex vertex : retained) {
    if (IsSet(query.force_in, vertex) &&
        flow.AddEdge(source, projected[vertex], infinity) !=
            FlowStatus::kOk) {
      return ResultWithStatus(IdealStatus::kFlowFailure, vertex_count);
    }
    if (IsSet(query.force_out, vertex) &&
        flow.AddEdge(projected[vertex], sink, infinity) !=
            FlowStatus::kOk) {
      return ResultWithStatus(IdealStatus::kFlowFailure, vertex_count);
    }
  }

  network_profile.Stop();
  const MaxFlowResult max_flow = flow.Compute(source, sink, infinity);
  if (max_flow.status == FlowStatus::kLimitReached) {
    return ResultWithStatus(IdealStatus::kInfeasible, vertex_count);
  }
  if (!max_flow.ok()) {
    return ResultWithStatus(
        max_flow.status == FlowStatus::kCapacityOverflow
            ? IdealStatus::kArithmeticOverflow
            : IdealStatus::kFlowFailure,
        vertex_count);
  }

  ProfileScope recover_profile("ideal.recover");
  std::int64_t signed_weight = 0;
  if (max_flow.value >= negative_sum) {
    const std::uint64_t positive = max_flow.value - negative_sum;
    if (positive >
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      return ResultWithStatus(IdealStatus::kArithmeticOverflow, vertex_count);
    }
    signed_weight = static_cast<std::int64_t>(positive);
  } else {
    const std::uint64_t magnitude = negative_sum - max_flow.value;
    constexpr std::uint64_t kMinMagnitude =
        std::uint64_t{1} << std::numeric_limits<std::int64_t>::digits;
    if (magnitude > kMinMagnitude) {
      return ResultWithStatus(IdealStatus::kArithmeticOverflow, vertex_count);
    }
    signed_weight = magnitude == kMinMagnitude
                        ? std::numeric_limits<std::int64_t>::min()
                        : -static_cast<std::int64_t>(magnitude);
  }

  DynamicBitset ideal(vertex_count);
  const std::vector<std::uint8_t> source_side =
      flow.ResidualReachable(source);
  for (const IdealVertex vertex : retained) {
    if (source_side[projected[vertex]] != 0) {
      ideal.set(vertex);
      pending.push_back(vertex);
    }
  }
  while (!pending.empty()) {
    const IdealVertex current = pending.back();
    pending.pop_back();
    for (const IdealVertex predecessor : predecessors.Of(current)) {
      if (!ideal.test(predecessor)) {
        ideal.set(predecessor);
        pending.push_back(predecessor);
      }
    }
  }

  return IdealResult{IdealStatus::kOk, true, signed_weight, std::move(ideal)};
}

}  // namespace sentryward::algorithm
