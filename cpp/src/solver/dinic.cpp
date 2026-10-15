#include "sentryward/algorithm/diagnostics/performance_profile.h"
#include "sentryward/algorithm/solver/dinic.h"

#include <algorithm>
#include <limits>
#include <queue>
#include <stdexcept>

namespace sentryward::algorithm {

Dinic::Dinic(std::size_t node_count) {
  if (node_count > static_cast<std::size_t>(
                       std::numeric_limits<FlowNode>::max())) {
    throw std::length_error("flow network has too many nodes");
  }
  adjacency_.resize(node_count);
  level_.resize(node_count);
  next_edge_.resize(node_count);
}

std::size_t Dinic::node_count() const noexcept { return adjacency_.size(); }

FlowStatus Dinic::AddEdge(FlowNode from,
                          FlowNode to,
                          FlowCapacity capacity) {
  if (computed_) {
    return FlowStatus::kAlreadyComputed;
  }
  if (!IsValidNode(from) || !IsValidNode(to)) {
    return FlowStatus::kInvalidNode;
  }

  const std::size_t forward_index = adjacency_[from].size();
  const std::size_t reverse_index =
      adjacency_[to].size() + (from == to ? 1 : 0);
  adjacency_[from].push_back(Edge{to, reverse_index, capacity});
  adjacency_[to].push_back(Edge{from, forward_index, 0});
  return FlowStatus::kOk;
}

MaxFlowResult Dinic::Compute(FlowNode source,
                             FlowNode sink,
                             std::optional<FlowCapacity> stop_at) {
  ProfileScope profile("ideal.maxflow");
  if (computed_) {
    return {FlowStatus::kAlreadyComputed, 0};
  }
  if (!IsValidNode(source) || !IsValidNode(sink)) {
    return {FlowStatus::kInvalidNode, 0};
  }
  if (source == sink) {
    return {FlowStatus::kSourceEqualsSink, 0};
  }
  computed_ = true;

  if (stop_at.has_value() && *stop_at == 0) {
    return {FlowStatus::kLimitReached, 0};
  }

  FlowCapacity flow = 0;
  constexpr FlowCapacity kCapacityMax =
      std::numeric_limits<FlowCapacity>::max();

  while (BuildLevels(source, sink)) {
    if (stop_at.has_value() && flow == *stop_at) {
      return {FlowStatus::kLimitReached, flow};
    }
    if (!stop_at.has_value() && flow == kCapacityMax) {
      return {FlowStatus::kCapacityOverflow, flow};
    }

    std::fill(next_edge_.begin(), next_edge_.end(), 0);
    while (true) {
      const FlowCapacity limit = stop_at.has_value()
                                     ? *stop_at - flow
                                     : kCapacityMax - flow;
      if (limit == 0) {
        if (stop_at.has_value()) {
          return {FlowStatus::kLimitReached, flow};
        }
        break;
      }

      const FlowCapacity pushed = SendFlow(source, sink, limit);
      if (residual_overflow_) {
        return {FlowStatus::kCapacityOverflow, flow};
      }
      if (pushed == 0) {
        break;
      }
      flow += pushed;

      if (stop_at.has_value() && flow == *stop_at) {
        return {FlowStatus::kLimitReached, flow};
      }
      if (!stop_at.has_value() && flow == kCapacityMax) {
        break;
      }
    }
  }

  return {FlowStatus::kOk, flow};
}

std::vector<std::uint8_t> Dinic::ResidualReachable(FlowNode source) const {
  std::vector<std::uint8_t> reachable(adjacency_.size(), 0);
  if (!IsValidNode(source)) {
    return reachable;
  }

  std::queue<FlowNode> queue;
  reachable[source] = 1;
  queue.push(source);
  while (!queue.empty()) {
    const FlowNode node = queue.front();
    queue.pop();
    for (const Edge& edge : adjacency_[node]) {
      if (edge.capacity == 0 || reachable[edge.to] != 0) {
        continue;
      }
      reachable[edge.to] = 1;
      queue.push(edge.to);
    }
  }
  return reachable;
}

bool Dinic::IsValidNode(FlowNode node) const noexcept {
  return static_cast<std::size_t>(node) < adjacency_.size();
}

bool Dinic::BuildLevels(FlowNode source, FlowNode sink) {
  std::fill(level_.begin(), level_.end(), -1);
  std::queue<FlowNode> queue;
  level_[source] = 0;
  queue.push(source);

  while (!queue.empty()) {
    const FlowNode node = queue.front();
    queue.pop();
    for (const Edge& edge : adjacency_[node]) {
      if (edge.capacity == 0 || level_[edge.to] != -1) {
        continue;
      }
      level_[edge.to] = level_[node] + 1;
      queue.push(edge.to);
    }
  }
  return level_[sink] != -1;
}

FlowCapacity Dinic::SendFlow(FlowNode node,
                             FlowNode sink,
                             FlowCapacity limit) {
  if (node == sink) {
    return limit;
  }

  for (std::size_t& index = next_edge_[node];
       index < adjacency_[node].size(); ++index) {
    Edge& edge = adjacency_[node][index];
    if (edge.capacity == 0 || level_[edge.to] != level_[node] + 1) {
      continue;
    }

    const FlowCapacity pushed =
        SendFlow(edge.to, sink, std::min(limit, edge.capacity));
    if (pushed == 0 || residual_overflow_) {
      continue;
    }

    Edge& reverse = adjacency_[edge.to][edge.reverse_index];
    if (reverse.capacity >
        std::numeric_limits<FlowCapacity>::max() - pushed) {
      residual_overflow_ = true;
      return 0;
    }
    edge.capacity -= pushed;
    reverse.capacity += pushed;
    return pushed;
  }
  return 0;
}

}  // namespace sentryward::algorithm
