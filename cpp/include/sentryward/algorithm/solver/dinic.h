#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace sentryward::algorithm {

using FlowNode = std::uint32_t;
using FlowCapacity = std::uint64_t;

enum class FlowStatus {
  kOk,
  kInvalidNode,
  kSourceEqualsSink,
  kAlreadyComputed,
  kLimitReached,
  kCapacityOverflow,
};

struct MaxFlowResult {
  FlowStatus status = FlowStatus::kOk;
  FlowCapacity value = 0;

  [[nodiscard]] bool ok() const noexcept { return status == FlowStatus::kOk; }
};

class Dinic {
 public:
  explicit Dinic(std::size_t node_count);

  [[nodiscard]] std::size_t node_count() const noexcept;

  [[nodiscard]] FlowStatus AddEdge(FlowNode from,
                                   FlowNode to,
                                   FlowCapacity capacity);

  [[nodiscard]] MaxFlowResult Compute(
      FlowNode source,
      FlowNode sink,
      std::optional<FlowCapacity> stop_at = std::nullopt);

  [[nodiscard]] std::vector<std::uint8_t> ResidualReachable(
      FlowNode source) const;

 private:
  struct Edge {
    FlowNode to = 0;
    std::size_t reverse_index = 0;
    FlowCapacity capacity = 0;
  };

  [[nodiscard]] bool IsValidNode(FlowNode node) const noexcept;
  [[nodiscard]] bool BuildLevels(FlowNode source, FlowNode sink);
  [[nodiscard]] FlowCapacity SendFlow(FlowNode node,
                                      FlowNode sink,
                                      FlowCapacity limit);

  std::vector<std::vector<Edge>> adjacency_;
  std::vector<std::int64_t> level_;
  std::vector<std::size_t> next_edge_;
  bool computed_ = false;
  bool residual_overflow_ = false;
};

}  // namespace sentryward::algorithm
