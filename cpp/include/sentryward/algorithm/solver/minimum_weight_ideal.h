#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "sentryward/algorithm/base/dynamic_bitset.h"

namespace sentryward::algorithm {

using IdealVertex = std::uint32_t;

struct DirectedEdge {
  IdealVertex from = 0;
  IdealVertex to = 0;

  friend bool operator==(const DirectedEdge&, const DirectedEdge&) = default;
};

struct DirectedGraph {
  std::size_t vertex_count = 0;
  std::vector<DirectedEdge> edges;
};

struct IdealQuery {
  DynamicBitset force_in;
  DynamicBitset force_out;
};

enum class IdealStatus {
  kOk,
  kInfeasible,
  kInvalidInput,
  kArithmeticOverflow,
  kFlowFailure,
};

struct IdealResult {
  IdealStatus status = IdealStatus::kInvalidInput;
  bool feasible = false;
  std::int64_t minimum_weight = 0;
  DynamicBitset ideal;

  [[nodiscard]] bool ok() const noexcept { return status == IdealStatus::kOk; }
};

class MinimumWeightIdealSolver {
 public:
  [[nodiscard]] IdealResult Solve(
      const DirectedGraph& graph,
      std::span<const std::int64_t> weights,
      const IdealQuery& query = {}) const;
};

}  // namespace sentryward::algorithm
