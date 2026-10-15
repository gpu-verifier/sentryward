#pragma once

#include <cstddef>
#include <functional>
#include <string_view>
#include "sentryward/value_analysis/memory_elimination.h"
#include "sentryward/value_analysis/view_normalization.h"

namespace sentryward::value_analysis {

struct ValuePassStatistics final {
  std::size_t nodes = 0;
  std::size_t instances = 0;
};

// Exact fixed-width packing rewrite: disjoint bitwise OR becomes addition.
ValuePassStatistics rewritePacking(value_ir::Graph& graph);
// Whole software exponent construction becomes ideal Exp2, including its clamp.
ValuePassStatistics abstractSoftwareExp2(value_ir::Graph& graph);

struct ValuePipelineResult final {
  ValuePassStatistics packing;
  ValuePassStatistics ideal_exp2;
  value_ir::TypeEdgeStatistics type_edges;
  MemoryEliminationResult memory;
  ViewNormalizationStatistics views;
};

// Requires completed checks or an explicit trusted-reference order contract.
ValuePipelineResult runValuePasses(value_ir::Graph& graph,
    MemoryActionRange actions, const CheckedOrderContract& contract,
    const std::function<void(std::string_view)>& before_pass = {});

}  // namespace sentryward::value_analysis
