#pragma once

#include "sentryward/value_ir/graph.h"

namespace sentryward::frontend::ptx {

// Returns an exact local rewrite; leaves the source graph and occurrences intact.
value_ir::Value NormalizeAddressExpression(const value_ir::Graph& graph,
    value_ir::Value value,
    const std::function<value_ir::Value(value_ir::Value)>& resolve = {});

}  // namespace sentryward::frontend::ptx
