#pragma once

#include <algorithm>
#include "sentryward/value_ir/ptx_operation.h"

namespace sentryward::value_analysis {

// The existing ideal-real abstraction must be identical at source matching
// and scalar projection. Classify the operation, without rebuilding its graph
// or enumerating its logical instances. Unknown modifiers stay exact/opaque.
inline bool isIdealExp2(const value_ir::Operation& operation) {
  namespace v = value_ir;
  if (dynamic_cast<const v::IdealExp2Op*>(&operation)) return true;
  const auto* ptx = dynamic_cast<const v::PtxScalarOp*>(&operation);
  if (!ptx || ptx->opcode() != v::PtxScalarOpcode::Exp2 ||
      ptx->resultTypes().size() != 1) return false;
  const auto& type = ptx->resultTypes()[0];
  if (!type.isScalar() || type.element().isParameterized() ||
      type.element().kind() != v::ElementKind::Floating || type.bitSize() != 32 ||
      ptx->sourceType() != type) return false;
  const auto& modifiers = ptx->attributes().modifiers;
  // Already used by scalar/Fold, not a new approximation contract.
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-ex2
  return std::find(modifiers.begin(), modifiers.end(), "approx") != modifiers.end() &&
      std::all_of(modifiers.begin(), modifiers.end(), [](const auto& modifier) {
        return modifier == "approx" || modifier == "ftz" || modifier == "f32";
      });
}

}  // namespace sentryward::value_analysis
