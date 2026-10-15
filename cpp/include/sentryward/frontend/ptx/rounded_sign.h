#pragma once
#include "sentryward/value_ir/graph.h"
#include <array>
#include <optional>
#include <string>

namespace sentryward::frontend::ptx {
// A complete ORIGINAL typed predicate shape, not an unconditional sign fact.
// Only a trusted captured true observation plus finite Numeric counterparts
// for ALL three operands permits the one-direction implication below. The
// matched a/b/c must be exact encoded Constant/Parameter/readonly Load values,
// not a newly idealized computed floating subtree.
struct RoundedPositiveSubAddShape {
  value_ir::Value predicate, addition, subtraction;
  std::array<value_ir::Value,3> operands; // a,b,c in Round(Round(a-b)+c)>0.
  std::string format;
};
std::optional<RoundedPositiveSubAddShape> roundedPositiveSubAddShape(
    const value_ir::Graph&,value_ir::Value);

// With a,b,c finite in the SAME IEEE format and monotone roundings fixing
// representables, Round2(Round1(a-b)+c)>0 implies a-b+c>0. c is represented
// in that format, not an approximate coefficient or arbitrary ideal real.
// Contrapositive: a-b+c<=0 => a-b<=-c => Round1(a-b)<=-c
// => Round2(Round1(a-b)+c)<=Round2(0)=0. Overflow is harmless to this
// implication; NaN, FTZ, saturation, mixed formats and the reverse direction
// are NOT admitted. This API does not establish operand finiteness, map memory,
// infer a branch truth, widen a member domain or authorize an exact union.
// PTX add/sub's default IEEE mode is independently specified as RN here:
// https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-add
// https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-sub
} // namespace sentryward::frontend::ptx
