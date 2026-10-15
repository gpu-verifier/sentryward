#pragma once

#include "sentryward/scalar_fold/artifact.h"
#include <chrono>
#include <optional>

namespace sentryward::equivalence {

// One complete software/direct natural-exp implementation, not a rewrite of a
// rounded decimal coefficient. Arguments remain opaque local DAG bindings.
struct NaturalExpIntrinsicMatch {
  scalar_fold::Ref argument;
  std::optional<scalar_fold::Ref> accumulator;
  bool negate_argument = false;
  bool fast_direct = false;
  // Only the complete direct Exp2 consumer admits these bounded extensions.
  // The second natural argument and the base-two residual remain opaque
  // bindings; their signs are independent of the first argument's sign.
  std::optional<scalar_fold::Ref> second_argument = std::nullopt;
  bool negate_second_argument = false;
  std::optional<scalar_fold::Ref> base_two_residual = std::nullopt;
  bool negate_base_two_residual = false;
  // Complete three-factor direct form Exp2(x*(RN(log2(e))*y)).
  // Both numerical arguments remain opaque local bindings.
  std::optional<scalar_fold::Ref> argument_factor = std::nullopt;
};

// Inspects at most min(max_inspected_nodes,128) distinct local nodes. It never
// traverses an argument's ancestry, Fold members, or output-family elements.
std::optional<NaturalExpIntrinsicMatch> matchNaturalExpIntrinsic(
    const scalar_fold::Graph&, scalar_fold::Ref,
    std::size_t max_inspected_nodes = 128, std::size_t* inspected_nodes = nullptr);

// Complete direct fast natural-exp consumer: f32 Exp2(Multiply(+/- the
// exact f32 RN log2(e) constant, x)), with admitted RN/lowered multiply mode,
// or its bounded additive wrapper. The wrapper inspects at most three bare
// Add/Subtract/Negate spine nodes, with at most two signed scaled bindings and
// one unscaled base-two residual. It becomes Exp(signed natural arguments)
// times Exp2(signed residual); a residual Log2 retains its original domain.
// A direct product with an unmatched literal coefficient is not a residual.
// This is an explicitly admitted approximation, not coefficient equality or
// pointwise floating equivalence. The argument remains an opaque DAG binding.
std::optional<NaturalExpIntrinsicMatch> matchFastNaturalExpIntrinsic(
    const scalar_fold::Graph&, scalar_fold::Ref,
    std::size_t max_inspected_nodes = 128, std::size_t* inspected_nodes = nullptr);

// Complete f32 hardware natural-log consumer: RN_f32(ln2)*PtxLog2(x),
// optionally RN_f32(ln2)*(PtxLog2(x)+offset). The latter retains the original
// coefficient*offset term. Only the positive exact bits 0x3f317218 and the
// shared typed approximate lg2 certificate are admitted, never canonical
// Log2 times an arbitrary decimal. Known nonpositive/nonfinite x are rejected;
// symbolic x is preserved without assuming it positive or finite. Property
// analysis must prove the Log domain before using finite/nonzero identities.
struct NaturalLogIntrinsicMatch {
  scalar_fold::Ref argument,coefficient;
  std::optional<scalar_fold::Ref> offset;
};
std::optional<NaturalLogIntrinsicMatch> matchFastNaturalLogIntrinsic(
    const scalar_fold::Graph&, scalar_fold::Ref,
    std::size_t max_inspected_nodes = 128, std::size_t* inspected_nodes = nullptr);

// Complete f32 NVIDIA-style software Log2 macro: subnormal rescaling,
// exact integer exponent/mantissa extraction, all eleven fixed coefficient
// bits and both zero/nonfinite exception selectors. This is an explicitly
// admitted ideal-real approximation, not a polynomial/coefficient identity.
// Known nonpositive/nonfinite inputs are excluded from the ideal-real domain;
// symbolic arguments remain opaque and are not assumed positive or finite.
struct SoftwareLog2IntrinsicMatch { scalar_fold::Ref argument; };
std::optional<SoftwareLog2IntrinsicMatch> matchSoftwareLog2Intrinsic(
    const scalar_fold::Graph&, scalar_fold::Ref,
    std::size_t max_inspected_nodes = 128, std::size_t* inspected_nodes = nullptr);

struct MathIntrinsicStatistics {
  std::size_t match_attempts = 0, inspected_nodes = 0, max_match_nodes = 0;
  std::size_t transformed_nodes = 0, natural_exp_nodes = 0;
  std::size_t natural_exp_accumulators = 0;
  std::size_t fast_natural_exp_nodes = 0;
  std::size_t natural_log_nodes = 0, natural_log_offsets = 0;
  std::size_t software_log2_nodes = 0;
};

// Invoke only under the explicitly authorized ideal-real mathematical contract.
// Admits COMPLETE software/direct fast natural-exp implementations as ideal
// Exp, complete hardware natural-log wrappers as ideal Log and complete
// software base-two-log wrappers as ideal Log2. This omits
// precision/FTZ/overflow/underflow effects, not typed semantics.
// Unknown/bit-observing uses remain exact. Work is O(stored DAG nodes + operand
// edges), with constant-bounded local recognition and at most two memoized
// versions (numeric/exact) of a shared node. No logical instances are expanded.
// A no-match consuming path returns the original artifact without rebuilding.
scalar_fold::Artifact abstractMathIntrinsics(
    scalar_fold::Artifact&&, std::size_t max_nodes,
    std::chrono::steady_clock::time_point deadline,
    MathIntrinsicStatistics* statistics = nullptr);

}  // namespace sentryward::equivalence
