#pragma once
#include <cstddef>
#include <optional>
#include <string_view>

namespace sentryward::equivalence {
// Shared operation/arity vocabulary for ordinary ideal-real scalar primitives.
// Callers independently certify node kind, value type, attributes and uses.
bool idealScalarOperation(std::string_view operation,std::size_t arity);

// Typed PTX primitive modifier certificate, not a software-math idiom
// recognizer. Approximate Sin/Cos/Log2 require f32.approx[.ftz][.f32]. This is
// an opt-in ideal-real vocabulary, not a floating-point error certificate;
// callers require an explicit ideal numerical use, an Operation node, one
// child, and no pointer provenance. Exact/bit/predicate uses remain exact.
std::optional<std::string_view> legacyIdealPrimitive(
    std::string_view operation,std::string_view type,
    std::string_view attributes,std::string_view operand_type);

// Proves only finiteness of a preserved conversion use, never equality to
// its integer input. All s/u8/16/32/64 values convert finitely to f32/f64
// under every ordinary IEEE rounding mode. Optional width-preserving integer
// or raw-bit Bitcast prefixes retain this fact. Unknown/saturating steps,
// floating bitcasts, non-f32/f64 targets and malformed chains are rejected.
// Scans at most 128 conversion steps, without graph traversal/allocation.
bool finiteIntegerFloatConversion(std::string_view source,
    std::string_view target,std::string_view conversion);
}
