#pragma once
#include "sentryward/equivalence/constant_values.h"
#include "sentryward/equivalence/integer_interval.h"
#include "sentryward/scalar_fold/graph.h"
#include <optional>
#include <string>

namespace sentryward::equivalence {
struct ConversionPrefix { std::string operation, target, remainder; };
// Parse one annotated edge, bounded to 128 bytes. Operation spelling is
// returned, not certified; each caller checks its own exact semantic subset.
std::optional<ConversionPrefix> leadingConversion(
    const std::string& source, const std::string& conversion);
struct BitcastPrefix { std::string target, remainder; };
std::optional<BitcastPrefix> leadingBitcast(
    const std::string& source, const std::string& conversion);
bool idealNumericConversion(const std::string& source, const std::string& target,
                            const std::string& conversion, bool storage_tail = false);
// Mathematical OUTPUT boundary only: floating quantization and saturation may
// be erased here. Internal numeric, predicate, address and bit uses must keep
// using idealNumericConversion or their exact conversion certificates.
// Checks a bounded, complete typed chain and an optional final storage bitcast.
bool idealNumericOutputConversion(const std::string& source, const std::string& target,
                                 const std::string& conversion);
// Every finite source value is exactly representable in the destination.
bool exactFloatingWidening(const std::string& source, const std::string& target);

// An explicit same-width storage Bitcast on an unchanged memory cell is a
// typed floating read, not a numerical reinterpret of a computed producer.
// This local certificate preserves the remaining use annotations verbatim;
// their arithmetic eligibility is checked separately below. Metadata is
// bounded and the address is inspected only as one typed edge, never walked.
std::optional<BitcastPrefix> floatingStorageReadPrefix(
    const scalar_fold::Graph&, const scalar_fold::Ref&);
bool idealFloatingStorageReadUse(
    const scalar_fold::Graph&, const scalar_fold::Ref&, bool storage_tail = false);

// One typed integer edge preserves its mathematical value only when the
// supplied interval fits the destination. Unsupported modifiers, wrapping,
// raw noninteger types and width-changing bitcasts remain Unknown. No graph
// traversal or conversion-chain expansion is performed.
integer_interval::Result integerConversionRange(
    const std::string& source, const std::string& target,
    const std::string& conversion, const integer_interval::Interval& value);

// Fold an integer singleton only if the existing exact encoder can represent
// it in f32/f64 without rounding. Bare Convert is NOT defined to mean RNE here;
// exact representability makes its admitted ordinary result mode-independent.
// Directed/unknown modes and small/quantized floating formats stay explicit.
std::optional<ConstantValue> exactIntegerFloatingValue(
    const std::string& source, const std::string& target,
    const std::string& conversion, std::int64_t value);

struct FloatingClampRange { double lower, upper; };
// Complete two-sided f32 min_num/max_num clamp, with two exact finite f32
// literals and no modifiers/provenance on either clamp operation. The middle
// operand is not inspected: NaN and infinities are handled by the default
// non-propagating min/max semantics. At most two operation heads/four operand
// headers are examined, independently of any Fold count or lexical domain.
std::optional<FloatingClampRange> exactFloatingClampRange(
    const scalar_fold::Graph&, const scalar_fold::Ref&);

// Remove ONLY a redundant first RN+saturate_finite f32->e4m3 modifier after
// the exact source clamp proves its complete range lies in [-448,448]. RN,
// producer, remaining conversion tail, target type and provenance are retained.
// Unsupported modes/qualifiers and unbounded/one-sided clamps stay explicit;
// this is value/bit equality, not an ideal quantization alias.
std::optional<scalar_fold::Ref> normalizeInactiveFiniteSaturation(
    const scalar_fold::Graph&, const scalar_fold::Ref&);

// Canonicalizes exact bit-preserving chains; unsupported steps remain explicit.
std::optional<std::string> normalizeConversions(
    const std::string& source, const std::string& target, const std::string& conversion);
}
