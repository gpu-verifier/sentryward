#pragma once
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string_view>
namespace sentryward::equivalence {
// The IEEE binary formats of the floating types f16, bf16, f32 and f64: `width` bits, of which
// `fraction` are fraction bits and width - 1 - fraction exponent bits. The one table of these
// parameters for constant evaluation (constant_values.cpp) and the equivalence import's value sets.
struct IeeeFormat { unsigned width, fraction; };
inline std::optional<IeeeFormat> ieeeFormat(std::string_view type) {
  if (type == "f16") return IeeeFormat{16, 10};
  if (type == "bf16") return IeeeFormat{16, 7};
  if (type == "f32") return IeeeFormat{32, 23};
  if (type == "f64") return IeeeFormat{64, 52};
  return std::nullopt;
}
// An exact dyadic value (-1)^negative x significand x 2^exponent.
struct DyadicValue { bool negative; std::uint64_t significand; int exponent; };
// The value of IEEE-style bits: `width` bits, `fraction` fraction bits and E = width - 1 - fraction
// exponent bits of bias 2^(E-1) - 1, subnormal when the exponent field is zero. An all-ones exponent
// field is read as a finite exponent: callers exclude infinities and NaNs. Not normalized.
// Requires width <= 64 and 1 <= E <= 30 (every shift and the exponent stay in range); other
// parameters throw std::invalid_argument. O(1).
inline DyadicValue ieeeDyadicValue(std::uint64_t bits, unsigned width, unsigned fraction) {
  if (width > 64 || fraction + 1 >= width || width - 1 - fraction > 30)
    throw std::invalid_argument("IEEE-style format outside 1..30 exponent bits within 64 bits");
  const unsigned exponent_bits = width - 1 - fraction;
  const auto field = (bits >> fraction) & ((std::uint64_t(1) << exponent_bits) - 1);
  const int bias = (1 << (exponent_bits - 1)) - 1;
  auto significand = bits & ((std::uint64_t(1) << fraction) - 1);
  if (field) significand |= std::uint64_t(1) << fraction;
  return {bool((bits >> (width - 1)) & 1), significand, (field ? int(field) - bias : 1 - bias) - int(fraction)};
}
}  // namespace sentryward::equivalence
