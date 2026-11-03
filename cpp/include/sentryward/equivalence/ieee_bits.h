#pragma once
#include <cstdint>
#include <stdexcept>
namespace sentryward::equivalence {
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
