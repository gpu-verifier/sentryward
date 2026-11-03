#pragma once
#include <optional>
#include <span>
#include <string>
namespace sentryward::equivalence {
struct ConstantValue { std::string type, literal; };
// Exact supported conversions; unknown qualifiers and inexact conversions remain symbolic.
std::optional<ConstantValue> constantValue(
    std::string type, std::string literal, std::string conversion = {},
    std::string converted_type = {});
std::optional<ConstantValue> constantOperation(
    const std::string& name, const std::string& type, const std::string& attributes,
    std::span<const ConstantValue> operands);
// IEEE-754 binary32/binary64 Add, Subtract, Multiply, Divide (two operands) or Fma (a*b+c) of
// exact constants of `type`, rounded to nearest even. `flush_to_zero` (binary32 only) flushes
// subnormal inputs and results to sign-preserving zero; a result that may have rounded up to the
// smallest normal magnitude from below is refused, as hardware tininess detection decides its flush.
// Non-finite results are refused.
std::optional<ConstantValue> ieeeArithmetic(const std::string& name, const std::string& type,
    std::span<const ConstantValue> operands, bool flush_to_zero);
enum class Rounding { NearestEven, TowardZero, TowardNegative, TowardPositive };
// An exact integer or IEEE constant rounded to the IEEE type `target` (f16, bf16, f32, f64).
// Overflow past the largest finite value is refused.
std::optional<ConstantValue> roundedConversion(const ConstantValue& value, const std::string& target,
    Rounding rounding);
}
