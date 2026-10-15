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
}
