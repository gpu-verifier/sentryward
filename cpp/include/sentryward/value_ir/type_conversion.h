#pragma once
#include <memory>
#include <string>
#include "sentryward/value_ir/operations.h"

namespace sentryward::value_ir {
enum class TypeConversionKind { Bitcast, Convert };

struct TypeConversionSpec final {
  TypeConversionKind kind;
  ValueType target;
  ValueType element;
  ConversionAttributes attributes;
  bool operator==(const TypeConversionSpec&) const = default;
};
struct TypeConversion final {
  Value source;
  std::shared_ptr<const TypeConversionSpec> spec;
  bool operator==(const TypeConversion& other) const {
    return source == other.source && (spec == other.spec || *spec == *other.spec);
  }
};
// Source-to-destination order, including integer truncation and rounding.
std::string typeConversionLabel(const Value& value);
}  // namespace sentryward::value_ir
