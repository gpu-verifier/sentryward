#pragma once

#include <cstddef>
#include <iosfwd>
#include <span>
#include <string>
#include <string_view>

#include "sentryward/value_ir/type.h"

namespace sentryward::value_ir {

// An extension defines its own semantics and validation. Graph and printers
// consume this interface without an opcode switch or an opaque-op fallback.
class Operation {
 public:
  virtual ~Operation() = default;
  [[nodiscard]] virtual std::string_view name() const noexcept = 0;
  [[nodiscard]] virtual std::span<const ValueType> resultTypes() const noexcept = 0;
  virtual void validateOperands(std::span<const ValueType> operands) const = 0;
  virtual void printAttributes(std::ostream& output) const = 0;
  [[nodiscard]] virtual std::string operandLabel(std::size_t index) const;
};

}  // namespace sentryward::value_ir
