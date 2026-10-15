#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "sentryward/value_ir/operation.h"

namespace sentryward::value_ir {

class TypedOperation : public Operation {
 public:
  [[nodiscard]] std::span<const ValueType> resultTypes() const noexcept final {
    return result_types_;
  }

 protected:
  explicit TypedOperation(std::vector<ValueType> result_types);
  [[nodiscard]] const ValueType& resultType() const;

 private:
  std::vector<ValueType> result_types_;
};

class InputParameterOp final : public TypedOperation {
 public:
  InputParameterOp(ValueType type, std::string symbol, std::optional<PointerProperties> pointer = {},
                   std::optional<std::string> specialized_literal = {});
  [[nodiscard]] const std::optional<PointerProperties>& pointer() const noexcept { return pointer_; }
  [[nodiscard]] std::string_view name() const noexcept override { return "InputParameter"; }
  void validateOperands(std::span<const ValueType>) const override;
  void printAttributes(std::ostream&) const override;
  [[nodiscard]] const std::string& symbol() const noexcept { return symbol_; }
  // A scalar .param value supplied by this analysis configuration, never tensor
  // contents or an inferred device pointer. The symbolic identity is retained.
  [[nodiscard]] const std::optional<std::string>& specializedLiteral() const noexcept { return specialized_literal_; }
 private:
  std::string symbol_;
  std::optional<PointerProperties> pointer_;
  std::optional<std::string> specialized_literal_;
};

enum class CoordinateKind { ThreadIndex, BlockIndex, ClusterBlockIndex, ClusterRank };
enum class Axis { X, Y, Z };
class CoordinateOp final : public TypedOperation {
 public:
  CoordinateOp(CoordinateKind kind, Axis axis = Axis::X);
  [[nodiscard]] std::string_view name() const noexcept override;
  [[nodiscard]] CoordinateKind kind() const noexcept { return kind_; }
  [[nodiscard]] Axis axis() const noexcept { return axis_; }
  void validateOperands(std::span<const ValueType>) const override;
  void printAttributes(std::ostream&) const override;
 private:
  CoordinateKind kind_;
  Axis axis_;
};

class ConstantOp final : public TypedOperation {
 public:
  ConstantOp(ValueType type, std::string literal, std::optional<PointerProperties> pointer = {});
  [[nodiscard]] const std::optional<PointerProperties>& pointer() const noexcept { return pointer_; }
  [[nodiscard]] std::string_view name() const noexcept override { return "Constant"; }
  void validateOperands(std::span<const ValueType>) const override;
  void printAttributes(std::ostream&) const override;
  [[nodiscard]] const std::string& literal() const noexcept { return literal_; }
 private:
  std::string literal_;
  std::optional<PointerProperties> pointer_;
};

enum class UnaryOpcode { Negate, Absolute, BitNot };
enum class BinaryOpcode {
  Add, Subtract, Multiply, Divide, Remainder,
  BitAnd, BitOr, BitXor, ShiftLeft, ShiftRight, Minimum, Maximum
};
enum class RoundingMode { None, NearestEven, TowardZero, TowardNegative, TowardPositive, TiesAway };
enum class IntegerMultiplyMode { Low, High, Wide };
struct ArithmeticAttributes final {
  RoundingMode rounding = RoundingMode::None;
  IntegerMultiplyMode multiply_mode = IntegerMultiplyMode::Low;
  bool saturate = false;
  bool flush_to_zero = false;
  bool propagate_nan = false;
};
struct ConversionAttributes final {
  bool operator==(const ConversionAttributes&) const = default;
  RoundingMode rounding = RoundingMode::None;
  bool saturate = false;
  bool flush_to_zero = false;
  bool saturate_finite = false;
};

class UnaryOp final : public TypedOperation {
 public:
  UnaryOp(UnaryOpcode opcode, ValueType type, ArithmeticAttributes attributes = {});
  [[nodiscard]] std::string_view name() const noexcept override;
  void validateOperands(std::span<const ValueType>) const override;
  void printAttributes(std::ostream&) const override;
  [[nodiscard]] UnaryOpcode opcode() const noexcept { return opcode_; }
  [[nodiscard]] const ArithmeticAttributes& attributes() const noexcept { return attributes_; }
 private:
  UnaryOpcode opcode_;
  ArithmeticAttributes attributes_;
};

class BinaryOp final : public TypedOperation {
 public:
  BinaryOp(BinaryOpcode opcode, ValueType type, ArithmeticAttributes attributes = {});
  [[nodiscard]] std::string_view name() const noexcept override;
  void validateOperands(std::span<const ValueType>) const override;
  void printAttributes(std::ostream&) const override;
  [[nodiscard]] std::string operandLabel(std::size_t index) const override;
  [[nodiscard]] BinaryOpcode opcode() const noexcept { return opcode_; }
  [[nodiscard]] const ArithmeticAttributes& attributes() const noexcept { return attributes_; }
 private:
  BinaryOpcode opcode_;
  ArithmeticAttributes attributes_;
};

class FmaOp final : public TypedOperation {
 public:
  explicit FmaOp(ValueType type, ArithmeticAttributes attributes = {});
  [[nodiscard]] std::string_view name() const noexcept override { return "Fma"; }
  void validateOperands(std::span<const ValueType>) const override;
  void printAttributes(std::ostream&) const override;
  [[nodiscard]] const ArithmeticAttributes& attributes() const noexcept { return attributes_; }
 private:
  ArithmeticAttributes attributes_;
};

class IdealExp2Op final : public TypedOperation {
 public:
  explicit IdealExp2Op(ValueType type);
  std::string_view name() const noexcept override { return "Exp2"; }
  void validateOperands(std::span<const ValueType>) const override;
  void printAttributes(std::ostream&) const override;
};

class ConvertOp final : public TypedOperation {
 public:
  explicit ConvertOp(ValueType type, ConversionAttributes attributes = {});
  [[nodiscard]] std::string_view name() const noexcept override { return "Convert"; }
  void validateOperands(std::span<const ValueType>) const override;
  void printAttributes(std::ostream&) const override;
  [[nodiscard]] const ConversionAttributes& attributes() const noexcept { return attributes_; }
 private:
  ConversionAttributes attributes_;
};

class BitcastOp final : public TypedOperation {
 public:
  explicit BitcastOp(ValueType type);
  [[nodiscard]] std::string_view name() const noexcept override { return "Bitcast"; }
  void validateOperands(std::span<const ValueType>) const override;
  void printAttributes(std::ostream&) const override {}
};

enum class ComparePredicate { Equal, NotEqual, Less, LessEqual, Greater, GreaterEqual, Ordered, Unordered };
class CompareOp final : public TypedOperation {
 public:
  CompareOp(ComparePredicate predicate, ValueType type, bool flush_to_zero = false, bool unordered_true = false);
  [[nodiscard]] bool flushToZero() const noexcept { return flush_to_zero_; }
  [[nodiscard]] bool unorderedTrue() const noexcept { return unordered_true_; }
  [[nodiscard]] std::string_view name() const noexcept override { return "Compare"; }
  void validateOperands(std::span<const ValueType>) const override;
  void printAttributes(std::ostream&) const override;
  [[nodiscard]] ComparePredicate predicate() const noexcept { return predicate_; }
 private:
  ComparePredicate predicate_;
  bool flush_to_zero_, unordered_true_;
};

class SelectOp final : public TypedOperation {
 public:
  explicit SelectOp(ValueType type);
  [[nodiscard]] std::string_view name() const noexcept override { return "Select"; }
  void validateOperands(std::span<const ValueType>) const override;
  void printAttributes(std::ostream&) const override {}
  [[nodiscard]] std::string operandLabel(std::size_t index) const override;
};

// Layout and descriptor fields describe address-to-element interpretation;
// they neither replace address operands nor introduce memory/order edges.
struct MatrixLayout final {
  std::string name;
  std::vector<std::uint64_t> strides;
  std::vector<std::uint32_t> permutation;
  std::optional<std::uint32_t> swizzle_bytes;
};
struct MemoryDescriptor final {
  std::string kind;
  std::vector<std::uint64_t> shape;
  std::vector<std::int64_t> byte_strides;
  std::optional<std::uint64_t> element_bytes;
};
struct MemoryAttributes final {
  std::optional<MatrixLayout> layout;
  std::optional<MemoryDescriptor> descriptor;
  enum class Interpretation { Ordinary, TensorMap, TmaTensor };
  Interpretation interpretation = Interpretation::Ordinary;
  std::vector<std::string> native_modifiers;
  // A disabled Load yields zero without reading; a disabled Store has no effect.
  bool guarded = false;
};

class LoadOp final : public TypedOperation {
 public:
  explicit LoadOp(ValueType type, MemoryAttributes attributes = {});
  [[nodiscard]] std::string_view name() const noexcept override { return "Load"; }
  void validateOperands(std::span<const ValueType>) const override;
  void printAttributes(std::ostream&) const override;
  [[nodiscard]] std::string operandLabel(std::size_t index) const override;
  [[nodiscard]] const MemoryAttributes& attributes() const noexcept { return attributes_; }
 private:
  MemoryAttributes attributes_;
};

class StoreOp final : public TypedOperation {
 public:
  explicit StoreOp(MemoryAttributes attributes = {});
  void validateOperands(std::span<const ValueType>) const override;
  void printAttributes(std::ostream&) const override;
  [[nodiscard]] std::string_view name() const noexcept override { return "Store"; }
  [[nodiscard]] std::string operandLabel(std::size_t index) const override;
  [[nodiscard]] const MemoryAttributes& attributes() const noexcept { return attributes_; }
 private:
  MemoryAttributes attributes_;
};

class MatMulOp final : public TypedOperation {
 public:
  explicit MatMulOp(ValueType type, bool accumulate = true);
  [[nodiscard]] std::string_view name() const noexcept override { return "MatMul"; }
  void validateOperands(std::span<const ValueType>) const override;
  void printAttributes(std::ostream&) const override;
  [[nodiscard]] std::string operandLabel(std::size_t index) const override;
  [[nodiscard]] bool accumulate() const noexcept { return accumulate_; }
 private:
  bool accumulate_;
};

struct MatMulAttributes final {
  MatrixLayout left_layout{"row-major", {}, {}, {}};
  MatrixLayout right_layout{"row-major", {}, {}, {}};
  std::uint64_t scale_block = 1;
  std::uint64_t sparse_kept = 2;
  std::uint64_t sparse_group = 4;
  std::string metadata_encoding = "2:4-indices";
};
class ExtendedMatMulOp : public TypedOperation {
 public:
  void validateOperands(std::span<const ValueType>) const override;
  void printAttributes(std::ostream&) const override;
  [[nodiscard]] std::string operandLabel(std::size_t index) const override;
  [[nodiscard]] const MatMulAttributes& attributes() const noexcept { return attributes_; }
 protected:
  ExtendedMatMulOp(ValueType result, bool scaled, bool sparse, MatMulAttributes attributes);
 private:
  bool scaled_, sparse_;
  MatMulAttributes attributes_;
};
class ScaledMatMulOp final : public ExtendedMatMulOp {
 public:
  explicit ScaledMatMulOp(ValueType result, MatMulAttributes attributes = {})
      : ExtendedMatMulOp(std::move(result), true, false, std::move(attributes)) {}
  [[nodiscard]] std::string_view name() const noexcept override { return "ScaledMatMul"; }
};
class SparseMatMulOp final : public ExtendedMatMulOp {
 public:
  explicit SparseMatMulOp(ValueType result, MatMulAttributes attributes = {})
      : ExtendedMatMulOp(std::move(result), false, true, std::move(attributes)) {}
  [[nodiscard]] std::string_view name() const noexcept override { return "SparseMatMul"; }
};
class SparseScaledMatMulOp final : public ExtendedMatMulOp {
 public:
  explicit SparseScaledMatMulOp(ValueType result, MatMulAttributes attributes = {})
      : ExtendedMatMulOp(std::move(result), true, true, std::move(attributes)) {}
  [[nodiscard]] std::string_view name() const noexcept override { return "SparseScaledMatMul"; }
};

}  // namespace sentryward::value_ir
