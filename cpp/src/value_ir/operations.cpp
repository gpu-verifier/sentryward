#include "sentryward/value_ir/operations.h"

#include <algorithm>
#include <cctype>
#include <initializer_list>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace sentryward::value_ir {
namespace {

void require(bool condition, std::string_view message) {
  if (!condition) throw std::invalid_argument(std::string(message));
}
void arity(std::span<const ValueType> operands, std::size_t count) {
  require(operands.size() == count, "incorrect operation operand count");
}
bool numeric(const ElementType& element) { return element.isNumeric() || element.isParameterized(); }
bool compatible(const ValueType& a, const ValueType& b) {
  return a.isScalar() == b.isScalar() &&
      (!a.hasStaticShape() || !b.hasStaticShape() || sameShape(a, b)) &&
      (a.element().isParameterized() || b.element().isParameterized() || a.element() == b.element());
}
bool rankTwo(const ValueType& type) { return type.isMatrix() && (!type.hasStaticShape() || type.shape().size() == 2); }
std::string_view roundingName(RoundingMode mode) {
  switch (mode) {
    case RoundingMode::None: return "none";
    case RoundingMode::NearestEven: return "nearest-even";
    case RoundingMode::TowardZero: return "toward-zero";
    case RoundingMode::TowardNegative: return "toward-negative";
    case RoundingMode::TowardPositive: return "toward-positive";
    case RoundingMode::TiesAway: return "ties-away";
  }
  throw std::invalid_argument("invalid rounding mode");
}
std::string_view multiplyName(IntegerMultiplyMode mode) {
  switch (mode) {
    case IntegerMultiplyMode::Low: return "low";
    case IntegerMultiplyMode::High: return "high";
    case IntegerMultiplyMode::Wide: return "wide";
  }
  throw std::invalid_argument("invalid multiply mode");
}
void validateArithmetic(const ArithmeticAttributes& attributes, const ValueType& type,
                        bool multiplication, bool permits_saturation, bool permits_nan = false) {
  static_cast<void>(roundingName(attributes.rounding));
  static_cast<void>(multiplyName(attributes.multiply_mode));
  require(multiplication || attributes.multiply_mode == IntegerMultiplyMode::Low,
          "multiply mode is only valid on Multiply");
  require(attributes.multiply_mode == IntegerMultiplyMode::Low || type.element().isInteger() || type.element().isParameterized(),
          "high/wide Multiply requires integer elements");
  require(attributes.rounding == RoundingMode::None || type.element().kind() == ElementKind::Floating || type.element().isParameterized(),
          "arithmetic rounding requires floating elements");
  require(!attributes.flush_to_zero || type.element().kind() == ElementKind::Floating || type.element().isParameterized(),
          "flush-to-zero requires floating elements");
  require(!attributes.saturate || permits_saturation, "saturation is invalid on this operation");
  require(!attributes.propagate_nan || (permits_nan && type.element().kind()==ElementKind::Floating),
          "NaN propagation requires floating Minimum or Maximum");
}
void printArithmetic(std::ostream& output, const ArithmeticAttributes& attributes) {
  if (attributes.rounding != RoundingMode::None) output << "rounding=" << roundingName(attributes.rounding) << ' ';
  if (attributes.multiply_mode != IntegerMultiplyMode::Low) output << "multiply=" << multiplyName(attributes.multiply_mode) << ' ';
  if (attributes.saturate) output << "saturate ";
  if (attributes.flush_to_zero) output << "flush_to_zero ";
  if (attributes.propagate_nan) output << "propagate_nan ";
}
void validatePermutation(std::span<const std::uint32_t> axes, std::size_t rank) {
  require(axes.size() == rank, "permutation must match matrix rank");
  std::vector<bool> seen(rank);
  for (const auto axis : axes) {
    require(axis < rank && !seen[axis], "invalid matrix permutation");
    seen[axis] = true;
  }
}
void validateLayout(const MatrixLayout& layout, std::size_t rank) {
  require(!layout.name.empty(), "layout name must not be empty");
  require(layout.strides.empty() || layout.strides.size() == rank,
          "layout strides must match matrix rank");
  if (!layout.permutation.empty()) validatePermutation(layout.permutation, rank);
  if (layout.swizzle_bytes) {
    const auto bytes = *layout.swizzle_bytes;
    require(bytes != 0 && (bytes & (bytes - 1)) == 0, "swizzle size must be a positive power of two");
  }
}
template <class T>
void printList(std::ostream& output, std::span<const T> values) {
  output << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) output << ',';
    output << values[i];
  }
  output << ']';
}
void printLayout(std::ostream& output, const MatrixLayout& layout) {
  output << "layout=" << layout.name;
  if (!layout.strides.empty()) {
    output << " strides=";
    printList(output, std::span(layout.strides));
  }
  if (!layout.permutation.empty()) {
    output << " permutation=";
    printList(output, std::span(layout.permutation));
  }
  if (layout.swizzle_bytes) output << " swizzle_bytes=" << *layout.swizzle_bytes;
}
void validateMemory(const MemoryAttributes& attributes) {
  if (attributes.descriptor) {
    const auto& descriptor = *attributes.descriptor;
    require(!descriptor.kind.empty(), "memory descriptor kind must not be empty");
    require(!descriptor.shape.empty(), "memory descriptor shape must not be empty");
    require(std::all_of(descriptor.shape.begin(), descriptor.shape.end(), [](auto extent) { return extent != 0; }),
            "memory descriptor shape has zero extent");
    require(descriptor.byte_strides.empty() || descriptor.byte_strides.size() == descriptor.shape.size(),
            "memory descriptor strides must match its rank");
    require(!descriptor.element_bytes || *descriptor.element_bytes != 0, "descriptor element byte size must be positive");
  }
}
void validateMemoryValue(const MemoryAttributes& attributes, const ValueType& data) {
  require(data.element().kind() != ElementKind::Predicate, "memory values cannot have predicate elements");
  if (attributes.layout) {
    require(data.isMatrix(), "memory layout requires a matrix value");
    const auto& layout = *attributes.layout;
    validateLayout(layout, data.hasStaticShape() ? data.shape().size() :
        !layout.strides.empty() ? layout.strides.size() : layout.permutation.size());
  }
  if (attributes.descriptor) {
    require(data.isMatrix(), "memory descriptor requires a matrix value");
    if (attributes.descriptor->element_bytes && !data.element().isParameterized()) {
      require(data.element().width() % 8 == 0 &&
                  data.element().width() / 8 == *attributes.descriptor->element_bytes,
              "memory descriptor element byte size differs from value element");
    }
  }
}
void validateAddress(const ValueType& address, const ValueType& data) {
  require(address.element().isInteger() || address.element().kind() == ElementKind::Bit,
          "memory address requires integer or bit elements");
  require(address.isScalar() || (data.isMatrix() && (!address.hasStaticShape() || !data.hasStaticShape() || sameShape(address, data))),
          "matrix addresses must match the memory value shape");
}
void printMemory(std::ostream& output, const MemoryAttributes& attributes) {
  switch (attributes.interpretation) {
    case MemoryAttributes::Interpretation::Ordinary: break;
    case MemoryAttributes::Interpretation::TensorMap: output << " interpretation=tensor_map"; break;
    case MemoryAttributes::Interpretation::TmaTensor: output << " interpretation=tma_tensor"; break;
  }
  for (const auto& modifier : attributes.native_modifiers) output << '.' << modifier;
  if (attributes.guarded) output << " guarded";
  if (attributes.layout) {
    output << ' ';
    printLayout(output, *attributes.layout);
  }
  if (attributes.descriptor) {
    const auto& descriptor = *attributes.descriptor;
    output << " descriptor=" << descriptor.kind << " shape=";
    printList(output, std::span(descriptor.shape));
    if (!descriptor.byte_strides.empty()) {
      output << " byte_strides=";
      printList(output, std::span(descriptor.byte_strides));
    }
    if (descriptor.element_bytes) output << " element_bytes=" << *descriptor.element_bytes;
  }
}

}  // namespace

TypedOperation::TypedOperation(std::vector<ValueType> result_types) : result_types_(std::move(result_types)) {}
const ValueType& TypedOperation::resultType() const { return resultTypes().front(); }
InputParameterOp::InputParameterOp(ValueType type, std::string symbol, std::optional<PointerProperties> pointer,
                                 std::optional<std::string> specialized_literal)
    : TypedOperation({std::move(type)}), symbol_(std::move(symbol)), pointer_(std::move(pointer)),
      specialized_literal_(std::move(specialized_literal)) {
  require(!symbol_.empty(), "Input symbol must not be empty");
  require(!pointer_ || (resultType().isScalar() && resultType().element().isBitLike()), "pointer parameter must be an integer/bit scalar");
  require(!specialized_literal_ || (!pointer_ && resultType().isScalar() &&
      resultType().element().isBitLike() && !specialized_literal_->empty()),
      "only nonpointer integer/bit input parameters have a specialized literal");
}
void InputParameterOp::validateOperands(std::span<const ValueType> operands) const { arity(operands, 0); }
void InputParameterOp::printAttributes(std::ostream& output) const {
  output << "symbol=" << symbol_;
  if (pointer_) output << " pointer=" << pointer_->identity << " addressing=" << pointer_->addressing;
  if (specialized_literal_) output << " specialized=" << *specialized_literal_;
}
ConstantOp::ConstantOp(ValueType type, std::string literal, std::optional<PointerProperties> pointer)
    : TypedOperation({std::move(type)}), literal_(std::move(literal)), pointer_(std::move(pointer)) {
  require(!pointer_ || (resultType().isScalar() && resultType().element().isBitLike()), "pointer constant must be an integer/bit scalar");
  require(!literal_.empty() && std::none_of(literal_.begin(), literal_.end(), [](unsigned char ch) {
    return std::isspace(ch) || std::iscntrl(ch);
  }), "Constant literal must be a nonempty token");
}
void ConstantOp::validateOperands(std::span<const ValueType> operands) const { arity(operands, 0); }
void ConstantOp::printAttributes(std::ostream& output) const {
  output << "literal=" << literal_;
  if (pointer_) {
    output << " pointer=" << pointer_->identity << " addressing=" << pointer_->addressing;
    if (pointer_->owner_cta) output << " cta=" << *pointer_->owner_cta;
    if (pointer_->owner_thread) output << " thread=" << *pointer_->owner_thread;
  }
}
CoordinateOp::CoordinateOp(CoordinateKind kind, Axis axis)
    : TypedOperation({ValueType::scalar(ElementType::unsignedInteger(32))}), kind_(kind), axis_(axis) {
  require(!name().empty() && static_cast<unsigned>(axis) <= 2, "invalid coordinate");
  require(kind != CoordinateKind::ClusterRank || axis == Axis::X, "cluster rank has no axis");
}
std::string_view CoordinateOp::name() const noexcept {
  switch (kind_) {
    case CoordinateKind::ThreadIndex: return "ThreadIndex";
    case CoordinateKind::BlockIndex: return "BlockIndex";
    case CoordinateKind::ClusterBlockIndex: return "ClusterBlockIndex";
    case CoordinateKind::ClusterRank: return "ClusterRank";
  }
  return {};
}
void CoordinateOp::validateOperands(std::span<const ValueType> operands) const { arity(operands, 0); }
void CoordinateOp::printAttributes(std::ostream& output) const {
  if (kind_ != CoordinateKind::ClusterRank) output << "axis=" << "xyz"[static_cast<unsigned>(axis_)];
}

UnaryOp::UnaryOp(UnaryOpcode opcode, ValueType type, ArithmeticAttributes attributes)
    : TypedOperation({std::move(type)}), opcode_(opcode), attributes_(attributes) {
  require(opcode == UnaryOpcode::Negate || opcode == UnaryOpcode::Absolute || opcode == UnaryOpcode::BitNot,
          "invalid unary opcode");
  require(opcode == UnaryOpcode::BitNot ? resultType().element().isBitLike() : numeric(resultType().element()),
          "unary opcode cannot operate on this element type");
  require(opcode == UnaryOpcode::Negate || resultType().isScalar(), "Absolute and BitNot are scalar-only");
  validateArithmetic(attributes, resultType(), false, opcode != UnaryOpcode::BitNot);
}
std::string_view UnaryOp::name() const noexcept {
  switch (opcode_) {
    case UnaryOpcode::Negate: return "Negate";
    case UnaryOpcode::Absolute: return "Absolute";
    case UnaryOpcode::BitNot: return "BitNot";
  }
  return {};
}
void UnaryOp::validateOperands(std::span<const ValueType> operands) const {
  arity(operands, 1);
  require(compatible(operands[0], resultType()), "unary operand type must equal its result type");
}
void UnaryOp::printAttributes(std::ostream& output) const { printArithmetic(output, attributes_); }

BinaryOp::BinaryOp(BinaryOpcode opcode, ValueType type, ArithmeticAttributes attributes)
    : TypedOperation({std::move(type)}), opcode_(opcode), attributes_(attributes) {
  bool bitwise = false;
  bool numeric = false;
  switch (opcode_) {
    case BinaryOpcode::BitAnd: case BinaryOpcode::BitOr: case BinaryOpcode::BitXor:
    case BinaryOpcode::ShiftLeft: case BinaryOpcode::ShiftRight: bitwise = true; break;
    case BinaryOpcode::Add: case BinaryOpcode::Subtract: case BinaryOpcode::Multiply:
    case BinaryOpcode::Divide: case BinaryOpcode::Minimum: case BinaryOpcode::Maximum:
      numeric = true; break;
    case BinaryOpcode::Remainder:
      require(resultType().element().isInteger(), "Remainder requires integer elements"); break;
    default: throw std::invalid_argument("invalid binary opcode");
  }
  require((!bitwise && opcode != BinaryOpcode::Remainder) || resultType().isScalar(), "bit operations and Remainder are scalar-only");
  require(!bitwise || resultType().element().isBitLike(), "bitwise operation requires bit-like elements");
  require(!numeric || resultType().element().isNumeric() || resultType().element().isParameterized(), "arithmetic operation requires numeric elements");
  if (opcode_ == BinaryOpcode::ShiftLeft || opcode_ == BinaryOpcode::ShiftRight) {
    require(resultType().element().kind() != ElementKind::Predicate, "cannot shift a predicate");
  }
  validateArithmetic(attributes, resultType(), opcode == BinaryOpcode::Multiply,
                     opcode == BinaryOpcode::Add || opcode == BinaryOpcode::Subtract ||
                     opcode == BinaryOpcode::Multiply,
                     opcode == BinaryOpcode::Minimum || opcode == BinaryOpcode::Maximum);
}
std::string_view BinaryOp::name() const noexcept {
  switch (opcode_) {
    case BinaryOpcode::Add: return "Add";
    case BinaryOpcode::Subtract: return "Subtract";
    case BinaryOpcode::Multiply: return "Multiply";
    case BinaryOpcode::Divide: return "Divide";
    case BinaryOpcode::Remainder: return "Remainder";
    case BinaryOpcode::BitAnd: return "BitAnd";
    case BinaryOpcode::BitOr: return "BitOr";
    case BinaryOpcode::BitXor: return "BitXor";
    case BinaryOpcode::ShiftLeft: return "ShiftLeft";
    case BinaryOpcode::ShiftRight: return "ShiftRight";
    case BinaryOpcode::Minimum: return "Minimum";
    case BinaryOpcode::Maximum: return "Maximum";
  }
  return {};
}
void BinaryOp::validateOperands(std::span<const ValueType> operands) const {
  arity(operands, 2);
  if (opcode_ == BinaryOpcode::Multiply && attributes_.multiply_mode == IntegerMultiplyMode::Wide) {
    require(operands[0] == operands[1] && sameShape(operands[0], resultType()) &&
                operands[0].element().kind() == resultType().element().kind() &&
                std::uint64_t(operands[0].element().width()) * 2 == resultType().element().width(),
            "wide Multiply result must double its matching integer operand width");
  } else if (opcode_ == BinaryOpcode::ShiftLeft || opcode_ == BinaryOpcode::ShiftRight) {
    require(operands[0] == resultType(), "shift source type must equal result type");
    require(operands[1].element().isInteger() || operands[1].element().kind() == ElementKind::Bit,
            "shift amount requires integer or bit elements");
    require(operands[1].isScalar() || sameShape(operands[1], resultType()), "shift amount shape mismatch");
  } else {
    require(compatible(operands[0], resultType()) && compatible(operands[1], resultType()), "binary operand types must equal result type");
  }
}
void BinaryOp::printAttributes(std::ostream& output) const { printArithmetic(output, attributes_); }

IdealExp2Op::IdealExp2Op(ValueType type) : TypedOperation({std::move(type)}) {
  require(resultType().isScalar() && resultType().element().kind() == ElementKind::Floating,
          "Ideal Exp2 requires a floating scalar");
}
void IdealExp2Op::validateOperands(std::span<const ValueType> operands) const {
  arity(operands, 1);
  require(operands[0] == resultType(), "Ideal Exp2 operand type mismatch");
}
void IdealExp2Op::printAttributes(std::ostream& output) const { output << "semantics=ideal"; }
std::string BinaryOp::operandLabel(std::size_t index) const { return index == 0 ? "left" : "right"; }

// Scalar FMA: https://docs.nvidia.com/cuda/floating-point/index.html#the-fused-multiply-add-fma
FmaOp::FmaOp(ValueType type, ArithmeticAttributes attributes)
    : TypedOperation({std::move(type)}), attributes_(attributes) {
  require(resultType().element().kind() == ElementKind::Floating || resultType().element().isParameterized(), "Fma requires floating elements");
  validateArithmetic(attributes, resultType(), false, true);
}
void FmaOp::validateOperands(std::span<const ValueType> operands) const {
  arity(operands, 3);
  for (const auto& operand : operands) require(compatible(operand, resultType()), "Fma operand/result types must match");
}
void FmaOp::printAttributes(std::ostream& output) const { printArithmetic(output, attributes_); }
ConvertOp::ConvertOp(ValueType type, ConversionAttributes attributes)
    : TypedOperation({std::move(type)}), attributes_(attributes) {
  require(resultType().element().isNumeric(), "Convert result must be numeric");
  require(!(attributes.saturate && attributes.saturate_finite), "Convert saturation modes are mutually exclusive");
  require(!attributes.saturate_finite || resultType().element().kind() == ElementKind::Floating,
          "Finite saturation requires a floating result");
  static_cast<void>(roundingName(attributes.rounding));
}
void ConvertOp::validateOperands(std::span<const ValueType> operands) const {
  arity(operands, 1);
  require(operands[0].element().isNumeric(), "Convert operand must be numeric");
  require(sameShape(operands[0], resultType()), "Convert preserves value shape");
  const bool any_float = operands[0].element().kind() == ElementKind::Floating ||
                         resultType().element().kind() == ElementKind::Floating;
  require(attributes_.rounding == RoundingMode::None || any_float, "rounding requires a floating Convert operand or result");
  require(!attributes_.flush_to_zero || any_float, "flush-to-zero requires a floating Convert operand or result");
}
void ConvertOp::printAttributes(std::ostream& output) const {
  if (attributes_.rounding != RoundingMode::None) output << "rounding=" << roundingName(attributes_.rounding) << ' ';
  if (attributes_.saturate) output << "saturate ";
  if (attributes_.flush_to_zero) output << "flush_to_zero ";
  if (attributes_.saturate_finite) output << "saturate_finite ";
}
BitcastOp::BitcastOp(ValueType type) : TypedOperation({std::move(type)}) { require(resultType().isScalar(), "Bitcast is scalar-only"); }
void BitcastOp::validateOperands(std::span<const ValueType> operands) const {
  arity(operands, 1);
  require(operands[0].isScalar() && operands[0].bitSize() == resultType().bitSize(), "Bitcast must preserve total bit size");
}
CompareOp::CompareOp(ComparePredicate predicate, ValueType type, bool flush_to_zero, bool unordered_true)
    : TypedOperation({std::move(type)}), predicate_(predicate), flush_to_zero_(flush_to_zero), unordered_true_(unordered_true) {
  require(resultType().isScalar() && resultType().element().kind() == ElementKind::Predicate, "Compare result elements must be predicates");
  switch (predicate_) {
    case ComparePredicate::Equal: case ComparePredicate::NotEqual: case ComparePredicate::Less:
    case ComparePredicate::LessEqual: case ComparePredicate::Greater: case ComparePredicate::GreaterEqual:
    case ComparePredicate::Ordered: case ComparePredicate::Unordered: break;
    default: throw std::invalid_argument("invalid comparison predicate");
  }
}
void CompareOp::validateOperands(std::span<const ValueType> operands) const {
  arity(operands, 2);
  require(operands[0] == operands[1] && sameShape(operands[0], resultType()), "Compare operand/result shape mismatch");
  const auto& element = operands[0].element();
  require((!flush_to_zero_ && !unordered_true_) || element.kind() == ElementKind::Floating, "floating comparison attributes require floating operands");
  if (predicate_ == ComparePredicate::Ordered || predicate_ == ComparePredicate::Unordered) {
    require(element.kind() == ElementKind::Floating, "Ordered/Unordered comparison requires floating elements");
  } else if (predicate_ != ComparePredicate::Equal && predicate_ != ComparePredicate::NotEqual) {
    require(element.isNumeric(), "relational comparison requires numeric elements");
  }
}
void CompareOp::printAttributes(std::ostream& output) const {
  if (flush_to_zero_) output << "flush_to_zero ";
  if (unordered_true_) output << "unordered_true ";
  output << "predicate=";
  switch (predicate_) {
    case ComparePredicate::Equal: output << "eq"; break;
    case ComparePredicate::NotEqual: output << "ne"; break;
    case ComparePredicate::Less: output << "lt"; break;
    case ComparePredicate::LessEqual: output << "le"; break;
    case ComparePredicate::Greater: output << "gt"; break;
    case ComparePredicate::GreaterEqual: output << "ge"; break;
    case ComparePredicate::Ordered: output << "ordered"; break;
    case ComparePredicate::Unordered: output << "unordered"; break;
  }
}
ThreadForwardOp::ThreadForwardOp(ValueType type) : TypedOperation({std::move(type)}) {}
void ThreadForwardOp::validateOperands(std::span<const ValueType> operands) const {
  arity(operands, 1);
  require(operands[0] == resultType(), "ThreadForward operand must have its result type");
}
SelectOp::SelectOp(ValueType type) : TypedOperation({std::move(type)}) {}
void SelectOp::validateOperands(std::span<const ValueType> operands) const {
  arity(operands, 3);
  require(operands[0].element().kind() == ElementKind::Predicate &&
              (operands[0].isScalar() || sameShape(operands[0], resultType())), "Select condition requires scalar or matching matrix predicates");
  require(compatible(operands[1], resultType()) && compatible(operands[2], resultType()), "Select alternatives must match its result type");
}
std::string SelectOp::operandLabel(std::size_t index) const {
  return index == 0 ? "condition" : (index == 1 ? "true" : "false");
}

LoadOp::LoadOp(ValueType type, MemoryAttributes attributes)
    : TypedOperation({std::move(type)}), attributes_(std::move(attributes)) {
  validateMemory(attributes_);
  validateMemoryValue(attributes_, resultType());
}
void LoadOp::validateOperands(std::span<const ValueType> operands) const {
  arity(operands, attributes_.guarded ? 2 : 1);
  validateAddress(operands[0], resultType());
  if (attributes_.guarded) require(operands[1].isScalar() &&
      operands[1].element().kind() == ElementKind::Predicate, "Load guard must be a scalar predicate");
}
void LoadOp::printAttributes(std::ostream& output) const { printMemory(output, attributes_); }
std::string LoadOp::operandLabel(std::size_t index) const { return index == 0 ? "address" : "predicate"; }
StoreOp::StoreOp(MemoryAttributes attributes)
    : TypedOperation({}), attributes_(std::move(attributes)) { validateMemory(attributes_); }
void StoreOp::validateOperands(std::span<const ValueType> operands) const {
  arity(operands, attributes_.guarded ? 3 : 2);
  validateAddress(operands[0], operands[1]);
  validateMemoryValue(attributes_, operands[1]);
  if (attributes_.guarded) require(operands[2].element().kind() == ElementKind::Predicate &&
      (operands[2].isScalar() || (operands[1].isMatrix() &&
       (!operands[1].hasStaticShape() || !operands[2].hasStaticShape() || sameShape(operands[1], operands[2])))),
      "Store guard must be a scalar or matching matrix predicate");
}
void StoreOp::printAttributes(std::ostream& output) const { printMemory(output, attributes_); }
std::string StoreOp::operandLabel(std::size_t index) const {
  return index == 0 ? "address" : (index == 1 ? "value" : "predicate");
}

MatMulOp::MatMulOp(ValueType type, bool accumulate)
    : TypedOperation({std::move(type)}), accumulate_(accumulate) {
  require(rankTwo(resultType()) && numeric(resultType().element()), "MatMul requires rank-two numeric result");
}
void MatMulOp::validateOperands(std::span<const ValueType> operands) const {
  arity(operands, accumulate_ ? 3 : 2);
  const auto& left = operands[0];
  const auto& right = operands[1];
  require(rankTwo(left) && rankTwo(right), "MatMul requires rank-two input matrices");
  require(numeric(left.element()) && numeric(right.element()) &&
      (left.element().isParameterized() || right.element().isParameterized() || left.element() == right.element()),
      "MatMul input element types must match and be numeric");
  require(left.element().isParameterized() || resultType().element().isParameterized() ||
      (left.element().kind() == resultType().element().kind() && left.element().width() <= resultType().element().width()),
          "MatMul result must have same numeric kind and at least the input width");
  if (left.hasStaticShape() && resultType().hasStaticShape())
    require(left.shape()[0] == resultType().shape()[0], "MatMul row count mismatch");
  if (right.hasStaticShape() && resultType().hasStaticShape())
    require(right.shape()[1] == resultType().shape()[1], "MatMul column count mismatch");
  if (left.hasStaticShape() && right.hasStaticShape())
    require(left.shape()[1] == right.shape()[0], "MatMul reduction dimension mismatch");
  if (accumulate_) require(compatible(operands[2], resultType()), "MatMul accumulator must match result type");
}
void MatMulOp::printAttributes(std::ostream& output) const { output << "accumulate=" << (accumulate_ ? "true" : "false"); }
std::string MatMulOp::operandLabel(std::size_t index) const {
  return index == 0 ? "left" : (index == 1 ? "right" : "accumulator");
}

ExtendedMatMulOp::ExtendedMatMulOp(ValueType result, bool scaled, bool sparse, MatMulAttributes attributes)
    : TypedOperation({std::move(result)}), scaled_(scaled), sparse_(sparse), attributes_(std::move(attributes)) {
  require(rankTwo(resultType()) && numeric(resultType().element()), "MMA result must be a numeric matrix");
  require(attributes_.scale_block > 0 && attributes_.sparse_kept > 0 &&
      attributes_.sparse_group >= attributes_.sparse_kept, "invalid MMA block/group shape");
  require(!sparse_ || !attributes_.metadata_encoding.empty(), "sparse MMA needs a metadata encoding");
  validateLayout(attributes_.left_layout, 2);
  validateLayout(attributes_.right_layout, 2);
}
void ExtendedMatMulOp::validateOperands(std::span<const ValueType> operands) const {
  arity(operands, 3 + (scaled_ ? 2 : 0) + (sparse_ ? 1 : 0));
  const auto& a = operands[0]; const auto& b = operands[1];
  require(rankTwo(a) && rankTwo(b) && numeric(a.element()) && numeric(b.element()),
      "MMA inputs must be numeric matrices");
  require(compatible(operands[2], resultType()), "MMA accumulator type mismatch");
  if (a.hasStaticShape() && resultType().hasStaticShape())
    require(a.shape()[0] == resultType().shape()[0], "MMA output row mismatch");
  if (b.hasStaticShape() && resultType().hasStaticShape())
    require(b.shape()[1] == resultType().shape()[1], "MMA output column mismatch");
  if (sparse_) {
    const auto& meta = operands.back();
    require(rankTwo(meta) && (meta.element().isBitLike() || meta.element().isParameterized()),
        "sparse metadata shape/type mismatch");
    if (meta.hasStaticShape() && a.hasStaticShape())
      require(meta.shape()[0] == a.shape()[0], "sparse metadata row mismatch");
    if (b.hasStaticShape()) {
      const auto k = b.shape()[0];
      require(k % attributes_.sparse_group == 0, "invalid sparse reduction dimension");
      if (a.hasStaticShape())
        require(a.shape()[1] == k / attributes_.sparse_group * attributes_.sparse_kept, "compressed A shape mismatch");
      if (meta.hasStaticShape())
        require(meta.shape()[1] == k / attributes_.sparse_group, "sparse metadata column mismatch");
    }
  } else if (a.hasStaticShape() && b.hasStaticShape())
    require(a.shape()[1] == b.shape()[0], "MMA reduction dimension mismatch");
  if (scaled_) {
    const auto& sa = operands[3]; const auto& sb = operands[4];
    require(rankTwo(sa) && rankTwo(sb) && numeric(sa.element()) && numeric(sb.element()),
        "MMA scale shape/type mismatch");
    if (sa.hasStaticShape() && a.hasStaticShape())
      require(sa.shape()[0] == a.shape()[0], "MMA scale row mismatch");
    if (b.hasStaticShape()) {
      const auto k = b.shape()[0];
      const auto blocks = k / attributes_.scale_block + (k % attributes_.scale_block != 0);
      if (sa.hasStaticShape()) require(sa.shape()[1] == blocks, "MMA left scale block mismatch");
      if (sb.hasStaticShape()) require(sb.shape()[0] == blocks && sb.shape()[1] == b.shape()[1],
          "MMA right scale shape mismatch");
    }
  }
}
void ExtendedMatMulOp::printAttributes(std::ostream& output) const {
  printLayout(output, attributes_.left_layout); output << " right_"; printLayout(output, attributes_.right_layout);
  if (scaled_) output << " scale_block=" << attributes_.scale_block;
  if (sparse_) output << " sparse=" << attributes_.sparse_kept << '/' << attributes_.sparse_group
                      << " metadata=" << attributes_.metadata_encoding;
}
std::string ExtendedMatMulOp::operandLabel(std::size_t index) const {
  if (index < 3) return index == 0 ? "left" : index == 1 ? "right" : "accumulator";
  if (scaled_ && index < 5) return index == 3 ? "scale_left" : "scale_right";
  return "sparse_metadata";
}

}  // namespace sentryward::value_ir
