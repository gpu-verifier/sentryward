#include "sentryward/value_ir/native_matrix.h"
#include <ostream>
#include <stdexcept>

namespace sentryward::value_ir {
namespace {
void require(bool condition, const char* message) { if (!condition) throw std::invalid_argument(message); }
bool bits(const ValueType& t) { return t.element().isBitLike() && t.element().kind() != ElementKind::Predicate; }
void modifiers(std::ostream& out, const std::vector<std::string>& values) {
  for (const auto& value : values) out << '.' << value;
}
}
WarpMmaOp::WarpMmaOp(bool bfloat)
    : TypedOperation({ValueType::matrix(ElementType::floating(32),{16,8})}), bfloat_(bfloat) {}
void WarpMmaOp::validateOperands(std::span<const ValueType> operands) const {
  require(operands.size()==320,"Warp MMA requires all 32 lane register fragments");
  const auto packed=ValueType::scalar(ElementType::bit(32));
  const auto accumulator=ValueType::scalar(ElementType::floating(32));
  for (std::size_t i=0;i<operands.size();++i)
    require(operands[i]==(i<192 ? packed : accumulator),"Warp MMA fragment type mismatch");
}
void WarpMmaOp::printAttributes(std::ostream& output) const {
  output << "m16n8k16 row.col f32." << (bfloat_ ? "bf16.bf16" : "f16.f16") << ".f32";
}
std::string WarpMmaOp::operandLabel(std::size_t index) const {
  const auto base=index<128 ? 0u : index<192 ? 128u : 192u;
  const auto width=index<128 ? 4u : index<192 ? 2u : 4u;
  return std::string(index<128 ? "A" : index<192 ? "B" : "C")+".lane"+
      std::to_string((index-base)/width)+".reg"+std::to_string((index-base)%width);
}
NativeMatrixOp::NativeMatrixOp(NativeMatrixOpcode op, ValueType type, NativeMatrixAttributes attributes)
    : TypedOperation({std::move(type)}), opcode_(op), attributes_(std::move(attributes)) {
  if (opcode_ == NativeMatrixOpcode::TcgenCopyValue) require(resultType().isMatrix(), "TMEM copy value must be a matrix");
  else if (opcode_ == NativeMatrixOpcode::TcgenMmaOutputMask)
    require(resultType().isMatrix() && resultType().element().kind() == ElementKind::Predicate, "MMA mask requires a predicate matrix");
  else require(bits(resultType()), "native addresses require integer/bit values");
  if (opcode_ == NativeMatrixOpcode::TmaGlobalAddress)
    require(attributes_.coordinate_count >= 1 && attributes_.coordinate_count <= 5, "invalid TMA coordinate count");
  require(!attributes_.packed_half || *attributes_.packed_half < 2, "packed register half must be zero or one");
}
std::string_view NativeMatrixOp::name() const noexcept {
  switch (opcode_) {
    case NativeMatrixOpcode::TmaGlobalAddress: return "TmaGlobalAddress";
    case NativeMatrixOpcode::TmaSharedAddress: return "TmaSharedAddress";
    case NativeMatrixOpcode::TcgenCopyValue: return "TcgenCopyValue";
    case NativeMatrixOpcode::TcgenMmaAddress: return "TcgenMmaAddress";
    case NativeMatrixOpcode::TcgenMmaOutputMask: return "TcgenMmaOutputMask";
    case NativeMatrixOpcode::TcgenRegisterAddress: return "TcgenRegisterAddress";
    case NativeMatrixOpcode::TcgenCopyAddress: return "TcgenCopyAddress";
    case NativeMatrixOpcode::TmemAllocation: return "TmemAllocation";
  }
  return {};
}
void NativeMatrixOp::validateOperands(std::span<const ValueType> operands) const {
  std::size_t count = 0;
  switch (opcode_) {
    case NativeMatrixOpcode::TmaGlobalAddress: count = 1 + attributes_.coordinate_count; break;
    case NativeMatrixOpcode::TmaSharedAddress: count = 4; break;
    case NativeMatrixOpcode::TcgenCopyValue: count = 1; break;
    case NativeMatrixOpcode::TcgenMmaAddress: count = 3; break;
    case NativeMatrixOpcode::TcgenMmaOutputMask: count = 2 + attributes_.mask_words; break;
    case NativeMatrixOpcode::TcgenRegisterAddress: count = 3; break;
    case NativeMatrixOpcode::TcgenCopyAddress: count = 2; break;
    case NativeMatrixOpcode::TmemAllocation: count = 1; break;
  }
  require(operands.size() == count, "native matrix operation arity mismatch");
  for (std::size_t i = 0; i < operands.size(); ++i) {
    if (opcode_ == NativeMatrixOpcode::TcgenCopyValue && i == 0) {
      require(operands[i].isMatrix(), "TMEM copy value must be a matrix"); continue;
    }
    require(bits(operands[i]), "native matrix parameter must have integer/bit elements");
    const bool descriptor = (opcode_ == NativeMatrixOpcode::TmaGlobalAddress && i == 0) ||
        (opcode_ == NativeMatrixOpcode::TmaSharedAddress && i == 1);
    if (!descriptor) require(operands[i].isScalar(), "native matrix parameter must be scalar");
  }
}
void NativeMatrixOp::printAttributes(std::ostream& out) const {
  modifiers(out, attributes_.modifiers);
  const auto role = [&]() -> const char* {
    switch (attributes_.role) {
      case MatrixRole::A: return "A"; case MatrixRole::B: return "B"; case MatrixRole::D: return "D";
      case MatrixRole::ScaleA: return "scale_A"; case MatrixRole::ScaleB: return "scale_B";
      case MatrixRole::SparseMetadata: return "sparse_metadata";
      case MatrixRole::Source: return "source"; case MatrixRole::Destination: return "destination";
    }
    return "invalid";
  };
  out << " role=" << role();
  if (attributes_.coordinate_count) out << " coordinates=" << attributes_.coordinate_count;
  if (attributes_.mask_words) out << " mask_words=" << attributes_.mask_words;
  if (opcode_ == NativeMatrixOpcode::TcgenRegisterAddress) out << " register=" << attributes_.register_index;
  if (attributes_.packed_half) out << " half=" << *attributes_.packed_half;
}
std::string NativeMatrixOp::operandLabel(std::size_t i) const {
  switch (opcode_) {
    case NativeMatrixOpcode::TmaGlobalAddress: return i ? "coordinate[" + std::to_string(i - 1) + "]" : "tensor_map";
    case NativeMatrixOpcode::TmaSharedAddress:
      return i == 0 ? "base" :
          i == 1 ? "tensor_map" : i == 2 ? "issuing_rank" : "target_rank";
    case NativeMatrixOpcode::TcgenMmaAddress: return i == 0 ? "base_or_matrix_descriptor" : i == 1 ? "instruction_descriptor" : "issuing_rank";
    case NativeMatrixOpcode::TcgenMmaOutputMask: return i == 0 ? "instruction_descriptor" : i == 1 ? "destination_base" : "disable_lanes[" + std::to_string(i-2) + "]";
    case NativeMatrixOpcode::TcgenRegisterAddress: return i == 0 ? "tmem_base" : i == 1 ? "thread_index" : "half_split_offset";
    case NativeMatrixOpcode::TcgenCopyAddress: return i == 0 ? "base_or_matrix_descriptor" : "issuing_rank";
    case NativeMatrixOpcode::TcgenCopyValue: return "shared_matrix";
    case NativeMatrixOpcode::TmemAllocation: return "columns";
  }
  return {};
}
TcgenMmaOp::TcgenMmaOp(ValueType type, TcgenMmaAttributes attributes)
    : TypedOperation({std::move(type)}), attributes_(std::move(attributes)) {
  require(resultType().isMatrix(), "MMA result must be a matrix");
}
void TcgenMmaOp::validateOperands(std::span<const ValueType> operands) const {
  const auto count = 3 + attributes_.accumulate + 2 * attributes_.scaled + attributes_.sparse + attributes_.scale_input_d;
  require(operands.size() == static_cast<std::size_t>(count), "MMA parameter count mismatch");
  for (std::size_t i = 0; i < operands.size(); ++i) {
    if (i == 2 || (attributes_.scale_input_d && i + 1 == operands.size()))
      require(operands[i].isScalar() && bits(operands[i]), "MMA descriptor/exponent must be integer/bit scalar");
    else require(operands[i].isMatrix(), "MMA data operand must be a matrix");
  }
}
void TcgenMmaOp::printAttributes(std::ostream& out) const {
  modifiers(out, attributes_.modifiers);
  out << " accumulate=" << attributes_.accumulate << " scaled=" << attributes_.scaled
      << " sparse=" << attributes_.sparse << " scale_input_d=" << attributes_.scale_input_d;
}
std::string TcgenMmaOp::operandLabel(std::size_t i) const {
  if (i < 3) return i == 0 ? "A" : i == 1 ? "B" : "instruction_descriptor";
  i -= 3;
  if (attributes_.accumulate) { if (!i) return "D"; --i; }
  if (attributes_.scaled) { if (i < 2) return i ? "scale_B" : "scale_A"; i -= 2; }
  if (attributes_.sparse) { if (!i) return "sparse_metadata"; --i; }
  return "scale_input_d";
}
} // namespace sentryward::value_ir
