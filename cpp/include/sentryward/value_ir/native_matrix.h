#pragma once
#include "sentryward/value_ir/operations.h"

namespace sentryward::value_ir {

// Lane-major register fragments: A[32][4] b32, B[32][2] b32, C[32][4] f32.
// https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float
class WarpMmaOp final : public TypedOperation {
 public:
  explicit WarpMmaOp(bool bfloat);
  std::string_view name() const noexcept override { return "WarpMatMul"; }
  void validateOperands(std::span<const ValueType>) const override;
  void printAttributes(std::ostream&) const override;
  std::string operandLabel(std::size_t) const override;
  bool bfloat() const noexcept { return bfloat_; }
 private:
  bool bfloat_;
};

enum class NativeMatrixOpcode {
  TmaGlobalAddress, TmaSharedAddress,
  TcgenMmaAddress, TcgenMmaOutputMask, TcgenRegisterAddress,
  TcgenCopyAddress, TcgenCopyValue, TmemAllocation
};
enum class MatrixRole { A, B, D, ScaleA, ScaleB, SparseMetadata, Source, Destination };
struct NativeMatrixAttributes final {
  std::vector<std::string> modifiers;
  MatrixRole role = MatrixRole::Source;
  unsigned coordinate_count = 0;
  unsigned register_index = 0;
  std::optional<unsigned> packed_half = std::nullopt;
  unsigned mask_words = 0;
};

// Parameterized PTX layout functions. Address results are ordinary integer values.
// https://docs.nvidia.com/cuda/parallel-thread-execution/#tensorcore-5th-generation-instructions
class NativeMatrixOp final : public TypedOperation {
 public:
  NativeMatrixOp(NativeMatrixOpcode opcode, ValueType type, NativeMatrixAttributes attributes = {});
  std::string_view name() const noexcept override;
  void validateOperands(std::span<const ValueType>) const override;
  void printAttributes(std::ostream&) const override;
  std::string operandLabel(std::size_t) const override;
  NativeMatrixOpcode opcode() const noexcept { return opcode_; }
  const NativeMatrixAttributes& attributes() const noexcept { return attributes_; }
 private:
  NativeMatrixOpcode opcode_;
  NativeMatrixAttributes attributes_;
};

struct TcgenMmaAttributes final {
  std::vector<std::string> modifiers;
  bool accumulate = false;
  bool scaled = false;
  bool sparse = false;
  bool scale_input_d = false;
};
// Operands: A, B, instruction descriptor, optional D, scales, metadata, D exponent.
// Descriptor-dependent signs, types and scale layouts remain part of the operation.
class TcgenMmaOp final : public TypedOperation {
 public:
  TcgenMmaOp(ValueType type, TcgenMmaAttributes attributes);
  std::string_view name() const noexcept override { return "TcgenMatMul"; }
  void validateOperands(std::span<const ValueType>) const override;
  void printAttributes(std::ostream&) const override;
  std::string operandLabel(std::size_t) const override;
  const TcgenMmaAttributes& attributes() const noexcept { return attributes_; }
 private:
  TcgenMmaAttributes attributes_;
};
} // namespace sentryward::value_ir
