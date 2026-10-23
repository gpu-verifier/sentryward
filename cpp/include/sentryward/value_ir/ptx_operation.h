#pragma once

#include "sentryward/value_ir/operations.h"
#include <memory>

namespace sentryward::value_ir {

// Opaque 128-bit response to an unsuccessful PTX cluster cancellation request.
// Its cancellation predicate is false; its payload is not an integer zero.
class FailedClcResponseOp final : public TypedOperation {
public:
  FailedClcResponseOp();
  std::string_view name() const noexcept override { return "FailedClcResponse"; }
  void validateOperands(std::span<const ValueType>) const override;
  void printAttributes(std::ostream&) const override;
};

// A value whose bits this IR does not determine: the content PTX leaves
// unspecified to the model, such as a tensor map after a field edit (the
// CUtensorMap byte layout is opaque). It names its producer. No evaluator gives
// it a value, so anything computed from it stays unresolved.
class OpaqueValueOp final : public TypedOperation {
public:
  OpaqueValueOp(ValueType type, std::string provenance);
  std::string_view name() const noexcept override { return "OpaqueValue"; }
  void validateOperands(std::span<const ValueType>) const override;
  void printAttributes(std::ostream&) const override;
  [[nodiscard]] const std::string& provenance() const noexcept { return provenance_; }
private:
  std::string provenance_;
};

// PTX-specific scalar operations whose semantics cannot be expressed by the
// portable arithmetic primitives without losing instruction information.
// An evaluator can dispatch on this public enum and inspect typed attributes;
// there is no arbitrary operation-name or caller-supplied signature escape.
enum class PtxScalarOpcode {
  AddCarry, SubtractCarry, MultiplyAdd, MultiplyAddCarry,
  SetPredicate, SetValue, SelectSign, AddressSpaceConvert, ClusterSharedAddress,
  BitReverse, BitFind, CountLeadingZeros, PopulationCount, BitExtract, BitInsert,
  BytePermute, Logic3, FunnelShift, CopySign, LogicalNot, TestPredicate,
  Sin, Cos, Log2, Exp2, Reciprocal, Sqrt, ReciprocalSqrt, Tanh
};

struct PtxScalarAttributes final {
  std::vector<std::string> modifiers;
  bool carry_in = false;
  bool carry_out = false;
};

class PtxScalarOp final : public TypedOperation {
public:
  PtxScalarOp(PtxScalarOpcode opcode, ValueType source_type,
              std::vector<ValueType> results, PtxScalarAttributes attributes = {});
  [[nodiscard]] std::string_view name() const noexcept override;
  [[nodiscard]] PtxScalarOpcode opcode() const noexcept { return opcode_; }
  [[nodiscard]] const ValueType &sourceType() const noexcept { return source_type_; }
  [[nodiscard]] const PtxScalarAttributes &attributes() const noexcept { return attributes_; }
  void validateOperands(std::span<const ValueType>) const override;
  void printAttributes(std::ostream &) const override;
private:
  PtxScalarOpcode opcode_;
  ValueType source_type_;
  PtxScalarAttributes attributes_;
};

} // namespace sentryward::value_ir
