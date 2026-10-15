#include "sentryward/value_ir/ptx_operation.h"
#include <algorithm>
#include <ostream>
#include <regex>
#include <set>
#include <stdexcept>

namespace sentryward::value_ir {
namespace {
void Require(bool condition, const char *message) {
  if (!condition) throw std::invalid_argument(message);
}
bool Predicate(const ValueType &type) {
  return type.isScalar() && type.element().kind() == ElementKind::Predicate;
}
bool Integer32(const ValueType &type) {
  return type.isScalar() && type.element().isBitLike() && type.element().width() == 32;
}
bool Has(const PtxScalarAttributes &attributes, std::string_view modifier) {
  return std::find(attributes.modifiers.begin(), attributes.modifiers.end(), modifier) != attributes.modifiers.end();
}
} // namespace
FailedClcResponseOp::FailedClcResponseOp()
    : TypedOperation({ValueType::scalar(ElementType::bit(128))}) {}
void FailedClcResponseOp::validateOperands(std::span<const ValueType> operands) const {
  Require(operands.empty(), "Failed CLC response takes no operands");
}
void FailedClcResponseOp::printAttributes(std::ostream& out) const { out << " canceled=false"; }
PtxScalarOp::PtxScalarOp(PtxScalarOpcode opcode, ValueType source_type,
                         std::vector<ValueType> results, PtxScalarAttributes attributes)
    : TypedOperation(std::move(results)), opcode_(opcode), source_type_(std::move(source_type)), attributes_(std::move(attributes)) {
  Require(!resultTypes().empty(), "PTX scalar operation must produce values");
  Require(resultTypes().size() <= 2, "PTX scalar operation has too many results");
  Require(!name().empty(), "Invalid PTX scalar opcode");
  const bool carry_op = opcode == PtxScalarOpcode::AddCarry || opcode == PtxScalarOpcode::SubtractCarry ||
      opcode == PtxScalarOpcode::MultiplyAdd || opcode == PtxScalarOpcode::MultiplyAddCarry;
  Require(carry_op || (!attributes_.carry_in && !attributes_.carry_out), "Carry attributes on a non-carry operation");
  if (attributes_.carry_out) Require(resultTypes().size() == 2 && Predicate(resultTypes()[1]), "Carry output must be a predicate result");
  if (opcode == PtxScalarOpcode::SetPredicate || opcode == PtxScalarOpcode::TestPredicate)
    for (const auto &type : resultTypes()) Require(Predicate(type), "PTX predicate operation needs predicate outputs");
  else Require(resultTypes().size() == (attributes_.carry_out ? 2u : 1u), "Unexpected PTX scalar result count");
  static const std::regex type_spelling("^(pred|bf[0-9]+(x[0-9]+)?|[busf][0-9]+(x[0-9]+)?)$");
  static const std::set<std::string> known_modifiers{
      "rn", "rz", "rm", "rp", "rna", "rni", "rzi", "rmi", "rpi", "approx", "ftz", "sat", "relu",
      "lo", "hi", "wide", "cc", "to", "global", "shared", "local", "const", "param", "shared::cta", "shared::cluster",
      "eq", "ne", "lt", "le", "gt", "ge", "ls", "hs", "equ", "neu", "ltu", "leu", "gtu", "geu", "num", "nan",
      "and", "or", "xor", "shiftamt", "f4e", "b4e", "rc8", "ecl", "ecr", "rc16", "l", "r", "wrap", "clamp",
      "finite", "infinite", "number", "notanumber", "normal", "subnormal"};
  for (const auto &modifier : attributes_.modifiers)
    Require(known_modifiers.contains(modifier) || std::regex_match(modifier, type_spelling), "Unknown PTX scalar modifier");
  const auto &result = resultTypes()[0];
  const auto &element = source_type_.element();
  const bool bits = element.isBitLike() && element.kind() != ElementKind::Predicate;
  const auto same = [&] { Require(result == source_type_, "PTX result must match source type"); };
  const auto integer = [&] { Require(source_type_.isScalar() && element.isInteger(), "PTX operation requires scalar integer type"); };
  const auto bitwise = [&] { Require(source_type_.isScalar() && bits, "PTX operation requires scalar integer/bit type"); };
  const auto floating = [&] { Require(element.kind() == ElementKind::Floating, "PTX operation requires floating type"); };
  switch (opcode_) {
  case PtxScalarOpcode::AddCarry: case PtxScalarOpcode::SubtractCarry:
    integer(); same(); Require(element.width() == 32 || element.width() == 64, "PTX carry width must be 32 or 64"); break;
  case PtxScalarOpcode::MultiplyAdd: case PtxScalarOpcode::MultiplyAddCarry:
    Require(element.isNumeric(), "PTX multiply-add requires numeric values");
    if (attributes_.carry_in || attributes_.carry_out || opcode_ == PtxScalarOpcode::MultiplyAddCarry) integer();
    if (Has(attributes_, "wide")) Require(element.isInteger() && result.element().kind() == element.kind() &&
        result.element().width() == element.width() * 2 && sameShape(source_type_, result), "Invalid wide PTX multiply-add result");
    else same();
    break;
  case PtxScalarOpcode::SetPredicate:
    Require(element.isNumeric() || element.isBitLike(), "Invalid PTX comparison source type"); break;
  case PtxScalarOpcode::SetValue:
    Require(element.isNumeric() && result.element().isNumeric() && sameShape(source_type_, result), "Invalid PTX set types"); break;
  case PtxScalarOpcode::SelectSign:
    Require(source_type_.isScalar() && (element.kind() == ElementKind::SignedInteger || element.kind() == ElementKind::Floating), "slct requires signed/float test value"); break;
  case PtxScalarOpcode::AddressSpaceConvert:
  case PtxScalarOpcode::ClusterSharedAddress:
    integer(); same(); Require(element.width() == 32 || element.width() == 64, "cvta requires 32/64-bit values"); break;
  case PtxScalarOpcode::BitFind: case PtxScalarOpcode::CountLeadingZeros: case PtxScalarOpcode::PopulationCount:
    bitwise(); Require(Integer32(result), "Bit count/find result must be 32-bit"); break;
  case PtxScalarOpcode::BitReverse: case PtxScalarOpcode::BitExtract: case PtxScalarOpcode::BitInsert: case PtxScalarOpcode::LogicalNot:
    bitwise(); same(); break;
  case PtxScalarOpcode::BytePermute: case PtxScalarOpcode::Logic3: case PtxScalarOpcode::FunnelShift:
    bitwise(); same(); Require(element.width() == 32, "PTX operation requires 32-bit components"); break;
  case PtxScalarOpcode::TestPredicate:
    floating(); break;
  default:
    floating(); same(); break;
  }
}
std::string_view PtxScalarOp::name() const noexcept {
  switch (opcode_) {
  case PtxScalarOpcode::AddCarry: return "PtxAddCarry";
  case PtxScalarOpcode::SubtractCarry: return "PtxSubtractCarry";
  case PtxScalarOpcode::MultiplyAdd: return "PtxMultiplyAdd";
  case PtxScalarOpcode::MultiplyAddCarry: return "PtxMultiplyAddCarry";
  case PtxScalarOpcode::SetPredicate: return "PtxSetPredicate";
  case PtxScalarOpcode::SetValue: return "PtxSetValue";
  case PtxScalarOpcode::SelectSign: return "PtxSelectSign";
  case PtxScalarOpcode::AddressSpaceConvert: return "PtxAddressSpaceConvert";
  case PtxScalarOpcode::ClusterSharedAddress: return "ClusterSharedAddress";
  case PtxScalarOpcode::BitReverse: return "PtxBitReverse";
  case PtxScalarOpcode::BitFind: return "PtxBitFind";
  case PtxScalarOpcode::CountLeadingZeros: return "PtxCountLeadingZeros";
  case PtxScalarOpcode::PopulationCount: return "PtxPopulationCount";
  case PtxScalarOpcode::BitExtract: return "PtxBitExtract";
  case PtxScalarOpcode::BitInsert: return "PtxBitInsert";
  case PtxScalarOpcode::BytePermute: return "PtxBytePermute";
  case PtxScalarOpcode::Logic3: return "PtxLogic3";
  case PtxScalarOpcode::FunnelShift: return "PtxFunnelShift";
  case PtxScalarOpcode::CopySign: return "PtxCopySign";
  case PtxScalarOpcode::LogicalNot: return "PtxLogicalNot";
  case PtxScalarOpcode::TestPredicate: return "PtxTestPredicate";
  case PtxScalarOpcode::Sin: return "PtxSin";
  case PtxScalarOpcode::Cos: return "PtxCos";
  case PtxScalarOpcode::Log2: return "PtxLog2";
  case PtxScalarOpcode::Exp2: return "PtxExp2";
  case PtxScalarOpcode::Reciprocal: return "PtxReciprocal";
  case PtxScalarOpcode::Sqrt: return "PtxSqrt";
  case PtxScalarOpcode::ReciprocalSqrt: return "PtxReciprocalSqrt";
  case PtxScalarOpcode::Tanh: return "PtxTanh";
  }
  return {};
}
void PtxScalarOp::validateOperands(std::span<const ValueType> operands) const {
  if (attributes_.carry_in) {
    Require(!operands.empty() && Predicate(operands.back()), "Missing carry predicate operand");
    operands = operands.first(operands.size() - 1);
  }
  const auto exact = [&](std::size_t count) { Require(operands.size() == count, "Wrong PTX scalar operand count"); };
  const auto source = [&](std::size_t index) { Require(operands[index] == source_type_, "PTX source operand type mismatch"); };
  switch (opcode_) {
  case PtxScalarOpcode::ClusterSharedAddress:
    exact(2); source(0); Require(Integer32(operands[1]), "Cluster rank must be 32-bit"); break;
  case PtxScalarOpcode::AddCarry: case PtxScalarOpcode::SubtractCarry:
    exact(2); source(0); source(1); break;
  case PtxScalarOpcode::MultiplyAdd: case PtxScalarOpcode::MultiplyAddCarry:
    exact(3); source(0); source(1);
    Require(operands[2] == resultTypes()[0], "PTX multiply-add accumulator type mismatch"); break;
  case PtxScalarOpcode::SetPredicate: case PtxScalarOpcode::SetValue:
    Require(operands.size() == 2 || operands.size() == 3, "Wrong PTX comparison operand count"); source(0); source(1);
    if (operands.size() == 3) {
      Require(Predicate(operands[2]), "PTX boolean comparison operand must be predicate");
      Require(Has(attributes_, "and") || Has(attributes_, "or") || Has(attributes_, "xor"), "PTX comparison missing boolean operator");
    }
    break;
  case PtxScalarOpcode::SelectSign:
    exact(3); Require(operands[0] == resultTypes()[0] && operands[1] == resultTypes()[0], "slct value type mismatch"); source(2); break;
  case PtxScalarOpcode::BitExtract:
    exact(3); source(0); Require(Integer32(operands[1]) && Integer32(operands[2]), "bfe offsets must be u32"); break;
  case PtxScalarOpcode::BitInsert:
    exact(4); source(0); source(1); Require(Integer32(operands[2]) && Integer32(operands[3]), "bfi offsets must be u32"); break;
  case PtxScalarOpcode::BytePermute: case PtxScalarOpcode::FunnelShift:
    exact(3); source(0); source(1); Require(Integer32(operands[2]), "PTX permute/shift control must be 32-bit"); break;
  case PtxScalarOpcode::Logic3:
    exact(4); source(0); source(1); source(2); Require(Integer32(operands[3]), "lop3 table must be 32-bit"); break;
  case PtxScalarOpcode::CopySign:
    exact(2); source(0); source(1); break;
  default:
    exact(1); source(0); break;
  }
}
void PtxScalarOp::printAttributes(std::ostream &out) const {
  out << "source=" << source_type_.toString();
  for (const auto &modifier : attributes_.modifiers) out << '.' << modifier;
  if (attributes_.carry_in) out << " carry-in";
  if (attributes_.carry_out) out << " carry-out";
}
} // namespace sentryward::value_ir
