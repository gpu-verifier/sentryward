#pragma once
#include "value_ir_impl.hpp"

namespace sentryward::frontend::ptx {
namespace {
Key ValueTemplateExtractor::Resolve(std::string_view name, const Scope &scope) const {
  auto search = scope;
  for (;;) {
    Key key{search, std::string(name)};
    if (declarations_.contains(key) || symbols_.contains(key)) return key;
    if (search.empty()) return key;
    search.pop_back();
  }
}
void ValueTemplateExtractor::Fail(std::string code, std::string message) {
  if(branch_capture_ && !demand_mode && cursor_<live_->size() && (*live_)[cursor_]==2) {
    branch_capture_->unavailable(code+": "+message);return;
  }
  diagnostics_.push_back({std::move(code), std::move(message), origin_});
}
void ValueTemplateExtractor::Bind(const std::string &name, Expression value) {
  if (name != "_") {
    const auto key = Resolve(name, instruction_->scope);
    const auto declared = declarations_.find(key);
    if (declared == declarations_.end()) {
      Fail("missing_register_declaration", "Destination '" + name + "' has no scoped register declaration");
      registers_.erase(key); invalidated_.insert(key); return;
    }
    if (declared->second.value.bitSize() != value.type.value.bitSize()) {
      if (declared->second.value.isScalar() && value.type.value.isScalar() &&
          declared->second.value.element().isBitLike() && value.type.value.element().isBitLike() &&
          value.type.bits < declared->second.bits && value.type.spelling != "pred" &&
          In(instruction_->opcode, {"ld", "ldu", "cvt"})) {
        const auto kind = value.type.value.element().kind() == v::ElementKind::SignedInteger ? "s" : "u";
        if (value.type.value.element().kind() == v::ElementKind::Bit) value = *Reinterpret(value, *PtxType("u" + std::to_string(value.type.bits)));
        const auto widened = *PtxType(std::string(kind) + std::to_string(declared->second.bits));
        value = {Add(std::make_unique<v::ConvertOp>(widened.value), {value.value}, origin_).result(), widened};
      } else {
        Fail("destination_width_mismatch", "Destination '" + name + "' declaration does not match the PTX result width");
        registers_.erase(key); invalidated_.insert(key); return;
      }
    }
    registers_.insert_or_assign(key, std::move(value));
    invalidated_.erase(key);
  }
}
void ValueTemplateExtractor::Invalidate() {
  for (const auto &name : instruction_->destinations) {
    const auto key = Resolve(name, instruction_->scope);
    registers_.erase(key);
    invalidated_.insert(key);
  }
  if (Has(*instruction_, "cc")) carry_.reset();
}
Expression ValueTemplateExtractor::Constant(const Type &type, const std::string &text) {
  return {Add(std::make_unique<v::ConstantOp>(type.value, text), {}, origin_).result(), type};
}
std::optional<Integer> ValueTemplateExtractor::Known(std::string text, bool uniform_only) const {
  text = Trim(text);
  if (auto literal = ParseInteger(text)) return literal;
  if (text == "true") return Integer(1);
  if (text == "false") return Integer(0);
  if (lifting_) return {};
  if (!uniform_only) {
    if (const auto* value = facts_.Value(*record_, text)) return ScalarInteger(*value);
    return {};
  }
  return facts_.LegacyUniformOnlyValue({lane_, cursor_}, members, text);
}
Expression ValueTemplateExtractor::LinearThread() {
  auto coordinate = [&](v::Axis axis) {
    return Expression{Add(std::make_unique<v::CoordinateOp>(v::CoordinateKind::ThreadIndex, axis), {}, origin_).result(), U32()};
  };
  auto result = coordinate(v::Axis::X);
  const auto& dim = facts_.launch().block_dim();
  for (unsigned axis = 1; axis < 3; ++axis) {
    if (dim[axis] == 1) continue;
    auto scale = Constant(U32(), std::to_string(axis == 1 ? dim[0] : dim[0] * dim[1]));
    auto part = Expression{Add(std::make_unique<v::BinaryOp>(v::BinaryOpcode::Multiply, U32().value),
        {coordinate(static_cast<v::Axis>(axis)).value, scale.value}, origin_).result(), U32()};
    result = {Add(std::make_unique<v::BinaryOp>(v::BinaryOpcode::Add, U32().value), {result.value, part.value}, origin_).result(), U32()};
  }
  return result;
}
std::optional<Expression> ValueTemplateExtractor::Special(std::string text) {
  for (const auto& [prefix, kind] : std::vector<std::pair<std::string, v::CoordinateKind>>{
      {"%tid.", v::CoordinateKind::ThreadIndex}, {"%ctaid.", v::CoordinateKind::BlockIndex},
      {"%cluster_ctaid.", v::CoordinateKind::ClusterBlockIndex}}) {
    if (text.starts_with(prefix) && text.size() == prefix.size()+1) {
      const auto axis = std::string("xyz").find(text.back());
      if (axis < 3) return Expression{Add(std::make_unique<v::CoordinateOp>(kind, static_cast<v::Axis>(axis)), {}, origin_).result(), U32()};
    }
  }
  if (text == "%cluster_ctarank")
    return Expression{Add(std::make_unique<v::CoordinateOp>(v::CoordinateKind::ClusterRank), {}, origin_).result(), U32()};
  // %clusterid = %ctaid / %cluster_nctaid per axis (ConfigForCta; the grid holds whole clusters): a coordinate
  // expression like %ctaid, not the CTA's own constant, so a unit's value view does not depend on its unit.
  if (text.starts_with("%clusterid.") && text.size() == 12 && std::string("xyz").find(text.back()) < 3) {
    const auto axis = std::string("xyz").find(text.back());
    Expression block{Add(std::make_unique<v::CoordinateOp>(v::CoordinateKind::BlockIndex, static_cast<v::Axis>(axis)), {}, origin_).result(), U32()};
    const auto width = frame_.cluster_dim()[axis];
    if (width == 1) return block;
    auto divisor = Constant(U32(), std::to_string(width));
    return Expression{Add(std::make_unique<v::BinaryOp>(v::BinaryOpcode::Divide, U32().value), {block.value, divisor.value}, origin_).result(), U32()};
  }
  if (text == "%laneid" || text == "%warpid") {
    auto thread = LinearThread(), width = Constant(U32(), std::to_string(facts_.launch().warp_size()));
    return Expression{Add(std::make_unique<v::BinaryOp>(text == "%laneid" ? v::BinaryOpcode::Remainder : v::BinaryOpcode::Divide, U32().value),
        {thread.value, width.value}, origin_).result(), U32()};
  }
  if (text.starts_with("%ntid.") && std::string("xyz").find(text.back()) < 3)
    return Constant(U32(), std::to_string(facts_.launch().block_dim()[std::string("xyz").find(text.back())]));
  if (text.starts_with("%clock") || text.starts_with("%globaltimer")) {
    Fail("unsupported_special_register", "Clock/timer values have no coordinate or parameter representation"); return {};
  }
  if (const auto known = facts_.Special(text))
    return Constant(U32(), known->convert_to<std::string>());
  return {};
}
std::optional<Expression> ValueTemplateExtractor::Bits(Expression expression) {
  if (expression.unassigned) expression = Unassigned(*expression.unassigned);
  if (expression.missing_definition) { Fail("missing_definition",*expression.missing_definition); return {}; }
  const auto bits = expression.type.value.bitSize();
  auto integer = PtxType("u" + std::to_string(bits));
  // Internal nibble storage, not a PTX register type. Packed e2m1 conversions
  // use it before widening to the eight-bit register holding both nibbles.
  if (bits == 4) integer = Type{v::ValueType::scalar(v::ElementType::unsignedInteger(4)), "u4", 4};
  if (!integer) { Fail("unsupported_register_width", "Register bit representation exceeds supported scalar width"); return {}; }
  if (expression.components.empty()) {
    if (expression.type.value == integer->value) return expression;
    if (!expression.type.value.isScalar()) { Fail("unresolved_register_components", "Matrix register components require an explicit mapping"); return {}; }
    return Expression{Add(std::make_unique<v::BitcastOp>(integer->value), {expression.value}, origin_).result(), *integer};
  }
  std::optional<Expression> result;
  std::uint32_t offset = 0;
  for (auto component : expression.components) {
    auto value = Bits(component);
    if (!value) return {};
    if (value->type.value != integer->value)
      value = Expression{Add(std::make_unique<v::ConvertOp>(integer->value), {value->value}, origin_).result(), *integer};
    if (offset) {
      auto shift = Constant(U32(), std::to_string(offset));
      value = Expression{Add(std::make_unique<v::BinaryOp>(v::BinaryOpcode::ShiftLeft, integer->value), {value->value, shift.value}, origin_).result(), *integer};
    }
    if (result) value = Expression{Add(std::make_unique<v::BinaryOp>(v::BinaryOpcode::BitOr, integer->value), {result->value, value->value}, origin_).result(), *integer};
    result = value;
    offset += static_cast<std::uint32_t>(component.type.value.bitSize());
  }
  if (offset != bits) { Fail("invalid_register_components", "Register components do not cover its bit width"); return {}; }
  return result;
}
std::optional<Expression> ValueTemplateExtractor::Part(Expression expression, const Type& type, std::uint32_t bit) {
  std::uint32_t offset = 0;
  for (auto component : expression.components) {
    if (offset == bit && component.type.value.bitSize() == type.value.bitSize()) return Reinterpret(component, type);
    offset += static_cast<std::uint32_t>(component.type.value.bitSize());
  }
  if (expression.components.empty() && expression.type.value.isMatrix() && expression.type.bits == type.bits && bit % type.bits == 0) {
    auto index = Constant(U32(), std::to_string(bit / type.bits));
    return Reinterpret({graph_.component(expression.value, {index.value}), Type{v::ValueType::scalar(expression.type.value.element()), expression.type.spelling, expression.type.bits}}, type);
  }
  auto bits = Bits(expression);
  if (!bits || bit + type.value.bitSize() > expression.type.value.bitSize()) return {};
  if (bit) {
    auto shift = Constant(U32(), std::to_string(bit));
    bits = Expression{Add(std::make_unique<v::BinaryOp>(v::BinaryOpcode::ShiftRight, bits->type.value), {bits->value, shift.value}, origin_).result(), bits->type};
  }
  const auto narrow = type.bits == 4
      ? Type{v::ValueType::scalar(v::ElementType::unsignedInteger(4)), "u4", 4}
      : *PtxType("u" + std::to_string(type.bits));
  if (narrow.value != bits->type.value)
    bits = Expression{Add(std::make_unique<v::ConvertOp>(narrow.value), {bits->value}, origin_).result(), narrow};
  return Reinterpret(*bits, type);
}
std::optional<Expression> ValueTemplateExtractor::Reinterpret(Expression expression, const Type &type) {
  if (expression.unassigned) expression = Unassigned(*expression.unassigned);
  if (expression.missing_definition) { Fail("missing_definition",*expression.missing_definition); return {}; }
  if (!expression.components.empty() && type.value.isScalar()) {
    auto bits = Bits(expression);
    if (!bits) return {};
    expression = *bits;
  }
  if (expression.type.value == type.value) return expression;
  if (expression.type.value.bitSize() != type.value.bitSize()) {
    Fail("operand_width_mismatch", "Operand width does not match PTX instruction type " + type.spelling);
    return {};
  }
  if (type.elements > 1) {
    const auto scalar = Type{v::ValueType::scalar(type.value.element()), type.spelling.substr(0, type.spelling.find('x')), type.bits};
    Expression result{{}, type};
    for (std::uint32_t n = 0; n < type.elements; ++n) {
      auto part = Part(expression, scalar, n * type.bits);
      if (!part) return {};
      result.components.push_back(*part);
    }
    return result;
  }
  if (!expression.components.empty()) {
    auto bits = Bits(expression);
    if (!bits) return {};
    return Reinterpret(*bits, type);
  }
  return Expression{Add(std::make_unique<v::BitcastOp>(type.value), {expression.value}, origin_).result(), type};
}
v::NodeUse ValueTemplateExtractor::Operation(std::string name, std::string attributes,
                                      const std::vector<Type> &results,
                                      const std::vector<Expression> &operands,
                                      std::vector<std::string> labels) {
  (void)attributes;
  std::vector<v::ValueType> result_types;
  std::vector<v::Value> values;
  for (const auto &type : results) result_types.push_back(type.value);
  for (const auto &operand : operands) values.push_back(operand.value);
  const bool helper = name == "ptx.address_add" || (name == "ptx.not" && labels.empty());
  const auto op = name.substr(4);
  const auto &i = *instruction_;
  v::ArithmeticAttributes arithmetic;
  if (Has(i, "rn") || Has(i, "rni")) arithmetic.rounding = v::RoundingMode::NearestEven;
  if (Has(i, "rz") || Has(i, "rzi")) arithmetic.rounding = v::RoundingMode::TowardZero;
  if (Has(i, "rm") || Has(i, "rmi")) arithmetic.rounding = v::RoundingMode::TowardNegative;
  if (Has(i, "rp") || Has(i, "rpi")) arithmetic.rounding = v::RoundingMode::TowardPositive;
  if (Has(i, "rna")) arithmetic.rounding = v::RoundingMode::TiesAway;
  arithmetic.saturate = Has(i, "sat");
  arithmetic.flush_to_zero = Has(i, "ftz");
  arithmetic.propagate_nan = Has(i, "NaN");
  if (Has(i, "wide")) arithmetic.multiply_mode = v::IntegerMultiplyMode::Wide;
  if (Has(i, "hi")) arithmetic.multiply_mode = v::IntegerMultiplyMode::High;
  std::unique_ptr<const v::Operation> operation;
  const auto check_modifiers = [&](std::initializer_list<std::string_view> allowed) {
    if (helper) return;
    for (const auto &modifier : i.modifiers)
      if (!PtxType(modifier) && !In(modifier, allowed))
        throw std::invalid_argument("PTX modifier '" + modifier + "' has no exact core semantics on '" + i.opcode + "'");
  };
  static const std::map<std::string, v::BinaryOpcode> binary{
      {"add", v::BinaryOpcode::Add}, {"sub", v::BinaryOpcode::Subtract},
      {"mul", v::BinaryOpcode::Multiply}, {"div", v::BinaryOpcode::Divide}, {"rem", v::BinaryOpcode::Remainder},
      {"and", v::BinaryOpcode::BitAnd}, {"or", v::BinaryOpcode::BitOr}, {"xor", v::BinaryOpcode::BitXor},
      {"shl", v::BinaryOpcode::ShiftLeft}, {"shr", v::BinaryOpcode::ShiftRight},
      {"min", v::BinaryOpcode::Minimum}, {"max", v::BinaryOpcode::Maximum}};
  if (name == "ptx.address_add") operation = std::make_unique<v::BinaryOp>(v::BinaryOpcode::Add, results[0].value);
  else if (auto found = binary.find(op); found != binary.end() && !Has(i, "cc")) {
    if (op == "mul") check_modifiers({"rn", "rz", "rm", "rp", "ftz", "sat", "lo", "hi", "wide"});
    else if (op == "div" && results[0].value.element().kind() == v::ElementKind::Floating)
      check_modifiers({"rn", "rz", "rm", "rp", "ftz", "sat", "approx", "full"});
    else if (In(op, {"add", "sub", "div"})) check_modifiers({"rn", "rz", "rm", "rp", "ftz", "sat"});
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-max
    else if (In(op, {"min", "max"})) check_modifiers({"ftz", "NaN"});
    else check_modifiers({});
    operation = std::make_unique<v::BinaryOp>(found->second, results[0].value, arithmetic);
  } else if (In(op, {"neg", "abs", "not"})) {
    if (op == "not") check_modifiers({});
    else check_modifiers({"ftz"});
    operation = std::make_unique<v::UnaryOp>(op == "neg" ? v::UnaryOpcode::Negate :
        op == "abs" ? v::UnaryOpcode::Absolute : v::UnaryOpcode::BitNot, results[0].value,
        helper ? v::ArithmeticAttributes{} : arithmetic);
  } else if (op == "mov") {
    check_modifiers({});
    const auto source_count = operands.size();
    if (results.size() > 1 || source_count > 1) {
      throw std::invalid_argument("register tuples must be resolved by the frontend");
    } else operation = std::make_unique<v::BitcastOp>(results[0].value);
  } else if (op == "cvt") {
    // PTX rni/rzi/rmi/rpi round a floating result to an integral value;
    // ordinary conversion rounding modes are not equivalent.
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt
    // .relu (here on f16/bf16 results; a packed cvt arrives per component) clamps a negative
    // result to 0 and makes a NaN result the canonical NaN: max.NaN(result, +0.0), since max
    // orders +0.0 above -0.0.
    const bool relu = Has(i, "relu") && In(results[0].spelling, {"f16", "bf16"});
    if (relu) check_modifiers({"rn", "rz", "rm", "rp", "rna", "ftz", "sat", "satfinite", "relu"});
    else check_modifiers({"rn", "rz", "rm", "rp", "rna", "ftz", "sat", "satfinite"});
    operation = std::make_unique<v::ConvertOp>(results[0].value,
        v::ConversionAttributes{arithmetic.rounding, arithmetic.saturate, arithmetic.flush_to_zero, Has(i, "satfinite")});
    if (relu) {
      const auto converted = Add(std::move(operation), std::move(values), origin_).result();
      const auto zero = Add(std::make_unique<v::ConstantOp>(results[0].value, "0"), {}, origin_).result();
      v::ArithmeticAttributes clamp;
      clamp.propagate_nan = true;
      return Add(std::make_unique<v::BinaryOp>(v::BinaryOpcode::Maximum, results[0].value, clamp),
                 {converted, zero}, origin_);
    }
  } else if (op == "fma") {
    check_modifiers({"rn", "rz", "rm", "rp", "ftz", "sat"});
    operation = std::make_unique<v::FmaOp>(results[0].value, arithmetic);
  } else if (op == "selp") {
    check_modifiers({});
    // PTX syntax is true-value,false-value,predicate; the core Select primitive
    // uses predicate,true-value,false-value.
    values = {operands[2].value, operands[0].value, operands[1].value};
    operation = std::make_unique<v::SelectOp>(results[0].value);
  } else if (op == "setp" && results.size() == 1 && operands.size() == 2 &&
      !Has(i, "ftz") && (Has(i, "eq") || Has(i, "ne") || Has(i, "lt") || Has(i, "le") || Has(i, "gt") || Has(i, "ge"))) {
    auto comparison = Has(i, "eq") ? v::ComparePredicate::Equal : Has(i, "ne") ? v::ComparePredicate::NotEqual :
        Has(i, "lt") ? v::ComparePredicate::Less : Has(i, "le") ? v::ComparePredicate::LessEqual :
        Has(i, "gt") ? v::ComparePredicate::Greater : v::ComparePredicate::GreaterEqual;
    operation = std::make_unique<v::CompareOp>(comparison, results[0].value);
  } else {
    static const std::map<std::string, v::PtxScalarOpcode> extensions{
        {"add", v::PtxScalarOpcode::AddCarry}, {"addc", v::PtxScalarOpcode::AddCarry},
        {"sub", v::PtxScalarOpcode::SubtractCarry}, {"subc", v::PtxScalarOpcode::SubtractCarry},
        {"mad", v::PtxScalarOpcode::MultiplyAdd}, {"madc", v::PtxScalarOpcode::MultiplyAddCarry},
        {"setp", v::PtxScalarOpcode::SetPredicate}, {"set", v::PtxScalarOpcode::SetValue},
        {"slct", v::PtxScalarOpcode::SelectSign}, {"cvta", v::PtxScalarOpcode::AddressSpaceConvert},
        {"brev", v::PtxScalarOpcode::BitReverse}, {"bfind", v::PtxScalarOpcode::BitFind},
        {"clz", v::PtxScalarOpcode::CountLeadingZeros}, {"popc", v::PtxScalarOpcode::PopulationCount},
        {"bfe", v::PtxScalarOpcode::BitExtract}, {"bfi", v::PtxScalarOpcode::BitInsert},
        {"prmt", v::PtxScalarOpcode::BytePermute}, {"lop3", v::PtxScalarOpcode::Logic3},
        {"shf", v::PtxScalarOpcode::FunnelShift}, {"copysign", v::PtxScalarOpcode::CopySign},
        {"cnot", v::PtxScalarOpcode::LogicalNot}, {"testp", v::PtxScalarOpcode::TestPredicate},
        {"sin", v::PtxScalarOpcode::Sin}, {"cos", v::PtxScalarOpcode::Cos},
        {"lg2", v::PtxScalarOpcode::Log2}, {"ex2", v::PtxScalarOpcode::Exp2},
        {"rcp", v::PtxScalarOpcode::Reciprocal}, {"sqrt", v::PtxScalarOpcode::Sqrt},
        {"rsqrt", v::PtxScalarOpcode::ReciprocalSqrt}, {"tanh", v::PtxScalarOpcode::Tanh}};
    const auto found = extensions.find(op);
    if (found == extensions.end()) throw std::invalid_argument("No public value operation for PTX instruction");
    auto source_type = operands.front().type.value;
    if (op == "slct") source_type = operands[2].type.value;
    operation = std::make_unique<v::PtxScalarOp>(found->second, source_type, result_types,
        v::PtxScalarAttributes{i.modifiers, In(op, {"addc", "subc", "madc"}), Has(i, "cc")});
  }
  return Add(std::move(operation), std::move(values), origin_);
}

std::optional<Expression> ValueTemplateExtractor::Operand(std::string text, const Type &type, bool defer_missing) {
  text = Trim(text);
  if (text.empty()) { Fail("invalid_operand", "Empty PTX operand"); return {}; }
  if (text.front() == '!') {
    auto source = Operand(text.substr(1), Pred());
    if (!source) return {};
    return Expression{Operation("ptx.not", "not.pred", {Pred()}, {*source}).result(), Pred()};
  }
  if (auto literal=ParseInteger(text)) return Constant(type,literal->convert_to<std::string>());
  if (In(text, {"true", "false"})) return Constant(type, text);
  // Preserve PTX hexadecimal floating encodings verbatim; no host float roundtrip.
  static const std::regex floating_bits("0([fF][0-9a-fA-F]{8}|[dD][0-9a-fA-F]{16})");
  if (std::regex_match(text, floating_bits)) return Constant(type, text);
  // A decimal floating literal is a double, converted to the type at its use (PTX ISA 4.5.2). The
  // ISA states no rounding for that conversion, so a value .f32 cannot hold exactly is refused.
  static const std::regex floating_decimal("-?(([0-9]+\\.[0-9]*|\\.[0-9]+)([eE][+-]?[0-9]+)?|[0-9]+[eE][+-]?[0-9]+)");
  if (In(type.spelling, {"f32", "f64"}) && std::regex_match(text, floating_decimal)) {
    double value = 0;
    const auto parsed = sentryward::equivalence::floatFromChars(text.data(), text.data() + text.size(), value);
    const auto single = static_cast<float>(value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
        (type.spelling == "f32" && static_cast<double>(single) != value)) {
      Fail("inexact_floating_constant", "Decimal floating constant '" + text + "' has no exact " + type.spelling + " value");
      return {};
    }
    std::ostringstream bits;
    bits << std::hex << std::setfill('0');
    if (type.spelling == "f64") bits << "0d" << std::setw(16) << std::bit_cast<std::uint64_t>(value);
    else bits << "0f" << std::setw(8) << std::bit_cast<std::uint32_t>(single);
    return Constant(type, bits.str());
  }
  const auto key = Resolve(text, instruction_->scope);
  if (demand_mode && declarations_.contains(key)) return DemandOperand(text, type);
  if (auto found = registers_.find(key); found != registers_.end()) return found->second;
  if (auto found = inputs_.find(key); found != inputs_.end() &&
      (!symbols_.contains(key) || found->second.type.value == type.value)) return found->second;
  if (symbols_.contains(key)) {
    const auto name = "symbol:" + ScopeText(key.first) + ':' + text;
    const auto space = symbol_spaces_.contains(key) ? symbol_spaces_.at(key) : "param";
    v::PointerProperties pointer{name, space, {}, {}};
    if (space == "shared" || space == "local") pointer.owner_cta = cta_;
    if (space == "local") pointer.owner_thread = origin_.thread;
    auto template_pointer = pointer;
    template_pointer.owner_thread.reset();
    const auto base = Add(std::make_unique<v::ConstantOp>(type.value, text, template_pointer), {}, origin_).result();
    Expression result{graph_.asPointer(base, pointer), type};
    // Symbol identity is independent of its address-register representation.
    inputs_.insert_or_assign(key, result);
    return result;
  }
  if (!declarations_.contains(key)) {
    if (auto parameter = facts_.launch().parameters().find(text); parameter != facts_.launch().parameters().end())
      return Constant(type, parameter->second.convert_to<std::string>());
  }
  if (!declarations_.contains(key) && text.starts_with('%')) {
    if (auto special = Special(text)) return special;
  }
  for (const auto &warp : facts_.launch().warps())
    if (warp.warp_id == facts_.Warp(lane_))
      if (auto value = warp.initial_registers.find(text); value != warp.initial_registers.end() && !invalidated_.contains(key)) {
        auto result = Constant(declarations_.contains(key) ? declarations_.at(key) : type,
                               value->second.convert_to<std::string>());
        registers_.emplace(key, result);
        return result;
      }
  // Neither bound nor invalidated: no instruction on this thread's path has assigned the register.
  const bool unassigned = declarations_.contains(key) && !invalidated_.contains(key);
  if (unassigned && !defer_missing) return Unassigned(key);
  const auto message="No recorded definition for operand '" + text + "' in scope " + ScopeText(key.first);
  if (defer_missing && declarations_.contains(key)) {
    Expression result{{},declarations_.at(key)};
    result.missing_definition=message;
    if (unassigned) result.unassigned=key;
    return result;
  }
  Fail("missing_definition", message);
  return {};
}
// A .reg variable has no initializer (PTX ISA 5.4.4: only .const and .global variables are
// initialized), so a read before any assignment sees a value PTX does not specify. It is one
// OpaqueValue of the declared type: no evaluator gives it a value, so equivalence must hold for
// every value it may take. The node is a thread template, one instance per member thread, so each
// thread's value is its own. It is bound while the register is unassigned, so later reads see the
// same value; a deferred read materialized after an assignment gets its own, independent value.
Expression ValueTemplateExtractor::Unassigned(const Key& key) {
  const auto provenance = "unassigned register " + key.second + " in scope " + ScopeText(key.first);
  if (const auto bound = registers_.find(key); bound != registers_.end() && !bound->second.missing_definition &&
      bound->second.components.empty()) {
    const auto& held = bound->second.value;
    const auto* opaque = dynamic_cast<const v::OpaqueValueOp*>(&graph_.definingNode(held).operation());
    if (opaque && opaque->provenance() == provenance && !held.pointer() && !held.conversion() &&
        held.indices().empty() && !held.threadSelector())
      return bound->second;
  }
  const auto& type = declarations_.at(key);
  Expression value{Add(std::make_unique<v::OpaqueValueOp>(type.value, provenance), {}, origin_).result(), type};
  if (!registers_.contains(key) && !invalidated_.contains(key)) registers_.emplace(key, value);
  return value;
}
std::optional<Expression> ValueTemplateExtractor::CurrentValue(const Key& key) {
  if (const auto found = registers_.find(key); found != registers_.end()) return found->second;
  if (declarations_.contains(key) && !invalidated_.contains(key)) return Unassigned(key);
  return std::nullopt;
}
std::optional<Expression> ValueTemplateExtractor::Address(std::string text) {
  text = Trim(text);
  if (text.size() < 2 || text.front() != '[' || text.back() != ']') {
    Fail("unsupported_address", "Expected a bracketed scalar address: " + text); return {};
  }
  auto split = SplitAddress(std::string_view(text).substr(1, text.size() - 2));
  if (!split) { Fail("unsupported_address", "Unsupported address expression: " + text); return {}; }
  auto base = Operand(split->first, U64());
  if (!base) return {};
  if (base->type.elements != 1 || base->type.spelling == "pred" ||
      base->type.spelling.starts_with('f') || base->type.spelling.starts_with("bf")) {
    Fail("unsupported_address", "Address is not an integer/bit scalar: " + text); return {};
  }
  if (base->type.value.element().kind() == v::ElementKind::Bit)
    base = Reinterpret(*base, *PtxType("u" + std::to_string(base->type.bits)));
  if (split->second != 0) {
    auto offset = Constant(base->type, split->second.convert_to<std::string>());
    base = Expression{Operation("ptx.address_add", "byte offset", {base->type}, {*base, offset}).result(), base->type};
  }
  const auto space = instruction_->semantics.memory_space.value_or("generic");
  v::PointerProperties pointer{"", space, {}, {}};
  if (space.starts_with("shared") || space == "local" || space == "tmem") pointer.owner_cta = cta_;
  if (space == "local") pointer.owner_thread = origin_.thread;
  base->value = graph_.asPointer(base->value, pointer);
  return base;
}
} // namespace
} // namespace sentryward::frontend::ptx
