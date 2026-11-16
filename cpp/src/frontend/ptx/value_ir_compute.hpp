#pragma once
#include "value_ir_impl.hpp"

namespace sentryward::frontend::ptx {
namespace {
void ValueTemplateExtractor::InstructionValue() {
  const auto &i = *instruction_;
  if (i.opcode == "cp" && Has(i, "reduce") && Has(i, "bulk") && !Has(i, "tensor")) { BulkReduce(); return; }
  if (i.semantics.atomic_add) {
    Fail("atomic_add_value_semantics_unsupported", "Atomic add numerical semantics are unsupported");
    Invalidate(); return;
  }
  if (record_->normalized_clc_result().has_value()) {
    for (const auto& destination : i.destinations)
      Bind(destination, Constant(Pred(), *record_->normalized_clc_result() ? "true" : "false"));
    return;
  }
  if (i.opcode == "setmaxnreg") return;
  if (i.opcode == "clusterlaunchcontrol" && i.semantics.operation == "try_cancel_complete_tx") {
    if (facts_.launch().ClusterLaunchCancellation()) {
      Fail("cluster_launch_control_outcome", "CLC numerical response needs the no-cancellation scenario"); return;
    }
    if (i.operands.size() != 2) throw std::invalid_argument("CLC response requires destination and barrier");
    auto address = Address(Trim(module_.view(i.operands[0])));
    if (!address) return;
    const auto response = Add(std::make_unique<v::FailedClcResponseOp>(), {}, origin_).result();
    const auto shape = frame_.cluster_dim();
    const auto count = Has(i,"multicast::cluster::all") ? shape[0]*shape[1]*shape[2] : 1u;
    for (unsigned rank = 0; rank < count; ++rank) {
      auto destination = address->value;
      if (Has(i,"multicast::cluster::all")) {
        const auto target = Constant(U32(), std::to_string(rank));
        destination = Add(std::make_unique<v::PtxScalarOp>(v::PtxScalarOpcode::ClusterSharedAddress,
            address->type.value, std::vector{address->type.value}), {destination, target.value}, origin_).result();
      }
      Add(std::make_unique<v::StoreOp>(), {destination, response}, origin_);
    }
    return;
  }
  if (i.opcode == "mapa") {
    if (i.operands.size() != 3 || !Has(i, "shared::cluster") || !(Has(i, "u32") || Has(i, "u64"))) {
      Fail("unsupported_mapa", "Expected mapa.shared::cluster.u32/u64"); Invalidate(); return;
    }
    const auto type = Has(i, "u64") ? U64() : U32();
    auto base = Operand(std::string(module_.view(i.operands[1])), type);
    auto rank = Operand(std::string(module_.view(i.operands[2])), U32());
    if (base) base = Reinterpret(*base, type);
    if (rank) rank = Reinterpret(*rank, U32());
    if (!base || !rank) { Invalidate(); return; }
    const auto value = Add(std::make_unique<v::PtxScalarOp>(v::PtxScalarOpcode::ClusterSharedAddress,
        type.value, std::vector{type.value}), {base->value, rank->value}, origin_).result();
    Bind(std::string(module_.view(i.operands[0])), {value, type});
    return;
  }
  if (i.control != "none") {
    if (i.control == "call") {
      Fail("unsupported_call", "Opaque calls do not provide value definitions"); Invalidate();
    }
    return;
  }
  if (In(i.opcode, {"prefetch", "prefetchu"})) {
    std::string modifiers;
    for (const auto& modifier : i.modifiers) {
      if (!modifiers.empty()) modifiers += '.';
      modifiers += modifier;
    }
    const bool valid = i.opcode == "prefetchu" ? modifiers == "L1" :
        In(modifiers, {"L1", "L2", "global.L1", "global.L2", "local.L1", "local.L2",
                       "global.L2::evict_last", "global.L2::evict_normal",
                       "L1::32B.valid_addr", "global.L1::32B.valid_addr",
                       "tensormap", "const.tensormap", "param.tensormap"});
    if (!valid || i.operands.size() != 1) {
      Fail("unsupported_prefetch", "Unsupported cache-prefetch form"); return;
    }
    const auto address = Trim(module_.view(i.operands[0]));
    if (!address.starts_with('[') || !address.ends_with(']')) {
      Fail("unsupported_prefetch", "Cache prefetch requires a bracketed address"); return;
    }
    return;
  }
  if (i.opcode == "tcgen05") { Tcgen(); return; }
  if (i.opcode == "cp" && Has(i,"reduce")) {
    Fail("atomic_reduction_value_not_modeled", "Atomic bulk reduction needs a read-modify-write value effect and reduction-aware memory elimination");
    Invalidate(); return;
  }
  if (i.opcode == "cp" && Has(i, "async") && Has(i, "bulk") && Has(i, "tensor") && !Has(i, "prefetch")) { Tma(); return; }
  if (i.semantics.category == "memory" && In(i.opcode, {"ld", "ldu", "st"}) && !i.semantics.asynchronous) {
    Memory(); return;
  }
  if (i.opcode == "cp" && i.semantics.operation == "copy") { AsyncCopy(); return; }
  if (i.opcode == "cp" && In(i.semantics.operation, {"bulk_complete_tx", "bulk_copy"})) { BulkCopy(); return; }
  if (i.opcode == "tensormap") { TensorMap(); return; }
  if (i.opcode == "mbarrier" && i.semantics.operation == "inval") { MbarrierInvalidate(); return; }
  if (i.semantics.category == "compute") { Compute(); return; }
  if (i.semantics.memory_effect || !i.destinations.empty() || !i.semantics.modeled) {
    Fail("unsupported_operation", "No value-IR model for '" + i.opcode + "' (including its memory effects/results)");
    Invalidate();
  }
  // Synchronization-only events have no scalar value semantics. Their identity
  // remains in the execution records, and memory-node origins support a future join.
}

void ValueTemplateExtractor::Compute() {
  const auto &i = *instruction_;
  if (i.opcode == "elect") {
    if (i.modifiers != std::vector<std::string>{"sync"} || i.operands.size() != 2) {
      Fail("unsupported_election", "elect.sync requires lane/predicate destinations and a membermask"); Invalidate(); return;
    }
    const auto destinations = Split(module_.view(i.operands[0]), '|');
    if (destinations.size() != 2 || destinations[0].empty() || destinations[1].empty() || destinations[1] == "_") {
      Fail("unsupported_election", "elect.sync requires d|p destinations"); Invalidate(); return;
    }
    const auto raw = Known(std::string(module_.view(i.operands[1])));
    if (!raw || *raw < -(Integer(1) << 31) || *raw > UINT32_MAX) {
      Fail("unsupported_election", "Election membermask is invalid or not statically determined"); Invalidate(); return;
    }
    const auto mask = (*raw & 0xffffffff).convert_to<std::uint32_t>();
    if (facts_.LaneId(lane_) >= 32 || !mask || !(mask & (1u << facts_.LaneId(lane_)))) {
      Fail("unsupported_election", "Election issuer is outside membermask"); Invalidate(); return;
    }
    auto chosen = facts_.launch().elect_lane();
    if (!chosen) {
      if (std::popcount(mask) != 1) {
        Fail("unsupported_election", "Election policy is not configured"); Invalidate(); return;
      }
      chosen = std::countr_zero(mask);
    } else if (*chosen >= 32 || !(mask & (1u << *chosen))) {
      chosen = std::countr_zero(mask);
    }
    if (destinations[0] != "_") Bind(destinations[0], Constant(U32(), std::to_string(*chosen)));
    Bind(destinations[1], Constant(Pred(), facts_.LaneId(lane_) == *chosen ? "true" : "false"));
    return;
  }
  if (i.opcode == "isspacep") {
    // The answer follows from the address's provenance, as the executor decided it.
    const auto *address = i.operands.size() == 2 && i.modifiers.size() == 1
        ? facts_.Value(*record_, Trim(module_.view(i.operands[1]))) : nullptr;
    const auto inside = address ? GenericAddressInSpace(i.modifiers[0], *address, facts_.launch().resources()) : std::nullopt;
    if (!inside) { Fail("unsupported_isspacep", "isspacep needs an address of known state space"); Invalidate(); return; }
    Bind(Trim(module_.view(i.operands[0])), Constant(Pred(), *inside ? "true" : "false"));
    return;
  }
  static const std::map<std::string, std::pair<unsigned, unsigned>> arities{
      {"mov", {1, 1}}, {"cvt", {1, 2}}, {"cvta", {1, 1}},
      {"abs", {1, 1}}, {"neg", {1, 1}}, {"not", {1, 1}}, {"cnot", {1, 1}},
      {"clz", {1, 1}}, {"popc", {1, 1}}, {"brev", {1, 1}}, {"bfind", {1, 1}},
      {"sin", {1, 1}}, {"cos", {1, 1}}, {"lg2", {1, 1}}, {"ex2", {1, 1}},
      {"rcp", {1, 1}}, {"sqrt", {1, 1}}, {"rsqrt", {1, 1}}, {"tanh", {1, 1}}, {"testp", {1, 1}},
      {"add", {2, 2}}, {"addc", {2, 2}}, {"sub", {2, 2}}, {"subc", {2, 2}},
      {"mul", {2, 2}}, {"div", {2, 2}}, {"rem", {2, 2}}, {"min", {2, 3}}, {"max", {2, 3}},
      {"and", {2, 2}}, {"or", {2, 2}}, {"xor", {2, 2}}, {"shl", {2, 2}}, {"shr", {2, 2}},
      {"copysign", {2, 2}}, {"setp", {2, 3}}, {"set", {2, 3}},
      {"mad", {3, 3}}, {"madc", {3, 3}}, {"fma", {3, 3}}, {"selp", {3, 3}}, {"slct", {3, 3}},
      {"bfe", {3, 3}}, {"prmt", {3, 3}}, {"bfi", {4, 4}}, {"lop3", {4, 5}}, {"shf", {3, 3}}};
  auto arity = arities.find(i.opcode);
  if (arity == arities.end()) {
    Fail("unsupported_operation", "No exact per-thread value model for compute operation '" + i.opcode + "'"); Invalidate(); return;
  }
  if (i.operands.empty() || i.operands.size() - 1 < arity->second.first || i.operands.size() - 1 > arity->second.second) {
    Fail("unsupported_operands", "Unsupported operand count for '" + i.opcode + "'"); Invalidate(); return;
  }
  std::vector<Type> types;
  static const std::set<std::string> modifiers{
      "rn", "rz", "rm", "rp", "rna", "rni", "rzi", "rmi", "rpi", "approx", "full", "ftz", "sat", "satfinite", "relu",
      "lo", "hi", "wide", "cc", "to", "global", "shared", "local", "const", "param", "shared::cta", "shared::cluster",
      "eq", "ne", "lt", "le", "gt", "ge", "lo", "ls", "hs", "equ", "neu", "ltu", "leu", "gtu", "geu", "num", "nan",
      "and", "or", "xor", "shiftamt", "f4e", "b4e", "rc8", "ecl", "ecr", "rc16", "l", "r", "wrap", "clamp",
      "finite", "infinite", "number", "notanumber", "normal", "subnormal", "NaN", "abs", "oob"};
  for (const auto &modifier : i.modifiers) {
    if (auto type = PtxType(modifier)) types.push_back(*type);
    else if (!modifiers.contains(modifier)) {
      Fail("unsupported_modifier", "Unsupported modifier '" + modifier + "' on '" + i.opcode + "'"); Invalidate(); return;
    }
  }
  if (types.empty()) { Fail("missing_type", "No supported PTX type for '" + i.opcode + "'"); Invalidate(); return; }
  auto result_type = types.front();
  auto source_type = (In(i.opcode, {"cvt", "set", "slct"}) && types.size() > 1) ? types.back() : types.front();
  if (In(i.opcode, {"setp", "testp"})) result_type = Pred();
  if (In(i.opcode, {"clz", "popc", "bfind"})) result_type = U32();
  if (Has(i, "wide")) {
    auto wide = PtxType(std::string(1, source_type.spelling.front()) + std::to_string(source_type.bits * 2));
    if (!wide) { Fail("unsupported_type", "Unsupported wide result type"); Invalidate(); return; }
    result_type = *wide;
  }
  // Scalar mixed add/sub widen only a; c and d already have f32 type.
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#mixed-precision-floating-point-instructions-add
  const bool mixed_add_sub = In(i.opcode, {"add", "sub"}) && types.size() == 2 &&
      types[0].spelling == "f32" && In(types[1].spelling, {"f16", "bf16"});
  auto destination_text = Trim(module_.view(i.operands.front()));
  const bool destination_vector = destination_text.starts_with('{') && destination_text.ends_with('}');
  if (destination_vector) destination_text = destination_text.substr(1, destination_text.size() - 2);
  auto destinations = Split(destination_text, destination_vector ? ',' : '|');
  if (i.opcode == "selp") {
    if (auto predicate = Known(std::string(module_.view(i.operands[3])), cohort_template)) {
      auto value = Operand(std::string(module_.view(i.operands[*predicate != 0 ? 1 : 2])), result_type);
      if (value) value = Reinterpret(*value, result_type);
      if (value) Bind(destinations.front(), *value); else Invalidate();
      return;
    }
  }
  std::vector<Type> result_types;
  for (const auto &destination : destinations) {
    auto type = result_type;
    if (destination_vector) {
      auto declared = declarations_.find(Resolve(destination, i.scope));
      if (i.opcode == "mov" && destination == "_" &&
          (destinations.size() == 2 || destinations.size() == 4) &&
          result_type.bits % destinations.size() == 0) {
        const auto width = result_type.bits / destinations.size();
        result_types.push_back(*PtxType("b" + std::to_string(width)));
        continue;
      }
      if (i.opcode != "mov" || declared == declarations_.end()) {
        Fail("unsupported_vector_compute", "Only explicit mov register packing/unpacking is supported"); Invalidate(); return;
      }
      type = declared->second;
    } else if (destinations.size() > 1 && i.opcode != "setp") {
      Fail("unsupported_multi_result", "Unsupported multi-destination compute operation"); Invalidate(); return;
    }
    result_types.push_back(type);
  }
  std::vector<Expression> operands;
  std::vector<std::string> labels;
  bool valid = true;
  for (std::size_t n = 1; n < i.operands.size(); ++n) {
    auto text = Trim(module_.view(i.operands[n]));
    auto type = source_type;
    if (i.opcode == "fma" && types.size() == 2 && n <= 2) type = types[1];
    if (mixed_add_sub && n == 1) type = types[1];
    if ((i.opcode == "selp" && n == 3) || (In(i.opcode, {"setp", "set"}) && n == 3)) type = Pred();
    if ((In(i.opcode, {"shl", "shr"}) && n == 2) || (i.opcode == "bfe" && n >= 2) ||
        (i.opcode == "bfi" && n >= 3) || (i.opcode == "lop3" && n == 4)) type = U32();
    if (i.opcode == "mad" && Has(i, "wide") && n == 3) type = result_type;
    if (i.opcode == "slct" && n < 3) type = result_type;
    if (text.starts_with('{') && text.ends_with('}')) {
      if (i.opcode != "mov") { Fail("unsupported_vector_compute", "Non-mov register packs require explicit layout support"); valid = false; continue; }
      const auto parts = Split(std::string_view(text).substr(1, text.size() - 2), ',');
      if ((parts.size() != 2 && parts.size() != 4) || type.bits % parts.size() != 0) {
        Fail("invalid_register_pack", "mov pack needs two or four equal-width components"); Invalidate(); return;
      }
      const auto component_type = PtxType("b" + std::to_string(type.bits / parts.size()));
      if (!component_type) { Fail("invalid_register_pack", "Unsupported mov component width"); Invalidate(); return; }
      for (const auto &part : parts) {
        const auto declared = declarations_.find(Resolve(part, i.scope));
        auto expression = Operand(part, declared == declarations_.end() ? *component_type : declared->second, true);
        if (expression) { operands.push_back(*expression); labels.push_back(part); } else valid = false;
      }
    } else {
      auto expression = Operand(text, type);
      if (expression && i.opcode == "cvt" && expression->type.value.isScalar() &&
          type.value.isScalar() && expression->type.value.element().isBitLike() &&
          type.value.element().isBitLike() && type.spelling != "pred" &&
          expression->type.value.bitSize() > type.value.bitSize()) {
        // https://docs.nvidia.com/cuda/parallel-thread-execution/#operand-size-exceeding-instruction-type-size
        expression = Part(*expression, type, 0);
      } else if (expression && i.opcode!="mov") expression = Reinterpret(*expression, type);
      if (expression && ((i.opcode == "fma" && types.size() == 2 && n <= 2) ||
                         (mixed_add_sub && n == 1)) && type.value != result_type.value) {
        expression = Expression{Add(std::make_unique<v::ConvertOp>(result_type.value), {expression->value}, origin_).result(), result_type};
      }
      if (expression) { operands.push_back(*expression); labels.push_back(text); } else valid = false;
    }
  }
  if (In(i.opcode, {"addc", "subc", "madc"})) {
    if (!carry_) { Fail("missing_carry_definition", "Carry-consuming operation has no recorded carry definition"); valid = false; }
    else { operands.push_back(*carry_); labels.push_back("CC.CF"); }
  }
  if (!valid) { Invalidate(); return; }
  if (i.opcode == "cvt" && result_type.spelling == "ue8m0x2") {
    // The reference conversion (cuda_fp8.hpp, __nv_cvt_float_to_e8m0 and __nv_cvt_bfloat16raw_to_e8m0):
    // .rz keeps the biased exponent field and ignores the sign, so zero and subnormals give 0 (2^-127)
    // and NaN or infinity give 0xFF (NaN); .satfinite turns an infinity into 0xFE. Source a fills the
    // upper byte. .rp rounds |x| up to bf16 (an f32 NaN becomes 0x7FFF; __internal_float_to_bf16raw_ru),
    // then increments the exponent field when the bf16 mantissa is nonzero and 2^-127 < |x| < infinity;
    // .satfinite then turns a non-NaN 0xFF into 0xFE. (.relu is not defined for .ue8m0x2.)
    const bool f32 = source_type.spelling == "f32" && operands.size() == 2;
    const bool up = Has(i, "rp");
    if (!(Has(i, "rz") || up) || Has(i, "relu") || !(f32 || (source_type.spelling == "bf16x2" && operands.size() == 1))) {
      Fail("unsupported_ue8m0_conversion", "Only cvt.{rz,rp}{.satfinite}.ue8m0x2 from .f32 or .bf16x2 has an exact value model");
      Invalidate(); return;
    }
    std::vector<Expression> sources;
    if (f32) sources = {operands[1], operands[0]};
    else for (std::uint32_t n = 0; n < 2; ++n) {
      auto half = Part(operands[0], *PtxType("bf16"), 16 * n);
      if (!half) { Invalidate(); return; }
      sources.push_back(*half);
    }
    const Type element{v::ValueType::scalar(result_type.value.element()), "ue8m0", 8};
    Expression packed{{}, result_type};
    for (const auto &source : sources) {
      auto bits = Bits(source);
      if (!bits) { Invalidate(); return; }
      const auto type = bits->type;
      const auto binary = [&](v::BinaryOpcode opcode, const Expression &a, const Expression &b) {
        return Expression{Add(std::make_unique<v::BinaryOp>(opcode, type.value), {a.value, b.value}, origin_).result(), type};
      };
      const bool wide = type.bits == 32;
      const auto compare = [&](v::ComparePredicate predicate, const Expression &a, const char *b) {
        return Expression{Add(std::make_unique<v::CompareOp>(predicate, Pred().value),
                              {a.value, Constant(type, b).value}, origin_).result(), Pred()};
      };
      const auto select = [&](const Expression &condition, const Expression &a, const Expression &b) {
        return Expression{Add(std::make_unique<v::SelectOp>(type.value),
                              {condition.value, a.value, b.value}, origin_).result(), type};
      };
      // .rp: the bf16 pattern of |x| rounded up (an f32 source; a bf16 source is that pattern),
      // then its exponent field plus one when the mantissa is nonzero and 0x40 < |pattern| < 0x7F80.
      const auto rounded_up = [&] {
        const auto one = Constant(type, "1"), zero = Constant(type, "0");
        auto half = *bits;
        if (wide) {
          const auto magnitude = binary(v::BinaryOpcode::BitAnd, *bits, Constant(type, "2147483647"));
          const auto carry = select(compare(v::ComparePredicate::Less, magnitude, "2139095040"),
              select(compare(v::ComparePredicate::NotEqual, binary(v::BinaryOpcode::BitAnd, magnitude, Constant(type, "65535")), "0"),
                     one, zero), zero);
          half = select(compare(v::ComparePredicate::Greater, magnitude, "2139095040"), Constant(type, "32767"),
                        binary(v::BinaryOpcode::Add, binary(v::BinaryOpcode::ShiftRight, magnitude, Constant(U32(), "16")), carry));
        }
        const auto field = binary(v::BinaryOpcode::BitAnd, half, Constant(type, "32767"));
        const auto increment = select(compare(v::ComparePredicate::NotEqual, binary(v::BinaryOpcode::BitAnd, half, Constant(type, "127")), "0"),
            select(compare(v::ComparePredicate::Greater, field, "64"),
                   select(compare(v::ComparePredicate::Less, field, "32640"), one, zero), zero), zero);
        auto exponent = binary(v::BinaryOpcode::Add, binary(v::BinaryOpcode::BitAnd,
            binary(v::BinaryOpcode::ShiftRight, half, Constant(U32(), "7")), Constant(type, "255")), increment);
        if (Has(i, "satfinite"))
          exponent = select(compare(v::ComparePredicate::Equal, exponent, "255"),
                            select(compare(v::ComparePredicate::LessEqual, field, "32640"), Constant(type, "254"), exponent),
                            exponent);
        return exponent;
      };
      const auto toward_zero = [&] {
        auto exponent = binary(v::BinaryOpcode::BitAnd, binary(v::BinaryOpcode::ShiftRight, *bits, Constant(U32(), wide ? "23" : "7")),
                               Constant(type, "255"));
        if (Has(i, "satfinite")) {
          const auto infinity = Expression{Add(std::make_unique<v::CompareOp>(v::ComparePredicate::Equal, Pred().value),
              {binary(v::BinaryOpcode::BitAnd, *bits, Constant(type, wide ? "2147483647" : "32767")).value,
               Constant(type, wide ? "2139095040" : "32640").value}, origin_).result(), Pred()};
          exponent = Expression{Add(std::make_unique<v::SelectOp>(type.value),
              {infinity.value, Constant(type, "254").value, exponent.value}, origin_).result(), type};
        }
        return exponent;
      };
      const auto exponent = up ? rounded_up() : toward_zero();
      auto byte = Part(exponent, element, 0);
      if (!byte) { Invalidate(); return; }
      packed.components.push_back(*byte);
    }
    Bind(destinations[0], packed); return;
  }
  if (i.opcode == "mov" && (destination_vector ||
      Trim(module_.view(i.operands[1])).starts_with('{'))) {
    std::uint64_t result_bits = 0, source_bits = 0;
    for (const auto &item : result_types) result_bits += item.value.bitSize();
    for (std::size_t n = 0; n < operands.size(); ++n)
      source_bits += operands[n].type.value.bitSize();
    if (result_bits != source_bits || result_bits != types.front().value.bitSize()) {
      Fail("invalid_register_pack", "mov vector components do not exactly cover the PTX bit width"); Invalidate(); return;
    }
  }
  if (i.opcode == "mov") {
    Expression source = operands.front();
    if (operands.size() > 1) source = {{}, types.front(), operands};
    if (destinations.size() == 1) {
      if (source.type.value == result_types[0].value) Bind(destinations[0], source);
      else if (auto result = Reinterpret(source, result_types[0])) Bind(destinations[0], *result);
    } else {
      std::uint32_t bit = 0;
      for (std::size_t n = 0; n < destinations.size(); ++n) {
        if (destinations[n]!="_") {
          auto result = Part(source, result_types[n], bit);
          if (result) Bind(destinations[n], *result); else { Invalidate(); return; }
        }
        bit += static_cast<std::uint32_t>(result_types[n].value.bitSize());
      }
    }
    return;
  }
  if (result_type.elements > 1) {
    if (!In(i.opcode, {"add", "sub", "mul", "fma", "cvt", "min", "max"})) {
      Fail("unsupported_packed_operation", "Packed operation needs a defined scalar component mapping"); Invalidate(); return;
    }
    Expression packed{{}, result_type};
    const auto scalar_result = Type{v::ValueType::scalar(result_type.value.element()), result_type.spelling.substr(0, result_type.spelling.find('x')), result_type.bits};
    for (std::uint32_t n = 0; n < result_type.elements; ++n) {
      std::vector<Expression> components;
      if (i.opcode == "cvt" && source_type.elements == 1 && operands.size() == result_type.elements) {
        // Packed cvt.f16x2/e4m3x2/e5m2x2/e2m1x2.f32 places the
        // second source in the low component (bits [3:0] for e2m1x2).
        // https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt
        components.push_back(operands[result_type.elements - 1 - n]);
      } else for (auto operand : operands) {
        const auto scalar_source = Type{v::ValueType::scalar(operand.type.value.element()), operand.type.spelling.substr(0, operand.type.spelling.find('x')), operand.type.bits};
        auto part = Part(operand, scalar_source, n * operand.type.bits);
        if (!part) { Invalidate(); return; }
        components.push_back(*part);
      }
      packed.components.push_back({Operation("ptx." + i.opcode, "", {scalar_result}, components).result(), scalar_result});
    }
    Bind(destinations[0], packed); return;
  }
  if (In(i.opcode, {"min", "max"}) && operands.size() == 3) {
    auto partial = Expression{Operation("ptx." + i.opcode, "", {result_type}, {operands[0], operands[1]}).result(), result_type};
    operands = {partial, operands[2]};
  }
  if (i.opcode == "cvta") {
    // Explicit symbol/pointer origins already identify the addressed object.
    if (operands[0].value.pointer() && !operands[0].value.pointer()->identity.empty()) {
      Bind(destinations[0], operands[0]); return;
    }
  }
  if (i.opcode == "setp") {
    const std::vector<std::pair<std::string, v::ComparePredicate>> relations{
      {"eq",v::ComparePredicate::Equal},{"ne",v::ComparePredicate::NotEqual},
      {"lt",v::ComparePredicate::Less},{"le",v::ComparePredicate::LessEqual},
      {"gt",v::ComparePredicate::Greater},{"ge",v::ComparePredicate::GreaterEqual},
      {"lo",v::ComparePredicate::Less},{"ls",v::ComparePredicate::LessEqual},
      {"hi",v::ComparePredicate::Greater},{"hs",v::ComparePredicate::GreaterEqual},
      {"num",v::ComparePredicate::Ordered},{"nan",v::ComparePredicate::Unordered}};
    std::optional<v::ComparePredicate> relation;
    bool unordered = false;
    for (const auto& [name, value] : relations) {
      if (Has(i,name)) relation = value;
      if (In(name, {"eq","ne","lt","le","gt","ge"}) && Has(i,name+"u")) { relation = value; unordered = true; }
    }
    if (!relation) { Fail("unsupported_comparison", "Comparison relation is not modeled"); Invalidate(); return; }
    auto p = Expression{Add(std::make_unique<v::CompareOp>(*relation, Pred().value, Has(i,"ftz"), unordered),
        {operands[0].value,operands[1].value}, origin_).result(), Pred()};
    auto combine = [&](Expression condition) {
      if (operands.size() == 2) return condition;
      if (!(Has(i,"and") || Has(i,"or") || Has(i,"xor"))) throw std::invalid_argument("Missing predicate combination operator");
      auto op = Has(i,"and") ? v::BinaryOpcode::BitAnd : Has(i,"or") ? v::BinaryOpcode::BitOr : v::BinaryOpcode::BitXor;
      return Expression{Add(std::make_unique<v::BinaryOp>(op,Pred().value), {condition.value,operands[2].value},origin_).result(),Pred()};
    };
    Bind(destinations[0],combine(p));
    if (destinations.size() == 2) {
      auto q = Expression{Add(std::make_unique<v::UnaryOp>(v::UnaryOpcode::BitNot,Pred().value),{p.value},origin_).result(),Pred()};
      Bind(destinations[1],combine(q));
    }
    return;
  }
  if (Has(i, "cc")) result_types.push_back(Pred());
  std::string mnemonic = i.opcode;
  for (const auto &modifier : i.modifiers) mnemonic += '.' + modifier;
  const auto &node = Operation("ptx." + i.opcode, mnemonic, result_types, operands, std::move(labels));
  for (std::size_t n = 0; n < destinations.size(); ++n) Bind(destinations[n], {node.result(n), result_types[n]});
  if (Has(i, "cc")) carry_ = Expression{node.result(result_types.size() - 1), Pred()};
}

} // namespace
} // namespace sentryward::frontend::ptx
