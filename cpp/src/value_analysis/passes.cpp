#include "sentryward/value_analysis/passes.h"
#include "sentryward/algorithm/diagnostics/performance_profile.h"
#include "sentryward/algorithm/diagnostics/compression_audit.h"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include "sentryward/value_ir/operations.h"
#include "sentryward/value_ir/ptx_operation.h"

namespace sentryward::value_analysis {
namespace {
namespace v = value_ir;
using B = v::BinaryOpcode;
using R = v::RoundingMode;

struct RuleGraph final {
  const v::Graph& graph;
  v::LocalRewriteMatch& match;
  mutable std::vector<std::pair<v::Value,std::uint64_t>> masks{};
  const v::ValueType& type(v::Value x) const { return graph.type(x); }
  const v::Node& definingNode(v::Value x) const { return graph.definingNode(x); }
  const v::Instance& instance(v::Value x) const { return match.instance(x); }
};

bool scalar(const RuleGraph& g, v::Value x) {
  return !x.conversion() && x.indices().empty() && !x.threadSelector() && !x.pointer() &&
      g.type(x).isScalar() && !g.type(x).element().isParameterized();
}
bool width(const RuleGraph& g, v::Value x, unsigned bits) {
  return scalar(g, x) && g.type(x).bitSize() == bits;
}
v::Value uncast(const RuleGraph& g, v::Value x) {
  for (unsigned depth = 0; depth < 32 && scalar(g, x); ++depth) {
    if (!dynamic_cast<const v::BitcastOp*>(&g.definingNode(x).operation())) break;
    const auto source = g.instance(x).operands.at(0);
    if (!scalar(g, source) || g.type(source).bitSize() != g.type(x).bitSize()) break;
    x = source;
  }
  return x;
}
const v::BinaryOp* binary(const RuleGraph& g, v::Value x, B op) {
  const auto* node = dynamic_cast<const v::BinaryOp*>(&g.definingNode(x).operation());
  return scalar(g, x) && node && node->opcode() == op ? node : nullptr;
}
std::optional<std::uint64_t> integer(const RuleGraph& g, v::Value x) {
  x = uncast(g, x);
  const auto* c = dynamic_cast<const v::ConstantOp*>(&g.definingNode(x).operation());
  if (!c || !scalar(g, x) || !g.type(x).element().isBitLike()) return {};
  auto text = std::string_view(c->literal());
  int base = 10;
  if (text.starts_with("0x") || text.starts_with("0X")) { text.remove_prefix(2); base = 16; }
  std::uint64_t value;
  auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value, base);
  if (error != std::errc{} || end != text.data() + text.size()) return {};
  return value;
}
std::uint64_t full(unsigned bits) {
  return bits == 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << bits) - 1;
}
bool plain(const v::ArithmeticAttributes& a, R rounding = R::None) {
  return a.rounding == rounding && a.multiply_mode == v::IntegerMultiplyMode::Low &&
      !a.saturate && !a.flush_to_zero && !a.propagate_nan;
}

// An over-approximation of the potentially nonzero bits.
std::uint64_t mask(const RuleGraph&, v::Value, unsigned depth = 0);
std::uint64_t computeMask(const RuleGraph& g, v::Value x, unsigned depth) {
  const auto& type = g.type(x);
  if (!type.isScalar() || type.element().isParameterized() ||
      type.bitSize() == 0 || type.bitSize() > 64) return ~std::uint64_t{0};
  const auto all = full(type.bitSize());
  if (!scalar(g, x) || depth > 32) return all;
  if (auto c = integer(g, x)) return *c & all;
  const auto& op = g.definingNode(x).operation();
  const auto& args = g.instance(x).operands;
  if (dynamic_cast<const v::BitcastOp*>(&op)) return mask(g, args[0], depth + 1) & all;
  if (const auto* convert = dynamic_cast<const v::ConvertOp*>(&op)) {
    const auto& a = convert->attributes();
    const auto& source = g.type(args[0]);
    if (source.isScalar() && !source.element().isParameterized() &&
        source.element().kind() == v::ElementKind::UnsignedInteger &&
        type.element().kind() == v::ElementKind::UnsignedInteger &&
        !a.saturate && !a.saturate_finite && !a.flush_to_zero && a.rounding == R::None)
      return mask(g, args[0], depth + 1) & all;
  }
  const auto* b = dynamic_cast<const v::BinaryOp*>(&op);
  if (!b || !plain(b->attributes())) return all;
  if (b->opcode() == B::BitOr)
    return (mask(g, args[0], depth + 1) | mask(g, args[1], depth + 1)) & all;
  if (b->opcode() == B::ShiftLeft)
    if (auto shift = integer(g, args[1]); shift && *shift < type.bitSize())
      return (mask(g, args[0], depth + 1) << *shift) & all;
  return all;
}
std::uint64_t mask(const RuleGraph& g, v::Value x, unsigned depth) {
  for(const auto& [value,bits]:g.masks)if(value==x)return bits;
  const auto bits=computeMask(g,x,depth);
  g.masks.emplace_back(x,bits);
  return bits;
}
void packingShifts(const RuleGraph& g, v::Value x, std::set<std::size_t>& result,
                   std::vector<v::Value>& visited, unsigned depth = 0) {
  if (depth > 32 || !scalar(g, x)) return;
  if(std::ranges::find(visited,x)!=visited.end())return;
  visited.push_back(x);
  const auto& op = g.definingNode(x).operation();
  if (const auto* b = dynamic_cast<const v::BinaryOp*>(&op)) {
    if (b->opcode() == B::ShiftLeft) { result.insert(x.nodeId()); return; }
    if (b->opcode() != B::BitOr) return;
  } else if (!dynamic_cast<const v::BitcastOp*>(&op) && !dynamic_cast<const v::ConvertOp*>(&op)) return;
  for (auto a : g.instance(x).operands) packingShifts(g, a, result, visited, depth + 1);
}
bool f32(const RuleGraph& g, v::Value x) {
  const auto& type = g.type(x);
  return type.isScalar() && type.element().kind() == v::ElementKind::Floating && type.bitSize() == 32;
}
// The `bits`-bit word a literal spells: a PTX binary32 hex literal 0fXXXXXXXX (32 bits only) or, for a
// constant of an integer or bit type (`integral`), a 0x hexadecimal or a decimal integer (with an
// optional '-') that fits the word as an unsigned or a signed value. A decimal literal of a floating
// type denotes a number, not a word: none. None for any other spelling.
std::optional<std::uint64_t> literalWord(std::string_view text, unsigned bits, bool integral) {
  int base = 10;
  bool negative = false;
  if (bits == 32 && text.size() == 10 && (text.starts_with("0f") || text.starts_with("0F"))) {
    text.remove_prefix(2); base = 16;
  } else if (!integral) {
    return {};
  } else if (text.starts_with("0x") || text.starts_with("0X")) {
    text.remove_prefix(2); base = 16;
  } else if (text.starts_with('-')) {
    text.remove_prefix(1); negative = true;
  }
  std::uint64_t value;
  auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value, base);
  if (text.empty() || error != std::errc{} || end != text.data() + text.size()) return {};
  const std::uint64_t all = full(bits);
  if (!negative) return value <= all ? std::optional(value) : std::nullopt;
  const std::uint64_t magnitude = std::uint64_t{1} << (bits - 1);  // -2^(bits-1) is the least signed value
  if (value > magnitude) return {};
  return (~value + 1) & all;
}
// The 32-bit word of a constant operand: a 32-bit Constant (seen through same-width bitcasts), or one
// half of a 64-bit Constant as the frontend spells a lane of a packed .f32x2 operand given as a 64-bit
// immediate: Convert to u32 of the constant (low half) or of the constant shifted right by 32 (high
// half). None for anything else. O(1).
std::optional<std::uint32_t> word32(const RuleGraph& g, v::Value x) {
  x = uncast(g, x);
  if (!scalar(g, x)) return {};
  const auto* constant = dynamic_cast<const v::ConstantOp*>(&g.definingNode(x).operation());
  if (constant) {
    if (g.type(x).bitSize() != 32 || constant->pointer()) return {};
    const auto word = literalWord(constant->literal(), 32, g.type(x).element().isBitLike());
    return word ? std::optional(std::uint32_t(*word)) : std::nullopt;
  }
  const auto* convert = dynamic_cast<const v::ConvertOp*>(&g.definingNode(x).operation());
  if (!convert || !width(g, x, 32) || g.type(x).element().kind() != v::ElementKind::UnsignedInteger ||
      !(convert->attributes() == v::ConversionAttributes{})) return {};
  auto source = uncast(g, g.instance(x).operands.at(0));
  unsigned shift = 0;
  if (const auto* right = binary(g, source, B::ShiftRight)) {
    if (!plain(right->attributes()) || !width(g, source, 64) ||
        integer(g, g.instance(source).operands.at(1)) != 32) return {};
    shift = 32;
    source = uncast(g, g.instance(source).operands.at(0));
  }
  if (!width(g, source, 64)) return {};
  const auto* wide = dynamic_cast<const v::ConstantOp*>(&g.definingNode(source).operation());
  if (!wide || wide->pointer()) return {};
  const auto word = literalWord(wide->literal(), 64, g.type(source).element().isBitLike());
  return word ? std::optional(std::uint32_t(*word >> shift)) : std::nullopt;
}
bool literal(const RuleGraph& g, v::Value x, std::uint32_t expected) {
  return word32(g, x) == expected;
}
// An f32 operation `opcode` with this rounding and flush-to-zero flag, otherwise plain.
bool floatBinary(const RuleGraph& g, v::Value x, B opcode, R round, bool ftz = false) {
  auto* b = binary(g, x, opcode);
  if (!f32(g, x) || !b) return false;
  auto attributes = b->attributes();
  if (attributes.flush_to_zero != ftz) return false;
  attributes.flush_to_zero = false;
  return plain(attributes, round);
}
// The software exp2 routine (FlashAttention-4's exp2 emulation), lane by lane:
//   x' = max(x, -127); t = add.rm(x', 1.5*2^23); j = t - 1.5*2^23; f = x' - j;
//   p = ((c3*f + c2)*f + c1)*f + 1 (RN FMAs; c3 = 0f3D9DF09D, c2 = 0f3E6906A4, c1 = 0f3F31F519);
//   result = bitcast_f32((bits(t) << 23) + bits(p)).
// Constants are matched by their binary32 words: hex literals, integer literals of the same word, or
// halves of a 64-bit immediate (word32). The rounding add, both subtractions and the three FMAs carry
// .ftz all together or not at all; the clamp is a plain max. Returns x: the pass replaces the result
// by ideal Exp2(x), so the polynomial error and the clamp region x < -127 are the abstraction the pass
// report names ("matched approximation and clamp replaced by ideal Exp2"). Fixed-size pattern.
std::optional<v::Value> exp2Input(const RuleGraph& g, v::Value result) {
  if (!f32(g, result)) return {};
  auto sum = uncast(g, result);
  auto* add = binary(g, sum, B::Add);
  if (!width(g, sum, 32) || !g.type(sum).element().isInteger() || !add || !plain(add->attributes())) return {};
  const auto& sumArgs = g.instance(sum).operands;
  auto shifted = uncast(g, sumArgs[0]), polynomial = uncast(g, sumArgs[1]);
  auto* shift = binary(g, shifted, B::ShiftLeft);
  if (!shift || !width(g, shifted, 32) || !plain(shift->attributes())) return {};
  const auto& shiftArgs = g.instance(shifted).operands;
  if (integer(g, shiftArgs[1]) != 23) return {};
  auto rounded = uncast(g, shiftArgs[0]);
  auto* rounding = binary(g, rounded, B::Add);
  if (!rounding) return {};
  const bool ftz = rounding->attributes().flush_to_zero;
  if (!floatBinary(g, rounded, B::Add, R::TowardNegative, ftz)) return {};
  const auto& roundArgs = g.instance(rounded).operands;
  auto clamped = uncast(g, roundArgs[0]);
  if (!literal(g, roundArgs[1], 0x4b400000u) ||
      !floatBinary(g, clamped, B::Maximum, R::None)) return {};
  const auto& clampArgs = g.instance(clamped).operands;
  if (!literal(g, clampArgs[1], 0xc2fe0000u) || !f32(g, clampArgs[0])) return {};
  std::optional<v::Value> fraction;
  // Horner polynomial: ((c3*f + c2)*f + c1)*f + 1.
  for (const auto coefficient : {0x3f800000u, 0x3f31f519u, 0x3e6906a4u}) {
    const auto* fma = dynamic_cast<const v::FmaOp*>(&g.definingNode(polynomial).operation());
    if (!f32(g, polynomial) || !fma || fma->attributes().flush_to_zero != ftz) return {};
    auto attributes = fma->attributes();
    attributes.flush_to_zero = false;
    if (!plain(attributes, R::NearestEven)) return {};
    const auto& args = g.instance(polynomial).operands;
    if (!literal(g, args[2], coefficient)) return {};
    const auto f = uncast(g, args[1]);
    if (fraction && *fraction != f) return {};
    fraction = f;
    polynomial = uncast(g, args[0]);
  }
  if (!literal(g, polynomial, 0x3d9df09du) || !fraction ||
      !floatBinary(g, *fraction, B::Subtract, R::NearestEven, ftz)) return {};
  const auto& fractionArgs = g.instance(*fraction).operands;
  if (uncast(g, fractionArgs[0]) != clamped) return {};
  auto integral = uncast(g, fractionArgs[1]);
  if (!floatBinary(g, integral, B::Subtract, R::NearestEven, ftz)) return {};
  const auto& integralArgs = g.instance(integral).operands;
  if (uncast(g, integralArgs[0]) != rounded || !literal(g, integralArgs[1], 0x4b400000u)) return {};
  return clampArgs[0];
}
bool rewrite(v::Graph& graph,std::size_t id,std::unique_ptr<const v::Operation> operation,
             const v::Graph::LocalRewrite& match,ValuePassStatistics& stats) {
  [[maybe_unused]] v::LocalRewriteStatistics local;
  v::LocalRewriteStatistics* observed=nullptr;
  SENTRYWARD_AUDIT((void)audit; observed=&local);
  const bool changed=graph.rewritePure(id,std::move(operation),match,observed);
  SENTRYWARD_AUDIT(audit->add(AuditCounter::rewrite_patterns,local.patterns);
    audit->add(AuditCounter::rewrite_local_operand_reads,local.operand_reads);
    audit->add(AuditCounter::rewrite_budget_rejections,local.budget_rejections);
    audit->add(AuditCounter::rewrite_domain_rejections,local.domain_rejections));
  if(changed){++stats.nodes;stats.instances+=graph.node(id).instanceCount();}
  return changed;
}
}  // namespace

ValuePassStatistics rewritePacking(v::Graph& graph) {
  SENTRYWARD_AUDIT_STAGE("packing");
  ValuePassStatistics stats;
  std::set<std::size_t> shifts;
  for (const auto& node : graph.nodes()) {
    const auto* op = dynamic_cast<const v::BinaryOp*>(&node.operation());
    if (!op || op->opcode() != B::BitOr || !plain(op->attributes())) continue;
    const auto& type=graph.type(node.result());
    // Validate the rule's result domain before constructing its replacement.
    // Raw PTX bit/predicate values are not numeric Add operands.
    if(!type.isScalar() || type.element().kind()!=v::ElementKind::UnsignedInteger ||
        type.bitSize()>64)continue;
    std::set<std::size_t> local;
    if(rewrite(graph,node.id(),std::make_unique<v::BinaryOp>(B::Add,graph.type(node.result())),
        [&](v::LocalRewriteMatch& match)->std::optional<std::vector<v::Value>> {
      RuleGraph g{graph,match};const auto value=match.root();
      const auto& type = graph.type(value);
      const auto& args = match.instance(value).operands;
      if (!scalar(g, value) || type.element().kind() != v::ElementKind::UnsignedInteger ||
          type.bitSize() > 64 || (mask(g, args[0]) & mask(g, args[1])) != 0)return {};
      std::vector<v::Value> visited;
      packingShifts(g,args[0],local,visited);packingShifts(g,args[1],local,visited);
      if(local.empty())return {};
      return args;
    },stats))
      shifts.insert(local.begin(), local.end());
  }
  std::map<std::pair<unsigned, std::uint64_t>, v::Value> powers;
  for (auto id : shifts) {
    const auto& node = graph.node(id);
    const auto& type = graph.type(node.result());
    const auto* op = dynamic_cast<const v::BinaryOp*>(&node.operation());
    if (!op || !plain(op->attributes()) || !type.isScalar() ||
        type.element().kind() != v::ElementKind::UnsignedInteger || type.bitSize() > 64) continue;
    rewrite(graph,id,std::make_unique<v::BinaryOp>(B::Multiply,type),
        [&](v::LocalRewriteMatch& match)->std::optional<std::vector<v::Value>> {
      RuleGraph g{graph,match};const auto& instance=match.instance(match.root());
      auto amount=integer(g,instance.operands[1]);
      if(!amount || *amount>=type.bitSize())return {};
      const auto key = std::pair<unsigned, std::uint64_t>{type.bitSize(), *amount};
      auto found = powers.find(key);
      if (found == powers.end()) {
        auto power = graph.add(std::make_unique<v::ConstantOp>(type,
            std::to_string(std::uint64_t{1} << *amount))).result();
        found = powers.emplace(key, power).first;
      }
      return std::vector<v::Value>{instance.operands[0],found->second};
    },stats);
  }
  return stats;
}

ValuePassStatistics abstractSoftwareExp2(v::Graph& graph) {
  SENTRYWARD_AUDIT_STAGE("software_exp2");
  ValuePassStatistics stats;
  for (const auto& node : graph.nodes()) {
    const auto id = node.id();
    if (!dynamic_cast<const v::BitcastOp*>(&node.operation())) continue;
    auto type = graph.type(node.result());
    if(!type.isScalar() || type.element().isParameterized() ||
        type.element().kind()!=v::ElementKind::Floating || type.bitSize()!=32)continue;
    rewrite(graph,id,std::make_unique<v::IdealExp2Op>(type),
        [&](v::LocalRewriteMatch& match)->std::optional<std::vector<v::Value>> {
      RuleGraph g{graph,match};auto input=exp2Input(g,match.root());
      if(!input)return {};
      return std::vector<v::Value>{*input};
    },stats);
  }
  return stats;
}

ValuePipelineResult runValuePasses(v::Graph& graph,
    MemoryActionRange actions, const CheckedOrderContract& contract,
    const std::function<void(std::string_view)>& before_pass) {
  if (!contract.permitsElimination())
    throw std::invalid_argument("Value passes require checked or explicit trusted-reference order");
  ValuePipelineResult result;
  if (before_pass) before_pass("packing");
  result.packing = algorithm::ProfileCall("value_analysis.packing",[&]{return rewritePacking(graph);});
  if (before_pass) before_pass("software_exp2");
  result.ideal_exp2 = algorithm::ProfileCall("value_analysis.software_exp2",[&]{return abstractSoftwareExp2(graph);});
  if (before_pass) before_pass("type_edges");
  result.type_edges = algorithm::ProfileCall("value_analysis.type_edges",[&]{return graph.lowerTypeConversions();});
  if (before_pass) before_pass("memory_elimination");
  result.memory = algorithm::ProfileCall("value_analysis.memory_elimination",[&]{return eliminateMemory(graph, actions, contract);});
  if (result.memory.complete) {
    if (before_pass) before_pass("view_normalization");
    result.views = algorithm::ProfileCall("value_analysis.view_normalization",[&]{return normalizeMemoryViews(graph, result.memory);});
  }
  return result;
}
}  // namespace sentryward::value_analysis
