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
bool literal(const RuleGraph& g, v::Value x, std::string_view expected) {
  x = uncast(g, x);
  if (!width(g, x, 32)) return false;
  const auto* c = dynamic_cast<const v::ConstantOp*>(&g.definingNode(x).operation());
  if (!c) return false;
  std::string text = c->literal();
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) { return std::tolower(ch); });
  return text == expected;
}
bool floatBinary(const RuleGraph& g, v::Value x, B opcode, R round) {
  auto* b = binary(g, x, opcode);
  return f32(g, x) && b && plain(b->attributes(), round);
}
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
  if (!floatBinary(g, rounded, B::Add, R::TowardNegative)) return {};
  const auto& roundArgs = g.instance(rounded).operands;
  auto clamped = uncast(g, roundArgs[0]);
  if (!literal(g, roundArgs[1], "0f4b400000") ||
      !floatBinary(g, clamped, B::Maximum, R::None)) return {};
  const auto& clampArgs = g.instance(clamped).operands;
  if (!literal(g, clampArgs[1], "0fc2fe0000") || !f32(g, clampArgs[0])) return {};
  std::optional<v::Value> fraction;
  // Horner polynomial: ((c3*f + c2)*f + c1)*f + 1.
  for (const auto coefficient : {"0f3f800000", "0f3f31f519", "0f3e6906a4"}) {
    const auto* fma = dynamic_cast<const v::FmaOp*>(&g.definingNode(polynomial).operation());
    if (!f32(g, polynomial) || !fma || !plain(fma->attributes(), R::NearestEven)) return {};
    const auto& args = g.instance(polynomial).operands;
    if (!literal(g, args[2], coefficient)) return {};
    const auto f = uncast(g, args[1]);
    if (fraction && *fraction != f) return {};
    fraction = f;
    polynomial = uncast(g, args[0]);
  }
  if (!literal(g, polynomial, "0f3d9df09d") || !fraction ||
      !floatBinary(g, *fraction, B::Subtract, R::NearestEven)) return {};
  const auto& fractionArgs = g.instance(*fraction).operands;
  if (uncast(g, fractionArgs[0]) != clamped) return {};
  auto integral = uncast(g, fractionArgs[1]);
  if (!floatBinary(g, integral, B::Subtract, R::NearestEven)) return {};
  const auto& integralArgs = g.instance(integral).operands;
  if (uncast(g, integralArgs[0]) != rounded || !literal(g, integralArgs[1], "0f4b400000")) return {};
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
  if(changed){++stats.nodes;stats.instances+=graph.node(id).instances().size();}
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
