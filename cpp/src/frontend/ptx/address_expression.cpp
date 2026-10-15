#include "sentryward/frontend/ptx/address_expression.h"
#include "sentryward/value_ir/operations.h"
#include "sentryward/value_ir/type_conversion.h"

#include <optional>

namespace sentryward::frontend::ptx {
namespace {
namespace v = value_ir;

struct Normalizer {
const v::Graph& graph;
const std::function<v::Value(v::Value)>& resolve;
v::Value Resolve(v::Value value) { return resolve ? resolve(value) : value; }
bool Scalar(v::Value value) {
  return value.indices().empty() && !value.threadSelector() && value.type(graph).isScalar();
}

v::Value Unwrap(v::Value value) {
  value = Resolve(value);
  for (unsigned depth = 0; depth < 16 && Scalar(value); ++depth) {
    if (value.conversion()) {
      if (value.conversion()->spec->kind != v::TypeConversionKind::Bitcast) break;
      value = Resolve(value.conversion()->source);
    } else if (dynamic_cast<const v::BitcastOp*>(&graph.definingNode(value).operation())) {
      value = Resolve(graph.instance(value).operands.at(0));
    } else break;
  }
  return value;
}

bool Unsigned(v::Value value, unsigned bits) {
  return Scalar(value) && value.type(graph).element() == v::ElementType::unsignedInteger(bits);
}

std::optional<v::Value> ConversionSource(v::Value value,
                                       unsigned from, unsigned to) {
  value = Unwrap(value);
  if (!Unsigned(value, to)) return {};
  v::Value source;
  if (value.conversion()) {
    if (value.conversion()->spec->kind != v::TypeConversionKind::Convert ||
        value.conversion()->spec->attributes != v::ConversionAttributes{}) return {};
    source = value.conversion()->source;
  } else {
    const auto* op = dynamic_cast<const v::ConvertOp*>(&graph.definingNode(value).operation());
    if (!op || op->attributes() != v::ConversionAttributes{}) return {};
    source = graph.instance(value).operands.at(0);
  }
  source = Resolve(source);
  if (!Unsigned(source, from)) return {};
  return source;
}

std::optional<v::Value> Shift32Source(v::Value value,
                                    v::BinaryOpcode opcode) {
  value = Unwrap(value);
  if (!Unsigned(value, 64) || value.conversion()) return {};
  const auto* op = dynamic_cast<const v::BinaryOp*>(&graph.definingNode(value).operation());
  if (!op || op->opcode() != opcode || op->attributes().saturate) return {};
  const auto operands = graph.instance(value).operands;
  const auto source = Resolve(operands.at(0));
  if (!Unsigned(source, 64)) return {};
  const auto amount = Unwrap(operands.at(1));
  if (!Unsigned(amount, 32) || amount.conversion()) return {};
  const auto* literal = dynamic_cast<const v::ConstantOp*>(&graph.definingNode(amount).operation());
  if (!literal || literal->pointer() || literal->literal() != "32") return {};
  return source;
}

std::optional<v::Value> HalfSource(v::Value value) {
  const auto narrow = ConversionSource(value, 32, 64);
  return narrow ? ConversionSource(*narrow, 64, 32) : std::nullopt;
}
v::Value Run(v::Value value) {
  value = Resolve(value);
  const auto candidate = Unwrap(value);
  if (!Unsigned(candidate, 64) || candidate.conversion()) return value;
  const auto* op = dynamic_cast<const v::BinaryOp*>(&graph.definingNode(candidate).operation());
  if (!op || op->opcode() != v::BinaryOpcode::BitOr || op->attributes().saturate) return value;
  const auto operands = graph.instance(candidate).operands;
  for (unsigned low = 0; low < 2; ++low) {
    const auto original = HalfSource(operands.at(low));
    const auto upper = Shift32Source(operands.at(1 - low), v::BinaryOpcode::ShiftLeft);
    const auto truncated = upper ? HalfSource(*upper) : std::nullopt;
    const auto high_original = truncated ? Shift32Source(*truncated, v::BinaryOpcode::ShiftRight) : std::nullopt;
    if (original && high_original && Unwrap(*original) == Unwrap(*high_original))
      return original->type(graph) == value.type(graph) ? *original :
          graph.bitcast(*original, value.type(graph));
  }
  return value;
}
};
}  // namespace
v::Value NormalizeAddressExpression(const v::Graph& graph, v::Value value,
    const std::function<v::Value(v::Value)>& resolve) {
  return Normalizer{graph, resolve}.Run(value);
}
}  // namespace sentryward::frontend::ptx
