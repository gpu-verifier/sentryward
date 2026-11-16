#pragma once
#include "value_ir_impl.hpp"

namespace sentryward::frontend::ptx {
namespace {
std::optional<Expression> ValueTemplateExtractor::DemandOperand(const std::string& text, const Type& requested) {
  const auto key = Resolve(text, instruction_->scope);
  const auto type = declarations_.contains(key) ? declarations_.at(key) : requested;
  if (const auto* found = facts_.Value(*record_, text)) {
    // A table atom (class mode) is its plain value here.
    const auto plain = PlainValue(*found);
    const auto* known = &plain;
    if (const auto number = ScalarInteger(*known))
      return Constant(type, number->convert_to<std::string>());
    if (const auto* pointer = std::get_if<AllocationPointer>(known); pointer && type.bits == 64) {
      v::PointerProperties properties{pointer->resource, "global", {}, {}};
      return Expression{Add(std::make_unique<v::ConstantOp>(type.value,
          pointer->offset.convert_to<std::string>(), properties), {}, origin_).result(), type};
    }
  }
  const auto producer = demand_producer(*record_, key);
  if (!producer || !demand_definition || demand_depth >= 128 || graph_.nodes().size() >= 100000) return {};
  const auto cache_key = std::pair{*producer, text};
  if (const auto found = demand_cache.find(cache_key); found != demand_cache.end()) return found->second;
  demand_cache.emplace(cache_key, std::nullopt);
  const auto definition = demand_definition(lane_, *producer);
  if (!definition || definition->predicate_value() == false ||
      !definition->opaque_compute_diamond_indices().empty()) return {};
  const auto& instruction = function_.instructions.at(definition->instruction_id());
  if (Has(instruction, "cc") || In(instruction.opcode, {"addc", "subc", "madc"})) return {};
  auto result = Demand(*definition, text);
  demand_cache[cache_key] = result;
  return result;
}

std::optional<Expression> ValueTemplateExtractor::Demand(const RecordView& record, const std::string& text) {
  struct Restore {
    ValueTemplateExtractor& self;
    std::optional<RecordView> record;
    const Instruction* instruction;
    v::SourceOrigin origin;
    std::map<Key, Expression> registers;
    std::uint64_t suboperation;
    ~Restore() {
      self.record_ = record; self.instruction_ = instruction; self.origin_ = std::move(origin);
      self.registers_ = std::move(registers); self.suboperation_ = suboperation; --self.demand_depth;
    }
  } restore{*this, record_, instruction_, origin_, std::move(registers_), suboperation_};
  ++demand_depth;
  record_ = record;
  instruction_ = &function_.instructions.at(record.instruction_id());
  suboperation_ = 0;
  BeginOrigin(record);
  try {
    if (text.starts_with('[')) return Address(text);
    if (instruction_->opcode == "shfl") Shuffle();
    else if (instruction_->opcode == "ld" || instruction_->opcode == "ldu") Memory();
    else if (instruction_->semantics.category == "compute") Compute();
    else return {};
    const auto found = registers_.find(Resolve(text, instruction_->scope));
    if (found != registers_.end()) return found->second;
  } catch (const std::exception&) {
    // Preserve an opaque address when its dependency slice is unsupported.
  }
  return {};
}

// Explicit DAG serialization avoids duplicating shared subexpressions in JSON.
class AddressDagWriter {
  using Json = nlohmann::json;
  const v::Graph& graph;
  Json nodes = Json::array();
  using Key = std::tuple<std::size_t, std::size_t, std::size_t, const v::TypeConversion*>;
  std::map<Key, std::size_t> ids;
  std::vector<v::Value> retained_values;
  bool canonical_terms = false;
  std::unordered_map<std::string, std::size_t> canonical_ids;
  std::size_t slice_visits = 0;
  std::uint64_t pool_id = 0;
public:
  explicit AddressDagWriter(const v::Graph& graph, bool canonical = false)
      : graph(graph), canonical_terms(canonical) {
    if (canonical) pool_id = algorithm::AnalysisScope::Current().Next(algorithm::AnalysisScope::Ids::kAddressPool);
  }
  std::optional<std::uint64_t> Selector(v::Value value, unsigned depth = 0) const {
    if (depth > 64) return {};
    if (value.threadSelector()) return Selector(Selected(value),depth+1);
    if (!value.indices().empty() ||
        !value.type(graph).isScalar() || !value.type(graph).element().isInteger()) return {};
    const auto bits = value.type(graph).element().width();
    if (!bits || bits > 64) return {};
    const auto mask = bits == 64 ? UINT64_MAX : (std::uint64_t(1) << bits)-1;
    if (value.conversion()) {
      if (value.conversion()->spec->kind != v::TypeConversionKind::Bitcast) return {};
      auto result = Selector(value.conversion()->source, depth+1);
      return result ? std::optional<std::uint64_t>(*result & mask) : std::nullopt;
    }
    const auto& op = graph.definingNode(value).operation();
    if (const auto* literal = dynamic_cast<const v::ConstantOp*>(&op)) {
      auto integer = ParseInteger(literal->literal());
      if (!integer || literal->pointer()) return {};
      return (*integer & Integer(mask)).convert_to<std::uint64_t>();
    }
    const auto instance = graph.instance(value);
    if (const auto* coordinate = dynamic_cast<const v::CoordinateOp*>(&op)) {
      const auto axis = static_cast<unsigned>(coordinate->axis());
      if (coordinate->kind() == v::CoordinateKind::ThreadIndex && instance.origin.thread_coordinates)
        return instance.origin.thread_coordinates->at(axis);
      if (coordinate->kind() == v::CoordinateKind::BlockIndex && instance.origin.cta_coordinates)
        return instance.origin.cta_coordinates->at(axis);
      return {};
    }
    if (const auto* binary = dynamic_cast<const v::BinaryOp*>(&op)) {
      if (binary->attributes().saturate) return {};
      const auto a = Selector(instance.operands.at(0),depth+1);
      const auto b = Selector(instance.operands.at(1),depth+1);
      if (!a || !b) return {};
      switch (binary->opcode()) {
        case v::BinaryOpcode::Add: return (*a + *b) & mask;
        case v::BinaryOpcode::Subtract: return (*a - *b) & mask;
        case v::BinaryOpcode::BitAnd: return *a & *b & mask;
        case v::BinaryOpcode::BitOr: return (*a | *b) & mask;
        case v::BinaryOpcode::BitXor: return (*a ^ *b) & mask;
        default: return {};
      }
    }
    return {};
  }
  v::Value Selected(v::Value value) const {
    if (!value.threadSelector()) return value;
    const auto thread = Selector(*value.threadSelector());
    if (!thread) throw std::invalid_argument("address thread selector is not constant");
    if (!value.indices().empty()) throw std::invalid_argument("indexed address thread selection");
    v::Value result;
    if (value.conversion()) {
      const auto& conversion = *value.conversion();
      auto source = Selected(graph.fromThread(conversion.source,*value.threadSelector()));
      result = conversion.spec->kind == v::TypeConversionKind::Bitcast
          ? graph.bitcast(source,conversion.spec->target)
          : graph.convert(source,conversion.spec->target,conversion.spec->attributes);
    } else {
      const auto& node = graph.definingNode(value);
      const auto member = node.memberIndex(*thread);
      if (!member) throw std::invalid_argument("address selected thread is absent");
      result = node.result(value.resultIndex(),*member);
      // A ThreadForward instance is that source thread's own value.
      if (dynamic_cast<const v::ThreadForwardOp*>(&node.operation())) {
        if (value.pointer())
          throw std::invalid_argument("an address forwarded from another thread carries a pointer annotation");
        return Selected(graph.forwarded(result));
      }
    }
    if (value.pointer()) result = graph.asPointer(result,*value.pointer());
    return result;
  }
  v::Value Normalize(v::Value value) const {
    return NormalizeAddressExpression(graph,value,[&](v::Value x) { return Selected(x); });
  }
  std::optional<std::string> Allocation(v::Value value, unsigned depth = 0) const {
    value = Normalize(value);
    if (depth > 128 || value.type(graph).bitSize() != 64) return {};
    if (value.conversion()) {
      if (value.conversion()->spec->kind != v::TypeConversionKind::Bitcast) return {};
      return Allocation(value.conversion()->source, depth + 1);
    }
    const auto& op = graph.definingNode(value).operation();
    if (dynamic_cast<const v::BitcastOp*>(&op))
      return Allocation(graph.instance(value).operands.at(0), depth + 1);
    if (const auto* constant = dynamic_cast<const v::ConstantOp*>(&op)) {
      if (constant->pointer() && constant->pointer()->addressing == "global") return constant->pointer()->identity;
    }
    if (const auto* input = dynamic_cast<const v::InputParameterOp*>(&op)) {
      if (input->pointer() && input->pointer()->addressing == "global") return input->pointer()->identity;
    }
    // cvta changes the address representation, not its originating object.
    // Retain the original DAG for offset reasoning: do not erase the conversion
    // or infer an object for an untyped integer / loaded pointer value.
    if (const auto* conversion = dynamic_cast<const v::PtxScalarOp*>(&op);
        conversion && conversion->opcode() == v::PtxScalarOpcode::AddressSpaceConvert &&
        !conversion->attributes().carry_in && !conversion->attributes().carry_out &&
        (conversion->attributes().modifiers == std::vector<std::string>{"to", "global", "u64"} ||
         conversion->attributes().modifiers == std::vector<std::string>{"global", "u64"})) {
      const auto operands = graph.instance(value).operands;
      if (operands.size() == 1 && operands[0].type(graph).bitSize() == 64)
        return Allocation(operands[0], depth + 1);
    }
    if (const auto* binary = dynamic_cast<const v::BinaryOp*>(&op);
        binary && (binary->opcode() == v::BinaryOpcode::Add || binary->opcode() == v::BinaryOpcode::Subtract)) {
      const auto operands = graph.instance(value).operands;
      const auto a = Allocation(operands[0],depth+1), b = Allocation(operands[1],depth+1);
      if (a && !b) return a;
      if (b && !a && binary->opcode() == v::BinaryOpcode::Add) return b;
    }
    return {};
  }
  std::size_t Node(v::Value value) {
    value = Normalize(value);
    if (!value.indices().empty() || value.threadSelector() || !value.type(graph).isScalar())
      throw std::invalid_argument("non-scalar address value");
    if (value.conversion() && value.conversion()->spec->kind == v::TypeConversionKind::Bitcast)
      return Node(value.conversion()->source);
    if (!value.conversion() && dynamic_cast<const v::BitcastOp*>(&graph.definingNode(value).operation()))
      return Node(graph.instance(value).operands.at(0));
    const Key key{value.nodeId(), value.instanceIndex(), value.resultIndex(), value.conversion().get()};
    if (const auto found = ids.find(key); found != ids.end()) return found->second;
    if (++slice_visits > 4096) throw std::invalid_argument("address slice exceeds serialization budget");
    retained_values.push_back(value);
    const auto& definition = graph.definingNode(value);
    const auto instance = graph.instance(value);
    Json node{{"op", std::string(definition.operation().name())}, {"type", value.type(graph).toString()},
              {"result", value.resultIndex()}, {"operands", Json::array()}, {"operand_types", Json::array()}};
    if (value.conversion()) {
      node["op"] = "TypeConversion";
      node["attributes"] = v::typeConversionLabel(value);
      node["operands"].push_back(Node(value.conversion()->source));
      node["operand_types"].push_back(value.conversion()->source.type(graph).toString());
    } else {
      std::ostringstream attributes;
      definition.operation().printAttributes(attributes);
      node["attributes"] = attributes.str();
      for (const auto& operand : instance.operands) {
        node["operands"].push_back(Node(operand));
        node["operand_types"].push_back(operand.type(graph).toString());
      }
      if (dynamic_cast<const v::CoordinateOp*>(&definition.operation())) {
        const auto* coordinate = dynamic_cast<const v::CoordinateOp*>(&definition.operation());
        const auto axis = static_cast<unsigned>(coordinate->axis());
        if (coordinate->kind() == v::CoordinateKind::ThreadIndex)
          node["binding"] = instance.origin.thread_coordinates->at(axis);
        else if (coordinate->kind() == v::CoordinateKind::BlockIndex)
          node["binding"] = instance.origin.cta_coordinates->at(axis);
        else throw std::invalid_argument("unbound cluster coordinate");
      }
      if (definition.operation().name() == "Load")
        node["occurrence"] = {instance.origin.source_file->str(), instance.origin.function->str(),
            instance.origin.cta_coordinates.value(), instance.origin.thread.value(),
            instance.origin.instruction_index.value(), instance.origin.occurrence.value(), instance.origin.lowering_index};
    }
    if (canonical_terms) {
      const auto id = canonical_ids.try_emplace(node.dump(), canonical_ids.size()).first->second;
      ids.emplace(key, id);
      return id;
    }
    const auto id = nodes.size();
    nodes.push_back(std::move(node)); ids.emplace(key, id);
    return id;
  }
  Json Write(v::Value value) {
    slice_visits = 0;
    const auto root = Node(value);
    if (canonical_terms)
      return {{"offset", {{"operator", "opaque"},
          {"provenance", Json::array({"canonical_value_dag", value.graphIdentity(), pool_id, root})}}}};
    Json result{{"offset", {{"operator", "value_dag"}, {"nodes", std::move(nodes)}, {"root", root}}}};
    if (const auto allocation = Allocation(value)) result["allocation"] = *allocation;
    return result;
  }
};
} // namespace
} // namespace sentryward::frontend::ptx
