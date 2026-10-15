#pragma once
#include "core.hpp"
namespace rg {
inline uint64_t input_unsigned(const J &value) {
  if (value.is_number_unsigned())
    return value.get<uint64_t>();
  if (value.is_number_integer()) {
    int64_t n = value.get<int64_t>();
    if (n >= 0)
      return static_cast<uint64_t>(n);
  }
  throw Unsupported("node/binder ID or count must be an unsigned integer");
}
inline Id input_id(const J &value) {
  uint64_t n = input_unsigned(value);
  if (n >= invalid)
    throw Unsupported("node/binder ID outside supported range");
  return static_cast<Id>(n);
}

struct Edge {
  Id node;
  std::string conversion;
  std::string converted_type, pointer_provenance;
  Edge(Id id, std::string step={}, std::string type={}, std::string pointer={})
      : node(id), conversion(std::move(step)), converted_type(std::move(type)), pointer_provenance(std::move(pointer)) {}
  explicit Edge(const J &e)
      : node(input_id(e.at("node"))), conversion(e.value("conversion", "")),
        converted_type(e.value("converted_type", "")), pointer_provenance(e.value("pointer_provenance", "")) {}
};
struct RawNode {
  RawNode() = default;
  std::string kind, name, type, attributes;
  std::vector<Edge> operands;
  Id level = 0;
  uint64_t count = 0;
  std::string reducer;
  explicit RawNode(const J &n)
      : kind(n.at("kind")), name(n.value("name", "")),
        type(n.value("type", "")), attributes(n.value("attributes", "")) {
    if (n.contains("operands")) {
      operands.reserve(n["operands"].size());
      for (const auto &e : n["operands"])
        operands.emplace_back(e);
    }
    if (kind == "Index")
      level = input_id(n.at("index_level"));
    if (kind == "Fold") {
      const auto &f = n.at("fold");
      level = input_id(f.at("binder_level"));
      count = input_unsigned(f.at("count"));
      reducer = f.at("reducer");
      if (reducer != "Sum" && reducer != "Max")
        throw Unsupported("unsupported Fold reducer " + reducer);
    }
  }
};
struct Document {
  J meta;
  std::vector<RawNode> nodes;
  Document() = default;
  Document(const J &d) : meta(d) {
    nodes.reserve(d.at("nodes").size());
    for (const auto &n : d.at("nodes"))
      nodes.emplace_back(n);
    meta.erase("nodes");
  }
  // Parser callback retains at most one raw-node JSON object. Raw expression
  // storage is O(nodes + operands + strings), without a second full JSON DOM.
  static Document load(std::istream &in) {
    Document d;
    std::string rootkey;
    bool in_nodes = false, saw_nodes = false;
    auto callback = [&](int depth, J::parse_event_t event, J &parsed) {
      if (depth == 1 && event == J::parse_event_t::key)
        rootkey = parsed.get<std::string>();
      if (depth == 1 && event == J::parse_event_t::array_start &&
          rootkey == "nodes") {
        in_nodes = true;
        saw_nodes = true;
      }
      if (in_nodes && depth == 2 && event == J::parse_event_t::object_end) {
        d.nodes.emplace_back(parsed);
        return false;
      }
      if (in_nodes && depth == 1 && event == J::parse_event_t::array_end) {
        in_nodes = false;
        return false;
      }
      return true;
    };
    d.meta = J::parse(in, callback);
    d.meta.erase("nodes");
    if (!saw_nodes)
      throw Unsupported("missing expression nodes array");
    return d;
  }
};
} // namespace rg
