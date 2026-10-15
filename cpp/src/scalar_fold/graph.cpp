#include "sentryward/scalar_fold/graph.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace sentryward::scalar_fold {
namespace {
void combine(std::size_t& h, std::size_t value) {
  h ^= value + std::size_t{0x9e3779b9} + (h << 6) + (h >> 2);
}
void hashString(std::size_t& h, const std::string& value) {
  combine(h, std::hash<std::string>{}(value));
}
std::size_t hashNode(const Node& n) {
  std::size_t h = static_cast<std::size_t>(n.kind);
  hashString(h, n.type);
  hashString(h, n.name);
  hashString(h, n.attributes);
  combine(h, n.index_level);
  for (const auto& ref : n.operands) {
    combine(h, ref.node);
    hashString(h, ref.conversion);
    hashString(h, ref.converted_type);
    hashString(h, ref.pointer_provenance);
  }
  combine(h, n.fold.has_value());
  if (n.fold) {
    combine(h, static_cast<std::size_t>(n.fold->reducer));
    combine(h, n.fold->binder_level);
    combine(h, std::hash<std::uint64_t>{}(n.fold->count));
  }
  return h;
}
std::vector<Id> reachable(const Graph& graph, std::span<const Ref> roots,std::uint32_t outer_binders,
                          std::span<const std::uint32_t> root_binders) {
  graph.validateRoots(roots,outer_binders,root_binders);
  std::unordered_set<Id> seen;
  std::vector<Id> pending;
  for (const auto& root : roots) pending.push_back(root.node);
  while (!pending.empty()) {
    const auto id = pending.back();
    pending.pop_back();
    if (!seen.insert(id).second) continue;
    for (const auto& child : graph.node(id).operands) pending.push_back(child.node);
  }
  std::vector<Id> result(seen.begin(), seen.end());
  std::sort(result.begin(), result.end());
  return result;
}
}  // namespace

std::size_t RefHash::operator()(const Ref& r) const {
  std::size_t h = std::hash<Id>{}(r.node);
  combine(h, std::hash<std::uint64_t>{}(r.owner));
  hashString(h, r.conversion);
  hashString(h, r.converted_type);
  hashString(h, r.pointer_provenance);
  return h;
}

Graph::Graph() {
  static std::atomic<std::uint64_t> next_owner{1};
  owner_ = next_owner.fetch_add(1, std::memory_order_relaxed);
  if (owner_ == 0) throw std::overflow_error("scalar-fold graph owner overflow");
}
void Graph::validateRef(const Ref& ref) const {
  if (ref.owner != owner_)
    throw std::invalid_argument("scalar-fold reference belongs to another graph");
  if (ref.node >= nodes_.size())
    throw std::out_of_range("scalar-fold reference node does not exist");
  if (ref.conversion.empty() != ref.converted_type.empty())
    throw std::invalid_argument("scalar-fold reference has incomplete conversion");
}
Ref Graph::ref(Id id) const {
  node(id);
  return Ref{id, {}, {}, owner_, {}};
}
const Node& Graph::node(Id id) const {
  if (id >= nodes_.size())
    throw std::out_of_range("scalar-fold node does not exist");
  return nodes_[id];
}
const Node& Graph::node(const Ref& ref) const {
  validateRef(ref);
  return nodes_[ref.node];
}
const std::string& Graph::type(const Ref& ref) const {
  validateRef(ref);
  return ref.converted_type.empty() ? nodes_[ref.node].type : ref.converted_type;
}
std::size_t Graph::collect(std::span<Ref> roots,std::span<std::optional<Id>> weak_nodes,
    const RemapObserver& observer) {
  for(const auto& id:weak_nodes)if(id)node(*id);
  std::vector<bool> live(nodes_.size(),false);
  std::vector<Id> work;
  std::size_t live_count=0;
  for(const auto& root:roots){validateRef(root);work.push_back(root.node);}
  while(!work.empty()) {
    auto id=work.back();work.pop_back();
    if(live[id])continue;
    live[id]=true;
    ++live_count;
    for(const auto& child:nodes_[id].operands)work.push_back(child.node);
  }
  Graph retained;
  if(live_count==nodes_.size()) {
    if(observer) {
      std::vector<Id> identity(nodes_.size());
      for(Id i=0;i<identity.size();++i)identity[i]=i;
      observer(identity,retained.owner_);
    }
    owner_=retained.owner_;
    for(auto& node:nodes_)for(auto& child:node.operands)child.owner=owner_;
    for(auto& root:roots)root.owner=owner_;
    return 0;
  }
  retained.nodes_.reserve(live_count);
  retained.buckets_.reserve(live_count);
  std::vector<Id> map(nodes_.size(),std::numeric_limits<Id>::max());
  const auto remap=[&](Ref value) {
    value.node=map[value.node];value.owner=retained.owner_;return value;
  };
  for(Id id=0;id<nodes_.size();++id)if(live[id]) {
    auto copy=nodes_[id];
    for(auto& child:copy.operands)child=remap(child);
    // Injective ID remapping preserves structural uniqueness.
    const auto next=retained.nodes_.size();
    retained.buckets_[hashNode(copy)].push_back(next);
    retained.nodes_.push_back(std::move(copy));
    map[id]=next;
  }
  const auto removed=nodes_.size()-retained.size();
  for(auto& root:roots)root=remap(root);
  for(auto& id:weak_nodes)if(id) {
    if(live[*id])*id=map[*id];
    else id.reset();
  }
  if(observer)observer(map,retained.owner_);
  *this=std::move(retained);
  return removed;
}

Ref Graph::intern(Node n) {
  for (const auto& child : n.operands) validateRef(child);
  const auto hash = hashNode(n);
  auto& bucket = buckets_[hash];
  for (const auto id : bucket)
    if (nodes_[id] == n) return ref(id);
  const Id id = nodes_.size();
  nodes_.push_back(std::move(n));
  bucket.push_back(id);
  return ref(id);
}
Ref Graph::constant(std::string type, std::string literal) {
  Node n;
  n.kind = Kind::Constant;
  n.type = std::move(type);
  n.name = std::move(literal);
  return intern(std::move(n));
}
Ref Graph::inputParameter(std::string type, std::string symbol) {
  if (symbol.empty()) throw std::invalid_argument("empty scalar parameter symbol");
  Node n;
  n.kind = Kind::Parameter;
  n.type = std::move(type);
  n.name = std::move(symbol);
  return intern(std::move(n));
}
Ref Graph::inputValue(std::string allocation, Ref offset, std::string type) {
  if (allocation.empty()) throw std::invalid_argument("empty initial allocation");
  Node n;
  n.kind = Kind::InitialCell;
  n.type = std::move(type);
  n.name = std::move(allocation);
  n.operands.push_back(std::move(offset));
  return intern(std::move(n));
}
Ref Graph::index(std::uint32_t level) {
  Node n;
  n.kind = Kind::Index;
  n.type = "u64";
  n.index_level = level;
  return intern(std::move(n));
}
Ref Graph::operation(std::string name, std::string type,
                     std::vector<Ref> operands, std::string attributes) {
  if (name.empty()) throw std::invalid_argument("empty scalar operation name");
  Node n;
  n.kind = Kind::Operation;
  n.type = std::move(type);
  n.name = std::move(name);
  n.operands = std::move(operands);
  n.attributes = std::move(attributes);
  return intern(std::move(n));
}
Ref Graph::fold(Reducer reducer, Ref initial, Ref body,
                std::uint32_t binder_level, std::uint64_t count,
                std::string type_name) {
  reducerName(reducer);
  validateRef(initial);
  validateRef(body);
  if ((!type(initial).empty() && !type_name.empty() && type(initial) != type_name) ||
      (!type(body).empty() && !type_name.empty() && type(body) != type_name))
    throw std::invalid_argument("scalar-fold operand and result types differ");
  if (count == 0) return initial;
  Node n;
  n.kind = Kind::Fold;
  n.type = std::move(type_name);
  n.operands = {std::move(initial), std::move(body)};
  n.fold = FoldInfo{reducer, binder_level, count};
  return intern(std::move(n));
}
Ref Graph::convert(Ref value, std::string target_type, std::string step) {
  const auto source_type = type(value);
  if (target_type.empty() || step.empty())
    throw std::invalid_argument("scalar conversion requires type and operation");
  const auto bitWidth = [](const std::string& type) -> unsigned {
    if (type == "bf16") return 16;
    if (type.size() < 2 || std::string("fbsu").find(type[0]) == std::string::npos) return 0;
    unsigned width = 0;
    for (std::size_t i = 1; i < type.size(); ++i) {
      if (type[i] < '0' || type[i] > '9' || width > 1024) return 0;
      width = width * 10 + unsigned(type[i] - '0');
    }
    return width;
  };
  if (step == "Bitcast" && bitWidth(source_type) &&
      bitWidth(source_type) == bitWidth(target_type)) {
    if (source_type == target_type) return value;
    // An initial raw-bit read followed by a floating bitcast is a typed read
    // of those same bytes, not a numeric conversion or an address rewrite.
    // Computed integers, pointers, and floating-to-bits uses keep their edges.
    const auto& producer=node(value);
    const bool floating_read=(producer.type=="b16" && (target_type=="f16" || target_type=="bf16")) ||
      (producer.type=="b32" && target_type=="f32") || (producer.type=="b64" && target_type=="f64");
    if(producer.kind==Kind::InputValue && value.conversion.empty() &&
        value.pointer_provenance.empty() && floating_read)
      return inputValue(producer.name,producer.operands.at(0),std::move(target_type));
    const auto inverse = "Bitcast(" + target_type + " -> " + source_type + ")";
    if (value.conversion == inverse && node(value).type == target_type) {
      value.conversion.clear();
      value.converted_type.clear();
      return value;
    }
    const auto suffix = "; " + inverse;
    if (value.conversion.ends_with(suffix)) {
      value.conversion.resize(value.conversion.size() - suffix.size());
      value.converted_type = std::move(target_type);
      return value;
    }
  }
  if (!value.conversion.empty()) value.conversion += "; ";
  value.conversion += step + "(" + source_type + " -> " + target_type + ")";
  value.converted_type = std::move(target_type);
  return value;
}

Ref Graph::withPointer(Ref value, std::string provenance) {
  validateRef(value);
  value.pointer_provenance = std::move(provenance);
  return value;
}

void Graph::validateRoots(std::span<const Ref> roots, std::uint32_t outer_binders,
                          std::span<const std::uint32_t> root_binders) const {
  if(!root_binders.empty() && root_binders.size()!=roots.size())
    throw std::invalid_argument("root binder count mismatch");
  struct Visit {
    Id id;
    std::uint32_t depth;
    bool operator==(const Visit&) const = default;
  };
  struct Hash {
    std::size_t operator()(const Visit& v) const {
      std::size_t h = v.id;
      combine(h, v.depth);
      return h;
    }
  };
  std::vector<Visit> pending;
  std::unordered_set<Visit, Hash> seen;
  for (std::size_t i=0;i<roots.size();++i) {
    const auto& root=roots[i];
    validateRef(root);
    pending.push_back({root.node, root_binders.empty()?outer_binders:root_binders[i]});
  }
  while (!pending.empty()) {
    const auto v = pending.back();
    pending.pop_back();
    if (!seen.insert(v).second) continue;
    const auto& n = node(v.id);
    if (n.kind == Kind::Index && n.index_level >= v.depth)
      throw std::invalid_argument("free scalar-fold index k" +
                                  std::to_string(n.index_level));
    if (n.kind == Kind::Fold) {
      if (n.fold->binder_level != v.depth)
        throw std::invalid_argument("scalar-fold binder has wrong lexical level");
      if (v.depth == std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("scalar-fold nesting overflow");
      pending.push_back({n.operands[0].node, v.depth});
      pending.push_back({n.operands[1].node, v.depth + 1});
    } else {
      for (const auto& child : n.operands)
        pending.push_back({child.node, v.depth});
    }
  }
}

std::string kindName(Kind kind) {
  switch (kind) {
    case Kind::Constant: return "Constant";
    case Kind::InputParameter: return "InputParameter";
    case Kind::InputValue: return "InputValue";
    case Kind::Index: return "Index";
    case Kind::Operation: return "Operation";
    case Kind::Fold: return "Fold";
  }
  throw std::invalid_argument("invalid scalar-fold node kind");
}
std::string reducerName(Reducer reducer) {
  switch (reducer) {
    case Reducer::Sum: return "Sum";
    case Reducer::Product: return "Product";
    case Reducer::Max: return "Max";
    case Reducer::Min: return "Min";
  }
  throw std::invalid_argument("invalid scalar-fold reducer");
}
Statistics statistics(const Graph& graph, std::span<const Ref> roots,std::uint32_t outer_binders,
                      std::span<const std::uint32_t> root_binders) {
  Statistics s;
  std::unordered_set<Ref, RefHash> bodies;
  std::unordered_map<Id, std::size_t> nesting;
  for (auto id : reachable(graph, roots,outer_binders,root_binders)) {
    const auto& n = graph.node(id);
    ++s.nodes;
    s.edges += n.operands.size();
    std::size_t depth = 0;
    for (const auto& child : n.operands) depth = std::max(depth, nesting.at(child.node));
    switch (n.kind) {
      case Kind::Constant: ++s.constants; break;
      case Kind::Parameter: ++s.parameters; break;
      case Kind::InitialCell: ++s.initial_cells; break;
      case Kind::Index: ++s.indices; break;
      case Kind::Operation: ++s.operations; break;
      case Kind::Fold: {
        ++s.folds;
        ++s.range_descriptions;
        bodies.insert(n.operands[1]);
        depth = std::max(nesting.at(n.operands[0].node),
                         nesting.at(n.operands[1].node) + 1);
        if (n.fold->count > std::numeric_limits<std::uint64_t>::max() -
                            s.logical_iterations) {
          s.logical_iterations = std::numeric_limits<std::uint64_t>::max();
          s.logical_iterations_overflow = true;
        } else {
          s.logical_iterations += n.fold->count;
        }
        break;
      }
    }
    nesting.emplace(id, depth);
    s.max_fold_nesting = std::max(s.max_fold_nesting, depth);
  }
  s.unique_fold_bodies = bodies.size();
  return s;
}
}  // namespace sentryward::scalar_fold
