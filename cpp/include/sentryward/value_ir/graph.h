#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <deque>
#include <memory>
#include <map>
#include <optional>
#include <span>
#include <ranges>
#include <stdexcept>
#include <vector>
#include <iosfwd>

#include "sentryward/value_ir/operation.h"
#include "sentryward/value_ir/origin.h"
#include "sentryward/value_ir/type_conversion.h"
#include "sentryward/value_ir/thread_members.h"

namespace sentryward::value_ir {

using ThreadGroupId = std::size_t;

// Members share an extraction prefix, not an execution time or memory value.
struct ThreadGroup final {
  ThreadGroupId id;
  std::optional<ThreadGroupId> parent;
  std::uint64_t cta;
  ThreadMembers threads;
};

struct ThreadTemplate final {
  std::vector<SourceOrigin> threads;
  ThreadMembers instances, members;
  std::optional<ThreadGroupId> cursor;
};

struct InstanceTypes final {
  std::vector<ValueType> results;
  std::vector<ValueType> elements;
};

struct Instance final {
  SourceOrigin origin;
  std::vector<Value> operands;
  std::span<const ValueType> result_types;
  std::span<const ValueType> element_types;
};

class InstanceView final : public std::ranges::transform_view<
    std::ranges::iota_view<std::size_t, std::size_t>, std::function<Instance(std::size_t)>> {
  using Base = std::ranges::transform_view<std::ranges::iota_view<std::size_t, std::size_t>,
      std::function<Instance(std::size_t)>>;
 public:
  InstanceView(std::size_t count, std::function<Instance(std::size_t)> read)
      : Base(std::ranges::iota_view<std::size_t, std::size_t>(0, count), std::move(read)) {}
  Instance at(std::size_t index) const {
    if (index >= size()) throw std::out_of_range("value instance index out of range");
    return (*this)[index];
  }
};

// Read-only operands of one logical instance, equal to its record's operands.
// Stored records and collective members are borrowed (valid until the next
// non-const Graph call); a rebound template member owns its list. Temporaries
// return copies, so no reference outlives an owned list.
class OperandsRef final {
 public:
  using const_iterator = std::vector<Value>::const_iterator;
  [[nodiscard]] bool borrowed() const noexcept { return !owned_; }
  [[nodiscard]] std::size_t size() const noexcept { return list().size(); }
  [[nodiscard]] bool empty() const noexcept { return list().empty(); }
  [[nodiscard]] std::span<const Value> span() const& noexcept { return list(); }
  void span() const&& = delete;
  [[nodiscard]] const_iterator begin() const& noexcept { return list().begin(); }
  [[nodiscard]] const_iterator end() const& noexcept { return list().end(); }
  void begin() const&& = delete;
  void end() const&& = delete;
  [[nodiscard]] const Value& operator[](std::size_t index) const& noexcept { return list()[index]; }
  [[nodiscard]] Value operator[](std::size_t index) const&& { return list()[index]; }
  [[nodiscard]] const Value& at(std::size_t index) const& { return list().at(index); }
  [[nodiscard]] Value at(std::size_t index) const&& { return list().at(index); }
  [[nodiscard]] const Value& front() const& { return list().front(); }
  [[nodiscard]] Value front() const&& { return list().front(); }
  [[nodiscard]] std::vector<Value> toVector() && { if (owned_) return std::move(*owned_); return *borrowed_; }
 private:
  friend class Node;
  explicit OperandsRef(const std::vector<Value>& stored) noexcept : borrowed_(&stored) {}
  explicit OperandsRef(std::vector<Value>&& rebound) : owned_(std::move(rebound)) {}
  const std::vector<Value>& list() const noexcept { return owned_ ? *owned_ : *borrowed_; }
  const std::vector<Value>* borrowed_ = nullptr;
  std::optional<std::vector<Value>> owned_;
};

class Node final {
 public:
  Node(Node&&) noexcept = default;
  Node& operator=(Node&&) = delete;
  Node(const Node&) = delete;
  Node& operator=(const Node&) = delete;

  [[nodiscard]] std::size_t id() const noexcept { return id_; }
  [[nodiscard]] const Operation& operation() const noexcept { return *operation_; }
  [[nodiscard]] std::span<const Value> operands() const noexcept { return instances_.at(0)->operands; }
  [[nodiscard]] const SourceOrigin& origin() const noexcept { return instances_.at(0)->origin; }
  // Provenance-only query: do not reconstruct/copy this instance's operands.
  [[nodiscard]] SourceOrigin instanceOrigin(std::size_t index) const {
    if (index >= instance_count_) throw std::out_of_range("value instance index out of range");
    return readOrigin(index);
  }
  // Exactly instances().at(index).operands, with the same range check, but
  // without building the origin or types; see OperandsRef for borrowing.
  [[nodiscard]] OperandsRef instanceOperands(std::size_t index) const;
  // Records built on demand. Counts use instanceCount(), operands instanceOperands().
  [[nodiscard]] InstanceView instances() const { return InstanceView(instance_count_, [this](std::size_t i) { return readInstance(i); }); }
  [[nodiscard]] std::size_t instanceCount() const noexcept { return instance_count_; }
  [[nodiscard]] std::size_t storedInstanceCount() const noexcept;
  // Visit stored operand patterns, never the logical thread instances. A
  // parameterized pattern rebinds same-thread operands using instanceThreads().
  void forEachInstancePattern(const std::function<void(
      const Instance&, std::size_t, std::size_t, bool)>&) const;
  [[nodiscard]] bool hasUniformThreadTemplate() const noexcept {
    return bool(origin_template_) && instances_.size()==1;
  }
  // This is the one collective occurrence emitted by addCollective, whose
  // operands are shared exactly. It is not ordinary thread-template sharing.
  [[nodiscard]] bool isCollectiveOccurrence() const noexcept {
    return bool(collective_origins_) && hasUniformThreadTemplate() && !rebind_operands_;
  }
  [[nodiscard]] const ThreadMembers& instanceThreads() const noexcept { return instance_threads_; }
  [[nodiscard]] std::optional<ThreadGroupId> groupId() const noexcept { return group_id_; }
  [[nodiscard]] std::size_t resultCount() const noexcept;
  [[nodiscard]] Value result(std::size_t index = 0, std::size_t instance = 0) const;

 private:
  friend class Graph;
  friend class GraphCheckpoint;
  friend class LocalRewriteMatch;
  Node(std::uint64_t owner, std::size_t id, std::unique_ptr<const Operation> operation,
       std::vector<Value> operands, SourceOrigin origin, const InstanceTypes& types);
  std::uint64_t owner_;
  std::size_t id_;
  std::unique_ptr<const Operation> operation_;
  std::size_t instance_count_=1;
  std::map<std::size_t,std::shared_ptr<Instance>> instances_;
  // A non-stored member is the prototype instances_.at(0) with these parts.
  // origin_template_ marks every template; operand_template_ is the one owner
  // of rebound member operands and is empty for collective members, which
  // share the prototype operands exactly.
  std::function<std::vector<Value>(std::size_t)> operand_template_;
  std::function<SourceOrigin(std::size_t)> origin_template_;
  std::function<const InstanceTypes&(std::size_t)> types_template_;
  // Retain the already-stored collective provenance for lossless checkpoints.
  // Ordinary thread templates use thread_bindings, not one origin per instance.
  std::shared_ptr<const std::vector<SourceOrigin>> collective_origins_;
  bool rebind_operands_ = false;
  ThreadMembers instance_threads_;
  const Instance* storedInstance(std::size_t index) const;
  Instance readInstance(std::size_t index) const;
  // Record of a non-stored member. No range check: extraction probes the next slot.
  Instance materialize(std::size_t index) const;
  SourceOrigin readOrigin(std::size_t index) const;
  std::optional<ThreadGroupId> group_id_;
};

class NodeView final {
 public:
  using Storage = std::deque<std::unique_ptr<Node>>;
  struct Iterator {
    using iterator_category = std::forward_iterator_tag;
    using value_type = Node;
    using difference_type = std::ptrdiff_t;
    using pointer = const Node*;
    using reference = const Node&;
    Storage::const_iterator current, end;
    void skip() { while (current != end && !*current) ++current; }
    const Node& operator*() const { return **current; }
    Iterator& operator++() { ++current; skip(); return *this; }
    Iterator operator++(int) { auto old=*this; ++*this; return old; }
    const Node* operator->() const { return current->get(); }
    bool operator==(const Iterator& other) const { return current == other.current; }
    bool operator!=(const Iterator& other) const { return current != other.current; }
  };
  NodeView(const Storage& storage, std::size_t size) : storage_(storage), size_(size) {}
  Iterator begin() const { Iterator i{storage_.begin(),storage_.end()}; i.skip(); return i; }
  Iterator end() const { return {storage_.end(),storage_.end()}; }
  std::size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }
 private:
  const Storage& storage_;
  std::size_t size_;
};

struct TypeEdgeStatistics final { std::size_t nodes = 0, instances = 0; };

struct NodeUse final {
  const Node& node;
  std::size_t instance;
  [[nodiscard]] Value result(std::size_t index = 0) const { return node.result(index, instance); }
};

struct LocalRewriteStatistics final {
  std::size_t patterns = 0, operand_reads = 0, budget_rejections = 0, domain_rejections = 0;
};

class Graph;
// A match exposes only a bounded, local operand subgraph. Its bindings are
// evidence for a rewrite, not arbitrary values that need an ancestry search.
class LocalRewriteMatch final {
 public:
  [[nodiscard]] Value root() const noexcept { return root_; }
  [[nodiscard]] const Instance& instance(Value);
 private:
  friend class Graph;
  friend class GraphCheckpoint;
  LocalRewriteMatch(const Graph&, Value, const Instance&, std::size_t, bool,
                    LocalRewriteStatistics&);
  const Graph& graph_;
  Value root_;
  std::size_t count_;
  bool rebind_;
  LocalRewriteStatistics& statistics_;
  std::deque<std::pair<Value,Instance>> inspected_;
  std::vector<Value> exposed_;
};

// Spatial instances are append-only during extraction; value passes preserve identities.
// Edges describe data dependencies, including component and thread selectors.
class Graph final {
 public:
  Graph();
  ~Graph();
  Graph(Graph&&) noexcept;
  Graph& operator=(Graph&&) noexcept;
  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;

  const Node& add(std::unique_ptr<const Operation> operation,
                  std::vector<Value> operands = {}, SourceOrigin origin = {});
  // Identical collective operands, with distinct logical member instances.
  const Node& addCollective(std::unique_ptr<const Operation> operation,
                           std::vector<Value> operands, std::vector<SourceOrigin> members);
  // The caller certifies a common extraction path with parameterized thread coordinates.
  void beginThreadTemplate(std::vector<SourceOrigin> threads);
  void endThreadTemplate();
  ThreadTemplate makeThreadTemplate(std::vector<SourceOrigin> threads,
                                   std::optional<ThreadGroupId> parent = {});
  void activateThreadTemplate(ThreadTemplate&);
  void suspendThreadTemplate(ThreadTemplate&);
  void finishThreadTemplate(const ThreadTemplate&);
  NodeUse addTemplate(std::unique_ptr<const Operation> operation,
                      std::vector<Value> operands, SourceOrigin origin);
  Value bindThread(Value value, std::uint64_t cta, std::uint64_t from, std::uint64_t to);
  NodeUse addInstance(std::unique_ptr<const Operation> operation,
                      std::vector<Value> operands, SourceOrigin origin);
  // Replaces a pure scalar operation after extraction, preserving all instance identities.
  // Subsequent extraction is rejected; source-free constants may still be appended.
  // Visits stored patterns/explicit exceptions, never logical members.
  // A failed local match leaves the whole node unchanged. No ancestry BFS.
  using LocalRewrite = std::function<std::optional<std::vector<Value>>(LocalRewriteMatch&)>;
  bool rewritePure(std::size_t node, std::unique_ptr<const Operation> operation,
                   const LocalRewrite&, LocalRewriteStatistics* statistics = nullptr);
  // Call once for each executed PTX occurrence, including occurrences without value nodes.
  ThreadGroupId observeOccurrence(const SourceOrigin& origin);
  // Retains graph queries; rejects future events in this CTA and releases build indexes.
  void sealCta(std::uint64_t cta);
  [[nodiscard]] std::optional<ThreadGroupId> groupAt(const SourceOrigin& origin) const;
  // Empty if absent or ambiguous across source functions or unroll contexts.
  [[nodiscard]] std::optional<ThreadGroupId> groupAtExecution(
      std::uint64_t cta, std::uint64_t thread, std::uint64_t instruction, std::uint64_t occurrence) const;
  [[nodiscard]] std::optional<ThreadGroupId> currentGroup(std::uint64_t cta, std::uint64_t thread) const;
  [[nodiscard]] const ThreadGroup& group(ThreadGroupId id) const;
  // Extraction-prefix containment, not execution order. The forest is indexed
  // once in O(groups); each subsequent query is O(1), with no member traversal.
  [[nodiscard]] bool groupContains(ThreadGroupId ancestor, ThreadGroupId descendant) const;
  // Effect of rebinding a uniform context to another member of its stored
  // domain. Fixed means this whole annotated handle is unchanged; SameDomain
  // means its varying coordinates use that exact domain and member index.
  // Unknown is not permission to infer behavior from a few members. Bounded
  // annotation inspection only, with no operand/ancestor or member traversal.
  enum class ThreadBindingRelation { Fixed, SameDomain, Unknown };
  [[nodiscard]] ThreadBindingRelation threadBindingRelation(Value value, Value context) const;
  struct ThreadBindingMap { bool fixed=false; std::int64_t base=0,stride=1; };
  // Producer member = base + stride * context member. Exact whole-domain
  // proof from cached thread progressions; no operand or member traversal.
  [[nodiscard]] std::optional<ThreadBindingMap> threadBindingMap(Value value,Value context) const;
  [[nodiscard]] const std::deque<ThreadGroup>& groups() const noexcept;
  [[nodiscard]] Value component(Value matrix, std::vector<Value> indices) const;
  [[nodiscard]] Value asPointer(Value value, PointerProperties pointer) const;
  [[nodiscard]] Value fromThread(Value value, Value selector) const;
  // A ThreadForward node outside any template: one stored instance per thread
  // of `threads` (distinct, of base's CTA), instance k forwarding values[k], with
  // base's origin moved to that thread. Its instance 0, for fromThread readers.
  Value addThreadUnion(const SourceOrigin& base, const std::vector<std::uint64_t>& threads,
                       std::vector<Value> values);
  // The operand a ThreadForward instance forwards; any other value unchanged.
  [[nodiscard]] Value forwarded(Value value) const;
  [[nodiscard]] Instance instance(const Value& value) const;
  // Exactly instance(value).operands / .origin, without building the other parts.
  [[nodiscard]] OperandsRef instanceOperands(const Value& value) const;
  [[nodiscard]] SourceOrigin instanceOrigin(const Value& value) const;
  [[nodiscard]] Value resolve(Value value) const;
  // Drop only bitcast chains that return to an identical type. Numeric
  // conversion, selection and pointer provenance remain semantic boundaries.
  [[nodiscard]] Value canonicalInterpretation(Value value) const;
  [[nodiscard]] Value bitcast(Value source, ValueType target) const;
  [[nodiscard]] Value convert(Value source, ValueType target, ConversionAttributes attributes) const;
  TypeEdgeStatistics lowerTypeConversions();
  [[nodiscard]] std::string structuralKey(Value value, const SourceOrigin& consumer) const;
  [[nodiscard]] std::string structuralTypeKey(const ValueType& type, const SourceOrigin& consumer) const;
  // Explicit launch metadata, visited once when building a coordinate index.
  void forEachThreadBinding(const std::function<void(const SourceOrigin&)>&) const;
  [[nodiscard]] bool contains(const Value& value) const noexcept;
  [[nodiscard]] const ValueType& type(const Value& value) const;
  [[nodiscard]] const Node& definingNode(const Value& value) const;
  [[nodiscard]] const Node& node(std::size_t id) const;
  [[nodiscard]] NodeView nodes() const noexcept;
  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] bool empty() const noexcept { return size() == 0; }

  // Debug checkpoint at the pre-value-pass boundary. Stored templates only;
  // replay does not resume frontend extraction or reconstruct sharing.

 private:
  void validateType(const ValueType& type) const;
  friend class GraphCheckpoint;
  struct Storage;
  static Value resolveStorage(Storage&, Value value);
  static Value rebindThread(Storage&, Value value, std::uint64_t cta, std::uint64_t from, std::uint64_t to);
  static std::function<std::vector<Value>(std::size_t)> memberOperands(
      Storage&, std::shared_ptr<const Instance> base, const Node&);
  std::pair<const Node*,std::size_t> locate(const Value& value) const;
  const ValueType& resolvedType(const Value& resolved, const Node* producer) const;
  void setInstanceTemplate(const Node&, ThreadGroupId, ThreadMembers);
  std::unique_ptr<Storage> storage_;
};

}  // namespace sentryward::value_ir
