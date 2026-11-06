#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>
#include <functional>

namespace sentryward::scalar_fold {

using Id = std::size_t;

enum class Kind { Constant, InputParameter, InputValue, Index, Operation, Fold,
                  Parameter = InputParameter, InitialCell = InputValue }; // legacy C++ aliases
enum class Reducer { Sum, Product, Max, Min };

// A numerical conversion belongs to this use of a scalar, not to its producer.
struct Ref {
  Id node = 0;
  std::string conversion;
  std::string converted_type;
  std::uint64_t owner = 0;
  std::string pointer_provenance;
  bool operator==(const Ref&) const = default;
};

struct RefHash {
  std::size_t operator()(const Ref&) const;
};

// The modes of the PTX instructions a node is the direct image of (Graph::instructionModes): the
// node computes that instruction's operation on the instruction's operands. A mul, add or sub
// without rounding modifier rounds to nearest even and may be contracted into an fma (`contractible`);
// `nearest`, `toward_zero`, `toward_negative` and `toward_positive` are the explicit .rn, .rz, .rm and
// .rp; `flush` and `no_flush` record .ftz (an ex2.approx carries only these). `synthesized` marks a
// node that a construction which is no single instruction also made (a reassociated, canonical or
// accumulated sum): it is then no instruction image. One interned node made several times carries
// the union of its marks. 0: no instruction image.
namespace instruction_mode {
inline constexpr std::uint8_t contractible = 1, nearest = 2, toward_zero = 4, toward_negative = 8,
    toward_positive = 16, flush = 32, no_flush = 64, synthesized = 128;
// Whether `modes` describes an instruction image.
inline bool instruction(std::uint8_t modes) { return modes && !(modes & synthesized); }
// A node generalizing two nodes (one per use) is an instruction image only when both are.
inline std::uint8_t common(std::uint8_t a, std::uint8_t b) { return a && b ? std::uint8_t(a | b) : 0; }
}

struct FoldInfo {
  Reducer reducer = Reducer::Sum;
  std::uint32_t binder_level = 0;
  std::uint64_t count = 0;
  bool operator==(const FoldInfo&) const = default;
};

struct Node {
  Kind kind = Kind::Constant;
  std::string type;
  std::string name;
  std::string attributes;
  std::vector<Ref> operands;
  std::optional<FoldInfo> fold;
  std::uint32_t index_level = 0;
  bool operator==(const Node&) const = default;
};

// Nodes are immutable and scalar-valued. Fold iterates [0,count) at unit stride.
// Index levels are absolute lexical depths; a fold binds its body, not its initial.
class Graph {
 public:
  Graph();
  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;
  Graph(Graph&&) noexcept = default;
  Graph& operator=(Graph&&) noexcept = default;

  Ref constant(std::string type, std::string literal);
  Ref inputParameter(std::string type, std::string symbol);
  Ref parameter(std::string type, std::string symbol) {
    return inputParameter(std::move(type),std::move(symbol));
  }
  Ref inputValue(std::string allocation, Ref byte_offset, std::string type);
  Ref initialCell(std::string allocation, Ref byte_offset, std::string type) {
    return inputValue(std::move(allocation),std::move(byte_offset),std::move(type));
  }
  Ref index(std::uint32_t level);
  Ref operation(std::string name, std::string type,
                std::vector<Ref> operands, std::string attributes = {});
  Ref fold(Reducer, Ref initial, Ref body, std::uint32_t binder_level,
           std::uint64_t count, std::string type);
  Ref convert(Ref, std::string target_type, std::string step);
  // Provenance, not identity: never part of a node's equality, hash or attributes. Adds `modes`
  // (instruction_mode bits) to an Operation node; any other node is left unmarked.
  void markInstruction(const Ref&, std::uint8_t modes);
  std::uint8_t instructionModes(Id id) const { return id < instructions_.size() ? instructions_[id] : 0; }
  Ref withPointer(Ref, std::string provenance);
  Ref ref(Id) const;

  const Node& node(Id) const;
  const Node& node(const Ref&) const;
  const std::string& type(const Ref&) const;
  std::size_t size() const { return nodes_.size(); }
  const std::vector<Node>& nodes() const { return nodes_; }

  // Reject free indices and folds nested at an inconsistent lexical level.
  // Share validation across roots, retaining lexical scope in the visit key.
  void validateRoots(std::span<const Ref>, std::uint32_t outer_binders = 0,
                     std::span<const std::uint32_t> root_binders = {}) const;
  // Retain supplied roots and descendants. Weak node IDs never retain nodes:
  // surviving IDs are remapped, dead IDs reset. All other handles expire. A
  // cache can refresh its surviving handles' node/owner without copying their
  // conversion and provenance strings into a second table during collection.
  using RemapObserver = std::function<void(std::span<const Id>,std::uint64_t)>;
  std::size_t collect(std::span<Ref> roots,
                      std::span<std::optional<Id>> weak_nodes = {},
                      const RemapObserver& observer = {});

 private:
  // Hash-consing index: open addressing over {node hash, id}, a power-of-two
  // capacity at load <= 1/2, linear probing. No allocation per node.
  struct IndexSlot {
    std::size_t hash;
    Id id;
  };
  Ref intern(Node);
  void validateRef(const Ref&) const;
  Id findNode(std::size_t hash, const Node&) const;
  void reserveIndex(std::size_t nodes);
  void indexNode(IndexSlot) noexcept;
  std::uint64_t owner_;
  std::vector<Node> nodes_;
  std::vector<IndexSlot> index_;
  // instruction_mode bits per node ID; shorter than nodes_ when the tail is unmarked.
  std::vector<std::uint8_t> instructions_;
};

struct Statistics {
  std::size_t nodes = 0;
  std::size_t edges = 0;
  std::size_t constants = 0;
  std::size_t parameters = 0;
  std::size_t initial_cells = 0;
  std::size_t indices = 0;
  std::size_t operations = 0;
  std::size_t folds = 0;
  std::size_t unique_fold_bodies = 0;
  std::size_t max_fold_nesting = 0;
  std::size_t range_descriptions = 0;
  std::uint64_t logical_iterations = 0;
  bool logical_iterations_overflow = false;
};

std::string kindName(Kind);
std::string reducerName(Reducer);
// Counts structurally reachable nodes; logical_iterations sums each distinct
// fold range once, saturating at uint64_t's limit.
Statistics statistics(const Graph&, std::span<const Ref> roots, std::uint32_t outer_binders = 0,
                      std::span<const std::uint32_t> root_binders = {});
}  // namespace sentryward::scalar_fold
