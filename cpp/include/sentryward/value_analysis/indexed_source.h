#pragma once
#include "sentryward/value_analysis/memory_elimination.h"
#include <memory>
#include <unordered_map>

namespace sentryward::value_analysis {
struct IndexedSourceLimit : std::runtime_error {
  std::string code;
  explicit IndexedSourceLimit(std::string message, std::string reason = "source_index_limit_exceeded")
      : std::runtime_error(std::move(message)), code(std::move(reason)) {}
};
// Cumulative byte ends of immutable AssembledRead pieces. One pass owns one
// index and lends it to every consumer of the same raw reads, so a reused
// assembly is measured once in O(P) and each slice is an O(log P) search.
// Keys are stable substitution addresses: the raw read map must outlive the
// index and stay unchanged. Each consumer keeps its own policy for a piece
// whose extent is not representable.
class AssembledPieceEnds final {
 public:
  enum class Stop : unsigned char { None, Dynamic, Overflow };
  struct Extents {
    // Ends of the leading representable pieces. Unless stop is None, piece
    // ends.size() is descriptor-sized (Dynamic) or ends past UINT64_MAX.
    std::vector<std::uint64_t> ends;
    Stop stop = Stop::None;
    bool built = false, requested = false;
  };
  const Extents& extents(const AssembledRead& read) {
    auto& entry = entries_[&read];
    if (!entry.built) build(read, entry);
    return entry;
  }
  // Every end of a representable assembly, or null when the caller should
  // scan instead: an unrepresentable extent, or a small assembly's first
  // request. A single-use small assembly does not pay for an O(P) index.
  const std::vector<std::uint64_t>* reusedEnds(const AssembledRead& read) {
    auto& entry = entries_[&read];
    if (!entry.built) {
      if (read.pieces.size() <= 8 && !entry.requested) { entry.requested = true; return nullptr; }
      build(read, entry);
    }
    return entry.stop == Stop::None ? &entry.ends : nullptr;
  }

 private:
  static void build(const AssembledRead& read, Extents& entry) {
    entry.ends.reserve(read.pieces.size());
    std::uint64_t end = 0;
    for (const auto& piece : read.pieces) {
      // Exactly the types whose bitSize() throws; record the piece instead.
      if (!piece.type.hasStaticShape() || piece.type.element().isParameterized()) { entry.stop = Stop::Dynamic; break; }
      const auto bytes = piece.type.bitSize() / 8;
      if (bytes > UINT64_MAX - end) { entry.stop = Stop::Overflow; break; }
      entry.ends.push_back(end += bytes);
    }
    entry.built = true;
  }
  std::unordered_map<const AssembledRead*, Extents> entries_;
};
// A disjoint domain of existing finite source records. A null source denotes
// one exact record. result_stride permits interleaved domains without a table.
// Planning performs no scalar/Fold expression construction.
struct IndexedSourceDomain final {
  std::size_t begin = 0;
  std::size_t count = 1;
  std::shared_ptr<const IndexedSource> source;
  std::size_t result_stride = 1;
};
// A byte slice of an existing operand. Used for native register fragments;
// no temporary Value nodes or per-output scalar expressions are constructed.
struct IndexedSourceSlice final {
  value_ir::Value value;
  std::uint64_t offset = 0, bytes = 0;
};
// One already-extracted spatial value template, not a list of member Values.
// prototype.instanceIndex() is the first member; temporal occurrences remain
// different nodes. Fixed component indices and interpretation edges stay on
// the prototype; they are not a reason to materialize the member domain.
// The matcher may certify the whole domain or reject it;
// this interface never materializes members or falls back to per-thread work.
struct IndexedSourceTemplate final {
  value_ir::Value prototype;
  std::size_t count;
};
// Bind one annotated operand in its consumer's existing thread domain.
// Visits interpretation edges and component indices only, not math ancestors
// or logical members. Matrix and index uses retain independent maps. Returns
// null when the existing graph cannot certify a member mapping.
std::shared_ptr<const IndexedSourceUse> bindSourceUse(
    const value_ir::Graph&, value_ir::Value operand, value_ir::Value consumer,
    std::int64_t base = 0, std::int64_t stride = 1);
// Partition stored source records by their existing node/result/interpretation
// identity and affine member mapping. Does not compare operand expressions.
// No logical members are created. Contiguous planning is linear in records
// times annotation depth. Interleaved planning also indexes template keys and
// compares the stored interpretation variants in each key's bucket.
// Optional payload offsets participate in the identity: interleaved byte
// slices are grouped without dropping their within-value offset.
std::vector<IndexedSourceDomain> mapSourceTemplates(
    const value_ir::Graph&, std::span<const value_ir::Value>, bool interleaved = false,
    std::span<const std::uint64_t> payload_offsets = {});
// One normalization pass owns one matcher. Raw read substitutions and the
// Value graph must remain immutable; normalized views may be added separately.
// Source interval indices and resolved slices are reused across prefix matches.
class IndexedSourceMatcher {
 public:
  IndexedSourceMatcher(const value_ir::Graph&, const MemoryEliminationResult&);
  // Shares the pass's assembled-read extents; the owner must outlive this.
  IndexedSourceMatcher(const value_ir::Graph&, const MemoryEliminationResult&, AssembledPieceEnds&);
  ~IndexedSourceMatcher();
  std::optional<IndexedSource> match(std::span<const value_ir::Value>);
  std::optional<IndexedSource> matchTemplate(const IndexedSourceTemplate&);
  // Exact finite validation of one native operand layout, including packed
  // scalar fields. Failure does not trigger element-wise math construction.
  std::optional<IndexedSource> matchSlices(std::span<const IndexedSourceSlice>);
  // Prove a rectangular source grid, optionally with one diagonal zero mask.
  // Validate all existing records, including every zero and every active use;
  // the retained representation has two coordinate coefficients per use.
  std::optional<IndexedSource> matchGrid(
      std::span<const value_ir::Value>, std::uint64_t columns);
  // Validate supplied stored slices as a full grid; no source-domain expansion.
  // Each slice must resolve to one complete scalar Value with its annotations.
  std::optional<IndexedSource> matchGridSlices(
      std::span<const IndexedSourceSlice>, std::uint64_t columns);
  // Every finite member is validated; this is not a template-only input API.
  // Immutable use nodes share children, not flattened leaf lists or tree paths.
  // Matching is memoized by (use node, full annotated target Value) and a
  // certified member interval. A changed use replaces its ancestors only;
  // another use of the same concrete Value retains its own index map.
  // With P finite roots and U distinct visited use/target pairs, work includes
  // O((P+U) log U) indexing plus their direct operands and read-slice queries.
  // This is not a template-only bound: genuinely different contexts cost work.
  // One affine prefix may gain a mixed-radix boundary and at most sixteen
  // dyadic quotient corrections per use. Every correction is identically zero
  // on the certified prefix; remaining finite records are still validated.
  // No arbitrary source/member table is encoded in these maps.
  // Refinement preserves all prior constraints; failure keeps the old root.
  // A failed boundary costs one local prefix attempt, never a group restart.
  // Read-from range certificates are produced by memory elimination. Together
  // with identical immutable thread-domain bindings, they permit reuse of a
  // pure template proof across that member range: O(P + template DAG work),
  // not P repetitions of the operand DAG. Existing root records are inspected
  // once to check their node/instance coordinate; no member is skipped merely
  // because its source resembles the first two samples. Uncertified mappings
  // retain the finite-validation bound above and are counted in debug builds.
  // Exact-value proofs, broadcasts invariant under thread rebinding, and
  // affine member-coordinate proofs are distinct. A fixed/singleton subtree
  // must not become coordinate-dependent merely because it is in a template;
  // interleaved affine proofs include their congruence-class step explicitly.
  std::vector<IndexedSourceDomain> partition(std::span<const value_ir::Value>);
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
// Validates every member. A failed prefix match leaves the sources independent.
std::optional<IndexedSource> matchIndexedSource(
    const value_ir::Graph& graph, const MemoryEliminationResult& memory,
    std::span<const value_ir::Value> values);
}
