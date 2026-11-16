#pragma once

#include "sentryward/scalar_fold/graph.h"
#include <array>
#include <map>
#include <memory>
#include <utility>

namespace sentryward::scalar_fold {

// Finite ideal-real reductions; three matching terms create an affine fold.
// Later terms extend that fold only after a read-only structural check.
class ReductionBuilder {
 public:
  explicit ReductionBuilder(Graph& graph) : graph_(graph) {}
  Ref append(Reducer, Ref accumulator, Ref term, std::string type,
             std::uint32_t binder_level = 0);
  // Append a certified finite reduction chunk without expanding its members.
  // First concatenate ideal-real Sum chunks when the next body is exactly
  // the previous body with i replaced by i + previous_count. One fixed
  // substitution only; read-only structural check capped at 4096 operand
  // pairs. Failure preserves the existing chunk/matching behavior.
  // Repeated chunk bodies use the same bounded structural matcher as append.
  // A mismatch retains the original Fold(initial, body), not a guessed merge.
  Ref appendFold(Reducer, Ref accumulator, Ref body, std::uint32_t binder_level,
                 std::uint64_t count, std::string type);
  // Retain live expressions and matching state; remaps roots and expires other handles.
  std::size_t collect(std::span<Ref> live_roots);
  // Keep matcher state only for expressions that may still be extended.
  std::size_t collect(std::span<Ref> live_roots, std::span<const Ref> extendable_roots,
                      std::span<std::optional<Id>> weak_nodes = {},
                      const Graph::RemapObserver& observer = {});
  // Reusing such an accumulator after matcher-state reset would lose the
  // ability to extend its recognized/pending Fold. Expected O(1), no traversal.
  bool hasMatchingState(const Ref& value) const {
    return runs_.contains(value) || pending_.contains(value) || blocks_.contains(value);
  }
  struct Statistics {
    std::size_t created = 0, extended = 0, rejected = 0, reclaimed = 0;
    std::size_t block_checks = 0, block_created = 0, block_extended = 0;
    std::size_t block_candidates=0, block_match_rejected=0, block_body_rejected=0;
    std::size_t template_nodes=0, template_edges=0;
  };
  const Statistics& statistics() const { return stats_; }
 private:
  struct Run {
    Reducer reducer;
    Ref initial, first, second, body;
    std::uint64_t count;
    std::uint32_t level;
    std::string type;
  };
  struct Pending {
    Reducer reducer;
    Ref initial, first;
    std::uint32_t level;
    std::string type;
  };
  void refreshBounds();
  // A block's current terms and the prefix (accumulator) before each: the view [begin, end)
  // of an append-only buffer that consecutive appends share. An append extends the buffer
  // in place when its view ends at the buffer's end, and otherwise (another append extended
  // it, or the view lies far into the buffer) moves its view into a fresh buffer, so an
  // append costs O(1) amortized instead of a copy of the window. The prototype terms
  // (first, second) never change once made and are shared by the copies of a block.
  struct Window {
    std::vector<Ref> terms, prefixes;
  };
  struct Block {
    Ref initial, body;
    std::shared_ptr<Window> window;
    std::size_t begin=0, end=0;
    std::shared_ptr<std::vector<Ref>> first, second;
    std::uint64_t count=0;
    std::uint32_t level=0;
    std::string type;
    std::size_t size() const { return end-begin; }
    const Ref& term(std::size_t i) const { return window->terms[begin+i]; }
    const Ref& prefix(std::size_t i) const { return window->prefixes[begin+i]; }
    void push(const Ref& term, const Ref& prefix);
    void clearTerms() { window.reset(); begin=end=0; }
  };
  std::optional<Ref> appendBlock(Ref accumulator,Ref term,const std::string& type,
                                 std::uint32_t level);
  std::size_t templateId(Ref);
  // Variable-length integer keys interned in first-insertion order: key i (from 0) is
  // id i + 1. Open addressing over {stored hash, key index} at load <= 1/2, keys packed in
  // one word array: no allocation or string per key. clear() keeps the capacity.
  class KeyTable {
   public:
    // The key's id and whether this call inserted it. Expected O(key length).
    std::pair<std::size_t,bool> intern(std::span<const std::uint64_t> key);
    std::size_t size() const { return starts_.size(); }
    void clear();
   private:
    void grow();
    std::vector<std::uint64_t> words_;
    std::vector<std::size_t> starts_, hashes_;
    std::vector<std::size_t> slots_;  // key index + 1; 0 marks an empty slot
  };
  // An injective numbering of the strings a template key contains (0: the empty string).
  // The numbering never affects template or parameter ids, which are first-insertion
  // positions of keys; it persists across graph epochs.
  std::uint64_t stringId(const std::string&);
  std::unordered_map<std::string,std::uint64_t> string_ids_;
  std::vector<std::uint64_t> key_scratch_;
  std::vector<std::array<std::size_t,2>> template_ids_;
  struct ParameterTree {
    std::optional<std::uint64_t> value;
    std::vector<std::size_t> children;
  };
  std::vector<ParameterTree> parameter_trees_{{std::nullopt,{}}};
  KeyTable parameter_keys_;
  std::vector<std::array<std::size_t,2>> parameter_roots_;
  std::map<std::uint32_t,std::unordered_map<Id,Ref>> lifted_by_level_;
  KeyTable template_keys_;
  std::optional<std::size_t> parameterDifference(std::size_t,std::size_t);
  std::map<std::pair<std::size_t,std::size_t>,std::optional<std::size_t>> parameter_differences_;
  std::unordered_map<std::string,std::size_t> difference_keys_;
  std::unordered_map<Ref,Block,RefHash> blocks_;
  Graph& graph_;
  Statistics stats_;
  // Memoized binder-level bounds for the current graph epoch (indexed by node id).
  std::vector<std::uint32_t> binder_bounds_;
  std::uint64_t bounds_owner_ = 0;
  std::unordered_map<Ref, Run, RefHash> runs_;
  std::unordered_map<Ref, Pending, RefHash> pending_;
};
}  // namespace sentryward::scalar_fold
