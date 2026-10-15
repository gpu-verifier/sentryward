#pragma once

#include "sentryward/scalar_fold/graph.h"
#include <array>
#include <map>

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
  struct Block {
    Ref initial, body;
    std::vector<Ref> terms, prefixes, first, second;
    std::uint64_t count=0;
    std::uint32_t level=0;
    std::string type;
  };
  std::optional<Ref> appendBlock(Ref accumulator,Ref term,const std::string& type,
                                 std::uint32_t level);
  std::size_t templateId(Ref);
  std::vector<std::array<std::size_t,2>> template_ids_;
  struct ParameterTree {
    std::optional<std::uint64_t> value;
    std::vector<std::size_t> children;
  };
  std::vector<ParameterTree> parameter_trees_{{std::nullopt,{}}};
  std::unordered_map<std::string,std::size_t> parameter_keys_;
  std::vector<std::array<std::size_t,2>> parameter_roots_;
  std::map<std::uint32_t,std::unordered_map<Id,Ref>> lifted_by_level_;
  std::unordered_map<std::string,std::size_t> template_keys_;
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
