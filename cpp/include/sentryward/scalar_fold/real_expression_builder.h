#pragma once

#include <functional>
#include <memory>
#include "sentryward/scalar_fold/reduction_builder.h"

namespace sentryward::scalar_fold {

// The exact canonical forms RealExpressionBuilder::canonicalize applies.
enum class CanonicalForms {
  Bf16Fields,         // typed bf16 field reads
  SumsAndBf16Fields,  // typed bf16 field reads and canonical ideal-real sums
};

// Constructs ideal-real expressions and extends structurally matched finite folds.
class RealExpressionBuilder {
 public:
  RealExpressionBuilder(Graph&, ReductionBuilder&, std::function<void()> poll = {});
  ~RealExpressionBuilder();
  RealExpressionBuilder(const RealExpressionBuilder&) = delete;
  RealExpressionBuilder& operator=(const RealExpressionBuilder&) = delete;

  // Exact PTX literal decoding precedes ideal-real update construction.
  Ref constant(std::string type, std::string literal);
  Ref operation(std::string name, std::string type, std::vector<Ref> operands,
                std::string attributes = {}, std::uint32_t binder_level = 0);
  // Exact canonical forms below `roots`, so that expressions which differ only in how an
  // equal value was computed become shifted copies of each other (output merging):
  //  - always: a bf16 field of a raw b32 cell widened by its bits (Bitcast(ShiftLeft(c, 16))
  //    or Bitcast(BitAnd(c, 0xffff0000)) to f32) becomes the typed read of that field
  //    converted bf16 -> f32. This is bit-exact (bf16 is the upper half of f32), and the low
  //    and high fields of a packed pair get one form;
  //  - with SumsAndBf16Fields: every maximal ideal-real sum (an ideal Add, flattened through
  //    its clean ideal Add operands of its type) is rebuilt as one left-deep sum of its terms
  //    in a shift-invariant order: by structure (literal values ignored), then by the first
  //    literal, in operand order, where two terms of one structure differ. Ideal-real
  //    addition is associative and commutative, so sums associated differently (e.g. by the
  //    lanes of a warp butterfly) become one sum; a sum of too many terms keeps its tree.
  //    Exact, but a flattened sum no longer shares its partial sums.
  // Rewrites `roots` in place; other handles stay valid. O(reachable nodes), plus sorting
  // them by ID and, for sums, flattened terms x log terms x the compared prefix of two terms.
  void canonicalize(std::span<Ref> roots, CanonicalForms forms);
  std::size_t collect(std::span<Ref> roots);
  std::size_t collect(std::span<Ref> roots,std::span<const Ref> extendable,
                      std::span<std::optional<Id>> weak_nodes = {});
  struct Statistics {
    std::size_t templates=0,compiles=0,cache_hits=0,cache_misses=0,
      cache_capacity_rejections=0,query_work=0,compile_nodes=0,
      budget_rejections=0,summary_nodes=0,summary_remaps=0;
  };
  const Statistics& statistics() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace sentryward::scalar_fold
