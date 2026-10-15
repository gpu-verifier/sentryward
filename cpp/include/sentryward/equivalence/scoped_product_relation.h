#pragma once

#include "sentryward/equivalence/scoped_body_relation.h"

namespace sentryward::equivalence {
struct ScopedProductResult {
  bool proved=false;
  std::string reason;
  std::size_t reference_factors=0,actual_factors=0,body_work=0,scope_identities=0;
  // Diagnostic counts only. No caller-visible assignment/proof handle.
  std::size_t context_entries=0,context_id_reads=0,context_prefix_hits=0;
};
// Exact ideal-real equality of products of ACTUAL body uses, not equality of
// their wrappers/Sums/outputs. Every original scope/path/type is validated.
// A factor may be lifted only across an actual nested BODY domain proven
// independent of that factor. Selected Multiply heads flatten locally to at
// most8 factors; multiplicity is retained and ambiguous alignment is Unknown.
// Same scoped matcher/ledger, no enode or widened expression/scope is created.
// Future Engine consumers reuse the private epoch factory, not this one-shot
// importer per point/update.
ScopedProductResult proveScopedProductRelation(const scalar_fold::Graph&,
    std::span<const ScopedBodyPath> reference_factors,const scalar_fold::Graph&,
    std::span<const ScopedBodyPath> actual_factors,std::span<const ScopedAxisBinding>,Options={});
} // namespace sentryward::equivalence
