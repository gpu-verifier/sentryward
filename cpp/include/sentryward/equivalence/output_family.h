#pragma once
#include "sentryward/scalar_fold/output_family.h"
#include "sentryward/equivalence/output_mapping.h"
#include "sentryward/equivalence/egraph.h"

namespace sentryward::equivalence {
struct FamilyResult {
  Result proof;
  std::uint64_t outputs=0, covered_bytes=0;
};
struct FamilyPair {
  const scalar_fold::OutputFamily* left=nullptr;
  const scalar_fold::OutputFamily* right=nullptr;
  const TensorOutputMapping* mapping=nullptr;
};
// A conjunction of independent whole-family proofs over ONE already matching
// index domain. Borrowed families/mappings belong to the two supplied graphs.
// No reshape, element expansion or cross-family mathematical assumption.
// Failure credits no outputs/bytes from this cohort.
FamilyResult compareOutputFamilySets(const scalar_fold::Graph&,
                                    const scalar_fold::Graph&,
                                    std::span<const FamilyPair>,Options = {});
// Input identities are already aligned. Each family covers its whole declared tensor.
FamilyResult compareOutputFamilies(const scalar_fold::Graph&,const scalar_fold::OutputFamily&,
                                  const scalar_fold::Graph&,const scalar_fold::OutputFamily&,
                                  const TensorOutputMapping&,Options = {});
}
