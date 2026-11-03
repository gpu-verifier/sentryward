#pragma once
#include "sentryward/scalar_fold/artifact.h"
namespace sentryward::scalar_fold {
// Merge adjacent flat families with exactly shifted scalar/fold bodies.
// Weak cache node IDs observe the final collection without retaining obsolete
// scalar expansions when adjacent outputs become one indexed family.
std::size_t consolidateOutputs(Artifact&,bool tensor_families=false,
                               std::span<std::optional<Id>> weak_nodes={});
// The merge of consolidateOutputs on a family list, in (allocation, offset) order, without
// collecting the graph; with tensor_families, contiguous equal rows also become one
// two-dimensional family. Returns the number of families merged away. O(graph) for the
// affine forms, then linear in the families plus the body pairs compared.
std::size_t mergeOutputFamilies(Graph&,std::vector<OutputFamily>&,bool tensor_families);
}
