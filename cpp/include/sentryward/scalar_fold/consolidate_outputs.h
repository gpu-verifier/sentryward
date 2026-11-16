#pragma once
#include "sentryward/scalar_fold/artifact.h"
namespace sentryward::scalar_fold {
// Merge adjacent flat families with exactly shifted scalar/fold bodies.
// Weak cache node IDs observe the final collection without retaining obsolete
// scalar expansions when adjacent outputs become one indexed family.
// With strided_rows (and no tensor_families), runs of at least 16 equal rows at any constant row
// stride, other families interleaved, become one two-dimensional family (outputs at symbolic
// positions excepted) when the rows lie in a regular tiling (every family between two of them is
// a row of the same shape); every row is checked against the generalized body.
// `observer` observes that final collection (Graph::collect).
std::size_t consolidateOutputs(Artifact&,bool tensor_families=false,
                               std::span<std::optional<Id>> weak_nodes={},bool strided_rows=false,
                               const Graph::RemapObserver& observer={});
// The merge of consolidateOutputs on a family list, in (allocation, offset) order, without
// collecting the graph; with tensor_families, contiguous equal rows also become one
// two-dimensional family. Returns the number of families merged away. O(graph) for the
// affine forms, then linear in the families plus the body pairs compared.
// With strided_runs, runs of at least 16 one-element families at one constant stride larger than
// the element first become one-dimensional strided families (every member checked against the
// generalized body); the class-mode lowering asks for it for pending outputs under node pressure.
std::size_t mergeOutputFamilies(Graph&,std::vector<OutputFamily>&,bool tensor_families,bool strided_rows=false,
                                bool strided_runs=false);
}
