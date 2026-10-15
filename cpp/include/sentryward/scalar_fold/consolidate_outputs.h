#pragma once
#include "sentryward/scalar_fold/artifact.h"
namespace sentryward::scalar_fold {
// Merge adjacent flat families with exactly shifted scalar/fold bodies.
// Weak cache node IDs observe the final collection without retaining obsolete
// scalar expansions when adjacent outputs become one indexed family.
std::size_t consolidateOutputs(Artifact&,bool tensor_families=false,
                               std::span<std::optional<Id>> weak_nodes={});
}
