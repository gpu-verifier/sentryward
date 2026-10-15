#pragma once
#include "sentryward/scalar_fold/artifact.h"
#include <chrono>
namespace sentryward::equivalence {
struct FoldRewriteStatistics { std::size_t attempts=0, merged=0; };
// Ideal-real Sum concatenation; bodies are matched after exact index substitution.
scalar_fold::Artifact mergeAdjacentFolds(const scalar_fold::Artifact&,
    std::size_t max_nodes,std::chrono::steady_clock::time_point deadline,
    FoldRewriteStatistics* statistics=nullptr);
// Retain admitted immutable input nodes. max_nodes bounds newly constructed
// rewrite nodes, not the already admitted input; input/e-graph limits are
// unchanged. No candidate means an identity with no graph reconstruction.
scalar_fold::Artifact mergeAdjacentFolds(scalar_fold::Artifact&&,
    std::size_t max_nodes,std::chrono::steady_clock::time_point deadline,
    FoldRewriteStatistics* statistics=nullptr);
}
