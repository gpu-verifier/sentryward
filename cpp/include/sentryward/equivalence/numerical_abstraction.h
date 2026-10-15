#pragma once
#include "sentryward/scalar_fold/artifact.h"
#include <chrono>

namespace sentryward::equivalence {
// Reassociate finite floating-point sums under the ideal-real contract.
scalar_fold::Artifact abstractNumericalReductions(
    const scalar_fold::Artifact&, std::size_t max_nodes,
    std::chrono::steady_clock::time_point deadline);
// Ownership transfer avoids rebuilding a typed artifact when ideal-real
// reassociation is inapplicable. No nodes are created on that path.
scalar_fold::Artifact abstractNumericalReductions(
    scalar_fold::Artifact&&, std::size_t max_nodes,
    std::chrono::steady_clock::time_point deadline);
}
