#pragma once
#include "comparison.hpp"
#include <map>
#include "sentryward/scalar_fold/artifact.h"
namespace rg {
Document fromArtifact(const sentryward::scalar_fold::Artifact&);
void alignOutputs(Document& left,Document& right,J& mapping);
// The bytes the mapping's outputs compare on the optimized side, per
// allocation; empty when the output mapping is malformed or names an
// allocation twice (the comparison refuses those itself).
std::map<std::string,sentryward::scalar_fold::ComparedOutput> comparedOutputs(const J& mapping);
// Consumer owns the scalar artifact; conversion is one stored-node/edge pass.
ComparisonResult compareArtifact(Document reference,
    const sentryward::scalar_fold::Artifact& optimized,J mapping,
    const ComparisonOptions& options={}, const ContextBuilder& build_context={});
}
