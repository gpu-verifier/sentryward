#pragma once
#include "comparison.hpp"
#include "sentryward/scalar_fold/artifact.h"
namespace rg {
Document fromArtifact(const sentryward::scalar_fold::Artifact&);
void alignOutputs(Document& left,Document& right,J& mapping);
// Consumer owns the scalar artifact; conversion is one stored-node/edge pass.
ComparisonResult compareArtifact(Document reference,
    const sentryward::scalar_fold::Artifact& optimized,J mapping,
    const ComparisonOptions& options={}, const ContextBuilder& build_context={});
}
