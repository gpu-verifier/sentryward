#pragma once
#include "sentryward/equivalence/output_family.h"
#include "sentryward/scalar_fold/artifact.h"
namespace sentryward::equivalence {
FamilyResult comparePartitionedOutputs(const scalar_fold::Artifact&,const scalar_fold::Artifact&,
                                      std::span<const TensorOutputMapping>,Options = {});
}
