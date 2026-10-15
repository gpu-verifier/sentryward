#pragma once

#include <map>
#include <memory>
#include <string>
#include "sentryward/scalar_fold/artifact.h"

namespace sentryward::equivalence {
class ComparisonContext;
struct InputTensorLayout {
  std::uint64_t source_offset=0, target_offset=0, element_bytes=0;
  std::vector<std::uint64_t> shape, source_strides_bytes, target_strides_bytes;
};
struct InputParameterConstant {
  std::string type, literal;
};
struct InputIdentityMapping {
  std::map<std::string,std::string> allocations, parameters;
  std::map<std::string,InputTensorLayout> layouts{};
  std::map<std::string,InputParameterConstant> parameter_constants{};
  // Null preserves unscoped behavior. Only exact, constant-offset reads in a
  // validated original fixed-read-only context can be specialized.
  std::shared_ptr<const ComparisonContext> comparison_context{};
};
struct AlignedInputs {
  scalar_fold::Artifact artifact;
  bool complete=false;
  std::string reason;
};
// Explicit injective renaming; byte offsets, conversions and provenance stay exact.
// Output allocation names are retained for the separate output coverage mapping.
AlignedInputs alignInputs(const scalar_fold::Artifact&,const InputIdentityMapping&,
                          bool normalize_values=false);
} // namespace sentryward::equivalence
