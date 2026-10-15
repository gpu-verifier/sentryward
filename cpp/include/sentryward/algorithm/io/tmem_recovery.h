#pragma once

#include <vector>

#include "sentryward/algorithm/io/source_json.h"

namespace sentryward::algorithm {

struct TmemDeallocationRecovery final {
  CtaId cta;
  InstId allocation;
  InstId retirement;
  std::vector<InstId> deallocations;
  SourceLocation allocation_location;
  SourceLocation deallocation_location;
};

// Recover the SSD pattern: one allocation, four complete deallocation warps.
[[nodiscard]] std::vector<TmemDeallocationRecovery>
RecoverSsdDeallocations(SourceArtifact& input);

}  // namespace sentryward::algorithm
