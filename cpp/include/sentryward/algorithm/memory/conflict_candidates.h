#pragma once

#include <optional>

#include "sentryward/algorithm/memory/access.h"

namespace sentryward::algorithm {

struct MemoryConflictPair final {
  AccessId first;
  AccessId second;
};

// Pair stream; completeness and uniqueness are specified by the producing method
// (MemoryModel::conflictCandidates or MemoryModel::conflictOrder).
// Pairs have distinct execution events, nonempty footprints and at least one write.
class MemoryConflictCandidates {
 public:
  virtual ~MemoryConflictCandidates() = default;
  [[nodiscard]] virtual std::optional<MemoryConflictPair> next() = 0;
};

}  // namespace sentryward::algorithm
