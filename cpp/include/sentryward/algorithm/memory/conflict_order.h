#pragma once

#include <memory>
#include <span>
#include <vector>

#include "sentryward/algorithm/memory/conflict_candidates.h"

namespace sentryward::algorithm {

// Regions in different groups are disjoint; within a group, byte intervals
// describe overlap exactly. Register footprints use the synthetic interval [0,1).
struct ConstantConflictRegion final {
  AccessId access;
  EventId event;
  bool write;
  std::size_t group;
  std::uint64_t offset;
  std::uint64_t bytes;
};

// Sufficient ordering obligations, NOT an enumeration of every conflicting pair.
// All obligations hold iff every conflicting pair is ordered. Each emitted pair
// itself conflicts. Only use for existence checks that stop on the first failure.
// Ranks must outlive the stream and be a topological rank for each event.
[[nodiscard]] std::unique_ptr<MemoryConflictCandidates> MakeConflictOrder(
    std::vector<ConstantConflictRegion> regions,
    std::span<const std::size_t> ranks);

}  // namespace sentryward::algorithm
