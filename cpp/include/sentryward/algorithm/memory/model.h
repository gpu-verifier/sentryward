#pragma once

#include <optional>
#include <memory>
#include <span>

#include "sentryward/algorithm/memory/access.h"
#include "sentryward/algorithm/memory/family_access.h"
#include "sentryward/algorithm/memory/overlap.h"
#include "sentryward/algorithm/memory/conflict_candidates.h"
#include "sentryward/algorithm/base/dynamic_bitset.h"

namespace sentryward::algorithm {

class MemoryStorageFilter {
 public:
  virtual ~MemoryStorageFilter() = default;
  // Clear only accesses proven storage-disjoint from the selected access.
  virtual void retainCompatible(AccessId access, DynamicBitset& candidates) = 0;
};

class MemoryModel {
 public:
  virtual ~MemoryModel() = default;

  // Null declines family classification; callers retain the exact legacy path.
  [[nodiscard]] virtual std::optional<FamilyFootprint> familyFootprint(
      const MemoryAccessFamily&) { return std::nullopt; }

  [[nodiscard]] virtual std::optional<MemoryAnalysisError> validate() const {
    return std::nullopt;
  }

  // Queries are symmetric and use one fixed address/constraint context.
  // Overlap establishes an intersection; Unknown leaves it unresolved.
  [[nodiscard]] virtual OverlapResult mayOverlap(const MemoryAccess& first,
                                                const MemoryAccess& second) = 0;

  // Optional coarse filter; IDs must index the supplied access table.
  // The returned filter owns its index and preserves all possible overlaps.
  [[nodiscard]] virtual std::unique_ptr<MemoryStorageFilter> storageFilter(
      MemoryAccessView) { return nullptr; }

  // Null selects the generic HB-first path. The stream must retain all Unknown pairs.
  [[nodiscard]] virtual std::unique_ptr<MemoryConflictCandidates> conflictCandidates(
      MemoryAccessView) { return nullptr; }

  // Optional exact existence shortcut. Emits a sufficient subset of conflicting
  // pairs oriented by topological rank: all ordered iff no conflict is unordered.
  // Every emitted pair must have known Overlap; duplicates are permitted. Null
  // declines the shortcut. Not suitable for counting pairs or ignoring failures
  // as warnings. Ranks must outlive the returned stream.
  [[nodiscard]] virtual std::unique_ptr<MemoryConflictCandidates> conflictOrder(
      MemoryAccessView, std::span<const std::size_t>) { return nullptr; }

  // Existence shortcut with pair exemptions. All potentially exempt writes must
  // be supplied. Preserve every conflicting pair involving their storage; other
  // pairs may use sufficient ordering constraints. No Unknown pair may be lost.
  // Complete scans preserve exemption counts; not for exhaustive race counting.
  // Ranks must outlive the stream. Null declines this shortcut.
  [[nodiscard]] virtual std::unique_ptr<MemoryConflictCandidates>
  conflictCandidatesWithExemptions(MemoryAccessView,
      std::span<const std::size_t>, std::span<const AccessId>) { return nullptr; }
};

}  // namespace sentryward::algorithm
