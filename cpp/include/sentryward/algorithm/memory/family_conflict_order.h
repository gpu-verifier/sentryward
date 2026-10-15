#pragma once
#include <vector>
#include "sentryward/algorithm/memory/family_access.h"

namespace sentryward::algorithm {
// Adjacent constraints for scalar intervals or isolated affine footprints.
class FamilyConflictOrder {
 public:
  FamilyConflictOrder(const std::vector<MemoryAccessFamily>& families,
      const std::vector<FamilyFootprint>& footprints,const TraceOrder* order,
      std::span<const std::uint8_t> exempt_families = {});
  const std::vector<std::size_t>* candidates(std::size_t family) const;
  std::size_t reducedFamilies() const { return reduced_families_; }
  std::size_t memberChecks() const { return member_checks_; }
 private:
  std::vector<std::size_t> buckets_;
  std::vector<std::vector<std::size_t>> peers_;
  std::size_t reduced_families_=0,member_checks_=0;
};
} // namespace sentryward::algorithm
