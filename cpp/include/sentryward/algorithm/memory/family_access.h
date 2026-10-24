#pragma once
#include "sentryward/algorithm/memory/access.h"
#include "sentryward/algorithm/memory/address_solver.h"
#include "sentryward/algorithm/analysis/member_relation.h"

#include <algorithm>
#include <utility>

namespace sentryward::algorithm {
struct MemoryAccessFamily {
  MemoryAccess first;
  MemoryAccessCodec::Delta delta;
  std::size_t count=0, event_family=0, first_member=0, extent=0;
  MemoryAccess at(std::size_t member) const {
    if(member<first_member || member-first_member>=count)throw std::out_of_range("access family member");
    auto result=first;MemoryAccessCodec::Binding binding;
    MemoryAccessCodec::split(result,binding);
    auto translated=MemoryAccessCodec::translate(binding,delta,member-first_member);
    if(!translated)throw std::overflow_error("access family binding");
    MemoryAccessCodec::bind(result,*translated);return result;
  }
  MemberSet members() const { return MemberRange(extent,first_member,first_member+count); }
};
// One term of a symbolic suffix that is a tensor-property lookup: coefficient x the element of
// `object` at byte `arg` (address-model symbol `symbol`). `uniform`: the model constrains every
// element of the symbol alike (no address invariant names one), so a decision depends only on
// which lookups coincide, not on their byte offsets.
struct GatherTerm {
  std::string symbol,object;
  address_solver::Int arg=0,coefficient=0;
  bool uniform=false;
};
struct FamilyFootprint {
  std::string space,allocation,alias,symbolic;
  address_solver::Int offset=0,step=0;
  std::uint64_t bytes=0;
  bool affine=false,empty=false;
  // The symbolic suffix as lookups, when all of its terms are tensor-property lookups (else empty).
  std::vector<GatherTerm> gathers;
  // A strided region (a TMA box; MemoryModel::crossUnitFootprint only): each member covers
  // `region_element` bytes at every offset + sum of stride x index (index < extent), within
  // `bytes` from its first byte. Empty: each member is `bytes` contiguous bytes.
  std::vector<std::pair<std::uint64_t,std::uint64_t>> region_axes = {};
  std::uint64_t region_element = 0;
};
// The bytes [lo, hi) that the members of an affine footprint start in, plus `bytes`: from the
// lowest member's offset to the highest member's offset + bytes.
inline std::pair<address_solver::Int,address_solver::Int> FootprintSpan(const MemoryAccessFamily& family,
    const FamilyFootprint& footprint) {
  using I=address_solver::Int;
  const I a=footprint.offset+footprint.step*I(family.first_member),
      b=footprint.offset+footprint.step*I(family.first_member+family.count-1);
  return {std::min(a,b),std::max(a,b)+I(footprint.bytes)};
}
// A differing symbolic suffix remains Unknown; equal suffixes cancel exactly.
struct FamilyOverlap {
  std::vector<MemberRelation> overlap,unknown;
  std::size_t member_rows=0;
};
Result<std::vector<MemoryAccessFamily>,MemoryAnalysisError> ExtractMemoryAccessFamilies(
    const HappensBeforeGraph& graph,const SourceProgram& source);
// Derive families from validated compute records without revisiting source operations.
std::vector<MemoryAccessFamily> BindMemoryAccessFamilies(
    const EventRecords& events, const MemoryAccessRecords& accesses);
FamilyOverlap ClassifyFamilyOverlap(const MemoryAccessFamily& a,const FamilyFootprint& x,
    const MemoryAccessFamily& b,const FamilyFootprint& y);
} // namespace sentryward::algorithm
