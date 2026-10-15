#pragma once
#include "sentryward/algorithm/memory/access.h"
#include "sentryward/algorithm/memory/address_solver.h"
#include "sentryward/algorithm/analysis/member_relation.h"

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
struct FamilyFootprint {
  std::string space,allocation,alias,symbolic;
  address_solver::Int offset=0,step=0;
  std::uint64_t bytes=0;
  bool affine=false,empty=false;
};
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
