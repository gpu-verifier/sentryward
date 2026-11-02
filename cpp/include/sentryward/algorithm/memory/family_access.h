#pragma once
#include "sentryward/algorithm/memory/access.h"
#include "sentryward/algorithm/memory/address_solver.h"
#include "sentryward/algorithm/analysis/member_relation.h"

#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace sentryward::algorithm {
struct MemoryAccessFamily {
  // The MemoryAccessCodec binding fields (MemoryAccessCodec::split) of the instruction and actor.
  static constexpr std::size_t kInstructionField=2,kActorField=3;
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
  // The actor of a member, affine in it, without reconstructing the access.
  ActorId actor(std::size_t member) const {
    const auto steps=std::int64_t(member)-std::int64_t(first_member);
    return ActorId(static_cast<std::uint32_t>(std::int64_t(first.actor.value())+delta[kActorField]*steps));
  }
  MemberSet members() const { return MemberRange(extent,first_member,first_member+count); }
};
// The CTA of each actor of a source program, by actor id (kNoCta for an unknown actor). O(actors).
// `contiguous`: no actor id of one CTA lies between two actor ids of another, so the members of an
// affine family that lie in one CTA form one member range (ForEachCtaRange).
class ActorCtas final {
 public:
  static constexpr std::uint32_t kNoCta=UINT32_MAX;
  explicit ActorCtas(std::span<const ActorInfo> actors) {
    for(const auto& actor:actors) {
      if(actor.id.value()>=ctas_.size())ctas_.resize(std::size_t(actor.id.value())+1,kNoCta);
      ctas_[actor.id.value()]=actor.cta.value();
    }
    // The CTA of each run of one CTA in id order: a CTA with two runs is not contiguous.
    std::vector<std::uint32_t> runs;
    for(const auto cta:ctas_)if(cta!=kNoCta && (runs.empty() || runs.back()!=cta))runs.push_back(cta);
    std::sort(runs.begin(),runs.end());
    contiguous_=std::adjacent_find(runs.begin(),runs.end())==runs.end();
  }
  std::uint32_t operator()(ActorId actor) const {
    return actor.value()<ctas_.size()?ctas_[actor.value()]:kNoCta;
  }
  // The CTA of a known actor; throws std::out_of_range for an unknown one.
  std::uint32_t at(ActorId actor) const {
    const auto cta=(*this)(actor);
    if(cta==kNoCta)throw std::out_of_range("actor has no CTA");
    return cta;
  }
  bool contiguous() const noexcept { return contiguous_; }
 private:
  std::vector<std::uint32_t> ctas_;
  bool contiguous_=true;
};
// visit(lo, hi) for each maximal member range of [begin, end) on which cta_of(member) is constant,
// where each value of cta_of holds one range (ActorCtas::contiguous, for the actors of an affine
// family). O(ranges x log(end - begin)) calls of cta_of.
template<class CtaOf,class Visit>
void ForEachCtaRange(std::size_t begin,std::size_t end,const CtaOf& cta_of,const Visit& visit) {
  while(begin<end) {
    const auto cta=cta_of(begin);
    std::size_t lo=begin,hi=end; // the last member of `cta` is in [lo, hi)
    if(cta_of(end-1)!=cta)
      while(hi-lo>1) {
        const auto middle=lo+(hi-lo)/2;
        (cta_of(middle)==cta?lo:hi)=middle;
      }
    else lo=end-1;
    visit(begin,lo+1);
    begin=lo+1;
  }
}
// One term of a symbolic suffix that is a tensor-property lookup: coefficient x the element of
// `object` at byte `arg` (address-model symbol `symbol`). `uniform`: the model constrains every
// element of the symbol alike (no address invariant names one), so a decision depends only on
// which lookups coincide, not on their byte offsets.
struct GatherTerm {
  std::string symbol,object;
  address_solver::Int arg=0,coefficient=0;
  bool uniform=false;
};
// The storage every member of a family is bound to, known even when its offsets are not.
struct FamilyStorage {
  std::string space,allocation,alias;
};
// A named address space (not empty, "generic" or "unknown"): accesses in two different known
// spaces never share storage.
inline bool KnownSpace(std::string_view space) {
  return !space.empty() && space!="generic" && space!="unknown";
}
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
