#pragma once
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "sentryward/algorithm/memory/family_access.h"

namespace sentryward::algorithm {
// Adjacent constraints for isolated affine footprints, and exact member-interval
// constraints for scalar or mutually overlapping footprints of one storage.
// A blocked family (of a duplicate-store exemption) disables the reduction of its whole storage:
// no family there has a candidate list, so the caller checks each against every overlapping family.
//
// Write obligations. An accepted same-value pair of two reduced families (addObligation) is
// unordered, so no chain of constraints may pass through it. Obligation groups are the connected
// components of those pairs; a family without one is a lone group. Two groups are linked when a
// constraint joins a family of each. extension() requires every overlapping pair inside a group and
// between linked groups. Once each required pair is ordered, or an obligation inside one group,
// every conflicting pair of reduced families is ordered or itself a constraint or required.
// Proof: conflicting accesses x and y share a slab (a member, for adjacent constraints), whose
// constraints join them by a path x = v_0, ..., v_n = y in rank order; a read on it is joined only
// to writes, and a write family has at most one member on it. The member sweep keeps only the first
// access of each event on a slab (MakeConflictOrder), so the proof needs every access of a group on
// the path itself: a family that shares an event with another family of its sweep takes no
// obligation (addObligation declines). From v_k jump to v_{u+1}, where v_u
// is the last access of v_k's group on the path (u = k for a read or a lone write). A jump joins
// two distinct groups that the constraint (v_u, v_{u+1}) links, so it is a constraint or required;
// it conflicts (a read jumps to a write); and it is ordered, as an unordered one would be a race or
// an obligation joining the two groups. If the jumps reach y, x is ordered before y. Otherwise they
// stop at an access of y's group, which is not lone: if that access is x, the pair is required;
// else the access reached before it lies in a linked group, so its pair with y is required and
// ordered, and ends the chain.
class FamilyConflictOrder {
 public:
  FamilyConflictOrder(const std::vector<MemoryAccessFamily>& families,
      const std::vector<FamilyFootprint>& footprints,const TraceOrder* order,
      std::span<const std::uint8_t> blocked_families = {});
  const std::vector<std::size_t>* candidates(std::size_t family) const;
  std::size_t reducedFamilies() const { return reduced_families_; }
  std::size_t memberChecks() const { return member_checks_; }
  // Records an accepted same-value write pair of two reduced families. False, recording nothing,
  // when either family shares an event with another family of its sweep; the caller then checks
  // the accesses without the reduction.
  [[nodiscard]] bool addObligation(std::size_t a,std::size_t b);
  // `overlapping(f, visit)` visits every family whose footprint may overlap that of family f.
  using Overlapping=std::function<void(std::size_t,const std::function<void(std::size_t)>&)>;
  // The required pairs of the groups joined since the last call that are neither constraints nor
  // returned before: symmetric candidate lists, by family in family order; empty when none is new.
  // For the families f of those groups, with p_f constraints and v_f visited families each:
  // O(sum of (p_f + v_f) log p_f) time, and one hash entry per returned pair.
  std::vector<std::pair<std::size_t,std::vector<std::size_t>>> extension(const Overlapping& overlapping);
 private:
  void reduceMembers(const std::vector<std::size_t>& items,const std::vector<MemoryAccessFamily>& families,
      const std::vector<FamilyFootprint>& footprints,const TraceOrder& order,std::size_t bucket);
  // The root of a family's obligation group (path halving).
  std::size_t group(std::size_t family);
  struct PairHash {
    std::size_t operator()(const std::pair<std::size_t,std::size_t>& pair) const noexcept {
      return std::hash<std::size_t>{}(pair.first*0x9e3779b97f4a7c15ULL^pair.second);
    }
  };
  std::vector<std::size_t> buckets_;
  std::vector<std::vector<std::size_t>> peers_;
  // Families of a member sweep with an event that also carries another family's access there.
  std::vector<std::uint8_t> shares_event_;
  std::size_t reduced_families_=0,member_checks_=0;
  // Obligation groups: union-find parents (allocated with the first obligation), the members of
  // each group of 2+ families by root, the roots joined since the last extension(), and the
  // pairs that extension() returned, as (lesser, greater) family.
  std::vector<std::size_t> parents_;
  std::unordered_map<std::size_t,std::vector<std::size_t>> groups_;
  std::vector<std::size_t> joined_;
  std::unordered_set<std::pair<std::size_t,std::size_t>,PairHash> required_;
};
} // namespace sentryward::algorithm
