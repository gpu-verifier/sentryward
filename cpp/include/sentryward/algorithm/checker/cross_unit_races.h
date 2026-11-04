#pragma once
// Cross-CTA class mode: the cross-unit global race pass. The quotient program holds only
// representative units; pairs of global accesses with at least one access in a member unit are
// checked here, exactly, on the representatives' families translated by each unit's δ (its
// allocations' offsets relative to its representative). Accesses of different units are never
// ordered (the unit-independence precondition, unit_classes.h), so every overlap with a write
// is a race, or a same-value write obligation that the caller leaves to the per-unit analysis.
// Representative pairs belong to the normal checker.
//
// Cost: O(P log P) for P global family parts, per ordered pair (A, B) of part groups (a class, or
// one piece of a class's piecewise δ) on one allocation O(|A| log |B|) to find the units whose δ
// falls in the hull window plus one O(1) memo lookup per candidate unit pair; each distinct
// translation d = δ_v - δ_u is decided once by a linear hull merge and, only if hulls meet, by
// exact ClassifyFamilyOverlap on the intersecting family pairs (an interval-tree query per write
// part). When the distinct δ of both groups form arithmetic progressions with one step (units
// tiling the allocation evenly, in any unit order), the translations are enumerated without the
// unit pairs: O(|A| + |B| + T log T) for the T < |A| + |B| translations, decided at the same first
// pair and in the same order as the window search. Two pieces of one class are also compared inside
// each member unit: O(|A|) differences of their δ, each distinct one decided once as above.
// Members are never expanded.
//
// Parts whose footprint is tensor-property lookups (FamilyFootprint::gathers) have no constant
// span. With a memory model, they are decided per allocation over pairs of unit groups (units of a
// class with equal δ on the allocation and the lookups' objects), by
// MemoryModel::translatedFamilyOverlap on both parts translated to their units. A decision is
// memoized by the relative translation, where a lookup object's shift only matters through which
// lookups it makes coincide (every element of a uniform lookup is constrained alike). Any verdict
// but Disjoint is Unknown: these parts never report a race here.
#include "sentryward/algorithm/checker/unit_quotient.h"
#include "sentryward/algorithm/memory/family_access.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace sentryward::algorithm {
class MemoryModel;
// One representative family part inside one CTA: member m in [family.first_member,
// family.first_member + family.count) is at footprint.offset + footprint.step * m (representative
// coordinates); footprint.space decides whether the part can meet another unit.
struct CrossUnitPart {
  MemoryAccessFamily family;
  FamilyFootprint footprint;
  bool write = false;
  std::uint32_t cta = 0; // linear id of the representative CTA
  // The slot site of every member, set on a written allocation with a piecewise δ in some class
  // (UnitQuotient::PiecewiseDelta): such a part is placed by its piece's δ.
  std::optional<UnitSite> site = std::nullopt;
};
// Work caps: exceeding either restarts the launch per unit. unit_pairs counts the unit pairs the
// window search visits, or the translations decided on progressions, and the member units the
// check of two pieces inside one unit visits.
struct CrossUnitLimits {
  std::uint64_t unit_pairs = 10'000'000, refinements = 1'000'000, symbolic_decisions = 100'000;
};
struct CrossUnitOutcome {
  enum class Status : std::uint8_t { kNone, kRace, kUnknown, kCap };
  Status status = Status::kNone;
  // kRace: parts[first_part] member first_member (a write) in unit first_unit meets
  // parts[second_part] member second_member in unit second_unit, on `allocation`, whose δ in the two
  // units are first_delta and second_delta; region: the overlapping bytes in launch coordinates.
  std::size_t first_part = 0, second_part = 0, first_member = 0, second_member = 0;
  std::uint32_t first_unit = 0, second_unit = 0;
  std::int64_t first_delta = 0, second_delta = 0;
  std::string allocation;
  std::uint64_t region_offset = 0, region_bytes = 0;
  std::string reason; // kUnknown / kCap: why
};
// Every pair with at least one member unit is decided exactly; parts outside global memory are
// ignored, unplaceable ones are Unknown. Allocations without a write part hold no conflicting pair
// and are skipped. On an allocation with a piecewise δ, each piece of a class is a group of its own,
// placed by the piece's δ, and two pieces of one class are also compared inside each member unit,
// where any overlap with a write is Unknown; a lookup part with a piecewise δ is Unknown.
[[nodiscard]] CrossUnitOutcome CheckCrossUnit(std::span<const CrossUnitPart> parts, const UnitQuotient &quotient,
                                              MemoryModel &model, const CrossUnitLimits &limits = {});
} // namespace sentryward::algorithm
