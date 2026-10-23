#pragma once
// Cross-CTA unit-relative value lifting: in class mode, the members' values are
// lifted from their class representative's.
//
// In lifting mode the scalar-fold lowering keeps the unit coordinate symbolic.
// Every BlockIndex leaf becomes the parameter "%unit.ctaid.<axis>", spelled
// "%unit.ctaid.<axis>+<k>" for a CTA k CTAs from its unit's origin along the
// axis (its %cluster_ctaid), so a body depends on the unit origin only, never
// on which rank of the unit it came from. Before today's consolidation every
// representative body is instantiated for its own unit and, for every member
// of its class, copied: the parameters become the unit's coordinates,
// representative-absolute global offsets move by the unit's δ and CTA-owned
// pointer provenance is renamed. A body the instantiation cannot reproduce
// exactly refuses lifting (scalar_fold_unit_parameter_unsupported), and the
// driver restarts the launch per unit.
//
// The site counters count, per process (one launch per process), where a
// BlockIndex value was declined (read as unknown) or read numerically. They
// are incremented only at BlockIndex hits; the fold and the driver compare
// them before and after a stage to decide whether lifting is refused.
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace sentryward::value_analysis {
enum class UnitSite : std::uint8_t {
  FoldKnown,             // knownAt() on BlockIndex: declined (unknown)
  FoldBindingKnown,      // knownAt() through a BlockIndex leaf binding: numeric
  FoldSpecialization,    // the lowering's ScalarSpecialization: declined (no literal)
  FoldTemplateLeaf,      // the lowering's PureSourceTemplates coordinate leaf: declined
  ValueSpecialization,   // ScalarSpecialization outside lifting (value passes, matchers): numeric
  ValueTemplateLeaf,     // PureSourceTemplates leaf outside lifting (view normalization): numeric
  Count
};
inline constexpr std::size_t kUnitSiteCount = static_cast<std::size_t>(UnitSite::Count);
inline std::array<std::atomic<std::uint64_t>, kUnitSiteCount> &UnitSiteCounters() noexcept {
  static std::array<std::atomic<std::uint64_t>, kUnitSiteCount> counters{};
  return counters;
}
inline void NoteUnitSite(UnitSite site, std::uint64_t n = 1) noexcept {
  UnitSiteCounters()[static_cast<std::size_t>(site)].fetch_add(n, std::memory_order_relaxed);
}
inline std::array<std::uint64_t, kUnitSiteCount> UnitSiteCounts() noexcept {
  std::array<std::uint64_t, kUnitSiteCount> counts{};
  for (std::size_t i = 0; i < kUnitSiteCount; ++i)
    counts[i] = UnitSiteCounters()[i].load(std::memory_order_relaxed);
  return counts;
}

// Audited row widths: the only modulus applied to an absolute
// global offset in the value passes and the fold is a row map's row width R
// (PureSourceTemplates: an initial-memory 2-D matrix leaf's coordinate is its
// absolute offset, and row maps take coordinate / R). Per allocation, the row
// widths of such leaves (a superset of the anchors used): every member's δ of
// the allocation must be a multiple of each (checked in run.cpp after value
// analysis). Value analysis is single-threaded.
inline std::map<std::string, std::set<std::uint64_t>> &UnitRowModuli() {
  static std::map<std::string, std::set<std::uint64_t>> moduli;
  return moduli;
}

// The launch geometry and class tables lifting needs
// (ScalarFoldOptions::unit_lifting). Class mode (`classes`, set by the
// driver): only the representative units were analysed; every representative
// output is instantiated for its own unit in place and, for every member of
// its class, appended as a copy at the member's origin, δ and CTA ids.
// Without `classes` the lowering refuses lifting.
struct UnitLifting {
  using Origin = std::array<std::uint64_t, 3>;
  std::array<std::uint64_t, 3> grid{1, 1, 1}, unit_shape{1, 1, 1};
  bool classes = false;
  struct Member {
    Origin origin{};                             // the member unit's first CTA (its %unit.ctaid)
    std::map<std::string, std::int64_t> delta;   // per global allocation, keyed as ConstantRegion::allocation
    std::map<std::uint64_t, std::uint64_t> ctas; // representative CTA linear id -> the member's (same rank)
  };
  // Per representative unit origin: its members in unit order. A
  // representative of a class without members has no entry.
  std::map<Origin, std::vector<Member>> members;
};
inline constexpr std::string_view kUnitCoordinatePrefix = "%unit.ctaid.";
// "%unit.ctaid.<axis>" or "%unit.ctaid.<axis>+<offset>".
inline std::string UnitCoordinateParameter(unsigned axis, std::uint64_t offset) {
  std::string name(kUnitCoordinatePrefix);
  name += "xyz"[axis];
  if (offset) name += '+' + std::to_string(offset);
  return name;
}
} // namespace sentryward::value_analysis
