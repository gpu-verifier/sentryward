#pragma once
// Cross-CTA class mode: the quotient tables. A launch unit is a cluster (or a CTA without
// clusters); units whose analysis is identical form a class, analysed once through its
// representative unit. The tables say how the analysed program, built from the representative
// units only, stands for the whole launch: every consumed CTA belongs to a representative unit,
// and consumers use the tables to weight per-occurrence evidence by class size, to expand
// bounded samples in the per-unit path's order and to run the cross-unit global race pass. Every pointer to
// it is null when class mode is off, and then nothing below is consulted.
//
// Cost: every query is O(1) or O(log) in units; nothing here enumerates members except
// the callers' bounded sample expansions.
#include "sentryward/algorithm/source_ir/program.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sentryward::algorithm {
// The launch units of a launch with whole-CTA analysis (a cluster, or a CTA without clusters), on
// both paths: units are numbered in unit order (z, y, x over the unit grid), CTAs linearly (z, y, x),
// and a CTA's rank is its position in its unit. Repeated entries name units by these indices
// (source_json.cpp).
struct LaunchUnits {
  using Coordinates = std::array<std::uint32_t, 3>;
  Coordinates grid{1, 1, 1}, unit_shape{1, 1, 1};

  [[nodiscard]] std::uint32_t Linear(const Coordinates &cta) const noexcept {
    return cta[0] + grid[0] * (cta[1] + grid[1] * cta[2]);
  }
  [[nodiscard]] Coordinates CtaCoordinates(std::uint32_t linear) const noexcept {
    return {linear % grid[0], (linear / grid[0]) % grid[1], linear / (grid[0] * grid[1])};
  }
  [[nodiscard]] std::uint32_t UnitOf(std::uint32_t linear) const noexcept {
    const auto c = CtaCoordinates(linear);
    const Coordinates g{grid[0] / unit_shape[0], grid[1] / unit_shape[1], grid[2] / unit_shape[2]};
    return c[0] / unit_shape[0] + g[0] * (c[1] / unit_shape[1] + g[1] * (c[2] / unit_shape[2]));
  }
  [[nodiscard]] std::uint32_t RankOf(std::uint32_t linear) const noexcept {
    const auto c = CtaCoordinates(linear);
    return c[0] % unit_shape[0] + unit_shape[0] * (c[1] % unit_shape[1] + unit_shape[1] * (c[2] % unit_shape[2]));
  }
};

// A slot's site: the PTX instruction index and occurrence of its record in the CTA of rank `rank` of
// its unit (the PtxOrigin of the source instructions lowered from that record).
struct UnitSite {
  std::uint32_t rank = 0, instruction = 0, occurrence = 0;
  friend auto operator<=>(const UnitSite &, const UnitSite &) = default;
};
// The δ of a global allocation whose slots move by different amounts in some unit of a class: its
// slot sites, grouped into pieces (the sites with one δ in every unit of the class), and each piece's
// δ per unit. Every access of the allocation in the class's units is placed by the slot at its own
// site (lowering resolves each footprint at the record it lowers).
struct PiecewiseDelta {
  std::vector<UnitSite> sites;                  // sorted, distinct
  std::vector<std::uint32_t> piece;             // per site: its piece
  std::vector<std::vector<std::int64_t>> delta; // per piece: δ per Class::units[i] (0 for i = 0)
  // The piece of `site`, nullopt when it is not a slot site of the allocation. O(log sites).
  [[nodiscard]] std::optional<std::uint32_t> PieceOf(const UnitSite &site) const {
    const auto found = std::lower_bound(sites.begin(), sites.end(), site);
    if (found == sites.end() || *found != site) return std::nullopt;
    return piece[static_cast<std::size_t>(found - sites.begin())];
  }
};

struct UnitQuotient : LaunchUnits {
  struct Unit {
    Coordinates kappa{};
    std::uint32_t klass = 0;
    std::vector<std::uint32_t> ctas; // linear CTA ids, rank order
  };
  struct Class {
    std::vector<std::uint32_t> units;                       // unit order; units[0] is the representative
    std::vector<std::map<std::string, std::int64_t>> delta; // per units[i]: δ per global allocation ({} for units[0])
    // The allocations with a piecewise δ, which no delta[i] holds.
    std::map<std::string, PiecewiseDelta> pieces;
    // Per units[i]: value lifting's δ, the unit's exact global offsets against the representative's
    // ({} for units[0]); nullopt when they are not one δ per allocation (a class-mode table atom
    // read other elements). Without table atoms it agrees with delta[i] on the allocations it names.
    std::vector<std::optional<std::map<std::string, std::int64_t>>> valued_delta;
    // Per units[i]: a containment member (its events are the representative's minus some plain
    // global accesses: frontend/ptx/query_log.h UnitDelta::contained). Its accesses are a subset
    // of the representative's translated ones, so the cross-unit pass over-approximates them.
    std::vector<bool> contained;
    // Per rank: the containment members' executed-record count changes per instruction, summed
    // (lowering's opcode counts add them to the representative's count times the class size).
    std::vector<std::map<std::uint32_t, std::int64_t>> occurrence_adjust;
    // The global allocations some containment member reads only through TMA loads compared without
    // their global placement (frontend/ptx/query_log.h UnitDelta::relaxed): no unit may write them.
    std::set<std::string> relaxed_reads;
    // Per units[i], for a containment member: per rank and lane (the index of the lane's actor in its
    // CTA), the representative's plain global accesses that the member does not perform, as sorted
    // (PTX instruction, occurrence) pairs; lanes with equal lists share one. Empty for other units.
    // Some containment member is marked (frontend/ptx/query_log.h UnitDelta::hazard): it may have
    // register pairs of one thread (two writes of a register by successive tcgen05.ld, and that
    // register's later accesses, with no read of it and no tcgen05.wait::ld ordering them) that are
    // unordered in the member and ordered in its representative. Everything else of the member is its
    // representative's. A class with such a member never reports a clean memory result
    // (ContainmentRestartReason).
    bool hazard = false;
    using RemovedAccesses = std::vector<std::pair<std::uint32_t, std::uint32_t>>;
    std::vector<std::vector<std::vector<std::shared_ptr<const RemovedAccesses>>>> removed;
  };
  std::vector<Unit> units;    // unit order (z, y, x over the unit grid)
  std::vector<Class> classes; // creation order
  // The class loop took planned founders first (frontend/ptx/query_log.h ContainmentPlan): a
  // representative need not be the lowest unit of its class, a class's members need not be in unit
  // order, and the analysed program is not numbered as the per-unit one. The driver accepts such an
  // attempt only when it ends in a clean pass.
  bool reordered = false;

  // Filled by lowering at Finish: how many orders the per-CTA lowering emitted before Finish
  // (lifecycle witness expansion).
  std::size_t add_time_orders = 0;

  [[nodiscard]] const Class &ClassOf(std::uint32_t linear) const { return classes.at(units.at(UnitOf(linear)).klass); }
  // The class size of the unit of `linear` (a consumed CTA's multiplicity).
  [[nodiscard]] std::uint64_t Weight(std::uint32_t linear) const { return ClassOf(linear).units.size(); }
  // The same-rank CTA of the representative of `linear`'s class (itself for a representative CTA).
  [[nodiscard]] std::uint32_t RepresentativeCta(std::uint32_t linear) const {
    return units.at(ClassOf(linear).units.front()).ctas.at(RankOf(linear));
  }
  [[nodiscard]] bool Shared() const noexcept { return classes.size() < units.size(); }
  // Whether some class has an allocation with a piecewise δ. O(classes).
  [[nodiscard]] bool Piecewise() const noexcept {
    return std::any_of(classes.begin(), classes.end(), [](const Class &c) { return !c.pieces.empty(); });
  }
  // Whether some class has a containment member. O(units).
  [[nodiscard]] bool Contained() const noexcept {
    return std::any_of(classes.begin(), classes.end(), [](const Class &c) {
      return std::find(c.contained.begin(), c.contained.end(), true) != c.contained.end();
    });
  }
  // The allocations no unit may write (Class::relaxed_reads of every class). O(classes x allocations).
  [[nodiscard]] std::set<std::string> RelaxedReads() const {
    std::set<std::string> result;
    for (const auto &c : classes) result.insert(c.relaxed_reads.begin(), c.relaxed_reads.end());
    return result;
  }
  // Whether Class::units[index] of class `klass`, a containment member, lacks its representative's
  // plain global access at (rank, lane, PTX instruction, occurrence). False for any other unit and
  // for an access outside the recorded lists. O(log removed accesses of the lane).
  [[nodiscard]] bool Removed(std::uint32_t klass, std::size_t index, std::uint32_t rank, std::uint32_t lane,
                             std::uint32_t instruction, std::uint32_t occurrence) const {
    const auto &c = classes.at(klass);
    if (index >= c.removed.size() || rank >= c.removed[index].size() || lane >= c.removed[index][rank].size())
      return false;
    const auto &list = c.removed[index][rank][lane];
    return list && std::binary_search(list->begin(), list->end(), std::pair{instruction, occurrence});
  }
  // Whether some class has a marked member (Class::hazard). O(classes).
  [[nodiscard]] bool Hazard() const noexcept {
    return std::any_of(classes.begin(), classes.end(), [](const Class &c) { return c.hazard; });
  }
  // Whether unit `unit` is a containment member. O(class size).
  [[nodiscard]] bool ContainedUnit(std::uint32_t unit) const {
    const auto &c = classes.at(units.at(unit).klass);
    for (std::size_t i = 0; i < c.units.size() && i < c.contained.size(); ++i)
      if (c.units[i] == unit) return c.contained[i];
    return false;
  }
  // Whether the analysed program numbers the representative CTA `linear` (its instruction, actor and
  // barrier lifetime ids) as the per-unit program does, which consumes every CTA in launch order: it
  // is the launch's first CTA, or units are runs of consecutive CTAs and no member unit precedes its
  // unit. `first_member`: FirstMemberUnit().
  [[nodiscard]] bool NumberedAsPerUnit(std::uint32_t linear, std::uint32_t first_member) const noexcept {
    if (reordered) return false;
    return linear == 0 || (unit_shape[1] == 1 && unit_shape[2] == 1 && UnitOf(linear) < first_member);
  }
  // The lowest unit analysed through its class's representative; units.size() when none. O(units).
  [[nodiscard]] std::uint32_t FirstMemberUnit() const noexcept {
    auto first = static_cast<std::uint32_t>(units.size());
    for (const auto &c : classes)
      for (std::size_t m = 1; m < c.units.size(); ++m) first = std::min(first, c.units[m]);
    return first;
  }
};

// The multiplicity of every actor of a quotient program: the class size of its CTA's unit.
// O(actors) to build; O(1) per query.
class ActorWeights {
public:
  ActorWeights(const UnitQuotient &quotient, std::span<const ActorInfo> actors) {
    std::uint32_t size = 0;
    for (const auto &actor : actors) size = std::max(size, actor.id.value() + 1);
    weights_.assign(size, 1);
    for (const auto &actor : actors) weights_[actor.id.value()] = quotient.Weight(actor.cta.value());
  }
  [[nodiscard]] std::uint64_t operator()(ActorId actor) const {
    return actor.value() < weights_.size() ? weights_[actor.value()] : 1;
  }

private:
  std::vector<std::uint64_t> weights_;
};

// Class mode cannot finish exactly (unit closure, a cross-unit Unknown or a work cap): the
// driver drops the quotient and re-runs the launch per unit. Deliberately not a
// std::exception, so that no analysis stage converts it into a report.
struct UnitClassRestart {
  std::string reason;
};

// Whether a clean memory search of the founders may stand for the containment members of their
// classes. OFF: containment is race-only until the containment argument (frontend/ptx/query_log.cpp
// ContainMap; a member's accesses are a subset of its representative's, and only a marked member has
// pairs its representative lacks) has had an independent review. With it off, the soundness of a
// class-mode result does not rest on that argument: founders are real units of the launch, checked
// in full, and nothing is reported for a containment member that its own lowering did not state.
// Setting it to true enables reporting clean through unmarked containment members (the path below
// and founder planning in the driver, both kept and unit-tested).
inline constexpr bool kContainmentReportsClean = false;

// Class mode with containment members: what the founders' memory search means for the launch
// (checker/memory_race_checker.cpp RunMemoryCheck). The reason of a per-unit restart, or empty when
// the search's result goes on as it is.
//  - A race: it stands. Its two accesses are accesses of analysed units, which are real units of the
//    launch lowered exactly, so it is a race of the launch whichever pair another search order would
//    name first (class mode reports it likewise for classes of equal units). The driver redoes a
//    planned-founder attempt in unit order first, so the reported pair comes from launch order.
//  - An unresolved pair, or a same-value obligation (counted by class size, which a member that
//    removed one of its writes would overstate): per-unit restart.
//  - A clean search, with `reports_clean` off (kContainmentReportsClean): per-unit restart whenever
//    some class holds a containment member, marked or not.
//  - A clean search, with `reports_clean` on: it covers every containment member (its accesses are a
//    subset of its representative's, translated), except a marked member's own register pairs
//    (Class::hazard), which no analysed unit shows: the class must not report clean, so per-unit
//    restart. The marked member's pair is not reported from here: the member is not lowered, and only
//    its own lowering states the pair's two events.
[[nodiscard]] inline std::string_view ContainmentRestartReason(const UnitQuotient &units, bool race, bool unresolved,
                                                               bool same_value_obligations,
                                                               bool reports_clean = kContainmentReportsClean) noexcept {
  if (race || !units.Contained()) return {};
  if (unresolved || same_value_obligations) return "containment_unclean";
  if (!reports_clean) return "containment_clean_not_reported";
  if (units.Hazard()) return "containment_register_hazard";
  return {};
}
} // namespace sentryward::algorithm
