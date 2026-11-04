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
#include <optional>
#include <span>
#include <string>
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
  };
  std::vector<Unit> units;    // unit order (z, y, x over the unit grid)
  std::vector<Class> classes; // creation order

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
  // Whether the analysed program numbers the representative CTA `linear` (its instruction, actor and
  // barrier lifetime ids) as the per-unit program does, which consumes every CTA in launch order: it
  // is the launch's first CTA, or units are runs of consecutive CTAs and no member unit precedes its
  // unit. `first_member`: FirstMemberUnit().
  [[nodiscard]] bool NumberedAsPerUnit(std::uint32_t linear, std::uint32_t first_member) const noexcept {
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
} // namespace sentryward::algorithm
