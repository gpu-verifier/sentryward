#pragma once
// Cross-CTA class mode: the quotient tables. A launch unit is a cluster (or a CTA without
// clusters); units whose analysis is identical form a class, analysed once through its
// representative unit. The tables say how the analysed program, built from the representative
// units only, stands for the whole launch: every consumed CTA belongs to a representative unit,
// and consumers use the tables to weight per-occurrence evidence by class size, to expand
// bounded samples in today's order and to run the cross-unit global race pass. Every pointer to
// it is null when class mode is off, and then nothing below is consulted.
//
// Cost: every query is O(1) or O(log) in units; nothing here enumerates members except
// the callers' bounded sample expansions.
#include "sentryward/algorithm/source_ir/program.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <map>
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

struct UnitQuotient : LaunchUnits {
  struct Unit {
    Coordinates kappa{};
    std::uint32_t klass = 0;
    std::vector<std::uint32_t> ctas; // linear CTA ids, rank order
  };
  struct Class {
    std::vector<std::uint32_t> units;                       // unit order; units[0] is the representative
    std::vector<std::map<std::string, std::int64_t>> delta; // per units[i]: δ per global allocation ({} for units[0])
  };
  std::vector<Unit> units;    // unit order (z, y, x over the unit grid)
  std::vector<Class> classes; // creation order

  // Filled by lowering at Finish: the CTA of every required order (that of its `from` actor) and how
  // many orders the per-CTA lowering emitted before Finish (lifecycle witness expansion).
  std::vector<std::uint32_t> order_ctas;
  std::size_t add_time_orders = 0;

  [[nodiscard]] const Class &ClassOf(std::uint32_t linear) const { return classes.at(units.at(UnitOf(linear)).klass); }
  // The class size of the unit of `linear` (a consumed CTA's multiplicity).
  [[nodiscard]] std::uint64_t Weight(std::uint32_t linear) const { return ClassOf(linear).units.size(); }
  // The same-rank CTA of the representative of `linear`'s class (itself for a representative CTA).
  [[nodiscard]] std::uint32_t RepresentativeCta(std::uint32_t linear) const {
    return units.at(ClassOf(linear).units.front()).ctas.at(RankOf(linear));
  }
  // The class's units as [first, last] runs of unit indices (the "members" field of repeated entries).
  [[nodiscard]] std::vector<std::pair<std::uint32_t, std::uint32_t>> UnitRuns(std::uint32_t klass) const {
    std::vector<std::pair<std::uint32_t, std::uint32_t>> runs;
    for (const auto unit : classes.at(klass).units)
      if (!runs.empty() && runs.back().second + 1 == unit) runs.back().second = unit;
      else runs.emplace_back(unit, unit);
    return runs;
  }
  [[nodiscard]] bool Shared() const noexcept { return classes.size() < units.size(); }
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
    ctas_.assign(size, UINT32_MAX);
    weights_.assign(size, 1);
    for (const auto &actor : actors) {
      ctas_[actor.id.value()] = actor.cta.value();
      weights_[actor.id.value()] = quotient.Weight(actor.cta.value());
    }
  }
  [[nodiscard]] std::uint64_t operator()(ActorId actor) const {
    return actor.value() < weights_.size() ? weights_[actor.value()] : 1;
  }
  [[nodiscard]] std::uint32_t Cta(ActorId actor) const {
    return actor.value() < ctas_.size() ? ctas_[actor.value()] : UINT32_MAX;
  }

private:
  std::vector<std::uint32_t> ctas_;
  std::vector<std::uint64_t> weights_;
};

// Class mode cannot finish exactly (unit closure, a cross-unit Unknown or a work cap, (I.5)): the
// driver drops the quotient and re-runs the launch per unit. Deliberately not a
// std::exception, so that no analysis stage converts it into a report.
struct UnitClassRestart {
  std::string reason;
};
} // namespace sentryward::algorithm
