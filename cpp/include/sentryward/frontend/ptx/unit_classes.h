#pragma once
// Cross-CTA launch-unit classes: launch units, the static precondition (I)
// (no access of one unit can be ordered with another unit's: no global atomic,
// reduction or multimem, and no strong access or memory fence wider than the
// unit) with the (I.3') rule admitting wider proxy fences, the class-mode
// switch and its gates, and the C1 structure key. Class mode runs where the
// plan allows it (query_log.h, run.cpp); with mode off every unit is analysed.
#include "sentryward/frontend/ptx/config.h"
#include "sentryward/frontend/ptx/module.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace sentryward::frontend::ptx {
class CtaFacts;

// No public option. On (the default): class mode where the plan allows it;
// members are not consumed. Off: the per-unit loop, every unit
// consumed; selected only by the developer setting SENTRYWARD_UNIT_CLASSES=off
// (run.cpp). (The development-only modes census, verify-members and singleton
// were removed.)
enum class UnitClassMode : std::uint8_t { Off, On };
[[nodiscard]] std::optional<UnitClassMode> ParseUnitClassMode(std::string_view text) noexcept;

// A cluster, or a CTA when the launch has no clusters.
struct LaunchUnit {
  std::uint32_t index = 0;       // unit order: z, y, x over the cluster grid (x fastest)
  Coordinates kappa{};           // the cluster id (ctaid / cluster_dim), or the ctaid
  std::vector<Coordinates> ctas; // rank order: rank = x + cx*(y + cy*z) (config.cpp:910-913)
};

// The static part of precondition (I) over the launched entry's instructions,
// with the (I.3') rule. Calls are opaque summaries (no corpus entry calls a
// .func with a body), so the entry's instructions are the scan.
struct PreconditionReport {
  // No site violates a rule (R-atom, R-reduce, R-strong, R-fence, R-multimem);
  // fence.proxy.* and tensormap.cp_fenceproxy sites wider than the unit are
  // admitted by (I.3').
  bool holds = true;
  std::uint64_t tensormap_copies = 0; // tensormap.cp_fenceproxy sites: (I.5) applies
  // R-atom sites (global atom/red, every scope), by instruction index: a site
  // violates (I) only where some CTA executes it, which class mode checks on
  // every executed CTA's counts (run.cpp) and restarts per unit.
  std::vector<std::uint32_t> atomic_sites;
};
[[nodiscard]] PreconditionReport PreconditionScan(const Module &module, const Function &function);

// The run options that keep class mode off.
struct UnitGates {
  // Values are lifted through the scalar fold; values analysed
  // without it have nothing to lift (class mode stays off).
  bool value_analysis = false, scalar_fold = false;
  bool value_thread_range = false, trusted_reference = false, value_dump = false;
  bool branch_scope = false, scalar_fold_dump = false;
  bool exhaustive_search = false, continue_search = false, development_exports = false;
  // A stdin reference that could not be kept for a per-unit restart (an exec
  // of the binary re-reads it from a spooled copy; run.cpp).
  bool reference_stdin = false;
};

struct UnitPlan {
  // The units are enumerated: more than one unit and no per-CTA analysis
  // range (whatever the mode).
  bool active = false;
  // Whether class mode (representatives-only consumption) is on.
  bool class_mode = false;
  // Class mode with values: the members' values are lifted from their
  // representative's. Values add no admission condition; instead every
  // member's δ is checked against the audited row widths after value
  // analysis (run.cpp).
  bool lift_values = false;
  Coordinates grid{}, unit_shape{1, 1, 1}, unit_grid{1, 1, 1};
  std::vector<LaunchUnit> units; // filled when active
  PreconditionReport precondition;
  // Input tensor maps with a global location (allocation_location or address):
  // (I.5c) checks that no cp_fenceproxy region overlaps them.
  struct InputMap {
    // The bound allocation object of allocation_location (its resource's concrete allocation, at
    // base + offset; the resource id for a pointer resource without one); empty when it cannot be
    // placed (an absolute `address`, an unknown resource).
    std::string allocation;
    std::uint64_t offset = 0;
  };
  std::vector<InputMap> input_maps;
  [[nodiscard]] std::uint32_t UnitOf(Coordinates cta) const noexcept;
  [[nodiscard]] std::uint32_t RankOf(Coordinates cta) const noexcept;
  [[nodiscard]] std::uint32_t UnitSize() const noexcept { return unit_shape[0] * unit_shape[1] * unit_shape[2]; }
};
// A tensormap.cp_fenceproxy destination: 128 bytes at `begin` of a global allocation object.
struct TensormapRegion {
  std::string object;
  std::uint64_t begin = 0;
};
// (I.5c), on both paths: some region overlaps an input tensor map's location, or an input map
// cannot be placed while regions exist. Class mode's (I.5) restarts on it (query_log.cpp); the
// per-unit run then reports the verification limitation tensormap_input_overwrite (run.cpp), since
// descriptor fetches are not modeled as reads. O(regions x input maps).
[[nodiscard]] bool TensormapInputOverlap(const std::vector<TensormapRegion> &regions,
                                         const std::vector<UnitPlan::InputMap> &maps) noexcept;

// `config`: the launch configuration (before ConfigForCta).
[[nodiscard]] UnitPlan PlanUnits(UnitClassMode mode, const FrontendConfig &config, const Module &module,
                                 const Function &function, const UnitGates &gates);

// C1, the structure key: the canonical bytes of one CTA's structure,
// computed by the driver before any consumer reads the facts. Lanes (warp,
// lane, path), the split tree and groups, one metadata sequence per path (per
// lane on the lane path) with the prototype's bound register ids (never their
// values), lane aggregates, static counts and the rank-relative shared sites.
// Equality is byte equality; `hash` only selects candidates.
struct StructureKey {
  std::string bytes;
  std::uint64_t hash = 0;
};
[[nodiscard]] StructureKey ComputeStructureKey(const CtaFacts &facts);
} // namespace sentryward::frontend::ptx
