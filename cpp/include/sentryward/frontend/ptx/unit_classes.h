#pragma once
// Cross-CTA launch-unit classes: launch units, the static part of the
// unit-independence precondition (no access of one unit can be ordered with
// another unit's: no global atomic, reduction or multimem, and no strong access
// or memory fence wider than the unit) with its exception admitting wider proxy
// fences, the class-mode gates, and the structure key. Class mode runs where
// the plan allows it (query_log.h, run.cpp); otherwise every unit is analysed.
#include "sentryward/frontend/ptx/config.h"
#include "sentryward/frontend/ptx/module.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace sentryward::frontend::ptx {
class CtaFacts;

// A cluster, or a CTA when the launch has no clusters.
struct LaunchUnit {
  std::uint32_t index = 0;       // unit order: z, y, x over the cluster grid (x fastest)
  Coordinates kappa{};           // the cluster id (ctaid / cluster_dim), or the ctaid
  std::vector<Coordinates> ctas; // rank order: rank = x + cx*(y + cy*z) (as ConfigForCta's %cluster_ctarank)
};

// The static part of the unit-independence precondition over the launched
// entry's instructions, with the proxy-fence exception. Calls are opaque
// summaries (no corpus entry calls a .func with a body), so the entry's
// instructions are the scan.
struct PreconditionReport {
  // No site is a global reduction, a multimem access, or a strong access or
  // memory fence wider than the unit (global atomics: atomic_sites);
  // fence.proxy.* and tensormap.cp_fenceproxy sites wider than the unit are
  // admitted as proxy fences.
  bool holds = true;
  std::uint64_t tensormap_copies = 0; // tensormap.cp_fenceproxy sites: the tensormap locality rule applies
  // Global atomic sites (global atom/red, every scope), by instruction index: a
  // site violates the precondition only where some CTA executes it, which class
  // mode checks on every executed CTA's counts (run.cpp) and restarts per unit.
  std::vector<std::uint32_t> atomic_sites;
};
[[nodiscard]] PreconditionReport PreconditionScan(const Function &function);

// The run options that keep class mode off.
struct UnitGates {
  // Values are lifted through the scalar fold; values analysed
  // without it have nothing to lift (class mode stays off).
  bool value_analysis = false, scalar_fold = false;
  bool value_thread_range = false, trusted_reference = false, value_dump = false;
  bool branch_scope = false, scalar_fold_dump = false;
  bool exhaustive_search = false, continue_search = false, development_exports = false;
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
  Coordinates grid{}, unit_shape{1, 1, 1};
  std::vector<LaunchUnit> units; // filled when active
  PreconditionReport precondition;
  // Input tensor maps with a global location (allocation_location or address):
  // the tensormap locality rule checks that no cp_fenceproxy region overlaps them.
  struct InputMap {
    // The bound allocation object of allocation_location (its resource's concrete allocation, at
    // base + offset; the resource id for a pointer resource without one); empty when it cannot be
    // placed (an absolute `address`, an unknown resource).
    std::string allocation;
    std::uint64_t offset = 0;
  };
  std::vector<InputMap> input_maps;
  [[nodiscard]] std::uint32_t UnitSize() const noexcept { return unit_shape[0] * unit_shape[1] * unit_shape[2]; }
};
// A tensormap.cp_fenceproxy destination: 128 bytes at `begin` of a global allocation object.
struct TensormapRegion {
  std::string object;
  std::uint64_t begin = 0;
};
// The input-map part of the tensormap locality rule, on both paths: some region overlaps an
// input tensor map's location, or an input map cannot be placed while regions exist. Class
// mode's locality check restarts on it (query_log.cpp); the per-unit run then reports the
// verification limitation tensormap_input_overwrite (run.cpp), since descriptor fetches are
// not modeled as reads. O(regions x input maps).
[[nodiscard]] bool TensormapInputOverlap(const std::vector<TensormapRegion> &regions,
                                         const std::vector<UnitPlan::InputMap> &maps) noexcept;

// `config`: the launch configuration (before ConfigForCta).
[[nodiscard]] UnitPlan PlanUnits(const FrontendConfig &config, const Function &function,
                                 const UnitGates &gates);

// The structure key: the canonical bytes of one CTA's structure,
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

// Containment classes (query_log.cpp): a unit whose structure key differs from a representative's may
// still be admitted when, lane by lane, its record sequence is the representative's with some
// records removed and some taken branches added: the removed records are plain synchronous global
// loads and stores, pure compute or opaque compute regions, whose lowering changed no state later
// records' lowering reads (ContainmentRecord::touched), and the added ones are `bra` records, which
// lower to nothing. Its events are then the representative's minus the removed records' events
// (UnitDelta::contained).
//   kAccess:  ld/st of the global state space with no async, bulk or strong semantics
//   kCompute: a register-only arithmetic, logic, move, comparison or conversion instruction
//   kBranch:  bra (no register operand)
//   kOpaque:  an opaque compute region (a data-dependent forward branch over pure computation whose
//             outcome the executor left open): removable like kCompute; a member may have the same
//             branch taken (kBranch) in its place
//   kFixed:   anything else, and any record with other executor metadata beyond its predicate
enum class ContainmentKind : std::uint8_t { kAccess, kCompute, kBranch, kOpaque, kFixed };
struct ContainmentRecord {
  std::uint32_t instruction = 0, occurrence = 0;
  // A hash of the record's register origins (the defining record of each register it reads), and of
  // every other ExecutionMetadata field but the predicate.
  std::uint64_t origins = 0, meta = 0;
  // The record's origins: [origin_begin, origin_begin + origin_count) of its path's `origins`.
  std::uint32_t origin_begin = 0, origin_count = 0;
  std::uint8_t predicate = 0;  // 0: unpredicated, 1: predicate true, 2: predicated off
  ContainmentKind kind = ContainmentKind::kFixed;
  // Lowering reads this record's register origins (a state token's arrival: 1) or follows its
  // definition chains (a cp.async copy size, a tcgen05.mma descriptor: 2); 0 otherwise. A member's
  // origins there (and along those chains) must be the representative's (query_log.cpp ContainMap).
  std::uint8_t chases = 0;
  // A founder's record whose lowering overwrote a register with a pending asynchronous write
  // (query::RegisterTouch). Removing it leaves its written registers pending in the member until
  // the next tcgen05.wait::ld (`clears`): later records may read them (lowering orders such reads
  // after the pending write); one that writes them is refused, or, a tcgen05.ld, marks the member
  // (query_log.cpp ContainMap; query_log.h UnitDelta::hazard).
  bool touched = false;
  // A founder's record whose lowering read a register with a pending asynchronous write
  // (query::RegisterTouch::reads): its events, and the thread's later ones, are ordered after that
  // write. Removing it leaves the registers it reads pending in the member in the same way.
  bool touched_reads = false;
  bool clears = false; // a tcgen05.wait::ld: no register write is pending after it
  // A tcgen05.ld: its register writes are asynchronous effects without a predecessor, pending until a
  // record reads or overwrites the register or a tcgen05.wait::ld completes them (lowering.cpp).
  bool async_load = false;
  // The record's register reads and writes (ids of `ContainmentProfile::register_ids`; kUnknownRegister
  // when a name is not one of them): [reads_begin, +reads_count) and [writes_begin, +writes_count) of
  // its path's `registers`.
  std::uint32_t reads_begin = 0, reads_count = 0, writes_begin = 0, writes_count = 0;
  friend bool operator==(const ContainmentRecord &, const ContainmentRecord &) = default;
};
inline constexpr std::uint32_t kUnknownRegister = UINT32_MAX;
// One CTA's records for containment admission: one record sequence per template path (the first
// lane of each template group; one per lane on the lane path), each lane's path, and `key`, the
// containment key: equal on every unit a containment admission can relate (lanes, warps, lane ids,
// per lane the path's kFixed records with predicates and metadata, and the rank-relative shared
// sites; the lane aggregates are accounted per unit). Empty (no lanes) for a CTA that reads
// runtime-written tensor maps.
// O(paths x records + lanes).
struct ContainmentProfile {
  std::vector<std::uint32_t> lane_path;
  std::vector<std::vector<ContainmentRecord>> paths;
  // Per path: the records' register origins, (instruction, occurrence) of each defining record.
  std::vector<std::vector<std::pair<std::uint32_t, std::uint32_t>>> origins;
  // Per path: the records' register reads and writes (register ids by name, the function's
  // instruction text order: equal on every CTA of the function).
  std::vector<std::vector<std::uint32_t>> registers;
  std::uint64_t key = 0;
};
[[nodiscard]] ContainmentProfile ComputeContainmentProfile(const CtaFacts &facts);
} // namespace sentryward::frontend::ptx
