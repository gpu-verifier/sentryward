#include "sentryward/frontend/ptx/unit_classes.h"

#include "sentryward/frontend/ptx/unit_facts.h"

#include <array>
#include <cstring>
#include <type_traits>
#include <utility>

namespace sentryward::frontend::ptx {
std::optional<UnitClassMode> ParseUnitClassMode(std::string_view text) noexcept {
  if (text == "off") return UnitClassMode::Off;
  if (text == "on") return UnitClassMode::On;
  return std::nullopt;
}

// ---- precondition (I) -------------------------------------------------------
namespace {
bool In(std::string_view value, std::initializer_list<std::string_view> set) noexcept {
  for (const auto item : set)
    if (value == item) return true;
  return false;
}
// Helpers of the per-site scan rules (state space, semantics and scope).
std::string_view SpaceOf(const std::vector<std::string> &mods) noexcept {
  for (const auto &p : mods) {
    if (p == "global") return "global";
    if (p.starts_with("shared")) return "shared";
    if (p == "local" || p == "const" || p.starts_with("param")) return "other";
  }
  return "generic";
}
struct SemScope {
  std::string_view sem, scope;
};
SemScope SemScopeOf(const std::vector<std::string> &mods) noexcept {
  SemScope result;
  for (const auto &p : mods) {
    if (result.sem.empty() &&
        In(p, {"relaxed", "acquire", "release", "acq_rel", "sc", "weak", "volatile", "mmio"}))
      result.sem = p;
    if (In(p, {"cta", "cluster", "gpu", "sys"})) result.scope = p;
  }
  return result;
}
bool Wide(std::string_view scope) noexcept { return scope == "gpu" || scope == "sys"; }
} // namespace

PreconditionReport PreconditionScan(const Module &module, const Function &function) {
  (void)module;
  PreconditionReport report;
  const auto violation = [&] { report.holds = false; };
  for (std::size_t index = 0; index < function.instructions.size(); ++index) {
    const auto &i = function.instructions[index];
    const auto &op = i.opcode;
    const auto &mods = i.modifiers;
    if (op == "atom" || op == "red") {
      if (!mods.empty() && mods[0] == "async") continue; // red.async: shared::cluster only
      const auto space = SpaceOf(mods);
      if (space != "global" && space != "generic") continue;
      report.atomic_sites.push_back(static_cast<std::uint32_t>(index)); // R-atom: every scope, where executed
    } else if (op == "ld" || op == "st") {
      if (!mods.empty() && (mods[0] == "async" || mods[0] == "bulk")) continue;
      const auto space = SpaceOf(mods);
      if (space != "global" && space != "generic") continue;
      const auto [sem, scope] = SemScopeOf(mods);
      if (sem == "volatile" || sem == "mmio") violation(); // R-strong
      else if (In(sem, {"relaxed", "acquire", "release"}) && Wide(scope)) violation(); // R-strong
    } else if (op == "multimem") {
      violation(); // R-multimem
    } else if (op == "fence") {
      const auto [sem, scope] = SemScopeOf(mods);
      (void)sem;
      if (!Wide(scope)) continue;
      // (I.3'): proxy fences only restrict base causality order; memory
      // fences (sc, acq_rel, acquire, release, mbarrier_init) stay rejected.
      if (mods.empty() || mods[0] != "proxy") violation(); // R-fence
    } else if (op == "membar") {
      if (!mods.empty() && (mods[0] == "gl" || mods[0] == "sys")) violation(); // R-fence
    } else if (op == "tensormap" && !mods.empty() && mods[0] == "cp_fenceproxy") {
      // A copy fused with fence.proxy.tensormap::generic.release.<scope>: a proxy fence (I.3'), checked by (I.5).
      ++report.tensormap_copies;
    } else if (op == "cp" && mods.size() >= 3 && mods[0] == "reduce" && mods[1] == "async" && mods[2] == "bulk") {
      for (std::size_t k = 3; k < mods.size(); ++k) {
        if (mods[k] == "global") {
          violation(); // R-reduce
          break;
        }
        if (mods[k].starts_with("shared")) break;
      }
    }
  }
  return report;
}

bool TensormapInputOverlap(const std::vector<TensormapRegion> &regions,
                           const std::vector<UnitPlan::InputMap> &maps) noexcept {
  if (regions.empty()) return false;
  for (const auto &map : maps) {
    if (map.allocation.empty()) return true;
    for (const auto &region : regions)
      if (region.object == map.allocation && region.begin < map.offset + 128 && map.offset < region.begin + 128)
        return true;
  }
  return false;
}

// ---- units and the plan -----------------------------------------------------
std::uint32_t UnitPlan::UnitOf(Coordinates cta) const noexcept {
  const Coordinates k{cta[0] / unit_shape[0], cta[1] / unit_shape[1], cta[2] / unit_shape[2]};
  return k[0] + unit_grid[0] * (k[1] + unit_grid[1] * k[2]);
}
std::uint32_t UnitPlan::RankOf(Coordinates cta) const noexcept {
  const Coordinates l{cta[0] % unit_shape[0], cta[1] % unit_shape[1], cta[2] % unit_shape[2]};
  return l[0] + unit_shape[0] * (l[1] + unit_shape[1] * l[2]);
}

UnitPlan PlanUnits(UnitClassMode mode, const FrontendConfig &config, const Module &module,
                   const Function &function, const UnitGates &gates) {
  UnitPlan plan;
  const auto [grid, block] = LaunchDimensions(config);
  (void)block;
  plan.grid = grid;
  plan.unit_shape = config.cluster_dim.value_or(Coordinates{1, 1, 1});
  plan.precondition = PreconditionScan(module, function);
  // An allocation_location names a launch resource; the slot resolver binds it to the resource's
  // concrete allocation object at its base offset (slot_resolver.cpp MemoryResource), or to the
  // resource id itself for a pointer without one. An unknown resource cannot be placed (empty name).
  using Json = nlohmann::json;
  const auto bound = [&](const std::string &id, std::uint64_t offset) -> UnitPlan::InputMap {
    for (const auto &r : config.resources) {
      if (!r.is_object() || r.value("id", std::string()) != id) continue;
      const auto allocation = r.value("allocation", Json::object());
      if (allocation.empty()) return {id, offset};
      if (allocation.value("resolution", "") != "concrete" || !allocation.contains("object") ||
          !allocation.contains("base_byte_offset"))
        return {};
      const auto base = JsonInteger(allocation["base_byte_offset"]);
      if (base < 0 || base > UINT64_MAX - offset) return {};
      return {allocation["object"].get<std::string>(), base.convert_to<std::uint64_t>() + offset};
    }
    return {};
  };
  for (auto it = config.tma_descriptors.begin(); it != config.tma_descriptors.end(); ++it) {
    const auto &d = it.value();
    if (!d.is_object()) continue;
    if (const auto location = d.find("allocation_location");
        location != d.end() && location->is_array() && location->size() == 2 && (*location)[0].is_string()) {
      const auto offset = JsonInteger((*location)[1]);
      if (offset < 0 || offset > UINT64_MAX) plan.input_maps.push_back({});
      else plan.input_maps.push_back(bound((*location)[0].get<std::string>(), offset.convert_to<std::uint64_t>()));
    } else if (const auto address = d.find("address"); address != d.end()) {
      const auto value = JsonInteger(*address);
      plan.input_maps.push_back({{}, value >= 0 && value <= UINT64_MAX ? value.convert_to<std::uint64_t>() : 0});
    }
  }
  bool exact = true;
  for (int axis = 0; axis < 3; ++axis) {
    if (plan.unit_shape[axis] == 0 || grid[axis] % plan.unit_shape[axis]) exact = false;
    else plan.unit_grid[axis] = grid[axis] / plan.unit_shape[axis];
  }
  const std::uint64_t units = exact ? std::uint64_t(plan.unit_grid[0]) * plan.unit_grid[1] * plan.unit_grid[2] : 0;
  // Class mode needs mode on (off: SENTRYWARD_UNIT_CLASSES=off, run.cpp), a
  // cluster grid that tiles the launch, more than one unit, none of the run
  // options in UnitGates and precondition (I). Values are lifted through the
  // scalar fold; without it nothing lifts them.
  const bool allowed = mode == UnitClassMode::On && exact && units > 1 && !gates.value_thread_range &&
                       !gates.trusted_reference && !gates.value_dump && !gates.branch_scope &&
                       !gates.scalar_fold_dump && !gates.exhaustive_search && !gates.continue_search &&
                       !gates.development_exports && !gates.reference_stdin && plan.precondition.holds &&
                       !(gates.value_analysis && !gates.scalar_fold);
  plan.lift_values = gates.value_analysis && gates.scalar_fold && allowed;
  // The units of every multi-unit launch with whole-CTA analysis, including
  // the ones class mode keeps off.
  plan.active = exact && units > 1 && !gates.value_thread_range;
  plan.class_mode = allowed;
  if (!plan.active) return plan;
  plan.units.reserve(units);
  const auto &d = plan.unit_shape;
  for (std::uint32_t z = 0; z < plan.unit_grid[2]; ++z)
    for (std::uint32_t y = 0; y < plan.unit_grid[1]; ++y)
      for (std::uint32_t x = 0; x < plan.unit_grid[0]; ++x) {
        LaunchUnit unit;
        unit.index = static_cast<std::uint32_t>(plan.units.size());
        unit.kappa = {x, y, z};
        unit.ctas.reserve(plan.UnitSize());
        for (std::uint32_t lz = 0; lz < d[2]; ++lz)
          for (std::uint32_t ly = 0; ly < d[1]; ++ly)
            for (std::uint32_t lx = 0; lx < d[0]; ++lx)
              unit.ctas.push_back({x * d[0] + lx, y * d[1] + ly, z * d[2] + lz});
        plan.units.push_back(std::move(unit));
      }
  return plan;
}

// ---- C1: the structure key --------------------------------------------------
namespace {
constexpr std::uint64_t Mix(std::uint64_t z) noexcept {
  z += 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
template <class T> struct IsOptional : std::false_type {};
template <class T> struct IsOptional<std::optional<T>> : std::true_type {};
// Appends a canonical byte encoding (fixed-width integers, length-prefixed
// strings and sequences). Two keys are equal iff their bytes are equal.
class KeyWriter {
public:
  explicit KeyWriter(std::string &bytes) noexcept : bytes_(bytes) {}
  void U64(std::uint64_t value) {
    char raw[8];
    std::memcpy(raw, &value, 8);
    bytes_.append(raw, 8);
  }
  void Str(std::string_view text) {
    U64(text.size());
    bytes_.append(text);
  }
  template <class T> void Put(const T &x) {
    if constexpr (std::is_same_v<T, bool>) U64(x ? 1 : 2);
    else if constexpr (std::is_integral_v<T> || std::is_enum_v<T>) U64(static_cast<std::uint64_t>(x));
    else if constexpr (std::is_same_v<T, std::string> || std::is_same_v<T, std::string_view>) Str(x);
    else if constexpr (std::is_same_v<T, Integer>) Str(x.str());
    else if constexpr (std::is_same_v<T, DynamicKey>) {
      U64(x.instruction);
      U64(x.occurrence);
    } else if constexpr (std::is_same_v<T, BranchScopeDecision>) {
      Put(x.taken);
      U64(x.visit);
      U64(x.user_decision);
    } else if constexpr (std::is_same_v<T, PathId>) U64(x.value);
    else if constexpr (std::is_same_v<T, TemplatePathNode>) {
      Put(x.parent);
      U64(x.prefix);
      U64(x.pc);
      Put(x.reason);
      Put(x.group);
    } else if constexpr (std::is_same_v<T, DataBranchDecision>) {
      Str(x.assumption);
      Str(x.rule);
      U64(x.taken);
      U64(x.not_taken);
    } else if constexpr (IsOptional<T>::value) {
      if (x) {
        U64(1);
        Put(*x);
      } else U64(0);
    } else if constexpr (requires { x.first; x.second; }) {
      Put(x.first);
      Put(x.second);
    } else {
      U64(x.size());
      for (const auto &element : x) Put(element);
    }
  }
  // Every ExecutionMetadata field of one record.
  void Record(const RecordView &r) {
    Put(r.instruction_id()); Put(r.occurrence()); Put(r.predicate_value()); Put(r.branch_scope_decision());
    Put(r.collapsed_wait_loop()); Put(r.normalized_wait_result()); Put(r.normalized_clc_result());
    Put(r.canonical_wait_probe()); Put(r.wait_exit_label()); Put(r.schedule_seed_outcome());
    Put(r.probe_fallback_key()); Put(r.opaque_compute_diamond_indices()); Put(r.collapsed_wait_indices());
    Put(r.opaque_compute_join_label()); Put(r.assumed_equality()); Put(r.opaque_compute_register_reads());
    Put(r.opaque_compute_register_writes()); Put(r.register_origins()); Put(r.merged_arm_join());
  }

private:
  std::string &bytes_;
};
} // namespace

StructureKey ComputeStructureKey(const CtaFacts &facts) {
  StructureKey key;
  KeyWriter w(key.bytes);
  const bool templates = facts.TemplateExecution();
  w.Put(templates);
  w.U64(facts.LaneCount());
  for (std::size_t index = 0; index < facts.LaneCount(); ++index) {
    const auto lane = facts.Lane(index);
    w.U64(facts.Warp(lane));
    w.U64(facts.LaneId(lane));
    w.Put(facts.TemplatePath(lane));
    w.U64(facts.Size(lane));
  }
  const auto tree = facts.TemplatePaths();
  w.U64(tree.size());
  for (const auto &node : tree) w.Put(node);
  w.U64(facts.GroupPaths().size());
  for (const auto path : facts.GroupPaths()) w.U64(path.value);
  const auto groups = facts.TemplateGroups();
  w.U64(groups.size());
  for (const auto &group : groups) w.Put(group);
  // One metadata sequence per path (a member of its finished group), or per lane on the lane path.
  std::vector<RegId> bound;
  const auto sequence = [&](LaneRef lane, bool prototype) {
    const auto size = facts.Size(lane);
    w.U64(size);
    for (std::size_t pos = 0; pos < size; ++pos) {
      w.Record(facts.Meta({lane, pos}));
      if (prototype) {
        facts.PrototypeRegisterIds({lane, pos}, bound);
        w.Put(bound);
      }
    }
  };
  if (templates) {
    for (const auto &node : tree)
      if (node.group && *node.group < groups.size() && !groups[*node.group].empty()) {
        w.U64(1);
        sequence(facts.Lane(groups[*node.group].front()), true);
      } else w.U64(0);
  } else
    for (std::size_t index = 0; index < facts.LaneCount(); ++index) sequence(facts.Lane(index), false);
  for (std::size_t index = 0; index < facts.LaneCount(); ++index) {
    const auto aggregates = facts.Aggregates(facts.Lane(index));
    w.U64(aggregates.assumed_assertions);
    w.Put(aggregates.opaque_regions);
    w.Put(aggregates.data_branch_decisions);
  }
  const auto counts = facts.StaticCounts();
  w.U64(counts.size());
  for (const auto count : counts) w.U64(count);
  const auto sites = facts.SharedSitesRelative();
  w.U64(sites.size());
  for (const auto &site : sites) {
    w.Put(site.owner ? std::optional<std::uint32_t>(site.owner->value) : std::nullopt);
    w.U64(site.thread);
    w.Put(site.instruction);
  }
  std::uint64_t h = 0x243f6a8885a308d3ULL;
  std::size_t i = 0;
  for (; i + 8 <= key.bytes.size(); i += 8) {
    std::uint64_t word;
    std::memcpy(&word, key.bytes.data() + i, 8);
    h = Mix(h ^ Mix(word));
  }
  std::uint64_t tail = 0;
  if (i < key.bytes.size()) std::memcpy(&tail, key.bytes.data() + i, key.bytes.size() - i);
  key.hash = Mix(h ^ Mix(tail ^ key.bytes.size()));
  return key;
}
} // namespace sentryward::frontend::ptx
