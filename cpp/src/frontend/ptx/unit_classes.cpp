#include "sentryward/frontend/ptx/unit_classes.h"

#include "sentryward/frontend/ptx/unit_facts.h"

#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <type_traits>
#include <utility>

namespace sentryward::frontend::ptx {
// ---- the unit-independence precondition -------------------------------------
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

PreconditionReport PreconditionScan(const Function &function) {
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
      report.atomic_sites.push_back(static_cast<std::uint32_t>(index)); // global atomic: every scope, where executed
    } else if (op == "ld" || op == "st") {
      if (!mods.empty() && (mods[0] == "async" || mods[0] == "bulk")) continue;
      const auto space = SpaceOf(mods);
      if (space != "global" && space != "generic") continue;
      const auto [sem, scope] = SemScopeOf(mods);
      if (sem == "volatile" || sem == "mmio") violation(); // a system-scope strong access
      else if (In(sem, {"relaxed", "acquire", "release"}) && Wide(scope)) violation(); // strong, wider than the unit
    } else if (op == "multimem") {
      violation(); // a multimem access
    } else if (op == "fence") {
      const auto scope = SemScopeOf(mods).scope;
      if (!Wide(scope)) continue;
      // The proxy-fence exception: proxy fences only restrict base causality order; memory
      // fences (sc, acq_rel, acquire, release, mbarrier_init) stay rejected.
      if (mods.empty() || mods[0] != "proxy") violation(); // a memory fence wider than the unit
    } else if (op == "membar") {
      if (!mods.empty() && (mods[0] == "gl" || mods[0] == "sys")) violation(); // a memory fence wider than the unit
    } else if (op == "tensormap" && !mods.empty() && mods[0] == "cp_fenceproxy") {
      // A copy fused with fence.proxy.tensormap::generic.release.<scope>: an admitted proxy fence,
      // checked by the tensormap locality rule.
      ++report.tensormap_copies;
    } else if (op == "cp" && mods.size() >= 3 && mods[0] == "reduce" && mods[1] == "async" && mods[2] == "bulk") {
      for (std::size_t k = 3; k < mods.size(); ++k) {
        if (mods[k] == "global") {
          violation(); // a global bulk reduction
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
      if (region.object == map.allocation &&
          (region.begin >= map.offset ? region.begin - map.offset : map.offset - region.begin) < 128)
        return true; // two 128-byte regions overlap (written without overflow)
  }
  return false;
}

// ---- units and the plan -----------------------------------------------------
UnitPlan PlanUnits(const FrontendConfig &config, const Function &function,
                   const UnitGates &gates) {
  UnitPlan plan;
  const auto grid = LaunchDimensions(config).first;
  plan.grid = grid;
  plan.unit_shape = config.cluster_dim.value_or(Coordinates{1, 1, 1});
  plan.precondition = PreconditionScan(function);
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
  Coordinates unit_grid{1, 1, 1};
  for (int axis = 0; axis < 3; ++axis) {
    if (plan.unit_shape[axis] == 0 || grid[axis] % plan.unit_shape[axis]) exact = false;
    else unit_grid[axis] = grid[axis] / plan.unit_shape[axis];
  }
  const std::uint64_t units = exact ? std::uint64_t(unit_grid[0]) * unit_grid[1] * unit_grid[2] : 0;
  // The units of every multi-unit launch with whole-CTA analysis, including
  // the ones class mode keeps off.
  plan.active = exact && units > 1 && !gates.value_thread_range;
  // Class mode needs a cluster grid that tiles the launch, more than one unit, none of the run
  // options in UnitGates and the unit-independence precondition. Values are lifted through the
  // scalar fold; without it nothing lifts them.
  plan.class_mode = plan.active && !gates.trusted_reference && !gates.value_dump &&
                    !gates.branch_scope && !gates.scalar_fold_dump && !gates.exhaustive_search &&
                    !gates.continue_search && !gates.development_exports && plan.precondition.holds &&
                    !(gates.value_analysis && !gates.scalar_fold);
  // Class mode with value analysis implies the scalar fold.
  plan.lift_values = plan.class_mode && gates.value_analysis;
  if (!plan.active) return plan;
  plan.units.reserve(units);
  const auto &d = plan.unit_shape;
  for (std::uint32_t z = 0; z < unit_grid[2]; ++z)
    for (std::uint32_t y = 0; y < unit_grid[1]; ++y)
      for (std::uint32_t x = 0; x < unit_grid[0]; ++x) {
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

// ---- the structure key ------------------------------------------------------
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
  w.Put(tree);
  w.Put(facts.GroupPaths());
  const auto groups = facts.TemplateGroups();
  w.Put(groups);
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
  w.Put(facts.StaticCounts());
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

// ---- containment profiles ---------------------------------------------------
namespace {
std::uint64_t HashBytes(const std::string &bytes) noexcept {
  std::uint64_t h = 0x243f6a8885a308d3ULL;
  std::size_t i = 0;
  for (; i + 8 <= bytes.size(); i += 8) {
    std::uint64_t word;
    std::memcpy(&word, bytes.data() + i, 8);
    h = Mix(h ^ Mix(word));
  }
  std::uint64_t tail = 0;
  if (i < bytes.size()) std::memcpy(&tail, bytes.data() + i, bytes.size() - i);
  return Mix(h ^ Mix(tail ^ bytes.size()));
}
template <class T> bool Present(const T &x) {
  if constexpr (std::is_same_v<T, bool>) return x;
  else if constexpr (IsOptional<T>::value) return x.has_value();
  else if constexpr (requires { x.empty(); }) return !x.empty();
  else return true;
}
// The instruction's containment kind (before its record's metadata).
ContainmentKind KindOf(const Instruction &i) {
  static const std::initializer_list<std::string_view> compute = {
      "add", "sub", "mul", "mad", "fma", "div", "rem", "abs", "neg", "min", "max", "and", "or", "xor", "not",
      "cnot", "shl", "shr", "mov", "setp", "selp", "set", "cvt", "cvta", "prmt", "bfe", "bfi", "popc", "clz",
      "brev", "lop3", "shf", "ex2", "lg2", "rcp", "sqrt", "rsqrt", "sin", "cos", "tanh", "mul24", "mad24", "sad",
      "testp", "copysign", "slct", "bfind", "dp4a", "dp2a", "addc", "subc", "madc", "szext", "bmsk"};
  if (i.opcode == "bra") return i.indirect_targets.empty() ? ContainmentKind::kBranch : ContainmentKind::kFixed;
  if (In(i.opcode, compute)) return ContainmentKind::kCompute;
  if (i.opcode != "ld" && i.opcode != "st") return ContainmentKind::kFixed;
  bool global = false;
  for (const auto &m : i.modifiers) {
    if (m == "global") global = true;
    else if (m == "async" || m == "bulk" || m == "volatile" || m == "mmio" || m == "relaxed" || m == "acquire" ||
             m == "release" || m == "weak" || m == "cta" || m == "cluster" || m == "gpu" || m == "sys" ||
             m.starts_with("shared") || m == "local" || m == "param" || m == "const" || m == "tensormap")
      return ContainmentKind::kFixed;
  }
  return global ? ContainmentKind::kAccess : ContainmentKind::kFixed;
}
} // namespace

ContainmentProfile ComputeContainmentProfile(const CtaFacts &facts) {
  ContainmentProfile profile;
  // Runtime-written tensor maps are read through frontend dependencies on their global copies: a
  // CTA with such reads gets no profile (no containment admission).
  if (const auto &reads = facts.TensorMapReads(); reads && !reads->empty()) return profile;
  const auto &instructions = facts.function().instructions;
  std::vector<ContainmentKind> kinds(instructions.size());
  for (std::size_t k = 0; k < instructions.size(); ++k) kinds[k] = KindOf(instructions[k]);
  // Where lowering reads origins (lowering.cpp: MBarrier's state-token wait, Copy's cp.async source
  // size, Mma's descriptors through Resources::FixedBits); conservatively every such instruction.
  std::vector<std::uint8_t> chases(instructions.size());
  for (std::size_t k = 0; k < instructions.size(); ++k) {
    const auto &i = instructions[k];
    const auto has = [&](std::string_view m) { return std::find(i.modifiers.begin(), i.modifiers.end(), m) != i.modifiers.end(); };
    if (i.opcode == "mbarrier" && (has("test_wait") || has("try_wait"))) chases[k] = 1;
    if ((i.opcode == "cp" && has("async") && !has("bulk")) || (i.opcode == "tcgen05" && has("mma"))) chases[k] = 2;
  }
  // Register ids by name (the executor's register names are the instruction text's), in the
  // function's instruction order.
  std::unordered_map<std::string_view, std::uint32_t> register_ids;
  const auto intern = [&](std::string_view name) { register_ids.try_emplace(name, register_ids.size()); };
  for (const auto &i : instructions) {
    for (const auto &r : i.registers) intern(r);
    for (const auto &r : i.destinations) intern(r);
    if (i.predicate) intern(*i.predicate);
  }
  const auto id_of = [&](std::string_view name) {
    const auto found = register_ids.find(name);
    return found == register_ids.end() ? kUnknownRegister : found->second;
  };
  // Per instruction: its register reads (every operand register but its destinations) and writes.
  std::vector<std::vector<std::uint32_t>> instruction_reads(instructions.size()), instruction_writes(instructions.size());
  std::vector<bool> clears(instructions.size());
  for (std::size_t k = 0; k < instructions.size(); ++k) {
    const auto &i = instructions[k];
    for (const auto &r : i.destinations) instruction_writes[k].push_back(id_of(r));
    for (const auto &r : i.registers)
      if (std::find(i.destinations.begin(), i.destinations.end(), r) == i.destinations.end())
        instruction_reads[k].push_back(id_of(r));
    clears[k] = i.opcode == "tcgen05" && i.semantics.operation == "wait_load";
  }
  std::string scratch;
  const auto sequence = [&](LaneRef lane) {
    std::vector<ContainmentRecord> records(facts.Size(lane));
    auto &origins = profile.origins.emplace_back();
    auto &registers = profile.registers.emplace_back();
    for (std::size_t pos = 0; pos < records.size(); ++pos) {
      const auto r = facts.Meta({lane, pos});
      auto &out = records[pos];
      out.instruction = static_cast<std::uint32_t>(r.instruction_id());
      out.occurrence = static_cast<std::uint32_t>(r.occurrence());
      out.predicate = !r.predicate_value() ? 0 : *r.predicate_value() ? 1 : 2;
      out.kind = out.instruction < kinds.size() ? kinds[out.instruction] : ContainmentKind::kFixed;
      out.chases = out.instruction < chases.size() ? chases[out.instruction] : 2;
      out.origin_begin = static_cast<std::uint32_t>(origins.size());
      for (const auto &[reg, key] : r.register_origins())
        origins.emplace_back(key.instruction, static_cast<std::uint32_t>(key.occurrence));
      out.origin_count = static_cast<std::uint32_t>(origins.size()) - out.origin_begin;
      if (out.instruction < instructions.size()) {
        out.clears = clears[out.instruction];
        out.async_load = instructions[out.instruction].opcode == "tcgen05" &&
                         instructions[out.instruction].semantics.operation == "load";
        const bool opaque = !r.opaque_compute_diamond_indices().empty();
        out.reads_begin = static_cast<std::uint32_t>(registers.size());
        if (opaque)
          for (const auto &name : r.opaque_compute_register_reads()) registers.push_back(id_of(name));
        else registers.insert(registers.end(), instruction_reads[out.instruction].begin(),
                              instruction_reads[out.instruction].end());
        out.reads_count = static_cast<std::uint32_t>(registers.size()) - out.reads_begin;
        out.writes_begin = static_cast<std::uint32_t>(registers.size());
        if (opaque)
          for (const auto &name : r.opaque_compute_register_writes()) registers.push_back(id_of(name));
        else registers.insert(registers.end(), instruction_writes[out.instruction].begin(),
                              instruction_writes[out.instruction].end());
        out.writes_count = static_cast<std::uint32_t>(registers.size()) - out.writes_begin;
      }
      scratch.clear();
      KeyWriter w(scratch);
      w.Put(r.branch_scope_decision()); w.Put(r.collapsed_wait_loop()); w.Put(r.normalized_wait_result());
      w.Put(r.normalized_clc_result()); w.Put(r.canonical_wait_probe()); w.Put(r.wait_exit_label());
      w.Put(r.schedule_seed_outcome()); w.Put(r.probe_fallback_key()); w.Put(r.opaque_compute_diamond_indices());
      w.Put(r.collapsed_wait_indices()); w.Put(r.opaque_compute_join_label()); w.Put(r.assumed_equality());
      w.Put(r.opaque_compute_register_reads()); w.Put(r.opaque_compute_register_writes());
      w.Put(r.merged_arm_join());
      out.meta = HashBytes(scratch);
      scratch.clear();
      w.Put(r.register_origins());
      out.origins = HashBytes(scratch);
      if (Present(r.branch_scope_decision()) || Present(r.collapsed_wait_loop()) ||
          Present(r.normalized_wait_result()) || Present(r.normalized_clc_result()) ||
          Present(r.canonical_wait_probe()) || Present(r.wait_exit_label()) || Present(r.schedule_seed_outcome()) ||
          Present(r.probe_fallback_key()) || Present(r.collapsed_wait_indices()) || Present(r.assumed_equality()) ||
          Present(r.merged_arm_join()))
        out.kind = ContainmentKind::kFixed;
      else if (Present(r.opaque_compute_diamond_indices()) || Present(r.opaque_compute_join_label()) ||
               Present(r.opaque_compute_register_reads()) || Present(r.opaque_compute_register_writes()))
        out.kind = out.kind == ContainmentKind::kBranch ? ContainmentKind::kOpaque : ContainmentKind::kFixed;
    }
    return records;
  };
  const auto lanes = facts.LaneCount();
  profile.lane_path.assign(lanes, UINT32_MAX);
  if (facts.TemplateExecution()) {
    const auto groups = facts.TemplateGroups();
    for (const auto &group : groups) {
      if (group.empty()) continue;
      const auto path = static_cast<std::uint32_t>(profile.paths.size());
      profile.paths.push_back(sequence(facts.Lane(group.front())));
      for (const auto lane : group) profile.lane_path.at(lane) = path;
    }
  }
  for (std::size_t lane = 0; lane < lanes; ++lane)
    if (profile.lane_path[lane] == UINT32_MAX) {
      profile.lane_path[lane] = static_cast<std::uint32_t>(profile.paths.size());
      profile.paths.push_back(sequence(facts.Lane(lane)));
    }
  // The key: per lane its warp, lane id and its path's kFixed records.
  std::vector<std::uint64_t> fixed(profile.paths.size());
  for (std::size_t p = 0; p < profile.paths.size(); ++p) {
    scratch.clear();
    KeyWriter w(scratch);
    for (const auto &r : profile.paths[p])
      if (r.kind == ContainmentKind::kFixed) {
        w.U64(r.instruction); w.U64(r.occurrence); w.U64(r.predicate); w.U64(r.meta);
      }
    fixed[p] = HashBytes(scratch);
  }
  std::string bytes;
  KeyWriter w(bytes);
  w.Put(facts.TemplateExecution());
  w.U64(lanes);
  for (std::size_t index = 0; index < lanes; ++index) {
    const auto lane = facts.Lane(index);
    w.U64(facts.Warp(lane));
    w.U64(facts.LaneId(lane));
    w.U64(fixed[profile.lane_path[index]]);
  }
  const auto sites = facts.SharedSitesRelative();
  w.U64(sites.size());
  for (const auto &site : sites) {
    w.Put(site.owner ? std::optional<std::uint32_t>(site.owner->value) : std::nullopt);
    w.U64(site.thread);
    w.Put(site.instruction);
  }
  profile.key = HashBytes(bytes);
  return profile;
}
} // namespace sentryward::frontend::ptx
