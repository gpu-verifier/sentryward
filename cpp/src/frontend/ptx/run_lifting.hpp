#pragma once
#include "run_support.hpp"

namespace {
// The value stage's verdict on same-value write obligations (MemoryRaceResult::
// same_value_write_groups, docs/assumptions.md): nullopt when memory elimination
// processed every obligation write and proved its bytes equal; the unproven write
// when a comparison failed; otherwise undecided, with the reason the value stage
// stopped. A write's checked access is that of the first run of `writes` holding
// it: O(runs).
std::optional<alg::SameValueFailure> SameValueOutcome(
    const std::optional<sentryward::value_analysis::MemoryEliminationResult> &eliminated,
    const Json &value_analysis, const std::vector<ptx::SameValueWriteRun> &writes) {
  if (eliminated && eliminated->unproven_same_value_write) {
    const auto &write = *eliminated->unproven_same_value_write;
    std::optional<alg::AccessId> access;
    for (const auto &run : writes)
      if (run.Holds(write)) {
        access = run.access;
        break;
      }
    if (access)
      return alg::SameValueFailure{"same_value_writes_unproven",
          "Unordered writes cover the same bytes, and the value analysis could not prove that they "
          "store equal values.", access};
  }
  if (eliminated && eliminated->pending_same_value_writes == 0) return std::nullopt;
  const auto reason = value_analysis.value("reason", std::string("value_analysis_incomplete"));
  return alg::SameValueFailure{"same_value_writes_unchecked",
      "Unordered writes cover the same bytes and are not races only if they store equal values; "
      "the value analysis stopped before proving this (" + reason + ").", std::nullopt};
}

// The fixed shape-array tables (fixed_input.global_memory_inputs) as TableValuation reads them:
// per table its allocation object, element width and raw element bits. "[]" without tables.
// O(table elements).
std::string TableAtomsJson(const ptx::FrontendConfig &config) {
  if (!config.fixed_input.contains("global_memory_inputs")) return "[]";
  Json tables = Json::array();
  for (const auto &entry : config.fixed_input["global_memory_inputs"]) {
    const auto resource = entry.at("resource").get<std::string>();
    std::string object = resource;
    for (const auto &r : config.resources)
      if (r.value("id", "") == resource && r.contains("allocation"))
        object = r["allocation"].value("object", resource);
    const auto bits = entry.at("element_bits").get<unsigned>();
    Json values = Json::array();
    for (const auto &value : entry.at("values"))
      values.push_back((ptx::JsonInteger(value) & ((ptx::Integer(1) << bits) - 1)).convert_to<std::uint64_t>());
    tables.push_back({{"object", object}, {"element_bytes", bits / 8}, {"values", std::move(values)}});
  }
  return tables.dump();
}
// Cross-CTA value lifting in class mode: the class tables the scalar fold
// instantiates members from. A unit is named by its origin (rank 0's
// coordinates); δ is keyed as ConstantRegion::allocation (AllocationIdentity);
// CTA ids are linear, renamed rank by rank. An allocation with a piecewise δ has no entry: such a
// quotient exists only when no value stage lifts (UnitClasses::AdmitPiecewise).
// O(units x unit size + δ entries).
sentryward::value_analysis::UnitLifting ClassLifting(const alg::UnitQuotient &q) {
  namespace va = sentryward::value_analysis;
  va::UnitLifting lifting;
  for (int axis = 0; axis < 3; ++axis) {
    lifting.grid[axis] = q.grid[axis];
    lifting.unit_shape[axis] = q.unit_shape[axis];
  }
  const auto origin = [&](std::uint32_t unit) {
    const auto c = q.CtaCoordinates(q.units.at(unit).ctas.at(0));
    return va::UnitLifting::Origin{c[0], c[1], c[2]};
  };
  for (const auto &c : q.classes) {
    if (c.units.size() < 2) continue;
    const auto &rep = q.units.at(c.units.front());
    auto &members = lifting.members[origin(c.units.front())];
    for (std::size_t i = 1; i < c.units.size(); ++i) {
      va::UnitLifting::Member member;
      member.origin = origin(c.units[i]);
      for (const auto &[object, d] : c.delta.at(i))
        member.delta.emplace(sentryward::value_analysis::AllocationIdentity("global", object), d);
      // The valued δ (the member's exact offsets) where table atoms made it differ from the forms' δ.
      if (i < c.valued_delta.size()) {
        if (!c.valued_delta[i]) {
          // A member with a piecewise δ: its class is refused lifting as piecewise_unit_delta.
          if (c.pieces.empty()) lifting.valued_unliftable = true;
        } else
          for (const auto &[object, d] : *c.valued_delta[i])
            member.delta[sentryward::value_analysis::AllocationIdentity("global", object)] = d;
      }
      const auto &unit = q.units.at(c.units[i]);
      for (std::size_t rank = 0; rank < unit.ctas.size(); ++rank)
        member.ctas.emplace(rep.ctas.at(rank), unit.ctas[rank]);
      members.push_back(std::move(member));
    }
  }
  return lifting;
}
// After value analysis: every member's δ of an allocation where a row map was
// anchored must be a multiple of each audited row width R of that allocation,
// i.e. of their lcm (no fixed floor; admission checks no modulus). The audited
// row widths are the only moduli value analysis applies to absolute global
// offsets. Returns whether some allocation violates it. O(row-map allocations x members x widths).
bool RowModulusViolation(const sentryward::value_analysis::UnitLifting &lifting) {
  for (const auto &[allocation, rows] : sentryward::value_analysis::UnitRowModuli())
    for (const auto &[rep, members] : lifting.members)
      for (const auto &member : members) {
        const auto found = member.delta.find(allocation);
        if (found == member.delta.end() || !found->second) continue;
        for (const auto row : rows) {
          if (!row || row > (std::uint64_t(1) << 40)) return true;
          if (found->second % static_cast<std::int64_t>(row)) return true;
        }
      }
  return false;
}
// The first-item rule: a class-mode quotient's fold failure is the one the
// per-unit fold would report first when the lowering made no lifting-specific
// choice, failed while lowering output records (phase 'e') at the quotient's
// first item with no output family and no completed output, with a
// context-free code and a literal message, and no member's record precedes
// that item's first record (for every class with records in its allocation
// X0, minoff(X0) + δ_u(X0) > o0 for each member u). Items never span
// CTAs and the fold handles them in record order from an empty state, so the
// per-unit fold then fails at the same item of the same CTA in the same way.
// Reads only quotient-side objects (the fold result, the quotient's records
// and value graph) and the class tables, never member-expanded evidence.
// Returns whether the rule holds.
// O(quotient records + members).
bool FirstFailureMatchesPerUnit(const sentryward::value_ir::Graph &values,
                                const sentryward::value_analysis::MemoryEliminationResult &records,
                                const sentryward::value_analysis::ScalarFoldResult &folded,
                                const sentryward::value_analysis::UnitLifting &lifting) {
  namespace va = sentryward::value_analysis;
  const auto &stop = folded.failure_site;
  if (!folded.diagnostic) return false;
  if (folded.lifting.parameters || folded.lifting.declined) return false; // a lifting-specific choice
  if (stop.phase != 'e') return false;                                     // not the output-record phase
  if (stop.output_families) return false;
  if (stop.items != 1 || !stop.record || *stop.record >= records.outputs.size()) return false;
  if (stop.completed_outputs) return false;
  if (!va::FoldFailureContextFree(folded.diagnostic->code)) return false; // a context-dependent code
  if (!stop.literal_message) return false;
  const auto unit_of = [&](const va::OutputWrite &output) -> std::optional<va::UnitLifting::Origin> {
    const auto cta = values.node(output.writer.node).instanceOrigin(output.writer.instance).cta_coordinates;
    if (!cta) return std::nullopt;
    va::UnitLifting::Origin origin{};
    for (unsigned axis = 0; axis < 3; ++axis) {
      const auto shape = lifting.unit_shape[axis];
      origin[axis] = shape ? (*cta)[axis] - (*cta)[axis] % shape : (*cta)[axis];
    }
    return origin;
  };
  const auto &first = records.outputs[*stop.record];
  const auto &x0 = first.region.allocation;
  const auto o0 = first.region.offset;
  std::map<va::UnitLifting::Origin, std::uint64_t> minimum; // per representative unit, records in X0
  for (const auto &output : records.outputs) {
    if (output.region.allocation != x0) continue;
    const auto unit = unit_of(output);
    if (!unit) return false; // member precedence needs every record's unit
    auto [at, fresh] = minimum.try_emplace(*unit, output.region.offset);
    if (!fresh) at->second = std::min(at->second, output.region.offset);
  }
  for (const auto &[rep, members] : lifting.members) {
    const auto found = minimum.find(rep);
    if (found == minimum.end()) continue;
    for (const auto &member : members) {
      const auto delta = member.delta.find(x0);
      if (delta == member.delta.end()) return false;
      if (static_cast<__int128>(found->second) + delta->second <= static_cast<__int128>(o0)) return false;
    }
  }
  return true;
}
// Class-mode binding diagnostics: every per-occurrence entry of the
// quotient's list stands for each CTA of its class at the same rank. The list
// is expanded in launch order, which is the per-unit run's order when the entries were
// produced CTA by CTA (the linker visits occurrences in (cta, ...) order) or
// are all equal (code and message carry no CTA). incomplete_value_semantics
// summarizes the launch's first value diagnostic once, in the per-unit run too: not
// weighted. So does the domain link's stop at its first failing group as the
// list's only entry: the per-unit link visits groups in CTA order, the least
// CTA with a failing group founds its class, and no earlier group raised an
// entry. Any other entry without a CTA, or an order that cannot be expanded, is
// not expandable (per-unit run). In CTA order, the list's first entry is the
// per-unit run's first: the least CTA with entries founds its class
// (launch_order). O(entries + CTAs).
struct BindingExpansion {
  bool expandable = true;
  bool launch_order = false;
  std::uint64_t count = 0;
};
BindingExpansion BindingEvidence(const ptx::SemanticProgram &program, const alg::UnitQuotient &q,
                                 Json *list = nullptr) {
  BindingExpansion result;
  const auto &diagnostics = program.diagnostics;
  std::map<std::uint32_t, std::vector<std::size_t>> by_cta;
  std::vector<std::size_t> plain;
  bool ordered = true, uniform = true, stop = false;
  std::optional<std::uint32_t> last;
  const ptx::SemanticDiagnostic *first = nullptr;
  for (std::size_t i = 0; i < diagnostics.size(); ++i) {
    const auto &d = diagnostics[i];
    if (d.code == "incomplete_value_semantics" || (d.first_group && diagnostics.size() == 1)) {
      stop = d.first_group;
      plain.push_back(i);
      continue;
    }
    if (!d.occurrence) {
      result.expandable = false;
      return result;
    }
    const auto cta = d.occurrence->cta;
    if (last && cta < *last) ordered = false;
    last = cta;
    if (!first) first = &d;
    else if (d.code != first->code || d.message != first->message) uniform = false;
    by_cta[cta].push_back(i);
  }
  if ((!plain.empty() && !by_cta.empty()) || (!ordered && !uniform)) {
    result.expandable = false;
    return result;
  }
  result.launch_order = ordered && (plain.empty() || stop);
  result.count = plain.size();
  for (const auto &[cta, entries] : by_cta) result.count += q.Weight(cta) * entries.size();
  if (list) {
    for (const auto i : plain) list->push_back({{"code", diagnostics[i].code}, {"message", diagnostics[i].message}});
    if (!by_cta.empty()) {
      const std::uint64_t ctas = std::uint64_t(q.grid[0]) * q.grid[1] * q.grid[2];
      for (std::uint64_t linear = 0; linear < ctas; ++linear) {
        const auto found = by_cta.find(q.RepresentativeCta(static_cast<std::uint32_t>(linear)));
        if (found == by_cta.end()) continue;
        for (const auto i : found->second)
          list->push_back({{"code", diagnostics[i].code}, {"message", diagnostics[i].message}});
      }
    }
  }
  return result;
}
// A per-unit restart caused only by value lifting (UnitClassRestart "value_lifting") keeps the
// class-mode synchronization verdict, and the re-run redoes only the value side. Class mode
// reports exactly the per-unit run's synchronization result, and its check
// (well-synchronization, lifecycle, the race search with the cross-unit pass) was complete and race
// free before any value stage ran. Lifting adds no admission condition (UnitClasses::Admit): class
// membership rests on structure keys, logged answers and slot replay up to δ, none of which reads a
// lifted value; a value_lifting refusal only says that the representatives' values do not stand
// for the members'. So the per-unit synchronization verdict is the kept one.
// The re-run builds the per-unit program and runs its well-synchronization check, whose
// happens-before certificate orders the value side's memory actions (a failure there is an
// internal error), and its source lifecycle check; it skips the required-order lifecycle check,
// the race search and the frontend value validations (AnalyzeSourceArtifact's established
// verdict). Its synchronization report is the class-mode verified report as is: race free and
// without memory warnings, so what comes from class mode is verification_elapsed_seconds, the
// warnings of a recovered program (duplicate barrier initializations), and the program,
// memory-analysis and happens-before statistics, which count only the analysed units. The frontend
// metadata, the value analysis, the scalar fold and the equivalence check are the re-run's.
// Kept only when the verdict does not depend on the value side (no same-value write obligations)
// and the value side needs nothing from the per-unit race search (no accepted unordered writes: no
// duplicate-write groups or warnings); otherwise the re-run checks synchronization itself. O(1).
bool KeepsVerdict(const alg::MemoryRaceResult &memory) {
  return memory.status == alg::MemoryRaceStatus::kRaceFree && memory.search_complete &&
         memory.duplicate_write_groups.empty() && memory.same_value_write_groups.empty() &&
         memory.warning_count == 0 && memory.duplicate_tma_store_warnings.empty();
}
// The class-mode diagnostic at which a value stage refused lifting, when it names one.
struct LiftingStop {
  std::string code, message;
};
template <class Diagnostic>
std::optional<LiftingStop> LiftingStopOf(const std::optional<Diagnostic> &diagnostic) {
  if (!diagnostic) return std::nullopt;
  return LiftingStop{diagnostic->code, diagnostic->message};
}
// The per-unit re-run's value view, estimated from the class-mode one: its value instances
// scaled from the analysed units to all units of the launch (members of one class have equal
// shapes, so equal instance counts). Above kPerUnitRestartValueInstances the re-run is not
// started: the refusing stage ends with the budget refusal unit_restart_limit_exceeded, which
// names the class-mode stop. O(value nodes).
// The limit is a fixed algorithm budget, not a share of the memory budget, so that whether a launch
// re-runs does not depend on the host. The re-run holds the value view of every unit at once, and
// its value passes, memory elimination and scalar fold keep state per value instance; about 2^26
// instances is where that state outgrows the memory of a large analysis host, so a re-run above it
// would more likely end late at the memory budget than finish.
constexpr std::uint64_t kPerUnitRestartValueInstances = std::uint64_t(1) << 26;
// The value view's logical instances. O(value nodes).
std::uint64_t ValueInstances(const sentryward::value_ir::Graph& values) {
  std::uint64_t instances = 0;
  for (const auto& node : values.nodes()) instances += node.instanceCount();
  return instances;
}
// `value_instances`: ValueInstances of the class-mode value view (counted before it was released).
std::optional<Json> PerUnitRestartBudget(std::uint64_t value_instances,
    const sentryward::value_analysis::UnitLifting& lifting, std::string_view stage,
    const std::optional<LiftingStop>& stop) {
  unsigned __int128 units = 1, members = 0, instances = value_instances;
  for (unsigned axis = 0; axis < 3; ++axis) {
    if (!lifting.unit_shape[axis] || lifting.grid[axis] % lifting.unit_shape[axis]) return std::nullopt;
    units *= lifting.grid[axis] / lifting.unit_shape[axis];
  }
  for (const auto& [origin, list] : lifting.members) members += list.size();
  if (members >= units) return std::nullopt;
  // Units with a value view: the analysed ones, or, once later founders built none
  // (SuspendValues), those whose CTAs built one.
  unsigned __int128 viewed = units - members;
  if (lifting.value_view_ctas) {
    const unsigned __int128 size = lifting.unit_shape[0] * lifting.unit_shape[1] * lifting.unit_shape[2];
    viewed = std::min<unsigned __int128>(viewed, *lifting.value_view_ctas / size);
    if (!viewed) return std::nullopt;
  }
  const auto estimate = instances * units / viewed;
  if (estimate <= kPerUnitRestartValueInstances) return std::nullopt;
  const auto clamp = [](unsigned __int128 x) { return x > UINT64_MAX ? UINT64_MAX : std::uint64_t(x); };
  std::string message = "The class-mode " + std::string(stage) + " refused lifting";
  Json report = {{"status", "analysis_budget_exceeded"}};
  if (stop) {
    message += " at " + stop->code + " (" + stop->message + ")";
    report["class_mode_diagnostic"] = {{"code", stop->code}, {"message", stop->message}};
  }
  message += "; the per-unit re-run would build about " +
      std::to_string(clamp(estimate)) + " value instances (" + std::to_string(clamp(units)) + " units from " +
      std::to_string(clamp(units - members)) + " analysed" +
      (viewed != units - members ? ", scaled from the " + std::to_string(clamp(viewed)) + " with a value view" : std::string()) +
      "), above the limit " +
      std::to_string(kPerUnitRestartValueInstances) + ".";
  report["diagnostic"] = {{"code", "unit_restart_limit_exceeded"}, {"message", message}};
  report["resource"] = {{"origin", "algorithm_budget"}, {"kind", "per_unit_value_instances"},
      {"limit", kPerUnitRestartValueInstances}, {"estimate", clamp(estimate)}};
  return report;
}
// Class mode: whether the per-unit value analysis, estimated as PerUnitRestartBudget does from the
// value views of the `viewed` units built so far, is over kPerUnitRestartValueInstances (`units`:
// the launch's). O(value nodes).
bool PerUnitValuesOverBudget(const sentryward::value_ir::Graph& values, std::uint64_t units, std::uint64_t viewed) {
  if (!viewed) return false;
  return static_cast<unsigned __int128>(ValueInstances(values)) * units / viewed > kPerUnitRestartValueInstances;
}
// A value stage of the class-mode attempt whose result cannot stand for the members refuses
// lifting; every such refusal (UnitClassRestart "value_lifting") comes here. It throws for the
// per-unit re-run, or, above the re-run's budget (PerUnitRestartBudget), returns the budget
// refusal, which the refusing stage reports as its result. `stage` names that stage.
// `value_instances`: ValueInstances of the class-mode value view.
Json RefuseLifting(std::uint64_t value_instances, const sentryward::value_analysis::UnitLifting& lifting,
                   std::string_view stage, const std::optional<LiftingStop>& stop = std::nullopt) {
  if (auto refused = PerUnitRestartBudget(value_instances, lifting, stage, stop)) return std::move(*refused);
  throw alg::UnitClassRestart{"value_lifting"};
}
// `lifting`: cross-CTA value lifting mode (unit_lifting.h), else null; a
// refusal throws UnitClassRestart, unless the per-unit re-run exceeds its budget.
} // namespace
