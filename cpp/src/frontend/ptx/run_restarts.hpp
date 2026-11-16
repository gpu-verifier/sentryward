#pragma once
#include "run_support.hpp"


// Per-unit restart: a class-mode attempt that cannot finish exactly throws
// UnitClassRestart, and main analyses the launch again in a second attempt with
// class mode off (RunAttempt). The reasons: the class tables cannot stand for
// the members (unit_closure, tensormap_locality, executed_global_atomic,
// binding_evidence, repeated_entries), the cross-unit race pass is undecided
// (cross_unit_unknown:*, cross_unit_cap:*, cross_unit_tie_break,
// cross_unit_same_value), or the representatives' values cannot stand for the
// members' (value_lifting, RefuseLifting). Each attempt owns its process-level
// state (AnalysisScope). Without a kept verdict the per-unit attempt analyses
// exactly as a run without classes; a value_lifting restart may also carry the
// class-mode synchronization verdict (KeepsVerdict), which the per-unit attempt
// reports while it redoes the value side. Its clock continues from the
// class-mode attempt's start, so the time budget and the elapsed times also
// count the release of the class-mode attempt. Like UnitClassRestart, not a
// std::exception, so that no stage of the attempt converts it into a report.
// Table-atom attempt: a class-mode attempt over an input with fixed shape-array tables, which reads
// their elements exactly, restarts with class mode's table atoms once the per-unit value analysis,
// estimated from the value views of the founders consumed so far, is over the per-unit re-run budget
// (UnitClassRestart "table_atoms", PerUnitValuesOverBudget): no value stage could then complete per
// unit, so the attempt admits units on the table-lookup forms of their addresses. Its clock continues
// from the first attempt's start. Not a std::exception, like PerUnitRestart.
struct TableAtomRestart {
  Clock::time_point started;
};
// Founder order: a class-mode attempt that admits containment members first plans which units found
// classes (ptx::ContainmentPlan) and takes those first. Its representatives are then not the first
// units of their classes in launch order, which the report's per-unit evidence (witness order,
// repeated entries, numbering) relies on; so when such an attempt does not end with a clean
// synchronization pass, the attempt is redone in unit order (same mode, the clock continues). Not a
// std::exception, like TableAtomRestart.
struct FounderOrderRestart {
  Clock::time_point started;
};
// Founder planning costs one more execution of every unit, so an attempt that admits containment
// members starts in unit order without a plan; when its class loop meets the necessary condition
// for a plan to change anything (UnitClasses::PlanUseful: usually within the first units), the
// attempt is restarted with the plan pass (same mode, the clock continues). A launch whose units
// are not contained in one another never pays for the plan. Not a std::exception, like
// TableAtomRestart.
struct FounderPlanRestart {
  Clock::time_point started;
};
struct PerUnitRestart {
  std::string reason; // UnitClassRestart::reason
  // When the class-mode attempt started (Monitor::start).
  Clock::time_point started;
  // The class-mode verified report and when it was reached, in seconds from
  // `started`; empty when the per-unit attempt checks synchronization itself.
  std::string verdict;
  double verdict_elapsed_s = 0;
};
// What one call reads once and both attempts share: input files by path and
// the stdin reference, so the per-unit attempt checks exactly the bytes the
// class-mode attempt checked. Retains O(input bytes) for the call.
class CallInputs {
public:
  // The file's bytes, or null when it cannot be opened (not cached; each reader
  // reports its own error).
  const std::string *File(const std::string &path) {
    if (const auto found = files_.find(path); found != files_.end()) return &found->second;
    std::ifstream file(path, std::ios::binary);
    if (!file) return nullptr;
    return &files_.emplace(path, std::string{std::istreambuf_iterator<char>(file), {}}).first->second;
  }
  // The file's bytes; FileError when it cannot be opened.
  const std::string &Read(const std::string &path) {
    if (const auto *bytes = File(path)) return *bytes;
    throw FileError(path, false);
  }
  // The stdin reference, read on first use.
  const std::string &ReferenceStdin() {
    if (!reference_stdin_)
      reference_stdin_.emplace(std::istreambuf_iterator<char>(std::cin), std::istreambuf_iterator<char>());
    return *reference_stdin_;
  }

private:
  std::map<std::string, std::string> files_;
  std::optional<std::string> reference_stdin_;
};
// The memory class mode's look-ahead may hold in executed units: a quarter of
// --rss-limit-mib, or without a limit a quarter of the memory the process may use (UsableMemory;
// docs/output-json.md, "Resource limits"). UINT64_MAX when neither is known.
std::uint64_t LookAheadBudget(const std::optional<std::uint64_t> &rss_mib) {
  if (rss_mib) return *rss_mib * 1048576 / 4;
  const auto usable = UsableMemory();
  return usable ? *usable / 4 : UINT64_MAX;
}
// Bytes the allocator has handed out (glibc), which sizes class mode's
// look-ahead window; nullopt elsewhere (the window is then the worker count).
std::optional<std::uint64_t> AllocatedBytes() {
#if defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 33))
  const auto info = mallinfo2();
  return std::uint64_t(info.uordblks) + std::uint64_t(info.hblkhd);
#else
  return std::nullopt;
#endif
}
// The driver's accounting of one executed CTA: what it adds to the launch's statistics and
// frontend assumptions, read from the CTA's facts once. A pure function of that CTA's
// execution, so class mode's look-ahead workers summarize the units they execute and the
// driver applies the summaries in unit order. The per-lane quantities are added up per CTA:
// counts are sums, and a branch decision's assumption and rule are those of its last lane, as
// when the driver adds lane by lane. O(lanes x per-lane decisions + static instructions).
struct CtaAccount {
  ptx::TemplateExecutionStatistics execution;
  ptx::ThreadRecordStatistics storage;
  std::uint64_t records = 0, assumed_assertions = 0;
  std::uint64_t representative_branch_uses = 0, bounded_loop_uses = 0;
  std::uint64_t election_uses = 0, clc_uses = 0, atomic_uses = 0;
  std::map<std::uint64_t, std::uint64_t> opaque_regions;
  std::map<std::uint64_t, ptx::DataBranchDecision> data_branch_decisions;
  std::set<std::uint32_t> static_choice_lines;
  // Whether the CTA executed one of `atomic_sites` (class mode's unit-independence precondition).
  bool executes_atomic_site = false;
};
// Adds a later decision at one branch line to its summary: the counts add up, and the
// assumption and rule become the later one's.
void AddDecision(ptx::DataBranchDecision &summary, const ptx::DataBranchDecision &decision) {
  summary.assumption = decision.assumption;
  summary.rule = decision.rule;
  summary.taken += decision.taken;
  summary.not_taken += decision.not_taken;
}
CtaAccount SummarizeCta(const ptx::CtaFacts &cta_facts, const ptx::Function &function,
                        std::span<const std::uint32_t> atomic_sites) {
  const ptx::FactSource::ConsumerScope driver(cta_facts, {ptx::Consumer::Driver, 0});
  CtaAccount account{.execution = cta_facts.TemplateStatistics(), .storage = cta_facts.RecordCompression()};
  for (std::size_t index = 0; index < cta_facts.LaneCount(); ++index) {
    const auto lane = cta_facts.Lane(index);
    const auto aggregates = cta_facts.Aggregates(lane);
    account.records += cta_facts.Size(lane);
    account.assumed_assertions += aggregates.assumed_assertions;
    for (const auto &[line, count] : aggregates.opaque_regions) account.opaque_regions[line] += count;
    for (const auto &[line, decision] : aggregates.data_branch_decisions) {
      AddDecision(account.data_branch_decisions[line], decision);
      (decision.assumption == "representative_branches" ? account.representative_branch_uses
                                                         : account.bounded_loop_uses) +=
          decision.taken + decision.not_taken;
    }
  }
  account.static_choice_lines = cta_facts.StaticBranchChoiceLines();
  const auto static_counts = cta_facts.StaticCounts();
  for (std::size_t id = 0; id < static_counts.size(); ++id) {
    const auto &op = function.instructions[id].opcode;
    const auto count = static_counts[id];
    if (op == "elect") account.election_uses += count;
    if (op == "clusterlaunchcontrol") account.clc_uses += count;
    if (op == "atom" || op == "red") account.atomic_uses += count;
  }
  for (const auto id : atomic_sites)
    if (id < static_counts.size() && static_counts[id] != 0) account.executes_atomic_site = true;
  return account;
}
// One analysis attempt of the launch: the class-mode attempt, or with
// `unit_restart` the per-unit attempt after it (PerUnitRestart). Prints the
// report and returns the exit status; a class-mode attempt that restarts throws
// PerUnitRestart instead, after its state is released.
