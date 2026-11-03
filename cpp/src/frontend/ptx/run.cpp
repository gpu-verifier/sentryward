#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/branch_headers.inc"
#endif
#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/graph_headers.inc"
#endif
#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/export_headers.inc"
#endif
#include "sentryward/result.h"
#include "sentryward/algorithm/base/analysis_scope.h"
#include "sentryward/algorithm/diagnostics/performance_profile.h"
#include "sentryward/algorithm/diagnostics/ir_size_profile.h"
#include "sentryward/algorithm/diagnostics/memory_profile.h"
#include "sentryward/frontend/ptx/config.h"
#include "sentryward/frontend/ptx/public_input.h"
#include "sentryward/frontend/ptx/execution.h"
#include "sentryward/frontend/ptx/lowering.h"
#include "sentryward/frontend/ptx/layouts.h"
#include "sentryward/frontend/ptx/value_ir.h"
#include "sentryward/frontend/ptx/branch_scope_capture.h"
#include "sentryward/frontend/ptx/semantic_program.h"
#include "sentryward/frontend/ptx/serialization.h"
#include "sentryward/frontend/ptx/unit_facts.h"
#include "sentryward/frontend/ptx/unit_classes.h"
#include "sentryward/algorithm/checker/unit_quotient.h"
#include "sentryward/frontend/ptx/query_log.h"
#include "sentryward/frontend/ptx/look_ahead.h"
#include "sentryward/value_ir/operations.h"
#include "sentryward/value_analysis/passes.h"
#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/checkpoint_headers.inc"
#endif
#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/checkpoint_version.inc"
#endif
#include <openssl/evp.h>
#if defined(__GLIBC__)
#include <malloc.h>
#endif
#include <filesystem>
#include "sentryward/value_analysis/scalar_fold_lowering.h"
#include "sentryward/value_analysis/scalar_fold_failure_codes.h"
#include "sentryward/scalar_fold/graph.h"
#include "sentryward/scalar_fold/artifact.h"
#include "production.hpp"
#include "fixed_input_context.hpp"
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <sys/resource.h>
#include <thread>
#include <unistd.h>
#ifdef __APPLE__
#include <mach/mach.h>
#endif
namespace ptx = sentryward::frontend::ptx;
namespace alg = sentryward::algorithm;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
namespace {
struct Options {
  std::string ptx, config, report, dump_source, dump_value_ir, dump_output_expressions, dump_scalar_fold;
  std::string cache_correctness, resume_correctness;
  std::string dump_branch_scope;
  std::string equivalence_mapping;
  std::string reference_provenance;
  bool reference_stdin = false;
  double equivalence_timeout = 100;
  std::optional<std::pair<std::uint64_t, std::uint64_t>> value_thread_range;
  bool extract = false;
  bool compact_source_cache = false;
  bool compress_thread_records = true;
  bool semantic = false;
  bool analyze_values = false;
  bool trusted_reference = false;
  bool analyze_scalar_fold = false;
  bool exhaustive_memory_search = false;
  bool continue_memory_search = false;
  double timeout = 0, sync_timeout = 600;
  std::optional<std::uint64_t> rss_mib;
  // --default-budget-jobs: the number of verifiers that run at once and share the default memory
  // budget (no --rss-limit-mib); each enforces that share only (EnforcedMemoryMib).
  std::optional<std::uint64_t> default_budget_jobs;
  std::size_t threads = 8;
  bool require_equivalence = false;
  std::string reference_failure_json;
  std::uint64_t max_steps_per_thread = 100000;
  alg::PtxAssumptions assumptions;
  // Cross-CTA class mode and value lifting are decided automatically (PlanUnits).
};
std::uint64_t Count(const std::string &text) {
  std::uint64_t n{};
  auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), n);
  if (error != std::errc{} || end != text.data() + text.size())
    throw std::invalid_argument("expected nonnegative integer: " + text);
  return n;
}
double Seconds(const std::string &text) {
  std::size_t end{};
  double n = std::stod(text, &end);
  if (end != text.size() || !std::isfinite(n))
    throw std::invalid_argument("expected finite timeout: " + text);
  return n;
}
// How often the monitor samples memory and elapsed time (Monitor::Watch).
constexpr auto kMonitorInterval = std::chrono::milliseconds(100);
// The default memory budget's share of the memory the process may use (UsableMemory,
// EnforcedMemoryMib), as numerator / denominator. The rest is headroom for what the budget does
// not see in time (the monitor samples every kMonitorInterval) and for the operating system, so
// that a run reaching the budget is stopped by the monitor rather than by the operating system.
constexpr std::uint64_t kDefaultBudgetNumerator = 9, kDefaultBudgetDenominator = 10;
Options Arguments(int argc, char **argv) {
  Options o;
  for (int k = 1; k < argc; ++k) {
    std::string a = argv[k];
    auto value = [&] {
      if (++k >= argc)
        throw std::invalid_argument("missing value after " + a);
      return std::string(argv[k]);
    };
    if (a == "--ptx")
      o.ptx = value();
    else if (a == "--config")
      o.config = value();
    else if (a == "--report")
      o.report = value();

#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/source_options.inc"
#endif
#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/checkpoint_options.inc"
#endif


#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/value_options.inc"
#endif
#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/branch_options.inc"
#endif

    else if (a == "--trusted-reference") {
      o.trusted_reference = true;
      o.analyze_values = true;
    }
    else if (a == "--require-equivalence") { o.require_equivalence=true; o.analyze_values=o.analyze_scalar_fold=true; }
    else if (a == "--reference-failure-json") o.reference_failure_json=value();
    else if (a == "--reference-stdin") {
      o.reference_stdin = true;
      o.analyze_values = o.analyze_scalar_fold = true;
    } else if (a == "--equivalence-mapping") o.equivalence_mapping = value();
    else if (a == "--reference-provenance") o.reference_provenance = value();
    else if (a == "--equivalence-timeout-seconds") o.equivalence_timeout = std::stod(value());
    else if (a == "--analyze-values")
      o.analyze_values = true;

#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/expression_options.inc"
#endif
    else if (a == "--analyze-scalar-fold") {
      o.analyze_scalar_fold = true;
      o.analyze_values = true;
    }
#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/fold_options.inc"
#endif
    else if (a == "--extract-only")
      throw std::invalid_argument("extract-only mode has been removed; request verification");
    else if (a == "--no-thread-record-compression")
      o.compress_thread_records = false;
    else if (a == "--semantic-only")
      throw std::invalid_argument("semantic-only mode has been removed; request verification");
    else if (a == "--exhaustive-memory-search")
      o.exhaustive_memory_search = true;
    else if (a == "--continue-memory-search")
      o.continue_memory_search = true;
    else if (a == "--timeout-seconds")
      o.timeout = Seconds(value());
    else if (a == "--wellsync-timeout-seconds")
      o.sync_timeout = Seconds(value());
    else if (a == "--rss-limit-mib")
      o.rss_mib = Count(value());
    else if (a == "--default-budget-jobs")
      o.default_budget_jobs = Count(value());
    else if (a == "--value-thread-range") {
      const auto text = value();
      const auto colon = text.find(':');
      if (colon == std::string::npos) throw std::invalid_argument("expected FIRST:COUNT");
      const auto first = Count(text.substr(0, colon)), count = Count(text.substr(colon+1));
      if (!count || first > UINT32_MAX || count > UINT32_MAX-first)
        throw std::invalid_argument("invalid thread range");
      o.value_thread_range = {{first, first+count}};
    } else if (a == "--max-steps-per-thread")
      o.max_steps_per_thread = Count(value());
    else if (a == "--sync-threads")
      o.threads = Count(value());
    else if (a == "--assume" || a == "--no-assume") {
      auto name = value();
      auto allowed = alg::DefaultPtxAssumptions();
      if (!allowed.contains(name))
        throw std::invalid_argument("unknown semantic assumption " + name);
      o.assumptions[name] = a == "--assume";
    } else if (a == "--help") {
      std::cout
          << "sentryward-ptx --ptx FILE --config FILE [--report FILE] "
             "\n  --wellsync-timeout-seconds "
             "600  WellSync-only budget (default)\n  --timeout-seconds SECONDS "
             "     Optional whole-process diagnostic budget\n  --rss-limit-mib "
             "MIB  Optional positive memory budget (default: "
          << kDefaultBudgetNumerator * 100 / kDefaultBudgetDenominator
          << "% of the memory the process may use: physical memory,\n"
             "                     bounded by its cgroup limit)\n"
             "  --default-budget-jobs N  Without --rss-limit-mib: N verifiers run at once and share the\n"
             "                     default budget; each enforces 1/N of it (default 1)\n  --sync-threads 8\n  "
             "--exhaustive-memory-search  Enumerate/count all race pairs\n"
#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/value_help.inc"
#endif
#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/branch_help.inc"
#endif
             "--value-thread-range FIRST:COUNT  Extract a thread subset; requires cross-subset validation\n"
             "--no-thread-record-compression  Disable template execution and shared records for comparison\n"
             "--max-steps-per-thread N  PTX instructions one thread may execute (default 100000); threads\n"
             "                           that run one shared path together get 100*N steps as a group\n"
             "--analyze-values  Verify both semantic views and eliminate constant-address memory\n"
#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/checkpoint_save_help.inc"
#endif
#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/checkpoint_replay_help.inc"
#endif
             "--trusted-reference  Trust correctness; preserve synchronous memory order without verification\n"
#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/expression_help.inc"
#endif
             "--analyze-scalar-fold  Lower checked output expressions to scalar + Fold\n"
#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/fold_help.inc"
#endif
             "--continue-memory-search  Scan all optimized ordering constraints; retain one witness\n";
#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/source_help.inc"
#endif
      std::exit(0);
    } else
      throw std::invalid_argument("unknown option " + a);
  }
#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/enable_export_values.inc"
#endif
  if (o.ptx.empty() || o.config.empty())
    throw std::invalid_argument("--ptx and --config are required");
#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/validate_exports.inc"
#endif
  if (o.value_thread_range && !o.analyze_values)
    throw std::invalid_argument("--value-thread-range requires --analyze-values");
  if (o.analyze_values && (o.extract || o.semantic))
    throw std::invalid_argument("--analyze-values requires verification");
#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/validate_expression_export.inc"
#endif
  if (o.trusted_reference && (o.exhaustive_memory_search || o.continue_memory_search))
    throw std::invalid_argument("--trusted-reference does not perform memory-race search");
  if (o.exhaustive_memory_search && (o.extract || o.semantic))
    throw std::invalid_argument("--exhaustive-memory-search requires verification");
  if (o.continue_memory_search &&
      (o.extract || o.semantic || o.exhaustive_memory_search))
    throw std::invalid_argument(
        "--continue-memory-search requires verification and cannot be combined with exhaustive search");
  if (o.default_budget_jobs && *o.default_budget_jobs == 0)
    throw std::invalid_argument("--default-budget-jobs must be positive");
  if (o.default_budget_jobs && o.rss_mib)
    throw std::invalid_argument("--default-budget-jobs shares the default memory budget; it cannot be combined "
                                "with --rss-limit-mib");
  if (o.timeout < 0 || o.sync_timeout <= 0 || (o.rss_mib && (*o.rss_mib == 0 || *o.rss_mib > UINT64_MAX / (1024 * 1024))) || o.threads < 1 ||
      o.threads > 64 || o.max_steps_per_thread == 0 || (o.extract && o.semantic))
    throw std::invalid_argument("invalid timeout/RSS/thread option");
  return o;
}
struct FileError : std::runtime_error {
  std::string path;
  bool writing;
  FileError(std::string p, bool w) : std::runtime_error(std::string(w ? "cannot write " : "cannot read ") + p),
      path(std::move(p)), writing(w) {}
};
Json FileFailure(const FileError& error) {
  return {{"schema","spverify.native-ptx-run/v1"}, {"status",error.writing ? "output_file_error" : "input_file_error"},
      {"stage","file_io"}, {"reason",error.writing ? "file_write_failed" : "file_read_failed"},
      {"path",error.path}, {"message",error.what()}, {"memory_analysis_complete",false},{"warnings_complete",false}};
}
void Write(const std::string &path, const Json &result) {
  if (path.empty())
    return;
  std::ofstream file(path);
  if (!file)
    throw FileError(path, true);
  file << result.dump(2) << '\n';
}
std::uint64_t Rss() {
#ifdef __linux__
  std::ifstream f("/proc/self/statm");
  std::uint64_t total = 0, resident = 0;
  f >> total >> resident;
  return resident * sysconf(_SC_PAGESIZE);
#else
  struct rusage r{};
  getrusage(RUSAGE_SELF, &r);
  return r.ru_maxrss;
#endif
}
std::uint64_t PeakRss() {
  struct rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
#ifdef __linux__
  return std::uint64_t(usage.ru_maxrss) * 1024;
#else
  return usage.ru_maxrss;
#endif
}
std::uint64_t MemoryFootprint() {
#ifdef __APPLE__
  task_vm_info_data_t info{};
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS)
    return info.phys_footprint;
#endif
  return Rss();
}
// The physical memory, sysconf(_SC_PHYS_PAGES) x _SC_PAGESIZE; nullopt when unknown.
std::optional<std::uint64_t> PhysicalMemory() {
  const auto pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGESIZE);
  if (pages <= 0 || page <= 0) return std::nullopt;
  return std::uint64_t(pages) * std::uint64_t(page);
}
#ifdef __linux__
// A cgroup memory limit at or above this is no limit: cgroup v1 reports none as its largest page
// counter (close to 2^63 bytes), far above any machine's memory.
constexpr std::uint64_t kUnlimitedCgroupBytes = std::uint64_t(1) << 60;
#endif
// The memory limit of this process's cgroup on Linux: the least limit among its cgroups and their
// ancestors, cgroup v2 memory.max under /sys/fs/cgroup and cgroup v1 memory.limit_in_bytes under
// /sys/fs/cgroup/memory, at the paths /proc/self/cgroup names. "max", unreadable files and
// kUnlimitedCgroupBytes and above are no limit; a path outside the visible hierarchy ("..") reads
// only the mount's root. nullopt when no limit is found. O(cgroup depth^2) file reads.
std::optional<std::uint64_t> CgroupMemoryLimit() {
  std::optional<std::uint64_t> least;
#ifdef __linux__
  std::ifstream cgroups("/proc/self/cgroup");
  for (std::string line; std::getline(cgroups, line);) {
    const auto first = line.find(':');
    const auto second = first == std::string::npos ? first : line.find(':', first + 1);
    if (second == std::string::npos) continue;
    const auto controllers = "," + line.substr(first + 1, second - first - 1) + ",";
    std::filesystem::path root;
    const char *file = nullptr;
    if (line.compare(0, first, "0") == 0 && controllers == ",,") {
      root = "/sys/fs/cgroup";
      file = "memory.max";
    } else if (controllers.find(",memory,") != std::string::npos) {
      root = "/sys/fs/cgroup/memory";
      file = "memory.limit_in_bytes";
    } else
      continue;
    std::vector<std::string> components;
    std::istringstream path(line.substr(second + 1));
    for (std::string part; std::getline(path, part, '/');) {
      if (part == "..") {
        components.clear();
        break;
      }
      if (!part.empty() && part != ".") components.push_back(part);
    }
    for (std::size_t depth = components.size() + 1; depth-- > 0;) {
      auto directory = root;
      for (std::size_t k = 0; k < depth; ++k) directory /= components[k];
      std::ifstream limit(directory / file);
      std::string text;
      if (!(limit >> text)) continue;
      std::uint64_t bytes{};
      const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), bytes);
      if (error != std::errc{} || end != text.data() + text.size() || !bytes || bytes >= kUnlimitedCgroupBytes)
        continue;
      least = least ? std::min(*least, bytes) : bytes;
    }
  }
#endif
  return least;
}
// The memory the process may use: the physical memory, bounded by its cgroup's limit; nullopt
// when neither is known.
std::optional<std::uint64_t> UsableMemory() {
  const auto physical = PhysicalMemory(), cgroup = CgroupMemoryLimit();
  if (physical && cgroup) return std::min(*physical, *cgroup);
  return physical ? physical : cgroup;
}
// The memory budget the monitor enforces, in MiB: --rss-limit-mib, or without it the default
// budget, kDefaultBudgetNumerator / kDefaultBudgetDenominator of UsableMemory, divided by
// --default-budget-jobs, so that a run reaching it ends with memory_budget_exceeded instead of
// being killed by the operating system (docs/output-json.md, "Resource limits"); nullopt when the
// usable memory is unknown. Only enforcement uses the default and its share: rss_limit_mib reports
// the given limit, and the budgets derived from it (class logs, LookAheadBudget) see none.
std::optional<std::uint64_t> EnforcedMemoryMib(const Options &options) {
  if (options.rss_mib) return options.rss_mib;
  const auto usable = UsableMemory();
  if (!usable) return std::nullopt;
  return *usable / kDefaultBudgetDenominator * kDefaultBudgetNumerator / 1048576 /
         options.default_budget_jobs.value_or(1);
}
struct Monitor {
  const Options &options;
  const std::optional<std::uint64_t> memory_limit_mib = EnforcedMemoryMib(options);
  // The enforced budget's origin: given (--rss-limit-mib) or the default.
  const char *memory_origin = options.rss_mib ? "configured_budget" : "default_budget";
  Clock::time_point start;
  std::mutex mutex;
  std::condition_variable changed;
  bool finished = false;
  std::string stage = "startup";
  Json frontend_checkpoint = Json::object();
  std::atomic<std::uint64_t> peak = 0;
  std::atomic<std::uint64_t> peak_footprint = 0;
  // The per-unit attempt's internal.frontend.unit_restart record, else null.
  const Json unit_restart;
  std::thread worker;
  // `started`: when the call's class-mode attempt started, for a per-unit attempt, whose clock
  // continues from it (PerUnitRestart::started); else now.
  Monitor(const Options &o, std::optional<Clock::time_point> started, Json restart)
      : options(o), start(started.value_or(Clock::now())), unit_restart(std::move(restart)),
        worker([this] { Watch(); }) {}
  // The memory resource of a report stopped by the memory budget; `limit` keeps the caller's
  // number type. A shared default budget also names its verifier count (--default-budget-jobs).
  Json MemoryResource(Json limit) const {
    Json resource{{"kind", "memory"}, {"origin", memory_origin}, {"limit", std::move(limit)}, {"unit", "MiB"}};
    if (options.default_budget_jobs) resource["default_budget_jobs"] = *options.default_budget_jobs;
    return resource;
  }
  ~Monitor() { Stop(); }
  double Elapsed() const {
    return std::chrono::duration<double>(Clock::now() - start).count();
  }
  void Stage(std::string s) {
    std::lock_guard lock(mutex);
    stage = std::move(s);
  }
  void Frontend(const Json &metadata, const alg::SourceArtifact &artifact) {
    Json checkpoint{{"frontend", metadata},
                    {"frontend_source_violations", Json::array()},
                    {"verification_limitations", Json::array()}};
    for (const auto &violation : artifact.source_violations)
      checkpoint["frontend_source_violations"].push_back(
          {{"code", violation.code}, {"message", violation.message}});
    for (const auto &limitation : artifact.verification_limitations)
      checkpoint["verification_limitations"].push_back(
          {{"code", limitation.code}, {"message", limitation.message}});
    std::lock_guard lock(mutex);
    // Merged: a per-unit attempt's checkpoint already holds the kept verdict (MemoryVerified).
    frontend_checkpoint.update(checkpoint);
  }
  void MemoryStrategy(const std::string &strategy, bool fallback) {
    std::lock_guard lock(mutex);
    stage = "verification.memory";
    frontend_checkpoint["memory_analysis_entered"] = true;
    auto &memory = frontend_checkpoint["memory_analysis"];
    memory = {{"strategy", strategy}, {"candidate_generation_fallback", fallback}};
    if (fallback)
      memory["strategy_note"] = {
          {"code", "memory_conflict_candidate_generation_unavailable"},
          {"message", alg::kMemoryCandidateFallback}};
  }
  // `at`: when the verdict was reached (a kept class-mode verdict's own time), else now.
  void MemoryVerified(std::string_view verified_report, std::optional<double> at = std::nullopt) {
    // One small report snapshot, not another graph traversal or verification.
    // Preserve counts, warning samples and assumption usage on a later timeout.
    auto checkpoint = Json::parse(verified_report);
    // Same-value write obligations not yet proven by the value analysis leave an unordered
    // write pair (source_json.cpp): not race free until SameValueWritesProven.
    const bool pending = checkpoint.value("same_value_writes", Json::object()).value("status", "") == "pending";
    checkpoint["verification_status"] = pending ? Json("potential_race") : checkpoint.at("status");
    checkpoint.erase("status");
    checkpoint.erase("schema");
    checkpoint["verification_elapsed_seconds"] = at ? *at : Elapsed();
    checkpoint["warnings_complete"] = true;
    std::lock_guard lock(mutex);
    stage = "value_analysis.memory_effect_binding";
    frontend_checkpoint.update(checkpoint);
  }
  void SameValueWritesProven() {
    std::lock_guard lock(mutex);
    frontend_checkpoint["verification_status"] = "race_free";
    frontend_checkpoint["same_value_writes"]["status"] = "proven";
  }
  // The settled verification report: its warnings, or the unproven pair as the race.
  void SameValueWritesSettled(const Json &report) {
    std::lock_guard lock(mutex);
    if (!frontend_checkpoint.contains("verification_status")) return;
    frontend_checkpoint["verification_status"] = report.at("status");
    for (const auto *key : {"same_value_writes", "warnings", "warning_count", "warning_samples_truncated",
                            "races", "race_count", "race_samples_truncated", "diagnostics"})
      if (report.contains(key)) frontend_checkpoint[key] = report[key];
  }
  void Serialized(const Json& statistics,bool complete) {
    std::lock_guard lock(mutex);
    frontend_checkpoint["value_analysis"]={{"serialization",statistics},{"serialization_complete",complete}};
  }
  void RecordUnitRestart(Json &report) const {
    if (unit_restart.is_null()) return;
    auto &frontend = report["frontend"];
    if (frontend.is_null()) frontend = Json::object();
    if (frontend.is_object()) frontend["unit_restart"] = unit_restart;
  }
  void Stop() {
    {
      std::lock_guard lock(mutex);
      finished = true;
    }
    changed.notify_all();
    if (worker.joinable())
      worker.join();
    peak.store(std::max(peak.load(), PeakRss()));
    peak_footprint.store(std::max(peak_footprint.load(), MemoryFootprint()));
  }
  void Watch() {
    std::unique_lock lock(mutex);
    while (!finished) {
      if (changed.wait_for(lock, kMonitorInterval, [this] { return finished; }))
        break;
      auto rss = Rss();
      peak.store(std::max(peak.load(), rss));
      auto footprint = MemoryFootprint();
      peak_footprint.store(std::max(peak_footprint.load(), footprint));
      double elapsed = Elapsed();
      bool timeout = options.timeout > 0 && elapsed >= options.timeout,
           limit = memory_limit_mib && std::max(rss, footprint) > *memory_limit_mib * 1024 * 1024;
      if (!timeout && !limit)
        continue;
      Json result{
          {"schema", "spverify.native-ptx-run/v1"},
          {"status", limit ? "memory_limit_exceeded" : "timeout"},
          {"timeout_scope", "whole_pipeline"},
          {"resource", limit ? MemoryResource(double(*memory_limit_mib))
                             : Json{{"kind", "time"}, {"origin", "configured_budget"},
                                    {"limit", options.timeout}, {"unit", "seconds"}}},
          {"elapsed_seconds", elapsed},
          {"peak_rss_mib", peak.load() / 1048576.0},
          {"peak_footprint_mib", peak_footprint.load() / 1048576.0},
          {"active_stage", stage},
          {"memory_analysis_complete", false},
          {"warnings_complete", false},
          {"implementation", "cpp"},
          {"rss_limit_mib", options.rss_mib ? Json(*options.rss_mib) : Json(nullptr)}};
      result.update(frontend_checkpoint);
      RecordUnitRestart(result);
      if(options.require_equivalence || options.reference_stdin) {
        if(stage=="equivalence.object_comparison")
          result["equivalence"]={{"status",limit?"memory_limit_exceeded":"timeout"},
            {"reason",limit?"rss_limit":"time_budget"},{"stage",stage}};
        else result["equivalence"]={{"status","blocked"},{"reason","verification_prerequisite_not_satisfied"},
          {"prerequisite_status",result.value("status","timeout")},{"stage",stage}};
      }
      result = sentryward::results::PublicReport(std::move(result), "kernel_verification");
      try {
        Write(options.report, result);
      } catch (...) {
      }
      std::string line = "SENTRYWARD_RUN " + result.dump() + "\n";
      std::cerr << line;
      std::cerr.flush();
      std::cout << result.dump() << '\n';
      std::cout.flush();
      std::_Exit(limit ? 137 : 124);
    }
  }
};

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

// Cross-CTA value lifting in class mode: the class tables the scalar fold
// instantiates members from. A unit is named by its origin (rank 0's
// coordinates); δ is keyed as ConstantRegion::allocation (AllocationIdentity);
// CTA ids are linear, renamed rank by rank.
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
std::optional<Json> PerUnitRestartBudget(const sentryward::value_ir::Graph& values,
    const sentryward::value_analysis::UnitLifting& lifting, std::string_view stage,
    const std::optional<LiftingStop>& stop) {
  unsigned __int128 units = 1, members = 0, instances = 0;
  for (unsigned axis = 0; axis < 3; ++axis) {
    if (!lifting.unit_shape[axis] || lifting.grid[axis] % lifting.unit_shape[axis]) return std::nullopt;
    units *= lifting.grid[axis] / lifting.unit_shape[axis];
  }
  for (const auto& [origin, list] : lifting.members) members += list.size();
  if (members >= units) return std::nullopt;
  for (const auto& node : values.nodes()) instances += node.instanceCount();
  const auto estimate = instances * units / (units - members);
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
      std::to_string(clamp(units - members)) + " analysed), above the limit " +
      std::to_string(kPerUnitRestartValueInstances) + ".";
  report["diagnostic"] = {{"code", "unit_restart_limit_exceeded"}, {"message", message}};
  report["resource"] = {{"origin", "algorithm_budget"}, {"kind", "per_unit_value_instances"},
      {"limit", kPerUnitRestartValueInstances}, {"estimate", clamp(estimate)}};
  return report;
}
// A value stage of the class-mode attempt whose result cannot stand for the members refuses
// lifting; every such refusal (UnitClassRestart "value_lifting") comes here. It throws for the
// per-unit re-run, or, above the re-run's budget (PerUnitRestartBudget), returns the budget
// refusal, which the refusing stage reports as its result. `stage` names that stage.
Json RefuseLifting(const sentryward::value_ir::Graph& values, const sentryward::value_analysis::UnitLifting& lifting,
                   std::string_view stage, const std::optional<LiftingStop>& stop = std::nullopt) {
  if (auto refused = PerUnitRestartBudget(values, lifting, stage, stop)) return std::move(*refused);
  throw alg::UnitClassRestart{"value_lifting"};
}
// `lifting`: cross-CTA value lifting mode (unit_lifting.h), else null; a
// refusal throws UnitClassRestart, unless the per-unit re-run exceeds its budget.
Json AnalyzeScalarFold(const sentryward::value_ir::Graph& values,
                       const sentryward::value_analysis::MemoryEliminationResult& eliminated,
                       const Options& options, Monitor& monitor,
                       std::optional<sentryward::scalar_fold::Artifact>& artifact,
                       const sentryward::value_analysis::UnitLifting* lifting,
                       const std::map<std::string, sentryward::scalar_fold::ComparedOutput>* compared) {
  namespace sf = sentryward::scalar_fold;
  Json report;
  try {
    monitor.Stage("value_analysis.scalar_fold");
    auto folded = alg::ProfileCall("value_analysis.scalar_fold", [&] {
      sentryward::value_analysis::ScalarFoldOptions fold_options;
      fold_options.compact_output_families=true;
      fold_options.unit_lifting=lifting;
      fold_options.compared_outputs=compared;
      return sentryward::value_analysis::lowerScalarFold(values, eliminated,fold_options);
    });
    if (lifting) {
      if (folded.lifting.refused) {
        // The first-item rule: a class-mode fold failure that is provably the per-unit
        // run's first failure is reported as computed; any other refusal restarts.
        bool matches = false;
        if (folded.diagnostic) matches = FirstFailureMatchesPerUnit(values, eliminated, folded, *lifting);
        if (!matches) return RefuseLifting(values, *lifting, "lowering", LiftingStopOf(folded.diagnostic));
      }
      // The fold's own row maps (audited row widths) after the value passes' ones.
      if (RowModulusViolation(*lifting)) return RefuseLifting(values, *lifting, "lowering");
    }
    report = {{"status", folded.complete ? "ready" : "unsupported"},
              {"source_outputs", folded.source_outputs},
              {"construction_collections", folded.construction_collections},
              {"peak_construction_nodes", folded.peak_construction_nodes},
              {"shared_view_projection_hits", folded.shared_view_projection_hits}};
    if (folded.diagnostic) {
      const auto& diagnostic = *folded.diagnostic;
      report["diagnostic"] = {{"code", diagnostic.code}, {"message", diagnostic.message}};
      if (diagnostic.code.ends_with("_limit_exceeded")) {
        report["status"] = "analysis_budget_exceeded";
        report["resource"] = {{"origin","algorithm_budget"},{"kind",diagnostic.code}};
      } else if (diagnostic.code == "index_representation_overflow") {
        report["status"] = "representation_limit_exceeded";
        report["resource"] = {{"origin","representation"},{"kind","integer_range"}};
      } else if (diagnostic.code == "invalid_analysis_budget") report["status"] = "invalid_configuration";
      else if (diagnostic.code == "source_dependency_cycle") report["status"] = "internal_error";
      if (diagnostic.operation) {
        const auto& operation = *diagnostic.operation;
        report["diagnostic"]["value_node"] = operation.node;
        report["diagnostic"]["instance"] = operation.instance;
        const auto& origin = values.node(operation.node).instances().at(operation.instance).origin;
        if (origin.source_line) report["diagnostic"]["line"] = *origin.source_line;
        report["diagnostic"]["instruction"] = origin.source_text;
      }
    }
    if (!folded.complete) {
      report["partial_node_count"] = folded.graph.size();
      report["partial_output_count"] = folded.outputs.size();
      return report;
    }
    std::vector<sf::Ref> roots;
    std::vector<std::uint32_t> root_binders;
    roots.reserve(folded.outputs.size());
    for (const auto& output : folded.outputs) { roots.push_back(output.value);root_binders.push_back(0); }
    std::uint64_t logical_outputs=folded.outputs.size();
    for (const auto& family:folded.output_families) {
      roots.push_back(family.body);root_binders.push_back(family.shape.size());
      std::uint64_t count=1;
      for(auto extent:family.shape) {
        if(extent && count>UINT64_MAX/extent)throw std::overflow_error("output count overflow");
        count*=extent;
      }
      if(count>UINT64_MAX-logical_outputs)throw std::overflow_error("output count overflow");
      logical_outputs+=count;
    }
    const auto stats = sf::statistics(folded.graph, roots,0,root_binders);
    report["output_count"] = logical_outputs;
    report["output_root_count"] = roots.size();
    report["output_family_count"] = folded.output_families.size();
    report["output_families_merged"] = folded.output_families_merged;
    report["spatial_families_projected"] = folded.spatial_families_projected;
    report["spatial_outputs_projected"] = folded.spatial_outputs_projected;
    report["spatial_template_operations"] = folded.spatial_template_operations;
    report["statistics"] = {
        {"nodes", stats.nodes}, {"storage_nodes", folded.graph.size()},
        {"nodes_before_final_collection", folded.nodes_before_final_collection},
        {"nodes_reclaimed", folded.nodes_reclaimed},
        {"edges", stats.edges}, {"constants", stats.constants},
        {"parameters", stats.parameters}, {"initial_cells", stats.initial_cells},
        {"indices", stats.indices}, {"operations", stats.operations}, {"folds", stats.folds},
        {"unique_fold_bodies", stats.unique_fold_bodies},
        {"max_fold_nesting", stats.max_fold_nesting},
        {"range_descriptions", stats.range_descriptions},
        {"logical_iterations", stats.logical_iterations},
        {"logical_iterations_overflow", stats.logical_iterations_overflow}};

#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/fold_export.inc"
#endif
    if (options.reference_stdin) {
      artifact.emplace();
      artifact->graph = std::move(folded.graph);
      artifact->output_families = std::move(folded.output_families);
      for (const auto& output : folded.outputs)
        artifact->outputs.push_back({output.region.allocation, output.region.offset,
                                    output.region.bytes, output.value});
    }
  } catch (const std::bad_alloc& error) {
    artifact.reset();
    report = {{"status","allocation_failed"}, {"stage","scalar_expression_construction"},
        {"resource",{{"kind","memory"},{"origin","allocator"},{"cause","unknown"}}}, {"message",error.what()}};
  } catch (const std::exception& error) {
    artifact.reset();
    report = {{"status", "internal_error"}, {"stage","scalar_expression_construction"}, {"reason", "unexpected_exception"}, {"message", error.what()}};
  }
  return report;
}
// The bounds of tensor_properties are equivalence premises too, as input-contract ranges of
// the mapped objects on both sides (the same trusted premises the race check and memory
// elimination use; injectivity is not one). Records without a bound beyond their type's
// range, objects the mapping does not name on both sides, and objects/types the mapping
// already declares are skipped.
void AppendTensorPropertyContract(Json& mapping, const Json& properties) {
  if (!properties.is_array() || properties.empty() || !mapping.is_object()) return;
  auto& context = mapping["checker_context"];
  if (!context.is_object()) context = Json::object();
  auto& contract = context["input_contract"];
  if (!contract.is_array()) contract = Json::array();
  std::set<std::pair<std::string, std::string>> declared;
  for (const auto& entry : contract)
    if (entry.is_object() && entry.contains("object") && entry.contains("element_type"))
      declared.emplace(entry["object"].get<std::string>(), entry["element_type"].get<std::string>());
  const auto mapped = [&](const char* side, const std::string& identity) -> std::optional<std::string> {
    if (!mapping.contains(side) || !mapping[side].contains("allocations")) return std::nullopt;
    const auto& table = mapping[side]["allocations"];
    const auto found = table.find(identity);
    if (found == table.end() || !found->is_string()) return std::nullopt;
    return found->get<std::string>();
  };
  for (const auto& record : properties) {
    const auto object = record.at("object").get<std::string>(), type = record.at("element_type").get<std::string>();
    const auto bits = record.at("element_bits").get<unsigned>();
    const bool sign = record.at("signed").get<bool>();
    const auto lower = ptx::JsonInteger(record.at("lower_bound")), upper = ptx::JsonInteger(record.at("upper_bound"));
    const ptx::Integer type_lower = sign ? ptx::Integer(-(ptx::Integer(1) << (bits - 1))) : ptx::Integer(0);
    const ptx::Integer type_upper = (ptx::Integer(1) << (bits - unsigned(sign))) - 1;
    if (lower <= type_lower && upper >= type_upper) continue;
    const auto identity = sentryward::value_analysis::AllocationIdentity("global", object);
    const auto left = mapped("left_inputs", identity), right = mapped("right_inputs", identity);
    if (!left || !right || *left != *right || !declared.emplace(*left, type).second) continue;
    contract.push_back({{"object", *left}, {"element_type", type}, {"source", "tensor_properties"},
        {"region", "whole_object"}, {"min_inclusive", record.at("lower_bound")}, {"max_inclusive", record.at("upper_bound")}});
  }
  if (contract.empty()) context.erase("input_contract");
  if (context.empty()) mapping.erase("checker_context");
}
} // namespace
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
// One analysis attempt of the launch: the class-mode attempt, or with
// `unit_restart` the per-unit attempt after it (PerUnitRestart). Prints the
// report and returns the exit status; a class-mode attempt that restarts throws
// PerUnitRestart instead, after its state is released.
int RunAttempt(int argc, char **argv, CallInputs &inputs, const std::optional<PerUnitRestart> &unit_restart) {
  bool parsing_arguments = true;
  try {
    auto options = Arguments(argc, argv);
    parsing_arguments = false;
    alg::AnalysisScope attempt_scope;
    Json restart_record;
    if (unit_restart)
      restart_record = {{"reason", unit_restart->reason},
                        {"kept_synchronization_verdict", !unit_restart->verdict.empty()}};
    Monitor monitor(options, unit_restart ? std::optional(unit_restart->started) : std::nullopt,
                    std::move(restart_record));
    // A per-unit attempt that keeps the class-mode synchronization verdict reports it
    // from the start, also when the value side stops early.
    const bool kept_verdict = unit_restart && !unit_restart->verdict.empty();
    if (kept_verdict) monitor.MemoryVerified(unit_restart->verdict, unit_restart->verdict_elapsed_s);
    std::optional<rg::Document> reference;
    std::optional<sentryward::scalar_fold::Artifact> expression;
    Json equivalence_mapping, reference_error={{"status","missing_reference"},{"reason","reference_expression_not_supplied"}};
    // The bytes the binding compares: the scalar fold projects only those.
    std::map<std::string, sentryward::scalar_fold::ComparedOutput> compared_outputs;
    if(!options.reference_failure_json.empty()) {
      reference_error=Json::parse(options.reference_failure_json);
      if(!reference_error.is_object() || !reference_error.contains("result") ||
          !sentryward::results::IsResult(reference_error.value("result","")) ||
          reference_error.value("result","")=="verified" || reference_error.value("result","")=="race_free")
        throw std::invalid_argument("invalid reference failure report");
    }
    if (options.reference_stdin) {
      monitor.Stage("equivalence.reference_import");
      try {
        if (!std::isfinite(options.equivalence_timeout) || options.equivalence_timeout <= 0)
          throw std::invalid_argument("invalid equivalence timeout");
        std::istringstream in(inputs.ReferenceStdin());
        reference = rg::Document::load(in);
        const auto *mapping = inputs.File(options.equivalence_mapping);
        if (!mapping) throw std::invalid_argument("missing equivalence mapping");
        std::istringstream mapping_in(*mapping);
        mapping_in >> equivalence_mapping;
        compared_outputs = rg::comparedOutputs(equivalence_mapping);
      } catch (const std::exception& error) {
        reference.reset();
        compared_outputs.clear();
        reference_error = {{"status","unsupported"},{"reason","reference_import_failed"},{"message",error.what()}};
      }
    }
    alg::ProfileSession profiling(true);
    alg::ProfileScope total("pipeline.total");
    auto poll = [] { alg::ProfileSnapshot(); };
    Json metadata, input_metadata = Json::object();
    // Normalized tensor_properties records: trusted premises of the equivalence check too.
    Json tensor_property_inputs = Json::array();
    std::uint64_t records = 0;
    ptx::ThreadRecordStatistics record_compression;
    ptx::TemplateExecutionStatistics template_execution;
    std::uint64_t template_ctas=0;
    std::uint32_t ctas = 0;
    Json result;
    std::unique_ptr<sentryward::value_ir::Graph> values;
    std::unique_ptr<ptx::ValueIrExtractor> value_extractor;
    std::unique_ptr<ptx::BranchScopeCapture> branch_capture;
    std::unique_ptr<ptx::SemanticProgramBuilder> semantic_builder;
    std::optional<ptx::SemanticProgram> semantic_program;
    std::optional<sentryward::value_analysis::MemoryEliminationResult> eliminated;
    Json value_analysis = {{"status", "not_run"}, {"reason", "verification_not_completed"}};
    Json serialization_statistics;
    Json cache_provenance;
    bool replayed_correctness=false;
    // Cross-CTA class mode: the quotient tables; an attempt that cannot finish
    // exactly restarts per unit (PerUnitRestart). Value lifting (class
    // mode with values): the class tables. Whether units share their analysis
    // does not change the results; analysis statistics (program counts such as
    // threads, dependency-compression statistics) count only the units analysed.
    std::unique_ptr<alg::UnitQuotient> quotient;
    std::optional<sentryward::value_analysis::UnitLifting> unit_lifting;
    // Class mode with value lifting: the verified report of the class-mode synchronization
    // check and when it was reached, when a value_lifting restart may keep it (KeepsVerdict).
    std::string class_verdict;
    double class_verdict_elapsed_s = 0;
    const auto restart_per_unit = [&](const alg::UnitClassRestart &restart) {
      if (unit_restart)
        throw std::logic_error("internal error: CTA class bookkeeping (restart requested by the run without classes)");
      PerUnitRestart next{restart.reason, monitor.start};
      if (restart.reason == "value_lifting") {
        next.verdict = class_verdict;
        next.verdict_elapsed_s = class_verdict_elapsed_s;
      }
      throw next;
    };
#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/dot_options.inc"
#endif
    if ((!options.dump_value_ir.empty() || !options.dump_branch_scope.empty()) && !options.analyze_values) {
      values = std::make_unique<sentryward::value_ir::Graph>();
      if (!options.dump_branch_scope.empty()) branch_capture = std::make_unique<ptx::BranchScopeCapture>(*values);
      value_extractor = std::make_unique<ptx::ValueIrExtractor>(*values, false, false, branch_capture.get());
    }
    auto effective_assumptions = alg::DefaultPtxAssumptions();
    std::function<void(ptx::CheckedMemoryActions)> run_values;
    // Value lifting: a value stage of the class-mode quotient whose result
    // cannot stand for the members refuses lifting (RefuseLifting); the launch
    // re-runs per unit before the later (more expensive) stages, or, above the
    // re-run's budget, the value analysis ends with the budget refusal.
    const auto refuse_lifting = [&](const std::optional<LiftingStop> &stop = std::nullopt) {
      value_analysis = RefuseLifting(semantic_program->values, *unit_lifting, "value analysis", stop);
      value_analysis["reason"] = value_analysis["diagnostic"]["code"];
      value_analysis["message"] = value_analysis["diagnostic"]["message"];
      value_analysis["stage"] = "value_lifting";
    };
    const auto class_lifting = [&] { return unit_lifting.has_value(); };
    run_values = [&](ptx::CheckedMemoryActions actions) {
      monitor.Stage("value_analysis.prepare");
      if (!semantic_program->valuesComplete()) {
        const auto& diagnostic = semantic_program->value_diagnostics.front();
        value_analysis = {{"status", "unsupported"}, {"reason", diagnostic.code},
            {"message", diagnostic.message}, {"stage", "value_lowering"},
            {"line", diagnostic.origin.source_line.value_or(0)}};
        return;
      }
      if (!actions.complete()) {
        const auto& diagnostic = actions.diagnostics.front();
        // A binding diagnostic (it names its occurrence, or is the domain
        // link's stop) heading a list in CTA order is the per-unit run's
        // first; any other first diagnostic of the quotient need not be.
        if (class_lifting() && !((diagnostic.occurrence || diagnostic.first_group) && quotient &&
                                 BindingEvidence(*semantic_program, *quotient).launch_order))
          return refuse_lifting(LiftingStop{diagnostic.code, diagnostic.message});
        value_analysis = {{"status", "unsupported"}, {"reason", diagnostic.code},
            {"message", diagnostic.message}, {"stage", "memory_effect_binding"}};
        if (diagnostic.occurrence) value_analysis["instruction_index"] = diagnostic.occurrence->instruction;
        if (diagnostic.value) {
          const auto& origin = semantic_program->values.node(diagnostic.value->node).instances().at(diagnostic.value->instance).origin;
          if (origin.source_line) value_analysis["line"] = *origin.source_line;
          value_analysis["instruction"] = origin.source_text;
        }
        return;
      }
      const auto sites_before = sentryward::value_analysis::UnitSiteCounts();
      auto pipeline = sentryward::value_analysis::runValuePasses(
          semantic_program->values, actions.actions, actions.contract, [&](std::string_view pass) {
            monitor.Stage("value_analysis." + std::string(pass));
          });
      if (class_lifting()) {
        // A BlockIndex read numerically by the value passes baked the
        // representative's coordinate into the values.
        namespace va = sentryward::value_analysis;
        const auto sites = va::UnitSiteCounts();
        const auto numeric = [&](va::UnitSite site) {
          return sites[std::size_t(site)] - sites_before[std::size_t(site)];
        };
        const auto read = numeric(va::UnitSite::ValueSpecialization) + numeric(va::UnitSite::ValueTemplateLeaf);
        if (read) return refuse_lifting();
        // A memory-elimination failure with the order-free identity is the
        // per-unit run's. Members fail on their addresses at the
        // representative's positions (admission compared their slot shapes,
        // and they write the same allocations), and the static key names no
        // unit. Any other failure restarts.
        const auto &stop = pipeline.memory.diagnostic;
        if (!pipeline.memory.complete && !(stop && stop->order_free)) return refuse_lifting(LiftingStopOf(stop));
        if (RowModulusViolation(*unit_lifting)) return refuse_lifting();
      }
      const auto packing = pipeline.packing, exp2 = pipeline.ideal_exp2;
      eliminated = std::move(pipeline.memory);
      value_analysis = {{"status", eliminated->complete ? "expressions_ready" : "unsupported"},
          {"passes", Json::array({
              {{"name", "packing"}, {"semantics", "exact_fixed_width"},
               {"rewritten_nodes", packing.nodes}, {"rewritten_instances", packing.instances}},
              {{"name", "software_exp2"}, {"semantics", "ideal_exp2"},
               {"rewritten_nodes", exp2.nodes}, {"rewritten_instances", exp2.instances},
               {"abstraction", "matched approximation and clamp replaced by ideal Exp2"}},
              {{"name", "type_edges"}, {"removed_nodes", pipeline.type_edges.nodes},
               {"converted_instances", pipeline.type_edges.instances}},
              {{"name", "memory_elimination"}},
              {{"name", "view_normalization"}, {"identities", pipeline.views.identities},
               {"elements", pipeline.views.elements}, {"slices", pipeline.views.slices},
               {"layouts", pipeline.views.layouts}, {"segmented", pipeline.views.segmented},
               {"original_pieces", pipeline.views.original_pieces},
               {"mapping_blocks", pipeline.views.mapping_blocks},
               {"reused_initial_reads", pipeline.views.reused_initial_reads},
               {"reused_views", pipeline.views.reused_views},
               {"stored_mapping_blocks", pipeline.views.stored_mapping_blocks},
               {"typed_output_regions", pipeline.views.typed_output_regions},
               {"typed_output_pieces", pipeline.views.typed_output_pieces}}})},
          {"processed_memory_actions", eliminated->processed_actions},
          {"processed_memory_families", eliminated->processed_families},
          {"memory_action_storage", {{"logical_actions",actions.actions.size()},
              {"stored_runs",actions.actions.storedRunCount()}}},
          {"duplicate_write_value_checks", eliminated->duplicate_write_value_checks},
          {"read_substitutions", eliminated->reads.size()+eliminated->shared_read_instances},
          {"stored_read_substitutions", eliminated->reads.size()},
          {"shared_read_families", eliminated->shared_read_families.size()},
          {"shared_read_instances", eliminated->shared_read_instances},
          {"initial_read_families", eliminated->initial_read_families.size()},
          {"initial_memory_regions", eliminated->initial_memory.size()},
          {"output_regions", eliminated->outputCount()},
          {"output_region_families", eliminated->output_families.size()},
          {"equivalence_check", "not_run"}, {"equivalence_reason", "reference_expression_not_supplied"}};
      if (options.value_thread_range && eliminated->complete) {
        using RegionKey = std::pair<std::string, bool>;
        std::map<RegionKey, std::vector<std::pair<std::uint64_t,std::uint64_t>>> regions;
        for (const auto& action : sentryward::value_analysis::MemoryActionRange(actions.actions)) {
          if (!action.region) throw std::runtime_error("missing partition footprint");
          const auto& op = semantic_program->values.node(action.operation.node).operation();
          const bool write = dynamic_cast<const sentryward::value_ir::StoreOp*>(&op) != nullptr;
          if (!write && !dynamic_cast<const sentryward::value_ir::LoadOp*>(&op))
            throw std::runtime_error("unknown partition memory effect");
          const auto& r = *action.region;
          if (r.bytes) regions[{r.allocation,write}].push_back({r.offset,r.offset+r.bytes});
        }
        value_analysis["partition_footprints"] = Json::array();
        for (auto& [key,ranges] : regions) {
          std::sort(ranges.begin(),ranges.end());
          std::vector<std::pair<std::uint64_t,std::uint64_t>> merged;
          for (const auto& r : ranges) {
            if (!merged.empty() && r.first <= merged.back().second)
              merged.back().second = std::max(merged.back().second,r.second);
            else merged.push_back(r);
          }
          for (const auto& r : merged)
            value_analysis["partition_footprints"].push_back({{"allocation",key.first},
                {"write",key.second},{"begin",r.first},{"end",r.second}});
        }
      }
      if (eliminated->diagnostic) {
        value_analysis["reason"] = eliminated->diagnostic->code;
        value_analysis["message"] = eliminated->diagnostic->message;
        if (const auto& operation = eliminated->diagnostic->operation) {
          const auto& origin = semantic_program->values.node(operation->node).instances().at(operation->instance).origin;
          if (origin.source_line) value_analysis["line"] = *origin.source_line;
          value_analysis["instruction"] = origin.source_text;
        }
      }
    };
    try {
      monitor.Stage("ptx_input.read");
      auto text = alg::ProfileCall("ptx_input.read",
                                   [&] { return inputs.Read(options.ptx); });

#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/checkpoint_identity.inc"
#endif
      monitor.Stage("ptx_parser");
      auto module = alg::ProfileCall("ptx_parser", [&] {
        return ptx::Parse(std::move(text), options.ptx);
      });
      monitor.Stage("input_configuration");
      auto config = alg::ProfileCall("input_configuration", [&] {
        auto supplied = ptx::LoadKernelInput(options.config, inputs.File(options.config));
        auto supplied_flags = effective_assumptions;
        const auto overrides = supplied.fields.value("assumptions", Json::object());
        alg::OverridePtxAssumptions(supplied_flags, overrides.get<alg::PtxAssumptions>());
        alg::OverridePtxAssumptions(supplied_flags, options.assumptions);
        supplied.fields["assumptions"] = supplied_flags;
        effective_assumptions = supplied_flags;
        ptx::InputOptions input_options;
        input_options.max_steps_per_thread = options.max_steps_per_thread;
        input_options.compress_thread_records = options.compress_thread_records;
        auto normalized = ptx::NormalizeKernelInput(supplied, module, input_options);
        const auto &entry = *std::find_if(module.functions.begin(), module.functions.end(),
            [&](const auto &f) { return f.name == normalized.entry; });
        auto inferred = ptx::InferPointerFields(module, entry, normalized);
        if (!inferred.empty()) normalized = ptx::NormalizeKernelInput(supplied, module, input_options, inferred);
        return normalized;
      });

#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/value_resource_bindings.inc"
#endif
      Json frontend_assumptions = {
          {"trap_only_equality_assertions_hold", config.fixed_input.value("assume_trap_assertions", false)},
          {"assertion_failure_forms", {"trap", "__assertfail"}},
          {"fixed_election", config.elect_lane.has_value()},
          {"election_policy", config.elect_lane ? Json{{"strategy","preferred_member"},
              {"preferred_lane",*config.elect_lane}, {"fallback","lowest_member"}} : Json(nullptr)},
          {"noalias_unbounded", false},
          {"pointer_contract", config.fixed_input.value("kernel_pointer_contract",Json::object())},
          {"isolated_launch", config.fixed_input.value("launch_contract",Json::object()).value("external_grids","") == "no_overlap"},
          {"clc_failure", config.fixed_input.contains("cluster_launch_cancellation") &&
              config.fixed_input["cluster_launch_cancellation"] == false},
          {"assertion_uses",0}, {"election_uses",0}, {"clc_uses",0},
          {"representative_branch_uses",0}, {"bounded_loop_uses",0}, {"atomic_uses",0},
          {"data_branch_decisions",Json::array()},
          {"opaque_compute_regions",Json::array()}};
      alg::OverridePtxAssumptions(effective_assumptions,
          config.raw.value("semantic_assumptions", Json::object()).get<alg::PtxAssumptions>());
      alg::OverridePtxAssumptions(effective_assumptions, options.assumptions);
      config.raw["semantic_assumptions"] = effective_assumptions;
      frontend_assumptions["fixed_election"] = effective_assumptions.at("representative_election");
      frontend_assumptions["clc_failure"] = effective_assumptions.at("clc_always_fails");
      input_metadata = {{"frontend_assumptions", frontend_assumptions},
          {"input_objects",config.raw.value("input_objects",Json::array())},
          {"user_branch_decisions",config.raw.value("user_branch_decisions",Json::array())}};
      if (config.fixed_input.contains("shape_input_contract"))
        input_metadata["frontend_assumptions"]["shape_input_contract"] =
            config.fixed_input["shape_input_contract"];
      if (config.fixed_input.contains("pointer_array_contract"))
        input_metadata["frontend_assumptions"]["pointer_array_contract"] =
            config.fixed_input["pointer_array_contract"];
      if (config.fixed_input.contains("tensor_property_contract")) {
        input_metadata["frontend_assumptions"]["tensor_properties"] = config.fixed_input["tensor_property_contract"];
        input_metadata["frontend_assumptions"]["tensor_property_decisions"] = Json::array();
      }
      const ptx::Function *function = nullptr;
      const bool explicit_prefix = config.entry.starts_with("prefix:");
      const auto selector =
          explicit_prefix ? config.entry.substr(7) : config.entry;
      if (!explicit_prefix)
        for (const auto &f : module.functions)
          if (f.name == selector) {
            function = &f;
            break;
          }
      if (!function)
        for (const auto &f : module.functions)
          if (f.name.starts_with(selector)) {
            if (function)
              throw std::invalid_argument(
                  "entry selector matches multiple functions");
            function = &f;
          }
      if (!function)
        throw std::invalid_argument("configured entry missing from PTX");
      monitor.Stage("launch_binding");
      config = alg::ProfileCall("launch_binding", [&] {
        return ptx::BindLaunchObjects(module, *function, config);
      });

#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/checkpoint_provenance.inc"
#else
      {
#endif
      auto [grid, block] = ptx::LaunchDimensions(config);
      (void)block;
      if (options.trusted_reference) {
        const std::set<std::string> forbidden = {"bar", "barrier", "mbarrier", "shfl",
            "vote", "match", "elect", "activemask", "redux", "mma", "wmma", "wgmma",
            "tcgen05", "cp", "clusterlaunchcontrol", "atom", "red", "call"};
        for (const auto& inst : function->instructions)
          if (forbidden.contains(inst.opcode) || inst.semantics.asynchronous)
            throw ptx::ExecutionError("unsupported_reference_order", inst.line,
                "Trusted-reference mode currently requires synchronous, noncollective instructions");
      }
      if (options.value_thread_range) {
        const auto total = std::uint64_t(grid[0])*grid[1]*grid[2]*config.Threads();
        if (options.value_thread_range->second > total ||
            (config.cluster_dim && *config.cluster_dim != ptx::Coordinates{1,1,1}))
          throw std::invalid_argument("thread subset exceeds launch or uses a cluster");
        const std::set<std::string> forbidden = {"bar", "barrier", "mbarrier", "shfl",
            "vote", "match", "elect", "activemask", "redux", "mma", "wmma", "wgmma",
            "tcgen05", "cp", "clusterlaunchcontrol", "atom", "red", "call"};
        for (const auto& inst : function->instructions)
          if (forbidden.contains(inst.opcode) || inst.semantics.asynchronous)
            throw std::invalid_argument("thread subset requires independent synchronous instructions: "+inst.opcode);
      }
      if (ptx::Integer(grid[0]) * grid[1] * grid[2] * config.Threads() >
          UINT32_MAX)
        throw std::invalid_argument("launch exceeds native actor ID domain");
      try {
      std::unique_ptr<ptx::ModelLowerer> lowerer;
      // Cross-CTA: launch units, the unit-independence precondition and, in
      // class mode, the unit-major loop below. With class mode off (a gate or
      // the per-unit re-run) nothing below observes the run.
      ptx::UnitGates unit_gates;
      unit_gates.value_analysis = options.analyze_values;
      unit_gates.scalar_fold = options.analyze_scalar_fold;
      unit_gates.value_thread_range = options.value_thread_range.has_value();
      unit_gates.trusted_reference = options.trusted_reference;
      unit_gates.value_dump = values != nullptr;
      unit_gates.branch_scope = !options.dump_branch_scope.empty();
      unit_gates.scalar_fold_dump = !options.dump_scalar_fold.empty();
      unit_gates.exhaustive_search = options.exhaustive_memory_search;
      unit_gates.continue_search = options.continue_memory_search;
#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
      unit_gates.development_exports = true;
#endif
      auto unit_plan = ptx::PlanUnits(config, *function, unit_gates);
      if (unit_restart) unit_plan.class_mode = false; // the per-unit attempt
      // Class mode's log budget, min(2 GiB, 10% of --rss-limit-mib).
      const std::uint64_t class_budget = std::min<std::uint64_t>(
          std::uint64_t(2) << 30, options.rss_mib ? *options.rss_mib * 1048576 / 10 : UINT64_MAX);
      if (options.analyze_values || (!options.extract && !options.semantic && !values)) {
        semantic_builder = std::make_unique<ptx::SemanticProgramBuilder>(module, *function, poll,
            ptx::SemanticProgramOptions{
                .collect_synchronization_groups = options.analyze_values && !options.trusted_reference,
                .compact_memory_linking = true,
                .link_numerical_memory = options.analyze_values,
                .defer_address_serialization = !options.trusted_reference,
                .capture_branch_scope = !options.dump_branch_scope.empty(),
                .atomics = effective_assumptions.at("order_independent_atomics"),
                .values_on_demand = !options.analyze_values,
                .threads = options.threads});
      } else if (!options.semantic && !values)
        lowerer = alg::ProfileCall("frontend.static_lowering_prepare", [&] {
          return std::make_unique<ptx::ModelLowerer>(module, *function, poll, false,
              effective_assumptions.at("order_independent_atomics"), options.threads);
        });
      // The classes of this launch (class mode only).
      std::unique_ptr<ptx::UnitClasses> classes;
      if (unit_plan.class_mode) {
        classes = std::make_unique<ptx::UnitClasses>(unit_plan, class_budget);
        // Lowering weights by the class tables, complete before its Finish.
        quotient = std::make_unique<alg::UnitQuotient>();
        if (semantic_builder) semantic_builder->TrackUnits(quotient.get());
        if (lowerer) lowerer->TrackUnits(quotient.get());
      }
      {
        alg::ProfileScope frontend("frontend.dynamic_pipeline");
        const auto purpose = branch_capture ? ptx::SpecializationPurpose::BranchScopeCapture
                            : !options.dump_branch_scope.empty() ? ptx::SpecializationPurpose::VerificationWithBranchScope
                            : values ? ptx::SpecializationPurpose::ValueExtraction
                                    : ptx::SpecializationPurpose::Verification;
        std::set<ptx::Coordinates> emitted;
        std::map<std::uint64_t, ptx::DataBranchDecision> data_branch_decisions;
        std::map<std::uint64_t, std::uint64_t> opaque_regions;
        std::map<std::string, std::set<std::uint64_t>> property_lines;
        std::set<std::uint64_t> static_choice_lines;
        // The driver's accounting of one executed CTA (every CTA, members included).
        const auto account = [&](const ptx::CtaFacts &cta_facts) {
              const ptx::FactSource::ConsumerScope driver(cta_facts, {ptx::Consumer::Driver, 0});
              const auto &execution=cta_facts.TemplateStatistics();
              template_ctas+=execution.used;
              template_execution.steps+=execution.steps;
              template_execution.logical_steps+=execution.logical_steps;
              template_execution.splits+=execution.splits;
              template_execution.value_nodes+=execution.value_nodes;
              template_execution.record_nodes+=execution.record_nodes;
              template_execution.record_bindings+=execution.record_bindings;
              template_execution.coordinate_bindings+=execution.coordinate_bindings;
              template_execution.path_entries+=execution.path_entries;
              template_execution.thread_memberships+=execution.thread_memberships;
              alg::ReportIrSize("frontend.execution", {{"completed_ctas",template_ctas},
                  {"logical_steps",template_execution.logical_steps},
                  {"record_nodes",template_execution.record_nodes},
                  {"value_nodes",template_execution.value_nodes},
                  {"record_bindings",template_execution.record_bindings},
                  {"coordinate_bindings",template_execution.coordinate_bindings},
                  {"path_entries",template_execution.path_entries}});
              const auto &storage = cta_facts.RecordCompression();
              record_compression.record_count += storage.record_count;
              record_compression.template_count += storage.template_count;
              record_compression.unique_binding_count += storage.unique_binding_count;
              record_compression.estimated_owned_bytes += storage.estimated_owned_bytes;
              for (std::size_t index = 0; index < cta_facts.LaneCount(); ++index) {
                const auto lane = cta_facts.Lane(index);
                const auto aggregates = cta_facts.Aggregates(lane);
                records += cta_facts.Size(lane);
                auto &facts = input_metadata["frontend_assumptions"];
                facts["assertion_uses"] = facts["assertion_uses"].get<std::uint64_t>() + aggregates.assumed_assertions;
                for (const auto &[line, count] : aggregates.opaque_regions) opaque_regions[line] += count;
                for (const auto &[line, decision] : aggregates.data_branch_decisions) {
                  auto &summary = data_branch_decisions[line];
                  summary.assumption = decision.assumption;
                  summary.rule = decision.rule;
                  summary.taken += decision.taken;
                  summary.not_taken += decision.not_taken;
                  const auto counter = decision.assumption == "representative_branches" ? "representative_branch_uses" : "bounded_loop_uses";
                  facts[counter] = facts[counter].get<std::uint64_t>() +
                      decision.taken + decision.not_taken;
                }
              }
              static_choice_lines.insert(cta_facts.StaticBranchChoiceLines().begin(),
                                         cta_facts.StaticBranchChoiceLines().end());
              const auto static_counts = cta_facts.StaticCounts();
              for (std::size_t id=0;id<static_counts.size();++id) {
                auto &facts=input_metadata["frontend_assumptions"];
                const auto &op=function->instructions[id].opcode;
                const auto count=static_counts[id];
                if (op=="elect") facts["election_uses"]=facts["election_uses"].get<std::uint64_t>()+count;
                if (op=="clusterlaunchcontrol") facts["clc_uses"]=facts["clc_uses"].get<std::uint64_t>()+count;
                if (op=="atom" || op=="red") facts["atomic_uses"]=facts["atomic_uses"].get<std::uint64_t>()+count;
              }
              // Unit independence forbids an executed global atomic: restart if this CTA executes one.
              if (unit_plan.class_mode)
                for (const auto id : unit_plan.precondition.atomic_sites)
                  if (id < static_counts.size() && static_counts[id] != 0)
                    throw alg::UnitClassRestart{"executed_global_atomic"};
        };
        // A CTA's tensor_properties decisions, read once every consumer is done with it: template
        // values evaluated after its run (lowering, class admission) are decided by bounds too.
        const auto account_properties = [&](const ptx::CtaFacts &cta_facts) {
          for (const auto &[object, sites] : cta_facts.PropertyDecisionsSnapshot())
            for (const auto id : sites) property_lines[object].insert(function->instructions[id].line);
        };
        const auto linear_id = [&](const ptx::Coordinates &c) {
          return static_cast<std::uint32_t>((std::uint64_t(c[2]) * grid[1] + c[1]) * grid[0] + c[0]);
        };
        if (unit_plan.class_mode) {
          // Class loop: units in unit order, every CTA executed and accounted for;
          // a unit joins a class before any of its CTAs is consumed, and only
          // the founding (representative) units are consumed, in rank order.
          std::uint32_t consumed = 0;
          // One unit's execution: its ranks' programs, their facts and structure keys
          // (pure functions of each CTA's own execution). Facts point into the
          // programs, so they are destroyed first.
          struct ExecutedUnit {
            std::vector<ptx::ExecutionProgram> programs;
            std::vector<std::unique_ptr<ptx::UnitFrame>> frames;
            std::vector<std::unique_ptr<ptx::CtaFacts>> facts;
            std::vector<ptx::StructureKey> keys;
            // Admission against the classes founded before this unit ran (look-ahead).
            ptx::UnitClasses::PreAdmission pre;
          };
          const auto execute = [&](std::size_t index, const std::function<void()> &unit_poll) {
            const auto &unit = unit_plan.units[index];
            const auto size = unit.ctas.size();
            auto result = std::make_unique<ExecutedUnit>();
            auto &programs = result->programs;
            programs.resize(size);
            for (std::size_t rank = 0; rank < size; ++rank) {
              const auto &coordinates = unit.ctas[rank];
              try {
                programs[rank] = alg::ProfileCall("ptx_execution", [&] {
                  return ptx::SpecializeFunction(module, *function, ptx::ConfigForCta(config, coordinates), unit_poll,
                                                 purpose);
                });
              } catch (const ptx::ExecutionError &error) {
                if (!error.shared_control) throw;
                // Shared control (as without classes, but per unit): run the whole
                // cluster; ranks executed individually before keep their programs.
                std::vector<ptx::FrontendConfig> configs;
                for (const auto &member : unit.ctas) configs.push_back(ptx::ConfigForCta(config, member));
                auto cluster = ptx::SpecializeCluster(module, *function, configs, unit_poll, purpose);
                for (std::size_t r = rank; r < size; ++r) programs[r] = std::move(cluster[r]);
                break;
              }
            }
            for (std::size_t rank = 0; rank < size; ++rank) {
              auto &program = programs[rank];
              result->frames.push_back(std::make_unique<ptx::UnitFrame>(unit.ctas[rank], program.config.cluster_dim));
              result->facts.push_back(
                  std::make_unique<ptx::CtaFacts>(program, *result->frames.back(), linear_id(unit.ctas[rank])));
              result->keys.push_back(alg::ProfileCall("class.structure_key",
                                                      [&] { return ptx::ComputeStructureKey(*result->facts.back()); }));
            }
            std::vector<const ptx::CtaFacts *> view;
            for (const auto &facts : result->facts) view.push_back(facts.get());
            result->pre = alg::ProfileCall("class.pre_admit", [&] { return classes->PreAdmit(view, result->keys); });
            // The replay's reads filled the template value caches; drop them before the
            // unit waits in the window (a later read, e.g. a founder's lowering, recomputes).
            for (const auto &facts : result->facts) facts->ReleaseValues();
            return result;
          };
          // Look-ahead: up to --sync-threads workers execute the next units while
          // this loop consumes in unit order; a unit's execution error is rethrown here,
          // at that unit, and leaving the loop (a restart included) joins the workers.
          // The first unit runs inline: the bytes it allocates bound the workers (each
          // keeps its own allocator arena at about one execution) and the executed units
          // in flight to LookAheadBudget (a quarter of the memory limit).
          const std::size_t workers = options.threads > 1 ? options.threads : 0;
          std::unique_ptr<ExecutedUnit> first;
          std::size_t window = workers;
          if (workers && !unit_plan.units.empty()) {
            monitor.Stage("ptx_execution.cta" + std::to_string(ctas));
            const auto before = AllocatedBytes();
            first = execute(0, poll);
            const auto after = AllocatedBytes();
            if (before && after && *after > *before)
              window = static_cast<std::size_t>(std::clamp<std::uint64_t>(
                  LookAheadBudget(options.rss_mib) / (*after - *before), 1, workers));
          }
          const std::size_t skip = first ? 1 : 0;
          [[maybe_unused]] std::size_t released_units = 0;
          std::optional<ptx::LookAhead<std::unique_ptr<ExecutedUnit>>> executions;
          executions.emplace(unit_plan.units.size() - skip, window, window,
              [&](std::size_t index, const std::function<void()> &unit_poll) { return execute(index + skip, unit_poll); },
              poll);
          for (const auto &unit : unit_plan.units) {
            const auto size = unit.ctas.size();
            monitor.Stage("ptx_execution.cta" + std::to_string(ctas));
            const auto executed_unit = first ? std::move(first) : executions->Take();
            auto &programs = executed_unit->programs;
            auto &unit_facts = executed_unit->facts;
            std::vector<const ptx::CtaFacts *> view;
            for (std::size_t rank = 0; rank < size; ++rank) {
              view.push_back(unit_facts[rank].get());
              account(*unit_facts[rank]);
              ++ctas;
            }
            const auto release = [&](std::size_t rank) {
              account_properties(*unit_facts[rank]);
              unit_facts[rank].reset();
              auto &program = programs[rank];
              alg::ProfileCall("frontend.release_execution",
                               [&] { program = ptx::ExecutionProgram{}; });
#if defined(__GLIBC__)
              // The executions were allocated in the workers' arenas, which keep the
              // freed pages: return them every 8 released units.
              if (workers && rank + 1 == size && ++released_units % 8 == 0) malloc_trim(0);
#endif
            };
            if (classes->Admit(unit.index, view, std::move(executed_unit->keys), std::move(executed_unit->pre))) {
              for (std::size_t rank = 0; rank < size; ++rank) release(rank);
              continue;
            }
            for (std::size_t rank = 0; rank < size; ++rank) {
              auto &cta_facts = *unit_facts[rank];
              ptx::QuerySink *const sink = classes->Record(unit.index, static_cast<std::uint32_t>(rank), cta_facts);
              {
                // The recorder is attached while the consumers read this CTA.
                struct Detach {
                  ptx::CtaFacts &facts;
                  bool attached;
                  ~Detach() { if (attached) facts.Attach(nullptr); }
                } detach{cta_facts, sink != nullptr};
                if (sink) cta_facts.Attach(sink);
                if (lowerer) {
                  monitor.Stage("source_ir.cta" + std::to_string(consumed));
                  lowerer->AddCta(cta_facts);
                }
                if (semantic_builder) {
                  monitor.Stage("semantic_ir.cta" + std::to_string(consumed));
                  semantic_builder->AddCta(cta_facts);
                }
                ++consumed;
              }
              classes->Recorded(unit.index, static_cast<std::uint32_t>(rank));
              release(rank);
            }
            // The units executed while this one was consumed were pre-admitted without its class:
            // try it on them now, in parallel (Admit would replay it here, one unit at a time).
            if (workers)
              executions->RefreshReady(workers, [&](std::unique_ptr<ExecutedUnit> &waiting) {
                std::vector<const ptx::CtaFacts *> waiting_view;
                for (const auto &facts : waiting->facts) waiting_view.push_back(facts.get());
                waiting->pre = classes->PreAdmit(waiting_view, waiting->keys, std::move(waiting->pre));
                for (const auto &facts : waiting->facts) facts->ReleaseValues();
              });
          }
          executions.reset();
#if defined(__GLIBC__)
          // The workers' arenas keep the pages of the programs the loop freed;
          // return them before the analysis stages (gigabytes on large launches).
          if (workers) malloc_trim(0);
#endif
        } else
        for (std::uint32_t z = 0; z < grid[2]; ++z)
          for (std::uint32_t y = 0; y < grid[1]; ++y)
            for (std::uint32_t x = 0; x < grid[0]; ++x) {
              ptx::Coordinates coordinates{x, y, z};
              if (emitted.contains(coordinates)) continue;
              auto local = ptx::ConfigForCta(config, coordinates);
              if (options.value_thread_range) {
                const auto base = (std::uint64_t(z)*grid[1]*grid[0]+std::uint64_t(y)*grid[0]+x)*config.Threads();
                const auto begin = std::max(base, options.value_thread_range->first);
                const auto end = std::min(base+config.Threads(), options.value_thread_range->second);
                if (begin >= end) continue;
                local.analysis_thread_range = {{static_cast<std::uint32_t>(begin-base),
                                               static_cast<std::uint32_t>(end-base)}};
              }
              monitor.Stage("ptx_execution.cta" + std::to_string(ctas));
              std::vector<std::pair<ptx::Coordinates,ptx::ExecutionProgram>> programs;
              try {
                programs.emplace_back(coordinates,alg::ProfileCall("ptx_execution", [&] {
                  return ptx::SpecializeFunction(module, *function, local, poll, purpose);
                }));
              } catch (const ptx::ExecutionError &error) {
                if (!error.shared_control) throw;
                const auto shape=config.cluster_dim.value_or(ptx::Coordinates{1,1,1});
                const ptx::Coordinates base{x/shape[0]*shape[0],y/shape[1]*shape[1],z/shape[2]*shape[2]};
                std::vector<ptx::FrontendConfig> configs;
                std::vector<ptx::Coordinates> members;
                for (unsigned dz=0;dz<shape[2];++dz)
                  for (unsigned dy=0;dy<shape[1];++dy)
                    for (unsigned dx=0;dx<shape[0];++dx) {
                      ptx::Coordinates member{base[0]+dx,base[1]+dy,base[2]+dz};
                      for (unsigned axis=0;axis<3;++axis) if (member[axis]>=grid[axis])
                        throw ptx::ExecutionError("shared_control_cluster",error.line,"incomplete CTA cluster in fixed launch");
                      members.push_back(member); configs.push_back(ptx::ConfigForCta(config,member));
                    }
                auto cluster=ptx::SpecializeCluster(module,*function,configs,poll,purpose);
                for (std::size_t rank=0;rank<cluster.size();++rank)
                  if (!emitted.contains(members[rank])) programs.emplace_back(members[rank],std::move(cluster[rank]));
              }
              for (auto &[member,program] : programs) {
              {
              // The only consumer view of this program; destroyed before the program is released.
              const ptx::UnitFrame frame(member, program.config.cluster_dim);
              ptx::CtaFacts cta_facts(program, frame, linear_id(member));
              account(cta_facts);
              if (lowerer) {
                monitor.Stage("source_ir.cta" + std::to_string(ctas));
                lowerer->AddCta(cta_facts);
              }
              if (value_extractor) {
                monitor.Stage("value_ir.cta" + std::to_string(ctas));
                value_extractor->AddCta(cta_facts);
              }
              if (semantic_builder) {
                monitor.Stage("semantic_ir.cta" + std::to_string(ctas));
                semantic_builder->AddCta(cta_facts);
              }
              account_properties(cta_facts);
              }
              alg::ProfileCall("frontend.release_execution",
                               [&] { program = ptx::ExecutionProgram{}; });
              ++ctas;
              emitted.insert(member);
              }
            }
        for (const auto &[line, count] : opaque_regions)
          input_metadata["frontend_assumptions"]["opaque_compute_regions"].push_back({{"line", line}, {"lanes", count}});
        for (const auto &[object, lines] : property_lines)
          input_metadata["frontend_assumptions"]["tensor_property_decisions"].push_back(
              {{"object", object}, {"lines", lines}});
        // Choices that a statically known predicate overrode on some visit (listed only when any).
        if (!static_choice_lines.empty())
          input_metadata["frontend_assumptions"]["branch_choices_with_static_visits"] = static_choice_lines;
        auto &decisions = input_metadata["frontend_assumptions"]["data_branch_decisions"];
        for (const auto &[line, decision] : data_branch_decisions)
          decisions.push_back({{"line", line}, {"assumption", decision.assumption}, {"rule", decision.rule},
                               {"taken", decision.taken}, {"not_taken", decision.not_taken}});
        if (unit_plan.class_mode) {
          // The quotient tables (lowering weights by them at Finish); the tensormap
          // locality rule needs every unit's slots, so it is decided only now. An
          // input map overwrite is found again by the per-unit run.
          *quotient = classes->Quotient();
          if (!classes->TensormapLocality()) throw alg::UnitClassRestart{"tensormap_locality"};
          // Values are lifted only when some class has members (fewer
          // classes than units); otherwise every unit was analysed (the per-unit run's fold).
          if (unit_plan.lift_values && quotient->Shared())
            unit_lifting = ClassLifting(*quotient);
        }
        classes.reset();
      }
      if (values) {
        result = {{"status", value_extractor->complete() ? "value_view_complete" : "value_view_incomplete"},
                  {"ctas", ctas}, {"executed_records", records},
                  {"verification_performed", false},
                  {"memory_analysis_complete", false},
                  {"diagnostics", Json::array()}};
        for (const auto &diagnostic : value_extractor->diagnostics())
          result["diagnostics"].push_back({{"code", diagnostic.code}, {"message", diagnostic.message},
              {"location", {{"file", diagnostic.origin.source_file.value_or(options.ptx)},
                            {"line", diagnostic.origin.source_line.value_or(0)}}},
              {"thread", diagnostic.origin.thread.value_or(0)},
              {"instruction_index", diagnostic.origin.instruction_index.value_or(0)},
              {"occurrence", diagnostic.origin.occurrence.value_or(0)}});
        result["frontend"] = input_metadata;
      } else if (options.semantic) {
        result = {{"status", "semantic_complete"},
                  {"ctas", ctas},
                  {"executed_records", records},
                  {"static_instructions", function->instructions.size()},
                  {"verification_performed", false},
                  {"memory_analysis_complete", false}};
        result["frontend"] = input_metadata;
      } else {
        monitor.Stage("source_ir.finalize");

#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/branch_failure_export.inc"
#endif
        if (semantic_builder) semantic_program.emplace(semantic_builder->Finish(
            [&](std::string_view stage) { monitor.Stage(std::string(stage)); }));
        // Binding diagnostics are weighted per CTA (BindingEvidence); one that
        // names no CTA, or a list whose order cannot be expanded in launch
        // order, cannot stand for the members: per-unit run.
        if (semantic_program && quotient && quotient->Shared() &&
            !BindingEvidence(*semantic_program, *quotient).expandable)
          throw alg::UnitClassRestart{"binding_evidence"};
        auto lowered = semantic_program ? std::move(semantic_program->lowering) : lowerer->Finish();
        // The tensormap input-map overlap, on every path: class mode restarted on
        // it, so this is the per-unit program's (every CTA lowered).
        if (ptx::TensormapInputOverlap(lowered.tensormap_destinations, unit_plan.input_maps))
          lowered.artifact.verification_limitations.push_back({"tensormap_input_overwrite",
              "A tensormap.cp_fenceproxy destination overlaps an input tensor map's location; descriptor "
              "fetches are not modeled as reads."});
        ptx::ValidateConstantInputs(config, lowered.artifact);
        alg::ProfileCall("frontend.release_builder", [&] { lowerer.reset(); });
        alg::ReportRecordGraphSize("source.frontend_ready", lowered.artifact.program.instructions, lowered.artifact.program.edges);
        alg::ReportSourceMemory(lowered.artifact, "frontend.ready");
        metadata = std::move(lowered.metadata);
        metadata.update(input_metadata);
        lowered.artifact.frontend_assumptions_json = input_metadata["frontend_assumptions"].dump();
        lowered.artifact.address_invariants_json =
            config.raw.value("address_invariants", Json::array()).dump();
        tensor_property_inputs = config.fixed_input.value("tensor_property_inputs", Json::array());
        lowered.artifact.tensor_properties_json = tensor_property_inputs.dump();
        metadata["executed_records"] = records;
        const auto assumptions =
            config.raw.value("semantic_assumptions", Json::object());
        for (auto it = assumptions.begin(); it != assumptions.end(); ++it)
          lowered.artifact.semantic_assumptions[it.key()] =
              it.value().get<bool>();
        alg::OverridePtxAssumptions(lowered.artifact.semantic_assumptions,
                                    options.assumptions);

#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/source_export.inc"
#endif
        if (!lowered.diagnostics.empty()) {
          result = {{"status", "event_view_incomplete"}, {"stage","event_view_construction"},
                    {"reason",lowered.diagnostics.front().value("code",std::string("event_view_construction_incomplete"))},
                    {"diagnostics", std::move(lowered.diagnostics)},
                    {"verification_performed", false},
                    {"memory_analysis_complete", false}};
          for (const auto &violation : lowered.artifact.source_violations)
            if (violation.code == "cluster_arrive_without_wait") {
              result["status"] = "program_constraint_violation";
              result["stage"] = "cluster_barrier_check";
              result["reason"] = violation.code;
              break;
            }
        } else if (options.extract) {
          result = {{"status", "extraction_complete"},
                    {"verification_performed", false},
                    {"memory_analysis_complete", false},
                    {"verification_limitations", Json::array()},
                    {"source_violations", Json::array()}};
          for (const auto &l : lowered.artifact.verification_limitations)
            result["verification_limitations"].push_back(
                {{"code", l.code}, {"message", l.message}});
          for (const auto &v : lowered.artifact.source_violations)
            result["source_violations"].push_back(
                {{"code", v.code}, {"message", v.message}});
        } else {
          monitor.Frontend(metadata, lowered.artifact);
          monitor.Stage("verification");
          alg::ReachabilityOptions reachability;
          reachability.worker_threads = options.threads;
          alg::MemoryRaceOptions memory;
          memory.collect_duplicate_write_groups=semantic_program.has_value() && options.analyze_values;
          memory.exhaustive = options.exhaustive_memory_search;
          memory.continue_after_race = options.continue_memory_search;
          memory.units = quotient.get();
          // The per-unit path's units: repeated entries are reported per unit.
          const alg::LaunchUnits launch_units{unit_plan.grid, unit_plan.unit_shape};
          if (!quotient && unit_plan.active) memory.launch_units = &launch_units;
          memory.on_search_strategy = [&](const std::string &strategy, bool fallback) {
            monitor.MemoryStrategy(strategy, fallback);
          };
          alg::VerifiedProgramObserver on_verified;
          if (semantic_program && options.analyze_values) {
            on_verified = [&](const alg::SourceArtifact& checked, const alg::PassCertificate& certificate,
                              const alg::MemoryRaceResult& memory_result,
                              const alg::MemoryAccessRecords& accesses, std::string_view verified_report)
                -> std::optional<alg::SameValueFailure> {
              monitor.MemoryVerified(verified_report,
                  kept_verdict ? std::optional(unit_restart->verdict_elapsed_s) : std::nullopt);
              if (unit_lifting && KeepsVerdict(memory_result)) {
                class_verdict.assign(verified_report);
                class_verdict_elapsed_s = monitor.Elapsed();
              }
              const auto serialization_start=Clock::now();
              auto actions = alg::ProfileCall("value_analysis.memory_effect_binding", [&] {
                alg::ProfileSnapshot();
                return ptx::SerializeCheckedMemory(*semantic_program, checked, certificate, memory_result, accesses, true);
              });
              const auto& s=actions.statistics;
              serialization_statistics={{"event_families",s.event_families},{"value_families",s.value_families},
                  {"relations",s.relations},{"scheduled_batches",s.scheduled_batches},
                  {"logical_events",s.logical_events},{"binding_patterns",s.binding_patterns},
                  {"action_patterns",s.action_patterns},{"logical_actions",s.logical_actions},
                  {"expanded_event_records",s.expanded_event_records},{"reachability_queries",s.reachability_queries}};
              serialization_statistics["domain_refinements"]=s.domain_refinements;
              serialization_statistics["singleton_batches"]=s.singleton_batches;
              serialization_statistics["largest_batch_members"]=s.largest_batch_members;
              serialization_statistics["elapsed_seconds"]=std::chrono::duration<double>(Clock::now()-serialization_start).count();
              serialization_statistics["rss_mib_at_completion"]=Rss()/1048576.0;
              serialization_statistics["peak_rss_mib_at_completion"]=PeakRss()/1048576.0;
              monitor.Serialized(serialization_statistics,actions.complete());

#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/checkpoint_save.inc"
#endif

              const auto same_value_writes = std::move(actions.same_value_writes);
              run_values(std::move(actions));
              if (memory_result.same_value_write_groups.empty()) return std::nullopt;
              auto failure = SameValueOutcome(eliminated, value_analysis, same_value_writes);
              if (!failure) monitor.SameValueWritesProven();
              return failure;
            };
          }
          if (options.trusted_reference) {
            monitor.Stage("value_analysis.trusted_reference_order");
            run_values(ptx::BuildTrustedReferenceMemoryActions(*semantic_program, lowered.artifact));
            result = {{"status", value_analysis.value("status", "unsupported") == "expressions_ready"
                                   ? "reference_expressions_ready" : "reference_unsupported"},
                      {"verification_performed", false}, {"memory_analysis_complete", false},
                      {"warnings", Json::array()}, {"warnings_complete", false},
                      {"execution_order_basis", "trusted_reference"}};
          } else {
            const bool canonical_addresses = Json::parse(lowered.artifact.address_invariants_json).empty();
            auto memory_addresses = std::move(lowered.artifact.addresses);
            result = Json::parse(alg::AnalyzeSourceArtifact(
                std::move(lowered.artifact), true, memory, {}, reachability,
                options.sync_timeout, on_verified,
                [addresses = std::move(memory_addresses), &semantic_program, canonical_addresses]() mutable {
                  if (semantic_program)
                    alg::ProfileCall("memory.value_addresses", [&] {
                      ptx::MaterializeValueAddresses(addresses, semantic_program->values, canonical_addresses);
                    });
                  return std::move(addresses);
                }, kept_verdict ? std::string_view(unit_restart->verdict) : std::string_view()));
            if (result.contains("same_value_writes")) monitor.SameValueWritesSettled(result);
          }
        }
        result["frontend"] = std::move(metadata);
      }
      } catch (const alg::UnitClassRestart &restart) {
        restart_per_unit(restart);
      }
      } // fresh PTX path; checkpoint replay already supplied the checked value inputs
    } catch (const FileError &error) {
      result = FileFailure(error);
    } catch (const ptx::LayoutError &error) {
      result = {{"status","memory_layout_unsupported"}, {"reason",error.code}, {"details",error.details},
          {"stage","memory_layout"}, {"failure_stage",monitor.stage}, {"message",error.what()},
          {"memory_analysis_complete",false}, {"warnings_complete",false}};
    } catch (const ptx::ParserError &error) {
      result = {{"status","parser_error"}, {"reason","invalid_ptx_syntax"}, {"stage","ptx_parser"},
          {"message",error.what()}, {"location",{{"file",error.file},{"line",error.line},{"column",error.column}}},
          {"memory_analysis_complete",false}, {"warnings_complete",false}};
    } catch (const std::bad_alloc &error) {
      result = {{"status","allocation_failed"}, {"reason","memory_allocation_failed"},
          {"stage",monitor.stage}, {"resource",{{"kind","memory"},{"origin","allocator"},{"cause","unknown"}}},
          {"message",error.what()}, {"memory_analysis_complete",false}, {"warnings_complete",false}};
    } catch (const ptx::ExecutionError &error) {
      input_metadata["required_parameters"] = error.required_parameters;
      result = {{"status","ptx_execution_unsupported"}, {"stage","ptx_execution"}, {"reason",error.code}, {"message",error.what()},
          {"location",{{"file",options.ptx},{"line",error.line},{"column",0}}},
          {"details",{{"required_parameters",error.required_parameters}}},
          {"frontend",input_metadata}, {"warnings",Json::array()}, {"memory_analysis_complete",false}};
      if (error.code == "execution_step_limit_exceeded") {
        result["status"]="analysis_budget_exceeded";
        result["resource"]={{"origin","algorithm_budget"},{"kind","steps_per_thread"},
            {"limit",options.max_steps_per_thread},{"unit","steps"}};
      }
    } catch (const ptx::InputError &error) {
      result = {{"status", error.status}, {"stage","input_configuration"}, {"reason", error.code}, {"path", error.path},
                {"message",error.what()}, {"frontend",input_metadata},
                {"failure_stage", monitor.stage}, {"warnings", Json::array()},
                {"completed_memory_check", false}, {"memory_analysis_complete", false}};
    } catch (const std::exception &error) {
      result = {{"status", "internal_error"}, {"reason","unexpected_exception"},
                {"stage", monitor.stage}, {"failure_stage", monitor.stage},
                {"message", error.what()},
                {"memory_analysis_complete", false},
                {"warnings_complete", false}};
    }
    if (options.analyze_values) {
      if (value_analysis["status"] == "not_run") {
        value_analysis["reason"] = "verification_prerequisite_not_satisfied";
        value_analysis["prerequisite_status"] = result.value("status", "incomplete");
      }
      result["value_analysis"] = value_analysis;
      if(!serialization_statistics.is_null())result["value_analysis"]["serialization"]=serialization_statistics;
      if (options.analyze_scalar_fold) {
        result["value_analysis"]["scalar_fold"] = {
            {"status", "not_run"}, {"reason", "memory_elimination_not_completed"}};
      }
      const auto* graph = semantic_program ? &semantic_program->values :
          semantic_builder ? &semantic_builder->graph() : nullptr;
      if (graph) {
        if (options.analyze_scalar_fold && eliminated && eliminated->complete) {
          try {
            result["value_analysis"]["scalar_fold"] = AnalyzeScalarFold(*graph, *eliminated, options, monitor, expression,
                unit_lifting ? &*unit_lifting : nullptr, reference ? &compared_outputs : nullptr);
          } catch (const alg::UnitClassRestart &restart) {
            // A lifting refusal re-runs per unit from execution.
            restart_per_unit(restart);
          }
        }
        std::size_t instances = 0;
        for (const auto& node : graph->nodes()) instances += node.instanceCount();
        result["value_ir_node_count"] = graph->size();
        result["value_ir_instance_count"] = instances;
        result["value_view_complete"] = semantic_program && semantic_program->valuesComplete();
        result["semantic_program"] = {{"thread_prefix_groups", graph->groups().size()},
            {"execution_cohorts", semantic_program ? semantic_program->execution_cohorts.size() : 0},
            {"synchronization_groups", semantic_program ? semantic_program->synchronization_groups.size() : 0},
            {"synthetic_synchronization_nodes", semantic_program ? semantic_program->synthetic_synchronization_nodes.size() : 0},
            {"memory_bindings", semantic_program ? semantic_program->memory_bindings.size() : 0},
            {"memory_binding_stored_rows", semantic_program ? semantic_program->memory_bindings.storedBindingCount() : 0},
            {"memory_binding_templates", semantic_program ? semantic_program->memory_bindings.bindingTemplateCount() : 0},
            {"memory_linking_templates_used", semantic_program && semantic_program->memory_linking_templates_used},
            {"mapping_complete", semantic_program && semantic_program->mappingComplete()},
            {"synchronization_groups_materialized", semantic_program && semantic_program->synchronization_groups_materialized}};
        if(replayed_correctness)result["semantic_program"]={{"thread_prefix_groups",graph->groups().size()},
            {"status","restored_value_graph_only"},{"source_bindings","not_rebuilt"}};
        if (semantic_program) {
          auto& info = result["semantic_program"];
          info["value_diagnostic_count"] = semantic_program->value_diagnostics.size();
          info["binding_diagnostic_count"] = semantic_program->diagnostics.size();
          info["value_diagnostics"] = Json::array();
          info["binding_diagnostics"] = Json::array();
          for (const auto& diagnostic : semantic_program->value_diagnostics) {
            info["value_diagnostics"].push_back({{"code", diagnostic.code}, {"message", diagnostic.message},
                {"line", diagnostic.origin.source_line.value_or(0)}});
          }
          if (quotient && quotient->Shared()) {
            // Weighted and expanded in launch order (checked expandable after Finish).
            const auto expansion = BindingEvidence(*semantic_program, *quotient, &info["binding_diagnostics"]);
            info["binding_diagnostic_count"] = expansion.count;
          } else
          for (const auto& diagnostic : semantic_program->diagnostics) {
            info["binding_diagnostics"].push_back({{"code", diagnostic.code}, {"message", diagnostic.message}});
          }
        }

#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/value_export.inc"
#endif
      }
    }
#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/branch_capture_result.inc"
#endif
    else if (values) {
      const auto edges = values->lowerTypeConversions();
      result["type_edges"] = {{"removed_nodes", edges.nodes}, {"converted_instances", edges.instances}};
      result["value_ir_node_count"] = values->size();

#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/dot_export.inc"
#endif
      result["value_ir_node_count"] = values->nodes().size();
      result["value_view_complete"] = result.value("status", "") == "value_view_complete";
      result["verification_performed"] = false;
    }
    if (options.reference_stdin || options.require_equivalence) {
      const auto verified = result.value("verification_status", result.value("status", ""));
      // The first value-stage failure, if any. A not-run scalar projection must
      // not hide a precise lowering/elimination diagnostic (e.g. atomic add), nor
      // relabel that failure as a generic scalar/Fold blockage. Without a
      // reference a complete fold keeps no artifact, so "ready" is no failure.
      const auto value_failure = [&]() -> std::optional<Json> {
        if (expression) return std::nullopt;
        const auto& analysis = result["value_analysis"];
        auto folded = analysis.value("scalar_fold", Json::object());
        auto stage = std::string("value_analysis.scalar_fold");
        if (folded.value("status", "not_run") == "not_run" &&
            analysis.value("status", "not_run") != "expressions_ready") {
          folded = analysis;
          folded.erase("scalar_fold");
          stage = analysis.value("stage", "value_analysis.memory_elimination");
        }
        const auto state = folded.value("status", "not_run");
        if (!reference && state == "ready") return std::nullopt;
        const auto diagnostic = folded.value("diagnostic", Json::object());
        const auto reason = diagnostic.value("code", folded.value("reason", "expression_not_ready"));
        // The public diagnostic's text: the failure's own message, else that of its
        // entry in the stage's (normalized) diagnostics.
        std::string message;
        const auto text = [](const Json& object) {
          return object.is_object() && object.contains("message") && object["message"].is_string()
              ? object["message"].get<std::string>() : std::string();
        };
        message = text(diagnostic);
        if (message.empty()) message = text(folded);
        if (message.empty() && folded.contains("diagnostics") && folded["diagnostics"].is_array())
          for (const auto& entry : folded["diagnostics"])
            if (entry.is_object() && entry.value("code", "") == reason && !(message = text(entry)).empty()) break;
        Json failure{
            {"status", state == "analysis_budget_exceeded" ? "analysis_budget_exceeded" : state == "representation_limit_exceeded" ? "unsupported" :
                       state == "unsupported" ? "unsupported" : "blocked"},
            {"reason", reason}, {"stage", stage}, {"expression_diagnostic", folded}};
        if (!message.empty()) failure["message"] = std::move(message);
        return failure;
      };
      if (verified != "race_free" || !result.value("memory_analysis_complete", false))
        result["equivalence"] = {{"status","blocked"},{"reason","verification_prerequisite_not_satisfied"}};
      // A failed value stage is reported before a missing or failed reference:
      // supplying the reference would not change the result. Both are diagnosed.
      else if (auto failure = value_failure()) {
        result["equivalence"] = std::move(*failure);
        if (!reference)
          result["equivalence"]["diagnostics"] = sentryward::results::Report(reference_error, "equivalence")["diagnostics"];
      }
      else if (!reference) result["equivalence"] = reference_error;
      else {
        // All diagnostic exports are complete. The checker needs only scalar/Fold
        // objects, not a second live copy of the memory-elimination pipeline.
        eliminated.reset();
        semantic_builder.reset();
        semantic_program.reset();
        monitor.Stage("equivalence.object_comparison");
        const auto started = Clock::now();
        rg::ContextBuilder context;
        context = [&](rg::Arena&, const Json& mapping,
            rg::EquivalenceContext& scope, Json& receipt) {
          std::string input_hash, provenance_hash;
          const auto input = rg::fixed_input::metadata(options.config, inputs.File(options.config), input_hash);
          if(input.value("shape_arrays",Json::array()).empty())return;
          if (!mapping.contains("provenance") && !options.reference_provenance.empty()) {
            const auto provenance = rg::fixed_input::metadata(
                options.reference_provenance, inputs.File(options.reference_provenance), provenance_hash);
            scope.fixed_inputs = rg::fixed_input::decode(input, result, provenance, mapping,
                input_hash, provenance_hash);
          } else scope.fixed_inputs = rg::fixed_input::decode_binding(input,result,mapping,input_hash,provenance_hash);
          for (auto& fixed : scope.fixed_inputs)
            fixed.reason = "current verified fixed-read-only input; input sha256=" + input_hash +
                           "; provenance sha256=" + provenance_hash;
          receipt = {{"kind","verified_current_fixed_input"},{"input_sha256",input_hash},
                     {"provenance_sha256",provenance_hash},{"elements",scope.fixed_inputs.size()}};
        };
        AppendTensorPropertyContract(equivalence_mapping, tensor_property_inputs);
        result["equivalence"] = alg::ProfileCall("equivalence.object_comparison", [&] {
          return rg::compareArtifact(std::move(*reference), *expression,
              std::move(equivalence_mapping),
              {.seconds = options.equivalence_timeout,
               .floating_range_premise = effective_assumptions.at("floating_values_in_range")},
              context).report;
        });
        result["equivalence"]["comparison_elapsed_seconds"] = std::chrono::duration<double>(Clock::now()-started).count();
        result["equivalence"]["transport"] = "in_memory_cpp_objects";
        result["equivalence"]["numerical_status"] = result["equivalence"]["status"];
        if (result["equivalence"].value("status", "") == "equivalent" &&
            result["equivalence"].value("access_validity", "") == "not_proven") {
          result["equivalence"]["status"] = "not_proven";
          result["equivalence"]["reason"] = "input_access_validity_not_proven";
        }
      }
      result["value_analysis"]["equivalence_check"] = result["equivalence"].value("result", result["equivalence"].value("status", std::string("not_run")));
      result["value_analysis"].erase("equivalence_reason");
    }
    total.Stop();
    monitor.Stop();
    // Normal completion and deadline reports expose the same completed
    // verification checkpoint independently of the numerical outcome.
    if (monitor.frontend_checkpoint.contains("verification_status")) {
      result["verification_status"] = monitor.frontend_checkpoint["verification_status"];
      result["verification_elapsed_seconds"] = monitor.frontend_checkpoint["verification_elapsed_seconds"];
    }
    if (monitor.memory_limit_mib &&
        std::max(monitor.peak.load(), monitor.peak_footprint.load()) > *monitor.memory_limit_mib * 1024 * 1024) {
      Json partial = std::move(result);
      result = {{"schema", "spverify.native-ptx-run/v1"},
                {"status", "memory_limit_exceeded"},
                {"resource", monitor.MemoryResource(*monitor.memory_limit_mib)},
                {"message", "Memory budget exceeded (RSS or physical footprint) before final monitor sample"},
                {"partial_report", std::move(partial)},
                {"memory_analysis_complete", false},
                {"warnings_complete", false}};
      result.update(monitor.frontend_checkpoint);
      if(options.require_equivalence || options.reference_stdin) {
        if(result["partial_report"].contains("equivalence")) {
          result["equivalence"]=std::move(result["partial_report"]["equivalence"]);
          result["partial_report"].erase("equivalence");
        } else result["equivalence"]={{"status","blocked"},{"reason","verification_prerequisite_not_satisfied"},
          {"prerequisite_status","memory_limit_exceeded"},{"stage",monitor.stage}};
      }
    }
    if (options.value_thread_range) {
      result["analysis_scope"] = {{"kind","thread_subset"},
          {"first",options.value_thread_range->first},
          {"count",options.value_thread_range->second-options.value_thread_range->first},
          {"whole_launch_verified",false}};
    }
    result["template_execution"]={{"completed_ctas",template_ctas},
        {"template_steps",template_execution.steps},{"logical_thread_steps",template_execution.logical_steps},
        {"splits",template_execution.splits},{"value_nodes",template_execution.value_nodes},
        {"record_nodes",template_execution.record_nodes},
        {"record_bindings",template_execution.record_bindings},
        {"coordinate_bindings",template_execution.coordinate_bindings},
        {"path_entries",template_execution.path_entries},
        {"thread_memberships",template_execution.thread_memberships}};
    result["thread_record_compression"] = {
        {"enabled", options.compress_thread_records}, {"completed_ctas", ctas},
        {"logical_records", records}, {"intern_calls", record_compression.record_count},
        {"templates", record_compression.template_count},
        {"unique_bindings", record_compression.unique_binding_count},
        {"record_handle_bytes", sizeof(ptx::ExecutedInstruction)},
        {"estimated_pool_bytes_sum", record_compression.estimated_owned_bytes}};
    result["pipeline_elapsed_seconds"] = monitor.Elapsed();
    result["semantic_assumptions"] = effective_assumptions;
    // Floating values stay in range: the equivalence import uses the premise where it drops a range
    // bound (counter range_premise_bounds_dropped). Unused only when the comparison imported every
    // output family without a drop; unmeasured (null) when it did not run or stopped early.
    std::optional<Json> floating_range_used;
    if (effective_assumptions.at("floating_values_in_range")) {
      floating_range_used = Json(nullptr);
      if (const auto equivalence = result.find("equivalence");
          equivalence != result.end() && equivalence->is_object() && equivalence->contains("counters")) {
        if (equivalence->at("counters").value("range_premise_bounds_dropped", std::uint64_t{0}) > 0)
          floating_range_used = Json(true);
        else if (equivalence->contains("total_output_families") &&
                 equivalence->value("finished_families", Json()) == equivalence->at("total_output_families"))
          floating_range_used = Json(false);
      }
    }
    if (!result.contains("assumptions")) {
      result["assumptions"] = Json::array();
      for (const auto& entry : alg::AssumptionPool()) {
        Json triggered = effective_assumptions.at(entry.key) ? Json(nullptr) : Json(false);
        if (effective_assumptions.at(entry.key) && result.value("status", "") == "extraction_complete") {
          const auto facts = input_metadata.value("frontend_assumptions", Json::object());
          if (entry.key == "representative_election") triggered = facts.value("election_uses", 0) > 0;
          if (entry.key == "clc_always_fails") triggered = facts.value("clc_uses", 0) > 0;
          if (entry.key == "representative_branches") triggered = facts.value("representative_branch_uses", 0) > 0;
          if (entry.key == "bounded_computation_loops") triggered = facts.value("bounded_loop_uses", 0) > 0;
          if (entry.key == "order_independent_atomics") triggered = facts.value("atomic_uses", 0) > 0;
        }
        if (entry.key == "floating_values_in_range" && floating_range_used) triggered = *floating_range_used;
        result["assumptions"].push_back({{"label",entry.label},{"key",entry.key},{"text",entry.text},
            {"enabled",effective_assumptions.at(entry.key)},
            {"triggered",triggered}});
      }
    } else if (floating_range_used && result["assumptions"].is_array()) {
      // The verification report's list (source_json.cpp RenderReport) saw no comparison.
      for (auto& entry : result["assumptions"])
        if (entry.value("key", "") == "floating_values_in_range") entry["triggered"] = *floating_range_used;
    }
    const auto reason = result.value("reason", "");
    if (reason == "branch_not_static" || reason == "missing_parameter_value") {
      result["assumption_hints"] = Json::array();
      for (const auto& entry : alg::AssumptionPool())
        if ((entry.key == "representative_branches" || entry.key == "bounded_computation_loops") &&
            !effective_assumptions.at(entry.key))
          result["assumption_hints"].push_back({{"label",entry.label},{"key",entry.key},
            {"suggested_value",true},{"message","Supply the missing control input for exact analysis. This assumption applies only if CFG classification identifies the corresponding structured control; polling and unclassified loops remain unsupported."}});
    }
    for (const auto& entry : alg::AssumptionPool())
      if (reason == "assumption_required_" + entry.key)
        result["assumption_hints"] = Json::array({{{"label",entry.label},{"key",entry.key},
            {"suggested_value",true},{"message",result.value("message", "")}}});
    result["rss_limit_mib"] = options.rss_mib ? Json(*options.rss_mib) : Json(nullptr);
    result["peak_rss_mib"] = monitor.peak.load() / 1048576.0;
    result["peak_footprint_mib"] = monitor.peak_footprint.load() / 1048576.0;
    result["implementation"] = "cpp";
    result["wellsync_timeout_seconds"] = options.sync_timeout;
    result["whole_pipeline_timeout_seconds"] = options.timeout;
    monitor.RecordUnitRestart(result);
    const auto exit_status = result.value("status", "");
    result = sentryward::results::PublicReport(
        std::move(result), options.extract ? "extraction" : options.semantic ? "semantic_analysis" : "kernel_verification");
    Write(options.report, result);

#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/branch_capture_export.inc"
#endif

    std::cout << result.dump(2) << '\n';
    auto status = exit_status;
    if (status == "memory_limit_exceeded")
      return 137;
    if (status == "invalid_configuration" || status == "parser_error" || status == "input_file_error" || status == "output_file_error") return 1;
    return status == "internal_error" ||
                   status == "value_view_incomplete" ||
                   status == "branch_scope_incomplete" ||
                   status == "event_view_incomplete"
               ? 2
               : 0;
  } catch (const FileError &error) {
    std::cout << sentryward::results::PublicReport(FileFailure(error), "kernel_verification").dump() << '\n';
    return 1;
  } catch (const std::bad_alloc &error) {
    std::cout << sentryward::results::PublicReport(Json{{"schema","spverify.native-ptx-run/v1"},{"status","allocation_failed"},
        {"resource",{{"kind","memory"},{"origin","allocator"},{"cause","unknown"}}},
        {"message",error.what()},{"memory_analysis_complete",false},{"warnings_complete",false}}, "kernel_verification").dump() << '\n';
    return 2;
  } catch (const std::exception &error) {
    const auto status = parsing_arguments ? "invalid_configuration" : "internal_error";
    std::cout << sentryward::results::PublicReport(Json{{"schema","spverify.native-ptx-run/v1"},{"status",status},
        {"stage",parsing_arguments ? "cli_arguments" : "pipeline"},
        {"reason",parsing_arguments ? "invalid_cli_arguments" : "unexpected_exception"},
        {"message",error.what()},{"memory_analysis_complete",false},{"warnings_complete",false}}, "kernel_verification").dump() << '\n';
    std::cerr << "sentryward-ptx: " << error.what() << '\n';
    return parsing_arguments ? 1 : 2;
  }
}

// At most two attempts: a class-mode attempt that restarts per unit leaves the
// call's inputs, its start time and its verdict to the per-unit attempt.
int main(int argc, char **argv) {
  CallInputs inputs;
  std::optional<PerUnitRestart> unit_restart;
  for (;;) {
    try {
      return RunAttempt(argc, argv, inputs, unit_restart);
    } catch (PerUnitRestart &restart) {
      unit_restart = std::move(restart);
#if defined(__GLIBC__)
      // The class-mode attempt's pages, freed by its unwinding, go back before
      // the per-unit attempt.
      malloc_trim(0);
#endif
    }
  }
}
