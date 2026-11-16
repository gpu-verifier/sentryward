#pragma once
#include "run_support.hpp"

namespace {
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

} // namespace
