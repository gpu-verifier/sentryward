#pragma once
#include "run_support.hpp"

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
  // A reported reference failure is the outcome of a required equivalence check.
  if (!o.reference_failure_json.empty()) o.require_equivalence = o.analyze_values = o.analyze_scalar_fold = true;
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
} // namespace
