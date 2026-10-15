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
#include "sentryward/value_ir/operations.h"
#include "sentryward/value_analysis/passes.h"
#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/checkpoint_headers.inc"
#endif
#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/checkpoint_version.inc"
#endif
#include <openssl/evp.h>
#include <filesystem>
#include "sentryward/value_analysis/scalar_fold_lowering.h"
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
  std::size_t threads = 8;
  bool require_equivalence = false;
  std::string reference_failure_json;
  std::uint64_t max_steps_per_thread = 100000;
  alg::PtxAssumptions assumptions;
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
             "MIB  Optional positive memory budget (default unlimited)\n  --sync-threads 8\n  "
             "--exhaustive-memory-search  Enumerate/count all race pairs\n"
#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/value_help.inc"
#endif
#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/branch_help.inc"
#endif
             "--value-thread-range FIRST:COUNT  Extract a thread subset; requires cross-subset validation\n"
             "--no-thread-record-compression  Disable template execution and shared records for comparison\n"
             "--max-steps-per-thread N  Per-thread PTX execution budget (default 100000)\n"
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
std::string Read(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw FileError(path, false);
  return {std::istreambuf_iterator<char>(file), {}};
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
struct Monitor {
  const Options &options;
  Clock::time_point start = Clock::now();
  std::mutex mutex;
  std::condition_variable changed;
  bool finished = false;
  std::string stage = "startup";
  Json frontend_checkpoint = Json::object();
  std::atomic<std::uint64_t> peak = 0;
  std::atomic<std::uint64_t> peak_footprint = 0;
  std::thread worker;
  explicit Monitor(const Options &o)
      : options(o), worker([this] { Watch(); }) {}
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
    frontend_checkpoint = std::move(checkpoint);
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
  void MemoryVerified(std::string_view verified_report) {
    // One small report snapshot, not another graph traversal or verification.
    // Preserve counts, warning samples and assumption usage on a later timeout.
    auto checkpoint = Json::parse(verified_report);
    checkpoint["verification_status"] = checkpoint.at("status");
    checkpoint.erase("status");
    checkpoint.erase("schema");
    checkpoint["verification_elapsed_seconds"] = Elapsed();
    checkpoint["warnings_complete"] = true;
    std::lock_guard lock(mutex);
    stage = "value_analysis.memory_effect_binding";
    frontend_checkpoint.update(checkpoint);
  }
  void Serialized(const Json& statistics,bool complete) {
    std::lock_guard lock(mutex);
    frontend_checkpoint["value_analysis"]={{"serialization",statistics},{"serialization_complete",complete}};
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
      if (changed.wait_for(lock, std::chrono::milliseconds(100),
                           [this] { return finished; }))
        break;
      auto rss = Rss();
      peak.store(std::max(peak.load(), rss));
      auto footprint = MemoryFootprint();
      peak_footprint.store(std::max(peak_footprint.load(), footprint));
      double elapsed = Elapsed();
      bool timeout = options.timeout > 0 && elapsed >= options.timeout,
           limit = options.rss_mib && std::max(rss, footprint) > *options.rss_mib * 1024 * 1024;
      if (!timeout && !limit)
        continue;
      Json result{
          {"schema", "spverify.native-ptx-run/v1"},
          {"status", limit ? "memory_limit_exceeded" : "timeout"},
          {"timeout_scope", "whole_pipeline"},
          {"resource",{{"kind",limit ? "memory" : "time"},{"origin","configured_budget"},
                       {"limit",limit ? double(*options.rss_mib) : options.timeout},
                       {"unit",limit ? "MiB" : "seconds"}}},
          {"elapsed_seconds", elapsed},
          {"peak_rss_mib", peak.load() / 1048576.0},
          {"peak_footprint_mib", peak_footprint.load() / 1048576.0},
          {"active_stage", stage},
          {"memory_analysis_complete", false},
          {"warnings_complete", false},
          {"implementation", "cpp"},
          {"rss_limit_mib", options.rss_mib ? Json(*options.rss_mib) : Json(nullptr)}};
      result.update(frontend_checkpoint);
      if(options.require_equivalence || options.reference_stdin) {
        if(stage=="equivalence.object_comparison")
          result["equivalence"]={{"status",limit?"memory_limit_exceeded":"timeout"},
            {"reason",limit?"rss_limit":"time_budget"},{"stage",stage}};
        else result["equivalence"]={{"status","blocked"},{"reason","verification_prerequisite_not_satisfied"},
          {"prerequisite_status",result.value("status","timeout")},{"stage",stage}};
      }
      sentryward::results::Normalize(result, "kernel_verification");
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

Json AnalyzeScalarFold(const sentryward::value_ir::Graph& values,
                       const sentryward::value_analysis::MemoryEliminationResult& eliminated,
                       const Options& options, Monitor& monitor,
                       std::optional<sentryward::scalar_fold::Artifact>& artifact) {
  namespace sf = sentryward::scalar_fold;
  Json report;
  try {
    monitor.Stage("value_analysis.scalar_fold");
    auto folded = alg::ProfileCall("value_analysis.scalar_fold", [&] {
      sentryward::value_analysis::ScalarFoldOptions fold_options;
      fold_options.compact_output_families=true;
      return sentryward::value_analysis::lowerScalarFold(values, eliminated,fold_options);
    });
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
} // namespace
int main(int argc, char **argv) {
  bool parsing_arguments = true;
  try {
    auto options = Arguments(argc, argv);
    parsing_arguments = false;
    Monitor monitor(options);
    std::optional<rg::Document> reference;
    std::optional<sentryward::scalar_fold::Artifact> expression;
    Json equivalence_mapping, reference_error={{"status","missing_reference"},{"reason","reference_expression_not_supplied"}};
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
        reference = rg::Document::load(std::cin);
        std::ifstream input(options.equivalence_mapping);
        if (!input) throw std::invalid_argument("missing equivalence mapping");
        input >> equivalence_mapping;
      } catch (const std::exception& error) {
        reference.reset();
        reference_error = {{"status","unsupported"},{"reason","reference_import_failed"},{"message",error.what()}};
      }
    }
    alg::ProfileSession profiling(true);
    alg::ProfileScope total("pipeline.total");
    auto poll = [] { alg::ProfileSnapshot(); };
    Json metadata, input_metadata = Json::object();
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
      auto pipeline = sentryward::value_analysis::runValuePasses(
          semantic_program->values, actions.actions, actions.contract, [&](std::string_view pass) {
            monitor.Stage("value_analysis." + std::string(pass));
          });
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
                                   [&] { return Read(options.ptx); });

#ifdef SENTRYWARD_DEVELOPMENT_EXPORTS
#include "sentryward/development/runner/checkpoint_identity.inc"
#endif
      monitor.Stage("ptx_parser");
      auto module = alg::ProfileCall("ptx_parser", [&] {
        return ptx::Parse(std::move(text), options.ptx);
      });
      monitor.Stage("input_configuration");
      auto config = alg::ProfileCall("input_configuration", [&] {
        auto supplied = ptx::LoadKernelInput(options.config);
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
          {"representative_branch_uses",0}, {"bounded_loop_uses",0}, {"data_branch_decisions",Json::array()},
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
      std::unique_ptr<ptx::ModelLowerer> lowerer;
      if (options.analyze_values || (!options.extract && !options.semantic && !values)) {
        semantic_builder = std::make_unique<ptx::SemanticProgramBuilder>(module, *function, poll,
            ptx::SemanticProgramOptions{options.analyze_values && !options.trusted_reference,
                                       true, options.analyze_values, !options.trusted_reference,
                                       !options.dump_branch_scope.empty()});
      } else if (!options.semantic && !values)
        lowerer = alg::ProfileCall("frontend.static_lowering_prepare", [&] {
          return std::make_unique<ptx::ModelLowerer>(module, *function, poll);
        });
      {
        alg::ProfileScope frontend("frontend.dynamic_pipeline");
        const auto purpose = branch_capture ? ptx::SpecializationPurpose::BranchScopeCapture
                            : !options.dump_branch_scope.empty() ? ptx::SpecializationPurpose::VerificationWithBranchScope
                            : values ? ptx::SpecializationPurpose::ValueExtraction
                                    : ptx::SpecializationPurpose::Verification;
        std::set<ptx::Coordinates> emitted;
        std::map<std::uint64_t, ptx::DataBranchDecision> data_branch_decisions;
        std::map<std::uint64_t, std::uint64_t> opaque_regions;
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
              const auto &execution=program.template_execution;
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
              const auto &storage = program.record_compression;
              record_compression.record_count += storage.record_count;
              record_compression.template_count += storage.template_count;
              record_compression.unique_binding_count += storage.unique_binding_count;
              record_compression.estimated_owned_bytes += storage.estimated_owned_bytes;
              for (const auto &lane : program.lane_traces) {
                records += lane.instructions.size();
                auto &facts = input_metadata["frontend_assumptions"];
                facts["assertion_uses"] = facts["assertion_uses"].get<std::uint64_t>() + lane.assumed_assertions;
                for (const auto &[line, count] : lane.opaque_regions) opaque_regions[line] += count;
                for (const auto &[line, decision] : lane.data_branch_decisions) {
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
              for (std::size_t id=0;id<program.static_execution_counts.size();++id) {
                auto &facts=input_metadata["frontend_assumptions"];
                const auto &op=function->instructions[id].opcode;
                const auto count=program.static_execution_counts[id];
                if (op=="elect") facts["election_uses"]=facts["election_uses"].get<std::uint64_t>()+count;
                if (op=="clusterlaunchcontrol") facts["clc_uses"]=facts["clc_uses"].get<std::uint64_t>()+count;
              }
              if (lowerer) {
                monitor.Stage("source_ir.cta" + std::to_string(ctas));
                const auto linear=(std::uint64_t(member[2])*grid[1]+member[1])*grid[0]+member[0];
                lowerer->AddCta(linear, member, program);
              }
              if (value_extractor) {
                monitor.Stage("value_ir.cta" + std::to_string(ctas));
                const auto linear = (std::uint64_t(member[2]) * grid[1] + member[1]) * grid[0] + member[0];
                value_extractor->AddCta(static_cast<std::uint32_t>(linear), member, program);
              }
              if (semantic_builder) {
                monitor.Stage("semantic_ir.cta" + std::to_string(ctas));
                const auto linear = (std::uint64_t(member[2]) * grid[1] + member[1]) * grid[0] + member[0];
                semantic_builder->AddCta(static_cast<std::uint32_t>(linear), member, program);
              }
              alg::ProfileCall("frontend.release_execution",
                               [&] { program = ptx::ExecutionProgram{}; });
              ++ctas;
              emitted.insert(member);
              }
            }
        for (const auto &[line, count] : opaque_regions)
          input_metadata["frontend_assumptions"]["opaque_compute_regions"].push_back({{"line", line}, {"lanes", count}});
        auto &decisions = input_metadata["frontend_assumptions"]["data_branch_decisions"];
        for (const auto &[line, decision] : data_branch_decisions)
          decisions.push_back({{"line", line}, {"assumption", decision.assumption}, {"rule", decision.rule},
                               {"taken", decision.taken}, {"not_taken", decision.not_taken}});
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
        auto lowered = semantic_program ? std::move(semantic_program->lowering) : lowerer->Finish();
        ptx::ValidateConstantInputs(config, lowered.artifact);
        alg::ProfileCall("frontend.release_builder", [&] { lowerer.reset(); });
        alg::ReportRecordGraphSize("source.frontend_ready", lowered.artifact.program.instructions, lowered.artifact.program.edges);
        alg::ReportSourceMemory(lowered.artifact, "frontend.ready");
        metadata = std::move(lowered.metadata);
        metadata.update(input_metadata);
        lowered.artifact.frontend_assumptions_json = input_metadata["frontend_assumptions"].dump();
        lowered.artifact.address_invariants_json =
            config.raw.value("address_invariants", Json::array()).dump();
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
          memory.on_search_strategy = [&](const std::string &strategy, bool fallback) {
            monitor.MemoryStrategy(strategy, fallback);
          };
          alg::VerifiedProgramObserver on_verified;
          if (semantic_program && options.analyze_values) {
            on_verified = [&](const alg::SourceArtifact& checked, const alg::PassCertificate& certificate,
                              const alg::MemoryRaceResult& memory_result,
                              const alg::MemoryAccessRecords& accesses, std::string_view verified_report) {
              monitor.MemoryVerified(verified_report);
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

              run_values(std::move(actions));
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
                }));
          }
        }
        result["frontend"] = std::move(metadata);
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
          result["value_analysis"]["scalar_fold"] = AnalyzeScalarFold(*graph, *eliminated, options, monitor, expression);
        }
        std::size_t instances = 0;
        for (const auto& node : graph->nodes()) instances += node.instances().size();
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
      if (verified != "race_free" || !result.value("memory_analysis_complete", false))
        result["equivalence"] = {{"status","blocked"},{"reason","verification_prerequisite_not_satisfied"}};
      else if (!reference) result["equivalence"] = reference_error;
      else if (!expression) {
        // Preserve the first failed prerequisite. A not-run scalar projection
        // must not hide a precise lowering/elimination diagnostic (e.g. atomic
        // add), nor relabel that failure as a generic scalar/Fold blockage.
        const auto& analysis = result["value_analysis"];
        auto folded = analysis.value("scalar_fold", Json::object());
        auto stage = std::string("value_analysis.scalar_fold");
        if (folded.value("status", "not_run") == "not_run" &&
            analysis.value("status", "not_run") != "expressions_ready") {
          folded = analysis;
          folded.erase("scalar_fold");
          stage = analysis.value("stage", "value_analysis.memory_elimination");
        }
        const auto diagnostic = folded.value("diagnostic", Json::object());
        const auto state = folded.value("status", "not_run");
        result["equivalence"] = {
            {"status", state == "analysis_budget_exceeded" ? "analysis_budget_exceeded" : state == "representation_limit_exceeded" ? "unsupported" :
                       state == "unsupported" ? "unsupported" : "blocked"},
            {"reason", diagnostic.value("code", folded.value("reason", "expression_not_ready"))},
            {"stage", stage}, {"expression_diagnostic", folded}};
      }
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
          const auto input = rg::fixed_input::metadata(options.config, input_hash);
          if(input.value("shape_arrays",Json::array()).empty())return;
          if (!mapping.contains("provenance") && !options.reference_provenance.empty()) {
            const auto provenance = rg::fixed_input::metadata(options.reference_provenance, provenance_hash);
            scope.fixed_inputs = rg::fixed_input::decode(input, result, provenance, mapping,
                input_hash, provenance_hash);
          } else scope.fixed_inputs = rg::fixed_input::decode_binding(input,result,mapping,input_hash,provenance_hash);
          for (auto& fixed : scope.fixed_inputs)
            fixed.reason = "current verified fixed-read-only input; input sha256=" + input_hash +
                           "; provenance sha256=" + provenance_hash;
          receipt = {{"kind","verified_current_fixed_input"},{"input_sha256",input_hash},
                     {"provenance_sha256",provenance_hash},{"elements",scope.fixed_inputs.size()}};
        };
        result["equivalence"] = alg::ProfileCall("equivalence.object_comparison", [&] {
          return rg::compareArtifact(std::move(*reference), *expression,
              std::move(equivalence_mapping), {options.equivalence_timeout, 0}, context).report;
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
    if (options.rss_mib && std::max(monitor.peak.load(), monitor.peak_footprint.load()) > *options.rss_mib * 1024 * 1024) {
      Json partial = std::move(result);
      result = {{"schema", "spverify.native-ptx-run/v1"},
                {"status", "memory_limit_exceeded"},
                {"resource",{{"kind","memory"},{"origin","configured_budget"},{"limit",*options.rss_mib},{"unit","MiB"}}},
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
        }
        result["assumptions"].push_back({{"label",entry.label},{"key",entry.key},{"text",entry.text},
            {"enabled",effective_assumptions.at(entry.key)},
            {"triggered",triggered}});
      }
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
    const auto exit_status = result.value("status", "");
    sentryward::results::Normalize(result, options.extract ? "extraction" : options.semantic ? "semantic_analysis" : "kernel_verification");
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
    std::cout << sentryward::results::Report(FileFailure(error), "kernel_verification").dump() << '\n';
    return 1;
  } catch (const std::bad_alloc &error) {
    std::cout << sentryward::results::Report(Json{{"schema","spverify.native-ptx-run/v1"},{"status","allocation_failed"},
        {"resource",{{"kind","memory"},{"origin","allocator"},{"cause","unknown"}}},
        {"message",error.what()},{"memory_analysis_complete",false},{"warnings_complete",false}}, "kernel_verification").dump() << '\n';
    return 2;
  } catch (const std::exception &error) {
    const auto status = parsing_arguments ? "invalid_configuration" : "internal_error";
    std::cout << sentryward::results::Report(Json{{"schema","spverify.native-ptx-run/v1"},{"status",status},
        {"stage",parsing_arguments ? "cli_arguments" : "pipeline"},
        {"reason",parsing_arguments ? "invalid_cli_arguments" : "unexpected_exception"},
        {"message",error.what()},{"memory_analysis_complete",false},{"warnings_complete",false}}, "kernel_verification").dump() << '\n';
    std::cerr << "sentryward-ptx: " << error.what() << '\n';
    return parsing_arguments ? 1 : 2;
  }
}
