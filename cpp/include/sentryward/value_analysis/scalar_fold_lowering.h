#pragma once
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <vector>
#include "sentryward/scalar_fold/graph.h"
#include "sentryward/scalar_fold/output_family.h"
#include "sentryward/value_analysis/memory_elimination.h"
#include "sentryward/value_analysis/unit_lifting.h"

namespace sentryward::value_analysis {
struct ScalarFoldOptions {
  bool compact_output_families = false;
  std::size_t max_nodes = 2000000;
  // Projection uses an explicit worklist; this bounds nested Fold binders.
  std::size_t max_recursion = 2048;
  std::size_t max_outputs = 1000000;
  std::size_t collection_interval = 128; // Completed outputs between live-root collections; zero disables.
  std::size_t max_pending_projections = 2000000;
  // Default: use a quarter of the graph budget as the allocation quantum.
  // Small live graphs must not trigger repeated eviction of reusable
  // projections every few thousand temporary nodes. Explicit small values
  // remain useful for moving-GC stress tests; zero disables construction GC.
  std::size_t construction_collection_interval = std::numeric_limits<std::size_t>::max();
  // Cross-CTA value lifting mode (unit_lifting.h): BlockIndex stays a unit
  // parameter and every body is instantiated for its unit before the output
  // tail. Null: today's lowering, unchanged.
  const UnitLifting* unit_lifting = nullptr;
};
struct ScalarFoldOutput {
  ConstantRegion region;
  scalar_fold::Ref value;
};
struct ScalarFoldDiagnostic {
  std::string code, message;
  std::optional<MemoryInstance> operation;
};
struct ScalarFoldResult {
  scalar_fold::Graph graph;
  std::vector<ScalarFoldOutput> outputs;
  std::vector<scalar_fold::OutputFamily> output_families;
  bool complete = false;
  std::optional<ScalarFoldDiagnostic> diagnostic;
  std::size_t source_outputs = 0;
  std::size_t nodes_before_final_collection = 0;
  std::size_t nodes_reclaimed = 0;
  std::size_t construction_collections = 0;
  std::size_t peak_construction_nodes = 0;
  std::size_t shared_view_projection_hits = 0;
  std::size_t output_families_merged = 0;
  std::size_t spatial_families_projected = 0;
  std::size_t spatial_outputs_projected = 0;
  std::size_t spatial_template_operations = 0;
  // Lifting mode only (ScalarFoldOptions::unit_lifting).
  struct Lifting {
    std::size_t parameters = 0; // unit coordinate leaves emitted
    std::size_t declined = 0;   // BlockIndex reads answered "unknown"
    // Set when lifting cannot finish exactly: the driver restarts per unit.
    std::optional<ScalarFoldDiagnostic> refused;
  } lifting;
  // Where the lowering stopped when `diagnostic` is set (for class mode's
  // first-item rule, run.cpp). Phases 'a'..'f':
  // prerequisites, normalized reads, budgets, output families, output records
  // (items in record order: a normalized output group, a contiguous scalar run
  // or one record), final instantiation and consolidation.
  struct FailureSite {
    char phase = 0;
    std::size_t items = 0;             // phase 'e' items started, the failing one included
    std::optional<std::size_t> record; // the last started item's first record (index of memory.outputs)
    bool output_families = false;      // memory.output_families was non-empty
    std::size_t completed_outputs = 0; // outputs completed when it failed
    bool literal_message = false;      // the message is a literal at its fail() site
  } failure_site;
};
// Pure, ideal-arithmetic projection after checked memory elimination.
// Preserves allocation identities and ordered numerical conversion attributes.
ScalarFoldResult lowerScalarFold(
    const value_ir::Graph&, const MemoryEliminationResult&,
    const ScalarFoldOptions& options = {});
} // namespace sentryward::value_analysis
