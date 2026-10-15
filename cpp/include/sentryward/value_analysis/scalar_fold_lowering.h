#pragma once
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <vector>
#include "sentryward/scalar_fold/graph.h"
#include "sentryward/scalar_fold/output_family.h"
#include "sentryward/value_analysis/memory_elimination.h"

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
};
// Pure, ideal-arithmetic projection after checked memory elimination.
// Preserves allocation identities and ordered numerical conversion attributes.
ScalarFoldResult lowerScalarFold(
    const value_ir::Graph&, const MemoryEliminationResult&,
    const ScalarFoldOptions& options = {});
} // namespace sentryward::value_analysis
