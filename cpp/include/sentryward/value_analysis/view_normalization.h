#pragma once
#include "sentryward/value_analysis/memory_elimination.h"

namespace sentryward::value_analysis {
struct ViewNormalizationStatistics final {
  std::size_t identities = 0, elements = 0, slices = 0, layouts = 0, segmented = 0;
  std::size_t original_pieces = 0, mapping_blocks = 0;
  std::size_t reused_initial_reads = 0;
  std::size_t reused_views = 0, stored_mapping_blocks = 0;
  std::size_t typed_output_regions = 0, typed_output_pieces = 0;
};
// Preserves source instances, allocation identity and ordered type conversions.
ViewNormalizationStatistics normalizeMemoryViews(
    const value_ir::Graph& graph, MemoryEliminationResult& result);
}  // namespace sentryward::value_analysis
