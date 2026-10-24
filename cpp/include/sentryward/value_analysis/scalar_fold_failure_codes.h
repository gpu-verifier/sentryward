#pragma once
// Cross-CTA class mode, first-item rule (run.cpp): every
// failure code of the scalar-fold lowering, classified once. A class-mode
// quotient's fold failure may stand for the launch only if its code is
// context-free: decided by the projection request and its cone alone, so the
// per-unit lowering raises it for the same item in the same way. A code is
// allowed only per site: the failure must also carry a literal message (the
// lowering's fail() records which overload raised it), so no CTA, offset or
// node id can enter the reported text. Codes outside this table are excluded.
// The table covers every code of the lowering's fail() sites and of the two
// foreign failure types it reports (IndexedSourceLimit,
// frontend::ptx::LayoutError).
#include <array>
#include <cstdint>
#include <string_view>

namespace sentryward::value_analysis {
enum class FoldFailureKind : std::uint8_t {
  ContextFree, // allowed (with a literal message)
  Budget,      // depends on how much was processed before
  Magnitude,   // representation range (excluded: a member's translated offsets could differ)
  LaunchWide,  // a check over all outputs
  KeyText,     // the message names projection keys
  Internal,    // an internal invariant
  Lifting,     // the lifting's own failure
  Dynamic,     // dynamic what() (LayoutError)
};
struct FoldFailureCode {
  std::string_view code;
  FoldFailureKind kind;
};
inline constexpr std::array kFoldFailureCodes{
    FoldFailureCode{"scalar_fold_cross_element", FoldFailureKind::ContextFree},
    FoldFailureCode{"scalar_fold_invalid_index", FoldFailureKind::ContextFree},
    FoldFailureCode{"scalar_fold_subbyte_value", FoldFailureKind::ContextFree},
    FoldFailureCode{"scalar_fold_wide_scalar", FoldFailureKind::ContextFree},
    FoldFailureCode{"scalar_fold_parameterized_type", FoldFailureKind::ContextFree},
    FoldFailureCode{"scalar_fold_parameterized_shape", FoldFailureKind::ContextFree},
    FoldFailureCode{"scalar_fold_parameterized_mma", FoldFailureKind::ContextFree},
    FoldFailureCode{"scalar_fold_mma_variant", FoldFailureKind::ContextFree},
    FoldFailureCode{"scalar_fold_mma_shape", FoldFailureKind::ContextFree},
    FoldFailureCode{"scalar_fold_mma_flags", FoldFailureKind::ContextFree},
    FoldFailureCode{"scalar_fold_subbyte_mma", FoldFailureKind::ContextFree},
    FoldFailureCode{"scalar_fold_component_rank", FoldFailureKind::ContextFree},
    FoldFailureCode{"scalar_fold_component_index", FoldFailureKind::ContextFree},
    FoldFailureCode{"scalar_fold_conversion_projection", FoldFailureKind::ContextFree},
    FoldFailureCode{"scalar_fold_thread_selector", FoldFailureKind::ContextFree},
    FoldFailureCode{"scalar_fold_matrix_operand", FoldFailureKind::ContextFree},
    FoldFailureCode{"scalar_fold_matrix_constant", FoldFailureKind::ContextFree},
    FoldFailureCode{"scalar_fold_operation_unsupported", FoldFailureKind::ContextFree},
    FoldFailureCode{"scalar_fold_spatial_projection", FoldFailureKind::ContextFree},
    FoldFailureCode{"scalar_fold_scale_replicas", FoldFailureKind::ContextFree},
    FoldFailureCode{"scalar_fold_missing_read", FoldFailureKind::ContextFree},
    FoldFailureCode{"scalar_fold_coordinate", FoldFailureKind::ContextFree},
    FoldFailureCode{"compact_view_unsupported", FoldFailureKind::ContextFree},
    FoldFailureCode{"symbolic_initial_base_unsupported", FoldFailureKind::ContextFree},
    FoldFailureCode{"expression_node_limit_exceeded", FoldFailureKind::Budget},
    FoldFailureCode{"projection_worklist_limit_exceeded", FoldFailureKind::Budget},
    FoldFailureCode{"output_family_limit_exceeded", FoldFailureKind::Budget},
    FoldFailureCode{"output_count_limit_exceeded", FoldFailureKind::Budget},
    FoldFailureCode{"output_domain_limit_exceeded", FoldFailureKind::Budget},
    FoldFailureCode{"layout_nesting_limit_exceeded", FoldFailureKind::Budget},
    FoldFailureCode{"expression_nesting_limit_exceeded", FoldFailureKind::Budget},
    FoldFailureCode{"domain_refinement_limit_exceeded", FoldFailureKind::Budget},
    FoldFailureCode{"invalid_analysis_budget", FoldFailureKind::Budget},
    FoldFailureCode{"source_index_limit_exceeded", FoldFailureKind::Budget}, // IndexedSourceLimit
    FoldFailureCode{"index_representation_overflow", FoldFailureKind::Magnitude},
    FoldFailureCode{"scalar_fold_index_overflow", FoldFailureKind::Magnitude},
    FoldFailureCode{"scalar_fold_prerequisite", FoldFailureKind::LaunchWide},
    FoldFailureCode{"scalar_fold_cycle", FoldFailureKind::KeyText},
    FoldFailureCode{"scalar_fold_internal_binding", FoldFailureKind::Internal},
    FoldFailureCode{"source_dependency_cycle", FoldFailureKind::Internal}, // IndexedSourceLimit
    FoldFailureCode{"scalar_fold_unit_parameter_unsupported", FoldFailureKind::Lifting},
    FoldFailureCode{"scalar_fold_mma_descriptor", FoldFailureKind::Dynamic}, // LayoutError and two literal fail() sites
};
inline bool FoldFailureContextFree(std::string_view code) noexcept {
  for (const auto &entry : kFoldFailureCodes)
    if (entry.code == code) return entry.kind == FoldFailureKind::ContextFree;
  return false;
}
} // namespace sentryward::value_analysis
