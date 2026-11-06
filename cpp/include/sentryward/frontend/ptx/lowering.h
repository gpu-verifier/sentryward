#pragma once
#include <initializer_list>
#include <string_view>
#include "sentryward/algorithm/io/source_json.h"
#include "sentryward/frontend/ptx/execution.h"
#include "sentryward/frontend/ptx/unit_classes.h"
#include "sentryward/frontend/ptx/unit_facts.h"
#include <functional>
#include <array>
#include <bitset>
#include <memory>
namespace sentryward::algorithm { struct UnitQuotient; }
namespace sentryward::frontend::ptx {
enum class MmaOperandRole { A, B, D, ScaleA, ScaleB, SparseMetadata };
struct MmaValueAccess {
  std::uint32_t effect, access;
  MmaOperandRole role;
  std::uint32_t cta_part;
};
// Lowering diagnostics that are reachable undefined behavior (result undefined_behavior):
// the lowering decided them from the concrete values of an executed instruction, and each
// violates a PTX ISA sentence that is explicit ("undefined") or a "must" (owner rule
// 2026-11-05: PTX is followed strictly). docs/output-json.md lists the sentences.
inline bool LoweringUndefinedBehavior(std::string_view code) noexcept {
  for (const auto known : {"cp_async_source_size", "tmem_lane_alignment_mismatch",
                           "invalid_bulk_transfer_size", "bulk_reduction_size", "bulk_reduction_alignment",
                           "tmem_address_not_uniform", "tmem_aligned_lanes_missing",
                           "tmem_lifetime_lanes_missing", "tmem_invalid_alloc_size",
                           "tmem_nonuniform_alloc_size", "tmem_nonuniform_dealloc_size",
                           "tmem_increasing_alloc_size", "mma_sparsity_version", "bulk_copy_alignment",
                           "tmem_alloc_destination_outside_shared_window", "tmem_alloc_after_relinquish",
                           "tcgen05_cta_group_mixed", "aligned_lanes_missing"})
    if (code == known) return true;
  return false;
}
// The diagnostic that decides a stopped lowering's result: the first reachable undefined behavior
// (LoweringUndefinedBehavior), wherever it is, else the first diagnostic. `diagnostics` is a
// nonempty array in lowering order. O(diagnostics).
inline std::size_t DecidingLoweringDiagnostic(const nlohmann::json &diagnostics) {
  for (std::size_t k = 0; k < diagnostics.size(); ++k)
    if (LoweringUndefinedBehavior(diagnostics[k].value("code", std::string()))) return k;
  return 0;
}
// Whether lowering an executed instruction registers its lane at the instruction's .aligned site,
// whose warp participation ModelLowerer checks: barrier.cluster .aligned, tcgen05.wait, tcgen05.ld
// and tcgen05.st, the tcgen05 lifetime operations, tensormap.cp_fenceproxy and the per-lane matrix
// accesses (ldmatrix, stmatrix). A lane whose lowering of such an instruction fails executed it.
bool RegistersAlignedLanes(const Instruction &);
// The participation of a full warp in one occurrence of an .aligned instruction: lanes that
// neither lowered it (`lowered`) nor executed it and failed to lower it (`failed`) are missing,
// which violates .aligned (undefined behavior); with no lane missing but a failed one, the warp's
// participation is incomplete, not undefined. O(32).
struct AlignedParticipation {
  std::vector<std::uint32_t> missing;
  bool incomplete = false;
};
inline AlignedParticipation AlignedSiteParticipation(const std::bitset<32> &lowered, const std::bitset<32> &failed) {
  AlignedParticipation result;
  for (std::uint32_t lane = 0; lane < 32; ++lane)
    if (!lowered.test(lane) && !failed.test(lane)) result.missing.push_back(lane);
  result.incomplete = result.missing.empty() && (failed & ~lowered).any();
  return result;
}
struct MmaValueLayout {
  algorithm::InstId instruction;
  std::uint64_t descriptor, a_base, b_descriptor, d_base;
  bool a_in_tmem;
  std::uint32_t cta_group;
  std::vector<MmaValueAccess> accesses;
  std::array<std::uint64_t,2> scale_bases{};
  // Weight-stationary (.ws): D uses the 128/M lane-group layouts D, E and G.
  bool ws = false;
};
struct LoweringResult {
  algorithm::SourceArtifact artifact;
  nlohmann::json metadata;
  nlohmann::json diagnostics = nlohmann::json::array();
  std::vector<MmaValueLayout> mma_layouts;
  // The consumed CTAs' tensormap.cp_fenceproxy destinations, checked against
  // input tensor maps in run.cpp.
  std::vector<TensormapRegion> tensormap_destinations;
};
// One CTA's execution/provenance is consumed at a time; model storage is
// native. `threads` (--sync-threads): with more than one, a CTA's lanes are
// lowered on that many worker threads and applied in lowering order; the result
// is the one-thread result for every thread count.
class ModelLowerer {
public:
  explicit ModelLowerer(const Module &, const Function &,
                        std::function<void()> poll = {}, bool retain_matrix_layout = false,
                        bool atomics = false, std::size_t threads = 0);
  ~ModelLowerer();
  // The address bridge: (lane of the CTA's facts, record, operand text).
  using ValueAddresses = std::function<std::optional<nlohmann::json>(
      LaneRef, const RecordView&, std::string_view)>;
  // The CTA id is facts.Linear(); the coordinates are facts.frame().SelfCoordinates().
  // `lane_workers`: whether lanes may be lowered on worker threads; false when a
  // bridge query may still build state the workers read (an on-demand value view).
  void AddCta(const CtaFacts &, ValueAddresses = {}, bool lane_workers = true);
  [[nodiscard]] const algorithm::SourceProgram &currentProgram() const noexcept;
  // Cross-CTA class mode, before the first AddCta: only representative CTAs are added, and
  // `quotient` (complete by Finish) says how many units each stands for. Lowering keeps per-CTA
  // occurrence counts and weights them at Finish (cta_count, thread_count, reached_opcodes), weights
  // the Finish-time diagnostics by their unit, re-renders per-CTA violation texts for every member,
  // checks unit closure and records the required orders' Add/Finish boundary.
  void TrackUnits(algorithm::UnitQuotient *quotient) noexcept;
  [[nodiscard]] LoweringResult Finish();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace sentryward::frontend::ptx
