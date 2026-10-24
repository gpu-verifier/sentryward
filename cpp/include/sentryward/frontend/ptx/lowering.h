#pragma once
#include "sentryward/algorithm/io/source_json.h"
#include "sentryward/frontend/ptx/execution.h"
#include "sentryward/frontend/ptx/unit_classes.h"
#include "sentryward/frontend/ptx/unit_facts.h"
#include <functional>
#include <array>
#include <memory>
namespace sentryward::algorithm { struct UnitQuotient; }
namespace sentryward::frontend::ptx {
enum class MmaOperandRole { A, B, D, ScaleA, ScaleB, SparseMetadata };
struct MmaValueAccess {
  std::uint32_t effect, access;
  MmaOperandRole role;
  std::uint32_t cta_part;
};
struct MmaValueLayout {
  algorithm::InstId instruction;
  std::uint64_t descriptor, a_base, b_descriptor, d_base;
  bool a_in_tmem;
  std::uint32_t cta_group;
  std::vector<MmaValueAccess> accesses;
  std::array<std::uint64_t,2> scale_bases{};
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
// native.
class ModelLowerer {
public:
  explicit ModelLowerer(const Module &, const Function &,
                        std::function<void()> poll = {}, bool retain_matrix_layout = false,
                        bool atomics = false);
  ~ModelLowerer();
  // The address bridge: (lane of the CTA's facts, record, operand text).
  using ValueAddresses = std::function<std::optional<nlohmann::json>(
      LaneRef, const RecordView&, std::string_view)>;
  // The CTA id is facts.Linear(); the coordinates are facts.frame().SelfCoordinates().
  void AddCta(const CtaFacts &, ValueAddresses = {});
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
