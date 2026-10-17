#pragma once
#include "sentryward/algorithm/io/source_json.h"
#include "sentryward/frontend/ptx/execution.h"
#include <functional>
#include <array>
#include <memory>
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
};
// One CTA's execution/provenance is consumed at a time; model storage is
// native.
class ModelLowerer {
public:
  explicit ModelLowerer(const Module &, const Function &,
                        std::function<void()> poll = {}, bool retain_matrix_layout = false,
                        bool atomics = false);
  ~ModelLowerer();
  using ValueAddresses = std::function<std::optional<nlohmann::json>(
      const LaneTrace&, const ExecutedInstruction&, std::string_view)>;
  void AddCta(std::uint32_t, Coordinates, const ExecutionProgram &, ValueAddresses = {});
  [[nodiscard]] const algorithm::SourceProgram &currentProgram() const noexcept;
  [[nodiscard]] LoweringResult Finish();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace sentryward::frontend::ptx
