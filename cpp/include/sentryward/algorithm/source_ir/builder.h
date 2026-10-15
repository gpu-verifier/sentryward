#pragma once

#include <map>
#include <set>
#include <span>
#include "sentryward/algorithm/source_ir/program.h"

namespace sentryward::algorithm {

// Builds dense instruction IDs and explicit PO once, as semantic records arrive.
// Random access is by ID; callers must not retain references across append.
class SourceProgramBuilder final {
 public:
  SourceProgram program;
  [[nodiscard]] InstId append(ActorId actor, SourceLocation location, SourceOp op,
                              std::optional<PtxOrigin> origin = std::nullopt,
                              bool program_order = true);
  void appendGeneratedThread(ActorId actor,std::size_t count,
      std::function<SourceInstruction(std::size_t)> generate,
      std::optional<std::pair<std::uint64_t,std::uintptr_t>> cohort={},
      SourceInstructions::AffineGenerator columns={},const void* family=nullptr);
  [[nodiscard]] SourceInstruction& editInstruction(InstId id);
  [[nodiscard]] SourceInstruction instruction(InstId id) const;
  void initializeNamedBarrier(BarrierLifetimeId barrier, std::uint32_t expected,
                             std::span<const InstId> uses);
  void alignWarp(std::span<const std::pair<InstId, InstId>> calls,
                 bool join_completions, std::size_t anchor = 0);
  void finishEdges();
 private:
  std::map<ActorId, InstId> first_, last_;
  std::map<ActorId, std::vector<InstId>> initializers_;
  std::set<std::pair<InstId, InstId>> replaced_edges_;
};
}  // namespace sentryward::algorithm
