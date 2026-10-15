#pragma once

#include <compare>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "sentryward/frontend/ptx/execution.h"

namespace sentryward::frontend::ptx {
struct TmemAllocationId {
  std::uint32_t warp{}, instruction_index{};
  std::uint64_t occurrence{};
  auto operator<=>(const TmemAllocationId &) const = default;
  [[nodiscard]] std::string ResourceName() const;
};
struct TmemPointer {
  TmemAllocationId allocation;
  std::uint32_t encoded_offset{};
  auto operator<=>(const TmemPointer &) const = default;
  [[nodiscard]] std::optional<TmemPointer>
  AddEncoded(const Integer &delta) const;
  [[nodiscard]] std::string ResourceName() const {
    return allocation.ResourceName();
  }
};
struct TmemAtomSite {
  std::uint32_t warp{}, instruction_index{};
  std::uint64_t occurrence{};
  std::uint32_t ptx_line{};
  std::string opcode;
  auto operator<=>(const TmemAtomSite &) const = default;
  [[nodiscard]] TmemAllocationId Key() const {
    return {warp, instruction_index, occurrence};
  }
};
struct TmemAllocation {
  TmemAllocationId identifier;
  TmemAtomSite site;
  Integer shared_slot;
  std::uint32_t columns{}, cta_group = 1;
};
struct TmemUse {
  TmemAtomSite site;
  TmemAllocationId allocation;
  std::uint32_t encoded_offset{};
  std::string operation;
  auto operator<=>(const TmemUse &) const = default;
};
struct TmemPublicationRead {
  TmemAtomSite site;
  TmemAllocationId allocation;
  auto operator<=>(const TmemPublicationRead &) const = default;
};
struct TmemPublicationCandidates {
  TmemAtomSite site;
  Integer shared_slot;
  std::vector<TmemAllocationId> allocations;
};
struct TmemLifetimeIssue {
  std::string code, message;
  TmemAtomSite site;
  auto operator<=>(const TmemLifetimeIssue &) const = default;
};
struct TmemOrderObligation {
  std::string kind;
  TmemAllocationId allocation;
  TmemAtomSite source, target;
  std::string message;
  auto operator<=>(const TmemOrderObligation &) const = default;
};
using ResolvedTmemOperand = std::variant<std::monostate, Integer, TmemPointer>;
using TmemLaneExecutions =
    std::vector<std::pair<std::uint32_t, const ExecutedInstruction *>>;
struct TmemAnalysisOptions {
  std::map<TmemAllocationId, TmemAllocationId> publication_resolution;
  bool allow_ambiguous_publications = false, allow_cta_group2 = false;
  std::function<void()> poll;
};
// ExecutionProgram and its lane records must remain alive and unmoved while
// this index is queried. Snapshots retain only pointer registers referenced by
// each static tcgen instruction, instead of copying a whole register map per
// event.
struct TmemProvenanceIndex {
  const ExecutionProgram *program = nullptr;
  std::vector<TmemAllocation> allocations;
  std::set<TmemAllocationId> zero_based_allocations;
  using Snapshot = std::vector<std::pair<std::string_view, TmemPointer>>;
  std::unordered_map<const ExecutedInstruction *, Snapshot>
      register_state_before;
  std::vector<TmemPublicationRead> publication_reads;
  std::vector<TmemPublicationCandidates> publication_candidates;
  std::vector<TmemLifetimeIssue> issues;
  [[nodiscard]] ResolvedTmemOperand Resolve(std::uint32_t warp,
                                            const ExecutedInstruction &,
                                            std::string_view operand,
                                            std::uint32_t lane) const;
  [[nodiscard]] ResolvedTmemOperand Resolve(std::uint32_t warp,
                                            const ExecutedInstruction &,
                                            std::size_t operand,
                                            std::uint32_t lane) const;
  [[nodiscard]] ResolvedTmemOperand
  ResolveUniform(std::uint32_t warp, const TmemLaneExecutions &,
                 std::string_view operand) const;
  [[nodiscard]] ResolvedTmemOperand ResolveUniform(std::uint32_t warp,
                                                   const TmemLaneExecutions &,
                                                   std::size_t operand) const;
  [[nodiscard]] std::optional<TmemPointer>
  NormalizeTaddr(const ResolvedTmemOperand &) const;
  [[nodiscard]] std::optional<TmemPointer>
  ResolveUniformTaddr(std::uint32_t warp, const TmemLaneExecutions &lanes,
                      std::string_view operand) const {
    return NormalizeTaddr(ResolveUniform(warp, lanes, operand));
  }
  [[nodiscard]] std::optional<TmemPointer>
  ResolveUniformTaddr(std::uint32_t warp, const TmemLaneExecutions &lanes,
                      std::size_t operand) const {
    return NormalizeTaddr(ResolveUniform(warp, lanes, operand));
  }
  [[nodiscard]] const TmemAllocation *AllocationById(TmemAllocationId) const;
};
struct TmemLifetimeAnalysis {
  TmemProvenanceIndex provenance;
  std::vector<TmemUse> uses;
  std::vector<std::pair<TmemAtomSite, TmemAllocationId>> deallocations;
  std::vector<TmemLifetimeIssue> issues;
  std::vector<TmemOrderObligation> ordering_obligations;
  bool cluster_pairing_required = false;
  [[nodiscard]] bool IdentityProven() const { return issues.empty(); }
  [[nodiscard]] bool Proven() const {
    return IdentityProven() && ordering_obligations.empty() &&
           !cluster_pairing_required;
  }
};
struct TmemSiteCertificate {
  TmemAtomSite site;
  std::string operation;
  std::optional<TmemAllocationId> allocation;
  // Shared analysis owns the issue and obligation inventories, avoiding their
  // repeated allocation at every lifetime atom.
  const TmemLifetimeAnalysis *analysis = nullptr;
  bool representable = false, fully_proven = false;
};
[[nodiscard]] TmemProvenanceIndex
BuildTmemProvenance(const ExecutionProgram &, const TmemAnalysisOptions & = {});
[[nodiscard]] TmemLifetimeAnalysis
AnalyzeTmemLifetimes(const ExecutionProgram &,
                     const TmemAnalysisOptions & = {});
[[nodiscard]] std::map<TmemAllocationId, TmemSiteCertificate>
BuildTmemSiteCertificates(const ExecutionProgram &,
                          const TmemLifetimeAnalysis &);
[[nodiscard]] nlohmann::json TmemLifetimeJson(const TmemLifetimeAnalysis &);
} // namespace sentryward::frontend::ptx
