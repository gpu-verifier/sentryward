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
#include "sentryward/frontend/ptx/unit_facts.h"

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
// (lane id, record with its values) per lane of one atom: the Expanded element,
// or (T3) a Full read the analysis holds.
using TmemLaneExecutions = std::vector<std::pair<std::uint32_t, RecordView>>;
struct TmemAnalysisOptions {
  std::map<TmemAllocationId, TmemAllocationId> publication_resolution;
  bool allow_ambiguous_publications = false, allow_cta_group2 = false;
  std::function<void()> poll;
};
// The facts, their program and its lane records must remain alive and
// unmoved while this index is queried. Snapshots retain only pointer registers
// referenced by each static tcgen instruction, instead of copying a whole
// register map per event. They are keyed by the position of the record they
// precede. Resolve looks one up for an Expanded view and, when `positioned`
// (T3, which reads lanes without expanding them), for a Full or Slice read of
// that position: the views lowering holds where an expanded lane gave it the
// Expanded element. Other kinds (a prototype, a copy) get none.
struct TmemProvenanceIndex {
  const FactSource *facts = nullptr;
  bool positioned = false;
  std::vector<TmemAllocation> allocations;
  std::set<TmemAllocationId> zero_based_allocations;
  using Snapshot = std::vector<std::pair<std::string_view, TmemPointer>>;
  std::unordered_map<RecordRef, Snapshot> register_state_before;
  std::vector<TmemPublicationRead> publication_reads;
  std::vector<TmemPublicationCandidates> publication_candidates;
  std::vector<TmemLifetimeIssue> issues;
  [[nodiscard]] ResolvedTmemOperand Resolve(std::uint32_t warp,
                                            const RecordView &,
                                            std::string_view operand,
                                            std::uint32_t lane) const;
  [[nodiscard]] ResolvedTmemOperand Resolve(std::uint32_t warp,
                                            const RecordView &,
                                            std::size_t operand,
                                            std::uint32_t lane) const;
  [[nodiscard]] std::optional<TmemPointer>
  NormalizeTaddr(const ResolvedTmemOperand &) const;
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
// T1 expands every lane of `facts`, at the same points as the per-lane loops
// did before the expansion was made explicit. T3 expands none: its passes walk
// metadata and read a record's values only where a step uses them.
[[nodiscard]] TmemLifetimeAnalysis
AnalyzeTmemLifetimes(const CtaFacts &, ExpansionSite,
                     const TmemAnalysisOptions & = {});
} // namespace sentryward::frontend::ptx
