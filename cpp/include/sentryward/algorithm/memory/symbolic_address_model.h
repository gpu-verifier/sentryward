#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "sentryward/algorithm/memory/constant_address_model.h"

namespace sentryward::algorithm {

struct SymbolicSolverOptions final {
  // Budget per pair across its environment and overlap checks; must be positive.
  unsigned timeout_ms = 100;

  friend bool operator==(const SymbolicSolverOptions&, const SymbolicSolverOptions&) = default;
};

struct SymbolicAddressStats final {
  // Pairs the constant model left Unknown and this model refined.
  std::size_t refined = 0;
  std::size_t cache_hits = 0;
  std::size_t layer0_disjoint = 0;
  // Decided from the address-invariant environment or a constant offset
  // difference, without calling the solver.
  std::size_t ground_disjoint = 0;
  std::size_t ground_overlap = 0;
  std::size_t solver_queries = 0;
  std::size_t solver_unsat = 0;
  std::size_t solver_sat = 0;
  std::size_t solver_unknown = 0;

  friend bool operator==(const SymbolicAddressStats&, const SymbolicAddressStats&) = default;
};

// Symbolic dynamic-address overlap (dynamic.pdf, layers 0 and 2).
//
// Addresses are allocation-tagged affine forms over interned atoms. Pairs the
// constant model decides (resolved constant footprints, registers, constant
// allocation separation) keep its exact answers and candidate streams. Pairs
// it leaves Unknown are refined by allocation separation (layer 0) and by a
// ground SMT query over the address-invariant environment Gamma (layer 2).
// Disjoint holds for every valuation satisfying Gamma; inconclusive queries
// stay Unknown. See docs/assumptions.md for the binding and invariant formats.
class SymbolicAddressModel final : public MemoryModel {
 public:
  // `address_invariants` is a JSON array of Gamma constraints (default empty).
  explicit SymbolicAddressModel(const std::vector<ConstantAddressBinding>& bindings,
                                std::string_view address_invariants = "[]",
                                SymbolicSolverOptions solver = {},
                                std::size_t cache_capacity = 4096);
  ~SymbolicAddressModel() override;
  SymbolicAddressModel(SymbolicAddressModel&&) noexcept;
  SymbolicAddressModel& operator=(SymbolicAddressModel&&) noexcept;
  SymbolicAddressModel(const SymbolicAddressModel&) = delete;
  SymbolicAddressModel& operator=(const SymbolicAddressModel&) = delete;

  [[nodiscard]] std::optional<MemoryAnalysisError> validate() const override;
  [[nodiscard]] OverlapResult mayOverlap(const MemoryAccess& first,
                                         const MemoryAccess& second) override;
  // Streams come from the constant model: they retain every pair it leaves
  // Unknown, a superset of the pairs this model does not prove disjoint.
  [[nodiscard]] std::unique_ptr<MemoryStorageFilter> storageFilter(
      MemoryAccessView accesses) override;
  [[nodiscard]] std::unique_ptr<MemoryConflictCandidates> conflictCandidates(
      MemoryAccessView accesses) override;
  [[nodiscard]] std::unique_ptr<MemoryConflictCandidates> conflictOrder(
      MemoryAccessView accesses, std::span<const std::size_t> ranks) override;
  [[nodiscard]] std::unique_ptr<MemoryConflictCandidates> conflictCandidatesWithExemptions(
      MemoryAccessView accesses, std::span<const std::size_t> ranks,
      std::span<const AccessId> exempt_writes) override;

  [[nodiscard]] const ConstantAddressModel& constantModel() const noexcept;
  [[nodiscard]] std::optional<FamilyFootprint> familyFootprint(const MemoryAccessFamily&) override;
  [[nodiscard]] SymbolicAddressStats stats() const noexcept;
  // Whether this build links an SMT backend, and its name ("z3" or "unavailable").
  [[nodiscard]] static bool SolverAvailable() noexcept;
  [[nodiscard]] static const char* SolverName() noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sentryward::algorithm
