#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "sentryward/algorithm/memory/constant_address_model.h"

namespace sentryward::algorithm {

// The internal table-lookup symbol of an input object's elements under tensor_properties:
// an address offset term {"operator":"gather","symbol":S,"args":[element byte offset]}.
inline std::string TensorPropertySymbol(std::string_view object, std::string_view element_type) {
  return "tensor_property:" + std::string(object) + ":" + std::string(element_type);
}

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
  // The offset difference is a constant plus a multiple of its coefficients'
  // gcd, and no such value lies in the overlap interval.
  std::size_t congruence_disjoint = 0;
  // Two different elements of one injective input object, with opposite
  // coefficients, differ by at least the coefficient.
  std::size_t injective_disjoint = 0;
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
  // `tensor_properties` are normalized input records (object, element_type, lower_bound,
  // upper_bound, injective); they bound the TensorPropertySymbol lookups (default none).
  explicit SymbolicAddressModel(const std::vector<ConstantAddressBinding>& bindings,
                                std::string_view address_invariants = "[]",
                                SymbolicSolverOptions solver = {},
                                std::size_t cache_capacity = 4096,
                                std::string_view tensor_properties = "[]");
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
  // Also a TMA region whose offset's symbolic part is tensor-property lookups (region_axes set).
  [[nodiscard]] std::optional<FamilyFootprint> crossUnitFootprint(const MemoryAccessFamily&) override;
  // Footprints whose symbolic suffix is tensor-property lookups (FamilyFootprint::gathers): each
  // family is its lowest member's address plus one progression axis, decided like a pair of
  // bindings. Unknown when either footprint has another suffix or is not affine.
  [[nodiscard]] OverlapResult translatedFamilyOverlap(const MemoryAccessFamily& a, const FamilyFootprint& x,
                                                      const MemoryAccessFamily& b, const FamilyFootprint& y) override;
  [[nodiscard]] SymbolicAddressStats stats() const noexcept;
  // Whether this build links an SMT backend, and its name ("z3" or "unavailable").
  [[nodiscard]] static bool SolverAvailable() noexcept;
  [[nodiscard]] static const char* SolverName() noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sentryward::algorithm
