#pragma once

#include <chrono>
#include <cstddef>
#include <string>
#include <span>
#include <memory>
#include <optional>
#include <vector>

#include "sentryward/scalar_fold/graph.h"

namespace sentryward::equivalence {
class CertifiedBranchScope;

enum class Status { Equivalent, NotProven, Unsupported };

struct Options {
  std::size_t max_nodes = 1000000;
  std::size_t max_iterations = 32;
  std::chrono::milliseconds timeout{1200000};
  // At mathematical OUTPUT uses only, ignore floating quantization/saturation.
  // Internal values, predicates, addresses and exact/bit consumers keep their
  // typed conversion semantics; this never globally aliases a converted class.
  bool ideal_numeric_outputs = false;
  std::uint32_t outer_binders = 0;
  // Validated domains for this comparison only. Empty means unbounded/unknown;
  // Fold binder extents are never inferred from a global index number.
  std::vector<std::uint64_t> outer_index_extents;
  // Zero leaves the existing comparison budget unchanged. Small update
  // compilations additionally cap all Engine work, not just enode count.
  std::size_t max_work = 0;
  // Explicit stored-byte comparison at validated output boundaries only.
  // Zero disables the certificate; it never merges internal value classes.
  std::uint64_t storage_output_bytes = 0;
  // Separate hash-bound original reached-scope evidence. Only consumer-local
  // Numeric implications may use it; exact predicates/addresses/bits retain
  // their source semantics. Null preserves the ordinary comparison path.
  std::shared_ptr<const CertifiedBranchScope> certified_branch_scope = {};
};

// Precompile a bounded, parameterized local expression. Parameters are opaque:
// callers bind them directly and validate domain/allocation/binder relations
// separately. No e-matching, import, union, rebuild or saturation at lookup.
struct Result;
class FrozenUpdateEgraph {
 public:
  struct Lookup { std::optional<std::size_t> eclass; std::size_t work=0; };
  FrozenUpdateEgraph(const scalar_fold::Graph&, std::span<const scalar_fold::Ref>,
                    Options, std::span<const std::string> positive_parameters = {});
  ~FrozenUpdateEgraph();
  FrozenUpdateEgraph(FrozenUpdateEgraph&&) noexcept;
  FrozenUpdateEgraph& operator=(FrozenUpdateEgraph&&) noexcept;
  // O(1) access to an already imported, normalized root. No new proof work.
  Lookup importedRoot(std::size_t index) const;
  Lookup lookup(const scalar_fold::Graph&, scalar_fold::Ref,
                std::size_t max_work=64) const;
  const Result& compilation() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct Statistics {
  std::size_t nodes = 0;
  std::size_t iterations = 0;
  std::size_t rewrite_unions = 0;
  std::size_t congruence_unions = 0;
  // Monotone e-class summaries; counters cover transfer evaluations, not
  // saturation complexity. No logical Fold elements are enumerated here.
  std::size_t property_evaluations = 0;
  std::size_t property_updates = 0;
};

struct Result {
  Status status = Status::NotProven;
  std::string reason;
  Statistics statistics;
  // Nonempty means conditional equivalence under separately captured original
  // branch choices; the fingerprint identifies exact sites/member provenance.
  std::string conditional_scope_fingerprint = {};
};

// Exact conversion and packed-field rewrites; no equality saturation.
scalar_fold::Ref normalizeExactUse(const scalar_fold::Graph&,scalar_fold::Ref);
// The mutable use path can emit at most one constant/ShiftRight pair after
// four local unsigned packing/widening heads; conversion tails stay explicit.
scalar_fold::Ref normalizeExactUse(scalar_fold::Graph&,scalar_fold::Ref);

// Parameter/allocation identities must already be aligned by the caller.
// Numerical arithmetic uses ideal reals; integer and address operations stay exact.
Result compareExpressions(const scalar_fold::Graph&, scalar_fold::Ref,
                          const scalar_fold::Graph&, scalar_fold::Ref,
                          Options = {});

// Corresponding roots share one import, rewrite run and resource budget.
Result compareExpressionSets(const scalar_fold::Graph&, std::span<const scalar_fold::Ref>,
                             const scalar_fold::Graph&, std::span<const scalar_fold::Ref>,
                             Options = {});

}  // namespace sentryward::equivalence
