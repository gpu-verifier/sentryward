#pragma once

#include <cstddef>
#include <span>

namespace sentryward::equivalence::epoch_work_allowance {
// Resource policy only: these values never admit a value, domain, body or
// predicate. Existing callers retain the original default allowance.
inline constexpr std::size_t default_work=1000000;
inline constexpr std::size_t maximum_work=32000000;
inline constexpr std::size_t default_retained_headers=4096,maximum_retained_headers=65536;
inline constexpr std::size_t metadata_bytes_per_header=4096;
// Fixed bookkeeping units for one stored header, one stored child edge, and
// four bounded metadata hash/copy visits across source attachment, local
// validation, scoped body relations and output connection. These conservative
// weights are a resource heuristic, NOT an upper bound on proof cost. One
// monotone ledger still debits every actual visit; no query/root resets it.
inline constexpr std::size_t header_work=64,edge_work=16,metadata_byte_work=4;
struct Size {
  // Saturating physical counts only. No logical Fold/output extent belongs in
  // these fields; metadata is independently bounded for EVERY stored header.
  std::size_t headers=0,edges=0,bounded_metadata_bytes=0;
  bool saturated=false;
};
struct Allowance {
  std::size_t max_work=default_work;
  Size physical;
};
// Resource capacity only, from the ACTUAL stored header count. Does not
// create a record, enumerate logical members or admit a value/proof/scope.
// Standalone algorithm defaults remain4096; opt-in callers use this result.
std::size_t retainedHeaderCapacity(const Size&) noexcept;
class PhysicalEpoch {
 public:
  // The caller supplies scalar lengths of this actual immutable header's
  // metadata, not strings. At most eight metadata fields are inspected; the
  // contribution is capped at4096 bytes even for huge/unsupported headers.
  // There is no traversal, allocation, logical member loop or budget reset.
  void addHeader(std::size_t stored_edges,std::span<const std::size_t> metadata_lengths={}) noexcept;
  // The caller floor is clamped to [default_work,maximum_work]. Prefix uses
  // its old1M floor; optional linear coverage uses its old2M floor.
  Allowance allowance(std::size_t floor=default_work) const noexcept;
 private:
  Size size_;
};
} // namespace sentryward::equivalence::epoch_work_allowance
