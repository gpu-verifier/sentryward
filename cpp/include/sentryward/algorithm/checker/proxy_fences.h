#pragma once
// Cross-proxy ordering (PTX ISA 9.5, "Proxies", "Async Proxy" and fence.proxy). Accesses through
// different memory proxies to the same bytes are ordered only through a proxy fence. This pass checks
// the pairs that the race check counts as ordered: a generic-proxy access g and an async-proxy access
// a (TMA, bulk copy and reduction, tcgen05.mma and tcgen05.cp shared operands) on the same shared or
// global bytes, at least one a write, with g happening before a. Such a pair is ordered only when
// some fence.proxy.async F covering the state space has g -> F -> a in happens-before.
//
// The fence may be in any thread. g's thread (the first fence g reaches) and the thread that issued a
// (the last fence reaching a) are searched first; the pairs left are decided over the happens-before
// graph by bitset passes that find a fence in any thread on a path between them.
//
// Async before generic needs no explicit fence here: the completion of an async operation, which is
// how happens-before orders its accesses before later ones, is followed by an implicit generic-async
// proxy fence (PTX ISA for cp.async.bulk, cp.reduce.async.bulk and wgmma). The same is assumed for
// tcgen05.mma and tcgen05.cp completion observed through tcgen05.commit, which the ISA does not state.
// Async/async pairs and tensormap-proxy reads (checked by the tensormap locality rule) are not checked.
// A .shared::cta fence is taken to cover all shared memory, including a peer CTA's.
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "sentryward/algorithm/analysis/hb_reachability.h"
#include "sentryward/algorithm/base/result.h"
#include "sentryward/algorithm/memory/access.h"
#include "sentryward/algorithm/memory/model.h"
#include "sentryward/algorithm/memory/overlap.h"
#include "sentryward/algorithm/source_ir/program.h"

namespace sentryward::algorithm {

inline constexpr std::size_t kMaxProxyFenceSites = 32;

struct ProxyFenceStats final {
  std::size_t async_accesses = 0, fences = 0;
  // Overlapping generic/async pairs with the generic access first in happens-before.
  std::size_t ordered_pairs = 0;
  // Those without a covering fence, by the generic access's and the async access's modes.
  std::size_t read_after_write = 0, write_after_write = 0, write_after_read = 0;
  // Pairs whose overlap or state space is unknown and that have no fence covering every space.
  std::size_t unresolved = 0;

  [[nodiscard]] std::size_t missing() const noexcept {
    return read_after_write + write_after_write + write_after_read;
  }
};

struct ProxyFenceFinding final {
  MemoryAccess generic;
  MemoryAccess async;
  OverlapEvidence overlap;
};

// Pairs without a fence, grouped by kind and by the PTX sites of the two instructions.
struct ProxyFenceSite final {
  const char* kind; // read_after_write, write_after_write or write_after_read
  MemoryAccess generic, async; // the first pair at these sites
  std::optional<std::uint32_t> generic_ptx, async_ptx; // PTX instruction indices
  std::size_t count = 0;
};

struct ProxyFenceResult final {
  ProxyFenceStats stats;
  std::vector<ProxyFenceSite> sites; // at most kMaxProxyFenceSites, in order of first finding
  std::optional<ProxyFenceFinding> first;
  std::optional<std::pair<MemoryAccess, MemoryAccess>> first_unresolved;
};

// O(N) to split the N accesses and one storage-filter query per distinct async address; the model's
// candidate streams over the async reads with the generic writes on their storage, and over the async
// writes with the generic accesses on theirs (no async/async pair is enumerated); then batched
// reachability queries: one per candidate pair, O(log F) per distinct access in a thread with F
// fences, and at most two per ordered pair; then, for pairs still without a fence, O(F / 64 x (V + E))
// over the V events and E edges between their ranks.
[[nodiscard]] Result<ProxyFenceResult, MemoryAnalysisError> CheckProxyFences(
    const SourceProgram& source, const HappensBeforeGraph& graph, StaticHbQueries& queries,
    MemoryModel& memory, MemoryAccessView accesses);

}  // namespace sentryward::algorithm
