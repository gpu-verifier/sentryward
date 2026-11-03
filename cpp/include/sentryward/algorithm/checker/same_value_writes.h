#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "sentryward/algorithm/checker/memory_race_checker.h"
#include "sentryward/algorithm/memory/family_access.h"

namespace sentryward::algorithm {
class ActorWeights;

// Same-value write obligations (docs/assumptions.md, "Same-value writes"). Two unordered plain
// generic-proxy writes that cover exactly the same bytes form a uniform-size data race. The race
// checker records such a pair as an obligation and continues; the value analysis must prove that
// both store the same bytes. Read/write pairs and partially overlapping writes remain races.
class SameValueWrites final {
 public:
  // Warning samples (sample, sampleRelation) need `ctas`, the CTA of each actor. `weights` (class
  // mode, else null) weights a warning by the size of its CTA's class; at most `max_samples`
  // samples are kept, and at least one. Without `ctas` only the predicates are available.
  explicit SameValueWrites(const SourceProgram& source, const ActorCtas* ctas = nullptr,
                           const ActorWeights* weights = nullptr, std::size_t max_samples = 0)
      : source_(&source), ctas_(ctas), weights_(weights), max_samples_(max_samples) {}
  // A plain (non-atomic) write of a memory endpoint through the generic proxy: st (not st.async,
  // st.bulk or st.mmio), stmatrix, or a non-bulk cp.async. One opcode projection per call.
  [[nodiscard]] bool eligible(const MemoryAccess& access) const;
  // Every member is such a write: one opcode projection per member instruction.
  [[nodiscard]] bool eligible(const MemoryAccessFamily& family) const;
  // Two distinct eligible writes whose overlap is exactly each footprint (same bytes, same width).
  [[nodiscard]] bool pair(const MemoryAccess& first, const MemoryAccess& second,
                          const OverlapEvidence& overlap) const;
  // pair() without its distinct-event test: two eligible writes of one width whose overlap, of
  // `overlap_bytes`, is each footprint.
  [[nodiscard]] bool sameBytes(const MemoryAccess& first, const MemoryAccess& second,
                               std::uint64_t overlap_bytes) const;
  // Warning sample: one per pair of write sites (source locations) and CTA, counted with the
  // weight of that CTA. The CTA is that of the write whose (site, CTA) is the lesser, so the count
  // does not depend on the order in which a search meets a pair. O(log sites).
  void sample(const MemoryAccess& first, const MemoryAccess& second, const OverlapEvidence& overlap,
              MemoryRaceResult& result);
  // sample() for every pair of `relation` between members of `first` and `second` (an accepted
  // obligation of the family search), starting with relation.witness(), without enumerating the
  // pairs: the members of a family share their site, and its members in one CTA form one range,
  // so one pair per CTA range decides every count. `evidence(x, y)` is the overlap of members x
  // and y of a kept sample. False, with nothing counted, when those ranges are not known
  // (ActorCtas::contiguous). O((member runs + CTA ranges) log members + CTA ranges log sites).
  [[nodiscard]] bool sampleRelation(const MemoryAccessFamily& first, const MemoryAccessFamily& second,
                                    const MemberRelation& relation,
                                    const std::function<OverlapEvidence(std::size_t, std::size_t)>& evidence,
                                    MemoryRaceResult& result);

 private:
  // Counts the pair of sites and CTA of two writes (see sample) when it is new.
  bool count(const SourceLocation& first, ActorId first_actor, const SourceLocation& second,
             ActorId second_actor, MemoryRaceResult& result);
  // Whether a new count is also kept as a sample.
  [[nodiscard]] bool keeps(const MemoryRaceResult& result) const;
  const SourceProgram* source_;
  const ActorCtas* ctas_;
  const ActorWeights* weights_;
  std::size_t max_samples_;
  std::set<std::tuple<std::string, std::uint32_t, std::uint32_t, std::string, std::uint32_t,
                      std::uint32_t, std::uint32_t>> sites_;
};

}  // namespace sentryward::algorithm
