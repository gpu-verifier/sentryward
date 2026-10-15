#pragma once
#include "sentryward/algorithm/io/source_json.h"
#include "sentryward/frontend/ptx/tmem_lifetime.h"
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace sentryward::frontend::ptx {
struct Group2LifetimeSite {
  Coordinates cta_coordinates{};
  std::string operation;
  std::uint32_t instruction_index{};
  std::uint64_t occurrence{};
  std::uint32_t warp{};
  std::optional<std::uint32_t> columns;
  std::optional<TmemAllocationId> allocation_key;
  std::vector<algorithm::ActorId> participant_actors;
  std::vector<algorithm::MilestoneRef> issue_refs, completion_refs,
      publication_refs, accesses;
};
struct Group2LifetimePair {
  Group2LifetimeSite even, odd;
};
struct Group2LifetimeResult {
  std::vector<Group2LifetimePair> pairs;
  std::vector<algorithm::RequiredOrder> required_orders;
  std::vector<algorithm::VerificationLimitation> verification_limitations;
  [[nodiscard]] bool ParticipationProven() const {
    return verification_limitations.empty();
  }
};
struct Group2ProgressResult {
  bool discharged = false;
  std::size_t checked_pairs{};
  std::vector<std::string> blocked, premises;
};
// Produces check-only obligations. Neither routine adds happens-before edges or
// modifies SourceProgram; paired deallocation need not be a CTA rendezvous.
[[nodiscard]] Group2LifetimeResult
PairGroup2Lifetimes(std::span<const Group2LifetimeSite>,
                    Coordinates cluster_dim);
[[nodiscard]] Group2ProgressResult
CheckGroup2OptionalBlocking(const algorithm::SourceProgram &,
                            const Group2LifetimeResult &,
                            std::function<void()> poll = {});
} // namespace sentryward::frontend::ptx
