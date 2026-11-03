#pragma once
#include "sentryward/algorithm/checker/memory_race_checker.h"

#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sentryward::algorithm {
std::optional<Result<MemoryRaceResult,MemoryAnalysisError>> TryFamilyMemoryCheck(
    const SourceProgram&,const PassCertificate&,MemoryModel&,const MemoryRaceOptions&,
    std::optional<std::span<const MemoryAccessFamily>> families = std::nullopt,
    StaticHbQueries* shared_hb = nullptr);

// The write-obligation groups of a MemoryRaceResult (duplicate TMA stores, same-value writes), for
// the two memory checkers: a union-find over the access ids of accepted write pairs, with one entry
// per access that joins a group. Each access keeps one partner, the other access of the first
// accepted pair that joined it.
class WriteObligationGroups final {
 public:
  // An accepted pair: one group, and each is the other's partner unless it already has one.
  void pair(AccessId first, AccessId second);
  // `access` joins the group of `anchor`, with `partner` (the other access of an accepted pair) as
  // its partner unless it already has one.
  void join(AccessId access, AccessId partner, AccessId anchor);
  // One group for two accesses; an absent access enters without a partner (join gives it one).
  void unite(AccessId first, AccessId second);
  // Whether two accesses are in one group; an absent access is alone. Inserts nothing.
  [[nodiscard]] bool connected(AccessId first, AccessId second);
  // Groups in order of their first access, each in access order.
  [[nodiscard]] std::vector<std::vector<AccessId>> groups();
  // (access, partner) for every grouped access that has one, in access order.
  [[nodiscard]] std::vector<std::pair<AccessId,AccessId>> partners() const;

 private:
  struct Entry {
    std::uint32_t parent,partner=0;
    bool paired=false;
  };
  // Adds `access` when absent, and gives it `partner` unless it already has one.
  void setPartner(AccessId access,AccessId partner);
  // The root of a present id (halving its path), or the id itself when absent. Inserts nothing.
  std::uint32_t root(std::uint32_t id);
  std::unordered_map<std::uint32_t,Entry> entries_;
};
}
