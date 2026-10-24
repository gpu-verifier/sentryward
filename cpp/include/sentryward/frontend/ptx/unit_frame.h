#pragma once
// Naming frame of one consumed or executing CTA.
// Every address-table allocation name, alias, barrier-table owner, coordinate
// JSON and owner decision goes through it. Owners are CtaRanks in code; table
// keys and names use the owner's absolute coordinates (Cta(rank)). The value-IR
// `cta` argument is not frame naming: the value view passes CtaFacts::Linear()
// and the address-value extractor keeps 0.
#include "sentryward/frontend/ptx/config.h"

#include <compare>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace sentryward::frontend::ptx {
// A CTA's rank inside its cluster: l0 + d0*(l1 + d1*l2) for its coordinates l
// relative to the cluster origin and the cluster dimensions d.
// Obtained only from UnitFrame::self/Rank/RankOf/TryRankOf, so always valid.
struct CtaRank {
  std::uint32_t value = 0;
  friend auto operator<=>(const CtaRank &, const CtaRank &) = default;
};

// An address-table allocation before formatting (Resources::Bind's grammar).
struct AllocationKey {
  enum class Kind : std::uint8_t { CtaShared, CtaTensor, ThreadLocal, Plain, Opaque };
  Kind kind = Kind::Plain;
  std::string space;         // the interned address space
  CtaRank owner{};           // CtaShared, CtaTensor, ThreadLocal
  std::uint32_t thread = 0;  // ThreadLocal: warp_id * 32 + lane_id
  std::string object, alias; // object: the resource (Plain, Opaque: the allocation verbatim); alias: Opaque only
  static AllocationKey Shared(CtaRank owner); // physical.shared.cta.0
  static AllocationKey Tensor(CtaRank owner, std::string resource);
  static AllocationKey Local(CtaRank owner, std::uint32_t thread, std::string resource);
  static AllocationKey Plain(std::string space, std::string resource);
  static AllocationKey Opaque(std::string space, std::string allocation, std::string alias);
};
struct FormattedAllocation {
  std::string space, allocation, alias;
};
// The validation half of Resources::Bind, in Bind's exact order: canonical
// space, then unresolved_shared_placement / active_thread_context_missing /
// unresolved_footprint. `active_thread` is warp_id * 32 + lane_id of the active lane.
AllocationKey BindKey(std::string space, std::string resource, CtaRank owner,
                      std::optional<std::uint32_t> active_thread);
// Bind's range check: invalid_memory_range.
void CheckRange(std::uint64_t offset, std::uint64_t bytes);

// A barrier-table key; the table stores Cta(owner).
struct BarrierKey {
  enum class Kind : std::uint8_t { Physical, Symbolic, Named };
  Kind kind = Kind::Physical;
  CtaRank owner{};
  std::string identity;
};
[[nodiscard]] std::string_view KindText(BarrierKey::Kind kind) noexcept;

class UnitFrame {
public:
  UnitFrame(Coordinates self, std::optional<Coordinates> cluster_dim) noexcept
      : self_(self), cluster_dim_(cluster_dim) {}
  // The CTA's own coordinates, as constructed.
  [[nodiscard]] Coordinates SelfCoordinates() const noexcept { return self_; }
  // cluster_dim.value_or({1,1,1}).
  [[nodiscard]] Coordinates cluster_dim() const noexcept {
    return cluster_dim_.value_or(Coordinates{1, 1, 1});
  }
  // == RankOf(SelfCoordinates()), computed at each call. The own CTA is
  // always inside its cluster, so this cannot throw.
  [[nodiscard]] CtaRank self() const noexcept {
    const auto d = cluster_dim();
    Coordinates local;
    for (int i = 0; i < 3; ++i)
      local[i] = self_[i] % d[i];
    return {local[0] + d[0] * (local[1] + d[1] * local[2])};
  }
  // `rank` itself; invalid_cluster_rank unless it is below the cluster size.
  [[nodiscard]] CtaRank Rank(std::uint32_t rank) const;
  // The rank of CTA `c` in this CTA's cluster; cluster_owner when `c` is
  // outside the cluster. Executor only.
  [[nodiscard]] CtaRank RankOf(Coordinates c) const;
  // RankOf without the throw; the facade's relative shared-site view only.
  [[nodiscard]] std::optional<CtaRank> TryRankOf(Coordinates c) const noexcept;
  // The coordinates of the CTA at `rank` in this CTA's cluster:
  // ClusterOrigin() plus the rank's coordinates inside the cluster.
  [[nodiscard]] Coordinates Cta(CtaRank rank) const noexcept {
    const auto d = cluster_dim();
    const auto r = rank.value;
    const auto origin = ClusterOrigin();
    Coordinates local{r % d[0], (r / d[0]) % d[1], r / (d[0] * d[1])}, result;
    for (int i = 0; i < 3; ++i)
      result[i] = origin[i] + local[i];
    return result;
  }
  // c - c % d: the cluster's origin CTA.
  [[nodiscard]] Coordinates ClusterOrigin() const noexcept {
    const auto d = cluster_dim();
    Coordinates origin;
    for (int k = 0; k < 3; ++k)
      origin[k] = self_[k] - self_[k] % d[k];
    return origin;
  }
  // The coordinate text of Cta(rank): "(x, y, z)".
  [[nodiscard]] std::string Text(CtaRank rank) const;
  // nlohmann::json(Cta(rank)) == [x, y, z].
  [[nodiscard]] nlohmann::json Json(CtaRank rank) const;
  // The only place the allocation grammar (cta:<owner>:...) is written. The
  // rvalue form moves the key's strings into the result, without a by-value
  // parameter's extra move; the lvalue form formats a copy.
  [[nodiscard]] FormattedAllocation Format(AllocationKey &&key) const;
  [[nodiscard]] FormattedAllocation Format(const AllocationKey &key) const { return Format(AllocationKey(key)); }

private:
  Coordinates self_;
  std::optional<Coordinates> cluster_dim_;
};

} // namespace sentryward::frontend::ptx
