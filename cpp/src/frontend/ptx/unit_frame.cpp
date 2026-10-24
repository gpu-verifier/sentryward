#include "sentryward/frontend/ptx/unit_frame.h"

#include "sentryward/frontend/ptx/resources.h"

#include <initializer_list>
#include <utility>

namespace sentryward::frontend::ptx {
namespace {
// One allocation per name; the parts are those of Resources::Bind's grammar.
std::string Join(std::initializer_list<std::string_view> parts) {
  std::size_t size = 0;
  for (const auto part : parts) size += part.size();
  std::string result;
  result.reserve(size);
  for (const auto part : parts) result.append(part);
  return result;
}
} // namespace

AllocationKey AllocationKey::Shared(CtaRank owner) {
  AllocationKey key;
  key.kind = Kind::CtaShared;
  key.space = "shared";
  key.owner = owner;
  return key;
}
AllocationKey AllocationKey::Tensor(CtaRank owner, std::string resource) {
  AllocationKey key;
  key.kind = Kind::CtaTensor;
  key.space = "tensor";
  key.owner = owner;
  key.object = std::move(resource);
  return key;
}
AllocationKey AllocationKey::Local(CtaRank owner, std::uint32_t thread, std::string resource) {
  AllocationKey key;
  key.kind = Kind::ThreadLocal;
  key.space = "local";
  key.owner = owner;
  key.thread = thread;
  key.object = std::move(resource);
  return key;
}
AllocationKey AllocationKey::Plain(std::string space, std::string resource) {
  AllocationKey key;
  key.kind = Kind::Plain;
  key.space = std::move(space);
  key.object = std::move(resource);
  return key;
}
AllocationKey AllocationKey::Opaque(std::string space, std::string allocation, std::string alias) {
  AllocationKey key;
  key.kind = Kind::Opaque;
  key.space = std::move(space);
  key.object = std::move(allocation);
  key.alias = std::move(alias);
  return key;
}

AllocationKey BindKey(std::string space, std::string resource, CtaRank owner,
                      std::optional<std::uint32_t> active_thread) {
  space = CanonicalSpace(std::move(space));
  if (space == "shared" || space == "cluster_shared") {
    if (resource != "physical.shared.cta.0")
      throw SemanticError("unresolved_shared_placement",
                          "shared allocation has no physical placement");
    return AllocationKey::Shared(owner);
  }
  if (space == "tensor")
    return AllocationKey::Tensor(owner, std::move(resource));
  if (space == "local") {
    if (!active_thread) throw SemanticError("active_thread_context_missing", "local access has no active thread");
    return AllocationKey::Local(owner, *active_thread, std::move(resource));
  }
  if (space == "generic")
    throw SemanticError("unresolved_footprint",
                        "memory allocation has no exact owner");
  return AllocationKey::Plain(std::move(space), std::move(resource));
}
void CheckRange(std::uint64_t offset, std::uint64_t bytes) {
  if (bytes > UINT64_MAX - offset)
    throw SemanticError("invalid_memory_range", "memory range overflows u64");
}
std::string_view KindText(BarrierKey::Kind kind) noexcept {
  switch (kind) {
  case BarrierKey::Kind::Physical: return "physical";
  case BarrierKey::Kind::Symbolic: return "symbolic";
  case BarrierKey::Kind::Named: return "named";
  }
  return {};
}

CtaRank UnitFrame::Rank(std::uint32_t rank) const {
  const auto d = cluster_dim();
  if (rank >= std::uint64_t(d[0]) * d[1] * d[2])
    throw SemanticError("invalid_cluster_rank", "cluster rank outside launch");
  return {rank};
}
CtaRank UnitFrame::RankOf(Coordinates c) const {
  if (const auto rank = TryRankOf(c)) return *rank;
  throw SemanticError("cluster_owner", "owner outside cluster");
}
std::optional<CtaRank> UnitFrame::TryRankOf(Coordinates c) const noexcept {
  const auto d = cluster_dim();
  const auto origin = ClusterOrigin();
  Coordinates local;
  for (int i = 0; i < 3; ++i) {
    if (c[i] < origin[i] || c[i] >= origin[i] + d[i]) return std::nullopt;
    local[i] = c[i] - origin[i];
  }
  return CtaRank{local[0] + d[0] * (local[1] + d[1] * local[2])};
}
std::string UnitFrame::Text(CtaRank rank) const { return CoordinateText(Cta(rank)); }
nlohmann::json UnitFrame::Json(CtaRank rank) const { return nlohmann::json(Cta(rank)); }
FormattedAllocation UnitFrame::Format(AllocationKey &&key) const {
  FormattedAllocation result;
  switch (key.kind) {
  case AllocationKey::Kind::CtaShared:
    result.allocation = Join({"cta:", Text(key.owner), ":shared:physical.shared.cta.0"});
    break;
  case AllocationKey::Kind::CtaTensor: {
    const auto owner = Text(key.owner);
    result.allocation = Join({"cta:", owner, ":tensor:", key.object});
    result.alias = Join({"cta:", owner, ":tensor"});
    break;
  }
  case AllocationKey::Kind::ThreadLocal:
    result.allocation = Join({"cta:", Text(key.owner), ":thread:", std::to_string(key.thread), ":", key.object});
    break;
  case AllocationKey::Kind::Plain:
    result.allocation = std::move(key.object);
    break;
  case AllocationKey::Kind::Opaque:
    result.allocation = std::move(key.object);
    result.alias = std::move(key.alias);
    break;
  }
  result.space = std::move(key.space);
  return result;
}

} // namespace sentryward::frontend::ptx
