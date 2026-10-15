#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace sentryward::algorithm {

enum class OverlapStatus { kDisjoint, kOverlap, kUnknown };

struct ByteOverlap final {
  std::string address_space;
  std::string allocation;
  std::uint64_t offset;
  std::uint64_t bytes;

  friend bool operator==(const ByteOverlap&, const ByteOverlap&) = default;
};

struct OverlapEvidence final {
  std::string description;
  std::optional<ByteOverlap> constant_region;

  friend bool operator==(const OverlapEvidence&,
                         const OverlapEvidence&) = default;
};

struct OverlapResult final {
  OverlapStatus status;
  std::optional<OverlapEvidence> evidence;
  std::string reason;
  std::string_view code = "address_overlap_unresolved";

  [[nodiscard]] static OverlapResult Disjoint() {
    return {OverlapStatus::kDisjoint, std::nullopt, {}};
  }

  [[nodiscard]] static OverlapResult Overlap(OverlapEvidence evidence) {
    return {OverlapStatus::kOverlap, std::move(evidence), {}};
  }

  [[nodiscard]] static OverlapResult Unknown(std::string reason, std::string_view code = "address_overlap_unresolved") {
    return {OverlapStatus::kUnknown, std::nullopt, std::move(reason), std::move(code)};
  }

  friend bool operator==(const OverlapResult&, const OverlapResult&) = default;
};

}  // namespace sentryward::algorithm
