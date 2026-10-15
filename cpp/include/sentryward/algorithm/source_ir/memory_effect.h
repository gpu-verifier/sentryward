#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <variant>

#include "sentryward/algorithm/base/strong_id.h"

namespace sentryward::algorithm {

enum class AccessMode { kRead, kWrite };
enum class AtomicScope { kCta, kGpu, kSys };

struct RegisterAddress final {
  // A physical register name is local to this actor, not the issuing event.
  ActorId actor;
  std::string name;

  friend bool operator==(const RegisterAddress&,
                         const RegisterAddress&) = default;
};

using MemoryEndpoint = std::variant<AddressId, RegisterAddress>;

[[nodiscard]] inline bool operator==(const MemoryEndpoint& endpoint,
                                     AddressId address) {
  const auto* memory = std::get_if<AddressId>(&endpoint);
  return memory != nullptr && *memory == address;
}

struct MemoryEffect final {
  AccessMode mode;
  MemoryEndpoint endpoint;
  // nullopt denotes an extent supplied by the abstract region.
  std::optional<std::uint64_t> bytes;
  std::optional<AtomicScope> atomic_scope = std::nullopt;

  friend bool operator==(const MemoryEffect&, const MemoryEffect&) = default;
};

}  // namespace sentryward::algorithm
