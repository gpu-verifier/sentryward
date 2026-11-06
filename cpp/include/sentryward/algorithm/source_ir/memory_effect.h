#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <variant>

#include "sentryward/algorithm/base/strong_id.h"

namespace sentryward::algorithm {

enum class AccessMode { kRead, kWrite };
enum class AtomicScope { kCta, kGpu, kSys };
// The memory proxy of an access (PTX ISA, "Proxies"). Generic and async accesses to the same
// bytes are ordered only through a proxy fence (fence.proxy.async) on the path between them, or
// through the implicit fence that follows an async operation's observed completion.
enum class MemoryProxy { kGeneric, kAsync, kTensorMap };

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
  MemoryProxy proxy = MemoryProxy::kGeneric;

  friend bool operator==(const MemoryEffect&, const MemoryEffect&) = default;
};

}  // namespace sentryward::algorithm
