#pragma once

#include <compare>
#include <cstdint>
#include <ostream>

namespace sentryward::algorithm {

template <typename Tag>
class StrongId final {
 public:
  constexpr StrongId() = default;
  explicit constexpr StrongId(std::uint32_t value) : value_(value) {}

  [[nodiscard]] constexpr std::uint32_t value() const noexcept {
    return value_;
  }

  friend constexpr bool operator==(StrongId, StrongId) noexcept = default;
  friend constexpr auto operator<=>(StrongId, StrongId) noexcept = default;

 private:
  std::uint32_t value_ = 0;
};

template <typename Tag>
std::ostream& operator<<(std::ostream& stream, StrongId<Tag> id) {
  return stream << id.value();
}

using InstId = StrongId<struct InstTag>;
using EventId = StrongId<struct EventTag>;
using ActorId = StrongId<struct ActorTag>;
using CtaId = StrongId<struct CtaTag>;
using WarpId = StrongId<struct WarpTag>;
using BarrierLifetimeId = StrongId<struct BarrierTag>;
using StateId = StrongId<struct StateTag>;
using AsyncOpId = StrongId<struct AsyncTag>;
using SyncId = StrongId<struct SyncTag>;
using AddressId = StrongId<struct AddressTag>;

}  // namespace sentryward::algorithm
