#include "sentryward/algorithm/analysis/phase_algebra.h"

#include <algorithm>
#include <limits>

#include "sentryward/algorithm/base/checked_int.h"

namespace sentryward::algorithm {
namespace {

// Separate magnitudes avoid signed intermediate overflow and dependence on
// effect order. Unsigned overflow still gives a safe, inconclusive result.
struct SignedSum final {
  std::uint64_t positive = 0;
  std::uint64_t negative = 0;

  [[nodiscard]] bool Add(std::int64_t value) {
    const bool is_negative = value < 0;
    const auto magnitude = is_negative
        ? std::uint64_t{0} - static_cast<std::uint64_t>(value)
        : static_cast<std::uint64_t>(value);
    auto& destination = is_negative ? negative : positive;
    const auto sum = CheckedAdd(destination, magnitude);
    if (!sum) return false;
    destination = sum.value();
    return true;
  }

  [[nodiscard]] std::optional<std::int64_t> Value() const {
    constexpr auto maximum =
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    if (positive >= negative) {
      const auto magnitude = positive - negative;
      if (magnitude > maximum) return std::nullopt;
      return static_cast<std::int64_t>(magnitude);
    }
    const auto magnitude = negative - positive;
    if (magnitude > maximum + 1) return std::nullopt;
    if (magnitude == maximum + 1)
      return std::numeric_limits<std::int64_t>::min();
    return -static_cast<std::int64_t>(magnitude);
  }
};

}  // namespace

std::optional<AlgebraicPhaseBounds> TryBoundPhaseWithoutDependencies(
    std::span<const EventId> events,
    std::span<const PhaseEffectWeights> effects, bool reject_negative_tx) {
  if (events.size() != effects.size()) return std::nullopt;
  SignedSum total;
  SignedSum transaction_bound;
  SignedSum zero_pending_bound;
  std::size_t nonarrivals = 0;
  for (const auto& effect : effects) {
    if (effect.arrival_increment != 0) return std::nullopt;
    const auto weight = effect.transaction_weight;
    const bool arrival = effect.arrival_weight > 0;
    if (!total.Add(weight)) return std::nullopt;
    // min over arbitrary subsets: forced contributions plus every optional
    // negative contribution. Every graph ideal is one of these subsets.
    if ((weight < 0 || (arrival && !reject_negative_tx)) &&
        !transaction_bound.Add(weight)) return std::nullopt;
    if ((arrival || weight < 0) && !zero_pending_bound.Add(weight))
      return std::nullopt;
    if (!arrival) ++nonarrivals;
  }
  if (total.positive != total.negative) return std::nullopt;
  const auto transaction = transaction_bound.Value();
  const auto zero_pending = zero_pending_bound.Value();
  if (!transaction || *transaction < 0 || !zero_pending) return std::nullopt;

  AlgebraicPhaseBounds result{*transaction, {}};
  result.proper_prefix_lower_bounds.reserve(nonarrivals);
  for (std::size_t index = 0; index < effects.size(); ++index) {
    if (effects[index].arrival_weight > 0) continue;
    // Forcing one optional event out removes its negative contribution.
    const auto omitted = CheckedSubtract(
        *zero_pending,
        std::min<std::int64_t>(effects[index].transaction_weight, 0));
    if (!omitted || omitted.value() <= 0) return std::nullopt;
    result.proper_prefix_lower_bounds.push_back({events[index], omitted.value()});
  }
  return result;
}

}  // namespace sentryward::algorithm
