#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>

#include "sentryward/algorithm/base/result.h"
#include "sentryward/algorithm/base/strong_id.h"

namespace sentryward::algorithm {

struct PhaseKey final {
  BarrierLifetimeId barrier;
  std::uint32_t phase;

  friend bool operator==(const PhaseKey&, const PhaseKey&) = default;
  friend auto operator<=>(const PhaseKey&, const PhaseKey&) = default;
};

struct StateToken final {
  BarrierLifetimeId barrier;
  std::uint32_t phase;

  friend bool operator==(const StateToken&, const StateToken&) = default;
};

enum class BarrierRuntimeErrorCode {
  kAlreadyInitialized,
  kZeroExpectedArrivals,
  kNotInitialized,
  kArrivalUnderflow,
  kTransactionOverflow,
  kTransactionUnderflow,
  kPhaseOverflow,
  kStateBarrierMismatch,
  kStaleState,
  kInvalidParity,
  kStaleParity,
  kWaitForIncompletePhase,
  kMissingSuccessfulWaitHandoff,
  kArrivalOverflow,
  kNamedArrivalCountMismatch,
};

struct BarrierRuntimeError final {
  BarrierRuntimeErrorCode code;
  BarrierLifetimeId barrier;
  std::uint32_t phase = 0;
};

struct BarrierTransition final {
  std::uint32_t effect_phase;
  std::optional<StateToken> produced_state;
  std::optional<PhaseKey> completed_phase;
};

// Mutable operational state for one initialized mbarrier lifetime. Counter
// methods perform their urgent phase transition before returning.
class BarrierRuntime final {
 public:
  explicit BarrierRuntime(BarrierLifetimeId barrier,
                          bool reject_negative_tx = false)
      : barrier_(barrier), reject_negative_tx_(reject_negative_tx) {}

  [[nodiscard]] Result<void, BarrierRuntimeError> Initialize(
      std::uint32_t expected_arrivals);

  [[nodiscard]] Result<BarrierTransition, BarrierRuntimeError> ApplyUpdate(
      std::uint32_t arrival_decrement, std::uint64_t transaction_increment,
      bool produces_state = false);

  [[nodiscard]] Result<BarrierTransition, BarrierRuntimeError> CompleteTx(
      std::uint64_t bytes);

  [[nodiscard]] Result<BarrierTransition, BarrierRuntimeError> NamedArrive(
      std::uint32_t expected_arrivals, std::uint32_t arrivals = 1);

  // Only the current pending count changes; the next-phase reset is unchanged.
  [[nodiscard]] Result<BarrierTransition, BarrierRuntimeError> AddArrivals(
      std::uint32_t arrivals);

  [[nodiscard]] Result<PhaseKey, BarrierRuntimeError> SelectStateWaitTarget(
      StateToken state) const;
  // An empty target denotes readiness from the initialized barrier state.
  [[nodiscard]] Result<std::optional<PhaseKey>, BarrierRuntimeError>
  SelectParityWaitTarget(std::uint8_t parity) const;

  [[nodiscard]] Result<void, BarrierRuntimeError> RecordSuccessfulWait(
      std::uint32_t phase);

  [[nodiscard]] bool initialized() const noexcept { return initialized_; }
  [[nodiscard]] BarrierLifetimeId barrier() const noexcept { return barrier_; }
  [[nodiscard]] std::uint32_t phase() const noexcept { return phase_; }
  [[nodiscard]] std::uint32_t expectedArrivals() const noexcept {
    return expected_arrivals_;
  }
  [[nodiscard]] std::uint32_t pendingArrivals() const noexcept {
    return pending_arrivals_;
  }
  [[nodiscard]] std::int64_t transactionCount() const noexcept {
    return transaction_count_;
  }
  [[nodiscard]] bool isPhaseCompleted(std::uint32_t phase) const noexcept {
    return initialized_ && phase < phase_;
  }
  [[nodiscard]] bool hasSuccessfulWait(std::uint32_t phase) const noexcept {
    return successful_waits_.contains(phase);
  }
  [[nodiscard]] std::size_t successfulWaitCount() const noexcept {
    return successful_waits_.size();
  }

 private:
  [[nodiscard]] Result<std::optional<PhaseKey>, BarrierRuntimeError>
  CompletePhaseIfReady();

  BarrierLifetimeId barrier_;
  bool reject_negative_tx_ = false;
  bool initialized_ = false;
  std::uint32_t phase_ = 0;
  std::uint32_t expected_arrivals_ = 0;
  std::uint32_t pending_arrivals_ = 0;
  std::int64_t transaction_count_ = 0;
  std::set<std::uint32_t> successful_waits_;
};

}  // namespace sentryward::algorithm
