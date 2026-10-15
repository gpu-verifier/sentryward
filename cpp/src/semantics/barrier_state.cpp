#include "sentryward/algorithm/semantics/barrier_state.h"

#include <limits>

#include "sentryward/algorithm/base/checked_int.h"

namespace sentryward::algorithm {

namespace {

BarrierRuntimeError error(BarrierRuntimeErrorCode code,
                          BarrierLifetimeId barrier, std::uint32_t phase) {
  return BarrierRuntimeError{code, barrier, phase};
}

constexpr std::uint64_t kSignedBias = std::uint64_t{1} << 63;

std::uint64_t biasedTransaction(std::int64_t value) {
  return static_cast<std::uint64_t>(value) + kSignedBias;
}

std::int64_t signedTransaction(std::uint64_t value) {
  if (value >= kSignedBias) {
    return static_cast<std::int64_t>(value - kSignedBias);
  }
  return std::numeric_limits<std::int64_t>::min() +
         static_cast<std::int64_t>(value);
}

}  // namespace

Result<void, BarrierRuntimeError> BarrierRuntime::Initialize(
    std::uint32_t expected_arrivals) {
  if (initialized_) {
    return Result<void, BarrierRuntimeError>::Failure(
        error(BarrierRuntimeErrorCode::kAlreadyInitialized, barrier_, phase_));
  }
  if (expected_arrivals == 0) {
    return Result<void, BarrierRuntimeError>::Failure(
        error(BarrierRuntimeErrorCode::kZeroExpectedArrivals, barrier_, 0));
  }

  initialized_ = true;
  phase_ = 0;
  expected_arrivals_ = expected_arrivals;
  pending_arrivals_ = expected_arrivals;
  transaction_count_ = 0;
  successful_waits_.clear();
  return Result<void, BarrierRuntimeError>::Success();
}

Result<BarrierTransition, BarrierRuntimeError> BarrierRuntime::NamedArrive(
    std::uint32_t expected_arrivals, std::uint32_t arrivals) {
  if (!expected_arrivals)
    return Result<BarrierTransition, BarrierRuntimeError>::Failure(
        error(BarrierRuntimeErrorCode::kZeroExpectedArrivals, barrier_, phase_));
  if (!initialized_ || pending_arrivals_ == expected_arrivals_) {
    initialized_ = true;
    expected_arrivals_ = pending_arrivals_ = expected_arrivals;
  } else if (expected_arrivals != expected_arrivals_) {
    return Result<BarrierTransition, BarrierRuntimeError>::Failure(
        error(BarrierRuntimeErrorCode::kNamedArrivalCountMismatch, barrier_, phase_));
  }
  const auto phase = phase_;
  if (!arrivals || arrivals > pending_arrivals_)
    return Result<BarrierTransition, BarrierRuntimeError>::Failure(
        error(BarrierRuntimeErrorCode::kArrivalUnderflow, barrier_, phase_));
  pending_arrivals_ -= arrivals;
  auto completion = CompletePhaseIfReady();
  if (!completion) return Result<BarrierTransition, BarrierRuntimeError>::Failure(completion.error());
  return Result<BarrierTransition, BarrierRuntimeError>::Success(
      BarrierTransition{phase, std::nullopt, completion.value()});
}

Result<BarrierTransition, BarrierRuntimeError> BarrierRuntime::ApplyUpdate(
    std::uint32_t arrival_decrement, std::uint64_t transaction_increment,
    bool produces_state) {
  if (!initialized_) {
    return Result<BarrierTransition, BarrierRuntimeError>::Failure(
        error(BarrierRuntimeErrorCode::kNotInitialized, barrier_, phase_));
  }
  if (arrival_decrement > 0 && phase_ > 0 &&
      !successful_waits_.contains(phase_ - 1)) {
    return Result<BarrierTransition, BarrierRuntimeError>::Failure(
        error(BarrierRuntimeErrorCode::kMissingSuccessfulWaitHandoff, barrier_,
              phase_));
  }
  if (arrival_decrement > pending_arrivals_) {
    return Result<BarrierTransition, BarrierRuntimeError>::Failure(
        error(BarrierRuntimeErrorCode::kArrivalUnderflow, barrier_, phase_));
  }
  const auto transaction =
      CheckedAdd(biasedTransaction(transaction_count_), transaction_increment);
  if (!transaction) {
    return Result<BarrierTransition, BarrierRuntimeError>::Failure(
        error(BarrierRuntimeErrorCode::kTransactionOverflow, barrier_, phase_));
  }

  const std::uint32_t effect_phase = phase_;
  const std::uint32_t new_pending = pending_arrivals_ - arrival_decrement;
  const std::int64_t new_transaction = signedTransaction(transaction.value());
  const bool completes = new_pending == 0 && new_transaction == 0;
  if (completes && phase_ == std::numeric_limits<std::uint32_t>::max()) {
    return Result<BarrierTransition, BarrierRuntimeError>::Failure(
        error(BarrierRuntimeErrorCode::kPhaseOverflow, barrier_, phase_));
  }

  pending_arrivals_ = new_pending;
  transaction_count_ = new_transaction;
  std::optional<StateToken> token;
  if (produces_state) {
    token = StateToken{barrier_, effect_phase};
  }

  auto completion = CompletePhaseIfReady();
  if (!completion) {
    return Result<BarrierTransition, BarrierRuntimeError>::Failure(
        completion.error());
  }
  return Result<BarrierTransition, BarrierRuntimeError>::Success(
      BarrierTransition{effect_phase, token, completion.value()});
}

Result<BarrierTransition, BarrierRuntimeError> BarrierRuntime::CompleteTx(
    std::uint64_t bytes) {
  if (!initialized_) {
    return Result<BarrierTransition, BarrierRuntimeError>::Failure(
        error(BarrierRuntimeErrorCode::kNotInitialized, barrier_, phase_));
  }
  const std::uint64_t biased = biasedTransaction(transaction_count_);
  if (bytes > biased) {
    return Result<BarrierTransition, BarrierRuntimeError>::Failure(error(
        BarrierRuntimeErrorCode::kTransactionUnderflow, barrier_, phase_));
  }

  const std::uint32_t effect_phase = phase_;
  const std::int64_t new_transaction_count = signedTransaction(biased - bytes);
  if (reject_negative_tx_ && new_transaction_count < 0) {
    return Result<BarrierTransition, BarrierRuntimeError>::Failure(error(
        BarrierRuntimeErrorCode::kTransactionUnderflow, barrier_, phase_));
  }
  const bool completes = pending_arrivals_ == 0 && new_transaction_count == 0;
  if (completes && phase_ == std::numeric_limits<std::uint32_t>::max()) {
    return Result<BarrierTransition, BarrierRuntimeError>::Failure(
        error(BarrierRuntimeErrorCode::kPhaseOverflow, barrier_, phase_));
  }

  transaction_count_ = new_transaction_count;
  auto completion = CompletePhaseIfReady();
  if (!completion) {
    return Result<BarrierTransition, BarrierRuntimeError>::Failure(
        completion.error());
  }
  return Result<BarrierTransition, BarrierRuntimeError>::Success(
      BarrierTransition{effect_phase, std::nullopt, completion.value()});
}

Result<BarrierTransition, BarrierRuntimeError> BarrierRuntime::AddArrivals(
    std::uint32_t arrivals) {
  if (!initialized_) {
    return Result<BarrierTransition, BarrierRuntimeError>::Failure(
        error(BarrierRuntimeErrorCode::kNotInitialized, barrier_, phase_));
  }
  const auto pending = CheckedAdd(pending_arrivals_, arrivals);
  if (!pending) {
    return Result<BarrierTransition, BarrierRuntimeError>::Failure(
        error(BarrierRuntimeErrorCode::kArrivalOverflow, barrier_, phase_));
  }
  pending_arrivals_ = pending.value();
  return Result<BarrierTransition, BarrierRuntimeError>::Success(
      BarrierTransition{phase_, std::nullopt, std::nullopt});
}

Result<PhaseKey, BarrierRuntimeError> BarrierRuntime::SelectStateWaitTarget(
    StateToken state) const {
  if (!initialized_) {
    return Result<PhaseKey, BarrierRuntimeError>::Failure(
        error(BarrierRuntimeErrorCode::kNotInitialized, barrier_, phase_));
  }
  if (state.barrier != barrier_) {
    return Result<PhaseKey, BarrierRuntimeError>::Failure(error(
        BarrierRuntimeErrorCode::kStateBarrierMismatch, barrier_, phase_));
  }
  if (state.phase != phase_ && !(phase_ > 0 && state.phase == phase_ - 1)) {
    return Result<PhaseKey, BarrierRuntimeError>::Failure(
        error(BarrierRuntimeErrorCode::kStaleState, barrier_, phase_));
  }
  return Result<PhaseKey, BarrierRuntimeError>::Success(
      PhaseKey{barrier_, state.phase});
}

Result<std::optional<PhaseKey>, BarrierRuntimeError>
BarrierRuntime::SelectParityWaitTarget(std::uint8_t parity) const {
  using TargetResult = Result<std::optional<PhaseKey>, BarrierRuntimeError>;
  if (!initialized_) {
    return TargetResult::Failure(
        error(BarrierRuntimeErrorCode::kNotInitialized, barrier_, phase_));
  }
  if (parity > 1) {
    return TargetResult::Failure(
        error(BarrierRuntimeErrorCode::kInvalidParity, barrier_, phase_));
  }

  if (parity == static_cast<std::uint8_t>(phase_ & 1U)) {
    return TargetResult::Success(PhaseKey{barrier_, phase_});
  }
  if (phase_ == 0) {
    return TargetResult::Success(std::nullopt);
  }
  return TargetResult::Success(PhaseKey{barrier_, phase_ - 1});
}

Result<void, BarrierRuntimeError> BarrierRuntime::RecordSuccessfulWait(
    std::uint32_t phase) {
  if (!initialized_) {
    return Result<void, BarrierRuntimeError>::Failure(
        error(BarrierRuntimeErrorCode::kNotInitialized, barrier_, phase_));
  }
  if (!isPhaseCompleted(phase)) {
    return Result<void, BarrierRuntimeError>::Failure(error(
        BarrierRuntimeErrorCode::kWaitForIncompletePhase, barrier_, phase));
  }
  successful_waits_.insert(phase);
  return Result<void, BarrierRuntimeError>::Success();
}

Result<std::optional<PhaseKey>, BarrierRuntimeError>
BarrierRuntime::CompletePhaseIfReady() {
  if (pending_arrivals_ != 0 || transaction_count_ != 0) {
    return Result<std::optional<PhaseKey>, BarrierRuntimeError>::Success(
        std::nullopt);
  }
  if (phase_ == std::numeric_limits<std::uint32_t>::max()) {
    return Result<std::optional<PhaseKey>, BarrierRuntimeError>::Failure(
        error(BarrierRuntimeErrorCode::kPhaseOverflow, barrier_, phase_));
  }

  const PhaseKey completed{barrier_, phase_};
  ++phase_;
  pending_arrivals_ = expected_arrivals_;
  transaction_count_ = 0;
  return Result<std::optional<PhaseKey>, BarrierRuntimeError>::Success(
      completed);
}

}  // namespace sentryward::algorithm
