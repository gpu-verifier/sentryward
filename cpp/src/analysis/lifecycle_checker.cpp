#include "sentryward/algorithm/analysis/lifecycle_checker.h"

#include <optional>
#include <utility>

namespace sentryward::algorithm {
namespace {

[[nodiscard]] LifecycleCheckError Error(
    LifecycleCheckErrorCode code, PhaseKey phase, EventId arrival,
    PhaseKey previous, std::vector<EventId> candidates = {}) {
  return LifecycleCheckError{
      code, LifecycleWitness{phase, arrival, previous, std::move(candidates)}};
}

}  // namespace

Result<LifecycleCertificate, LifecycleCheckError> LifecycleChecker::Check(
    const PhaseDatabase& phases, const HappensBeforeGraph& proof,
    ReachabilityOracle& reachability) const {
  LifecycleCertificate certificate;
  for (const PhaseInfo& phase : phases.phases()) {
    if (phase.phase == 0 || phase.named) {
      continue;
    }
    const PhaseKey previous{phase.barrier, phase.phase - 1};
    const PhaseInfo* previous_info = phases.find(previous);
    std::optional<std::vector<std::optional<EventId>>> selected_waits;
    for (std::size_t index = 0; index < phase.positive_arrivals.size(); ++index) {
      const EventId arrival = phase.positive_arrivals[index];
      if (previous_info == nullptr) {
        return Result<LifecycleCertificate, LifecycleCheckError>::Failure(
            Error(LifecycleCheckErrorCode::kMissingPreviousPhase,
                  phase.key(), arrival, previous));
      }
      if (!previous_info->completed) {
        return Result<LifecycleCertificate, LifecycleCheckError>::Failure(
            Error(LifecycleCheckErrorCode::kPreviousPhaseIncomplete,
                  phase.key(), arrival, previous,
                  previous_info->waits_done_for_phase));
      }
      if (!proof.contains(arrival)) {
        return Result<LifecycleCertificate, LifecycleCheckError>::Failure(
            Error(LifecycleCheckErrorCode::kInvalidPositiveArrival,
                  phase.key(), arrival, previous,
                  previous_info->waits_done_for_phase));
      }
      auto decoded = GetPhaseEffectWeights(proof.event(arrival));
      if (!decoded || !decoded.value().has_value() ||
          decoded.value()->barrier != phase.barrier ||
          decoded.value()->arrival_weight == 0) {
        return Result<LifecycleCertificate, LifecycleCheckError>::Failure(
            Error(LifecycleCheckErrorCode::kInvalidPositiveArrival,
                  phase.key(), arrival, previous,
                  previous_info->waits_done_for_phase));
      }

      if (!selected_waits)
        selected_waits = reachability.FindAncestors(
            previous_info->waits_done_for_phase, phase.positive_arrivals);
      const std::optional<EventId> selected_wait = (*selected_waits)[index];
      if (!selected_wait.has_value()) {
        return Result<LifecycleCertificate, LifecycleCheckError>::Failure(
            Error(LifecycleCheckErrorCode::kMissingSuccessfulWaitHandoff,
                  phase.key(), arrival, previous,
                  previous_info->waits_done_for_phase));
      }
      certificate.handoffs.push_back(LifecycleHandoffCertificate{
          phase.key(), arrival, *selected_wait});
    }
  }
  return Result<LifecycleCertificate, LifecycleCheckError>::Success(
      std::move(certificate));
}

}  // namespace sentryward::algorithm
