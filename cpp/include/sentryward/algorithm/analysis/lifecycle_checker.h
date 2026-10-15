#pragma once

#include <vector>

#include "sentryward/algorithm/analysis/phase_database.h"
#include "sentryward/algorithm/analysis/happens_before_graph.h"
#include "sentryward/algorithm/analysis/reachability.h"
#include "sentryward/algorithm/base/result.h"

namespace sentryward::algorithm {

struct LifecycleHandoffCertificate final {
  PhaseKey phase;
  EventId positive_arrival;
  EventId previous_phase_wait_done;

  friend bool operator==(const LifecycleHandoffCertificate&,
                         const LifecycleHandoffCertificate&) = default;
};

struct LifecycleCertificate final {
  std::vector<LifecycleHandoffCertificate> handoffs;

  friend bool operator==(const LifecycleCertificate&,
                         const LifecycleCertificate&) = default;
};

struct LifecycleWitness final {
  PhaseKey phase;
  EventId positive_arrival;
  PhaseKey previous_phase;
  std::vector<EventId> candidate_waits;

  friend bool operator==(const LifecycleWitness&,
                         const LifecycleWitness&) = default;
};

enum class LifecycleCheckErrorCode {
  kMissingPreviousPhase,
  kPreviousPhaseIncomplete,
  kInvalidPositiveArrival,
  kMissingSuccessfulWaitHandoff,
};

struct LifecycleCheckError final {
  LifecycleCheckErrorCode code =
      LifecycleCheckErrorCode::kMissingSuccessfulWaitHandoff;
  LifecycleWitness witness;
};

class LifecycleChecker final {
 public:
  [[nodiscard]] Result<LifecycleCertificate, LifecycleCheckError> Check(
      const PhaseDatabase& phases, const HappensBeforeGraph& proof,
      ReachabilityOracle& reachability) const;
};

}  // namespace sentryward::algorithm
