#pragma once

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include "sentryward/algorithm/analysis/phase_database.h"
#include "sentryward/algorithm/analysis/happens_before_graph.h"
#include "sentryward/algorithm/analysis/reachability.h"
#include "sentryward/algorithm/analysis/wait_reachability_batch.h"
#include "sentryward/algorithm/base/result.h"
#include "sentryward/algorithm/event_ir/program.h"
#include "sentryward/algorithm/semantics/reference_trace.h"

namespace sentryward::algorithm {

struct LowerFreshnessCertificate final {
  PhaseKey previous_phase;
  std::vector<EventId> effects;
  EventId completion;

  friend bool operator==(const LowerFreshnessCertificate&,
                         const LowerFreshnessCertificate&) = default;
};

struct UpperFreshnessCertificate final {
  PhaseKey next_phase;
  EventId reachable_effect;

  friend bool operator==(const UpperFreshnessCertificate&,
                         const UpperFreshnessCertificate&) = default;
};

struct StateWaitCertificate final {
  StateId state;
  EventId definition;

  friend bool operator==(const StateWaitCertificate&,
                         const StateWaitCertificate&) = default;
};

struct ParityWaitCertificate final {
  std::uint8_t parity = 0;

  friend bool operator==(const ParityWaitCertificate&,
                         const ParityWaitCertificate&) = default;
};

struct NamedWaitCertificate final {
  friend bool operator==(const NamedWaitCertificate&, const NamedWaitCertificate&) = default;
};
using WaitKindCertificate =
    std::variant<StateWaitCertificate, ParityWaitCertificate, NamedWaitCertificate>;

struct WaitCertificate final {
  EventId issue;
  EventId done;
  // Initial readiness has no numbered target or completion event.
  std::optional<PhaseKey> target;
  std::optional<EventId> initialization;
  std::optional<EventId> target_completion;
  std::optional<LowerFreshnessCertificate> lower_freshness;
  std::optional<UpperFreshnessCertificate> upper_freshness;
  WaitKindCertificate kind;

  friend bool operator==(const WaitCertificate&, const WaitCertificate&) =
      default;
};

struct WaitEventWitness final {
  EventId issue;
  std::optional<EventId> related_event;
  std::optional<PhaseKey> phase;

  friend bool operator==(const WaitEventWitness&,
                         const WaitEventWitness&) = default;
};

struct WaitReachabilityWitness final {
  EventId issue;
  PhaseKey phase;
  std::vector<EventId> required_events;

  friend bool operator==(const WaitReachabilityWitness&,
                         const WaitReachabilityWitness&) = default;
};

struct WaitStateWitness final {
  EventId issue;
  StateId state;
  PhaseKey target;
  std::optional<EventId> definition;
  std::optional<StateToken> token;

  friend bool operator==(const WaitStateWitness&,
                         const WaitStateWitness&) = default;
};

struct WaitParityWitness final {
  EventId issue;
  PhaseKey target;
  std::uint8_t actual = 0;
  std::uint8_t expected = 0;

  friend bool operator==(const WaitParityWitness&,
                         const WaitParityWitness&) = default;
};

using WaitCheckWitness =
    std::variant<WaitEventWitness, WaitReachabilityWitness, WaitStateWitness,
                 WaitParityWitness, HappensBeforeGraphMutationError>;

enum class WaitCheckErrorCode {
  kUnknownEvent,
  kNotWaitIssue,
  kMissingWaitTarget,
  kMissingWaitOrigin,
  kMissingWaitMilestones,
  kMismatchedWaitDone,
  kMissingInitialization,
  kInitializationNotOrdered,
  kUnknownTargetPhase,
  kTargetPhaseIncomplete,
  kMissingTargetCompletion,
  kTargetCompletionNotOrderedBeforeDone,
  kIssueNotOrderedBeforeDone,
  kMissingPreviousPhase,
  kPreviousPhaseNotOrderedBeforeIssue,
  kMissingPreviousPhaseCompletion,
  kPhaseNumberOverflow,
  kMissingNextPhase,
  kIssueNotOrderedBeforeNextPhase,
  kMissingStateDefinition,
  kInvalidStateDefinition,
  kStateBarrierMismatch,
  kMissingStateToken,
  kStatePhaseMismatch,
  kStateDefinitionNotOrderedBeforeIssue,
  kParityMismatch,
  kHappensBeforeGraphMutationFailed,
};

struct WaitCheckError final {
  WaitCheckErrorCode code = WaitCheckErrorCode::kUnknownEvent;
  WaitCheckWitness witness;
};

class ParallelReachability;

class WaitChecker final {
 public:
  [[nodiscard]] Result<WaitCertificate, WaitCheckError> Check(
      EventId issue, const LoweredProgram& program,
      const ReferenceTrace& trace, const PhaseDatabase& phases,
      HappensBeforeGraphBuilder& proof, ReachabilityOracle& reachability) const;

  [[nodiscard]] Result<std::vector<WaitCertificate>, WaitCheckError> CheckAll(
      const LoweredProgram& program, const ReferenceTrace& trace,
      const PhaseDatabase& phases, HappensBeforeGraphBuilder& proof,
      ReachabilityOracle& reachability,
      ParallelReachability* parallel = nullptr,
      WaitReachabilityBatchOptions batch_options = {}) const;
};

}  // namespace sentryward::algorithm
