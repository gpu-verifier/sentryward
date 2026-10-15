#include "sentryward/algorithm/source_ir/diagnostic.h"

namespace sentryward::algorithm {

const char* ToString(DiagnosticClass classification) noexcept {
  switch (classification) {
    case DiagnosticClass::kInvalidInput:
      return "invalid-input";
    case DiagnosticClass::kUnsupported:
      return "unsupported";
  }
  return "unknown";
}

const char* ToString(DiagnosticCode code) noexcept {
  switch (code) {
    case DiagnosticCode::kDuplicateActorId:
      return "duplicate-actor-id";
    case DiagnosticCode::kDuplicateActorCoordinates:
      return "duplicate-actor-coordinates";
    case DiagnosticCode::kInvalidActorLane:
      return "invalid-actor-lane";
    case DiagnosticCode::kDuplicateInstructionId:
      return "duplicate-instruction-id";
    case DiagnosticCode::kUnknownInstructionActor:
      return "unknown-instruction-actor";
    case DiagnosticCode::kDuplicateAsyncOpId:
      return "duplicate-async-op-id";
    case DiagnosticCode::kUnknownEdgeEndpoint:
      return "unknown-edge-endpoint";
    case DiagnosticCode::kInvalidEdgeKind:
      return "invalid-edge-kind";
    case DiagnosticCode::kProgramOrderCrossesActors:
      return "program-order-crosses-actors";
    case DiagnosticCode::kIncompleteThreadProgramOrder:
      return "incomplete-thread-program-order";
    case DiagnosticCode::kDuplicateEdge:
      return "duplicate-edge";
    case DiagnosticCode::kCyclicSourceGraph:
      return "cyclic-source-graph";
    case DiagnosticCode::kZeroInitialArrivalCount:
      return "zero-initial-arrival-count";
    case DiagnosticCode::kDuplicateBarrierInitialization:
      return "duplicate-barrier-initialization";
    case DiagnosticCode::kMissingBarrierInitialization:
      return "missing-barrier-initialization";
    case DiagnosticCode::kDuplicateStateDefinition:
      return "duplicate-state-definition";
    case DiagnosticCode::kMissingStateDefinition:
      return "missing-state-definition";
    case DiagnosticCode::kInvalidParity:
      return "invalid-parity";
    case DiagnosticCode::kTransactionValueOutOfRange:
      return "transaction-value-out-of-range";
    case DiagnosticCode::kDuplicateCollectiveSpec:
      return "duplicate-collective-spec";
    case DiagnosticCode::kUnknownCollective:
      return "unknown-collective";
    case DiagnosticCode::kDuplicateCollectiveParticipant:
      return "duplicate-collective-participant";
    case DiagnosticCode::kUnknownCollectiveParticipant:
      return "unknown-collective-participant";
    case DiagnosticCode::kDuplicateCollectiveCall:
      return "duplicate-collective-call";
    case DiagnosticCode::kCollectiveKindMismatch:
      return "collective-kind-mismatch";
    case DiagnosticCode::kCollectiveParticipantMismatch:
      return "collective-participant-mismatch";
    case DiagnosticCode::kCollectiveCtaMismatch:
      return "collective-cta-mismatch";
    case DiagnosticCode::kCollectiveWarpMismatch:
      return "collective-warp-mismatch";
    case DiagnosticCode::kCollectiveMaskMismatch:
      return "collective-mask-mismatch";
    case DiagnosticCode::kIncompatibleCollectiveOrder:
      return "incompatible-collective-order";
    case DiagnosticCode::kRawTryWaitUnsupported:
      return "raw-try-wait-unsupported";
    case DiagnosticCode::kInvalidMilestoneReference:
      return "invalid-milestone-reference";
    case DiagnosticCode::kInvalidMemoryEffect:
      return "invalid-memory-effect";
    case DiagnosticCode::kInvalidDeferredArrival:
      return "invalid-deferred-arrival";
  }
  return "unknown";
}

}  // namespace sentryward::algorithm
