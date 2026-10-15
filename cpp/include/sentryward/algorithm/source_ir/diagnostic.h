#pragma once

#include <optional>
#include <string>

#include "sentryward/algorithm/base/source_location.h"
#include "sentryward/algorithm/base/strong_id.h"

namespace sentryward::algorithm {

enum class DiagnosticClass {
  kInvalidInput,
  kUnsupported,
};

enum class DiagnosticCode {
  kDuplicateActorId,
  kDuplicateActorCoordinates,
  kInvalidActorLane,
  kDuplicateInstructionId,
  kUnknownInstructionActor,
  kDuplicateAsyncOpId,
  kUnknownEdgeEndpoint,
  kInvalidEdgeKind,
  kProgramOrderCrossesActors,
  kIncompleteThreadProgramOrder,
  kDuplicateEdge,
  kCyclicSourceGraph,
  kZeroInitialArrivalCount,
  kDuplicateBarrierInitialization,
  kMissingBarrierInitialization,
  kDuplicateStateDefinition,
  kMissingStateDefinition,
  kInvalidParity,
  kTransactionValueOutOfRange,
  kDuplicateCollectiveSpec,
  kUnknownCollective,
  kDuplicateCollectiveParticipant,
  kUnknownCollectiveParticipant,
  kDuplicateCollectiveCall,
  kCollectiveKindMismatch,
  kCollectiveParticipantMismatch,
  kCollectiveCtaMismatch,
  kCollectiveWarpMismatch,
  kCollectiveMaskMismatch,
  kIncompatibleCollectiveOrder,
  kRawTryWaitUnsupported,
  kInvalidMilestoneReference,
  kInvalidMemoryEffect,
  kInvalidDeferredArrival,
};

struct Diagnostic final {
  DiagnosticClass classification;
  DiagnosticCode code;
  std::string message;
  std::optional<InstId> instruction;
  std::optional<SourceLocation> location;

  friend bool operator==(const Diagnostic&, const Diagnostic&) = default;
};

[[nodiscard]] const char* ToString(DiagnosticClass classification) noexcept;
[[nodiscard]] const char* ToString(DiagnosticCode code) noexcept;

}  // namespace sentryward::algorithm
