#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

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

// An id rendered into a diagnostic message: its kind, its value and the
// [begin, begin + size) span of its decimal text. Consumers compare messages
// with ids taken relative to their CTA (repeated entries, source_json.cpp).
enum class MessageIdKind { kInstruction, kActor, kBarrierLifetime, kOther };
struct MessageId final {
  MessageIdKind kind;
  std::uint32_t value;
  std::size_t begin;
  std::size_t size;

  friend bool operator==(const MessageId&, const MessageId&) = default;
};

// A message under construction and the ids rendered into it, in text order.
struct DiagnosticMessage final {
  std::string text;
  std::vector<MessageId> ids;

  DiagnosticMessage(const char* message) : text(message) {}
  DiagnosticMessage(std::string message) : text(std::move(message)) {}
  friend DiagnosticMessage operator+(DiagnosticMessage left,
                                     const DiagnosticMessage& right) {
    for (auto id : right.ids) {
      id.begin += left.text.size();
      left.ids.push_back(id);
    }
    left.text += right.text;
    return left;
  }
};

// The decimal text of `id`, recorded with its kind.
template <typename Id>
[[nodiscard]] DiagnosticMessage MessageIdText(Id id) {
  MessageIdKind kind = MessageIdKind::kOther;
  if constexpr (std::is_same_v<Id, InstId>) kind = MessageIdKind::kInstruction;
  else if constexpr (std::is_same_v<Id, ActorId>) kind = MessageIdKind::kActor;
  else if constexpr (std::is_same_v<Id, BarrierLifetimeId>) kind = MessageIdKind::kBarrierLifetime;
  DiagnosticMessage message(std::to_string(id.value()));
  message.ids.push_back({kind, id.value(), 0, message.text.size()});
  return message;
}

struct Diagnostic final {
  DiagnosticClass classification;
  DiagnosticCode code;
  std::string message;
  std::optional<InstId> instruction;
  std::optional<SourceLocation> location;
  // The ids rendered into `message` (DiagnosticMessage), in text order.
  std::vector<MessageId> message_ids;

  friend bool operator==(const Diagnostic&, const Diagnostic&) = default;
};

[[nodiscard]] const char* ToString(DiagnosticClass classification) noexcept;
[[nodiscard]] const char* ToString(DiagnosticCode code) noexcept;

}  // namespace sentryward::algorithm
