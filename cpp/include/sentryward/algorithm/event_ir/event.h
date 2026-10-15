#pragma once

#include <cstdint>
#include <optional>
#include <variant>

#include "sentryward/algorithm/base/strong_id.h"
#include "sentryward/algorithm/source_ir/async.h"

namespace sentryward::algorithm {

enum class MemoryDirection {
  kLoad,
  kStore,
};

enum class MemoryMode {
  kImmediate,
  kAsync,
};

enum class EventSchedulingClass {
  kNormal,
  kUrgentDerived,
};

struct ImmediateMemoryIssueEvent final {
  MemoryDirection direction;
  AddressId address;
  std::uint32_t bytes;

  friend bool operator==(const ImmediateMemoryIssueEvent&,
                         const ImmediateMemoryIssueEvent&) = default;
};

struct ImmediateMemoryExecEvent final {
  MemoryDirection direction;
  AddressId address;
  std::uint32_t bytes;

  friend bool operator==(const ImmediateMemoryExecEvent&,
                         const ImmediateMemoryExecEvent&) = default;
};

struct AsyncMemoryIssueEvent final {
  MemoryDirection direction;
  AsyncOpId async_id;
  AddressId source;
  AddressId destination;
  std::uint64_t bytes;

  friend bool operator==(const AsyncMemoryIssueEvent&,
                         const AsyncMemoryIssueEvent&) = default;
};

struct AsyncMemoryExecEvent final {
  MemoryDirection direction;
  AsyncOpId async_id;
  AddressId source;
  AddressId destination;
  std::uint64_t bytes;
  std::optional<CompleteTxEffect> completion;

  friend bool operator==(const AsyncMemoryExecEvent&,
                         const AsyncMemoryExecEvent&) = default;
};

struct GenericAsyncIssueEvent final {
  AsyncOpId async_id;
  std::vector<BarrierLifetimeId> barriers = {};
  friend bool operator==(const GenericAsyncIssueEvent&,
                         const GenericAsyncIssueEvent&) = default;
};

struct GenericAsyncEffectEvent final {
  AsyncOpId async_id;
  std::uint32_t effect_index;
  SharedArray<MemoryEffect> accesses;
  std::optional<CompletionAction> completion;
  // Access metadata lives in the compute input, keyed by origin/effect index.
  bool detached_accesses = false;
  bool checks_issue_initialization = true;
  std::uint32_t detached_access_count = 0;
  bool detached_transfer = false;
  friend bool operator==(const GenericAsyncEffectEvent&,
                         const GenericAsyncEffectEvent&) = default;
};

struct OperationIssueEvent final {
  friend bool operator==(const OperationIssueEvent&, const OperationIssueEvent&) = default;
};

struct OperationCompleteEvent final {
  friend bool operator==(const OperationCompleteEvent&, const OperationCompleteEvent&) = default;
};

struct GenericAsyncReadsCompleteEvent final {
  AsyncOpId async_id;
  friend bool operator==(const GenericAsyncReadsCompleteEvent&,
                         const GenericAsyncReadsCompleteEvent&) = default;
};

struct GenericAsyncCompleteEvent final {
  AsyncOpId async_id;
  friend bool operator==(const GenericAsyncCompleteEvent&,
                         const GenericAsyncCompleteEvent&) = default;
};

struct AwaitIssueEvent final {
  friend bool operator==(const AwaitIssueEvent&,
                         const AwaitIssueEvent&) = default;
};

struct AwaitDoneEvent final {
  friend bool operator==(const AwaitDoneEvent&,
                         const AwaitDoneEvent&) = default;
};

struct MBarrierInitEvent final {
  BarrierLifetimeId barrier;
  std::uint32_t expected_arrivals;

  friend bool operator==(const MBarrierInitEvent&,
                         const MBarrierInitEvent&) = default;
};

struct MBarrierUpdateEvent final {
  BarrierLifetimeId barrier;
  std::uint32_t arrival_decrement;
  std::uint64_t transaction_increment;
  std::optional<StateId> produced_state;

  friend bool operator==(const MBarrierUpdateEvent&,
                         const MBarrierUpdateEvent&) = default;
};

struct MBarrierCompleteTxEvent final {
  BarrierLifetimeId barrier;
  std::uint64_t bytes;

  friend bool operator==(const MBarrierCompleteTxEvent&,
                         const MBarrierCompleteTxEvent&) = default;
};

struct MBarrierAddArrivalsEvent final {
  BarrierLifetimeId barrier;
  std::uint32_t arrivals;

  friend bool operator==(const MBarrierAddArrivalsEvent&,
                         const MBarrierAddArrivalsEvent&) = default;
};

struct WaitStateIssueEvent final {
  BarrierLifetimeId barrier;
  StateId state;

  friend bool operator==(const WaitStateIssueEvent&,
                         const WaitStateIssueEvent&) = default;
};

struct WaitStateDoneEvent final {
  BarrierLifetimeId barrier;
  StateId state;

  friend bool operator==(const WaitStateDoneEvent&,
                         const WaitStateDoneEvent&) = default;
};

struct WaitParityIssueEvent final {
  BarrierLifetimeId barrier;
  std::uint8_t parity;

  friend bool operator==(const WaitParityIssueEvent&,
                         const WaitParityIssueEvent&) = default;
};

struct WaitParityDoneEvent final {
  BarrierLifetimeId barrier;
  std::uint8_t parity;

  friend bool operator==(const WaitParityDoneEvent&,
                         const WaitParityDoneEvent&) = default;
};

enum class CollectiveKind {
  kSyncThreads,
  kSyncWarp,
  kWarpData,
};

struct CollectiveIssueEvent final {
  SyncId sync_id;
  CollectiveKind kind;

  friend bool operator==(const CollectiveIssueEvent&,
                         const CollectiveIssueEvent&) = default;
};

struct CollectiveDoneEvent final {
  SyncId sync_id;
  CollectiveKind kind;

  friend bool operator==(const CollectiveDoneEvent&,
                         const CollectiveDoneEvent&) = default;
};

struct CollectiveCompleteEvent final {
  SyncId sync_id;
  CollectiveKind kind;

  friend bool operator==(const CollectiveCompleteEvent&,
                         const CollectiveCompleteEvent&) = default;
};

struct MBarrierPhaseCompleteEvent final {
  BarrierLifetimeId barrier;
  std::uint32_t phase;

  friend bool operator==(const MBarrierPhaseCompleteEvent&,
                         const MBarrierPhaseCompleteEvent&) = default;
};

struct NamedBarrierArriveEvent final {
  BarrierLifetimeId barrier;
  std::uint32_t expected_arrivals;
  bool blocking;
  friend bool operator==(const NamedBarrierArriveEvent&, const NamedBarrierArriveEvent&) = default;
};

struct NamedBarrierReturnEvent final {
  BarrierLifetimeId barrier;
  friend bool operator==(const NamedBarrierReturnEvent&, const NamedBarrierReturnEvent&) = default;
};

using EventPayload =
    std::variant<ImmediateMemoryIssueEvent, ImmediateMemoryExecEvent,
                 AsyncMemoryIssueEvent, AsyncMemoryExecEvent, MBarrierInitEvent,
                 MBarrierUpdateEvent, MBarrierCompleteTxEvent,
                 WaitStateIssueEvent, WaitStateDoneEvent, WaitParityIssueEvent,
                 WaitParityDoneEvent, CollectiveIssueEvent, CollectiveDoneEvent,
                 CollectiveCompleteEvent, MBarrierPhaseCompleteEvent,
                 GenericAsyncIssueEvent, GenericAsyncEffectEvent,
                 GenericAsyncReadsCompleteEvent, GenericAsyncCompleteEvent,
                 AwaitIssueEvent, AwaitDoneEvent, MBarrierAddArrivalsEvent,
                 NamedBarrierArriveEvent, NamedBarrierReturnEvent,
                 OperationIssueEvent, OperationCompleteEvent>;

struct Event final {
  EventId id;
  EventSchedulingClass scheduling_class;
  std::optional<InstId> origin;
  std::optional<ActorId> actor;
  EventPayload payload;

  [[nodiscard]] bool isUrgentDerived() const noexcept {
    return scheduling_class == EventSchedulingClass::kUrgentDerived;
  }

  friend bool operator==(const Event&, const Event&) = default;
};

}  // namespace sentryward::algorithm
