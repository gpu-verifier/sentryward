#pragma once

#include <cstddef>
#include <map>
#include <optional>
#include <span>
#include <variant>
#include <vector>

#include "sentryward/algorithm/base/compact_id_map.h"
#include "sentryward/algorithm/base/affine_sequence.h"
#include "sentryward/algorithm/base/strong_id.h"
#include "sentryward/algorithm/event_ir/event_families.h"
#include "sentryward/algorithm/event_ir/family_values.h"
#include "sentryward/algorithm/semantics/barrier_state.h"

namespace sentryward::algorithm {

using TraceIndex = std::size_t;

struct NormalTraceEntry final {
  EventId event;

  friend bool operator==(const NormalTraceEntry&,
                         const NormalTraceEntry&) = default;
};

struct BarrierPhaseCompletionTraceEntry final {
  PhaseKey phase;

  friend bool operator==(const BarrierPhaseCompletionTraceEntry&,
                         const BarrierPhaseCompletionTraceEntry&) = default;
};

struct CollectiveCompletionTraceEntry final {
  SyncId sync_id;
  EventId event;

  friend bool operator==(const CollectiveCompletionTraceEntry&,
                         const CollectiveCompletionTraceEntry&) = default;
};

using ReferenceTraceEntry =
    std::variant<NormalTraceEntry, BarrierPhaseCompletionTraceEntry,
                 CollectiveCompletionTraceEntry>;

struct TraceEventIdCodec final {
  using Delta = std::int64_t;
  static std::optional<Delta> difference(EventId a, EventId b) {
    return std::int64_t(b.value()) - a.value();
  }
  static EventId advance(EventId first, Delta delta, std::size_t index) {
    return EventId(static_cast<std::uint32_t>(first.value() + delta * std::int64_t(index)));
  }
};
struct TraceEntryCodec final {
  using Delta = std::int64_t;
  static std::optional<Delta> difference(const ReferenceTraceEntry& a, const ReferenceTraceEntry& b) {
    auto x = std::get_if<NormalTraceEntry>(&a), y = std::get_if<NormalTraceEntry>(&b);
    if (!x || !y) return {};
    return TraceEventIdCodec::difference(x->event, y->event);
  }
  static ReferenceTraceEntry advance(ReferenceTraceEntry first, Delta delta, std::size_t index) {
    if (auto* value = std::get_if<NormalTraceEntry>(&first))
      value->event = TraceEventIdCodec::advance(value->event, delta, index);
    return first;
  }
};
using TraceEntries = AffineSequence<ReferenceTraceEntry, TraceEntryCodec>;
using TraceEventIds = AffineSequence<EventId, TraceEventIdCodec>;

// Records normal and urgent events from one reference execution.
class ReferenceTrace final {
public:
  [[nodiscard]] const TraceEntries& entries() const noexcept;
  [[nodiscard]] const TraceEventIds& normalEvents() const noexcept;
  [[nodiscard]] const TraceEventIds& executedEvents() const noexcept;

  [[nodiscard]] std::optional<TraceIndex> indexOf(EventId event) const;
  [[nodiscard]] std::optional<std::uint32_t> effectPhase(EventId event) const;
  [[nodiscard]] std::optional<PhaseKey> waitTarget(EventId event) const;
  [[nodiscard]] std::optional<BarrierLifetimeId>
  initialWaitBarrier(EventId event) const;
  [[nodiscard]] std::optional<StateToken> stateToken(StateId state) const;
  [[nodiscard]] std::optional<TraceIndex> completionIndex(PhaseKey phase) const;
  [[nodiscard]] std::optional<TraceIndex>
  collectiveCompletionIndex(SyncId sync_id) const;

  [[nodiscard]] const FamilyValues<std::uint32_t>&
  effectPhases() const noexcept;
  [[nodiscard]] const FamilyValues<PhaseKey>& waitTargets() const noexcept;
  [[nodiscard]] const FamilyValues<BarrierLifetimeId>&
  initialWaits() const noexcept;
  [[nodiscard]] const std::map<StateId, StateToken>&
  stateTokens() const noexcept;

private:
  friend class ReferenceSimulator;
  friend class ReferenceTraceRecorder;
  friend void ReportTraceMemory(const ReferenceTrace& trace,
                                const char* checkpoint);

  void reserveEvents(std::size_t count);
  void appendNormal(EventId event);
  void appendNormalBatch(EventId first, std::uint32_t stride, std::size_t count);
  void appendBarrierCompletion(PhaseKey phase);
  void appendCollectiveCompletion(SyncId sync_id, EventId event);
  void discardLastNormal(EventId event);
  void recordEffectPhase(EventId event, std::uint32_t phase);
  void recordWaitTarget(EventId event, PhaseKey phase);
  void recordInitialWait(EventId event, BarrierLifetimeId barrier);
  void recordStateToken(StateId state, StateToken token);

  TraceEntries entries_;
  TraceEventIds normal_events_;
  TraceEventIds executed_events_;
  CompactIdMap<EventId, TraceIndex> event_indices_;
  FamilyTraceIndex family_indices_;
  FamilyValues<std::uint32_t> effect_phases_;
  FamilyValues<PhaseKey> wait_targets_;
  FamilyValues<BarrierLifetimeId> initial_waits_;
  std::map<StateId, StateToken> state_tokens_;
  std::map<PhaseKey, TraceIndex> phase_completion_indices_;
  std::map<SyncId, TraceIndex> collective_completion_indices_;
};

} // namespace sentryward::algorithm
