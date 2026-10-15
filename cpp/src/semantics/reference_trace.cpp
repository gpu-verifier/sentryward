#include "sentryward/algorithm/semantics/reference_trace.h"

namespace sentryward::algorithm {

const TraceEntries& ReferenceTrace::entries() const noexcept {
  return entries_;
}

const TraceEventIds& ReferenceTrace::normalEvents() const noexcept {
  return normal_events_;
}

const TraceEventIds& ReferenceTrace::executedEvents() const noexcept {
  return executed_events_;
}

std::optional<TraceIndex> ReferenceTrace::indexOf(EventId event) const {
  if (family_indices_.enabled()) return family_indices_.get(event);
  const TraceIndex* found = event_indices_.get(event);
  return found ? std::optional(*found) : std::nullopt;
}

std::optional<std::uint32_t> ReferenceTrace::effectPhase(EventId event) const {
  return effect_phases_.get(event);
}

std::optional<PhaseKey> ReferenceTrace::waitTarget(EventId event) const {
  return wait_targets_.get(event);
}

std::optional<StateToken> ReferenceTrace::stateToken(StateId state) const {
  const auto found = state_tokens_.find(state);
  return found == state_tokens_.end() ? std::nullopt
                                      : std::optional(found->second);
}

std::optional<BarrierLifetimeId>
ReferenceTrace::initialWaitBarrier(EventId event) const {
  return initial_waits_.get(event);
}

const FamilyValues<BarrierLifetimeId>&
ReferenceTrace::initialWaits() const noexcept {
  return initial_waits_;
}

std::optional<TraceIndex>
ReferenceTrace::completionIndex(PhaseKey phase) const {
  const auto found = phase_completion_indices_.find(phase);
  return found == phase_completion_indices_.end()
             ? std::nullopt
             : std::optional(found->second);
}

std::optional<TraceIndex>
ReferenceTrace::collectiveCompletionIndex(SyncId sync_id) const {
  const auto found = collective_completion_indices_.find(sync_id);
  return found == collective_completion_indices_.end()
             ? std::nullopt
             : std::optional(found->second);
}

const FamilyValues<std::uint32_t>&
ReferenceTrace::effectPhases() const noexcept {
  return effect_phases_;
}

const FamilyValues<PhaseKey>&
ReferenceTrace::waitTargets() const noexcept {
  return wait_targets_;
}

const std::map<StateId, StateToken>&
ReferenceTrace::stateTokens() const noexcept {
  return state_tokens_;
}

void ReferenceTrace::reserveEvents(std::size_t count) {
  event_indices_.reserveDense(count);
}

void ReferenceTrace::appendNormal(EventId event) {
  const TraceIndex index = entries_.size();
  entries_.push_back(NormalTraceEntry{event});
  normal_events_.push_back(event);
  executed_events_.push_back(event);
  if (family_indices_.enabled()) family_indices_.insert(event, index);
  else event_indices_.emplace(event, index);
}

void ReferenceTrace::appendNormalBatch(EventId first, std::uint32_t stride, std::size_t count) {
  const TraceIndex index = entries_.size();
  entries_.append(NormalTraceEntry{first}, stride, count);
  normal_events_.append(first, stride, count);
  executed_events_.append(first, stride, count);
  if (family_indices_.enabled()) family_indices_.insert(first, index, count);
  else for (std::size_t i = 0; i < count; ++i)
    event_indices_.emplace(EventId(first.value() + stride * i), index + i);
}

void ReferenceTrace::appendBarrierCompletion(PhaseKey phase) {
  const TraceIndex index = entries_.size();
  entries_.push_back(BarrierPhaseCompletionTraceEntry{phase});
  phase_completion_indices_.emplace(phase, index);
}

void ReferenceTrace::appendCollectiveCompletion(SyncId sync_id, EventId event) {
  const TraceIndex index = entries_.size();
  entries_.push_back(CollectiveCompletionTraceEntry{sync_id, event});
  executed_events_.push_back(event);
  if (family_indices_.enabled()) family_indices_.insert(event, index);
  else event_indices_.emplace(event, index);
  collective_completion_indices_.emplace(sync_id, index);
}

void ReferenceTrace::discardLastNormal(EventId event) {
  if (entries_.empty() || normal_events_.empty() || executed_events_.empty() ||
      normal_events_.back() != event || executed_events_.back() != event ||
      !std::holds_alternative<NormalTraceEntry>(entries_.back()) ||
      std::get<NormalTraceEntry>(entries_.back()).event != event) {
    return;
  }
  entries_.pop_back();
  normal_events_.pop_back();
  executed_events_.pop_back();
  if (family_indices_.enabled()) family_indices_.erase(event);
  else event_indices_.erase(event);
}

void ReferenceTrace::recordEffectPhase(EventId event, std::uint32_t phase) {
  effect_phases_.emplace(event, phase);
}

void ReferenceTrace::recordWaitTarget(EventId event, PhaseKey phase) {
  wait_targets_.emplace(event, phase);
}

void ReferenceTrace::recordInitialWait(EventId event, BarrierLifetimeId barrier) {
  initial_waits_.emplace(event, barrier);
}

void ReferenceTrace::recordStateToken(StateId state, StateToken token) {
  state_tokens_.emplace(state, token);
}

} // namespace sentryward::algorithm
