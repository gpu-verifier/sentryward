#include "sentryward/algorithm/semantics/collective_state.h"
#include <algorithm>

namespace sentryward::algorithm {

CollectiveRuntime::CollectiveRuntime(SyncId sync_id,
                                     std::span<const EventId> issue_events)
    : sync_id_(sync_id) {
  std::vector<EventId> sorted(issue_events.begin(), issue_events.end());
  std::sort(sorted.begin(), sorted.end());
  sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
  expected_ = sorted.size();
  for (auto id : sorted) {
    if (!groups_.empty()) {
      auto& group = groups_.back();
      if (group.count == 1) { group.stride = id.value() - group.first.value(); ++group.count; continue; }
      if (id.value() == group.first.value() + group.stride * group.count) { ++group.count; continue; }
    }
    groups_.push_back({id, 1, 1, {}});
  }
}

std::size_t CollectiveRuntime::groupIndex(EventId event) const {
  auto it = std::upper_bound(groups_.begin(), groups_.end(), event,
      [](auto id, const auto& group) { return id < group.first; });
  if (it == groups_.begin()) return groups_.size();
  --it;
  const auto offset = event.value() - it->first.value();
  if (offset % it->stride || offset / it->stride >= it->count) return groups_.size();
  return it - groups_.begin();
}

std::size_t CollectiveRuntime::batchPrefix(EventId first, std::uint32_t stride, std::size_t count) const {
  const auto index = groupIndex(first);
  if (index == groups_.size()) return 0;
  const auto& group = groups_[index];
  if (stride != group.stride) return std::min<std::size_t>(count, 1);
  return std::min(count, group.count - (first.value() - group.first.value()) / stride);
}

Result<void, CollectiveRuntimeError> CollectiveRuntime::Arrive(EventId issue) {
  return ArriveBatch(issue, 0, 1);
}

Result<void, CollectiveRuntimeError> CollectiveRuntime::ArriveBatch(EventId issue, std::uint32_t stride, std::size_t count) {
  if (!count || batchPrefix(issue, stride, count) != count) {
    return Result<void, CollectiveRuntimeError>::Failure(
        {CollectiveRuntimeErrorCode::kUnexpectedParticipant, sync_id_, issue});
  }
  auto& group = groups_[groupIndex(issue)];
  auto begin = (issue.value() - group.first.value()) / group.stride;
  auto end = begin + count;
  auto next = group.arrived.lower_bound(begin);
  if ((next != group.arrived.end() && next->first < end) ||
      (next != group.arrived.begin() && std::prev(next)->second > begin)) {
    return Result<void, CollectiveRuntimeError>::Failure(
        {CollectiveRuntimeErrorCode::kDuplicateArrival, sync_id_, issue});
  }
  if (next != group.arrived.begin() && std::prev(next)->second == begin) {
    const auto previous = std::prev(next);
    begin = previous->first; group.arrived.erase(previous);
  }
  if (next != group.arrived.end() && next->first == end) {
    end = next->second; group.arrived.erase(next);
  }
  group.arrived.emplace(begin, end);
  arrived_ += count;
  return Result<void, CollectiveRuntimeError>::Success();
}

Result<void, CollectiveRuntimeError> CollectiveRuntime::Complete(
    EventId completion_event) {
  if (completed_) {
    return Result<void, CollectiveRuntimeError>::Failure(
        {CollectiveRuntimeErrorCode::kDuplicateCompletion, sync_id_,
         completion_event});
  }
  if (!readyToComplete()) {
    return Result<void, CollectiveRuntimeError>::Failure(
        {CollectiveRuntimeErrorCode::kCompletionBeforeAllArrivals, sync_id_,
         completion_event});
  }
  completed_ = true;
  return Result<void, CollectiveRuntimeError>::Success();
}

}  // namespace sentryward::algorithm
