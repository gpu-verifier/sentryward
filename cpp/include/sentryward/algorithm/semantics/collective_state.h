#pragma once

#include <cstddef>
#include <map>
#include <span>
#include <vector>

#include "sentryward/algorithm/base/result.h"
#include "sentryward/algorithm/base/strong_id.h"

namespace sentryward::algorithm {

enum class CollectiveRuntimeErrorCode {
  kUnexpectedParticipant,
  kDuplicateArrival,
  kCompletionBeforeAllArrivals,
  kDuplicateCompletion,
};

struct CollectiveRuntimeError final {
  CollectiveRuntimeErrorCode code;
  SyncId sync_id;
  EventId event;
};

class CollectiveRuntime final {
 public:
  CollectiveRuntime(SyncId sync_id, std::span<const EventId> issue_events);

  [[nodiscard]] Result<void, CollectiveRuntimeError> Arrive(EventId issue);
  [[nodiscard]] std::size_t batchPrefix(EventId first, std::uint32_t stride, std::size_t count) const;
  [[nodiscard]] Result<void, CollectiveRuntimeError> ArriveBatch(EventId first, std::uint32_t stride, std::size_t count);
  [[nodiscard]] Result<void, CollectiveRuntimeError> Complete(
      EventId completion_event);

  [[nodiscard]] SyncId syncId() const noexcept { return sync_id_; }
  [[nodiscard]] bool readyToComplete() const noexcept {
    return arrived_ == expected_ && !completed_;
  }
  [[nodiscard]] bool completed() const noexcept { return completed_; }
  [[nodiscard]] std::size_t expectedCount() const noexcept {
    return expected_;
  }
  [[nodiscard]] std::size_t arrivalCount() const noexcept {
    return arrived_;
  }

 private:
  SyncId sync_id_;
  struct Group {
    EventId first;
    std::uint32_t stride;
    std::size_t count;
    std::map<std::size_t, std::size_t> arrived;
  };
  std::size_t groupIndex(EventId event) const;
  std::vector<Group> groups_;
  std::size_t expected_ = 0, arrived_ = 0;
  bool completed_ = false;
};

}  // namespace sentryward::algorithm
