#pragma once

#include <condition_variable>
#include <mutex>
#include <optional>
#include <thread>

namespace sentryward::algorithm {
inline constexpr double kDefaultWellSyncTimeoutSeconds = 600;

// Hard wall-clock deadline for a process-isolated checker invocation. Expiry
// writes a machine-readable stderr marker and exits the process with code 124.
// No verifier/profiler state is accessed by the watchdog. Destruction disarms
// and joins it, so neither work before construction nor later stages is timed.
class ScopedWellSyncDeadline final {
 public:
  explicit ScopedWellSyncDeadline(std::optional<double> seconds);
  ~ScopedWellSyncDeadline();
  ScopedWellSyncDeadline(const ScopedWellSyncDeadline&) = delete;
  ScopedWellSyncDeadline& operator=(const ScopedWellSyncDeadline&) = delete;

 private:
  std::mutex mutex_;
  std::condition_variable condition_;
  bool finished_ = false;
  std::thread watchdog_;
};

}  // namespace sentryward::algorithm
