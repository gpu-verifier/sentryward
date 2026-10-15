#include "sentryward/algorithm/diagnostics/scoped_deadline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace sentryward::algorithm {

ScopedWellSyncDeadline::ScopedWellSyncDeadline(std::optional<double> seconds) {
  if (!seconds) return;
  if (!std::isfinite(*seconds) || *seconds <= 0)
    throw std::invalid_argument("WellSync timeout must be finite and positive");
  const auto started = std::chrono::steady_clock::now();
  watchdog_ = std::thread([this, started, seconds = *seconds] {
    std::unique_lock lock(mutex_);
    // Subtract thread startup latency without converting an arbitrarily large
    // double timeout into the clock's integral duration representation.
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    double remaining = seconds - elapsed;
    while (!finished_ && remaining > 0) {
      // Bounded waits also avoid overflowing condition_variable's clock math.
      condition_.wait_for(lock, std::chrono::duration<double>(
          std::min(remaining, 60.0)), [this] { return finished_; });
      remaining = seconds - std::chrono::duration<double>(
          std::chrono::steady_clock::now() - started).count();
    }
    if (finished_) return;
    std::fputs("\nSENTRYWARD_TIMEOUT wellsync\n", stderr);
    std::fflush(stderr);
    std::_Exit(124);
  });
}

ScopedWellSyncDeadline::~ScopedWellSyncDeadline() {
  if (!watchdog_.joinable()) return;
  {
    std::lock_guard lock(mutex_);
    finished_ = true;
  }
  condition_.notify_one();
  watchdog_.join();
}

}  // namespace sentryward::algorithm
