#pragma once
// Look-ahead production on worker threads (class mode's unit executions,
// run.cpp). Items 0..count-1 are produced on up to `threads` workers, at most
// `window` items ahead of the consumer, and taken in index order on the
// consumer's thread. A producer's exception is rethrown by the Take of its item,
// so the consumer sees failures exactly where sequential production would.
// Destruction cancels: the workers' poll throws at its next call, finished items
// are dropped, and the workers are joined (before, e.g., a restart's exec).
// With threads == 0, Take produces inline with the consumer's own poll.
//
// Cost: O(1) locking per item; at most `window` items are alive beyond the one
// being consumed (the caller bounds the memory of in-flight items by `window`).
#include "sentryward/algorithm/diagnostics/performance_profile.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

namespace sentryward::frontend::ptx {
template <class T> class LookAhead {
public:
  // `poll` is handed to the producer: the consumer's own poll inline, a
  // cancellation check on the workers.
  using Produce = std::function<T(std::size_t index, const std::function<void()> &poll)>;
  LookAhead(std::size_t count, std::size_t threads, std::size_t window, Produce produce,
            std::function<void()> inline_poll)
      : count_(count), window_(window == 0 ? 1 : window), produce_(std::move(produce)),
        inline_poll_(std::move(inline_poll)), profile_(algorithm::CurrentPerformanceProfile()) {
    try {
      for (std::size_t i = 0; i < threads && i < count; ++i) workers_.emplace_back([this] { Work(); });
    } catch (...) {
      Stop();
      throw;
    }
  }
  ~LookAhead() { Stop(); }
  LookAhead(const LookAhead &) = delete;
  LookAhead &operator=(const LookAhead &) = delete;

  // The next item in index order.
  T Take() {
    const auto index = taken_;
    if (workers_.empty()) {
      ++taken_;
      return produce_(index, inline_poll_);
    }
    std::unique_lock lock(mutex_);
    produced_.wait(lock, [&] { return ready_.contains(index); });
    auto node = ready_.extract(index);
    ++taken_;
    lock.unlock();
    space_.notify_all();
    auto &slot = node.mapped();
    if (slot.error) std::rethrow_exception(slot.error);
    return std::move(*slot.value);
  }

private:
  struct Cancelled {};
  struct Slot {
    std::optional<T> value;
    std::exception_ptr error;
  };
  void Work() {
    algorithm::ProfileWorkerSession session(profile_);
    const std::function<void()> poll = [this] {
      if (stopping_.load(std::memory_order_relaxed)) throw Cancelled{};
    };
    for (;;) {
      std::size_t index;
      {
        std::unique_lock lock(mutex_);
        space_.wait(lock, [&] { return stopping_.load() || next_ >= count_ || next_ < taken_ + window_; });
        if (stopping_.load() || next_ >= count_) return;
        index = next_++;
      }
      Slot slot;
      try {
        slot.value.emplace(produce_(index, poll));
      } catch (...) {
        slot.error = std::current_exception();
      }
      {
        std::lock_guard lock(mutex_);
        if (stopping_.load()) return;
        ready_.emplace(index, std::move(slot));
      }
      produced_.notify_all();
    }
  }
  void Stop() noexcept {
    {
      std::lock_guard lock(mutex_);
      stopping_.store(true);
    }
    space_.notify_all();
    for (auto &worker : workers_)
      if (worker.joinable()) worker.join();
    workers_.clear();
  }

  const std::size_t count_, window_;
  Produce produce_;
  std::function<void()> inline_poll_;
  algorithm::PerformanceProfile *profile_;
  std::mutex mutex_;
  std::condition_variable space_, produced_;
  std::atomic<bool> stopping_{false};
  std::size_t next_ = 0, taken_ = 0; // under mutex_ once workers exist (taken_ is written by Take only)
  std::map<std::size_t, Slot> ready_;
  std::vector<std::thread> workers_;
};
} // namespace sentryward::frontend::ptx
