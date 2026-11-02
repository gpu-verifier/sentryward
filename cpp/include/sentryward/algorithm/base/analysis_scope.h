#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <typeindex>
#include <utility>
#include <vector>

namespace sentryward::algorithm {

// The process-level state of one analysis attempt. The driver (run.cpp)
// analyses a launch in one attempt, or in two when a class-mode attempt
// restarts per unit. Each attempt owns its identifier counters and the tables
// that analysis passes share without an owner object, so a per-unit attempt
// starts from the same state as a fresh process. One scope is installed at a
// time, for the whole attempt; worker threads start and finish inside it.
//
// Identifiers restart at 1 in every attempt and carry no attempt tag: some
// reach the report (value provenance), which must not depend on whether an
// attempt came first. An identifier therefore never outlives its attempt:
// every object that holds one is created after the scope by the attempt
// (run.cpp's RunAttempt) and destroyed before it.
class AnalysisScope {
 public:
  // Identifier spaces, each numbered 1, 2, ... per attempt.
  enum class Ids : unsigned {
    kValueGraph,
    kAddressPool,
    kScalarFoldGraph,
    kEGraphContext,
    kCausalRegionCertificate,
    kFinitePrefixCertificate,
    kMomentCertificate,
    kCount
  };

  AnalysisScope() {
    for (auto &next : next_) next.store(1, std::memory_order_relaxed);
    AnalysisScope *expected = nullptr;
    if (!current_.compare_exchange_strong(expected, this, std::memory_order_acq_rel))
      throw std::logic_error("internal error: nested analysis scope");
  }
  ~AnalysisScope() { current_.store(nullptr, std::memory_order_release); }
  AnalysisScope(const AnalysisScope &) = delete;
  AnalysisScope &operator=(const AnalysisScope &) = delete;

  static AnalysisScope &Current() {
    auto *scope = current_.load(std::memory_order_acquire);
    if (!scope) throw std::logic_error("internal error: analysis state used outside an analysis scope");
    return *scope;
  }
  // The next identifier of `space`.
  std::uint64_t Next(Ids space) {
    auto &next = next_[static_cast<std::size_t>(space)];
    auto value = next.load(std::memory_order_relaxed);
    do {
      if (value == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("analysis identifier space exhausted");
    } while (!next.compare_exchange_weak(value, value + 1, std::memory_order_relaxed));
    return value;
  }
  // The attempt's T, value-initialized on first use. O(kinds of T) under a lock.
  // For state of a higher layer (value analysis) that this base layer cannot
  // name as a member.
  template <class T> T &Local() {
    std::lock_guard lock(mutex_);
    for (const auto &[type, value] : locals_)
      if (type == std::type_index(typeid(T))) return *static_cast<T *>(value.get());
    auto value = std::make_shared<T>();
    auto &result = *value;
    locals_.emplace_back(std::type_index(typeid(T)), std::move(value));
    return result;
  }

 private:
  inline static std::atomic<AnalysisScope *> current_ = nullptr;
  std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(Ids::kCount)> next_;
  std::mutex mutex_;
  std::vector<std::pair<std::type_index, std::shared_ptr<void>>> locals_;
};

}  // namespace sentryward::algorithm
