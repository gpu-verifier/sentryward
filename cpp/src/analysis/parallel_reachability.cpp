#include "sentryward/algorithm/analysis/parallel_reachability.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "sentryward/algorithm/diagnostics/performance_profile.h"

namespace sentryward::algorithm {

struct ParallelReachability::Impl {
  const HappensBeforeGraph& graph;
  ReachabilityOptions options;
  std::size_t threads;
  std::vector<std::thread> workers;
  std::mutex mutex;
  std::condition_variable ready, finished;
  bool stopping = false;
  std::size_t generation = 0, remaining = 0, count = 0;
  std::atomic<std::size_t> next{0};
  std::function<void(std::size_t, ReachabilityOracle&)> task;
  std::exception_ptr failure;
  PerformanceProfile* profile = nullptr;

  Impl(const HappensBeforeGraph& g, ReachabilityOptions o, std::size_t n)
      : graph(g), options(o), threads(n) {
    if (n == 0 || n > 64) throw std::invalid_argument("sync threads must be 1..64");
  }
  ~Impl() { Stop(); }
  void Stop() {
    { std::lock_guard lock(mutex); stopping = true; }
    ready.notify_all();
    for (auto& worker : workers) if (worker.joinable()) worker.join();
  }
  void Start() {
    if (!workers.empty()) return;
    try {
      for (std::size_t i = 0; i < threads; ++i) {
        workers.emplace_back([this] {
          // Lazy construction makes allocation failures propagate via For.
          std::unique_ptr<ReachabilityOracle> oracle;
          std::size_t seen = 0;
          for (;;) {
            {
              std::unique_lock lock(mutex);
              ready.wait(lock, [&] { return stopping || generation != seen; });
              if (stopping) return;
              seen = generation;
            }
            try {
              ProfileWorkerSession session(profile);
              if (!oracle) oracle = std::make_unique<ReachabilityOracle>(graph, options);
              for (;;) {
                const auto index = next.fetch_add(1, std::memory_order_relaxed);
                if (index >= count) break;
                task(index, *oracle);
              }
            } catch (...) {
              std::lock_guard lock(mutex);
              if (!failure) failure = std::current_exception();
              next.store(count, std::memory_order_relaxed);
            }
            {
              std::lock_guard lock(mutex);
              if (--remaining == 0) finished.notify_one();
            }
          }
        });
      }
    } catch (...) {
      Stop();
      workers.clear();
      stopping = false;
      throw;
    }
  }
};

ParallelReachability::ParallelReachability(const HappensBeforeGraph& graph,
    ReachabilityOptions options, std::size_t threads)
    : impl_(std::make_unique<Impl>(graph, options, threads)) {}
ParallelReachability::~ParallelReachability() = default;
bool ParallelReachability::enabled() const noexcept { return impl_->threads > 1; }

void ParallelReachability::For(std::size_t count,
    const std::function<void(std::size_t, ReachabilityOracle&)>& task) {
  if (count == 0) return;
  ProfileScope batch("parallel.batch");
  batch.nodes = count;
  impl_->Start();
  std::unique_lock lock(impl_->mutex);
  impl_->count = count;
  impl_->next.store(0, std::memory_order_relaxed);
  impl_->task = task;
  impl_->failure = nullptr;
  impl_->profile = CurrentPerformanceProfile();
  impl_->remaining = impl_->workers.size();
  ++impl_->generation;
  impl_->ready.notify_all();
  impl_->finished.wait(lock, [&] { return impl_->remaining == 0; });
  impl_->task = {};
  if (impl_->failure) std::rethrow_exception(impl_->failure);
}

}  // namespace sentryward::algorithm
