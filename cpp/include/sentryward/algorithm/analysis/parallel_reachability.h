#pragma once

#include <cstddef>
#include <functional>
#include <memory>

#include "sentryward/algorithm/analysis/reachability.h"

namespace sentryward::algorithm {

// A persistent pool sharing only the read-only graph. Each worker owns its
// oracle, scratch space and bounded cache. The caller must not mutate the graph
// until For returns. Batches are synchronous, non-reentrant, and may throw.
class ParallelReachability final {
 public:
  ParallelReachability(const HappensBeforeGraph& graph,
                       ReachabilityOptions options, std::size_t threads);
  ~ParallelReachability();
  ParallelReachability(const ParallelReachability&) = delete;
  ParallelReachability& operator=(const ParallelReachability&) = delete;

  [[nodiscard]] bool enabled() const noexcept;
  void For(std::size_t count,
           const std::function<void(std::size_t, ReachabilityOracle&)>& task);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sentryward::algorithm
