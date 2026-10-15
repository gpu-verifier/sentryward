#pragma once

#include <optional>

#include "sentryward/algorithm/analysis/reachability.h"
#include "sentryward/algorithm/checker/result.h"
#include "sentryward/algorithm/source_ir/program.h"
#include "sentryward/algorithm/event_ir/program.h"

namespace sentryward::algorithm {

struct CheckerOptions final {
  bool reject_negative_tx = false;
  bool build_counterexample_schedule = true;
  bool use_bitset_reachability_after_freeze = true;
  // Failure ideals are always retained.
  bool retain_successful_ideals = false;
  // Share wait reachability work after all phase boundaries and ordering have
  // been certified. Generic WaitChecker calls do not assume that precondition.
  bool use_batched_wait_reachability = true;
  ReachabilityOptions reachability = {};
  // Optional hard process deadline, active only while Run is on the stack.
  std::optional<double> timeout_seconds = std::nullopt;
};

class WellSyncChecker final {
 public:
  [[nodiscard]] CheckResult Run(
      const SourceProgram& source,
      const CheckerOptions& options = CheckerOptions{}) const;

  // Pipeline entry: prepared must come from a successful Lowerer::Run on this
  // unchanged source with CollectiveOrdering::kExecution. It is borrowed only
  // for the duration of the call.
  [[nodiscard]] CheckResult RunPrepared(const SourceProgram& source,
      const LoweredProgram& prepared, const CheckerOptions& options = {}) const;
 private:
  [[nodiscard]] CheckResult RunImpl(const SourceProgram& source,
      const CheckerOptions& options, const LoweredProgram* prepared) const;
  // Synchronization checks on the execution-order relation. The certificate's
  // proof graph still carries the rendezvous of warp data collectives.
  [[nodiscard]] CheckResult RunExecutionOrder(const SourceProgram& source,
      const CheckerOptions& options, const LoweredProgram* prepared) const;
};

}  // namespace sentryward::algorithm
