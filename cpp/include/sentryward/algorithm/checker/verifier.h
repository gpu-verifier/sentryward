#pragma once

#include <optional>

#include "sentryward/algorithm/checker/memory_race_checker.h"
#include "sentryward/algorithm/checker/wellsync_checker.h"

namespace sentryward::algorithm {

enum class VerificationStatus {
  kSynchronizationRejected,
  kUnsupported,
  kInvalidInput,
  kRaceFound,
  kRaceFree,
  kUnresolved,
};

struct VerificationOptions final {
  CheckerOptions synchronization;
  MemoryRaceOptions memory;
};

struct VerificationResult final {
  CheckResult synchronization;
  std::optional<MemoryRaceResult> memory;
  std::optional<MemoryAnalysisError> memory_error;

  [[nodiscard]] VerificationStatus status() const;
};

class Verifier final {
 public:
  [[nodiscard]] VerificationResult Run(
      const SourceProgram& source, MemoryModel& memory,
      const VerificationOptions& options = {}) const;
};

}  // namespace sentryward::algorithm
