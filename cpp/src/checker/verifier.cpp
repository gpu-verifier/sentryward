#include "sentryward/algorithm/checker/verifier.h"

#include <stdexcept>
#include <utility>

namespace sentryward::algorithm {

VerificationStatus VerificationResult::status() const {
  switch (synchronization.status()) {
    case CheckStatus::kRejected:
      return VerificationStatus::kSynchronizationRejected;
    case CheckStatus::kUnsupported:
      return VerificationStatus::kUnsupported;
    case CheckStatus::kInvalidInput:
      return VerificationStatus::kInvalidInput;
    case CheckStatus::kPassed:
      break;
  }
  if (memory_error) {
    return VerificationStatus::kInvalidInput;
  }
  if (!memory) {
    throw std::logic_error("successful synchronization lacks a memory result");
  }
  switch (memory->status) {
    case MemoryRaceStatus::kRaceFound:
      return VerificationStatus::kRaceFound;
    case MemoryRaceStatus::kRaceFree:
      return VerificationStatus::kRaceFree;
    case MemoryRaceStatus::kUnresolved:
      return VerificationStatus::kUnresolved;
  }
  throw std::logic_error("invalid memory race status");
}

VerificationResult Verifier::Run(const SourceProgram& source,
                                 MemoryModel& memory,
                                 const VerificationOptions& options) const {
  VerificationResult result{
      WellSyncChecker{}.Run(source, options.synchronization), {}, {}};
  if (!result.synchronization.passed()) {
    return result;
  }
  auto analysis = MemoryRaceChecker{}.Run(
      source, *result.synchronization.certificate(), memory, options.memory);
  if (analysis) {
    result.memory = std::move(analysis).value();
  } else {
    result.memory_error = std::move(analysis).error();
  }
  return result;
}

}  // namespace sentryward::algorithm
