#pragma once

#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "sentryward/algorithm/analysis/phase_boundary.h"
#include "sentryward/algorithm/analysis/phase_database.h"
#include "sentryward/algorithm/analysis/happens_before_graph.h"
#include "sentryward/algorithm/analysis/wait_checker.h"
#include "sentryward/algorithm/checker/certificate.h"
#include "sentryward/algorithm/checker/witness.h"
#include "sentryward/algorithm/source_ir/diagnostic.h"

namespace sentryward::algorithm {

enum class CheckStatus {
  kPassed,
  kRejected,
  kUnsupported,
  kInvalidInput,
};

enum class UnsupportedReason {
  kSourceFeature,
  kSignedTransactionCertificateUnsupported,
};

struct UnsupportedInput final {
  UnsupportedReason reason = UnsupportedReason::kSourceFeature;
  std::vector<Diagnostic> diagnostics;
  std::string message;
};

struct DerivedEdgeTraceOrderError final {
  EventId from;
  EventId to;
  std::optional<TraceIndex> from_index;
  std::optional<TraceIndex> to_index;

  friend bool operator==(const DerivedEdgeTraceOrderError&,
                         const DerivedEdgeTraceOrderError&) = default;
};

using InvalidInputCause =
    std::variant<std::vector<Diagnostic>, Diagnostic, SimulationError,
                 PhaseDatabaseError, PhaseBoundaryError, WaitCheckError,
                 LifecycleCheckError, HappensBeforeGraphMutationError,
                 DerivedEdgeTraceOrderError>;

struct InvalidInput final {
  InvalidInputCause cause;
  std::string message;
};

using CheckDetail = std::variant<PassCertificate, RejectionWitness,
                                 UnsupportedInput, InvalidInput>;

class CheckResult final {
public:
  [[nodiscard]] static CheckResult Passed(PassCertificate certificate) {
    return CheckResult(CheckStatus::kPassed, std::move(certificate));
  }

  [[nodiscard]] static CheckResult
  Rejected(RejectionWitness witness,
           std::optional<RejectionContext> context = std::nullopt) {
    CheckResult result(CheckStatus::kRejected, std::move(witness));
    result.rejection_context_ = std::move(context);
    return result;
  }

  [[nodiscard]] static CheckResult Unsupported(UnsupportedInput input) {
    return CheckResult(CheckStatus::kUnsupported, std::move(input));
  }

  [[nodiscard]] static CheckResult Invalid(InvalidInput input) {
    return CheckResult(CheckStatus::kInvalidInput, std::move(input));
  }

  [[nodiscard]] CheckStatus status() const noexcept { return status_; }
  [[nodiscard]] bool passed() const noexcept {
    return status_ == CheckStatus::kPassed;
  }
  [[nodiscard]] const CheckDetail& detail() const noexcept { return detail_; }

  [[nodiscard]] const PassCertificate* certificate() const noexcept {
    return std::get_if<PassCertificate>(&detail_);
  }
  [[nodiscard]] PassCertificate* certificate() noexcept {
    return std::get_if<PassCertificate>(&detail_);
  }

  [[nodiscard]] const RejectionWitness* rejection() const noexcept {
    return std::get_if<RejectionWitness>(&detail_);
  }

  [[nodiscard]] const RejectionContext* rejectionContext() const noexcept {
    return rejection_context_ ? &*rejection_context_ : nullptr;
  }

  [[nodiscard]] const UnsupportedInput* unsupported() const noexcept {
    return std::get_if<UnsupportedInput>(&detail_);
  }

  [[nodiscard]] const InvalidInput* invalidInput() const noexcept {
    return std::get_if<InvalidInput>(&detail_);
  }

private:
  template <typename Detail>
  CheckResult(CheckStatus status, Detail detail)
      : status_(status), detail_(std::move(detail)) {}

  CheckStatus status_;
  CheckDetail detail_;
  std::optional<RejectionContext> rejection_context_;
};

} // namespace sentryward::algorithm
