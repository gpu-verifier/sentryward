#pragma once

#include <cstdint>
#include <functional>
#include <iosfwd>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "sentryward/algorithm/analysis/reachability.h"
#include "sentryward/algorithm/base/shared_optional.h"
#include "sentryward/algorithm/base/shared_string.h"
#include "sentryward/algorithm/memory/symbolic_address_model.h"
#include "sentryward/algorithm/checker/memory_race_checker.h"
#include "sentryward/algorithm/source_ir/program.h"
#include "sentryward/algorithm/source_ir/ptx_assumptions.h"

namespace sentryward::algorithm {

struct RequiredOrder final {
  SharedString kind;
  MilestoneRef from;
  MilestoneRef to;
  SharedString message;
  // Conjunction of effect endpoints; absent preserves the literal milestone.
  SharedOptional<std::vector<std::uint32_t>> from_effects = std::nullopt;
  // A universal target domain, emitted before taking a source/target product.
  // Absent means the singleton `to`; present storage is immutable and shared
  // across source selections. `to` remains the first endpoint for diagnostics.
  SharedOptional<std::vector<MilestoneRef>> target_domain = std::nullopt;
  [[nodiscard]] std::span<const MilestoneRef> targets() const {
    if (target_domain) return *target_domain;
    return {&to, 1};
  }
};

struct VerificationLimitation final {
  std::string code;
  std::string message;
};

struct SourceViolation final {
  std::string code;
  std::string message;
};

struct SourceArtifact final {
  SourceProgram program;
  std::vector<ConstantAddressBinding> addresses;
  std::vector<RequiredOrder> required_orders;
  std::vector<VerificationLimitation> verification_limitations;
  std::vector<SourceViolation> source_violations;
  PtxAssumptions semantic_assumptions = DefaultPtxAssumptions();
  std::string frontend_assumptions_json = "{}";
  std::string mma_operand_ordering_json = "{}";
  // Address-invariant environment Gamma (JSON array) and per-pair SMT budget
  // for the symbolic address model.
  std::string address_invariants_json = "[]";
  // Normalized tensor_properties records (frontend fixed_input.tensor_property_inputs).
  std::string tensor_properties_json = "[]";
  SymbolicSolverOptions address_solver;
};

// Invoked while the checked program and its certificate are still alive.
// The finalized verification report (including warnings) is available before
// the consumer starts numerical work; the string view is valid for this call.
using VerifiedProgramObserver = std::function<void(
    const SourceArtifact&, const PassCertificate&, const MemoryRaceResult&,
    const MemoryAccessRecords&, std::string_view)>;

// Called once, after successful synchronization, before memory analysis.
using MemoryInputProvider = std::function<std::vector<ConstantAddressBinding>()>;

// Throws std::invalid_argument for malformed or unsupported input.
[[nodiscard]] SourceArtifact ReadSourceArtifact(std::string_view text);
[[nodiscard]] SourceArtifact ReadSourceArtifact(std::istream& input);

// Returns a JSON validation or verification report.
// Native producers transfer ownership directly; no source JSON is materialized.
[[nodiscard]] std::string AnalyzeSourceArtifact(SourceArtifact input,
                                               bool verify = false,
                                               const MemoryRaceOptions& memory = {},
                                               const PtxAssumptions& assumptions = {},
                                               ReachabilityOptions reachability = {},
                                               std::optional<double> wellsync_timeout_seconds = std::nullopt,
                                               const VerifiedProgramObserver& on_verified = {},
                                               const MemoryInputProvider& memory_input = {});
[[nodiscard]] std::string AnalyzeSourceArtifact(std::string_view text,
                                               bool verify = false,
                                               const MemoryRaceOptions& memory = {},
                                               const PtxAssumptions& assumptions = {},
                                               ReachabilityOptions reachability = {},
                                               std::optional<double> wellsync_timeout_seconds = std::nullopt,
                                               const VerifiedProgramObserver& on_verified = {});
[[nodiscard]] std::string AnalyzeSourceArtifact(std::istream& input,
                                               bool verify = false,
                                               const MemoryRaceOptions& memory = {},
                                               const PtxAssumptions& assumptions = {},
                                               ReachabilityOptions reachability = {},
                                               std::optional<double> wellsync_timeout_seconds = std::nullopt,
                                               const VerifiedProgramObserver& on_verified = {});

}  // namespace sentryward::algorithm
