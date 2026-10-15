#pragma once

#include "sentryward/frontend/ptx/execution.h"
#include "sentryward/value_ir/graph.h"
#include "sentryward/algorithm/memory/constant_address_model.h"
#include <string>
#include <vector>

namespace sentryward::frontend::ptx {
class BranchScopeCapture;

// CTA-local index of memory address operands in the completed value DAG.
class ValueAddressIndex {
public:
  ValueAddressIndex(const value_ir::Graph&, const ExecutionProgram&, std::uint32_t cta,
                    std::size_t first_node, bool defer_serialization = false);
  ~ValueAddressIndex();
  std::optional<nlohmann::json> Extract(const LaneTrace&, const ExecutedInstruction&, std::string_view);
  std::size_t queries() const;
  std::size_t resolved() const;
  std::string firstUnavailableReason() const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Resolve compute-DAG references at the memory-analysis boundary or for export.
void MaterializeValueAddresses(std::vector<algorithm::ConstantAddressBinding>&,
                               const value_ir::Graph&, bool canonical_terms = false);

// Demand-driven scalar address slice using the numerical PTX lowering rules.
class AddressValueExtractor {
public:
  using Definition = std::function<const ExecutedInstruction*(const LaneTrace&, DynamicKey)>;
  AddressValueExtractor(const ExecutionProgram&, Coordinates);
  ~AddressValueExtractor();
  std::optional<nlohmann::json> Extract(const LaneTrace&, const ExecutedInstruction&,
                                      std::string_view, const Definition&);
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct ValueIrDiagnostic {
  std::string code;
  std::string message;
  value_ir::SourceOrigin origin;
};

struct ValueLoweringStatistics {
  std::uint64_t template_steps=0, splits=0, final_templates=0;
};

// Opt-in, expression-only export of the recorded per-thread path. This does not
// evaluate values, infer synchronization, or replace the verifier's lowering.
// The destination graph owns all expressions and copied source provenance.
class ValueIrExtractor {
public:
  explicit ValueIrExtractor(value_ir::Graph &graph, bool stop_on_unsupported = false,
                            bool prune_unused_compute = false, BranchScopeCapture* branch_capture = nullptr)
      : graph_(graph), stop_on_unsupported_(stop_on_unsupported), prune_unused_compute_(prune_unused_compute),
        branch_capture_(branch_capture) {}
  void AddCta(std::uint32_t cta, Coordinates coordinates,
              const ExecutionProgram &program);
  [[nodiscard]] bool complete() const { return diagnostics_.empty(); }
  [[nodiscard]] const ValueLoweringStatistics& statistics() const { return statistics_; }
  [[nodiscard]] const std::vector<ValueIrDiagnostic> &diagnostics() const {
    return diagnostics_;
  }

private:
  value_ir::Graph &graph_;
  std::vector<ValueIrDiagnostic> diagnostics_;
  bool stop_on_unsupported_ = false;
  bool prune_unused_compute_ = false;
  BranchScopeCapture* branch_capture_ = nullptr;
  ValueLoweringStatistics statistics_;
};
} // namespace sentryward::frontend::ptx
