#pragma once

#include "sentryward/frontend/ptx/execution.h"
#include "sentryward/frontend/ptx/unit_facts.h"
#include "sentryward/value_ir/graph.h"
#include "sentryward/algorithm/memory/constant_address_model.h"
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace sentryward::frontend::ptx {
class BranchScopeCapture;

// CTA-local index of memory address operands in the completed value DAG.
class ValueAddressIndex {
public:
  // The CTA id is facts.Linear().
  ValueAddressIndex(const value_ir::Graph&, const CtaFacts&,
                    std::size_t first_node, bool defer_serialization = false);
  ~ValueAddressIndex();
  // The address bridge: `lane` is a lane of the same facts.
  std::optional<nlohmann::json> Extract(LaneRef lane, const RecordView&, std::string_view);
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
  // The defining record of a dynamic key on a lane (Resources' Definition).
  using Definition = std::function<std::optional<RecordView>(LaneRef, DynamicKey)>;
  // Names through the frame of the Resources that owns it; its value-IR CTA id is 0.
  AddressValueExtractor(const FactSource&, const UnitFrame&);
  ~AddressValueExtractor();
  std::optional<nlohmann::json> Extract(LaneRef, const RecordView&,
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

// One function's Partition rules, shared by every CTA and built on first use.
// Sparse: an entry only for an instruction that reached Partition and whose
// opcode a rule can whitelist (cp, ld, mma, ldmatrix, stmatrix, shfl, elect,
// tcgen05, vote); null where Partition returns at its whitelist or
// constant-copy exit. Static; never consumer state.
struct PartitionSpecCache {
  const Function *function = nullptr;
  std::unordered_map<std::uint32_t, std::unique_ptr<const PartitionSpec>> rules;
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
  void AddCta(const CtaFacts &facts);
  [[nodiscard]] bool complete() const { return diagnostics_.empty(); }
  [[nodiscard]] const std::vector<ValueIrDiagnostic> &diagnostics() const {
    return diagnostics_;
  }

private:
  value_ir::Graph &graph_;
  std::vector<ValueIrDiagnostic> diagnostics_;
  bool stop_on_unsupported_ = false;
  bool prune_unused_compute_ = false;
  BranchScopeCapture* branch_capture_ = nullptr;
  PartitionSpecCache partition_specs_;
};
} // namespace sentryward::frontend::ptx
