#pragma once

#include "sentryward/frontend/ptx/execution.h"
#include "sentryward/value_ir/graph.h"
#include <memory>
#include <map>
#include <span>

namespace sentryward::frontend::ptx {
// Evidence for one actually observed user-supplied control decision. It is not
// a numerical theorem and cannot be applied outside reached_group. The raw
// predicate keeps Load operations; memory binding is a separate checked step.
struct BranchScopeObservation {
  value_ir::Value predicate;
  value_ir::ThreadGroupId reached_group{};
  value_ir::SourceOrigin branch;
  std::string predicate_register;
  DynamicKey register_origin;
  BranchScopeDecision decision;
  bool predicate_negated{};
  bool predicateTruth() const { return decision.taken != predicate_negated; }
};
class BranchScopeCapture {
 public:
  explicit BranchScopeCapture(value_ir::Graph& graph) : graph_(graph) {}
  // Called only by the opt-in extractor at the pre-decision register read.
  // Does not traverse operands, members, or graph ancestors. Header failure
  // fails capture; it never creates an assumption or coerces a predicate.
  void observe(BranchScopeObservation);
  void unavailable(std::string reason) { if (failure_.empty()) failure_=std::move(reason); }
  const std::string& failure() const { return failure_; }
  std::span<const BranchScopeObservation> observations() const { return observations_; }
  const value_ir::Graph& graph() const { return graph_; }
 private:
  value_ir::Graph& graph_;
  std::vector<BranchScopeObservation> observations_;
  std::string failure_;
};

} // namespace sentryward::frontend::ptx
