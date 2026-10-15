#pragma once
#include "equivalence_context.hpp"
#include "input.hpp"

namespace rg {
// A native caller may build Arena-scoped premises before either expression
// is imported. This is not a
// new user-facing configuration format. The receipt is included in the report.
using ContextBuilder = std::function<void(Arena &, const J &, EquivalenceContext &, J &)>;
enum class ComparisonStatus { Equivalent, PartialScopeEquivalent, NotProven, Blocked,
                              Timeout, ResourceLimit, ImplementationError };
enum class TotalDefinedness { Proved, NotProven, Unreported };
struct ComparisonOptions {
  double seconds=100; std::size_t family_limit=0;
  bool ep_profile=false, ep_reuse_cache=false;
};
struct ComparisonResult {
  ComparisonStatus status;
  J report;
  // Equivalence of partial functions does not imply total definedness.
  TotalDefinedness totalDefinedness() const {
    if(!report.contains("definedness"))return TotalDefinedness::Unreported;
    return report.at("definedness").value("status", "not_proven")=="proved"
      ? TotalDefinedness::Proved : TotalDefinedness::NotProven;
  }
  static ComparisonResult fromReport(J report) {
    const auto s=report.value("status", "internal_error");
    auto status=s=="equivalent"?ComparisonStatus::Equivalent:
      s=="partial_scope_equivalent"?ComparisonStatus::PartialScopeEquivalent:
      s=="not_proven"?ComparisonStatus::NotProven:
      s=="translation_or_supported_fragment_blocked"?ComparisonStatus::Blocked:
      s=="timeout"?ComparisonStatus::Timeout:
      s=="allocation_failed"?ComparisonStatus::ResourceLimit:ComparisonStatus::ImplementationError;
    return {status,std::move(report)};
  }
};
// Owned documents move into importers. No file I/O, stdout or process control.
ComparisonResult compare(Document left, Document right, J mapping,
    const ComparisonOptions& options={}, EquivalenceContext context={},
    InputContract contract={}, const ContextBuilder& build_context={});
}
