#include "sentryward/frontend/ptx/branch_scope_capture.h"
namespace sentryward::frontend::ptx {
namespace {
namespace v=value_ir;
constexpr std::size_t maxFacts=4096;
void validateHeader(const v::Graph& graph, const BranchScopeObservation& fact) {
  const auto predicate = fact.predicate;
  if (!graph.contains(predicate) || !predicate.indices().empty() || predicate.threadSelector() ||
      predicate.pointer() || predicate.conversion() || !graph.type(predicate).isScalar() ||
      graph.type(predicate).element().kind() != v::ElementKind::Predicate)
    throw std::invalid_argument("branch-scope root must be an exact, bare scalar predicate");
  const auto& o = fact.branch;
  if (!o.cta || !o.thread || !o.instruction_index || !o.occurrence || !o.source_line ||
      !o.source_text || !o.function || !o.source_file || !o.scope || !o.predicate ||
      o.executed != true || o.unroll_context.size() > 64 ||
      fact.predicate_register.empty() || fact.predicate_register.size() > 128)
    throw std::invalid_argument("incomplete branch-scope provenance");
  for (const auto* field : {&o.source_text, &o.function, &o.source_file, &o.scope, &o.predicate})
    if ((*field)->view().size() > 8192) throw std::length_error("branch-scope provenance limit");
  const auto& domain = graph.group(fact.reached_group);
  if (domain.cta != *o.cta || domain.threads.empty() || !domain.threads.indexOf(*o.thread))
    throw std::invalid_argument("branch-scope reached domain mismatch");
  const auto source = graph.node(predicate.nodeId()).instanceOrigin(predicate.instanceIndex());
  if (source.instruction_index != fact.register_origin.instruction ||
      source.occurrence != fact.register_origin.occurrence || source.cta != o.cta || source.thread != o.thread)
    throw std::invalid_argument("branch-scope register producer mismatch");
}
}
void BranchScopeCapture::observe(BranchScopeObservation fact) {
  if (observations_.size() == maxFacts) throw std::length_error("branch-scope observation limit");
  validateHeader(graph_, fact);
  observations_.push_back(std::move(fact));
}
}
