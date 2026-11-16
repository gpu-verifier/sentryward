#include "production.hpp"
#include "output_adapter_utils.hpp"

namespace rg {
using namespace output_alignment_detail;
namespace {
Edge artifactEdge(const sentryward::scalar_fold::Ref& e) {
  return Edge(e.node,e.conversion,e.converted_type,e.pointer_provenance);
}
} // namespace
Document fromArtifact(const sentryward::scalar_fold::Artifact& a) {
  Document d;d.nodes.reserve(a.graph.size());
  for(const auto& n:a.graph.nodes()) {
    RawNode r;r.kind=sentryward::scalar_fold::kindName(n.kind);r.name=n.name;r.type=n.type;r.attributes=n.attributes;r.level=n.index_level;
    r.instruction=a.graph.instructionModes(d.nodes.size());
    for(const auto& e:n.operands)r.operands.push_back(artifactEdge(e));
    if(n.fold){r.level=n.fold->binder_level;r.count=n.fold->count;r.reducer=sentryward::scalar_fold::reducerName(n.fold->reducer);}
    d.nodes.push_back(std::move(r));
  }
  d.meta={{"format","spverify.scalar-fold.v2"},{"numerical_semantics","ideal_real"},{"outputs",J::array()},{"output_families",J::array()}};
  for(const auto& f:a.output_families)d.meta["output_families"].push_back({{"allocation",f.allocation},{"offset",f.offset},{"element_bytes",f.element_bytes},{"shape",f.shape},{"strides_bytes",f.strides_bytes},{"body",edgeJson(artifactEdge(f.body))}});
  if(!a.outputs.empty())throw Unsupported("point outputs require a family adapter");
  return d;
}
std::map<std::string,sentryward::scalar_fold::ComparedOutput> comparedOutputs(const J& mapping) {
  std::map<std::string,sentryward::scalar_fold::ComparedOutput> result;
  try {
    for(const auto& m:outputTensors(mapping)) {
      sentryward::scalar_fold::ComparedOutput layout{input_unsigned(m.at("right_offset")),input_unsigned(m.at("element_bytes")),
          m.at("shape").get<std::vector<uint64_t>>(),m.at("right_strides_bytes").get<std::vector<uint64_t>>()};
      if(!result.emplace(m.at("right_allocation").get<std::string>(),std::move(layout)).second)return {};
    }
  } catch(const std::exception&) {return {};}
  return result;
}
ComparisonResult compareArtifact(Document reference,const sentryward::scalar_fold::Artifact& optimized,J mapping,const ComparisonOptions& options,const ContextBuilder& build_context) {
  try {auto right=fromArtifact(optimized);alignOutputs(reference,right,mapping);return compare(std::move(reference),std::move(right),std::move(mapping),options,{}, {},build_context);}
  catch(const Unsupported& e){return ComparisonResult::fromReport({{"status","translation_or_supported_fragment_blocked"},{"blocked_stage","object_adapter"},{"reason",e.code},{"diagnostics",J::array({{{"code",e.code},{"stage","equivalence_import"},{"message",e.what()},{"details",e.details}}})}});}
  catch(const std::bad_alloc& e){return ComparisonResult::fromReport({{"status","allocation_failed"},{"blocked_stage","object_adapter"},{"reason","memory_allocation_failed"},{"message",e.what()}});}
  catch(const std::exception& e){return ComparisonResult::fromReport({{"status","internal_error"},{"blocked_stage","object_adapter"},{"reason","unexpected_exception"},{"message",e.what()}});}
}
} // namespace rg
