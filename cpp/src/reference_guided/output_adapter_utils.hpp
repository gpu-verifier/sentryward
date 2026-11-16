#pragma once
#include "input.hpp"

namespace rg::output_alignment_detail {
inline J edgeJson(const Edge& e) {
  J j={{"node",e.node}};
  if(!e.conversion.empty())j["conversion"]=e.conversion;
  if(!e.converted_type.empty())j["converted_type"]=e.converted_type;
  if(!e.pointer_provenance.empty())j["pointer_provenance"]=e.pointer_provenance;
  return j;
}
inline J outputTensors(const J& mapping) {
  const auto& outputs=mapping.at("outputs");
  return outputs.value("kind","")=="dense_tensor"?J::array({outputs}):outputs.at("tensors");
}
} // namespace rg::output_alignment_detail
