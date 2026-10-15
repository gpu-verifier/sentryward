#pragma once
#include "equivalence_context.hpp"
namespace rg {
// Once per comparison, after premise validation. Traverses only the supplied
// predicate DAG, not the kernel expressions or individual output families.
inline J contextDeclaration(const Arena& arena,const EquivalenceContext& context) {
  J report={{"inputs",J::array()},{"assumptions",J::array()},
            {"fixed_inputs",J::array()},{"predicate_nodes",J::array()}};
  std::vector<Id> pending;
  for(const auto& input:context.inputs)report["inputs"].push_back(
      {{"reference",input.reference},{"optimized",input.optimized},{"shared",input.shared}});
  for(const auto& assumption:context.assumptions)std::visit([&](const auto& p){
    using T=std::decay_t<decltype(p)>;
    J declaration;
    if constexpr(std::is_same_v<T,InputElementAssumption>) {
      declaration={{"kind","input_elements"},{"input",p.input},
                   {"interpretation",p.interpretation},{"reason",p.reason}};
      if(const auto* interval=std::get_if<Interval>(&p.property)) {
        declaration["interval"]={{"lower",interval->lower?J(str(*interval->lower)):J(nullptr)},
          {"upper",interval->upper?J(str(*interval->upper)):J(nullptr)},
          {"lower_inclusive",interval->lower_inclusive},{"upper_inclusive",interval->upper_inclusive}};
      } else declaration["finite"]=true;
    } else {
      declaration={{"kind","predicate"},{"predicate",p.predicate}};
      pending.push_back(p.predicate);
      if constexpr(std::is_same_v<T,ScopedPredicateAssumption>) {
        declaration["bound_indices"]=p.bound_indices;declaration["reason"]=p.reason;
        pending.insert(pending.end(),p.bound_indices.begin(),p.bound_indices.end());
      } else declaration["reason"]="explicit native caller premise";
    }
    report["assumptions"].push_back(std::move(declaration));
  },assumption);
  for(const auto& p:context.fixed_inputs) {
    J coordinates=J::array();for(const auto& c:p.coordinates)coordinates.push_back(c.template convert_to<std::string>());
    report["fixed_inputs"].push_back({{"input",p.input},{"interpretation",p.interpretation},
      {"coordinates",std::move(coordinates)},{"value",str(p.value)},{"reason",p.reason}});
  }
  std::set<Id> seen;
  while(!pending.empty()) {
    auto id=pending.back();pending.pop_back();if(!seen.insert(id).second)continue;
    const auto& n=arena.nodes.at(id);
    J data=J::array();for(const auto& value:n.data)data.push_back(str(value));
    report["predicate_nodes"].push_back({{"id",id},{"op",n.op},{"args",n.args},{"data",data},{"tag",n.tag}});
    pending.insert(pending.end(),n.args.begin(),n.args.end());
  }
  return report;
}
}
