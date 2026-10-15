#pragma once
#include "equivalence_context.hpp"
namespace rg {
// Explicit caller predicates, in topological DAG order. No kernel recognition,
// implicit branch-to-predicate inference, or recursive expression parsing.
inline void appendCallerPredicates(Arena& a,const J& mapping,EquivalenceContext& context) {
  const auto config=mapping.value("checker_context",J::object());
  for(auto it=config.begin();it!=config.end();++it)
    if(it.key()!="input_contract" && it.key()!="predicates")throw Unsupported("unknown checker context field: "+it.key());
  const auto predicates=config.value("predicates",J::array());
  if(!predicates.is_array())throw Unsupported("caller predicates must be an array");
  for(const auto& predicate:predicates) {
    const auto reason=predicate.at("source").get<std::string>();
    if(reason.empty())throw Unsupported("caller predicate requires provenance");
    std::vector<Id> binders,nodes;
    for(const auto& b:predicate.at("binders")) {
      Q lo=jq(b.at("min_inclusive")),hi=jq(b.at("max_inclusive"));
      if(lo.denominator()!=1 || hi.denominator()!=1 || lo>hi)throw Unsupported("invalid caller binder domain");
      binders.push_back(a.var("caller-predicate-"+std::to_string(context.assumptions.size())+"-"+std::to_string(binders.size()),lo,hi));
    }
    for(const auto& n:predicate.at("nodes")) {
      const auto op=n.at("op").get<std::string>();
      auto operand=[&](const J& raw)->Id {
        if(!raw.is_number_unsigned() && (!raw.is_number_integer() || raw.get<int64_t>()<0))throw Unsupported("invalid caller operand");
        const auto i=raw.get<size_t>();if(i>=nodes.size())throw Unsupported("caller predicate is not a topological DAG");return nodes[i];
      };
      if(op=="constant")nodes.push_back(a.constant(jq(n.at("value"))));
      else if(op=="index") {
        const auto i=n.at("binder").get<size_t>();if(i>=binders.size())throw Unsupported("unknown caller binder");
        nodes.push_back(binders[i]);
      } else if(op=="read") {
        const auto object=n.at("object").get<std::string>(),type=n.at("element_type").get<std::string>();
        unsigned bytes=type=="f32"?4:type=="f64"?8:type=="f16"||type=="bf16"?2:0;
        if(!bytes)throw Unsupported("caller predicate read requires supported real type");
        std::vector<Id> coords;for(const auto& c:n.at("coordinates"))coords.push_back(operand(c));
        J canonical;
        for(const auto* side:{"left_inputs","right_inputs"}) {
          const auto& inputs=mapping.at(side);const J* layout=nullptr;
          for(auto it=inputs.at("allocations").begin();it!=inputs.at("allocations").end();++it)if(it.value()==object) {
            if(layout)throw Unsupported("ambiguous caller predicate input");
            layout=&inputs.at("layouts").at(it.key());
          }
          if(!layout || layout->at("element_bytes")!=bytes || layout->at("shape").size()!=coords.size())throw Unsupported("caller predicate input layout mismatch");
          J target={{"shape",layout->at("shape")},{"strides",layout->value("target_strides_bytes",layout->at("source_strides_bytes"))},{"offset",layout->value("target_offset",0)}};
          if(!canonical.is_null() && canonical!=target)throw Unsupported("caller predicate canonical layouts differ");
          canonical=target;
          for(size_t i=0;i<coords.size();++i) {
            const auto& b=a.nodes.at(coords[i]).bounds;
            if(!b || b->first<0 || b->second>=jq(layout->at("shape").at(i)))throw Unsupported("caller predicate read outside declared domain");
          }
        }
        nodes.push_back(a.read(object,std::move(coords),"real:"+type));
      } else {
        std::vector<Id> args;for(const auto& arg:n.at("args"))args.push_back(operand(arg));
        if(op=="add" && args.size()==2)nodes.push_back(a.add(args));
        else if(op=="subtract" && args.size()==2)nodes.push_back(a.add({args[0],a.neg(args[1])}));
        else if(op=="multiply" && args.size()==2)nodes.push_back(a.mul(args));
        else if(op=="compare" && args.size()==2)nodes.push_back(a.compare(n.at("predicate"),args[0],args[1]));
        else throw Unsupported("unsupported caller predicate operator: "+op);
      }
    }
    const auto root=predicate.at("root").get<size_t>();if(root>=nodes.size())throw Unsupported("invalid caller predicate root");
    context.assumptions.emplace_back(ScopedPredicateAssumption{nodes[root],std::move(binders),reason});
  }
}
}
