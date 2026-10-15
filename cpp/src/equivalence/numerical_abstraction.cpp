#include "sentryward/equivalence/numerical_abstraction.h"
#include "sentryward/scalar_fold/reduction_builder.h"
#include "sentryward/equivalence/conversion_uses.h"
#include "sentryward/equivalence/numeric_semantics.h"
#include "sentryward/equivalence/structural_order.h"
#include "sentryward/algorithm/diagnostics/compression_audit.h"
#include <algorithm>
#include <map>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace sentryward::equivalence {
namespace {
namespace s=scalar_fold;
bool floating(const std::string& t) {
  return t=="real" || t=="f16" || t=="bf16" || t=="f32" || t=="f64";
}
bool attributes(const std::string& text) {
  std::istringstream in(text);std::string word;
  while(in>>word)if(word!="ideal_arithmetic" && word!="flush_to_zero" &&
      word!="rounding=nearest-even" && word!="rounding=toward-zero" &&
      word!="rounding=toward-positive" && word!="rounding=toward-negative")return false;
  return true;
}
bool legacyNumericPrimitive(const s::Graph& graph,const s::Node& n) {
  return n.kind==s::Kind::Operation && n.operands.size()==1 &&
      legacyIdealPrimitive(n.name,n.type,n.attributes,graph.type(n.operands[0])).has_value();
}
bool numericOperation(const s::Graph& graph,const s::Node& n) {
  if(legacyNumericPrimitive(graph,n))return true;
  if(n.kind!=s::Kind::Operation || !floating(n.type) ||
      (!attributes(n.attributes) && !(n.name=="Exp2" && n.attributes=="semantics=ideal")))return false;
  return idealScalarOperation(n.name,n.operands.size());
}
bool integer(const std::string& type) {
  return type=="u8" || type=="u16" || type=="u32" || type=="u64" ||
      type=="s8" || type=="s16" || type=="s32" || type=="s64";
}
bool comparison(const s::Node& n) {
  if(n.kind!=s::Kind::Operation || n.type!="pred" || n.operands.size()!=2)return false;
  if(n.name!="Compare")return n.attributes.empty() &&
      (n.name=="Less" || n.name=="LessEqual" || n.name=="Greater" ||
       n.name=="GreaterEqual" || n.name=="Equal" || n.name=="NotEqual");
  std::istringstream in(n.attributes);std::string word,predicate;bool unordered=false,seen=false;
  while(in>>word) {
    if(word.starts_with("predicate=") && !seen){predicate=word.substr(10);seen=true;}
    else if(word=="unordered_true" && !unordered)unordered=true;
    else return false;
  }
  return predicate=="eq" || predicate=="ne" || predicate=="lt" ||
      predicate=="le" || predicate=="gt" || predicate=="ge";
}
bool select(const s::Node& n) {
  return n.kind==s::Kind::Operation && n.name=="Select" && floating(n.type) &&
      n.operands.size()==3 && (n.attributes.empty() || n.attributes=="ideal_arithmetic");
}
struct Transform {
  const s::Graph& input;s::Graph output;
  std::size_t limit,work=0;std::chrono::steady_clock::time_point deadline;
  std::map<std::tuple<s::Id,unsigned,bool>,s::Ref> cache;
  std::vector<unsigned char> uses;
  StructuralOrder order{output,[this]{check();}};
  std::unordered_map<s::Ref,s::Ref,s::RefHash> address_envelopes{};
  void check() {
    if(++work>limit*32 || output.size()>limit) {
      SENTRYWARD_AUDIT(audit->event("NORMALIZATION_LIMIT",output.size()>limit?"created_nodes":"work",
          work,output.size(),order.visitedSegments()));
      throw std::invalid_argument("numerical_abstraction_node_budget");
    }
    if((work&255)==0 && std::chrono::steady_clock::now()>=deadline)
      throw std::invalid_argument("numerical_abstraction_time_budget");
  }
  bool sum(s::Ref r) const {
    const auto& n=input.node(r);
    return r.conversion.empty() && r.pointer_provenance.empty() &&
      n.kind==s::Kind::Operation && floating(n.type) && attributes(n.attributes) &&
      ((n.name=="Add" && n.operands.size()==2) || (n.name=="Fma" && n.operands.size()==3));
  }
  s::Ref addressSum(const std::vector<s::Ref>& args) {
    std::vector<s::Ref> pending=args,terms;std::uint64_t constant=0;
    while(!pending.empty()) {
      check();auto a=pending.back();pending.pop_back();const auto n=output.node(a);
      if(a.conversion.empty() && a.pointer_provenance.empty() && n.type=="u64") {
        if(n.kind==s::Kind::Constant) {constant+=std::stoull(n.name);continue;}
        if(n.kind==s::Kind::Operation && n.name=="Add" && n.attributes.empty()) {
          pending.insert(pending.end(),n.operands.begin(),n.operands.end());continue;
        }
      }
      terms.push_back(a);
    }
    std::stable_sort(terms.begin(),terms.end(),[&](const auto& a,const auto& b){return order.less(a,b);});
    auto value=output.constant("u64",std::to_string(constant));
    for(auto term:terms)value=output.operation("Add","u64",{value,term});
    return value;
  }
  s::Ref numericReadAddress(s::Ref address) {
    // Existing Add-address normalization already has an explicit constant
    // offset. Give a plain zero-offset Index/Multiply root the SAME shape so
    // ReductionBuilder can recognize stored affine taps using its existing
    // complete matcher. This is exact u64 Add(0,x), not an affine inference,
    // ancestor walk, read identity change or Fold-element generation.
    if(!address.conversion.empty() || !address.converted_type.empty() ||
        !address.pointer_provenance.empty() || output.type(address)!="u64")return address;
    const auto& n=output.node(address);
    if(!n.attributes.empty() || n.fold)return address;
    bool eligible=n.kind==s::Kind::Index && n.operands.empty();
    if(n.kind==s::Kind::Operation && n.name=="Multiply" && n.operands.size()==2) {
      eligible=true;
      for(const auto& operand:n.operands)
        eligible &= operand.conversion.empty() && operand.converted_type.empty() &&
            operand.pointer_provenance.empty() && output.type(operand)=="u64";
    }
    if(!eligible)return address;
    check();
    if(auto found=address_envelopes.find(address);found!=address_envelopes.end())return found->second;
    auto zero=output.constant("u64","0");check();
    auto result=output.operation("Add","u64",{zero,address});check();
    // Every key/value is one plain existing/new output Ref. No expanded
    // address metadata is retained; the normal construction cap also bounds
    // this memo, independently of logical output/Fold extents.
    if(address_envelopes.size()>=limit)throw std::invalid_argument("numerical_abstraction_node_budget");
    address_envelopes.emplace(address,result);return result;
  }
  s::Ref visit(s::Ref r,unsigned level,bool arithmetic=true,bool output_root=false) {
    check();
    if(!r.pointer_provenance.empty() || (!output_root && !r.conversion.empty() &&
        !idealNumericConversion(input.node(r).type,input.type(r),r.conversion) &&
        !idealFloatingStorageReadUse(input,r)))arithmetic=false;
    auto ck=std::tuple{r.node,level,arithmetic};s::Ref result;
    if(auto it=cache.find(ck);it!=cache.end())result=it->second;
    else {
      const auto n=input.node(r);
      if(arithmetic && sum(input.ref(r.node))) {
        std::vector<s::Ref> stack{input.ref(r.node)},terms;
        while(!stack.empty()) {
          check();auto a=stack.back();stack.pop_back();
          // Flatten exclusive chains, but retain shared sums as memoized DAG
          // atoms. Otherwise Add(x,x) recursively expands x twice at every level.
          if(sum(a) && input.node(a).type==n.type && (a.node==r.node || uses[a.node]<2)) {
            const auto term=input.node(a);
            if(term.name=="Add") {stack.push_back(term.operands[1]);stack.push_back(term.operands[0]);}
            else {
              auto x=visit(term.operands[0],level),y=visit(term.operands[1],level);
              if(order.less(y,x))std::swap(x,y);
              terms.push_back(output.operation("Multiply",n.type,{x,y}));
              stack.push_back(term.operands[2]);
            }
          } else terms.push_back(visit(a,level));
        }
        std::stable_sort(terms.begin(),terms.end(),[&](const auto& a,const auto& b){return order.less(a,b);});
        s::ReductionBuilder reductions(output);result=output.constant(n.type,"0");
        for(auto term:terms) {
          const auto& v=output.node(term);
          if(v.kind==s::Kind::Constant && v.name=="0" && term.conversion.empty())continue;
          // Existing reductions are terms too. ReductionBuilder checks their
          // complete structure and lifts inner binders when introducing the
          // outer affine reduction; it never enumerates their iterations.
          result=reductions.append(s::Reducer::Sum,result,term,n.type,level);check();
        }
      } else {
        std::vector<s::Ref> args;
        for(std::size_t i=0;i<n.operands.size();++i)
          args.push_back(visit(n.operands[i],n.fold && i==1?n.fold->binder_level+1:level,
                              arithmetic && floating(n.type) && !(select(n) && i==0)));
        switch(n.kind) {
          case s::Kind::Constant:result=output.constant(n.type,n.name);break;
          case s::Kind::Parameter:result=output.parameter(n.type,n.name);break;
          case s::Kind::Index:result=output.index(n.index_level);break;
          case s::Kind::InitialCell:
            if(arithmetic && floating(n.type) && n.attributes.empty() && args.size()==1 &&
                n.operands[0].conversion.empty() && n.operands[0].converted_type.empty() &&
                n.operands[0].pointer_provenance.empty())args[0]=numericReadAddress(args[0]);
            result=output.initialCell(n.name,args.at(0),n.type);break;
          case s::Kind::Operation:
            result=arithmetic && n.name=="Add" && n.type=="u64" && n.attributes.empty()?
                addressSum(args):output.operation(n.name,n.type,args,n.attributes);break;
          case s::Kind::Fold:result=output.fold(n.fold->reducer,args.at(0),args.at(1),n.fold->binder_level,n.fold->count,n.type);break;
        }
      }
      cache.emplace(ck,result);
    }
    if(!r.conversion.empty()) {
      if(!result.conversion.empty())result.conversion+="; ";
      result.conversion+=r.conversion;result.converted_type=r.converted_type;
    }
    result.pointer_provenance=r.pointer_provenance;return result;
  }
};
}
namespace {
struct RootEligibility {
  // Outputs precede output_families, matching normalize's root traversal.
  std::vector<bool> numeric;
  std::size_t numeric_count=0;
};
RootEligibility eligibleRoots(const scalar_fold::Artifact& input,
                              std::chrono::steady_clock::time_point deadline) {
  // Predicate values are control certificates, not numeric operands. Their
  // descendants are copied in exact context by Transform (including arithmetic
  // shared with a Select arm), preserving rounding and comparison attributes.
  std::vector<unsigned char> numeric(input.graph.size()),predicate(input.graph.size());
  const auto numericUse=[&](const s::Ref& operand) {
    const auto& source=input.graph.node(operand);
    const auto& type=input.graph.type(operand);
    // An explicitly converted integer expression is a numeric ATOM, not
    // permission to reassociate its integer ancestry. Transform::visit keeps
    // this complete use in exact context. This includes packed signed-element
    // decoding under floating sums, without assuming a value for that decode.
    if(integer(source.type) && source.type!="pred" && operand.pointer_provenance.empty() &&
        operand.conversion.size()<=128) {
      const auto step=leadingConversion(source.type,operand.conversion);
      if(step && finiteIntegerFloatConversion(source.type,step->target,
          operand.conversion.substr(0,operand.conversion.find("; "))) &&
          idealNumericConversion(step->target,type,step->remainder))return true;
    }
    return numeric.at(operand.node) && operand.pointer_provenance.empty() &&
        (floating(type) || type=="e4m3" || type=="e5m2") &&
        (operand.conversion.empty() || idealNumericConversion(source.type,type,operand.conversion) ||
         idealFloatingStorageReadUse(input.graph,operand));
  };
  const auto comparisonUse=[&](const s::Ref& operand) {
    // Comparands are exact control inputs. In particular an integer predicate
    // may have arbitrary integer arithmetic ancestry, which is not reassociated.
    const auto& type=input.graph.type(operand);
    if(!operand.pointer_provenance.empty())return false;
    if(integer(type))return operand.conversion.empty();
    return (floating(type) || type=="e4m3" || type=="e5m2") &&
        (operand.conversion.empty() || idealNumericConversion(input.graph.node(operand).type,type,operand.conversion) ||
         idealFloatingStorageReadUse(input.graph,operand));
  };
  for(scalar_fold::Id i=0;i<input.graph.size();++i) {
    if((i&255)==0 && std::chrono::steady_clock::now()>=deadline)
      throw std::invalid_argument("numerical_abstraction_time_budget");
    const auto& n=input.graph.node(i);
    bool safe=n.kind==scalar_fold::Kind::InitialCell || n.kind==scalar_fold::Kind::Constant ||
        n.kind==scalar_fold::Kind::Parameter;
    if(safe && n.type=="pred" && n.attributes.empty())predicate[i]=true;
    if(comparison(n)) {
      predicate[i]=comparisonUse(n.operands[0]) && comparisonUse(n.operands[1]) &&
          input.graph.type(n.operands[0])==input.graph.type(n.operands[1]);
      safe=false;
    } else if(select(n)) {
      const auto& condition=n.operands[0];
      safe=predicate.at(condition.node) && input.graph.type(condition)=="pred" &&
          condition.conversion.empty() && condition.pointer_provenance.empty();
      for(std::size_t j=1;j<3;++j)
        safe &= numericUse(n.operands[j]) && input.graph.type(n.operands[j])==n.type;
    } else if(n.kind==scalar_fold::Kind::Operation || n.kind==scalar_fold::Kind::Fold) {
      // Do not reassociate arithmetic through an opaque floating consumer or
      // a malformed control node; these remain on the typed/exact path.
      safe=n.kind==scalar_fold::Kind::Fold?floating(n.type):numericOperation(input.graph,n);
      for(const auto& operand:n.operands)safe &= numericUse(operand);
    }
    numeric[i]=safe;
  }
  RootEligibility roots;
  roots.numeric.reserve(input.outputs.size()+input.output_families.size());
  const auto append=[&](const s::Ref& ref) {
    const auto& source=input.graph.node(ref);
    // This pass is called under the explicit ideal-float output contract.
    // Reuse its approved OUTPUT conversion/storage-tail policy. Internal
    // saturation, integer, predicate and bit uses still use numericUse()'s
    // exact conversion eligibility, not this output-boundary exception.
    const bool value=numeric.at(ref.node) && ref.pointer_provenance.empty() &&
        (ref.conversion.empty()?
          (floating(source.type) || exactFloatingWidening(source.type,source.type)):
          (idealNumericOutputConversion(source.type,input.graph.type(ref),ref.conversion) ||
           idealFloatingStorageReadUse(input.graph,ref,true)));
    roots.numeric.push_back(value);roots.numeric_count+=value;
  };
  for(const auto& out:input.outputs)append(out.value);
  for(const auto& family:input.output_families)append(family.body);
  return roots;
}
scalar_fold::Artifact normalize(const scalar_fold::Artifact& input,std::size_t max_nodes,
    std::chrono::steady_clock::time_point deadline,const RootEligibility& eligibility) {
  Transform transform{input.graph,{},max_nodes,0,deadline,{},
      std::vector<unsigned char>(input.graph.size(),0)};
  const auto used=[&](s::Ref ref){auto& count=transform.uses.at(ref.node);if(count<2)++count;};
  for(const auto& node:input.graph.nodes())for(const auto& operand:node.operands)used(operand);
  for(const auto& out:input.outputs)used(out.value);
  for(const auto& family:input.output_families)used(family.body);
  SENTRYWARD_AUDIT(audit->event("NORMALIZATION_INPUT",eligibility.numeric_count?"ideal_arithmetic":"typed_passthrough",
      input.graph.size(),input.output_families.size(),max_nodes));
  SENTRYWARD_AUDIT(audit->event("NORMALIZATION_ROOT_CONTEXTS","numeric_vs_exact",
      eligibility.numeric_count,eligibility.numeric.size()-eligibility.numeric_count));
  scalar_fold::Artifact result;result.outputs=input.outputs;result.output_families=input.output_families;
  // Unsupported siblings are exact contexts, not a global veto. The cache
  // key includes node, lexical level AND numeric/exact context, so a shared
  // arithmetic DAG consumed both numerically and by bits/opaque math retains
  // independent views. Each context is memoized; no logical Fold expansion.
  std::size_t cursor=0;
  for(auto& out:result.outputs)out.value=transform.visit(out.value,0,eligibility.numeric.at(cursor++),true);
  for(auto& family:result.output_families)
    family.body=transform.visit(family.body,family.shape.size(),eligibility.numeric.at(cursor++),true);
  std::vector<scalar_fold::Ref> roots;
  for(auto& out:result.outputs)roots.push_back(out.value);
  for(auto& family:result.output_families)roots.push_back(family.body);
  transform.output.collect(roots);std::size_t i=0;
  for(auto& out:result.outputs)out.value=roots[i++];
  for(auto& family:result.output_families)family.body=roots[i++];
  result.graph=std::move(transform.output);return result;
}
} // namespace
scalar_fold::Artifact abstractNumericalReductions(const scalar_fold::Artifact& input,
    std::size_t max_nodes,std::chrono::steady_clock::time_point deadline) {
  SENTRYWARD_AUDIT_STAGE("equivalence.numerical_abstraction");
  return normalize(input,max_nodes,deadline,eligibleRoots(input,deadline));
}
scalar_fold::Artifact abstractNumericalReductions(scalar_fold::Artifact&& input,
    std::size_t max_nodes,std::chrono::steady_clock::time_point deadline) {
  SENTRYWARD_AUDIT_STAGE("equivalence.numerical_abstraction");
  const auto eligibility=eligibleRoots(input,deadline);
  if(!eligibility.numeric_count) {
    SENTRYWARD_AUDIT(audit->event("NORMALIZATION_IDENTITY","typed_artifact_retained",input.graph.size(),0));
    return std::move(input);
  }
  return normalize(input,max_nodes,deadline,eligibility);
}
}
