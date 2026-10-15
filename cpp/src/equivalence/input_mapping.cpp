#include "sentryward/algorithm/diagnostics/development_trace.h"
#include "sentryward/equivalence/input_mapping.h"
#include "sentryward/equivalence/float_from_chars.h"
#include "sentryward/equivalence/comparison_context.h"
#include "sentryward/equivalence/constant_values.h"
#include "sentryward/equivalence/egraph.h"
#include "sentryward/equivalence/conversion_uses.h"
#include "sentryward/equivalence/representation_rewriting.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>

namespace sentryward::equivalence {
namespace {
void validate(const std::map<std::string,std::string>& mapping) {
  std::set<std::string> targets;
  for(const auto& [source,target]:mapping)
    if(source.empty() || target.empty() || !targets.insert(target).second)
      throw std::invalid_argument("input_mapping_not_injective");
}
void validateConstant(const InputParameterConstant& value) {
  const auto& type=value.type;const auto& text=value.literal;
  const auto invalid=[] { throw std::invalid_argument("invalid_input_parameter_constant"); };
  if(type.empty() || text.empty())invalid();
  if(type=="f16" || type=="bf16" || type=="f32" || type=="f64" || type=="real") {
    double parsed=0;auto result=floatFromChars(text.data(),text.data()+text.size(),parsed);
    if(result.ec!=std::errc{} || result.ptr!=text.data()+text.size() || !std::isfinite(parsed))invalid();
    return;
  }
  unsigned width=0;auto bits=std::from_chars(type.data()+1,type.data()+type.size(),width);
  if(bits.ec!=std::errc{} || bits.ptr!=type.data()+type.size() ||
     (width!=8 && width!=16 && width!=32 && width!=64))invalid();
  if(type.front()=='s') {
    std::int64_t parsed=0;auto result=std::from_chars(text.data(),text.data()+text.size(),parsed);
    if(result.ec!=std::errc{} || result.ptr!=text.data()+text.size())invalid();
    if(width<64 && (parsed<-(std::int64_t(1)<<(width-1)) || parsed>=(std::int64_t(1)<<(width-1))))invalid();
  } else if(type.front()=='u' || type.front()=='b') {
    std::uint64_t parsed=0;auto result=std::from_chars(text.data(),text.data()+text.size(),parsed);
    if(result.ec!=std::errc{} || result.ptr!=text.data()+text.size() || (width<64 && parsed>=(std::uint64_t(1)<<width)))invalid();
  } else invalid();
}
const std::string& mapped(const std::map<std::string,std::string>& mapping,
                          const std::string& name,const char* reason) {
  auto it=mapping.find(name);
  if(it==mapping.end())throw std::invalid_argument(std::string(reason)+": "+name);
  return it->second;
}
std::uint64_t checkedAdd(std::uint64_t a,std::uint64_t b) {
  if(b>UINT64_MAX-a)throw std::invalid_argument("input_layout_overflow");return a+b;
}
std::uint64_t checkedMultiply(std::uint64_t a,std::uint64_t b) {
  if(b && a>UINT64_MAX/b)throw std::invalid_argument("input_layout_overflow");return a*b;
}
std::optional<std::uint64_t> unsignedLiteral(const std::string& text) {
  std::uint64_t value=0;auto result=std::from_chars(text.data(),text.data()+text.size(),value);
  if(result.ec!=std::errc{} || result.ptr!=text.data()+text.size())return {};return value;
}
std::uint64_t layoutExtent(const InputTensorLayout& layout) {
  if(layout.shape.empty() || !layout.element_bytes)throw std::invalid_argument("invalid_input_layout");
  const auto validate=[&](const auto& strides) {
    if(strides.size()!=layout.shape.size())throw std::invalid_argument("input_layout_rank_mismatch");
    std::vector<std::pair<std::uint64_t,std::uint64_t>> axes;
    for(std::size_t i=0;i<strides.size();++i) {
      if(!layout.shape[i])throw std::invalid_argument("empty_input_layout_axis");
      if(layout.shape[i]>1)axes.emplace_back(strides[i],layout.shape[i]);
    }
    std::sort(axes.begin(),axes.end());auto extent=layout.element_bytes;
    for(auto [stride,count]:axes) {
      if(stride!=extent)throw std::invalid_argument("input_layout_not_dense_permutation");
      extent=checkedMultiply(extent,count);
    }
    return extent;
  };
  const auto extent=validate(layout.source_strides_bytes);
  if(extent!=validate(layout.target_strides_bytes))throw std::invalid_argument("input_layout_size_mismatch");
  checkedAdd(layout.source_offset,extent);checkedAdd(layout.target_offset,extent);return extent;
}
struct AffineOffset {
  std::uint64_t base=0;
  std::map<std::uint32_t,std::uint64_t> terms;
};
using Bounds=std::pair<std::uint64_t,std::uint64_t>;
std::optional<Bounds> offsetBounds(const scalar_fold::Graph& graph,const scalar_fold::Node& n,
    const std::vector<std::optional<Bounds>>& previous,
    const std::map<std::uint32_t,std::uint64_t>& domains) {
  using K=scalar_fold::Kind;
  if(n.type!="u64")return {};
  if(n.kind==K::Constant) {
    if(auto value=unsignedLiteral(n.name))return Bounds{*value,*value};return {};
  }
  if(n.kind==K::Index) {
    const auto found=domains.find(n.index_level);
    if(found!=domains.end() && found->second)return Bounds{0,found->second-1};return {};
  }
  if(n.kind!=K::Operation || !n.attributes.empty() || n.operands.size()!=2)return {};
  // Modular b + (-c)*q equals b - c*q. Bound the complete subtraction,
  // not the wrapped intermediate product. Only inspect a fixed local pattern
  // and reuse the bottom-up range of q; never enumerate index members.
  if(n.name=="Add")for(unsigned side=0;side<2;++side) {
    const auto base=n.operands[side],product=n.operands[1-side];
    if(!base.conversion.empty() || !base.pointer_provenance.empty() ||
       !product.conversion.empty() || !product.pointer_provenance.empty() ||
       !previous.at(base.node))continue;
    const auto& p=graph.node(product);
    if(p.kind!=K::Operation || p.type!="u64" || p.name!="Multiply" ||
        !p.attributes.empty() || p.operands.size()!=2)continue;
    for(unsigned factor=0;factor<2;++factor) {
      const auto c=p.operands[factor],q=p.operands[1-factor];
      if(!c.conversion.empty() || !c.pointer_provenance.empty() ||
         !q.conversion.empty() || !q.pointer_provenance.empty() ||
         graph.type(q)!="u64" || !previous.at(q.node))continue;
      const auto& constant=graph.node(c);
      if(constant.kind!=K::Constant || constant.type!="u64" ||
          constant.name.empty() || constant.name.front()!='-')continue;
      const auto magnitude=unsignedLiteral(constant.name.substr(1));
      if(!magnitude)continue;
      const auto [lo,hi]=*previous[base.node];
      const auto [a,b]=*previous[q.node];
      const auto min_delta=checkedMultiply(*magnitude,a),max_delta=checkedMultiply(*magnitude,b);
      if(lo>=max_delta)return Bounds{lo-max_delta,hi-min_delta};
    }
  }
  for(const auto& ref:n.operands)
    if(!ref.conversion.empty() || !ref.pointer_provenance.empty() || !previous.at(ref.node))return {};
  auto [lo,hi]=*previous[n.operands[0].node];auto [a,b]=*previous[n.operands[1].node];
  if(n.name=="Add")return Bounds{checkedAdd(lo,a),checkedAdd(hi,b)};
  if(n.name=="Multiply")return Bounds{checkedMultiply(lo,a),checkedMultiply(hi,b)};
  if(n.name=="Subtract" && lo>=b)return Bounds{lo-b,hi-a};
  if(n.name=="Divide" && a)return Bounds{lo/b,hi/a};
  if(n.name=="Remainder" && a==b && a)return Bounds{0,std::min(hi,a-1)};
  return {};
}
std::optional<AffineOffset> affine(const scalar_fold::Graph& graph,const scalar_fold::Node& n,
                                 const std::vector<std::optional<AffineOffset>>& previous,
                                 const std::map<std::uint32_t,std::uint64_t>& domains) {
  using K=scalar_fold::Kind;
  if(n.type!="u64")return {};
  if(n.kind==K::Index)return AffineOffset{0,{{n.index_level,1}}};
  if(n.kind==K::Constant) {
    if(auto value=unsignedLiteral(n.name))return AffineOffset{*value,{}};return {};
  }
  if(n.kind!=K::Operation || !n.attributes.empty() || n.operands.size()!=2)return {};
  for(const auto& ref:n.operands)if(!ref.conversion.empty() || !ref.pointer_provenance.empty() ||
      graph.type(ref)!="u64" || !previous.at(ref.node))return {};
  auto left=*previous[n.operands[0].node],right=*previous[n.operands[1].node];
  if(n.name=="Add") {
    left.base=checkedAdd(left.base,right.base);
    for(auto [level,c]:right.terms)left.terms[level]=checkedAdd(left.terms[level],c);return left;
  }
  if(n.name=="Multiply") {
    if(!right.terms.empty())std::swap(left,right);
    if(!right.terms.empty())return {};
    left.base=checkedMultiply(left.base,right.base);
    for(auto& [level,c]:left.terms){(void)level;c=checkedMultiply(c,right.base);}return left;
  }
  if((n.name=="Divide" || n.name=="Remainder") && right.terms.empty() && right.base) {
    const auto divisor=right.base;
    AffineOffset quotient{left.base/divisor,{}},remainder{left.base%divisor,{}};
    auto maximum=remainder.base;
    for(auto [level,c]:left.terms) {
      const auto found=domains.find(level);if(found==domains.end() || !found->second)return {};
      if(c/divisor)quotient.terms[level]=c/divisor;
      if(c%divisor)remainder.terms[level]=c%divisor;
      maximum=checkedAdd(maximum,checkedMultiply(c%divisor,found->second-1));
    }
    if(maximum<divisor)return n.name=="Divide"?quotient:remainder;
  }
  return {};
}
std::uint64_t scalarBytes(const std::string& type) {
  if(type=="e4m3" || type=="e5m2")return 1;
  static const std::set<std::string> types{"b8","b16","b32","b64","u8","u16","u32","u64",
      "s8","s16","s32","s64","f16","bf16","f32","f64"};
  if(!types.contains(type))throw std::invalid_argument("input_layout_unknown_element_type");
  auto first=type.find_first_of("0123456789");
  if(first==std::string::npos)throw std::invalid_argument("input_layout_unknown_element_type");
  auto bits=unsignedLiteral(type.substr(first));
  if(!bits || !*bits || *bits%8)throw std::invalid_argument("input_layout_unknown_element_type");
  return *bits/8;
}
std::uint64_t affineMaximum(const AffineOffset& value,const std::map<std::uint32_t,std::uint64_t>& domains) {
  auto maximum=value.base;
  for(auto [level,c]:value.terms) {
    const auto found=domains.find(level);
    if(found==domains.end())throw std::invalid_argument("input_layout_unbound_index");
    maximum=checkedAdd(maximum,checkedMultiply(c,found->second?found->second-1:0));
  }
  return maximum;
}
std::optional<AffineOffset> affineQuotient(const AffineOffset& value,std::uint64_t divisor,
    const std::map<std::uint32_t,std::uint64_t>& domains) {
  AffineOffset quotient{value.base/divisor,{}},remainder{value.base%divisor,{}};
  for(auto [level,c]:value.terms) {
    if(domains.at(level)<=1)continue;
    if(c/divisor)quotient.terms[level]=c/divisor;
    if(c%divisor)remainder.terms[level]=c%divisor;
  }
  if(affineMaximum(remainder,domains)>=divisor)return {};
  return quotient;
}
std::optional<AffineOffset> affineRemainder(const AffineOffset& value,std::uint64_t divisor,
    const std::map<std::uint32_t,std::uint64_t>& domains) {
  AffineOffset remainder{value.base%divisor,{}};
  for(auto [level,c]:value.terms)
    if(domains.at(level)>1 && c%divisor)remainder.terms[level]=c%divisor;
  if(affineMaximum(remainder,domains)>=divisor)return {};
  return remainder;
}
scalar_fold::Ref affineExpression(scalar_fold::Graph& graph,const AffineOffset& value) {
  const auto constant=[&](auto n){return graph.constant("u64",std::to_string(n));};
  std::optional<scalar_fold::Ref> result;
  if(value.base)result=constant(value.base);
  for(auto [level,c]:value.terms)if(c) {
    auto term=graph.index(level);
    if(c!=1)term=graph.operation("Multiply","u64",{term,constant(c)});
    result=result?graph.operation("Add","u64",{*result,term}):term;
  }
  return result.value_or(constant(0));
}
scalar_fold::Ref layoutOffset(scalar_fold::Graph& graph,scalar_fold::Ref offset,
    const AffineOffset& affine,const std::map<std::uint32_t,std::uint64_t>& domains,
    const InputTensorLayout& layout,std::uint64_t extent,const std::string& type) {
  const auto bytes=scalarBytes(type);
  if(affine.base<layout.source_offset)throw std::invalid_argument("input_layout_out_of_bounds");
  auto maximum=affine.base;
  for(auto [level,c]:affine.terms) {
    auto found=domains.find(level);
    if(found==domains.end())throw std::invalid_argument("input_layout_unbound_index");
    maximum=checkedAdd(maximum,checkedMultiply(c,found->second?found->second-1:0));
    if(c%layout.element_bytes)throw std::invalid_argument("input_layout_unaligned_index");
  }
  if(maximum-layout.source_offset>extent || bytes>extent-(maximum-layout.source_offset))
    throw std::invalid_argument("input_layout_out_of_bounds");
  if(layout.source_strides_bytes==layout.target_strides_bytes) {
    auto translated=affine;
    translated.base=checkedAdd(affine.base-layout.source_offset,layout.target_offset);
    return affineExpression(graph,translated);
  }
  auto remainder=(affine.base-layout.source_offset)%layout.element_bytes;
  if(bytes>layout.element_bytes-remainder)throw std::invalid_argument("input_layout_crosses_element");
  const auto constant=[&](auto n){return graph.constant("u64",std::to_string(n));};
  if(affine.terms.empty()) {
    auto translated=layout.target_offset+remainder;
    for(std::size_t i=0;i<layout.shape.size();++i)if(layout.shape[i]>1)
      translated=checkedAdd(translated,checkedMultiply(
        ((affine.base-layout.source_offset)/layout.source_strides_bytes[i])%layout.shape[i],
        layout.target_strides_bytes[i]));
    return constant(translated);
  }
  auto relative_affine=affine;relative_affine.base-=layout.source_offset;
  std::vector<std::optional<AffineOffset>> coordinates(layout.shape.size());
  bool all_affine=true;
  for(std::size_t i=0;i<layout.shape.size();++i) {
    if(layout.shape[i]==1){coordinates[i]=AffineOffset{};continue;}
    if(auto q=affineQuotient(relative_affine,layout.source_strides_bytes[i],domains))
      coordinates[i]=affineRemainder(*q,layout.shape[i],domains);
    all_affine=all_affine && coordinates[i].has_value();
  }
  if(all_affine) {
    AffineOffset translated{layout.target_offset+remainder,{}};
    for(std::size_t i=0;i<coordinates.size();++i) {
      const auto& coordinate=*coordinates[i];const auto stride=layout.target_strides_bytes[i];
      translated.base=checkedAdd(translated.base,checkedMultiply(coordinate.base,stride));
      for(auto [level,c]:coordinate.terms)
        translated.terms[level]=checkedAdd(translated.terms[level],checkedMultiply(c,stride));
    }
    return affineExpression(graph,translated);
  }
  auto relative=layout.source_offset?graph.operation("Subtract","u64",{offset,constant(layout.source_offset)}):offset;
  auto result=constant(layout.target_offset+remainder);
  for(std::size_t axis=0;axis<layout.shape.size();++axis)if(layout.shape[axis]>1) {
    auto coordinate=relative;
    if(coordinates[axis])coordinate=affineExpression(graph,*coordinates[axis]);
    else {
      if(layout.source_strides_bytes[axis]!=1)
        coordinate=graph.operation("Divide","u64",{coordinate,constant(layout.source_strides_bytes[axis])});
      coordinate=graph.operation("Remainder","u64",{coordinate,constant(layout.shape[axis])});
    }
    if(layout.target_strides_bytes[axis]!=1)
      coordinate=graph.operation("Multiply","u64",{coordinate,constant(layout.target_strides_bytes[axis])});
    result=graph.operation("Add","u64",{result,coordinate});
  }
  return result;
}
} // namespace
AlignedInputs alignInputs(const scalar_fold::Artifact& source,const InputIdentityMapping& mapping,
                          bool normalize_values) {
  namespace s=scalar_fold;
  AlignedInputs result;
  try {
    validate(mapping.allocations);validate(mapping.parameters);
    for(const auto& [name,value]:mapping.parameter_constants) {
      validateConstant(value);
      if(name.empty())
        throw std::invalid_argument("invalid_input_parameter_constant");
      if(mapping.parameters.contains(name))
        throw std::invalid_argument("input_parameter_both_renamed_and_specialized");
    }
    std::map<std::string,std::uint64_t> extents;
    for(const auto& [name,layout]:mapping.layouts) {
      mapped(mapping.allocations,name,"layout_has_no_allocation_mapping");
      extents.emplace(name,layoutExtent(layout));
    }
    std::vector<s::Ref> roots;
    for(const auto& output:source.outputs)roots.push_back(output.value);
    source.graph.validateRoots(roots);
    for(const auto& family:source.output_families) {
      source.graph.validateRoots(std::span<const s::Ref>(&family.body,1),family.shape.size());
      roots.push_back(family.body);
    }
    std::vector<bool> live(source.graph.size());
    std::vector<s::Id> pending;
    for(const auto& root:roots)pending.push_back(root.node);
    while(!pending.empty()) {
      auto id=pending.back();pending.pop_back();
      if(live.at(id))continue;
      live[id]=true;
      for(const auto& operand:source.graph.node(id).operands)pending.push_back(operand.node);
    }
    using Domains=std::map<std::uint32_t,std::uint64_t>;
    std::vector<Domains> scoped_domains(source.graph.size());
    const auto mergeDomains=[](Domains& target,const Domains& source) {
      for(auto [level,count]:source)target[level]=std::max(target[level],count);
    };
    for(const auto& family:source.output_families)
      for(std::size_t i=0;i<family.shape.size();++i)
        scoped_domains[family.body.node][i]=std::max(scoped_domains[family.body.node][i],family.shape[i]);
    // Nodes are topologically stored; propagate only enclosing binder domains.
    for(std::size_t next=source.graph.size();next>0;--next) {
      const auto id=next-1;if(!live[id])continue;
      const auto& n=source.graph.node(id);
      for(std::size_t i=0;i<n.operands.size();++i) {
        auto& target=scoped_domains[n.operands[i].node];
        mergeDomains(target,scoped_domains[id]);
        if(n.fold && i==1)
          target[n.fold->binder_level]=std::max(target[n.fold->binder_level],n.fold->count);
      }
    }
    std::vector<std::optional<AffineOffset>> offsets(source.graph.size());
    std::vector<std::optional<Bounds>> bounds(source.graph.size());
    std::vector<s::Ref> replacements(source.graph.size());
    auto& graph=result.artifact.graph;
    RepresentationRewriter representation_rewriter(graph);
    const bool trace=normalize_values && sentryward::debug::environment("SENTRYWARD_REPRESENTATION_TRACE");
    const auto traceStats=[&](const char* phase,std::size_t root,std::size_t old,
                              std::size_t mapped,std::size_t conversion_bytes) {
      if(!trace)return;
      const auto stats=representation_rewriter.statistics();const auto& bits=stats.bit_origins;
      SENTRYWARD_DEBUG_PRINT(input_mapping_0);
    };
    const auto use=[&](s::Ref old) {
      auto value=replacements.at(old.node);
      if(!old.conversion.empty()) {
        if(!value.conversion.empty())value.conversion+="; ";
        value.conversion+=old.conversion;value.converted_type=old.converted_type;
      }
      value.pointer_provenance=old.pointer_provenance;
      if(normalize_values) {
        value=representation_rewriter.rewrite(std::move(value),scoped_domains.at(old.node));
      }
      const auto& node=graph.node(value);
      if(normalize_values && node.kind==s::Kind::Constant && value.pointer_provenance.empty()) {
        if(auto exact=constantValue(node.type,node.name,value.conversion,value.converted_type))
          return graph.constant(exact->type,exact->literal);
      }
      return value;
    };
    for(s::Id id=0;id<source.graph.size();++id)if(live[id]) {
      const auto& n=source.graph.node(id);
      const auto& domains=scoped_domains[id];
      if(!mapping.layouts.empty() || (normalize_values && n.type=="u64")) {
        try { offsets[id]=affine(source.graph,n,offsets,domains); }
        catch(const std::invalid_argument&) { offsets[id].reset(); }
        try { bounds[id]=offsetBounds(source.graph,n,bounds,domains); }
        catch(const std::invalid_argument&) { bounds[id].reset(); }
        // Reuse the existing exact, bottom-up offset summary even when no
        // physical-layout remapping is requested. Require a nonwrapping range
        // for this new normalization path, including every intermediate.
        // This visits stored nodes/lexical axes, never logical Fold members.
        if(mapping.layouts.empty() && !bounds[id])offsets[id].reset();
      }
      std::vector<s::Ref> args;
      for(const auto& operand:n.operands)args.push_back(use(operand));
      s::Ref value;
      switch(n.kind) {
        case s::Kind::Constant:value=graph.constant(n.type,n.name);break;
        case s::Kind::Parameter:
          if(const auto bound=mapping.parameter_constants.find(n.name);bound!=mapping.parameter_constants.end()) {
            if(bound->second.type!=n.type)throw std::invalid_argument("input_parameter_constant_type_mismatch");
            value=graph.constant(n.type,bound->second.literal);
          } else value=graph.parameter(n.type,mapped(mapping.parameters,n.name,"unmapped_input_parameter"));
          break;
        case s::Kind::InitialCell: {
          auto offset=args.at(0);
          if(const auto layout=mapping.layouts.find(n.name);layout!=mapping.layouts.end()) {
            const auto& original=n.operands.at(0);
            if(!original.conversion.empty() || !original.pointer_provenance.empty())
              throw std::invalid_argument("input_layout_non_affine_offset");
            if(offsets.at(original.node))
              offset=layoutOffset(graph,offset,*offsets[original.node],domains,layout->second,extents.at(n.name),n.type);
            else {
              const auto& l=layout->second;auto range=bounds.at(original.node);
              if(!range || l.source_strides_bytes!=l.target_strides_bytes) {
                throw std::invalid_argument("input_layout_non_affine_offset");
              }
              const auto extent=extents.at(n.name),bytes=scalarBytes(n.type);
              if(range->first<l.source_offset || range->second-l.source_offset>extent ||
                  bytes>extent-(range->second-l.source_offset))
                throw std::invalid_argument("input_layout_out_of_bounds");
              if(l.source_offset)offset=graph.operation("Subtract","u64",{offset,graph.constant("u64",std::to_string(l.source_offset))});
              if(l.target_offset)offset=graph.operation("Add","u64",{offset,graph.constant("u64",std::to_string(l.target_offset))});
            }
          }
          const auto& allocation=mapped(mapping.allocations,n.name,"unmapped_input_allocation");
          const auto& address=graph.node(offset);
          if(mapping.comparison_context && offset.conversion.empty() && offset.pointer_provenance.empty() &&
              address.kind==s::Kind::Constant && address.type=="u64") {
            if(const auto at=unsignedLiteral(address.name))
              if(const auto literal=mapping.comparison_context->literal(allocation,*at,n.type)) {
                value=graph.constant(n.type,*literal);break;
              }
          }
          value=graph.initialCell(allocation,offset,n.type);break;
        }
        case s::Kind::Index:value=graph.index(n.index_level);break;
        case s::Kind::Operation: {
          if(normalize_values && n.attributes.empty() && n.name=="Select" && args.size()==3) {
            const auto condition=graph.node(args[0]);
            if(args[0].conversion.empty() && condition.kind==s::Kind::Constant && condition.type=="pred" &&
                (condition.name=="true" || condition.name=="false")) {
              value=args[condition.name=="true"?1:2];break;
            }
          }
          if(normalize_values && offsets[id]) {value=affineExpression(graph,*offsets[id]);break;}
          std::vector<ConstantValue> constants;
          for(const auto& arg:args) {
            const auto& operand=graph.node(arg);
            if(operand.kind!=s::Kind::Constant || !arg.conversion.empty() || !arg.pointer_provenance.empty())break;
            constants.push_back({operand.type,operand.name});
          }
          auto exact=normalize_values && constants.size()==args.size()?constantOperation(n.name,n.type,n.attributes,constants):std::nullopt;
          if(normalize_values && n.attributes.empty() && n.name=="Equal" && n.type=="pred" && constants.size()==2 &&
              constants[0].type=="u64" && constants[1].type=="u64") {
            value=graph.constant("pred",constants[0].literal==constants[1].literal?"true":"false");break;
          }
          value=exact?graph.constant(exact->type,exact->literal):graph.operation(n.name,n.type,std::move(args),n.attributes);
          break;
        }
        case s::Kind::Fold:
          value=graph.fold(n.fold->reducer,args.at(0),args.at(1),n.fold->binder_level,n.fold->count,n.type);break;
      }
      replacements[id]=value;
    }
    traceStats("cloned",0,0,0,0);
    roots.clear();
    std::size_t traced_roots=0;
    for(const auto& output:source.outputs) {
      if(traced_roots<8)traceStats("root_before",traced_roots,output.value.node,
                                 replacements[output.value.node].node,output.value.conversion.size());
      auto mapped_output=output;mapped_output.value=use(output.value);
      if(traced_roots<8)traceStats("root_after",traced_roots,output.value.node,
                                 mapped_output.value.node,mapped_output.value.conversion.size());
      ++traced_roots;
      roots.push_back(mapped_output.value);result.artifact.outputs.push_back(std::move(mapped_output));
    }
    graph.validateRoots(roots);
    for(const auto& family:source.output_families) {
      if(traced_roots<8)traceStats("root_before",traced_roots,family.body.node,
                                 replacements[family.body.node].node,family.body.conversion.size());
      auto mapped_family=family;mapped_family.body=use(family.body);
      if(traced_roots<8)traceStats("root_after",traced_roots,family.body.node,
                                 mapped_family.body.node,mapped_family.body.conversion.size());
      ++traced_roots;
      graph.validateRoots(std::span<const s::Ref>(&mapped_family.body,1),mapped_family.shape.size());
      result.artifact.output_families.push_back(std::move(mapped_family));
    }
    traceStats("end",traced_roots,0,0,0);
    result.complete=true;
  } catch(const std::invalid_argument& e) {result.reason=e.what();}
    catch(const std::out_of_range& e) {result.reason=e.what();}
  return result;
}
} // namespace sentryward::equivalence
