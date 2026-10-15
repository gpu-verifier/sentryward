#include "sentryward/scalar_fold/consolidate_outputs.h"
#include <algorithm>
#include <charconv>
#include <map>
#include <optional>
#include <set>
#include <unordered_set>
#include <stdexcept>

namespace sentryward::scalar_fold {
namespace {
struct Affine { std::uint64_t base=0;std::map<std::uint32_t,std::uint64_t> coefficients; };
struct Forms : std::vector<std::optional<Affine>> {
  std::vector<bool> depends_on_output;
};
struct PairHash {
  std::size_t operator()(const std::pair<Id,Id>& pair) const {
    return std::hash<Id>{}(pair.first) ^ (std::hash<Id>{}(pair.second)+0x9e3779b9+(pair.first<<6)+(pair.first>>2));
  }
};
bool plain(const Ref& ref) {return ref.conversion.empty() && ref.pointer_provenance.empty();}
void extendForms(const Graph& graph,Forms& result) {
  const auto begin=result.size();
  result.resize(graph.size());
  result.depends_on_output.resize(graph.size());
  for(Id id=begin;id<graph.size();++id) {
    const auto& n=graph.node(id);
    bool depends=n.kind==Kind::Index && n.index_level==0;
    for(const auto& operand:n.operands)depends=depends || result.depends_on_output[operand.node];
    result.depends_on_output[id]=depends;
    if(n.type!="u64" || !n.attributes.empty())continue;
    if(n.kind==Kind::Index){result[id]=Affine{0,{{n.index_level,1}}};continue;}
    if(n.kind==Kind::Constant) {
      std::uint64_t value=0;auto parsed=std::from_chars(n.name.data(),n.name.data()+n.name.size(),value);
      if(parsed.ec==std::errc{} && parsed.ptr==n.name.data()+n.name.size())result[id]=Affine{value,{}};
      continue;
    }
    if(n.kind!=Kind::Operation || n.operands.size()!=2 || !plain(n.operands[0]) || !plain(n.operands[1]))continue;
    auto a=result[n.operands[0].node],b=result[n.operands[1].node];if(!a || !b)continue;
    if(n.name=="Add" || n.name=="Subtract") {
      auto sign=n.name=="Add"?std::uint64_t{1}:UINT64_MAX;
      a->base+=sign*b->base;
      for(auto [axis,c]:b->coefficients)a->coefficients[axis]+=sign*c;
    } else if(n.name=="Multiply") {
      if(!a->coefficients.empty() && !b->coefficients.empty())continue;
      if(a->coefficients.empty())std::swap(a,b);
      a->base*=b->base;for(auto& [axis,c]:a->coefficients)c*=b->base;
    } else continue;
    std::erase_if(a->coefficients,[](const auto& item){return !item.second;});
    result[id]=std::move(a);
  }
}
bool shifted(const Graph& graph,Ref first,Ref next,std::uint64_t shift,
             const Forms& affine,bool singleton) {
  std::vector<std::pair<Ref,Ref>> work{{first,next}};
  std::unordered_set<std::pair<Id,Id>,PairHash> seen;
  while(!work.empty()) {
    auto [x,y]=work.back();work.pop_back();
    if(x.conversion!=y.conversion || x.converted_type!=y.converted_type || x.pointer_provenance!=y.pointer_provenance)return false;
    if(x.node==y.node && !affine.depends_on_output[x.node])continue;
    if(!seen.emplace(x.node,y.node).second)continue;
    const auto& a=affine[x.node];const auto& b=affine[y.node];
    if(a && b) {
      auto ac=a->coefficients,bc=b->coefficients;
      if(singleton){ac.erase(0);bc.erase(0);}
      if(ac!=bc)return false;
      auto found=a->coefficients.find(0);auto c=found==a->coefficients.end()?0:found->second;
      if(a->base+c*shift!=b->base)return false;
      continue;
    }
    const auto& p=graph.node(x);const auto& q=graph.node(y);
    if(p.kind!=q.kind || p.type!=q.type || p.name!=q.name || p.attributes!=q.attributes ||
        p.fold!=q.fold || p.index_level!=q.index_level || p.operands.size()!=q.operands.size())return false;
    for(std::size_t i=0;i<p.operands.size();++i)work.emplace_back(p.operands[i],q.operands[i]);
  }
  return true;
}
std::optional<Ref> generalizeSingletons(Graph& graph,Ref first,Ref next,
                                      const Forms& affine) {
  using Pair=std::pair<Id,Id>;
  std::map<Pair,Ref> result;
  std::vector<std::pair<Ref,Ref>> work{{first,next}};
  auto decorated=[](Ref value,const Ref& source){
    value.conversion=source.conversion;value.converted_type=source.converted_type;
    value.pointer_provenance=source.pointer_provenance;return value;
  };
  while(!work.empty()) {
    const auto [x,y]=work.back();
    if(x.conversion!=y.conversion || x.converted_type!=y.converted_type ||
       x.pointer_provenance!=y.pointer_provenance)return std::nullopt;
    const Pair key{x.node,y.node};
    if(result.contains(key)){work.pop_back();continue;}
    if(x.node==y.node && !affine.depends_on_output[x.node]) {
      result.emplace(key,graph.ref(x.node));work.pop_back();continue;
    }
    const auto a=affine[x.node],b=affine[y.node];
    if(a && b) {
      auto coefficients=a->coefficients,other=b->coefficients;
      coefficients.erase(0);other.erase(0);
      if(coefficients!=other)return std::nullopt;
      coefficients[0]=b->base-a->base;
      auto value=graph.constant("u64",std::to_string(a->base));
      for(auto [axis,c]:coefficients)if(c) {
        auto term=graph.index(axis);
        if(c!=1)term=graph.operation("Multiply","u64",{term,graph.constant("u64",std::to_string(c))});
        value=graph.operation("Add","u64",{value,term});
      }
      result.emplace(key,value);work.pop_back();continue;
    }
    const auto p=graph.node(x),q=graph.node(y);
    if(p.kind!=q.kind || p.type!=q.type || p.name!=q.name || p.attributes!=q.attributes ||
       p.fold!=q.fold || p.index_level!=q.index_level || p.operands.size()!=q.operands.size())return std::nullopt;
    bool ready=true;
    for(std::size_t i=0;i<p.operands.size();++i) {
      const auto left=p.operands[i],right=q.operands[i];
      if(left.conversion!=right.conversion || left.converted_type!=right.converted_type ||
         left.pointer_provenance!=right.pointer_provenance)return std::nullopt;
      if(!result.contains({left.node,right.node})){work.emplace_back(left,right);ready=false;}
    }
    if(!ready)continue;
    std::vector<Ref> operands;
    for(std::size_t i=0;i<p.operands.size();++i)
      operands.push_back(decorated(result.at({p.operands[i].node,q.operands[i].node}),p.operands[i]));
    Ref value;
    switch(p.kind) {
      case Kind::Constant:value=graph.constant(p.type,p.name);break;
      case Kind::Parameter:value=graph.parameter(p.type,p.name);break;
      case Kind::Index:value=graph.index(p.index_level);break;
      case Kind::InitialCell:value=graph.initialCell(p.name,operands.at(0),p.type);break;
      case Kind::Operation:value=graph.operation(p.name,p.type,std::move(operands),p.attributes);break;
      case Kind::Fold:value=graph.fold(p.fold->reducer,operands.at(0),operands.at(1),
        p.fold->binder_level,p.fold->count,p.type);break;
    }
    result.emplace(key,value);work.pop_back();
  }
  return decorated(result.at({first.node,next.node}),first);
}
bool flat(const OutputFamily& family) {
  return family.element_bytes && family.shape.size()==1 && family.shape[0] &&
    family.strides_bytes==std::vector<std::uint64_t>{family.element_bytes} &&
    family.shape[0]<=UINT64_MAX/family.element_bytes &&
    family.shape[0]*family.element_bytes<=UINT64_MAX-family.offset;
}
std::optional<std::uint64_t> scalarBytes(const std::string& type) {
  if(type=="bf16")return 2;
  if(type.size()<2 || (type[0]!='b' && type[0]!='u' && type[0]!='s' && type[0]!='f'))return {};
  unsigned bits=0;
  auto parsed=std::from_chars(type.data()+1,type.data()+type.size(),bits);
  if(parsed.ec!=std::errc{} || parsed.ptr!=type.data()+type.size() || !bits || bits%8 || bits>64)return {};
  return bits/8;
}
void recoverCells(const Graph& graph,std::vector<OutputFamily>& families,
                  const std::vector<std::optional<Affine>>& affine) {
  std::vector<bool> depends(graph.size());
  for(Id id=0;id<graph.size();++id) {
    const auto& node=graph.node(id);
    depends[id]=node.kind==Kind::Index && node.index_level==0;
    for(const auto& operand:node.operands)depends[id]=depends[id] || depends[operand.node];
  }
  for(auto& family:families) {
    if(!flat(family) || !plain(family.body) || !family.body.converted_type.empty())continue;
    const auto& node=graph.node(family.body);
    if(node.kind!=Kind::Operation || node.name!="ExtractBits" || node.operands.size()!=2 ||
       node.attributes!="width="+std::to_string(8*family.element_bytes) ||
       scalarBytes(node.type)!=family.element_bytes || !plain(node.operands[1]))continue;
    const auto source=node.operands[0];
    const auto bytes=scalarBytes(graph.type(source));
    if(!bytes || *bytes!=family.shape[0]*family.element_bytes || depends[source.node])continue;
    const auto& offset=affine[node.operands[1].node];
    if(!offset || offset->base || offset->coefficients!=std::map<std::uint32_t,std::uint64_t>{{0,8*family.element_bytes}})continue;
    family.element_bytes=*bytes;family.shape={1};family.strides_bytes={*bytes};family.body=source;
  }
}
Ref addOuterBinder(Graph& graph,Ref root,std::map<Id,Ref>& mapped) {
  const auto use=[&](const Ref& old) {
    auto result=mapped.at(old.node);result.conversion=old.conversion;
    result.converted_type=old.converted_type;result.pointer_provenance=old.pointer_provenance;return result;
  };
  std::vector<std::pair<Id,bool>> work{{root.node,false}};
  while(!work.empty()) {
    auto [id,ready]=work.back();work.pop_back();if(mapped.contains(id))continue;
    const auto node=graph.node(id);
    if(!ready) {
      work.emplace_back(id,true);
      for(const auto& arg:node.operands)if(!mapped.contains(arg.node))work.emplace_back(arg.node,false);
      continue;
    }
    std::vector<Ref> args;for(const auto& arg:node.operands)args.push_back(use(arg));
    Ref value;
    const auto level=[](std::uint32_t old) {
      if(old==UINT32_MAX)throw std::invalid_argument("output_binder_overflow");return old+1;
    };
    switch(node.kind) {
      case Kind::Constant:value=graph.constant(node.type,node.name);break;
      case Kind::Parameter:value=graph.parameter(node.type,node.name);break;
      case Kind::InitialCell:value=graph.initialCell(node.name,args.at(0),node.type);break;
      case Kind::Index:value=graph.index(level(node.index_level));break;
      case Kind::Operation:value=graph.operation(node.name,node.type,std::move(args),node.attributes);break;
      case Kind::Fold:value=graph.fold(node.fold->reducer,args.at(0),args.at(1),level(node.fold->binder_level),node.fold->count,node.type);break;
    }
    mapped.emplace(id,value);
  }
  return use(root);
}
std::vector<OutputFamily> tensorize(Graph& graph,const std::vector<OutputFamily>& families,
                                  Forms& affine) {
  std::vector<OutputFamily> result;std::map<Id,Ref> shifted_roots;
  for(std::size_t i=0;i<families.size();) {
    const auto& first=families[i];
    auto end=i+1;
    if(flat(first) && first.shape[0]>1)while(end<families.size()) {
      const auto& next=families[end];
      const auto offset=static_cast<__int128>(first.offset)+static_cast<__int128>(end-i)*first.shape[0]*first.element_bytes;
      if(!flat(next) || next.allocation!=first.allocation || next.shape!=first.shape ||
         next.element_bytes!=first.element_bytes || offset!=next.offset)break;
      ++end;
    }
    if(end-i>1) {
      auto a=addOuterBinder(graph,first.body,shifted_roots);
      auto b=addOuterBinder(graph,families[i+1].body,shifted_roots);
      extendForms(graph,affine);
      if(auto generalized=generalizeSingletons(graph,a,b,affine)) {
        extendForms(graph,affine);std::size_t count=2;
        while(i+count<end) {
          auto next=addOuterBinder(graph,families[i+count].body,shifted_roots);
          extendForms(graph,affine);
          if(!shifted(graph,*generalized,next,count,affine,true))break;
          ++count;
        }
        auto combined=first;combined.shape={count,first.shape[0]};
        combined.strides_bytes={first.shape[0]*first.element_bytes,first.element_bytes};
        combined.body=*generalized;graph.validateRoots(std::span(&combined.body,1),2);
        result.push_back(std::move(combined));i+=count;continue;
      }
    }
    result.push_back(first);++i;
  }
  return result;
}
}
std::size_t consolidateOutputs(Artifact& artifact,bool tensor_families,
                               std::span<std::optional<Id>> weak_nodes) {
  auto& graph=artifact.graph;
  std::map<std::uint32_t,std::vector<Ref>> validation_roots;
  for(const auto& output:artifact.outputs)validation_roots[0].push_back(output.value);
  for(const auto& f:artifact.output_families)validation_roots[f.shape.size()].push_back(f.body);
  for(const auto& [depth,roots]:validation_roots)graph.validateRoots(roots,depth);
  Forms affine;
  extendForms(graph,affine);
  auto families=artifact.output_families;
  recoverCells(graph,families,affine);
  std::stable_sort(families.begin(),families.end(),[](const auto& a,const auto& b){
    return std::tie(a.allocation,a.offset)<std::tie(b.allocation,b.offset);
  });
  std::vector<OutputFamily> merged;
  for(const auto& family:families) {
    if(!merged.empty()) {
      auto& previous=merged.back();
      if(flat(previous) && flat(family) && previous.allocation==family.allocation &&
          previous.element_bytes==family.element_bytes &&
          previous.offset+previous.shape[0]*previous.element_bytes==family.offset) {
        bool matches=shifted(graph,previous.body,family.body,previous.shape[0],affine,family.shape[0]==1);
        if(!matches && previous.shape[0]==1 && family.shape[0]==1) {
          auto generalized=generalizeSingletons(graph,previous.body,family.body,affine);
          extendForms(graph,affine);
          if(generalized){previous.body=*generalized;matches=true;}
        }
        if(matches){previous.shape[0]+=family.shape[0];continue;}
      }
    }
    merged.push_back(family);
  }
  if(tensor_families)merged=tensorize(graph,merged,affine);
  const auto removed=artifact.output_families.size()-merged.size();
  artifact.output_families=std::move(merged);
  std::vector<Ref> roots;
  for(const auto& output:artifact.outputs)roots.push_back(output.value);
  for(const auto& family:artifact.output_families)roots.push_back(family.body);
  graph.collect(roots,weak_nodes);
  std::size_t i=0;
  for(auto& output:artifact.outputs)output.value=roots[i++];
  for(auto& family:artifact.output_families)family.body=roots[i++];
  return removed;
}
}
