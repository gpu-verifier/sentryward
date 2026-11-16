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
  // Read-only: the pairs point into first, next and graph storage, which stays put.
  std::vector<std::pair<const Ref*,const Ref*>> work{{&first,&next}};
  std::unordered_set<std::pair<Id,Id>,PairHash> seen;
  while(!work.empty()) {
    const auto [px,py]=work.back();work.pop_back();
    const Ref& x=*px;const Ref& y=*py;
    if(x.conversion!=y.conversion || x.converted_type!=y.converted_type || x.pointer_provenance!=y.pointer_provenance)return false;
    if(x.node==y.node && !affine.depends_on_output[x.node])continue;
    if(!seen.emplace(x.node,y.node).second)continue;
    const auto& a=affine[x.node];const auto& b=affine[y.node];
    if(a && b) {
      // The coefficient maps compared without copies (without axis 0 for a singleton).
      const auto skip=[&](const auto& map){auto it=map.begin();if(singleton && it!=map.end() && it->first==0)++it;return it;};
      if(!std::equal(skip(a->coefficients),a->coefficients.end(),skip(b->coefficients),b->coefficients.end()))return false;
      auto found=a->coefficients.find(0);auto c=found==a->coefficients.end()?0:found->second;
      if(a->base+c*shift!=b->base)return false;
      continue;
    }
    const auto& p=graph.node(x);const auto& q=graph.node(y);
    if(p.kind!=q.kind || p.type!=q.type || p.name!=q.name || p.attributes!=q.attributes ||
        p.fold!=q.fold || p.index_level!=q.index_level || p.operands.size()!=q.operands.size())return false;
    for(std::size_t i=0;i<p.operands.size();++i)work.emplace_back(&p.operands[i],&q.operands[i]);
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
      case Kind::Operation:
        value=graph.operation(p.name,p.type,std::move(operands),p.attributes);
        graph.markInstruction(value,instruction_mode::common(graph.instructionModes(x.node),graph.instructionModes(y.node)));
        break;
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
      case Kind::Operation:
        value=graph.operation(node.name,node.type,std::move(args),node.attributes);
        graph.markInstruction(value,graph.instructionModes(id));
        break;
      case Kind::Fold:value=graph.fold(node.fold->reducer,args.at(0),args.at(1),level(node.fold->binder_level),node.fold->count,node.type);break;
    }
    mapped.emplace(id,value);
  }
  return use(root);
}
// Class mode's member families (mergeRepresentativeFamilies): contiguous rows only.
std::vector<OutputFamily> tensorizeContiguous(Graph& graph,const std::vector<OutputFamily>& families,
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
// Rows of one output tensor: flat families of one shape whose bodies are one body over an outer
// row binder (generalizeSingletons on the first two, then `shifted` checks every further row
// exactly). The row stride is the distance to a partner row: the adjacent family when rows are
// contiguous, or a later one when other families (another CTA's rows of a [row, head, column]
// tensor) interleave. The first kTensorPartners candidate partners are tried and the longest run
// is kept. Every member is checked against the generalized body; nothing is extrapolated.
// O(F log F) lookups plus O(DAG) per tried partner and per member for F families.
constexpr std::size_t kTensorPartners=4;
// A run is kept only from this many rows on: it then replaces at least this many families (the
// per-family work of the comparison); shorter runs keep their row families.
constexpr std::size_t kTensorMinRows=16;
// Shape signatures of family bodies: equal signatures are necessary for generalizeSingletons on two
// bodies (addOuterBinder shifts every level of both alike). A node hashes its kind, type, name,
// attributes, fold and index level and its operand edges (node signature and decorations), except
// that an affine u64 node hashes only its coefficients: its base is what generalization varies.
// Unequal signatures skip a partner without building its shifted copy; equal ones are still
// generalized and checked exactly, so a hash collision only costs that work. O(DAG) per new node,
// memoized across families (each node once).
class ShapeSignatures {
  std::vector<std::size_t> memo;std::vector<char> done;
  static void mix(std::size_t& h,std::size_t x){h^=x+0x9e3779b97f4a7c15ULL+(h<<6)+(h>>2);}
  static std::size_t edge(std::size_t node,const Ref& ref) {
    std::size_t h=node;const std::hash<std::string> text;
    mix(h,text(ref.conversion));mix(h,text(ref.converted_type));mix(h,text(ref.pointer_provenance));
    return h;
  }
 public:
  std::size_t of(const Graph& graph,const Forms& affine,const Ref& root) {
    if(memo.size()<graph.size()){memo.resize(graph.size());done.resize(graph.size());}
    std::vector<std::pair<Id,bool>> work{{root.node,false}};
    while(!work.empty()) {
      const auto [id,ready]=work.back();work.pop_back();
      if(done[id])continue;
      const auto& n=graph.node(id);
      if(!ready) {
        work.emplace_back(id,true);
        if(!affine[id])for(const auto& operand:n.operands)if(!done[operand.node])work.emplace_back(operand.node,false);
        continue;
      }
      std::size_t h=std::size_t(n.kind)+1;const std::hash<std::string> text;
      if(const auto& form=affine[id]) {
        mix(h,0xa5a5);for(const auto& [level,c]:form->coefficients){mix(h,level);mix(h,c);}
      } else {
        mix(h,text(n.type));mix(h,text(n.name));mix(h,text(n.attributes));mix(h,n.index_level);
        if(n.fold){mix(h,std::size_t(n.fold->reducer));mix(h,n.fold->binder_level);mix(h,n.fold->count);}
        for(const auto& operand:n.operands)mix(h,edge(memo[operand.node],operand));
      }
      memo[id]=h;done[id]=1;
    }
    return edge(memo[root.node],root);
  }
};
std::vector<OutputFamily> tensorizeRows(Graph& graph,const std::vector<OutputFamily>& families,
                                  Forms& affine) {
  std::vector<OutputFamily> result;std::map<Id,Ref> shifted_roots;
  std::map<std::pair<std::string,std::uint64_t>,std::size_t> at;
  for(std::size_t i=0;i<families.size();++i)at.emplace(std::pair{families[i].allocation,families[i].offset},i);
  std::vector<bool> used(families.size());
  const auto row=[&](const OutputFamily& f){return static_cast<unsigned __int128>(f.shape[0])*f.element_bytes;};
  // An output at a symbolic position (kPlacementOperation) keeps its own family: its offset is
  // only the constant part of its position.
  const auto placed=[&](const OutputFamily& f) {
    const auto& node=graph.node(f.body);
    return node.kind==Kind::Operation && node.name==kPlacementOperation;
  };
  const auto sameRow=[&](const OutputFamily& f,const OutputFamily& first) {
    return flat(f) && !placed(f) && f.allocation==first.allocation && f.shape==first.shape &&
        f.element_bytes==first.element_bytes;
  };
  // Rows are merged only inside a regular tiling: a maximal range of consecutive families (in
  // (allocation, offset) order) that are all rows of one shape, so every family between two
  // members of a run (another tensor's interleaved row) has that shape too. A row cut into pieces
  // of different lengths (column ranges computed differently, such as chunk prefixes) keeps its
  // row families, whose per-row comparison those pieces need. O(F).
  ShapeSignatures shapes;
  std::vector<std::size_t> tile(families.size());
  for(std::size_t k=1;k<families.size();++k)
    tile[k]=tile[k-1]+!(flat(families[k-1]) && !placed(families[k-1]) && sameRow(families[k],families[k-1]));
  for(std::size_t i=0;i<families.size();++i) {
    if(used[i])continue;
    const auto& first=families[i];
    std::vector<std::size_t> best{i};std::optional<Ref> best_body;std::uint64_t best_stride=0;
    if(flat(first) && !placed(first) && first.shape[0]>1) {
      std::size_t tried=0;
      for(auto j=i+1;j<families.size() && tried<kTensorPartners;++j) {
        const auto& next=families[j];
        if(next.allocation!=first.allocation || tile[j]!=tile[i])break;
        if(used[j] || !sameRow(next,first) || static_cast<unsigned __int128>(next.offset)<first.offset+row(first))continue;
        ++tried;
        const auto signature=shapes.of(graph,affine,first.body);
        if(signature!=shapes.of(graph,affine,next.body))continue;
        const auto stride=next.offset-first.offset;
        // A run is kept only from kTensorMinRows rows on, and each of its rows is an unused row of
        // this tiling with the first row's signature at first + k * stride: look the first
        // kTensorMinRows up before building any shifted copy. A partner that fails this can only
        // give a shorter run, which is not kept. O(kTensorMinRows log F).
        bool reaches=true;
        for(std::size_t k=2;k<kTensorMinRows && reaches;++k) {
          const auto offset=static_cast<unsigned __int128>(first.offset)+static_cast<unsigned __int128>(k)*stride;
          const auto found=offset>UINT64_MAX?at.end():at.find({first.allocation,static_cast<std::uint64_t>(offset)});
          reaches=found!=at.end() && !used[found->second] && tile[found->second]==tile[i] &&
              sameRow(families[found->second],first) && shapes.of(graph,affine,families[found->second].body)==signature;
        }
        if(!reaches)continue;
        auto a=addOuterBinder(graph,first.body,shifted_roots);
        auto b=addOuterBinder(graph,next.body,shifted_roots);
        extendForms(graph,affine);
        auto generalized=generalizeSingletons(graph,a,b,affine);
        if(!generalized)continue;
        extendForms(graph,affine);
        std::vector<std::size_t> members{i,j};
        for(;;) {
          const auto offset=static_cast<unsigned __int128>(first.offset)+static_cast<unsigned __int128>(members.size())*stride;
          if(offset>UINT64_MAX)break;
          const auto found=at.find({first.allocation,static_cast<std::uint64_t>(offset)});
          if(found==at.end() || used[found->second] || tile[found->second]!=tile[i] ||
             !sameRow(families[found->second],first))break;
          auto member=addOuterBinder(graph,families[found->second].body,shifted_roots);
          extendForms(graph,affine);
          if(!shifted(graph,*generalized,member,members.size(),affine,true))break;
          members.push_back(found->second);
        }
        if(members.size()>best.size()){best=std::move(members);best_body=*generalized;best_stride=stride;}
      }
    }
    if(best.size()>=kTensorMinRows) {
      for(const auto k:best)used[k]=true;
      auto combined=first;combined.shape={best.size(),first.shape[0]};
      combined.strides_bytes={best_stride,first.element_bytes};
      combined.body=*best_body;graph.validateRoots(std::span(&combined.body,1),2);
      result.push_back(std::move(combined));continue;
    }
    used[i]=true;result.push_back(first);
  }
  return result;
}
// Runs of at least kTensorMinRows one-element families of one allocation and element width at one
// constant stride larger than the element become one one-dimensional family with that stride (one
// scalar per row of a [row, head] tensor whose other heads are stored elsewhere or computed
// differently). The first two members give the generalized body (generalizeSingletons); every
// further member is checked as that body at its index (shifted), so a member that differs ends
// the run. Families between the members are untouched; an output at a symbolic position keeps
// its own family. Partners: the next kTensorPartners one-element families of the allocation;
// equal shape signatures and the first kTensorMinRows offsets are looked up before any body is
// built. O(F log F) lookups, plus the bodies of the runs compared once per member.
std::vector<OutputFamily> stridedRuns(Graph& graph,const std::vector<OutputFamily>& families,Forms& affine) {
  std::vector<OutputFamily> result;
  std::map<std::pair<std::string,std::uint64_t>,std::size_t> at;
  for(std::size_t i=0;i<families.size();++i)at.emplace(std::pair{families[i].allocation,families[i].offset},i);
  std::vector<bool> used(families.size());
  const auto single=[&](const OutputFamily& f) {
    if(!flat(f) || f.shape[0]!=1)return false;
    const auto& node=graph.node(f.body);
    return !(node.kind==Kind::Operation && node.name==kPlacementOperation);
  };
  ShapeSignatures shapes;
  for(std::size_t i=0;i<families.size();++i) {
    if(used[i])continue;
    const auto& first=families[i];
    std::vector<std::size_t> best{i};std::optional<Ref> best_body;std::uint64_t best_stride=0;
    if(single(first)) {
      const auto signature=shapes.of(graph,affine,first.body);
      const auto member=[&](std::size_t k) {
        const auto& f=families[k];
        return !used[k] && single(f) && f.element_bytes==first.element_bytes &&
            shapes.of(graph,affine,f.body)==signature;
      };
      std::size_t tried=0;
      for(auto j=i+1;j<families.size() && tried<kTensorPartners;++j) {
        const auto& next=families[j];
        if(next.allocation!=first.allocation)break;
        if(used[j] || !single(next) || next.element_bytes!=first.element_bytes ||
           next.offset-first.offset<=first.element_bytes)continue;
        ++tried;
        if(!member(j))continue;
        const auto stride=next.offset-first.offset;
        const auto find=[&](std::size_t k) {
          const auto offset=static_cast<unsigned __int128>(first.offset)+static_cast<unsigned __int128>(k)*stride;
          return offset>UINT64_MAX?at.end():at.find({first.allocation,static_cast<std::uint64_t>(offset)});
        };
        bool reaches=true;
        for(std::size_t k=2;k<kTensorMinRows && reaches;++k) {
          const auto found=find(k);
          reaches=found!=at.end() && member(found->second);
        }
        if(!reaches)continue;
        auto generalized=generalizeSingletons(graph,first.body,next.body,affine);
        extendForms(graph,affine);
        if(!generalized)continue;
        std::vector<std::size_t> members{i,j};
        for(;;) {
          const auto found=find(members.size());
          if(found==at.end() || !member(found->second) ||
             !shifted(graph,*generalized,families[found->second].body,members.size(),affine,true))break;
          members.push_back(found->second);
        }
        if(members.size()>best.size()){best=std::move(members);best_body=*generalized;best_stride=stride;}
      }
    }
    if(best.size()>=kTensorMinRows) {
      for(const auto k:best)used[k]=true;
      auto combined=first;combined.shape={best.size()};combined.strides_bytes={best_stride};
      combined.body=*best_body;graph.validateRoots(std::span(&combined.body,1),1);
      result.push_back(std::move(combined));continue;
    }
    used[i]=true;result.push_back(first);
  }
  return result;
}
}
// Rows that continue an existing strided-row family: a one-dimensional family of the row length at
// the family's next row offset joins it when its body is the family's body at that row index
// (shifted, the check tensorizeRows makes for every member). tensorizeRows only builds a family
// from at least kTensorMinRows single rows; a lowering that merges its pending outputs again and
// again (class mode under node pressure) completes rows a few at a time, and without this each
// later row would stay a family of its own beside the family it continues. Families are in
// (allocation, offset) order, so a row follows its family. O(F log F) lookups, plus one shifted
// copy and one comparison per row tried; a row that differs ends the extension.
std::vector<OutputFamily> extendRowFamilies(Graph& graph,const std::vector<OutputFamily>& families,Forms& affine) {
  std::map<std::pair<std::string,std::uint64_t>,std::size_t> at;
  for(std::size_t i=0;i<families.size();++i)at.emplace(std::pair{families[i].allocation,families[i].offset},i);
  const auto placed=[&](const OutputFamily& f) {
    const auto& node=graph.node(f.body);
    return node.kind==Kind::Operation && node.name==kPlacementOperation;
  };
  std::vector<bool> used(families.size());
  std::map<Id,Ref> shifted_roots;
  std::vector<OutputFamily> result;
  for(std::size_t i=0;i<families.size();++i) {
    if(used[i])continue;
    used[i]=true;
    auto tensor=families[i];
    const bool rows=tensor.element_bytes && tensor.shape.size()==2 && tensor.strides_bytes.size()==2 &&
        tensor.strides_bytes[1]==tensor.element_bytes && tensor.shape[0] && tensor.shape[1]>1 && !placed(tensor);
    while(rows) {
      const auto offset=static_cast<unsigned __int128>(tensor.offset)+
          static_cast<unsigned __int128>(tensor.shape[0])*tensor.strides_bytes[0];
      if(offset>UINT64_MAX)break;
      const auto found=at.find({tensor.allocation,static_cast<std::uint64_t>(offset)});
      if(found==at.end() || used[found->second])break;
      const auto& row=families[found->second];
      if(!flat(row) || placed(row) || row.element_bytes!=tensor.element_bytes || row.shape[0]!=tensor.shape[1])break;
      const auto member=addOuterBinder(graph,row.body,shifted_roots);
      extendForms(graph,affine);
      if(!shifted(graph,tensor.body,member,tensor.shape[0],affine,true))break;
      used[found->second]=true;++tensor.shape[0];
    }
    result.push_back(std::move(tensor));
  }
  return result;
}
std::size_t mergeOutputFamilies(Graph& graph,std::vector<OutputFamily>& families,bool tensor_families,bool strided_rows,
                                bool strided_runs) {
  Forms affine;
  extendForms(graph,affine);
  const auto before=families.size();
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
  if(strided_runs)merged=stridedRuns(graph,merged,affine);
  if(strided_runs && strided_rows)merged=extendRowFamilies(graph,merged,affine);
  if(tensor_families)merged=tensorizeContiguous(graph,merged,affine);
  else if(strided_rows)merged=tensorizeRows(graph,merged,affine);
  families=std::move(merged);
  return before-families.size();
}
std::size_t consolidateOutputs(Artifact& artifact,bool tensor_families,
                               std::span<std::optional<Id>> weak_nodes,bool strided_rows,
                               const Graph::RemapObserver& observer) {
  auto& graph=artifact.graph;
  std::map<std::uint32_t,std::vector<Ref>> validation_roots;
  for(const auto& output:artifact.outputs)validation_roots[0].push_back(output.value);
  for(const auto& f:artifact.output_families)validation_roots[f.shape.size()].push_back(f.body);
  for(const auto& [depth,roots]:validation_roots)graph.validateRoots(roots,depth);
  const auto removed=mergeOutputFamilies(graph,artifact.output_families,tensor_families,strided_rows);
  std::vector<Ref> roots;
  for(const auto& output:artifact.outputs)roots.push_back(output.value);
  for(const auto& family:artifact.output_families)roots.push_back(family.body);
  graph.collect(roots,weak_nodes,observer);
  std::size_t i=0;
  for(auto& output:artifact.outputs)output.value=roots[i++];
  for(auto& family:artifact.output_families)family.body=roots[i++];
  return removed;
}
}
