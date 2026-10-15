#include "sentryward/equivalence/fold_rewriting.h"
#include "sentryward/equivalence/conversion_uses.h"
#include "sentryward/algorithm/diagnostics/compression_audit.h"
#include <algorithm>
#include <charconv>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_set>

namespace sentryward::equivalence {
namespace {
namespace s=scalar_fold;
bool floating(const std::string& t){return t=="real" || t=="f32" || t=="f64" || t=="f16" || t=="bf16";}
bool idealAttributes(const std::string& text) {
  std::istringstream in(text);std::string word;
  while(in>>word)if(word!="ideal_arithmetic" && word!="flush_to_zero" &&
      word!="rounding=nearest-even")return false;
  return true;
}
bool plain(s::Ref r){return r.conversion.empty() && r.pointer_provenance.empty();}
void validate(const s::Artifact& input) {
  std::map<unsigned,std::vector<s::Ref>> roots;
  for(const auto& out:input.outputs)roots[0].push_back(out.value);
  for(const auto& family:input.output_families)roots[family.shape.size()].push_back(family.body);
  for(const auto& [rank,values]:roots)input.graph.validateRoots(values,rank);
}
bool hasCandidate(const s::Artifact& input,std::chrono::steady_clock::time_point deadline) {
  const auto& graph=input.graph;
  std::vector<unsigned char> active(graph.size()),folds(graph.size());
  const auto activate=[&](const s::Ref& ref,bool root=false) {
    if(ref.pointer_provenance.empty() && (root || ref.conversion.empty() ||
       idealNumericConversion(graph.node(ref).type,graph.type(ref),ref.conversion)))active[ref.node]=1;
  };
  for(const auto& out:input.outputs)activate(out.value,true);
  for(const auto& family:input.output_families)activate(family.body,true);
  const auto check=[&](std::size_t i) {
    if((i&255)==0 && std::chrono::steady_clock::now()>=deadline)
      throw std::invalid_argument("fold_rewrite_time_budget");
  };
  for(std::size_t next=graph.size();next>0;--next) {
    const auto id=next-1;check(id);const auto& n=graph.node(id);
    if(active[id] && floating(n.type))for(const auto& operand:n.operands)activate(operand);
  }
  for(s::Id id=0;id<graph.size();++id) {
    check(id);const auto& n=graph.node(id);
    // A periodic predicate may live under an exact integer Select whose
    // result is converted to a numerical value. Candidate discovery is only
    // a hint; copy() still requires an ideal Sum consumer before partitioning.
    if(n.kind==s::Kind::Operation && n.name=="Remainder" && n.type=="u64" &&
        n.attributes.empty() && n.operands.size()==2 && plain(n.operands[0]) && plain(n.operands[1]) &&
        graph.node(n.operands[0]).kind==s::Kind::Index &&
        graph.node(n.operands[1]).kind==s::Kind::Constant)return true;
    if(n.fold && n.fold->reducer==s::Reducer::Sum && n.fold->count && floating(n.type))folds[id]=1;
    else if(n.kind==s::Kind::Operation && n.name=="Add" && n.operands.size()==2 &&
            floating(n.type) && idealAttributes(n.attributes)) {
      unsigned terms=0;
      for(const auto& operand:n.operands)
        if(plain(operand) && graph.type(operand)==n.type)terms+=folds[operand.node];
      folds[id]=static_cast<unsigned char>(std::min(terms,2u));
      if(active[id] && folds[id]>=2)return true;
    }
  }
  return false;
}
struct Rewriter {
  s::Graph graph;const s::Graph& source;
  // An admitted input DAG is not a rewrite allocation. The owning path keeps
  // its immutable nodes and charges the construction budget only for additions.
  // The parser's input cap and the e-graph's node cap remain independent.
  std::size_t retained=0;
  std::size_t limit,work=0;std::chrono::steady_clock::time_point deadline;
  FoldRewriteStatistics stats;
  std::map<std::tuple<s::Id,bool,bool>,s::Ref> copied;
  std::map<std::tuple<s::Id,unsigned,std::uint64_t>,std::optional<s::Ref>> shifted;
  std::vector<unsigned char> source_uses,fold_terms;
  std::unordered_set<s::Id> shared_sums;
  std::vector<std::map<unsigned,std::uint64_t>> periodic_hints;
  std::map<std::tuple<s::Id,unsigned,std::uint64_t,std::uint64_t>,s::Ref> specialized;
  Rewriter(const s::Graph& input,std::size_t budget,std::chrono::steady_clock::time_point until)
      :source(input),limit(budget),deadline(until) { countUses(); }
  Rewriter(s::Graph&& input,std::size_t budget,std::chrono::steady_clock::time_point until)
      :graph(std::move(input)),source(graph),retained(graph.size()),limit(budget),deadline(until) { countUses(); }
  void addUse(s::Ref r) { if(source_uses[r.node]<2)++source_uses[r.node]; }
  void countUses() {
    source_uses.resize(source.size());
    for(const auto& node:source.nodes())for(const auto& operand:node.operands)addUse(operand);
  }
  void check() {
    if(++work>limit*64 || graph.size()-retained>limit)throw std::invalid_argument("fold_rewrite_node_budget");
    if((work&255)==0 && std::chrono::steady_clock::now()>=deadline)
      throw std::invalid_argument("fold_rewrite_time_budget");
  }
  std::optional<std::uint64_t> number(s::Ref r) const {
    const auto& n=graph.node(r);if(!plain(r) || n.kind!=s::Kind::Constant || n.type!="u64")return {};
    std::uint64_t value=0;auto p=std::from_chars(n.name.data(),n.name.data()+n.name.size(),value);
    if(p.ec!=std::errc{} || p.ptr!=n.name.data()+n.name.size())return {};return value;
  }
  s::Ref use(s::Ref value,const s::Ref& old) {
    if(!old.conversion.empty()) {
      if(!value.conversion.empty())value.conversion+="; ";
      value.conversion+=old.conversion;value.converted_type=old.converted_type;
    }
    value.pointer_provenance=old.pointer_provenance;return value;
  }
  s::Ref integer(const std::string& op,const std::vector<s::Ref>& args) {
    using Key=std::tuple<s::Id,std::string,std::string,std::string>;
    std::map<Key,std::pair<s::Ref,std::uint64_t>> terms;
    std::vector<std::pair<s::Ref,std::uint64_t>> pending;
    if(op=="Add")for(auto a:args)pending.emplace_back(a,1);
    else if(auto n=number(args[0]))pending.emplace_back(args[1],*n);
    else if(auto n=number(args[1]))pending.emplace_back(args[0],*n);
    else return graph.operation(op,"u64",args);
    std::uint64_t constant=0;
    while(!pending.empty()) {
      check();auto [a,coefficient]=pending.back();pending.pop_back();
      if(!coefficient)continue;
      if(auto n=number(a)){constant+=*n*coefficient;continue;}
      const auto n=graph.node(a);
      if(plain(a) && n.type=="u64" && n.kind==s::Kind::Operation && n.attributes.empty() && n.operands.size()==2) {
        if(n.name=="Add") {for(auto b:n.operands)pending.emplace_back(b,coefficient);continue;}
        if(n.name=="Multiply") {
          if(auto c=number(n.operands[0])){pending.emplace_back(n.operands[1],coefficient**c);continue;}
          if(auto c=number(n.operands[1])){pending.emplace_back(n.operands[0],coefficient**c);continue;}
        }
      }
      auto key=Key{a.node,a.conversion,a.converted_type,a.pointer_provenance};
      auto& term=terms[key];term.first=a;term.second+=coefficient;
    }
    std::optional<s::Ref> value;
    if(constant)value=graph.constant("u64",std::to_string(constant));
    for(const auto& [key,term]:terms) {
      (void)key;if(!term.second)continue;auto ref=term.first;
      if(term.second!=1)ref=graph.operation("Multiply","u64",{ref,graph.constant("u64",std::to_string(term.second))});
      value=value?graph.operation("Add","u64",{*value,ref}):ref;
    }
    return value.value_or(graph.constant("u64","0"));
  }
  s::Ref create(const s::Node& n,std::vector<s::Ref> args) {
    switch(n.kind) {
      case s::Kind::Constant:return graph.constant(n.type,n.name);
      case s::Kind::Parameter:return graph.parameter(n.type,n.name);
      case s::Kind::Index:return graph.index(n.index_level);
      case s::Kind::InitialCell:return graph.initialCell(n.name,args.at(0),n.type);
      case s::Kind::Fold:return graph.fold(n.fold->reducer,args[0],args[1],n.fold->binder_level,n.fold->count,n.type);
      case s::Kind::Operation:
        if(n.type=="u64" && n.attributes.empty() && args.size()==2 && (n.name=="Add" || n.name=="Multiply"))return integer(n.name,args);
        return graph.operation(n.name,n.type,args,n.attributes);
    }
    throw std::invalid_argument("unknown_fold_rewrite_node");
  }
  void periodicHints() {
    // Bounded metadata per immutable DAG node; no repeated body discovery.
    while(periodic_hints.size()<graph.size()) {
      check();const auto n=graph.node(periodic_hints.size());std::map<unsigned,std::uint64_t> hints;
      for(const auto& a:n.operands)for(auto [axis,period]:periodic_hints[a.node]) {
        if(n.fold && axis==n.fold->binder_level)continue;
        auto [it,fresh]=hints.emplace(axis,period);
        if(!fresh)it->second=(!it->second || !period)?0:std::lcm(it->second,period);
        if(it->second>8)it->second=0;
      }
      if(n.kind==s::Kind::Operation && n.type=="pred" && n.name=="Equal" &&
          n.attributes.empty() && n.operands.size()==2)for(unsigned side=0;side<2;++side) {
        const auto& r=graph.node(n.operands[side]);
        if(!plain(n.operands[side]) || !number(n.operands[1-side]) ||
            r.kind!=s::Kind::Operation || r.name!="Remainder" || r.type!="u64" ||
            !r.attributes.empty() || r.operands.size()!=2 || !plain(r.operands[0]))continue;
        const auto& index=graph.node(r.operands[0]);auto p=number(r.operands[1]);
        if(index.kind==s::Kind::Index && p && *p>=2 && *p<=8)hints[index.index_level]=*p;
      }
      if(hints.size()>8)hints.clear(); // No hint means keep the original Fold.
      periodic_hints.push_back(std::move(hints));
    }
  }
  s::Ref specialize(s::Ref r,unsigned axis,std::uint64_t period,std::uint64_t residue) {
    check();const auto key=std::tuple{r.node,axis,period,residue};
    if(auto it=specialized.find(key);it!=specialized.end())return use(it->second,r);
    const auto n=graph.node(r);s::Ref result;
    if(n.kind==s::Kind::Index && n.index_level==axis)
      result=integer("Add",{integer("Multiply",{graph.index(axis),graph.constant("u64",std::to_string(period))}),
          graph.constant("u64",std::to_string(residue))});
    else if(n.kind==s::Kind::Operation && n.type=="u64" && n.attributes.empty() &&
        (n.name=="Divide" || n.name=="Remainder") && n.operands.size()==2 &&
        plain(n.operands[0]) && graph.node(n.operands[0]).kind==s::Kind::Index &&
        graph.node(n.operands[0]).index_level==axis && number(n.operands[1])==period)
      result=n.name=="Divide"?graph.index(axis):graph.constant("u64",std::to_string(residue));
    else {
      std::vector<s::Ref> args;
      for(std::size_t i=0;i<n.operands.size();++i)
        args.push_back(n.fold && i==1 && n.fold->binder_level==axis?n.operands[i]:
            specialize(n.operands[i],axis,period,residue));
      if(n.kind==s::Kind::Operation && n.type=="pred" && n.name=="Equal" &&
          n.attributes.empty() && args.size()==2 && number(args[0]) && number(args[1]))
        result=graph.constant("pred",*number(args[0])==*number(args[1])?"1":"0");
      else if(n.kind==s::Kind::Operation && n.name=="Select" && n.attributes.empty() && args.size()==3 &&
          plain(args[0]) && graph.node(args[0]).kind==s::Kind::Constant && graph.type(args[0])=="pred" &&
          (graph.node(args[0]).name=="0" || graph.node(args[0]).name=="1"))
        result=args[graph.node(args[0]).name=="1"?1:2];
      else result=create(n,std::move(args));
    }
    specialized.emplace(key,result);return use(result,r);
  }
  s::Ref partitionPeriodic(s::Ref r) {
    const auto n=graph.node(r);
    if(!plain(r) || !n.fold || n.fold->reducer!=s::Reducer::Sum || !floating(n.type) ||
        !n.attributes.empty() || !n.fold->count)return r;
    periodicHints();const auto& hints=periodic_hints[n.operands[1].node];
    auto it=hints.find(n.fold->binder_level);
    if(it==hints.end() || !it->second || it->second>8)return r;
    const auto period=it->second;auto result=n.operands[0];
    // At most eight congruence classes, not count iterations. The seed is
    // retained ONCE; an odd tail is included by its exact per-residue extent.
    // period*j+residue < count proves that the substituted Index cannot wrap.
    for(std::uint64_t residue=0;residue<period && residue<n.fold->count;++residue) {
      auto body=specialize(n.operands[1],n.fold->binder_level,period,residue);
      result=graph.fold(s::Reducer::Sum,result,body,n.fold->binder_level,
          1+(n.fold->count-1-residue)/period,n.type);
    }
    return result;
  }
  std::optional<s::Ref> shift(s::Ref r,unsigned level,std::uint64_t amount) {
    check();auto key=std::tuple{r.node,level,amount};
    if(auto it=shifted.find(key);it!=shifted.end())return it->second?std::optional{use(*it->second,r)}:std::nullopt;
    const auto n=graph.node(r);std::optional<s::Ref> result;
    if(n.kind==s::Kind::Fold) {shifted[key]={};return {};}
    if(n.kind==s::Kind::Index && n.index_level==level)
      result=integer("Add",{graph.index(level),graph.constant("u64",std::to_string(amount))});
    else {
      std::vector<s::Ref> args;
      for(auto a:n.operands) {
        auto next=shift(a,level,amount);if(!next){shifted[key]={};return {};}
        args.push_back(*next);
      }
      result=create(n,args);
    }
    shifted[key]=result;return use(*result,r);
  }
  bool sum(s::Ref r) const {
    const auto& n=graph.node(r);
    return plain(r) && floating(n.type) && n.kind==s::Kind::Operation && n.name=="Add" &&
        n.operands.size()==2 && idealAttributes(n.attributes);
  }
  std::optional<s::Ref> join(s::Ref a,s::Ref b) {
    check();const auto x=graph.node(a),y=graph.node(b);
    if(!plain(a) || !plain(b) || !x.fold || !y.fold || !floating(x.type) || x.type!=y.type ||
        x.fold->reducer!=s::Reducer::Sum || y.fold->reducer!=s::Reducer::Sum ||
        x.fold->binder_level!=y.fold->binder_level || !x.fold->count || !y.fold->count ||
        x.fold->count>UINT64_MAX-y.fold->count)return {};
    ++stats.attempts;
    auto next=shift(x.operands[1],x.fold->binder_level,x.fold->count);
    if(!next || *next!=y.operands[1])return {};
    const auto zero=[&](s::Ref r){const auto& n=graph.node(r);return plain(r) && n.kind==s::Kind::Constant && n.name=="0";};
    auto initial=x.operands[0];
    if(zero(initial))initial=y.operands[0];
    else if(!zero(y.operands[0]))initial=graph.operation("Add",x.type,{initial,y.operands[0]});
    ++stats.merged;
    return graph.fold(s::Reducer::Sum,initial,x.operands[1],x.fold->binder_level,x.fold->count+y.fold->count,x.type);
  }
  s::Ref coalesce(s::Ref root) {
    // Summaries are appended once per immutable graph node. Do not walk an
    // Add ancestry merely to discover that it contains fewer than two Folds.
    while(fold_terms.size()<graph.size()) {
      check();const auto& n=graph.node(fold_terms.size());unsigned count=0;
      if(n.fold && n.fold->reducer==s::Reducer::Sum && n.fold->count && floating(n.type))count=1;
      else if(n.kind==s::Kind::Operation && n.name=="Add" && floating(n.type) && idealAttributes(n.attributes))
        for(const auto& operand:n.operands)
          if(plain(operand) && graph.type(operand)==n.type)count+=fold_terms[operand.node];
      fold_terms.push_back(static_cast<unsigned char>(std::min(count,2u)));
    }
    if(fold_terms[root.node]<2)return root;
    const auto type=graph.type(root);std::vector<s::Ref> pending{root},terms;
    while(!pending.empty()) {
      check();auto a=pending.back();pending.pop_back();
      if(sum(a) && graph.type(a)==type && (a.node==root.node || !shared_sums.contains(a.node))) {
        const auto n=graph.node(a);pending.push_back(n.operands[1]);pending.push_back(n.operands[0]);
      } else terms.push_back(a);
    }
    bool changed=false,again=true;
    while(again) {
      again=false;
      for(std::size_t i=0;i<terms.size() && !again;++i)for(std::size_t j=i+1;j<terms.size();++j) {
        if(!graph.node(terms[i]).fold || !graph.node(terms[j]).fold)continue;
        auto merged=join(terms[i],terms[j]);if(!merged)merged=join(terms[j],terms[i]);
        if(merged) {terms[i]=*merged;terms.erase(terms.begin()+j);again=changed=true;break;}
      }
    }
    if(!changed)return root;
    auto result=terms.at(0);for(std::size_t i=1;i<terms.size();++i)result=graph.operation("Add",type,{result,terms[i]});
    return result;
  }
  s::Ref copy(s::Ref r,bool arithmetic,bool output_root=false,bool defer_sum=false) {
    check();
    if(!r.pointer_provenance.empty() || (!output_root && !r.conversion.empty() &&
        !idealNumericConversion(source.node(r).type,source.type(r),r.conversion)))arithmetic=false;
    auto key=std::tuple{r.node,arithmetic,defer_sum};s::Ref result;
    if(auto it=copied.find(key);it!=copied.end())result=it->second;
    else {
      const auto n=source.node(r);std::vector<s::Ref> args;
      const bool sum_region=arithmetic && n.kind==s::Kind::Operation && n.name=="Add" &&
          n.operands.size()==2 && floating(n.type) && idealAttributes(n.attributes);
      for(auto a:n.operands) {
        const auto& child=source.node(a);
        // Visit each exclusive Add region once, at its root, instead of
        // rescanning all prefixes. Shared sums remain DAG atoms at consumers.
        const bool defer=sum_region && source_uses[a.node]==1 && plain(a) && child.type==n.type &&
            child.kind==s::Kind::Operation && child.name=="Add" && child.operands.size()==2 && idealAttributes(child.attributes);
        args.push_back(copy(a,arithmetic && floating(n.type),false,defer));
      }
      result=create(n,args);
      if(arithmetic && n.fold)result=partitionPeriodic(result);
      if(arithmetic && !defer_sum && sum(result))result=coalesce(result);
      if(source_uses[r.node]>1 && sum(result))shared_sums.insert(result.node);
      copied[key]=result;
    }
    return use(result,r);
  }
};
scalar_fold::Artifact finish(Rewriter& r,scalar_fold::Artifact out,FoldRewriteStatistics* statistics) {
  for(const auto& x:out.outputs)r.addUse(x.value);
  for(const auto& x:out.output_families)r.addUse(x.body);
  for(auto& x:out.outputs)x.value=r.copy(x.value,true,true);
  for(auto& x:out.output_families)x.body=r.copy(x.body,true,true);
  r.check();
  SENTRYWARD_AUDIT(audit->event("FOLD_REWRITE_STORAGE","retained_and_constructed",r.retained,r.graph.size()-r.retained,r.work));
  // Collection is unnecessary for an unchanged owned graph; importantly, it
  // would invalidate every existing handle for an identity transformation.
  if(r.graph.size()!=r.retained) {
    std::vector<scalar_fold::Ref> roots;
    for(const auto& x:out.outputs)roots.push_back(x.value);
    for(const auto& x:out.output_families)roots.push_back(x.body);
    r.graph.collect(roots);std::size_t i=0;
    for(auto& x:out.outputs)x.value=roots[i++];
    for(auto& x:out.output_families)x.body=roots[i++];
  }
  if(statistics)*statistics=r.stats;out.graph=std::move(r.graph);return out;
}
}
scalar_fold::Artifact mergeAdjacentFolds(const scalar_fold::Artifact& input,std::size_t max_nodes,
    std::chrono::steady_clock::time_point deadline,FoldRewriteStatistics* statistics) {
  validate(input);
  Rewriter r(input.graph,max_nodes,deadline);
  scalar_fold::Artifact out;out.outputs=input.outputs;out.output_families=input.output_families;
  return finish(r,std::move(out),statistics);
}
scalar_fold::Artifact mergeAdjacentFolds(scalar_fold::Artifact&& input,std::size_t max_nodes,
    std::chrono::steady_clock::time_point deadline,FoldRewriteStatistics* statistics) {
  SENTRYWARD_AUDIT_STAGE("equivalence.fold_rewriting");
  validate(input);
  if(!hasCandidate(input,deadline)) {
    SENTRYWARD_AUDIT(audit->event("NORMALIZATION_IDENTITY","no_adjacent_fold_candidates",input.graph.size(),0));
    if(statistics)*statistics={};return std::move(input);
  }
  Rewriter r(std::move(input.graph),max_nodes,deadline);
  return finish(r,std::move(input),statistics);
}
}
