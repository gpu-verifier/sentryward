#pragma once
#include "egraph_engine.hpp"

namespace sentryward::equivalence {
namespace {

inline auto Engine::emitCommonAddFactor(Class id,const Key& k,Class factor,
                                         Class left,Class right) -> std::optional<Class> {
  // Both callers certify this SAME local finite ideal identity. Return its
  // compact residue so a frozen additive DAG can carry the proof upward
  // without awaiting another global rewrite epoch. Never distribute back.
  if(factored_sums_.contains(id))return {};
  Key residue;residue.kind=static_cast<int>(s::Kind::Operation);residue.type=k.type;
  residue.name="Add";residue.attributes="ideal_arithmetic";residue.children={left,right};
  const auto body=intern(std::move(residue));
  Key product;product.kind=static_cast<int>(s::Kind::Operation);product.type=k.type;
  product.name="Multiply";product.attributes="ideal_arithmetic";product.children={factor,body};
  rewrite(id,intern(std::move(product)));factored_sums_.insert(id);return body;
}

inline auto Engine::factorAdditiveForests(const std::vector<Key>& snapshot,
                            const std::vector<bool>& nonrecursive) -> void {
  namespace l=bounded_linear_reduction;
  // Optional bounded scheduling of the existing common-factor identity.
  // Only selected Add edges are followed; Fold, products and factor/residue
  // values remain opaque. Every stored class is summarized once per epoch.
  constexpr std::size_t max_records=65536,max_product_heads=4,max_factors=8;
  std::size_t remaining=2000000;
  const auto charge=[&](std::size_t amount=1) {
    poll();if(amount>remaining)return false;remaining-=amount;
    if(options_.max_work) {
      if(amount>options_.max_work-work_)throw Limit{"work_budget"};
      work_+=amount;
    }
    return true;
  };
  const auto n=snapshot.size();if(n>1000000)return;
  std::vector<Class> owners(n);
  std::vector<bool> unit(n);
  for(Class id=0;id<n;++id) {
    if(!charge())return;
    owners[id]=find(id);if(owners[id]>=n)return; // Appended owner: Unknown for this frozen factory.
    const auto& k=snapshot[id];
    if(k.kind==static_cast<int>(s::Kind::Constant) && k.type=="real" && k.name=="1" &&
        k.attributes.empty() && k.provenance.empty())unit[owners[id]]=true;
  }
  l::RankedDagStatistics statistics;
  auto ranked=l::rankedDagHeads(owners,[&](Class id)->std::span<const Class> {
    return snapshot[id].children;
  },n,remaining,&statistics);
  if(!charge(statistics.work) || !ranked)return;
  struct Descriptor {Class factor,residue;};
  struct Record {
    Class sum=0;bool has_sum=false,unknown=false;unsigned char state=0;
    std::vector<Class> products;
    std::vector<Descriptor> factors;
  };
  std::unordered_map<Class,Record> records;
  const auto eligible=[&](const Key& k) {
    return k.type=="real" && idealOperation(k) && k.provenance.empty() &&
      k.children.size()==2 && (k.name=="Add" || k.name=="Multiply") &&
      snapshot[k.children[0]].type==k.type && snapshot[k.children[1]].type==k.type &&
      finite_[owners[k.children[0]]] && finite_[owners[k.children[1]]];
  };
  for(Class id=0;id<n;++id) {
    if(!charge())return;const auto& k=snapshot[id];
    if(!nonrecursive[id] || !eligible(k))continue;
    const auto cls=owners[id];
    if(!records.contains(cls) && records.size()==max_records)return;
    auto& record=records[cls];
    if(k.name=="Add" && (*ranked)[id] && !record.has_sum) {
      record.sum=id;record.has_sum=true;
    }
    if(k.name=="Multiply") {
      // Late proven product aliases need not precede their owner rank.
      // They are inspected ONLY as local opaque identities, never followed
      // recursively; the SCC mask and explicit self-use guard reject cycles.
      if(owners[k.children[0]]==cls || owners[k.children[1]]==cls)continue;
      if(record.products.size()==max_product_heads)record.unknown=true;
      else record.products.push_back(id);
    }
  }
  const auto insert=[&](Record& r,Descriptor d) {
    auto at=std::lower_bound(r.factors.begin(),r.factors.end(),d.factor,
        [](const Descriptor& a,Class b){return a.factor<b;});
    if(at!=r.factors.end() && at->factor==d.factor)return true;
    if(r.factors.size()==max_factors)return false;
    r.factors.insert(at,d);return true;
  };
  struct Task {Class cls;bool ready;};
  for(auto& [root,unused]:records) {
    (void)unused;if(records.at(root).state==2)continue;
    std::vector<Task> pending{{root,false}};
    while(!pending.empty()) {
      if(!charge())return;
      auto t=pending.back();pending.pop_back();auto& r=records.at(t.cls);
      if(r.state==2)continue;
      if(!t.ready) {
        if(r.state==1){r.unknown=true;r.state=2;continue;}
        r.state=1;pending.push_back({t.cls,true});
        if(r.has_sum)for(auto child:snapshot[r.sum].children) {
          const auto cls=owners[child];auto c=records.find(cls);
          if(c!=records.end() && c->second.state!=2)pending.push_back({cls,false});
        }
        continue;
      }
      r.state=2;if(r.unknown)continue;
      for(auto head:r.products)for(unsigned side=0;side<2;++side) {
        if(!charge())return;const auto& k=snapshot[head];
        const auto factor=owners[k.children[side]];
        if(unit[factor])continue; // Never grow a self-alias by factoring one.
        if(!insert(r,{factor,owners[k.children[1-side]]})){r.unknown=true;break;}
      }
      if(r.unknown || !r.has_sum || factored_sums_.contains(r.sum))continue;
      const auto k=snapshot[r.sum];
      auto a=records.find(owners[k.children[0]]),b=records.find(owners[k.children[1]]);
      if(a==records.end() || b==records.end() || a->second.unknown || b->second.unknown)continue;
      const auto& left=a->second.factors;const auto& right=b->second.factors;
      std::size_t i=0,j=0;
      while(i<left.size() && j<right.size()) {
        if(!charge())return;
        if(left[i].factor<right[j].factor){++i;continue;}
        if(right[j].factor<left[i].factor){++j;continue;}
        const auto factor=left[i].factor;
        auto existing=std::lower_bound(r.factors.begin(),r.factors.end(),factor,
            [](const Descriptor& a,Class f){return a.factor<f;});
        if(existing!=r.factors.end() && existing->factor==factor)break;
        if(r.factors.size()==max_factors){r.unknown=true;break;}
        // The two descriptors are complete proven use identities, including
        // type/capture. Their opaque residues may be newly interned IDs;
        // no vector element reference survives the shared emitter's append.
        const auto residue=emitCommonAddFactor(r.sum,k,factor,left[i].residue,right[j].residue);
        if(residue)insert(r,{factor,*residue});
        break; // One common witness, at most two enodes per stored Add head.
      }
    }
  }
}

inline auto Engine::mergePolynomialNormalForms(const std::vector<Key>& snapshot,
                                const std::vector<bool>& nonrecursive) -> void {
  namespace p=polynomial_certificate;
  // Exact bounded certificates for finite ideal-real scalar algebra. Fold,
  // Select and other math stay atomic. Only literal nonzero denominators can
  // be expanded; arbitrary division/cancellation is never licensed here.
  // No distributed expression trees are interned. Memoization is per frozen
  // class/type and epoch; hard limits bound terms, degree and integer bits.
  std::map<std::string,std::vector<Class>> groups;
  const auto n=snapshot.size();std::vector<Class> owners(n);
  for(Class id=0;id<n;++id) {
    poll();
    owners[id]=find(id);
    if(nonrecursive[id] && realType(snapshot[id].type) &&
        snapshot[id].provenance.empty() && finite_[owners[id]])
      groups[snapshot[id].type].push_back(id);
  }
  std::vector<std::pair<Class,Class>> equalities;
  for(const auto& [type,heads]:groups) {
    p::Limits limits;
    if(options_.max_work)limits.max_work=std::min(limits.max_work,options_.max_work-work_);
    p::Builder builder(limits);
    std::size_t charged=0;
    const auto charge=[&] {
      const auto delta=builder.work()-charged;charged=builder.work();
      if(options_.max_work && delta>options_.max_work-work_)throw Limit{"work_budget"};
      if(options_.max_work)work_+=delta;
    };
    std::unordered_map<Class,Class> selected;
    std::unordered_map<Class,std::string> literals;
    for(auto id:heads)if(snapshot[id].kind==static_cast<int>(s::Kind::Constant) && snapshot[id].attributes.empty())
      literals.try_emplace(owners[id],snapshot[id].name);
    const auto eligible=[&,&type=type](const Key& k) {
      if(!idealOperation(k))return false;
      if(!std::all_of(k.children.begin(),k.children.end(),[&](auto c) {
        return snapshot[c].type==type && finite_[owners[c]];
      }))return false;
      if(k.name=="Add" || k.name=="Subtract" || k.name=="Multiply" || k.name=="Negate")return true;
      if(k.name=="Divide")return literals.contains(owners[k.children[1]]);
      if(k.name=="Reciprocal")return literals.contains(owners[k.children[0]]);
      return false;
    };
    for(auto id:heads){poll();if(eligible(snapshot[id]))selected.try_emplace(owners[id],id);}
    std::unordered_map<Class,p::Result> summaries;
    std::unordered_map<Class,unsigned char> state;
    const auto evaluate=[&](const Key& k)->p::Result {
      std::vector<const p::CanonicalPolynomial*> operands;
      for(auto child:k.children) {
        const auto& summary=summaries[owners[child]];if(!summary)return {};
        operands.push_back(&*summary);
      }
      if(k.name=="Reciprocal") {
        auto one=builder.literal("1");if(!one)return {};
        return builder.divideByLiteral(*one,literals.at(owners[k.children[0]]));
      }
      if(k.name=="Divide")return builder.divideByLiteral(*operands[0],literals.at(owners[k.children[1]]));
      auto op=p::Operation::Add;
      if(k.name=="Subtract")op=p::Operation::Subtract;
      if(k.name=="Multiply")op=p::Operation::Multiply;
      if(k.name=="Negate")op=p::Operation::Negate;
      return builder.apply(op,operands);
    };
    struct Task {Class cls;bool ready;};
    for(auto id:heads) {
      if(state[owners[id]]==2)continue;
      std::vector<Task> work{{owners[id],false}};
      while(!work.empty() && !builder.exhausted()) {
        poll();auto [cls,ready]=work.back();work.pop_back();if(state[cls]==2)continue;
        if(auto literal=literals.find(cls);literal!=literals.end()) {
          summaries[cls]=builder.literal(literal->second);state[cls]=2;continue;
        }
        auto head=selected.find(cls);
        if(head==selected.end()){summaries[cls]=builder.atom(cls);state[cls]=2;continue;}
        const auto& k=snapshot[head->second];
        if(ready) {
          summaries[cls]=evaluate(k);
          if(!summaries[cls])summaries[cls]=builder.atom(cls);
          state[cls]=2;continue;
        }
        if(state[cls]==1){summaries[cls]=std::nullopt;state[cls]=2;continue;}
        state[cls]=1;work.push_back({cls,true});
        for(auto child:k.children)if(state[owners[child]]!=2)work.push_back({owners[child],false});
      }
      charge();if(builder.exhausted())break;
    }
    std::map<p::CanonicalPolynomial,Class> signatures;
    for(auto id:heads) {
      poll();
      const auto& k=snapshot[id];p::Result signature;
      const auto owner=owners[id];
      const auto choice=selected.find(owner);
      if((choice!=selected.end() && choice->second==id) ||
          (choice==selected.end() && (!eligible(k) || literals.contains(owner)))) {
        if(auto found=summaries.find(owner);found!=summaries.end())signature=found->second;
      } else if(!builder.exhausted()) {
        if(k.kind==static_cast<int>(s::Kind::Constant) && k.attributes.empty())signature=builder.literal(k.name);
        else if(eligible(k))signature=evaluate(k);
        else signature=builder.atom(owner);
      }
      charge();
      if(!signature)continue;
      auto [entry,inserted]=signatures.emplace(std::move(*signature),id);
      if(!inserted)equalities.emplace_back(id,entry->second);
    }
  }
  for(auto [a,b]:equalities)rewrite(a,b);
}

inline auto Engine::mergeAssociativeNormalForms(const std::vector<Key>& snapshot,
                                 const std::vector<bool>& nonrecursive) -> void {
  // A bounded congruence analysis, not an expansive rewrite. Freeze owners
  // and one nonrecursive decomposition per class/operator for this epoch;
  // only then union equal complete signatures. No binary enodes are emitted.
  // B=256 DISTINCT scalar atoms per summary. Counts preserve shared DAGs
  // such as Add(x,x) without materializing their logical occurrences. A cap
  // or count overflow makes this optional certificate unavailable, not true.
  // Cost O(B*(N+E)+B*N*log N) per epoch, space O(B*N), with no Fold expansion.
  constexpr std::size_t max_atoms=256;
  using Terms=std::vector<std::pair<Class,std::uint64_t>>;
  const auto n=snapshot.size();
  std::vector<Class> owners(n);
  for(Class id=0;id<n;++id)owners[id]=find(id);
  std::map<std::pair<std::string,std::string>,std::vector<Class>> groups;
  for(Class id=0;id<n;++id) {
    const auto& k=snapshot[id];
    if(nonrecursive[id] && arithmetic(k) && k.provenance.empty() &&
        (k.name=="Add" || k.name=="Multiply") &&
        finite_[owners[k.children[0]]] && finite_[owners[k.children[1]]])
      groups[{k.name,k.type}].push_back(id);
  }
  std::vector<std::pair<Class,Class>> equalities;
  const auto merge=[&](const Terms& a,const Terms& b)->std::optional<Terms> {
    Terms result;result.reserve(std::min(max_atoms,a.size()+b.size()));
    std::size_t i=0,j=0;
    while(i<a.size() || j<b.size()) {
      poll();std::pair<Class,std::uint64_t> term;
      if(j==b.size() || (i<a.size() && a[i].first<b[j].first))term=a[i++];
      else if(i==a.size() || b[j].first<a[i].first)term=b[j++];
      else {
        if(b[j].second>UINT64_MAX-a[i].second)return {};
        term={a[i].first,a[i].second+b[j].second};++i;++j;
      }
      if(result.size()==max_atoms)return {};result.push_back(term);
    }
    return result;
  };
  for(const auto& [spec,heads]:groups) {
    std::unordered_map<Class,Class> selected;
    for(auto id:heads)selected.try_emplace(owners[id],id);
    std::unordered_map<Class,std::optional<Terms>> summaries;
    std::unordered_map<Class,unsigned char> state;
    struct Task {Class cls;bool ready;};
    for(auto id:heads) {
      std::vector<Task> work{{owners[id],false}};
      while(!work.empty()) {
        poll();auto [cls,ready]=work.back();work.pop_back();
        if(state[cls]==2)continue;
        auto head=selected.find(cls);
        if(head==selected.end()) {
          summaries[cls]=Terms{{cls,1}};state[cls]=2;continue;
        }
        const auto& k=snapshot[head->second];
        auto a=owners[k.children[0]],b=owners[k.children[1]];
        if(ready) {
          auto& left=summaries[a];auto& right=summaries[b];
          summaries[cls]=left && right?merge(*left,*right):std::nullopt;
          state[cls]=2;continue;
        }
        if(state[cls]==1) {summaries[cls]=std::nullopt;state[cls]=2;continue;}
        state[cls]=1;work.push_back({cls,true});
        if(state[b]!=2)work.push_back({b,false});
        if(state[a]!=2)work.push_back({a,false});
      }
    }
    std::map<Terms,Class> signatures;
    for(auto id:heads) {
      poll();const auto& k=snapshot[id];
      const auto& a=summaries[owners[k.children[0]]];
      const auto& b=summaries[owners[k.children[1]]];
      if(!a || !b)continue;
      auto signature=merge(*a,*b);if(!signature)continue;
      auto [entry,inserted]=signatures.emplace(std::move(*signature),id);
      if(!inserted)equalities.emplace_back(id,entry->second);
    }
  }
  for(auto [a,b]:equalities)rewrite(a,b);
}

inline auto Engine::nonrecursiveHeads(const std::vector<Key>& snapshot) -> std::vector<bool> {
  // Equality creates recursive alternatives such as x=(x+y)-y. They are
  // useful proven unions, not finite trees to unfold repeatedly. Compute
  // class SCCs once per epoch (O(enodes+operand edges)); expansive matching
  // uses only heads which leave their result's SCC. Never discard a union.
  // Rewriters can append and unite new product/residue nodes while this
  // epoch's source snapshot stays frozen. SCC owners and paths therefore
  // range over the CURRENT stored graph, not merely its snapshot prefix.
  const auto n=nodes_.size();
  std::vector<std::vector<Class>> out(n),in(n);
  for(Class id=0;id<n;++id)for(auto child:nodes_[id].children) {
    poll();auto a=find(id),b=find(child);out[a].push_back(b);in[b].push_back(a);
  }
  std::vector<bool> seen(n);std::vector<Class> order;order.reserve(n);
  for(Class root=0;root<n;++root)if(find(root)==root && !seen[root]) {
    seen[root]=true;std::vector<std::pair<Class,std::size_t>> work{{root,0}};
    while(!work.empty()) {
      poll();auto& [id,cursor]=work.back();
      if(cursor==out[id].size()){order.push_back(id);work.pop_back();continue;}
      auto child=out[id][cursor++];if(seen[child])continue;
      seen[child]=true;work.emplace_back(child,0);
    }
  }
  std::vector<Class> component(n,n);
  for(auto cursor=order.rbegin();cursor!=order.rend();++cursor)if(component[*cursor]==n) {
    component[*cursor]=*cursor;std::vector<Class> work{*cursor};
    while(!work.empty()) {
      poll();auto id=work.back();work.pop_back();
      for(auto parent:in[id])if(component[parent]==n){component[parent]=*cursor;work.push_back(parent);}
    }
  }
  std::vector<bool> result(snapshot.size(),true);
  for(Class id=0;id<snapshot.size();++id)for(auto child:snapshot[id].children)
    if(component[find(id)]==component[find(child)]){result[id]=false;break;}
  return result;
}

inline auto Engine::mergeAlphaEquivalentFolds() -> void {
  // One import normalization, before any unions can introduce cyclic
  // e-class alternatives. Private signatures use de Bruijn bound indices;
  // free indices remain absolute. ONLY whole Fold results signed under the
  // empty environment are unioned, never context-dependent bodies/indices.
  struct PairHash {
    std::size_t operator()(const std::pair<Class,Class>& p) const {
      return p.first^(p.second+0x9e3779b9+(p.first<<6)+(p.first>>2));
    }
  };
  struct Scope {Class parent;std::uint32_t level;};
  std::vector<Scope> scopes{{0,0}};
  std::unordered_map<std::pair<Class,Class>,Class,PairHash> scope_ids,values;
  std::map<std::vector<std::pair<std::uint32_t,std::uint32_t>>,Class> context_ids;
  context_ids.emplace(std::vector<std::pair<std::uint32_t,std::uint32_t>>{},0);
  std::unordered_map<Key,Class,Hash> signatures;
  std::vector<std::pair<Class,Class>> folds;
  const auto bound=[&](std::uint32_t level,Class scope)->std::optional<std::uint32_t> {
    std::uint32_t distance=0;
    for(auto current=scope;current;current=scopes[current].parent,++distance) {
      poll();if(scopes[current].level==level)return distance;
    }
    return {};
  };
  const auto context=[&](Class node,Class scope) {
    // Unused enclosing binders cannot multiply the cost of a closed DAG.
    std::vector<std::pair<std::uint32_t,std::uint32_t>> used;
    for(auto level:free_indices_[node])if(auto position=bound(level,scope))used.emplace_back(level,*position);
    auto found=context_ids.find(used);
    if(found==context_ids.end()) {
      if(context_ids.size()>=options_.max_nodes)throw Limit{"alpha_normalization_context_budget"};
      found=context_ids.emplace(std::move(used),context_ids.size()).first;
    }
    return std::pair{node,found->second};
  };
  const auto bodyScope=[&](Class parent,std::uint32_t level) {
    auto key=std::pair{parent,Class(level)};
    if(auto found=scope_ids.find(key);found!=scope_ids.end())return found->second;
    if(scopes.size()>=options_.max_nodes)throw Limit{"alpha_normalization_context_budget"};
    auto id=scopes.size();scopes.push_back({parent,level});scope_ids.emplace(key,id);return id;
  };
  struct Task {Class node,scope;bool ready;};
  for(Class root=0;root<nodes_.size();++root) {
    poll();
    if(nodes_[root].kind!=static_cast<int>(s::Kind::Fold))continue;
    std::vector<Task> work{{root,0,false}};
    while(!work.empty()) {
      poll();const auto task=work.back();work.pop_back();
      const auto ck=context(task.node,task.scope);
      if(values.contains(ck))continue;
      auto k=nodes_[task.node];
      const auto is_fold=k.kind==static_cast<int>(s::Kind::Fold);
      const auto nested=is_fold?bodyScope(task.scope,k.fold->binder_level):task.scope;
      if(!task.ready) {
        work.push_back({task.node,task.scope,true});
        for(std::size_t i=0;i<k.children.size();++i) {
          const auto scope=is_fold && i==1?nested:task.scope;
          if(!values.contains(context(k.children[i],scope)))work.push_back({k.children[i],scope,false});
        }
        continue;
      }
      const bool commutative=(arithmetic(k) && finite_[k.children[0]] && finite_[k.children[1]]) ||
          (integerCommutative(k) && nodes_[k.children[0]].type==k.type && nodes_[k.children[1]].type==k.type);
      for(std::size_t i=0;i<k.children.size();++i)
        k.children[i]=values.at(context(k.children[i],is_fold && i==1?nested:task.scope));
      if(commutative && k.children[1]<k.children[0])std::swap(k.children[0],k.children[1]);
      if(k.kind==static_cast<int>(s::Kind::Index)) {
        if(auto position=bound(k.index_level,task.scope)){k.kind=9;k.index_level=*position;}
        else k.kind=8;
      }
      if(is_fold)k.fold->binder_level=0;
      auto found=signatures.find(k);
      if(found==signatures.end()) {
        if(signatures.size()>=options_.max_nodes)throw Limit{"alpha_normalization_node_budget"};
        found=signatures.emplace(std::move(k),signatures.size()).first;
      }
      if(values.size()>=options_.max_nodes)throw Limit{"alpha_normalization_context_budget"};
      values.emplace(ck,found->second);
    }
    folds.emplace_back(values.at(context(root,0)),root);
  }
  std::unordered_map<Class,Class> first;
  const auto before=unions_;
  for(auto [signature,root]:folds) {
    auto [found,inserted]=first.try_emplace(signature,root);
    if(!inserted)rewrite(root,found->second);
  }
  if(sentryward::debug::environment("SENTRYWARD_EGRAPH_TRACE"))SENTRYWARD_DEBUG_PRINT(egraph_60);
}

} // namespace
} // namespace sentryward::equivalence
