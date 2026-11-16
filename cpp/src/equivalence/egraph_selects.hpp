#pragma once
#include "egraph_engine.hpp"

namespace sentryward::equivalence {
namespace {

inline auto Engine::indexSelectPatterns(const std::vector<Key>& snapshot,const std::vector<bool>& nonrecursive) -> SelectIndex {
  SelectIndex result;
  for(Class id=0;id<snapshot.size();++id) {
    if(!nonrecursive[id])continue;
    poll();const auto& k=snapshot[id];const auto root=find(id);
    if(k.kind==static_cast<int>(s::Kind::Constant) && realType(k.type)) {
      result.constants.insert(root);
      if(k.name=="1" && k.attributes.empty() && k.provenance.empty())result.ones.insert(root);
    }
    if(k.kind==static_cast<int>(s::Kind::Constant) &&
        (k.type=="u64" || k.type=="exact:u64") && k.attributes.empty() &&
        k.provenance.empty() && k.name.size()<=20) {
      std::uint64_t value=0;
      const auto parsed=std::from_chars(k.name.data(),k.name.data()+k.name.size(),value);
      if(parsed.ec==std::errc{} && parsed.ptr==k.name.data()+k.name.size())
        result.u64_constants.try_emplace(root,SelectIndex::U64Constant{k.type,value});
    }
    if(idealSelect(k))result.heads[root].conditions.try_emplace(find(k.children[0]),id);
    if(idealOperation(k) && k.provenance.empty() && k.children.size()==1)
      result.heads[root].unary.try_emplace(k.name,id);
  }
  for(Class id=0;id<snapshot.size();++id) {
    if(!nonrecursive[id])continue;
    poll();const auto& k=snapshot[id];
    if(k.kind==static_cast<int>(s::Kind::Operation) && k.name=="Add" &&
        (k.type=="u64" || k.type=="exact:u64") && k.children.size()==2 &&
        k.attributes.empty() && k.provenance.empty()) {
      const auto a=find(k.children[0]),b=find(k.children[1]);
      if(a<snapshot.size() && b<snapshot.size() && snapshot[a].type==k.type && snapshot[b].type==k.type)
        for(unsigned side=0;side<2;++side) {
          const auto constant=result.u64_constants.find(side?b:a);
          if(constant==result.u64_constants.end() || constant->second.type!=k.type)continue;
          result.u64_add_constants.try_emplace(find(id),
              SelectIndex::U64AddConstant{k.type,id,side?a:b,constant->second.value});
          break; // One acyclic immediate witness per class, never products of alternatives.
        }
    }
    if(!idealOperation(k) || !k.provenance.empty() || k.children.size()!=2 ||
        (k.name!="Add" && k.name!="Multiply" && k.name!="Subtract" && k.name!="Divide"))continue;
    auto& heads=result.heads[find(id)];
    auto a=find(k.children[0]),b=find(k.children[1]);
    if(k.name=="Add" || k.name=="Multiply") {
      heads.factors.try_emplace(std::pair{k.name,a},b);
      heads.factors.try_emplace(std::pair{k.name,b},a);
    }
    if(k.name=="Multiply") {
      // Zero-arm factoring prefers a constant invariant factor. This is a
      // local canonical choice, not a search through a numerical subtree.
      if(!result.constants.contains(a) && result.constants.contains(b))std::swap(a,b);
      if(!heads.product || (!result.constants.contains(heads.product->first) && result.constants.contains(a)))
        heads.product=std::pair{a,b};
    }
    if(k.name=="Subtract" && !heads.difference)heads.difference=std::pair{a,b};
    if(k.name=="Divide" && !heads.quotient)heads.quotient=std::pair{a,b};
    if(k.name=="Add" && !heads.difference)for(unsigned side=0;side<2;++side) {
      const auto& unary=result.get(find(k.children[side])).unary;
      const auto neg=unary.find("Negate");
      if(neg==unary.end())continue;
      heads.difference=std::pair{find(k.children[1-side]),find(snapshot[neg->second].children[0])};
      break;
    }
  }
  return result;
}

inline auto Engine::rewriteExactU64Add(Class id,const Key& k,const std::vector<Key>& snapshot,
                        const SelectIndex& index) -> void {
  if(k.kind!=static_cast<int>(s::Kind::Operation) || k.name!="Add" ||
      (k.type!="u64" && k.type!="exact:u64") || k.children.size()!=2 ||
      !k.attributes.empty() || !k.provenance.empty())return;
  const auto a=find(k.children[0]),b=find(k.children[1]);
  if(a>=snapshot.size() || b>=snapshot.size() || snapshot[a].type!=k.type || snapshot[b].type!=k.type)return;
  const auto literal=[&](Class root)->std::optional<std::uint64_t> {
    const auto found=index.u64_constants.find(root);
    if(found==index.u64_constants.end() || found->second.type!=k.type)return {};
    return found->second.value;
  };
  const auto constant=[&](std::uint64_t value) {
    Key result;result.kind=static_cast<int>(s::Kind::Constant);
    result.type=k.type;result.name=std::to_string(value);return intern(std::move(result));
  };
  const auto sum=[](std::uint64_t x,std::uint64_t y) {
    // Reuse the exact u64 affine ring: overflow is modulo 2^64, not an
    // assumed absence of wrap. Both coefficient maps are empty, O(1).
    return combineAffine("Add",IndexAffine{x,{}},IndexAffine{y,{}})->constant;
  };
  const auto first=literal(a),second=literal(b);
  if(first && second){rewrite(id,constant(sum(*first,*second)));return;}
  for(unsigned side=0;side<2;++side) {
    poll();const auto outer=side?second:first;
    if(!outer)continue;
    const auto inner=side?a:b;
    if(*outer==0){rewrite(id,inner);return;}
    const auto head=index.u64_add_constants.find(inner);
    if(head==index.u64_add_constants.end() || head->second.type!=k.type)continue;
    const auto value=find(head->second.value);
    if(value>=snapshot.size() || snapshot[value].type!=k.type ||
        find(head->second.source)==find(id) || value==find(id))continue;
    const auto folded=sum(*outer,head->second.constant);
    if(!folded){rewrite(id,value);return;}
    Key result;result.kind=static_cast<int>(s::Kind::Operation);
    result.name="Add";result.type=k.type;result.children={value,constant(folded)};
    rewrite(id,intern(std::move(result)));return; // At most one replacement head.
  }
}

inline auto Engine::rewriteKnownPredicateSelect(Class id,const Key& k,
    const std::vector<Key>& snapshot) -> bool {
  if(k.kind!=static_cast<int>(s::Kind::Operation) || k.name!="Select" ||
      k.children.size()!=3 || !k.provenance.empty() ||
      (!k.attributes.empty() && k.attributes!="ideal_arithmetic"))return false;
  const auto p=find(k.children[0]),a=find(k.children[1]),b=find(k.children[2]);
  if(p>=snapshot.size() || a>=snapshot.size() || b>=snapshot.size() ||
      (snapshot[p].type!="pred" && snapshot[p].type!="exact:pred") ||
      snapshot[a].type!=k.type || snapshot[b].type!=k.type)return false;
  // Existing three-valued proof: Unknown is never interpreted as False.
  // Selection is exact for typed storage/integer/opaque arms too. No finite
  // fact, arithmetic interpretation, new head or alternative scan is needed.
  const auto known=predicate_[p];if(!known)return false;
  rewrite(id,known==1?a:b);return true;
}

inline auto Engine::rewriteFiniteCompareSelectExtremum(Class id,const Key& k,
    const std::vector<Key>& snapshot) -> bool {
  if(!idealSelect(k) || k.type!="real" || !k.provenance.empty())return false;
  // Keep this stored predicate head: an earlier local union in this epoch
  // may have made its current owner a constant without removing the witness.
  const auto p=k.children[0],a=find(k.children[1]),b=find(k.children[2]);
  if(p>=snapshot.size())return false;
  const auto& condition=snapshot[p];const auto predicate=comparison(condition);
  if(!predicate || (*predicate!="lt" && *predicate!="le" &&
                    *predicate!="gt" && *predicate!="ge"))return false;
  const auto x=find(condition.children[0]),y=find(condition.children[1]);
  if(x>=snapshot.size() || y>=snapshot.size() || snapshot[x].type!="real" ||
      snapshot[y].type!="real" || !finite_[x] || !finite_[y] ||
      !((a==x && b==y) || (a==y && b==x)))return false;
  // Both finite numerical arms must be exactly the comparands. The shared
  // comparison certificate also admits unordered_true on these finite inputs.
  // Exact/bit/opaque heads and unknown predicates retain their original form.
  const bool true_is_x=a==x,true_is_smaller=*predicate=="lt" || *predicate=="le";
  Key extrema;extrema.kind=static_cast<int>(s::Kind::Operation);extrema.type=k.type;
  extrema.name=true_is_x==true_is_smaller?"Minimum":"Maximum";
  extrema.attributes="ideal_arithmetic";extrema.children={x,y};
  rewrite(id,intern(std::move(extrema)));return true; // At most one new head.
}

inline auto Engine::rewriteSelectPatterns(Class id,const Key& k,const std::vector<Class>& original,
    const std::vector<Key>& snapshot,const SelectIndex& index) -> void {
  const auto make=[&](std::string name,std::vector<Class> children) {
    Key operation;operation.kind=static_cast<int>(s::Kind::Operation);
    operation.type=k.type;operation.name=std::move(name);
    operation.attributes="ideal_arithmetic";operation.children=std::move(children);
    return intern(std::move(operation));
  };
  const auto choose=[&](Class p,Class a,Class b) {
    return make("Select",{find(p),find(a),find(b)});
  };
  const auto zero=[&] {
    Key value;value.kind=static_cast<int>(s::Kind::Constant);value.type=k.type;value.name="0";
    return intern(std::move(value));
  };
  const auto finite=[&](Class a){return finite_[find(a)];};
  const auto nonzero=[&](Class a){return nonzero_[find(a)];};
  if(idealSelect(k)) {
    const auto p=k.children[0],a=k.children[1],b=k.children[2];
    if(a==b){rewrite(id,a);return;}
    for(unsigned arm=0;arm<2;++arm) {
      const auto& heads=index.get(original[arm+1]).conditions;
      if(auto found=heads.find(p);found!=heads.end()) {
        const auto& nested=snapshot[found->second];
        if(nested.type!=k.type)continue;
        auto branches=k.children;branches[arm+1]=find(nested.children[arm+1]);
        rewrite(id,make("Select",std::move(branches)));
      }
    }
    if(factored_selects_.contains(id) || !finite(a) || !finite(b))return;
    // Unary factoring is oriented out of Select, not both ways. Exp2(0)=1
    // supplies the identity arm without an unrestricted inverse rewrite.
    for(const auto* name:{"Exp2","Exp","Negate","Reciprocal"}) {
      std::optional<Class> args[2];
      for(unsigned arm=0;arm<2;++arm) {
        const auto& heads=index.get(original[arm+1]).unary;
        if(auto found=heads.find(name);found!=heads.end()) {
          const auto& unary=snapshot[found->second];
          if(unary.type==k.type)args[arm]=find(unary.children[0]);
        } else if((std::string_view(name)=="Exp2" || std::string_view(name)=="Exp") && index.ones.contains(original[arm+1]))args[arm]=zero();
        else if(std::string_view(name)=="Negate" && zero_[find(original[arm+1])])args[arm]=zero();
      }
      if(!args[0] || !args[1] || !finite(*args[0]) || !finite(*args[1]) ||
          (std::string_view(name)=="Reciprocal" && (!nonzero(*args[0]) || !nonzero(*args[1]))))continue;
      rewrite(id,make(name,{choose(p,*args[0],*args[1])}));
      factored_selects_.insert(id);return;
    }
    // Intersect local factor heads rather than cross-product all e-class
    // alternatives. Each successful source enode emits only one factoring.
    const auto& left=index.get(original[1]).factors;
    const auto& right=index.get(original[2]).factors;
    const auto& small=left.size()<right.size()?left:right;
    const auto& other=left.size()<right.size()?right:left;
    for(const auto& [head,variable]:small) {
      poll();auto found=other.find(head);if(found==other.end())continue;
      const auto [name,common]=head;
      const auto x=left.size()<right.size()?variable:found->second;
      const auto y=left.size()<right.size()?found->second:variable;
      if(!finite(common) || !finite(x) || !finite(y))continue;
      rewrite(id,make(name,{find(common),choose(p,x,y)}));
      factored_selects_.insert(id);return;
    }
    for(unsigned arm=0;arm<2;++arm) {
      if(!zero_[find(original[arm+1])])continue;
      const auto& heads=index.get(original[2-arm]);
      if(heads.product) {
        auto [factor,value]=*heads.product;
        if(finite(factor) && finite(value)) {
          auto selected=arm==0?choose(p,zero(),value):choose(p,value,zero());
          rewrite(id,make("Multiply",{find(factor),selected}));
          factored_selects_.insert(id);return;
        }
      }
      // Select(p,0,x-y)=x-Select(p,x,y), and the opposite arm.
      // This fixed Add/Negate/Subtract pattern exposes correlated recentering
      // without distributing arbitrary arithmetic across one Select.
      if(heads.difference) {
        auto [x,y]=*heads.difference;
        if(finite(x) && finite(y)) {
          auto selected=arm==0?choose(p,x,y):choose(p,y,x);
          rewrite(id,make("Add",{find(x),make("Negate",{selected})}));
          factored_selects_.insert(id);return;
        }
      }
    }
    return;
  }
  if(!idealOperation(k) || !k.provenance.empty() || k.children.size()!=2 ||
      merged_selects_.contains(id))return;
  const auto& left=index.get(original[0]).conditions;
  const auto& right=index.get(original[1]).conditions;
  const auto& small=left.size()<right.size()?left:right;
  const auto& other=left.size()<right.size()?right:left;
  for(auto [condition,member]:small) {
    poll();auto found=other.find(condition);if(found==other.end())continue;
    const auto& x=snapshot[left.size()<right.size()?member:found->second];
    const auto& y=snapshot[left.size()<right.size()?found->second:member];
    if(x.type!=k.type || y.type!=k.type ||
        !finite(x.children[1]) || !finite(x.children[2]) ||
        !finite(y.children[1]) || !finite(y.children[2]) ||
        (k.name=="Divide" && (!nonzero(y.children[1]) || !nonzero(y.children[2]))))continue;
    auto a=make(k.name,{find(x.children[1]),find(y.children[1])});
    auto b=make(k.name,{find(x.children[2]),find(y.children[2])});
    rewrite(id,choose(condition,a,b));merged_selects_.insert(id);return;
  }
}

inline auto Engine::compactImported(std::vector<Class>& a,std::vector<Class>& b) -> void {
  std::vector<bool> live(nodes_.size());
  std::vector<Class> todo=a;todo.insert(todo.end(),b.begin(),b.end());
  while(!todo.empty()) {
    poll();auto id=todo.back();todo.pop_back();
    if(live[id])continue;
    live[id]=true;todo.insert(todo.end(),nodes_[id].children.begin(),nodes_[id].children.end());
  }
  auto old=std::move(nodes_);
  nodes_.clear();parents_.clear();sizes_.clear();minimum_.clear();finite_.clear();
  class_members_.clear();real_head_ids_.clear();fold_head_ids_.clear();
  nonnegative_.clear();positive_.clear();nonzero_.clear();predicate_.clear();
  zero_.clear();negative_infinity_.clear();positive_infinity_.clear();
  integer_ranges_.clear();
  free_indices_.clear();dependents_.clear();property_queue_.clear();property_queued_.clear();
  max_index_.clear();has_fold_.clear();
  memo_.clear();factored_folds_.clear();
  // Import/projection matcher keys contain class IDs, which this compaction
  // remaps. No structural certificate survives a different ID epoch.
  shifted_body_matches_.clear();
  factored_selects_.clear();merged_selects_.clear();
  negated_products_.clear();
  combined_exponentials_.clear();
  coalesced_forests_.clear();
  flattened_folds_.clear();
  numeric_projection_.clear();exact_projection_.clear();predicate_projection_.clear();
  if(projection_scopes_ && sentryward::debug::environment("SENTRYWARD_EGRAPH_TRACE")) {
    std::size_t admitted=0;
    for(const auto& entry:projection_sat_certificates_)admitted+=entry.second;
    const auto scopes=projection_scopes_->statistics();
    SENTRYWARD_DEBUG_PRINT(egraph_76);
  }
  projection_body_heads_.reset();projection_affine_cache_.clear();
  // All source IDs are reassigned below. No descriptor source slot or
  // borrowed metadata view can survive compaction; the next physical
  // planner bootstrap remints exact descriptors under a new generation.
  complete_headers_.reset();
  if(++complete_header_generation_==0)throw std::overflow_error("complete header epoch overflow");
  projection_source_ids_.clear();projection_sat_sensitive_.clear();projection_branch_sensitive_.clear();
  projection_integer_sensitive_.clear();
  branch_candidates_.clear();
  // A compaction epoch reassigns node IDs. Only newly certified wrappers
  // may acquire a conditional predicate property in the new arena.
  branch_proved_wrappers_.clear();
  projection_max_samples_.clear();projection_max_membership_.clear();
  projection_sat_certificates_.clear();projection_sample_count_=0;
  projection_scopes_.reset();projection_root_scope_=lexical_scope::unknown;
  projection_sat_remaining_=2000000;projection_sat_requests_=0;
  std::vector<Class> mapped(old.size());
  for(std::size_t id=0;id<old.size();++id)if(live[id]) {
    poll();auto key=std::move(old[id]);
    for(auto& child:key.children)child=mapped[child];
    mapped[id]=intern(std::move(key));
  }
  for(auto& id:a)id=mapped[id];
  for(auto& id:b)id=mapped[id];
  if(branch_fact_anchor_)branch_fact_anchor_=live[*branch_fact_anchor_]?
      std::optional<Class>{mapped[*branch_fact_anchor_]}:std::nullopt;
  branch_scope_ids_.clear(); // Catalog IDs are epoch-local; retained wrappers own their old records.
}

inline auto Engine::normalizeExponentialRatio(Class id,std::string_view base) -> Class {
  const auto root=nodes_[id];
  if(!idealOperation(root) || root.name!="Divide" ||
      !finite_[find(root.children[0])] || !positive_[find(root.children[1])] ||
      !finite_[find(root.children[1])])return id;
  using Powers=std::map<Class,std::int64_t>;
  struct Term {Powers powers;Class weight;};
  std::vector<Term> numerator,denominator;
  std::size_t budget=4096;
  const auto operation=[&](std::string name,std::vector<Class> children) {
    Key key;key.kind=static_cast<int>(s::Kind::Operation);key.type=root.type;
    key.name=std::move(name);key.attributes="ideal_arithmetic";key.children=std::move(children);
    return intern(std::move(key));
  };
  const auto constant=[&](std::string text) {
    Key key;key.kind=static_cast<int>(s::Kind::Constant);key.type=root.type;key.name=std::move(text);
    return intern(std::move(key));
  };
  const auto literal=[&](Class value,const char* text) {
    const auto& key=nodes_[value];
    return key.kind==static_cast<int>(s::Kind::Constant) && key.type==root.type &&
        key.provenance.empty() && key.attributes.empty() && key.name==text;
  };
  const auto one=constant("1");
  std::function<bool(Class,std::int64_t,Powers&)> exponent;
  exponent=[&](Class value,std::int64_t sign,Powers& powers) {
    poll();if(!budget)return false;--budget;
    const auto key=nodes_[value];
    if(!finite_[find(value)] || key.type!=root.type || !key.provenance.empty())return false;
    if(literal(value,"0"))return true;
    if(idealOperation(key)) {
      if(key.name=="Add" || key.name=="Subtract")
        return exponent(key.children[0],sign,powers) &&
            exponent(key.children[1],key.name=="Add"?sign:-sign,powers);
      if(key.name=="Negate")return exponent(key.children[0],-sign,powers);
    }
    std::int64_t coefficient;
    if(__builtin_add_overflow(powers[value],sign,&coefficient))return false;
    if(coefficient)powers[value]=coefficient;else powers.erase(value);
    return true;
  };
  std::function<bool(Class,Powers,std::vector<Term>&)> collect;
  collect=[&](Class value,Powers powers,std::vector<Term>& terms) {
    poll();if(!budget || terms.size()>=128)return false;--budget;
    const auto key=nodes_[value];
    if(!finite_[find(value)] || key.type!=root.type || !key.provenance.empty())return false;
    if(literal(value,"0"))return true;
    if(idealOperation(key)) {
      if(key.name=="Add")return collect(key.children[0],powers,terms) && collect(key.children[1],powers,terms);
      if(key.name=="Multiply") {
        for(unsigned side=0;side<2;++side) {
          const auto factor=nodes_[key.children[side]];
          if(idealOperation(factor) && factor.name==base) {
            if(!exponent(factor.children[0],1,powers))return false;
            return collect(key.children[1-side],std::move(powers),terms);
          }
        }
      }
      if(key.name==base) {
        if(!exponent(key.children[0],1,powers))return false;
        terms.push_back({std::move(powers),one});return true;
      }
    }
    terms.push_back({std::move(powers),value});return true;
  };
  if(!collect(root.children[0],{},numerator) || !collect(root.children[1],{},denominator) ||
      numerator.empty() || denominator.empty())return id;
  Powers common;
  for(const auto* terms:{&numerator,&denominator})for(const auto& term:*terms)
    for(auto [atom,coefficient]:term.powers)common.emplace(atom,coefficient);
  bool changed=false;
  for(auto& [atom,coefficient]:common) {
    for(const auto* terms:{&numerator,&denominator})for(const auto& term:*terms) {
      auto found=term.powers.find(atom);
      coefficient=std::min(coefficient,found==term.powers.end()?0:found->second);
    }
    changed|=coefficient!=0;
  }
  if(!changed)return id;
  // Both sides are divided by the same strictly positive finite Exp2 factor.
  for(auto* terms:{&numerator,&denominator})for(auto& term:*terms) {
    for(auto [atom,coefficient]:common) {
      auto& current=term.powers[atom];
      if(__builtin_sub_overflow(current,coefficient,&current))return id;
    }
    std::erase_if(term.powers,[](const auto& item){return item.second==0;});
  }
  const auto build=[&](std::vector<Term>& terms) {
    std::sort(terms.begin(),terms.end(),[](const auto& a,const auto& b) {
      return std::tie(a.powers,a.weight)<std::tie(b.powers,b.weight);
    });
    auto sum=constant("0");
    for(const auto& term:terms) {
      auto power=constant("0");
      for(auto [atom,coefficient]:term.powers) {
        auto value=coefficient==1?atom:operation("Multiply",{constant(std::to_string(coefficient)),atom});
        power=literal(power,"0")?value:operation("Add",{power,value});
      }
      auto value=term.weight;
      if(!term.powers.empty()) {
        auto factor=operation(std::string(base),{power});
        value=literal(value,"1")?factor:operation("Multiply",{value,factor});
      }
      sum=literal(sum,"0")?value:operation("Add",{sum,value});
    }
    return sum;
  };
  auto top=build(numerator),bottom=build(denominator);
  return operation("Divide",{top,bottom});
}

inline auto Engine::numericOutput(Class id) const -> bool {
  const auto& k=nodes_[id];
  if(!k.provenance.empty())return false;
  if(realType(k.type))return true;
  return k.kind==6 && k.provenance.empty() && k.children.size()==1 &&
      idealNumericOutputConversion(nodes_[k.children[0]].type,k.type,k.name);
}

inline auto Engine::numericOutputProducer(Class id) const -> Class {
  // Erase an approved conversion USE at the comparison root, not its e-class.
  // A shared producer's internal/predicate/exact consumers still project from
  // their original typed nodes. No global union or ancestor search is needed.
  // Import represents the bounded annotation chain as successive use nodes.
  // Only its outermost edge may be the same-width storage bitcast; never walk
  // through an integer producer, a mathematical op, or an unrecognized cast.
  std::size_t metadata=0;
  for(unsigned step=0;step<16;++step) {
    const auto& k=nodes_[id];
    if(k.kind!=6 || !k.provenance.empty() || k.children.size()!=1 ||
        (step && !realType(k.type)) || k.name.size()>128-metadata ||
        !idealNumericOutputConversion(nodes_[k.children[0]].type,k.type,k.name))break;
    metadata+=k.name.size();id=k.children[0];
  }
  return id;
}

} // namespace
} // namespace sentryward::equivalence
