#pragma once
#include "egraph_engine.hpp"

namespace sentryward::equivalence {
namespace {

inline auto Engine::branchCharge(std::size_t amount) -> bool {
  poll();if(amount>branch_remaining_){branch_remaining_=0;return false;}
  branch_remaining_-=amount;
  if(options_.max_work && amount>1) {
    if(amount-1>options_.max_work-work_)throw Limit{"work_budget"};
    work_+=amount-1;
  }
  return true;
}

inline auto Engine::branchPredicateSource(const Key& k,const std::vector<Key>& arena) -> bool {
  if(comparison(k)!=std::optional<std::string>{"gt"} ||
      (k.attributes!="predicate=gt" && !k.attributes.empty()) ||
      !k.provenance.empty() || k.children.size()!=2)return false;
  const auto& argument=arena[k.children[0]];const auto& zero=arena[k.children[1]];
  const auto plain=[&](const Key& h,std::string_view name) {
    return h.kind==static_cast<int>(s::Kind::Operation) && h.type==argument.type &&
        h.name==name && h.children.size()==2 && h.provenance.empty() &&
        (h.attributes.empty() || h.attributes=="ideal_arithmetic");
  };
  if((argument.type!="f32" && argument.type!="f64") || !plain(argument,"Add") ||
      zero.kind!=static_cast<int>(s::Kind::Constant) || zero.type!=argument.type ||
      zero.name.size()>32 || !zero.attributes.empty() || !zero.provenance.empty())return false;
  // Import already decoded finite typed constants; do not parse a literal
  // again at every node-intern/sensitivity transfer.
  if(zero.name!="0")return false;
  for(auto child:argument.children)if(plain(arena[child],"Subtract"))return true;
  return false;
}

inline auto Engine::predicateFloatConversion(const Key& k,const std::vector<Key>& arena) -> bool {
  if(k.kind!=6 || !k.provenance.empty() || !k.attributes.empty() ||
      k.children.size()!=1 || arena[k.children[0]].type!="pred" ||
      (k.type!="f32" && k.type!="f64"))return false;
  const auto step=leadingConversion("pred",k.name);
  return step && step->target==k.type && step->remainder.empty() &&
      (step->operation=="Convert" || step->operation=="Convert rounding=nearest-even");
}

inline auto Engine::prepareBranchFacts() -> void {
  if(branch_fact_anchor_ || !options_.certified_branch_scope)return;
  const auto& evidence=*options_.certified_branch_scope;
  if(evidence.pieces().empty() || evidence.pieces().size()>64 || !branchCharge(evidence.pieces().size()*4))return;
  std::vector<s::Ref> refs;
  for(const auto& piece:evidence.pieces()) {
    refs.push_back(piece.expression);
    refs.insert(refs.end(),piece.encoded_operands.begin(),piece.encoded_operands.end());
  }
  const auto encoded=import(evidence.graph(),refs,nullptr,{},nullptr,true);
  Key anchor;anchor.kind=7;anchor.type="branch-facts";anchor.name="CapturedConditionalFacts";
  anchor.attributes=evidence.digest();
  for(auto ref:encoded)anchor.children.push_back(project(ref,true));
  branch_fact_anchor_=intern(std::move(anchor));
}

inline auto Engine::branchScope(ScopeId id) -> std::optional<std::size_t> {
  if(auto it=branch_scope_ids_.find(id);it!=branch_scope_ids_.end())return it->second;
  if(!projection_scopes_ || id==lexical_scope::unknown || branch_scopes_.size()>=512)return {};
  const auto counts=projection_scopes_->counts(id);
  if(!counts || counts->size()>64 || !branchCharge(counts->size()+1))return {};
  BranchScopeRecord scope;
  for(auto count:*counts){if(!count || !*count)return {};scope.counts.push_back(*count);}
  const auto result=branch_scopes_.size();branch_scopes_.push_back(std::move(scope));
  branch_scope_ids_.emplace(id,result);return result;
}

inline auto Engine::branchWrapper(const Key& k) -> std::optional<std::pair<std::size_t,std::string>> {
  if(k.kind!=7 || k.type!="pred" || k.name!="ScopedPositivePredicate" ||
      !k.provenance.empty() || k.children.size()!=2 || k.attributes.size()>32)return {};
  const auto colon=k.attributes.find(':');if(colon==std::string::npos)return {};
  std::size_t scope=0;const auto parsed=std::from_chars(k.attributes.data(),k.attributes.data()+colon,scope);
  const auto format=k.attributes.substr(colon+1);
  if(parsed.ec!=std::errc{} || parsed.ptr!=k.attributes.data()+colon ||
      (format!="f32" && format!="f64"))return {};
  return std::pair{scope,format};
}

// A mask describes only a subset of an already validated ACTUAL binder.
// This is fixed-head exact-u64 geometry, not a branch-note interpretation.
inline auto Engine::branchMask(
    Class mask,std::uint32_t binder,std::uint64_t extent,const std::vector<Key>& snapshot,
    BodyHeadIndex& heads) -> std::optional<std::vector<std::pair<std::uint64_t,std::uint64_t>>> {
  mask=find(mask);if(mask>=snapshot.size() || !branchCharge())return {};
  if(predicate_[mask]==1)return std::vector<std::pair<std::uint64_t,std::uint64_t>>{{0,extent}};
  if(predicate_[mask]==2)return std::vector<std::pair<std::uint64_t,std::uint64_t>>{};
  // A certified conjunction can share an e-class with its exact mask while
  // the canonical owner still names the now-cyclic private conjunction.
  // Reuse the bounded existing selected-head catalog, not that owner head.
  const auto witness=heads.compact(mask);if(!witness || *witness>=snapshot.size())return {};
  const auto& k=snapshot[*witness];const auto cmp=comparison(k);
  if(cmp!=std::optional<std::string>{"ne"} || k.attributes!="predicate=ne" ||
      k.children.size()!=2 || !k.provenance.empty())return {};
  for(unsigned side=0;side<2;++side) {
    const auto coordinate=find(k.children[side]),point=find(k.children[1-side]);
    if(coordinate>=snapshot.size() || point>=snapshot.size())continue;
    if(snapshot[coordinate].type!=snapshot[point].type)continue;
    std::optional<IndexAffine> a,b;
    if(snapshot[coordinate].type=="u64" || snapshot[coordinate].type=="exact:u64") {
      a=heads.affine(coordinate);b=heads.affine(point);
    } else if(snapshot[coordinate].type=="s64" || snapshot[coordinate].type=="exact:s64") {
      // ONE exact value-preserving integer conversion is only a geometry
      // view. The original typed comparison/bit consumer is never aliased.
      // At most TWO same-type exact Add(0,x)/Multiply(1,x) envelopes.
      // This fixed local identity view does not follow arbitrary ancestry.
      Class terminal=coordinate;std::optional<Class> selected;
      for(unsigned depth=0;depth<3;++depth) {
        selected=heads.compact(terminal);if(!selected || !branchCharge(4))break;
        const auto& h=snapshot[*selected];
        if(h.kind==6)break;
        if(depth==2 || h.kind!=static_cast<int>(s::Kind::Operation) ||
            h.type!=snapshot[coordinate].type || !h.attributes.empty() || !h.provenance.empty() ||
            h.children.size()!=2 || (h.name!="Add" && h.name!="Multiply")){selected.reset();break;}
        bool identity=false;
        for(unsigned side=0;side<2;++side) {
          const auto c=find(h.children[side]),x=find(h.children[1-side]);
          if(c>=snapshot.size() || x>=snapshot.size())continue;
          const auto& constant=snapshot[c];
          if(constant.kind==static_cast<int>(s::Kind::Constant) && constant.type==h.type &&
              constant.attributes.empty() && constant.provenance.empty() &&
              constant.name==(h.name=="Add"?"0":"1") && snapshot[x].type==h.type) {
            terminal=x;identity=true;break;
          }
        }
        if(!identity){selected.reset();break;}
      }
      if(!selected)continue;
      const auto& cast=snapshot[*selected];const auto& literal=snapshot[point];
      if(cast.kind!=6 || cast.type!=snapshot[coordinate].type || !cast.attributes.empty() || !cast.provenance.empty() ||
          cast.children.size()!=1 || literal.kind!=static_cast<int>(s::Kind::Constant) ||
          !literal.attributes.empty() || !literal.provenance.empty() || literal.name.size()>32 ||
          cast.name!="Convert(u64 -> s64)" || !branchCharge(8))continue;
      const auto child=find(cast.children[0]);if(child>=snapshot.size() ||
          (snapshot[child].type!="u64" && snapshot[child].type!="exact:u64"))continue;
      a=heads.affine(child);
      if(!a || a->coefficients.size()!=1 || !a->coefficients.contains(binder) ||
          a->coefficients.at(binder)!=1 || !extent ||
          a->constant>std::uint64_t(INT64_MAX) || extent-1>std::uint64_t(INT64_MAX)-a->constant)continue;
      auto range=integer_interval::interval({false,64},static_cast<std::int64_t>(a->constant),
          static_cast<std::int64_t>(a->constant+extent-1));
      auto point_range=integer_interval::literal("s64",literal.name);
      if(!range || !integerConversionRange("u64","s64",cast.name,*range) ||
          !point_range || point_range->lower!=point_range->upper || point_range->lower<0)continue;
      b=IndexAffine{static_cast<std::uint64_t>(point_range->lower),{}};
    } else continue;
    if(!a || !b || a->coefficients.size()!=1 || !a->coefficients.contains(binder) ||
        a->coefficients.at(binder)!=1 || !b->coefficients.empty() ||
        a->constant>UINT64_MAX-extent)continue;
    const auto end=a->constant+extent;
    if(b->constant<a->constant || b->constant>=end)
      return std::vector<std::pair<std::uint64_t,std::uint64_t>>{{0,extent}};
    const auto omitted=b->constant-a->constant;
    std::vector<std::pair<std::uint64_t,std::uint64_t>> intervals;
    if(omitted)intervals.emplace_back(0,omitted);
    if(omitted+1<extent)intervals.emplace_back(omitted+1,extent);
    return intervals;
  }
  return {};
}

inline auto Engine::rewriteScopedBranchPredicates(const std::vector<Key>& snapshot,
    const std::vector<std::vector<Class>>& members) -> void {
  if(!options_.certified_branch_scope || !branch_remaining_)return;
  AffineCache affine;BodyHeadIndex heads(*this,snapshot,members,affine,&branch_remaining_);
  // Share one comparison-wide TRACE allowance with the existing paired-map
  // denial printer. No extra proof query, rendering, or ancestor traversal.
  if(sentryward::debug::environment("SENTRYWARD_EGRAPH_TRACE"))heads.denial_trace_remaining=&branch_trace_remaining_;
  namespace l=bounded_linear_reduction;
  l::Limits limits;limits.max_intervals=128;limits.max_work=262144;
  l::Builder intervals(limits,[&](std::size_t amount){return branchCharge(amount);});
  const auto one=intervals.coefficient("1");if(!one)return;
  std::map<std::pair<Class,Class>,bool> memo;
  const auto certificate=[&](Class wrapper,Class mask)->bool {
    wrapper=find(wrapper);mask=mask==SIZE_MAX?mask:find(mask);
    const auto key=std::pair{wrapper,mask};
    if(auto it=memo.find(key);it!=memo.end())return it->second;
    const auto finish=[&](bool proof){if(memo.size()<512)memo.emplace(key,proof);return proof;};
    if(memo.size()>=512 || branch_requests_>=2048 || !branchCharge() || wrapper>=snapshot.size())return false;
    ++branch_requests_;
    const auto parsed=branchWrapper(snapshot[wrapper]);
    if(!parsed || parsed->first>=branch_scopes_.size())return finish(false);
    const auto& wrapper_head=snapshot[wrapper];
    const auto cmp_id=find(wrapper_head.children[0]),anchor_id=find(wrapper_head.children[1]);
    if(cmp_id>=snapshot.size() || anchor_id>=snapshot.size())return finish(false);
    const auto& cmp=snapshot[cmp_id];const auto& anchor=snapshot[anchor_id];
    if(comparison(cmp)!=std::optional<std::string>{"gt"} || cmp.attributes!="predicate=gt" ||
        anchor.kind!=7 || anchor.type!="branch-facts" || anchor.name!="CapturedConditionalFacts" ||
        anchor.attributes!=options_.certified_branch_scope->digest() || !anchor.provenance.empty() ||
        anchor.children.size()!=options_.certified_branch_scope->pieces().size()*4 ||
        !finite_[find(cmp.children[0])] || !zero_[find(cmp.children[1])] ||
        !finite_[find(cmp.children[1])])return finish(false);
    const auto expression=find(cmp.children[0]);
    if(expression>=snapshot.size() || has_fold_[expression])return finish(false);
    const auto& free=free_indices_[expression];
    if(free.size()!=1)return finish(false);
    const auto binder=free.front();const auto& scope=branch_scopes_[parsed->first];
    if(binder>=scope.counts.size())return finish(false);
    const auto extent=scope.counts[binder];
    auto wanted=mask==SIZE_MAX?
        std::optional{std::vector<std::pair<std::uint64_t,std::uint64_t>>{{0,extent}}}:
        branchMask(mask,binder,extent,snapshot,heads);
    if(heads.denial_trace_remaining && *heads.denial_trace_remaining) {
      --*heads.denial_trace_remaining;
      SENTRYWARD_DEBUG_PRINT(egraph_143);
      if(mask==SIZE_MAX)SENTRYWARD_DEBUG_PRINT(egraph_144);else SENTRYWARD_DEBUG_PRINT(egraph_145);
      SENTRYWARD_DEBUG_PRINT(egraph_146);
      if(!wanted)SENTRYWARD_DEBUG_PRINT(egraph_147);
      else if(wanted->empty())SENTRYWARD_DEBUG_PRINT(egraph_148);
      else for(const auto& [begin,end]:*wanted)SENTRYWARD_DEBUG_PRINT(egraph_149);
      SENTRYWARD_DEBUG_PRINT(egraph_150);
    }
    if(!wanted)return finish(false);
    if(wanted->empty())return finish(true);
    std::vector<l::WeightedInterval> covered,required;
    for(auto [begin,end]:*wanted)required.push_back({begin,end,*one});
    const auto pieces=options_.certified_branch_scope->pieces();
    for(std::size_t n=0;n<pieces.size();++n) {
      if(!branchCharge(5))return finish(false);
      if(pieces[n].format!=parsed->second || !pieces[n].count)continue;
      const auto fact=find(anchor.children[4*n]);bool finite=true;
      if(fact>=snapshot.size() || has_fold_[fact])continue;
      for(unsigned j=0;j<4;++j)finite=finite && finite_[find(anchor.children[4*n+j])];
      if(!finite)continue;
      const bool point=pieces[n].count==1 &&
          !std::binary_search(free_indices_[fact].begin(),free_indices_[fact].end(),branch_fact_binder);
      const auto image=affineBodyMapWithHeads(expression,fact,binder,snapshot,point,&branch_remaining_,heads,branch_fact_binder);
      if(!image) {if(heads.exhausted || !branch_remaining_)return finish(false);continue;}
      if(image->first!=1 || image->second>UINT64_MAX-pieces[n].count)continue;
      const auto begin=image->second,end=std::min(extent,begin+pieces[n].count);
      for(auto [first,last]:*wanted) {
        const auto a=std::max(first,begin),b=std::min(last,end);
        if(a<b)covered.push_back({a,b,*one});
      }
    }
    // Every input weight is +1: canonical nonzero runs are exactly the
    // union of certified reached images, even if observations overlap.
    auto canonical=intervals.canonicalIntervals(covered);if(!canonical)return finish(false);
    for(auto& piece:*canonical)piece.weight=*one;
    const auto equal=intervals.intervalsEqual(*canonical,required);
    const bool proved=equal && *equal;
    branch_admitted_+=proved;
    branch_scope_used_=branch_scope_used_ || proved;
    return finish(proved);
  };
  const auto candidate_count=branch_candidates_.size();
  for(std::size_t candidate=0;candidate<candidate_count;++candidate) {
    if(!branchCharge())break;
    const auto id=branch_candidates_[candidate];if(id>=snapshot.size())continue;
    const auto& k=snapshot[id];
    if(branchWrapper(k)) {
      if(!predicate_[find(id)] && certificate(id,SIZE_MAX) && branch_proved_wrappers_.insert(id).second)
        enqueueProperty(id); // Private scoped fact, NEVER a global GT/true union.
    } else if(k.kind==7 && k.type=="pred" && k.name=="ScopedPredicateAnd" &&
        k.attributes.empty() && k.provenance.empty() && k.children.size()==2) {
      for(unsigned side=0;side<2;++side) {
        const auto a=find(k.children[side]),b=find(k.children[1-side]);
        if(predicate_[a]==1 || certificate(a,b)){rewrite(id,b);break;}
        if(predicate_[a]==2){rewrite(id,a);break;}
      }
    } else if(k.kind==static_cast<int>(s::Kind::Fold) && k.type=="real" && k.fold &&
        k.fold->reducer==s::Reducer::Sum && k.fold->count && k.attributes.empty() &&
        k.provenance.empty() && k.children.size()==2 && zero_[find(k.children[0])]) {
      const auto body=find(k.children[1]);if(body>=snapshot.size())continue;
      const auto& select=snapshot[body];
      if(!idealSelect(select) || select.type!="real" || !zero_[find(select.children[2])])continue;
      const auto value_id=find(select.children[1]);if(value_id>=snapshot.size())continue;
      const auto witness=heads.compact(value_id);if(!witness || *witness>=snapshot.size())continue;
      const auto& value=snapshot[*witness];
      if(value.kind!=static_cast<int>(s::Kind::Constant) || value.type!="real" || value.name!="1" ||
          !value.attributes.empty() || !value.provenance.empty())continue;
      const auto mask=branchMask(select.children[0],k.fold->binder_level,k.fold->count,snapshot,heads);
      if(!mask)continue;
      std::uint64_t count=0;for(auto [begin,end]:*mask)count+=end-begin;
      Key constant;constant.kind=static_cast<int>(s::Kind::Constant);constant.type="real";constant.name=std::to_string(count);
      rewrite(id,intern(std::move(constant)));++branch_indicators_;
    }
  }
}

} // namespace
} // namespace sentryward::equivalence
