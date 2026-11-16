#pragma once
#include "egraph_engine.hpp"

namespace sentryward::equivalence {
namespace {

inline auto Engine::ScopedSumProof::plainFold(Class id) -> bool {
  if(id>=physical_nodes || !heads.charge(5))return false;
  const auto& k=heads.snapshot[id];
  return k.kind==static_cast<int>(s::Kind::Fold) && k.fold &&
      k.fold->reducer==s::Reducer::Sum && k.fold->count && k.type=="real" &&
      k.attributes.empty() && k.provenance.empty() && k.children.size()==2 &&
      k.children[0]<physical_nodes && k.children[1]<physical_nodes &&
      heads.engine.finite_[heads.owner(id)] &&
      heads.snapshot[k.children[0]].type==k.type && heads.snapshot[k.children[1]].type==k.type &&
      heads.engine.finite_[heads.owner(k.children[0])] && heads.engine.finite_[heads.owner(k.children[1])];
}

inline auto Engine::ScopedSumProof::fold(Class cls,Scope parent) -> std::optional<Class> {
  if(parent>=registry.scopes.size())return {};
  const auto rank=registry.bounds.rank(registry.scopes[parent].bounds);if(!rank)return {};
  if(heads.engine.unions_!=selection_epoch){heads.cap();return {};}
  cls=heads.owner(cls);
  if(!heads.charge(2))return {};
  // This existing conservative OR property is eagerly true for every
  // physical Fold and merged alias; a false value therefore excludes a
  // Fold witness without a new catalog. Adapter IDs are another namespace.
  if(!heads.adapter && cls<heads.engine.has_fold_.size() && !heads.engine.has_fold_[cls]) {
    if(trace_work)++fold_selection_no_fold;
    return {};
  }
  const auto selection_key=std::pair{cls,*rank};
  if(!heads.charge(3+std::bit_width(fold_selections.size()+1)))return {};
  if(auto cached=fold_selections.find(selection_key);cached!=fold_selections.end()) {
    if(trace_work)++fold_selection_hits;
    return cached->second;
  }
  const auto* choices=heads.rootChoices(cls);if(!choices)return {};
  if(trace_work)++fold_selection_scans;
  std::optional<Class> result;
  // ONE minimum actual witness at the original lexical depth. Certified
  // alpha aliases are not independently retried on proof failure.
  for(const auto& [key,choice]:*choices) {
    (void)key;if(!heads.charge())return {};
    const bool eligible=plainFold(choice.head);
    // plainFold() may deny the LAST choice on a per-query charge without
    // making the index sticky-exhausted. Never cache that partial scan,
    // including a prior candidate or an apparent absence. Exact quota
    // depletion conservatively remains uncached too.
    if(heads.exhausted || (heads.remaining && !*heads.remaining) ||
        heads.query_guards.exhausted())return {};
    if(eligible && heads.snapshot[choice.head].fold->binder_level==*rank &&
        (!result || choice.head<*result))result=choice.head;
  }
  if(!registry.contextCapacity(fold_selections.size(),"scoped_sum_fold_selection") ||
      !heads.charge(3+std::bit_width(fold_selections.size()+1)))return {};
  fold_selections.emplace(selection_key,result);
  if(!result && trace_work)++fold_selection_absences;
  return result;
}

inline auto Engine::ScopedSumProof::forest(Class cls) -> std::optional<Class> {
  cls=heads.owner(cls);
  if(heads.engine.unions_!=selection_epoch || !heads.engine.finite_[cls])return {};
  const auto eligible=[&](Class id) {
    if(id>=physical_nodes || !heads.charge(4))return false;
    const auto& k=heads.snapshot[id];
    return k.kind==static_cast<int>(s::Kind::Operation) && !k.fold &&
        k.children.size()==2 && k.type=="real" && k.provenance.empty() &&
        idealOperation(k) && (k.name=="Add" || k.name=="Subtract");
  };
  auto preferred=heads.compact(cls);if(!preferred)return {};
  if(eligible(*preferred))return preferred;
  // A certified Fold alias may have the wrong ORIGINAL lexical depth for
  // this use. Select ONE existing linear forest header before the proof;
  // never rebind that Fold, search ancestors, or retry a failed body walk.
  const auto* choices=heads.rootChoices(cls);if(!choices)return {};
  std::optional<Class> selected;
  for(const auto& [key,choice]:*choices) {
    (void)key;const bool admitted=eligible(choice.head);
    if(heads.exhausted || (heads.remaining && !*heads.remaining) ||
        heads.query_guards.exhausted())return {};
    if(admitted && (!selected || choice.head<*selected))selected=choice.head;
  }
  return selected;
}

inline auto Engine::ScopedSumProof::compatible(Class a,Class b) -> bool {
  // O(64) root-header scheduling only. It does not inspect an ancestor,
  // invent a Fold, or certify any child/seed/address relation.
  const auto* choices=heads.rootChoices(heads.owner(a));if(!choices)return false;
  bool sum=false;for(const auto& [key,c]:*choices){(void)key;if(!heads.charge())return false;sum|=plainFold(c.head);}
  if(!sum || !heads.engine.finite_[heads.owner(b)])return false;
  auto selected=heads.compact(heads.owner(b));if(!selected)return false;
  return plainFold(*selected) || bool(forest(b));
}

inline auto Engine::ScopedSumProof::select(Class a,Class b,Scope r,Scope actual) -> Selection {
  // A seeded Sum may share a certified Add alias with its mate. Choose
  // its ONE original-depth actual witness BEFORE structural matching;
  // otherwise that Add can pair a seed with a partial Fold. This is only
  // header dispatch: related() or the unchanged ordinary paired walk
  // must prove every actual seed/body/address below, without a retry.
  auto p=fold(a,r);if(!p)return {};
  auto q=fold(b,actual);
  if(q) {
    if(heads.snapshot[*p].fold->count!=heads.snapshot[*q].fold->count)
      return {{},true};
    auto x=heads.header(*p),y=heads.header(*q);
    if(x && y && *x==*y)return {std::pair{*p,*q},false};
    return {};
  }
  if(forest(b))return {{},true};
  return {};
}

inline auto Engine::ScopedSumProof::domain(const Axis& axis,Scope r,Scope a) -> bool {
  if(r>=registry.scopes.size() || a>=registry.scopes.size() ||
      axis.reference_scope>=registry.scopes.size() || axis.actual_scope>=registry.scopes.size() ||
      !heads.charge(6))return false;
  const auto& rs=registry.scopes[r];const auto& origin=registry.scopes[axis.reference_scope];
  if(axis.reference>=rs.domains.size() || axis.reference>=origin.domains.size() ||
      rs.domains[axis.reference]!=origin.domains[axis.reference] ||
      registry.count(r,axis.reference)!=axis.reference_count)return false;
  if(axis.actual) {
    const auto& as=registry.scopes[a];const auto& ao=registry.scopes[axis.actual_scope];
    if(*axis.actual>=as.domains.size() || *axis.actual>=ao.domains.size() ||
        as.domains[*axis.actual]!=ao.domains[*axis.actual] ||
        registry.count(a,*axis.actual)!=axis.actual_count)return false;
  } else {
    // A point has no actual binder. Nested equal-count BODY scopes may
    // extend its original parent, but may not replace a captured domain.
    const auto& parent=registry.scopes[axis.actual_scope];const auto& current=registry.scopes[a];
    if(axis.actual_count!=1 || parent.domains.size()>current.domains.size())return false;
    for(std::size_t i=0;i<parent.domains.size();++i) {
      if(!heads.charge() || parent.domains[i]!=current.domains[i] ||
          registry.count(axis.actual_scope,i)!=registry.count(a,i))return false;
    }
  }
  return true;
}

inline auto Engine::ScopedSumProof::independent(Class cls,const Assignment& assignment,Scope r,Scope a,
    std::span<const std::pair<std::uint32_t,std::uint32_t>> lexical) -> bool {
  if(!valid(assignment))return false;
  std::array<std::uint32_t,10> covered{};std::size_t n=0;
  for(const auto& axis:assignment.axes) {
    if(!domain(axis,r,a))return false;
    const bool identity=axis.actual && *axis.actual==axis.reference &&
        axis.stride==1 && axis.offset==0;
    if(!identity && (heads.depends(cls,axis.reference) ||
        (axis.actual && heads.depends(cls,*axis.actual))))return false;
    if(identity)covered[n++]=axis.reference;
  }
  for(auto [x,y]:lexical) {
    if(!heads.charge() || (x!=y && (heads.depends(cls,x) || heads.depends(cls,y))))return false;
    if(x==y)covered[n++]=x;
  }
  auto rv=registry.scopeView(r),av=registry.scopeView(a);
  return rv && av && registry.sameAssignment(cls,*rv,*av,std::span(covered.data(),n));
}

inline auto Engine::ScopedSumProof::sameOwnerIdentity(Class cls,Geometry& geometry,Scope r,Scope a,
    std::span<const std::pair<std::uint32_t,std::uint32_t>> lexical) -> bool {
  // Congruence of the SAME complete function at identity coordinates. This
  // is a completed task boundary, not an orientation/header hint and not
  // an inference that this function is injective. No inner Fold is rebound.
  if(heads.engine.options_.certified_branch_scope || heads.adapter ||
      heads.engine.unions_!=selection_epoch || !valid(geometry.assignment) ||
      lexical.size()>8 || r>=registry.scopes.size() || a>=registry.scopes.size() ||
      cls>=physical_nodes)return false;
  cls=heads.owner(cls);
  const auto free_size=heads.engine.free_indices_[cls].size();
  if(free_size>64 || !heads.charge(16+8*(geometry.assignment.axes.size()+lexical.size())+
      4*(geometry.assignment.axes.size()+lexical.size()+1)*std::bit_width(free_size+1)))return false;
  auto candidate=geometry.assignment; // At most two fixed-size Axis records.
  for(auto& axis:candidate.axes) {
    // Even unused axes must retain their checked ORIGINAL domains.
    if(!domain(axis,r,a))return false;
    const bool used=heads.depends(cls,axis.reference) ||
        (axis.actual && heads.depends(cls,*axis.actual));
    if(!used)continue;
    if(!axis.actual || *axis.actual!=axis.reference || !axis.actual_count ||
        axis.actual_count>axis.reference_count ||
        (axis.stride && *axis.stride!=1) || (axis.offset && *axis.offset!=0))return false;
    axis.stride=1;axis.offset=0;
  }
  for(auto [x,y]:lexical) {
    if(!heads.depends(cls,x) && !heads.depends(cls,y))continue;
    const auto rc=registry.count(r,x),ac=registry.count(a,y);
    if(x!=y || !rc || !*rc || rc!=ac)return false;
  }
  // Existing whole-owner cached absence/capture guards include seeds.
  // All unmapped free coordinates retain exact occurrence tokens/counts;
  // nonidentity lexical bindings cannot capture a coordinate of this owner.
  if(!independent(cls,candidate,r,a,lexical))return false;
  geometry.assignment=std::move(candidate);++identity_assignment_hits;return true;
}

inline auto Engine::ScopedSumProof::leaf(IndexAffine x,IndexAffine y,Geometry& geometry,Scope r,Scope a,
    std::span<const std::pair<std::uint32_t,std::uint32_t>> lexical) -> bool {
  if(!valid(geometry.assignment) || geometry.equations.size()>=4096 ||
      !registry.boundedAffine(x,r) || !registry.boundedAffine(y,a) ||
      !heads.charge(4*(x.coefficients.size()+y.coefficients.size()+lexical.size()+3)*
          (1+std::bit_width(x.coefficients.size()+y.coefficients.size()+1))))return false;
  Equation equation;
  for(std::size_t i=0;i<geometry.assignment.axes.size();++i) {
    auto& axis=geometry.assignment.axes[i];if(!domain(axis,r,a))return false;
    const auto ca=x.coefficients.contains(axis.reference)?x.coefficients.at(axis.reference):0;
    const auto cb=axis.actual && y.coefficients.contains(*axis.actual)?y.coefficients.at(*axis.actual):0;
    if(ca) {
      if((axis.actual && (!cb || cb%ca)) || (!axis.actual && cb))return false;
      const auto stride=axis.actual?cb/ca:1;
      if(!stride || (axis.stride && *axis.stride!=stride))return false;
      axis.stride=stride;
    } else if(cb)return false;
    equation.coefficients[i]=ca;x.coefficients.erase(axis.reference);
    if(axis.actual)y.coefficients.erase(*axis.actual);
  }
  for(auto [xlevel,ylevel]:lexical) {
    const auto ca=x.coefficients.contains(xlevel)?x.coefficients.at(xlevel):0;
    const auto cb=y.coefficients.contains(ylevel)?y.coefficients.at(ylevel):0;
    if(ca!=cb)return false;x.coefficients.erase(xlevel);y.coefficients.erase(ylevel);
  }
  if(y.constant<x.constant)return false;equation.value=y.constant-x.constant;
  x.constant=y.constant=0;
  if(!registry.finishLeaf(x,y,r,a))return false;
  geometry.equations.push_back(equation);return true;
}

inline auto Engine::ScopedSumProof::solve(Geometry& geometry) -> bool {
  if(!valid(geometry.assignment) || !heads.charge(32*geometry.equations.size()+8))return false;
  auto& axes=geometry.assignment.axes;
  // Unit pivots are common typed-memory anchors. At most two passes can
  // resolve two axes; unresolved coupled offsets are Unknown, not zero.
  for(unsigned pass=0;pass<2;++pass)for(const auto& e:geometry.equations) {
    std::uint64_t known=0;std::optional<std::size_t> unknown;bool multiple=false;
    for(std::size_t i=0;i<axes.size();++i)if(e.coefficients[i]) {
      if(axes[i].offset) {
        std::uint64_t part;if(__builtin_mul_overflow(e.coefficients[i],*axes[i].offset,&part) ||
            __builtin_add_overflow(known,part,&known))return false;
      } else if(unknown)multiple=true;else unknown=i;
    }
    if(known>e.value)return false;
    if(unknown && !multiple) {
      const auto value=e.value-known,coefficient=e.coefficients[*unknown];
      if(value%coefficient)return false;axes[*unknown].offset=value/coefficient;
    } else if(!unknown && known!=e.value)return false;
  }
  // A mixed-radix address can determine two point coordinates without a
  // second address equation: H*x + L*y = offset, 0 <= y < N, H > L*(N-1).
  // Use the ORIGINAL finite domains, not guessed layout dimensions. The
  // gap check and final substitution below also reject addresses in holes.
  // This is constant work per stored equation; no domain is enumerated.
  if(axes.size()==2 && !axes[0].offset && !axes[1].offset) {
    for(const auto& e:geometry.equations) {
      if(!e.coefficients[0] || !e.coefficients[1])continue;
      const auto high=e.coefficients[0]>=e.coefficients[1]?0u:1u,low=1u-high;
      const auto h=e.coefficients[high],l=e.coefficients[low];
      std::uint64_t span;
      if(!axes[low].reference_count ||
          __builtin_mul_overflow(l,axes[low].reference_count-1,&span) || h<=span)continue;
      const auto x=e.value/h,rest=e.value%h;
      if(rest%l)return false;
      axes[high].offset=x;axes[low].offset=rest/l;break;
    }
  }
  for(const auto& axis:axes) {
    if(!axis.stride || !axis.offset || !axis.reference_count || !axis.actual_count)return false;
    std::uint64_t last;
    if(__builtin_mul_overflow(axis.actual_count-1,*axis.stride,&last) ||
        __builtin_add_overflow(last,*axis.offset,&last) || last>=axis.reference_count)return false;
  }
  for(const auto& e:geometry.equations) {
    std::uint64_t value=0;
    for(std::size_t i=0;i<axes.size();++i) {
      std::uint64_t part;if(__builtin_mul_overflow(e.coefficients[i],*axes[i].offset,&part) ||
          __builtin_add_overflow(value,part,&value))return false;
    }
    if(value!=e.value)return false;
  }
  return true;
}

inline auto Engine::ScopedSumProof::project(Class target,Class actual,Scope r,Scope a,
    const Assignment& incoming) -> std::optional<Projection> {
  if(!valid(incoming) || target>=physical_nodes || actual>=physical_nodes ||
      !heads.charge(8+4*incoming.axes.size()))return {};
  const auto left=heads.owner(target),right=heads.owner(actual);
  Projection result{root(),{}};
  for(std::size_t i=0;i<incoming.axes.size();++i) {
    const auto& axis=incoming.axes[i];
    // An omitted axis retains its ORIGINAL lexical domains. Absence from
    // the conservative whole-expression union includes the actual seeds;
    // presence is Unknown, never a request for another dependency walk.
    if(!domain(axis,r,a) || !heads.charge(2+
        std::bit_width(heads.engine.free_indices_[left].size()+1)+
        std::bit_width(heads.engine.free_indices_[right].size()+1)))return {};
    const bool reference_used=heads.depends(left,axis.reference);
    const bool actual_used=axis.actual && heads.depends(right,*axis.actual);
    if(!reference_used && !actual_used)continue;
    result.original[result.local.axes.size()]=i;
    result.local.axes.push_back(axis);
  }
  return result;
}

inline auto Engine::ScopedSumProof::merge(const Projection& projection,const Assignment& proved,
    const Assignment& incoming,Assignment& result) -> bool {
  if(!valid(projection.local) || !valid(proved) || !valid(incoming) ||
      proved.axes.size()!=projection.local.axes.size() ||
      !heads.charge(8+8*incoming.axes.size()))return false;
  result=incoming;
  for(std::size_t i=0;i<proved.axes.size();++i) {
    const auto original=projection.original[i];if(original>=incoming.axes.size())return false;
    const auto& before=incoming.axes[original];const auto& after=proved.axes[i];
    if(projection.local.axes[i]!=before ||
        std::tie(before.reference_scope,before.actual_scope,before.reference,before.actual,
            before.reference_count,before.actual_count)!=
        std::tie(after.reference_scope,after.actual_scope,after.reference,after.actual,
            after.reference_count,after.actual_count) ||
        (before.stride && before.stride!=after.stride) ||
        (before.offset && before.offset!=after.offset))return false;
    // Only genuinely derived retained-axis facts are returned. Omitted
    // axes, including unknown geometry, are copied unchanged. The caller's
    // equations remain in their ORIGINAL index order and must still pass
    // its final solve(); this local proof invents no zero offset/stride.
    if(after.stride)result.axes[original].stride=after.stride;
    if(after.offset)result.axes[original].offset=after.offset;
  }
  return true;
}

inline auto Engine::ScopedSumProof::qualify(Class witness,Class actual,Scope r,Scope a,
    const Assignment& active,const Environment& enclosing) -> std::optional<QualifiedKey> {
  // Certified branch anchors have a separate proof-metadata namespace
  // deliberately absent from free_indices_. Keep their old full-key path.
  if(heads.engine.options_.certified_branch_scope || !valid(active) || active.axes.size()>=2 ||
      r>=registry.scopes.size() || a>=registry.scopes.size() ||
      heads.engine.unions_!=selection_epoch ||
      !heads.charge(8+8*(active.axes.size()+enclosing.size())))return {};
  auto rr=registry.bounds.rank(registry.scopes[r].bounds);
  auto ar=registry.bounds.rank(registry.scopes[a].bounds);
  if(!rr || !ar)return {};
  for(const auto& axis:active.axes)if(!domain(axis,r,a))return {};
  auto rv=registry.scopeView(r),av=registry.scopeView(a);if(!rv || !av)return {};
  const auto capture=[&](Class cls,const auto& view,bool reference)->std::optional<Captures> {
    const auto& free=heads.engine.free_indices_[heads.owner(cls)];
    if(free.size()>64 || !heads.charge(4*free.size()+1))return {};
    Captures result;result.reserve(free.size());
    for(auto level:free) {
      bool mapped=false;
      for(const auto& axis:active.axes)
        mapped|=reference?axis.reference==level:axis.actual==level;
      if(mapped)continue;
      auto token=registry.domain(view,level);auto count=registry.count(view,level);
      if(!token || !count || !*count)return {};
      result.emplace_back(level,token->first,token->second,*count);
    }
    return result;
  };
  auto left=capture(witness,*rv,true),right=capture(actual,*av,false);
  if(!left || !right)return {};
  return QualifiedKey{witness,heads.owner(actual),*rr,*ar,active.axes,
      std::move(*left),std::move(*right),enclosing};
}

inline auto Engine::ScopedSumProof::rebind(const CompletedRelation& completed,const Projection& projection,
    Class witness,Class actual,Scope r,Scope a,const Assignment& incoming,Assignment& result) -> bool {
  if(!completed.key || !completed.record || completed.record->gray ||
      !completed.record->projected_result || heads.engine.unions_!=selection_epoch)return false;
  const auto& key=*completed.key;
  if(std::get<0>(key)!=witness || std::get<1>(key)!=heads.owner(actual))return false;
  auto old_r=registry.scopeView(std::get<2>(key)),new_r=registry.scopeView(r);
  auto old_a=registry.scopeView(std::get<3>(key)),new_a=registry.scopeView(a);
  if(!old_r || !new_r || !old_a || !new_a)return false;
  std::array<std::uint32_t,2> reference{},actual_levels{};std::size_t rn=0,an=0;
  for(const auto& axis:projection.local.axes) {
    // These are the ONLY skipped capture tokens; both requests must use
    // the SAME immutable original axis definitions and their domains.
    if(!domain(axis,std::get<2>(key),std::get<3>(key)) || !domain(axis,r,a))return false;
    reference[rn++]=axis.reference;
    if(axis.actual)actual_levels[an++]=*axis.actual;
  }
  if(!registry.sameAssignment(witness,*old_r,*new_r,std::span(reference.data(),rn)) ||
      !registry.sameAssignment(actual,*old_a,*new_a,std::span(actual_levels.data(),an)))return false;
  return merge(projection,*completed.record->projected_result,incoming,result);
}

inline auto Engine::ScopedSumProof::remember(QualifiedKey key,const GoalKey& original,const Record& completed) -> void {
  if(!heads.charge(16*(std::get<4>(key).size()+std::get<5>(key).size()+
      std::get<6>(key).size()+std::get<7>(key).size()+8)*(1+std::bit_width(qualified.size()+1))))return;
  if(qualified.find(key)!=qualified.end())return;
  if(!registry.contextCapacity(qualified.size(),"scoped_sum_completed_relation"))return;
  // map-owned keys/records remain stable throughout this sealed epoch.
  // No graph node, growing vector element or callback is borrowed here.
  qualified.emplace(std::move(key),CompletedRelation{&original,&completed});
}

inline auto Engine::ScopedSumProof::sameCoefficient(const bounded_linear_reduction::Coefficient& x,
    const bounded_linear_reduction::Coefficient& y) -> bool {
  if(x.terms().size()>1 || y.terms().size()>1 ||
      (!x.terms().empty() && !x.terms()[0].monomial.powers.empty()) ||
      (!y.terms().empty() && !y.terms()[0].monomial.powers.empty()))return false;
  std::size_t work=2;
  for(const auto* coefficient:{&x,&y})for(const auto& term:coefficient->terms())
    work+=term.coefficient.numerator.backend().size()+term.coefficient.denominator.backend().size()+1;
  return heads.charge(work) && x==y;
}

// Compare a two-axis zero-seeded Sum with a stored additive forest.
// Each source atom is a point or an inner-axis Fold. Retain its parameter
// range and prove the complete scalar body under a two-axis assignment.
// Rows are indexed by STORED pieces, never generated from the target count.
inline auto Engine::ScopedSumProof::nestedPointCoverage(Class outer,Scope r,Scope a,const Environment& enclosing,
    const Linear& source,const Assignment& incoming) -> bool {
  if(!source || !incoming.axes.empty())return false;
  const auto& f=heads.snapshot[outer];
  if(!plainFold(outer) || !heads.engine.zero_[heads.owner(f.children[0])] ||
      !source->constant().terms().empty())return false;
  auto rs=registry.insideFold(r,outer);if(!rs)return false;
  auto inner=fold(f.children[1],*rs);if(!inner || !plainFold(*inner))return false;
  const auto& g=heads.snapshot[*inner];
  if(g.type!=f.type || g.fold->binder_level!=f.fold->binder_level+1 ||
      !heads.engine.zero_[heads.owner(g.children[0])] ||
      !f.fold->count || !g.fold->count)return false;
  auto body_scope=registry.insideFold(*rs,*inner);if(!body_scope)return false;
  const auto one=builder.coefficient("1");synchronize();if(!one)return false;
  struct Piece {std::uint64_t stride,offset,count;};
  std::map<std::uint64_t,std::vector<Piece>> rows;
  if(source->terms().empty() || source->terms().size()>512)return false;
  for(const auto& term:source->terms()) {
    if(!heads.charge(32) || !sameCoefficient(term.weight,*one))return false;
    auto sf=reduction(term.atom);Class body;Scope ab=a;
    std::uint64_t count=1;std::optional<std::uint32_t> axis;
    if(sf) {
      if(!plainFold(*sf))return false;const auto& h=heads.snapshot[*sf];
      if(h.type!=f.type || (term.atom<physical_nodes &&
          !heads.engine.zero_[heads.owner(h.children[0])]))return false;
      auto scope=registry.insideFold(a,*sf);if(!scope)return false;
      ab=*scope;body=h.children[1];count=h.fold->count;axis=h.fold->binder_level;
    } else {
      if(term.atom>=physical_nodes)return false;body=static_cast<Class>(term.atom);
    }
    Assignment map=incoming;
    map.axes.push_back({*rs,a,f.fold->binder_level,{},f.fold->count,1,{},{}});
    map.axes.push_back({*body_scope,ab,g.fold->binder_level,axis,g.fold->count,count,{},{}});
    if(!heads.engine.affineBodyMapWithHeads(g.children[1],body,UINT32_MAX,
        heads.snapshot,false,heads.remaining,heads,UINT32_MAX,nullptr,nullptr,nullptr,this,
        &map,*body_scope,ab,&enclosing))return false;
    if(!map.axes[0].offset || !map.axes[1].offset || !map.axes[1].stride)return false;
    rows[*map.axes[0].offset].push_back({*map.axes[1].stride,*map.axes[1].offset,count});
  }
  // Every row must occur, but checking the map size and adjacent stored
  // keys establishes this without iterating a potentially huge extent.
  if(rows.size()!=f.fold->count)return false;
  std::uint64_t expected=0;
  for(const auto& [row,pieces]:rows) {
    if(row!=expected++)return false;
    std::uint64_t lattice=1;
    for(const auto& p:pieces) {
      if(!heads.charge(4))return false;
      std::uint64_t next;
      if(!p.stride || __builtin_mul_overflow(lattice/std::gcd(lattice,p.stride),p.stride,&next) ||
          next>64)return false;
      lattice=next;
    }
    for(std::uint64_t residue=0;residue<lattice;++residue) {
      std::vector<bounded_linear_reduction::WeightedInterval> left,right;
      if(residue<g.fold->count)right.push_back({0,1+(g.fold->count-1-residue)/lattice,*one});
      for(const auto& p:pieces) {
        if(!heads.charge(8))return false;
        const auto delta=(residue+lattice-p.offset%lattice)%lattice;
        if(delta%p.stride)continue;
        const auto first=delta/p.stride;if(first>=p.count)continue;
        std::uint64_t coordinate,end;
        if(__builtin_mul_overflow(first,p.stride,&coordinate) ||
            __builtin_add_overflow(p.offset,coordinate,&coordinate))return false;
        const auto begin=coordinate/lattice,n=1+(p.count-1-first)/(lattice/p.stride);
        if(__builtin_add_overflow(begin,n,&end))return false;
        left.push_back({begin,end,*one});
      }
      const auto equal=builder.intervalsEqual(left,right);synchronize();
      if(!equal || !*equal)return false;
    }
  }
  return true;
}

inline auto Engine::ScopedSumProof::related(Class target,Class actual,Scope r,Scope a,const Environment& enclosing,
    Assignment& incoming) -> bool {
  // This operation proves ONE stored forest, not one individual mapper
  // body. Suspend only the paused caller's local pathology guard; every
  // child body installs its own guard and keeps the SAME aggregate.
  BodyHeadIndex::MapperPause aggregation(heads);
  if(!valid(incoming) || enclosing.size()>8 ||
      !heads.charge(24+8*(incoming.axes.size()+enclosing.size())))return false;
  auto witness=fold(target,r);if(!witness)return false;
  auto projection=project(target,actual,r,a,incoming);
  if(!projection || projection->local.axes.size()>=2)return false;
  // Memoize the FULL original assignment, not the shortened local view.
  // Original scopes/enclosing bindings also remain complete key fields.
  const auto key=GoalKey{*witness,heads.owner(actual),r,a,incoming.axes,enclosing};
  if(!heads.charge(4*(incoming.axes.size()+enclosing.size()+8)*(1+std::bit_width(records.size()+1))))return false;
  if(auto found=records.find(key);found!=records.end()) {
    ++hits;if(found->second.gray || !found->second.result)return false;
    incoming=*found->second.result;return true;
  }
  if(!registry.contextCapacity(records.size(),"scoped_sum_goal"))return false;
  auto [entry,inserted]=records.emplace(key,Record{});(void)inserted;++queries;
  GoalTrace report(*this,*witness,heads.owner(actual),r,a,incoming.axes.size());
  // Every exit seals this exact goal as success or Unknown. No alternative
  // prototype, budget reset, or failed full-proof retry is attempted.
  struct Seal {Record& r;~Seal(){r.gray=false;}} seal{entry->second};
  auto qualified_key=qualify(*witness,actual,r,a,projection->local,enclosing);
  if(qualified_key) {
    if(!heads.charge(16*(projection->local.axes.size()+std::get<5>(*qualified_key).size()+
        std::get<6>(*qualified_key).size()+enclosing.size()+8)*
        (1+std::bit_width(qualified.size()+1))))return false;
    if(auto found=qualified.find(*qualified_key);found!=qualified.end()) {
      Assignment rebound;
      // ONE completed prototype, never a gray/Unknown record or retry of
      // another scope after a failed transport. Full request stays keyed.
      if(!rebind(found->second,*projection,*witness,actual,r,a,incoming,rebound))return false;
      entry->second.result=rebound;
      entry->second.projected_result=found->second.record->projected_result;
      incoming=std::move(rebound);++qualified_hits;++completed;
      if(report.enabled)report.success=true;
      return true;
    }
  }
  const auto& target_fold=heads.snapshot[*witness];
  auto rb=registry.insideFold(r,*witness);if(!rb)return false;
  report.next(1);
  auto seed=summarize(heads.owner(target_fold.children[0]));
  if(report.enabled && seed)report.seed_terms=seed->terms().size();
  report.next(2);
  auto source=summarize(heads.owner(actual));
  if(report.enabled && source)report.source_terms=source->terms().size();
  if(!seed || !source)return false;
  report.next(3);
  if(nestedPointCoverage(*witness,r,a,enclosing,source,projection->local)) {
    // No enclosing nonidentity axis was inferred; preserve the complete
    // incoming assignment and the ordinary goal memo's cycle protection.
    entry->second.result=incoming;entry->second.projected_result=projection->local;
    ++completed;if(report.enabled)report.success=true;return true;
  }
  // Linear coefficients are exact constant rationals. Scalar backend
  // limb counts account for equality WITHOUT copying a bigint/record.
  if(!sameCoefficient(seed->constant(),source->constant()))return false;
  const auto& residual=seed->terms();std::vector<bool> consumed(residual.size());
  if(!heads.charge(8*(residual.size()+source->terms().size()+1)))return false;
  struct Piece {std::uint64_t stride,offset,count;};std::vector<Piece> pieces;
  const auto one=builder.coefficient("1");synchronize();if(!one)return false;
  Assignment committed=projection->local;
  for(const auto& term:source->terms()) {
    // Preserve exact outside-binder seed atoms once. A captured atom may
    // not cross a nonidentity/unknown outer assignment just by class ID.
    if(!heads.charge(1+std::bit_width(residual.size()+1)))return false;
    auto found=std::lower_bound(residual.begin(),residual.end(),term.atom,
        [](const auto& s,auto atom){return s.atom<atom;});
    const auto position=static_cast<std::size_t>(found-residual.begin());
    if(found!=residual.end() && found->atom==term.atom && !consumed[position] &&
        sameCoefficient(found->weight,term.weight) && term.atom<physical_nodes &&
        independent(static_cast<Class>(term.atom),committed,r,a,enclosing)) {
      consumed[position]=true;continue;
    }
    if(!sameCoefficient(term.weight,*one) || pieces.size()>=512)return false;
    auto source_fold=reduction(term.atom);
    Class body;Scope ab=a;std::uint64_t count=1;std::optional<std::uint32_t> actual_axis;
    if(source_fold) {
      if(!plainFold(*source_fold))return false;const auto& f=heads.snapshot[*source_fold];
      auto nested=registry.insideFold(a,*source_fold);if(!nested)return false;
      // A private body-only atom excludes the actual seed; a physical
      // ordinary Fold atom is admissible here ONLY with certified zero.
      if(term.atom<physical_nodes && !heads.engine.zero_[heads.owner(f.children[0])])return false;
      body=f.children[1];ab=*nested;count=f.fold->count;actual_axis=f.fold->binder_level;
    } else {
      if(term.atom>=physical_nodes)return false;body=static_cast<Class>(term.atom);
    }
    Geometry local{committed,{}};
    const auto axis=target_fold.fold->binder_level;
    for(const auto& outer:local.assignment.axes)if(outer.reference==axis ||
        (actual_axis && outer.actual==actual_axis))return false;
    local.assignment.axes.push_back({*rb,ab,axis,actual_axis,target_fold.fold->count,count,{},{}});
    PieceTrace piece_report(*this,target_fold.children[1],body,*rb,ab,count);
    if(!heads.engine.affineBodyMapWithHeads(target_fold.children[1],body,UINT32_MAX,
        heads.snapshot,false,heads.remaining,heads,UINT32_MAX,nullptr,nullptr,nullptr,this,
        &local.assignment,*rb,ab,&enclosing))return false;
    if(piece_report.enabled)piece_report.complete=true;
    const auto image=local.assignment.axes.back();
    if(!image.stride || !image.offset)return false;
    pieces.push_back({*image.stride,*image.offset,count});
    if(report.enabled)report.pieces=pieces.size();
    local.assignment.axes.pop_back();
    committed=std::move(local.assignment);
  }
  if(std::find(consumed.begin(),consumed.end(),false)!=consumed.end() || pieces.empty())return false;
  report.next(4);
  // Finite lattice quotient, not Fold enumeration. At most64 residue
  // classes and1024 symbolic intervals retain exact additive multiplicity.
  std::uint64_t lattice=1;
  for(const auto& p:pieces) {
    if(!heads.charge(1+std::bit_width(lattice)+std::bit_width(p.stride)))return false;
    const auto g=std::gcd(lattice,p.stride);std::uint64_t next;
    if(__builtin_mul_overflow(lattice/g,p.stride,&next) || next>64)return false;lattice=next;
  }
  std::vector<bounded_linear_reduction::WeightedInterval> left,right;
  for(std::uint64_t residue=0;residue<lattice;++residue) {
    const auto total=target_fold.fold->count;
    if(residue<total)right.push_back({0,1+(total-1-residue)/lattice,*one});
    for(const auto& p:pieces) {
      if(!heads.charge(8))return false;
      const auto initial=p.offset%lattice;
      const auto delta=(residue+lattice-initial)%lattice;
      if(delta%p.stride)continue;const auto first=delta/p.stride;
      if(first>=p.count)continue;
      std::uint64_t coordinate;
      if(__builtin_mul_overflow(first,p.stride,&coordinate) ||
          __builtin_add_overflow(p.offset,coordinate,&coordinate))return false;
      const auto step=lattice/p.stride,count=1+(p.count-1-first)/step;
      const auto begin=coordinate/lattice;std::uint64_t end;
      if(__builtin_add_overflow(begin,count,&end) || left.size()+right.size()>=1024)return false;
      left.push_back({begin,end,*one});
    }
    const auto equal=builder.intervalsEqual(left,right);synchronize();
    if(!equal || !*equal) {
      // Bounded TRACE sample of intervals already constructed by the proof;
      // no new ancestor, member, or logical reduction traversal.
      if(trace_work && lattice_failure_reports++<8) {
        SENTRYWARD_DEBUG_PRINT(egraph_86);
        for(std::size_t i=0;i<std::min<std::size_t>(left.size(),16);++i)
          SENTRYWARD_DEBUG_PRINT(egraph_87);
        SENTRYWARD_DEBUG_PRINT(egraph_88);
      }
      return false;
    }
    left.clear();right.clear();
  }
  report.next(5);
  Assignment merged;
  if(!merge(*projection,committed,incoming,merged))return false;
  entry->second.result=merged;entry->second.projected_result=committed;
  entry->second.gray=false;
  // Exact incoming and completed geometry have separate keys, not wildcard
  // facts. Two slots avoid repeating an initially-Unknown offset proof
  // once that proof certified it; known conflicts never hit either key.
  if(qualified_key)remember(std::move(*qualified_key),entry->first,entry->second);
  if(auto solved=qualify(*witness,actual,r,a,committed,enclosing))
    remember(std::move(*solved),entry->first,entry->second);
  incoming=std::move(merged);++completed;
  if(report.enabled)report.success=true;
  return true;
}

} // namespace
} // namespace sentryward::equivalence
