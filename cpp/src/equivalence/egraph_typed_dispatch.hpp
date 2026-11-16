#pragma once
#include "egraph_engine.hpp"

namespace sentryward::equivalence {
namespace {

inline auto Engine::TypedDispatch::next() -> std::optional<Step> {
  auto& heads=registry.heads;const auto& snapshot=heads.snapshot;
  constexpr Class none=std::numeric_limits<Class>::max();
  if(closed || scope>=registry.scopes.size() || !heads.charge() || !active.insert(current).second)return {};
  auto selected=heads.compact(current);if(!selected)return {};
  const auto& select=snapshot[*selected];
  if(!idealSelect(select)) {
    auto domain=image?registry.view(*image):std::nullopt;
    if(!domain || domain->derived || domain->end<=domain->begin ||
        excluded.size()+1!=domain->end-domain->begin ||
        !coverage.cover(excluded,{domain->begin,domain->end-1}) ||
        !heads.charge(8+3*((sizeof(Registry::DispatchPointId)+sizeof(Class)-1)/sizeof(Class))))return {};
    // All earlier actual Eq exclusions are unit-weight, disjoint and
    // complete below end-1. The actual false-path remainder is exactly
    // this last singleton; this token is a scalar proof, never a Class.
    Registry::DispatchPointId point;point.owner=&registry;point.epoch=registry.context_epoch;
    point.image=image->id;point.root=root;point.branch=current;point.value=domain->end-1;
    closed=true;return Step{point.value,point,current,true,true};
  }
  auto predicate_head=heads.compact(select.children[0]);if(!predicate_head)return {};
  const auto& predicate=snapshot[*predicate_head];
  if(predicate.children.size()!=2 || comparison(predicate)!=std::optional<std::string>{"eq"})return {};
  Class point=none,axis=none;std::uint64_t endpoint=0;
  for(unsigned side=0;side<2;++side) {
    const auto a=heads.owner(predicate.children[side]),b=heads.owner(predicate.children[1-side]);
    if(snapshot[a].type!=snapshot[b].type || (snapshot[a].type!="u64" && snapshot[a].type!="exact:u64"))return {};
    auto pa=heads.affine(a),ba=heads.affine(b);
    if(pa && ba && pa->coefficients.empty() && ba->coefficients.size()==1 &&
        ba->coefficients.begin()->second==1 &&
        ba->coefficients.begin()->first<heads.engine.options_.outer_binders) {
      if(point!=none)return {};point=a;axis=b;endpoint=pa->constant;
    }
  }
  if(point==none)return {};
  if(coordinate==none) {
    coordinate=axis;auto a=heads.affine(axis);
    image=registry.image(scope,a->coefficients.begin()->first,axis);if(!image)return {};
  } else if(axis!=coordinate)return {};
  auto domain=registry.view(*image);if(!domain)return {};
  const bool inside=endpoint>=domain->begin && endpoint<domain->end;
  if(inside) {
    if(endpoint==UINT64_MAX || excluded.size()>=512 || !heads.charge(2))return {};
    excluded.emplace_back(endpoint,endpoint+1);
  }
  const auto branch=heads.owner(select.children[1]);current=heads.owner(select.children[2]);
  return Step{endpoint,point,branch,inside,false};
}

} // namespace
} // namespace sentryward::equivalence
