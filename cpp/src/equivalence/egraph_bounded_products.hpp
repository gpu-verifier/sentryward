#pragma once
#include "egraph_engine.hpp"

namespace sentryward::equivalence {
namespace {

inline auto Engine::BoundedProducts::related(Use r,Use a,const Registry::Context& context) -> bool {
  // Resident, complete-key proof memo; eviction never resets the shared
  // allowance. Original scopes and private point/image handles remain in
  // equality even though the hash is only a cheap indexing discriminator.
  if(!registry.heads.charge(24))return false; // Two fixed complete-context hash slots.
  const RelationKey key{r,a,context};
  return memoRelation(key,[&] {
    return bool(registry.heads.engine.affineBodyMapWithHeads(r.body,a.body,UINT32_MAX,
        registry.heads.snapshot,false,registry.heads.remaining,registry.heads,UINT32_MAX,&registry,&context));
  });
}

inline auto Engine::BoundedProducts::related(Use r,Use a,Registry::ContextId id) -> bool {
  const auto* context=registry.context(id);
  if(!context || r.scope!=context->reference_scope || a.scope!=context->actual_scope ||
      !registry.heads.charge(12))return false;
  // Two actual Uses plus an opaque owned ID: no Context record copy,
  // complete-field hash or initial coordinate validation on a memo hit.
  const RelationKey key{r,a,id};
  return memoRelation(key,[&] {
    return bool(registry.heads.engine.affineBodyMapWithHeads(r.body,a.body,UINT32_MAX,
        registry.heads.snapshot,false,registry.heads.remaining,registry.heads,UINT32_MAX,&registry,context,&id));
  });
}

inline auto Engine::BoundedProducts::common(std::span<const Use> uses) -> std::optional<Registry::Id> {
  if(uses.empty() || uses.size()>8 || uses.front().scope>=registry.scopes.size())return {};
  auto deepest=uses.front().scope;
  for(const auto& use:uses) {
    if(use.scope>=registry.scopes.size() || !registry.heads.charge())return {};
    if(registry.scopes[use.scope].domains.size()>registry.scopes[deepest].domains.size())deepest=use.scope;
  }
  for(const auto& use:uses)if(!registry.prefix(use.scope,deepest))return {};
  return deepest;
}

inline auto Engine::BoundedProducts::admitted(Use use) -> bool {
  use.body=registry.heads.owner(use.body);
  if(use.scope>=registry.scopes.size() || use.body>=registry.heads.snapshot.size() ||
      !registry.heads.charge())return false;
  const auto& k=registry.heads.snapshot[use.body];
  if(!realType(k.type) || !k.provenance.empty() || !registry.heads.engine.finite_[use.body])return false;
  const auto& free=registry.heads.engine.free_indices_[use.body];
  if(free.size()>8 || !registry.heads.charge(free.size()+1))return false;
  for(auto level:free)if(level>=registry.scopes[use.scope].domains.size())return false;
  return true;
}

inline auto Engine::BoundedProducts::factors(std::span<const Use> uses,bool strict_real_products) -> std::optional<std::vector<Use>> {
  if(uses.empty() || uses.size()>8 || !registry.heads.charge(uses.size()))return {};
  struct Task {Use use;bool ready=false;};std::vector<Task> tasks;
  for(auto use:uses)tasks.push_back({use});
  std::unordered_set<Class> active;std::vector<Use> result;std::size_t steps=0;
  while(!tasks.empty()) {
    if(++steps>48 || !registry.heads.charge())return {};
    auto task=tasks.back();tasks.pop_back();task.use.body=registry.heads.owner(task.use.body);
    if(task.ready){active.erase(task.use.body);continue;}
    if(active.contains(task.use.body) || !admitted(task.use))return {};
    auto head=registry.heads.compact(task.use.body);if(!head)return {};
    const auto& k=registry.heads.snapshot[*head];
    if(k.type!=registry.heads.snapshot[task.use.body].type || !k.provenance.empty())return {};
    if(k.kind==static_cast<int>(s::Kind::Operation) && k.name=="Multiply") {
      // Numeric projection's existing marker is an admitted ideal
      // arithmetic head, not an unknown modifier. Other modifiers remain
      // opaque/Unknown; leaf encodings are compared by the full mapper.
      if(!arithmetic(k) || k.children.size()!=2 || (strict_real_products && k.type!="real"))return {};
      active.insert(task.use.body);tasks.push_back({task.use,true});
      tasks.push_back({{k.children[0],task.use.scope}});tasks.push_back({{k.children[1],task.use.scope}});
    } else {
      // Existing exact ideal unit identity. No rounded coefficient, SAT or
      // arbitrary conversion is discarded; unknown constants stay factors.
      if(k.kind==static_cast<int>(s::Kind::Constant) && k.name=="1" && k.attributes.empty())continue;
      if(result.size()==8)return {};result.push_back(task.use);
    }
  }
  return result;
}

inline auto Engine::BoundedProducts::activeAxes(Use r,Use a,const Registry::Context& context) -> std::optional<unsigned> {
  if(!registry.prefix(r.scope,context.reference_scope) || !registry.prefix(a.scope,context.actual_scope))return {};
  if(!registry.heads.charge(2+12*context.axis_count))return {};
  unsigned active=0;
  for(unsigned i=0;i<context.axis_count;++i) {
    const auto& axis=context.axes[i];
    const bool ref_present=bool(registry.count(r.scope,axis.reference_level));
    const bool actual_present=!axis.actual_level || bool(registry.count(a.scope,*axis.actual_level));
    const bool used=registry.heads.depends(r.body,axis.reference_level) ||
        (axis.actual_level && registry.heads.depends(a.body,*axis.actual_level));
    // A proved dependency ABSENCE on both original factor uses permits
    // removing an irrelevant assignment, even when that axis is present.
    // This is checked per expression, never inferred from cached geometry.
    if(!used)continue;
    if(!ref_present || !actual_present)return {};
    active|=1u<<i;
  }
  return active;
}

inline auto Engine::BoundedProducts::contextFor(Use r,Use a,const Registry::Context& context) -> std::optional<Registry::Context> {
  auto active=activeAxes(r,a,context);if(!active)return {};
  Registry::Context result{r.scope,a.scope};
  for(unsigned i=0;i<context.axis_count;++i)if(*active&(1u<<i)) {
    auto axis=context.axes[i];
    if(axis.point) {
      auto point=registry.pointIn(*axis.point,r.scope,a.scope);if(!point)return {};axis.point=*point;
    }
    result.axes[result.axis_count++]=axis;
  }
  return result;
}

inline auto Engine::BoundedProducts::contextFor(Use r,Use a,Registry::ContextId parent) -> std::optional<Registry::ContextId> {
  const auto* context=registry.context(parent);
  if(!context || !registry.prefix(r.scope,context->reference_scope) ||
      !registry.prefix(a.scope,context->actual_scope))return {};
  // These are EXPRESSION checks, not cached geometry. An already minted
  // subcontext cannot authorize a different factor to cross a live axis.
  auto active=activeAxes(r,a,*context);if(!active)return {};
  if(r.scope==context->reference_scope && a.scope==context->actual_scope &&
      *active==((1u<<context->axis_count)-1))return parent;
  auto ordinal=registry.contextOrdinal(parent);if(!ordinal ||
      !registry.heads.charge(6+4*std::bit_width(prefix_contexts.size()+1)))return {};
  const auto key=std::tuple{*ordinal,r.scope,a.scope,*active};
  if(auto found=prefix_contexts.find(key);found!=prefix_contexts.end()) {
    ++registry.context_prefix_hits;return found->second;
  }
  if(!registry.contextCapacity(prefix_contexts.size(),"factor_prefix_plan"))return {};
  // Parent records have stable deque storage through this append. The
  // existing contextFor validates actual prefixes and rebases owned points;
  // admission checks the complete newly derived original-scope geometry.
  auto derived=contextFor(r,a,*context);if(!derived)return {};
  auto admitted=registry.admitContext(*derived);if(!admitted)return {};
  prefix_contexts.emplace(key,*admitted);return admitted;
}

// Functional assignment transport, not body/family identity by shape.
// The returned value retains the exact ORIGINAL use and the full scoped
// proof context. Internal future Fold binders keep the ORIGINAL rank.
inline auto Engine::BoundedProducts::canonicalUse(Use original,Registry::ContextId assignment,bool coordinate) -> std::optional<AssignedUse> {
  const auto* context=registry.context(assignment);
  if(!context || original.scope>=registry.scopes.size() || original.body>=registry.heads.snapshot.size() ||
      !registry.heads.charge(context->axis_count+2))return {};
  for(unsigned i=0;i<context->axis_count;++i) {
    const auto& axis=context->axes[i];
    if(!axis.actual_level || *axis.actual_level!=axis.reference_level || axis.offset || axis.actual_shift ||
        axis.point || axis.coordinate_image ||
        registry.count(context->reference_scope,axis.reference_level)!=std::optional{axis.extent} ||
        registry.count(context->actual_scope,*axis.actual_level)!=std::optional{axis.extent})return {};
  }
  original.body=registry.heads.owner(original.body);
  if(coordinate) {
    const auto& k=registry.heads.snapshot[original.body];
    if((k.type!="u64" && k.type!="exact:u64") || !k.provenance.empty())return {};
    auto affine=registry.heads.affine(original.body);
    if(!affine || !registry.boundedAffine(*affine,original.scope))return {};
  } else if(!admitted(original))return {};
  const auto rank=registry.bounds.rank(registry.scopes[original.scope].bounds);if(!rank)return {};
  auto target=registry.prefixScope(context->reference_scope,*rank);if(!target)return {};
  const Use canonical{original.body,*target};
  auto local=contextFor(canonical,original,assignment);
  if(!local || !related(canonical,original,*local) || !registry.heads.charge(6))return {};
  AssignedUse result;result.owner=this;result.epoch=registry.context_epoch;
  result.original=original;result.canonical=canonical;result.context=*local;return result;
}

inline auto Engine::BoundedProducts::group(std::span<const Use> input,bool strict_real_products) -> std::optional<Groups> {
  auto normalized=factors(input,strict_real_products);if(!normalized || !registry.heads.charge(2))return {};
  Groups result;result.factor_count=normalized->size();
  for(auto use:*normalized) {
    auto tag=registry.heads.shallow(use.body);if(!tag)return {};
    std::size_t i=0;
    for(;i<result.size;++i) {
      if(!registry.heads.charge())return {};
      if(result.data[i].tag==*tag)break;
    }
    if(i<result.size) {
      if(result.data[i].use!=use || !registry.heads.charge())return {};
      ++result.data[i].copies;
    } else {
      if(result.size==result.data.size() || !registry.heads.charge(4))return {};
      result.data[result.size++]={*tag,use,1};
    }
  }
  if(!registry.heads.charge(4*result.size))return {}; // Active-only optional return copy.
  return result;
}

inline auto Engine::BoundedProducts::pairedFactors(Use reference,Use actual) -> std::optional<std::vector<FactorPair>> {
  // SAME selected <=8-factor factory and multiplicity-preserving matching,
  // exposed as actual Uses, not a relation proof or fabricated Context.
  // Physical Engine properties may not admit an adapter-owned namespace.
  if(registry.heads.adapter)return {};
  // This new ideal-real paired route may not reassociate a raw rounded
  // f32/f64 product nested beneath an explicit real parent. Keep the SAME
  // selected factor parser; existing callers retain their original policy.
  auto r=group(std::span(&reference,1),true),a=group(std::span(&actual,1),true);
  if(!r || !a || !registry.heads.charge(4*r->size+1))return {};
  std::vector<FactorPair> result;result.reserve(r->size);
  if(!pairGroups(*r,*a,[&](Use p,Use q){result.emplace_back(p,q);return true;}))return {};
  return result;
}

inline auto Engine::BoundedProducts::equal(std::span<const Use> reference,
    std::span<const Use> actual,const Registry::Context& context) -> std::optional<std::pair<std::size_t,std::size_t>> {
  if(!registry.validate(context))return {};
  auto plan=referenceProduct(reference);if(!plan)return {};
  return equalReference(*plan,actual,[&](Use r,Use a) {
    auto local=contextFor(r,a,context);return local && related(r,a,*local);
  });
}

inline auto Engine::BoundedProducts::equal(ReferenceId reference,
    std::span<const Use> actual,Registry::ContextId context) -> std::optional<std::pair<std::size_t,std::size_t>> {
  if(!registry.context(context))return {};
  return equalReference(reference,actual,[&](Use r,Use a) {
    auto local=contextFor(r,a,context);return local && related(r,a,*local);
  });
}

} // namespace
} // namespace sentryward::equivalence
