#pragma once
#include "egraph_engine.hpp"

namespace sentryward::equivalence {
namespace {

// Thin actual-occurrence metadata only; Catalog deduplicates the exact
// bounds. No persistent source-domain identity is minted per case.
inline auto Engine::BodyScopeRegistry::bodyView(Id parent,Class fold) -> std::optional<ScopeView> {
  if(parent>=scopes.size() || fold>UINT32_MAX || fold>=heads.snapshot.size() || !heads.charge(4))return {};
  const auto& k=heads.snapshot[fold];const auto cls=heads.owner(fold);
  if(k.kind!=static_cast<int>(s::Kind::Fold) || !k.fold || k.fold->reducer!=s::Reducer::Sum ||
      !k.fold->count || k.children.size()!=2 || k.type!="real" ||
      !k.attributes.empty() || !k.provenance.empty() ||
      !heads.engine.finite_[cls] || !heads.engine.zero_[heads.owner(k.children[0])])return {};
  auto b=bounds.bindBody(scopes[parent].bounds,k.fold->binder_level,k.fold->count);if(!b)return {};
  ScopeView v;v.owner=this;v.epoch=context_epoch;v.bounds=*b;v.parent=parent;v.body_fold=fold;
  ++original_body_views;return v;
}

inline auto Engine::BodyScopeRegistry::domain(const ScopeView& v,std::uint32_t level) const -> std::optional<std::pair<unsigned,std::uint64_t>> {
  if(!valid(v))return {};
  if(level<scopes[v.parent].domains.size())return std::pair{0u,std::uint64_t(scopes[v.parent].domains[level])};
  if(v.body_fold && level==scopes[v.parent].domains.size()) {
    // Distinct ORIGINAL Fold BODY occurrences never compare equal merely
    // by binder/count. This value token cannot alias a resident domain ID.
    return std::pair{1u,(std::uint64_t(v.parent)<<32)|*v.body_fold};
  }
  return {};
}

inline auto Engine::BodyScopeRegistry::sameAssignment(Class cls,const ScopeView& reference,const ScopeView& actual,
    std::span<const std::uint32_t> covered) -> bool {
  if(!valid(reference) || !valid(actual))return false;
  cls=heads.owner(cls);const auto& free=heads.engine.free_indices_[cls];
  if(free.size()>64 || !heads.charge(free.size()+1))return false;
  for(auto level:free) {
    if(std::find(covered.begin(),covered.end(),level)!=covered.end())continue;
    auto r=domain(reference,level),a=domain(actual,level);
    auto rc=count(reference,level),ac=count(actual,level);
    if(!r || !a || r!=a || !rc || rc!=ac)return false;
  }
  return true;
}

inline auto Engine::BodyScopeRegistry::bodyUse(const ScopeView& v,std::span<const unsigned> path) -> std::optional<OccurrenceUse> {
  if(!valid(v) || !v.body_fold || path.size()>8 || !heads.charge(path.size()+2))return {};
  const auto& fold=heads.snapshot[*v.body_fold];Class current=heads.owner(fold.children[1]);
  std::array<Class,8> active{};unsigned used=0;
  for(auto edge:path) {
    if(std::find(active.begin(),active.begin()+used,current)!=active.begin()+used)return {};
    active[used++]=current;
    auto selected=heads.compact(current);if(!selected)return {};
    const auto& k=heads.snapshot[*selected];
    // A further actual Fold BODY requires its OWN binding/view. No path
    // may silently cross a binder or reuse an ancestor's occurrence token.
    if(k.kind==static_cast<int>(s::Kind::Fold) || edge>=k.children.size())return {};
    current=heads.owner(k.children[edge]);
  }
  if(std::find(active.begin(),active.begin()+used,current)!=active.begin()+used)return {};
  const auto& k=heads.snapshot[current];
  if(k.type!="real" || !k.provenance.empty() || !heads.engine.finite_[current])return {};
  OccurrenceUse use;use.scope=v;use.body=current;use.path_size=path.size();
  std::copy(path.begin(),path.end(),use.body_path.begin());return use;
}

inline auto Engine::BodyScopeRegistry::transport(const OccurrenceUse& use,Id canonical,std::uint32_t bound_axis) -> std::optional<Class> {
  if(!valid(use.scope) || !use.scope.body_fold || !heads.charge(3))return {};
  auto target=scopeView(canonical);if(!target)return {};
  const auto& fold=heads.snapshot[*use.scope.body_fold];
  const auto a=count(use.scope,bound_axis),r=count(*target,bound_axis);
  if(fold.fold->binder_level!=bound_axis || !a || a!=r ||
      bounds.rank(use.scope.bounds)!=bounds.rank(target->bounds) ||
      heads.engine.free_indices_[use.body].size()>8)return {};
  const std::array<std::uint32_t,1> mapped{bound_axis};
  // The SAME full matcher's identity-assignment guard. Only the actual
  // bound axis is explicitly unit/same-count mapped; EVERY captured axis
  // must retain its exact ORIGINAL occurrence and count.
  if(!sameAssignment(use.body,*target,use.scope,mapped))return {};
  ++transported_body_uses;return use.body;
}

inline auto Engine::BodyScopeRegistry::output() -> std::optional<Id> {
  const auto& o=heads.engine.options_;
  if(o.outer_binders>max_depth || o.outer_index_extents.size()!=o.outer_binders ||
      !heads.charge(o.outer_binders+1))return {};
  if(!scopes.empty())return Id{0};
  auto root_bounds=bounds.root(o.outer_binders,o.outer_index_extents);if(!root_bounds)return {};
  Scope root{*root_bounds,{}};
  for(auto count:o.outer_index_extents) {
    if(!count)return {};root.domains.push_back(next_domain++);
  }
  identity_slots+=root.domains.size()+1;
  scopes.push_back(std::move(root));return Id{0};
}

inline auto Engine::BodyScopeRegistry::insideFold(Id parent,Class head) -> std::optional<Id> {
  if(parent>=scopes.size() || head>=heads.snapshot.size() || !heads.charge())return {};
  const auto& k=heads.snapshot[head];
  if(k.kind!=static_cast<int>(s::Kind::Fold) || !k.fold ||
      k.children.size()!=2 || !k.fold->count ||
      !k.attributes.empty() || !k.provenance.empty())return {};
  const auto key=std::pair{parent,head};
  if(!heads.charge(1+std::bit_width(body_scopes.size()+1)))return {};
  if(auto found=body_scopes.find(key);found!=body_scopes.end())return found->second;
  auto nested=bind(parent,k.fold->binder_level,k.fold->count);if(!nested)return {};
  const auto id=static_cast<Id>(scopes.size());scopes.push_back(std::move(*nested));
  body_scopes.emplace(key,id);return id;
}

inline auto Engine::BodyScopeRegistry::bind(Id parent,std::uint32_t binder,std::uint64_t count) -> std::optional<Scope> {
  if(parent>=scopes.size() || !count || !retainedCapacity(scopes.size(),"occurrence_scope"))return {};
  const auto rank=bounds.rank(scopes[parent].bounds);
  if(!rank || *rank>max_depth)return {};
  if(identity_slots>max_identity_slots || *rank+2>max_identity_slots-identity_slots) {
    // TRACE-only first retained identity rejection. Read scalar metadata;
    // no expression/domain walk, proof charge, or admission change.
    if(!identity_cap_reported && sentryward::debug::environment("SENTRYWARD_EGRAPH_TRACE")) {
      identity_cap_reported=true;const auto stats=bounds.statistics();
      SENTRYWARD_DEBUG_PRINT(egraph_81);
    }
    return {};
  }
  if(!heads.charge(*rank+1))return {};
  auto next=bounds.bindBody(scopes[parent].bounds,binder,count);if(!next)return {};
  // Only <=8 source identity fields are copied; count storage is Catalog's.
  Scope nested{*next,scopes[parent].domains,parent};nested.domains.push_back(next_domain++);
  identity_slots+=nested.domains.size()+1;return nested;
}

inline auto Engine::BodyScopeRegistry::prefixScope(Id scope,std::uint32_t rank) -> std::optional<Id> {
  // Follow <=8 ACTUAL scope metadata edges. Never move a use into a
  // deeper scope: a future internal Fold binder could otherwise capture.
  for(unsigned depth=0;depth<=max_depth;++depth) {
    if(scope>=scopes.size() || !heads.charge())return {};
    const auto current=bounds.rank(scopes[scope].bounds);if(!current || *current<rank)return {};
    if(*current==rank)return scope;
    if(!scopes[scope].parent)return {};scope=*scopes[scope].parent;
  }
  return {};
}

inline auto Engine::BodyScopeRegistry::insideSourceFold(Id parent,const s::Graph& graph,unsigned graph_tag,s::Id head) -> std::optional<Id> {
  if(parent>=scopes.size() || head>=graph.size() || !heads.charge())return {};
  const auto& k=graph.node(head);
  if(k.kind!=s::Kind::Fold || !k.fold || k.operands.size()!=2 || !k.fold->count ||
      !k.attributes.empty())return {};
  const auto key=std::tuple{parent,graph_tag,head};
  if(!heads.charge(1+std::bit_width(source_scopes.size()+1)))return {};
  if(auto found=source_scopes.find(key);found!=source_scopes.end())return found->second;
  auto nested=bind(parent,k.fold->binder_level,k.fold->count);if(!nested)return {};
  const auto id=static_cast<Id>(scopes.size());scopes.push_back(std::move(*nested));source_scopes.emplace(key,id);return id;
}

inline auto Engine::BodyScopeRegistry::boundedAffine(const IndexAffine& affine,Id scope) -> bool {
  if(scope>=scopes.size() || affine.coefficients.size()>64 ||
      !heads.charge(affine.coefficients.size()+1))return false;
  auto maximum=affine.constant;
  for(auto [level,coefficient]:affine.coefficients) {
    const auto extent=count(scope,level);if(!extent)return false;
    std::uint64_t term;
    if(__builtin_mul_overflow(coefficient,*extent-1,&term) ||
        __builtin_add_overflow(maximum,term,&maximum))return false;
  }
  return true;
}

// Private geometry handle: the interval is read from an actual original
// BODY domain and a reachable pure exact-u64 unit-coordinate witness.
// No caller interval/extent or selected expression at widened bounds.
inline auto Engine::BodyScopeRegistry::image(Id scope,std::uint32_t level,Class coordinate) -> std::optional<ImageId> {
  if(scope>=scopes.size() ||
      !heads.charge(1+std::bit_width(image_keys.size()+1)))return {};
  coordinate=heads.owner(coordinate);const auto key=std::tuple{scope,level,coordinate};
  if(auto it=image_keys.find(key);it!=image_keys.end())return it->second;
  if(!imageCapacity("unit_image"))return {};
  auto affine=heads.affine(coordinate);auto extent=count(scope,level);
  if(!extent || !affine || affine->coefficients.size()!=1 ||
      !affine->coefficients.contains(level) || affine->coefficients.at(level)!=1 ||
      !boundedAffine(*affine,scope))return {};
  std::uint64_t end;if(__builtin_add_overflow(affine->constant,*extent,&end))return {};
  ImageId id;id.owner=this;id.id=images.size();
  images.push_back({scope,level,coordinate,affine->constant,end});image_keys.emplace(key,id);return id;
}

// Mint only from TWO admitted original outer unit images under the SAME
// actual lexical assignment. The difference is read from their checked
// affine constants, not guessed from a frame's declared begin. Neither
// coordinate has another free/captured coefficient, and both full ranges
// were checked for exact-u64 wrap by image().
inline auto Engine::BodyScopeRegistry::commonTranslation(ImageId source,ImageId actual) -> std::optional<CommonTranslationId> {
  auto s=view(source),a=view(actual);
  if(!s || !a || s->derived || a->derived || s->scope!=a->scope || s->level!=a->level ||
      scopes[s->scope].parent || s->end-s->begin!=a->end-a->begin ||
      s->end-s->begin!=count(s->scope,s->level) || !heads.charge(12))return {};
  const auto key=std::pair{source.id,actual.id};
  if(!heads.charge(1+std::bit_width(common_translation_keys.size()+1)))return {};
  if(auto f=common_translation_keys.find(key);f!=common_translation_keys.end())return f->second;
  if(!retainedCapacity(common_translations.size(),"common_translation") || !heads.charge(12))return {};
  CommonTranslationId id;id.owner=this;id.epoch=context_epoch;id.id=common_translations.size();
  common_translations.push_back({source,actual,s->begin>=a->begin,
      s->begin>=a->begin?s->begin-a->begin:a->begin-s->begin});
  common_translation_keys.emplace(key,id);return id;
}

// Keep the actual LOCAL unit witness. Only its private image interval is
// translated, with the SAME original outer assignment/captures, checked
// containment and no-wrap arithmetic. Future sampling still proves ALL
// P/Q/C numeric leaves through their original scopes and local assignment.
inline auto Engine::BodyScopeRegistry::commonImage(CommonTranslationId translation,ImageId local) -> std::optional<ImageId> {
  const auto* t=commonTranslation(translation);auto a=view(local);
  if(!t || !a || a->derived || !heads.charge(12))return {};
  // Copy before images append, so no borrowed vector element is retained.
  const auto output=images[t->actual.id],source=images[t->source.id];
  if(!prefix(output.scope,a->scope) || a->begin<output.begin || a->end>output.end ||
      a->end-a->begin!=count(a->scope,a->level))return {};
  // Identity changes neither the actual witness nor its bounds. All
  // owner/epoch/original-scope/domain/containment checks above still run;
  // do not allocate derived metadata or reroute existing point transport.
  if(!t->difference)return local;
  const auto key=std::pair{translation.id,local.id};
  if(!heads.charge(1+std::bit_width(common_image_keys.size()+1)))return {};
  if(auto f=common_image_keys.find(key);f!=common_image_keys.end())return f->second;
  if(!imageCapacity("common_image"))return {};
  std::uint64_t begin,end;
  if(t->add) {
    if(__builtin_add_overflow(a->begin,t->difference,&begin) ||
        __builtin_add_overflow(a->end,t->difference,&end))return {};
  } else {
    if(a->begin<t->difference || a->end<t->difference)return {};
    begin=a->begin-t->difference;end=a->end-t->difference;
  }
  if(begin>=end || begin<source.begin || end>source.end || !heads.charge(18))return {};
  ImageId id;id.owner=this;id.id=images.size();
  images.push_back({a->scope,a->level,a->coordinate,begin,end,true});
  common_image_keys.emplace(key,id);common_image_translations.emplace(id.id,translation.id);return id;
}

inline auto Engine::BodyScopeRegistry::commonImagePair(CommonTranslationId translation,ImageId a,ImageId b) -> bool {
  const auto* t=commonTranslation(translation);if(!t)return false;
  if(!t->difference) {
    // No derived record exists for identity images. Validate BOTH original
    // images through the same owned token; a foreign scope/shifted image
    // cannot become a common order-preserving assignment by equal length.
    auto x=commonImage(translation,a),y=commonImage(translation,b);
    return x && y && *x==a && *y==b;
  }
  if(a.owner!=this || b.owner!=this ||
      !heads.charge(3+2*std::bit_width(common_image_translations.size()+1)))return false;
  const auto x=common_image_translations.find(a.id),y=common_image_translations.find(b.id);
  return x!=common_image_translations.end() && y!=common_image_translations.end() &&
      x->second==translation.id && y->second==translation.id;
}

inline auto Engine::BodyScopeRegistry::point(ImageId image,Class coordinate,Id actual_scope) -> std::optional<PointId> {
  if(image.owner!=this || image.id>=images.size() || actual_scope>=scopes.size() ||
      !heads.charge())return {};
  coordinate=heads.owner(coordinate);
  const auto piece=images[image.id];auto global=heads.affine(coordinate);
  if(!global || !global->coefficients.empty() || !boundedAffine(*global,actual_scope) ||
      global->constant<piece.begin || global->constant>=piece.end)return {};
  // Both endpoints are checked above; subtraction cannot wrap. This
  // value handle is owned by the same immutable registry epoch. It creates
  // no retained per-update point table and copies only fixed scalar fields.
  PointId id;id.owner=this;id.image=image.id;id.coordinate=coordinate;
  id.actual_scope=actual_scope;id.local=global->constant-piece.begin;return id;
}

inline auto Engine::BodyScopeRegistry::globalValue(const ScalarPoint& point) -> std::optional<std::uint64_t> {
  if(auto literal=std::get_if<Class>(&point)) {
    if(*literal>=heads.snapshot.size())return {};
    auto value=heads.affine(*literal);
    return value && value->coefficients.empty()?std::optional{value->constant}:std::nullopt;
  }
  if(auto p=std::get_if<TypedPointId>(&point)) {
    if(p->owner!=this || p->epoch!=context_epoch || p->image>=images.size() ||
        p->cell>=heads.snapshot.size() || p->address>=heads.snapshot.size() ||
        p->scope>=scopes.size() || !heads.charge(8+std::bit_width(typed_time_sources.size()+1)))return {};
    auto source=typed_time_sources.find(p->image);if(source==typed_time_sources.end())return {};
    auto assignment=context(source->second.context);
    return assignment && assignment->actual_scope==p->scope?std::optional{p->value}:std::nullopt;
  }
  const auto& p=std::get<DispatchPointId>(point);
  if(p.owner!=this || p.epoch!=context_epoch || p.image>=images.size() ||
      p.root>=heads.snapshot.size() || p.branch>=heads.snapshot.size() || !heads.charge(5))return {};
  const auto& source=images[p.image];
  if(source.derived || p.value<source.begin || p.value>=source.end)return {};
  return p.value;
}

inline auto Engine::BodyScopeRegistry::pointWitness(const ScalarPoint& point) -> std::optional<Class> {
  if(auto literal=std::get_if<Class>(&point))
    return *literal<heads.snapshot.size()?std::optional{heads.owner(*literal)}:std::nullopt;
  if(!globalValue(point))return {};
  if(auto p=std::get_if<TypedPointId>(&point))return p->address;
  return images[std::get<DispatchPointId>(point).image].coordinate;
}

inline auto Engine::BodyScopeRegistry::point(ImageId image,const ScalarPoint& coordinate,Id actual_scope) -> std::optional<PointId> {
  if(auto literal=std::get_if<Class>(&coordinate))return point(image,*literal,actual_scope);
  if(auto source=std::get_if<TypedPointId>(&coordinate)) {
    auto global=globalValue(coordinate);
    if(!global || image.owner!=this || image.id>=images.size() || actual_scope>=scopes.size() ||
        !prefix(source->scope,actual_scope) || !heads.charge(6))return {};
    for(std::uint32_t axis=scopes[source->scope].domains.size();axis<scopes[actual_scope].domains.size();++axis)
      if(!heads.charge(2) || heads.depends(source->cell,axis) || heads.depends(source->address,axis))return {};
    if(!heads.charge(4+2*std::bit_width(typed_time_sources.size()+1)))return {};
    const auto piece=images[image.id];auto role=typed_time_sources.find(image.id);
    auto original=typed_time_sources.find(source->image);
    // Equal numeric point values alone never identify two time roles.
    // Every target tile must use the SAME actual anchor/header lattice.
    if(role==typed_time_sources.end() || original==typed_time_sources.end() ||
        role->second.anchor!=original->second.anchor || role->second.stride!=original->second.stride ||
        role->second.anchor_constant!=original->second.anchor_constant ||
        *global<piece.begin || *global>=piece.end)return {};
    auto a=context(role->second.context),b=context(original->second.context);
    if(!a || !b || a->actual_scope!=b->actual_scope)return {};
    PointId id;id.owner=this;id.image=image.id;id.coordinate=source->address;
    id.actual_scope=actual_scope;id.local=*global-piece.begin;id.dispatch=true;return id;
  }
  auto global=globalValue(coordinate);const auto& source=std::get<DispatchPointId>(coordinate);
  if(!global || image.owner!=this || image.id>=images.size() || actual_scope>=scopes.size() ||
      !prefix(images[source.image].scope,actual_scope) || !heads.charge(6))return {};
  const auto& piece=images[image.id];
  if(*global<piece.begin || *global>=piece.end)return {};
  PointId id;id.owner=this;id.image=image.id;id.coordinate=images[source.image].coordinate;
  id.actual_scope=actual_scope;id.local=*global-piece.begin;id.dispatch=true;return id;
}

inline auto Engine::BodyScopeRegistry::prefix(Id parent,Id child) -> bool {
  if(parent>=scopes.size() || child>=scopes.size() ||
      scopes[parent].domains.size()>scopes[child].domains.size() ||
      !heads.charge(scopes[parent].domains.size()+1))return false;
  for(std::uint32_t level=0;level<scopes[parent].domains.size();++level)
    if(scopes[parent].domains[level]!=scopes[child].domains[level] ||
        count(parent,level)!=count(child,level))return false;
  return true;
}

inline auto Engine::BodyScopeRegistry::pointIn(PointId p,Id reference_scope,Id actual_scope) -> std::optional<PointId> {
  if(p.owner!=this || p.image>=images.size() ||
      !prefix(reference_scope,images[p.image].scope) || !prefix(actual_scope,p.actual_scope))return {};
  const auto original=images[p.image];
  // These derived images, unlike a byte-address image, retain a proved
  // UNIT coordinate witness. They may rebase ONLY to original scope
  // prefixes via the same checked common assignment factory. Typed-cell
  // images below keep their stricter unchanged no-reinterpretation rule.
  auto translation=common_image_translations.end();
  if(original.derived && !common_image_translations.empty()) {
    if(!heads.charge(1+std::bit_width(common_image_translations.size()+1)))return {};
    translation=common_image_translations.find(p.image);
  }
  if(translation!=common_image_translations.end()) {
    CommonTranslationId token;token.owner=this;token.epoch=context_epoch;token.id=translation->second;
    auto local=image(reference_scope,original.level,original.coordinate);
    auto image_here=local?commonImage(token,*local):std::nullopt;
    if(!image_here)return {};
    if(!p.dispatch)return point(*image_here,p.coordinate,actual_scope);
    auto target=view(*image_here);std::uint64_t global;
    if(!target || p.local>=original.end-original.begin ||
        __builtin_add_overflow(original.begin,p.local,&global) || global<target->begin || global>=target->end ||
        !heads.charge(8))return {};
    PointId result=p;result.image=image_here->id;result.actual_scope=actual_scope;
    result.local=global-target->begin;return result;
  }
  if(p.dispatch) {
    if(!heads.charge(8))return {};
    // The private point already proves this scalar assignment, not the
    // value of its reachable source-coordinate witness. Rebase only the
    // owned reference image and actual ORIGINAL scope prefixes.
    std::optional<ImageId> image_here;
    if(original.derived) {
      if(reference_scope!=original.scope)return {};
      ImageId here;here.owner=this;here.id=p.image;image_here=here;
    } else image_here=image(reference_scope,original.level,original.coordinate);
    auto target=image_here?view(*image_here):std::nullopt;
    std::uint64_t global;
    if(!target || p.local>=original.end-original.begin ||
        __builtin_add_overflow(original.begin,p.local,&global) || global<target->begin || global>=target->end)return {};
    PointId result=p;result.image=image_here->id;result.actual_scope=actual_scope;
    result.local=global-target->begin;return result;
  }
  // Derived images retain their actual typed-cell proof. Never reinterpret
  // its byte-address witness as a unit graph coordinate in a new scope.
  if(original.derived) {
    if(reference_scope!=original.scope)return {};
    ImageId image_here;image_here.owner=this;image_here.id=p.image;
    return point(image_here,p.coordinate,actual_scope);
  }
  auto image_here=image(reference_scope,original.level,original.coordinate);
  return image_here?point(*image_here,p.coordinate,actual_scope):std::nullopt;
}

inline auto Engine::BodyScopeRegistry::validate(const Context& c) -> bool {
  if(heads.adapter || c.reference_scope>=scopes.size() || c.actual_scope>=scopes.size() ||
      c.axis_count>c.axes.size() || !heads.charge(c.axis_count+1))return false;
  const auto& reference=scopes[c.reference_scope];const auto& actual=scopes[c.actual_scope];
  std::optional<std::uint32_t> previous_reference;
  for(unsigned i=0;i<c.axis_count;++i) {
    ++coordinate_checks;
    const auto& a=c.axes[i];
    if(a.reference_level>=reference.domains.size() || !a.extent ||
        (previous_reference && a.reference_level<=*previous_reference))return false;
    previous_reference=a.reference_level;
    if(a.actual_shift && !a.actual_level)return false;
    const auto extent=count(a.actual_shift?c.actual_scope:c.reference_scope,
        a.actual_shift?*a.actual_level:a.reference_level);
    if(!extent || a.offset>=*extent || a.extent>*extent-a.offset)return false;
    std::optional<IndexAffine> coordinate;
    if(a.point) {
      if(a.coordinate_image || a.actual_level || a.actual_shift || a.point->owner!=this || a.point->image>=images.size() ||
          !heads.charge())return false;
      const auto& p=*a.point;const auto& image=images[p.image];
      if(image.scope!=c.reference_scope || image.level!=a.reference_level ||
          p.actual_scope!=c.actual_scope || p.coordinate!=heads.owner(a.actual_coordinate) ||
          p.local>=image.end-image.begin)return false;
      coordinate.emplace();coordinate->constant=p.local;
    } else if(a.coordinate_image) {
      auto image=view(*a.coordinate_image);
      if(!image || !a.actual_level ||
          image->scope!=(a.actual_shift?c.reference_scope:c.actual_scope) ||
          image->level!=(a.actual_shift?a.reference_level:*a.actual_level) ||
          image->coordinate!=heads.owner(a.actual_coordinate))return false;
      // Borrow a checked one-term logical image; no graph node is created
      // and the original narrow BODY scope is never widened.
      coordinate.emplace();coordinate->constant=image->begin;
      coordinate->coefficients.emplace(image->level,1);
    } else coordinate=heads.affine(a.actual_coordinate);
    if(!coordinate || !boundedAffine(*coordinate,a.actual_shift?c.reference_scope:c.actual_scope) ||
        coordinate->constant!=a.offset)return false;
    if(a.actual_level) {
      // Both axes are already live in their ORIGINAL lexical scopes.
      // Their semantic roles may be permuted (e.g. finite Sum interchange),
      // but the assignment must remain injective. Scope BODY extension
      // still requires a fresh binder beyond every existing live axis.
      for(unsigned j=0;j<i;++j)
        if(c.axes[j].actual_level==a.actual_level)return false;
      if(*a.actual_level>=actual.domains.size() ||
          a.extent!=count(a.actual_shift?c.reference_scope:c.actual_scope,
              a.actual_shift?a.reference_level:*a.actual_level) ||
          coordinate->coefficients.size()!=1 ||
          !coordinate->coefficients.contains(a.actual_shift?a.reference_level:*a.actual_level) ||
          coordinate->coefficients.at(a.actual_shift?a.reference_level:*a.actual_level)!=1)return false;
    } else if(a.extent!=1 || !coordinate->coefficients.empty())return false;
  }
  return true;
}

inline auto Engine::BodyScopeRegistry::lessContext(const Context& a,const Context& b) -> bool {
  // Intern construction is O(log stored contexts), not an uncharged hash
  // collision-chain walk. Every comparison is precharged and bounded.
  if(!heads.charge(4))throw ContextCap{};
  const auto ah=std::tie(a.reference_scope,a.actual_scope,a.axis_count);
  const auto bh=std::tie(b.reference_scope,b.actual_scope,b.axis_count);
  if(ah!=bh)return ah<bh;
  for(unsigned i=0;i<a.axis_count;++i) {
    if(!heads.charge(20))throw ContextCap{};
    const auto& x=a.axes[i];const auto& y=b.axes[i];
    const auto xh=std::tuple{x.reference_level,x.actual_level,x.offset,x.extent,x.actual_coordinate,
        x.actual_shift,bool(x.point),bool(x.coordinate_image)};
    const auto yh=std::tuple{y.reference_level,y.actual_level,y.offset,y.extent,y.actual_coordinate,
        y.actual_shift,bool(y.point),bool(y.coordinate_image)};
    if(xh!=yh)return xh<yh;
    // validate() admitted all private owners before interning. They are
    // this registry, so these complete varying owned fields finish the key.
    if(x.point) {
      if(!heads.charge(12))throw ContextCap{};
      const auto xp=std::tie(x.point->image,x.point->coordinate,x.point->actual_scope,x.point->local,x.point->dispatch);
      const auto yp=std::tie(y.point->image,y.point->coordinate,y.point->actual_scope,y.point->local,y.point->dispatch);
      if(xp!=yp)return xp<yp;
    }
    if(x.coordinate_image) {
      if(!heads.charge(2))throw ContextCap{};
      if(x.coordinate_image->id!=y.coordinate_image->id)return x.coordinate_image->id<y.coordinate_image->id;
    }
  }
  return false;
}

// Role order (time,state) is independent of lexical binder order. Keep
// semantic image/box arrays unchanged; canonicalize only the at-most-two
// assignment records so validation and memo keys retain one ordering.
inline auto Engine::BodyScopeRegistry::orderContextAxes(Context& context) -> bool {
  if(context.axis_count>context.axes.size())return false;
  if(context.axis_count==2 && context.axes[0].reference_level>context.axes[1].reference_level) {
    constexpr auto swap_work=3*((sizeof(Binding)+sizeof(Class)-1)/sizeof(Class));
    if(!heads.charge(swap_work))return false;
    std::swap(context.axes[0],context.axes[1]);
  }
  return true;
}

inline auto Engine::BodyScopeRegistry::admitContext(const Context& source) -> std::optional<ContextId> {
  if(source.axis_count>source.axes.size())return {};
  // Three bounded record copies and insertion metadata. Comparator work
  // below is charged separately, once during explicit construction.
  constexpr auto construction_work=3*((sizeof(Context)+sizeof(Class)-1)/sizeof(Class))+8;
  if(!heads.charge(construction_work))return {};
  Context key=source;
  for(unsigned i=key.axis_count;i<key.axes.size();++i)key.axes[i]=Binding{};
  if(!orderContextAxes(key) || !validate(key))return {};
  try {
    auto found=context_keys.lower_bound(key);ContextLess less{this};
    if(found!=context_keys.end() && !less(key,found->first))return found->second;
    if(!contextCapacity(contexts.size(),"owned_assignment"))return {};
    ContextId id;id.owner=this;id.epoch=context_epoch;id.id=contexts.size();
    // Insert first: a comparator cap cannot leave an unindexed record.
    context_keys.emplace_hint(found,key,id);contexts.push_back(std::move(key));return id;
  } catch(const ContextCap&) {return {};}
}

// Compile ONLY an exact original-image assignment. A memo hit reads a
// private ID, never reconstructs/copies/hashes/validates a full Context.
// Expression/leaf/capture checks still occur in the full body relation.
inline auto Engine::BodyScopeRegistry::sampleContext(ImageId time,std::optional<ImageId> state,
    Id actual_scope,ScalarPoint global_point,std::uint32_t state_axis) -> std::optional<ContextId> {
  if(time.owner!=this || time.id>=images.size() ||
      (state && (state->owner!=this || state->id>=images.size())) ||
      actual_scope>=scopes.size() || !heads.charge(9))return {};
  if(auto literal=std::get_if<Class>(&global_point)) {
    if(*literal>=heads.snapshot.size())return {};*literal=heads.owner(*literal);
  } else {
    const auto copies=std::visit([](const auto& p){return 2*((sizeof(p)+sizeof(Class)-1)/sizeof(Class));},global_point);
    if(!globalValue(global_point) || !heads.charge(copies))return {};
  }
  if(!state)state_axis=UINT32_MAX;
  const SampleKey key{time,state,actual_scope,global_point,state_axis};
  if(auto it=sample_contexts.find(key);it!=sample_contexts.end()) {++sample_context_hits;return it->second;}
  if(!contextCapacity(sample_contexts.size(),"sample_plan"))return {};
  auto t=view(time);auto local=point(time,global_point,actual_scope);if(!t || !local)return {};
  Context result{t->scope,actual_scope};result.axis_count=1;
  auto witness=pointWitness(global_point);if(!witness)return {};
  result.axes[0]={t->level,{},local->local,1,*witness,false,*local};
  if(state) {
    auto s=view(*state);
    if(!s || s->scope!=t->scope || s->level==t->level)return {};
    result.axis_count=2;
    result.axes[1]={s->level,state_axis,s->begin,s->end-s->begin,s->coordinate,true};
    if(s->derived)result.axes[1].coordinate_image=*state;
  }
  auto admitted=admitContext(result);if(!admitted || !heads.charge(8))return {};
  sample_contexts.emplace(key,*admitted);return admitted;
}

// Both maps are small checked exact-u64 summaries from the shared factory.
// Reference/actual namespaces remain separate, even when axis labels match.
inline auto Engine::BodyScopeRegistry::projectLeaf(IndexAffine& reference,IndexAffine& actual,const Context& initial,
    unsigned active,Id reference_scope,Id actual_scope,
    std::span<const std::pair<std::uint32_t,std::uint32_t>> lexical) -> bool {
  ++affine_checks;
  if(!boundedAffine(reference,reference_scope) || !boundedAffine(actual,actual_scope) ||
      !heads.charge((lexical.size()+initial.axis_count+1)*
          (1+std::bit_width(reference.coefficients.size()+actual.coefficients.size()+1))))return false;
  for(unsigned i=0;i<initial.axis_count;++i)if(active&(1u<<i)) {
    const auto& a=initial.axes[i];
    const auto found=reference.coefficients.find(a.reference_level);
    const auto coefficient=found==reference.coefficients.end()?0:found->second;
    std::uint64_t shift;
    if(__builtin_mul_overflow(coefficient,a.offset,&shift))return false;
    auto& shifted=a.actual_shift?actual.constant:reference.constant;
    if(__builtin_add_overflow(shifted,shift,&shifted))return false;
    reference.coefficients.erase(a.reference_level);
    if(a.actual_level) {
      const auto sample=actual.coefficients.find(*a.actual_level);
      if(coefficient!=(sample==actual.coefficients.end()?0:sample->second))return false;
      actual.coefficients.erase(*a.actual_level);
    }
  }
  for(auto [a,b]:lexical) {
    const auto ca=reference.coefficients.contains(a)?reference.coefficients.at(a):0;
    const auto cb=actual.coefficients.contains(b)?actual.coefficients.at(b):0;
    if(ca!=cb)return false;reference.coefficients.erase(a);actual.coefficients.erase(b);
  }
  return true;
}

inline auto Engine::BodyScopeRegistry::finishLeaf(const IndexAffine& reference,const IndexAffine& actual,
    Id reference_scope,Id actual_scope) -> bool {
  if(reference.constant!=actual.constant || reference.coefficients!=actual.coefficients)return false;
  for(auto [level,coefficient]:reference.coefficients) {
    if(!heads.charge() || level>=scopes[reference_scope].domains.size() ||
        level>=scopes[actual_scope].domains.size() ||
        scopes[reference_scope].domains[level]!=scopes[actual_scope].domains[level] ||
        count(reference_scope,level)!=count(actual_scope,level))return false;
  }
  return true;
}

inline auto Engine::BodyScopeRegistry::typedCellPair(Class piece,Class full) -> std::optional<std::pair<Class,Class>> {
  if(piece>=heads.snapshot.size() || full>=heads.snapshot.size())return {};
  piece=heads.owner(piece);full=heads.owner(full);
  if(piece>=heads.snapshot.size() || full>=heads.snapshot.size() || !heads.charge())return {};
  auto p=heads.compact(piece),f=heads.compact(full);if(!p || !f)return {};
  const auto& a=heads.snapshot[*p];const auto& b=heads.snapshot[*f];
  std::size_t metadata=0;
  for(std::string_view field:{a.type,a.name,a.attributes,a.provenance,b.type,b.name,b.attributes,b.provenance}) {
    if(field.size()>4096)return {};metadata+=field.size();
  }
  if(!heads.charge(metadata+2))return {};
  if(a.kind!=static_cast<int>(s::Kind::InitialCell) || b.kind!=a.kind ||
      !realType(a.type) || a.type!=b.type || a.name!=b.name || a.attributes!=b.attributes ||
      !a.provenance.empty() || !b.provenance.empty() || a.children.size()!=1 || b.children.size()!=1 ||
      !heads.engine.finite_[piece] || !heads.engine.finite_[full])return {};
  return std::pair{*p,*f};
}

// Relative time geometry anchored to TWO actual typed potential cells.
// The first source cell names zero only inside this private address
// lattice. No graph coordinate, widened time scope, or branch fact exists.
inline auto Engine::BodyScopeRegistry::typedTimeImage(Context known,Class piece,Class anchor,std::uint32_t time_axis) -> std::optional<ImageId> {
  if(piece>=heads.snapshot.size() || anchor>=heads.snapshot.size())return {};
  piece=heads.owner(piece);anchor=heads.owner(anchor);
  if(!validate(known))return {};
  for(unsigned i=0;i<known.axis_count;++i)
    if(known.axes[i].reference_level==time_axis)return {};
  auto ids=typedCellPair(piece,anchor);auto n=count(known.reference_scope,time_axis);
  if(!ids || !n)return {};
  auto context_id=admitContext(known);auto ordinal=context_id?contextOrdinal(*context_id):std::nullopt;
  if(!ordinal || !heads.charge(12+4*std::bit_width(typed_time_keys.size()+1)))return {};
  const auto key=std::tuple{piece,anchor,*ordinal,time_axis};
  if(auto found=typed_time_keys.find(key);found!=typed_time_keys.end())return found->second;
  if(!imageCapacity("typed_time_image"))return {};
  const auto& a=heads.snapshot[ids->first];const auto& b=heads.snapshot[ids->second];
  auto pa=heads.affine(a.children[0]),fa=heads.affine(b.children[0]);
  if(!pa || !fa || !projectLeaf(*pa,*fa,known,(1u<<known.axis_count)-1,
      known.reference_scope,known.actual_scope,{}) || !pa->coefficients.contains(time_axis))return {};
  const auto stride=pa->coefficients.at(time_axis);
  if(!stride || pa->constant<fa->constant || (pa->constant-fa->constant)%stride)return {};
  const auto begin=(pa->constant-fa->constant)/stride;std::uint64_t end;
  if(__builtin_add_overflow(begin,*n,&end))return {};
  const auto anchor_constant=fa->constant;
  pa->coefficients.erase(time_axis);pa->constant=fa->constant;
  if(!finishLeaf(*pa,*fa,known.reference_scope,known.actual_scope) || !heads.charge(24))return {};
  ImageId image;image.owner=this;image.id=images.size();
  images.push_back({known.reference_scope,time_axis,heads.owner(a.children[0]),begin,end,true});
  typed_time_sources.emplace(image.id,TypedTimeSource{piece,anchor,*context_id,stride,anchor_constant});
  typed_time_keys.emplace(key,image);return image;
}

inline auto Engine::BodyScopeRegistry::typedPoint(ImageId role,Class actual,Id actual_scope) -> std::optional<TypedPointId> {
  if(role.owner!=this || role.id>=images.size() || actual_scope>=scopes.size() ||
      !heads.charge(10+std::bit_width(typed_time_sources.size()+1)))return {};
  auto found=typed_time_sources.find(role.id);if(found==typed_time_sources.end())return {};
  const auto source=found->second;auto known=context(source.context);
  if(!known || known->actual_scope!=actual_scope)return {};
  if(actual>=heads.snapshot.size())return {};
  actual=heads.owner(actual);auto ids=typedCellPair(source.piece,actual);if(!ids)return {};
  const auto& a=heads.snapshot[ids->first];const auto& b=heads.snapshot[ids->second];
  auto pa=heads.affine(a.children[0]),fa=heads.affine(b.children[0]);
  if(!pa || !fa || !projectLeaf(*pa,*fa,*known,(1u<<known->axis_count)-1,
      known->reference_scope,known->actual_scope,{}) ||
      !pa->coefficients.contains(images[role.id].level) ||
      pa->coefficients.at(images[role.id].level)!=source.stride ||
      fa->constant<source.anchor_constant || (fa->constant-source.anchor_constant)%source.stride)return {};
  const auto value=(fa->constant-source.anchor_constant)/source.stride;
  pa->coefficients.erase(images[role.id].level);pa->constant=fa->constant;
  if(!finishLeaf(*pa,*fa,known->reference_scope,known->actual_scope) || !heads.charge(14))return {};
  TypedPointId point;point.owner=this;point.epoch=context_epoch;point.image=role.id;
  point.cell=actual;point.address=heads.owner(b.children[0]);point.scope=actual_scope;point.value=value;
  return point;
}

// A geometry-only image from ACTUAL typed cells, not an invented reachable
// Index+origin node. Known axes are validated by the SAME scoped affine
// substitution factory; remaining captures retain occurrence identities.
// Full typed body relation remains a separate mandatory proof obligation.
inline auto Engine::BodyScopeRegistry::imageFromTypedCells(Context known,Class piece,Class full,
    std::uint32_t piece_axis,std::uint32_t full_axis) -> std::optional<ImageId> {
  piece=heads.owner(piece);full=heads.owner(full);
  if(!validate(known) || piece>=heads.snapshot.size() || full>=heads.snapshot.size() ||
      !heads.charge())return {};
  const TypedImageKey key{piece,full,known,piece_axis,full_axis};
  if(auto it=typed_image_keys.find(key);it!=typed_image_keys.end())return it->second;
  if(!imageCapacity("typed_cell_image"))return {};
  for(unsigned i=0;i<known.axis_count;++i)
    if(known.axes[i].reference_level==piece_axis ||
        known.axes[i].actual_level==full_axis)return {};
  auto pair=typedCellPair(piece,full);if(!pair)return {};
  const auto& a=heads.snapshot[pair->first];const auto& b=heads.snapshot[pair->second];
  auto pa=heads.affine(a.children[0]),fa=heads.affine(b.children[0]);
  if(!pa || !fa || !projectLeaf(*pa,*fa,known,(1u<<known.axis_count)-1,
      known.reference_scope,known.actual_scope,{}))return {};
  auto pn=count(known.reference_scope,piece_axis),fn=count(known.actual_scope,full_axis);
  if(!pn || !fn || !pa->coefficients.contains(piece_axis) || !fa->coefficients.contains(full_axis))return {};
  const auto coefficient=pa->coefficients.at(piece_axis);
  if(!coefficient || coefficient!=fa->coefficients.at(full_axis) || pa->constant<fa->constant)return {};
  const auto difference=pa->constant-fa->constant;
  if(difference%coefficient)return {};
  const auto origin=difference/coefficient;
  if(origin>=*fn || *pn>*fn-origin)return {};
  pa->coefficients.erase(piece_axis);fa->coefficients.erase(full_axis);
  pa->constant=fa->constant;
  if(!finishLeaf(*pa,*fa,known.reference_scope,known.actual_scope))return {};
  ImageId id;id.owner=this;id.id=images.size();
  images.push_back({known.reference_scope,piece_axis,heads.owner(a.children[0]),origin,origin+*pn,true});
  if(!heads.charge(8))return {};
  typed_image_sources.emplace(id.id,TypedImageSource{full,known.actual_scope,full_axis});
  typed_image_keys.emplace(key,id);return id;
}

// Compose a proved local typed-cell image with the FULL cell's owned
// global coordinate image. Original BODY domains remain unchanged.
// Geometry alone still admits no P/Q or state-body equality.
inline auto Engine::BodyScopeRegistry::globalTypedImage(ImageId local,ImageId reference,Class full) -> std::optional<ImageId> {
  if(local.owner!=this || reference.owner!=this || local.id>=images.size() || reference.id>=images.size() ||
      !heads.charge(12+2*std::bit_width(translated_typed_images.size()+1)))return {};
  const auto key=std::pair{local.id,reference.id};
  auto source=typed_image_sources.find(local.id);if(source==typed_image_sources.end())return {};
  const auto& a=images[local.id];const auto& b=images[reference.id];
  if(!a.derived || source->second.full!=heads.owner(full) || source->second.scope!=b.scope ||
      source->second.axis!=b.level || b.end-b.begin!=count(b.scope,b.level) || a.end>b.end-b.begin)return {};
  if(auto it=translated_typed_images.find(key);it!=translated_typed_images.end())return it->second;
  if(!imageCapacity("global_typed_image"))return {};
  std::uint64_t begin,end;
  if(__builtin_add_overflow(a.begin,b.begin,&begin) || __builtin_add_overflow(a.end,b.begin,&end) ||
      begin>=end || end>b.end || !heads.charge(16))return {};
  ImageId id;id.owner=this;id.id=images.size();images.push_back({a.scope,a.level,a.coordinate,begin,end,true});
  translated_typed_images.emplace(key,id);return id;
}

inline auto Engine::BodyScopeRegistry::relativeImage(ImageId actual,ImageId reference) -> std::optional<ImageId> {
  if(actual.owner!=this || reference.owner!=this || actual.id>=images.size() || reference.id>=images.size() ||
      !heads.charge(10+2*std::bit_width(relative_image_keys.size()+1)))return {};
  const auto& a=images[actual.id];const auto& b=images[reference.id];
  if(a.begin<b.begin || a.end>b.end || a.end-a.begin!=count(a.scope,a.level) ||
      b.end-b.begin!=count(b.scope,b.level))return {};
  const auto key=std::pair{actual.id,reference.id};
  if(auto it=relative_image_keys.find(key);it!=relative_image_keys.end())return it->second;
  if(!imageCapacity("relative_image") || !heads.charge(12))return {};
  ImageId id;id.owner=this;id.id=images.size();
  images.push_back({a.scope,a.level,a.coordinate,a.begin-b.begin,a.end-b.begin,true});
  relative_image_keys.emplace(key,id);return id;
}

// Reuse owned geometry ONLY in an ACTUAL original descendant scope.
// The retained axis/count/capture identities cannot change. Both the
// physical coordinate and its original typed source use must be proved
// independent of EVERY added axis before a cache hit can be returned.
// This creates no graph coordinate or lexical scope and proves no value
// equality; the full scoped body relation still checks all numeric leaves.
inline auto Engine::BodyScopeRegistry::imageInOriginalScope(ImageId source,Id scope,Class source_use) -> std::optional<ImageId> {
  if(source_use>=heads.snapshot.size())return {};
  auto image=view(source);source_use=heads.owner(source_use);
  if(!image || source_use>=heads.snapshot.size() ||
      !prefix(image->scope,scope) || count(image->scope,image->level)!=count(scope,image->level) ||
      !heads.charge(6))return {};
  for(std::uint32_t axis=scopes[image->scope].domains.size();axis<scopes[scope].domains.size();++axis)
    if(!heads.charge(2) || heads.depends(image->coordinate,axis) || heads.depends(source_use,axis))return {};
  if(scope==image->scope)return source;
  const auto key=std::pair{source.id,scope};
  if(!heads.charge(1+std::bit_width(original_scope_image_keys.size()+1)))return {};
  if(auto found=original_scope_image_keys.find(key);found!=original_scope_image_keys.end())return found->second;
  if(!imageCapacity("original_scope_image") || !heads.charge(12))return {};
  ImageId id;id.owner=this;id.id=images.size();
  images.push_back({scope,image->level,image->coordinate,image->begin,image->end,image->derived});
  original_scope_image_keys.emplace(key,id);return id;
}

} // namespace
} // namespace sentryward::equivalence
