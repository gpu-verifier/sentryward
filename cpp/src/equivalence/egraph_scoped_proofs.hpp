#pragma once
#include "egraph_engine.hpp"

namespace sentryward::equivalence {
namespace {

inline auto Engine::result(Status status, std::string reason) const -> Result {
  auto stats = statistics_;
  stats.nodes = nodes_.size();
  if(options_.certified_branch_scope && sentryward::debug::environment("SENTRYWARD_EGRAPH_TRACE"))
    SENTRYWARD_DEBUG_PRINT(egraph_3);
  return {status, std::move(reason), stats,branch_scope_used_ && options_.certified_branch_scope?
      options_.certified_branch_scope->digest():std::string{}};
}

inline auto Engine::scopedBodyProof(Class reference_wrapper,Class actual_wrapper,
    Class reference,Class actual,const s::Graph& reference_graph,std::span<const s::Id> reference_folds,
    const s::Graph& actual_graph,std::span<const s::Id> actual_folds,
    std::span<const ScopedAxisBinding> bindings,std::span<const Class> coordinates,
    std::span<const std::optional<Class>> image_coordinates) -> ScopedBodyResult {
  ScopedBodyResult result;
  if(!options_.ideal_numeric_outputs || bindings.size()>2 || coordinates.size()!=bindings.size() ||
      (!image_coordinates.empty() && image_coordinates.size()!=bindings.size())) {
    result.reason="unsupported_scoped_context";return result;
  }
  if(!numericOutput(reference_wrapper) || !numericOutput(actual_wrapper)) {
    result.reason="unsupported_numeric_root";return result;
  }
  reference_wrapper=project(reference_wrapper,true);actual_wrapper=project(actual_wrapper,true);
  if(!realType(nodes_[find(reference_wrapper)].type) || !realType(nodes_[find(actual_wrapper)].type)) {
    result.reason="unsupported_root_policy";return result;
  }
  reference=project(reference,true);actual=project(actual,true);propagateProperties();
  if(!realType(nodes_[find(reference)].type) || nodes_[find(reference)].type!=nodes_[find(actual)].type ||
      !finite_[find(reference)] || !finite_[find(actual)]) {
    result.reason="nonfinite_or_exact_body";return result;
  }
  // One epoch catalog over stored heads, not per query root reachability sets.
  std::vector<std::vector<Class>> members(nodes_.size());
  for(Class id=0;id<nodes_.size();++id){poll();members[find(id)].push_back(id);}
  std::size_t remaining=131072;AffineCache affine;
  BodyHeadIndex heads(*this,nodes_,members,affine,&remaining);BodyScopeRegistry registry(heads);
  auto root=registry.output();
  if(!root){result.reason="outer_domain_unknown";return result;}
  auto reference_scope=*root,actual_scope=*root;
  for(auto head:reference_folds) {
    auto next=registry.insideSourceFold(reference_scope,reference_graph,0,head);
    if(!next){result.reason="reference_scope_unknown";return result;}reference_scope=*next;
  }
  for(auto head:actual_folds) {
    auto next=registry.insideSourceFold(actual_scope,actual_graph,1,head);
    if(!next){result.reason="actual_scope_unknown";return result;}actual_scope=*next;
  }
  BodyScopeRegistry::Context context{reference_scope,actual_scope};
  context.axis_count=static_cast<unsigned>(bindings.size());
  for(unsigned i=0;i<context.axis_count;++i) {
    const auto& b=bindings[i];
    const auto extent=b.actual_level?registry.count(b.actual_shift?reference_scope:actual_scope,
        b.actual_shift?b.reference_level:*b.actual_level):std::optional<std::uint64_t>{1};
    if(!extent){result.reason="binding_domain_unknown";return result;}
    context.axes[i]={b.reference_level,b.actual_level,b.offset,*extent,coordinates[i],b.actual_shift};
    if(!image_coordinates.empty() && image_coordinates[i]) {
      if(b.actual_level || b.actual_shift){result.reason="invalid_derived_point";return result;}
      auto image=registry.image(reference_scope,b.reference_level,*image_coordinates[i]);
      auto point=image?registry.point(*image,coordinates[i],actual_scope):std::nullopt;
      if(!point){result.reason="derived_point_unknown";return result;}
      context.axes[i].point=*point;
    }
  }
  const auto matched=affineBodyMapWithHeads(reference,actual,UINT32_MAX,nodes_,false,&remaining,heads,
      UINT32_MAX,&registry,&context);
  result.proved=bool(matched);result.reason=result.proved?"scoped_body_certificate":
      (!remaining || heads.exhausted?"scoped_body_local_cap":"scoped_body_unknown");
  result.body_work=131072-remaining;result.scope_identities=registry.scopes.size();
  result.coordinate_checks=registry.coordinate_checks;result.affine_checks=registry.affine_checks;return result;
}

inline auto Engine::joinedBodyProof(Class reference_wrapper,Class reference,
    const s::Graph& reference_graph,std::span<const s::Id> reference_folds,
    std::span<const std::uint32_t> reference_axes,
    const s::Graph& actual_graph,std::span<const Class> actual_wrappers,
    std::span<const Class> bodies,std::span<const std::vector<s::Id>> folds,
    std::span<const std::array<Class,2>> coordinates,std::span<const JoinedBodyPiece> requests) -> JoinedBodyResult {
  JoinedBodyResult result;
  if(!options_.ideal_numeric_outputs || reference_axes.empty() || reference_axes.size()>2 ||
      requests.empty() || requests.size()>1024 || bodies.size()!=requests.size() ||
      actual_wrappers.size()!=requests.size() || folds.size()!=requests.size() ||
      coordinates.size()!=requests.size()) {result.reason="invalid_join_configuration";return result;}
  if(!numericOutput(reference_wrapper)) {result.reason="unsupported_numeric_root";return result;}
  reference_wrapper=project(reference_wrapper,true);reference=project(reference,true);
  std::vector<Class> numeric_bodies;numeric_bodies.reserve(bodies.size());
  for(std::size_t i=0;i<bodies.size();++i) {
    poll();if(!numericOutput(actual_wrappers[i])) {result.reason="unsupported_numeric_root";return result;}
    auto wrapper=project(actual_wrappers[i],true);
    if(!realType(nodes_[find(wrapper)].type)){result.reason="unsupported_root_policy";return result;}
    numeric_bodies.push_back(project(bodies[i],true));
  }
  // Coordinate witnesses and numeric cell addresses must use the SAME exact
  // projection before the immutable certificate epoch is constructed.
  std::vector<std::array<Class,2>> exact_coordinates(coordinates.begin(),coordinates.end());
  for(std::size_t i=0;i<requests.size();++i)
    for(unsigned j=0;j<requests[i].axis_count && j<2;++j)
      exact_coordinates[i][j]=project(exact_coordinates[i][j],false);
  propagateProperties();
  if(!realType(nodes_[find(reference_wrapper)].type) || !realType(nodes_[find(reference)].type) ||
      !finite_[find(reference)]) {result.reason="nonfinite_or_exact_body";return result;}
  std::vector<std::vector<Class>> members(nodes_.size());
  for(Class id=0;id<nodes_.size();++id){poll();members[find(id)].push_back(id);}
  std::size_t remaining=131072;AffineCache affine;
  BodyHeadIndex heads(*this,nodes_,members,affine,&remaining);BodyScopeRegistry registry(heads);
  auto root=registry.output();if(!root){result.reason="outer_domain_unknown";return result;}
  auto reference_scope=*root;
  for(auto head:reference_folds) {
    auto next=registry.insideSourceFold(reference_scope,reference_graph,0,head);
    if(!next){result.reason="reference_scope_unknown";return result;}reference_scope=*next;
  }
  std::array<std::pair<std::uint64_t,std::uint64_t>,2> domain{};
  for(std::size_t i=0;i<reference_axes.size();++i) {
    auto extent=registry.count(reference_scope,reference_axes[i]);
    if(!extent || (i && reference_axes[i]==reference_axes[i-1])) {
      result.reason="join_domain_unknown";return result;
    }
    domain[i]={0,*extent};
  }
  JoinedFamilies catalog(registry);
  std::vector<JoinedFamilies::Piece> pieces;pieces.reserve(requests.size());
  for(std::size_t i=0;i<requests.size();++i) {
    if(!heads.charge()){result.reason="join_local_cap";return result;}
    const auto& request=requests[i];auto scope=*root;
    if(request.axis_count!=reference_axes.size()){result.reason="join_axis_mismatch";return result;}
    for(auto head:folds[i]) {
      auto next=registry.insideSourceFold(scope,actual_graph,1,head);
      if(!next){result.reason="actual_scope_unknown";return result;}scope=*next;
    }
    JoinedFamilies::Piece piece;piece.body=numeric_bodies[i];piece.scope=scope;
    for(unsigned j=0;j<request.axis_count;++j) {
      auto image=registry.image(scope,request.axes[j].level,exact_coordinates[i][j]);
      if(!image && request.axis_count==1) {
        // Rank-one typed-cell fallback: the caller still supplies an actual
        // reachable byte-offset witness. Geometry cannot bypass the later
        // full joined-body certificate or introduce a graph coordinate.
        auto head=heads.compact(piece.body);
        if(head && heads.snapshot[*head].kind==static_cast<int>(s::Kind::InitialCell) &&
            heads.snapshot[*head].children.size()==1 &&
            heads.owner(heads.snapshot[*head].children[0])==heads.owner(exact_coordinates[i][j]))
          image=registry.imageFromTypedCells({scope,reference_scope},piece.body,reference,
              request.axes[j].level,reference_axes[j]);
      }
      if(!image){result.reason="join_image_unknown";return result;}piece.images[j]=*image;
    }
    pieces.push_back(std::move(piece));
  }
  auto family=catalog.join(pieces,static_cast<unsigned>(reference_axes.size()),domain);
  if(family && catalog.uniform(*family,reference,reference_scope,reference_axes)) {
    result.proved=true;result.reason="joined_body_certificate";result.pieces=pieces.size();
  } else result.reason=!remaining || heads.exhausted || catalog.exhausted()?"join_local_cap":"join_unknown";
  result.body_work=131072-remaining;result.coverage_work=catalog.coverageWork();
  result.scope_identities=registry.scopes.size();return result;
}

inline auto Engine::productBodyProof(const s::Graph& reference_graph,
    std::span<const Class> reference_wrappers,std::span<const Class> reference_bodies,
    std::span<const std::vector<s::Id>> reference_folds,const s::Graph& actual_graph,
    std::span<const Class> actual_wrappers,std::span<const Class> actual_bodies,
    std::span<const std::vector<s::Id>> actual_folds,std::span<const ScopedAxisBinding> bindings,
    std::span<const Class> coordinates,std::span<const std::optional<Class>> image_coordinates) -> ScopedProductResult {
  ScopedProductResult result;
  if(!options_.ideal_numeric_outputs || reference_bodies.empty() || actual_bodies.empty() ||
      reference_bodies.size()>8 || actual_bodies.size()>8 || bindings.size()>2 ||
      reference_wrappers.size()!=reference_bodies.size() || reference_folds.size()!=reference_bodies.size() ||
      actual_wrappers.size()!=actual_bodies.size() || actual_folds.size()!=actual_bodies.size() ||
      coordinates.size()!=bindings.size() || image_coordinates.size()!=bindings.size()) {
    result.reason="invalid_product_configuration";return result;
  }
  const auto projectBodies=[&](std::span<const Class> wrappers,std::span<const Class> bodies) {
    std::vector<Class> result;result.reserve(bodies.size());
    for(std::size_t i=0;i<bodies.size();++i) {
      poll();if(!numericOutput(wrappers[i]))return std::optional<std::vector<Class>>{};
      auto wrapper=project(wrappers[i],true);
      if(!realType(nodes_[find(wrapper)].type))return std::optional<std::vector<Class>>{};
      result.push_back(project(bodies[i],true));
    }
    return std::optional{std::move(result)};
  };
  auto rb=projectBodies(reference_wrappers,reference_bodies),ab=projectBodies(actual_wrappers,actual_bodies);
  if(!rb || !ab){result.reason="unsupported_numeric_root";return result;}propagateProperties();
  std::vector<std::vector<Class>> members(nodes_.size());
  for(Class id=0;id<nodes_.size();++id){poll();members[find(id)].push_back(id);}
  std::size_t remaining=131072;AffineCache affine;
  BodyHeadIndex heads(*this,nodes_,members,affine,&remaining);BodyScopeRegistry registry(heads);
  auto root=registry.output();if(!root){result.reason="outer_domain_unknown";return result;}
  const auto uses=[&](const s::Graph& graph,unsigned tag,std::span<const Class> bodies,
      std::span<const std::vector<s::Id>> folds) {
    std::vector<BoundedProducts::Use> result;result.reserve(bodies.size());
    for(std::size_t i=0;i<bodies.size();++i) {
      auto scope=*root;
      for(auto head:folds[i]) {
        auto next=registry.insideSourceFold(scope,graph,tag,head);
        if(!next)return std::optional<std::vector<BoundedProducts::Use>>{};scope=*next;
      }
      result.push_back({bodies[i],scope});
    }
    return std::optional{std::move(result)};
  };
  auto r=uses(reference_graph,0,*rb,reference_folds),a=uses(actual_graph,1,*ab,actual_folds);
  if(!r || !a){result.reason="product_scope_unknown";return result;}
  BoundedProducts factory(registry);auto rs=factory.common(*r),as=factory.common(*a);
  if(!rs || !as){result.reason="incompatible_product_scopes";return result;}
  BodyScopeRegistry::Context context{*rs,*as};context.axis_count=static_cast<unsigned>(bindings.size());
  for(unsigned i=0;i<context.axis_count;++i) {
    const auto& b=bindings[i];
    const auto extent=b.actual_level?registry.count(b.actual_shift?*rs:*as,
        b.actual_shift?b.reference_level:*b.actual_level):std::optional<std::uint64_t>{1};
    if(!extent){result.reason="binding_domain_unknown";return result;}
    context.axes[i]={b.reference_level,b.actual_level,b.offset,*extent,coordinates[i],b.actual_shift};
    if(image_coordinates[i]) {
      if(b.actual_level || b.actual_shift){result.reason="invalid_derived_point";return result;}
      auto image=registry.image(*rs,b.reference_level,*image_coordinates[i]);
      auto point=image?registry.point(*image,coordinates[i],*as):std::nullopt;
      if(!point){result.reason="derived_point_unknown";return result;}context.axes[i].point=*point;
    }
  }
  // Compile this ORIGINAL assignment once. Product factors retain their
  // own source paths/scopes and use only owned prefix-derived IDs below.
  auto admitted=registry.admitContext(context);
  auto proof=admitted?factory.equal(*r,*a,*admitted):std::nullopt;
  result.proved=bool(proof);result.reason=result.proved?"scoped_product_certificate":
      (!remaining || heads.exhausted?"product_local_cap":"product_unknown");
  if(proof){result.reference_factors=proof->first;result.actual_factors=proof->second;}
  result.body_work=131072-remaining;result.scope_identities=registry.scopes.size();
  result.context_entries=registry.contexts.size();result.context_id_reads=registry.context_reads;
  result.context_prefix_hits=registry.context_prefix_hits;return result;
}

inline auto Engine::indexStoredOutputs() -> void {
  if(storage_copy_nodes_==nodes_.size() && storage_copy_unions_==unions_)return;
  storage_copies_.assign(nodes_.size(),{});
  // One bounded scan per congruence epoch, independent of output count.
  // Only immediate, exact InitialCell heads are inspected: no ancestor walk,
  // numeric conversion removal, or internal cross-type union is performed.
  for(Class id=0;id<nodes_.size();++id) {
    poll();const auto& k=nodes_[id];
    if(k.kind!=static_cast<int>(s::Kind::InitialCell) || k.name.empty() ||
        !k.attributes.empty() || !k.provenance.empty() || k.fold || k.children.size()!=1)continue;
    const auto type=k.type.starts_with("exact:")?k.type.substr(6):k.type;
    if(bitWidth(type)!=8*options_.storage_output_bytes)continue;
    const auto offset=find(k.children[0]);const auto& address=nodes_[offset];
    if(address.type!="u64" && address.type!="exact:u64")continue;
    auto& entry=storage_copies_[find(id)];
    if(!entry.cell) {entry.cell=id;entry.offset=offset;}
    else if(nodes_[*entry.cell].name!=k.name || entry.offset!=offset)entry.ambiguous=true;
  }
  storage_copy_nodes_=nodes_.size();storage_copy_unions_=unions_;
}

inline auto Engine::sameStoredOutput(Class a,Class b) -> bool {
  // Known scalar storage types are at most eight bytes; reject overflow,
  // unknown/unbounded formats, and the default-disabled contract up front.
  if(!options_.storage_output_bytes || options_.storage_output_bytes>8)return false;
  indexStoredOutputs();
  const auto& x=storage_copies_[find(a)];const auto& y=storage_copies_[find(b)];
  return x.cell && y.cell && !x.ambiguous && !y.ambiguous &&
      x.offset==y.offset && nodes_[*x.cell].name==nodes_[*y.cell].name;
}

// RESOURCE policy only. Shared by candidate-gated optional certificate
// epochs, never a source/domain/finite admission. Count stored headers and
// edges plus bounded scalar metadata lengths once; no string/record copies,
// no logical extent and no per-root replenishment. Preserve all prior work.
inline auto Engine::physicalWorkPlan(const std::vector<Key>& snapshot,
    std::size_t floor,std::size_t already_spent,bool account_global_work) -> PhysicalWorkPlan {
  namespace w=epoch_work_allowance;
  PhysicalWorkPlan out;w::PhysicalEpoch size;
  if(already_spent>w::maximum_work) {
    out.initial=w::maximum_work;out.remaining=0;out.exhausted=true;return out;
  }
  for(const auto& k:snapshot) {
    poll();constexpr std::size_t cost=6; // Header/edges/four scalar lengths.
    if(cost>w::maximum_work-already_spent-out.sizing) {
      out.initial=w::maximum_work;out.remaining=out.initial-already_spent-out.sizing;
      out.physical=size.allowance(floor).physical;out.exhausted=true;return out;
    }
    if(account_global_work && options_.max_work) {
      if(work_>options_.max_work || cost>options_.max_work-work_)throw Limit{"work_budget"};
      work_+=cost;
    }
    out.sizing+=cost; // Precharge before actual scalar header inspection.
    const std::array<std::size_t,4> lengths{
        k.type.size(),k.name.size(),k.attributes.size(),k.provenance.size()};
    size.addHeader(k.children.size(),lengths);
  }
  const auto allowance=size.allowance(floor);out.initial=allowance.max_work;out.physical=allowance.physical;
  if(already_spent>out.initial || out.sizing>out.initial-already_spent) {
    out.remaining=0;out.exhausted=true;return out;
  }
  out.remaining=out.initial-already_spent-out.sizing;return out;
}

} // namespace
} // namespace sentryward::equivalence
