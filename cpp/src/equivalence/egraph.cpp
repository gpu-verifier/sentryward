#include "egraph_engine.hpp"

namespace sentryward::equivalence {
struct FrozenUpdateEgraph::Impl {
  FrozenIndex index;
  Result result;
};
FrozenUpdateEgraph::FrozenUpdateEgraph(const s::Graph& graph,std::span<const s::Ref> roots,
    Options options,std::span<const std::string> positives):impl_(std::make_unique<Impl>()) {
  Engine engine(options,positives);
  std::vector<Class> imported_roots;
  if(!options.max_nodes || !options.max_work || options.timeout.count()<=0 || roots.empty()) {
    impl_->result=engine.result(Status::Unsupported,"invalid_update_budget");return;
  }
  try {
    imported_roots=engine.import(graph,roots);
    s::Graph other;auto sentinel=other.parameter(graph.type(roots[0]),"$update_never_equal");
    auto y=engine.import(other,std::span(&sentinel,1));
    // Full bounded saturation, not an equality query that could return early.
    std::vector<Class> unmatched(imported_roots.size(),y[0]);
    impl_->result=engine.compare(imported_roots,unmatched,true);
  } catch(const Limit& limit) {
    impl_->result=engine.result(Status::NotProven,limit.reason);
  } catch(const std::invalid_argument& error) {
    impl_->result=engine.result(Status::Unsupported,error.what());
  }
  // Even a budget stop retains only proven unions. Queries can miss, never
  // infer inequivalence or introduce a new union from an unproved candidate.
  impl_->index=engine.freeze(imported_roots);
}
FrozenUpdateEgraph::~FrozenUpdateEgraph()=default;
FrozenUpdateEgraph::FrozenUpdateEgraph(FrozenUpdateEgraph&&) noexcept=default;
FrozenUpdateEgraph& FrozenUpdateEgraph::operator=(FrozenUpdateEgraph&&) noexcept=default;
const Result& FrozenUpdateEgraph::compilation() const{return impl_->result;}
FrozenUpdateEgraph::Lookup FrozenUpdateEgraph::importedRoot(std::size_t index) const {
  if(index>=impl_->index.imported_roots.size())return {};
  return {impl_->index.imported_roots[index],1};
}
FrozenUpdateEgraph::Lookup FrozenUpdateEgraph::lookup(const s::Graph& graph,s::Ref root,std::size_t cap) const {
  Lookup result;
  std::unordered_map<s::Id,Class> known;
  std::vector<s::Ref> pending{root};
  while(!pending.empty()) {
    if(++result.work>cap)return result;
    const auto value=pending.back();
    if(!value.conversion.empty() || !value.pointer_provenance.empty())return result;
    if(known.contains(value.node)){pending.pop_back();continue;}
    const auto& n=graph.node(value);
    if(n.kind!=s::Kind::Parameter && n.kind!=s::Kind::Constant && n.kind!=s::Kind::Operation)return result;
    bool ready=true;
    for(const auto& child:n.operands) {
      if(!child.conversion.empty() || !child.pointer_provenance.empty())return result;
      if(!known.contains(child.node)) {pending.push_back(child);ready=false;}
    }
    if(!ready)continue;
    Key key{static_cast<int>(n.kind),n.type,n.name,n.attributes,{},n.fold,n.index_level,{}};
    for(const auto& child:n.operands)key.children.push_back(known.at(child.node));
    if(idealOperation(key) || idealSelect(key))key.attributes="ideal_arithmetic";
    if(arithmetic(key) && impl_->index.finite[key.children[0]] && impl_->index.finite[key.children[1]] &&
        key.children[1]<key.children[0])std::swap(key.children[0],key.children[1]);
    const auto found=impl_->index.memo.find(key);
    if(found==impl_->index.memo.end())return result;
    known.emplace(value.node,found->second);pending.pop_back();
  }
  result.eclass=known.at(root.node);return result;
}

scalar_fold::Ref normalizeExactUse(const scalar_fold::Graph& graph,scalar_fold::Ref ref) {
  for(;;) {
    if(!ref.conversion.empty()) {
      if(auto inactive=normalizeInactiveFiniteSaturation(graph,ref))ref=std::move(*inactive);
      if(auto normalized=normalizeConversions(graph.node(ref).type,graph.type(ref),ref.conversion)) {
        ref.conversion=*normalized;
        if(ref.conversion.empty())ref.converted_type.clear();
      }
    }
    auto reduced=packedLowUse(graph,ref);
    if(!reduced)reduced=packedHighUse(graph,ref);
    if(!reduced)return ref;
    ref=std::move(*reduced);
  }
}

scalar_fold::Ref normalizeExactUse(scalar_fold::Graph& graph,scalar_fold::Ref ref) {
  // Validate bounded metadata before the const path can canonicalize it.
  // In particular, a dangling delimiter must not become a valid field tail.
  if(!ref.conversion.empty() && ref.conversion.size()<=128 && ref.converted_type.size()<=128 &&
      !completeBoundedConversionUse(graph.node(ref).type,ref))return ref;
  ref=normalizeExactUse(static_cast<const scalar_fold::Graph&>(graph),std::move(ref));
  if(auto field=packedShiftedFieldUse(graph,ref))
    return normalizeExactUse(static_cast<const scalar_fold::Graph&>(graph),std::move(*field));
  return ref;
}

Result compareExpressionSets(const s::Graph& left, std::span<const s::Ref> a,
                             const s::Graph& right, std::span<const s::Ref> b, Options options) {
  Engine engine(options);
  if (!options.max_nodes || options.timeout.count() <= 0)
    return engine.result(Status::Unsupported, "invalid_budget");
  if(options.outer_index_extents.size()>options.outer_binders ||
      std::any_of(options.outer_index_extents.begin(),options.outer_index_extents.end(),[](auto n){return !n;}))
    return engine.result(Status::Unsupported,"invalid_outer_index_domain");
  if (a.size()!=b.size()) return engine.result(Status::Unsupported, "output_mapping_size_mismatch");
  if (a.empty()) return engine.result(Status::Unsupported, "empty_output_mapping");
  try {
    std::vector<unsigned char> storage_a,storage_b;
    const auto x=engine.import(left,a,options.storage_output_bytes?&storage_a:nullptr);
    const auto y=engine.import(right,b,options.storage_output_bytes?&storage_b:nullptr);
    for (std::size_t i=0; i<a.size(); ++i)
      if (!options.ideal_numeric_outputs && !options.storage_output_bytes &&
          left.type(a[i]) != right.type(b[i]))
        return engine.result(Status::NotProven,"different_result_types");
    return engine.compare(x,y,false,storage_a,storage_b);
  } catch (const Limit& limit) {
    return engine.result(Status::NotProven,limit.reason);
  } catch (const std::invalid_argument& error) {
    return engine.result(Status::Unsupported,error.what());
  } catch (const std::out_of_range& error) {
    return engine.result(Status::Unsupported,error.what());
  }
}

Result compareExpressions(const s::Graph& left, s::Ref a,
                          const s::Graph& right, s::Ref b, Options options) {
  return compareExpressionSets(left,std::span(&a,1),right,std::span(&b,1),options);
}

ScopedBodyResult proveScopedBodyRelation(const s::Graph& reference,const ScopedBodyPath& reference_path,
    const s::Graph& actual,const ScopedBodyPath& actual_path,
    std::span<const ScopedAxisBinding> bindings,Options options) {
  ScopedBodyResult result;
  if(!options.max_nodes || options.timeout.count()<=0 || bindings.size()>2 ||
      !options.ideal_numeric_outputs || options.outer_index_extents.size()!=options.outer_binders ||
      std::any_of(options.outer_index_extents.begin(),options.outer_index_extents.end(),[](auto n){return !n;})) {
    result.reason="invalid_scoped_configuration";return result;
  }
  if(!boundedScopedRef(reference_path.wrapper) || !boundedScopedRef(actual_path.wrapper) ||
      std::any_of(bindings.begin(),bindings.end(),[&](const auto& b){
        return !boundedScopedRef(b.coordinate) || (b.image_coordinate && !boundedScopedRef(*b.image_coordinate));})) {
    result.reason="scoped_metadata_cap";return result;
  }
  try {
    // Actual graph-root validation in import() precedes all domain minting.
    const auto r=extractScopedBody(reference,reference_path),a=extractScopedBody(actual,actual_path);
    std::vector<s::Ref> reference_witnesses{r.body},actual_witnesses{a.body};
    std::array<std::pair<bool,std::size_t>,2> locations{};
    std::array<std::optional<std::size_t>,2> image_locations{};
    for(std::size_t i=0;i<bindings.size();++i) {
      auto& witnesses=bindings[i].actual_shift?reference_witnesses:actual_witnesses;
      locations[i]={bindings[i].actual_shift,witnesses.size()};witnesses.push_back(bindings[i].coordinate);
      if(bindings[i].image_coordinate) {
        image_locations[i]=reference_witnesses.size();reference_witnesses.push_back(*bindings[i].image_coordinate);
      }
    }
    Engine engine(options);std::vector<Class> rw,aw;
    const auto rr=engine.import(reference,std::span(&reference_path.wrapper,1),nullptr,reference_witnesses,&rw);
    const auto ar=engine.import(actual,std::span(&actual_path.wrapper,1),nullptr,actual_witnesses,&aw);
    std::array<Class,2> coordinates{};
    std::array<std::optional<Class>,2> image_coordinates{};
    for(std::size_t i=0;i<bindings.size();++i)
      coordinates[i]=locations[i].first?rw[locations[i].second]:aw[locations[i].second];
    for(std::size_t i=0;i<bindings.size();++i)
      if(image_locations[i])image_coordinates[i]=rw[*image_locations[i]];
    return engine.scopedBodyProof(rr.front(),ar.front(),rw.front(),aw.front(),reference,r.folds,actual,a.folds,
        bindings,std::span(coordinates.data(),bindings.size()),std::span(image_coordinates.data(),bindings.size()));
  } catch(const Limit& limit) {result.reason=limit.reason;}
    catch(const std::invalid_argument& error) {result.reason=error.what();}
    catch(const std::out_of_range& error) {result.reason=error.what();}
  return result;
}

JoinedBodyResult proveJoinedBodyRelation(const s::Graph& reference,const ScopedBodyPath& reference_path,
    std::span<const std::uint32_t> reference_axes,const s::Graph& actual,
    std::span<const JoinedBodyPiece> pieces,Options options) {
  JoinedBodyResult result;
  if(!options.max_nodes || options.timeout.count()<=0 || !options.ideal_numeric_outputs ||
      options.outer_index_extents.size()!=options.outer_binders || reference_axes.empty() ||
      reference_axes.size()>2 || pieces.empty() || pieces.size()>=1024 ||
      std::any_of(options.outer_index_extents.begin(),options.outer_index_extents.end(),[](auto n){return !n;})) {
    result.reason="invalid_join_configuration";return result;
  }
  if(!boundedScopedRef(reference_path.wrapper)) {result.reason="scoped_metadata_cap";return result;}
  try {
    auto target=extractScopedBody(reference,reference_path);
    std::vector<s::Ref> actual_roots,actual_witnesses;
    std::vector<std::vector<s::Id>> folds;
    std::vector<std::array<std::size_t,3>> locations;
    actual_roots.reserve(pieces.size());folds.reserve(pieces.size());locations.reserve(pieces.size());
    for(const auto& piece:pieces) {
      if(piece.axis_count!=reference_axes.size() || !boundedScopedRef(piece.body.wrapper))
        throw std::invalid_argument("invalid_join_piece");
      auto use=extractScopedBody(actual,piece.body);std::array<std::size_t,3> location{};
      location[0]=actual_witnesses.size();actual_witnesses.push_back(use.body);
      for(unsigned axis=0;axis<piece.axis_count;++axis) {
        if(!boundedScopedRef(piece.axes[axis].coordinate))throw std::invalid_argument("scoped_metadata_cap");
        location[axis+1]=actual_witnesses.size();actual_witnesses.push_back(piece.axes[axis].coordinate);
      }
      actual_roots.push_back(piece.body.wrapper);folds.push_back(std::move(use.folds));locations.push_back(location);
    }
    Engine engine(options);std::vector<Class> rw,aw;
    const std::array<s::Ref,1> reference_witnesses{target.body};
    auto rr=engine.import(reference,std::span(&reference_path.wrapper,1),nullptr,reference_witnesses,&rw);
    auto ar=engine.import(actual,actual_roots,nullptr,actual_witnesses,&aw);
    std::vector<Class> bodies;std::vector<std::array<Class,2>> coordinates;
    bodies.reserve(pieces.size());coordinates.reserve(pieces.size());
    for(std::size_t i=0;i<pieces.size();++i) {
      bodies.push_back(aw[locations[i][0]]);std::array<Class,2> c{};
      for(unsigned axis=0;axis<pieces[i].axis_count;++axis)c[axis]=aw[locations[i][axis+1]];
      coordinates.push_back(c);
    }
    return engine.joinedBodyProof(rr.front(),rw.front(),reference,target.folds,reference_axes,
        actual,ar,bodies,folds,coordinates,pieces);
  } catch(const Limit& limit) {result.reason=limit.reason;}
    catch(const std::invalid_argument& error) {result.reason=error.what();}
    catch(const std::out_of_range& error) {result.reason=error.what();}
  return result;
}

ScopedProductResult proveScopedProductRelation(const s::Graph& reference,
    std::span<const ScopedBodyPath> reference_paths,const s::Graph& actual,
    std::span<const ScopedBodyPath> actual_paths,std::span<const ScopedAxisBinding> bindings,Options options) {
  ScopedProductResult result;
  if(!options.max_nodes || options.timeout.count()<=0 || !options.ideal_numeric_outputs ||
      options.outer_index_extents.size()!=options.outer_binders || reference_paths.empty() ||
      actual_paths.empty() || reference_paths.size()>8 || actual_paths.size()>8 || bindings.size()>2 ||
      std::any_of(options.outer_index_extents.begin(),options.outer_index_extents.end(),[](auto n){return !n;})) {
    result.reason="invalid_product_configuration";return result;
  }
  try {
    struct Input {
      std::vector<s::Ref> roots,witnesses;std::vector<std::vector<s::Id>> folds;
      std::vector<std::size_t> body_locations;
    };
    const auto input=[&](const s::Graph& graph,std::span<const ScopedBodyPath> paths) {
      Input result;
      for(const auto& path:paths) {
        if(!boundedScopedRef(path.wrapper))throw std::invalid_argument("scoped_metadata_cap");
        auto use=extractScopedBody(graph,path);result.roots.push_back(path.wrapper);
        result.body_locations.push_back(result.witnesses.size());result.witnesses.push_back(use.body);
        result.folds.push_back(std::move(use.folds));
      }
      return result;
    };
    auto r=input(reference,reference_paths),a=input(actual,actual_paths);
    std::array<std::pair<bool,std::size_t>,2> locations{};
    std::array<std::optional<std::size_t>,2> image_locations{};
    for(std::size_t i=0;i<bindings.size();++i) {
      const auto& b=bindings[i];
      if(!boundedScopedRef(b.coordinate) || (b.image_coordinate && !boundedScopedRef(*b.image_coordinate)))
        throw std::invalid_argument("scoped_metadata_cap");
      auto& witnesses=b.actual_shift?r.witnesses:a.witnesses;
      locations[i]={b.actual_shift,witnesses.size()};witnesses.push_back(b.coordinate);
      if(b.image_coordinate){image_locations[i]=r.witnesses.size();r.witnesses.push_back(*b.image_coordinate);}
    }
    Engine engine(options);std::vector<Class> rw,aw;
    auto rr=engine.import(reference,r.roots,nullptr,r.witnesses,&rw);
    auto ar=engine.import(actual,a.roots,nullptr,a.witnesses,&aw);
    std::vector<Class> rb,ab;
    for(auto location:r.body_locations)rb.push_back(rw[location]);
    for(auto location:a.body_locations)ab.push_back(aw[location]);
    std::array<Class,2> coordinates{};std::array<std::optional<Class>,2> images{};
    for(std::size_t i=0;i<bindings.size();++i) {
      coordinates[i]=locations[i].first?rw[locations[i].second]:aw[locations[i].second];
      if(image_locations[i])images[i]=rw[*image_locations[i]];
    }
    return engine.productBodyProof(reference,rr,rb,r.folds,actual,ar,ab,a.folds,bindings,
        std::span(coordinates.data(),bindings.size()),std::span(images.data(),bindings.size()));
  } catch(const Limit& limit) {result.reason=limit.reason;}
    catch(const std::invalid_argument& error) {result.reason=error.what();}
    catch(const std::out_of_range& error) {result.reason=error.what();}
  return result;
}
}  // namespace sentryward::equivalence
