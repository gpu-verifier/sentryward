#pragma once
#include "egraph_engine.hpp"

namespace sentryward::equivalence {
namespace {

inline auto Engine::projectionCharge(std::size_t amount) -> bool {
  poll();
  if(amount>projection_sat_remaining_){projection_sat_remaining_=0;return false;}
  projection_sat_remaining_-=amount;
  if(options_.max_work && amount>1) {
    if(amount-1>options_.max_work-work_)throw Limit{"work_budget"};
    work_+=amount-1;
  }
  return true;
}

inline auto Engine::scopedSatHead(const Key& k,const std::vector<Key>& arena) -> bool {
  if(k.kind!=6 || k.type!="e4m3" || !k.attributes.empty() || !k.provenance.empty() ||
      k.children.size()!=1 || arena[k.children[0]].type!="f32")return false;
  const auto step=leadingConversion("f32",k.name);
  return step && step->operation=="Convert rounding=nearest-even saturate_finite" &&
      step->target=="e4m3" && step->remainder.empty();
}

inline auto Engine::scopedExponentialSource(Class exponential) const -> bool {
  // Keep all complete local typing/modifier/provenance guards BEFORE type
  // erasure. The body proof later consumes only the Numeric views.
  const auto op=[&](Class id,std::string_view name,std::size_t arity) {
    const auto& k=nodes_[id];
    return k.type=="f32" && k.provenance.empty() && idealOperation(k,true) &&
        k.name==name && k.children.size()==arity &&
        std::all_of(k.children.begin(),k.children.end(),
            [&](Class child){return nodes_[child].type=="f32";});
  };
  if(!op(exponential,"Exp2",1))return false;
  const auto argument=nodes_[exponential].children[0];
  if(!op(argument,"Add",2))return false;
  for(unsigned side=0;side<2;++side) {
    const auto a=nodes_[argument].children[side],b=nodes_[argument].children[1-side];
    if(auto difference=localDifference(a,nodes_,op); difference && op(b,"Multiply",2)) {
      const auto product=difference->second;if(!op(product,"Multiply",2))continue;
      for(unsigned x=0;x<2;++x)for(unsigned y=0;y<2;++y)
        if(nodes_[product].children[x]==nodes_[b].children[y])return true;
    } else if(op(b,"Multiply",2)) {
      for(auto child:nodes_[b].children)if(localDifference(child,nodes_,op))return true;
    }
  }
  return false;
}

inline auto Engine::initializeProjectionScope() -> void {
  if(projection_scopes_)return;
  projection_scopes_=std::make_unique<lexical_scope::Catalog>(lexical_scope::Limits{},
      [this](std::size_t amount){return projectionCharge(amount);});
  projection_root_scope_=projection_scopes_->root(options_.outer_binders,
      options_.outer_index_extents).value_or(lexical_scope::unknown);
  projection_source_ids_.resize(nodes_.size());
  std::iota(projection_source_ids_.begin(),projection_source_ids_.end(),Class{0});
  // Explicit arena callbacks, not IDs from a second private graph. This is
  // one Engine's append-only projection epoch, BEFORE unions/rebuild. Child
  // properties are complete before their parent is interned. All borrowed
  // headers/spans live only within a query which itself does not intern.
  BodyHeadIndex::Adapter adapter;
  adapter.owner=[this](Class id) {
    return id<nodes_.size() && parents_[id]==id?id:std::numeric_limits<Class>::max();
  };
  adapter.affine=[this](Class id,AffineCache& cache,std::size_t& remaining) {
    if(id>=nodes_.size() || parents_[id]!=id)return std::optional<IndexAffine>{};
    return affineValue(id,cache,0,&remaining,0,64);
  };
  adapter.depends=[this](Class id,std::uint32_t binder) {
    if(id>=nodes_.size() || parents_[id]!=id)return true;
    const auto& levels=free_indices_[id];
    return std::binary_search(levels.begin(),levels.end(),binder);
  };
  projection_body_heads_=std::make_unique<BodyHeadIndex>(*this,nodes_,projection_source_ids_,
      projection_affine_cache_,&projection_sat_remaining_,std::move(adapter));
}

inline auto Engine::projectionMaxSamples(Class maximum) -> const MaxSamples* {
  if(auto found=projection_max_samples_.find(maximum);found!=projection_max_samples_.end())
    return &found->second;
  if(projection_max_samples_.size()>=128)return nullptr;
  MaxSamples result;bool valid=true;
  std::vector<MaxSample> todo{{maximum,0,1,true}};
  std::set<std::tuple<Class,std::uint32_t,std::uint64_t,bool>> seen;
  while(!todo.empty()) {
    if(!projectionCharge()){valid=false;break;}
    const auto sample=todo.back();todo.pop_back();const auto id=sample.body;
    if(!seen.emplace(id,sample.binder,sample.count,sample.singleton).second)continue;
    if(seen.size()>1024 || id>=nodes_.size()){valid=false;break;}
    const auto& k=nodes_[id];
    if(k.type=="real" && k.provenance.empty() && negative_infinity_[id])continue;
    if(!projectionFinite(id)){valid=false;break;}
    if(projectedOperation(id,"Maximum",2)) {
      for(auto child:k.children)todo.push_back({child,sample.binder,sample.count,sample.singleton});
      continue;
    }
    if(k.kind==static_cast<int>(s::Kind::Fold) && k.fold &&
        k.fold->reducer==s::Reducer::Max && k.children.size()==2 &&
        (k.attributes.empty() || k.attributes=="ideal_arithmetic")) {
      if(!sample.singleton || !k.fold->count || !projectionFinite(k.children[1]) ||
          std::any_of(free_indices_[k.children[0]].begin(),free_indices_[k.children[0]].end(),
              [&](auto level){return level>=k.fold->binder_level;}) ||
          std::any_of(free_indices_[k.children[1]].begin(),free_indices_[k.children[1]].end(),
              [&](auto level){return level>k.fold->binder_level;})){valid=false;break;}
      todo.push_back({k.children[1],k.fold->binder_level,k.fold->count,false});
      if(!negative_infinity_[k.children[0]])todo.push_back({k.children[0],0,1,true});
    } else {
      if(result.samples.size()>=1024 || projection_sample_count_>=4096){valid=false;break;}
      result.samples.push_back(sample);++projection_sample_count_;
    }
  }
  result.valid=valid;
  return &projection_max_samples_.emplace(maximum,std::move(result)).first->second;
}

inline auto Engine::projectedMaxContains(Class maximum,Class score,ScopeId scope) -> bool {
  const auto key=std::tuple{maximum,score,scope};
  if(auto found=projection_max_membership_.find(key);found!=projection_max_membership_.end())
    return found->second;
  if(projection_max_membership_.size()>=2048)return false;
  const auto finish=[&](bool answer) {
    projection_max_membership_.emplace(key,answer);return answer;
  };
  if(!projection_scopes_ || scope==lexical_scope::unknown ||
      !projectionFinite(maximum) || !projectionFinite(score))return finish(false);
  auto counts=projection_scopes_->counts(scope);if(!counts)return finish(false);
  // Unknown and empty domains never supply point coverage. A rank-zero
  // known scope is valid and can certify an identical scalar Max sample.
  for(auto count:*counts)if(!count || !*count)return finish(false);
  std::optional<std::uint32_t> binder;
  for(auto level:free_indices_[score]) {
    if(!projectionCharge() || level>=counts->size())return finish(false);
    if(!std::binary_search(free_indices_[maximum].begin(),free_indices_[maximum].end(),level)) {
      if(binder)return finish(false);binder=level;
    }
  }
  const auto* collection=projectionMaxSamples(maximum);
  if(!collection || !collection->valid)return finish(false);
  if(!binder) {
    for(const auto& sample:collection->samples) {
      if(!projectionCharge())return finish(false);
      if(sample.singleton && sample.body==score)return finish(true);
    }
    return finish(false);
  }
  const auto count=projection_scopes_->count(scope,*binder);
  if(!count || !*count)return finish(false);
  struct Progression {std::uint64_t begin,stride,count;};
  std::vector<Progression> progressions;std::uint64_t lattice=1;
  for(const auto& sample:collection->samples) {
    if(!projectionCharge())return finish(false);
    // A Fold binds its BODY, not its seed. Mismatched lexical depth is
    // Unknown, and inferred mapped j must stay in an actual stored interval.
    if(!sample.singleton) {
      const auto parent=projection_scopes_->prefix(scope,sample.binder);
      if(!parent || !projection_scopes_->bindBody(*parent,sample.binder,sample.count))continue;
    }
    const auto map=affineBodyMapWithHeads(score,sample.body,*binder,nodes_,
        sample.singleton,&projection_sat_remaining_,*projection_body_heads_,
        sample.singleton?*binder:sample.binder);
    if(!map) {
      if(!projection_sat_remaining_ || projection_body_heads_->exhausted)return finish(false);
      continue; // A Maximum dominates any certified subset of its samples.
    }
    if(!map->first)continue;
    if(map->second>=*count)continue;
    std::uint64_t next;
    if(__builtin_mul_overflow(lattice/std::gcd(lattice,map->first),map->first,&next) || next>1024)
      continue; // This subset cannot be represented within the proof budget.
    lattice=next;
    progressions.push_back({map->second,map->first,
        std::min(sample.count,1+(*count-1-map->second)/map->first)});
  }
  // A Max dominates the UNION of its parameterized samples, including
  // interleaved blocks. Unlike Sum coverage, overlap is harmless here.
  for(std::uint64_t residue=0;residue<lattice && residue<*count;++residue) {
    std::vector<std::pair<std::uint64_t,std::uint64_t>> intervals;
    for(const auto& p:progressions) {
      if(!projectionCharge(8))return finish(false);
      const auto delta=(residue+lattice-p.begin%lattice)%lattice;
      if(delta%p.stride)continue;
      const auto first=delta/p.stride;if(first>=p.count)continue;
      const auto begin=(p.begin+delta)/lattice;
      intervals.emplace_back(begin,begin+1+(p.count-1-first)/(lattice/p.stride));
    }
    if(!projectionCharge(intervals.size()*(1+std::bit_width(intervals.size()+1))))return finish(false);
    std::sort(intervals.begin(),intervals.end());std::uint64_t end=0;
    for(auto [begin,next]:intervals){if(begin>end)return finish(false);end=std::max(end,next);}
    if(end<1+(*count-1-residue)/lattice)return finish(false);
  }
  return finish(true);
}

inline auto Engine::projectedInactiveSaturation(Class conversion,Class exponential,ScopeId scope) -> bool {
  const ProjectionKey key{conversion,scope};
  if(auto found=projection_sat_certificates_.find(key);found!=projection_sat_certificates_.end())
    return found->second;
  if(projection_sat_certificates_.size()>=2048 || projection_sat_requests_>=256)return false;
  ++projection_sat_requests_;
  const auto finish=[&](bool answer) {
    projection_sat_certificates_.emplace(key,answer);return answer;
  };
  if(!projectionCharge() || !projectedOperation(exponential,"Exp2",1))return finish(false);
  const auto argument=nodes_[exponential].children[0];
  if(!projectedOperation(argument,"Add",2))return finish(false);
  std::optional<Class> score,maximum,coefficient,offset;
  const auto op=[&](Class id,std::string_view name,std::size_t arity) {
    return projectedOperation(id,name,arity);
  };
  // COMPLETE bounded local forms (p-c*M)+c*S and p+c*(S-M), sharing the
  // same positive finite coefficient. No descendant search/distribution.
  for(unsigned side=0;side<2 && !score;++side) {
    const auto a=nodes_[argument].children[side],b=nodes_[argument].children[1-side];
    if(auto difference=localDifference(a,nodes_,op); difference && projectedOperation(b,"Multiply",2)) {
      const auto p=difference->first,product=difference->second;
      if(!projectedOperation(product,"Multiply",2))continue;
      for(unsigned x=0;x<2 && !score;++x)for(unsigned y=0;y<2 && !score;++y) {
        const auto c=nodes_[product].children[x],d=nodes_[b].children[y];
        if(c==d && projectionFinite(c) && positive_[c]) {
          offset=p;coefficient=c;maximum=nodes_[product].children[1-x];score=nodes_[b].children[1-y];
        }
      }
    } else if(projectedOperation(b,"Multiply",2)) {
      for(unsigned x=0;x<2 && !score;++x) {
        const auto c=nodes_[b].children[x],difference=nodes_[b].children[1-x];
        const auto parts=localDifference(difference,nodes_,op);
        if(!projectionFinite(c) || !positive_[c] || !parts)continue;
        offset=a;coefficient=c;score=parts->first;maximum=parts->second;
      }
    }
  }
  if(!score || !maximum || !coefficient || !offset)return finish(false);
  const auto& p=nodes_[*offset];double bound=0;
  if(p.name.size()>128)return finish(false);
  const auto parsed=floatFromChars(p.name.data(),p.name.data()+p.name.size(),bound);
  if(p.kind!=static_cast<int>(s::Kind::Constant) || p.type!="real" ||
      !p.attributes.empty() || !p.provenance.empty() || parsed.ec!=std::errc{} ||
      parsed.ptr!=p.name.data()+p.name.size() || !std::isfinite(bound) || bound>8)return finish(false);
  if(idealSelect(nodes_[*maximum])) {
    const auto& select=nodes_[*maximum];const auto& pred=nodes_[select.children[0]];
    const auto cmp=comparison(pred);
    if(!cmp || *cmp!="ne" || !zero_[select.children[2]] ||
        pred.children.size()!=2 || pred.children[0]!=select.children[1] ||
        !negative_infinity_[pred.children[1]])return finish(false);
    maximum=select.children[1];
  }
  if(!projectionFinite(*maximum) || !projectionFinite(*score))return finish(false);
  // Numeric ONLY: S<=M,c>0,p<=8 imply ideal Exp2<=256<448. This is NOT an
  // IEEE RN/FTZ/approx theorem or a bit alias for quantized/NaN exact values.
  // PTX RN/SAT/NaN storage semantics remain on the untouched Exact view:
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt
  return finish(projectedMaxContains(*maximum,*score,scope));
}

inline auto Engine::scopedIndexRemainderHead(const Key& k,const std::vector<Key>& arena) -> bool {
  if(k.kind!=static_cast<int>(s::Kind::Operation) || k.name!="Remainder" ||
      (k.type!="u64" && k.type!="exact:u64") || !k.attributes.empty() ||
      !k.provenance.empty() || k.fold || k.children.size()!=2)return false;
  const auto& index=arena[k.children[0]];const auto& divisor=arena[k.children[1]];
  return index.kind==static_cast<int>(s::Kind::Index) && index.type==k.type &&
      index.attributes.empty() && index.provenance.empty() && index.children.empty() && !index.fold &&
      divisor.kind==static_cast<int>(s::Kind::Constant) && divisor.type==k.type &&
      divisor.attributes.empty() && divisor.provenance.empty() && divisor.children.empty() &&
      !divisor.fold && divisor.name.size()<=32;
}

inline auto Engine::scopedIndexRemainder(const Key& source,ScopeId scope) -> bool {
  if(!projection_scopes_ || scope==lexical_scope::unknown ||
      !scopedIndexRemainderHead(source,nodes_) ||
      !projectionCharge(8+nodes_[source.children[1]].name.size()))return false;
  const auto& index=nodes_[source.children[0]];
  const auto count=projection_scopes_->count(scope,index.index_level);
  if(!count || !*count)return false;
  const auto range=integer_interval::externalIndex(source.type,*count);
  const auto divisor=integer_interval::literal(source.type,nodes_[source.children[1]].name);
  if(!range || !divisor || divisor->lower!=divisor->upper || divisor->lower<=0 ||
      range->upper>=divisor->lower)return false;
  // Exact LOCAL use identity, not an e-class union or a global binder fact.
  // The only bound is this actual output/Fold BODY scope. Empty, unknown,
  // out-of-range and values outside the shared interval model are denied.
  return integer_interval::remainderByPositiveLiteral(*range,divisor->lower)==range;
}

inline auto Engine::project(Class root,bool numeric) -> Class {
  // Predicate ancestry is a third memoized use context, not a repeated
  // ancestor search. Existing numerical comparisons retain their vocabulary;
  // newly admitted approximate trig/log instructions stay exact beneath
  // predicates, even when their producer also has ordinary numeric users.
  enum class Context {Exact,Numeric,NumericPredicate};
  if(numeric && (projection_sat_sensitive_[root] || projection_integer_sensitive_[root] ||
      (options_.certified_branch_scope && projection_branch_sensitive_[root])))initializeProjectionScope();
  const auto projection=[&](Context context)->ProjectionMap& {
    return context==Context::Exact?exact_projection_:
        context==Context::Numeric?numeric_projection_:predicate_projection_;
  };
  const auto keyFor=[&](Class id,Context context,ScopeId scope) {
    const bool scoped=projection_integer_sensitive_[id] ||
        (context!=Context::Exact && options_.certified_branch_scope && projection_branch_sensitive_[id]);
    return ProjectionKey{id,scoped || (context==Context::Numeric && projection_sat_sensitive_[id])?scope:0};
  };
  const auto projected=[&](Class id,Context context,ScopeId scope) {
    return projection(context).at(keyFor(id,context,scope));
  };
  struct Task {Class id;Context context;ScopeId scope;bool ready;};
  const auto root_context=numeric?Context::Numeric:Context::Exact;
  // Exact ROOTS never acquire a Numeric consumer's lexical assumptions.
  // Exact address/mask children below Numeric retain the enclosing scope,
  // with full scope keys on the sensitive integer node AND its ancestors.
  const auto root_scope=numeric?projection_root_scope_:lexical_scope::unknown;
  std::vector<Task> work{{root,root_context,root_scope,false}};
  while(!work.empty()) {
    poll();const auto task=work.back();work.pop_back();
    auto& cache=projection(task.context);
    const auto cache_key=keyFor(task.id,task.context,task.scope);
    if(cache.contains(cache_key))continue;
    const auto source=nodes_[task.id];
    const bool numerical=task.context!=Context::Exact;
    const bool boolean_conversion=numerical && options_.certified_branch_scope && predicateFloatConversion(source,nodes_);
    const bool conjunction=numerical && options_.certified_branch_scope && predicateConjunction(source,nodes_);
    const bool boolean_constant=numerical && options_.certified_branch_scope &&
        source.kind==static_cast<int>(s::Kind::Constant) && source.type=="pred" &&
        source.attributes.empty() && source.provenance.empty() &&
        (source.name=="0" || source.name=="1" || source.name=="true" || source.name=="false");
    const bool sat_candidate=task.context==Context::Numeric &&
        task.scope!=lexical_scope::unknown && scopedSatHead(source,nodes_) &&
        scopedExponentialSource(source.children[0]);
    bool unwrap=numerical && source.kind==6 && source.provenance.empty() &&
        source.children.size()==1 &&
        idealNumericConversion(nodes_[source.children[0]].type,source.type,source.name,true);
    if(task.ready && sat_candidate)unwrap=projectedInactiveSaturation(task.id,
        projected(source.children[0],Context::Numeric,task.scope),task.scope);
    const bool predicate_approximate=task.context==Context::NumericPredicate &&
        (source.name=="PtxSin" || source.name=="PtxCos" || source.name=="PtxLog2");
    const auto legacy_primitive=!predicate_approximate && source.children.size()==1?
        legacyIdealPrimitive(source,nodes_[source.children[0]].type):std::nullopt;
    const bool math=numerical &&
        (boolean_conversion || conjunction || boolean_constant ||
         idealOperation(source,true) || legacy_primitive || idealSelect(source) || comparison(source) ||
         (source.kind==static_cast<int>(s::Kind::Fold) && realType(source.type)));
    const bool leaf=numerical && realType(source.type) &&
        (source.kind==static_cast<int>(s::Kind::InitialCell) ||
         source.kind==static_cast<int>(s::Kind::Parameter) ||
         source.kind==static_cast<int>(s::Kind::Constant));
    const bool opaque=numerical && !unwrap && !math && !leaf;
    const auto childContext=[&](std::size_t i) {
      if(!math)return Context::Exact;
      if(boolean_conversion || conjunction)return task.context;
      if(comparison(source))return realType(nodes_[source.children[i]].type)?
          Context::NumericPredicate:Context::Exact;
      if(source.name=="Select" && i==0)return comparison(nodes_[source.children[i]])?
          Context::NumericPredicate:Context::Exact;
      return task.context;
    };
    const auto childScope=[&](std::size_t i) {
      if(task.scope!=lexical_scope::unknown && source.kind==static_cast<int>(s::Kind::Fold) &&
          source.fold && i==1 && projection_scopes_ &&
          (projection_sat_sensitive_[source.children[i]] || projection_branch_sensitive_[source.children[i]] ||
           projection_integer_sensitive_[source.children[i]]))
        return projection_scopes_->bindBody(task.scope,source.fold->binder_level,
            source.fold->count).value_or(lexical_scope::unknown);
      return task.scope;
    };
    if(!task.ready) {
      work.push_back({task.id,task.context,task.scope,true});
      if(sat_candidate) {
        // One ideal child for the local proof plus the unchanged exact fallback.
        // No temporary SAT alias escapes to another use/context.
        work.push_back({task.id,Context::Exact,task.scope,false});
        work.push_back({source.children[0],Context::Numeric,task.scope,false});
      } else if(opaque)work.push_back({task.id,Context::Exact,task.scope,false});
      else if(unwrap)work.push_back({source.children[0],task.context,task.scope,false});
      else for(std::size_t i=0;i<source.children.size();++i)
        work.push_back({source.children[i],childContext(i),childScope(i),false});
      continue;
    }
    Class value;
    if(task.context==Context::Exact && scopedIndexRemainder(source,task.scope)) {
      value=projected(source.children[0],Context::Exact,task.scope);
    } else if(boolean_conversion) {
      Key one;one.kind=static_cast<int>(s::Kind::Constant);one.type="real";one.name="1";
      auto zero=one;zero.name="0";
      Key select;select.kind=static_cast<int>(s::Kind::Operation);select.type="real";select.name="Select";
      select.children={projected(source.children[0],task.context,task.scope),intern(std::move(one)),intern(std::move(zero))};
      value=intern(std::move(select));
    } else if(unwrap)value=projected(source.children[0],task.context,task.scope);
    else if(opaque) {
      Key k;k.kind=7;k.type="real";k.name="ExactNumericValue";
      k.attributes="encoding="+source.type;
      k.children={projected(task.id,Context::Exact,task.scope)};
      value=intern(std::move(k));
    } else {
      auto k=source;k.children.clear();
      for(std::size_t i=0;i<source.children.size();++i)
        k.children.push_back(projected(source.children[i],childContext(i),childScope(i)));
      if(numerical) {
        if(source.kind==static_cast<int>(s::Kind::InitialCell) ||
            source.kind==static_cast<int>(s::Kind::Parameter))
          k.attributes="encoding="+source.type+";"+source.attributes;
        k.type=comparison(source) || conjunction || boolean_constant?"pred":"real";
        if(conjunction){k.kind=7;k.name="ScopedPredicateAnd";k.attributes.clear();}
        if(source.kind==static_cast<int>(s::Kind::Constant) && !boolean_constant) {
          if(auto decoded=constantValue(source.type,source.name))k.name=decoded->literal;
          else {
            const auto infinity=source.provenance.empty() && source.attributes.empty()?
                floatingInfinity(source.type,source.name):std::nullopt;
            if(infinity)k.name=*infinity?"-inf":"inf";
            // Undecodable constants must retain their declared encoding. A
            // finite-only format's invalid "inf" literal is not real infinity
            // merely because projection erased its result type. This also
            // keeps malformed widths and NaN payloads typed and opaque.
            else if(source.type!="real" && source.attributes.empty())k.attributes="encoding="+source.type;
          }
        }
      } else k.type="exact:"+source.type;
      if(numerical)concatenateProjectedFold(k);
      if(math && source.kind==static_cast<int>(s::Kind::Operation)) {
        if(legacy_primitive)k.name=*legacy_primitive;
        else if(k.name=="Rsqrt")k.name="ReciprocalSqrt";
        if(k.name=="Fma") {
          auto product=k;product.name="Multiply";product.attributes="ideal_arithmetic";
          product.children.resize(2);
          k.name="Add";k.children={intern(std::move(product)),k.children[2]};
        }
        if(!comparison(k) && k.name!="Select" && !conjunction)k.attributes="ideal_arithmetic";
      }
      value=intern(std::move(k));
      if(numerical && branch_fact_anchor_ && branchPredicateSource(source,nodes_))
        if(const auto scope=branchScope(task.scope)) {
          Key wrapper;wrapper.kind=7;wrapper.type="pred";wrapper.name="ScopedPositivePredicate";
          wrapper.attributes=std::to_string(*scope)+":"+nodes_[source.children[0]].type;
          wrapper.children={value,*branch_fact_anchor_};value=intern(std::move(wrapper));
        }
    }
    if(cache.size()>=options_.max_nodes)throw Limit{"projection_memo_budget"};
    cache.emplace(cache_key,value);
  }
  return projected(root,root_context,root_scope);
}

} // namespace
} // namespace sentryward::equivalence
