#pragma once
#include "egraph_engine.hpp"

namespace sentryward::equivalence {
namespace {

inline auto Engine::affineBodyMapWithHeads(
    Class target,Class block,std::uint32_t binder,const std::vector<Key>& snapshot,
    bool singleton,std::size_t* remaining,BodyHeadIndex& heads,
    std::uint32_t block_binder,BodyScopeRegistry* registry,
    const BodyScopeRegistry::Context* initial,
    const BodyScopeRegistry::ContextId* admitted,ScopedSumProof* sums,
    ScopedSumProof::Assignment* assignment,
    BodyScopeRegistry::Id reference_scope,BodyScopeRegistry::Id actual_scope,
    const ScopedSumProof::Environment* enclosing) -> std::optional<std::pair<std::uint64_t,std::uint64_t>> {
  // ONE existing paired proof, shared by reduction coverage and Numeric SAT.
  // The optional source binder supports a sample Fold at another lexical depth.
  if(block_binder==UINT32_MAX)block_binder=binder;
  constexpr std::size_t max_pairs=4096,max_environments=4096,max_depth=8;
  const bool sum_context=sums && assignment;
  constexpr std::size_t mapper_limit=131072;
  std::size_t mapper_remaining=mapper_limit;
  // Aggregate borrowing does not enlarge or refresh the physical epoch.
  // Nested coverage and body calls reuse the original pointer. Legacy
  // callers retain their old131k allowance unchanged.
  std::size_t coverage_remaining=heads.remaining?*heads.remaining:mapper_limit;
  const bool scoped_coverage=sum_context || heads.coverage_active;
  BodyHeadIndex::QueryBudget budget(heads,&mapper_remaining,&coverage_remaining,sum_context);
  if(bool(sums)!=bool(assignment) || (sum_context && (registry || initial || admitted ||
      &sums->heads!=&heads || !sums->valid(*assignment))))return {};
  std::optional<ScopedSumProof::QueryRelay> query_relay;
  if(sum_context)query_relay.emplace(sums->planner_query,&heads.query_guards);
  auto* geometry_registry=sum_context?&sums->registry:registry;
  if(bool(registry)!=bool(initial) || (admitted && !registry) ||
      (registry && &registry->heads!=&heads))return {};
  if(admitted) {
    // No caller bool: require this EXACT immutable record, owned by the
    // current registry/epoch. All actual body/leaf/scope checks below remain.
    if(registry->context(*admitted)!=initial)return {};
  } else if(registry && !registry->validate(*initial))return {};
  const bool explicit_context=initial!=nullptr;
  using Environment=std::vector<std::pair<std::uint32_t,std::uint32_t>>;
  std::vector<Environment> environments(1);
  if(enclosing){if(!sum_context || enclosing->size()>max_depth)return {};environments[0]=*enclosing;}
  std::map<Environment,unsigned> environment_keys{{environments[0],0}};
  ScopedSumProof::Geometry geometry;
  if(sum_context)geometry.assignment=*assignment;
  // This selected product view only supplies factor tasks. All tasks retain
  // the SAME unfinished assignment, original scopes, gray/cycle sets and
  // final affine-equation solve; no independent scoped relation is minted.
  std::optional<BoundedProducts> scoped_products;
  struct Task {
    Class a,b;unsigned environment=0;bool substitute=true,ready=false;
    unsigned fixed_mask=0;
    BodyScopeRegistry::Id reference_scope=0,actual_scope=0;
  };
  std::vector<Task> work{{target,block}};
  if(explicit_context) {
    work.front().substitute=false;work.front().fixed_mask=(1u<<initial->axis_count)-1;
    work.front().reference_scope=initial->reference_scope;work.front().actual_scope=initial->actual_scope;
  }
  if(sum_context) {
    if(reference_scope>=geometry_registry->scopes.size() || actual_scope>=geometry_registry->scopes.size())return {};
    work.front().substitute=false;work.front().reference_scope=reference_scope;work.front().actual_scope=actual_scope;
  }
  std::map<std::tuple<Class,Class,unsigned,bool,unsigned,
      BodyScopeRegistry::Id,BodyScopeRegistry::Id>,unsigned char> visited;
  struct MapperTrace {
    ScopedSumProof* sums;const decltype(visited)& pairs;std::size_t tasks=0;
    ~MapperTrace(){if(sums){sums->mapper_visits+=pairs.size();sums->mapper_tasks+=tasks;}}
  } mapper_report{sum_context && sums->trace_work?sums:nullptr,visited};
  // Validate only the heads actually chosen by this paired proof. No eager
  // ancestor walk: shared independent eclasses and exact affine summaries
  // are already certified opaque boundaries. A selected side may not reenter
  // an active class under a different counterpart or lexical environment.
  std::unordered_set<Class> active_left,active_right;
  std::optional<std::pair<std::uint64_t,std::uint64_t>> map;
  const auto deny=[&](const char* reason,const Task& task,
      std::optional<std::pair<Class,Class>> chosen=std::nullopt,
      std::optional<std::pair<std::uint64_t,std::uint64_t>> observed=std::nullopt)
      ->std::optional<std::pair<std::uint64_t,std::uint64_t>> {
    if(heads.denial_trace_remaining && *heads.denial_trace_remaining) {
      --*heads.denial_trace_remaining;
      SENTRYWARD_DEBUG_PRINT(egraph_102);
      if(chosen)SENTRYWARD_DEBUG_PRINT(egraph_103);else SENTRYWARD_DEBUG_PRINT(egraph_104);
      SENTRYWARD_DEBUG_PRINT(egraph_105);
      // This walk already admits at most8 lexical bindings; no ancestor or
      // expression rendering is used, and orientation probes do not spend
      // this separate first16-denials-per-reducer/epoch ledger.
      for(auto [a,b]:environments[task.environment])SENTRYWARD_DEBUG_PRINT(egraph_106);
      SENTRYWARD_DEBUG_PRINT(egraph_107);
      if(map)SENTRYWARD_DEBUG_PRINT(egraph_108);else SENTRYWARD_DEBUG_PRINT(egraph_109);
      if(observed)SENTRYWARD_DEBUG_PRINT(egraph_110);
      if(sum_context) {
        SENTRYWARD_DEBUG_PRINT(egraph_111);
        for(const auto& axis:geometry.assignment.axes) {
          SENTRYWARD_DEBUG_PRINT(egraph_112);if(axis.actual)SENTRYWARD_DEBUG_PRINT(egraph_113);else SENTRYWARD_DEBUG_PRINT(egraph_114);
          SENTRYWARD_DEBUG_PRINT(egraph_115);if(axis.stride)SENTRYWARD_DEBUG_PRINT(egraph_116);else SENTRYWARD_DEBUG_PRINT(egraph_117);
          SENTRYWARD_DEBUG_PRINT(egraph_118);if(axis.offset)SENTRYWARD_DEBUG_PRINT(egraph_119);else SENTRYWARD_DEBUG_PRINT(egraph_120);
          SENTRYWARD_DEBUG_PRINT(egraph_121);
        }
      }
      const auto active_remaining=scoped_coverage?mapper_remaining:
          (heads.query_remaining?*heads.query_remaining:mapper_remaining);
      SENTRYWARD_DEBUG_PRINT(egraph_122);
      // Read only these immutable stored headers. Never select a witness,
      // inspect class members, render ancestors, or debit proof work merely
      // to explain one of the existing first16 TRACE denials.
      const auto text=[&](std::string_view value) {
        SENTRYWARD_DEBUG_PRINT(egraph_123);
        const auto size=std::min<std::size_t>(value.size(),128);
        constexpr char hex[]="0123456789abcdef";
        for(std::size_t i=0;i<size;++i) {
          const auto c=static_cast<unsigned char>(value[i]);
          if(c=='"' || c=='\\')SENTRYWARD_DEBUG_PRINT(egraph_124);
          else if(c<32 || c==127)SENTRYWARD_DEBUG_PRINT(egraph_125);
          else SENTRYWARD_DEBUG_PRINT(egraph_126);
        }
        SENTRYWARD_DEBUG_PRINT(egraph_127);
        if(size<value.size())SENTRYWARD_DEBUG_PRINT(egraph_128);
      };
      const auto plain=[&](const char* label,Class id) {
        SENTRYWARD_DEBUG_PRINT(egraph_129);
        if(id>=snapshot.size()){SENTRYWARD_DEBUG_PRINT(egraph_130);return;}
        const auto& k=snapshot[id];
        SENTRYWARD_DEBUG_PRINT(egraph_131);text(k.name);
        SENTRYWARD_DEBUG_PRINT(egraph_132);text(k.type);
        SENTRYWARD_DEBUG_PRINT(egraph_133);text(k.attributes);
        SENTRYWARD_DEBUG_PRINT(egraph_134);text(k.provenance);
        SENTRYWARD_DEBUG_PRINT(egraph_135);
        for(std::size_t i=0;i<std::min<std::size_t>(k.children.size(),4);++i)
          SENTRYWARD_DEBUG_PRINT(egraph_136);
        if(k.children.size()>4)SENTRYWARD_DEBUG_PRINT(egraph_137);
        if(k.fold)SENTRYWARD_DEBUG_PRINT(egraph_138);
        SENTRYWARD_DEBUG_PRINT(egraph_139);
      };
      plain("plain_left",heads.owner(task.a));
      plain("plain_right",heads.owner(task.b));
      if(chosen) {
        plain("chosen_left",chosen->first);plain("chosen_right",chosen->second);
      }
      SENTRYWARD_DEBUG_PRINT(egraph_140);
    }
    return {};
  };
  const auto independent=[&](Class cls,unsigned environment,bool substitute,unsigned fixed_mask,
      BodyScopeRegistry::Id reference_scope,BodyScopeRegistry::Id actual_scope) {
    if(substitute && (heads.depends(cls,binder) || heads.depends(cls,block_binder)))return false;
    for(auto [a,b]:environments[environment]) {
      if(!heads.charge())return false;
      if(a!=b && (heads.depends(cls,a) || heads.depends(cls,b)))return false;
    }
    if(explicit_context) {
      for(unsigned i=0;i<initial->axis_count;++i)if(fixed_mask&(1u<<i)) {
        if(!heads.charge())return false;const auto& a=initial->axes[i];
        const bool identity=a.actual_level && *a.actual_level==a.reference_level && !a.offset;
        if(!identity && (heads.depends(cls,a.reference_level) ||
            (a.actual_level && heads.depends(cls,*a.actual_level))))return false;
      }
      std::array<std::uint32_t,max_depth+2> covered{};std::size_t n=0;
      for(unsigned i=0;i<initial->axis_count;++i)if(fixed_mask&(1u<<i)) {
        const auto& a=initial->axes[i];
        if(a.actual_level && a.reference_level==*a.actual_level && !a.offset)covered[n++]=a.reference_level;
      }
      for(auto [a,b]:environments[environment])if(a==b)covered[n++]=a;
      auto r=registry->scopeView(reference_scope),a=registry->scopeView(actual_scope);
      if(!r || !a || !registry->sameAssignment(cls,*r,*a,std::span(covered.data(),n)))return false;
    }
    if(sum_context && !sums->independent(cls,geometry.assignment,reference_scope,actual_scope,
        environments[environment]))return false;
    return true;
  };
  while(!work.empty()) {
    if(mapper_report.sums)++mapper_report.tasks;
    if(!heads.charge())return deny("work",work.back());auto t=work.back();work.pop_back();t.a=heads.owner(t.a);t.b=heads.owner(t.b);
    if(t.a>=heads.memberCount() || t.b>=heads.memberCount())return deny("member_bounds",t);
    if(snapshot[t.a].type!=snapshot[t.b].type)return deny("type",t);
    const auto pair=std::tuple{t.a,t.b,t.environment,t.substitute,t.fixed_mask,t.reference_scope,t.actual_scope};
    if(t.ready) {
      active_left.erase(t.a);active_right.erase(t.b);visited[pair]=2;continue;
    }
    if(visited.size()>=max_pairs && !visited.contains(pair))return deny("pair_cap",t);
    const auto invariant=[&](Class cls) {
      return independent(cls,t.environment,t.substitute,t.fixed_mask,t.reference_scope,t.actual_scope);
    };
    if(t.a==t.b && (invariant(t.a) || (sum_context &&
        sums->sameOwnerIdentity(t.a,geometry,t.reference_scope,t.actual_scope,
            environments[t.environment])))){visited[pair]=2;continue;}
    auto x=heads.affine(t.a),y=heads.affine(t.b);
    if(heads.query_guards.exhausted() ||
        (remaining && !*remaining))return deny("affine_work",t);
    if(x && y) {
      if(sum_context) {
        if(!sums->leaf(std::move(*x),std::move(*y),geometry,t.reference_scope,t.actual_scope,
            environments[t.environment]))return deny("sum_scoped_affine_leaf",t);
        visited[pair]=2;continue;
      }
      if(explicit_context) {
        if(!registry->leaf(*x,*y,*initial,t.fixed_mask,t.reference_scope,t.actual_scope,
            environments[t.environment]))return deny("scoped_affine_leaf",t);
        visited[pair]=2;continue;
      }
      if(!heads.charge((x->coefficients.size()+y->coefficients.size()+environments[t.environment].size()+1)*
          (1+std::bit_width(x->coefficients.size()+y->coefficients.size()+1))))return deny("affine_coeff_work",t);
      for(auto [a,b]:environments[t.environment]) {
        const auto ca=x->coefficients.contains(a)?x->coefficients.at(a):0;
        const auto cb=y->coefficients.contains(b)?y->coefficients.at(b):0;
        if(ca!=cb)return deny("bound_coefficient",t);x->coefficients.erase(a);y->coefficients.erase(b);
      }
      auto c=x->coefficients.contains(binder)?x->coefficients.at(binder):0;
      auto d=y->coefficients.contains(block_binder)?y->coefficients.at(block_binder):0;
      if(!t.substitute)c=d=0;
      if(c) {
        if((!d && !singleton) || (d && singleton) || d%c ||
            y->constant<x->constant || (y->constant-x->constant)%c)return deny("affine_stride_offset",t);
        // A scalar piece covers exactly one point. Its address contains no
        // reduction binder; only the offset is observable, not the stride.
        const auto candidate=std::pair{singleton?std::uint64_t{1}:d/c,(y->constant-x->constant)/c};
        if(map && *map!=candidate)return deny("affine_image_conflict",t,std::nullopt,candidate);map=candidate;
      } else if(d || x->constant!=y->constant)return deny("affine_constant",t);
      if(t.substitute){x->coefficients.erase(binder);y->coefficients.erase(block_binder);}
      if(x->coefficients!=y->coefficients)return deny("captured_coefficients",t);
      visited[pair]=2;continue;
    }
    if(active_left.contains(t.a) || active_right.contains(t.b))return deny("active_cycle",t);
    if(auto v=visited.find(pair);v!=visited.end()){if(v->second==1)return deny("paired_cycle",t);continue;}
    std::optional<std::pair<Class,Class>> matched;
    if(sum_context) {
      const auto selected=sums->select(t.a,t.b,t.reference_scope,t.actual_scope);
      if(selected.coverage) {
        if(!sums->related(t.a,t.b,t.reference_scope,t.actual_scope,
            environments[t.environment],geometry.assignment))return deny("sum_coverage",t);
        visited[pair]=2;continue;
      }
      matched=selected.ordinary;
    }
    // Reuse the ordinary fixed-radius witness selector before its existing
    // root-only fallback. A certified alias may move a Fold's seed into an
    // Add; selecting the first common root header misses that valid witness.
    if(!matched)matched=heads.matched(t.a,t.b);
    if(!matched) {
      if(sum_context && sums->compatible(t.a,t.b) &&
          sums->related(t.a,t.b,t.reference_scope,t.actual_scope,environments[t.environment],geometry.assignment)) {
        visited[pair]=2;continue;
      }
      return deny("head_match",t);
    }
    const auto& p=snapshot[matched->first];const auto& q=snapshot[matched->second];
    if(p.kind==static_cast<int>(s::Kind::Index) || p.children.size()!=q.children.size())return deny("index_or_arity",t,matched);
    unsigned body_environment=t.environment;
    auto reference_body_scope=t.reference_scope,actual_body_scope=t.actual_scope;
    auto body_mask=t.fixed_mask;
    if(p.fold) {
      if(!q.fold || p.fold->reducer!=q.fold->reducer || p.fold->count!=q.fold->count ||
          p.children.size()!=2 || p.kind!=static_cast<int>(s::Kind::Fold))return deny("fold_metadata",t,matched);
      const auto& enclosing=environments[t.environment];
      if(enclosing.size()>=max_depth || (!enclosing.empty() &&
          (p.fold->binder_level<=enclosing.back().first || q.fold->binder_level<=enclosing.back().second)))return deny("fold_lexical_order",t,matched);
      // Reject capture/rebinding, including an apparently harmless identical
      // eclass reused under different lexical pairings.
      for(auto [a,b]:enclosing)if(a==p.fold->binder_level || b==q.fold->binder_level)return deny("fold_capture",t,matched);
      if(!heads.charge(enclosing.size()+1))return deny("fold_work",t,matched);
      if(explicit_context || sum_context) {
        auto a=geometry_registry->insideFold(t.reference_scope,matched->first);
        auto b=geometry_registry->insideFold(t.actual_scope,matched->second);
        if(!a || !b)return deny("scoped_fold_body",t,matched);reference_body_scope=*a;actual_body_scope=*b;
        // Registry requires fresh, strictly deeper actual BODY binders.
        // Keep the mask explicit in memo keys even for conservative shadowing.
        for(unsigned i=0;explicit_context && i<initial->axis_count;++i)if(body_mask&(1u<<i)) {
          const auto& axis=initial->axes[i];
          if(axis.reference_level==p.fold->binder_level ||
              (axis.actual_level && *axis.actual_level==q.fold->binder_level))return deny("scoped_fold_capture",t,matched);
        }
        if(sum_context)for(const auto& axis:geometry.assignment.axes)
          if(axis.reference==p.fold->binder_level || axis.actual==q.fold->binder_level)
            return deny("sum_scoped_fold_capture",t,matched);
      }
      auto nested=enclosing;nested.emplace_back(p.fold->binder_level,q.fold->binder_level);
      if(!heads.charge((nested.size()+1)*(1+std::bit_width(environment_keys.size()+1))))return deny("environment_work",t,matched);
      auto found=environment_keys.find(nested);
      if(found==environment_keys.end()) {
        if(environments.size()>=max_environments)return deny("environment_cap",t,matched);
        body_environment=static_cast<unsigned>(environments.size());
        environments.push_back(nested);environment_keys.emplace(std::move(nested),body_environment);
      } else body_environment=found->second;
    }
    bool swap=false;
    std::optional<std::vector<BoundedProducts::FactorPair>> factor_pairs;
    if((arithmetic(p) || integerCommutative(p)) && p.children.size()==2) {
      auto p0=heads.owner(p.children[0]),p1=heads.owner(p.children[1]);
      auto q0=heads.owner(q.children[0]),q1=heads.owner(q.children[1]);
      const bool common_direct=(p0==q0 && invariant(p0)) || (p1==q1 && invariant(p1));
      const bool common_reverse=(p0==q1 && invariant(p0)) || (p1==q0 && invariant(p1));
      const auto root_orientation=[&]()->std::optional<bool> {
        if(!sum_context || heads.adapter)return {};
        // At most FOUR existing complete-root selections. They schedule
        // one orientation only; no child equality, affine assignment or
        // invariant is inferred from a root header. Ambiguity retains the
        // old hints BEFORE full child proof, never retries a failed walk.
        ++heads.root_orientation_attempts;
        const bool direct=bool(heads.rootMatched(p0,q0)) && bool(heads.rootMatched(p1,q1));
        const bool reverse=bool(heads.rootMatched(p0,q1)) && bool(heads.rootMatched(p1,q0));
        if(heads.exhausted || (remaining && !*remaining) ||
            heads.query_guards.exhausted())return {};
        if(direct!=reverse || (direct && (p0==p1 || q0==q1))) {
          ++heads.root_orientation_hits;return !direct;
        }
        return {};
      };
      if(common_direct)swap=false;
      else if(common_reverse)swap=true;
      else if(auto orientation=root_orientation())swap=*orientation;
      else {
        if(sum_context && (heads.exhausted || (remaining && !*remaining) ||
            heads.query_guards.exhausted()))return deny("orientation_root_work",t,matched);
        auto a=heads.shallow(p0),b=heads.shallow(p1),c=heads.shallow(q0),d=heads.shallow(q1);
        if((!a || !b || !c || !d) && !sum_context)return deny("orientation_hints",t,matched);
        bool direct=a && b && c && d && *a==*c && *b==*d;
        bool reverse=a && b && c && d && *a==*d && *b==*c;
        if(!direct && !reverse) {
          // Preferred-child shallow shapes can differ only because an
          // equivalent child prefers another stored alias. A fixed-local
          // root-header discriminator may orient ONE compatible pairing;
          // the SAME scoped walk still proves every chosen operand below.
          const auto compatible=[&](Class a,Class b) {
            return heads.localRootCompatible(a,b) || (sum_context && sums->compatible(a,b));
          };
          direct=compatible(p0,q0) && compatible(p1,q1);
          reverse=compatible(p0,q1) && compatible(p1,q0);
        }
        const auto ambiguous=[&] {return direct==reverse && (!direct || (p0!=p1 && q0!=q1));};
        if(ambiguous() && sum_context) {
          // Complete immediate root headers are a weaker scheduling hint,
          // not a proof veto based on the association of their children.
          const auto compatible=[&](Class a,Class b) {
            return bool(heads.rootMatched(a,b)) || sums->compatible(a,b);
          };
          direct=compatible(p0,q0) && compatible(p1,q1);
          reverse=compatible(p0,q1) && compatible(p1,q0);
        }
        if(ambiguous() && sum_context && p.name=="Add") {
          // An exact Negate alias can distinguish v + (-sum) even when
          // both children also have Multiply aliases. This only chooses
          // an orientation: the ordinary scoped walk proves BOTH terms.
          const auto negated=[&](Class cls) {
            const auto* choices=heads.rootChoices(cls);if(!choices)return false;
            for(const auto& [key,choice]:*choices) {
              (void)key;if(!heads.charge())return false;
              const auto& head=snapshot[choice.head];
              if(idealOperation(head) && head.name=="Negate" && head.children.size()==1)return true;
            }
            return false;
          };
          const bool a=negated(p0),b=negated(p1),c=negated(q0),d=negated(q1);
          if(a!=b && c!=d){direct=a==c;reverse=!direct;}
        }
        if(ambiguous()) {
          if(sum_context && !heads.adapter && p.name=="Multiply" && q.name=="Multiply" &&
              p.type=="real" && q.type=="real" && p.provenance.empty() && q.provenance.empty() &&
              arithmetic(p) && arithmetic(q) && finite_[heads.owner(t.a)] && finite_[heads.owner(t.b)]) {
            // Construct only when used: the factory owns deque storage,
            // whose empty construction otherwise allocates on every query.
            if(!scoped_products)scoped_products.emplace(*geometry_registry);
            factor_pairs=scoped_products->pairedFactors({t.a,t.reference_scope},{t.b,t.actual_scope});
          }
          if(!factor_pairs)return deny("orientation_root_ambiguous",t,matched);
        }
        swap=!direct;
      }
    }
    active_left.insert(t.a);active_right.insert(t.b);
    visited[pair]=1;work.push_back({t.a,t.b,t.environment,t.substitute,true,
        t.fixed_mask,t.reference_scope,t.actual_scope});
    if(factor_pairs) {
      // Repeated equal-shape factors have identical actual Uses and checked
      // occurrence counts; one proof per distinct pair suffices. Different
      // same-tag uses were denied by the EXISTING product group factory.
      for(const auto& [a,b]:*factor_pairs) {
        if(!heads.charge())return deny("product_child_work",t,matched);
        work.push_back({a.body,b.body,t.environment,t.substitute,false,
            t.fixed_mask,a.scope,b.scope});
      }
      continue;
    }
    for(std::size_t i=0;i<p.children.size();++i) {
      const bool shadow=p.fold && i==1 && p.fold->binder_level==binder;
      if(!heads.charge())return deny("child_work",t,matched);
      work.push_back({p.children[i],q.children[swap?1-i:i],
          p.fold && i==1?body_environment:t.environment,t.substitute && !shadow,false,
          p.fold && i==1?body_mask:t.fixed_mask,
          p.fold && i==1?reference_body_scope:t.reference_scope,
          p.fold && i==1?actual_body_scope:t.actual_scope});
    }
  }
  if(sum_context) {
    if(!sums->solve(geometry))return deny("sum_affine_assignment",Task{target,block});
    *assignment=std::move(geometry.assignment);
    return std::pair{std::uint64_t{1},std::uint64_t{0}};
  }
  if(!explicit_context && !map)return deny("no_observable_axis",Task{target,block});
  return explicit_context?std::optional{std::pair{std::uint64_t{1},std::uint64_t{0}}}:map;
}

inline auto Engine::coalesceInterleavedFolds(Class id,const Key& root,const std::vector<Key>& snapshot,
    const std::vector<std::vector<Class>>& members,
    const std::map<std::tuple<std::string,std::uint32_t,std::uint64_t>,std::vector<Class>>& targets) -> void {
  // At most 64 compact Fold pieces in this bounded local Add pattern. The
  // bound limits matching work, NOT reduction count (possibly billions).
  struct Piece {Class body;std::uint64_t count;bool singleton;};
  // One local ledger and immutable selected-head/affine caches for the
  // ENTIRE stored forest/target loop. A failed target must not reset the
  // body-match budget or recopy the same signature metadata. No union is
  // made until a complete certificate succeeds and this call returns.
  std::size_t remaining=rectangle_work_limit;
  AffineCache affine_cache;
  BodyHeadIndex heads(*this,snapshot,members,affine_cache,&remaining);
  if(!heads.charge(root.children.size()))return;
  std::vector<Piece> pieces;std::vector<Class> work=root.children;std::unordered_set<Class> seen;
  std::optional<std::uint32_t> binder;std::uint64_t total=0;
  while(!work.empty()) {
    if(!heads.charge())return;auto cls=find(work.back());work.pop_back();
    if(cls>=members.size())return;
    if(!seen.insert(cls).second || seen.size()>127)return;
    std::optional<Class> fold,add;
    for(auto member:members[cls]) {
      const auto& k=snapshot[member];
      if(!BodyHeadIndex::bounded(k) ||
          !heads.charge(1+k.type.size()+k.name.size()+k.attributes.size()+k.provenance.size()))return;
      if(k.kind==static_cast<int>(s::Kind::Fold) && k.type==root.type && k.attributes.empty() &&
          k.provenance.empty() && k.fold->reducer==s::Reducer::Sum && k.fold->count &&
          k.children.size()==2 && zero_[find(k.children[0])] && finite_[cls]){fold=member;break;}
      if(arithmetic(k) && k.type==root.type && k.name=="Add" && !add)add=member;
    }
    if(fold) {
      const auto& k=snapshot[*fold];
      if(binder && *binder!=k.fold->binder_level)return;binder=k.fold->binder_level;
      if(k.fold->count>UINT64_MAX-total)return;total+=k.fold->count;
      pieces.push_back({k.children[1],k.fold->count,false});
      if(pieces.size()>64)return;
    } else if(add)for(auto child:snapshot[*add].children) {
      if(!heads.charge())return;work.push_back(child);
    }
    else {
      if(!finite_[cls] || snapshot[cls].type!=root.type || !snapshot[cls].provenance.empty() || total==UINT64_MAX)return;
      ++total;pieces.push_back({cls,1,true});if(pieces.size()>64)return;
    }
  }
  if(pieces.size()<2 || !binder)return;
  if(!heads.charge(1+root.type.size()+std::bit_width(targets.size()+1)))return;
  auto candidates=targets.find({root.type,*binder,total});if(candidates==targets.end())return;
  for(auto target:candidates->second) {
    if(!heads.charge())return;
    if(find(target)==find(id))continue;
    if(bitPermutedFoldCover(pieces,snapshot[target].children[1],*binder,snapshot,heads)) {
      rewrite(id,target);coalesced_forests_.insert(id);return;
    }
    if(!remaining || heads.exhausted)return;
    std::map<std::uint64_t,std::vector<std::pair<std::uint64_t,std::uint64_t>>> domains;
    std::optional<std::uint64_t> stride;bool valid=true;
    struct Domain {std::uint64_t begin,count;bool singleton;};
    std::vector<Domain> mapped;
    for(const auto& piece:pieces) {
      if(!heads.charge())return;
      if(piece.singleton && dependsOn(piece.body,*binder)){valid=false;break;}
      auto map=affineBodyMap(snapshot[target].children[1],piece.body,*binder,snapshot,members,
          piece.singleton,&remaining,&affine_cache,&heads);
      if(!remaining || heads.exhausted)return;
      if(!map || (!piece.singleton && stride && *stride!=map->first) || !map->first || map->first>pieces.size()){valid=false;break;}
      if(!piece.singleton)stride=map->first;
      mapped.push_back({map->second,piece.count,piece.singleton});
    }
    if(!valid || !stride)continue;
    for(auto domain:mapped) {
      if(!heads.charge(1+std::bit_width(domains.size()+1)))return;
      const auto start=domain.begin/ *stride,count=domain.count;
      if(count>UINT64_MAX-start){valid=false;break;}
      domains[domain.begin% *stride].emplace_back(start,start+count);
    }
    if(!valid || !stride || domains.size()!=*stride)continue;
    for(auto& [residue,ranges]:domains) {
      if(!heads.charge(1+ranges.size()*(1+std::bit_width(ranges.size()+1))))return;
      std::sort(ranges.begin(),ranges.end());std::uint64_t end=0;
      for(auto [begin,next]:ranges) {
        if(!heads.charge())return;if(begin!=end){valid=false;break;}end=next;
      }
      const auto required=total/ *stride+(residue<total% *stride);
      if(end!=required)valid=false;
    }
    if(valid) {
      if(!heads.charge())return;
      rewrite(id,target);coalesced_forests_.insert(id);return;
    }
  }
}

inline auto Engine::affineValue(Class id,AffineCache& cache,unsigned depth,
                                      std::size_t* remaining,
                                      std::size_t max_entries,std::size_t max_terms) -> std::optional<IndexAffine> {
    poll();if(remaining){if(!*remaining)return {};--*remaining;}
    id=find(id);if(depth>256)return {};
    if(auto found=cache.find(id);found!=cache.end()) {
      // A shared cache may have been populated by a less restrictive
      // caller. Enforce this query's term bound BEFORE copying its map.
      if(max_terms && found->second && found->second->coefficients.size()>max_terms) {
        if(remaining)*remaining=0;return {};
      }
      if(remaining && found->second) {
        const auto copied=found->second->coefficients.size();
        if(copied>*remaining){*remaining=0;return {};}*remaining-=copied;
      }
      return found->second;
    }
    if(max_entries && cache.size()>=max_entries) {
      if(remaining)*remaining=0;return {};
    }
    const auto& node=nodes_[id];std::optional<IndexAffine> result;
    // Zero cache-entry limit does not disable bounded-query metadata checks.
    if((max_entries || max_terms) && (node.type.size()>128 || node.name.size()>4096 ||
        node.attributes.size()>128 || node.provenance.size()>4096 || node.children.size()>64)) {
      if(remaining)*remaining=0;return {};
    }
    if((node.type=="exact:u64" || node.type=="u64") && node.provenance.empty()) {
      if(node.kind==static_cast<int>(s::Kind::Index))
        result=IndexAffine{0,{{node.index_level,1}}};
      else if(node.kind==static_cast<int>(s::Kind::Constant)) {
        std::uint64_t value=0;
        auto parsed=std::from_chars(node.name.data(),node.name.data()+node.name.size(),value);
        if(parsed.ec==std::errc{} && parsed.ptr==node.name.data()+node.name.size())
          result=IndexAffine{value,{}};
      } else if(node.kind==static_cast<int>(s::Kind::Operation) &&
                node.attributes.empty() && node.children.size()==2) {
        auto a=affineValue(node.children[0],cache,depth+1,remaining,max_entries,max_terms),
             b=affineValue(node.children[1],cache,depth+1,remaining,max_entries,max_terms);
        if(a && b) {
          if(remaining) {
            const auto copied=a->coefficients.size()+b->coefficients.size();
            if(copied>*remaining){*remaining=0;return {};}*remaining-=copied;
          }
          result=combineAffine(node.name,*a,*b);
        }
      }
    }
    if(max_terms && result && result->coefficients.size()>max_terms) {
      if(remaining)*remaining=0;return {};
    }
    if(remaining && result) {
      const auto copied=result->coefficients.size();
      if(copied>*remaining){*remaining=0;return {};}*remaining-=copied;
    }
    if(max_entries && cache.size()>=max_entries) {
      if(remaining)*remaining=0;return {};
    }
    cache.emplace(id,result);return result;
}

inline auto Engine::shiftedProjectedBody(Class first,Class next,std::uint32_t binder,std::uint64_t shift,
                          std::uint64_t count) -> bool {
  first=find(first);next=find(next);
  const auto query=std::tuple{first,next,binder,shift,count};
  if(auto found=shifted_body_matches_.find(query);found!=shifted_body_matches_.end())return found->second;
  constexpr std::size_t max_queries=4096,max_pairs=4096,max_affine_nodes=4096,max_terms=64;
  constexpr std::size_t max_local_work=131072;
  const auto capped=[&](const char* ledger) {
    if(!shifted_body_cap_reported_ && sentryward::debug::environment("SENTRYWARD_EGRAPH_TRACE")) {
      shifted_body_cap_reported_=true;
      SENTRYWARD_DEBUG_PRINT(egraph_142);
    }
    return false;
  };
  if(shifted_body_matches_.size()>=max_queries)return capped("queries");
  // A body match visits template pairs, not reduction members. Memoization
  // lasts for one congruence epoch; a new union can expose a new match.
  const auto finish=[&](bool result){shifted_body_matches_.emplace(query,result);return result;};
  std::size_t remaining=max_local_work;
  const auto charge=[&](std::size_t amount) {
    if(amount>remaining){remaining=0;return false;}
    remaining-=amount;poll();
    if(options_.max_work && amount>1) {
      if(amount-1>options_.max_work-work_)throw Limit{"work_budget"};
      work_+=amount-1;
    }
    return true;
  };
  const auto boundedHead=[](const Key& key) {
    return key.type.size()<=128 && key.name.size()<=4096 && key.attributes.size()<=128 &&
           key.provenance.size()<=4096 && key.children.size()<=64;
  };
  AffineCache cache;
  namespace bit=bounded_index_translation;
  std::optional<bit::Session> euclidean;
  if(options_.outer_index_extents.size()<64) {
  std::vector<bit::IndexExtent> extents;
  for(std::size_t level=0;level<options_.outer_index_extents.size();++level) {
    if(!charge(1))return finish(capped("work"));
    extents.push_back({static_cast<std::uint32_t>(level),options_.outer_index_extents[level]});
  }
  // One scoped dual-DAG session per body match. Only stored exact u64 heads
  // are resolved; no e-class-member, ancestor, or logical Fold-member scan.
  euclidean.emplace([&](Class id)->std::optional<bit::Node> {
    const auto& k=nodes_[find(id)];
    if(!boundedHead(k) || !k.attributes.empty() || !k.provenance.empty() || k.fold || k.children.size()>2)return {};
    bit::Node result;result.type=k.type;result.admitted=true;
    if(k.kind==static_cast<int>(s::Kind::Constant)) {
      result.kind=bit::Kind::Literal;result.literal=k.name;
    } else if(k.kind==static_cast<int>(s::Kind::Index)) {
      result.kind=bit::Kind::Index;result.index_level=k.index_level;
    } else if(k.kind==static_cast<int>(s::Kind::Operation)) {
      if(k.name=="Add")result.kind=bit::Kind::Add;
      else if(k.name=="Multiply")result.kind=bit::Kind::Multiply;
      else if(k.name=="Divide")result.kind=bit::Kind::Divide;
      else if(k.name=="Remainder")result.kind=bit::Kind::Remainder;
      else return {};
    } else return {};
    result.child_count=k.children.size();
    std::copy(k.children.begin(),k.children.end(),result.children.begin());return result;
  },extents,binder,shift,count,bit::Limits{},charge);
  }
  struct Task {Class first,next;bool ready=false;};
  std::vector<Task> work{{first,next}};
  // A cycle introduced by e-class equality is not a structural proof. Only
  // completed pairs may be reused; recursive pairs conservatively fail.
  std::map<std::pair<Class,Class>,unsigned char> visited;
  while(!work.empty()) {
    if(!charge(1))return finish(capped("work"));
    auto [a,b,ready]=work.back();work.pop_back();a=find(a);b=find(b);
    const auto pair=std::pair{a,b};
    if(ready){visited[pair]=2;continue;}
    if(auto found=visited.find(pair);found!=visited.end()) {
      if(found->second==1)return finish(false);
      continue;
    }
    if(visited.size()>=max_pairs)return finish(capped("pairs"));
    if(a==b && !dependsOn(a,binder)){visited[pair]=2;continue;}
    if(euclidean && euclidean->equivalent(a,b)){visited[pair]=2;continue;}
    auto x=affineValue(a,cache,0,&remaining,max_affine_nodes,max_terms),
         y=affineValue(b,cache,0,&remaining,max_affine_nodes,max_terms);
    if(!remaining)return finish(capped("affine_or_work"));
    if(x && y) {
      if(!shiftedAffine(*x,*y,binder,shift))return finish(false);
      visited[pair]=2;
      continue;
    }
    const auto& lhs=nodes_[a];const auto& rhs=nodes_[b];
    if(!boundedHead(lhs) || !boundedHead(rhs))return finish(capped("metadata"));
    if(lhs.kind!=rhs.kind || lhs.type!=rhs.type || lhs.name!=rhs.name ||
        lhs.attributes!=rhs.attributes || lhs.provenance!=rhs.provenance ||
        lhs.fold!=rhs.fold || lhs.index_level!=rhs.index_level ||
        lhs.children.size()!=rhs.children.size())return finish(false);
    if(lhs.kind==static_cast<int>(s::Kind::Index) && lhs.index_level==binder && shift)return finish(false);
    visited[pair]=1;work.push_back({a,b,true});
    const auto sameHead=[&](const Key& x,const Key& y) {
      return boundedHead(x) && boundedHead(y) && x.kind==y.kind && x.type==y.type && x.name==y.name && x.attributes==y.attributes &&
          x.provenance==y.provenance && x.fold==y.fold && x.index_level==y.index_level &&
          x.children.size()==y.children.size();
    };
    // E-class IDs need not sort corresponding commutative operands alike
    // after an index shift. Select a uniquely determined head permutation;
    // ambiguous heads stay conservative, with no exponential backtracking.
    bool swapped=false;
    if((arithmetic(lhs) && arithmetic(rhs)) ||
        (integerCommutative(lhs) && integerCommutative(rhs))) {
      const auto& x=nodes_[find(lhs.children[0])];const auto& y=nodes_[find(lhs.children[1])];
      const auto& u=nodes_[find(rhs.children[0])];const auto& v=nodes_[find(rhs.children[1])];
      swapped=(!sameHead(x,u) || !sameHead(y,v)) && sameHead(x,v) && sameHead(y,u);
    }
    for(std::size_t i=0;i<lhs.children.size();++i) {
      if(lhs.kind==static_cast<int>(s::Kind::Fold) && i==1 && lhs.fold->binder_level==binder) {
        if(find(lhs.children[i])!=find(rhs.children[i]))return finish(false);
        continue;
      }
      work.push_back({lhs.children[i],rhs.children[swapped?1-i:i],false});
    }
  }
  return finish(true);
}

inline auto Engine::concatenateProjectedFold(Key& key) -> void {
  if(key.kind!=static_cast<int>(s::Kind::Fold) || key.type!="real" ||
      key.children.size()!=2 || !key.provenance.empty())return;
  const auto& prefix=nodes_[key.children[0]];
  if(prefix.kind!=key.kind || prefix.type!=key.type || prefix.children.size()!=2 ||
      !prefix.provenance.empty() || prefix.fold->reducer!=key.fold->reducer ||
      prefix.fold->binder_level!=key.fold->binder_level ||
      key.fold->count>UINT64_MAX-prefix.fold->count)return;
  if(!shiftedProjectedBody(prefix.children[1],key.children[1],
                           key.fold->binder_level,prefix.fold->count,key.fold->count))return;
  key.fold->count+=prefix.fold->count;
  key.children=prefix.children;
}

} // namespace
} // namespace sentryward::equivalence
