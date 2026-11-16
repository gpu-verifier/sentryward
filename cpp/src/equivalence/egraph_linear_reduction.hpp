#pragma once
#include "egraph_engine.hpp"

namespace sentryward::equivalence {
namespace {

inline auto Engine::mergeLinearReductionNormalForms(const std::vector<Key>& snapshot,
    const std::vector<bool>&,std::span<const Class> left,
    std::span<const Class> right) -> void {
  namespace l=bounded_linear_reduction;
  // One frozen RANKED selected-head DAG and one summary cache per epoch. Candidate
  // roots are indexed by type/context/body skeleton, then exact weighted
  // interval keys: no root-pair product, no generated enodes or Fold elements.
  constexpr std::size_t max_candidates=1024,max_active_nodes=4096,max_cached_terms=32768;
  constexpr std::size_t max_root_candidates=max_candidates/2;
  constexpr std::size_t max_signatures=1024,max_interval_entries=32768,max_groups=256,max_mappings=8192;
  // ONE physical stored-header capacity for retained planner metadata.
  // Active walks, candidate fanout and complete mapper geometry keep their
  // separate fixed limits: more memo slots are never an equality fact.
  epoch_work_allowance::Size summary_basis;summary_basis.headers=snapshot.size();
  const auto max_retained_summary_nodes=epoch_work_allowance::retainedHeaderCapacity(summary_basis);
  struct LocalCap {};
  std::size_t initial_work=2000000,remaining=initial_work,policy_work=0,charged_helper=0;
  epoch_work_allowance::Size physical_epoch;
  const char* phase="linear_setup";const char* cap_ledger="optional_work";
  std::size_t cap_entries=0;bool shape_capped=false;
  const auto localCap=[&](const char* ledger,std::size_t entries=0) {
    cap_ledger=ledger;cap_entries=entries;throw LocalCap{};
  };
  scoped_work_budget::Guards* scoped_sum_query=nullptr;bool synchronizing_helper=false;
  const auto charge=[&](std::size_t amount=1) {
    poll();
    // Resolver/catalog work uses this same pre-work path. The sync delta
    // has already passed the primitive hook and may not debit query twice.
    if(scoped_sum_query && !synchronizing_helper) {
      if(!scoped_sum_query->charge(amount))localCap("scoped_sum_query",0);
    }
    if(amount>remaining)localCap("optional_work",initial_work-remaining);remaining-=amount;
    if(options_.max_work) {
      if(amount>options_.max_work-work_)throw Limit{"work_budget"};work_+=amount;
    }
  };
  const auto traceCap=[&](std::size_t helper_work=0) {
    if(sentryward::debug::environment("SENTRYWARD_EGRAPH_TRACE"))
      SENTRYWARD_DEBUG_PRINT(egraph_4);
  };
  std::vector<std::pair<Class,Class>> equalities;
  // Reducer contexts share the original optional ledgers. Extrema use
  // interval-first selected-DAG summaries, never a max/min -> real sum rule.
  std::size_t candidate_entries=0,summary_entries=0,cached_terms=0;
  std::size_t interval_entries=0,residual_entries=0,mapping_entries=0;
  std::size_t signature_entries=0,group_entries=0,shape_entries=0,lexical_entries=0,coverage_entries=0,constant_entries=0;
  std::array<bool,3> modes{};
  try {
  for(auto id:fold_head_ids_) {
    if(id>=snapshot.size())break;
    const auto& k=snapshot[id];
    charge();if(k.kind!=static_cast<int>(s::Kind::Fold) || !k.fold || !k.fold->count)continue;
    if(k.fold->reducer==s::Reducer::Max)modes[0]=true;
    if(k.fold->reducer==s::Reducer::Min)modes[1]=true;
    if(k.fold->reducer==s::Reducer::Sum)modes[2]=true;
  }
  } catch(const LocalCap&) {traceCap();return;}
  if(std::none_of(modes.begin(),modes.end(),[](bool value){return value;}))return;
  const auto plan=physicalWorkPlan(snapshot,initial_work,initial_work-remaining,true);
  initial_work=plan.initial;remaining=plan.remaining;policy_work=plan.sizing;physical_epoch=plan.physical;
  if(plan.exhausted){traceCap();return;}
  // PLAN FIRST: neither helper has done work before the mode gate/sizing.
  // One lifetime allowance borrows the SAME physically derived remainder;
  // reducer/query/root transitions never reconstruct or replenish it.
  l::Limits limits;limits.max_nodes=max_retained_summary_nodes;limits.max_cached_terms=max_cached_terms;
  limits.max_work=remaining;
  if(options_.max_work)limits.max_work=std::min(limits.max_work,options_.max_work-work_);
  limits.coefficients.max_work=limits.max_work;
  // Linear and exact-coefficient hooks precharge the SAME coverage aggregate
  // and currently active body guard. The existing sync() still owns the ONE
  // epoch/global debit. Legacy planner queries keep a null relay.
  l::Builder builder(limits,[&](std::size_t amount) {
    if(!scoped_sum_query)return true;
    return scoped_sum_query->charge(amount);
  });
  const auto sync=[&] {
    const auto delta=builder.work()-charged_helper;
    struct SyncGuard {bool& active;bool previous;SyncGuard(bool& a):active(a),previous(a){active=true;}
      ~SyncGuard(){active=previous;}} guard(synchronizing_helper);
    charge(delta);
    charged_helper=builder.work();
    if(builder.exhausted())localCap(scoped_sum_query && scoped_sum_query->exhausted()?
        "scoped_sum_query":"linear_builder",builder.work());
  };
  const auto settleHelperFailure=[&] {
    // Primitive work already passed the active-query precharge, but a
    // resolver/cache LocalCap may bypass sync(). Invoice its pending delta
    // once even when the optional epoch has no room left. No proof activity,
    // fresh allowance, query debit, or throwing unwinding destructor occurs.
    const auto delta=builder.work()-charged_helper;
    if(!delta)return;
    if(options_.max_work) {
      if(delta>options_.max_work-work_)throw Limit{"work_budget"};
      work_+=delta;
    }
    remaining=delta>remaining?0:remaining-delta;
    charged_helper=builder.work();
  };
  // All modes borrow ONE immutable epoch catalog. Preserve the full stored
  // class minimum-rank theorem and all address/body witnesses; only later
  // numeric scheduling scans use sparse ascending stored-head ID lists.
  const auto n=snapshot.size();
  struct ModeCatalog {
    std::unordered_map<Class,Class> selected,ordinary_folds,seeded_folds,sum_folds;
    std::vector<Class> eligible_ids,fold_ids,schedule_ids;
  };
  std::array<ModeCatalog,3> mode_catalogs;
  // No whole-node owner/rank/member reconstruction. Union maintains minimum_
  // and actual class membership; this immutable proof epoch borrows them.
  const auto owner_epoch=unions_;
  struct Owners {
    std::function<Class(Class)> lookup;
    Class operator[](Class id) const {return lookup(id);}
  } owners{[&](Class id) {
    if(unions_!=owner_epoch)throw std::logic_error("linear catalog epoch changed");
    if(id>=n)localCap("frozen_owner",id);
    const auto owner=find(id);if(owner>=n)localCap("frozen_owner",owner);
    return owner;
  }};
  struct RankView {
    std::function<bool(Class)> lookup;
    bool operator[](Class id) const {return lookup(id);}
  };
  std::unordered_map<Class,bool> rank_cache;
  RankView ranked{[&](Class id) {
    if(unions_!=owner_epoch)throw std::logic_error("linear catalog epoch changed");
    if(auto found=rank_cache.find(id);found!=rank_cache.end())return found->second;
    charge();const auto owner=owners[id];const auto rank=minimum_[owner];
    bool result=true;
    for(auto child:snapshot[id].children) {
      charge();if(minimum_[owners[child]]>=rank)result=false;
    }
    rank_cache.emplace(id,result);return result;
  }};
  struct SparseFlags {
    std::unordered_set<Class> values;
    bool operator[](Class id) const {return values.contains(id);}
  } admitted;
  struct MemberView {
    std::size_t boundary;
    std::function<std::span<const Class>(Class)> lookup;
    std::size_t size() const {return boundary;}
    std::span<const Class> operator[](Class id) const {return lookup(id);}
  };
  // Map nodes retain their vectors across rehash; only this epoch owns views.
  // Full mapper views cap actual PREFIX members at64, not appended members.
  // Ranked lexical/rectangle views retain every old ranked alternative.
  std::unordered_map<Class,std::optional<std::vector<Class>>> ranked_views,body_views;
  MemberView members{n,[&](Class cls)->std::span<const Class> {
    cls=owners[cls];
    auto found=ranked_views.find(cls);
    if(found==ranked_views.end()) {
      auto view=class_members_.prefix(cls,n,n,[&](std::size_t amount){charge(amount);return true;});
      if(view) {
        auto out=view->begin();
        for(auto id:*view){charge();if(ranked[id])*out++=id;}
        view->erase(out,view->end());
      }
      found=ranked_views.emplace(cls,std::move(view)).first;
    }
    return found->second?std::span<const Class>{*found->second}:std::span<const Class>{};
  }};
  MemberView body_members{n,[&](Class cls)->std::span<const Class> {
    cls=owners[cls];
    auto found=body_views.find(cls);
    if(found==body_views.end()) {
      auto view=class_members_.prefix(cls,n,BodyHeadIndex::max_members,
          [&](std::size_t amount){charge(amount);return true;});
      found=body_views.emplace(cls,std::move(view)).first;
    }
    return found->second?std::span<const Class>{*found->second}:std::span<const Class>{};
  }};
  std::vector<Class> owner_ids,numeric_candidates;
  std::unordered_map<Class,std::string> literals;
  bool catalog_ready=false,catalog_failed=false;
  // Optional scheduling is demanded by the SAME paired-root half pool, not
  // every numerical head ever produced by saturation. Each reached eclass
  // discovers dependencies through ALL physical snapshot aliases. Numerical
  // admission still uses the original minimum-rank theorem. Exact addresses
  // are opaque to this numeric scheduling walk;
  // the full scoped mapper still borrows ALL original body/address members.
  std::vector<Class> demanded_heads;
  std::size_t demanded_owners=0,demanded_root_entries=0,demanded_aliases=0,demanded_unranked=0;
  bool demand_ready=false;
  const auto ensureDemand=[&] {
    if(demand_ready)return;
    std::unordered_set<Class> reached,root_window;
    std::vector<Class> pending;
    const auto reach=[&](Class cls) {
      charge();cls=owners[cls];
      if(realType(snapshot[cls].type) && reached.insert(cls).second) {
        charge(3);pending.push_back(cls);
      }
    };
    const auto root=[&](Class cls) {
      cls=owners[cls];
      if(root_window.insert(cls).second){++demanded_root_entries;reach(cls);}
    };
    for(std::size_t i=0;i<std::max(left.size(),right.size());++i) {
      charge();
      const bool certified=i<left.size() && i<right.size() && owners[left[i]]==owners[right[i]];
      if(certified) {
        // Explicit Fold output targets preserve their existing nullable body
        // group/collision index, even when the mate is already certified.
        // Do not recursively promote equal non-Fold output wrappers/noise.
        if(snapshot[left[i]].kind==static_cast<int>(s::Kind::Fold) ||
            snapshot[right[i]].kind==static_cast<int>(s::Kind::Fold))reach(left[i]);
        continue;
      }
      // Reserve the two possible mate slots exactly as appendRoot does.
      // Missing mates remain ordinary optional roots, never an equality.
      if(demanded_root_entries+2>max_root_candidates)continue;
      if(i<left.size())root(left[i]);
      if(i<right.size())root(right[i]);
    }
    while(!pending.empty()) {
      charge();const auto cls=pending.back();pending.pop_back();++demanded_owners;
      charge(3);owner_ids.push_back(cls);
      // Discovery is NOT an acyclic proof witness. A later compact Fold may
      // fail the min-owner rank test yet be chosen by the full scoped mapper;
      // its actual children still need numerical prerequisite catalogs. Scan
      // all physical aliases once, then compact the SAME owned prefix vector
      // into the original ranked view. Reached owners break alias cycles.
      auto view=class_members_.prefix(cls,n,n,
          [&](std::size_t amount){charge(amount);return true;});
      if(!view)localCap("demand_members",cls);
      // Existing literal precedence is also a discovery atom boundary.
      // Otherwise zero's unrelated x-x/zero-Fold alternatives reconnect
      // every numerical ancestor despite the demanded-root window. This
      // is the SAME finite/plain real literal guard as ensureCatalog(),
      // not an intersection dependency fact or a new equality.
      bool literal_boundary=false;
      for(auto head:*view) {
        charge();const auto& k=snapshot[head];
        if(k.kind==static_cast<int>(s::Kind::Constant) && realType(k.type) &&
            k.attributes.empty() && k.provenance.empty() && finite_[cls] && ranked[head]) {
          literal_boundary=true;break;
        }
      }
      auto ranked_out=view->begin();
      for(auto head:*view) {
        charge();const auto& k=snapshot[head];++demanded_aliases;
        const bool head_ranked=ranked[head];
        if(head_ranked) {
          *ranked_out++=head;
          if(realType(k.type)){charge(3);demanded_heads.push_back(head);}
        } else ++demanded_unranked;
        // InitialCell reads preserve exact address/provenance semantics on
        // their untouched body-map view. No numeric rank/catalog is required
        // for their u64 address, raw bit or opaque exact producer ancestors.
        if(!literal_boundary)for(auto child:k.children) {
          charge();if(realType(snapshot[owners[child]].type))reach(child);
        }
      }
      view->erase(ranked_out,view->end());
      charge();ranked_views.try_emplace(cls,std::move(view));
    }
    if(!stored_class_catalog::Members::orderIds(demanded_heads,n,
        [&](std::size_t amount){charge(amount);return true;}))localCap("demand_order",demanded_heads.size());
    if(!stored_class_catalog::Members::orderIds(owner_ids,n,
        [&](std::size_t amount){charge(amount);return true;}))localCap("demand_owner_order",owner_ids.size());
    demand_ready=true;
  };
  const auto modeIndex=[](s::Reducer reducer) {
    return reducer==s::Reducer::Max?0U:reducer==s::Reducer::Min?1U:2U;
  };
  const auto eligibleFor=[&](const Key& k,s::Reducer reducer) {
    if(!idealOperation(k) || !k.provenance.empty())return false;
    if(reducer!=s::Reducer::Sum) {
      if(k.type!="real" || k.children.size()!=2 ||
          k.name!=(reducer==s::Reducer::Max?"Maximum":"Minimum"))return false;
    } else if(k.name!="Add" && k.name!="Subtract" && k.name!="Negate" && k.name!="Multiply" &&
        k.name!="Divide" && k.name!="Reciprocal")return false;
    return std::all_of(k.children.begin(),k.children.end(),[&](auto c) {
      return snapshot[c].type==k.type && finite_[owners[c]];
    });
  };
  const auto ensureCatalog=[&] {
    if(catalog_ready)return;
    if(n>std::numeric_limits<l::Atom>::max()/2)localCap("private_atom_ids",n);
    ensureDemand();
    for(auto id:demanded_heads) {
      charge();const auto& k=snapshot[id];const auto owner=owners[id];
      // The separately ordered owner list includes a cyclic/opaque minimum
      // owner too, when its LATER ranked alias is the actual chosen witness.
      // Requiring the minimum head itself to rank would discard that alias.
      if(!realType(k.type) || !k.provenance.empty() || !finite_[owner] || !ranked[id])continue;
      admitted.values.insert(owner);
      if(k.kind==static_cast<int>(s::Kind::Constant) && k.attributes.empty())literals.try_emplace(owner,k.name);
      if(k.kind==static_cast<int>(s::Kind::Fold) || k.kind==static_cast<int>(s::Kind::Operation))
        numeric_candidates.push_back(id);
    }
    catalog_ready=true;
  };
  const auto run_mode=[&](s::Reducer reducer) {
  if(catalog_failed)return;
  const bool extrema=reducer!=s::Reducer::Sum;
  l::Statistics linear_statistics;
  const bool trace=sentryward::debug::environment("SENTRYWARD_EGRAPH_TRACE")!=nullptr;
  const auto preparation_start=remaining;
  std::optional<std::size_t> setup_end,groups_end,scheduling_end;
  struct MapTrace {std::size_t calls=0,work=0,start=0;bool active=false;};
  MapTrace origin_trace,domain_trace;
  BodyHeadIndex::HeaderStatistics header_statistics;
  std::size_t origin_identity=0,domain_queries=0,domain_hits=0,domain_identity=0;
  std::size_t frontier_nodes=0,frontier_edges=0,frontier_targets=0;
  std::size_t constant_head_calls=0,constant_head_tasks=0;
  std::size_t constant_cached_false_hits=0,constant_child_cached_false=0;
  std::size_t domain_fallback_reports=16;
  std::size_t domain_denial_reports=16;
  std::size_t coverage_denial_reports=16;
  const auto tracedMap=[&](MapTrace& stats,auto&& invoke) {
    if(trace){++stats.calls;stats.start=remaining;stats.active=true;}
    auto result=invoke();
    if(trace){stats.work+=stats.start-remaining;stats.active=false;}
    return result;
  };
  try {
    phase="linear_setup";
    ensureCatalog();
    auto& catalog=mode_catalogs[modeIndex(reducer)];
    const auto& selected=catalog.selected;
    const auto& ordinary_folds=catalog.ordinary_folds;
    const auto& seeded_folds=catalog.seeded_folds;
    const auto& sum_folds=catalog.sum_folds;
    // Use this strict rank mask exclusively, never OR with the global SCC
    // mask. sum=2*mean can create a proven class SCC even though the original
    // stored Divide(Add(samples),2) is an acyclic lower-ranked witness.
    const auto eligible=[&](const Key& k) {return eligibleFor(k,reducer);};
    // Mode admission remains lazy and ordered: a future Sum catalog cannot
    // spend optional work before an earlier Max/Min proof. The demand catalog
    // retains EVERY finite ranked Operation/Fold alternative of each reached
    // owner, not only a selected head or a recursively expanded shape hint.
    for(auto id:numeric_candidates) {
      charge();const auto& k=snapshot[id];const auto owner=owners[id];
      if(k.kind==static_cast<int>(s::Kind::Fold) && k.fold && k.fold->reducer==reducer &&
          k.children.size()==2 && k.attributes.empty() &&
          snapshot[k.children[0]].type==k.type && snapshot[k.children[1]].type==k.type &&
          (!extrema || k.type=="real") && finite_[owners[k.children[1]]] &&
          (finite_[owners[k.children[0]]] ||
           (reducer==s::Reducer::Max && negative_infinity_[owners[k.children[0]]]) ||
           (reducer==s::Reducer::Min && positive_infinity_[owners[k.children[0]]]))) {
        catalog.sum_folds.try_emplace(owner,id);
        const bool neutral=reducer==s::Reducer::Sum?zero_[owners[k.children[0]]]:
            reducer==s::Reducer::Max?(negative_infinity_[owners[k.children[0]]] ||
                (k.fold->count && zero_[owners[k.children[0]]] && nonnegative_[owners[k.children[1]]])):
            positive_infinity_[owners[k.children[0]]];
        auto& index=neutral?catalog.ordinary_folds:catalog.seeded_folds;
        if(index.try_emplace(owner,id).second) {
          catalog.fold_ids.push_back(id);catalog.schedule_ids.push_back(id);
        }
      }
      if(eligible(k)) {
        // Keep EVERY eligible alternate for moment primitive indexing.
        catalog.eligible_ids.push_back(id);
        if(catalog.selected.try_emplace(owner,id).second)catalog.schedule_ids.push_back(id);
      }
    }
    // Exact dependencies of ONE selected expression, not the conservative
    // union of dependencies across all eclass alternatives (x-x can pollute
    // a zero class with a binder that this chosen constant does not use).
    std::unordered_map<Class,std::optional<std::vector<std::uint32_t>>> lexical;
    const auto lexicalFree=[&](Class root)->std::optional<std::vector<std::uint32_t>> {
      struct Task {Class cls,head;bool ready;};
      std::vector<Task> pending{{owners[root],n,false}};std::unordered_set<Class> active;
      while(!pending.empty()) {
        charge();auto t=pending.back();pending.pop_back();t.cls=owners[t.cls];
        if(lexical.contains(t.cls))continue;
        if(lexical_entries+active.size()>=max_retained_summary_nodes)
          localCap("lexical_nodes",lexical_entries+active.size());
        if(active.size()>=max_active_nodes)localCap("lexical_active",active.size());
        // The public dependency property is the conservative UNION of all
        // alternatives after the epoch's worklist reaches a fixed point.
        // Empty therefore certifies every chosen witness is closed; it is
        // not an Unknown initialized to empty or an intersection guess.
        if(free_indices_[t.cls].empty()) {
          lexical.emplace(t.cls,std::vector<std::uint32_t>{});++lexical_entries;active.erase(t.cls);continue;
        }
        if(!t.ready) {
          if(!active.insert(t.cls).second)localCap("lexical_cycle",active.size());
          if(literals.contains(t.cls)) {
            lexical.emplace(t.cls,std::vector<std::uint32_t>{});++lexical_entries;active.erase(t.cls);continue;
          }
          if(auto h=selected.find(t.cls);h!=selected.end())t.head=h->second;
          else if(auto h=ordinary_folds.find(t.cls);h!=ordinary_folds.end())t.head=h->second;
          else if(!members[t.cls].empty())t.head=members[t.cls].front();
          if(t.head==n) {lexical.emplace(t.cls,std::nullopt);++lexical_entries;active.erase(t.cls);continue;}
          pending.push_back({t.cls,t.head,true});
          for(auto c:snapshot[t.head].children)if(!lexical.contains(owners[c]))pending.push_back({owners[c],n,false});
          continue;
        }
        const auto& k=snapshot[t.head];std::vector<std::uint32_t> indices;
        bool valid=k.provenance.empty();
        if(k.kind==static_cast<int>(s::Kind::Index))indices.push_back(k.index_level);
        for(std::size_t i=0;i<k.children.size() && valid;++i) {
          const auto c=lexical.find(owners[k.children[i]]);if(c==lexical.end() || !c->second){valid=false;break;}
          for(auto level:*c->second)if(!(k.fold && i==1 && level==k.fold->binder_level))indices.push_back(level);
        }
        charge(indices.size());std::sort(indices.begin(),indices.end());
        indices.erase(std::unique(indices.begin(),indices.end()),indices.end());
        if(indices.size()>8)valid=false;
        lexical.emplace(t.cls,valid?std::optional{std::move(indices)}:std::nullopt);++lexical_entries;active.erase(t.cls);
      }
      return lexical.at(owners[root]);
    };
    // Recursive shapes partition only the bounded stored Fold targets.
    // Candidate atoms use the fixed-local body index below, never this
    // ancestor walk. affineBodyMap still checks every actual leaf/context.
    AffineCache affine_cache;
    BodyHeadIndex body_heads_index(*this,snapshot,body_members,affine_cache,&remaining);
    // ONE opt-in token memo per immutable planner epoch, not per local Add
    // candidate or standalone body query on the same large snapshot.
    body_heads_index.initializeSoleHints(max_retained_summary_nodes,true);
    if(trace)body_heads_index.header_statistics=&header_statistics;
    std::unordered_map<Class,std::optional<Class>> shapes;
    std::unordered_map<Key,Class,Hash> shape_keys;
    std::unordered_set<Class> active_shapes;
    // Large closed factors are exact opaque CLASS identities in this private
    // candidate filter. A small scalar point still receives its normal shape
    // (so constants can match a Fold address); a closed subtree gets at most
    // 64 newly visited heads before retaining exact identity, never a wildcard.
    // Full body/context/affine checks remain the equality certificate.
    constexpr std::size_t closed_shape_visits=64;
    std::function<std::optional<Class>(Class,unsigned,std::size_t*)> shapeImpl;
    shapeImpl=[&](Class cls,unsigned depth,std::size_t* closed_remaining)->std::optional<Class> {
      charge();cls=owners[cls];
      if(auto found=shapes.find(cls);found!=shapes.end())return found->second;
      if(!closed_remaining && free_indices_[cls].empty()) {
        std::size_t local_remaining=closed_shape_visits;
        return shapeImpl(cls,depth,&local_remaining);
      }
      if(depth>128)return {};
      if(shape_entries+active_shapes.size()>=max_retained_summary_nodes){shape_capped=true;return {};}
      if(!active_shapes.insert(cls).second)return {};
      const auto finish=[&](std::optional<Class> result) {
        active_shapes.erase(cls);shapes.emplace(cls,result);++shape_entries;return result;
      };
      Key key;
      if(closed_remaining && !*closed_remaining) {
        if(!free_indices_[cls].empty())return finish({});
        key.kind=10;key.type="closed-eclass-shape-filter";key.children={cls};
      } else if(affineValue(cls,affine_cache,0,&remaining)) {
        if(closed_remaining)--*closed_remaining;
        key.kind=8;key.type="affine-u64-filter";
      }
      else {
        if(closed_remaining)--*closed_remaining;
        Class head=n;
        if(auto choice=body_heads_index.preferred(cls))head=*choice;
        // Local compact heads are scheduling hints only. The shape's own
        // active/depth/closed-factor limits bound this walk; the actual
        // paired body certificate below checks its chosen witness cycles.
        if(head==n)return finish({});
        const auto& source=snapshot[head];
        if(source.type.size()>4096 || source.name.size()>4096 || source.attributes.size()>4096 ||
            source.provenance.size()>4096 || source.children.size()>4096)return finish({});
        charge(source.type.size()+source.name.size()+source.attributes.size()+source.provenance.size()+source.children.size());
        key=snapshot[head];key.children.clear();
        // Candidate-only alpha shape. Every actual binder/count/reducer and
        // captured leaf is subsequently checked in the paired lexical map.
        if(key.fold)key.fold->binder_level=0;
        for(auto child:snapshot[head].children) {
          auto child_shape=shapeImpl(child,depth+1,closed_remaining);if(!child_shape)return finish({});
          key.children.push_back(*child_shape);
        }
        // Class IDs can place a common closed factor on opposite sides of
        // a template and a scalar point. Canonicalize only this candidate
        // filter; affineBodyMap still validates every actual operand/scope.
        if((arithmetic(source) || integerCommutative(source)) && key.children.size()==2 &&
            key.children[1]<key.children[0])std::swap(key.children[0],key.children[1]);
      }
      if(!remaining)return finish({});
      charge(key.children.size());
      auto [entry,inserted]=shape_keys.try_emplace(std::move(key),shape_keys.size());(void)inserted;
      return finish(entry->second);
    };
    const auto shape=[&](Class cls,unsigned depth) {return shapeImpl(cls,depth,nullptr);};
    const auto scopeFor=[&](Class cls)->std::optional<std::vector<std::uint32_t>> {
      cls=owners[cls];
      if(literals.contains(cls))return std::vector<std::uint32_t>{};
      // Fixed-point dependency UNION is a conservative lexical upper bound.
      // Missing binder proves absence; extra captured levels only reject a
      // match. This PRIVATE filter never establishes expression equality.
      // A chosen Fold's literal seed/body has certified empty dependencies,
      // even when x-x alternatives polluted that literal's public union.
      if(!extrema)if(auto f=sum_folds.find(cls);f!=sum_folds.end()) {
        const auto& k=snapshot[f->second];std::vector<std::uint32_t> levels;
        for(unsigned side=0;side<2;++side) {
          const auto child=owners[k.children[side]];if(literals.contains(child))continue;
          const auto& bound=free_indices_[child];charge(bound.size());
          if(bound.size()>8)return {};
          for(auto level:bound)if(side==0 || level!=k.fold->binder_level)levels.push_back(level);
        }
        charge(levels.size());std::sort(levels.begin(),levels.end());
        levels.erase(std::unique(levels.begin(),levels.end()),levels.end());
        if(levels.size()>8)return {};return levels;
      }
      // The full-body map below still checks every actual index operand,
      // seed and capture. Moments use a separate EXACT lexical index below.
      const auto& levels=free_indices_[cls];charge(levels.size());
      if(levels.size()>8)return {};return levels;
    };
    using BucketKey=std::tuple<std::string,std::vector<std::uint32_t>,Class,s::Reducer>;
    // The last token names a SCHEDULING component, not an equality fact.
    // One wildcard body skeleton can contain functions with incompatible
    // relative affine origins (e.g. x(i)*state(i+page)). Full maps below are
    // still the only body/coordinate certificates.
    using GroupKey=std::tuple<std::string,std::vector<std::uint32_t>,Class,s::Reducer,Class>;
    using Captures=std::vector<std::pair<std::uint32_t,std::uint64_t>>;
    using AnchorFamily=std::tuple<Class,std::uint64_t,std::uint32_t,Captures,std::uint64_t>;
    struct AnchorInterval {const AnchorFamily* family;std::uint64_t begin,end;};
    struct Group {
      Class fold;
      std::array<std::unordered_map<Class,std::optional<std::pair<std::uint64_t,std::uint64_t>>>,2> mapped;
      std::optional<AnchorInterval> anchor;
    };
    struct ReadAnchor {Class tag;IndexAffine address;};
    const auto uniqueReadAnchor=[&](Class body)->std::optional<ReadAnchor> {
      // Exactly THREE stored header levels, at most 1+2+4 heads. No address
      // ancestor walk, Fold BODY descent, candidate Cartesian search or
      // recursive signature expansion. The existing affine cache is shared.
      std::array<std::pair<Class,unsigned>,7> pending{};std::size_t size=1;
      pending[0]={owners[body],0};std::optional<Class> read;
      while(size) {
        charge();const auto [cls,depth]=pending[--size];
        auto head=body_heads_index.preferred(cls);if(!head)return {};
        const auto& k=snapshot[*head];
        const bool memory=k.kind==static_cast<int>(s::Kind::InitialCell) ||
            k.kind==static_cast<int>(s::Kind::InputValue);
        if(memory && realType(k.type) && k.children.size()==1 && finite_[owners[*head]]) {
          if(read && owners[*read]!=owners[*head])return {};
          read=*head;continue;
        }
        if(depth==2 || k.kind!=static_cast<int>(s::Kind::Operation) ||
            !idealOperation(k) || k.children.size()>2)continue;
        for(auto child:k.children)if(realType(snapshot[owners[child]].type)) {
          charge();if(size==pending.size())return {};
          pending[size++]={owners[child],depth+1};
        }
      }
      if(!read)return {};
      auto tag=body_heads_index.header(*read);if(!tag)return {};
      auto address=body_heads_index.affine(snapshot[*read].children[0]);
      if(!address || address->coefficients.size()>64)return {};
      charge(address->coefficients.size());return ReadAnchor{*tag,std::move(*address)};
    };
    struct Prototype {Class fold;const GroupKey* key;};
    struct Partition {
      // Each actual target contributes at most one immutable anchor record.
      // A collision at a coordinate is nullable, not alternative search.
      std::map<AnchorFamily,std::map<std::uint64_t,std::optional<Prototype>>> prototypes;
      std::size_t components=0;
      bool anchored=false,unsupported=false;
    };
    std::map<BucketKey,Partition> partitions;
    std::size_t prototype_entries=0;
    const auto targetAnchor=[&](const Key& k)->std::optional<std::pair<AnchorFamily,AnchorInterval>> {
      auto read=uniqueReadAnchor(k.children[1]);if(!read || !k.fold || !k.fold->count)return {};
      const auto binder=k.fold->binder_level;
      auto coefficient=read->address.coefficients.find(binder);
      if(coefficient==read->address.coefficients.end() || !coefficient->second)return {};
      const auto stride=coefficient->second,constant=read->address.constant;
      // Check the CONSTANT coordinate lattice, not an invented no-wrap
      // property for captured addresses. Full u64-ring/capture checks remain
      // mandatory. A wrapping lattice conservatively has no scheduling key.
      if(k.fold->count-1>(UINT64_MAX-constant)/stride)return {};
      const auto begin=constant/stride;
      if(k.fold->count>UINT64_MAX-begin)return {};
      Captures captures;captures.reserve(read->address.coefficients.size()-1);
      for(auto [level,value]:read->address.coefficients)if(level!=binder) {
        charge();if(level>=binder)return {};captures.push_back({level,value});
      }
      charge(captures.size()+8);
      // An absolute bound label is not part of this scheduling family.
      // Distinct depths can denote the same sample function. Keep EVERY
      // actual label in the full origin proof and canonical Fold below;
      // captured affine coefficients stay exact, never alpha-erased here.
      AnchorFamily family{read->tag,stride,0,std::move(captures),constant%stride};
      return std::pair{std::move(family),AnchorInterval{nullptr,begin,begin+k.fold->count}};
    };
    const auto sameBoundFoldBody=[&](const Key& a,const Key& b) {
      charge(8);
      // Pointwise equality of the SAME proven body, not a scheduling/header
      // hint. Callers separately retain the captured query scope and seed.
      // A different binder must still use the complete lexical body map.
      return a.kind==static_cast<int>(s::Kind::Fold) && b.kind==a.kind &&
          a.fold && b.fold && a.children.size()==2 && b.children.size()==2 &&
          a.type==b.type && a.fold->reducer==b.fold->reducer &&
          a.fold->binder_level==b.fold->binder_level &&
          a.attributes.empty() && b.attributes.empty() &&
          a.provenance.empty() && b.provenance.empty() &&
          snapshot[a.children[0]].type==a.type && snapshot[a.children[1]].type==a.type &&
          snapshot[b.children[0]].type==b.type && snapshot[b.children[1]].type==b.type &&
          owners[a.children[1]]==owners[b.children[1]];
    };
    std::map<GroupKey,Group> groups;
    // An admitted ACTUAL Fold header can retain its existing target group
    // without asking a lossy scalar-point hint to rediscover that group.
    // std::map keys are stable for this immutable epoch; no body or domain
    // equality is asserted by the borrowed key. Conflicting heads are Unknown.
    struct FoldMembership {Class head;const GroupKey* key;std::optional<AnchorInterval> anchor;};
    std::unordered_map<Class,std::optional<FoldMembership>> fold_memberships;
    std::vector<Class> membership_owners;
    std::set<std::pair<int,std::string>> body_heads;
    // Fixed-local metadata only. Do not recursively shape/map every stored
    // Fold before an actual certificate candidate has been selected.
    std::vector<Class> potential_target_ids;
    std::set<std::pair<std::string,std::vector<std::uint32_t>>> group_scopes;
    for(auto id:catalog.fold_ids) {
      charge();const auto owner=owners[id];
      const bool ordinary=ordinary_folds.contains(owner) && ordinary_folds.at(owner)==id;
      const bool seeded=extrema && seeded_folds.contains(owner) && seeded_folds.at(owner)==id;
      if((!ordinary && !seeded) || !snapshot[id].fold->count)continue;
      const auto body=owners[snapshot[id].children[1]];
      if(!extrema && (literals.contains(body) ||
          !dependsOn(body,snapshot[id].fold->binder_level)))continue;
      auto scope=scopeFor(owner);if(!scope)continue;
      const auto body_head=selected.contains(body)?selected.at(body):
          members[body].empty()?body:members[body].front();
      if(snapshot[body_head].name.size()>4096)continue;
      charge(8+snapshot[id].type.size()+scope->size()+snapshot[body_head].name.size());
      potential_target_ids.push_back(id);
      group_scopes.emplace(snapshot[id].type,std::move(*scope));
      body_heads.emplace(snapshot[body_head].kind,snapshot[body_head].name);
    }

    // Only a necessary scheduling discriminator, never a body certificate.
    // Construct it lazily, once per reducer/epoch, after the root-only filter
    // below. Unrelated output Divide/Multiply heads must not fill the shared
    // body matcher's retained-entry caches before any coverage query can run.
    bool body_shallow_ready=false;
    std::unordered_set<Class> body_shallow;
    const auto pointBodyHead=[&](Class owner) {
      const auto& alternatives=body_members[owner];
      if(alternatives.empty() || alternatives.size()>BodyHeadIndex::max_members)return false;
      bool possible=false;
      for(auto id:alternatives) {
        const auto& k=snapshot[id];
        if(k.name.size()>4096)continue;
        charge(1+k.name.size());
        if(body_heads.contains({k.kind,k.name})){possible=true;break;}
      }
      if(!possible)return false;
      if(!body_shallow_ready) {
        for(auto id:potential_target_ids) {
          charge();
          if(auto signature=body_heads_index.shallow(snapshot[id].children[1]))
            body_shallow.insert(*signature);
        }
        body_shallow_ready=true;
      }
      if(auto signature=body_heads_index.shallow(owner))return body_shallow.contains(*signature);
      return false;
    };
    // Coverage filters may carry conservative extra dependencies. Moment
    // semantics must not use those keys as an exact lexical context. Build
    // one lazy index of the existing (at most max_groups) stored targets,
    // retaining their public tokens only after an exact selected-DAG scope
    // check. Colliding target contexts are Unknown, never alternative search.
    bool exact_groups_ready=false;
    std::map<GroupKey,std::optional<GroupKey>> exact_groups;
    std::set<std::pair<std::string,std::vector<std::uint32_t>>> exact_group_scopes;
    const auto exactGroups=[&] {
      if(exact_groups_ready)return;
      for(const auto& [key,group]:groups) {
        charge();auto scope=lexicalFree(owners[group.fold]);if(!scope)continue;
        charge(std::get<0>(key).size()+scope->size());
        GroupKey exact{std::get<0>(key),*scope,std::get<2>(key),reducer,std::get<4>(key)};
        auto [entry,inserted]=exact_groups.try_emplace(exact,key);
        if(!inserted && entry->second && *entry->second!=key)entry->second.reset();
        exact_group_scopes.emplace(std::get<0>(key),std::move(*scope));
      }
      exact_groups_ready=true;
    };
    const auto admittedFoldGroup=[&](Class head,const std::string& type,
        const std::vector<std::uint32_t>& scope,bool semantic=false)->std::optional<GroupKey> {
      charge();if(head>=n)return {};
      const auto& source=snapshot[head];
      if(source.kind!=static_cast<int>(s::Kind::Fold) || !source.fold ||
          source.type!=type || source.fold->reducer!=reducer)return {};
      auto member=fold_memberships.find(owners[head]);
      if(member==fold_memberships.end() || !member->second || member->second->head!=head)return {};
      const auto& key=*member->second->key;
      if(std::get<0>(key)!=type || std::get<3>(key)!=reducer)return {};
      if(!semantic)return std::get<1>(key)==scope?std::optional{key}:std::nullopt;
      // Family mean queries remove an outer axis. Do not silently substitute
      // that different query scope for this Fold's actual enclosing context.
      auto actual=lexicalFree(owners[head]);if(!actual || *actual!=scope)return {};
      exactGroups();charge(type.size()+scope.size());
      auto found=exact_groups.find(GroupKey{type,scope,std::get<2>(key),reducer,std::get<4>(key)});
      return found==exact_groups.end()?std::nullopt:found->second;
    };
    // Index at most max_groups existing target groups by their shared public
    // fixed-radius scheduling signature. A shallow collision NEVER chooses a
    // target or merges groups: only a unique existing GroupKey is returned.
    // This is a necessary filter, not a body/coordinate equality fact.
    std::array<std::map<BucketKey,std::optional<GroupKey>>,2> body_groups;
    using AnchorSchema=std::pair<Class,Captures>;
    struct AnchorSegment {std::uint64_t begin,end;const GroupKey* key;};
    struct AnchorEvent {std::uint64_t coordinate;const GroupKey* key;bool begin;};
    struct AnchorLattice {
      std::uint64_t stride;std::uint32_t binder;
      std::map<std::uint64_t,std::vector<AnchorEvent>> events;
      std::map<std::uint64_t,std::vector<AnchorSegment>> segments;
    };
    struct AnchorBucket {
      bool partitioned=false;
      // The point has no derivative from which to guess a target stride.
      // Different strides/binders for one read/capture schema are Unknown.
      std::map<AnchorSchema,std::optional<AnchorLattice>> lattices;
    };
    std::array<std::map<BucketKey,AnchorBucket>,2> anchor_buckets;
    std::array<bool,2> body_groups_ready{};
    const auto bodyGroups=[&](bool semantic) {
      const unsigned mode=semantic?1U:0U;
      if(body_groups_ready[mode])return;
      if(semantic)exactGroups();
      std::map<const GroupKey*,BucketKey> group_queries;
      for(const auto& [key,group]:groups) {
        charge();auto scope=semantic?lexicalFree(owners[group.fold]):
            std::optional{std::get<1>(key)};
        if(!scope)continue;
        auto signature=body_heads_index.shallow(snapshot[group.fold].children[1]);
        if(!signature)continue;
        std::optional<GroupKey> value=key;
        if(semantic) {
          GroupKey exact{std::get<0>(key),*scope,std::get<2>(key),reducer,std::get<4>(key)};
          auto found=exact_groups.find(exact);
          value=found==exact_groups.end()?std::nullopt:found->second;
        }
        charge(std::get<0>(key).size()+scope->size());
        BucketKey query{std::get<0>(key),std::move(*scope),*signature,reducer};
        charge(32+std::get<0>(query).size()+std::get<1>(query).size());
        // Semantic anchor lookup must not bypass the existing nullable
        // exact lexical index, even though it remains scheduling only.
        if(value)group_queries.emplace(&key,query);
        auto& anchors=anchor_buckets[mode][query];
        const auto& partition=partitions.at(BucketKey{std::get<0>(key),std::get<1>(key),std::get<2>(key),reducer});
        anchors.partitioned|=partition.anchored && (partition.components>1 || partition.unsupported);
        auto [entry,inserted]=body_groups[mode].try_emplace(std::move(query),value);
        if(!inserted && entry->second!=value) {
          // Print only the FIRST distinct-key collision in a bucket. Once
          // nullable, that bucket remains Unknown without more diagnostics.
          if(sentryward::debug::environment("SENTRYWARD_EGRAPH_TRACE") && entry->second && value) {
            SENTRYWARD_DEBUG_PRINT(egraph_5);
            for(auto level:std::get<1>(entry->first))SENTRYWARD_DEBUG_PRINT(egraph_6);
            SENTRYWARD_DEBUG_PRINT(egraph_7);
          }
          entry->second.reset();
        }
      }
      // Sweep actual stored target intervals ONCE, not a group scan per
      // scalar point. Retained targets yield at most twice as many events;
      // the separate aggregate interval-entry cap still applies to segments.
      // Overlap between distinct components is Unknown; gaps
      // stay gaps. Neither a hull nor an anchor is an equality certificate.
      for(auto owner:membership_owners) {
        charge();const auto& member=fold_memberships.at(owner);
        if(!member || !member->anchor)continue;
        auto query=group_queries.find(member->key);if(query==group_queries.end())continue;
        auto& bucket=anchor_buckets[mode].at(query->second);if(!bucket.partitioned)continue;
        const auto& interval=*member->anchor;const auto& family=*interval.family;
        // Only full origin certificates join this component. Its ACTUAL
        // canonical binder is shared by its scheduling records, even when
        // an equivalent original target uses another absolute bound label.
        const auto canonical_binder=snapshot[groups.at(*member->key).fold].fold->binder_level;
        AnchorSchema schema{std::get<0>(family),std::get<3>(family)};
        charge(4*(schema.second.size()+8)*(1+std::bit_width(bucket.lattices.size()+1)));
        auto [entry,inserted]=bucket.lattices.try_emplace(std::move(schema),
            AnchorLattice{std::get<1>(family),canonical_binder,{}, {}});
        if(!inserted && entry->second && (entry->second->stride!=std::get<1>(family) ||
            entry->second->binder!=canonical_binder))entry->second.reset();
        if(!entry->second)continue;
        auto& events=entry->second->events[std::get<4>(family)];
        charge(8);events.push_back({interval.begin,member->key,true});
        events.push_back({interval.end,member->key,false});
      }
      for(auto& [query,bucket]:anchor_buckets[mode]) {
        (void)query;if(!bucket.partitioned)continue;
        for(auto& [schema,lattice]:bucket.lattices) {
          (void)schema;if(!lattice)continue;
          for(auto& [residue,events]:lattice->events) {
            std::sort(events.begin(),events.end(),[&](const auto& a,const auto& b) {
              charge();return a.coordinate<b.coordinate;
            });
            std::map<const GroupKey*,std::size_t> active;
            auto& segments=lattice->segments[residue];
            for(std::size_t i=0;i<events.size();) {
              const auto coordinate=events[i].coordinate;
              do {
                charge(4*(1+std::bit_width(active.size()+1)));
                const auto& event=events[i++];
                if(event.begin)++active[event.key];
                else if(auto found=active.find(event.key);found!=active.end()) {
                  if(!--found->second)active.erase(found);
                } else throw std::logic_error("anchor interval imbalance");
              } while(i<events.size() && events[i].coordinate==coordinate);
              if(i==events.size() || active.size()!=1)continue;
              const auto end=events[i].coordinate;const auto key=active.begin()->first;
              charge(4);
              if(!segments.empty() && segments.back().end==coordinate && segments.back().key==key)
                segments.back().end=end;
              else {
                if(interval_entries>=max_interval_entries)localCap("coverage_intervals",interval_entries);
                ++interval_entries;segments.push_back({coordinate,end,key});
              }
            }
            if(!active.empty())throw std::logic_error("anchor interval imbalance");
          }
          // No borrowed vector/span survives this release. The owned final
          // segments are immutable for the remainder of the reducer epoch.
          lattice->events.clear();
        }
      }
      body_groups_ready[mode]=true;
    };
    const auto candidateGroup=[&](Class body,const std::string& type,
        const std::vector<std::uint32_t>& scope,bool semantic=false,
        bool* ambiguous=nullptr)->std::optional<GroupKey> {
      charge();body=owners[body];
      // Existing bounded root/header discriminator avoids indexing every
      // unrelated finite residual atom. No recursive private shape is read.
      if(!pointBodyHead(body))return {};
      bodyGroups(semantic);
      auto signature=body_heads_index.shallow(body);if(!signature)return {};
      charge(type.size()+scope.size());
      const auto& index=body_groups[semantic?1:0];
      const BucketKey query{type,scope,*signature,reducer};
      auto bucket=anchor_buckets[semantic?1:0].find(query);
      if(bucket!=anchor_buckets[semantic?1:0].end() && bucket->second.partitioned) {
        // No legacy/cross-component fallback in a partitioned bucket. A
        // denied possible sample must not be demoted to an opaque residual.
        if(ambiguous)*ambiguous=true;
        auto read=uniqueReadAnchor(body);if(!read)return {};
        Captures captures;captures.reserve(read->address.coefficients.size());
        for(auto entry:read->address.coefficients){charge();captures.push_back(entry);}
        charge(4*(captures.size()+8)*(1+std::bit_width(bucket->second.lattices.size()+1)));
        auto lattice=bucket->second.lattices.find(AnchorSchema{read->tag,std::move(captures)});
        if(lattice==bucket->second.lattices.end() || !lattice->second)return {};
        const auto& value=*lattice->second;
        auto residue=value.segments.find(read->address.constant%value.stride);
        if(residue==value.segments.end())return {};
        const auto point=read->address.constant/value.stride;
        charge(4*(1+std::bit_width(residue->second.size()+1)));
        auto segment=std::upper_bound(residue->second.begin(),residue->second.end(),point,
            [](std::uint64_t p,const auto& s){return p<s.begin;});
        if(segment==residue->second.begin())return {};--segment;
        if(point>=segment->end)return {};
        if(ambiguous)*ambiguous=false;
        return *segment->key;
      }
      auto found=index.find(query);
      if(found!=index.end() && !found->second && ambiguous)*ambiguous=true;
      return found==index.end()?std::nullopt:found->second;
    };
    // Constant-only classification is a separate, memoized selected-DAG
    // property. Do it before asking for linear forms: proving that neither
    // factor is a scalar must not materialize two large variance/mean forms.
    std::unordered_map<Class,bool> constant_heads;
    const auto trivialConstantHead=[&](Class cls)->std::optional<bool> {
      cls=owners[cls];
      // Literal alternatives take precedence over conservative dependency
      // contamination from equivalent nonliteral witnesses. Otherwise a
      // literal-only arithmetic witness must be closed and have a selected
      // arithmetic head. These necessary negatives never create facts.
      if(literals.contains(cls))return true;
      if(!selected.contains(cls) || !free_indices_[cls].empty())return false;
      return {};
    };
    const auto immediateConstantHead=[&](Class cls)->std::optional<bool> {
      if(auto known=trivialConstantHead(cls))return known;
      if(auto cached=constant_heads.find(cls);cached!=constant_heads.end()) {
        if(trace && !cached->second)++constant_cached_false_hits;
        return cached->second;
      }
      // At most two immediate operands. An opaque/free-index operand makes
      // this literal-only classification false without storing another
      // negative entry for every ordinary variable product or sum.
      bool all=true;
      for(auto child:snapshot[selected.at(cls)].children) {
        charge();const auto known=trivialConstantHead(owners[child]);
        // Diagnostic only: count an existing negative child memo that this
        // current classifier does not consult. Do not change its result,
        // recurse, materialize a form, or create another memo entry.
        if(trace && !known) {
          auto cached=constant_heads.find(owners[child]);
          if(cached!=constant_heads.end() && !cached->second)++constant_child_cached_false;
        }
        if(known && !*known)return false;
        if(!known)all=false;
      }
      return all?std::optional{true}:std::nullopt;
    };
    const auto constantHead=[&](Class root) {
      if(trace)++constant_head_calls;
      struct Task {Class cls;bool ready;};
      std::vector<Task> pending{{owners[root],false}};std::unordered_set<Class> active_constants;
      while(!pending.empty()) {
        if(trace)++constant_head_tasks;
        charge();auto t=pending.back();pending.pop_back();t.cls=owners[t.cls];
        if(immediateConstantHead(t.cls))continue;
        if(constant_entries+active_constants.size()>=max_retained_summary_nodes)
          localCap("constant_nodes",constant_entries+active_constants.size());
        if(active_constants.size()>=max_active_nodes)localCap("constant_active",active_constants.size());
        auto head=selected.find(t.cls);
        const auto& k=snapshot[head->second];
        if(!t.ready) {
          if(!active_constants.insert(t.cls).second)localCap("constant_cycle",active_constants.size());
          pending.push_back({t.cls,true});
          for(auto child:k.children)if(!immediateConstantHead(owners[child]))pending.push_back({owners[child],false});
        } else {
          const bool all=std::all_of(k.children.begin(),k.children.end(),
              [&](auto child){return immediateConstantHead(owners[child]).value();});
          constant_heads.emplace(t.cls,all);++constant_entries;active_constants.erase(t.cls);
        }
      }
      return immediateConstantHead(owners[root]).value();
    };
    const auto linearHead=[&](const Key& k) {
      if(k.name=="Multiply")return constantHead(k.children[0]) || constantHead(k.children[1]);
      if(k.name=="Divide")return constantHead(k.children[1]);
      if(k.name=="Reciprocal")return constantHead(k.children[0]);
      return true;
    };
    // Candidate indexing is optional scheduling, not a constant/equality
    // fact. Unknown immediate coefficients leave their descendants eligible
    // as separate components; only an actual summary asks the bounded deep
    // classifier. Do not fill its memo by walking every unrelated head.
    const auto indexedLinearHead=[&](const Key& k) {
      const auto knownConstant=[&](Class child) {
        charge();return immediateConstantHead(owners[child]).value_or(false);
      };
      if(k.name=="Multiply")return knownConstant(k.children[0]) || knownConstant(k.children[1]);
      if(k.name=="Divide")return knownConstant(k.children[1]);
      if(k.name=="Reciprocal")return knownConstant(k.children[0]);
      return true;
    };
    // One public compact catalog, not a private cache of every expanded
    // prefix form. The shared session aggregates one exact scalar adjoint
    // per reached stored node; only the caller-owned final form is expanded.
    // Virtual IDs are bounded scalar headers, never logical Fold elements.
    auto session_limits=limits;
    session_limits.max_nodes=max_retained_summary_nodes-summary_entries;
    session_limits.max_cached_terms=max_cached_terms-cached_terms;
    l::DagSession linear_session(builder,[&](l::NodeId id)->std::optional<l::Node> {
      charge();l::Node node;node.admitted=true;
      if(id>=2*n) {
        const auto cls=id-2*n;
        if(cls>=n || !selected.contains(cls) || snapshot[selected.at(cls)].name!="Reciprocal")return {};
        node.kind=l::NodeKind::Literal;node.literal="1";node.type=snapshot[cls].type;return node;
      }
      if(id>=n) {
        const auto cls=id-n;
        if(cls>=n || !seeded_folds.contains(cls) || !snapshot[seeded_folds.at(cls)].fold->count)return {};
        node.kind=l::NodeKind::Atom;node.atom=static_cast<l::Atom>(id);node.type=snapshot[cls].type;return node;
      }
      const auto cls=static_cast<Class>(id);
      if(owners[cls]!=cls || !admitted[cls])return {};
      node.type=snapshot[cls].type;
      if(literals.contains(cls)) {node.kind=l::NodeKind::Literal;node.literal=literals.at(cls);return node;}
      if(!ordinary_folds.contains(cls) && seeded_folds.contains(cls)) {
        const auto& k=snapshot[seeded_folds.at(cls)];
        // The seed stays OUTSIDE the binder, once. The private region atom
        // retains the existing reductionFold()/mappedDomain semantic key.
        node.kind=k.fold->count?l::NodeKind::Add:l::NodeKind::Identity;
        node.children={owners[k.children[0]],n+cls};node.child_count=k.fold->count?2:1;return node;
      }
      if(auto head=selected.find(cls);head!=selected.end() && linearHead(snapshot[head->second])) {
        const auto& k=snapshot[head->second];node.child_count=k.children.size();
        if(k.name=="Reciprocal") {node.kind=l::NodeKind::Divide;node.children={2*n+cls,owners[k.children[0]]};node.child_count=2;return node;}
        if(node.child_count>2)return {};
        for(std::size_t i=0;i<node.child_count;++i)node.children[i]=owners[k.children[i]];
        node.kind=l::NodeKind::Add;
        if(k.name=="Subtract")node.kind=l::NodeKind::Subtract;
        if(k.name=="Negate")node.kind=l::NodeKind::Negate;
        if(k.name=="Multiply")node.kind=l::NodeKind::Multiply;
        if(k.name=="Divide")node.kind=l::NodeKind::Divide;
        return node;
      }
      // Unselected/nonlinear admitted heads are the SAME finite atom. No
      // nonconstant denominator or operand is cancelled or opened here.
      node.kind=l::NodeKind::Atom;node.atom=cls;return node;
    },session_limits);
    std::size_t session_nodes=0,session_terms=0;
    const auto summarize=[&](Class root)->l::Result {
      auto result=linear_session.summarize(owners[root]);
      const auto stats=linear_session.statistics();linear_statistics=stats;
      if(stats.nodes-session_nodes>max_retained_summary_nodes-summary_entries)localCap("summary_nodes",summary_entries);
      if(stats.cached_terms-session_terms>max_cached_terms-cached_terms)localCap("summary_terms",cached_terms);
      summary_entries+=stats.nodes-session_nodes;cached_terms+=stats.cached_terms-session_terms;
      session_nodes=stats.nodes;session_terms=stats.cached_terms;sync();
      if(!result) {
        switch(linear_session.cap()) {
          case l::DagSession::Cap::Nodes:localCap("summary_nodes",summary_entries);break;
          case l::DagSession::Cap::CachedTerms:localCap("summary_terms",cached_terms);break;
          case l::DagSession::Cap::Atoms:localCap("summary_atoms",limits.max_atoms);break;
          case l::DagSession::Cap::Work:localCap("linear_builder",builder.work());break;
          case l::DagSession::Cap::Coefficients:localCap("summary_coefficients",session_nodes);break;
          case l::DagSession::Cap::None:localCap("summary_unknown",session_nodes);break;
        }
      }
      return result;
    };
    struct CoverageKey {
      GroupKey group;
      l::Coefficient constant;
      std::vector<l::WeightedInterval> intervals;
      // Exact finite atoms outside the one certified reduction-body group.
      // A nonliteral seed is retained once, outside the Fold binder, rather
      // than discarded or treated as a uniform value at every sample.
      std::vector<l::Term> residual;
    };
    const auto less=[&](const CoverageKey& a,const CoverageKey& b) {
      charge(8);
      if(a.group!=b.group)return a.group<b.group;
      // Compare the cheap interval geometry before exact coefficient
      // payloads. Distinct short Fold domains should not spend the bounded
      // rational-comparison allowance merely to order scalar endpoints.
      const auto size=std::min(a.intervals.size(),b.intervals.size());
      for(std::size_t i=0;i<size;++i) {
        charge();const auto& x=a.intervals[i];const auto& y=b.intervals[i];
        if(x.begin!=y.begin)return x.begin<y.begin;
        if(x.end!=y.end)return x.end<y.end;
        charge(128);
        if(x.weight!=y.weight)return x.weight<y.weight;
      }
      if(a.intervals.size()!=b.intervals.size())return a.intervals.size()<b.intervals.size();
      charge(128);if(a.constant!=b.constant)return a.constant<b.constant;
      const auto residual_size=std::min(a.residual.size(),b.residual.size());
      for(std::size_t i=0;i<residual_size;++i) {
        charge();const auto& x=a.residual[i];const auto& y=b.residual[i];
        if(x.atom!=y.atom)return x.atom<y.atom;
        charge(128);
        if(x.weight!=y.weight)return x.weight<y.weight;
      }
      return a.residual.size()<b.residual.size();
    };
    std::map<CoverageKey,Class,decltype(less)> signatures(less);
    setup_end=remaining;
    std::vector<Class> candidates;std::unordered_set<Class> candidate_set;
    std::unordered_set<Class> linear_parented;
    const auto append=[&](Class id) {
      const auto owner=owners[id];
      if(candidate_entries<max_candidates && admitted[owner] && candidate_set.insert(owner).second) {
        candidates.push_back(owner);++candidate_entries;
      }
    };
    const auto appendRoot=[&](Class id,bool room) {
      charge();const auto owner=owners[id];
      // Whole outputs already have their own bounded priority pool. They
      // must not refill the prerequisite pool as "maximal components" when
      // many outputs depend on the SAME interior Fold/linear certificate.
      charge();linear_parented.insert(owner);
      if(!room || candidate_entries>=max_root_candidates)return;
      // Complete linear roots and reduction regions are normal candidates.
      // A nonlinear root is one opaque sample only when its bounded local
      // head matches an indexed Fold BODY head; do not scan its descendants.
      if(ordinary_folds.contains(owner) || seeded_folds.contains(owner) ||
          (selected.contains(owner) && indexedLinearHead(snapshot[selected.at(owner)]))) {
        append(id);return;
      }
      if(candidate_entries>=max_candidates || !admitted[owner] || candidate_set.contains(owner))return;
      // This lookup schedules only. Coverage still checks the complete
      // selected witness, lexical context, affine address and weighted domain.
      if(pointBodyHead(owner))append(id);
    };
    // Index each complete output pair before unrelated later roots. A large
    // unsupported root must not exhaust a local certificate ledger between
    // two small mates which would already have a complete equality key.
    for(std::size_t i=0;i<std::max(left.size(),right.size());++i) {
      // Reserve both possible mate slots before admitting this root pair.
      // The remaining half stays available for certificate prerequisites.
      // A frozen e-class equality is already a complete mate certificate,
      // not a head/shape hint. Keep both parent markers via appendRoot, but
      // do not spend optional root slots or promote their interior boundaries.
      const bool certified=i<left.size() && i<right.size() && owners[left[i]]==owners[right[i]];
      const bool room=!certified && candidate_entries+2<=max_root_candidates;
      if(i<left.size())appendRoot(left[i],room);
      if(i<right.size())appendRoot(right[i],room);
    }
    const auto root_candidates=candidates.size();
    phase="maximal_linear";
    // Fold-containing maximal components precede optional moments and the
    // many individual Fold candidates. Scalar-only components are deferred:
    // eagerly summarizing every partial quadratic-merge mean/correction can
    // consume the shared work allowance before a complete moment proof.
    // Both phases share the original 1024 candidate/work/summary limits and
    // the one half-pool component cap; no growing prefix can fill that pool.
    for(const auto& [cls,head]:selected) {
      charge();if(literals.contains(cls) || !indexedLinearHead(snapshot[head]))continue;
      for(auto child:snapshot[head].children){charge();linear_parented.insert(owners[child]);}
    }
    // A maximal heterogeneous Add/Subtract need not itself have one coverage
    // group. Keep its immediate linear region as a prerequisite instead of
    // hiding that region behind an unrelated opaque scalar/reduction arm.
    // Inspect only maximal heads (or already admitted whole roots), never
    // every Add prefix or an ancestor path. The bounded scope/header tests
    // schedule a candidate; its complete weighted-body certificate is final.
    std::unordered_set<Class> linear_boundaries;
    // Keep only one bounded stored-head schedule, not another whole-DAG
    // traversal. Earliest IDs are deterministic even though selected is an
    // unordered index; omitted boundaries remain ordinary later candidates.
    std::set<Class> boundary_heads;
    for(const auto& [cls,head]:selected) {
      charge();const auto& k=snapshot[head];
      if((linear_parented.contains(cls) && !candidate_set.contains(cls)) ||
          (k.name!="Add" && k.name!="Subtract") || k.children.size()!=2)continue;
      const std::array<Class,2> children{owners[k.children[0]],owners[k.children[1]]};
      const auto linearChild=[&](Class child) {
        charge();return admitted[child] && !literals.contains(child) &&
            (ordinary_folds.contains(child) || seeded_folds.contains(child) ||
             (selected.contains(child) && indexedLinearHead(snapshot[selected.at(child)]))) &&
            !immediateConstantHead(child).value_or(false);
      };
      const bool a=linearChild(children[0]),b=linearChild(children[1]);
      if(a==b)continue;
      const auto region=children[a?0:1],other=children[a?1:0];
      if(literals.contains(other) || immediateConstantHead(other).value_or(false))continue;
      auto scope=scopeFor(region);if(!scope)continue;
      charge();if(!group_scopes.contains({snapshot[region].type,*scope}) ||
          !linear_boundaries.insert(region).second)continue;
      const auto region_head=ordinary_folds.contains(region)?ordinary_folds.at(region):
          seeded_folds.contains(region)?seeded_folds.at(region):selected.at(region);
      // set operations and their bounded metadata copies are charged. This
      // shares the existing candidate cap, not a second admission allowance.
      charge(32);
      if(boundary_heads.size()<max_candidates || region_head<*boundary_heads.rbegin()) {
        boundary_heads.insert(region_head);
        if(boundary_heads.size()>max_candidates)boundary_heads.erase(std::prev(boundary_heads.end()));
      }
    }
    constexpr std::size_t max_component_candidates=max_candidates/2;
    std::size_t component_candidates=0;
    // Reserve component room within the SAME fixed pool. Only an opaque
    // boundary region precedes complete components; parented/intermediate
    // Fold aliases are optional later candidates, not prerequisites merely
    // because a rewrite produced another stored Fold head.
    const auto early_fold_cap=(max_candidates-candidate_entries)/2;
    std::size_t early_fold_candidates=0;
    const auto appendBoundaries=[&](bool folded) {
      for(auto id:boundary_heads) {
        charge();const auto owner=owners[id];
        if(candidate_entries>=max_candidates || component_candidates>=max_component_candidates)break;
        if(has_fold_[owner]!=folded)continue;
        const bool fold_region=(ordinary_folds.contains(owner) && ordinary_folds.at(owner)==id) ||
            (seeded_folds.contains(owner) && seeded_folds.at(owner)==id);
        if(fold_region) {
          if(!folded || early_fold_candidates>=early_fold_cap)continue;
          const auto before=candidates.size();append(id);
          if(candidates.size()!=before)++early_fold_candidates;
        } else if(selected.contains(owner) && selected.at(owner)==id &&
            indexedLinearHead(snapshot[id]) && !immediateConstantHead(owner).value_or(false)) {
          const auto before=candidates.size();append(id);
          if(candidates.size()!=before)++component_candidates;
        }
      }
    };
    const auto appendComponents=[&](bool folded) {
      for(auto id:catalog.schedule_ids) {
        charge();const auto owner=owners[id];
        if(candidate_entries>=max_candidates || component_candidates>=max_component_candidates)break;
        // A selected linear parent can summarize this region as one atom.
        // Its opaque/nonlinear boundary, however, may need a whole-region
        // equality before a complete output can match by congruence.
        if(folded && ((ordinary_folds.contains(owner) && ordinary_folds.at(owner)==id) ||
            (seeded_folds.contains(owner) && seeded_folds.at(owner)==id)) &&
            (!linear_parented.contains(owner) || linear_boundaries.contains(owner)) &&
            early_fold_candidates<early_fold_cap) {
          const auto before=candidates.size();append(id);
          if(candidates.size()!=before)++early_fold_candidates;
          continue;
        }
        if(selected.contains(owner) && selected.at(owner)==id &&
            has_fold_[owner]==folded &&
            (!linear_parented.contains(owner) || linear_boundaries.contains(owner)) &&
            indexedLinearHead(snapshot[id]) && !immediateConstantHead(owner).value_or(false)) {
          const auto before=candidates.size();append(id);
          if(candidates.size()!=before)++component_candidates;
        }
      }
    };
    appendBoundaries(true);
    appendComponents(true);
    if(extrema){appendBoundaries(false);appendComponents(false);}
    const auto priority_candidates=candidates.size();
    // Collect the SAME complete candidate window before publishing any
    // coordinate key. Only proof execution is deferred past moments.
    phase="scalar_linear_components";
    if(!extrema){appendBoundaries(false);appendComponents(false);}
    phase="fold_alias_fallback";
    // Retain the original region alternatives after complete components
    // (and Sum's optional moments/scalar components). Heterogeneous parents
    // can still need an interior Fold equality; it is never dropped as a
    // semantic assertion. All categories share the original total pool.
    for(auto id:catalog.fold_ids) {
      if(candidate_entries>=max_candidates)break;
      charge();const auto owner=owners[id];
      if((ordinary_folds.contains(owner) && ordinary_folds.at(owner)==id) ||
          (seeded_folds.contains(owner) && seeded_folds.at(owner)==id))append(id);
    }
    phase="linear_prefix_fallback";
    // Heterogeneous parents need not have a coverage key even when an
    // interior component does. Retain the old bounded prefix fallback LAST;
    // neither this scan nor a cap adds an equality.
    if(!extrema)for(auto id:catalog.schedule_ids) {
      charge();if(selected.contains(owners[id]) && selected.at(owners[id])==id &&
          !immediateConstantHead(owners[id]).value_or(false))append(id);
    }

    scheduling_end=remaining;
    if(trace) {
      // Scalar census of the already sealed, capped candidate window. Do
      // not select another witness, render expressions or traverse bodies.
      SENTRYWARD_DEBUG_PRINT(egraph_8);
      if(!left.empty())SENTRYWARD_DEBUG_PRINT(egraph_9);else SENTRYWARD_DEBUG_PRINT(egraph_10);
      SENTRYWARD_DEBUG_PRINT(egraph_11);if(!right.empty())SENTRYWARD_DEBUG_PRINT(egraph_12);else SENTRYWARD_DEBUG_PRINT(egraph_13);
      SENTRYWARD_DEBUG_PRINT(egraph_14);
      for(auto cls:candidates) {
        cls=owners[cls];Class head=cls;
        if(auto f=ordinary_folds.find(cls);f!=ordinary_folds.end())head=f->second;
        else if(auto f=seeded_folds.find(cls);f!=seeded_folds.end())head=f->second;
        else if(auto h=selected.find(cls);h!=selected.end())head=h->second;
        SENTRYWARD_DEBUG_PRINT(egraph_15);
        if(head<n && snapshot[head].fold)SENTRYWARD_DEBUG_PRINT(egraph_16);
        SENTRYWARD_DEBUG_PRINT(egraph_17);
      }
      SENTRYWARD_DEBUG_PRINT(egraph_18);
    }
    phase="target_frontier";
    // ONE union-of-candidate selected-DAG frontier. An ordinary Fold retains
    // its ACTUAL template even if its class also has a selected linear alias;
    // this does not choose the summary's atom or assert a body equality.
    // A seeded Fold retains its original seed once and its opaque body-only
    // boundary. No Fold BODY/logical instance is traversed.
    // Nonlinear atoms stop here. Their independently queued Fold/linear
    // children remain roots in the unchanged boundary/component schedule.
    std::unordered_set<Class> target_frontier_seen,target_frontier_owners;
    std::vector<Class> target_frontier_pending;
    charge(candidates.size());target_frontier_pending=candidates;
    const auto frontierPush=[&](Class cls) {
      charge();cls=owners[cls];
      if(!target_frontier_seen.contains(cls)) {
        if(target_frontier_pending.size()>=max_retained_summary_nodes)
          localCap("target_frontier_active",target_frontier_pending.size());
        charge(3);target_frontier_pending.push_back(cls);
      }
    };
    while(!target_frontier_pending.empty()) {
      charge();const auto cls=owners[target_frontier_pending.back()];target_frontier_pending.pop_back();
      if(target_frontier_seen.contains(cls))continue;
      if(target_frontier_seen.size()>=max_retained_summary_nodes)
        localCap("target_frontier_nodes",target_frontier_seen.size());
      charge(3);target_frontier_seen.insert(cls);++frontier_nodes;
      if(!admitted[cls] || literals.contains(cls))continue;
      if(auto f=ordinary_folds.find(cls);f!=ordinary_folds.end()) {
        charge(3);target_frontier_owners.insert(cls);
      }
      if(auto f=seeded_folds.find(cls);f!=seeded_folds.end() && !ordinary_folds.contains(cls)) {
        const auto& k=snapshot[f->second];
        // Extrema retain an actual non-neutral seed in the complete form.
        // Sum's seed/body split is exactly the existing DagSession resolver.
        if(extrema){charge(3);target_frontier_owners.insert(cls);}
        frontierPush(k.children[0]);
        // A certified Add(seed, neutral-Fold) alias can provide the ACTUAL
        // body-only target template. The summary may prefer its private
        // seeded atom, but target discovery must retain that stored alias.
        // Follow only ranked selected linear children, never the Fold BODY.
        if(auto alias=selected.find(cls);alias!=selected.end() && linearHead(snapshot[alias->second]))
          for(auto child:snapshot[alias->second].children){++frontier_edges;frontierPush(child);}
        continue;
      }
      if(auto f=selected.find(cls);f!=selected.end() && linearHead(snapshot[f->second])) {
        for(auto child:snapshot[f->second].children){++frontier_edges;frontierPush(child);}
      }
    }
    // IDs were already admitted in deterministic ascending snapshot order.
    // This scalar filter never reopens a denied target or whole Fold body.
    std::vector<Class> frontier_fold_ids;
    for(auto id:potential_target_ids) {
      charge();if(target_frontier_owners.contains(owners[id])) {
        charge(3);frontier_fold_ids.push_back(id);
      }
    }
    frontier_targets=frontier_fold_ids.size();
    phase="group_shapes";
    // Metadata-sized target scan; the selected Fold body remains unchanged.
    for(auto id:frontier_fold_ids) {
      charge();const auto owner=owners[id];
      const bool ordinary=ordinary_folds.contains(owner) && ordinary_folds.at(owner)==id;
      const bool seeded=extrema && seeded_folds.contains(owner) && seeded_folds.at(owner)==id;
      if((!ordinary && !seeded) || !snapshot[id].fold->count)continue;
      const auto body=owners[snapshot[id].children[1]];
      // A coordinate-coverage target needs an observable reduction axis.
      // Fixed-point absence (or a certified literal) proves this body is
      // invariant; it cannot supply an affine interval origin. This is only
      // target admission: the Fold and its normal algebra remain unchanged.
      if(!extrema && (literals.contains(body) ||
          !dependsOn(body,snapshot[id].fold->binder_level)))continue;
      auto scope=scopeFor(owner);if(!scope)continue;
      auto body_shape=shape(snapshot[id].children[1],0);if(!body_shape)continue;
      const auto body_head=selected.contains(body)?selected.at(body):
          members[body].empty()?body:members[body].front();
      body_heads.emplace(snapshot[body_head].kind,snapshot[body_head].name);
      const BucketKey bucket{snapshot[id].type,*scope,*body_shape,reducer};
      charge(snapshot[id].type.size()+scope->size()+32);
      if(group_entries>=max_groups && !partitions.contains(bucket))continue;
      auto& partition=partitions[bucket];
      auto anchor=targetAnchor(snapshot[id]);
      if(!anchor)partition.unsupported=true;
      std::optional<AnchorInterval> interval;
      const GroupKey* component=nullptr;
      std::map<std::uint64_t,std::optional<Prototype>>* prototypes=nullptr;
      if(anchor) {
        charge(4*(std::get<3>(anchor->first).size()+8)*(1+std::bit_width(partition.prototypes.size()+1)));
        if(group_entries>=max_groups && !partition.prototypes.contains(anchor->first))continue;
        auto [family,new_family]=partition.prototypes.try_emplace(std::move(anchor->first));
        (void)new_family;interval=anchor->second;interval->family=&family->first;
        prototypes=&family->second;partition.anchored=true;
        // ONE deterministic nearest compatible prototype and ONE oriented
        // full proof. Failed/ambiguous/exhausted proofs create no equality;
        // no second prototype or opposite orientation is tried.
        charge(4*(1+std::bit_width(prototypes->size()+1)));
        auto nearest=prototypes->lower_bound(interval->begin);
        if(nearest!=prototypes->begin()) {
          auto lower=std::prev(nearest);
          if(nearest==prototypes->end() || interval->begin-lower->first<=nearest->first-interval->begin)nearest=lower;
        }
        if(nearest!=prototypes->end() && nearest->second && !body_heads_index.exhausted) {
          const auto& prototype=*nearest->second;
          const auto& p=snapshot[prototype.fold];const auto& c=snapshot[id];
          const bool candidate_lower=interval->begin<nearest->first;
          const auto expected=candidate_lower?nearest->first-interval->begin:interval->begin-nearest->first;
          bool related=false;
          if(!expected && sameBoundFoldBody(p,c)) {related=true;if(trace)++origin_identity;}
          else {
            auto map=tracedMap(origin_trace,[&] {
              return affineBodyMapWithHeads(candidate_lower?c.children[1]:p.children[1],
                  candidate_lower?p.children[1]:c.children[1],
                  candidate_lower?c.fold->binder_level:p.fold->binder_level,
                  snapshot,false,&remaining,body_heads_index,
                  candidate_lower?p.fold->binder_level:c.fold->binder_level);
            });
            related=map && map->first==1 && map->second==expected;
          }
          if(!remaining)localCap("optional_work",initial_work);
          if(related)component=prototype.key;
        }
      }
      GroupKey key=component?*component:GroupKey{snapshot[id].type,*scope,*body_shape,reducer,
          (anchor || partition.anchored)?id+1:0};
      if(group_entries>=max_groups && !groups.contains(key))continue;
      auto [entry,inserted]=groups.try_emplace(std::move(key),Group{id,{},interval});
      if(inserted)++group_entries;
      if(inserted)++partition.components;
      if(!inserted && interval) {
        if(entry->second.anchor && interval->family==entry->second.anchor->family &&
            interval->begin<entry->second.anchor->begin) {
          // The one complete origin proof above certified this component.
          // Changing its canonical origin is scheduling only; EVERY atom
          // will still be fully mapped to the final canonical body below.
          entry->second.fold=id;entry->second.anchor=interval;
        }
      } else if(!inserted && !partition.anchored) {
        // A WHOLLY unsupported anchor bucket retains the original path.
        // Its token0 never joins an anchored component by a wildcard hint.
        const auto& current=snapshot[entry->second.fold];const auto& candidate=snapshot[id];
        if(candidate.fold->binder_level==current.fold->binder_level) {
          if(sameBoundFoldBody(candidate,current)) {
            if(trace)++origin_identity;
          } else {
          // Choose a lower coordinate origin only after proving the ENTIRE
          // current body is candidate(i+b), with unit stride and b>0.
          // Node/visit order, allocation names and a single address leaf do
          // not establish this direction. One check per stored Fold body;
          // shared shape/affine caches and the aggregate work cap bound it.
          auto lower=tracedMap(origin_trace,[&] {
            return affineBodyMap(candidate.children[1],current.children[1],
                candidate.fold->binder_level,snapshot,members,false,&remaining,&affine_cache,&body_heads_index);
          });
          if(!remaining)localCap("optional_work",initial_work);
          if(lower && lower->first==1 && lower->second>0)entry->second.fold=id;
          }
        }
      }
      if(prototypes && (fold_memberships.contains(owner) || fold_memberships.size()<max_retained_summary_nodes)) {
        charge(8*(1+std::bit_width(prototypes->size()+1)));
        if(prototypes->contains(interval->begin) || prototype_entries<max_retained_summary_nodes) {
          auto [record,new_record]=prototypes->try_emplace(interval->begin,Prototype{id,&entry->first});
          if(new_record)++prototype_entries;
          else if(record->second && record->second->key!=&entry->first)record->second.reset();
        }
      }
      // O(1) owner/header records use the SAME physical retained capacity,
      // not another copied group key or ancestor walk.
      if(fold_memberships.contains(owner) || fold_memberships.size()<max_retained_summary_nodes) {
        charge(32);
        auto [member,new_member]=fold_memberships.try_emplace(owner,FoldMembership{id,&entry->first,interval});
        if(new_member){charge(3);membership_owners.push_back(owner);}
        if(!new_member && member->second &&
            (member->second->head!=id || member->second->key!=&entry->first))member->second.reset();
      }
    }
    groups_end=remaining;
    if(sentryward::debug::environment("SENTRYWARD_EGRAPH_TRACE")) {
#if defined(SENTRYWARD_DEVELOPMENT_DIAGNOSTICS) || defined(SENTRYWARD_ENABLE_COMPRESSION_AUDIT)
#include "sentryward/development/blocks/egraph_trace_1.inc"
#endif
}

    if(groups.empty())return;
    std::array<std::unordered_map<Class,std::optional<CoverageKey>>,2> coverage_caches;
    const auto reductionFold=[&](l::Atom atom)->std::optional<Class> {
      if(atom>=n) {
        const auto cls=static_cast<Class>(atom-n);
        if(auto found=seeded_folds.find(cls);found!=seeded_folds.end())return found->second;
        return {};
      }
      if(auto found=ordinary_folds.find(static_cast<Class>(atom));found!=ordinary_folds.end())return found->second;
      return {};
    };
    // The same full-body/scoped mapping factory serves every reducer. Shape
    // identity merely schedules this exact certificate; it is not a proof.
    const auto tracedDomainMap=[&](auto&& invoke) {
      BodyHeadIndex::FallbackTrace reports(body_heads_index,trace?&domain_fallback_reports:nullptr,
          trace?&domain_denial_reports:nullptr);
      return tracedMap(domain_trace,std::forward<decltype(invoke)>(invoke));
    };
    const auto mappedDomain=[&](Group& group,l::Atom atom,
        const std::vector<std::uint32_t>& scope,bool semantic=false) {
      if(trace)++domain_queries;
      auto& cache=group.mapped[semantic?1:0];
      auto mapped=cache.find(atom);
      if(mapped==cache.end()) {
        const auto& target=snapshot[group.fold];const auto binder=target.fold->binder_level;
        std::optional<std::pair<std::uint64_t,std::uint64_t>> domain;
        if(auto fold=reductionFold(atom)) {
          const auto& k=snapshot[*fold];auto atom_scope=semantic?lexicalFree(owners[*fold]):scopeFor(owners[*fold]);
          if(k.type==target.type && k.fold->reducer==target.fold->reducer && atom_scope && *atom_scope==scope) {
            if(k.fold->binder_level==binder) {
              if(sameBoundFoldBody(target,k)) {
                if(trace)++domain_identity;
                // Count is the ACTUAL source domain, never the target count.
                // Seeded atoms represent body-only sums; their exact seed
                // remains once in the existing residual/constant form.
                domain=std::pair{std::uint64_t{0},k.fold->count};
              } else {
                auto map=tracedDomainMap([&] {
                  return affineBodyMap(target.children[1],k.children[1],binder,snapshot,members,false,&remaining,&affine_cache,&body_heads_index);
                });
                if(map && map->first==1 && k.fold->count<=UINT64_MAX-map->second)domain=std::pair{map->second,map->second+k.fold->count};
              }
            } else {
              std::size_t rectangle_work=0;
              const std::vector<RectangleAxis> axes{{k.fold->binder_level,k.fold->count}};
              auto map=tracedDomainMap([&] {
                auto result=rectangleBodyMap(target.children[1],k.children[1],binder,
                    axes,snapshot,members,affine_cache,rectangle_work);charge(rectangle_work);return result;
              });
              if(map && map->coefficients.size()==1 && map->coefficients[0]==1 &&
                  k.fold->count<=UINT64_MAX-map->begin)domain=std::pair{map->begin,map->begin+k.fold->count};
            }
          }
        } else if(atom<n) {
          auto atom_scope=semantic?lexicalFree(static_cast<Class>(atom)):scopeFor(static_cast<Class>(atom));
          if(atom_scope && !std::binary_search(atom_scope->begin(),atom_scope->end(),binder) &&
              snapshot[atom].type==target.type && *atom_scope==scope) {
            auto map=tracedDomainMap([&] {
              return affineBodyMap(target.children[1],static_cast<Class>(atom),binder,snapshot,members,true,&remaining,&affine_cache,&body_heads_index);
            });
            if(map && map->second!=UINT64_MAX)domain=std::pair{map->second,map->second+1};
          }
        }
        if(!remaining)localCap("optional_work",initial_work);
        if(mapping_entries>=max_mappings)localCap("affine_mappings",mapping_entries);++mapping_entries;
        mapped=cache.emplace(atom,domain).first;
      } else if(trace)++domain_hits;
      return mapped->second;
    };
    struct ExtremaForm {
      std::optional<GroupKey> group;
      std::vector<l::WeightedInterval> intervals;
      std::vector<l::Term> residual;
    };
    std::unordered_map<Class,std::optional<ExtremaForm>> extrema_summaries;
    std::unordered_map<Class,unsigned char> extrema_state;
    std::optional<l::Coefficient> unit,private_zero;
    if(extrema) {
      unit=builder.coefficient("1");private_zero=builder.coefficient("0");sync();
      if(!unit || !private_zero)localCap("private_coefficients");
    }
    const auto summarizeExtrema=[&](Class root)->const ExtremaForm* {
      struct Task {Class cls;bool ready;};std::vector<Task> pending{{owners[root],false}};
      while(!pending.empty()) {
        charge();auto [cls,ready]=pending.back();pending.pop_back();
        if(extrema_summaries.contains(cls))continue;
        if(!extrema_state.contains(cls)) {
          if(summary_entries>=max_retained_summary_nodes)localCap("summary_nodes",summary_entries);
          extrema_state.emplace(cls,0);++summary_entries;
        }
        std::optional<ExtremaForm> result;const auto fold_head=reductionFold(
            seeded_folds.contains(cls)?static_cast<l::Atom>(n+cls):cls);
        const auto head=selected.find(cls);
        // A complete matching body is one sample even if that body's head
        // itself is Maximum/Minimum. Never distribute the body operation.
        auto scope=scopeFor(cls);std::optional<GroupKey> body_group;
        const auto chosen=head!=selected.end()?head->second:members[cls].empty()?cls:members[cls].front();
        // Only bounded local signatures classify a complete body sample;
        // the shared typed/scoped mapping remains the equality certificate.
        const auto& chosen_name=snapshot[chosen].name;
        if(chosen_name.size()>4096)localCap("sample_head_metadata",chosen_name.size());
        charge(chosen_name.size());
        bool ambiguous=false;
        if(scope && fold_head)body_group=admittedFoldGroup(*fold_head,snapshot[cls].type,*scope);
        if(!body_group && scope && (fold_head || body_heads.contains({snapshot[chosen].kind,chosen_name})))
          body_group=candidateGroup(fold_head?snapshot[*fold_head].children[1]:cls,
              snapshot[cls].type,*scope,false,&ambiguous);
        if(ambiguous) {extrema_summaries.emplace(cls,std::nullopt);continue;}
        if(scope && body_group && admitted[cls]) {
          if(auto group=groups.find(*body_group);group!=groups.end()) {
            auto domain=mappedDomain(group->second,fold_head && seeded_folds.contains(cls)?n+cls:cls,*scope);
            if(domain)result=ExtremaForm{*body_group,{{domain->first,domain->second,*unit}},{}};
            else {extrema_summaries.emplace(cls,std::nullopt);continue;}
          }
        }
        const bool seeded=fold_head && seeded_folds.contains(cls);
        const bool combine=!result && head!=selected.end();
        if(seeded || combine) {
          const auto& k=snapshot[seeded?*fold_head:head->second];
          if(!ready) {
            if(extrema_state.at(cls)==1)localCap("summary_cycle",summary_entries);
            extrema_state[cls]=1;pending.push_back({cls,true});
            if(seeded)pending.push_back({owners[k.children[0]],false});
            else for(auto c:k.children)pending.push_back({owners[c],false});
            continue;
          }
          if(seeded && !result) {extrema_summaries.emplace(cls,std::nullopt);continue;}
          const auto merge=[&](const ExtremaForm& a,const ExtremaForm& b)->std::optional<ExtremaForm> {
            if(a.group && b.group && *a.group!=*b.group)return {};
            const auto pieces=a.intervals.size()+b.intervals.size();
            const auto atoms=a.residual.size()+b.residual.size();charge(pieces+atoms);
            if(pieces>limits.max_intervals || atoms>max_cached_terms-cached_terms)
              localCap("summary_terms",cached_terms);
            std::vector<l::WeightedInterval> intervals=a.intervals;
            intervals.insert(intervals.end(),b.intervals.begin(),b.intervals.end());
            auto canonical=builder.canonicalIntervals(intervals);sync();if(!canonical)return {};
            std::vector<l::Term> residual;residual.reserve(atoms);
            std::merge(a.residual.begin(),a.residual.end(),b.residual.begin(),b.residual.end(),
                std::back_inserter(residual),[](const auto& x,const auto& y){return x.atom<y.atom;});
            return ExtremaForm{a.group?a.group:b.group,std::move(*canonical),std::move(residual)};
          };
          auto a=extrema_summaries.find(owners[k.children[0]]);
          if(seeded) {
            if(a!=extrema_summaries.end() && a->second)result=merge(*result,*a->second);else result.reset();
          } else {
            auto b=extrema_summaries.find(owners[k.children[1]]);
            if(a!=extrema_summaries.end() && a->second && b!=extrema_summaries.end() && b->second) {
              // A finite nonnegative real makes Max(0,x)'s zero inactive.
              // This is not a NaN/IEEE rule or an arbitrary extrema weight.
              if(reducer==s::Reducer::Max && zero_[owners[k.children[0]]] && nonnegative_[owners[k.children[1]]])result=*b->second;
              else if(reducer==s::Reducer::Max && zero_[owners[k.children[1]]] && nonnegative_[owners[k.children[0]]])result=*a->second;
              else result=merge(*a->second,*b->second);
            }
          }
        } else if(!result && admitted[cls])result=ExtremaForm{{},{},{{cls,*unit}}};
        if(result) {
          const auto pieces=result->intervals.size(),atoms=result->residual.size();
          if(pieces+atoms>max_cached_terms-cached_terms)localCap("summary_terms",cached_terms);
          if(pieces>max_interval_entries-interval_entries)localCap("coverage_intervals",interval_entries);
          if(atoms>max_cached_terms-residual_entries)localCap("coverage_residuals",residual_entries);
          cached_terms+=pieces+atoms;interval_entries+=pieces;residual_entries+=atoms;
        }
        extrema_summaries.emplace(cls,std::move(result));extrema_state[cls]=2;
      }
      const auto& result=extrema_summaries.at(owners[root]);return result?&*result:nullptr;
    };
    const auto coverage=[&](Class candidate,bool semantic=false)->const CoverageKey* {
      candidate=owners[candidate];charge();
      auto& coverage_cache=coverage_caches[semantic?1:0];
      if(auto cached=coverage_cache.find(candidate);cached!=coverage_cache.end())return cached->second?&*cached->second:nullptr;
      const std::vector<std::uint32_t>* diagnostic_scope=nullptr;
      const auto fail=[&](const char* reason,l::Atom atom=std::numeric_limits<l::Atom>::max(),
          std::size_t terms=0)->const CoverageKey* {
        if(trace && coverage_denial_reports) {
          --coverage_denial_reports;
          SENTRYWARD_DEBUG_PRINT(egraph_24);
          if(!left.empty())SENTRYWARD_DEBUG_PRINT(egraph_25);else SENTRYWARD_DEBUG_PRINT(egraph_26);
          SENTRYWARD_DEBUG_PRINT(egraph_27);if(!right.empty())SENTRYWARD_DEBUG_PRINT(egraph_28);else SENTRYWARD_DEBUG_PRINT(egraph_29);
          SENTRYWARD_DEBUG_PRINT(egraph_30);
          if(atom!=std::numeric_limits<l::Atom>::max())SENTRYWARD_DEBUG_PRINT(egraph_31);else SENTRYWARD_DEBUG_PRINT(egraph_32);
          SENTRYWARD_DEBUG_PRINT(egraph_33);
          if(diagnostic_scope) {
            for(std::size_t i=0;i<std::min<std::size_t>(diagnostic_scope->size(),8);++i)
              SENTRYWARD_DEBUG_PRINT(egraph_34);
            if(diagnostic_scope->size()>8)SENTRYWARD_DEBUG_PRINT(egraph_35);
          } else SENTRYWARD_DEBUG_PRINT(egraph_36);
          const auto text=[&](std::string_view value) {
            SENTRYWARD_DEBUG_PRINT(egraph_37);constexpr char hex[]="0123456789abcdef";
            const auto size=std::min<std::size_t>(value.size(),128);
            for(std::size_t i=0;i<size;++i) {
              const auto c=static_cast<unsigned char>(value[i]);
              if(c=='"' || c=='\\')SENTRYWARD_DEBUG_PRINT(egraph_38);
              else if(c<32 || c==127)SENTRYWARD_DEBUG_PRINT(egraph_39);
              else SENTRYWARD_DEBUG_PRINT(egraph_40);
            }
            SENTRYWARD_DEBUG_PRINT(egraph_41);if(size<value.size())SENTRYWARD_DEBUG_PRINT(egraph_42);
          };
          const auto catalogHead=[&](Class id) {
            if(id>=n)return id;const auto cls=owners[id];
            if(auto f=ordinary_folds.find(cls);f!=ordinary_folds.end())return f->second;
            if(auto f=seeded_folds.find(cls);f!=seeded_folds.end())return f->second;
            if(auto h=selected.find(cls);h!=selected.end())return h->second;
            return cls;
          };
          const auto plain=[&](Class id) {
            SENTRYWARD_DEBUG_PRINT(egraph_43);if(id>=n){SENTRYWARD_DEBUG_PRINT(egraph_44);return;}
            const auto& k=snapshot[id];SENTRYWARD_DEBUG_PRINT(egraph_45);text(k.name);
            SENTRYWARD_DEBUG_PRINT(egraph_46);text(k.type);SENTRYWARD_DEBUG_PRINT(egraph_47);text(k.attributes);
            SENTRYWARD_DEBUG_PRINT(egraph_48);text(k.provenance);SENTRYWARD_DEBUG_PRINT(egraph_49);
            for(std::size_t i=0;i<std::min<std::size_t>(k.children.size(),4);++i)SENTRYWARD_DEBUG_PRINT(egraph_50);
            if(k.children.size()>4)SENTRYWARD_DEBUG_PRINT(egraph_51);
            if(k.fold)SENTRYWARD_DEBUG_PRINT(egraph_52);
            SENTRYWARD_DEBUG_PRINT(egraph_53);
          };
          // At most FOUR fixed-local headers. Catalog hints are labelled
          // diagnostics only; they are not selected by the full body proof.
          auto fold=atom!=std::numeric_limits<l::Atom>::max()?reductionFold(atom):std::nullopt;
          const auto head=fold?*fold:catalogHead(atom<n?static_cast<Class>(atom):candidate);
          SENTRYWARD_DEBUG_PRINT(egraph_54);plain(head);
          if(head<n && snapshot[head].fold && snapshot[head].children.size()==2) {
            const auto body=catalogHead(snapshot[head].children[1]);plain(body);
            if(body<n)for(std::size_t i=0;i<std::min<std::size_t>(snapshot[body].children.size(),2);++i)
              plain(catalogHead(snapshot[body].children[i]));
          } else if(head<n)for(std::size_t i=0;i<std::min<std::size_t>(snapshot[head].children.size(),3);++i)
            plain(catalogHead(snapshot[head].children[i]));
          SENTRYWARD_DEBUG_PRINT(egraph_55);
        }
        coverage_cache.emplace(candidate,std::nullopt);++coverage_entries;return nullptr;
      };
      if(coverage_entries>=max_retained_summary_nodes)localCap("coverage_nodes",coverage_entries);
      if(!admitted[candidate])return fail("not_admitted");
      auto scope=semantic?lexicalFree(candidate):scopeFor(candidate);if(!scope)return fail("scope_unknown");
      diagnostic_scope=&*scope;
      if(semantic)exactGroups();
      const auto& scopes=semantic?exact_group_scopes:group_scopes;
      charge();if(!scopes.contains({snapshot[candidate].type,*scope}))return fail("scope_unindexed");
      if(extrema) {
        const auto* form=summarizeExtrema(candidate);
        if(!form || !form->group || form->intervals.empty() ||
            std::get<0>(*form->group)!=snapshot[candidate].type || std::get<1>(*form->group)!=*scope)return fail("extrema_form");
        if(form->intervals.size()>max_interval_entries-interval_entries)localCap("coverage_intervals",interval_entries);
        if(form->residual.size()>max_cached_terms-residual_entries)localCap("coverage_residuals",residual_entries);
        interval_entries+=form->intervals.size();residual_entries+=form->residual.size();
        CoverageKey key{*form->group,*private_zero,form->intervals,form->residual};
        charge(form->intervals.size()+form->residual.size());++coverage_entries;
        auto entry=coverage_cache.emplace(candidate,std::move(key)).first;return &*entry->second;
      }
      auto result=summarize(candidate);if(!result || result->terms().empty())return fail("empty_summary");
      const auto& form=*result;
      std::optional<GroupKey> group_key;bool valid=true;
      const char* denial="no_group";l::Atom denied_atom=std::numeric_limits<l::Atom>::max();
      std::vector<bool> reduction_terms;reduction_terms.reserve(form.terms().size());
      for(const auto& term:form.terms()) {
        charge();
        const auto atom=static_cast<Class>(term.atom);
        const auto fold=reductionFold(term.atom);
        denied_atom=term.atom;
        if(!fold && (atom>=n || !admitted[atom])){denial="invalid_atom";valid=false;break;}
        bool ambiguous=false;
        auto key=fold?admittedFoldGroup(*fold,snapshot[candidate].type,*scope,semantic):std::nullopt;
        if(!key)key=candidateGroup(fold?snapshot[*fold].children[1]:atom,
            snapshot[candidate].type,*scope,semantic,&ambiguous);
        if(ambiguous){denial="ambiguous_group";valid=false;break;}
        const bool reduction=key.has_value();
        reduction_terms.push_back(reduction);
        if(!reduction)continue;
        // Two different admitted Fold-body groups are ambiguous here. Do
        // not guess an origin, demote a conflicting mapped group to an atom,
        // or enumerate root/target combinations.
        if(group_key && *group_key!=*key){denial="conflicting_groups";valid=false;break;}group_key=std::move(*key);
      }
      if(!valid || !group_key)return fail(denial,denied_atom,form.terms().size());
      auto& group=groups.at(*group_key);
      std::vector<l::WeightedInterval> intervals;intervals.reserve(form.terms().size());
      std::vector<l::Term> residual;residual.reserve(form.terms().size());
      for(std::size_t i=0;i<form.terms().size();++i) {
        charge();const auto& term=form.terms()[i];
        if(!reduction_terms[i]) {
          // Builder terms already have sorted distinct atom IDs, exact
          // bounded rational weights, and certified finite source domains.
          // Equality compares this entire residual multiset verbatim.
          charge(128);residual.push_back(term);continue;
        }
        auto domain=mappedDomain(group,term.atom,*scope,semantic);
        if(!domain){denied_atom=term.atom;valid=false;break;}
        intervals.push_back({domain->first,domain->second,term.weight});
      }
      if(!valid)return fail("domain_unknown",denied_atom,form.terms().size());
      auto canonical=builder.canonicalIntervals(intervals);sync();if(!canonical)return fail("interval_canonicalization",denied_atom,form.terms().size());
      if(canonical->size()>max_interval_entries-interval_entries)localCap("coverage_intervals",interval_entries);
      interval_entries+=canonical->size();
      if(residual.size()>max_cached_terms-residual_entries)localCap("coverage_residuals",residual_entries);
      residual_entries+=residual.size();
      CoverageKey key{*group_key,form.constant(),std::move(*canonical),std::move(residual)};
      ++coverage_entries;auto entry=coverage_cache.emplace(candidate,std::move(key)).first;return &*entry->second;
    };
    const auto indexCandidate=[&](Class candidate) {
      auto key=coverage(candidate);if(!key)return;
      if(signature_entries>=max_signatures)localCap("coverage_signatures",signature_entries);
      charge(128*(key->intervals.size()+key->residual.size()+1));
      auto [entry,inserted]=signatures.emplace(*key,candidate);
      if(inserted)++signature_entries;
      if(!inserted)equalities.emplace_back(candidate,entry->second);
    };
    phase="linear_prerequisites";
    // Complete interior certificates unblock many outputs in the next
    // congruence epoch. Give them one existing half-pool share before roots,
    // not an unbounded ancestor query or a fresh proof-work allowance.
    // When there are no admitted whole roots, there is no root work to defer.
    const auto prerequisite_end=root_candidates?std::min(priority_candidates,
        root_candidates+max_component_candidates):priority_candidates;
    for(std::size_t i=root_candidates;i<prerequisite_end;++i)indexCandidate(candidates[i]);
    phase="linear_candidates";
    for(std::size_t i=0;i<root_candidates;++i)indexCandidate(candidates[i]);
    for(std::size_t i=prerequisite_end;i<priority_candidates;++i)indexCandidate(candidates[i]);
    if(!extrema)try {
    phase="moment_setup";
    // Exact moments use the SAME affine linear-mean coverage keys above.
    // Local quadratic merge identities telescope over those private keys;
    // no sample cross-products, Fold members or new enodes are constructed.
    namespace m=moment_certificate;
    m::Limits moment_limits;moment_limits.max_work=std::min<std::size_t>(1000000,remaining);
    m::Builder moments(moment_limits);std::size_t charged_moments=0;
    const auto momentSync=[&] {
      const auto delta=moments.work()-charged_moments;charged_moments=moments.work();charge(delta);
      if(moments.exhausted())localCap("moment_builder",moments.work());
    };
    std::map<GroupKey,std::size_t> group_tokens;
    for(const auto& [key,value]:groups){(void)value;group_tokens.emplace(key,group_tokens.size());}
    struct MeanRecord {m::MeanId id;const Group* group;const CoverageKey* coverage;};
    std::unordered_map<Class,std::optional<MeanRecord>> mean_records;
    const auto mean=[&](Class cls)->std::optional<MeanRecord> {
      cls=owners[cls];charge();if(auto found=mean_records.find(cls);found!=mean_records.end())return found->second;
      if(mean_records.size()>=max_retained_summary_nodes)localCap("moment_means",mean_records.size());
      std::optional<MeanRecord> result;
      if(auto key=coverage(cls,true);key && key->residual.empty()) {
        auto id=moments.mean({group_tokens.at(key->group),0,key->constant,key->intervals,finite_[cls]});
        momentSync();if(id)result=MeanRecord{*id,&groups.at(key->group),key};
      }
      mean_records.emplace(cls,result);return result;
    };
    struct FamilyMeanRecord {m::MeanId id;const Group* group;std::uint64_t step;};
    using FamilyQuery=std::tuple<Class,std::uint32_t,std::uint64_t>;
    std::map<FamilyQuery,std::optional<FamilyMeanRecord>> family_means;
    struct FamilyDomain {std::uint64_t begin,end,step;};
    using FamilyMapQuery=std::tuple<std::size_t,l::Atom,std::uint32_t,std::uint64_t>;
    std::map<FamilyMapQuery,std::optional<FamilyDomain>> family_mappings;
    const auto familyMean=[&](Class cls,std::uint32_t axis,std::uint64_t extent)->std::optional<FamilyMeanRecord> {
      cls=owners[cls];charge(48);const FamilyQuery query{cls,axis,extent};
      if(auto found=family_means.find(query);found!=family_means.end())return found->second;
      const auto fail=[&]()->std::optional<FamilyMeanRecord>{family_means.emplace(query,std::nullopt);return {};};
      if(family_means.size()>=max_retained_summary_nodes)localCap("moment_family_means",family_means.size());
      if(!extent || !admitted[cls])return fail();
      auto scope=lexicalFree(cls);if(!scope)return fail();
      scope->erase(std::remove(scope->begin(),scope->end(),axis),scope->end());
      exactGroups();charge();if(!exact_group_scopes.contains({snapshot[cls].type,*scope}))return fail();
      auto result=summarize(cls);if(!result || result->terms().empty())return fail();
      const auto& form=*result;std::optional<GroupKey> group_key;
      for(const auto& term:form.terms()) {
        charge();const auto atom=term.atom;
        const auto fold=reductionFold(atom);
        if(!fold && (atom>=n || !admitted[atom]))return fail();
        auto key=fold?admittedFoldGroup(*fold,snapshot[cls].type,*scope,true):std::nullopt;
        if(!key)key=candidateGroup(fold?snapshot[*fold].children[1]:static_cast<Class>(atom),
            snapshot[cls].type,*scope,true);
        if(!key || (group_key && *group_key!=*key))return fail();group_key=std::move(*key);
      }
      if(!group_key)return fail();
      auto& group=groups.at(*group_key);const auto& target=snapshot[group.fold];
      const auto token=group_tokens.at(*group_key);std::optional<std::uint64_t> step;
      std::vector<l::WeightedInterval> intervals;intervals.reserve(form.terms().size());
      for(const auto& term:form.terms()) {
        charge(48);const auto atom=term.atom;const FamilyMapQuery key{token,atom,axis,extent};
        auto mapped=family_mappings.find(key);
        if(mapped==family_mappings.end()) {
          std::optional<FamilyDomain> domain;const auto fold=reductionFold(atom);
          const auto source=fold?*fold:static_cast<Class>(atom);
          if(source>=n)return fail();
          // A private seeded-region atom denotes its body ONLY. Its seed is
          // already a separate actual linear term; never index n+owner as a
          // source node or import the complete seeded Fold as a body point.
          const auto body=fold?snapshot[*fold].children[1]:source;
          auto atom_scope=lexicalFree(body);
          if(atom_scope && fold) {
            const auto inner=snapshot[*fold].fold->binder_level;
            atom_scope->erase(std::remove(atom_scope->begin(),atom_scope->end(),inner),atom_scope->end());
          }
          if(atom_scope)atom_scope->erase(std::remove(atom_scope->begin(),atom_scope->end(),axis),atom_scope->end());
          if(atom_scope && *atom_scope==*scope && snapshot[source].type==target.type) {
            std::vector<RectangleAxis> axes{{axis,extent}};std::uint64_t count=1;
            bool valid=true;
            if(fold) {
              const auto& f=snapshot[*fold];count=f.fold->count;
              if(!count || f.fold->binder_level<=axis)valid=false;
              else axes.push_back({f.fold->binder_level,count});
            }
            std::size_t rectangle_work=0;
            auto map=valid?rectangleBodyMap(target.children[1],body,target.fold->binder_level,
                axes,snapshot,members,affine_cache,rectangle_work):std::nullopt;
            charge(rectangle_work);
            if(map && map->coefficients.size()==axes.size() &&
                (axes.size()==1 || map->coefficients[1]==1) && count<=UINT64_MAX-map->begin &&
                (!map->coefficients[0] || extent-1<=(UINT64_MAX-map->begin-count)/map->coefficients[0]))
              domain=FamilyDomain{map->begin,map->begin+count,map->coefficients[0]};
          }
          if(mapping_entries>=max_mappings)localCap("moment_family_mappings",mapping_entries);++mapping_entries;
          mapped=family_mappings.emplace(key,domain).first;
        }
        if(!mapped->second || (step && *step!=mapped->second->step))return fail();step=mapped->second->step;
        intervals.push_back({mapped->second->begin,mapped->second->end,term.weight});
      }
      auto canonical=builder.canonicalIntervals(intervals);sync();if(!canonical || !step)return fail();
      auto id=moments.mean({token,0,form.constant(),*canonical,finite_[cls]});momentSync();if(!id)return fail();
      FamilyMeanRecord record{*id,&group,*step};family_means.emplace(query,record);return record;
    };
    const auto constantForm=[&](Class cls)->std::optional<l::LinearForm> {
      cls=owners[cls];if(!constantHead(cls))return {};
      auto result=summarize(cls);
      if(result && result->terms().empty())return result;return {};
    };
    const auto combine=[&](l::Operation op,const l::LinearForm& a,const l::LinearForm& b) {
      auto result=builder.apply(op,std::array{&a,&b});sync();return result;
    };
    auto one=builder.literal("1");sync();if(!one)localCap("moment_one",builder.work());
    // Fixed epoch head index: one nonrecursive admitted representative per
    // pattern/owner, then O(1) primitive lookups. Add+Negate uses this unary
    // index; no member-pair search or repeated eclass member traversal.
    std::unordered_map<Class,Class> negates,squares;
    std::unordered_map<Class,std::pair<Class,Class>> differences;
    for(auto id:catalog.eligible_ids) {
      charge();const auto& k=snapshot[id];
      if(!ranked[id] || !finite_[owners[id]] || !eligible(k))continue;
      if(k.name=="Negate" && k.children.size()==1)negates.try_emplace(owners[id],owners[k.children[0]]);
      if(k.name=="Subtract" && k.children.size()==2)
        differences.try_emplace(owners[id],owners[k.children[0]],owners[k.children[1]]);
      if(k.name=="Multiply" && k.children.size()==2 && owners[k.children[0]]==owners[k.children[1]])
        squares.try_emplace(owners[id],owners[k.children[0]]);
    }
    for(auto id:catalog.eligible_ids) {
      charge();const auto& k=snapshot[id];
      if(!ranked[id] || !finite_[owners[id]] || !eligible(k) || k.name!="Add" || k.children.size()!=2)continue;
      for(unsigned side=0;side<2;++side)if(auto neg=negates.find(owners[k.children[side]]);neg!=negates.end()) {
        differences.try_emplace(owners[id],owners[k.children[1-side]],neg->second);break;
      }
    }
    const auto difference=[&](Class cls)->std::optional<std::pair<Class,Class>> {
      charge();auto found=differences.find(owners[cls]);if(found==differences.end())return {};return found->second;
    };
    const auto square=[&](Class cls)->std::optional<Class> {
      charge();auto found=squares.find(owners[cls]);if(found==squares.end())return {};return found->second;
    };
    enum class MomentKind {Zero,Correction,IndexedCorrection,Centered,Add,Subtract,Negate,Scale};
    struct MomentNode {
      MomentKind kind=MomentKind::Zero;std::vector<Class> children;
      std::vector<Class> consumed;
      l::LinearForm factor;std::optional<m::Builder::Correction> correction;
      m::Builder::Translation translation;
      std::optional<m::MeanId> center;std::vector<std::pair<std::uint64_t,std::uint64_t>> intervals;
      bool has_moment=false;
    };
    const auto primitive=[&](Class cls)->std::optional<MomentNode> {
      cls=owners[cls];if(!admitted[cls])return {};
      if(zero_[cls])return MomentNode{};
      if(auto fold=ordinary_folds.find(cls);fold!=ordinary_folds.end()) {
        const auto& f=snapshot[fold->second];if(!f.fold->count)return {};
        // A rectangular centered sum can cover separated blocks. Prove its
        // support against the independently certified mean's STORED ranges;
        // never expand the outer or inner reduction elements.
        if(zero_[owners[f.children[0]]]) {
          auto nested=ordinary_folds.find(owners[f.children[1]]);
          if(nested!=ordinary_folds.end()) {
            const auto& h=snapshot[nested->second];
            if(h.fold->count && h.fold->binder_level>f.fold->binder_level && zero_[owners[h.children[0]]])
              if(auto squared=square(h.children[1]))if(auto delta=difference(*squared))
                for(unsigned side=0;side<2;++side) {
                  const auto center=side?delta->first:delta->second;
                  const auto sample=side?delta->second:delta->first;
                  auto c=mean(center);auto deps=lexicalFree(center);
                  if(!c || !deps || !finite_[sample] ||
                      std::binary_search(deps->begin(),deps->end(),f.fold->binder_level) ||
                      std::binary_search(deps->begin(),deps->end(),h.fold->binder_level))continue;
                  const auto& target=snapshot[c->group->fold];std::size_t rectangle_work=0;
                  const std::vector<RectangleAxis> axes{{f.fold->binder_level,f.fold->count},
                      {h.fold->binder_level,h.fold->count}};
                  auto map=rectangleBodyMap(target.children[1],sample,target.fold->binder_level,
                      axes,snapshot,members,affine_cache,rectangle_work);charge(rectangle_work);
                  if(!map || map->coefficients.size()!=2 || map->coefficients[1]!=1 ||
                      map->coefficients[0]<h.fold->count)continue;
                  const auto step=map->coefficients[0],width=h.fold->count;
                  const auto& support=c->coverage->intervals;
                  bool valid=true;std::uint64_t rows=0;
                  MomentNode out;out.kind=MomentKind::Centered;out.center=c->id;out.has_moment=true;
                  for(const auto& interval:support) {
                    charge();const auto begin=interval.begin,end=interval.end;
                    if(begin<map->begin || end<begin || (begin-map->begin)%step ||
                        (begin-map->begin)/step!=rows){valid=false;break;}
                    const auto length=end-begin;
                    if(!length || length%width || (step!=width && length!=width) ||
                        length/width>f.fold->count-rows){valid=false;break;}
                    rows+=length/width;out.intervals.emplace_back(begin,end);
                  }
                  if(valid && rows==f.fold->count)return out;
                }
          }
        }
        if(auto squared=square(f.children[1]))if(auto delta=difference(*squared))
          for(unsigned side=0;side<2;++side) {
            auto c=mean(side?delta->first:delta->second);if(!c)continue;
            const auto sample=side?delta->second:delta->first;
            auto deps=lexicalFree(side?delta->first:delta->second);
            if(!deps || std::binary_search(deps->begin(),deps->end(),f.fold->binder_level) || !finite_[sample])continue;
            const auto& target=snapshot[c->group->fold];std::size_t rectangle_work=0;
            const std::vector<RectangleAxis> axes{{f.fold->binder_level,f.fold->count}};
            auto map=rectangleBodyMap(target.children[1],sample,target.fold->binder_level,
                axes,snapshot,members,affine_cache,rectangle_work);charge(rectangle_work);
            if(!map || map->coefficients.size()!=1 || map->coefficients[0]!=1 ||
                f.fold->count>UINT64_MAX-map->begin)continue;
            MomentNode out;out.kind=MomentKind::Centered;out.center=c->id;
            out.intervals.emplace_back(map->begin,map->begin+f.fold->count);out.has_moment=true;return out;
          }
      }
      if(auto fold=sum_folds.find(cls);fold!=sum_folds.end()) {
        const auto& f=snapshot[fold->second];const auto binder=f.fold->binder_level;
        auto seed_scope=lexicalFree(f.children[0]);
        if(f.fold->count && seed_scope && !std::binary_search(seed_scope->begin(),seed_scope->end(),binder)) {
          auto body=owners[f.children[1]];l::LinearForm coefficient=*one;
          for(unsigned depth=0;depth<8;++depth) {
            charge();if(auto squared=square(body))if(auto delta=difference(*squared)) {
              auto a=familyMean(delta->first,binder,f.fold->count),b=familyMean(delta->second,binder,f.fold->count);
              if(!a || !b || a->group!=b->group || a->step!=b->step)break;
              auto correction=moments.correction(a->id,b->id,coefficient.constant());momentSync();if(!correction)break;
              MomentNode out;out.kind=MomentKind::IndexedCorrection;out.correction=std::move(correction);
              out.translation={0,a->step,f.fold->count};out.children.push_back(owners[f.children[0]]);
              out.has_moment=true;return out;
            }
            auto head=selected.find(body);if(head==selected.end())break;
            const auto& k=snapshot[head->second];std::optional<l::LinearForm> factor;Class next=n;
            if(k.name=="Multiply")for(unsigned side=0;side<2 && !factor;++side)
              if(auto c=constantForm(k.children[side])){factor=std::move(c);next=owners[k.children[1-side]];}
            if(k.name=="Divide")if(auto c=constantForm(k.children[1])) {
              factor=combine(l::Operation::Divide,*one,*c);next=owners[k.children[0]];
            }
            if(!factor || next==n)break;
            auto product=combine(l::Operation::Multiply,coefficient,*factor);if(!product)break;
            coefficient=std::move(*product);body=next;
          }
        }
      }
      // Complete scalar coefficient chain, capped at eight immediate heads.
      // A bare or partially scaled square is NOT a Welford correction unless
      // its exact coefficient already satisfies the local count identity.
      auto current=cls;l::LinearForm coefficient=*one;std::vector<Class> consumed;
      for(unsigned depth=0;depth<8;++depth) {
        charge();if(auto squared=square(current))if(auto delta=difference(*squared)) {
          auto a=mean(delta->first),b=mean(delta->second);if(!a || !b)return {};
          auto correction=moments.correction(a->id,b->id,coefficient.constant());momentSync();
          if(!correction)return {};
          MomentNode out;out.kind=MomentKind::Correction;out.correction=std::move(correction);
          out.consumed=std::move(consumed);out.has_moment=true;return out;
        }
        auto head=selected.find(current);if(head==selected.end())break;
        const auto& k=snapshot[head->second];std::optional<l::LinearForm> factor;Class next=n;
        if(k.name=="Multiply")for(unsigned side=0;side<2 && !factor;++side)
          if(auto c=constantForm(k.children[side])){factor=std::move(c);next=owners[k.children[1-side]];}
        if(k.name=="Divide")if(auto c=constantForm(k.children[1])) {
          factor=combine(l::Operation::Divide,*one,*c);next=owners[k.children[0]];
        }
        if(!factor || next==n)break;
        auto product=combine(l::Operation::Multiply,coefficient,*factor);if(!product)break;
        coefficient=std::move(*product);current=next;consumed.push_back(current);
      }
      return {};
    };
    std::unordered_map<Class,std::optional<MomentNode>> plans;
    std::unordered_map<Class,MomentNode> drafts;
    std::vector<Class> order;std::unordered_set<Class> active;
    const auto plan=[&](Class root) {
      struct Task {Class cls;bool ready;};std::vector<Task> pending{{owners[root],false}};
      while(!pending.empty()) {
        charge();auto [cls,ready]=pending.back();pending.pop_back();cls=owners[cls];
        if(plans.contains(cls))continue;
        if(plans.size()+active.size()>=max_retained_summary_nodes)
          localCap("moment_plans",plans.size()+active.size());
        if(active.size()>=max_active_nodes)localCap("moment_active",active.size());
        if(!ready) {
          if(!active.insert(cls).second)localCap("moment_cycle",active.size());
          if(auto p=primitive(cls)) {
            if(p->children.empty()){plans.emplace(cls,std::move(p));order.push_back(cls);active.erase(cls);continue;}
            auto draft=drafts.emplace(cls,std::move(*p)).first;
            pending.push_back({cls,true});
            for(auto child:draft->second.children){charge();if(!plans.contains(child))pending.push_back({child,false});}
            continue;
          }
          auto head=selected.find(cls);
          if(!admitted[cls] || head==selected.end()) {plans.emplace(cls,std::nullopt);active.erase(cls);continue;}
          const auto& k=snapshot[head->second];MomentNode p;bool valid=true;
          if(k.name=="Add" || k.name=="Subtract" || k.name=="Negate") {
            p.kind=k.name=="Add"?MomentKind::Add:k.name=="Subtract"?MomentKind::Subtract:MomentKind::Negate;
            for(auto c:k.children)p.children.push_back(owners[c]);
          } else if(k.name=="Multiply" || k.name=="Divide") {
            std::optional<l::LinearForm> factor;Class child=n;
            if(k.name=="Multiply")for(unsigned side=0;side<2 && !factor;++side)
              if(auto c=constantForm(k.children[side])){factor=std::move(c);child=owners[k.children[1-side]];}
            if(k.name=="Divide")if(auto c=constantForm(k.children[1])) {
              factor=combine(l::Operation::Divide,*one,*c);child=owners[k.children[0]];
            }
            if(!factor || child==n)valid=false;
            else {p.kind=MomentKind::Scale;p.factor=std::move(*factor);p.children.push_back(child);}
          } else valid=false;
          if(!valid) {plans.emplace(cls,std::nullopt);active.erase(cls);continue;}
          // Keep pending internal records separate until all children are
          // validated; no partially completed component is a certificate.
          pending.push_back({cls,true});
          for(auto child:p.children)if(!plans.contains(child))pending.push_back({child,false});
          drafts.emplace(cls,std::move(p));
          continue;
        }
        MomentNode p=std::move(drafts.at(cls));drafts.erase(cls);bool valid=true;
        for(auto child:p.children) {
          const auto q=plans.find(child);if(q==plans.end() || !q->second){valid=false;break;}
          p.has_moment|=q->second->has_moment;
        }
        plans.emplace(cls,valid?std::optional{std::move(p)}:std::nullopt);order.push_back(cls);active.erase(cls);
      }
    };
    phase="moment_plans";
    for(auto id:owner_ids)if(admitted[id])plan(id);
    std::unordered_set<Class> parented;
    for(const auto& [cls,p]:plans)if(p && p->has_moment) {
      for(auto child:p->children)if(plans.at(child)->has_moment)parented.insert(child);
      // Complete corrections consume at most eight immediate scalar heads.
      // Those internal partial factors are not independent candidate roots;
      // otherwise a balanced tree could spend the fixed root cap on them.
      for(auto child:p->consumed){charge();parented.insert(child);}
    }
    constexpr std::size_t moment_candidate_cap=16;
    std::vector<Class> moment_candidates;std::unordered_set<Class> moment_candidate_set;
    const auto appendMoment=[&](Class cls) {
      charge();if(moment_candidate_set.contains(cls))return true;
      if(moment_candidates.size()>=moment_candidate_cap)return false;
      moment_candidate_set.insert(cls);moment_candidates.push_back(cls);return true;
    };
    const auto appendMomentTail=[&](Class cls) {
      if(!appendMoment(cls))return;auto child=cls;
      // Also certify the unscaled M2 beneath a bounded final scalar tail.
      for(unsigned depth=0;depth<8 && plans.at(child)->kind==MomentKind::Scale;++depth) {
        child=plans.at(child)->children[0];if(!appendMoment(child))break;
      }
    };
    // Output-reachable complete components precede optional interior roots.
    // Traverse one ranked witness DAG once; when a whole plan is found its
    // descendants need not compete for the fixed sixteen-certificate cap.
    std::vector<Class> frontier;
    std::unordered_set<Class> frontier_seen;
    const auto appendFrontier=[&](Class cls) {
      charge();cls=owners[cls];if(frontier_seen.contains(cls) || frontier.size()>=max_active_nodes)return;
      frontier_seen.insert(cls);frontier.push_back(cls);
    };
    for(auto cls:left)appendFrontier(cls);for(auto cls:right)appendFrontier(cls);
    for(std::size_t i=0;i<frontier.size();++i) {
      charge();auto cls=frontier[i];
      if(auto p=plans.find(cls);p!=plans.end() && p->second && p->second->has_moment) {
        appendMomentTail(cls);continue;
      }
      Class head=n;
      if(literals.contains(cls))continue;
      if(auto h=selected.find(cls);h!=selected.end())head=h->second;
      else if(auto h=ordinary_folds.find(cls);h!=ordinary_folds.end())head=h->second;
      else if(!members[cls].empty())head=members[cls].front();
      if(head==n)continue;
      for(auto child:snapshot[head].children)appendFrontier(child);
    }
    for(const auto& [cls,p]:plans) {
      charge();if(moment_candidates.size()>=moment_candidate_cap)break;
      if(p && p->has_moment && !parented.contains(cls))appendMomentTail(cls);
    }
    std::vector<std::pair<Class,m::Builder::FamilySignature>> moment_signatures;
    for(auto root:moment_candidates) {
      // Reverse selected-DAG adjoints accumulate one scalar coefficient per
      // node. Never cache/copy a growing polynomial/term vector per prefix.
      std::unordered_map<Class,l::LinearForm> weights;weights.emplace(root,*one);
      m::Builder::FamilyForm form;std::map<m::MeanId,std::vector<m::Interval>> centered_pieces;
      std::size_t centered_entries=0;bool valid=true;
      for(auto it=order.rbegin();it!=order.rend() && valid;++it) {
        charge();const auto cls=*it;auto found=weights.find(cls);if(found==weights.end())continue;
        const auto& p=*plans.at(cls);const auto weight=found->second;
        if(p.kind==MomentKind::Zero)continue;
        if(p.kind==MomentKind::Correction) {
          if(!moments.addFixed(form,*p.correction,weight.constant()))valid=false;
          momentSync();continue;
        }
        if(p.kind==MomentKind::IndexedCorrection) {
          if(!moments.addFamily(form,*p.correction,p.translation,weight.constant(),true))valid=false;
          momentSync();if(!valid)break;
          // Fall through to the seed child with the SAME outer coefficient;
          // the seed is outside the Fold binder and is never repeated.
        }
        if(p.kind==MomentKind::Centered) {
          if(p.intervals.size()>1024-centered_entries){valid=false;break;}
          centered_entries+=p.intervals.size();
          for(auto [begin,end]:p.intervals) {
            charge();centered_pieces[*p.center].push_back({begin,end,weight.constant()});
          }
          continue;
        }
        for(std::size_t i=0;i<p.children.size();++i) {
          auto next=std::optional{weight};
          if((p.kind==MomentKind::Subtract && i==1) || p.kind==MomentKind::Negate) {
            next=builder.apply(l::Operation::Negate,std::array{&weight});sync();
          }
          if(p.kind==MomentKind::Scale)next=combine(l::Operation::Multiply,weight,p.factor);
          if(!next){valid=false;break;}
          auto previous=weights.find(p.children[i]);
          if(previous!=weights.end())next=combine(l::Operation::Add,previous->second,*next);
          if(!next){valid=false;break;}weights.insert_or_assign(p.children[i],std::move(*next));
        }
      }
      if(!valid)continue;
      for(const auto& [center,pieces]:centered_pieces) {
        if(!moments.addCentered(form,center,pieces,true)){valid=false;break;}momentSync();
      }
      if(!valid)continue;
      auto signature=moments.finishFamily(form);momentSync();
      if(!signature)continue;
      for(const auto& [other,key]:moment_signatures) {
        charge();auto equal=moments.equalFamilies(*signature,key);momentSync();
        if(equal && *equal)equalities.emplace_back(root,other);
      }
      moment_signatures.emplace_back(root,std::move(*signature));
    }
    } catch(const LocalCap&) {
      // Optional moment planning cannot discard complete linear certificates
      // or prevent the final bounded prefix fallback from being attempted.
      settleHelperFailure();
      traceCap(builder.work());
    }
    phase="scalar_linear_components";
    for(std::size_t i=priority_candidates;i<candidates.size();++i)indexCandidate(candidates[i]);
    sync();
    if(!extrema) {
      phase="scoped_nested_sums";
      if(trace){domain_fallback_reports=16;domain_denial_reports=16;}
      // Preserve every old linear/moment opportunity before this optional
      // complete-root proof. It borrows the SAME unrefreshed epoch ledger.
      // This is an existing paired-root window, not a per-output ancestor
      // scheduler; context-local Sum facts never create internal unions.
      BodyScopeRegistry scopes(body_heads_index,max_retained_summary_nodes);
      ScopedSumProof sums(scopes,builder,summarize,reductionFold,sync,scoped_sum_query);
      auto root_scope=scopes.output();
      if(root_scope) {
        std::size_t root_slots=0;
        const auto pairs=std::min(left.size(),right.size());
        for(std::size_t i=0;i<pairs && root_slots+2<=max_root_candidates;++i) {
          charge();const auto a=owners[left[i]],b=owners[right[i]];
          if(a==b)continue;root_slots+=2;
          if(a>=n || b>=n || snapshot[a].type!="real" || snapshot[b].type!="real" ||
              !finite_[a] || !finite_[b])continue;
          auto assignment=sums.root();
          BodyHeadIndex::FallbackTrace reports(body_heads_index,trace?&domain_fallback_reports:nullptr,
              trace?&domain_denial_reports:nullptr);
          if(affineBodyMapWithHeads(a,b,UINT32_MAX,snapshot,false,&remaining,body_heads_index,
              UINT32_MAX,nullptr,nullptr,nullptr,&sums,&assignment,*root_scope,*root_scope))
            equalities.emplace_back(a,b);
        }
      }
      sync();
      if(trace)SENTRYWARD_DEBUG_PRINT(egraph_56);
    }
  } catch(const LocalCap&) {
    if(!catalog_ready)catalog_failed=true;
    // Optional proof cap, never a partial certificate. Earlier completed
    // whole-root certificates are still sound and can be retained.
    settleHelperFailure();
    traceCap(builder.work());
  }
  if(linear_statistics.queries && sentryward::debug::environment("SENTRYWARD_EGRAPH_TRACE"))
    SENTRYWARD_DEBUG_PRINT(egraph_57);
  if(trace) {
    // One compact line per reducer epoch, including an interrupted mapper's
    // consumed work. Map components overlap group/proof costs; no per-node
    // trace, new walk, reset allowance or proof assumption is introduced.
    SENTRYWARD_DEBUG_PRINT(egraph_58);
  }
  if(shape_capped && sentryward::debug::environment("SENTRYWARD_EGRAPH_TRACE"))
    SENTRYWARD_DEBUG_PRINT(egraph_59);
  };
  // Extrema first: they can discharge a shared max/min scale before Sum
  // moments/prefix candidates consume the same fixed aggregate pool.
  if(modes[0])run_mode(s::Reducer::Max);
  if(modes[1])run_mode(s::Reducer::Min);
  if(modes[2])run_mode(s::Reducer::Sum);
  for(auto [a,b]:equalities)rewrite(a,b);
}

} // namespace
} // namespace sentryward::equivalence
