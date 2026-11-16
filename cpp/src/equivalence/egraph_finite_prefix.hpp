#pragma once
#include "egraph_engine.hpp"

namespace sentryward::equivalence {
namespace {

// Bounded actual-source cohort bridge. A zero-entry frame output is
// certified only after the recurrence,
// dispatch, projection and causal region all share a complete scoped proof.
// One immutable cohort epoch/aggregate allowance; no allowance per output.
inline auto Engine::proveFirstFinitePrefixOutput(std::span<const Class> left,std::span<const Class> right) -> bool {
  namespace fp=finite_prefix_certificate;
  namespace cr=causal_region_certificate;
  using Registry=BodyScopeRegistry;
  using Use=BoundedProducts::Use;
  constexpr Class none=std::numeric_limits<Class>::max();
  constexpr Registry::Id virtual_scope=UINT32_MAX;
  std::size_t initial_work=epoch_work_allowance::default_work,remaining=initial_work,policy_work=0;
  epoch_work_allowance::Size physical_epoch;
  std::vector<std::pair<Class,Class>> proved_roots;
  std::size_t roots_attempted=0,completed_frames=0;
  std::size_t source_cases=0,attached=0,products_count=0,p_samples=0,q_samples=0;
  std::size_t proved_updates=0,scope_count=0,coverage_work=0;
  std::size_t potential_coverage_work=0,product_coverage_work=0;
  bool potential_coverage_cap=false,product_coverage_cap=false;
  unsigned candidate_flags=0,candidate_factors=0,source_factors_count=0;
  unsigned selected_projection_count=0,selected_weight_count=0;
  std::size_t bounds_scopes=0,header_entries=0,relation_entries=0,original_views=0,transported_uses=0;
  std::size_t context_entries=0,sample_context_entries=0,sample_context_hits=0,reference_plans=0,reference_plan_reads=0;
  std::size_t context_capacity=0,context_plan_rejections=0;
  std::size_t retained_record_rejections=0,image_entries=0,identity_slots=0;
  const char* retained_record_cap="none";
  const char* context_plan_cap="none";
  unsigned product_calls=0,last_product_kind=0,target_calls=0,mask_calls=0;
  std::string_view entry_stage="none";
  bool region_complete=false,dispatch_complete=false,region_cap=false;
  cr::Statistics region_statistics;
  bool body_cap=false,scope_cap=false;
  const char* stage="configuration";bool component=false,output_proved=false;
  const bool prefix_trace=sentryward::debug::environment("SENTRYWARD_EGRAPH_PREFIX_TRACE")!=nullptr;
  enum class PrefixPhase : unsigned {Source,Templates,Sampling,Output,Carry,Region,Dispatch,Count};
  std::array<std::size_t,static_cast<unsigned>(PrefixPhase::Count)> phase_work{};
  struct PhaseTrace {
    bool enabled;std::size_t& remaining;decltype(phase_work)& work;
    std::size_t previous;PrefixPhase phase=PrefixPhase::Source;
    void finish() {
      if(enabled){work[static_cast<unsigned>(phase)]+=previous-remaining;previous=remaining;}
    }
    void next(PrefixPhase value) {finish();phase=value;}
    ~PhaseTrace(){finish();}
  };
  struct RootTrace {
    std::size_t ordinal,work,cases,attachments,updates,frames,proved;
    const char* stage;std::string_view entry;
    struct Coordinate {
      Class predicate=UINT32_MAX,time=UINT32_MAX,output=UINT32_MAX;
      std::array<Class,2> children{UINT32_MAX,UINT32_MAX};
      std::array<std::uint64_t,2> constants{},coefficients{};
      std::array<std::uint32_t,2> axes{UINT32_MAX,UINT32_MAX};
      std::array<std::size_t,2> terms{};
      std::uint64_t source_begin=0,source_end=0,time_begin=0,time_end=0;
      Registry::Id original_scope=UINT32_MAX,piece_scope=UINT32_MAX;
      std::uint32_t binder=UINT32_MAX;unsigned candidate=0;
    } coordinate;
  };
  RootTrace::Coordinate coordinate_trace;
  std::array<RootTrace,64> root_trace{};std::size_t root_trace_size=0;
  unsigned carry_denial_trace=0;
  std::size_t member_queries=0,member_hits=0,member_entries=0;
  fp::Statistics prefix_statistics{};
  const auto run=[&]() {
    if(left.empty() || left.size()!=right.size() || !options_.ideal_numeric_outputs ||
        !options_.outer_binders || options_.outer_binders+1>=Registry::max_depth ||
        options_.outer_index_extents.size()!=options_.outer_binders)return;
    // Necessary O(stored roots) syntax filter before any epoch catalog.
    // This schedules only; every actual head/path is still re-read below.
    bool candidate=false;
    for(std::size_t i=0;i<left.size();++i) {
      poll();
      if(!remaining){stage="configuration_cap";return;}--remaining;
      const auto& l=nodes_[find(left[i])];const auto& r=nodes_[find(right[i])];
      if((idealSelect(l) || (l.type=="real" && arithmetic(l) && l.name=="Add")) &&
          ((arithmetic(r) && r.name=="Add") ||
          (r.kind==static_cast<int>(s::Kind::Fold) && r.fold && r.fold->reducer==s::Reducer::Sum)))
        {candidate=true;break;}
    }
    if(!candidate)return;
    const auto plan=physicalWorkPlan(nodes_,initial_work,initial_work-remaining);
    initial_work=plan.initial;remaining=plan.remaining;policy_work=plan.sizing;physical_epoch=plan.physical;
    if(plan.exhausted){stage="configuration_cap";return;}
    propagateProperties();
    // Borrow the maintained immutable membership epoch. A requested owner
    // copies only its capped physical prefix, once; never reconstruct all
    // stored classes or walk the graph to discover two exact Index heads.
    const auto boundary=nodes_.size(),owner_epoch=unions_;
    struct MemberView {
      std::size_t boundary;std::function<std::span<const Class>(Class)> lookup;
      std::size_t size() const {return boundary;}
      std::span<const Class> operator[](Class id) const {return lookup(id);}
    };
    std::unordered_map<Class,std::vector<Class>> member_views;
    BodyHeadIndex* member_budget=nullptr;
    MemberView members{boundary,[&](Class cls)->std::span<const Class> {
      if(unions_!=owner_epoch || nodes_.size()!=boundary)
        throw std::logic_error("prefix catalog epoch changed");
      if(cls>=boundary || !member_budget)return {};
      ++member_queries;cls=find(cls);
      if(auto found=member_views.find(cls);found!=member_views.end()) {
        // Like the existing linear body_members view, a cached span stays
        // stable for repeated size/front reads within ONE head operation.
        // Do not turn its second read into an empty span on local-cap exit;
        // the unchanged actual head/matcher charge points deny that query.
        ++member_hits;return found->second;
      }
      if(!member_budget->charge())return {};
      auto view=class_members_.prefix(cls,boundary,BodyHeadIndex::max_members,
          [&](std::size_t amount){return member_budget->charge(amount);});
      // A failed query-local allowance must not poison later cohort uses.
      if(!view || !member_budget->charge(2))return {};
      auto found=member_views.emplace(cls,std::move(*view)).first;
      ++member_entries;return found->second;
    }};
    AffineCache affine;BodyHeadIndex heads(*this,nodes_,members,affine,&remaining);
    member_budget=&heads;
    // ONE immutable cohort index. The existing local header/sole tokens are
    // shared across all roots; no new allowance or per-root array is created.
    heads.initializeSoleHints();
    const auto indexCoordinate=[&](std::uint32_t level) {
      Class selected=none;
      for(const auto* type:{"u64","exact:u64"}) {
        if(!heads.charge(16+3*std::char_traits<char>::length(type)))return none;
        Key probe;probe.kind=static_cast<int>(s::Kind::Index);probe.type=type;probe.index_level=level;
        auto found=memo_.find(probe);
        if(found!=memo_.end())selected=std::min(selected,found->second);
      }
      return selected==none?none:find(selected);
    };
    const Class state_coordinate=indexCoordinate(options_.outer_binders);
    const Class nested_state_coordinate=indexCoordinate(options_.outer_binders+1);
    Registry registry(heads,epoch_work_allowance::retainedHeaderCapacity(physical_epoch));
    JoinedProducts products(registry,initial_work);JoinedFamilies potentials(registry,initial_work);
    struct Counters {
      Registry& registry;JoinedProducts& products;JoinedFamilies& potentials;
      std::size_t& scopes;std::size_t& bounds;std::size_t& headers;std::size_t& relations;std::size_t& coverage;
      bool& body_cap;bool& scope_cap;std::size_t& originals;std::size_t& transports;
      std::size_t& potential_work;std::size_t& product_work;bool& potential_cap;bool& product_cap;
      std::size_t& contexts;std::size_t& samples;std::size_t& sample_hits;std::size_t& plans;std::size_t& plan_reads;
      std::size_t& context_capacity;std::size_t& context_rejections;const char*& context_cap;
      std::size_t& record_rejections;const char*& record_cap;std::size_t& images;std::size_t& identities;
      ~Counters() {
        scopes=registry.scopes.size();const auto s=registry.bounds.statistics();bounds=s.scopes;scope_cap=s.exhausted;
        headers=registry.heads.roots_cache.size();relations=products.products.relations.size();body_cap=registry.heads.exhausted;
        coverage=potentials.coverageWork()+products.geometry.coverageWork();
        potential_work=potentials.coverageWork();product_work=products.geometry.coverageWork();
        potential_cap=potentials.intervalExhausted();product_cap=products.geometry.intervalExhausted();
        originals=registry.original_body_views;transports=registry.transported_body_uses;
        contexts=registry.contexts.size();samples=registry.sample_contexts.size();sample_hits=registry.sample_context_hits;
        plans=products.products.reference_products.size();plan_reads=products.products.reference_reads;
        context_capacity=registry.max_retained_records;context_rejections=registry.context_plan_rejections;
        context_cap=registry.context_plan_cap;
        record_rejections=registry.retained_record_rejections;record_cap=registry.retained_record_cap;
        images=registry.images.size();identities=registry.identity_slots;
      }
    } counters{registry,products,potentials,scope_count,bounds_scopes,header_entries,relation_entries,
        coverage_work,body_cap,scope_cap,original_views,transported_uses,
        potential_coverage_work,product_coverage_work,potential_coverage_cap,product_coverage_cap,
        context_entries,sample_context_entries,sample_context_hits,reference_plans,reference_plan_reads,
        context_capacity,context_plan_rejections,context_plan_cap,
        retained_record_rejections,retained_record_cap,image_entries,identity_slots};
    auto outer=registry.output();if(!outer || state_coordinate==none){stage="outer_scope";return;}
    const auto head=[&](Class cls)->const Key* {
      if(cls>=nodes_.size())return nullptr;
      auto id=heads.compact(cls);return id?&nodes_[*id]:nullptr;
    };
    const auto prefix_node=[&](fp::ExprId expression,fp::ScopeId scope)->std::optional<fp::Node> {
      if(expression>=nodes_.size() || scope>=registry.scopes.size())return {};
      const auto cls=find(static_cast<Class>(expression));auto k=head(cls);
      if(!k || !realType(k->type) || !k->provenance.empty() || !finite_[cls] || !heads.charge())return {};
      fp::Node node;node.type=k->type;node.attributes=k->attributes;node.provenance=k->provenance;
      const auto& free=free_indices_[cls];
      if(free.size()>node.free_indices.size() || !heads.charge(free.size()+1))return {};
      for(auto level:free)if(!registry.count(scope,level))return {};
      std::copy(free.begin(),free.end(),node.free_indices.begin());node.free_count=free.size();
      // A conservative certified dependency upper bound is sufficient for
      // absence checks; it can reject, but cannot invent independence.
      node.exact_dependencies=true;node.finite_ideal=true;
      // This is an existing certified value fact of this e-class.
      // Selected Fold/Multiply aliases must not hide it from the SAME parser.
      if(zero_[cls]) {node.kind=fp::Kind::Zero;node.attributes={};return node;}
      if(k->kind==static_cast<int>(s::Kind::Constant) && k->attributes.empty()) {
        if(k->name=="0")node.kind=fp::Kind::Zero;
        else if(k->name=="1")node.kind=fp::Kind::One;
      } else if(k->kind==static_cast<int>(s::Kind::InitialCell)) {
        if(k->children.size()!=1)return {};
      } else if(arithmetic(*k) || idealOperation(*k)) {
        if(k->name=="Add")node.kind=fp::Kind::Add;
        else if(k->name=="Subtract")node.kind=fp::Kind::Subtract;
        else if(k->name=="Negate")node.kind=fp::Kind::Negate;
        else if(k->name=="Multiply")node.kind=fp::Kind::Multiply;
        else if(k->name=="Exp")node.kind=fp::Kind::NaturalExp;
        else return {};
        if(k->children.size()>node.children.size())return {};
        node.arity=k->children.size();
        for(unsigned i=0;i<node.arity;++i)node.children[i]=find(k->children[i]);
      } else return {};
      return node;
    };
    // Private semantic descriptors live for this ONE immutable proof epoch.
    // Every descriptor is populated only from its completed joined-family
    // admission; different frames never share a virtual expression identity.
    std::unordered_map<fp::ExprId,fp::Node> virtual_prefix_nodes;
    std::unordered_map<cr::ExprId,cr::Node> joined_projection_nodes;
    const auto region_node=[&](cr::Use use)->std::optional<cr::Node> {
      if(use.scope==virtual_scope) {
        auto found=joined_projection_nodes.find(use.expression);
        return found==joined_projection_nodes.end()?std::nullopt:std::optional{found->second};
      }
      auto p=prefix_node(use.expression,use.scope);auto k=head(use.expression);
      if(!k || use.scope>=registry.scopes.size())return {};
      cr::Node n;
      if(p) {
        n.type=p->type;n.attributes=p->attributes;n.provenance=p->provenance;
        n.finite_ideal=p->finite_ideal;n.exact_dependencies=p->exact_dependencies;
        n.free_indices=p->free_indices;n.free_count=p->free_count;n.arity=p->arity;
        for(unsigned i=0;i<n.arity;++i)n.children[i]=p->children[i];
        switch(p->kind) {
          case fp::Kind::Zero:n.kind=cr::Kind::Zero;break;
          case fp::Kind::Add:n.kind=cr::Kind::Add;break;
          case fp::Kind::Subtract:n.kind=cr::Kind::Subtract;break;
          case fp::Kind::Negate:n.kind=cr::Kind::Negate;break;
          case fp::Kind::Multiply:n.kind=cr::Kind::Multiply;break;
          case fp::Kind::NaturalExp:n.kind=cr::Kind::NaturalExp;break;
          default:break;
        }
        return n;
      }
      const auto cls=find(use.expression);
      if(!realType(k->type) || !k->provenance.empty() || !finite_[cls] || !heads.charge())return {};
      if(k->kind==static_cast<int>(s::Kind::Fold) && k->fold && k->fold->reducer==s::Reducer::Sum &&
          k->children.size()==2 && k->attributes.empty()) {
        n.kind=cr::Kind::Sum;n.binder=k->fold->binder_level;n.count=k->fold->count;
      } else if(idealSelect(*k))n.kind=cr::Kind::Select;else return {};
      n.type=k->type;n.attributes=k->attributes;n.provenance=k->provenance;
      n.finite_ideal=true;n.exact_dependencies=true;n.arity=k->children.size();
      for(unsigned i=0;i<n.arity;++i)n.children[i]=find(k->children[i]);
      const auto& free=free_indices_[cls];if(free.size()>8 || !heads.charge(free.size()+1))return {};
      for(auto level:free)if(!registry.count(use.scope,level))return {};
      std::copy(free.begin(),free.end(),n.free_indices.begin());n.free_count=free.size();return n;
    };
    // Source syntax is selected locally; every actual update is later
    // rechecked by the existing core, and every P/Q leaf by the full mapper.
    struct Update {Class previous,current_p,previous_p,injection;};
    const auto update=[&](Class state,Class zero_baseline=std::numeric_limits<Class>::max())->std::optional<Update> {
      auto k=head(state);if(!k || !arithmetic(*k) || k->name!="Add")return {};
      std::optional<Update> result;
      for(unsigned arm=0;arm<2;++arm) {
        auto m=head(k->children[arm]);if(!m || !arithmetic(*m) || m->name!="Multiply")continue;
        for(unsigned side=0;side<2;++side) {
          auto exp=head(m->children[side]);
          if(!exp || !idealOperation(*exp) || exp->name!="Exp" || exp->children.size()!=1)continue;
          auto difference=head(exp->children[0]);if(!difference)return {};
          std::optional<std::pair<Class,Class>> pair;
          if(idealOperation(*difference) && difference->name=="Subtract" && difference->children.size()==2)
            pair=std::pair{find(difference->children[0]),find(difference->children[1])};
          else if(arithmetic(*difference) && difference->name=="Add")
            for(unsigned sign=0;sign<2;++sign) {
              auto negative=head(difference->children[sign]);
              if(negative && idealOperation(*negative) && negative->name=="Negate" && negative->children.size()==1) {
                if(pair)return {};pair=std::pair{find(difference->children[1-sign]),find(negative->children[0])};
              }
            }
          // The existing core admits Exp(P) as Exp(P-E) only when this
          // ACTUAL baseline is a certified ideal-real zero. State carry is
          // a separate nonzero operand and is never erased by this hint.
          if(!pair && zero_baseline<nodes_.size() && nodes_[find(zero_baseline)].type=="real" &&
              finite_[find(zero_baseline)] && zero_[find(zero_baseline)])
            pair=std::pair{find(exp->children[0]),find(zero_baseline)};
          if(pair) {
            if(result)return {};
            result=Update{find(m->children[1-side]),pair->first,pair->second,find(k->children[1-arm])};
          }
        }
      }
      return result;
    };
    // ONE core/region builder and ONE aggregate ledger, not a replay or
    // fresh allowance per root. Actual query callbacks are installed only
    // while their complete original-root attachment records are alive.
    fp::Relate core_relate;
    cr::RelateProduct region_product;cr::ProveMask region_mask;cr::ProveOutputTarget region_target;
    cr::ProveStateTarget region_state_target;
    fp::Limits prefix_limits;prefix_limits.max_work=initial_work;
    prefix_limits.max_states=epoch_work_allowance::retainedHeaderCapacity(physical_epoch);
    cr::Limits region_limits;region_limits.max_work=initial_work;
    region_limits.max_nodes=std::max(region_limits.max_nodes,
        epoch_work_allowance::retainedHeaderCapacity(physical_epoch));
    fp::Builder core([&](auto e,auto scope)->std::optional<fp::Node> {
          if(scope==virtual_scope) {
            auto found=virtual_prefix_nodes.find(e);
            return found==virtual_prefix_nodes.end()?std::nullopt:std::optional{found->second};
          }
          return prefix_node(e,scope);
        },[&](const auto& request){return core_relate?core_relate(request):fp::RelationResult{};},
        prefix_limits,[&](auto n){return heads.charge(n);});
    struct PrefixStatisticsSnapshot {
      fp::Builder& core;fp::Statistics& stored;
      ~PrefixStatisticsSnapshot(){stored=core.statistics();}
    } prefix_statistics_snapshot{core,prefix_statistics};
    cr::Builder regions(core,region_node,
        [&](cr::Use use,std::size_t allowance) {
          const auto before=remaining;std::size_t local=allowance;BodyHeadIndex::QueryBudget guard(heads,&local);
          auto selected=heads.compact(use.expression);
          auto scope=selected?registry.insideFold(use.scope,*selected):std::nullopt;
          return cr::ScopeResult{scope,before-remaining};
        },[&](const auto& request){return region_product?region_product(request):cr::ProductResult{};},
        [&](const auto& request){return region_mask?region_mask(request):fp::RelationResult{};},
        [&](auto n){return heads.charge(n);},region_limits,
        [&](const auto& request){return region_target?region_target(request):cr::OutputTargetResult{};},
        [&](const auto& request){return region_state_target?region_state_target(request):cr::StateTargetResult{};});
    struct RegionStatisticsSnapshot {
      cr::Builder& regions;cr::Statistics& stored;bool& stopped;bool enabled;
      ~RegionStatisticsSnapshot(){if(enabled){stored=regions.statistics();stopped=regions.exhausted();}}
    } region_statistics_snapshot{regions,region_statistics,region_cap,prefix_trace};
    // This index schedules a real same-count original BODY scope. It is
    // never an equality fact: EVERY other occurrence is attached by the
    // existing complete same-assignment guard, including all captures.
    std::map<std::uint64_t,Registry::Id> canonical_sources;
    std::map<std::uint64_t,Registry::Id> canonical_carry_sources;
    std::map<std::pair<std::uint64_t,std::uint64_t>,Registry::Id> canonical_rhs_scopes;
    const auto samplePotential=[&](JoinedFamilies::Id definition,Use actual,const Registry::ScalarPoint& global_point) {
      const auto* family=potentials.view(definition);auto point=registry.globalValue(global_point);
      if(!family || !point ||
          !heads.charge(1+std::bit_width(family->boxes.size()+1)))return false;
      auto it=std::lower_bound(family->boxes.begin(),family->boxes.end(),*point,
          [](const auto& box,std::uint64_t n){return box[0].second<=n;});
      if(it==family->boxes.end() || *point<(*it)[0].first)return false;
      const auto& piece=family->pieces[it-family->boxes.begin()];
      auto context=registry.sampleContext(piece.images[0],{},actual.scope,global_point);
      const auto* assignment=context?registry.context(*context):nullptr;
      return assignment && assignment->reference_scope==piece.scope &&
          products.products.related({piece.body,piece.scope},actual,*context);
    };
    struct Terminal {
      fp::Builder::StateId state;fp::Builder::FrameId frame;
      fp::Builder::CompletedPrefixFamilyId completed;
      Registry::Id scope;std::uint32_t axis;std::uint64_t extent;
      fp::Interval time;Class expression,potential;Registry::ScalarPoint point;Class zero;
      JoinedFamilies::Id p;JoinedProducts::Id q;
      fp::ExprId virtual_p,virtual_q;
    };
    using TerminalKey=std::tuple<Class,Registry::Id,std::uint32_t,std::uint64_t,std::uint64_t>;
    std::deque<Terminal> terminals;
    std::map<TerminalKey,std::optional<std::size_t>> terminal_index;
    struct Case {
      std::uint64_t endpoint;Registry::ScalarPoint point;Class state,projection,seed;
      Registry::ScopeView body_scope;
      Registry::OccurrenceUse state_use,projection_use;
      // The ACTUAL selected Fold/member and parent remain in body_scope;
      // both original BODY operand paths remain in the two use handles.
    };
    struct SourceFrameRequest {
      std::span<const Case> cases;std::span<const Update> updates;
      Registry::Id state_scope;
      std::uint32_t actual_state_axis,template_time_axis,template_state_axis;
      std::optional<std::uint32_t> output_time_axis;
      std::uint64_t state_extent;fp::Interval time;
      Class entry_state,entry_potential;
      const Terminal* carry;
      JoinedFamilies::Id potential;JoinedProducts::Id injection;
      fp::ExprId virtual_p,virtual_q;
    };
    struct SourceFrame {
      fp::Builder::FrameRequest definition;
      fp::Builder::FrameId frame;fp::Builder::StateId terminal;
      fp::Builder::CompletedPrefixFamilyId completed;
      std::vector<fp::Builder::StateId> states;
    };
    // ONE existing exact source-frame proof. Producers keep their ORIGINAL
    // output/coordinate/path attachments; this callable does not certify a
    // producer's whole output, fabricate a projection, or admit a point.
    // Every stored update still passes the SAME core and full P/Q samplers.
    const auto certifySourceFrame=[&](const SourceFrameRequest& source,PhaseTrace& phase)
        ->std::optional<SourceFrame> {
      if(!heads.charge(16) || source.cases.empty() || source.cases.size()!=source.updates.size())return {};
      if(source.time.begin>=source.time.end ||
          source.time.end-source.time.begin!=source.cases.size())return {};
      auto endpoint=source.time.begin;
      for(const auto& c:source.cases) {
        if(!heads.charge(3) || c.endpoint!=endpoint)return {};
        auto point=registry.globalValue(c.point);if(!point || *point!=endpoint)return {};
        ++endpoint;
      }
      if(endpoint!=source.time.end)return {};
      const auto* q_family=products.view(source.injection);
      const auto* p_family=potentials.view(source.potential);
      const auto* q_geometry=q_family?products.geometry.view(q_family->geometry):nullptr;
      if(!p_family || !q_geometry || p_family->rank!=1 || q_geometry->rank!=2 ||
          p_family->type!=q_geometry->type || source.template_time_axis==source.template_state_axis ||
          p_family->domain[0]!=std::pair{source.time.begin,source.time.end} ||
          q_geometry->domain[0]!=std::pair{source.time.begin,source.time.end} ||
          q_geometry->domain[1]!=std::pair<std::uint64_t,std::uint64_t>{0,source.state_extent} ||
          registry.count(source.state_scope,source.actual_state_axis)!=std::optional{source.state_extent})return {};
      // The explicit virtual labels name the ACTUAL admitted template roles,
      // not arbitrary new free axes. Original captures retain their labels.
      for(const auto& piece:p_family->pieces) {
        if(!heads.charge(3))return {};
        auto time=registry.view(piece.images[0]);
        if(!time || time->scope!=piece.scope || time->level!=source.template_time_axis)return {};
      }
      for(const auto& piece:q_geometry->pieces) {
        if(!heads.charge(6))return {};
        auto time=registry.view(piece.images[0]),state=registry.view(piece.images[1]);
        if(!time || !state || time->scope!=piece.scope || state->scope!=piece.scope ||
            time->level!=source.template_time_axis || state->level!=source.template_state_axis)return {};
      }
      const auto virtual_node=[&](fp::ExprId expression,fp::ScopeId scope)->std::optional<fp::Node> {
        if(scope!=virtual_scope || (expression!=source.virtual_p && expression!=source.virtual_q))return {};
        fp::Node n;n.type=p_family->type;n.finite_ideal=true;n.exact_dependencies=true;
        std::vector<std::uint32_t> free{source.template_time_axis};
        if(expression==source.virtual_q)free.push_back(source.template_state_axis);
        const auto& captures=expression==source.virtual_q?q_geometry->captures:p_family->captures;
        for(auto [level,identity]:captures) { (void)identity;free.push_back(level); }
        std::sort(free.begin(),free.end());free.erase(std::unique(free.begin(),free.end()),free.end());
        if(free.size()>n.free_indices.size() || !heads.charge(free.size()+1))return {};
        std::copy(free.begin(),free.end(),n.free_indices.begin());n.free_count=free.size();return n;
      };
      auto pn=virtual_node(source.virtual_p,virtual_scope),qn=virtual_node(source.virtual_q,virtual_scope);
      if(!pn || !qn || !heads.charge(2+pn->free_count+qn->free_count))return {};
      if(virtual_prefix_nodes.contains(source.virtual_p) || virtual_prefix_nodes.contains(source.virtual_q))return {};
      virtual_prefix_nodes.emplace(source.virtual_p,*pn);virtual_prefix_nodes.emplace(source.virtual_q,*qn);
      // The callback borrows only this invocation's source span. Clear it
      // before that request/span can expire, including every failed return.
      struct ResetRelation {fp::Relate& slot;~ResetRelation(){slot={};}} reset_relation{core_relate};
      core_relate=[&](const fp::RelationRequest& request) {
        const auto before=remaining;fp::RelationResult result;
        const auto finish=[&]() {result.work=before-remaining;return result;};
        if(request.reference_scope!=virtual_scope || request.actual_scope!=source.state_scope || request.axis_count!=2 ||
            request.axes[0].reference_axis!=source.template_time_axis || request.axes[0].actual_axis || request.axes[0].extent!=1 ||
            request.axes[1].reference_axis!=source.template_state_axis || request.axes[1].actual_axis!=source.actual_state_axis ||
            request.axes[1].offset || request.axes[1].extent!=source.state_extent || request.actual>=nodes_.size())return result;
        std::size_t local=request.max_work;BodyHeadIndex::QueryBudget guard(heads,&local);
        // The factory certified an exact consecutive endpoint/owned-point
        // span once. Reuse that invariant; do not search it on each P/Q use.
        if(!heads.charge(3) || request.axes[0].offset<source.time.begin ||
            request.axes[0].offset>=source.time.end)return finish();
        const auto position=request.axes[0].offset-source.time.begin;
        if(position>=source.cases.size())return finish();
        const auto* c=&source.cases[position];
        if(c->endpoint!=request.axes[0].offset)return finish();
        Use actual{find(request.actual),source.state_scope};bool proved=false;
        if(request.reference==source.virtual_p){++p_samples;proved=samplePotential(source.potential,actual,c->point);}
        if(request.reference==source.virtual_q){++q_samples;proved=products.sample(source.injection,std::span(&actual,1),c->point,source.actual_state_axis);}
        if(proved)result.status=fp::RelationStatus::Proved;return finish();
      };
      fp::Builder::FrameRequest frame;frame.potential_template=source.virtual_p;frame.injection_template=source.virtual_q;
      frame.template_scope=virtual_scope;frame.state_scope=source.state_scope;
      frame.time=source.time;frame.template_time_axis=source.template_time_axis;
      frame.template_state_axis=source.template_state_axis;frame.actual_state_axis=source.actual_state_axis;
      frame.state_extent=source.state_extent;frame.output_time_axis=source.output_time_axis;
      frame.entry_state=source.entry_state;frame.entry_potential=source.entry_potential;
      if(source.carry)frame.carry=source.carry->state;
      phase.next(PrefixPhase::Sampling);
      stage="core_begin";auto f=core.begin(frame);if(!f)return {};
      std::optional<fp::Builder::StateId> previous;std::vector<fp::Builder::StateId> certified_states;
      if(!heads.charge(source.cases.size()))return {};
      certified_states.reserve(source.cases.size());
      stage="core_advance";
      for(std::size_t i=0;i<source.cases.size();++i) {
        const auto& c=source.cases[i];const auto& u=source.updates[i];
        auto next=core.advance(*f,previous,{c.state,u.current_p,u.injection,c.endpoint});
        if(!next)return {};previous=next;certified_states.push_back(*next);++proved_updates;
      }
      auto completed=previous?core.complete(*previous):std::nullopt;
      component=component || bool(completed);stage=completed?"completed_first_frame":"complete_unknown";
      if(!completed)return {};
      ++completed_frames;
      phase.next(PrefixPhase::Output);
      if(!heads.charge(72+5*std::bit_width(terminal_index.size()+1)))return {};
      const auto& last=source.cases.back();
      const TerminalKey terminal_key{last.state,source.state_scope,source.actual_state_axis,source.state_extent,last.endpoint};
      auto [position,inserted]=terminal_index.try_emplace(terminal_key,terminals.size());
      if(inserted)terminals.push_back({*previous,*f,*completed,source.state_scope,source.actual_state_axis,
          source.state_extent,frame.time,last.state,source.updates.back().current_p,last.point,source.cases.front().seed,
          source.potential,source.injection,source.virtual_p,source.virtual_q});
      else if(position->second && terminals[*position->second].frame!=*f)
        position->second.reset(); // Ambiguous semantic frame is not a shape-selected carry.
      constexpr auto returned_fields=(sizeof(fp::Builder::FrameRequest)+sizeof(Class)-1)/sizeof(Class)+8;
      if(!heads.charge(returned_fields))return {};
      return SourceFrame{frame,*f,*previous,*completed,std::move(certified_states)};
    };
    struct StateFunctionProof {cr::Builder::Certificate source,region;fp::Interval state;};
    // One complete Identity implementation serves actual entry restrictions
    // and direct terminal roots. Every numeric sample still uses the shared
    // typed/scoped matcher; geometry does not union a partial source function.
    using StateFunction=std::function<std::optional<StateFunctionProof>(const Terminal&,Use,Class,Class,
        std::uint32_t,std::uint32_t,Registry::ImageId,Class,Registry::ContextId,bool)>;
    if(!heads.charge(24))return;
    StateFunction proveStateFunction;
    proveStateFunction=[&](const Terminal& terminal,Use actual_use,
        Class source_coordinate,Class rhs_coordinate,std::uint32_t actual_state_axis,
        std::uint32_t actual_time_axis,Registry::ImageId component_image,
        Class dependency_witness,Registry::ContextId assignment,bool restricted)
        ->std::optional<StateFunctionProof> {
      stage="carry_state_header";
      const auto* carried=&terminal;const auto* actual=&actual_use;
      const auto canonical_scope=terminal.scope,carry_scope=actual_use.scope;
      const auto state_axis=terminal.axis;
      const auto state_coordinate=source_coordinate;
      const std::optional<std::uint64_t> state_extent=terminal.extent;
      const std::optional<Registry::ImageId> image=component_image;
      const std::optional<Registry::ContextId> component_context=assignment;
      auto view=registry.view(component_image);
      const auto* old_p=potentials.view(terminal.p);const auto* old_q=products.view(terminal.q);
      const auto* old_geometry=old_q?products.geometry.view(old_q->geometry):nullptr;
      auto old_frame=core.view(terminal.frame);
      if(!view || !old_p || !old_q || !old_geometry || !old_frame ||
          view->scope!=carry_scope || view->level!=actual_state_axis ||
          view->begin>=view->end || view->end>terminal.extent ||
          registry.count(carry_scope,actual_state_axis)!=std::optional{view->end-view->begin} ||
          !registry.context(assignment))return {};
      const auto entry_count=view->end-view->begin;
      stage="carry_state_assignment";
      const auto* assigned=registry.context(assignment);
      if(!assigned || assigned->reference_scope!=canonical_scope || assigned->actual_scope!=carry_scope ||
          assigned->axis_count!=1 || assigned->axes[0].reference_level!=state_axis ||
          assigned->axes[0].actual_level!=std::optional{actual_state_axis} ||
          assigned->axes[0].offset!=view->begin || assigned->axes[0].extent!=entry_count ||
          assigned->axes[0].actual_shift || assigned->axes[0].point ||
          assigned->axes[0].coordinate_image!=std::optional{component_image})return {};
      const fp::Interval component_state{view->begin,view->end};
      const cr::StateTarget state_target{1,1,carry_scope,component_state};
      const cr::StateTarget source_target{1,1,canonical_scope,component_state};
      struct IdentityPiece {
        const cr::Candidate* syntax;Registry::ImageId time;
        Use p;Registry::Context p_context,q_context;
        BoundedProducts::ReferenceId q;
      };
      std::vector<IdentityPiece> identity_pieces;
      using IdentityKey=std::tuple<cr::ExprId,Registry::Id,Registry::Id>;
      std::map<IdentityKey,std::size_t> identity_index;
      stage="carry_state_syntax";
      auto syntax=regions.candidates({actual->body,actual->scope},cr::CandidateMode::Identity);
      if(!syntax)return {};
      struct IdentityEntry {
        const cr::Candidate* syntax;cr::Weight weight;Use actual;
        StateFunctionProof previous;
      };
      std::optional<IdentityEntry> identity_entry;
      for(const auto& piece:*syntax) {
        if(!piece.time_fold) {
          stage="carry_state_entry";
          if(identity_entry || piece.state_fold || piece.predicate || !old_frame->carry ||
              !heads.charge(8+2*piece.factor_count))return {};
          if(!carried->time.begin || !heads.charge(8+5*std::bit_width(terminal_index.size()+1)))return {};
          auto previous_terminal=terminal_index.find({find(old_frame->entry_state),canonical_scope,
              state_axis,*state_extent,carried->time.begin-1});
          if(previous_terminal==terminal_index.end() || !previous_terminal->second)return {};
          const auto* prior=&terminals[*previous_terminal->second];
          if(prior->state!=*old_frame->carry || prior->time.end!=carried->time.begin || prior->time.end>=carried->time.end ||
              prior->scope!=canonical_scope || prior->axis!=state_axis || prior->extent!=*state_extent ||
              prior->expression!=find(old_frame->entry_state))return {};
          std::optional<cr::Weight> weight;std::optional<Use> entry;
          for(unsigned i=0;i<piece.factor_count;++i) {
            const auto factor=piece.factors[i];if(factor.expression>=nodes_.size())return {};
            if(piece.weight && factor==piece.weight->factor) {
              if(weight)return {};weight=*piece.weight;
            } else {
              const auto k=head(factor.expression);
              if(!piece.weight && k && k->type=="real" && idealOperation(*k) &&
                  k->name=="Exp" && k->children.size()==1 &&
                  find(old_frame->entry_potential)<zero_.size() && zero_[find(old_frame->entry_potential)]) {
                if(weight)return {};
                weight=cr::Weight{factor,{k->children[0],factor.scope},
                    {old_frame->entry_potential,canonical_scope}};
              } else {if(entry)return {};entry=Use{find(factor.expression),factor.scope};}
            }
          }
          if(!weight || !entry || entry->scope!=carry_scope ||
              registry.count(entry->scope,actual_state_axis)!=std::optional{entry_count})return {};
          auto prior_image=registry.imageInOriginalScope(component_image,entry->scope,entry->body);
          if(!prior_image)return {};
          // The private carry's source scope/count are unchanged. Only its
          // actual component function is restricted by this SAME assignment.
          auto previous=proveStateFunction(*prior,*entry,source_coordinate,rhs_coordinate,
              actual_state_axis,actual_time_axis,*prior_image,entry->body,assignment,restricted);
          if(!previous || previous->state!=component_state || !heads.charge(24))return {};
          identity_entry=IdentityEntry{&piece,*weight,*entry,*previous};continue;
        }
        stage="carry_state_piece_header";
        if(!piece.time_fold || piece.state_fold || !piece.weight || identity_pieces.size()>=1023)return {};
        const auto& tf=*piece.time_fold;
        if(tf.binder!=actual_time_axis || piece.weight->source_potential.scope!=tf.body_scope)return {};
        struct PotentialPiece {Use use;Registry::ImageId time;std::size_t index;};
        std::optional<PotentialPiece> p_definition;
        stage="carry_state_p_image";
        for(std::size_t i=0;i<old_p->pieces.size();++i) {
          if(!heads.charge())return {};
          const auto& reference=old_p->pieces[i];
          auto reference_time=registry.view(reference.images[0]);if(!reference_time)return {};
          auto local=registry.imageFromTypedCells({piece.weight->source_potential.scope,reference.scope},
              find(piece.weight->source_potential.expression),reference.body,tf.binder,reference_time->level);
          auto global=local?registry.globalTypedImage(*local,reference.images[0],reference.body):std::nullopt;
          if(!global)continue;
          auto range=registry.view(*global);
          if(!range || range->begin<carried->time.begin || range->end>carried->time.end)continue;
          if(p_definition)return {};p_definition=PotentialPiece{{reference.body,reference.scope},*global,i};
        }
        if(!p_definition)return {};
        const auto time=p_definition->time;auto time_view=registry.view(time);if(!time_view)return {};
        stage="carry_state_q_box";
        std::optional<std::size_t> q_index;
        for(std::size_t i=0;i<old_geometry->boxes.size();++i) {
          if(!heads.charge())return {};const auto& box=old_geometry->boxes[i];
          if(box[0].first<=time_view->begin && box[0].second>=time_view->end &&
              box[1].first<=component_state.begin && box[1].second>=component_state.end) {
            if(q_index)return {};q_index=i;
          }
        }
        if(!q_index)return {};const auto& qr=old_q->pieces[*q_index];
        if(!qr.reference_product || !qr.count)return {};
        stage="carry_state_relative_images";
        auto relative_q=registry.relativeImage(time,qr.images[0]);
        auto qv=relative_q?registry.view(*relative_q):std::nullopt;if(!qv)return {};
        // Keep the selected physical piece from the charged scan above;
        // do not perform an uncharged second walk over the family.
        const auto& pr=old_p->pieces[p_definition->index];
        auto pr_time=registry.view(pr.images[0]),qr_time=registry.view(qr.images[0]),qr_state=registry.view(qr.images[1]);
        if(!pr_time || !qr_time || !qr_state)return {};
        auto relative_p=registry.relativeImage(time,pr.images[0]);
        auto pv=relative_p?registry.view(*relative_p):std::nullopt;if(!pv)return {};
        auto state_here=registry.imageInOriginalScope(*image,tf.body_scope,dependency_witness);
        auto relative_state=state_here?registry.relativeImage(*state_here,qr.images[1]):std::nullopt;
        auto sv=relative_state?registry.view(*relative_state):std::nullopt;if(!sv)return {};
        stage="carry_state_contexts";
        Registry::Context pc{p_definition->use.scope,tf.body_scope};pc.axis_count=1;
        pc.axes[0]={pr_time->level,tf.binder,pv->begin,tf.count,pv->coordinate,false,{},*relative_p};
        // The product's admitted geometry owns the complete original
        // BODY scope. A first scalar factor may live in a shorter prefix;
        // equalReference/contextFor trims only proved unused axes per use.
        Registry::Context qc{old_geometry->pieces[*q_index].scope,tf.body_scope};qc.axis_count=2;
        qc.axes[0]={qr_time->level,tf.binder,qv->begin,tf.count,qv->coordinate,false,{},*relative_q};
        qc.axes[1]={qr_state->level,actual_state_axis,sv->begin,entry_count,sv->coordinate,false,{},*relative_state};
        constexpr auto context_fields=(sizeof(Registry::Context)+sizeof(Class)-1)/sizeof(Class);
        if(!heads.charge(3*context_fields+12+4*std::bit_width(identity_index.size()+1)))return {};
        const IdentityKey key{tf.head.expression,tf.head.scope,tf.body_scope};
        if(!identity_index.emplace(key,identity_pieces.size()).second)return {};
        identity_pieces.push_back({&piece,time,p_definition->use,pc,qc,*qr.reference_product});
      }
      // Recursive old-frame proofs finish BEFORE installing these callbacks;
      // their strictly decreasing time endpoints prohibit a certificate cycle.
      struct ClearStateCallbacks {
        cr::ProveStateTarget& target;cr::RelateProduct& product;cr::ProveMask& mask;
        ~ClearStateCallbacks(){target={};product={};mask={};}
      } clear_callbacks{region_state_target,region_product,region_mask};
      region_state_target=[&](const cr::StateTargetRequest& request) {
        const auto before=remaining;cr::StateTargetResult result;
        const auto finish=[&]() {result.work=before-remaining;return result;};
        std::size_t local=request.max_work;BodyHeadIndex::QueryBudget guard(heads,&local);
        const cr::Use source{carried->expression,canonical_scope},coord{state_coordinate,canonical_scope};
        const cr::Use narrow_coordinate{rhs_coordinate,carry_scope};
        if(!heads.charge() || request.terminal!=carried->state || request.terminal_state.frame!=carried->frame ||
            request.terminal_state.expression!=carried->expression || request.terminal_state.endpoint!=carried->time.end-1 ||
            request.source_state!=source || request.source_coordinate!=coord ||
            (request.region_root!=source && request.region_root!=cr::Use{actual->body,actual->scope}) ||
            request.frame.state_scope!=canonical_scope || request.frame.actual_state_axis!=state_axis ||
            request.frame.state_extent!=*state_extent || request.frame.time!=carried->time)return finish();
        if(restricted) {
          // Geometry and source-terminal attachment ONLY. Numeric region
          // equality is still proved by the complete causal callbacks.
          if(request.mode!=cr::StateTargetMode::PairedRestriction ||
              request.region_root!=cr::Use{actual->body,actual->scope} ||
              request.state_coordinate!=narrow_coordinate ||
              registry.count(carry_scope,actual_state_axis)!=std::optional{entry_count} ||
              !registry.context(*component_context))return finish();
          auto full_coordinate=registry.image(canonical_scope,state_axis,state_coordinate);
          auto local_coordinate=registry.image(carry_scope,actual_state_axis,rhs_coordinate);
          auto full=full_coordinate?registry.view(*full_coordinate):std::nullopt;
          auto narrow=local_coordinate?registry.view(*local_coordinate):std::nullopt;
          if(!full || !narrow || full->begin || full->end!=*state_extent ||
              narrow->begin || narrow->end!=entry_count ||
              !registry.prefix(*outer,canonical_scope) || !registry.prefix(*outer,carry_scope))return finish();
          result.source_target=source_target;
        } else if(request.mode!=cr::StateTargetMode::Identity || request.state_coordinate!=coord)return finish();
        result.status=fp::RelationStatus::Proved;result.target=state_target;return finish();
      };
      region_product=[&](const cr::ProductRequest& request) {
        const auto before=remaining;cr::ProductResult result;
        const auto finish=[&]() {result.work=before-remaining;return result;};
        std::size_t local=request.max_work;BodyHeadIndex::QueryBudget guard(heads,&local);
        if(!heads.charge() || !request.endpoint ||
            request.endpoint->frame!=carried->frame || request.endpoint->expression!=carried->expression ||
            request.endpoint->endpoint!=carried->time.end-1 || request.state_output!=state_target ||
            request.completed || request.output || request.projection || request.state_fold || !request.weight ||
            request.frame.potential_template!=carried->virtual_p || request.frame.injection_template!=carried->virtual_q ||
            request.frame.template_scope!=virtual_scope || request.frame.state_scope!=canonical_scope ||
            request.frame.entry_state!=old_frame->entry_state || request.frame.entry_potential!=old_frame->entry_potential ||
            request.frame.carry!=old_frame->carry || request.frame.time!=carried->time ||
            request.frame.state_extent!=*state_extent || request.factor_count>8)return finish();
        if(request.kind==cr::ProductKind::EntryIdentity) {
          if(request.time_fold || !identity_entry || request.factor_count!=1 ||
              request.factors[0]!=cr::Use{identity_entry->actual.body,identity_entry->actual.scope} ||
              request.weight->factor!=identity_entry->weight.factor ||
              request.weight->target_potential!=identity_entry->weight.target_potential ||
              request.weight->source_potential!=identity_entry->weight.source_potential ||
              !regions.equal(identity_entry->previous.source,identity_entry->previous.region).value_or(false) ||
              !samplePotential(carried->p,{find(request.weight->target_potential.expression),request.weight->target_potential.scope},carried->point))return finish();
          const Use expected{old_frame->entry_potential,canonical_scope};
          const Use actual_e{find(request.weight->source_potential.expression),request.weight->source_potential.scope};
          auto baseline=products.products.contextFor(expected,actual_e,assignment);
          if(!baseline || !products.products.related(expected,actual_e,*baseline))return finish();
          result.status=fp::RelationStatus::Proved;result.state=component_state;
          result.geometry_token=identity_pieces.size()+1;return finish();
        }
        if(request.kind!=cr::ProductKind::InjectionIdentity || !request.time_fold)return finish();
        const auto& tf=*request.time_fold;
        if(!heads.charge(4+4*std::bit_width(identity_index.size()+1)))return finish();
        auto found=identity_index.find({tf.head.expression,tf.head.scope,tf.body_scope});if(found==identity_index.end())return finish();
        const auto& record=identity_pieces[found->second];const auto& original=*record.syntax;
        if(tf.binder!=original.time_fold->binder || tf.count!=original.time_fold->count ||
            request.weight->factor!=original.weight->factor || request.weight->target_potential!=original.weight->target_potential ||
            request.weight->source_potential!=original.weight->source_potential ||
            !samplePotential(carried->p,{find(request.weight->target_potential.expression),request.weight->target_potential.scope},carried->point) ||
            !products.products.related(record.p,{find(request.weight->source_potential.expression),request.weight->source_potential.scope},record.p_context))return finish();
        std::array<Use,8> actual_factors{};
        if(!heads.charge(2*request.factor_count))return finish();
        for(unsigned i=0;i<request.factor_count;++i) {
          if(request.factors[i].expression>=nodes_.size())return finish();
          actual_factors[i]={find(request.factors[i].expression),request.factors[i].scope};
        }
        auto context=registry.admitContext(record.q_context);
        if(!context || !products.products.equal(record.q,std::span(actual_factors.data(),request.factor_count),*context))return finish();
        auto range=registry.view(record.time);if(!range)return finish();
        result.status=fp::RelationStatus::Proved;result.time={range->begin,range->end};result.state=component_state;
        result.geometry_token=found->second+1;return finish();
      };
      region_mask=[&](const cr::MaskRequest& request) {
        const auto before=remaining;fp::RelationResult result;
        const auto finish=[&]() {result.work=before-remaining;return result;};
        std::size_t local=request.max_work;BodyHeadIndex::QueryBudget guard(heads,&local);
        if(!heads.charge() || !request.endpoint || *request.endpoint!=carried->time.end-1 || request.state_output!=state_target ||
            !request.geometry_token || request.geometry_token>identity_pieces.size())return finish();
        const auto& record=identity_pieces[request.geometry_token-1];const auto& original=*record.syntax;
        if(!original.predicate || request.predicate!=*original.predicate || request.selected_when_true!=original.selected_when_true)return finish();
        auto k=head(request.predicate.expression);auto relation=k?comparison(*k):std::nullopt;
        if(!k || !relation || k->children.size()!=2 || !k->provenance.empty())return finish();
        const auto range=[&](Class cls)->std::optional<std::pair<std::uint64_t,std::uint64_t>> {
          const auto& n=nodes_[find(cls)];if(n.type!="u64" && n.type!="exact:u64")return {};
          auto a=heads.affine(cls);if(!a || !registry.boundedAffine(*a,request.predicate.scope))return {};
          auto end=a->constant;
          for(auto [axis,coefficient]:a->coefficients) {
            auto count=registry.count(request.predicate.scope,axis);std::uint64_t term;
            if(!count || !heads.charge() || __builtin_mul_overflow(coefficient,*count-1,&term) ||
                __builtin_add_overflow(end,term,&end))return {};
          }
          return std::pair{a->constant,end};
        };
        auto a=range(k->children[0]),b=range(k->children[1]);if(!a || !b)return finish();
        std::optional<bool> predicate;
        if(*relation=="le") {if(a->second<=b->first)predicate=true;else if(a->first>b->second)predicate=false;}
        if(*relation=="lt") {if(a->second<b->first)predicate=true;else if(a->first>=b->second)predicate=false;}
        if(*relation=="ge") {if(a->first>=b->second)predicate=true;else if(a->second<b->first)predicate=false;}
        if(*relation=="gt") {if(a->first>b->second)predicate=true;else if(a->second<=b->first)predicate=false;}
        if(predicate && *predicate==request.selected_when_true)result.status=fp::RelationStatus::Proved;return finish();
      };
      stage="carry_identity_region";
      const cr::Use src{carried->expression,canonical_scope},coord{state_coordinate,canonical_scope};
      const cr::Use actual_root{actual->body,actual->scope},actual_coordinate{rhs_coordinate,actual->scope};
      std::optional<cr::Builder::StateRestrictionId> restriction;
      if(restricted) {
        restriction=regions.stateRestriction({actual_root,actual_coordinate,src,coord,carried->state});
        if(!restriction){stage="carry_restriction_attachment_unknown";return {};}
      }
      auto source=regions.identityState({src,coord,src,coord,carried->state,restriction});
      auto region=regions.identityRegion({actual_root,actual_coordinate,src,coord,carried->state,restriction});
      if(!source || !region || !regions.equal(*source,*region).value_or(false) || !heads.charge(18))return {};
      return StateFunctionProof{*source,*region,component_state};
    };
    // Private expression IDs do not enter the graph or overlap normal
    // producers' reserved three IDs per physically stored root pair.
    fp::ExprId next_terminal_virtual=UINT64_MAX;
    if(left.size()>(UINT64_MAX-3)/3)return;
    next_terminal_virtual-=3*left.size();
    const auto proveTerminalRoot=[&](Class left_root,Class right_root) {
      PhaseTrace phase{prefix_trace,remaining,phase_work,remaining};
      entry_stage="none";coordinate_trace={};
      stage="terminal_source_header";++roots_attempted;
      const auto& free=free_indices_[find(left_root)];
      if(free.size()!=1 || !heads.charge(4))return;
      const auto state_axis=*free.begin(),time_axis=options_.outer_binders;
      const auto source_coordinate=indexCoordinate(state_axis);
      auto extent=registry.count(*outer,state_axis);
      if(source_coordinate==none || !extent || state_axis>=time_axis)return;
      auto state_image=registry.image(*outer,state_axis,source_coordinate);
      auto state_view=state_image?registry.view(*state_image):std::nullopt;
      if(!state_view || state_view->derived || state_view->begin || state_view->end!=*extent)return;

      struct StoredUpdate {Class state;Update update;};
      std::vector<StoredUpdate> backwards;std::unordered_set<Class> active;
      Class initial=find(left_root);
      while(backwards.size()<1024) {
        if(!heads.charge(24) || !active.insert(initial).second)return;
        auto selected=update(initial);
        if(!selected)break;
        backwards.push_back({initial,*selected});initial=selected->previous;
      }
      // A first injection has no P operand in its source expression. Its
      // actual P is the retained previous-P operand of the first update,
      // never a manufactured ordinal, constant or load.
      if(backwards.empty() || backwards.size()==1024 || !heads.charge(backwards.size()))return;
      std::reverse(backwards.begin(),backwards.end());
      const auto anchor=backwards.front().update.previous_p;
      if(!prefix_node(initial,*outer) || !prefix_node(anchor,*outer))return;

      struct TemplatePiece {const cr::Candidate* syntax;Registry::ImageId time;JoinedProducts::Piece q;};
      std::vector<TemplatePiece> templates;
      std::vector<Use> syntax_roots{{find(right_root),*outer}};
      std::set<std::pair<Class,Registry::Id>> syntax_active;
      std::optional<Registry::ImageId> point_role;Class zero_seed=none;
      while(!syntax_roots.empty()) {
        auto root=syntax_roots.back();syntax_roots.pop_back();
        if(syntax_active.size()>=256 || !heads.charge(16+3*std::bit_width(syntax_active.size()+1)) ||
            !syntax_active.emplace(root.body,root.scope).second)return;
        const auto syntax=regions.candidates({root.body,root.scope},cr::CandidateMode::Identity);
        if(!syntax)return;
        for(const auto& piece:*syntax) {
          if(!heads.charge(12+2*piece.factor_count))return;
          if(!piece.time_fold) {
            // Untrusted parser hints only. The complete Identity proof
            // later checks the actual weight/E and prior private carry.
            std::optional<Use> old;unsigned weights=0;
            for(unsigned i=0;i<piece.factor_count;++i) {
              const auto use=piece.factors[i];auto k=head(use.expression);
              if(!k)return;
              if((piece.weight && use==piece.weight->factor) ||
                  (!piece.weight && k->type=="real" && idealOperation(*k) && k->name=="Exp" && k->children.size()==1))++weights;
              else {if(old)return;old=Use{find(use.expression),use.scope};}
            }
            if(piece.state_fold || piece.predicate || weights!=1 || !old || old->scope!=*outer)return;
            syntax_roots.push_back(*old);continue;
          }
          const auto& tf=*piece.time_fold;
          if(piece.state_fold || !piece.weight || tf.binder!=time_axis || templates.size()>=1023 ||
              piece.weight->source_potential.scope!=tf.body_scope)return;
          auto selected=head(tf.head.expression);
          if(!selected || selected->kind!=static_cast<int>(s::Kind::Fold) || !selected->fold ||
              selected->fold->reducer!=s::Reducer::Sum || selected->children.size()!=2 ||
              !selected->attributes.empty() || !selected->provenance.empty())return;
          // The SAME parser retains every actual outside-binder seed as
          // another contribution. A later seeded tile may carry an earlier
          // tile; it is not silently zeroed by this template hint.
          const auto seed=find(selected->children[0]);
          if(zero_seed==none && zero_[seed] && finite_[seed])zero_seed=seed;
          Registry::Context known{tf.body_scope,*outer};known.axis_count=1;
          known.axes[0]={state_axis,state_axis,0,*extent,source_coordinate,false};
          auto image=registry.typedTimeImage(known,find(piece.weight->source_potential.expression),anchor,time_axis);
          auto range=image?registry.view(*image):std::nullopt;
          auto state=registry.image(tf.body_scope,state_axis,source_coordinate);
          if(!range || !state)return;
          if(!range->begin && !point_role)point_role=*image;
          JoinedProducts::Piece q;q.images={*image,*state};
          for(unsigned i=0;i<piece.factor_count;++i)if(piece.factors[i]!=piece.weight->factor) {
            if(q.count==q.factors.size())return;
            q.factors[q.count++]={find(piece.factors[i].expression),piece.factors[i].scope};
          }
          constexpr auto template_fields=(sizeof(TemplatePiece)+sizeof(Class)-1)/sizeof(Class);
          // Vector growth is geometric; precharge its bounded fixed record
          // allocation/copies, not just the number of active factors.
          if(!q.count || !heads.charge(3*template_fields+2*q.count))return;
          templates.push_back({&piece,*image,q});
        }
      }
      if(!point_role || zero_seed==none || templates.empty())return;
      auto first_point=registry.typedPoint(*point_role,anchor,*outer);
      auto first_value=first_point?registry.globalValue(Registry::ScalarPoint{*first_point}):std::nullopt;
      if(!first_value || *first_value)return;
      constexpr auto case_fields=(sizeof(Case)+sizeof(Class)-1)/sizeof(Class);
      if(!heads.charge((backwards.size()+1)*(case_fields+8)))return;
      std::vector<Case> cases;std::vector<Update> updates;std::vector<std::size_t> begins{0};
      cases.reserve(backwards.size()+1);updates.reserve(backwards.size()+1);begins.reserve(backwards.size()+1);
      cases.push_back({0,Registry::ScalarPoint{*first_point},initial,none,zero_seed,{},{},{}});
      updates.push_back({zero_seed,anchor,zero_seed,initial});
      for(const auto& stored:backwards) {
        auto point=registry.typedPoint(*point_role,stored.update.current_p,*outer);
        auto value=point?registry.globalValue(Registry::ScalarPoint{*point}):std::nullopt;
        if(!value || cases.back().endpoint==UINT64_MAX || *value!=cases.back().endpoint+1 ||
            stored.update.previous!=cases.back().state)return;
        auto prior=registry.typedPoint(*point_role,stored.update.previous_p,*outer);
        auto prior_value=prior?registry.globalValue(Registry::ScalarPoint{*prior}):std::nullopt;
        if(!prior_value || *prior_value!=cases.back().endpoint)begins.push_back(cases.size());
        cases.push_back({*value,Registry::ScalarPoint{*point},stored.state,none,zero_seed,{},{},{}});
        updates.push_back(stored.update);++source_cases;
      }
      begins.push_back(cases.size());
      const Terminal* carried=nullptr;
      for(std::size_t frame_index=0;frame_index+1<begins.size();++frame_index) {
        const auto begin=begins[frame_index],end=begins[frame_index+1];
        if(begin>=end || cases[end-1].endpoint==UINT64_MAX)return;
        const fp::Interval time{cases[begin].endpoint,cases[end-1].endpoint+1};
        std::vector<JoinedFamilies::Piece> ps;std::vector<JoinedProducts::Piece> qs;
        constexpr auto family_fields=(sizeof(JoinedFamilies::Piece)+sizeof(JoinedProducts::Piece)+sizeof(Class)-1)/sizeof(Class);
        if(!heads.charge(templates.size()*2*family_fields))return;
        ps.reserve(templates.size());qs.reserve(templates.size());
        for(const auto& piece:templates) {
          if(!heads.charge())return;auto range=registry.view(piece.time);if(!range)return;
          if(range->begin<time.begin || range->end>time.end)continue;
          const auto& p=*piece.syntax->weight;
          ps.push_back({find(p.source_potential.expression),p.source_potential.scope,{piece.time,{}}});qs.push_back(piece.q);
        }
        JoinedFamilies::Box domain{{{time.begin,time.end},{0,*extent}}};
        auto p=potentials.join(ps,1,domain);auto q=products.join(qs,domain);
        if(!p || !q || next_terminal_virtual<2 || next_terminal_virtual-2<=nodes_.size() || !heads.charge(8))return;
        const auto virtual_p=next_terminal_virtual--,virtual_q=next_terminal_virtual--;
        const Class entry=begin?updates[begin].previous:zero_seed;
        const Class baseline=begin?updates[begin].previous_p:zero_seed;
        SourceFrameRequest request{std::span(cases).subspan(begin,end-begin),std::span(updates).subspan(begin,end-begin),
            *outer,state_axis,time_axis,state_axis,{},*extent,time,entry,baseline,carried,*p,*q,virtual_p,virtual_q};
        auto certified=certifySourceFrame(request,phase);if(!certified)return;
        const TerminalKey key{cases[end-1].state,*outer,state_axis,*extent,cases[end-1].endpoint};
        if(!heads.charge(12+5*std::bit_width(terminal_index.size()+1)))return;
        auto position=terminal_index.find(key);
        if(position==terminal_index.end() || !position->second)return;
        carried=&terminals[*position->second];
        if(carried->state!=certified->terminal || carried->frame!=certified->frame)return;
      }
      if(!carried || carried->expression!=find(left_root))return;
      Registry::Context state_assignment{*outer,*outer};state_assignment.axis_count=1;
      state_assignment.axes[0]={state_axis,state_axis,0,*extent,source_coordinate,false,{},*state_image};
      auto assignment=registry.admitContext(state_assignment);if(!assignment)return;
      phase.next(PrefixPhase::Region);stage="terminal_identity_region";
      auto identity=proveStateFunction(*carried,{find(right_root),*outer},source_coordinate,source_coordinate,
          state_axis,time_axis,*state_image,right_root,*assignment,false);
      if(!identity || identity->state!=fp::Interval{0,*extent} || !heads.charge(2))return;
      proved_roots.emplace_back(left_root,right_root);output_proved=true;stage="terminal_output_proved";
    };
    const auto proveRoot=[&](Class left_root,Class right_root,std::size_t ordinal) {
    if(ordinal>(UINT64_MAX-3)/3 || !heads.charge())return;
    PhaseTrace phase{prefix_trace,remaining,phase_work,remaining};
    entry_stage="none";
    coordinate_trace={};
    const fp::ExprId virtual_p=UINT64_MAX-3*ordinal;
    const fp::ExprId virtual_q=virtual_p-1,virtual_c=virtual_p-2;
    ++roots_attempted;
    std::vector<Case> cases;
    std::optional<Registry::Id> canonical_source;
    TypedDispatch dispatch_cursor(registry,potentials,*outer,left_root);
    Class coordinate=none;std::optional<Registry::ImageId> source_image;
    stage="source_dispatch";
    const auto appendCase=[&](std::uint64_t endpoint,const Registry::ScalarPoint& point,Class branch) {
        if(std::holds_alternative<Registry::DispatchPointId>(point) &&
            !heads.charge(2*((sizeof(Registry::DispatchPointId)+sizeof(Class)-1)/sizeof(Class))))return false;
        auto sum_id=heads.compact(branch);
        if(!sum_id)return false;const auto& sum=nodes_[*sum_id];
        if(sum.kind!=static_cast<int>(s::Kind::Fold) || !sum.fold || sum.fold->reducer!=s::Reducer::Sum ||
            sum.fold->binder_level!=options_.outer_binders || !sum.fold->count ||
            sum.children.size()!=2 || !sum.attributes.empty() || !sum.provenance.empty() ||
            !zero_[find(sum.children[0])])return false;
        auto original=registry.bodyView(*outer,*sum_id);if(!original)return false;
        if(!canonical_source) {
          if(!heads.charge(2+std::bit_width(canonical_sources.size()+1)))return false;
          if(auto found=canonical_sources.find(sum.fold->count);found!=canonical_sources.end())
            canonical_source=found->second;
          else {
            canonical_source=registry.insideFold(*outer,*sum_id);if(!canonical_source)return false;
            canonical_sources.emplace(sum.fold->count,*canonical_source);
          }
        }
        if(!canonical_source)return false;
        auto product=head(sum.children[1]);if(!product || !arithmetic(*product) || product->name!="Multiply")return false;
        std::optional<std::pair<Class,Class>> split;unsigned projection_side=0;
        for(unsigned side=0;side<2;++side) {
          auto c=head(product->children[side]);
          if(c && c->kind==static_cast<int>(s::Kind::InitialCell)) {
            if(split)return false;split=std::pair{find(product->children[1-side]),find(product->children[side])};
            projection_side=side;
          }
        }
        if(!split)return false;
        const std::array<unsigned,1> sp{1-projection_side},cp{projection_side};
        auto state_use=registry.bodyUse(*original,sp),projection_use=registry.bodyUse(*original,cp);
        if(!state_use || !projection_use ||
            registry.transport(*state_use,*canonical_source,sum.fold->binder_level)!=split->first ||
            registry.transport(*projection_use,*canonical_source,sum.fold->binder_level)!=split->second)return false;
        if(!heads.charge(18))return false;
        cases.push_back({endpoint,point,split->first,split->second,find(sum.children[0]),
            *original,*state_use,*projection_use});++source_cases;return true;
    };
    while(cases.size()<512) {
      auto next=dispatch_cursor.next();if(!next) {
        stage=potentials.intervalExhausted()?"source_dispatch_interval_cap":"source_dispatch_unknown";return;
      }
      coordinate=dispatch_cursor.coordinate;source_image=dispatch_cursor.image;
      if(next->inside && !appendCase(next->endpoint,next->point,next->branch))return;
      if(next->done)break;
      auto domain=source_image?registry.view(*source_image):std::nullopt;if(!domain)return;
      if(cases.size()==domain->end-domain->begin)break; // Exact coverage checked below before use.
    }
    if(cases.size()<2 || !source_image)return;
    if(!heads.charge(cases.size()*(1+std::bit_width(cases.size()+1))))return;
    std::sort(cases.begin(),cases.end(),[](const auto& a,const auto& b){return a.endpoint<b.endpoint;});
    auto source_domain=registry.view(*source_image);if(!source_domain)return;
    std::vector<std::pair<std::uint64_t,std::uint64_t>> intervals;
    for(const auto& c:cases) {if(c.endpoint==UINT64_MAX)return;intervals.emplace_back(c.endpoint,c.endpoint+1);}
    if(!potentials.cover(intervals,{source_domain->begin,source_domain->end}))return;
    if(!canonical_source)return;
    const auto canonical_scope=*canonical_source;
    const auto state_axis=options_.outer_binders;
    auto state_extent=registry.count(canonical_scope,state_axis);if(!state_extent)return;
    // Attach only ACTUAL reachable source BODY uses to this canonical state
    // scope. The shared full mapper certifies identical index assignment;
    // it does not traverse or infer a recurrence from shared P/Q headers.
    for(const auto& c:cases) {
      if(registry.count(c.body_scope,state_axis)!=state_extent)return;
      // Both ACTUAL original operand attachments were certified once during
      // Case construction by the SAME identity-assignment factory above.
      ++attached;
    }
    const Terminal* carried=nullptr;
    Class entry_state=cases[0].seed,entry_potential=cases[0].seed;
    std::vector<Update> updates;updates.reserve(cases.size());
    if(!source_domain->begin) {
      auto first=update(cases[1].state);
      if(!first || first->previous!=cases[0].state){stage="initial_update_unknown";return;}
      updates.push_back({entry_state,first->previous_p,entry_potential,cases[0].state});
    } else {
      auto first=update(cases[0].state,cases[0].seed);
      if(!first || !heads.charge(8+5*std::bit_width(terminal_index.size()+1)))return;
      auto found=terminal_index.find({first->previous,canonical_scope,state_axis,*state_extent,source_domain->begin-1});
      if(found==terminal_index.end() || !found->second){stage="carry_attachment_unknown";return;}
      carried=&terminals[*found->second];
      if(carried->time.end!=source_domain->begin || carried->expression!=first->previous ||
          carried->scope!=canonical_scope || carried->axis!=state_axis || carried->extent!=*state_extent)return;
      entry_state=first->previous;entry_potential=first->previous_p;updates.push_back(*first);
    }
    for(std::size_t i=1;i<cases.size();++i) {
      auto u=update(cases[i].state);
      if(!u || u->previous!=cases[i-1].state || u->previous_p!=updates.back().current_p)return;
      updates.push_back(*u);
    }
    stage="actual_region_candidates";
    phase.next(PrefixPhase::Templates);
    auto candidates=regions.candidates({find(right_root),*outer});if(!candidates)return;
    std::vector<JoinedProducts::Piece> q_pieces;std::vector<JoinedFamilies::Piece> p_pieces;
    // Original uses and the SAME full relation contexts are retained until
    // this root's complete dispatch/region attachment has been certified.
    std::deque<BoundedProducts::AssignedUse> assignment_attachments;
    struct ProductRecord {
      const cr::Candidate* syntax;
      Use projection;
      Registry::ImageId time,state;
    };
    using FoldKey=std::tuple<cr::ExprId,Registry::Id,cr::ExprId,Registry::Id>;
    std::map<FoldKey,std::size_t> product_records;
    std::vector<ProductRecord> records;
    std::vector<JoinedProducts::Piece> c_pieces;
    std::optional<std::pair<std::uint64_t,std::uint64_t>> projection_band;
    Class output_coordinate=none;
    const auto output_axis=source_domain->level;
    std::optional<Registry::CommonTranslationId> common_time;
    const auto globalTimeImage=[&](Registry::Id scope,std::uint32_t axis,Class coordinate)
        ->std::optional<Registry::ImageId> {
      auto local=registry.image(scope,axis,coordinate);
      return local && common_time?registry.commonImage(*common_time,*local):std::nullopt;
    };
    std::unordered_set<Class> potential_folds;
    const auto same_cell_header=[](const Key& a,const Key& b) {
      return a.kind==static_cast<int>(s::Kind::InitialCell) && b.kind==a.kind && a.name==b.name &&
          a.type==b.type && a.attributes==b.attributes && a.provenance==b.provenance && a.children.size()==1 && b.children.size()==1;
    };
    const auto* projection_head=head(cases.front().projection);if(!projection_head)return;
    stage="checked_joined_templates";
    for(const auto& candidate:*candidates) {
      // The SAME parser also retains the projected entry contribution.
      // It is not a P/Q injection-template piece: preserve it unchanged for
      // the subsequent whole-region certificate, which must validate the
      // actual entry/carry and baseline. This component admits no equality.
      if(!candidate.time_fold)continue;
      if(prefix_trace) {
        ++coordinate_trace.candidate;
        coordinate_trace.source_begin=source_domain->begin;
        coordinate_trace.source_end=source_domain->end;
        coordinate_trace.binder=candidate.time_fold->binder;
      }
      if(q_pieces.size()>=1023)return; // Existing joined-family admission cap.
      candidate_flags=bool(candidate.time_fold)+2*bool(candidate.state_fold)+
          4*bool(candidate.weight)+8*bool(candidate.predicate);
      candidate_factors=candidate.factor_count;
      stage="candidate_headers";
      if(!candidate.time_fold || !candidate.state_fold || !candidate.weight || !candidate.predicate ||
          candidate.time_fold->binder!=state_axis || candidate.state_fold->binder!=state_axis+1)return;
      if(nested_state_coordinate==none)return;
      const auto original_piece_scope=candidate.state_fold->body_scope;
      const auto scope_key=std::pair{candidate.time_fold->count,candidate.state_fold->count};
      if(!heads.charge(3+2*std::bit_width(canonical_rhs_scopes.size()+1)))return;
      auto position=canonical_rhs_scopes.find(scope_key);
      const auto piece_scope=position==canonical_rhs_scopes.end()?original_piece_scope:position->second;
      if(prefix_trace) {
        coordinate_trace.original_scope=original_piece_scope;
        coordinate_trace.piece_scope=piece_scope;
      }
      if(position==canonical_rhs_scopes.end())canonical_rhs_scopes.emplace(scope_key,piece_scope);
      Registry::Context original_assignment{piece_scope,original_piece_scope};original_assignment.axis_count=2;
      original_assignment.axes[0]={state_axis,state_axis,0,scope_key.first,state_coordinate,false};
      original_assignment.axes[1]={state_axis+1,state_axis+1,0,scope_key.second,nested_state_coordinate,false};
      auto assignment=registry.admitContext(original_assignment);if(!assignment)return;
      const auto assigned=[&](Use original,bool coordinate_use=false)->std::optional<Use> {
        auto proof=products.products.canonicalUse(original,*assignment,coordinate_use);
        auto value=proof?products.products.assigned(*proof):std::nullopt;
        if(!value || !heads.charge(10))return {};
        assignment_attachments.push_back(*proof);return value;
      };
      auto predicate=head(candidate.predicate->expression);if(!predicate || !comparison(*predicate))return;
      Class time_coordinate=none;
      stage="candidate_time_coordinate";
      if(prefix_trace) {
        coordinate_trace.predicate=find(candidate.predicate->expression);
        coordinate_trace.time=none;coordinate_trace.output=output_coordinate;
        coordinate_trace.children={none,none};coordinate_trace.constants={};
        coordinate_trace.coefficients={};coordinate_trace.axes={UINT32_MAX,UINT32_MAX};
        coordinate_trace.terms={};coordinate_trace.time_begin=coordinate_trace.time_end=0;
      }
      unsigned predicate_child=0;
      for(auto child:predicate->children) {
        auto a=heads.affine(child);
        if(prefix_trace && predicate_child<2) {
          coordinate_trace.children[predicate_child]=find(child);
          if(a) {
            coordinate_trace.constants[predicate_child]=a->constant;
            coordinate_trace.terms[predicate_child]=a->coefficients.size();
            if(!a->coefficients.empty()) {
              coordinate_trace.axes[predicate_child]=a->coefficients.begin()->first;
              coordinate_trace.coefficients[predicate_child]=a->coefficients.begin()->second;
            }
          } else coordinate_trace.terms[predicate_child]=SIZE_MAX;
        }
        ++predicate_child;
        if(a && a->coefficients.size()==1 && a->coefficients.begin()->first==candidate.time_fold->binder &&
            a->coefficients.begin()->second==1) {
          if(time_coordinate!=none){stage="candidate_time_duplicate";return;}time_coordinate=find(child);
          if(prefix_trace)coordinate_trace.time=time_coordinate;
        }
        if(a && a->coefficients.size()==1 && a->coefficients.begin()->first==output_axis &&
            a->coefficients.begin()->second==1) {
          if(output_coordinate!=none && output_coordinate!=find(child)){stage="candidate_output_conflict";return;}
          output_coordinate=find(child);
          if(prefix_trace)coordinate_trace.output=output_coordinate;
        }
      }
      if(time_coordinate==none || output_coordinate==none){stage="candidate_time_axes_missing";return;}
      if(!common_time) {
        // The source selector and RHS mask are actual unit coordinates of
        // the SAME outer assignment. Their origins may differ; this private
        // geometric token never asserts their raw graph values are equal.
        auto actual_output=registry.image(*outer,output_axis,output_coordinate);
        if(!actual_output)return;
        common_time=registry.commonTranslation(*source_image,*actual_output);if(!common_time)return;
      }
      stage="candidate_time_transport";
      auto actual_time=assigned({time_coordinate,original_piece_scope},true);if(!actual_time || actual_time->scope!=piece_scope)return;
      time_coordinate=actual_time->body;
      if(prefix_trace)coordinate_trace.time=time_coordinate;
      stage="candidate_time_image";
      auto time=globalTimeImage(piece_scope,candidate.time_fold->binder,time_coordinate);if(!time)return;
      auto time_view=registry.view(*time);if(!time_view)return;
      if(prefix_trace){coordinate_trace.time_begin=time_view->begin;coordinate_trace.time_end=time_view->end;}
      stage="candidate_time_anchor";
      auto anchor=std::lower_bound(cases.begin(),cases.end(),time_view->begin,
          [](const auto& c,std::uint64_t p){return c.endpoint<p;});
      if(anchor==cases.end() || anchor->endpoint!=time_view->begin)return;
      const auto index=anchor-cases.begin();
      Use injection{updates[index].injection,canonical_scope};
      stage="source_injection_factors";
      auto source_factors=products.products.factors(std::span(&injection,1));if(!source_factors)return;
      source_factors_count=source_factors->size();
      std::optional<Use> source_cell;
      for(auto use:*source_factors) {
        auto k=head(use.body);
        if(k && k->kind==static_cast<int>(s::Kind::InitialCell) && heads.depends(use.body,state_axis)) {
          if(source_cell)return;source_cell=use;
        }
      }
      if(!source_cell)return;
      const auto* source_cell_head=head(source_cell->body);if(!source_cell_head)return;
      JoinedProducts::Piece piece;std::optional<Use> piece_cell,piece_projection,original_projection;
      unsigned projection_count=0,weight_count=0;
      stage="candidate_factor_selection";
      for(unsigned j=0;j<candidate.factor_count;++j) {
        const auto use=candidate.factors[j];auto k=head(use.expression);if(!k)return;
        if(find(use.expression)==find(candidate.weight->factor.expression)) {++weight_count;continue;}
        auto value=assigned({find(use.expression),use.scope});if(!value)return;
        if(same_cell_header(*k,*projection_head)) {
          ++projection_count;piece_projection=*value;original_projection=Use{find(use.expression),use.scope};continue;
        }
        if(piece.count==piece.factors.size())return;
        piece.factors[piece.count++]=*value;
        if(same_cell_header(*k,*source_cell_head) && heads.depends(use.expression,candidate.state_fold->binder)) {
          if(piece_cell)return;piece_cell=*value;
        }
      }
      selected_projection_count=projection_count;selected_weight_count=weight_count;
      if(projection_count!=1 || weight_count!=1 || !piece_cell || !piece_projection ||
          piece_cell->scope!=piece_scope || piece_projection->scope!=piece_scope)return;
      auto point=registry.point(*time,anchor->point,canonical_scope);if(!point)return;
      Registry::Context known{piece_scope,canonical_scope};known.axis_count=1;
      stage="candidate_state_image";
      auto witness=registry.pointWitness(anchor->point);if(!witness)return;
      known.axes[0]={candidate.time_fold->binder,{},anchor->endpoint-time_view->begin,1,*witness,false,*point};
      auto state=registry.imageFromTypedCells(known,piece_cell->body,source_cell->body,
          candidate.state_fold->binder,state_axis);if(!state)return;
      piece.images={*time,*state};q_pieces.push_back(piece);++products_count;
      const auto& tf=*candidate.time_fold;const auto& sf=*candidate.state_fold;
      const FoldKey fold_key{tf.head.expression,tf.head.scope,sf.head.expression,sf.head.scope};
      if(!heads.charge(4+std::bit_width(product_records.size()+1)) ||
          !product_records.emplace(fold_key,records.size()).second)return;
      // Final root products still consume the ORIGINAL parser occurrence.
      // Its canonical value is only a proved definition for sampling.
      if(!original_projection)return;
      records.push_back({&candidate,*original_projection,*time,*state});
      // One actual time band suffices to define C(T,s), but EVERY later use
      // is still mapped to this definition before any region certificate.
      // Its output-time image is read in the original BODY scope; no scope
      // is widened and the inner U binder is never silently captured.
      const auto band=std::pair{time_view->begin,time_view->end};
      if(!projection_band)projection_band=band;
      if(*projection_band==band) {
        stage="projection_state_image";
        auto ct=globalTimeImage(piece_scope,output_axis,output_coordinate);if(!ct)return;
        auto cp=registry.point(*ct,cases.front().point,canonical_scope);if(!cp)return;
        Registry::Context ck{piece_scope,canonical_scope};ck.axis_count=1;
        auto witness=registry.pointWitness(cases.front().point);if(!witness)return;
        ck.axes[0]={output_axis,{},0,1,*witness,false,*cp};
        auto cs=registry.imageFromTypedCells(ck,piece_projection->body,cases.front().projection,
            sf.binder,state_axis);if(!cs)return;
        JoinedProducts::Piece c;c.count=1;c.factors[0]=*piece_projection;c.images={*ct,*cs};c_pieces.push_back(c);
      }
      if(potential_folds.insert(find(candidate.time_fold->head.expression)).second) {
        const auto use=candidate.weight->source_potential;
        auto value=assigned({find(use.expression),use.scope});if(!value)return;
        auto image=globalTimeImage(value->scope,candidate.time_fold->binder,time_coordinate);if(!image)return;
        p_pieces.push_back({value->body,value->scope,{*image,{}}});
      }
    }
    JoinedFamilies::Box domain{{{source_domain->begin,source_domain->end},{0,*state_extent}}};
    auto q=products.join(q_pieces,domain);
    auto p=potentials.join(p_pieces,1,domain);
    stage="joined_projection";
    auto c=products.join(c_pieces,domain);
    if(!q || !p || !c || output_coordinate==none || nested_state_coordinate==none)return;
    auto time_role=registry.view(*source_image);if(!time_role)return;
    SourceFrameRequest frame_source{cases,updates,canonical_scope,state_axis,state_axis,state_axis+1,
        time_role->level,*state_extent,{source_domain->begin,source_domain->end},
        entry_state,entry_potential,carried,*p,*q,virtual_p,virtual_q};
    auto certified=certifySourceFrame(frame_source,phase);if(!certified)return;
    const auto& frame=certified->definition;
    const auto f=std::optional{certified->frame};
    const auto completed=std::optional{certified->completed};
    const auto& certified_states=certified->states;
    const auto* c_family=products.view(*c);
    const auto* c_geometry=c_family?products.geometry.view(c_family->geometry):nullptr;
    if(!c_family || !c_geometry)return;
    // An owner-bound C descriptor, NEVER a graph expression at a widened
    // lexical scope. Only region_node resolves this private virtual use.
    cr::Node cn;cn.type=c_geometry->type;cn.finite_ideal=true;cn.exact_dependencies=true;
    std::vector<std::uint32_t> cfree{output_axis,state_axis+1};
    for(auto [level,identity]:c_geometry->captures){(void)identity;cfree.push_back(level);}
    if(!heads.charge(cfree.size()*(1+std::bit_width(cfree.size()+1))))return;
    std::sort(cfree.begin(),cfree.end());cfree.erase(std::unique(cfree.begin(),cfree.end()),cfree.end());
    if(cfree.size()>cn.free_indices.size())return;
    std::copy(cfree.begin(),cfree.end(),cn.free_indices.begin());cn.free_count=cfree.size();
    if(!heads.charge(cfree.size()+1) || joined_projection_nodes.contains(virtual_c))return;
    joined_projection_nodes.emplace(virtual_c,cn);
    const cr::ProjectionRequest projection_definition{{virtual_c,virtual_scope},output_axis,state_axis+1,*state_extent};
    stage="output_projection";auto projection=regions.projection(projection_definition);if(!projection)return;

    // Exact output-coordinate equality is independent of P being injective.
    // A constant potential never proves an output time assignment.
    auto output_image=globalTimeImage(*outer,output_axis,output_coordinate);if(!output_image || !common_time)return;
    auto output_domain=registry.view(*output_image);if(!output_domain ||
        output_domain->begin!=source_domain->begin || output_domain->end!=source_domain->end)return;
    const auto localOutput=[&](Registry::Id scope)->std::optional<Registry::ImageId> {
      auto actual=globalTimeImage(scope,output_axis,output_coordinate);
      return actual?registry.relativeImage(*actual,*output_image):std::nullopt;
    };
    auto output_assignment=localOutput(*outer);if(!output_assignment)return;
    Registry::Context output_context{*outer,*outer};output_context.axis_count=1;
    output_context.axes[0]={output_axis,output_axis,0,source_domain->end-source_domain->begin,
        output_coordinate,false,{},*output_assignment};
    if(!registry.commonOutputAssignment(*common_time,coordinate,output_coordinate,*outer))return;
    const cr::OutputTarget checked_output{1,1,*outer,{output_domain->begin,output_domain->end}};
    bool dispatch_attached=false;
    region_target=[&](const cr::OutputTargetRequest& request) {
      ++target_calls;
      const auto before=remaining;cr::OutputTargetResult result;
      const auto finish=[&]() {result.work=before-remaining;return result;};
      std::size_t local=request.max_work;BodyHeadIndex::QueryBudget guard(heads,&local);
      if(!heads.charge() || request.completed!=*completed || request.completed_prefix.frame!=*f ||
          request.source_dispatch!=cr::Use{find(left_root),*outer} ||
          request.source_coordinate!=cr::Use{coordinate,*outer} ||
          request.projection!=projection_definition ||
          (request.region_root!=request.source_dispatch && request.region_root!=cr::Use{find(right_root),*outer}) ||
          request.output_coordinate.scope!=*outer)return finish();
      // Source/source or actual/actual coordinate attachment stays a full
      // scoped proof. The DIFFERENT raw origins are related only by the
      // owned common assignment token above, never by an e-class equality.
      const auto requested_coordinate=request.region_root==request.source_dispatch?coordinate:output_coordinate;
      if(request.output_coordinate.expression>=nodes_.size() ||
          !registry.commonOutputAssignment(*common_time,coordinate,output_coordinate,*outer) ||
          !products.products.related({requested_coordinate,*outer},
            {find(request.output_coordinate.expression),*outer},output_context))return finish();
      if(!dispatch_attached) {
        // The ACTUAL selector path, all exact case predicates and complete
        // domain coverage were checked once above. Here attach EVERY branch
        // to its private StateId and the same complete C definition, retaining
        // its actual state Sum count/zero seed and original BODY occurrence.
        for(std::size_t i=0;i<cases.size();++i) {
          if(!heads.charge())return finish();auto state=core.view(certified_states[i]);
          if(!state || state->frame!=*f || state->expression!=cases[i].state || state->endpoint!=cases[i].endpoint ||
              !zero_[cases[i].seed] || registry.count(cases[i].body_scope,state_axis)!=state_extent)return finish();
          // The retained original projection_use attachment, including its
          // actual Fold/BODY path, authorizes this canonical value use.
          Use cp{cases[i].projection,canonical_scope};
          if(!products.sample(*c,std::span(&cp,1),cases[i].point,state_axis))return finish();
        }
        dispatch_attached=true;
      }
      result.status=fp::RelationStatus::Proved;result.target=checked_output;return finish();
    };

    std::map<std::pair<Class,Registry::Id>,bool> target_potential_proofs,projection_proofs;
    const auto checked_c=[&](const ProductRecord& record) {
      const auto key=std::pair{record.projection.body,record.projection.scope};
      if(!heads.charge(1+std::bit_width(projection_proofs.size()+1)))return false;
      if(auto it=projection_proofs.find(key);it!=projection_proofs.end())return it->second;
      auto state=registry.view(record.state);if(!state)return false;
      if(!heads.charge(1+std::bit_width(c_geometry->boxes.size()+1)))return false;
      auto it=std::lower_bound(c_geometry->boxes.begin(),c_geometry->boxes.end(),state->begin,
          [](const auto& box,std::uint64_t point){return box[1].second<=point;});
      if(it==c_geometry->boxes.end() || (*it)[1]!=std::pair{state->begin,state->end})return false;
      const auto& piece=c_family->pieces[it-c_geometry->boxes.begin()];
      if(piece.count!=1)return false;
      auto local=localOutput(record.projection.scope);if(!local)return false;
      Registry::Context context{piece.factors[0].scope,record.projection.scope};context.axis_count=2;
      context.axes[0]={output_axis,output_axis,0,output_domain->end-output_domain->begin,
          output_coordinate,false,{},*local};
      context.axes[1]={state->level,state->level,0,state->end-state->begin,nested_state_coordinate,false};
      const bool proved=products.products.related(piece.factors[0],record.projection,context);
      if(!heads.exhausted && remaining)projection_proofs.emplace(key,proved);return proved;
    };
    const auto checked_target_p=[&](cr::Use target) {
      if(target.expression>=nodes_.size())return false;
      const auto key=std::pair{find(target.expression),target.scope};
      if(!heads.charge(1+std::bit_width(target_potential_proofs.size()+1)))return false;
      if(!target_potential_proofs.contains(key)) {
        Use full{key.first,*outer},original{key.first,target.scope};
        auto local=localOutput(original.scope);if(!local)return false;
        Registry::Context lift{*outer,original.scope};lift.axis_count=1;
        lift.axes[0]={output_axis,output_axis,0,output_domain->end-output_domain->begin,
            output_coordinate,false,{},*local};
        const std::array<std::uint32_t,1> axis{output_axis};
        const std::array<Registry::ImageId,1> images{*output_image};
        const bool proved=products.products.related(full,original,lift) && potentials.uniform(*p,full.body,full.scope,axis,images);
        if(heads.exhausted || !remaining)return false;target_potential_proofs.emplace(key,proved);
      }
      return target_potential_proofs.at(key);
    };
    const auto checked_p=[&](const cr::Weight& weight,const ProductRecord& record) {
      if(!checked_target_p(weight.target_potential))return false;
      // Source P(U) is related to the exact admitted time-piece definition,
      // not merely selected by an allocation/header or equal interval length.
      const auto& syntax=*record.syntax;
      Registry::Context context{syntax.weight->source_potential.scope,weight.source_potential.scope};context.axis_count=1;
      context.axes[0]={syntax.time_fold->binder,syntax.time_fold->binder,0,syntax.time_fold->count,state_coordinate,false};
      return products.products.related({find(syntax.weight->source_potential.expression),context.reference_scope},
          {find(weight.source_potential.expression),context.actual_scope},context);
    };
    struct EntryRecord {
      const cr::Candidate* syntax;Use projection,carry,weight;
      cr::Builder::Certificate source,region;
      fp::Interval state;
    };
    std::vector<EntryRecord> entry_records;
    if(carried) {
      // Preprove the ACTUAL entry State function before entering the main
      // EntryProjection callback. No nested query allowance is replaced.
      for(const auto& entry:*candidates)if(!entry.time_fold) {
        phase.next(PrefixPhase::Carry);
        stage="carry_entry_header";
        if(!entry.state_fold || entry.predicate || entry.state_fold->binder!=state_axis ||
            !entry.state_fold->count || entry.state_fold->count>*state_extent)return;
        const auto entry_count=entry.state_fold->count;
        stage="carry_entry_factors";
        std::array<Use,8> raw{};for(unsigned i=0;i<entry.factor_count;++i)
          raw[i]={find(entry.factors[i].expression),entry.factors[i].scope};
        auto factors=products.products.factors(std::span(raw.data(),entry.factor_count));if(!factors)return;
        std::optional<Use> entry_c,entry_s,entry_weight;unsigned weights=0;
        for(auto use:*factors) {
          stage="carry_entry_factor_head";
          auto k=head(use.body);if(!k)return;
          stage="carry_entry_factor_roles";
          if(idealOperation(*k) && k->name=="Exp" && k->children.size()==1){++weights;entry_weight=use;continue;}
          if(same_cell_header(*k,*projection_head)) {if(entry_c)return;entry_c=use;}
          else {if(entry_s)return;entry_s=use;}
        }
        stage="carry_entry_factor_scope";
        if(weights!=1 || !entry_c || !entry_s || entry_s->scope!=entry.state_fold->body_scope ||
            entry_c->scope!=entry.state_fold->body_scope)return;
        const auto original_entry_c=*entry_c,original_entry_s=*entry_s;
        stage="carry_entry_transport_context";
        // Same-count original BODY transport only. A narrow16-state use
        // never acquires the canonical source's128-state lexical domain.
        if(!heads.charge(3+2*std::bit_width(canonical_carry_sources.size()+1)))return;
        auto carry_position=canonical_carry_sources.find(entry_count);
        const auto carry_scope=entry_count==*state_extent?canonical_scope:
            carry_position==canonical_carry_sources.end()?entry_s->scope:carry_position->second;
        Registry::Context transport{carry_scope,entry_s->scope};transport.axis_count=1;
        transport.axes[0]={state_axis,state_axis,0,entry_count,state_coordinate,false};
        auto transport_id=registry.admitContext(transport);if(!transport_id)return;
        stage="carry_entry_state_transport";
        auto attachment=products.products.canonicalUse(*entry_s,*transport_id);
        auto actual=attachment?products.products.assigned(*attachment):std::nullopt;
        if(!actual || actual->scope!=carry_scope || !heads.charge(10))return;
        assignment_attachments.push_back(*attachment);
        stage="carry_entry_projection_transport";
        auto c_attachment=products.products.canonicalUse(*entry_c,*transport_id);
        auto actual_c=c_attachment?products.products.assigned(*c_attachment):std::nullopt;
        if(!actual_c || actual_c->scope!=carry_scope || !heads.charge(10))return;
        assignment_attachments.push_back(*c_attachment);
        if(entry_count!=*state_extent && carry_position==canonical_carry_sources.end())
          canonical_carry_sources.emplace(entry_count,carry_scope);
        entry_c=actual_c;
        stage="carry_entry_time_image";
        auto ct=globalTimeImage(entry_c->scope,output_axis,output_coordinate);
        auto cp=ct?registry.point(*ct,cases.front().point,canonical_scope):std::nullopt;if(!cp)return;
        Registry::Context known{entry_c->scope,canonical_scope};known.axis_count=1;
        auto witness=registry.pointWitness(cases.front().point);if(!witness)return;
        known.axes[0]={output_axis,{},0,1,*witness,false,*cp};
        stage="carry_entry_state_image";
        auto image=registry.imageFromTypedCells(known,entry_c->body,cases.front().projection,state_axis,state_axis);
        auto view=image?registry.view(*image):std::nullopt;
        if(!view || view->scope!=carry_scope || view->level!=state_axis ||
            view->begin>=view->end || view->end>*state_extent || view->end-view->begin!=entry_count)return;
        const bool restricted=entry_count!=*state_extent || view->begin!=0 || view->end!=*state_extent;
        if(!restricted) {
          stage="carry_entry_full_projection";
          // Preserve the existing full-state route, including families
          // whose complete C definition comprises several state boxes.
          const std::array<std::uint32_t,2> c_axes{output_axis,state_axis};
          const std::array<Registry::ImageId,2> c_images{*ct,*image};
          if(!products.geometry.uniform(c_family->geometry,entry_c->body,entry_c->scope,c_axes,c_images))return;
        } else {
        stage="carry_entry_projection_box";
        // One uniquely containing ACTUAL admitted C box, then the SAME full
        // typed scoped relation. Geometry is not body equality and does not
        // ask a16-state representative to be uniform over all128 states.
        std::optional<std::size_t> c_index;
        for(std::size_t i=0;i<c_geometry->boxes.size();++i) {
          if(!heads.charge())return;const auto& box=c_geometry->boxes[i];
          if(box[0]==std::pair{output_domain->begin,output_domain->end} &&
              box[1].first<=view->begin && box[1].second>=view->end) {
            if(c_index)return;c_index=i;
          }
        }
        if(!c_index)return;const auto& expected_c=c_family->pieces[*c_index];
        if(expected_c.count!=1)return;
        stage="carry_entry_projection_image";
        auto c_time=registry.relativeImage(*ct,expected_c.images[0]);
        auto c_state=registry.relativeImage(*image,expected_c.images[1]);
        auto ctv=c_time?registry.view(*c_time):std::nullopt,csv=c_state?registry.view(*c_state):std::nullopt;
        if(!ctv || !csv)return;
        Registry::Context c_relation{expected_c.factors[0].scope,entry_c->scope};c_relation.axis_count=2;
        c_relation.axes[0]={output_axis,output_axis,ctv->begin,output_domain->end-output_domain->begin,
            ctv->coordinate,false,{},*c_time};
        auto expected_state=registry.view(expected_c.images[1]);if(!expected_state)return;
        c_relation.axes[1]={expected_state->level,state_axis,csv->begin,entry_count,
            csv->coordinate,false,{},*c_state};
        stage="carry_entry_projection_relation";
        if(!products.products.related(expected_c.factors[0],*entry_c,c_relation))return;
        }
        stage="carry_entry_component_context";
        Registry::Context component_assignment{canonical_scope,carry_scope};component_assignment.axis_count=1;
        component_assignment.axes[0]={state_axis,state_axis,view->begin,entry_count,view->coordinate,false,{},*image};
        auto component_context=registry.admitContext(component_assignment);if(!component_context)return;
        stage="carry_entry_state_function";
        auto identity=proveStateFunction(*carried,*actual,state_coordinate,state_coordinate,
            state_axis,state_axis+1,*image,entry_c->body,*component_context,restricted);
        if(!identity)return;
        entry_records.push_back({&entry,original_entry_c,original_entry_s,*entry_weight,
            identity->source,identity->region,identity->state});
      }
      stage="carry_entry_records";
      if(entry_records.empty())return;
    }
    region_product=[&](const cr::ProductRequest& request) {
      ++product_calls;last_product_kind=1+static_cast<unsigned>(request.kind);
      if(request.kind==cr::ProductKind::EntryProjection)entry_stage="header";
      const auto before=remaining;cr::ProductResult result;
      const auto finish=[&]() {result.work=before-remaining;return result;};
      std::size_t local=request.max_work;BodyHeadIndex::QueryBudget guard(heads,&local);
      if(!heads.charge() ||
          !request.completed || *request.completed!=*completed || !request.output || *request.output!=checked_output ||
          !request.projection || *request.projection!=projection_definition ||
          !request.state_fold || !request.weight ||
          request.frame.potential_template!=virtual_p || request.frame.injection_template!=virtual_q ||
          request.frame.template_scope!=virtual_scope || request.frame.state_scope!=canonical_scope ||
          request.frame.entry_state!=entry_state || request.frame.entry_potential!=entry_potential ||
          request.frame.carry!=frame.carry || request.frame.time!=fp::Interval{source_domain->begin,source_domain->end} ||
          request.frame.state_extent!=*state_extent || request.factor_count>8)return finish();
      if(request.kind==cr::ProductKind::EntryProjection) {
        entry_stage="entry_header";
        if(!carried || request.time_fold || request.endpoint || request.state_output ||
            request.weight->source_potential.expression>=nodes_.size())return finish();
        const auto& sf=*request.state_fold;const EntryRecord* found=nullptr;std::size_t index=0;
        entry_stage="record";
        for(std::size_t i=0;i<entry_records.size();++i) {
          if(!heads.charge())return finish();const auto& record=entry_records[i];
          const auto& original=*record.syntax->state_fold;
          if(sf.head==original.head && sf.body_scope==original.body_scope && sf.binder==original.binder && sf.count==original.count) {
            if(found)return finish();found=&record;index=i;
          }
        }
        if(!found)return finish();entry_stage="weight_occurrence";
        if(request.weight->factor!=cr::Use{found->weight.body,found->weight.scope})return finish();
        entry_stage="identity_certificate";
        if(!regions.equal(found->source,found->region).value_or(false))return finish();
        entry_stage="target_potential";
        if(!checked_target_p(request.weight->target_potential))return finish();
        // The private Identity certificate relates the original carry
        // function to the actual prior terminal. The baseline remains the
        // actual new frame E, independently of that prior terminal P.
        entry_stage="baseline_image";
        auto local=localOutput(request.weight->source_potential.scope);if(!local)return finish();
        Registry::Context baseline{canonical_scope,request.weight->source_potential.scope};baseline.axis_count=2;
        baseline.axes[0]={output_axis,output_axis,0,output_domain->end-output_domain->begin,output_coordinate,false,{},*local};
        baseline.axes[1]={state_axis,state_axis,found->state.begin,sf.count,state_coordinate,false};
        entry_stage="baseline_relation";
        const Use expected_e{entry_potential,canonical_scope};
        const Use actual_e{find(request.weight->source_potential.expression),request.weight->source_potential.scope};
        // E may be outside the state Fold. Derive its original-use context
        // through the SAME conservative dependency-absence factory; never
        // move that scalar into a wider scope or omit a used/captured axis.
        auto scalar_context=products.products.contextFor(expected_e,actual_e,baseline);
        if(!scalar_context || !products.products.related(expected_e,actual_e,*scalar_context))return finish();
        const std::array<Use,2> expected{found->projection,found->carry};std::array<Use,8> actual{};
        entry_stage="factor_occurrences";
        if(request.factor_count!=expected.size() || !heads.charge(2*request.factor_count))return finish();
        for(unsigned i=0;i<request.factor_count;++i) {
          if(request.factors[i].expression>=nodes_.size())return finish();
          actual[i]={find(request.factors[i].expression),request.factors[i].scope};
        }
        entry_stage="factor_image";
        auto identity_output=localOutput(sf.body_scope);if(!identity_output)return finish();
        Registry::Context identity{sf.body_scope,sf.body_scope};identity.axis_count=2;
        identity.axes[0]={output_axis,output_axis,0,output_domain->end-output_domain->begin,output_coordinate,false,{},*identity_output};
        identity.axes[1]={state_axis,state_axis,0,sf.count,state_coordinate,false};
        entry_stage="factor_relation";
        if(!products.products.equal(expected,std::span(actual.data(),request.factor_count),identity))return finish();
        result.status=fp::RelationStatus::Proved;result.state=found->state;
        entry_stage="complete";
        result.geometry_token=records.size()+index+1;return finish();
      }
      if(request.kind!=cr::ProductKind::InjectionProjection || !request.time_fold)return finish();
      const auto& tf=*request.time_fold;const auto& sf=*request.state_fold;
      if(!heads.charge(4+std::bit_width(product_records.size()+1)))return finish();
      auto found=product_records.find({tf.head.expression,tf.head.scope,sf.head.expression,sf.head.scope});
      if(found==product_records.end())return finish();
      const auto& record=records[found->second];const auto& syntax=*record.syntax;
      if(tf.binder!=syntax.time_fold->binder || tf.count!=syntax.time_fold->count || tf.body_scope!=syntax.time_fold->body_scope ||
          sf.binder!=syntax.state_fold->binder || sf.count!=syntax.state_fold->count || sf.body_scope!=syntax.state_fold->body_scope ||
          request.weight->factor!=syntax.weight->factor || request.weight->target_potential!=syntax.weight->target_potential ||
          request.weight->source_potential!=syntax.weight->source_potential ||
          !checked_c(record) || !checked_p(*request.weight,record))return finish();
      std::array<Use,8> expected{},actual{};unsigned count=0;
      for(unsigned i=0;i<syntax.factor_count;++i)if(syntax.factors[i]!=syntax.weight->factor) {
        if(count>=8)return finish();expected[count++]={find(syntax.factors[i].expression),syntax.factors[i].scope};
      }
      if(count!=request.factor_count || !count || !heads.charge(count))return finish();
      for(unsigned i=0;i<count;++i) {
        if(request.factors[i].expression>=nodes_.size())return finish();
        actual[i]={find(request.factors[i].expression),request.factors[i].scope};
      }
      Registry::Context context{sf.body_scope,sf.body_scope};context.axis_count=2;
      context.axes[0]={tf.binder,tf.binder,0,tf.count,state_coordinate,false};
      context.axes[1]={sf.binder,sf.binder,0,sf.count,nested_state_coordinate,false};
      if(!products.products.equal(std::span(expected.data(),count),std::span(actual.data(),count),context))return finish();
      auto time=registry.view(record.time),state=registry.view(record.state);if(!time || !state)return finish();
      result.status=fp::RelationStatus::Proved;result.time={time->begin,time->end};result.state={state->begin,state->end};
      result.geometry_token=found->second+1;return finish();
    };
    region_mask=[&](const cr::MaskRequest& request) {
      ++mask_calls;
      const auto before=remaining;fp::RelationResult result;
      const auto finish=[&]() {result.work=before-remaining;return result;};
      std::size_t local=request.max_work;BodyHeadIndex::QueryBudget guard(heads,&local);
      if(!heads.charge() || !request.output || *request.output!=checked_output || request.endpoint ||
          !request.geometry_token || request.geometry_token>records.size())return finish();
      const auto& record=records[request.geometry_token-1];const auto& syntax=*record.syntax;
      if(!syntax.predicate || request.predicate!=*syntax.predicate || request.selected_when_true!=syntax.selected_when_true)return finish();
      auto k=head(request.predicate.expression);if(!k || k->children.size()!=2 ||
          !k->provenance.empty())return finish();
      auto relation=comparison(*k);if(!relation)return finish();
      auto time=registry.view(record.time);if(!time)return finish();
      // The actual predicate compares LOCAL unit witnesses. Its two
      // semantic images were translated through this SAME proved assignment
      // (with checked no-wrap bounds), so <=/>= order is preserved on BOTH
      // sides. A one-sided shift or another frame's token cannot certify it.
      if(!registry.commonImagePair(*common_time,record.time,*output_image))return finish();
      const auto side=[&](Class child,Class wanted) {
        if(child>=nodes_.size() || wanted>=nodes_.size())return false;
        const auto raw=[](const std::string& t){return t.starts_with("exact:")?t.substr(6):t;};
        if(raw(nodes_[find(child)].type)!="u64" || raw(nodes_[find(wanted)].type)!="u64")return false;
        auto a=heads.affine(child),b=heads.affine(wanted);
        return a && b && registry.boundedAffine(*a,request.predicate.scope) &&
            registry.boundedAffine(*b,request.predicate.scope) && a->constant==b->constant && a->coefficients==b->coefficients;
      };
      const bool ut=side(k->children[0],time->coordinate) && side(k->children[1],output_coordinate);
      const bool tu=side(k->children[1],time->coordinate) && side(k->children[0],output_coordinate);
      const bool proved=request.selected_when_true?((*relation=="le" && ut)||(*relation=="ge" && tu)):
          ((*relation=="gt" && ut)||(*relation=="lt" && tu));
      if(proved)result.status=fp::RelationStatus::Proved;return finish();
    };
    phase.next(PrefixPhase::Region);
    stage="whole_output_region";
    const cr::Builder::ParameterizedRequest target{{find(right_root),*outer},{output_coordinate,*outer},
        {find(left_root),*outer},{coordinate,*outer},*completed,*projection};
    auto region=regions.parameterizedRegion(target);
    region_complete=bool(region);region_cap=regions.exhausted();
    const cr::Builder::ParameterizedRequest source{{find(left_root),*outer},{coordinate,*outer},
        {find(left_root),*outer},{coordinate,*outer},*completed,*projection};
    phase.next(PrefixPhase::Dispatch);
    auto dispatch=regions.parameterizedDispatch(source);
    dispatch_complete=bool(dispatch);region_cap=region_cap || regions.exhausted();
    const bool proved=region && dispatch && regions.equal(*region,*dispatch).value_or(false);
    if(proved) {
      if(!heads.charge(3))return;
      proved_roots.emplace_back(left_root,right_root);output_proved=true;
    }
    stage=proved?"whole_output_proved":"whole_output_unknown";
    scope_count=registry.scopes.size();coverage_work=potentials.coverageWork()+products.geometry.coverageWork();
    };
    for(std::size_t i=0;i<left.size();++i) {
      if(!remaining || heads.exhausted || regions.exhausted() || core.exhausted())break;
      if(!heads.charge())break;
      const auto& l=nodes_[find(left[i])];
      const bool dispatch=idealSelect(l);
      const bool terminal=l.type=="real" && arithmetic(l) && l.name=="Add";
      if(!dispatch && !terminal)continue;
      const auto root_attempts=roots_attempted,root_work=remaining,root_cases=source_cases,root_attached=attached,
          root_updates=proved_updates,root_frames=completed_frames,root_proved=proved_roots.size();
      if(dispatch)proveRoot(find(left[i]),find(right[i]),i);
      else proveTerminalRoot(find(left[i]),find(right[i]));
      // Report only a failed existing carry stage. No new witness selection,
      // traversal, rendering, semantic fact or proof-work debit is involved.
      if(prefix_trace && roots_attempted>root_attempts && proved_roots.size()==root_proved &&
          carry_denial_trace<16) {
        ++carry_denial_trace;
        SENTRYWARD_DEBUG_PRINT(egraph_89);
      }
      if(prefix_trace && roots_attempted>root_attempts && root_trace_size<root_trace.size())
        root_trace[root_trace_size++]={i,root_work-remaining,
          source_cases-root_cases,attached-root_attached,proved_updates-root_updates,
          completed_frames-root_frames,proved_roots.size()-root_proved,stage,entry_stage,coordinate_trace};
      // A callback must never outlive the root-specific original-path
      // records it borrows. Earlier private certificates remain valid; no
      // later query can accidentally call a previous root's attachment.
      core_relate={};region_product={};region_mask={};region_target={};region_state_target={};
    }
  };
  run();
  if(prefix_trace)SENTRYWARD_DEBUG_PRINT(egraph_90);
  if(prefix_trace) {
    constexpr std::array names{"source","templates","sampling","output","carry","region","dispatch"};
    SENTRYWARD_DEBUG_PRINT(egraph_91);
    SENTRYWARD_DEBUG_PRINT(egraph_92);
    std::size_t accounted=0;SENTRYWARD_DEBUG_PRINT(egraph_93);
    for(unsigned i=0;i<phase_work.size();++i){accounted+=phase_work[i];SENTRYWARD_DEBUG_PRINT(egraph_94);}
    SENTRYWARD_DEBUG_PRINT(egraph_95);
    for(std::size_t i=0;i<root_trace_size;++i) {
      const auto& r=root_trace[i];SENTRYWARD_DEBUG_PRINT(egraph_96);
      if(r.coordinate.candidate) {
        const auto& c=r.coordinate;
        SENTRYWARD_DEBUG_PRINT(egraph_97);
        for(unsigned j=0;j<2;++j)SENTRYWARD_DEBUG_PRINT(egraph_98);
        SENTRYWARD_DEBUG_PRINT(egraph_99);
      }
    }
  }
  // All borrowed scopes/heads/core/region certificates have now died. Only
  // their completed whole-root equality is transferred to the live Engine.
  // No private template/component handle creates an internal union.
  for(auto [a,b]:proved_roots)rewrite(a,b);
  return output_proved;
}

} // namespace
} // namespace sentryward::equivalence
