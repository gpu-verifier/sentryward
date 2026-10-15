#include "sentryward/equivalence/causal_region_certificate.h"
#include "sentryward/equivalence/epoch_work_allowance.h"

#include <algorithm>
#include <atomic>
#include <unordered_map>

namespace sentryward::equivalence::causal_region_certificate {
namespace linear=bounded_linear_reduction;
namespace {
std::atomic<std::uint64_t> next_owner{1};
std::uint64_t allocateOwner() {
  auto value=next_owner.load();
  for(;;) {
    if(!value)return 0;
    const auto next=value==UINT64_MAX?std::uint64_t{0}:value+1;
    if(next_owner.compare_exchange_weak(value,next))return value;
  }
}
Limits bounded(Limits limits) {
  limits.max_nodes=std::min(limits.max_nodes,epoch_work_allowance::maximum_retained_headers);
  limits.max_projections=std::min<std::size_t>(limits.max_projections,256);
  limits.max_certificates=std::min<std::size_t>(limits.max_certificates,4096);
  limits.max_pieces=std::min<std::size_t>(limits.max_pieces,4096);
  limits.max_work=std::min(limits.max_work,epoch_work_allowance::maximum_work);
  return limits;
}
struct UseHash {
  std::size_t operator()(Use u) const {
    return std::hash<ExprId>{}(u.expression)^(std::size_t(u.scope)*0x9e3779b9u);
  }
};
void mixHash(std::size_t& h,std::size_t v) {h^=v+0x9e3779b9u+(h<<6)+(h>>2);}
bool primitiveAttributes(std::string_view a) {return a.empty() || a=="ideal_arithmetic";}
linear::Limits linearLimits(Limits limits) {
  linear::Limits out;
  // Retained selected headers are physical epoch storage, not per-query
  // atoms/pieces. Honor the explicitly configured bounded capacity in BOTH
  // catalogs; DagSession also intersects with this parent Builder limit.
  out.max_nodes=limits.max_nodes;
  out.max_atoms=std::min(out.max_atoms,limits.max_pieces);
  // An explicitly configured physical-epoch allowance is shared by this
  // builder's primitive hook, not an independent fresh allowance. Preserve
  // the public default1M while avoiding a second hidden cumulative1M cap.
  out.max_work=limits.max_work;out.coefficients.max_work=out.max_work;
  return out;
}
}
struct Builder::Impl {
  prefix::Builder& core;
  Resolve resolve;BindBody bind;RelateProduct product;ProveMask mask;ProveOutputTarget output_target;ProveStateTarget state_target;
  std::function<bool(std::size_t)> shared;Limits limits;Statistics stats;
  std::uint64_t owner=allocateOwner();bool stopped=false;
  linear::Builder linear;
  std::unordered_map<Use,std::optional<Node>,UseHash> headers;
  std::unordered_map<Use,std::optional<ScopeId>,UseHash> scopes;
  struct Slot {Use use;bool body=false;};
  struct SlotKey {Use use;bool body;bool operator==(const SlotKey&) const=default;};
  struct SlotHash {
    std::size_t operator()(const SlotKey& key) const {
      return UseHash{}(key.use)^(std::size_t(key.body)*0x517cc1b7u);
    }
  };
  std::vector<Slot> slots;
  std::unordered_map<SlotKey,std::size_t,SlotHash> slot_ids;
  std::unique_ptr<linear::DagSession> forests;
  std::optional<linear::Coefficient> one;
  std::vector<ProjectionRequest> projections;
  struct Key {
    enum class Mode {FixedContraction,TimeFunction,StateFunction,RestrictedStateFunction};
    prefix::Builder::FrameId frame;
    std::uint64_t endpoint;
    std::size_t projection;
    std::optional<OutputTarget> output;
    Mode mode=Mode::FixedContraction;
    std::optional<StateRestrictionId> restriction={};
    std::optional<prefix::Builder::StateId> terminal={};
    bool operator==(const Key&) const=default;
  };
  std::vector<Key> certificates;
  struct OutputDomain {
    prefix::Builder::FrameId frame;std::size_t projection;OutputTarget target;
    Key::Mode mode=Key::Mode::TimeFunction;
    bool operator==(const OutputDomain&) const=default;
  };
  std::vector<OutputDomain> output_domains;
  struct StateRestriction {
    StateRestrictionRequest request;
    StateTarget source,region;
  };
  std::vector<StateRestriction> state_restrictions;
  struct RestrictionHash {
    std::size_t operator()(const StateRestrictionRequest& q) const {
      std::size_t h=q.terminal.hash();mixHash(h,UseHash{}(q.region_root));
      mixHash(h,UseHash{}(q.state_coordinate));mixHash(h,UseHash{}(q.source_state));
      mixHash(h,UseHash{}(q.source_coordinate));return h;
    }
  };
  std::unordered_map<StateRestrictionRequest,std::optional<std::size_t>,RestrictionHash> restriction_queries;
  struct Query {
    Use root;prefix::Builder::StateId state;std::size_t projection;bool region;
    bool operator==(const Query&) const=default;
  };
  struct QueryHash {
    std::size_t operator()(const Query& q) const {
      return UseHash{}(q.root)^q.state.hash()^(q.projection<<1)^std::size_t(q.region);
    }
  };
  std::unordered_map<Query,std::optional<std::size_t>,QueryHash> queries;
  struct ParameterQuery {
    Use root,coordinate,dispatch,source_coordinate;
    prefix::Builder::CompletedPrefixFamilyId completed;
    std::size_t projection;bool dispatch_only;
    bool operator==(const ParameterQuery&) const=default;
  };
  struct ParameterHash {
    std::size_t operator()(const ParameterQuery& q) const {
      std::size_t h=q.completed.hash();mixHash(h,UseHash{}(q.root));mixHash(h,UseHash{}(q.coordinate));
      mixHash(h,UseHash{}(q.dispatch));mixHash(h,UseHash{}(q.source_coordinate));
      mixHash(h,q.projection);mixHash(h,std::size_t(q.dispatch_only));return h;
    }
  };
  std::unordered_map<ParameterQuery,std::optional<std::size_t>,ParameterHash> parameter_queries;
  struct IdentityQuery {
    Use root,coordinate,source,source_coordinate;
    prefix::Builder::StateId terminal;bool source_only;
    std::optional<StateRestrictionId> restriction={};
    bool operator==(const IdentityQuery&) const=default;
  };
  struct IdentityHash {
    std::size_t operator()(const IdentityQuery& q) const {
      std::size_t h=q.terminal.hash();mixHash(h,UseHash{}(q.root));mixHash(h,UseHash{}(q.coordinate));
      mixHash(h,UseHash{}(q.source));mixHash(h,UseHash{}(q.source_coordinate));
      if(q.restriction){mixHash(h,q.restriction->owner);mixHash(h,q.restriction->index);}
      mixHash(h,std::size_t(q.source_only));return h;
    }
  };
  std::unordered_map<IdentityQuery,std::optional<std::size_t>,IdentityHash> identity_queries;
  struct CandidateGroup {std::optional<FoldUse> time;std::size_t begin,end;};
  struct Parsed {
    std::vector<Candidate> records;std::vector<CandidateGroup> groups;
  };
  // unique_ptr keeps published record spans stable across map rehash. Parsed
  // vectors are built privately once and never append after publication.
  struct SyntaxKey {Use root;CandidateMode mode;bool operator==(const SyntaxKey&) const=default;};
  struct SyntaxHash {
    std::size_t operator()(const SyntaxKey& key) const {
      return UseHash{}(key.root)^std::size_t(key.mode);
    }
  };
  std::unordered_map<SyntaxKey,std::unique_ptr<Parsed>,SyntaxHash> parsed_roots;
  std::size_t pending_queries=0;
  struct QueryReservation {
    std::size_t& count;
    explicit QueryReservation(std::size_t& n):count(n){++count;}
    ~QueryReservation(){--count;}
  };
  struct Context {
    std::optional<prefix::Builder::StateView> endpoint;
    prefix::Builder::FrameView frame;
    std::optional<ProjectionRequest> projection;
    std::optional<prefix::Builder::CompletedPrefixFamilyId> completed;
    std::optional<OutputTarget> output;
    std::optional<StateTarget> state_output;
    std::optional<OutputDomainId> output_domain;
  };
  struct Mask {Use predicate;bool positive;};
  struct Product {std::vector<Use> factors;std::optional<Mask> mask;};
  Impl(prefix::Builder& c,Resolve r,BindBody b,RelateProduct p,
      ProveMask m,std::function<bool(std::size_t)> s,Limits cap,ProveOutputTarget target,ProveStateTarget state):
      core(c),resolve(std::move(r)),bind(std::move(b)),product(std::move(p)),
      mask(std::move(m)),output_target(std::move(target)),state_target(std::move(state)),shared(std::move(s)),limits(bounded(cap)),
      linear(linearLimits(limits),[this](std::size_t n){return charge(n);}) {
    linear::Limits lc;lc.max_nodes=limits.max_nodes;lc.max_atoms=limits.max_pieces;
    lc.max_cached_terms=limits.max_nodes;lc.max_work=limits.max_work;
    forests=std::make_unique<linear::DagSession>(linear,[this](auto id){return forestHead(id);},lc);
    one=linear.coefficient("1");
    if(!owner || !one)stopped=true;
  }
  bool charge(std::size_t n=1) {
    if(stopped || n>limits.max_work-stats.work || !shared || !shared(n)) {
      stopped=true;return false;
    }
    stats.work+=n;return true;
  }
  bool externalWork(std::size_t n) {
    // Adapter already charged its shared ledger BEFORE work. Do not debit it
    // twice. Incoming max_work bounds this local delta as well.
    if(stopped || n>limits.max_work-stats.work){stopped=true;return false;}
    stats.work+=n;return true;
  }
  std::size_t queryCount() const {
    return queries.size()+parameter_queries.size()+identity_queries.size()+restriction_queries.size()+
        parsed_roots.size()+pending_queries;
  }
  std::size_t certificateCount() const {
    return certificates.size()+output_domains.size()+state_restrictions.size();
  }
  std::optional<Node> read(Use u) {
    if(!charge())return {};
    if(auto f=headers.find(u);f!=headers.end())return f->second;
    if(headers.size()>=limits.max_nodes){stopped=true;return {};}
    auto n=resolve?resolve(u):std::optional<Node>{};
    if(n) {
      const unsigned arity=n->kind==Kind::Zero || n->kind==Kind::Opaque?0:
          n->kind==Kind::Negate || n->kind==Kind::NaturalExp?1:n->kind==Kind::Select?3:2;
      if(n->type!="real" || !n->finite_ideal || !n->provenance.empty() ||
          n->arity!=arity || n->free_count>8 || n->attributes.size()>128 ||
          (n->kind==Kind::Sum?!n->attributes.empty():
            n->kind!=Kind::Opaque && !primitiveAttributes(n->attributes)))n.reset();
      if(n && n->exact_dependencies)for(unsigned i=1;i<n->free_count;++i)
        if(n->free_indices[i-1]>=n->free_indices[i]){n.reset();break;}
    }
    headers.emplace(u,n);stats.nodes=headers.size();return n;
  }
  bool independent(Use u,std::uint32_t axis) {
    auto n=read(u);return n && n->exact_dependencies &&
      std::find(n->free_indices.begin(),n->free_indices.begin()+n->free_count,axis)==
        n->free_indices.begin()+n->free_count;
  }
  std::optional<FoldUse> bodyScope(Use u) {
    auto n=read(u);if(!n || n->kind!=Kind::Sum || !n->count)return {};
    std::optional<ScopeId> result;
    if(auto f=scopes.find(u);f!=scopes.end())result=f->second;
    else {
      if(scopes.size()>=limits.max_nodes){stopped=true;return {};}
      const auto remaining=limits.max_work-stats.work;
      auto proof=bind?bind(u,remaining):ScopeResult{};
      if(proof.work>remaining){stopped=true;return {};}
      if(!externalWork(proof.work))return {};
      result=proof.scope;scopes.emplace(u,result);
    }
    if(!result)return {};return FoldUse{u,*result,n->binder,n->count};
  }
  std::optional<std::size_t> slot(Use u,bool body=false) {
    const SlotKey key{u,body};
    if(auto f=slot_ids.find(key);f!=slot_ids.end())return f->second;
    if(!charge() || slots.size()>=limits.max_nodes){stopped=true;return {};}
    auto id=slots.size();slots.push_back({u,body});slot_ids.emplace(key,id);return id;
  }
  std::optional<linear::Node> forestHead(std::size_t id) {
    if(id>=slots.size())return {};const auto source=slots[id]; // Copy before slot append.
    auto n=read(source.use);if(!n)return {};
    linear::Node out;out.type="real";out.admitted=true;
    out.kind=linear::NodeKind::Atom;out.atom=id;
    if(source.body)return n->kind==Kind::Sum && n->count?std::optional{out}:std::nullopt;
    if(n->kind==Kind::Zero){out.kind=linear::NodeKind::Literal;out.literal="0";return out;}
    if(n->kind==Kind::Add || n->kind==Kind::Sum) {
      const auto a=slot({n->children[0],source.use.scope});
      const auto b=n->kind==Kind::Sum?slot(source.use,true):slot({n->children[1],source.use.scope});
      if(!a || !b)return {};
      if(n->kind==Kind::Sum && !n->count) {
        out.kind=linear::NodeKind::Identity;out.children[0]=*a;out.child_count=1;
      } else {out.kind=linear::NodeKind::Add;out.children={*a,*b};out.child_count=2;}
    }
    return out;
  }
  std::optional<std::vector<Slot>> forest(Use root) {
    auto id=slot(root);if(!id)return {};
    ++stats.forest_queries;auto form=forests->summarize(*id);
    if(!form || !form->constant().terms().empty())return {};
    std::vector<Slot> result;
    if(!charge(form->terms().size()) || form->terms().size()>limits.max_pieces)return {};
    for(const auto& t:form->terms()) {
      if(t.weight!=*one || t.atom>=slots.size())return {};
      result.push_back(slots[t.atom]); // Duplicate contribution never deduplicated.
    }
    return result;
  }
  std::optional<Product> factors(Use root) {
    Product result;std::vector<Use> pending{root};
    // Binary product tree with <=8 occurrences visits <=15 product/leaf heads.
    // Shared occurrences remain repeated; a selected cycle exceeds this cap.
    for(unsigned visits=0;!pending.empty();++visits) {
      if(visits>=31 || !charge())return {};
      auto u=pending.back();pending.pop_back();auto n=read(u);if(!n)return {};
      if(n->kind==Kind::Multiply) {
        pending.push_back({n->children[0],u.scope});pending.push_back({n->children[1],u.scope});
      } else if(n->kind==Kind::Select) {
        if(result.mask)return {};
        auto a=read({n->children[1],u.scope}),b=read({n->children[2],u.scope});
        if(!a || !b)return {};
        const bool az=a->kind==Kind::Zero,bz=b->kind==Kind::Zero;
        if(az==bz)return {};
        result.mask=Mask{{n->children[0],u.scope},bz};
        pending.push_back({n->children[bz?1:2],u.scope});
      } else {
        result.factors.push_back(u);if(result.factors.size()>8)return {};
      }
    }
    return result;
  }
  std::optional<Weight> weight(Use u) {
    auto n=read(u);if(!n || n->kind!=Kind::NaturalExp)return {};
    const Use argument{n->children[0],u.scope};auto a=read(argument);if(!a)return {};
    if(a->kind==Kind::Subtract)return Weight{u,{a->children[0],u.scope},{a->children[1],u.scope}};
    if(a->kind==Kind::Add)for(unsigned i=0;i<2;++i) {
      const Use negative{a->children[i],u.scope};auto b=read(negative);
      if(b && b->kind==Kind::Negate)return Weight{u,{a->children[1-i],u.scope},{b->children[0],u.scope}};
    }
    return {};
  }
  bool covered(std::span<const Interval> pieces,Interval expected) {
    if(!one || expected.begin>=expected.end || pieces.empty() ||
        pieces.size()>limits.max_pieces || !charge(pieces.size()))return false;
    std::vector<linear::WeightedInterval> input;input.reserve(pieces.size());
    for(auto p:pieces) {
      if(p.begin>=p.end || p.begin<expected.begin || p.end>expected.end)return false;
      input.push_back({p.begin,p.end,*one});
    }
    const std::array<linear::WeightedInterval,1> target{{{expected.begin,expected.end,*one}}};
    auto equal=linear.intervalsEqual(input,target);return equal && *equal;
  }
  std::optional<ProductResult> relate(const Context& c,ProductKind kind,
      std::optional<FoldUse> time,std::optional<FoldUse> state,Product p) {
    ProductRequest request;request.kind=kind;request.frame=c.frame;request.endpoint=c.endpoint;
    request.completed=c.completed;request.output=c.output;request.state_output=c.state_output;request.output_domain=c.output_domain;
    request.projection=c.projection;request.time_fold=time;request.state_fold=state;
    for(auto u:p.factors) {
      auto w=weight(u);
      if(!w && (kind==ProductKind::EntryProjection || kind==ProductKind::EntryIdentity)) {
        auto n=read(u),initial=read({c.frame.entry_potential,c.frame.state_scope});
        // Exp(P) is Exp(P-E) ONLY for the actual certified Zero baseline.
        // Never silently set an arbitrary entry potential to zero.
        if(n && n->kind==Kind::NaturalExp && initial && initial->kind==Kind::Zero)
          w=Weight{u,{n->children[0],u.scope},{c.frame.entry_potential,c.frame.state_scope}};
      }
      if(w) {
        if(request.weight)return {};request.weight=*w;
      } else {
        if(request.factor_count>=8)return {};request.factors[request.factor_count++]=u;
      }
    }
    if((kind==ProductKind::StateProjection)==bool(request.weight) || !request.factor_count)return {};
    if(!charge() || stats.pieces+stats.syntax_candidates>=limits.max_pieces){stopped=true;return {};}
    ++stats.products;request.max_work=limits.max_work-stats.work;
    auto proof=product?product(request):ProductResult{};
    if(proof.work>request.max_work){stopped=true;return {};}
    if(!externalWork(proof.work) ||
        proof.status!=prefix::RelationStatus::Proved || !proof.geometry_token ||
        proof.state.begin>=proof.state.end || proof.state.end>c.frame.state_extent ||
        (state?proof.state.end-proof.state.begin!=state->count:
          !c.state_output || proof.state!=c.state_output->image))return {};
    if(time && (proof.time.begin>=proof.time.end || proof.time.begin<c.frame.time.begin ||
        proof.time.end>c.frame.time.end || proof.time.end-proof.time.begin!=time->count))return {};
    if(p.mask) {
      if(kind!=ProductKind::InjectionProjection && kind!=ProductKind::InjectionIdentity)return {};
      MaskRequest m;m.predicate=p.mask->predicate;m.selected_when_true=p.mask->positive;
      m.geometry_token=proof.geometry_token;m.endpoint=c.endpoint?
          std::optional{c.endpoint->endpoint}:std::nullopt;m.output=c.output;m.state_output=c.state_output;m.output_domain=c.output_domain;
      m.max_work=limits.max_work-stats.work;++stats.masks;
      auto result=mask?mask(m):prefix::RelationResult{};
      if(result.work>m.max_work){stopped=true;return {};}
      if(!externalWork(result.work) || result.status!=prefix::RelationStatus::Proved)return {};
    } else if(time && proof.time.end-1>(c.endpoint?c.endpoint->endpoint:c.output->image.begin))return {};
    ++stats.pieces;return proof;
  }
  bool parseStatePieces(Parsed& parsed,std::optional<FoldUse> time,
      Use root,std::span<const Use> outside,std::optional<Mask> outer_mask) {
    auto terms=forest(root);if(!terms || terms->empty())return {};
    const auto begin=parsed.records.size();
    for(auto term:*terms) {
      if(!term.body)return false;auto fold=bodyScope(term.use);if(!fold)return false;
      // Factors crossing an actual state Sum retain original scopes and require
      // exact dependency absence. The SAME mapper checks moves again.
      for(auto u:outside)if(!independent(u,fold->binder))return false;
      if(outer_mask) {
        // Predicate is exact/boolean, so its independence is validated by the
        // shared mask factory against the coordinate token, not numeric read().
      }
      auto head=read(term.use);auto p=factors({head->children[1],fold->body_scope});if(!p)return false;
      if(outer_mask && p->mask)return false;if(outer_mask)p->mask=outer_mask;
      if(p->factors.size()+outside.size()>8)return false;
      p->factors.insert(p->factors.end(),outside.begin(),outside.end());
      if(!appendCandidate(parsed,time,*fold,*p))return false;
    }
    if(!charge())return false;
    parsed.groups.push_back({time,begin,parsed.records.size()});return true;
  }
  bool appendCandidate(Parsed& parsed,std::optional<FoldUse> time,
      std::optional<FoldUse> state,const Product& p) {
    if(!charge(p.factors.size()+1) || stats.pieces+stats.syntax_candidates>=limits.max_pieces) {
      stopped=true;return false;
    }
    Candidate candidate;candidate.group=parsed.groups.size();candidate.time_fold=time;
    candidate.state_fold=state;candidate.factor_count=p.factors.size();
    std::copy(p.factors.begin(),p.factors.end(),candidate.factors.begin());
    if(p.mask) {
      candidate.predicate=p.mask->predicate;candidate.selected_when_true=p.mask->positive;
    }
    for(auto u:p.factors)if(auto w=weight(u)) {
      if(candidate.weight)return false;candidate.weight=*w;
    }
    parsed.records.push_back(candidate);++stats.syntax_candidates;return true;
  }
  bool parseIdentityProduct(Parsed& parsed,std::optional<FoldUse> time,Use body) {
    auto p=factors(body);if(!p || (!time && p->mask))return false;
    const auto begin=parsed.records.size();
    if(!appendCandidate(parsed,time,{},*p) || !charge())return false;
    parsed.groups.push_back({time,begin,parsed.records.size()});return true;
  }
  std::optional<Interval> proveGroup(const Context& c,ProductKind kind,
      std::span<const Candidate> candidates,std::optional<FoldUse> time) {
    std::vector<Interval> pieces;std::optional<Interval> time_image;
    for(const auto& candidate:candidates) {
      if(!charge(candidate.factor_count+1))return {};
      Product p;p.factors.assign(candidate.factors.begin(),candidate.factors.begin()+candidate.factor_count);
      if(candidate.predicate)p.mask=Mask{*candidate.predicate,candidate.selected_when_true};
      if(!candidate.state_fold)return {};
      auto proof=relate(c,kind,time,candidate.state_fold,std::move(p));if(!proof)return {};
      if(time_image && *time_image!=proof->time)return {};time_image=proof->time;
      pieces.push_back(proof->state);
    }
    if(!covered(pieces,{0,c.frame.state_extent}))return {};
    return time?time_image:std::optional{Interval{0,c.frame.state_extent}};
  }
  bool parseProjected(Parsed& parsed,Use root) {
    auto p=factors(root);if(!p || p->mask)return false;
    std::optional<Use> sum;std::vector<Use> outside;
    for(auto u:p->factors) {
      auto n=read(u);if(!n)return false;
      if(n->kind==Kind::Sum || n->kind==Kind::Add) {
        if(sum)return false;sum=u;
      } else outside.push_back(u);
    }
    return sum && parseStatePieces(parsed,{},*sum,outside,{});
  }
  bool projected(const Context& c,ProductKind kind,Use root) {
    Parsed parsed;
    return parseProjected(parsed,root) && bool(proveGroup(c,kind,parsed.records,{}));
  }
  const Parsed* parseCausal(Use root,CandidateMode mode=CandidateMode::Contraction) {
    const SyntaxKey key{root,mode};
    if(auto found=parsed_roots.find(key);found!=parsed_roots.end())return found->second.get();
    if(!charge() || queryCount()>=limits.max_certificates) {
      stopped=true;return nullptr;
    }
    auto parsed=std::make_unique<Parsed>();
    const auto deny=[&]() -> const Parsed* {parsed_roots.emplace(key,nullptr);return nullptr;};
    auto terms=forest(root);if(!terms)return deny();
    bool entry=false;Parsed projected_entry;
    for(auto term:*terms) {
      if(!term.body) {
        // A stored additive forest can split ONE state projection into
        // several outside-time terms. Keep every original record, then prove
        // their combined exact state coverage once. Identity has no state
        // reduction/coverage theorem, so its existing single-entry guard stays.
        if(mode==CandidateMode::Identity) {
          if(entry || !parseIdentityProduct(*parsed,{},term.use))return deny();
          entry=true;
        } else if(!parseProjected(projected_entry,term.use))return deny();
        continue;
      }
      auto time=bodyScope(term.use);if(!time)return deny();
      if(mode==CandidateMode::Identity) {
        auto n=read(term.use);
        if(!parseIdentityProduct(*parsed,time,{n->children[1],time->body_scope}))return deny();
        continue;
      }
      auto n=read(term.use);auto p=factors({n->children[1],time->body_scope});if(!p)return deny();
      std::optional<Use> state_sum;std::vector<Use> outside;
      for(auto u:p->factors) {
        auto head=read(u);if(!head)return deny();
        if(head->kind==Kind::Sum || head->kind==Kind::Add) {
          if(state_sum)return deny();state_sum=u;
        } else outside.push_back(u);
      }
      if(!state_sum)return deny();
      if(!parseStatePieces(*parsed,time,*state_sum,outside,p->mask)) {
        return deny();
      }
    }
    if(!projected_entry.records.empty()) {
      // One bounded relocation before publishing immutable candidate spans.
      // Covering this group below rejects overlap, gaps and duplicate terms;
      // each actual weight, state source and entry potential is still related.
      constexpr auto record_work=(sizeof(Candidate)+sizeof(ExprId)-1)/sizeof(ExprId);
      const auto total=parsed->records.size()+projected_entry.records.size();
      if(total>limits.max_pieces || !charge(total*record_work+1))return deny();
      const auto begin=parsed->records.size(),group=parsed->groups.size();
      parsed->records.reserve(total);
      for(const auto& candidate:projected_entry.records) {
        parsed->records.push_back(candidate);parsed->records.back().group=group;
      }
      parsed->groups.push_back({{},begin,parsed->records.size()});
    }
    auto pointer=parsed.get();parsed_roots.emplace(key,std::move(parsed));return pointer;
  }
  bool causal(const Context& c,Use root) {
    const auto* parsed=parseCausal(root,c.state_output?CandidateMode::Identity:CandidateMode::Contraction);if(!parsed)return false;
    std::vector<Interval> time_pieces;bool entry=false;
    for(const auto& group:parsed->groups) {
      const auto records=std::span<const Candidate>(parsed->records).subspan(group.begin,group.end-group.begin);
      std::optional<Interval> image;
      if(c.state_output) {
        if(records.size()!=1 || records.front().state_fold)return false;
        const auto& record=records.front();if(!charge(record.factor_count+1))return false;
        Product p;p.factors.assign(record.factors.begin(),record.factors.begin()+record.factor_count);
        if(record.predicate)p.mask=Mask{*record.predicate,record.selected_when_true};
        auto proof=relate(c,group.time?ProductKind::InjectionIdentity:ProductKind::EntryIdentity,group.time,{},std::move(p));
        if(proof)image=group.time?proof->time:proof->state;
      } else image=proveGroup(c,group.time?ProductKind::InjectionProjection:ProductKind::EntryProjection,records,group.time);
      if(!image)return false;
      if(group.time)time_pieces.push_back(*image);else entry=true;
    }
    auto initial=read({c.frame.entry_state,c.frame.state_scope});
    if(!initial || (!entry && initial->kind!=Kind::Zero))return false;
    return covered(time_pieces,c.frame.time);
  }
  std::optional<std::size_t> certify(const Request& r,bool is_region) {
    if(r.projection.owner!=owner || r.projection.index>=projections.size())return {};
    Query query{r.root,r.endpoint,r.projection.index,is_region};
    if(auto f=queries.find(query);f!=queries.end())return f->second;
    if(!charge() || queryCount()>=limits.max_certificates){stopped=true;return {};}
    QueryReservation reservation(pending_queries);
    std::optional<std::size_t> answer;
    const auto core_before=core.statistics().work;
    if(!core.chargesSharedWork() && !charge(2))return {};
    auto state=core.view(r.endpoint);
    auto frame=state?core.view(state->frame):std::optional<prefix::Builder::FrameView>{};
    if(core.chargesSharedWork() && !externalWork(core.statistics().work-core_before))return {};
    if(state && frame && frame->template_state_axis && frame->actual_state_axis &&
        frame->state_extent && state->endpoint>=frame->time.begin && state->endpoint<frame->time.end &&
        projections[r.projection.index].state_extent==frame->state_extent) {
      Context c{state,*frame,projections[r.projection.index],{}, {}, {}, {}};
      if(is_region?causal(c,r.root):projected(c,ProductKind::StateProjection,r.root)) {
        Key key{state->frame,state->endpoint,r.projection.index,{}};
        if(charge(certificates.size()+1)) {
          auto found=std::find(certificates.begin(),certificates.end(),key);
          if(found!=certificates.end())answer=static_cast<std::size_t>(found-certificates.begin());
          else if(certificateCount()<limits.max_certificates) {
            answer=certificates.size();certificates.push_back(key);
          } else stopped=true;
        }
      }
    }
    queries.emplace(query,answer);return answer;
  }
  std::optional<std::size_t> parameterized(const ParameterizedRequest& r,bool dispatch_only) {
    if(r.projection.owner!=owner || r.projection.index>=projections.size())return {};
    ParameterQuery query{r.root,r.output_coordinate,r.source_dispatch,r.source_coordinate,
        r.completed,r.projection.index,dispatch_only};
    if(auto it=parameter_queries.find(query);it!=parameter_queries.end())return it->second;
    if(!charge() || queryCount()>=limits.max_certificates) {
      stopped=true;return {};
    }
    QueryReservation reservation(pending_queries);
    std::optional<std::size_t> answer;
    const auto core_before=core.statistics().work;
    // Completed view performs its checked handle lookup and Frame view (2).
    if(!core.chargesSharedWork() && !charge(2))return {};
    auto completed=core.view(r.completed);
    if(core.chargesSharedWork() && !externalWork(core.statistics().work-core_before))return {};
    if(completed && completed->definition.template_state_axis && completed->definition.actual_state_axis &&
        completed->definition.state_extent &&
        projections[r.projection.index].state_extent==completed->definition.state_extent &&
        r.root.scope==r.output_coordinate.scope && r.source_dispatch.scope==r.source_coordinate.scope &&
        (!dispatch_only || (r.root==r.source_dispatch && r.output_coordinate==r.source_coordinate)) &&
        read(r.root) && read(r.source_dispatch)) {
      OutputTargetRequest request;request.region_root=r.root;request.output_coordinate=r.output_coordinate;
      request.source_dispatch=r.source_dispatch;request.source_coordinate=r.source_coordinate;
      request.completed=r.completed;request.completed_prefix=*completed;
      request.projection=projections[r.projection.index];request.max_work=limits.max_work-stats.work;
      auto proof=output_target?output_target(request):OutputTargetResult{};
      if(proof.work>request.max_work){stopped=true;return {};}
      if(externalWork(proof.work) &&
          proof.status==prefix::RelationStatus::Proved && proof.target.coordinate_token &&
          proof.target.time_role_token && proof.target.original_scope==r.output_coordinate.scope &&
          proof.target.image.begin<proof.target.image.end &&
          proof.target.image.begin>=completed->definition.time.begin &&
          proof.target.image.end<=completed->definition.time.end) {
        const OutputDomain definition{completed->frame,r.projection.index,proof.target};
        if(!charge(output_domains.size()+1))return {};
        auto found_domain=std::find(output_domains.begin(),output_domains.end(),definition);
        std::size_t domain_index=static_cast<std::size_t>(found_domain-output_domains.begin());
        if(found_domain==output_domains.end()) {
          if(certificateCount()>=limits.max_certificates){stopped=true;return {};}
          output_domains.push_back(definition);
        }
        OutputDomainId domain;domain.owner=owner;domain.index=domain_index;
        Context c{{},completed->definition,request.projection,r.completed,proof.target,{},domain};
        if(dispatch_only || causal(c,r.root)) {
          Key key{completed->frame,0,r.projection.index,proof.target,Key::Mode::TimeFunction};
          if(charge(certificates.size()+1)) {
            auto found=std::find(certificates.begin(),certificates.end(),key);
            if(found!=certificates.end())answer=found-certificates.begin();
            else if(certificateCount()<limits.max_certificates) {
              answer=certificates.size();certificates.push_back(key);
            } else stopped=true;
          }
        }
      }
    }
    parameter_queries.emplace(query,answer);return answer;
  }
  std::optional<std::size_t> restriction(const StateRestrictionRequest& r) {
    if(auto found=restriction_queries.find(r);found!=restriction_queries.end())return found->second;
    if(!charge() || queryCount()>=limits.max_certificates || certificateCount()>=limits.max_certificates) {
      stopped=true;return {};
    }
    // Pending Unknown prevents the factory from recursively minting its own
    // prerequisite. No private handle is exposed until both attachments pass.
    restriction_queries.emplace(r,std::nullopt);
    std::optional<std::size_t> answer;
    const auto core_before=core.statistics().work;
    if(!core.chargesSharedWork() && !charge(2))return {};
    auto state=core.view(r.terminal);
    auto frame=state?core.view(state->frame):std::optional<prefix::Builder::FrameView>{};
    if(core.chargesSharedWork() && !externalWork(core.statistics().work-core_before))return {};
    if(state && frame && frame->template_state_axis && frame->actual_state_axis && frame->state_extent &&
        frame->time.begin<frame->time.end && state->endpoint==frame->time.end-1 &&
        r.region_root.scope==r.state_coordinate.scope && r.source_state.scope==r.source_coordinate.scope &&
        read(r.region_root) && read(r.source_state)) {
      StateTargetRequest request{r.region_root,r.state_coordinate,r.source_state,r.source_coordinate,
          r.terminal,*state,*frame,limits.max_work-stats.work,StateTargetMode::PairedRestriction};
      auto proof=state_target?state_target(request):StateTargetResult{};
      if(proof.work>request.max_work){stopped=true;return {};}
      if(externalWork(proof.work) && proof.status==prefix::RelationStatus::Proved && proof.source_target &&
          proof.target.coordinate_token && proof.target.state_role_token &&
          proof.target.original_scope==r.state_coordinate.scope &&
          proof.source_target->original_scope==r.source_coordinate.scope &&
          proof.target.coordinate_token==proof.source_target->coordinate_token &&
          proof.target.state_role_token==proof.source_target->state_role_token &&
          proof.target.image==proof.source_target->image && proof.target.image.begin<proof.target.image.end &&
          proof.target.image.end<=frame->state_extent &&
          charge((sizeof(StateRestriction)+sizeof(ExprId)-1)/sizeof(ExprId))) {
        if(certificateCount()>=limits.max_certificates){stopped=true;return {};}
        answer=state_restrictions.size();state_restrictions.push_back({r,*proof.source_target,proof.target});
      }
    }
    restriction_queries.find(r)->second=answer;return answer;
  }
  std::optional<std::size_t> identity(const IdentityRequest& r,bool source_only) {
    if(r.restriction && (r.restriction->owner!=owner || r.restriction->index>=state_restrictions.size()))return {};
    const IdentityQuery query{r.root,r.state_coordinate,r.source_state,r.source_coordinate,r.terminal,source_only,r.restriction};
    if(auto found=identity_queries.find(query);found!=identity_queries.end())return found->second;
    if(!charge() || queryCount()>=limits.max_certificates){stopped=true;return {};}
    QueryReservation reservation(pending_queries);std::optional<std::size_t> answer;
    const auto core_before=core.statistics().work;
    if(!core.chargesSharedWork() && !charge(2))return {};
    auto state=core.view(r.terminal);
    auto frame=state?core.view(state->frame):std::optional<prefix::Builder::FrameView>{};
    if(core.chargesSharedWork() && !externalWork(core.statistics().work-core_before))return {};
    if(state && frame && frame->template_state_axis && frame->actual_state_axis && frame->state_extent &&
        frame->time.begin<frame->time.end && state->endpoint==frame->time.end-1 &&
        r.root.scope==r.state_coordinate.scope && r.source_state.scope==r.source_coordinate.scope &&
        (!source_only || (r.root==r.source_state && r.state_coordinate==r.source_coordinate)) &&
        read(r.root) && read(r.source_state)) {
      StateTargetResult proof;bool attachment=false;
      if(r.restriction) {
        // Copy before any callback/parse can append immutable records. Queries
        // retain all original uses and scopes; only the semantic component key
        // below uses the checked paired restriction instead of a graph scope.
        if(!charge((sizeof(StateRestriction)+sizeof(ExprId)-1)/sizeof(ExprId)))return {};
        const auto pair=state_restrictions[r.restriction->index];
        attachment=pair.request.terminal==r.terminal && pair.request.source_state==r.source_state &&
            pair.request.source_coordinate==r.source_coordinate &&
            (source_only || (pair.request.region_root==r.root && pair.request.state_coordinate==r.state_coordinate));
        if(attachment) {
          proof.status=prefix::RelationStatus::Proved;proof.target=source_only?pair.source:pair.region;
        }
      } else {
        StateTargetRequest request{r.root,r.state_coordinate,r.source_state,r.source_coordinate,
          r.terminal,*state,*frame,limits.max_work-stats.work};
        proof=state_target?state_target(request):StateTargetResult{};
        if(proof.work>request.max_work){stopped=true;return {};}
        attachment=externalWork(proof.work);
      }
      if(attachment && proof.status==prefix::RelationStatus::Proved &&
          proof.target.coordinate_token && proof.target.state_role_token &&
          proof.target.original_scope==r.state_coordinate.scope &&
          proof.target.image.begin<proof.target.image.end && proof.target.image.end<=frame->state_extent) {
        const OutputTarget encoded{proof.target.coordinate_token,proof.target.state_role_token,
          proof.target.original_scope,proof.target.image};
        const OutputDomain definition{state->frame,0,encoded,Key::Mode::StateFunction};
        if(!charge(output_domains.size()+1))return {};
        auto found=std::find(output_domains.begin(),output_domains.end(),definition);
        const auto index=static_cast<std::size_t>(found-output_domains.begin());
        if(found==output_domains.end()) {
          if(certificateCount()>=limits.max_certificates){stopped=true;return {};}
          output_domains.push_back(definition);
        }
        OutputDomainId domain;domain.owner=owner;domain.index=index;
        Context context{state,*frame,{}, {}, {},proof.target,domain};
        if(source_only || causal(context,r.root)) {
          // Restricted source and region have different ORIGINAL scopes. The
          // SAME private pair denotes exactly their common image; neither
          // certificate grants whole-state/root equality or changes a scope.
          const Key key=r.restriction?
              Key{state->frame,state->endpoint,0,{},Key::Mode::RestrictedStateFunction,r.restriction,r.terminal}:
              Key{state->frame,state->endpoint,0,encoded,Key::Mode::StateFunction};
          if(charge(certificates.size()+1)) {
            auto existing=std::find(certificates.begin(),certificates.end(),key);
            if(existing!=certificates.end())answer=static_cast<std::size_t>(existing-certificates.begin());
            else if(certificateCount()<limits.max_certificates) {
              answer=certificates.size();certificates.push_back(key);
            } else stopped=true;
          }
        }
      }
    }
    identity_queries.emplace(query,answer);return answer;
  }
};
Builder::Builder(prefix::Builder& c,Resolve r,BindBody b,RelateProduct p,
    ProveMask m,std::function<bool(std::size_t)> s,Limits limits,ProveOutputTarget target,ProveStateTarget state):
    impl_(std::make_unique<Impl>(c,std::move(r),std::move(b),std::move(p),std::move(m),std::move(s),limits,std::move(target),std::move(state))) {}
Builder::~Builder()=default;
std::optional<Builder::ProjectionId> Builder::projection(const ProjectionRequest& request) {
  for(std::size_t i=0;i<impl_->projections.size();++i)if(impl_->projections[i]==request) {
    ProjectionId result;result.owner=impl_->owner;result.index=i;return result;
  }
  if(!impl_->charge(impl_->projections.size()+1) || !request.state_extent ||
      request.time_axis==request.state_axis || impl_->projections.size()>=impl_->limits.max_projections ||
      !impl_->read(request.body))return {};
  ProjectionId result;result.owner=impl_->owner;result.index=impl_->projections.size();
  impl_->projections.push_back(request);return result;
}
std::optional<std::span<const Candidate>> Builder::candidates(Use actual_root,CandidateMode mode) {
  const auto* parsed=impl_->parseCausal(actual_root,mode);if(!parsed)return {};
  return std::span<const Candidate>(parsed->records);
}
std::optional<Builder::Certificate> Builder::contraction(const Request& request) {
  auto id=impl_->certify(request,false);if(!id)return {};
  Certificate result;result.owner=impl_->owner;result.index=*id;return result;
}
std::optional<Builder::Certificate> Builder::region(const Request& request) {
  auto id=impl_->certify(request,true);if(!id)return {};
  Certificate result;result.owner=impl_->owner;result.index=*id;return result;
}
std::optional<Builder::Certificate> Builder::parameterizedRegion(const ParameterizedRequest& request) {
  auto id=impl_->parameterized(request,false);if(!id)return {};
  Certificate result;result.owner=impl_->owner;result.index=*id;return result;
}
std::optional<Builder::Certificate> Builder::parameterizedDispatch(const ParameterizedRequest& request) {
  auto id=impl_->parameterized(request,true);if(!id)return {};
  Certificate result;result.owner=impl_->owner;result.index=*id;return result;
}
std::optional<Builder::StateRestrictionId> Builder::stateRestriction(const StateRestrictionRequest& request) {
  auto id=impl_->restriction(request);if(!id)return {};
  StateRestrictionId result;result.owner=impl_->owner;result.index=*id;return result;
}
std::optional<Builder::Certificate> Builder::identityState(const IdentityRequest& request) {
  auto id=impl_->identity(request,true);if(!id)return {};
  Certificate result;result.owner=impl_->owner;result.index=*id;return result;
}
std::optional<Builder::Certificate> Builder::identityRegion(const IdentityRequest& request) {
  auto id=impl_->identity(request,false);if(!id)return {};
  Certificate result;result.owner=impl_->owner;result.index=*id;return result;
}
std::optional<bool> Builder::equal(Certificate a,Certificate b) const {
  if(a.owner!=impl_->owner || b.owner!=impl_->owner || a.index>=impl_->certificates.size() ||
      b.index>=impl_->certificates.size())return {};
  return impl_->certificates[a.index]==impl_->certificates[b.index];
}
Statistics Builder::statistics() const {
  auto result=impl_->stats;
  result.stored_slots=impl_->slots.size();result.queries=impl_->queryCount();
  result.certificates=impl_->certificateCount();return result;
}
bool Builder::exhausted() const {return impl_->stopped || impl_->forests->exhausted() || impl_->linear.exhausted();}
} // namespace sentryward::equivalence::causal_region_certificate
