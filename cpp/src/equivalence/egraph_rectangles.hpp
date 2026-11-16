#pragma once
#include "egraph_engine.hpp"

namespace sentryward::equivalence {
namespace {

inline auto Engine::completeHeaderCharge(std::size_t amount) -> void {
  poll();
  if(options_.max_work && amount>1) {
    if(amount-1>options_.max_work-work_)throw Limit{"work_budget"};
    work_+=amount-1;
  }
}

inline auto Engine::prepareCompleteHeaders(std::size_t bound,BodyHeadIndex& heads) -> bool {
  if(heads.adapter || bound>nodes_.size() || heads.query_remaining)return false;
  if(!complete_headers_) {
    if(!heads.charge(1))return false;
    complete_headers_=std::make_unique<CompleteHeaderCatalog>(*this);
  }
  auto& catalog=*complete_headers_;
  struct RegistrationScope {
    CompleteHeaderCatalog& catalog;
    explicit RegistrationScope(CompleteHeaderCatalog& c,BodyHeadIndex& h):catalog(c){catalog.preparing=&h;}
    ~RegistrationScope(){catalog.preparing=nullptr;}
  } scope(catalog,heads);
  try {
    if(catalog.sources.capacity()<bound) {
      // One bulk scalar allocation for the original physical prefix. The
      // later append path retains amortized geometric growth accounting.
      catalog.charge(bound+catalog.sources.size()+1);catalog.sources.reserve(bound);
    }
    while(catalog.sources.size()<bound) {
      const auto id=catalog.sources.size();catalog.registerSource(id,nodes_[id]);
    }
    return true;
  } catch(const BodyHeadIndex::HeaderBudget&) {return false;}
}

inline auto Engine::rectanglePoll(std::size_t& work,std::size_t amount) -> bool {
  poll();
  if(work>rectangle_limit_ || amount>rectangle_limit_-work) {
    work=rectangle_limit_+1;return false;
  }
  // poll() already accounts for one unit. Other rectangle consumers may
  // have no outer aggregate ledger, so retain global work-budget coverage.
  if(options_.max_work && amount>1) {
    if(amount-1>options_.max_work-work_)throw Limit{"work_budget"};
    work_+=amount-1;
  }
  work+=amount;return true;
}

inline auto Engine::flattenAffineFold(Class id,const Key& root,const std::vector<Key>& snapshot,
    const std::vector<std::vector<Class>>& members,
    std::unordered_map<Class,Class>* import_replacements) -> void {
  // Sum_i Sum_j f(N*i+j) -> Sum_k f(k), only after EVERY dependent
  // index leaf proves this same affine substitution. No logical elements.
  if(root.kind!=static_cast<int>(s::Kind::Fold) || !root.fold ||
      root.fold->reducer!=s::Reducer::Sum || root.fold->count<=1 ||
      !root.attributes.empty() || !root.provenance.empty() ||
      !realType(root.type) || !finite_[find(id)] ||
      !zero_[find(root.children[0])] || flattened_folds_.contains(id))return;
  const auto body=find(root.children[1]);if(body>=members.size())return;
  std::size_t work=0;
  std::optional<Class> inner;
  for(auto candidate:members[body]) {
    if(!rectanglePoll(work))return;
    const auto& h=snapshot[candidate];
    if(h.fold && h.fold->reducer==s::Reducer::Sum && h.fold->count &&
        h.kind==root.kind && h.type==root.type && h.attributes.empty() && h.provenance.empty() &&
        h.fold->binder_level>root.fold->binder_level && zero_[find(h.children[0])]) {
      inner=candidate;break;
    }
  }
  if(!inner)return;const auto nested=snapshot[*inner];
  // Singleton binders already have their own lexical-substitution rule.
  // Do not change that representation while combining nontrivial blocks.
  if(nested.fold->count<=1)return;
  std::uint64_t count;
  if(__builtin_mul_overflow(root.fold->count,nested.fold->count,&count))return;
  const auto outer_axis=root.fold->binder_level,inner_axis=nested.fold->binder_level;
  AffineCache affine;
  struct Plan {Class source;Class head;std::optional<IndexAffine> address;};
  std::vector<Plan> plans;std::map<Class,unsigned char> state;
  std::vector<std::pair<Class,bool>> pending{{find(nested.children[1]),false}};
  bool mapped=false,dense=true;
  while(!pending.empty()) {
    if(!rectanglePoll(work) || state.size()>4096)return;
    auto [node,ready]=pending.back();pending.pop_back();node=find(node);
    if(node>=snapshot.size())return;
    if(node>=members.size() || members[node].empty())return;
    const auto head=members[node].front();
    if(ready){state[node]=2;plans.push_back({node,head,{}});continue;}
    if(auto found=state.find(node);found!=state.end()) {
      if(found->second==1)return;continue;
    }
    if(!dependsOn(node,outer_axis) && !dependsOn(node,inner_axis)){state[node]=2;continue;}
    // A union leader can be a recursive alternative (e.g. 0=x-x).
    // Rebuild the admitted finite head, never unfold the union leader.
    const auto& h=snapshot[head];
    if(h.kind==static_cast<int>(s::Kind::Constant)){state[node]=2;continue;}
    auto index=rectangleAffine(node,snapshot,members,affine,work);
    if(index) {
      const auto a=index->coefficients.contains(outer_axis)?index->coefficients.at(outer_axis):0;
      const auto b=index->coefficients.contains(inner_axis)?index->coefficients.at(inner_axis):0;
      std::uint64_t expected;
      if(__builtin_mul_overflow(b,nested.fold->count,&expected) || a!=expected)dense=false;
      if(!dense && root.fold->count!=2)return;
      mapped|=a!=0 || b!=0;state[node]=2;plans.push_back({node,head,std::move(index)});continue;
    }
    if(h.kind==static_cast<int>(s::Kind::Index) || h.children.empty())return;
    if(h.fold && (h.fold->binder_level==outer_axis || h.fold->binder_level==inner_axis))return;
    if(!rectanglePoll(work,h.children.size()))return;
    state[node]=1;pending.push_back({node,true});
    for(auto child:h.children)pending.push_back({find(child),false});
  }
  if(!mapped)return;
  const auto integer=[&](std::uint64_t n) {Key k;k.kind=static_cast<int>(s::Kind::Constant);
    k.type="exact:u64";k.name=std::to_string(n);return intern(std::move(k));};
  const auto operation=[&](const char* name,Class a,Class b) {
    Key k;k.kind=static_cast<int>(s::Kind::Operation);k.type="exact:u64";
    k.name=name;k.children={a,b};return intern(std::move(k));};
  const auto emit=[&](unsigned block)->std::optional<Class> {
  std::map<Class,Class> replacement;
  for(const auto& plan:plans) {
    if(!rectanglePoll(work,4+(plan.address?plan.address->coefficients.size()*8:
        snapshot[plan.head].children.size()*4)))return {};
    if(plan.address) {
      auto address=*plan.address;
      const auto outer=address.coefficients.contains(outer_axis)?address.coefficients.at(outer_axis):0;
      const auto inner=address.coefficients.contains(inner_axis)?address.coefficients.at(inner_axis):0;
      address.coefficients.erase(outer_axis);
      if(dense) {
        address.coefficients.erase(inner_axis);
        if(inner)address.coefficients.emplace(outer_axis,inner);
      } else address.constant+=block?outer:0; // exact u64 ring arithmetic
      Class value=integer(address.constant);
      for(auto [axis,coefficient]:address.coefficients) {
        Key k;k.kind=static_cast<int>(s::Kind::Index);k.type="exact:u64";k.index_level=axis;
        auto term=intern(std::move(k));
        if(coefficient!=1)term=operation("Multiply",term,integer(coefficient));
        value=operation("Add",value,term);
      }
      replacement.emplace(plan.source,value);
    } else {
      auto k=snapshot[plan.head];
      for(auto& child:k.children) {
        child=find(child);if(auto found=replacement.find(child);found!=replacement.end())child=found->second;
      }
      replacement.emplace(plan.source,intern(std::move(k)));
    }
  }
  auto result=dense?root:nested;if(dense)result.fold->count=count;
  result.children[1]=replacement.at(find(nested.children[1]));
  return intern(std::move(result));
  };
  auto first=emit(0);if(!first)return;
  if(dense) {
    if(import_replacements)import_replacements->emplace(id,*first);
    else rewrite(id,*first);
  }
  else {
    // Fixed binary partition law: two separated BLOCKS remain two Folds.
    // Never iterate over either block's elements or arbitrary outer counts.
    auto second=emit(1);if(!second)return;
    Key sum;sum.kind=static_cast<int>(s::Kind::Operation);sum.type=root.type;
    sum.name="Add";sum.attributes="ideal_arithmetic";
    sum.children={*first,*second};
    auto result=intern(std::move(sum));
    if(import_replacements)import_replacements->emplace(id,result);
    else rewrite(id,result);
  }
  flattened_folds_.insert(id);
}

inline auto Engine::coalesceRectangularFolds(Class id,const Key& root,const std::vector<Key>& snapshot,
    const std::vector<std::vector<Class>>& members,
    const std::map<std::tuple<std::string,std::uint32_t,std::uint64_t>,std::vector<Class>>& targets) -> void {
  // Bounded template certificate: scalar seeds retain their enclosing axes,
  // while bodies acquire exactly one more axis. No new enodes or elements.
  if(!root.provenance.empty() ||
      (!root.attributes.empty() && !(arithmetic(root) && root.name=="Add")))return;
  struct Scope {unsigned parent;RectangleAxis axis;std::uint64_t volume;unsigned depth;};
  struct Task {Class id;unsigned scope;};
  struct Piece {Class body;unsigned scope;};
  std::vector<Scope> scopes{{0,{0,1},1,0}};
  std::map<std::tuple<unsigned,std::uint32_t,std::uint64_t>,unsigned> interned;
  std::vector<Task> pending{{id,0}};std::vector<Piece> pieces;
  std::set<std::pair<Class,unsigned>> visited;
  std::optional<std::uint32_t> binder;
  std::uint64_t total=0;std::size_t work=0;
  while(!pending.empty()) {
    if(!rectanglePoll(work))return;
    auto t=pending.back();pending.pop_back();t.id=find(t.id);
    if(t.id>=members.size() || !finite_[t.id])return;
    if(zero_[t.id])continue;
    if(!visited.emplace(t.id,t.scope).second || visited.size()>4096)return;
    std::optional<Class> fold,add,leaf;
    for(auto member:members[t.id]) {
      if(!rectanglePoll(work))return;
      const auto& k=snapshot[member];
      if(k.type!=root.type || !k.provenance.empty())continue;
      if(k.kind==static_cast<int>(s::Kind::Fold) && k.attributes.empty() &&
          k.fold->reducer==s::Reducer::Sum && k.children.size()==2){fold=member;break;}
      if(arithmetic(k) && k.name=="Add" && !add)add=member;
      if(!leaf)leaf=member;
    }
    if(fold) {
      const auto& k=snapshot[*fold];const auto count=k.fold->count;
      pending.push_back({k.children[0],t.scope});
      if(!count)continue;
      if(scopes[t.scope].depth>=8 || (t.scope && k.fold->binder_level<=scopes[t.scope].axis.binder))return;
      if(count>UINT64_MAX/scopes[t.scope].volume)return;
      binder=binder?std::min(*binder,k.fold->binder_level):k.fold->binder_level;
      const auto key=std::tuple{t.scope,k.fold->binder_level,count};
      auto found=interned.find(key);unsigned child;
      if(found==interned.end()) {
        if(scopes.size()>=128)return;
        child=static_cast<unsigned>(scopes.size());interned.emplace(key,child);
        scopes.push_back({t.scope,{k.fold->binder_level,count},scopes[t.scope].volume*count,scopes[t.scope].depth+1});
      } else child=found->second;
      pending.push_back({k.children[1],child});
    } else if(add)for(auto child:snapshot[*add].children)pending.push_back({child,t.scope});
    else {
      if(!leaf || pieces.size()>=1024 || scopes[t.scope].volume>UINT64_MAX-total)return;
      total+=scopes[t.scope].volume;pieces.push_back({t.id,t.scope});
    }
  }
  if(!binder || !total || pieces.empty())return;
  const auto candidates=targets.find({root.type,*binder,total});if(candidates==targets.end())return;
  // Only a complete matching-volume stored forest receives this ledger.
  // No logical Fold element is visited. Keep small unrelated proofs at the
  // ordinary budget; this allowance is local to this coverage certificate.
  struct BudgetScope {
    std::size_t& slot;std::size_t previous;
    ~BudgetScope(){slot=previous;}
  } budget_scope{rectangle_limit_,rectangle_limit_};
  rectangle_limit_=std::min<std::size_t>(8388608,
      rectangle_work_limit+pieces.size()*8192);
  AffineCache cache;
  std::size_t head_remaining=0;
  BodyHeadIndex witness_heads(*this,snapshot,members,cache,&head_remaining);
  for(auto target:candidates->second) {
    if(!rectanglePoll(work))return;
    if(find(target)==find(id) || free_indices_[find(target)]!=free_indices_[find(id)])continue;
    struct Domain {
      std::uint64_t begin,stride,count;
      std::vector<std::pair<std::uint64_t,std::uint64_t>> digits;
    };
    std::vector<Domain> domains;bool valid=true;std::uint64_t lattice=1;
    for(const auto& piece:pieces) {
      std::vector<RectangleAxis> axes;
      for(auto scope=piece.scope;scope;scope=scopes[scope].parent)axes.push_back(scopes[scope].axis);
      std::reverse(axes.begin(),axes.end());
      auto map=rectangleBodyMap(snapshot[target].children[1],piece.body,*binder,axes,snapshot,members,cache,work,&witness_heads);
      if(!map || map->begin>=total){
        valid=false;break;
      }
      std::vector<std::pair<std::uint64_t,std::uint64_t>> strides;
      for(std::size_t i=0;i<axes.size();++i)if(axes[i].count>1)strides.emplace_back(map->coefficients[i],axes[i].count);
      std::sort(strides.begin(),strides.end());
      const auto step=strides.empty()?std::uint64_t{1}:strides.back().first;
      if(!step){valid=false;break;}
      // A rectangle can have holes: e.g. 256*i + 32*j + k,
      // i,j in [0,3), k in [0,32). Do not require dense AP flattening.
      // Superincreasing byte/index strides give a unique bounded digit
      // decomposition. The largest axis stays an interval; smaller axes
      // are tested symbolically at a bounded residue, never enumerated.
      std::uint64_t span=0;
      for(auto [stride,count]:strides) {
        std::uint64_t part;
        if(stride<=span || __builtin_mul_overflow(stride,count-1,&part) ||
            __builtin_add_overflow(span,part,&span)){valid=false;break;}
      }
      if(!valid)break;
      const auto count=strides.empty()?std::uint64_t{1}:strides.back().second;
      std::uint64_t next;
      if(span>=total-map->begin ||
          __builtin_mul_overflow(lattice/std::gcd(lattice,step),step,&next) || next>4096)
        {valid=false;break;}
      if(!strides.empty())strides.pop_back();
      lattice=next;domains.push_back({map->begin,step,count,std::move(strides)});
    }
    if(work>rectangle_limit_)return;
    if(!valid)continue;
    // A rectangular block may collapse to a strided AP, not only a dense
    // interval. Compare all APs in a bounded common residue lattice using
    // the existing exact weighted-interval engine. Work depends on stored
    // pieces and <=4096 residues, never their logical iteration counts.
    bounded_linear_reduction::Limits limits;
    limits.max_work=rectangle_limit_;
    bounded_linear_reduction::Builder coverage(limits,[&](std::size_t amount){return rectanglePoll(work,amount);});
    auto one=coverage.coefficient("1");if(!one)return;
    using IntervalShape=std::vector<std::pair<std::uint64_t,std::uint64_t>>;
    std::set<std::pair<IntervalShape,IntervalShape>> certified_residues;
    for(std::uint64_t residue=0;residue<lattice && valid;++residue) {
      std::vector<bounded_linear_reduction::WeightedInterval> actual,required;
      if(residue<total)required.push_back({0,1+(total-1-residue)/lattice,*one});
      for(const auto& d:domains) {
        if(!rectanglePoll(work,2+2*d.digits.size()))return;
        const auto delta=(residue+lattice-d.begin%lattice)%lattice;
        auto remainder=delta%d.stride;bool covered=true;
        for(auto it=d.digits.rbegin();it!=d.digits.rend();++it) {
          const auto digit=remainder/it->first;
          if(digit>=it->second){covered=false;break;}
          remainder%=it->first;
        }
        if(!covered || remainder)continue;
        const auto first=delta/d.stride;if(first>=d.count)continue;
        const auto start=(d.begin+delta)/lattice;
        const auto count=1+(d.count-1-first)/(lattice/d.stride);
        actual.push_back({start,start+count,*one});
      }
      // Adjacent residues frequently ask exactly the same interval proof.
      // Reuse that proof, not a representative sample of a different set.
      IntervalShape actual_shape,required_shape;
      for(const auto& interval:actual)actual_shape.emplace_back(interval.begin,interval.end);
      for(const auto& interval:required)required_shape.emplace_back(interval.begin,interval.end);
      auto signature=std::pair{std::move(actual_shape),std::move(required_shape)};
      if(!rectanglePoll(work,(1+actual.size()+required.size())*
          (1+std::bit_width(certified_residues.size()+1))))return;
      if(certified_residues.contains(signature))continue;
      auto equal=coverage.intervalsEqual(actual,required);
      if(!equal || !*equal)valid=false;
      else certified_residues.insert(std::move(signature));
    }
    if(valid){rewrite(id,target);coalesced_forests_.insert(id);return;}
  }
}

} // namespace
} // namespace sentryward::equivalence
