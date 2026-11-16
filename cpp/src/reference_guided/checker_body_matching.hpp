#pragma once
#include "checker.hpp"

namespace rg {

// Reconstruct the reference body under a candidate index correspondence.
// Opaque leaves must already exist in this Arena: no facts are transferred
// from a differently indexed reference read.
// twin_target != invalid: the instantiation of one candidate's body against that target (twin mode).
// A read the Arena does not hold is the target's read at the same storage element (local_twin_at).
// In twin mode, or with build_max, a maximum the Arena does not hold is built by its constructor (an
// exact value); nothing else changes. These instances are neither cached nor read from the cache.
inline auto Checker::instantiate_body(Id root, const std::vector<Id> &cap, Id twin_target, bool build_max) -> Id {
  build_max |= twin_target != invalid;
  auto eligibility=body_instance_cacheable.find(root);
  const bool reusable=!build_max && (eligibility==body_instance_cacheable.end() || eligibility->second);
  if(reusable) if(auto hit=body_instance_cache.find({root,cap});hit!=body_instance_cache.end()) {
    ++a.stats["ep_body_instance_cache_hits"];return hit->second;
  }
  bool encountered_call=false;
  std::unordered_map<Id,Id> memo;
  std::function<Id(Id,unsigned)> visit = [&](Id id,unsigned depth)->Id {
    if(depth>128 || memo.size()>2048) return invalid;
    if(auto it=memo.find(id);it!=memo.end()) return it->second;
    const auto &n=patterns.nodes[id];
    auto slot=[&](size_t j){return j<cap.size()?cap[j]:invalid;};
    auto index=[&](size_t j){return n.data[j].numerator().convert_to<size_t>();};
    Id out=invalid;
    if(n.op=="slot") return slot(index(0));
    if(n.op=="affine_slot" || n.op=="affine_param") {
      Id axis=slot(index(0)); if(axis==invalid)return invalid;
      if(n.op=="affine_slot") out=a.add({a.mul({a.constant(n.data[1]),axis}),a.constant(n.data[2])});
      else {Id residual=slot(index(1));if(residual==invalid)return invalid;
        out=a.add({a.mul({a.constant(n.data[2]),axis}),residual});}
    } else if(n.op=="derived_coordinate") {
      Id axis=slot(index(0));if(axis==invalid)return invalid;
      out=CoordinateRelations::evaluate(a,axis,n.data);
    } else if(const Id decided=n.op=="select" && n.args.size()==3 && n.data.empty() && n.tag.empty()
                               ? visit(n.args[0],depth+1) : invalid;
              decided!=invalid && a.value(decided)) {
      // Rule (decided arm of an instantiated Select): a condition instantiated to a constant selects
      // one arm, which is the Select's value; the other arm is not instantiated (a concatenation's
      // reads of the other source at this index need not exist). Exact.
      ++a.stats["ep_body_decided_select_arms"];
      out=visit(n.args[*a.value(decided)!=0?1:2],depth+1);
    } else {
      std::vector<Id> xs;for(Id child:n.args){Id v=visit(child,depth+1);if(v==invalid)return invalid;xs.push_back(v);}
      if(n.op=="coordinate_sum") {
        std::vector<Id> terms{a.constant(n.data[2])};
        if(n.data[1]>=0){Id v=slot(index(1));if(v==invalid)return invalid;terms.push_back(v);}
        for(size_t j=0;j<xs.size();++j)terms.push_back(a.mul({a.constant(n.data[3+j]),xs[j]}));
        out=a.add(terms);
      } else if(n.op=="const") out=a.constant(n.data[0]);
      else if(n.op=="add" || n.op=="add_raw") {
        for(size_t j=0;j<xs.size();++j)xs[j]=a.mul({a.constant(n.data[j]),xs[j]});
        out=a.add(xs);
      } else if(n.op=="mul") {xs.push_back(a.constant(n.data[0]));out=a.mul(xs);}
      else if(n.op=="idiv" || n.op=="imod") {
        // Captured indices may simplify a reference quotient/remainder to
        // a different operator (or a literal). Reuse the exact coordinate
        // constructors, just as instantiated_coordinate() does for the
        // candidate check; an unsimplified node need not exist in Arena.
        if(xs.size()!=1 || n.data.size()!=1 ||
           n.data[0].denominator()!=1 || n.data[0]<=0) return invalid;
        out=n.op=="idiv" ? a.idiv(xs[0],n.data[0].numerator())
                         : a.imod(xs[0],n.data[0].numerator());
        ++a.stats["ep_body_coordinate_rebuilds"];
      }
      else if(n.op=="signed") {
        if(xs.size()!=1 || n.data.size()!=1 || n.data[0].denominator()!=1 ||
           n.data[0]<=0 || n.data[0]>256) return invalid;
        out=a.signed_word(xs[0],n.data[0].numerator().convert_to<unsigned>());
      }
      else if(n.op=="word_or" && xs.size()==2 && n.data.size()==1)
        out=a.word_or(xs[0],xs[1],n.data[0].numerator().convert_to<unsigned>());
      else if(n.op=="word_shr" && xs.size()==2 && n.data.size()==2)
        out=a.word_shr(xs[0],xs[1],n.data[0].numerator().convert_to<unsigned>(),n.data[1]!=0);
      else if(n.op=="word_shl" && xs.size()==2 && n.data.size()==1)
        out=a.word_shl(xs[0],xs[1],n.data[0].numerator().convert_to<unsigned>());
      else if(n.op=="extractbits" && xs.size()==1 && n.data.size()==2)
        out=a.bit_slice(xs[0],n.data[0].numerator().convert_to<unsigned>(),n.data[1].numerator().convert_to<unsigned>());
      else if(n.op=="exp" || n.op=="exp2")out=a.exponential(xs[0],n.op=="exp2");
      else if(n.op=="recip")out=a.divide(a.one,xs[0]);
      else if(n.op=="compare") {
        // Candidate substitution can prove a predicate on a narrower binder
        // domain. Derive the result from the actual operands, not facts
        // attached to the reference pattern or an opaque-node lookup.
        if(xs.size()!=2 || !n.data.empty() ||
           (n.tag!="eq" && n.tag!="ne" && n.tag!="lt" && n.tag!="le" &&
            n.tag!="gt" && n.tag!="ge"))return invalid;
        out=a.compare(n.tag,xs[0],xs[1]);
        ++a.stats["ep_body_predicate_rebuilds"];
      }
      else if(n.op=="select") {
        if(xs.size()!=3 || !n.data.empty() || !n.tag.empty())return invalid;
        out=a.select(xs[0],xs[1],xs[2]);
        ++a.stats["ep_body_select_rebuilds"];
      }
      else if(n.op=="call") {
        encountered_call=true;
        // A captured parameter changes the instantiated definition, not
        // only a Call's argument list. Reuse the same exact definition
        // canonicalizer as registration and ordinary reconstruction.
        if(n.data.size()!=1 || n.data[0].denominator()!=1 ||
           n.data[0]<0 || n.data[0]>=Q(specs.size()))return invalid;
        Id sid=n.data[0].numerator().convert_to<Id>();
        if(xs.size()+1!=specs[sid].slots)return invalid;
        out=call(sid,xs);
        ++a.stats["ep_body_call_rebuilds"];
      }
      else {
        if(auto hit=a.find_node(n.op,xs,n.data,n.tag))out=*hit;
        else if(twin_target!=invalid && (n.op=="read" || n.op=="readbits" || n.op=="packedbits")) {
          out=local_twin_at(twin_target,ReadKind{n.op,n.tag,n.data},xs);
          if(out!=invalid)++a.stats["ep_body_twin_reads"];
        } else if(build_max && n.op=="max" && !xs.empty() && n.data.empty() && n.tag.empty()) {
          out=a.maximum(xs);++a.stats["ep_body_twin_maxima"];
        }
        else ++a.stats["ep_body_missing_opaque_"+n.op];
      }
    }
    memo.emplace(id,out);return out;
  };
  Id result=visit(root,0);
  if(build_max)return result;
  // A successful call-free construction depends only on immutable template
  // and capture IDs. Missing opaque nodes can become available later, so
  // failures are never cached. Call construction also registers active
  // definitions and must retain that side effect on every invocation.
  if(encountered_call)body_instance_cacheable[root]=false;
  else if(result!=invalid) {
    body_instance_cacheable[root]=true;
    body_instance_cache.emplace(BodyInstanceKey{root,cap},result);
    ++a.stats["ep_body_instance_cache_entries"];
  }
  return result;
}

// Whether the node `read` occurs in the scope of target's anchor inventory: exactly membership in
// that inventory had no discovery limit cut it (the same node identity test as the inventory
// lookup). Only a truncated inventory needs the index; an untruncated one is that set already.
// The first query for a truncated target builds its index in O(scoped nodes + edges) of the
// target; every lookup is a binary search, O(log reads).
inline auto Checker::target_scope_contains_read(Id target, Id read) -> bool {
  if(!truncated_target_anchors.contains(target))return false;
  auto found=target_read_index.find(target);
  if(found==target_read_index.end()) {
    if(read_index_marks.size()<a.nodes.size())read_index_marks.resize(a.nodes.size(),0);
    if(++read_index_build==0) {  // the build numbers wrapped: no stale mark may match
      std::fill(read_index_marks.begin(),read_index_marks.end(),0);read_index_build=1;
    }
    const auto build=read_index_build;
    const auto first_visit=[&](Id id) {
      if(read_index_marks[id]==build)return false;
      read_index_marks[id]=build;return true;
    };
    std::vector<Id> reads,todo{target};first_visit(target);
    size_t visited=0;
    while(!todo.empty()) {
      if((++visited&1023)==0)a.check();
      const Id id=todo.back();todo.pop_back();
      const auto &n=a.nodes[id];
      if(n.op=="read" || n.op=="readbits" || n.op=="packedbits"){reads.push_back(id);continue;}
      if(n.op=="slot" || n.op=="var" || n.op=="call" || n.op=="certified_state")continue;
      const size_t children=n.op=="fold"?std::min<size_t>(1,n.args.size()):n.args.size();
      for(size_t j=0;j<children;++j)if(first_visit(n.args[j]))todo.push_back(n.args[j]);
    }
    std::sort(reads.begin(),reads.end());
    a.stats["ep_body_target_read_index_nodes"]+=visited;
    ++a.stats["ep_body_target_read_index_builds"];
    found=target_read_index.emplace(target,std::move(reads)).first;
  }
  return std::binary_search(found->second.begin(),found->second.end(),read);
}

// A factor whose EP normal form is not the zero polynomial: a nonzero constant, an opaque atom or
// an exponential (a monomial times nonzero factors), a product of such, or a sum or logarithm
// whose value is proved nonzero (its normal form is exact, so it is not zero either).
inline auto Checker::nonzero_normal_form(Id id,size_t &visits) -> bool {
  if(++visits>required_call_visits)return false;
  const auto &n=a.nodes[id];
  if(n.op=="const")return n.data.size()==1 && n.data[0]!=0;
  if(n.op=="mul") {
    if(n.data.size()!=1 || n.data[0]==0)return false;
    for(Id f:n.args)if(!nonzero_normal_form(f,visits))return false;
    return true;
  }
  if(n.op=="add" || n.op=="add_raw" || n.op=="log" || n.op=="log2")return a.nonzero(id);
  return true;
}

inline auto Checker::missing_required_call(Id x,Id target) -> bool {
  // Calls in the linear shell of X.
  std::vector<Id> shell,todo{x};
  for(size_t visits=0;!todo.empty();) {
    if(++visits>required_call_visits)return false;
    const Id id=todo.back();todo.pop_back();
    const auto &n=a.nodes[id];
    if(n.op=="call"){if(std::find(shell.begin(),shell.end(),id)==shell.end())shell.push_back(id);}
    else if(n.op=="mul" || n.op=="add" || n.op=="add_raw")todo.insert(todo.end(),n.args.begin(),n.args.end());
  }
  if(shell.empty())return false;
  const auto &present=target_scope_calls(target);
  size_t candidates=0;
  for(Id call:shell) {
    if(++candidates>required_call_candidates)return false;
    const auto same=[&](Id other) {
      return other==call || (a.nodes[other].op=="call" && a.nodes[other].data==a.nodes[call].data &&
                             scalar.alpha.equal(other,call));
    };
    if(std::any_of(present.begin(),present.end(),same))continue;
    // holds(id): a node alpha-equal to the Call occurs in the DAG of id (any operator).
    std::unordered_map<Id,bool> memo;size_t visits=0;bool bounded=true;
    std::function<bool(Id)> holds=[&](Id id)->bool {
      if(auto it=memo.find(id);it!=memo.end())return it->second;
      if(++visits>required_call_visits){bounded=false;return true;}
      bool result=same(id);
      if(!result)for(Id child:std::vector<Id>(a.nodes[id].args))if(holds(child)){result=true;break;}
      memo.emplace(id,result);return result;
    };
    Id cursor=x;bool unique=true;
    while(unique && cursor!=call) {
      const auto &n=a.nodes[cursor];
      const bool product=n.op=="mul" && n.data.size()==1 && n.data[0]!=0;
      const bool sum=(n.op=="add" || n.op=="add_raw") && n.args.size()==n.data.size();
      if(!product && !sum){unique=false;break;}
      const auto args=n.args;const auto data=n.data;
      size_t holder=args.size();
      for(size_t j=0;j<args.size() && unique;++j)if(holds(args[j])) {
        if(holder!=args.size())unique=false;
        holder=j;
      }
      if(!bounded || !unique || holder==args.size() || (sum && data[holder]==0)){unique=false;break;}
      if(product)for(size_t j=0;j<args.size() && unique;++j)
        if(j!=holder && !nonzero_normal_form(args[j],visits))unique=false;
      cursor=args[holder];
    }
    if(unique && bounded)return true;
  }
  return false;
}

inline auto Checker::respelled_call_captures(Id pat,Id target,const std::vector<Id> &cap,size_t &used) -> Captures {
  const auto &pn=patterns.nodes[pat];
  if(pn.op!="call" || pn.data.size()!=1 || pn.data[0].denominator()!=1 || pn.data[0]<0 ||
     pn.data[0]>=Q(specs.size()))return {};
  const Id sid=pn.data[0].numerator().convert_to<Id>();
  const auto targs=a.nodes[target].args;  // a copy: call() may grow the node store
  const size_t k=pn.args.size(),n=targs.size();
  if(k+1!=specs[sid].slots || n>8 || k>8 || (n>=k?n-k:k-n)>2)return {};
  if(n==k && a.nodes[target].data==pn.data)return {};  // the same spelling: the ordinary match
  Captures out;
  const auto attempt=[&](const std::vector<Id> &list) {
    if(used==respelling_limit)return;
    ++used;++a.stats["ep_body_call_respelling_attempts"];
    if(call(sid,list)!=target)return;
    const auto raw=a.find_node("call",list,pn.data,"");
    if(!raw || *raw==target)return;
    for(auto &c:matches(pat,*raw,cap))if(std::find(out.begin(),out.end(),c)==out.end())out.push_back(std::move(c));
  };
  if(n>=k) {
    const auto without=[&](size_t first,size_t second) {  // the dropped positions (n when unused)
      std::vector<Id> kept;
      for(size_t j=0;j<n;++j)if(j!=first && j!=second)kept.push_back(targs[j]);
      attempt(kept);
    };
    if(n==k)without(n,n);
    else if(n==k+1)for(size_t i=0;i<n;++i)without(i,n);
    else for(size_t i=0;i<n;++i)for(size_t j=i+1;j<n;++j)without(i,j);
    return out;
  }
  // n < k: the lists of length k that repeat k - n of T's arguments. Already captured slot
  // arguments of the pattern Call select among them before anything is instantiated.
  const auto agrees=[&](const std::vector<Id> &list) {
    for(size_t i=0;i<k;++i) {
      const auto &arg=patterns.nodes[pn.args[i]];
      if(arg.op!="slot" || arg.data.size()!=1 || arg.data[0].denominator()!=1 || arg.data[0]<0)continue;
      const auto slot=arg.data[0].numerator().convert_to<size_t>();
      if(slot<cap.size() && cap[slot]!=invalid && cap[slot]!=list[i])return false;
    }
    return true;
  };
  std::vector<std::vector<Id>> tried;
  const auto repeated=[&](const std::vector<Id> &list) {
    if(list.size()!=k || !agrees(list) || std::find(tried.begin(),tried.end(),list)!=tried.end())return;
    tried.push_back(list);attempt(list);
  };
  // The enumeration ends with the instantiation limit: no list is built after it.
  const auto insertions=[&](const std::vector<Id> &base,const auto &then) {
    for(size_t position=0;position<=base.size();++position)for(size_t source=0;source<n;++source) {
      if(used==respelling_limit)return;
      auto list=base;list.insert(list.begin()+std::ptrdiff_t(position),targs[source]);
      then(list);
    }
  };
  if(k==n+1)insertions(targs,repeated);
  else insertions(targs,[&](const std::vector<Id> &once){insertions(once,repeated);});
  return out;
}

// Anchor matching proposes coordinates only; ScalarEqual checks the entire
// body. Bounded DAG traversal never enumerates the reduction domain.
inline auto Checker::body_candidates(const Spec &s,Id target,const std::vector<Id> &initial) -> Captures {
  if(initial.size()!=s.slots)return {};
  auto anchors=[&](Arena &arena,Id root,bool pattern)->const std::vector<Id>&{
    auto &cache=pattern?pattern_anchor_cache:target_anchor_cache;
    if(auto found=cache.find(root);found!=cache.end()) {
      ++a.counter(pattern?Arena::Hot::ep_body_pattern_anchor_cache_hits:Arena::Hot::ep_body_target_anchor_cache_hits);
      return found->second;
    }
    // Coordinates in a direct operand must not be starved by one sibling's
    // numerical history. Traverse stored nodes breadth first; anchor nodes
    // remain opaque, and all proposals still require a complete body proof.
    // Bound discovery itself, including wide operand arrays and the queue.
    std::vector<Id> out,todo{root};std::unordered_set<Id> seen{root};
    size_t cursor=0,edges=0;bool limited=false;
    while(cursor<todo.size() && out.size()<anchor_limit){
      Id id=todo[cursor++];
      const auto &n=arena.nodes[id];
      if(n.op=="slot" || n.op=="var" || n.op=="read" || n.op=="readbits" || n.op=="packedbits" || n.op=="call" || n.op=="certified_state")out.push_back(id);
      else {
        // A nested Fold owns its body's coordinates. Discovering a read
        // there would bind this scalar query to a bound member of an entire
        // reduction. The reduction consumer handles that body separately;
        // only the seed remains in the current lexical scope. Registered
        // Calls above are already opaque and retain delayed rebinding.
        const bool binder=n.op=="fold";
        if(binder)++a.stats["ep_body_fold_scope_boundaries"];
        const size_t children=binder?std::min<size_t>(1,n.args.size()):n.args.size();
        for(size_t j=0;j<children;++j){
          Id child=n.args[j];
          if(edges==anchor_edge_limit){limited=true;break;}
          ++edges;
          if(seen.contains(child))continue;
          if(seen.size()==anchor_node_limit){limited=true;continue;}
          seen.insert(child);todo.push_back(child);
        }
      }
    }
    a.stats["ep_body_anchor_nodes"]+=cursor;
    a.stats["ep_body_anchor_edges"]+=edges;
    if(limited || cursor<todo.size()) {
      ++a.stats["ep_body_anchor_limited"];
      if(!pattern)truncated_target_anchors.insert(root);
    }
    if(pattern)std::stable_partition(out.begin(),out.end(),[&](Id id){
      const auto &op=arena.nodes[id].op;return op!="slot" && op!="var";
    });
    return cache.emplace(root,std::move(out)).first->second;
  };
  const auto &ps=anchors(patterns,s.pattern,true), &ts=anchors(a,target,false);
  // Reads/calls bind coordinates even when a bounds-proved predicate has
  // disappeared from the actual body. A redundant bare slot is not another
  // required occurrence of that coordinate. This orders candidate discovery,
  // never proof: instantiate_body + ScalarEqual still check the whole body.
  Captures caps{initial};
  size_t attempts=0,respellings=0;
  // Rule (anchor order): in a search with fixed reduction parameters, an anchor that discovers
  // nothing while a slot is still unbound in some capture is tried once more after the
  // remaining anchors, which may bind that slot (a gathered row whose index is read through another
  // input anchor). Every anchor still has to be met once by every proposed capture; candidates
  // only, never a proof. (Fixed parameters only: an open search would retry nearly every anchor.)
  const bool retry_anchors=std::any_of(initial.begin(),initial.end(),[](Id x){return x!=invalid;});
  std::vector<Id> order(ps.begin(),ps.end());
  std::unordered_set<Id> retried;
  for(size_t position=0;position<order.size();++position){
    const Id pat=order[position];
    Captures next;
    const auto &pn=patterns.nodes[pat];
    for(const auto &cap:caps){
      // Only abstract anchors may have a different registered spelling of
      // the same instantiated definition. Direct input anchors still have
      // to agree on the object, typed view and captured coordinates, even
      // when an earlier read has already bound every slot. Otherwise a
      // proper factor of a body can trigger a full EP proof of that body.
      const bool deferred_definition = pn.op=="call" || pn.op=="certified_state";
      if(deferred_definition && std::find(cap.begin(),cap.end(),invalid)==cap.end()) {
        ++a.stats["ep_body_deferred_call_anchors"];
        next.push_back(cap);continue;
      }
      if(pn.op=="var" || (pn.op=="slot" && pn.data.size()==1 &&
          pn.data[0].denominator()==1 && pn.data[0]>=0 && pn.data[0]<Q(cap.size()) &&
          cap[pn.data[0].numerator().convert_to<size_t>()]!=invalid)) {
        next.push_back(cap);continue;
      }
      if(cacheable_read_anchor(pat) &&
         std::find(cap.begin(),cap.end(),invalid)==cap.end()) {
        Id expected=instantiate_body(pat,cap);
        // The inventory ts may be cut by a discovery limit. Then presence is decided by the
        // target's exact read index instead (same node identity); absence still falls back.
        const bool listed=expected!=invalid && std::find(ts.begin(),ts.end(),expected)!=ts.end();
        if(listed || (expected!=invalid && target_scope_contains_read(target,expected))) {
          const auto validated=match_anchor(pat,expected,cap);
          if(std::find(validated.begin(),validated.end(),cap)!=validated.end()) {
            ++a.stats[listed?"ep_body_bound_read_exact_hits":"ep_body_bound_read_index_hits"];
            next.push_back(cap);continue;
          }
        }
        // A miss is not a failed constraint. The general anchor matcher
        // remains available; no unchecked reconstruction is accepted.
        ++a.stats["ep_body_bound_read_matcher_fallbacks"];
      } else if(std::find(cap.begin(),cap.end(),invalid)==cap.end() && plain_read_anchor(pat,&cap)) {
        // Rule (bound plain read anchor): with every slot captured, an input anchor whose
        // coordinates are ordinary operations over slots and constants (no derived, affine or
        // summed coordinate recipe, whose scope guards only the matcher checks) is the exact
        // substitution of the captures. It is met when the target holds that very read node
        // (same identity, or the exact read index of a cut inventory). Candidate only: the
        // complete body proof follows. A miss falls back to the general matcher as before.
        const Id expected=instantiate_body(pat,cap);
        const bool listed=expected!=invalid && std::find(ts.begin(),ts.end(),expected)!=ts.end();
        if(listed || (expected!=invalid && target_scope_contains_read(target,expected))) {
          ++a.stats["ep_body_bound_plain_read_hits"];
          next.push_back(cap);continue;
        }

      } else if(scoped_read_anchor(pat,cap)) {
        // Coordinates that combine derived digits arithmetically (a packed nibble address)
        // are not matched structurally. With every captured axis an integer inside each
        // digit's recorded scope, their exact instantiation names one target read or none.
        // One uncaptured slot that is a plain term of the address is solved exactly from a
        // target address (integer, then re-instantiated). Candidates only; the body proof
        // remains the complete EP equality. The general matcher's budgets apply.
        const auto candidates=scoped_read_captures(pat,cap,ts,attempts);
        if(!candidates){++a.stats["ep_body_candidate_budget"];return {};}
        bool hit=false;
        for(const auto &candidate:*candidates) {
          Id expected=instantiate_body(pat,candidate);
          if(expected!=invalid && std::find(ts.begin(),ts.end(),expected)!=ts.end()) {
            ++a.stats["ep_body_bound_read_scoped_hits"];
            if(std::find(next.begin(),next.end(),candidate)==next.end()) {
              next.push_back(candidate);
              if(next.size()>body_candidate_limit){++a.stats["ep_body_candidate_budget"];return {};}
            }
            hit=true;
          }
        }
        if(hit)continue;
      }
      bool discovered=false;
      for(Id t:ts){
      if(++attempts>body_candidate_attempts){++a.stats["ep_body_candidate_budget"];return {};}
      const auto &tn=a.nodes[t];
      if(pn.op=="slot" ? tn.op!="var" : (pn.op!=tn.op || pn.tag!=tn.tag))continue;
      for(auto c:match_anchor(pat,t,cap)) {
        discovered=true;
        if(std::find(next.begin(),next.end(),c)==next.end()) {
          next.push_back(std::move(c));
          if(next.size()>body_candidate_limit){++a.stats["ep_body_candidate_budget"];return {};}}
      }
      }
      // A Call anchor that no target Call matches as spelled may name one under another
      // registered spelling of the same instantiated definition (respelled_call_captures). These
      // captures are added to the delayed proposal below, which is kept as before; they leave
      // room for it under body_candidate_limit.
      // Targets spelled with at least as many arguments first, then those with fewer, so the
      // latter only use what the former leave of the instantiation limit.
      if(!discovered && pn.op=="call") for(const bool fewer:{false,true}) for(Id t:ts) {
        if(a.nodes[t].op!="call" || (a.nodes[t].args.size()<pn.args.size())!=fewer)continue;
        for(auto &c:respelled_call_captures(pat,t,cap,respellings))
          if(std::find(next.begin(),next.end(),c)==next.end() && next.size()+1<body_candidate_limit) {
            next.push_back(std::move(c));
            ++a.stats["ep_body_respelled_call_anchors"];
          }
      }
      // A Call can be rebound only after independent anchors supply its
      // actual parameters. Keep that delayed proposal, not arbitrary failed
      // read proposals. This is bounded candidate discovery, not a proof of
      // inequality: unsupported/absent anchors decline this search path;
      // every admitted candidate still requires the complete body proof.
      if(!discovered && !deferred_definition &&
          (pn.op=="read" || pn.op=="readbits" || pn.op=="packedbits"))
        ++a.counter(Arena::Hot::ep_body_read_anchor_rejections);
      if(!discovered && deferred_definition && std::find(next.begin(),next.end(),cap)==next.end()) {
        next.push_back(cap);
        ++a.stats["ep_body_deferred_unmatched_anchors"];
        ++a.stats["ep_body_deferred_call_anchors"];
        if(next.size()>body_candidate_limit){++a.stats["ep_body_candidate_budget"];return {};}
      }
    }
    // Rule (anchor in a decided arm): a capture whose instantiated Select conditions exclude every
    // occurrence of the anchor (a concatenation's arm for the other source, decided by the captured
    // index) need not meet it and is kept as it is. Candidates only, never a proof.
    for(const auto &cap:caps)
      if(std::find(next.begin(),next.end(),cap)==next.end() && excluded_anchor(s.pattern,pat,{cap})) {
        next.push_back(cap);
        ++a.stats["ep_body_excluded_arm_anchors"];
        if(next.size()>body_candidate_limit){++a.stats["ep_body_candidate_budget"];return {};}
      }
    if(next.empty()) {
      if(retry_anchors && position+1<order.size() && !retried.contains(pat)) {
        // A slot may also enter the anchor through a derived coordinate's axis (data[0]), which
        // coordinate_slots does not list: any unbound slot of a capture qualifies.
        const bool unbound=std::any_of(caps.begin(),caps.end(),[&](const std::vector<Id> &cap) {
          return std::find(cap.begin(),cap.end(),invalid)!=cap.end();
        });
        if(unbound) {
          retried.insert(pat);order.push_back(pat);
          ++a.stats["ep_body_deferred_anchor_retries"];
          continue;
        }
      }
      return {};
    }
    caps=std::move(next);
  }
  return caps;
}

} // namespace rg
