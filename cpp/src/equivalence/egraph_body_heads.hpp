#pragma once
#include "egraph_engine.hpp"

namespace sentryward::equivalence {
namespace {

inline auto Engine::BodyHeadIndex::charge(std::size_t amount) -> bool {
  engine.poll();
  if(exhausted || (remaining && amount>*remaining)) {
    // Exhausting a per-query budget does not invalidate earlier epoch
    // witnesses. The outer ledger, when exhausted, stays exhausted.
    if(remaining && amount>*remaining)*remaining=0;
    return false;
  }
  if(!query_guards.charge(amount))return false;
  if(remaining && remaining!=query_remaining && remaining!=mapper_remaining)*remaining-=amount;
  if(engine.options_.max_work && amount>1) {
    if(amount-1>engine.options_.max_work-engine.work_)throw Limit{"work_budget"};
    engine.work_+=amount-1;
  }
  return true;
}

inline auto Engine::BodyHeadIndex::initializeSoleHints(std::size_t retained_capacity,
    bool complete_descriptors) -> void {
  retained_entries=std::clamp(retained_capacity,max_entries,
      epoch_work_allowance::maximum_retained_headers);
  const auto bound=snapshot.size(),words=bound/64+(bound%64!=0);
  if(bound>std::numeric_limits<std::size_t>::max()/2)return;
  const auto header_slots=2*bound,header_words=header_slots/64+(header_slots%64!=0);
  if(words>(std::numeric_limits<std::size_t>::max()-header_words-3)/2 ||
      !charge(2*words+header_words+3))return;
  // No parentheses: fundamental Class slots are NOT value-initialized.
  // A validity bit is always tested before reading its corresponding slot.
  auto roots=std::unique_ptr<Class[]>(new Class[bound]);
  auto shallow=std::unique_ptr<Class[]>(new Class[bound]);
  auto headers=std::unique_ptr<Class[]>(new Class[header_slots]);
  sole_root_valid.resize(words,0);sole_shallow_valid.resize(words,0);
  source_header_valid.resize(header_words,0);
  sole_root_tags=std::move(roots);sole_shallow_tags=std::move(shallow);
  source_header_tags=std::move(headers);
  sole_bound=bound;sole_epoch=engine.unions_;
  // Full descriptor registration is an explicit preparation opt-in, not
  // a prerequisite for the original cheap local/sole-source token cache.
  // Prefix/local consumers must not bootstrap an unrelated metadata arena.
  // When requested, bootstrap charges THIS epoch/global work before a
  // query; later intern misses charge ordinary Engine work once.
  if(complete_descriptors && !adapter && engine.prepareCompleteHeaders(bound,*this))
    descriptor_generation=engine.complete_header_generation_;
}

inline auto Engine::BodyHeadIndex::sourceHeaderHint(Class id) -> std::optional<Class> {
  if(!source_header_tags || id/2>=sole_bound)return {};
  if(engine.unions_!=sole_epoch){cap();return {};}
  if(!charge(2))return {};
  if(!(source_header_valid[id/64]&(std::uint64_t{1}<<(id%64))))return {};
  ++source_header_hits;return source_header_tags[id];
}

inline auto Engine::BodyHeadIndex::retainSourceHeader(Class id,Class tag) -> bool {
  if(source_header_tags && id/2<sole_bound) {
    if(engine.unions_!=sole_epoch)return cap();
    if(!charge(2))return false;
    const auto bit=std::uint64_t{1}<<(id%64);
    source_header_tags[id]=tag;
    if(!(source_header_valid[id/64]&bit))++source_header_tokens;
    source_header_valid[id/64]|=bit;return true;
  }
  // Local/append-only callers keep the old bounded sparse representation.
  if(header_cache.size()<retained_entries)header_cache.emplace(id,tag);
  return true;
}

inline auto Engine::BodyHeadIndex::soleHint(Class cls,bool shallow) -> std::optional<Class> {
  if(cls>=sole_bound)return {};
  if(engine.unions_!=sole_epoch){cap();return {};}
  if(!charge(2))return {};
  const auto& valid=shallow?sole_shallow_valid:sole_root_valid;
  if(!(valid[cls/64]&(std::uint64_t{1}<<(cls%64))))return {};
  if(shallow){++sole_shallow_hits;return sole_shallow_tags[cls];}
  ++sole_root_hits;return sole_root_tags[cls];
}

inline auto Engine::BodyHeadIndex::rememberSoleHint(Class cls,Class tag,bool shallow) -> bool {
  if(cls>=sole_bound)return true; // Append-only projected IDs stay uncached.
  if(engine.unions_!=sole_epoch)return cap();
  if(!charge(2))return false;
  auto& valid=shallow?sole_shallow_valid:sole_root_valid;
  const auto bit=std::uint64_t{1}<<(cls%64);
  if(shallow) {
    sole_shallow_tags[cls]=tag;if(!(valid[cls/64]&bit))++sole_shallow_tokens;
  } else {
    sole_root_tags[cls]=tag;if(!(valid[cls/64]&bit))++sole_root_tokens;
  }
  valid[cls/64]|=bit;return true;
}

inline auto Engine::BodyHeadIndex::affine(Class cls) -> std::optional<IndexAffine> {
  cls=owner(cls);
  if(cls>=snapshot.size() ||
      (snapshot[cls].type!="u64" && snapshot[cls].type!="exact:u64"))return {};
  std::size_t available=remaining?*remaining:131072;
  available=query_guards.available(available);
  const auto before=available;
  auto result=adapter?adapter->affine(cls,affine_cache,available):
      engine.affineValue(cls,affine_cache,0,&available,0,64);
  const auto used=before-available;
  // The callee used at most the prechecked minimum. No global double
  // charge: affineValue already owns its actual primitive invoice.
  if(!query_guards.charge(used))return {};
  if(remaining && remaining!=query_remaining && remaining!=mapper_remaining)*remaining-=used;
  return result;
}

inline auto Engine::BodyHeadIndex::header(Class source_id,bool affine_address) -> std::optional<Class> {
  if(source_id>=snapshot.size())return {};
  // Explicit stable IDs: no pointer arithmetic, retained node reference or
  // header span may survive an append to the borrowed arena.
  const auto& source=snapshot[source_id];
  const auto id=source_id*2+affine_address;
  if(source_header_tags && source_id<sole_bound) {
    if(auto tag=sourceHeaderHint(id))return tag;
    if(exhausted || (remaining && !*remaining) || query_guards.exhausted())return {};
  } else {
    if(auto found=header_cache.find(id);found!=header_cache.end()) {
      if(!charge())return {};return found->second;
    }
  }
  if(!bounded(source))return {};
  std::optional<Class> descriptor;
  const HeaderKey* registered=nullptr;
  if(!affine_address && !adapter && descriptor_generation) {
    if(descriptor_generation!=engine.complete_header_generation_) {cap();return {};}
    if(!charge(2))return {};
    descriptor=engine.completeHeaderId(source_id);
    if(descriptor) {
      if(auto found=descriptor_tags.find(*descriptor);found!=descriptor_tags.end()) {
        ++descriptor_hits;
        if(!retainSourceHeader(id,found->second))return {};return found->second;
      }
      registered=engine.completeHeader(*descriptor);
      if(!registered)return {};
    }
  }
  auto probe=headerProbe(source);
  if(affine_address)probe={8,source.type,"affine-address-body-filter",{},{},{},0,0};
  else if(probe.fold)probe.fold->binder_level=0; // Scheduling only; full scope remains mandatory.
  const auto bytes=probe.type.size()+probe.name.size()+probe.attributes.size()+probe.provenance.size();
  // Hash the transient fields once. Owning entries retain this scalar
  // fingerprint, so rehash never rescans strings or throws mid-insertion.
  if(registered) {
    // The immutable Engine-owned descriptor already performed exact full
    // collision equality. Borrow it only for this operation, never retain
    // a view into nodes_ or substitute its birth ID for local tag order.
    if(!charge(8))return {};probe=headerProbe(registered->value);probe.hash=registered->hash;
  } else {
    if(!charge(8+bytes))return {};probe.hash=HeaderHash::compute(probe);
  }
  ++header_probes;
  try {
    // Equality charges every ACTUAL collision comparison. No string/vector
    // is copied on a hit; hash equality is never an equality proof.
    if(auto found=headers.find(probe);found!=headers.end()) {
      ++header_probe_hits;
      if(descriptor) {
        if(!charge(2))return {};descriptor_tags.emplace(*descriptor,found->second);++descriptor_translations;
      }
      if(!retainSourceHeader(id,found->second))return {};
      return found->second;
    }
    if(headers.size()>=retained_entries){cap();return {};}
    // One owned key plus the existing owned header_values copy, only on
    // a distinct-header miss. Child zeros are charged before construction.
    if(!charge(2*bytes+2*probe.arity+1))return {};
    Key key;key.kind=probe.kind;key.type=probe.type;key.name=probe.name;
    key.attributes=probe.attributes;key.provenance=probe.provenance;
    key.fold=probe.fold;key.index_level=probe.index_level;key.children.resize(probe.arity,0);
    const auto tag=header_values.size();header_values.push_back(key);
    bool inserted=false;Class retained_tag=tag;
    try {
      const auto entry=headers.emplace(HeaderKey{std::move(key),probe.hash},tag);
      inserted=entry.second;retained_tag=entry.first->second;
    } catch(...) {
      // Cached hash is noexcept; insertion has no effect if equality or
      // allocation throws. Keep the owned table/arena tag invariant even
      // on a local allowance denial or a propagated global limit.
      header_values.pop_back();throw;
    }
    if(!inserted) {
      header_values.pop_back();
      if(descriptor) {
        if(!charge(2))return {};descriptor_tags.emplace(*descriptor,retained_tag);++descriptor_translations;
      }
      if(!retainSourceHeader(id,retained_tag))return {};return retained_tag;
    }
    header_copied_bytes+=2*bytes;
    if(descriptor) {
      if(!charge(2))return {};descriptor_tags.emplace(*descriptor,tag);++descriptor_translations;
    }
    if(!retainSourceHeader(id,tag))return {};return tag;
  } catch(const HeaderBudget&) {return {};}
}

inline auto Engine::BodyHeadIndex::priority(const Key& k) -> unsigned {
  if(k.kind==static_cast<int>(s::Kind::Constant))return 0;
  if(k.kind==static_cast<int>(s::Kind::Fold))return 1;
  if(k.kind==static_cast<int>(s::Kind::Operation)) {
    if(k.name=="Reciprocal" || k.name=="Negate")return 2;
    if(k.name=="Multiply")return 3;
    if(k.name=="Add")return 4;
    if(k.name=="Divide")return 5;
  }
  return 6;
}

inline auto Engine::BodyHeadIndex::roots(Class cls) -> std::optional<Roots> {
  cls=owner(cls);
  if(auto found=roots_cache.find(cls);found!=roots_cache.end())return found->second;
  if(cls>=memberCount() || classMembers(cls).empty() || classMembers(cls).size()>max_members)return {};
  const bool sole=classMembers(cls).size()==1;
  if(sole) {
    if(auto tag=soleHint(cls,false))return Roots{classMembers(cls).front(),*tag};
    if(exhausted || (remaining && !*remaining) || query_guards.exhausted())return {};
  }
  // An affine address is a wildcard only in the filter. Its complete ring
  // coefficients, including captured indices, are checked by affineBodyMap.
  if((snapshot[cls].type=="u64" || snapshot[cls].type=="exact:u64") && affine(cls)) {
    auto tag=header(classMembers(cls).front(),true);if(!tag)return {};
    // The shared exact affine cache already owns this complete summary.
    // Repeated address headers need no second retained per-class hint.
    if(sole && !rememberSoleHint(cls,*tag,false))return {};
    return Roots{classMembers(cls).front(),*tag};
  }
  if((remaining && !*remaining) || query_guards.exhausted())return {};
  if(sole) {
    // One ACTUAL snapshot-prefix witness has no header choice to memoize.
    // Preserve roots()'s metadata/direct-self checks; an indirect cycle is
    // still rejected by the complete paired walk, never by this hint.
    const auto id=classMembers(cls).front();const auto& k=snapshot[id];
    if(!bounded(k) || !charge(k.children.size()))return {};
    if(std::any_of(k.children.begin(),k.children.end(),
        [&](Class child){return owner(child)==cls;}))return {};
    auto tag=header(id);if(!tag || !rememberSoleHint(cls,*tag,false))return {};
    return Roots{id,*tag};
  }
  if(roots_cache.size()>=retained_entries){cap();return {};}
  std::optional<Roots> result;
  for(auto id:classMembers(cls)) {
    if(!charge(snapshot[id].children.size()))return {};
    // Do not let a direct cyclic alias (notably Fold0(seed,...)=seed)
    // hide a normal scheduling shape. This is not an admission guard:
    // choices() still indexes these heads, and a chosen cyclic pair is
    // rejected by the actual per-side witness walk, with no retry.
    if(std::any_of(snapshot[id].children.begin(),snapshot[id].children.end(),
        [&](Class child){return owner(child)==cls;}))continue;
    auto tag=header(id);if(!tag)return {};
    if(!result || preferredTag(*tag,result->tag))result=Roots{id,*tag};
  }
  if(!result)return {};
  roots_cache.emplace(cls,*result);return result;
}

inline auto Engine::BodyHeadIndex::compact(Class cls) -> std::optional<Class> {
  cls=owner(cls);
  if(auto found=compact_cache.find(cls);found!=compact_cache.end())return found->second;
  if(cls<memberCount() && classMembers(cls).size()==1) {
    // There is NO scheduling alternative or child-tag tie to resolve.
    // Preserve exactly roots()/compact()'s local bounded/self guard, but
    // avoid constructing/caching irrelevant headers for a sole witness.
    // Indirect selected cycles remain the actual paired walker's duty.
    const auto id=classMembers(cls).front();
    if(id>=snapshot.size() || !bounded(snapshot[id]) ||
        !charge(1+snapshot[id].children.size()))return {};
    if(std::any_of(snapshot[id].children.begin(),snapshot[id].children.end(),
        [&](Class child){return owner(child)==cls;}))return {};
    return id;
  }
  if(compact_cache.size()>=retained_entries){cap();return {};}
  auto root=roots(cls);if(!root)return {};
  if(header_values[root->tag].kind==8) {
    compact_cache.emplace(cls,root->preferred);return root->preferred;
  }
  // Tie-break one common root header by immediate child headers only.
  // In particular Multiply(x,1/d) chooses an already stored Reciprocal
  // witness when available. No recursive compact()/choices() descent.
  Class result=root->preferred;std::optional<std::vector<Class>> best;
  for(auto id:classMembers(cls)) {
    const auto& k=snapshot[id];auto tag=header(id);if(!tag)return {};
    if(*tag!=root->tag)continue;
    if(!charge(k.children.size()))return {};
    if(std::any_of(k.children.begin(),k.children.end(),
        [&](Class child){return owner(child)==cls;}))continue;
    std::vector<Class> tags;tags.reserve(k.children.size());
    for(auto child:k.children) {auto part=roots(child);if(!part)return {};tags.push_back(part->tag);}
    if((arithmetic(k) || integerCommutative(k)) && tags.size()==2 && preferredTag(tags[1],tags[0]))
      std::swap(tags[0],tags[1]);
    bool less=!best;
    if(best)for(std::size_t i=0;i<tags.size();++i) {
      const auto& a=header_values[tags[i]];const auto& b=header_values[(*best)[i]];
      if(!charge(1+std::min(a.type.size(),b.type.size())+std::min(a.name.size(),b.name.size())))return {};
      if(tags[i]!=(*best)[i]){less=preferredTag(tags[i],(*best)[i]);break;}
    }
    if(less){result=id;best=std::move(tags);}
  }
  compact_cache.emplace(cls,result);return result;
}

inline auto Engine::BodyHeadIndex::shallow(Class cls) -> std::optional<Class> {
  cls=owner(cls);
  if(auto found=shallow_cache.find(cls);found!=shallow_cache.end())return found->second;
  const bool sole=cls<memberCount() && classMembers(cls).size()==1;
  if(sole) {
    if(auto tag=soleHint(cls,true))return tag;
    if(exhausted || (remaining && !*remaining) || query_guards.exhausted())return {};
  }
  if(!sole && shallow_cache.size()>=retained_entries){cap();return {};}
  auto root=roots(cls);if(!root)return {};
  auto head=preferred(cls);if(!head)return {};
  const auto& k=snapshot[*head];Signature signature{root->tag,{}};
  const bool address=header_values[root->tag].kind==8;
  const auto arity=address?std::size_t{0}:k.children.size();
  if(!charge(arity))return {};signature.second.reserve(arity<=2?3*arity:arity);
  if(arity<=2) {
    // Exactly two local edges, not a recursive shallow()/choices() walk.
    // Binary/unary roots contribute at most four terminal headers. A
    // child's header carries its arity, so these flattened blocks remain
    // unambiguous. Affine addresses stop at their arity-zero wildcard.
    std::array<std::vector<Class>,2> parts;
    for(std::size_t i=0;i<arity;++i) {
      const auto child=k.children[i];auto tag=roots(child);if(!tag)return {};
      parts[i].push_back(tag->tag);
      if(header_values[tag->tag].kind==8)continue;
      const auto child_arity=header_values[tag->tag].children.size();
      // Large-arity heads retain the old immediate hint. Never fan out
      // another layer beneath one, or recursively expand a grandchild.
      if(child_arity>2)continue;
      auto child_head=preferred(child);if(!child_head)return {};
      const auto& child_key=snapshot[*child_head];
      if(!charge(child_arity))return {};
      for(auto grandchild:child_key.children) {
        auto leaf=roots(grandchild);if(!leaf)return {};parts[i].push_back(leaf->tag);
      }
      if((arithmetic(child_key) || integerCommutative(child_key)) && child_arity==2 &&
          parts[i][2]<parts[i][1])std::swap(parts[i][1],parts[i][2]);
    }
    if((arithmetic(k) || integerCommutative(k)) && arity==2 && parts[1]<parts[0])
      std::swap(parts[0],parts[1]);
    if(!charge(3*arity))return {};
    for(std::size_t i=0;i<arity;++i)
      signature.second.insert(signature.second.end(),parts[i].begin(),parts[i].end());
  } else {
    // The existing bounded header-only behavior for larger arities is a
    // conservative scheduling hint, not a different proof procedure.
    for(auto child:k.children) {
      auto tag=roots(child);if(!tag)return {};signature.second.push_back(tag->tag);
    }
  }
  if(!charge((signature.second.size()+1)*(1+std::bit_width(shallow_keys.size()+1))))return {};
  auto found=shallow_keys.find(signature);
  if(found==shallow_keys.end()) {
    if(shallow_keys.size()>=retained_entries){cap();return {};}
    found=shallow_keys.emplace(std::move(signature),shallow_keys.size()).first;
  }
  // Both kinds retain the SAME complete bounded interned signature. Sole
  // classes need only a scalar token; multi-witness/ambiguous cache rows
  // and all distinct-signature/header/choice capacities stay unchanged.
  if(sole) {
    if(!rememberSoleHint(cls,found->second,true))return {};
  } else shallow_cache.emplace(cls,found->second);
  return found->second;
}

inline auto Engine::BodyHeadIndex::choices(Class cls) -> const Choices* {
  cls=owner(cls);
  if(auto found=choices_cache.find(cls);found!=choices_cache.end())return &found->second;
  if(cls>=memberCount() || classMembers(cls).empty() || classMembers(cls).size()>max_members)return nullptr;
  if(choices_cache.size()+root_choices_cache.size()>=retained_entries){cap();return nullptr;}
  Choices result;
  for(auto id:classMembers(cls)) {
    const auto& k=snapshot[id];auto tag=header(id);if(!tag)return nullptr;
    Signature signature{*tag,{}};
    if(!charge(k.children.size()))return nullptr;signature.second.reserve(k.children.size());
    for(auto child:k.children) {
      auto part=shallow(child);if(!part)return nullptr;signature.second.push_back(*part);
    }
    if((arithmetic(k) || integerCommutative(k)) && signature.second.size()==2 &&
        signature.second[1]<signature.second[0])std::swap(signature.second[0],signature.second[1]);
    if(!charge((signature.second.size()+1)*(1+std::bit_width(choice_keys.size()+1))))return nullptr;
    auto found=choice_keys.find(signature);
    if(found==choice_keys.end()) {
      if(choice_keys.size()>=retained_entries){cap();return nullptr;}
      found=choice_keys.emplace(std::move(signature),choice_keys.size()).first;
    }
    const auto binder=k.fold?k.fold->binder_level:std::numeric_limits<std::uint32_t>::max();
    auto [entry,inserted]=result.try_emplace(ChoiceKey{found->second,binder},Choice{id});
    if(inserted) {
      if(choice_entries>=retained_entries){cap();return nullptr;}++choice_entries;
    } else {
      const auto& previous=snapshot[entry->second.head];
      bool same=previous.fold==k.fold && previous.index_level==k.index_level;
      if(!charge(k.children.size()))return nullptr;
      for(std::size_t i=0;i<k.children.size();++i)same &= owner(previous.children[i])==owner(k.children[i]);
      entry->second.unique &= same;
    }
  }
  return &choices_cache.emplace(cls,std::move(result)).first->second;
}

inline auto Engine::BodyHeadIndex::matchChoices(const Choices& x,const Choices& y,
    bool unique_root) -> std::optional<std::pair<Class,Class>> {
  std::optional<std::pair<Class,Class>> result;std::optional<Class> best;
  auto p=x.begin(),q=y.begin();
  while(p!=x.end() && q!=y.end()) {
    if(!charge())return {};
    if(p->first.first<q->first.first){++p;continue;}
    if(q->first.first<p->first.first){++q;continue;}
    const auto signature=p->first.first;
    auto p_end=p,q_end=q;
    while(p_end!=x.end() && p_end->first.first==signature) {
      if(!charge())return {};++p_end;
    }
    while(q_end!=y.end() && q_end->first.first==signature) {
      if(!charge())return {};++q_end;
    }
    // A linear merge, not a variant-pair product. Prefer one unique exact
    // binder match. Only singleton unique buckets may pair alpha labels;
    // the actual chosen witness below still validates every lexical edge,
    // seed, capture and cycle. Never retry another head after it fails.
    std::optional<std::pair<Class,Class>> candidate;
    auto a=p,b=q;
    while(a!=p_end && b!=q_end) {
      if(!charge())return {};
      if(a->first.second<b->first.second){++a;continue;}
      if(b->first.second<a->first.second){++b;continue;}
      if(a->second.unique && b->second.unique) {
        if(unique_root && candidate)return {};
        candidate=std::pair{a->second.head,b->second.head};
        if(!unique_root)break;
      }
      ++a;++b;
    }
    if(!candidate && std::next(p)==p_end && std::next(q)==q_end &&
        p->second.unique && q->second.unique)
      candidate=std::pair{p->second.head,q->second.head};
    if(candidate) {
      if(unique_root && result)return {};
      auto tag=header(candidate->first);if(!tag)return {};
      if(!best || preferredTag(*tag,*best)) {
        result=*candidate;best=*tag;
      }
    }
    p=p_end;q=q_end;
  }
  return result;
}

inline auto Engine::BodyHeadIndex::rootChoices(Class cls) -> const Choices* {
  cls=owner(cls);
  if(auto found=root_choices_cache.find(cls);found!=root_choices_cache.end()) {
    if(!charge())return nullptr;return &found->second;
  }
  if(cls>=memberCount() || classMembers(cls).empty() || classMembers(cls).size()>max_members)return nullptr;
  if(choices_cache.size()+root_choices_cache.size()>=retained_entries){cap();return nullptr;}
  // At most max_members ACTUAL complete root headers. Child shallow
  // signatures are optional scheduling hints, not semantic premises.
  // A shared invariant may have no nonrecursive child header after a
  // certified Add/Subtract alias creates an SCC; the paired walk can
  // still certify that child as an independent same-class opaque leaf.
  // This cache shares the existing resident/entry/work allowances.
  Choices result;
  for(auto id:classMembers(cls)) {
    if(!charge(2+std::bit_width(result.size()+1)))return nullptr;
    const auto& k=snapshot[id];auto tag=header(id);if(!tag)return nullptr;
    const auto binder=k.fold?k.fold->binder_level:std::numeric_limits<std::uint32_t>::max();
    auto [entry,inserted]=result.try_emplace(ChoiceKey{*tag,binder},Choice{id});
    if(inserted) {
      if(choice_entries>=retained_entries){cap();return nullptr;}++choice_entries;
      continue;
    }
    const auto& previous=snapshot[entry->second.head];
    bool same=previous.fold==k.fold && previous.index_level==k.index_level;
    if(!charge(k.children.size()))return nullptr;
    for(std::size_t i=0;i<k.children.size();++i)
      same &= owner(previous.children[i])==owner(k.children[i]);
    entry->second.unique &= same;
  }
  return &root_choices_cache.emplace(cls,std::move(result)).first->second;
}

inline auto Engine::BodyHeadIndex::canonicalRootPair(const Choices& x,const Choices& y) -> std::optional<std::pair<Class,Class>> {
  // Alternate certified operators in ONE eclass are not alternate
  // meanings. Select one common complete root header by the existing
  // fixed priority, BEFORE examining child-witness uniqueness. A chosen
  // ambiguous header stays Unknown; never retry a less preferred header.
  // Both indexes have at most max_members entries. This is a linear
  // ordered merge, not a candidate-pair product or an ancestor walk.
  auto p=x.begin(),q=y.begin();
  auto p_begin=x.end(),p_end=x.end(),q_begin=y.end(),q_end=y.end();
  std::optional<Class> best;
  while(p!=x.end() && q!=y.end()) {
    if(!charge())return {};
    if(p->first.first<q->first.first){++p;continue;}
    if(q->first.first<p->first.first){++q;continue;}
    const auto tag=p->first.first;
    auto a=p,b=q;
    while(a!=x.end() && a->first.first==tag) {if(!charge())return {};++a;}
    while(b!=y.end() && b->first.first==tag) {if(!charge())return {};++b;}
    if(!best || preferredTag(tag,*best)) {
      best=tag;p_begin=p;p_end=a;q_begin=q;q_end=b;
    }
    p=a;q=b;
  }
  if(!best)return {};
  std::optional<std::pair<Class,Class>> result;
  auto a=p_begin,b=q_begin;
  while(a!=p_end && b!=q_end) {
    if(!charge())return {};
    if(a->first.second<b->first.second){++a;continue;}
    if(b->first.second<a->first.second){++b;continue;}
    if(a->second.unique && b->second.unique) {
      if(result)return {}; // Within-header binder variants are ambiguous.
      result=std::pair{a->second.head,b->second.head};
    }
    ++a;++b;
  }
  if(!result && std::next(p_begin)==p_end && std::next(q_begin)==q_end &&
      p_begin->second.unique && q_begin->second.unique)
    result=std::pair{p_begin->second.head,q_begin->second.head};
  return result;
}

inline auto Engine::BodyHeadIndex::fixedFoldPair(const Choices& x,const Choices& y) -> std::optional<std::pair<Class,Class>> {
  // Binder labels distinguish actual witnesses, not the meanings of
  // already-certified aliases in one eclass. If the old unique-bucket
  // route cannot choose, select ONE stored Fold pair deterministically.
  // No candidate product or retry: every seed/body/lexical edge below is
  // still proved by the ordinary paired walk. Adapter-owned IDs have no
  // Engine finite-property certificate and conservatively keep Unknown.
  if(adapter)return {};
  auto p=x.begin(),q=y.begin();std::optional<Class> best;
  std::optional<std::pair<Class,Class>> result;
  while(p!=x.end() && q!=y.end()) {
    if(!charge())return {};
    if(p->first.first<q->first.first){++p;continue;}
    if(q->first.first<p->first.first){++q;continue;}
    const auto tag=p->first.first;auto a=p,b=q;
    auto left=p->second.head,right=q->second.head;
    while(a!=x.end() && a->first.first==tag) {
      if(!charge())return {};left=std::min(left,a->second.head);++a;
    }
    while(b!=y.end() && b->first.first==tag) {
      if(!charge())return {};right=std::min(right,b->second.head);++b;
    }
    const auto& h=header_values[tag];
    if(h.kind==static_cast<int>(s::Kind::Fold) && h.fold && h.type=="real" &&
        h.attributes.empty() && h.provenance.empty() && h.children.size()==2 &&
        (!best || preferredTag(tag,*best))) {
      best=tag;result=std::pair{left,right};
    }
    p=a;q=b;
  }
  if(!result)return {};
  // Do not search another alias if this ONE actual witness is ineligible.
  for(auto id:{result->first,result->second}) {
    if(!charge(7) || id>=snapshot.size())return {};
    const auto& k=snapshot[id];
    if(!engine.finite_[owner(id)])return {};
    for(auto child:k.children)if(child>=snapshot.size() || snapshot[child].type!="real" ||
        !engine.finite_[owner(child)])return {};
  }
  return result;
}

inline auto Engine::BodyHeadIndex::rootMatched(Class a,Class b) -> std::optional<std::pair<Class,Class>> {
  if(exhausted || (remaining && !*remaining) || query_guards.exhausted())return {};
  const auto* rx=rootChoices(a);if(!rx)return {};const auto* ry=rootChoices(b);if(!ry)return {};
  // ONE canonical common root pair, never retry/Cartesian alternatives.
  // Uniqueness is required WITHIN the chosen complete root header, not
  // across already-certified Multiply/Divide or Rsqrt/Reciprocal aliases.
  // Actual children, affine leaves, binder captures and selected cycles
  // still pass through the SAME complete scoped witness walk below.
  auto result=canonicalRootPair(*rx,*ry);
  if(!result)result=fixedFoldPair(*rx,*ry);
  if(fallback_trace_remaining && *fallback_trace_remaining) {
    --*fallback_trace_remaining;
    SENTRYWARD_DEBUG_PRINT(egraph_77);
    if(result)SENTRYWARD_DEBUG_PRINT(egraph_78);
    else SENTRYWARD_DEBUG_PRINT(egraph_79);
    SENTRYWARD_DEBUG_PRINT(egraph_80);
  }
  return result;
}

inline auto Engine::BodyHeadIndex::localRootCompatible(Class a,Class b) -> bool {
  auto pair=rootMatched(a,b);if(!pair)return false;
  const auto& p=snapshot[pair->first];const auto& q=snapshot[pair->second];
  // Fixed ONE extra local edge (<=2 child headers), never recursive
  // signatures or a trial full proof. A Fold preserves seed/body order.
  if(p.children.size()>2 || p.children.size()!=q.children.size())return false;
  const auto compatible=[&](std::size_t i,std::size_t j) {
    // Address shapes stay wildcards at this scheduling boundary. Their
    // exact coefficients and captured binders are checked by the walk.
    auto x=roots(p.children[i]),y=roots(q.children[j]);
    if(!x || !y)return false;
    if(header_values[x->tag].kind==8 && header_values[y->tag].kind==8)
      return x->tag==y->tag;
    return bool(rootMatched(p.children[i],q.children[j]));
  };
  if(p.children.size()==2 && (arithmetic(p) || integerCommutative(p))) {
    const bool direct=compatible(0,0) && compatible(1,1);
    const bool reverse=compatible(0,1) && compatible(1,0);
    // Repeating either operand makes the two orientations the SAME
    // multiset of child-pair obligations, not two possible proofs. All
    // obligations still pass through the full scoped walk below.
    return direct!=reverse || (direct &&
        (owner(p.children[0])==owner(p.children[1]) ||
         owner(q.children[0])==owner(q.children[1])));
  }
  for(std::size_t i=0;i<p.children.size();++i)if(!compatible(i,i))return false;
  return true;
}

inline auto Engine::BodyHeadIndex::matched(Class a,Class b,bool complete_root_first) -> std::optional<std::pair<Class,Class>> {
  a=owner(a);b=owner(b);
  if(exhausted || (remaining && !*remaining) || query_guards.exhausted())return {};
  if(a<memberCount() && b<memberCount() &&
      classMembers(a).size()==1 && classMembers(b).size()==1) {
    // No alternative or within-header ambiguity exists on either side.
    // Equal COMPLETE headers select the SAME sole actual witnesses; this
    // is not equality by shape. Every child, original Fold seed/binder,
    // capture, affine coefficient and indirect cycle is still checked by
    // the ordinary paired walk below. Different headers keep the old path.
    const auto p=classMembers(a).front(),q=classMembers(b).front();
    if(!charge(2))return {};
    auto x=header(p),y=header(q);if(!x || !y)return {};
    if(*x==*y)return std::pair{p,q};
  }
  if(!adapter) {
    // Preserve a common unary wrapper before consulting child-shape
    // hints. Its child may be a Fold on one side and a stored Sum forest
    // on the other; that is for the scoped reduction proof, not this
    // fixed-radius witness selector. Expanded binary aliases must not
    // obscure an already available matching wrapper.
    if(auto direct=rootMatched(a,b)) {
      const auto& p=snapshot[direct->first];
      if(idealOperation(p) && p.children.size()==1)return direct;
    }
    if(exhausted || (remaining && !*remaining) || query_guards.exhausted())return {};
  }
  if(complete_root_first && !adapter) {
    // Original-rank scoped Sum dispatch has ALREADY run. Use the existing
    // canonical complete-header priority to select ONE actual pair before
    // preparing descendant hints. Ambiguity within the chosen header is
    // still Unknown; other certified aliases are not alternate meanings.
    // A Fold stays on its existing lexical route. No selected child proof
    // is retried, and no header match itself certifies a value or geometry.
    ++root_first_attempts;
    if(auto direct=rootMatched(a,b)) {
      const auto& p=snapshot[direct->first];
      if(p.kind!=static_cast<int>(s::Kind::Fold) && !p.fold) {
        ++root_first_hits;return direct;
      }
    }
    if(exhausted || (remaining && !*remaining) || query_guards.exhausted())return {};
  }
  const auto* x=choices(a);const auto* y=choices(b);
  if(x && y)if(auto direct=matchChoices(*x,*y))return direct;
  // Unknown child hints are not a failed full witness. Select ONE bounded
  // complete root pair, then run the SAME scoped proof exactly once.
  return rootMatched(a,b);
}

} // namespace
} // namespace sentryward::equivalence
