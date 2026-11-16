#include "sentryward/scalar_fold/reduction_builder.h"
#include "sentryward/algorithm/diagnostics/compression_audit.h"

#include <algorithm>
#include <charconv>
#include <iterator>
#include <unordered_set>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>

namespace sentryward::scalar_fold {
namespace {
std::string operation(Reducer r) {
  switch (r) {
    case Reducer::Sum: return "Add";
    case Reducer::Product: return "Multiply";
    case Reducer::Max: return "Maximum";
    case Reducer::Min: return "Minimum";
  }
  throw std::invalid_argument("invalid reducer");
}
std::string identity(Reducer r) {
  switch (r) {
    case Reducer::Sum: return "0";
    case Reducer::Product: return "1";
    case Reducer::Max: return "-inf";
    case Reducer::Min: return "inf";
  }
  throw std::invalid_argument("invalid reducer");
}
bool sameUse(const Ref& a, const Ref& b) {
  return a.owner == b.owner && a.conversion == b.conversion &&
      a.converted_type == b.converted_type && a.pointer_provenance == b.pointer_provenance;
}
bool sameShape(const Node& a, const Node& b) {
  return a.kind == b.kind && a.type == b.type && a.name == b.name &&
      a.attributes == b.attributes && a.fold == b.fold && a.index_level == b.index_level &&
      a.operands.size() == b.operands.size();
}
std::optional<std::uint64_t> offset(const Node& n) {
  if (n.kind != Kind::Constant || (n.type != "u64" && n.type != "s64") || !n.attributes.empty()) return {};
  std::uint64_t value;
  const auto [end,error] = std::from_chars(n.name.data(), n.name.data()+n.name.size(), value);
  if (error != std::errc{} || end != n.name.data()+n.name.size()) return {};
  if(n.type=="s64" && value>INT64_MAX)return {};
  return value;
}
std::optional<std::uint64_t> indexControl(const Node& n) {
  if(n.kind!=Kind::Constant || !n.attributes.empty() ||
      (n.type!="u64" && n.type!="s64" && n.type!="u32" && n.type!="b64" && n.type!="b32"))return {};
  std::uint64_t value;
  const auto [end,error]=std::from_chars(n.name.data(),n.name.data()+n.name.size(),value);
  if(error!=std::errc{} || end!=n.name.data()+n.name.size())return {};
  if(n.type=="s64" && value>INT64_MAX)return {};
  if((n.type=="u32" || n.type=="b32") && value>UINT32_MAX)return {};
  return value;
}
constexpr std::uint32_t kUnknownBound=std::numeric_limits<std::uint32_t>::max();
// Check exactly one substitution: next(i) == previous(i + previous_count).
// Read-only and bounded by visited operand pairs, never logical iterations.
// No algebraic search, candidate construction, or sampling of body values.
bool consecutiveBody(const Graph& graph, Ref previous, Ref next,
                     std::uint32_t level, std::uint64_t previous_count) {
  std::vector<std::pair<Ref,Ref>> work{{previous,next}};
  std::size_t budget=4096;
  while(!work.empty()) {
    if(!budget--)return false;
    auto [a,b]=std::move(work.back());work.pop_back();
    if(!sameUse(a,b))return false;
    const auto& x=graph.node(a);const auto& y=graph.node(b);
    if(x.kind==Kind::Index && x.index_level==level) {
      if(y.kind!=Kind::Operation || y.name!="Add" || y.type!=x.type ||
         !y.attributes.empty() || y.operands.size()!=2)return false;
      const auto plain=[](const Ref& r) {
        return r.conversion.empty() && r.converted_type.empty() && r.pointer_provenance.empty();
      };
      bool matched=false;
      for(unsigned side=0;side<2;++side) {
        const auto& index=y.operands[side];const auto& shift=y.operands[1-side];
        const auto& inode=graph.node(index);const auto& cnode=graph.node(shift);
        matched|=plain(index) && plain(shift) && inode.kind==Kind::Index &&
            inode.index_level==level && inode.type==x.type && cnode.type==x.type &&
            offset(cnode)==previous_count;
      }
      if(!matched)return false;
      continue;
    }
    if(!sameShape(x,y))return false;
    // Do not substitute through a shadowing binder. Well-formed lexical
    // bodies use deeper levels, but rejecting here also protects raw builders.
    if(x.kind==Kind::Fold && x.fold->binder_level==level)return false;
    if(x.operands.size()>budget || work.size()>budget-x.operands.size())return false;
    for(std::size_t i=0;i<x.operands.size();++i)
      work.emplace_back(x.operands[i],y.operands[i]);
  }
  return true;
}
struct Match {
  Graph& graph;
  std::uint32_t level;
  // Per-node memo shared across matches of one graph epoch; see bound().
  std::vector<std::uint32_t>& bounds;
  std::unordered_map<Id,Ref>& lifted;
  std::size_t remaining = 4096;
  struct Pair {
    Ref a,b;bool address;
    bool operator==(const Pair&) const = default;
  };
  // An accepted triple's three Refs share one use (checkImpl requires sameUse), so its
  // memo entry is the three nodes, k, the address flag, the owner and that use, numbered by
  // `uses` (0: no conversion or provenance): equal entries are exactly equal triples.
  struct Triple {
    Id a,b,actual;std::uint64_t k,owner;std::size_t use;bool address;
    bool operator==(const Triple&) const = default;
  };
  struct PairHash {
    std::size_t operator()(const Pair& p) const {
      const auto h=RefHash{}(p.a),q=RefHash{}(p.b);
      return h^(q+0x9e3779b9+(h<<6)+(h>>2))^std::size_t(p.address);
    }
  };
  struct TripleHash {
    std::size_t operator()(const Triple& p) const {
      std::size_t h=p.a;
      for(const std::size_t v:{p.b,p.actual,std::size_t(p.k),std::size_t(p.owner),p.use,std::size_t(p.address)})
        h^=v+0x9e3779b97f4a7c15ULL+(h<<6)+(h>>2);
      return h;
    }
  };
  // Local to one immutable-epoch match and binder level. Proof/body traversal
  // follows distinct DAG pairs, not paths through a diamond. Full Ref keys
  // preserve use-site conversion and pointer annotations, as well as whether
  // a node is being interpreted as an address or a numerical expression.
  std::unordered_set<Triple,TripleHash> checked{};
  std::vector<std::array<std::string,3>> uses{};
  std::unordered_map<Pair,Ref,PairHash> bodies{};
  // The number of r's use in `uses`, added when `add`; nullopt when absent.
  std::optional<std::size_t> useNumber(const Ref& r,bool add) {
    if(r.conversion.empty() && r.converted_type.empty() && r.pointer_provenance.empty())return 0;
    for(std::size_t i=0;i<uses.size();++i)
      if(uses[i][0]==r.conversion && uses[i][1]==r.converted_type && uses[i][2]==r.pointer_provenance)return i+1;
    if(!add)return std::nullopt;
    uses.push_back({r.conversion,r.converted_type,r.pointer_provenance});
    return uses.size();
  }

  // One plus the deepest Index level or Fold binder in a subtree; zero if none.
  // Each node is evaluated once per epoch, so lifting never rescans unaffected subtrees.
  std::uint32_t bound(Id root) {
    if(bounds.size()<graph.size())bounds.resize(graph.size(),kUnknownBound);
    if(bounds[root]!=kUnknownBound)return bounds[root];
    const auto next=[](std::uint32_t x){return x>=kUnknownBound-1?kUnknownBound-1:x+1;};
    std::vector<Id> work{root};
    while(!work.empty()) {
      const auto id=work.back();
      if(bounds[id]!=kUnknownBound){work.pop_back();continue;}
      const auto& node=graph.node(id);
      bool ready=true;
      for(const auto& child:node.operands)
        if(bounds[child.node]==kUnknownBound){work.push_back(child.node);ready=false;}
      if(!ready)continue;
      std::uint32_t value=0;
      for(const auto& child:node.operands)value=std::max(value,bounds[child.node]);
      if(node.kind==Kind::Index)value=std::max(value,next(node.index_level));
      if(node.kind==Kind::Fold)value=std::max(value,next(node.fold->binder_level));
      bounds[id]=value;work.pop_back();
    }
    return bounds[root];
  }
  static Ref use(Ref result,const Ref& source) {
    result.conversion=source.conversion;
    result.converted_type=source.converted_type;
    result.pointer_provenance=source.pointer_provenance;
    return result;
  }
  // value must not refer into graph storage: interning below may move it.
  Ref lift(const Ref& value) {
    // Subtrees without binders at or above this level are unchanged by lifting.
    if(bound(value.node)<=level)return value;
    if(const auto found=lifted.find(value.node);found!=lifted.end())return use(found->second,value);
    std::unordered_set<Id> seen;
    std::vector<Id> work{value.node},ordered;
    while(!work.empty()) {
      const auto id=work.back();work.pop_back();
      if(lifted.contains(id) || !seen.insert(id).second)continue;
      if(bound(id)<=level){lifted.emplace(id,graph.ref(id));continue;}
      ordered.push_back(id);
      for(const auto& child:graph.node(id).operands)work.push_back(child.node);
    }
    std::sort(ordered.begin(),ordered.end());
    for(auto id:ordered) {
      const auto node=graph.node(id);
      std::vector<Ref> args;
      bool changed=false;
      for(const auto& child:node.operands) {
        auto replacement=use(lifted.at(child.node),child);
        changed |= replacement!=child;
        args.push_back(std::move(replacement));
      }
      auto result=graph.ref(id);
      if(node.kind==Kind::Index && node.index_level>=level) {
        if(node.index_level==UINT32_MAX)throw std::overflow_error("fold binder overflow");
        result=graph.index(node.index_level+1);
      } else if(node.kind==Kind::Fold && (changed || node.fold->binder_level>=level)) {
        auto binder=node.fold->binder_level;
        if(binder>=level) {
          if(binder==UINT32_MAX)throw std::overflow_error("fold binder overflow");
          ++binder;
        }
        result=graph.fold(node.fold->reducer,args.at(0),args.at(1),binder,node.fold->count,node.type);
      } else if(changed) {
        if(node.kind==Kind::InitialCell)result=graph.initialCell(node.name,args.at(0),node.type);
        else {
          result=graph.operation(node.name,node.type,std::move(args),node.attributes);
          graph.markInstruction(result,graph.instructionModes(id));
        }
      }
      lifted.emplace(id,result);
    }
    return use(lifted.at(value.node),value);
  }

  // Read-only: interns nothing, so references into node operands and matcher
  // tables stay valid for the whole walk.
  bool check(const Ref& a, const Ref& b, const Ref& actual, std::uint64_t k,
             bool address = false, unsigned depth = 0) {
    // The overwhelmingly common fixed factor needs no cache entry or walk.
    if(a==b) {
      SENTRYWARD_AUDIT(audit->add(AuditCounter::fold_match_visits);
        audit->add(AuditCounter::fold_shared_identity_checks));
      return a==actual;
    }
    // Only triples with one use are ever accepted (and memoized).
    if(sameUse(a,b) && sameUse(a,actual))
      if(const auto use=useNumber(a,false);use && checked.contains(Triple{a.node,b.node,actual.node,k,a.owner,*use,address}))
        return true;
    const bool accepted=checkImpl(a,b,actual,k,address,depth);
    if(accepted)checked.insert(Triple{a.node,b.node,actual.node,k,a.owner,*useNumber(a,true),address});
    return accepted;
  }
  bool checkImpl(const Ref& a, const Ref& b, const Ref& actual, std::uint64_t k,
                bool address, unsigned depth) {
    SENTRYWARD_AUDIT(audit->add(AuditCounter::fold_match_visits));
    if (!remaining || depth > 128) {
      SENTRYWARD_AUDIT(audit->add(!remaining?AuditCounter::fold_budget_rejections:AuditCounter::fold_depth_rejections);
        audit->event("FOLD_REJECT",!remaining?"match_work_budget":"match_depth_limit",depth,remaining,a.node));
      return false;
    }
    --remaining;
    if (!sameUse(a,b) || !sameUse(a,actual)) return false;
    // Read-only walk: references stay valid because check() interns nothing.
    const auto& x=graph.node(a);
    const auto& y=graph.node(b);
    const auto& z=graph.node(actual);
    if (address && a.conversion.empty()) {
      const auto p=offset(x), q=offset(y), r=offset(z);
      if (p && q && r && x.type==y.type && x.type==z.type) {
        const bool ascending=*q >= *p;
        const auto step=ascending ? *q-*p : *p-*q;
        std::uint64_t delta;
        if (__builtin_mul_overflow(step,k,&delta)) return false;
        if (ascending) {
          std::uint64_t expected;
          return !__builtin_add_overflow(*p,delta,&expected) && expected == *r;
        }
        return delta <= *p && *p-delta == *r;
      }
    }
    if (!sameShape(x,y) || !sameShape(x,z)) {
      SENTRYWARD_AUDIT(audit->add(AuditCounter::fold_shape_rejections));
      return false;
    }
    if (x.kind != Kind::Operation && x.kind != Kind::InitialCell && x.kind != Kind::Fold) return false;
    // The body takes the common instruction modes of the two prototypes (bodyImpl); every further
    // instance must be an image whose modes the body includes, or the body must be no image.
    if (x.kind == Kind::Operation) {
      const auto modes=instruction_mode::common(graph.instructionModes(a.node),graph.instructionModes(b.node));
      const auto own=graph.instructionModes(actual.node);
      if (instruction_mode::instruction(modes) && (!instruction_mode::instruction(own) || (own & ~modes))) return false;
    }
    if (address && x.kind != Kind::InitialCell) {
      // sameUse already proved identical edge conversions. Keep them around
      // the parameterized operation, including narrowing and bitcasts; do not
      // reject or erase an exact layout expression merely for having one.
      if (x.kind != Kind::Operation)
        return false;
      if (x.name=="ExtractBits") {
        // This is exact template substitution, not arithmetic simplification:
        // keep the extraction attributes and its fixed bit position intact.
        if(x.operands.size()!=2 || x.operands[1]!=y.operands[1] ||
            x.operands[1]!=z.operands[1])return false;
      } else if (x.name=="PtxMultiplyAdd") {
        if(x.operands.size()!=3)return false;
        const auto fixed=[&](std::size_t i) {
          return x.operands[i]==y.operands[i] && x.operands[i]==z.operands[i];
        };
        if(!fixed(0) && !fixed(1))return false;
      } else if (x.operands.size()!=2) {
        return false;
      } else if (x.name=="Multiply") {
        const auto fixed=[&](std::size_t i) {
          return x.operands[i]==y.operands[i] && x.operands[i]==z.operands[i] &&
              x.operands[i].conversion.empty() && indexControl(graph.node(x.operands[i])).has_value();
        };
        if (!fixed(0) && !fixed(1)) return false;
      } else if (x.name=="Divide" || x.name=="Remainder" ||
                 x.name=="ShiftLeft" || x.name=="ShiftRight" ||
                 x.name=="BitAnd" || x.name=="BitOr" || x.name=="BitXor") {
        // Preserve the exact index expression, not an affine approximation of
        // its result. Only its varying leaves are parameterized. Native layout
        // formulas routinely apply a fixed quotient, mask or shift to them.
        // check() still checks each appended term and all use-site attributes.
        if((x.type!="u64" && x.type!="s64" && x.type!="u32" && x.type!="b32" && x.type!="b64") || x.operands[1]!=y.operands[1] ||
            x.operands[1]!=z.operands[1] || !x.operands[1].conversion.empty())return false;
        const auto control=indexControl(graph.node(x.operands[1]));
        if(!control)return false;
        if((x.name=="Divide" || x.name=="Remainder") && !*control)return false;
        const unsigned width=x.type=="u32" || x.type=="b32"?32:64;
        if((x.name=="ShiftLeft" || x.name=="ShiftRight") && *control>=width)return false;
      } else if (x.name!="Add" && x.name!="Subtract") return false;
    }
    for (std::size_t i=0;i<x.operands.size();++i)
      if (!check(x.operands[i],y.operands[i],z.operands[i],k,
                 address || x.kind == Kind::InitialCell,depth+1)) return false;
    return true;
  }
  // a and b must not refer into graph storage (e.g. graph.node(x).operands):
  // interning below may reallocate it while they are still read. Callers
  // pass local copies or matcher-table fields, which interning never moves.
  Ref body(const Ref& a, const Ref& b, bool address = false) {
    if(a==b)return lift(a);
    const Pair key{a,b,address};
    if(const auto found=bodies.find(key);found!=bodies.end())return found->second;
    const auto result=bodyImpl(a,b,address);
    bodies.emplace(key,result);return result;
  }
  Ref bodyImpl(const Ref& a, const Ref& b, bool address) {
    // One stable copy of x; of y only its operands, read before any interning.
    auto x=graph.node(a);
    const auto& yn=graph.node(b);
    const auto q=offset(yn);
    const auto y_operands=yn.operands;
    Ref result;
    const auto p=offset(x);
    if (address && p && q && a.conversion.empty()) {
      const bool ascending=*q >= *p;
      const auto step=ascending ? *q-*p : *p-*q;
      auto delta=graph.operation("Multiply","u64",{graph.index(level),graph.constant("u64",std::to_string(step))});
      // Build in u64, where checked offsets and differences are representable.
      // Every accepted s64 occurrence is nonnegative and <= INT64_MAX; restore
      // its original type on this use, including beneath packed-address ops.
      auto base=graph.constant("u64",std::to_string(*p));
      result=graph.operation(ascending ? "Add" : "Subtract","u64",{base,delta});
      if(x.type=="s64")result=graph.convert(result,"s64","Convert");
    } else {
      std::vector<Ref> args;args.reserve(x.operands.size());
      for (std::size_t i=0;i<x.operands.size();++i)
        args.push_back(body(x.operands[i],y_operands[i],address || x.kind == Kind::InitialCell));
      // x is not read after this construction: its strings move into it.
      if(x.kind==Kind::Fold) {
        auto binder=x.fold->binder_level;
        if(binder>=level) {
          if(binder==UINT32_MAX)throw std::overflow_error("fold binder overflow");
          ++binder;
        }
        result=graph.fold(x.fold->reducer,args.at(0),args.at(1),binder,x.fold->count,std::move(x.type));
      } else {
        result=x.kind == Kind::InitialCell ? graph.initialCell(std::move(x.name),args.at(0),std::move(x.type))
            : graph.operation(std::move(x.name),std::move(x.type),std::move(args),std::move(x.attributes));
        graph.markInstruction(result,instruction_mode::common(graph.instructionModes(a.node),graph.instructionModes(b.node)));
      }
    }
    if(!a.conversion.empty() || result.conversion.empty()) {
      result.conversion=a.conversion;
      result.converted_type=a.converted_type;
    }
    result.pointer_provenance=a.pointer_provenance;
    return result;
  }
};
}  // namespace

std::size_t ReductionBuilder::collect(std::span<Ref> live_roots) {
  return collect(live_roots, live_roots);
}

std::size_t ReductionBuilder::collect(std::span<Ref> live_roots,
                                      std::span<const Ref> extendable_roots,
                                      std::span<std::optional<Id>> weak_nodes,
                                      const Graph::RemapObserver& observer) {
  // Dense marks over this epoch's IDs: the membership answers of a node set,
  // without a heap node and rehash per live node.
  // Node IDs are topological: one descending sweep marks every descendant of a root,
  // reading live nodes in storage order. O(nodes + live edges).
  std::vector<char> live(graph_.size());
  const auto isLive=[&](Id id){return id<live.size() && live[id];};
  for(const auto& root:extendable_roots){graph_.node(root);live[root.node]=1;}
  const auto& nodes=graph_.nodes();
  for(Id id=live.size();id-->0;)if(live[id])
    for(const auto& child:nodes[id].operands)live[child.node]=1;
  // Live matcher entries leave their tables as node handles, in iteration
  // order; their Refs travel through the collector by move and return to the
  // same handles. No Run/Pending/Block copy and no node reallocation for a
  // survivor. extendable_roots (possibly aliasing live_roots) is not read
  // after marking. RealExpressionBuilder::collect already polled in refresh(),
  // so only terminal exceptions (allocation, invariant violations) can leave
  // after the first move. No poll may be added inside ReductionBuilder::collect,
  // Graph::collect or the remap observer it runs (RealExpressionBuilder::Impl::
  // remap): a recoverable Failure would lose the moved entries.
  const auto extractLive=[&](auto& table,auto& out) {
    for(auto it=table.begin();it!=table.end();) {
      const auto next=std::next(it);
      if(isLive(it->first.node))out.push_back(table.extract(it));
      it=next;
    }
  };
  std::vector<decltype(runs_)::node_type> runs;
  std::vector<decltype(pending_)::node_type> pending;
  std::vector<decltype(blocks_)::node_type> blocks;
  extractLive(runs_,runs);extractLive(pending_,pending);extractLive(blocks_,blocks);
  // Every Ref of an extracted entry, in one fixed order per table kind.
  // Counting, flattening into roots and restoring all walk this order, so
  // each Ref returns from its own root position.
  const auto forEachRef=[](auto& e,auto&& f) {
    auto& v=e.mapped();
    using Value=std::remove_reference_t<decltype(v)>;
    f(e.key());f(v.initial);
    if constexpr(std::is_same_v<Value,Run>){f(v.first);f(v.second);f(v.body);}
    else if constexpr(std::is_same_v<Value,Pending>)f(v.first);
    else {
      static_assert(std::is_same_v<Value,Block>);
      if(v.count)f(v.body);
      // A Ref shared by several entries' views is passed once per entry: each pass remaps
      // it to the same node.
      if(v.window) {
        for(auto i=v.begin;i<v.end;++i)f(v.window->terms[i]);
        for(auto i=v.begin;i<v.end;++i)f(v.window->prefixes[i]);
      }
      for(auto* values:{v.first.get(),v.second.get()})
        if(values)for(auto& value:*values)f(value);
    }
  };
  const auto forEachEntryRef=[&](auto&& f) {
    for(auto& e:runs)forEachRef(e,f);
    for(auto& e:pending)forEachRef(e,f);
    for(auto& e:blocks)forEachRef(e,f);
  };
  auto count=live_roots.size();
  forEachEntryRef([&](Ref&){++count;});
  std::vector<Ref> roots;roots.reserve(count);
  for(auto& r:live_roots)roots.push_back(std::move(r));
  // An entry's Ref goes to the collector as its node and owner only (the collector reads and
  // remaps no more of a root); its use strings stay in place. The collector's check of a
  // root's conversion fields is made here.
  forEachEntryRef([&](Ref& x){
    if(x.conversion.empty()!=x.converted_type.empty())
      throw std::invalid_argument("scalar-fold reference has incomplete conversion");
    roots.push_back(Ref{x.node,{},{},x.owner,{}});
  });
  const auto removed=graph_.collect(roots,weak_nodes,
      [&](std::span<const Id> remap,std::uint64_t owner) {
        // These caches describe immutable nodes, not a scheduling epoch.
        // Preserve surviving entries using the collector's existing ID map.
        // They are weak: neither a source nor its lifted result is rooted.
        std::vector<std::uint32_t> bounds;
        for(std::size_t old=0;old<binder_bounds_.size();++old) {
          const auto next=remap[old];
          if(next==std::numeric_limits<Id>::max())continue;
          if(bounds.size()<=next)bounds.resize(next+1,kUnknownBound);
          bounds[next]=binder_bounds_[old];
        }
        binder_bounds_=std::move(bounds);
        for(auto& [level,entries]:lifted_by_level_) {
          (void)level;
          std::unordered_map<Id,Ref> retained;
          for(const auto& [source,result]:entries) {
            const auto a=remap[source],b=remap[result.node];
            if(a==std::numeric_limits<Id>::max() || b==std::numeric_limits<Id>::max())continue;
            auto value=result;value.node=b;value.owner=owner;
            retained.emplace(a,std::move(value));
          }
          entries=std::move(retained);
        }
        if(observer)observer(remap,owner);
      });
  bounds_owner_=graph_.size()?graph_.ref(0).owner:0;
  template_ids_.clear();template_keys_.clear();parameter_differences_.clear();difference_keys_.clear();blocks_.clear();
  parameter_roots_.clear();parameter_keys_.clear();parameter_trees_={{std::nullopt,{}}};
  std::size_t position=0;
  for(auto& r:live_roots)r=std::move(roots[position++]);
  // The same keys go back in the same order into tables whose bucket count
  // clear() keeps, so each table's layout matches emplacing those entries;
  // the survivors' nodes are reused.
  runs_.clear();pending_.clear();
  const auto restore=[&](Ref& x){const auto& r=roots[position++];x.node=r.node;x.owner=r.owner;};
  for(auto& e:runs){forEachRef(e,restore);runs_.insert(std::move(e));}
  for(auto& e:pending){forEachRef(e,restore);pending_.insert(std::move(e));}
  for(auto& e:blocks){forEachRef(e,restore);blocks_.insert(std::move(e));}
  stats_.reclaimed+=removed;
  return removed;
}

void ReductionBuilder::refreshBounds() {
  // Node ids are stable only within one graph epoch; collection or replacement renumbers them.
  const auto owner=graph_.size()?graph_.ref(0).owner:0;
  if(owner!=bounds_owner_ || binder_bounds_.size()>graph_.size()) {
    binder_bounds_.clear();bounds_owner_=owner;
    template_ids_.clear();template_keys_.clear();parameter_differences_.clear();difference_keys_.clear();blocks_.clear();
    parameter_roots_.clear();parameter_keys_.clear();parameter_trees_={{std::nullopt,{}}};
    lifted_by_level_.clear();
  }
}

std::pair<std::size_t,bool> ReductionBuilder::KeyTable::intern(std::span<const std::uint64_t> key) {
  std::uint64_t h=0x9e3779b97f4a7c15ULL^key.size();
  for(const auto word:key) {
    h^=word+0x9e3779b97f4a7c15ULL+(h<<6)+(h>>2);
    h*=0xff51afd7ed558ccdULL;
  }
  const auto hash=static_cast<std::size_t>(h^(h>>29));
  if(2*(size()+1)>slots_.size())grow();
  const auto mask=slots_.size()-1;
  const auto equal=[&](std::size_t k) {
    const auto begin=starts_[k],end=k+1<starts_.size()?starts_[k+1]:words_.size();
    return end-begin==key.size() && std::equal(key.begin(),key.end(),words_.begin()+std::ptrdiff_t(begin));
  };
  for(auto i=hash&mask;;i=(i+1)&mask) {
    const auto slot=slots_[i];
    if(!slot) {
      starts_.push_back(words_.size());hashes_.push_back(hash);
      words_.insert(words_.end(),key.begin(),key.end());
      slots_[i]=starts_.size();
      return {starts_.size(),true};
    }
    if(hashes_[slot-1]==hash && equal(slot-1))return {slot,false};
  }
}
void ReductionBuilder::KeyTable::grow() {
  std::vector<std::size_t> grown(slots_.empty()?std::size_t{64}:2*slots_.size(),0);
  const auto mask=grown.size()-1;
  for(std::size_t k=0;k<starts_.size();++k) {
    auto i=hashes_[k]&mask;
    while(grown[i])i=(i+1)&mask;
    grown[i]=k+1;
  }
  slots_=std::move(grown);
}
void ReductionBuilder::KeyTable::clear() {
  words_.clear();starts_.clear();hashes_.clear();
  std::fill(slots_.begin(),slots_.end(),0);
}
std::uint64_t ReductionBuilder::stringId(const std::string& value) {
  if(value.empty())return 0;
  if(const auto found=string_ids_.find(value);found!=string_ids_.end())return found->second;
  return string_ids_.emplace(value,string_ids_.size()+1).first->second;
}

std::size_t ReductionBuilder::templateId(Ref value) {
  // Demand-driven postorder, once per reachable node per graph epoch. A small
  // candidate must not summarize unrelated outputs already stored in the DAG.
  // Keys contain direct child IDs, never serialized ancestor trees.
  // A key is the sequence of its fields: a node key (tag 0) has the node's kind, type,
  // name (or "<index>" for an address offset), attributes, binder level, fold (when
  // present) and, per operand, the operand's template ID and use (conversion, converted
  // type, pointer provenance); a use key (tag 1) has a template ID and a use. Two keys are
  // equal exactly when all their fields are, so IDs (first-insertion positions) identify
  // the same templates as length-prefixed key strings of those fields. O(node fields) per
  // node, without key strings.
  static const std::string index_name="<index>";
  auto& key=key_scratch_;
  template_ids_.resize(graph_.size());
  parameter_roots_.resize(graph_.size());
  std::vector<Id> work{value.node};
  while(!work.empty()) {
    const auto id=work.back();
    if(template_ids_[id][0]) {work.pop_back();continue;}
    const auto& n=graph_.node(id);
    bool ready=true;
    for(const auto& child:n.operands)if(!template_ids_.at(child.node)[0]) {
      work.push_back(child.node);ready=false;
    }
    if(!ready)continue;
    work.pop_back();
    ++stats_.template_nodes;
    stats_.template_edges+=n.operands.size();
    std::array<std::size_t,2> ids,parameter_ids;
    const auto node_offset=offset(n);
    const auto type=stringId(n.type),name=stringId(n.name),attributes=stringId(n.attributes);
    for(unsigned address=0;address<2;++address) {
      const unsigned side=address || n.kind==Kind::InitialCell;
      key.clear();
      key.push_back(0);key.push_back(std::uint64_t(n.kind));key.push_back(type);
      key.push_back(address && node_offset?stringId(index_name):name);
      key.push_back(attributes);key.push_back(n.index_level);
      key.push_back(n.fold.has_value());
      if(n.fold) {
        key.push_back(std::uint64_t(n.fold->reducer));key.push_back(n.fold->binder_level);key.push_back(n.fold->count);
      }
      for(const auto& child:n.operands) {
        key.push_back(template_ids_.at(child.node)[side]);
        key.push_back(stringId(child.conversion));key.push_back(stringId(child.converted_type));
        key.push_back(stringId(child.pointer_provenance));
      }
      ids[address]=template_keys_.intern(key).first;
      // Persistent parameter-only DAG. Unary numerical structure disappears;
      // shared parameter subtrees stay shared instead of copying leaf arrays.
      // Its key: (0, value) for an offset leaf, (1, children...) otherwise.
      const auto leaf=address?node_offset:std::nullopt;
      key.clear();
      if(leaf){key.push_back(0);key.push_back(*leaf);}
      else {
        key.push_back(1);
        for(const auto& child:n.operands)
          if(const auto root=parameter_roots_.at(child.node)[side])key.push_back(root);
      }
      if(!leaf && key.size()==1)parameter_ids[address]=0;
      else if(!leaf && key.size()==2)parameter_ids[address]=key[1];
      else {
        const auto [position,inserted]=parameter_keys_.intern(key);
        if(inserted) {
          if(position!=parameter_trees_.size())throw std::logic_error("parameter tree table out of step");
          ParameterTree tree;tree.value=leaf;
          if(!leaf)tree.children.assign(key.begin()+1,key.end());
          parameter_trees_.push_back(std::move(tree));
        }
        parameter_ids[address]=position;
      }
    }
    template_ids_[id]=ids;
    parameter_roots_[id]=parameter_ids;
  }
  key.clear();
  key.push_back(1);key.push_back(template_ids_.at(value.node)[0]);
  key.push_back(stringId(value.conversion));key.push_back(stringId(value.converted_type));
  key.push_back(stringId(value.pointer_provenance));
  return template_keys_.intern(key).first;
}

std::optional<std::size_t> ReductionBuilder::parameterDifference(std::size_t a,std::size_t b) {
  std::size_t remaining=4096,leaves=0;
  std::function<std::optional<std::size_t>(std::size_t,std::size_t,unsigned)> visit;
  visit=[&](std::size_t x,std::size_t y,unsigned depth)->std::optional<std::size_t> {
    if(x==y)return 0;
    const auto key=std::pair{x,y};
    if(auto it=parameter_differences_.find(key);it!=parameter_differences_.end())return it->second;
    if(!remaining || depth>128)return {};
    --remaining;++stats_.block_checks;
    const auto& p=parameter_trees_.at(x);const auto& q=parameter_trees_.at(y);
    std::string signature;
    if(p.value && q.value) {
      if(++leaves>256)return {};
      signature=*q.value>=*p.value?"+"+std::to_string(*q.value-*p.value):"-"+std::to_string(*p.value-*q.value);
    } else {
      if(p.value || q.value || p.children.size()!=q.children.size())return {};
      signature="c";
      for(std::size_t i=0;i<p.children.size();++i) {
        if(!remaining)return {};--remaining;++stats_.block_checks;
        auto child=visit(p.children[i],q.children[i],depth+1);if(!child)return {};
        signature+="/"+std::to_string(*child);
      }
    }
    const auto id=difference_keys_.try_emplace(std::move(signature),difference_keys_.size()+1).first->second;
    parameter_differences_.emplace(key,id);return id;
  };
  auto result=visit(a,b,0);
  // Cache failed roots too: the same oversized proposal is not retried on
  // each subsequent append. Successful subproblems remain reusable.
  parameter_differences_.insert_or_assign(std::pair{a,b},result);return result;
}

void ReductionBuilder::Block::push(const Ref& term,const Ref& prefix) {
  // A view that starts this far into its buffer moves to a fresh one: a buffer then holds at
  // most a constant number of dead entries per live one, and the copy is amortized over them.
  constexpr std::size_t compact_after=64;
  if(!window || window->terms.size()!=end || begin>=compact_after) {
    auto fresh=std::make_shared<Window>();
    if(window) {
      fresh->terms.assign(window->terms.begin()+std::ptrdiff_t(begin),window->terms.begin()+std::ptrdiff_t(end));
      fresh->prefixes.assign(window->prefixes.begin()+std::ptrdiff_t(begin),window->prefixes.begin()+std::ptrdiff_t(end));
    }
    window=std::move(fresh);end-=begin;begin=0;
  }
  window->terms.push_back(term);window->prefixes.push_back(prefix);++end;
}

std::optional<Ref> ReductionBuilder::appendBlock(Ref accumulator,Ref term,
    const std::string& type,std::uint32_t level) {
  constexpr std::size_t window=24,max_period=8,budget=4096;
  Block state;
  if(auto it=blocks_.find(accumulator);it!=blocks_.end() &&
      it->second.type==type && it->second.level==level)state=it->second;
  else {state.initial=accumulator;state.type=type;state.level=level;}
  state.push(term,accumulator);
  if(state.count) {
    const auto pos=state.size()-1;
    Match match{graph_,level,binder_bounds_,lifted_by_level_[level]};match.remaining=budget;
    const bool accepted=match.check((*state.first)[pos],(*state.second)[pos],term,state.count);
    stats_.block_checks+=budget-match.remaining;
    if(accepted) {
      if(state.size()==state.first->size()) {
        auto out=graph_.fold(Reducer::Sum,state.initial,state.body,level,++state.count,type);
        state.clearTerms();blocks_.insert_or_assign(out,std::move(state));
        ++stats_.block_extended;return out;
      }
    } else {
      state=Block{};state.initial=accumulator;state.type=type;state.level=level;
      state.push(term,accumulator);
    }
  } else {
    if(state.size()>window) {
      ++state.begin;
      state.initial=state.prefix(0);
    }
    // KMP proposes ONE shortest structural period; no loop over block lengths.
    std::vector<std::size_t> ids,pi(state.size());
    for(std::size_t i=0;i<state.size();++i)ids.push_back(templateId(state.term(i)));
    for(std::size_t i=1;i<ids.size();++i) {
      auto j=pi[i-1];while(j && ids[i]!=ids[j])j=pi[j-1];
      if(ids[i]==ids[j])++j;pi[i]=j;
    }
    auto p=ids.size()-pi.back();
    // Equal shapes may still have a periodic index increment (0,1,8,9,...).
    // Extract bounded parameter vectors, then run the same single-candidate
    // period detector on exact differences. Never try all possible lengths.
    if(p==1 && ids.size()>=6) {
      std::vector<std::size_t> differences;
      bool complete=true;
      for(std::size_t i=1;i<state.size();++i) {
        auto difference=parameterDifference(parameter_roots_.at(state.term(i-1).node)[0],
                                             parameter_roots_.at(state.term(i).node)[0]);
        if(!difference){complete=false;break;}
        differences.push_back(*difference);
      }
      if(complete) {
        std::vector<std::size_t> prefix(differences.size());
        for(std::size_t i=1;i<differences.size();++i) {
          auto j=prefix[i-1];while(j && differences[i]!=differences[j])j=prefix[j-1];
          if(differences[i]==differences[j])++j;prefix[i]=j;
        }
        p=differences.size()-prefix.back();
      }
    }
    if(p>1 && p<=max_period && ids.size()==3*p) {
      ++stats_.block_candidates;
      Match match{graph_,level,binder_bounds_,lifted_by_level_[level]};match.remaining=budget;
      bool accepted=true;
      for(std::size_t i=0;i<p && accepted;++i)
        accepted=match.check(state.term(i),state.term(p+i),state.term(2*p+i),2);
      stats_.block_checks+=budget-match.remaining;
      if(!accepted)++stats_.block_match_rejected;
      // Varying pairs were bounded by check(). Fixed subgraphs use the shared
      // (graph epoch, binder level, node) lifting cache, not a second ancestor
      // preflight per candidate. First-time lifting costs the affected DAG;
      // later bodies at this level reuse it. This is not O(1) construction.
      if(accepted) {
        auto body=graph_.constant(type,"0");
        for(std::size_t i=0;i<p;++i) {
          body=graph_.operation("Add",type,{body,match.body(state.term(i),state.term(p+i))});
          graph_.markInstruction(body,instruction_mode::synthesized);
        }
        const auto terms=state.window->terms.begin()+std::ptrdiff_t(state.begin);
        state.first=std::make_shared<std::vector<Ref>>(terms,terms+std::ptrdiff_t(p));
        state.second=std::make_shared<std::vector<Ref>>(terms+std::ptrdiff_t(p),terms+std::ptrdiff_t(2*p));
        state.body=body;state.count=3;state.clearTerms();
        auto out=graph_.fold(Reducer::Sum,state.initial,body,level,3,type);
        blocks_.insert_or_assign(out,std::move(state));++stats_.block_created;return out;
      }
    }
  }
  auto out=graph_.operation("Add",type,{accumulator,term});
  blocks_.insert_or_assign(out,std::move(state));return out;
}

Ref ReductionBuilder::appendFold(Reducer reducer,Ref accumulator,Ref body,
    std::uint32_t level,std::uint64_t count,std::string type) {
  // Integer/PTX bit arithmetic must not inherit ideal-real reassociation.
  const bool real=type=="real" || type=="f16" || type=="bf16" || type=="f32" || type=="f64";
  if(!real || !count)return graph_.fold(reducer,accumulator,body,level,count,std::move(type));
  const auto& previous=graph_.node(accumulator);
  if(reducer==Reducer::Sum && accumulator.conversion.empty() &&
     accumulator.converted_type.empty() && accumulator.pointer_provenance.empty() &&
     previous.kind==Kind::Fold && previous.type==type && previous.attributes.empty() &&
     previous.fold->reducer==reducer && previous.fold->binder_level==level &&
     previous.fold->count<=std::numeric_limits<std::uint64_t>::max()-count &&
     consecutiveBody(graph_,previous.operands[1],body,level,previous.fold->count)) {
    // Copy only the two refs before intern can reallocate graph storage.
    const auto initial=previous.operands[0],combined_body=previous.operands[1];
    const auto combined_count=previous.fold->count+count;
    return graph_.fold(reducer,initial,combined_body,level,combined_count,std::move(type));
  }
  const auto unit=graph_.constant(type,identity(reducer));
  const auto chunk=graph_.fold(reducer,unit,body,level,count,type);
  const auto joined=append(reducer,accumulator,chunk,type,level);
  if(runs_.contains(joined))return joined;
  const auto pending=pending_.find(joined);
  if(pending==pending_.end())return joined; // Existing matcher proved a Fold.
  // append's fallback is an ordinary binary reduction. Spell this certified
  // chunk with its original initializer, and keep exactly the same matcher
  // state for a later chunk. No source expression traversal or K enumeration.
  const auto state=pending->second;
  const auto out=graph_.fold(reducer,accumulator,body,level,count,type);
  pending_.insert_or_assign(out,state);
  return out;
}

Ref ReductionBuilder::append(Reducer reducer, Ref accumulator, Ref term,
                              std::string type, std::uint32_t level) {
  if (graph_.type(accumulator) != type || graph_.type(term) != type)
    throw std::invalid_argument("reduction operand types differ");
  // Fixed-width integer arithmetic retains ordinary operation nodes.
  const bool real=type == "real" || type == "f16" || type == "bf16" ||
      type == "f32" || type == "f64";
  if (real) {
    refreshBounds();
    if (const auto it=runs_.find(accumulator);it!=runs_.end()) {
      // check() and fold() do not touch runs_; the one inserted copy is below.
      const auto& run=it->second;
      if (run.reducer==reducer && run.type==type && run.level==level &&
          run.count<std::numeric_limits<std::uint64_t>::max()) {
        Match match{graph_,level,binder_bounds_,lifted_by_level_[level]};
        if (match.check(run.first,run.second,term,run.count)) {
          auto out=graph_.fold(reducer,run.initial,run.body,level,run.count+1,type);
          auto extended=run; ++extended.count;
          runs_.insert_or_assign(out,std::move(extended));
          ++stats_.extended;
          return out;
        }
        ++stats_.rejected;
      }
    }
    if (const auto it=pending_.find(accumulator);it!=pending_.end()) {
      // Matching and graph interning never modify pending_; both references
      // are read before the insertion below, which targets runs_.
      const auto& second=it->second;
      const auto first=pending_.find(second.initial);
      const auto compatible=[&](const Pending& p) {
        return p.reducer==reducer && p.type==type && p.level==level;
      };
      if (compatible(second) && first!=pending_.end() && compatible(first->second)) {
        const auto& prefix=first->second;
        Match match{graph_,level,binder_bounds_,lifted_by_level_[level]};
        if (match.check(prefix.first,second.first,term,2)) {
          auto body=match.body(prefix.first,second.first);
          auto out=graph_.fold(reducer,prefix.initial,body,level,3,type);
          runs_.insert_or_assign(out,Run{reducer,prefix.initial,prefix.first,second.first,body,3,level,type});
          ++stats_.created;
          return out;
        }
        ++stats_.rejected;
      } else if (compatible(second) && first==pending_.end() && !runs_.contains(second.initial)) {
        const auto& initial=graph_.node(second.initial);
        const auto& sample=graph_.node(second.first);
        const auto unit=identity(reducer);
        const bool explicit_identity=initial.kind==Kind::Constant && initial.name==unit;
        if (!explicit_identity && initial.kind==sample.kind) {
        Match match{graph_,level,binder_bounds_,lifted_by_level_[level]};
          if (match.check(second.initial,second.first,term,2)) {
            auto body=match.body(second.initial,second.first);
            auto init=graph_.constant(type,unit);
            auto out=graph_.fold(reducer,init,body,level,3,type);
            runs_.insert_or_assign(out,Run{reducer,init,second.initial,second.first,body,3,level,type});
            ++stats_.created;
            return out;
          }
          ++stats_.rejected;
        }
      }
    }
  }
  auto out=graph_.operation(operation(reducer),type,{accumulator,term});
  if(real && reducer==Reducer::Sum)out=*appendBlock(accumulator,term,type,level);
  if (real) pending_.insert_or_assign(out,Pending{reducer,accumulator,term,level,std::move(type)});
  return out;
}
}  // namespace sentryward::scalar_fold
