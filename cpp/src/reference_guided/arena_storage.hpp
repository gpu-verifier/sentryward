#pragma once
#include "core.hpp"

namespace rg {

// std::hash<string_view> equals std::hash<string> on the same characters.
inline auto Arena::hash(std::string_view op, std::span<const Id> as,
                   std::span<const Q> ds, std::string_view tag) -> size_t {
  size_t h = std::hash<std::string_view>{}(op);
  mix(h, std::hash<std::string_view>{}(tag));
  for (auto x : as)
    mix(h, x);
  for (const auto &x : ds)
    mix(h, hashq(x));
  return h;
}

inline auto Arena::find_node(size_t h, std::string_view op, std::span<const Id> as,
                            std::span<const Q> ds, std::string_view tag) const -> std::optional<Id> {
  if (intern_index.empty())
    return {};
  for (size_t m = intern_index.size() - 1, i = index_home(h, intern_index_shift);;
       i = (i + 1) & m) {
    const auto &s = intern_index[i];
    if (s.id == invalid)
      return {};
    if (s.hash != h)
      continue;
    const auto &n = nodes[s.id];
    if (n.op == op && std::equal(n.args.begin(), n.args.end(), as.begin(), as.end()) &&
        std::equal(n.data.begin(), n.data.end(), ds.begin(), ds.end()) && n.tag == tag)
      return s.id;
  }
}

inline auto Arena::intern(std::string_view op, std::vector<Id> as, std::vector<Q> ds,
          std::string_view tag, Facts f, Bounds b) -> Id {
  return intern_lazy(op, as, ds, tag, [&](size_t h) {
    return insert_node(h, op, std::move(as), std::move(ds), tag, f, std::move(b));
  });
}

inline auto Arena::insert_node(size_t h, std::string_view op, std::vector<Id> as, std::vector<Q> ds,
               std::string_view tag, Facts f, Bounds b) -> Id {
  if (nodes.size() >= invalid)
    throw Unsupported("arena ID capacity");
  Id id = Id(nodes.size());
  std::vector<Id> fs;
  for (auto x : as) {
    std::vector<Id> tmp;
    std::set_union(fs.begin(), fs.end(), nodes[x].free.begin(),
                   nodes[x].free.end(), std::back_inserter(tmp));
    fs.swap(tmp);
  }
  if (op == "var")
    fs = {id};
  if (op == "fold") {
    fs = nodes[as[1]].free;
    Id v = ds[0].numerator().convert_to<Id>();
    fs.erase(std::remove(fs.begin(), fs.end(), v), fs.end());
    std::vector<Id> tmp;
    std::set_union(fs.begin(), fs.end(), nodes[as[0]].free.begin(),
                   nodes[as[0]].free.end(), std::back_inserter(tmp));
    fs.swap(tmp);
    if (!b) b = boolean_fold_bounds(tag, as, ds);
  }
  // Grow before appending: a failed allocation leaves nodes and the index
  // unchanged, and placing the new slot after push_back cannot throw.
  reserve_index();
  nodes.push_back({std::string(op), std::string(tag), std::move(as),
                   std::move(ds), f, std::move(b), std::move(fs)});
  place_index(intern_index, intern_index_shift, {h, id});
  return id;
}

// The bounds of a Max fold max(s, e(0), ..., e(n-1)) whose initial s and body e are finite and
// both lie within [0, 1] (a predicate fold: the warp vote of the value view, vote.any as Fold Max)
// or both within [-1, 0] (the importer's Min fold over predicates, -max(-s, -e(k))): for n >= 1
// [max(l_s, l_e), max(u_s, u_e)], for n = 0 the initial's bounds. The body's bounds hold at every
// binder value (the binder is a variable of its own range [0, n - 1]), and the maximum of finitely
// many values in an interval lies in it. Exact. Every other fold keeps no bounds. O(1), at creation
// of the node, so every constructor of a fold gives the same node the same bounds.
inline auto Arena::boolean_fold_bounds(std::string_view tag, const std::vector<Id> &as,
                           const std::vector<Q> &ds) -> Bounds {
  if (tag != "Max" || as.size() != 2 || ds.size() != 2) return {};
  const Q &count = ds[1];
  if (count.denominator() != 1 || count < 0) return {};
  const auto &s = nodes[as[0]], &e = nodes[as[1]];
  const auto within = [](const Node &n, const Q &lo, const Q &hi) {
    return n.facts.finite && n.bounds && n.bounds->first >= lo && n.bounds->second <= hi;
  };
  if (!(within(s, Q(0), Q(1)) && within(e, Q(0), Q(1))) &&
      !(within(s, Q(-1), Q(0)) && within(e, Q(-1), Q(0))))
    return {};
  ++stats["boolean_fold_bounds"];
  if (count == 0) return s.bounds;
  return std::pair(std::max(s.bounds->first, e.bounds->first),
                   std::max(s.bounds->second, e.bounds->second));
}

inline auto Arena::reserve_index() -> void {
  if (4 * (nodes.size() + 1) <= 3 * intern_index.size())
    return;
  const size_t size = intern_index.empty() ? 256 : 2 * intern_index.size();
  std::vector<IndexSlot> grown(size, IndexSlot{0, invalid});
  const unsigned shift = 64 - unsigned(std::countr_zero(size));
  for (const auto &s : intern_index)
    if (s.id != invalid)
      place_index(grown, shift, s);
  intern_index.swap(grown);
  intern_index_shift = shift;
}

inline auto Arena::place_index(std::vector<IndexSlot> &slots, unsigned shift, IndexSlot s) noexcept -> void {
  for (size_t m = slots.size() - 1, i = index_home(s.hash, shift);; i = (i + 1) & m)
    if (slots[i].id == invalid) {
      slots[i] = s;
      return;
    }
}

// One coordinate witness, not an element enumeration or a numerical-DAG
// substitution. Invariant nodes are reused by identity. All callers use
// this same bounded interpreter before constructing the reduction node.
inline auto Arena::coordinate_at_zero(Id root,Id axis) -> Id {
  auto key=std::pair(root,axis);
  if(auto it=coordinate_witness_cache.find(key);it!=coordinate_witness_cache.end())return it->second;
  std::unordered_map<Id,Id> memo;
  std::vector<std::pair<Id,bool>> todo{{root,false}};
  size_t work=0;
  while(!todo.empty()) {
    check();auto [id,ready]=todo.back();todo.pop_back();
    if(memo.contains(id))continue;
    if(++work>512)return coordinate_witness_cache.emplace(key,invalid).first->second;
    if(id==axis){memo[id]=zero;continue;}
    if(!depends(id,axis)){memo[id]=id;continue;}
    const auto &n=nodes[id];const auto &op=n.op;
    bool supported=op=="add" || op=="add_raw" || op=="mul" ||
      op=="compare" || op=="select" || op=="max" || op=="idiv" ||
      op=="imod" || op=="word_or" || op=="word_shr" || op=="word_shl" || op=="signed" || op=="extractbits";
    if(!supported || n.args.size()>32){memo[id]=invalid;continue;}
    if(!ready) {
      todo.emplace_back(id,true);
      for(Id child:n.args)if(!memo.contains(child))todo.emplace_back(child,false);
      continue;
    }
    std::vector<Id> xs;
    for(Id child:n.args)xs.push_back(memo.at(child));
    if(std::find(xs.begin(),xs.end(),invalid)!=xs.end()){memo[id]=invalid;continue;}
    Id result=invalid;
    try {
      if(op=="add" || op=="add_raw") {
        if(xs.size()==n.data.size()) {
          for(size_t i=0;i<xs.size();++i)xs[i]=mul({constant(n.data[i]),xs[i]});
          result=add(xs);
        }
      } else if(op=="mul" && n.data.size()==1) {
        xs.push_back(constant(n.data[0]));result=mul(xs);
      } else if(op=="compare" && xs.size()==2)result=compare(n.tag,xs[0],xs[1]);
      else if(op=="select" && xs.size()==3)result=select(xs[0],xs[1],xs[2],true);
      else if(op=="max")result=maximum(xs);
      else if((op=="idiv" || op=="imod") && xs.size()==1 && n.data.size()==1 &&
              n.data[0].denominator()==1 && n.data[0]>0)
        result=op=="idiv"?idiv(xs[0],n.data[0].numerator()):imod(xs[0],n.data[0].numerator());
      else if(op=="word_or" && xs.size()==2 && n.data.size()==1)
        result=word_or(xs[0],xs[1],n.data[0].numerator().convert_to<unsigned>());
      else if(op=="word_shr" && xs.size()==2 && n.data.size()==2)
        result=word_shr(xs[0],xs[1],n.data[0].numerator().convert_to<unsigned>(),n.data[1]!=0);
      else if(op=="word_shl" && xs.size()==2 && n.data.size()==1)
        result=word_shl(xs[0],xs[1],n.data[0].numerator().convert_to<unsigned>());
      else if(op=="signed" && xs.size()==1 && n.data.size()==1)
        result=signed_word(xs[0],n.data[0].numerator().convert_to<unsigned>());
      else if(op=="extractbits" && xs.size()==1 && n.data.size()==2)
        result=bit_slice(xs[0],n.data[0].numerator().convert_to<unsigned>(),n.data[1].numerator().convert_to<unsigned>());
    } catch(const Unsupported &) {result=invalid;}
    memo[id]=result;++stats["sum_witness_coordinate_nodes"];
  }
  return coordinate_witness_cache.emplace(key,memo.at(root)).first->second;
}

inline auto Arena::reduction_witness_at_zero(Id root,Id axis,const Z &count,bool positive) -> bool {
  if(count<=0 || axis==invalid)return false;
  auto key=std::tuple(root,axis,positive);
  if(auto it=reduction_witness_cache.find(key);it!=reduction_witness_cache.end())return it->second;
  Id id=root;bool result=false;
  for(size_t work=0;work<128;++work) {
    check();const auto &n=nodes[id];
    if(n.facts.finite && (!positive || n.facts.pos)){result=true;break;}
    if(positive?(!n.facts.finite || !n.facts.nn):!n.facts.finite_or_negative_infinity())break;
    if(n.op!="select" || n.args.size()!=3)break;
    Id p=coordinate_at_zero(n.args[0],axis);
    if(p!=one && p!=zero)break;
    id=n.args[p==one?1:2];
  }
  if(result)++stats[positive?"sum_positive_coordinate_witnesses":"max_finite_coordinate_witnesses"];
  return reduction_witness_cache.emplace(key,result).first->second;
}

inline auto Arena::fold_facts(Id initial,Id body,const Q &count,bool maximum,Id axis) -> Facts {
  if(count==0)return nodes[initial].facts;
  if(count<0)return {};
  const auto &s=nodes[initial].facts,&b=nodes[body].facts;
  bool extended=s.finite_or_negative_infinity() && b.finite_or_negative_infinity();
  if(maximum) {
    bool finite=extended && (s.finite || b.finite);
    if(!finite && extended && count.denominator()==1)
      finite=reduction_witness_at_zero(body,axis,count.numerator(),false);
    return {finite,finite && (s.pos || b.pos),finite && (s.nn || b.nn),extended};
  }
  bool nn=s.nn && b.nn;
  bool finite=s.finite && b.finite,positive=nn && (s.pos || b.pos);
  if(finite && nn && !positive && count.denominator()==1)
    positive=positive_at_zero(body,axis,count.numerator());
  return {finite,positive,nn,extended};
}

} // namespace rg
