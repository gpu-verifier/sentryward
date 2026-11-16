#pragma once
#include "core.hpp"

namespace rg {

inline auto Arena::bit_slice(Id source,unsigned lo,unsigned count) -> Id {
  // Every actual one-bit producer, including direct PTX extraction, prepares
  // the same result consumed by signed-comparison probes. Key before local
  // slice composition. Do not memoize every multi-byte FP decoder slice.
  if(count!=1)return compute_bit_slice(source,lo,count);
  auto key=std::tuple(source,lo,count);
  if(auto it=resolved_bit_slice_cache.find(key);it!=resolved_bit_slice_cache.end())return it->second;
  Id result=compute_bit_slice(source,lo,count);
  // Local composition can have computed only the child's facts. Prepare
  // the original query source here, never later in a speculative Max probe.
  if(!known_bits_cache.contains(source))known_bits(source);
  resolved_bit_slice_cache.emplace(key,result);
  return result;
}

inline auto Arena::compute_bit_slice(Id source,unsigned lo,unsigned count) -> Id {
  if(count==0)return zero;
  if(count>256)throw Unsupported("bit slice width");
  // Preserve truncation when composing slices. This is one local shell,
  // not an ancestor search. A slice past an unsigned field reads zero; a
  // partially overlapping slice retains only the remaining low bits.
  const auto &original=nodes[source];
  bool slice=original.op=="extractbits" && original.args.size()==1 && original.data.size()==2;
  bool truncation=original.op=="word" && original.args.size()==1 && original.data.size()==1;
  if((slice || truncation) && original.facts.finite) {
    Q start=slice?original.data[0]:Q(0), field_width=original.data[slice?1:0];
    if(start.denominator()==1 && start>=0 && start<=Q(UINT_MAX) &&
       field_width.denominator()==1 && field_width>=0 && field_width<=256) {
      unsigned w=field_width.numerator().convert_to<unsigned>();
      if(lo>=w){++stats["bit_slice_truncation_zeros"];return zero;}
      const uint64_t combined=uint64_t(lo)+start.numerator().convert_to<unsigned>();
      if(combined<=UINT_MAX) {
        count=std::min(count,w-lo);lo=unsigned(combined);source=original.args[0];
        ++stats["bit_slice_local_compositions"];
      }
    }
  }
  if(auto v=value(source);v && v->denominator()==1) {
    const Z integer=v->numerator(), magnitude=integer<0?-integer:integer;
    // Avoid allocating 2^lo for a large offset once its result is known.
    if(magnitude==0 || (lo>0 && Z(boost::multiprecision::msb(magnitude))<Z(lo)))
      return integer<0?constant(Q(pow2(count)-1)):zero;
    return constant(Q(modz(floorz(integer,pow2(lo)),pow2(count))));
  }
  const auto source_bounds=nodes[source].bounds;
  const auto bits=known_bits(source);
  if(bits.integer && nodes[source].facts.finite && source_bounds) {
    // All supported word payloads fit 256 bits. Compare safely without
    // unsigned lo+count arithmetic, or a huge allocation for a large lo.
    unsigned cutoff=std::min(lo,256u);
    if(source_bounds->first>=0 && source_bounds->second<Q(pow2(cutoff))) {
      ++stats["bit_slice_range_zeros"];return zero;
    }
    if(lo>=256 && source_bounds->first>=Q(-pow2(256)) && source_bounds->second<0)
      return constant(Q(pow2(count)-1));
  }
  if(lo<256 && count<=256-lo) {
    const Z mask=pow2(count)-1;
    if(bits.integer && (((bits.zero|bits.one)>>lo)&mask)==mask) {
      ++stats["bit_slice_known_values"];
      return constant(Q((bits.one>>lo)&mask));
    }
  }
  // One bit of a word built from symbolic left shifts; only words reaching a word_shl leaf try it.
  if(count==1 && lo<256)
    if(auto bit=shifted_word_bit(source,constant(Q(lo))))return *bit;
  if(nodes[source].op=="word_or") {
    // Slice the stored OR DAG once per requested window, without recursive
    // descent or enumerating the member domain.
    std::vector<std::pair<Id,bool>> todo{{source,false}};
    while(!todo.empty()) {
      auto [id,ready]=todo.back();todo.pop_back();check();
      auto key=std::tuple(id,lo,count);if(resolved_bit_slice_cache.contains(key))continue;
      const auto& n=nodes[id];
      if(n.op!="word_or") {resolved_bit_slice_cache.emplace(key,bit_slice(id,lo,count));continue;}
      if(!ready) {todo.emplace_back(id,true);for(Id arg:n.args)todo.emplace_back(arg,false);continue;}
      Id l=resolved_bit_slice_cache.at(std::tuple(n.args[0],lo,count)),r=resolved_bit_slice_cache.at(std::tuple(n.args[1],lo,count));
      resolved_bit_slice_cache.emplace(key,word_or(l,r,count));++stats["word_or_slice_nodes"];
    }
    return resolved_bit_slice_cache.at(std::tuple(source,lo,count));
  }
  if(nodes[source].op=="word_shr") {
    // Copy scalar fields before interning; Arena storage can reallocate.
    auto args=nodes[source].args;auto data=nodes[source].data;
    unsigned width=data[0].numerator().convert_to<unsigned>();
    auto v=value(args[0]);
    if(data[1]==0 && v && *v==Q(pow2(width)-1)) {
      std::vector<Id> bits;
      for(unsigned b=0;lo<width && b<count && b<width-lo;++b)
        bits.push_back(mul({constant(Q(pow2(b))),compare("lt",args[1],constant(Q(width-lo-b)))}));
      return add(bits);
    }
    if(auto view=shifted_byte_view_slice(source,lo,count))return *view;
  }
  const auto &n=nodes[source];
  if(n.op=="readbits" && lo%8==0 && count%8==0 && count>0 && Q(lo)+Q(count)<=n.data[0]) {
    std::string tag=n.tag;Id address=n.args[0];
    return intern("readbits",{add({address,constant(Q(lo/8))})},{Q(count)},tag,
                  {true,false,true},std::pair(Q(0),Q(pow2(count)-1)));
  }
  return intern("extractbits",{source},{Q(lo),Q(count)},"",{true,false,true},
                std::pair(Q(0),Q(pow2(count)-1)));
}

// Whether a word_shl leaf is reachable from root through the shells shifted_word_bit reads (Select
// arms, word_or, word, signed, extractbits, idiv, imod, the source of an unsigned word_shr, sum
// terms). Memoized per node: one visit per shell node over the Arena's lifetime.
inline auto Arena::shift_dependent(Id root) -> bool {
  if(!word_shl_requests) return false;
  std::vector<std::pair<Id,bool>> todo{{root,false}};
  while(!todo.empty()) {
    auto [id,ready]=todo.back();todo.pop_back();check();
    if(shift_dependent_cache.contains(id))continue;
    const auto &n=nodes[id];
    if(n.op=="word_shl"){shift_dependent_cache.emplace(id,true);continue;}
    size_t first=0,last=0;
    if(n.op=="select" && n.args.size()==3){first=1;last=3;}
    else if(n.op=="word_shr" && n.args.size()==2){first=0;last=1;}
    else if(n.op=="word_or" || n.op=="word" || n.op=="signed" || n.op=="add" || n.op=="extractbits" ||
            n.op=="idiv" || n.op=="imod"){first=0;last=n.args.size();}
    if(first==last){shift_dependent_cache.emplace(id,false);continue;}
    const std::vector<Id> args(n.args.begin()+first,n.args.begin()+last);
    if(!ready) {
      todo.emplace_back(id,true);
      for(Id arg:args)if(!shift_dependent_cache.contains(arg))todo.emplace_back(arg,false);
      continue;
    }
    bool found=false;
    for(Id arg:args)found=found || shift_dependent_cache.at(arg);
    shift_dependent_cache.emplace(id,found);
  }
  return shift_dependent_cache.at(root);
}

// Resolve a bit in [0,256) through exact word, slice, shift, OR and Select shells.
// A word_shl constant contributes [shift == position-k] for each set bit k;
// ((1 << shift)-1) contributes [shift > position], including PTX's clamped shifts.
// Only a result decided by a word_shl leaf is accepted. Unknown shells decline.
// After the cached dependency check, visit each (node, position) at most once,
// up to shifted_bit_visit_limit. Guard absorption must agree on both arms.
inline auto Arena::shifted_word_bit(Id v, Id position) -> std::optional<Id> {
  if(!valid_bit_position(position)) return {};
  if(!shift_dependent(v)) return packed_predicate_bit(v,position);
  ShiftedBitMemo memo;
  size_t visits=0;
  const auto bit=visit_shifted_bit(v,position,0,memo,visits);
  if(!bit || !bit->shifted) return {};
  ++stats["shifted_word_bits"];
  return bit->value;
}

// Whether a Select of two integer constants is reachable from root through the shells the bit
// analysis reads (Select arms, word_or, word, signed, extractbits, idiv, imod, the source of an
// unsigned word_shr, sum terms): a word assembled from predicates, as a validity mask packed into
// bits is. Memoized per node: one visit per shell node over the Arena's lifetime.
inline auto Arena::predicate_packed_word(Id root) -> bool {
  std::vector<std::pair<Id,bool>> todo{{root,false}};
  while(!todo.empty()) {
    auto [id,ready]=todo.back();todo.pop_back();check();
    if(predicate_packed_cache.contains(id))continue;
    const auto &n=nodes[id];
    size_t first=0,last=0;
    if(n.op=="select" && n.args.size()==3) {
      const auto yes=value(n.args[1]), no=value(n.args[2]);
      if(yes && no && yes->denominator()==1 && no->denominator()==1){predicate_packed_cache.emplace(id,true);continue;}
      first=1;last=3;
    }
    else if(n.op=="word_shr" && n.args.size()==2){first=0;last=1;}
    else if(n.op=="word_or" || n.op=="word" || n.op=="signed" || n.op=="add" || n.op=="extractbits" ||
            n.op=="idiv" || n.op=="imod"){first=0;last=n.args.size();}
    if(first==last){predicate_packed_cache.emplace(id,false);continue;}
    const std::vector<Id> args(n.args.begin()+first,n.args.begin()+last);
    if(!ready) {
      todo.emplace_back(id,true);
      for(Id arg:args)if(!predicate_packed_cache.contains(arg))todo.emplace_back(arg,false);
      continue;
    }
    bool found=false;
    for(Id arg:args)found=found || predicate_packed_cache.at(arg);
    predicate_packed_cache.emplace(id,found);
  }
  return predicate_packed_cache.at(root);
}

// Bit `position` (a constant or symbolic integer in [0, 256)) of a word assembled from predicate
// Selects and holding no symbolic left shift: the bit analysis of shifted_word_bit in packed mode,
// which adds the carry-free sum of terms with disjoint possible one bits (compute_shifted_bit). A
// result exists only when a Select of two constant bits decided it (the bit is then that Select's
// Boolean condition, its complement, or a Boolean combination of such), so a word without one keeps
// its form. Exact. Cost: as shifted_word_bit, after the memoized predicate_packed_word check.
inline auto Arena::packed_predicate_bit(Id v, Id position) -> std::optional<Id> {
  if(packed_bit_mode || !predicate_packed_word(v)) return {};
  struct Mode { bool &flag; ~Mode(){flag=false;} } mode{packed_bit_mode};
  packed_bit_mode=true;
  ShiftedBitMemo memo;
  size_t visits=0;
  const auto bit=visit_shifted_bit(v,position,0,memo,visits);
  if(!bit || !bit->predicate) return {};
  ++stats["packed_predicate_bits"];
  return bit->value;
}

// v & mask for a constant mask: the sum over the one bits b of the mask of 2^b times bit b of v
// (shifted_word_bit). Every bit must be decided, one of them by a word_shl leaf.
// O(popcount(mask)) bit queries sharing one memo, each within shifted_bit_visit_limit visits.
inline auto Arena::shifted_word_mask(Id v, const Z &mask) -> std::optional<Id> {
  if(mask<=0 || mask>=pow2(256)) return {};
  if(!shift_dependent(v)) return packed_predicate_mask(v,mask);
  ShiftedBitMemo memo;
  bool shifted=false;
  std::vector<Id> bits;
  for(unsigned b=0;b<256 && pow2(b)<=mask;++b) {
    if(!boost::multiprecision::bit_test(mask,b)) continue;
    size_t visits=0;
    const auto bit=visit_shifted_bit(v,constant(Q(b)),0,memo,visits);
    if(!bit) return {};
    shifted=shifted || bit->shifted;
    bits.push_back(mul({constant(Q(pow2(b))),bit->value}));
  }
  if(!shifted) return {};
  ++stats["shifted_word_masks"];
  return add(std::move(bits));
}

// v & mask for a word assembled from predicate Selects (packed_predicate_bit): every mask bit must
// be decided, one of them by a predicate Select. O(popcount(mask)) bit queries sharing one memo.
inline auto Arena::packed_predicate_mask(Id v, const Z &mask) -> std::optional<Id> {
  if(packed_bit_mode || !predicate_packed_word(v)) return {};
  struct Mode { bool &flag; ~Mode(){flag=false;} } mode{packed_bit_mode};
  packed_bit_mode=true;
  ShiftedBitMemo memo;
  bool predicate=false;
  std::vector<Id> bits;
  for(unsigned b=0;b<256 && pow2(b)<=mask;++b) {
    if(!boost::multiprecision::bit_test(mask,b)) continue;
    size_t visits=0;
    const auto bit=visit_shifted_bit(v,constant(Q(b)),0,memo,visits);
    if(!bit) return {};
    predicate=predicate || bit->predicate;
    bits.push_back(mul({constant(Q(pow2(b))),bit->value}));
  }
  if(!predicate) return {};
  ++stats["packed_predicate_masks"];
  return add(std::move(bits));
}

inline auto Arena::visit_shifted_bit(Id v, Id position, unsigned depth, ShiftedBitMemo &memo,
                                            size_t &visits) -> std::optional<ShiftedBit> {
  check();
  if(auto it=memo.find(std::pair(v,position));it!=memo.end()) return it->second;
  if(depth>shifted_bit_depth_limit || ++visits>shifted_bit_visit_limit) return {};
  auto result=compute_shifted_bit(v,position,depth,memo,visits);
  memo.emplace(std::pair(v,position),result);
  return result;
}

inline auto Arena::compute_shifted_bit(Id v, Id position, unsigned depth, ShiftedBitMemo &memo,
                                              size_t &visits) -> std::optional<ShiftedBit> {
  if(!valid_bit_position(position)) return {};
  const Q low=nodes[position].bounds->first, high=nodes[position].bounds->second;
  const auto below=[&](const Q &width) { return high<width; };
  const auto decided=[&](Id value, bool shifted) {
    ShiftedBit out; out.value=value; out.shifted=shifted; return out;
  };
  if(auto c=value(v)) {
    if(c->denominator()!=1) return {};
    const Z x=c->numerator();
    const unsigned lo=floorq(low).convert_to<unsigned>(), hi=floorq(high).convert_to<unsigned>();
    const Z field=modz(floorz(x,pow2(lo)),pow2(hi-lo+1));
    if(field==0) return decided(zero,false);
    if(field==pow2(hi-lo+1)-1) return decided(one,false);
    return {};
  }
  // Copy the fields: constructors below may grow the Arena.
  const std::string op=nodes[v].op;
  const std::vector<Id> args=nodes[v].args;
  const std::vector<Q> data=nodes[v].data;
  if(op=="word_shl" && args.size()==2 && data.size()==1) {
    if(low>=data[0]) return decided(zero,true);
    const auto c=value(args[0]);
    if(!below(data[0]) || !c || c->denominator()!=1 || *c<=0) return {};
    const Z source=c->numerator();
    const unsigned width=data[0].numerator().convert_to<unsigned>();
    std::vector<Id> terms;
    Id rhs=invalid;
    for(unsigned k=0;k<width && Q(k)<=high;++k) {
      if(!boost::multiprecision::bit_test(source,k)) continue;
      if(terms.size()==64) return {};
      rhs=k?add({position,constant(Q(-int(k)))}):position;
      terms.push_back(count_comparison(ShiftedBit::equal,args[1],rhs));
    }
    ShiftedBit out=decided(add(terms),true);
    if(terms.size()==1){out.kind=ShiftedBit::equal;out.count=args[1];out.rhs=rhs;}
    ++stats["shifted_one_hot_bits"];
    return out;
  }
  if(op=="add" && packed_bit_mode) {
    // Packed mode only. Bit p (a constant position) of a sum of non-negative integer terms 2^s x
    // whose possible one bits (the complement of known_bits' zero mask, moved by s) are pairwise
    // disjoint: such a sum has no carry, so it is the bitwise OR of its terms, and bit p is bit
    // p - s of the one term that can have it (0 when none can). O(terms) memoized known-bits lookups.
    if(args.empty() || args.size()!=data.size() || low!=high) return {};
    const unsigned p=floorq(low).convert_to<unsigned>();
    const Z all=pow2(256)-1;
    Z taken=0;
    std::optional<size_t> owner;
    unsigned owner_shift=0;
    for(size_t j=0;j<args.size();++j) {
      if(data[j].denominator()!=1 || data[j]<=0) return {};
      const Z c=data[j].numerator();
      if((c&(c-1))!=0) return {};
      const unsigned s=boost::multiprecision::msb(c);
      const auto bounds=nodes[args[j]].bounds;
      if(!nodes[args[j]].facts.finite || !bounds || bounds->first<0 || s>=256 ||
         bounds->second>=Q(pow2(256-s))) return {};
      const KnownBits k=known_bits(args[j]);
      if(!k.integer) return {};
      const Z support=Z((all^k.zero)<<s)&all; // the term is below 2^(256-s): no bit is lost
      if((support&taken)!=0) return {};
      taken|=support;
      if(boost::multiprecision::bit_test(support,p)){owner=j;owner_shift=s;}
    }
    ++stats["packed_disjoint_sum_bits"];
    if(!owner) return decided(zero,false);
    return visit_shifted_bit(args[*owner],constant(Q(p-owner_shift)),depth+1,memo,visits);
  }
  if(op=="add" && args.size()==2 && data.size()==2) {
    // word_shl(1, s, W) + d with d = -1 modulo 2^W: the low-bits mask of s bits (bits below W).
    const size_t term=nodes[args[0]].op=="word_shl"?0:1;
    const Id mask=args[term];
    const Q &d=data[1-term];
    if(nodes[mask].op!="word_shl" || args[1-term]!=one || data[term]!=1 || d.denominator()!=1)
      return {};
    const auto shl=nodes[mask];
    if(shl.args.size()!=2 || shl.data.size()!=1 || value(shl.args[0])!=Q(1) || !below(shl.data[0]))
      return {};
    const Z modulus=pow2(shl.data[0].numerator().convert_to<unsigned>());
    if(modz(d.numerator(),modulus)!=modulus-1) return {};
    ShiftedBit out=decided(count_comparison(ShiftedBit::above,shl.args[1],position),true);
    out.kind=ShiftedBit::above;out.count=shl.args[1];out.rhs=position;
    ++stats["shifted_low_mask_bits"];
    return out;
  }
  if((op=="word" || op=="signed") && args.size()==1 && data.size()==1) {
    if(!below(data[0])) return {};
    return visit_shifted_bit(args[0],position,depth+1,memo,visits);
  }
  // Bit p of extractbits(x, lo, n) is bit p + lo of x for p < n, and 0 for p >= n; of x / 2^k
  // (floor) bit p + k of x; of x mod 2^k bit p of x for p < k, and 0 for p >= k.
  if(op=="extractbits" && args.size()==1 && data.size()==2) {
    if(low>=data[1]) return decided(zero,false);
    if(!below(data[1]) || data[0].denominator()!=1 || data[0]<0) return {};
    return visit_shifted_bit(args[0],data[0]==0?position:add({position,constant(data[0])}),depth+1,memo,visits);
  }
  if((op=="idiv" || op=="imod") && args.size()==1 && data.size()==1 && data[0].denominator()==1 &&
     data[0]>0) {
    const Z d=data[0].numerator();
    if((d&(d-1))!=0) return {};
    const unsigned k=boost::multiprecision::msb(d);
    if(op=="idiv") return visit_shifted_bit(args[0],k?add({position,constant(Q(k))}):position,depth+1,memo,visits);
    if(low>=Q(k)) return decided(zero,false);
    if(!below(Q(k))) return {};
    return visit_shifted_bit(args[0],position,depth+1,memo,visits);
  }
  if(op=="word_shr" && args.size()==2 && data.size()==2 && data[1]==0) {
    const Id moved=add({position,args[1]});
    if(!valid_bit_position(moved) || !below(data[0]) || !(nodes[moved].bounds->second<data[0])) return {};
    return visit_shifted_bit(args[0],moved,depth+1,memo,visits);
  }
  if(op=="word_or" && args.size()==2 && data.size()==1) {
    if(!below(data[0])) return {};
    std::vector<Id> stack{v}, leaves;
    std::set<Id> seen;
    while(!stack.empty()) {
      const Id id=stack.back();stack.pop_back();check();
      if(!seen.insert(id).second) continue;
      if(seen.size()>shifted_bit_visit_limit) return {};
      const auto &n=nodes[id];
      if(n.op=="word_or" && n.args.size()==2 && n.data==data) {stack.push_back(n.args[0]);stack.push_back(n.args[1]);}
      else leaves.push_back(id);
    }
    std::sort(leaves.begin(),leaves.end());
    std::vector<Id> complements;
    bool shifted=false, predicate=false;
    for(Id leaf:leaves) {
      const auto bit=visit_shifted_bit(leaf,position,depth+1,memo,visits);
      if(!bit) return {};
      shifted=shifted || bit->shifted;
      predicate=predicate || bit->predicate;
      complements.push_back(sub(one,bit->value));
    }
    ++stats["shifted_or_bits"];
    auto out=decided(sub(one,mul(std::move(complements))),shifted);
    out.predicate=predicate;
    return out;
  }
  if(op=="select" && args.size()==3) {
    auto yes=visit_shifted_bit(args[1],position,depth+1,memo,visits);
    if(!yes) return {};
    auto no=visit_shifted_bit(args[2],position,depth+1,memo,visits);
    if(!no) return {};
    const bool shifted=yes->shifted || no->shifted;
    const bool predicate=yes->predicate || no->predicate;
    // An arm bit that is the condition itself (or its complement) is decided in its arm.
    if(const auto complement=boolean_select(args[0],false,true)) {
      for(auto *arm:{&*yes,&*no}) {
        const bool truth=arm==&*yes;
        if(arm->value==args[0]) *arm=decided(truth?one:zero,arm->shifted);
        else if(arm->value==*complement) *arm=decided(truth?zero:one,arm->shifted);
      }
    }
    if(auto kept=absorbed_shift_guard(args[0],*yes,*no)) {kept->shifted=shifted;kept->predicate=predicate;return kept;}
    const auto t=value(yes->value), f=value(no->value);
    if(t && f && (*t==0 || *t==1) && (*f==0 || *f==1))
      if(const auto both=boolean_select(args[0],*t==1,*f==1)) {
        // Two constant bits: the Select's Boolean condition, its complement, or a constant.
        auto out=decided(*both,shifted);
        out.predicate=predicate || !value(*both);
        return out;
      }
    auto out=decided(select(args[0],yes->value,no->value,true),shifted);
    out.predicate=predicate;
    return out;
  }
  return {};
}

// select(p, x, y) where y's bit is a constant K and x's bit is a leaf [s == r] or [s > r] (or
// symmetrically): p compares a value v with a constant, so it holds on an interval of v. The Select
// is replaced by one comparison when that is exact everywhere:
//   (a) where the leaf arm is taken, s = v + d (affine_on), and [v + d == r] (or >) is K wherever
//       the constant arm is taken: the Select is [v == r - d] (or [v > r - d]);
//   (b) the leaf itself is K wherever the constant arm is taken (interval_on of s there, bounds
//       of r): the Select is the leaf;
//   an empty constant-arm interval means that arm is never taken: the Select is the leaf.
// Both arms then agree wherever the constant arm is taken, and the result is the leaf arm's bit
// wherever it is taken. O(size of the count expression).
inline auto Arena::absorbed_shift_guard(Id p, const ShiftedBit &yes, const ShiftedBit &no) -> std::optional<ShiftedBit> {
  const auto &condition=nodes[p];
  if(condition.op!="compare" || condition.args.size()!=2 || !condition.data.empty()) return {};
  const Id v=value(condition.args[1])?condition.args[0]:value(condition.args[0])?condition.args[1]:invalid;
  if(v==invalid) return {};
  for(const bool leaf_is_yes:{true,false}) {
    const ShiftedBit &leaf=leaf_is_yes?yes:no, &other=leaf_is_yes?no:yes;
    const auto k=value(other.value);
    if(leaf.kind==ShiftedBit::other || !k || !nodes[leaf.rhs].bounds) continue;
    const auto r=*nodes[leaf.rhs].bounds;
    const auto taken=count_interval(p,v,leaf_is_yes), skipped=count_interval(p,v,!leaf_is_yes);
    if(!taken || !skipped) continue;
    if(skipped->first>skipped->second) {++stats["shifted_bit_guard_absorptions"];return leaf;}
    if(taken->first<=taken->second)
      if(const auto d=affine_on(leaf.count,GuardRegion{p,leaf_is_yes,v,*taken})) {
        if(decide_shifted_leaf(leaf.kind,*skipped,{r.first-*d,r.second-*d})==k) {
          ShiftedBit out=leaf;
          out.count=v;out.rhs=*d==0?leaf.rhs:add({leaf.rhs,constant(-*d)});
          out.value=count_comparison(leaf.kind,v,out.rhs);
          ++stats["shifted_bit_guard_absorptions"];++stats["shifted_bit_guard_rebased_counts"];
          return out;
        }
      }
    if(const auto counts=interval_on(leaf.count,GuardRegion{p,!leaf_is_yes,v,*skipped}))
      if(decide_shifted_leaf(leaf.kind,*counts,r)==k) {++stats["shifted_bit_guard_absorptions"];return leaf;}
  }
  return {};
}

// The truth of [s == r] (equal) or [s > r] (above) for every s in counts and r in rhs, if constant.
inline auto Arena::decide_shifted_leaf(ShiftedBit::Kind kind, const std::pair<Q,Q> &counts,
                                            const std::pair<Q,Q> &rhs) -> std::optional<Q> {
  if(kind==ShiftedBit::equal) {
    if(counts.second<rhs.first || counts.first>rhs.second) return Q(0);
    if(counts.first==counts.second && rhs.first==rhs.second && counts.first==rhs.first) return Q(1);
  } else if(kind==ShiftedBit::above) {
    if(counts.first>rhs.second) return Q(1);
    if(counts.second<=rhs.first) return Q(0);
  }
  return {};
}

// [count == rhs] or [count > rhs]. A count that is one Select of two constants under constant
// shells (selected_constants) is compared arm by arm against a constant rhs: a constant, the
// Boolean condition, or its complement. Exact. O(size of the count's shells).
inline auto Arena::count_comparison(ShiftedBit::Kind kind, Id count, Id rhs) -> Id {
  const std::string op=kind==ShiftedBit::equal?"eq":"gt";
  if(const auto r=value(rhs))
    if(const auto arms=selected_constants(count)) {
      const auto &[q,yes,no]=*arms;
      if(const auto decided=boolean_select(q,op=="eq"?yes==*r:yes>*r,op=="eq"?no==*r:no>*r)) {
        ++stats["shifted_bit_selected_counts"];
        return *decided;
      }
    }
  return compare(op,count,rhs);
}

// select(q, when_true, when_false) of two truth values for a Boolean (0/1 integer) q: a constant,
// q itself, or its complement (the complementary comparison of finite operands, else 1 - q).
inline auto Arena::boolean_select(Id q, bool when_true, bool when_false) -> std::optional<Id> {
  if(when_true==when_false) return when_true?one:zero;
  const auto &b=nodes[q].bounds;
  if(!b || b->first<0 || b->second>1 || !known_bits(q).integer) return {};
  if(when_true) return q;
  const auto c=nodes[q];
  if(c.op=="compare" && c.args.size()==2 && c.data.empty() && nodes[c.args[0]].facts.finite &&
     nodes[c.args[1]].facts.finite)
    return compare(complement_comparison(c.tag),c.args[0],c.args[1]);
  return sub(one,q);
}

// e as select(q, a, b) of two constants: e is one Select of two constants under shells of sums
// with constants, scalings, words, signed reinterpretations, max with constants and clamps, which
// apply to each arm. Exact
// (each shell is a function of the Select's value). O(size of e up to the depth limit).
inline auto Arena::selected_constants(Id e, unsigned depth) -> std::optional<std::tuple<Id,Q,Q>> {
  if(depth>=guard_region_depth_limit) return {};
  const std::string op=nodes[e].op;
  const std::vector<Id> args=nodes[e].args;
  const std::vector<Q> data=nodes[e].data;
  if(op=="select" && args.size()==3) {
    const auto yes=value(args[1]), no=value(args[2]);
    if(yes && no) return std::tuple(args[0],*yes,*no);
    return {};
  }
  const auto apply=[&](Id inner, auto f) -> std::optional<std::tuple<Id,Q,Q>> {
    const auto arms=selected_constants(inner,depth+1);
    if(!arms) return {};
    const auto &[q,yes,no]=*arms;
    const auto a=f(yes), b=f(no);
    if(!a || !b) return {};
    return std::tuple(q,*a,*b);
  };
  if(op=="add" && args.size()==data.size()) {
    Id inner=invalid;Q k(0),offset(0);
    for(size_t i=0;i<args.size();++i) {
      if(args[i]==one) {offset+=data[i];continue;}
      if(inner!=invalid) return {};
      inner=args[i];k=data[i];
    }
    if(inner==invalid) return {};
    return apply(inner,[&](const Q &x) -> std::optional<Q> { return k*x+offset; });
  }
  if(op=="mul" && args.size()==1 && data.size()==1)
    return apply(args[0],[&](const Q &x) -> std::optional<Q> { return data[0]*x; });
  if(op=="word" && args.size()==1 && data.size()==1 && data[0].denominator()==1)
    return apply(args[0],[&](const Q &x) -> std::optional<Q> {
      if(x.denominator()!=1) return {};
      return Q(modz(x.numerator(),pow2(data[0].numerator().convert_to<unsigned>())));
    });
  if(op=="clamp" && args.size()==1 && data.size()==2)
    return apply(args[0],[&](const Q &x) -> std::optional<Q> { return std::min(std::max(x,data[0]),data[1]); });
  if(op=="signed" && args.size()==1 && data.size()==1 && data[0].denominator()==1 && data[0]>0 && data[0]<=256)
    return apply(args[0],[&](const Q &x) -> std::optional<Q> {
      // The signed reinterpretation of the W-bit word x mod 2^W.
      if(x.denominator()!=1) return {};
      const unsigned w=data[0].numerator().convert_to<unsigned>();
      const Z modulus=pow2(w), v=modz(x.numerator(),modulus);
      return Q(v>=pow2(w-1)?Z(v-modulus):v);
    });
  if(op=="max") {
    Id inner=invalid;std::optional<Q> largest;
    for(Id x:args) {
      if(const auto c=value(x)) {largest=largest?std::max(*largest,*c):*c;continue;}
      if(inner!=invalid) return {};
      inner=x;
    }
    if(inner==invalid) return {};
    return apply(inner,[&](const Q &x) -> std::optional<Q> { return largest?std::max(x,*largest):x; });
  }
  return {};
}

// Whether the Select condition q is decided in the region: q is p itself, or q compares v with a
// constant and the region's interval lies within one side.
inline auto Arena::decided_condition(Id q, const GuardRegion &g) -> std::optional<bool> {
  if(q==g.p) return g.truth;
  const auto holds=count_interval(q,g.v,true), fails=count_interval(q,g.v,false);
  const auto within=[&](const std::optional<std::pair<Q,Q>> &side) {
    return side && side->first<=g.interval.first && g.interval.second<=side->second;
  };
  if(within(holds)) return true;
  if(within(fails)) return false;
  return {};
}

// An interval containing e for every v in the region (interval arithmetic over v, constants,
// sums, scalings, max, clamp, words that cannot wrap and Selects decided in the region; any
// other node contributes its own bounds). O(size of e up to guard_region_depth_limit levels).
inline auto Arena::interval_on(Id e, const GuardRegion &g, unsigned depth) -> std::optional<std::pair<Q,Q>> {
  if(e==g.v) return g.interval;
  if(auto c=value(e)) return std::pair(*c,*c);
  std::optional<std::pair<Q,Q>> own;
  if(nodes[e].bounds && nodes[e].facts.finite) own=*nodes[e].bounds;
  if(depth>=guard_region_depth_limit) return own;
  const std::string op=nodes[e].op;
  const std::vector<Id> args=nodes[e].args;
  const std::vector<Q> data=nodes[e].data;
  const auto child=[&](Id x) { return interval_on(x,g,depth+1); };
  if(op=="select" && args.size()==3) {
    if(const auto chosen=decided_condition(args[0],g)) return child(args[*chosen?1:2]);
    const auto a=child(args[1]), b=child(args[2]);
    if(!a || !b) return own;
    return std::pair(std::min(a->first,b->first),std::max(a->second,b->second));
  }
  if(op=="add" && args.size()==data.size()) {
    std::pair<Q,Q> sum(0,0);
    for(size_t i=0;i<args.size();++i) {
      const auto term=child(args[i]);
      if(!term) return own;
      const Q lo=term->first*data[i], hi=term->second*data[i];
      sum.first+=std::min(lo,hi);sum.second+=std::max(lo,hi);
    }
    return sum;
  }
  if(op=="mul" && args.size()==1 && data.size()==1) {
    const auto term=child(args[0]);
    if(!term) return own;
    const Q lo=term->first*data[0], hi=term->second*data[0];
    return std::pair(std::min(lo,hi),std::max(lo,hi));
  }
  if(op=="max" && !args.empty()) {
    std::optional<std::pair<Q,Q>> out;
    for(Id x:args) {
      const auto term=child(x);
      if(!term) return own;
      out=out?std::pair(std::max(out->first,term->first),std::max(out->second,term->second)):*term;
    }
    return out;
  }
  if(op=="clamp" && args.size()==1 && data.size()==2) {
    const auto term=child(args[0]);
    if(!term) return own;
    return std::pair(std::min(std::max(term->first,data[0]),data[1]),std::min(std::max(term->second,data[0]),data[1]));
  }
  if(op=="word" && args.size()==1 && data.size()==1 && data[0].denominator()==1) {
    // A word is its operand minus k 2^W when the whole interval lies in one window [k 2^W, (k+1) 2^W).
    const auto term=child(args[0]);
    const Z window=pow2(data[0].numerator().convert_to<unsigned>());
    if(term) {
      const Z k=floorq(term->first/Q(window));
      if(floorq(term->second/Q(window))==k) return std::pair(term->first-Q(k*window),term->second-Q(k*window));
    }
  }
  return own;
}

// d with e = v + d for every v in the region: v itself, a sum of one such term (coefficient 1)
// and a constant, a word that cannot wrap, a max whose other operands never exceed it, a clamp
// that never clips it, or a Select decided in the region. O(size of e up to the depth limit).
inline auto Arena::affine_on(Id e, const GuardRegion &g, unsigned depth) -> std::optional<Q> {
  if(e==g.v) return Q(0);
  if(depth>=guard_region_depth_limit) return {};
  const std::string op=nodes[e].op;
  const std::vector<Id> args=nodes[e].args;
  const std::vector<Q> data=nodes[e].data;
  const auto fits=[&](const Q &d, const Q &lo, const Q &hi) {
    return g.interval.first+d>=lo && g.interval.second+d<=hi;
  };
  if(op=="select" && args.size()==3) {
    if(const auto chosen=decided_condition(args[0],g)) return affine_on(args[*chosen?1:2],g,depth+1);
    return {};
  }
  if(op=="add" && args.size()==data.size()) {
    std::optional<Q> d;Q offset(0);
    for(size_t i=0;i<args.size();++i) {
      if(args[i]==one) {offset+=data[i];continue;}
      if(d || data[i]!=1) return {};
      d=affine_on(args[i],g,depth+1);
      if(!d) return {};
    }
    if(!d) return {};
    return *d+offset;
  }
  if(op=="word" && args.size()==1 && data.size()==1 && data[0].denominator()==1) {
    const auto d=affine_on(args[0],g,depth+1);
    if(d && fits(*d,Q(0),Q(pow2(data[0].numerator().convert_to<unsigned>())-1))) return d;
    return {};
  }
  if(op=="clamp" && args.size()==1 && data.size()==2) {
    const auto d=affine_on(args[0],g,depth+1);
    if(d && fits(*d,data[0],data[1])) return d;
    return {};
  }
  if(op=="max") {
    for(size_t i=0;i<args.size();++i) {
      const auto d=affine_on(args[i],g,depth+1);
      if(!d) continue;
      bool dominates=true;
      for(size_t j=0;j<args.size() && dominates;++j) {
        if(j==i) continue;
        const auto other=interval_on(args[j],g,depth+1);
        dominates=other && other->second<=g.interval.first+*d;
      }
      if(dominates) return d;
    }
  }
  return {};
}

// The integer interval of x (bounded, integer valued) where p = compare(op, x, c) or
// compare(op, c, x), c constant, holds (holds) or fails (!holds); empty when first > second.
// nullopt for any other p and for an inequality x != c with c strictly inside x's bounds. O(1).
inline auto Arena::count_interval(Id p, Id x, bool holds) -> std::optional<std::pair<Q,Q>> {
  const auto &condition=nodes[p];
  if(condition.op!="compare" || condition.args.size()!=2 || !condition.data.empty()) return {};
  std::string op=condition.tag;
  std::optional<Q> c;
  if(condition.args[0]==x) c=value(condition.args[1]);
  else if(condition.args[1]==x) {c=value(condition.args[0]);op=reverse_comparison(op);}
  const auto &b=nodes[x].bounds;
  if(!c || !b || !nodes[x].facts.finite || !known_bits(x).integer) return {};
  if(!holds) op=complement_comparison(op);
  Q lo(-floorq(-b->first)), hi(floorq(b->second));
  const Q floor_c(floorq(*c)), ceil_c(-floorq(-*c));
  if(op=="eq") {
    if(c->denominator()!=1) return std::pair(Q(1),Q(0));
    lo=std::max(lo,*c);hi=std::min(hi,*c);
  } else if(op=="ne") {
    // An interval only when c is outside the bounds or at one end of them.
    if(c->denominator()!=1 || *c<lo || *c>hi) return std::pair(lo,hi);
    if(*c==lo) lo+=1;
    else if(*c==hi) hi-=1;
    else return {};
  } else if(op=="lt") hi=std::min(hi,ceil_c-1);
  else if(op=="le") hi=std::min(hi,floor_c);
  else if(op=="gt") lo=std::max(lo,floor_c+1);
  else if(op=="ge") lo=std::max(lo,ceil_c);
  else return {};
  return std::pair(lo,hi);
}

} // namespace rg
