#pragma once
#include "core.hpp"

namespace rg {

// readbits is a little-endian byte view (bit_slice above reads bits [8k, 8k+8) of R_W(a) as
// R_8(a+k)). So a sum of adjacent views of one allocation, sum_i 2^(8 o_i) R_(w_i)(a + o_i) with
// o_0 = 0 and o_(i+1) = o_i + w_i/8, is the single view R_W(a), W = sum_i w_i <= 256: both are
// sum_k 2^(8k) byte(a+k). Anything else is not a view. O(terms).
inline auto Arena::concatenated_byte_view(Id source) -> std::optional<Id> {
  if(nodes[source].op=="readbits")return source;
  const auto args=nodes[source].args;const auto data=nodes[source].data;
  if(nodes[source].op!="add" || args.size()<2 || args.size()!=data.size())return {};
  std::vector<std::pair<unsigned,Id>> parts;
  for(size_t i=0;i<args.size();++i) {
    const auto &view=nodes[args[i]];
    if(data[i].denominator()!=1 || data[i]<=0 || view.op!="readbits" || view.args.size()!=1 ||
       view.data.size()!=1)return {};
    const Z coefficient=data[i].numerator();
    if((coefficient&(coefficient-1))!=0)return {};
    const unsigned shift=boost::multiprecision::msb(coefficient);
    if(shift%8)return {};
    parts.emplace_back(shift,args[i]);
  }
  std::sort(parts.begin(),parts.end());
  const std::string tag=nodes[parts.front().second].tag;
  const Id base=nodes[parts.front().second].args[0];
  unsigned next=0;
  for(const auto &[shift,view]:parts) {
    const Q bits=nodes[view].data[0];
    if(shift!=next || nodes[view].tag!=tag || bits.denominator()!=1 || bits<=0 ||
       modz(bits.numerator(),8)!=0 || bits>Q(256-next))return {};
    if(shift && nodes[view].args[0]!=add({base,constant(Q(shift/8))}))return {};
    next+=bits.numerator().convert_to<unsigned>();
  }
  ++stats["concatenated_byte_views"];
  return intern("readbits",{base},{Q(next)},tag,{true,false,true},std::pair(Q(0),Q(pow2(next)-1)));
}

// Bits [lo, lo+count) of word_shr(V, s), V a byte view R_W(a) (or a concatenation of adjacent
// views) no wider than the shifted word, are R_count(a + s/8 + lo/8) when lo and count are byte
// multiples, s is an integer multiple of 8 and 0 <= s, s + lo + count <= W: PTX shr then shifts
// exactly (its clamp at the word width is not reached) and the slice selects whole bytes of V.
// A constant s is already reduced by word_shr. O(terms of V).
inline auto Arena::shifted_byte_view_slice(Id source,unsigned lo,unsigned count) -> std::optional<Id> {
  const auto args=nodes[source].args;const auto data=nodes[source].data;
  if(nodes[source].op!="word_shr" || args.size()!=2 || data.size()!=2 || data[1]!=0 ||
     lo%8 || count%8 || !count || value(args[1]))return {};
  const auto view=concatenated_byte_view(args[0]);
  if(!view || nodes[*view].data[0]>data[0])return {};
  const Id shift=args[1];
  const auto shift_bounds=nodes[shift].bounds;
  if(!shift_bounds || shift_bounds->first<0 || !nodes[shift].facts.finite ||
     shift_bounds->second+Q(lo)+Q(count)>nodes[*view].data[0])return {};
  const auto shift_bits=known_bits(shift);
  if(!shift_bits.integer || (shift_bits.zero&7)!=7)return {};
  const std::string tag=nodes[*view].tag;
  const Id address=add({nodes[*view].args[0],idiv(shift,8),constant(Q(lo/8))});
  ++stats["shifted_byte_view_slices"];
  return intern("readbits",{address},{Q(count)},tag,{true,false,true},std::pair(Q(0),Q(pow2(count)-1)));
}

// gcd of d and every integer coefficient of ts; 1 when a coefficient is not
// an integer.
inline auto Arena::common_factor(const std::vector<std::pair<Id, Q>> &ts, Z d) -> Z {
  for (const auto &[_, c] : ts) {
    if (c.denominator() != 1)
      return 1;
    d = gcdz(d, c.numerator());
  }
  return d;
}

inline auto Arena::idiv(Id n, Z d) -> Id {
  if (d <= 0)
    throw Unsupported("index divisor must be positive");
  if (d == 1)
    return n;
  if((nodes[n].op=="word_or" || (nodes[n].op=="word_shr" && nodes[n].data[1]==0)) && (d&(d-1))==0) {
    unsigned lo=boost::multiprecision::msb(d);
    unsigned width=nodes[n].data[0].numerator().convert_to<unsigned>();
    return lo>=width?zero:bit_slice(n,lo,width-lo);
  }
  if(nodes[n].op=="idiv") return idiv(nodes[n].args[0],d*nodes[n].data[0].numerator());
  // x = q*(d*m) + r with 0 <= r < d*m gives floor(r/d) = floor(x/d) mod m:
  // the quotient of a residue is the residue of the quotient.
  if(nodes[n].op=="imod" && nodes[n].args.size()==1 && nodes[n].data[0].denominator()==1 &&
     modz(nodes[n].data[0].numerator(),d)==0) {
    const Id x=nodes[n].args[0];const Z m=nodes[n].data[0].numerator()/d;
    return imod(idiv(x,d),m);
  }
  if(nodes[n].op=="readbits" && (d & (d-1))==0) {
    unsigned shift=boost::multiprecision::msb(d);
    unsigned bits=nodes[n].data[0].numerator().convert_to<unsigned>();
    if(shift>=bits) return zero;
    if(shift%8==0 && bits%8==0) return bit_slice(n,shift,bits-shift);
  }
  auto ts = terms(n);
  Z g = common_factor(ts, d);
  if (g > 1) {
    std::vector<Id> rs;
    for (auto &[t, c] : ts)
      rs.push_back(mul({constant(c / Q(g)), t}));
    return idiv(add(rs), d / g);
  }
  // Rule (quotient of an offset multiple): floor((a*x + c0) / (a*m)) = floor(x / m) for one integer
  // term x with integer coefficient a dividing d, and an integer offset 0 < c0 < a: x + c0/a lies in
  // (x, x + 1). (A record field's byte address, record * size + field, divided by a block of whole
  // records.) Exact. O(1).
  if (ts.size() == 2) {
    Z a_common = 0; Q offset(0); bool eligible = true;
    for (auto &[t, c] : ts) {
      if (c.denominator() != 1) { eligible = false; break; }
      if (t == one) { offset += c; continue; }
      if (a_common != 0 || !nodes[t].facts.finite || !known_bits(t).integer) { eligible = false; break; }
      a_common = boost::multiprecision::abs(c.numerator());
    }
    if (eligible && a_common > 1 && modz(d, a_common) == 0 && offset > 0 && offset < Q(a_common)) {
      std::vector<Id> ys;
      for (auto &[t, c] : ts) if (t != one) ys.push_back(mul({constant(c / Q(a_common)), t}));
      ++stats["offset_multiple_quotients"];
      return idiv(add(std::move(ys)), d / a_common);
    }
  }
  std::vector<Id> qs, rs;
  for (auto &[t, c] : ts) {
    if (c.denominator() != 1)
      return intern("idiv", {n}, {Q(d)}, "", {true, false, false});
    Z q = floorz(c.numerator(), d), r = modz(c.numerator(), d);
    if (q != 0)
      qs.push_back(mul({constant(Q(q)), t}));
    if (r != 0)
      rs.push_back(mul({constant(Q(r)), t}));
  }
  Id r = add(rs);
  auto b = nodes[r].bounds;
  if (b && floorq(b->first / Q(d)) == floorq(b->second / Q(d))) {
    qs.push_back(constant(Q(floorq(b->first / Q(d)))));
    return add(qs);
  }
  // Packed decoding often contains (d-small)*bit. A centered remainder
  // exposes its carry without enumerating bit values. Try only short affine
  // sums and prove the remaining quotient is constant from exact bounds.
  if(ts.size()<=8) {
    Q low(0), high(0);bool bounded=true,changed=false;
    for(const auto &[term,coefficient]:ts) {
      Z rem=modz(coefficient.numerator(),d);
      if(rem>d/2) {rem-=d;changed=true;}
      if(rem==0)continue;
      auto range=nodes[term].bounds;
      if(!range || !nodes[term].facts.finite) {bounded=false;break;}
      Q l=Q(rem)*range->first,h=Q(rem)*range->second;
      low+=std::min(l,h);high+=std::max(l,h);
    }
    if(changed && bounded && floorq(low/Q(d))==floorq(high/Q(d))) {
      std::vector<Id> centered{constant(Q(floorq(low/Q(d))))};
      for(const auto &[term,coefficient]:ts) {
        Z rem=modz(coefficient.numerator(),d);if(rem>d/2)rem-=d;
        Z quotient=(coefficient.numerator()-rem)/d;
        if(quotient!=0)centered.push_back(mul({constant(Q(quotient)),term}));
      }
      ++stats["bounded_centered_quotients"];
      return add(centered);
    }
  }
  Bounds bound;
  if (b)
    bound =
        std::pair(Q(floorq(b->first / Q(d))), Q(floorq(b->second / Q(d))));
  qs.push_back(intern("idiv", {r}, {Q(d)}, "", {true, false, false}, bound));
  return add(qs);
}

inline auto Arena::imod(Id n, Z d) -> Id {
  if (d <= 0)
    throw Unsupported("index modulus must be positive");
  if((nodes[n].op=="word_shr" || nodes[n].op=="word_or") && (d&(d-1))==0)
    return bit_slice(n,0,boost::multiprecision::msb(d));
  if(nodes[n].op=="imod" && modz(nodes[n].data[0].numerator(),d)==0)
    return imod(nodes[n].args[0],d);
  if(nodes[n].op=="readbits" && (d & (d-1))==0) {
    unsigned count=boost::multiprecision::msb(d);
    unsigned bits=nodes[n].data[0].numerator().convert_to<unsigned>();
    if(count>=bits) return n;
    if(count>0 && count%8==0 && bits%8==0) return bit_slice(n,0,count);
  }
  if (nodes[n].op == "word" && modz(pow2(nodes[n].data[0].numerator().convert_to<unsigned>()), d) == 0)
    return imod(nodes[n].args[0], d);
  auto ts = terms(n);
  // A term c imod(a, d') with integer c and d' a multiple of d is congruent to c a modulo d
  // (imod(a, d') = a - d' q for an integer q), so imod(... + c imod(a, d') + ..., d) is
  // imod(... + c a + ..., d). Exact for integer operands. O(terms).
  if (std::any_of(ts.begin(), ts.end(), [&](const auto &term) {
        const auto &m = nodes[term.first];
        return term.second.denominator() == 1 && m.op == "imod" && m.args.size() == 1 && m.data.size() == 1 &&
               m.data[0].denominator() == 1 && modz(m.data[0].numerator(), d) == 0; })) {
    std::vector<Id> rs;
    for (const auto &[t, c] : ts) {
      Id operand = t;
      {
        const auto &m = nodes[t];
        if (c.denominator() == 1 && m.op == "imod" && m.args.size() == 1 && m.data.size() == 1 &&
            m.data[0].denominator() == 1 && modz(m.data[0].numerator(), d) == 0)
          operand = m.args[0];
      }
      const Id k = constant(c);
      rs.push_back(mul({k, operand}));
    }
    ++stats["nested_remainder_terms"];
    return imod(add(rs), d);
  }
  Z g = common_factor(ts, d);
  if (g > 1) {
    std::vector<Id> rs;
    for (auto &[t, c] : ts)
      rs.push_back(mul({constant(c / Q(g)), t}));
    if (d / g == 1)
      return zero;
    return mul({constant(Q(g)), imod(add(rs), d / g)});
  }
  Id q = idiv(n, d);
  if (nodes[q].op != "idiv" && nodes[q].op != "add")
    return sub(n, mul({constant(Q(d)), q}));
  std::vector<Id> rs;
  for (auto &[t, c] : ts) {
    if (c.denominator() != 1)
      return intern("imod", {n}, {Q(d)}, "", {true, false, true},
                    std::pair(Q(0), Q(d - 1)));
    Z rem = modz(c.numerator(), d);
    if (rem != 0)
      rs.push_back(mul({constant(Q(rem)), t}));
  }
  Id r = add(rs);
  auto b = nodes[r].bounds;
  if (b && floorq(b->first / Q(d)) == floorq(b->second / Q(d)))
    return sub(r, constant(Q(floorq(b->first / Q(d)) * d)));
  // Coefficient reduction can expose a word beneath an aligned constant.
  // Only peel it when truncation preserves this modulus; never wrap addresses.
  if (r != n && nodes[r].op == "word" &&
      modz(pow2(nodes[r].data[0].numerator().convert_to<unsigned>()), d) == 0)
    return imod(nodes[r].args[0], d);
  return intern("imod", {r}, {Q(d)}, "", {true, false, true},
                std::pair(Q(0), Q(d - 1)));
}

inline auto Arena::read(std::string alloc, std::vector<Id> coords, std::string type,
        Bounds b) -> Id {
  bool finite = type.starts_with("real") || bool(b);
  return intern("read", coords, {}, alloc + "\n" + type,
                {finite, b && b->first > 0, b && b->first >= 0}, b);
}

} // namespace rg
