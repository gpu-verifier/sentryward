#pragma once
#include "core.hpp"

namespace rg {

inline auto Arena::known_bits(Id root) -> KnownBits {
  const Z mask=pow2(256)-1;
  auto exact=[&](Z v) {v=modz(v,pow2(256));return KnownBits{mask^v,v,true};};
  auto scale=[&](KnownBits k,const Q& c) {
    if(!k.integer || c.denominator()!=1)return KnownBits{};
    Z v=modz(c.numerator(),pow2(256));
    if(v==0)return exact(0);
    if((k.zero|k.one)==mask)return exact(k.one*v);
    if((v&(v-1))==0) {
      unsigned s=boost::multiprecision::lsb(v);
      return KnownBits{Z(((k.zero<<s)|(pow2(s)-1))&mask),Z((k.one<<s)&mask),true};
    }
    unsigned s=boost::multiprecision::lsb(v);
    return KnownBits{pow2(s)-1,0,true};
  };
  auto sum=[&](KnownBits l,KnownBits r) {
    if(!l.integer || !r.integer)return KnownBits{};
    KnownBits out{0,0,true};unsigned carries=1;
    for(unsigned b=0;b<256;++b) {
      Z bit=pow2(b);
      unsigned ls=(l.zero&bit)!=0?1:(l.one&bit)!=0?2:3;
      unsigned rs=(r.zero&bit)!=0?1:(r.one&bit)!=0?2:3;
      unsigned values=0,next=0;
      for(unsigned x=0;x<2;++x)for(unsigned y=0;y<2;++y)for(unsigned c=0;c<2;++c)
        if((ls&(1u<<x)) && (rs&(1u<<y)) && (carries&(1u<<c))) {
          unsigned v=x+y+c;values|=1u<<(v&1);next|=1u<<(v>>1);
        }
      if(values==1)out.zero|=bit;else if(values==2)out.one|=bit;
      carries=next;
    }
    return out;
  };
  // A Max fold with bounds is a predicate fold (boolean_fold_bounds): its value is its initial or
  // one instance of its body, so it has the bits both have (as a Select of the two).
  auto boolean_fold=[](const Node& n) {
    return n.op=="fold" && n.tag=="Max" && n.args.size()==2 && n.bounds.has_value();
  };
  auto descends=[&](const Node& n) {
    return n.op=="add" || n.op=="mul" || n.op=="select" || n.op=="word_or" ||
           n.op=="word" || n.op=="signed" || n.op=="extractbits" ||
           n.op=="constant_bit_mask" || n.op=="idiv" || n.op=="imod" || n.op=="word_shr" ||
           n.op=="word_shl" || n.op=="max" || n.op=="clamp" || boolean_fold(n);
  };
  std::vector<std::pair<Id,bool>> todo{{root,false}};
  while(!todo.empty()) {
    auto [id,ready]=todo.back();todo.pop_back();check();
    if(known_bits_cache.contains(id)){++stats["known_bits_cache_hits"];continue;}
    const auto& n=nodes[id];
    if(!ready && descends(n)) {
      todo.emplace_back(id,true);
      for(Id child:n.args)if(!known_bits_cache.contains(child))todo.emplace_back(child,false);
      continue;
    }
    auto child=[&](size_t j){return known_bits_cache.at(n.args.at(j));};
    KnownBits out;
    if(n.op=="const" && n.data[0].denominator()==1)out=exact(n.data[0].numerator());
    else if(n.op=="var" || n.op=="readbits" || n.op=="packedbits" || n.op=="floatbits" || n.op=="first_argmax" ||
            n.op=="compare" || n.op=="floor")out.integer=true;
    else if(n.op=="read") {
      auto split=n.tag.rfind('\n');std::string type=split==std::string::npos?"":n.tag.substr(split+1);
      out.integer=type.size()>1 && (type[0]=='u'||type[0]=='s'||type[0]=='b') &&
        std::all_of(type.begin()+1,type.end(),[](char c){return c>='0'&&c<='9';});
    }
    else if(n.op=="add") {
      out=exact(0);
      for(size_t j=0;j<n.args.size();++j)out=sum(out,scale(child(j),n.data[j]));
    } else if(n.op=="mul") {
      out=exact(1);
      for(size_t j=0;j<n.args.size();++j) {
        auto k=child(j);
        if(!k.integer || !out.integer){out={};break;}
        if((out.zero|out.one)==mask)out=scale(k,Q(out.one));
        else if((k.zero|k.one)==mask)out=scale(out,Q(k.one));
        else out={0,0,true};
      }
      out=scale(out,n.data[0]);
    } else if(n.op=="select") {
      auto l=child(1),r=child(2);out={Z(l.zero&r.zero),Z(l.one&r.one),l.integer&&r.integer};
    } else if(boolean_fold(n)) {
      auto l=child(0),r=child(1);out={Z(l.zero&r.zero),Z(l.one&r.one),l.integer&&r.integer};
      if(out.integer)++stats["boolean_fold_integer_bits"];
    } else if(n.op=="word_or") {
      auto l=child(0),r=child(1);out={Z(l.zero&r.zero),Z(l.one|r.one),true};
    } else if(n.op=="word" || n.op=="signed" || n.op=="extractbits" || n.op=="constant_bit_mask") {
      out=child(0);out.integer=true;
      if(n.op=="constant_bit_mask") {
        Z m=modz(n.data[0].numerator(),pow2(256));out.zero|=mask^m;out.one&=m;
      } else {
        unsigned lo=n.op=="extractbits"?n.data[0].numerator().convert_to<unsigned>():0;
        unsigned w=n.data[n.op=="extractbits"?1:0].numerator().convert_to<unsigned>();
        if(w>0 && w<=256 && lo<=256-w) {
          Z m=pow2(w)-1;out.zero=(out.zero>>lo)&m;out.one=(out.one>>lo)&m;
          if(n.op!="signed")out.zero|=mask^m;
          else if((out.zero&pow2(w-1))!=0)out.zero|=mask^m;
          else if((out.one&pow2(w-1))!=0)out.one|=mask^m;
        } else out={0,0,true};
      }
    } else if(n.op=="idiv" || n.op=="imod") {
      out.integer=n.op=="idiv" || child(0).integer;
      Z d=n.data[0].numerator();
      if(out.integer && child(0).integer && d>0 && (d&(d-1))==0) {
        unsigned s=boost::multiprecision::msb(d);
        if(s<256) {
          if(n.op=="imod") {Z m=pow2(s)-1;out={Z((child(0).zero&m)|(mask^m)),Z(child(0).one&m),true};}
          else out={Z(child(0).zero>>s),Z(child(0).one>>s),true};
        }
      }
    } else if(n.op=="word_shr" || n.op=="word_shl")out.integer=true;
    else if(n.op=="max" || n.op=="clamp") {
      out.integer=std::all_of(n.args.begin(),n.args.end(),[&](Id x){return known_bits_cache.at(x).integer;});
      for(const auto& d:n.data)out.integer &= d.denominator()==1;
    }
    // Intersect with the shared high prefix of a proven integer interval.
    if(out.integer && n.bounds) {
      Z lo=floorq(n.bounds->first),hi=floorq(n.bounds->second);
      if(lo<=hi && floorz(lo,pow2(256))==floorz(hi,pow2(256))) {
        lo=modz(lo,pow2(256));hi=modz(hi,pow2(256));Z differing=lo^hi;
        Z prefix=differing==0?mask:Z(mask^(pow2(boost::multiprecision::msb(differing)+1)-1));
        out.one|=lo&prefix;out.zero|=(mask^lo)&prefix;
      }
    }
    if(!out.integer)out={};
    known_bits_cache.emplace(id,std::move(out));++stats["known_bits_nodes"];
    stats["known_bits_edges"]+=descends(n)?n.args.size():0;
  }
  return known_bits_cache.at(root);
}

inline auto Arena::word_or(Id lhs,Id rhs,unsigned width) -> Id {
  if(width==0 || width>256)throw Unsupported("word OR width");
  Z mask=pow2(width)-1;
  // Typed integer inputs; substitution may enlarge their coordinate range.
  for(Id* operand:{&lhs,&rhs}) {
    auto b=nodes[*operand].bounds;
    if(!b || b->first<0 || b->second>Q(mask))*operand=imod(*operand,pow2(width));
  }
  if(lhs==rhs || rhs==zero)return lhs;
  if(lhs==zero)return rhs;
  auto l=known_bits(lhs),r=known_bits(rhs);
  Z ones=(l.one|r.one)&mask, zeros=(l.zero&r.zero)&mask;
  if((ones|zeros)==mask)return constant(Q(ones));
  if(((mask^l.zero)&(mask^r.zero)&mask)==0) {
    ++stats["disjoint_integer_or_fields"];return add({lhs,rhs});
  }
  if(rhs<lhs)std::swap(lhs,rhs);
  return intern("word_or",{lhs,rhs},{Q(width)},"",{true,false,true},std::pair(Q(ones),Q(mask^zeros)));
}

// PTX shr: the u32 shift count is clamped, never taken modulo width.
// The source is already interpreted in the declared signed/unsigned type.
inline auto Arena::word_shr(Id source, Id shift, unsigned width, bool sign) -> Id {
  if(width==0 || width>256) throw Unsupported("right shift width");
  auto sb=nodes[shift].bounds;
  if(!sb || sb->first<0) throw Unsupported("right shift requires unsigned count");
  // Compose only the immediate, same-width unsigned operation. The input
  // word and both counts must still have their typed integer semantics.
  // The synthesized count is an ideal integer sum: wrapping it to u32 would
  // turn e.g. UINT32_MAX + 1 into zero, contrary to two clamped PTX shifts.
  const auto &inner=nodes[source];
  if(!sign && inner.op=="word_shr" && inner.args.size()==2 && inner.data.size()==2 &&
     inner.data[0]==Q(width) && inner.data[1]==0) {
    const auto args=inner.args;
    const auto original_bounds=nodes[args[0]].bounds, first_bounds=nodes[args[1]].bounds;
    if(original_bounds && original_bounds->first>=0 && original_bounds->second<Q(pow2(width)) &&
       first_bounds && first_bounds->first>=0 && nodes[args[0]].facts.finite &&
       nodes[args[1]].facts.finite && nodes[shift].facts.finite &&
       known_bits(args[0]).integer && known_bits(args[1]).integer && known_bits(shift).integer) {
      source=args[0]; shift=add({args[1],shift}); sb=nodes[shift].bounds;
      ++stats["unsigned_nested_shift_compositions"];
    }
  }
  if(auto s=value(shift)) {
    if(s->denominator()!=1) throw Unsupported("noninteger right shift count");
    unsigned amount=*s>=Q(width)?width:s->numerator().convert_to<unsigned>();
    if(auto v=value(source);v && v->denominator()==1)
      return constant(Q(floorz(v->numerator(),pow2(amount))));
    if(!sign && nodes[source].op=="readbits")
      return amount>=width?zero:bit_slice(source,amount,width-amount);
    return idiv(source,pow2(amount));
  }
  if(source==zero)return zero;
  if(sb->first>=Q(width))return sign?idiv(source,pow2(width)):zero;
  Q type_lo=sign?Q(-pow2(width-1)):Q(0),type_hi=sign?Q(pow2(width-1)-1):Q(pow2(width)-1);
  Bounds result=std::pair(type_lo,type_hi);
  auto source_bounds=nodes[source].bounds;
  if(source_bounds && nodes[source].facts.finite && source_bounds->first>=type_lo && source_bounds->second<=type_hi) {
    Z lo=-floorq(-source_bounds->first),hi=floorq(source_bounds->second);
    Z first=-floorq(-sb->first),last=floorq(sb->second);
    if(lo<=hi && first<=last) {
      unsigned s0=first>=Z(width)?width:first.convert_to<unsigned>();
      unsigned s1=last>=Z(width)?width:last.convert_to<unsigned>();
      Z a=floorz(lo,pow2(s0)),b=floorz(lo,pow2(s1));
      Z c=floorz(hi,pow2(s0)),d=floorz(hi,pow2(s1));
      result=std::pair(Q(std::min(a,b)),Q(std::max(c,d)));
      ++stats["word_shift_range_transfers"];
    }
  }
  return intern("word_shr",{source,shift},{Q(width),Q(sign)},"",
                {nodes[source].facts.finite,result->first>0,result->first>=0},result);
}

// PTX shl.bW: d = a * 2^s mod 2^W for the u32 count s < W, and 0 for s >= W (the count is clamped
// at the word width, never taken modulo it). source is a W-bit word in [0, 2^W), shift a nonnegative
// integer. A constant count is the exact product (reduced modulo 2^W only when its bounds may wrap);
// a symbolic count is the word_shl node, whose bits shifted_word_bit reads. Its bounds: the largest
// nonzero result is source * 2^min(s, W-1) when that cannot wrap, else 2^W - 1. O(1).
inline auto Arena::word_shl(Id source, Id shift, unsigned width) -> Id {
  if(width==0 || width>256) throw Unsupported("left shift width");
  const Z modulus=pow2(width);
  const auto source_bounds=nodes[source].bounds, shift_bounds=nodes[shift].bounds;
  if(!nodes[source].facts.finite || !source_bounds || source_bounds->first<0 ||
     source_bounds->second>=Q(modulus) || !known_bits(source).integer)
    throw Unsupported("left shift source must be a word");
  if(!nodes[shift].facts.finite || !shift_bounds || shift_bounds->first<0 || !known_bits(shift).integer)
    throw Unsupported("left shift requires unsigned count");
  if(auto s=value(shift)) {
    if(*s>=Q(width)) {++stats["clamped_left_shifts"];return zero;}
    const Id product=mul({source,constant(Q(pow2(s->numerator().convert_to<unsigned>())))});
    const auto b=nodes[product].bounds;
    return b && b->first>=0 && b->second<Q(modulus)?product:imod(product,modulus);
  }
  if(source==zero || shift_bounds->first>=Q(width)) return zero;
  const unsigned top=shift_bounds->second>=Q(width-1)?width-1:floorq(shift_bounds->second).convert_to<unsigned>();
  const Q largest=source_bounds->second*Q(pow2(top));
  ++word_shl_requests;++stats["symbolic_left_shifts"];
  return intern("word_shl",{source,shift},{Q(width)},"",{true,false,true},
                std::pair(Q(0),largest<Q(modulus)?largest:Q(modulus-1)));
}

// Shared signed reinterpretation for import and all reconstruction paths.
// Operands are integer values; modulo width is retained unless range proves
// it irrelevant. No recursive polynomial/modulo expansion is necessary.
inline auto Arena::signed_word(Id source,unsigned width) -> Id {
  if(width==0 || width>256)throw Unsupported("signed word width");
  Z modulus=pow2(width),half=pow2(width-1);
  if(auto v=value(source);v && v->denominator()==1) {
    Z bits=modz(v->numerator(),modulus);
    return constant(Q(bits<half?bits:Z(bits-modulus)));
  }
  auto b=nodes[source].bounds;
  if(nodes[source].facts.finite && b && b->first>=Q(-half) && b->second<Q(half))return source;
  // signed(word(x, w), w) = x for an integer x inside [-2^(w-1), 2^(w-1)) (signed_arm), and
  // signed(select(c, A, B), w) = select(c, signed(A, w), signed(B, w)): the reinterpretation
  // applies to whichever arm is taken; used when both arms resolve without a signed shell.
  // Inspects one shell. Exact.
  if(auto direct=signed_arm(source,width)) {++stats["signed_word_identities"];return *direct;}
  if(nodes[source].op=="select" && nodes[source].args.size()==3) {
    const auto arms=nodes[source].args;
    const auto yes=signed_arm(arms[1],width), no=signed_arm(arms[2],width);
    if(yes && no) {++stats["signed_word_select_arms"];return select(arms[0],*yes,*no,true);}
  }
  Id bits=source;
  if(!b || b->first<0 || b->second>=Q(modulus)) {
    bits=intern("word",{source},{Q(width)},"",{nodes[source].facts.finite,false,true},
                std::pair(Q(0),Q(modulus-1)));
    b=nodes[bits].bounds;
  }
  if(nodes[bits].facts.finite && b && b->second<Q(half))return bits;
  if(nodes[bits].facts.finite && b && b->first>=Q(half))return sub(bits,constant(Q(modulus)));
  return intern("signed",{bits},{Q(width)},"",{nodes[bits].facts.finite,false,false},
                std::pair(Q(-half),Q(half-1)));
}

// The signed value at width w of x where no signed shell is needed: a constant's, or y for x = y
// or x = word(y, w) with y an integer inside [-2^(w-1), 2^(w-1)) (the word is y modulo 2^w, and the
// signed reinterpretation is the representative of that class in the signed range). O(1).
inline auto Arena::signed_arm(Id x,unsigned width) -> std::optional<Id> {
  const Z half=pow2(width-1), modulus=pow2(width);
  if(auto v=value(x)) {
    if(v->denominator()!=1) return {};
    const Z bits=modz(v->numerator(),modulus);
    return constant(Q(bits<half?bits:Z(bits-modulus)));
  }
  Id inner=x;
  {
    const auto &n=nodes[x];
    if(n.op=="word" && n.args.size()==1 && n.data.size()==1 && n.data[0]==Q(width)) inner=n.args[0];
  }
  const auto b=nodes[inner].bounds;
  if(nodes[inner].facts.finite && b && b->first>=Q(-half) && b->second<Q(half) && known_bits(inner).integer)
    return inner;
  return {};
}

} // namespace rg
