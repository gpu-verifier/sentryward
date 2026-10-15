#pragma once
#include "core.hpp"
namespace rg {
// A local, exact finite-domain proof. The pass never expands an array, follows
// an address producer, or assumes a numeric range for an unknown storage read.
// Only one contiguous four-bit slice may influence a candidate decoder.
struct PackedDecodePass {
  struct Bits {
    std::string allocation;
    Id address = invalid; // canonical bit address, not a byte address
    unsigned count = 0;
  };
  using Resolve = std::function<std::optional<Bits>(Id, unsigned, unsigned)>;
  struct Slice {
    Id source = invalid;
    unsigned lo = 0, count = 0;
    std::optional<Bits> canonical;
  };
  struct Instruction {
    Id source;
    std::vector<unsigned> args;
    std::optional<Bits> bits;
    unsigned bit_shift = 0;
  };
  enum Decoder { Unknown, Signed4, FP4 };
  Arena &a;
  // Unknown enables both decoder kinds for bounded component experiments.
  // Keep the target immutable because normalized results are cached by DAG ID.
  const Decoder target;
  std::unordered_map<Id, std::optional<Slice>> slices;
  std::unordered_map<Id, Id> normalized;
  std::unordered_set<Id> known_integers;
  std::map<std::tuple<Id,unsigned,unsigned>,std::optional<Bits>> resolved;
  std::unordered_map<std::string, Decoder> patterns;
  static constexpr unsigned max_nodes = 32, max_depth = 24;
  explicit PackedDecodePass(Arena &arena, Decoder kind = Unknown) : a(arena), target(kind) {}

  static std::optional<unsigned> power_of_two(const Q &q) {
    if(q.denominator()!=1 || q<=0 || q>Q(pow2(256)))return {};
    Z z=q.numerator();
    if((z & (z-1))!=0)return {};
    return boost::multiprecision::msb(z);
  }
  std::optional<Bits> resolve(const Slice &s,const Resolve &mapper) {
    if(s.canonical)return s.canonical;
    auto key=std::tuple(s.source,s.lo,s.count);
    if(auto it=resolved.find(key);it!=resolved.end())return it->second;
    auto value=mapper(s.source,s.lo,s.count);
    resolved.emplace(key,value);
    return value;
  }
  bool integer_impl(Id id,unsigned &budget,unsigned depth=0) const {
    if(depth>8 || budget==0)return false;
    --budget;
    if(known_integers.contains(id))return true;
    const auto &n=a.nodes[id];
    if(n.op=="const")return n.data[0].denominator()==1;
    // floatbits is the integer storage encoding, not the floating value.
    // Its integral nature must survive exact pack/unpack arithmetic.
    if(n.op=="readbits" || n.op=="extractbits" || n.op=="packedbits" || n.op=="floatbits" || n.op=="idiv" || n.op=="word" || n.op=="signed" || n.op=="word_shr" || n.op=="word_or")return true;
    if(n.op=="imod")return integer_impl(n.args[0],budget,depth+1) && n.data[0].denominator()==1;
    if((n.op=="add" || n.op=="mul") && n.args.size()<=4) {
      for(const auto &q:n.data)if(q.denominator()!=1)return false;
      for(Id arg:n.args)if(!integer_impl(arg,budget,depth+1))return false;
      return true;
    }
    return false;
  }
  bool integer(Id id) const {
    unsigned budget=max_nodes;
    return integer_impl(id,budget);
  }
  // A demanded packed slice can discard a proved low radix digit:
  // floor((s*y+r)/(s*d)) = floor(y/d), for integer y and 0 <= r < s.
  // Keep this local to packed transport; do not add another search to general
  // arithmetic construction. At most eight addends and eight power-of-two
  // radices are considered, sharing the transport visit budget.
  std::optional<Id> mixed_radix_quotient(Id dividend, const Z &divisor,
                                        unsigned &budget) {
    const auto &n = a.nodes[dividend];
    if (n.op != "add" || n.args.size() > 8 || n.args.size() != n.data.size() ||
        !n.facts.finite || !power_of_two(Q(divisor))) return {};
    std::set<Z> radices;
    for (const auto &coefficient : n.data) {
      if (coefficient.denominator() != 1 || !small(coefficient)) return {};
      Z radix = boost::multiprecision::gcd(
          divisor, Z(boost::multiprecision::abs(coefficient.numerator())));
      if (radix > 1) radices.insert(std::move(radix));
    }
    for (const Z &radix : radices) {
      if (budget == 0) return {};
      --budget;
      ++a.stats["packed_radix_probes"];
      Q low(0), high(0);
      bool valid = true, has_quotient = false;
      for (size_t j = 0; j < n.args.size(); ++j) {
        if (budget == 0) return {};
        --budget;
        Z quotient = floorz(n.data[j].numerator(), radix);
        Z remainder = modz(n.data[j].numerator(), radix);
        if (quotient != 0) {
          has_quotient = true;
          if (!integer_impl(n.args[j], budget)) { valid = false; break; }
        }
        if (remainder != 0) {
          const auto &term = a.nodes[n.args[j]];
          if (!term.facts.finite || !term.bounds) { valid = false; break; }
          low += Q(remainder) * term.bounds->first;
          high += Q(remainder) * term.bounds->second;
        }
      }
      if (!valid || !has_quotient || low < 0 || high >= Q(radix)) continue;
      std::vector<Id> high_terms;
      for (size_t j = 0; j < n.args.size(); ++j) {
        Z quotient = floorz(n.data[j].numerator(), radix);
        if (quotient != 0)
          high_terms.push_back(a.mul({a.constant(Q(quotient)), n.args[j]}));
      }
      ++a.stats["packed_radix_rewrites"];
      return a.idiv(a.add(std::move(high_terms)), divisor / radix);
    }
    return {};
  }
  // Preserve the dependence between an index's parity and its chosen nibble.
  // This is a two-point Boolean identity, not an assumption about the index.
  Id select_address(Id predicate,Id yes,Id no) {
    const auto &p=a.nodes[predicate];
    if(p.op=="compare" && (p.tag=="eq" || p.tag=="ne")) {
      for(unsigned i=0;i<2;++i) {
        auto constant=a.value(p.args[i]);
        Id bit=p.args[1-i];const auto &b=a.nodes[bit].bounds;
        if(!constant || (*constant!=0 && *constant!=1) || !b || b->first<0 || b->second>1)continue;
        // Only proved integer values are accepted, not a real interval [0,1].
        if(!integer(bit))continue;
        bool at_zero=(*constant==0)==(p.tag=="eq");
        auto delta=a.value(a.sub(yes,no));
        if(!delta)continue;
        return at_zero?a.add({yes,a.mul({a.constant(-*delta),bit})})
                      :a.add({no,a.mul({a.constant(*delta),bit})});
      }
    }
    return a.select(predicate,yes,no);
  }
  std::optional<Slice> transport_impl(Id id,const Resolve &mapper,unsigned &budget,unsigned depth=0) {
    if(depth>max_depth || budget==0)return {};
    --budget;
    ++a.stats["packed_transport_visits"];
    if(auto it=slices.find(id);it!=slices.end())return it->second;
    const auto &n=a.nodes[id];std::optional<Slice> result;
    if(n.op=="readbits" && n.args.size()==1 && n.data.size()==1 &&
       n.data[0].denominator()==1 && n.data[0]>0 && n.data[0]<=256) {
      result=Slice{id,0,n.data[0].numerator().convert_to<unsigned>(),{}};
    } else if(n.op=="packedbits" && n.args.size()==1 && n.data.size()==1 && n.data[0]==4) {
      result=Slice{invalid,0,4,Bits{n.tag,n.args[0],4}};
    } else if(n.args.size()==1 && (n.op=="extractbits" || n.op=="idiv" || n.op=="imod" || n.op=="word")) {
      unsigned lo=0,count=256;
      bool valid=true;
      if(n.op=="extractbits") {
        valid=n.data.size()==2 && n.data[0].denominator()==1 && n.data[1].denominator()==1 &&
              n.data[0]>=0 && n.data[0]<=256 && n.data[1]>0 && n.data[1]<=256;
        if(valid) {lo=n.data[0].numerator().convert_to<unsigned>();count=n.data[1].numerator().convert_to<unsigned>();}
      } else if(n.op=="word") {
        valid=n.data.size()==1 && n.data[0].denominator()==1 && n.data[0]>0 && n.data[0]<=256;
        if(valid)count=n.data[0].numerator().convert_to<unsigned>();
      } else {
        auto shift=n.data.size()==1?power_of_two(n.data[0]):std::optional<unsigned>{};
        valid=shift.has_value();
        if(valid) {if(n.op=="idiv")lo=*shift;else count=*shift;}
      }
      if(valid)if(auto p=transport_impl(n.args[0],mapper,budget,depth+1);p && lo<p->count) {
        if(n.op!="extractbits" || lo+count<=p->count) {
          count=std::min(count,p->count-lo);
          if(count) {
            if(p->canonical) {
              auto b=*p->canonical;b.address=a.add({b.address,a.constant(lo)});b.count=count;
              result=Slice{invalid,0,count,b};
            } else result=Slice{p->source,p->lo+lo,count,{}};
          }
        }
      }
      if (!result && valid && n.op == "idiv")
        if (auto reduced = mixed_radix_quotient(n.args[0], n.data[0].numerator(), budget))
          result = transport_impl(*reduced, mapper, budget, depth+1);
    } else if(n.op=="select" && n.args.size()==3) {
      auto yes=transport_impl(n.args[1],mapper,budget,depth+1),no=transport_impl(n.args[2],mapper,budget,depth+1);
      if(yes && no && yes->count==no->count && yes->count<=4) {
        auto y=resolve(*yes,mapper),z=resolve(*no,mapper);
        if(y && z && y->allocation==z->allocation)
          result=Slice{invalid,0,y->count,Bits{y->allocation,select_address(n.args[0],y->address,z->address),y->count}};
      }
    }
    slices.emplace(id,result);
    return result;
  }
  std::optional<Slice> transport(Id id,const Resolve &mapper) {
    unsigned budget=max_nodes;
    return transport_impl(id,mapper,budget);
  }
  static bool small(const Q &q) {
    Z numerator=q.numerator();if(numerator<0)numerator=-numerator;
    return (numerator==0 || boost::multiprecision::msb(numerator)<512) &&
           boost::multiprecision::msb(q.denominator())<512;
  }
  static bool allowed(const Node &n) {
    if(n.args.size()>8)return false;
    for(const auto &q:n.data)if(!small(q))return false;
    if(n.op=="const" || n.op=="add" || n.op=="mul")return true;
    if((n.op=="idiv" || n.op=="imod" || n.op=="word" || n.op=="signed" || n.op=="exp2") && n.args.size()==1)return true;
    if((n.op=="xor" || n.op=="compare" || n.op=="word_or") && n.args.size()==2)return true;
    return n.op=="select" && n.args.size()==3;
  }
  std::optional<unsigned> program(Id id,const Resolve &mapper,std::vector<Instruction> &out,
                                  std::unordered_map<Id,unsigned> &memo,unsigned depth=0) {
    if(depth>max_depth)return {};
    if(auto it=memo.find(id);it!=memo.end())return it->second;
    if(out.size()>=max_nodes)return {};
    Instruction step{id,{},{}};
    if(auto s=transport(id,mapper);s && s->count<=4)step.bits=resolve(*s,mapper);
    if(!step.bits) {
      const auto &n=a.nodes[id];
      if(!allowed(n))return {};
      for(Id arg:n.args) {
        auto slot=program(arg,mapper,out,memo,depth+1);
        if(!slot)return {};
        step.args.push_back(*slot);
      }
    }
    if(out.size()>=max_nodes)return {};
    unsigned slot=unsigned(out.size());out.push_back(std::move(step));memo.emplace(id,slot);return slot;
  }
  std::optional<Q> evaluate(const Instruction &step,const std::vector<Q> &values,unsigned input) {
    if(step.bits)return Q((input>>step.bit_shift)&((1u<<step.bits->count)-1));
    const auto &n=a.nodes[step.source];
    auto arg=[&](unsigned j)->const Q& {return values.at(step.args.at(j));};
    if(n.op=="const")return n.data[0];
    if(n.op=="add") {Q r=0;for(unsigned j=0;j<step.args.size();++j) {r+=n.data[j]*arg(j);if(!small(r))return {};}return r;}
    if(n.op=="mul") {Q r=n.data[0];for(unsigned j=0;j<step.args.size();++j) {r*=arg(j);if(!small(r))return {};}return r;}
    if(n.op=="idiv" || n.op=="imod") {
      if(n.data.size()!=1 || n.data[0]<=0)return {};
      Q quotient(floorq(arg(0)/n.data[0]));return n.op=="idiv"?quotient:arg(0)-quotient*n.data[0];
    }
    if(n.op=="word" || n.op=="signed") {
      if(arg(0).denominator()!=1 || n.data[0].denominator()!=1 || n.data[0]<=0 || n.data[0]>256)return {};
      unsigned width=n.data[0].numerator().convert_to<unsigned>();Z z=modz(arg(0).numerator(),pow2(width));
      if(n.op=="signed" && z>=pow2(width-1))z-=pow2(width);
      return Q(z);
    }
    if(n.op=="xor") {
      if(arg(0).denominator()!=1 || arg(1).denominator()!=1 || arg(0)<0 || arg(1)<0)return {};
      return Q(Z(arg(0).numerator()^arg(1).numerator()));
    }
    if(n.op=="word_or") {
      if(arg(0).denominator()!=1 || arg(1).denominator()!=1 || n.data.size()!=1 ||
         n.data[0]<1 || n.data[0]>256 || n.data[0].denominator()!=1)return {};
      Z modulus=pow2(n.data[0].numerator().convert_to<unsigned>());
      return Q(Z(modz(arg(0).numerator(),modulus)|modz(arg(1).numerator(),modulus)));
    }
    if(n.op=="compare") {
      bool b;
      if(n.tag=="eq")b=arg(0)==arg(1);else if(n.tag=="ne")b=arg(0)!=arg(1);
      else if(n.tag=="lt")b=arg(0)<arg(1);else if(n.tag=="le")b=arg(0)<=arg(1);
      else if(n.tag=="gt")b=arg(0)>arg(1);else if(n.tag=="ge")b=arg(0)>=arg(1);else return {};
      return Q(b);
    }
    if(n.op=="select")return arg(0)!=0?arg(1):arg(2);
    if(n.op=="exp2") {
      if(arg(0).denominator()!=1 || arg(0)<-256 || arg(0)>256)return {};
      int e=arg(0).numerator().convert_to<int>();return e>=0?Q(pow2(unsigned(e))):Q(1,pow2(unsigned(-e)));
    }
    return {};
  }
  Id normalize(Id root,const Resolve &mapper) {
    const auto &n=a.nodes[root];
    // Cheap rejection before memo allocation or any producer inspection.
    if((n.op!="add" && n.op!="select" && n.op!="signed" && n.op!="mul") ||
       n.args.size()>8 || !n.facts.finite || !n.bounds || n.bounds->first < -32 || n.bounds->second > 32)return root;
    // A full four-bit decoder must be capable of its entire output range.
    // Reject other bounded arithmetic before allocating or inspecting producers.
    bool signed_candidate=target!=FP4 && n.bounds->first<=-8 && n.bounds->second>=7;
    bool fp4_candidate=target!=Signed4 && n.bounds->first<=-6 && n.bounds->second>=6;
    if(!signed_candidate && !fp4_candidate)return root;
    return prove_decoder(root,mapper,signed_candidate,fp4_candidate);
  }
  // Only explicit importer boundaries use this entry. Shape recognition is a
  // candidate filter, never evidence of a decoder identity: the full bounded
  // program and all sixteen storage values must still pass below. Reference
  // Exp2/Select formulas need not carry expensive propagated interval facts.
  Id normalize_fp4_boundary(Id root,const Resolve &mapper) {
    const auto &n=a.nodes[root];
    if(target!=FP4 || n.op!="select" || n.args.size()!=3 || !n.facts.finite)return root;
    return prove_decoder(root,mapper,false,true);
  }
  Id prove_decoder(Id root,const Resolve &mapper,bool signed_candidate,bool fp4_candidate) {
    const auto &n=a.nodes[root];
    if(auto it=normalized.find(root);it!=normalized.end())return it->second;
    normalized.emplace(root,root);++a.stats["packed_decoder_candidates"];
    // A pure decoder commutes with selecting its storage bits. Reference INT4
    // commonly decodes both nibbles before selecting by the reduction parity.
    if(n.op=="select" && n.args.size()==3) {
      const auto &yes=a.nodes[n.args[1]],&no=a.nodes[n.args[2]];
      if(((signed_candidate && yes.op=="decode_s4") || (fp4_candidate && yes.op=="decode_fp4")) && yes.op==no.op &&
         yes.args.size()==1 && no.args.size()==1) {
        const auto &yb=a.nodes[yes.args[0]],&nb=a.nodes[no.args[0]];
        if(yb.op=="packedbits" && nb.op=="packedbits" && yb.tag==nb.tag &&
           yb.data==std::vector<Q>{Q(4)} && nb.data==yb.data) {
          Id address=select_address(n.args[0],yb.args[0],nb.args[0]);
          Id bits=a.intern("packedbits",{address},{Q(4)},yb.tag,{true,false,true},std::pair(Q(0),Q(15)));
          Id result=a.intern(yes.op,{bits},{},"",yes.facts,yes.bounds);
          normalized[root]=result;++a.stats["packed_decoder_selected_reads"];return result;
        }
      }
    }
    std::vector<Instruction> ops;std::unordered_map<Id,unsigned> memo;
    if(!program(root,mapper,ops,memo))return root;
    std::optional<Bits> first;Q low=0,high=0;
    for(const auto &step:ops)if(step.bits) {
      if(!first) {first=step.bits;high=Q(first->count);}
      if(step.bits->allocation!=first->allocation)return root;
      auto delta=a.value(a.sub(step.bits->address,first->address));
      if(!delta || delta->denominator()!=1)return root;
      low=std::min(low,*delta);high=std::max(high,*delta+Q(step.bits->count));
    }
    if(!first || high-low!=4)return root;
    Id base=a.add({first->address,a.constant(low)});
    std::string signature;
    auto field=[&](const std::string &value) {signature+=std::to_string(value.size())+":"+value;};
    for(auto &step:ops) {
      const auto &node=a.nodes[step.source];
      if(step.bits) {
        auto shift=a.value(a.sub(step.bits->address,base));
        if(!shift || shift->denominator()!=1 || *shift<0 || *shift+Q(step.bits->count)>4)return root;
        step.bit_shift=shift->numerator().convert_to<unsigned>();
        field("bits");field(std::to_string(step.bit_shift));field(std::to_string(step.bits->count));
      } else {
        field(node.op);field(node.tag);field(std::to_string(node.data.size()));
        for(const auto &q:node.data)field(str(q));
        field(std::to_string(step.args.size()));
        for(unsigned arg:step.args)field(std::to_string(arg));
      }
    }
    Decoder decoder=Unknown;
    if(auto it=patterns.find(signature);it!=patterns.end()) {decoder=it->second;++a.stats["packed_decoder_pattern_cache_hits"];}
    else {
      bool signed4=true,fp4=true;const Q magnitudes[]={Q(0),Q(1,2),Q(1),Q(3,2),Q(2),Q(3),Q(4),Q(6)};
      for(unsigned input=0;input<16 && (signed4 || fp4);++input) {
        std::vector<Q> values;values.reserve(ops.size());
        for(const auto &step:ops) {
          auto value=evaluate(step,values,input);
          if(!value || !small(*value)) {signed4=fp4=false;break;}
          values.push_back(*value);
        }
        if(!signed4 && !fp4)break;
        Q observed=values.back();
        signed4 &= observed==Q(input<8?int(input):int(input)-16);
        fp4 &= observed==(input<8?magnitudes[input]:-magnitudes[input-8]);
      }
      decoder=signed4?Signed4:(fp4?FP4:Unknown);patterns.emplace(std::move(signature),decoder);
      ++a.stats["packed_decoder_truth_tables"];
    }
    if(decoder==Unknown || (target!=Unknown && decoder!=target))return root;
    Id bits=a.intern("packedbits",{base},{Q(4)},first->allocation,{true,false,true},std::pair(Q(0),Q(15)));
    Id result=a.intern(decoder==Signed4?"decode_s4":"decode_fp4",{bits},{},"",{true,false,false},
                       decoder==Signed4?Bounds(std::pair(Q(-8),Q(7))):Bounds(std::pair(Q(-6),Q(6))));
    normalized[root]=result;++a.stats["packed_decoder_replacements"];return result;
  }
};
} // namespace rg
