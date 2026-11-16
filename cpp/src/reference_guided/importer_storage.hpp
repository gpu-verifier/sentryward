#pragma once
#include "importer.hpp"

namespace rg {

inline auto Importer::word(Id x, unsigned w) -> Id {
  auto b = a.nodes[x].bounds;
  Z lim = pow2(w);
  if (b && b->first >= 0 && b->first <= b->second && b->second < Q(lim))
    return x;
  const auto &n = a.nodes[x];
  if (n.op == "signed" && n.data[0] == Q(w))
    return word(n.args[0], w);
  if (n.op == "word" && n.data[0] <= Q(w))
    return x;
  auto ts = a.terms(x);
  if (std::all_of(ts.begin(), ts.end(),
                  [](auto &t) { return t.second.denominator() == 1; })) {
    std::vector<Id> rs;
    for (auto &[t, c] : ts) {
      Z coefficient = modz(c.numerator(), lim);
      // Keep negative residues negative when this proves a small unsigned
      // result (e.g. packed INT4 xor sign toggles). Congruence is unchanged.
      if (t != a.one && c < 0 && coefficient != 0) coefficient -= lim;
      // Reducing a coefficient modulo 2^w is valid only for integral
      // terms. Unknown terms retain their exact coefficient, rather than
      // being rejected or silently treated as integers.
      Q exact = Q(coefficient);
      if (exact != c && t != a.one && !packed_decoders.integer(t)) exact = c;
      rs.push_back(a.mul({a.constant(exact), t}));
    }
    Id r = a.add(rs);
    auto rb = a.nodes[r].bounds;
    if (rb && rb->first >= 0 && rb->first <= rb->second &&
        rb->second < Q(lim))
      return r;
    // A negative integer stride can arrive encoded as an unsigned residue.
    // Choose balanced coefficients only for proved integer atoms; changing
    // a coefficient by 2^w is otherwise not a congruence (e.g. x=1/2).
    // The whole candidate must still fit the unsigned word domain.
    rs.clear();
    bool changed = false, integral = true;
    for (const auto &[t, c] : ts) {
      if (t != a.one && !packed_decoders.integer(t)) { integral = false; break; }
      Z coefficient = modz(c.numerator(), lim);
      if (coefficient > lim / 2) { coefficient -= lim; changed = true; }
      rs.push_back(a.mul({a.constant(Q(coefficient)), t}));
    }
    if (integral && changed) {
      Id candidate = a.add(rs);
      auto cb = a.nodes[candidate].bounds;
      if (cb && cb->first >= 0 && cb->first <= cb->second && cb->second < Q(lim)) {
        ++a.stats["bounded_unsigned_stride_recoveries"];
        return candidate;
      }
    }
  }
  return a.intern("word", {x}, {Q(w)}, "", {true, false, true},
                  std::pair(Q(0), Q(lim - 1)));
}

inline auto Importer::unwrap(Id x, unsigned w) -> Id {
  while (a.nodes[x].op == "signed" && a.nodes[x].data[0] == Q(w))
    x = a.nodes[x].args[0];
  return a.nodes[x].op == "word" && a.nodes[x].data[0] == Q(w)
             ? a.nodes[x].args[0]
             : x;
}

// Recognize a bounded bit slice of an initial packed load. No value/IEEE
// inference: shifts and truncations only identify the original storage bits.
inline auto Importer::initial_bit_slice(Id x, unsigned lo, unsigned count, unsigned depth) -> std::optional<Id> {
  if(depth>8 || !count) return {};
  const auto &n=a.nodes[x];
  if(n.op=="readbits") {
    if(Q(lo+count)>n.data[0]) return {};
    return a.bit_slice(x,lo,count);
  }
  if(n.op=="extractbits") {
    if(Q(lo+count)>n.data[1]) return {};
    return initial_bit_slice(n.args[0],lo+n.data[0].numerator().convert_to<unsigned>(),count,depth+1);
  }
  if(n.op=="word") {
    if(Q(lo+count)>n.data[0]) return {};
    return initial_bit_slice(n.args[0],lo,count,depth+1);
  }
  if(n.op=="idiv" || n.op=="imod") {
    const Q &q=n.data[0];
    if(q.denominator()!=1 || q<=0) return {};
    Z divisor=q.numerator();if((divisor & (divisor-1))!=0) return {};
    unsigned shift=boost::multiprecision::msb(divisor);
    if(shift>256) return {};
    if(n.op=="imod") {
      if(lo+count>shift)return {};
      return initial_bit_slice(n.args[0],lo,count,depth+1);
    }
    auto bounds=a.nodes[n.args[0]].bounds;
    if(!bounds || bounds->first<0)return {};
    return initial_bit_slice(n.args[0],lo+shift,count,depth+1);
  }
  return {};
}

inline auto Importer::initial_typed_bits(Id x,const std::string &type) -> std::optional<Id> {
  // Raw transport does not establish UE4M3 format/padding validity. Only
  // explicitly typed initial inputs enter its finite-domain decoder.
  if (type == "ue4m3") return {};
  unsigned bits=width(type);if(bits%8) return {};
  const auto &n=a.nodes[x];
  Id read=x;Q bit_offset(0);
  if(n.op=="extractbits" && n.data[1]==Q(bits)) {read=n.args[0];bit_offset=n.data[0];}
  else if(n.op!="readbits" || n.data[0]!=Q(bits)) return {};
  const auto &r=a.nodes[read];
  if(r.op!="readbits" || bit_offset.denominator()!=1 || modz(bit_offset.numerator(),8)!=0) return {};
  if(!m.contains("layouts") || !m["layouts"].contains(r.tag) ||
     m["layouts"][r.tag]["element_bytes"].get<unsigned>()*8!=bits) return {};
  std::string raw=r.tag;Id address=r.args[0];
  return typed_read(raw,a.add({address,a.constant(bit_offset/Q(8))}),type);
}

// NumericalInterpretation of an already typed UE4M3 input.  This follows
// the seven payload bits in the saved expression's unsigned format, not an
// arbitrary b8 bitcast or a proof that PTX padding requirements were met.
// E4M3 finite excludes exactly the same low-seven-bit NaN payload (127);
// it does not constrain bit 7 and must not be treated as a padding premise.
inline auto Importer::typed_ue4m3_input(const std::string &raw, Id offset) -> Id {
  if (!ideal)
    throw Unsupported("typed UE4M3 input requires ideal_real numerical semantics");
  const std::string allocation = mapped_name(m, "allocations", raw);
  const auto *rule = contract ? contract->find(allocation, "e4m3") : nullptr;
  if (!rule || !rule->finite)
    throw Unsupported("typed UE4M3 input requires an explicit finite E4M3 premise: " + allocation);
  // Validate the complete byte and the correspondence before consuming the
  // premise; an unknown/out-of-range or uncovered address cannot use it.
  typed_read(raw, offset, "u8", true);
  require_contract_coverage(*rule, raw, offset, 1);
  Id payload = a.intern("readbits", {offset}, {Q(8)}, raw,
                        {true, false, true}, std::pair(Q(0), Q(255)));
  Id fraction = a.imod(payload, Z(8));
  // Use the same exact unsigned shift representation as the raw reference
  // decoder. The final modulus still excludes storage bit 7 from the value.
  Id exponent = a.imod(a.bit_slice(payload, 3, 5), Z(16));
  Id denormal = a.mul({a.constant(Q(1, 512)), fraction});
  Id normal = a.mul({a.exp2(a.sub(exponent, a.constant(7))),
                     a.add({a.one, a.mul({a.constant(Q(1, 8)), fraction})})});
  Id result = a.select(a.compare("eq", exponent, a.zero), denormal, normal);
  contract->applied.emplace(side, allocation, "e4m3");
  ++a.stats["typed_ue4m3_initial_decodes"];
  return result;
}

inline auto Importer::typed_e8m0_input(const std::string& raw,Id offset) -> Id {
  const auto allocation=mapped_name(m,"allocations",raw);
  const auto* rule=contract?contract->find(allocation,"e8m0"):nullptr;
  if(!ideal || !rule || !rule->finite)throw Unsupported("typed E8M0 input requires explicit finite E8M0 premise: "+allocation);
  typed_read(raw,offset,"u8",true);
  require_contract_coverage(*rule,raw,offset,1);
  Id payload=a.intern("readbits",{offset},{Q(8)},raw,{true,false,true},std::pair(Q(0),Q(255)));
  contract->applied.emplace(side,allocation,"e8m0");++a.stats["typed_e8m0_initial_decodes"];
  return a.exp2(a.sub(payload,a.constant(127)));
}

// Whether the region of a premise covers the `bytes` bytes of the element with canonical
// coordinates `coordinates` of raw input `raw` (InputRegion, input_contract.hpp). The canonical byte
// address is c = target offset + sum of coordinate x target stride (the layout both sides share);
// a record region needs c >= offset_bytes and (c - offset_bytes) mod record_stride in
// [first_byte, first_byte + byte_count - bytes], both from exact Arena bounds (affine residues reduce
// exactly; an unknown residue is not covered). O(rank) Arena terms, no enumeration.
inline auto Importer::region_covers(const InputRegion &region,const std::string &raw,const std::vector<Id> &coordinates,unsigned bytes) -> bool {
  if(region.whole())return true;
  if(!m.contains("layouts") || !m["layouts"].contains(raw))return false;
  const auto &l=m["layouts"][raw];
  const auto &target=l.contains("target_strides_bytes")?l["target_strides_bytes"]:l["source_strides_bytes"];
  if(target.size()!=coordinates.size())return false;
  std::vector<Id> terms{a.constant(jq(l.value("target_offset",J(0))))};
  for(size_t j=0;j<coordinates.size();++j)terms.push_back(a.mul({a.constant(jq(target[j])),coordinates[j]}));
  const Id address=a.add(terms);
  const auto &records=*region.records;
  const auto &address_bounds=a.nodes[address].bounds;
  if(!address_bounds || address_bounds->first<Q(records.offset))return false;
  const Id residue=a.imod(a.sub(address,a.constant(Q(records.offset))),records.stride);
  const auto &residue_bounds=a.nodes[residue].bounds;
  return residue_bounds && residue_bounds->first>=Q(records.first) &&
         residue_bounds->second+Q(bytes)<=Q(records.first+records.count);
}

// The same for a read at source byte `offset` of raw input `raw`: its canonical coordinates are those
// of the validated byte read (typed_read, which checks alignment, layout digits and extent). A fixed
// byte (no coordinate read) is not covered. Throws when the region does not cover the read.
inline auto Importer::require_contract_coverage(const InputContract::Rule &rule,const std::string &raw,Id offset,unsigned bytes) -> void {
  if(rule.region.whole())return;
  const Id byte=typed_read(raw,offset,"u8",true);
  const std::vector<Id> coordinates=a.nodes[byte].args; // copied: region_covers interns
  if(a.nodes[byte].op!="read" || !region_covers(rule.region,raw,coordinates,bytes))
    throw Unsupported("Could not prove that the input premise's region covers the read",
                      "input_contract_region_not_covered",
                      {{"object",rule.object},{"element_type",rule.element_type},
                       {"region",rule.region.declaration()},{"offset_expression",a.render(offset,4)}});
  ++a.stats["input_contract_region_covered_reads"];
}

// Decode a storage byte only under its object's covering finite-E8M0 premise.
// Follow exact, nonnegative integer transport: slices, power-of-two arithmetic,
// nonoverlapping replicas/fields, and Select arms. Initial reads require aligned
// little-endian bytes; constants decode as 2^(byte-127), rejecting NaN code 255.
// Unknown transport or uncovered bytes decline. Memoized by (node, bit offset),
// with at most 16 transport steps and four nested selections.
inline auto Importer::premise_e8m0_byte(Id x,unsigned lo,unsigned depth) -> std::optional<Id> {
  if(!contract || !ideal || depth>4)return {};
  const auto key=std::pair(x,lo);
  if(auto it=premise_e8m0_bytes.find(key);it!=premise_e8m0_bytes.end())return it->second;
  const auto nonnegative_integer=[&](Id y) {return transport_integer(y);};
  const auto remember=[&](Id result) -> std::optional<Id> {
    premise_e8m0_bytes.emplace(key,result);return result;
  };
  for(unsigned step=0;step<16;++step) {
    if(lo>256)return {};
    const auto& n=a.nodes[x];
    if(n.op=="const") {
      const Q& v=n.data[0];
      if(v.denominator()!=1 || v<0)return {};
      const Z byte=modz(floorz(v.numerator(),pow2(lo)),Z(256));
      if(byte==255)throw Unsupported("E8M0 NaN encoding (byte 255) in a raw E8M0 bitcast");
      ++a.stats["premise_e8m0_constant_bytes"];
      return remember(a.constant(Q(byte)>=Q(127)?Q(pow2((byte-127).convert_to<unsigned>())):
                                                  Q(Z(1),pow2((127-byte).convert_to<unsigned>()))));
    }
    if(n.op=="readbits") {
      if(n.args.size()!=1 || n.data.size()!=1 || lo%8 || Q(lo+8)>n.data[0])return {};
      const std::string raw=n.tag;const Id address=n.args[0]; // copied: interning moves nodes
      const auto* rule=contract->find(mapped_name(m,"allocations",raw),"e8m0");
      if(!rule || !rule->finite)return {};
      const Id byte=typed_e8m0_input(raw,a.add({address,a.constant(Q(lo/8))}));
      ++a.stats["premise_e8m0_storage_bytes"];
      return remember(byte);
    }
    if(n.op=="select" && n.args.size()==3) {
      const Id p=n.args[0],yes=n.args[1],no=n.args[2];
      if(!nonnegative_integer(yes) || !nonnegative_integer(no))return {};
      auto first=premise_e8m0_byte(yes,lo,depth+1);
      if(!first)return {};
      auto second=premise_e8m0_byte(no,lo,depth+1);
      if(!second)return {};
      ++a.stats["premise_e8m0_selected_bytes"];
      return remember(a.select(p,*first,*second));
    }
    if(n.op=="add") {
      const auto field=disjoint_sum_field(x,lo,8);
      if(!field)return {};
      x=field->first;lo=field->second;continue;
    }
    if(n.args.size()!=1 || !nonnegative_integer(n.args[0]))return {};
    if(n.op=="word" || n.op=="imod") {
      std::optional<unsigned> bits;
      if(n.op=="word") {
        if(n.data.size()==1 && n.data[0].denominator()==1 && n.data[0]>0 && n.data[0]<=256)
          bits=n.data[0].numerator().convert_to<unsigned>();
      } else if(n.data.size()==1)bits=PackedDecodePass::power_of_two(n.data[0]);
      if(!bits || lo+8>*bits)return {};
    } else if(n.op=="idiv") {
      auto bits=n.data.size()==1?PackedDecodePass::power_of_two(n.data[0]):std::optional<unsigned>{};
      if(!bits)return {};
      lo+=*bits;
    } else if(n.op=="extractbits") {
      if(n.data.size()!=2 || n.data[0].denominator()!=1 || n.data[1].denominator()!=1 ||
         n.data[0]<0 || n.data[0]>256 || n.data[1]<=0 || n.data[1]>256 || Q(lo+8)>n.data[1])return {};
      lo+=n.data[0].numerator().convert_to<unsigned>();
    } else if(n.op=="mul" && n.data.size()==1) {
      const auto copy=replicated_copy(n.data[0],n.args[0],lo,8);
      if(!copy)return {};
      lo-=*copy;
    } else return {};
    x=n.args[0];
  }
  return {};
}

// Demand one complete typed floating lane from exact integer transport.
// The walk follows only one containing field; other fields are checked by
// bounded integer/range summaries, never by visiting numeric ancestors.
// A failed proof does not make an integer payload a numeric conversion.
inline auto Importer::floating_lane(Id source,const std::string& type) -> std::optional<Id> {
  const auto key=std::pair(source,type);
  if(auto it=floating_lane_values.find(key);it!=floating_lane_values.end())return it->second;
  const unsigned count=width(type);
  if(type!="f16" && type!="bf16" && type!="f32" && type!="f64")return {};
  unsigned lo=0,budget=PackedDecodePass::max_nodes;
  Id value=source;
  auto remember=[&](Id result) -> std::optional<Id> {
    floating_lane_values.emplace(key,result);++a.stats["exact_packed_floating_lanes"];return result;
  };
  while(budget) {
    --budget;++a.stats["floating_lane_transport_visits"];
    // No arena insertion occurs until a terminal field is found. References
    // below cannot be invalidated by arithmetic interning during traversal.
    const auto& n=a.nodes[value];
    if(n.op=="floatbits") {
      if(lo==0 && n.args.size()==1 && n.data==std::vector<Q>{Q(count)} && n.tag==type)
        return remember(n.args[0]);
      return {};
    }
    if(n.op=="readbits") {
      if(n.args.size()!=1 || n.data.size()!=1 || lo%8 || Q(lo+count)>n.data[0])return {};
      const auto raw=n.tag;const auto address=n.args[0];
      return remember(typed_read(raw,a.add({address,a.constant(lo/8)}),type));
    }
    if(n.args.size()==1 && (n.op=="word" || n.op=="imod" || n.op=="idiv" || n.op=="extractbits")) {
      unsigned shift=0,available=256;
      if(n.op=="word") {
        if(n.data.size()!=1 || n.data[0].denominator()!=1 || n.data[0]<=0 || n.data[0]>256)return {};
        available=n.data[0].numerator().convert_to<unsigned>();
      } else if(n.op=="extractbits") {
        if(n.data.size()!=2 || n.data[0].denominator()!=1 || n.data[1].denominator()!=1 ||
           n.data[0]<0 || n.data[0]>256 || n.data[1]<=0 || n.data[1]>256)return {};
        shift=n.data[0].numerator().convert_to<unsigned>();available=n.data[1].numerator().convert_to<unsigned>();
      } else {
        auto radix=n.data.size()==1?PackedDecodePass::power_of_two(n.data[0]):std::optional<unsigned>{};
        if(!radix)return {};
        if(n.op=="idiv")shift=*radix;else available=*radix;
      }
      if(lo+count>available || lo+shift+count>256)return {};
      const auto& input=a.nodes[n.args[0]];
      if(!input.bounds || input.bounds->first<0 || !packed_decoders.integer_impl(n.args[0],budget))return {};
      value=n.args[0];lo+=shift;continue;
    }
    if(n.op=="mul" && n.args.size()==1 && n.data.size()==1) {
      auto shift=PackedDecodePass::power_of_two(n.data[0]);
      if(!shift || lo<*shift)return {};
      const auto& input=a.nodes[n.args[0]];
      if(!input.bounds || input.bounds->first<0 || !packed_decoders.integer_impl(n.args[0],budget))return {};
      value=n.args[0];lo-=*shift;continue;
    }
    if(n.op!="add" || n.args.empty() || n.args.size()>8 || n.args.size()!=n.data.size())return {};
    struct Field {Id value;unsigned first,last;};
    std::vector<Field> fields;fields.reserve(n.args.size());
    std::optional<Field> selected;
    for(size_t i=0;i<n.args.size();++i) {
      if(!budget)return {};
      --budget;
      auto shift=PackedDecodePass::power_of_two(n.data[i]);const auto& part=a.nodes[n.args[i]];
      if(!shift || !part.bounds || part.bounds->first<0 || !part.facts.finite ||
         !packed_decoders.integer_impl(n.args[i],budget))return {};
      const Z high=floorq(part.bounds->second);
      if(high<0)return {};
      if(high==0)continue;
      if(high>=pow2(256))return {};
      unsigned bits=boost::multiprecision::msb(high)+1;
      if(*shift+bits>256)return {};
      Field field{n.args[i],*shift,*shift+bits};
      for(auto previous:fields)
        if(field.first<previous.last && previous.first<field.last)return {};
      fields.push_back(field);
      if(lo>=field.first && lo+count<=field.last)selected=field;
    }
    if(!selected)return {}; // crossing fields or including padding is not a typed lane
    value=selected->value;lo-=selected->first;
  }
  return {};
}

// Whether y is a nonnegative integer with finite bounds, also through selections of integers (a
// selection yields one alternative) and remainders, sums and products with integer coefficients of
// integers; other forms use the packed decoders' proof. Bounded: depth 8, 32 nodes. Shared by the
// bit-transport rules below (sliced_selection, premise_e8m0_byte).
inline auto Importer::transport_integer(Id y) -> bool {
  const auto& b=a.nodes[y].bounds;
  if(!b || b->first<0 || !a.nodes[y].facts.finite)return false;
  unsigned budget=PackedDecodePass::max_nodes;
  const auto integral=[&](const auto& self,Id z,unsigned level) -> bool {
    if(level>8 || budget==0)return false;
    --budget;
    const auto& n=a.nodes[z];
    if(n.op=="select" && n.args.size()==3)return self(self,n.args[1],level+1) && self(self,n.args[2],level+1);
    if(n.op=="imod" && n.args.size()==1)return n.data[0].denominator()==1 && self(self,n.args[0],level+1);
    if((n.op=="add" || n.op=="mul") && !n.args.empty() && n.args.size()<=8) {
      for(const auto& q:n.data)if(q.denominator()!=1)return false;
      for(Id arg:n.args)if(arg!=a.one && !self(self,arg,level+1))return false;
      return true;
    }
    return packed_decoders.integer(z);
  };
  return integral(integral,y,0);
}

// Bits [lo, lo + width) of c*y, for an integer c > 0 and an integer y in [0, 2^b): when c = sum of 2^s_i
// with s_(i+1) - s_i >= b (the copies y*2^s_i do not overlap, so c*y is their carry-free sum) and the
// slice lies inside one copy's span [s_i, s_(i+1)) (the last span unbounded), the slice is y's bits from
// lo - s_i: returns s_i. A single power of two is the plain shift. mul.lo x, 257 and prmt with a
// repeated byte selector duplicate a byte this way. O(set bits of c).
inline auto Importer::replicated_copy(const Q& c,Id y,unsigned lo,unsigned width) -> std::optional<unsigned> {
  if(c.denominator()!=1 || c<=0 || c>Q(pow2(256)))return {};
  const auto& in=a.nodes[y].bounds;
  if(!in || in->first<0)return {};
  const Z high=floorq(in->second);
  const unsigned bits=high>0?unsigned(boost::multiprecision::msb(high))+1:1;
  std::vector<unsigned> copies;
  for(Z rest=c.numerator();rest!=0;rest&=rest-1)copies.push_back(unsigned(boost::multiprecision::lsb(rest)));
  std::optional<unsigned> copy;
  for(size_t i=0;i<copies.size();++i) {
    if(i+1<copies.size() && copies[i+1]-copies[i]<bits)return {};
    if(copies[i]<=lo && (i+1==copies.size() || lo+width<=copies[i+1]))copy=copies[i];
  }
  if(copy && copies.size()>1)++a.stats["replicated_copy_slices"];
  return copy;
}

// Bits [lo, lo + width) of a sum of terms c_i * y_i (each y_i a nonnegative integer, transport_integer;
// each c_i a positive integer): every set bit s of c_i places a copy of y_i on [s, s + b_i), b_i the bit
// width of y_i's upper bound. When all copies of all terms are pairwise disjoint the sum has no carries
// and is the union of the copies, so a slice that no copy other than the last one starting at or below
// lo meets reads that copy's term at lo - s (its bits past b_i are zero, as are the sum's there).
// Returns (term, bit offset). Example: prmt with selector 0x11 gives b1*257 + b0*0x01010000. At most 8
// terms and 64 copies.
inline auto Importer::disjoint_sum_field(Id sum,unsigned lo,unsigned width) -> std::optional<std::pair<Id,unsigned>> {
  const auto& n=a.nodes[sum];
  if(n.op!="add" || n.args.empty() || n.args.size()>8 || n.args.size()!=n.data.size())return {};
  struct Copy {Id term;unsigned first,last;};
  std::vector<Copy> copies;
  const auto terms=n.args;const auto coefficients=n.data;
  for(size_t i=0;i<terms.size();++i) {
    const Q& c=coefficients[i];
    if(c.denominator()!=1 || c<=0 || c>Q(pow2(256)) || !transport_integer(terms[i]))return {};
    const Z high=floorq(a.nodes[terms[i]].bounds->second);
    if(high==0)continue;
    const unsigned bits=unsigned(boost::multiprecision::msb(high))+1;
    for(Z rest=c.numerator();rest!=0;rest&=rest-1) {
      const unsigned first=unsigned(boost::multiprecision::lsb(rest));
      if(copies.size()>=64 || first+bits>512)return {};
      copies.push_back({terms[i],first,first+bits});
    }
  }
  for(size_t i=0;i<copies.size();++i)
    for(size_t j=i+1;j<copies.size();++j)
      if(copies[i].first<copies[j].last && copies[j].first<copies[i].last)return {};
  std::optional<Copy> chosen;
  for(const auto& copy:copies)
    if(copy.first<=lo && (!chosen || copy.first>chosen->first))chosen=copy;
  if(!chosen)return {};
  for(const auto& copy:copies)
    if(copy.first!=chosen->first && copy.first<lo+width && lo<copy.last)return {};
  ++a.stats["disjoint_sum_field_slices"];
  return std::pair(chosen->term,lo-chosen->first);
}

// Bits [lo, lo + width) of x as an exact value, following the same exact bit transport as
// sliced_selection (truncation, remainder/quotient by 2^k, bit extraction, replicated products,
// disjoint sums, each over a nonnegative integer) and constant masks: x & m restricted to the slice
// passes when m's slice bits are all ones, narrows to its low k bits when they are 2^k - 1 (the rest
// are zero), and is 0 when they are zero. It stops at the first value y at bit offset 0 with
// y < 2^width (the field is y itself), else ends with Arena::bit_slice. Never inspects arithmetic
// producers. At most 16 shells.
inline auto Importer::exact_bit_field(Id x,unsigned lo,unsigned width) -> Id {
  const auto fits=[&](Id y) {
    const auto& b=a.nodes[y].bounds;
    return lo==0 && b && b->first>=0 && b->second<Q(pow2(width)) && transport_integer(y);
  };
  for(unsigned depth=0;depth<16 && width;++depth) {
    if(fits(x))return x;
    const auto& n=a.nodes[x];
    if(n.op=="const")break;
    if(n.op=="add") {
      const auto field=disjoint_sum_field(x,lo,width);
      if(!field)break;
      x=field->first;lo=field->second;continue;
    }
    if(n.args.size()!=1 || !transport_integer(n.args[0]))break;
    const Id input=n.args[0];
    if(n.op=="constant_bit_mask" && n.data.size()==2 && n.data[0].denominator()==1 && n.data[0]>=0) {
      const Z slice=(n.data[0].numerator()>>lo)&(pow2(width)-1);
      if(slice==0)return a.zero;
      unsigned run=0;
      while(run<width && ((slice>>run)&1)!=0)++run;
      if((slice>>run)!=0)break; // not a low run of ones
      width=run;
    } else if(n.op=="word" || n.op=="imod") {
      std::optional<unsigned> bits;
      if(n.op=="word") {
        if(n.data.size()==1 && n.data[0].denominator()==1 && n.data[0]>0 && n.data[0]<=256)
          bits=n.data[0].numerator().convert_to<unsigned>();
      } else if(n.data.size()==1)bits=PackedDecodePass::power_of_two(n.data[0]);
      if(!bits || lo>=*bits)break;
      width=std::min(width,*bits-lo);
    } else if(n.op=="idiv") {
      auto bits=n.data.size()==1?PackedDecodePass::power_of_two(n.data[0]):std::optional<unsigned>{};
      if(!bits)break;
      lo+=*bits;
    } else if(n.op=="extractbits") {
      if(n.data.size()!=2 || n.data[0].denominator()!=1 || n.data[1].denominator()!=1 ||
         n.data[0]<0 || n.data[0]>256 || n.data[1]<=0 || n.data[1]>256 || Q(lo)>=n.data[1])break;
      width=std::min<unsigned>(width,n.data[1].numerator().convert_to<unsigned>()-lo);
      lo+=n.data[0].numerator().convert_to<unsigned>();
    } else if(n.op=="mul" && n.data.size()==1) {
      const auto copy=replicated_copy(n.data[0],input,lo,width);
      if(!copy)break;
      lo-=*copy;
    } else break;
    if(lo+width>256)break;
    x=input;
  }
  if(fits(x))return x;
  return a.bit_slice(x,lo,width);
}

// prmt.b32 d, a, b, c with a selector c that is not a constant: byte i of d is byte n_i of the
// 64-bit {b, a} (a in bytes 0..3), n_i = bits [4i, 4i+3) of c, when bit 4i+3 (the sign-replication
// mode) is provably zero (exact_bit_field). That byte is bits [0, 8) of an unsigned word shift by
// 8 n_i: of a (32 bits) when n_i < 4, of b by 8 (n_i - 4) when n_i >= 4, of {b, a} (64 bits) when the
// bound of n_i allows both (PTX shr clamps a shift >= the width to 0, which never happens below
// 32 for a). Returns nothing when a mode bit is not proved zero. O(4) shifts.
inline auto Importer::symbolic_byte_permute(Id x,Id y,Id control) -> std::optional<Id> {
  std::vector<Id> bytes;
  for(unsigned i=0;i<4;++i) {
    const Id mode=exact_bit_field(control,4*i+3,1);
    if(mode!=a.zero)return {};
    const Id lane=exact_bit_field(control,4*i,3);
    const auto& lb=a.nodes[lane].bounds;
    if(!lb || lb->first<0 || lb->second>7 || !transport_integer(lane))return {};
    Id byte;
    if(lb->second<4)byte=a.bit_slice(a.word_shr(x,a.mul({a.constant(8),lane}),32,false),0,8);
    else if(lb->first>=4)byte=a.bit_slice(a.word_shr(y,a.mul({a.constant(8),a.sub(lane,a.constant(4))}),32,false),0,8);
    else byte=a.bit_slice(a.word_shr(a.add({x,a.mul({a.constant(Q(pow2(32))),y})}),a.mul({a.constant(8),lane}),64,false),0,8);
    bytes.push_back(a.mul({a.constant(Q(pow2(8*i))),byte}));
  }
  ++a.stats["symbolic_byte_permutes"];
  return a.add(bytes);
}

// Push a bit slice through exact nonnegative-integer transport to a Select,
// then slice each arm. Supported shells are truncation, power-of-two arithmetic,
// bit extraction, nonoverlapping replicas and disjoint fields. Infer no new
// facts about either arm. Unknown transport declines; at most 16 shells.
inline auto Importer::sliced_selection(Id x,unsigned width) -> std::optional<Id> {
  if(!width || width>256)return {};
  const auto nonnegative_integer=[&](Id y) {return transport_integer(y);};
  unsigned lo=0;
  for(unsigned depth=0;depth<16;++depth) {
    const auto& n=a.nodes[x];
    if(n.op=="select" && n.args.size()==3) {
      const Id p=n.args[0],yes=n.args[1],no=n.args[2];
      if(!nonnegative_integer(yes) || !nonnegative_integer(no))return {};
      const Id sliced_yes=a.bit_slice(yes,lo,width);
      const Id sliced_no=a.bit_slice(no,lo,width);
      return a.select(p,sliced_yes,sliced_no);
    }
    if(n.op=="add") {
      const auto field=disjoint_sum_field(x,lo,width);
      if(!field)return {};
      x=field->first;lo=field->second;
      continue;
    }
    // An unsigned word shift by a symbolic count of a selection is the selection of the two shifted
    // alternatives (the shift applies to the selected value): the bits are the selection of their bits.
    if(n.op=="word_shr" && n.args.size()==2 && n.data.size()==2 && n.data[1]==0 && n.data[0].denominator()==1 &&
       n.data[0]>0 && n.data[0]<=256) {
      const auto& source=a.nodes[n.args[0]];
      if(source.op!="select" || source.args.size()!=3)return {};
      const Id p=source.args[0],yes=source.args[1],no=source.args[2],shift=n.args[1];
      const unsigned bits=n.data[0].numerator().convert_to<unsigned>();
      if(!transport_integer(yes) || !transport_integer(no) || !transport_integer(shift))return {};
      ++a.stats["sliced_shifted_selections"];
      return a.select(p,a.bit_slice(a.word_shr(yes,shift,bits,false),lo,width),
                      a.bit_slice(a.word_shr(no,shift,bits,false),lo,width));
    }
    if(n.args.size()!=1)return {};
    const Id input=n.args[0];
    const auto& in=a.nodes[input];
    const bool select_input=in.op=="select" && in.args.size()==3;
    if(!select_input && !nonnegative_integer(input))return {};
    if(n.op=="word" || n.op=="imod" || n.op=="idiv") {
      std::optional<unsigned> bits;
      if(n.op=="word") {
        if(n.data.size()==1 && n.data[0].denominator()==1 && n.data[0]>0 && n.data[0]<=256)
          bits=n.data[0].numerator().convert_to<unsigned>();
      } else if(n.data.size()==1)bits=PackedDecodePass::power_of_two(n.data[0]);
      if(!bits)return {};
      if(n.op=="idiv")lo+=*bits;
      else if(lo+width>*bits)return {}; // the slice would read the truncated (zero) bits
    } else if(n.op=="extractbits") {
      if(n.data.size()!=2 || n.data[0].denominator()!=1 || n.data[1].denominator()!=1 ||
         n.data[0]<0 || n.data[0]>256 || n.data[1]<=0 || n.data[1]>256)return {};
      if(Q(lo+width)>n.data[1])return {};
      lo+=n.data[0].numerator().convert_to<unsigned>();
    } else if(n.op=="mul" && n.data.size()==1) {
      const auto copy=replicated_copy(n.data[0],input,lo,width);
      if(!copy)return {};
      lo-=*copy;
    } else return {};
    if(lo+width>256)return {};
    x=input;
  }
  return {};
}

// One node rebuilt over new arguments `xs` through the Arena's own constructors, so it takes its
// canonical form. Input reads, storage bits and floating bit patterns keep their tag, data, facts and
// bounds (they depend on their coordinates only through these operands). With `existing_reads`, such a
// node must already exist (find_node), else invalid: the Arena keys nodes without facts and bounds, so
// a new one would give the element it reads these facts and bounds for every later read of it, which
// that element need not have (an input premise may cover some bytes of a record only). Any other
// operator is not rebuilt: invalid.
inline auto Importer::rebuild_node(const std::string& op,const std::string& tag,const std::vector<Q>& data,const Facts& facts,
                const Bounds& bounds,std::vector<Id> xs,bool existing_reads) -> Id {
  const auto natural=[&](size_t k)->std::optional<unsigned> {
    if(k>=data.size() || data[k].denominator()!=1 || data[k]<0 || data[k]>256)return {};
    return data[k].numerator().convert_to<unsigned>();
  };
  try {
    if((op=="add" || op=="add_raw") && xs.size()==data.size()) {
      for(size_t i=0;i<xs.size();++i)xs[i]=a.mul({a.constant(data[i]),xs[i]});
      return a.add(xs);
    }
    if(op=="mul" && data.size()==1) {xs.push_back(a.constant(data[0]));return a.mul(xs);}
    if(op=="select" && xs.size()==3 && data.empty())return a.select(xs[0],xs[1],xs[2]);
    if(op=="compare" && xs.size()==2 && data.empty())return a.compare(tag,xs[0],xs[1]);
    if((op=="idiv" || op=="imod") && xs.size()==1 && data.size()==1 && data[0].denominator()==1 && data[0]>0)
      return op=="idiv"?a.idiv(xs[0],data[0].numerator()):a.imod(xs[0],data[0].numerator());
    if(op=="max" && !xs.empty())return a.maximum(xs);
    if((op=="exp" || op=="exp2") && xs.size()==1)return a.exponential(xs[0],op=="exp2");
    if(op=="clamp" && xs.size()==1 && data.size()==2)return a.clamp(xs[0],data[0],data[1]);
    if(op=="signed" && xs.size()==1 && natural(0))return a.signed_word(xs[0],*natural(0));
    if(op=="word" && xs.size()==1 && natural(0))return word(xs[0],*natural(0));
    if(op=="extractbits" && xs.size()==1 && natural(0) && natural(1))return a.bit_slice(xs[0],*natural(0),*natural(1));
    if(op=="read" || op=="readbits" || op=="floatbits") {
      if(!existing_reads)return a.intern(op,xs,data,tag,facts,bounds);
      const auto found=a.find_node(op,xs,data,tag);
      return found?*found:invalid;
    }
  } catch(const Unsupported &) {}
  return invalid;
}

// The value of `root` with the variable `var` replaced by `value`, rebuilt bottom-up (rebuild_node,
// with `existing_reads`). Only nodes that depend on `var` are visited (Arena free-variable sets);
// invalid when one of them is not rebuildable. At most `budget` rebuilt nodes.
inline auto Importer::instantiate(Id root,Id var,Id value,bool existing_reads,size_t budget) -> Id {
  std::unordered_map<Id,Id> memo;
  std::vector<std::pair<Id,bool>> todo{{root,false}};
  while(!todo.empty()) {
    a.check();
    auto [id,ready]=todo.back();todo.pop_back();
    if(memo.contains(id))continue;
    if(id==var){memo[id]=value;continue;}
    if(!a.depends(id,var)){memo[id]=id;continue;}
    if(!ready) {
      if(!budget--)return invalid;
      todo.emplace_back(id,true);
      for(Id child:a.nodes[id].args)if(!memo.contains(child))todo.emplace_back(child,false);
      continue;
    }
    // Copy the node's fields: the constructors may move Arena storage.
    const std::string op=a.nodes[id].op,tag=a.nodes[id].tag;
    const std::vector<Q> data=a.nodes[id].data;
    const Facts facts=a.nodes[id].facts;const Bounds bounds=a.nodes[id].bounds;
    std::vector<Id> xs;
    for(Id child:a.nodes[id].args)xs.push_back(memo.at(child));
    if(std::find(xs.begin(),xs.end(),invalid)!=xs.end())return invalid;
    const Id result=rebuild_node(op,tag,data,facts,bounds,std::move(xs),existing_reads);
    if(result==invalid)return invalid;
    memo[id]=result;
  }
  return memo.at(root);
}

// A proposed template T(x) of two values with T(k0) = A and T(k0 + 1) = B, or invalid. Equal values
// are shared. Different integer terms whose difference is a constant d give A + d(x - k0); quotients
// and remainders by constants are first brought to one common divisor (idiv(N, D) = idiv(cN, cD),
// c imod(N, D) = imod(cN, cD)) and their numerators interpolated the same way. Otherwise nodes with the
// same operator, tag, data and arity are matched argument by argument and rebuilt (rebuild_node).
// Only a proposal: record_lane_template verifies T at every lane. At most 2048 visited pairs.
inline auto Importer::lane_anti_unify(Id A,Id B,Id x,const Z &k0,std::map<std::pair<Id,Id>,Id> &memo) -> Id {
  if(A==B)return A;
  if(auto it=memo.find({A,B});it!=memo.end())return it->second;
  if(memo.size()>2048)return invalid;
  memo[{A,B}]=invalid;
  Id result=invalid;
  const auto integral=[&](Id y){
    return a.nodes[y].facts.finite && (a.known_bits(y).integer || packed_decoders.integer(y));
  };
  if(integral(A) && integral(B)) {
    const Id lane_offset=a.sub(x,a.constant(Q(k0)));
    // y = scale * op(numerator, divisor) for op idiv (scale 1) or imod.
    const auto modular=[&](Id y,std::string& op,Z& divisor,Id& numerator,Z& scale)->bool {
      scale=1;Id inner=y;
      if(a.nodes[y].op=="mul" && a.nodes[y].args.size()==1 && a.nodes[y].data.size()==1 &&
         a.nodes[y].data[0].denominator()==1 && a.nodes[y].data[0]>0) {
        scale=a.nodes[y].data[0].numerator();inner=a.nodes[y].args[0];
      }
      const auto& m=a.nodes[inner];
      if((m.op!="idiv" && m.op!="imod") || m.args.size()!=1 || m.data.size()!=1 ||
         m.data[0].denominator()!=1 || m.data[0]<=0 || (m.op=="idiv" && scale!=1))return false;
      op=m.op;divisor=m.data[0].numerator();numerator=m.args[0];return true;
    };
    if(auto d=a.value(a.sub(B,A)))
      result=a.add({A,a.mul({a.constant(*d),lane_offset})});
    else {
      std::string op0,op1;Z d0,d1,s0,s1;Id n0=invalid,n1=invalid;
      if(modular(A,op0,d0,n0,s0) && modular(B,op1,d1,n1,s1) && op0==op1) {
        const Z D0=d0*s0,D1=d1*s1,L=D0/gcdz(D0,D1)*D1;
        const Id N0=a.mul({a.constant(Q(L/d0)),n0}),N1=a.mul({a.constant(Q(L/d1)),n1});
        if(auto d=a.value(a.sub(N1,N0))) {
          const Id N=a.add({N0,a.mul({a.constant(*d),lane_offset})});
          result=op0=="idiv"?a.idiv(N,L):a.imod(N,L);
        }
      }
    }
  }
  // Commutative sums, products and maxima keep their operands sorted by Arena id, so the lane-varying
  // operands need not sit at the same positions: match equal operands (with equal coefficients) first,
  // then pair the remaining ones in order of operator, tag and data (each pair anti-unified).
  const auto& opA=a.nodes[A].op;
  if(result==invalid && opA==a.nodes[B].op && (opA=="add" || opA=="mul" || opA=="max") &&
     a.nodes[A].args.size()==a.nodes[B].args.size() && a.nodes[A].args.size()>1 &&
     (opA!="add" || a.nodes[A].data.size()==a.nodes[A].args.size()) &&
     (opA!="add" || a.nodes[B].data.size()==a.nodes[B].args.size()) &&
     (opA=="add" || a.nodes[A].data==a.nodes[B].data)) {
    const std::string op=opA,tag=a.nodes[A].tag;
    const Facts facts=a.nodes[A].facts;const Bounds bounds=a.nodes[A].bounds;
    const auto argsA=a.nodes[A].args,argsB=a.nodes[B].args;
    const auto dataA=a.nodes[A].data,dataB=a.nodes[B].data;
    const bool sum=op=="add";
    std::vector<bool> usedB(argsB.size(),false);
    std::vector<size_t> restA;
    std::vector<Id> children;std::vector<Q> coefficients;
    for(size_t i=0;i<argsA.size();++i) {
      bool matched=false;
      for(size_t j=0;j<argsB.size() && !matched;++j)
        if(!usedB[j] && argsB[j]==argsA[i] && (!sum || dataB[j]==dataA[i])) {
          usedB[j]=matched=true;children.push_back(argsA[i]);if(sum)coefficients.push_back(dataA[i]);
        }
      if(!matched)restA.push_back(i);
    }
    bool failed=false;
    for(size_t i:restA) {
      bool matched=false;
      for(size_t j=0;j<argsB.size() && !matched;++j) {
        if(usedB[j] || (sum && dataB[j]!=dataA[i]) || a.nodes[argsB[j]].op!=a.nodes[argsA[i]].op ||
           a.nodes[argsB[j]].tag!=a.nodes[argsA[i]].tag)continue;
        const Id child=lane_anti_unify(argsA[i],argsB[j],x,k0,memo);
        if(child==invalid)continue;
        usedB[j]=matched=true;children.push_back(child);if(sum)coefficients.push_back(dataA[i]);
      }
      if(!matched){failed=true;break;}
    }
    if(!failed && !restA.empty())
      result=rebuild_node(op,tag,sum?coefficients:dataA,facts,bounds,std::move(children));
  }
  if(result==invalid && a.nodes[A].op==a.nodes[B].op && a.nodes[A].tag==a.nodes[B].tag &&
     a.nodes[A].data==a.nodes[B].data && a.nodes[A].args.size()==a.nodes[B].args.size() && !a.nodes[A].args.empty()) {
    const std::string op=a.nodes[A].op,tag=a.nodes[A].tag;
    const std::vector<Q> data=a.nodes[A].data;
    const Facts facts=a.nodes[A].facts;const Bounds bounds=a.nodes[A].bounds;
    const auto argsA=a.nodes[A].args,argsB=a.nodes[B].args;
    std::vector<Id> children;bool failed=false;
    for(size_t i=0;i<argsA.size() && !failed;++i) {
      children.push_back(lane_anti_unify(argsA[i],argsB[i],x,k0,memo));
      failed=children.back()==invalid;
    }
    if(!failed)result=children==argsA?A:rebuild_node(op,tag,data,facts,bounds,std::move(children));
  }
  memo[{A,B}]=result;
  return result;
}

// For a packed record R = sum over lanes k of 2^(w k) * floatbits_w(V_k) (same floating type, one field
// per lane, lanes k0..k1 present): templates with V_k = T_(k mod p)(k div p) for every k in [k0, k1], for
// the smallest period p in {1, 2, 4} that has them (p > 1: interleaved lanes, e.g. the low and high
// nibbles of packed FP4 bytes). Each residue class's template is proposed from its first two lanes
// (lane_anti_unify) and verified by instantiating it at every lane of the class and comparing Arena
// identity (exact: then T_r(m) = V_(p m + r) for every class index m of the range). Input reads, storage
// bits and bit patterns of a template keep their facts and bounds at a symbolic index (rebuild_node), so
// every lane's instance must carry exactly the same ones. The instances at the lanes are found, never
// made (rebuild_node's existing_reads): a candidate that fails leaves no read of an element the
// importer has not read. Memoized per record; O(template x lanes) for each period tried.
inline auto Importer::record_lane_template(Id record,const std::vector<Id>& lanes,unsigned w,const Z& k0,const Z& k1) -> LaneTemplate {
  const auto key=std::tuple(record,w,k0,k1);
  if(auto it=record_lane_templates.find(key);it!=record_lane_templates.end())return it->second;
  const Z count=k1-k0+1;
  LaneTemplate found;
  for(unsigned p=1;p<=4 && found.period==0 && Z(2*p)<=count;p*=2) {
    const Id x=a.var("record-lane:"+std::to_string(record)+":"+std::to_string(w)+":"+std::to_string(p),
                     Q(floorz(k0,Z(p))),Q(floorz(k1,Z(p))));
    std::vector<Id> templates(p,invalid);
    bool ok=true;
    for(unsigned r=0;r<p && ok;++r) {
      std::vector<Z> ks;
      for(Z k=k0;k<=k1;++k)if(modz(k,Z(p))==Z(r))ks.push_back(k);
      if(ks.size()<2){ok=false;break;}
      std::map<std::pair<Id,Id>,Id> memo;
      Id T=lane_anti_unify(lanes[ks[0].convert_to<size_t>()],lanes[ks[1].convert_to<size_t>()],x,floorz(ks[0],Z(p)),memo);
      for(size_t i=0;T!=invalid && i<ks.size();++i)
        if(instantiate(T,x,a.constant(Q(floorz(ks[i],Z(p)))),true)!=lanes[ks[i].convert_to<size_t>()])T=invalid;
      if(T!=invalid) {
        std::vector<Id> work{T},leaves;std::unordered_set<Id> seen;
        while(!work.empty()) {
          Id y=work.back();work.pop_back();
          if(!a.depends(y,x) || !seen.insert(y).second)continue;
          const auto& op=a.nodes[y].op;
          if(op=="read" || op=="readbits" || op=="floatbits")leaves.push_back(y);
          for(Id z:a.nodes[y].args)work.push_back(z);
        }
        for(Id leaf:leaves)
          for(size_t i=0;T!=invalid && i<ks.size();++i) {
            const Id instance=instantiate(leaf,x,a.constant(Q(floorz(ks[i],Z(p)))),true);
            const auto& f=a.nodes[leaf].facts;
            if(instance==invalid || a.nodes[instance].bounds!=a.nodes[leaf].bounds ||
               a.nodes[instance].facts.finite!=f.finite || a.nodes[instance].facts.pos!=f.pos ||
               a.nodes[instance].facts.nn!=f.nn || a.nodes[instance].facts.finite_or_neginf!=f.finite_or_neginf)T=invalid;
          }
      }
      ok=T!=invalid;templates[r]=T;
    }
    if(ok)found=LaneTemplate{p,std::move(templates),x};
  }
  ++a.stats[found.period?"record_lane_templates_verified":"record_lane_templates_refused"];
  if(found.period>1)++a.stats["record_lane_interleaved_templates"];
  return record_lane_templates.emplace(key,std::move(found)).first->second;
}

// A symbolic lane of a packed record of same-type floating fields, as a bitcast to that type: bits
// [lo, lo + w) of word_shr(R, s) (unsigned, record width W), with R = sum over lanes k of
// 2^(w k) * floatbits_w(V_k) for one floating type, s = w*j' a proven multiple of w and lane
// j = j' + lo/w in [k0, k1] with s + lo + w <= W (so shr's clamp is never reached). The bits are lane
// j's field, i.e. the value V_j = T_(j mod p)(j div p) for the verified lane templates (record_lane_template;
// a selection on j mod p when p > 1). Example:
// an MMA operand read at a dynamic 16-bit offset of a 128-bit record of eight bf16 values. O(template
// size x lanes) once per record, then one instantiation per use.
inline auto Importer::dynamic_record_lane(Id x,const std::string& dst,unsigned w) -> std::optional<Id> {
  if(!w || w>64)return {};
  const auto& n=a.nodes[x];
  Id shifted=invalid;unsigned lo=0;
  if(n.op=="extractbits" && n.args.size()==1 && n.data.size()==2 && n.data[0].denominator()==1 &&
     n.data[1]==Q(w) && n.data[0]>=0 && n.data[0]<=256) {shifted=n.args[0];lo=n.data[0].numerator().convert_to<unsigned>();}
  else if(n.op=="word" && n.args.size()==1 && n.data.size()==1 && n.data[0]==Q(w))shifted=n.args[0];
  else if(n.op=="imod" && n.args.size()==1 && n.data.size()==1 && n.data[0]==Q(pow2(w)))shifted=n.args[0];
  if(shifted==invalid || lo%w)return {};
  const auto& sh=a.nodes[shifted];
  if(sh.op!="word_shr" || sh.args.size()!=2 || sh.data.size()!=2 || sh.data[1]!=0 || sh.data[0].denominator()!=1)return {};
  const Id record=sh.args[0],shift=sh.args[1];
  const Z W=sh.data[0].numerator();
  if(W%w)return {};
  const auto& r=a.nodes[record];
  if(r.op!="add" || r.args.size()!=r.data.size() || r.args.empty())return {};
  const size_t L=(W/w).convert_to<size_t>();
  std::vector<Id> lanes(L,invalid);
  for(size_t i=0;i<r.args.size();++i) {
    const auto shift_bits=PackedDecodePass::power_of_two(r.data[i]);
    const auto& f=a.nodes[r.args[i]];
    if(!shift_bits || *shift_bits%w || *shift_bits/w>=L || f.op!="floatbits" || f.tag!=dst || f.args.size()!=1 ||
       f.data.size()!=1 || f.data[0]!=Q(w) || lanes[*shift_bits/w]!=invalid)return {};
    lanes[*shift_bits/w]=f.args[0];
  }
  // The lane index: s must be a proven multiple of w.
  const auto& sb=a.nodes[shift].bounds;
  if(!sb || sb->first<0 || !a.nodes[shift].facts.finite || !packed_decoders.integer(shift))return {};
  if(a.imod(shift,Z(w))!=a.zero)return {};
  const Id lane=a.add({a.idiv(shift,Z(w)),a.constant(Q(lo/w))});
  const auto& lb=a.nodes[lane].bounds;
  if(!lb || lb->first.denominator()!=1 || lb->second.denominator()!=1)return {};
  const Z k0=lb->first.numerator(),k1=lb->second.numerator();
  if(k0<0 || k1>=Z(L) || (k1+1)*w>W)return {};
  for(Z k=k0;k<=k1;++k)if(lanes[k.convert_to<size_t>()]==invalid)return {};
  if(k0==k1)return lanes[k0.convert_to<size_t>()];
  const LaneTemplate lane_template=record_lane_template(record,lanes,w,k0,k1);
  const unsigned p=lane_template.period;
  if(!p)return {};
  // Lane j is T_(j mod p)(j div p): a selection over the residue for p > 1.
  const Id index=p==1?lane:a.idiv(lane,Z(p));
  Id value=instantiate(lane_template.templates[p-1],lane_template.index,index);
  for(unsigned r=p-1;value!=invalid && r-->0;) {
    const Id arm=instantiate(lane_template.templates[r],lane_template.index,index);
    value=arm==invalid?invalid:a.select(a.compare("eq",a.imod(lane,Z(p)),a.constant(Q(r))),arm,value);
  }
  if(value==invalid)return {};
  ++a.stats["dynamic_record_lanes"];
  return value;
}

} // namespace rg
