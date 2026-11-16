#pragma once
#include "importer.hpp"

namespace rg {

inline auto Importer::cast(Id x, const std::string &src, const std::string &dst,
        bool bitcast) -> Id {
  if (src == dst)
    return x;
  if((src=="e8m0" || src=="ue8m0" || dst=="e8m0" || dst=="ue8m0") && bitcast) {
    if((dst=="e8m0" || dst=="ue8m0") && (src=="u8" || src=="b8"))
      if(auto value=premise_e8m0_byte(x)) {
        ++a.stats["premise_e8m0_byte_bitcasts"];
        return *value;
      }
    throw Unsupported("raw E8M0 bitcast requires format/domain evidence");
  }
  if(dst=="e8m0" || dst=="ue8m0")throw Unsupported("E8M0 construction outside typed-input interpretation");
  // Typed input numerical interpretation is supported above.  A raw byte
  // bitcast has no established UE4M3 padding/NaN domain, and constructing new
  // UE4M3 values would require the specified quantization semantics.
  if ((src == "ue4m3" || dst == "ue4m3") && bitcast)
    throw Unsupported("raw UE4M3 bitcast requires explicit format/domain evidence");
  if (dst == "ue4m3")
    throw Unsupported("UE4M3 value construction is outside typed-input interpretation");
  if(src=="pred" && !bitcast &&
     ((floating(dst) && ideal) || dst.starts_with("u") || dst.starts_with("s"))) {
    auto bounds=a.nodes[x].bounds;
    if(!bounds || bounds->first<0 || bounds->second>1)
      throw Unsupported("predicate conversion outside Boolean range");
    if(!floating(dst) && width(dst)<(dst.starts_with("s")?2u:1u))
      throw Unsupported("predicate conversion destination cannot represent one");
    return x;
  }
  unsigned sw = width(src), dw = width(dst);
  if (bitcast && sw != dw)
    throw Unsupported("bitcast changes width");
  if (floating(src) && floating(dst) && !bitcast) {
    if (!ideal)
      throw Unsupported("floating convert without ideal_real");
    return x;
  }
  if (floating(dst) && bitcast) {
    if(src=="b32" && dst=="f32")
      if(auto found=power_bit_values.find(x);found!=power_bit_values.end()) {
        ++a.stats["fp_idiom_numeric_replacements"];
        return found->second;
      }
    const auto &n = a.nodes[x];
    // A floating source holds a real value, not its bits: every rule below would read the value
    // (or its bounds) as a bit pattern. Only zero has known bits (all zero, or only the sign set),
    // which read as zero in every IEEE-style format; any other value is refused.
    if (floating(src)) {
      if (n.op == "const" && n.data[0] == 0) return a.zero;
      throw Unsupported("bitcast to floating is not a numeric conversion: " + src + " -> " + dst);
    }
    if(dst=="e2m1" && src=="b4") {
      if(!ideal)throw Unsupported("FP4 numerical interpretation requires ideal_real");
      auto bounds=n.bounds;
      if(!n.facts.finite || !bounds || bounds->first<0 || bounds->second>15 ||
         (n.op=="const" && n.data[0].denominator()!=1))
        throw Unsupported("FP4 payload outside four-bit storage");
      ++a.stats["exact_e2m1_decodes"];
      if(auto it=typed_fp4_values.find(x);it!=typed_fp4_values.end()) {
        ++a.stats["typed_e2m1_decode_cache_hits"];
        return it->second;
      }
      // The explicit b4->e2m1 edge already defines the sixteen-value decoder.
      // Transport only proves which four stored bits it consumes, including
      // full original-load extent and canonical layout checks in the mapper.
      auto mapper=[&](Id source,unsigned lo,unsigned count) {
        return canonical_packed_slice(source,lo,count);
      };
      if(auto slice=fp4_boundaries.transport(x,mapper);slice && slice->count==4)
        if(auto bits=fp4_boundaries.resolve(*slice,mapper);bits && bits->count==4) {
          Id payload=a.intern("packedbits",{bits->address},{Q(4)},bits->allocation,
                              {true,false,true},std::pair(Q(0),Q(15)));
          Id decoded=a.intern("decode_fp4",{payload},{},"",{true,false,false},
                              std::pair(Q(-6),Q(6)));
          typed_fp4_values.emplace(x,decoded);
          ++a.stats["typed_e2m1_canonical_decodes"];
          return decoded;
        }
      // Non-storage payloads and unresolved provenance retain the existing
      // exact arithmetic interpretation; no storage identity is invented.
      Id fraction=a.imod(x,2), exponent=a.imod(a.idiv(x,2),4);
      Id half=a.mul({a.constant(Q(1,2)),fraction});
      Id scale=a.select(a.compare("eq",exponent,a.one),a.one,
          a.select(a.compare("eq",exponent,a.constant(2)),a.constant(2),a.constant(4)));
      Id magnitude=a.select(a.compare("eq",exponent,a.zero),half,
          a.mul({scale,a.add({a.one,half})}));
      Id decoded=a.select(a.compare("ge",x,a.constant(8)),a.neg(magnitude),magnitude);
      typed_fp4_values.emplace(x,decoded);
      return decoded;
    }
    // Decode IEEE binary32 bits, never numerically cast the integer payload.
    // Fixed-width work: no traversal of the producer DAG or host FP rounding.
    if (dst == "f32" && (src == "b32" || src == "u32" || src == "s32") &&
        n.op == "const") {
      const Q &value = n.data[0];
      Z lower = src == "s32" ? -pow2(31) : Z(0);
      Z upper = src == "s32" ? pow2(31) : pow2(32);
      if (value.denominator() != 1 || value.numerator() < lower ||
          value.numerator() >= upper)
        throw Unsupported(
            "binary32 bitcast payload outside source integer range");
      uint32_t bits =
          modz(value.numerator(), pow2(32)).convert_to<uint32_t>();
      if (((bits >> 23) & 255u) == 255u) {
        if((bits & 0x7fffffu)==0)return a.infinity((bits>>31)?-1:1);
        throw Unsupported("NaN binary32 constant bitcast");
      }
      if (bits == 0x80000000u)
        throw Unsupported("negative-zero binary32 constant bitcast requires "
                          "signed-zero semantics");
      ++a.stats["constant_binary32_bitcasts"];
      return a.constant(ieee_value(bits, 32, 23));
    }
    // Exact binary16/BF16/FP8 literal decoding, including reduction sentinels.
    // A numeric cast of the integer payload would be incorrect. E5M2 has
    // IEEE infinities and NaNs; E4M3 has no infinity and one NaN pattern
    // per sign (S.1111.111), its other all-ones-exponent codes are finite.
    if((((dst=="f16" || dst=="bf16") && sw==16) || ((dst=="e5m2" || dst=="e4m3") && sw==8)) && n.op=="const") {
      const auto value=n.data[0];const bool signed_source=src.starts_with("s");
      const Z range=pow2(sw),half=range/2;
      const Z lowest=signed_source?Z(-half):Z(0),highest=signed_source?Z(half-1):Z(range-1);
      if(value.denominator()!=1 || value<Q(lowest) || value>Q(highest))
        throw Unsupported(std::to_string(sw)+"-bit floating payload outside source range");
      const unsigned bits=modz(value.numerator(),range).convert_to<unsigned>();
      const unsigned fraction_bits=dst=="f16"?10:dst=="bf16"?7:dst=="e5m2"?2:3;
      const unsigned max_exponent=(1u<<(sw-1-fraction_bits))-1;
      const unsigned exponent=(bits>>fraction_bits)&max_exponent, fraction=bits&((1u<<fraction_bits)-1);
      const unsigned sign=bits>>(sw-1);
      if(dst=="e4m3" ? exponent==max_exponent && fraction==(1u<<fraction_bits)-1 : exponent==max_exponent) {
        if(dst!="e4m3" && fraction==0)return a.infinity(sign?-1:1);
        throw Unsupported("NaN "+std::to_string(sw)+"-bit floating constant bitcast");
      }
      if(bits==(1u<<(sw-1)))
        throw Unsupported("negative-zero "+std::to_string(sw)+"-bit floating constant bitcast requires signed-zero semantics");
      return a.constant(ieee_value(bits,sw,fraction_bits));
    }
    if (n.op == "floatbits" && n.tag == dst && n.data[0] == Q(dw))
      return n.args[0];
    if (n.op == "readbits" && n.data[0] == Q(dw))
      return typed_read(n.tag, n.args[0], dst);
    if (n.op == "extractbits" && modz(n.data[0].numerator(), 8) == 0 &&
        n.data[1] == Q(dw)) {
      const auto &p = a.nodes[n.args[0]];
      if (p.op == "readbits")
        return typed_read(
            p.tag, a.add({p.args[0], a.constant(n.data[0] / Q(8))}), dst);
    }
    // Packed BF16 lanes may first be sliced from a 64-bit load and shifted
    // under a 32-bit truncation. Recover exactly the contributing 16 bits.
    if(dst=="f32" && sw==32) {
      bool truncated=n.op=="word" && n.data[0]==Q(32);
      const auto &expanded=a.nodes[truncated?n.args[0]:x];
      if(expanded.op=="mul" && expanded.args.size()==1 && expanded.data[0]==Q(65536)) {
        Id bits=expanded.args[0];auto bound=a.nodes[bits].bounds;
        if(truncated || (bound && bound->first>=0 && bound->second<65536))
          if(auto slice=initial_bit_slice(bits,0,16)) {
            ++a.stats["packed_bf16_lane_recoveries"];
            return cast(*slice,"u16","bf16",true);
          }
      }
    }
    // BF16 register expansion: zero-extend its bits and shift into f32's
    // high half. Recover the original numeric BF16 input, not an integer cast.
    if (dst == "f32" && sw == 32 && n.op == "mul" &&
        n.args.size() == 1 && n.data[0] == Q(65536)) {
      Id bits = n.args[0];
      auto bound = a.nodes[bits].bounds;
      if (bound && bound->first >= 0 && bound->second < 65536)
        return cast(bits, "u16", "bf16", true);
    }
    // A BF16 power of two built in its exponent field, before the field decoders below
    // (they give the same value with an explicit fraction factor). f32/f16: at the end.
    if(dst=="bf16")if(auto power=exponent_field_power(x,src,dst,false))return *power;
    // Constructed BF16 values with fixed sign/exponent are affine in their
    // seven fraction bits. No approximate host float or instance enumeration.
    if (dst == "bf16")
      if (auto field = fixed_header_float(x, sw, dst)) return *field;
    if (dst == "bf16" && sw == 16 && n.bounds) {
      auto bound = *n.bounds;
      bool positive = bound.first >= 128 && bound.second < 32640;
      bool negative = bound.first >= 32896 && bound.second < 65408;
      if (positive || negative) {
        // Interval facts may cross an exponent boundary. Decode the field
        // exactly instead of assuming its exponent is constant.
        Id header = a.idiv(x, Z(128));
        Id exponent = a.sub(header, a.constant(negative ? 390 : 134));
        Id significand = a.add({a.constant(128), a.imod(x, Z(128))});
        return a.mul({a.constant(negative ? -1 : 1), significand, a.exp2(exponent)});
      }
    }
    if(auto lane=floating_lane(x,dst))return *lane;
    // A bitcast of a selection is the selection of the alternatives'
    // bitcasts (both are bit patterns of the source type). Memoized per
    // selection node: O(distinct selections), depth-bounded.
    if(n.op=="select" && n.args.size()==3) {
      const auto key=std::tuple(x,src,dst);
      if(auto it=select_bitcasts.find(key);it!=select_bitcasts.end())return it->second;
      if(select_bitcast_depth>=select_bitcast_depth_limit)
        throw Unsupported("bitcast of nested selections exceeds depth limit");
      const Id p=n.args[0],t=n.args[1],f=n.args[2];
      ++select_bitcast_depth;
      struct Depth {unsigned& n;~Depth(){--n;}} depth{select_bitcast_depth};
      const Id yes=cast(t,src,dst,true),no=cast(f,src,dst,true);
      ++a.stats["select_floating_bitcasts"];
      return select_bitcasts.emplace(key,a.select(p,yes,no)).first->second;
    }
    // f16 and f32 fields, after the lane and selection rules. A bf16 field already took the same rule
    // above, before the bf16 range rule; it is not counted in fixed_header_float_bitcasts.
    if(dst!="bf16")
      if(auto field=fixed_header_float(x,sw,dst)) {
        ++a.stats["fixed_header_float_bitcasts"];
        return *field;
      }
    if(auto power=exponent_field_power(x,src,dst,true))return *power;
    // Last, so no rule above changes: a bit slice of a selection, then the selection rule.
    if(auto selection=sliced_selection(x,dw)) {
      ++a.stats["sliced_selection_bitcasts"];
      return cast(*selection,src,dst,true);
    }
    // After every rule above: a symbolic lane of a packed record of same-type floating fields.
    if(auto value=dynamic_record_lane(x,dst,dw))return *value;
    throw Unsupported("bitcast to floating is not a numeric conversion: " +
                      src + " -> " + dst);
  }
  if (floating(src)) {
    if (bitcast && sw == dw)
      return a.intern("floatbits", {x}, {Q(sw)}, src, {true, false, true},
                      std::pair(Q(0), Q(pow2(sw) - 1)));
    throw Unsupported("float to integer/bit conversion: " + src + " -> " +
                      dst);
  }
  if (floating(dst)) {
    if (!ideal)
      throw Unsupported("integer to float exact mode");
    return x;
  }
  if (dw < sw) {
    const auto &n = a.nodes[x];
    if (n.op == "readbits")
      x = a.bit_slice(x,0,dw);
    else if (n.op == "extractbits" && n.data[1] >= Q(dw))
      x = a.intern("extractbits", n.args, {n.data[0], Q(dw)}, "",
                   {true, false, true}, std::pair(Q(0), Q(pow2(dw) - 1)));
    else
      x = word(x, dw);
  }
  if (dst.starts_with("s")) {
    if (auto read=initial_typed_bits(x,dst)) {
      ++a.stats["signed_input_transport_reads"];
      return *read;
    }
    return a.signed_word(x, dw);
  }
  return word(x, dw);
}

inline auto Importer::edge_conversion(Id x, const std::string &conv, bool output) -> Id {
  if (conv.empty())
    return x;
  static const std::regex re(
      R"(\s*(Bitcast|Convert|NumericInterpretation)([^()]*)\(\s*([^ ()]+)\s*->\s*([^ ()]+)\s*\)\s*)");
  std::stringstream ss(conv);
  std::string part;
  // The fold packs byte parts into one word with ZeroExtend (indexed scalar packing): the source's
  // unsigned word, embedded unchanged. Exact when the destination holds 2^K - 1 (unsigned or bit of at
  // least K bits, signed of more); a signed source or a narrowing is refused.
  static const std::regex zero_extend(R"(\s*ZeroExtend\(\s*([bu][0-9]+)\s*->\s*([bus][0-9]+)\s*\)\s*)");
  while (std::getline(ss, part, ';')) {
    std::smatch m;
    if (std::regex_match(part, m, zero_extend)) {
      const std::string src = m[1], dst = m[2];
      const unsigned sw = width(src), dw = width(dst);
      if (dw < sw || (dst[0] == 's' && dw == sw))
        throw Unsupported("unsupported zero extension: " + part);
      x = word(x, sw);
      ++a.stats["zero_extensions"];
      continue;
    }
    if (!std::regex_match(part, m, re))
      throw Unsupported("unparsed edge conversion: " + part);
    std::string attr = m[2], src = m[3], dst = m[4];
    // An e2m1 satfinite result read as its four bits is encoded exactly (e2m1_bits). The encoding
    // is total in its input, NaN and infinities included, so it needs no finiteness proof. Any
    // other use of the result keeps the saturation path below.
    if (ideal && attr == " rounding=nearest-even saturate_finite " && src == "f32" && dst == "e2m1") {
      const auto next = ss.tellg();
      std::string bits;
      std::smatch b;
      if (std::getline(ss, bits, ';') && std::regex_match(bits, b, re) && b[1] == "Bitcast" &&
          std::string(b[2]).find_first_not_of(' ') == std::string::npos && b[3] == "e2m1" && b[4] == "u4") {
        x = e2m1_bits(x);
        continue;
      }
      ss.clear();
      ss.seekg(next);
    }
    if (attr.find("saturat") != attr.npos &&
        !(output && floating(src) && floating(dst) &&
          mapping.value("output_value_semantics", "") == "ideal_real")) {
      if (!ideal || !floating(src) || !floating(dst))
        throw Unsupported("Saturation requires supported ideal-real floating semantics",
                          "saturation_semantics_unsupported",
                          {{"ideal_real",ideal},{"source_type",src},{"target_type",dst},{"conversion",part}});
      // cvt .sat limits the result to [0.0, 1.0] (PTX ISA, cvt): -inf becomes 0.
      if (attr == " saturate " && a.neginf(x)) {
        ++a.stats["saturated_negative_infinities"];
        x = cast(a.zero, src, dst, m[1] == "Bitcast");
        continue;
      }
      if (!a.nodes[x].facts.finite)
        throw Unsupported("Could not prove that the saturation input is finite",
                          "saturation_input_finiteness_not_proven",
                          {{"input_expression",a.render(x,3)},{"source_type",src},{"target_type",dst},{"conversion",part}});
      Q lo, hi;
      if (attr == " saturate ") { lo = Q(0); hi = Q(1); }
      else if (attr == " rounding=nearest-even saturate_finite " && dst == "e4m3") {
        lo = Q(-448); hi = Q(448);
      } else throw Unsupported("unsupported intermediate saturation mode: " + part);
      x = a.neg(a.maximum({a.neg(a.maximum({x,a.constant(lo)})), a.constant(-hi)}));
    }
    x = cast(x, src, dst, m[1] == "Bitcast");
  }
  return x;
}

// The four bits of cvt.rn.satfinite.e2m1 of a binary32 y, for every y. E2M1 has a sign, two
// exponent bits and one mantissa bit (magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6) and no infinity or
// NaN. NaN gives +MAXNORM, 0b0111. Otherwise bit 3 is the sign bit of y's binary32 encoding (also
// for results that round to zero and for saturated ones), and bits 0..2 are the code of |y|
// rounded to nearest even, saturated to MAXNORM 6 above. Of the midpoints, 0.25, 1.25, 2.5 and 5
// round down to the even code, 0.75, 1.75 and 3.5 round up to it. O(1).
inline auto Importer::e2m1_bits(Id y) -> Id {
  struct Midpoint { Q value; bool rounds_up; };
  static const Midpoint midpoints[] = {{Q(1, 4), false}, {Q(3, 4), true}, {Q(5, 4), false},
                                       {Q(7, 4), true}, {Q(5, 2), false}, {Q(7, 2), true},
                                       {Q(5), false}};
  const Id magnitude = a.absolute(y);
  Id code = a.zero;
  for (unsigned k = 0; k < std::size(midpoints); ++k)
    code = a.select(a.compare(midpoints[k].rounds_up ? "ge" : "gt", magnitude,
                              a.constant(midpoints[k].value)),
                    a.constant(Q(k + 1)), code, true);
  const Id word = cast(y, "f32", "u32", true);
  const Id sign = a.compare("ne", a.mul({a.constant(Q(pow2(31))), a.imod(a.idiv(word, pow2(31)), 2)}),
                            a.zero);
  ++a.stats["e2m1_satfinite_encodings"];
  const Id bits = a.select(a.compare("ne", y, y), a.constant(Q(7)),
                           a.add({code, a.mul({sign, a.constant(Q(8))})}), true);
  // Integers for every y (the code and sign also surface as terms of a packing sum once a finite
  // y removes the NaN selection): word() may reduce their packing coefficients.
  packed_decoders.known_integers.insert({code, sign, bits});
  return bits;
}

// f16, bf16 or f32 bits whose bounds fix the sign and a normal exponent field (e.g. the uint4
// dequantization idiom (w & 0x000F000F) | 0x64006400) are affine in the fraction bits:
// +-2^(e-bias-f) (bits - header*2^f + 2^f). O(1).
inline auto Importer::fixed_header_float(Id x, unsigned w, const std::string &dst) -> std::optional<Id> {
  unsigned fraction = 0, exponent_bits = 0;
  if (dst == "f16" && w == 16) { fraction = 10; exponent_bits = 5; }
  else if (dst == "bf16" && w == 16) { fraction = 7; exponent_bits = 8; }
  else if (dst == "f32" && w == 32) { fraction = 23; exponent_bits = 8; }
  else return std::nullopt;
  const auto &bound = a.nodes[x].bounds;
  const Q unit = Q(pow2(fraction));
  if (!bound || bound->first < 0 || bound->second >= Q(pow2(w)) ||
      floorq(bound->first / unit) != floorq(bound->second / unit)) return std::nullopt;
  const Z header = floorq(bound->first / unit);
  const Z exponent = modz(header, pow2(exponent_bits));
  if (exponent == 0 || exponent == pow2(exponent_bits) - 1) return std::nullopt;
  const int bias = (1 << (exponent_bits - 1)) - 1;
  const int shift = exponent.convert_to<int>() - bias - int(fraction);
  const Q scale = shift >= 0 ? Q(pow2(unsigned(shift))) : Q(1, pow2(unsigned(-shift)));
  Id significand = a.add({x, a.constant(unit - Q(header) * unit)});
  return a.mul({a.constant(header >= pow2(exponent_bits) ? -scale : scale), significand});
}

// An integer word read as f32, f16 or bf16 whose fraction field is zero and whose biased exponent field is
// proved inside the normal range [1, 2^E - 2] is exactly 2^(field - bias): the exponent built by integer
// arithmetic, e.g. (e + 127) << 23. The word's bits are its residue modulo 2^width (a signed source is the
// two's-complement reading of its word). Fraction and field come from the arena's exact residue and quotient
// and their interval bounds. With `refuse`, a zero fraction whose field is not proved normal (zero,
// subnormal, infinity, NaN or a negative sign) is refused precisely; otherwise none is returned and the
// other bitcast rules apply. O(affine terms of the word).
inline auto Importer::exponent_field_power(Id x, const std::string &src, const std::string &dst, bool refuse) -> std::optional<Id> {
  const unsigned w = width(src);
  unsigned fraction_bits = 0, bias = 0;
  if (dst == "f32" && w == 32) { fraction_bits = 23; bias = 127; }
  else if (dst == "f16" && w == 16) { fraction_bits = 10; bias = 15; }
  else if (dst == "bf16" && w == 16) { fraction_bits = 7; bias = 127; }
  else return std::nullopt;
  if (!a.known_bits(x).integer) return std::nullopt;
  const auto &n = a.nodes[x];
  Id word = n.op == "signed" && n.args.size() == 1 && n.data.size() == 1 && n.data[0] == Q(w) ? n.args[0] : x;
  Id bits = a.imod(word, pow2(w));
  if (a.imod(bits, pow2(fraction_bits)) != a.zero) return std::nullopt;
  Id field = a.idiv(bits, pow2(fraction_bits));
  const Z top = pow2(w - 1 - fraction_bits) - 2;
  const auto &range = a.nodes[field].bounds;
  if (!range || range->first < Q(1) || range->second > Q(top)) {
    if (!refuse) return std::nullopt;
    throw Unsupported("integer-built " + dst + " exponent field is not proved in the normal range [1," +
                      top.str() + "]" + (range ? " (bounds " + str(range->first) + ".." + str(range->second) + ")" : ""));
  }
  ++a.stats["exponent_field_powers"];
  return a.exp2(a.sub(field, a.constant(Q(bias))));
}

inline auto Importer::literal(const std::string &s, const std::string &t) -> Id {
  if (s == "-inf" || s == "-infinity" || s == "-Infinity")
    return a.infinity(-1);
  if (s == "inf" || s == "+inf" || s == "infinity" || s == "Infinity")
    return a.infinity(1);
  if (t == "pred" && (s == "true" || s == "false"))
    return s == "true" ? a.one : a.zero;
  try {
    Q q;
    // A binary32/binary64 literal denotes its nearest IEEE value (as a host float literal does),
    // not the decimal number spelled. floatFromChars rounds once, directly to the literal's type,
    // and accepts an exact subnormal; only overflow and underflow to zero are out of range. O(digits).
    const auto binary = [&](auto parsed, unsigned fraction) {
      decltype(parsed) f = 0;
      const auto r = sentryward::equivalence::floatFromChars(s.data(), s.data() + s.size(), f);
      if (r.ec != std::errc{} || r.ptr != s.data() + s.size() || !std::isfinite(f))
        throw Unsupported("constant " + s + " type " + t);
      using Word = std::conditional_t<sizeof(f) == 4, uint32_t, uint64_t>;
      return ieee_value(std::bit_cast<Word>(f), 8 * sizeof(f), fraction);
    };
    if (t == "f32") q = binary(float{}, 23);
    else if (t == "f64") q = binary(double{}, 52);
    else if (s.starts_with("0x") || s.starts_with("0X"))
      q = Q(Z(s));
    else
      q = decimal(s);
    // PTX reads an integer constant used as a predicate as C does: zero is false, nonzero true
    // (ISA, Predicate Constants), so `mov.pred %p, -1` is true, not the number -1.
    if (t == "pred")
      return q == 0 ? a.zero : a.one;
    if (!floating(t)) {
      Z v = modz(q.numerator() / q.denominator(), pow2(width(t)));
      if (t.starts_with("s") && v >= pow2(width(t) - 1))
        v -= pow2(width(t));
      q = Q(v);
    }
    return a.constant(q);
  } catch (const Unsupported &) {
    throw;
  } catch (...) {
    throw Unsupported("constant " + s + " type " + t);
  }
}

} // namespace rg
