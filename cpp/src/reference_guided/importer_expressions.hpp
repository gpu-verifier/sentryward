#pragma once
#include "importer.hpp"

namespace rg {

// At a Maximum (Minimum), an operand imported as the type's most negative (largest) finite value
// -MAX (MAX) never decides it in hardware when another operand's hardware value set lies at or above
// -MAX (at or below MAX); it is dropped, without the range premise of in_range. Records
// range_bounds_proved. O(operands) besides the memoized sets.
inline auto Importer::proved_in_range(std::vector<Id> x, const RawNode &n, bool maximum, const Env &env) -> std::vector<Id> {
  const auto format = float_format(n.type);
  if (!hardware_values || !format || x.size() != n.operands.size()) return x;
  const Q bound = maximum ? -largest_finite(*format) : largest_finite(*format);
  std::vector<uint8_t> at_bound(x.size());
  for (size_t j = 0; j < x.size(); ++j)
    if (const auto v = a.value(x[j]); v && *v == bound) at_bound[j] = 1;
  if (std::none_of(at_bound.begin(), at_bound.end(), [](uint8_t b) { return b; })) return x;
  bool dominated = false;
  const Extended limit{0, bound};
  for (size_t j = 0; j < x.size() && !dominated; ++j) {
    if (at_bound[j]) continue;
    const auto s = edge_set(n.operands[j], &env, 0);
    dominated = !s.nan && (maximum ? !before(s.lo, limit) : !before(limit, s.hi));
  }
  if (!dominated) return x;
  std::vector<Id> kept;
  for (size_t j = 0; j < x.size(); ++j)
    if (!at_bound[j]) kept.push_back(x[j]);
  a.stats["range_bounds_proved"] += x.size() - kept.size();
  return kept;
}

// A comparison of a Select of two constants with a constant, decided arm by arm: the Select's
// Boolean condition, its complement 1 - p, or a constant. This is a predicate carried through an
// integer (selp 1, 0, then setp.ne 0) or any other constant pair. The Select may sit under
// shells that apply to each arm (Arena::selected_constants: a sum with constants, a scaling, a
// word, a max with constants, a clamp), e.g. word(select(q, 0, 111) - 1) <= 62. Exact.
// O(size of the shells, at most Arena::guard_region_depth_limit levels).
inline auto Importer::selected_comparison(const std::string &op, Id x, Id y) -> std::optional<Id> {
  if (!Arena::valid_comparison(op)) return {};
  for (unsigned side = 0; side != 2; ++side) {
    const Id chosen = side ? y : x;
    const auto other = a.value(side ? x : y);
    if (!other) continue;
    const auto arms = a.selected_constants(chosen);
    if (!arms) {
      // One constant arm, decided by the Select's own condition: [select(c, A, K) op v] for a Boolean
      // c, a constant K and a finite A is [A op v] where c holds and the constant truth d of [K op v]
      // elsewhere. When c implies [A op v] (Arena::boolean_implies) that is d ? 1 : c; when c implies
      // its negation, d ? not c : 0 (not x is [x == 0]). With K on the true arm it is c ? d : [A op v]:
      // d ? 1 : not c when not c implies [A op v], and d ? c : 0 when not c implies its negation.
      // Every other case keeps its Select. Inspects this one Select. Exact.
      const auto chosen_node = a.nodes[chosen];
      if (chosen_node.op != "select" || chosen_node.args.size() != 3 || !a.boolean_node(chosen_node.args[0])) continue;
      const auto yes_constant = a.value(chosen_node.args[1]), no_constant = a.value(chosen_node.args[2]);
      if (bool(yes_constant) == bool(no_constant)) continue;
      const bool constant_when_true = bool(yes_constant);
      const Q k = constant_when_true ? *yes_constant : *no_constant;
      const Id open = chosen_node.args[constant_when_true ? 2 : 1], condition = chosen_node.args[0];
      if (!a.nodes[open].facts.finite) continue;
      const Q &l = side ? *other : k, &r = side ? k : *other;
      const bool decided = op == "eq" ? l == r : op == "ne" ? l != r : op == "lt" ? l < r :
                           op == "le" ? l <= r : op == "gt" ? l > r : l >= r;
      const Id partner = side ? x : y;
      const Id inner = side ? a.compare(op, partner, open) : a.compare(op, open, partner);
      const auto negate = [&](Id b) { return a.compare("eq", b, a.zero); };
      Id result = invalid;
      if (!constant_when_true) {
        if (inner == a.one || a.boolean_implies(condition, inner)) result = decided ? a.one : condition;
        else if (inner == a.zero || a.boolean_implies(condition, negate(inner))) result = decided ? negate(condition) : a.zero;
      } else {
        const Id not_condition = negate(condition);
        if (inner == a.one || a.boolean_implies(not_condition, inner)) result = decided ? a.one : not_condition;
        else if (inner == a.zero || a.boolean_implies(not_condition, negate(inner))) result = decided ? condition : a.zero;
      }
      if (result == invalid) continue;
      ++a.stats["condition_decided_select_comparisons"];
      return result;
    }
    const auto &[condition, yes_value, no_value] = *arms;
    const std::optional<Q> yes = yes_value, no = no_value;
    const auto &bounds = a.nodes[condition].bounds;
    if (!bounds || bounds->first < 0 || bounds->second > 1 ||
        !a.known_bits(condition).integer) continue;
    if (a.nodes[chosen].op != "select") ++a.stats["constant_select_shell_comparisons"];
    const auto holds = [&](const Q &arm) {
      const Q &l = side ? *other : arm, &r = side ? arm : *other;
      return op == "eq" ? l == r : op == "ne" ? l != r : op == "lt" ? l < r :
             op == "le" ? l <= r : op == "gt" ? l > r : l >= r;
    };
    const bool when_true = holds(*yes), when_false = holds(*no);
    ++a.stats["constant_select_comparisons"];
    if (when_true == when_false) return when_true ? a.one : a.zero;
    if (when_true) return condition;
    // The complement of a comparison of finite values is the complementary comparison.
    const auto p = a.nodes[condition];
    if (p.op == "compare" && p.args.size() == 2 && p.data.empty() &&
        a.nodes[p.args[0]].facts.finite && a.nodes[p.args[1]].facts.finite)
      return a.compare(Arena::complement_comparison(p.tag), p.args[0], p.args[1]);
    return a.sub(a.one, condition);
  }
  return {};
}

// select(p, select(p, a, b), c) = select(p, a, c) and select(p, a, select(p, b, c)) = select(p, a, c):
// an arm that is itself a Select on the same predicate takes the branch the outer Select fixes.
// The inner Select is already canonical (Arena::select turns ne/ge/gt into eq/lt/le with swapped
// arms), so p is compared both as itself and as that canonical complement (the arms then swap).
// Only this importer Select; the Arena constructor is unchanged. Exact. O(1).
inline auto Importer::same_condition_arms(Id p, Id yes, Id no) -> std::pair<Id, Id> {
  Id complement = invalid;
  const auto &c = a.nodes[p];
  if (c.op == "compare" && c.args.size() == 2 && c.data.empty() &&
      (c.tag == "ne" || c.tag == "ge" || c.tag == "gt") &&
      a.nodes[c.args[0]].facts.finite && a.nodes[c.args[1]].facts.finite) {
    const auto args = c.args;
    complement = a.compare(c.tag == "ne" ? "eq" : c.tag == "ge" ? "lt" : "le", args[0], args[1]);
  }
  // The branch of select(q, u, v) taken when p holds (taken) or fails (!taken), if q is p or its complement.
  const auto branch = [&](Id arm, bool taken) -> Id {
    const auto &inner = a.nodes[arm];
    if (inner.op != "select" || inner.args.size() != 3) return arm;
    const Id q = inner.args[0];
    if (q == p) { ++a.stats["same_condition_select_arms"]; return inner.args[taken ? 1 : 2]; }
    if (complement != invalid && q == complement) {
      ++a.stats["same_condition_select_arms"]; return inner.args[taken ? 2 : 1];
    }
    return arm;
  };
  return {branch(yes, true), branch(no, false)};
}

// Bit b of an integer word v in [0, 2^w), exactly: a constant's bit; for a Select of two such words,
// the Select of their bits; for the w-bit complement 2^w - 1 - z of such a word z, one minus z's
// bit; for a sum of nonnegative integer terms whose possible one bits (known_bits) are pairwise
// disjoint, which is their OR, the bit of the one term (or constant) that may set it; otherwise
// the arena's bit slice. O(depth * terms) known-bits lookups.
inline auto Importer::bit_of(Id v, unsigned b, unsigned w, unsigned depth) -> Id {
  const auto in_word = [&](Id z) { return whole_word(z, w); };
  if (auto value = a.value(v); value && value->denominator() == 1 && *value >= 0)
    return a.constant(Q(modz(floorz(value->numerator(), pow2(b)), Z(2))));
  const auto n = a.nodes[v];
  if (n.op == "select" && n.args.size() == 3 && depth < bit_of_depth_limit &&
      in_word(n.args[1]) && in_word(n.args[2])) {
    const Id yes = bit_of(n.args[1], b, w, depth + 1), no = bit_of(n.args[2], b, w, depth + 1);
    return a.select(n.args[0], yes, no, true);
  }
  if (n.op != "add" || depth >= bit_of_depth_limit || n.args.size() > bit_of_sum_terms)
    return a.bit_slice(v, b, 1);
  Q offset = 0;
  std::vector<std::pair<Id, Q>> parts;
  for (const auto &[t, c] : a.terms(v)) {
    if (t == a.one) offset += c;
    else parts.emplace_back(t, c);
  }
  const auto all = [&](auto predicate) {
    return std::all_of(parts.begin(), parts.end(), [&](const auto &p) { return predicate(p.second); });
  };
  if (offset == Q(pow2(w) - 1) && all([](const Q &c) { return c < 0; })) {
    std::vector<Id> negated;
    for (const auto &[t, c] : parts) negated.push_back(a.mul({a.constant(-c), t}));
    const Id z = a.add(std::move(negated));
    if (in_word(z)) return a.sub(a.one, bit_of(z, b, w, depth + 1));
  }
  if (offset >= 0 && offset.denominator() == 1 &&
      all([](const Q &c) { return c > 0 && c.denominator() == 1; })) {
    const Z mask = pow2(w) - 1;
    Z taken = offset.numerator();
    Id owner = invalid;
    bool disjoint = taken <= mask;
    for (const auto &[t, c] : parts) {
      const Id term = c == 1 ? t : a.mul({a.constant(c), t});
      if (!in_word(term)) { disjoint = false; break; }
      const Z possible = mask ^ (a.known_bits(term).zero & mask);
      if ((possible & taken) != 0) { disjoint = false; break; }
      taken |= possible;
      if (((possible >> b) & 1) != 0) owner = term;
    }
    if (disjoint) {
      if (((offset.numerator() >> b) & 1) != 0) return a.one;
      return owner == invalid ? a.zero : bit_of(owner, b, w, depth + 1);
    }
  }
  return a.bit_slice(v, b, 1);
}

// x & y of an integer type whose operands are both non-constant: the sum over the bits b of
// 2^b * bit(x, b) * bit(y, b), exact for the two w-bit words. O(w) bit products.
inline auto Importer::bitwise_and(Id x, Id y, const std::string &t) -> std::optional<Id> {
  const unsigned w = width(t);
  if (w == 0 || w > bitwise_and_width_limit) return {};
  const Id lhs = word(x, w), rhs = word(y, w);
  std::vector<Id> terms;
  for (unsigned b = 0; b < w; ++b) {
    const Id l = bit_of(lhs, b, w);
    if (l == a.zero) continue;
    const Id r = bit_of(rhs, b, w);
    if (r == a.zero) continue;
    terms.push_back(a.mul({a.constant(Q(pow2(b))), l, r}));
  }
  ++a.stats["bitwise_and_bit_products"];
  const Id result = a.add(std::move(terms));
  return t.starts_with("s") ? cast(result, "u" + std::to_string(w), t) : result;
}

inline auto Importer::make(const RawNode &n, const std::vector<Id> &x, const Env &ctx,
        const Env &body) -> Id {
  const std::string &k = n.kind, &t = n.type, &attr = n.attributes;
  std::string name = n.name;
  if (k == "Constant")
    return literal(name, t);
  if (k == "Index") {
    Id index=ctx.at(n.level);
    packed_decoders.known_integers.insert(index);
    fp4_boundaries.known_integers.insert(index);
    return index;
  }
  if (k == "InputParameter") {
    if (m.contains("parameter_constants") &&
        m["parameter_constants"].contains(name)) {
      const auto &c = m["parameter_constants"][name];
      return literal(c["literal"], c["type"]);
    }
    return typed_integer_fact(a.intern("atom", {}, {}, mapped_name(m, "parameters", name),
                    {true, false, false}),t);
  }
  if (k == "InitialCell" || k == "InputValue") {
    if (x.size() != 1)
      throw Unsupported("input read expects one byte-offset operand");
    if(t=="e8m0" || t=="ue8m0")return typed_e8m0_input(name,x[0]);
    if (t == "ue4m3")
      return typed_ue4m3_input(name, x[0]);
    if ((t.starts_with("b") && !t.starts_with("bf")) || t.starts_with("u")) {
      if (auto fixed = fixed_bits(name, x[0], width(t))) return *fixed;
      if (auto element = contracted_word(name, x[0], width(t))) return *element;
      return a.intern("readbits", x, {Q(width(t))}, name, {true, false, true},
                      std::pair(Q(0), Q(pow2(width(t)) - 1)));
    }
    return typed_read(name, x[0], t);
  }
  if (k == "Fold") {
    const auto &f = n;
    Id var = body.at(f.level);
    Q count = Q(f.count);
    const std::string &reducer = f.reducer;
    if (reducer == "Min") {
      // Ideal-real minimum shares the maximum normal form, as Minimum does:
      // min(s, b_0, ..., b_{n-1}) = -max(-s, -b_0, ..., -b_{n-1}) exactly,
      // for integers and extended reals (the +inf identity becomes -inf).
      Id initial = a.neg(x.at(0)), element = a.neg(x.at(1));
      return a.neg(a.intern("fold", {initial, element}, {Q(var), count}, "Max",
                            a.fold_facts(initial, element, count, true, var)));
    }
    // The PTX side's artifact bypasses the document reader's reducer check;
    // every rule below interprets a non-Max fold as a sum.
    if (reducer != "Sum" && reducer != "Max")
      throw Unsupported("unsupported Fold reducer " + reducer);
    return a.intern("fold", x, {Q(var), count}, reducer,
                    a.fold_facts(x[0],x[1],count,reducer=="Max",var));
  }
  if (k != "Operation")
    throw Unsupported("node kind " + k);
  if (attr.find("saturat") != attr.npos)
    throw Unsupported("arithmetic saturation is not ideal-real identity");
  bool integer = !floating(t) && t != "pred";
  if(name=="BitNot") {
    if(!attr.empty() || x.size()!=1)throw Unsupported("BitNot attributes/arity; type="+t);
    if(t=="pred") {
      const auto bounds=a.nodes[x[0]].bounds;
      if(!bounds || bounds->first<0 || bounds->second>1)
        throw Unsupported("predicate NOT operand outside Boolean range");
      return a.sub(a.one,x[0]);
    }
    if(!integer || t.empty() || (t[0]!='s' && t[0]!='u' && t[0]!='b'))
      throw Unsupported("BitNot requires integer/bit type; type="+t);
    const unsigned w=width(t);
    const auto result=a.sub(a.constant(Q(pow2(w)-1)),word(x[0],w));
    return t.starts_with("s")?cast(result,"u"+std::to_string(w),t):result;
  }
  if (integer && (name == "Add" || name == "Subtract" || name == "Multiply" ||
                  name == "PtxMultiplyAdd" || name == "ShiftLeft")) {
    unsigned w = width(t);
    std::vector<Id> xx;
    for (Id z : x)
      xx.push_back(unwrap(z, w));
    Id y;
    if (name == "Add") {
      auto r = reconstruct(xx);
      y = r ? *r : a.add(xx);
    } else if (name == "Subtract")
      y = a.sub(xx.at(0), xx.at(1));
    else if (name == "Multiply")
      y = a.mul(xx);
    else if (name == "PtxMultiplyAdd") {
      if (attr.find(".lo.") == attr.npos)
        throw Unsupported("multiply-add mode " + attr);
      y = a.add({a.mul({xx.at(0), xx.at(1)}), xx.at(2)});
    } else if (auto s = a.value(xx.at(1)); s && s->denominator() == 1 && *s >= 0 && *s < Q(w)) {
      y = a.mul({xx[0],
                 a.constant(Q(pow2(s->numerator().convert_to<unsigned>())))});
    } else {
      // PTX shl: the count is the unsigned value of its (u32) operand; a count >= w gives 0
      // (clamped, not taken modulo w). A symbolic count is Arena::word_shl.
      if (n.operands.size() != 2) throw Unsupported("left shift arity");
      const auto &edge = n.operands[1];
      const std::string count_type = edge.converted_type.empty() ? ns.at(edge.node).type : edge.converted_type;
      if (floating(count_type)) throw Unsupported("left shift count type " + count_type);
      y = a.word_shl(word(x.at(0), w), word(x.at(1), width(count_type)), w);
    }
    if (t.starts_with("s")) return a.signed_word(y, w);
    return word(y, w);
  }
  if (name == "PtxBytePermute" && t == "b32" &&
      attr == "source=b32.b32 source_type=b32") {
    auto control = a.value(x.at(2));
    if (!control || control->denominator() != 1 || *control < 0 || *control > 65535) {
      if (auto permuted = symbolic_byte_permute(word(x.at(0),32), word(x.at(1),32), word(x.at(2),32)))
        return *permuted;
      throw Unsupported("byte permute needs constant 16-bit selector");
    }
    return byte_permute(word(x.at(0),32), word(x.at(1),32), control->numerator().convert_to<unsigned>());
  }
  if (name == "PtxLogic3" && t == "b32" &&
      attr == "source=b32.b32 source_type=b32") {
    auto b = a.value(x.at(1)), c = a.value(x.at(2)), lut = a.value(x.at(3));
    if (!b || !c || !lut || b->denominator()!=1 || c->denominator()!=1 ||
        lut->denominator()!=1 || *b<0 || *c<0 || *b>=Q(pow2(32)) || *c>=Q(pow2(32)) || *lut<0 || *lut>255)
      throw Unsupported("logic3 requires two constant masks and 8-bit LUT");
    return logic3_constants(word(x.at(0),32), b->numerator().convert_to<uint32_t>(),
                            c->numerator().convert_to<uint32_t>(), lut->numerator().convert_to<unsigned>());
  }
  if (name == "PtxBitExtract") {
    // PTX bfe.{u32,u64,s32,s64} d, a, pos, len with constant pos and len.
    // ISA: pos = b & 0xff, len = c & 0xff; d[i] = a[pos+i] for i < len and
    // pos+i <= msb, else sbit; sbit = 0 for unsigned or len == 0, else
    // a[min(pos+len-1, msb)]. So d is the field of width min(len, msb+1-pos)
    // at pos, sign-extended from its top bit for signed types; past msb the
    // field is empty and a signed d replicates a[msb].
    const unsigned w = width(t);
    if (!integer || (w != 32 && w != 64) || x.size() != 3 || t[0] == 'b')
      throw Unsupported("bit extract type/arity; type=" + t);
    auto pos_value = a.value(x[1]), len_value = a.value(x[2]);
    if (!pos_value || !len_value || pos_value->denominator() != 1 || len_value->denominator() != 1 ||
        *pos_value < 0 || *len_value < 0 || *pos_value >= Q(pow2(32)) || *len_value >= Q(pow2(32)))
      throw Unsupported("bit extract requires constant position and length");
    const unsigned pos = static_cast<unsigned>(pos_value->numerator().convert_to<uint64_t>() & 0xff);
    const unsigned len = static_cast<unsigned>(len_value->numerator().convert_to<uint64_t>() & 0xff);
    const bool sign = t[0] == 's';
    ++a.stats["constant_bit_extracts"];
    if (len == 0) return a.zero;
    const Id storage = word(x[0], w);
    if (pos >= w) return sign ? a.signed_word(a.bit_slice(storage, w - 1, 1), 1) : a.zero;
    const unsigned field = std::min(len, w - pos);
    const Id bits = a.bit_slice(storage, pos, field);
    return sign ? a.signed_word(bits, field) : bits;
  }
  if (name == "ExtractBits") {
    auto off = a.value(x.at(1));
    std::smatch m;
    static const std::regex re("width=(\\d+)");
    if (!std::regex_search(attr, m, re))
      throw Unsupported("extract bits metadata");
    unsigned w = std::stoul(m[1]);
    if (w == 0 || w > 256) throw Unsupported("extract bits width");
    if (!off) {
      auto bounds = a.nodes[x.at(1)].bounds;
      if(n.operands.size()!=2)throw Unsupported("dynamic extract bits requires source type");
      const auto &source_edge=n.operands[0], &offset_edge=n.operands[1];
      std::string source_type=source_edge.converted_type.empty()?ns.at(source_edge.node).type:source_edge.converted_type;
      std::string offset_type=offset_edge.converted_type.empty()?ns.at(offset_edge.node).type:offset_edge.converted_type;
      unsigned source_width=width(source_type),result_width=width(t);
      if(floating(offset_type) || !bounds || bounds->first<0 || bounds->second+Q(w)>Q(source_width) ||
         result_width<w || (floating(t) && result_width!=w))
        throw Unsupported("dynamic extract bits requires in-range typed coordinate","expression_form_unsupported",
            J{{"source_type",source_type},{"offset_type",offset_type},{"result_type",t},{"slice_width",w},
              {"offset_lower",bounds?str(bounds->first):"unknown"},{"offset_upper",bounds?str(bounds->second):"unknown"}});
      Id storage=x[0];
      if(floating(source_type))storage=cast(storage,source_type,"b"+std::to_string(source_width),true);
      storage=word(storage,source_width);
      Id bits=a.bit_slice(a.word_shr(storage,x[1],source_width,false),0,w);
      ++a.stats["parameterized_bit_extracts"];
      if(storage==a.zero)++a.stats["zero_dynamic_bit_extracts"];
      return floating(t)||t.starts_with("s")?
          cast(bits,(floating(t)?"b":"u")+std::to_string(result_width),t,true):bits;
    }
    if (off->denominator() != 1 || *off < 0 || *off>Q(UINT_MAX-w))
      throw Unsupported("extract bits offset");
    const auto result_width=width(t);
    if(result_width<w || (floating(t) && result_width!=w))
      throw Unsupported("extract bits result type width mismatch");
    // Projection's typed ExtractBits includes the interpretation of the
    // selected storage field. In particular bf16 is not the unsigned value
    // of its encoding. Reuse the same exact slice/cast path as type edges.
    unsigned source_width=256;
    auto storage=x[0];
    if(!n.operands.empty()) {
      const auto& e=n.operands[0];
      const auto source_type=e.converted_type.empty()?ns.at(e.node).type:e.converted_type;
      source_width=width(source_type);
      if(floating(source_type))storage=cast(storage,source_type,"b"+std::to_string(source_width),true);
    }
    if(floating(t) || t.starts_with("s")) {
      if(*off+Q(w)>Q(source_width))throw Unsupported("typed extract exceeds source width");
      const auto masked=a.nodes[storage];
      if(floating(t) && (t=="f16" || t=="bf16" || t=="f32" || t=="f64") &&
         masked.op=="constant_bit_mask" && masked.args.size()==1 && masked.data.size()==2 &&
         *off+Q(w)<=masked.data[1]) {
        const auto shift=off->numerator().convert_to<unsigned>();
        Z lane=(masked.data[0].numerator()>>shift)&(pow2(w)-1);
        if(lane==pow2(w-1)-1) {
          auto selected=a.intern("extractbits",{masked.args[0]},{*off,Q(w)},"",
              {true,false,true},std::pair(Q(0),Q(pow2(w)-1)));
          if(auto value=initial_typed_bits(selected,t)) {
            ++a.stats["typed_input_sign_mask_abs"];
            return a.absolute(*value);
          }
        }
      }
      auto bits=word(a.idiv(storage,pow2(off->numerator().convert_to<unsigned>())),w);
      return cast(bits,(floating(t)?"b":"u")+std::to_string(result_width),t,true);
    }
    // The low w bits of an integer already in [0, 2^w) are the integer itself.
    if(*off==0 && whole_word(storage,w))return storage;
    return a.bit_slice(storage,off->numerator().convert_to<unsigned>(),w);
  }
  if (name == "ShiftRight") {
    unsigned w = width(t);
    if(!integer || x.size()!=2)throw Unsupported("right shift operand type");
    Id source=word(x[0],w),shift=word(x[1],32);
    bool sign=t.starts_with("s");
    if(sign)source=cast(source,"u"+std::to_string(w),t);
    return a.word_shr(source,shift,w,sign);
  }
  if (name == "BitOr" && integer && x.size() == 2) {
    const unsigned w=width(t);
    Id lhs=word(x[0],w), rhs=word(x[1],w);
    const auto finish=[&](Id value) {return t.starts_with("s")?
        cast(value,"u"+std::to_string(w),t):value;};
    return finish(a.word_or(lhs,rhs,w));
  }
  if (name == "BitXor" && integer) {
    unsigned w = width(t);
    Id result = a.bitxor(word(x.at(0), w), word(x.at(1), w));
    return t.starts_with("s") ? cast(result, "u" + std::to_string(w), t)
                              : result;
  }
  if (name == "Absolute") {
    if (!floating(t))
      throw Unsupported("integer Absolute adapter");
    return a.absolute(x.at(0));
  }
  if ((name == "BitAnd" || name == "BitOr" || name == "BitXor") && t == "pred") {
    const std::string logic=name=="BitAnd"?"AND":name=="BitOr"?"OR":"XOR";
    if(x.size()!=2) throw Unsupported("predicate "+logic+" arity");
    for(Id v:x) if(!a.nodes[v].bounds || a.nodes[v].bounds->first<0 || a.nodes[v].bounds->second>1)
      throw Unsupported("predicate "+logic+" operand outside Boolean range");
    // On {0,1}: x&y = xy, x|y = 1-(1-x)(1-y) (De Morgan over NOT = 1-x), x^y = (x|y)(1-xy).
    // Each form is exact and keeps the result's bounds within [0,1].
    const Id both=a.mul(x);
    if(name=="BitAnd")return both;
    const Id either=a.sub(a.one,a.mul({a.sub(a.one,x[0]),a.sub(a.one,x[1])}));
    return name=="BitOr"?either:a.mul({either,a.sub(a.one,both)});
  }
  if (name == "BitAnd") {
    auto m = a.value(x.at(1));
    if ((!m || m->denominator() != 1) && integer)
      if (auto both = bitwise_and(x.at(0), x.at(1), t)) return *both;
    if (!m || m->denominator() != 1)
      throw Unsupported("nonconstant mask");
    Z mask = m->numerator();
    if (mask == 0)
      return a.zero;
    if (mask < 0)
      throw Unsupported("negative mask");
    // A constant mask of a Select of two integer constants is the Select of the two masked
    // constants (each alternative is one whole word). Exact. O(1).
    if (integer && mask < pow2(width(t))) {
      const unsigned w = width(t);
      const auto &chosen = a.nodes[unwrap(x[0], w)];
      if (chosen.op == "select" && chosen.args.size() == 3) {
        const auto yes = a.value(chosen.args[1]), no = a.value(chosen.args[2]);
        if (yes && no && yes->denominator() == 1 && no->denominator() == 1) {
          const Id condition = chosen.args[0];
          const Z masked_yes = modz(yes->numerator(), pow2(w)) & mask, masked_no = modz(no->numerator(), pow2(w)) & mask;
          ++a.stats["masked_constant_selects"];
          const Id selected = a.select(condition, a.constant(Q(masked_yes)), a.constant(Q(masked_no)), true);
          return t.starts_with("s") ? a.signed_word(selected, w) : selected;
        }
      }
    }
    // A word built from symbolic left shifts: each mask bit is decided by Arena::shifted_word_bit.
    if (integer && mask < pow2(width(t)))
      if (auto masked = a.shifted_word_mask(x[0], mask))
        return t.starts_with("s") ? a.signed_word(*masked, width(t)) : *masked;
    Z low = mask & -mask, span = mask / low + 1;
    if((a.nodes[x[0]].op=="word_shr" || a.nodes[x[0]].op=="word_or") && mask<pow2(width(t))) {
      std::vector<Id> bits;
      for(unsigned b=0;b<width(t);++b)if((mask&pow2(b))!=0)
        bits.push_back(a.mul({a.constant(Q(pow2(b))),a.bit_slice(x[0],b,1)}));
      return a.add(bits);
    }
    if ((span & (span - 1)) != 0) {
      if(!integer || mask>=pow2(width(t)))throw Unsupported("mask exceeds integer width");
      const Id storage=word(x[0],width(t));
      // A constant word under a constant mask is the constant (word & mask), the node's own value.
      if(auto v=a.value(storage);v && v->denominator()==1 && *v>=0) {
        ++a.stats["constant_bit_mask_folds"];
        return a.constant(Q(Z(v->numerator() & mask)));
      }
      return a.intern("constant_bit_mask",{storage},{Q(mask),Q(width(t))},"",
          {true,false,true},std::pair(Q(0),Q(mask)));
    }
    return a.mul({a.constant(Q(low)),
                  a.imod(a.idiv(unwrap(x[0], width(t)), low), span)});
  }
  if ((name == "FloorDivide" || name == "Modulo") && integer) {
    auto divisor = a.value(x.at(1));
    if (!divisor || divisor->denominator() != 1 || *divisor <= 0)
      throw Unsupported("FloorDivide requires positive constant integer divisor");
    return name == "FloorDivide" ? integer_index_quotient(x.at(0), divisor->numerator())
                                 : a.imod(x.at(0), divisor->numerator());
  }
  if (integer && (name == "Divide" || name == "Remainder")) {
    auto d = a.value(x.at(1));
    if (!d || d->denominator() != 1 || *d <= 0)
      throw Unsupported("dynamic index divisor");
    auto b = a.nodes[x[0]].bounds;
    if (t.starts_with("s") && (!b || b->first < 0))
      throw Unsupported("signed truncation not floor");
    return name == "Divide" ? integer_index_quotient(x[0], d->numerator())
                            : a.imod(x[0], d->numerator());
  }
  if (name == "Add")
    return a.add(x);
  if (name == "Subtract")
    return a.sub(x.at(0), x.at(1));
  if (name == "Multiply") {
    if (auto lse = compiler_lse(n, ctx)) return *lse;
    return a.mul(x);
  }
  if (name == "Fma")
    return split_constant_low_part(n) ? x.at(2) : a.add({a.mul({x.at(0), x.at(1)}), x.at(2)});
  if (name == "Negate")
    return a.neg(x.at(0));
  if (name == "Exp")
    return a.neginf(x.at(0)) ? a.zero : a.exp(x[0]);
  // The rounded log2(e) literal is 1/ln 2 (ideal_rounded_operands), so the compiler's exp lowering
  // exp2(x * log2(e)) is exp(x) by the exact base change of Arena::exponential.
  if (name == "Exp2")
    return a.neginf(x.at(0)) ? a.zero : a.exp2(x[0]);
  if (name == "Reciprocal")
    return a.recip(x.at(0));
  if (name == "Divide")
    return a.divide(x.at(0), x.at(1));
  if (name == "Select") {
    const auto [yes, no] = same_condition_arms(x.at(0), x.at(1), x.at(2));
    return a.select(x.at(0), yes, no, !floating(n.type));
  }
  if (name == "Maximum")
    return a.maximum(in_range(proved_in_range(x, n, true, ctx), n.type, true));
  if (name == "Minimum") {
    // Ideal-real minimum shares the maximum normal form.
    std::vector<Id> negatives;
    for (Id operand : in_range(proved_in_range(x, n, false, ctx), n.type, false))
      negatives.push_back(a.neg(operand));
    return a.neg(a.maximum(std::move(negatives)));
  }
  if (name == "Compare") {
    std::smatch m;
    static const std::regex re("predicate=(\\w+)");
    if (!std::regex_search(attr, m, re))
      throw Unsupported("missing predicate metadata");
    if (auto decided = selected_comparison(m[1], x.at(0), x.at(1))) return *decided;
    return a.compare(m[1], x.at(0), x.at(1));
  }
  static const std::map<std::string, std::string> cmp{
      {"Less", "lt"},         {"LessEqual", "le"}, {"Greater", "gt"},
      {"GreaterEqual", "ge"}, {"Equal", "eq"},     {"NotEqual", "ne"}};
  if (cmp.contains(name)) {
    if (auto decided = selected_comparison(cmp.at(name), x.at(0), x.at(1))) return *decided;
    return a.compare(cmp.at(name), x.at(0), x.at(1));
  }
  if (name == "Log" || name == "Power") {
    if (!ideal || !floating(t))
      throw Unsupported("real Log/Power requires ideal_real floating expression");
    return name == "Log" ? a.logarithm(x.at(0)) : a.power(x.at(0), x.at(1));
  }
  if (name == "PtxLog2") {
    if (!ideal || !floating(t)) throw Unsupported("PtxLog2 requires ideal_real floating expression");
    return a.logarithm(x.at(0), true);
  }
  if (name == "PtxSqrt") {
    if (!ideal) throw Unsupported("PtxSqrt requires ideal_real");
    name = "Sqrt";
  }
  if (name == "Rsqrt" || name == "PtxReciprocalSqrt")
    name = "ReciprocalSqrt";
  if (name == "PtxSin")
    name = "Sin";
  if (name == "PtxCos")
    name = "Cos";
  if (name == "Sqrt" || name == "ReciprocalSqrt" || name == "Sin" ||
      name == "Cos" || name == "Exp" || name == "Tanh") {
    bool finite = std::all_of(x.begin(), x.end(),
                              [&](Id z) { return a.nodes[z].facts.finite; });
    if (name == "Sqrt") finite &= a.nodes[x[0]].facts.nn;
    if (name == "ReciprocalSqrt") finite &= a.nodes[x[0]].facts.pos;
    bool pos = finite && (name == "ReciprocalSqrt" || name == "Exp" ||
                         (name == "Sqrt" && a.nodes[x[0]].facts.pos));
    bool nn = pos || (finite && name == "Sqrt");
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return a.intern(name, x, {}, "", {finite, pos, nn});
  }
  throw Unsupported("operation " + name + " [" + attr + "]; type="+t+
      "; arity="+std::to_string(x.size()));
}

} // namespace rg
