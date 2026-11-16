#pragma once
#include "core.hpp"

namespace rg {

inline auto Arena::reverse_comparison(const std::string &op) -> std::string {
  if(op=="lt")return "gt";
  if(op=="le")return "ge";
  if(op=="gt")return "lt";
  if(op=="ge")return "le";
  return op;
}

inline auto Arena::complement_comparison(const std::string &op) -> std::string {
  if(op=="eq")return "ne";
  if(op=="ne")return "eq";
  if(op=="lt")return "ge";
  if(op=="le")return "gt";
  if(op=="gt")return "le";
  return "lt";
}

inline auto Arena::sign_comparison(const std::string &op,Id a,Id b) const -> std::optional<SignComparison> {
  Id id=a;auto boundary=value(b);std::string order=op;
  if(!boundary && value(a)){id=b;boundary=value(a);order=reverse_comparison(op);}
  const auto &n=nodes[id];
  if(!boundary || n.op!="signed" || n.args.size()!=1 || n.data.size()!=1 ||
     n.data[0].denominator()!=1 || n.data[0]<1 || n.data[0]>256 || !n.facts.finite)
    return {};
  const bool negative=(*boundary==0 && order=="lt") || (*boundary==-1 && order=="le");
  const bool nonnegative=(*boundary==0 && order=="ge") || (*boundary==-1 && order=="gt");
  if(!negative && !nonnegative)return {};
  return SignComparison{n.args[0],n.data[0].numerator().convert_to<unsigned>()-1,negative};
}

inline auto Arena::comparison_difference(Id a,Id b) const -> std::optional<ComparisonDifference> {
  if(!nodes[a].facts.finite || !nodes[b].facts.finite)return {};
  ComparisonDifference form;size_t terms_count=0,factor_count=0;
  auto atom=[&](Id id,const Q &coefficient) {
    if(++terms_count>max_comparison_terms)return false;
    const auto &n=nodes[id];
    if(n.op=="const"){form.offset+=coefficient*n.data[0];return true;}
    ComparisonAtom factors;Q c=coefficient;
    if(n.op=="mul") {
      if(n.data.size()!=1 || n.args.size()>max_comparison_terms-factor_count)return false;
      factor_count+=n.args.size();factors=n.args;c*=n.data[0];
    } else {
      if(++factor_count>max_comparison_terms)return false;
      factors.push_back(id);
    }
    if(factors.empty())form.offset+=c;
    else form.coefficients[std::move(factors)]+=c;
    return true;
  };
  auto side=[&](Id id,const Q &sign) {
    const auto &n=nodes[id];
    if(n.op!="add")return atom(id,sign);
    if(n.args.size()>max_comparison_terms-terms_count || n.data.size()!=n.args.size())return false;
    for(size_t i=0;i<n.args.size();++i)if(!atom(n.args[i],sign*n.data[i]))return false;
    return true;
  };
  if(!side(a,Q(1)) || !side(b,Q(-1)))return {};
  for(auto it=form.coefficients.begin();it!=form.coefficients.end();)
    if(it->second==0)it=form.coefficients.erase(it);else ++it;
  return form;
}

inline auto Arena::normalize_boolean_comparison(std::string &op,Id &a,Id &b) const -> BooleanComparisonStep {
  BooleanComparisonStep out;
  if(op!="eq" && op!="ne")return out;
  if(value(a) && !value(b))std::swap(a,b);
  const auto &scaled=nodes[a];
  if(b==zero && scaled.op=="mul" && scaled.args.size()==1 &&
     scaled.data.size()==1 && scaled.data[0]!=0 && nodes[scaled.args[0]].op=="compare") {
    a=scaled.args[0];out.scaled=true;
  }
  const auto &inner=nodes[a];
  if(inner.op!="compare" || inner.args.size()!=2 || !inner.data.empty() ||
     !valid_comparison(inner.tag) || (b!=zero && b!=one))return out;
  if((op=="ne" && b==zero)||(op=="eq" && b==one)) {out.resolved=a;return out;}
  if(nodes[inner.args[0]].facts.finite && nodes[inner.args[1]].facts.finite) {
    op=complement_comparison(inner.tag);a=inner.args[0];b=inner.args[1];out.complemented=true;
  }
  return out;
}

// Both real comparison construction and Max recognition use this exact
// normal form. Only already-known integer facts are read here. At most 32
// stored terms/factors are inspected; no numerical ancestors are visited.
inline auto Arena::normalize_comparison(std::string op,Id a,Id b,
    std::optional<ComparisonDifference> form) const -> ComparisonDescription {
  ComparisonDescription out;out.op=op;out.lhs=a;out.rhs=b;
  if(!valid_comparison(op)){out.kind=ComparisonDescription::NotPrepared;return out;}
  if(!form)return out;
  for(const auto &[factors,c]:form->coefficients) {
    (void)c;
    for(Id id:factors) {
      auto it=known_bits_cache.find(id);
      if(!nodes[id].bounds || it==known_bits_cache.end() || !it->second.integer)return out;
    }
  }
  if(form->coefficients.empty()) {
    out.kind=ComparisonDescription::Constant;
    const Q &c=form->offset;
    out.truth=op=="eq"?c==0:op=="ne"?c!=0:op=="lt"?c<0:op=="le"?c<=0:op=="gt"?c>0:c>=0;
    return out;
  }
  Z denominator=1;
  for(const auto &[_,c]:form->coefficients)
    denominator=denominator/gcdz(denominator,c.denominator())*c.denominator();
  Z divisor=0;
  for(const auto &[_,c]:form->coefficients) {
    Z v=c.numerator()*(denominator/c.denominator());
    divisor=gcdz(divisor,v);
  }
  Q scale(denominator,divisor);
  if(form->coefficients.begin()->second<0){scale=-scale;op=reverse_comparison(op);}
  Q threshold=-form->offset*scale;
  if(op=="lt"){op="le";threshold=Q(-floorq(-threshold)-1);}
  else if(op=="gt"){op="ge";threshold=Q(floorq(threshold)+1);}
  else if(op=="le")threshold=Q(floorq(threshold));
  else if(op=="ge")threshold=Q(-floorq(-threshold));
  else if(threshold.denominator()!=1) {
    out.kind=ComparisonDescription::Constant;out.truth=op=="ne";return out;
  }
  for(auto &[_,c]:form->coefficients)c*=scale;
  form->offset=0;
  out.kind=ComparisonDescription::Affine;out.op=std::move(op);
  out.form=std::move(*form);out.threshold=std::move(threshold);
  return out;
}

inline auto Arena::describe_comparison(std::string op,Id a,Id b) const -> ComparisonDescription {
  if(!valid_comparison(op)) {
    ComparisonDescription out;out.kind=ComparisonDescription::NotPrepared;return out;
  }
  // Fixed local Boolean shells only. Resolved bit queries may themselves be
  // comparisons (e.g. a bit of an all-ones parameterized right shift).
  for(unsigned shell=0;shell<4;++shell) {
    if(auto sign=sign_comparison(op,a,b)) {
      auto bits=known_bits_cache.find(sign->source);
      auto slice=resolved_bit_slice_cache.find(std::tuple(sign->source,sign->bit,1u));
      if(bits==known_bits_cache.end() || !bits->second.integer || slice==resolved_bit_slice_cache.end()) {
        ComparisonDescription out;out.kind=ComparisonDescription::NotPrepared;return out;
      }
      op=sign->negative?"ne":"eq";a=slice->second;b=zero;continue;
    }
    auto boolean=normalize_boolean_comparison(op,a,b);
    if(boolean.resolved!=invalid) {
      const auto &n=nodes[boolean.resolved];op=n.tag;a=n.args[0];b=n.args[1];continue;
    }
    if(boolean.complemented)continue;
    return normalize_comparison(op,a,b,comparison_difference(a,b));
  }
  ComparisonDescription out;out.kind=ComparisonDescription::NotPrepared;return out;
}

inline auto Arena::same_comparison(const ComparisonDescription &a,const ComparisonDescription &b) -> bool {
  if(a.kind!=b.kind || a.kind==ComparisonDescription::NotPrepared)return false;
  if(a.kind==ComparisonDescription::Constant)return a.truth==b.truth;
  if(a.kind==ComparisonDescription::Affine)
    return a.op==b.op && a.threshold==b.threshold && a.form.coefficients==b.form.coefficients;
  return (a.op==b.op && a.lhs==b.lhs && a.rhs==b.rhs) ||
         (a.op==reverse_comparison(b.op) && a.lhs==b.rhs && a.rhs==b.lhs);
}

inline auto Arena::select_is_max(Id p,Id a,Id b) const -> bool {
  const auto &n=nodes[p];
  if(n.op!="compare" || n.args.size()!=2 || !n.data.empty() || !valid_comparison(n.tag) ||
     !nodes[a].facts.finite || !nodes[b].facts.finite)return false;
  if(((n.tag=="gt" || n.tag=="ge") && n.args[0]==a && n.args[1]==b) ||
     ((n.tag=="lt" || n.tag=="le") && n.args[0]==b && n.args[1]==a))return true;
  // Ordinary equality-controlled decoders are not order predicates. Retain
  // the signed-word boundary case, whose real comparison can become eq/ne.
  if((n.tag=="eq" || n.tag=="ne") &&
     !sign_comparison("gt",a,b) && !sign_comparison("ge",a,b))return false;
  auto predicate=describe_comparison(n.tag,n.args[0],n.args[1]);
  return same_comparison(predicate,describe_comparison("gt",a,b)) ||
         same_comparison(predicate,describe_comparison("ge",a,b));
}

// An (in)equality of an IEEE binary16/bfloat16/binary32/binary64 encoding
// floatbits_w(x) with a constant pattern, read through the encoding:
// - the w-1 magnitude bits floatbits_w(x) mod 2^(w-1) are zero exactly for
//   +0 and -0, so that test is the floating zero test x == 0;
// - every encoding of a positive x (finite, rounded to +0, or +infinity)
//   has a clear sign bit, so a pattern with the sign bit set differs.
// Other patterns keep the integer comparison. O(1).
inline auto Arena::encoding_comparison(const std::string &p, Id lhs, Id rhs) -> std::optional<Id> {
  if (p != "eq" && p != "ne") return {};
  if (value(lhs) && !value(rhs)) std::swap(lhs, rhs);
  const auto pattern = value(rhs);
  if (!pattern || pattern->denominator() != 1 || *pattern < 0) return {};
  auto encoded = [&](Id id) -> std::optional<std::pair<Id, unsigned>> {
    const auto &n = nodes[id];
    if (n.op != "floatbits" || n.args.size() != 1 || n.data.size() != 1 ||
        (n.tag != "f16" && n.tag != "bf16" && n.tag != "f32" && n.tag != "f64"))
      return {};
    const unsigned w = n.data[0].numerator().convert_to<unsigned>();
    if (Q(w) != n.data[0] || w != (n.tag == "f64" ? 64u : n.tag == "f32" ? 32u : 16u)) return {};
    return std::pair(n.args[0], w);
  };
  const auto &n = nodes[lhs];
  if (n.op == "imod" && n.args.size() == 1 && n.data.size() == 1 && *pattern == 0)
    if (auto source = encoded(n.args[0]); source && n.data[0] == Q(pow2(source->second - 1))) {
      ++stats["floating_encoding_zero_tests"];
      return compare(p, source->first, zero);
    }
  if (auto source = encoded(lhs); source && *pattern >= Q(pow2(source->second - 1)) &&
      *pattern < Q(pow2(source->second)) && nodes[source->first].facts.finite &&
      nodes[source->first].facts.pos) {
    ++stats["floating_encoding_sign_tests"];
    return constant(p == "ne");
  }
  return {};
}

inline auto Arena::compare(std::string p, Id a, Id b) -> Id {
  if (!valid_comparison(p))
    throw Unsupported("comparison " + p);
  auto av = value(a), bv = value(b);
  std::optional<int> c;
  if(a==b && nodes[a].facts.finite)c=0;
  // Two infinity constants of the extended reals: -inf = -inf, +inf = +inf, -inf < +inf. Exact
  // (neither is NaN). O(1).
  else if (nodes[a].op == "infinity" && nodes[b].op == "infinity") {
    const int sa = nodes[a].data[0].numerator().convert_to<int>(), sb = nodes[b].data[0].numerator().convert_to<int>();
    c = sa < sb ? -1 : sa > sb ? 1 : 0;
    ++stats["infinity_constant_comparisons"];
  }
  else if (av && bv)
    c = *av < *bv ? -1 : *av > *bv ? 1 : 0;
  else if (nodes[a].op == "infinity" && nodes[b].facts.finite)
    c = nodes[a].data[0].numerator().convert_to<int>();
  else if (nodes[b].op == "infinity" && nodes[a].facts.finite)
    c = -nodes[b].data[0].numerator().convert_to<int>();
  if (!c && nodes[a].facts.finite && nodes[b].facts.finite && nodes[a].bounds && nodes[b].bounds) {
    const auto &ab=*nodes[a].bounds, &bb=*nodes[b].bounds;
    if (ab.second < bb.first) c=-1;
    else if (ab.first > bb.second) c=1;
    else if (ab.first==ab.second && ab==bb) c=0;
    else if (p=="lt" && ab.first>=bb.second) return zero;
    else if (p=="ge" && ab.first>=bb.second) return one;
    else if (p=="gt" && ab.second<=bb.first) return zero;
    else if (p=="le" && ab.second<=bb.first) return one;
  }
  if (c) {
    bool r = p == "eq"   ? *c == 0
             : p == "ne" ? *c != 0
             : p == "lt" ? *c < 0
             : p == "le" ? *c <= 0
             : p == "gt" ? *c > 0
                         : *c >= 0;
    return constant(int(r));
  }
  // Rule (-inf test of an extended sum): see neginf_sum_condition. Exact.
  if (p == "eq" && (neginf(a) || neginf(b))) {
    const Id sum = neginf(a) ? b : a;
    if (nodes[sum].op == "add_raw")
      if (auto condition = neginf_sum_condition(sum)) {
        ++stats["neginf_sum_conditions"];
        return *condition;
      }
  }
  if (b == zero && nodes[a].facts.pos &&
      (p == "ne" || p == "gt" || p == "ge"))
    return one;
  if (b == zero && nodes[a].facts.finite && nodes[a].facts.pos &&
      (p == "eq" || p == "lt" || p == "le"))
    return zero;
  // An (in)equality of a raw 32-bit input word, while an importer that can read its typed element
  // is active (Importer::typed_word_predicate). O(1) unless an operand is such a word.
  if (word_predicate && (p == "eq" || p == "ne") &&
      (raw_word32(a) || raw_word32(b)))
    if (auto typed = word_predicate(p, a, b)) return *typed;
  if (auto decoded = encoding_comparison(p, a, b)) return *decoded;
  // Predicate constructors are shared by import, substitution and reference
  // instantiation. Inspect only this comparison's immediate Boolean shell;
  // never distribute Selects or search numerical ancestors.
  // A signed word crossing its sign boundary is not an unknown real value.
  // Sign-only comparisons consume the same parametric bit query used by
  // extraction. Preserve the declared width; do not split its member domain.
  if(auto query=sign_comparison(p,a,b);query && known_bits(query->source).integer) {
    Id sign=bit_slice(query->source,query->bit,1);
    ++stats["signed_word_sign_queries"];
    return compare(query->negative?"ne":"eq",sign,zero);
  }
  auto boolean=normalize_boolean_comparison(p,a,b);
  if(boolean.scaled)++stats["predicate_scaled_boolean_zero"];
  if(boolean.complemented)++stats["predicate_boolean_complements"];
  if(boolean.resolved!=invalid)return boolean.resolved;

  // Actual construction may prepare integer facts. The shared normalizer
  // and its read-only consumers never do so, and never call terms().
  Id original_a=a,original_b=b;std::string original_p=p;
  auto form=comparison_difference(a,b);
  if(form) {
    bool ready=true;
    for(const auto &[factors,c]:form->coefficients) {
      (void)c;
      for(Id id:factors)if(!nodes[id].bounds || !known_bits(id).integer){ready=false;break;}
      if(!ready)break;
    }
    auto description=normalize_comparison(p,a,b,std::move(form));
    if(description.kind==ComparisonDescription::Constant)return description.truth?one:zero;
    if(description.kind==ComparisonDescription::Affine) {
      std::vector<Id> parts;
      for(const auto &[factors,c]:description.form.coefficients) {
        std::vector<Id> xs=factors;
        if(c!=1)xs.push_back(constant(c));
        parts.push_back(xs.size()==1?xs[0]:mul(std::move(xs)));
      }
      a=add(std::move(parts));b=constant(description.threshold);p=description.op;
      if(a!=original_a || b!=original_b || p!=original_p) {
        ++stats["predicate_integer_affine_normalizations"];
        return compare(p,a,b);
      }
    }
  }
  const std::array<Id, 2> operands{a, b};
  return intern_lazy("compare", operands, {}, p, [&](size_t hashed) {
    return insert_node(hashed, "compare", {a, b}, {}, p, {true, false, true},
                       std::pair(Q(0), Q(1)));
  });
}

// The interval of a Select arm where it is taken: when p (taken) or its
// complement is a comparison of a bounded value x with a constant, x's
// interval is narrowed by it inside an arm that is x, k*x or an affine sum
// of bounded terms. Otherwise the arm's own interval. Arms must be bounded;
// O(arm arity).
inline auto Arena::guarded_bounds(Id p, Id arm, bool taken) const -> std::pair<Q, Q> {
  const auto own = *nodes[arm].bounds;
  const auto &condition = nodes[p];
  if (condition.op != "compare" || condition.args.size() != 2 || !condition.data.empty())
    return own;
  std::string op = taken ? condition.tag : complement_comparison(condition.tag);
  Id x = condition.args[0];
  auto c = value(condition.args[1]);
  if (!c) {
    c = value(condition.args[0]);
    x = condition.args[1];
    op = reverse_comparison(op);
  }
  if (!c || value(x) || !nodes[x].bounds || !nodes[x].facts.finite) return own;
  auto narrowed = *nodes[x].bounds;
  if (op == "le" || op == "lt") narrowed.second = std::min(narrowed.second, *c);
  else if (op == "ge" || op == "gt") narrowed.first = std::max(narrowed.first, *c);
  else if (op == "eq") narrowed = std::pair(*c, *c);
  else return own;
  if (narrowed.first > narrowed.second) return own;
  auto interval = [&](Id term) -> Bounds {
    if (term == x) return narrowed;
    return nodes[term].bounds;
  };
  Bounds result;
  const auto &n = nodes[arm];
  if (arm == x) result = narrowed;
  else if ((n.op == "add" || n.op == "mul") && std::find(n.args.begin(), n.args.end(), x) != n.args.end()) {
    const bool sum = n.op == "add";
    if (!sum && n.args.size() != 1) return own;
    result = std::pair(Q(0), Q(0));
    for (size_t i = 0; i < n.args.size() && result; ++i) {
      const auto term = interval(n.args[i]);
      if (!term) { result.reset(); break; }
      const Q &k = sum ? n.data[i] : n.data[0];
      const Q lo = term->first * k, hi = term->second * k;
      result->first += std::min(lo, hi);
      result->second += std::max(lo, hi);
    }
  }
  if (!result) return own;
  return std::pair(std::max(own.first, result->first), std::min(own.second, result->second));
}

// On one arm of a Select on condition c, an arm recip(x) with x finite where c rules out x <= 0
// there (compare(le, x, k) or compare(lt, x, k') on the false arm; compare(lt, k, x) or
// compare(le, k', x) on the true arm; k >= 0, k' > 0) is the reciprocal of a finite real > 0,
// a finite positive real: 2. Where c rules out x == 0 there (compare(eq, x, 0) or
// compare(eq, 0, x) on the false arm) it is finite: 1. Otherwise 0. Only that occurrence is
// refined: the facts of recip(x) and x are unchanged. O(1).
inline auto Arena::guarded_reciprocal(const Node &c, Id arm, bool on_true) const -> int {
  if (nodes[arm].op != "recip" || nodes[arm].args.size() != 1 || c.op != "compare" ||
      c.args.size() != 2 || !c.data.empty()) return 0;
  const Id x = nodes[arm].args[0];
  if (!nodes[x].facts.finite) return 0;
  const auto left = value(c.args[0]), right = value(c.args[1]);
  if (!on_true && c.args[0] == x && right) {
    if ((c.tag == "le" && *right >= 0) || (c.tag == "lt" && *right > 0)) return 2;
    if (c.tag == "eq" && *right == 0) return 1;
  }
  if (!on_true && c.args[1] == x && left && c.tag == "eq" && *left == 0) return 1;
  if (on_true && c.args[1] == x && left &&
      ((c.tag == "lt" && *left >= 0) || (c.tag == "le" && *left > 0))) return 2;
  return 0;
}

inline auto Arena::select(Id p, Id a, Id b, bool infer_bounds) -> Id {
  if (p == one)
    return a;
  if (p == zero)
    return b;
  if (a == b)
    return a;
  // A complementary condition with swapped arms denotes the same Select,
  // including when an arm is -infinity. Only the comparison operands must
  // be proved finite/ordered; requiring finite arms would lose masked Max.
  // Reuse compare() for integer strictness and affine normalization. The
  // preferred polarities (eq/le/lt) never switch back on the next rebuild.
  const auto &predicate = nodes[p];
  if (predicate.op == "compare" && predicate.args.size() == 2 &&
      predicate.data.empty() && nodes[predicate.args[0]].facts.finite &&
      nodes[predicate.args[1]].facts.finite &&
      (predicate.tag == "ne" || predicate.tag == "ge" || predicate.tag == "gt")) {
    const std::string opposite = predicate.tag == "ne" ? "eq"
                                 : predicate.tag == "ge" ? "lt" : "le";
    p = compare(opposite, predicate.args[0], predicate.args[1]);
    std::swap(a, b);
    ++stats["select_complementary_polarities"];
    if (p == one) return a;
    if (p == zero) return b;
  }
  // Rule (clamped sentinel): select([x == lo], lo + 1, x) = max(lo + 1, x) for an integer x whose
  // lower bound is lo. At x = lo both are lo + 1; elsewhere x >= lo + 1, so both are x. The two
  // spellings of clamping a -1 sentinel to row 0 (a Select on the sentinel, a Max with 0) become one
  // node. Inspects this Select only. Exact.
  if (auto clamped = sentinel_clamp(p, a, b)) return *clamped;
  // Rule (all-masked guard): select([m == -inf], 0, X) = X when m = -inf forces X = 0
  // (neginf_forces_zero): where the condition holds both are 0, elsewhere both are X. Exact.
  if (a == zero && nodes[b].facts.finite && neginf_max_comparison(p) != invalid &&
      neginf_forces_zero(p, b)) {
    ++stats["neginf_max_zero_selects"];
    return b;
  }
  // Rule (guard of a zero test): in select([X == 0], A, B) the arm B is taken only where X != 0.
  // A Select on a condition c = [m == -inf] that forces X = 0 is false there, so inside B it is
  // its false arm (neginf_guarded_arm). Exact.
  if (auto guarded = neginf_guarded_arm(p, b); guarded && *guarded != b) {
    ++stats["neginf_max_guard_resolutions"];
    return select(p, a, *guarded, infer_bounds);
  }
  // Recognition is a const query over bounded stored shells and prepared
  // facts. In particular it cannot call compare/add/mul/bit_slice/known_bits.
  ++stats["select_max_probe_calls"];
  const size_t probe_nodes=nodes.size(),probe_facts=known_bits_cache.size();
  const bool is_max=select_is_max(p,a,b);
  if(nodes.size()!=probe_nodes)stats["select_max_probe_temporary_nodes"]+=nodes.size()-probe_nodes;
  if(known_bits_cache.size()!=probe_facts)stats["select_max_probe_ancestor_nodes"]+=known_bits_cache.size()-probe_facts;
  if(is_max) {
    ++stats["select_max_probe_hits"];
    return maximum({a,b});
  }
  if(auto tournament=tournament_select(p,a,b,false))return *tournament;
  if(auto merged=argmax_point_select(p,a,b))return *merged;
  if(auto payload=payload_tournament_select(p,a,b))return *payload;
  const auto &condition=nodes[p];
  // Refine only the selected occurrence of the compared value. No inherited
  // path context is added to the Arena, and no fact escapes to that value.
  Facts af=nodes[a].facts,bf=nodes[b].facts;
  if(condition.op=="compare" && condition.args.size()==2 &&
     (condition.tag=="eq" || condition.tag=="ne")) {
    Id compared=invalid;
    if(neginf(condition.args[0]))compared=condition.args[1];
    else if(neginf(condition.args[1]))compared=condition.args[0];
    if(compared!=invalid && nodes[compared].facts.finite_or_negative_infinity()) {
      if(condition.tag=="ne" && a==compared)af.finite=true;
      if(condition.tag=="eq" && b==compared)bf.finite=true;
    }
  }
  // Likewise for a reciprocal arm whose condition excludes a zero divisor on that arm (facts only).
  for(const bool on_true:{true,false})if(const int g=guarded_reciprocal(condition,on_true?a:b,on_true)) {
    Facts &f=on_true?af:bf;
    f.finite=f.finite_or_neginf=true;
    if(g==2)f.pos=f.nn=true;
    ++stats["guarded_reciprocal_select_arms"];
  }
  const Id exponential_arm = a == one ? b : b == one ? a : invalid;
  if (exponential_arm != invalid && nodes[exponential_arm].op == "exp2" &&
      nodes[nodes[exponential_arm].args[0]].facts.finite) {
    // Keep the identity independent of predicate polarity. Canonicalizing
    // a complementary Select must not disable the existing rescale rule.
    const Id one_predicate = a == one ? p : compare("eq", p, zero);
    Id d = nodes[exponential_arm].args[0];
    const auto &n = nodes[d];
    if (n.op == "add" && n.args.size() == 2 && n.data[0] == -n.data[1]) {
      Id x = n.args[0], y = n.args[1];
      Q c = n.data[0];
      if (c < 0) {
        std::swap(x, y);
        c = -c;
      }
      if (nodes[x].facts.finite && nodes[y].facts.finite) {
        ++stats["select_rescaling_identity"];
        return exp2(mul({constant(c), sub(x, select(one_predicate, x, y))}));
      }
    }
    return exp2(select(one_predicate, zero, d));
  }
  Bounds bounds;
  // Demand-driven: propagating every floating Select interval makes all
  // downstream add/mul rebuilds perform otherwise unused rational arithmetic.
  if (infer_bounds && nodes[a].bounds && nodes[b].bounds) {
    const auto ab = guarded_bounds(p, a, true), bb = guarded_bounds(p, b, false);
    bounds = std::pair(std::min(ab.first, bb.first), std::max(ab.second, bb.second));
  }
  return intern("select", {p, a, b}, {}, "",
                {af.finite && bf.finite, af.pos && bf.pos, af.nn && bf.nn,
                 af.finite_or_negative_infinity() && bf.finite_or_negative_infinity()}, bounds);
}

inline auto Arena::clamp(Id x, Q lo, Q hi) -> Id {
  if (lo > hi) throw Unsupported("invalid clamp interval");
  if (auto v=value(x)) return constant(std::min(hi,std::max(lo,*v)));
  if (lo == hi && nodes[x].facts.finite) return constant(lo);
  if (auto b=nodes[x].bounds; b && nodes[x].facts.finite) {
    if(b->second <= lo) return constant(lo);
    if(b->first >= hi) return constant(hi);
    if(b->first >= lo && b->second <= hi) return x;
  }
  return intern("clamp",{x},{lo,hi},"",{nodes[x].facts.finite,lo>0,lo>=0},std::pair(lo,hi));
}

} // namespace rg
