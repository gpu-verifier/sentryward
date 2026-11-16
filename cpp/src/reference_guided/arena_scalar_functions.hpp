#pragma once
#include "core.hpp"

namespace rg {

// Integer operands are decoded at the import boundary. This operation uses
// unbounded nonnegative integer XOR; no logical-domain enumeration.
inline auto Arena::bitxor(Id x, Id y) -> Id {
  auto xb = nodes[x].bounds, yb = nodes[y].bounds;
  if (!xb || !yb || xb->first < 0 || yb->first < 0)
    throw Unsupported("XOR requires nonnegative integer operands");
  auto xv = value(x), yv = value(y);
  if ((xv && xv->denominator() != 1) || (yv && yv->denominator() != 1))
    throw Unsupported("XOR requires integer constants");
  if (xv && yv)
    return constant(Q(Z(xv->numerator() ^ yv->numerator())));
  if (x == y)
    return zero;
  if (x == zero)
    return y;
  if (y == zero)
    return x;
  if (xv) {
    std::swap(x, y);
    std::swap(xv, yv);
    std::swap(xb, yb);
  }
  if (yv) {
    const auto &n = nodes[x];
    if (n.op == "xor" && n.args[1] == y)
      return n.args[0];
    Z mask = yv->numerator();
    if (mask > 0 && (mask & (mask - 1)) == 0 && xb->second < Q(mask * 2)) {
      // Toggle one bit, valid only over the proved two-half interval.
      return sub(add({x, y}), mul({constant(Q(mask * 2)), idiv(x, mask)}));
    }
  }
  if (!yv && y < x)
    std::swap(x, y);
  Z hi = std::max(floorq(xb->second), floorq(yb->second));
  unsigned bits = hi == 0 ? 0 : boost::multiprecision::msb(hi) + 1;
  return intern("xor", {x, y}, {}, "", {true, false, true},
                std::pair(Q(0), Q(pow2(bits) - 1)));
}

inline auto Arena::absolute(Id x) -> Id {
  const auto &n = nodes[x];
  if (auto v = value(x))
    return constant(*v < 0 ? -*v : *v);
  if (n.op == "abs")
    return x;
  if (n.op == "infinity")
    return infinity(1);
  if (n.facts.finite && n.facts.nn)
    return x;
  if (n.bounds && n.bounds->second <= 0)
    return neg(x);
  Bounds b;
  if (n.bounds)
    b = std::pair(Q(0), std::max(-n.bounds->first, n.bounds->second));
  bool finite = n.facts.finite;
  return intern("abs", {x}, {}, "", {finite, finite && n.facts.pos, finite},
                b);
}

// The exact constant ln 2 = log(2), with a rational enclosure as bounds. Every log of the constant 2 is
// this node. The importer reads the rounded binary32/binary64 ln 2 and log2(e) literals as ln 2 and
// 1/ln 2 (ideal_rounded_operands).
inline auto Arena::natural_log2() -> Id {
  if (exact_ln2 == invalid)
    exact_ln2 = intern("log", {constant(2)}, {}, "", {true, true, true},
                       std::pair(Q(Z("69314718055994530"), Z("100000000000000000")),
                                 Q(Z("69314718055994531"), Z("100000000000000000"))));
  return exact_ln2;
}

// Exact ln 2 factors of one product: ln 2 * (1/ln 2) = 1 and log2(y) * ln 2 = log(y) (finite
// logarithms only). A 1/ln 2 factor is left for the base change exp2(y / ln 2) = exp(y)
// (exponent_bases), not merged into a logarithm. O(factors) per rewrite.
inline auto Arena::natural_log2_factors(std::vector<Id> &factors) -> void {
  const auto find = [&](auto &&match) {
    for (size_t i = 0; i < factors.size(); ++i) if (match(factors[i])) return i;
    return factors.size();
  };
  for (bool changed = true; changed;) {
    changed = false;
    const size_t ln2 = find([&](Id f) { return is_natural_log2(f); });
    const size_t inverse = find([&](Id f) { return is_inverse_natural_log2(f); });
    if (ln2 < factors.size() && inverse < factors.size()) {
      factors.erase(factors.begin() + std::max(ln2, inverse));
      factors.erase(factors.begin() + std::min(ln2, inverse));
      changed = true;
    } else if (ln2 < factors.size()) {
      const size_t log = find([&](Id f) {
        return nodes[f].op == "log2" && nodes[f].args.size() == 1 && nodes[f].facts.finite; });
      if (log < factors.size()) {
        factors[log] = logarithm(nodes[factors[log]].args[0]);
        factors.erase(factors.begin() + ln2);
        changed = true;
      }
    }
    if (changed) ++stats["exact_natural_log2_factor_rewrites"];
  }
}

// exp(y * ln 2) = exp2(y) and exp2(y / ln 2) = exp(y), exact for the ideal constant ln 2. A term of a
// finite exponent with exactly one ln 2 factor (natural base) or one 1/ln 2 factor (base two) moves to
// the other base without that factor. Returns the new (natural, binary) exponents, or nullopt when no
// term moves. A read-only scan decides first, so an exponent without such a term builds nothing.
// O(terms * factors).
inline auto Arena::exponent_bases(Id natural, Id binary) -> std::optional<std::pair<Id,Id>> {
  if (exact_ln2 == invalid) return {};
  const auto carries = [&](Id f, bool base2) { return base2 ? is_inverse_natural_log2(f) : is_natural_log2(f); };
  // The term t (a product with any scalar, or a single factor) carries exactly one such factor.
  const auto carries_one = [&](Id t, bool base2) {
    if (carries(t, base2)) return true;
    const auto &n = nodes[t];
    return n.op == "mul" && std::count_if(n.args.begin(), n.args.end(), [&](Id f) { return carries(f, base2); }) == 1;
  };
  const auto any = [&](Id e, bool base2) {
    if (e == zero) return false;
    const auto &n = nodes[e];
    if (n.op != "add") return carries_one(e, base2);
    return std::any_of(n.args.begin(), n.args.end(), [&](Id t) { return carries_one(t, base2); });
  };
  if (!any(natural, false) && !any(binary, true)) return {};
  std::vector<Id> natural_terms, binary_terms;
  const auto scan = [&](Id exponent, bool base2) {
    if (exponent == zero) return;
    for (const auto &[term, coefficient] : terms(exponent)) {
      std::optional<Id> rest;
      if (carries(term, base2)) rest = one;
      else if (nodes[term].op == "mul" && nodes[term].data[0] == 1) {
        const auto args = nodes[term].args;
        if (std::count_if(args.begin(), args.end(), [&](Id f) { return carries(f, base2); }) == 1) {
          std::vector<Id> others;
          for (Id f : args) if (!carries(f, base2)) others.push_back(f);
          rest = mul(std::move(others));
        }
      }
      auto &target = (rest ? !base2 : base2) ? binary_terms : natural_terms;
      target.push_back(mul({constant(coefficient), rest ? *rest : term}));
    }
  };
  scan(natural, false);
  scan(binary, true);
  return std::pair(add(std::move(natural_terms)), add(std::move(binary_terms)));
}

// The scale q > 0 with x = q*y, read without building y: a finite product's scalar magnitude, or that
// of the product arm of a mask select(p, q*y, -inf); 1 otherwise. O(1).
inline auto Arena::scale_of(Id x) const -> Q {
  const auto &n = nodes[x];
  if (n.op == "mul" && n.facts.finite && n.data[0] != 0 && n.data[0] != 1)
    return n.data[0] < 0 ? Q(-n.data[0]) : n.data[0];
  if (n.op == "select" && n.args.size() == 3 && neginf(n.args[1]) != neginf(n.args[2])) {
    const auto &arm = nodes[n.args[neginf(n.args[2]) ? 1 : 2]];
    if (arm.op == "mul" && arm.facts.finite && arm.data[0] != 0 && arm.data[0] != 1)
      return arm.data[0] < 0 ? Q(-arm.data[0]) : arm.data[0];
  }
  return Q(1);
}

// x as q*y with q = scale_of(x): a product's scalar magnitude (y keeps the sign), or the scale of the
// product arm of a mask select(p, q*y, -inf) (y masked the same way); (1, x) otherwise. O(1).
inline auto Arena::positive_scale(Id x) -> std::pair<Q,Id> {
  const Q q = scale_of(x);
  if (q == 1) return {q, x};
  const auto &n = nodes[x];
  const auto descale = [&](Id product) {
    const Q s = nodes[product].data[0];
    const auto args = nodes[product].args;
    const Id y = mul(args);
    return s < 0 ? neg(y) : y;
  };
  if (n.op == "mul") return {q, descale(x)};
  const auto args = n.args;
  const bool finite_true = neginf(args[2]);
  const Id y = descale(args[finite_true ? 1 : 2]);
  return {q, select(args[0], finite_true ? y : args[1], finite_true ? args[2] : y)};
}

// Rule: q*. with q > 0 is monotone, so max(q*y_1, .., q*y_n, k) = q*max(y_1, .., y_n, k/q). A
// positive constant common to every operand comes out of the Max, as a constant coefficient stays
// outside a sum: q*m and q'*m then share the atom m. The canonical Max Fold pulls its body's
// constant likewise (Checker::canonicalize_fold). nullopt unless every non-constant operand has the
// same scale q != 1; the scales are read before anything is built. O(operands).
inline auto Arena::common_positive_scale(const std::vector<Id> &xs, const std::optional<Q> &largest) -> std::optional<Id> {
  if (xs.empty()) return {};
  const Q common = scale_of(xs[0]);
  if (common == 1) return {};
  for (Id x : xs) if (scale_of(x) != common) return {};
  std::vector<Id> ys;
  for (Id x : xs) ys.push_back(positive_scale(x).second);
  if (largest) ys.push_back(constant(*largest / common));
  ++stats["maximum_scale_pulls"];
  return mul({constant(common), maximum(std::move(ys))});
}

// select([v <= k], c, v) with k >= 0 and c > 0 constants and v finite (or select([v < k], c, v) with
// k > 0): c > 0 where the condition holds and v > k >= 0 (v >= k > 0) where it fails, so the Select
// is a finite positive value. Read only by logarithm() below. O(1).
inline auto Arena::positive_view(Id s) const -> bool {
  const auto &n = nodes[s];
  if (n.op != "select" || n.args.size() != 3) return false;
  const auto &c = nodes[n.args[0]];
  if (c.op != "compare" || c.args.size() != 2 || !c.data.empty() || c.args[0] != n.args[2] ||
      !nodes[n.args[2]].facts.finite) return false;
  const auto k = value(c.args[1]), arm = value(n.args[1]);
  return k && arm && *arm > 0 && ((c.tag == "le" && *k >= 0) || (c.tag == "lt" && *k > 0));
}

inline auto Arena::logarithm(Id x, bool base2) -> Id {
  if (x == one) return zero;
  if (!base2 && value(x) == Q(2)) return natural_log2();
  // Preserve a partial real function when positivity is not established.
  bool finite = nodes[x].facts.finite && (nodes[x].facts.pos || positive_view(x));
  // The logarithm is increasing: a proved lower bound above (at least) 1
  // makes it positive (nonnegative), e.g. log(2) > 0.
  const auto &b = nodes[x].bounds;
  bool positive = finite && b && b->first > 1, nonnegative = finite && b && b->first >= 1;
  return intern(base2 ? "log2" : "log", {x}, {}, "", {finite, positive, nonnegative});
}

inline auto Arena::power(Id base, Id exponent) -> Id {
  if(nodes[base].facts.finite) {
    if(exponent==one) return base;
    if(auto v=value(exponent);v && *v==2) return mul({base,base});
  }
  // Positive finite base and finite exponent suffice for a real result.
  // No floating approximation or power expansion is needed for import.
  bool positive = nodes[base].facts.finite && nodes[base].facts.pos &&
                  nodes[exponent].facts.finite;
  if (positive && exponent == zero) return one;
  if (positive && exponent == one) return base;
  if (positive && base == one) return one;
  // A positive-base real power and the compiler's lg2/mul/ex2 lowering
  // share an exact symbolic form. Storage constants retain their exact
  // imported values; no numerical log approximation is performed here.
  if (positive)
    return exp2(mul({logarithm(base, true), exponent}));
  return intern("power", {base, exponent}, {}, "", {positive, positive, positive});
}

inline auto Arena::recip(Id n) -> Id {
  auto v = value(n);
  if (v && *v != 0)
    return constant(Q(1) / *v);
  // 1/(1/ln 2) = ln 2 for the exact constant.
  if (is_inverse_natural_log2(n)) return natural_log2();
  bool p = nodes[n].facts.finite && nodes[n].facts.pos;
  const auto &node = nodes[n];
  if (node.op == "sqrt" && p) {
    Id x=node.args[0];
    return intern("reciprocalsqrt",{x},{},"",{true,true,true});
  }
  if ((node.op == "exp2" || node.op == "exp") && p) {
    bool base2 = node.op == "exp2";
    Id exponent = node.args[0];
    return exponential(neg(exponent), base2);
  }
  // Only expose exponential factors when every denominator factor is known
  // finite and nonzero. Never cancel a possibly-zero or nonfinite denominator.
  if (node.op == "mul" && node.data[0] != 0 &&
      std::all_of(node.args.begin(), node.args.end(),
                  [&](Id x) { return nonzero(x); })) {
    std::vector<Id> other, powers;
    Q scalar = node.data[0];
    for (Id x : node.args) {
      const auto &factor = nodes[x];
      if ((factor.op == "exp2" || factor.op == "exp") &&
          nodes[factor.args[0]].facts.finite)
        powers.push_back(x);
      else other.push_back(x);
    }
    if (!powers.empty()) {
      other.push_back(constant(scalar));
      Id remaining = mul(other);
      std::vector<Id> inverse{recip(remaining)};
      for (Id x : powers) inverse.push_back(recip(x));
      ++stats["reciprocal_exponential_factors"];
      return mul(inverse);
    }
  }
  return intern("recip", {n}, {}, "", {p, p, p});
}

inline auto Arena::factors(Id n) -> Factors {
  const auto &x = nodes[n];
  if (x.op == "const")
    return {{}, x.data[0]};
  std::map<Id, int> fs;
  if (x.op == "mul") {
    for (Id a : x.args)
      ++fs[a];
    return {fs, x.data[0]};
  }
  return {{{n, 1}}, Q(1)};
}

inline auto Arena::common(const std::map<Id, int> &a,
                                const std::map<Id, int> &b) -> std::map<Id, int> {
  std::map<Id, int> r;
  for (auto [x, k] : a) {
    auto it = b.find(x);
    if (it != b.end())
      r[x] = std::min(k, it->second);
  }
  return r;
}

inline auto Arena::product(Factors f, const std::map<Id, int> &remove) -> Id {
  std::vector<Id> xs{constant(f.second)};
  for (auto [n, k] : f.first) {
    auto it = remove.find(n);
    if (it != remove.end())
      k -= it->second;
    for (int j = 0; j < k; ++j)
      xs.push_back(n);
  }
  return mul(xs);
}

inline auto Arena::divide(Id x, Id y) -> Id {
  if (x == y && nonzero(y))
    return one;
  auto af = factors(x), bf = factors(y);
  auto c = common(af.first, bf.first);
  for (auto it = c.begin(); it != c.end();)
    if (!nonzero(it->first))
      it = c.erase(it);
    else
      ++it;
  if (!c.empty()) {
    Id a = product(af, c), b = product(bf, c);
    if (nonzero(b)) {
      ++stats["cancel_nonzero_factor"];
      return mul({a, recip(b)});
    }
  }
  return mul({x, recip(y)});
}

inline auto Arena::factor_sum(Id n) -> Id {
  const auto &x = nodes[n];
  if (x.op != "add")
    return n;
  std::vector<Factors> ps;
  for (Id a : x.args)
    ps.push_back(factors(a));
  auto c = ps[0].first;
  for (size_t i = 1; i < ps.size(); ++i)
    c = common(c, ps[i].first);
  if (c.empty())
    return n;
  std::vector<Id> rs;
  for (size_t i = 0; i < ps.size(); ++i) {
    ps[i].second *= x.data[i];
    rs.push_back(product(ps[i], c));
  }
  return mul({product({c, Q(1)}), add(rs)});
}

} // namespace rg
