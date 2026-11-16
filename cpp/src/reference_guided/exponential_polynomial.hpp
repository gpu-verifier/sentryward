#include <unordered_set>
#pragma once
#include "core.hpp"
#include "alpha_identity.hpp"

namespace rg {
// Proof-local sparse normal forms. Arena remains compact and immutable; opaque
// nodes (reads, calls, folds, sqrt, integer operations) retain their identities.
struct ExponentialPolynomialBudget {
  size_t terms = 512, degree = 24, depth = 128, cached_terms = 32768;
  uint64_t work = 100000;
  size_t coefficient_bits = 4096, new_nodes = 20000;
};
class ExponentialPolynomial {
public:
  using Monomial = std::vector<Id>;
  using Polynomial = std::map<Monomial, Q>;
  struct Miss { std::string reason; };
  Arena &a;
  FoldAlphaIdentity alpha;
  ExponentialPolynomialBudget budget;
  std::string last_failure;
  uint64_t work = 0;
  bool profiling = false, reuse_cache = false;
  struct Timer {
    uint64_t *total; std::chrono::steady_clock::time_point start;
    Timer(uint64_t *p) : total(p) { if(total) start=std::chrono::steady_clock::now(); }
    ~Timer(){if(total)*total+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();}
  };
  struct Visits {
    uint64_t requests=0, unique_query_nodes=0, repeated_across_queries=0,
      previously_completed=0, shared_hits=0, computations=0;
    std::vector<uint64_t> seen, completed;
  };
  Visits poly_visits, normal_visits, fraction_visits;
  uint64_t query=0, equal_ns=0, reset_ns=0, polynomial_ns=0, fraction_ns=0, cross_ns=0,
    match_ns=0, instantiate_ns=0, body_scalar_ns=0, shared_evictions=0;
  J profile() const {
    auto visits=[](const Visits &v){return J{{"requests",v.requests},{"unique_query_nodes",v.unique_query_nodes},
      {"repeated_across_queries",v.repeated_across_queries},{"previously_completed",v.previously_completed},
      {"shared_hits",v.shared_hits},{"computations",v.computations}};};
    return {{"enabled",profiling},{"reuse_cache",reuse_cache},{"query_count",query},
      {"equal_s",equal_ns/1e9},{"reset_s",reset_ns/1e9},{"ordinary_polynomial_s",polynomial_ns/1e9},
      {"fraction_witness_s",fraction_ns/1e9},{"cross_product_s",cross_ns/1e9},
      {"match_total_s",match_ns/1e9},{"body_instantiation_s",instantiate_ns/1e9},
      {"body_scalar_s",body_scalar_ns/1e9},{"candidate_discovery_and_bookkeeping_s",(match_ns-instantiate_ns-body_scalar_ns)/1e9},
      {"polynomial",visits(poly_visits)},{"normal",visits(normal_visits)},
      {"fraction",visits(fraction_visits)},{"shared_evictions",shared_evictions}};
  }
private:
  std::set<Id> conditional_nodes, conditional_nonzero;
  size_t initial_nodes, stored_terms = 0;
  Id inverse_log2 = invalid;
  std::map<Id, Polynomial> cache;
  // Shared normal forms this query has charged to its budget, used in place: a shared entry is
  // never erased during a query (clear_shared runs only between queries, after reset_query),
  // so the reference stays valid, and the per-query budget is charged once, as a copy into
  // `cache` was. Saves copying (often large) polynomials into every query.
  std::unordered_map<Id, const Polynomial *> shared_refs;
  std::unordered_map<Id, Id> normal_cache;
  struct Fraction { Id numerator, denominator; };
  std::unordered_map<Id, Fraction> fraction_cache;
  std::map<Id,Polynomial> shared_poly;
  std::unordered_map<Id,Id> shared_normal;
  std::unordered_map<Id,Fraction> shared_fraction;
  size_t shared_terms=0;
  bool shared_full=false;
  void observe(Visits &v,Id id) {
    if(!profiling)return;
    ++v.requests;
    if(v.seen.size()<=id){v.seen.resize(size_t(id)+1);v.completed.resize(size_t(id)+1);}
    if(v.seen[id]!=query){++v.unique_query_nodes;if(v.seen[id])++v.repeated_across_queries;
      if(v.completed[id] && v.completed[id]<query)++v.previously_completed;
      v.seen[id]=query;
    }
  }
  void computed(Visits &v,Id id) {if(profiling){++v.computations;v.completed[id]=query;}}
  void clear_shared(){shared_poly.clear();shared_normal.clear();shared_fraction.clear();shared_terms=0;shared_full=false;}
  void tick() {
    a.check();
    if (++work > budget.work) throw Miss{"work"};
    if (a.nodes.size() - initial_nodes > budget.new_nodes) throw Miss{"new_nodes"};
  }
  void depth_check(size_t depth) {
    tick();
    if (depth > budget.depth) throw Miss{"depth"};
  }
  size_t bits(const Z &x) const {
    return x == 0 ? 0 : size_t(boost::multiprecision::msb(x < 0 ? -x : x)) + 1;
  }
  void coefficient(const Q &q) {
    if (bits(q.numerator()) > budget.coefficient_bits ||
        bits(q.denominator()) > budget.coefficient_bits) throw Miss{"coefficient_bits"};
  }
  void insert(Polynomial &p, Monomial key, const Q &value) {
    tick(); coefficient(value);
    if (key.size() > budget.degree) throw Miss{"degree"};
    if (value == 0) return;
    auto [it, added] = p.try_emplace(std::move(key), Q(0));
    (void)added;
    it->second += value;
    coefficient(it->second);
    if (it->second == 0) p.erase(it);
    if (p.size() > budget.terms) throw Miss{"terms"};
  }
  void add(Polynomial &to, const Polynomial &from, const Q &scale = Q(1)) {
    coefficient(scale);
    for (const auto &[key, value] : from) insert(to, key, scale * value);
  }
  // Every exponential in a product contributes its exponent to one sum per
  // base. exp and exp2 remain distinct; no approximate log(2) conversion. The
  // only base change is exact: a term carrying the ideal constant ln 2 (or
  // 1/ln 2) moves to the other base (Arena::exponent_bases), and ln 2 * (1/ln 2)
  // cancels.
  Monomial product_key(const Monomial &left, const Monomial &right, size_t depth) {
    tick();
    if (left.size() + right.size() > budget.degree) throw Miss{"degree"};
    Monomial result;
    std::vector<Id> natural, binary;
    for (const auto *part : {&left, &right}) for (Id id : *part) {
      const Node &n = a.nodes[id];
      if (n.op == "exp") natural.push_back(n.args[0]);
      else if (n.op == "exp2") binary.push_back(n.args[0]);
      else result.push_back(id);
    }
    for (bool cancelled = true; cancelled;) {
      const auto ln2 = std::find_if(result.begin(), result.end(), [&](Id f) { return a.is_natural_log2(f); });
      const auto inverse = std::find_if(result.begin(), result.end(), [&](Id f) { return a.is_inverse_natural_log2(f); });
      cancelled = ln2 != result.end() && inverse != result.end();
      if (cancelled) {
        const size_t i = size_t(ln2 - result.begin()), j = size_t(inverse - result.begin());
        result.erase(result.begin() + std::max(i, j));
        result.erase(result.begin() + std::min(i, j));
      }
    }
    Id natural_exponent = natural.empty() ? a.zero : exponent_normal(a.add(natural), depth + 1);
    Id binary_exponent = binary.empty() ? a.zero : exponent_normal(a.add(binary), depth + 1);
    if (auto bases = a.exponent_bases(natural_exponent, binary_exponent)) {
      natural_exponent = exponent_normal(bases->first, depth + 1);
      binary_exponent = exponent_normal(bases->second, depth + 1);
    }
    auto append = [&](Id exponent, bool base2) {
      if (exponent == a.zero) return;
      Id value = a.exponential(exponent, base2);
      const Node &v = a.nodes[value];
      // A residual base change returns exp(.)*exp2(.): keep each as its own factor.
      if (v.op == "mul" && v.data[0] == 1 && std::all_of(v.args.begin(), v.args.end(), [&](Id f) {
            return a.nodes[f].op == "exp" || a.nodes[f].op == "exp2"; })) {
        const auto factors = v.args;
        for (Id f : factors) result.push_back(alpha.representative(f));
      } else if (value != a.one) result.push_back(alpha.representative(value));
    };
    append(natural_exponent, false); append(binary_exponent, true);
    std::sort(result.begin(), result.end());
    return result;
  }
  Polynomial multiply(const Polynomial &left, const Polynomial &right, size_t depth) {
    Polynomial result;
    for (const auto &[lk, lv] : left) for (const auto &[rk, rv] : right)
      insert(result, product_key(lk, rk, depth), lv * rv);
    return result;
  }
  Id emit(const Polynomial &p) {
    std::vector<Id> terms;
    terms.reserve(p.size());
    for (const auto &[key, value] : p) {
      tick();
      std::vector<Id> factors(key);
      factors.push_back(a.constant(value));
      terms.push_back(a.mul(std::move(factors)));
    }
    return a.add(std::move(terms));
  }
  Id normal(Id root, size_t depth) {
    depth_check(depth);
    observe(normal_visits,root);
    if (auto it = normal_cache.find(root); it != normal_cache.end()) return it->second;
    if(reuse_cache)if(auto it=shared_normal.find(root);it!=shared_normal.end()){
      ++normal_visits.shared_hits;normal_cache.emplace(root,it->second);return it->second;}
    Id result = emit(polynomial(root, depth + 1));
    normal_cache.emplace(root, result);computed(normal_visits,root);
    if(reuse_cache && !shared_full){if(shared_normal.size()<budget.cached_terms)shared_normal.emplace(root,result);else shared_full=true;}
    return result;
  }
  bool nonzero(Id id) {
    const auto &n = a.nodes[id];
    return (!conditional_nonzero.empty() && conditional_nonzero.contains(alpha.representative(id))) || a.nonzero(id) || (n.facts.finite && n.bounds &&
                            (n.bounds->first > 0 || n.bounds->second < 0));
  }
  // Preserve a compact fraction witness: repeated prefix construction must
  // not flatten/copy all earlier factors or summands (quadratic on a DAG chain).
  Id compact_product(Id x, Id y) {
    tick();
    if (x == a.one) return y;
    if (y == a.one) return x;
    return a.intern("mul", {x,y}, {Q(1)}, "", {true,false,false});
  }
  Id compact_sum(Id x, Id y) {
    tick();
    return a.intern("add_raw", {x,y}, {Q(1),Q(1)}, "", {true,false,false});
  }
  struct ExponentialForm {
    Id remainder;
    std::vector<Id> numerator, denominator;
  };
  // Exact same-base log/exp cancellation inside an additive exponent. Scan
  // only its bounded scalar normal form, never the logarithm's argument DAG.
  // Integer coefficients become repeated factors under the existing degree
  // budget; a fractional coefficient or different log base remains opaque.
  ExponentialForm exponential_form(Id exponent, bool base2, size_t depth) {
    const Id normalized = exponent_normal(exponent, depth + 1);
    const Node &n = a.nodes[normalized]; // Arena owns stable deque records.
    const bool sum = n.op == "add" || n.op == "add_raw";
    const size_t arity = sum ? n.args.size() : 1;
    if (arity > budget.terms) throw Miss{"terms"};
    ExponentialForm result{a.one,{},{}};
    std::vector<Id> residual;
    for (size_t j = 0; j < arity; ++j) {
      tick();
      const Id original = sum ? n.args[j] : normalized;
      const Q scale = sum ? n.data[j] : Q(1);
      Id term = original; Q coefficient = scale;
      if (a.nodes[term].op == "mul" && a.nodes[term].args.size() == 1) {
        coefficient *= a.nodes[term].data[0]; term = a.nodes[term].args[0];
      }
      const char *logop = base2 ? "log2" : "log";
      if (base2) if (Id natural = natural_log_of_base2_term(term, coefficient); natural != invalid) {
        term = natural; logop = "log";
      }
      const auto &candidate = a.nodes[term];
      if (candidate.op == logop && candidate.args.size() == 1 &&
          candidate.facts.finite && a.nodes[candidate.args[0]].facts.finite &&
          a.nodes[candidate.args[0]].facts.pos && coefficient.denominator() == 1) {
        const Q absolute = coefficient < 0 ? -coefficient : coefficient;
        if (absolute > Q(budget.degree) ||
            Q(result.numerator.size()+result.denominator.size())+absolute > Q(budget.degree))
          throw Miss{"degree"};
        const Id argument = candidate.args[0];
        const size_t count = absolute.numerator().convert_to<size_t>();
        auto &factors = coefficient < 0 ? result.denominator : result.numerator;
        for (size_t k = 0; k < count; ++k) { tick(); factors.push_back(argument); }
      } else residual.push_back(a.mul({a.constant(scale),original}));
    }
    result.remainder = a.exponential(a.add(std::move(residual)),base2);
    return result;
  }
  // 1/log(2), the factor of every base-2 logarithm in the normal form.
  Id inverse_natural_log2() {
    if (inverse_log2 == invalid) inverse_log2 = a.inverse_natural_log2();
    return inverse_log2;
  }
  // Rule: for finite positive factors, log(k * f_1 * ... * f_n) = log k + sum_i log f_i, with
  // log(exp z) = z, log(exp2 z) = z * ln 2 and log(2^j) = j * ln 2, all exact real identities. Only an
  // argument whose normal form is one monomial with a positive coefficient and factors each proved
  // finite and positive is split; any other argument keeps its logarithm atom. Returns the natural
  // logarithm. O(monomial degree + exponent normal forms).
  std::optional<Polynomial> logarithm_of_product(Id argument, size_t depth) {
    const Polynomial argument_polynomial = polynomial(argument, depth + 1);
    if (argument_polynomial.size() != 1) return {};
    const auto [key, coefficient] = *argument_polynomial.begin();
    if (coefficient <= 0) return {};
    for (Id f : key) {
      const Node &n = a.nodes[f];
      const bool exponential = (n.op == "exp" || n.op == "exp2") && n.args.size() == 1 &&
                               a.nodes[n.args[0]].facts.finite;
      if (!exponential && (!n.facts.finite || !n.facts.pos)) return {};
    }
    Polynomial p;
    const Id ln2 = alpha.representative(a.natural_log2());
    const auto power_of_two = [](const Z &x) { return x > 0 && (x & (x - 1)) == 0; };
    const Z numerator = coefficient.numerator(), denominator = coefficient.denominator();
    if (coefficient != 1) {
      if (denominator == 1 && power_of_two(numerator))
        insert(p, {ln2}, Q(Z(boost::multiprecision::msb(numerator))));
      else if (numerator == 1 && power_of_two(denominator))
        insert(p, {ln2}, -Q(Z(boost::multiprecision::msb(denominator))));
      else insert(p, {alpha.representative(a.logarithm(a.constant(coefficient)))}, Q(1));
    }
    for (Id f : key) {
      tick();
      const Node &n = a.nodes[f];
      if (n.op == "exp") add(p, polynomial(n.args[0], depth + 1));
      else if (n.op == "exp2") {
        Polynomial scale; insert(scale, {ln2}, Q(1));
        add(p, multiply(polynomial(n.args[0], depth + 1), scale, depth + 1));
      } else insert(p, {alpha.representative(a.logarithm(f))}, Q(1));
    }
    ++a.stats["logarithm_product_splits"];
    return p;
  }
  // The normal form of c*log2(x) is c*log(x)*(1/log 2) (logarithm_polynomial).
  // Returns log(x) and folds c into the coefficient; invalid otherwise.
  Id natural_log_of_base2_term(Id term, Q &coefficient) {
    const Node &n = a.nodes[term];
    if (n.op != "mul" || n.args.size() != 2) return invalid;
    const Id inverse = inverse_natural_log2();
    for (unsigned i = 0; i < 2; ++i)
      if (n.args[1 - i] == inverse && a.nodes[n.args[i]].op == "log") {
        coefficient *= n.data[0];
        return n.args[i];
      }
    return invalid;
  }
  // A logarithm of a finite positive x is keyed by the normal form of x, and
  // log2(x) is the monomial log(x)*(1/log 2), so equal arguments of either
  // shape and either base meet. Both rewrites are exact real identities. An
  // argument whose normal form leaves the bounded fragment, or does not keep
  // its proved positivity, keeps its own atom. O(normal form of x), once per
  // logarithm node and query.
  std::optional<Polynomial> logarithm_polynomial(Id root, size_t depth) {
    const Node &n = a.nodes[root];
    if ((n.op != "log" && n.op != "log2") || n.args.size() != 1 || !n.facts.finite) return {};
    const bool base2 = n.op == "log2";
    Id argument = n.args[0];
    try {
      if (auto split = logarithm_of_product(argument, depth)) {
        if (!base2) return split;
        Polynomial inverse; insert(inverse, {alpha.representative(inverse_natural_log2())}, Q(1));
        return multiply(*split, inverse, depth + 1);
      }
    } catch (const Miss &) {}
    try {
      const Id normalized = normal(argument, depth + 1);
      if (a.nodes[normalized].facts.finite && a.nodes[normalized].facts.pos) argument = normalized;
    } catch (const Miss &) {}
    Polynomial p;
    const Id natural = a.logarithm(argument);
    if (natural == a.zero) return p;
    Monomial key{alpha.representative(natural)};
    if (base2) key.push_back(alpha.representative(inverse_natural_log2()));
    std::sort(key.begin(), key.end());
    insert(p, std::move(key), Q(1));
    return p;
  }
  // Used only by exponent keys and a failed scalar comparison. This is not
  // the stored representation of every expression. No gcd/factorization.
  Fraction fraction(Id root, size_t depth) {
    depth_check(depth);
    observe(fraction_visits,root);
    if (auto it = fraction_cache.find(root); it != fraction_cache.end()) return it->second;
    if(reuse_cache)if(auto it=shared_fraction.find(root);it!=shared_fraction.end()){
      ++fraction_visits.shared_hits;fraction_cache.emplace(root,it->second);return it->second;}
    const Node &n = a.nodes[root];
    if (!n.facts.finite && !conditional_nodes.contains(root)) throw Miss{"nonfinite_or_undefined"};
    Fraction result{root, a.one};
    if (n.op == "recip") {
      if (!nonzero(n.args[0])) throw Miss{"denominator_nonzero"};
      Fraction inner = fraction(n.args[0], depth + 1);
      result = {inner.denominator, inner.numerator};
    } else if (n.op == "mul") {
      result = {a.constant(n.data[0]), a.one};
      for (Id child : n.args) {
        tick(); auto f = fraction(child, depth + 1);
        result = {compact_product(result.numerator, f.numerator),
                  compact_product(result.denominator, f.denominator)};
      }
    } else if (n.op == "add" || n.op == "add_raw") {
      result = {a.zero, a.one};
      for (size_t j = 0; j < n.args.size(); ++j) {
        tick(); auto f = fraction(n.args[j], depth + 1);
        Id numerator = compact_product(a.constant(n.data[j]), f.numerator);
        if (result.numerator == a.zero) result = {numerator, f.denominator};
        else if (result.denominator == f.denominator)
          result.numerator = compact_sum(result.numerator, numerator);
        else result = {compact_sum(compact_product(result.numerator, f.denominator),
                                   compact_product(numerator, result.denominator)),
                       compact_product(result.denominator, f.denominator)};
      }
    } else if (n.op == "exp" || n.op == "exp2") {
      const bool base2 = n.op == "exp2";
      auto factors = exponential_form(n.args[0],base2,depth+1);
      result = {factors.remainder,a.one};
      for (Id factor : factors.numerator) {
        auto f = fraction(factor,depth+1);
        result = {compact_product(result.numerator,f.numerator),compact_product(result.denominator,f.denominator)};
      }
      for (Id factor : factors.denominator) {
        auto f = fraction(factor,depth+1);
        result = {compact_product(result.numerator,f.denominator),compact_product(result.denominator,f.numerator)};
      }
    }
    fraction_cache.emplace(root, result);computed(fraction_visits,root);
    if(reuse_cache && !shared_full){if(shared_fraction.size()<budget.cached_terms)shared_fraction.emplace(root,result);else shared_full=true;}
    return result;
  }
  Id exponent_normal(Id root, size_t depth) {
    depth_check(depth);
    // Canonical numerator/denominator provides common-denominator identities
    // without an unbounded all-pairs equivalence search over exponent keys.
    Fraction f = fraction(root, depth + 1);
    Id numerator = normal(f.numerator, depth + 1);
    if (f.denominator == a.one) return numerator;
    Id denominator = normal(f.denominator, depth + 1);
    // The original fraction checks established this denominator is nonzero;
    // preserve that fact locally even when expansion loses positivity facts.
    if (numerator == a.zero) return a.zero;
    if (numerator == denominator) return a.one;
    if (auto c = a.value(denominator)) {
      if (*c == 0) throw Miss{"zero_denominator"};
      return normal(a.mul({numerator, a.constant(Q(1) / *c)}), depth + 1);
    }
    Id inverse = a.intern("recip", {denominator}, {}, "", {true, false, false});
    return normal(a.mul({numerator, inverse}), depth + 1);
  }
  // A Select whose arms share a constant factor k (both k*x and k*y, or one of them and a constant):
  // (k, the Select of the unscaled arms). nullopt otherwise (and for k = 0 or 1). O(1).
  std::optional<std::pair<Q, Id>> common_arm_scale(Id root) {
    const auto &n = a.nodes[root];
    if (n.op != "select" || n.args.size() != 3) return std::nullopt;
    const auto scale = [&](Id x) -> std::optional<Q> {
      const auto &m = a.nodes[x];
      if (m.op == "mul" && m.data.size() == 1 && m.data[0] != 0 && m.data[0] != 1 && !m.args.empty()) return m.data[0];
      return std::nullopt;
    };
    const Id c = n.args[0], x = n.args[1], y = n.args[2];
    const auto kx = scale(x), ky = scale(y);
    const auto vx = a.value(x), vy = a.value(y);
    std::optional<Q> k;
    if (kx && ky) { if (*kx == *ky) k = kx; }
    else if (kx && vy) k = kx;
    else if (ky && vx) k = ky;
    if (!k) return std::nullopt;
    const auto unscaled = [&](Id z) {
      if (const auto v = a.value(z)) return a.constant(*v / *k);
      const auto args = a.nodes[z].args;
      return a.mul(args);
    };
    const Id ux = unscaled(x), uy = unscaled(y);
    // Only for arms built without -infinity: no -infinity node within the arms (Folds and Calls are not
    // entered; at most 256 nodes visited, beyond that no rule). Where a pivot is built from values that may be
    // -infinity (masked rows), the chain lemmas of the linear-reduction pattern reason about its spellings.
    {
      std::vector<Id> todo{ux, uy}; std::unordered_set<Id> seen;
      while (!todo.empty()) {
        const Id z = todo.back(); todo.pop_back();
        if (!seen.insert(z).second) continue;
        if (seen.size() > 256 || a.neginf(z)) return std::nullopt;
        const auto &m = a.nodes[z];
        if (m.op == "fold" || m.op == "call") continue;
        todo.insert(todo.end(), m.args.begin(), m.args.end());
      }
    }
    const Id inner = a.select(c, ux, uy);
    if (!a.nodes[inner].facts.finite) return std::nullopt;
    return std::pair(*k, inner);
  }
  const Polynomial &polynomial(Id root, size_t depth) {
    depth_check(depth);
    observe(poly_visits,root);
    if (auto it = cache.find(root); it != cache.end()) return it->second;
    if (auto it = shared_refs.find(root); it != shared_refs.end()) return *it->second;
    if(reuse_cache)if(auto it=shared_poly.find(root);it!=shared_poly.end()){
      if(stored_terms+it->second.size()>budget.cached_terms)throw Miss{"cached_terms"};
      stored_terms+=it->second.size();++poly_visits.shared_hits;
      shared_refs.emplace(root,&it->second);
      return it->second;
    }
    const Node &n = a.nodes[root];
    if (!n.facts.finite && !conditional_nodes.contains(root)) throw Miss{"nonfinite_or_undefined"};
    Polynomial p;
    if (n.op == "const") insert(p, {}, n.data[0]);
    else if (n.op == "add" || n.op == "add_raw") {
      for (size_t j = 0; j < n.args.size(); ++j)
        add(p, polynomial(n.args[j], depth + 1), n.data[j]);
    } else if (n.op == "mul") {
      insert(p, {}, n.data[0]);
      for (Id child : n.args) p = multiply(p, polynomial(child, depth + 1), depth + 1);
    } else if (n.op == "exp" || n.op == "exp2") {
      const bool base2 = n.op == "exp2";
      auto factors = exponential_form(n.args[0],base2,depth+1);
      insert(p, factors.remainder == a.one ? Monomial{} : Monomial{alpha.representative(factors.remainder)}, Q(1));
      for (Id factor : factors.numerator) p = multiply(p,polynomial(factor,depth+1),depth+1);
      for (Id factor : factors.denominator) {
        Polynomial inverse; insert(inverse,{alpha.representative(a.recip(factor))},Q(1));
        p = multiply(p,inverse,depth+1);
      }
    } else if (auto logarithm = logarithm_polynomial(root, depth)) {
      p = std::move(*logarithm);
    } else if (auto pulled = common_arm_scale(root)) {
      // Rule (common scale of the arms): select(c, k*x, k*y) = k*select(c, x, y), and
      // select(c, k*x, v) = k*select(c, x, v/k) (or mirrored) for constants k (not 0, 1) and v: pointwise the
      // Select takes one arm, which is k times that arm of the unscaled Select. The normal form keeps the
      // unscaled Select as the atom with coefficient k, so a pivot kept in score units and in scaled units
      // (k*select(c, m, p) and select(c, k*m, k*p)) is one atom. Only the normal form changes; the
      // expression keeps its spelling. Exact. O(1) plus the unscaled Select's constructor.
      insert(p, {}, pulled->first);
      p = multiply(p, polynomial(pulled->second, depth + 1), depth + 1);
      ++a.stats["ep_select_common_scales"];
    } else {
      // Reciprocal is an opaque factor in the ordinary normal form. Its
      // original domain remains intact; fraction() handles crossing later.
      insert(p, {alpha.representative(root)}, Q(1));
    }
    if (stored_terms + p.size() > budget.cached_terms) throw Miss{"cached_terms"};
    stored_terms += p.size();computed(poly_visits,root);
    if(reuse_cache && !shared_full && !shared_poly.contains(root)) {
      if(shared_terms+p.size()<=budget.cached_terms && shared_poly.size()<budget.cached_terms) {
        // Stored once, shared; this query uses it in place (shared_refs).
        shared_terms+=p.size();
        const auto &stored=shared_poly.emplace(root,std::move(p)).first->second;
        shared_refs.emplace(root,&stored);
        return stored;
      }
      else shared_full=true;
    }
    return cache.emplace(root, std::move(p)).first->second;
  }
public:
  explicit ExponentialPolynomial(Arena &arena, ExponentialPolynomialBudget b = {})
      : a(arena), alpha(arena), budget(b), initial_nodes(arena.nodes.size()) {}
  void reset() {clear_shared();reset_query();}
  void reset_query() {
    cache.clear(); shared_refs.clear(); normal_cache.clear(); fraction_cache.clear();
    stored_terms=0; work=0; initial_nodes=a.nodes.size(); last_failure.clear();
  }
  // Canonicalization is bounded; equal() starts fresh per-query budgets/caches.
  Id canonicalize(Id root) {
    last_failure.clear();
    try { return normal(root, 0); }
    catch (const Miss &m) {
      last_failure = m.reason; ++a.stats["ep_decline_" + m.reason]; return root;
    }
  }
  struct Domain {
    bool supported = true;
    std::set<Id> nonzero, conditional;
  };
  // Bounded strict arithmetic fragment. Total subexpressions are opaque.
  // Unknown partial operators, conditional branches and partial denominators
  // decline rather than guessing their domains.
  Domain domain(Id root) {
    Domain result;
    std::vector<std::pair<Id, size_t>> todo{{root, 0}};
    std::unordered_set<Id> seen;
    uint64_t visits = 0;
    while (!todo.empty()) {
      auto [id, depth] = todo.back(); todo.pop_back();
      a.check();
      if (++visits > budget.work || depth > budget.depth) { result.supported = false; break; }
      if (!seen.insert(id).second) continue;
      const auto &n = a.nodes[id];
      if (n.facts.finite) continue;
      result.conditional.insert(id);
      if (n.op == "recip" && n.args.size() == 1 && a.nodes[n.args[0]].facts.finite) {
        result.nonzero.insert(alpha.representative(n.args[0]));
      } else if (n.op == "add" || n.op == "add_raw" || n.op == "mul") {
        if (n.args.size() > budget.work - visits) { result.supported = false; break; }
        for (Id child : n.args) todo.emplace_back(child, depth + 1);
      } else { result.supported = false; break; }
    }
    return result;
  }
  // This proves equality as partial functions, not global definedness.
  // Conditional caches never escape into the ordinary total-value engine.
  bool equal_partial(Id left, Id right) {
    auto l = domain(left), r = domain(right);
    if (!l.supported || !r.supported || l.nonzero != r.nonzero) {
      last_failure = "partial_domain_mismatch_or_unsupported"; return false;
    }
    struct Cleanup {
      ExponentialPolynomial &ep;
      ~Cleanup() { ep.conditional_nodes.clear(); ep.conditional_nonzero.clear(); ep.reset(); }
    } cleanup{*this};
    reset();
    conditional_nodes = std::move(l.conditional);
    conditional_nodes.insert(r.conditional.begin(), r.conditional.end());
    conditional_nonzero = std::move(l.nonzero);
    bool result = equal(left, right);
    if (result) ++a.stats["ep_partial_function_proofs"];
    return result;
  }
  bool equal(Id left, Id right) {
    Timer total(profiling?&equal_ns:nullptr);
    {Timer timer(profiling?&reset_ns:nullptr);reset_query();
      if(!reuse_cache)clear_shared();else if(shared_full){clear_shared();++shared_evictions;}}
    ++query; ++a.stats["ep_scalar_attempts"];
    try {
      // Even identity must not establish definedness of a nonfinite value.
      if ((!a.nodes[left].facts.finite && !conditional_nodes.contains(left)) ||
          (!a.nodes[right].facts.finite && !conditional_nodes.contains(right)))
        throw Miss{"nonfinite_or_undefined"};
      if (left == right) {
        ++a.stats["ep_identity_proofs"]; ++a.stats["ep_scalar_proofs"]; return true;
      }
      bool same;
      {Timer timer(profiling?&polynomial_ns:nullptr);same=polynomial(left,0)==polynomial(right,0);}
      if (same) {
        ++a.stats["ep_polynomial_proofs"]; ++a.stats["ep_scalar_proofs"]; return true;
      }
      Fraction l,r;
      {Timer timer(profiling?&fraction_ns:nullptr);l=fraction(left,0);r=fraction(right,0);}
      if (l.denominator == a.one && r.denominator == a.one) {
        last_failure = "nonzero_normal_form"; return false;
      }
      ++a.stats["ep_cross_products"];
      Timer cross_timer(profiling?&cross_ns:nullptr);
      auto lp = multiply(polynomial(l.numerator, 0), polynomial(r.denominator, 0), 0);
      auto rp = multiply(polynomial(r.numerator, 0), polynomial(l.denominator, 0), 0);
      if (lp == rp) { ++a.stats["ep_fraction_proofs"]; ++a.stats["ep_scalar_proofs"]; return true; }
      last_failure = "nonzero_cross_product"; return false;
    } catch (const Miss &m) {
      last_failure = m.reason; ++a.stats["ep_decline_" + m.reason]; return false;
    }
  }
};
} // namespace rg
