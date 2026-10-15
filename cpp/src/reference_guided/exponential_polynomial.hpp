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
  std::map<Id, Polynomial> cache;
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
  // base. exp and exp2 remain distinct; no approximate log(2) conversion.
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
    auto append = [&](const std::vector<Id> &exponents, bool base2) {
      if (exponents.empty()) return;
      Id exponent = exponent_normal(a.add(exponents), depth + 1);
      Id value = a.exponential(exponent, base2);
      if (value != a.one) result.push_back(alpha.representative(value));
    };
    append(natural, false); append(binary, true);
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
      const auto &candidate = a.nodes[term];
      if (candidate.op == (base2 ? "log2" : "log") && candidate.args.size() == 1 &&
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
  const Polynomial &polynomial(Id root, size_t depth) {
    depth_check(depth);
    observe(poly_visits,root);
    if (auto it = cache.find(root); it != cache.end()) return it->second;
    if(reuse_cache)if(auto it=shared_poly.find(root);it!=shared_poly.end()){
      if(stored_terms+it->second.size()>budget.cached_terms)throw Miss{"cached_terms"};
      stored_terms+=it->second.size();++poly_visits.shared_hits;
      return cache.emplace(root,it->second).first->second;
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
    } else {
      // Reciprocal is an opaque factor in the ordinary normal form. Its
      // original domain remains intact; fraction() handles crossing later.
      insert(p, {alpha.representative(root)}, Q(1));
    }
    if (stored_terms + p.size() > budget.cached_terms) throw Miss{"cached_terms"};
    stored_terms += p.size();computed(poly_visits,root);
    if(reuse_cache && !shared_full && !shared_poly.contains(root)) {
      if(shared_terms+p.size()<=budget.cached_terms && shared_poly.size()<budget.cached_terms){shared_terms+=p.size();shared_poly.emplace(root,p);}
      else shared_full=true;
    }
    return cache.emplace(root, std::move(p)).first->second;
  }
public:
  explicit ExponentialPolynomial(Arena &arena, ExponentialPolynomialBudget b = {})
      : a(arena), alpha(arena), budget(b), initial_nodes(arena.nodes.size()) {}
  void reset() {clear_shared();reset_query();}
  void reset_query() {
    cache.clear(); normal_cache.clear(); fraction_cache.clear();
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
