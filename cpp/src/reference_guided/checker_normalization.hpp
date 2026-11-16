#pragma once
#include "checker.hpp"

namespace rg {

// log(x) * (1/log 2) is log2(x) for finite positive x (a finite log), the
// shape PTX lg2 imports to. Both sides share this canonical form. Only a
// product with exactly one finite natural log and one 1/log(2) factor is
// contracted. O(factors).
inline auto Checker::base2_logarithm_product(const std::vector<Id> &args, const Q &scalar) -> std::optional<Id> {
  size_t inverse = args.size(), logarithm = args.size(), inverses = 0, logarithms = 0;
  for (size_t i = 0; i < args.size(); ++i) {
    const auto &n = a.nodes[args[i]];
    if (n.op == "recip" && n.args.size() == 1 && a.nodes[n.args[0]].op == "log" &&
        a.value(a.nodes[n.args[0]].args[0]) == Q(2)) { inverse = i; ++inverses; }
    else if (n.op == "log" || n.op == "log2") { logarithm = i; ++logarithms; }
  }
  if (inverses != 1 || logarithms != 1 || a.nodes[args[logarithm]].op != "log" ||
      !a.nodes[args[logarithm]].facts.finite) return {};
  std::vector<Id> xs{a.constant(scalar), a.logarithm(a.nodes[args[logarithm]].args[0], true)};
  for (size_t i = 0; i < args.size(); ++i)
    if (i != inverse && i != logarithm) xs.push_back(args[i]);
  ++a.stats["canonical_base2_logarithms"];
  return a.mul(std::move(xs));
}

// strict: the rebuilt node must take its facts from its new operands. Where the rebuild would
// give it the original node's facts or bounds (the floor branch and the generic fallback) and an
// operand changed, nothing is built and `invalid` is returned. For a caller whose new operands
// equal the old ones only under an assumption (the case analysis of the linear reduction).
inline auto Checker::reapply(Id original, const std::vector<Id> &args, bool strict) -> Id {
  const auto &n = a.nodes[original];
  const auto &op = n.op;
  const auto &d = n.data;
  if (op == "call") {
    if(d.size()!=1 || d[0].denominator()!=1 || d[0]<0 || d[0]>=Q(specs.size()))
      throw Unsupported("reduction Call reconstruction identity");
    Id sid=d[0].numerator().convert_to<Id>();
    if(args.size()+1!=specs[sid].slots)
      throw Unsupported("reduction Call reconstruction arity");
    return call(sid,args);
  }
  if (op == "fold" && args.size() == 2 && d.size() == 2 &&
      d[0].denominator() == 1 && d[1].denominator() == 1 && d[1] >= 0 &&
      (n.tag == "Sum" || n.tag == "Max")) {
    return canonicalize_fold(original, args, strict);
  }
  if (op == "add" || op == "add_raw") {
    std::vector<Id> xs;
    for (size_t j = 0; j < args.size(); ++j)
      xs.push_back(a.mul({a.constant(d[j]), args[j]}));
    return a.factor_sum(a.add(xs));
  }
  if (op == "mul")
    if (auto contracted = base2_logarithm_product(args, d[0])) return *contracted;
  if (op == "mul") {
    std::vector<Id> rec, others{a.constant(d[0])};
    for (Id x : args)
      if (a.nodes[x].op == "recip")
        rec.push_back(a.nodes[x].args[0]);
      else
        others.push_back(x);
    if (rec.size() == 1)
      return a.divide(a.mul(others), rec[0]);
    auto xs = args;
    xs.push_back(a.constant(d[0]));
    return a.mul(xs);
  }
  if (op == "signed") {
    if(args.size()!=1 || d.size()!=1 || d[0].denominator()!=1 || d[0]<=0 || d[0]>256)
      throw Unsupported("signed word reconstruction arity/width");
    return a.signed_word(args[0],d[0].numerator().convert_to<unsigned>());
  }
  if (op == "word_or")
    return a.word_or(args[0],args[1],d[0].numerator().convert_to<unsigned>());
  if (op == "word_shr")
    return a.word_shr(args[0],args[1],d[0].numerator().convert_to<unsigned>(),d[1]!=0);
  if (op == "word_shl" && args.size() == 2 && d.size() == 1)
    return a.word_shl(args[0],args[1],d[0].numerator().convert_to<unsigned>());
  if (op == "extractbits")
    return a.bit_slice(args[0],d[0].numerator().convert_to<unsigned>(),d[1].numerator().convert_to<unsigned>());
  if ((op == "idiv" || op == "imod") && args.size() == 1 &&
      d.size() == 1 && d[0].denominator() == 1 && d[0] > 0)
    return op == "idiv" ? a.idiv(args[0], d[0].numerator())
                        : a.imod(args[0], d[0].numerator());
  if (op == "floor") {
    if (strict && args != n.args) return invalid;
    return a.floor_value(args[0],n.bounds);
  }
  if (op == "exp2" || op == "exp")
    return a.exponential(args[0], op == "exp2");
  if (op == "select")
    return equal_arm_select(a.select(args[0], args[1], args[2]));
  if (op == "compare")
    return a.compare(n.tag, args[0], args[1]);
  if (op == "xor")
    return a.bitxor(args.at(0), args.at(1));
  if (op == "abs")
    return a.absolute(args.at(0));
  if (op == "log" || op == "log2")
    return a.logarithm(args.at(0), op == "log2");
  if (op == "power")
    return a.power(args.at(0), args.at(1));
  if (op == "recip")
    return a.recip(args[0]);
  if (op == "max")
    return a.maximum(args);
  if (op == "clamp")
    return a.clamp(args[0],d[0],d[1]);
  if (op == "sqrt" || op == "reciprocalsqrt" || op == "sin" || op == "cos" ||
      op == "exp" || op == "tanh") {
    bool f = std::all_of(args.begin(), args.end(),
                         [&](Id x) { return a.nodes[x].facts.finite; });
    if (op == "sqrt") f &= a.nodes[args[0]].facts.nn;
    if (op == "reciprocalsqrt") f &= a.nodes[args[0]].facts.pos;
    bool p = f && (op == "reciprocalsqrt" || op == "exp" ||
                          (op == "sqrt" && a.nodes[args[0]].facts.pos));
    bool nn = p || (f && op == "sqrt");
    return a.intern(op, args, d, n.tag, {f, p, nn});
  }
  if (args == n.args) return original;
  if (strict) return invalid;
  return a.intern(op, args, d, n.tag, n.facts, n.bounds);
}

// Cached free-variable sets make each independence query logarithmic in the
// factor's free-variable count; this never traverses its producer DAG.
inline auto Checker::split_invariant_factors(Id body, Id axis) -> std::pair<Id, Id> {
  const auto &n = a.nodes[body];
  if (!n.facts.finite) return {a.one, body};
  if (!a.depends(body, axis)) return {body, a.one};
  if (n.op != "mul" || n.data.size() != 1) return {a.one, body};
  std::vector<Id> invariant{a.constant(n.data[0])}, varying;
  for (Id factor : n.args) {
    a.check();
    ++a.stats["canonical_factor_visits"];
    (a.depends(factor, axis) ? varying : invariant).push_back(factor);
  }
  return {a.mul(invariant), a.mul(varying)};
}

// Capture-safe restriction of one free coordinate to a subset of its
// declared domain. Invariant subgraphs are reused. Facts on immutable reads
// remain valid under restriction; coordinate constructors recompute bounds.
// The iteration domain is dependent stored DAG nodes, never Fold elements.
// root with the free variable `var` replaced by `value`, where `value` may lie outside var's declared
// domain: every node that depends on var is rebuilt by reapply in strict mode (a constructor that
// derives facts and bounds from the new operands); invalid where a node would copy facts or bounds
// proved for var's domain (a floor, a read, any generic operator). A Fold binding var is not entered.
// Not memoized in restricted_axis_cache. O(nodes that depend on var).
inline auto Checker::substitute_strict(Id root, Id var, Id value) -> Id {
  std::unordered_map<Id, Id> memo;
  std::vector<std::pair<Id, bool>> todo{{root, false}};
  while (!todo.empty()) {
    a.check();
    auto [id, ready] = todo.back(); todo.pop_back();
    if (memo.contains(id)) continue;
    if (id == var) { memo.emplace(id, value); continue; }
    if (!a.depends(id, var)) { memo.emplace(id, id); continue; }
    const auto &n = a.nodes[id];
    const bool shadow = n.op == "fold" && n.args.size() == 2 && n.data.size() == 2 && n.data[0] == Q(var);
    if (!ready) {
      todo.emplace_back(id, true);
      for (size_t j = 0; j < n.args.size(); ++j)
        if (!(shadow && j == 1) && !memo.contains(n.args[j])) todo.emplace_back(n.args[j], false);
      continue;
    }
    std::vector<Id> args;
    args.reserve(n.args.size());
    for (size_t j = 0; j < n.args.size(); ++j) args.push_back(shadow && j == 1 ? n.args[j] : memo.at(n.args[j]));
    const Id rebuilt = reapply(id, args, /*strict=*/true);
    if (rebuilt == invalid) { ++a.stats["strict_substitution_declines"]; return invalid; }
    memo.emplace(id, rebuilt);
  }
  return memo.at(root);
}

inline auto Checker::restrict_axis(Id root, Id old_axis, Id new_axis) -> Id {
  auto key = [&](Id id) { return std::tuple(old_axis, new_axis, id); };
  std::vector<std::pair<Id, bool>> todo{{root, false}};
  while (!todo.empty()) {
    a.check();
    auto [id, ready] = todo.back(); todo.pop_back();
    if (restricted_axis_cache.contains(key(id))) {
      ++a.stats["canonical_domain_rebind_cache_hits"];
      continue;
    }
    if (id == old_axis) {
      restricted_axis_cache.emplace(key(id), new_axis);
      continue;
    }
    if (!a.depends(id, old_axis)) {
      restricted_axis_cache.emplace(key(id), id);
      ++a.stats["canonical_domain_invariant_reuse"];
      continue;
    }
    const auto &n = a.nodes[id];
    const bool shadow = n.op == "fold" && n.args.size() == 2 &&
                        n.data.size() == 2 && n.data[0] == Q(old_axis);
    if (!ready) {
      todo.emplace_back(id, true);
      for (size_t j = 0; j < n.args.size(); ++j)
        if (!(shadow && j == 1) && !restricted_axis_cache.contains(key(n.args[j])))
          todo.emplace_back(n.args[j], false);
      continue;
    }
    std::vector<Id> args;
    args.reserve(n.args.size());
    for (size_t j = 0; j < n.args.size(); ++j)
      args.push_back(shadow && j == 1 ? n.args[j]
                                    : restricted_axis_cache.at(key(n.args[j])));
    restricted_axis_cache.emplace(key(id), reapply(id, args));
    ++a.stats["canonical_domain_rebind_nodes"];
  }
  return restricted_axis_cache.at(key(root));
}

// Same orientation for reference and optimized, before either is abstracted:
// Fold(z, c*f) -> z + c*Fold(0, f), never Fold(c*z, c*f).
// The small shells materialize the SumNF view in the existing IR. No element
// expansion, new IR operator, or reference-dependent alternative is needed.
// A Max over a -infinity-masked body keeps only the possible prefix of its
// mask (masked_max_prefix), rebound to a fresh binder as for Sum. Returns
// the new (binder, body) and narrows count; nothing if no index is dropped.
inline auto Checker::masked_max_domain(Id value, Id axis, Z &count) -> std::optional<std::pair<Id, Id>> {
  const Z prefix = masked_max_prefix(a, value, axis, count);
  if (prefix >= count) return {};
  ++a.stats["canonical_max_trimmed_domains"];
  count = prefix;
  if (prefix == 0) return std::pair(axis, value);
  const auto key = std::tuple(value, axis, prefix);
  auto found = restricted_max_bodies.find(key);
  if (found == restricted_max_bodies.end()) {
    Id fresh;
    size_t before;
    do {
      before = a.nodes.size();
      fresh = a.var("canonical-max-domain:" + std::to_string(domain_binder_serial++),
                    Q(0), Q(prefix - 1));
    } while (fresh < before);
    found = restricted_max_bodies.emplace(key, std::pair(fresh, restrict_axis(value, axis, fresh))).first;
  }
  return found->second;
}

// strict (reapply): a malformed fold rebuilt with changed operands would keep the original
// node's facts and bounds; refused instead, as in reapply's other fact-copying branches.
inline auto Checker::canonicalize_fold(Id original, const std::vector<Id> &args, bool strict) -> Id {
  const auto &n = a.nodes[original];
  if (n.args.size() != 2 || n.data.size() != 2 ||
      n.data[0].denominator() != 1 || n.data[1].denominator() != 1 ||
      n.data[0] < 0 || n.data[0] >= Q(a.nodes.size()) || n.data[1] < 0) {
    if (args == n.args) return original;
    if (strict) return invalid;
    return a.intern(n.op, args, n.data, n.tag, n.facts, n.bounds);
  }
  Id axis = n.data[0].numerator().convert_to<Id>();
  Z count = n.data[1].numerator();
  if (count == 0) return args[0];
  bool sum = n.tag == "Sum";
  Id value = args[1];
  if (sum) {
    auto key = std::tuple(value, axis, count);
    auto it = masked_prefix_cache.find(key);
    if (it == masked_prefix_cache.end())
      it = masked_prefix_cache.emplace(key, masked_sum_prefix(a, value, axis, count)).first;
    if (it->second < count) {
      count = it->second;
      ++a.stats["canonical_sum_trimmed_domains"];
      if (count == 0) {
        ++a.stats["canonical_sum_zero_domains"];
        return args[0];
      }
      auto found = restricted_sum_bodies.find(key);
      if (found == restricted_sum_bodies.end()) {
        // A newly allocated binder cannot already occur free or bound in
        // the old body. Cache the restriction, not a potentially capturing
        // name shared by unrelated lexical scopes.
        Id fresh;
        size_t before;
        do {
          before = a.nodes.size();
          fresh = a.var("canonical-sum-domain:" + std::to_string(domain_binder_serial++),
                        Q(0), Q(count - 1));
        } while (fresh < before);
        Id restricted = restrict_axis(value, axis, fresh);
        found = restricted_sum_bodies.emplace(key, std::pair(fresh, restricted)).first;
      }
      axis = found->second.first;
      value = found->second.second;
    }
  } else if (auto trimmed = masked_max_domain(value, axis, count)) {
    if (count == 0) return args[0];
    std::tie(axis, value) = *trimmed;
  }
  // A Max Fold seeded by body(-1) becomes an -infinity-seeded Fold of
  // body(w-1) over [0,n+1): both contain exactly body(-1),...,body(n-1).
  // Require exact hash-consed seed identity and finite-or-negative-infinity
  // facts for seed and body. Rebuild both substitutions with substitute_strict;
  // never reuse bounds from [0,n) or store them in restricted_axis_cache.
  // Unsupported rebuilds decline. Cost: O(body) for each of two rebuilds.
  if (n.tag == "Max" && !a.neginf(args[0]) && a.depends(value, axis) && count > 0 &&
      a.nodes[value].facts.finite_or_negative_infinity() && a.nodes[args[0]].facts.finite_or_negative_infinity()) {
    const Id previous = substitute_strict(value, axis, a.constant(Q(-1)));
    if (previous != invalid && previous == args[0]) {
      // One fresh binder per (body, binder, count), reused as restricted_sum_bodies reuses its own.
      const auto key = std::tuple(value, axis, count);
      auto found = unpeeled_max_bodies.find(key);
      if (found == unpeeled_max_bodies.end()) {
        Id fresh;
        size_t before;
        do {
          before = a.nodes.size();
          fresh = a.var("canonical-max-unpeeled:" + std::to_string(domain_binder_serial++), Q(0), Q(count));
        } while (fresh < before);
        found = unpeeled_max_bodies.emplace(key, std::pair(fresh, substitute_strict(value, axis, a.sub(fresh, a.one)))).first;
      }
      const auto [fresh, body] = found->second;
      if (body != invalid) {
        const Id seed = a.infinity(-1);
        const Id unpeeled = a.intern("fold", {seed, body}, {Q(fresh), Q(count + 1)}, "Max",
                                     reduction_facts(seed, body, fresh, count + 1, true));
        ++a.stats["canonical_max_unpeeled_folds"];
        return canonicalize_fold(unpeeled, {seed, body}, strict);
      }
    }
  }
  // Rule (Arena::common_positive_scale): q*. with q > 0 is monotone, so a Max Fold of q*y is
  // q * (Max Fold of y), with the seed divided by q (a -infinity seed stays). The constant then
  // sits outside, as a Max node's common scale does. O(1) beyond the rebuilt body.
  if (n.tag == "Max") {
    const auto [q, body] = a.positive_scale(value);
    const auto [seed_scale, seed] = a.positive_scale(args[0]);
    const auto seed_value = a.value(args[0]);
    if (q != 1 && (a.neginf(args[0]) || seed_value || seed_scale == q)) {
      const Id initial = a.neginf(args[0]) ? args[0] :
                         seed_value ? a.constant(*seed_value / q) : seed;
      Id pure = a.intern("fold", {initial, body}, {Q(axis), Q(count)}, "Max",
                         reduction_facts(initial, body, axis, count, true));
      ++a.stats["canonical_max_scale_pulls"];
      return a.mul({a.constant(q), pure});
    }
  }
  // Max is idempotent: count >= 1 copies of an axis-invariant body b give
  // max(z, b), and its identity -infinity leaves the seed z.
  if (n.tag == "Max" && !a.depends(value, axis)) {
    ++a.stats["canonical_max_invariant_bodies"];
    return a.maximum({args[0], value});
  }
  std::vector<Q> domain{Q(axis), Q(count)};
  Facts facts = reduction_facts(args[0], value, axis, count, n.tag == "Max");
  if (sum && facts.finite && !a.depends(args[0], axis)) {
    if (auto scalar_sum = simplify_scalar_sum(a, args[0], value, axis, count))
      return *scalar_sum;
    auto [coefficient, body] = split_invariant_factors(value, axis);
    if (coefficient != a.one || args[0] != a.zero) {
      if (coefficient != a.one) ++a.stats["canonical_sum_factor_pulls"];
      if (args[0] != a.zero) ++a.stats["canonical_sum_initials"];
      Id pure = a.intern("fold", {a.zero, body}, domain, "Sum",
                         reduction_facts(a.zero, body, axis, count, false));
      // The seed is outside the multiplication and still appears once.
      return a.factor_sum(a.add({args[0], a.mul({coefficient, pure})}));
    }
  }
  return a.intern("fold", {args[0], value}, domain, n.tag, facts);
}

inline auto Checker::canonicalize(Id root) -> Id {
  std::vector<std::pair<Id, bool>> todo{{root, false}};
  // Normalize newly emitted shells too. Already normalized immutable DAG
  // nodes stay cached; this is not a repeated whole-graph fixed-point scan.
  std::unordered_map<Id, Id> rewritten;
  while (!todo.empty()) {
    a.check();
    auto [id, ready] = todo.back(); todo.pop_back();
    if (canonical_cache.contains(id)) continue;
    if (auto it = rewritten.find(id); it != rewritten.end()) {
      canonical_cache.emplace(id, canonical_cache.at(it->second));
      continue;
    }
    const auto &n = a.nodes[id];
    const bool empty = n.op == "fold" && n.args.size() == 2 &&
                       n.data.size() == 2 && n.data[1] == 0;
    if (!ready) {
      todo.emplace_back(id, true);
      for (size_t j = 0; j < (empty ? 1 : n.args.size()); ++j)
        if (!canonical_cache.contains(n.args[j])) todo.emplace_back(n.args[j], false);
      continue;
    }
    if (empty) {
      canonical_cache.emplace(id, canonical_cache.at(n.args[0]));
      ++a.counter(Arena::Hot::canonical_node_visits);
      continue;
    }
    std::vector<Id> args;
    args.reserve(n.args.size());
    for (Id child : n.args) args.push_back(canonical_cache.at(child));
    Id result = n.op == "fold" ? canonicalize_fold(id, args) : reapply(id, args);
    if (result != id && !canonical_cache.contains(result)) {
      rewritten.emplace(id, result);
      todo.emplace_back(id, true);
      todo.emplace_back(result, false);
    } else {
      canonical_cache.emplace(id, result == id ? id : canonical_cache.at(result));
    }
    ++a.counter(Arena::Hot::canonical_node_visits);
  }
  return canonical_cache.at(root);
}

// This second stage is intentionally reference-guided: register definitions
// on the reference side, prove contributions on the optimized side. The
// algebraic canonicalizer above has no such side flag or active-spec state.
inline auto Checker::normalize(Id root, bool reference) -> Id {
  root = canonicalize(root);
  std::unordered_map<Id, Entry> local;
  if (reference) {
    active.clear();
    reference_binders.clear();
  } else if (active != opt_version) {
    opt_version = active;
    opt_cache.clear();
    ++a.stats["normalization_cache_context_changes"];
  }
  auto &cache = reference ? local : opt_cache;
  std::vector<std::pair<Id, bool>> todo{{root, false}};
  std::unordered_map<Id, Id> redirects;
  while (!todo.empty()) {
    a.check();
    auto [old, ready] = todo.back();
    todo.pop_back();
    if (cache.contains(old))
      continue;
    if (auto it = redirects.find(old); it != redirects.end()) {
      cache.emplace(old, cache.at(it->second));
      continue;
    }
    const auto &raw = a.nodes[old];
    if (!ready) {
      // Visit the Fold before its body, so reg() can recognize an enclosing
      // reference axis without a second DAG traversal or variable-name tests.
      if (raw.op == "fold") {
        const Id binder = raw.data[0].numerator().convert_to<Id>();
        fold_binders.insert(binder);
        if (reference) reference_binders.insert(binder);
      }
      todo.emplace_back(old, true);
      for (size_t j = raw.args.size(); j-- > 0;)
        if (!cache.contains(raw.args[j]))
          todo.emplace_back(raw.args[j], false);
      continue;
    }
    std::vector<Id> args;
    std::vector<const Summary *> child;
    for (Id x : raw.args) {
      auto &entry = cache.at(x);
      args.push_back(entry.n);
      child.push_back(&entry.summary);
    }
    Summary summary;
    Id n;
    if (raw.op == "fold") {
      // Child abstraction can reveal a simpler predicate or finite zero
      // mask. Re-enter the same normal form before binding summaries or
      // registering a definition; never bind with a stale domain/axis.
      Id rebuilt = canonicalize_fold(old, args);
      const auto &normalized = a.nodes[rebuilt];
      if (normalized.op != "fold" || normalized.args != args ||
          normalized.data != raw.data || normalized.tag != raw.tag) {
        redirects.emplace(old, rebuilt);
        todo.emplace_back(old, true);
        todo.emplace_back(rebuilt, false);
        ++a.stats["normalization_fold_rebuild_redirects"];
        continue;
      }
      Id var = raw.data[0].numerator().convert_to<Id>();
      Z count = raw.data[1].numerator();
      bool max = raw.tag == "Max";
      if (reference) {
        if(count==0) n=args[0];
        else {
          Id call=reg(args[1],var,count,max);
          n=max ? a.maximum({args[0],call}) : a.add({args[0],call});
        }
      }
      else {
        Summary initial = select(*child[0], max),
                bound = bind(select(*child[1], max), var, count);
        if (!initial.empty() && !bound.empty())
          summary = combine({initial, bound});
        else if (!bound.empty())
          summary = residual(bound, args[0]);
        n = a.intern("fold", args, raw.data, raw.tag,
                     reduction_facts(args[0], args[1], var, count, max));
        if (count == 0 || (!max && args[1] == a.zero)) {
          n = args[0];
          summary = *child[0];
        }
      }
    } else {
      n = reapply(old, args);
      if (raw.op == "add" || raw.op == "add_raw") {
        std::vector<Summary> parts;
        std::vector<Id> unmatched;
        for (size_t j = 0; j < child.size(); ++j) {
          auto s = select(*child[j], false);
          if (s.empty())
            unmatched.push_back(a.mul({a.constant(raw.data[j]), args[j]}));
          else
            parts.push_back(scale(s, a.constant(raw.data[j])));
        }
        summary = residual(combine(parts), a.add(unmatched));
      } else if (raw.op == "max") {
        std::vector<Summary> parts;
        std::vector<Id> unmatched;
        for (size_t j = 0; j < child.size(); ++j) {
          auto s = select(*child[j], true);
          if (s.empty())
            unmatched.push_back(args[j]);
          else
            parts.push_back(std::move(s));
        }
        summary = residual(combine(parts), a.maximum(unmatched));
      } else if (raw.op == "mul") {
        size_t participating = 0, index = 0;
        Summary selected;
        for (size_t j = 0; j < child.size(); ++j) {
          auto s = select(*child[j], false);
          if (!s.empty()) {
            ++participating;
            index = j;
            selected = std::move(s);
          }
        }
        if (participating == 1) {
          std::vector<Id> xs{a.constant(raw.data[0])};
          for (size_t j = 0; j < args.size(); ++j)
            if (j != index)
              xs.push_back(args[j]);
          summary = scale(selected, a.mul(xs));
        }
      }
    }
    // Keep the expression DAG compact; normalize only scalar obligations.
    if (!reference) {
      auto c = close(summary);
      if (c) {
        n = *c;
        summary.clear();
      }
      if (!exponential_mode) n = a.factor_sum(n);
      auto it = active_byroot.find(a.nodes[n].op);
      std::vector<Id> candidates;
      if(it!=active_byroot.end())candidates=it->second;
      if(exponential_mode && (a.nodes[n].op=="add" || a.nodes[n].op=="mul" || a.nodes[n].op=="recip" || a.nodes[n].op=="exp" || a.nodes[n].op=="exp2"))
        for(Id sid:active)if(candidates.size()<64 && std::find(candidates.begin(),candidates.end(),sid)==candidates.end())candidates.push_back(sid);
      for (Id sid : candidates) {
          ++a.counter(Arena::Hot::local_pattern_candidates);
          for (const auto &m : match_term(sid, n)) {
            Key k{sid, m.params, m.coef, neutral(sid)};
            Id p = leaf(m.index);
            auto oldpiece = summary.find(k);
            if (oldpiece == summary.end())
              summary.emplace(std::move(k), p);
            else
              oldpiece->second = concat(oldpiece->second, p);
          }
        }
    }
    cache.emplace(old, Entry{n, std::move(summary)});
    ++a.counter(Arena::Hot::normalization_node_visits);
  }
  if (reference) {
    active_byroot.clear();
    for (Id sid : active)
      active_byroot[patterns.nodes[specs[sid].pattern].op].push_back(sid);
  }
  return cache.at(root).n;
}

} // namespace rg
