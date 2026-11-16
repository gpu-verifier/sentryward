#pragma once
#include "linear_reduction_pattern.hpp"

namespace rg {

// The exponents e_1..e_n (one base) of a product of exponentials with a scalar in (0, 1], not all
// finite: the product is at most 1 when every e_j is finite or -inf and the case fixes X < k or
// X <= k with k <= 0 (a true lt/le, or a false ge/gt: X has no NaN), X the unreordered sum of
// exactly these exponents (add_raw, every coefficient 1, the same multiset). If some e_j is -inf,
// its exponential is 0 and the others are finite and positive, so the product is 0; otherwise X is
// the real sum E of the exponents and E < k <= 0 (or E <= k <= 0), so the product is at most its
// scalar. Exact. O(fixed * n log n).
inline auto LinearReductionPattern::raw_exponents_at_most_zero(const std::vector<Id> &exponents, const CaseFix &fixed) -> bool {
  for (Id e : exponents) if (!a.nodes[e].facts.finite_or_negative_infinity()) return false;
  std::vector<Id> wanted = exponents;
  std::sort(wanted.begin(), wanted.end());
  for (const auto &[c, truth] : fixed) {
    const auto &m = a.nodes[c];
    if (m.op != "compare" || m.args.size() != 2) continue;
    if (!(truth && (m.tag == "lt" || m.tag == "le")) && !(!truth && (m.tag == "ge" || m.tag == "gt"))) continue;
    const auto k = a.value(m.args[1]);
    if (!k || *k > 0) continue;
    const auto &sum = a.nodes[m.args[0]];
    if (sum.op != "add_raw" || sum.data != std::vector<Q>(sum.args.size(), Q(1))) continue;
    std::vector<Id> terms = sum.args;
    std::sort(terms.begin(), terms.end());
    if (terms == wanted) { ++a.stats["linear_reduction_vote_raw_exponent_bounds"]; return true; }
  }
  return false;
}

inline auto LinearReductionPattern::at_most_one_by_arms(Id x, CaseFix &path, size_t &leaves) -> bool {
  const auto &n = a.nodes[x];
  if (n.op == "select" && n.args.size() == 3) {
    if (path.size() >= vote_arm_depth) return false;
    const Id c = n.args[0], yes = n.args[1], no = n.args[2];
    for (const bool truth : {true, false}) {
      path.emplace_back(c, truth);
      const bool ok = at_most_one_by_arms(truth ? yes : no, path, leaves);
      path.pop_back();
      if (!ok) return false;
    }
    return true;
  }
  ++leaves;
  return at_most_one(x, path);
}

// x <= 1 in a resolved case: a constant at most 1, or a product of exponentials (scalar factor
// in (0, 1]) whose summed exponent E is at most 0, by its bounds or because the case fixes
// X < k or X <= k (a true lt/le or a false ge/gt of finite X) with E - X a constant d and
// k + d <= 0. Exact.
// A bounded structural order proof. Max is above each operand; a Select
// is above a value only if both arms are. No predicate assumptions are added.
inline auto LinearReductionPattern::maximum_dominates(Id upper, Id lower) -> bool {
  std::map<std::pair<Id, Id>, bool> memo;
  size_t work = 0;
  std::function<bool(Id, Id)> prove = [&](Id u, Id l) -> bool {
    a.check();
    if (++work > 512) return false;
    if (u == l) return true;
    if (!a.nodes[u].facts.finite || !a.nodes[l].facts.finite) return false;
    const auto key = std::pair(u, l);
    if (auto it = memo.find(key); it != memo.end()) return it->second;
    const auto &un = a.nodes[u], &ln = a.nodes[l];
    bool result = false;
    if (ln.op == "max")
      result = std::all_of(ln.args.begin(), ln.args.end(), [&](Id v) { return prove(u, v); });
    else if (ln.op == "select" && ln.args.size() == 3)
      result = prove(u, ln.args[1]) && prove(u, ln.args[2]);
    else if (un.op == "max")
      result = std::any_of(un.args.begin(), un.args.end(), [&](Id v) { return prove(v, l); });
    else if (un.op == "select" && un.args.size() == 3)
      result = prove(un.args[1], l) && prove(un.args[2], l);
    memo.emplace(key, result);
    return result;
  };
  return prove(upper, lower);
}

inline auto LinearReductionPattern::at_most_one(Id x, const CaseFix &fixed) -> bool {
  if (auto v = a.value(x)) return *v <= 1;
  const auto &n = a.nodes[x];
  std::vector<Id> exponents; bool base2 = true;
  if (n.op == "exp2" || n.op == "exp") { exponents.push_back(n.args[0]); base2 = n.op == "exp2"; }
  else if (n.op == "mul" && n.data.size() == 1 && n.data[0] > 0 && n.data[0] <= 1) {
    for (Id f : n.args) {
      const auto &m = a.nodes[f];
      if ((m.op != "exp2" && m.op != "exp") || m.args.size() != 1) return false;
      if (!exponents.empty() && base2 != (m.op == "exp2")) return false;
      base2 = m.op == "exp2"; exponents.push_back(m.args[0]);
    }
  } else return false;
  if (exponents.empty()) return false;
  if (std::any_of(exponents.begin(), exponents.end(), [&](Id e) { return !a.nodes[e].facts.finite; }))
    return raw_exponents_at_most_zero(exponents, fixed);
  const Id sum = a.add(exponents);
  if (a.nodes[sum].bounds && a.nodes[sum].bounds->second <= 0) return true;
  const auto &difference = a.nodes[sum];
  if (difference.op == "add" && difference.args.size() == 2 && difference.data.size() == 2 &&
      difference.data[0] == -difference.data[1] && difference.data[0] != 0) {
    const size_t positive = difference.data[0] > 0 ? 0 : 1;
    if (maximum_dominates(difference.args[1 - positive], difference.args[positive])) {
      ++a.stats["linear_reduction_maximum_rescale_bounds"];
      return true;
    }
  }
  // X < k or X <= k: a true lt/le, or a false ge/gt (complements on finite operands).
  for (const auto &[c, truth] : fixed) {
    const auto &m = a.nodes[c];
    if (m.op != "compare" || m.args.size() != 2) continue;
    if (!(truth && (m.tag == "lt" || m.tag == "le")) && !(!truth && (m.tag == "ge" || m.tag == "gt"))) continue;
    const auto k = a.value(m.args[1]);
    if (!k || !a.nodes[m.args[0]].facts.finite) continue;
    if (auto d = a.value(a.sub(sum, m.args[0])); d && *k + *d <= 0) return true;
  }
  return false;
}

// Whether x = y wherever guard != 0: in every case of the Select conditions met in x, y and the
// guard, the guard resolves to 0 or x = y by the exact EP equality (both finite). At most
// 2^case_variable_limit cases, each a rebuild of at most case_rebuild_limit nodes plus one EP
// equality.
// Whether x is a finite value in the case `fixed`: by its facts, or built from such values by
// sums, products and exponentials, by the reciprocal of a value the case fixes nonzero, or by the
// logarithm of a value the case fixes positive (the criteria of resolve()'s reciprocal and
// logarithm rules). Anything else, and anything beyond case_rebuild_limit nodes, counts as not
// finite. Used only to label an accepted case (case_equal), never to accept one.
inline auto LinearReductionPattern::finite_in_case(Id x, const CaseFix &fixed) -> bool {
  std::unordered_map<Id, bool> known; std::unordered_map<Id, Id> memo; std::vector<Id> open; size_t work = 0;
  const auto truth = [&](const char *tag, Id v) {
    return case_truth(resolve(a.compare(tag, v, a.zero), fixed, memo, open, work), fixed);
  };
  std::function<bool(Id)> finite = [&](Id id) -> bool {
    if (a.nodes[id].facts.finite) return true;
    if (auto it = known.find(id); it != known.end()) return it->second;
    if (++work > case_rebuild_limit) throw CaseMiss{};
    const std::string op = a.nodes[id].op; const auto args = a.nodes[id].args;
    bool result = false;
    if (op == "mul" || op == "add" || op == "add_raw" || op == "exp" || op == "exp2")
      result = !args.empty() && std::all_of(args.begin(), args.end(), finite);
    else if (op == "recip" && args.size() == 1)
      result = finite(args[0]) && (truth("eq", args[0]) == 0 || truth("le", args[0]) == 0 || truth("ge", args[0]) == 0);
    else if ((op == "log" || op == "log2") && args.size() == 1)
      result = finite(args[0]) && truth("le", args[0]) == 0;
    known.emplace(id, result);
    return result;
  };
  try { return finite(x); } catch (const CaseMiss &) { return false; }
}

// A case in which both values resolve to one node is accepted: the two sides are then the same
// expression, so the same value or the same undefinedness. Where that node is not a finite value
// in the case (finite_in_case; -inf of an empty log-sum-exp row, for example), this is equality of
// partial functions and not a total value proof (the EP engine refuses a non-finite side). Each
// successful analysis with such a case is counted in linear_reduction_case_nonfinite_identities;
// the comparison labels a family proved with one accordingly (comparison.cpp,
// equivalence_contract).
inline auto LinearReductionPattern::case_equal(Id x, Id y, Id guard) -> bool {
  CaseFix fixed; size_t cases = 0; bool equal = false, nonfinite_identity = false;
  try {
    equal = all_cases({x, y, guard},
        [&](const std::vector<Id> &v, const CaseFix &f) {
          // The guard is 0 in this case: a constant 0, or a fixed (or complementary) condition.
          if (case_truth(v[2], f) == 0) return true;
          if (v[0] == v[1]) { nonfinite_identity = nonfinite_identity || !finite_in_case(v[0], f); return true; }
          if (!a.nodes[v[0]].facts.finite || !a.nodes[v[1]].facts.finite) {
            // Logarithms of values this case fixes positive are finite here: as finite_part atoms
            // (case_finite_rebuild) both values may become finite sums the EP engine compares. An
            // identity proved with such an atom holds for every real value of it. Exact.
            bool ok = true;
            const Id x = case_finite_rebuild(v[0], f, ok), y = case_finite_rebuild(v[1], f, ok);
            if (!ok || !a.nodes[x].facts.finite || !a.nodes[y].facts.finite) return false;
            ++a.stats["linear_reduction_case_finite_logarithm_comparisons"];
            try { return checker.scalar.equal(x, y); } catch (const Unsupported &) { return false; }
          }
          try { return checker.scalar.equal(v[0], v[1]); } catch (const Unsupported &) { return false; }
        }, fixed, cases, {2});
  } catch (const CaseMiss &) { equal = false; }
  a.stats["linear_reduction_case_split_cases"] += cases;
  if (!equal) ++a.stats["linear_reduction_case_split_declines"];
  else if (nonfinite_identity) ++a.stats["linear_reduction_case_nonfinite_identities"];
  return equal;
}

// v rebuilt for one case: each logarithm L (log or log2) of a finite nonnegative argument that the
// case fixes positive ([arg <= 0] false) becomes the atom finite_part(L), and sums (add, add_raw)
// and products above it are rebuilt by their constructors (so finite parts merge). A finite node
// is kept. Any other non-finite node sets ok = false. O(nodes of v).
// finite_part(x) denotes x where x is finite and 0 elsewhere: a total function, so the node's
// facts (finite; no sign, no bounds) hold on every input, not only in this case, and a hash-consed
// finite_part node reached from any other expression means the same. The EP engine keeps a finite
// node as an opaque atom. Within the case L is finite (arg > 0), so finite_part(L) = L and the
// rebuilt value equals v there; an identity proved with the atom holds for every real value of it.
inline auto LinearReductionPattern::case_finite_rebuild(Id v, const CaseFix &fixed, bool &ok) -> Id {
  if (!ok) return v;
  const auto &n = a.nodes[v];
  if (n.facts.finite) return v;
  const std::string op = n.op;
  const auto args = n.args;
  const auto data = n.data;
  if ((op == "log" || op == "log2") && args.size() == 1 && a.nodes[args[0]].facts.finite && a.nodes[args[0]].facts.nn) {
    std::unordered_map<Id, Id> memo; std::vector<Id> open; size_t work = 0;
    try {
      if (case_truth(resolve(a.compare("le", args[0], a.zero), fixed, memo, open, work), fixed) == 0)
        return a.intern("finite_part", {v}, {}, "", {true, false, false});
    } catch (const CaseMiss &) {}
    ok = false; return v;
  }
  if ((op == "add" || op == "add_raw") && args.size() == data.size()) {
    std::vector<Id> terms;
    for (size_t j = 0; j < args.size(); ++j) terms.push_back(a.mul({a.constant(data[j]), case_finite_rebuild(args[j], fixed, ok)}));
    return ok ? a.add(std::move(terms)) : v;
  }
  if (op == "mul" && data.size() == 1) {
    std::vector<Id> factors{a.constant(data[0])};
    for (Id x : args) factors.push_back(case_finite_rebuild(x, fixed, ok));
    return ok ? a.mul(std::move(factors)) : v;
  }
  ok = false;
  return v;
}

// x and y without the factors their products share (same node, with multiplicity): for
// x = d K_1..K_n X and y = e K_1..K_n Y returns (d X, e Y); (x, y) itself when they share no
// factor or one of them is not finite. O(factors log factors).
inline auto LinearReductionPattern::without_common_factors(Id x, Id y) -> std::pair<Id, Id> {
  if (!a.nodes[x].facts.finite || !a.nodes[y].facts.finite) return {x, y};
  const auto factors = [&](Id v) {
    const auto &n = a.nodes[v];
    return n.op == "mul" && n.data.size() == 1 ? std::pair(n.args, n.data[0]) : std::pair(std::vector<Id>{v}, Q(1));
  };
  auto [fx, dx] = factors(x);
  auto [fy, dy] = factors(y);
  std::multiset<Id> other(fy.begin(), fy.end());
  std::vector<Id> rx{a.constant(dx)}, ry{a.constant(dy)};
  bool shared = false;
  for (Id f : fx) {
    if (auto it = other.find(f); it != other.end()) { other.erase(it); shared = true; }
    else rx.push_back(f);
  }
  if (!shared) return {x, y};
  ry.insert(ry.end(), other.begin(), other.end());
  return {a.mul(rx), a.mul(ry)};
}

inline auto LinearReductionPattern::merge_case_equal_groups(Form &form) -> void {
  std::map<size_t, std::vector<Key>> by_target;
  for (const auto &[key, group] : form.groups) by_target[key.target].push_back(key);
  for (auto &[target, keys] : by_target) {
    // Each group's guard before any merge (1 when unknown), and the premise of each group that joins
    // (support_premises below).
    std::map<Id, Id> guard_before, premise;
    for (const auto &key : keys) {
      const Id g = form.groups.at(key).guard;
      guard_before.emplace(key.coefficient, g == invalid ? a.one : g);
    }
    if (keys.size() < 2) { support_premises[target] = {guard_before.begin()->second}; continue; }
    Key base = keys.front();
    for (const auto &key : keys)
      if (form.groups.at(key).multiplicity > form.groups.at(base).multiplicity) base = key;
    for (const auto &key : keys) {
      if (key.coefficient == base.coefficient) continue;
      const Group &group = form.groups.at(key);
      const Id guard = group.guard == invalid ? a.one : group.guard;
      // c_g = K * x and c_b = K * y with the same factors K: x = y gives c_g = c_b. The shared
      // factors' own Selects then stay out of the case analysis; the whole coefficients are
      // compared only where that declines.
      const auto [x, y] = without_common_factors(key.coefficient, base.coefficient);
      // Rule (coefficients equal in local form, Checker::local_index_form): two coefficients of one
      // target whose local forms agree are one value on every input (e.g. two spellings of the same
      // running maximum, nested differently); the groups merge as on a case equality. Pair-local.
      if (!((x != key.coefficient || y != base.coefficient) && case_equal(x, y, guard)) &&
          !case_equal(key.coefficient, base.coefficient, guard)) {
        // A coefficient holds the running maxima of whole rows of scores: a larger walk than an index.
        if (checker.local_index_form(key.coefficient, merge_form_limit) !=
            checker.local_index_form(base.coefficient, merge_form_limit)) continue;
        ++a.stats["linear_reduction_local_form_merges"]; ++local_form_merges;
      }
      const Group moved = group;
      form.groups.erase(key);
      Group &into = form.groups.at(base);
      into.piece = concat(into.piece, moved.piece);
      into.witness = sum_witness(into.witness, moved.witness);
      into.multiplicity += moved.multiplicity;
      if (into.guard != moved.guard) into.guard = a.one;
      ++a.stats["linear_reduction_case_split_merges"];
      premise.emplace(key.coefficient, guard_before.at(key.coefficient));
    }
    if (std::any_of(keys.begin(), keys.end(), [&](const Key &key) {
          return key.coefficient != base.coefficient && form.groups.contains(key); }))
      merge_by_finiteness_chain(form, keys, base, premise);
    if (std::none_of(keys.begin(), keys.end(), [&](const Key &key) {
          return key.coefficient != base.coefficient && form.groups.contains(key); })) {
      std::vector<Id> premises{guard_before.at(base.coefficient)};
      for (const auto &[c, f] : premise)
        if (std::find(premises.begin(), premises.end(), f) == premises.end()) premises.push_back(f);
      support_premises[target] = std::move(premises);
    }
  }
}

} // namespace rg
