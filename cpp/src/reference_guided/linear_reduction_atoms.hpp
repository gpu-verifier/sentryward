#pragma once
#include "linear_reduction_pattern.hpp"

namespace rg {

inline auto LinearReductionPattern::reduction_atom(Id x, const Target &target) -> Id {
  // Keyed by the mode too: with residual_atoms an atom may keep a finite residual.
  const auto key = std::tuple(x, residual_atoms, target.sid, target.parameters);
  if (auto it = atom_cache.find(key); it != atom_cache.end()) return it->second;
  Id result = invalid; bool exhausted = false;
  // As attempt(): term-wise distribution first; only a distribution-budget miss repeats the
  // decomposition once with binder-independent weights factored out (the same exact mode).
  for (const bool factored : {false, true}) {
    LinearReductionPattern inner(checker, remaining_budget());
    inner.closure_query = closure_query;  // charged here, so part of an enclosing closure query
    inner.targets = {target};
    inner.factor_weights = factored;
    bool retry = false;
    try {
      Form form = inner.decompose(x);
      inner.merge_case_equal_groups(form);
      if (form.groups.size() == 1) {
        const auto &[group_key, group] = *form.groups.begin();
        if (inner.complete(group_key, group)) {
          result = a.mul({group_key.coefficient, checker.call(target.sid, target.parameters)});
          // (Light case mode, residual_atoms.) A finite residual R stays beside the closed group:
          // X = R + C * call(T) (a sum under a reciprocal such as mass + sink). The identity is the
          // decomposition itself. Exact.
          if (form.residual != invalid && !residual_atoms) result = invalid;
          else if (form.residual != invalid) {
            const Id rest = inner.materialize(form.residual);
            if (a.nodes[rest].facts.finite) { result = a.add({rest, result}); ++a.stats["linear_reduction_atoms_with_residual"]; }
            else result = invalid;
          }
          // The support of a residual-free atom X = C * call(T) (the only atoms that become aliases).
          if (form.residual == invalid)
            if (auto it = inner.support_premises.find(0); it != inner.support_premises.end())
              atom_support_cache[key] = it->second;
        }
      }
    } catch (const Miss &miss) {
      const std::string_view reason = miss.reason;
      exhausted = reason == "node_budget" || reason == "piece_budget" || reason == "polynomial_budget";
      retry = !factored && reason == "distribution_budget";
    } catch (const Unsupported &) {}
    charge(inner);
    if (!retry) break;
    ++a.stats["linear_reduction_atom_factored_retries"];
  }
  if (result != invalid || !exhausted) atom_cache.emplace(key, result);
  return result;
}

// The light attempt of an output check (light_cases, residual_atoms): reduction atoms (now also with
// a finite residual) substituted again, then the exact EP equality or the exact case analysis; where
// no atom is left but normalization already closed the optimized Sums into the reference's own Calls
// (shares_reference_call), the case analysis of the two values themselves. Exact; the ordinary
// attempt's result is not changed, this only runs after it declined.
inline auto LinearReductionPattern::light_attempt(Id reference, Id optimized) -> bool {
  FlagScope light(light_cases, true), residuals(residual_atoms, true);
  const Id substituted = substitute_reduction_atoms(reference, optimized);
  bool closed = substituted != optimized &&
                (checker.scalar.equal(reference, substituted) || case_equal(reference, substituted, a.one));
  if (!closed && substituted == optimized && shares_reference_call(reference, optimized))
    closed = case_equal(reference, optimized, a.one);
  if (closed) ++a.stats["linear_reduction_light_case_proofs"];
  return closed;
}

// Whether a Call of the reference (outside Fold bodies) occurs in the optimized value (outside
// Fold bodies), as the same node. O(nodes of both), charged to tick().
inline auto LinearReductionPattern::shares_reference_call(Id reference, Id optimized) -> bool {
  std::unordered_set<Id> calls;
  for (const bool reference_side : {true, false}) {
    std::vector<Id> todo{reference_side ? reference : optimized}; std::unordered_set<Id> seen;
    while (!todo.empty()) {
      tick(); const Id id = todo.back(); todo.pop_back();
      if (!seen.insert(id).second) continue;
      const auto &n = a.nodes[id];
      if (n.op == "call") {
        if (reference_side) calls.insert(id);
        else if (calls.contains(id)) return true;
        continue;
      }
      if (n.op == "fold") { if (!n.args.empty()) todo.push_back(n.args[0]); continue; }
      todo.insert(todo.end(), n.args.begin(), n.args.end());
    }
    if (calls.empty()) return false;
  }
  return false;
}

inline auto LinearReductionPattern::substitute_reduction_atoms(Id reference, Id optimized, bool with_products) -> Id {
  atom_products = false;
  // Reference Calls, anywhere outside Fold bodies.
  std::vector<Target> calls; std::set<std::pair<Id, std::vector<Id>>> seen_calls;
  std::vector<Id> todo{reference}; std::unordered_set<Id> seen;
  while (!todo.empty()) {
    tick(); const Id id = todo.back(); todo.pop_back();
    if (!seen.insert(id).second) continue;
    const auto &n = a.nodes[id];
    if (n.op == "call") {
      if (n.data.size() != 1 || n.data[0].denominator() != 1 || n.data[0] < 0 ||
          n.data[0] >= Q(checker.specs.size())) continue;
      const Id sid = n.data[0].numerator().convert_to<Id>();
      const auto &spec = checker.specs[sid];
      if (!checker.active.contains(sid) || spec.maximum || !spec.facts.finite || !n.facts.finite ||
          spec.count <= 0 || n.args.size() + 1 != spec.slots || !checker.literal_binder_free(sid)) continue;
      if (seen_calls.emplace(sid, n.args).second) calls.push_back({sid, n.args});
      if (calls.size() > budget.targets) return optimized;
      continue;
    }
    const size_t children = n.op == "fold" ? std::min<size_t>(1, n.args.size()) : n.args.size();
    for (size_t j = 0; j < children; ++j) todo.push_back(n.args[j]);
  }
  if (calls.empty()) return optimized;
  // Sums under reciprocals, logarithms and comparisons of the optimized value. A product with
  // exactly one Sum factor there (a factored Sum w * (x_1 + ... + x_n)) is a candidate too, in
  // whatever room the Sum nodes leave under atom_limit.
  const auto sum_node = [&](Id y) { return a.nodes[y].op == "add" || a.nodes[y].op == "add_raw"; };
  // Candidates in the order met; `compared` holds the Sums met so far only under comparisons, and
  // `primary` counts those met under a reciprocal or logarithm. More than atom_limit of the latter
  // skips the route, as any excess did before; Sums met only under comparisons (the threshold tests
  // of a lazily rescaled pivot, say) fill only the room the others leave, in their order (below).
  // With at most atom_limit candidates in all, the list is the same as before. At most atom_limit
  // Sums are decomposed either way.
  std::vector<Id> atoms, products; todo = {optimized}; seen.clear();
  std::unordered_set<Id> compared; size_t primary = 0;
  while (!todo.empty()) {
    tick(); const Id id = todo.back(); todo.pop_back();
    if (!seen.insert(id).second) continue;
    const auto &n = a.nodes[id];
    if (n.op == "fold" || n.op == "call") continue;
    const bool nonlinear = n.op == "recip" || n.op == "log" || n.op == "log2" || n.op == "compare";
    for (Id y : n.args) {
      const auto &m = a.nodes[y];
      if (nonlinear && sum_node(y) && m.facts.finite) {
        const bool known = std::find(atoms.begin(), atoms.end(), y) != atoms.end();
        if (!known) atoms.push_back(y);
        if (n.op == "compare") { if (!known) compared.insert(y); }
        else if (!known || compared.erase(y)) { if (++primary > atom_limit) return optimized; }
      } else if (nonlinear && m.op == "mul" && m.data.size() == 1 && m.args.size() <= 16 &&
                 m.facts.finite && products.size() < atom_limit &&
                 std::count_if(m.args.begin(), m.args.end(), sum_node) == 1 &&
                 std::find(products.begin(), products.end(), y) == products.end()) {
        products.push_back(y);
      }
      todo.push_back(y);
    }
  }
  if (atoms.size() > atom_limit) {
    std::vector<Id> kept; size_t room = atom_limit - primary;
    for (Id y : atoms)
      if (!compared.contains(y)) kept.push_back(y);
      else if (room > 0) { kept.push_back(y); --room; }
    atoms = std::move(kept);
    ++a.stats["linear_reduction_atom_comparison_cuts"];
  }
  atom_products = !products.empty();
  if (with_products) for (Id y : products) if (atoms.size() < atom_limit) atoms.push_back(y);
  std::unordered_map<Id, Id> replace;
  std::unordered_map<Id, std::vector<Id>> replaced_support;
  for (Id x : atoms)
    for (const auto &target : calls)
      if (const Id value = reduction_atom(x, target); value != invalid) {
        replace.emplace(x, value);
        if (auto it = atom_support_cache.find(std::tuple(x, residual_atoms, target.sid, target.parameters)); it != atom_support_cache.end())
          replaced_support.emplace(x, it->second);
        break;
      }
  if (replace.empty()) return optimized;
  a.stats["linear_reduction_reduction_atoms"] += replace.size();
  // Each replaced Sum X = C * call with C finite and positive has the sign of the call.
  for (const auto &[x, value] : replace) {
    const auto &v = a.nodes[value];
    if (v.op == "call") {
      if (atom_aliases.emplace(value, std::pair(x, a.one)).second && replaced_support.contains(x))
        atom_supports.emplace(value, replaced_support.at(x));
      continue;
    }
    if (v.op != "mul" || v.data.size() != 1 || v.data[0] <= 0) continue;
    const auto args = v.args;
    for (Id f : args)
      if (a.nodes[f].op == "call" && positive_multiple(value, f)) {
        std::vector<Id> rest{a.constant(v.data[0])};
        bool skipped = false;
        for (Id g : args) { if (g == f && !skipped) { skipped = true; continue; } rest.push_back(g); }
        if (atom_aliases.emplace(f, std::pair(x, a.mul(rest))).second && replaced_support.contains(x))
          atom_supports.emplace(f, replaced_support.at(x));
        break;
      }
  }
  // Rebuild with the replacements; Folds and Calls are kept.
  std::unordered_map<Id, Id> memo;
  std::function<Id(Id)> rebuild = [&](Id x) -> Id {
    if (auto it = replace.find(x); it != replace.end()) return it->second;
    if (auto it = memo.find(x); it != memo.end()) return it->second;
    tick();
    const auto &n = a.nodes[x];
    Id result = x;
    if (!n.args.empty() && n.op != "fold" && n.op != "call") {
      const auto args = n.args;
      std::vector<Id> rebuilt; bool changed = false;
      for (Id y : args) { rebuilt.push_back(rebuild(y)); changed |= rebuilt.back() != y; }
      if (changed) result = checker.reapply(x, rebuilt);
    }
    memo.emplace(x, result);
    return result;
  };
  return rebuild(optimized);
}

// y when x = 1 - y (an add of the constant 1 and -y), else invalid.
inline auto LinearReductionPattern::complement_of(Id x) const -> Id {
  const auto &n = a.nodes[x];
  if (n.op != "add" || n.args.size() != 2 || n.data.size() != 2) return invalid;
  for (unsigned i = 0; i < 2; ++i)
    if (n.args[i] == a.one && n.data[i] == 1 && n.data[1 - i] == -1) return n.args[1 - i];
  return invalid;
}

// w when x is the comparison of a 0/1 value w with 0 or 1 named by `same` (x = w on 0/1 values) or
// by `opposite` (x = 1 - w), else invalid; `negated` reports which.
inline auto LinearReductionPattern::boolean_comparison_operand(Id x, bool &negated) -> Id {
  const auto &n = a.nodes[x];
  if (n.op != "compare" || n.args.size() != 2 || !n.data.empty()) return invalid;
  const Id w = n.args[0], k = n.args[1];
  if (k != a.zero && k != a.one) return invalid;
  const std::string &t = n.tag;
  const bool same = k == a.zero ? (t == "ne" || t == "gt") : (t == "ge" || t == "eq");
  const bool opposite = k == a.zero ? (t == "eq" || t == "le") : (t == "lt" || t == "ne");
  if (!same && !opposite) return invalid;
  if (!zero_one(w)) return invalid;
  negated = opposite;
  return w;
}

// A reading that fails leaves members as it found them.
inline auto LinearReductionPattern::zero_forces(Id v, std::vector<Id> &members, size_t &work) -> bool {
  const size_t before = members.size();
  const bool read = zero_forces_value(v, members, work);
  if (!read) members.resize(before);
  return read;
}

inline auto LinearReductionPattern::one_forces(Id u, std::vector<Id> &members, size_t &work) -> bool {
  const size_t before = members.size();
  const bool read = one_forces_value(u, members, work);
  if (!read) members.resize(before);
  return read;
}

inline auto LinearReductionPattern::zero_forces_value(Id v, std::vector<Id> &members, size_t &work) -> bool {
  if (++work > vote_read_limit || !zero_one(v)) return false;
  const auto &n = a.nodes[v];
  if (n.op == "compare" && n.args.size() == 2) {
    if (n.tag == "lt" && n.args[1] == a.one) members.push_back(n.args[0]);
    bool negated = false;
    // c = [w != 0] (or [w > 0], [w >= 1]) of a 0/1 value w: c = 0 is w = 0. When w is not read,
    // c is still a comparison forced false, with no further member.
    if (const Id w = boolean_comparison_operand(v, negated); w != invalid && !negated)
      zero_forces(w, members, work);
    return true;
  }
  if (const Id u = complement_of(v); u != invalid) return one_forces(u, members, work);
  if (n.op == "fold" && n.tag == "Max" && n.args.size() == 2) return zero_forces(n.args[0], members, work);
  return false;
}

inline auto LinearReductionPattern::one_forces_value(Id u, std::vector<Id> &members, size_t &work) -> bool {
  if (++work > vote_read_limit || !zero_one(u)) return false;
  const auto &n = a.nodes[u];
  if (n.op == "mul" && n.data.size() == 1 && n.data[0] == 1) {
    // Every factor is a 0/1 value (checked first): only then is a product equal to 1 a product of
    // factors equal to 1. A factor that is not read adds no member.
    const auto factors = n.args;
    for (Id f : factors) if (!zero_one(f)) return false;
    for (Id f : factors) one_forces(f, members, work);
    return work <= vote_read_limit;
  }
  if (const Id v = complement_of(u); v != invalid) return zero_forces(v, members, work);
  bool negated = false;
  if (const Id w = boolean_comparison_operand(u, negated); w != invalid && negated)
    return zero_forces(w, members, work);
  return false;
}

// Rule: select(p, R*X, X) is R*X when p = 0 forces R = 1. Here p = 1 - prod_k (1 - [x_k < 1])
// (an any-of vote of comparisons; every factor is 0 or 1, so p = 0 makes each [x_k < 1] = 0), or
// the single comparison [x_0 < 1], R is one of the x_k, and R <= 1 in every case of its
// conditions (at_most_one), so p = 0 gives 1 <= R <= 1. The mirror select(prod_k (...), X, R*X)
// is R*X likewise. Where p is not of that chain shape, p is read through 0/1 values instead
// (zero_forces / one_forces; the vote as a Fold over the lanes): the arm chosen when p = 0
// (zero_forces(p)) or when p = 1 (one_forces(p); p is a 0/1 value, so p != 0 is p = 1) is the
// still arm X, and R must be among the comparisons [x < 1] that p's value there forces false.
// Returns the arm R*X or invalid. O(vote members + arm factors) plus the bounded case analysis
// of R.
inline auto LinearReductionPattern::vote_rescale_arm(Id node) -> Id {
  const auto &n = a.nodes[node];
  if (n.op != "select" || n.args.size() != 3) return invalid;
  const Id p = n.args[0];
  // Readings of p: which arm is chosen when no member holds, and the members.
  std::vector<std::pair<size_t, std::vector<Id>>> readings;
  const auto chain = chain_vote(p);
  if (chain) readings.push_back(*chain);
  else {
    std::vector<Id> members; size_t work = 0;
    if (zero_forces(p, members, work)) readings.emplace_back(2, std::move(members));
    members.clear(); work = 0;
    if (one_forces(p, members, work)) readings.emplace_back(1, std::move(members));
    if (readings.empty()) return invalid;
    ++a.stats["linear_reduction_vote_zero_one_readings"];
  }
  // The chain shape has one reading, decided exactly as before; a 0/1 reading may have two (the
  // arms decide which one has the moved factor).
  for (const auto &[quiet, members] : readings)
    if (const Id arm = vote_rescale_reading(n.args, quiet, members, !chain); arm != invalid || chain)
      return arm;
  return invalid;
}

// The chain shape of the rule above (a product of 1 - comparison factors, its complement, or one
// comparison [x < 1]): (quiet arm, members), else nothing.
inline auto LinearReductionPattern::chain_vote(Id p) -> std::optional<std::pair<size_t, std::vector<Id>>> {
  Id product = invalid; size_t quiet = 0;
  const auto &c = a.nodes[p];
  if (c.op == "add" && c.args.size() == 2 && c.data.size() == 2) {
    for (unsigned i = 0; i < 2; ++i)
      if (c.args[i] == a.one && c.data[i] == 1 && c.data[1 - i] == -1) { product = c.args[1 - i]; quiet = 2; }
  } else if (c.op == "mul") { product = p; quiet = 1; }
  std::vector<Id> members;
  // A vote of one member is the comparison itself: p = [R < 1], quiet on the second arm.
  const bool single = c.op == "compare" && c.tag == "lt" && c.args.size() == 2 && c.args[1] == a.one;
  if (single) { members.push_back(c.args[0]); quiet = 2; }
  else if (product == invalid || a.nodes[product].op != "mul" || a.nodes[product].data != std::vector<Q>{Q(1)}) return {};
  if (!single) for (Id f : a.nodes[product].args) {
    const auto &g = a.nodes[f];
    if (g.op != "add" || g.args.size() != 2 || g.data.size() != 2) return {};
    const size_t k = g.args[0] == a.one ? 0 : g.args[1] == a.one ? 1 : 2;
    if (k == 2 || g.data[k] != 1 || g.data[1 - k] != -1) return {};
    const auto &cmp = a.nodes[g.args[1 - k]];
    if (cmp.op != "compare") return {};
    if (cmp.tag == "lt" && cmp.args.size() == 2 && cmp.args[1] == a.one) members.push_back(cmp.args[0]);
  }
  return std::pair(quiet, std::move(members));
}

// The arms of select(p, ., .) for one reading: the still arm args[quiet] = X and the moved arm
// R*X, R among the members and at most 1 in every case of its conditions. Returns R*X or invalid.
inline auto LinearReductionPattern::vote_rescale_reading(const std::vector<Id> &args, size_t quiet, const std::vector<Id> &members,
                        bool zero_one_reading) -> Id {
  const Id still = args[quiet], moved = args[3 - quiet];
  const auto factors = [&](Id x) {
    return a.nodes[x].op == "mul" ? std::pair(a.nodes[x].args, a.nodes[x].data) : std::pair(std::vector<Id>{x}, std::vector<Q>{Q(1)});
  };
  const auto [fs, ds] = factors(still);
  const auto [fm, dm] = factors(moved);
  if (ds != dm || fm.size() != fs.size() + 1) return invalid;
  std::multiset<Id> rest(fm.begin(), fm.end());
  for (Id f : fs) { auto it = rest.find(f); if (it == rest.end()) return invalid; rest.erase(it); }
  const Id r = *rest.begin();
  if (std::find(members.begin(), members.end(), r) == members.end()) return invalid;
  CaseFix fixed; size_t cases = 0;
  bool bounded = at_most_one(r, fixed);
  try {
    if (!bounded) bounded = all_cases({r}, [&](const std::vector<Id> &v, const CaseFix &f) { return at_most_one(v[0], f); }, fixed, cases);
  } catch (const CaseMiss &) { bounded = false; }
  a.stats["linear_reduction_vote_bound_cases"] += cases;
  if (!bounded) {
    // Where the case analysis of all of R's conditions declines (it splits the innermost first and
    // stops at case_variable_limit), R's own Select tree is followed from its root instead.
    CaseFix path; size_t arms = 0;
    bounded = at_most_one_by_arms(r, path, arms);
    a.stats["linear_reduction_vote_arm_paths"] += arms;
    if (bounded) ++a.stats["linear_reduction_vote_arm_bounds"];
  }
  if (!bounded) { ++a.stats["linear_reduction_vote_rescale_declines"]; return invalid; }
  ++a.stats["linear_reduction_vote_rescale_selects"];
  if (zero_one_reading) ++a.stats["linear_reduction_vote_zero_one_selects"];
  return moved;
}

} // namespace rg
