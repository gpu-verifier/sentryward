#pragma once
#include "linear_reduction_pattern.hpp"

namespace rg {

// What a lemma's case analysis may read besides its three nodes: the Arena's constructors, and in
// resolve() this pattern's atom_aliases (Call T -> (X, C)). Every alias comes from reduction_atom: an
// exact, complete decomposition X = C * call(T) with C = 1 or C a finite positive co-factor
// (substitute_reduction_atoms), i.e. an identity on every input, so a lemma proved with it holds in every
// later proof. support_finite (chain_case_equal's support facts) is empty whenever a lemma is decided
// (prepare_support_finite fills it only after its own lemmas); a lemma decided with nonempty support
// facts would still be true but is not shared (kept out of the cache by the check below).
inline auto LinearReductionPattern::chain_lemma(int kind, Id u, Id v, Id w) -> bool {
  const auto key = std::tuple(kind, u, v, w);
  if (auto it = checker.proved_chain_lemmas.find(key); it != checker.proved_chain_lemmas.end()) {
    ++a.stats["linear_reduction_chain_lemma_reuses"];
    // A reused proof that accepted a case by identical non-finite values is counted again, as a fresh
    // proof is (case_equal), so that the family's label reports it.
    if (it->second) ++a.stats["linear_reduction_case_nonfinite_identities"];
    return true;
  }
  if (declined_chain_lemmas.contains(key)) return false;
  bool ok = false;
  const auto identities_before = a.stats["linear_reduction_case_nonfinite_identities"];
  {
    ChainScope scope(*this, false);
    if (kind == 0) ok = case_equal(u, v, w);
    else {
      CaseFix fixed; size_t cases = 0;
      try {
        ok = all_cases({v, u}, [&](const std::vector<Id> &r, const CaseFix &f) {
               return case_truth(r[1], f) == 0 || case_truth(r[0], f) == 1; }, fixed, cases, {1});
      } catch (const CaseMiss &) { ok = false; }
      a.stats["linear_reduction_chain_implication_cases"] += cases;
    }
  }
  const bool nonfinite_identity = a.stats["linear_reduction_case_nonfinite_identities"] != identities_before;
  if (!ok) declined_chain_lemmas.insert(key);
  else if (support_finite.empty()) checker.proved_chain_lemmas.emplace(key, nonfinite_identity);
  ++a.stats[ok ? "linear_reduction_chain_lemmas" : "linear_reduction_chain_lemma_declines"];
  return ok;
}

// The predicates [v != -inf] of the values v that x or y compares with -inf (eq or ne; v finite or
// -inf and not finite by its facts), nearest to x and y first (Folds and Calls not entered), at most
// chain_predicate_limit of them within chain_predicate_depth operand edges.
inline auto LinearReductionPattern::finiteness_predicates(Id x, Id y) -> std::vector<Id> {
  std::vector<Id> values, level{x, y}, next;
  std::unordered_set<Id> seen;
  for (size_t depth = 0; depth <= chain_predicate_depth && !level.empty() && values.size() < chain_predicate_limit; ++depth) {
    for (Id id : level) {
      if (!seen.insert(id).second) continue;
      tick();
      const std::string op = a.nodes[id].op, tag = a.nodes[id].tag;
      const std::vector<Id> args = a.nodes[id].args;
      if (op == "compare" && (tag == "eq" || tag == "ne") && args.size() == 2)
        for (unsigned i = 0; i < 2; ++i) {
          const Id v = args[1 - i];
          if (a.neginf(args[i]) && !a.nodes[v].facts.finite && a.nodes[v].facts.finite_or_negative_infinity() &&
              std::find(values.begin(), values.end(), v) == values.end() && values.size() < chain_predicate_limit)
            values.push_back(v);
        }
      if (op == "fold" || op == "call") continue;
      next.insert(next.end(), args.begin(), args.end());
    }
    level.swap(next); next.clear();
  }
  std::vector<Id> predicates;
  for (Id v : values) predicates.push_back(a.compare("ne", v, a.infinity(-1)));
  return predicates;
}

// For each group (coefficient, guard) of one target: whether it joins the base coefficient by a walk
// (above); false for the base itself and for a group not in `walk` (when given), which is only an
// intermediate coefficient. Coefficients are distinct; a group whose guard is invalid has guard 1.
inline auto LinearReductionPattern::finiteness_chain_joins(const std::vector<std::pair<Id, Id>> &groups, Id base,
                                         const std::vector<bool> &walk, std::vector<Id> *terminal) -> std::vector<bool> {
  if (terminal) terminal->assign(groups.size(), invalid);
  std::map<Id, std::multiset<Id>> factors;
  for (const auto &[c, guard] : groups) {
    const auto &n = a.nodes[c];
    factors.emplace(c, n.op == "mul" && n.data.size() == 1 ? std::multiset<Id>(n.args.begin(), n.args.end())
                                                          : std::multiset<Id>{c});
  }
  std::vector<bool> joins(groups.size(), false);
  if (!factors.contains(base)) return joins;
  const auto distance = [&](Id u, Id v) {
    const auto &fu = factors.at(u), &fv = factors.at(v);
    std::vector<Id> d;
    std::set_symmetric_difference(fu.begin(), fu.end(), fv.begin(), fv.end(), std::back_inserter(d));
    return d.size();
  };
  std::map<Id, Id> successor;
  for (const auto &[c, guard] : groups) {
    if (c == base) continue;
    const size_t own = distance(c, base);
    Id best = invalid; std::pair<size_t, size_t> rank;
    for (const auto &[d, other_guard] : groups) {
      if (d == c) continue;
      const size_t nearer = distance(d, base);
      if (nearer >= own) continue;
      const std::pair<size_t, size_t> r{distance(c, d), nearer};
      if (best == invalid || r < rank) { best = d; rank = r; }
    }
    if (best != invalid) successor.emplace(c, best);
  }
  std::map<Id, Id> invariant;  // c -> F with c = successor(c) wherever F holds; invalid: none found
  const auto link = [&](Id c) -> Id {
    if (auto it = invariant.find(c); it != invariant.end()) return it->second;
    const Id s = successor.at(c);
    const auto [x, y] = without_common_factors(c, s);
    Id found = invalid;
    for (Id f : finiteness_predicates(x, y))
      if (chain_lemma(0, x, y, f) || ((x != c || y != s) && chain_lemma(0, c, s, f))) { found = f; break; }
    invariant.emplace(c, found);
    return found;
  };
  for (size_t j = 0; j < groups.size(); ++j) {
    Id c = groups[j].first, premise = groups[j].second == invalid ? a.one : groups[j].second;
    if (c == base || (!walk.empty() && !walk[j])) continue;
    bool joined = true;
    for (size_t links = 0; c != base; ++links) {
      const auto next = successor.find(c);
      if (next == successor.end() || links >= groups.size()) { joined = false; break; }
      const Id f = link(c);
      if (f == invalid || !chain_lemma(1, premise, f, invalid)) { joined = false; break; }
      premise = f; c = next->second;
    }
    joins[j] = joined;
    if (joined && terminal) (*terminal)[j] = premise;
  }
  return joins;
}

inline auto LinearReductionPattern::merge_by_finiteness_chain(Form &form, const std::vector<Key> &keys, const Key &base, std::map<Id, Id> &premise) -> void {
  // Groups merged already keep their coefficients as intermediate nodes of a walk.
  std::vector<std::pair<Id, Id>> groups;
  std::vector<bool> walk;
  for (const auto &key : keys) {
    const bool present = form.groups.contains(key);
    groups.emplace_back(key.coefficient, present ? form.groups.at(key).guard : a.one);
    walk.push_back(present);
  }
  std::vector<Id> terminal;
  const auto joins = finiteness_chain_joins(groups, base.coefficient, walk, &terminal);
  for (size_t j = 0; j < keys.size(); ++j) {
    const Key &key = keys[j];
    if (!joins[j] || !form.groups.contains(key)) continue;
    premise.emplace(key.coefficient, terminal[j]);
    const Group moved = form.groups.at(key);
    form.groups.erase(key);
    Group &into = form.groups.at(base);
    into.piece = concat(into.piece, moved.piece);
    into.witness = sum_witness(into.witness, moved.witness);
    into.multiplicity += moved.multiplicity;
    if (into.guard != moved.guard) into.guard = a.one;
    ++a.stats["linear_reduction_chain_merges"];
  }
}

// The values finite or -inf and not finite by their facts nearest to the roots (Folds and Calls not
// entered), at most chain_predicate_limit within chain_predicate_depth operand edges.
inline auto LinearReductionPattern::extended_values(const std::vector<Id> &roots) -> std::vector<Id> {
  std::vector<Id> values, level = roots, next;
  std::unordered_set<Id> seen;
  for (size_t depth = 0; depth <= chain_predicate_depth && !level.empty() && values.size() < chain_predicate_limit; ++depth) {
    for (Id id : level) {
      if (!seen.insert(id).second) continue;
      tick();
      const auto &n = a.nodes[id];
      if (!n.facts.finite && n.facts.finite_or_negative_infinity() && !a.neginf(id) && values.size() < chain_predicate_limit)
        values.push_back(id);
      if (n.op == "fold" || n.op == "call") continue;
      next.insert(next.end(), n.args.begin(), n.args.end());
    }
    level.swap(next); next.clear();
  }
  return values;
}

inline auto LinearReductionPattern::prepare_support_finite(const std::vector<Id> &roots) -> void {
  support_finite.clear();
  if (atom_supports.empty()) return;
  const std::vector<Id> values = extended_values(roots);
  std::vector<std::pair<Id, Id>> facts;
  for (const auto &[call, premises] : atom_supports) {
    const auto alias = atom_aliases.find(call);
    if (alias == atom_aliases.end() || premises.empty()) continue;
    for (Id v : values) {
      const Id finite = a.compare("ne", v, a.infinity(-1));
      if (std::all_of(premises.begin(), premises.end(), [&](Id p) { return chain_lemma(1, p, finite, invalid); }))
        facts.emplace_back(alias->second.first, v);
    }
  }
  support_finite = std::move(facts);
}

// The exact case analysis of x and y (case_equal) in the chain mode, after the ordinary analysis
// declined: values a case fixes finite are their finite views (support facts included), the nearest
// open condition is split first, empty cases hold, and proved Sums are used as they are.
inline auto LinearReductionPattern::chain_case_equal(Id x, Id y) -> bool {
  ChainScope scope(*this, true);
  prepare_support_finite({x, y});
  const bool ok = case_equal(x, y, a.one);
  ++a.stats[ok ? "linear_reduction_chain_output_proofs" : "linear_reduction_chain_output_declines"];
  return ok;
}

} // namespace rg
