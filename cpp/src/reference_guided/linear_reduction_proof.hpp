#pragma once
#include "linear_reduction_pattern.hpp"

namespace rg {

// The Checker's exponent closure (Checker::exponent_closure, installed by the comparison). One
// closure query (value, spec, parameters) runs close_target in its own pattern on one standard
// linear-reduction budget. Every closure that query reaches in turn (a query from its own term
// match: Checker::closure_owner, set by OwnerScope in match, is then a closure pattern) runs on what
// the enclosing query has left and is charged to it, the contract of every nested sub-proof, so all
// work of a query, nested queries included, is bounded by that one budget. A nested query cut off by
// the enclosing budget unwinds with the enclosing query's Miss and is not stored. Every query's
// result is stored (Checker::exponent_closure_cache: the identity, or invalid when the decomposition
// did not close within its budget) and a recurring query costs nothing, so the total closure work is
// at most one budget per distinct query, however many matches reach it. Exact: only close_target's
// own result is returned.
inline auto LinearReductionPattern::owned_closure(Checker &checker, Id value, Id sid, const std::vector<Id> &parameters) -> std::optional<Id> {
  auto key = std::tuple(value, sid, parameters);
  if (auto it = checker.exponent_closure_cache.find(key); it != checker.exponent_closure_cache.end()) {
    ++checker.a.stats["masked_exponent_closure_reuses"];
    if (it->second == invalid) return std::nullopt;
    // The Call's construction registers its active definition (Checker::call): kept on reuse.
    checker.call(sid, parameters);
    return it->second;
  }
  LinearReductionPattern *owner = checker.closure_owner;
  const bool nested = owner && owner->closure_query;
  LinearReductionPattern closure(checker, nested ? owner->remaining_budget() : LinearReductionBudget{});
  closure.mark_closure_query();
  const auto closed = closure.close_target(value, sid, parameters);
  if (nested) owner->charge(closure);
  else checker.a.stats["masked_exponent_closure_nodes"] += closure.visits;  // nested queries' work included
  ++checker.a.stats[nested ? "masked_exponent_closure_nested_queries" : "masked_exponent_closure_queries"];
  checker.exponent_closure_cache.emplace(std::move(key), closed ? *closed : invalid);
  return closed;
}

// The value as R + C * call(sid, parameters) when its decomposition with that single target
// (the ordinary exact route: weighted terms, coordinate-dependent input scales kept with each
// term, mandatory body proofs, case-split merges) has exactly one group, it is complete
// (exact multiplicity-one coverage), and its coefficient C is finite; R is the materialized
// residual. nullopt otherwise. Used by the Checker's masked matcher to close a reduction inside
// a merged exponent (owned_closure). This pattern's budgets apply.
inline auto LinearReductionPattern::close_target(Id value, Id sid, const std::vector<Id> &parameters) -> std::optional<Id> {
  visits = piece_count = scalar_work = choices = 0; distributed_terms = 0;
  targets = {Target{sid, parameters}}; witnesses.clear(); original_witnesses.clear(); coverage_cache.clear();
  scalar_cache.clear(); witness_cache.clear(); materialized.clear(); completed_targets.clear();
  try {
    Form form = decompose(value);
    merge_case_equal_groups(form);
    if (form.groups.size() != 1) return {};
    const auto &[key, group] = *form.groups.begin();
    if (!a.nodes[key.coefficient].facts.finite || !complete(key, group)) return {};
    ++a.stats["linear_reduction_target_closures"];
    // The witness forest's raw sums (materialize keeps them unflattened) are rebuilt as ordinary
    // sums where finite, so the caller reads the residual's terms. Same value.
    std::unordered_map<Id, Id> flat;
    const std::function<Id(Id)> ordinary = [&](Id x) -> Id {
      if (auto it = flat.find(x); it != flat.end()) return it->second;
      tick();
      const auto &n = a.nodes[x];
      Id result = x;
      if (n.facts.finite && n.op == "add_raw" && n.args.size() == n.data.size()) {
        const auto args = n.args; const auto data = n.data;
        std::vector<Id> terms;
        for (size_t j = 0; j < args.size(); ++j) terms.push_back(a.mul({a.constant(data[j]), ordinary(args[j])}));
        result = a.add(std::move(terms));
      }
      flat.emplace(x, result);
      return result;
    };
    return a.add({ordinary(materialize(form.residual)), a.mul({key.coefficient, checker.call(sid, parameters)})});
  } catch (const Miss &) { return {}; } catch (const Unsupported &) { return {}; }
}

inline auto LinearReductionPattern::prove_attempt(Id reference, Id optimized,
                                                 bool reuse_existing) -> std::optional<LinearReductionProof> {
  visits = piece_count = scalar_work = choices = 0; distributed_terms = 0; factored_sums = 0; instantiated_receipt = J::array(); attempt_nested_closures = attempt_nested_gaps = 0;
  closed_groups = partial_groups = 0; local_form_merges = 0;
  targets.clear(); witnesses.clear(); original_witnesses.clear(); coverage_cache.clear();
  scalar_cache.clear(); witness_cache.clear(); materialized.clear(); completed_targets.clear(); last_failure.clear();
  shared_targets_skipped = false;
  try {
    if (!collect_targets(reference, optimized, reuse_existing)) return {};
    ++a.stats["linear_reduction_pattern_attempts"];
    const Polynomial expected = checker.exponential_mode ? Polynomial{} : polynomial(reference);
    Form form = decompose(optimized);
    merge_case_equal_groups(form);
    // Every successful certificate must close at least one group for every
    // reference target. Check this existing necessary condition before
    // expanding residual polynomials, which can be large even on a miss.
    std::set<Key> complete_groups;
    for (const auto &[key, group] : form.groups) {
      if (complete(key, group) || (nested_mode && complete_with_nested_gaps(key, group))) {
        complete_groups.insert(key);
        completed_targets.insert(key.target);
      }
    }
    std::map<size_t, Id> members;
    if (checker.exponential_mode)
      for (size_t t = 0; t < targets.size(); ++t)
        if (!completed_targets.contains(t))
          if (Id sum = instantiated_target(targets[t]); sum != invalid) {
            members.emplace(t, sum);
            completed_targets.insert(t);
            instantiated_receipt.push_back({{"spec", targets[t].sid},
                                            {"count", checker.specs[targets[t].sid].count.str()}});
          }
    if (completed_targets.size() != targets.size()) throw Miss{"incomplete_targets"};
    const Id expanded = members.empty() ? reference : expand_targets(reference, members);
    Polynomial actual;
    std::vector<Id> ep_terms;
    if(checker.exponential_mode)ep_terms.push_back(materialize(form.residual));
    else actual=witness_polynomial(form.residual);
    J closures = J::array();
    for (const auto &[key, group] : form.groups) {
      if (complete_groups.contains(key)) {
        const auto &target = targets[key.target];
        Id value = a.mul({key.coefficient, checker.call(target.sid, target.parameters)});
        if(checker.exponential_mode)ep_terms.push_back(value);
        else add_polynomial(actual, polynomial(value));
        ++closed_groups;
        closures.push_back({{"spec", target.sid}, {"count", checker.specs[target.sid].count.str()},
                            {"coefficient", a.render(key.coefficient, 4)}});
      } else {
        ++partial_groups;
        if(checker.exponential_mode)ep_terms.push_back(materialize(group.witness));
        else add_polynomial(actual, witness_polynomial(group.witness));
      }
    }
    if(checker.exponential_mode) {
      ++a.stats["ep_reduction_outer_obligations"];
      // Rule (one spelling per running maximum, Checker::flat_maxima): when groups of this attempt merged
      // by equal local forms (two spellings of one running maximum in their coefficients), the kernel value
      // is taken with each maximum replaced by its flat local form (equal values), so its reduction atoms
      // and coefficients spell each running maximum once. Exact.
      const Id kernel_value = local_form_merges ? checker.flat_maxima(a.add(ep_terms)) : a.add(ep_terms);
      if (local_form_merges) ++a.stats["linear_reduction_flat_maxima_outputs"];
      if(!checker.scalar.equal(expanded,kernel_value)) {
        // Reduction atoms of the kernel value, then the exact case analysis of both values;
        // where that declines and a factored Sum is a further atom candidate, once more with it.
        const auto closes = [&](Id substituted) {
          return (substituted != kernel_value && checker.scalar.equal(expanded, substituted)) ||
                 case_equal(expanded, substituted, a.one) || chain_case_equal(expanded, substituted);
        };
        const Id substituted = substitute_reduction_atoms(expanded, kernel_value);
        if (!closes(substituted)) {
          const Id wider = atom_products ? substitute_reduction_atoms(expanded, kernel_value, true) : substituted;
          if ((wider == substituted || !closes(wider)) && !light_attempt(expanded, kernel_value)) throw Miss{"ep_output_scalar"};
        }
        ++a.stats["linear_reduction_case_split_outputs"];
      }
    } else if (expected != actual) {
      // Coverage and residual accounting stay unchanged. Use the shared
      // scalar engine for the final algebraic comparison in EP mode.
      auto expression = [&](const Polynomial &poly) {
        std::vector<Id> terms;
        for (const auto &[key, coefficient] : poly) {
          scalar_tick();
          std::vector<Id> factors(key); factors.push_back(a.constant(coefficient));
          terms.push_back(a.mul(std::move(factors)));
        }
        return a.add(std::move(terms));
      };
      if (!checker.exponential_mode ||
          !checker.scalar.equal(expression(expected), expression(actual)))
        throw Miss{"output_polynomial"};
    }
    ++a.stats["linear_reduction_pattern_proofs"];
    a.stats["linear_reduction_closed_groups"] += closed_groups;
    a.stats["linear_reduction_preserved_partial_groups"] += partial_groups;
    J receipt{
      {"method", "linear_reduction_combination_certificate"}, {"closures", closures},
      {"coverage", "exact_multiplicity_one_per_closed_group"},
      {"preserved_partial_groups", partial_groups}, {"visited_nodes", visits},
      {"contribution_pieces", piece_count}, {"witness_nodes", witnesses.size()},
      {"weighted_add_edges", distributed_terms},
      {"factor_choices", choices}, {"polynomial_work", scalar_work},
      {"output_polynomial_terms", actual.size()}};
    if (factor_weights) receipt["factored_weighted_sums"] = factored_sums;
    if (!instantiated_receipt.empty()) receipt["instantiated_targets"] = instantiated_receipt;
    if (nested_mode) {
      receipt["route"] = "nested";
      receipt["nested_closures"] = attempt_nested_closures;
      receipt["nested_gap_completions"] = attempt_nested_gaps;
    }
    return LinearReductionProof{reference, std::move(receipt)};
  } catch (const Miss &miss) {
    last_failure = miss.reason;
    ++a.stats["linear_reduction_decline_" + last_failure]; return {};
  } catch (const Unsupported &) {
    // An optional recognizer exhausting the existing local matcher's budget
    // must not turn an already imported comparison into a translation blocker.
    last_failure = "unsupported_local_match";
    ++a.stats["linear_reduction_decline_" + last_failure]; return {};
  }
}

// Term-wise distribution keeps a weight that belongs to the reference body
// matchable. Only when it exhausts its budget is the same attempt repeated
// with binder-independent weights factored out; that adds at most one more
// bounded attempt and leaves every distribution-budget-free proof unchanged.
inline auto LinearReductionPattern::attempt(Id reference, Id optimized, bool reuse_existing) -> std::optional<LinearReductionProof> {
  factor_weights = false;
  auto result = prove_attempt(reference, optimized, reuse_existing);
  if (result || last_failure != "distribution_budget") return result;
  ++a.stats["linear_reduction_factored_weight_retries"];
  FlagScope factored(factor_weights, true);
  return prove_attempt(reference, optimized, reuse_existing);
}

inline auto LinearReductionPattern::direct(Id reference, Id optimized) -> std::optional<LinearReductionProof> {
  auto result = attempt(reference, optimized, true);
  if (result || !shared_targets_skipped) return result;
  // A shared call can coexist with still-expanded contributions to that
  // same sum (for example a second LayerNorm coefficient). Preserve the old
  // full-target proof as one bounded fallback; never discard the residual.
  ++a.stats["linear_reduction_shared_target_fallbacks"];
  return attempt(reference, optimized, false);
}

// select(p, A, B) = select(p, A', B') for every p when A = A' and B = B' as
// total functions. Selects are descended only while both sides test the
// same predicate node. Every remaining arm pair is proved by the scalar
// engine or by a direct reduction proof (never by a further descent), so
// the work is the pairs' sub-proofs, each with this pattern's budgets.
inline auto LinearReductionPattern::select_arms(Id reference, Id optimized) -> std::optional<LinearReductionProof> {
  auto same_predicate = [&](Id l, Id r) {
    const auto &x = a.nodes[l], &y = a.nodes[r];
    return x.op == "select" && y.op == "select" && x.args.size() == 3 &&
           y.args.size() == 3 && x.args[0] == y.args[0];
  };
  if (!same_predicate(reference, optimized)) return {};
  size_t selects = 0, scalar_pairs = 0;
  J reductions = J::array();
  std::vector<std::pair<Id, Id>> todo{{reference, optimized}};
  while (!todo.empty()) {
    a.check();
    auto [l, r] = todo.back(); todo.pop_back();
    if (same_predicate(l, r)) {
      ++selects;
      todo.emplace_back(a.nodes[l].args[2], a.nodes[r].args[2]);
      todo.emplace_back(a.nodes[l].args[1], a.nodes[r].args[1]);
      continue;
    }
    if (selects + scalar_pairs + reductions.size() > max_select_arm_pairs) {
      last_failure = "select_arm_budget";
      ++a.stats["linear_reduction_decline_" + last_failure]; return {};
    }
    if (checker.scalar.equal(l, r)) { ++scalar_pairs; continue; }
    LinearReductionPattern arm(checker, budget);
    auto proof = arm.direct(l, r);
    if (!proof) {
      last_failure = "select_arm" + (arm.last_failure.empty() ? "" : ":" + arm.last_failure);
      ++a.stats["linear_reduction_select_arm_declines"]; return {};
    }
    reductions.push_back(std::move(proof->receipt));
  }
  ++a.stats["linear_reduction_select_arm_proofs"];
  return LinearReductionProof{reference, {
      {"method", "select_arm_congruence"}, {"coverage", "same_predicate_nodes_equal_arms"},
      {"selects", selects}, {"scalar_arm_pairs", scalar_pairs},
      {"reduction_arm_pairs", reductions.size()}, {"first_reduction_arm", reductions.empty() ? J() : reductions[0]}}};
}

// Whether a root-level reference call has a nested-call body (cheap scan
// through add/mul shells; the nested route runs only then).
inline auto LinearReductionPattern::nested_targets(Id reference) -> bool {
  std::vector<Id> todo{reference};
  std::unordered_set<Id> seen{reference};
  size_t work = 0;
  while (!todo.empty()) {
    a.check(); Id id = todo.back(); todo.pop_back();
    if (++work > budget.nodes) return false;
    const auto &n = a.nodes[id];
    if (n.op == "call" && n.data.size() == 1 && n.data[0].denominator() == 1 && n.data[0] >= 0 &&
        n.data[0] < Q(checker.specs.size()) &&
        nested_body(checker.specs[n.data[0].numerator().convert_to<Id>()])) return true;
    if (n.op == "add" || n.op == "add_raw" || n.op == "mul")
      for (Id child : n.args) if (seen.insert(child).second) todo.push_back(child);
  }
  return false;
}

} // namespace rg
