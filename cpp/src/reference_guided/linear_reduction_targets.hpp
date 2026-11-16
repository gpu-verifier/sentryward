#pragma once
#include "linear_reduction_pattern.hpp"

namespace rg {

inline auto LinearReductionPattern::collect_targets(Id reference, Id optimized, bool reuse_existing) -> bool {
  std::vector<Id> todo{reference}; std::unordered_set<Id> seen;
  std::map<std::pair<Id, std::vector<Id>>, Id> found;
  while (!todo.empty()) {
    tick(); Id id = todo.back(); todo.pop_back();
    if (!seen.insert(id).second) continue;
    const auto &n = a.nodes[id];
    if (n.op == "call") {
      if (n.data.size() != 1 || n.data[0].denominator() != 1 || n.data[0] < 0 ||
          n.data[0] >= Q(checker.specs.size())) return false;
      Id sid = n.data[0].numerator().convert_to<Id>();
      const auto &spec = checker.specs[sid];
      if (!checker.active.contains(sid) || spec.maximum || !spec.facts.finite || !n.facts.finite ||
          spec.count <= 0 || n.args.size() + 1 != spec.slots || !checker.literal_binder_free(sid)) return false;
      found.emplace(std::pair(sid, n.args), id);
      if (found.size() > budget.targets) throw Miss{"target_budget"};
    } else if (n.op == "add" || n.op == "add_raw" || n.op == "mul") {
      todo.insert(todo.end(), n.args.begin(), n.args.end());
    }
  }
  // Do not scan the optimized DAG when there is no eligible reference call.
  if (found.empty()) return false;
  if (!reuse_existing) {
    for (const auto &[key, id] : found)
      targets.push_back({key.first, key.second});
    return true;
  }
  // A call already present in the optimized linear expression is an exact
  // scalar atom. Re-proving its reduction would make a product of that
  // scalar and an unfinished reduction look like two participating groups.
  // This scan records presence only; final polynomial equality still checks
  // every occurrence, parameter, coefficient, and surrounding operation.
  std::unordered_set<Id> existing_calls;
  todo = {optimized}; seen.clear();
  while (!todo.empty()) {
    tick(); Id id = todo.back(); todo.pop_back();
    if (!seen.insert(id).second) continue;
    const auto &n = a.nodes[id];
    if (n.op == "call") existing_calls.insert(id);
    else if (n.op == "add" || n.op == "add_raw" || n.op == "mul")
      todo.insert(todo.end(), n.args.begin(), n.args.end());
  }
  for (const auto &[key, id] : found) {
    if (existing_calls.contains(id)) {
      shared_targets_skipped = true;
      ++a.stats["linear_reduction_existing_scalar_calls"];
      continue;
    }
    targets.push_back({key.first, key.second});
  }
  return !targets.empty();
}

// Products are flattened by Arena. Try bounded factor subsets, then verify
// each selected product with the ordinary exact matcher. Binding later proves
// that any extracted factor is independent of the reduction coordinate.
inline auto LinearReductionPattern::match(Id value) -> std::optional<Candidate> {
  const auto &n = a.nodes[value];
  if (!n.facts.finite) return {};
  if (n.op == "mul" && n.args.size() > 16) return {};
  std::optional<Candidate> best; bool ambiguous = false;
  size_t local_choices = 0;
  auto accept = [&](Candidate c) {
    if (!a.nodes[c.coefficient].facts.finite) return;
    if (!best || c.specificity > best->specificity) { best = c; ambiguous = false; }
    else if (c.specificity == best->specificity &&
             std::tie(c.target, c.index, c.coefficient) !=
             std::tie(best->target, best->index, best->coefficient)) ambiguous = true;
  };
  // An exponent closure reached by the matcher below sees this pattern as its owner (owned_closure:
  // charged here when this pattern is itself a closure query).
  OwnerScope owner(checker.closure_owner, this);
  for (size_t t = 0; t < targets.size(); ++t) {
    const auto &target = targets[t];
    const auto &p = checker.patterns.nodes[checker.specs[target.sid].pattern];
    if (!checker.exponential_mode && p.op != n.op) continue;
    size_t specificity = p.op == "mul" ? p.args.size() : 1;
    for (const auto &m : checker.match_term(target.sid, value, target.parameters))
      if (m.params == target.parameters) accept({t, specificity, m.index, m.coef});
    if (n.op != "mul" || p.data.size() != 1 || n.data.size() != 1 || p.data[0] == 0 ||
        p.args.empty() || p.args.size() > n.args.size()) continue;
    if (p.args.size() == n.args.size() && p.data == n.data) continue;
    const size_t take = p.args.size(), size = n.args.size();
    std::vector<size_t> selected(take);
    for (size_t j = 0; j < take; ++j) selected[j] = j;
    while (true) {
      tick(); ++choices;
      if (++local_choices > budget.factor_choices) throw Miss{"factor_budget"};
      // Both product arity and the number of local subset choices are bounded.
      std::vector<Id> chosen{a.constant(p.data[0])}, rest{a.constant(n.data[0] / p.data[0])};
      size_t cursor = 0;
      for (size_t j = 0; j < size; ++j) {
        if (cursor < take && selected[cursor] == j) { chosen.push_back(n.args[j]); ++cursor; }
        else rest.push_back(n.args[j]);
      }
      Id candidate = a.mul(chosen);
      for (const auto &m : checker.match_term(target.sid, candidate, target.parameters))
        if (m.params == target.parameters)
          accept({t, specificity, m.index, a.mul({m.coef, a.mul(rest)})});
      size_t j = take;
      while (j && selected[j - 1] == size - take + j - 1) --j;
      if (j == 0) break;
      ++selected[j - 1];
      for (size_t k = j; k < take; ++k) selected[k] = selected[k - 1] + 1;
    }
  }
  if (ambiguous) { ++a.stats["linear_reduction_ambiguous_terms"]; return {}; }
  if (!best && nested_mode) return nested_match(value);
  return best;
}

inline auto LinearReductionPattern::nested_body(const Spec &spec) const -> std::optional<NestedBody> {
  const auto &p = checker.patterns.nodes[spec.pattern];
  if (p.op != "mul" || p.data.size() != 1 || p.data[0] == 0 || p.args.size() > 16) return {};
  NestedBody result;
  for (Id factor : p.args) {
    if (checker.patterns.nodes[factor].op != "call") { result.rest.push_back(factor); continue; }
    if (result.call != invalid) return {};
    result.call = factor;
  }
  if (result.call == invalid) return {};
  return result;
}

// The nested target at a concrete value of the enclosing binder: the
// nested Call's arguments instantiated at the capture. pattern() slots an
// enclosing reference binder as an explicit Call parameter, so the binder's
// value arrives through these arguments; a nested spec that still holds a
// binder literal is refused (literal_binder_free) instead.
inline auto LinearReductionPattern::nested_target(Id call_pattern, const std::vector<Id> &cap) -> std::optional<Target> {
  const auto &cp = checker.patterns.nodes[call_pattern];
  if (cp.data.size() != 1 || cp.data[0].denominator() != 1 || cp.data[0] < 0 ||
      cp.data[0] >= Q(checker.specs.size())) return {};
  const Id sid = cp.data[0].numerator().convert_to<Id>();
  const Spec &nested = checker.specs[sid];
  if (nested.maximum || !nested.facts.finite || cp.args.size() + 1 != nested.slots ||
      !checker.literal_binder_free(sid)) return {};
  std::vector<Id> args;
  for (Id x : cp.args) {
    const Id v = checker.instantiate_body(x, cap);
    if (v == invalid) return {};
    args.push_back(v);
  }
  return Target{sid, std::move(args)};
}

inline auto LinearReductionPattern::closes_as(Id value, const Target &target) -> bool {
  const auto key = std::tuple(value, target.sid, target.parameters);
  if (auto it = closure_cache.find(key); it != closure_cache.end()) return it->second;
  LinearReductionPattern inner(checker, remaining_budget());
  inner.closure_query = closure_query;  // charged here, so part of an enclosing closure query
  inner.targets = {target};
  bool closed = false, exhausted = false;
  try {
    Form form = inner.decompose(value);
    if (form.residual == invalid && form.groups.size() == 1) {
      const auto &[group_key, group] = *form.groups.begin();
      closed = group_key.coefficient == a.one && inner.complete(group_key, group);
    }
  } catch (const Miss &miss) {
    const std::string_view reason = miss.reason;
    exhausted = reason == "node_budget" || reason == "piece_budget" || reason == "polynomial_budget";
  } catch (const Unsupported &) {}
  charge(inner);
  // A decline that ran out of the remaining budget may close later with
  // more. Every other decline depends only on (value, target) and the fixed
  // structural limits, so it is final.
  if (closed || !exhausted) closure_cache.emplace(key, closed);
  return closed;
}

inline auto LinearReductionPattern::nested_match(Id value) -> std::optional<Candidate> {
  const auto &n = a.nodes[value];
  if (n.op != "mul" || n.data.size() != 1 || n.args.size() > 16 || !n.facts.finite) return {};
  for (size_t t = 0; t < targets.size(); ++t) {
    const auto &target = targets[t];
    const Spec &spec = checker.specs[target.sid];
    auto body = nested_body(spec);
    if (!body) continue;
    const std::vector<Id> initial = capture(target, invalid);
    const Q scale = n.data[0] / checker.patterns.nodes[spec.pattern].data[0];
    for (const auto &cap : checker.body_candidates(spec, value, initial)) {
      tick();
      if (cap.size() != spec.slots || std::find(cap.begin(), cap.end(), invalid) != cap.end()) continue;
      auto index = a.value(cap[spec.axis]);
      if (!index || index->denominator() != 1 || *index < 0 || *index >= Q(spec.count)) continue;
      std::vector<Id> rest = n.args;
      bool present = true;
      for (Id factor : body->rest) {
        const Id expected = checker.instantiate_body(factor, cap);
        auto it = std::find(rest.begin(), rest.end(), expected);
        if (expected == invalid || it == rest.end()) { present = false; break; }
        rest.erase(it);
      }
      if (!present || rest.empty()) continue;
      // The factors outside G(r) form S; a flattened product is one S too.
      const Id inner = rest.size() == 1 ? rest.front() : a.mul(std::move(rest));
      auto nested = nested_target(body->call, cap);
      if (!nested || !closes_as(inner, *nested)) continue;
      ++a.stats["linear_reduction_nested_closures"]; ++attempt_nested_closures;
      return Candidate{t, body->rest.size() + 1, cap[spec.axis], a.constant(scale)};
    }
  }
  return {};
}

// Coverage with nested-empty gaps: plain constant leaves, each once; at
// every missing r the nested Call at the instantiated arguments has a mask
// that is zero on its whole domain (existing zero_on_interval), so
// body_t(r) = G(r) * 0. At most
// max_case_values missing indices; O(count + gaps) work.
inline auto LinearReductionPattern::complete_with_nested_gaps(const Key &key, const Group &group) -> bool {
  const auto &target = targets[key.target];
  const Spec &spec = checker.specs[target.sid];
  auto body = nested_body(spec);
  if (!body || group.multiplicity >= spec.count || spec.count > Z(max_nested_count)) return false;
  const size_t count = spec.count.convert_to<size_t>();
  std::vector<char> present(count, 0);
  std::vector<Id> todo{group.piece};
  while (!todo.empty()) {
    tick(); const auto &p = checker.pieces[todo.back()]; todo.pop_back();
    if (p.kind == PieceNode::Concat) { todo.push_back(p.right); todo.push_back(p.left); continue; }
    if (p.kind != PieceNode::Leaf) return false;
    auto index = a.value(p.index);
    if (!index || index->denominator() != 1 || *index < 0 || *index >= Q(spec.count) ||
        present[index->numerator().convert_to<size_t>()]++) return false;
  }
  size_t gaps = 0;
  for (size_t r = 0; r < count; ++r) {
    if (present[r]) continue;
    if (++gaps > max_case_values) return false;
    auto nested = nested_target(body->call, capture(target, a.constant(Q(Z(r)))));
    if (!nested) return false;
    const Z &inner = checker.specs[nested->sid].count;
    if (!zero_on_interval(*nested, Z(0), inner)) return false;
  }
  ++a.stats["linear_reduction_nested_gap_completions"]; ++attempt_nested_gaps;
  return true;
}

inline auto LinearReductionPattern::decompose(Id root) -> Form {
  // The scale is part of the memo identity, not a rewrite of the Fold body.
  // Work is bounded by stored (node,weight) states and their operand edges,
  // not by logical reduction length. Distinct weights may create distinct
  // states; tick()/distributed_terms limit that real cost explicitly.
  using State = std::pair<Id, Id>;
  enum class Kind { Add, Bind, WeightedChild, Product, Scaled, Alias };
  // scale: for Scaled, the factor applied to the child's form (the whole weight when invalid).
  struct Plan { Kind kind; std::vector<State> children; Id axis = invalid; Z count = 0; Id scale = invalid; };
  std::map<State, Form> memo;
  std::map<State, Plan> plans;
  State start{root, a.one};
  std::vector<std::pair<State, bool>> todo{{start, false}};
  while (!todo.empty()) {
    tick(); auto [state, ready] = todo.back(); todo.pop_back();
    if (memo.contains(state)) continue;
    auto [id, weight] = state;
    const auto &n = a.nodes[id];
    if (!ready) {
      ++a.stats["linear_reduction_weighted_states"];
      Id weighted = weight == a.one ? id : a.mul({weight, id});
      Form form; bool identified = false;
      for (size_t t = 0; t < targets.size(); ++t) if (target_call(id, targets[t])) {
        const Z count = checker.specs[targets[t].sid].count;
        Id var = a.var("linear-reduction-domain:" + std::to_string(targets[t].sid), 0, Q(count - 1));
        insert(form, {t, weight}, {bind_piece(leaf(var), var, count), original(weighted), count, a.one});
        identified = true; break;
      }
      // Complete weighted summands still go through the ordinary mandatory
      // EP body proof. An anchor match alone never establishes equality.
      if (!identified) {
        auto candidate = match(weighted);
        // The full weighted term is needed when the weight belongs to the
        // reference body (e.g. coordinate-dependent input scales). If only
        // the unweighted body matches, retain the known outside coefficient
        // explicitly instead of asking a leaf pattern to absorb it. This
        // reuses the same exact body matcher; bound() still checks that the
        // resulting coefficient is independent of every crossed binder.
        if (!candidate && weight != a.one && a.nodes[weight].facts.finite) {
          candidate = match(id);
          if (candidate)
            candidate->coefficient = a.mul({weight, candidate->coefficient});
        }
        // Rule: (m * r) * x = r * (m * x). A weight of several factors may hold the reference
        // summand's mask m (a zero-armed Select) next to factors that are not part of it (another
        // mask such as a guarded reciprocal, a scale). Each mask factor is tried alone with the
        // summand; the rest of the weight is the known outside coefficient, as above. At most
        // four more matches per summand, only where the two matches above found nothing.
        if (!candidate && weight != a.one && a.nodes[weight].facts.finite) {
          const auto &w = a.nodes[weight];
          if (w.op == "mul" && w.data.size() == 1 && w.args.size() >= 2 && w.args.size() <= 4) {
            const auto factors = w.args; const Q scalar = w.data[0];
            for (size_t j = 0; j < factors.size() && !candidate; ++j) {
              const auto &m = a.nodes[factors[j]];
              if (m.op != "select" || m.args.size() != 3 || (m.args[1] != a.zero && m.args[2] != a.zero)) continue;
              candidate = match(a.mul({factors[j], id}));
              if (!candidate) continue;
              std::vector<Id> rest{a.constant(scalar)};
              for (size_t k = 0; k < factors.size(); ++k) if (k != j) rest.push_back(factors[k]);
              candidate->coefficient = a.mul({a.mul(rest), candidate->coefficient});
              ++a.stats["linear_reduction_single_mask_weights"];
            }
          }
        }
        if (candidate) {
          insert(form, {candidate->target, candidate->coefficient},
                 {leaf(candidate->index), original(weighted), Z(1),
                  reference_guard(targets[candidate->target], candidate->index)});
          identified = true;
        }
      }
      if (identified) { memo.emplace(state, std::move(form)); continue; }
      std::optional<Plan> plan;
      if (n.facts.finite && a.nodes[weight].facts.finite) {
        // Masks of a weighted Sum whose own terms carry none: zero-armed Select factors of the
        // weight. They stay with each term (a distributed state, within the distribution budget),
        // so a term can match a masked reference summand; the rest of the weight scales the
        // child's form as in the factored case. Without room in the budget, nothing changes.
        Id masks = a.one, unmasked = weight;
        const bool sum = (n.op == "add" || n.op == "add_raw") && n.args.size() == n.data.size();
        if (sum && weight != a.one) {
          const auto zero_armed = [&](Id x) {
            const auto &m = a.nodes[x];
            return m.op == "select" && m.args.size() == 3 && (m.args[1] == a.zero || m.args[2] == a.zero);
          };
          const auto &w = a.nodes[weight];
          std::vector<Id> mask_factors, other{a.constant(w.op == "mul" && w.data.size() == 1 ? w.data[0] : Q(1))};
          for (Id f : w.op == "mul" && w.data.size() == 1 ? w.args : std::vector<Id>{weight})
            (zero_armed(f) ? mask_factors : other).push_back(f);
          bool bare = true;
          for (Id term : n.args) {
            const auto &m = a.nodes[term];
            if (zero_armed(term) || (m.op == "mul" && std::any_of(m.args.begin(), m.args.end(), zero_armed))) { bare = false; break; }
          }
          if (!mask_factors.empty() && bare && distributed_terms <= budget.distributed_terms &&
              n.args.size() <= budget.distributed_terms - distributed_terms) {
            masks = a.mul(mask_factors); unmasked = a.mul(other);
          }
        }
        if (masks != a.one && unmasked != a.one && outside_binders(unmasked, root)) {
          plan = Plan{Kind::Scaled, {{id, masks}}};
          plan->scale = unmasked;
          ++a.stats["linear_reduction_masked_weighted_sums"];
        } else if (sum && weight != a.one && masks == a.one && factor_weights && outside_binders(weight, root)) {
          // w * Sum_j c_j x_j: the unweighted Sum is one memoized state.
          plan = Plan{Kind::Scaled, {{id, a.one}}};
          ++factored_sums; ++a.stats["linear_reduction_factored_weighted_sums"];
        } else if ((n.op == "add" || n.op == "add_raw") && n.args.size() == n.data.size()) {
          if (weight != a.one) {
            if (distributed_terms > budget.distributed_terms ||
                n.args.size() > budget.distributed_terms - distributed_terms) {
              ++a.stats["linear_reduction_distribution_budget_declines"];
              throw Miss{"distribution_budget"};
            }
            distributed_terms += n.args.size();
            a.stats["linear_reduction_weighted_add_edges"] += n.args.size();
            ++a.stats["linear_reduction_distributed_products"];
          }
          plan = Plan{Kind::Add, {}};
          for (size_t j = 0; j < n.args.size(); ++j) {
            tick();
            plan->children.emplace_back(n.args[j], a.mul({weight, a.constant(n.data[j])}));
          }
        } else if (n.op == "fold" && n.tag == "Sum" && n.args.size() == 2 &&
                   n.data.size() == 2 && n.data[0].denominator() == 1 &&
                   n.data[1].denominator() == 1 && n.data[0] >= 0 &&
                   n.data[0] < Q(a.nodes.size()) && n.data[1] >= 0) {
          Id axis = n.data[0].numerator().convert_to<Id>();
          // An outside free variable must not be captured by this binder.
          if (!a.depends(weight, axis) && !a.depends(n.args[0], axis)) {
            plan = Plan{Kind::Bind, {{n.args[0], weight}}, axis, n.data[1].numerator()};
            if (plan->count != 0) plan->children.emplace_back(n.args[1], weight);
          }
        } else if (n.op == "mul" && n.data.size() == 1) {
          size_t selected = 0, number = 0;
          // Canonical scalar products are flattened by Arena. Inspect only
          // their operands; nonlinear nodes and Calls remain boundaries.
          for (size_t j = 0; j < n.args.size(); ++j) {
            tick();
            const auto &child = a.nodes[n.args[j]];
            bool shell = child.op == "add" || child.op == "add_raw" ||
                         (child.op == "fold" && child.tag == "Sum");
            if (shell) { ++number; selected = j; }
          }
          if (number == 1) {
            std::vector<Id> factors{weight, a.constant(n.data[0])};
            for (size_t j = 0; j < n.args.size(); ++j) {
              tick(); if (j != selected) factors.push_back(n.args[j]);
            }
            plan = Plan{Kind::WeightedChild, {{n.args[selected], a.mul(factors)}}};
          } else {
            // Products of reductions are not expanded into a Cartesian
            // domain. Retain the existing single-participating-group rule.
            plan = Plan{Kind::Product, {}};
            for (Id child : n.args) { tick(); plan->children.emplace_back(child, a.one); }
          }
        }
      }
      // A vote-guarded rescale select(p, R*X, X) is the arm R*X (vote_rescale_arm); decompose it.
      if (!plan && n.op == "select" && n.facts.finite && a.nodes[weight].facts.finite)
        if (const Id arm = vote_rescale_arm(id); arm != invalid) plan = Plan{Kind::Alias, {{arm, weight}}};
      if (!plan) {
        form.residual = original(weighted); memo.emplace(state, std::move(form)); continue;
      }
      todo.emplace_back(state, true);
      for (const State &child : plan->children)
        if (!memo.contains(child)) todo.emplace_back(child, false);
      plans.emplace(state, std::move(*plan));
      continue;
    }
    Form form;
    const Plan &plan = plans.at(state);
    if (plan.kind == Kind::Add) {
      for (const State &child : plan.children) { tick(); append(form, memo.at(child), a.one); }
    } else if (plan.kind == Kind::Scaled) {
      append(form, memo.at(plan.children[0]), plan.scale == invalid ? weight : plan.scale);
    } else if (plan.kind == Kind::Alias) {
      append(form, memo.at(plan.children[0]), a.one);
    } else if (plan.kind == Kind::WeightedChild) {
      append(form, memo.at(plan.children[0]), a.one);
    } else if (plan.kind == Kind::Bind) {
      append(form, memo.at(plan.children[0]), a.one); // weighted seed, once
      if (plan.count != 0)
        append(form, bound(memo.at(plan.children[1]), plan.axis, plan.count), a.one);
    } else {
      size_t participating = 0, chosen = 0;
      for (size_t j = 0; j < plan.children.size(); ++j) {
        tick(); if (!memo.at(plan.children[j]).groups.empty()) { ++participating; chosen = j; }
      }
      if (participating == 1) {
        std::vector<Id> factors{weight, a.constant(n.data[0])};
        for (size_t j = 0; j < n.args.size(); ++j) {
          tick(); if (j != chosen) factors.push_back(n.args[j]);
        }
        append(form, memo.at(plan.children[chosen]), a.mul(factors));
      } else if (participating == 0) {
        // Rule: w * (S_1 ... S_k * P) = (w * S_1 ... S_k) * P. Finite Select factors with no
        // zero arm (data-dependent scales such as a rescale; a zero-armed Select is a mask and
        // stays with the summand) join the weight, and the rest P is decomposed as one
        // weighted summand, exactly as a weight outside the product. Only where no factor of
        // the product participates; one more state per such product.
        const auto scale = [&](Id f) {
          const auto &m = a.nodes[f];
          return m.op == "select" && m.args.size() == 3 && m.facts.finite &&
                 m.args[1] != a.zero && m.args[2] != a.zero;
        };
        // A product without such a factor is left exactly as before (no work is charged).
        std::vector<Id> scales{weight}, rest{a.constant(n.data[0])};
        if (std::any_of(n.args.begin(), n.args.end(), scale))
          for (Id f : n.args) { tick(); (scale(f) ? scales : rest).push_back(f); }
        if (scales.size() > 1 && rest.size() > 1) {
          const State child{a.mul(rest), a.mul(scales)};
          if (a.nodes[child.first].facts.finite && a.nodes[child.second].facts.finite) {
            ++a.stats["linear_reduction_scale_factor_weights"];
            plans.insert_or_assign(state, Plan{Kind::WeightedChild, {child}});
            todo.emplace_back(state, true);
            if (!memo.contains(child)) todo.emplace_back(child, false);
            continue;
          }
        }
      }
    }
    if (form.groups.empty()) {
      form.residual = original(weight == a.one ? id : a.mul({weight, id}));
    }
    memo.emplace(state, std::move(form));
  }
  return std::move(memo.at(start));
}

} // namespace rg
