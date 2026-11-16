#pragma once
#include "linear_reduction_pattern.hpp"

namespace rg {

inline auto LinearReductionPattern::fused_form(Id reference) -> std::optional<FusedForm> {
  const Polynomial poly = polynomial(reference);
  std::map<size_t, std::vector<Id>> parts;
  for (const auto &[monomial, coefficient] : poly) {
    scalar_tick();
    std::optional<size_t> member;
    std::vector<Id> rest{a.constant(coefficient)};
    for (Id atom : monomial) {
      std::optional<size_t> hit;
      for (size_t t = 0; t < targets.size(); ++t) if (target_call(atom, targets[t])) hit = t;
      if (!hit) { rest.push_back(atom); continue; }
      if (member) return {};
      member = hit;
    }
    if (member) parts[*member].push_back(a.mul(std::move(rest)));
  }
  if (parts.size() < 2) return {};
  FusedForm form;
  for (auto &[t, terms] : parts) {
    const Z &count = checker.specs[targets[t].sid].count;
    if (form.members.empty()) form.count = count;
    else if (count != form.count) return {};
    Id alpha = a.add(std::move(terms));
    if (!a.nodes[alpha].facts.finite) return {};
    form.members.push_back(t); form.alphas.push_back(alpha);
  }
  return form;
}

inline auto LinearReductionPattern::fused_body(const FusedForm &form, const Z &index) -> Id {
  std::vector<Id> parts;
  for (size_t m = 0; m < form.members.size(); ++m) {
    const auto &target = targets[form.members[m]];
    const auto &spec = checker.specs[target.sid];
    Id body = checker.instantiate_body(spec.pattern, capture(target, a.constant(Q(index))));
    if (body == invalid) return invalid;
    parts.push_back(a.mul({form.alphas[m], body}));
  }
  return a.add(std::move(parts));
}

inline auto LinearReductionPattern::add_closure_targets(Id reference) -> void {
  std::vector<Id> todo{reference};
  std::unordered_set<Id> seen{reference};
  while (!todo.empty() && targets.size() < max_closure_targets) {
    tick(); Id id = todo.back(); todo.pop_back();
    const auto &n = a.nodes[id];
    if (n.op == "call") {
      if (n.data.size() != 1 || n.data[0].denominator() != 1 || n.data[0] < 0 ||
          n.data[0] >= Q(checker.specs.size())) continue;
      Id sid = n.data[0].numerator().convert_to<Id>();
      const auto &spec = checker.specs[sid];
      if (!checker.active.contains(sid) || spec.maximum || !spec.facts.finite || !n.facts.finite || spec.count <= 0 ||
          n.args.size() + 1 != spec.slots || !checker.literal_binder_free(sid)) continue;
      if (std::none_of(targets.begin(), targets.end(), [&](const Target &t) { return target_call(id, t); }))
        targets.push_back({sid, n.args});
      continue;
    }
    if (n.op == "fold") continue;
    for (Id child : n.args) if (seen.insert(child).second) todo.push_back(child);
  }
}

// Integer value of a coordinate built from constants, one variable u,
// add/mul and floor idiv/imod, at u = value. Anything else has no value.
inline auto LinearReductionPattern::coordinate_at(Id id, Id u, const Z &value, unsigned depth) -> std::optional<Z> {
  if (depth > 16) return {};
  tick();
  const auto &n = a.nodes[id];
  if (id == u) return value;
  if (auto c = a.value(id)) { if (c->denominator() != 1) return {}; return c->numerator(); }
  if (n.op == "add" && n.args.size() == n.data.size()) {
    Z sum = 0;
    for (size_t j = 0; j < n.args.size(); ++j) {
      auto x = coordinate_at(n.args[j], u, value, depth + 1);
      if (!x || n.data[j].denominator() != 1) return {};
      sum += n.data[j].numerator() * *x;
    }
    return sum;
  }
  if (n.op == "mul" && n.data.size() == 1 && n.data[0].denominator() == 1) {
    Z product = n.data[0].numerator();
    for (Id child : n.args) {
      auto x = coordinate_at(child, u, value, depth + 1);
      if (!x) return {};
      product *= *x;
    }
    return product;
  }
  // Floor quotient and remainder by a positive constant (Arena idiv/imod).
  if ((n.op == "idiv" || n.op == "imod") && n.args.size() == 1 && n.data.size() == 1 &&
      n.data[0].denominator() == 1 && n.data[0] > 0) {
    auto x = coordinate_at(n.args[0], u, value, depth + 1);
    if (!x) return {};
    const Z m = n.data[0].numerator();
    Z q = *x / m, r = *x % m;
    if (r < 0) { r += m; --q; }
    return n.op == "idiv" ? q : r;
  }
  return {};
}

inline auto LinearReductionPattern::complete_by_cases(const Key &key, const Group &group) -> bool {
  const Z &count = checker.specs[targets[key.target].sid].count;
  if (group.multiplicity != count) return false;
  std::vector<Id> leaves, todo{group.piece};
  while (!todo.empty()) {
    tick(); const auto &p = checker.pieces[todo.back()]; todo.pop_back();
    if (p.kind == PieceNode::Concat) { todo.push_back(p.right); todo.push_back(p.left); }
    else if (p.kind == PieceNode::Leaf) leaves.push_back(p.index);
    else return false;
  }
  Id u = invalid;
  for (Id leaf : leaves) for (Id v : a.nodes[leaf].free) {
    if (u != invalid && v != u) return false;
    u = v;
  }
  const auto &var = a.nodes[u == invalid ? a.zero : u];
  if (u == invalid || var.op != "var" || !var.bounds || var.bounds->first.denominator() != 1 ||
      var.bounds->second.denominator() != 1 ||
      var.bounds->second - var.bounds->first + 1 > Q(max_case_values)) return false;
  const size_t total = count.convert_to<size_t>();
  for (Z value = var.bounds->first.numerator(); value <= var.bounds->second.numerator(); ++value) {
    std::vector<char> seen(total, 0);
    for (Id leaf : leaves) {
      auto x = coordinate_at(leaf, u, value);
      if (!x || *x < 0 || *x >= count) return false;
      if (seen[x->convert_to<size_t>()]++) return false;
    }
  }
  ++a.stats["linear_reduction_case_split_closures"];
  return true;
}

inline auto LinearReductionPattern::close_shared_sums(Id root) -> Id {
  std::unordered_map<Id, size_t> uses;
  std::vector<Id> todo{root};
  std::unordered_set<Id> seen{root};
  while (!todo.empty()) {
    tick(); Id id = todo.back(); todo.pop_back();
    const auto &n = a.nodes[id];
    const bool unary = unary_numeric(n.op);
    if (n.op != "add" && n.op != "add_raw" && n.op != "mul" && !unary) continue;
    for (Id child : n.args) {
      tick();
      const auto &c = a.nodes[child];
      // A sum under a unary numerical node (e.g. a norm under rsqrt) is
      // never a linear part of the outer sums, so it counts as shared.
      if ((n.op == "mul" || unary) && (c.op == "add" || c.op == "add_raw")) uses[child] += unary ? 2 : 1;
      if (seen.insert(child).second) todo.push_back(child);
    }
  }
  std::unordered_map<Id, Id> memo;
  std::vector<std::pair<Id, bool>> work{{root, false}};
  while (!work.empty()) {
    tick(); auto [id, ready] = work.back(); work.pop_back();
    if (memo.contains(id)) continue;
    const auto &n = a.nodes[id];
    if (n.op != "add" && n.op != "add_raw" && n.op != "mul" && !unary_numeric(n.op)) {
      memo.emplace(id, id); continue;
    }
    if (!ready) {
      work.emplace_back(id, true);
      for (Id child : n.args) if (!memo.contains(child)) work.emplace_back(child, false);
      continue;
    }
    std::vector<Id> args;
    bool changed = false;
    for (Id child : n.args) { args.push_back(memo.at(child)); changed |= args.back() != child; }
    Id value = changed ? checker.reapply(id, args) : id;
    if (id != root && uses[id] >= 2 && a.nodes[value].facts.finite &&
        (a.nodes[value].op == "add" || a.nodes[value].op == "add_raw")) {
      try {
        Form form = decompose(value);
        std::vector<Id> closed{materialize(form.residual)};
        bool any = false;
        for (const auto &[key, group] : form.groups) {
          if (complete(key, group) || complete_by_cases(key, group)) {
            closed.push_back(a.mul({key.coefficient, checker.call(targets[key.target].sid, targets[key.target].parameters)}));
            any = true;
          } else closed.push_back(materialize(group.witness));
        }
        if (any) { value = a.add(std::move(closed)); ++inner_closures; }
      } catch (const Miss &) {
        // A shared sum that does not close keeps its explicit form.
      } catch (const Unsupported &) {
        // Likewise when its local matcher declines.
      }
    }
    memo.emplace(id, value);
  }
  return memo.at(root);
}

// Fused proof of one explicit sum (under an outside product of factors):
// every matched term is W * c_j * term_j = F(i) for one index i, the matched
// indices cover [0, N) exactly once, and the reference equals the residual
// terms plus Sum_t alpha_t call_t. Work O(terms) candidate and scalar proofs.
inline auto LinearReductionPattern::fused_attempt(Id reference, Id optimized) -> std::optional<LinearReductionProof> {
  visits = piece_count = scalar_work = choices = 0; distributed_terms = 0; factored_sums = 0; instantiated_receipt = J::array(); attempt_nested_closures = attempt_nested_gaps = 0;
  closed_groups = partial_groups = 0; inner_closures = 0; factor_weights = false; closed_rewrite = invalid;
  targets.clear(); witnesses.clear(); original_witnesses.clear(); coverage_cache.clear();
  scalar_cache.clear(); witness_cache.clear(); materialized.clear(); completed_targets.clear();
  shared_targets_skipped = false;
  if (!checker.exponential_mode) return {};
  try {
    if (!collect_targets(reference, optimized, false)) return {};
    // A fused form needs two targets of one count; check before any work.
    bool pair = false;
    for (size_t x = 0; x < targets.size() && !pair; ++x)
      for (size_t y = x + 1; y < targets.size() && !pair; ++y)
        pair = checker.specs[targets[x].sid].count == checker.specs[targets[y].sid].count;
    if (!pair) return {};
    add_closure_targets(reference);
    const Id rewritten = close_shared_sums(optimized);
    targets.clear(); shared_targets_skipped = false;
    closed_rewrite = rewritten;
    if (!collect_targets(reference, rewritten, true)) throw Miss{"fused_targets"};
    auto form = fused_form(reference);
    if (!form) throw Miss{"fused_form"};
    const auto &root = a.nodes[rewritten];
    Id sum = invalid;
    std::vector<Id> outside;
    if (root.op == "add" || root.op == "add_raw") sum = rewritten;
    else if (root.op == "mul" && root.data.size() == 1) {
      outside.push_back(a.constant(root.data[0]));
      for (Id factor : root.args) {
        const auto &op = a.nodes[factor].op;
        if (op == "add" || op == "add_raw") {
          if (sum != invalid) throw Miss{"fused_shape"};
          sum = factor;
        } else outside.push_back(factor);
      }
    }
    if (sum == invalid || form->count > Z(a.nodes[sum].args.size())) throw Miss{"fused_shape"};
    const Id weight = a.mul(std::move(outside));
    const auto &spec = checker.specs[targets[form->members.front()].sid];
    const std::vector<Id> initial = capture(targets[form->members.front()], invalid);
    const size_t count = form->count.convert_to<size_t>();
    std::vector<char> covered(count, 0);
    std::vector<Id> residual;
    size_t matched = 0;
    const auto &terms = a.nodes[sum];
    for (size_t j = 0; j < terms.args.size(); ++j) {
      tick();
      const Id term = a.mul({weight, a.constant(terms.data[j]), terms.args[j]});
      std::optional<size_t> index;
      std::set<Z> tried;
      for (const auto &cap : checker.body_candidates(spec, term, initial)) {
        scalar_tick();
        auto value = cap.size() == spec.slots && cap[spec.axis] != invalid ? a.value(cap[spec.axis]) : std::nullopt;
        if (!value || value->denominator() != 1 || *value < 0 || *value >= Q(form->count) ||
            !tried.insert(value->numerator()).second) continue;
        const Id body = fused_body(*form, value->numerator());
        if (body != invalid && checker.scalar.equal(term, body)) { index = value->numerator().convert_to<size_t>(); break; }
      }
      if (!index) { residual.push_back(term); continue; }
      if (covered[*index]++) throw Miss{"fused_multiplicity"};
      ++matched;
    }
    if (matched != count) throw Miss{"fused_coverage"};
    for (size_t m = 0; m < form->members.size(); ++m) {
      const auto &target = targets[form->members[m]];
      residual.push_back(a.mul({form->alphas[m], checker.call(target.sid, target.parameters)}));
    }
    ++a.stats["ep_reduction_outer_obligations"];
    if (!checker.scalar.equal(reference, a.add(std::move(residual)))) throw Miss{"fused_output_scalar"};
    ++a.stats["linear_reduction_fused_proofs"];
    return LinearReductionProof{reference, {
        {"method", "fused_linear_reduction_certificate"},
        {"coverage", "exact_multiplicity_one_shared_index"},
        {"members", form->members.size()}, {"count", form->count.str()},
        {"matched_terms", matched}, {"inner_closures", inner_closures},
        {"visited_nodes", visits}, {"polynomial_work", scalar_work}}};
  } catch (const Miss &miss) {
    last_failure = miss.reason;
    ++a.stats["linear_reduction_decline_" + last_failure]; return {};
  } catch (const Unsupported &) {
    last_failure = "unsupported_local_match";
    ++a.stats["linear_reduction_decline_" + last_failure]; return {};
  }
}

} // namespace rg
