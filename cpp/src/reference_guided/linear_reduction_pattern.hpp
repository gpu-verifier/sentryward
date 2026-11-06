#pragma once
#include "checker.hpp"

namespace rg {
struct LinearReductionBudget {
  uint64_t nodes = 100000, pieces = 30000;
  size_t groups = 32, targets = 4;
  size_t factor_choices = 128, polynomial_terms = 512, polynomial_degree = 24;
  size_t distributed_terms = 128;
  uint64_t polynomial_work = 100000;
  // Members of a reference target that may be expanded explicitly when no
  // group closes it: a short explicit sum, not a reduction to enumerate.
  size_t instantiated_members = 8;
};
struct LinearReductionProof { Id replacement = invalid; J receipt; };

// A single exact decomposition, not a collection of competing explanations:
// value = residual + sum(group.coefficient * target(group.coverage)).
// Every group also retains its actual value as a persistent witness. A partial
// match (including the output's g_j) therefore remains part of the expression.
class LinearReductionPattern {
  Checker &checker;
  Arena &a;
  LinearReductionBudget budget;
  struct Miss { const char *reason; };
  uint64_t visits = 0, piece_count = 0, scalar_work = 0, choices = 0;
  size_t distributed_terms = 0;
  // Retry mode after a distribution-budget miss: a weighted Sum whose weight
  // depends on no binder below the decomposed root is decomposed once without
  // the weight, and its form is scaled (exact distributivity, O(groups)).
  bool factor_weights = false;
  // Receipt record of the current attempt: instantiated targets, and the
  // nested closures and nested gap completions it used.
  J instantiated_receipt = J::array();
  size_t attempt_nested_closures = 0, attempt_nested_gaps = 0;
  // Restores a mode flag on every exit, including a Timeout.
  struct FlagScope {
    bool &flag; bool saved;
    FlagScope(bool &f, bool value) : flag(f), saved(f) { flag = value; }
    ~FlagScope() { flag = saved; }
  };
  size_t factored_sums = 0;
  size_t closed_groups = 0, partial_groups = 0;
  struct Target { Id sid; std::vector<Id> parameters; };
  std::vector<Target> targets;
  struct Key {
    size_t target; Id coefficient;
    bool operator<(const Key &o) const {
      return std::tie(target, coefficient) < std::tie(o.target, o.coefficient);
    }
  };
  // The witness denotes the weighted value, including the coefficient.
  struct Group { Id piece, witness; Z multiplicity; };
  struct Form { Id residual = invalid; std::map<Key, Group> groups; };
  struct Witness {
    enum Kind { Original, Add, Scale, Bind } kind;
    Id left = invalid, right = invalid, value = invalid, variable = invalid;
    Z count = 0;
  };
  std::vector<Witness> witnesses;
  std::unordered_map<Id, Id> original_witnesses;
  std::map<std::pair<Id, size_t>, bool> coverage_cache;
  std::set<size_t> completed_targets;
  bool shared_targets_skipped = false;

  void tick() { a.check(); if (++visits > budget.nodes) throw Miss{"node_budget"}; }
  // A nested closure sub-proof runs with what remains of this pattern's
  // budgets, and its work is charged back here afterwards, so one attempt's
  // node, piece and polynomial budgets bound its nested sub-proofs too.
  LinearReductionBudget remaining_budget() const {
    LinearReductionBudget rest = budget;
    rest.nodes = budget.nodes > visits ? budget.nodes - visits : 0;
    rest.pieces = budget.pieces > piece_count ? budget.pieces - piece_count : 0;
    rest.polynomial_work = budget.polynomial_work > scalar_work ? budget.polynomial_work - scalar_work : 0;
    return rest;
  }
  void charge(const LinearReductionPattern &child) {
    visits += child.visits; piece_count += child.piece_count; scalar_work += child.scalar_work;
    if (visits > budget.nodes) throw Miss{"node_budget"};
    if (piece_count > budget.pieces) throw Miss{"piece_budget"};
    if (scalar_work > budget.polynomial_work) throw Miss{"polynomial_budget"};
  }
  void scalar_tick() {
    a.check(); if (++scalar_work > budget.polynomial_work) throw Miss{"polynomial_budget"};
  }
  Id witness(Witness n) {
    tick(); witnesses.push_back(std::move(n)); return Id(witnesses.size() - 1);
  }
  Id original(Id n) {
    if (n == a.zero) return invalid;
    if (auto it = original_witnesses.find(n); it != original_witnesses.end()) return it->second;
    Id result = witness({Witness::Original, invalid, invalid, n});
    original_witnesses.emplace(n, result); return result;
  }
  Id sum_witness(Id x, Id y) {
    if (x == invalid) return y;
    if (y == invalid) return x;
    return witness({Witness::Add, x, y});
  }
  Id scale_witness(Id x, Id coefficient) {
    if (x == invalid || coefficient == a.zero) return invalid;
    if (coefficient == a.one) return x;
    return witness({Witness::Scale, x, invalid, coefficient});
  }
  Id bind_witness(Id x, Id variable, const Z &count) {
    if (x == invalid || count == 0) return invalid;
    return witness({Witness::Bind, x, invalid, invalid, variable, count});
  }
  void piece_tick() {
    tick(); if (++piece_count > budget.pieces) throw Miss{"piece_budget"};
  }
  Id leaf(Id index) { piece_tick(); return checker.leaf(index); }
  Id concat(Id x, Id y) { piece_tick(); return checker.concat(x, y); }
  Id bind_piece(Id p, Id v, const Z &count) {
    piece_tick(); return checker.bind_piece(p, v, count);
  }
  void insert(Form &form, const Key &key, const Group &group) {
    if (key.coefficient == a.zero || group.multiplicity == 0) return;
    auto [it, added] = form.groups.emplace(key, group);
    if (!added) {
      it->second.piece = concat(it->second.piece, group.piece);
      it->second.witness = sum_witness(it->second.witness, group.witness);
      it->second.multiplicity += group.multiplicity;
    }
    if (form.groups.size() > budget.groups) throw Miss{"group_budget"};
  }
  void append(Form &to, const Form &from, Id coefficient) {
    if (coefficient == a.zero) return;
    if (!a.nodes[coefficient].facts.finite) throw Miss{"nonfinite_coefficient"};
    to.residual = sum_witness(to.residual, scale_witness(from.residual, coefficient));
    for (const auto &[key, group] : from.groups) {
      tick();
      insert(to, {key.target, a.mul({key.coefficient, coefficient})},
             {group.piece, scale_witness(group.witness, coefficient), group.multiplicity});
    }
  }
  Form bound(const Form &body, Id variable, const Z &count) {
    Form result;
    if (count == 0) return result;
    Id residual = body.residual;
    for (const auto &[key, group] : body.groups) {
      tick(); const auto &target = targets[key.target];
      bool independent = !a.depends(key.coefficient, variable) &&
        std::none_of(target.parameters.begin(), target.parameters.end(),
                     [&](Id p) { return a.depends(p, variable); });
      if (!independent) {
        residual = sum_witness(residual, group.witness);
        ++a.stats["linear_reduction_dependent_groups_preserved"];
      } else {
        insert(result, key, {bind_piece(group.piece, variable, count),
            bind_witness(group.witness, variable, count), group.multiplicity * count});
      }
    }
    result.residual = bind_witness(residual, variable, count);
    return result;
  }

  // The slot capture of a target's spec: its parameters in the non-axis
  // slots and axis_value (or invalid) in the axis slot.
  std::vector<Id> capture(const Target &target, Id axis_value) const {
    const Spec &spec = checker.specs[target.sid];
    std::vector<Id> cap(spec.slots, invalid);
    size_t parameter = 0;
    for (size_t slot = 0; slot < spec.slots; ++slot)
      cap[slot] = slot == spec.axis ? axis_value : target.parameters[parameter++];
    return cap;
  }
  bool target_call(Id node, const Target &target) const {
    const auto &n = a.nodes[node];
    return n.op == "call" && n.data == std::vector<Q>{Q(target.sid)} &&
           n.args == target.parameters;
  }
  bool collect_targets(Id reference, Id optimized, bool reuse_existing) {
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
  struct Candidate {
    size_t target, specificity; Id index, coefficient;
  };
  // Products are flattened by Arena. Try bounded factor subsets, then verify
  // each selected product with the ordinary exact matcher. Binding later proves
  // that any extracted factor is independent of the reduction coordinate.
  std::optional<Candidate> match(Id value) {
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

  // Nested reductions. A root-level target whose body is a product with one
  // nested reduction call, body_t(r) = G(r) * C(args(r)), may have a nested
  // definition C that depends on the target's own reduction binder (e.g. a
  // triangular mask s < r): pattern() slots that binder as an argument of C,
  // so instantiating body_t at a concrete r instantiates C's arguments at r.
  // A term matches body_t(r) when its factors are G(r) times the rest S, and
  // S closes as exactly one complete group of C at those arguments (masked
  // gaps included).
  bool nested_mode = false;
  // Bounds the presence bitmap of a nested-gap completion (one byte per index).
  static constexpr size_t max_nested_count = 65536;
  struct NestedBody { Id call = invalid; std::vector<Id> rest; };
  std::optional<NestedBody> nested_body(const Spec &spec) const {
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
  std::optional<Target> nested_target(Id call_pattern, const std::vector<Id> &cap) {
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
  // Memoized per (value, target): a closure is an exact property of these
  // immutable nodes and specs. The memo lives as long as this pattern.
  std::map<std::tuple<Id, Id, std::vector<Id>>, bool> closure_cache;
  bool closes_as(Id value, const Target &target) {
    const auto key = std::tuple(value, target.sid, target.parameters);
    if (auto it = closure_cache.find(key); it != closure_cache.end()) return it->second;
    LinearReductionPattern inner(checker, remaining_budget());
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
  std::optional<Candidate> nested_match(Id value) {
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
  bool complete_with_nested_gaps(const Key &key, const Group &group) {
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
  // The free variables of the decomposed root are output coordinates; every
  // binder crossed by bound() below it is absent from that set. A weight over
  // root-free variables only therefore keeps each scaled coefficient
  // independent of every crossed binder. Cost O(|free(weight)| log |free(root)|).
  bool outside_binders(Id weight, Id root) const {
    const auto &free = a.nodes[root].free;
    return std::all_of(a.nodes[weight].free.begin(), a.nodes[weight].free.end(),
                       [&](Id v) { return std::binary_search(free.begin(), free.end(), v); });
  }
  Form decompose(Id root) {
    // The scale is part of the memo identity, not a rewrite of the Fold body.
    // Work is bounded by stored (node,weight) states and their operand edges,
    // not by logical reduction length. Distinct weights may create distinct
    // states; tick()/distributed_terms limit that real cost explicitly.
    using State = std::pair<Id, Id>;
    enum class Kind { Add, Bind, WeightedChild, Product, Scaled };
    struct Plan { Kind kind; std::vector<State> children; Id axis = invalid; Z count = 0; };
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
          insert(form, {t, weight}, {bind_piece(leaf(var), var, count), original(weighted), count});
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
          if (candidate) {
            insert(form, {candidate->target, candidate->coefficient},
                   {leaf(candidate->index), original(weighted), Z(1)});
            identified = true;
          }
        }
        if (identified) { memo.emplace(state, std::move(form)); continue; }
        std::optional<Plan> plan;
        if (n.facts.finite && a.nodes[weight].facts.finite) {
          if ((n.op == "add" || n.op == "add_raw") && n.args.size() == n.data.size() &&
              weight != a.one && factor_weights && outside_binders(weight, root)) {
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
        append(form, memo.at(plan.children[0]), weight);
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
        }
      }
      if (form.groups.empty()) {
        form.residual = original(weight == a.one ? id : a.mul({weight, id}));
      }
      memo.emplace(state, std::move(form));
    }
    return std::move(memo.at(start));
  }
  // A missing interval can count as a contribution only when the complete
  // reference summand has a finite, explicitly zero-valued mask there. Inspect
  // only a shallow product/Select shell; the predicate operands have a separate
  // small pattern-DAG budget. Unknown bounds are a miss, never a zero proof.
  bool zero_on_interval(const Target &target, const Z &lo, const Z &hi) {
    const Spec &spec = checker.specs[target.sid];
    if (lo >= hi || !spec.facts.finite || spec.maximum || spec.slots > 128 ||
        target.parameters.size() + 1 != spec.slots) return false;
    const auto &root = checker.patterns.nodes[spec.pattern];
    if (root.args.size() > 16) return false;
    std::vector<Id> candidates;
    if (root.op == "select") candidates.push_back(spec.pattern);
    else if (root.op == "mul") candidates = root.args;
    else return false;
    std::vector<Id> cap = capture(target, invalid);
    // A fresh binder is confined to this proof. Its interval is the exact gap,
    // not a widening of an existing binder whose facts may already be used.
    cap[spec.axis] = a.var("linear-zero-gap:" + std::to_string(target.sid) + ":" +
                          lo.str() + ":" + hi.str(), Q(lo), Q(hi - 1));
    auto is_zero = [&](Id id) {
      const auto &n = checker.patterns.nodes[id];
      if (n.op == "const") return n.data.size() == 1 && n.data[0] == 0;
      if (n.op != "slot" || n.data.size() != 1 || n.data[0].denominator() != 1 ||
          n.data[0] < 0 || n.data[0] >= Q(cap.size())) return false;
      return cap[n.data[0].numerator().convert_to<size_t>()] == a.zero;
    };
    auto small_predicate_operand = [&](Id id) {
      std::vector<Id> todo{id}; std::unordered_set<Id> seen;
      size_t edges = 0;
      while (!todo.empty()) {
        tick(); Id next = todo.back(); todo.pop_back();
        if (!seen.insert(next).second) continue;
        if (seen.size() > 64) return false;
        const auto &n = checker.patterns.nodes[next];
        if (n.args.size() > 8 || n.args.size() > 128 - edges) return false;
        edges += n.args.size();
        if (n.op != "slot" && n.op != "const" && n.op != "var" &&
            n.op != "add" && n.op != "mul" && n.op != "affine_slot" &&
            n.op != "affine_param") return false;
        todo.insert(todo.end(), n.args.begin(), n.args.end());
      }
      return true;
    };
    for (Id candidate : candidates) {
      tick(); const auto &mask = checker.patterns.nodes[candidate];
      if (mask.op != "select" || mask.args.size() != 3) continue;
      const bool true_zero = is_zero(mask.args[1]), false_zero = is_zero(mask.args[2]);
      if (!true_zero && !false_zero) continue;
      const auto &predicate = checker.patterns.nodes[mask.args[0]];
      if (predicate.op != "compare" || predicate.args.size() != 2 ||
          !small_predicate_operand(predicate.args[0]) ||
          !small_predicate_operand(predicate.args[1])) continue;
      Id lhs = checker.instantiate_body(predicate.args[0], cap);
      Id rhs = checker.instantiate_body(predicate.args[1], cap);
      if (lhs == invalid || rhs == invalid) continue;
      Id truth = a.compare(predicate.tag, lhs, rhs);
      if ((true_zero && truth == a.one) || (false_zero && truth == a.zero)) {
        ++a.stats["linear_reduction_zero_gap_proofs"];
        return true;
      }
    }
    return false;
  }
  // A target that no complete group closes and whose domain has at most
  // instantiated_members members is expanded instead: its call denotes
  // sum_k body(k), so the final EP obligation compares these explicit members
  // with the optimized contributions one by one, including a member whose
  // factor folds to a constant there (exp(0) on a diagonal). Returns that sum,
  // or invalid when a member cannot be instantiated. O(count * pattern).
  Id instantiated_target(const Target &target) {
    const Spec &spec = checker.specs[target.sid];
    if (spec.maximum || spec.count <= 0 || spec.count > Z(budget.instantiated_members) ||
        !spec.facts.finite || target.parameters.size() + 1 != spec.slots) return invalid;
    std::vector<Id> cap = capture(target, invalid);
    std::vector<Id> terms;
    for (Z k = 0; k < spec.count; ++k) {
      tick(); cap[spec.axis] = a.constant(Q(k));
      const Id member = checker.instantiate_body(spec.pattern, cap);
      if (member == invalid || !a.nodes[member].facts.finite) return invalid;
      terms.push_back(member);
    }
    ++a.stats["linear_reduction_instantiated_targets"];
    return a.add(std::move(terms));
  }
  // The reference with each instantiated target call replaced by its sum of
  // members: an exact rewrite, since the call denotes that sum. Memoized over
  // the reference DAG; operators are rebuilt by the shared reapply.
  Id expand_targets(Id root, const std::map<size_t, Id> &members) {
    std::unordered_map<Id, Id> memo;
    std::vector<std::pair<Id, bool>> todo{{root, false}};
    while (!todo.empty()) {
      tick(); auto [id, ready] = todo.back(); todo.pop_back();
      if (memo.contains(id)) continue;
      const auto &n = a.nodes[id];
      if (n.op == "call") {
        Id value = id;
        for (const auto &[t, sum] : members) if (target_call(id, targets[t])) value = sum;
        memo.emplace(id, value); continue;
      }
      if (!ready) {
        todo.emplace_back(id, true);
        for (Id child : n.args) if (!memo.contains(child)) todo.emplace_back(child, false);
        continue;
      }
      std::vector<Id> args; args.reserve(n.args.size());
      bool changed = false;
      for (Id child : n.args) { args.push_back(memo.at(child)); changed |= args.back() != child; }
      memo.emplace(id, changed ? checker.reapply(id, args) : id);
    }
    return memo.at(root);
  }
  bool complete_with_zero_gaps(std::vector<Box> boxes, const Target &target) {
    const Z &count = checker.specs[target.sid].count;
    std::vector<std::pair<Z,Z>> intervals;
    for (const auto &box : boxes) {
      tick(); if (box.mult != 1) return false;
      auto axes = box.axes;
      std::sort(axes.begin(), axes.end());
      Z size = 1;
      for (const auto &[stride, extent] : axes) {
        tick(); if (extent <= 0 || stride <= 0) return false;
        if (extent == 1) continue;
        if (stride != size) return false; // Not a dense interval; do not enumerate.
        size *= extent;
      }
      Z end = box.start + size;
      if (box.start < 0 || end > count) return false;
      intervals.emplace_back(box.start, std::move(end));
    }
    std::sort(intervals.begin(), intervals.end());
    Z cursor = 0;
    for (const auto &[begin,end] : intervals) {
      tick();
      if (begin < cursor) return false; // A repeated contribution is not a gap.
      if (begin > cursor) {
        if (!zero_on_interval(target,cursor,begin)) return false;
        boxes.push_back({cursor,1,{{1,begin-cursor}}});
      }
      cursor = end;
    }
    if (cursor < count) {
      if (!zero_on_interval(target,cursor,count)) return false;
      boxes.push_back({cursor,1,{{1,count-cursor}}});
    }
    // Keep the independent multiplicity/coverage certificate after adding only
    // proven-zero missing contributions. Present terms were matched in full.
    return cover_boxes(boxes,count,a);
  }
  bool complete(const Key &key, const Group &group) {
    const Z &count = checker.specs[targets[key.target].sid].count;
    if (group.multiplicity > count) return false;
    if (group.multiplicity < count) {
      const auto &root = checker.patterns.nodes[checker.specs[targets[key.target].sid].pattern];
      bool masked = root.op == "select";
      if (root.op == "mul" && root.args.size() <= 16)
        for (Id child : root.args) masked |= checker.patterns.nodes[child].op == "select";
      // Unmasked incomplete groups retain their cheap cardinality rejection.
      if (!masked) return false;
    }
    auto cache_key = std::pair(group.piece, key.target);
    if (auto it = coverage_cache.find(cache_key); it != coverage_cache.end()) return it->second;
    struct Work { Id piece; bool leave; };
    std::vector<Work> todo{{group.piece, false}};
    std::vector<std::pair<Id, Z>> domains; std::vector<Box> boxes;
    bool valid = true;
    while (!todo.empty()) {
      tick(); auto w = todo.back(); todo.pop_back();
      const auto &p = checker.pieces[w.piece];
      if (w.leave) domains.pop_back();
      else if (p.kind == PieceNode::Concat) {
        todo.push_back({p.right, false}); todo.push_back({p.left, false});
      } else if (p.kind == PieceNode::Bind) {
        domains.emplace_back(p.var, p.count);
        todo.push_back({w.piece, true}); todo.push_back({p.left, false});
      } else {
        auto b = checker.box(p.index, domains);
        if (!b) { valid = false; break; }
        boxes.push_back(std::move(*b));
      }
    }
    valid = valid && (group.multiplicity == count ? cover_boxes(boxes, count, a)
                  : complete_with_zero_gaps(std::move(boxes),targets[key.target]));
    coverage_cache.emplace(cache_key, valid); return valid;
  }

  // Bounded ring normalization only after reduction closure. Calls, reads and
  // remaining Fold nodes are indivisible atoms; their bodies are not expanded.
  using Monomial = std::vector<Id>;
  using Polynomial = std::map<Monomial, Q>;
  std::unordered_map<Id, Polynomial> scalar_cache, witness_cache;
  void add_polynomial(Polynomial &to, const Polynomial &from, Q scale = Q(1)) {
    if (scale == 0) return;
    for (const auto &[key, c] : from) {
      scalar_tick(); to[key] += scale * c;
      if (to[key] == 0) to.erase(key);
      if (to.size() > budget.polynomial_terms) throw Miss{"polynomial_terms"};
    }
  }
  Polynomial multiply(const Polynomial &x, const Polynomial &y) {
    Polynomial result;
    for (const auto &[kx, cx] : x) for (const auto &[ky, cy] : y) {
      scalar_tick();
      if (kx.size() + ky.size() > budget.polynomial_degree) throw Miss{"polynomial_degree"};
      Monomial key; key.reserve(kx.size() + ky.size());
      std::merge(kx.begin(), kx.end(), ky.begin(), ky.end(), std::back_inserter(key));
      result[key] += cx * cy;
      if (result[key] == 0) result.erase(key);
      if (result.size() > budget.polynomial_terms) throw Miss{"polynomial_terms"};
    }
    return result;
  }
  const Polynomial &polynomial(Id root) {
    std::vector<std::pair<Id, bool>> todo{{root, false}};
    while (!todo.empty()) {
      scalar_tick(); auto [id, ready] = todo.back(); todo.pop_back();
      if (scalar_cache.contains(id)) continue;
      const auto &n = a.nodes[id];
      if (!n.facts.finite) throw Miss{"nonfinite_scalar"};
      if (!ready && (n.op == "add" || n.op == "add_raw" || n.op == "mul")) {
        todo.emplace_back(id, true);
        for (Id child : n.args) if (!scalar_cache.contains(child)) todo.emplace_back(child, false);
        continue;
      }
      Polynomial result;
      if (auto constant = a.value(id)) { if (*constant != 0) result[{}] = *constant; }
      else if (n.op == "add" || n.op == "add_raw") {
        for (size_t j = 0; j < n.args.size(); ++j) add_polynomial(result, scalar_cache.at(n.args[j]), n.data[j]);
      } else if (n.op == "mul") {
        result[{}] = n.data[0];
        for (Id child : n.args) result = multiply(result, scalar_cache.at(child));
      } else result[{id}] = Q(1);
      scalar_cache.emplace(id, std::move(result));
    }
    return scalar_cache.at(root);
  }
  // Bind witnesses only arise in partial/unrecognized residues. Keep the
  // original Sum semantics as an opaque scalar rather than distributing it.
  std::unordered_map<Id, Id> materialized;
  Id materialize(Id root) {
    if (root == invalid) return a.zero;
    std::vector<std::pair<Id, bool>> todo{{root, false}};
    while (!todo.empty()) {
      tick(); auto [id, ready] = todo.back(); todo.pop_back();
      if (materialized.contains(id)) continue;
      const Witness &w = witnesses[id];
      if (!ready) {
        todo.emplace_back(id, true);
        if (w.left != invalid && !materialized.contains(w.left)) todo.emplace_back(w.left, false);
        if (w.right != invalid && !materialized.contains(w.right)) todo.emplace_back(w.right, false);
        continue;
      }
      Id value;
      if (w.kind == Witness::Original) value = w.value;
      else if (w.kind == Witness::Add) {
        // add_raw preserves the witness forest; ordinary a.add would repeatedly
        // flatten growing partial prefixes before we know whether they close.
        Id l = materialized.at(w.left), r = materialized.at(w.right);
        const auto &lf = a.nodes[l].facts, &rf = a.nodes[r].facts;
        value = a.intern("add_raw", {l, r}, {Q(1), Q(1)}, "",
                         {lf.finite && rf.finite, false, false});
      } else if (w.kind == Witness::Scale) {
        Id inner = materialized.at(w.left);
        value = a.intern("mul", {w.value, inner}, {Q(1)}, "",
                         {a.nodes[w.value].facts.finite && a.nodes[inner].facts.finite, false, false});
      }
      else {
        Id body = materialized.at(w.left);
        value = a.intern("fold", {a.zero, body}, {Q(w.variable), Q(w.count)}, "Sum",
                         {a.nodes[body].facts.finite, false, false});
      }
      materialized.emplace(id, value);
    }
    return materialized.at(root);
  }
  const Polynomial &witness_polynomial(Id root) {
    if (root == invalid) return polynomial(a.zero);
    if (auto it = witness_cache.find(root); it != witness_cache.end()) return it->second;
    // Aggregate path weights over the shared forest, then add each leaf once.
    // This avoids copying a growing polynomial at every Add-prefix node, while
    // preserving multiplicity when a witness is reachable through two edges.
    std::vector<std::pair<Id, bool>> todo{{root, false}};
    std::unordered_set<Id> seen;
    std::vector<Id> postorder;
    while (!todo.empty()) {
      scalar_tick(); auto [id, ready] = todo.back(); todo.pop_back();
      if (seen.contains(id)) continue;
      const Witness &w = witnesses[id];
      if (!ready && (w.kind == Witness::Add || w.kind == Witness::Scale)) {
        todo.emplace_back(id, true);
        if (!seen.contains(w.left)) todo.emplace_back(w.left, false);
        if (w.right != invalid && !seen.contains(w.right)) todo.emplace_back(w.right, false);
        continue;
      }
      seen.insert(id); postorder.push_back(id);
    }
    std::unordered_map<Id, Polynomial> weights;
    weights[root][{}] = Q(1);
    Polynomial result;
    for (auto it = postorder.rbegin(); it != postorder.rend(); ++it) {
      scalar_tick(); Id id = *it;
      Polynomial weight = std::move(weights.at(id)); weights.erase(id);
      const Witness &w = witnesses[id];
      if (w.kind == Witness::Add) {
        add_polynomial(weights[w.left], weight); add_polynomial(weights[w.right], weight);
      } else if (w.kind == Witness::Scale) {
        add_polynomial(weights[w.left], multiply(weight, polynomial(w.value)));
      } else {
        const Polynomial &leaf = polynomial(w.kind == Witness::Original ? w.value : materialize(id));
        add_polynomial(result, multiply(weight, leaf));
      }
    }
    return witness_cache.emplace(root, std::move(result)).first->second;
  }

public:
  std::string last_failure;
  explicit LinearReductionPattern(Checker &owner, LinearReductionBudget limits = {})
      : checker(owner), a(owner.a), budget(limits) {}
private:
  std::optional<LinearReductionProof> prove_attempt(Id reference, Id optimized,
                                                   bool reuse_existing) {
    visits = piece_count = scalar_work = choices = 0; distributed_terms = 0; factored_sums = 0; instantiated_receipt = J::array(); attempt_nested_closures = attempt_nested_gaps = 0;
    closed_groups = partial_groups = 0;
    targets.clear(); witnesses.clear(); original_witnesses.clear(); coverage_cache.clear();
    scalar_cache.clear(); witness_cache.clear(); materialized.clear(); completed_targets.clear(); last_failure.clear();
    shared_targets_skipped = false;
    try {
      if (!collect_targets(reference, optimized, reuse_existing)) return {};
      ++a.stats["linear_reduction_pattern_attempts"];
      const Polynomial expected = checker.exponential_mode ? Polynomial{} : polynomial(reference);
      Form form = decompose(optimized);
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
        if(!checker.scalar.equal(expanded,a.add(ep_terms)))throw Miss{"ep_output_scalar"};
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
  std::optional<LinearReductionProof> attempt(Id reference, Id optimized, bool reuse_existing) {
    factor_weights = false;
    auto result = prove_attempt(reference, optimized, reuse_existing);
    if (result || last_failure != "distribution_budget") return result;
    ++a.stats["linear_reduction_factored_weight_retries"];
    FlagScope factored(factor_weights, true);
    return prove_attempt(reference, optimized, reuse_existing);
  }
  std::optional<LinearReductionProof> direct(Id reference, Id optimized) {
    auto result = attempt(reference, optimized, true);
    if (result || !shared_targets_skipped) return result;
    // A shared call can coexist with still-expanded contributions to that
    // same sum (for example a second LayerNorm coefficient). Preserve the old
    // full-target proof as one bounded fallback; never discard the residual.
    ++a.stats["linear_reduction_shared_target_fallbacks"];
    return attempt(reference, optimized, false);
  }
  // Each arm pair is one bounded sub-proof with this pattern's own budgets:
  // one direct() proof, i.e. at most two attempt() calls of at most two
  // budgeted prove_attempt() passes each. This bounds their number per root
  // pair (a select chain over a state-pool slot table has one pair per slot),
  // so select_arms does at most max_select_arm_pairs * 4 budgeted passes.
  static constexpr size_t max_select_arm_pairs = 1024;
  // select(p, A, B) = select(p, A', B') for every p when A = A' and B = B' as
  // total functions. Selects are descended only while both sides test the
  // same predicate node. Every remaining arm pair is proved by the scalar
  // engine or by a direct reduction proof (never by a further descent), so
  // the work is the pairs' sub-proofs, each with this pattern's budgets.
  std::optional<LinearReductionProof> select_arms(Id reference, Id optimized) {
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
  bool nested_targets(Id reference) {
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
  // Fused reductions. When every monomial of the reference holds at most one
  // target call, R = r0 + Sum_t alpha_t call_t, and the targets in that form
  // share one count N with the alpha_t free of their reduction coordinate,
  // Sum_t alpha_t call_t = Sum_{i<N} F(i) with F(i) = Sum_t alpha_t body_t(i).
  // One term of an explicit optimized sum then matches F(i) as a whole, with
  // one scalar proof, where term-wise matching would first have to distribute
  // that term's factors into every constituent body.
  struct FusedForm { std::vector<size_t> members; std::vector<Id> alphas; Z count; };
  std::optional<FusedForm> fused_form(Id reference) {
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
  Id fused_body(const FusedForm &form, const Z &index) {
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
  // A sum that several terms of the optimized sums share (a common factor
  // such as an inner prediction) is closed first when its own decomposition
  // completes a target group: value = residual + Sum coefficient * call is
  // the same exact identity the final certificate uses. Only add nodes that
  // occur as a product factor below the root, referenced at least twice, are
  // candidates; the scan visits stored add/mul shells once.
  size_t inner_closures = 0;
  Id closed_rewrite = invalid;
  // Closure targets also include eligible reference calls below nonlinear
  // nodes (e.g. a norm inside rsqrt): an inner closure needs the call itself,
  // not a linear position of it in the reference. Bounded scan of the stored
  // reference DAG; at most max_closure_targets targets in all.
  static constexpr size_t max_closure_targets = 8;
  void add_closure_targets(Id reference) {
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
  static bool unary_numeric(const std::string &op) {
    return op == "recip" || op == "reciprocalsqrt" || op == "sqrt" || op == "exp" || op == "exp2" ||
           op == "log" || op == "log2";
  }
  // Integer value of a coordinate built from constants, one variable u,
  // add/mul and floor idiv/imod, at u = value. Anything else has no value.
  std::optional<Z> coordinate_at(Id id, Id u, const Z &value, unsigned depth = 0) {
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
  // A group of plain leaves (an explicit sum, no binders) whose coordinates
  // depend on one free integer variable u with at most max_case_values
  // values is complete when, for EVERY value of u, the leaves' coordinates
  // are exactly [0, count) once each. Cost O(values * leaves).
  static constexpr size_t max_case_values = 64;
  bool complete_by_cases(const Key &key, const Group &group) {
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
  Id close_shared_sums(Id root) {
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
  std::optional<LinearReductionProof> fused_attempt(Id reference, Id optimized) {
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
public:
  // Work per prove(), in budgeted passes (each pass resets and is bounded by
  // the node, piece and polynomial budgets): direct() at most 4;
  // select_arms() at most max_select_arm_pairs * 4; the fused attempt 1; a
  // closed-rewrite direct() at most 4; a nested-mode direct() at most 4.
  // Nested closures run inside their pass's remaining budget and are charged
  // to it.
  std::optional<LinearReductionProof> prove(Id reference, Id optimized) {
    if (auto result = direct(reference, optimized)) return result;
    if (auto result = select_arms(reference, optimized)) return result;
    // The fused route runs only where every proof above declined; its own
    // declines are counted, and the reported reason stays the earlier one.
    const std::string declined = last_failure;
    if (auto result = fused_attempt(reference, optimized)) return result;
    // Without a fused form, the inner closures alone may complete the
    // ordinary proof: the rewritten expression is exactly the optimized one.
    if (const Id rewritten = closed_rewrite; rewritten != invalid && rewritten != optimized) {
      const size_t closures = inner_closures;
      if (auto result = direct(reference, rewritten)) {
        result->receipt["inner_closures"] = closures;
        ++a.stats["linear_reduction_closed_rewrite_proofs"];
        return result;
      }
    }
    if (nested_targets(reference)) {
      FlagScope nested(nested_mode, true);
      if (auto result = direct(reference, optimized)) { ++a.stats["linear_reduction_nested_proofs"]; return result; }
    }
    last_failure = declined;
    return {};
  }
};
} // namespace rg
