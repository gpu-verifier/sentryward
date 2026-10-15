#pragma once
#include "checker.hpp"

namespace rg {
struct LinearReductionBudget {
  uint64_t nodes = 100000, pieces = 30000;
  size_t groups = 32, targets = 4;
  size_t factor_choices = 128, polynomial_terms = 512, polynomial_degree = 24;
  size_t distributed_terms = 128;
  uint64_t polynomial_work = 100000;
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
        if (!checker.active.contains(sid) || spec.maximum || !spec.facts.finite ||
            spec.count <= 0 || n.args.size() + 1 != spec.slots) return false;
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
    return best;
  }

  Form decompose(Id root) {
    // The scale is part of the memo identity, not a rewrite of the Fold body.
    // Work is bounded by stored (node,weight) states and their operand edges,
    // not by logical reduction length. Distinct weights may create distinct
    // states; tick()/distributed_terms limit that real cost explicitly.
    using State = std::pair<Id, Id>;
    enum class Kind { Add, Bind, WeightedChild, Product };
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
          if ((n.op == "add" || n.op == "add_raw") && n.args.size() == n.data.size()) {
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
    std::vector<Id> cap(spec.slots);
    size_t parameter = 0;
    for (size_t slot = 0; slot < spec.slots; ++slot)
      if (slot != spec.axis) cap[slot] = target.parameters[parameter++];
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
    visits = piece_count = scalar_work = choices = 0; distributed_terms = 0;
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
        if (complete(key, group)) {
          complete_groups.insert(key);
          completed_targets.insert(key.target);
        }
      }
      if (completed_targets.size() != targets.size()) throw Miss{"incomplete_targets"};
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
        if(!checker.scalar.equal(reference,a.add(ep_terms)))throw Miss{"ep_output_scalar"};
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
      return LinearReductionProof{reference, {
        {"method", "linear_reduction_combination_certificate"}, {"closures", closures},
        {"coverage", "exact_multiplicity_one_per_closed_group"},
        {"preserved_partial_groups", partial_groups}, {"visited_nodes", visits},
        {"contribution_pieces", piece_count}, {"witness_nodes", witnesses.size()},
        {"weighted_add_edges", distributed_terms},
        {"factor_choices", choices}, {"polynomial_work", scalar_work},
        {"output_polynomial_terms", actual.size()}}};
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
public:
  std::optional<LinearReductionProof> prove(Id reference, Id optimized) {
    auto result = prove_attempt(reference, optimized, true);
    if (result || !shared_targets_skipped) return result;
    // A shared call can coexist with still-expanded contributions to that
    // same sum (for example a second LayerNorm coefficient). Preserve the old
    // full-target proof as one bounded fallback; never discard the residual.
    ++a.stats["linear_reduction_shared_target_fallbacks"];
    return prove_attempt(reference, optimized, false);
  }
};
} // namespace rg
