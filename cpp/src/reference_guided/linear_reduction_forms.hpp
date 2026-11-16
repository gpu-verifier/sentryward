#pragma once
#include "linear_reduction_pattern.hpp"

namespace rg {

// A missing interval can count as a contribution only when the complete
// reference summand has a finite, explicitly zero-valued mask there. Inspect
// only a shallow product/Select shell; the predicate operands have a separate
// small pattern-DAG budget. Unknown bounds are a miss, never a zero proof.
inline auto LinearReductionPattern::zero_on_interval(const Target &target, const Z &lo, const Z &hi) -> bool {
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
      // Integer quotients and remainders by a constant (a block index idiv(k, 64), say) are
      // instantiated by the Arena's own constructors, which bound them exactly.
      if (n.op != "slot" && n.op != "const" && n.op != "var" &&
          n.op != "add" && n.op != "mul" && n.op != "affine_slot" &&
          n.op != "affine_param" && n.op != "idiv" && n.op != "imod") return false;
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
    // The truth of a small comparison on the gap, if it is decided there.
    const auto decided = [&](Id comparison) -> Id {
      const auto &c = checker.patterns.nodes[comparison];
      if (c.op != "compare" || c.args.size() != 2 ||
          !small_predicate_operand(c.args[0]) || !small_predicate_operand(c.args[1])) return invalid;
      Id lhs = checker.instantiate_body(c.args[0], cap);
      Id rhs = checker.instantiate_body(c.args[1], cap);
      if (lhs == invalid || rhs == invalid) return invalid;
      return a.compare(c.tag, lhs, rhs);
    };
    // A product condition is 0 on the gap when one of its comparison factors is false there and
    // every factor, instantiated on the gap, is finite: the Select then takes its false arm.
    // (Pattern nodes carry no facts; the instantiated factors do.)
    if (false_zero && predicate.op == "mul" && predicate.args.size() <= 16) {
      const auto factors = predicate.args;
      bool finite = true;
      for (Id factor : factors) {
        const Id value = checker.instantiate_body(factor, cap);
        if (value == invalid || !a.nodes[value].facts.finite) { finite = false; break; }
      }
      if (!finite) continue;
      for (Id factor : factors)
        if (decided(factor) == a.zero) {
          ++a.stats["linear_reduction_zero_gap_product_proofs"];
          return true;
        }
      continue;
    }
    const Id truth = decided(mask.args[0]);
    if (truth == invalid) continue;
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
inline auto LinearReductionPattern::instantiated_target(const Target &target) -> Id {
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
inline auto LinearReductionPattern::expand_targets(Id root, const std::map<size_t, Id> &members) -> Id {
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

inline auto LinearReductionPattern::complete_with_zero_gaps(std::vector<Box> boxes, const Target &target) -> bool {
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

inline auto LinearReductionPattern::complete(const Key &key, const Group &group) -> bool {
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
                : complete_with_zero_gaps(boxes, targets[key.target]) ||
                  complete_with_zero_edge(std::move(boxes), targets[key.target], group.multiplicity));
  coverage_cache.emplace(cache_key, valid); return valid;
}

// A masked target whose matched contributions (any box shapes, interleaved strides included) leave
// out exactly one interval at the end of the reduction domain, or at its start: the interval
// [covered, count) (or [0, count - covered)) is proved to be a zero of the complete reference
// summand (zero_on_interval: the mask is false there), and the boxes plus that interval cover the
// domain exactly once (cover_boxes: the exact generating-polynomial identity). Then the sum of the
// contributions is the complete reduction: the left-out terms are 0. Exact. Used only where the
// dense-interval gap rule above declines. O(boxes) plus at most two zero proofs.
inline auto LinearReductionPattern::complete_with_zero_edge(std::vector<Box> boxes, const Target &target, const Z &covered) -> bool {
  const Z &count = checker.specs[target.sid].count;
  if (covered <= 0 || covered >= count) return false;
  for (const bool suffix : {true, false}) {
    const Z lo = suffix ? covered : Z(0), hi = suffix ? count : count - covered;
    if (!zero_on_interval(target, lo, hi)) continue;
    boxes.push_back({lo, 1, {{1, hi - lo}}});
    if (cover_boxes(boxes, count, a)) { ++a.stats["linear_reduction_zero_edge_coverage_proofs"]; return true; }
    boxes.pop_back();
  }
  return false;
}

inline auto LinearReductionPattern::add_polynomial(Polynomial &to, const Polynomial &from, Q scale) -> void {
  if (scale == 0) return;
  for (const auto &[key, c] : from) {
    scalar_tick(); to[key] += scale * c;
    if (to[key] == 0) to.erase(key);
    if (to.size() > budget.polynomial_terms) throw Miss{"polynomial_terms"};
  }
}

inline auto LinearReductionPattern::multiply(const Polynomial &x, const Polynomial &y) -> Polynomial {
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

inline auto LinearReductionPattern::materialize(Id root) -> Id {
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

} // namespace rg
