#pragma once
#include "linear_reduction_pattern.hpp"

namespace rg {

inline auto LinearReductionPattern::family_spec(Id sid, size_t arguments) -> bool {
  if (sid >= checker.specs.size()) return false;
  const Spec &spec = checker.specs[sid];
  return !spec.maximum && spec.facts.finite && spec.count > 0 && arguments + 1 == spec.slots &&
         checker.literal_binder_free(sid);
}

// Every target's spec, with its own coordinate slot at a fresh formal
// variable over its domain, then every family's spec with the coordinate
// slot left open (a nested Call that names it is no family of A). The
// variable is named after the spec and its domain only, so one family
// has one formal node. O(pattern nodes) per scanned spec.
inline auto LinearReductionPattern::collect_families() -> void {
  families.clear(); row_coordinates.clear();
  struct Work { Target owner; Id axis; bool slot_axis; size_t depth; };
  std::vector<Work> todo;
  for (const auto &target : targets) {
    const Spec &spec = checker.specs[target.sid];
    if (spec.maximum || spec.count <= 0) continue;
    const Id axis = a.var("linear-parametric-row:" + std::to_string(target.sid) + ":" + spec.count.str(),
                          Q(0), Q(spec.count - 1));
    todo.push_back({target, axis, true, 0});
  }
  while (!todo.empty() && families.size() < max_families) {
    const Work work = todo.back(); todo.pop_back();
    const Spec &spec = checker.specs[work.owner.sid];
    const std::vector<Id> cap = capture(work.owner, work.slot_axis ? work.axis : invalid);
    std::vector<Id> walk{spec.pattern};
    std::unordered_set<Id> seen{spec.pattern};
    while (!walk.empty() && families.size() < max_families) {
      tick(); const Id id = walk.back(); walk.pop_back();
      const auto &n = checker.patterns.nodes[id];
      if (n.op == "call") {
        if (n.data.size() != 1 || n.data[0].denominator() != 1 || n.data[0] < 0 ||
            n.data[0] >= Q(checker.specs.size())) continue;
        const Id sid = n.data[0].numerator().convert_to<Id>();
        if (!family_spec(sid, n.args.size())) continue;
        std::vector<Id> args;
        for (Id x : n.args) {
          const Id value = checker.instantiate_body(x, cap);
          if (value == invalid) { args.clear(); break; }
          args.push_back(value);
        }
        if (args.size() != n.args.size()) continue;
        auto exact = std::find(args.begin(), args.end(), work.axis);
        if (exact == args.end()) continue;
        if (std::any_of(families.begin(), families.end(), [&](const Family &f) {
              return f.target.sid == sid && f.target.parameters == args && f.axis == work.axis; })) continue;
        Family family;
        family.target = {sid, args}; family.axis = work.axis;
        family.position = size_t(exact - args.begin()); family.depth = work.depth + 1;
        families.push_back(std::move(family));
        ++a.stats["linear_reduction_parametric_families"];
        if (work.depth + 1 < max_family_depth) todo.push_back({{sid, args}, work.axis, false, work.depth + 1});
        continue;
      }
      for (Id child : n.args) if (seen.insert(child).second) walk.push_back(child);
    }
  }
  // Innermost families first: a row's own inner rows close before it.
  std::stable_sort(families.begin(), families.end(),
                   [](const Family &x, const Family &y) { return x.depth > y.depth; });
}

// An integer coordinate inside A's domain for every value of its own free
// variables (the condition the instance check states again).
inline auto LinearReductionPattern::row_value(Id value, Id axis) -> bool {
  const auto &range = a.nodes[axis].bounds;
  const auto &n = a.nodes[value];
  return range && n.bounds && range->first <= n.bounds->first && n.bounds->second <= range->second &&
         !a.depends(value, axis) && CoordinateRelations::integer(a, value);
}

// The row coordinate of a candidate: the value that one term's anchors
// bind to the family's A-parameter, with every other A-dependent parameter
// its instance at that value. A proposal only (body_candidates proposes,
// never proves); the full instance check decides. Memoized per (family,
// node) across the passes of one attempt.
inline auto LinearReductionPattern::row_coordinate(size_t index, Id value) -> std::optional<Id> {
  const auto key = std::pair(index, value);
  if (auto it = row_coordinates.find(key); it != row_coordinates.end()) return it->second;
  const Family &f = families[index];
  const Spec &spec = checker.specs[f.target.sid];
  std::vector<Id> initial = capture(f.target, invalid);
  std::vector<size_t> dependent;
  size_t slot_of_position = spec.slots;
  for (size_t slot = 0, parameter = 0; slot < spec.slots; ++slot) {
    if (slot == spec.axis) continue;
    if (parameter == f.position) slot_of_position = slot;
    if (a.depends(f.target.parameters[parameter], f.axis)) { initial[slot] = invalid; dependent.push_back(slot); }
    ++parameter;
  }
  std::optional<Id> found;
  const auto &n = a.nodes[value];
  for (size_t t = 0; t < n.args.size() && t < max_discovery_terms && !found; ++t) {
    tick();
    for (const auto &cap : checker.body_candidates(spec, n.args[t], initial)) {
      tick();
      if (cap.size() != spec.slots || slot_of_position >= spec.slots) continue;
      const Id j = cap[slot_of_position];
      if (j == invalid || !row_value(j, f.axis)) continue;
      bool consistent = true;
      for (size_t slot : dependent) {
        const size_t parameter = slot < spec.axis ? slot : slot - 1;
        consistent = cap[slot] != invalid &&
                     checker.restrict_axis(f.target.parameters[parameter], f.axis, j) == cap[slot];
        if (!consistent) break;
      }
      if (consistent) { found = j; break; }
    }
  }
  row_coordinates.emplace(key, found);
  return found;
}

// The theorem: L(A) decomposed with the single target(A); one complete
// group (coverage of the target's whole domain) gives L(A) = C(A). The
// decomposition runs on this attempt's remaining budgets and is charged
// back. Samples only propose L; a failed build or theorem drops the older
// sample, at most max_family_attempts times per family.
inline auto LinearReductionPattern::prove_family(Family &f, std::vector<std::pair<Id, Id>> &samples) -> void {
  if (samples.size() < 2) return;
  ++f.attempts;
  auto drop = [&](const char *why) {
    ++a.stats[std::string("linear_reduction_parametric_decline_") + why];
    samples.erase(samples.begin());
    if (f.attempts >= max_family_attempts) f.failed = true;
  };
  auto rows = std::make_unique<WholeExpressionTemplates>(
      checker, std::vector<Id>{f.axis}, ++a.stats["linear_reduction_parametric_templates"]);
  rows->allow_reference_calls();
  const auto [first, first_row] = samples[0];
  const auto [second, second_row] = samples[1];
  const bool built = rows->build({{first, first, {first_row}}, {second, second, {second_row}}});
  structure_tick(rows->last_visits);
  if (!built) return drop("template");
  const Id row = rows->left_template();
  LinearReductionPattern inner(checker, remaining_budget());
  inner.closure_query = closure_query;  // charged here, so part of an enclosing closure query
  inner.targets = {f.target};
  Id closed = invalid;
  try {
    Form form = inner.decompose(row);
    std::vector<Id> parts{inner.materialize(form.residual)};
    bool complete = false;
    for (const auto &[key, group] : form.groups) {
      if (inner.complete(key, group) || inner.complete_by_cases(key, group)) {
        parts.push_back(a.mul({key.coefficient, checker.call(f.target.sid, f.target.parameters)}));
        complete = true;
      } else parts.push_back(inner.materialize(group.witness));
    }
    if (complete) closed = a.add(std::move(parts));
  } catch (const Miss &) {
  } catch (const Unsupported &) {}
  charge(inner);
  if (closed == invalid || !a.nodes[closed].facts.finite) return drop("theorem");
  f.rows = std::move(rows); f.closed = closed;
  ++a.stats["linear_reduction_parametric_theorems"];
}

inline auto LinearReductionPattern::close_row(Id value, bool &proved_new) -> std::optional<Id> {
  for (size_t index = 0; index < families.size(); ++index) {
    Family &f = families[index];
    if (f.failed) continue;
    if (f.closed != invalid) {
      // An instance has the template row's operator and arity; anything
      // else is no instance worth instantiating (a filter, not a proof).
      const auto &shape = a.nodes[f.rows->left_template()], &candidate = a.nodes[value];
      if (shape.op != candidate.op || shape.args.size() != candidate.args.size()) continue;
    }
    std::optional<Id> row;
    try { row = row_coordinate(index, value); }
    catch (const Unsupported &) { row_coordinates[{index, value}] = std::nullopt; }
    if (!row) continue;
    if (f.closed == invalid) {
      auto &samples = f.samples[a.nodes[value].args.size()];
      if (std::none_of(samples.begin(), samples.end(),
                       [&](const auto &s) { return s.first == value || s.second == *row; }))
        samples.emplace_back(value, *row);
      prove_family(f, samples);
      if (f.closed == invalid) continue;
      proved_new = true;
    }
    const bool instance = f.rows->matches_reference_identity(value, {*row});
    structure_tick(f.rows->last_visits);
    if (!instance) { ++f.declines; ++a.stats["linear_reduction_parametric_instance_declines"]; continue; }
    const Id replacement = checker.restrict_axis(f.closed, f.axis, *row);
    if (a.depends(replacement, f.axis) || !a.nodes[replacement].facts.finite) continue;
    ++f.instances; ++a.stats["linear_reduction_parametric_instances"];
    return replacement;
  }
  return {};
}

// The ordinary closure of a sum below a unary numeric node (a norm over
// the closed rows), against the attempt's targets: the same identity as
// close_shared_sums. A miss keeps the explicit sum.
inline auto LinearReductionPattern::close_ordinary(Id value) -> std::optional<Id> {
  ++a.stats["linear_reduction_parametric_unary_attempts"];
  try {
    Form form = decompose(value);
    std::vector<Id> parts{materialize(form.residual)};
    bool any = false;
    for (const auto &[key, group] : form.groups) {
      if (complete(key, group) || complete_by_cases(key, group)) {
        parts.push_back(a.mul({key.coefficient, checker.call(targets[key.target].sid, targets[key.target].parameters)}));
        any = true;
      } else parts.push_back(materialize(group.witness));
    }
    if (any) { ++inner_closures; return a.add(std::move(parts)); }
    ++a.stats[form.groups.empty() ? "linear_reduction_parametric_unary_no_group"
                                  : "linear_reduction_parametric_unary_incomplete_group"];
  } catch (const Miss &miss) {
    if (std::string_view(miss.reason) == "structure_budget") throw;
    ++a.stats[std::string("linear_reduction_parametric_unary_decline_") + miss.reason];
  } catch (const Unsupported &) {
    ++a.stats["linear_reduction_parametric_unary_decline_unsupported"];
  }
  return {};
}

inline auto LinearReductionPattern::parametric_scan(Id root, ParametricScan &scan) -> void {
  std::vector<Id> todo{root};
  std::unordered_set<Id> seen{root};
  while (!todo.empty()) {
    structure_tick(); const Id id = todo.back(); todo.pop_back();
    const auto &n = a.nodes[id];
    const bool unary = unary_numeric(n.op);
    if (n.op != "add" && n.op != "add_raw" && n.op != "mul" && !unary) continue;
    for (Id child : n.args) {
      const auto &c = a.nodes[child];
      if ((n.op == "mul" || unary) && (c.op == "add" || c.op == "add_raw")) {
        scan.uses[child] += unary ? 2 : 1;
        if (unary) scan.unary_arguments.insert(child);
      }
      if (seen.insert(child).second) todo.push_back(child);
    }
  }
}

// One bottom-up rewrite from the original optimized DAG. A family proved
// during the pass applies from that point on; its samples (already passed)
// are closed by the next pass. Ordinary closures below unary nodes run
// only in a pass that proved no new family, i.e. the final one.
inline auto LinearReductionPattern::parametric_pass(Id root, const ParametricScan &scan, bool &proved_new) -> Id {
  std::unordered_map<Id, Id> memo;
  std::vector<std::pair<Id, bool>> work{{root, false}};
  while (!work.empty()) {
    a.check(); auto [id, ready] = work.back(); work.pop_back();
    if (memo.contains(id)) continue;
    const auto &n = a.nodes[id];
    if (n.op != "add" && n.op != "add_raw" && n.op != "mul" && !unary_numeric(n.op)) {
      structure_tick(); memo.emplace(id, id); continue;
    }
    if (!ready) {
      work.emplace_back(id, true);
      for (Id child : n.args) if (!memo.contains(child)) work.emplace_back(child, false);
      continue;
    }
    structure_tick();
    std::vector<Id> args;
    bool changed = false;
    for (Id child : n.args) { args.push_back(memo.at(child)); changed |= args.back() != child; }
    Id value = changed ? checker.reapply(id, args) : id;
    const auto use = scan.uses.find(id);
    if (id != root && use != scan.uses.end() && use->second >= 2 && a.nodes[value].op == "add" &&
        a.nodes[value].facts.finite) {
      if (auto row = close_row(value, proved_new)) value = *row;
      else if (!proved_new && scan.unary_arguments.contains(id))
        if (auto closed = close_ordinary(value)) value = *closed;
    }
    memo.emplace(id, value);
  }
  return memo.at(root);
}

// The fresh coordinate of level `depth` is named by that level, so a nested
// comparison never reuses an enclosing level's coordinate (which its
// arguments may hold); arguments that hold it anyway are refused.
inline auto LinearReductionPattern::call_body(Id call, Id &axis, unsigned depth) -> std::optional<std::pair<Id, Id>> {
  const auto &n = a.nodes[call];
  if (n.op != "call" || n.data.size() != 1 || n.data[0].denominator() != 1 || n.data[0] < 0 ||
      n.data[0] >= Q(checker.specs.size())) return {};
  const Id sid = n.data[0].numerator().convert_to<Id>();
  const Spec &spec = checker.specs[sid];
  if (spec.maximum || spec.count <= 0 || !spec.facts.finite || n.args.size() + 1 != spec.slots ||
      !checker.literal_binder_free(sid)) return {};
  if (axis == invalid)
    axis = a.var("linear-call-equality:" + std::to_string(depth) + ":" + spec.count.str(), Q(0), Q(spec.count - 1));
  else if (a.nodes[axis].bounds->second != Q(spec.count - 1)) return {};
  if (a.depends(call, axis)) return {};
  const Id body = construct_body(spec.pattern, capture({sid, n.args}, axis));
  if (body == invalid || !a.nodes[body].facts.finite) return {};
  return std::pair(sid, body);
}

// A spec body at a capture, built with the exact constructors (a summand
// at a fresh coordinate reads input elements no stored node reads yet).
// Real-typed reads, sums, products, the unary numeric operators, compare,
// select and nested Calls are rebuilt; coordinate recipes and every other
// node go through instantiate_body. invalid when anything is missing.
inline auto LinearReductionPattern::construct_body(Id root, const std::vector<Id> &cap) -> Id {
  std::unordered_map<Id, Id> memo;
  std::vector<std::pair<Id, bool>> todo{{root, false}};
  auto constructive = [](const std::string &op) {
    return op == "read" || op == "add" || op == "mul" || op == "exp" || op == "exp2" || op == "recip" ||
           op == "reciprocalsqrt" || op == "sqrt" || op == "log" || op == "log2" || op == "compare" ||
           op == "select" || op == "call";
  };
  while (!todo.empty()) {
    structure_tick(); auto [id, ready] = todo.back(); todo.pop_back();
    if (memo.contains(id)) continue;
    const auto &n = checker.patterns.nodes[id];
    if (!constructive(n.op)) {
      const Id value = checker.instantiate_body(id, cap);
      if (value == invalid) return invalid;
      memo.emplace(id, value); continue;
    }
    if (!ready) {
      todo.emplace_back(id, true);
      for (Id child : n.args) if (!memo.contains(child)) todo.emplace_back(child, false);
      continue;
    }
    std::vector<Id> xs;
    for (Id child : n.args) xs.push_back(memo.at(child));
    Id out = invalid;
    if (n.op == "add" && xs.size() == n.data.size()) {
      for (size_t j = 0; j < xs.size(); ++j) xs[j] = a.mul({a.constant(n.data[j]), xs[j]});
      out = a.add(std::move(xs));
    } else if (n.op == "mul" && n.data.size() == 1) {
      xs.push_back(a.constant(n.data[0])); out = a.mul(std::move(xs));
    } else if (n.op == "read") {
      const auto split = n.tag.rfind('\n');
      if (split == std::string::npos || !n.tag.substr(split + 1).starts_with("real")) return invalid;
      out = a.read(n.tag.substr(0, split), std::move(xs), n.tag.substr(split + 1));
    } else if ((n.op == "exp" || n.op == "exp2") && xs.size() == 1) out = a.exponential(xs[0], n.op == "exp2");
    else if (n.op == "recip" && xs.size() == 1) out = a.recip(xs[0]);
    else if ((n.op == "log" || n.op == "log2") && xs.size() == 1) out = a.logarithm(xs[0], n.op == "log2");
    else if ((n.op == "reciprocalsqrt" || n.op == "sqrt") && xs.size() == 1) {
      const auto &x = a.nodes[xs[0]].facts;
      const bool finite = x.finite && (n.op == "sqrt" ? x.nn : x.pos);
      const bool positive = finite && (n.op == "reciprocalsqrt" || x.pos);
      out = a.intern(n.op, xs, n.data, n.tag, {finite, positive, finite});
    } else if (n.op == "compare" && xs.size() == 2 && n.data.empty() &&
               (n.tag == "eq" || n.tag == "ne" || n.tag == "lt" || n.tag == "le" || n.tag == "gt" || n.tag == "ge"))
      out = a.compare(n.tag, xs[0], xs[1]);
    else if (n.op == "select" && xs.size() == 3 && n.data.empty() && n.tag.empty())
      out = a.select(xs[0], xs[1], xs[2]);
    else if (n.op == "call" && n.data.size() == 1 && n.data[0].denominator() == 1 && n.data[0] >= 0 &&
             n.data[0] < Q(checker.specs.size()) &&
             xs.size() + 1 == checker.specs[n.data[0].numerator().convert_to<Id>()].slots)
      out = checker.call(n.data[0].numerator().convert_to<Id>(), xs);
    if (out == invalid) return invalid;
    memo.emplace(id, out);
  }
  return memo.at(root);
}

inline auto LinearReductionPattern::call_atoms(Id root) -> std::vector<Id> {
  std::vector<Id> calls, todo{root};
  std::unordered_set<Id> seen{root};
  while (!todo.empty()) {
    structure_tick(); const Id id = todo.back(); todo.pop_back();
    const auto &n = a.nodes[id];
    if (n.op == "call") { calls.push_back(id); continue; }
    for (Id child : n.args) if (seen.insert(child).second) todo.push_back(child);
  }
  std::sort(calls.begin(), calls.end());
  return calls;
}

inline auto LinearReductionPattern::replace_atoms(Id root, const std::unordered_map<Id, Id> &replacement) -> Id {
  if (replacement.empty()) return root;
  std::unordered_map<Id, Id> memo;
  std::vector<std::pair<Id, bool>> todo{{root, false}};
  while (!todo.empty()) {
    structure_tick(); auto [id, ready] = todo.back(); todo.pop_back();
    if (memo.contains(id)) continue;
    if (auto it = replacement.find(id); it != replacement.end()) { memo.emplace(id, it->second); continue; }
    const auto &n = a.nodes[id];
    if (!ready) {
      todo.emplace_back(id, true);
      for (Id child : n.args) if (!memo.contains(child)) todo.emplace_back(child, false);
      continue;
    }
    std::vector<Id> args;
    bool changed = false;
    for (Id child : n.args) { args.push_back(memo.at(child)); changed |= args.back() != child; }
    memo.emplace(id, changed ? checker.reapply(id, args) : id);
  }
  return memo.at(root);
}

// Each Call of `actual` that is no Call of `expected` but equals one is
// replaced by it.
inline auto LinearReductionPattern::equal_call_pairs(const std::vector<Id> &expected, const std::vector<Id> &actual,
                                            unsigned depth) -> std::unordered_map<Id, Id> {
  // Only Calls with one multiset of arguments are compared (a filter).
  auto arguments = [&](Id call) { auto args = a.nodes[call].args; std::sort(args.begin(), args.end()); return args; };
  std::unordered_map<Id, Id> replacement;
  for (Id y : actual) {
    if (std::binary_search(expected.begin(), expected.end(), y)) continue;
    const auto wanted = arguments(y);
    for (Id x : expected)
      if (!std::binary_search(actual.begin(), actual.end(), x) && arguments(x) == wanted &&
          equal_calls(x, y, depth)) {
        replacement.emplace(y, x); break;
      }
  }
  return replacement;
}

inline auto LinearReductionPattern::equal_calls(Id x, Id y, unsigned depth) -> bool {
  if (x == y) return true;
  if (depth >= max_call_equality_depth) return false;
  const auto key = std::minmax(x, y);
  if (auto it = call_equalities.find(key); it != call_equalities.end()) return it->second;
  call_equalities.emplace(key, false);  // no cyclic use of the pair being proved
  tick();
  Id axis = invalid;
  const auto left = call_body(x, axis, depth), right = call_body(y, axis, depth);
  bool equal = false;
  if (left && right) {
    const auto nested = equal_call_pairs(call_atoms(left->second), call_atoms(right->second), depth + 1);
    ++a.stats["ep_reduction_outer_obligations"];
    equal = checker.scalar.equal(left->second, replace_atoms(right->second, nested));
  }
  call_equalities[key] = equal;
  ++a.stats[equal ? "linear_reduction_parametric_equal_calls" : "linear_reduction_parametric_unequal_calls"];
  return equal;
}

// Route: families from the reference, closure passes until no new family
// is proved (at most families + 1), Calls of the result that equal a
// reference Call replaced by it, then the exact normal form or one
// ordinary direct() proof.
inline auto LinearReductionPattern::parametric_attempt(Id reference, Id optimized) -> std::optional<LinearReductionProof> {
  visits = piece_count = scalar_work = choices = 0; distributed_terms = 0; factored_sums = 0;
  instantiated_receipt = J::array(); attempt_nested_closures = attempt_nested_gaps = 0;
  closed_groups = partial_groups = 0; inner_closures = 0; factor_weights = false; closed_rewrite = invalid;
  targets.clear(); witnesses.clear(); original_witnesses.clear(); coverage_cache.clear();
  scalar_cache.clear(); witness_cache.clear(); materialized.clear(); completed_targets.clear();
  shared_targets_skipped = false; structure = 0; parametric_report = J::object();
  call_equalities.clear();
  if (!checker.exponential_mode) return {};
  try {
    if (!collect_targets(reference, optimized, false)) return {};
    add_closure_targets(reference);
    collect_families();
    if (families.empty()) return {};
    ++a.stats["linear_reduction_parametric_attempts"];
    ParametricScan scan;
    parametric_scan(optimized, scan);
    Id rewritten = optimized;
    size_t passes = 0;
    for (bool proved_new = true; proved_new && passes <= families.size();) {
      proved_new = false; ++passes;
      for (auto &f : families) f.instances = f.declines = 0;
      rewritten = parametric_pass(optimized, scan, proved_new);
    }
    size_t theorems = 0, instances = 0, declines = 0;
    J proved = J::array(), all = J::array();
    for (const auto &f : families) {
      instances += f.instances; declines += f.declines;
      J item{{"spec", f.target.sid}, {"count", checker.specs[f.target.sid].count.str()},
             {"rows", str(a.nodes[f.axis].data[1] + Q(1))}, {"instances", f.instances},
             {"instance_declines", f.declines}};
      all.push_back(item);
      all.back()["proved"] = f.closed != invalid;
      all.back()["attempts"] = f.attempts;
      if (f.closed == invalid) continue;
      ++theorems;
      proved.push_back(std::move(item));
    }
    parametric_report = {{"families", all}, {"theorems", theorems}, {"passes", passes},
                         {"visited_nodes", visits}, {"structure_nodes", structure}};
    a.stats["linear_reduction_parametric_structure_peak"] =
        std::max<uint64_t>(a.stats["linear_reduction_parametric_structure_peak"], structure);
    a.stats["linear_reduction_parametric_visits_peak"] =
        std::max<uint64_t>(a.stats["linear_reduction_parametric_visits_peak"], visits);
    if (instances == 0 || rewritten == optimized) throw Miss{"no_instance"};
    const size_t norm_closures = inner_closures;
    const Id expected = reference;
    const auto identified = equal_call_pairs(call_atoms(reference), call_atoms(rewritten), 0);
    const Id actual = replace_atoms(rewritten, identified);
    parametric_report["identified_calls"] = identified.size();
    parametric_report["expected"] = expected; parametric_report["actual"] = actual;
    J receipt{{"method", "row_parametric_closure"}, {"families", std::move(proved)},
              {"row_instances", instances}, {"row_instance_declines", declines},
              {"unary_argument_closures", norm_closures}, {"passes", passes},
              {"visited_nodes", visits}, {"structure_nodes", structure},
              {"coverage", "one formal-row theorem per family; every replaced row an exact instance"}};
    if (checker.scalar.equal(expected, actual)) receipt["then"] = "exact_normal_form";
    else {
      auto result = direct(expected, actual);
      if (!result) {
        last_failure = "parametric_" + last_failure; parametric_report["decline"] = last_failure;
        ++a.stats["linear_reduction_decline_" + last_failure]; return {};
      }
      receipt["then"] = std::move(result->receipt);
    }
    ++a.stats["linear_reduction_parametric_proofs"];
    return LinearReductionProof{reference, std::move(receipt)};
  } catch (const Miss &miss) {
    last_failure = std::string("parametric_") + miss.reason;
    ++a.stats["linear_reduction_decline_" + last_failure];
  } catch (const Unsupported &) {
    last_failure = "parametric_unsupported_local_match";
    ++a.stats["linear_reduction_decline_" + last_failure];
  }
  parametric_report["decline"] = last_failure;
  parametric_report["structure_nodes"] = structure; parametric_report["visited_nodes"] = visits;
  return {};
}

// Work per prove(), in budgeted passes (each pass resets and is bounded by
// the node, piece and polynomial budgets): direct() at most 4;
// select_arms() at most max_select_arm_pairs * 4; the fused attempt 1; a
// closed-rewrite direct() at most 4; a nested-mode direct() at most 4;
// the row-parametric attempt 1 (its structural work bounded by
// max_structure_nodes) plus one direct() of at most 4.
// Nested closures run inside their pass's remaining budget and are charged
// to it.
inline auto LinearReductionPattern::prove(Id reference, Id optimized) -> std::optional<LinearReductionProof> {
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
  // Complete reductions under non-linear operators of the optimized value (a logarithm,
  // reciprocal or comparison of a Sum): replace each by C * call (substitute_reduction_atoms),
  // then require the exact EP equality or the exact case analysis of the two values.
  try {
    visits = piece_count = scalar_work = 0;
    const auto closes = [&](Id substituted) {
      return substituted != optimized &&
             (checker.scalar.equal(reference, substituted) || case_equal(reference, substituted, a.one) ||
              chain_case_equal(reference, substituted));
    };
    Id substituted = substitute_reduction_atoms(reference, optimized);
    bool closed = closes(substituted);

    if (!closed && atom_products) {
      // Once more with factored Sums as atom candidates.
      const Id wider = substitute_reduction_atoms(reference, optimized, true);
      closed = wider != substituted && closes(wider);
    }
    if (!closed) closed = light_attempt(reference, optimized);
    if (closed) {
      ++a.stats["linear_reduction_atom_proofs"];
      return LinearReductionProof{reference, {{"method", "complete_reductions_under_nonlinear_operators"},
          {"equality", "exact EP equality or exact case analysis of the substituted value"}}};
    }
  } catch (const Miss &) {} catch (const Unsupported &) {}
  // Last: only where every route above declined.
  if (auto result = parametric_attempt(reference, optimized)) return result;
  last_failure = declined;
  return {};
}

} // namespace rg
