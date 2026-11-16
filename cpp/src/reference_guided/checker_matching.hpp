#pragma once
#include "checker.hpp"

namespace rg {

// Nonrecursive local matching. Only commutative nodes branch. Exact capture
// agreement is enforced on every occurrence, even with shared pattern nodes.
inline auto Checker::instantiated_coordinate(Id root, const std::vector<Id> &captures) -> std::optional<Id> {
  // Only the bounded coordinate pattern is evaluated, never an operand's
  // numerical ancestry. All variables must already have exact captures.
  std::unordered_map<Id, Id> memo;
  std::vector<std::pair<Id, bool>> todo{{root, false}};
  size_t visits = 0;
  while (!todo.empty()) {
    a.check();
    if (++visits > 128) return {};
    auto [id, ready] = todo.back(); todo.pop_back();
    if (memo.contains(id)) continue;
    const auto &p = patterns.nodes[id];
    if (p.args.size() > 8) return {};
    if (p.op == "slot") {
      size_t slot = p.data.at(0).numerator().convert_to<size_t>();
      if (slot >= captures.size() || captures[slot] == invalid) return {};
      memo[id] = captures[slot]; continue;
    }
    if (p.op == "var" && p.args.empty()) {
      // An invariant variable outside read coordinates remains a literal
      // pattern leaf. Reuse its exact Arena identity; do not invent facts.
      auto existing = a.find_node(p.op, p.args, p.data, p.tag);
      if (!existing) return {};
      memo[id] = *existing; continue;
    }
    if (p.op != "const" && p.op != "add" && p.op != "mul" &&
        p.op != "idiv" && p.op != "imod") return {};
    if (!ready) {
      todo.emplace_back(id, true);
      for (Id child : p.args) if (!memo.contains(child)) todo.emplace_back(child, false);
      continue;
    }
    std::vector<Id> args; for (Id child : p.args) args.push_back(memo.at(child));
    Id value = invalid;
    if (p.op == "const" && p.data.size() == 1) value = a.constant(p.data[0]);
    else if (p.op == "add" && args.size() == p.data.size()) {
      for (size_t j = 0; j < args.size(); ++j) args[j] = a.mul({a.constant(p.data[j]), args[j]});
      value = a.add(args);
    } else if (p.op == "mul" && p.data.size() == 1) {
      args.push_back(a.constant(p.data[0])); value = a.mul(args);
    } else if ((p.op == "idiv" || p.op == "imod") && args.size() == 1 &&
               p.data.size() == 1 && p.data[0].denominator() == 1 && p.data[0] > 0)
      value = p.op == "idiv" ? a.idiv(args[0], p.data[0].numerator())
                                : a.imod(args[0], p.data[0].numerator());
    if (value == invalid) return {};
    memo[id] = value;
  }
  ++a.stats["instantiated_coordinate_patterns"];
  return memo.at(root);
}

inline auto Checker::deferred_coordinate_syntax(Id root) -> bool {
  if (auto it = deferred_coordinate_syntax_cache.find(root);
      it != deferred_coordinate_syntax_cache.end()) return it->second;
  std::vector<Id> todo{root};
  std::unordered_set<Id> seen;
  bool supported = true;
  while (!todo.empty()) {
    Id id = todo.back(); todo.pop_back();
    if (!seen.insert(id).second) continue;
    const auto &n = patterns.nodes[id];
    if (seen.size() > 128 || n.args.size() > 8 ||
        (n.op != "const" && n.op != "slot" && n.op != "var" &&
         n.op != "add" && n.op != "mul" && n.op != "idiv" && n.op != "imod")) {
      supported = false; break;
    }
    for (Id child : n.args) todo.push_back(child);
  }
  return deferred_coordinate_syntax_cache.emplace(root, supported).first->second;
}

inline auto Checker::matches(Id pat, Id target, std::vector<Id> captures, bool fixed_parameters) -> Captures {
  struct Task {
    Id pattern, target;
    size_t child = SIZE_MAX;
    std::vector<Id> remaining;
    bool coordinate = false;
    bool predicate_structural = false;
  };
  struct State {
    std::vector<Task> todo;
    std::vector<Id> cap;
    std::vector<Task> deferred;
    std::vector<std::pair<Id, Id>> coordinate_constraints;
    std::vector<std::pair<Id, Id>> coordinate_sums;
    std::vector<std::pair<Id, Id>> fixed_coordinates;
    std::vector<std::pair<Id, Id>> predicates;
  };
  std::vector<State> states{
      {{{pat, target, SIZE_MAX, {}}}, std::move(captures), {}, {}, {}, {}, {}}};
  Captures out;
  uint64_t work = 0;
  // Arena orders addends by local node Id. Substitution can change that
  // order, so small sums match by (coefficient, child), not by position.
  // Keep wider sums on the existing ordered path; the global work budget
  // also bounds combinations of nested small commutative nodes.
  auto small_add = [](const auto &n) {
    return n.op == "add" && n.args.size() > 1 && n.args.size() <= 4 &&
           n.data.size() == n.args.size();
  };
  auto compatible = [&](Id p, Id n, const std::vector<Id> &cap, bool deferred_predicate = true) {
    const auto &pn = patterns.nodes[p];
    if (pn.op == "slot") {
      size_t slot = pn.data[0].numerator().convert_to<size_t>();
      return slot >= cap.size() || cap[slot] == invalid || cap[slot] == n;
    }
    if (pn.op == "affine_slot" || pn.op == "affine_param" || pn.op == "derived_coordinate" ||
        pn.op == "coordinate_sum") return true;
    const auto &an = a.nodes[n];
    if (deferred_predicate && pn.op == "compare" &&
        (an.op == "compare" || n == a.zero || n == a.one) && affine_predicate_slots(p))
      return true;
    if (pn.op != an.op || pn.tag != an.tag || pn.args.size() != an.args.size())
      return false;
    if (small_add(pn) && small_add(an))
      return std::is_permutation(pn.data.begin(), pn.data.end(), an.data.begin());
    return pn.data == an.data;
  };
  while (!states.empty()) {
    State s = std::move(states.back());
    states.pop_back();
    bool ok = true;
    while (!s.todo.empty()) {
      a.check();
      if (++work > 1000000)
        throw Unsupported("local pattern matching work budget exceeded");
      Task task = std::move(s.todo.back());
      s.todo.pop_back();
      const auto &pn = patterns.nodes[task.pattern];
      const auto &an = a.nodes[task.target];
      if (task.child != SIZE_MAX) {
        if (task.child == pn.args.size())
          continue;
        // Branch only over candidates surviving this child's shallow shape
        // and current captures, then check the child before the next choice.
        std::vector<Id> choices;
        std::set<Id> seen;
        for (Id j : task.remaining)
          if ((pn.op != "add" || pn.data[task.child] == an.data[j]) &&
              seen.insert(an.args[j]).second &&
              compatible(pn.args[task.child], an.args[j], s.cap))
            choices.push_back(j);
        if (choices.empty()) {
          ok = false;
          break;
        }
        auto schedule = [&](State &next, Id j) {
          Task continuation = task;
          ++continuation.child;
          continuation.remaining.erase(
              std::find(continuation.remaining.begin(),
                        continuation.remaining.end(), j));
          next.todo.push_back(std::move(continuation));
          next.todo.push_back(
              {pn.args[task.child], an.args[j], SIZE_MAX, {}});
        };
        for (size_t i = 1; i < choices.size(); ++i) {
          State next = s;
          schedule(next, choices[i]);
          states.push_back(std::move(next));
          ++a.stats["pattern_branch_states"];
        }
        schedule(s, choices[0]);
        continue;
      }
      ++a.counter(Arena::Hot::pattern_node_visits);
      if (!task.predicate_structural && pn.op == "compare" &&
          (an.op == "compare" || task.target == a.zero || task.target == a.one) &&
          affine_predicate_slots(task.pattern)) {
        const auto predicate = std::pair(task.pattern, task.target);
        if (std::find(s.predicates.begin(), s.predicates.end(), predicate) == s.predicates.end()) {
          if (s.predicates.size() == CoordinateRelations::max_constraints) { ok = false; break; }
          s.predicates.push_back(predicate);
        }
        continue;
      }
      if (fixed_parameters && task.coordinate &&
          (pn.op == "add" || pn.op == "mul" || pn.op == "idiv" || pn.op == "imod") &&
          deferred_coordinate_syntax(task.pattern)) {
        if (s.fixed_coordinates.size() == CoordinateRelations::max_constraints) { ok = false; break; }
        s.fixed_coordinates.emplace_back(task.pattern, task.target);
        continue;
      }
      // Rule (coordinate in local form, local_coordinate_match): a read coordinate whose pattern is
      // fully captured is met when its instantiation has the same local index form as the target's
      // coordinate (spelled differently, or not structurally decomposable). Checked on this pair only;
      // otherwise the structural match below proceeds as before.
      if (task.coordinate && local_coordinate_match(task.pattern, task.target, s.cap)) continue;
      // For an Add with one coordinate recipe, assign every other summand to a
      // distinct target term with the same coefficient, then match the coordinate
      // recipe to the remaining target divided by its coefficient. These are
      // capture proposals only; the complete body proof still follows. Bound the
      // search by rest_summands, rest_terms and rest_branches.
      if (task.coordinate && pn.op == "add" && pn.args.size() == pn.data.size() && pn.args.size() >= 2 &&
          pn.args.size() <= rest_summands && !compatible(task.pattern, task.target, s.cap, !task.predicate_structural)) {
        const auto recipe = [&](Id p) {
          const auto &op = patterns.nodes[p].op;
          return op == "slot" || op == "derived_coordinate" || op == "affine_slot" || op == "affine_param" ||
                 op == "coordinate_sum";
        };
        const auto pargs = pn.args; const auto pdata = pn.data;
        size_t coordinate = pargs.size(), recipes = 0;
        for (size_t i = 0; i < pargs.size(); ++i)
          if (recipe(pargs[i])) { coordinate = i; ++recipes; }
          else if (pdata[i] == 0) recipes = 2;
        const auto terms = a.terms(task.target);
        if (recipes == 1 && pdata[coordinate] != 0 && terms.size() <= rest_terms) {
          std::vector<size_t> others;
          for (size_t i = 0; i < pargs.size(); ++i) if (i != coordinate) others.push_back(i);
          std::vector<size_t> chosen;
          std::vector<bool> used(terms.size(), false);
          size_t branches = 0;
          std::function<void(size_t)> assign = [&](size_t position) {
            if (branches == rest_branches) return;
            if (position == others.size()) {
              std::vector<Id> rest;
              for (size_t j = 0; j < terms.size(); ++j)
                if (!used[j]) rest.push_back(a.mul({a.constant(terms[j].second), terms[j].first}));
              State next = s;
              next.todo.push_back({pargs[coordinate],
                                   a.mul({a.constant(Q(1) / pdata[coordinate]), a.add(std::move(rest))}),
                                   SIZE_MAX, {}, true});
              for (size_t k = 0; k < others.size(); ++k)
                next.todo.push_back({pargs[others[k]], terms[chosen[k]].first, SIZE_MAX, {}, false});
              states.push_back(std::move(next));
              ++branches;
              ++a.stats["pattern_coordinate_rest_branches"];
              return;
            }
            const size_t i = others[position];
            for (size_t j = 0; j < terms.size(); ++j) {
              if (used[j] || terms[j].first == a.one || terms[j].second != pdata[i]) continue;
              if (!compatible(pargs[i], terms[j].first, s.cap)) continue;
              used[j] = true; chosen.push_back(j);
              assign(position + 1);
              used[j] = false; chosen.pop_back();
            }
          };
          assign(0);
          ok = false;  // this state continues only in the branches pushed above
          break;
        }
      }
      if (!compatible(task.pattern, task.target, s.cap, !task.predicate_structural)) {
        ok = false;
        break;
      }
      if (pn.op == "slot") {
        size_t slot = pn.data[0].numerator().convert_to<size_t>();
        if (slot >= s.cap.size())
          s.cap.resize(slot + 1, invalid);
        s.cap[slot] = task.target;
        continue;
      }
      if (pn.op == "derived_coordinate") {
        auto constraint = std::pair(task.pattern, task.target);
        if (std::find(s.coordinate_constraints.begin(), s.coordinate_constraints.end(), constraint) ==
            s.coordinate_constraints.end()) {
          if (s.coordinate_constraints.size() == CoordinateRelations::max_constraints) { ok = false; break; }
          s.coordinate_constraints.push_back(constraint);
        }
        continue;
      }
      if (pn.op == "coordinate_sum") {
        // A decoder reuses its input in the fraction, exponent and sign.
        // Identical constraints are one proof obligation, not independent
        // work. Different actual coordinates remain separate obligations.
        auto constraint = std::pair(task.pattern, task.target);
        if (std::find(s.coordinate_sums.begin(), s.coordinate_sums.end(), constraint) == s.coordinate_sums.end()) {
          if (s.coordinate_sums.size() == CoordinateRelations::max_constraints) { ok = false; break; }
          s.coordinate_sums.push_back(constraint);
        }
        continue;
      }
      if (pn.op == "affine_param") {
        s.deferred.push_back(std::move(task));
        continue;
      }
      if (pn.op == "affine_slot") {
        size_t slot = pn.data[0].numerator().convert_to<size_t>();
        Id captured = a.mul({a.constant(Q(1) / pn.data[1]),
                             a.sub(task.target, a.constant(pn.data[2]))});
        if (slot >= s.cap.size()) s.cap.resize(slot + 1, invalid);
        if (s.cap[slot] != invalid && s.cap[slot] != captured) {
          ok = false;
          break;
        }
        s.cap[slot] = captured;
        ++a.stats["affine_coordinate_captures"];
        continue;
      }
      if (((pn.op == "mul" || pn.op == "max") && pn.args.size() > 1) ||
          small_add(pn)) {
        std::vector<Id> remaining(an.args.size());
        std::iota(remaining.begin(), remaining.end(), Id(0));
        s.todo.push_back(
            {task.pattern, task.target, 0, std::move(remaining)});
      } else {
        for (size_t j = pn.args.size(); j-- > 0;)
          s.todo.push_back({pn.args[j], an.args[j], SIZE_MAX, {},
              pn.op == "read" || pn.op == "readbits" || pn.op == "packedbits" || pn.op == "call"});
      }
    }
    // Known output parameters anchor the one unknown axis directly. This
    // is a single affine equation, not a search for an index transformation.
    if (ok && fixed_parameters) for (const auto &task : s.deferred) {
      const auto &p = patterns.nodes[task.pattern];
      size_t axis = p.data[0].numerator().convert_to<size_t>();
      size_t residual = p.data[1].numerator().convert_to<size_t>();
      if (axis < s.cap.size() && s.cap[axis] == invalid && residual < s.cap.size() &&
          s.cap[residual] != invalid && p.data[2] != 0 &&
          a.nodes[task.target].args.size() <= 8 && a.nodes[s.cap[residual]].args.size() <= 8) {
        s.cap[axis] = a.mul({a.constant(Q(1) / p.data[2]), a.sub(task.target, s.cap[residual])});
        ++a.stats["parameter_anchored_reduction_axes"];
      }
    }
    if (ok && !s.coordinate_constraints.empty()) {
      // Rule (remainder candidates of an axis, CoordinateRelations::remainder_candidates): one state
      // per candidate value of an axis only a remainder constrains; each is completed as usual.
      if (auto alternatives = CoordinateRelations::remainder_candidates(a, patterns, s.coordinate_constraints, s.cap)) {
        ++a.stats["remainder_axis_candidate_forks"];
        for (Id value : alternatives->second) {
          State copy = s;
          if (alternatives->first >= copy.cap.size()) copy.cap.resize(alternatives->first + 1, invalid);
          copy.cap[alternatives->first] = value;
          states.push_back(std::move(copy));
        }
        continue;
      }
      ok = CoordinateRelations::resolve(a, patterns, s.coordinate_constraints, s.cap, work);
    }
    // Parameterized affine residuals require an axis captured independently:
    // an ordinary/literal-affine coordinate or a verified quotient/remainder
    // pair. Residual tasks never determine their own axis.
    if (ok) for (const auto &task : s.deferred) {
      a.check();
      if (++work > 1000000)
        throw Unsupported("local pattern matching work budget exceeded");
      const auto &pn = patterns.nodes[task.pattern];
      size_t axis = pn.data[0].numerator().convert_to<size_t>();
      size_t residual = pn.data[1].numerator().convert_to<size_t>();
      if (axis >= s.cap.size() || s.cap[axis] == invalid ||
          !a.nodes[s.cap[axis]].facts.finite ||
          !a.nodes[task.target].facts.finite ||
          a.nodes[s.cap[axis]].args.size() > 8 ||
          a.nodes[task.target].args.size() > 8) {
        ok = false;
        break;
      }
      Id captured = a.sub(task.target,
                          a.mul({a.constant(pn.data[2]), s.cap[axis]}));
      if (residual >= s.cap.size()) s.cap.resize(residual + 1, invalid);
      if (s.cap[residual] != invalid && s.cap[residual] != captured) {
        ok = false;
        break;
      }
      s.cap[residual] = captured;
      ++a.stats["parameterized_affine_coordinate_captures"];
    }
    // Short coordinate sums may only use an axis already captured by an
    // ordinary coordinate or a verified quotient/remainder pair. Each
    // residual parameter must agree with every other occurrence; existing
    // summary binding still rejects parameters dependent on a Fold binder.
    if (ok) for (const auto &[pattern, actual] : s.coordinate_sums) {
      if (!CoordinateRelations::resolve_sum(a, patterns, pattern, actual, s.cap, work)) {
        ok = false;
        break;
      }
    }
    if (ok) for (const auto &[pattern, actual] : s.fixed_coordinates) {
      auto expected = instantiated_coordinate(pattern, s.cap);
      if ((!expected || *expected != actual) && !local_coordinate_match(pattern, actual, s.cap)) { ok = false; break; }
    }
    if (ok) for (const auto &[pattern, actual] : s.predicates) {
      const auto &slots = *affine_predicate_slots(pattern);
      const bool bound = std::all_of(slots.begin(), slots.end(), [&](size_t slot) {
        return slot < s.cap.size() && s.cap[slot] != invalid;
      });
      if (!bound) {
        // No independent coordinate anchor: preserve the original exact
        // structural match. This one-way requeue cannot defer itself again.
        s.todo.push_back({pattern, actual, SIZE_MAX, {}, false, true});
        ++a.stats["predicate_structural_fallbacks"];
        continue;
      }
      const auto &predicate = patterns.nodes[pattern];
      auto lhs = instantiated_coordinate(predicate.args[0], s.cap);
      auto rhs = instantiated_coordinate(predicate.args[1], s.cap);
      const auto &actual_node = a.nodes[actual];
      Id canonical_actual = actual_node.op == "compare" && actual_node.args.size() == 2
          ? a.compare(actual_node.tag, actual_node.args[0], actual_node.args[1]) : actual;
      ++a.stats["deferred_predicate_checks"];
      if (!lhs || !rhs || a.compare(predicate.tag, *lhs, *rhs) != canonical_actual) {
        ++a.stats["deferred_predicate_mismatches"]; ok = false; break;
      }
    }
    if (ok && !s.todo.empty()) {
      // The completed coordinate obligations remain valid: captures only
      // become bound, never change. Do not repeat their work after fallback.
      s.deferred.clear(); s.coordinate_constraints.clear(); s.coordinate_sums.clear();
      s.fixed_coordinates.clear(); s.predicates.clear();
      states.push_back(std::move(s));
      continue;
    }
    if (ok)
      out.push_back(std::move(s.cap));
  }
  return out;
}

// A read anchor (not cacheable_read_anchor) whose coordinates are integer additions and
// constant multiples of derived coordinates, slots and constants, with every derived
// coordinate's captured axis an integer inside its scope. O(pattern coordinate nodes).
inline auto Checker::scoped_read_anchor(Id pattern,const std::vector<Id> &cap) -> bool {
  const auto &p=patterns.nodes[pattern];
  if((p.op!="read" && p.op!="readbits" && p.op!="packedbits") || p.args.size()>read_anchor_coordinate_limit)
    return false;
  std::vector<Id> todo(p.args.begin(),p.args.end());
  for(size_t visits=0;!todo.empty();++visits) {
    if(visits>scoped_anchor_visit_limit)return false;
    const auto &n=patterns.nodes[todo.back()];todo.pop_back();
    if(n.op=="derived_coordinate") {
      if(!n.args.empty() || !CoordinateRelations::valid(n.data))return false;
      const auto axis=n.data[0].numerator().convert_to<size_t>();
      if(axis>=cap.size() || cap[axis]==invalid || !CoordinateRelations::within_axis(a,cap[axis],n.data))return false;
    } else if(n.op=="add" || n.op=="mul") {
      for(const auto &q:n.data)if(q.denominator()!=1)return false;
      todo.insert(todo.end(),n.args.begin(),n.args.end());
    } else if((n.op!="slot" && n.op!="const") || !n.args.empty())return false;
  }
  return true;
}

// The captures under which a scoped read anchor may name a target: `cap` itself when every
// slot is captured; otherwise, for one missing slot that is a plain term c * slot of a
// single-coordinate address, the integer value (target address - rest) / c per same-kind
// target. Each target solved for the slot is one attempt of the body-candidate search
// (`attempts`); none when that budget is spent. O(targets).
inline auto Checker::scoped_read_captures(Id pattern,const std::vector<Id> &cap,const std::vector<Id> &ts,
                                             size_t &attempts) -> std::optional<Captures> {
  const auto missing=std::count(cap.begin(),cap.end(),invalid);
  if(!missing)return Captures{cap};
  const auto &p=patterns.nodes[pattern];
  if(missing!=1 || p.args.size()!=1)return Captures{};
  const auto slot=size_t(std::find(cap.begin(),cap.end(),invalid)-cap.begin());
  const auto &address=patterns.nodes[p.args[0]];
  if(address.op!="add" || address.args.size()!=address.data.size())return Captures{};
  std::optional<Q> coefficient;
  for(size_t j=0;j<address.args.size();++j) {
    const auto &term=patterns.nodes[address.args[j]];
    if(term.op=="slot" && term.data.size()==1 && term.data[0]==Q(slot)) {
      if(coefficient)return Captures{};
      coefficient=address.data[j];
    }
  }
  if(!coefficient || *coefficient==0)return Captures{};
  auto zero=cap;zero[slot]=a.zero;
  const Id rest=instantiate_body(pattern,zero);
  if(rest==invalid || a.nodes[rest].args.size()!=1)return Captures{};
  const Id rest_address=a.nodes[rest].args[0];
  const auto rest_data=a.nodes[rest].data;
  Captures result;
  for(Id t:ts) {
    if(a.nodes[t].op!=p.op || a.nodes[t].tag!=p.tag || a.nodes[t].data!=rest_data || a.nodes[t].args.size()!=1)continue;
    if(++attempts>body_candidate_attempts)return std::nullopt;
    const Id target_address=a.nodes[t].args[0];
    Id value=a.mul({a.constant(Q(1)/ *coefficient),a.sub(target_address,rest_address)});
    if(!CoordinateRelations::integer(a,value))continue;
    auto solved=cap;solved[slot]=value;result.push_back(std::move(solved));
  }
  return result;
}

// Whether every occurrence of the anchor `anchor` in the pattern `root` lies in a Select arm whose
// condition, instantiated with each capture of `caps`, is a constant excluding that arm. Paths are
// followed through ordinary pattern nodes (not into anchors), at most 512 visits; an occurrence on a
// path without such an excluded arm, a capture that leaves a condition open, or a cut search gives
// false. O(visits * captures).
inline auto Checker::excluded_anchor(Id root, Id anchor, const Captures &caps) -> bool {
  if (caps.empty() || root == anchor) return false;
  struct Visit { Id node; std::vector<std::pair<Id, unsigned>> arms; };
  std::vector<Visit> todo{{root, {}}};
  size_t visits = 0; bool found = false;
  while (!todo.empty()) {
    if (++visits > 512) return false;
    Visit v = std::move(todo.back()); todo.pop_back();
    if (v.node == anchor) {
      found = true;
      bool excluded = false;
      for (const auto &[select, arm] : v.arms) {
        bool all = true;
        for (const auto &cap : caps) {
          const Id condition = instantiate_body(patterns.nodes[select].args[0], cap);
          const auto value = condition == invalid ? std::nullopt : a.value(condition);
          if (!value || (arm == 1) == (*value != 0)) { all = false; break; }
        }
        if (all) { excluded = true; break; }
      }
      if (!excluded) return false;
      continue;
    }
    const auto &n = patterns.nodes[v.node];
    if (n.op == "read" || n.op == "readbits" || n.op == "packedbits" || n.op == "call" || n.op == "certified_state") continue;
    const auto args = n.args;
    for (size_t j = 0; j < args.size(); ++j) {
      Visit next{args[j], v.arms};
      if (n.op == "select" && args.size() == 3 && j > 0) next.arms.emplace_back(v.node, unsigned(j));
      todo.push_back(std::move(next));
    }
  }
  return found;
}

} // namespace rg
