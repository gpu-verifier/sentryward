#pragma once
#include "core.hpp"

namespace rg {

// Whether the Boolean c implies the Boolean q, by two local checks: q is c or one of c's factors
// (c a product of Booleans holds only when every factor does); or q compares an integer x with a
// constant, c has a factor [x != e] with e an end of x's bounds [lo, hi] (or a factor [x == e]), and
// q holds on every integer of the remaining interval (at e). Sufficient, not complete. O(factors).
inline auto Arena::boolean_implies(Id c,Id q) -> bool {
  if(c==q) return true;
  std::vector<Id> factors;
  {
    const auto &n=nodes[c];
    if(n.op=="mul" && n.data.size()==1 && n.data[0]==1) factors=n.args; else factors={c};
  }
  for(Id f:factors) if(f==q) return true;
  for(Id f:factors) if(!boolean_node(f)) return false;
  const auto goal=nodes[q];
  if(goal.op!="compare" || goal.args.size()!=2 || !goal.data.empty()) return false;
  const auto k=value(goal.args[1]);
  const Id x=goal.args[0];
  const auto b=nodes[x].bounds;
  if(!k || !b || !nodes[x].facts.finite || !known_bits(x).integer ||
     b->first.denominator()!=1 || b->second.denominator()!=1) return false;
  for(Id f:factors) {
    const auto &fn=nodes[f];
    if(fn.op!="compare" || (fn.tag!="ne" && fn.tag!="eq") || fn.args.size()!=2 || !fn.data.empty() || fn.args[0]!=x) continue;
    const auto excluded=value(fn.args[1]);
    if(!excluded || excluded->denominator()!=1) continue;
    Q lo=b->first, hi=b->second;
    if(fn.tag=="eq") lo=hi=*excluded; // the factor fixes x
    else if(*excluded==lo) lo+=1; else if(*excluded==hi) hi-=1; else continue;
    if(lo>hi) return true; // the factor never holds
    const std::string &t=goal.tag;
    if((t=="ge" && lo>=*k) || (t=="gt" && lo>*k) || (t=="le" && hi<=*k) || (t=="lt" && hi<*k) ||
       (t=="ne" && (*k<lo || *k>hi)) || (t=="eq" && lo==hi && lo==*k)) return true;
  }
  return false;
}

// select([x == lo], lo + 1, x) as max(lo + 1, x) (Arena::select, rule clamped sentinel): x a finite
// integer whose lower bound is the integer lo, either operand order of the comparison. O(1).
inline auto Arena::sentinel_clamp(Id p, Id yes, Id no) -> std::optional<Id> {
  const auto &c = nodes[p];
  if (c.op != "compare" || c.tag != "eq" || c.args.size() != 2 || !c.data.empty()) return {};
  for (unsigned side = 0; side != 2; ++side) {
    const Id x = c.args[side];
    const auto k = value(c.args[1 - side]);
    const auto &b = nodes[x].bounds;
    if (x != no || !k || k->denominator() != 1 || !nodes[x].facts.finite || !b || b->first != *k ||
        value(yes) != *k + 1 || !known_bits(x).integer) continue;
    ++stats["sentinel_clamp_selects"];
    return maximum({yes, x});
  }
  return {};
}

// The Max m of a condition c = [m == -inf] (either operand order) when m is not finite, else invalid.
inline auto Arena::neginf_max_comparison(Id c) const -> Id {
  const auto &n = nodes[c];
  if (n.op != "compare" || n.tag != "eq" || n.args.size() != 2 || !n.data.empty()) return invalid;
  for (unsigned side = 0; side != 2; ++side) {
    const Id m = n.args[side];
    if (!neginf(n.args[1 - side]) || nodes[m].facts.finite || !nodes[m].facts.finite_or_negative_infinity())
      continue;
    Id inner = m;
    if (nodes[m].op == "mul" && nodes[m].args.size() == 1 && nodes[m].data.size() == 1 && nodes[m].data[0] > 0)
      inner = nodes[m].args[0];
    if (nodes[inner].op == "max") return m;
  }
  return invalid;
}

// Whether c = [m == -inf] forces x = 0: x is a finite Select select(q, 0, y) for a condition q that
// m = -inf forces (neginf_max_conditions), a finite product with such a factor (the other factors are
// finite), or a finite Sum, or product with a Sum factor, whose terms are all forced 0 this way (at
// most three nested Sums). Sufficient, not complete. O(visited terms * factors + conditions).
inline auto Arena::neginf_forces_zero(Id c, Id x) -> bool {
  const Id m = neginf_max_comparison(c);
  if (m == invalid || !nodes[x].facts.finite) return false;
  const auto &forced = neginf_max_conditions(m);
  if (forced.empty()) return false;
  const auto zero_factor = [&](Id f) {
    const auto &s = nodes[f];
    return s.op == "select" && s.args.size() == 3 && s.args[1] == zero &&
           std::find(forced.begin(), forced.end(), s.args[0]) != forced.end();
  };
  // A finite term is 0 when it is such a Select, a product with such a factor, or a Sum (or a
  // product with a Sum factor) whose every term is 0 in this sense; at most three nested Sums.
  std::function<bool(Id, unsigned)> zero_term = [&](Id t, unsigned depth) -> bool {
    const auto &n = nodes[t];
    if (!n.facts.finite) return false;
    if (zero_factor(t)) return true;
    if (n.op == "add" && n.args.size() == n.data.size()) {
      if (depth == 0 || n.args.size() > 4096) return false;
      const auto terms = n.args;
      return std::all_of(terms.begin(), terms.end(), [&](Id u) { return zero_term(u, depth - 1); });
    }
    if (n.op != "mul" || n.data.size() != 1 || n.args.size() > 16) return false;
    const auto factors = n.args;
    return std::any_of(factors.begin(), factors.end(), [&](Id f) {
      return zero_factor(f) || (nodes[f].op == "add" && zero_term(f, depth));
    });
  };
  return zero_term(x, 3);
}

// [s == -inf] for an extended sum s = sum_i c_i t_i (a raw sum): when every non-finite term has a
// positive coefficient and is either select(C, -inf, F) with F finite (it is -inf exactly when
// C != 0) or a product of finite positive factors and one log(X) or log2(X) of a finite
// nonnegative X (it is -inf exactly when X = 0), then s = -inf exactly when one of those
// conditions holds (the finite terms add a finite amount; no +inf and no undefined term occurs).
// When one condition is [X == 0] and every other condition c is [m == -inf] forcing X = 0
// (neginf_forces_zero), that disjunction is [X == 0] itself. Returns it, or nullopt. Exact.
// O(terms + forced-zero checks).
inline auto Arena::neginf_sum_condition(Id s) -> std::optional<Id> {
  const auto &n = nodes[s];
  if (n.args.size() != n.data.size() || n.args.size() < 2 || n.args.size() > 64) return {};
  const auto args = n.args;
  const auto data = n.data;
  std::vector<Id> masks;
  Id mass = invalid;
  for (size_t i = 0; i < args.size(); ++i) {
    const auto &t = nodes[args[i]];
    if (t.facts.finite) continue;
    if (data[i] <= 0) return {};
    if (t.op == "select" && t.args.size() == 3 && neginf(t.args[1]) && nodes[t.args[2]].facts.finite) {
      masks.push_back(t.args[0]);
      continue;
    }
    const std::vector<Id> factors = t.op == "mul" && t.data.size() == 1 && t.data[0] > 0 ? t.args : std::vector<Id>{args[i]};
    if (t.op == "mul" && (t.data.size() != 1 || t.data[0] <= 0)) return {};
    Id logarithm = invalid;
    for (Id f : factors) {
      const auto &g = nodes[f];
      if (g.facts.finite && g.facts.pos) continue;
      if ((g.op == "log" || g.op == "log2") && g.args.size() == 1 && logarithm == invalid) { logarithm = f; continue; }
      return {};
    }
    if (logarithm == invalid || mass != invalid) return {};
    const Id x = nodes[logarithm].args[0];
    if (!nodes[x].facts.finite || !nodes[x].facts.nn) return {};
    mass = x;
  }
  if (mass == invalid) return {};
  for (Id c : masks)
    if (neginf_max_comparison(c) == invalid || !neginf_forces_zero(c, mass)) return {};
  return compare("eq", mass, zero);
}

// The arm B of select([X == 0], A, B) (Arena::select, rule guard of a zero test) with every Select
// select(c, u, w) on a condition c = [m == -inf] that forces X = 0 (neginf_forces_zero) replaced by w:
// B is taken only where X != 0, where c is false. Only non-finite shells are searched (such a Select
// with an -inf arm is not finite; finite subterms are kept), through add, add_raw, mul, recip, exp,
// exp2 and Select arms, at most 64 nodes; each rebuilt shell is built by its constructor. Returns
// B itself when nothing is replaced. Exact.
inline auto Arena::neginf_guarded_arm(Id p, Id arm) -> std::optional<Id> {
  const auto &c = nodes[p];
  if (c.op != "compare" || c.tag != "eq" || c.args.size() != 2 || !c.data.empty()) return {};
  Id x = invalid;
  if (c.args[1] == zero) x = c.args[0]; else if (c.args[0] == zero) x = c.args[1];
  if (x == invalid || !nodes[x].facts.finite || nodes[arm].facts.finite) return {};
  std::unordered_map<Id, Id> memo;
  std::unordered_map<Id, bool> forcing;
  size_t visits = 0;
  bool failed = false, replaced = false;
  const std::function<Id(Id)> rebuild = [&](Id id) -> Id {
    if (failed) return id;
    if (nodes[id].facts.finite || nodes[id].args.empty()) return id;
    if (auto it = memo.find(id); it != memo.end()) return it->second;
    if (++visits > 64) { failed = true; return id; }
    const std::string op = nodes[id].op;
    const auto args = nodes[id].args;
    const auto data = nodes[id].data;
    Id result = id;
    if (op == "select" && args.size() == 3) {
      bool forces = false;
      if (neginf_max_comparison(args[0]) != invalid) {
        auto [it, fresh] = forcing.emplace(args[0], false);
        if (fresh) it->second = neginf_forces_zero(args[0], x);
        forces = it->second;
      }
      if (forces) { replaced = true; result = rebuild(args[2]); }
      else {
        const Id yes = rebuild(args[1]), no = rebuild(args[2]);
        if (yes != args[1] || no != args[2]) result = select(args[0], yes, no);
      }
    } else if (op == "add" || op == "add_raw" || op == "mul" || op == "recip" || op == "exp" || op == "exp2") {
      std::vector<Id> rebuilt; bool changed = false;
      for (Id y : args) { rebuilt.push_back(rebuild(y)); changed |= rebuilt.back() != y; }
      if (changed) {
        if (op == "add" || op == "add_raw") {
          if (data.size() != rebuilt.size()) { failed = true; return id; }
          std::vector<Id> terms;
          for (size_t j = 0; j < rebuilt.size(); ++j) terms.push_back(mul({constant(data[j]), rebuilt[j]}));
          result = add(std::move(terms));
        } else if (op == "mul") {
          if (data.size() != 1) { failed = true; return id; }
          rebuilt.push_back(constant(data[0]));
          result = mul(std::move(rebuilt));
        } else if (op == "recip") result = recip(rebuilt[0]);
        else result = exponential(rebuilt[0], op == "exp2");
      }
    }
    memo.emplace(id, result);
    return result;
  };
  const Id result = rebuild(arm);
  if (failed || !replaced) return {};
  return result;
}

} // namespace rg
