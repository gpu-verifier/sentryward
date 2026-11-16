#pragma once
#include "linear_reduction_pattern.hpp"

namespace rg {

// A nested closure sub-proof runs with what remains of this pattern's
// budgets, and its work is charged back here afterwards, so one attempt's
// node, piece and polynomial budgets bound its nested sub-proofs too.
inline auto LinearReductionPattern::remaining_budget() const -> LinearReductionBudget {
  LinearReductionBudget rest = budget;
  rest.nodes = budget.nodes > visits ? budget.nodes - visits : 0;
  rest.pieces = budget.pieces > piece_count ? budget.pieces - piece_count : 0;
  rest.polynomial_work = budget.polynomial_work > scalar_work ? budget.polynomial_work - scalar_work : 0;
  return rest;
}

inline auto LinearReductionPattern::charge(const LinearReductionPattern &child) -> void {
  visits += child.visits; piece_count += child.piece_count; scalar_work += child.scalar_work;
  if (visits > budget.nodes) throw Miss{"node_budget"};
  if (piece_count > budget.pieces) throw Miss{"piece_budget"};
  if (scalar_work > budget.polynomial_work) throw Miss{"polynomial_budget"};
}

inline auto LinearReductionPattern::original(Id n) -> Id {
  if (n == a.zero) return invalid;
  if (auto it = original_witnesses.find(n); it != original_witnesses.end()) return it->second;
  Id result = witness({Witness::Original, invalid, invalid, n});
  original_witnesses.emplace(n, result); return result;
}

inline auto LinearReductionPattern::insert(Form &form, const Key &key, const Group &group) -> void {
  if (key.coefficient == a.zero || group.multiplicity == 0) return;
  auto [it, added] = form.groups.emplace(key, group);
  if (!added) {
    it->second.piece = concat(it->second.piece, group.piece);
    it->second.witness = sum_witness(it->second.witness, group.witness);
    it->second.multiplicity += group.multiplicity;
    if (it->second.guard != group.guard) it->second.guard = a.one;
  }
  if (form.groups.size() > budget.groups) throw Miss{"group_budget"};
}

inline auto LinearReductionPattern::append(Form &to, const Form &from, Id coefficient) -> void {
  if (coefficient == a.zero) return;
  if (!a.nodes[coefficient].facts.finite) throw Miss{"nonfinite_coefficient"};
  to.residual = sum_witness(to.residual, scale_witness(from.residual, coefficient));
  for (const auto &[key, group] : from.groups) {
    tick();
    insert(to, {key.target, a.mul({key.coefficient, coefficient})},
           {group.piece, scale_witness(group.witness, coefficient), group.multiplicity, group.guard});
  }
}

inline auto LinearReductionPattern::bound(const Form &body, Id variable, const Z &count) -> Form {
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
      // A binder-dependent guard is not one predicate for all bound members.
      insert(result, key, {bind_piece(group.piece, variable, count),
          bind_witness(group.witness, variable, count), group.multiplicity * count,
          a.depends(group.guard, variable) ? a.one : group.guard});
    }
  }
  result.residual = bind_witness(residual, variable, count);
  return result;
}

// The slot capture of a target's spec: its parameters in the non-axis
// slots and axis_value (or invalid) in the axis slot.
inline auto LinearReductionPattern::capture(const Target &target, Id axis_value) const -> std::vector<Id> {
  const Spec &spec = checker.specs[target.sid];
  std::vector<Id> cap(spec.slots, invalid);
  size_t parameter = 0;
  for (size_t slot = 0; slot < spec.slots; ++slot)
    cap[slot] = slot == spec.axis ? axis_value : target.parameters[parameter++];
  return cap;
}

// The mask predicate of the reference summand at one index: the summand is a product whose
// zero-armed Select factors share this predicate, so the summand is 0 whenever it is 0
// (Checker::masked_exponential_term); one when it has none or cannot be instantiated.
// Cached instantiation; O(summand shell) otherwise.
inline auto LinearReductionPattern::reference_guard(const Target &target, Id index) -> Id {
  const Spec &spec = checker.specs[target.sid];
  if (target.parameters.size() + 1 != spec.slots) return a.one;
  const Id body = checker.instantiate_body(spec.pattern, capture(target, index));
  if (body == invalid) return a.one;
  const auto term = checker.masked_exponential_term(body);
  return term ? term->predicate : a.one;
}

// The truth of `c` as a Select condition (nonzero selects the first arm): 1, 0 or -1 (open).
inline auto LinearReductionPattern::case_truth(Id c, const CaseFix &fixed) -> int {
  if (auto v = a.value(c)) return *v != 0;
  // Copies: a.sub below may grow the node store.
  const std::string op = a.nodes[c].op, tag = a.nodes[c].tag;
  const std::vector<Id> args = a.nodes[c].args;
  // (Light case mode, light_cases.) For a finite nonnegative x, [x <= 0] is [x == 0] and [x > 0]
  // is [x != 0]. Exact.
  if (light_cases && op == "compare" && args.size() == 2 && args[1] == a.zero && (tag == "le" || tag == "gt") &&
      a.nodes[args[0]].facts.finite && a.nodes[args[0]].facts.nn) {
    const Id zero_test = a.compare("eq", args[0], a.zero);
    if (zero_test != c) {
      const int truth = case_truth(zero_test, fixed);
      if (truth >= 0) { ++a.stats["linear_reduction_case_nonnegative_zero_tests"]; return tag == "le" ? truth : !truth; }
    }
  }
  for (const auto &[f, truth] : fixed) {
    if (f == c) return truth;
    const std::string fop = a.nodes[f].op, ftag = a.nodes[f].tag;
    const std::vector<Id> fargs = a.nodes[f].args;
    if (op != "compare" || fop != "compare" || args.size() != 2 || fargs.size() != 2) continue;
    const bool same = args == fargs, swapped = args[0] == fargs[1] && args[1] == fargs[0];
    const auto complement = [&](const std::string &x, const std::string &y) {
      return (tag == x && ftag == y) || (tag == y && ftag == x);
    };
    // eq/ne are complements on every value; lt/ge and le/gt only without NaN operands.
    const bool ordered = a.nodes[args[0]].facts.finite_or_negative_infinity() &&
                         a.nodes[args[1]].facts.finite_or_negative_infinity();
    if ((same || swapped) && complement("eq", "ne")) return !truth;
    if (same && ordered && (complement("lt", "ge") || complement("le", "gt"))) return !truth;
    // Equalities of finite operands with the same difference up to sign: x = y iff x - y = 0.
    const bool equalities = (tag == "eq" || tag == "ne") && (ftag == "eq" || ftag == "ne");
    const bool finite = a.nodes[args[0]].facts.finite && a.nodes[args[1]].facts.finite &&
                        a.nodes[fargs[0]].facts.finite && a.nodes[fargs[1]].facts.finite;
    if (equalities && finite) {
      const Id dn = a.sub(args[0], args[1]), dm = a.sub(fargs[0], fargs[1]);
      if (dn == dm || dn == a.neg(dm)) return tag == ftag ? truth : !truth;
    }
    // Comparisons with 0 of x and k*x for a finite positive k have the same truth.
    if (args[1] == a.zero && fargs[1] == a.zero && a.nodes[args[0]].facts.finite &&
        a.nodes[fargs[0]].facts.finite && positive_multiple(args[0], fargs[0])) {
      if (tag == ftag) return truth;
      if (complement("eq", "ne") || complement("lt", "ge") || complement("le", "gt")) return !truth;
    }
    // (Light case mode.) An aliased Call T with a proved Sum X = C * T (C finite and positive, atom_aliases):
    // comparisons with 0 of T, of a finite positive multiple of T, and of X have the same truth.
    if (light_cases && args[1] == a.zero && fargs[1] == a.zero && !atom_aliases.empty() &&
        a.nodes[args[0]].facts.finite && a.nodes[fargs[0]].facts.finite) {
      for (const auto &[call, alias] : atom_aliases) {
        const auto names = [&](Id v) { return v == call || v == alias.first || positive_multiple(v, call); };
        if (!names(args[0]) || !names(fargs[0])) continue;
        ++a.stats["linear_reduction_case_alias_truths"];
        if (tag == ftag) return truth;
        if (complement("eq", "ne") || complement("lt", "ge") || complement("le", "gt")) return !truth;
      }
    }
  }
  return -1;
}

// log2 of a positive value: log2(exp2(y)) = y and log2(exp(y)) = y / ln 2 for finite y; otherwise
// the Arena's log2 node. Exact.
inline auto LinearReductionPattern::log2_of(Id x) -> Id {
  const auto &n = a.nodes[x];
  if ((n.op == "exp2" || n.op == "exp") && n.args.size() == 1 && a.nodes[n.args[0]].facts.finite)
    return n.op == "exp2" ? n.args[0] : a.mul({n.args[0], a.inverse_natural_log2()});
  return a.logarithm(x, true);
}

// Whether one of x, y is the other times a finite positive factor (the rest of a product). O(factors).
inline auto LinearReductionPattern::positive_multiple(Id x, Id y) -> bool {
  for (const auto &[u, v] : {std::pair(x, y), std::pair(y, x)}) {
    const auto &n = a.nodes[u];
    if (n.op != "mul" || n.data.size() != 1) continue;
    const auto args = n.args;
    const auto at = std::find(args.begin(), args.end(), v);
    if (at == args.end()) continue;
    std::vector<Id> rest{a.constant(n.data[0])};
    for (auto it = args.begin(); it != args.end(); ++it) if (it != at) rest.push_back(*it);
    const Id k = a.mul(rest);
    if (a.nodes[k].facts.finite && a.nodes[k].facts.pos) return true;
  }
  return false;
}

// Whether every fixed condition compares with 0 an aliased Call, a finite positive multiple of
// one, or the proved Sum of an alias (atom_aliases); true when nothing is fixed. A proved Sum X is
// then used as it is instead of rebuilt in the case: the two are the same value in every case, so
// this only forgoes exposing X's own Selects, which such conditions do not fix. O(fixed * aliases).
inline auto LinearReductionPattern::alias_conditions_only(const CaseFix &fixed) -> bool {
  for (const auto &[f, truth] : fixed) {
    (void)truth;
    const auto &c = a.nodes[f];
    if (c.op != "compare" || c.args.size() != 2 || c.args[1] != a.zero) return false;
    bool aliased = false;
    for (const auto &[call, alias] : atom_aliases)
      if (c.args[0] == call || c.args[0] == alias.first || positive_multiple(c.args[0], call)) { aliased = true; break; }
    if (!aliased) return false;
  }
  return true;
}

inline auto LinearReductionPattern::resolve(Id x, const CaseFix &fixed, std::unordered_map<Id, Id> &memo, std::vector<Id> &open,
           size_t &work) -> Id {
  if (auto it = memo.find(x); it != memo.end()) return it->second;
  if (++work > case_rebuild_limit) throw CaseMiss{};
  a.check();
  const auto &n = a.nodes[x];
  Id result = x;
  // (Light case mode.) A finite value Max without bounds (a pivot such as max(floor, running
  // maxima)) is one atom here, like a Fold or a Call: its value is the same in every case, and its
  // inner guards are neither rebuilt nor split candidates. (Integer coordinate maxima carry bounds.)
  if (n.args.empty() || n.op == "fold" || n.op == "call" ||
      (light_cases && n.op == "max" && n.facts.finite && !n.bounds)) result = x;
  else if (n.op == "select" && n.args.size() == 3) {
    const auto args = n.args;
    const Id condition = resolve(args[0], fixed, memo, open, work);
    const int truth = case_truth(condition, fixed);
    if (truth >= 0) result = resolve(args[truth ? 1 : 2], fixed, memo, open, work);
    else {
      if (std::find(open.begin(), open.end(), condition) == open.end()) open.push_back(condition);
      const Id yes = resolve(args[1], fixed, memo, open, work), no = resolve(args[2], fixed, memo, open, work);
      result = condition == args[0] && yes == args[1] && no == args[2] ? x
               : checker.reapply(x, {condition, yes, no});
    }
  } else {
    const bool reciprocal = n.op == "recip" && n.args.size() == 1;
    const bool comparison = n.op == "compare" && n.args.size() == 2;
    const std::string n_op = n.op;
    const bool logarithm = (n.op == "log" || n.op == "log2") && n.args.size() == 1;
    const auto args = n.args;
    std::vector<Id> rebuilt;
    bool changed = false;
    for (Id y : args) { rebuilt.push_back(resolve(y, fixed, memo, open, work)); changed |= rebuilt.back() != y; }
    // The rebuilt operands equal the old ones only in this case, and the rebuilt node is shared
    // by every later use. Where the rebuild would keep the original node's facts or bounds
    // instead of deriving them from the new operands (Checker::reapply, strict), the case
    // analysis declines.
    if (changed) {
      result = checker.reapply(x, rebuilt, /*strict=*/true);
      if (result == invalid) { ++a.stats["linear_reduction_case_inherited_fact_declines"]; throw CaseMiss{}; }
    } else result = x;
    // A comparison with 0 of an aliased Call (or a positive multiple of one) is the comparison of
    // its proved Sum, resolved in this case (atom_aliases).
    // (Light case mode: a comparison the case already decides needs no alias; where only alias
    // conditions are fixed, the proved Sum is used without being rebuilt: alias_conditions_only.)
    if (comparison && !atom_aliases.empty() && a.nodes[result].op == "compare" &&
        a.nodes[result].args.size() == 2 && a.nodes[result].args[1] == a.zero &&
        (!light_cases || case_truth(result, fixed) < 0)) {
      const Id lhs = a.nodes[result].args[0];
      Id source = invalid;
      if (auto it = atom_aliases.find(lhs); it != atom_aliases.end()) source = it->second.first;
      else if (a.nodes[lhs].op == "mul")
        for (Id f : std::vector<Id>(a.nodes[lhs].args))
          if (auto it = atom_aliases.find(f); it != atom_aliases.end() && positive_multiple(lhs, f)) { source = it->second.first; break; }
      if (source != invalid) {
        // The Sum's own Selects are resolved apart: its open conditions are not split
        // candidates (a still-open comparison is itself one, at its Select).
        std::unordered_map<Id, Id> source_memo; std::vector<Id> source_open;
        // The operator of the rebuilt comparison, not of the original node: the constructor may
        // have returned another comparison ([B != 0] and [B == 1] are B, [B == 0] is B's
        // complement, an affine predicate may be normalized).
        const std::string tag = a.nodes[result].tag;
        // In the chain mode (finite_cases) the proved Sum is used as it is (it equals itself in every
        // case); its resolution is not needed for the case analysis there. In the light case mode the
        // same where only alias conditions are fixed (alias_conditions_only). The two modes are never
        // set together on one pattern.
        result = a.compare(tag, finite_cases || (light_cases && alias_conditions_only(fixed)) ? source
                                : resolve(source, fixed, source_memo, source_open, work), a.zero);
        ++a.stats["linear_reduction_case_aliased_comparisons"];
      }
    }
    // Logarithms of a value the case fixes positive (X <= 0 false): ln X = ln 2 * log2 X, and for
    // X = d f_1 ... f_k with d > 0 and every factor but at most one positive, every factor is
    // positive, so log2 X = log2 d + log2 f_1 + ... + log2 f_k. Exact.
    // log2 of a positive factor; an aliased Call T with X = C * T is log2 X - log2 C (X > 0 here).
    // Chain mode (finite_cases): a factor f positive in this case and not by its facts is taken as its
    // positive view select([f <= 0], 1, f), equal to f in this case and positive by
    // Arena::positive_view, so its logarithm is finite.
    const auto log2_positive = [&](Id f) -> Id {
      const auto &m = a.nodes[f];
      if (!finite_cases || m.facts.pos || !m.facts.finite ||
          ((m.op == "exp2" || m.op == "exp") && m.args.size() == 1 && a.nodes[m.args[0]].facts.finite))
        return log2_of(f);
      ++a.stats["linear_reduction_chain_positive_views"];
      return log2_of(a.select(a.compare("le", f, a.zero), a.one, f));
    };
    const auto log2_term = [&](Id f) -> Id {
      if (auto it = atom_aliases.find(f); it != atom_aliases.end()) {
        ++a.stats["linear_reduction_case_aliased_logarithms"];
        // Chain mode: the Sum as it is (positive: X = C * T with C > 0 and T > 0 here), the co-factor C
        // resolved in this case (both equal themselves on its inputs).
        if (finite_cases)
          return a.sub(log2_positive(it->second.first), log2_of(resolve(it->second.second, fixed, memo, open, work)));
        std::unordered_map<Id, Id> source_memo; std::vector<Id> source_open;
        const Id source = light_cases && alias_conditions_only(fixed) ? it->second.first
                          : resolve(it->second.first, fixed, source_memo, source_open, work);
        return a.sub(log2_of(source), log2_of(it->second.second));
      }
      return log2_positive(f);
    };
    if (logarithm) {
      const Id value = rebuilt[0];
      if (a.nodes[value].facts.finite &&
          case_truth(resolve(a.compare("le", value, a.zero), fixed, memo, open, work), fixed) == 0) {
        std::vector<Id> terms;
        const auto &m = a.nodes[value];
        if (m.op == "mul" && m.data.size() == 1 && m.data[0] > 0 && m.args.size() >= 2 &&
            std::count_if(m.args.begin(), m.args.end(), [&](Id f) {
              return !(a.nodes[f].facts.finite && a.nodes[f].facts.pos); }) <= 1 &&
            std::all_of(m.args.begin(), m.args.end(), [&](Id f) { return a.nodes[f].facts.finite; })) {
          const auto factors = m.args; const Q d = m.data[0];
          if (d != 1) terms.push_back(a.logarithm(a.constant(d), true));
          for (Id f : factors) terms.push_back(log2_term(f));
          ++a.stats["linear_reduction_case_logarithm_products"];
        } else terms.push_back(log2_term(value));
        const Id base2 = a.add(terms);
        result = n_op == "log2" ? base2 : a.mul({a.natural_log2(), base2});
      }
    }
    // 1/(d f_1 ... f_k) = (1/d) (1/f_1) ... (1/f_k) for finite factors when the case fixes the
    // product nonzero (X = 0 false, X <= 0 false or X >= 0 false): then every factor is nonzero.
    if (reciprocal) {
      const Id product = rebuilt[0];
      const auto &m = a.nodes[product];
      if (m.op == "mul" && m.data.size() == 1 && m.data[0] != 0 && m.args.size() >= 2 &&
          std::all_of(m.args.begin(), m.args.end(), [&](Id f) { return a.nodes[f].facts.finite; }) &&
          (case_truth(resolve(a.compare("eq", product, a.zero), fixed, memo, open, work), fixed) == 0 ||
           case_truth(resolve(a.compare("le", product, a.zero), fixed, memo, open, work), fixed) == 0 ||
           case_truth(resolve(a.compare("ge", product, a.zero), fixed, memo, open, work), fixed) == 0)) {
        const auto factors = m.args; const Q d = m.data[0];
        std::vector<Id> inverse{a.constant(Q(1) / d)};
        for (Id f : factors) inverse.push_back(a.recip(f));
        result = a.mul(inverse);
        ++a.stats["linear_reduction_case_reciprocal_products"];
      }
    }
    // Chain mode (finite_cases): the reciprocal of a value v the case fixes positive ([v <= 0] false),
    // v = d f_1 ... f_k (d > 0) or a single factor, every factor finite and all but one positive by their
    // facts (so that one, f, is positive in this case too): 1/v = (1/d) (1/f_1) ... (1/f_k), and 1/f is
    // 2^(-log2 u) for f's positive view u = select([f <= 0], 1, f) (u = f here, log2 u finite by
    // Arena::positive_view). Exact in this case.
    if (reciprocal && finite_cases && a.nodes[rebuilt[0]].facts.finite &&
        case_truth(resolve(a.compare("le", rebuilt[0], a.zero), fixed, memo, open, work), fixed) == 0) {
      const Id value = rebuilt[0];
      std::vector<Id> factors{value}; Q d(1);
      if (a.nodes[value].op == "mul" && a.nodes[value].data.size() == 1 && a.nodes[value].data[0] > 0) {
        factors = a.nodes[value].args; d = a.nodes[value].data[0];
      }
      const auto positive = [&](Id f) { return a.nodes[f].facts.finite && a.nodes[f].facts.pos; };
      if (std::all_of(factors.begin(), factors.end(), [&](Id f) { return a.nodes[f].facts.finite; }) &&
          std::count_if(factors.begin(), factors.end(), [&](Id f) { return !positive(f); }) == 1) {
        std::vector<Id> inverse{a.constant(Q(1) / d)};
        for (Id f : factors)
          inverse.push_back(positive(f) ? a.recip(f)
                            : a.exp2(a.neg(a.logarithm(a.select(a.compare("le", f, a.zero), a.one, f), true))));
        result = a.mul(inverse);
        ++a.stats["linear_reduction_chain_positive_reciprocals"];
      }
    }
  }
  // Chain lemmas only (finite_cases): a value the case fixes finite (case_finite, by its original or
  // its rebuilt node) is its finite view, equal to it on every input of the case. If it resolves to
  // -inf instead, no input is in the case (EmptyCase).
  if (finite_cases && (case_finite.contains(x) || case_finite.contains(result))) {
    if (a.neginf(result)) throw EmptyCase{};
    result = finite_view(result);
  }
  memo.emplace(x, result);
  return result;
}

// The nodes a case fixes finite, each finite or -inf by its facts and not finite by them. Each
// conclusion holds on every input of the case:
//   - [v != -inf] or [v > -inf] (or [-inf < v]) true, [v == -inf] or [v <= -inf] (or [-inf >= v])
//     false, v finite or -inf: v is not -inf, so finite;
//   - [X < k] or [X <= k] false, [X >= k] or [X > k] true, with k a finite constant and
//     X = d_1 e_1 + ... + d_n e_n (add or add_raw) where every e_j is finite, or finite or -inf with
//     d_j > 0: X is then finite or -inf (no NaN, no +inf), X >= k (or X > k) excludes X = -inf, so
//     no term is -inf: every e_j is finite;
//   - a node fixed finite that is q * y with q > 0 a constant and y finite or -inf: y is finite.
// O(fixed + their sums' terms).
inline auto LinearReductionPattern::collect_case_finite(const CaseFix &fixed) -> void {
  case_finite.clear();
  std::vector<Id> found;
  const auto candidate = [&](Id v) {
    if (!a.nodes[v].facts.finite && a.nodes[v].facts.finite_or_negative_infinity() && !a.neginf(v)) found.push_back(v);
  };
  for (const auto &[c, truth] : fixed) {
    const auto &n = a.nodes[c];
    if (n.op != "compare" || n.args.size() != 2 || !n.data.empty()) continue;
    const std::string tag = n.tag;
    const Id left = n.args[0], right = n.args[1];
    if (a.neginf(left) || a.neginf(right)) {
      const bool on_right = a.neginf(right);
      const bool excluded = (tag == "ne" && truth) || (tag == "eq" && !truth) ||
          (on_right && ((tag == "gt" && truth) || (tag == "le" && !truth))) ||
          (!on_right && ((tag == "lt" && truth) || (tag == "ge" && !truth)));
      if (excluded) candidate(on_right ? left : right);
      continue;
    }
    // A proved Sum X = C * T (C finite positive) that the case fixes nonzero: T != 0, so a value v of
    // support_finite for X is finite (prepare_support_finite).
    if (!support_finite.empty() && right == a.zero &&
        ((tag == "le" && !truth) || (tag == "eq" && !truth) || (tag == "ne" && truth) || (tag == "gt" && truth)))
      for (const auto &[x, v] : support_finite) if (x == left) candidate(v);
    const bool below = (truth && (tag == "ge" || tag == "gt")) || (!truth && (tag == "lt" || tag == "le"));
    const auto k = a.value(right);
    const auto &s = a.nodes[left];
    if (!below || !k || (s.op != "add" && s.op != "add_raw") || s.data.size() != s.args.size()) continue;
    bool terms = true;
    for (size_t j = 0; j < s.args.size() && terms; ++j) {
      const auto &f = a.nodes[s.args[j]].facts;
      terms = f.finite || (f.finite_or_negative_infinity() && s.data[j] > 0);
    }
    if (!terms) continue;
    const std::vector<Id> operands = s.args;
    for (Id e : operands) candidate(e);
  }
  for (size_t i = 0; i < found.size(); ++i) {
    if (!case_finite.insert(found[i]).second) continue;
    const auto &m = a.nodes[found[i]];
    if (m.op == "mul" && m.data.size() == 1 && m.data[0] > 0 && m.args.size() == 1) candidate(m.args[0]);
  }
}

// The finite view of v in a case where v is not -inf: Select([v == -inf], 0, v), which is v there and
// finite by the Arena's facts (its arm v is taken only where v != -inf). v itself when v is finite
// already, may be NaN, +inf or undefined, or the view is not finite by its facts.
inline auto LinearReductionPattern::finite_view(Id v) -> Id {
  if (a.nodes[v].facts.finite || !a.nodes[v].facts.finite_or_negative_infinity()) return v;
  const Id view = a.select(a.compare("eq", v, a.infinity(-1)), a.zero, v);
  if (!a.nodes[view].facts.finite) return v;
  ++a.stats["linear_reduction_chain_finite_views"];
  return view;
}

// The open condition nearest to a resolved root (fewest operand edges; Folds and Calls are not
// entered, as resolve() does not enter them), ties in the order met; the first open condition when
// none is reached. O(nodes reached), at most case_rebuild_limit.
inline auto LinearReductionPattern::nearest_open(const std::vector<Id> &roots, const std::vector<Id> &open) -> Id {
  std::unordered_map<Id, size_t> rank;
  for (size_t j = 0; j < open.size(); ++j) rank.emplace(open[j], j);
  std::unordered_set<Id> seen;
  std::vector<Id> level = roots, next;
  size_t work = 0;
  while (!level.empty()) {
    Id best = invalid;
    for (Id x : level)
      if (auto it = rank.find(x); it != rank.end() && (best == invalid || it->second < rank.at(best))) best = x;
    if (best != invalid) return best;
    for (Id x : level) {
      if (!seen.insert(x).second) continue;
      if (++work > case_rebuild_limit) return open.front();
      const auto &n = a.nodes[x];
      if (n.op == "fold" || n.op == "call") continue;
      next.insert(next.end(), n.args.begin(), n.args.end());
    }
    level.swap(next); next.clear();
  }
  return open.front();
}

} // namespace rg
