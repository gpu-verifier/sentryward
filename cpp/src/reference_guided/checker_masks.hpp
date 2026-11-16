#pragma once
#include "checker.hpp"

namespace rg {

inline auto Checker::admit_payload_coordinates(Id expected,Id actual) -> bool {
  if(expected==actual)return true;
  // The reduction consumer already owns bounded factor decomposition and
  // the later binder-independence check. Do not ask a whole-body equality
  // query to discard an extra ordinary input factor. This is exact root
  // factor containment, not division and not a recovered coefficient.
  const auto &e=a.nodes[expected], &v=a.nodes[actual];
  if(e.op=="mul" && v.op=="mul" && e.data==v.data &&
     e.args.size()<v.args.size() && v.args.size()<=16) {
    std::vector<bool> used(v.args.size(),false);bool contained=true;
    for(Id factor:e.args) {
      size_t j=0;while(j<v.args.size() && (used[j] || v.args[j]!=factor))++j;
      if(j==v.args.size()){contained=false;break;}
      used[j]=true;
    }
    if(contained) {
      bool ordinary=true;
      for(size_t j=0;j<v.args.size();++j)if(!used[j]) {
        const auto &extra=a.nodes[v.args[j]];
        ordinary &= extra.facts.finite &&
          (extra.op=="read" || extra.op=="readbits" || extra.op=="packedbits");
      }
      if(ordinary){++a.stats["ep_body_factor_decomposition_admissions"];return false;}
    }
  }
  const auto &wanted=payload_admission(expected), &found=payload_admission(actual);
  if(!wanted.positive_shell || !found.positive_shell)return true;
  // A different Exp/Select/Call can require genuine algebra or delayed
  // definition binding. Never classify it by looking through its ancestry.
  for(Id opaque:found.opaque)if(!wanted.opaque.contains(opaque))return true;
  for(const auto &[source,reads]:found.reads) {
    // An unrelated outside input factor cannot hide conflicting coordinates
    // of a source already constrained by this candidate. It is left for the
    // existing factor-decomposition consumer, not interpreted here.
    if(!wanted.reads.contains(source))continue;
    const auto &allowed=wanted.reads.at(source);
    for(Id read:reads)if(!allowed.contains(read)) {
      ++a.stats["ep_body_payload_admission_rejections"];
      return false;
    }
  }
  return true;
}

// The literal that "condition != 0 is `nonzero`" establishes: for a 0/1 condition c the literal
// (c, nonzero); for [y == 0] or [y != 0] of a 0/1 value y the literal on y itself. O(1).
inline auto Checker::mask_literal(Id condition, bool nonzero) -> std::optional<MaskLiteral> {
  const auto &c = a.nodes[condition];
  if (c.op == "compare" && c.args.size() == 2 && c.data.empty() && (c.tag == "eq" || c.tag == "ne")) {
    for (unsigned side = 0; side != 2; ++side)
      if (c.args[1 - side] == a.zero && a.boolean_node(c.args[side]))
        return MaskLiteral{c.args[side], (c.tag == "ne") == nonzero};
  }
  if (a.boolean_node(condition)) return MaskLiteral{condition, nonzero};
  return {};
}

// The truth of `c != 0` where `lit` holds: 1, 0, or -1 (open). The same literal, the literal of
// a 0/1 value whose [y == 0]/[y != 0] form c is, or (lit true) a literal implied by lit or whose
// complement lit implies (Arena::boolean_implies: a factor of a product of predicates, or a
// bounds-excluded end of an integer). Sufficient, not complete. O(factors).
inline auto Checker::decide_under(Id c, const MaskLiteral &lit) -> int {
  if (c == lit.node) return lit.truth;
  const auto lc = mask_literal(c, true);  // c != 0 exactly when lc->node == lc->truth
  if (!lc) return -1;
  if (lc->node == lit.node) return lc->truth == lit.truth;
  if (!lit.truth) return -1;
  if (a.boolean_implies(lit.node, lc->node)) return lc->truth;
  const auto &q = a.nodes[lc->node];
  if (q.op == "compare" && q.args.size() == 2 && q.data.empty() &&
      a.nodes[q.args[0]].facts.finite && a.nodes[q.args[1]].facts.finite) {
    const Id complement = a.compare(Arena::complement_comparison(q.tag), q.args[0], q.args[1]);
    if (a.boolean_implies(lit.node, complement)) return !lc->truth;
  }
  return -1;
}

inline auto Checker::resolve_under(Id x, const MaskLiteral &lit, bool coordinate, size_t &work, bool &failed, bool deep) -> Id {
  if (failed) return x;
  const auto key = std::tuple(x, lit.node, lit.truth, coordinate, deep);
  if (auto it = mask_resolution_cache.find(key); it != mask_resolution_cache.end()) return it->second;
  if (++work > mask_resolution_limit) { failed = true; return x; }
  a.check();
  const std::string op = a.nodes[x].op, tag = a.nodes[x].tag;
  const auto args = a.nodes[x].args;
  const auto data = a.nodes[x].data;
  Id result = x;
  if (x == lit.node) result = lit.truth ? a.one : a.zero;
  else if (args.empty() || op == "call" || op == "certified_state" || (op == "max" && !coordinate)) result = x;
  else if (op == "select" && args.size() == 3) {
    // deep: the condition is rebuilt as well (a test on a value the literal decides inside it, e.g. the
    // sign of a nibble of a byte that is zeroed for an invalid token); it equals the original where lit holds.
    const Id condition = deep ? resolve_under(args[0], lit, coordinate, work, failed, deep) : args[0];
    const int truth = decide_under(condition, lit);
    if (truth >= 0) result = resolve_under(args[truth ? 1 : 2], lit, coordinate, work, failed, deep);
    else {
      const Id yes = resolve_under(args[1], lit, coordinate, work, failed, deep);
      const Id no = resolve_under(args[2], lit, coordinate, work, failed, deep);
      if (condition != args[0] || yes != args[1] || no != args[2]) result = a.select(condition, yes, no);
    }
  } else if (op == "compare" && decide_under(x, lit) >= 0) {
    result = decide_under(x, lit) ? a.one : a.zero;
  } else if (op == "fold" && args.size() == 2 && data.size() == 2 && data[0].denominator() == 1 &&
             data[0] >= 0 && data[0] < Q(a.nodes.size()) &&
             a.depends(lit.node, data[0].numerator().convert_to<Id>())) {
    result = x;
  } else {
    const bool inner = coordinate || op == "read" || op == "readbits" || op == "packedbits";
    std::vector<Id> rebuilt; bool changed = false;
    for (Id y : args) { rebuilt.push_back(resolve_under(y, lit, inner, work, failed, deep)); changed |= rebuilt.back() != y; }
    if (changed && !failed) {
      if (op == "readbits" || op == "packedbits") {
        const auto &bounds = a.nodes[x].bounds;
        if (data.empty() || data[0].denominator() != 1 || data[0] <= 0 || data[0] > 64 || !bounds ||
            bounds->first != 0 || bounds->second != Q(pow2(data[0].numerator().convert_to<unsigned>()) - 1))
          failed = true;
        else result = a.intern(op, rebuilt, data, tag, {true, false, true}, bounds);
      } else if (op == "read") {
        const auto newline = tag.find('\n');
        if (newline == std::string::npos || !tag.substr(newline + 1).starts_with("real") || a.nodes[x].bounds)
          failed = true;
        else result = a.intern("read", rebuilt, data, tag, {true, false, false}, {});
      } else if (deep && op == "word" && rebuilt.size() == 1 && data.size() == 1 && data[0].denominator() == 1 &&
                 data[0] > 0 && data[0] <= 64) {
        // A word is its operand modulo 2^w: an integer in [0, 2^w - 1] for any integer operand.
        const unsigned width = data[0].numerator().convert_to<unsigned>();
        const auto v = a.value(rebuilt[0]);
        result = v && v->denominator() == 1 ? a.constant(Q(modz(v->numerator(), pow2(width))))
               : a.intern("word", rebuilt, data, tag, {a.nodes[rebuilt[0]].facts.finite, false, true},
                          std::pair(Q(0), Q(pow2(width) - 1)));
      } else {
        result = reapply(x, rebuilt, /*strict=*/true);
        if (result == invalid) { failed = true; result = x; }
      }
    }
  }
  if (!failed) mask_resolution_cache.emplace(key, result);
  return result;
}

// Rule (mask resolution): a finite product with a zero-armed factor Z = select(c, X, 0) (or
// select(c, 0, X)) equals the product in which Z's arm X and every other factor are rebuilt where
// Z's literal holds (resolve_under). Where the literal holds every rebuilt factor equals its
// original; elsewhere Z is 0 and every factor is finite (checked on the rebuilt ones), so both
// products are 0. The first such factor (in the product's order) is kept as the mask; the others
// are ordinary factors (one of them may resolve to its arm). Returns n itself when nothing changes
// or the rebuild declines. Cost: the rebuilt nodes, memoized across terms (mask_resolution_cache).
inline auto Checker::mask_resolved(Id n) -> Id {
  const auto &node = a.nodes[n];
  if (node.op != "mul" || node.data.size() != 1 || !node.facts.finite || node.args.size() > 16) return n;
  const auto factors = node.args;
  const Q scalar = node.data[0];
  size_t keeper = factors.size(); std::optional<MaskLiteral> lit;
  for (size_t j = 0; j < factors.size() && keeper == factors.size(); ++j) {
    const auto &f = a.nodes[factors[j]];
    if (f.op != "select" || f.args.size() != 3 || (f.args[1] != a.zero && f.args[2] != a.zero)) continue;
    lit = mask_literal(f.args[0], f.args[2] == a.zero);
    if (lit) keeper = j;
  }
  if (!lit) return n;
  size_t work = 0; bool failed = false;
  std::vector<Id> rebuilt{a.constant(scalar)};
  for (size_t j = 0; j < factors.size(); ++j) {
    if (j == keeper) {
      const auto s = a.nodes[factors[j]].args;
      const bool first = s[2] == a.zero;
      const Id arm = resolve_under(s[first ? 1 : 2], *lit, false, work, failed);
      rebuilt.push_back(arm == s[first ? 1 : 2] ? factors[j] : a.select(s[0], first ? arm : a.zero, first ? a.zero : arm));
    } else rebuilt.push_back(resolve_under(factors[j], *lit, false, work, failed));
    if (failed || !a.nodes[rebuilt.back()].facts.finite) return n;
  }
  const Id result = a.mul(std::move(rebuilt));
  if (result != n) ++a.stats["mask_resolved_products"];
  return a.nodes[result].facts.finite ? result : n;
}

inline auto Checker::masked_exponential_term(Id root) -> std::optional<MaskedExponentialTerm> {
  if (!a.nodes[root].facts.finite) return {};
  Id predicate = a.one;
  std::vector<Id> exponent, residual{a.one}, todo{root};
  std::optional<bool> base2;
  size_t work = 0;
  while (!todo.empty()) {
    Id id = todo.back(); todo.pop_back();
    if (++work > 24) return {};
    const auto &node = a.nodes[id];
    if (!node.facts.finite) return {};
    if (node.op == "mul" && node.data.size() == 1 && node.args.size() <= 8) {
      const auto factors = node.args;
      const Q scalar = node.data[0];
      residual.push_back(a.constant(scalar));
      todo.insert(todo.end(), factors.begin(), factors.end());
    } else if (node.op == "select" && node.args.size() == 3 &&
               (node.args[1] == a.zero || node.args[2] == a.zero)) {
      const auto branches = node.args;
      Id condition = branches[0];
      if (branches[1] == a.zero) condition = a.compare("eq", condition, a.zero);
      else if (boolean_value(condition)) {
        // A numeric condition proved 0/1-valued (e.g. a product of predicates, or 1 - P with P a
        // product of predicates) selects exactly when [c != 0]; compare() gives that predicate its
        // canonical comparison form, as a comparison condition already has. Matching only.
        const Id truth = a.compare("ne", condition, a.zero);
        if (truth != condition) ++a.stats["masked_boolean_condition_canonicalizations"];
        condition = truth;
      }
      // Two masks of one term are one predicate when equal, or equal in their local index forms (exact).
      if (predicate != a.one && !scalar.alpha.equal(predicate, condition) &&
          local_index_form(predicate) != local_index_form(condition)) return {};
      predicate = condition;
      Id value = branches[branches[1] == a.zero ? 2 : 1];
      // Only one mask layer per factor. Its numerical child is not searched
      // for another condition; otherwise this would become path expansion.
      if (a.nodes[value].op == "select") return {};
      todo.push_back(value);
    } else if ((node.op == "exp" || node.op == "exp2") && node.args.size() == 1 &&
               a.nodes[node.args[0]].facts.finite) {
      bool binary = node.op == "exp2";
      if (base2 && *base2 != binary) return {};
      base2 = binary;
      exponent.push_back(node.args[0]);
    } else residual.push_back(id);
  }
  if (!base2 || exponent.empty()) return {};
  return MaskedExponentialTerm{predicate, a.add(exponent), a.mul(residual), *base2};
}

// x with each node of `replace` substituted everywhere (DAG walk, memoized), parents rebuilt by their
// constructors (reapply) so constant operands fold; a word of a constant is that constant modulo
// 2^width, and 2^k of an integer constant |k| <= 64 is that rational. At most 4096 distinct nodes,
// else invalid. Exact (substitution).
inline auto Checker::substitute_nodes(Id x, const std::unordered_map<Id, Id> &replace) -> Id {
  std::unordered_map<Id, Id> memo;
  std::vector<std::pair<Id, bool>> todo{{x, false}};
  while (!todo.empty()) {
    a.check();
    auto [id, ready] = todo.back(); todo.pop_back();
    if (memo.contains(id)) continue;
    if (memo.size() > 4096) return invalid;
    if (auto it = replace.find(id); it != replace.end()) { memo.emplace(id, it->second); continue; }
    const auto args = a.nodes[id].args;
    if (args.empty() || a.nodes[id].op == "fold") { memo.emplace(id, id); continue; }
    if (!ready) {
      todo.emplace_back(id, true);
      for (Id y : args) if (!memo.contains(y)) todo.emplace_back(y, false);
      continue;
    }
    std::vector<Id> xs; bool changed = false;
    for (Id y : args) { xs.push_back(memo.at(y)); changed = changed || xs.back() != y; }
    Id out = id;
    if (changed) {
      const auto &n = a.nodes[id];
      const auto v = xs.size() == 1 ? a.value(xs[0]) : std::nullopt;
      if (n.op == "word" && n.data.size() == 1 && v && v->denominator() == 1)
        out = a.constant(Q(modz(v->numerator(), pow2(n.data[0].numerator().convert_to<unsigned>()))));
      else if (n.op == "exp2" && v && v->denominator() == 1 && boost::multiprecision::abs(v->numerator()) <= 64) {
        const unsigned k = boost::multiprecision::abs(v->numerator()).convert_to<unsigned>();
        out = a.constant(v->numerator() >= 0 ? Q(pow2(k)) : Q(Z(1), pow2(k)));
      } else out = reapply(id, xs);
    }
    memo.emplace(id, out);
  }
  return memo.at(x);
}

inline auto Checker::byte_cases_equal(Id x, Id y) -> bool {
  if (!a.nodes[x].facts.finite || !a.nodes[y].facts.finite) return false;
  const auto key = std::pair(x, y);
  if (auto it = byte_case_cache.find(key); it != byte_case_cache.end()) return it->second;
  const bool result = byte_cases_equal_uncached(x, y);
  byte_case_cache.emplace(key, result);
  return result;
}

inline auto Checker::byte_cases_equal_uncached(Id x, Id y) -> bool {
  Id byte = invalid;
  std::vector<Id> conditions;
  std::unordered_set<Id> seen; std::vector<Id> todo{x, y};
  while (!todo.empty()) {
    const Id id = todo.back(); todo.pop_back();
    if (!seen.insert(id).second) continue;
    if (seen.size() > 4096) return false;
    const auto &n = a.nodes[id];
    if (n.op == "readbits" && n.data.size() == 1 && n.data[0] >= 1 && n.data[0] <= 8 && n.bounds &&
        n.bounds->first >= 0 && n.bounds->second <= Q(pow2(n.data[0].numerator().convert_to<unsigned>()) - 1)) {
      if (byte != invalid && byte != id) return false;
      byte = id;
      continue;
    }
    if (n.op == "fold") continue;
    if (n.op == "select" && n.args.size() == 3 && a.nodes[n.args[0]].op == "compare" &&
        std::find(conditions.begin(), conditions.end(), n.args[0]) == conditions.end())
      conditions.push_back(n.args[0]);
    todo.insert(todo.end(), n.args.begin(), n.args.end());
  }
  if (byte == invalid) return false;
  std::vector<Id> guards;
  const auto holds_byte = [&](Id root) {  // whether `byte` occurs in root (at most 512 nodes; beyond: assume it does)
    std::unordered_set<Id> visited; std::vector<Id> stack{root};
    while (!stack.empty()) {
      const Id id = stack.back(); stack.pop_back();
      if (id == byte) return true;
      if (!visited.insert(id).second) continue;
      if (visited.size() > 512) return true;
      stack.insert(stack.end(), a.nodes[id].args.begin(), a.nodes[id].args.end());
    }
    return false;
  };
  for (Id c : conditions) if (!holds_byte(c)) guards.push_back(c);
  if (guards.size() > 2) return false;
  // Only the values the byte takes: its Arena bounds (inside [0, 2^width - 1], checked above), so no
  // substituted node is built at a value the byte never has.
  const auto &byte_bounds = *a.nodes[byte].bounds;
  const Z lo = -floorz(-byte_bounds.first.numerator(), byte_bounds.first.denominator());
  const Z hi = floorz(byte_bounds.second.numerator(), byte_bounds.second.denominator());
  if (lo > hi) return false;
  const unsigned first = lo.convert_to<unsigned>(), last = hi.convert_to<unsigned>();
  for (unsigned g = 0; g < (1u << guards.size()); ++g)
    for (unsigned v = first; v <= last; ++v) {
      std::unordered_map<Id, Id> replace{{byte, a.constant(Q(v))}};
      for (size_t j = 0; j < guards.size(); ++j) replace.emplace(guards[j], (g >> j) & 1 ? a.one : a.zero);
      const Id xv = substitute_nodes(x, replace), yv = substitute_nodes(y, replace);
      if (xv == invalid || yv == invalid) return false;
      try { if (xv != yv && !scalar.equal(xv, yv)) return false; } catch (const Unsupported &) { return false; }
    }
  ++a.stats["byte_case_equalities"];
  return true;
}

inline auto Checker::output_coordinate_guard(Id x) -> std::optional<CoordinateGuard> {
  const auto &n = a.nodes[x];
  const std::vector<Id> factors = n.op == "mul" ? n.args : std::vector<Id>{x};
  for (Id f : factors) {
    const auto &s = a.nodes[f];
    if (s.op != "select" || s.args.size() != 3) continue;
    const auto &c = a.nodes[s.args[0]];
    if (c.op != "compare" || c.args.size() != 2 || !c.data.empty()) continue;
    for (unsigned side = 0; side != 2; ++side) {
      const Id v = c.args[side];
      const auto &axis = a.nodes[v];
      const auto k = a.value(c.args[1 - side]);
      if (axis.op != "var" || !axis.tag.starts_with("output:") || !axis.bounds || !k ||
          axis.bounds->first.denominator() != 1 || axis.bounds->second.denominator() != 1) continue;
      const Z lo = axis.bounds->first.numerator(), hi = axis.bounds->second.numerator();
      const std::string op = side ? Arena::reverse_comparison(c.tag) : c.tag;
      Z first = lo, last = hi;
      if (op == "le") last = floorq(*k);
      else if (op == "lt") last = -floorq(-*k) - 1;
      else if (op == "ge") first = -floorq(-*k);
      else if (op == "gt") first = floorq(*k) + 1;
      else continue;
      first = std::max(first, lo); last = std::min(last, hi);
      if (first > last || (first == lo && last == hi)) continue;
      return CoordinateGuard{v, first, last};
    }
    // Otherwise a comparison of integer arithmetic over one output coordinate alone (no stored
    // read: e.g. [v mod 2 == 0]) with at most max_coordinate_points values: its points.
    const auto &free = a.nodes[s.args[0]].free;
    if (free.size() != 1 || !coordinate_arithmetic(s.args[0])) continue;
    const auto &axis = a.nodes[free[0]];
    if (axis.op != "var" || !axis.tag.starts_with("output:") || !axis.bounds ||
        axis.bounds->first.denominator() != 1 || axis.bounds->second.denominator() != 1 ||
        axis.bounds->second - axis.bounds->first + 1 > Q(max_coordinate_points)) continue;
    return CoordinateGuard{free[0], axis.bounds->first.numerator(), axis.bounds->second.numerator(), true};
  }
  return {};
}

// Whether x is built only from variables, constants, add, mul, idiv, imod and comparisons (no read,
// Call, Fold or other opaque leaf). At most 64 nodes. O(nodes).
inline auto Checker::coordinate_arithmetic(Id x) const -> bool {
  std::vector<Id> todo{x}; std::unordered_set<Id> seen;
  while (!todo.empty()) {
    const Id id = todo.back(); todo.pop_back();
    if (!seen.insert(id).second) continue;
    if (seen.size() > 64) return false;
    const auto &n = a.nodes[id];
    if (n.op != "var" && n.op != "const" && n.op != "add" && n.op != "mul" && n.op != "idiv" &&
        n.op != "imod" && n.op != "compare") return false;
    todo.insert(todo.end(), n.args.begin(), n.args.end());
  }
  return true;
}

inline auto Checker::masked_body_coefficient_direct(Id expected, Id actual, bool allow_proved_mask) -> std::optional<Id> {
  auto left = masked_exponential_term(expected), right = masked_exponential_term(actual);
  if (!left || !right || left->base2 != right->base2 ||
      (!scalar.alpha.equal(left->predicate, right->predicate) &&
       local_index_form(left->predicate) != local_index_form(right->predicate))) return {};
  if (left->predicate == a.one && !allow_proved_mask) return {};
  // If a reference exponent has a reduction Call absent from the actual exponent,
  // normalize the complete actual Sum again with exact contribution coverage.
  // Under a term's mask, rebuild its exponent and residual using resolve_under:
  // outside the mask the term is zero; inside, the rebuilt expressions agree.
  // A declined rebuild keeps the original expression.
  if (exponential_mode && right->predicate != a.one && a.nodes[right->exponent].facts.finite)
    if (const auto lit = mask_literal(right->predicate, true)) {
      size_t work = 0; bool failed = false;
      const Id rebuilt = resolve_under(right->exponent, *lit, false, work, failed, /*deep=*/true);
      if (!failed && rebuilt != right->exponent && a.nodes[rebuilt].facts.finite) {
        right->exponent = rebuilt;
        ++a.stats["masked_exponent_mask_resolutions"];
      }
      // Rule (residual under its mask): likewise for the term's residual factor (a stored value zeroed
      // for an invalid token where the predicate fails). Exact for the masked term. A declined rebuild
      // keeps the residual.
      size_t residual_work = 0; bool residual_failed = false;
      const Id residual = resolve_under(right->residual, *lit, false, residual_work, residual_failed, /*deep=*/true);
      if (!residual_failed && residual != right->residual && a.nodes[residual].facts.finite) {
        right->residual = residual;
        ++a.stats["masked_residual_mask_resolutions"];
      }
    }
  if (exponential_mode && a.nodes[left->exponent].facts.finite && a.nodes[right->exponent].facts.finite) {
    const auto right_terms = a.terms(right->exponent);
    for (const auto &[term, scale] : a.terms(left->exponent)) {
      if (scale == 0 || a.nodes[term].op != "call" || !exponent_closure ||
          std::any_of(right_terms.begin(), right_terms.end(), [&](const auto &t) { return t.first == term; }))
        continue;
      const auto &call_node = a.nodes[term];
      if (call_node.data.size() != 1 || call_node.data[0].denominator() != 1 || call_node.data[0] < 0 ||
          call_node.data[0] >= Q(specs.size())) continue;
      const Id sid = call_node.data[0].numerator().convert_to<Id>();
      const std::vector<Id> parameters = call_node.args;
      if (const auto closed = exponent_closure(right->exponent, sid, parameters);
          closed && a.nodes[*closed].facts.finite) {
        right->exponent = *closed;
        ++a.stats["masked_exponent_closures"];
      }
      break;
    }
  }
  // Match the actual reference exponent before constructing a scale. The
  // tautology exp(T)=exp(S)*exp(T-S) is not a useful reduction match: choosing
  // S from a different key would manufacture one unique compensation factor
  // per anchor, even when both indices are constants. Reuse the ordinary
  // exponent match's local Add/coefficient contract instead. Numerical
  // ancestors remain opaque identities; there is no candidate key search.
  constexpr size_t maximum_exponent_terms = 32;
  auto bounded_shell = [&](Id exponent) {
    const auto &shell = a.nodes[exponent];
    return (shell.op != "add" && shell.op != "mul") ||
           shell.args.size() <= maximum_exponent_terms;
  };
  if (!bounded_shell(left->exponent) || !bounded_shell(right->exponent)) return {};
  auto expected_terms = a.terms(left->exponent), actual_terms = a.terms(right->exponent);
  if (expected_terms.size() > maximum_exponent_terms || actual_terms.size() > maximum_exponent_terms)
    return {};
  std::map<Id, Q> remaining, expected_by_identity;
  for (const auto &[term, scale] : actual_terms)
    remaining[scalar.alpha.representative(term)] += scale;
  for (const auto &[term, scale] : expected_terms)
    expected_by_identity[scalar.alpha.representative(term)] += scale;
  bool matched_anchor = false;
  for (const auto &[term, scale] : expected_by_identity) {
    if (scale == 0) continue;
    if (term == a.one) { remaining[term] -= scale; continue; }
    auto found = remaining.find(term);
    if (found == remaining.end() || found->second != scale) {
      ++a.stats["masked_exponential_unmatched_score_terms"];
      return {};
    }
    matched_anchor = true;
    remaining.erase(found);
  }
  if (!matched_anchor) return {};
  if (left->residual != right->residual &&
      (!exponential_mode || !(scalar.equal(left->residual, right->residual) ||
                              byte_cases_equal(left->residual, right->residual)))) {
    // Rule (read alignment of the residuals, align_reads): the reference residual's reads that the
    // actual residual holds under another spelling of their coordinates are its reads (a copy for this
    // comparison only); then the same two comparisons.
    if (!exponential_mode) return {};
    const Id aligned = align_reads(left->residual, right->residual);
    if (aligned == left->residual || !(scalar.equal(aligned, right->residual) || byte_cases_equal(aligned, right->residual)))
      return {};
    ++a.stats["masked_residual_read_alignments"];
  }
  std::vector<Id> offsets;
  for (const auto &[term, scale] : remaining)
    if (scale != 0) offsets.push_back(a.mul({a.constant(scale), term}));
  Id difference = a.add(std::move(offsets));
  if (!a.nodes[difference].facts.finite) return {};
  Id coefficient = canonical_coefficient(a.exponential(difference, left->base2));
  if (!a.nodes[coefficient].facts.finite || !a.nodes[coefficient].facts.pos) return {};
  ++a.stats["masked_exponential_coefficient_proofs"];
  return coefficient;
}

} // namespace rg
