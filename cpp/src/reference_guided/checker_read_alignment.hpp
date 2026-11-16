#pragma once
#include "checker.hpp"

namespace rg {

inline auto Checker::local_shape(const Arena &arena, Id root, std::unordered_map<Id, bool> &cache) -> bool {
  if (auto it = cache.find(root); it != cache.end()) return it->second;
  std::vector<Id> todo{root}; std::unordered_set<Id> seen{root}; bool found = false;
  while (!todo.empty() && !found) {
    if (seen.size() > 64) { found = true; break; }
    const auto &n = arena.nodes[todo.back()]; todo.pop_back();
    if (n.op == "idiv" || n.op == "imod" || n.op == "select" || n.op == "compare" || n.op == "word" ||
        n.op == "derived_coordinate") { found = true; break; }
    if (n.op == "read" || n.op == "readbits" || n.op == "packedbits" || n.op == "fold" || n.op == "call") continue;
    for (Id x : n.args) if (seen.insert(x).second) todo.push_back(x);
  }
  cache.emplace(root, found);
  return found;
}

inline auto Checker::local_coordinate_match(Id pattern, Id actual, const std::vector<Id> &cap) -> bool {
  const auto &p = patterns.nodes[pattern];
  if (p.op != "add" && p.op != "mul" && p.op != "idiv" && p.op != "imod") return false;
  if (!local_shape(patterns, pattern, pattern_local_shape_cache) && !local_shape(a, actual, local_shape_cache)) return false;
  const auto &t = a.nodes[actual];
  if (!t.facts.finite || !a.known_bits(actual).integer) return false;
  // Uncached instantiation (build_max): coordinate patterns are small, and their captures vary.
  const Id expected = instantiate_body(pattern, cap, invalid, /*build_max=*/true);
  if (expected == invalid || !a.nodes[expected].facts.finite) return false;
  if (expected == actual) return true;
  if (local_index_form(expected) != local_index_form(actual)) return false;
  ++a.stats["local_coordinate_matches"];
  return true;
}

inline auto Checker::local_quotient(Id n, const Z &d, size_t &visits) -> Id {
  Id q = a.idiv(n, d);
  for (unsigned round = 0; round < 8; ++round) {
    if (++visits > local_index_limit_now) throw LocalIndexCut{};
    const auto &qn = a.nodes[q];
    if (qn.op != "idiv" || qn.args.size() != 1 || qn.data.size() != 1 || qn.data[0].denominator() != 1) return q;
    const Id dividend = qn.args[0]; const Z dd = qn.data[0].numerator();
    const auto ts = a.terms(dividend);
    std::optional<size_t> chosen; bool ambiguous = false;
    for (size_t i = 0; i < ts.size() && !ambiguous; ++i) {
      const auto &[t, c] = ts[i];
      if (t == a.one || c.denominator() != 1) continue;
      const Z ai = boost::multiprecision::abs(c.numerator());
      if (ai <= 1 || modz(dd, ai) != 0 || !a.nodes[t].facts.finite || !a.known_bits(t).integer) continue;
      Q lo(0), hi(0); bool bounded = true;
      for (size_t j = 0; j < ts.size() && bounded; ++j) {
        if (j == i) continue;
        const auto &[u, k] = ts[j];
        if (k.denominator() != 1) { bounded = false; break; }
        if (u == a.one) { lo += k; hi += k; continue; }
        const auto &b = a.nodes[u].bounds;
        if (!a.nodes[u].facts.finite || !b || !a.known_bits(u).integer) { bounded = false; break; }
        lo += k > 0 ? k * b->first : k * b->second;
        hi += k > 0 ? k * b->second : k * b->first;
      }
      if (!bounded || lo < 0 || hi > Q(ai - 1)) continue;
      if (chosen) ambiguous = true; else chosen = i;
    }
    if (!chosen || ambiguous) return q;
    const auto &[t, c] = ts[*chosen];
    const Z ai = boost::multiprecision::abs(c.numerator());
    ++a.stats["local_index_offset_quotients"];
    q = a.idiv(a.mul({a.constant(c / Q(ai)), t}), dd / ai);
  }
  return q;
}

inline auto Checker::local_sum(Id sum, size_t &visits) -> Id {
  for (unsigned round = 0; round < 8; ++round) {
    const auto ts = a.terms(sum);
    bool changed = false;
    for (const auto &[t, c] : ts) {
      const auto &n = a.nodes[t];
      if (n.op != "imod" || n.args.size() != 1 || n.data.size() != 1 || n.data[0].denominator() != 1 || n.data[0] <= 0)
        continue;
      const Z m = n.data[0].numerator();
      const Id y = n.args[0];
      const Id q = local_quotient(y, m, visits);
      const Q want = c * Q(m);
      if (!std::any_of(ts.begin(), ts.end(), [&](const auto &p) { return p.first == q && p.second == want; })) continue;
      sum = a.add({sum, a.mul({a.constant(-want), q}), a.mul({a.constant(-c), t}), a.mul({a.constant(c), y})});
      ++a.stats["local_index_recombinations"];
      changed = true;
      break;
    }
    if (!changed) break;
  }
  return sum;
}

inline auto Checker::flat_maxima(Id x) -> Id {
  std::unordered_map<Id, Id> memo;
  std::vector<std::pair<Id, bool>> todo{{x, false}};
  while (!todo.empty()) {
    a.check();
    auto [id, ready] = todo.back(); todo.pop_back();
    if (memo.contains(id)) continue;
    if (memo.size() > flat_maxima_limit) return x;
    const auto &n = a.nodes[id];
    if (n.op == "max") {
      const Id f = local_index_form(id, size_t(1) << 16);
      memo.emplace(id, f);
      continue;
    }
    const auto args = n.args;
    if (args.empty() || n.op == "fold" || n.op == "call" || n.op == "certified_state" || n.op == "read" ||
        n.op == "readbits" || n.op == "packedbits") { memo.emplace(id, id); continue; }
    if (!ready) {
      todo.emplace_back(id, true);
      for (Id y : args) if (!memo.contains(y)) todo.emplace_back(y, false);
      continue;
    }
    std::vector<Id> xs; bool changed = false;
    for (Id y : args) { xs.push_back(memo.at(y)); changed |= xs.back() != y; }
    Id out = id;
    if (changed) {
      try { out = reapply(id, xs); } catch (const Unsupported &) { return x; }
      if (out == invalid) return x;
    }
    memo.emplace(id, out);
  }
  return memo.at(x);
}

// An extended sum (add_raw) with its (operand, coefficient) pairs in Id order: the same value (a sum
// is order-free; add_raw keeps construction order). Other nodes unchanged. O(operands log operands).
inline auto Checker::sorted_extended_sum(Id x) -> Id {
  const auto &n = a.nodes[x];
  if (n.op != "add_raw" || n.args.size() != n.data.size() || n.args.size() < 2) return x;
  std::vector<std::pair<Id, Q>> terms;
  for (size_t i = 0; i < n.args.size(); ++i) terms.emplace_back(n.args[i], n.data[i]);
  if (std::is_sorted(terms.begin(), terms.end(), [](const auto &l, const auto &r) { return l.first < r.first; })) return x;
  std::sort(terms.begin(), terms.end(), [](const auto &l, const auto &r) { return l.first < r.first; });
  std::vector<Id> args; std::vector<Q> data;
  for (auto &[t, c] : terms) { args.push_back(t); data.push_back(c); }
  const Facts f = n.facts; const Bounds b = n.bounds; const std::string tag = n.tag;
  ++a.stats["local_index_sorted_extended_sums"];
  return a.intern("add_raw", std::move(args), std::move(data), tag, f, b);
}

// 1 when x is select(c, -inf, u), 2 when select(c, u, -inf) (u not -inf), else -1.
inline auto Checker::infinity_arm(Id x) const -> int {
  const auto &n = a.nodes[x];
  if (n.op != "select" || n.args.size() != 3) return -1;
  const bool t = a.neginf(n.args[1]), e = a.neginf(n.args[2]);
  return t == e ? -1 : t ? 1 : 2;
}

// A Select with one -inf arm (a masked score): its finite arm rebuilt where the arm is taken
// (resolve_under on the condition's literal: a value zeroed for an invalid token, inside its Sums,
// is the stored value there). Exact: the arm is used only where it is taken. Declines (nullopt) when
// nothing changes or the rebuild fails.
inline auto Checker::masked_infinity_select(Id c, Id then_arm, Id else_arm) -> std::optional<Id> {
  const bool t = a.neginf(then_arm), e = a.neginf(else_arm);
  if (t == e) return {};
  const Id arm = t ? else_arm : then_arm;
  const auto lit = mask_literal(c, /*nonzero=*/!t);
  if (!lit) return {};
  size_t work = 0; bool failed = false;
  const Id resolved = resolve_under(arm, *lit, false, work, failed);
  if (failed || resolved == arm) return {};
  ++a.stats["local_index_masked_arms"];
  return t ? a.select(c, then_arm, resolved) : a.select(c, resolved, else_arm);
}

inline auto Checker::local_index_form(Id root, size_t limit) -> Id {
  if (auto it = local_index_cache.find(root); it != local_index_cache.end()) return it->second;
  size_t visits = 0;
  struct LimitScope { size_t &slot, saved; LimitScope(size_t &s, size_t v) : slot(s), saved(s) { slot = v; } ~LimitScope() { slot = saved; } } limit_scope(local_index_limit_now, limit);
  std::function<Id(Id)> form = [&](Id x) -> Id {
    if (auto it = local_index_cache.find(x); it != local_index_cache.end()) return it->second;
    if (++visits > limit) throw LocalIndexCut{};
    a.check();
    const std::string op = a.nodes[x].op, tag = a.nodes[x].tag;
    const auto args = a.nodes[x].args;
    const auto data = a.nodes[x].data;
    Id out = x;
    // A maximum: the flat maximum of its operands, through nested maxima and positive multiples of
    // maxima (max(max(a, b), c) = max(a, b, c); k max(a, b) = max(k a, k b) for k > 0; both exact on
    // the extended reals, and order-free). The operands that are not maxima stay as they are.
    if (op == "max") {
      std::vector<Id> leaves; std::unordered_set<Id> seen_leaves;
      std::set<std::pair<Id, Q>> seen_maxima;  // a shared operand is collected once per multiple
      std::function<void(Id, Q)> collect = [&](Id y, Q k) {
        if (++visits > limit) throw LocalIndexCut{};
        const auto &n = a.nodes[y];
        if ((n.op == "max" || n.op == "mul") && !seen_maxima.emplace(y, k).second) return;
        if (n.op == "max") { const auto ys = n.args; for (Id z : ys) collect(z, k); return; }
        if (n.op == "mul" && n.data.size() == 1 && n.data[0] > 0 && n.args.size() == 1 && a.nodes[n.args[0]].op == "max") {
          collect(n.args[0], k * n.data[0]); return;
        }
        // An extended sum (add_raw) keeps its operands in construction order; its value does not depend
        // on that order, so a leaf's extended sum is taken with its operands sorted (also under a scale).
        Q scale = k; Id base = y;
        if (n.op == "mul" && n.data.size() == 1 && n.args.size() == 1) { scale = k * n.data[0]; base = n.args[0]; }
        base = sorted_extended_sum(base);
        const Id leaf = scale == 1 ? base : a.mul({a.constant(scale), base});
        if (seen_leaves.insert(leaf).second) leaves.push_back(leaf);
      };
      for (Id y : args) collect(y, Q(1));
      out = a.maximum(std::move(leaves));
      if (out != x) ++a.stats["local_index_flat_maxima"];
      local_index_cache.emplace(x, out);
      return out;
    }
    // Reads (and nodes that bind or defer) are leaves: a local form never builds a read node.
    if (!(args.empty() || op == "fold" || op == "call" || op == "certified_state" || op == "read" ||
          op == "readbits" || op == "packedbits")) {
      std::vector<Id> xs; bool changed = false;
      for (Id y : args) { xs.push_back(form(y)); changed |= xs.back() != y; }
      if (op == "select" && xs.size() == 3) {
        const auto &c = a.nodes[xs[0]];
        Id kept = invalid;
        if (c.op == "compare" && c.args.size() == 2 && (c.tag == "eq" || c.tag == "ne")) {
          const Id u = c.args[0], v = c.args[1];
          const Id same = c.tag == "eq" ? xs[2] : xs[1], other = c.tag == "eq" ? xs[1] : xs[2];
          if ((same == u && other == v) || (same == v && other == u)) kept = same;
        }
        if (kept != invalid) { out = kept; ++a.stats["local_index_sentinel_selects"]; }
        else if (const auto masked = masked_infinity_select(xs[0], xs[1], xs[2])) {
          out = *masked;
        } else if (changed) out = a.select(xs[0], xs[1], xs[2]);
      } else if ((op == "add" || op == "add_raw") && xs.size() == data.size() && xs.size() >= 2 &&
                 [&] { size_t m = 0; for (Id y : xs) m += infinity_arm(y) >= 0; return m == 1; }()) {
        // One masked term select(c, -inf, u) with a positive coefficient and finite other terms:
        // select(c, -inf, u) + r = select(c, -inf, u + r) (both are -inf where c selects -inf). Exact.
        size_t j = 0; while (infinity_arm(xs[j]) < 0) ++j;
        bool finite_rest = data[j] > 0;
        for (size_t i = 0; i < xs.size() && finite_rest; ++i) if (i != j) finite_rest = a.nodes[xs[i]].facts.finite;
        if (finite_rest) {
          const auto &sel = a.nodes[xs[j]];
          const int arm = infinity_arm(xs[j]);  // 1: then-arm is -inf, 2: else-arm is -inf
          const Id finite_arm = sel.args[arm == 1 ? 2 : 1], condition = sel.args[0];
          std::vector<Id> terms{a.mul({a.constant(data[j]), finite_arm})};
          for (size_t i = 0; i < xs.size(); ++i) if (i != j) terms.push_back(a.mul({a.constant(data[i]), xs[i]}));
          const Id sum = a.add(std::move(terms));
          const auto masked = arm == 1 ? masked_infinity_select(condition, a.infinity(-1), sum)
                                       : masked_infinity_select(condition, sum, a.infinity(-1));
          out = masked ? *masked : (arm == 1 ? a.select(condition, a.infinity(-1), sum) : a.select(condition, sum, a.infinity(-1)));
          ++a.stats["local_index_masked_sums"];
        } else if (changed) out = reapply(x, xs);
        if (out == invalid) out = x;
      } else if (op == "compare" && xs.size() == 2) {
        std::string t = tag; Id u = xs[0], v = xs[1];
        const auto vu = a.value(u), vv = a.value(v);
        if ((vu || vv) && !(vu && vv)) {
          const Id y = vu ? v : u; const Q k = vu ? *vu : *vv;
          const std::string q = vu ? Arena::reverse_comparison(t) : t;
          const auto &yb = a.nodes[y].bounds;
          if (a.nodes[y].facts.finite && yb && ((q == "le" && k == yb->first) || (q == "ge" && k == yb->second))) {
            t = "eq"; u = y; v = a.constant(k); changed = true; ++a.stats["local_index_bound_comparisons"];
          }
        }
        if (changed) out = a.compare(t, u, v);
      } else if (op == "idiv" && xs.size() == 1 && data.size() == 1 && data[0].denominator() == 1 && data[0] > 0) {
        out = local_quotient(xs[0], data[0].numerator(), visits);
      } else if (op == "imod" && xs.size() == 1 && data.size() == 1 && data[0].denominator() == 1 && data[0] > 0) {
        out = changed ? a.imod(xs[0], data[0].numerator()) : x;
      } else if (op == "add" && xs.size() == data.size()) {
        std::vector<Id> terms;
        for (size_t j = 0; j < xs.size(); ++j) terms.push_back(a.mul({a.constant(data[j]), xs[j]}));
        out = local_sum(changed ? a.add(std::move(terms)) : x, visits);
      } else if (op == "word" && xs.size() == 1 && data.size() == 1 && data[0].denominator() == 1 && data[0] > 0 && data[0] <= 64) {
        const unsigned width = data[0].numerator().convert_to<unsigned>();
        const auto v = a.value(xs[0]);
        out = v && v->denominator() == 1 ? a.constant(Q(modz(v->numerator(), pow2(width))))
            : changed ? a.intern("word", xs, data, tag, a.nodes[x].facts, a.nodes[x].bounds) : x;
      } else if (changed) {
        out = reapply(x, xs);
        if (out == invalid) out = x;
      }
    }
    local_index_cache.emplace(x, out);
    return out;
  };
  try { return form(root); }
  catch (const LocalIndexCut &) { ++a.stats["local_index_cuts"]; return root; }
  catch (const Unsupported &) { ++a.stats["local_index_cuts"]; return root; }
}

// The reads in the scope of `root`, as the exact target read index walks it (target_scope_contains_read):
// every node except Fold bodies (a Fold's initial value only), Calls, certified states and the reads'
// own coordinates. O(nodes of that scope).
inline auto Checker::scope_reads(Id root) -> std::vector<Id> {
  std::vector<Id> reads, todo{root}; std::unordered_set<Id> seen{root};
  size_t visited = 0;
  while (!todo.empty()) {
    if ((++visited & 1023) == 0) a.check();
    const Id id = todo.back(); todo.pop_back();
    const auto &n = a.nodes[id];
    if (n.op == "read" || n.op == "readbits" || n.op == "packedbits") { reads.push_back(id); continue; }
    if (n.op == "slot" || n.op == "var" || n.op == "call" || n.op == "certified_state") continue;
    const size_t children = n.op == "fold" ? std::min<size_t>(1, n.args.size()) : n.args.size();
    for (size_t j = 0; j < children; ++j) if (seen.insert(n.args[j]).second) todo.push_back(n.args[j]);
  }
  return reads;
}

inline auto Checker::local_twin_at(Id target, const ReadKind &kind, const std::vector<Id> &coords) -> Id {
  auto &tr = target_reads(target);
  auto group = tr.by_coordinates.find(kind);
  if (group == tr.by_coordinates.end()) {
    std::vector<Id> same;
    for (Id t : tr.reads) {
      const auto &n = a.nodes[t];
      if (n.op == std::get<0>(kind) && n.tag == std::get<1>(kind) && n.data == std::get<2>(kind)) same.push_back(t);
    }
    std::sort(same.begin(), same.end());
    std::map<std::vector<Id>, Id> forms;
    for (Id t : same) {
      std::vector<Id> f;
      for (Id c : std::vector<Id>(a.nodes[t].args)) f.push_back(local_index_form(c));
      auto [slot, inserted] = forms.emplace(std::move(f), t);
      if (!inserted && slot->second != t) slot->second = invalid;
    }
    group = target_reads(target).by_coordinates.emplace(kind, std::move(forms)).first;
  }
  if (group->second.empty()) return invalid;
  std::vector<Id> want;
  for (Id c : coords) want.push_back(local_index_form(c));
  const auto hit = group->second.find(want);
  if (hit == group->second.end() || hit->second == invalid) return invalid;
  ++a.stats["local_read_twin_hits"];
  return hit->second;
}

// The target read at the storage element of an existing read node: `read` itself when the target
// holds it; otherwise, for a read whose coordinates the local identities change, local_twin_at
// (search order: a read spelled in local form is met only by its own node).
inline auto Checker::local_read_twin(Id target, Id read) -> Id {
  if (target_reads(target).reads.contains(read)) return read;
  const auto &r = a.nodes[read];
  const std::vector<Id> coords = r.args;
  // Coordinates without a node a local identity acts on are their own local forms (local_shape).
  bool canonical = true;
  for (Id c : coords) if (local_shape(a, c, local_shape_cache) && local_index_form(c) != c) { canonical = false; break; }
  if (canonical) return invalid;
  ++a.stats["local_read_twin_searches"];
  return local_twin_at(target, ReadKind{r.op, r.tag, r.data}, coords);
}

inline auto Checker::align_reads(Id expected, Id target) -> Id {
  if (expected == invalid) return expected;
  if (auto it = align_cache.find({expected, target}); it != align_cache.end()) return it->second;
  const Id aligned = align_reads_uncached(expected, target);
  align_cache.emplace(std::pair(expected, target), aligned);
  return aligned;
}

inline auto Checker::align_reads_uncached(Id expected, Id target) -> Id {
  std::unordered_map<Id, Id> replace;
  for (Id id : scope_reads(expected)) {
    const Id twin = local_read_twin(target, id);
    if (twin != invalid && twin != id) replace.emplace(id, twin);
  }
  if (replace.empty()) return expected;
  const Id aligned = substitute_nodes(expected, replace);
  if (aligned == invalid) return expected;
  ++a.stats["aligned_body_reads"];
  return aligned;
}

inline auto Checker::plain_read_anchor(Id pattern, const std::vector<Id> *cap) const -> bool {
  const auto &p=patterns.nodes[pattern];
  if((p.op!="read" && p.op!="readbits" && p.op!="packedbits") || p.args.size()>read_anchor_coordinate_limit)
    return false;
  std::vector<Id> todo(p.args.begin(),p.args.end());
  std::unordered_set<Id> seen(todo.begin(),todo.end());
  while(!todo.empty()) {
    if(seen.size()>64)return false;
    const auto &n=patterns.nodes[todo.back()];todo.pop_back();
    if(n.op=="derived_coordinate" && cap && n.args.empty() && CoordinateRelations::valid(n.data)) {
      const size_t axis=n.data[0].numerator().convert_to<size_t>();
      if(axis<cap->size() && (*cap)[axis]!=invalid && CoordinateRelations::within_axis(a,(*cap)[axis],n.data))continue;
      return false;
    }
    if(n.op=="derived_coordinate" || n.op=="coordinate_sum" || n.op=="affine_slot" || n.op=="affine_param" ||
       n.op=="call" || n.op=="certified_state" || n.op=="fold")return false;
    for(Id x:n.args)if(seen.insert(x).second)todo.push_back(x);
  }
  return true;
}

inline auto Checker::cacheable_read_anchor(Id pattern) const -> bool {
  const auto &p=patterns.nodes[pattern];
  if((p.op!="read" && p.op!="readbits" && p.op!="packedbits") || p.args.size()>read_anchor_coordinate_limit)
    return false;
  for(Id coordinate:p.args) {
    const auto &n=patterns.nodes[coordinate];
    if(n.op=="coordinate_sum") {
      if(n.args.empty() || n.args.size()>4)return false;
      for(Id relation:n.args) {
        const auto &child=patterns.nodes[relation];
        if(child.op!="derived_coordinate" || !child.args.empty())return false;
      }
    } else if(!n.args.empty() || (n.op!="slot" && n.op!="affine_slot" && n.op!="affine_param" &&
              n.op!="derived_coordinate" && n.op!="const" && n.op!="var")) {
      return false;
    }
  }
  return true;
}

inline auto Checker::match_anchor(Id pattern,Id target,const std::vector<Id> &captures) -> Captures {
  // These are closed local constraints: no opaque lookup, deferred
  // predicate, Call registration or new Arena nodes can change the result.
  // Cache the exact existing matcher result, including an empty result.
  // Short coordinate recipes run all original integer/scope/residual guards
  // on the first query. Only that validated result is reusable; merely
  // reconstructing an existing Read is never a domain proof.
  if(!cacheable_read_anchor(pattern))return matches(pattern,target,captures);
  ReadAnchorKey key{pattern,target,captures};
  if(auto found=read_anchor_matches.find(key);found!=read_anchor_matches.end()) {
    ++a.stats["ep_body_read_match_cache_hits"];return found->second;
  }
  auto result=matches(pattern,target,captures);
  read_anchor_matches.emplace(std::move(key),result);
  ++a.stats["ep_body_read_match_cache_entries"];
  return result;
}

} // namespace rg
