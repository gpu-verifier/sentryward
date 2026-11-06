#pragma once
#include "checker.hpp"

namespace rg {
// Select(c, A, B) whose arms agree wherever c holds is B. The condition c recognized here says that a
// nonnegative weighted sum of Boolean products is zero (a warp ballot of a per-lane predicate,
// masked and compared with zero), so every product is zero and in each one some factor is false.
// A product [M <= m]·[M' != m] whose M and M' are maxima over the same operand multiset X gives
// max(X) >= m: either max(X) = m, or not max(X) <= m. Under these facts a maximum over X and m is
// the maximum over X. A and B are rewritten under the facts, every maximum by its flattened operand
// multiset; when the rewrites are the same node, A = B wherever c holds, so the Select is B for
// every input. Exact. Cost: each stored node is visited once per comparison (memoized across
// families), plus one bounded rewrite of A and B per Select with such a condition.
class ConditionalSelect {
  Checker &checker;
  Arena &a;
  // Bounds on the shapes and the work of one elimination.
  static constexpr size_t max_products = 64, max_leaves = 4096, max_rewrite_nodes = 200000;
  struct Fact {
    std::vector<Id> operands;  // sorted maximum operands X
    Id floor = invalid;        // m, with max(X) >= m
  };
  std::unordered_map<Id, std::optional<std::vector<Id>>> leaf_cache;
  // Elimination results, shared by every family of one comparison: a node's result depends only on
  // the node, so each stored node is visited once per comparison, not once per family.
  std::unordered_map<Id, Id> eliminated;
  size_t work = 0;

  // The sorted operand multiset of a maximum, nested maxima flattened.
  std::optional<std::vector<Id>> leaves(Id root) {
    if (auto it = leaf_cache.find(root); it != leaf_cache.end()) return it->second;
    std::vector<Id> out, todo{root};
    while (!todo.empty()) {
      Id id = todo.back(); todo.pop_back();
      if (a.nodes[id].op == "max") for (Id x : a.nodes[id].args) todo.push_back(x);
      else out.push_back(id);
      if (out.size() + todo.size() > max_leaves) return leaf_cache[root] = std::nullopt;
    }
    std::sort(out.begin(), out.end());
    return leaf_cache[root] = out;
  }
  // A Boolean as one comparison and its polarity: compare(op, X, k) itself, select(compare, 1, 0)
  // likewise, select(compare, 0, 1) as its complement, and 1 - b as the complement of b.
  struct Literal { Id compare = invalid; bool positive = true; };
  std::optional<Literal> literal(Id x) const {
    const auto &n = a.nodes[x];
    if (n.op == "compare" && n.args.size() == 2) return Literal{x, true};
    if (n.op == "select" && n.args.size() == 3 && a.nodes[n.args[0]].op == "compare") {
      if (n.args[1] == a.zero && n.args[2] == a.one) return Literal{n.args[0], false};
      if (n.args[1] == a.one && n.args[2] == a.zero) return Literal{n.args[0], true};
    }
    const auto t = const_cast<Arena &>(a).terms(x);
    if (t.size() == 2)
      for (unsigned j = 0; j < 2; ++j)
        if (t[j].first == a.one && t[j].second == 1 && t[1-j].second == -1) {
          auto inner = literal(t[1-j].first);
          if (inner) { inner->positive = !inner->positive; return inner; }
        }
    return {};
  }
  // The sum S whose zero is the condition c.
  std::optional<Id> zero_sum(Id c) const {
    auto l = literal(c);
    if (!l) return {};
    const auto &cmp = a.nodes[l->compare];
    const bool eq = cmp.tag == "eq", ne = cmp.tag == "ne";
    if ((!eq && !ne) || (eq != l->positive) || cmp.args[1] != a.zero) return {};
    return cmp.args[0];
  }
  // max(X) >= m from one zero product [M <= m]·[M' != m].
  std::optional<Fact> product_fact(Id product) {
    const auto &n = a.nodes[product];
    if (n.op != "mul" || n.data.size() != 1 || n.data[0] <= 0 || n.args.size() != 2) return {};
    std::optional<Id> at_most, unequal;
    Id bound = invalid;
    for (Id factor : n.args) {
      auto l = literal(factor);
      if (!l) return {};
      const auto &cmp = a.nodes[l->compare];
      if (cmp.args.size() != 2 || !a.value(cmp.args[1])) return {};
      const bool le = cmp.tag == "le" && l->positive, gt = cmp.tag == "gt" && !l->positive;
      const bool ne = (cmp.tag == "ne" && l->positive) || (cmp.tag == "eq" && !l->positive);
      if (le || gt) at_most = cmp.args[0];
      else if (ne) unequal = cmp.args[0];
      else return {};
      if (bound != invalid && bound != cmp.args[1]) return {};
      bound = cmp.args[1];
    }
    if (!at_most || !unequal) return {};
    auto x = leaves(*at_most), y = leaves(*unequal);
    if (!x || !y || *x != *y || !a.nodes[*at_most].facts.finite) return {};
    return Fact{*x, bound};
  }
  std::optional<std::vector<Fact>> facts(Id c) {
    auto sum = zero_sum(c);
    if (!sum) return {};
    const auto &s = a.nodes[*sum];
    std::vector<std::pair<Id, Q>> terms;
    if (s.op == "add" && s.args.size() == s.data.size()) for (size_t i = 0; i < s.args.size(); ++i) terms.emplace_back(s.args[i], s.data[i]);
    else terms.emplace_back(*sum, Q(1));
    if (terms.empty() || terms.size() > max_products) return {};
    std::vector<Fact> out;
    for (const auto &[t, k] : terms) {
      if (k <= 0) return {};
      auto f = product_fact(t);
      if (!f) return {};
      out.push_back(std::move(*f));
    }
    return out;
  }
  // root rewritten under the facts: a maximum whose operands are X, or X and m, becomes the flat
  // maximum over X. Other nodes are rebuilt from their rewritten operands.
  Id rewrite(Id root, const std::vector<Fact> &facts, std::unordered_map<Id, Id> &memo) {
    std::vector<std::pair<Id, bool>> todo{{root, false}};
    while (!todo.empty()) {
      a.check();
      auto [id, ready] = todo.back(); todo.pop_back();
      if (memo.contains(id)) continue;
      if (++work > max_rewrite_nodes) throw Unsupported("conditional select rewrite budget");
      const auto &n = a.nodes[id];
      if (n.op == "max") {
        if (auto x = leaves(id)) {
          Id replaced = invalid;
          for (const auto &f : facts) {
            if (*x == f.operands) { replaced = a.maximum(f.operands); break; }
            std::vector<Id> with = f.operands;
            with.insert(std::upper_bound(with.begin(), with.end(), f.floor), f.floor);
            if (*x == with) { replaced = a.maximum(f.operands); break; }
          }
          if (replaced != invalid) { memo.emplace(id, replaced); continue; }
        }
      }
      if (n.args.empty() || n.op == "fold" || n.op == "call") { memo.emplace(id, id); continue; }
      if (!ready) {
        todo.emplace_back(id, true);
        for (Id x : n.args) if (!memo.contains(x)) todo.emplace_back(x, false);
        continue;
      }
      std::vector<Id> args;
      args.reserve(n.args.size());
      bool changed = false;
      for (Id x : n.args) { args.push_back(memo.at(x)); changed |= args.back() != x; }
      memo.emplace(id, changed ? checker.reapply(id, args) : id);
    }
    return memo.at(root);
  }

public:
  std::string last_failure;
  explicit ConditionalSelect(Checker &owner) : checker(owner), a(owner.a) {}
  // root with every Select(c, A, B) of the recognized kind whose arms agree under c replaced by
  // B; root itself when none is. Bottom-up; results are memoized across calls (`eliminated`), and
  // the work budget counts only nodes not visited before.
  Id eliminate(Id root) {
    last_failure.clear(); work = 0;
    auto &done = eliminated;
    std::vector<std::pair<Id, bool>> todo{{root, false}};
    size_t removed = 0;
    try {
      while (!todo.empty()) {
        a.check();
        auto [id, ready] = todo.back(); todo.pop_back();
        if (done.contains(id)) continue;
        if (++work > max_rewrite_nodes) throw Unsupported("conditional select budget");
        const auto &n = a.nodes[id];
        if (n.args.empty() || n.op == "fold" || n.op == "call") { done.emplace(id, id); continue; }
        if (!ready) {
          todo.emplace_back(id, true);
          for (Id x : n.args) if (!done.contains(x)) todo.emplace_back(x, false);
          continue;
        }
        std::vector<Id> args;
        args.reserve(n.args.size());
        bool changed = false;
        for (Id x : n.args) { args.push_back(done.at(x)); changed |= args.back() != x; }
        Id result = changed ? checker.reapply(id, args) : id;
        const auto &s = a.nodes[result];
        if (s.op == "select" && s.args.size() == 3)
          if (auto f = facts(s.args[0])) {
            std::unordered_map<Id, Id> memo;
            if (rewrite(s.args[1], *f, memo) == rewrite(s.args[2], *f, memo)) {
              result = s.args[2];
              ++removed;
            }
          }
        done.emplace(id, result);
      }
    } catch (const Unsupported &e) {
      last_failure = e.what();
      return root;
    }
    if (removed) a.stats["conditional_select_eliminations"] += removed;
    return done.at(root);
  }
};
} // namespace rg
