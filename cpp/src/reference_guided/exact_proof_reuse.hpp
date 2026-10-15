#pragma once
#include "checker.hpp"

namespace rg {
// One comparison runner owns this bank, so Arena identities and immutable input
// premises never cross runs. These are exact, already-proved equalities, not
// parameterized rules: neither output coordinates nor Fold binders are renamed.
class ExactProofReuse {
  Checker &checker;
  Arena &a;
  std::unordered_map<Id, Id> certified, memo;
  size_t version = 0, memo_version = 0;
  size_t node_budget;
public:
  explicit ExactProofReuse(Checker &owner, size_t budget = 100000)
      : checker(owner), a(owner.a), node_budget(budget) {}
  bool empty() const { return certified.empty(); }
  // Called only after a complete output or exact reduction-coverage equality
  // has a successful certificate. Partial contribution matches are insufficient.
  // Decreasing Id orientation prevents cycles without walking either operand.
  bool record(Id reference, Id optimized) {
    if (reference == optimized) return false;
    if (reference >= optimized || !a.nodes[reference].facts.finite ||
        !a.nodes[optimized].facts.finite) {
      ++a.stats["exact_proof_reuse_orientation_declines"];
      return false;
    }
    auto [it, fresh] = certified.emplace(optimized, reference);
    if (fresh) { ++version; ++a.stats["exact_proof_reuse_certificates"]; }
    return fresh;
  }
  std::optional<Id> rewrite(Id root) {
    if (empty()) return root;
    ++a.stats["exact_proof_reuse_attempts"];
    // Only unresolved outputs call rewrite. Recording thousands of successful
    // outputs therefore does not invalidate the ordinary normalization cache.
    if (memo_version != version) { memo.clear(); memo_version = version; }
    if (memo.size() >= node_budget) memo.clear();
    size_t work = 0;
    std::unordered_map<Id, Id> local;
    std::vector<std::pair<Id, bool>> todo{{root, false}};
    auto get = [&](Id id) -> std::optional<Id> {
      if (auto it = local.find(id); it != local.end()) return it->second;
      if (auto it = memo.find(id); it != memo.end()) return it->second;
      return {};
    };
    while (!todo.empty()) {
      a.check();
      if (++work > node_budget) {
        ++a.stats["exact_proof_reuse_budget_declines"];
        return {};
      }
      auto [old, ready] = todo.back(); todo.pop_back();
      if (get(old)) continue;
      if (auto it = certified.find(old); it != certified.end()) {
        Id replacement = it->second;
        if (auto value = get(replacement)) {
          local.emplace(old, *value);
          ++a.stats["exact_proof_reuse_hits"];
        } else {
          todo.emplace_back(old, false);
          todo.emplace_back(replacement, false);
        }
        continue;
      }
      const auto &node = a.nodes[old];
      if (!ready) {
        // Count edges before pushing: one very wide Add must not allocate
        // unbounded pending work or hide behind a one-node accounting unit.
        if (node.args.size() > node_budget - work) {
          ++a.stats["exact_proof_reuse_budget_declines"];
          return {};
        }
        work += node.args.size();
        todo.emplace_back(old, true);
        for (Id child : node.args) if (!get(child)) todo.emplace_back(child, false);
        continue;
      }
      bool changed = false;
      std::vector<Id> args; args.reserve(node.args.size());
      for (Id child : node.args) {
        Id value = *get(child); args.push_back(value); changed |= value != child;
      }
      // Congruence also works through nonlinear scalar boundaries such as
      // rsqrt. Only their children change; reapply retains operation semantics.
      Id result = old;
      try {
        if (changed) result = checker.reapply(old, args);
      } catch (const Unsupported &) {
        // Equivalent replacements can carry weaker interval metadata. An
        // optional rewriter must not upgrade an unresolved comparison to an
        // importer blocker when an operator needs that missing evidence.
        ++a.stats["exact_proof_reuse_reapply_declines"];
        return {};
      }
      local.emplace(old, result);
      ++a.stats["exact_proof_reuse_node_visits"];
    }
    Id result = *get(root);
    if (memo.size() + local.size() > node_budget) memo.clear();
    memo.insert(local.begin(), local.end());
    return result;
  }
};
} // namespace rg
