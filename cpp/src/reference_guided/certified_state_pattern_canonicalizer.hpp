#pragma once
#include "core.hpp"

namespace rg {
struct CertifiedStatePatternCanonical {
  Id pattern = invalid;
  size_t axis = 0;
  // The caller must reorder actual slot_values with this same bijection.
  // Constants/input coordinates are not generalized or discarded.
  std::vector<Id> old_to_new;
};
struct CertifiedStatePatternCanonicalLimits {
  size_t nodes = 128, edges = 128, work = 16384, entries = 1024;
};

// Canonicalization of a small *definition*, not a proof of any new equality.
// Only coordinate-slot renaming and Add/Mul/Max operand permutations occur.
// A certified_state tag stays exact, including its proof-bank/schema identity.
class CertifiedStatePatternCanonicalizer {
  Arena &patterns, &owner;
  CertifiedStatePatternCanonicalLimits limits;
  using Key = std::tuple<Id, size_t, size_t>;
  std::map<Key, std::optional<CertifiedStatePatternCanonical>> cache;
  size_t work = 0;
  struct Decline { const char *reason; };
  void tick(size_t amount = 1) {
    owner.check();
    if (amount > limits.work - std::min(work, limits.work)) throw Decline{"work_budget"};
    work += amount;
  }
  bool plain_node(const Node &n, size_t slots) const {
    if (n.op == "slot")
      return n.args.empty() && n.tag.empty() && n.data.size() == 1 &&
             n.data[0].denominator() == 1 && n.data[0] >= 0 && n.data[0] < Q(slots);
    if (n.op == "add") return n.tag.empty() && n.data.size() == n.args.size();
    if (n.op == "mul") return n.tag.empty() && n.data.size() == 1;
    if (n.op == "max") return n.tag.empty() && n.data.empty();
    if (n.op == "const") return n.args.empty() && n.tag.empty() && n.data.size() == 1;
    if (n.op == "atom") return n.args.empty() && n.data.empty();
    if (n.op == "certified_state")
      return !n.tag.empty() && !n.args.empty() && n.args.size() <= 4 && n.data.empty();
    if (n.op == "read") return !n.tag.empty() && n.data.empty();
    if (n.op == "call")
      return n.tag.empty() && n.data.size() == 1 && n.data[0].denominator() == 1 && n.data[0] >= 0;
    if (n.op == "recip" || n.op == "reciprocalsqrt" || n.op == "sqrt" ||
        n.op == "abs" || n.op == "sin" || n.op == "cos" || n.op == "tanh" ||
        n.op == "exp" || n.op == "exp2" || n.op == "log" || n.op == "log2" || n.op == "floor")
      return n.tag.empty() && n.args.size() == 1 && n.data.empty();
    if (n.op == "power") return n.tag.empty() && n.args.size() == 2 && n.data.empty();
    if (n.op == "clamp") return n.tag.empty() && n.args.size() == 1 && n.data.size() == 2;
    if (n.op == "select") return n.tag.empty() && n.args.size() == 3 && n.data.empty();
    if (n.op == "compare") return n.args.size() == 2 && n.data.empty();
    // In particular, affine_slot, affine_param, derived_coordinate and
    // coordinate_sum encode slot IDs in other data fields and must decline.
    return false;
  }
public:
  std::string last_failure;
  CertifiedStatePatternCanonicalizer(Arena &pattern_arena, Arena &owner_arena,
                                    CertifiedStatePatternCanonicalLimits budget = {})
      : patterns(pattern_arena), owner(owner_arena), limits(budget) {
    limits.nodes = std::min<size_t>(limits.nodes, 128);
    limits.edges = std::min<size_t>(limits.edges, 128);
    limits.work = std::min<size_t>(limits.work, 16384);
    limits.entries = std::min<size_t>(limits.entries, 1024);
  }
  std::optional<CertifiedStatePatternCanonical> canonicalize(Id root, size_t slots, size_t axis) {
    last_failure.clear(); work = 0;
    const Key key{root, slots, axis};
    if (auto found = cache.find(key); found != cache.end()) {
      ++owner.stats["certified_state_pattern_cache_hits"];
      return found->second;
    }
    if (cache.size() >= limits.entries) {
      last_failure = "cache_budget"; ++owner.stats["certified_state_pattern_cache_declines"]; return {};
    }
    std::optional<CertifiedStatePatternCanonical> result;
    try {
      if (slots == 0 || slots > 5 || axis >= slots || root >= patterns.nodes.size())
        throw Decline{"slot_domain"};
      bool has_state = false;
      std::vector<bool> used(slots, false);
      std::set<Id> seen;
      std::vector<Id> todo{root};
      size_t edges = 0;
      while (!todo.empty()) {
        Id id = todo.back(); todo.pop_back();
        if (!seen.insert(id).second) continue;
        tick();
        if (seen.size() > limits.nodes) throw Decline{"node_budget"};
        const auto &n = patterns.nodes[id];
        if (n.args.size() > limits.edges - std::min(edges, limits.edges)) throw Decline{"edge_budget"};
        edges += n.args.size(); tick(n.args.size());
        if (!plain_node(n, slots)) throw Decline{"nonplain_pattern"};
        if (n.op == "slot") used[n.data[0].numerator().convert_to<size_t>()] = true;
        has_state |= n.op == "certified_state";
        for (Id child : n.args) {
          // Arena construction is topological. Reject malformed/cyclic input
          // before any copied argument list or recursive traversal is made.
          if (child >= id) throw Decline{"pattern_dag"};
          todo.push_back(child);
        }
      }
      if (!has_state) throw Decline{"no_certified_state"};
      if (std::find(used.begin(), used.end(), false) != used.end()) throw Decline{"unused_slot"};
      std::vector<Id> old_slots, labels;
      for (size_t i = 0; i < slots; ++i) if (i != axis) old_slots.push_back(Id(i));
      for (size_t i = 1; i < slots; ++i) labels.push_back(Id(i));
      do {
        tick();
        std::vector<Id> permutation(slots); permutation[axis] = 0;
        for (size_t i = 0; i < old_slots.size(); ++i) permutation[old_slots[i]] = labels[i];
        std::map<Id, Id> rebuilt;
        for (Id id : seen) {
          tick(); const auto &n = patterns.nodes[id];
          tick(n.args.size() + n.data.size());
          if (n.op == "slot") {
            Id slot = n.data[0].numerator().convert_to<Id>();
            rebuilt.emplace(id, patterns.intern("slot", {}, {Q(permutation[slot])}));
            continue;
          }
          std::vector<Id> args; args.reserve(n.args.size());
          for (Id child : n.args) args.push_back(rebuilt.at(child));
          std::vector<Q> data = n.data;
          if (n.op == "add") {
            std::vector<std::pair<Id, Q>> terms; terms.reserve(args.size());
            for (size_t i = 0; i < args.size(); ++i) terms.emplace_back(args[i], data[i]);
            std::sort(terms.begin(), terms.end(), [&](const auto &x, const auto &y) { tick(); return x < y; });
            for (size_t i = 0; i < terms.size(); ++i) { args[i] = terms[i].first; data[i] = terms[i].second; }
          } else if (n.op == "mul" || n.op == "max") {
            std::sort(args.begin(), args.end(), [&](Id x, Id y) { tick(); return x < y; });
          }
          rebuilt.emplace(id, patterns.intern(n.op, std::move(args), std::move(data), n.tag));
        }
        Id candidate = rebuilt.at(root);
        if (!result || candidate < result->pattern)
          result = CertifiedStatePatternCanonical{candidate, 0, std::move(permutation)};
        ++owner.stats["certified_state_pattern_permutations"];
      } while (std::next_permutation(labels.begin(), labels.end()));
      ++owner.stats["certified_state_pattern_canonicalizations"];
    } catch (const Decline &miss) {
      // An incomplete orbit cannot establish a stable representative.
      result.reset(); last_failure = miss.reason;
      ++owner.stats["certified_state_pattern_decline_" + last_failure];
    }
    owner.stats["certified_state_pattern_work"] += work;
    cache.emplace(key, result);
    return result;
  }
};
} // namespace rg
