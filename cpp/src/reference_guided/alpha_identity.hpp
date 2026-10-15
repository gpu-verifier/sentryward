#pragma once
#include "core.hpp"

namespace rg {
// Exact identity modulo lexical Fold binder names. This is not reduction
// matching: it never guesses coordinates, changes a domain, or visits members.
// Representatives are existing Arena nodes; the physical bindings used by
// reference registration, substitution and domain proofs remain untouched.
class FoldAlphaIdentity {
  Arena &a;
  struct Signature {
    std::string op, tag;
    std::vector<Id> args;
    std::vector<Q> data;
    bool operator==(const Signature &) const = default;
  };
  using Environment = std::vector<std::pair<Id, size_t>>;
  struct State {
    Id node;
    size_t depth;
    Environment env;
    bool operator<(const State &o) const {
      return std::tie(node, depth, env) < std::tie(o.node, o.depth, o.env);
    }
  };
  std::deque<Signature> signatures;
  std::unordered_map<size_t, std::vector<Id>> buckets;
  std::map<State, Id> memo;
  std::unordered_map<Id, Id> roots, representatives;

  Id intern(Signature s) {
    a.check();
    const size_t hash = Arena::hash(s.op, s.args, s.data, s.tag);
    auto &bucket = buckets[hash];
    for (Id id : bucket) if (signatures[id] == s) return id;
    if (signatures.size() >= invalid) throw Unsupported("alpha identity capacity");
    Id id = Id(signatures.size());
    a.stats["fold_alpha_signature_edges"] += s.args.size();
    signatures.push_back(std::move(s)); bucket.push_back(id);
    ++a.stats["fold_alpha_signatures"];
    return id;
  }
  State state(Id id, size_t depth, const Environment &env) {
    State s{id, depth, {}};
    // Cached free sets identify the only outer bindings relevant to this
    // subtree. Fold free sets include its seed, but exclude its body binder.
    for (const auto &binding : env) {
      a.check();
      if (a.depends(id, binding.first)) s.env.push_back(binding);
    }
    return s;
  }
  bool fold(const Node &n) const {
    return n.op == "fold" && n.args.size() == 2 && n.data.size() == 2 &&
      (n.tag == "Sum" || n.tag == "Max") && n.data[0].denominator() == 1 &&
      n.data[0] >= 0 && n.data[0] < Q(a.nodes.size()) &&
      n.data[1].denominator() == 1 && n.data[1] >= 0 &&
      a.nodes[n.data[0].numerator().convert_to<Id>()].op == "var";
  }
  State child(const State &s, size_t index) {
    const Node &n = a.nodes[s.node];
    if (!fold(n) || index == 0) return state(n.args[index], s.depth, s.env);
    const Id axis = n.data[0].numerator().convert_to<Id>();
    Environment env = s.env;
    auto it = std::lower_bound(env.begin(), env.end(), axis,
      [](const auto &entry, Id id) { return entry.first < id; });
    if (it != env.end() && it->first == axis) it->second = s.depth;
    else env.insert(it, {axis, s.depth});
    return state(n.args[index], s.depth + 1, env);
  }
  static std::string reverse(std::string p) {
    if (p == "le") return "ge";
    if (p == "ge") return "le";
    if (p == "lt") return "gt";
    if (p == "gt") return "lt";
    return p;
  }
  // Arena's affine predicate constructor chooses a sign using raw atom IDs.
  // Remove only that storage-order choice, using these exact alpha keys. The
  // shell is bounded (the same 32-term contract as Arena::compare); no search
  // or recursive polynomial normalization is performed here.
  std::optional<Id> comparison(const Node &n, const std::vector<Id> &args) {
    if (n.op != "compare" || args.size() != 2 || !n.data.empty() ||
        !a.nodes[n.args[0]].facts.finite || !a.nodes[n.args[1]].facts.finite ||
        (n.tag != "eq" && n.tag != "ne" && n.tag != "lt" &&
         n.tag != "le" && n.tag != "gt" && n.tag != "ge")) return {};
    std::map<Id, Q> terms; Q offset = 0;
    auto append = [&](Id id, Q scale) {
      const auto &s = signatures[id];
      if (s.op == "const" && s.data.size() == 1) offset += scale * s.data[0];
      else if (s.op == "mul" && s.args.size() == 1 && s.data.size() == 1)
        terms[s.args[0]] += scale * s.data[0];
      else terms[id] += scale;
    };
    size_t visits = 0;
    for (size_t side = 0; side < 2; ++side) {
      const auto &s = signatures[args[side]];
      Q scale = side == 0 ? Q(1) : Q(-1);
      if (s.op == "add" && s.args.size() == s.data.size()) {
        if ((visits += s.args.size()) > 32) return {};
        for (size_t j = 0; j < s.args.size(); ++j) append(s.args[j], scale * s.data[j]);
      } else { if (++visits > 32) return {}; append(args[side], scale); }
    }
    for (auto it = terms.begin(); it != terms.end();)
      if (it->second == 0) it = terms.erase(it); else ++it;
    std::string tag = n.tag;
    const Q scale = !terms.empty() && terms.begin()->second < 0 ? Q(-1) : Q(1);
    if (scale < 0) tag = reverse(tag);
    Signature key{"$alpha_compare", tag, {}, {offset * scale}};
    for (const auto &[id, coefficient] : terms) {
      key.args.push_back(id); key.data.push_back(coefficient * scale);
    }
    return intern(std::move(key));
  }
  Id finish(const State &s, std::vector<Id> args) {
    const Node &n = a.nodes[s.node];
    if (n.op == "var") {
      auto it = std::lower_bound(s.env.begin(), s.env.end(), s.node,
        [](const auto &entry, Id id) { return entry.first < id; });
      if (it == s.env.end() || it->first != s.node)
        return intern({"$alpha_free", "", {}, {Q(s.node)}});
      std::vector<Q> data{Q(it->second)};
      data.insert(data.end(), n.data.begin(), n.data.end());
      if (n.bounds) { data.push_back(n.bounds->first); data.push_back(n.bounds->second); }
      return intern({"$alpha_bound", n.bounds ? "bounded" : "", {}, std::move(data)});
    }
    if (fold(n)) {
      const auto &axis = a.nodes[n.data[0].numerator().convert_to<Id>()];
      // Count AND declared domain are retained. A wider interval may have
      // changed already-applied simplifications, so it is not alpha renaming.
      std::vector<Q> data{n.data[1], Q(axis.data.size())};
      data.insert(data.end(), axis.data.begin(), axis.data.end());
      data.push_back(Q(bool(axis.bounds)));
      if (axis.bounds) { data.push_back(axis.bounds->first); data.push_back(axis.bounds->second); }
      return intern({"fold", n.tag, std::move(args), std::move(data)});
    }
    if (auto result = comparison(n, args)) return *result;
    if (n.op == "select" && args.size() == 3 && signatures[args[0]].op == "$alpha_compare") {
      // Arena orders affine atoms by physical IDs before selecting a Boolean
      // polarity. Binding abstraction may reverse that order. Canonicalize
      // the already bounded predicate signature together with its two arms;
      // do not rebuild any physical expression or compare numerical bodies.
      Signature predicate = signatures[args[0]];
      bool invert = predicate.tag == "ne" || predicate.tag == "ge" || predicate.tag == "gt";
      if (invert) {
        predicate.tag = predicate.tag == "ne" ? "eq" : predicate.tag == "ge" ? "lt" : "le";
        std::swap(args[1], args[2]);
      }
      // $alpha_compare denotes (sum(coeff*atom) + offset) op 0. For an
      // integer-valued difference, d<0 is exactly d+1<=0. Integer endpoints
      // alone do not establish this: reuse the cached exact provenance of
      // the original operands, whose finiteness comparison() already checked.
      const auto &original = a.nodes[n.args[0]];
      if (predicate.tag == "lt" && original.op == "compare" && original.args.size() == 2 &&
          a.known_bits(original.args[0]).integer &&
          a.known_bits(original.args[1]).integer) {
        predicate.tag = "le";
        predicate.data[0] += 1;
      }
      args[0] = intern(std::move(predicate));
    }
    std::vector<Q> data = n.data;
    if (n.op == "add" && n.facts.finite && args.size() == data.size()) {
      std::map<Id, Q> terms;
      for (size_t j = 0; j < args.size(); ++j) terms[args[j]] += data[j];
      args.clear(); data.clear();
      for (const auto &[id, coefficient] : terms) if (coefficient != 0) {
        args.push_back(id); data.push_back(coefficient);
      }
      if (args.empty()) return intern({"const", "", {}, {Q(0)}});
      if (args.size() == 1 && data[0] == 1) return args[0];
    } else if (n.op == "mul" || n.op == "max") {
      std::sort(args.begin(), args.end());
      if (n.op == "max") {
        args.erase(std::unique(args.begin(), args.end()), args.end());
        if (args.size() == 1) return args[0];
      }
    }
    return intern({n.op, n.tag, std::move(args), std::move(data)});
  }
  Id identity(Id root) {
    if (auto it = roots.find(root); it != roots.end()) {
      ++a.stats["fold_alpha_root_cache_hits"]; return it->second;
    }
    struct Frame { State s; bool ready; };
    State start = state(root, 0, {});
    std::vector<Frame> todo{{start, false}};
    while (!todo.empty()) {
      a.check();
      auto frame = std::move(todo.back()); todo.pop_back();
      if (memo.contains(frame.s)) continue;
      const auto &n = a.nodes[frame.s.node];
      // Unknown Fold layouts are kept opaque, not interpreted as binders.
      if (n.op == "fold" && !fold(n)) {
        std::vector<Q> context{Q(frame.s.node), Q(frame.s.depth)};
        for (const auto &[variable, level] : frame.s.env) {
          context.push_back(Q(variable)); context.push_back(Q(level));
        }
        Id key = intern({"$alpha_opaque", "", {}, std::move(context)});
        memo.emplace(std::move(frame.s), key);
        continue;
      }
      if (!frame.ready) {
        todo.push_back({frame.s, true});
        for (size_t j = 0; j < n.args.size(); ++j) {
          State next = child(frame.s, j);
          if (!memo.contains(next)) todo.push_back({std::move(next), false});
        }
        continue;
      }
      std::vector<Id> args; args.reserve(n.args.size());
      for (size_t j = 0; j < n.args.size(); ++j) args.push_back(memo.at(child(frame.s, j)));
      Id result = finish(frame.s, std::move(args));
      memo.emplace(std::move(frame.s), result);
      ++a.stats["fold_alpha_states"];
    }
    return roots.emplace(root, memo.at(start)).first->second;
  }
public:
  explicit FoldAlphaIdentity(Arena &arena) : a(arena) {}
  Id representative(Id root) {
    return representatives.try_emplace(identity(root), root).first->second;
  }
  bool equal(Id left, Id right) { return identity(left) == identity(right); }
};
} // namespace rg
