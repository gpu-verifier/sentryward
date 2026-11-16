#pragma once
#include "core.hpp"

namespace rg {

inline auto Arena::render(Id id, int depth) const -> std::string {
  const auto &n = nodes[id];
  if (n.op == "const")
    return str(n.data[0]);
  if (depth <= 0)
    return n.op + "#" + std::to_string(id);
  std::string s = n.op + "[" + n.tag;
  for (const auto &q : n.data)
    s += "," + str(q);
  s += "](";
  for (Id x : n.args)
    s += render(x, depth - 1) + ",";
  return s + ")";
}

// The diagnostic text of a large expression DAG (an unproved family's two
// sides). It differs from render in two ways. A node with operands is
// expanded only once, as op#id[...](...); every later occurrence is op#id.
// The text ends after `limit` bytes, at a UTF-8 character boundary, and is
// then marked "...(truncated)". O(limit + nodes expanded within depth).
inline auto Arena::render_shared(Id root, int depth, std::size_t limit) const -> std::string {
  std::string s;
  std::unordered_set<Id> expanded;
  const auto go = [&](const auto &self, Id id, int remaining) -> void {
    if (s.size() > limit)
      return;
    const auto &n = nodes[id];
    if (n.op == "const") {
      s += str(n.data[0]);
      return;
    }
    if (remaining <= 0 || (!n.args.empty() && !expanded.insert(id).second)) {
      s += n.op + "#" + std::to_string(id);
      return;
    }
    s += n.args.empty() ? n.op : n.op + "#" + std::to_string(id);
    s += "[" + n.tag;
    for (const auto &q : n.data)
      s += "," + str(q);
    s += "](";
    for (Id x : n.args) {
      self(self, x, remaining - 1);
      if (s.size() > limit)
        return;
      s += ",";
    }
    s += ")";
  };
  go(go, root, depth);
  if (s.size() <= limit)
    return s;
  std::size_t cut = limit;
  while (cut > 0 && (static_cast<unsigned char>(s[cut]) & 0xC0) == 0x80)
    --cut;
  s.resize(cut);
  return s + "...(truncated)";
}

} // namespace rg
