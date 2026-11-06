#pragma once
#include "checker.hpp"

namespace rg {
// Proof by cases on one output coordinate. When the root of either side is Select(c, A, B) and c
// compares a free output coordinate v (an integer in [lo, hi]) with a constant, v is split: into
// its points when the domain has at most max_points of them, else into the interval where c holds
// and the rest. Each side is restricted to each part (v replaced by the point, or by a fresh
// coordinate over the interval) and the restricted sides are compared by the caller. The parts
// cover v's domain exactly once and a restriction is a capture-safe substitution, so equal parts
// make equal families. A condition that holds on, or fails on, more than one interval (eq/ne
// strictly inside the domain) declines. Cost: one restriction per part and side, O(nodes that
// depend on v) each (Checker::restrict_axis); at most max_points parts.
class OutputGuardSplit {
  Checker &checker;
  Arena &a;
  uint64_t serial = 0;

public:
  // Small coordinate domains (e.g. a state-slot pool) are split into points, so that index
  // arithmetic such as (v + 7) mod 8 folds; the bound keeps the work per family small.
  static constexpr unsigned max_points = 16;
  // Selections with one identical condition are compared arm by arm, at most this deep.
  static constexpr unsigned max_select_depth = 8;
  struct Part {
    Id left = invalid, right = invalid;
    Z first = 0, last = 0;
  };
  struct Guard {
    Id axis = invalid;
    Z first = 0, last = 0;  // the interval of v's domain where the condition holds
  };
  std::string last_failure;
  explicit OutputGuardSplit(Checker &owner) : checker(owner), a(owner.a) {}

  // The coordinate guard of a root Select, if its condition compares one output coordinate with a
  // constant and holds on one interval of the coordinate's domain.
  std::optional<Guard> guard(Id root) const {
    const auto &n = a.nodes[root];
    if (n.op != "select" || n.args.size() != 3) return {};
    const auto &c = a.nodes[n.args[0]];
    if (c.op != "compare" || c.args.size() != 2 || !c.data.empty()) return {};
    for (unsigned side = 0; side != 2; ++side) {
      const Id v = c.args[side];
      const auto &axis = a.nodes[v];
      const auto constant = a.value(c.args[1 - side]);
      if (axis.op != "var" || !axis.tag.starts_with("output:") || !axis.bounds || !constant ||
          axis.bounds->first.denominator() != 1 || axis.bounds->second.denominator() != 1)
        continue;
      const Z lo = axis.bounds->first.numerator(), hi = axis.bounds->second.numerator();
      // The comparison as `v op k`.
      std::string op = side ? Arena::reverse_comparison(c.tag) : c.tag;
      const Q k = *constant;
      Z first = lo, last = hi;
      if (op == "le") last = floorq(k);
      else if (op == "lt") last = -floorq(-k) - 1;
      else if (op == "ge") first = -floorq(-k);
      else if (op == "gt") first = floorq(k) + 1;
      else if (op == "eq" || op == "ne") {
        if (k.denominator() != 1 || (k.numerator() != lo && k.numerator() != hi)) return {};
        const Z point = k.numerator();
        if (op == "eq") first = last = point;
        else if (point == lo) first = lo + 1;
        else last = hi - 1;
      } else return {};
      first = std::max(first, lo);
      last = std::min(last, hi);
      // Both the condition's interval and its complement must be nonempty and within the domain.
      if (first > last || (first == lo && last == hi)) return {};
      return Guard{v, first, last};
    }
    return {};
  }

  // The two restricted pairs of (left, right) for the first root coordinate guard of either side.
  std::optional<std::vector<Part>> split(Id left, Id right) {
    last_failure.clear();
    std::optional<Guard> g = guard(left);
    if (!g) g = guard(right);
    if (!g) return {};
    const auto &bounds = *a.nodes[g->axis].bounds;
    const Z lo = bounds.first.numerator(), hi = bounds.second.numerator();
    std::vector<std::pair<Z, Z>> intervals;
    if (hi - lo < Z(max_points))
      for (Z point = lo; point <= hi; ++point) intervals.emplace_back(point, point);
    else {
      intervals.emplace_back(g->first, g->last);
      if (g->first > lo) intervals.emplace_back(lo, g->first - 1);
      if (g->last < hi) intervals.emplace_back(g->last + 1, hi);
      if (intervals.size() != 2) { last_failure = "guard_interval"; return {}; }
    }
    std::vector<Part> parts;
    try {
      for (const auto &[first, last] : intervals) {
        Id value;
        if (first == last) value = a.constant(Q(first));
        else {
          // A fresh coordinate: a node that already existed would alias another scope's variable.
          size_t before;
          do {
            before = a.nodes.size();
            value = a.var("output-guard-split:" + std::to_string(serial++), Q(first), Q(last));
          } while (value < before);
        }
        parts.push_back({checker.restrict_axis(left, g->axis, value),
                         checker.restrict_axis(right, g->axis, value), first, last});
      }
    } catch (const Unsupported &) {
      last_failure = "restriction";
      return {};
    }
    ++a.stats["output_guard_splits"];
    return parts;
  }
  // The pairs whose equality implies left == right: two Selects with the same condition are equal
  // when their true arms and their false arms are; identical subterms need nothing. Appends to
  // `pairs` (no duplicates). Exact sufficient condition; O(depth).
  void obligations(Id left, Id right, std::vector<std::pair<Id, Id>> &pairs, unsigned depth = 0) const {
    if (left == right) return;
    const auto &l = a.nodes[left], &r = a.nodes[right];
    if (depth < max_select_depth && l.op == "select" && r.op == "select" && l.args.size() == 3 &&
        r.args.size() == 3 && l.args[0] == r.args[0]) {
      const Id lt = l.args[1], lf = l.args[2], rt = r.args[1], rf = r.args[2];
      obligations(lt, rt, pairs, depth + 1);
      obligations(lf, rf, pairs, depth + 1);
      return;
    }
    if (std::find(pairs.begin(), pairs.end(), std::pair(left, right)) == pairs.end())
      pairs.emplace_back(left, right);
  }
};
} // namespace rg
