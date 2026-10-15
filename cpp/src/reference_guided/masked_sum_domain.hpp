#pragma once
#include "core.hpp"

namespace rg::masked_sum_detail {

// An exact constant slope in the reduction coordinate, plus an interval that
// encloses the coordinate-independent remainder. Bounds do not prove equality
// of expressions; here they only prove that a suffix cannot select a nonzero
// branch. Unknown or nonlinear coordinate dependence is a conservative miss.
struct AffineRange {
  Q slope = 0, low = 0, high = 0;
};

inline AffineRange scale(AffineRange value, const Q &factor) {
  value.slope *= factor;
  Q lo = value.low * factor, hi = value.high * factor;
  value.low = std::min(lo, hi); value.high = std::max(lo, hi);
  return value;
}

class AffineRangeReader {
  Arena &a;
  Id axis;
  std::unordered_map<Id, std::optional<AffineRange>> memo;
  size_t work = 0, edges = 0;
  bool exhausted = false;
  static constexpr size_t maximum_nodes = 128, maximum_edges = 256,
                          maximum_work = 768, maximum_arity = 32;

  static std::optional<AffineRange> product(const AffineRange &left,
                                           const AffineRange &right) {
    if (left.slope != 0) {
      if (right.slope != 0 || right.low != right.high) return {};
      return scale(left, right.low);
    }
    if (right.slope != 0) {
      if (left.low != left.high) return {};
      return scale(right, left.low);
    }
    const Q ll = left.low * right.low, lh = left.low * right.high,
            hl = left.high * right.low, hh = left.high * right.high;
    return AffineRange{0, std::min(std::min(ll, lh), std::min(hl, hh)),
                         std::max(std::max(ll, lh), std::max(hl, hh))};
  }

public:
  AffineRangeReader(Arena &arena, Id variable) : a(arena), axis(variable) {}

  std::optional<AffineRange> read(Id root) {
    if (exhausted || root >= a.nodes.size()) return {};
    if (auto it = memo.find(root); it != memo.end()) return it->second;
    std::vector<std::pair<Id, bool>> todo{{root, false}};
    while (!todo.empty()) {
      a.check();
      if (++work > maximum_work || memo.size() >= maximum_nodes) {
        exhausted = true; return {};
      }
      auto [id, ready] = todo.back(); todo.pop_back();
      if (memo.contains(id)) continue;
      if (id >= a.nodes.size()) { exhausted = true; return {}; }
      if (!ready) ++a.stats["masked_sum_predicate_nodes"];
      const auto &node = a.nodes[id];
      if (!node.facts.finite) { memo.emplace(id, std::nullopt); continue; }
      if (id == axis) { memo.emplace(id, AffineRange{1, 0, 0}); continue; }
      // depends() reads the immutable cached free-variable set, not ancestors.
      if (!a.depends(id, axis)) {
        if (node.bounds && node.bounds->first <= node.bounds->second)
          memo.emplace(id, AffineRange{0, node.bounds->first, node.bounds->second});
        else memo.emplace(id, std::nullopt);
        continue;
      }
      const bool sum = (node.op == "add" || node.op == "add_raw") &&
                       node.args.size() == node.data.size();
      const bool mul = node.op == "mul" && node.data.size() == 1;
      if ((!sum && !mul) || node.args.size() > maximum_arity) {
        memo.emplace(id, std::nullopt); continue;
      }
      if (!ready) {
        if (node.args.size() > maximum_edges - edges) {
          exhausted = true; return {};
        }
        edges += node.args.size();
        a.stats["masked_sum_predicate_edges"] += node.args.size();
        todo.emplace_back(id, true);
        for (Id child : node.args)
          if (!memo.contains(child)) todo.emplace_back(child, false);
        continue;
      }
      std::optional<AffineRange> result = sum ? AffineRange{}
          : AffineRange{0, node.data[0], node.data[0]};
      for (size_t j = 0; j < node.args.size() && result; ++j) {
        const auto &child = memo.at(node.args[j]);
        if (!child) { result.reset(); break; }
        if (sum) {
          auto term = scale(*child, node.data[j]);
          result->slope += term.slope;
          result->low += term.low; result->high += term.high;
        } else result = product(*result, *child);
      }
      memo.emplace(id, std::move(result));
    }
    return memo.at(root);
  }
};

inline std::string inverse(std::string predicate) {
  if (predicate == "lt") return "ge";
  if (predicate == "le") return "gt";
  if (predicate == "gt") return "le";
  if (predicate == "ge") return "lt";
  if (predicate == "eq") return "ne";
  if (predicate == "ne") return "eq";
  return {};
}

// Return a prefix containing every index where the predicate can be true.
// Interior holes may remain. In particular this never counts or enumerates
// the satisfying indices and never changes a Select to an unconditional value.
inline Z possible_prefix(AffineRange difference, std::string predicate,
                         const Z &count) {
  auto clip = [&](Z end) -> Z {
    if (end < 0) return Z(0);
    return end > count ? count : end;
  };
  if (predicate == "le" || predicate == "lt") {
    difference = scale(std::move(difference), Q(-1));
    predicate = predicate == "le" ? "ge" : "gt";
  }
  const Q &c = difference.slope;
  const Q &lo = difference.low, &hi = difference.high;
  if (predicate == "ge" || predicate == "gt") {
    if (c < 0) {
      const Q cutoff = hi / -c;
      return clip(predicate == "ge" ? Z(floorq(cutoff) + 1)
                                     : Z(-floorq(-cutoff)));
    }
    // A nondecreasing active condition has no removable suffix unless the
    // entire domain is inactive. The endpoint is the largest possible value.
    const Q last = hi + c * Q(count - 1);
    if (predicate == "ge" ? last < 0 : last <= 0) return Z(0);
    return count;
  }
  if (predicate == "eq") {
    if (c > 0) return clip(Z(floorq(-lo / c) + 1));
    if (c < 0) return clip(Z(floorq(hi / -c) + 1));
    return lo > 0 || hi < 0 ? Z(0) : count;
  }
  if (predicate == "ne" && c == 0 && lo == 0 && hi == 0) return Z(0);
  return count;
}
} // namespace rg::masked_sum_detail

namespace rg {
// A local ideal-real Sum identity, shared by producer/rebuild consumers.
// Keeps the original seed and body unchanged; only the returned count may be
// smaller. The caller owns capture-safe binder/domain reconstruction and any
// cache keyed by (body, axis, count). No Arena nodes or assumptions are added.
inline Z masked_sum_prefix(Arena &a, Id body, Id axis, const Z &count) {
  ++a.stats["masked_sum_prefix_queries"];
  if (count <= 0 || body >= a.nodes.size() || axis >= a.nodes.size()) return count;
  const auto &variable = a.nodes[axis];
  if (variable.op != "var" || variable.data.size() != 2 || !variable.bounds ||
      variable.data[0].denominator() != 1 || variable.data[1].denominator() != 1 ||
      variable.bounds->first.denominator() != 1 || variable.bounds->second.denominator() != 1 ||
      variable.data[0] > 0 || variable.data[1] < Q(count - 1) ||
      variable.bounds->first > 0 || variable.bounds->second < Q(count - 1)) return count;
  if (body == a.zero) return Z(0);
  const auto &node = a.nodes[body];
  if (!node.facts.finite) return count;
  std::vector<Id> candidates;
  if (node.op == "select") candidates.push_back(body);
  else if (node.op == "mul" && node.data.size() == 1 && node.args.size() <= 8) {
    if (std::any_of(node.args.begin(), node.args.end(),
                    [&](Id child) { return !a.nodes[child].facts.finite; })) return count;
    if (node.data[0] == 0 || std::find(node.args.begin(), node.args.end(), a.zero) != node.args.end())
      return Z(0);
    for (Id child : node.args) if (a.nodes[child].op == "select") candidates.push_back(child);
  } else return count;

  masked_sum_detail::AffineRangeReader reader(a, axis);
  Z result = count;
  for (Id candidate : candidates) {
    a.check();
    const auto &mask = a.nodes[candidate];
    if (mask.args.size() != 3) continue;
    const bool zero_true = mask.args[1] == a.zero, zero_false = mask.args[2] == a.zero;
    if (zero_true && zero_false) return Z(0);
    if (!zero_true && !zero_false) continue;
    const auto &predicate = a.nodes[mask.args[0]];
    if (predicate.op != "compare" || predicate.args.size() != 2 || !predicate.data.empty()) continue;
    const std::string active = zero_true ? masked_sum_detail::inverse(predicate.tag) : predicate.tag;
    if (active != "lt" && active != "le" && active != "gt" && active != "ge" &&
        active != "eq" && active != "ne") continue;
    const auto left = reader.read(predicate.args[0]), right = reader.read(predicate.args[1]);
    if (!left || !right) continue;
    masked_sum_detail::AffineRange difference{left->slope - right->slope,
                                            left->low - right->high,
                                            left->high - right->low};
    result = std::min(result, masked_sum_detail::possible_prefix(std::move(difference), active, count));
    if (result == 0) break;
  }
  return result;
}
} // namespace rg
