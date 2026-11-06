#pragma once
#include "core.hpp"

namespace rg {
// An exact congruence context for optional local rewrites. When both sides
// are products (or sums) whose operands agree except for the remaining
// factors (addends) of each side, the pair of remainders is the place where
// they differ: if a rewrite proves left' = left and right' = right inside it,
// rebuilding the same shell around them gives equal outputs. Peeling stops
// when the left (reference) remainder is a Fold (Sum or Max) or a Select,
// which the projection stage recognizes itself; the right side is not tested.
// O(depth * arity) per output; no node is visited twice.
class CommonShell {
  struct Layer {
    bool product;
    Q scalar;                                  // product: common scalar
    std::vector<std::pair<Id, Q>> common;      // shared factors or (addend, coefficient)
  };
  std::vector<Layer> layers;
  Id left_root = invalid, right_root = invalid;
  static constexpr size_t max_depth = 8, max_arity = 64;

  // Multiset difference of two operand lists; empty common part is no layer.
  static bool split(const std::vector<std::pair<Id, Q>> &l, const std::vector<std::pair<Id, Q>> &r,
                    std::vector<std::pair<Id, Q>> &common, std::vector<std::pair<Id, Q>> &left,
                    std::vector<std::pair<Id, Q>> &right) {
    auto x = l, y = r;
    std::sort(x.begin(), x.end()); std::sort(y.begin(), y.end());
    std::set_intersection(x.begin(), x.end(), y.begin(), y.end(), std::back_inserter(common));
    std::set_difference(x.begin(), x.end(), common.begin(), common.end(), std::back_inserter(left));
    std::set_difference(y.begin(), y.end(), common.begin(), common.end(), std::back_inserter(right));
    return !common.empty() && !left.empty() && !right.empty();
  }
  static Id build(Arena &a, const Layer &layer, const std::vector<std::pair<Id, Q>> &rest) {
    std::vector<Id> xs;
    if (layer.product) {
      xs.push_back(a.constant(layer.scalar));
      for (const auto &[id, unused] : layer.common) { (void)unused; xs.push_back(id); }
      for (const auto &[id, unused] : rest) { (void)unused; xs.push_back(id); }
      return a.mul(std::move(xs));
    }
    for (const auto *part : {&layer.common, &rest})
      for (const auto &[id, c] : *part) xs.push_back(a.mul({a.constant(c), id}));
    return a.add(std::move(xs));
  }
  Id rebuild(Arena &a, Id inner, Id peeled, Id root) const {
    if (inner == peeled) return root;
    for (size_t i = layers.size(); i-- > 0;) inner = build(a, layers[i], {{inner, Q(1)}});
    return inner;
  }

public:
  Id left = invalid, right = invalid;
  bool peeled() const { return !layers.empty(); }
  static CommonShell of(Arena &a, Id l, Id r) {
    CommonShell s; s.left_root = s.left = l; s.right_root = s.right = r;
    for (size_t depth = 0; depth < max_depth; ++depth) {
      const auto &x = a.nodes[s.left], &y = a.nodes[s.right];
      if (x.op != y.op || (x.op != "mul" && x.op != "add") || !x.facts.finite || !y.facts.finite ||
          x.args.size() > max_arity || y.args.size() > max_arity) break;
      const bool product = x.op == "mul";
      if (product && (x.data.size() != 1 || y.data.size() != 1 || x.data[0] != y.data[0])) break;
      if (!product && (x.data.size() != x.args.size() || y.data.size() != y.args.size())) break;
      std::vector<std::pair<Id, Q>> xl, yl;
      for (size_t i = 0; i < x.args.size(); ++i) xl.emplace_back(x.args[i], product ? Q(1) : x.data[i]);
      for (size_t i = 0; i < y.args.size(); ++i) yl.emplace_back(y.args[i], product ? Q(1) : y.data[i]);
      Layer layer{product, product ? x.data[0] : Q(1), {}};
      std::vector<std::pair<Id, Q>> lrest, rrest;
      if (!split(xl, yl, layer.common, lrest, rrest)) break;
      // The remaining operands of each side form its inner value (a product
      // with a unit scalar, or a sum).
      auto inner = [&](const std::vector<std::pair<Id, Q>> &rest) {
        if (rest.size() == 1 && rest[0].second == 1) return rest[0].first;
        return build(a, Layer{product, Q(1), {}}, rest);
      };
      s.left = inner(lrest); s.right = inner(rrest);
      s.layers.push_back(std::move(layer));
      const auto &op = a.nodes[s.left].op;
      if (op == "fold" || op == "select") break;
    }
    return s;
  }
  Id rebuild_left(Arena &a, Id inner) const { return rebuild(a, inner, left, left_root); }
  Id rebuild_right(Arena &a, Id inner) const { return rebuild(a, inner, right, right_root); }
};
} // namespace rg
