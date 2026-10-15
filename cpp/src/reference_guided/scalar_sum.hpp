#pragma once
#include "core.hpp"

namespace rg::scalar_sum_detail {
struct Affine {
  Q slope = 0, offset = 0;
};

// This is a deliberately small expression recognizer, not a domain solver.
// Both work and recursion depth depend on the expression, never the Fold count.
struct AffineReader {
  Arena &a;
  Id binder;
  size_t remaining = 128;
  std::unordered_map<Id, std::optional<Affine>> memo;

  AffineReader(Arena &arena, Id variable) : a(arena), binder(variable) {}

  std::optional<Affine> read(Id id, unsigned depth = 0) {
    if (remaining == 0 || depth > 32)
      return {};
    --remaining;
    a.check();
    if (auto it = memo.find(id); it != memo.end())
      return it->second;
    const Node &n = a.nodes[id];
    std::optional<Affine> result;
    if (!n.facts.finite)
      return {};
    if (id == binder)
      result = Affine{Q(1), Q(0)};
    else if (n.op == "const" && n.data.size() == 1)
      result = Affine{Q(0), n.data[0]};
    else if (n.op == "add" && n.args.size() == n.data.size() &&
             n.args.size() <= remaining) {
      Affine sum;
      bool valid = true;
      for (size_t i = 0; i < n.args.size(); ++i) {
        auto term = read(n.args[i], depth + 1);
        if (!term) { valid = false; break; }
        sum.slope += n.data[i] * term->slope;
        sum.offset += n.data[i] * term->offset;
      }
      if (valid) result = std::move(sum);
    } else if (n.op == "mul" && n.data.size() == 1 &&
               n.args.size() <= remaining) {
      Affine product{Q(0), n.data[0]};
      bool valid = true;
      for (Id factor : n.args) {
        auto term = read(factor, depth + 1);
        if (!term || (product.slope != 0 && term->slope != 0)) {
          valid = false;
          break;
        }
        product.slope = product.slope * term->offset +
                        product.offset * term->slope;
        product.offset *= term->offset;
      }
      if (valid) result = std::move(product);
    }
    memo.emplace(id, result);
    return result;
  }
};

inline std::optional<Z> predicate_count(Arena &a, Id comparison,
                                       Id binder, const Z &count) {
  const Node &n = a.nodes[comparison];
  if (n.op != "compare" || n.args.size() != 2)
    return {};
  std::string predicate = n.tag;
  if (predicate != "eq" && predicate != "ne" && predicate != "lt" &&
      predicate != "le" && predicate != "gt" && predicate != "ge")
    return {};
  AffineReader reader{a, binder};
  auto lhs = reader.read(n.args[0]), rhs = reader.read(n.args[1]);
  if (!lhs || !rhs)
    return {};
  Q slope = lhs->slope - rhs->slope;
  Q offset = lhs->offset - rhs->offset;
  if (slope == 0) {
    bool truth = predicate == "eq" ? offset == 0
               : predicate == "ne" ? offset != 0
               : predicate == "lt" ? offset < 0
               : predicate == "le" ? offset <= 0
               : predicate == "gt" ? offset > 0 : offset >= 0;
    return truth ? count : Z(0);
  }
  const Q cutoff = -offset / slope;
  if (slope < 0) {
    if (predicate == "lt") predicate = "gt";
    else if (predicate == "le") predicate = "ge";
    else if (predicate == "gt") predicate = "lt";
    else if (predicate == "ge") predicate = "le";
  }
  auto clip = [&](Z end) -> Z {
    if (end < 0) return Z(0);
    if (end > count) return count;
    return end;
  };
  if (predicate == "eq" || predicate == "ne") {
    const bool included = cutoff.denominator() == 1 &&
                          cutoff.numerator() >= 0 &&
                          cutoff.numerator() < count;
    const Z equal = included ? Z(1) : Z(0);
    return predicate == "eq" ? equal : Z(count - equal);
  }
  // Integer i < cutoff ends at ceil(cutoff); i <= cutoff ends at floor+1.
  const Z less = clip(Z(-floorq(-cutoff)));
  const Z less_equal = clip(Z(floorq(cutoff) + 1));
  if (predicate == "lt") return less;
  if (predicate == "le") return less_equal;
  if (predicate == "gt") return Z(count - less_equal);
  return Z(count - less);
}
} // namespace rg::scalar_sum_detail

namespace rg {
// Exact ideal-real Sum: seed + sum_{i=0}^{count-1} body[i/binder].
// The seed occurs once, including when the body has zero satisfying instances.
// Nonfinite bodies are declined before any zero multiplication or cancellation.
// A nonempty Fold requires the binder's declared integer interval to contain
// the entire Fold domain: earlier Arena rewrites may have used that interval.
inline std::optional<Id> simplify_scalar_sum(Arena &a, Id seed, Id body,
                                            Id binder, const Z &count) {
  if (count < 0)
    return {};
  if (count == 0) {
    ++a.stats["scalar_sum_empty_folds"];
    return seed;
  }
  const Node &variable = a.nodes[binder];
  if (variable.op != "var" || variable.data.size() != 2 ||
      variable.data[0].denominator() != 1 ||
      variable.data[1].denominator() != 1 || variable.data[0] > 0 ||
      variable.data[1] < Q(count - 1))
    return {};
  if (!a.nodes[body].facts.finite)
    return {};
  if (!a.depends(body, binder)) {
    ++a.stats["scalar_sum_constant_folds"];
    return a.add({seed, a.mul({a.constant(Q(count)), body})});
  }
  Id comparison = body;
  Q scalar(1);
  std::vector<Id> invariant;
  const Node &n = a.nodes[body];
  if (n.op == "mul") {
    if (n.data.size() != 1 || n.args.size() > 128)
      return {};
    scalar = n.data[0];
    comparison = invalid;
    for (Id factor : n.args) {
      a.check();
      if (!a.nodes[factor].facts.finite)
        return {};
      if (!a.depends(factor, binder))
        invariant.push_back(factor);
      else if (comparison == invalid)
        comparison = factor;
      else
        return {}; // A binder-dependent coefficient is not invariant.
    }
    if (comparison == invalid)
      return {};
  }
  auto satisfying = scalar_sum_detail::predicate_count(a, comparison, binder,
                                                       count);
  if (!satisfying)
    return {};
  invariant.push_back(a.constant(scalar * Q(*satisfying)));
  ++a.stats["scalar_sum_predicate_folds"];
  return a.add({seed, a.mul(std::move(invariant))});
}
} // namespace rg
