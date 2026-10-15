#pragma once
#include "core.hpp"
#include <array>

namespace rg {
// Exact coordinate constraints, independent of tensor contents. The data is
// [axis-slot, a, b, axis-low, axis-high, (operation, positive divisor)...],
// where operation 0 is floor division and 1 is Euclidean remainder, applied
// from inner to outer to a*k+b. Only the matched coordinate is replaced;
// surrounding read identities, storage types and other coordinates still match.
struct CoordinateRelations {
  static constexpr size_t max_steps = 4, max_groups = 8, max_constraints = 32;
  static constexpr size_t integer_nodes = 64, integer_edges = 128;

  static bool integer(Arena &a, Id root) {
    std::vector<Id> todo{root}; std::unordered_set<Id> seen;
    size_t edges = 0;
    while (!todo.empty()) {
      a.check(); Id id = todo.back(); todo.pop_back();
      if (!seen.insert(id).second) continue;
      if (seen.size() > integer_nodes) return false;
      const auto &n = a.nodes[id];
      if (!n.facts.finite) return false;
      if (n.op == "const") {
        if (n.data.size() != 1 || n.data[0].denominator() != 1) return false;
      } else if (n.op == "var") {
        // Arena::var is the integer index opcode. Bounds do not turn an atom
        // or a numerical read into an integer.
        if (!n.bounds || n.bounds->first.denominator() != 1 ||
            n.bounds->second.denominator() != 1) return false;
      } else if (n.op == "readbits" || n.op == "packedbits" || n.op == "extractbits" ||
                 n.op == "word" || n.op == "signed" || n.op == "floor" || n.op == "word_shr" || n.op == "word_or") {
        continue;
      } else if (n.op == "read") {
        auto separator = n.tag.rfind('\n');
        if (separator == std::string::npos) return false;
        std::string type = n.tag.substr(separator + 1);
        if (type.size() < 2 || (type[0] != 'u' && type[0] != 's') ||
            !std::all_of(type.begin() + 1, type.end(),
                         [](char c) { return c >= '0' && c <= '9'; })) return false;
      } else if (n.op == "idiv") {
        if (n.args.size() != 1 || n.data.size() != 1 ||
            n.data[0].denominator() != 1 || n.data[0] <= 0) return false;
        // Floor of a finite real expression is an integer.
        if (!a.nodes[n.args[0]].facts.finite) return false;
      } else if (n.op == "imod" || n.op == "add" || n.op == "mul" || n.op == "xor") {
        if (n.args.size() > 8 || n.args.empty()) return false;
        if (n.op == "imod" && (n.args.size() != 1 || n.data.size() != 1 || n.data[0] <= 0)) return false;
        for (const auto &coefficient : n.data)
          if (coefficient.denominator() != 1) return false;
        edges += n.args.size(); if (edges > integer_edges) return false;
        todo.insert(todo.end(), n.args.begin(), n.args.end());
      } else return false;
    }
    return true;
  }

  // Describe only a small existing idiv/imod chain with a literal affine
  // dividend. Parameters outside that dividend retain ordinary slot matching.
  static std::optional<std::vector<Q>> describe(Arena &a, Id root, Id axis) {
    if (a.nodes[root].op != "idiv" && a.nodes[root].op != "imod") return {};
    const auto &v = a.nodes[axis];
    if (v.op != "var" || !v.bounds || v.bounds->first.denominator() != 1 ||
        v.bounds->second.denominator() != 1 || v.bounds->first > v.bounds->second) return {};
    Q low = v.bounds->first, high = v.bounds->second;
    std::vector<std::pair<Q, Q>> reverse_steps;
    Id base = root;
    while (a.nodes[base].op == "idiv" || a.nodes[base].op == "imod") {
      const auto &n = a.nodes[base];
      if (reverse_steps.size() == max_steps || n.args.size() != 1 || n.data.size() != 1 ||
          n.data[0].denominator() != 1 || n.data[0] <= 0) return {};
      reverse_steps.emplace_back(Q(n.op == "imod" ? 1 : 0), n.data[0]);
      base = n.args[0];
    }
    if (a.nodes[base].args.size() > 8) return {};
    Q coefficient(0), offset(0);
    for (const auto &[term, scale] : a.terms(base)) {
      if (scale.denominator() != 1) return {};
      if (term == axis) coefficient += scale;
      else if (term == a.one) offset += scale;
      else return {};
    }
    if (coefficient == 0) return {};
    std::vector<Q> data{Q(0), coefficient, offset, low, high};
    for (auto it = reverse_steps.rbegin(); it != reverse_steps.rend(); ++it) {
      data.push_back(it->first); data.push_back(it->second);
    }
    return data;
  }

  struct SumRecipe {
    std::vector<std::vector<Q>> relations;
    std::vector<Q> coefficients;
    Q constant = 0;
    Id residual = invalid;
  };
  // A short sum of the existing exact quotient/remainder chains. All
  // axis-independent, nonliteral terms stay together as one parameter; this
  // recipe never provides its own reduction-axis anchor.
  static std::optional<SumRecipe> describe_sum(Arena &a, Id root, Id axis) {
    const auto &node = a.nodes[root];
    if ((node.op != "add" && node.op != "mul") || node.args.size() > 4 ||
        !node.facts.finite) return {};
    auto terms = a.terms(root);
    if (terms.empty() || terms.size() > 4) return {};
    SumRecipe recipe; std::vector<Id> residual;
    for (const auto &[term, coefficient] : terms) {
      if (coefficient.denominator() != 1) return {};
      if (term == a.one) recipe.constant += coefficient;
      else if (!a.depends(term, axis)) {
        if (!a.nodes[term].facts.finite || !integer(a, term)) return {};
        residual.push_back(a.mul({a.constant(coefficient), term}));
      } else {
        auto relation = describe(a, term, axis);
        if (!relation) return {};
        recipe.relations.push_back(std::move(*relation));
        recipe.coefficients.push_back(coefficient);
      }
    }
    if (recipe.relations.empty()) return {};
    if (!residual.empty()) recipe.residual = a.add(std::move(residual));
    return recipe;
  }

  static Id evaluate(Arena &a, Id index, const std::vector<Q> &data) {
    Id expected = a.add({a.mul({a.constant(data[1]), index}), a.constant(data[2])});
    for (size_t step = 5; step < data.size(); step += 2)
      expected = data[step] == 0 ? a.idiv(expected, data[step + 1].numerator())
                                : a.imod(expected, data[step + 1].numerator());
    return expected;
  }

  // Pattern data: [axis-slot, residual-slot or -1, literal offset,
  //                coefficient for each chain argument]. Every argument is
  // an ordinary derived_coordinate description with the same axis and scope.
  static bool resolve_sum(Arena &a, Arena &patterns, Id pattern, Id actual,
                          std::vector<Id> &captures, uint64_t &work) {
    a.check(); if (++work > 1000000) throw Unsupported("local pattern matching work budget exceeded");
    const auto &node = patterns.nodes[pattern];
    if (node.op != "coordinate_sum" || node.args.empty() || node.args.size() > 4 ||
        node.data.size() != node.args.size() + 3 || !integer(a, actual)) return false;
    const auto &data = node.data;
    for (const auto &value : data) if (value.denominator() != 1) return false;
    if (data[0] < 0 || data[0] > 1024 || data[1] < -1 || data[1] > 1024) return false;
    size_t axis = data[0].numerator().convert_to<size_t>();
    if (axis >= captures.size() || captures[axis] == invalid ||
        a.nodes[captures[axis]].args.size() > 8) return false;
    std::vector<Id> terms{a.constant(data[2])};
    std::optional<std::pair<Q, Q>> scope;
    for (size_t i = 0; i < node.args.size(); ++i) {
      a.check(); if (++work > 1000000) throw Unsupported("local pattern matching work budget exceeded");
      const auto &relation = patterns.nodes[node.args[i]];
      if (relation.op != "derived_coordinate" || !valid(relation.data) ||
          relation.data[0] != data[0] || !within_axis(a, captures[axis], relation.data)) return false;
      auto domain = std::pair(relation.data[3], relation.data[4]);
      if (scope && *scope != domain) return false;
      scope = domain;
      terms.push_back(a.mul({a.constant(data[i + 3]), evaluate(a, captures[axis], relation.data)}));
    }
    Id expected = a.add(std::move(terms));
    if (data[1] >= 0) {
      size_t slot = data[1].numerator().convert_to<size_t>();
      if (slot == axis || a.nodes[actual].args.size() > 8) return false;
      Id residual = a.sub(actual, expected);
      if (!integer(a, residual)) return false;
      if (slot >= captures.size()) captures.resize(slot + 1, invalid);
      if (captures[slot] != invalid && captures[slot] != residual) return false;
      captures[slot] = residual;
      expected = a.add({expected, residual});
    }
    if (expected != actual) return false;
    ++a.stats["coordinate_sum_constraints_checked"];
    return true;
  }

  static bool valid(const std::vector<Q> &data) {
    if (data.size() < 7 || data.size() > 5 + 2 * max_steps || data.size() % 2 != 1 ||
        data[0].denominator() != 1 || data[0] < 0 || data[0] > Q(1024) ||
        data[1].denominator() != 1 || data[1] == 0 || data[2].denominator() != 1 ||
        data[3].denominator() != 1 || data[4].denominator() != 1 || data[3] > data[4]) return false;
    for (size_t i = 5; i < data.size(); i += 2)
      if ((data[i] != 0 && data[i] != 1) || data[i + 1].denominator() != 1 || data[i + 1] <= 0) return false;
    return true;
  }
  static std::vector<Q> pair_key(const std::vector<Q> &data) {
    // A one-step pair shares axis, literal affine dividend, scope and divisor.
    return {data[0], data[1], data[2], data[3], data[4], data[6]};
  }
  static bool within_axis(Arena &a, Id value, const std::vector<Q> &data) {
    const auto &n = a.nodes[value];
    return n.bounds && n.bounds->first >= data[3] && n.bounds->second <= data[4] &&
           integer(a, value);
  }
  static bool resolve(Arena &a, Arena &patterns,
                      const std::vector<std::pair<Id, Id>> &constraints,
                      std::vector<Id> &captures, uint64_t &work) {
    if (constraints.size() > max_constraints) return false;
    std::map<std::vector<Q>, std::array<Id, 2>> pairs;
    for (const auto &[pattern, actual] : constraints) {
      a.check(); if (++work > 1000000) throw Unsupported("local pattern matching work budget exceeded");
      const auto &data = patterns.nodes[pattern].data;
      if (!valid(data) || !integer(a, actual)) return false;
      if (data.size() != 7) continue;
      auto [it, fresh] = pairs.try_emplace(pair_key(data), std::array<Id, 2>{invalid, invalid});
      if (pairs.size() > max_groups) return false;
      size_t kind = data[5].numerator().convert_to<size_t>();
      Id &stored = it->second[kind];
      if (stored != invalid && stored != actual) return false;
      stored = actual;
    }
    // A complete pair may anchor an otherwise uncaptured axis. Restrict the
    // inverse to unit stride; arbitrary a would additionally need divisibility.
    for (const auto &[key, values] : pairs) {
      size_t axis = key[0].numerator().convert_to<size_t>();
      if (axis < captures.size() && captures[axis] != invalid) continue;
      if (values[0] == invalid || values[1] == invalid || (key[1] != 1 && key[1] != -1)) continue;
      const auto remainder_range = a.nodes[values[1]].bounds;
      if (!remainder_range || remainder_range->first < 0 || remainder_range->second >= key[5]) return false;
      Id dividend = a.add({a.mul({a.constant(key[5]), values[0]}), values[1]});
      Id index = a.mul({a.constant(Q(1) / key[1]), a.sub(dividend, a.constant(key[2]))});
      if (!within_axis(a, index, key)) return false;
      if (axis >= captures.size()) captures.resize(axis + 1, invalid);
      captures[axis] = index;
      ++a.stats["quotient_remainder_coordinate_anchors"];
    }
    // Re-evaluate every original relation, even after a different pair or an
    // ordinary coordinate anchored the same axis. No quotient/remainder slot
    // becomes an independent fixed parameter and no constraint is discarded.
    for (const auto &[pattern, actual] : constraints) {
      a.check(); if (++work > 1000000) throw Unsupported("local pattern matching work budget exceeded");
      const auto &data = patterns.nodes[pattern].data;
      size_t axis = data[0].numerator().convert_to<size_t>();
      if (axis >= captures.size() || captures[axis] == invalid ||
          !within_axis(a, captures[axis], data)) return false;
      if (a.nodes[captures[axis]].args.size() > 8) return false;
      Id expected = evaluate(a, captures[axis], data);
      if (expected != actual) return false;
      ++a.stats["derived_coordinate_constraints_checked"];
    }
    return true;
  }
};
} // namespace rg
