#pragma once
#include "checker.hpp"
#include "importer.hpp"

namespace rg {
// A fixed ideal-real Welford certificate. This does not make a kernel-name or
// output-count assumption. The reference supplies the input recipe/domain;
// raw weighted means supply a checked merge tree; its local M2 corrections
// must exactly account for the optimized expression before a proof is issued.
struct WelfordPatternPass {
  Importer &left, &right;
  Arena &a;
  J receipt = J::object();
  using Env = Importer::Env;
  static constexpr size_t max_work = 500000, max_candidates = 64;
  static constexpr unsigned max_depth = 128;
  struct Miss {};
  struct Axis { Id var; Z count; };
  struct Reduction {
    Edge body;
    Env env;
    std::vector<Axis> axes;
    Z count = 1;
    explicit Reduction(Edge e, Env v) : body(std::move(e)), env(std::move(v)) {}
  };
  struct Recipe {
    std::string tag;
    std::vector<Id> fixed;
    // A positive entry means this canonical read coordinate is reduced.
    std::vector<Z> extents;
    Z count = 1;
    bool operator==(const Recipe &o) const {
      return tag == o.tag && fixed == o.fixed && extents == o.extents && count == o.count;
    }
  };
  enum Projection { Mean, Variance, Std, InvStd };
  struct Spec { Recipe recipe; Projection projection = Mean; Id epsilon = invalid; };
  struct State {
    Z count;
    Id mean, first = invalid, second = invalid, index = invalid;
    // One valid local contribution to M2; children retain theirs persistently.
    Id correction = invalid;
  };
  struct Scale {
    Edge core;
    Q coefficient;
    Scale(Edge e, Q q) : core(std::move(e)), coefficient(std::move(q)) {}
  };
  using Quadratic = std::map<std::vector<Id>, Q>;
  Env outenv;
  std::optional<Spec> spec;
  std::deque<State> states;
  std::map<std::tuple<Id, std::string, Z>, Id> state_cache;
  std::map<std::pair<Id, std::string>, Scale> scale_cache;
  std::map<Id, std::pair<Id, Q>> primitive_cache;
  size_t work = 0, local_merges = 0, local_updates = 0, next_domain = 0;

  ExponentialPolynomial *scalar_engine = nullptr;
  WelfordPatternPass(Importer &l, Importer &r, ExponentialPolynomial *ep=nullptr) : left(l), right(r), a(l.a), scalar_engine(ep) {}
  static Edge raw(Id n) { return Edge(J{{"node", n}}); }
  void tick() {
    a.check();
    if (++work > max_work) throw Miss{};
  }
  static bool arithmetic(const RawNode &n, const char *name, size_t arity) {
    return n.kind == "Operation" && n.name == name && floating(n.type) &&
           (n.attributes.empty() || n.attributes == "ideal_arithmetic") && n.operands.size() == arity;
  }
  Id scalar(Importer &im, const Edge &e, const Env &env) {
    tick();
    return im.edge(im.import(e.node, env), e);
  }
  Id scalar(const Edge &e) { return scalar(right, e, outenv); }
  Reduction reduction(Edge e, const Env &base) {
    Reduction r(e, base);
    std::set<Id> levels;
    while (left.ns.at(r.body.node).kind == "Fold") {
      tick();
      const auto &n = left.ns.at(r.body.node);
      if (!r.body.conversion.empty() || !floating(n.type) || n.reducer != "Sum" || n.count == 0 ||
          n.operands.size() != 2 || !levels.insert(n.level).second || r.axes.size() == 4 ||
          scalar(left, n.operands[0], r.env) != a.zero) throw Miss{};
      Id v = n.count == 1 ? a.zero :
          a.var("welford:domain:" + std::to_string(next_domain++) + ":" + std::to_string(n.level),
                0, Q(n.count) - Q(1));
      r.env[n.level] = v;
      r.axes.push_back({v, Z(n.count)});
      r.count *= n.count;
      r.body = n.operands[1];
    }
    if (r.axes.empty()) throw Miss{};
    return r;
  }
  Recipe recipe(const Reduction &r, const Edge &sample) {
    Id value = scalar(left, sample, r.env);
    const auto &read = a.nodes[value];
    if (read.op != "read" || !read.facts.finite || read.tag.find("real:byte-address:") != std::string::npos)
      throw Miss{};
    Recipe p;
    p.tag = read.tag;
    p.count = r.count;
    p.fixed = read.args;
    p.extents.assign(read.args.size(), Z(0));
    for (const auto &axis : r.axes) {
      // Count-one axes may already have become literal zero. They add no
      // coverage choices and cannot establish an otherwise missing binder.
      if (axis.count == 1) continue;
      size_t slot = read.args.size();
      for (size_t j = 0; j < read.args.size(); ++j) {
        if (read.args[j] == axis.var) {
          if (slot != read.args.size()) throw Miss{};
          slot = j;
        } else if (a.depends(read.args[j], axis.var)) throw Miss{};
      }
      if (slot == read.args.size() || p.extents[slot] != 0) throw Miss{};
      p.extents[slot] = axis.count;
      p.fixed[slot] = invalid;
    }
    Z size = 1;
    for (const Z &n : p.extents) if (n != 0) size *= n;
    if (size != p.count) throw Miss{};
    return p;
  }
  std::pair<Reduction, Q> divided_sum(const Edge &e, const Env &base) {
    const auto &n = left.ns.at(e.node);
    if (!e.conversion.empty() || !arithmetic(n, "Divide", 2)) throw Miss{};
    auto denominator = a.value(scalar(left, n.operands[1], base));
    if (!denominator || denominator->denominator() != 1 || *denominator <= 0) throw Miss{};
    Reduction r = reduction(n.operands[0], base);
    if (*denominator != Q(r.count)) throw Miss{};
    return {std::move(r), *denominator};
  }
  Recipe reference_mean(const Edge &e, const Env &base) {
    auto [r, n] = divided_sum(e, base);
    return recipe(r, r.body);
  }
  Spec reference_variance(const Edge &e, const Env &base) {
    auto [r, n] = divided_sum(e, base);
    const auto &square = left.ns.at(r.body.node);
    if (!r.body.conversion.empty() || !arithmetic(square, "Multiply", 2) ||
        square.operands[0].node != square.operands[1].node ||
        square.operands[0].conversion != square.operands[1].conversion) throw Miss{};
    const auto &delta = left.ns.at(square.operands[0].node);
    if (!square.operands[0].conversion.empty() || !arithmetic(delta, "Subtract", 2)) throw Miss{};
    Recipe p = recipe(r, delta.operands[0]);
    Recipe mean = reference_mean(delta.operands[1], base);
    if (!(p == mean)) throw Miss{};
    return {std::move(p), Variance, a.zero};
  }
  Spec reference(const J &family) {
    Edge e(family.at("body"));
    // prove() separately requires the complete output conversion to preserve
    // this imported numeric value; bit reinterpretation is not assumed equal.
    e.conversion.clear();
    const auto &n = left.ns.at(e.node);
    if (arithmetic(n, "Divide", 2)) {
      try { return {reference_mean(e, outenv), Mean, a.zero}; }
      catch (const Miss &) { return reference_variance(e, outenv); }
    }
    Projection projection;
    if (arithmetic(n, "Rsqrt", 1) || arithmetic(n, "ReciprocalSqrt", 1)) projection = InvStd;
    else if (arithmetic(n, "Sqrt", 1)) projection = Std;
    else throw Miss{};
    const Edge &argument = n.operands[0];
    const auto &sum = left.ns.at(argument.node);
    if (!argument.conversion.empty() || !arithmetic(sum, "Add", 2)) throw Miss{};
    for (unsigned j = 0; j < 2; ++j) {
      try {
        Spec s = reference_variance(sum.operands[j], outenv);
        s.epsilon = scalar(left, sum.operands[1 - j], outenv);
        // This first fixed fragment deliberately excludes InvStd's zero/zero
        // special case. Positivity is a proved premise, not an added one.
        if (!a.nodes[s.epsilon].facts.finite || !a.nodes[s.epsilon].facts.pos) throw Miss{};
        s.projection = projection;
        return s;
      } catch (const Miss &) {}
    }
    throw Miss{};
  }
  std::optional<Id> sample_index(Id value) {
    const auto &p = spec->recipe;
    const auto &n = a.nodes[value];
    if (n.op != "read" || n.tag != p.tag || !n.facts.finite || n.args.size() != p.fixed.size()) return {};
    Z index = 0;
    for (size_t j = 0; j < n.args.size(); ++j) {
      if (p.extents[j] == 0) {
        if (n.args[j] != p.fixed[j]) return {};
      } else {
        auto coordinate = a.value(n.args[j]);
        if (!coordinate || coordinate->denominator() != 1 || *coordinate < 0 ||
            *coordinate >= Q(p.extents[j])) return {};
        index = index * p.extents[j] + coordinate->numerator();
      }
    }
    return a.constant(Q(index));
  }
  Scale split_scale(const Edge &e) {
    tick();
    auto key = std::pair(e.node, e.conversion);
    if (auto it = scale_cache.find(key); it != scale_cache.end()) return it->second;
    const auto &n = right.ns.at(e.node);
    Scale result(e, Q(1));
    // Peel exactly one coefficient boundary. Peeling recursively through a
    // child's mean would cancel n_child * (1/n_child) and lose its state
    // boundary, recreating the scalar-DAG problem this pass is solving.
    if (arithmetic(n, "Multiply", 2)) {
      for (unsigned j = 0; j < 2; ++j) {
        auto c = a.value(scalar(n.operands[j]));
        if (!c) continue;
        result = Scale(n.operands[1 - j], *c);
        break;
      }
    } else if (arithmetic(n, "Divide", 2)) {
      auto c = a.value(scalar(n.operands[1]));
      if (c && *c != 0) {
        result = Scale(n.operands[0], Q(1) / *c);
      }
    } else if (scalar(e) == a.zero) {
      result.coefficient = Q(0);
    }
    // Conversion/saturation/bit transport cannot be silently crossed.
    if (scalar(e) != a.mul({a.constant(result.coefficient), scalar(result.core)})) throw Miss{};
    scale_cache.emplace(std::move(key), result);
    return result;
  }
  Id append(State s) {
    if (states.size() >= max_work) throw Miss{};
    states.push_back(std::move(s));
    return Id(states.size() - 1);
  }
  std::optional<std::array<Scale, 2>> weighted_add(const Edge &e) {
    const auto &n = right.ns.at(e.node);
    if (arithmetic(n, "Add", 2))
      return std::array<Scale, 2>{split_scale(n.operands[0]), split_scale(n.operands[1])};
    // Preserve the exported FMA and rounding metadata. Only the checker's
    // established ideal-real interpretation may expose this local sum view.
    if (!right.ideal || n.kind != "Operation" || n.name != "Fma" ||
        !floating(n.type) || n.operands.size() != 3) return {};
    for (unsigned j = 0; j < 2; ++j) {
      auto coefficient = a.value(scalar(n.operands[j]));
      if (!coefficient) continue;
      Id expected = a.add({a.mul({a.constant(*coefficient), scalar(n.operands[1-j])}), scalar(n.operands[2])});
      if (scalar(e) != expected) return {};
      ++a.stats["welford_ideal_fma_boundaries"];
      return std::array<Scale, 2>{Scale(n.operands[1-j], *coefficient), split_scale(n.operands[2])};
    }
    return {};
  }
  Id state(const Edge &e, const Z &count, unsigned depth = 0) {
    tick();
    if (depth > max_depth || count < 0 || count > spec->recipe.count) return invalid;
    auto key = std::tuple(e.node, e.conversion, count);
    if (auto it = state_cache.find(key); it != state_cache.end()) return it->second;
    Id result = invalid, value = scalar(e);
    if (!a.nodes[value].facts.finite) return invalid;
    if (count == 0 && value == a.zero) result = append({0, a.zero});
    else if (count == 1) {
      if (auto index = sample_index(value)) result = append({1, value, invalid, invalid, *index});
    }
    if (result == invalid && count > 0) {
      Scale outer = split_scale(e);
      if (auto sum = weighted_add(outer.core); sum && outer.core.conversion.empty()) {
        const Scale &x = (*sum)[0], &y = (*sum)[1];
        Q nx = outer.coefficient * x.coefficient * Q(count);
        Q ny = outer.coefficient * y.coefficient * Q(count);
        if (nx.denominator() == 1 && ny.denominator() == 1 &&
            nx >= 0 && ny >= 0 && nx + ny == Q(count)) {
          // An exactly zero weighted, finite operand contributes no samples;
          // the operand itself need not be an empty-state mean (FMA padding
          // often computes 0 * (unused_mean - mean)).
          Id l = nx == 0 ? append({0, a.zero}) : state(x.core, nx.numerator(), depth + 1);
          Id r = ny == 0 ? append({0, a.zero}) : state(y.core, ny.numerator(), depth + 1);
          if (l != invalid && r != invalid) {
            const State &ls = states[l], &rs = states[r];
            Id expected = a.mul({a.constant(Q(1, count)),
                a.add({a.mul({a.constant(Q(ls.count)), ls.mean}), a.mul({a.constant(Q(rs.count)), rs.mean})})});
            if (expected == value) {
              Id delta = a.sub(ls.mean, rs.mean);
              Id correction = a.mul({a.constant(Q(ls.count * rs.count, count)), delta, delta});
              result = append({count, value, l, r, invalid, correction});
              ++local_merges;
            }
          }
        }
      }
      // Covers both one-sample updates and parallel weighted updates:
      // mu' = muA + (muB-muA) * nB/(nA+nB).
      // The existing M2 comparison checks the complete correction afterward.
      auto update = weighted_add(e);
      if (result == invalid && update && e.conversion.empty()) {
        for (unsigned j = 0; j < 2 && result == invalid; ++j) {
          const Scale &base = (*update)[j], &step = (*update)[1-j];
          Q right_count = step.coefficient * Q(count);
          if (base.coefficient != 1 || right_count.denominator() != 1 ||
              right_count <= 0 || right_count > Q(count)) continue;
          Z nr = right_count.numerator(), nl = count - nr;
          Id previous = state(base.core, nl, depth + 1);
          if (previous == invalid) continue;
          const auto &difference = right.ns.at(step.core.node);
          if (!step.core.conversion.empty() ||
              !(arithmetic(difference, "Subtract", 2) || arithmetic(difference, "Add", 2))) continue;
          for (unsigned k = 0; k < 2 && result == invalid; ++k) {
            Id x = scalar(difference.operands[k]);
            Id other = state(difference.operands[k], nr, depth + 1);
            if (other == invalid) continue;
            Id delta = a.sub(x, states[previous].mean);
            if (scalar(step.core) != delta || value != a.add({states[previous].mean,
                a.mul({a.constant(Q(nr, count)), delta})})) continue;
            Id correction = a.mul({a.constant(Q(nl * nr, count)), delta, delta});
            result = append({count, value, previous, other, invalid, correction});
            ++local_updates;
          }
        }
      }
    }
    state_cache.emplace(std::move(key), result);
    return result;
  }
  bool coverage(Id root) {
    std::vector<Id> todo{root};
    std::vector<Box> boxes;
    while (!todo.empty()) {
      tick();
      const auto &s = states[todo.back()]; todo.pop_back();
      if (s.count == 0) continue;
      if (s.index != invalid) {
        auto coordinate = a.value(s.index);
        if (!coordinate || coordinate->denominator() != 1) return false;
        boxes.push_back({coordinate->numerator(), 1, {}});
      } else {
        if (s.first == invalid || s.second == invalid) return false;
        todo.push_back(s.first); todo.push_back(s.second);
      }
    }
    ++a.stats["welford_coverage_checks"];
    return cover_boxes(boxes, spec->recipe.count, a);
  }
  // Canonicalize only the scale of a linear factor, including its sign. No
  // square is distributed over input variables. Thus (muA-muB)^2 and the
  // reverse orientation have the same exact key without a quadratic expansion.
  std::pair<Id, Q> primitive(Id value) {
    if (auto it = primitive_cache.find(value); it != primitive_cache.end()) return it->second;
    tick();
    if (!a.nodes[value].facts.finite) throw Miss{};
    auto terms = a.terms(value);
    if (terms.empty()) return {a.one, Q(0)};
    Q scale = terms.front().second;
    if (scale == 0) throw Miss{};
    std::vector<Id> normalized;
    normalized.reserve(terms.size());
    for (auto [term, coefficient] : terms) {
      tick();
      normalized.push_back(a.mul({a.constant(coefficient / scale), term}));
    }
    auto result = std::pair(a.add(normalized), scale);
    primitive_cache.emplace(value, result);
    return result;
  }
  void accumulate(Quadratic &out, Id value, Q weight = Q(1)) {
    if (weight == 0 || value == a.zero) return;
    if (!a.nodes[value].facts.finite) throw Miss{};
    for (auto [term, coefficient] : a.terms(value)) {
      tick();
      Q factor = weight * coefficient;
      std::vector<Id> key;
      const auto &n = a.nodes[term];
      std::vector<Id> factors;
      if (n.op == "mul") { factor *= n.data[0]; factors = n.args; }
      else if (auto constant = a.value(term)) factor *= *constant;
      else factors.push_back(term);
      if (factors.size() > 2) throw Miss{};
      for (Id f : factors) {
        auto [basis, scalar] = primitive(f);
        factor *= scalar;
        if (basis != a.one) key.push_back(basis);
      }
      std::sort(key.begin(), key.end());
      out[key] += factor;
      if (out[key] == 0) out.erase(key);
    }
  }
  Quadratic moment(Id root) {
    Quadratic result;
    std::vector<Id> todo{root};
    while (!todo.empty()) {
      tick();
      const auto &s = states[todo.back()]; todo.pop_back();
      if (s.count == 0 || s.index != invalid) continue;
      if (s.correction == invalid || s.first == invalid || s.second == invalid) throw Miss{};
      accumulate(result, s.correction, Q(1, spec->recipe.count));
      todo.push_back(s.first); todo.push_back(s.second);
    }
    if (spec->projection == Std || spec->projection == InvStd) accumulate(result, spec->epsilon);
    return result;
  }
  bool prove(const J &lf, const J &rf, Id lr, Id rr) {
    receipt = J::object();
    // Cheap reference gate: this fragment starts at a numeric mean/variance
    // division or a sqrt projection, and always contains a Sum Fold.
    const auto &root = left.ns.at(input_id(lf.at("body").at("node")));
    if (root.kind != "Operation" || (root.name != "Divide" && root.name != "Rsqrt" &&
        root.name != "ReciprocalSqrt" && root.name != "Sqrt")) return false;
    ++a.stats["welford_reference_candidates"];
    work = local_merges = local_updates = next_domain = 0;
    spec.reset(); states.clear(); state_cache.clear(); scale_cache.clear(); primitive_cache.clear();
    outenv = left.family_environment(lf);
    try {
      if (&left.a != &right.a || !left.ideal || !right.ideal || lf.at("shape") != rf.at("shape")) throw Miss{};
      spec = reference(lf);
      if (spec->recipe.count <= 0 || spec->recipe.count > Z(max_work)) throw Miss{};
      if (left.family(lf) != lr || right.family(rf, left.family_type(lf)) != rr) throw Miss{};
      if (scalar(left, raw(input_id(lf.at("body").at("node"))), outenv) != lr) throw Miss{};
      outenv = right.family_environment(rf);
      Id target = rr;
      if (spec->projection == Std || spec->projection == InvStd) {
        const auto &n = a.nodes[rr];
        const char *expected = spec->projection == Std ? "sqrt" : "reciprocalsqrt";
        if (n.op == expected && n.args.size() == 1) target = n.args[0];
        else if (spec->projection == InvStd && n.op == "recip" && n.args.size() == 1 &&
                 a.nodes[n.args[0]].op == "sqrt" && a.nodes[n.args[0]].args.size() == 1)
          // The local M2 certificate below proves nonnegativity even when
          // scalar Facts did not see that delta*(x-new_mean) is a square.
          target = a.nodes[n.args[0]].args[0];
        else throw Miss{};
      }
      std::vector<Edge> candidates;
      if (spec->projection == Mean) candidates.emplace_back(rf.at("body"));
      else {
        // Other explicitly mapped output roots can contain the complete mean
        // even when the variance consumer only uses its child means. They are
        // witnesses, not assumptions: state(), coverage() and moment() verify
        // their exact sample recipe and the entire actual variance expression.
        for (const auto &family : right.doc.at("output_families")) {
          tick();
          if (family.at("shape") != rf.at("shape")) continue;
          if (right.family_environment(family) != outenv) continue;
          Edge root(family.at("body"));
          if (state(root, spec->recipe.count) == invalid) continue;
          if (candidates.size() == max_candidates) throw Miss{};
          candidates.push_back(root);
        }
        // Visit stored raw DAG nodes once, preserving boundaries that Arena
        // has flattened. Only reciprocal-count weighted means are candidates.
        std::vector<Id> todo{input_id(rf.at("body").at("node"))};
        std::unordered_set<Id> seen;
        while (!todo.empty()) {
          tick(); Id i = todo.back(); todo.pop_back();
          if (!seen.insert(i).second) continue;
          const auto &n = right.ns.at(i);
          for (const auto &e : n.operands) todo.push_back(e.node);
          if (n.kind == "Operation" && n.name == "Fma" && floating(n.type) && n.operands.size() == 3) {
            if (state(raw(i), spec->recipe.count) != invalid) {
              if (candidates.size() == max_candidates) throw Miss{};
              candidates.push_back(raw(i));
            }
            continue;
          }
          if (arithmetic(n, "Add", 2)) {
            bool update = false;
            for (const auto &edge : n.operands) {
              Scale step = split_scale(edge);
              const auto &delta = right.ns.at(step.core.node);
              update |= step.coefficient == Q(1, spec->recipe.count) &&
                        (arithmetic(delta, "Subtract", 2) || arithmetic(delta, "Add", 2));
            }
            if (update) {
              if (candidates.size() == max_candidates) throw Miss{};
              candidates.push_back(raw(i));
            }
            continue;
          }
          if (!arithmetic(n, "Multiply", 2) && !arithmetic(n, "Divide", 2)) continue;
          Scale scaled = split_scale(raw(i));
          if (scaled.coefficient != Q(1, spec->recipe.count)) continue;
          const auto &sum = right.ns.at(scaled.core.node);
          if (!arithmetic(sum, "Add", 2)) continue;
          Scale x = split_scale(sum.operands[0]), y = split_scale(sum.operands[1]);
          if (x.coefficient + y.coefficient != Q(spec->recipe.count)) continue;
          if (candidates.size() == max_candidates) throw Miss{};
          candidates.push_back(raw(i));
        }
      }
      Quadratic actual;
      if (spec->projection != Mean) accumulate(actual, target);
      for (const Edge &candidate : candidates) {
        ++a.stats["welford_state_candidates"];
        Id certified = state(candidate, spec->recipe.count);
        if (certified == invalid || !coverage(certified)) continue;
        auto expression=[&](const Quadratic &poly){std::vector<Id> terms;
          for(const auto &[factors,c]:poly){auto fs=factors;fs.push_back(a.constant(c));terms.push_back(a.mul(fs));}return a.add(terms);};
        if(scalar_engine){
          Id expected=spec->projection==Mean?states[certified].mean:expression(moment(certified));
          Id observed=spec->projection==Mean?rr:expression(actual);
          if(!scalar_engine->equal(expected,observed))continue;
        } else if (spec->projection == Mean ? states[certified].mean != rr : moment(certified) != actual) continue;
        J domain = J::array();
        for (size_t j = 0; j < spec->recipe.extents.size(); ++j) {
          if (spec->recipe.extents[j] != 0)
            domain.push_back({{"coordinate", j}, {"count", spec->recipe.extents[j].str()}});
          else domain.push_back({{"coordinate", j}, {"fixed", a.render(spec->recipe.fixed[j], 3)}});
        }
        receipt = {{"method", "welford_fixed_state_certificate"},
                   {"count", spec->recipe.count.str()}, {"sample_recipe", spec->recipe.tag},
                   {"domain", domain},
                   {"projection", spec->projection == Mean ? "mean" : spec->projection == Variance ? "variance" :
                                  spec->projection == Std ? "std" : "inverse_std"},
                   {"coverage", "exact_multiplicity_one"}, {"local_merge_rules", local_merges},
                   {"local_update_rules", local_updates}, {"work", work}};
        ++a.stats["welford_proven_families"];
        a.stats["welford_verified_merge_rules"] += local_merges;
        a.stats["welford_verified_update_rules"] += local_updates;
        return true;
      }
    } catch (const Miss &) {
      ++a.stats["welford_fragment_misses"];
    }
    return false;
  }
};
} // namespace rg
