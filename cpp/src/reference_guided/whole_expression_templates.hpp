#pragma once
#include "checker.hpp"
#include "finite_piece_union.hpp"
#include <array>

namespace rg {
struct WholeTemplateSample {
  Id left = invalid, right = invalid;
  std::vector<Id> coordinates;
};
struct WholeTemplateLimits {
  size_t seeds = 8, axes = 4, depth = 96, binder_depth = 8;
  uint64_t work = 400000, cache_entries = 2000000, template_nodes = 100000;
  size_t coordinate_terms = 8, ac_arity = 256;
};

// Builds a candidate only. A candidate is NOT a theorem: every complete proved
// piece must pass validate_piece(), and the separate finite-domain coverage
// certificate must seal before a caller may use rewrite_optimized().
class WholeExpressionTemplates {
  Arena &a;
  std::vector<Id> axes_;
  uint64_t context_;
  uint64_t bank_id;
  WholeTemplateLimits limits;
  Id left_ = invalid, right_ = invalid, first_private_id;
  uint64_t work = 0, visits = 0, alpha_hits = 0, instance_hits = 0;
  struct Decline { const char *why; };
  using Binding = std::pair<Id, Id>;
  struct Key {
    Id node = invalid;
    unsigned depth = 0, used = 0;
    std::array<Binding, 8> bindings{};
    bool operator==(const Key &) const = default;
  };
  struct KeyHash {
    size_t operator()(const Key &key) const {
      size_t result = key.node; mix(result, key.depth);
      for (unsigned i = 0; i < key.used; ++i) {
        mix(result, key.bindings[i].first); mix(result, key.bindings[i].second);
      }
      return result;
    }
  };
  std::unordered_map<Key, Id, KeyHash> alpha_cache, instance_cache;
  std::unordered_map<Id, bool> fold_cache;
  std::map<std::pair<unsigned, Z>, Id> binders;
  std::unordered_set<Id> private_binders;
  std::vector<WholeTemplateSample> seeds_;
  std::unordered_set<Id> local_axes, common;
  std::map<std::pair<bool, std::vector<Id>>, Id> unify_cache;
  std::unordered_map<uint64_t, size_t> signature_cache;

  void tick() { ++visits; a.check(); if (++work > limits.work) throw Decline{"work_budget"}; }
  void charge_edges(size_t edges) {
    if (edges > limits.work - work) throw Decline{"edge_budget"};
    work += edges;
  }
  void begin_attempt() {
    work = visits = alpha_hits = instance_hits = 0; last_failure.clear();
    failed_expected = failed_actual = invalid;
  }
  void finish_attempt(const char *stage, bool accepted) {
    const std::string prefix = std::string("whole_template_") + stage;
    ++a.stats[prefix + "_attempts"];
    a.stats[prefix + "_work"] += work; a.stats[prefix + "_visits"] += visits;
    a.stats[prefix + "_alpha_cache_hits"] += alpha_hits;
    a.stats[prefix + "_instance_cache_hits"] += instance_hits;
    if (!accepted) ++a.stats[prefix + "_decline_" + last_failure];
    last_work = work; last_visits = visits;
  }
  void depth_check(unsigned depth) {
    tick(); if (depth > limits.depth) throw Decline{"depth_budget"};
  }
  void cache_check() {
    if (alpha_cache.size() + instance_cache.size() + fold_cache.size() > limits.cache_entries)
      throw Decline{"cache_budget"};
  }
  Key key(Id id, unsigned depth, const std::vector<Binding> &bindings) {
    Key result; result.node = id; result.depth = depth;
    for (const auto &binding : bindings) if (a.depends(id, binding.first)) {
      if (result.used == result.bindings.size()) throw Decline{"binding_budget"};
      result.bindings[result.used++] = binding;
    }
    return result;
  }
  bool touches_local(Id id) const {
    for (Id v : local_axes) if (a.depends(id, v)) return true;
    return false;
  }
  bool contains_fold(Id id, unsigned depth = 0) {
    depth_check(depth);
    if (auto it = fold_cache.find(id); it != fold_cache.end()) return it->second;
    const auto &n = a.nodes[id]; bool yes = n.op == "fold";
    if (!yes) {
      charge_edges(n.args.size());
      for (Id child : n.args) if (contains_fold(child, depth + 1)) { yes = true; break; }
    }
    fold_cache.emplace(id, yes); cache_check(); return yes;
  }
  bool whole_coordinates(const std::vector<Id> &coordinates) {
    if (coordinates.size() != axes_.size()) return false;
    for (size_t i = 0; i < axes_.size(); ++i) {
      const auto &range = a.nodes[axes_[i]].bounds;
      const auto &value = a.nodes[coordinates[i]];
      if (!range || !value.bounds || value.bounds->first < range->first ||
          value.bounds->second > range->second || !CoordinateRelations::integer(a, coordinates[i])) return false;
      // The supplied coordinates are free at this local template boundary.
      // A private binder can never be introduced as a free actual coordinate.
      charge_edges(value.free.size());
      for (Id variable : value.free) if (private_binders.contains(variable)) return false;
    }
    return true;
  }
  Id canonical_binder(unsigned depth, const Z &count) {
    if (depth >= limits.binder_depth || count <= 0) throw Decline{"binder_domain"};
    auto k = std::pair(depth, count);
    if (auto it = binders.find(k); it != binders.end()) return it->second;
    Id value = a.var("whole-template:" + std::to_string(context_) + ":" +
                    std::to_string(first_private_id) + ":" + std::to_string(bank_id) + ":binder:" +
                    std::to_string(depth) + ":" + count.str(),
                    Q(0), Q(count - 1));
    if (value < first_private_id || std::find(axes_.begin(), axes_.end(), value) != axes_.end())
      throw Decline{"binder_name_collision"};
    private_binders.insert(value); binders.emplace(std::move(k), value); return value;
  }

  // Never copy inferred prototype Bounds/Facts into a larger formal domain.
  // Only exact opcode semantics and globally meaningful input types rebuild a
  // changed node. Unknown operations and locally bounded numeric reads decline.
  Id rebuild(Id prototype, const std::vector<Id> &args, Id binder = invalid) {
    tick(); const auto &n = a.nodes[prototype]; const auto &op = n.op;
    if (args == n.args && (binder == invalid || (n.op == "fold" && n.data[0] == Q(binder)))) return prototype;
    charge_edges(args.size());
    if (op == "add" && n.data.size() == args.size()) {
      std::vector<Id> terms; terms.reserve(args.size());
      for (size_t i = 0; i < args.size(); ++i) terms.push_back(a.mul({a.constant(n.data[i]), args[i]}));
      return a.add(std::move(terms));
    }
    if (op == "mul" && n.data.size() == 1) {
      auto terms = args; terms.push_back(a.constant(n.data[0])); return a.mul(std::move(terms));
    }
    if (op == "fold" && args.size() == 2 && n.data.size() == 2 && binder != invalid) {
      return a.intern("fold", args, {Q(binder), n.data[1]}, n.tag,
                      a.fold_facts(args[0],args[1],n.data[1],n.tag=="Max",binder));
    }
    if (op == "read") {
      auto split = n.tag.rfind('\n');
      if (split == std::string::npos || n.bounds) throw Decline{"bounded_or_untyped_read"};
      std::string type = n.tag.substr(split + 1);
      if (!type.starts_with("real")) throw Decline{"nonreal_read"};
      return a.read(n.tag.substr(0, split), args, type);
    }
    if ((op == "readbits" || op == "packedbits") && args.size() == 1 && n.data.size() == 1 &&
        n.data[0].denominator() == 1 && n.data[0] > 0 && n.data[0] <= 256) {
      unsigned bits = n.data[0].numerator().convert_to<unsigned>();
      return a.intern(op, args, n.data, n.tag, {true, false, true}, std::pair(Q(0), Q(pow2(bits) - 1)));
    }
    if (op == "extractbits" && args.size() == 1 && n.data.size() == 2 &&
        n.data[0].denominator() == 1 && n.data[1].denominator() == 1 &&
        n.data[0] >= 0 && n.data[0] <= 256 && n.data[1] > 0 && n.data[1] <= 256)
      return a.bit_slice(args[0], n.data[0].numerator().convert_to<unsigned>(), n.data[1].numerator().convert_to<unsigned>());
    if ((op == "idiv" || op == "imod") && args.size() == 1 && n.data.size() == 1 &&
        n.data[0].denominator() == 1 && n.data[0] > 0)
      return op == "idiv" ? a.idiv(args[0], n.data[0].numerator()) : a.imod(args[0], n.data[0].numerator());
    if (op == "signed") {
      if(args.size()!=1 || n.data.size()!=1 || n.data[0].denominator()!=1 ||
         n.data[0]<=0 || n.data[0]>256) throw Decline{"signed_word_width"};
      return a.signed_word(args[0],n.data[0].numerator().convert_to<unsigned>());
    }
    if (op == "word_or" && args.size()==2 && n.data.size()==1)
      return a.word_or(args[0],args[1],n.data[0].numerator().convert_to<unsigned>());
    if (op == "word_shr" && args.size()==2 && n.data.size()==2) {
      unsigned width=n.data[0].numerator().convert_to<unsigned>();bool sign=n.data[1]!=0;
      return a.word_shr(args[0],args[1],width,sign);
    }
    if (op == "floor" && args.size() == 1) return a.floor_value(args[0]);
    if ((op == "exp" || op == "exp2") && args.size() == 1) return a.exponential(args[0], op == "exp2");
    if (op == "recip" && args.size() == 1) return a.recip(args[0]);
    if (op == "abs" && args.size() == 1) return a.absolute(args[0]);
    if ((op == "log" || op == "log2") && args.size() == 1) return a.logarithm(args[0], op == "log2");
    if (op == "power" && args.size() == 2) return a.power(args[0], args[1]);
    if (op == "compare" && args.size() == 2) return a.compare(n.tag, args[0], args[1]);
    if (op == "select" && args.size() == 3) return a.select(args[0], args[1], args[2]);
    if (op == "max") return a.maximum(args);
    if (op == "clamp" && args.size() == 1 && n.data.size() == 2) return a.clamp(args[0], n.data[0], n.data[1]);
    if (op == "xor" && args.size() == 2) return a.bitxor(args[0], args[1]);
    if ((op == "sqrt" || op == "reciprocalsqrt" || op == "sin" || op == "cos" || op == "tanh") && args.size() == 1) {
      bool finite = a.nodes[args[0]].facts.finite;
      if (op == "sqrt" || op == "reciprocalsqrt") finite &= a.nodes[args[0]].facts.pos;
      bool positive = finite && (op == "sqrt" || op == "reciprocalsqrt");
      return a.intern(op, args, n.data, n.tag, {finite, positive, positive});
    }
    if ((op == "decode_fp4" || op == "decode_s4") && args.size() == 1)
      return a.intern(op, args, n.data, n.tag, {true, false, false},
                      op == "decode_fp4" ? Bounds(std::pair(Q(-6), Q(6))) : Bounds(std::pair(Q(-8), Q(7))));
    throw Decline{"unsupported_rebuild"};
  }

  Id alpha(Id id, const std::vector<Binding> &env, unsigned lexical = 0, unsigned depth = 0) {
    depth_check(depth);
    const auto &n = a.nodes[id];
    if (n.op == "var") {
      for (auto it = env.rbegin(); it != env.rend(); ++it) if (it->first == id) return it->second;
      if (private_binders.contains(id)) throw Decline{"free_private_binder"};
      return id;
    }
    bool folds = contains_fold(id);
    Key k = key(id, folds ? lexical : 0, env);
    if (!folds && k.used == 0) return id;
    if (auto it = alpha_cache.find(k); it != alpha_cache.end()) { ++alpha_hits; return it->second; }
    charge_edges(n.args.size());
    Id result;
    if (n.op == "fold") {
      if (n.args.size() != 2 || n.data.size() != 2 || n.tag != "Sum" ||
          n.data[0].denominator() != 1 || n.data[1].denominator() != 1 || n.data[1] <= 0)
        throw Decline{"fold_form"};
      if (n.data[0] < 0 || n.data[0] >= Q(a.nodes.size())) throw Decline{"fold_binder_id"};
      Id old = n.data[0].numerator().convert_to<Id>(); const auto &v = a.nodes[old];
      if (v.op != "var" || !v.bounds || v.bounds->first != 0 || v.bounds->second != n.data[1] - 1 ||
          a.depends(n.args[0], old)) throw Decline{"fold_scope"};
      for (const auto &[from, to] : env) if (from == old || to == old) throw Decline{"shadowed_binder"};
      Id fresh = canonical_binder(lexical, n.data[1].numerator());
      auto body_env = env; body_env.emplace_back(old, fresh);
      result = rebuild(id, {alpha(n.args[0], env, lexical, depth + 1),
                            alpha(n.args[1], body_env, lexical + 1, depth + 1)}, fresh);
    } else {
      std::vector<Id> args; args.reserve(n.args.size());
      for (Id child : n.args) args.push_back(alpha(child, env, lexical, depth + 1));
      result = rebuild(id, args);
    }
    alpha_cache.emplace(std::move(k), result); cache_check(); return result;
  }

  Id instantiate(Id id, const std::vector<Id> &coordinates, unsigned depth = 0) {
    depth_check(depth);
    for (size_t i = 0; i < axes_.size(); ++i) if (id == axes_[i]) return coordinates[i];
    Key k; k.node = id;
    for (size_t i = 0; i < axes_.size(); ++i) if (a.depends(id, axes_[i]))
      k.bindings[k.used++] = {axes_[i], coordinates[i]};
    if (k.used == 0) return id;
    if (auto it = instance_cache.find(k); it != instance_cache.end()) { ++instance_hits; return it->second; }
    const auto &n = a.nodes[id];
    charge_edges(n.args.size());
    std::vector<Id> args; args.reserve(n.args.size());
    for (Id child : n.args) args.push_back(instantiate(child, coordinates, depth + 1));
    Id binder = n.op == "fold" ? n.data[0].numerator().convert_to<Id>() : invalid;
    Id result = rebuild(id, args, binder);
    instance_cache.emplace(std::move(k), result); cache_check(); return result;
  }

  using Linear = std::map<Id, Q>;
  static bool small_integer(const Q &value) {
    if (value.denominator() != 1) return false;
    Z magnitude = boost::multiprecision::abs(value.numerator());
    return magnitude == 0 || boost::multiprecision::msb(magnitude) < 256;
  }
  std::optional<Linear> linear(Id id) {
    if (a.nodes[id].args.size() > limits.coordinate_terms) return {};
    Linear result;
    for (const auto &[term, coefficient] : a.terms(id)) {
      if (!small_integer(coefficient)) return {};
      if (term != a.one && a.nodes[term].op != "var" &&
          (!common.contains(term) || touches_local(term) || !CoordinateRelations::integer(a, term))) return {};
      result[term] += coefficient;
    }
    return result;
  }
  std::optional<Id> coordinate_fit(const std::vector<Id> &nodes) {
    std::vector<Linear> y; std::set<Id> atoms;
    std::vector<std::vector<Linear>> x(nodes.size());
    for (size_t s = 0; s < nodes.size(); ++s) {
      auto value = linear(nodes[s]); if (!value) return {};
      y.push_back(*value);
      for (const auto &[atom, c] : *value) if (c != 0) atoms.insert(atom);
      for (Id coordinate : seeds_[s].coordinates) {
        auto input = linear(coordinate); if (!input) return {};
        x[s].push_back(*input);
        for (const auto &[atom, c] : *input) if (c != 0) atoms.insert(atom);
      }
    }
    if (atoms.size() > 32) return {};
    auto coefficient = [](const Linear &f, Id atom) { auto it = f.find(atom); return it == f.end() ? Q(0) : it->second; };
    std::vector<Id> invariant_terms;
    std::set<Id> input_atoms;
    for (const auto &sample : x) for (const auto &f : sample)
      for (const auto &[atom, c] : f) if (c != 0) input_atoms.insert(atom);
    for (Id atom : atoms) if (atom != a.one && !input_atoms.contains(atom)) {
      Q c = coefficient(y[0], atom);
      for (size_t s = 1; s < y.size(); ++s) if (coefficient(y[s], atom) != c) return {};
      if (touches_local(atom)) return {};
      if (c != 0) invariant_terms.push_back(a.mul({a.constant(c), atom}));
      for (auto &f : y) f.erase(atom);
    }
    atoms.insert(a.one);
    const size_t columns = axes_.size() + 1;
    std::vector<std::vector<Q>> matrix;
    for (size_t s = 0; s < nodes.size(); ++s) for (Id atom : atoms) {
      if (atom != a.one && !input_atoms.contains(atom)) continue;
      std::vector<Q> row(columns + 1, Q(0));
      for (size_t axis = 0; axis < axes_.size(); ++axis) row[axis] = coefficient(x[s][axis], atom);
      row[axes_.size()] = atom == a.one ? Q(1) : Q(0); row[columns] = coefficient(y[s], atom);
      matrix.push_back(std::move(row));
      if (matrix.size() > 128) return {};
    }
    size_t pivot = 0; std::vector<size_t> pivots;
    for (size_t column = 0; column < columns && pivot < matrix.size(); ++column) {
      tick(); size_t found = pivot;
      while (found < matrix.size() && matrix[found][column] == 0) ++found;
      if (found == matrix.size()) continue;
      std::swap(matrix[pivot], matrix[found]); Q divisor = matrix[pivot][column];
      for (size_t j = column; j <= columns; ++j) matrix[pivot][j] /= divisor;
      for (size_t i = 0; i < matrix.size(); ++i) if (i != pivot && matrix[i][column] != 0) {
        tick();
        Q scale = matrix[i][column];
        for (size_t j = column; j <= columns; ++j) matrix[i][j] -= scale * matrix[pivot][j];
      }
      pivots.push_back(column); ++pivot;
    }
    for (const auto &row : matrix) {
      bool zero = true; for (size_t i = 0; i < columns; ++i) zero &= row[i] == 0;
      if (zero && row[columns] != 0) return {};
    }
    std::vector<Q> answer(columns, Q(0));
    for (size_t i = 0; i < pivots.size(); ++i) answer[pivots[i]] = matrix[i][columns];
    for (const Q &value : answer) if (!small_integer(value)) return {};
    invariant_terms.push_back(a.constant(answer.back()));
    for (size_t i = 0; i < axes_.size(); ++i) invariant_terms.push_back(a.mul({a.constant(answer[i]), axes_[i]}));
    Id candidate = a.add(std::move(invariant_terms));
    for (size_t s = 0; s < nodes.size(); ++s) if (instantiate(candidate, seeds_[s].coordinates) != nodes[s]) return {};
    return candidate;
  }

  size_t signature(Id id, bool coordinate = false, unsigned depth = 0) {
    depth_check(depth); uint64_t key = (uint64_t(id) << 1) | coordinate;
    if (auto it = signature_cache.find(key); it != signature_cache.end()) return it->second;
    size_t hash = 0;
    if (coordinate) hash = 17;
    else if (common.contains(id) && !touches_local(id)) { hash = 31; mix(hash, id); }
    else {
      const auto &n = a.nodes[id]; hash = std::hash<std::string>{}(n.op); mix(hash, std::hash<std::string>{}(n.tag));
      charge_edges(n.args.size());
      bool child_coordinate = n.op == "read" || n.op == "readbits" || n.op == "packedbits";
      std::vector<size_t> children;
      for (size_t i = 0; i < n.args.size(); ++i) {
        size_t h = signature(n.args[i], child_coordinate, depth + 1);
        if (n.op == "add") mix(h, hashq(n.data[i]));
        children.push_back(h);
      }
      if (n.op == "add" || n.op == "mul") std::sort(children.begin(), children.end());
      for (size_t h : children) mix(hash, h);
      if (n.op != "add") for (const Q &value : n.data) mix(hash, hashq(value));
    }
    signature_cache.emplace(key, hash); return hash;
  }
  Id unify(const std::vector<Id> &nodes, bool coordinate = false, unsigned depth = 0) {
    depth_check(depth);
    auto cache_key = std::pair(coordinate, nodes);
    if (auto it = unify_cache.find(cache_key); it != unify_cache.end()) return it->second;
    if (unify_cache.size() > limits.template_nodes) throw Decline{"template_budget"};
    if (std::all_of(nodes.begin(), nodes.end(), [&](Id id) { return id == nodes[0]; }) && !touches_local(nodes[0])) return nodes[0];
    if (coordinate) if (auto fitted = coordinate_fit(nodes)) {
      unify_cache.emplace(std::move(cache_key), *fitted); return *fitted;
    }
    const auto &first = a.nodes[nodes[0]];
    bool ac = first.op == "add" || first.op == "mul";
    if (first.op == "var" || first.op == "const" || (ac && first.args.size() > limits.ac_arity))
      throw Decline{"template_shape"};
    std::vector<std::vector<Id>> children(nodes.size());
    std::vector<std::vector<Q>> coefficients(nodes.size());
    for (size_t s = 0; s < nodes.size(); ++s) {
      const auto &n = a.nodes[nodes[s]];
      if (n.op != first.op || n.tag != first.tag || n.args.size() != first.args.size() ||
          (n.op != "add" && n.data != first.data)) throw Decline{"template_shape"};
      charge_edges(n.args.size());
      std::vector<size_t> order(n.args.size()); std::iota(order.begin(), order.end(), 0);
      if (ac) std::stable_sort(order.begin(), order.end(), [&](size_t x, size_t y) {
        auto sx = std::pair(signature(n.args[x]), n.op == "add" ? n.data[x] : Q(0));
        auto sy = std::pair(signature(n.args[y]), n.op == "add" ? n.data[y] : Q(0));
        return sx < sy;
      });
      for (size_t i : order) { children[s].push_back(n.args[i]); if (n.op == "add") coefficients[s].push_back(n.data[i]); }
      if (s && coefficients[s] != coefficients[0]) throw Decline{"template_coefficients"};
    }
    std::vector<Id> args;
    bool child_coordinate = coordinate || first.op == "read" || first.op == "readbits" || first.op == "packedbits";
    for (size_t i = 0; i < first.args.size(); ++i) {
      std::vector<Id> sample; for (const auto &v : children) sample.push_back(v[i]);
      args.push_back(unify(sample, child_coordinate, depth + 1));
    }
    Id result;
    if (first.op == "add") {
      std::vector<Id> terms;
      for (size_t i = 0; i < args.size(); ++i) terms.push_back(a.mul({a.constant(coefficients[0][i]), args[i]}));
      result = a.add(std::move(terms));
    } else {
      Id binder = first.op == "fold" ? first.data[0].numerator().convert_to<Id>() : invalid;
      result = rebuild(nodes[0], args, binder);
    }
    unify_cache.emplace(std::move(cache_key), result); return result;
  }
  std::unordered_set<Id> reachable(Id root) {
    std::unordered_set<Id> seen; std::vector<Id> todo{root};
    while (!todo.empty()) {
      tick(); Id id = todo.back(); todo.pop_back(); if (!seen.insert(id).second) continue;
      const auto &n = a.nodes[id]; charge_edges(n.args.size()); todo.insert(todo.end(), n.args.begin(), n.args.end());
    }
    return seen;
  }
  Id build_side(const std::vector<Id> &roots) {
    common = reachable(roots[0]);
    for (size_t i = 1; i < roots.size(); ++i) {
      auto ids = reachable(roots[i]);
      for (auto it = common.begin(); it != common.end();) if (!ids.contains(*it)) it = common.erase(it); else ++it;
    }
    unify_cache.clear(); signature_cache.clear();
    Id candidate = unify(roots);
    for (size_t i = 0; i < roots.size(); ++i)
      if (instantiate(candidate, seeds_[i].coordinates) != roots[i]) throw Decline{"seed_instance"};
    if (!a.nodes[candidate].facts.finite) throw Decline{"nonfinite_template"};
    for (Id free : a.nodes[candidate].free)
      if (std::find(axes_.begin(), axes_.end(), free) == axes_.end()) throw Decline{"template_free_variable"};
    return candidate;
  }

  bool matches_side(Id target, const std::vector<Id> &coordinates, bool reference) {
    begin_attempt();
    const char *stage = reference ? "reference_match" : "optimized_match";
    try {
      if (left_ == invalid || right_ == invalid || !whole_coordinates(coordinates))
        throw Decline{"target_coordinates"};
      // Match only the requested side. Optimized-state abstraction does not
      // construct the reference instance, including its nested reductions.
      Id expected = instantiate(reference ? left_ : right_, coordinates);
      Id actual = alpha(target, {});
      if (expected != actual) {
        failed_expected = expected; failed_actual = actual;
        throw Decline{"whole_target_instance"};
      }
      finish_attempt(stage, true); return true;
    } catch (const Decline &miss) { last_failure = miss.why; }
      catch (const Unsupported &) { last_failure = "unsupported"; }
    finish_attempt(stage, false); return false;
  }

public:
  std::string last_failure;
  uint64_t last_work = 0, last_visits = 0;
  Id failed_expected = invalid, failed_actual = invalid;
  WholeExpressionTemplates(Checker &checker, std::vector<Id> formal_axes, uint64_t context,
                           WholeTemplateLimits budget = {})
      : a(checker.a), axes_(std::move(formal_axes)), context_(context),
        bank_id(++a.stats["whole_template_banks"]), limits(budget), first_private_id(Id(a.nodes.size())) {}
  Id left_template() const { return left_; }
  Id right_template() const { return right_; }
  const std::vector<Id> &axes() const { return axes_; }
  uint64_t context() const { return context_; }
  bool build(const std::vector<WholeTemplateSample> &samples) {
    begin_attempt(); left_ = right_ = invalid;
    try {
      if (samples.size() < 2 || samples.size() > limits.seeds || axes_.empty() || axes_.size() > limits.axes || axes_.size() > 4)
        throw Decline{"schema_size"};
      std::set<Id> distinct;
      for (Id axis : axes_) {
        const auto &n = a.nodes[axis];
        if (!distinct.insert(axis).second || n.op != "var" || !n.bounds ||
            n.bounds->first.denominator() != 1 || n.bounds->second.denominator() != 1 || n.bounds->first > n.bounds->second)
          throw Decline{"formal_domain"};
      }
      seeds_ = samples; local_axes.clear(); std::vector<Id> left, right;
      for (const auto &sample : samples) {
        if (!whole_coordinates(sample.coordinates)) throw Decline{"sample_coordinates"};
        for (Id value : sample.coordinates) for (Id v : a.nodes[value].free) {
          if (distinct.contains(v)) throw Decline{"formal_alias"};
          local_axes.insert(v);
        }
        left.push_back(alpha(sample.left, {})); right.push_back(alpha(sample.right, {}));
      }
      left_ = build_side(left); right_ = build_side(right);
      ++a.stats["whole_template_candidates"]; finish_attempt("build", true); return true;
    } catch (const Decline &miss) { last_failure = miss.why; }
      catch (const Unsupported &) { last_failure = "unsupported"; }
    left_ = right_ = invalid; finish_attempt("build", false); return false;
  }
  std::optional<FinitePieceInstanceWitness> validate_piece(Id left, Id right, const std::vector<Id> &coordinates) {
    begin_attempt();
    try {
      if (left_ == invalid || right_ == invalid || !whole_coordinates(coordinates)) throw Decline{"piece_coordinates"};
      Id expected_left = instantiate(left_, coordinates), actual_left = alpha(left, {});
      if (expected_left != actual_left) {
        failed_expected = expected_left; failed_actual = actual_left;
        throw Decline{"whole_piece_left_instance"};
      }
      Id expected_right = instantiate(right_, coordinates), actual_right = alpha(right, {});
      if (expected_right != actual_right) {
        failed_expected = expected_right; failed_actual = actual_right;
        throw Decline{"whole_piece_right_instance"};
      }
      ++a.stats["whole_template_validated_pieces"]; finish_attempt("validate", true);
      // These original body IDs are certified by full substitution and exact
      // capture-free alpha-normalized whole-root equality above. They are not
      // unchecked caller-provided substitutes for the instantiated roots.
      return FinitePieceInstanceWitness{left_, right_, left, right, coordinates, context_};
    } catch (const Decline &miss) { last_failure = miss.why; }
      catch (const Unsupported &) { last_failure = "unsupported"; }
    finish_attempt("validate", false); return {};
  }
  // These are exact full-instance predicates, not equality theorems. A caller
  // may identify the two sides only after the complete piece union has sealed.
  bool matches_reference(Id target, const std::vector<Id> &coordinates) {
    return matches_side(target, coordinates, true);
  }
  bool matches_optimized(Id target, const std::vector<Id> &coordinates) {
    return matches_side(target, coordinates, false);
  }
  std::optional<Id> rewrite_optimized(Id target, const std::vector<Id> &coordinates) {
    begin_attempt();
    try {
      if (left_ == invalid || right_ == invalid || !whole_coordinates(coordinates)) throw Decline{"target_coordinates"};
      Id expected = instantiate(right_, coordinates), actual = alpha(target, {});
      if (expected != actual) {
        failed_expected = expected; failed_actual = actual; throw Decline{"whole_target_instance"};
      }
      Id result = instantiate(left_, coordinates); finish_attempt("rewrite", true); return result;
    } catch (const Decline &miss) { last_failure = miss.why; }
      catch (const Unsupported &) { last_failure = "unsupported"; }
    finish_attempt("rewrite", false); return {};
  }
};
} // namespace rg
