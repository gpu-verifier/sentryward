#pragma once
#include "checker.hpp"
#include <cstdint>

namespace rg {

struct AffineStateLimits {
  uint64_t work = 20000000, stored_nodes = 2000000, states = 131072;
  size_t segments = 8192, steps = 4096, factors = 8, rank = 4;
  size_t expression_nodes = 128, expression_depth = 96;
};
struct AffineStateLocation { size_t segment = 0, index = 0; };
struct AffineStateSegment {
  // These are ordinary finite expressions. axis is a fresh integer binder on
  // [0,count). Every original_states[k] has been checked against the recurrence
  // represented by potential/update at k; no unseen time points are inferred.
  Id axis = invalid, potential = invalid, update = invalid, incoming = invalid;
  size_t count = 0;
  std::vector<Id> original_states;
};
struct AffineStateProof { Id replacement = invalid; J receipt; };

// A bounded recognizer for an already stored affine recurrence, not a solver
// for arbitrary recurrences. It checks each actual Add/Mul update once. Only
// typed finite input reads with affine integer coordinates may form the update
// template. Numeric coefficients, input identities and all other coordinates
// remain frozen. The resulting Sum is justified by finite induction and
// exp(a-b)*exp(b-c)=exp(a-c), under the Arena's existing ideal-real Exp semantics.
class AffineStatePattern {
  Checker &checker;
  Arena &a;
  AffineStateLimits limits;
  uint64_t bank, work = 0, certified_updates = 0, version = 0;
  struct Decline { const char *reason; };
  struct Coordinate { Id fixed = invalid; Z offset = 0, slope = 0; };
  struct ReadModel { std::string tag, type; std::vector<Coordinate> coordinates; };
  struct ProductModel { Q coefficient = 1; std::vector<ReadModel> reads; };
  struct Product { Q coefficient = 1; std::map<std::string, Id> reads; };
  struct State {
    Id potential = invalid, update = invalid, incoming = invalid;
    size_t segment = SIZE_MAX, index = 0;
  };
  struct Model { ReadModel potential; ProductModel update; bool dirty = true; };
  struct Decay { Id previous = invalid, current_potential = invalid, previous_potential = invalid; };
  std::vector<AffineStateSegment> segments_;
  std::vector<Model> models;
  std::unordered_map<Id, State> states;
  std::unordered_set<Id> scanned;
  std::unordered_map<Id, bool> eligible_roots;
  std::map<std::tuple<Id, Id, Id>, Id> substitution_cache;
  std::map<std::pair<size_t, Id>, Id> prefix_cache;
  std::unordered_map<Id, Id> rewrite_cache;

  void tick(uint64_t amount = 1) {
    a.check();
    if (amount > limits.work - std::min(work, limits.work)) throw Decline{"work_budget"};
    work += amount;
  }
  bool integer_leaf(Id id) const {
    const auto &n = a.nodes[id];
    return n.facts.finite && ((n.op == "const" && n.data.size() == 1 && n.data[0].denominator() == 1) ||
           (n.op == "var" && n.data.size() == 2 &&
            n.data[0].denominator() == 1 && n.data[1].denominator() == 1));
  }
  bool integer_coordinate(Id id) {
    if(integer_leaf(id))return true;
    const auto &n=a.nodes[id];
    // Arena already stores affine Add coefficients separately. Check only
    // those immediate terms; no coordinate reconstruction or ancestor search.
    // This permits a fixed component such as 1+2*output while keeping its
    // exact identity. Only concrete changing coordinates infer time slopes.
    if(!n.facts.finite || !n.bounds || n.bounds->first.denominator()!=1 ||
       n.bounds->second.denominator()!=1 || n.args.empty() ||
       n.args.size()>limits.factors)return false;
    tick(n.args.size()+1);
    if(n.op=="mul")return n.args.size()==1 && n.data.size()==1 &&
      n.data[0].denominator()==1 && integer_leaf(n.args[0]);
    if(n.op!="add" || n.data.size()!=n.args.size())return false;
    for(size_t k=0;k<n.args.size();++k)
      if(n.data[k].denominator()!=1 || !integer_leaf(n.args[k]))return false;
    return true;
  }
  bool typed_read(Id id) {
    const auto &n = a.nodes[id];
    auto split = n.tag.rfind('\n');
    if (n.op != "read" || n.args.empty() || n.args.size() > limits.rank ||
        !n.data.empty() || !n.facts.finite || n.bounds || split == std::string::npos ||
        !n.tag.substr(split + 1).starts_with("real")) return false;
    for (Id x : n.args) if (!integer_coordinate(x)) return false;
    return true;
  }
  std::optional<Product> product(Id root) {
    tick();
    Product out;
    const auto &n = a.nodes[root];
    const auto *children = &n.args;
    std::vector<Id> singleton;
    if (n.op == "mul") {
      if (n.data.size() != 1 || !n.facts.finite || n.args.size() > limits.factors) return {};
      out.coefficient = n.data[0];
    } else { singleton = {root}; children = &singleton; }
    if (out.coefficient == 0 || children->empty()) return {};
    for (Id child : *children) {
      tick(); if (!typed_read(child)) return {};
      // Repeated allocation/type identities would require a separate matching
      // proof; the narrow recognizer does not guess their pairing.
      if (!out.reads.emplace(a.nodes[child].tag, child).second) return {};
    }
    return out;
  }
  std::optional<ReadModel> infer_read(Id first, Id second, bool potential) {
    tick(); if (!typed_read(first) || !typed_read(second)) return {};
    const auto &x = a.nodes[first], &y = a.nodes[second];
    if (x.tag != y.tag || x.args.size() != y.args.size()) return {};
    size_t split = x.tag.rfind('\n');
    ReadModel result{x.tag.substr(0, split), x.tag.substr(split + 1), {}};
    size_t changing = 0;
    for (size_t i = 0; i < x.args.size(); ++i) {
      tick();
      if (x.args[i] == y.args[i]) result.coordinates.push_back({x.args[i], 0, 0});
      else {
        auto p = a.value(x.args[i]), q = a.value(y.args[i]);
        if (!p || !q || p->denominator() != 1 || q->denominator() != 1) return {};
        Z slope = q->numerator() - p->numerator();
        if (potential && slope != 1) return {};
        result.coordinates.push_back({invalid, p->numerator(), slope}); ++changing;
      }
    }
    if (potential && changing != 1) return {};
    return result;
  }
  std::optional<ProductModel> infer_update(Id first, Id second) {
    auto p = product(first), q = product(second);
    if (!p || !q || p->coefficient != q->coefficient || p->reads.size() != q->reads.size()) return {};
    ProductModel result; result.coefficient = p->coefficient;
    bool changing = false;
    for (const auto &[key, id] : p->reads) {
      auto it = q->reads.find(key); if (it == q->reads.end()) return {};
      auto model = infer_read(id, it->second, false); if (!model) return {};
      for (const auto &coordinate : model->coordinates) changing |= coordinate.fixed == invalid;
      result.reads.push_back(std::move(*model));
    }
    if (!changing) return {};
    return result;
  }
  bool matches_read(const ReadModel &model, Id target, size_t index) {
    tick(); if (!typed_read(target)) return false;
    const auto &n = a.nodes[target];
    if (n.tag != model.tag + "\n" + model.type || n.args.size() != model.coordinates.size()) return false;
    for (size_t i = 0; i < n.args.size(); ++i) {
      tick(); const auto &coordinate = model.coordinates[i];
      if (coordinate.fixed != invalid) { if (n.args[i] != coordinate.fixed) return false; }
      else {
        auto actual = a.value(n.args[i]);
        if (!actual || *actual != Q(coordinate.offset + coordinate.slope * Z(index))) return false;
      }
    }
    return true;
  }
  bool matches_update(const ProductModel &model, Id target, size_t index) {
    auto p = product(target);
    if (!p || p->coefficient != model.coefficient || p->reads.size() != model.reads.size()) return false;
    for (const auto &read : model.reads) {
      auto it = p->reads.find(read.tag + "\n" + read.type);
      if (it == p->reads.end() || !matches_read(read, it->second, index)) return false;
    }
    return true;
  }
  Id render_read(const ReadModel &model, Id axis) {
    std::vector<Id> coordinates;
    for (const auto &c : model.coordinates) {
      tick(); coordinates.push_back(c.fixed != invalid ? c.fixed :
          a.add({a.constant(Q(c.offset)), a.mul({a.constant(Q(c.slope)), axis})}));
    }
    // The integer range is exactly the fully checked sample interval; no
    // prototype numeric bounds are copied to the generalized input read.
    return a.read(model.tag, coordinates, model.type);
  }
  Id render_update(const ProductModel &model, Id axis) {
    std::vector<Id> factors{a.constant(model.coefficient)};
    for (const auto &read : model.reads) factors.push_back(render_read(read, axis));
    return a.mul(std::move(factors));
  }
  std::optional<Decay> decay(Id term) {
    tick(); const auto &n = a.nodes[term];
    if (n.op != "mul" || n.data.size() != 1 || n.data[0] != 1 ||
        n.args.size() < 2 || n.args.size() > limits.factors + 1 || !n.facts.finite) return {};
    Id exponential = invalid; std::vector<Id> rest;
    for (Id factor : n.args) {
      if (a.nodes[factor].op == "exp") {
        if (exponential != invalid) return {};
        exponential = factor;
      } else rest.push_back(factor);
    }
    if (exponential == invalid) return {};
    const auto &e = a.nodes[exponential];
    if (e.args.size() != 1 || !e.facts.finite || !e.facts.pos) return {};
    Id current = invalid, previous = a.zero;
    const auto &p = a.nodes[e.args[0]];
    if (typed_read(e.args[0])) current = e.args[0];
    else if (p.op == "add" && p.args.size() == 2 && p.data.size() == 2) {
      previous = invalid;
      for (size_t j = 0; j < 2; ++j) {
        if (!typed_read(p.args[j])) return {};
        if (p.data[j] == 1 && current == invalid) current = p.args[j];
        else if (p.data[j] == -1 && previous == invalid) previous = p.args[j];
        else return {};
      }
    }
    if (current == invalid || previous == invalid) return {};
    return Decay{a.mul(std::move(rest)), current, previous};
  }
  bool contains(Id root, Id needle) {
    std::vector<Id> todo{root}; size_t seen = 0;
    while (!todo.empty()) {
      Id id = todo.back(); todo.pop_back(); tick();
      if (id == needle) return true;
      if (++seen > limits.expression_nodes) return true; // conservative decline
      const auto &n = a.nodes[id]; tick(n.args.size());
      todo.insert(todo.end(), n.args.begin(), n.args.end());
    }
    return false;
  }
  void insert_state(Id root, State state) {
    if (states.size() >= limits.states) throw Decline{"state_budget"};
    states.emplace(root, std::move(state));
  }
  bool eligible(Id root) {
    if (scanned.contains(root) || states.contains(root)) return true;
    if (auto it = eligible_roots.find(root); it != eligible_roots.end()) return it->second;
    // Inspect only a small arithmetic/dispatcher shell. Ordinary pointwise,
    // normalization and reduction outputs do not trigger a full DAG walk.
    struct Item { Id id; unsigned depth; };
    std::vector<Item> todo{{root, 0}};
    size_t visits = 0; bool found = false;
    while (!todo.empty() && visits < 64) {
      auto [id, depth] = todo.back(); todo.pop_back(); ++visits; tick();
      const auto &n = a.nodes[id];
      if (n.op == "add" && n.args.size() == 2 && n.data.size() == 2 &&
          n.data[0] == 1 && n.data[1] == 1) {
        for (Id term : n.args) {
          const auto &m = a.nodes[term];
          if (m.op != "mul" || m.args.size() > limits.factors + 1) continue;
          for (Id factor : m.args) if (a.nodes[factor].op == "exp") { found = true; break; }
          if (found) break;
        }
      }
      if (found) break;
      if (depth == 8) continue;
      if (n.op == "select" && n.args.size() == 3) {
        todo.push_back({n.args[2], depth + 1}); todo.push_back({n.args[1], depth + 1});
      } else if (n.op == "fold" && n.tag == "Sum" && n.args.size() == 2) {
        todo.push_back({n.args[1], depth + 1});
      } else if ((n.op == "add" || n.op == "mul") && n.args.size() <= limits.factors) {
        for (Id child : n.args) todo.push_back({child, depth + 1});
      }
    }
    eligible_roots.emplace(root, found); return found;
  }
  bool recognize(Id root) {
    if (states.contains(root)) return true;
    const auto &n = a.nodes[root];
    if (n.op != "add" || n.args.size() != 2 || n.data.size() != 2 ||
        n.data[0] != 1 || n.data[1] != 1 || !n.facts.finite) return false;
    for (size_t side = 0; side < 2; ++side) {
      auto d = decay(n.args[side]); if (!d) continue;
      Id update = n.args[1 - side];
      if (!product(update) || contains(update, d->previous) ||
          contains(d->current_potential, d->previous) ||
          contains(d->previous_potential, d->previous)) continue;
      auto prior = states.find(d->previous);
      if (d->previous_potential == a.zero) {
        // A new local potential origin: preserve the whole incoming state.
        // It is never confused with a cross-chunk difference g_new-g_old.
        if (prior == states.end() || !a.nodes[d->previous].facts.finite) continue;
        insert_state(root, {d->current_potential, update, d->previous, SIZE_MAX, 0});
        ++certified_updates; return true;
      }
      if (prior == states.end()) {
        // Zero initialization removes the first exp factor and leaves V_0.
        // A finite product can be a base only after the next exact update
        // establishes both potential and update templates below.
        if (!product(d->previous)) continue;
        auto pm = infer_read(d->previous_potential, d->current_potential, true);
        auto um = infer_update(d->previous, update);
        if (!pm || !um) continue;
        if (segments_.size() >= limits.segments) throw Decline{"segment_budget"};
        size_t sid = segments_.size();
        segments_.push_back({invalid, invalid, invalid, a.zero, 2, {d->previous, root}});
        models.push_back({std::move(*pm), std::move(*um), true});
        insert_state(d->previous, {d->previous_potential, d->previous, a.zero, sid, 0});
        insert_state(root, {d->current_potential, update, a.zero, sid, 1});
        ++certified_updates; ++version; return true;
      }
      State previous = prior->second;
      if (previous.potential != d->previous_potential) continue;
      if (previous.segment == SIZE_MAX) {
        auto pm = infer_read(previous.potential, d->current_potential, true);
        auto um = infer_update(previous.update, update);
        if (!pm || !um) continue;
        if (segments_.size() >= limits.segments) throw Decline{"segment_budget"};
        size_t sid = segments_.size();
        segments_.push_back({invalid, invalid, invalid, previous.incoming, 2, {d->previous, root}});
        models.push_back({std::move(*pm), std::move(*um), true});
        states.at(d->previous).segment = sid;
        insert_state(root, {d->current_potential, update, previous.incoming, sid, 1});
        ++certified_updates; ++version; return true;
      }
      size_t sid = previous.segment, index = previous.index + 1;
      if (index >= limits.steps || !matches_read(models[sid].potential, d->current_potential, index) ||
          !matches_update(models[sid].update, update, index)) continue;
      auto &segment = segments_[sid];
      if (index < segment.original_states.size()) {
        if (segment.original_states[index] != root) continue;
      } else {
        if (index != segment.original_states.size()) continue;
        segment.original_states.push_back(root); segment.count = index + 1;
        models[sid].dirty = true; ++version;
      }
      insert_state(root, {d->current_potential, update, previous.incoming, sid, index});
      ++certified_updates; return true;
    }
    return false;
  }
  void finalize() {
    for (size_t sid = 0; sid < models.size(); ++sid) {
      if (!models[sid].dirty) continue;
      auto &segment = segments_[sid];
      size_t before = a.nodes.size();
      segment.axis = a.var("affine-state:" + std::to_string(bank) + ":" +
                          std::to_string(sid) + ":" + std::to_string(segment.count),
                          0, Q(Z(segment.count - 1)));
      if (segment.axis < before) throw Decline{"binder_name_collision"};
      segment.potential = render_read(models[sid].potential, segment.axis);
      segment.update = render_update(models[sid].update, segment.axis);
      models[sid].dirty = false;
    }
    prefix_cache.clear(); rewrite_cache.clear();
  }
  Id rebuild(Id old, const std::vector<Id> &args) {
    const auto &n = a.nodes[old];
    if (args == n.args) return old;
    if (n.op == "read") {
      auto split = n.tag.rfind('\n');
      if (split == std::string::npos || n.bounds || !n.tag.substr(split + 1).starts_with("real"))
        throw Decline{"substitute_read"};
      return a.read(n.tag.substr(0, split), args, n.tag.substr(split + 1));
    }
    if (n.op == "fold") {
      if (n.tag != "Sum" || n.args.size() != 2 || n.data.size() != 2) throw Decline{"substitute_fold"};
      bool finite = a.nodes[args[0]].facts.finite && a.nodes[args[1]].facts.finite;
      return a.intern("fold", args, n.data, n.tag, {finite, false, false});
    }
    if (n.op == "add" || n.op == "mul" || n.op == "exp" || n.op == "select" || n.op == "compare")
      return checker.reapply(old, args);
    throw Decline{"substitute_operator"};
  }
  Id substitute_impl(Id source, Id axis, Id value) {
    // The reference can already contain a long, finite chain of state updates.
    // Traverse that stored DAG once per substitution, not recursively through
    // its depth and not once per logical recurrence step after abstraction.
    auto resolved = [&](Id id) -> std::optional<Id> {
      if (id == axis) return value;
      if (!a.depends(id,axis)) return id;
      auto it=substitution_cache.find(std::tuple(id,axis,value));
      return it==substitution_cache.end()?std::nullopt:std::optional<Id>(it->second);
    };
    struct Work { Id id; bool ready; };
    std::vector<Work> todo{{source,false}};
    while (!todo.empty()) {
      tick(); auto [id,ready]=todo.back();todo.pop_back();
      if (resolved(id)) continue;
      const auto &n=a.nodes[id];
      if (n.args.size()>limits.expression_nodes) throw Decline{"substitute_arity"};
      if (n.op=="fold") {
        Id binder=n.data[0].numerator().convert_to<Id>();
        if (binder==axis || a.depends(value,binder)) throw Decline{"substitute_capture"};
      }
      tick(n.args.size());
      if (!ready) {
        todo.push_back({id,true});
        for (Id child:n.args) if (!resolved(child)) todo.push_back({child,false});
        continue;
      }
      std::vector<Id> args;args.reserve(n.args.size());
      for (Id child:n.args) args.push_back(resolved(child).value());
      substitution_cache.emplace(std::tuple(id,axis,value),rebuild(id,args));
    }
    return resolved(source).value();
  }
  Id prefix_impl(size_t sid, Id index, size_t depth = 0) {
    tick(); if (sid >= segments_.size() || depth > limits.expression_depth) throw Decline{"prefix_segment"};
    const auto &segment = segments_[sid];
    if (segment.axis == invalid || segment.count < 2 || a.depends(index, segment.axis) ||
        !a.nodes[index].bounds || a.nodes[index].bounds->first < 0 ||
        a.nodes[index].bounds->second >= Q(Z(segment.count)) ||
        !CoordinateRelations::integer(a, index)) throw Decline{"prefix_index"};
    auto key = std::pair(sid, index);
    if (auto it = prefix_cache.find(key); it != prefix_cache.end()) return it->second;
    Id current = substitute_impl(segment.potential, segment.axis, index);
    Id term = a.mul({a.exp(a.sub(current, segment.potential)), segment.update});
    if (auto end = a.value(index); !end || *end != Q(Z(segment.count - 1)))
      term = a.select(a.compare("ge", index, segment.axis), term, a.zero);
    Id sum = a.intern("fold", {a.zero, term}, {Q(segment.axis), Q(Z(segment.count))},
                     "Sum", {a.nodes[term].facts.finite, false, false});
    Id incoming = segment.incoming;
    if (auto prior = lookup(incoming)) incoming = prefix_impl(prior->segment, a.constant(Q(Z(prior->index))), depth + 1);
    Id result = a.add({a.mul({a.exp(current), incoming}), sum});
    prefix_cache.emplace(key, result); ++a.stats["affine_state_closed_prefixes"]; return result;
  }
  Id rewrite_impl(Id root) {
    struct Work { Id id; bool ready; };
    std::vector<Work> todo{{root, false}};
    while (!todo.empty()) {
      auto w = todo.back(); todo.pop_back(); tick();
      if (rewrite_cache.contains(w.id)) continue;
      if (auto state = lookup(w.id)) {
        rewrite_cache.emplace(w.id, prefix_impl(state->segment, a.constant(Q(Z(state->index)))));
        continue;
      }
      const auto &n = a.nodes[w.id];
      if (!w.ready) {
        todo.push_back({w.id, true}); tick(n.args.size());
        for (Id child : n.args) if (!rewrite_cache.contains(child)) todo.push_back({child, false});
      } else {
        std::vector<Id> args; for (Id child : n.args) args.push_back(rewrite_cache.at(child));
        rewrite_cache.emplace(w.id, rebuild(w.id, args));
      }
    }
    return rewrite_cache.at(root);
  }
public:
  std::string last_failure;
  explicit AffineStatePattern(Checker &owner, AffineStateLimits budget = {})
      : checker(owner), a(owner.a), limits(budget), bank(++a.stats["affine_state_banks"]) {}
  AffineStatePattern(const AffineStatePattern &) = delete;
  AffineStatePattern &operator=(const AffineStatePattern &) = delete;
  const std::vector<AffineStateSegment> &segments() const { return segments_; }
  std::optional<AffineStateLocation> lookup(Id state) const {
    auto it = states.find(state);
    if (it == states.end() || it->second.segment == SIZE_MAX || models[it->second.segment].dirty) return {};
    return AffineStateLocation{it->second.segment, it->second.index};
  }
  bool collect(Id root) {
    last_failure.clear(); uint64_t before = version, before_work = work;
    try {
      if (!eligible(root)) return true;
      struct Work { Id id; bool ready; };
      std::vector<Work> todo{{root, false}};
      while (!todo.empty()) {
        auto w = todo.back(); todo.pop_back(); tick();
        if (scanned.contains(w.id)) continue;
        if (scanned.size() >= limits.stored_nodes) throw Decline{"stored_node_budget"};
        const auto &n = a.nodes[w.id];
        if (!w.ready) {
          todo.push_back({w.id, true}); tick(n.args.size());
          for (Id child : n.args) if (!scanned.contains(child)) todo.push_back({child, false});
        } else { recognize(w.id); scanned.insert(w.id); }
      }
      if (before != version) finalize();
      a.stats["affine_state_certified_updates"] = certified_updates;
      a.stats["affine_state_segments"] = segments_.size();
      a.stats["affine_state_collect_work"] += work - before_work;
      return true;
    } catch (const Decline &d) {
      last_failure = d.reason; ++a.stats["affine_state_decline_" + last_failure]; return false;
    }
  }
  Id substitute(Id source, Id axis, Id value) {
    try { return substitute_impl(source, axis, value); }
    catch (const Decline &d) { last_failure = d.reason; return invalid; }
  }
  Id closed_prefix(size_t sid, Id index) {
    try { return prefix_impl(sid, index); }
    catch (const Decline &d) { last_failure = d.reason; return invalid; }
  }
  std::optional<AffineStateProof> rewrite(Id root) {
    if (!collect(root)) return {};
    try {
      Id replacement = rewrite_impl(root);
      if (replacement == root) return {};
      return AffineStateProof{replacement, {{"kind", "finite_affine_state"},
          {"segments", segments_.size()}, {"stored_updates_verified", certified_updates},
          {"rule", "exp potential difference induction; exact typed-read update instances"}}};
    } catch (const Decline &d) {
      last_failure = d.reason; ++a.stats["affine_state_decline_" + last_failure]; return {};
    }
  }
};
} // namespace rg
