#pragma once
#include "core.hpp"
#include "exponential_polynomial.hpp"
#include "scalar_sum.hpp"
#include "masked_sum_domain.hpp"
#include "coordinate_relations.hpp"
#include "xor_coordinate.hpp"
#include "certified_state_pattern_canonicalizer.hpp"
namespace rg {
struct Box {
  Z start, mult = 1;
  std::vector<std::pair<Z, Z>> axes;
};
inline bool cover_boxes(const std::vector<Box> &boxes, const Z &count,
                        Arena &a, const Z &uniform_multiplicity = Z(1)) {
  Z size = 0;
  std::map<Z, int> den{{Z(1), 1}};
  for (const auto &b : boxes) {
    Z sz = b.mult;
    std::map<Z, int> ds;
    for (const auto &[s, n] : b.axes) {
      if (s <= 0 || n < 0)
        return false;
      sz *= n;
      ++ds[s];
    }
    size += sz;
    for (auto &[s, k] : ds)
      den[s] = std::max(den[s], k);
  }
  if (size != count * uniform_multiplicity)
    return false;
  if (count == 0)
    return size == 0;
  using Poly = std::map<Z, Z>;
  Poly diff;
  auto factor = [&](Poly p, const Z &e) {
    Poly out = p;
    for (const auto &[pow, c] : p) {
      a.check();
      out[pow + e] -= c;
      ++a.stats["coverage_polynomial_visits"];
    }
    for (auto it = out.begin(); it != out.end();)
      if (it->second == 0)
        it = out.erase(it);
      else
        ++it;
    a.stats["coverage_polynomial_peak"] =
        std::max<uint64_t>(a.stats["coverage_polynomial_peak"], out.size());
    return out;
  };
  auto accumulate = [&](const Box &b) {
    if (b.mult == 0)
      return;
    Poly p{{b.start, b.mult}};
    auto ds = den;
    for (const auto &[s, n] : b.axes) {
      p = factor(std::move(p), s * n);
      --ds[s];
    }
    for (const auto &[s, k] : ds)
      for (int j = 0; j < k; ++j)
        p = factor(std::move(p), s);
    for (const auto &[e, c] : p)
      diff[e] += c;
  };
  for (const auto &b : boxes)
    accumulate(b);
  accumulate({0, -uniform_multiplicity, {{1, count}}});
  return std::all_of(diff.begin(), diff.end(),
                     [](const auto &p) { return p.second == 0; });
}
struct Spec {
  Id pattern;
  size_t slots, axis;
  Z count;
  Facts facts;
  bool maximum = false;
};
struct Key {
  Id sid;
  std::vector<Id> params;
  Id coef, residual;
  bool operator<(const Key &o) const {
    return std::tie(sid, params, coef, residual) <
           std::tie(o.sid, o.params, o.coef, o.residual);
  }
};
using Summary = std::map<Key, Id>;
// Persistent, flat contribution forest. Concatenation and binding are O(1);
// leaf/box materialization occurs only at closure and preserves multiplicity.
struct PieceNode {
  enum Kind { Leaf, Concat, Bind } kind;
  Id left = invalid, right = invalid, index = invalid, var = invalid;
  Z count = 0;
};
struct Checker {
  Arena &a;
  Arena patterns;
  CertifiedStatePatternCanonicalizer state_patterns;
  std::vector<Spec> specs;
  std::map<std::tuple<Id, size_t, size_t, Z, bool>, Id> spec_index;
  // Exact instantiated definition identities, local to this Checker. These
  // never enumerate reduction elements or infer equality from samples.
  std::unordered_map<Id, Id> call_definition_cache;
  std::map<std::tuple<Id, Z, bool>, Id> call_definitions;
  std::set<Id> active, opt_version;
  // Only actual reference reduction axes may create shifted max domains.
  // Free output coordinates are parameters, even when they have integer bounds.
  std::unordered_set<Id> reference_binders;
  std::map<std::string, std::vector<Id>> active_byroot;
  struct Entry {
    Id n;
    Summary summary;
  };
  std::unordered_map<Id, Entry> opt_cache;
  // Pure semantic normalization is shared by both sides. Unlike opt_cache it
  // contains no reference patterns, contribution summaries or proof results.
  std::unordered_map<Id, Id> canonical_cache;
  // Domain restriction changes a binder, not a logical member. Reuse each
  // dependent stored node once per (old axis, narrowed axis) substitution.
  std::map<std::tuple<Id, Id, Id>, Id> restricted_axis_cache;
  std::map<std::tuple<Id, Id, Z>, Z> masked_prefix_cache;
  std::map<std::tuple<Id, Id, Z>, std::pair<Id, Id>> restricted_sum_bodies;
  uint64_t domain_binder_serial = 0;
  struct Match {
    std::vector<Id> params;
    Id index, coef;
    bool operator<(const Match &o) const {
      return std::tie(params, index, coef) <
             std::tie(o.params, o.index, o.coef);
    }
  };
  std::map<std::tuple<Id, Id, std::vector<Id>>, std::vector<Match>> match_cache;
  // Anchor inventories contain no captures or proof results. Successful
  // reconstruction below is separately keyed by its complete capture tuple.
  std::unordered_map<Id, std::vector<Id>> pattern_anchor_cache, target_anchor_cache;
  struct BodyInstanceKey {
    Id pattern; std::vector<Id> captures;
    bool operator==(const BodyInstanceKey &) const = default;
  };
  struct BodyInstanceHash {
    size_t operator()(const BodyInstanceKey &key) const {
      size_t hash=std::hash<Id>{}(key.pattern);
      for(Id value:key.captures) hash^=std::hash<Id>{}(value)+0x9e3779b9+(hash<<6)+(hash>>2);
      return hash;
    }
  };
  std::unordered_map<BodyInstanceKey,Id,BodyInstanceHash> body_instance_cache;
  std::unordered_map<Id,bool> body_instance_cacheable;
  struct ReadAnchorKey {
    Id pattern,target; std::vector<Id> captures;
    bool operator==(const ReadAnchorKey &) const = default;
  };
  struct ReadAnchorHash {
    size_t operator()(const ReadAnchorKey &key) const {
      size_t hash=std::hash<Id>{}(key.pattern);
      hash^=std::hash<Id>{}(key.target)+0x9e3779b9+(hash<<6)+(hash>>2);
      for(Id value:key.captures)hash^=std::hash<Id>{}(value)+0x9e3779b9+(hash<<6)+(hash>>2);
      return hash;
    }
  };
  std::unordered_map<ReadAnchorKey,std::vector<std::vector<Id>>,ReadAnchorHash> read_anchor_matches;
  using PayloadSource = std::tuple<std::string, std::string, std::vector<Q>>;
  struct PayloadAdmission {
    bool positive_shell = true;
    std::set<Id> opaque;
    std::map<PayloadSource, std::set<Id>> reads;
  };
  std::unordered_map<Id, PayloadAdmission> payload_admission_cache;
  // Only bounded affine predicate shells enter deferred matching. The cached
  // descriptor contains slot numbers, never instantiated numerical ancestry.
  std::unordered_map<Id, std::optional<std::vector<size_t>>> predicate_coordinate_slots;
  // A witness for one actual parameter tuple must never become a property of
  // every instantiation of the reference template.
  std::map<std::pair<Id, std::vector<Id>>, Facts> scoped_call_facts;
  std::unordered_map<Id, Id> coefficient_normal_cache;
  std::map<std::pair<Id, Id>, bool> coefficient_equal_cache;
  std::vector<PieceNode> pieces;
  std::map<std::string, uint64_t> failures;
  // Different complete coverage proofs describe the same exact contribution.
  // Export these equalities to the existing per-comparison proof reuse bank;
  // partial matches never enter this list.
  std::vector<std::pair<Id,Id>> complete_reduction_equalities;
  bool exponential_mode = false;
  ExponentialPolynomial scalar;
  Checker(Arena &ar, bool ep = false) : a(ar), state_patterns(patterns, a),
      exponential_mode(ep), scalar(ar) { patterns.deadline = a.deadline; }
  Id canonical_coefficient(Id value) {
    if (!exponential_mode || !a.nodes[value].facts.finite) return value;
    if (auto it = coefficient_normal_cache.find(value); it != coefficient_normal_cache.end())
      return it->second;
    scalar.reset_query();
    Id result = scalar.canonicalize(value);
    coefficient_normal_cache.emplace(value, result);
    ++a.stats["coefficient_normalizations"];
    return result;
  }
  bool equal_coefficients(Id left, Id right) {
    if (left == right) return true;
    left = canonical_coefficient(left); right = canonical_coefficient(right);
    if (left == right) return true;
    if (!exponential_mode) return false;
    auto key = std::minmax(left, right);
    auto [it, fresh] = coefficient_equal_cache.emplace(std::pair(key.first, key.second), false);
    if (fresh) it->second = scalar.equal(left, right);
    else ++a.stats["coefficient_equality_cache_hits"];
    return it->second;
  }
  // The same producer-side witness interface is used by import and all
  // reconstruction paths; Checker does not keep a second property engine.
  Id coordinate_at_zero(Id root, Id axis) {
    return a.coordinate_at_zero(root, axis);
  }
  bool reduction_witness_at_zero(Id root, Id axis, const Z &count, bool positive) {
    return a.reduction_witness_at_zero(root, axis, count, positive);
  }
  bool positive_at_zero(Id body, Id axis, const Z &count) {
    return a.positive_at_zero(body, axis, count);
  }
  Facts reduction_facts(Id initial, Id body, Id axis, const Z &count, bool maximum) {
    return a.fold_facts(initial, body, Q(count), maximum, axis);
  }
  Id leaf(Id index) {
    pieces.push_back({PieceNode::Leaf, invalid, invalid, index, invalid, 0});
    return Id(pieces.size() - 1);
  }
  Id concat(Id l, Id r) {
    pieces.push_back({PieceNode::Concat, l, r, invalid, invalid, 0});
    ++a.stats["summary_concat_nodes"];
    return Id(pieces.size() - 1);
  }
  Id bind_piece(Id p, Id v, Z n) {
    pieces.push_back({PieceNode::Bind, p, invalid, invalid, v, std::move(n)});
    ++a.stats["summary_bind_nodes"];
    return Id(pieces.size() - 1);
  }
  std::tuple<Id, size_t, size_t, std::vector<Id>> pattern(
      Id root, Id var, bool allow_parameterized = true, bool allow_derived = true) {
    std::unordered_map<Id, Id> slots;
    std::vector<Id> slot_values;
    std::unordered_map<uint64_t, Id> memo;
    struct Work {
      Id n;
      bool coord, ready;
    };
    std::vector<Work> todo{{root, false, false}};
    uint64_t visits = 0;
    bool anchored = false, used_parameterized = false, used_derived = false;
    std::map<std::vector<Q>, unsigned> coordinate_pairs;
    auto key = [](Id n, bool c) { return (uint64_t(n) << 1) | c; };
    while (!todo.empty()) {
      a.check();
      auto w = todo.back();
      todo.pop_back();
      auto k = key(w.n, w.coord);
      if (memo.contains(k))
        continue;
      const auto &n = a.nodes[w.n];
      if (w.n == var || (w.coord && !a.depends(w.n, var))) {
        if (w.n == var) anchored = true;
        auto [it, newslot] = slots.emplace(w.n, Id(slots.size()));
        if (newslot)
          slot_values.push_back(w.n);
        memo[k] = patterns.intern("slot", {}, {Q(it->second)});
        continue;
      }
      // Record quotient/remainder relationships before the binder subtree
      // becomes a slot. The surrounding read/call structure remains exact.
      if (allow_derived && w.coord) {
        if (auto relation = CoordinateRelations::describe(a, w.n, var)) {
          auto [it, fresh] = slots.emplace(var, Id(slots.size()));
          if (fresh) slot_values.push_back(var);
          (*relation)[0] = Q(it->second);
          if (relation->size() == 7) {
            auto key = CoordinateRelations::pair_key(*relation);
            if (!coordinate_pairs.contains(key) && coordinate_pairs.size() == CoordinateRelations::max_groups)
              return pattern(root, var, allow_parameterized, false);
            coordinate_pairs[key] |= (*relation)[5] == 0 ? 1u : 2u;
          }
          used_derived = true;
          memo[k] = patterns.intern("derived_coordinate", {}, std::move(*relation));
          continue;
        }
        if (auto recipe = CoordinateRelations::describe_sum(a, w.n, var)) {
          auto [it, fresh] = slots.emplace(var, Id(slots.size()));
          if (fresh) slot_values.push_back(var);
          Id axis = it->second;
          Q residual_slot(-1);
          if (recipe->residual != invalid) {
            auto [ri, added] = slots.emplace(recipe->residual, Id(slots.size()));
            if (added) slot_values.push_back(recipe->residual);
            residual_slot = Q(ri->second);
          }
          std::vector<Id> relations;
          for (auto &relation : recipe->relations) {
            relation[0] = Q(axis);
            relations.push_back(patterns.intern("derived_coordinate", {}, std::move(relation)));
          }
          std::vector<Q> data{Q(axis), residual_slot, recipe->constant};
          data.insert(data.end(), recipe->coefficients.begin(), recipe->coefficients.end());
          memo[k] = patterns.intern("coordinate_sum", std::move(relations), std::move(data));
          used_derived = true;
          continue;
        }
      }
      // A coordinate such as k+128 is one affine capture, not a tree that
      // must have the same constant/parenthesization as each optimized tile.
      // A parameterized residual is captured only after another coordinate
      // has anchored the binder. No multivariable address equation is solved.
      if (w.coord && (n.op == "add" || n.op == "mul") && n.args.size() <= 8) {
        auto terms = a.terms(w.n);
        Q coefficient(0), offset(0);
        bool affine = true, parameterized = false;
        for (auto [term, scale] : terms) {
          if (scale.denominator() != 1) { affine = false; break; }
          if (term == var) coefficient += scale;
          else if (term == a.one) offset += scale;
          else if (!a.depends(term, var) && a.nodes[term].facts.finite)
            parameterized = true;
          else { affine = false; break; }
        }
        if (affine && coefficient != 0 && coefficient.denominator() == 1 &&
            offset.denominator() == 1 && (allow_parameterized || !parameterized)) {
          auto [it, added] = slots.emplace(var, Id(slots.size()));
          if (added) slot_values.push_back(var);
          Id axis = it->second;
          if (parameterized) {
            used_parameterized = true;
            std::vector<Id> residual_terms;
            for (auto [term, scale] : terms)
              if (term != var)
                residual_terms.push_back(a.mul({a.constant(scale), term}));
            Id residual = a.add(std::move(residual_terms));
            auto [ri, fresh] = slots.emplace(residual, Id(slots.size()));
            if (fresh) slot_values.push_back(residual);
            memo[k] = patterns.intern("affine_param", {},
                                     {Q(axis), Q(ri->second), coefficient});
          } else {
            anchored = true;
            memo[k] = patterns.intern("affine_slot", {},
                                     {Q(axis), coefficient, offset});
          }
          continue;
        }
      }
      if (!w.coord && n.op == "fold")
        throw Unsupported("reference nested reduction not closed");
      bool childcoord = w.coord || n.op == "read" || n.op == "readbits" ||
                        n.op == "packedbits" || n.op == "call" || n.op == "certified_state";
      if (!w.ready) {
        todo.push_back({w.n, w.coord, true});
        for (size_t i = n.args.size(); i-- > 0;)
          if (!memo.contains(key(n.args[i], childcoord)))
            todo.push_back({n.args[i], childcoord, false});
        continue;
      }
      std::vector<Id> as;
      as.reserve(n.args.size());
      for (Id x : n.args)
        as.push_back(memo.at(key(x, childcoord)));
      memo[k] = patterns.intern(n.op, std::move(as), n.data, n.tag);
      ++visits;
    }
    a.stats["pattern_build_unique_nodes"] += visits;
    a.stats["pattern_build_peak_nodes"] =
        std::max(a.stats["pattern_build_peak_nodes"], visits);
    // Without an independent anchor, retain the previous structural path:
    // equal-shaped parameterized addresses can still bind k and each input
    // separately. Rebuild this whole pattern at most once, never per coordinate.
    bool paired = std::any_of(coordinate_pairs.begin(), coordinate_pairs.end(),
                             [](const auto &entry) {
                               return entry.second == 3 && (entry.first[1] == 1 || entry.first[1] == -1);
                             });
    if (allow_derived && used_derived && !anchored && !paired) {
      ++a.stats["derived_coordinate_structural_fallbacks"];
      return pattern(root, var, allow_parameterized, false);
    }
    if (allow_parameterized && used_parameterized && !anchored && !paired) {
      ++a.stats["parameterized_affine_structural_fallbacks"];
      return pattern(root, var, false, allow_derived);
    }
    if (!slots.contains(var))
      throw Unsupported(
          "reference has no reduction coordinate capture");
    return {memo.at(key(root, false)), slots.size(), slots.at(var),
            slot_values};
  }
  Id intern_spec(Id p, size_t slots, size_t axis, Z count, Facts f,
                 bool maximum = false) {
    auto k = std::tuple(p, slots, axis, count, maximum);
    auto it = spec_index.find(k);
    if (it != spec_index.end())
      return it->second;
    Id id = Id(specs.size());
    spec_index.emplace(std::move(k), id);
    specs.push_back({p, slots, axis, std::move(count), f, maximum});
    return id;
  }
  Id call(Id sid, const std::vector<Id> &args) {
    const auto s = specs[sid];
    Facts facts{s.facts.finite, s.count > 0 && s.facts.pos,
                s.count == 0 || s.facts.nn};
    if (s.maximum && s.count > 0)
      facts.finite_or_neginf = s.facts.finite_or_negative_infinity();
    if (auto it = scoped_call_facts.find({sid, args}); it != scoped_call_facts.end()) facts = it->second;
    Id raw = a.intern("call", args, {Q(sid)}, "",
                    facts);
    if (auto it = call_definition_cache.find(raw); it != call_definition_cache.end()) {
      active.insert(a.nodes[it->second].data[0].numerator().convert_to<Id>());
      return it->second;
    }
    if (!s.facts.finite || args.size()+1 != s.slots || s.slots>128) return raw;
    std::vector<Id> captures(s.slots); size_t next=0;
    for(size_t i=0;i<s.slots;++i) {
      if(i==s.axis) captures[i]=patterns.intern("definition_bound_axis",{},{});
      else {
        Id arg=args[next++];
        captures[i]=a.value(arg) ? patterns.constant(*a.value(arg))
          : patterns.intern("definition_actual",{},{Q(arg)});
      }
    }
    std::unordered_map<Id,Id> memo;
    std::vector<std::pair<Id,bool>> todo{{s.pattern,false}};
    bool supported=true; size_t work=0;
    while(!todo.empty()) {
      a.check(); auto [id,ready]=todo.back();todo.pop_back();
      if(memo.contains(id))continue;
      if(++work>512 || memo.size()>=128){supported=false;break;}
      const auto &n=patterns.nodes[id];
      if(n.args.size()>128){supported=false;break;}
      ++a.stats["call_definition_node_visits"];
      if(n.op=="slot") {memo[id]=captures.at(n.data[0].numerator().convert_to<size_t>());continue;}
      // Conservative exact structural specialization. Rich coordinate recipes
      // keep their existing identity until separately proven equivalent.
      if(n.op!="const" && n.op!="read" && n.op!="mul" && n.op!="add") {supported=false;break;}
      if(!ready){todo.emplace_back(id,true);for(Id child:n.args)if(!memo.contains(child))todo.emplace_back(child,false);continue;}
      std::vector<Id> children;for(Id child:n.args)children.push_back(memo.at(child));
      auto data=n.data;
      if(n.op=="mul")std::sort(children.begin(),children.end());
      if(n.op=="add") {
        std::vector<std::pair<Id,Q>> terms;for(size_t i=0;i<children.size();++i)terms.emplace_back(children[i],data[i]);
        std::sort(terms.begin(),terms.end());
        for(size_t i=0;i<terms.size();++i){children[i]=terms[i].first;data[i]=terms[i].second;}
      }
      memo[id]=patterns.intern(n.op,std::move(children),std::move(data),n.tag);
    }
    Id result=raw;
    if(supported) {
      auto [it,inserted]=call_definitions.emplace(std::tuple(memo.at(s.pattern),s.count,s.maximum),raw);
      result=it->second;
      if(!inserted && result!=raw)++a.stats["identical_instantiated_reduction_calls"];
    }
    call_definition_cache.emplace(raw,result);
    active.insert(a.nodes[result].data[0].numerator().convert_to<Id>());
    return result;
  }
  // A paged score Call carries the logical axis through e.g.
  // page_table[k/D] and k%D, so no individual argument is the axis itself.
  // Register its maximum vocabulary only for a real enclosing reference
  // reduction binder and an exact one-step quotient/remainder pair. This
  // merely defines a template; OnlineSoftmax still proves every Max domain's
  // complete, multiplicity-one coverage before canonicalizing any pivot.
  void register_paired_maximum(Id call_id, Id reduced_axis) {
    constexpr size_t max_binders = 8, max_nodes = 32, max_edges = 64;
    const auto &call_node = a.nodes[call_id];
    if (call_node.op != "call" || !call_node.facts.finite || call_node.args.size() > 8 ||
        reference_binders.size() > max_binders) return;
    if (std::none_of(reference_binders.begin(), reference_binders.end(),
          [&](Id axis) { return axis != reduced_axis && a.depends(call_id, axis); })) return;
    // Check the whole stored argument DAG before invoking pattern(), whose
    // dependency queries otherwise could walk a large unrelated parameter.
    std::vector<Id> todo = call_node.args, coordinate_nodes;
    std::unordered_set<Id> seen;
    size_t edges = 0;
    while (!todo.empty()) {
      a.check(); Id id = todo.back(); todo.pop_back();
      if (!seen.insert(id).second) continue;
      if (seen.size() > max_nodes) return;
      const auto &node = a.nodes[id];
      if (node.args.size() > 8 || edges + node.args.size() > max_edges) return;
      if (node.op == "fold" || node.op == "call") return;
      edges += node.args.size();
      if (node.op == "idiv" || node.op == "imod") coordinate_nodes.push_back(id);
      todo.insert(todo.end(), node.args.begin(), node.args.end());
    }
    if (coordinate_nodes.empty()) return;
    for (Id axis : reference_binders) {
      if (axis == reduced_axis || !seen.contains(axis)) continue;
      const auto &variable = a.nodes[axis];
      if (variable.op != "var" || variable.data.size() != 2 || !variable.bounds ||
          variable.bounds->first != 0 || variable.bounds->second < 1 ||
          variable.bounds->second.denominator() != 1 ||
          variable.data[0] != variable.bounds->first || variable.data[1] != variable.bounds->second) continue;
      std::map<std::vector<Q>, unsigned> pairs;
      for (Id coordinate : coordinate_nodes) {
        auto relation = CoordinateRelations::describe(a, coordinate, axis);
        if (!relation || relation->size() != 7 ||
            ((*relation)[1] != 1 && (*relation)[1] != -1)) continue;
        auto key = CoordinateRelations::pair_key(*relation);
        pairs[key] |= (*relation)[5] == 0 ? 1u : 2u;
      }
      if (std::none_of(pairs.begin(), pairs.end(),
                       [](const auto &entry) { return entry.second == 3; })) continue;
      auto [p, slots, dimension, values] = pattern(call_id, axis);
      Id auxiliary = intern_spec(p, slots, dimension, variable.bounds->second.numerator() + 1,
                                 call_node.facts, true);
      if (active.insert(auxiliary).second) ++a.stats["paired_maximum_reference_domains"];
    }
  }
  Id reg(Id body, Id var, Z count, bool maximum=false) {
    // Scalar normalization belongs to the body equality query.
    auto [p, ns, axis, slots] = pattern(body, var);
    if (!maximum && a.nodes[body].facts.finite) {
      if (auto canonical = state_patterns.canonicalize(p, ns, axis)) {
        std::vector<Id> reordered(ns);
        for (size_t i = 0; i < ns; ++i) reordered[canonical->old_to_new[i]] = slots[i];
        p = canonical->pattern;
        axis = canonical->axis;
        slots = std::move(reordered);
      }
    }
    Id sid = intern_spec(p, ns, axis, count, a.nodes[body].facts, maximum);
    active.insert(sid);
    std::vector<Id> args;
    for (size_t i = 0; i < slots.size(); ++i)
      if (i != axis)
        args.push_back(slots[i]);
    const auto &bf = a.nodes[body].facts;
    if (maximum && count > 0 && bf.finite_or_negative_infinity() &&
        reduction_witness_at_zero(body, var, count, false))
      scoped_call_facts.emplace(std::pair(sid, args), Facts{true, bf.pos, bf.nn, true});
    else if (!maximum && bf.finite && bf.nn && count > 0 && positive_at_zero(body, var, count))
      scoped_call_facts.emplace(std::pair(sid, args), Facts{true, true, true});
    Id c = call(sid, args);
    for (size_t j = 0; j < args.size(); ++j) {
      const auto &n = a.nodes[args[j]];
      // A reference split may pass k+offset to its score Call. Keep this
      // window in the maximum definition: matching an absolute key must then
      // recover k=key-offset, not widen the window to [0,offset+count).
      // Only one isolated unit-stride coordinate is supported here. Checking
      // the immediate Add before pattern() keeps work proportional to Call
      // arity; independent coordinates are captured without producer walks.
      if (n.op == "add" && n.args.size() == 2 && n.data.size() == 2) {
        Id index = invalid;
        std::optional<Q> offset;
        for (size_t k = 0; k < n.args.size(); ++k) {
          if (n.args[k] == a.one && n.data[k].denominator() == 1)
            offset = n.data[k];
          else if (n.data[k] == 1 && a.nodes[n.args[k]].op == "var")
            index = n.args[k];
        }
        if (index != invalid && offset && reference_binders.contains(index)) {
          const auto &v = a.nodes[index];
          bool isolated = true;
          for (size_t k = 0; k < args.size(); ++k)
            if (k != j && a.depends(args[k], index)) { isolated = false; break; }
          if (isolated && v.data.size() == 2 && v.bounds &&
              v.data[0] == 0 && v.data[1].denominator() == 1 && v.data[1] >= 1 &&
              v.bounds->first == v.data[0] && v.bounds->second == v.data[1] &&
              n.bounds && n.bounds->first == *offset &&
              n.bounds->second == *offset + v.data[1]) {
            auto [mp, ms, ma, values] = pattern(c, index);
            Id aux = intern_spec(mp, ms, ma, v.data[1].numerator() + 1,
                                 a.nodes[c].facts, true);
            active.insert(aux);
            ++a.stats["shifted_maximum_reference_domains"];
          }
        }
        continue;
      }
      if (n.op != "var" || !n.bounds || n.bounds->first != 0 ||
          n.bounds->second < 1)
        continue;
      std::vector<Id> ps;
      for (size_t i = 0; i < args.size(); ++i)
        ps.push_back(patterns.intern("slot", {}, {Q(i)}));
      Id pat = patterns.intern("call", ps, {Q(sid)});
      Id aux =
          intern_spec(pat, args.size(), j, n.bounds->second.numerator() + 1,
                      a.nodes[c].facts, true);
      active.insert(aux);
    }
    register_paired_maximum(c, var);
    return c;
  }
  Id neutral(Id sid) { return specs[sid].maximum ? a.infinity(-1) : a.zero; }
  Id plus(Id sid, std::vector<Id> xs) {
    return specs[sid].maximum ? a.maximum(xs) : a.add(xs);
  }
  Summary select(const Summary &s, bool max) {
    Summary r;
    for (const auto &[k, p] : s)
      if (specs[k.sid].maximum == max)
        r.emplace(k, p);
    return r;
  }
  // Nonrecursive local matching. Only commutative nodes branch. Exact capture
  // agreement is enforced on every occurrence, even with shared pattern nodes.
  std::optional<Id> instantiated_coordinate(Id root, const std::vector<Id> &captures) {
    // Only the bounded coordinate pattern is evaluated, never an operand's
    // numerical ancestry. All variables must already have exact captures.
    std::unordered_map<Id, Id> memo;
    std::vector<std::pair<Id, bool>> todo{{root, false}};
    size_t visits = 0;
    while (!todo.empty()) {
      a.check();
      if (++visits > 128) return {};
      auto [id, ready] = todo.back(); todo.pop_back();
      if (memo.contains(id)) continue;
      const auto &p = patterns.nodes[id];
      if (p.args.size() > 8) return {};
      if (p.op == "slot") {
        size_t slot = p.data.at(0).numerator().convert_to<size_t>();
        if (slot >= captures.size() || captures[slot] == invalid) return {};
        memo[id] = captures[slot]; continue;
      }
      if (p.op == "var" && p.args.empty()) {
        // An invariant variable outside read coordinates remains a literal
        // pattern leaf. Reuse its exact Arena identity; do not invent facts.
        auto existing = a.find_node(p.op, p.args, p.data, p.tag);
        if (!existing) return {};
        memo[id] = *existing; continue;
      }
      if (p.op != "const" && p.op != "add" && p.op != "mul" &&
          p.op != "idiv" && p.op != "imod") return {};
      if (!ready) {
        todo.emplace_back(id, true);
        for (Id child : p.args) if (!memo.contains(child)) todo.emplace_back(child, false);
        continue;
      }
      std::vector<Id> args; for (Id child : p.args) args.push_back(memo.at(child));
      Id value = invalid;
      if (p.op == "const" && p.data.size() == 1) value = a.constant(p.data[0]);
      else if (p.op == "add" && args.size() == p.data.size()) {
        for (size_t j = 0; j < args.size(); ++j) args[j] = a.mul({a.constant(p.data[j]), args[j]});
        value = a.add(args);
      } else if (p.op == "mul" && p.data.size() == 1) {
        args.push_back(a.constant(p.data[0])); value = a.mul(args);
      } else if ((p.op == "idiv" || p.op == "imod") && args.size() == 1 &&
                 p.data.size() == 1 && p.data[0].denominator() == 1 && p.data[0] > 0)
        value = p.op == "idiv" ? a.idiv(args[0], p.data[0].numerator())
                                  : a.imod(args[0], p.data[0].numerator());
      if (value == invalid) return {};
      memo[id] = value;
    }
    ++a.stats["instantiated_coordinate_patterns"];
    return memo.at(root);
  }
  const std::optional<std::vector<size_t>> &affine_predicate_slots(Id root) {
    if (auto it = predicate_coordinate_slots.find(root); it != predicate_coordinate_slots.end())
      return it->second;
    const auto &predicate = patterns.nodes[root];
    bool valid = predicate.op == "compare" && predicate.args.size() == 2 && predicate.data.empty() &&
                 (predicate.tag == "eq" || predicate.tag == "ne" || predicate.tag == "lt" ||
                  predicate.tag == "le" || predicate.tag == "gt" || predicate.tag == "ge");
    std::vector<size_t> slots;
    std::vector<Id> todo = valid ? predicate.args : std::vector<Id>{};
    std::unordered_set<Id> seen;
    size_t edges = 0;
    while (valid && !todo.empty()) {
      a.check(); Id id = todo.back(); todo.pop_back();
      if (!seen.insert(id).second) continue;
      const auto &n = patterns.nodes[id];
      if (seen.size() > 32 || n.args.size() > 8 || n.args.size() > 64 - edges) { valid = false; break; }
      edges += n.args.size();
      if (n.op == "slot" && n.args.empty() && n.data.size() == 1 &&
          n.data[0].denominator() == 1 && n.data[0] >= 0 && n.data[0] <= Q(SIZE_MAX))
        slots.push_back(n.data[0].numerator().convert_to<size_t>());
      else if ((n.op == "const" && n.args.empty() && n.data.size() == 1) ||
               (n.op == "var" && n.args.empty() && n.data.size() == 2)) {}
      else if ((n.op == "add" && n.data.size() == n.args.size()) ||
               (n.op == "mul" && n.args.size() == 1 && n.data.size() == 1))
        todo.insert(todo.end(), n.args.begin(), n.args.end());
      else valid = false;
    }
    std::optional<std::vector<size_t>> result;
    if (valid) {
      std::sort(slots.begin(), slots.end());
      slots.erase(std::unique(slots.begin(), slots.end()), slots.end());
      result = std::move(slots);
    }
    return predicate_coordinate_slots.emplace(root, std::move(result)).first->second;
  }
  std::vector<std::vector<Id>> matches(Id pat, Id target,
                                       std::vector<Id> captures, bool fixed_parameters = false) {
    struct Task {
      Id pattern, target;
      size_t child = SIZE_MAX;
      std::vector<Id> remaining;
      bool coordinate = false;
      bool predicate_structural = false;
    };
    struct State {
      std::vector<Task> todo;
      std::vector<Id> cap;
      std::vector<Task> deferred;
      std::vector<std::pair<Id, Id>> coordinate_constraints;
      std::vector<std::pair<Id, Id>> coordinate_sums;
      std::vector<std::pair<Id, Id>> fixed_coordinates;
      std::vector<std::pair<Id, Id>> predicates;
    };
    std::vector<State> states{
        {{{pat, target, SIZE_MAX, {}}}, std::move(captures), {}, {}, {}, {}, {}}};
    std::vector<std::vector<Id>> out;
    uint64_t work = 0;
    // Arena orders addends by local node Id. Substitution can change that
    // order, so small sums match by (coefficient, child), not by position.
    // Keep wider sums on the existing ordered path; the global work budget
    // also bounds combinations of nested small commutative nodes.
    auto small_add = [](const auto &n) {
      return n.op == "add" && n.args.size() > 1 && n.args.size() <= 4 &&
             n.data.size() == n.args.size();
    };
    auto compatible = [&](Id p, Id n, const std::vector<Id> &cap, bool deferred_predicate = true) {
      const auto &pn = patterns.nodes[p];
      if (pn.op == "slot") {
        size_t slot = pn.data[0].numerator().convert_to<size_t>();
        return slot >= cap.size() || cap[slot] == invalid || cap[slot] == n;
      }
      if (pn.op == "affine_slot" || pn.op == "affine_param" || pn.op == "derived_coordinate" ||
          pn.op == "coordinate_sum") return true;
      const auto &an = a.nodes[n];
      if (deferred_predicate && pn.op == "compare" &&
          (an.op == "compare" || n == a.zero || n == a.one) && affine_predicate_slots(p))
        return true;
      if (pn.op != an.op || pn.tag != an.tag || pn.args.size() != an.args.size())
        return false;
      if (small_add(pn) && small_add(an))
        return std::is_permutation(pn.data.begin(), pn.data.end(), an.data.begin());
      return pn.data == an.data;
    };
    while (!states.empty()) {
      State s = std::move(states.back());
      states.pop_back();
      bool ok = true;
      while (!s.todo.empty()) {
        a.check();
        if (++work > 1000000)
          throw Unsupported("local pattern matching work budget exceeded");
        Task task = std::move(s.todo.back());
        s.todo.pop_back();
        const auto &pn = patterns.nodes[task.pattern];
        const auto &an = a.nodes[task.target];
        if (task.child != SIZE_MAX) {
          if (task.child == pn.args.size())
            continue;
          // Branch only over candidates surviving this child's shallow shape
          // and current captures, then check the child before the next choice.
          std::vector<Id> choices;
          std::set<Id> seen;
          for (Id j : task.remaining)
            if ((pn.op != "add" || pn.data[task.child] == an.data[j]) &&
                seen.insert(an.args[j]).second &&
                compatible(pn.args[task.child], an.args[j], s.cap))
              choices.push_back(j);
          if (choices.empty()) {
            ok = false;
            break;
          }
          auto schedule = [&](State &next, Id j) {
            Task continuation = task;
            ++continuation.child;
            continuation.remaining.erase(
                std::find(continuation.remaining.begin(),
                          continuation.remaining.end(), j));
            next.todo.push_back(std::move(continuation));
            next.todo.push_back(
                {pn.args[task.child], an.args[j], SIZE_MAX, {}});
          };
          for (size_t i = 1; i < choices.size(); ++i) {
            State next = s;
            schedule(next, choices[i]);
            states.push_back(std::move(next));
            ++a.stats["pattern_branch_states"];
          }
          schedule(s, choices[0]);
          continue;
        }
        ++a.counter(Arena::Hot::pattern_node_visits);
        if (!task.predicate_structural && pn.op == "compare" &&
            (an.op == "compare" || task.target == a.zero || task.target == a.one) &&
            affine_predicate_slots(task.pattern)) {
          const auto predicate = std::pair(task.pattern, task.target);
          if (std::find(s.predicates.begin(), s.predicates.end(), predicate) == s.predicates.end()) {
            if (s.predicates.size() == CoordinateRelations::max_constraints) { ok = false; break; }
            s.predicates.push_back(predicate);
          }
          continue;
        }
        if (fixed_parameters && task.coordinate &&
            (pn.op == "add" || pn.op == "mul" || pn.op == "idiv" || pn.op == "imod")) {
          if (s.fixed_coordinates.size() == CoordinateRelations::max_constraints) { ok = false; break; }
          s.fixed_coordinates.emplace_back(task.pattern, task.target);
          continue;
        }
        if (!compatible(task.pattern, task.target, s.cap, !task.predicate_structural)) {
          ok = false;
          break;
        }
        if (pn.op == "slot") {
          size_t slot = pn.data[0].numerator().convert_to<size_t>();
          if (slot >= s.cap.size())
            s.cap.resize(slot + 1, invalid);
          s.cap[slot] = task.target;
          continue;
        }
        if (pn.op == "derived_coordinate") {
          auto constraint = std::pair(task.pattern, task.target);
          if (std::find(s.coordinate_constraints.begin(), s.coordinate_constraints.end(), constraint) ==
              s.coordinate_constraints.end()) {
            if (s.coordinate_constraints.size() == CoordinateRelations::max_constraints) { ok = false; break; }
            s.coordinate_constraints.push_back(constraint);
          }
          continue;
        }
        if (pn.op == "coordinate_sum") {
          // A decoder reuses its input in the fraction, exponent and sign.
          // Identical constraints are one proof obligation, not independent
          // work. Different actual coordinates remain separate obligations.
          auto constraint = std::pair(task.pattern, task.target);
          if (std::find(s.coordinate_sums.begin(), s.coordinate_sums.end(), constraint) == s.coordinate_sums.end()) {
            if (s.coordinate_sums.size() == CoordinateRelations::max_constraints) { ok = false; break; }
            s.coordinate_sums.push_back(constraint);
          }
          continue;
        }
        if (pn.op == "affine_param") {
          s.deferred.push_back(std::move(task));
          continue;
        }
        if (pn.op == "affine_slot") {
          size_t slot = pn.data[0].numerator().convert_to<size_t>();
          Id captured = a.mul({a.constant(Q(1) / pn.data[1]),
                               a.sub(task.target, a.constant(pn.data[2]))});
          if (slot >= s.cap.size()) s.cap.resize(slot + 1, invalid);
          if (s.cap[slot] != invalid && s.cap[slot] != captured) {
            ok = false;
            break;
          }
          s.cap[slot] = captured;
          ++a.stats["affine_coordinate_captures"];
          continue;
        }
        if (((pn.op == "mul" || pn.op == "max") && pn.args.size() > 1) ||
            small_add(pn)) {
          std::vector<Id> remaining(an.args.size());
          std::iota(remaining.begin(), remaining.end(), Id(0));
          s.todo.push_back(
              {task.pattern, task.target, 0, std::move(remaining)});
        } else {
          for (size_t j = pn.args.size(); j-- > 0;)
            s.todo.push_back({pn.args[j], an.args[j], SIZE_MAX, {},
                pn.op == "read" || pn.op == "readbits" || pn.op == "packedbits" || pn.op == "call"});
        }
      }
      // Known output parameters anchor the one unknown axis directly. This
      // is a single affine equation, not a search for an index transformation.
      if (ok && fixed_parameters) for (const auto &task : s.deferred) {
        const auto &p = patterns.nodes[task.pattern];
        size_t axis = p.data[0].numerator().convert_to<size_t>();
        size_t residual = p.data[1].numerator().convert_to<size_t>();
        if (axis < s.cap.size() && s.cap[axis] == invalid && residual < s.cap.size() &&
            s.cap[residual] != invalid && p.data[2] != 0 &&
            a.nodes[task.target].args.size() <= 8 && a.nodes[s.cap[residual]].args.size() <= 8) {
          s.cap[axis] = a.mul({a.constant(Q(1) / p.data[2]), a.sub(task.target, s.cap[residual])});
          ++a.stats["parameter_anchored_reduction_axes"];
        }
      }
      if (ok && !s.coordinate_constraints.empty())
        ok = CoordinateRelations::resolve(a, patterns, s.coordinate_constraints, s.cap, work);
      // Parameterized affine residuals require an axis captured independently:
      // an ordinary/literal-affine coordinate or a verified quotient/remainder
      // pair. Residual tasks never determine their own axis.
      if (ok) for (const auto &task : s.deferred) {
        a.check();
        if (++work > 1000000)
          throw Unsupported("local pattern matching work budget exceeded");
        const auto &pn = patterns.nodes[task.pattern];
        size_t axis = pn.data[0].numerator().convert_to<size_t>();
        size_t residual = pn.data[1].numerator().convert_to<size_t>();
        if (axis >= s.cap.size() || s.cap[axis] == invalid ||
            !a.nodes[s.cap[axis]].facts.finite ||
            !a.nodes[task.target].facts.finite ||
            a.nodes[s.cap[axis]].args.size() > 8 ||
            a.nodes[task.target].args.size() > 8) {
          ok = false;
          break;
        }
        Id captured = a.sub(task.target,
                            a.mul({a.constant(pn.data[2]), s.cap[axis]}));
        if (residual >= s.cap.size()) s.cap.resize(residual + 1, invalid);
        if (s.cap[residual] != invalid && s.cap[residual] != captured) {
          ok = false;
          break;
        }
        s.cap[residual] = captured;
        ++a.stats["parameterized_affine_coordinate_captures"];
      }
      // Short coordinate sums may only use an axis already captured by an
      // ordinary coordinate or a verified quotient/remainder pair. Each
      // residual parameter must agree with every other occurrence; existing
      // summary binding still rejects parameters dependent on a Fold binder.
      if (ok) for (const auto &[pattern, actual] : s.coordinate_sums) {
        if (!CoordinateRelations::resolve_sum(a, patterns, pattern, actual, s.cap, work)) {
          ok = false;
          break;
        }
      }
      if (ok) for (const auto &[pattern, actual] : s.fixed_coordinates) {
        auto expected = instantiated_coordinate(pattern, s.cap);
        if (!expected || *expected != actual) { ok = false; break; }
      }
      if (ok) for (const auto &[pattern, actual] : s.predicates) {
        const auto &slots = *affine_predicate_slots(pattern);
        const bool bound = std::all_of(slots.begin(), slots.end(), [&](size_t slot) {
          return slot < s.cap.size() && s.cap[slot] != invalid;
        });
        if (!bound) {
          // No independent coordinate anchor: preserve the original exact
          // structural match. This one-way requeue cannot defer itself again.
          s.todo.push_back({pattern, actual, SIZE_MAX, {}, false, true});
          ++a.stats["predicate_structural_fallbacks"];
          continue;
        }
        const auto &predicate = patterns.nodes[pattern];
        auto lhs = instantiated_coordinate(predicate.args[0], s.cap);
        auto rhs = instantiated_coordinate(predicate.args[1], s.cap);
        const auto &actual_node = a.nodes[actual];
        Id canonical_actual = actual_node.op == "compare" && actual_node.args.size() == 2
            ? a.compare(actual_node.tag, actual_node.args[0], actual_node.args[1]) : actual;
        ++a.stats["deferred_predicate_checks"];
        if (!lhs || !rhs || a.compare(predicate.tag, *lhs, *rhs) != canonical_actual) {
          ++a.stats["deferred_predicate_mismatches"]; ok = false; break;
        }
      }
      if (ok && !s.todo.empty()) {
        // The completed coordinate obligations remain valid: captures only
        // become bound, never change. Do not repeat their work after fallback.
        s.deferred.clear(); s.coordinate_constraints.clear(); s.coordinate_sums.clear();
        s.fixed_coordinates.clear(); s.predicates.clear();
        states.push_back(std::move(s));
        continue;
      }
      if (ok)
        out.push_back(std::move(s.cap));
    }
    return out;
  }

  bool cacheable_read_anchor(Id pattern) const {
    const auto &p=patterns.nodes[pattern];
    if((p.op!="read" && p.op!="readbits" && p.op!="packedbits") || p.args.size()>8)return false;
    for(Id coordinate:p.args) {
      const auto &n=patterns.nodes[coordinate];
      if(n.op=="coordinate_sum") {
        if(n.args.empty() || n.args.size()>4)return false;
        for(Id relation:n.args) {
          const auto &child=patterns.nodes[relation];
          if(child.op!="derived_coordinate" || !child.args.empty())return false;
        }
      } else if(!n.args.empty() || (n.op!="slot" && n.op!="affine_slot" && n.op!="affine_param" &&
                n.op!="derived_coordinate" && n.op!="const" && n.op!="var")) {
        return false;
      }
    }
    return true;
  }
  std::vector<std::vector<Id>> match_anchor(Id pattern,Id target,const std::vector<Id> &captures) {
    // These are closed local constraints: no opaque lookup, deferred
    // predicate, Call registration or new Arena nodes can change the result.
    // Cache the exact existing matcher result, including an empty result.
    // Short coordinate recipes run all original integer/scope/residual guards
    // on the first query. Only that validated result is reusable; merely
    // reconstructing an existing Read is never a domain proof.
    if(!cacheable_read_anchor(pattern))return matches(pattern,target,captures);
    ReadAnchorKey key{pattern,target,captures};
    if(auto found=read_anchor_matches.find(key);found!=read_anchor_matches.end()) {
      ++a.stats["ep_body_read_match_cache_hits"];return found->second;
    }
    auto result=matches(pattern,target,captures);
    read_anchor_matches.emplace(std::move(key),result);
    ++a.stats["ep_body_read_match_cache_entries"];
    return result;
  }

  // Reconstruct the reference body under a candidate index correspondence.
  // Opaque leaves must already exist in this Arena: no facts are transferred
  // from a differently indexed reference read.
  Id instantiate_body(Id root, const std::vector<Id> &cap) {
    auto eligibility=body_instance_cacheable.find(root);
    const bool reusable=eligibility==body_instance_cacheable.end() || eligibility->second;
    if(reusable) if(auto hit=body_instance_cache.find({root,cap});hit!=body_instance_cache.end()) {
      ++a.stats["ep_body_instance_cache_hits"];return hit->second;
    }
    bool encountered_call=false;
    std::unordered_map<Id,Id> memo;
    std::function<Id(Id,unsigned)> visit = [&](Id id,unsigned depth)->Id {
      if(depth>128 || memo.size()>2048) return invalid;
      if(auto it=memo.find(id);it!=memo.end()) return it->second;
      const auto &n=patterns.nodes[id];
      auto slot=[&](size_t j){return j<cap.size()?cap[j]:invalid;};
      auto index=[&](size_t j){return n.data[j].numerator().convert_to<size_t>();};
      Id out=invalid;
      if(n.op=="slot") return slot(index(0));
      if(n.op=="affine_slot" || n.op=="affine_param") {
        Id axis=slot(index(0)); if(axis==invalid)return invalid;
        if(n.op=="affine_slot") out=a.add({a.mul({a.constant(n.data[1]),axis}),a.constant(n.data[2])});
        else {Id residual=slot(index(1));if(residual==invalid)return invalid;
          out=a.add({a.mul({a.constant(n.data[2]),axis}),residual});}
      } else if(n.op=="derived_coordinate") {
        Id axis=slot(index(0));if(axis==invalid)return invalid;
        out=CoordinateRelations::evaluate(a,axis,n.data);
      } else {
        std::vector<Id> xs;for(Id child:n.args){Id v=visit(child,depth+1);if(v==invalid)return invalid;xs.push_back(v);}
        if(n.op=="coordinate_sum") {
          std::vector<Id> terms{a.constant(n.data[2])};
          if(n.data[1]>=0){Id v=slot(index(1));if(v==invalid)return invalid;terms.push_back(v);}
          for(size_t j=0;j<xs.size();++j)terms.push_back(a.mul({a.constant(n.data[3+j]),xs[j]}));
          out=a.add(terms);
        } else if(n.op=="const") out=a.constant(n.data[0]);
        else if(n.op=="add" || n.op=="add_raw") {
          for(size_t j=0;j<xs.size();++j)xs[j]=a.mul({a.constant(n.data[j]),xs[j]});
          out=a.add(xs);
        } else if(n.op=="mul") {xs.push_back(a.constant(n.data[0]));out=a.mul(xs);}
        else if(n.op=="idiv" || n.op=="imod") {
          // Captured indices may simplify a reference quotient/remainder to
          // a different operator (or a literal). Reuse the exact coordinate
          // constructors, just as instantiated_coordinate() does for the
          // candidate check; an unsimplified node need not exist in Arena.
          if(xs.size()!=1 || n.data.size()!=1 ||
             n.data[0].denominator()!=1 || n.data[0]<=0) return invalid;
          out=n.op=="idiv" ? a.idiv(xs[0],n.data[0].numerator())
                           : a.imod(xs[0],n.data[0].numerator());
          ++a.stats["ep_body_coordinate_rebuilds"];
        }
        else if(n.op=="signed") {
          if(xs.size()!=1 || n.data.size()!=1 || n.data[0].denominator()!=1 ||
             n.data[0]<=0 || n.data[0]>256) return invalid;
          out=a.signed_word(xs[0],n.data[0].numerator().convert_to<unsigned>());
        }
        else if(n.op=="word_or" && xs.size()==2 && n.data.size()==1)
          out=a.word_or(xs[0],xs[1],n.data[0].numerator().convert_to<unsigned>());
        else if(n.op=="word_shr" && xs.size()==2 && n.data.size()==2)
          out=a.word_shr(xs[0],xs[1],n.data[0].numerator().convert_to<unsigned>(),n.data[1]!=0);
        else if(n.op=="extractbits" && xs.size()==1 && n.data.size()==2)
          out=a.bit_slice(xs[0],n.data[0].numerator().convert_to<unsigned>(),n.data[1].numerator().convert_to<unsigned>());
        else if(n.op=="exp" || n.op=="exp2")out=a.exponential(xs[0],n.op=="exp2");
        else if(n.op=="recip")out=a.divide(a.one,xs[0]);
        else if(n.op=="compare") {
          // Candidate substitution can prove a predicate on a narrower binder
          // domain. Derive the result from the actual operands, not facts
          // attached to the reference pattern or an opaque-node lookup.
          if(xs.size()!=2 || !n.data.empty() ||
             (n.tag!="eq" && n.tag!="ne" && n.tag!="lt" && n.tag!="le" &&
              n.tag!="gt" && n.tag!="ge"))return invalid;
          out=a.compare(n.tag,xs[0],xs[1]);
          ++a.stats["ep_body_predicate_rebuilds"];
        }
        else if(n.op=="select") {
          if(xs.size()!=3 || !n.data.empty() || !n.tag.empty())return invalid;
          out=a.select(xs[0],xs[1],xs[2]);
          ++a.stats["ep_body_select_rebuilds"];
        }
        else if(n.op=="call") {
          encountered_call=true;
          // A captured parameter changes the instantiated definition, not
          // only a Call's argument list. Reuse the same exact definition
          // canonicalizer as registration and ordinary reconstruction.
          if(n.data.size()!=1 || n.data[0].denominator()!=1 ||
             n.data[0]<0 || n.data[0]>=Q(specs.size()))return invalid;
          Id sid=n.data[0].numerator().convert_to<Id>();
          if(xs.size()+1!=specs[sid].slots)return invalid;
          out=call(sid,xs);
          ++a.stats["ep_body_call_rebuilds"];
        }
        else {
          if(auto hit=a.find_node(n.op,xs,n.data,n.tag))out=*hit;
          if(out==invalid)++a.stats["ep_body_missing_opaque_"+n.op];
        }
      }
      memo.emplace(id,out);return out;
    };
    Id result=visit(root,0);
    // A successful call-free construction depends only on immutable template
    // and capture IDs. Missing opaque nodes can become available later, so
    // failures are never cached. Call construction also registers active
    // definitions and must retain that side effect on every invocation.
    if(encountered_call)body_instance_cacheable[root]=false;
    else if(result!=invalid) {
      body_instance_cacheable[root]=true;
      body_instance_cache.emplace(BodyInstanceKey{root,cap},result);
      ++a.stats["ep_body_instance_cache_entries"];
    }
    return result;
  }
  // Anchor matching proposes coordinates only; ScalarEqual checks the entire
  // body. Bounded DAG traversal never enumerates the reduction domain.
  std::vector<std::vector<Id>> body_candidates(const Spec &s,Id target,
                                             const std::vector<Id> &initial) {
    if(initial.size()!=s.slots)return {};
    auto anchors=[&](Arena &arena,Id root,bool pattern)->const std::vector<Id>&{
      auto &cache=pattern?pattern_anchor_cache:target_anchor_cache;
      if(auto found=cache.find(root);found!=cache.end()) {
        ++a.counter(pattern?Arena::Hot::ep_body_pattern_anchor_cache_hits:Arena::Hot::ep_body_target_anchor_cache_hits);
        return found->second;
      }
      // Coordinates in a direct operand must not be starved by one sibling's
      // numerical history. Traverse stored nodes breadth first; anchor nodes
      // remain opaque, and all proposals still require a complete body proof.
      // Bound discovery itself, including wide operand arrays and the queue.
      std::vector<Id> out,todo{root};std::unordered_set<Id> seen{root};
      size_t cursor=0,edges=0;bool limited=false;
      while(cursor<todo.size() && out.size()<16){
        Id id=todo[cursor++];
        const auto &n=arena.nodes[id];
        if(n.op=="slot" || n.op=="var" || n.op=="read" || n.op=="readbits" || n.op=="packedbits" || n.op=="call" || n.op=="certified_state")out.push_back(id);
        else {
          // A nested Fold owns its body's coordinates. Discovering a read
          // there would bind this scalar query to a bound member of an entire
          // reduction. The reduction consumer handles that body separately;
          // only the seed remains in the current lexical scope. Registered
          // Calls above are already opaque and retain delayed rebinding.
          const bool binder=n.op=="fold";
          if(binder)++a.stats["ep_body_fold_scope_boundaries"];
          const size_t children=binder?std::min<size_t>(1,n.args.size()):n.args.size();
          for(size_t j=0;j<children;++j){
            Id child=n.args[j];
            if(edges==2048){limited=true;break;}
            ++edges;
            if(seen.contains(child))continue;
            if(seen.size()==512){limited=true;continue;}
            seen.insert(child);todo.push_back(child);
          }
        }
      }
      a.stats["ep_body_anchor_nodes"]+=cursor;
      a.stats["ep_body_anchor_edges"]+=edges;
      if(limited || cursor<todo.size())++a.stats["ep_body_anchor_limited"];
      if(pattern)std::stable_partition(out.begin(),out.end(),[&](Id id){
        const auto &op=arena.nodes[id].op;return op!="slot" && op!="var";
      });
      return cache.emplace(root,std::move(out)).first->second;
    };
    const auto &ps=anchors(patterns,s.pattern,true), &ts=anchors(a,target,false);
    // Reads/calls bind coordinates even when a bounds-proved predicate has
    // disappeared from the actual body. A redundant bare slot is not another
    // required occurrence of that coordinate. This orders candidate discovery,
    // never proof: instantiate_body + ScalarEqual still check the whole body.
    std::vector<std::vector<Id>> caps{initial};
    size_t attempts=0;
    for(Id pat:ps){
      std::vector<std::vector<Id>> next;
      const auto &pn=patterns.nodes[pat];
      for(const auto &cap:caps){
        // Only abstract anchors may have a different registered spelling of
        // the same instantiated definition. Direct input anchors still have
        // to agree on the object, typed view and captured coordinates, even
        // when an earlier read has already bound every slot. Otherwise a
        // proper factor of a body can trigger a full EP proof of that body.
        const bool deferred_definition = pn.op=="call" || pn.op=="certified_state";
        if(deferred_definition && std::find(cap.begin(),cap.end(),invalid)==cap.end()) {
          ++a.stats["ep_body_deferred_call_anchors"];
          next.push_back(cap);continue;
        }
        if(pn.op=="var" || (pn.op=="slot" && pn.data.size()==1 &&
            pn.data[0].denominator()==1 && pn.data[0]>=0 && pn.data[0]<Q(cap.size()) &&
            cap[pn.data[0].numerator().convert_to<size_t>()]!=invalid)) {
          next.push_back(cap);continue;
        }
        if(cacheable_read_anchor(pat) &&
           std::find(cap.begin(),cap.end(),invalid)==cap.end()) {
          Id expected=instantiate_body(pat,cap);
          if(expected!=invalid && std::find(ts.begin(),ts.end(),expected)!=ts.end()) {
            const auto validated=match_anchor(pat,expected,cap);
            if(std::find(validated.begin(),validated.end(),cap)!=validated.end()) {
              ++a.stats["ep_body_bound_read_exact_hits"];
              next.push_back(cap);continue;
            }
          }
          // A miss is not a failed constraint. The general anchor matcher
          // remains available; no unchecked reconstruction is accepted.
          ++a.stats["ep_body_bound_read_matcher_fallbacks"];
        }
        bool discovered=false;
        for(Id t:ts){
        if(++attempts>256){++a.stats["ep_body_candidate_budget"];return {};}
        const auto &tn=a.nodes[t];
        if(pn.op=="slot" ? tn.op!="var" : (pn.op!=tn.op || pn.tag!=tn.tag))continue;
        for(auto c:match_anchor(pat,t,cap)) {
          discovered=true;
          if(std::find(next.begin(),next.end(),c)==next.end()) {
            next.push_back(std::move(c));if(next.size()>32){++a.stats["ep_body_candidate_budget"];return {};}}
        }
        }
        // A Call can be rebound only after independent anchors supply its
        // actual parameters. Keep that delayed proposal, not arbitrary failed
        // read proposals. This is bounded candidate discovery, not a proof of
        // inequality: unsupported/absent anchors decline this search path;
        // every admitted candidate still requires the complete body proof.
        if(!discovered && !deferred_definition &&
            (pn.op=="read" || pn.op=="readbits" || pn.op=="packedbits"))
          ++a.counter(Arena::Hot::ep_body_read_anchor_rejections);
        if(!discovered && deferred_definition && std::find(next.begin(),next.end(),cap)==next.end()) {
          next.push_back(cap);
          ++a.stats["ep_body_deferred_unmatched_anchors"];
          ++a.stats["ep_body_deferred_call_anchors"];
          if(next.size()>32){++a.stats["ep_body_candidate_budget"];return {};}
        }
      }
      if(next.empty())return {};
      caps=std::move(next);
    }
    return caps;
  }

  // This is a conservative search filter for positive polynomial shells, not
  // an inequality theorem. It must not reject cancellation or exponential
  // coefficient identities: negative coefficients and unfamiliar opaque
  // atoms leave admission undecided and retain the complete EP proof.
  const PayloadAdmission &payload_admission(Id root) {
    if(auto found=payload_admission_cache.find(root);found!=payload_admission_cache.end())
      return found->second;
    PayloadAdmission result;
    std::vector<Id> todo{root};std::unordered_set<Id> seen;
    size_t edges=0;
    while(!todo.empty() && result.positive_shell) {
      Id id=todo.back();todo.pop_back();
      if(!seen.insert(id).second)continue;
      if(seen.size()>128){result.positive_shell=false;break;}
      const auto &node=a.nodes[id];
      if(!node.facts.finite){result.positive_shell=false;break;}
      if(node.op=="const") {
        if(node.data.size()!=1 || node.data[0]<0)result.positive_shell=false;
      } else if(node.op=="add" || node.op=="mul") {
        if(node.data.empty() || std::any_of(node.data.begin(),node.data.end(),
                                           [](const Q &coefficient){return coefficient<=0;})) {
          result.positive_shell=false;break;
        }
        if(edges+node.args.size()>256){result.positive_shell=false;break;}
        edges+=node.args.size();todo.insert(todo.end(),node.args.begin(),node.args.end());
      } else if(node.op=="recip") {
        result.positive_shell=false;
      } else if(node.op=="read" || node.op=="readbits" || node.op=="packedbits") {
        result.reads[{node.op,node.tag,node.data}].insert(id);
      } else if(node.op=="decode_fp4" && node.args.size()==1 &&
                (a.nodes[node.args[0]].op=="packedbits" || a.nodes[node.args[0]].op=="readbits")) {
        // A typed decoded input is an EP payload atom, not a traversal into
        // its bit-level arithmetic or an enumeration of packed elements.
        const auto &input=a.nodes[node.args[0]];
        result.reads[{node.op+"/"+input.op,input.tag,input.data}].insert(id);
      } else result.opaque.insert(id);
    }
    a.stats["ep_body_payload_summary_nodes"]+=seen.size();
    return payload_admission_cache.emplace(root,std::move(result)).first->second;
  }
  bool admit_payload_coordinates(Id expected,Id actual) {
    if(expected==actual)return true;
    // The reduction consumer already owns bounded factor decomposition and
    // the later binder-independence check. Do not ask a whole-body equality
    // query to discard an extra ordinary input factor. This is exact root
    // factor containment, not division and not a recovered coefficient.
    const auto &e=a.nodes[expected], &v=a.nodes[actual];
    if(e.op=="mul" && v.op=="mul" && e.data==v.data &&
       e.args.size()<v.args.size() && v.args.size()<=16) {
      std::vector<bool> used(v.args.size(),false);bool contained=true;
      for(Id factor:e.args) {
        size_t j=0;while(j<v.args.size() && (used[j] || v.args[j]!=factor))++j;
        if(j==v.args.size()){contained=false;break;}
        used[j]=true;
      }
      if(contained) {
        bool ordinary=true;
        for(size_t j=0;j<v.args.size();++j)if(!used[j]) {
          const auto &extra=a.nodes[v.args[j]];
          ordinary &= extra.facts.finite &&
            (extra.op=="read" || extra.op=="readbits" || extra.op=="packedbits");
        }
        if(ordinary){++a.stats["ep_body_factor_decomposition_admissions"];return false;}
      }
    }
    const auto &wanted=payload_admission(expected), &found=payload_admission(actual);
    if(!wanted.positive_shell || !found.positive_shell)return true;
    // A different Exp/Select/Call can require genuine algebra or delayed
    // definition binding. Never classify it by looking through its ancestry.
    for(Id opaque:found.opaque)if(!wanted.opaque.contains(opaque))return true;
    for(const auto &[source,reads]:found.reads) {
      // An unrelated outside input factor cannot hide conflicting coordinates
      // of a source already constrained by this candidate. It is left for the
      // existing factor-decomposition consumer, not interpreted here.
      if(!wanted.reads.contains(source))continue;
      const auto &allowed=wanted.reads.at(source);
      for(Id read:reads)if(!allowed.contains(read)) {
        ++a.stats["ep_body_payload_admission_rejections"];
        return false;
      }
    }
    return true;
  }

  // One bounded local congruence: a product of finite factors may contain a
  // single zero-mask condition and exponentials of one base. No nested mask
  // paths are enumerated. Equal masks and equal residual factors leave only
  // an exponential coefficient to be proved independent of the Sum binder.
  struct MaskedExponentialTerm {
    Id predicate, exponent, residual;
    bool base2;
  };
  std::optional<MaskedExponentialTerm> masked_exponential_term(Id root) {
    if (!a.nodes[root].facts.finite) return {};
    Id predicate = a.one;
    std::vector<Id> exponent, residual{a.one}, todo{root};
    std::optional<bool> base2;
    size_t work = 0;
    while (!todo.empty()) {
      Id id = todo.back(); todo.pop_back();
      if (++work > 24) return {};
      const auto &node = a.nodes[id];
      if (!node.facts.finite) return {};
      if (node.op == "mul" && node.data.size() == 1 && node.args.size() <= 8) {
        const auto factors = node.args;
        const Q scalar = node.data[0];
        residual.push_back(a.constant(scalar));
        todo.insert(todo.end(), factors.begin(), factors.end());
      } else if (node.op == "select" && node.args.size() == 3 &&
                 (node.args[1] == a.zero || node.args[2] == a.zero)) {
        const auto branches = node.args;
        Id condition = branches[0];
        if (branches[1] == a.zero) condition = a.compare("eq", condition, a.zero);
        if (predicate != a.one && !scalar.alpha.equal(predicate, condition)) return {};
        predicate = condition;
        Id value = branches[branches[1] == a.zero ? 2 : 1];
        // Only one mask layer per factor. Its numerical child is not searched
        // for another condition; otherwise this would become path expansion.
        if (a.nodes[value].op == "select") return {};
        todo.push_back(value);
      } else if ((node.op == "exp" || node.op == "exp2") && node.args.size() == 1 &&
                 a.nodes[node.args[0]].facts.finite) {
        bool binary = node.op == "exp2";
        if (base2 && *base2 != binary) return {};
        base2 = binary;
        exponent.push_back(node.args[0]);
      } else residual.push_back(id);
    }
    if (!base2 || exponent.empty()) return {};
    return MaskedExponentialTerm{predicate, a.add(exponent), a.mul(residual), *base2};
  }
  std::optional<Id> masked_body_coefficient(Id expected, Id actual, bool allow_proved_mask = false) {
    auto left = masked_exponential_term(expected), right = masked_exponential_term(actual);
    if (!left || !right || left->base2 != right->base2 ||
        !scalar.alpha.equal(left->predicate, right->predicate)) return {};
    if (left->predicate == a.one && !allow_proved_mask) return {};
    // Match the actual reference exponent before constructing a scale. The
    // tautology exp(T)=exp(S)*exp(T-S) is not a useful reduction match: choosing
    // S from a different key would manufacture one unique compensation factor
    // per anchor, even when both indices are constants. Reuse the ordinary
    // exponent match's local Add/coefficient contract instead. Numerical
    // ancestors remain opaque identities; there is no candidate key search.
    constexpr size_t maximum_exponent_terms = 32;
    auto bounded_shell = [&](Id exponent) {
      const auto &shell = a.nodes[exponent];
      return (shell.op != "add" && shell.op != "mul") ||
             shell.args.size() <= maximum_exponent_terms;
    };
    if (!bounded_shell(left->exponent) || !bounded_shell(right->exponent)) return {};
    auto expected_terms = a.terms(left->exponent), actual_terms = a.terms(right->exponent);
    if (expected_terms.size() > maximum_exponent_terms || actual_terms.size() > maximum_exponent_terms)
      return {};
    std::map<Id, Q> remaining, expected_by_identity;
    for (const auto &[term, scale] : actual_terms)
      remaining[scalar.alpha.representative(term)] += scale;
    for (const auto &[term, scale] : expected_terms)
      expected_by_identity[scalar.alpha.representative(term)] += scale;
    bool matched_anchor = false;
    for (const auto &[term, scale] : expected_by_identity) {
      if (scale == 0) continue;
      if (term == a.one) { remaining[term] -= scale; continue; }
      auto found = remaining.find(term);
      if (found == remaining.end() || found->second != scale) {
        ++a.stats["masked_exponential_unmatched_score_terms"];
        return {};
      }
      matched_anchor = true;
      remaining.erase(found);
    }
    if (!matched_anchor) return {};
    if (left->residual != right->residual &&
        (!exponential_mode || !scalar.equal(left->residual, right->residual))) return {};
    std::vector<Id> offsets;
    for (const auto &[term, scale] : remaining)
      if (scale != 0) offsets.push_back(a.mul({a.constant(scale), term}));
    Id difference = a.add(std::move(offsets));
    if (!a.nodes[difference].facts.finite) return {};
    Id coefficient = canonical_coefficient(a.exponential(difference, left->base2));
    if (!a.nodes[coefficient].facts.finite || !a.nodes[coefficient].facts.pos) return {};
    ++a.stats["masked_exponential_coefficient_proofs"];
    return coefficient;
  }

  const std::vector<Match> &match_term(Id sid, Id n, const std::vector<Id> &parameters = {}) {
    auto key = std::tuple(sid, n, parameters);
    auto it = match_cache.find(key);
    if (it != match_cache.end())
      return it->second;
    ExponentialPolynomial::Timer match_timer(scalar.profiling?&scalar.match_ns:nullptr);
    const auto &s = specs[sid];
    const auto &p = patterns.nodes[s.pattern];
    const auto &node = a.nodes[n];
    bool masked_reference = p.op == "select";
    if (p.op == "mul" && p.args.size() <= 8)
      for (Id child : p.args) masked_reference |= patterns.nodes[child].op == "select";
    std::vector<Id> initial(s.slots, invalid);
    bool fixed = !parameters.empty();
    if (fixed) {
      if (parameters.size() + 1 != s.slots) throw Unsupported("reduction parameter arity mismatch");
      size_t j = 0;
      for (size_t k = 0; k < s.slots; ++k) if (k != s.axis) initial[k] = parameters[j++];
    }
    std::vector<std::pair<std::vector<Id>, Id>> rs;
    for (auto cap : matches(s.pattern, n, initial, fixed))
      rs.emplace_back(std::move(cap), a.one);
    auto expmatches = [&](Id ep, Id target, const std::vector<Id> &cap,
                          Id factor) {
      const auto &t = a.nodes[target];
      if (t.op != patterns.nodes[ep].op || !a.nodes[t.args[0]].facts.finite)
        return;
      const auto &sum = a.nodes[t.args[0]];
      if (sum.op != "add")
        return;
      for (size_t j = 0; j < sum.args.size(); ++j) {
        Id candidate = a.mul({a.constant(sum.data[j]), sum.args[j]});
        auto caps = matches(patterns.nodes[ep].args[0], candidate, cap, fixed);
        if (caps.empty())
          continue;
        std::vector<Id> rest;
        for (size_t k = 0; k < sum.args.size(); ++k)
          if (k != j)
            rest.push_back(a.mul({a.constant(sum.data[k]), sum.args[k]}));
        Id f = a.mul({factor, a.exponential(a.add(rest), t.op == "exp2")});
        for (auto c : caps)
          rs.emplace_back(std::move(c), f);
      }
    };
    if (rs.empty() && (p.op == "exp2" || p.op == "exp") && node.op == p.op)
      expmatches(s.pattern, n, initial, a.one);
    if (rs.empty() && p.op == "mul" && node.op == "mul" &&
        p.data == node.data && p.args.size() == 2 && node.args.size() == 2) {
      for (size_t j = 0; j < 2; ++j) {
        Id ep = p.args[j], other = p.args[1 - j];
        if (patterns.nodes[ep].op != "exp2" && patterns.nodes[ep].op != "exp")
          continue;
        for (size_t k = 0; k < 2; ++k)
          for (auto cap : matches(other, node.args[1 - k], initial, fixed))
            expmatches(ep, node.args[k], cap, a.one);
      }
    }
    if (exponential_mode && rs.empty())
      for(auto cap:body_candidates(s,n,initial))rs.emplace_back(std::move(cap),a.one);
    std::set<Match> unique;
    for (auto &[cap, f] : rs) {
      if (cap.size() != s.slots ||
          std::find(cap.begin(), cap.end(), invalid) != cap.end())
        continue;
      // EP anchors propose candidates; they cannot override parameters fixed
      // by the enclosing reduction query.
      if (fixed) {
        bool same=true;
        for (size_t k=0;k<cap.size();++k)
          if(k!=s.axis && cap[k]!=initial[k])same=false;
        if(!same)continue;
      }
      if(exponential_mode) {
        Id expected;
        {ExponentialPolynomial::Timer timer(scalar.profiling?&scalar.instantiate_ns:nullptr);expected=instantiate_body(s.pattern,cap);}
        if(expected==invalid){++a.stats["ep_body_instantiation_declines"];continue;}
        if(!admit_payload_coordinates(a.mul({expected,f}),n))continue;
        ++a.stats["ep_body_obligations"];
        bool equal;
        {ExponentialPolynomial::Timer timer(scalar.profiling?&scalar.body_scalar_ns:nullptr);
          // This theorem proves the complete masked product, not just its
          // positive branch. An opaque EP atom must not erase the mask.
          if (auto coefficient = masked_reference ? masked_body_coefficient(expected, n, true) : std::optional<Id>{}) {
            f = *coefficient; equal = true;
          } else equal=scalar.equal(a.mul({expected,f}),n);
        }
        if(!equal)continue;
        ++a.stats["ep_body_proofs"];
      }
      std::vector<Id> params;
      for (size_t j = 0; j < cap.size(); ++j)
        if (j != s.axis)
          params.push_back(cap[j]);
      unique.insert({std::move(params), cap[s.axis], canonical_coefficient(f)});
    }
    return match_cache
        .emplace(key, std::vector<Match>(unique.begin(), unique.end()))
        .first->second;
  }
  Summary combine(const std::vector<Summary> &parts) {
    Summary result;
    bool first = true;
    for (const auto &s : parts) {
      if (s.empty())
        continue;
      if (first) {
        result = s;
        first = false;
        continue;
      }
      using Prefix = std::pair<Id, std::vector<Id>>;
      std::map<Prefix, std::vector<std::pair<Key, Id>>> index;
      for (const auto &[k, p] : s)
        index[{k.sid, k.params}].emplace_back(k, p);
      Summary next;
      for (const auto &[k, p] : result) {
        auto it = index.find({k.sid, k.params});
        if (it == index.end())
          continue;
        for (const auto &[other, p2] : it->second) {
          a.check();
          if (!equal_coefficients(k.coef, other.coef)) continue;
          Key key = k;
          key.coef = canonical_coefficient(key.coef);
          key.residual = plus(k.sid, {k.residual, other.residual});
          next[key] = concat(p, p2);
        }
      }
      result = std::move(next);
    }
    return result;
  }
  Summary scale(const Summary &s, Id c) {
    Summary r;
    for (const auto &[k, p] : s) {
      Key key = k;
      key.coef = canonical_coefficient(a.mul({k.coef, c}));
      key.residual = a.mul({k.residual, c});
      r[key] = p;
    }
    return r;
  }
  Summary residual(const Summary &s, Id n) {
    Summary r;
    for (const auto &[k, p] : s) {
      Key key = k;
      key.residual = plus(k.sid, {k.residual, n});
      r[key] = p;
    }
    return r;
  }
  Summary bind(const Summary &s, Id var, const Z &count) {
    Summary r;
    for (const auto &[k, p] : s) {
      if (a.depends(k.coef, var) ||
          std::any_of(k.params.begin(), k.params.end(),
                      [&](Id x) { return a.depends(x, var); })) {
        ++failures["binder_in_fixed_parameter_or_factor"];
        continue;
      }
      if (a.depends(k.residual, var)) {
        ++failures["unmatched_binder_dependent_residual"];
        continue;
      }
      Key key = k;
      if (!specs[k.sid].maximum)
        key.residual = a.mul({a.constant(Q(count)), k.residual});
      r[key] = bind_piece(p, var, count);
    }
    return r;
  }
  std::optional<Box> box(Id index,
                         const std::vector<std::pair<Id, Z>> &domains) {
    std::map<Id, Q> ts;
    for (auto [t, c] : a.terms(index))
      ts[t] = c;
    Q start = 0;
    if (ts.contains(a.one)) {
      start = ts[a.one];
      ts.erase(a.one);
    }
    Box b;
    for (const auto &[v, n] : domains) {
      Q c = 0;
      if (ts.contains(v)) {
        c = ts[v];
        ts.erase(v);
      }
      if (c.denominator() != 1 || n < 0)
        return {};
      if (c == 0) {
        b.mult *= n;
        continue;
      }
      if (c < 0) {
        start += c * Q(n - 1);
        c = -c;
      }
      b.axes.emplace_back(c.numerator(), n);
    }
    for (auto it = ts.begin(); it != ts.end();)
      if (it->second == 0)
        it = ts.erase(it);
      else
        ++it;
    if (!ts.empty() || start.denominator() != 1)
      return {};
    b.start = start.numerator();
    return b;
  }
  std::optional<std::vector<Box>> boxes(Id p, const Z &count, bool &permuted) {
    struct W {
      Id id;
      bool leave;
    };
    std::vector<W> todo{{p, false}};
    std::vector<std::pair<Id, Z>> domains;
    std::vector<Box> out;
    std::optional<Id> common_mask;
    bool plain = false;
    permuted = false;
    while (!todo.empty()) {
      a.check();
      auto w = todo.back();
      todo.pop_back();
      const auto &n = pieces[w.id];
      if (w.leave) {
        domains.pop_back();
        continue;
      }
      if (n.kind == PieceNode::Concat) {
        todo.push_back({n.right, false});
        todo.push_back({n.left, false});
      } else if (n.kind == PieceNode::Bind) {
        domains.emplace_back(n.var, n.count);
        todo.push_back({w.id, true});
        todo.push_back({n.left, false});
      } else {
        Id index = n.index;
        const auto &coordinate = a.nodes[index];
        if (coordinate.op == "index_xor") {
          if (plain || coordinate.args.size() != 2 || coordinate.data.size() != 1 ||
              (coordinate.data[0] != 6 && coordinate.data[0] != 7) ||
              count != pow2(coordinate.data[0].numerator().convert_to<unsigned>())) return {};
          Id mask = coordinate.args[1];
          if (!bounded_coordinate_integer(a, coordinate.args[0], count) ||
              !bounded_coordinate_integer(a, mask, count) ||
              (common_mask && *common_mask != mask)) return {};
          for (const auto &[variable, length] : domains) {
            (void)length;
            if (a.depends(mask, variable)) return {};
          }
          common_mask = mask; permuted = true; index = coordinate.args[0];
        } else {
          if (common_mask) return {};
          plain = true;
        }
        auto b = box(index, domains);
        if (!b)
          return {};
        out.push_back(std::move(*b));
        ++a.stats["coverage_stored_piece_visits"];
      }
    }
    return out;
  }
  std::optional<Id> close(const Summary &s) {
    std::set<Id> closed;
    for (const auto &[k, p] : s) {
      bool permuted = false;
      auto bs = boxes(p, specs[k.sid].count, permuted);
      if (bs && cover_boxes(*bs, specs[k.sid].count, a)) {
        closed.insert(
            plus(k.sid, {k.residual, a.mul({k.coef, call(k.sid, k.params)})}));
        ++a.stats["completed_reductions"];
        if (permuted) ++a.stats["xor_permutation_complete_domains"];
      }
    }
    if (closed.size() > 1) {
      const auto representative=*closed.begin();
      for(auto alternative:closed)if(alternative!=representative)
        complete_reduction_equalities.emplace_back(representative,alternative);
      ++a.stats["multiple_complete_reduction_proofs"];
    }
    if (closed.empty())
      return {};
    return *closed.begin();
  }
  Id reapply(Id original, const std::vector<Id> &args) {
    const auto &n = a.nodes[original];
    const auto &op = n.op;
    const auto &d = n.data;
    if (op == "call") {
      if(d.size()!=1 || d[0].denominator()!=1 || d[0]<0 || d[0]>=Q(specs.size()))
        throw Unsupported("reduction Call reconstruction identity");
      Id sid=d[0].numerator().convert_to<Id>();
      if(args.size()+1!=specs[sid].slots)
        throw Unsupported("reduction Call reconstruction arity");
      return call(sid,args);
    }
    if (op == "fold" && args.size() == 2 && d.size() == 2 &&
        d[0].denominator() == 1 && d[1].denominator() == 1 && d[1] >= 0 &&
        (n.tag == "Sum" || n.tag == "Max")) {
      return canonicalize_fold(original, args);
    }
    if (op == "add" || op == "add_raw") {
      std::vector<Id> xs;
      for (size_t j = 0; j < args.size(); ++j)
        xs.push_back(a.mul({a.constant(d[j]), args[j]}));
      return a.factor_sum(a.add(xs));
    }
    if (op == "mul") {
      std::vector<Id> rec, others{a.constant(d[0])};
      for (Id x : args)
        if (a.nodes[x].op == "recip")
          rec.push_back(a.nodes[x].args[0]);
        else
          others.push_back(x);
      if (rec.size() == 1)
        return a.divide(a.mul(others), rec[0]);
      auto xs = args;
      xs.push_back(a.constant(d[0]));
      return a.mul(xs);
    }
    if (op == "signed") {
      if(args.size()!=1 || d.size()!=1 || d[0].denominator()!=1 || d[0]<=0 || d[0]>256)
        throw Unsupported("signed word reconstruction arity/width");
      return a.signed_word(args[0],d[0].numerator().convert_to<unsigned>());
    }
    if (op == "word_or")
      return a.word_or(args[0],args[1],d[0].numerator().convert_to<unsigned>());
    if (op == "word_shr")
      return a.word_shr(args[0],args[1],d[0].numerator().convert_to<unsigned>(),d[1]!=0);
    if (op == "extractbits")
      return a.bit_slice(args[0],d[0].numerator().convert_to<unsigned>(),d[1].numerator().convert_to<unsigned>());
    if ((op == "idiv" || op == "imod") && args.size() == 1 &&
        d.size() == 1 && d[0].denominator() == 1 && d[0] > 0)
      return op == "idiv" ? a.idiv(args[0], d[0].numerator())
                          : a.imod(args[0], d[0].numerator());
    if (op == "floor")
      return a.floor_value(args[0],n.bounds);
    if (op == "exp2" || op == "exp")
      return a.exponential(args[0], op == "exp2");
    if (op == "select")
      return a.select(args[0], args[1], args[2]);
    if (op == "compare")
      return a.compare(n.tag, args[0], args[1]);
    if (op == "xor")
      return a.bitxor(args.at(0), args.at(1));
    if (op == "abs")
      return a.absolute(args.at(0));
    if (op == "log" || op == "log2")
      return a.logarithm(args.at(0), op == "log2");
    if (op == "power")
      return a.power(args.at(0), args.at(1));
    if (op == "recip")
      return a.recip(args[0]);
    if (op == "max")
      return a.maximum(args);
    if (op == "clamp")
      return a.clamp(args[0],d[0],d[1]);
    if (op == "sqrt" || op == "reciprocalsqrt" || op == "sin" || op == "cos" ||
        op == "exp" || op == "tanh") {
      bool f = std::all_of(args.begin(), args.end(),
                           [&](Id x) { return a.nodes[x].facts.finite; });
      if (op == "sqrt") f &= a.nodes[args[0]].facts.nn;
      if (op == "reciprocalsqrt") f &= a.nodes[args[0]].facts.pos;
      bool p = f && (op == "reciprocalsqrt" || op == "exp" ||
                            (op == "sqrt" && a.nodes[args[0]].facts.pos));
      bool nn = p || (f && op == "sqrt");
      return a.intern(op, args, d, n.tag, {f, p, nn});
    }
    return args == n.args ? original
                          : a.intern(op, args, d, n.tag, n.facts, n.bounds);
  }
  // Cached free-variable sets make each independence query logarithmic in the
  // factor's free-variable count; this never traverses its producer DAG.
  std::pair<Id, Id> split_invariant_factors(Id body, Id axis) {
    const auto &n = a.nodes[body];
    if (!n.facts.finite) return {a.one, body};
    if (!a.depends(body, axis)) return {body, a.one};
    if (n.op != "mul" || n.data.size() != 1) return {a.one, body};
    std::vector<Id> invariant{a.constant(n.data[0])}, varying;
    for (Id factor : n.args) {
      a.check();
      ++a.stats["canonical_factor_visits"];
      (a.depends(factor, axis) ? varying : invariant).push_back(factor);
    }
    return {a.mul(invariant), a.mul(varying)};
  }

  // Capture-safe restriction of one free coordinate to a subset of its
  // declared domain. Invariant subgraphs are reused. Facts on immutable reads
  // remain valid under restriction; coordinate constructors recompute bounds.
  // The iteration domain is dependent stored DAG nodes, never Fold elements.
  Id restrict_axis(Id root, Id old_axis, Id new_axis) {
    auto key = [&](Id id) { return std::tuple(old_axis, new_axis, id); };
    std::vector<std::pair<Id, bool>> todo{{root, false}};
    while (!todo.empty()) {
      a.check();
      auto [id, ready] = todo.back(); todo.pop_back();
      if (restricted_axis_cache.contains(key(id))) {
        ++a.stats["canonical_domain_rebind_cache_hits"];
        continue;
      }
      if (id == old_axis) {
        restricted_axis_cache.emplace(key(id), new_axis);
        continue;
      }
      if (!a.depends(id, old_axis)) {
        restricted_axis_cache.emplace(key(id), id);
        ++a.stats["canonical_domain_invariant_reuse"];
        continue;
      }
      const auto &n = a.nodes[id];
      const bool shadow = n.op == "fold" && n.args.size() == 2 &&
                          n.data.size() == 2 && n.data[0] == Q(old_axis);
      if (!ready) {
        todo.emplace_back(id, true);
        for (size_t j = 0; j < n.args.size(); ++j)
          if (!(shadow && j == 1) && !restricted_axis_cache.contains(key(n.args[j])))
            todo.emplace_back(n.args[j], false);
        continue;
      }
      std::vector<Id> args;
      args.reserve(n.args.size());
      for (size_t j = 0; j < n.args.size(); ++j)
        args.push_back(shadow && j == 1 ? n.args[j]
                                      : restricted_axis_cache.at(key(n.args[j])));
      restricted_axis_cache.emplace(key(id), reapply(id, args));
      ++a.stats["canonical_domain_rebind_nodes"];
    }
    return restricted_axis_cache.at(key(root));
  }

  // Same orientation for reference and optimized, before either is abstracted:
  // Fold(z, c*f) -> z + c*Fold(0, f), never Fold(c*z, c*f).
  // The small shells materialize the SumNF view in the existing IR. No element
  // expansion, new IR operator, or reference-dependent alternative is needed.
  Id canonicalize_fold(Id original, const std::vector<Id> &args) {
    const auto &n = a.nodes[original];
    if (n.args.size() != 2 || n.data.size() != 2 ||
        n.data[0].denominator() != 1 || n.data[1].denominator() != 1 ||
        n.data[0] < 0 || n.data[0] >= Q(a.nodes.size()) || n.data[1] < 0)
      return args == n.args ? original : a.intern(n.op, args, n.data, n.tag, n.facts, n.bounds);
    Id axis = n.data[0].numerator().convert_to<Id>();
    Z count = n.data[1].numerator();
    if (count == 0) return args[0];
    bool sum = n.tag == "Sum";
    Id value = args[1];
    if (sum) {
      auto key = std::tuple(value, axis, count);
      auto it = masked_prefix_cache.find(key);
      if (it == masked_prefix_cache.end())
        it = masked_prefix_cache.emplace(key, masked_sum_prefix(a, value, axis, count)).first;
      if (it->second < count) {
        count = it->second;
        ++a.stats["canonical_sum_trimmed_domains"];
        if (count == 0) {
          ++a.stats["canonical_sum_zero_domains"];
          return args[0];
        }
        auto found = restricted_sum_bodies.find(key);
        if (found == restricted_sum_bodies.end()) {
          // A newly allocated binder cannot already occur free or bound in
          // the old body. Cache the restriction, not a potentially capturing
          // name shared by unrelated lexical scopes.
          Id fresh;
          size_t before;
          do {
            before = a.nodes.size();
            fresh = a.var("canonical-sum-domain:" + std::to_string(domain_binder_serial++),
                          Q(0), Q(count - 1));
          } while (fresh < before);
          Id restricted = restrict_axis(value, axis, fresh);
          found = restricted_sum_bodies.emplace(key, std::pair(fresh, restricted)).first;
        }
        axis = found->second.first;
        value = found->second.second;
      }
    }
    std::vector<Q> domain{Q(axis), Q(count)};
    Facts facts = reduction_facts(args[0], value, axis, count, n.tag == "Max");
    if (sum && facts.finite && !a.depends(args[0], axis)) {
      if (auto scalar_sum = simplify_scalar_sum(a, args[0], value, axis, count))
        return *scalar_sum;
      auto [coefficient, body] = split_invariant_factors(value, axis);
      if (coefficient != a.one || args[0] != a.zero) {
        if (coefficient != a.one) ++a.stats["canonical_sum_factor_pulls"];
        if (args[0] != a.zero) ++a.stats["canonical_sum_initials"];
        Id pure = a.intern("fold", {a.zero, body}, domain, "Sum",
                           reduction_facts(a.zero, body, axis, count, false));
        // The seed is outside the multiplication and still appears once.
        return a.factor_sum(a.add({args[0], a.mul({coefficient, pure})}));
      }
    }
    return a.intern("fold", {args[0], value}, domain, n.tag, facts);
  }

  Id canonicalize(Id root) {
    std::vector<std::pair<Id, bool>> todo{{root, false}};
    // Normalize newly emitted shells too. Already normalized immutable DAG
    // nodes stay cached; this is not a repeated whole-graph fixed-point scan.
    std::unordered_map<Id, Id> rewritten;
    while (!todo.empty()) {
      a.check();
      auto [id, ready] = todo.back(); todo.pop_back();
      if (canonical_cache.contains(id)) continue;
      if (auto it = rewritten.find(id); it != rewritten.end()) {
        canonical_cache.emplace(id, canonical_cache.at(it->second));
        continue;
      }
      const auto &n = a.nodes[id];
      const bool empty = n.op == "fold" && n.args.size() == 2 &&
                         n.data.size() == 2 && n.data[1] == 0;
      if (!ready) {
        todo.emplace_back(id, true);
        for (size_t j = 0; j < (empty ? 1 : n.args.size()); ++j)
          if (!canonical_cache.contains(n.args[j])) todo.emplace_back(n.args[j], false);
        continue;
      }
      if (empty) {
        canonical_cache.emplace(id, canonical_cache.at(n.args[0]));
        ++a.counter(Arena::Hot::canonical_node_visits);
        continue;
      }
      std::vector<Id> args;
      args.reserve(n.args.size());
      for (Id child : n.args) args.push_back(canonical_cache.at(child));
      Id result = n.op == "fold" ? canonicalize_fold(id, args) : reapply(id, args);
      if (result != id && !canonical_cache.contains(result)) {
        rewritten.emplace(id, result);
        todo.emplace_back(id, true);
        todo.emplace_back(result, false);
      } else {
        canonical_cache.emplace(id, result == id ? id : canonical_cache.at(result));
      }
      ++a.counter(Arena::Hot::canonical_node_visits);
    }
    return canonical_cache.at(root);
  }

  // This second stage is intentionally reference-guided: register definitions
  // on the reference side, prove contributions on the optimized side. The
  // algebraic canonicalizer above has no such side flag or active-spec state.
  Id normalize(Id root, bool reference = false) {
    root = canonicalize(root);
    std::unordered_map<Id, Entry> local;
    if (reference) {
      active.clear();
      reference_binders.clear();
    } else if (active != opt_version) {
      opt_version = active;
      opt_cache.clear();
      ++a.stats["normalization_cache_context_changes"];
    }
    auto &cache = reference ? local : opt_cache;
    std::vector<std::pair<Id, bool>> todo{{root, false}};
    std::unordered_map<Id, Id> redirects;
    while (!todo.empty()) {
      a.check();
      auto [old, ready] = todo.back();
      todo.pop_back();
      if (cache.contains(old))
        continue;
      if (auto it = redirects.find(old); it != redirects.end()) {
        cache.emplace(old, cache.at(it->second));
        continue;
      }
      const auto &raw = a.nodes[old];
      if (!ready) {
        // Visit the Fold before its body, so reg() can recognize an enclosing
        // reference axis without a second DAG traversal or variable-name tests.
        if (reference && raw.op == "fold")
          reference_binders.insert(raw.data[0].numerator().convert_to<Id>());
        todo.emplace_back(old, true);
        for (size_t j = raw.args.size(); j-- > 0;)
          if (!cache.contains(raw.args[j]))
            todo.emplace_back(raw.args[j], false);
        continue;
      }
      std::vector<Id> args;
      std::vector<const Summary *> child;
      for (Id x : raw.args) {
        auto &entry = cache.at(x);
        args.push_back(entry.n);
        child.push_back(&entry.summary);
      }
      Summary summary;
      Id n;
      if (raw.op == "fold") {
        // Child abstraction can reveal a simpler predicate or finite zero
        // mask. Re-enter the same normal form before binding summaries or
        // registering a definition; never bind with a stale domain/axis.
        Id rebuilt = canonicalize_fold(old, args);
        const auto &normalized = a.nodes[rebuilt];
        if (normalized.op != "fold" || normalized.args != args ||
            normalized.data != raw.data || normalized.tag != raw.tag) {
          redirects.emplace(old, rebuilt);
          todo.emplace_back(old, true);
          todo.emplace_back(rebuilt, false);
          ++a.stats["normalization_fold_rebuild_redirects"];
          continue;
        }
        Id var = raw.data[0].numerator().convert_to<Id>();
        Z count = raw.data[1].numerator();
        bool max = raw.tag == "Max";
        if (reference) {
          if(count==0) n=args[0];
          else {
            Id call=reg(args[1],var,count,max);
            n=max ? a.maximum({args[0],call}) : a.add({args[0],call});
          }
        }
        else {
          Summary initial = select(*child[0], max),
                  bound = bind(select(*child[1], max), var, count);
          if (!initial.empty() && !bound.empty())
            summary = combine({initial, bound});
          else if (!bound.empty())
            summary = residual(bound, args[0]);
          n = a.intern("fold", args, raw.data, raw.tag,
                       reduction_facts(args[0], args[1], var, count, max));
          if (count == 0 || (!max && args[1] == a.zero)) {
            n = args[0];
            summary = *child[0];
          }
        }
      } else {
        n = reapply(old, args);
        if (raw.op == "add" || raw.op == "add_raw") {
          std::vector<Summary> parts;
          std::vector<Id> unmatched;
          for (size_t j = 0; j < child.size(); ++j) {
            auto s = select(*child[j], false);
            if (s.empty())
              unmatched.push_back(a.mul({a.constant(raw.data[j]), args[j]}));
            else
              parts.push_back(scale(s, a.constant(raw.data[j])));
          }
          summary = residual(combine(parts), a.add(unmatched));
        } else if (raw.op == "max") {
          std::vector<Summary> parts;
          std::vector<Id> unmatched;
          for (size_t j = 0; j < child.size(); ++j) {
            auto s = select(*child[j], true);
            if (s.empty())
              unmatched.push_back(args[j]);
            else
              parts.push_back(std::move(s));
          }
          summary = residual(combine(parts), a.maximum(unmatched));
        } else if (raw.op == "mul") {
          size_t participating = 0, index = 0;
          Summary selected;
          for (size_t j = 0; j < child.size(); ++j) {
            auto s = select(*child[j], false);
            if (!s.empty()) {
              ++participating;
              index = j;
              selected = std::move(s);
            }
          }
          if (participating == 1) {
            std::vector<Id> xs{a.constant(raw.data[0])};
            for (size_t j = 0; j < args.size(); ++j)
              if (j != index)
                xs.push_back(args[j]);
            summary = scale(selected, a.mul(xs));
          }
        }
      }
      // Keep the expression DAG compact; normalize only scalar obligations.
      if (!reference) {
        auto c = close(summary);
        if (c) {
          n = *c;
          summary.clear();
        }
        if (!exponential_mode) n = a.factor_sum(n);
        auto it = active_byroot.find(a.nodes[n].op);
        std::vector<Id> candidates;
        if(it!=active_byroot.end())candidates=it->second;
        if(exponential_mode && (a.nodes[n].op=="add" || a.nodes[n].op=="mul" || a.nodes[n].op=="recip" || a.nodes[n].op=="exp" || a.nodes[n].op=="exp2"))
          for(Id sid:active)if(candidates.size()<64 && std::find(candidates.begin(),candidates.end(),sid)==candidates.end())candidates.push_back(sid);
        for (Id sid : candidates) {
            ++a.counter(Arena::Hot::local_pattern_candidates);
            for (const auto &m : match_term(sid, n)) {
              Key k{sid, m.params, m.coef, neutral(sid)};
              Id p = leaf(m.index);
              auto oldpiece = summary.find(k);
              if (oldpiece == summary.end())
                summary.emplace(std::move(k), p);
              else
                oldpiece->second = concat(oldpiece->second, p);
            }
          }
      }
      cache.emplace(old, Entry{n, std::move(summary)});
      ++a.counter(Arena::Hot::normalization_node_visits);
    }
    if (reference) {
      active_byroot.clear();
      for (Id sid : active)
        active_byroot[patterns.nodes[specs[sid].pattern].op].push_back(sid);
    }
    return cache.at(root).n;
  }
};
} // namespace rg
