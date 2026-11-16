#pragma once
#include "checker.hpp"

namespace rg {

inline auto Checker::canonical_coefficient(Id value) -> Id {
  if (!exponential_mode || !a.nodes[value].facts.finite) return value;
  if (auto it = coefficient_normal_cache.find(value); it != coefficient_normal_cache.end())
    return it->second;
  scalar.reset_query();
  Id result = scalar.canonicalize(value);
  coefficient_normal_cache.emplace(value, result);
  ++a.stats["coefficient_normalizations"];
  return result;
}

// select(p, A, B) denotes A for every p when both arms are total and the
// scalar engine proves A = B (normal-form or fraction cross-product). Only
// arms built from that engine's algebra (sums, products, reciprocals,
// exponentials) over the same free coordinates can be such a pair; any
// other pair is kept without a query. One memoized query per arm pair; the
// kept arm is the older node, independent of the predicate's polarity.
inline auto Checker::equal_arm_select(Id id) -> Id {
  const auto &n = a.nodes[id];
  if (!exponential_mode || n.op != "select" || n.args.size() != 3) return id;
  const Id x = n.args[1], y = n.args[2];
  auto algebraic = [&](Id arm) {
    const auto &op = a.nodes[arm].op;
    return a.nodes[arm].facts.finite &&
           (op == "add" || op == "add_raw" || op == "mul" || op == "recip" ||
            op == "exp" || op == "exp2");
  };
  if (!algebraic(x) || !algebraic(y) || a.nodes[x].free != a.nodes[y].free) return id;
  auto [it, fresh] = equal_arm_cache.emplace(std::minmax(x, y), false);
  if (fresh) {
    it->second = scalar.equal(x, y);
    ++a.stats[it->second ? "select_equal_arm_proofs" : "select_equal_arm_declines"];
  }
  return it->second ? std::min(x, y) : id;
}

inline auto Checker::equal_coefficients(Id left, Id right) -> bool {
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

// Every slot a coordinate pattern refers to, including the residual slots
// that affine_param and coordinate_sum keep in their data. O(pattern nodes).
inline auto Checker::coordinate_slots(Id root, std::vector<size_t> &out) const -> void {
  std::vector<Id> todo{root};
  std::unordered_set<Id> seen{root};
  while (!todo.empty()) {
    const auto &n = patterns.nodes[todo.back()];
    todo.pop_back();
    if (n.op == "slot") out.push_back(n.data[0].numerator().convert_to<size_t>());
    else if ((n.op == "affine_param" || n.op == "coordinate_sum") && n.data.size() > 1 && n.data[1] >= 0)
      out.push_back(n.data[1].numerator().convert_to<size_t>());
    for (Id child : n.args) if (seen.insert(child).second) todo.push_back(child);
  }
}

inline auto Checker::pattern(Id root, Id var, bool allow_parameterized, bool allow_derived) -> SpecPattern {
  std::unordered_map<Id, Id> slots;
  std::vector<Id> slot_values;
  std::vector<bool> valued, coordinate;
  auto slot = [&](Id value, bool coordinate_position) {
    auto [it, fresh] = slots.emplace(value, Id(slots.size()));
    if (fresh) {
      slot_values.push_back(value);
      valued.push_back(false);
      coordinate.push_back(false);
    }
    if (coordinate_position) coordinate[it->second] = true;
    return it->second;
  };
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
    if (w.n == var || (w.coord && !a.depends(w.n, var)) || enclosing_binder(w.n, var)) {
      if (w.n == var) anchored = true;
      const Id index = slot(w.n, w.coord || w.n == var);
      if (!w.coord && w.n != var) valued[index] = true;
      memo[k] = patterns.intern("slot", {}, {Q(index)});
      continue;
    }
    // Record quotient/remainder relationships before the binder subtree
    // becomes a slot. The surrounding read/call structure remains exact.
    if (allow_derived && w.coord) {
      if (auto relation = CoordinateRelations::describe(a, w.n, var)) {
        (*relation)[0] = Q(slot(var, true));
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
        Id axis = slot(var, true);
        Q residual_slot(-1);
        if (recipe->residual != invalid) residual_slot = Q(slot(recipe->residual, true));
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
        Id axis = slot(var, true);
        if (parameterized) {
          used_parameterized = true;
          std::vector<Id> residual_terms;
          for (auto [term, scale] : terms)
            if (term != var)
              residual_terms.push_back(a.mul({a.constant(scale), term}));
          Id residual = a.add(std::move(residual_terms));
          memo[k] = patterns.intern("affine_param", {},
                                   {Q(axis), Q(slot(residual, true)), coefficient});
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
    // A nested Call's Facts depend on the arguments its domains name.
    if (n.op == "call" && n.data.size() == 1 && n.data[0].denominator() == 1 &&
        n.data[0] >= 0 && n.data[0] < Q(specs.size())) {
      std::vector<size_t> inside;
      for (const auto &domain : specs[n.data[0].numerator().convert_to<Id>()].domains)
        if (domain.argument < as.size()) coordinate_slots(as[domain.argument], inside);
      for (size_t index : inside)
        if (slot_values[index] != var) valued[index] = true;
    }
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
          std::move(slot_values), std::move(valued), std::move(coordinate)};
}

inline auto Checker::literal_binder_free(Id sid) -> bool {
  if (literal_free_binders != fold_binders.size()) {
    literal_free_specs.clear();
    literal_free_binders = fold_binders.size();
  }
  if (auto it = literal_free_specs.find(sid); it != literal_free_specs.end()) return it->second;
  bool free = true;
  std::vector<Id> todo{specs[sid].pattern};
  std::unordered_set<Id> seen{specs[sid].pattern};
  while (!todo.empty() && free) {
    a.check(); const Id id = todo.back(); todo.pop_back();
    const auto &n = patterns.nodes[id];
    if (n.op == "var" && n.args.empty()) {
      const auto actual = a.find_node(n.op, n.args, n.data, n.tag);
      free = actual && !fold_binders.contains(*actual);
    } else if (n.op == "call") {
      free = n.data.size() == 1 && n.data[0].denominator() == 1 && n.data[0] >= 0 && n.data[0] < Q(sid) &&
             literal_binder_free(n.data[0].numerator().convert_to<Id>());
    }
    for (Id child : n.args) if (seen.insert(child).second) todo.push_back(child);
  }
  literal_free_specs.emplace(sid, free);
  return free;
}

inline auto Checker::intern_spec(Id p, size_t slots, size_t axis, Z count, Facts f,
               bool maximum, std::vector<SlotDomain> domains) -> Id {
  auto k = std::tuple(p, slots, axis, count, maximum, domains);
  auto it = spec_index.find(k);
  if (it != spec_index.end())
    return it->second;
  Id id = Id(specs.size());
  spec_index.emplace(std::move(k), id);
  specs.push_back({p, slots, axis, std::move(count), f, maximum, std::move(domains)});
  return id;
}

// The integer interval of a valued slot's value, narrowed by its sign
// Facts; nothing when the value is not provably an integer with bounds.
inline auto Checker::integer_domain(Id value) -> std::optional<std::pair<Q, Q>> {
  const auto &n = a.nodes[value];
  if (!n.bounds || n.bounds->first.denominator() != 1 || n.bounds->second.denominator() != 1 ||
      !CoordinateRelations::integer(a, value)) return {};
  Q lo = n.bounds->first;
  if (n.facts.pos) lo = std::max(lo, Q(1));
  else if (n.facts.nn) lo = std::max(lo, Q(0));
  return std::pair(lo, n.bounds->second);
}

// Domains of the valued slots, by Call argument position; nothing when a
// valued slot's value has no integer domain.
inline auto Checker::slot_domains(const std::vector<Id> &values,
                                                    const std::vector<bool> &valued, size_t axis) -> std::optional<std::vector<SlotDomain>> {
  std::vector<SlotDomain> domains;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i == axis || !valued[i]) continue;
    auto domain = integer_domain(values[i]);
    if (!domain) return {};
    domains.push_back({i < axis ? i : i - 1, domain->first, domain->second});
  }
  return domains;
}

// A Call denotes the spec's body at its arguments for any argument values.
// Its Facts are the spec's only at arguments inside every domain; elsewhere
// nothing is known. O(domains) per Call.
inline auto Checker::call(Id sid, const std::vector<Id> &args) -> Id {
  const Spec &s = specs[sid];
  Facts facts;
  if (std::all_of(s.domains.begin(), s.domains.end(), [&](const SlotDomain &domain) {
        return domain.argument < args.size() && in_domain(args[domain.argument], domain);
      })) {
    facts = {s.facts.finite, s.count > 0 && s.facts.pos, s.count == 0 || s.facts.nn};
    if (s.maximum && s.count > 0)
      facts.finite_or_neginf = s.facts.finite_or_negative_infinity();
  } else {
    ++a.stats["reference_call_domain_misses"];
  }
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
    // Rule (one spelling per instantiated shape): the definition above with every actual argument
    // replaced by the index of its first occurrence among the distinct actual arguments is the
    // reduction as a function of those arguments (constants, reads, products and sums of the
    // pattern; the same count and kind). Two specs with one shape denote the same function: a spec
    // whose equal literal parameters share one slot (row and head both 0, say) and the general
    // spec. The first spec registered for a shape is its spelling; a Call of another spec with that
    // shape is respelled as a Call of the first with the corresponding arguments (constants where
    // the shape has constants, the k-th distinct actual argument where it has marker k). Calls in
    // Fold bodies with renamed binders then share one spec, so alpha equality can identify them.
    // Exact: the respelled Call instantiates to the same definition. O(pattern) for the shape.
    std::vector<Id> distinct;
    std::vector<Id> shape_captures(s.slots);
    std::vector<std::pair<bool,Id>> recipe;  // per non-axis slot: (constant?, constant node or marker index)
    {
      size_t at=0;
      for(size_t i=0;i<s.slots;++i) {
        if(i==s.axis) { shape_captures[i]=captures[i]; continue; }
        const Id arg=args[at++];
        if(const auto v=a.value(arg)) { shape_captures[i]=captures[i]; recipe.emplace_back(true,arg); continue; }
        auto found=std::find(distinct.begin(),distinct.end(),arg);
        const size_t k=size_t(found-distinct.begin());
        if(found==distinct.end()) distinct.push_back(arg);
        shape_captures[i]=patterns.intern("definition_actual_index",{},{Q(k)});
        recipe.emplace_back(false,Id(k));
      }
    }
    std::unordered_map<Id,Id> shape_memo;
    std::vector<std::pair<Id,bool>> shape_todo{{s.pattern,false}};
    bool shaped=true; size_t shape_work=0;
    while(!shape_todo.empty() && shaped) {
      a.check(); auto [id,ready]=shape_todo.back();shape_todo.pop_back();
      if(shape_memo.contains(id))continue;
      if(++shape_work>512 || shape_memo.size()>=128){shaped=false;break;}
      const auto &n=patterns.nodes[id];
      if(n.op=="slot") {shape_memo[id]=shape_captures.at(n.data[0].numerator().convert_to<size_t>());continue;}
      if(!ready){shape_todo.emplace_back(id,true);for(Id child:n.args)if(!shape_memo.contains(child))shape_todo.emplace_back(child,false);continue;}
      std::vector<Id> children;for(Id child:n.args)children.push_back(shape_memo.at(child));
      auto data=n.data;
      if(n.op=="mul")std::sort(children.begin(),children.end());
      if(n.op=="add") {
        std::vector<std::pair<Id,Q>> terms;for(size_t i=0;i<children.size();++i)terms.emplace_back(children[i],data[i]);
        std::sort(terms.begin(),terms.end());
        for(size_t i=0;i<terms.size();++i){children[i]=terms[i].first;data[i]=terms[i].second;}
      }
      shape_memo[id]=patterns.intern(n.op,std::move(children),std::move(data),n.tag);
    }
    if(shaped) {
      const auto key=std::tuple(shape_memo.at(s.pattern),s.count,s.maximum);
      auto [spelling,fresh]=call_spellings.emplace(key,std::pair(sid,recipe));
      const auto &[canonical,canonical_recipe]=spelling->second;
      // Only a spec with fewer slots (equal literal parameters sharing one slot) is respelled, as the
      // registered spec of the shape with more slots; a spelling with as many slots or more is kept.
      if(!fresh && canonical!=sid && specs[canonical].slots>s.slots) {
        std::vector<Id> respelled;
        bool ok=true;
        for(const auto &[constant,value]:canonical_recipe) {
          if(constant) respelled.push_back(value);
          else if(value<distinct.size()) respelled.push_back(distinct[value]);
          else { ok=false; break; }
        }
        if(ok && respelled.size()+1==specs[canonical].slots) {
          ++a.stats["reduction_call_respellings"];
          result=call(canonical,respelled);
          call_definition_cache.emplace(raw,result);
          return result;
        }
      }
    }
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
inline auto Checker::register_paired_maximum(Id call_id, Id reduced_axis) -> void {
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
    auto shape = pattern(call_id, axis);
    auto domains = slot_domains(shape.values, shape.valued, shape.axis);
    Id auxiliary = intern_spec(shape.pattern, shape.slots, shape.axis, variable.bounds->second.numerator() + 1,
                               domains ? call_node.facts : Facts{}, true, domains ? std::move(*domains) : std::vector<SlotDomain>{});
    if (active.insert(auxiliary).second) ++a.stats["paired_maximum_reference_domains"];
  }
}

inline auto Checker::reg(Id body, Id var, Z count, bool maximum) -> Id {
  // Scalar normalization belongs to the body equality query.
  auto shape = pattern(body, var);
  auto &[p, ns, axis, slots, valued, coordinate] = shape;
  if (!maximum && a.nodes[body].facts.finite) {
    if (auto canonical = state_patterns.canonicalize(p, ns, axis)) {
      std::vector<Id> reordered(ns);
      std::vector<bool> reordered_valued(ns), reordered_coordinate(ns);
      for (size_t i = 0; i < ns; ++i) {
        const size_t to = canonical->old_to_new[i];
        reordered[to] = slots[i];
        reordered_valued[to] = valued[i];
        reordered_coordinate[to] = coordinate[i];
      }
      p = canonical->pattern;
      axis = canonical->axis;
      slots = std::move(reordered);
      valued = std::move(reordered_valued);
      coordinate = std::move(reordered_coordinate);
    }
  }
  // The body's Facts were derived with each valued slot at its value. They
  // are the spec's only together with those values' domains.
  auto domains = slot_domains(slots, valued, axis);
  Id sid = intern_spec(p, ns, axis, count, domains ? a.nodes[body].facts : Facts{}, maximum,
                       domains ? std::move(*domains) : std::vector<SlotDomain>{});
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
          auto window = pattern(c, index);
          auto window_domains = slot_domains(window.values, window.valued, window.axis);
          Id aux = intern_spec(window.pattern, window.slots, window.axis, v.data[1].numerator() + 1,
                               window_domains ? a.nodes[c].facts : Facts{}, true,
                               window_domains ? std::move(*window_domains) : std::vector<SlotDomain>{});
          active.insert(aux);
          ++a.stats["shifted_maximum_reference_domains"];
        }
      }
      continue;
    }
    // A binder that occurs only as a value (a mask bound, an addend) is no
    // coordinate of a Max domain.
    if (n.op != "var" || !n.bounds || n.bounds->first != 0 ||
        n.bounds->second < 1 || !coordinate[j < axis ? j : j + 1])
      continue;
    std::vector<Id> ps;
    for (size_t i = 0; i < args.size(); ++i)
      ps.push_back(patterns.intern("slot", {}, {Q(i)}));
    Id pat = patterns.intern("call", ps, {Q(sid)});
    // The axis j ranges over exactly its own domain [0, hi]; the other
    // domains move to the auxiliary's argument positions.
    std::vector<SlotDomain> aux_domains;
    for (const auto &domain : specs[sid].domains)
      if (domain.argument != j)
        aux_domains.push_back({domain.argument < j ? domain.argument : domain.argument - 1, domain.lo, domain.hi});
    Id aux =
        intern_spec(pat, args.size(), j, n.bounds->second.numerator() + 1,
                    a.nodes[c].facts, true, std::move(aux_domains));
    active.insert(aux);
  }
  register_paired_maximum(c, var);
  return c;
}

inline auto Checker::select(const Summary &s, bool max) -> Summary {
  Summary r;
  for (const auto &[k, p] : s)
    if (specs[k.sid].maximum == max)
      r.emplace(k, p);
  return r;
}

} // namespace rg
