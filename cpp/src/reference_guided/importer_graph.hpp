#pragma once
#include "importer.hpp"

namespace rg {

inline auto Importer::import(Id root, const Env &env) -> Id {
  // The typed-word predicate rule applies to predicates built while this side imports.
  struct WordPredicateScope {
    Arena &arena;
    std::function<std::optional<Id>(const std::string &, Id, Id)> saved;
    ~WordPredicateScope() { arena.word_predicate = std::move(saved); }
  } word_scope{a, a.word_predicate};
  a.word_predicate = [this](const std::string &op, Id x, Id y) { return typed_word_predicate(op, x, y); };
  struct Frame {
    Id i;
    Env env;
    bool ready;
  };
  std::vector<Frame> todo{{root, env, false}};
  // Reused key buffers: k holds this frame's key for the whole iteration,
  // probe the transient child keys. The cold log2, scale-guard and proved
  // Select paths below build their own keys.
  std::vector<Id> k, probe;
  while (!todo.empty()) {
    a.check();
    Frame f = std::move(todo.back());
    todo.pop_back();
    if (cache.contains(key(k, f.i, f.env)))
      continue;
    const auto &n = ns[f.i];
    const auto &edges = n.operands;
    bool fold = n.kind == "Fold", empty = fold && Q(n.count) == 0;
    size_t len = empty ? 1 : edges.size();
    if (fold && !empty && unrolls_stacked_fold(f.i)) {
      std::vector<std::pair<Id, Env>> missing;
      if (auto result = unrolled_fold(f.i, f.env, missing, "stacked_folds_unrolled")) {
        cache.emplace(k, context ? context->apply(a, *result) : *result);
        continue;
      }
      todo.push_back({f.i, f.env, false});
      for (auto &[raw, scope] : missing) todo.push_back({raw, std::move(scope), false});
      continue;
    }
    Env body = f.env;
    if (fold && !empty)
      body[n.level] = occurrence_binder(f.i, f.env);
    // Import the complete recognized numerical operation's input first.
    // Its bit decomposition is not a numeric cast; only a finite positive
    // input under ideal_real permits this complete compiler approximation.
    if(ideal)if(Id input=fp_idioms.log2_input(f.i);input!=invalid) {
      const auto input_key=key(input,f.env);
      if(!cache.contains(input_key)) {
        todo.push_back({f.i,f.env,false});todo.push_back({input,f.env,false});continue;
      }
      Id x=cache.at(input_key);
      if(a.nodes[x].facts.finite && a.nodes[x].facts.pos) {
        if(hardware_infinity(input,f.env))hardware_decline(f.i,{},f.env,"a floating-point idiom");
        auto result=a.logarithm(x,true);
        cache.emplace(k,context?context->apply(a,result):result);
        ++a.stats["ideal_compiler_log2_polynomials"];continue;
      }
    }
    // The same contract for the complete log1pf expansion: ideal log(1 + u) on a finite u >= 0.
    // There log1p is finite and the expansion returns its main path (-0 for u = -0).
    if(ideal)if(Id input=fp_idioms.log1p_input(f.i);input!=invalid) {
      const auto input_key=key(input,f.env);
      if(!cache.contains(input_key)) {
        todo.push_back({f.i,f.env,false});todo.push_back({input,f.env,false});continue;
      }
      Id x=cache.at(input_key);
      if(a.nodes[x].facts.finite && a.nodes[x].facts.nn) {
        if(hardware_infinity(input,f.env))hardware_decline(f.i,{},f.env,"a floating-point idiom");
        auto result=a.logarithm(a.add({a.one,x}));
        cache.emplace(k,context?context->apply(a,result):result);
        ++a.stats["ideal_compiler_log1p_polynomials"];continue;
      }
    }
    if (auto read = finite_scale_guard(n)) {
      const auto &input=ns.at(read->first);
      const auto &address=input.operands[0];
      const auto &chosen=edges[2]; // finite format excludes its exact NaN encoding
      auto address_key=key(address.node,f.env), chosen_key=key(chosen.node,f.env);
      if (!cache.contains(address_key)) {
        todo.push_back({f.i,f.env,false});
        todo.push_back({address.node,f.env,false});
        continue;
      }
      Id offset=edge(cache.at(address_key),address);
      // Reuse checked layout, alignment and range validation for this byte.
      typed_read(input.name,offset,"u8",true);
      require_contract_coverage(*contract->find(mapped_name(m,"allocations",input.name),read->second),input.name,offset,1);
      contract->applied.emplace(side,mapped_name(m,"allocations",input.name),read->second);
      if (!cache.contains(chosen_key)) {
        todo.push_back({f.i,f.env,false});
        todo.push_back({chosen.node,f.env,false});
        continue;
      }
      cache.emplace(k,edge(cache.at(chosen_key),chosen));
      if(hardware_infinity(chosen,f.env))hardware_infinity_keys.insert(k);
      ++a.stats["finite_"+read->second+"_nan_branches_eliminated"];
      continue;
    }
    // A proved condition must be evaluated before importing either arm.
    // In particular an unreachable NaN/unsupported arm is not a blocker.
    // Unknown conditions still require both arms; no speculative assumption.
    if(n.kind=="Operation" && n.name=="Select" && n.attributes.empty() && edges.size()==3) {
      const auto condition_key=key(edges[0].node,f.env);
      if(!cache.contains(condition_key)) {
        todo.push_back({f.i,f.env,false});todo.push_back({edges[0].node,f.env,false});continue;
      }
      Id condition=edge(cache.at(condition_key),edges[0]);
      if(context)condition=context->apply(a,condition);
      if(condition==a.zero || condition==a.one) {
        const auto& chosen=edges[condition==a.one?1:2];const auto selected_key=key(chosen.node,f.env);
        if(!cache.contains(selected_key)) {
          todo.push_back({f.i,f.env,false});todo.push_back({chosen.node,f.env,false});continue;
        }
        Id selected=edge(cache.at(selected_key),chosen);
        if(hardware_infinity(chosen,f.env))hardware_infinity_keys.insert(k);
        cache.emplace(k,context?context->apply(a,selected):selected);
        ++a.stats["proved_select_arms_imported"];continue;
      }
    }
    if (!f.ready) {
      todo.push_back({f.i, f.env, true});
      for (size_t j = len; j-- > 0;) {
        const Env &ctx = fold && j == 1 ? body : f.env;
        Id r = edges[j].node;
        if (!cache.contains(key(probe, r, ctx)))
          todo.push_back({r, ctx, false});
      }
      continue;
    }
    // The ready frame chooses the Fold's binder again. A leaf Fold of the same level and count
    // imported in between may have recorded a shared binder (occurrence_binder), so the choice
    // can differ from the initial frame's; the body is then imported under this one. Every
    // admitted binder is exact (alpha-equivalence), and the next choice is the same.
    if (fold && !empty && !cache.contains(key(probe, edges[1].node, body))) {
      todo.push_back({f.i, f.env, true});
      todo.push_back({edges[1].node, body, false});
      continue;
    }
    if (fold && !empty && n.reducer == "Min" && first_argmax_fold_pattern(f.i)) {
      std::vector<std::pair<Id, Env>> missing;
      if (auto result = first_argmax_fold(f.i, f.env, missing)) {
        if (const auto tainted = tainted_operands(f.i, f.env, body); !tainted.empty())
          carry_hardware_infinity(f.i, k, {cache.at(key(probe, edges[0].node, f.env)),
                                           cache.at(key(probe, edges[1].node, body))}, tainted, *result);
        cache.emplace(k, context ? context->apply(a, *result) : *result);
        continue;
      }
      if (!missing.empty()) {
        todo.push_back({f.i, f.env, true});
        for (auto &[raw, scope] : missing) todo.push_back({raw, std::move(scope), false});
        continue;
      }
    }
    if (fold && !empty && unrolls_fixed_table_fold(f.i) &&
        reads_fixed_table_at(cache.at(key(probe, edges[1].node, body)), body.at(n.level))) {
      std::vector<std::pair<Id, Env>> missing;
      if (auto result = unrolled_fold(f.i, f.env, missing)) {
        cache.emplace(k, context->apply(a, *result));
        continue;
      }
      todo.push_back({f.i, f.env, true});
      for (auto &[raw, scope] : missing) todo.push_back({raw, std::move(scope), false});
      continue;
    }
    std::vector<Id> xs;
    for (size_t j = 0; j < len; ++j) {
      const Env &ctx = fold && j == 1 ? body : f.env;
      Id operand=cache.at(key(probe,edges[j].node,ctx));
      xs.push_back(fp_idioms.is_power(f.i) && j==0 ? operand : edge(operand,edges[j]));
    }
    const auto tainted = tainted_operands(f.i, f.env, body);
    Id result;
    if((fp_idioms.is_rounding(f.i) || fp_idioms.is_power(f.i)) && !tainted.empty())
      hardware_decline(f.i, xs, f.env, "a floating-point idiom");
    if(fp_idioms.is_rounding(f.i)) {
      result=fp_idioms.lower_rounding(a,xs);
      ++a.stats["fp_idiom_rounding_instances"];
    } else if(fp_idioms.is_power(f.i)) {
      Id exponent=a.sub(xs[0],a.constant(12583039));
      Id biased=a.add({exponent,a.constant(127)});
      // The proved biased exponent is an integer in [1,253]; mantissa/sign
      // are zero after the 32-bit shift. Other integer consumers keep exact bits.
      result=a.mul({a.constant(Q(pow2(23))),biased});
      Id power = a.exp2(exponent);
      power_bit_values.emplace(result,power);
      ++a.stats["fp_idiom_power_instances"];
    } else if (auto exact = hardware_value(f.i, xs, tainted, f.env)) {
      result = *exact;
    } else {
      const auto ideal_xs = empty ? std::nullopt : ideal_rounded_operands(n, xs);
      result = empty ? xs[0] : make(n, ideal_xs ? *ideal_xs : xs, f.env, body);
      if (!tainted.empty()) carry_hardware_infinity(f.i, k, xs, tainted, result);
    }
    // Restrict local decoder proofs to arithmetic/selection boundaries. Raw
    // transports keep their original provenance for subsequent typed casts.
    if(n.kind=="Operation" && (n.name=="Add" || n.name=="Subtract" || n.name=="Multiply" ||
        n.name=="PtxMultiplyAdd" || n.name=="Fma" || n.name=="Select")) {
      if(n.name=="Select" && fp4_reference_boundaries.contains(f.i))
        result=fp4_boundaries.normalize_fp4_boundary(result,
            [&](Id source,unsigned lo,unsigned count) {
              return canonical_packed_slice(source,lo,count);
            });
      result=normalize_packed_decoder(result);
    }
    result=gathered_at_argmax(n,xs,result);
    cache.emplace(k, context ? context->apply(a,result) : result);
    ++a.counter(Arena::Hot::imported_context_nodes);
  }
  return cache.at(key(probe, root, env));
}

// Recognize one complete binary32 compiler LSE tail under ideal_real.
// This node means ln(mass) + ln(2)*pivot_in_log2_units. Only the
// witnessed tail's rounded ln(2) is idealized; neither the mass's exp2
// coefficients nor unrelated ln(2)/log2(e) literals are changed.
// The raw scan has fixed depth and arity (at most four orientations).
inline auto Importer::compiler_lse(const RawNode &root, const Env &env) -> std::optional<Id> {
  if (!ideal) return {};
  auto operation = [](const RawNode &node, const char *name, size_t arity) {
    return node.kind == "Operation" && node.name == name && node.type == "f32" &&
           (node.attributes.empty() || node.attributes == "ideal_arithmetic") &&
           node.operands.size() == arity &&
           std::all_of(node.operands.begin(), node.operands.end(),
                       [](const Edge &edge) { return edge.conversion.empty(); });
  };
  auto literal = [&](Id raw, const Q &value) {
    const auto &node = ns.at(raw);
    if (node.kind != "Constant" || node.type != "f32" || !node.operands.empty()) return false;
    // Constants have no free indices. Reuse their checked binary32 value;
    // probing this optional idiom must not reparse infinity/hex spellings.
    const Id *found = cache.find(std::span<const Id>(&raw, 1));
    if (!found) return false;
    auto decoded = a.value(*found);
    return decoded && *decoded == value;
  };
  auto imported = [&](Id raw) -> std::optional<Id> {
    if (const Id *found = cache.find(key(raw, env))) return *found;
    return {};
  };
  if (!operation(root, "Multiply", 2)) return {};
  static const Q compiler_ln2(1453635, 2097152);
  static const Q compiler_score_scale(12102203, 67108864);
  for (unsigned side = 0; side != 2; ++side) {
    const auto &sum = ns.at(root.operands[1-side].node);
    // The exporter retains fused arithmetic and its operand-edge types.
    // Recognize the same complete compiler tail without rewriting that
    // provenance into a guessed Multiply/Add sequence.
    if (literal(root.operands[side].node, compiler_ln2) &&
        sum.kind == "Operation" && sum.name == "Fma" && sum.type == "f32" &&
        sum.attributes == "rounding=nearest-even " && sum.operands.size() == 3 &&
        sum.operands[2].conversion.empty()) {
      const auto &log = ns.at(sum.operands[2].node);
      if (log.kind == "Operation" && log.name == "PtxLog2" && log.type == "f32" &&
          log.attributes == "source=f32.approx.ftz.f32 source_type=f32" &&
          log.operands.size() == 1 && log.operands[0].conversion.empty() &&
          ns.at(log.operands[0].node).type == "f32") {
        for (unsigned j = 0; j < 2; ++j) {
          const Edge &coefficient = sum.operands[j], &value = sum.operands[1-j];
          const auto &raw = ns.at(coefficient.node);
          bool typed_constant = raw.kind == "Constant" &&
              ((raw.type == "f32" && coefficient.conversion.empty()) ||
               (raw.type == "b32" && coefficient.conversion == "Bitcast(b32 -> f32)"));
          if (!typed_constant || !value.conversion.empty() || ns.at(value.node).type != "f32") continue;
          auto c = imported(coefficient.node), v = imported(value.node), mass = imported(log.operands[0].node);
          if (!c || !v || !mass || a.value(edge(*c, coefficient)) != compiler_score_scale ||
              !a.nodes[*v].facts.finite || !a.nodes[*mass].facts.finite || !a.nodes[*mass].facts.pos) continue;
          Id pivot = a.mul({a.constant(compiler_score_scale), *v});
          ++a.stats["ideal_compiler_lse_sequences"];
          ++a.stats["ideal_compiler_fma_lse_sequences"];
          return a.intern("ideal_lse", {*mass, pivot}, {}, "ptx_lg2_f32", {true, false, false});
        }
      }
    }
    if (!operation(sum, "Add", 2) ||
        !literal(root.operands[side].node, compiler_ln2)) continue;
    for (unsigned log_side = 0; log_side != 2; ++log_side) {
      const auto &log = ns.at(sum.operands[log_side].node);
      Id scaled_raw = sum.operands[1-log_side].node;
      const auto &scaled = ns.at(scaled_raw);
      if (log.kind != "Operation" || log.name != "PtxLog2" || log.type != "f32" ||
          log.attributes != "source=f32.approx.ftz.f32 source_type=f32" ||
          log.operands.size() != 1 || !log.operands[0].conversion.empty() ||
          ns.at(log.operands[0].node).type != "f32" ||
          !operation(scaled, "Multiply", 2)) continue;
      bool scale_witness = false;
      for (unsigned scalar_side = 0; scalar_side != 2; ++scalar_side)
        scale_witness |= literal(scaled.operands[scalar_side].node, compiler_score_scale) &&
                         ns.at(scaled.operands[1-scalar_side].node).type == "f32";
      if (!scale_witness) continue;
      auto mass = imported(log.operands[0].node), pivot = imported(scaled_raw);
      if (!mass || !pivot || !a.nodes[*mass].facts.finite || !a.nodes[*mass].facts.pos ||
          !a.nodes[*pivot].facts.finite) continue;
      ++a.stats["ideal_compiler_lse_sequences"];
      return a.intern("ideal_lse", {*mass, *pivot}, {}, "ptx_lg2_f32", {true, false, false});
    }
  }
  return {};
}

// The value of a literal operand after its edge conversion, or nullopt.
inline auto Importer::literal_operand(const Edge &e) -> std::optional<Q> {
  const auto &raw = ns.at(e.node);
  if (raw.kind != "Constant") return {};
  try { return a.value(edge(literal(raw.name, raw.type), e)); }
  catch (const Unsupported &) { return {}; }
}

// Rule (ideal_real): a literal whose value is the round-to-nearest value of ln 2 or log2(e) in the
// format of the floating Add, Subtract, Multiply, Fma, Divide or Reciprocal it is an operand of denotes that exact
// constant: ln 2 (Arena::natural_log2) or 1/ln 2; a negated literal denotes the negated constant. This
// idealizes the constant's own rounding error (about 3e-9 relative in binary32), as lg2.approx and
// ex2.approx are ideal. Any other use (comparison, selection, bit operation, hardware-exact value) keeps
// the literal's binary value. Returns the operands with the replacements, or nullopt if none. O(operands).
inline auto Importer::ideal_rounded_operands(const RawNode &n, const std::vector<Id> &xs) -> std::optional<std::vector<Id>> {
  if (!ideal || !ideal_arithmetic_operation(n) || xs.size() != n.operands.size()) return {};
  std::optional<std::vector<Id>> result;
  for (size_t j = 0; j < xs.size(); ++j) {
    if (ns.at(n.operands[j].node).kind != "Constant") continue;
    const auto value = a.value(xs[j]);
    if (!value) continue;
    for (const auto &c : rounded_constants(n.type)) {
      if (*value != c.high && *value != -c.high) continue;
      const Id exact = c.ln2 ? a.natural_log2() : a.inverse_natural_log2();
      if (!result) result = xs;
      (*result)[j] = *value == c.high ? exact : a.neg(exact);
      ++a.stats["ideal_rounded_constants"];
      break;
    }
  }
  return result;
}

// A high/low split constant: Fma(x, RN(c - RN(c)), Fma(x, RN(c), z)) is x*c + z, as nvcc's expf
// computes x*log2(e). With the high part read as c (ideal_rounded_operands), the low part adds no
// further term, so the outer Fma is the inner one's value. The multiplicand x must be the same edge
// in both, and the low part must carry the high part's sign. O(1).
inline auto Importer::split_constant_low_part(const RawNode &n) -> bool {
  if (!ideal || !ideal_arithmetic_operation(n) || n.name != "Fma" || n.operands.size() != 3 ||
      !n.operands[2].conversion.empty()) return false;
  const auto &inner = ns.at(n.operands[2].node);
  if (!ideal_arithmetic_operation(inner) || inner.name != "Fma" || inner.type != n.type ||
      inner.operands.size() != 3) return false;
  const auto same = [](const Edge &x, const Edge &y) { return x.node == y.node && x.conversion == y.conversion; };
  for (unsigned i = 0; i != 2; ++i) for (unsigned j = 0; j != 2; ++j) {
    if (!same(n.operands[1 - i], inner.operands[1 - j])) continue;
    const auto low = literal_operand(n.operands[i]), high = literal_operand(inner.operands[j]);
    if (!low || !high) continue;
    for (const auto &c : rounded_constants(n.type))
      if ((*high == c.high && *low == c.low) || (*high == -c.high && *low == -c.low)) {
        ++a.stats["ideal_split_constant_pairs"];
        return true;
      }
  }
  return false;
}

} // namespace rg
