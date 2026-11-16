#pragma once
#include "importer.hpp"

namespace rg {

inline auto Importer::fp4_reference_candidate(const RawNode &n) const -> bool {
  auto operation=[](const RawNode &x,const char *name,size_t arity) {
    return x.kind=="Operation" && x.name==name && x.operands.size()==arity &&
           std::all_of(x.operands.begin(),x.operands.end(),
                       [](const Edge &e){return e.conversion.empty();});
  };
  auto constant=[&](Id id,const char *value) {
    const auto &x=ns[id];
    return x.kind=="Constant" && x.type=="u8" && x.name==value;
  };
  // Actual reference boundary: sign ? -magnitude : magnitude. The sign
  // comes from payload & 8, and magnitude is itself a Select. Wrong scales,
  // mantissas or sign sources still reach (and fail) the exact local proof.
  if(n.type!="f32" || !operation(n,"Select",3))return false;
  const auto &condition=ns[n.operands[0].node],&negative=ns[n.operands[1].node];
  const auto &magnitude=ns[n.operands[2].node];
  if(!operation(condition,"NotEqual",2) || !operation(negative,"Negate",1) ||
     negative.operands[0].node!=n.operands[2].node || !operation(magnitude,"Select",3))return false;
  for(unsigned j=0;j<2;++j) {
    if(!constant(condition.operands[j].node,"0"))continue;
    const auto &mask=ns[condition.operands[1-j].node];
    if(operation(mask,"BitAnd",2) &&
       (constant(mask.operands[0].node,"8") || constant(mask.operands[1].node,"8")))return true;
  }
  return false;
}

// Fills k with the memo key [i, env[v] for each free level v of i]; the hot
// import() loop reuses its buffers instead of allocating a key per probe.
inline auto Importer::key(std::vector<Id> &k, Id i, const Env &env) const -> std::span<const Id> {
  if (i >= ns.size())
    throw Unsupported("raw node reference outside input DAG");
  k.assign(1, i);
  for (Id v : free[i]) {
    auto it = env.find(v);
    if (it == env.end())
      throw Unsupported("unbound index level " + std::to_string(v));
    k.push_back(it->second);
  }
  return k;
}

inline auto Importer::key(Id i, const Env &env) const -> std::vector<Id> {
  std::vector<Id> k;
  k.reserve(1 + (i < ns.size() ? free[i].size() : 0));
  key(k, i, env);
  return k;
}

// The ordinary raw-id/projected-free-environment memo key above is unchanged.
// A separate, bounded alpha-binder bank makes shared raw producers reusable
// across sibling Folds and scope-independent copies. Use exactly the free
// environment already computed for this Fold (including its seed): unused
// enclosing levels cannot affect its value, and its bound raw level is only
// a local name. The body is still imported with that original level bound.
// Nested-body and shadowed-level
// cases below keep their original binders to avoid capturing a cached Fold.
inline auto Importer::fold_binder(Id raw, const Env &env) -> Id {
  const auto &n = ns[raw];
  auto original = [&]() {
    std::string label = side + ":binder:" + std::to_string(raw);
    for (auto [v, x] : env)
      label += ":" + std::to_string(v) + "=" + std::to_string(x);
    return a.var(label, 0, Q(n.count) - Q(1));
  };
  // Closed raw Folds may already be memoized outside the present lexical
  // scope. Do not alpha-share an outer binder across any nested Fold, or a
  // shadowed level, because the cached inner binder cannot then be renamed.
  if (raw_has_fold[n.operands[1].node]) {
    ++a.stats["importer_scope_nested_body_fallbacks"];
    return original();
  }
  if (env.contains(n.level)) {
    ++a.stats["importer_scope_shadow_fallbacks"];
    return original();
  }
  if (env.size() > scope_environment_limit) {
    ++a.stats["importer_scope_environment_fallbacks"];
    return original();
  }
  ++a.stats["importer_scope_leaf_lookups"];
  std::vector<Id> scope;
  scope.reserve(2 + 2 * free[raw].size());
  scope.push_back(Id(n.count));
  scope.push_back(Id(n.count >> 32));
  for (Id level : free[raw]) {
    const auto binding = env.find(level);
    if (binding == env.end()) throw Unsupported("unbound index level " + std::to_string(level));
    Id value = binding->second;
    scope.push_back(level);
    scope.push_back(value);
  }
  auto found = sibling_binders.find(scope);
  if (found != sibling_binders.end()) {
    if (found->second == invalid) return original();
    // Check expressions, not only IDs: an outer coordinate may itself use
    // the candidate variable. Existing Arena nodes and their free sets are
    // immutable, so a rejected key stays rejected on the ready frame too.
    // The semantic key drops unused levels, but an admitted binder must
    // still be distinct from every active enclosing coordinate. Keep this
    // stronger hygiene check even when an outer level is unused here.
    for (auto [level, value] : env) {
      (void)level;
      if (a.depends(value, found->second)) {
        found->second = invalid;
        ++a.stats["importer_scope_capture_fallbacks"];
        return original();
      }
    }
    ++a.stats["importer_scope_binder_hits"];
    return found->second;
  }
  if (sibling_binders.size() >= scope_binder_limit) {
    ++a.stats["importer_scope_budget_fallbacks"];
    return original();
  }
  if (scope_binder_bank == 0)
    scope_binder_bank = ++a.stats["importer_scope_banks"];
  std::string label = side + ":shared-binder:" +
      std::to_string(scope_binder_bank) + ":" +
      std::to_string(++scope_binder_serial);
  const size_t before = a.nodes.size();
  Id binder = a.var(label, 0, Q(n.count) - Q(1));
  // Only a fresh Arena variable is admitted; hence it cannot occur free in
  // any existing environment expression. Memorize a collision decline so
  // the import scheduler's initial and ready frames use the same choice.
  if (binder < before) {
    sibling_binders.emplace(std::move(scope), invalid);
    ++a.stats["importer_scope_name_collision_fallbacks"];
    return original();
  }
  sibling_binders.emplace(std::move(scope), binder);
  ++a.stats["importer_scope_binders_created"];
  return binder;
}

inline auto Importer::occurrence_binder(Id raw, const Env &env) -> Id {
  const auto &n = ns[raw];
  if (auto it = occurrence_binders.find({n.level, n.count});
      it != occurrence_binders.end() && !env.contains(n.level) && !raw_has_fold[n.operands[1].node]) {
    const bool captured = std::any_of(env.begin(), env.end(),
        [&](const auto &binding) { return a.depends(binding.second, it->second); });
    if (!captured) {
      ++a.stats["importer_occurrence_binder_hits"];
      return it->second;
    }
    ++a.stats["importer_occurrence_capture_fallbacks"];
  }
  const Id binder = fold_binder(raw, env);
  if (a.nodes[binder].tag.starts_with(side + ":shared-binder:"))
    occurrence_binders.try_emplace(std::pair(n.level, n.count), binder);
  return binder;
}

inline auto Importer::unsigned_word(const std::string &t) -> bool {
  return t.size() > 1 && (t[0] == 'u' || t[0] == 'b') &&
         std::all_of(t.begin() + 1, t.end(), [](char c) {
           return c >= '0' && c <= '9';
         });
}

inline auto Importer::pack_constant(const Edge &e) -> unsigned {
  const auto &n = ns.at(e.node);
  if (!e.conversion.empty() || n.kind != "Constant")
    throw Unsupported("output packing requires constant selector/shift");
  Q q = decimal(n.name);
  if (q.denominator() != 1 || q < 0 || q > 65535)
    throw Unsupported("output packing constant out of range");
  return q.numerator().convert_to<unsigned>();
}

inline auto Importer::packed_edge(const Edge &e, unsigned depth) -> Packed {
  if (depth > 64)
    throw Unsupported("output packing depth limit");
  // Only this exact encoding is recognized; arbitrary saturating arithmetic
  // and conversions retain the ordinary importer rejection.
  static const std::regex quant(
      R"(\s*Convert rounding=nearest-even saturate_finite \(f32 -> e4m3\); Bitcast\(e4m3 -> u8\); Convert \(u8 -> u16\)\s*)");
  if (std::regex_match(e.conversion, quant)) {
    if (ns.at(e.node).type != "f32")
      throw Unsupported("output quantization source type mismatch");
    return {e.node, invalid};
  }
  Packed p = packed_node(e.node, depth + 1);
  static const std::regex cv(
      R"(\s*(Bitcast|Convert)\s*\(\s*([bu][0-9]+)\s*->\s*([bu][0-9]+)\s*\)\s*)");
  std::stringstream stream(e.conversion);
  std::string part;
  while (std::getline(stream, part, ';')) {
    std::smatch m;
    if (!std::regex_match(part, m, cv))
      throw Unsupported("nontransport output packing conversion");
    unsigned sw = width(m[2]), dw = width(m[3]);
    if (sw != p.size() * 8 || dw % 8 ||
        (m[1] == "Bitcast" && sw != dw))
      throw Unsupported("output packing conversion width mismatch");
    p.resize(dw / 8, invalid);
  }
  return p;
}

inline auto Importer::packed_node(Id id, unsigned depth) -> Packed {
  a.check();
  if (depth > 64)
    throw Unsupported("output packing depth limit");
  if (auto it = packed_cache.find(id); it != packed_cache.end())
    return it->second;
  const auto &n = ns.at(id);
  if (!unsigned_word(n.type) || width(n.type) % 8)
    throw Unsupported("nonword output packing node");
  unsigned bytes = width(n.type) / 8;
  Packed p(bytes, invalid);
  const auto &es = n.operands;
  if (n.kind == "Constant" && decimal(n.name) == 0) {
  } else if (n.kind != "Operation" || (!n.attributes.empty() &&
                 !(n.name == "PtxBytePermute" &&
                   n.attributes == "source=b32.b32 source_type=b32"))) {
    throw Unsupported("nontransport output packing node");
  } else if (n.name == "Add" && es.size() == 2) {
    p = packed_edge(es[0], depth + 1);
    auto q = packed_edge(es[1], depth + 1);
    if (p.size() != bytes || q.size() != bytes)
      throw Unsupported("output packing add width mismatch");
    for (unsigned i = 0; i < bytes; ++i) {
      if (p[i] != invalid && q[i] != invalid)
        throw Unsupported("overlapping output packing addition");
      if (q[i] != invalid) p[i] = q[i];
    }
  } else if ((n.name == "Multiply" || n.name == "ShiftRight") && es.size() == 2) {
    unsigned c = pack_constant(es[1]);
    unsigned shift = 0;
    if (n.name == "Multiply") {
      if (c != 256) throw Unsupported("unsupported output packing multiplier");
      shift = 1;
    } else {
      if (c % 8 || c >= bytes * 8)
        throw Unsupported("nonbyte output packing shift");
      shift = c / 8;
    }
    auto q = packed_edge(es[0], depth + 1);
    if (q.size() != bytes) throw Unsupported("output packing shift width mismatch");
    for (unsigned i = 0; i < bytes; ++i) {
      if (n.name == "Multiply") {
        if (i >= shift) p[i] = q[i - shift];
      } else if (i + shift < bytes) p[i] = q[i + shift];
    }
  } else if (n.name == "PtxBytePermute" && bytes == 4 && es.size() == 3) {
    unsigned selector = pack_constant(es[2]);
    auto x = packed_edge(es[0], depth + 1), y = packed_edge(es[1], depth + 1);
    if (x.size() != 4 || y.size() != 4)
      throw Unsupported("output byte permute width mismatch");
    for (unsigned i = 0; i < 4; ++i) {
      unsigned lane = (selector >> (4 * i)) & 15;
      if (lane >= 8) throw Unsupported("output byte permute sign replication");
      p[i] = lane < 4 ? x[lane] : y[lane - 4];
    }
  } else throw Unsupported("nontransport output packing operation");
  packed_cache.emplace(id, p);
  return p;
}

inline auto Importer::packed_output(const J &body, const Env &env) -> std::optional<Id> {
  if (!ideal || mapping.value("output_value_semantics", "") != "ideal_real")
    return {};
  std::string conversion = body.value("conversion", "");
  const std::string suffix = "Bitcast(u8 -> e4m3)";
  if (!conversion.ends_with(suffix)) return {};
  conversion.resize(conversion.size() - suffix.size());
  auto end = conversion.find_last_not_of(" ;\t");
  conversion = end == std::string::npos ? "" : conversion.substr(0, end + 1);
  Edge e(J{{"node", body.at("node")}, {"conversion", conversion}});
  auto p = packed_edge(e);
  if (p.size() != 1 || p[0] == invalid)
    throw Unsupported("output byte does not identify a quantized scalar");
  ++a.stats["output_packed_quantized_scalars"];
  return import(p[0], env);
}

inline auto Importer::family_environment(const J &f, const OutputPoint &point) -> Env {
  Env env;
  for (size_t i = 0; i < f["shape"].size(); ++i) {
    Q n = jq(f["shape"][i]);
    env[Id(i)] = point && point->first == i ? a.constant(Q(point->second)) :
        n == 1 ? a.zero : a.var("output:" + std::to_string(i), 0, n - Q(1));
  }
  if(f.contains("coordinate_bindings")) {
    Env source;
    const auto& bindings=f.at("coordinate_bindings");
    if(!bindings.is_array())throw Unsupported("invalid output coordinate bindings");
    for(size_t i=0;i<bindings.size();++i) {
      const auto& binding=bindings[i];
      source[Id(i)]=edge(import(input_id(binding.at("node")),env),binding);
    }
    env=std::move(source);
  }
  return env;
}

inline auto Importer::family(const J &f, const std::string &expected_type, const OutputPoint &point) -> Id {
  Env env=family_environment(f, point);
  if (auto result = packed_output(f["body"], env))
    return refuse_raw_byte_views(*result);
  Id n = edge(import(input_id(f["body"]["node"]), env), f["body"], true);
  if (mapping.value("output_value_semantics", "") == "ideal_real" &&
      a.nodes[n].op == "floatbits")
    return refuse_raw_byte_views(a.nodes[n].args[0]);
  if (mapping.value("output_value_semantics", "") == "ideal_real" &&
      floating(expected_type) && (a.nodes[n].op=="readbits" || a.nodes[n].op=="extractbits"))
    if (auto interpreted=initial_typed_bits(n,expected_type)) {
      ++a.stats["typed_output_transport_reads"];
      return refuse_raw_byte_views(*interpreted);
    }
  return byte_view_output(n);
}

inline auto Importer::family_type(const J &f) const -> std::string {
  const auto &body=f.at("body");
  if(body.contains("converted_type"))return body.at("converted_type");
  if(!body.value("conversion","").empty())return "";
  return ns.at(input_id(body.at("node"))).type;
}

} // namespace rg
