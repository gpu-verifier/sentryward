#pragma once
#include "core.hpp"
#include "input.hpp"
#include "equivalence_context.hpp"
#include "floating_idiom_pass.hpp"
#include "packed_decode.hpp"
#include "xor_coordinate.hpp"
#include <regex>
namespace rg {
inline bool floating(const std::string &t) {
  return t == "e2m1" || t == "ue4m3" || t == "ue8m0" || t == "e8m0" || t.starts_with("f") || t.starts_with("bf") || t.starts_with("e4") ||
         t.starts_with("e5") || t.starts_with("tf");
}
inline unsigned width(const std::string &t) {
  if (t == "e2m1") return 4;
  if (t == "e4m3" || t == "ue4m3" || t == "e5m2" || t == "e8m0" || t == "ue8m0")
    return 8;
  auto p = t.find_first_of("0123456789");
  if (p == std::string::npos || p == 0 || t.size() - p > 3)
    throw Unsupported("type width: " + t);
  const std::string prefix = t.substr(0, p);
  if (prefix != "b" && prefix != "u" && prefix != "s" &&
      prefix != "f" && prefix != "bf" && prefix != "tf")
    throw Unsupported("unrecognized numeric type: " + t);
  unsigned w = 0;
  for (size_t i = p; i < t.size(); ++i) {
    if (t[i] < '0' || t[i] > '9')
      throw Unsupported("unrecognized compound type: " + t);
    w = w * 10 + unsigned(t[i] - '0');
  }
  if (w == 0 || w > 256)
    throw Unsupported("type width out of supported range");
  return w;
}
// Lookup one alias without copying the complete allocation/parameter map.
inline std::string mapped_name(const J &mapping, const char *section,
                               const std::string &name) {
  auto table = mapping.find(section);
  if (table == mapping.end())
    return name;
  auto value = table->find(name);
  return value == table->end() ? name : value->get<std::string>();
}
struct VecHash {
  size_t operator()(const std::vector<Id> &v) const {
    size_t h = 0;
    for (Id x : v)
      mix(h, x);
    return h;
  }
};
struct Importer {
  Arena &a;
  Document document;
  const J &doc;
  const std::vector<RawNode> &ns;
  const J &mapping;
  const J &m;
  std::string side;
  FloatingIdiomPlan fp_idioms;
  PackedDecodePass packed_decoders;
  XorCoordinateNormalizer xor_coordinates;
  // Separate caches and entry points: ordinary arithmetic remains Signed4-only.
  PackedDecodePass fp4_boundaries;
  std::unordered_set<Id> fp4_reference_boundaries;
  std::unordered_map<Id,Id> typed_fp4_values;
  std::map<std::pair<Id,std::string>,Id> floating_lane_values;
  std::unordered_set<std::string> packed_dense_layouts;
  std::unordered_set<std::string> packed_identity_layouts;
  std::unordered_map<Id,Id> power_bit_values;
  std::unordered_set<Id> encoded_powers;
  bool ideal;
  const InputContract *contract;
  const PreparedContext *context = nullptr;
  std::set<std::pair<std::string,Id>> unresolved_accesses;
  std::set<std::pair<std::string,std::string>> consumed_contracts;
  std::vector<std::vector<Id>> free;
  // Computed with free variables in the existing topological raw-DAG scan.
  // Leaf Fold bodies can share an alpha binder without placing that binder
  // above a cached inner Fold that already binds the same Arena variable.
  std::vector<uint8_t> raw_has_fold;
  std::unordered_map<std::vector<Id>, Id, VecHash> sibling_binders;
  uint64_t scope_binder_bank = 0, scope_binder_serial = 0;
  static constexpr size_t scope_binder_limit = 4096, scope_environment_limit = 16;
  std::unordered_map<std::vector<Id>, Id, VecHash> cache;
  std::map<std::string, Bounds> ranges;
  using Env = std::map<Id, Id>;
  Importer(Arena &ar, Document d, const J &mp, std::string s, const InputContract *premises = nullptr)
      : a(ar), document(std::move(d)), doc(document.meta), ns(document.nodes),
        mapping(mp), m(mp.at(s + "_inputs")), side(s), packed_decoders(a,PackedDecodePass::Signed4), xor_coordinates(a),
        fp4_boundaries(a,PackedDecodePass::FP4),
        ideal(doc.value("numerical_semantics", "") == "ideal_real"), contract(premises) {
    if (contract) contract->validate_mapping(mapping);
    std::map<std::string, J> canonical;
    for (auto is : {"left_inputs", "right_inputs"}) {
      const auto &sm = mp.at(is);
      if (!sm.contains("layouts"))
        continue;
      for (auto it = sm["layouts"].begin(); it != sm["layouts"].end(); ++it) {
        auto &l = it.value();
        std::string alloc = mapped_name(sm, "allocations", it.key());
        J desc = J::array(
            {l["shape"],
             l.value("target_strides_bytes", l["source_strides_bytes"]),
             l.value("target_offset", J(0)), l["element_bytes"]});
        if (canonical.contains(alloc) && canonical[alloc] != desc)
          throw Unsupported("canonical input layout mismatch: " + alloc);
        canonical[alloc] = desc;
      }
    }
    if (m.contains("layouts"))
      for (auto it = m["layouts"].begin(); it != m["layouts"].end(); ++it) {
        const auto &l = it.value();
        Z span = jq(l["element_bytes"]).numerator();
        std::vector<std::pair<Z, Z>> dims;
        for (size_t k = 0; k < l["shape"].size(); ++k) {
          Z n = jq(l["shape"][k]).numerator();
          if (n > 1)
            dims.emplace_back(jq(l["source_strides_bytes"][k]).numerator(), n);
        }
        std::sort(dims.begin(), dims.end());
        for (auto &[s, n] : dims) {
          if (s != span)
            throw Unsupported("source input layout is not dense permutation: " +
                              it.key());
          span *= n;
        }
        // The local bit-address representation requires a bijective canonical
        // layout. Typed-coordinate reads may still use other existing layouts;
        // they simply do not participate in this packed decoder pass.
        const auto &target=l.contains("target_strides_bytes")?l["target_strides_bytes"]:l["source_strides_bytes"];
        dims.clear();span=jq(l["element_bytes"]).numerator();
        bool dense=span>0 && target.size()==l["shape"].size() &&
            jq(l.value("source_offset",J(0))).denominator()==1 &&
            jq(l.value("target_offset",J(0))).denominator()==1;
        for(size_t k=0;dense && k<l["shape"].size();++k) {
          Q shape=jq(l["shape"][k]),stride=jq(target[k]);
          if(shape.denominator()!=1 || shape<=0 || stride.denominator()!=1 ||
             jq(l["source_strides_bytes"][k]).denominator()!=1) {dense=false;break;}
          if(shape>1)dims.emplace_back(stride.numerator(),shape.numerator());
        }
        std::sort(dims.begin(),dims.end());
        for(const auto &[stride,count]:dims) {
          if(stride!=span) {dense=false;break;}
          span*=count;
        }
        if(dense) {
          packed_dense_layouts.insert(it.key());
          if(target==l["source_strides_bytes"])packed_identity_layouts.insert(it.key());
        }
      }
    for (const auto &c : mp.value("input_memory_constraints", J::array())) {
      if (!c.contains("value_range"))
        continue;
      bool ok = true, found = false;
      for (auto it = c.begin(); it != c.end(); ++it)
        if (it.key() != "allocation" && it.key() != "element_type" &&
            it.key() != "element_count" && it.key() != "value_range")
          ok = false;
      if (!ok)
        continue;
      std::string alloc = c["allocation"];
      if (m.contains("layouts"))
        for (auto it = m["layouts"].begin(); it != m["layouts"].end(); ++it)
          if (mapped_name(m, "allocations", it.key()) == alloc) {
            found = true;
            Z count = 1;
            for (auto &n : it.value()["shape"])
              count *= jq(n).numerator();
            if (!c.contains("element_count") ||
                jq(c["element_count"]) != Q(count) ||
                width(c["element_type"]) !=
                    8 * it.value()["element_bytes"].get<unsigned>())
              ok = false;
          }
      if (ok && found)
        ranges[alloc] = std::pair(jq(c["value_range"]["min_inclusive"]),
                                  jq(c["value_range"]["max_exclusive"]) - Q(1));
    }
    free.reserve(ns.size());
    raw_has_fold.reserve(ns.size());
    for (size_t i = 0; i < ns.size(); ++i) {
      a.check();
      const auto &n = ns[i];
      std::set<Id> fs;
      bool has_fold = n.kind == "Fold";
      std::vector<Id> refs;
      for (auto &e : n.operands) {
        Id r = e.node;
        if (r >= i)
          throw Unsupported("input DAG not topologically ordered");
        refs.push_back(r);
        has_fold = has_fold || raw_has_fold[r];
        fs.insert(free[r].begin(), free[r].end());
      }
      const std::string &k = n.kind;
      if (k == "Index")
        fs = {n.level};
      if (k == "Fold") {
        auto &f = n;
        if (refs.size() != 2)
          throw Unsupported("Fold arity");
        fs = std::set<Id>(free[refs[0]].begin(), free[refs[0]].end());
        if (Q(f.count) != 0)
          for (Id v : free[refs[1]])
            if (v != f.level)
              fs.insert(v);
      }
      free.emplace_back(fs.begin(), fs.end());
      raw_has_fold.push_back(uint8_t(has_fold));
      // Constant-depth checks while already visiting each raw template once.
      // Never scan scalar producers to discover FP4 candidates at import time.
      if (ideal && fp4_reference_candidate(n)) {
        fp4_reference_boundaries.insert(Id(i));
        ++a.stats["fp4_reference_boundary_templates"];
      }
    }
    fp_idioms=FloatingIdiomPlan(ns,a);
    a.stats["fp_idiom_rm_producers"]+=fp_idioms.producers;
    a.stats["fp_idiom_power_constructions"]+=fp_idioms.powers;
  }
  bool fp4_reference_candidate(const RawNode &n) const {
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
  Importer(Arena &ar, Document d, const J &mp, std::string s, const PreparedContext &ctx)
      : Importer(ar,std::move(d),mp,std::move(s),&ctx.contract) { context=&ctx; }
  std::vector<Id> key(Id i, const Env &env) {
    if (i >= ns.size())
      throw Unsupported("raw node reference outside input DAG");
    std::vector<Id> k{i};
    k.reserve(1 + free[i].size());
    for (Id v : free[i]) {
      auto it = env.find(v);
      if (it == env.end())
        throw Unsupported("unbound index level " + std::to_string(v));
      k.push_back(it->second);
    }
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
  Id fold_binder(Id raw, const Env &env) {
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
  // Byte provenance only: invalid denotes a proven zero byte. A source ID
  // denotes an opaque quantized e4m3 byte, never its ideal-real numeric value.
  using Packed = std::vector<Id>;
  std::map<Id, Packed> packed_cache;
  static bool unsigned_word(const std::string &t) {
    return t.size() > 1 && (t[0] == 'u' || t[0] == 'b') &&
           std::all_of(t.begin() + 1, t.end(), [](char c) {
             return c >= '0' && c <= '9';
           });
  }
  unsigned pack_constant(const Edge &e) {
    const auto &n = ns.at(e.node);
    if (!e.conversion.empty() || n.kind != "Constant")
      throw Unsupported("output packing requires constant selector/shift");
    Q q = decimal(n.name);
    if (q.denominator() != 1 || q < 0 || q > 65535)
      throw Unsupported("output packing constant out of range");
    return q.numerator().convert_to<unsigned>();
  }
  Packed packed_edge(const Edge &e, unsigned depth = 0) {
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
  Packed packed_node(Id id, unsigned depth) {
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
  std::optional<Id> packed_output(const J &body, const Env &env) {
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
  Env family_environment(const J &f) {
    Env env;
    for (size_t i = 0; i < f["shape"].size(); ++i) {
      Q n = jq(f["shape"][i]);
      env[Id(i)] =
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
  Id family(const J &f, const std::string &expected_type = "") {
    Env env=family_environment(f);
    if (auto result = packed_output(f["body"], env))
      return *result;
    Id n = edge(import(input_id(f["body"]["node"]), env), f["body"], true);
    if (mapping.value("output_value_semantics", "") == "ideal_real" &&
        a.nodes[n].op == "floatbits")
      return a.nodes[n].args[0];
    if (mapping.value("output_value_semantics", "") == "ideal_real" &&
        floating(expected_type) && (a.nodes[n].op=="readbits" || a.nodes[n].op=="extractbits"))
      if (auto interpreted=initial_typed_bits(n,expected_type)) {
        ++a.stats["typed_output_transport_reads"];
        return *interpreted;
      }
    return n;
  }
  std::string family_type(const J &f) const {
    const auto &body=f.at("body");
    if(body.contains("converted_type"))return body.at("converted_type");
    if(!body.value("conversion","").empty())return "";
    return ns.at(input_id(body.at("node"))).type;
  }
  // Recognize only the exact e4m3 NaN encoding test on initial byte reads.
  // Whole-object constraints are caller premises; address coverage is checked
  // separately before selecting a branch. No constraints on computed bytes.
  std::optional<std::pair<Id,std::string>> finite_scale_guard(const RawNode &select) const {
    if (!contract || select.kind != "Operation" || select.name != "Select" ||
        !select.attributes.empty() || select.operands.size()!=3) return {};
    const auto &ce=select.operands[0];
    if(!ce.conversion.empty()) return {};
    const auto &cmp=ns.at(ce.node);
    if(cmp.kind!="Operation" || cmp.name!="Equal" || cmp.type!="pred" ||
       !cmp.attributes.empty() || cmp.operands.size()!=2) return {};
    // E8M0 has exactly one nonfinite code, 255. This implication is about
    // decoded E8M0 values, not generic byte-valued inputs.
    for(unsigned side=0;side<2;++side) {
      const auto &value=cmp.operands[side], &test=cmp.operands[1-side];
      const auto &v=ns.at(value.node),&c=ns.at(test.node);
      if(value.conversion.empty() && test.conversion.empty() && c.kind=="Constant" && c.type=="u8" && c.name=="255" &&
         (v.kind=="InitialCell" || v.kind=="InputValue") && v.type=="u8" && v.operands.size()==1) {
        const auto* rule=contract->find(mapped_name(m,"allocations",v.name),"e8m0");
        if(rule && rule->finite)return std::pair(value.node,std::string("e8m0"));
      }
    }
    auto plain127=[&](const Edge &e) {
      const auto &v=ns.at(e.node);
      return e.conversion.empty() && v.kind=="Constant" && v.type=="u8" && v.name=="127";
    };
    for(unsigned side=0;side<2;++side) {
      if(!plain127(cmp.operands[1-side]) || !cmp.operands[side].conversion.empty()) continue;
      const auto &mask=ns.at(cmp.operands[side].node);
      if(mask.kind!="Operation" || mask.name!="BitAnd" || mask.type!="u8" ||
         !mask.attributes.empty() || mask.operands.size()!=2) continue;
      for(unsigned j=0;j<2;++j) {
        if(!plain127(mask.operands[1-j]) || !mask.operands[j].conversion.empty()) continue;
        Id read=mask.operands[j].node; const auto &v=ns.at(read);
        if((v.kind!="InitialCell" && v.kind!="InputValue") || v.type!="u8" || v.operands.size()!=1) continue;
        auto name=mapped_name(m,"allocations",v.name);
        const auto *rule=contract->find(name,"e4m3");
        if(rule && rule->finite) return std::pair(read,std::string("e4m3"));
      }
    }
    return {};
  }
  Id import(Id root, const Env &env) {
    struct Frame {
      Id i;
      Env env;
      bool ready;
    };
    std::vector<Frame> todo{{root, env, false}};
    while (!todo.empty()) {
      a.check();
      Frame f = std::move(todo.back());
      todo.pop_back();
      auto k = key(f.i, f.env);
      if (cache.contains(k))
        continue;
      const auto &n = ns[f.i];
      const auto &edges = n.operands;
      bool fold = n.kind == "Fold", empty = fold && Q(n.count) == 0;
      size_t len = empty ? 1 : edges.size();
      Env body = f.env;
      if (fold && !empty)
        body[n.level] = fold_binder(f.i, f.env);
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
          auto result=a.logarithm(x,true);
          cache.emplace(std::move(k),context?context->apply(a,result):result);
          ++a.stats["ideal_compiler_log2_polynomials"];continue;
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
        contract->applied.emplace(side,mapped_name(m,"allocations",input.name),read->second);
        if (!cache.contains(chosen_key)) {
          todo.push_back({f.i,f.env,false});
          todo.push_back({chosen.node,f.env,false});
          continue;
        }
        cache.emplace(std::move(k),edge(cache.at(chosen_key),chosen));
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
          cache.emplace(std::move(k),context?context->apply(a,selected):selected);
          ++a.stats["proved_select_arms_imported"];continue;
        }
      }
      if (!f.ready) {
        todo.push_back({f.i, f.env, true});
        for (size_t j = len; j-- > 0;) {
          const Env &ctx = fold && j == 1 ? body : f.env;
          Id r = edges[j].node;
          if (!cache.contains(key(r, ctx)))
            todo.push_back({r, ctx, false});
        }
        continue;
      }
      std::vector<Id> xs;
      for (size_t j = 0; j < len; ++j) {
        const Env &ctx = fold && j == 1 ? body : f.env;
        Id operand=cache.at(key(edges[j].node,ctx));
        xs.push_back(fp_idioms.is_power(f.i) && j==0 ? operand : edge(operand,edges[j]));
      }
      Id result;
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
        encoded_powers.insert(power);
        ++a.stats["fp_idiom_power_instances"];
      } else result = empty ? xs[0] : make(n, xs, f.env, body);
      // Restrict local decoder proofs to arithmetic/selection boundaries. Raw
      // transports keep their original provenance for subsequent typed casts.
      if(n.kind=="Operation" && (n.name=="Add" || n.name=="Subtract" ||
          n.name=="Multiply" || n.name=="PtxMultiplyAdd" || n.name=="Select")) {
        if(n.name=="Select" && fp4_reference_boundaries.contains(f.i))
          result=fp4_boundaries.normalize_fp4_boundary(result,
              [&](Id source,unsigned lo,unsigned count) {
                return canonical_packed_slice(source,lo,count);
              });
        result=normalize_packed_decoder(result);
      }
      cache.emplace(std::move(k), context ? context->apply(a,result) : result);
      ++a.stats["imported_context_nodes"];
    }
    return cache.at(key(root, env));
  }
  // Recognize one complete binary32 compiler LSE tail under ideal_real.
  // This node means ln(mass) + ln(2)*pivot_in_log2_units. Only the
  // witnessed tail's rounded ln(2) is idealized; neither the mass's exp2
  // coefficients nor unrelated ln(2)/log2(e) literals are changed.
  // The raw scan has fixed depth and arity (at most four orientations).
  std::optional<Id> compiler_lse(const RawNode &root, const Env &env) {
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
      auto found = cache.find(std::vector<Id>{raw});
      if (found == cache.end()) return false;
      auto decoded = a.value(found->second);
      return decoded && *decoded == value;
    };
    auto imported = [&](Id raw) -> std::optional<Id> {
      auto found = cache.find(key(raw, env));
      return found == cache.end() ? std::optional<Id>{} : found->second;
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
  Id numerical_product(Id x, Id y) {
    Id product = a.mul({x, y});
    if (!ideal || a.nodes[x].op != "exp2" || a.nodes[y].op != "exp2" ||
        encoded_powers.empty() ||
        (!encoded_powers.contains(x) && !encoded_powers.contains(y)))
      return product;
    // Ideal exp contract for the recognized split-log2(e) exponent encoding.
    // This is intentionally NOT a global approximate-constant equality rule.
    // One operand must be a power produced by the checked fma.rm/bit idiom.
    const auto &node = a.nodes[product];
    if (node.op != "exp2" || !a.nodes[node.args[0]].facts.finite) return product;
    const auto &e = a.nodes[node.args[0]];
    static const Q log2e_split(Z("25380159564675"), Z("17592186044416"));
    Id argument = invalid;
    if (e.op == "mul" && e.args.size() == 1 && e.data[0] == log2e_split)
      argument = e.args[0];
    else if (e.op == "add" && e.args.size() == 2 &&
             ((e.data[0] == log2e_split && e.data[1] == -log2e_split) ||
              (e.data[1] == log2e_split && e.data[0] == -log2e_split))) {
      Id positive = e.args[e.data[0] > 0 ? 0 : 1];
      Id negative = e.args[e.data[0] > 0 ? 1 : 0];
      argument = a.sub(positive, negative);
    }
    if (argument == invalid) return product;
    ++a.stats["ideal_compiler_exp_sequences"];
    return a.exp(argument);
  }
  Id word(Id x, unsigned w) {
    auto b = a.nodes[x].bounds;
    Z lim = pow2(w);
    if (b && b->first >= 0 && b->first <= b->second && b->second < Q(lim))
      return x;
    const auto &n = a.nodes[x];
    if (n.op == "signed" && n.data[0] == Q(w))
      return word(n.args[0], w);
    if (n.op == "word" && n.data[0] <= Q(w))
      return x;
    auto ts = a.terms(x);
    if (std::all_of(ts.begin(), ts.end(),
                    [](auto &t) { return t.second.denominator() == 1; })) {
      std::vector<Id> rs;
      for (auto &[t, c] : ts) {
        Z coefficient = modz(c.numerator(), lim);
        // Keep negative residues negative when this proves a small unsigned
        // result (e.g. packed INT4 xor sign toggles). Congruence is unchanged.
        if (t != a.one && c < 0 && coefficient != 0) coefficient -= lim;
        // Reducing a coefficient modulo 2^w is valid only for integral
        // terms. Unknown terms retain their exact coefficient, rather than
        // being rejected or silently treated as integers.
        Q exact = Q(coefficient);
        if (exact != c && t != a.one && !packed_decoders.integer(t)) exact = c;
        rs.push_back(a.mul({a.constant(exact), t}));
      }
      Id r = a.add(rs);
      auto rb = a.nodes[r].bounds;
      if (rb && rb->first >= 0 && rb->first <= rb->second &&
          rb->second < Q(lim))
        return r;
      // A negative integer stride can arrive encoded as an unsigned residue.
      // Choose balanced coefficients only for proved integer atoms; changing
      // a coefficient by 2^w is otherwise not a congruence (e.g. x=1/2).
      // The whole candidate must still fit the unsigned word domain.
      rs.clear();
      bool changed = false, integral = true;
      for (const auto &[t, c] : ts) {
        if (t != a.one && !packed_decoders.integer(t)) { integral = false; break; }
        Z coefficient = modz(c.numerator(), lim);
        if (coefficient > lim / 2) { coefficient -= lim; changed = true; }
        rs.push_back(a.mul({a.constant(Q(coefficient)), t}));
      }
      if (integral && changed) {
        Id candidate = a.add(rs);
        auto cb = a.nodes[candidate].bounds;
        if (cb && cb->first >= 0 && cb->first <= cb->second && cb->second < Q(lim)) {
          ++a.stats["bounded_unsigned_stride_recoveries"];
          return candidate;
        }
      }
    }
    return a.intern("word", {x}, {Q(w)}, "", {true, false, true},
                    std::pair(Q(0), Q(lim - 1)));
  }
  Id unwrap(Id x, unsigned w) {
    while (a.nodes[x].op == "signed" && a.nodes[x].data[0] == Q(w))
      x = a.nodes[x].args[0];
    return a.nodes[x].op == "word" && a.nodes[x].data[0] == Q(w)
               ? a.nodes[x].args[0]
               : x;
  }
  // Recognize a bounded bit slice of an initial packed load. No value/IEEE
  // inference: shifts and truncations only identify the original storage bits.
  std::optional<Id> initial_bit_slice(Id x, unsigned lo, unsigned count, unsigned depth=0) {
    if(depth>8 || !count) return {};
    const auto &n=a.nodes[x];
    if(n.op=="readbits") {
      if(Q(lo+count)>n.data[0]) return {};
      return a.bit_slice(x,lo,count);
    }
    if(n.op=="extractbits") {
      if(Q(lo+count)>n.data[1]) return {};
      return initial_bit_slice(n.args[0],lo+n.data[0].numerator().convert_to<unsigned>(),count,depth+1);
    }
    if(n.op=="word") {
      if(Q(lo+count)>n.data[0]) return {};
      return initial_bit_slice(n.args[0],lo,count,depth+1);
    }
    if(n.op=="idiv" || n.op=="imod") {
      const Q &q=n.data[0];
      if(q.denominator()!=1 || q<=0) return {};
      Z divisor=q.numerator();if((divisor & (divisor-1))!=0) return {};
      unsigned shift=boost::multiprecision::msb(divisor);
      if(shift>256) return {};
      if(n.op=="imod") {
        if(lo+count>shift)return {};
        return initial_bit_slice(n.args[0],lo,count,depth+1);
      }
      auto bounds=a.nodes[n.args[0]].bounds;
      if(!bounds || bounds->first<0)return {};
      return initial_bit_slice(n.args[0],lo+shift,count,depth+1);
    }
    return {};
  }
  std::optional<Id> initial_typed_bits(Id x,const std::string &type) {
    // Raw transport does not establish UE4M3 format/padding validity. Only
    // explicitly typed initial inputs enter its finite-domain decoder.
    if (type == "ue4m3") return {};
    unsigned bits=width(type);if(bits%8) return {};
    const auto &n=a.nodes[x];
    Id read=x;Q bit_offset(0);
    if(n.op=="extractbits" && n.data[1]==Q(bits)) {read=n.args[0];bit_offset=n.data[0];}
    else if(n.op!="readbits" || n.data[0]!=Q(bits)) return {};
    const auto &r=a.nodes[read];
    if(r.op!="readbits" || bit_offset.denominator()!=1 || modz(bit_offset.numerator(),8)!=0) return {};
    if(!m.contains("layouts") || !m["layouts"].contains(r.tag) ||
       m["layouts"][r.tag]["element_bytes"].get<unsigned>()*8!=bits) return {};
    std::string raw=r.tag;Id address=r.args[0];
    return typed_read(raw,a.add({address,a.constant(bit_offset/Q(8))}),type);
  }
  // NumericalInterpretation of an already typed UE4M3 input.  This follows
  // the seven payload bits in the saved expression's unsigned format, not an
  // arbitrary b8 bitcast or a proof that PTX padding requirements were met.
  // E4M3 finite excludes exactly the same low-seven-bit NaN payload (127);
  // it does not constrain bit 7 and must not be treated as a padding premise.
  Id typed_ue4m3_input(const std::string &raw, Id offset) {
    if (!ideal)
      throw Unsupported("typed UE4M3 input requires ideal_real numerical semantics");
    const std::string allocation = mapped_name(m, "allocations", raw);
    const auto *rule = contract ? contract->find(allocation, "e4m3") : nullptr;
    if (!rule || !rule->finite)
      throw Unsupported("typed UE4M3 input requires an explicit finite E4M3 premise: " + allocation);
    // Validate the complete byte and the correspondence before consuming a
    // whole-object premise; an unknown/out-of-range address cannot use it.
    typed_read(raw, offset, "u8", true);
    Id payload = a.intern("readbits", {offset}, {Q(8)}, raw,
                          {true, false, true}, std::pair(Q(0), Q(255)));
    Id fraction = a.imod(payload, Z(8));
    // Use the same exact unsigned shift representation as the raw reference
    // decoder. The final modulus still excludes storage bit 7 from the value.
    Id exponent = a.imod(a.bit_slice(payload, 3, 5), Z(16));
    Id denormal = a.mul({a.constant(Q(1, 512)), fraction});
    Id normal = a.mul({a.exp2(a.sub(exponent, a.constant(7))),
                       a.add({a.one, a.mul({a.constant(Q(1, 8)), fraction})})});
    Id result = a.select(a.compare("eq", exponent, a.zero), denormal, normal);
    contract->applied.emplace(side, allocation, "e4m3");
    ++a.stats["typed_ue4m3_initial_decodes"];
    return result;
  }
  Id typed_e8m0_input(const std::string& raw,Id offset) {
    const auto allocation=mapped_name(m,"allocations",raw);
    const auto* rule=contract?contract->find(allocation,"e8m0"):nullptr;
    if(!ideal || !rule || !rule->finite)throw Unsupported("typed E8M0 input requires explicit finite E8M0 premise: "+allocation);
    typed_read(raw,offset,"u8",true);
    Id payload=a.intern("readbits",{offset},{Q(8)},raw,{true,false,true},std::pair(Q(0),Q(255)));
    contract->applied.emplace(side,allocation,"e8m0");++a.stats["typed_e8m0_initial_decodes"];
    return a.exp2(a.sub(payload,a.constant(127)));
  }
  // Demand one complete typed floating lane from exact integer transport.
  // The walk follows only one containing field; other fields are checked by
  // bounded integer/range summaries, never by visiting numeric ancestors.
  // A failed proof does not make an integer payload a numeric conversion.
  std::optional<Id> floating_lane(Id source,const std::string& type) {
    const auto key=std::pair(source,type);
    if(auto it=floating_lane_values.find(key);it!=floating_lane_values.end())return it->second;
    const unsigned count=width(type);
    if(type!="f16" && type!="bf16" && type!="f32" && type!="f64")return {};
    unsigned lo=0,budget=PackedDecodePass::max_nodes;
    Id value=source;
    auto remember=[&](Id result) -> std::optional<Id> {
      floating_lane_values.emplace(key,result);++a.stats["exact_packed_floating_lanes"];return result;
    };
    while(budget) {
      --budget;++a.stats["floating_lane_transport_visits"];
      // No arena insertion occurs until a terminal field is found. References
      // below cannot be invalidated by arithmetic interning during traversal.
      const auto& n=a.nodes[value];
      if(n.op=="floatbits") {
        if(lo==0 && n.args.size()==1 && n.data==std::vector<Q>{Q(count)} && n.tag==type)
          return remember(n.args[0]);
        return {};
      }
      if(n.op=="readbits") {
        if(n.args.size()!=1 || n.data.size()!=1 || lo%8 || Q(lo+count)>n.data[0])return {};
        const auto raw=n.tag;const auto address=n.args[0];
        return remember(typed_read(raw,a.add({address,a.constant(lo/8)}),type));
      }
      if(n.args.size()==1 && (n.op=="word" || n.op=="imod" || n.op=="idiv" || n.op=="extractbits")) {
        unsigned shift=0,available=256;
        if(n.op=="word") {
          if(n.data.size()!=1 || n.data[0].denominator()!=1 || n.data[0]<=0 || n.data[0]>256)return {};
          available=n.data[0].numerator().convert_to<unsigned>();
        } else if(n.op=="extractbits") {
          if(n.data.size()!=2 || n.data[0].denominator()!=1 || n.data[1].denominator()!=1 ||
             n.data[0]<0 || n.data[0]>256 || n.data[1]<=0 || n.data[1]>256)return {};
          shift=n.data[0].numerator().convert_to<unsigned>();available=n.data[1].numerator().convert_to<unsigned>();
        } else {
          auto radix=n.data.size()==1?PackedDecodePass::power_of_two(n.data[0]):std::optional<unsigned>{};
          if(!radix)return {};
          if(n.op=="idiv")shift=*radix;else available=*radix;
        }
        if(lo+count>available || lo+shift+count>256)return {};
        const auto& input=a.nodes[n.args[0]];
        if(!input.bounds || input.bounds->first<0 || !packed_decoders.integer_impl(n.args[0],budget))return {};
        value=n.args[0];lo+=shift;continue;
      }
      if(n.op=="mul" && n.args.size()==1 && n.data.size()==1) {
        auto shift=PackedDecodePass::power_of_two(n.data[0]);
        if(!shift || lo<*shift)return {};
        const auto& input=a.nodes[n.args[0]];
        if(!input.bounds || input.bounds->first<0 || !packed_decoders.integer_impl(n.args[0],budget))return {};
        value=n.args[0];lo-=*shift;continue;
      }
      if(n.op!="add" || n.args.empty() || n.args.size()>8 || n.args.size()!=n.data.size())return {};
      struct Field {Id value;unsigned first,last;};
      std::vector<Field> fields;fields.reserve(n.args.size());
      std::optional<Field> selected;
      for(size_t i=0;i<n.args.size();++i) {
        if(!budget)return {};
        --budget;
        auto shift=PackedDecodePass::power_of_two(n.data[i]);const auto& part=a.nodes[n.args[i]];
        if(!shift || !part.bounds || part.bounds->first<0 || !part.facts.finite ||
           !packed_decoders.integer_impl(n.args[i],budget))return {};
        const Z high=floorq(part.bounds->second);
        if(high<0)return {};
        if(high==0)continue;
        if(high>=pow2(256))return {};
        unsigned bits=boost::multiprecision::msb(high)+1;
        if(*shift+bits>256)return {};
        Field field{n.args[i],*shift,*shift+bits};
        for(auto previous:fields)
          if(field.first<previous.last && previous.first<field.last)return {};
        fields.push_back(field);
        if(lo>=field.first && lo+count<=field.last)selected=field;
      }
      if(!selected)return {}; // crossing fields or including padding is not a typed lane
      value=selected->value;lo-=selected->first;
    }
    return {};
  }
  Id cast(Id x, const std::string &src, const std::string &dst,
          bool bitcast = false) {
    if (src == dst)
      return x;
    if((src=="e8m0" || src=="ue8m0" || dst=="e8m0" || dst=="ue8m0") && bitcast)
      throw Unsupported("raw E8M0 bitcast requires format/domain evidence");
    if(dst=="e8m0" || dst=="ue8m0")throw Unsupported("E8M0 construction outside typed-input interpretation");
    // Typed input numerical interpretation is supported above.  A raw byte
    // bitcast has no established UE4M3 padding/NaN domain, and constructing new
    // UE4M3 values would require the specified quantization semantics.
    if ((src == "ue4m3" || dst == "ue4m3") && bitcast)
      throw Unsupported("raw UE4M3 bitcast requires explicit format/domain evidence");
    if (dst == "ue4m3")
      throw Unsupported("UE4M3 value construction is outside typed-input interpretation");
    if(src=="pred" && !bitcast &&
       ((floating(dst) && ideal) || dst.starts_with("u") || dst.starts_with("s"))) {
      auto bounds=a.nodes[x].bounds;
      if(!bounds || bounds->first<0 || bounds->second>1)
        throw Unsupported("predicate conversion outside Boolean range");
      if(!floating(dst) && width(dst)<(dst.starts_with("s")?2u:1u))
        throw Unsupported("predicate conversion destination cannot represent one");
      return x;
    }
    unsigned sw = width(src), dw = width(dst);
    if (bitcast && sw != dw)
      throw Unsupported("bitcast changes width");
    if (floating(src) && floating(dst) && !bitcast) {
      if (!ideal)
        throw Unsupported("floating convert without ideal_real");
      return x;
    }
    if (floating(dst) && bitcast) {
      if(src=="b32" && dst=="f32")
        if(auto found=power_bit_values.find(x);found!=power_bit_values.end()) {
          ++a.stats["fp_idiom_numeric_replacements"];
          return found->second;
        }
      const auto &n = a.nodes[x];
      if(dst=="e2m1" && src=="b4") {
        if(!ideal)throw Unsupported("FP4 numerical interpretation requires ideal_real");
        auto bounds=n.bounds;
        if(!n.facts.finite || !bounds || bounds->first<0 || bounds->second>15 ||
           (n.op=="const" && n.data[0].denominator()!=1))
          throw Unsupported("FP4 payload outside four-bit storage");
        ++a.stats["exact_e2m1_decodes"];
        if(auto it=typed_fp4_values.find(x);it!=typed_fp4_values.end()) {
          ++a.stats["typed_e2m1_decode_cache_hits"];
          return it->second;
        }
        // The explicit b4->e2m1 edge already defines the sixteen-value decoder.
        // Transport only proves which four stored bits it consumes, including
        // full original-load extent and canonical layout checks in the mapper.
        auto mapper=[&](Id source,unsigned lo,unsigned count) {
          return canonical_packed_slice(source,lo,count);
        };
        if(auto slice=fp4_boundaries.transport(x,mapper);slice && slice->count==4)
          if(auto bits=fp4_boundaries.resolve(*slice,mapper);bits && bits->count==4) {
            Id payload=a.intern("packedbits",{bits->address},{Q(4)},bits->allocation,
                                {true,false,true},std::pair(Q(0),Q(15)));
            Id decoded=a.intern("decode_fp4",{payload},{},"",{true,false,false},
                                std::pair(Q(-6),Q(6)));
            typed_fp4_values.emplace(x,decoded);
            ++a.stats["typed_e2m1_canonical_decodes"];
            return decoded;
          }
        // Non-storage payloads and unresolved provenance retain the existing
        // exact arithmetic interpretation; no storage identity is invented.
        Id fraction=a.imod(x,2), exponent=a.imod(a.idiv(x,2),4);
        Id half=a.mul({a.constant(Q(1,2)),fraction});
        Id scale=a.select(a.compare("eq",exponent,a.one),a.one,
            a.select(a.compare("eq",exponent,a.constant(2)),a.constant(2),a.constant(4)));
        Id magnitude=a.select(a.compare("eq",exponent,a.zero),half,
            a.mul({scale,a.add({a.one,half})}));
        Id decoded=a.select(a.compare("ge",x,a.constant(8)),a.neg(magnitude),magnitude);
        typed_fp4_values.emplace(x,decoded);
        return decoded;
      }
      // Decode IEEE binary32 bits, never numerically cast the integer payload.
      // Fixed-width work: no traversal of the producer DAG or host FP rounding.
      if (dst == "f32" && (src == "b32" || src == "u32" || src == "s32") &&
          n.op == "const") {
        const Q &value = n.data[0];
        Z lower = src == "s32" ? -pow2(31) : Z(0);
        Z upper = src == "s32" ? pow2(31) : pow2(32);
        if (value.denominator() != 1 || value.numerator() < lower ||
            value.numerator() >= upper)
          throw Unsupported(
              "binary32 bitcast payload outside source integer range");
        uint32_t bits =
            modz(value.numerator(), pow2(32)).convert_to<uint32_t>();
        unsigned exponent = (bits >> 23) & 255u;
        unsigned fraction = bits & 0x7fffffu;
        if (exponent == 255) {
          if(fraction==0)return a.infinity((bits>>31)?-1:1);
          throw Unsupported("NaN binary32 constant bitcast");
        }
        if (bits == 0x80000000u)
          throw Unsupported("negative-zero binary32 constant bitcast requires "
                            "signed-zero semantics");
        Z significand = fraction + (exponent ? 0x800000u : 0u);
        int shift = (exponent ? int(exponent) - 127 : -126) - 23;
        Q decoded = shift >= 0 ? Q(significand << shift)
                               : Q(significand, pow2(unsigned(-shift)));
        if (bits >> 31)
          decoded = -decoded;
        ++a.stats["constant_binary32_bitcasts"];
        return a.constant(decoded);
      }
      // Exact binary16/BF16 literal decoding, including reduction sentinels.
      // A numeric cast of the integer payload would be incorrect.
      if((dst=="f16" || dst=="bf16") && sw==16 && n.op=="const") {
        const auto value=n.data[0];const bool signed_source=src=="s16";
        if(value.denominator()!=1 || value< Q(signed_source?-32768:0) || value>Q(signed_source?32767:65535))
          throw Unsupported("16-bit floating payload outside source range");
        const unsigned bits=modz(value.numerator(),pow2(16)).convert_to<unsigned>();
        const unsigned fraction_bits=dst=="f16"?10:7, bias=dst=="f16"?15:127;
        const unsigned max_exponent=dst=="f16"?31:255;
        const unsigned exponent=(bits>>fraction_bits)&max_exponent, fraction=bits&((1u<<fraction_bits)-1);
        if(exponent==max_exponent) {
          if(fraction==0)return a.infinity((bits>>15)?-1:1);
          throw Unsupported("NaN 16-bit floating constant bitcast");
        }
        if(bits==0x8000)throw Unsupported("negative-zero 16-bit floating constant bitcast requires signed-zero semantics");
        Z significand=fraction+(exponent?(1u<<fraction_bits):0u);
        const int shift=(exponent?int(exponent)-int(bias):1-int(bias))-int(fraction_bits);
        Q decoded=shift>=0?Q(significand<<shift):Q(significand,pow2(unsigned(-shift)));
        return a.constant((bits>>15)?-decoded:decoded);
      }
      if (n.op == "floatbits" && n.tag == dst && n.data[0] == Q(dw))
        return n.args[0];
      if (n.op == "readbits" && n.data[0] == Q(dw))
        return typed_read(n.tag, n.args[0], dst);
      if (n.op == "extractbits" && modz(n.data[0].numerator(), 8) == 0 &&
          n.data[1] == Q(dw)) {
        const auto &p = a.nodes[n.args[0]];
        if (p.op == "readbits")
          return typed_read(
              p.tag, a.add({p.args[0], a.constant(n.data[0] / Q(8))}), dst);
      }
      // Packed BF16 lanes may first be sliced from a 64-bit load and shifted
      // under a 32-bit truncation. Recover exactly the contributing 16 bits.
      if(dst=="f32" && sw==32) {
        bool truncated=n.op=="word" && n.data[0]==Q(32);
        const auto &expanded=a.nodes[truncated?n.args[0]:x];
        if(expanded.op=="mul" && expanded.args.size()==1 && expanded.data[0]==Q(65536)) {
          Id bits=expanded.args[0];auto bound=a.nodes[bits].bounds;
          if(truncated || (bound && bound->first>=0 && bound->second<65536))
            if(auto slice=initial_bit_slice(bits,0,16)) {
              ++a.stats["packed_bf16_lane_recoveries"];
              return cast(*slice,"u16","bf16",true);
            }
        }
      }
      // BF16 register expansion: zero-extend its bits and shift into f32's
      // high half. Recover the original numeric BF16 input, not an integer cast.
      if (dst == "f32" && sw == 32 && n.op == "mul" &&
          n.args.size() == 1 && n.data[0] == Q(65536)) {
        Id bits = n.args[0];
        auto bound = a.nodes[bits].bounds;
        if (bound && bound->first >= 0 && bound->second < 65536)
          return cast(bits, "u16", "bf16", true);
      }
      // Constructed BF16 values with fixed sign/exponent are affine in their
      // seven fraction bits. No approximate host float or instance enumeration.
      if (dst == "bf16" && sw == 16) {
        auto bound = n.bounds;
        if (bound && bound->first >= 0 && bound->second < 65536 &&
            floorq(bound->first / Q(128)) == floorq(bound->second / Q(128))) {
          unsigned header = floorq(bound->first / Q(128)).convert_to<unsigned>();
          unsigned exponent = header & 255;
          if (exponent > 0 && exponent < 255) {
            Id significand = a.add({x, a.constant(Q(128 - int(header * 128)))});
            int shift = int(exponent) - 134;
            Q scale = shift >= 0 ? Q(pow2(unsigned(shift))) : Q(1, pow2(unsigned(-shift)));
            return a.mul({a.constant(header >= 256 ? -scale : scale), significand});
          }
        }
      }
      if (dst == "bf16" && sw == 16 && n.bounds) {
        auto bound = *n.bounds;
        bool positive = bound.first >= 128 && bound.second < 32640;
        bool negative = bound.first >= 32896 && bound.second < 65408;
        if (positive || negative) {
          // Interval facts may cross an exponent boundary. Decode the field
          // exactly instead of assuming its exponent is constant.
          Id header = a.idiv(x, Z(128));
          Id exponent = a.sub(header, a.constant(negative ? 390 : 134));
          Id significand = a.add({a.constant(128), a.imod(x, Z(128))});
          return a.mul({a.constant(negative ? -1 : 1), significand, a.exp2(exponent)});
        }
      }
      if(auto lane=floating_lane(x,dst))return *lane;
      throw Unsupported("bitcast to floating is not a numeric conversion: " +
                        src + " -> " + dst);
    }
    if (floating(src)) {
      if (bitcast && sw == dw)
        return a.intern("floatbits", {x}, {Q(sw)}, src, {true, false, true},
                        std::pair(Q(0), Q(pow2(sw) - 1)));
      throw Unsupported("float to integer/bit conversion: " + src + " -> " +
                        dst);
    }
    if (floating(dst)) {
      if (!ideal)
        throw Unsupported("integer to float exact mode");
      return x;
    }
    if (dw < sw) {
      const auto &n = a.nodes[x];
      if (n.op == "readbits")
        x = a.bit_slice(x,0,dw);
      else if (n.op == "extractbits" && n.data[1] >= Q(dw))
        x = a.intern("extractbits", n.args, {n.data[0], Q(dw)}, "",
                     {true, false, true}, std::pair(Q(0), Q(pow2(dw) - 1)));
      else
        x = word(x, dw);
    }
    if (dst.starts_with("s")) {
      if (auto read=initial_typed_bits(x,dst)) {
        ++a.stats["signed_input_transport_reads"];
        return *read;
      }
      return a.signed_word(x, dw);
    }
    return word(x, dw);
  }
  Id edge(Id x, const J &e, bool output = false) {
    return edge_conversion(x, e.value("conversion", ""), output);
  }
  Id edge(Id x, const Edge &e, bool output = false) {
    return edge_conversion(x, e.conversion, output);
  }
  Id edge_conversion(Id x, const std::string &conv, bool output = false) {
    if (conv.empty())
      return x;
    static const std::regex re(
        R"(\s*(Bitcast|Convert|NumericInterpretation)([^()]*)\(\s*([^ ()]+)\s*->\s*([^ ()]+)\s*\)\s*)");
    std::stringstream ss(conv);
    std::string part;
    while (std::getline(ss, part, ';')) {
      std::smatch m;
      if (!std::regex_match(part, m, re))
        throw Unsupported("unparsed edge conversion: " + part);
      std::string attr = m[2], src = m[3], dst = m[4];
      if (attr.find("saturat") != attr.npos &&
          !(output && floating(src) && floating(dst) &&
            mapping.value("output_value_semantics", "") == "ideal_real")) {
        if (!ideal || !floating(src) || !floating(dst))
          throw Unsupported("Saturation requires supported ideal-real floating semantics",
                            "saturation_semantics_unsupported",
                            {{"ideal_real",ideal},{"source_type",src},{"target_type",dst},{"conversion",part}});
        if (!a.nodes[x].facts.finite)
          throw Unsupported("Could not prove that the saturation input is finite",
                            "saturation_input_finiteness_not_proven",
                            {{"input_expression",a.render(x,3)},{"source_type",src},{"target_type",dst},{"conversion",part}});
        Q lo, hi;
        if (attr == " saturate ") { lo = Q(0); hi = Q(1); }
        else if (attr == " rounding=nearest-even saturate_finite " && dst == "e4m3") {
          lo = Q(-448); hi = Q(448);
        } else throw Unsupported("unsupported intermediate saturation mode: " + part);
        x = a.neg(a.maximum({a.neg(a.maximum({x,a.constant(lo)})), a.constant(-hi)}));
      }
      x = cast(x, src, dst, m[1] == "Bitcast");
    }
    return x;
  }
  Id literal(const std::string &s, const std::string &t) {
    if (s == "-inf" || s == "-infinity" || s == "-Infinity")
      return a.infinity(-1);
    if (s == "inf" || s == "+inf" || s == "infinity" || s == "Infinity")
      return a.infinity(1);
    try {
      Q q;
      if (t == "f32") {
        size_t used;
        float f = std::stof(s, &used);
        if (used != s.size() || !std::isfinite(f))
          throw Unsupported("constant " + s + " type " + t);
        uint32_t bits = std::bit_cast<uint32_t>(f);
        int ex = int((bits >> 23) & 255);
        Z sig = (bits & 0x7fffff) + (ex ? 0x800000 : 0);
        int shift = (ex ? ex - 127 : 1 - 127) - 23;
        q = shift >= 0 ? Q(sig << shift) : Q(sig, pow2(unsigned(-shift)));
        if (bits >> 31)
          q = -q;
      } else if (s.starts_with("0x") || s.starts_with("0X"))
        q = Q(Z(s));
      else
        q = decimal(s);
      if (!floating(t) && t != "pred") {
        Z v = modz(q.numerator() / q.denominator(), pow2(width(t)));
        if (t.starts_with("s") && v >= pow2(width(t) - 1))
          v -= pow2(width(t));
        q = Q(v);
      }
      return a.constant(q);
    } catch (const Unsupported &) {
      throw;
    } catch (...) {
      throw Unsupported("constant " + s + " type " + t);
    }
  }
  std::optional<PackedDecodePass::Bits> canonical_packed_slice(Id source,unsigned lo,unsigned count) {
    const auto &read=a.nodes[source];
    if(read.op!="readbits" || read.args.size()!=1 || !count || count>4 ||
       lo%8+count>8 || Q(lo+count)>read.data[0] ||
       !m.contains("layouts") || !m["layouts"].contains(read.tag) ||
       !packed_dense_layouts.contains(read.tag))return {};
    const auto &layout=m["layouts"][read.tag];
    unsigned bytes=layout["element_bytes"].get<unsigned>();
    if(!bytes)return {};
    Id off=a.sub(read.args[0],a.constant(jq(layout.value("source_offset",J(0)))));
    Z extent=bytes;
    for(size_t j=0;j<layout["shape"].size();++j)
      extent+=(jq(layout["shape"][j]).numerator()-1)*jq(layout["source_strides_bytes"][j]).numerator();
    auto bounds=a.nodes[off].bounds;
    // Establish coverage of the original load, not merely its selected nibble.
    // Unknown accesses retain the original expression and no new assumptions.
    if(!bounds || bounds->first<0 || bounds->second+Q((read.data[0].numerator()+7)/8)>Q(extent))return {};
    Id selected=a.add({off,a.constant(lo/8)});
    if(packed_identity_layouts.contains(read.tag)) {
      // Same dense coordinates: no need to decompose and reassemble each axis.
      Id byte_address=a.add({selected,a.constant(jq(layout.value("target_offset",J(0))))});
      Id bit_address=a.add({a.mul({a.constant(8),byte_address}),a.constant(lo%8)});
      ++a.stats["packed_slice_identity_mappings"];
      return PackedDecodePass::Bits{mapped_name(m,"allocations",read.tag),bit_address,count};
    }
    const auto &target=layout.contains("target_strides_bytes")?layout["target_strides_bytes"]:layout["source_strides_bytes"];
    std::vector<Id> terms{a.constant(jq(layout.value("target_offset",J(0)))),a.imod(selected,Z(bytes))};
    for(size_t j=0;j<layout["shape"].size();++j) {
      Id coordinate=a.imod(a.idiv(selected,jq(layout["source_strides_bytes"][j]).numerator()),jq(layout["shape"][j]).numerator());
      terms.push_back(a.mul({a.constant(jq(target[j])),coordinate}));
    }
    Id bit_address=a.add({a.mul({a.constant(8),a.add(terms)}),a.constant(lo%8)});
    ++a.stats["packed_slice_layout_mappings"];
    return PackedDecodePass::Bits{mapped_name(m,"allocations",read.tag),bit_address,count};
  }
  Id normalize_packed_decoder(Id result) {
    return packed_decoders.normalize(result,[&](Id source,unsigned lo,unsigned count) {
      return canonical_packed_slice(source,lo,count);
    });
  }
  Id typed_integer_fact(Id value, const std::string &type) {
    // Preserve an existing declared integer type at its producer; this is
    // not inferred from a numeric range and adds no input assumption.
    if (!floating(type) && type != "pred" &&
        (type.starts_with("s") || type.starts_with("u") || type.starts_with("b"))) {
      packed_decoders.known_integers.insert(value);
      fp4_boundaries.known_integers.insert(value);
    }
    return value;
  }
  Id typed_read(const std::string &raw, Id offset, const std::string &t, bool require_valid = false) {
    if(t=="e8m0" || t=="ue8m0")return typed_e8m0_input(raw,offset);
    if (t == "ue4m3") return typed_ue4m3_input(raw, offset);
    std::string alloc = mapped_name(m, "allocations", raw);
    if (!m.contains("layouts") || !m["layouts"].contains(raw))
      throw Unsupported("missing input layout " + raw);
    const auto &l = m["layouts"][raw];
    Id off = a.sub(offset, a.constant(jq(l.value("source_offset", J(0)))));
    unsigned bytes = l["element_bytes"];
    if (width(t) != 8 * bytes)
      throw Unsupported(
          "typed input interpretation crosses logical storage elements: " +
          alloc + " " + t);
    // An exact digit certificate is applied before extent and typed-coordinate
    // construction. All enclosing strides must respect the last-axis span;
    // packed lanes arrive here separately with their exact byte displacement.
    if (floating(t) && bytes == 2 && xor_coordinates.eligible(off) && !l["shape"].empty() &&
        jq(l["source_strides_bytes"].back()) == Q(bytes)) {
      Q last = jq(l["shape"].back());
      if (last == 64 || last == 128) {
        Z span = Z(bytes) * last.numerator();
        bool aligned = true;
        for (size_t j = 0; j + 1 < l["shape"].size(); ++j)
          if (modz(jq(l["source_strides_bytes"][j]).numerator(), span) != 0) aligned = false;
        if (aligned) off = xor_coordinates.address(off, bytes, last == 64 ? 6 : 7);
      }
    }
    Id remainder = a.imod(off, bytes);
    if (remainder != a.zero)
      throw Unsupported("Could not prove that the input read is element-aligned",
                        "input_alignment_not_proven",
                        {{"allocation",alloc},{"element_bytes",bytes},
                         {"offset_expression",a.render(off,3)},{"remainder_expression",a.render(remainder,3)}});
    Z extent = bytes;
    for (size_t j = 0; j < l["shape"].size(); ++j)
      extent += (jq(l["shape"][j]).numerator() - 1) *
                jq(l["source_strides_bytes"][j]).numerator();
    auto b = a.nodes[off].bounds;
    bool valid = b && b->first >= 0 &&
        b->second + Q(std::max(1u, width(t) / 8)) <= Q(extent);
    if (!valid) {
      if(require_valid || l.value("target_strides_bytes",l["source_strides_bytes"]) != l["source_strides_bytes"])
        throw Unsupported("input offset range not established for layout/constraint: " + alloc);
      unresolved_accesses.emplace(alloc,offset);
      ++a.stats["symbolic_byte_reads_without_extent_proof"];
      Id canonical=a.add({off,a.constant(jq(l.value("target_offset",J(0))))});
      Bounds storage;
      if(t.starts_with("s") || t.starts_with("u")) {
        unsigned bits=width(t);
        storage=t.starts_with("s")?Bounds(std::pair(Q(-pow2(bits-1)),Q(pow2(bits-1)-1))):Bounds(std::pair(Q(0),Q(pow2(bits)-1)));
      }
      // A separate interpretation tag prevents collisions with coordinate reads.
      // Object contracts deliberately do not apply outside established coverage.
      return typed_integer_fact(a.read(alloc,{canonical},(floating(t)?"real:byte-address:":"byte-address:")+t,storage),t);
    }
    std::vector<Id> coords;
    for (size_t j = 0; j < l["shape"].size(); ++j)
      coords.push_back(
          a.imod(a.idiv(off, jq(l["source_strides_bytes"][j]).numerator()),
                 jq(l["shape"][j]).numerator()));
    if (floating(t) && !ideal)
      throw Unsupported("non-ideal numeric input");
    Bounds bound = ranges.contains(alloc) ? ranges[alloc] : Bounds{};
    if (!bound && !floating(t) && (t.starts_with("s") || t.starts_with("u"))) {
      unsigned bits = width(t);
      bound = t.starts_with("s") ? Bounds(std::pair(Q(-pow2(bits-1)), Q(pow2(bits-1)-1)))
                                  : Bounds(std::pair(Q(0), Q(pow2(bits)-1)));
    }
    if (contract) if (const auto *rule = contract->find(alloc,t)) {
      Bounds effective=rule->range;
      if(!effective && bound && (rule->lower || rule->upper))
        effective=std::pair(rule->lower.value_or(bound->first),rule->upper.value_or(bound->second));
      if (effective) {
        auto incoming=*effective;
        if (bound) incoming={std::max(bound->first,incoming.first),std::min(bound->second,incoming.second)};
        if(incoming.first>incoming.second) throw Unsupported("conflicting input ranges: "+alloc);
        bound=incoming;
      }
      consumed_contracts.emplace(alloc,t);
      contract->applied.emplace(side,alloc,t);
    }
    // The optimized artifact already contains its saved specialization. Bind
    // reference reads only after the ordinary identity, storage type, bounds
    // and coordinate checks above; never specialize by expression node ID.
    if(context && side=="left")
      if(auto fixed=context->fixed_read(a,alloc,t,coords)) {
        if(bound && (a.nodes[*fixed].data[0]<bound->first || a.nodes[*fixed].data[0]>bound->second))
          throw Unsupported("fixed input conflicts with mapped input range: "+alloc);
        return *fixed;
      }
    return typed_integer_fact(a.read(alloc, coords, floating(t) ? "real:" + t : t, bound),t);
  }
  std::optional<Id> reconstruct(const std::vector<Id> &xs) {
    if (xs.size() != 2)
      return {};
    for (unsigned j = 0; j < 2; ++j) {
      const auto &lo = a.nodes[xs[j]];
      const auto &hi = a.nodes[xs[1 - j]];
      if (lo.op != "extractbits" || lo.data != std::vector<Q>{Q(0), Q(32)} ||
          hi.op != "mul" || hi.data != std::vector<Q>{Q(pow2(32))} ||
          hi.args.size() != 1)
        continue;
      const auto &h = a.nodes[hi.args[0]];
      if (h.op == "extractbits" && h.data == std::vector<Q>{Q(32), Q(32)} &&
          h.args == lo.args) {
        ++a.stats["exact_word_reconstruction"];
        return unwrap(word(lo.args[0], 64), 64);
      }
    }
    return {};
  }
  Id mask32(Id x, uint32_t mask) {
    std::vector<Id> pieces;
    for (unsigned lo = 0; lo < 32;) {
      if (!((mask >> lo) & 1u)) { ++lo; continue; }
      unsigned hi = lo + 1;
      while (hi < 32 && ((mask >> hi) & 1u)) ++hi;
      pieces.push_back(a.mul({a.constant(Q(pow2(lo))),
          a.imod(a.idiv(x, pow2(lo)), pow2(hi - lo))}));
      lo = hi;
    }
    return a.add(pieces);
  }
  Id byte_permute(Id x, Id y, unsigned control) {
    std::vector<Id> bytes;
    for (unsigned i = 0; i < 4; ++i) {
      unsigned selector = (control >> (4 * i)) & 15;
      unsigned lane = selector & 7;
      Id source = lane < 4 ? x : y;
      Id byte = a.imod(a.idiv(source, pow2(8 * (lane % 4))), Z(256));
      if (selector & 8) byte = a.mul({a.constant(255), a.idiv(byte, Z(128))});
      bytes.push_back(a.mul({a.constant(Q(pow2(8 * i))), byte}));
    }
    return a.add(bytes);
  }
  Id logic3_constants(Id x, uint32_t b, uint32_t c, unsigned lut) {
    uint32_t positive = 0, negative = 0, ones = 0;
    for (unsigned i = 0; i < 32; ++i) {
      unsigned index = (((b >> i) & 1u) << 1) | ((c >> i) & 1u);
      bool low = (lut >> index) & 1u, high = (lut >> (index + 4)) & 1u;
      if (low && high) ones |= uint32_t(1) << i;
      else if (high) positive |= uint32_t(1) << i;
      else if (low) negative |= uint32_t(1) << i;
    }
    return a.add({a.constant(Q(Z(ones) + Z(negative))), mask32(x, positive),
                  a.neg(mask32(x, negative))});
  }
  // A demanded integer quotient may discard a bounded low radix digit.
  // Keep the proof at the typed import boundary: general Arena arithmetic and
  // numerical real division do not acquire another address search.
  Id integer_index_quotient(Id dividend, const Z &divisor,
                            unsigned work_limit = PackedDecodePass::max_nodes) {
    unsigned budget = std::min(work_limit, PackedDecodePass::max_nodes);
    auto reduce_once = [&](Id operand, const Z &denominator) -> std::optional<Id> {
      const auto &node = a.nodes[operand];
      if (!budget || node.op != "add" || node.args.size() > 8 || denominator <= 1 ||
          (denominator & (denominator - 1)) != 0) return {};
      auto result = packed_decoders.mixed_radix_quotient(operand, denominator, budget);
      if (result) ++a.stats["input_index_radix_quotients"];
      return result;
    };
    auto first = reduce_once(dividend, divisor);
    Id result = first ? *first : a.idiv(dividend, divisor);
    // Discarding a lane digit can expose another short quotient, such as
    // floor((8*k+head)/1024) after the byte lane was removed. Follow only the
    // returned root or direct quotient terms of a small Add, for four rounds
    // total. Every probe shares the original 32-unit transport budget.
    for (unsigned round = 1; round < 4 && budget; ++round) {
      const auto &node = a.nodes[result];
      auto follow = [&](Id id) -> Id {
        const auto &part = a.nodes[id];
        if (part.op != "idiv" || part.args.size() != 1 || part.data.size() != 1 ||
            part.data[0].denominator() != 1) return id;
        Z denominator = part.data[0].numerator();
        // Do not repeat an initial miss on an unchanged quotient.
        if (!first && part.args[0] == dividend && denominator == divisor) return id;
        auto next = reduce_once(part.args[0], denominator);
        if (!next || *next == id) return id;
        ++a.stats["input_index_radix_followups"];
        return *next;
      };
      Id next = result;
      if (node.op == "idiv") next = follow(result);
      else if (node.op == "add" && node.args.size() <= 8 && node.args.size() == node.data.size()) {
        auto args = node.args; auto coefficients = node.data;
        bool changed = false;
        for (Id &argument : args) {
          Id rewritten = follow(argument);
          changed |= rewritten != argument;
          argument = rewritten;
        }
        if (changed) {
          std::vector<Id> terms;
          for (size_t j = 0; j < args.size(); ++j)
            terms.push_back(a.mul({a.constant(coefficients[j]), args[j]}));
          next = a.add(std::move(terms));
        }
      }
      if (next == result) break;
      result = next;
    }
    return result;
  }

  // The same typed compiler exp idiom can be distributed over an additive
  // exponent. Keep unscaled terms in base two: Exp2(L*x + r) is idealized as
  // Exp(x)*Exp2(r), not Exp(x+r/L). Only inspect this bounded raw arithmetic
  // skeleton; imported leaves (including large reductions) stay opaque.
  std::optional<Id> additive_compiler_exp(const RawNode &n, Id exponent,
                                        const Env &ctx) {
    if (!ideal || n.kind != "Operation" || n.name != "Exp2" || n.type != "f32" ||
        (n.attributes != "" && n.attributes != "ideal_arithmetic") ||
        n.operands.size() != 1 || !n.operands[0].conversion.empty() ||
        !a.nodes[exponent].facts.finite) return {};
    const auto &root = ns[n.operands[0].node];
    if (root.kind != "Operation" || (root.name != "Add" && root.name != "Subtract")) return {};
    static const Q log2e(12102203,8388608);
    struct Term { Edge edge; int sign; };
    std::vector<Term> todo{{n.operands[0],1}};
    std::vector<Id> natural, binary;
    size_t work = 0;
    auto imported = [&](const Edge &e) -> std::optional<Id> {
      auto it = cache.find(key(e.node,ctx));
      if (it == cache.end()) return {};
      return edge(it->second,e);
    };
    while (!todo.empty()) {
      if (++work > 32) return {};
      auto term = std::move(todo.back()); todo.pop_back();
      const auto &raw = ns[term.edge.node];
      const bool arithmetic = term.edge.conversion.empty() && raw.kind == "Operation" &&
        raw.type == "f32" && (raw.attributes.empty() || raw.attributes == "ideal_arithmetic");
      if (arithmetic && raw.operands.size() == 2 && (raw.name == "Add" || raw.name == "Subtract")) {
        todo.push_back({raw.operands[1],term.sign*(raw.name == "Subtract" ? -1 : 1)});
        todo.push_back({raw.operands[0],term.sign}); continue;
      }
      if (arithmetic && raw.name == "Negate" && raw.operands.size() == 1) {
        todo.push_back({raw.operands[0],-term.sign}); continue;
      }
      bool matched = false;
      if (arithmetic && raw.name == "Multiply" && raw.operands.size() == 2) {
        for (unsigned j = 0; j < 2; ++j) {
          const auto &ce = raw.operands[j]; const auto &c = ns[ce.node];
          if (!ce.conversion.empty() || c.kind != "Constant" || c.type != "f32") continue;
          const auto constant = imported(ce);
          if (!constant) return {};
          const auto literal = a.value(*constant);
          if (!literal) continue; // NaN/Inf are not rational compiler constants.
          const Q coefficient = *literal;
          if (coefficient != log2e && coefficient != -log2e) continue;
          const auto &ve = raw.operands[1-j];
          if ((ve.conversion.empty() ? ns[ve.node].type : ve.converted_type) != "f32") continue;
          auto value = imported(ve);
          if (!value || !a.nodes[*value].facts.finite) return {};
          natural.push_back(term.sign*(coefficient == log2e ? 1 : -1) == 1 ? *value : a.neg(*value));
          matched = true; break;
        }
      }
      if (!matched) {
        auto value = imported(term.edge);
        if (!value || !a.nodes[*value].facts.finite) return {};
        binary.push_back(term.sign == 1 ? *value : a.neg(*value));
      }
    }
    if (natural.empty()) return {};
    Id argument = a.add(std::move(natural)), residual = a.add(std::move(binary));
    ++a.stats["ideal_single_constant_exp_sequences"];
    ++a.stats["ideal_additive_compiler_exp_sequences"];
    return a.mul({a.exp(argument),a.exp2(residual)});
  }

  Id make(const RawNode &n, const std::vector<Id> &x, const Env &ctx,
          const Env &body) {
    const std::string &k = n.kind, &t = n.type, &attr = n.attributes;
    std::string name = n.name;
    if (k == "Constant")
      return literal(name, t);
    if (k == "Index") {
      Id index=ctx.at(n.level);
      packed_decoders.known_integers.insert(index);
      fp4_boundaries.known_integers.insert(index);
      return index;
    }
    if (k == "InputParameter") {
      if (m.contains("parameter_constants") &&
          m["parameter_constants"].contains(name)) {
        const auto &c = m["parameter_constants"][name];
        return literal(c["literal"], c["type"]);
      }
      return typed_integer_fact(a.intern("atom", {}, {}, mapped_name(m, "parameters", name),
                      {true, false, false}),t);
    }
    if (k == "InitialCell" || k == "InputValue") {
      if (x.size() != 1)
        throw Unsupported("input read expects one byte-offset operand");
      if(t=="e8m0" || t=="ue8m0")return typed_e8m0_input(name,x[0]);
      if (t == "ue4m3")
        return typed_ue4m3_input(name, x[0]);
      if ((t.starts_with("b") && !t.starts_with("bf")) || t.starts_with("u"))
        return a.intern("readbits", x, {Q(width(t))}, name, {true, false, true},
                        std::pair(Q(0), Q(pow2(width(t)) - 1)));
      return typed_read(name, x[0], t);
    }
    if (k == "Fold") {
      const auto &f = n;
      Id var = body.at(f.level);
      Q count = Q(f.count);
      const std::string &reducer = f.reducer;
      return a.intern("fold", x, {Q(var), count}, reducer,
                      a.fold_facts(x[0],x[1],count,reducer=="Max",var));
    }
    if (k != "Operation")
      throw Unsupported("node kind " + k);
    if (attr.find("saturat") != attr.npos)
      throw Unsupported("arithmetic saturation is not ideal-real identity");
    bool integer = !floating(t) && t != "pred";
    if(name=="BitNot") {
      if(!attr.empty() || x.size()!=1)throw Unsupported("BitNot attributes/arity; type="+t);
      if(t=="pred") {
        const auto bounds=a.nodes[x[0]].bounds;
        if(!bounds || bounds->first<0 || bounds->second>1)
          throw Unsupported("predicate NOT operand outside Boolean range");
        return a.sub(a.one,x[0]);
      }
      if(!integer || t.empty() || (t[0]!='s' && t[0]!='u' && t[0]!='b'))
        throw Unsupported("BitNot requires integer/bit type; type="+t);
      const unsigned w=width(t);
      const auto result=a.sub(a.constant(Q(pow2(w)-1)),word(x[0],w));
      return t.starts_with("s")?cast(result,"u"+std::to_string(w),t):result;
    }
    if (integer && (name == "Add" || name == "Subtract" || name == "Multiply" ||
                    name == "PtxMultiplyAdd" || name == "ShiftLeft")) {
      unsigned w = width(t);
      std::vector<Id> xx;
      for (Id z : x)
        xx.push_back(unwrap(z, w));
      Id y;
      if (name == "Add") {
        auto r = reconstruct(xx);
        y = r ? *r : a.add(xx);
      } else if (name == "Subtract")
        y = a.sub(xx.at(0), xx.at(1));
      else if (name == "Multiply")
        y = a.mul(xx);
      else if (name == "PtxMultiplyAdd") {
        if (attr.find(".lo.") == attr.npos)
          throw Unsupported("multiply-add mode " + attr);
        y = a.add({a.mul({xx.at(0), xx.at(1)}), xx.at(2)});
      } else {
        auto s = a.value(xx.at(1));
        if (!s || s->denominator() != 1 || *s < 0 || *s >= Q(w))
          throw Unsupported("symbolic/out-of-range shift");
        y = a.mul({xx[0],
                   a.constant(Q(pow2(s->numerator().convert_to<unsigned>())))});
      }
      if (t.starts_with("s")) return a.signed_word(y, w);
      return word(y, w);
    }
    if (name == "PtxBytePermute" && t == "b32" &&
        attr == "source=b32.b32 source_type=b32") {
      auto control = a.value(x.at(2));
      if (!control || control->denominator() != 1 || *control < 0 || *control > 65535)
        throw Unsupported("byte permute needs constant 16-bit selector");
      return byte_permute(word(x.at(0),32), word(x.at(1),32), control->numerator().convert_to<unsigned>());
    }
    if (name == "PtxLogic3" && t == "b32" &&
        attr == "source=b32.b32 source_type=b32") {
      auto b = a.value(x.at(1)), c = a.value(x.at(2)), lut = a.value(x.at(3));
      if (!b || !c || !lut || b->denominator()!=1 || c->denominator()!=1 ||
          lut->denominator()!=1 || *b<0 || *c<0 || *b>=Q(pow2(32)) || *c>=Q(pow2(32)) || *lut<0 || *lut>255)
        throw Unsupported("logic3 requires two constant masks and 8-bit LUT");
      return logic3_constants(word(x.at(0),32), b->numerator().convert_to<uint32_t>(),
                              c->numerator().convert_to<uint32_t>(), lut->numerator().convert_to<unsigned>());
    }
    if (name == "ExtractBits") {
      auto off = a.value(x.at(1));
      std::smatch m;
      static const std::regex re("width=(\\d+)");
      if (!std::regex_search(attr, m, re))
        throw Unsupported("extract bits metadata");
      unsigned w = std::stoul(m[1]);
      if (w == 0 || w > 256) throw Unsupported("extract bits width");
      if (!off) {
        auto bounds = a.nodes[x.at(1)].bounds;
        if(n.operands.size()!=2)throw Unsupported("dynamic extract bits requires source type");
        const auto &source_edge=n.operands[0], &offset_edge=n.operands[1];
        std::string source_type=source_edge.converted_type.empty()?ns.at(source_edge.node).type:source_edge.converted_type;
        std::string offset_type=offset_edge.converted_type.empty()?ns.at(offset_edge.node).type:offset_edge.converted_type;
        unsigned source_width=width(source_type),result_width=width(t);
        if(floating(offset_type) || !bounds || bounds->first<0 || bounds->second+Q(w)>Q(source_width) ||
           result_width<w || (floating(t) && result_width!=w))
          throw Unsupported("dynamic extract bits requires in-range typed coordinate","expression_form_unsupported",
              J{{"source_type",source_type},{"offset_type",offset_type},{"result_type",t},{"slice_width",w},
                {"offset_lower",bounds?str(bounds->first):"unknown"},{"offset_upper",bounds?str(bounds->second):"unknown"}});
        Id storage=x[0];
        if(floating(source_type))storage=cast(storage,source_type,"b"+std::to_string(source_width),true);
        storage=word(storage,source_width);
        Id bits=a.bit_slice(a.word_shr(storage,x[1],source_width,false),0,w);
        ++a.stats["parameterized_bit_extracts"];
        if(storage==a.zero)++a.stats["zero_dynamic_bit_extracts"];
        return floating(t)||t.starts_with("s")?
            cast(bits,(floating(t)?"b":"u")+std::to_string(result_width),t,true):bits;
      }
      if (off->denominator() != 1 || *off < 0 || *off>Q(UINT_MAX-w))
        throw Unsupported("extract bits offset");
      const auto result_width=width(t);
      if(result_width<w || (floating(t) && result_width!=w))
        throw Unsupported("extract bits result type width mismatch");
      // Projection's typed ExtractBits includes the interpretation of the
      // selected storage field. In particular bf16 is not the unsigned value
      // of its encoding. Reuse the same exact slice/cast path as type edges.
      unsigned source_width=256;
      auto storage=x[0];
      if(!n.operands.empty()) {
        const auto& e=n.operands[0];
        const auto source_type=e.converted_type.empty()?ns.at(e.node).type:e.converted_type;
        source_width=width(source_type);
        if(floating(source_type))storage=cast(storage,source_type,"b"+std::to_string(source_width),true);
      }
      if(floating(t) || t.starts_with("s")) {
        if(*off+Q(w)>Q(source_width))throw Unsupported("typed extract exceeds source width");
        const auto masked=a.nodes[storage];
        if(floating(t) && (t=="f16" || t=="bf16" || t=="f32" || t=="f64") &&
           masked.op=="constant_bit_mask" && masked.args.size()==1 && masked.data.size()==2 &&
           *off+Q(w)<=masked.data[1]) {
          const auto shift=off->numerator().convert_to<unsigned>();
          Z lane=(masked.data[0].numerator()>>shift)&(pow2(w)-1);
          if(lane==pow2(w-1)-1) {
            auto selected=a.intern("extractbits",{masked.args[0]},{*off,Q(w)},"",
                {true,false,true},std::pair(Q(0),Q(pow2(w)-1)));
            if(auto value=initial_typed_bits(selected,t)) {
              ++a.stats["typed_input_sign_mask_abs"];
              return a.absolute(*value);
            }
          }
        }
        auto bits=word(a.idiv(storage,pow2(off->numerator().convert_to<unsigned>())),w);
        return cast(bits,(floating(t)?"b":"u")+std::to_string(result_width),t,true);
      }
      return a.bit_slice(storage,off->numerator().convert_to<unsigned>(),w);
    }
    if (name == "ShiftRight") {
      unsigned w = width(t);
      if(!integer || x.size()!=2)throw Unsupported("right shift operand type");
      Id source=word(x[0],w),shift=word(x[1],32);
      bool sign=t.starts_with("s");
      if(sign)source=cast(source,"u"+std::to_string(w),t);
      return a.word_shr(source,shift,w,sign);
    }
    if (name == "BitOr" && integer && x.size() == 2) {
      const unsigned w=width(t);
      Id lhs=word(x[0],w), rhs=word(x[1],w);
      const auto finish=[&](Id value) {return t.starts_with("s")?
          cast(value,"u"+std::to_string(w),t):value;};
      return finish(a.word_or(lhs,rhs,w));
    }
    if (name == "BitXor" && integer) {
      unsigned w = width(t);
      Id result = a.bitxor(word(x.at(0), w), word(x.at(1), w));
      return t.starts_with("s") ? cast(result, "u" + std::to_string(w), t)
                                : result;
    }
    if (name == "Absolute") {
      if (!floating(t))
        throw Unsupported("integer Absolute adapter");
      return a.absolute(x.at(0));
    }
    if (name == "BitAnd" && t == "pred") {
      if(x.size()!=2) throw Unsupported("predicate AND arity");
      for(Id v:x) if(!a.nodes[v].bounds || a.nodes[v].bounds->first<0 || a.nodes[v].bounds->second>1)
        throw Unsupported("predicate AND operand outside Boolean range");
      return a.mul(x);
    }
    if (name == "BitAnd") {
      auto m = a.value(x.at(1));
      if (!m || m->denominator() != 1)
        throw Unsupported("nonconstant mask");
      Z mask = m->numerator();
      if (mask == 0)
        return a.zero;
      if (mask < 0)
        throw Unsupported("negative mask");
      Z low = mask & -mask, span = mask / low + 1;
      if((a.nodes[x[0]].op=="word_shr" || a.nodes[x[0]].op=="word_or") && mask<pow2(width(t))) {
        std::vector<Id> bits;
        for(unsigned b=0;b<width(t);++b)if((mask&pow2(b))!=0)
          bits.push_back(a.mul({a.constant(Q(pow2(b))),a.bit_slice(x[0],b,1)}));
        return a.add(bits);
      }
      if ((span & (span - 1)) != 0) {
        if(!integer || mask>=pow2(width(t)))throw Unsupported("mask exceeds integer width");
        return a.intern("constant_bit_mask",{word(x[0],width(t))},{Q(mask),Q(width(t))},"",
            {true,false,true},std::pair(Q(0),Q(mask)));
      }
      return a.mul({a.constant(Q(low)),
                    a.imod(a.idiv(unwrap(x[0], width(t)), low), span)});
    }
    if ((name == "FloorDivide" || name == "Modulo") && integer) {
      auto divisor = a.value(x.at(1));
      if (!divisor || divisor->denominator() != 1 || *divisor <= 0)
        throw Unsupported("FloorDivide requires positive constant integer divisor");
      return name == "FloorDivide" ? integer_index_quotient(x.at(0), divisor->numerator())
                                   : a.imod(x.at(0), divisor->numerator());
    }
    if (integer && (name == "Divide" || name == "Remainder")) {
      auto d = a.value(x.at(1));
      if (!d || d->denominator() != 1 || *d <= 0)
        throw Unsupported("dynamic index divisor");
      auto b = a.nodes[x[0]].bounds;
      if (t.starts_with("s") && (!b || b->first < 0))
        throw Unsupported("signed truncation not floor");
      return name == "Divide" ? integer_index_quotient(x[0], d->numerator())
                              : a.imod(x[0], d->numerator());
    }
    if (name == "Add")
      return a.add(x);
    if (name == "Subtract")
      return a.sub(x.at(0), x.at(1));
    if (name == "Multiply") {
      if (auto lse = compiler_lse(n, ctx)) return *lse;
      return x.size() == 2 ? numerical_product(x[0], x[1]) : a.mul(x);
    }
    if (name == "Fma")
      return a.add({numerical_product(x.at(0), x.at(1)), x.at(2)});
    if (name == "Negate")
      return a.neg(x.at(0));
    if (name == "Exp")
      return a.neginf(x.at(0)) ? a.zero : a.exp(x[0]);
    if (name == "Exp2") {
      if (a.neginf(x.at(0))) return a.zero;
      if (auto expanded = additive_compiler_exp(n,x.at(0),ctx)) return *expanded;
      // Under ideal numerical semantics, the compiler's single-constant
      // exp lowering is an Exp idiom. Match the exact binary32 log2(e)
      // coefficient, not arbitrary nearby constants or unrelated arithmetic.
      const auto &exponent = a.nodes[x.at(0)];
      static const Q compiler_log2e(12102203,8388608);
      if (ideal && exponent.facts.finite && n.operands.size()==1 && n.operands[0].conversion.empty()) {
        // Preserve recognition when other literal factors have already merged
        // into the canonical coefficient (e.g. -5*log2(e)). Walk only a bounded
        // raw multiplication tree and require the explicit compiler constant.
        std::vector<Id> todo{n.operands[0].node};unsigned work=0;int sign=0;
        while(!todo.empty() && work++<16 && !sign) {
          Id id=todo.back();todo.pop_back();const auto &raw=ns[id];
          if(raw.kind=="Constant" && raw.type=="f32") {
            Q value=decimal(raw.name);
            if(value==compiler_log2e)sign=1;
            else if(value==-compiler_log2e)sign=-1;
          } else if(raw.kind=="Operation" && raw.name=="Multiply" && raw.type=="f32" &&
                    (raw.attributes.empty() || raw.attributes=="ideal_arithmetic")) {
            for(const auto &edge:raw.operands) if(edge.conversion.empty())todo.push_back(edge.node);
          }
        }
        if(sign) {
          ++a.stats["ideal_single_constant_exp_sequences"];
          // A scalar Multiply can distribute log2(e) into an Add before a
          // later Multiply introduces other factors. Cancel that exact common
          // scalar locally, under the same explicit raw compiler witness.
          // Bound both scans; leave wider or nonuniform sums on the old path.
          if (exponent.op == "mul" && exponent.data.size() == 1 && exponent.args.size() <= 8) {
            for (size_t j = 0; j < exponent.args.size(); ++j) {
              const auto &sum = a.nodes[exponent.args[j]];
              if (sum.op != "add" || sum.args.size() < 2 || sum.args.size() > 4 ||
                  sum.data.size() != sum.args.size()) continue;
              Q common = sum.data[0];
              if ((common != compiler_log2e && common != -compiler_log2e) ||
                  !std::all_of(sum.data.begin(), sum.data.end(),
                               [&](const Q &coefficient) { return coefficient == common; })) continue;
              // Copy before arena construction can invalidate node references.
              auto factors = exponent.args, addends = sum.args;
              Q scalar = exponent.data[0] * common / compiler_log2e;
              factors[j] = a.add(std::move(addends));
              factors.push_back(a.constant(scalar));
              ++a.stats["ideal_distributed_log2e_cancellations"];
              return a.exp(a.mul(std::move(factors)));
            }
          }
          return a.exp(a.mul({a.constant(Q(1)/compiler_log2e),x[0]}));
        }
      }
      if (ideal && exponent.op == "mul" && exponent.facts.finite &&
          (exponent.data[0] == compiler_log2e || exponent.data[0] == -compiler_log2e)) {
        auto factors = exponent.args;
        bool negative = exponent.data[0] < 0;
        Id argument = a.mul(factors);
        if (negative) argument = a.neg(argument);
        ++a.stats["ideal_single_constant_exp_sequences"];
        return a.exp(argument);
      }
      return a.exp2(x[0]);
    }
    if (name == "Reciprocal")
      return a.recip(x.at(0));
    if (name == "Divide")
      return a.divide(x.at(0), x.at(1));
    if (name == "Select")
      return a.select(x.at(0), x.at(1), x.at(2), !floating(n.type));
    if (name == "Maximum")
      return a.maximum(x);
    if (name == "Minimum") {
      // Ideal-real minimum shares the maximum normal form.
      std::vector<Id> negatives;
      negatives.reserve(x.size());
      for (Id operand : x)
        negatives.push_back(a.neg(operand));
      return a.neg(a.maximum(std::move(negatives)));
    }
    if (name == "Compare") {
      std::smatch m;
      static const std::regex re("predicate=(\\w+)");
      if (!std::regex_search(attr, m, re))
        throw Unsupported("missing predicate metadata");
      return a.compare(m[1], x.at(0), x.at(1));
    }
    static const std::map<std::string, std::string> cmp{
        {"Less", "lt"},         {"LessEqual", "le"}, {"Greater", "gt"},
        {"GreaterEqual", "ge"}, {"Equal", "eq"},     {"NotEqual", "ne"}};
    if (cmp.contains(name))
      return a.compare(cmp.at(name), x.at(0), x.at(1));
    if (name == "Log" || name == "Power") {
      if (!ideal || !floating(t))
        throw Unsupported("real Log/Power requires ideal_real floating expression");
      return name == "Log" ? a.logarithm(x.at(0)) : a.power(x.at(0), x.at(1));
    }
    if (name == "PtxLog2") {
      if (!ideal || !floating(t)) throw Unsupported("PtxLog2 requires ideal_real floating expression");
      return a.logarithm(x.at(0), true);
    }
    if (name == "PtxSqrt") {
      if (!ideal) throw Unsupported("PtxSqrt requires ideal_real");
      name = "Sqrt";
    }
    if (name == "Rsqrt" || name == "PtxReciprocalSqrt")
      name = "ReciprocalSqrt";
    if (name == "PtxSin")
      name = "Sin";
    if (name == "PtxCos")
      name = "Cos";
    if (name == "Sqrt" || name == "ReciprocalSqrt" || name == "Sin" ||
        name == "Cos" || name == "Exp" || name == "Tanh") {
      bool finite = std::all_of(x.begin(), x.end(),
                                [&](Id z) { return a.nodes[z].facts.finite; });
      if (name == "Sqrt") finite &= a.nodes[x[0]].facts.nn;
      if (name == "ReciprocalSqrt") finite &= a.nodes[x[0]].facts.pos;
      bool pos = finite && (name == "ReciprocalSqrt" || name == "Exp" ||
                           (name == "Sqrt" && a.nodes[x[0]].facts.pos));
      bool nn = pos || (finite && name == "Sqrt");
      std::transform(name.begin(), name.end(), name.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      return a.intern(name, x, {}, "", {finite, pos, nn});
    }
    throw Unsupported("operation " + name + " [" + attr + "]; type="+t+
        "; arity="+std::to_string(x.size()));
  }
};
} // namespace rg
