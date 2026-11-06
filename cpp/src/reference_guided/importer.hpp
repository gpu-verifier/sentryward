#pragma once
#include "core.hpp"
#include "alpha_identity.hpp"
#include "input.hpp"
#include "equivalence_context.hpp"
#include "floating_idiom_pass.hpp"
#include "packed_decode.hpp"
#include "sentryward/equivalence/float_from_chars.h"
#include "sentryward/equivalence/ieee_bits.h"
#include "sentryward/scalar_fold/graph.h"
#include "xor_coordinate.hpp"
#include <memory>
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
// The one key-hash formula of sibling_binders and ImportMemo; a vector key
// converts to the span implicitly.
struct VecHash {
  size_t operator()(std::span<const Id> v) const {
    size_t h = 0;
    for (Id x : v)
      mix(h, x);
    return h;
  }
};
// The importer's (raw id, projected free environment) -> id memo. Each key is
// stored once as [size, value, ids...] in fixed-size word chunks that never
// move, and a slot keeps VecHash's hash and a pointer to the stored key in a
// power-of-two table probed linearly at load <= 3/4 from Arena::index_home.
// No node per entry, no key vector per probe and no copy of stored keys on
// growth. Never erased or iterated; emplace keeps an existing value, as
// unordered_map::emplace, and at() keeps unordered_map::at's exception.
class ImportMemo {
  struct Slot {
    size_t hash;
    const Id *key;
  };
  static constexpr size_t chunk_words = size_t(1) << 16;
  std::vector<Slot> slots;
  std::vector<std::unique_ptr<Id[]>> chunks;
  Id *cursor = nullptr, *limit = nullptr;
  size_t entries = 0;
  unsigned shift = 64;
  const Id *find(std::span<const Id> k, size_t h) const {
    if (slots.empty())
      return nullptr;
    for (size_t m = slots.size() - 1, i = Arena::index_home(h, shift);; i = (i + 1) & m) {
      const Slot &s = slots[i];
      if (!s.key)
        return nullptr;
      if (s.hash == h && s.key[0] == k.size() && std::equal(k.begin(), k.end(), s.key + 2))
        return s.key + 1;
    }
  }
  // Grow before writing: a failed allocation leaves the stored entries intact.
  void reserve_slot() {
    if (4 * (entries + 1) <= 3 * slots.size())
      return;
    const size_t n = slots.empty() ? 256 : 2 * slots.size();
    std::vector<Slot> grown(n, Slot{0, nullptr});
    const unsigned grown_shift = 64 - unsigned(std::countr_zero(n));
    for (const auto &x : slots)
      if (x.key)
        place(grown, grown_shift, x);
    slots.swap(grown);
    shift = grown_shift;
  }
  static void place(std::vector<Slot> &v, unsigned home_shift, Slot s) noexcept {
    for (size_t m = v.size() - 1, i = Arena::index_home(s.hash, home_shift);; i = (i + 1) & m)
      if (!v[i].key) {
        v[i] = s;
        return;
      }
  }
public:
  // Slots, cursor and limit point into chunks, so the memo is neither copied
  // nor moved (the deleted copy also suppresses the implicit move).
  ImportMemo() = default;
  ImportMemo(const ImportMemo &) = delete;
  ImportMemo &operator=(const ImportMemo &) = delete;
  // A returned value pointer stays valid: stored entries never move or change.
  const Id *find(std::span<const Id> k) const { return find(k, VecHash{}(k)); }
  bool contains(std::span<const Id> k) const { return find(k); }
  Id at(std::span<const Id> k) const {
    if (const Id *v = find(k))
      return *v;
    throw std::out_of_range("unordered_map::at");
  }
  void emplace(std::span<const Id> k, Id value) {
    const size_t h = VecHash{}(k);
    if (find(k, h))
      return;
    reserve_slot();
    const size_t n = k.size() + 2;
    if (size_t(limit - cursor) < n) {
      const size_t words = std::max(chunk_words, n);
      chunks.push_back(std::unique_ptr<Id[]>(new Id[words]));
      cursor = chunks.back().get();
      limit = cursor + words;
    }
    Id *w = cursor;
    cursor += n;
    w[0] = Id(k.size());
    w[1] = value;
    std::copy(k.begin(), k.end(), w + 2);
    place(slots, shift, {h, w});
    ++entries;
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
  // Separate caches and entry points: ordinary arithmetic admits Signed4 and Affine4 decoders, never FP4.
  PackedDecodePass fp4_boundaries;
  std::unordered_set<Id> fp4_reference_boundaries;
  std::unordered_map<Id,Id> typed_fp4_values;
  std::map<std::pair<Id,std::string>,Id> floating_lane_values;
  std::map<std::tuple<Id,std::string,std::string>,Id> select_bitcasts;
  unsigned select_bitcast_depth=0;
  // Bounds the recursion (stack depth) of a bitcast through nested selections; deeper is refused.
  static constexpr unsigned select_bitcast_depth_limit=256;
  // A strided source layout (not a dense permutation) whose axes of extent > 1, in ascending
  // stride order, nest: every stride is a positive multiple of the element size and of the
  // next smaller stride, and no axis overlaps the next (stride >= the inner axis's extent x
  // stride). Then every element offset has exactly one digit expansion: axis j's coordinate
  // is (offset mod the next larger stride) / stride_j, the outermost one's offset / stride;
  // and an offset denotes an element exactly when it is a multiple of the innermost stride and
  // every coordinate is below its extent (typed_read proves both from the offset's bounds).
  // Per layout key: those axes, innermost first. Strides that do not nest but each cover the
  // byte extent of all smaller-stride axes are kept too (dominant_layouts); any other non-dense
  // layout is refused.
  struct NestedAxis { Z stride, count; size_t dim; };
  std::unordered_map<std::string, std::vector<NestedAxis>> nested_layouts;
  // Nested-layout keys whose strides do not divide each other (decoded greedily).
  std::unordered_set<std::string> dominant_layouts;
  std::unordered_set<std::string> packed_dense_layouts;
  std::unordered_set<std::string> packed_identity_layouts;
  std::unordered_map<Id,Id> power_bit_values;
  std::unordered_set<Id> encoded_powers;
  bool ideal;
  // The floating range premise of in_range (assumption floating_values_in_range); set by the comparison.
  bool range_premise = true;
  // The hardware value rule applies to this side (hardware_value); set by the comparison for the PTX
  // side only (ComparisonOptions::kernel_hardware_values).
  bool hardware_values = false;
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
  ImportMemo cache;
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
        bool permutation = true;
        for (auto &[s, n] : dims) {
          if (s != span) permutation = false;
          span *= n;
        }
        if (!permutation) {
          std::vector<NestedAxis> axes;
          for (size_t k = 0; k < l["shape"].size(); ++k) {
            Z n = jq(l["shape"][k]).numerator();
            if (n > 1) axes.push_back({jq(l["source_strides_bytes"][k]).numerator(), n, k});
          }
          std::sort(axes.begin(), axes.end(), [](const NestedAxis &x, const NestedAxis &y) {
            return std::tie(x.stride, x.dim) < std::tie(y.stride, y.dim);
          });
          const Z bytes = jq(l["element_bytes"]).numerator();
          // Strides that do not nest (a larger stride that is no multiple of a smaller one, e.g.
          // [1040, 256, 2] for shape [., 4, 128]) are exact when each stride is at least the byte
          // extent of all smaller-stride axes: offsets then decompose greedily from the largest
          // stride, one element per offset (injective). Padded rows whose stride is a multiple of
          // the inner stride (e.g. [1536, 256, 2]) nest.
          bool nests = true;
          Z inner = bytes;
          for (size_t j = 0; j < axes.size(); ++j) {
            const auto &axis = axes[j];
            if (axis.stride <= 0 || bytes <= 0 || modz(axis.stride, bytes) != 0)
              throw Unsupported("source input layout stride is not a positive multiple of its element size: " + it.key());
            inner += (axis.count - 1) * axis.stride;
            if (j + 1 < axes.size()) {
              nests = nests && modz(axes[j + 1].stride, axis.stride) == 0;
              if (axes[j + 1].stride < inner)
                throw Unsupported(nests ? "source input layout axes overlap: " + it.key()
                                        : "source input layout is not proved injective (a stride below the extent "
                                          "of the smaller-stride axes): " + it.key());
            }
          }
          if (!nests) dominant_layouts.insert(it.key());
          nested_layouts.emplace(it.key(), std::move(axes));
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
  // Fills k with the memo key [i, env[v] for each free level v of i]; the hot
  // import() loop reuses its buffers instead of allocating a key per probe.
  std::span<const Id> key(std::vector<Id> &k, Id i, const Env &env) const {
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
  std::vector<Id> key(Id i, const Env &env) const {
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
  // One binder per (bound level, domain) of a leaf Fold (a Fold whose body
  // holds no Fold) for all of its imports, in every output family. A bound
  // variable's name carries no meaning (alpha-equivalence), and leaf Folds
  // never enclose one another, so reusing the name is exact; with one name, the
  // parts of a Fold that do not depend on a family's outer coordinates (a
  // reference scan's state chain, say) are the same Arena nodes across
  // families instead of a fresh copy per family. The first import takes
  // fold_binder's choice; only its alpha-shared binders are recorded, so
  // nested, shadowed and fallback cases keep their own original binders. A
  // recorded binder is reused only when no enclosing coordinate depends on it
  // and the bound level is not shadowed, the same hygiene as the sibling bank.
  // O(environment) per Fold import.
  std::map<std::pair<Id, uint64_t>, Id> occurrence_binders;
  Id occurrence_binder(Id raw, const Env &env) {
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
  std::string family_type(const J &f) const {
    const auto &body=f.at("body");
    if(body.contains("converted_type"))return body.at("converted_type");
    if(!body.value("conversion","").empty())return "";
    return ns.at(input_id(body.at("node"))).type;
  }
  // A readbits node addresses raw bytes of this side's source layout. The same raw address denotes
  // different elements on the two sides when their source layouts of one shared input differ, so
  // views of such inputs are compared only at canonical (target-layout) addresses.
  // Per raw layout key of this side: whether the other side lays out the same input differently.
  std::unordered_map<std::string,bool> byte_layout_differs;
  std::optional<bool> any_byte_layout_differs;
  // Per Arena node: already scanned for raw views of such inputs (shared by this side's families).
  std::vector<uint8_t> byte_view_scanned;
  bool source_layout_differs(const std::string &raw) {
    if(auto known=byte_layout_differs.find(raw);known!=byte_layout_differs.end())return known->second;
    bool differs=false;
    const auto &other=mapping.at(side=="left"?"right_inputs":"left_inputs");
    const std::string shared=mapped_name(m,"allocations",raw);
    if(m.contains("layouts") && m["layouts"].contains(raw) && other.contains("layouts"))
      for(auto it=other["layouts"].begin();it!=other["layouts"].end();++it)
        if(mapped_name(other,"allocations",it.key())==shared) {
          const auto &mine=m["layouts"][raw],&theirs=it.value();
          differs=differs || mine["source_strides_bytes"]!=theirs["source_strides_bytes"] ||
                  jq(mine.value("source_offset",J(0)))!=jq(theirs.value("source_offset",J(0)));
        }
    byte_layout_differs.emplace(raw,differs);
    return differs;
  }
  // The canonical address of a view of `bytes_read` bytes at raw `address`: target_offset +
  // (offset mod element_bytes) + sum_j target_stride_j * ((offset / stride_j) mod extent_j), with
  // offset = address - source_offset. Exact for a dense permuted source layout when the offset lies
  // in the layout's extent and the view stays inside one element (both proved from bounds); an
  // identity layout only moves the offset. O(rank) Arena terms.
  std::optional<Id> canonical_byte_address(const std::string &raw,Id address,const Z &bytes_read) {
    const auto &l=m["layouts"][raw];
    const auto &target=l.contains("target_strides_bytes")?l["target_strides_bytes"]:l["source_strides_bytes"];
    const Id off=a.sub(address,a.constant(jq(l.value("source_offset",J(0)))));
    const Id target_offset=a.constant(jq(l.value("target_offset",J(0))));
    if(target==l["source_strides_bytes"])return a.add({off,target_offset});
    if(!packed_dense_layouts.contains(raw) || nested_layouts.contains(raw))return {};
    const Z bytes=l["element_bytes"].get<unsigned>();
    Z extent=bytes;
    for(size_t j=0;j<l["shape"].size();++j)
      extent+=(jq(l["shape"][j]).numerator()-1)*jq(l["source_strides_bytes"][j]).numerator();
    const auto bounds=a.nodes[off].bounds;
    if(!bounds || bounds->first<0 || bounds->second+Q(bytes_read)>Q(extent))return {};
    const Id within=a.imod(off,bytes);
    const auto inside=a.nodes[within].bounds;
    if(!inside || inside->first<0 || inside->second+Q(bytes_read)>Q(bytes))return {};
    std::vector<Id> terms{target_offset,within};
    for(size_t j=0;j<l["shape"].size();++j) {
      // A shape-1 axis has coordinate 0; its stride is not checked and may be 0.
      if(jq(l["shape"][j])==1)continue;
      const Id coordinate=a.imod(a.idiv(off,jq(l["source_strides_bytes"][j]).numerator()),
                                 jq(l["shape"][j]).numerator());
      terms.push_back(a.mul({a.constant(jq(target[j])),coordinate}));
    }
    ++a.stats["canonical_byte_view_layout_mappings"];
    return a.add(terms);
  }
  // Whether the other side lays out some input of this side differently. Computed once per side.
  bool some_byte_layout_differs() {
    if(!any_byte_layout_differs) {
      bool any=false;
      if(m.contains("layouts"))
        for(auto it=m["layouts"].begin();it!=m["layouts"].end();++it)any=source_layout_differs(it.key()) || any;
      any_byte_layout_differs=any;
    }
    return *any_byte_layout_differs;
  }
  // Refuses an output value whose cone holds a raw byte view of such an input; returns the value
  // otherwise. Every return of family() passes here. The scan visits each Arena node at most once
  // per side (byte_view_scanned).
  Id refuse_raw_byte_views(Id n) {
    if(!some_byte_layout_differs())return n;
    std::vector<Id> todo{n};
    while(!todo.empty()) {
      const Id x=todo.back();todo.pop_back();
      if(byte_view_scanned.size()<=x)byte_view_scanned.resize(a.nodes.size(),0);
      if(byte_view_scanned[x])continue;
      byte_view_scanned[x]=1;a.check();
      const auto &node=a.nodes[x];
      if(node.op=="readbits" && source_layout_differs(node.tag))
        throw Unsupported("A raw byte view of an input laid out differently on the two sides reaches an output",
                          "byte_view_layout_unsupported",
                          {{"allocation",mapped_name(m,"allocations",node.tag)},{"expression",a.render(x,3)}});
      todo.insert(todo.end(),node.args.begin(),node.args.end());
    }
    return n;
  }
  // An output that is a byte view of such an input is read at its canonical address under the
  // shared input name; both sides apply the same rule. Any other raw view of such an input in the
  // output is refused. O(rank) for the root, then refuse_raw_byte_views.
  Id byte_view_output(Id n) {
    if(!some_byte_layout_differs())return n;
    const auto &root=a.nodes[n];
    if(root.op=="readbits" && root.args.size()==1 && root.data.size()==1 && source_layout_differs(root.tag) &&
       root.data[0].denominator()==1 && root.data[0]>0 && modz(root.data[0].numerator(),8)==0) {
      const std::string raw=root.tag;const Id address=root.args[0];const Q bits=root.data[0];
      const auto canonical=canonical_byte_address(raw,address,bits.numerator()/8);
      if(!canonical)
        throw Unsupported("A byte-view output of an input laid out differently on the two sides has no proved "
                          "canonical address","byte_view_layout_unsupported",
                          {{"allocation",mapped_name(m,"allocations",raw)},{"bits",str(bits)},
                           {"address_expression",a.render(address,3)}});
      n=a.intern("readbits",{*canonical},{bits},"canonical:"+mapped_name(m,"allocations",raw),
                 {true,false,true},std::pair(Q(0),Q(pow2(bits.numerator().convert_to<unsigned>())-1)));
    }
    return refuse_raw_byte_views(n);
  }
  // Verified fixed read-only inputs (shape arrays) hold their declared values for every read of
  // either side: the native run proved that the kernel never writes them.
  // A byte view of exactly one such element at a constant offset is that element's stored bits,
  // its value modulo 2^bits. Coordinates use the side's source layout, as typed_read does. O(rank).
  std::optional<Id> fixed_bits(const std::string &raw,Id offset,unsigned bits) {
    if(!context || context->fixed_inputs.empty() || !m.contains("layouts") || !m["layouts"].contains(raw))return {};
    const auto &l=m["layouts"][raw];
    const unsigned bytes=l["element_bytes"].get<unsigned>();
    const auto address=a.value(offset);
    if(bits!=8*bytes || !address || address->denominator()!=1)return {};
    const Z off=address->numerator()-jq(l.value("source_offset",J(0))).numerator();
    Z extent=bytes,reconstructed=0;
    for(size_t j=0;j<l["shape"].size();++j)
      extent+=(jq(l["shape"][j]).numerator()-1)*jq(l["source_strides_bytes"][j]).numerator();
    if(off<0 || off+bytes>extent || modz(off,bytes)!=0)return {};
    std::vector<Id> coordinates;
    for(size_t j=0;j<l["shape"].size();++j) {
      // A shape-1 axis has coordinate 0.
      const Z count=jq(l["shape"][j]).numerator(),stride=jq(l["source_strides_bytes"][j]).numerator();
      const Z coordinate=count==1 ? Z(0) : modz(floorz(off,stride),count);
      reconstructed+=coordinate*stride;
      coordinates.push_back(a.constant(Q(coordinate)));
    }
    if(reconstructed!=off)return {};
    const std::string alloc=mapped_name(m,"allocations",raw);
    const auto range=ranges.find(alloc);
    for(const std::string type:{"s"+std::to_string(bits),"u"+std::to_string(bits)})
      if(auto fixed=context->fixed_read(a,alloc,type,coordinates,side!="left")) {
        // Input-contract intervals are checked against fixed values when the context is prepared.
        require_fixed_in_range(alloc,*fixed,range!=ranges.end() ? range->second : Bounds{});
        return a.constant(Q(modz(a.nodes[*fixed].data[0].numerator(),pow2(bits))));
      }
    return {};
  }
  // A verified fixed value outside the input's mapped range contradicts the binding: refused.
  void require_fixed_in_range(const std::string &alloc,Id fixed,const Bounds &bound) const {
    const Q &value=a.nodes[fixed].data[0];
    if(bound && (value<bound->first || value>bound->second))
      throw Unsupported("fixed input conflicts with mapped input range: "+alloc);
  }
  // Per raw node: the index levels on which the offset of some read of a fixed input inside it
  // depends (a Fold drops its own binder from its element's levels). Computed once, in one
  // topological scan, and only when the comparison has fixed inputs.
  std::optional<std::vector<std::vector<Id>>> fixed_read_levels;
  // A fixed table holds at most the context's fixed-element cap of elements (fixed_input::decode_verified).
  static constexpr uint64_t fixed_table_unroll_limit=fixed_input_element_limit;
  // The reducers whose definition unrolled_fold builds. A PTX-side Fold carries its scalar-fold
  // reducer verbatim (Product among them), without the document reader's check.
  static bool unrollable_reducer(const std::string &reducer) {
    return reducer=="Sum" || reducer=="Max" || reducer=="Min";
  }
  bool unrolls_fixed_table_fold(Id raw) {
    if(!context || context->fixed_inputs.empty() || ns[raw].count>fixed_table_unroll_limit ||
       ns[raw].operands.size()!=2 || !unrollable_reducer(ns[raw].reducer))return false;
    if(!fixed_read_levels) {
      std::vector<std::vector<Id>> levels(ns.size());
      for(size_t i=0;i<ns.size();++i) {
        const auto &n=ns[i];
        std::set<Id> s;
        for(size_t j=0;j<n.operands.size();++j)
          for(Id v:levels[n.operands[j].node])
            if(!(n.kind=="Fold" && j==1 && v==n.level))s.insert(v);
        if((n.kind=="InputValue" || n.kind=="InitialCell") && n.operands.size()==1 &&
           context->fixed_input_types.contains(mapped_name(m,"allocations",n.name)))
          s.insert(free[n.operands[0].node].begin(),free[n.operands[0].node].end());
        levels[i].assign(s.begin(),s.end());
      }
      fixed_read_levels=std::move(levels);
    }
    const auto &n=ns[raw];
    const auto &element=(*fixed_read_levels)[n.operands[1].node];
    return std::binary_search(element.begin(),element.end(),n.level);
  }
  // A stacked reduction (torch.stack(...).sum/amax): the element depends on the binder only through
  // Select(binder == constant, A, rest) chains whose arms do not depend on it. Unrolling is the Fold's
  // definition; the element is imported once per binder value, at most stacked_fold_unroll_limit times.
  static constexpr uint64_t stacked_fold_unroll_limit=64;
  bool unrolls_stacked_fold(Id raw) const {
    const auto &n=ns[raw];
    if(n.operands.size()!=2 || n.count>stacked_fold_unroll_limit || !unrollable_reducer(n.reducer))return false;
    auto independent=[&](Id node){return !std::binary_search(free[node].begin(),free[node].end(),n.level);};
    Id node=n.operands[1].node;
    for(bool first=true;;first=false) {
      const auto &select=ns[node];
      if(!first && independent(node))return true;
      if(select.kind!="Operation" || select.name!="Select" || !select.attributes.empty() || select.operands.size()!=3)return false;
      const auto &equal=ns[select.operands[0].node];
      if(equal.kind!="Operation" || equal.name!="Equal" || equal.operands.size()!=2)return false;
      bool binder=false,constant=false;
      for(const auto &side:equal.operands) {
        const auto &x=ns[side.node];
        if(x.kind=="Index" && x.level==n.level)binder=true;
        else if(x.kind=="Constant")constant=true;
      }
      if(!binder || !constant || !independent(select.operands[1].node))return false;
      node=select.operands[2].node;
    }
  }
  // Per first_argmax node built from a reference Fold: that reference's Max fold over the same values.
  std::unordered_map<Id,Id> argmax_maxima;
  // Per imported node: the read of an input at a first_argmax coordinate it contains (one per node).
  // Constants are not recorded: they contain no read.
  std::unordered_map<Id,Id> argmax_reads;
  // The most distinct nodes one replace_read visits; past it the gather is declined.
  static constexpr size_t argmax_replace_visit_limit=size_t(1)<<16;
  // The read R with every occurrence replaced by `to`, rebuilt through the exact Arena constructors of
  // add, mul, exp/exp2 and recip; nodes without R are kept. Empty when R sits under another operation
  // or the walk exceeds argmax_replace_visit_limit. A post-order walk on an explicit stack (no
  // recursion depth), each node rebuilt once: O(min(|root DAG|, limit)).
  std::optional<Id> replace_read(Id root,Id from,Id to) {
    if(root==from)return to;
    std::unordered_map<Id,Id> rebuilt;
    std::vector<std::pair<Id,bool>> todo{{root,false}};
    size_t visits=0;
    while(!todo.empty()) {
      const auto [x,ready]=todo.back();todo.pop_back();
      if(rebuilt.contains(x))continue;
      const auto &node=a.nodes[x];
      if(!ready) {
        if(++visits>argmax_replace_visit_limit) {
          ++a.stats["argmax_gather_visit_limit_declines"];
          return std::nullopt;
        }
        a.check();
        todo.emplace_back(x,true);
        for(size_t j=node.args.size();j-- > 0;)
          if(node.args[j]!=from && !rebuilt.contains(node.args[j]))todo.emplace_back(node.args[j],false);
        continue;
      }
      std::vector<Id> args;bool changed=false;
      args.reserve(node.args.size());
      for(Id arg:node.args) {
        const Id r=arg==from ? to : rebuilt.at(arg);
        changed|=r!=arg;args.push_back(r);
      }
      Id out=x;
      if(changed) {
        if(node.op=="add"){std::vector<Id> t;for(size_t j=0;j<args.size();++j)t.push_back(a.mul({a.constant(node.data[j]),args[j]}));out=a.add(t);}
        else if(node.op=="mul"){args.push_back(a.constant(node.data[0]));out=a.mul(args);}
        else if((node.op=="exp" || node.op=="exp2") && args.size()==1)out=a.exponential(args[0],node.op=="exp2");
        else if(node.op=="recip" && args.size()==1)out=a.recip(args[0]);
        else return std::nullopt;
      }
      rebuilt.emplace(x,out);
    }
    return rebuilt.at(root);
  }
  // E(R) with R = read(x, .., F, ..), F = first_argmax(v_0..v_{n-1}; d) is v_F when E(read(x, .., d_j, ..)) is v_j
  // for every j (equal up to Fold binder names): then E = v_F = max_j v_j. Also when v_j = select(G == d_j, c, u_j)
  // with u_j = E(read(.., d_j, ..)), G a first_argmax over the same indices and constant c below every u_j: the
  // argmax F of v avoids G (v_G = c < v_k for k != G), so E = u_F = v_F = max v: the reference Max fold that
  // defined F, or for masked leaves their masked_max. Checked at Divide and Multiply results containing such a
  // read. O(n x min(|E|, argmax_replace_visit_limit)).
  Id gathered_at_argmax(const RawNode &n,const std::vector<Id> &xs,Id result) {
    if(a.nodes[result].op=="read")
      for(Id coordinate:a.nodes[result].args)
        if(a.nodes[coordinate].op=="first_argmax"){argmax_reads.emplace(result,result);return result;}
    Id read=invalid;
    for(Id x:xs)if(auto it=argmax_reads.find(x);it!=argmax_reads.end()){read=it->second;break;}
    if(read==invalid)return result;
    // The result keeps the read for its consumers unless it is a constant.
    auto keep=[&] {
      if(!a.value(result))argmax_reads.emplace(result,read);
      return result;
    };
    if(n.kind!="Operation" || (n.name!="Divide" && n.name!="Multiply"))return keep();
    const auto node=a.nodes[read];
    size_t axis=node.args.size();
    for(size_t j=0;j<node.args.size();++j)if(a.nodes[node.args[j]].op=="first_argmax")axis=j;
    const Id argmax=node.args[axis];
    const auto values=a.nodes[argmax].args;const auto indices=a.nodes[argmax].data;
    if(values.size()<2)return keep();
    Id masked_by=invalid;std::optional<Q> mask;bool nonnegative=true;
    for(size_t j=0;j<values.size();++j) {
      auto coordinates=node.args;coordinates[axis]=a.constant(indices[j]);
      const Id instance_read=a.intern("read",coordinates,node.data,node.tag,node.facts,node.bounds);
      const auto instance=replace_read(result,read,instance_read);
      if(!instance)return keep();
      nonnegative=nonnegative && a.nodes[*instance].facts.finite && a.nodes[*instance].facts.nn;
      if(argmax_alpha.equal(*instance,values[j]))continue;
      const auto &leaf=a.nodes[values[j]];
      const auto &test=a.nodes[leaf.op=="select" ? leaf.args[0] : values[j]];
      const auto c=leaf.op=="select" ? a.value(leaf.args[1]) : std::nullopt;
      const bool masked=leaf.op=="select" && c && test.op=="compare" && test.tag=="eq" && test.args.size()==2 &&
          a.nodes[test.args[0]].op=="first_argmax" && a.nodes[test.args[0]].data==indices &&
          a.value(test.args[1]) && *a.value(test.args[1])==indices[j] &&
          (masked_by==invalid || masked_by==test.args[0]) && (!mask || *mask==*c) &&
          a.nodes[*instance].facts.nn && *c<0 && argmax_alpha.equal(*instance,leaf.args[2]);
      if(!masked)return keep();
      masked_by=test.args[0];mask=c;
    }
    // A masked leaf is below the others only when every instance is nonnegative.
    if(masked_by!=invalid && !nonnegative)return keep();
    if(masked_by!=invalid)
      if(auto masked=a.masked_maximum(std::set<Id>(values.begin(),values.end()))) {
        ++a.stats["argmax_gathers"];
        return *masked;
      }
    const auto fold=argmax_maxima.find(argmax);
    if(fold==argmax_maxima.end())return keep();
    ++a.stats["argmax_gathers"];
    return fold->second;
  }
  // Alpha identity of imported instances (binder names differ between the two scans of B).
  FoldAlphaIdentity argmax_alpha{a};
  // Raw shape of a first argmax: Fold(Min, init, Select(Equal(B, M), j, c)) over binder j, where M is a
  // Fold(Max) over its own binder of the same count that does not depend on j, and neither the Min's
  // element nor the Select's condition is converted. Raw structure only.
  // first_argmax_fold imports B and M's element once per index: an axis longer than this (a
  // vocabulary-sized argmax, say) keeps the generic Fold import instead of 2 x count imports.
  static constexpr uint64_t first_argmax_unroll_limit=4096;
  bool first_argmax_fold_pattern(Id raw) const {
    const auto &n=ns[raw];
    if(n.operands.size()!=2 || n.count>first_argmax_unroll_limit || !n.operands[1].conversion.empty())return false;
    const auto &select=ns[n.operands[1].node];
    if(select.kind!="Operation" || select.name!="Select" || !select.attributes.empty() || select.operands.size()!=3 ||
       !select.operands[0].conversion.empty())return false;
    const auto &equal=ns[select.operands[0].node];
    if(equal.kind!="Operation" || equal.name!="Equal" || equal.operands.size()!=2)return false;
    for(const auto &side:equal.operands) {
      const auto &m=ns[side.node];
      if(side.conversion.empty() && m.kind=="Fold" && m.reducer=="Max" && m.count==n.count &&
         !std::binary_search(free[side.node].begin(),free[side.node].end(),n.level))return true;
    }
    return false;
  }
  // Min_j select(B(j) == max_k B'(k), j, c) with init >= n-1 (or +inf) and c >= n-1 is the least j attaining
  // the max, when B'(k) is B(k) for every k (imported instances, equal up to Fold binder names), Max's init
  // is -inf and every B(j) is finite (a finite maximum is attained). Imports B(j) and B'(j) per j. O(n x B).
  std::optional<Id> first_argmax_fold(Id raw,const Env &env,std::vector<std::pair<Id,Env>> &missing) {
    const auto &n=ns[raw];
    const auto &init=n.operands[0];
    const auto &select=ns[n.operands[1].node];
    const auto &equal=ns[select.operands[0].node];
    size_t which=0;
    for(size_t j=0;j<2;++j) {
      const auto &m=ns[equal.operands[j].node];
      if(equal.operands[j].conversion.empty() && m.kind=="Fold" && m.reducer=="Max" && m.count==n.count)which=j;
    }
    const auto &maximum=ns[equal.operands[which].node];
    const auto &body=equal.operands[1-which];
    std::vector<Id> scratch;
    auto need=[&](const Edge &e,const Env &scope)->std::optional<Id> {
      if(cache.contains(key(scratch,e.node,scope)))return edge(cache.at(key(scratch,e.node,scope)),e);
      missing.emplace_back(e.node,scope);return {};
    };
    auto start=need(init,env);
    auto floor=need(maximum.operands[0],env);
    auto whole=need(equal.operands[which],env);
    std::vector<std::pair<Q,Id>> members;
    bool complete=start && floor && whole;
    for(uint64_t j=0;j<n.count;++j) {
      Env scope=env,inner=env;
      scope[n.level]=a.constant(Q(Z(j)));
      inner[maximum.level]=a.constant(Q(Z(j)));
      auto value=need(body,scope),instance=need(maximum.operands[1],inner);
      auto index=need(select.operands[1],scope),other=need(select.operands[2],scope);
      if(!value || !instance || !index || !other){complete=false;continue;}
      const auto d=a.value(*index),c=a.value(*other);
      if(!argmax_alpha.equal(*value,*instance) || !a.nodes[*value].facts.finite || !d || *d!=Q(Z(j)) ||
         !c || *c<Q(Z(n.count-1)))return {};
      members.emplace_back(*d,*value);
    }
    if(!complete)return {};
    const auto lowest=a.value(*start);
    if(!a.neginf(*floor) || !(a.nodes[*start].op=="infinity" && a.nodes[*start].data[0]==1) &&
       !(lowest && *lowest>=Q(Z(n.count-1))))return {};
    ++a.stats["first_argmax_folds"];
    // Copies of the same argmax (one per reference use) are equal up to Fold binder names: share one node.
    const Id argmax=argmax_alpha.representative(a.first_argmax(std::move(members)));
    argmax_maxima.emplace(argmax,*whole);
    return argmax;
  }
  // Whether the element, imported with the Fold's binder variable, still reads a fixed input at a
  // coordinate that depends on the binder (raw offsets can mention the binder and simplify it away).
  // Visits only nodes that depend on the binder (cached free sets); stops at the first such read.
  bool reads_fixed_table_at(Id element,Id binder) {
    std::vector<Id> todo{element};
    std::unordered_set<Id> seen;
    while(!todo.empty()) {
      const Id x=todo.back();todo.pop_back();
      if(!a.depends(x,binder) || !seen.insert(x).second)continue;
      a.check();
      const auto &node=a.nodes[x];
      if(node.op=="read")
        if(const auto split=node.tag.find('\n');split!=std::string::npos &&
           context->fixed_input_types.contains(node.tag.substr(0,split)))
          for(Id coordinate:node.args)if(a.depends(coordinate,binder))return true;
      todo.insert(todo.end(),node.args.begin(),node.args.end());
    }
    return false;
  }
  // The definition of a finite Fold: init + sum_k e(k) (Sum), max(init, e(0), ..., e(count-1)) (Max),
  // and Min as -max(-init, -e(k)). Each e(k) is imported with the binder bound to the constant k,
  // so a fixed-table read at the binder's coordinate meets its concrete element. Missing operand
  // imports are returned for the caller to schedule. O(count x element).
  std::optional<Id> unrolled_fold(Id raw,const Env &env,std::vector<std::pair<Id,Env>> &missing,
                                  const char *counter="fixed_table_folds_unrolled") {
    const auto &n=ns[raw];
    if(!unrollable_reducer(n.reducer))throw Unsupported("unsupported Fold reducer "+n.reducer);
    const auto &init=n.operands[0],&element=n.operands[1];
    std::vector<Id> scratch;
    if(!cache.contains(key(scratch,init.node,env)))missing.emplace_back(init.node,env);
    std::vector<Env> scopes;
    scopes.reserve(n.count);
    for(uint64_t j=0;j<n.count;++j) {
      Env scope=env;
      scope[n.level]=a.constant(Q(Z(j)));
      if(!cache.contains(key(scratch,element.node,scope)))missing.emplace_back(element.node,scope);
      scopes.push_back(std::move(scope));
    }
    if(!missing.empty())return {};
    std::vector<Id> terms{edge(cache.at(key(scratch,init.node,env)),init)};
    for(const auto &scope:scopes)terms.push_back(edge(cache.at(key(scratch,element.node,scope)),element));
    // Taint (hardware_infinity_keys) per term: a Max or Min carries it on, a Sum stops the family.
    std::vector<uint8_t> tainted;
    for(size_t j=0;j<terms.size();++j)
      if(hardware_infinity(j?element.node:init.node,j?scopes[j-1]:env)){tainted.resize(terms.size());tainted[j]=1;}
    if(!tainted.empty() && n.reducer!="Max" && n.reducer!="Min")
      hardware_decline(raw,terms,env,"a reassociated reduction");
    ++a.stats[counter];
    Id result;
    if(n.reducer=="Max")result=a.maximum(terms);
    else if(n.reducer=="Min") {
      std::vector<Id> negated;
      for(Id term:terms)negated.push_back(a.neg(term));
      result=a.neg(a.maximum(negated));
    } else result=a.add(terms);
    if(!tainted.empty())carry_hardware_infinity(raw,key(raw,env),terms,tainted,result);
    return result;
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
        encoded_powers.insert(power);
        ++a.stats["fp_idiom_power_instances"];
      } else if (auto exact = hardware_value(f.i, xs, tainted, f.env)) {
        result = *exact;
      } else {
        result = empty ? xs[0] : make(n, xs, f.env, body);
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
  // __logf as nvcc expands it (with and without -ftz): lg2.approx.f32(y) * RN32(ln 2), whose
  // operands x are already imported. On a proved finite positive y this is ideal log(y), under the
  // same ideal_real contract as the lg2 LSE tail; only this witnessed RN32(ln 2) literal is
  // idealized, other lg2 uses keep base two. O(1).
  std::optional<Id> compiler_log(const RawNode &root, const std::vector<Id> &x, const Env &env) {
    if (!ideal || root.type != "f32" || root.operands.size() != 2 || x.size() != 2 ||
        (!root.attributes.empty() && root.attributes != "ideal_arithmetic")) return {};
    static const Q compiler_ln2(1453635, 2097152);
    for (unsigned side = 0; side != 2; ++side) {
      const Edge &log_edge = root.operands[side], &scale = root.operands[1-side];
      const auto &log = ns.at(log_edge.node), &literal = ns.at(scale.node);
      if (!log_edge.conversion.empty() || log.kind != "Operation" || log.name != "PtxLog2" ||
          log.type != "f32" || (log.attributes != "source=f32.approx.f32 source_type=f32" &&
                                log.attributes != "source=f32.approx.ftz.f32 source_type=f32") ||
          log.operands.size() != 1 || !log.operands[0].conversion.empty() ||
          ns.at(log.operands[0].node).type != "f32") continue;
      const bool typed_constant = literal.kind == "Constant" &&
          ((literal.type == "f32" && scale.conversion.empty()) ||
           (literal.type == "b32" && scale.conversion == "Bitcast(b32 -> f32)"));
      if (!typed_constant || a.value(x[1-side]) != compiler_ln2) continue;
      const Id *input = cache.find(key(log.operands[0].node, env));
      if (!input || !a.nodes[*input].facts.finite || !a.nodes[*input].facts.pos) continue;
      ++a.stats["ideal_compiler_log_sequences"];
      return a.logarithm(*input);
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
      // A floating source holds a real value, not its bits: every rule below would read the value
      // (or its bounds) as a bit pattern. Only zero has known bits (all zero, or only the sign set),
      // which read as zero in every IEEE-style format; any other value is refused.
      if (floating(src)) {
        if (n.op == "const" && n.data[0] == 0) return a.zero;
        throw Unsupported("bitcast to floating is not a numeric conversion: " + src + " -> " + dst);
      }
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
        if (((bits >> 23) & 255u) == 255u) {
          if((bits & 0x7fffffu)==0)return a.infinity((bits>>31)?-1:1);
          throw Unsupported("NaN binary32 constant bitcast");
        }
        if (bits == 0x80000000u)
          throw Unsupported("negative-zero binary32 constant bitcast requires "
                            "signed-zero semantics");
        ++a.stats["constant_binary32_bitcasts"];
        return a.constant(ieee_value(bits, 32, 23));
      }
      // Exact binary16/BF16/FP8 literal decoding, including reduction sentinels.
      // A numeric cast of the integer payload would be incorrect. E5M2 has
      // IEEE infinities and NaNs; E4M3 has no infinity and one NaN pattern
      // per sign (S.1111.111), its other all-ones-exponent codes are finite.
      if((((dst=="f16" || dst=="bf16") && sw==16) || ((dst=="e5m2" || dst=="e4m3") && sw==8)) && n.op=="const") {
        const auto value=n.data[0];const bool signed_source=src.starts_with("s");
        const Z range=pow2(sw),half=range/2;
        const Z lowest=signed_source?Z(-half):Z(0),highest=signed_source?Z(half-1):Z(range-1);
        if(value.denominator()!=1 || value<Q(lowest) || value>Q(highest))
          throw Unsupported(std::to_string(sw)+"-bit floating payload outside source range");
        const unsigned bits=modz(value.numerator(),range).convert_to<unsigned>();
        const unsigned fraction_bits=dst=="f16"?10:dst=="bf16"?7:dst=="e5m2"?2:3;
        const unsigned max_exponent=(1u<<(sw-1-fraction_bits))-1;
        const unsigned exponent=(bits>>fraction_bits)&max_exponent, fraction=bits&((1u<<fraction_bits)-1);
        const unsigned sign=bits>>(sw-1);
        if(dst=="e4m3" ? exponent==max_exponent && fraction==(1u<<fraction_bits)-1 : exponent==max_exponent) {
          if(dst!="e4m3" && fraction==0)return a.infinity(sign?-1:1);
          throw Unsupported("NaN "+std::to_string(sw)+"-bit floating constant bitcast");
        }
        if(bits==(1u<<(sw-1)))
          throw Unsupported("negative-zero "+std::to_string(sw)+"-bit floating constant bitcast requires signed-zero semantics");
        return a.constant(ieee_value(bits,sw,fraction_bits));
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
      // A BF16 power of two built in its exponent field, before the field decoders below
      // (they give the same value with an explicit fraction factor). f32/f16: at the end.
      if(dst=="bf16")if(auto power=exponent_field_power(x,src,dst,false))return *power;
      // Constructed BF16 values with fixed sign/exponent are affine in their
      // seven fraction bits. No approximate host float or instance enumeration.
      if (dst == "bf16")
        if (auto field = fixed_header_float(x, sw, dst)) return *field;
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
      // A bitcast of a selection is the selection of the alternatives'
      // bitcasts (both are bit patterns of the source type). Memoized per
      // selection node: O(distinct selections), depth-bounded.
      if(n.op=="select" && n.args.size()==3) {
        const auto key=std::tuple(x,src,dst);
        if(auto it=select_bitcasts.find(key);it!=select_bitcasts.end())return it->second;
        if(select_bitcast_depth>=select_bitcast_depth_limit)
          throw Unsupported("bitcast of nested selections exceeds depth limit");
        const Id p=n.args[0],t=n.args[1],f=n.args[2];
        ++select_bitcast_depth;
        struct Depth {unsigned& n;~Depth(){--n;}} depth{select_bitcast_depth};
        const Id yes=cast(t,src,dst,true),no=cast(f,src,dst,true);
        ++a.stats["select_floating_bitcasts"];
        return select_bitcasts.emplace(key,a.select(p,yes,no)).first->second;
      }
      // f16 and f32 fields, after the lane and selection rules. A bf16 field already took the same rule
      // above, before the bf16 range rule; it is not counted in fixed_header_float_bitcasts.
      if(dst!="bf16")
        if(auto field=fixed_header_float(x,sw,dst)) {
          ++a.stats["fixed_header_float_bitcasts"];
          return *field;
        }
      if(auto power=exponent_field_power(x,src,dst,true))return *power;
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
      // An e2m1 satfinite result read as its four bits is encoded exactly (e2m1_bits). The encoding
      // is total in its input, NaN and infinities included, so it needs no finiteness proof. Any
      // other use of the result keeps the saturation path below.
      if (ideal && attr == " rounding=nearest-even saturate_finite " && src == "f32" && dst == "e2m1") {
        const auto next = ss.tellg();
        std::string bits;
        std::smatch b;
        if (std::getline(ss, bits, ';') && std::regex_match(bits, b, re) && b[1] == "Bitcast" &&
            std::string(b[2]).find_first_not_of(' ') == std::string::npos && b[3] == "e2m1" && b[4] == "u4") {
          x = e2m1_bits(x);
          continue;
        }
        ss.clear();
        ss.seekg(next);
      }
      if (attr.find("saturat") != attr.npos &&
          !(output && floating(src) && floating(dst) &&
            mapping.value("output_value_semantics", "") == "ideal_real")) {
        if (!ideal || !floating(src) || !floating(dst))
          throw Unsupported("Saturation requires supported ideal-real floating semantics",
                            "saturation_semantics_unsupported",
                            {{"ideal_real",ideal},{"source_type",src},{"target_type",dst},{"conversion",part}});
        // cvt .sat limits the result to [0.0, 1.0] (PTX ISA, cvt): -inf becomes 0.
        if (attr == " saturate " && a.neginf(x)) {
          ++a.stats["saturated_negative_infinities"];
          x = cast(a.zero, src, dst, m[1] == "Bitcast");
          continue;
        }
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
  // The four bits of cvt.rn.satfinite.e2m1 of a binary32 y, for every y. E2M1 has a sign, two
  // exponent bits and one mantissa bit (magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6) and no infinity or
  // NaN. NaN gives +MAXNORM, 0b0111. Otherwise bit 3 is the sign bit of y's binary32 encoding (also
  // for results that round to zero and for saturated ones), and bits 0..2 are the code of |y|
  // rounded to nearest even, saturated to MAXNORM 6 above. Of the midpoints, 0.25, 1.25, 2.5 and 5
  // round down to the even code, 0.75, 1.75 and 3.5 round up to it. O(1).
  Id e2m1_bits(Id y) {
    struct Midpoint { Q value; bool rounds_up; };
    static const Midpoint midpoints[] = {{Q(1, 4), false}, {Q(3, 4), true}, {Q(5, 4), false},
                                         {Q(7, 4), true}, {Q(5, 2), false}, {Q(7, 2), true},
                                         {Q(5), false}};
    const Id magnitude = a.absolute(y);
    Id code = a.zero;
    for (unsigned k = 0; k < std::size(midpoints); ++k)
      code = a.select(a.compare(midpoints[k].rounds_up ? "ge" : "gt", magnitude,
                                a.constant(midpoints[k].value)),
                      a.constant(Q(k + 1)), code, true);
    const Id word = cast(y, "f32", "u32", true);
    const Id sign = a.compare("ne", a.mul({a.constant(Q(pow2(31))), a.imod(a.idiv(word, pow2(31)), 2)}),
                              a.zero);
    ++a.stats["e2m1_satfinite_encodings"];
    const Id bits = a.select(a.compare("ne", y, y), a.constant(Q(7)),
                             a.add({code, a.mul({sign, a.constant(Q(8))})}), true);
    // Integers for every y (the code and sign also surface as terms of a packing sum once a finite
    // y removes the NaN selection): word() may reduce their packing coefficients.
    packed_decoders.known_integers.insert({code, sign, bits});
    return bits;
  }
  // f16, bf16 or f32 bits whose bounds fix the sign and a normal exponent field (e.g. the uint4
  // dequantization idiom (w & 0x000F000F) | 0x64006400) are affine in the fraction bits:
  // +-2^(e-bias-f) (bits - header*2^f + 2^f). O(1).
  std::optional<Id> fixed_header_float(Id x, unsigned w, const std::string &dst) {
    unsigned fraction = 0, exponent_bits = 0;
    if (dst == "f16" && w == 16) { fraction = 10; exponent_bits = 5; }
    else if (dst == "bf16" && w == 16) { fraction = 7; exponent_bits = 8; }
    else if (dst == "f32" && w == 32) { fraction = 23; exponent_bits = 8; }
    else return std::nullopt;
    const auto &bound = a.nodes[x].bounds;
    const Q unit = Q(pow2(fraction));
    if (!bound || bound->first < 0 || bound->second >= Q(pow2(w)) ||
        floorq(bound->first / unit) != floorq(bound->second / unit)) return std::nullopt;
    const Z header = floorq(bound->first / unit);
    const Z exponent = modz(header, pow2(exponent_bits));
    if (exponent == 0 || exponent == pow2(exponent_bits) - 1) return std::nullopt;
    const int bias = (1 << (exponent_bits - 1)) - 1;
    const int shift = exponent.convert_to<int>() - bias - int(fraction);
    const Q scale = shift >= 0 ? Q(pow2(unsigned(shift))) : Q(1, pow2(unsigned(-shift)));
    Id significand = a.add({x, a.constant(unit - Q(header) * unit)});
    return a.mul({a.constant(header >= pow2(exponent_bits) ? -scale : scale), significand});
  }
  // An integer word read as f32, f16 or bf16 whose fraction field is zero and whose biased exponent field is
  // proved inside the normal range [1, 2^E - 2] is exactly 2^(field - bias): the exponent built by integer
  // arithmetic, e.g. (e + 127) << 23. The word's bits are its residue modulo 2^width (a signed source is the
  // two's-complement reading of its word). Fraction and field come from the arena's exact residue and quotient
  // and their interval bounds. With `refuse`, a zero fraction whose field is not proved normal (zero,
  // subnormal, infinity, NaN or a negative sign) is refused precisely; otherwise none is returned and the
  // other bitcast rules apply. O(affine terms of the word).
  std::optional<Id> exponent_field_power(Id x, const std::string &src, const std::string &dst, bool refuse) {
    const unsigned w = width(src);
    unsigned fraction_bits = 0, bias = 0;
    if (dst == "f32" && w == 32) { fraction_bits = 23; bias = 127; }
    else if (dst == "f16" && w == 16) { fraction_bits = 10; bias = 15; }
    else if (dst == "bf16" && w == 16) { fraction_bits = 7; bias = 127; }
    else return std::nullopt;
    if (!a.known_bits(x).integer) return std::nullopt;
    const auto &n = a.nodes[x];
    Id word = n.op == "signed" && n.args.size() == 1 && n.data.size() == 1 && n.data[0] == Q(w) ? n.args[0] : x;
    Id bits = a.imod(word, pow2(w));
    if (a.imod(bits, pow2(fraction_bits)) != a.zero) return std::nullopt;
    Id field = a.idiv(bits, pow2(fraction_bits));
    const Z top = pow2(w - 1 - fraction_bits) - 2;
    const auto &range = a.nodes[field].bounds;
    if (!range || range->first < Q(1) || range->second > Q(top)) {
      if (!refuse) return std::nullopt;
      throw Unsupported("integer-built " + dst + " exponent field is not proved in the normal range [1," +
                        top.str() + "]" + (range ? " (bounds " + str(range->first) + ".." + str(range->second) + ")" : ""));
    }
    ++a.stats["exponent_field_powers"];
    return a.exp2(a.sub(field, a.constant(Q(bias))));
  }
  Id literal(const std::string &s, const std::string &t) {
    if (s == "-inf" || s == "-infinity" || s == "-Infinity")
      return a.infinity(-1);
    if (s == "inf" || s == "+inf" || s == "infinity" || s == "Infinity")
      return a.infinity(1);
    if (t == "pred" && (s == "true" || s == "false"))
      return s == "true" ? a.one : a.zero;
    try {
      Q q;
      // A binary32/binary64 literal denotes its nearest IEEE value (as a host float literal does),
      // not the decimal number spelled. floatFromChars rounds once, directly to the literal's type,
      // and accepts an exact subnormal; only overflow and underflow to zero are out of range. O(digits).
      const auto binary = [&](auto parsed, unsigned fraction) {
        decltype(parsed) f = 0;
        const auto r = sentryward::equivalence::floatFromChars(s.data(), s.data() + s.size(), f);
        if (r.ec != std::errc{} || r.ptr != s.data() + s.size() || !std::isfinite(f))
          throw Unsupported("constant " + s + " type " + t);
        using Word = std::conditional_t<sizeof(f) == 4, uint32_t, uint64_t>;
        return ieee_value(std::bit_cast<Word>(f), 8 * sizeof(f), fraction);
      };
      if (t == "f32") q = binary(float{}, 23);
      else if (t == "f64") q = binary(double{}, 52);
      else if (s.starts_with("0x") || s.starts_with("0X"))
        q = Q(Z(s));
      else
        q = decimal(s);
      // PTX reads an integer constant used as a predicate as C does: zero is false, nonzero true
      // (ISA, Predicate Constants), so `mov.pred %p, -1` is true, not the number -1.
      if (t == "pred")
        return q == 0 ? a.zero : a.one;
      if (!floating(t)) {
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
  // The layout mapping below reads each source coordinate as (offset / stride_j) mod extent_j, the
  // digits of a dense source layout only. A strided source layout (nested_layouts, which holds the
  // dominant ones too) is refused: with padding, e.g. strides [1536, 256, 2] for shape [7, 4, 128],
  // that formula maps offset 1536 to (1, 2, 0) instead of (1, 0, 0) and padding bytes to elements.
  // An identity layout (target strides = source strides, dense) is never strided.
  std::optional<PackedDecodePass::Bits> canonical_packed_slice(Id source,unsigned lo,unsigned count) {
    const auto &read=a.nodes[source];
    if(read.op!="readbits" || read.args.size()!=1 || !count || count>4 ||
       lo%8+count>8 || Q(lo+count)>read.data[0] ||
       !m.contains("layouts") || !m["layouts"].contains(read.tag) ||
       !packed_dense_layouts.contains(read.tag) || nested_layouts.contains(read.tag))return {};
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
    // A nested strided layout: the offset must be a multiple of the innermost stride and each
    // inner digit below its axis's extent x stride (the outermost digit is bounded by the
    // extent above); then its digits are the element's coordinates. O(rank) Arena terms.
    std::vector<Id> nested_coordinates;
    if (const auto nested = nested_layouts.find(raw); valid && nested != nested_layouts.end()) {
      const auto &axes = nested->second;
      nested_coordinates.assign(l["shape"].size(), a.zero);
      if (dominant_layouts.contains(raw)) {
        // Greedy digits from the largest stride: each remainder must lie below its axis's
        // extent x stride, and the last remainder must be zero. O(rank) Arena terms.
        Id rest = off;
        for (size_t j = axes.size(); valid && j-- > 0;) {
          const auto &bound = a.nodes[rest].bounds;
          if (j + 1 < axes.size() &&
              (!bound || bound->first < 0 || bound->second >= Q(axes[j].stride * axes[j].count))) valid = false;
          nested_coordinates[axes[j].dim] = a.idiv(rest, axes[j].stride);
          rest = a.imod(rest, axes[j].stride);
        }
        if (rest != a.zero) valid = false;
      } else {
        if (!axes.empty() && a.imod(off, axes.front().stride) != a.zero) valid = false;
        for (size_t j = 0; valid && j < axes.size(); ++j) {
          Id digit = off;
          if (j + 1 < axes.size()) {
            digit = a.imod(off, axes[j + 1].stride);
            const auto &bound = a.nodes[digit].bounds;
            if (!bound || bound->first < 0 || bound->second >= Q(axes[j].stride * axes[j].count)) valid = false;
          }
          nested_coordinates[axes[j].dim] = a.idiv(digit, axes[j].stride);
        }
      }
    }
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
    std::vector<Id> coords=std::move(nested_coordinates);
    if (coords.empty())
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
    // Bind fixed read-only inputs on both sides (see fixed_bits) only after the
    // ordinary identity, storage type, bounds and coordinate checks above;
    // never specialize by expression node ID.
    if(context)
      if(auto fixed=context->fixed_read(a,alloc,t,coords,side!="left")) {
        require_fixed_in_range(alloc,*fixed,bound);
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
      if (hi.op != "mul" || hi.data != std::vector<Q>{Q(pow2(32))} ||
          hi.args.size() != 1 || lo.args.size() != 1)
        continue;
      const auto &h = a.nodes[hi.args[0]];
      if (h.args != lo.args)
        continue;
      const Id source = lo.args[0];
      if (lo.op == "extractbits" && lo.data == std::vector<Q>{Q(0), Q(32)} &&
          h.op == "extractbits" && h.data == std::vector<Q>{Q(32), Q(32)}) {
        ++a.stats["exact_word_reconstruction"];
        return unwrap(word(source, 64), 64);
      }
      // The same split in arithmetic form (a u64 shifted right by 32 and
      // truncated to u32): x mod 2^32 + 2^32*floor(x/2^32) = x for every
      // integer x, so the sum is congruent to x modulo 2^64.
      if (lo.op == "word" && lo.data == std::vector<Q>{Q(32)} &&
          h.op == "idiv" && h.data == std::vector<Q>{Q(pow2(32))} &&
          a.known_bits(source).integer) {
        ++a.stats["exact_word_arithmetic_reconstruction"];
        return unwrap(word(source, 64), 64);
      }
    }
    return {};
  }
  // Calls f(lo, count) for each maximal run of set bits of `mask`, lowest first. O(32).
  template <class F> static void for_each_bit_run(uint32_t mask, F f) {
    for (unsigned lo = 0; lo < 32;) {
      if (!((mask >> lo) & 1u)) { ++lo; continue; }
      unsigned hi = lo + 1;
      while (hi < 32 && ((mask >> hi) & 1u)) ++hi;
      f(lo, hi - lo);
      lo = hi;
    }
  }
  Id mask32(Id x, uint32_t mask) {
    std::vector<Id> pieces;
    for_each_bit_run(mask, [&](unsigned lo, unsigned count) {
      pieces.push_back(a.mul({a.constant(Q(pow2(lo))), a.imod(a.idiv(x, pow2(lo)), pow2(count))}));
    });
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
    // (x & positive) | ones with disjoint masks (e.g. lop3 0xEA, the dequantization idiom
    // (w & 0x00F000F0) | 0x64006400): keep each selected run a bounded bit field of x, so
    // the fields stay visible to typed slices and packed decoders. O(runs).
    if (negative == 0) {
      std::vector<Id> pieces{a.constant(Q(Z(ones)))};
      for_each_bit_run(positive, [&](unsigned lo, unsigned count) {
        pieces.push_back(a.mul({a.constant(Q(pow2(lo))), a.bit_slice(x, lo, count)}));
      });
      return a.add(std::move(pieces));
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
      const Id *found = cache.find(key(e.node,ctx));
      if (!found) return {};
      return edge(*found,e);
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

  // Range premise (assumption floating_values_in_range, docs/assumptions.md): a finite ideal value of
  // an IEEE type lies within that type's finite range. Hence, beside a finite operand, the type's most
  // negative finite value never decides a Maximum (nor its largest finite value a Minimum); such
  // operands are dropped. Without the premise nothing is dropped. O(operands).
  std::vector<Id> in_range(std::vector<Id> x, const std::string &type, bool maximum) {
    const auto format = float_format(type);
    if (!range_premise || !format) return x;
    const Q largest = largest_finite(*format);
    const Id bound = a.constant(maximum ? -largest : largest);
    if (std::none_of(x.begin(), x.end(), [&](Id o) { return o != bound && a.nodes[o].facts.finite; })) return x;
    if (const auto dropped = std::erase(x, bound)) a.stats["range_premise_bounds_dropped"] += dropped;
    return x;
  }
  // Hardware values of the PTX side (docs/reference-binding-json.md, numerical model). An IEEE
  // binary format: precision p (significand bits, the hidden bit included), largest exponent emax
  // and smallest normal exponent -min_normal_shift.
  struct FloatFormat { unsigned precision, max_exponent, min_normal_shift; };
  static std::optional<FloatFormat> float_format(const std::string &t) {
    const auto ieee = sentryward::equivalence::ieeeFormat(t);
    if (!ieee) return std::nullopt;
    const unsigned max_exponent = (1u << (ieee->width - 2 - ieee->fraction)) - 1;
    return FloatFormat{ieee->fraction + 1, max_exponent, max_exponent - 1};
  }
  // The largest finite magnitude MAX = 2^(emax+1) - 2^(emax+1-p).
  static Q largest_finite(const FloatFormat &f) {
    return Q(pow2(f.max_exponent + 1) - pow2(f.max_exponent + 1 - f.precision));
  }
  // Round to nearest even carries a result to infinity from the midpoint 2^(emax+1) - 2^(emax-p)
  // between MAX and 2^(emax+1) on (IEEE 754-2019, 4.3.1 and 7.4).
  static Q overflow_threshold(const FloatFormat &f) {
    return Q(pow2(f.max_exponent + 1) - pow2(f.max_exponent - f.precision));
  }
  // Whether q is a finite value of format f: |q| <= MAX and q = odd 2^z with at most p significant bits
  // and z at least the subnormal spacing exponent emin + 1 - p. O(digits).
  static bool representable(const Q &q, const FloatFormat &f) {
    if (q == 0) return true;
    const Z n = boost::multiprecision::abs(q.numerator()), d = q.denominator();
    if ((d & (d - 1)) != 0 || Q(n, d) > largest_finite(f)) return false;
    const unsigned zeros = unsigned(boost::multiprecision::lsb(n));
    const long exponent = long(zeros) - long(boost::multiprecision::msb(d));
    return boost::multiprecision::msb(Z(n >> zeros)) + 1 <= f.precision &&
           exponent >= 1 - long(f.min_normal_shift) - long(f.precision);
  }
  // An extended real: -inf (infinity -1), a rational (0) or +inf (1).
  struct Extended { int infinity = 0; Q value = Q(0); };
  static bool before(const Extended &x, const Extended &y) {
    if (x.infinity != y.infinity) return x.infinity < y.infinity;
    return !x.infinity && x.value < y.value;
  }
  // The hardware value set of a floating value: every extended real in [lo, hi] may occur, and NaN
  // when `nan`. The default admits every value.
  struct ValueSet { Extended lo{-1}, hi{1}; bool nan = true; };
  static ValueSet exactly(const Q &q) { return {{0, q}, {0, q}, false}; }
  static ValueSet infinite(int sign) { return {{sign}, {sign}, false}; }
  // The sign of the one infinity that `s` holds, else 0.
  static int single_infinity(const ValueSet &s) {
    return !s.nan && s.lo.infinity && s.lo.infinity == s.hi.infinity ? s.lo.infinity : 0;
  }
  static bool finite_set(const ValueSet &s) { return !s.nan && !s.lo.infinity && !s.hi.infinity; }
  static ValueSet hull(const ValueSet &x, const ValueSet &y) {
    return {before(y.lo, x.lo) ? y.lo : x.lo, before(x.hi, y.hi) ? y.hi : x.hi, x.nan || y.nan};
  }
  static ValueSet negated(const ValueSet &s) {
    const auto flip = [](Extended e) { e.infinity = -e.infinity; e.value = -e.value; return e; };
    return {flip(s.hi), flip(s.lo), s.nan};
  }
  // The exact sum of two sets: NaN only from opposite infinities, which also widen the bound.
  static ValueSet sum(const ValueSet &x, const ValueSet &y) {
    const auto plus = [](const Extended &u, const Extended &v, int widen) -> Extended {
      if (u.infinity && v.infinity && u.infinity != v.infinity) return {widen};
      if (u.infinity) return u;
      if (v.infinity) return v;
      return {0, u.value + v.value};
    };
    return {plus(x.lo, y.lo, -1), plus(x.hi, y.hi, 1),
            x.nan || y.nan || (x.hi.infinity > 0 && y.lo.infinity < 0) || (x.lo.infinity < 0 && y.hi.infinity > 0)};
  }
  // The exact product of two sets: the extremes lie at the corners (a zero times an infinity bounds as
  // zero), and NaN arises from a zero times an infinity.
  static ValueSet product(const ValueSet &x, const ValueSet &y) {
    const auto sign = [](const Extended &e) { return e.infinity ? e.infinity : (e.value > 0) - (e.value < 0); };
    const auto times = [&](const Extended &u, const Extended &v) -> Extended {
      if (u.infinity || v.infinity) return {sign(u) * sign(v)};
      return {0, u.value * v.value};
    };
    const auto zero = [](const ValueSet &s) { return !before(Extended{}, s.lo) && !before(s.hi, Extended{}); };
    const auto unbounded = [](const ValueSet &s) { return s.lo.infinity || s.hi.infinity; };
    const std::array<Extended, 4> corners{times(x.lo, y.lo), times(x.lo, y.hi), times(x.hi, y.lo), times(x.hi, y.hi)};
    return {*std::min_element(corners.begin(), corners.end(), before),
            *std::max_element(corners.begin(), corners.end(), before),
            x.nan || y.nan || (zero(x) && unbounded(y)) || (zero(y) && unbounded(x))};
  }
  // A flush to zero (.ftz) replaces a subnormal by a zero of its sign: a bound strictly between zero
  // and the smallest normal magnitude moves to zero.
  static ValueSet flushed(ValueSet s, const FloatFormat &f) {
    const Q normal(1, pow2(f.min_normal_shift));
    if (!s.lo.infinity && s.lo.value > 0 && s.lo.value < normal) s.lo.value = Q(0);
    if (!s.hi.infinity && s.hi.value < 0 && s.hi.value > -normal) s.hi.value = Q(0);
    return s;
  }
  // The value of format f at or below (`up`: at or above) finite q, on f's grid without range
  // limits: q rounded down (up) to a multiple of 2^(e+1-p), e the exponent of q but at least emin. O(digits).
  static Q on_grid(const Q &q, const FloatFormat &f, bool up) {
    if (q == 0) return q;
    const Q magnitude = q < 0 ? -q : q;
    long e = long(boost::multiprecision::msb(magnitude.numerator())) -
             long(boost::multiprecision::msb(magnitude.denominator()));
    const auto power = [](long k) { return k >= 0 ? Q(pow2(unsigned(k))) : Q(1, pow2(unsigned(-k))); };
    if (magnitude < power(e)) --e;
    const Q spacing = power(std::max(e, -long(f.min_normal_shift)) + 1 - long(f.precision));
    const Q steps = q / spacing;
    Q whole(floorq(steps));
    if (up && whole != steps) whole += Q(1);
    return whole * spacing;
  }
  // Exact values in `s` rounded once to format f in the rounding modes `modes` (instruction_mode
  // bits; when none is known, in every mode), flushed when `flush`. A finite bound moves to f's grid
  // outward (every mode rounds a value between the two neighbours on that grid). IEEE 754-2019 7.4
  // gives the overflow results: to nearest, at and beyond the threshold the infinity of the sign;
  // toward zero ±MAX; toward -inf, -inf below -MAX and MAX above MAX; toward +inf, +inf above MAX
  // and -MAX below -MAX.
  static ValueSet rounded(const ValueSet &s, const FloatFormat &f, uint8_t modes, bool flush) {
    namespace mode = sentryward::scalar_fold::instruction_mode;
    enum Direction : unsigned { nearest = 1, zero = 2, down = 4, up = 8 };
    unsigned directions = (modes & (mode::contractible | mode::nearest) ? nearest : 0) |
                          (modes & mode::toward_zero ? zero : 0) | (modes & mode::toward_negative ? down : 0) |
                          (modes & mode::toward_positive ? up : 0);
    if (!directions) directions = nearest | zero | down | up;
    const Q largest = largest_finite(f), threshold = overflow_threshold(f);
    const auto bound = [&](const Extended &e, bool upper, unsigned d) -> Extended {
      if (e.infinity) return e;
      const Q &q = e.value;
      if (d == nearest && q <= -threshold) return {-1};
      if (d == nearest && q >= threshold) return {1};
      if (q < -largest) return d == down ? Extended{-1} : Extended{0, -largest};
      if (q > largest) return d == up ? Extended{1} : Extended{0, largest};
      return {0, on_grid(q, f, upper)};
    };
    std::optional<ValueSet> result;
    for (unsigned d = nearest; d <= up; d <<= 1) {
      if (!(directions & d)) continue;
      const ValueSet r{bound(s.lo, false, d), bound(s.hi, true, d), s.nan};
      result = result ? hull(*result, r) : r;
    }
    return flush ? flushed(*result, f) : *result;
  }
  // The values a max_num (min_num) of operands in `sets` can take: a NaN operand is ignored unless
  // .NaN propagates it (PTX ISA, max/min); an operand that is surely a number bounds a maximum from
  // below, every operand from above.
  static ValueSet extremum(std::vector<ValueSet> sets, bool maximum, bool propagate) {
    if (!maximum)
      for (auto &s : sets) s = negated(s);
    ValueSet r{{1}, {-1}, false};
    std::optional<Extended> sure;
    bool any = false, all = true;
    for (const auto &s : sets) {
      if (before(r.hi, s.hi)) r.hi = s.hi;
      if (before(s.lo, r.lo)) r.lo = s.lo;
      if ((!s.nan || propagate) && (!sure || before(*sure, s.lo))) sure = s.lo;
      any = any || s.nan;
      all = all && s.nan;
    }
    if (sure) r.lo = *sure;
    r.nan = propagate ? any : all;
    return maximum ? r : negated(r);
  }
  // The leading Bitcasts of `conversion` from `type`: the type they reach, the number of parts they
  // take and the remaining parts, or none when a Bitcast changes width or does not start at `type`.
  struct BitcastChain { std::string type, rest; size_t casts = 0; };
  static std::optional<BitcastChain> bitcast_chain(std::string type, const std::string &conversion) {
    static const std::regex bitcast(R"(\s*Bitcast\(\s*([^ ()]+)\s*->\s*([^ ()]+)\s*\)\s*)");
    std::vector<std::string> parts;
    std::stringstream text(conversion);
    for (std::string part; std::getline(text, part, ';');) parts.push_back(part);
    BitcastChain chain;
    std::smatch m;
    for (; chain.casts < parts.size() && std::regex_match(parts[chain.casts], m, bitcast); ++chain.casts) {
      if (m[1] != type || width(m[1]) != width(m[2])) return std::nullopt;
      type = m[2];
    }
    chain.type = type;
    for (size_t i = chain.casts; i < parts.size(); ++i) chain.rest += (chain.rest.empty() ? "" : ";") + parts[i];
    return chain;
  }
  // The exact value of a raw Constant through `conversion`: none, or a chain of Bitcasts of integer
  // bits of one width to an IEEE type; one floating Convert may follow (converted). An f16 or bf16
  // literal that is no value of its type admits every value. Decoded without interning; memoized per
  // raw node and conversion (constant_set). O(digits).
  ValueSet parse_constant(const RawNode &n, const std::string &conversion) {
    try {
      const auto chain = bitcast_chain(n.type, conversion);
      if (!chain) return {};
      const size_t casts = chain->casts;
      const std::string &type = chain->type;
      const auto format = float_format(type);
      if (!format) return {};
      ValueSet value;
      if (casts == 0) {
        const std::string &s = n.name;
        if (s == "-inf" || s == "-infinity" || s == "-Infinity") value = infinite(-1);
        else if (s == "inf" || s == "+inf" || s == "infinity" || s == "Infinity") value = infinite(1);
        else {
          Q q;
          const auto binary = [&](auto parsed, unsigned fraction) -> bool {
            decltype(parsed) number = 0;
            const auto r = sentryward::equivalence::floatFromChars(s.data(), s.data() + s.size(), number);
            if (r.ec != std::errc{} || r.ptr != s.data() + s.size() || !std::isfinite(number)) return false;
            using Word = std::conditional_t<sizeof(number) == 4, uint32_t, uint64_t>;
            q = ieee_value(std::bit_cast<Word>(number), 8 * sizeof(number), fraction);
            return true;
          };
          if (type == "f32" ? !binary(float{}, 23) : type == "f64" ? !binary(double{}, 52) : false) return {};
          if (type != "f32" && type != "f64") {
            q = decimal(s);
            if (!representable(q, *format)) return {};
          }
          value = exactly(q);
        }
      } else {
        if (floating(n.type)) return {};
        const unsigned w = width(type), fraction = format->precision - 1;
        const Q q = n.name.starts_with("0x") || n.name.starts_with("0X") ? Q(Z(n.name)) : decimal(n.name);
        if (q.denominator() != 1) return {};
        const Z bits = modz(q.numerator(), pow2(w));
        const Z exponent_mask = pow2(w - 1) - pow2(fraction);
        if ((bits & exponent_mask) == exponent_mask) {
          if ((bits & (pow2(fraction) - 1)) != 0) return {};
          value = infinite(bits >= pow2(w - 1) ? -1 : 1);
        } else value = exactly(ieee_value(bits.convert_to<uint64_t>(), w, fraction));
      }
      return converted(value, chain->rest);
    } catch (...) {
      return {};
    }
  }
  std::unordered_map<Id, std::vector<std::pair<std::string, ValueSet>>> constant_sets;
  const ValueSet &constant_set(Id node, const std::string &conversion) {
    auto &sets = constant_sets[node];
    for (const auto &[text, set] : sets)
      if (text == conversion) return set;
    return sets.emplace_back(conversion, parse_constant(ns[node], conversion)).second;
  }
  // An edge conversion as the value sets read it, parsed once per spelling: one floating Convert
  // rounds once in the rounding mode it names (none: an exact widening), .sat limits to [0.0, 1.0] and
  // flushes NaN to +0.0 (PTX ISA, cvt). Any other conversion admits every value.
  struct Conversion { enum Kind { every, saturate, round } kind = every; FloatFormat target{}; uint8_t modes = 0; bool flush = false; };
  std::unordered_map<std::string, Conversion> conversions;
  ValueSet converted(const ValueSet &s, const std::string &conversion) {
    if (conversion.empty()) return s;
    auto found = conversions.find(conversion);
    if (found == conversions.end()) {
      Conversion c;
      static const std::regex convert(R"(\s*Convert([^();]*)\(\s*([^ ()]+)\s*->\s*([^ ()]+)\s*\)\s*)");
      std::smatch m;
      if (std::regex_match(conversion, m, convert) && float_format(m[2]))
        if (const auto target = float_format(m[3])) {
          const std::string attributes = m[1];
          namespace mode = sentryward::scalar_fold::instruction_mode;
          if (attributes.find("satfinite") != std::string::npos || attributes.find("saturate_finite") != std::string::npos)
            c.kind = Conversion::every;
          else if (attributes.find("saturat") != std::string::npos) c.kind = Conversion::saturate;
          else {
            c.kind = Conversion::round;
            c.target = *target;
            if (attributes.find("rounding=nearest-even") != std::string::npos) c.modes |= mode::nearest;
            if (attributes.find("rounding=toward-zero") != std::string::npos) c.modes |= mode::toward_zero;
            if (attributes.find("rounding=toward-negative") != std::string::npos) c.modes |= mode::toward_negative;
            if (attributes.find("rounding=toward-positive") != std::string::npos) c.modes |= mode::toward_positive;
            if (!c.modes && attributes.find("rounding=") == std::string::npos) c.modes = mode::nearest;
            c.flush = attributes.find("flush_to_zero") != std::string::npos;
          }
        }
      found = conversions.emplace(conversion, c).first;
    }
    const auto &c = found->second;
    if (c.kind == Conversion::saturate) return {{0, Q(0)}, {0, Q(1)}, false};
    if (c.kind == Conversion::round) return rounded(s, c.target, c.modes, c.flush);
    return {};
  }
  // Hardware value sets of raw floating values, independent of the import scope (memoized per raw
  // node): constants; Select (either arm), Negate, Maximum/Minimum and Max/Min Folds (an initial value
  // bounds a running maximum from below); and PTX instruction images (RawNode::instruction) of add,
  // sub, mul, fma and ex2.approx. Any other node admits every value, NaN included: an input element, a
  // reassociated or accumulated sum, an MMA, a division, a software function. The walk
  // enters only the operands that node_set reads (set_operands), so a node that admits every value
  // ends it. O(nodes and edges of the floating cones read) over all queries.
  std::unordered_map<Id, ValueSet> value_sets;
  // The operands of raw node `id` whose sets node_set reads (an add's contraction reads its mul's
  // operands through the mul's own entry). O(operands).
  void set_operands(Id id, std::vector<Id> &out) const {
    const auto &n = ns[id];
    if (!float_format(n.type)) return;
    const auto from = [&](size_t first) {
      for (size_t j = first; j < n.operands.size(); ++j) out.push_back(n.operands[j].node);
    };
    if (n.kind == "Fold") {
      if (n.operands.size() == 2 && (n.reducer == "Max" || n.reducer == "Min")) from(0);
    } else if (n.kind == "Operation") {
      if (n.name == "Select") { if (n.operands.size() == 3) from(1); }
      else if (n.name == "Negate" || n.name == "Maximum" || n.name == "Minimum") from(0);
      else if (sentryward::scalar_fold::instruction_mode::instruction(n.instruction) &&
               (n.name == "Exp2" || n.name == "Multiply" || n.name == "Fma" || n.name == "Add" || n.name == "Subtract"))
        from(0);
    }
  }
  const ValueSet &raw_set(Id root) {
    if (const auto found = value_sets.find(root); found != value_sets.end()) return found->second;
    std::vector<std::pair<Id, bool>> todo{{root, false}};
    std::unordered_set<Id> expanded;
    std::vector<Id> operands;
    while (!todo.empty()) {
      const auto [id, ready] = todo.back();
      todo.pop_back();
      if (value_sets.contains(id) || (!ready && !expanded.insert(id).second)) continue;
      a.check();
      if (!ready) {
        todo.emplace_back(id, true);
        operands.clear();
        set_operands(id, operands);
        for (const Id o : operands)
          if (!value_sets.contains(o) && ns[o].kind != "Constant") todo.emplace_back(o, false);
        continue;
      }
      value_sets.emplace(id, node_set(ns[id], nullptr, 0));
    }
    return value_sets.at(root);
  }
  // Scope-dependent value sets of instruction images whose imported value is a constant, per import
  // key and query depth (so that a result does not depend on the order of queries); bounded by
  // scoped_set_depth nested scoped queries.
  std::unordered_map<std::vector<Id>, ValueSet, VecHash> scoped_sets;
  static constexpr unsigned scoped_set_depth = 16;
  // Whether scope binds every free index level of raw node `node` (so that it has an import key there).
  bool bound_in(Id node, const Env &scope) const {
    return std::all_of(free[node].begin(), free[node].end(), [&](Id level) { return scope.contains(level); });
  }
  // The hardware value set of raw edge e, in import scope `scope` (or in every scope when null): an
  // imported infinity is that infinity, a Select that this scope decided is its chosen arm, a Negate
  // negates its operand's set in this scope, an instruction image imported as a constant is evaluated
  // from its operands in this scope; any other value has its scope-independent set.
  ValueSet edge_set(const Edge &e, const Env *scope, unsigned depth) {
    const auto &n = ns[e.node];
    if (n.kind == "Constant") return constant_set(e.node, e.conversion);
    if (scope && depth < scoped_set_depth && bound_in(e.node, *scope))
      if (const Id *found = cache.find(key(e.node, *scope))) {
        const auto &value = a.nodes[*found];
        if (value.op == "infinity") return converted(infinite(value.data[0] < 0 ? -1 : 1), e.conversion);
        if (n.kind == "Operation" && n.name == "Select" && n.operands.size() == 3)
          if (const Id *condition = cache.find(key(n.operands[0].node, *scope))) {
            Id decided = edge(*condition, n.operands[0]);
            if (context) decided = context->apply(a, decided);
            if (decided == a.one || decided == a.zero)
              return converted(edge_set(n.operands[decided == a.one ? 1 : 2], scope, depth + 1), e.conversion);
          }
        if (n.kind == "Operation" && n.name == "Negate" && n.operands.size() == 1)
          return converted(node_set(n, scope, depth + 1), e.conversion);
        if (value.op == "const" && sentryward::scalar_fold::instruction_mode::instruction(n.instruction)) {
          auto k = key(e.node, *scope);
          k.push_back(Id(depth));
          auto memo = scoped_sets.find(k);
          if (memo == scoped_sets.end()) memo = scoped_sets.emplace(std::move(k), node_set(n, scope, depth + 1)).first;
          return converted(memo->second, e.conversion);
        }
      }
    return converted(raw_set(e.node), e.conversion);
  }
  // The hardware value set of raw node n from its direct operands' sets (edge_set, in `scope`). An
  // add or sub without rounding modifier that uses a mul without rounding modifier (directly or
  // through the Negate of a sub) may be contracted into an fma (PTX ISA, mul): its set is then also
  // the exact product plus the other addend, rounded once.
  ValueSet node_set(const RawNode &n, const Env *scope, unsigned depth) {
    namespace mode = sentryward::scalar_fold::instruction_mode;
    const auto format = float_format(n.type);
    if (!format) return {};
    const auto operand = [&](size_t j) { return edge_set(n.operands[j], scope, depth); };
    const bool flush = !mode::instruction(n.instruction) || (n.instruction & mode::flush);
    if (n.kind == "Constant") return parse_constant(n, "");
    if (n.kind == "Fold") {
      // The body is imported under the Fold's binder: its scope-independent set holds for every step.
      if (n.operands.size() != 2 || (n.reducer != "Max" && n.reducer != "Min")) return {};
      const auto initial = flushed(edge_set(n.operands[0], scope, depth), *format);
      if (n.count == 0) return initial;
      return flushed(extremum({initial, flushed(edge_set(n.operands[1], nullptr, depth), *format)},
                              n.reducer == "Max", false), *format);
    }
    if (n.kind != "Operation") return {};
    const auto &name = n.name;
    const size_t arity = n.operands.size();
    if (name == "Select" && arity == 3) return hull(operand(1), operand(2));
    if (name == "Negate" && arity == 1) return negated(flush ? flushed(operand(0), *format) : operand(0));
    if ((name == "Maximum" || name == "Minimum") && arity >= 1) {
      std::vector<ValueSet> sets;
      for (size_t j = 0; j < arity; ++j) sets.push_back(flush ? flushed(operand(j), *format) : operand(j));
      const bool propagate = n.attributes.find("propagate_nan") != std::string::npos;
      const auto r = extremum(std::move(sets), name == "Maximum", propagate);
      return flush ? flushed(r, *format) : r;
    }
    if (!mode::instruction(n.instruction)) return {};
    const auto input = [&](size_t j) { return flush ? flushed(operand(j), *format) : operand(j); };
    const auto round = [&](const ValueSet &s) { return rounded(s, *format, n.instruction, flush); };
    if (name == "Exp2" && arity == 1) {
      // ex2.approx: -Inf gives +0.0 and +Inf gives +Inf (PTX ISA, ex2); a finite argument gives a
      // value within 2 ulp of the correctly rounded power, never below zero.
      const auto x = input(0);
      if (single_infinity(x) < 0) return exactly(Q(0));
      if (single_infinity(x) > 0) return infinite(1);
      return {{0, Q(0)}, {1}, x.nan};
    }
    if (name == "Multiply" && arity == 2) return round(product(input(0), input(1)));
    if (name == "Fma" && arity == 3) return round(sum(product(input(0), input(1)), input(2)));
    if ((name != "Add" && name != "Subtract") || arity != 2) return {};
    ValueSet first = input(0), second = input(1);
    if (name == "Subtract") second = negated(second);
    ValueSet r = round(sum(first, second));
    if (!(n.instruction & mode::contractible)) return r;
    for (size_t j = 0; j < 2; ++j) {
      // The mul this operand reads, through the Negate of a sub (unconverted edges only).
      const Edge *edge_to = &n.operands[j];
      bool negative = name == "Subtract" && j == 1;
      if (edge_to->conversion.empty() && ns[edge_to->node].kind == "Operation" && ns[edge_to->node].name == "Negate" &&
          ns[edge_to->node].operands.size() == 1) {
        edge_to = &ns[edge_to->node].operands[0];
        negative = !negative;
      }
      const auto &mul = ns[edge_to->node];
      if (!edge_to->conversion.empty() || mul.kind != "Operation" || mul.name != "Multiply" ||
          mul.operands.size() != 2 || !mode::instruction(mul.instruction) || !(mul.instruction & mode::contractible) ||
          mul.type != n.type) continue;
      const bool mul_flush = mul.instruction & mode::flush;
      auto factor = [&](size_t i) {
        const auto s = edge_set(mul.operands[i], scope, depth);
        return mul_flush ? flushed(s, *format) : s;
      };
      const auto exact = product(factor(0), factor(1));
      r = hull(r, round(sum(negative ? negated(exact) : exact, j == 0 ? second : first)));
    }
    return r;
  }
  // The taint of introduced infinities (docs/reference-binding-json.md, numerical model): the import
  // keys whose import may be, or hold the bits of, an infinity that the hardware value rule
  // introduced in place of a value the model computes as finite. One mechanism on every import path:
  // a key is tainted when it carries a tainted data operand on (carry_hardware_infinity; a Select's
  // arms, not its condition) or when the rule concludes an infinity that does not follow from literal
  // infinities alone (hardware_value). Whatever cannot pass a tainted operand on exactly stops the
  // family (hardware_decline). Touched only while some key is tainted.
  std::unordered_set<std::vector<Id>, VecHash> hardware_infinity_keys;
  std::vector<Id> taint_probe;
  bool hardware_infinity(Id node, const Env &env) {
    if (hardware_infinity_keys.empty() || !bound_in(node, env)) return false;
    key(taint_probe, node, env);
    return hardware_infinity_keys.contains(taint_probe);
  }
  bool hardware_infinity(const Edge &e, const Env &env) { return hardware_infinity(e.node, env); }
  // The data operands of raw node `raw` whose keys are tainted (a Select's arms, a Fold's element in
  // scope `body`); empty when none is. O(operands).
  std::vector<uint8_t> tainted_operands(Id raw, const Env &env, const Env &body) {
    std::vector<uint8_t> tainted;
    if (hardware_infinity_keys.empty()) return tainted;
    const auto &n = ns[raw];
    const bool fold = n.kind == "Fold";
    const size_t len = fold && n.count == 0 ? 1 : n.operands.size();
    for (size_t j = n.kind == "Operation" && n.name == "Select" ? 1 : 0; j < len; ++j)
      if (hardware_infinity(n.operands[j], fold && j == 1 ? body : env)) {
        tainted.resize(len);
        tainted[j] = 1;
      }
    return tainted;
  }
  static bool comparison(const RawNode &n) {
    return n.type == "pred" || (n.kind == "Operation" && (n.name == "Compare" || n.name == "Less" ||
           n.name == "LessEqual" || n.name == "Greater" || n.name == "GreaterEqual" || n.name == "Equal" ||
           n.name == "NotEqual"));
  }
  // Records the taint of import key k of raw node `raw`, imported as `result` from operands x of
  // which `tainted` are tainted, once hardware_value let the import proceed. A comparison or other
  // predicate carries none: it is exact on the infinity. A maximum (minimum) carries it from an
  // operand that may decide it, that is anything but -inf (+inf), and when its result is an
  // infinity. A floating instruction image whose value set hardware_value found finite carries none.
  // Every other node carries it on: a Select arm, a negation or absolute value, an integer or bit
  // operation (a Bitcast is an edge conversion of its operand), an empty Fold. O(operands).
  void carry_hardware_infinity(Id raw, const std::vector<Id> &k, const std::vector<Id> &x,
                               const std::vector<uint8_t> &tainted, Id result) {
    const auto &n = ns[raw];
    if (comparison(n)) return;
    const bool fold = n.kind == "Fold", floating = float_format(n.type).has_value();
    const bool maximum = fold ? n.reducer == "Max" : n.name == "Maximum";
    const bool extremum = maximum || (fold ? n.reducer == "Min" : n.name == "Minimum");
    if (floating && a.nodes[result].op != "infinity") {
      if (extremum) {
        bool decides = false;
        for (size_t j = 0; j < tainted.size() && j < x.size(); ++j)
          decides = decides || (tainted[j] && !(a.nodes[x[j]].op == "infinity" && (a.nodes[x[j]].data[0] < 0) == maximum));
        if (!decides) return;
      } else if (n.kind == "Operation" && n.name != "Select" && n.name != "Negate" && n.name != "Absolute") return;
    }
    hardware_infinity_keys.insert(k);
  }
  static std::string render_set(const ValueSet &s) {
    const auto end = [](const Extended &e) {
      return e.infinity ? std::string(e.infinity < 0 ? "-inf" : "inf") : str(e.value);
    };
    return "[" + end(s.lo) + ", " + end(s.hi) + "]" + (s.nan ? " or NaN" : "");
  }
  // Stops the family at raw node `raw` (imported operands x in scope env): a tainted operand reaches
  // an operation whose hardware value it does not determine (hardware_value_not_determined), with
  // each operand's value set.
  [[noreturn]] void hardware_decline(Id raw, const std::vector<Id> &x, const Env &env, const std::string &why,
                                     const std::optional<ValueSet> &set = std::nullopt) {
    namespace mode = sentryward::scalar_fold::instruction_mode;
    const auto &n = ns[raw];
    ++a.stats["hardware_value_declines"];
    J operands = J::array();
    for (size_t j = 0; j < n.operands.size() && j < x.size(); ++j) {
      const auto &o = ns[n.operands[j].node];
      operands.push_back({{"kind", o.kind}, {"name", o.name}, {"type", o.type},
                          {"instruction", mode::instruction(o.instruction)},
                          {"value_set", render_set(edge_set(n.operands[j], n.kind == "Fold" && j == 1 ? nullptr : &env, 0))},
                          {"imported", a.render(x[j], 2)}});
    }
    J details{{"operation", n.kind == "Fold" ? "Fold " + n.reducer : n.name}, {"type", n.type},
              {"instruction", mode::instruction(n.instruction)}, {"operands", operands}};
    if (set) details["value_set"] = render_set(*set);
    throw Unsupported("An infinity reaches an operation whose hardware value it does not determine (" + why + ")",
                      "hardware_value_not_determined", details);
  }
  // The hardware value rule at raw node `raw` with imported operands x in scope env, of which
  // `tainted` are tainted: an operation whose hardware value set is one infinity imports as that
  // infinity (counter hardware_infinities). A tainted operand reaching an add, sub, mul, fma, ex2 or
  // other arithmetic whose hardware value it does not determine (an instruction image whose set is not
  // finite, or any operation that is no instruction image), a Sum Fold or a .NaN max/min that a NaN
  // may decide stops the family (hardware_decline). An infinity from a literal keeps the model's
  // extended-real handling there, as before. Select, Negate, Absolute, max/min (a deciding infinity is
  // the result), Max/Min Folds and comparisons pass an infinity on, and so do conversions, which are
  // edges (carry_hardware_infinity); an ex2.approx of -Inf is +0.0. Evaluated for an instruction image
  // only when an operand is an imported constant or infinity or tainted. Otherwise none: the model
  // imports the node as before.
  std::optional<Id> hardware_value(Id raw, const std::vector<Id> &x, const std::vector<uint8_t> &tainted,
                                   const Env &env) {
    if (!hardware_values || !ideal) return {};
    namespace mode = sentryward::scalar_fold::instruction_mode;
    const auto &n = ns[raw];
    if ((n.kind != "Operation" && n.kind != "Fold") || !float_format(n.type)) return {};
    const bool infinite_operand =
        std::any_of(x.begin(), x.end(), [&](Id v) { return a.nodes[v].op == "infinity"; });
    const bool constant_operand = std::any_of(x.begin(), x.end(), [&](Id v) { return a.value(v).has_value(); });
    const bool introduced_operand = !tainted.empty();
    const auto is_tainted = [&](size_t j) { return j < tainted.size() && tainted[j]; };
    const auto introduce = [&](int sign) {
      ++a.stats["hardware_infinities"];
      // An infinity that follows from literal infinities alone (an untainted infinite operand, no
      // tainted one) is the value the extended-real model gives: it keeps literal status.
      bool literal = false;
      for (size_t j = 0; j < x.size(); ++j) literal = literal || (a.nodes[x[j]].op == "infinity" && !is_tainted(j));
      if (introduced_operand || !literal) hardware_infinity_keys.insert(key(raw, env));
      return a.infinity(sign);
    };
    if (n.kind == "Fold") {
      if (introduced_operand && n.count > 0 && n.reducer != "Max" && n.reducer != "Min")
        hardware_decline(raw, x, env, "a reassociated reduction");
      return {};
    }
    const auto &name = n.name;
    const bool rounding = name == "Add" || name == "Subtract" || name == "Multiply" || name == "Fma";
    if (name == "Maximum" || name == "Minimum") {
      // max_num(+inf, x) is +inf, min_num(-inf, x) is -inf, also for a NaN x; the other infinity
      // never decides and the model drops it. With .NaN a NaN operand decides instead.
      if (!infinite_operand && !introduced_operand) return {};
      const auto s = node_set(n, &env, 0);
      if (const int sign = single_infinity(s)) return introduce(sign);
      if (s.nan && n.attributes.find("propagate_nan") != std::string::npos)
        for (size_t j = 0; j < x.size(); ++j)
          if (is_tainted(j) && !(a.nodes[x[j]].op == "infinity" && (a.nodes[x[j]].data[0] < 0) == (name == "Maximum")))
            hardware_decline(raw, x, env, "a NaN operand of max.NaN or min.NaN may decide it", s);
      return {};
    }
    if (mode::instruction(n.instruction) && (rounding || name == "Exp2") &&
        (infinite_operand || constant_operand || introduced_operand)) {
      const auto s = node_set(n, &env, 0);
      if (const int sign = single_infinity(s)) return introduce(sign);
      if (introduced_operand && !finite_set(s)) hardware_decline(raw, x, env, "its hardware value set is not finite", s);
      return {};
    }
    if (introduced_operand && name != "Select" && name != "Negate" && name != "Absolute")
      hardware_decline(raw, x, env, "it is no PTX instruction image");
    return {};
  }
  // At a Maximum (Minimum), an operand imported as the type's most negative (largest) finite value
  // -MAX (MAX) never decides it in hardware when another operand's hardware value set lies at or above
  // -MAX (at or below MAX); it is dropped, without the range premise of in_range. Records
  // range_bounds_proved. O(operands) besides the memoized sets.
  std::vector<Id> proved_in_range(std::vector<Id> x, const RawNode &n, bool maximum, const Env &env) {
    const auto format = float_format(n.type);
    if (!hardware_values || !format || x.size() != n.operands.size()) return x;
    const Q bound = maximum ? -largest_finite(*format) : largest_finite(*format);
    std::vector<uint8_t> at_bound(x.size());
    for (size_t j = 0; j < x.size(); ++j)
      if (const auto v = a.value(x[j]); v && *v == bound) at_bound[j] = 1;
    if (std::none_of(at_bound.begin(), at_bound.end(), [](uint8_t b) { return b; })) return x;
    bool dominated = false;
    const Extended limit{0, bound};
    for (size_t j = 0; j < x.size() && !dominated; ++j) {
      if (at_bound[j]) continue;
      const auto s = edge_set(n.operands[j], &env, 0);
      dominated = !s.nan && (maximum ? !before(s.lo, limit) : !before(limit, s.hi));
    }
    if (!dominated) return x;
    std::vector<Id> kept;
    for (size_t j = 0; j < x.size(); ++j)
      if (!at_bound[j]) kept.push_back(x[j]);
    a.stats["range_bounds_proved"] += x.size() - kept.size();
    return kept;
  }
  // A comparison of a Select of two constants with a constant, decided arm by arm: the Select's
  // Boolean condition, its complement 1 - p, or a constant. This is a predicate carried through an
  // integer (selp 1, 0, then setp.ne 0) or any other constant pair. Exact. O(1).
  std::optional<Id> selected_comparison(const std::string &op, Id x, Id y) {
    if (!Arena::valid_comparison(op)) return {};
    for (unsigned side = 0; side != 2; ++side) {
      const Id chosen = side ? y : x;
      const auto other = a.value(side ? x : y);
      const auto n = a.nodes[chosen];
      if (!other || n.op != "select" || n.args.size() != 3) continue;
      const auto yes = a.value(n.args[1]), no = a.value(n.args[2]);
      const Id condition = n.args[0];
      const auto &bounds = a.nodes[condition].bounds;
      if (!yes || !no || !bounds || bounds->first < 0 || bounds->second > 1 ||
          !a.known_bits(condition).integer) continue;
      const auto holds = [&](const Q &arm) {
        const Q &l = side ? *other : arm, &r = side ? arm : *other;
        return op == "eq" ? l == r : op == "ne" ? l != r : op == "lt" ? l < r :
               op == "le" ? l <= r : op == "gt" ? l > r : l >= r;
      };
      const bool when_true = holds(*yes), when_false = holds(*no);
      ++a.stats["constant_select_comparisons"];
      if (when_true == when_false) return when_true ? a.one : a.zero;
      if (when_true) return condition;
      // The complement of a comparison of finite values is the complementary comparison.
      const auto p = a.nodes[condition];
      if (p.op == "compare" && p.args.size() == 2 && p.data.empty() &&
          a.nodes[p.args[0]].facts.finite && a.nodes[p.args[1]].facts.finite)
        return a.compare(Arena::complement_comparison(p.tag), p.args[0], p.args[1]);
      return a.sub(a.one, condition);
    }
    return {};
  }
  // Widest integer word whose AND of two non-constant operands is expanded bit by bit (PTX and.b64).
  static constexpr unsigned bitwise_and_width_limit = 64;
  // Selections and sums bit_of descends through before it slices the word itself, and the most
  // terms of a sum it splits (a w-bit word has at most w disjoint nonzero fields, plus a constant).
  static constexpr unsigned bit_of_depth_limit = 8, bit_of_sum_terms = bitwise_and_width_limit + 1;
  // A finite integer proved to lie in [0, 2^w) (bounds first; known bits only then).
  bool whole_word(Id z, unsigned w) {
    const auto bounds = a.nodes[z].bounds;
    return a.nodes[z].facts.finite && bounds && bounds->first >= 0 && bounds->second < Q(pow2(w)) &&
           a.known_bits(z).integer;
  }
  // Bit b of an integer word v in [0, 2^w), exactly: a constant's bit; for a Select of two such words,
  // the Select of their bits; for the w-bit complement 2^w - 1 - z of such a word z, one minus z's
  // bit; for a sum of nonnegative integer terms whose possible one bits (known_bits) are pairwise
  // disjoint, which is their OR, the bit of the one term (or constant) that may set it; otherwise
  // the arena's bit slice. O(depth * terms) known-bits lookups.
  Id bit_of(Id v, unsigned b, unsigned w, unsigned depth = 0) {
    const auto in_word = [&](Id z) { return whole_word(z, w); };
    if (auto value = a.value(v); value && value->denominator() == 1 && *value >= 0)
      return a.constant(Q(modz(floorz(value->numerator(), pow2(b)), Z(2))));
    const auto n = a.nodes[v];
    if (n.op == "select" && n.args.size() == 3 && depth < bit_of_depth_limit &&
        in_word(n.args[1]) && in_word(n.args[2])) {
      const Id yes = bit_of(n.args[1], b, w, depth + 1), no = bit_of(n.args[2], b, w, depth + 1);
      return a.select(n.args[0], yes, no, true);
    }
    if (n.op != "add" || depth >= bit_of_depth_limit || n.args.size() > bit_of_sum_terms)
      return a.bit_slice(v, b, 1);
    Q offset = 0;
    std::vector<std::pair<Id, Q>> parts;
    for (const auto &[t, c] : a.terms(v)) {
      if (t == a.one) offset += c;
      else parts.emplace_back(t, c);
    }
    const auto all = [&](auto predicate) {
      return std::all_of(parts.begin(), parts.end(), [&](const auto &p) { return predicate(p.second); });
    };
    if (offset == Q(pow2(w) - 1) && all([](const Q &c) { return c < 0; })) {
      std::vector<Id> negated;
      for (const auto &[t, c] : parts) negated.push_back(a.mul({a.constant(-c), t}));
      const Id z = a.add(std::move(negated));
      if (in_word(z)) return a.sub(a.one, bit_of(z, b, w, depth + 1));
    }
    if (offset >= 0 && offset.denominator() == 1 &&
        all([](const Q &c) { return c > 0 && c.denominator() == 1; })) {
      const Z mask = pow2(w) - 1;
      Z taken = offset.numerator();
      Id owner = invalid;
      bool disjoint = taken <= mask;
      for (const auto &[t, c] : parts) {
        const Id term = c == 1 ? t : a.mul({a.constant(c), t});
        if (!in_word(term)) { disjoint = false; break; }
        const Z possible = mask ^ (a.known_bits(term).zero & mask);
        if ((possible & taken) != 0) { disjoint = false; break; }
        taken |= possible;
        if (((possible >> b) & 1) != 0) owner = term;
      }
      if (disjoint) {
        if (((offset.numerator() >> b) & 1) != 0) return a.one;
        return owner == invalid ? a.zero : bit_of(owner, b, w, depth + 1);
      }
    }
    return a.bit_slice(v, b, 1);
  }
  // x & y of an integer type whose operands are both non-constant: the sum over the bits b of
  // 2^b * bit(x, b) * bit(y, b), exact for the two w-bit words. O(w) bit products.
  std::optional<Id> bitwise_and(Id x, Id y, const std::string &t) {
    const unsigned w = width(t);
    if (w == 0 || w > bitwise_and_width_limit) return {};
    const Id lhs = word(x, w), rhs = word(y, w);
    std::vector<Id> terms;
    for (unsigned b = 0; b < w; ++b) {
      const Id l = bit_of(lhs, b, w);
      if (l == a.zero) continue;
      const Id r = bit_of(rhs, b, w);
      if (r == a.zero) continue;
      terms.push_back(a.mul({a.constant(Q(pow2(b))), l, r}));
    }
    ++a.stats["bitwise_and_bit_products"];
    const Id result = a.add(std::move(terms));
    return t.starts_with("s") ? cast(result, "u" + std::to_string(w), t) : result;
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
      if ((t.starts_with("b") && !t.starts_with("bf")) || t.starts_with("u")) {
        if (auto fixed = fixed_bits(name, x[0], width(t))) return *fixed;
        return a.intern("readbits", x, {Q(width(t))}, name, {true, false, true},
                        std::pair(Q(0), Q(pow2(width(t)) - 1)));
      }
      return typed_read(name, x[0], t);
    }
    if (k == "Fold") {
      const auto &f = n;
      Id var = body.at(f.level);
      Q count = Q(f.count);
      const std::string &reducer = f.reducer;
      if (reducer == "Min") {
        // Ideal-real minimum shares the maximum normal form, as Minimum does:
        // min(s, b_0, ..., b_{n-1}) = -max(-s, -b_0, ..., -b_{n-1}) exactly,
        // for integers and extended reals (the +inf identity becomes -inf).
        Id initial = a.neg(x.at(0)), element = a.neg(x.at(1));
        return a.neg(a.intern("fold", {initial, element}, {Q(var), count}, "Max",
                              a.fold_facts(initial, element, count, true, var)));
      }
      // The PTX side's artifact bypasses the document reader's reducer check;
      // every rule below interprets a non-Max fold as a sum.
      if (reducer != "Sum" && reducer != "Max")
        throw Unsupported("unsupported Fold reducer " + reducer);
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
      // The low w bits of an integer already in [0, 2^w) are the integer itself.
      if(*off==0 && whole_word(storage,w))return storage;
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
    if ((name == "BitAnd" || name == "BitOr" || name == "BitXor") && t == "pred") {
      const std::string logic=name=="BitAnd"?"AND":name=="BitOr"?"OR":"XOR";
      if(x.size()!=2) throw Unsupported("predicate "+logic+" arity");
      for(Id v:x) if(!a.nodes[v].bounds || a.nodes[v].bounds->first<0 || a.nodes[v].bounds->second>1)
        throw Unsupported("predicate "+logic+" operand outside Boolean range");
      // On {0,1}: x&y = xy, x|y = 1-(1-x)(1-y) (De Morgan over NOT = 1-x), x^y = (x|y)(1-xy).
      // Each form is exact and keeps the result's bounds within [0,1].
      const Id both=a.mul(x);
      if(name=="BitAnd")return both;
      const Id either=a.sub(a.one,a.mul({a.sub(a.one,x[0]),a.sub(a.one,x[1])}));
      return name=="BitOr"?either:a.mul({either,a.sub(a.one,both)});
    }
    if (name == "BitAnd") {
      auto m = a.value(x.at(1));
      if ((!m || m->denominator() != 1) && integer)
        if (auto both = bitwise_and(x.at(0), x.at(1), t)) return *both;
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
      if (auto log = compiler_log(n, x, ctx)) return *log;
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
      return a.maximum(in_range(proved_in_range(x, n, true, ctx), n.type, true));
    if (name == "Minimum") {
      // Ideal-real minimum shares the maximum normal form.
      std::vector<Id> negatives;
      for (Id operand : in_range(proved_in_range(x, n, false, ctx), n.type, false))
        negatives.push_back(a.neg(operand));
      return a.neg(a.maximum(std::move(negatives)));
    }
    if (name == "Compare") {
      std::smatch m;
      static const std::regex re("predicate=(\\w+)");
      if (!std::regex_search(attr, m, re))
        throw Unsupported("missing predicate metadata");
      if (auto decided = selected_comparison(m[1], x.at(0), x.at(1))) return *decided;
      return a.compare(m[1], x.at(0), x.at(1));
    }
    static const std::map<std::string, std::string> cmp{
        {"Less", "lt"},         {"LessEqual", "le"}, {"Greater", "gt"},
        {"GreaterEqual", "ge"}, {"Equal", "eq"},     {"NotEqual", "ne"}};
    if (cmp.contains(name)) {
      if (auto decided = selected_comparison(cmp.at(name), x.at(0), x.at(1))) return *decided;
      return a.compare(cmp.at(name), x.at(0), x.at(1));
    }
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
