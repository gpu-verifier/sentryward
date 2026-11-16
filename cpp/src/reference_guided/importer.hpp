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
#include "import_memo.hpp"
#include "import_types.hpp"
#include <regex>
namespace rg {
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
  // premise_e8m0_byte results per (node, bit offset of the byte).
  std::map<std::pair<Id,unsigned>,Id> premise_e8m0_bytes;
  // A verified lane template of a packed record (record_lane_template): lane k is
  // templates[k mod period] at class index k div period, the variable `index`. period 0: none.
  struct LaneTemplate { unsigned period = 0; std::vector<Id> templates; Id index = invalid; };
  // record_lane_template results per (record, lane width, first lane, last lane).
  std::map<std::tuple<Id,unsigned,Z,Z>,LaneTemplate> record_lane_templates;
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
  std::unordered_map<std::vector<Id>, Id, IdSequenceHash> sibling_binders;
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
  bool fp4_reference_candidate(const RawNode &n) const;
  Importer(Arena &ar, Document d, const J &mp, std::string s, const PreparedContext &ctx)
      : Importer(ar,std::move(d),mp,std::move(s),&ctx.contract) { context=&ctx; }
  std::span<const Id> key(std::vector<Id> &k, Id i, const Env &env) const;
  std::vector<Id> key(Id i, const Env &env) const;
  Id fold_binder(Id raw, const Env &env);
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
  Id occurrence_binder(Id raw, const Env &env);
  // Byte provenance only: invalid denotes a proven zero byte. A source ID
  // denotes an opaque quantized e4m3 byte, never its ideal-real numeric value.
  using Packed = std::vector<Id>;
  std::map<Id, Packed> packed_cache;
  static bool unsigned_word(const std::string &t);
  unsigned pack_constant(const Edge &e);
  Packed packed_edge(const Edge &e, unsigned depth = 0);
  Packed packed_node(Id id, unsigned depth);
  std::optional<Id> packed_output(const J &body, const Env &env);
  // One output coordinate fixed at a value: the family's instance at that value.
  using OutputPoint = std::optional<std::pair<size_t, Z>>;
  Env family_environment(const J &f, const OutputPoint &point = {});
  Id family(const J &f, const std::string &expected_type = "", const OutputPoint &point = {});
  std::string family_type(const J &f) const;
  // A readbits node addresses raw bytes of this side's source layout. The same raw address denotes
  // different elements on the two sides when their source layouts of one shared input differ, so
  // views of such inputs are compared only at canonical (target-layout) addresses.
  // Per raw layout key of this side: whether the other side lays out the same input differently.
  std::unordered_map<std::string,bool> byte_layout_differs;
  std::optional<bool> any_byte_layout_differs;
  // Per Arena node: already scanned for raw views of such inputs (shared by this side's families).
  std::vector<uint8_t> byte_view_scanned;
  bool source_layout_differs(const std::string &raw);
  std::optional<Id> canonical_byte_address(const std::string &raw,Id address,const Z &bytes_read);
  bool some_byte_layout_differs();
  Id refuse_raw_byte_views(Id n);
  Id byte_view_output(Id n);
  std::optional<Id> fixed_bits(const std::string &raw,Id offset,unsigned bits);
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
  bool unrolls_fixed_table_fold(Id raw);
  // A stacked reduction (torch.stack(...).sum/amax): the element depends on the binder only through
  // Select(binder == constant, A, rest) chains whose arms do not depend on it. Unrolling is the Fold's
  // definition; the element is imported once per binder value, at most stacked_fold_unroll_limit times.
  static constexpr uint64_t stacked_fold_unroll_limit=64;
  bool unrolls_stacked_fold(Id raw) const;
  // Per first_argmax node built from a reference Fold: that reference's Max fold over the same values.
  std::unordered_map<Id,Id> argmax_maxima;
  // Per imported node: the read of an input at a first_argmax coordinate it contains (one per node).
  // Constants are not recorded: they contain no read.
  std::unordered_map<Id,Id> argmax_reads;
  // The most distinct nodes one replace_read visits; past it the gather is declined.
  static constexpr size_t argmax_replace_visit_limit=size_t(1)<<16;
  std::optional<Id> replace_read(Id root,Id from,Id to);
  Id gathered_at_argmax(const RawNode &n,const std::vector<Id> &xs,Id result);
  // Alpha identity of imported instances (binder names differ between the two scans of B).
  FoldAlphaIdentity argmax_alpha{a};
  // Raw shape of a first argmax: Fold(Min, init, Select(Equal(B, M), j, c)) over binder j, where M is a
  // Fold(Max) over its own binder of the same count that does not depend on j, and neither the Min's
  // element nor the Select's condition is converted. Raw structure only.
  // first_argmax_fold imports B and M's element once per index: an axis longer than this (a
  // vocabulary-sized argmax, say) keeps the generic Fold import instead of 2 x count imports.
  static constexpr uint64_t first_argmax_unroll_limit=4096;
  bool first_argmax_fold_pattern(Id raw) const;
  std::optional<Id> first_argmax_fold(Id raw,const Env &env,std::vector<std::pair<Id,Env>> &missing);
  bool reads_fixed_table_at(Id element,Id binder);
  std::optional<Id> unrolled_fold(Id raw,const Env &env,std::vector<std::pair<Id,Env>> &missing,
                                  const char *counter="fixed_table_folds_unrolled");
  std::optional<std::pair<Id,std::string>> finite_scale_guard(const RawNode &select) const;
  // The integer interpretations (s32/u32) under which the other side's document reads each shared
  // input allocation (integer_reads of that document); set by the comparison. Only
  // typed_word_predicate consults it.
  std::map<std::string, std::set<std::string>> paired_integer_reads;
  static std::map<std::string, std::set<std::string>> integer_reads(const Document &d, const J &side_inputs);
  std::optional<Id> typed_word_predicate(const std::string &op, Id x, Id y);
  std::optional<Id> contracted_word(const std::string &raw, Id offset, unsigned bits);
  Id import(Id root, const Env &env);
  std::optional<Id> compiler_lse(const RawNode &root, const Env &env);
  // The round-to-nearest value of ln 2 and of log2(e) in a binary format, and the round-to-nearest
  // value of the residual c - RN(c) (the low part of a high/low split constant).
  struct RoundedConstant { Q high, low; bool ln2; };
  static const std::vector<RoundedConstant> &rounded_constants(const std::string &type) {
    static const std::vector<RoundedConstant> binary32{
        {Q(1453635, 2097152), Q(Z(-1072225), Z("562949953421312")), true},
        {Q(12102203, 8388608), Q(Z(338819), Z("17592186044416")), false}};
    static const std::vector<RoundedConstant> binary64{
        {Q(Z("6243314768165359"), Z("9007199254740992")),
         Q(Z("7525737178955839"), Z("324518553658426726783156020576256")), true},
        {Q(Z("3248660424278399"), Z("2251799813685248")),
         Q(Z("1651415998432073"), Z("81129638414606681695789005144064")), false}};
    static const std::vector<RoundedConstant> none;
    return type == "f32" ? binary32 : type == "f64" ? binary64 : none;
  }
  // A floating arithmetic operation; make() imports it as ideal arithmetic whatever its rounding or
  // flush-to-zero attributes (ideal_real), so the rounded-constant rule takes the same operations.
  // Reciprocal included: the scalar fold writes div.approx x, y as x * Reciprocal(y).
  static bool ideal_arithmetic_operation(const RawNode &n) {
    return n.kind == "Operation" && (n.type == "f32" || n.type == "f64") &&
           (n.name == "Add" || n.name == "Subtract" || n.name == "Multiply" || n.name == "Fma" ||
            n.name == "Divide" || n.name == "Reciprocal");
  }
  std::optional<Q> literal_operand(const Edge &e);
  std::optional<std::vector<Id>> ideal_rounded_operands(const RawNode &n, const std::vector<Id> &xs);
  bool split_constant_low_part(const RawNode &n);
  Id word(Id x, unsigned w);
  Id unwrap(Id x, unsigned w);
  std::optional<Id> initial_bit_slice(Id x, unsigned lo, unsigned count, unsigned depth=0);
  std::optional<Id> initial_typed_bits(Id x,const std::string &type);
  Id typed_ue4m3_input(const std::string &raw, Id offset);
  Id typed_e8m0_input(const std::string& raw,Id offset);
  bool region_covers(const InputRegion &region,const std::string &raw,const std::vector<Id> &coordinates,unsigned bytes);
  void require_contract_coverage(const InputContract::Rule &rule,const std::string &raw,Id offset,unsigned bytes);
  std::optional<Id> premise_e8m0_byte(Id x,unsigned lo=0,unsigned depth=0);
  std::optional<Id> floating_lane(Id source,const std::string& type);
  bool transport_integer(Id y);
  std::optional<unsigned> replicated_copy(const Q& c,Id y,unsigned lo,unsigned width);
  std::optional<std::pair<Id,unsigned>> disjoint_sum_field(Id sum,unsigned lo,unsigned width);
  Id exact_bit_field(Id x,unsigned lo,unsigned width);
  std::optional<Id> symbolic_byte_permute(Id x,Id y,Id control);
  std::optional<Id> sliced_selection(Id x,unsigned width);
  Id rebuild_node(const std::string& op,const std::string& tag,const std::vector<Q>& data,const Facts& facts,
                  const Bounds& bounds,std::vector<Id> xs,bool existing_reads=false);
  Id instantiate(Id root,Id var,Id value,bool existing_reads=false,size_t budget=4096);
  Id lane_anti_unify(Id A,Id B,Id x,const Z &k0,std::map<std::pair<Id,Id>,Id> &memo);
  LaneTemplate record_lane_template(Id record,const std::vector<Id>& lanes,unsigned w,const Z& k0,const Z& k1);
  std::optional<Id> dynamic_record_lane(Id x,const std::string& dst,unsigned w);
  Id cast(Id x, const std::string &src, const std::string &dst,
          bool bitcast = false);
  Id edge(Id x, const J &e, bool output = false) {
    return edge_conversion(x, e.value("conversion", ""), output);
  }
  Id edge(Id x, const Edge &e, bool output = false) {
    return edge_conversion(x, e.conversion, output);
  }
  Id edge_conversion(Id x, const std::string &conv, bool output = false);
  Id e2m1_bits(Id y);
  std::optional<Id> fixed_header_float(Id x, unsigned w, const std::string &dst);
  std::optional<Id> exponent_field_power(Id x, const std::string &src, const std::string &dst, bool refuse);
  Id literal(const std::string &s, const std::string &t);
  std::optional<PackedDecodePass::Bits> canonical_packed_slice(Id source,unsigned lo,unsigned count);
  Id normalize_packed_decoder(Id result) {
    return packed_decoders.normalize(result,[&](Id source,unsigned lo,unsigned count) {
      return canonical_packed_slice(source,lo,count);
    });
  }
  Id typed_integer_fact(Id value, const std::string &type);
  Id typed_read(const std::string &raw, Id offset, const std::string &t, bool require_valid = false);
  std::optional<Id> reconstruct(const std::vector<Id> &xs);
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
  Id mask32(Id x, uint32_t mask);
  Id byte_permute(Id x, Id y, unsigned control);
  Id logic3_constants(Id x, uint32_t b, uint32_t c, unsigned lut);
  Id integer_index_quotient(Id dividend, const Z &divisor,
                            unsigned work_limit = PackedDecodePass::max_nodes);

  std::vector<Id> in_range(std::vector<Id> x, const std::string &type, bool maximum);
  // Hardware values of the PTX side (docs/reference-binding-json.md, numerical model). An IEEE
  // binary format: precision p (significand bits, the hidden bit included), largest exponent emax
  // and smallest normal exponent -min_normal_shift.
  struct FloatFormat { unsigned precision, max_exponent, min_normal_shift; };
  static std::optional<FloatFormat> float_format(const std::string &t);
  // The largest finite magnitude MAX = 2^(emax+1) - 2^(emax+1-p).
  static Q largest_finite(const FloatFormat &f) {
    return Q(pow2(f.max_exponent + 1) - pow2(f.max_exponent + 1 - f.precision));
  }
  // Round to nearest even carries a result to infinity from the midpoint 2^(emax+1) - 2^(emax-p)
  // between MAX and 2^(emax+1) on (IEEE 754-2019, 4.3.1 and 7.4).
  static Q overflow_threshold(const FloatFormat &f) {
    return Q(pow2(f.max_exponent + 1) - pow2(f.max_exponent - f.precision));
  }
  static bool representable(const Q &q, const FloatFormat &f);
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
  static ValueSet sum(const ValueSet &x, const ValueSet &y);
  static ValueSet product(const ValueSet &x, const ValueSet &y);
  static ValueSet flushed(ValueSet s, const FloatFormat &f);
  static Q on_grid(const Q &q, const FloatFormat &f, bool up);
  static ValueSet rounded(const ValueSet &s, const FloatFormat &f, uint8_t modes, bool flush);
  static ValueSet extremum(std::vector<ValueSet> sets, bool maximum, bool propagate);
  // The leading Bitcasts of `conversion` from `type`: the type they reach, the number of parts they
  // take and the remaining parts, or none when a Bitcast changes width or does not start at `type`.
  struct BitcastChain { std::string type, rest; size_t casts = 0; };
  static std::optional<BitcastChain> bitcast_chain(std::string type, const std::string &conversion);
  ValueSet parse_constant(const RawNode &n, const std::string &conversion);
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
  ValueSet converted(const ValueSet &s, const std::string &conversion);
  // Hardware value sets of raw floating values, independent of the import scope (memoized per raw
  // node): constants; Select (either arm), Negate, Maximum/Minimum and Max/Min Folds (an initial value
  // bounds a running maximum from below); and PTX instruction images (RawNode::instruction) of add,
  // sub, mul, fma and ex2.approx. Any other node admits every value, NaN included: an input element, a
  // reassociated or accumulated sum, an MMA, a division, a software function. The walk
  // enters only the operands that node_set reads (set_operands), so a node that admits every value
  // ends it. O(nodes and edges of the floating cones read) over all queries.
  std::unordered_map<Id, ValueSet> value_sets;
  void set_operands(Id id, std::vector<Id> &out) const;
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
  std::unordered_map<std::vector<Id>, ValueSet, IdSequenceHash> scoped_sets;
  static constexpr unsigned scoped_set_depth = 16;
  // Whether scope binds every free index level of raw node `node` (so that it has an import key there).
  bool bound_in(Id node, const Env &scope) const {
    return std::all_of(free[node].begin(), free[node].end(), [&](Id level) { return scope.contains(level); });
  }
  ValueSet edge_set(const Edge &e, const Env *scope, unsigned depth);
  ValueSet node_set(const RawNode &n, const Env *scope, unsigned depth);
  // The taint of introduced infinities (docs/reference-binding-json.md, numerical model): the import
  // keys whose import may be, or hold the bits of, an infinity that the hardware value rule
  // introduced in place of a value the model computes as finite. One mechanism on every import path:
  // a key is tainted when it carries a tainted data operand on (carry_hardware_infinity; a Select's
  // arms, not its condition) or when the rule concludes an infinity that does not follow from literal
  // infinities alone (hardware_value). Whatever cannot pass a tainted operand on exactly stops the
  // family (hardware_decline). Touched only while some key is tainted.
  std::unordered_set<std::vector<Id>, IdSequenceHash> hardware_infinity_keys;
  std::vector<Id> taint_probe;
  bool hardware_infinity(Id node, const Env &env) {
    if (hardware_infinity_keys.empty() || !bound_in(node, env)) return false;
    key(taint_probe, node, env);
    return hardware_infinity_keys.contains(taint_probe);
  }
  bool hardware_infinity(const Edge &e, const Env &env) { return hardware_infinity(e.node, env); }
  std::vector<uint8_t> tainted_operands(Id raw, const Env &env, const Env &body);
  static bool comparison(const RawNode &n) {
    return n.type == "pred" || (n.kind == "Operation" && (n.name == "Compare" || n.name == "Less" ||
           n.name == "LessEqual" || n.name == "Greater" || n.name == "GreaterEqual" || n.name == "Equal" ||
           n.name == "NotEqual"));
  }
  void carry_hardware_infinity(Id raw, const std::vector<Id> &k, const std::vector<Id> &x,
                               const std::vector<uint8_t> &tainted, Id result);
  static std::string render_set(const ValueSet &s);
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
  std::optional<Id> hardware_value(Id raw, const std::vector<Id> &x, const std::vector<uint8_t> &tainted,
                                   const Env &env);
  std::vector<Id> proved_in_range(std::vector<Id> x, const RawNode &n, bool maximum, const Env &env);
  std::optional<Id> selected_comparison(const std::string &op, Id x, Id y);
  std::pair<Id, Id> same_condition_arms(Id p, Id yes, Id no);
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
  Id bit_of(Id v, unsigned b, unsigned w, unsigned depth = 0);
  std::optional<Id> bitwise_and(Id x, Id y, const std::string &t);
  Id make(const RawNode &n, const std::vector<Id> &x, const Env &ctx,
          const Env &body);
};
} // namespace rg

#include "importer_binders.hpp"
#include "importer_inputs.hpp"
#include "importer_graph.hpp"
#include "importer_storage.hpp"
#include "importer_conversions.hpp"
#include "importer_reads.hpp"
#include "importer_hardware.hpp"
#include "importer_expressions.hpp"
