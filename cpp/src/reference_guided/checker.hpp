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
// A Call argument position (the axis excluded) and the integer interval its
// value had when the spec's Facts were derived.
struct SlotDomain {
  size_t argument;
  Q lo, hi;
  bool operator<(const SlotDomain &o) const {
    return std::tie(argument, lo, hi) < std::tie(o.argument, o.lo, o.hi);
  }
};
struct Spec {
  Id pattern;
  size_t slots, axis;
  Z count;
  Facts facts;
  bool maximum = false;
  // The Facts depend on these arguments' values. A Call has them only when
  // each such argument is an integer inside its domain.
  std::vector<SlotDomain> domains;
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
class LinearReductionPattern;
struct Checker {
  Arena &a;
  Arena patterns;
  CertifiedStatePatternCanonicalizer state_patterns;
  // A deque, as the Arena's nodes: appending never invalidates a reference to an existing element,
  // so a `const Spec &` held across a call that may register more (the masked matcher's exponent
  // closure re-enters matching) stays valid. Accessed only by index, size and push_back.
  std::deque<Spec> specs;
  std::map<std::tuple<Id, size_t, size_t, Z, bool, std::vector<SlotDomain>>, Id> spec_index;
  // Exact instantiated definition identities, local to this Checker. These
  // never enumerate reduction elements or infer equality from samples.
  std::unordered_map<Id, Id> call_definition_cache;
  std::map<std::tuple<Id, Z, bool>, Id> call_definitions;
  // Shape of an instantiated reduction definition (actual arguments as first-occurrence markers) ->
  // (the spec registered first for it, its argument recipe): Checker::call respells other specs.
  std::map<std::tuple<Id, Z, bool>, std::pair<Id, std::vector<std::pair<bool, Id>>>> call_spellings;
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
  // (body, binder, count) of a peeled Max Fold -> (fresh binder, body at fresh - 1) (canonicalize_fold).
  std::map<std::tuple<Id, Id, Z>, std::pair<Id, Id>> unpeeled_max_bodies;
  std::map<std::tuple<Id, Id, Z>, std::pair<Id, Id>> restricted_max_bodies;
  // Lemmas of LinearReductionPattern's chain merge (merge_by_finiteness_chain) proved by its exact case
  // analysis: (kind, three nodes) -> whether the proof accepted a case by identical non-finite values.
  // Each is a property of those nodes alone (see chain_lemma), so it holds in every later proof of this
  // comparison.
  std::map<std::tuple<int, Id, Id, Id>, bool> proved_chain_lemmas;
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
  // Target roots whose anchor inventory above was cut by a discovery limit and, built on demand for
  // such a root, the exact set (sorted node ids) of read, readbits and packedbits nodes in the
  // lexical scope of that inventory: anchors are opaque and a nested Fold contributes only its
  // seed. Nodes are immutable and hash-consed, so an entry stays valid for the whole comparison and
  // is shared by every family and spec that queries the same target node.
  std::unordered_set<Id> truncated_target_anchors;
  std::unordered_map<Id, std::vector<Id>> target_read_index;
  // Visit marks of one read-index build: a node is visited in the build whose number it holds.
  // Dense over Arena ids (the store only grows), so a build allocates and hashes nothing per node.
  std::vector<std::uint32_t> read_index_marks;
  std::uint32_t read_index_build = 0;
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
  // A set of candidate captures: per candidate, one Id per pattern slot (invalid when unbound).
  using Captures = std::vector<std::vector<Id>>;
  std::unordered_map<ReadAnchorKey,Captures,ReadAnchorHash> read_anchor_matches;
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
  std::map<std::pair<Id, Id>, bool> equal_arm_cache;
  std::deque<PieceNode> pieces;  // a deque for the same reason as specs
  std::map<std::string, uint64_t> failures;
  // Different complete coverage proofs describe the same exact contribution.
  // Export these equalities to the existing per-comparison proof reuse bank;
  // partial matches never enter this list.
  std::vector<std::pair<Id,Id>> complete_reduction_equalities;
  bool exponential_mode = false;
  // Closes a reduction Call inside a Sum (LinearReductionPattern::owned_closure), set by the
  // comparison; unset, the masked matcher's exponent closure is not attempted.
  std::function<std::optional<Id>(Id, Id, const std::vector<Id> &)> exponent_closure;
  // The linear-reduction pattern whose term match is running (LinearReductionPattern::OwnerScope),
  // else null. A closure query reached from a closure query's own match is charged to it.
  LinearReductionPattern *closure_owner = nullptr;
  // Exponent closure queries (LinearReductionPattern::owned_closure): (value, spec, parameters) ->
  // R + C * call(spec, parameters), or invalid when the query did not close within its budget (a
  // decline, also for a query that ran out of it). The result depends on nothing else: the target
  // is given explicitly (no active-spec state is read), the budget is one fixed standard budget per
  // query, and a nested query cut off by its enclosing budget is never stored. A success is an exact
  // identity of immutable hash-consed nodes, so it holds wherever the same query recurs.
  std::map<std::tuple<Id, Id, std::vector<Id>>, Id> exponent_closure_cache;
  ExponentialPolynomial scalar;
  Checker(Arena &ar, bool ep = false) : a(ar), state_patterns(patterns, a),
      exponential_mode(ep), scalar(ar) { patterns.deadline = a.deadline; }
  Id canonical_coefficient(Id value);
  Id equal_arm_select(Id id);
  bool equal_coefficients(Id left, Id right);
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
  // An enclosing reference reduction binder other than var, wherever it
  // occurs in the body (a mask such as s < r, too). pattern() slots it as a
  // Call parameter instead of keeping it as a literal: the Call's arguments,
  // and hence its free set, then carry the dependence, instantiate_body
  // substitutes it, and the outer Fold's invariance tests see it.
  bool enclosing_binder(Id id, Id var) const {
    return id != var && a.nodes[id].op == "var" && reference_binders.contains(id);
  }
  struct SpecPattern {
    Id pattern;
    size_t slots, axis;
    std::vector<Id> values;
    // Per slot. valued: the body's Facts depend on the slot's value, because
    // it occurs outside every coordinate position (an enclosing binder in a
    // sum or mask) or inside an argument a nested Call's Facts depend on.
    // coordinate: it occurs in a coordinate position, or is the axis.
    std::vector<bool> valued, coordinate;
  };
  void coordinate_slots(Id root, std::vector<size_t> &out) const;
  SpecPattern pattern(Id root, Id var, bool allow_parameterized = true, bool allow_derived = true);
  // Every variable that was a Fold binder in a normalization by this Checker,
  // on either side. Never cleared: a spec outlives the normalization that
  // registered it, and a re-normalized part of a reference may hold a binder
  // of the whole reference as a free variable.
  std::unordered_set<Id> fold_binders;
  // True when no spec reachable from sid through nested Calls keeps a Fold
  // binder as a literal. Such a literal is a dangling reduction binder: the
  // Call's value depends on it, but its arguments and free set do not.
  // pattern() slots enclosing binders; this is the independent check for proof
  // rules that evaluate a spec's members. A literal that never was a Fold
  // binder (an output coordinate, a template axis) is a parameter of the whole
  // proof, which no reduction iterates. Memoized per spec while fold_binders
  // is unchanged; a Call names an earlier spec, so the recursion terminates.
  // O(total pattern nodes).
  std::unordered_map<Id, bool> literal_free_specs;
  size_t literal_free_binders = 0;
  bool literal_binder_free(Id sid);
  Id intern_spec(Id p, size_t slots, size_t axis, Z count, Facts f,
                 bool maximum = false, std::vector<SlotDomain> domains = {});
  std::optional<std::pair<Q, Q>> integer_domain(Id value);
  std::optional<std::vector<SlotDomain>> slot_domains(const std::vector<Id> &values,
                                                      const std::vector<bool> &valued, size_t axis);
  // The same narrowing as integer_domain, so a reference value is inside the
  // domain it defined.
  bool in_domain(Id argument, const SlotDomain &domain) {
    if (auto v = a.value(argument)) return v->denominator() == 1 && domain.lo <= *v && *v <= domain.hi;
    const auto range = integer_domain(argument);
    return range && domain.lo <= range->first && range->second <= domain.hi;
  }
  Id call(Id sid, const std::vector<Id> &args);
  void register_paired_maximum(Id call_id, Id reduced_axis);
  Id reg(Id body, Id var, Z count, bool maximum=false);
  Id neutral(Id sid) { return specs[sid].maximum ? a.infinity(-1) : a.zero; }
  Id plus(Id sid, std::vector<Id> xs) {
    return specs[sid].maximum ? a.maximum(xs) : a.add(xs);
  }
  Summary select(const Summary &s, bool max);
  std::optional<Id> instantiated_coordinate(Id root, const std::vector<Id> &captures);
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
  // Bounds of the coordinate-rest rule in matches(): pattern summands, target terms, assignments.
  static constexpr size_t rest_summands = 4, rest_terms = 8, rest_branches = 8;
  // Defer only coordinate syntax the deferred evaluator can instantiate.
  // Gathered reads and derived quotient/remainder recipes must instead take
  // the ordinary structural matcher, which can infer the unknown axis.
  std::unordered_map<Id, bool> deferred_coordinate_syntax_cache;
  bool deferred_coordinate_syntax(Id root);
  Captures matches(Id pat, Id target, std::vector<Id> captures, bool fixed_parameters = false);

  // A read anchor with more coordinates is matched only by the general matcher (no cached or
  // scoped matching). scoped_read_anchor visits at most scoped_anchor_visit_limit + 1 pattern nodes
  // of an anchor's coordinates; an anchor whose walk needs more is not scoped.
  static constexpr size_t read_anchor_coordinate_limit = 8, scoped_anchor_visit_limit = 64;
  // Bounded body-candidate search: at most this many distinct captures per anchor and this many
  // anchor/target attempts per search; beyond either the search declines (proposes no candidate).
  static constexpr size_t body_candidate_limit = 32, body_candidate_attempts = 256;
  // Anchor discovery per body (body_candidates): at most this many anchors, edges followed and
  // distinct nodes queued; a cut discovery is counted in ep_body_anchor_limited.
  static constexpr size_t anchor_limit = 16, anchor_edge_limit = 2048, anchor_node_limit = 512;
  bool scoped_read_anchor(Id pattern,const std::vector<Id> &cap);
  std::optional<Captures> scoped_read_captures(Id pattern,const std::vector<Id> &cap,const std::vector<Id> &ts,
                                               size_t &attempts);
  bool excluded_anchor(Id root, Id anchor, const Captures &caps);
  // A read, readbits or packedbits pattern whose coordinate subtrees hold only slots, constants and
  // ordinary operations (no derived_coordinate, coordinate_sum, affine_slot or affine_param recipe,
  // no Call). At most 64 visited pattern nodes. O(visited nodes).
  // With a capture: also a derived coordinate whose axis the capture binds to an integer inside the
  // coordinate's recorded scope (CoordinateRelations::within_axis), which is then the exact
  // quotient/remainder of that value (a gathered row read through a concatenated index list).
  // A read coordinate `actual` against its pattern coordinate (an integer sum, product, quotient or
  // remainder): every slot the pattern needs is captured, its instantiation is an integer, and the
  // instantiation and `actual` have the same local index form (the same integer on every input).
  // Candidates only; the complete body proof follows. O(pattern coordinate + memoized forms).
  // Whether an integer expression holds a node a local identity acts on (a quotient, a remainder, a
  // Select, a comparison, a word, or a pattern's derived coordinate, a quotient or remainder of its
  // axis); reads are leaves. Where neither side holds one, local forms are the
  // stored forms and local_coordinate_match adds nothing to the structural match. Memoized per node;
  // at most 64 nodes visited (beyond: true).
  std::unordered_map<Id, bool> local_shape_cache, pattern_local_shape_cache;
  bool local_shape(const Arena &arena, Id root, std::unordered_map<Id, bool> &cache);
  bool local_coordinate_match(Id pattern, Id actual, const std::vector<Id> &cap);
  // ---- Local index forms (exact, pair-local) -------------------------------------------------
  // A canonical copy of an integer index expression, built only to compare two specific nodes (an
  // anchor's instantiated read with a target read; the reads of one instantiated body with its target's
  // reads; two masks of one masked term). It is never substituted into a stored normal form, a pattern,
  // or a cache another query reads: callers only test equality of two local forms. Children first, then
  // exact integer identities under the Arena bounds of the leaves:
  //   floor((a y + R) / (a m)) = floor(y / m) for the one integer term y with coefficient +-a (a > 1,
  //     a | d) whose other terms R (integers, a constant included) have bounds inside [0, a - 1]:
  //     a y + R lies in [a y, a y + a), so the quotient by a m is that of a y;
  //   c m floor(y / m) + c (y mod m) = c y, with floor(y / m) taken in its local form (so a quotient the
  //     Arena stores flattened, floor(floor(z / b) / m) = floor(z / (b m)), still pairs with its remainder);
  //   select([y == k], k, y) = y and select([y != k], y, k) = y (where y = k both arms are k);
  //   [y <= lo] = [y == lo] and [y >= hi] = [y == hi] at the bounds [lo, hi] of a finite y;
  //   a word of a constant is the constant modulo 2^w;
  //   a maximum is the flat maximum of its operands (nested maxima and positive multiples of maxima).
  // Other nodes are rebuilt by their constructors (reapply). Reads, Folds, Calls and certified states are
  // leaves (a read's own coordinates are compared by local_read_twin, which never builds a read). Memoized
  // per node (local_index_cache); at most local_index_visit_limit new nodes per query, beyond which the
  // query returns its root unchanged.
  struct LocalIndexCut {};
  std::unordered_map<Id, Id> local_index_cache;
  static constexpr size_t local_index_visit_limit = 4096;
  Id local_quotient(Id n, const Z &d, size_t &visits);
  Id local_sum(Id sum, size_t &visits);
  // x with every maximum replaced by its flat local form (local_index_form, coefficient-sized walk): the same
  // value, one node for two spellings of one running maximum (nested differently, or over extended sums
  // whose operands are in another order). Maxima are not entered; other nodes are rebuilt by their
  // constructors (reapply). Folds, Calls and reads are leaves. At most flat_maxima_limit distinct nodes,
  // else x unchanged. Exact.
  static constexpr size_t flat_maxima_limit = size_t(1) << 17;
  Id flat_maxima(Id x);
  Id sorted_extended_sum(Id x);
  int infinity_arm(Id x) const;
  std::optional<Id> masked_infinity_select(Id c, Id then_arm, Id else_arm);
  size_t local_index_limit_now = local_index_visit_limit;  // the limit of the query in progress
  Id local_index_form(Id root, size_t limit = local_index_visit_limit);
  std::vector<Id> scope_reads(Id root);
  // The target read at the storage element of a read of kind (op, tag, data) at coordinates `coords`:
  // the unique read of `target`'s scope (scope_reads, or the target's complete anchor inventory) of that
  // kind whose coordinates have, one by one, the same local index forms, or invalid (none, or several).
  // A read is the stored value of its allocation and type at its coordinates; equal coordinates name
  // the same element, so the twin equals the read. The target's reads are grouped by kind and their
  // coordinate forms computed once per target and kind; O(arity) per lookup after that.
  using ReadKind = std::tuple<std::string, std::string, std::vector<Q>>;
  struct TargetReads {
    std::unordered_set<Id> reads;
    std::map<ReadKind, std::map<std::vector<Id>, Id>> by_coordinates;
  };
  std::unordered_map<Id, TargetReads> target_local_reads;
  TargetReads &target_reads(Id target) {
    auto it = target_local_reads.find(target);
    if (it != target_local_reads.end()) return it->second;
    TargetReads tr;
    if (auto inv = target_anchor_cache.find(target); inv != target_anchor_cache.end() && !truncated_target_anchors.contains(target)) {
      for (Id id : inv->second) {
        const auto &op = a.nodes[id].op;
        if (op == "read" || op == "readbits" || op == "packedbits") tr.reads.insert(id);
      }
    } else {
      for (Id id : scope_reads(target)) tr.reads.insert(id);
    }
    return target_local_reads.emplace(target, std::move(tr)).first->second;
  }
  Id local_twin_at(Id target, const ReadKind &kind, const std::vector<Id> &coords);
  Id local_read_twin(Id target, Id read);
  // Rule (read alignment, pair-local): the reads of an instantiated body `expected` that its target
  // does not hold are replaced, in this copy of the body only, by the unique target read at the same
  // storage element (local_read_twin: same allocation, type and arity, equal local index forms). A read
  // denotes the stored value at its coordinates, so the copy equals the body. Returns expected itself
  // when nothing is replaced. Only reads in the body's scope (scope_reads) are aligned. O(nodes of both
  // scopes, memoized forms); memoized per (body, target) pair (align_cache).
  std::map<std::pair<Id, Id>, Id> align_cache;
  Id align_reads(Id expected, Id target);
  Id align_reads_uncached(Id expected, Id target);
  bool plain_read_anchor(Id pattern, const std::vector<Id> *cap = nullptr) const;
  bool cacheable_read_anchor(Id pattern) const;
  Captures match_anchor(Id pattern,Id target,const std::vector<Id> &captures);

  Id instantiate_body(Id root, const std::vector<Id> &cap, Id twin_target = invalid, bool build_max = false);
  bool target_scope_contains_read(Id target, Id read);
  // ---- Exact refutation before a body proof: a reduction Call the body needs --------------------
  // The Call nodes in the scope of `target` (the anchor scope above: reads, slots, variables, Calls
  // and certified states are leaves and a nested Fold contributes only its seed). Every atom of an
  // EP normal form of the target that is a Call is one of these: the EP engine descends only
  // through sums, products, exponentials, logarithms and reciprocals, and creates no Call. Built
  // once per target, O(scoped nodes + edges).
  std::unordered_map<Id, std::vector<Id>> target_call_index;
  const std::vector<Id> &target_scope_calls(Id target) {
    if(auto found=target_call_index.find(target);found!=target_call_index.end())return found->second;
    if(read_index_marks.size()<a.nodes.size())read_index_marks.resize(a.nodes.size(),0);
    if(++read_index_build==0) {  // the build numbers wrapped: no stale mark may match
      std::fill(read_index_marks.begin(),read_index_marks.end(),0);read_index_build=1;
    }
    const auto build=read_index_build;
    std::vector<Id> calls,todo{target};read_index_marks[target]=build;
    size_t visited=0;
    while(!todo.empty()) {
      if((++visited&1023)==0)a.check();
      const Id id=todo.back();todo.pop_back();
      const auto &n=a.nodes[id];
      if(n.op=="call"){calls.push_back(id);continue;}
      if(n.op=="read" || n.op=="readbits" || n.op=="packedbits" || n.op=="slot" || n.op=="var" ||
         n.op=="certified_state")continue;
      const size_t children=n.op=="fold"?std::min<size_t>(1,n.args.size()):n.args.size();
      for(size_t j=0;j<children;++j) {
        const Id child=n.args[j];
        if(read_index_marks[child]!=build){read_index_marks[child]=build;todo.push_back(child);}
      }
    }
    a.stats["ep_body_target_call_index_nodes"]+=visited;
    ++a.stats["ep_body_target_call_index_builds"];
    return target_call_index.emplace(target,std::move(calls)).first->second;
  }
  bool nonzero_normal_form(Id id,size_t &visits);
  // Rule: let X = A + B * C, where C is a Call that X reaches by exactly one path of products and
  // sums, no other operand on that path holds a node alpha-equal to C anywhere below it, and every
  // other factor of the products on the path has a nonzero normal form. Then the atom C has degree
  // one in the EP normal form of X and in the numerator of its fraction, and degree zero in the
  // denominator. A target with no Call alpha-equal to C in its scope has no atom C in any normal
  // form, so neither the polynomial nor the cross-multiplied EP equality X = target can hold
  // (the atom's degree differs on the two sides). Returns whether this refutes X = target; false
  // leaves the complete EP proof to decide. At most required_call_visits nodes of X per walk and
  // required_call_candidates Calls; one scope scan per target (target_scope_calls).
  static constexpr size_t required_call_visits = 256, required_call_candidates = 4;
  bool missing_required_call(Id x,Id target);
  // Rule: two Calls denote the same value when their instantiated definitions are one registered
  // definition (call(): call_definitions). Which Call is the registered one depends on which was
  // instantiated first, so a pattern Call of spec P with k arguments may have to name a target Call
  // T spelled with another spec, with more arguments (n > k: the same literal as two parameters of
  // T) or with fewer (n < k: two parameters of P that T has as one). The candidate argument lists
  // for P are T's arguments without n - k of them, kept in order, or with k - n of them repeated at
  // some position; a list names T when it instantiates P to T itself: call(P, list) == T. The
  // pattern Call is then matched against that respelled node, which captures its slots as any Call
  // of P does (coordinate relations included). A proposal only: every capture still needs the
  // complete body proof, whose instantiation of the pattern Call is T again; the choice of the
  // registered spelling therefore changes which candidates are found, never what is accepted.
  // |n - k| <= 2, n and k <= 8; at most respelling_limit instantiations per candidate search
  // (`used`), each one call() plus one bounded match. A list with a repeated argument is tried only
  // if it agrees with every argument of the pattern Call that is an already captured slot (the
  // match would refuse it otherwise).
  static constexpr size_t respelling_limit = 32;
  Captures respelled_call_captures(Id pat,Id target,const std::vector<Id> &cap,size_t &used);
  Captures body_candidates(const Spec &s,Id target,const std::vector<Id> &initial);

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
  bool admit_payload_coordinates(Id expected,Id actual);

  // ---- Mask resolution of a masked product (exact) -------------------------------------------
  // A Boolean literal: the 0/1 node `node` has the value `truth`.
  struct MaskLiteral { Id node = invalid; bool truth = true; };
  std::optional<MaskLiteral> mask_literal(Id condition, bool nonzero);
  int decide_under(Id c, const MaskLiteral &lit);
  // A node rebuilt where `lit` holds: a Select or comparison whose condition the literal decides
  // (decide_under) is its taken arm or its constant truth, the literal's own node is its value, and
  // every other node is rebuilt from its rebuilt operands by its constructor (Checker::reapply,
  // strict: no inherited facts). Not entered: Max nodes outside read coordinates (a value Max is a
  // statistic shared by many guarded terms; rebuilding a guard inside it would give each term its
  // own copy), Calls, and Folds whose binder the literal depends on. A read whose coordinates change
  // is rebuilt with the facts of its element type only (a raw storage word: integer in
  // [0, 2^width - 1]; a typed real element: finite; any other typed read declines, since its bounds
  // may come from a premise). Equal to the original wherever lit holds. Memoized per (node, literal,
  // coordinate position). `work` counts visited nodes; beyond the limit the rebuild fails.
  // deep (only the masked matcher's exponent rebuild, rule exponent under its mask): a Select's condition
  // and a word of a rebuilt operand are rebuilt too; the ordinary mask resolution (mask_resolved) keeps
  // them as they are. The memo is keyed by the mode.
  std::map<std::tuple<Id, Id, bool, bool, bool>, Id> mask_resolution_cache;
  static constexpr size_t mask_resolution_limit = 20000;
  Id resolve_under(Id x, const MaskLiteral &lit, bool coordinate, size_t &work, bool &failed, bool deep = false);
  Id mask_resolved(Id n);

  // One bounded local congruence: a product of finite factors may contain a
  // single zero-mask condition and exponentials of one base. No nested mask
  // paths are enumerated. Equal masks and equal residual factors leave only
  // an exponential coefficient to be proved independent of the Sum binder.
  struct MaskedExponentialTerm {
    Id predicate, exponent, residual;
    bool base2;
  };
  // A finite value proved to be 0 or 1: integer known bits and bounds within [0, 1]. O(known_bits).
  bool boolean_value(Id x) {
    const auto &n = a.nodes[x];
    return n.facts.finite && n.bounds && n.bounds->first >= 0 && n.bounds->second <= 1 &&
           a.known_bits(x).integer;
  }
  std::optional<MaskedExponentialTerm> masked_exponential_term(Id root);
  Id substitute_nodes(Id x, const std::unordered_map<Id, Id> &replace);
  // Rule (cases of a stored byte): two finite values whose only stored readbits leaf of at most 8 bits
  // is one node B (an e2m1 nibble decoded from the same byte by different formulas, for example) are
  // equal when they are equal (EP) in every case of B's 2^w values and of the comparisons, at most
  // two, that are Select conditions in either value and do not depend on B (a nibble chosen by an
  // index's parity, for example; each is 0 or 1, so fixing it to either value covers every input).
  // B is an integer in [0, 2^w - 1]. At most 1024 substituted equalities; values of more than 4096
  // nodes, a second such leaf, or more such conditions decline. Exact.
  std::map<std::pair<Id, Id>, bool> byte_case_cache;  // (x, y) -> result: a function of immutable nodes
  bool byte_cases_equal(Id x, Id y);
  bool byte_cases_equal_uncached(Id x, Id y);
  std::optional<Id> masked_body_coefficient(Id expected, Id actual, bool allow_proved_mask = false) {
    return masked_body_coefficient_direct(expected, actual, allow_proved_mask);
  }
  // A Select factor of x (a product's factor, or x itself) whose condition compares a free output
  // coordinate v (an integer variable "output:...") with a constant, as (v, first, last): the
  // interval of v's domain where the condition holds, which must leave a nonempty complement that
  // is one interval too. O(factors).
  struct CoordinateGuard { Id axis; Z first, last; bool points = false; };
  uint64_t mask_case_serial = 0;
  std::optional<CoordinateGuard> output_coordinate_guard(Id x);
  static constexpr unsigned max_coordinate_points = 16;
  bool coordinate_arithmetic(Id x) const;
  std::optional<Id> masked_body_coefficient_direct(Id expected, Id actual, bool allow_proved_mask);

  const std::vector<Match> &match_term(Id sid, Id n, const std::vector<Id> &parameters = {}) {
    auto key = std::tuple(sid, n, parameters);
    auto it = match_cache.find(key);
    if (it != match_cache.end())
      return it->second;
    ExponentialPolynomial::Timer match_timer(scalar.profiling?&scalar.match_ns:nullptr);
    const auto &s = specs[sid];
    const auto &p = patterns.nodes[s.pattern];
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
    std::set<Match> unique = match_term_target(sid, n, initial, fixed, masked_reference);
    // Rule (mask resolution, mask_resolved): where nothing matched a masked reference body, the same
    // matcher once more on the term rebuilt under its own mask (an equal value). Every candidate
    // still needs the complete body proof against that equal value.
    Id target = n;
    if (unique.empty() && exponential_mode && masked_reference) {
      const Id resolved = mask_resolved(n);
      if (resolved != n) {
        target = resolved;
        unique = match_term_target(sid, resolved, initial, fixed, masked_reference);
        if (!unique.empty()) ++a.stats["mask_resolved_matches"];
      }
    }
    // Rule (match by cases on an output coordinate): with the reduction's parameters fixed, a target
    // with a factor Select on a comparison of a free output coordinate v with a constant
    // (output_coordinate_guard) is matched on each part of v's domain: the interval where the
    // comparison holds and the rest, v replaced by a fresh coordinate over the part in the target
    // and in every fixed parameter (restrict_axis, a capture-safe substitution; instantiation
    // commutes with it); for any other comparison whose only free variable is v (at most
    // max_coordinate_points values), each point of v's domain. It matches the body at index i with
    // coefficient C when each part gives exactly that one match, with the same i and C (so neither
    // depends on v). The parts cover v's domain exactly once, so target = C * body(parameters, i)
    // on all of it. Exact. Two parts, or at most max_coordinate_points points.
    if (unique.empty() && exponential_mode && masked_reference && fixed)
      if (const auto guard = output_coordinate_guard(target)) {
        const auto &bounds = *a.nodes[guard->axis].bounds;
        const Z lo = bounds.first.numerator(), hi = bounds.second.numerator();
        std::vector<std::pair<Z, Z>> parts;
        if (guard->points) for (Z point = lo; point <= hi; ++point) parts.emplace_back(point, point);
        else {
          parts.emplace_back(guard->first, guard->last);
          if (guard->first > lo) parts.emplace_back(lo, guard->first - 1);
          if (guard->last < hi) parts.emplace_back(guard->last + 1, hi);
        }
        std::optional<Match> common;
        bool all = guard->points ? parts.size() >= 2 : parts.size() == 2;
        for (const auto &[first, last] : parts) {
          if (!all) break;
          Id value;
          if (first == last) value = a.constant(Q(first));
          else {
            size_t before;
            do {
              before = a.nodes.size();
              value = a.var("match-case:" + std::to_string(mask_case_serial++), Q(first), Q(last));
            } while (value < before);
          }
          std::vector<Id> restricted = initial;
          for (Id &x : restricted) if (x != invalid) x = restrict_axis(x, guard->axis, value);
          // Rule (restricted body of a case part): the bodies this part's match instantiates are
          // restricted as its target is (only this call's own instantiations, not nested queries).
          const std::pair<Id, Id> restriction{guard->axis, value};
          const auto part = match_term_target(sid, restrict_axis(target, guard->axis, value), restricted, true, masked_reference, &restriction);
          if (part.size() != 1) { all = false; break; }
          const Match &m = *part.begin();
          if (a.depends(m.index, value) || a.depends(m.coef, value) ||
              (common && (common->index != m.index || common->coef != m.coef))) { all = false; break; }
          common = Match{parameters, m.index, m.coef};
        }
        if (all && common) {
          unique = {*common};
          ++a.stats["match_coordinate_cases"];
        }
      }
    return match_cache
        .emplace(key, std::vector<Match>(unique.begin(), unique.end()))
        .first->second;
  }
  std::set<Match> match_term_target(Id sid, Id n, const std::vector<Id> &initial, bool fixed, bool masked_reference,
                                    const std::pair<Id, Id> *case_restriction = nullptr);
  bool stored_input_scale(Id pattern_node);
  Summary combine(const std::vector<Summary> &parts);
  Summary scale(const Summary &s, Id c);
  Summary residual(const Summary &s, Id n);
  Summary bind(const Summary &s, Id var, const Z &count);
  std::optional<Box> box(Id index,
                         const std::vector<std::pair<Id, Z>> &domains);
  std::optional<std::vector<Box>> boxes(Id p, const Z &count, bool &permuted);
  std::optional<Id> close(const Summary &s);
  std::optional<Id> base2_logarithm_product(const std::vector<Id> &args, const Q &scalar);
  Id reapply(Id original, const std::vector<Id> &args, bool strict = false);
  std::pair<Id, Id> split_invariant_factors(Id body, Id axis);

  Id substitute_strict(Id root, Id var, Id value);
  Id restrict_axis(Id root, Id old_axis, Id new_axis);

  std::optional<std::pair<Id, Id>> masked_max_domain(Id value, Id axis, Z &count);
  Id canonicalize_fold(Id original, const std::vector<Id> &args, bool strict = false);

  Id canonicalize(Id root);

  Id normalize(Id root, bool reference = false);
};
} // namespace rg

#include "checker_patterns.hpp"
#include "checker_matching.hpp"
#include "checker_read_alignment.hpp"
#include "checker_body_matching.hpp"
#include "checker_masks.hpp"
#include "checker_reductions.hpp"
#include "checker_normalization.hpp"
