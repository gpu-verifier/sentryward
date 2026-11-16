#pragma once
#include <algorithm>
#include <array>
#include <bit>
#include "rational.hpp"
#include "sentryward/equivalence/ieee_bits.h"
#include <chrono>
#include <climits>
#include <cmath>
#include <deque>
#include <fstream>
#include <functional>
#include <map>
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>
namespace rg {
using J = nlohmann::json;
using Q = Rational; // Z and Rational come from rational.hpp
using Id = uint32_t;
constexpr Id invalid = UINT32_MAX;
// String/literal equality for all of namespace rg: libstdc++'s own
// operator==(const string&, const char*) formula, forced inline so that a
// literal's length folds; GCC otherwise calls one out-of-line copy (plus
// strlen) from large functions. As a non-template it is preferred over std's
// template for every rg string == literal, its reversed form and rewritten !=.
[[nodiscard, gnu::always_inline]] inline constexpr bool operator==(const std::string &lhs,
                                                                   const char *rhs) {
  return lhs.size() == std::char_traits<char>::length(rhs) &&
         !std::char_traits<char>::compare(lhs.data(), rhs, lhs.size());
}
// A null operand stays ill-formed (std's template cannot deduce from it).
bool operator==(const std::string &, std::nullptr_t) = delete;
struct Unsupported : std::runtime_error {
  std::string code = "expression_form_unsupported";
  J details = J::object();
  using runtime_error::runtime_error;
  Unsupported(std::string message, std::string diagnostic_code, J evidence)
      : runtime_error(std::move(message)), code(std::move(diagnostic_code)), details(std::move(evidence)) {}
};
struct Timeout : std::runtime_error {
  Timeout() : runtime_error("wall clock deadline") {};
};
// One-limb operands take one hardware divide or gcd; each result equals the
// cpp_int formula below it.
inline Z floorz(const Z &n, const Z &d) {
  rational_detail::Limb a, b;
  bool na, nb;
  if (rational_detail::limb(n, a, na) && rational_detail::limb(d, b, nb) && b != 0) {
    // Truncated quotient, minus one when the truncated remainder (sign of n) is negative.
    const rational_detail::SignedWide q(a / b), v = (na != nb ? -q : q) - (na && a % b != 0);
    return rational_detail::make(rational_detail::Wide(v < 0 ? -v : v), v < 0);
  }
  Z q = n / d, r = n % d;
  if (r < 0)
    --q;
  return q;
}
inline Z modz(const Z &n, const Z &d) {
  rational_detail::Limb a, b;
  bool na, nb;
  if (rational_detail::limb(n, a, na) && rational_detail::limb(d, b, nb) && !nb && b != 0) {
    // d > 0: n - floorz(n, d)*d is the truncated remainder, plus d when it is negative.
    const rational_detail::Limb r = a % b;
    return Z((na && r != 0) ? rational_detail::Limb(b - r) : r);
  }
  return n - floorz(n, d) * d;
}
inline Z floorq(const Q &q) { return floorz(q.numerator(), q.denominator()); }
inline Z pow2(unsigned w) {
  return w < rational_detail::limb_bits ? Z(rational_detail::Limb(1) << w) : Z(Z(1) << w);
}
// gcd(|x|, |y|) with a one-limb fast path.
inline Z gcdz(const Z &x, const Z &y) {
  rational_detail::Limb a, b;
  if (rational_detail::limb(x, a) && rational_detail::limb(y, b))
    return Z(rational_detail::Limb(std::gcd(a, b)));
  return boost::multiprecision::gcd(boost::multiprecision::abs(x), boost::multiprecision::abs(y));
}
inline std::string str(const Q &q) {
  return q.numerator().str() +
         (q.denominator() == 1 ? "" : "/" + q.denominator().str());
}
inline void mix(size_t &h, size_t v) {
  h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
}
inline size_t hashq(const Q &q) {
  size_t h = boost::multiprecision::hash_value(q.numerator());
  mix(h, boost::multiprecision::hash_value(q.denominator()));
  return h;
}
inline Q decimal(std::string s) {
  auto pos = s.find('/');
  if (pos != s.npos)
    return Q(Z(s.substr(0, pos)), Z(s.substr(pos + 1)));
  int e = 0;
  pos = s.find_first_of("eE");
  if (pos != s.npos) {
    e = std::stoi(s.substr(pos + 1));
    s.resize(pos);
  }
  pos = s.find('.');
  if (pos != s.npos) {
    e -= int(s.size() - pos - 1);
    s.erase(pos, 1);
  }
  bool neg = false;
  if (!s.empty() && (s[0] == '-' || s[0] == '+')) {
    neg = s[0] == '-';
    s.erase(0, 1);
  }
  Z n = 0;
  for (char c : s) {
    if (c < '0' || c > '9')
      throw Unsupported("invalid decimal");
    n = n * 10 + (c - '0');
  }
  Z d = 1;
  while (e < 0) {
    d *= 10;
    ++e;
  }
  while (e > 0) {
    n *= 10;
    --e;
  }
  return Q(neg ? -n : n, d);
}
inline Q jq(const J &j) {
  return decimal(j.is_string() ? j.get<std::string>() : j.dump());
}
// The exact rational value of IEEE-style binary bits (sentryward::equivalence::ieeeDyadicValue): an
// all-ones exponent field is read as a finite exponent; callers exclude infinities and NaNs.
inline Q ieee_value(uint64_t bits, unsigned width, unsigned fraction) {
  const auto value = sentryward::equivalence::ieeeDyadicValue(bits, width, fraction);
  const Z significand(value.significand);
  const Q magnitude = value.exponent >= 0 ? Q(significand << value.exponent)
                                          : Q(significand, pow2(unsigned(-value.exponent)));
  return value.negative ? -magnitude : magnitude;
}
using Bounds = std::optional<std::pair<Q, Q>>;
struct Facts {
  bool finite = false, pos = false, nn = false;
  // A value is either a finite real or negative infinity. This is not a
  // branch/path set: unknown, NaN, positive infinity and undefined are excluded.
  bool finite_or_neginf = false;
  bool finite_or_negative_infinity() const { return finite || finite_or_neginf; }
};
struct Node {
  std::string op, tag;
  std::vector<Id> args;
  std::vector<Q> data;
  Facts facts;
  Bounds bounds;
  std::vector<Id> free;
};
struct Arena {
  // Immutable node-local facts, modulo 2^256. Unknown bits are not zero bits.
  struct KnownBits { Z zero=0, one=0; bool integer=false; };
  std::unordered_map<Id,KnownBits> known_bits_cache;
  // Resolved one-bit queries and OR slices. Read-only comparison probes only
  // consume this cache; ordinary multi-byte payload slicing adds no entries.
  std::map<std::tuple<Id,unsigned,unsigned>,Id> resolved_bit_slice_cache;
  std::map<std::pair<Id,Id>,Id> coordinate_witness_cache;
  std::map<std::tuple<Id,Id,bool>,bool> reduction_witness_cache;
  // Installed by an importer for the duration of its import (Importer::import): rewrites an eq/ne
  // predicate on a raw 32-bit input word to the same predicate on the typed element, or declines.
  std::function<std::optional<Id>(const std::string &, Id, Id)> word_predicate;
  bool raw_word32(Id x) const {
    const auto &n = nodes[x];
    return n.op == "readbits" && n.args.size() == 1 && n.data.size() == 1 && n.data[0] == 32;
  }
  // word_shl constructions so far (0: no symbolic left shift exists, so no shifted-bit rule can apply)
  // and, per node, whether a word_shl leaf is reachable through the shells shifted_word_bit reads.
  uint64_t word_shl_requests = 0;
  std::unordered_map<Id,bool> shift_dependent_cache;
  // Words assembled from predicate Selects (packed_predicate_word) and the mode of a bit query on one.
  std::unordered_map<Id,bool> predicate_packed_cache;
  bool packed_bit_mode = false;
  // The exact constant ln 2 = log(2) once created (natural_log2), else invalid. While it is invalid no
  // node carries ln 2 or 1/ln 2, so the base-change and cancellation rules have nothing to do.
  Id exact_ln2 = invalid;
  std::deque<Node> nodes;
  // Flat intern index: one (key hash, id) slot per node, linear probing over a
  // power-of-two table at load <= 3/4. Nodes are never erased and an equal key
  // is never inserted twice, so a probe finds at most one match. A probe tests
  // the stored hash before touching any Node; growth walks slots only.
  struct IndexSlot {
    size_t hash;
    Id id;
  };
  std::vector<IndexSlot> intern_index;
  unsigned intern_index_shift = 64;
  std::map<std::string, uint64_t> stats;
  // Hot counters keep a pointer to their own stats entry, created by the first
  // increment exactly where stats[key] would create it. stats is never erased
  // and an Arena is neither copied nor moved, so the pointers stay valid and
  // the counters report remains a plain dump of stats.
  enum class Hot : unsigned char {
    intern_requests, add_term_visits, pattern_node_visits, imported_context_nodes,
    canonical_node_visits, normalization_node_visits, local_pattern_candidates,
    ep_body_pattern_anchor_cache_hits, ep_body_target_anchor_cache_hits,
    ep_body_read_anchor_rejections, count
  };
  static constexpr std::array<const char *, size_t(Hot::count)> hot_keys{
      "intern_requests", "add_term_visits", "pattern_node_visits",
      "imported_context_nodes", "canonical_node_visits", "normalization_node_visits",
      "local_pattern_candidates", "ep_body_pattern_anchor_cache_hits",
      "ep_body_target_anchor_cache_hits", "ep_body_read_anchor_rejections"};
  std::array<uint64_t *, size_t(Hot::count)> hot_slots{};
  std::chrono::steady_clock::time_point deadline;
  Id zero, one;
  uint64_t ticks = 0;
  Arena(double seconds = 300)
      : deadline(std::chrono::steady_clock::now() +
                 std::chrono::milliseconds(int64_t(seconds * 1000))) {
    zero = constant(0);
    one = constant(1);
    stats["select_max_probe_temporary_nodes"] = 0;
    stats["select_max_probe_ancestor_nodes"] = 0;
  }
  Arena(const Arena &) = delete;
  Arena &operator=(const Arena &) = delete;
  uint64_t &counter(Hot key) {
    auto &slot = hot_slots[size_t(key)];
    if (!slot)
      slot = &stats[hot_keys[size_t(key)]];
    return *slot;
  }
  void check() {
    if ((++ticks & 4095) == 0 && std::chrono::steady_clock::now() > deadline)
      throw Timeout();
  }
  static size_t hash(std::string_view op, std::span<const Id> as,
                     std::span<const Q> ds, std::string_view tag);
  static size_t index_home(size_t h, unsigned shift) {
    return size_t((uint64_t(h) * 0x9E3779B97F4A7C15ULL) >> shift);
  }
  std::optional<Id> find_node(size_t h, std::string_view op, std::span<const Id> as,
                              std::span<const Q> ds, std::string_view tag) const;
  std::optional<Id> find_node(std::string_view op, std::span<const Id> as,
                              std::span<const Q> ds = {}, std::string_view tag = "") const {
    return find_node(hash(op, as, ds, tag), op, as, ds, tag);
  }
  // Every node request takes this one path: check(), intern_requests, probe.
  // A hit returns the existing id and never reads facts or bounds, so make(h)
  // builds them (and the owned key vectors) on a miss only, then insert_node.
  template <class Make>
  Id intern_lazy(std::string_view op, std::span<const Id> as, std::span<const Q> ds,
                 std::string_view tag, Make &&make) {
    check();
    ++counter(Hot::intern_requests);
    const size_t h = hash(op, as, ds, tag);
    if (auto hit = find_node(h, op, as, ds, tag))
      return *hit;
    return make(h);
  }
  Id intern(std::string_view op, std::vector<Id> as = {}, std::vector<Q> ds = {},
            std::string_view tag = "", Facts f = {}, Bounds b = {});
  Id insert_node(size_t h, std::string_view op, std::vector<Id> as, std::vector<Q> ds,
                 std::string_view tag, Facts f, Bounds b);
  Bounds boolean_fold_bounds(std::string_view tag, const std::vector<Id> &as,
                             const std::vector<Q> &ds);
  void reserve_index();
  static void place_index(std::vector<IndexSlot> &slots, unsigned shift, IndexSlot s) noexcept;
  bool depends(Id n, Id v) const {
    return std::binary_search(nodes[n].free.begin(), nodes[n].free.end(), v);
  }
  Id coordinate_at_zero(Id root,Id axis);
  bool reduction_witness_at_zero(Id root,Id axis,const Z &count,bool positive);
  bool positive_at_zero(Id body,Id axis,const Z &count) {
    return reduction_witness_at_zero(body,axis,count,true);
  }
  Facts fold_facts(Id initial,Id body,const Q &count,bool maximum,Id axis=invalid);
  Id constant(const Q &q) {
    return intern_lazy("const", {}, std::span<const Q>(&q, 1), "", [&](size_t h) {
      return insert_node(h, "const", {}, {q}, "", {true, q > 0, q >= 0}, std::pair(q, q));
    });
  }
  std::optional<Q> value(Id n) const {
    if (nodes[n].op == "const")
      return nodes[n].data[0];
    return {};
  }
  Id var(std::string key, Q lo, Q hi);
  Id infinity(int sign) { return intern("infinity", {}, {Q(sign)}, "", {false,false,false,sign<0}); }
  bool neginf(Id n) const {
    return nodes[n].op == "infinity" && nodes[n].data[0] == -1;
  }
  std::vector<std::pair<Id, Q>> terms(Id n);
  Id add(std::vector<Id> xs);
  Id neg(Id n) { return mul({constant(-1), n}); }
  Id sub(Id a, Id b) { return add({a, neg(b)}); }
  Id mul(std::vector<Id> xs);
  Id floor_value(Id n, Bounds established = {});
  Id exponential(Id n, bool base2);
  Id exp2(Id n) { return exponential(n, true); }
  Id exp(Id n) { return exponential(n, false); }
  static bool valid_comparison(const std::string &op) {
    return op=="eq" || op=="ne" || op=="lt" || op=="le" || op=="gt" || op=="ge";
  }
  static std::string reverse_comparison(const std::string &op);
  static std::string complement_comparison(const std::string &op);
  struct SignComparison {
    Id source; unsigned bit; bool negative;
  };
  std::optional<SignComparison> sign_comparison(const std::string &op,Id a,Id b) const;
  // A virtual unit-coefficient product is a bounded vector of existing factor
  // IDs, NOT a newly interned Mul. This replaces terms() for comparison queries:
  // terms() is a constructor and can intern a stripped product.
  using ComparisonAtom = std::vector<Id>;
  struct ComparisonDifference {
    std::map<ComparisonAtom,Q> coefficients;
    Q offset=0;
  };
  static constexpr size_t max_comparison_terms=32;
  std::optional<ComparisonDifference> comparison_difference(Id a,Id b) const;
  struct ComparisonDescription {
    enum Kind { Raw, Affine, Constant, NotPrepared } kind=Raw;
    std::string op;
    Id lhs=invalid,rhs=invalid;
    ComparisonDifference form;
    Q threshold=0;
    bool truth=false;
  };
  struct BooleanComparisonStep {
    Id resolved=invalid;
    bool scaled=false,complemented=false;
  };
  BooleanComparisonStep normalize_boolean_comparison(std::string &op,Id &a,Id &b) const;
  ComparisonDescription normalize_comparison(std::string op,Id a,Id b,
      std::optional<ComparisonDifference> form) const;
  ComparisonDescription describe_comparison(std::string op,Id a,Id b) const;
  static bool same_comparison(const ComparisonDescription &a,const ComparisonDescription &b);
  bool select_is_max(Id p,Id a,Id b) const;
  std::optional<Id> encoding_comparison(const std::string &p, Id lhs, Id rhs);
  Id compare(std::string p, Id a, Id b);
  std::pair<Q, Q> guarded_bounds(Id p, Id arm, bool taken) const;
  int guarded_reciprocal(const Node &c, Id arm, bool on_true) const;
  Id select(Id p, Id a, Id b, bool infer_bounds = false);
  Id clamp(Id x, Q lo, Q hi);
  Id first_argmax(std::vector<std::pair<Q,Id>> members);
  std::set<Id> max_leaves(Id value) const;
  std::vector<std::pair<Q,Id>> tournament_members(Id arm,Id compared);
  std::optional<std::pair<Id,std::vector<Q>>> argmax_point_set(Id predicate);
  std::optional<Id> argmax_point_select(Id p,Id a,Id b);
  std::optional<std::pair<Id,Id>> max_selection_operands(Id p) const;
  Id tournament_maximum(Id x,Id y);
  std::optional<Id> masked_maximum(const std::set<Id> &leaves);
  // A masked tournament leaf select(G == d, c, u): G first_argmax, d and c constants.
  struct MaskedLeaf { Id argmax=invalid; Q index, mask; Id payload=invalid; };
  std::optional<MaskedLeaf> masked_leaf(Id w);
  std::optional<Id> payload_tournament_select(Id p,Id a,Id b);
  std::optional<Id> tournament_select(Id p,Id a,Id b,bool leaf);
  Id maximum(std::vector<Id> xs);
  Id bitxor(Id x, Id y);
  Id absolute(Id x);
  Id natural_log2();
  Id inverse_natural_log2() { return recip(natural_log2()); }
  bool is_natural_log2(Id n) const { return exact_ln2 != invalid && n == exact_ln2; }
  bool is_inverse_natural_log2(Id n) const {
    const auto &x = nodes[n];
    return exact_ln2 != invalid && x.op == "recip" && x.args.size() == 1 && x.args[0] == exact_ln2;
  }
  void natural_log2_factors(std::vector<Id> &factors);
  std::optional<std::pair<Id,Id>> exponent_bases(Id natural, Id binary);
  Q scale_of(Id x) const;
  std::pair<Q,Id> positive_scale(Id x);
  std::optional<Id> common_positive_scale(const std::vector<Id> &xs, const std::optional<Q> &largest);
  bool positive_view(Id s) const;
  Id logarithm(Id x, bool base2 = false);
  Id power(Id base, Id exponent);
  bool nonzero(Id n) {
    auto v = value(n);
    return (v && *v != 0) || (nodes[n].facts.finite && nodes[n].facts.pos);
  }
  Id recip(Id n);
  using Factors = std::pair<std::map<Id, int>, Q>;
  Factors factors(Id n);
  static std::map<Id, int> common(const std::map<Id, int> &a,
                                  const std::map<Id, int> &b);
  Id product(Factors f, const std::map<Id, int> &remove = {});
  Id divide(Id x, Id y);
  Id factor_sum(Id n);
  KnownBits known_bits(Id root);
  Id word_or(Id lhs,Id rhs,unsigned width);
  Id word_shr(Id source, Id shift, unsigned width, bool sign);
  Id word_shl(Id source, Id shift, unsigned width);
  Id signed_word(Id source,unsigned width);
  std::optional<Id> signed_arm(Id x,unsigned width);
  bool boolean_node(Id q) {
    const auto &b=nodes[q].bounds;
    return nodes[q].facts.finite && b && b->first>=0 && b->second<=1 && known_bits(q).integer;
  }
  bool boolean_implies(Id c,Id q);
  std::optional<Id> sentinel_clamp(Id p, Id yes, Id no);
  Id neginf_max_comparison(Id c) const;
  // The conditions q forced by m = -inf, for a value Max m: m = max(x_1, ..., x_n) through nested
  // Max nodes and positive constant multiples k * m' (k > 0). When m = -inf every leaf x is -inf (a
  // finite leaf would make m finite; no leaf may be +inf or undefined: each is checked to be finite
  // or -inf). A leaf select(q, -inf, h) with h finite is -inf exactly when q != 0, and so is a raw sum
  // c * select(q, -inf, h) + (finite terms) with c > 0. So m = -inf forces q != 0 for each such leaf.
  // Leaves of any other form force nothing. At most 1024 visited nodes (else no conditions). Memoized
  // per m. O(visited nodes).
  std::unordered_map<Id, std::vector<Id>> neginf_max_cache;
  const std::vector<Id> &neginf_max_conditions(Id m) {
    if (auto it = neginf_max_cache.find(m); it != neginf_max_cache.end()) return it->second;
    std::vector<Id> conditions, todo{m};
    std::unordered_set<Id> seen{m};
    size_t visits = 0;
    const auto masked_leaf = [&](Id s) -> Id {
      const auto &n = nodes[s];
      if (n.op == "select" && n.args.size() == 3 && neginf(n.args[1]) && nodes[n.args[2]].facts.finite)
        return n.args[0];
      return invalid;
    };
    while (!todo.empty()) {
      if (++visits > 1024) { conditions.clear(); break; }
      const Id id = todo.back(); todo.pop_back();
      const auto &n = nodes[id];
      if (!n.facts.finite_or_negative_infinity()) { conditions.clear(); break; }
      if (n.op == "max" || (n.op == "mul" && n.args.size() == 1 && n.data.size() == 1 && n.data[0] > 0)) {
        for (Id x : n.args) if (seen.insert(x).second) todo.push_back(x);
        continue;
      }
      Id q = masked_leaf(id);
      if (q == invalid && n.op == "add_raw" && n.args.size() == n.data.size()) {
        size_t extended = 0; Id found = invalid;
        for (size_t i = 0; i < n.args.size(); ++i) {
          if (nodes[n.args[i]].facts.finite) continue;
          ++extended;
          if (n.data[i] > 0) found = masked_leaf(n.args[i]);
        }
        if (extended == 1) q = found;
      }
      if (q != invalid && std::find(conditions.begin(), conditions.end(), q) == conditions.end())
        conditions.push_back(q);
    }
    ++stats["neginf_max_condition_scans"];
    return neginf_max_cache.emplace(m, std::move(conditions)).first->second;
  }
  bool neginf_forces_zero(Id c, Id x);
  std::optional<Id> neginf_sum_condition(Id s);
  std::optional<Id> neginf_guarded_arm(Id p, Id arm);
  Id bit_slice(Id source,unsigned lo,unsigned count);
  Id compute_bit_slice(Id source,unsigned lo,unsigned count);
  // One bit of a word built from symbolic left shifts, with what it says about a shift count when it
  // comes from one word_shl leaf alone: [count == rhs] (equal) or [count > rhs] (above).
  struct ShiftedBit {
    enum Kind { other, equal, above } kind = other;
    Id value = invalid, count = invalid, rhs = invalid;
    bool shifted = false; // a word_shl leaf decided it
    bool predicate = false; // a Select of two constant bits (its Boolean condition) decided it
  };
  using ShiftedBitMemo = std::map<std::pair<Id,Id>,std::optional<ShiftedBit>>;
  static constexpr size_t shifted_bit_visit_limit = 4096;
  static constexpr unsigned shifted_bit_depth_limit = 64;
  bool shift_dependent(Id root);
  std::optional<Id> shifted_word_bit(Id v, Id position);
  bool predicate_packed_word(Id root);
  std::optional<Id> packed_predicate_bit(Id v, Id position);
  std::optional<Id> shifted_word_mask(Id v, const Z &mask);
  std::optional<Id> packed_predicate_mask(Id v, const Z &mask);
  bool valid_bit_position(Id position) {
    const auto &b=nodes[position].bounds;
    return b && nodes[position].facts.finite && b->first>=0 && b->second<Q(256) && known_bits(position).integer;
  }
  std::optional<ShiftedBit> visit_shifted_bit(Id v, Id position, unsigned depth, ShiftedBitMemo &memo,
                                              size_t &visits);
  std::optional<ShiftedBit> compute_shifted_bit(Id v, Id position, unsigned depth, ShiftedBitMemo &memo,
                                                size_t &visits);
  std::optional<ShiftedBit> absorbed_shift_guard(Id p, const ShiftedBit &yes, const ShiftedBit &no);
  static std::optional<Q> decide_shifted_leaf(ShiftedBit::Kind kind, const std::pair<Q,Q> &counts,
                                              const std::pair<Q,Q> &rhs);
  Id count_comparison(ShiftedBit::Kind kind, Id count, Id rhs);
  std::optional<Id> boolean_select(Id q, bool when_true, bool when_false);
  std::optional<std::tuple<Id,Q,Q>> selected_constants(Id e, unsigned depth=0);
  // Where a Select condition p has a known truth and v (compared by p with a constant) lies in an
  // integer interval.
  struct GuardRegion { Id p; bool truth; Id v; std::pair<Q,Q> interval; };
  std::optional<bool> decided_condition(Id q, const GuardRegion &g);
  static constexpr unsigned guard_region_depth_limit = 8;
  std::optional<std::pair<Q,Q>> interval_on(Id e, const GuardRegion &g, unsigned depth=0);
  std::optional<Q> affine_on(Id e, const GuardRegion &g, unsigned depth=0);
  std::optional<std::pair<Q,Q>> count_interval(Id p, Id x, bool holds);
  std::optional<Id> concatenated_byte_view(Id source);
  std::optional<Id> shifted_byte_view_slice(Id source,unsigned lo,unsigned count);
  static Z common_factor(const std::vector<std::pair<Id, Q>> &ts, Z d);
  Id idiv(Id n, Z d);
  Id imod(Id n, Z d);
  Id read(std::string alloc, std::vector<Id> coords, std::string type,
          Bounds b = {});
  std::string render(Id id, int depth = 4) const;
  std::string render_shared(Id root, int depth, std::size_t limit) const;
};
} // namespace rg

#include "arena_storage.hpp"
#include "arena_arithmetic.hpp"
#include "arena_comparison.hpp"
#include "arena_maximum.hpp"
#include "arena_scalar_functions.hpp"
#include "arena_words.hpp"
#include "arena_nonfinite.hpp"
#include "arena_bit_analysis.hpp"
#include "arena_integer.hpp"
#include "arena_render.hpp"
