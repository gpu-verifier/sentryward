#pragma once
#include "core.hpp"

namespace rg {
// These are internal proof records, not an input format. The comparison runner
// may construct an equality only after proving the COMPLETE imported bodies
// equal for every assignment in their integer-variable domains. In particular,
// equal normalized roots are a valid complete proof too. The context identity
// names the immutable inputs, numerical contract and premises of that runner.
struct FinitePieceEquality {
  Id left_body = invalid, right_body = invalid;
  uint64_t context = 0;
};

// Produced only by the full-template instantiator, using capture-free
// simultaneous substitution of coordinates for the schema's formal axes.
// It may certify a body ID by exact whole-root comparison after capture-free
// alpha canonicalization of Fold binders. The instance IDs below then identify
// those certified ORIGINAL bodies, retaining the native proof's provenance.
// This module checks their exact IDs against the complete proof; it does not
// accept a matching read anchor or a shape as an instance proof.
struct FinitePieceInstanceWitness {
  Id left_template = invalid, right_template = invalid;
  Id left_instance = invalid, right_instance = invalid;
  std::vector<Id> coordinates;
  uint64_t context = 0;
};

struct FinitePieceUnionSchema {
  Id left_template = invalid, right_template = invalid;
  std::vector<Id> axes;
  uint64_t context = 0;
};

// Accepted images have fixed coordinates on every prefix axis and offset+t on
// the last axis. The domain of t comes from its Arena integer-var bounds; the
// domain of the rectangle comes from the schema's formal integer-var bounds.
// This deliberately excludes arbitrary strides, permutations and unions inside
// one piece. It includes symbolic four-element windows without expanding them.
struct FinitePieceCoordinates {
  Id local_axis = invalid;
  std::vector<Z> fixed_prefix;
  Z last_offset = 0;
};

struct FinitePieceUnionLimits {
  size_t pieces = 8192;
  size_t work = 524288;
};

class FinitePieceUnion {
public:
  static constexpr size_t max_axes = 4;
  static constexpr size_t max_pieces = 65536;
  // Bound integer-operation costs as well as the number of pieces. No array
  // extent is materialized; huge integer domains simply decline recognition.
  static constexpr unsigned integer_bits = 256;
  struct Piece {
    FinitePieceEquality equality;
    FinitePieceInstanceWitness instance;
    FinitePieceCoordinates coordinates;
    Z local_low, local_high, start, length;
  };
private:
  Arena &a;
  FinitePieceUnionSchema schema_;
  FinitePieceUnionLimits limits;
  std::vector<Z> lows, highs, strides;
  std::vector<Piece> pieces_;
  Z total = 0;
  size_t work_ = 0, sort_comparisons_ = 0;
  bool valid = false, sealed_ = false;
  struct BudgetExceeded {};

  bool id_valid(Id id) const { return id != invalid && size_t(id) < a.nodes.size(); }
  static bool small(const Z &n) {
    static const Z limit = pow2(integer_bits);
    return n > -limit && n < limit;
  }
  bool tick(size_t amount = 1) {
    a.check();
    if (amount > limits.work - std::min(work_, limits.work)) {
      work_ = limits.work;
      return decline("work_budget");
    }
    work_ += amount;
    return true;
  }
  bool decline(const char *reason) {
    last_failure = reason;
    ++a.stats[std::string("finite_piece_union_decline_") + reason];
    return false;
  }
  bool integer_var(Id id, Z &lo, Z &hi) const {
    if (!id_valid(id)) return false;
    const auto &n = a.nodes[id];
    if (n.op != "var" || !n.facts.finite || !n.bounds ||
        n.bounds->first.denominator() != 1 || n.bounds->second.denominator() != 1 ||
        !small(n.bounds->first.numerator()) || !small(n.bounds->second.numerator()) ||
        n.bounds->first > n.bounds->second) return false;
    lo = n.bounds->first.numerator(); hi = n.bounds->second.numerator();
    return true;
  }

public:
  std::string last_failure;
  FinitePieceUnion(Arena &arena, const FinitePieceUnionSchema &schema,
                   FinitePieceUnionLimits budget = {})
      : a(arena), limits(budget) {
    limits.pieces = std::min(limits.pieces, max_pieces);
    if (!tick()) return;
    if (!schema.context || schema.axes.empty() || schema.axes.size() > max_axes ||
        !id_valid(schema.left_template) || !id_valid(schema.right_template) ||
        !a.nodes[schema.left_template].facts.finite ||
        !a.nodes[schema.right_template].facts.finite) {
      decline("schema"); return;
    }
    // Validate rank before copying the caller's coordinate list.
    schema_ = schema;
    std::set<Id> seen;
    for (Id axis : schema_.axes) {
      if (!tick()) return;
      Z lo, hi;
      if (!seen.insert(axis).second || !integer_var(axis, lo, hi)) {
        decline("schema_axis"); return;
      }
      lows.push_back(lo); highs.push_back(hi);
    }
    // The theorem is a closed function of these coordinates. Keeping an old
    // local t as an extra free parameter could certify only j=offset+t while
    // falsely claiming equality for j independent of that t after union.
    // Numerical input values are read/atom nodes, not additional index vars.
    for (Id root : {schema_.left_template, schema_.right_template}) {
      const auto &free = a.nodes[root].free;
      if (free.size() > schema_.axes.size() ||
          !std::all_of(free.begin(), free.end(), [&](Id v) { return seen.contains(v); })) {
        decline("schema_free_variable"); return;
      }
    }
    strides.resize(lows.size()); total = 1;
    for (size_t d = lows.size(); d-- > 0;) {
      strides[d] = total;
      total *= highs[d] - lows[d] + 1;
      if (!small(total)) { decline("domain_size"); return; }
    }
    valid = true;
  }

  // Trust boundary: equality is a previously successful complete comparison;
  // instance is a checked full-template substitution. Everything connecting
  // those records to the coordinate cover is checked here. Rejection never
  // installs a partial piece, and does not change an already sealed theorem.
  bool add_proved_piece(const FinitePieceEquality &equality,
                        const FinitePieceCoordinates &coordinates,
                        const FinitePieceInstanceWitness &instance) {
    if (sealed_) return decline("sealed");
    if (!valid) return decline("invalid_schema");
    const size_t rank = schema_.axes.size();
    if (!tick(16 + 8 * rank)) return false;
    if (pieces_.size() >= limits.pieces) return decline("piece_budget");
    if (equality.context != schema_.context || instance.context != schema_.context)
      return decline("context");
    if (instance.left_template != schema_.left_template ||
        instance.right_template != schema_.right_template)
      return decline("template_identity");
    if (!id_valid(equality.left_body) || !id_valid(equality.right_body) ||
        !a.nodes[equality.left_body].facts.finite ||
        !a.nodes[equality.right_body].facts.finite ||
        instance.left_instance != equality.left_body ||
        instance.right_instance != equality.right_body)
      return decline("whole_body_instance");
    if (coordinates.fixed_prefix.size() != rank - 1 || instance.coordinates.size() != rank)
      return decline("coordinate_arity");
    Z local_low, local_high;
    if (!integer_var(coordinates.local_axis, local_low, local_high) ||
        std::find(schema_.axes.begin(), schema_.axes.end(), coordinates.local_axis) != schema_.axes.end())
      return decline("local_axis");
    // Each proved piece quantifies exactly its local coordinate (or a subset
    // when the body is constant). Fold binders are already absent from .free.
    for (Id root : {equality.left_body, equality.right_body}) {
      const auto &free = a.nodes[root].free;
      if (free.size() > 1 || (!free.empty() && free[0] != coordinates.local_axis))
        return decline("body_free_variable");
    }
    if (!small(coordinates.last_offset)) return decline("coordinate_size");

    Z start = 0;
    for (size_t d = 0; d + 1 < rank; ++d) {
      const Z &fixed = coordinates.fixed_prefix[d];
      if (!small(fixed) || fixed < lows[d] || fixed > highs[d])
        return decline("coordinate_range");
      if (instance.coordinates[d] != a.constant(Q(fixed)))
        return decline("coordinate_instance");
      start += (fixed - lows[d]) * strides[d];
    }
    const Z begin = coordinates.last_offset + local_low;
    const Z end = coordinates.last_offset + local_high;
    if (begin < lows.back() || end > highs.back()) return decline("coordinate_range");
    const Id expected = a.add({a.constant(Q(coordinates.last_offset)), coordinates.local_axis});
    // A singleton interval may have been exported with a constant coordinate.
    // This is exact substitution of its only domain value, not extrapolation.
    if (instance.coordinates.back() != expected &&
        !(local_low == local_high && instance.coordinates.back() == a.constant(Q(begin))))
      return decline("coordinate_instance");
    start += begin - lows.back();
    const Z length = local_high - local_low + 1;
    pieces_.push_back({equality, instance, coordinates, local_low, local_high, start, length});
    ++a.stats["finite_piece_union_accepted_pieces"];
    last_failure.clear();
    return true;
  }

  // Sorting B stored intervals costs O(B log B) comparisons and O(B) auxiliary
  // indices. No iteration runs over total, local t, or individual tensor cells.
  // Template construction/instantiation is deliberately outside this budget.
  bool seal() {
    if (sealed_) return true;
    if (!valid) return decline("invalid_schema");
    if (pieces_.empty()) return decline("empty");
    if (!tick(pieces_.size())) return false;
    std::vector<size_t> order(pieces_.size());
    std::iota(order.begin(), order.end(), size_t(0));
    try {
      std::sort(order.begin(), order.end(), [&](size_t x, size_t y) {
        if (!tick()) throw BudgetExceeded{};
        ++sort_comparisons_;
        if (pieces_[x].start != pieces_[y].start)
          return pieces_[x].start < pieces_[y].start;
        return pieces_[x].length < pieces_[y].length;
      });
    } catch (const BudgetExceeded &) { return false; }
    Z next = 0;
    for (size_t i : order) {
      if (!tick()) return false;
      const auto &piece = pieces_[i];
      if (piece.start < next) return decline("overlap");
      if (piece.start > next) return decline("hole");
      next += piece.length;
    }
    if (next != total) return decline("incomplete");
    sealed_ = true;
    ++a.stats["finite_piece_union_complete_domains"];
    last_failure.clear();
    return true;
  }

  bool sealed() const { return sealed_; }
  size_t work() const { return work_; }
  size_t sort_comparisons() const { return sort_comparisons_; }
  const FinitePieceUnionSchema &schema() const { return schema_; }
  const std::vector<Piece> &pieces() const { return pieces_; }
  const Z &domain_size() const { return total; }
  J receipt() const {
    J domains = J::array();
    for (size_t d = 0; d < lows.size(); ++d)
      domains.push_back({{"axis", schema_.axes[d]}, {"low", lows[d].str()}, {"high", highs[d].str()}});
    return {{"method", "finite_piece_union"}, {"complete", sealed_},
            {"context", schema_.context}, {"left_template", schema_.left_template},
            {"right_template", schema_.right_template}, {"domains", domains},
            {"pieces", pieces_.size()}, {"domain_size", total.str()},
            {"coverage", sealed_ ? "sorted_contiguous_intervals_exact_multiplicity_one" : "not_certified"},
            {"instances", "both_complete_bodies_equal_checked_template_instances"},
            {"work", work_}, {"sort_comparisons", sort_comparisons_}};
  }
};
} // namespace rg
