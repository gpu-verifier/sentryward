#include "sentryward/frontend/ptx/execution.h"
#include "template_execution.h"
#include "sentryward/frontend/ptx/shared_execution.h"
#include "sentryward/frontend/ptx/resources.h"
#include "sentryward/algorithm/diagnostics/performance_profile.h"

#include <algorithm>
#include <bit>
#include <cctype>
#include <cmath>
#include <coroutine>
#include <deque>
#include <exception>
#include <limits>
#include <map>
#include <regex>
#include <array>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace sentryward::frontend::ptx {
namespace {
// Last-resort re-executions after ordinary execution, including the shared-memory
// cluster retry, stopped on undetermined control (one kernel per process; per
// thread, since class mode executes units on look-ahead workers):
//   exact_control_pass: regions with no memory or synchronization effect;
//   data_branch_pass:   additionally Representative control flow representative outcomes.
thread_local bool exact_control_pass = false, data_branch_pass = false;
bool Assumed(const FrontendConfig &c, const char *key) {
  return c.raw.value("semantic_assumptions", nlohmann::json::object()).value(key, false);
}
bool DataBranchesAssumed(const FrontendConfig &c) { return Assumed(c, "representative_branches") || Assumed(c, "bounded_computation_loops"); }
} // namespace
std::vector<LocalMemoryDeclaration> LocalMemoryDeclarations(const Module& module, const Function& function) {
  static const std::regex pattern(R"(\.(?:b|u|s|f)(8|16|32|64)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\[\s*([0-9]+)\s*\]\s*;?\s*$)");
  std::vector<LocalMemoryDeclaration> result;
  for (const auto& directive : function.directives) {
    if (directive.name != ".local") continue;
    const auto text = std::string(module.view(directive.source));
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) continue;
    const auto width = std::stoull(match[1]) / 8;
    const auto count = std::stoull(match[3]);
    if (!count || count > UINT64_MAX / width) continue;
    std::string resource = "local:";
    for (const auto id : directive.scope) resource += std::to_string(id) + '/';
    resource += match[2].str();
    result.push_back({match[2], std::move(resource), directive.scope, count * width});
  }
  return result;
}
namespace {
using Json = nlohmann::json;
using sentryward::algorithm::ProfileScope;
constexpr std::uint32_t kMissing = UINT32_MAX;
std::string_view TrimView(std::string_view x) {
  while (!x.empty() && std::isspace(static_cast<unsigned char>(x.front())))
    x.remove_prefix(1);
  while (!x.empty() && std::isspace(static_cast<unsigned char>(x.back())))
    x.remove_suffix(1);
  return x;
}
bool In(std::string_view x, std::initializer_list<std::string_view> xs) {
  return std::find(xs.begin(), xs.end(), x) != xs.end();
}
bool Unknown(const Scalar &s) {
  return std::holds_alternative<std::monostate>(s);
}
std::optional<bool> Truth(const Scalar &s) {
  auto n = ScalarInteger(s);
  if (!n)
    return {};
  return *n != 0;
}
Scalar NegatePredicate(const Scalar &s) {
  auto truth = Truth(s);
  return truth ? Scalar(!*truth) : Scalar{};
}
Scalar PredicateBinary(std::string_view op, const Scalar &a, const Scalar &b) {
  auto x = Truth(a), y = Truth(b);
  if (op == "and") {
    if ((x && !*x) || (y && !*y))
      return false;
    if (x && y)
      return true;
  }
  if (op == "or") {
    if ((x && *x) || (y && *y))
      return true;
    if (x && y)
      return false;
  }
  if (op == "xor" && x && y)
    return *x != *y;
  return {};
}
std::optional<unsigned> Width(const Instruction &i) {
  for (const auto &m : i.modifiers)
    if (m.size() > 1 && In(m.substr(0, 1), {"b", "u", "s"}) &&
        std::all_of(m.begin() + 1, m.end(),
                    [](unsigned char c) { return std::isdigit(c); }))
      return std::stoul(m.substr(1));
  return {};
}
bool Floating(const Instruction &i) {
  for (const auto &m : i.modifiers)
    if (In(m, {"f16", "f16x2", "f32", "f64", "bf16", "bf16x2"}))
      return true;
  return false;
}
// Whether some modifier is a signed type: 's' followed by a digit.
bool SignedModifier(const Instruction &i) {
  return std::any_of(i.modifiers.begin(), i.modifiers.end(), [](const auto &modifier) {
    return modifier.size() > 1 && modifier[0] == 's' && std::isdigit(modifier[1]);
  });
}
std::int64_t Int64(const Integer &n, std::string_view what) {
  if (n < std::numeric_limits<std::int64_t>::min() ||
      n > std::numeric_limits<std::int64_t>::max())
    throw std::runtime_error(std::string(what) +
                             " outside signed 64-bit address range");
  return n.convert_to<std::int64_t>();
}
// Register integers are only meaningful modulo 2^width (see Compare), so
// operations that are not congruence-preserving first reduce operands to the
// instruction's width and signedness. Shift amounts are always u32.
Integer Reduce(const Integer &n, unsigned width, bool sign) {
  Integer modulus = Integer(1) << width, r = n & (modulus - 1);
  if (sign && (r & (modulus >> 1)) != 0) r -= modulus;
  return r;
}
// Interval arithmetic on BoundedInteger operands (tensor_properties); exact like Integer.
bool FitsType(const BoundedInteger &v, unsigned width, bool sign) {
  const Integer low = sign ? -(Integer(1) << (width - 1)) : Integer(0);
  const Integer high = (Integer(1) << (width - unsigned(sign))) - 1;
  return low <= v->lower && v->upper <= high;
}
std::vector<std::string> BoundedSources(std::initializer_list<const Scalar *> operands) {
  std::vector<std::string> sources;
  for (const auto *operand : operands) {
    if (const auto *b = std::get_if<BoundedInteger>(operand))
      sources.insert(sources.end(), (*b)->sources.begin(), (*b)->sources.end());
    if (const auto *p = std::get_if<BoundedAllocationPointer>(operand))
      sources.insert(sources.end(), (*p)->sources.begin(), (*p)->sources.end());
  }
  std::sort(sources.begin(), sources.end());
  sources.erase(std::unique(sources.begin(), sources.end()), sources.end());
  return sources;
}
Scalar MakeBounded(Integer lower, Integer upper, std::string identity, std::vector<std::string> sources,
                   std::optional<AffineForm> affine = std::nullopt, std::optional<BoundedTable> table = std::nullopt) {
  if (upper < lower) std::swap(lower, upper);
  return BoundedInteger{std::make_shared<const BoundedValue>(BoundedValue{std::move(lower), std::move(upper),
      std::move(identity), std::move(sources), std::move(affine), std::move(table)})};
}
Scalar MakeBoundedPointer(std::string resource, std::optional<std::uint64_t> extent, AffineForm offset, Integer lower,
                          Integer upper, std::vector<std::string> sources, std::string offset_identity = {},
                          std::optional<BoundedTable> table = std::nullopt) {
  return BoundedAllocationPointer{std::make_shared<const BoundedPointerValue>(BoundedPointerValue{
      std::move(resource), extent, std::move(offset), std::move(lower), std::move(upper), std::move(sources),
      std::move(offset_identity), std::move(table)})};
}
// Tables cover keys with at most this many values (O(entries) per tabulated operation).
constexpr std::uint64_t kBoundedTableLimit = 256;
// The exact values of a bounded value over a small-domain key: its own table, or, for a
// non-affine value with at most kBoundedTableLimit values, the identity table of itself.
// Affine values keep their affine forms instead (no table is built for them).
std::optional<BoundedTable> TableOf(const BoundedInteger &v) {
  if (v->table) return v->table;
  if (v->affine || v->upper - v->lower >= Integer(kBoundedTableLimit)) return std::nullopt;
  BoundedTable table{v->identity, v->lower, {}};
  for (Integer k = v->lower; k <= v->upper; ++k) table.values.push_back(k);
  return table;
}
// Whether two tables cover the same key: its name, first value and length.
bool SameKey(const BoundedTable &a, const BoundedTable &b) {
  return a.key == b.key && a.key_lower == b.key_lower && a.values.size() == b.values.size();
}
Scalar TableValue(BoundedTable table, std::string identity, std::vector<std::string> sources) {
  const auto [low, high] = std::minmax_element(table.values.begin(), table.values.end());
  return MakeBounded(*low, *high, std::move(identity), std::move(sources), std::nullopt, std::move(table));
}
std::string BoundedName(const Scalar &value) {
  if (const auto *b = std::get_if<BoundedInteger>(&value)) return (*b)->identity;
  return ScalarInteger(value)->convert_to<std::string>();
}
// The exact affine form of an integer or a bounded value, if it has one.
std::optional<AffineForm> AffineOf(const Scalar &value) {
  if (const auto *b = std::get_if<BoundedInteger>(&value)) return (*b)->affine;
  if (const auto n = ScalarInteger(value)) return AffineForm{*n, {}};
  return std::nullopt;
}
AffineForm AffineSum(const AffineForm &a, const AffineForm &b, const Integer &scale_b = Integer(1)) {
  AffineForm result{a.constant + scale_b * b.constant, {}};
  std::map<ElementAtom, Integer> terms;
  for (const auto &[atom, coefficient] : a.terms) terms[atom] += coefficient;
  for (const auto &[atom, coefficient] : b.terms) terms[atom] += scale_b * coefficient;
  for (auto &[atom, coefficient] : terms) if (coefficient != 0) result.terms.emplace_back(atom, coefficient);
  return result;
}
AffineForm AffineScale(const AffineForm &a, const Integer &scale) {
  AffineForm result{a.constant * scale, {}};
  if (scale != 0) for (const auto &[atom, coefficient] : a.terms) result.terms.emplace_back(atom, coefficient * scale);
  return result;
}
Scalar IntegerBinary(std::string_view op, const Scalar &a, const Scalar &b, const Instruction *i);
// std::nullopt when neither operand is bounded; unknown when the result has no bound here.
std::optional<Scalar> BoundedBinary(std::string_view op, const Scalar &a, const Scalar &b,
                                    const Instruction *i) {
  const auto *p = std::get_if<BoundedInteger>(&a), *q = std::get_if<BoundedInteger>(&b);
  if (!p && !q) return std::nullopt;
  const auto x = ScalarInteger(a), y = ScalarInteger(b);
  if ((!p && !x) || (!q && !y)) return Scalar{};
  const Integer al = p ? (*p)->lower : *x, ah = p ? (*p)->upper : *x, bl = q ? (*q)->lower : *y, bh = q ? (*q)->upper : *y;
  const auto identity = std::string(op) + "(" + BoundedName(a) + "," + BoundedName(b) + ")";
  const auto sources = BoundedSources({&a, &b});
  // Tables over one key (or a table and a constant): concrete execution's operator per
  // entry, exact and at least as tight as the intervals. Any entry it leaves unknown
  // (or two keys) falls back to the interval rules below.
  {
    const auto ta = p ? TableOf(*p) : std::nullopt, tb = q ? TableOf(*q) : std::nullopt;
    if ((!p || ta) && (!q || tb) && (ta || tb) && (!ta || !tb || SameKey(*ta, *tb))) {
      const auto &key = ta ? *ta : *tb;
      BoundedTable table{key.key, key.key_lower, {}};
      table.values.reserve(key.values.size());
      for (std::size_t k = 0; k < key.values.size(); ++k) {
        const auto entry = ScalarInteger(IntegerBinary(op, Scalar(ta ? ta->values[k] : *x),
                                                       Scalar(tb ? tb->values[k] : *y), i));
        if (!entry) { table.values.clear(); break; }
        table.values.push_back(*entry);
      }
      if (!table.values.empty()) return TableValue(std::move(table), identity, sources);
    }
  }
  const auto fa = AffineOf(a), fb = AffineOf(b);
  if (op == "add") return MakeBounded(al + bl, ah + bh, identity, sources,
      fa && fb ? std::optional(AffineSum(*fa, *fb)) : std::nullopt);
  if (op == "sub") return MakeBounded(al - bh, ah - bl, identity, sources,
      fa && fb ? std::optional(AffineSum(*fa, *fb, Integer(-1))) : std::nullopt);
  if (op == "mul") {
    const std::array<Integer, 4> corners{al * bl, al * bh, ah * bl, ah * bh};
    std::optional<AffineForm> affine;
    if (!p && fb) affine = AffineScale(*fb, *x);
    if (!q && fa) affine = AffineScale(*fa, *y);
    return MakeBounded(*std::min_element(corners.begin(), corners.end()),
                       *std::max_element(corners.begin(), corners.end()), identity, sources, affine);
  }
  if (op == "shl" && !q && *y >= 0 && *y < 64) {
    const Integer scale = Integer(1) << y->convert_to<unsigned>();
    return MakeBounded(al * scale, ah * scale, identity, sources,
                       fa ? std::optional(AffineScale(*fa, scale)) : std::nullopt);
  }
  // shr, div and rem by a constant of a value that fits the instruction's type:
  // concrete execution's own operators applied to the bounds. Both shifts (logical
  // on a nonnegative value, arithmetic) and division by c > 0 are monotone; a
  // nonnegative remainder stays in [0, c-1], exactly [lo%c, hi%c] within one quotient.
  if (In(op, {"shr", "div", "rem"}) && !q && i) {
    const auto width = Width(*i);
    if (!width) return Scalar{};
    const bool sign = Has(*i, "s" + std::to_string(*width));
    if (!FitsType(*p, *width, sign)) return Scalar{};
    if (op == "shr") {
      const Integer n = Reduce(*y, 32, false);
      if (n > 1000000) return Scalar{};
      const auto k = n.convert_to<unsigned>();
      return MakeBounded(Integer(al >> k), Integer(ah >> k), identity, sources);
    }
    const Integer c = Reduce(*y, *width, sign);
    if (c != *y || c <= 0) return Scalar{};
    if (op == "div") return MakeBounded(Integer(al / c), Integer(ah / c), identity, sources);
    if (al < 0) return Scalar{};
    if (al / c == ah / c) return MakeBounded(Integer(al % c), Integer(ah % c), identity, sources);
    return MakeBounded(Integer(0), std::min<Integer>(ah, c - 1), identity, sources);
  }
  // A nonnegative value masked by a nonnegative constant stays within [0, min(value, mask)];
  // a low-bit mask that covers the whole range leaves the value unchanged.
  if (op == "and" && !q && *y >= 0 && al >= 0) {
    if (((*y + 1) & *y) == 0 && ah <= *y) return a;
    return MakeBounded(Integer(0), std::min(ah, *y), identity, sources);
  }
  if (In(op, {"min", "max"}) && i) {
    const auto width = Width(*i);
    if (!width) return Scalar{};
    const bool sign = Has(*i, "s" + std::to_string(*width));
    for (const auto *v : {p, q}) if (v && !FitsType(*v, *width, sign)) return Scalar{};
    if (!p && Reduce(*x, *width, sign) != *x) return Scalar{};
    if (!q && Reduce(*y, *width, sign) != *y) return Scalar{};
    // An operation the bounds decide returns that operand unchanged.
    if (op == "min" ? ah <= bl : al >= bh) return a;
    if (op == "min" ? bh <= al : bl >= ah) return b;
    return op == "min" ? MakeBounded(std::min(al, bl), std::min(ah, bh), identity, sources)
                       : MakeBounded(std::max(al, bl), std::max(ah, bh), identity, sources);
  }
  return Scalar{};
}
Scalar IntegerBinary(std::string_view op, const Scalar &a, const Scalar &b,
                     const Instruction *i = nullptr) {
  if (auto bounded = BoundedBinary(op, a, b, i)) return *bounded;
  auto x = ScalarInteger(a), y = ScalarInteger(b);
  if (op == "mul" && (Unknown(a) || x) && (Unknown(b) || y) &&
      ((x && *x == 0) || (y && *y == 0))) return Integer(0);
  if (!x || !y)
    return {};
  if (auto width = i ? Width(*i) : std::nullopt;
      width && In(op, {"shr", "min", "max", "div", "rem"})) {
    const bool sign = Has(*i, "s" + std::to_string(*width));
    *x = Reduce(*x, *width, sign);
    *y = op == "shr" ? Reduce(*y, 32, false) : Reduce(*y, *width, sign);
    if (op == "div" || op == "rem") {
      if (*y == 0) return {};
      // boost::multiprecision truncates toward zero, as PTX div/rem do.
      return op == "div" ? Integer(*x / *y) : Integer(*x % *y);
    }
  }
  if (op == "add")
    return Integer(*x + *y);
  if (op == "sub")
    return Integer(*x - *y);
  if (op == "mul")
    return Integer(*x * *y);
  if (op == "and")
    return Integer(*x & *y);
  if (op == "or")
    return Integer(*x | *y);
  if (op == "xor")
    return Integer(*x ^ *y);
  if (op == "min")
    return Integer(std::min(*x, *y));
  if (op == "max")
    return Integer(std::max(*x, *y));
  if (op == "shl" || op == "shr") {
    if (*y < 0)
      return {};
    if (*y > 1000000)
      throw std::runtime_error(
          "PTX shift exceeds finite execution resource limit");
    auto n = y->convert_to<unsigned>();
    if (op == "shl")
      return Integer(*x << n);
    return Integer(*x >> n);
  }
  if (op == "div" || op == "rem") {
    if (*y == 0)
      return {};
    Integer q = *x / *y;
    Integer r = *x % *y;
    if (r != 0 && ((*x < 0) != (*y < 0))) {
      --q;
      r += *y;
    }
    return op == "div" ? q : r;
  }
  return {};
}
std::string AllocationIdentity(const AllocationPointer &p, const FrontendConfig &c) {
  for (const auto &r : c.resources)
    if (r.value("id", "") == p.resource && r.contains("allocation"))
      return r["allocation"].value("object", p.resource);
  return p.resource;
}
// A pointer plus a bounded affine offset stays inside its resource (tensor_properties).
// The exact offset range must lie in [0, 2^63): it then equals the hardware address
// (bounded values are exact and congruent to their registers modulo 2^64).
std::optional<Scalar> BoundedPointerArithmetic(std::string_view op, const Scalar &a, const Scalar &b) {
  const bool pointer_a = std::holds_alternative<BoundedAllocationPointer>(a) ||
      (std::holds_alternative<AllocationPointer>(a) && std::holds_alternative<BoundedInteger>(b));
  const bool pointer_b = std::holds_alternative<BoundedAllocationPointer>(b) ||
      (std::holds_alternative<AllocationPointer>(b) && std::holds_alternative<BoundedInteger>(a));
  if (!pointer_a && !pointer_b) return std::nullopt;
  if ((op != "add" && op != "sub") || (op == "sub" && !pointer_a)) return Scalar{};
  const auto &pointer = pointer_a ? a : b, &offset = pointer_a ? b : a;
  std::string resource;
  std::optional<std::uint64_t> extent;
  AffineForm base;
  Integer lower, upper;
  const BoundedPointerValue *tabled = nullptr; // a base with a non-affine offset
  if (const auto *p = std::get_if<AllocationPointer>(&pointer)) {
    resource = p->resource; extent = p->extent_bytes; base = {p->offset, {}}; lower = upper = p->offset;
  } else if (const auto *q = std::get_if<BoundedAllocationPointer>(&pointer)) {
    resource = (*q)->resource; extent = (*q)->extent_bytes; base = (*q)->offset;
    lower = (*q)->lower; upper = (*q)->upper;
    if (!(*q)->offset_identity.empty()) tabled = q->value.get();
  } else return Scalar{};
  const auto form = AffineOf(offset);
  const Integer sign = op == "sub" ? -1 : 1;
  if (!form || tabled) {
    // A non-affine offset: its exact byte offsets over one small-domain key (a table
    // base plus a constant or a same-key table; a concrete base plus a table).
    const auto *bounded = std::get_if<BoundedInteger>(&offset);
    const auto constant = ScalarInteger(offset);
    const auto offsets_of = bounded ? TableOf(*bounded) : std::nullopt;
    BoundedTable offsets;
    if (tabled) {
      if (!tabled->table) return Scalar{};
      offsets = *tabled->table;
      if (constant) {
        for (auto &value : offsets.values) value += sign * *constant;
      } else {
        if (!offsets_of || !SameKey(*offsets_of, offsets)) return Scalar{};
        for (std::size_t k = 0; k < offsets.values.size(); ++k) offsets.values[k] += sign * offsets_of->values[k];
      }
    } else {
      if (!offsets_of || !base.terms.empty()) return Scalar{};
      offsets = *offsets_of;
      for (auto &value : offsets.values) value = base.constant + sign * value;
    }
    const auto [low, high] = std::minmax_element(offsets.values.begin(), offsets.values.end());
    if (*low < 0 || *high >= (Integer(1) << 63)) return Scalar{};
    const auto identity = (tabled ? tabled->offset_identity : base.constant.convert_to<std::string>()) +
        (op == "sub" ? "-" : "+") + BoundedName(offset);
    const Integer lo = *low, hi = *high;
    return MakeBoundedPointer(std::move(resource), extent, AffineForm{0, {}}, lo, hi, BoundedSources({&a, &b}),
                              identity, std::move(offsets));
  }
  Integer low, high;
  if (const auto *bounded = std::get_if<BoundedInteger>(&offset)) { low = (*bounded)->lower; high = (*bounded)->upper; }
  else low = high = *ScalarInteger(offset);
  if (sign < 0) { std::swap(low, high); low = -low; high = -high; }
  lower += low; upper += high;
  if (lower < 0 || upper >= (Integer(1) << 63)) return Scalar{};
  return MakeBoundedPointer(std::move(resource), extent, AffineSum(base, *form, sign), std::move(lower),
                            std::move(upper), BoundedSources({&a, &b}));
}
std::optional<Scalar> PointerArithmetic(std::string_view op, const Scalar &a,
                                        const Scalar &b, const FrontendConfig &c) {
  if (auto bounded = BoundedPointerArithmetic(op, a, b)) return bounded;
  bool ap = a.index() >= 3 && !std::holds_alternative<DescriptorFragment>(a) &&
      !std::holds_alternative<BoundedInteger>(a);
  bool bp = b.index() >= 3 && !std::holds_alternative<DescriptorFragment>(b) &&
      !std::holds_alternative<BoundedInteger>(b);
  if (!ap && !bp)
    return {};
  auto x = ScalarInteger(a), y = ScalarInteger(b);
  if (op=="shr" && y && *y==32 && std::holds_alternative<AllocationPointer>(a))
    return Scalar(*PointerFragment(a,32,32));
  if (const auto *bits=std::get_if<PointerBits>(&a); bits && op=="shl" && y && *y>=0 && *y+bits->destination_bit+bits->width<=64) {
    auto shifted=*bits; shifted.destination_bit+=y->convert_to<unsigned>(); return Scalar(shifted);
  }
  if (op=="or") {
    const auto *left=std::get_if<PointerBits>(&a), *right=std::get_if<PointerBits>(&b);
    if (left && right) {
      if (left->destination_bit>right->destination_bit) std::swap(left,right);
      if (*left->pointer==*right->pointer && left->source_bit==0 && left->destination_bit==0 && left->width==32 &&
          right->source_bit==32 && right->destination_bit==32 && right->width==32) {
        auto whole=*left; whole.width=64; return RestorePointer(whole);
      }
    }
  }
  auto advance = [](const Scalar &p, const Integer &n) -> Scalar {
    if (auto v = std::get_if<ParameterPointer>(&p))
      return ParameterPointer{v->region, Integer(v->offset + n)};
    if (auto v = std::get_if<AllocationPointer>(&p))
      return AllocationPointer{v->resource, Integer(v->offset + n),
                               v->extent_bytes};
    if (auto v = std::get_if<ClusterSharedPointer>(&p))
      return ClusterSharedPointer{Integer(v->local_address + n),
                                  v->target_rank};
    return {};
  };
  if (op == "add") {
    if (ap && y)
      return advance(a, *y);
    if (bp && x)
      return advance(b, *x);
  }
  if (op == "sub") {
    if (ap && y)
      return advance(a, Integer(-*y));
    if (auto left = std::get_if<ParameterPointer>(&a))
      if (auto right = std::get_if<ParameterPointer>(&b);
          right && left->region == right->region)
        return Scalar(Integer(left->offset - right->offset));
    if (auto left = std::get_if<AllocationPointer>(&a))
      if (auto right = std::get_if<AllocationPointer>(&b);
          right && AllocationIdentity(*left,c) == AllocationIdentity(*right,c))
        return Scalar(Integer(left->offset - right->offset));
    if (auto left = std::get_if<ClusterSharedPointer>(&a))
      if (auto right = std::get_if<ClusterSharedPointer>(&b);
          right && left->target_rank == right->target_rank)
        return Scalar(Integer(left->local_address - right->local_address));
  }
  return Scalar{};
}
Scalar IntegerProduct(const Instruction &i, const Scalar &a, const Scalar &b) {
  if (std::holds_alternative<BoundedInteger>(a) || std::holds_alternative<BoundedInteger>(b)) {
    // mul.lo is congruent to the exact product; mul.wide is exact for in-range operands.
    const auto width = Width(i);
    if (!width || Has(i, "hi")) return {};
    const bool sign = Has(i, "s" + std::to_string(*width));
    for (const auto *v : {&a, &b}) {
      if (const auto *bounded = std::get_if<BoundedInteger>(v); bounded && !FitsType(*bounded, *width, sign)) return {};
      if (const auto n = ScalarInteger(*v); n && Reduce(*n, *width, sign) != *n) return {};
    }
    return BoundedBinary("mul", a, b, &i).value_or(Scalar{});
  }
  auto x = ScalarInteger(a), y = ScalarInteger(b);
  if ((Unknown(a) || x) && (Unknown(b) || y) &&
      ((x && *x == 0) || (y && *y == 0))) return Integer(0);
  if (!x || !y)
    return {};
  auto width = Width(i);
  if (!width)
    return Integer(*x * *y);
  Integer modulus = Integer(1) << *width, mask = modulus - 1;
  bool sign = Has(i, "s" + std::to_string(*width));
  Integer lhs = *x & mask, rhs = *y & mask;
  if (sign) {
    if ((lhs & (modulus >> 1)) != 0)
      lhs -= modulus;
    if ((rhs & (modulus >> 1)) != 0)
      rhs -= modulus;
  }
  Integer product = lhs * rhs;
  if (Has(i, "wide"))
    return product;
  if (Has(i, "hi"))
    return Integer(product >> *width);
  Integer result = product & mask;
  if (sign && (result & (modulus >> 1)) != 0)
    result -= modulus;
  return result;
}
Scalar CompareInteger(std::string_view relation, const Integer &a,
                      const Integer &b) {
  if (relation == "eq")
    return a == b;
  if (relation == "ne")
    return a != b;
  if (relation == "lt")
    return a < b;
  if (relation == "le")
    return a <= b;
  if (relation == "gt")
    return a > b;
  if (relation == "ge")
    return a >= b;
  return {};
}
Scalar ExactIntegerToFloat(const Scalar &value, const Instruction &instruction) {
  std::vector<std::string> types;
  for (const auto &m : instruction.modifiers)
    if (In(m,{"s8","u8","s16","u16","s32","u32","s64","u64","f16","f32","f64"})) types.push_back(m);
    else if (!In(m,{"rn","rz","rm","rp","ftz"})) return {};
  auto number=ScalarInteger(value);
  if (!number || types.size()!=2 || types[0][0]!='f' || !In(types[1].substr(0,1),{"s","u"})) return {};
  const auto width=std::stoul(types[1].substr(1));
  Integer n=*number&((Integer(1)<<width)-1);
  if (types[1][0]=='s' && ((n>>(width-1))&1)!=0) n-=Integer(1)<<width;
  if (n==0) return Integer(0);
  const bool negative=n<0; if (negative) n=-n;
  const unsigned fraction=types[0]=="f16"?10:types[0]=="f32"?23:52;
  const unsigned exponent_bits=types[0]=="f16"?5:types[0]=="f32"?8:11;
  const unsigned high=boost::multiprecision::msb(n), bias=(1U<<(exponent_bits-1))-1;
  if (high>bias) return {};
  Integer significand;
  if (high>fraction) {
    const auto shift=high-fraction;
    if ((n&((Integer(1)<<shift)-1))!=0) return {};
    significand=n>>shift;
  } else significand=n<<(fraction-high);
  return Integer((Integer(negative)<<(fraction+exponent_bits)) |
      (Integer(high+bias)<<fraction) | (significand&((Integer(1)<<fraction)-1)));
}
// Whether one of the six integer relations holds between a value and itself.
bool ReflexiveRelation(std::string_view relation) {
  return relation == "eq" || relation == "le" || relation == "ge";
}
// The integer relation between values in [al, ah] and [bl, bh], where the ranges
// decide it: eq and ne for disjoint ranges, and the four order relations.
std::optional<bool> IntervalOrder(std::string_view relation, const Integer &al, const Integer &ah,
                                  const Integer &bl, const Integer &bh) {
  if (relation == "eq" || relation == "ne") {
    if (ah < bl || bh < al) return relation == "ne";
    return std::nullopt;
  }
  if (relation == "lt") { if (ah < bl) return true; if (al >= bh) return false; }
  if (relation == "le") { if (ah <= bl) return true; if (al > bh) return false; }
  if (relation == "gt") { if (al > bh) return true; if (ah <= bl) return false; }
  if (relation == "ge") { if (al >= bh) return true; if (ah < bl) return false; }
  return std::nullopt;
}
// An integer comparison decided by the operands' bounds; values that may wrap at
// the comparison width stay unknown.
Scalar BoundedCompare(const Instruction &i, std::string_view relation, const Scalar &a, const Scalar &b) {
  const auto *p = std::get_if<BoundedInteger>(&a), *q = std::get_if<BoundedInteger>(&b);
  const auto x = ScalarInteger(a), y = ScalarInteger(b);
  const auto width = Width(i);
  if ((!p && !x) || (!q && !y) || !width || Floating(i)) return {};
  const bool sign = Has(i, "s" + std::to_string(*width));
  for (const auto *v : {p, q}) if (v && !FitsType(*v, *width, sign)) return {};
  const Integer al = p ? (*p)->lower : Reduce(*x, *width, sign), ah = p ? (*p)->upper : al;
  const Integer bl = q ? (*q)->lower : Reduce(*y, *width, sign), bh = q ? (*q)->upper : bl;
  if (p && q && *p == *q && In(relation, {"eq", "ne", "le", "ge", "lt", "gt"}))
    return ReflexiveRelation(relation);
  // Tables over one key (or a table and a constant): the relation per entry, decided
  // when every entry agrees.
  if (In(relation, {"eq", "ne", "le", "ge", "lt", "gt"})) {
    const auto ta = p ? TableOf(*p) : std::nullopt, tb = q ? TableOf(*q) : std::nullopt;
    if ((!p || ta) && (!q || tb) && (ta || tb) && (!ta || !tb || SameKey(*ta, *tb))) {
      const auto size = (ta ? *ta : *tb).values.size();
      std::optional<bool> decided;
      bool agree = true;
      for (std::size_t k = 0; agree && k < size; ++k) {
        const Integer u = ta ? ta->values[k] : al, v = tb ? tb->values[k] : bl;
        const bool holds = relation == "eq" ? u == v : relation == "ne" ? u != v : relation == "lt" ? u < v :
                           relation == "le" ? u <= v : relation == "gt" ? u > v : u >= v;
        if (!decided) decided = holds;
        else if (*decided != holds) agree = false;
      }
      if (agree && decided) return *decided;
    }
  }
  if (const auto order = IntervalOrder(relation, al, ah, bl, bh)) return *order;
  // Both exact (one value each) and not disjoint: the same value.
  if ((relation == "eq" || relation == "ne") && al == ah && bl == bh) return relation == "eq";
  return {};
}
Scalar Compare(const Instruction &i, std::string relation, const Scalar &a,
               const Scalar &b, const FrontendConfig &c) {
  auto x = ScalarInteger(a), y = ScalarInteger(b);
  if (Unknown(a) || Unknown(b))
    return {};
  if (std::holds_alternative<BoundedInteger>(a) || std::holds_alternative<BoundedInteger>(b))
    return BoundedCompare(i, relation, a, b);
  if (std::holds_alternative<BoundedAllocationPointer>(a) || std::holds_alternative<BoundedAllocationPointer>(b)) {
    // A bounded pointer is never null; pointers into different allocations differ; within one
    // resource, the offset ranges decide the comparison or leave it unknown.
    const auto range = [](const Scalar &v) -> std::optional<std::tuple<std::string, Integer, Integer>> {
      if (const auto *p = std::get_if<AllocationPointer>(&v)) return std::tuple{p->resource, p->offset, p->offset};
      if (const auto *q = std::get_if<BoundedAllocationPointer>(&v)) return std::tuple{(*q)->resource, (*q)->lower, (*q)->upper};
      return std::nullopt;
    };
    const auto left = range(a), right = range(b);
    if ((!left && x && *x == 0) || (!right && y && *y == 0))
      return In(relation, {"eq", "ne"}) ? Scalar(relation == "ne") : Scalar{};
    if (!left || !right || Floating(i)) return {};
    const auto &[left_resource, al, ah] = *left;
    const auto &[right_resource, bl, bh] = *right;
    if (left_resource != right_resource) {
      const AllocationPointer lp{left_resource, 0, std::nullopt}, rp{right_resource, 0, std::nullopt};
      if (AllocationIdentity(lp, c) != AllocationIdentity(rp, c) && In(relation, {"eq", "ne"})) return relation == "ne";
      return {};
    }
    if (a == b && In(relation, {"eq", "ne", "le", "ge", "lt", "gt"}))
      return ReflexiveRelation(relation);
    if (const auto order = IntervalOrder(relation, al, ah, bl, bh)) return *order;
    return {};
  }
  if (Has(i, "f32")) {
    if (!x || !y)
      return {};
    auto bits = [](const Integer &n, bool ftz) {
      auto b = (n & 0xffffffff).convert_to<std::uint32_t>();
      if (ftz && (b & 0x7f800000) == 0 && (b & 0x7fffff))
        b &= 0x80000000;
      return b;
    };
    float lhs = std::bit_cast<float>(bits(*x, Has(i, "ftz"))),
          rhs = std::bit_cast<float>(bits(*y, Has(i, "ftz")));
    bool nan = std::isnan(lhs) || std::isnan(rhs);
    if (relation == "num")
      return !nan;
    if (relation == "nan")
      return nan;
    if (In(relation, {"equ", "neu", "ltu", "leu", "gtu", "geu"})) {
      if (nan)
        return true;
      relation.pop_back();
    } else if (nan)
      return false;
    if (relation == "eq")
      return lhs == rhs;
    if (relation == "ne")
      return lhs != rhs;
    if (relation == "lt")
      return lhs < rhs;
    if (relation == "le")
      return lhs <= rhs;
    if (relation == "gt")
      return lhs > rhs;
    if (relation == "ge")
      return lhs >= rhs;
    return {};
  }
  if (a.index() >= 3 || b.index() >= 3) {
    if (std::holds_alternative<DescriptorFragment>(a) || std::holds_alternative<DescriptorFragment>(b)) {
      if (a==b && In(relation,{"eq","ne"})) return relation=="eq";
      return {};
    }
    if (std::holds_alternative<PointerBits>(a) || std::holds_alternative<PointerBits>(b)) {
      if (a==b && In(relation,{"eq","ne"})) return relation=="eq";
      return {};
    }
    if (a.index() == b.index()) {
      if (auto l = std::get_if<AllocationPointer>(&a)) {
        const auto &r = std::get<AllocationPointer>(b);
        if (AllocationIdentity(*l,c) == AllocationIdentity(r,c))
          return CompareInteger(relation,l->offset,r.offset);
        if (relation == "eq" || relation == "ne") return relation == "ne";
        return {};
      }
      if (relation == "eq")
        return a == b;
      if (relation == "ne")
        return a != b;
      if (auto l = std::get_if<ParameterPointer>(&a)) {
        const auto &r = std::get<ParameterPointer>(b);
        if (l->region == r.region)
          return CompareInteger(relation, l->offset, r.offset);
      }
      if (auto l = std::get_if<AllocationPointer>(&a)) {
        const auto &r = std::get<AllocationPointer>(b);
        if (l->resource == r.resource)
          return CompareInteger(relation, l->offset, r.offset);
      }
    } else if ((std::holds_alternative<ParameterPointer>(a) ||
                std::holds_alternative<AllocationPointer>(a)) &&
               y && *y == 0 && In(relation, {"eq", "ne"}))
      return relation == "ne";
    else if ((std::holds_alternative<ParameterPointer>(b) ||
              std::holds_alternative<AllocationPointer>(b)) &&
             x && *x == 0 && In(relation, {"eq", "ne"}))
      return relation == "ne";
    return {};
  }
  if (!x || !y)
    return {};
  if (auto width = Width(i)) {
    Integer modulus = Integer(1) << *width, mask = modulus - 1;
    *x &= mask;
    *y &= mask;
    if (Has(i, "s" + std::to_string(*width))) {
      if ((*x & (modulus >> 1)) != 0)
        *x -= modulus;
      if ((*y & (modulus >> 1)) != 0)
        *y -= modulus;
    }
  }
  return CompareInteger(relation, *x, *y);
}
struct Operand {
  std::string text;
  std::uint32_t reg = kMissing;
  Scalar constant;
  std::vector<std::uint32_t> registers;
  std::optional<std::pair<std::string, Integer>> address;
};
struct Decoded {
  const Instruction *instruction = nullptr;
  std::vector<Operand> operands;
  std::vector<std::uint32_t> uses, destinations;
  std::vector<std::uint32_t> destination_slots;
  std::uint32_t predicate = kMissing;
  std::optional<std::pair<std::string, std::uint32_t>> wait_loop;
  bool replay_wait_result = false;
};
struct State {
  std::vector<Scalar> registers;
  std::vector<std::optional<DynamicKey>> origins;
  std::vector<std::uint64_t> occurrences, branch_visits;
  std::map<std::string, std::uint64_t> backedges;
};
struct Deferred {
  std::uint32_t lane, instruction;
  std::uint64_t occurrence;
  bool shared = false;
};
class LaneTask {
public:
  struct promise_type {
    std::optional<LaneTrace> result;
    std::optional<Deferred> dependency;
    std::exception_ptr error;
    LaneTask get_return_object() {
      return LaneTask(std::coroutine_handle<promise_type>::from_promise(*this));
    }
    std::suspend_always initial_suspend() noexcept { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }
    std::suspend_always yield_value(Deferred value) noexcept {
      dependency = value;
      return {};
    }
    void return_value(LaneTrace trace) { result = std::move(trace); }
    void unhandled_exception() noexcept { error = std::current_exception(); }
  };
  explicit LaneTask(std::coroutine_handle<promise_type> handle) : handle_(handle) {}
  LaneTask(LaneTask &&other) noexcept : handle_(std::exchange(other.handle_, {})) {}
  LaneTask(const LaneTask &) = delete;
  ~LaneTask() { if (handle_) handle_.destroy(); }
  bool Done() const { return handle_.done(); }
  const auto &Dependency() const { return handle_.promise().dependency; }
  void Resume() {
    handle_.promise().dependency.reset();
    handle_.resume();
    if (handle_.promise().error) std::rethrow_exception(handle_.promise().error);
  }
  LaneTrace TakeResult() { return std::move(*handle_.promise().result); }
private:
  std::coroutine_handle<promise_type> handle_;
};
struct ExecutorResources;
struct WarpContext {
  std::map<std::tuple<std::uint32_t, std::uint64_t, std::uint32_t>, Scalar>
      shuffle;
  std::map<std::tuple<std::uint32_t, std::uint64_t, std::uint32_t>, std::uint32_t>
      vote_masks;
  SharedExecution *shared = nullptr;
  ExecutorResources *resources = nullptr;
};
// A producer Resources with the ExecutorFacts and UnitFrame it reads and names
// through, in one allocation (facts and frame first: they outlive the
// Resources). Its holders keep one pointer, and the template task's
// coroutine frame does not grow.
struct ExecutorResources {
  ExecutorFacts facts;
  UnitFrame frame;
  Resources resources;
  ExecutorResources(const ExecutionProgram &in_progress, Coordinates coordinates, AddressTable &addresses,
                    BarrierTable &barriers)
      : facts(in_progress), frame(coordinates, in_progress.config.cluster_dim),
        resources(facts, frame, addresses, barriers, SlotConfig{Consumer::Executor, {}}) {}
};
class TemplateTask {
public:
  struct promise_type {
    std::optional<ExecutionProgram> result;
    std::exception_ptr error;
    TemplateTask get_return_object() {return TemplateTask(std::coroutine_handle<promise_type>::from_promise(*this));}
    std::suspend_always initial_suspend() noexcept {return {};}
    std::suspend_always final_suspend() noexcept {return {};}
    std::suspend_always yield_value(bool) noexcept {return {};}
    void return_value(std::optional<ExecutionProgram> value) {result=std::move(value);}
    void unhandled_exception() noexcept {error=std::current_exception();}
  };
  explicit TemplateTask(std::coroutine_handle<promise_type> handle):handle_(handle){}
  TemplateTask(TemplateTask&& other) noexcept:handle_(std::exchange(other.handle_,{})){}
  TemplateTask(const TemplateTask&)=delete;
  ~TemplateTask(){if(handle_)handle_.destroy();}
  bool Done()const{return handle_.done();}
  void Resume(){handle_.resume();if(handle_.promise().error)std::rethrow_exception(handle_.promise().error);}
  std::optional<ExecutionProgram> TakeResult(){return std::move(handle_.promise().result);}
private:
  std::coroutine_handle<promise_type> handle_;
};
struct ProbeValue {
  DynamicKey key;
  Scalar alternative;
};
struct WaitIdentity {
  std::string kind;
  Scalar address, target;
  bool operator==(const WaitIdentity &) const = default;
};
struct WaitProbe {
  std::size_t trace_index;
  WaitIdentity identity;
};
// Whether a vote.sync mask is an integer naming every lane of the warp.
bool FullWarpMask(const Scalar &value) {
  const auto *mask = std::get_if<Integer>(&value);
  return mask && (*mask & Integer(0xffffffffu)) == Integer(0xffffffffu);
}
class Executor : public std::enable_shared_from_this<Executor> {
public:
  Executor(const Module &m, const Function &f, const FrontendConfig &c,
           std::function<void()> poll, SpecializationPurpose purpose)
      : m_(m), f_(f), c_(c), poll_(std::move(poll)), purpose_(purpose) {
    if (purpose_ == SpecializationPurpose::BranchScopeCapture ||
        purpose_ == SpecializationPurpose::VerificationWithBranchScope) {
      const auto choices = c_.raw.value("user_branch_decisions", Json::array());
      if (!choices.is_array() || choices.size() > 4096)
        throw std::length_error("branch-scope user-choice limit");
      for (std::size_t index = 0; index < choices.size(); ++index) {
        const auto& choice = choices[index];
        const auto line = choice.at("line").get<std::uint32_t>();
        const auto taken = choice.contains("resolved_taken")
            ? choice.at("resolved_taken").get<bool>() : choice.at("taken").get<bool>();
        if (!capture_decisions_.emplace(line, std::pair{index, taken}).second)
          throw std::invalid_argument("ambiguous branch-scope user choice");
      }
    }
    Prepare();
  }
  ExecutionProgram Run();
  std::optional<ExecutionProgram> RunTemplates();
  TemplateTask RunTemplateTask(SharedExecution*);
  static std::optional<std::vector<ExecutionProgram>> RunTemplateCluster(const Module &,const Function &,
      const std::vector<FrontendConfig>&,const std::function<void()>&,SpecializationPurpose);
  static std::vector<ExecutionProgram> RunCluster(const Module &, const Function &,
      const std::vector<FrontendConfig> &, const std::function<void()> &, SpecializationPurpose);

private:
  const Module &m_;
  const Function &f_;
  const FrontendConfig c_;
  std::function<void()> poll_;
  SpecializationPurpose purpose_;
  std::map<std::uint32_t, std::pair<std::size_t, bool>> capture_decisions_;
  std::optional<BranchScopeDecision> CaptureDecision(std::uint32_t line,
      bool taken, std::uint64_t visit) const {
    const auto found = capture_decisions_.find(line);
    if (found == capture_decisions_.end() || found->second.second != taken) return {};
    return BranchScopeDecision{taken, visit, found->second.first};
  }
  std::set<std::uint32_t> label_entries_;
  // Two-arm merge, shared by the template and per-lane executors: a forward one-armed diamond
  // whose straight-line fall-through arm holds only listed compute opcodes, private reads and
  // tcgen05 ld/st/wait (no synchronization) and whose written registers are referenced nowhere
  // outside the arm, so the arm's registers are dead after the join (consumers of the lane
  // traces never see an arm value after the join). `uniform` is the static proof of a
  // warp-uniform predicate: it is computed in the branch's basic block from a full-warp
  // vote.sync and constants only, so every lane of the warp reaches the branch with the same
  // value. Lowering checks that no tcgen05 ordering is pending at the branch.
  struct MergedArm {
    std::vector<std::uint32_t> writes;
    bool aligned = false, uniform = false;
  };
  mutable std::map<std::uint32_t, std::optional<MergedArm>> merged_arms_;
  mutable std::vector<std::pair<std::uint32_t, std::uint32_t>> reference_ranges_;
  const MergedArm *MergedArmAt(std::uint32_t pc) const;
  bool StaticWarpUniform(std::uint32_t use, std::uint32_t reg, unsigned depth) const;
  mutable ThreadRecordPool record_pool_;
  std::vector<std::string> names_;
  std::unordered_map<std::string, std::uint32_t> ids_;
  std::vector<Decoded> decoded_;
  std::map<std::string, std::uint32_t> labels_;
  std::map<std::tuple<std::string, std::uint64_t, std::uint32_t>,
           const PointerParameter *>
      pointers_;
  std::map<std::string, ParameterRegion> constant_inputs_;
  // Pointer tables (input pointer_arrays): table resource -> slot byte offset -> entry
  // (an AllocationPointer, or Integer 0 for a null entry).
  std::map<std::string, std::map<std::uint64_t, Scalar>> pointer_tables_;
  // tensor_properties: object resource -> element bound; loads of whole elements are
  // BoundedInteger. property_sites_ records the instructions those bounds decided.
  struct TensorProperty {
    std::string object; unsigned bits; bool sign; Integer lower, upper;
    // The element type: s or u, then the width in bits.
    std::string Type() const { return std::string(sign ? "s" : "u") + std::to_string(bits); }
  };
  std::map<std::string, TensorProperty> tensor_properties_;
  mutable std::map<std::string, std::set<std::uint32_t>> property_sites_;
  // An element whose bounds coincide is that value: appends it to `values` and records
  // `instruction` as decided by the bound.
  bool ExactElement(const TensorProperty &property, std::uint32_t instruction, std::vector<Scalar> &values) const {
    if (property.lower != property.upper) return false;
    values.push_back(property.lower);
    property_sites_[property.object].insert(instruction);
    return true;
  }
  void NoteBoundedDecision(std::uint32_t instruction, std::initializer_list<const Scalar *> operands) const {
    for (const auto &object : BoundedSources(operands)) property_sites_[object].insert(instruction);
  }
  std::uint32_t Intern(std::string name) {
    auto [it, inserted] = ids_.emplace(name, names_.size());
    if (inserted)
      names_.push_back(std::move(name));
    return it->second;
  }
  void Prepare();
  struct ShuffleSelection { std::uint32_t lane; bool valid; };
  std::optional<ShuffleSelection> SelectShuffle(const Decoded&, const State&,
                                               std::uint32_t warp, std::uint32_t lane) const;
  std::optional<std::pair<std::string, std::uint32_t>>
  WaitLoop(const Instruction &) const;
  enum class ControlLoop { None, Computation, Polling, Unknown };
  struct ControlBlock {
    std::uint32_t begin{}, end{};
    std::vector<std::uint32_t> successors, predecessors, loops;
  };
  struct NaturalLoop {
    std::uint32_t header{};
    std::set<std::uint32_t> blocks, latches;
    ControlLoop kind = ControlLoop::Unknown;
  };
  struct ControlChoice {
    ControlLoop kind = ControlLoop::Unknown;
    std::uint32_t loop = kMissing;
    bool taken_stays = false, post_test = false;
  };
  mutable bool control_ready_ = false, control_irreducible_ = false;
  mutable std::vector<ControlBlock> control_blocks_;
  mutable std::vector<std::uint32_t> control_block_for_pc_, control_postdom_;
  mutable std::vector<NaturalLoop> control_loops_;
  mutable std::map<std::uint32_t, ControlChoice> control_choices_;
  void PrepareControl() const;
  ControlChoice ClassifyControl(std::uint32_t pc) const;

  std::optional<std::pair<std::uint32_t, Integer>>
  Assertion(const Instruction &, std::uint32_t) const;
  State Initial(const WarpContract &, std::uint32_t) const;
  Scalar Value(const Operand &o, const State &s) const {
    return o.reg == kMissing ? o.constant
           : Unknown(s.registers[o.reg]) && !s.origins[o.reg] &&
                   !Unknown(o.constant)
               ? o.constant
               : s.registers[o.reg];
  }
  Scalar ValueText(std::string_view text, const State &s) const;
  void Set(const Decoded &, State &, std::vector<Scalar>) const;
  void Invalidate(const Decoded &d, State &s) const {
    for (auto id : d.destinations)
      s.registers[id] = Scalar{};
  }
  void Origins(const Decoded &, State &, std::uint64_t) const;
  ExecutedInstruction Record(const Decoded &, const State &, std::uint64_t,
                             std::optional<bool>) const;
  void Data(const Decoded &, State &, std::uint32_t, std::uint32_t,
            std::uint64_t, const Json &, WarpContext &,
            bool probe = false) const;
  LaneTask Lane(const WarpContract &, std::uint32_t, WarpContext &) const;
  bool SharedData(const Decoded &, State &, std::uint32_t, std::uint32_t,
      std::uint64_t, const Json &, WarpContext &, const LaneTrace &, bool wait = false) const;
  [[noreturn]] void UnresolvedControl(const Instruction &, const State &,
                                     const LaneTrace &, std::uint32_t) const;
  std::optional<WaitIdentity> Identity(const Decoded &, const State &) const;
  std::optional<std::pair<std::uint32_t, std::set<std::uint32_t>>>
  Fallback(const Decoded &, const State &, std::uint32_t, const WaitIdentity &,
           std::uint32_t, std::uint32_t, WarpContext &) const;
  [[noreturn]] void Error(const Instruction &i, std::uint32_t warp,
                          std::uint32_t lane, std::string message, std::string code = "ptx_execution_not_supported") const {
    throw ExecutionError(std::move(code), i.line, "warp " + std::to_string(warp) + " lane " +
                             std::to_string(lane) + ", PTX line " +
                             std::to_string(i.line) + " (" + i.opcode +
                             "): " + message);
  }
};
bool IsWait(const Instruction &i) {
  return i.semantics.category == "mbarrier" &&
         In(i.semantics.operation, {"try_wait_parity", "try_wait_state",
                                    "test_wait_parity", "test_wait_state"});
}
void Executor::UnresolvedControl(const Instruction &instruction, const State &state,
                                  const LaneTrace &trace, std::uint32_t reg) const {
  std::map<DynamicKey,const ExecutedInstruction *> definitions;
  for (const auto &e : detail::ExecutionStorage::Records(trace.instructions)) definitions[e.Key()] = &e;
  std::set<std::pair<DynamicKey, std::uint32_t>> visited;
  std::set<std::tuple<std::string,std::uint64_t,unsigned>> missing;
  Json required = Json::array();
  bool shared_control = false;
  std::function<void(DynamicKey, std::uint32_t)> visit = [&](DynamicKey key, std::uint32_t result) {
    if (!visited.emplace(key, result).second || !definitions.contains(key)) return;
    const auto &e = *definitions.at(key);
    const auto &d = decoded_[e.instruction_id()];
    const auto &i = *d.instruction;
    shared_control |= i.opcode == "ld" &&
        (Has(i,"shared") || Has(i,"shared::cta") || Has(i,"shared::cluster"));
    if (i.opcode == "ld" && Has(i,"param") && i.memory_width && !d.destinations.empty() &&
        d.operands.size() == 2 && d.operands[1].address) {
      auto [name,offset] = *d.operands[1].address;
      if (auto id = ids_.find(name); id != ids_.end())
        for (const auto &[r,v] : detail::ExecutionStorage::Bindings(e)) if (r == id->second)
          if (auto p = std::get_if<ParameterPointer>(&v)) { name = p->region; offset += p->offset; }
      auto region = c_.parameter_regions.find(name);
      const auto width = *i.memory_width / d.destinations.size();
      if (region != c_.parameter_regions.end())
        for (std::size_t j = 0; j < d.destinations.size(); ++j) {
          if (d.destinations[j] != result) continue;
          const Integer begin = offset + j * width;
          if (begin < 0 || begin > INT64_MAX || begin + width > region->second.size_bytes) continue;
          const auto off = begin.convert_to<std::uint64_t>();
          if (!pointers_.contains({name,off,width}) && !region->second.ReadInteger(off,width,false) &&
              missing.emplace(name,off,width).second)
            required.push_back({{"parameter",name}, {"offset",off}, {"bytes",width}, {"ptx_line",i.line}});
        }
    }
    for (const auto &[operand,origin] : e.register_origins()) visit(origin, operand);
  };
  if (reg < state.origins.size() && state.origins[reg]) visit(*state.origins[reg], reg);
  const std::string code = required.empty() ? "branch_not_static" : "missing_parameter_value";
  ExecutionError error(code, instruction.line, "control predicate is not statically determined", std::move(required));
  error.shared_control = shared_control;
  throw error;
}
std::optional<std::pair<std::uint32_t, Integer>>
Executor::Assertion(const Instruction &branch, std::uint32_t target) const {
  if (!branch.predicate || branch.predicate_negated || branch.id == 0 ||
      target <= branch.id + 1 || target > decoded_.size()) return {};
  for (auto k = branch.id + 1; k < target; ++k)
    if (decoded_[k].instruction->predicate) return {};
  const auto &terminal = *decoded_[target - 1].instruction;
  if (terminal.control == "trap") {
    if (target != branch.id + 2) return {};
  } else if (terminal.control == "call" && !terminal.operands.empty() &&
             TrimView(m_.view(terminal.operands[0])) == "__assertfail") {
    for (auto k = branch.id + 1; k + 1 < target; ++k) {
      const auto &i = *decoded_[k].instruction;
      if (!(In(i.opcode, {"mov", "cvta"}) || (i.opcode == "st" && Has(i, "param")))) return {};
    }
  } else return {};
  const auto &compare = decoded_[branch.id - 1];
  const auto &i = *compare.instruction;
  if (i.opcode != "setp" || !Has(i, "eq") || i.predicate ||
      compare.operands.size() != 3 || compare.operands[0].text != *branch.predicate) return {};
  for (auto [r, literal] : {std::pair{1u,2u}, std::pair{2u,1u}})
    if (compare.operands[r].reg != kMissing)
      if (auto n = ParseInteger(compare.operands[literal].text)) return {{compare.operands[r].reg, *n}};
  return {};
}
void Executor::Prepare() {
  const auto local_declarations = LocalMemoryDeclarations(m_, f_);
  ProfileScope prepare("frontend.execute.prepare");
  for (const auto &entry : c_.fixed_input.value("global_memory_inputs", Json::array())) {
    const auto name = entry.at("resource").get<std::string>();
    const auto bits = entry.at("element_bits").get<unsigned>();
    if (bits != 8 && bits != 16 && bits != 32 && bits != 64)
      throw std::invalid_argument("constant input has unsupported element width");
    ParameterRegion region;
    region.name = name;
    region.source = {{"kind", "typed_fields"}, {"fields", Json::array()}};
    auto offset = entry.value("byte_offset", std::uint64_t(0));
    for (const auto &value : entry.at("values")) {
      Integer n = JsonInteger(value) & ((Integer(1) << bits) - 1);
      region.offset_values[offset] = {bits / 8, n};
      if (offset > UINT64_MAX - bits / 8) throw std::invalid_argument("constant input overflows address range");
      offset += bits / 8;
    }
    region.size_bytes = offset;
    if (!constant_inputs_.emplace(name, std::move(region)).second)
      throw std::invalid_argument("duplicate constant input");
  }
  for (const auto &table : c_.fixed_input.value("pointer_array_inputs", Json::array()))
    for (const auto &entry : table.at("entries")) {
      auto &slot = pointer_tables_[table.at("resource").get<std::string>()][entry.at("slot_offset").get<std::uint64_t>()];
      if (entry.value("null", false)) slot = Integer(0);
      else
        slot = AllocationPointer{entry.at("resource").get<std::string>(), Integer(entry.at("offset").get<std::uint64_t>()),
                                 entry.contains("extent_bytes") ? std::optional(entry.at("extent_bytes").get<std::uint64_t>())
                                                                : std::nullopt};
    }
  for (const auto &entry : c_.fixed_input.value("tensor_property_inputs", Json::array()))
    tensor_properties_[entry.at("resource").get<std::string>()] = {entry.at("object").get<std::string>(),
        entry.at("element_bits").get<unsigned>(), entry.at("signed").get<bool>(),
        JsonInteger(entry.at("lower_bound")), JsonInteger(entry.at("upper_bound"))};
  if (!f_.has_body || f_.instructions.empty())
    throw std::runtime_error("PTX entry has no executable body");
  ProfileScope registers("frontend.execute.prepare.registers");
  for (const auto &i : f_.instructions) {
    if (i.id >= f_.instructions.size())
      throw std::runtime_error("non-dense PTX instruction ids");
    for (const auto &r : i.registers)
      Intern(r);
    for (const auto &r : i.destinations)
      Intern(r);
    if (i.predicate)
      Intern(*i.predicate);
    for (auto range : i.operands) {
      auto text = TrimView(m_.view(range));
      for (auto r : RegistersIn(text))
        Intern(r);
      if (text.starts_with('[') && text.ends_with(']'))
        if (auto split = SplitAddress(text.substr(1, text.size() - 2)))
          if (!ParseInteger(split->first))
            Intern(split->first);
    }
  }
  for (const auto &[r, v] : c_.special_registers)
    Intern(r);
  for (const auto &w : c_.warps)
    for (const auto &[r, v] : w.initial_registers)
      Intern(r);
  for (auto r : {"%laneid", "%warpid", "%nwarpid", "%tid.x", "%tid.y", "%tid.z",
                 "%ntid.x", "%ntid.y", "%ntid.z", "%ctaid.x", "%ctaid.y",
                 "%ctaid.z", "%cluster_ctarank", "%cluster_nctaid.x",
                 "%cluster_nctaid.y", "%cluster_nctaid.z"})
    Intern(r);
  std::map<std::string, std::set<std::uint32_t>> raw_labels;
  for (const auto &label : f_.labels) {
    label_entries_.insert(label.instruction);
    raw_labels[label.name].insert(label.instruction);
    std::string qualified;
    for (auto id : label.scope) {
      if (!qualified.empty())
        qualified += '/';
      qualified += std::to_string(id);
    }
    if (!qualified.empty())
      qualified += "::";
    qualified += label.name;
    labels_[qualified] = label.instruction;
  }
  for (const auto &[name, targets] : raw_labels)
    if (targets.size() == 1)
      labels_[name] = *targets.begin();
  registers.nodes = names_.size();
  registers.edges = labels_.size();
  registers.Stop();
  ProfileScope decode("frontend.execute.prepare.decode");
  decoded_.resize(f_.instructions.size());
  for (const auto &i : f_.instructions) {
    auto &d = decoded_[i.id];
    d.instruction = &i;
    if (i.predicate)
      d.predicate = ids_.at(*i.predicate);
    for (const auto &r : i.destinations)
      d.destinations.push_back(ids_.at(r));
    for (auto range : i.operands) {
      Operand o;
      o.text = std::string(TrimView(m_.view(range)));
      for (const auto &r : RegistersIn(o.text))
        o.registers.push_back(ids_.at(r));
      if (o.text.starts_with('[') && o.text.ends_with(']'))
        o.address =
            SplitAddress(std::string_view(o.text).substr(1, o.text.size() - 2));
      if (o.address) {
        auto found = ids_.find(o.address->first);
        if (found != ids_.end() &&
            std::find(o.registers.begin(), o.registers.end(), found->second) ==
                o.registers.end())
          o.registers.push_back(found->second);
      }
      if (auto n = ParseInteger(o.text))
        o.constant = *n;
      else if (o.text.size() >= 3 && o.text[0] == '0' &&
               In(o.text.substr(1, 1), {"f", "F", "d", "D"}) &&
               o.text.size() == (std::tolower(o.text[1]) == 'f' ? 10u : 18u)) {
        auto n = ParseInteger("0x" + o.text.substr(2));
        if (n)
          o.constant = *n;
      } else if (o.text == "true" || o.text == "p")
        o.constant = true;
      else if (o.text == "false" || o.text == "!p")
        o.constant = false;
      else if (c_.parameter_regions.contains(o.text))
        o.constant = ParameterPointer{o.text, 0};
      else {
        if (auto found = ids_.find(o.text); found != ids_.end())
          o.reg = found->second;
        if (auto found = c_.parameters.find(o.text);
            found != c_.parameters.end())
          o.constant = found->second;
      }
      const LocalMemoryDeclaration* local = nullptr;
      for (const auto& declaration : local_declarations)
        if (declaration.name == o.text && declaration.scope.size() <= i.scope.size() &&
            std::equal(declaration.scope.begin(), declaration.scope.end(), i.scope.begin()) &&
            (!local || declaration.scope.size() > local->scope.size())) local = &declaration;
      if (local) o.constant = AllocationPointer{local->resource, 0, local->bytes};
      for (auto id : o.registers)
        if (std::find(d.uses.begin(), d.uses.end(), id) == d.uses.end())
          d.uses.push_back(id);
      d.operands.push_back(std::move(o));
    }
    if (d.predicate != kMissing &&
        std::find(d.uses.begin(), d.uses.end(), d.predicate) == d.uses.end())
      d.uses.push_back(d.predicate);
    if (!d.operands.empty() &&
        d.operands[0].text.find('|') != std::string::npos) {
      auto text = d.operands[0].text;
      std::size_t begin = 0;
      while (true) {
        auto end = text.find('|', begin);
        auto regs = RegistersIn(std::string_view(text).substr(
            begin,
            end == std::string::npos ? text.size() - begin : end - begin));
        d.destination_slots.push_back(regs.empty() ? kMissing
                                                   : ids_.at(regs[0]));
        if (end == std::string::npos)
          break;
        begin = end + 1;
      }
    } else
      d.destination_slots = d.destinations;
    d.wait_loop = WaitLoop(i);
    d.replay_wait_result = d.wait_loop && i.id+1<f_.instructions.size() &&
        f_.instructions[i.id+1].opcode=="selp";
    decode.edges += d.operands.size();
  }
  decode.nodes = decoded_.size();
  decode.Stop();
  ProfileScope bindings("frontend.execute.prepare.pointer_bindings");
  std::map<std::string, std::vector<std::pair<std::uint64_t, std::uint64_t>>>
      ranges;
  for (const auto &p : c_.pointer_parameters) {
    const Parameter *parameter = nullptr;
    if (p.parameter_index) {
      if (*p.parameter_index >= f_.parameters.size())
        throw std::runtime_error("pointer parameter index outside PTX ABI");
      parameter = &f_.parameters[*p.parameter_index];
    } else
      for (const auto &item : f_.parameters)
        if (item.name == p.parameter) {
          parameter = &item;
          break;
        }
    if (!parameter || !parameter->name)
      throw std::runtime_error("pointer parameter missing from PTX ABI");
    std::optional<std::uint64_t> size;
    if (parameter->array) {
      auto text = *parameter->array;
      if (text.size() > 2 && text.front() == '[' && text.back() == ']') {
        auto n =
            ParseInteger(std::string_view(text).substr(1, text.size() - 2));
        if (n && *n >= 0 && *n <= UINT64_MAX)
          size = n->convert_to<std::uint64_t>();
      }
    } else
      for (const auto &type : parameter->types)
        if (type.size() > 2 && type.front() == '.' &&
            In(type.substr(1, 1), {"b", "u", "s", "f"})) {
          auto n = ParseInteger(type.substr(2));
          if (n && *n > 0 && *n % 8 == 0)
            size = (*n / 8).convert_to<std::uint64_t>();
        }
    if (!size || p.byte_offset > *size || p.width_bytes > *size - p.byte_offset)
      throw std::runtime_error(
          "pointer field exceeds PTX ABI parameter storage");
    auto end = p.byte_offset + p.width_bytes;
    for (auto [a, b] : ranges[*parameter->name])
      if (p.byte_offset < b && a < end)
        throw std::runtime_error("overlapping resolved pointer fields");
    ranges[*parameter->name].push_back({p.byte_offset, end});
    pointers_.emplace(
        std::tuple{*parameter->name, p.byte_offset, p.width_bytes}, &p);
  }
  bindings.nodes = pointers_.size();
  prepare.nodes = decoded_.size();
  prepare.edges = names_.size();
}
// Build a basic-block CFG once per executor. Dominators use immediate-parent
// chains, rather than a quadratic matrix over PTX instructions.
void Executor::PrepareControl() const {
  if (control_ready_)
    return;
  control_ready_ = true;
  const auto size = static_cast<std::uint32_t>(decoded_.size());
  std::set<std::uint32_t> leaders{0, size};
  for (std::uint32_t pc = 0; pc < size; ++pc) {
    const auto &i = *decoded_[pc].instruction;
    if (i.branch_target)
      leaders.insert(*i.branch_target);
    if (i.control != "none" && i.control != "call")
      leaders.insert(pc + 1);
  }
  control_block_for_pc_.resize(size);
  for (auto it = leaders.begin(); std::next(it) != leaders.end(); ++it) {
    ControlBlock block;
    block.begin = *it;
    block.end = *std::next(it);
    if (block.end > size) {
      control_irreducible_ = true;
      return;
    }
    for (auto pc = block.begin; pc < block.end; ++pc)
      control_block_for_pc_[pc] = control_blocks_.size();
    control_blocks_.push_back(std::move(block));
  }
  const auto exit = static_cast<std::uint32_t>(control_blocks_.size());
  control_blocks_.push_back(ControlBlock{size, size, {}, {}, {}});
  for (std::uint32_t b = 0; b < exit; ++b) {
    auto &block = control_blocks_[b];
    const auto &i = *decoded_[block.end - 1].instruction;
    auto edge = [&](std::uint32_t to) {
      auto &out = block.successors;
      if (std::find(out.begin(), out.end(), to) == out.end()) {
        out.push_back(to);
        control_blocks_[to].predecessors.push_back(b);
      }
    };
    if (i.control == "direct_branch" && i.branch_target &&
        *i.branch_target < size) {
      edge(control_block_for_pc_[*i.branch_target]);
      if (i.predicate)
        edge(b + 1);
    } else if (In(i.control, {"return", "exit", "trap"})) {
      edge(exit);
      if (i.predicate)
        edge(b + 1);
    } else {
      edge(b + 1);
      if (i.control == "indirect_branch")
        control_irreducible_ = true;
    }
  }
  auto dominators = [&](std::uint32_t root, bool reverse) {
    const auto count = control_blocks_.size();
    auto outgoing = [&](std::uint32_t b) -> const std::vector<std::uint32_t> & {
      return reverse ? control_blocks_[b].predecessors
                     : control_blocks_[b].successors;
    };
    std::vector<bool> seen(count);
    std::vector<std::pair<std::uint32_t, std::size_t>> stack{{root, 0}};
    std::vector<std::uint32_t> order;
    seen[root] = true;
    while (!stack.empty()) {
      auto &[b, next] = stack.back();
      const auto &edges = outgoing(b);
      if (next == edges.size()) {
        order.push_back(b);
        stack.pop_back();
      } else {
        auto n = edges[next++];
        if (!seen[n]) {
          seen[n] = true;
          stack.emplace_back(n, 0);
        }
      }
    }
    std::reverse(order.begin(), order.end());
    std::vector<std::uint32_t> rank(count, kMissing), parent(count, kMissing);
    for (std::uint32_t n = 0; n < order.size(); ++n)
      rank[order[n]] = n;
    parent[root] = root;
    bool changed = true;
    while (changed) {
      changed = false;
      for (auto b : order) {
        if (b == root)
          continue;
        auto value = kMissing;
        const auto &incoming = reverse ? control_blocks_[b].successors
                                       : control_blocks_[b].predecessors;
        for (auto p : incoming)
          if (parent[p] != kMissing) {
            if (value == kMissing)
              value = p;
            else {
              auto x = p;
              while (x != value) {
                while (rank[x] > rank[value])
                  x = parent[x];
                while (rank[value] > rank[x])
                  value = parent[value];
              }
            }
          }
        if (value != parent[b]) {
          parent[b] = value;
          changed = true;
        }
      }
    }
    return parent;
  };
  const auto dom = dominators(0, false);
  control_postdom_ = dominators(exit, true);
  auto dominates = [&](std::uint32_t h, std::uint32_t b) {
    while (b != kMissing && b != h && dom[b] != b)
      b = dom[b];
    return b == h;
  };
  // Keep strictly nested natural loops even when the compiler gives them the
  // same header (for (...) { while (wait) {} ... }). Combining by header would
  // incorrectly turn the inner polling latch into an outer computation latch.
  for (std::uint32_t b = 0; b < exit; ++b)
    for (auto h : control_blocks_[b].successors) {
      if (dom[b] == kMissing || !dominates(h, b))
        continue;
      NaturalLoop loop;
      loop.header = h;
      loop.latches.insert(b);
      loop.blocks.insert(h);
      std::vector<std::uint32_t> pending{b};
      while (!pending.empty()) {
        auto n = pending.back();
        pending.pop_back();
        if (!loop.blocks.insert(n).second)
          continue;
        for (auto p : control_blocks_[n].predecessors)
          if (dom[p] != kMissing)
            pending.push_back(p);
      }
      control_loops_.push_back(std::move(loop));
    }
  bool merged = true;
  while (merged) {
    merged = false;
    for (std::size_t a = 0; a < control_loops_.size() && !merged; ++a)
      for (std::size_t b = a + 1; b < control_loops_.size(); ++b) {
        auto &left = control_loops_[a];
        auto &right = control_loops_[b];
        if (left.header != right.header)
          continue;
        const bool l_contains =
            std::includes(left.blocks.begin(), left.blocks.end(),
                          right.blocks.begin(), right.blocks.end());
        const bool r_contains =
            std::includes(right.blocks.begin(), right.blocks.end(),
                          left.blocks.begin(), left.blocks.end());
        if (l_contains != r_contains)
          continue; // Strict nesting.
        left.blocks.insert(right.blocks.begin(), right.blocks.end());
        left.latches.insert(right.latches.begin(), right.latches.end());
        control_loops_.erase(control_loops_.begin() + b);
        merged = true;
        break;
      }
  }
  // Every DFS backedge in a reducible graph must target a dominator.
  std::vector<unsigned char> color(control_blocks_.size());
  std::vector<std::pair<std::uint32_t, std::size_t>> stack{{0, 0}};
  color[0] = 1;
  while (!stack.empty()) {
    auto &[b, next] = stack.back();
    const auto &edges = control_blocks_[b].successors;
    if (next == edges.size()) {
      color[b] = 2;
      stack.pop_back();
      continue;
    }
    auto n = edges[next++];
    if (color[n] == 1 && !dominates(n, b))
      control_irreducible_ = true;
    if (!color[n]) {
      color[n] = 1;
      stack.emplace_back(n, 0);
    }
  }
  for (std::uint32_t id = 0; id < control_loops_.size(); ++id)
    for (auto b : control_loops_[id].blocks)
      control_blocks_[b].loops.push_back(id);
  for (auto &block : control_blocks_)
    std::sort(block.loops.begin(), block.loops.end(), [&](auto a, auto b) {
      return control_loops_[a].blocks.size() > control_loops_[b].blocks.size();
    });
  for (auto &loop : control_loops_) {
    std::map<std::uint32_t, std::vector<std::uint32_t>> definitions;
    std::vector<std::uint32_t> tests;
    bool retry_only = true;
    for (auto b : loop.blocks) {
      const auto &block = control_blocks_[b];
      for (auto pc = block.begin; pc < block.end; ++pc) {
        const auto& i = *decoded_[pc].instruction;
        // A fixed-address read is also used to receive the next scheduler
        // task. It cannot prove that an effectful outer loop is a wait.
        // Keep such loops Unknown, never promote them to bounded computation.
        if (!(i.semantics.category == "compute" || i.control == "direct_branch" ||
              i.opcode == "nanosleep" || IsWait(i) ||
              (i.semantics.category == "memory" && i.semantics.memory_mode == "read")))
          retry_only = false;
        for (auto reg : decoded_[pc].destinations)
          definitions[reg].push_back(pc);
      }
      for (auto out : block.successors)
        if (!loop.blocks.contains(out)) {
          auto pred = decoded_[block.end - 1].predicate;
          if (pred != kMissing)
            tests.push_back(pred);
        }
    }
    bool induction = false, polling = false, unknown = false;
    std::set<std::uint32_t> visited;
    while (!tests.empty()) {
      auto reg = tests.back();
      tests.pop_back();
      if (reg == kMissing || !visited.insert(reg).second)
        continue;
      const auto found = definitions.find(reg);
      if (found == definitions.end())
        continue;
      for (auto pc : found->second) {
        const auto &d = decoded_[pc];
        const auto &i = *d.instruction;
        if (i.semantics.category != "compute") {
          bool invariant = true;
          for (const auto &operand : d.operands)
            if (operand.address)
              for (auto address_reg : operand.registers)
                if (definitions.contains(address_reg))
                  invariant = false;
          if (IsWait(i) || (i.semantics.category == "memory" &&
                            i.semantics.memory_mode == "read" && invariant))
            polling = true;
          else
            unknown = true;
          continue;
        }
        if (!i.predicate && found->second.size() == 1 &&
            (i.opcode == "add" || i.opcode == "sub") &&
            d.operands.size() == 3 && d.operands[1].reg == reg) {
          auto step = ParseInteger(d.operands[2].text);
          if (step && *step != 0)
            induction = true;
        }
        for (std::size_t n = 1; n < d.operands.size(); ++n)
          for (auto input : d.operands[n].registers)
            tests.push_back(input);
        if (d.predicate != kMissing)
          tests.push_back(d.predicate);
      }
    }
    loop.kind = polling     ? (retry_only ? ControlLoop::Polling : ControlLoop::Unknown)
                : unknown   ? ControlLoop::Unknown
                : induction ? ControlLoop::Computation
                            : ControlLoop::Unknown;
  }
}
Executor::ControlChoice Executor::ClassifyControl(std::uint32_t pc) const {
  PrepareControl();
  if (auto it = control_choices_.find(pc); it != control_choices_.end())
    return it->second;
  auto classify = [&]() -> ControlChoice {
    if (control_irreducible_)
      return {};
    const auto b = control_block_for_pc_[pc];
    const auto &block = control_blocks_[b];
    const auto &i = *decoded_[pc].instruction;
    const bool terminal = In(i.control, {"return", "exit", "trap"});
    if ((!terminal && (!i.branch_target || *i.branch_target >= decoded_.size())) ||
        block.successors.size() != 2 || pc + 1 >= decoded_.size())
      return {};
    auto target = terminal ? static_cast<std::uint32_t>(control_blocks_.size() - 1)
                           : control_block_for_pc_[*i.branch_target];
    auto fall = control_block_for_pc_[pc + 1];
    if (!block.loops.empty()) {
      const auto id = block.loops.back();
      const auto &loop = control_loops_[id];
      bool t = loop.blocks.contains(target), f = loop.blocks.contains(fall);
      if (t != f) {
        if (loop.kind == ControlLoop::Polling)
          return {ControlLoop::Polling};
        if (loop.latches.size() != 1)
          return {};
        const auto stays = t ? target : fall;
        bool post = loop.latches.contains(b) && stays == loop.header;
        // A conditional exit followed by a dedicated unconditional backedge
        // is also a post-test. Do not cross body effects, joins, or continues.
        if (!post && loop.latches.contains(stays)) {
          const auto &latch = control_blocks_[stays];
          if (latch.end == latch.begin + 1 &&
              latch.predecessors.size() == 1 && latch.predecessors.front() == b &&
              latch.successors.size() == 1 && latch.successors.front() == loop.header) {
            const auto &jump = *decoded_[latch.begin].instruction;
            post = jump.opcode == "bra" && !jump.predicate;
          }
        }
        const bool pre = b == loop.header && !post;
        if (!pre && !post)
          return {}; // Unknown break is not a representative if.
        // A pre-test block must not already execute body effects before its
        // condition; otherwise a numeric bound would execute an extra prefix.
        if (pre)
          for (auto k = block.begin; k < pc; ++k) {
            const auto &op = *decoded_[k].instruction;
            if (op.semantics.category != "compute" &&
                !(op.semantics.category == "memory" &&
                  op.semantics.memory_mode == "read"))
              return {};
          }
        return {loop.kind, id, t, post};
      }
      if (target == loop.header || fall == loop.header)
        return {}; // Unknown continue.
    }
    // Only ordinary diamonds with a join in the same loop nesting are eligible.
    auto join = control_postdom_[b];
    if (join == kMissing || join == b ||
        control_blocks_[join].loops != block.loops)
      return {};
    for (auto id : block.loops)
      if (control_loops_[id].header == join)
        return {};
    return {ControlLoop::None};
  };
  auto result = classify();
  control_choices_.emplace(pc, result);
  return result;
}
std::optional<std::pair<std::string, std::uint32_t>>
Executor::WaitLoop(const Instruction &i) const {
  if (!IsWait(i) || i.destinations.empty())
    return {};
  // A retry may branch to a short, idempotent address/parity setup before
  // the wait. Every value overwritten by the loop must be defined before
  // being read in this prefix: no carried counter, address, or probe result.
  const auto stable_prefix=[&](std::uint32_t target,std::uint32_t branch) {
    if(target==i.id)return true;
    if(target>i.id || i.id-target>8)return false;
    const auto entry=label_entries_.upper_bound(target);
    if(entry!=label_entries_.end() && *entry<=i.id)return false;
    std::set<std::string> overwritten,defined;
    for(auto pc=target;pc<=branch;++pc)
      for(const auto &name:f_.instructions[pc].destinations)overwritten.insert(name);
    for(auto pc=target;pc<i.id;++pc) {
      const auto &setup=f_.instructions[pc];
      if(setup.predicate || setup.control!="none" || Has(setup,"cc") || setup.destinations.size()!=1 ||
         !In(setup.opcode,{"mov","cvt","cvta","add","sub","shl","shr","and","or","xor"}))return false;
      for(std::size_t k=1;k<setup.operands.size();++k) {
        const auto value=std::string(TrimView(m_.view(setup.operands[k])));
        if(value.find_first_of("!{}[]| ")!=std::string::npos)return false;
        // Special registers sampled inside the loop need their own stability
        // proof. Ordinary declared registers are invariant unless overwritten.
        if(value.starts_with("%clock") || value.starts_with("%globaltimer") ||
           value.starts_with("%pm") || value=="%smid")return false;
        if(overwritten.contains(value) && !defined.contains(value))return false;
      }
      defined.insert(setup.destinations.front());
    }
    return true;
  };
  // NVVM also encodes !ready as selp(flag,1,0,ready); setp.eq(retry,flag,0).
  // Accept only this side-effect-free, same-target loop. Replay the conversion
  // instructions after successful waiting so live-out flag/predicate values
  // and their value origins are retained in both execution paths.
  if(i.id+3<f_.instructions.size()) {
    const auto &select=f_.instructions[i.id+1];
    const auto &compare=f_.instructions[i.id+2];
    const auto &branch=f_.instructions[i.id+3];
    const auto operand=[&](const Instruction &inst,std::size_t index) {
      return TrimView(m_.view(inst.operands.at(index)));
    };
    if(select.opcode=="selp" && Has(select,"b32") && !select.predicate && select.operands.size()==4 &&
       compare.opcode=="setp" && Has(compare,"eq") && Has(compare,"b32") && !compare.predicate && compare.operands.size()==3 &&
       branch.control=="direct_branch" && branch.branch_target && stable_prefix(*branch.branch_target,i.id+3) && branch.predicate && !branch.predicate_negated &&
       select.destinations.size()==1 && compare.destinations.size()==1 &&
       operand(select,1)=="1" && operand(select,2)=="0" && operand(select,3)==i.destinations.front() &&
       operand(compare,1)==select.destinations.front() && operand(compare,2)=="0" &&
       *branch.predicate==compare.destinations.front())
      return {{"pc:"+std::to_string(i.id+4),i.id+4}};
  }
  std::optional<std::pair<std::string, std::uint32_t>> success;
  for (std::uint32_t offset = 1; offset <= 8; ++offset) {
    auto pc = i.id + offset;
    if (pc >= f_.instructions.size())
      return {};
    const auto &candidate = f_.instructions[pc];
    if (candidate.control == "direct_branch") {
      if (!candidate.branch_target)
        return {};
      auto target = *candidate.branch_target;
      if (candidate.predicate && *candidate.predicate == i.destinations[0] &&
          !candidate.predicate_negated && target > pc) {
        if (success)
          return {};
        success = {{candidate.branch_label, target}};
        continue;
      }
      if (stable_prefix(target,pc)) {
        if (candidate.predicate && *candidate.predicate == i.destinations[0] &&
            candidate.predicate_negated)
          return {{"pc:" + std::to_string(pc + 1), pc + 1}};
        if (!candidate.predicate && success)
          return success;
      }
      return {};
    }
    if (candidate.opcode == "nanosleep") {
      if (candidate.predicate && (*candidate.predicate != i.destinations[0] ||
                                  !candidate.predicate_negated))
        return {};
      continue;
    }
    return {};
  }
  return {};
}
State Executor::Initial(const WarpContract &w, std::uint32_t lane) const {
  State s;
  s.registers.resize(names_.size());
  s.origins.resize(names_.size());
  s.occurrences.resize(decoded_.size());
  s.branch_visits.resize(decoded_.size());
  auto set = [&](std::string_view name, Integer value) {
    s.registers[ids_.at(std::string(name))] = std::move(value);
  };
  auto thread = w.warp_id * c_.warp_size + lane;
  set("%laneid", lane);
  set("%warpid", w.warp_id);
  set("%nwarpid", c_.warps.size());
  set("%tid.x", thread % c_.block_dim[0]);
  set("%tid.y", thread / c_.block_dim[0] % c_.block_dim[1]);
  set("%tid.z", thread / (c_.block_dim[0] * c_.block_dim[1]));
  for (std::size_t j = 0; j < 3; ++j) {
    set(std::string("%ntid.") + "xyz"[j], c_.block_dim[j]);
    set(std::string("%ctaid.") + "xyz"[j], 0);
    if (c_.cluster_dim)
      set(std::string("%cluster_nctaid.") + "xyz"[j], (*c_.cluster_dim)[j]);
  }
  set("%cluster_ctarank", 0);
  for (const auto &[name, v] : c_.special_registers)
    set(name, v);
  for (const auto &[name, v] : w.initial_registers)
    set(name, v);
  return s;
}
Scalar Executor::ValueText(std::string_view text, const State &s) const {
  text = TrimView(text);
  if (auto n = ParseInteger(text))
    return *n;
  if (c_.parameter_regions.contains(std::string(text)))
    return ParameterPointer{std::string(text), 0};
  if (auto it = ids_.find(std::string(text));
      it != ids_.end() &&
      (!Unknown(s.registers[it->second]) || s.origins[it->second]))
    return s.registers[it->second];
  if (auto it = c_.parameters.find(std::string(text));
      it != c_.parameters.end())
    return it->second;
  if (text == "true" || text == "p")
    return true;
  if (text == "false" || text == "!p")
    return false;
  return {};
}
void Executor::Set(const Decoded &d, State &s,
                   std::vector<Scalar> values) const {
  for (std::size_t n = 0; n < d.destination_slots.size(); ++n)
    if (d.destination_slots[n] != kMissing)
      s.registers[d.destination_slots[n]] =
          n < values.size() ? std::move(values[n]) : Scalar{};
}
void Executor::Origins(const Decoded &d, State &s,
                       std::uint64_t occurrence) const {
  DynamicKey origin{d.instruction->id, occurrence};
  std::vector<unsigned> widths;
  if (d.instruction->opcode == "cvt")
    for (const auto &modifier : d.instruction->modifiers)
      if (modifier.size() > 1 && In(modifier.substr(0,1), {"b","u","s"}) &&
          std::all_of(modifier.begin()+1,modifier.end(), [](unsigned char c) { return std::isdigit(c); }))
        widths.push_back(std::stoul(modifier.substr(1)));
  const bool narrowing = widths.size() == 2 && widths[0] < widths[1];
  if (!narrowing && In(d.instruction->opcode, {"mov", "cvt"}) && d.operands.size() > 1 &&
      d.operands[1].registers.size() == 1)
    if (auto known = s.origins[d.operands[1].registers[0]])
      origin = *known;
  for (auto id : d.destinations)
    s.origins[id] = origin;
}
ExecutedInstruction Executor::Record(const Decoded &d, const State &s,
                                     std::uint64_t occurrence,
                                     std::optional<bool> predicate) const {
  ExecutedInstruction out;
  auto &metadata = out.mutableMetadata();
  auto &bindings = out.mutableKnownRegisters();
  metadata.instruction_id = d.instruction->id;
  metadata.occurrence = occurrence;
  metadata.predicate_value = predicate;
  bindings.reserve(d.uses.size());
  metadata.register_origins.reserve(d.uses.size());
  for (auto id : d.uses) {
    if (!Unknown(s.registers[id]))
      bindings.emplace_back(id, s.registers[id]);
    if (s.origins[id])
      metadata.register_origins.emplace_back(id, *s.origins[id]);
  }
  return out;
}
std::optional<Executor::ShuffleSelection> Executor::SelectShuffle(
    const Decoded& d,const State& state,std::uint32_t warp,std::uint32_t lane) const {
  const auto& i=*d.instruction;
  auto source=ScalarInteger(Value(d.operands[2],state));
  auto clamp=ScalarInteger(Value(d.operands[3],state));
  auto mask=ScalarInteger(Value(d.operands[4],state));
  if(!source || !clamp || !mask)return {};
  const auto members=(*mask&0xffffffff).convert_to<std::uint32_t>();
  if(!(members&(1u<<lane)))Error(i,warp,lane,"executing lane absent from shuffle membermask");
  const auto b=(*source&31).convert_to<unsigned>();
  const auto c=(*clamp&31).convert_to<unsigned>();
  const auto segment=((*clamp>>8)&31).convert_to<unsigned>();
  const auto inverse=(~segment)&31;
  const auto maximum=(lane&segment)|(c&inverse);
  int from=(lane&segment)|(b&inverse);
  if(i.modifiers[1]=="up")from=int(lane)-int(b);
  if(i.modifiers[1]=="down")from=lane+b;
  if(i.modifiers[1]=="bfly")from=lane^b;
  const bool valid=i.modifiers[1]=="up"?from>=int(maximum):from<=int(maximum);
  if(!valid)from=lane;
  if(!(members&(1u<<from)))Error(i,warp,lane,"selected shuffle source absent from membermask");
  return ShuffleSelection{static_cast<std::uint32_t>(from),valid};
}
void Executor::Data(const Decoded &d, State &s, std::uint32_t warp,
                    std::uint32_t lane, std::uint64_t occurrence,
                    const Json &contract, WarpContext &context,
                    bool probe) const {
  const auto &i = *d.instruction;
  const auto &op = i.opcode;
  auto n = d.operands.size();
  auto val = [&](std::size_t index) -> Scalar {
    return index < n ? Value(d.operands[index], s) : Scalar{};
  };
  if (IsWait(i)) {
    if (contract.contains("register_results"))
      Error(i, warp, lane,
            "mbarrier wait results cannot come from fixed-input "
            "register_results");
    if (probe)
      Set(d, s, {false});
    else
      Invalidate(d, s);
    return;
  }
  if (op == "ld" && Has(i, "param")) {
    if (n < 2)
      Error(i, warp, lane, "malformed ld.param");
    if (!i.memory_width || d.destinations.empty()) {
      Invalidate(d, s);
      return;
    }
    if (*i.memory_width % d.destinations.size())
      Error(i, warp, lane, "parameter load cannot split across destinations");
    auto width = *i.memory_width / d.destinations.size();
    auto address = d.operands[1].text;
    auto view = TrimView(address);
    if (view.starts_with('[') && view.ends_with(']'))
      view = TrimView(view.substr(1, view.size() - 2));
    auto split = SplitAddress(view);
    std::string name = split ? split->first : std::string(view);
    Integer offset = split ? split->second : Integer(0);
    auto base = ValueText(name, s);
    if (auto p = std::get_if<ParameterPointer>(&base)) {
      name = p->region;
      offset += p->offset;
    }
    const bool sign = SignedModifier(i);
    std::vector<Scalar> values;
    for (std::size_t index = 0; index < d.destinations.size(); ++index) {
      auto field = Int64(offset + index * width, "parameter offset");
      if (field >= 0) {
        auto pointer = pointers_.find({name, std::uint64_t(field), width});
        if (pointer != pointers_.end()) {
          const auto &p = *pointer->second;
          values.emplace_back(AllocationPointer{p.resource, 0, p.extent_bytes});
          continue;
        }
      }
      const auto &reg = names_[d.destinations[index]];
      std::optional<DescriptorFragment> descriptor_fragment;
      for (auto it=c_.tma_descriptors.begin();it!=c_.tma_descriptors.end();++it) {
        const auto &map=it.value();
        if (!map.contains("parameter_location")) continue;
        const auto &location=map.at("parameter_location");
        const auto begin=location.at(1).get<std::int64_t>();
        if (location.at(0)==name && field>=begin && Integer(field)+width<=Integer(begin)+128) {
          descriptor_fragment=DescriptorFragment{it.key(),static_cast<unsigned>(field-begin),static_cast<unsigned>(width)};
          break;
        }
      }
      if (descriptor_fragment) { values.emplace_back(*descriptor_fragment); continue; }
      if (contract.contains("register_results") &&
          contract["register_results"].is_object() &&
          contract["register_results"].contains(reg) &&
          (contract["register_results"][reg].is_number_integer() ||
           contract["register_results"][reg].is_boolean())) {
        values.emplace_back(JsonInteger(contract["register_results"][reg]));
        continue;
      }
      auto region = c_.parameter_regions.find(name);
      if (region != c_.parameter_regions.end()) {
        auto v = (field < 0 || std::uint64_t(field) < region->second.size_bytes)
                     ? region->second.ReadInteger(field, width, sign)
                     : std::optional<Integer>{};
        values.push_back(v ? Scalar(*v) : Scalar{});
        continue;
      }
      auto exact = c_.parameters.find(std::string(view));
      values.push_back(d.destinations.size() == 1 &&
                               exact != c_.parameters.end()
                           ? Scalar(exact->second)
                           : Scalar{});
    }
    Set(d, s, std::move(values));
    return;
  }
  // Generic loads qualify too: the pointer must still resolve to a global
  // fixed-input allocation below.
  const bool global_or_generic =
      Has(i, "global") ||
      std::none_of(i.modifiers.begin(), i.modifiers.end(), [](const auto &m) {
        return m.starts_with("shared") || In(m, {"local", "param", "const"});
      });
  if (op == "ld" && global_or_generic && !Floating(i) && n == 2u + Has(i, "L2::cache_hint") &&
      d.operands[1].address && i.memory_width && !d.destination_slots.empty()) {
    const auto &[base, displacement] = *d.operands[1].address;
    const auto pointer_value = ValueText(base, s);
    // A whole 8-byte slot of a pointer table yields its entry; any other access is unknown.
    if (auto pointer = std::get_if<AllocationPointer>(&pointer_value);
        pointer && pointer_tables_.contains(pointer->resource)) {
      const auto &table = pointer_tables_.at(pointer->resource);
      const auto count = d.destination_slots.size();
      std::vector<Scalar> values;
      for (std::size_t index = 0; index < count; ++index) {
        const auto offset = Int64(pointer->offset + displacement + index * 8, "pointer table offset");
        const auto found = offset < 0 ? table.end() : table.find(std::uint64_t(offset));
        values.push_back(*i.memory_width == 8 * count && found != table.end() ? found->second : Scalar{});
      }
      Set(d, s, std::move(values));
      return;
    }
    // A pointer with a non-affine offset reads every candidate: a constant input gives
    // the loaded values over the offset's key; an object with tensor_properties gives its
    // bounds, the element named by the offset (read-only: one offset, one value).
    if (auto bounded = std::get_if<BoundedAllocationPointer>(&pointer_value);
        bounded && (*bounded)->table &&
        (constant_inputs_.contains((*bounded)->resource) || tensor_properties_.contains((*bounded)->resource))) {
      const auto &pointer = *bounded->value;
      const auto &offsets = *pointer.table;
      const auto count = d.destination_slots.size();
      std::vector<Scalar> values;
      if (const auto region = constant_inputs_.find(pointer.resource); region != constant_inputs_.end()) {
        if (*i.memory_width % count != 0) { Invalidate(d, s); return; }
        const auto width = *i.memory_width / count;
        const bool sign = SignedModifier(i);
        for (std::size_t index = 0; index < count; ++index) {
          BoundedTable table{offsets.key, offsets.key_lower, {}};
          for (const auto &candidate : offsets.values) {
            const Integer offset = candidate + displacement + Integer(index * width);
            const auto read = offset < 0 || offset > Integer(region->second.size_bytes) ||
                    Integer(width) > Integer(region->second.size_bytes) - offset
                ? std::optional<Integer>{} : region->second.ReadInteger(offset.convert_to<std::int64_t>(), width, sign);
            if (!read) { table.values.clear(); break; }
            table.values.push_back(*read);
          }
          values.push_back(table.values.empty() ? Scalar{} : TableValue(std::move(table),
              "load:" + pointer.resource + "[" + pointer.offset_identity + "+" +
                  (displacement + Integer(index * width)).convert_to<std::string>() + "]:" + std::to_string(width),
              pointer.sources));
        }
      } else {
        const auto &property = tensor_properties_.at(pointer.resource);
        const auto bytes = property.bits / 8;
        if (*i.memory_width != count * bytes) { Invalidate(d, s); return; }
        for (std::size_t index = 0; index < count; ++index) {
          const Integer shift = displacement + Integer(index * bytes);
          const bool aligned = std::all_of(offsets.values.begin(), offsets.values.end(), [&](const Integer &v) {
            return v + shift >= 0 && (v + shift) % bytes == 0;
          });
          if (!aligned) { values.push_back(Scalar{}); continue; }
          if (ExactElement(property, i.id, values)) continue;
          const auto type = property.Type();
          auto sources = pointer.sources;
          if (std::find(sources.begin(), sources.end(), property.object) == sources.end()) sources.push_back(property.object);
          values.push_back(MakeBounded(property.lower, property.upper,
              "element:" + property.object + "+[" + pointer.offset_identity + "+" + shift.convert_to<std::string>() +
                  "]:" + type, std::move(sources)));
        }
      }
      Set(d, s, std::move(values));
      return;
    }
    if (auto pointer = std::get_if<AllocationPointer>(&pointer_value);
        pointer && constant_inputs_.contains(pointer->resource)) {
      const auto count = d.destination_slots.size();
      if (*i.memory_width % count != 0) { Invalidate(d, s); return; }
      const auto width = *i.memory_width / count;
      const auto &region = constant_inputs_.at(pointer->resource);
      const bool sign = SignedModifier(i);
      std::vector<Scalar> values;
      for (std::size_t index = 0; index < count; ++index) {
        const auto offset = Int64(pointer->offset + displacement + index * width, "constant input offset");
        const auto value = offset < 0 || std::uint64_t(offset) > region.size_bytes ||
                width > region.size_bytes - std::uint64_t(offset)
            ? std::optional<Integer>{} : region.ReadInteger(offset, width, sign);
        values.push_back(value ? Scalar(*value) : Scalar{});
      }
      Set(d, s, std::move(values));
      return;
    }
    // A whole aligned element of an object with tensor_properties is bounded; its
    // identity is the element, which the read-only contract keeps unchanged.
    if (auto pointer = std::get_if<AllocationPointer>(&pointer_value);
        pointer && tensor_properties_.contains(pointer->resource)) {
      const auto &property = tensor_properties_.at(pointer->resource);
      const auto bytes = property.bits / 8;
      const auto count = d.destination_slots.size();
      if (*i.memory_width != count * bytes) { Invalidate(d, s); return; }
      std::vector<Scalar> values;
      for (std::size_t index = 0; index < count; ++index) {
        const Integer offset = pointer->offset + displacement + Integer(index * bytes);
        if (offset < 0 || offset % bytes != 0) { values.push_back(Scalar{}); continue; }
        if (ExactElement(property, i.id, values)) continue;
        const auto type = property.Type();
        values.push_back(MakeBounded(property.lower, property.upper,
            "element:" + property.object + "+" + offset.convert_to<std::string>() + ":" + type,
            {property.object}, AffineForm{0, {{ElementAtom{property.object, type, offset}, Integer(1)}}}));
      }
      Set(d, s, std::move(values));
      return;
    }
  }
  if (op == "clusterlaunchcontrol" && Has(i, "query_cancel") && Has(i, "is_canceled") &&
      c_.fixed_input.contains("cluster_launch_cancellation") &&
      c_.fixed_input["cluster_launch_cancellation"] == false) {
    Set(d, s, {false});
    return;
  }
  if (op == "mov" && n > 1) {
    if (d.destinations.empty())
      return;
    auto width = Width(i);
    auto value = val(1);
    auto integer = ScalarInteger(value);
    if (Has(i, "pred")) {
      Set(d, s, {integer ? Scalar(*integer != 0) : Scalar{}});
      return;
    }
    auto destination = TrimView(d.operands[0].text);
    if (destination.starts_with('{') && destination.ends_with('}')) {
      std::vector<std::string> slots;
      auto text = destination.substr(1, destination.size() - 2);
      std::size_t begin = 0;
      while (true) {
        auto end = text.find(',', begin);
        slots.emplace_back(TrimView(text.substr(begin, end == std::string_view::npos
                                                       ? text.size() - begin
                                                       : end - begin)));
        if (end == std::string_view::npos)
          break;
        begin = end + 1;
      }
      if (slots.size() > 1 && width && *width % slots.size() == 0) {
        std::vector<Scalar> result;
        auto part = *width / slots.size();
        Integer mask = (Integer(1) << part) - 1;
        for (std::size_t j = 0; j < slots.size(); ++j)
          if (slots[j] != "_")
            result.push_back(
                integer ? Scalar(Integer((*integer >> (j * part)) & mask))
                        : Scalar{});
        Set(d, s, std::move(result));
        return;
      }
    }
    if (d.destinations.size() > 1) {
      if (!width || *width % d.destinations.size() || !integer) {
        Invalidate(d, s);
        return;
      }
      auto part = *width / d.destinations.size();
      Integer mask = (Integer(1) << part) - 1;
      std::vector<Scalar> result;
      for (std::size_t j = 0; j < d.destinations.size(); ++j)
        result.emplace_back(Integer((*integer >> (j * part)) & mask));
      Set(d, s, std::move(result));
      return;
    }
    if (d.operands[1].text.starts_with('{') && d.operands[1].text.ends_with('}')) {
      std::vector<std::string> parts;
      auto source = std::string_view(d.operands[1].text);
      source.remove_prefix(1); source.remove_suffix(1);
      for (std::size_t begin = 0;;) {
        auto end = source.find(',', begin);
        parts.emplace_back(TrimView(source.substr(begin, end == source.npos ? source.size() - begin : end - begin)));
        if (end == source.npos) break;
        begin = end + 1;
      }
      auto count = parts.size();
      if (!width || *width % count) {
        Invalidate(d, s);
        return;
      }
      auto part = *width / count;
      Integer packed = 0, mask = (Integer(1) << part) - 1;
      for (std::size_t j = 0; j < count; ++j) {
        auto v = ScalarInteger(ValueText(parts[j], s));
        if (!v) {
          Invalidate(d, s);
          return;
        }
        packed |= (*v & mask) << (j * part);
      }
      Set(d, s, {packed});
      return;
    }
    if (std::holds_alternative<ClusterSharedPointer>(value) &&
        (!width || *width < 32))
      value = Scalar{};
    Set(d, s, {value});
    return;
  }
  if (In(op, {"brev", "bfind", "clz", "popc"}) && n == 2) {
    auto value = ScalarInteger(val(1));
    auto width = Width(i);
    if (!value || !width || (*width != 32 && *width != 64)) {
      Invalidate(d, s); return;
    }
    const Integer mask = (Integer(1) << *width) - 1;
    Integer bits = *value & mask, result = 0;
    if (op == "bfind" && Has(i, "s" + std::to_string(*width)) &&
        (bits & (Integer(1) << (*width - 1))) != 0) bits ^= mask;
    if (op == "brev") {
      for (unsigned bit = 0; bit < *width; ++bit)
        if ((bits & (Integer(1) << bit)) != 0) result |= Integer(1) << (*width - 1 - bit);
    } else if (op == "popc") {
      for (unsigned bit = 0; bit < *width; ++bit)
        if ((bits & (Integer(1) << bit)) != 0) ++result;
    } else {
      int top = int(*width) - 1;
      while (top >= 0 && (bits & (Integer(1) << top)) == 0) --top;
      if (op == "clz") result = int(*width) - 1 - top;
      else if (top < 0) result = UINT32_MAX;
      else result = Has(i, "shiftamt") ? int(*width) - 1 - top : top;
    }
    Set(d, s, {result}); return;
  }
  if (op == "bfe" && n == 4) {
    auto value = ScalarInteger(val(1)), position = ScalarInteger(val(2)), length = ScalarInteger(val(3));
    auto width = Width(i);
    if (!value || !position || !length || !width || (*width != 32 && *width != 64)) {
      Invalidate(d, s);
      return;
    }
    const auto start = (*position & 255).convert_to<unsigned>();
    const auto count = (*length & 255).convert_to<unsigned>();
    const Integer mask = (Integer(1) << *width) - 1;
    Integer bits = *value & mask;
    const bool sign = Has(i, "s" + std::to_string(*width));
    if (sign && (bits & (Integer(1) << (*width - 1))) != 0)
      bits -= Integer(1) << *width;
    Integer result = 0;
    if (count) {
      result = (bits >> start) & ((Integer(1) << count) - 1);
      if (sign && ((result >> (count - 1)) & 1) != 0)
        result -= Integer(1) << count;
    }
    Set(d, s, {Integer(result & mask)});
    return;
  }
  if (In(op, {"cvt", "cvta"}) && n > 1) {
    if (op == "cvt" && Floating(i)) {
      Set(d,s,{ExactIntegerToFloat(val(1),i)});
      return;
    }
    auto value = val(1);
    auto width = Width(i);
    if (std::holds_alternative<ClusterSharedPointer>(value) &&
        (!width || *width < 32))
      value = Scalar{};
    // Integer cvt: read the source at its own width and signedness (sign- or zero-
    // extending), then wrap, or clamp under .sat, to the destination type.
    std::vector<std::string> types;
    for (const auto &m : i.modifiers)
      if (m.size() > 1 && In(m.substr(0, 1), {"b", "u", "s"}) &&
          std::all_of(m.begin() + 1, m.end(), [](unsigned char c) { return std::isdigit(c); }))
        types.push_back(m);
    if (const auto *bounded = std::get_if<BoundedInteger>(&value); bounded && op == "cvt" && types.size() == 2) {
      // Exact when the value fits both the source and the destination type.
      if (!FitsType(*bounded, std::stoul(types[1].substr(1)), types[1][0] == 's') ||
          !FitsType(*bounded, std::stoul(types[0].substr(1)), types[0][0] == 's'))
        value = Scalar{};
    } else if (std::holds_alternative<BoundedAllocationPointer>(value) && op == "cvt" &&
               (types.size() != 2 || types[0].substr(1) != "64" || types[1].substr(1) != "64")) {
      value = Scalar{};
    } else if (auto n = ScalarInteger(value); op == "cvt" && n && types.size() == 2) {
      const auto dest = std::stoul(types[0].substr(1)), source = std::stoul(types[1].substr(1));
      const bool dest_sign = types[0][0] == 's';
      Integer x = Reduce(*n, source, types[1][0] == 's');
      if (Has(i, "sat")) {
        const Integer low = dest_sign ? -(Integer(1) << (dest - 1)) : Integer(0);
        const Integer high = (Integer(1) << (dest - dest_sign)) - 1;
        x = x < low ? low : x > high ? high : x;
      } else
        x = Reduce(x, dest, dest_sign);
      value = x;
    }
    Set(d, s, {value});
    return;
  }
  if (In(op, {"and", "or", "xor"}) && Has(i, "pred") && n > 2) {
    Set(d, s, {PredicateBinary(op, val(1), val(2))});
    return;
  }
  if (op == "mul" && n > 2) {
    if (Floating(i))
      Invalidate(d, s);
    else
      Set(d, s, {IntegerProduct(i, val(1), val(2))});
    return;
  }
  if (In(op, {"add", "sub", "shl", "shr", "and", "or", "xor", "min", "max",
              "rem", "div"}) &&
      n > 2) {
    if (Floating(i)) {
      Invalidate(d, s);
      return;
    }
    auto left = val(1), right = val(2);
    auto pointer = PointerArithmetic(op, left, right, c_);
    Set(d, s, {pointer ? *pointer : IntegerBinary(op, left, right, &i)});
    return;
  }
  if (op == "neg" && n > 1) {
    auto x = ScalarInteger(val(1));
    Set(d, s, {!Floating(i) && x ? Scalar(Integer(-*x)) : Scalar{}});
    return;
  }
  if (op == "not" && n > 1) {
    auto value = val(1);
    auto x = ScalarInteger(value);
    Set(d, s,
        {Has(i, "pred") ? NegatePredicate(value)
         : x            ? Scalar(Integer(~*x))
                        : Scalar{}});
    return;
  }
  if (op == "mad" && n > 3) {
    if (Floating(i)) {
      Invalidate(d, s);
      return;
    }
    auto left = IntegerBinary("mul", val(1), val(2)), right = val(3);
    auto pointer = PointerArithmetic("add", left, right, c_);
    Set(d, s, {pointer ? *pointer : IntegerBinary("add", left, right)});
    return;
  }
  if (op == "setp" && n > 2) {
    std::string relation;
    for (const auto &modifier : i.modifiers)
      if (In(modifier, {"eq", "ne", "lt", "le", "gt", "ge", "equ", "neu", "ltu",
                        "leu", "gtu", "geu", "num", "nan"})) {
        relation = modifier;
        break;
      }
    const auto left = val(1), right = val(2);
    auto comparison = Compare(i, relation, left, right, c_);
    if (!Unknown(comparison)) NoteBoundedDecision(i.id, {&left, &right});
    std::string boolean;
    for (const auto &modifier : i.modifiers)
      if (In(modifier, {"and", "or", "xor"})) {
        boolean = modifier;
        break;
      }
    if (!boolean.empty() && n > 3)
      Set(d, s,
          {PredicateBinary(boolean, comparison, val(3)),
           PredicateBinary(boolean, NegatePredicate(comparison), val(3))});
    else
      Set(d, s, {comparison, NegatePredicate(comparison)});
    return;
  }
  if (op == "selp" && n > 3) {
    auto cond = Truth(val(3));
    Set(d, s, {cond ? val(*cond ? 1 : 2) : Scalar{}});
    return;
  }
  if (op == "mapa" && n > 2) {
    auto address = ScalarInteger(val(1)), rank = ScalarInteger(val(2));
    if (address && *address >= 0 && *address < (Integer(1) << 32) && rank &&
        *rank >= 0 && c_.cluster_dim &&
        *rank < Integer((*c_.cluster_dim)[0]) * (*c_.cluster_dim)[1] *
                    (*c_.cluster_dim)[2])
      Set(d, s,
          {ClusterSharedPointer{*address, rank->convert_to<std::uint32_t>()}});
    else
      Invalidate(d, s);
    return;
  }
  if (op == "elect" && !i.modifiers.empty() && i.modifiers[0] == "sync") {
    if (n != 2) Error(i, warp, lane, "elect.sync requires destination and membermask");
    auto raw = ScalarInteger(val(1));
    if (!raw || *raw < -(Integer(1) << 31) || *raw > UINT32_MAX)
      Error(i, warp, lane, "invalid or unresolved election mask");
    const auto mask = (*raw & 0xffffffff).convert_to<std::uint32_t>();
    if (!mask || !(mask & (1u << lane))) Error(i, warp, lane, "elect issuer is outside membermask");
    auto chosen = c_.elect_lane;
    if (!chosen) {
      if (std::popcount(mask) != 1) Error(i, warp, lane, "election policy not configured");
      chosen = std::countr_zero(mask);
    } else if (*chosen >= 32 || !(mask & (1u << *chosen))) chosen = std::countr_zero(mask);
    Set(d, s, {Integer(*chosen), lane == *chosen});
    return;
  }
  if (op == "shfl") {
    if (i.modifiers.size() >= 2 && i.modifiers[0] == "sync" &&
        In(i.modifiers[1], {"idx", "up", "down", "bfly"}) && n == 5) {
      context.shuffle[{i.id, occurrence, lane}] = val(1);
      const auto selected=SelectShuffle(d,s,warp,lane);
      if (!selected) {
        Invalidate(d, s);
        return;
      }
      const auto from=selected->lane;
      auto it = context.shuffle.find({i.id, occurrence, from});
      if (it == context.shuffle.end())
        throw Deferred{static_cast<std::uint32_t>(from), i.id, occurrence};
      Set(d, s, {it->second, selected->valid});
      return;
    }
    if (contract.value("source_is_warp_uniform", false) && n > 1) {
      auto v = val(1);
      Set(d, s, {v, Unknown(v) ? Scalar{} : Scalar(true)});
      return;
    }
    if (contract.contains("result") &&
        (contract["result"].is_number_integer() ||
         contract["result"].is_boolean())) {
      Set(d, s, {JsonInteger(contract["result"]), true});
      return;
    }
    Invalidate(d, s);
    return;
  }
  if (op == "vote" && i.modifiers.size() == 3 && i.modifiers[0] == "sync" &&
      In(i.modifiers[1], {"ballot", "all", "any", "uni"}) && n == 3) {
    auto mask = ScalarInteger(val(2));
    if (!mask) { Invalidate(d, s); return; }
    const auto members = (*mask & 0xffffffff).convert_to<std::uint32_t>();
    if (!(members & (1u << lane)))
      Error(i, warp, lane, "executing lane absent from vote membermask");
    auto predicate = val(1);
    auto text = TrimView(d.operands[1].text);
    if (text.starts_with('!')) {
      auto value = ScalarInteger(ValueText(text.substr(1), s));
      predicate = value ? Scalar(*value == 0) : Scalar{};
    }
    context.shuffle[{i.id, occurrence, lane}] = predicate;
    context.vote_masks[{i.id, occurrence, lane}] = members;
    std::uint32_t bits = 0;
    bool unknown = false;
    for (unsigned peer = 0; peer < 32; ++peer) {
      if (!(members & (1u << peer))) continue;
      auto it = context.shuffle.find({i.id, occurrence, peer});
      if (it == context.shuffle.end()) throw Deferred{peer, i.id, occurrence};
      if (context.vote_masks.at({i.id, occurrence, peer}) != members)
        Error(i, warp, lane, "vote participants have different membermasks");
      auto value = ScalarInteger(it->second);
      if (!value) unknown = true;
      else if (*value != 0) bits |= 1u << peer;
    }
    if (unknown) { Invalidate(d, s); return; }
    const auto &mode = i.modifiers[1];
    if (mode == "ballot") Set(d, s, {Integer(bits)});
    else Set(d, s, {mode == "all" ? bits == members :
                   mode == "any" ? bits != 0 : bits == 0 || bits == members});
    return;
  }
  if (contract.contains("register_results") &&
      contract["register_results"].is_object()) {
    const auto &results = contract["register_results"];
    for (auto id : d.destinations) {
      auto found = results.find(names_[id]);
      s.registers[id] = found != results.end() && (found->is_number_integer() ||
                                                   found->is_boolean())
                            ? Scalar(JsonInteger(*found))
                            : Scalar{};
    }
    return;
  }
  Invalidate(d, s);
}
std::optional<WaitIdentity> Executor::Identity(const Decoded &d,
                                               const State &s) const {
  if (d.operands.size() < 3)
    return {};
  auto text = TrimView(d.operands[1].text);
  if (text.starts_with('[') && text.ends_with(']'))
    text = TrimView(text.substr(1, text.size() - 2));
  auto split = SplitAddress(text);
  if (!split)
    return {};
  auto address = ValueText(split->first, s);
  if (auto n = ScalarInteger(address))
    address = Integer(*n + split->second);
  else if (auto p = std::get_if<ClusterSharedPointer>(&address))
    p->local_address += split->second;
  else if (split->second != 0)
    return {};
  auto target = Value(d.operands[2], s);
  if (Unknown(address) || Unknown(target))
    return {};
  return WaitIdentity{d.instruction->semantics.operation.ends_with("parity")
                          ? "parity"
                          : "state",
                      std::move(address), std::move(target)};
}
std::optional<std::pair<std::uint32_t, std::set<std::uint32_t>>>
Executor::Fallback(const Decoded &branch, const State &state,
                   std::uint32_t target, const WaitIdentity &identity,
                   std::uint32_t warp, std::uint32_t lane,
                   WarpContext &context) const {
  auto pc = branch.instruction->id;
  if (target <= pc)
    return {};
  State shadow = state;
  std::set<std::uint32_t> changed;
  for (std::uint32_t index = pc + 1; index < std::min(target, pc + 40);
       ++index) {
    if (index >= decoded_.size())
      return {};
    const auto &d = decoded_[index];
    const auto &i = *d.instruction;
    if (i.predicate)
      return {};
    if (d.wait_loop) {
      changed.insert(d.destinations.begin(), d.destinations.end());
      if(d.replay_wait_result)
        for(auto replay=index+1;replay<d.wait_loop->second;++replay)
          changed.insert(decoded_[replay].destinations.begin(),decoded_[replay].destinations.end());
      auto exit = d.wait_loop->second;
      std::set<std::uint32_t> visited;
      while (exit != target && visited.insert(exit).second) {
        if (exit >= decoded_.size())
          return {};
        const auto &jump = *decoded_[exit].instruction;
        if (jump.control != "direct_branch" || jump.predicate ||
            !jump.branch_target)
          return {};
        exit = *jump.branch_target;
      }
      auto same = Identity(d, shadow);
      if (exit == target && same && *same == identity)
        return {{index, changed}};
      return {};
    }
    if (i.control != "none" || i.semantics.category != "compute" ||
        !In(i.opcode,
            {"mov", "cvt", "cvta", "add", "sub", "mul", "mad", "shl", "shr",
             "and", "or", "xor", "not", "neg", "setp", "selp"}))
      return {};
    Data(d, shadow, warp, lane, state.occurrences[index], Json::object(),
         context);
    changed.insert(d.destinations.begin(), d.destinations.end());
  }
  return {};
}
bool Executor::SharedData(const Decoded &d, State &state, std::uint32_t warp,
    std::uint32_t lane, std::uint64_t occurrence, const Json &contract,
    WarpContext &context, const LaneTrace &trace, bool wait) const {
  if (!context.shared) return false;
  const auto &i=*d.instruction;
  auto &runtime=*context.shared;
  const auto rank=c_.fixed_input.value("cluster_ctarank",0U);
  const SharedActor actor=(std::uint64_t(rank)<<32) | (warp*32+lane);
  const DynamicKey key{i.id,occurrence};
  const SharedValueSite site{c_.fixed_input.value("cta_coordinate",Coordinates{0,0,0}),warp*32+lane,key};
  auto value=[&](std::string_view text) { return ValueText(TrimView(text),state); };
  auto number=[&](std::string_view text) -> std::int64_t {
    auto n=ScalarInteger(value(text));
    if (!n) throw ExecutionError("shared_control_operand",i.line,"unknown shared-control operand: "+std::string(text));
    return Int64(*n,"shared-control operand");
  };
  const auto record=Record(d,state,occurrence,true);
  auto address=[&](std::string_view text) -> std::optional<std::uint64_t> {
    text=TrimView(text);
    if (text.starts_with('[') && text.ends_with(']')) text=TrimView(text.substr(1,text.size()-2));
    auto split=SplitAddress(text); if (!split) return {};
    auto v=value(split->first); auto owner=rank; std::optional<Integer> offset;
    if (auto p=std::get_if<ClusterSharedPointer>(&v)) { owner=p->target_rank; offset=Integer(p->local_address+split->second); }
    else if ((offset=ScalarInteger(v))) {
      // The record as the producer Resources read it: detached, on this trace's
      // lane, registered once where the lane is set.
      auto &executor=*context.resources;
      const auto record_lane=executor.facts.Register(trace);
      executor.resources.SetLane(record_lane);
      const auto detached=executor.facts.Detached(record,record_lane);
      if (auto inferred=executor.resources.SharedOwner(detached,split->first))
        owner=*inferred;
      *offset=executor.resources.AddressRegisterBits(detached,split->first,Integer(*offset+split->second));
    }
    if (!offset) return {};
    if (*offset<0 || *offset>UINT32_MAX)
      throw ExecutionError("shared_control_operand",i.line,"shared address outside u32");
    return (std::uint64_t(owner)<<32) | offset->convert_to<std::uint32_t>();
  };
  auto required_address=[&](std::string_view text) {
    auto a=address(text);
    if (!a) throw ExecutionError("shared_control_operand",i.line,"unknown shared address");
    return *a;
  };
  auto operand=[&](unsigned index) -> const std::string & { return d.operands.at(index).text; };
  auto encoded_values=[&](std::string_view text,unsigned width) {
    text=TrimView(text);
    if (text.starts_with('{') && text.ends_with('}')) text=text.substr(1,text.size()-2);
    std::vector<std::string_view> parts;
    for (;;) {
      auto comma=text.find(','); parts.push_back(TrimView(text.substr(0,comma)));
      if (comma==text.npos) break;
      text.remove_prefix(comma+1);
    }
    if (width%parts.size()) throw ExecutionError("shared_control_operand",i.line,"unaligned register vector width");
    std::vector<SharedByte> result;
    for (auto part : parts) {
      auto bytes=SharedBytes(value(part),width/parts.size()); result.insert(result.end(),bytes.begin(),bytes.end());
    }
    return result;
  };
  const auto cluster=c_.cluster_dim.value_or(Coordinates{1,1,1});
  const auto cluster_size=cluster[0]*cluster[1]*cluster[2];
  if (wait) {
    runtime.Wait(required_address(operand(1)),number(operand(2)),actor,Has(i,"parity"));
    return true;
  }
  if (i.opcode=="tensormap" && Has(i,"replace") && Has(i,"tile")) {
    auto v=value(operand(d.operands.size()-1));
    std::optional<std::int64_t> ordinal;
    if (d.operands.size()==3) ordinal=number(operand(1));
    const auto target=required_address(operand(0));
    if (!runtime.tensor_fields.contains(target)) {
      auto original=runtime.Read(site,target,128);
      if (auto fragment=std::get_if<DescriptorFragment>(&original);
          fragment && fragment->offset==0 && fragment->bytes==128 && c_.tma_descriptors.contains(fragment->descriptor))
        runtime.tensor_templates[target]=c_.tma_descriptors.at(fragment->descriptor);
    }
    runtime.tensor_fields[target][{i.modifiers.at(2),ordinal}]=std::move(v);
    runtime.constants.Invalidate(target,128);
    return true;
  }
  if (i.opcode=="tensormap" && Has(i,"cp_fenceproxy")) {
    auto text=TrimView(operand(0)); if (text.starts_with('[') && text.ends_with(']')) text=TrimView(text.substr(1,text.size()-2));
    const auto split=SplitAddress(text);
    auto v=split?value(split->first):Scalar{};
    auto *destination=std::get_if<AllocationPointer>(&v);
    if (!destination && split) {
      auto reg=ids_.find(split->first);
      if (reg!=ids_.end()) {
        try { UnresolvedControl(i,state,trace,reg->second); }
        catch (ExecutionError &error) {
          if (error.code=="branch_not_static") error.code="unresolved_tma_descriptor_location";
          throw ExecutionError(error.code,i.line,
              "tensor map copy destination cannot be resolved",error.required_parameters);
        }
      }
    }
    if (!destination || number(operand(2))!=128) throw ExecutionError("unresolved_tma_descriptor_location",i.line,"tensor map copy destination or size unresolved");
    destination->offset+=split->second;
    auto fields=runtime.tensor_fields.find(required_address(operand(1)));
    if (fields!=runtime.tensor_fields.end()) {
      auto decoded=DecodeTensorMap(fields->second,*destination);
      if (auto initial=runtime.tensor_templates.find(fields->first);initial!=runtime.tensor_templates.end())
        decoded["initial_descriptor"]=initial->second;
      PublishTensorMap(runtime.tensor_maps,decoded);
    }
    return true;
  }
  if (i.opcode=="mbarrier") {
    if (Has(i,"init")) runtime.Init(required_address(operand(0)),number(operand(1)));
    else if (Has(i,"inval")) { runtime.barriers.erase(required_address(operand(0))); ++runtime.revision; }
    else if (Has(i,"arrive")) {
      auto arrivals=std::int64_t(1), tx=std::int64_t(0);
      if (Has(i,"expect_tx")) tx=number(operand(2));
      else if (d.operands.size()>2) arrivals=number(operand(2));
      Set(d,state,{Integer(runtime.Update(required_address(operand(1)),arrivals,tx,actor))});
    } else if (Has(i,"expect_tx") || Has(i,"complete_tx"))
      runtime.Update(required_address(operand(0)),0,
          (Has(i,"expect_tx")?1:-1)*number(operand(1)),actor);
    else if (IsWait(i)) return false;
    else throw ExecutionError("shared_control_mbarrier_variant",i.line,"unsupported shared-control mbarrier variant");
    return true;
  }
  if (i.opcode=="bar" || i.opcode=="barrier") {
    auto prefix="cta."+std::to_string(rank)+".";
    if (Has(i,"warp") && Has(i,"sync")) {
      const auto mask=std::uint32_t(number(operand(0)));
      if (!(mask&(1U<<lane))) throw ExecutionError("shared_control_invalid_sync",i.line,"executing lane absent from warp mask");
      runtime.Collective(prefix+"warp."+std::to_string(warp)+"."+std::to_string(mask),std::popcount(mask),actor,key);
    } else if (Has(i,"cluster")) {
      if (Has(i,"arrive")) {
        runtime.Collective("cluster",c_.Threads()*cluster_size,actor,key,false);
        runtime.cluster_tickets[actor]=key;
      } else if (Has(i,"wait")) {
        if (!runtime.cluster_tickets.contains(actor)) throw ExecutionError("shared_control_invalid_sync",i.line,"cluster wait has no arrival");
        runtime.Collective("cluster",c_.Threads()*cluster_size,actor,runtime.cluster_tickets.at(actor));
      } else throw ExecutionError("shared_control_collective_variant",i.line,"unsupported cluster collective");
    } else if (Has(i,"sync") || Has(i,"arrive"))
      runtime.Collective(prefix+"named."+std::to_string(number(operand(0))),
          d.operands.size()>1?number(operand(1)):std::int64_t(c_.Threads()),actor,key,Has(i,"sync"));
    else throw ExecutionError("shared_control_collective_variant",i.line,"unsupported shared-control collective");
    return true;
  }
  if (i.opcode=="clusterlaunchcontrol" && i.semantics.operation=="try_cancel_complete_tx") {
    const auto destination=required_address(operand(0));
    const auto barrier=required_address(operand(1));
    const bool multicast=Has(i,"multicast::cluster::all");
    const auto count=multicast?cluster_size:1U;
    for (unsigned target=0;target<count;++target) {
      const auto owner=std::uint64_t(multicast?target:rank)<<32;
      // The response is opaque even when cancellation is assumed to fail.
      // Publish unknown bytes, and complete the 16-byte transaction in each
      // destination CTA. Query normalization remains responsible for its value.
      runtime.Defer(owner|(barrier&UINT32_MAX),0,-16,
          {{site,owner|(destination&UINT32_MAX),std::vector<SharedByte>(16)}});
    }
    return true;
  }
  if (i.opcode=="st" && Has(i,"async") && Has(i,"mbarrier::complete_tx::bytes")) {
    if (!i.memory_width) throw ExecutionError("shared_control_operand",i.line,"unknown asynchronous store width");
    runtime.Defer(required_address(operand(2)),0,-std::int64_t(*i.memory_width),
        {{site,required_address(operand(0)),encoded_values(operand(1),*i.memory_width)}});
    return true;
  }
  if (In(i.opcode,{"ld","st"}) && (Has(i,"shared") || Has(i,"shared::cta") || Has(i,"shared::cluster"))) {
    const bool store=i.opcode=="st"; auto pointer=address(operand(store?0:1));
    if (store) {
      if (!pointer || !i.memory_width) runtime.Clear();
      else {
        auto bytes=encoded_values(operand(1),*i.memory_width);
        if (std::none_of(bytes.begin(),bytes.end(),[](auto b){return b.has_value();}))
          runtime.Invalidate(*pointer,bytes.size());
        else runtime.Write({site,*pointer,std::move(bytes)});
      }
      ++runtime.revision;
    } else {
      std::vector<Scalar> values;
      const auto count=d.destination_slots.size();
      const auto width=i.memory_width && count?*i.memory_width/count:0;
      for (std::size_t k=0;k<count;++k) {
        auto loaded=pointer && width?runtime.Read(site,*pointer+k*width,width):Scalar{};
        if (auto n=ScalarInteger(loaded); n && Has(i,"s"+std::to_string(width*8)) && ((*n>>(width*8-1))&1)!=0)
          loaded=Integer(*n-(Integer(1)<<(width*8)));
        values.push_back(std::move(loaded));
      }
      Set(d,state,std::move(values));
    }
    return true;
  }
  if (i.opcode=="cp" && Has(i,"tensor") && Has(i,"mbarrier::complete_tx::bytes")) {
    const auto &facts=context.resources->facts; const auto record_lane=facts.Register(trace);
    auto &adapter=context.resources->resources; adapter.SetLane(record_lane);
    TmaAccesses transfer;
    // The runtime maps are searched after the program's own.
    try { transfer=adapter.Tma(facts.Detached(record,record_lane),TmaRequest(contract,false,&runtime.tensor_maps)); }
    catch (const SemanticError &error) { throw ExecutionError(error.code,i.line,error.what()); }
    std::vector<std::pair<std::uint64_t,std::uint64_t>> completions;
    for (const auto &[id,bytes] : transfer.completions) {
      const auto [kind,owner,identity]=adapter.interner().Location(id);
      if (kind!="physical") throw ExecutionError("shared_control_operand",i.line,"nonphysical TMA completion address");
      const auto target=(std::uint64_t(adapter.frame.RankOf(owner).value)<<32) | std::stoull(identity);
      if (!runtime.barriers.contains(target)) throw SharedBlocked{};
      completions.emplace_back(target,bytes);
    }
    for (const auto &pair : transfer.pairs) if (pair.destination) {
      const auto &binding=adapter.Binding(pair.destination->first);
      if (!binding.expression.empty() || !pair.destination->second) runtime.Clear();
      else runtime.Invalidate((std::uint64_t(pair.target_rank)<<32)|binding.offset,*pair.destination->second);
    }
    for (const auto &[target,bytes] : completions) runtime.Update(target,0,-std::int64_t(bytes),actor);
    return true;
  }
  if (i.opcode=="cp" && Has(i,"async") && !Has(i,"bulk")) {
    auto &pending=runtime.pending_copies[actor]; auto &groups=runtime.copy_groups[actor];
    if (Has(i,"mbarrier") && Has(i,"arrive")) {
      const auto target=required_address(operand(0));
      if (!Has(i,"noinc")) runtime.Update(target,-1,0,actor);
      for (auto &group : groups) {
        for (const auto &copy : group) runtime.Write(copy);
        group.clear();
      }
      for (const auto &copy : pending) runtime.Write(copy);
      pending.clear();
      runtime.Defer(target,1,0); return true;
    }
    if (Has(i,"commit_group") || Has(i,"wait_all")) { groups.push_back(std::move(pending)); pending.clear(); }
    if (Has(i,"wait_group") || Has(i,"wait_all")) {
      const auto keep=Has(i,"wait_group")?number(operand(0)):0;
      if (keep<0 || keep>7) throw ExecutionError("shared_control_operand",i.line,"invalid cp.async wait count");
      while (groups.size()>std::uint64_t(keep)) {
        for (const auto &copy : groups.front()) runtime.Write(copy);
        groups.pop_front(); ++runtime.revision;
      }
      return true;
    }
    if (Has(i,"commit_group")) return true;
    if ((Has(i,"shared") || Has(i,"shared::cta")) && Has(i,"global") && (d.operands.size()==3 || d.operands.size()==4)) {
      const auto width=number(operand(2)), source_width=d.operands.size()==4?number(operand(3)):width;
      if (!In(std::to_string(width),{"4","8","16"}) || source_width<0 || source_width>width)
        throw ExecutionError("shared_control_operand",i.line,"invalid cp.async width");
      Scalar loaded;
      auto text=TrimView(operand(1)); text=text.substr(1,text.size()-2);
      if (auto split=SplitAddress(text)) {
        auto source=value(split->first);
        if (auto p=std::get_if<AllocationPointer>(&source);p && constant_inputs_.contains(p->resource) && source_width) {
          auto n=constant_inputs_.at(p->resource).ReadInteger(Int64(p->offset+split->second,"source address"),source_width,false);
          if (n) loaded=*n;
        }
      }
      auto bytes=SharedBytes(loaded,source_width); bytes.resize(width,std::uint8_t(0));
      pending.push_back({site,required_address(operand(0)),std::move(bytes)}); return true;
    }
  }
  if (i.opcode=="tcgen05" && Has(i,"commit")) {
    const auto target=required_address(operand(0));
    if (Has(i,"multicast::cluster")) {
      const auto mask=std::uint32_t(number(operand(1)))&65535;
      if (!mask || mask>=(1U<<std::min(cluster_size,16U)))
        throw ExecutionError("shared_control_async_completion",i.line,"commit multicast mask outside cluster");
      for (unsigned r=0;r<cluster_size;++r) if (mask&(1U<<r))
        runtime.Defer((std::uint64_t(r)<<32)|(target&UINT32_MAX),1,0);
    } else runtime.Defer(target,1,0);
    return true;
  }
  if (i.opcode=="cp" && Has(i,"async") && Has(i,"bulk") &&
      (Has(i,"commit_group") || Has(i,"wait_group") || (Has(i,"tensor") && Has(i,"bulk_group")))) return false;
  if (In(i.opcode,{"cp","st"}) && Has(i,"async"))
    throw ExecutionError("shared_control_async_memory",i.line,"unsupported shared-control asynchronous transfer");
  return false;
}
LaneTask Executor::Lane(const WarpContract &warp, std::uint32_t lane,
                        WarpContext &context) const {

  std::uint32_t pc = 0;
  if (warp.entry_label) {
    auto found = labels_.find(*warp.entry_label);
    if (found == labels_.end())
      throw std::runtime_error("unknown warp entry label");
    pc = found->second;
  }
  ProfileScope initialize("frontend.execute.lane.initialize");
  State state = Initial(warp, lane);
  initialize.nodes = state.registers.size();
  initialize.Stop();
  LaneTrace trace;
  trace.warp_id = warp.warp_id;
  trace.lane_id = lane;
  trace.instructions.reserve(2048);
  std::uint64_t record_cursor = 0;
  bool normalized_records = false;
  auto append = [&](ExecutedInstruction record) {
    if (c_.compress_thread_records)
      record_pool_.Intern(record, record_cursor);
    trace.instructions.push_back(std::move(record));
  };
  if (data_branch_pass) PrepareControl();
  std::uint32_t previous_block = kMissing;
  std::map<std::uint32_t, std::uint64_t> loop_iterations;
  // Branches and predicated terminal instructions must use identical control
  // classification. Treating @p ret as an ordinary guarded instruction can
  // otherwise force an unknown polling loop to terminate.
  auto representative_outcome = [&](std::uint32_t pc) -> std::optional<bool> {
    const auto &i = *decoded_[pc].instruction;
    std::optional<bool> taken;
    const auto choice = ClassifyControl(pc);
    if (choice.kind == ControlLoop::Polling)
      Error(i, warp.warp_id, lane,
            "Polling control requires a modeled wait; neither assumption can "
            "force completion.",
            "polling_wait_not_modeled");
    if (choice.kind == ControlLoop::Unknown)
      Error(i, warp.warp_id, lane,
            "Control is not a structured if/else or a classified "
            "computation-loop test; supply explicit control information.",
            "loop_control_not_classified");
    const bool ordinary = choice.kind == ControlLoop::None;
    const char *key =
        ordinary ? "representative_branches" : "bounded_computation_loops";
    if (Assumed(c_, key)) {
      if (ordinary)
        taken = false;
      else {
        const auto limit = c_.raw.value("data_branch_loop_iterations", 4u);
        const auto completed =
            loop_iterations.at(choice.loop) + (choice.post_test ? 1 : 0);
        taken = (completed < limit) == choice.taken_stays;
      }
      auto &decision = trace.data_branch_decisions[i.line];
      decision.assumption = key;
      decision.rule = ordinary ? "fallthrough" : "body_iteration_bound";
      ++(*taken ? decision.taken : decision.not_taken);
    }
    return taken;
  };
  std::map<std::uint32_t, ProbeValue> probe_values;
  std::map<DynamicKey, WaitProbe> wait_probes;
  std::map<std::uint32_t, std::map<std::uint32_t, ProbeValue>> probe_joins;
  for (std::uint64_t steps = 0;; ++steps) {
    if ((steps & 1023) == 0) {
      if (poll_)
        poll_();
    }
    if (steps >= c_.max_steps_per_lane)
      throw ExecutionError("execution_step_limit_exceeded", pc < f_.instructions.size() ? f_.instructions[pc].line : 0, "warp " + std::to_string(warp.warp_id) +
                               " lane " + std::to_string(lane) +
                               ": exceeded max_steps_per_lane at pc " +
                               std::to_string(pc));
    if (pc >= decoded_.size())
      throw std::runtime_error("PTX control fell outside instruction stream");
    if (data_branch_pass) {
      const auto block = control_block_for_pc_[pc];
      // Update only at block transitions (including a one-block loop backedge).
      if (block != previous_block || pc == control_blocks_[block].begin) {
        const auto &active = control_blocks_[block].loops;
        if (previous_block != kMissing)
          for (auto id : control_blocks_[previous_block].loops)
            if (std::find(active.begin(), active.end(), id) == active.end()) loop_iterations.erase(id);
        for (auto id : active) {
          const auto &loop = control_loops_[id];
          auto [it, entered] = loop_iterations.try_emplace(id, 0);
          if (!entered && block == loop.header && loop.latches.contains(previous_block)) ++it->second;
        }
        previous_block = block;
      }
    }
    const auto &d = decoded_[pc];
    const auto &i = *d.instruction;
    if (auto join = probe_joins.find(pc); join != probe_joins.end()) {
      for (const auto &[reg, value] : join->second)
        if (value.alternative != state.registers[reg])
          probe_values[reg] = value;
      probe_joins.erase(join);
    }
    const auto &contract = c_.InstructionContract(i.line, i.id, warp.warp_id,
                                                  state.occurrences[pc]);
    if (i.control == "direct_branch") {
      if (!i.branch_target)
        Error(i, warp.warp_id, lane, "unresolved direct branch");
      auto target = *i.branch_target;
      std::optional<std::pair<std::string, Integer>> assumed_equality;
      if (c_.fixed_input.value("assume_trap_assertions", false))
        if (auto assertion = Assertion(i, target)) {
          const auto &[reg, expected] = *assertion;
          if (!Unknown(state.registers[reg]) && state.registers[reg] != Scalar(expected))
            Error(i, warp.warp_id, lane, "assertion assumption contradicts a known value");
          state.registers[reg] = expected;
          state.registers[d.predicate] = true;
          assumed_equality = {names_[reg], expected};
          ++trace.assumed_assertions;
        }
      if (d.predicate != kMissing) {
        auto probe = probe_values.find(d.predicate);
        if (probe != probe_values.end()) {
          auto actual = Truth(state.registers[d.predicate]),
               alternative = Truth(probe->second.alternative);
          if (!actual || !alternative)
            Error(i, warp.warp_id, lane,
                  "probe-dependent branch has unresolved alternative");
          bool a = *actual, b = *alternative;
          if (i.predicate_negated) {
            a = !a;
            b = !b;
          }
          if (a != b) {
            auto fallback =
                !a && b
                    ? Fallback(d, state, target,
                               wait_probes.at(probe->second.key).identity,
                               warp.warp_id, lane, context)
                    : std::optional<
                          std::pair<std::uint32_t, std::set<std::uint32_t>>>{};
            if (!fallback || c_.BranchOutcome(i.line, warp.warp_id, lane,
                                              state.branch_visits[pc]))
              Error(i, warp.warp_id, lane,
                    "schedule-dependent probe lacks same-target "
                    "side-effect-free fallback wait", "schedule_dependent_wait_unsupported");
            auto &record = trace.instructions[wait_probes.at(probe->second.key)
                                                  .trace_index];
            normalized_records = true;
            record.mutableMetadata().canonical_wait_probe = true;
            record.mutableMetadata().schedule_seed_outcome.reset();
            record.mutableMetadata().probe_fallback_key =
                DynamicKey{fallback->first, state.occurrences[fallback->first]};
            auto &join = probe_joins[target];
            for (auto reg : fallback->second) {
              auto previous = probe_values.find(reg);
              join[reg] = {probe->second.key,
                           previous != probe_values.end() &&
                                   previous->second.key == probe->second.key
                               ? previous->second.alternative
                               : state.registers[reg]};
            }
          }
        }
      }
      bool diamond = d.predicate != kMissing &&
                     Unknown(state.registers[d.predicate]) &&
                     !c_.BranchOutcome(i.line, warp.warp_id, lane,
                                       state.branch_visits[pc]) &&
                     target > pc + 1;
      for (std::uint32_t index = pc + 1; diamond && index < target; ++index) {
        if (index >= decoded_.size()) {
          diamond = false;
          break;
        }
        const auto &candidate = *decoded_[index].instruction;
        bool pure = candidate.semantics.category == "compute",
             private_read = candidate.semantics.category == "memory" &&
                            candidate.semantics.memory_mode == "read" &&
                            candidate.semantics.memory_space &&
                            In(*candidate.semantics.memory_space,
                               {"param", "local", "const"});
        if (candidate.control != "none" || (!pure && !private_read))
          diamond = false;
      }
      // In the exact last-resort pass, the same treatment extends to a forward region with
      // several arms (if-else, nested ifs): every path from the branch reaches one join
      // without memory, synchronization or control effects beyond forward branches inside it.
      std::uint32_t join = target;
      if (!diamond && exact_control_pass && d.predicate != kMissing &&
          Unknown(state.registers[d.predicate]) && target > pc &&
          !c_.BranchOutcome(i.line, warp.warp_id, lane, state.branch_visits[pc])) {
        diamond = true;
        for (std::uint32_t index = pc + 1; diamond && index < join; ++index) {
          if (index >= decoded_.size()) { diamond = false; break; }
          const auto &candidate = *decoded_[index].instruction;
          // Registers declared in a nested scope (inline asm blocks) cannot be named in
          // the region's record outside that scope.
          if (candidate.scope != i.scope) { diamond = false; break; }
          if (candidate.control == "direct_branch") {
            if (!candidate.branch_target || *candidate.branch_target <= index) diamond = false;
            else join = std::max(join, *candidate.branch_target);
            continue;
          }
          bool pure = candidate.semantics.category == "compute",
               private_read = candidate.semantics.category == "memory" &&
                              candidate.semantics.memory_mode == "read" &&
                              candidate.semantics.memory_space &&
                              In(*candidate.semantics.memory_space, {"param", "local", "const"});
          if (candidate.control != "none" || (!pure && !private_read)) diamond = false;
        }
        diamond = diamond && join < decoded_.size();
        if (diamond) ++trace.opaque_regions[i.line];
      }
      if (diamond) {
        auto occurrence = state.occurrences[pc]++;
        ++state.branch_visits[pc];
        for (std::uint32_t index = pc + 1; index < join; ++index)
          for (auto reg : decoded_[index].destinations) {
            state.registers[reg] = Scalar{};
            state.origins[reg].reset();
            probe_values.erase(reg);
          }
        auto record = Record(d, state, occurrence, {});
        std::set<std::string> reads, writes;
        for (auto index = pc + 1; index < join; ++index) {
          const auto &skipped = decoded_[index];
          for (auto reg : skipped.destinations) writes.insert(names_[reg]);
          if (skipped.instruction->control == "direct_branch") {
            // A branch inside a multi-arm region reads only its guard; the target is a label.
            if (skipped.predicate != kMissing) reads.insert(names_[skipped.predicate]);
            continue;
          }
          for (auto reg : skipped.uses) {
            bool source = skipped.predicate == reg;
            for (std::size_t operand = skipped.destinations.empty() ? 0 : 1;
                 operand < skipped.operands.size(); ++operand)
              source |= std::find(skipped.operands[operand].registers.begin(),
                  skipped.operands[operand].registers.end(), reg) != skipped.operands[operand].registers.end();
            if (source) reads.insert(names_[reg]);
          }
        }
        record.mutableMetadata().opaque_compute_register_reads.assign(reads.begin(), reads.end());
        record.mutableMetadata().opaque_compute_register_writes.assign(writes.begin(), writes.end());
        for (auto index = pc + 1; index < join; ++index)
          record.mutableMetadata().opaque_compute_diamond_indices.push_back(index);
        record.mutableMetadata().opaque_compute_join_label =
            join == target ? i.branch_label : "pc:" + std::to_string(join);
        append(std::move(record));
        pc = join;
        continue;
      }
      auto visit = state.branch_visits[pc]++;
      std::optional<bool> bound;
      if (target <= pc)
        if (auto loop = [&]() {
              auto found = c_.FindLoopBound(i.branch_label, warp.warp_id);
              return found ? found
                           : c_.FindLoopBound(m_.view(i.operands[0]),
                                              warp.warp_id);
            }()) {
          if (loop->backedge_line && *loop->backedge_line != i.line)
            Error(i, warp.warp_id, lane,
                  "loop bound names a different backedge line");
          auto count = state.backedges[i.branch_label]++;
          bound = count < loop->trip_count - 1;
        }
      auto override = c_.BranchOutcome(i.line, warp.warp_id, lane, visit);
      auto predicate = d.predicate == kMissing
                           ? std::optional<bool>(true)
                           : Truth(state.registers[d.predicate]);
      if (predicate && i.predicate_negated)
        *predicate = !*predicate;
      auto supplied = bound ? bound : override;
      if (supplied && predicate && *supplied != *predicate)
        Error(i, warp.warp_id, lane,
              "known predicate disagrees with fixed-input branch contract");
      auto taken = supplied ? supplied : predicate;
      if (!taken && d.predicate != kMissing && target > pc + 1)
        if (const auto *arm = MergedArmAt(pc); arm && (!arm->aligned || arm->uniform)) {
          // Two-arm merge: this lane runs the fall-through arm (the superset arm); the arm's
          // registers are dead after the join: nothing after it reads a register the arm writes.
          auto record = Record(d, state, state.occurrences[pc]++, {});
          record.mutableMetadata().merged_arm_join = target;
          append(std::move(record));
          ++pc;
          continue;
        }
      if (!taken && data_branch_pass)
        taken = representative_outcome(pc);
      if (!taken)
        UnresolvedControl(i, state, trace, d.predicate);
      auto occurrence = state.occurrences[pc]++;
      const auto capture = override && !bound && d.predicate != kMissing
          ? CaptureDecision(i.line, *taken, visit) : std::optional<BranchScopeDecision>{};
      if (*taken || capture) {
        ExecutedInstruction record;
        if (capture) {
          record = Record(d, state, occurrence, true);
          record.mutableMetadata().branch_scope_decision = capture;
        } else {
          record.mutableMetadata().instruction_id = pc;
          record.mutableMetadata().occurrence = occurrence;
          record.mutableMetadata().predicate_value = true;
        }
        record.mutableMetadata().assumed_equality = std::move(assumed_equality);
        append(std::move(record));
      }
      pc = *taken ? target : pc + 1;
      continue;
    }
    if (d.predicate != kMissing && probe_values.contains(d.predicate))
      Error(i, warp.warp_id, lane,
            "schedule-dependent probe guards noncanonical instruction");
    bool active = true;
    if (d.predicate != kMissing) {
      auto known = Truth(state.registers[d.predicate]);
      if (known)
        active = *known;
      else if (contract.contains("active_lanes") &&
               contract["active_lanes"].is_array())
        active = std::find(contract["active_lanes"].begin(),
                           contract["active_lanes"].end(),
                           Json(lane)) != contract["active_lanes"].end();
      else if (contract.contains("predicate") &&
               contract["predicate"].is_boolean())
        active = contract["predicate"].get<bool>();
      else if (data_branch_pass && In(i.control, {"return", "exit", "trap"})) {
        const auto chosen = representative_outcome(pc);
        if (!chosen)
          UnresolvedControl(i, state, trace, d.predicate);
        // The common guard path below applies predicate negation once.
        active = i.predicate_negated ? !*chosen : *chosen;
      } else if (data_branch_pass && Assumed(c_, "representative_branches")) {
        // Representative control flow: an instruction whose guard is not
        // determined by the fixed input executes.
        active = !i.predicate_negated;
        auto &decision = trace.data_branch_decisions[i.line];
        decision.assumption = "representative_branches";
        decision.rule = "guard_executes";
        ++decision.taken;
      } else
        UnresolvedControl(i, state, trace, d.predicate);
      if (i.predicate_negated)
        active = !active;
    }
    auto occurrence = state.occurrences[pc]++;
    if (!active) {
      ++pc;
      continue;
    }
    const auto& assumptions = c_.raw.at("semantic_assumptions");
    if (i.opcode == "elect" && !assumptions.at("representative_election").get<bool>())
      throw ExecutionError("assumption_required_representative_election", i.line,
          "Election outcomes are unsupported. Enable Representative election (representative_election) to analyze one representative lane.");
    if (i.opcode == "clusterlaunchcontrol" && !assumptions.at("clc_always_fails").get<bool>())
      throw ExecutionError("assumption_required_clc_always_fails", i.line,
          "CLC outcomes are unsupported. Enable Failed launch cancellation (clc_always_fails) to analyze failed requests.");
    std::map<std::uint32_t, ProbeValue> updates;
    if (!probe_values.empty()) {
      std::map<std::uint32_t, ProbeValue> dependencies;
      std::set<DynamicKey> keys;
      for (std::size_t operand = d.destinations.empty() ? 0 : 1;
           operand < d.operands.size(); ++operand)
        for (auto reg : d.operands[operand].registers)
          if (auto found = probe_values.find(reg);
              found != probe_values.end()) {
            dependencies.emplace(*found);
            keys.insert(found->second.key);
          }
      if (!dependencies.empty()) {
        if (i.semantics.category != "compute" ||
            !In(i.opcode,
                {"mov", "cvt", "selp", "setp", "and", "or", "xor", "not"}) ||
            keys.size() != 1)
          Error(i, warp.warp_id, lane,
                "schedule-dependent probe escapes canonical fallback control");
        State shadow = state;
        for (const auto &[reg, value] : dependencies)
          shadow.registers[reg] = value.alternative;
        Data(d, shadow, warp.warp_id, lane, occurrence, contract, context);
        for (auto reg : d.destinations) {
          if (Unknown(shadow.registers[reg]))
            Error(i, warp.warp_id, lane, "unresolved probe-result computation");
          updates[reg] = {*keys.begin(), shadow.registers[reg]};
        }
      }
    }
    if (d.wait_loop) {
      auto record = Record(d, state, occurrence, true);
      if (context.shared) for (;;) {
        bool blocked=false;
        try { SharedData(d,state,warp.warp_id,lane,occurrence,contract,context,trace,true); }
        catch (const SharedBlocked &) { blocked=true; }
        if (!blocked) break;
        co_yield Deferred{lane,pc,occurrence,true};
      }
      record.mutableMetadata().collapsed_wait_loop = true;
      record.mutableMetadata().normalized_wait_result = true;
      record.mutableMetadata().wait_exit_label = d.wait_loop->first;
      for (auto index = pc; index < (d.replay_wait_result ? pc+1 : d.wait_loop->second); ++index)
        record.mutableMetadata().collapsed_wait_indices.push_back(index);
      append(std::move(record));
      Set(d, state, {true});
      Origins(d, state, occurrence);
      for (auto reg : d.destinations)
        probe_values.erase(reg);
      pc = d.replay_wait_result ? pc+1 : d.wait_loop->second;
      continue;
    }
    auto record = Record(d, state, occurrence, true);
    bool probe = IsWait(i);
    if (probe)
      record.mutableMetadata().schedule_seed_outcome = "WaitAgain";
    append(std::move(record));
    if (i.control == "indirect_branch") {
      const auto index=ScalarInteger(Value(d.operands.front(),state));
      if (!index) UnresolvedControl(i,state,trace,d.operands.front().reg);
      if (*index<0 || *index>=i.indirect_targets.size())
        Error(i,warp.warp_id,lane,"indirect branch index is outside its target table");
      pc=i.indirect_targets[index->convert_to<std::size_t>()];
      continue;
    }
    if (i.control == "call") {
      if (!contract.value("opaque_call", false) ||
          !contract.value("contains_no_sync_or_memory", false))
        Error(i, warp.warp_id, lane,
              "call lacks explicit no-sync/no-memory summary");
      Invalidate(d, state);
      Origins(d, state, occurrence);
      ++pc;
      continue;
    }
    if (i.control == "return" || i.control == "exit")
      break;
    if (i.control == "trap")
      Error(i, warp.warp_id, lane, "fixed path reaches trap");
    for (;;) {
      std::optional<Deferred> dependency;
      try {
        if (!SharedData(d,state,warp.warp_id,lane,occurrence,contract,context,trace))
          Data(d, state, warp.warp_id, lane, occurrence, contract, context, probe);
      } catch (const Deferred &waiting) {
        dependency = waiting;
      } catch (const SharedBlocked &) {
        dependency = Deferred{lane,pc,occurrence,true};
      }
      if (!dependency) break;
      co_yield *dependency;
    }
    if (i.opcode == "clusterlaunchcontrol" && Has(i, "query_cancel") && Has(i, "is_canceled") &&
        !d.destinations.empty()) {
      normalized_records = true;
      trace.instructions.back().mutableMetadata().normalized_clc_result = Truth(state.registers[d.destinations.front()]);
    }
    Origins(d, state, occurrence);
    for (auto reg : d.destinations)
      probe_values.erase(reg);
    for (auto &[reg, value] : updates)
      if (value.alternative != state.registers[reg])
        probe_values[reg] = std::move(value);
    if (probe) {
      auto identity = Identity(d, state);
      if (!identity)
        Error(i, warp.warp_id, lane,
              "nonblocking wait probe has unresolved barrier or target");
      DynamicKey key{pc, occurrence};
      wait_probes[key] = {trace.instructions.size() - 1, *identity};
      for (auto reg : d.destinations)
        probe_values[reg] = {key, true};
    }
    ++pc;
  }
  for (const auto &[key, probe] : wait_probes) {
    normalized_records = true;
    auto &record = trace.instructions[probe.trace_index];
    record.mutableMetadata().canonical_wait_probe = true;
    record.mutableMetadata().normalized_wait_result = false;
    record.mutableMetadata().schedule_seed_outcome.reset();
  }
  if (c_.compress_thread_records && normalized_records) {
    record_cursor = 0;
    std::size_t position = 0;
    for (auto &record : trace.instructions) {
      if ((position++ & 1023) == 0 && poll_) poll_();
      record_pool_.Intern(record, record_cursor);
    }
  }
  for (std::uint32_t index = 0; index < state.branch_visits.size(); ++index)
    if (state.branch_visits[index])
      trace.branch_visits[index] = state.branch_visits[index];
  trace.loop_backedges = std::move(state.backedges);
  co_return trace;
}

const Executor::MergedArm *Executor::MergedArmAt(std::uint32_t pc) const {
  auto [entry, inserted] = merged_arms_.try_emplace(pc);
  if (!inserted) return entry->second ? &*entry->second : nullptr;
  const auto &branch = *decoded_[pc].instruction;
  if (branch.control != "direct_branch" || decoded_[pc].predicate == kMissing || !branch.branch_target ||
      *branch.branch_target <= pc + 1 || *branch.branch_target > decoded_.size())
    return nullptr;
  MergedArm arm;
  std::set<std::uint32_t> writes;
  for (auto index = pc + 1; index < *branch.branch_target; ++index) {
    const auto &decoded = decoded_[index];
    const auto &candidate = *decoded.instruction;
    const auto &s = candidate.semantics;
    if (candidate.control != "none") return nullptr;
    // "compute" is also the default category; only these opcodes are register-only.
    const bool pure = s.category == "compute" && In(candidate.opcode, {"mov", "add", "sub", "mul", "mad", "fma",
        "div", "rem", "neg", "abs", "min", "max", "and", "or", "xor", "not", "cnot", "shl", "shr", "shf", "lop3",
        "prmt", "bfe", "bfi", "bfind", "brev", "clz", "popc", "cvt", "cvta", "selp", "setp", "set", "slct", "rcp",
        "sqrt", "rsqrt", "ex2", "lg2", "sin", "cos", "tanh", "copysign", "testp", "mul24", "mad24", "sad", "dp2a",
        "dp4a", "fns"});
    const bool private_read = s.category == "memory" && s.memory_mode == "read" && s.memory_space &&
        In(*s.memory_space, {"param", "local", "const"});
    const bool tmem = candidate.opcode == "tcgen05" && In(s.operation, {"load", "store", "wait_load", "wait_store"});
    if (!pure && !private_read && !tmem) return nullptr;
    arm.aligned |= tmem;
    writes.insert(decoded.destinations.begin(), decoded.destinations.end());
  }
  if (reference_ranges_.empty()) {
    reference_ranges_.assign(names_.size(), {UINT32_MAX, 0});
    for (std::uint32_t index = 0; index < decoded_.size(); ++index)
      for (const auto *list : {&decoded_[index].uses, &decoded_[index].destinations})
        for (auto reg : *list) {
          auto &range = reference_ranges_.at(reg);
          range = {std::min(range.first, index), std::max(range.second, index)};
        }
  }
  for (auto reg : writes)
    if (reference_ranges_.at(reg).first <= pc || reference_ranges_.at(reg).second >= *branch.branch_target)
      return nullptr;
  arm.writes.assign(writes.begin(), writes.end());
  arm.uniform = StaticWarpUniform(pc, decoded_[pc].predicate, 0);
  entry->second = std::move(arm);
  return &*entry->second;
}
// The static proof that `reg` is warp-uniform at `use`: its reaching definition lies in use's
// basic block (no label entry in between) and is an unpredicated setp/selp/not/and/or/xor/mov/cvt
// of constants and such registers, or a vote.sync (any/all/uni/ballot) whose mask is the full
// warp.
bool Executor::StaticWarpUniform(std::uint32_t use, std::uint32_t reg, unsigned depth) const {
  if (depth > 16) return false;
  for (auto index = use; index-- > 0;) {
    if (label_entries_.contains(index + 1)) return false;
    const auto &d = decoded_[index];
    if (std::find(d.destinations.begin(), d.destinations.end(), reg) == d.destinations.end()) continue;
    const auto &i = *d.instruction;
    if (d.predicate != kMissing) return false;
    if (i.opcode == "vote") {
      if (i.modifiers.size() != 3 || i.modifiers[0] != "sync" || !In(i.modifiers[1], {"any", "all", "uni", "ballot"}) ||
          d.operands.size() != 3)
        return false;
      const auto &mask = d.operands[2];
      if (mask.reg == kMissing && mask.registers.empty()) return FullWarpMask(mask.constant);
      if (mask.reg == kMissing) return false;
      // A register mask: its reaching definition is a mov of a full constant.
      for (auto at = index; at-- > 0;) {
        if (label_entries_.contains(at + 1)) return false;
        const auto &m = decoded_[at];
        if (std::find(m.destinations.begin(), m.destinations.end(), mask.reg) == m.destinations.end()) continue;
        return m.predicate == kMissing && m.instruction->opcode == "mov" && m.operands.size() == 2 &&
               m.operands[1].reg == kMissing && m.operands[1].registers.empty() && FullWarpMask(m.operands[1].constant);
      }
      return false;
    }
    if (!In(i.opcode, {"setp", "selp", "not", "and", "or", "xor", "mov", "cvt"}) || d.operands.empty()) return false;
    for (std::size_t operand = 1; operand < d.operands.size(); ++operand) {
      std::vector<std::uint32_t> sources = d.operands[operand].registers;
      if (d.operands[operand].reg != kMissing) sources.push_back(d.operands[operand].reg);
      for (auto source : sources)
        if (!StaticWarpUniform(index, source, depth + 1)) return false;
    }
    return true;
  }
  return false;
}

std::optional<ExecutionProgram> Executor::RunTemplates() {
  auto task=RunTemplateTask(nullptr);task.Resume();
  if(!task.Done())throw std::logic_error("noncooperative template task suspended");
  return task.TakeResult();
}

TemplateTask Executor::RunTemplateTask(SharedExecution* runtime) {
  using namespace detail;
  if (!c_.compress_thread_records ||
      !c_.loop_bounds.empty() ||
      !c_.instruction_contracts.empty()) co_return std::nullopt;
  const auto supports_template=[](const Decoded& d) {
    const auto &i=*d.instruction;
    if (!In(i.control,{"none","direct_branch","indirect_branch","return","exit"})) return false;
    if(i.opcode=="vote" && (i.modifiers.size()!=3 || i.modifiers[0]!="sync" ||
        !In(i.modifiers[1],{"ballot","all","any","uni"}) || d.operands.size()!=3))return false;
    if(i.opcode=="shfl" && (i.modifiers.size()<2 || i.modifiers[0]!="sync" ||
        !In(i.modifiers[1],{"idx","up","down","bfly"}) || d.operands.size()!=5))return false;
    return true;
  };

  std::vector<std::vector<std::uint32_t>> source_registers(decoded_.size());
  for (const auto &d:decoded_) {
    auto &reads=source_registers[d.instruction->id];
    auto add=[&](std::uint32_t reg) {
      if (reg!=kMissing && std::find(reads.begin(),reads.end(),reg)==reads.end()) reads.push_back(reg);
    };
    for (std::size_t k=1;k<d.operands.size();++k) {
      const auto &operand=d.operands[k];
      for (auto reg:operand.registers) add(reg);
      add(operand.reg);
      // Register packs can contain component-qualified special registers.
      for (std::uint32_t reg=0;reg<names_.size();++reg) if (names_[reg].find('.')!=std::string::npos) {
        auto at=operand.text.find(names_[reg]);
        if (at==std::string::npos) continue;
        auto end=at+names_[reg].size();
        if (end==operand.text.size() || (!std::isalnum(static_cast<unsigned char>(operand.text[end])) &&
            operand.text[end]!='_' && operand.text[end]!='.')) add(reg);
      }
    }
  }
  struct Member {std::uint32_t warp,lane;};
  struct Group {
    State state;
    std::vector<std::size_t> registers;
    std::shared_ptr<std::vector<std::size_t>> path=std::make_shared<std::vector<std::size_t>>();
    PathId path_id{}; // the identity of `path` (the root is PathId 0)
    algorithm::SharedArray<std::size_t> members;
    std::uint32_t pc{};
    std::uint64_t steps{};
    std::uint64_t assumed_assertions{};
    std::map<std::uint32_t,ProbeValue> probe_values;
    std::map<DynamicKey,std::size_t> probes;
    std::map<std::uint32_t,std::map<std::uint32_t,ProbeValue>> probe_joins;
    struct SharedWork {
      std::vector<std::size_t> pending;
      std::vector<std::vector<Scalar>> outputs;
      bool handled=false;
    };
    std::optional<SharedWork> shared;
    std::vector<std::size_t> collective_results;
    // Join pc of the merged one-armed diamond being executed and the registers its arm
    // writes; they become unknown at the join (straight-line arms never nest).
    std::optional<std::pair<std::uint32_t,std::vector<std::uint32_t>>> merged_join;
  };
  auto arena=std::make_shared<TemplateArena>(); arena->poll=poll_;
  arena->Constant(Scalar{});
  std::vector<Member> members;
  auto member_coordinates=std::make_shared<std::vector<std::uint32_t>>();
  for (const auto &warp:c_.warps) {
    if (warp.entry_label) co_return std::nullopt;
    for (std::uint32_t lane=0;lane<c_.warp_size;++lane) {
      auto tid=warp.warp_id*c_.warp_size+lane;
      if (c_.analysis_thread_range && (tid<c_.analysis_thread_range->first || tid>=c_.analysis_thread_range->second)) continue;
      members.push_back({warp.warp_id,lane});member_coordinates->push_back(tid);
    }
  }
  if (members.empty()) co_return std::nullopt;
  TemplateValue thread_coordinate;
  thread_coordinate.member_value=[member_coordinates](std::size_t member)->Scalar{return Integer(member_coordinates->at(member));};
  const auto coordinate_value=arena->values.size();
  arena->values.push_back(std::move(thread_coordinate));
  arena->caches.resize(members.size());
  const auto first_warp=std::find_if(c_.warps.begin(),c_.warps.end(),[&](const auto& warp){return warp.warp_id==members.front().warp;});
  Group first; first.state=Initial(*first_warp,members.front().lane);
  auto default_warp=*first_warp;default_warp.initial_registers.clear();
  const auto default_initial=Initial(default_warp,members.front().lane);
  std::vector<std::size_t> first_members;
  auto member_indices=std::make_shared<std::map<std::uint32_t,std::size_t>>();
  for (std::size_t t=0;t<members.size();++t) {
    first_members.push_back(t);
    member_indices->emplace(members[t].warp*c_.warp_size+members[t].lane,t);
  }
  first.members=std::move(first_members);
  for (std::size_t reg=0;reg<names_.size();++reg) {
    const auto& name=names_[reg];
    unsigned kind=name=="%laneid"?1:name=="%warpid"?2:name=="%tid.x"?3:name=="%tid.y"?4:name=="%tid.z"?5:0;
    if(c_.special_registers.contains(name))kind=0;
    std::map<std::uint32_t,Integer> overrides;
    for(const auto& warp:c_.warps)if(auto found=warp.initial_registers.find(name);found!=warp.initial_registers.end())
      overrides.emplace(warp.warp_id,found->second);
    if(!kind && overrides.empty()) {
      first.registers.push_back(arena->Constant(first.state.registers[reg]));continue;
    }
    Scalar fallback=default_initial.registers[reg];
    TemplateValue node;node.operands={coordinate_value};
    node.evaluate=[kind,overrides=std::move(overrides),fallback,dimensions=c_.block_dim,warp_size=c_.warp_size](const std::vector<Scalar>& args)->Scalar {
      const auto thread=std::get<Integer>(args.at(0)).convert_to<std::uint32_t>();
      if(auto found=overrides.find(thread/warp_size);found!=overrides.end())return found->second;
      switch(kind) {
        case 1:return Integer(thread%warp_size);
        case 2:return Integer(thread/warp_size);
        case 3:return Integer(thread%dimensions[0]);
        case 4:return Integer(thread/dimensions[0]%dimensions[1]);
        case 5:return Integer(thread/(dimensions[0]*dimensions[1]));
        default:return fallback;
      }
    };
    arena->FoldCoordinates(node);
    first.state.registers[reg]=node.uniform.value_or(Scalar{});
    if(node.uniform)first.registers.push_back(arena->Constant(*node.uniform));
    else {first.registers.push_back(arena->values.size());arena->values.push_back(std::move(node));}
  }
  ExecutionProgram result; result.module=&m_; result.function=&f_; result.config=c_;
  result.register_names=names_; result.static_execution_counts.resize(decoded_.size());
  result.lane_traces.resize(members.size()); result.template_execution.used=true;
  // One split-tree node per path object, in creation order. `child` gets a copy
  // of `parent`'s path with the next PathId; the parent keeps its own.
  result.template_paths.emplace_back();
  const auto fork_path=[&result](Group& child,const Group& parent,SplitReason reason) {
    child.path=std::make_shared<std::vector<std::size_t>>(*parent.path);
    const PathId id{static_cast<std::uint32_t>(result.template_paths.size())};
    result.template_paths.push_back({parent.path_id,static_cast<std::uint32_t>(parent.path->size()),parent.pc,reason,{}});
    child.path_id=id;
  };
  AddressTable shared_addresses;BarrierTable shared_barriers;
  std::unique_ptr<ExecutorResources> shared_resources; // facts over the in-progress result
  if(runtime)shared_resources=std::make_unique<ExecutorResources>(result,c_.fixed_input.value("cta_coordinate",Coordinates{0,0,0}),shared_addresses,shared_barriers);
  std::map<std::uint32_t,WarpContext> shared_contexts;
  std::deque<Group> pending; pending.push_back(std::move(first));
  std::deque<Group> waiting;
  // A group's step is one step of every member: its budget is the per-thread one
  // times kTemplateStepFactor (practical shapes run long, cheap shared loops).
  const std::uint64_t template_step_budget =
      c_.max_steps_per_lane > UINT64_MAX / FrontendConfig::kTemplateStepFactor
          ? UINT64_MAX : c_.max_steps_per_lane * FrontendConfig::kTemplateStepFactor;
  std::map<DynamicKey,std::shared_ptr<TemplateExchange>> exchanges;
  std::map<DynamicKey,std::shared_ptr<TemplateVote>> votes;
  std::vector<std::pair<std::size_t,algorithm::SharedArray<std::size_t>>> vote_uses;
  struct ExchangeUse {
    std::shared_ptr<TemplateExchange> exchange;
    algorithm::SharedArray<std::size_t> members;
    std::size_t selection;
  };
  std::vector<ExchangeUse> exchange_uses;
  std::size_t exchange_revision=0,retried_revision=SIZE_MAX;
  std::uint64_t shared_revision=UINT64_MAX;
  bool split_collective_waits=false,stalled=false;
  auto owner=shared_from_this();
  struct EvaluationScratch {
    State state;
    std::vector<std::uint32_t> touched;
  };
  auto scratch=std::make_shared<EvaluationScratch>();
  scratch->state.registers.resize(names_.size());
  scratch->state.origins.resize(names_.size());
  struct ComputeDiamond {
    std::vector<std::uint32_t> writes,indices;
    std::vector<std::string> read_names,write_names;
  };
  std::map<std::uint32_t,std::optional<ComputeDiamond>> diamonds;
  const auto compute_diamond=[&](std::uint32_t pc)->const ComputeDiamond* {
    auto [entry,inserted]=diamonds.try_emplace(pc);
    if(!inserted)return entry->second?&*entry->second:nullptr;
    const auto& branch=*decoded_[pc].instruction;
    if(!branch.branch_target || *branch.branch_target<=pc+1)return nullptr;
    ComputeDiamond diamond;
    std::set<std::uint32_t> writes;
    std::set<std::string> read_names,write_names;
    for(auto index=pc+1;index<*branch.branch_target;++index) {
      if(index>=decoded_.size())return nullptr;
      const auto& decoded=decoded_[index];const auto& candidate=*decoded.instruction;
      const bool pure=candidate.semantics.category=="compute";
      const bool private_read=candidate.semantics.category=="memory" && candidate.semantics.memory_mode=="read" &&
        candidate.semantics.memory_space && In(*candidate.semantics.memory_space,{"param","local","const"});
      if(candidate.control!="none" || (!pure && !private_read))return nullptr;
      diamond.indices.push_back(index);
      for(auto reg:decoded.destinations){writes.insert(reg);write_names.insert(names_[reg]);}
      for(auto reg:decoded.uses) {
        bool source=decoded.predicate==reg;
        for(std::size_t operand=decoded.destinations.empty()?0:1;operand<decoded.operands.size();++operand)
          source|=std::find(decoded.operands[operand].registers.begin(),decoded.operands[operand].registers.end(),reg)!=decoded.operands[operand].registers.end();
        if(source)read_names.insert(names_[reg]);
      }
    }
    diamond.writes.assign(writes.begin(),writes.end());
    diamond.read_names.assign(read_names.begin(),read_names.end());
    diamond.write_names.assign(write_names.begin(),write_names.end());
    entry->second=std::move(diamond);return &*entry->second;
  };
  // Every warp with a member in the domain has all its CTA members in it: a warp split
  // across template groups could disagree on the predicate (aligned arms would be UB).
  std::map<std::uint32_t,std::size_t> warp_members;
  const auto whole_warps=[&](const std::vector<std::size_t>& domain) {
    if(warp_members.empty())for(const auto& member:members)++warp_members[member.warp];
    std::map<std::uint32_t,std::size_t> present;
    for(auto member:domain)++present[members[member].warp];
    return std::all_of(present.begin(),present.end(),[&](const auto& item){return item.second==warp_members.at(item.first);});
  };
  // The warp-uniform predicate for the template executor when the static proof
  // (MergedArm::uniform) fails: a provably warp-uniform template value (bounded walk of its
  // value DAG): a known uniform value, a full-warp vote.sync result, or a computation of such
  // values. A vote's mask node also reads the voted predicate, so an unfolded mask is
  // evaluated per member.
  const auto warp_uniform=[&](std::size_t root,const std::vector<std::size_t>& domain) {
    std::vector<std::size_t> stack{root};std::set<std::size_t> seen;
    while(!stack.empty()) {
      const auto id=stack.back();stack.pop_back();
      if(!seen.insert(id).second)continue;
      if(seen.size()>256)return false;
      const auto& node=arena->values[id];
      if(node.uniform){if(Unknown(*node.uniform))return false;continue;}
      if(node.vote) {
        if(node.operands.size()<2)return false;
        const auto mask=node.operands[1];
        if(const auto& uniform=arena->values[mask].uniform){if(!FullWarpMask(*uniform))return false;}
        else for(auto member:domain)if(!FullWarpMask(arena->Value(mask,member)))return false;
        continue;
      }
      if(!node.evaluate || node.exchange || node.member_value || !node.coordinates.empty())return false;
      stack.insert(stack.end(),node.operands.begin(),node.operands.end());
    }
    return true;
  };
  try {
    while (!pending.empty() || !waiting.empty()) {
      if(!waiting.empty() && (retried_revision!=exchange_revision ||
          (runtime && shared_revision!=runtime->revision))) {
        if(runtime){runtime->Progress();shared_revision=runtime->revision;}
        retried_revision=exchange_revision;
        pending.insert(pending.end(),std::make_move_iterator(waiting.begin()),std::make_move_iterator(waiting.end()));
        waiting.clear();
        // Progress resumed: waiting groups stay intact again.
        split_collective_waits=false;
      }
      if(pending.empty()) {
        if(retried_revision==exchange_revision && (!runtime || shared_revision==runtime->revision)) {
          // Only a stall that peer progress cannot resolve separates members
          // waiting for collective results (see the collective check below).
          // With a runtime, peers first get one chance to progress.
          if(!split_collective_waits && (!runtime || stalled))split_collective_waits=true;
          else {
            if(!runtime)throw TemplateFallback{};
            stalled=true;co_yield true;
          }
        }
        retried_revision=exchange_revision;
        if(runtime){runtime->Progress();shared_revision=runtime->revision;}
        pending=std::move(waiting);waiting.clear();
      }
      auto group=std::move(pending.front()); pending.pop_front();
      try {
      unsigned quantum=1;
      for (;;) {
        if(runtime && !quantum--) {pending.push_back(std::move(group));break;}
        if(!group.collective_results.empty()) {
          // Result availability follows peer groups' progress, not control
          // flow: members wait together and stay one thread cohort unless the
          // whole schedule has stalled. An intact group rethrows at its first
          // blocked member.
          std::vector<std::size_t> ready,blocked;
          for(auto member:group.members) {
            try {
              for(auto value:group.collective_results)(void)arena->Value(value,member);
              ready.push_back(member);
            } catch(const TemplatePending&) {
              if(!split_collective_waits)throw;
              blocked.push_back(member);
            }
          }
          if(!blocked.empty()) {
            if(!ready.empty()) {
              auto child=group;child.members=std::move(ready);child.collective_results.clear();
              fork_path(child,group,SplitReason::CollectiveWait);
              pending.push_back(std::move(child));++result.template_execution.splits;
              group.members=std::move(blocked);
            }
            throw TemplatePending{};
          }
          group.collective_results.clear();
        }
        if ((group.steps&1023)==0 && poll_) poll_();
        if (group.steps>=template_step_budget || group.pc>=decoded_.size()) throw TemplateFallback{};
        const auto &d=decoded_[group.pc]; const auto &i=*d.instruction;
        std::optional<std::pair<std::string,Integer>> assumed_equality;
        if(i.control=="direct_branch" && i.branch_target && c_.fixed_input.value("assume_trap_assertions",false))
          if(const auto assertion=Assertion(i,*i.branch_target)) {
            const auto& [reg,expected]=*assertion;
            for(auto member:group.members) {
              const auto value=arena->Value(group.registers[reg],member);
              if(!Unknown(value) && value!=Scalar(expected))
                Error(i,members[member].warp,members[member].lane,"assertion assumption contradicts a known value");
            }
            group.state.registers[reg]=expected;group.registers[reg]=arena->Constant(expected);
            group.state.registers[d.predicate]=true;group.registers[d.predicate]=arena->Constant(true);
            assumed_equality={names_[reg],expected};
          }
        if(group.merged_join && group.merged_join->first==group.pc) {
          for(auto reg:group.merged_join->second) {
            group.state.registers[reg]=Scalar{};group.state.origins[reg].reset();
            group.registers[reg]=arena->Constant(Scalar{});group.probe_values.erase(reg);
          }
          group.merged_join.reset();
        }
        if(auto join=group.probe_joins.find(group.pc);join!=group.probe_joins.end()) {
          for(const auto& [reg,value]:join->second)
            if(value.alternative!=group.state.registers[reg])group.probe_values[reg]=value;
          group.probe_joins.erase(join);
        }
        if(i.control=="direct_branch" && d.predicate!=kMissing) {
          if(auto probe=group.probe_values.find(d.predicate);probe!=group.probe_values.end()) {
            auto actual=Truth(group.state.registers[d.predicate]), alternative=Truth(probe->second.alternative);
            if(!actual || !alternative)throw TemplateFallback{};
            const bool a=*actual!=i.predicate_negated,b=*alternative!=i.predicate_negated;
            if(a!=b) {
              if(a || !b || !i.branch_target)throw TemplateFallback{};
              for(auto member:group.members)
                if(c_.BranchOutcome(i.line,members[member].warp,members[member].lane,group.state.branch_visits[group.pc]))
                  throw TemplateFallback{};
              const auto path_index=group.probes.at(probe->second.key);
              const auto record_index=group.path->at(path_index);
              const auto& original=decoded_[probe->second.key.instruction];
              std::set<std::uint32_t> needed;
              for(auto index=group.pc+1;index<std::min(*i.branch_target,group.pc+40);++index)
                needed.insert(decoded_[index].uses.begin(),decoded_[index].uses.end());
              std::optional<std::pair<std::uint32_t,std::set<std::uint32_t>>> fallback;
              State concrete=group.state, before=group.state;
              for(auto member_index:group.members) {
                for(auto reg:needed)concrete.registers[reg]=arena->Value(group.registers[reg],member_index);
                const auto record=arena->Read(record_index,member_index);
                for(auto reg:original.uses)before.registers[reg]=Scalar{};
                for(auto [reg,value]:detail::ExecutionStorage::Bindings(record))before.registers[reg]=value;
                const auto identity=Identity(original,before);
                if(!identity)throw TemplateFallback{};
                const auto member=members[member_index];WarpContext context;
                auto candidate=Fallback(d,concrete,*i.branch_target,*identity,member.warp,member.lane,context);
                if(!candidate || (fallback && fallback!=candidate))throw TemplateFallback{};
                fallback=std::move(candidate);
              }
              auto record=arena->records.at(record_index);
              record.record.mutableMetadata().probe_fallback_key=DynamicKey{fallback->first,group.state.occurrences[fallback->first]};
              group.path->at(path_index)=arena->records.size();arena->records.push_back(std::move(record));
              auto& join=group.probe_joins[*i.branch_target];
              for(auto reg:fallback->second) {
                auto previous=group.probe_values.find(reg);
                join[reg]={probe->second.key,previous!=group.probe_values.end() && previous->second.key==probe->second.key
                  ?previous->second.alternative:group.state.registers[reg]};
              }
            }
          }
        }
        // Partition before changing occurrence counters; children never rejoin.
        if(i.control!="direct_branch" && d.predicate!=kMissing && group.probe_values.contains(d.predicate))
          throw TemplateFallback{};
        bool active=true;
        const ComputeDiamond* opaque=nullptr;
        const MergedArm* merged=nullptr;
        if (d.predicate!=kMissing || i.control=="direct_branch") {
          std::vector<std::size_t> yes,no,unknown;
          for (auto t:group.members) {
            auto truth=d.predicate==kMissing?std::optional<bool>(true):Truth(arena->Value(group.registers[d.predicate],t));
            if(truth && i.predicate_negated)*truth=!*truth;
            if(i.control=="direct_branch") {
              const auto supplied=c_.BranchOutcome(i.line,members[t].warp,members[t].lane,group.state.branch_visits[group.pc]);
              if(supplied && truth && *supplied!=*truth)
                Error(i,members[t].warp,members[t].lane,"known predicate disagrees with fixed-input branch contract");
              if(supplied)truth=supplied;
            }
            if(!truth)unknown.push_back(t);
            else (*truth?yes:no).push_back(t);
          }
          if(!unknown.empty()) {
            if(i.control!="direct_branch")throw TemplateFallback{};
            if(!(opaque=compute_diamond(group.pc)) && (!(merged=MergedArmAt(group.pc)) || (merged->aligned &&
                !merged->uniform && (!whole_warps(unknown) || !warp_uniform(group.registers[d.predicate],unknown)))))
              throw TemplateFallback{};
            for(auto* domain:{&yes,&no})if(!domain->empty()) {
              auto child=group;child.members=std::move(*domain);
              fork_path(child,group,SplitReason::OpaqueDomain);
              pending.push_back(std::move(child));++result.template_execution.splits;
            }
            group.members=std::move(unknown);
          } else if (!yes.empty() && !no.empty()) {
            auto child=group; child.members=std::move(no); group.members=std::move(yes);
            fork_path(child,group,SplitReason::Predicate);
            pending.push_back(std::move(child)); ++result.template_execution.splits;
          } else active=!yes.empty();
        }
        if(active && !supports_template(d))throw TemplateFallback{};
        if(active && i.opcode=="elect" && !c_.raw.at("semantic_assumptions").at("representative_election").get<bool>())
          throw ExecutionError("assumption_required_representative_election",i.line,
              "Election outcomes are unsupported. Enable Representative election (representative_election) to analyze one representative lane.");
        if(active && i.opcode=="clusterlaunchcontrol" && !c_.raw.at("semantic_assumptions").at("clc_always_fails").get<bool>())
          throw ExecutionError("assumption_required_clc_always_fails",i.line,
              "CLC outcomes are unsupported. Enable Failed launch cancellation (clc_always_fails) to analyze failed requests.");
        std::map<std::uint32_t,ProbeValue> probe_updates;
        if(active && i.control!="direct_branch" && !group.probe_values.empty()) {
          State shadow=group.state;
          std::set<DynamicKey> keys;
          for(auto reg:source_registers[group.pc])if(auto dependency=group.probe_values.find(reg);dependency!=group.probe_values.end()) {
            keys.insert(dependency->second.key);shadow.registers[reg]=dependency->second.alternative;
          }
          if(!keys.empty()) {
            if(keys.size()!=1 || i.semantics.category!="compute" ||
                !In(i.opcode,{"mov","cvt","selp","setp","and","or","xor","not"}))throw TemplateFallback{};
            const auto member=members[group.members.front()];WarpContext context;
            Data(d,shadow,member.warp,member.lane,group.state.occurrences[group.pc],Json::object(),context);
            for(auto reg:d.destinations) {
              if(Unknown(shadow.registers[reg]))throw TemplateFallback{};
              probe_updates[reg]={*keys.begin(),shadow.registers[reg]};
            }
          }
        }
        if(active && (i.opcode=="elect" || i.opcode=="shfl")) {
          const auto& reads=source_registers[group.pc];
          State validation=group.state;
          std::uint32_t checked_lanes=0;
          const bool uniform_inputs=std::all_of(reads.begin(),reads.end(),[&](auto reg){return arena->values[group.registers[reg]].uniform.has_value();});
          for(auto t:group.members) {
            const auto member=members[t];
            const auto bit=std::uint32_t(1)<<member.lane;
            if(uniform_inputs && (checked_lanes&bit))continue;
            checked_lanes|=bit;
            for(auto reg:reads) {
              if(i.opcode=="shfl" && reg==d.operands[1].reg &&
                  std::none_of(d.operands.begin()+2,d.operands.end(),[&](const auto& operand){return operand.reg==reg;}))continue;
              validation.registers[reg]=arena->Value(group.registers[reg],t);
            }
            if(i.opcode=="shfl") (void)SelectShuffle(d,validation,member.warp,member.lane);
            else {
              WarpContext context;
              Data(d,validation,member.warp,member.lane,group.state.occurrences[group.pc],Json::object(),context);
            }
          }
        }
        if(active && IsWait(i) && !d.wait_loop) {
          State probe_state=group.state;
          for(auto member:group.members) {
            for(auto reg:source_registers[group.pc])probe_state.registers[reg]=arena->Value(group.registers[reg],member);
            if(!Identity(d,probe_state))throw TemplateFallback{};
          }
        }
        bool shared_handled=false;
        if(runtime && active && (In(i.opcode,{"bar","barrier","cp","tensormap"}) ||
            (i.opcode=="mbarrier" && (!IsWait(i) || d.wait_loop)) ||
            (i.opcode=="tcgen05" && Has(i,"commit")) ||
            (i.opcode=="clusterlaunchcontrol" && i.semantics.operation=="try_cancel_complete_tx") ||
            (In(i.opcode,{"ld","st"}) && (Has(i,"shared") || Has(i,"shared::cta") || Has(i,"shared::cluster"))))) {
          if(!group.shared) {
            group.shared=Group::SharedWork{};
            for(std::size_t slot=0;slot<group.members.size();++slot)group.shared->pending.push_back(slot);
            group.shared->outputs.resize(d.destinations.size(),std::vector<Scalar>(group.members.size()));
          }
          auto& work=*group.shared;
          std::vector<std::size_t> blocked;
          State concrete=group.state;
          for(auto slot:work.pending) {
            const auto member_index=group.members[slot];const auto member=members[member_index];
            try {for(auto reg:d.uses)concrete.registers[reg]=arena->Value(group.registers[reg],member_index);}
            catch(const TemplatePending&){blocked.push_back(slot);continue;}
            auto& context=shared_contexts[member.warp];context.shared=runtime;context.resources=shared_resources.get();
            auto& trace=result.lane_traces[member_index];trace.warp_id=member.warp;trace.lane_id=member.lane;
            trace.instructions=arena->Expand(arena,group.path,member_index,group.path_id);
            bool handled=false;
            try {handled=SharedData(d,concrete,member.warp,member.lane,group.state.occurrences[group.pc],Json::object(),context,trace,d.wait_loop.has_value());}
            catch(const SharedBlocked&){blocked.push_back(slot);continue;}
            work.handled|=handled;
            if(handled)for(std::size_t j=0;j<d.destinations.size();++j)
              work.outputs[j][slot]=concrete.registers[d.destinations[j]];
          }
          work.pending=std::move(blocked);
          if(!work.pending.empty())throw TemplatePending{};
          shared_handled=work.handled;
        }
        ++group.steps; ++result.template_execution.steps;
        result.template_execution.logical_steps+=group.members.size();
        auto occurrence=group.state.occurrences[group.pc]++;
        const auto capture = i.control == "direct_branch" && !opaque && !merged && d.predicate != kMissing
            ? CaptureDecision(i.line, active, group.state.branch_visits[group.pc])
            : std::optional<BranchScopeDecision>{};
        if (i.control=="direct_branch") ++group.state.branch_visits[group.pc];
        if (!active && !capture) {++group.pc; continue;}
        TemplateRecord record;
        if(opaque)for(auto reg:opaque->writes) {
          group.state.registers[reg]=Scalar{};group.state.origins[reg].reset();
          group.registers[reg]=arena->Constant(Scalar{});group.probe_values.erase(reg);
        }
        if (i.control=="direct_branch" && !opaque && !merged && !capture) {
          auto &m=record.record.mutableMetadata(); m.instruction_id=group.pc; m.occurrence=occurrence; m.predicate_value=true;
          if(assumed_equality){m.assumed_equality=assumed_equality;++group.assumed_assertions;}
        } else {
          record.record=Record(d,group.state,occurrence,opaque || merged?std::optional<bool>{}:std::optional<bool>(true));
          if(merged)record.record.mutableMetadata().merged_arm_join=*i.branch_target;
          record.record.mutableMetadata().branch_scope_decision = capture;
          if (capture && active && assumed_equality) {
            record.record.mutableMetadata().assumed_equality = assumed_equality;
            ++group.assumed_assertions;
          }
          record.record.mutableKnownRegisters().clear();
          for (auto reg:d.uses) {
            auto id=group.registers[reg];
            if (!arena->values[id].uniform || !Unknown(*arena->values[id].uniform)) record.bindings.emplace_back(reg,id);
          }
        }
        if(opaque) {
          auto& metadata=record.record.mutableMetadata();
          metadata.opaque_compute_register_reads=opaque->read_names;
          metadata.opaque_compute_register_writes=opaque->write_names;
          metadata.opaque_compute_diamond_indices=opaque->indices;
          metadata.opaque_compute_join_label=i.branch_label;
        }
        if (std::all_of(record.bindings.begin(),record.bindings.end(),[&](auto binding){return arena->values[binding.second].uniform.has_value();})) {
          auto &bindings=record.record.mutableKnownRegisters();
          for (auto [reg,id]:record.bindings) bindings.emplace_back(reg,*arena->values[id].uniform);
          record.bindings.clear();
        }
        if (d.wait_loop) {
          auto& metadata=record.record.mutableMetadata();
          metadata.collapsed_wait_loop=true;
          metadata.normalized_wait_result=true;
          metadata.wait_exit_label=d.wait_loop->first;
          for(auto index=group.pc;index<(d.replay_wait_result?group.pc+1:d.wait_loop->second);++index)
            metadata.collapsed_wait_indices.push_back(index);
        } else if(IsWait(i)) {
          auto& metadata=record.record.mutableMetadata();
          metadata.canonical_wait_probe=true;
          metadata.normalized_wait_result=false;
        }
        group.path->push_back(arena->records.size()); arena->records.push_back(std::move(record));
        result.static_execution_counts[group.pc]+=group.members.size();
        // False decisions are control observations only, never executed data
        // effects. This record exists solely in BranchScopeCapture mode.
        if (!active) { ++group.pc; continue; }
        if(opaque){group.pc=*i.branch_target;continue;}
        if(merged){group.merged_join={*i.branch_target,merged->writes};++group.pc;continue;}
        if(group.shared) {
          if(shared_handled)for(std::size_t j=0;j<d.destinations.size();++j) {
            const auto reg=d.destinations[j];const auto& outputs=group.shared->outputs[j];
            const auto value=arena->SharedResult({group.members.data(),group.members.size()},outputs,
                *member_coordinates,coordinate_value,c_.warp_size,c_.block_dim);
            group.registers[reg]=value;
            group.state.registers[reg]=arena->values[value].uniform.value_or(Scalar{});
          }
          group.shared.reset();
        }
        if (d.wait_loop) {
          Set(d,group.state,{true});
          for(auto reg:d.destinations)group.registers[reg]=arena->Constant(group.state.registers[reg]);
          Origins(d,group.state,occurrence);
          for(auto reg:d.destinations)group.probe_values.erase(reg);
          group.pc=d.replay_wait_result?group.pc+1:d.wait_loop->second;
          continue;
        }
        if(IsWait(i)) {
          const DynamicKey key{group.pc,occurrence};
          group.probes[key]=group.path->size()-1;
          Set(d,group.state,{false});
          for(auto reg:d.destinations) {
            group.registers[reg]=arena->Constant(group.state.registers[reg]);
            group.probe_values[reg]={key,true};
          }
          Origins(d,group.state,occurrence);++group.pc;continue;
        }
        if (i.control=="return" || i.control=="exit") {
          result.template_paths[group.path_id.value].group=static_cast<std::uint32_t>(result.thread_template_groups.size());
          result.group_paths.push_back(group.path_id);
          result.thread_template_groups.emplace_back(group.members.begin(),group.members.end());
          result.template_execution.path_entries+=group.path->size();
          result.template_execution.thread_memberships+=group.members.size();
          std::shared_ptr<const std::vector<std::size_t>> path=std::move(group.path);
          for (auto t:group.members) {
            auto &trace=result.lane_traces[t]; trace.warp_id=members[t].warp; trace.lane_id=members[t].lane;
            trace.assumed_assertions=group.assumed_assertions;
            trace.instructions=arena->Expand(arena,path,t,group.path_id);
            for (std::uint32_t b=0;b<group.state.branch_visits.size();++b)
              if (group.state.branch_visits[b]) trace.branch_visits[b]=group.state.branch_visits[b];
          }
          break;
        }
        if (i.control=="direct_branch") {
          if (!i.branch_target) throw TemplateFallback{};
          group.pc=*i.branch_target; continue;
        }
        if (i.control=="indirect_branch") {
          std::map<std::uint32_t,std::vector<std::size_t>> targets;
          State concrete=group.state;
          for(auto member:group.members) {
            for(auto reg:d.uses)concrete.registers[reg]=arena->Value(group.registers[reg],member);
            const auto index=ScalarInteger(Value(d.operands.front(),concrete));
            if(!index)throw TemplateFallback{};
            if(*index<0 || *index>=i.indirect_targets.size())
              Error(i,members[member].warp,members[member].lane,"indirect branch index is outside its target table");
            targets[i.indirect_targets[index->convert_to<std::size_t>()]].push_back(member);
          }
          for(auto target=std::next(targets.begin());target!=targets.end();++target) {
            auto child=group;child.pc=target->first;child.members=std::move(target->second);
            fork_path(child,group,SplitReason::IndirectBranch);
            pending.push_back(std::move(child));++result.template_execution.splits;
          }
          group.pc=targets.begin()->first;group.members=std::move(targets.begin()->second);
          continue;
        }
        if(i.opcode=="vote") {
          auto& vote=votes[{group.pc,occurrence}];
          if(!vote) {
            vote=std::make_shared<TemplateVote>();vote->members=member_indices;vote->mode=i.modifiers[1];
            vote->error=[owner,pc=group.pc](std::uint32_t actor,const std::string& message) {
              owner->Error(*owner->decoded_[pc].instruction,actor/owner->c_.warp_size,actor%owner->c_.warp_size,message);
            };
          }
          const auto reads=source_registers[group.pc];
          const auto operand=[&](std::size_t index) {
            TemplateValue node;
            for(auto reg:reads)node.operands.push_back(group.registers[reg]);
            node.evaluate=[owner,scratch,reads,pc=group.pc,index](const std::vector<Scalar>& args)->Scalar {
              auto& state=scratch->state;
              for(auto reg:scratch->touched){state.registers[reg]=Scalar{};state.origins[reg].reset();}
              scratch->touched=reads;
              for(std::size_t k=0;k<reads.size();++k)state.registers[reads[k]]=args[k];
              const auto& decoded=owner->decoded_[pc];
              auto text=TrimView(decoded.operands[index].text);
              if(index==1 && text.starts_with('!')) {
                auto value=ScalarInteger(owner->ValueText(text.substr(1),state));
                return value?Scalar(*value==0):Scalar{};
              }
              return owner->Value(decoded.operands[index],state);
            };
            arena->FoldCoordinates(node);
            if(node.uniform)return arena->Constant(*node.uniform);
            const auto id=arena->values.size();arena->values.push_back(std::move(node));return id;
          };
          const auto predicate=operand(1),mask=operand(2);
          vote->predicates->producers.push_back({group.members,predicate});
          vote->masks->producers.push_back({group.members,mask});++exchange_revision;
          TemplateValue node;node.operands={coordinate_value,mask};node.vote=vote;
          const auto id=arena->values.size();arena->values.push_back(std::move(node));
          vote_uses.emplace_back(id,group.members);
          group.collective_results.push_back(id);
          for(auto reg:d.destinations){group.registers[reg]=id;group.state.registers[reg]=Scalar{};}
          Origins(d,group.state,occurrence);++group.pc;continue;
        }
        if(i.opcode=="shfl") {
          const DynamicKey key{group.pc,occurrence};
          auto& exchange=exchanges[key];
          if(!exchange)exchange=std::make_shared<TemplateExchange>();
          const auto source=d.operands[1].reg==kMissing
              ?arena->Constant(Value(d.operands[1],group.state)):group.registers[d.operands[1].reg];
          exchange->producers.push_back({group.members,source});++exchange_revision;
          std::vector<std::uint32_t> selector_registers;
          for(std::size_t k=2;k<d.operands.size();++k)
            for(auto reg:d.operands[k].registers)
              if(std::find(selector_registers.begin(),selector_registers.end(),reg)==selector_registers.end())selector_registers.push_back(reg);
          auto selection_node=[&](bool index) {
            TemplateValue node;
            for(auto reg:selector_registers)node.operands.push_back(group.registers[reg]);
            node.operands.push_back(coordinate_value);
            node.evaluate=[owner,scratch,selector_registers,pc=group.pc,member_indices,index](const std::vector<Scalar>& args)->Scalar {
              auto& state=scratch->state;
              for(auto reg:scratch->touched){state.registers[reg]=Scalar{};state.origins[reg].reset();}
              scratch->touched=selector_registers;
              for(std::size_t k=0;k<selector_registers.size();++k)state.registers[selector_registers[k]]=args[k];
              const auto thread=std::get<Integer>(args.back()).convert_to<std::uint32_t>();
              const auto selected=owner->SelectShuffle(owner->decoded_[pc],state,thread/owner->c_.warp_size,thread%owner->c_.warp_size);
              if(!selected)return {};
              if(!index)return selected->valid;
              const auto actor=thread-thread%owner->c_.warp_size+selected->lane;
              auto found=member_indices->find(actor);
              if(found==member_indices->end())throw TemplateFallback{};
              return Integer(found->second);
            };
            const auto id=arena->values.size();arena->values.push_back(std::move(node));return id;
          };
          const auto selection=selection_node(true);
          exchange_uses.push_back({exchange,group.members,selection});
          TemplateValue completion;completion.operands={selection};completion.exchange=exchange;
          const auto completion_id=arena->values.size();
          group.collective_results.push_back(completion_id);
          arena->values.push_back(std::move(completion));
          if(!d.destination_slots.empty() && d.destination_slots[0]!=kMissing) {
            const auto reg=d.destination_slots[0];
            group.registers[reg]=completion_id;
            group.state.registers[reg]=Scalar{};
          }
          if(d.destination_slots.size()>1 && d.destination_slots[1]!=kMissing) {
            const auto reg=d.destination_slots[1];group.registers[reg]=selection_node(false);
            group.state.registers[reg]=Scalar{};
          }
          Origins(d,group.state,occurrence);++group.pc;continue;
        }
        if(shared_handled)Origins(d,group.state,occurrence);
        if (!shared_handled && !d.destinations.empty()) {
          const auto &reads=source_registers[group.pc];
          const bool uses_thread=i.opcode=="elect";
          const bool uniform_inputs=std::all_of(reads.begin(),reads.end(),[&](auto reg){return arena->values[group.registers[reg]].uniform.has_value();});
          bool uniform=!uses_thread && uniform_inputs;
          if(i.opcode=="clusterlaunchcontrol" && Has(i,"query_cancel") && Has(i,"is_canceled"))uniform=true;
          if (i.opcode=="ld" && !Has(i,"param") && constant_inputs_.empty() && pointer_tables_.empty() &&
              tensor_properties_.empty()) uniform=true;
          if (uniform) {
            WarpContext context; const auto member=members[group.members.front()];
            Data(d,group.state,member.warp,member.lane,occurrence,Json::object(),context);
            for (auto reg:d.destinations) group.registers[reg]=arena->Constant(group.state.registers[reg]);
          } else {
            std::vector<std::size_t> args; std::vector<std::optional<DynamicKey>> origins;
            for (auto reg:reads) {args.push_back(group.registers[reg]); origins.push_back(group.state.origins[reg]);}
            if(uses_thread)args.push_back(coordinate_value);
            for (auto dest:d.destinations) {
              const bool reusable=!Has(i,"cc") && In(i.opcode,{"add","sub","mul","mad","shl","shr","and","or","xor"}) &&
                  std::none_of(args.begin(),args.end(),[&](auto id) {
                    return arena->values[id].uniform && Unknown(*arena->values[id].uniform);
                  });
              TemplateArena::ComputationKey computation{group.pc,dest,args};
              if(reusable)if(const auto cached=arena->FindComputation(computation)) {
                group.registers[dest]=*cached;
                group.state.registers[dest]=arena->values[*cached].uniform.value_or(Scalar{});
                continue;
              }
              TemplateValue node; node.operands=args;
              node.evaluate=[owner,scratch,reads,origins,pc=group.pc,dest,occurrence,uses_thread](const std::vector<Scalar> &args) {
                auto &state=scratch->state;
                for (auto reg:scratch->touched) {
                  state.registers[reg]=Scalar{};
                  state.origins[reg].reset();
                }
                scratch->touched=reads;
                const auto &decoded=owner->decoded_[pc];
                scratch->touched.insert(scratch->touched.end(),decoded.destinations.begin(),decoded.destinations.end());
                for (std::size_t j=0;j<reads.size();++j) {state.registers[reads[j]]=args[j]; state.origins[reads[j]]=origins[j];}
                WarpContext context;
                const auto thread=uses_thread?std::get<Integer>(args.back()).convert_to<std::uint32_t>():0;
                owner->Data(decoded,state,thread/owner->c_.warp_size,thread%owner->c_.warp_size,occurrence,Json::object(),context);
                return state.registers[dest];
              };
              arena->FoldCoordinates(node);
              group.state.registers[dest]=node.uniform.value_or(Scalar{});
              if(reusable)arena->RememberComputation(std::move(computation),arena->values.size());
              group.registers[dest]=arena->values.size(); arena->values.push_back(std::move(node));
            }
          }
          Origins(d,group.state,occurrence);
        }
        for(auto reg:d.destinations)group.probe_values.erase(reg);
        if(i.opcode=="clusterlaunchcontrol" && Has(i,"query_cancel") && Has(i,"is_canceled") && !d.destinations.empty())
          arena->records.at(group.path->back()).record.mutableMetadata().normalized_clc_result=Truth(group.state.registers[d.destinations.front()]);
        for(auto& [reg,value]:probe_updates)
          if(value.alternative!=group.state.registers[reg])group.probe_values[reg]=std::move(value);
        ++group.pc;
      }
      } catch(const TemplatePending&) {waiting.push_back(std::move(group));}
    }
    for(const auto& [value,domain]:vote_uses)for(auto member:domain)(void)arena->Value(value,member);
    for(const auto& use:exchange_uses)for(auto member:use.members) {
      const auto value=arena->Value(use.selection,member);
      if(auto index=std::get_if<Integer>(&value))
        if(!use.exchange->Source(index->convert_to<std::size_t>()))throw TemplateFallback{};
    }
  } catch (const TemplateFallback &) {co_return std::nullopt;}
    catch (const TemplatePending &) {co_return std::nullopt;}
  result.template_execution.value_nodes=arena->values.size();
  result.template_execution.record_nodes=arena->records.size();
  for(const auto& value:arena->values)
    result.template_execution.coordinate_bindings+=value.coordinates.size();
  for(const auto& record:arena->records)
    result.template_execution.record_bindings+=record.bindings.size()+detail::ExecutionStorage::Bindings(record.record).size();
  result.release_value_caches=[weak=std::weak_ptr<TemplateArena>(arena)](std::optional<std::size_t> member) {
    const auto held=weak.lock();
    if(!held)return;
    if(!member)for(auto& cache:held->caches)std::unordered_map<std::size_t,Scalar>{}.swap(cache);
    else if(*member<held->caches.size())std::unordered_map<std::size_t,Scalar>{}.swap(held->caches[*member]);
  };
  result.release_value_caches(std::nullopt);
  if(runtime)for(auto it=runtime->tensor_maps.begin();it!=runtime->tensor_maps.end();++it)
    result.config.tma_descriptors["executed."+it.key()]=it.value();
  result.tensor_property_sites=property_sites_;
  co_return std::move(result);
}

ExecutionProgram Executor::Run() {
  ProfileScope lanes("frontend.execute.lanes");
  ExecutionProgram result;
  result.module = &m_;
  result.function = &f_;
  result.config = c_;
  result.register_names = names_;
  result.static_execution_counts.resize(decoded_.size());
  result.lane_traces.reserve(c_.warps.size() * c_.warp_size);
  for (const auto &warp : c_.warps) {
    WarpContext context;
    std::vector<LaneTask> tasks;
    tasks.reserve(c_.warp_size);
    for (std::uint32_t lane = 0; lane < c_.warp_size; ++lane) {
      const auto tid = warp.warp_id * c_.warp_size + lane;
      if (c_.analysis_thread_range && (tid < c_.analysis_thread_range->first ||
          tid >= c_.analysis_thread_range->second)) continue;
      tasks.push_back(Lane(warp, lane, context));
    }
    std::size_t remaining = tasks.size();
    while (remaining) {
      bool progress = false;
      for (auto &task : tasks) {
        if (task.Done()) continue;
        const auto &dependency = task.Dependency();
        if (dependency && !context.shuffle.contains(
                {dependency->instruction, dependency->occurrence, dependency->lane}))
          continue;
        ProfileScope resume("frontend.execute.lane_resume");
        task.Resume();
        ++resume.edges;
        progress = true;
        if (task.Done()) --remaining;
      }
      if (poll_) poll_();
      if (!progress)
        throw std::runtime_error("shuffle source has no matching dynamic occurrence");
    }
    ProfileScope collect("frontend.execute.trace_collection");
    for (auto &task : tasks) {
      auto trace = task.TakeResult();
      for (const auto &record : trace.instructions)
        ++result.static_execution_counts[record.instruction_id()];
      collect.nodes += trace.instructions.size();
      result.lane_traces.push_back(std::move(trace));
    }
    lanes.nodes += collect.nodes;
    lanes.edges += tasks.size();
  }
  result.record_compression = record_pool_.Statistics();
  result.tensor_property_sites = property_sites_;
  return result;
}
std::optional<std::vector<ExecutionProgram>> Executor::RunTemplateCluster(const Module& module,const Function& function,
    const std::vector<FrontendConfig>& configs,const std::function<void()>& poll,SpecializationPurpose purpose) {
  SharedExecution runtime;
  std::vector<std::shared_ptr<Executor>> executors;
  std::vector<TemplateTask> tasks;
  std::vector<ExecutionProgram> results(configs.size());
  for(std::size_t rank=0;rank<configs.size();++rank) {
    if(configs[rank].fixed_input.value("cluster_ctarank",std::uint32_t(rank))!=rank)
      throw ExecutionError("shared_control_cluster",0,"cooperative configurations must be in CTA rank order");
    executors.push_back(std::make_shared<Executor>(module,function,configs[rank],poll,purpose));
    tasks.push_back(executors.back()->RunTemplateTask(&runtime));
  }
  std::size_t remaining=tasks.size();
  while(remaining) {
    const auto before=std::pair{runtime.revision,remaining};
    for(std::size_t rank=0;rank<tasks.size();++rank)if(!tasks[rank].Done()) {
      tasks[rank].Resume();
      if(tasks[rank].Done()) {
        auto result=tasks[rank].TakeResult();
        if(!result)return {};
        results[rank]=std::move(*result);--remaining;
      }
    }
    runtime.Progress();if(poll)poll();
    if(remaining && before==std::pair{runtime.revision,remaining})return {};
  }
  for(auto &result:results)runtime.Export(result);
  return results;
}

std::vector<ExecutionProgram> Executor::RunCluster(const Module &module, const Function &function,
    const std::vector<FrontendConfig> &configs, const std::function<void()> &poll,
    SpecializationPurpose purpose) {
  if(auto compact=RunTemplateCluster(module,function,configs,poll,purpose))return std::move(*compact);
  SharedExecution runtime;
  std::vector<ExecutionProgram> results(configs.size());
  std::vector<std::unique_ptr<Executor>> executors;
  std::vector<AddressTable> addresses(configs.size());
  std::vector<BarrierTable> barriers(configs.size());
  std::vector<std::unique_ptr<ExecutorResources>> adapters;
  std::map<std::pair<std::size_t,std::uint32_t>,WarpContext> contexts;
  struct Task { std::size_t rank; WarpContext *context; LaneTask task; };
  std::vector<Task> tasks;
  for (std::size_t rank=0;rank<configs.size();++rank) {
    const auto &config=configs[rank];
    if (config.fixed_input.value("cluster_ctarank",std::uint32_t(rank))!=rank)
      throw ExecutionError("shared_control_cluster",0,"cooperative configurations must be in CTA rank order");
    executors.push_back(std::make_unique<Executor>(module,function,config,poll,purpose));
    auto &executor=*executors.back(); auto &result=results[rank];
    result.module=&module; result.function=&function; result.config=config;
    result.register_names=executor.names_; result.static_execution_counts.resize(function.instructions.size());
    const auto coordinates=config.fixed_input.value("cta_coordinate",Coordinates{0,0,0});
    adapters.push_back(std::make_unique<ExecutorResources>(result,coordinates,addresses[rank],barriers[rank]));
    for (const auto &warp : config.warps) {
      auto &context=contexts[{rank,warp.warp_id}]; context.shared=&runtime; context.resources=adapters.back().get();
      for (std::uint32_t lane=0;lane<config.warp_size;++lane)
        tasks.push_back({rank,&context,executor.Lane(warp,lane,context)});
    }
  }
  std::size_t remaining=tasks.size();
  auto progress=[&] {
    std::size_t shuffles=0; for (const auto &[key,context] : contexts) { (void)key; shuffles+=context.shuffle.size(); }
    return std::tuple{runtime.revision,remaining,shuffles};
  };
  while (remaining) {
    const auto before=progress();
    for (auto &entry : tasks) {
      auto &task=entry.task; if (task.Done()) continue;
      const auto &dep=task.Dependency();
      if (dep && !dep->shared && !entry.context->shuffle.contains({dep->instruction,dep->occurrence,dep->lane})) continue;
      task.Resume(); if (task.Done()) --remaining;
    }
    runtime.Progress(); if (poll) poll();
    if (remaining && before==progress()) {
      std::uint32_t line=0;
      for (const auto &entry : tasks) if (!entry.task.Done() && entry.task.Dependency()) {
        line=function.instructions[entry.task.Dependency()->instruction].line; break;
      }
      throw ExecutionError("shared_control_blocked",line,"shared-value propagation has no enabled publication or collective");
    }
  }
  for (auto &entry : tasks) {
    auto trace=entry.task.TakeResult(); auto &result=results[entry.rank];
    for (const auto &e : trace.instructions) ++result.static_execution_counts[e.instruction_id()];
    result.lane_traces.push_back(std::move(trace));
  }
  for (std::size_t rank = 0; rank < results.size(); ++rank) {
    auto &result = results[rank];
    runtime.Export(result);
    result.record_compression = executors[rank]->record_pool_.Statistics();
    for (auto it=runtime.tensor_maps.begin();it!=runtime.tensor_maps.end();++it)
      result.config.tma_descriptors["executed."+it.key()]=it.value();
  }
  return results;
}
} // namespace
std::optional<Integer> ScalarInteger(const Scalar &value) {
  if (auto n = std::get_if<Integer>(&value))
    return *n;
  if (auto b = std::get_if<bool>(&value))
    return Integer(*b);
  return {};
}
std::optional<PointerBits> PointerFragment(const Scalar &value,unsigned source,unsigned width) {
  if (!width || source+width>64) return {};
  if (const auto *bits=std::get_if<PointerBits>(&value)) {
    if (bits->destination_bit || source+width>bits->width) return {};
    auto result=*bits; result.source_bit+=source; result.width=width; return result;
  }
  std::shared_ptr<const PointerValue> pointer;
  if (auto p=std::get_if<AllocationPointer>(&value)) pointer=std::make_shared<PointerValue>(*p);
  else if (auto p=std::get_if<ParameterPointer>(&value)) pointer=std::make_shared<PointerValue>(*p);
  else if (auto p=std::get_if<ClusterSharedPointer>(&value)) pointer=std::make_shared<PointerValue>(*p);
  if (!pointer) return {};
  return PointerBits{std::move(pointer),source,width,0};
}
Scalar RestorePointer(const PointerBits &bits) {
  if (bits.source_bit==0 && bits.width==64 && bits.destination_bit==0)
    return std::visit([](const auto &pointer)->Scalar{return pointer;},*bits.pointer);
  return bits;
}
std::optional<Integer> ParseInteger(std::string_view text) {
  text = TrimView(text);
  if (text.empty())
    return {};
  const bool unsigned_literal=text.back()=='U';
  if(unsigned_literal)text.remove_suffix(1);
  if(text.empty())return {};
  bool negative = false;
  if (text.front() == '+' || text.front() == '-') {
    negative = text.front() == '-';
    text.remove_prefix(1);
  }
  if (text.empty())
    return {};
  unsigned base = 10;
  if (text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
    base = 16;
    text.remove_prefix(2);
  } else if (text.size()>2 && text[0]=='0' && (text[1]=='b' || text[1]=='B')) {
    base=2;
    text.remove_prefix(2);
  } else if(text.size()>1 && text[0]=='0') {
    base=8;
  }
  if (text.empty())
    return {};
  Integer value = 0;
  for (unsigned char ch : text) {
    unsigned digit;
    if (ch >= '0' && ch <= '9')
      digit = ch - '0';
    else if (ch >= 'a' && ch <= 'f')
      digit = ch - 'a' + 10;
    else if (ch >= 'A' && ch <= 'F')
      digit = ch - 'A' + 10;
    else
      return {};
    if (digit >= base)
      return {};
    value *= base;
    value += digit;
  }
  // PTX ISA 4.5.1: U selects unsigned 64-bit literal arithmetic.
  if(negative && (unsigned_literal || value>std::numeric_limits<std::int64_t>::max()))
    return Integer((-value)&((Integer(1)<<64)-1));
  return negative ? Integer(-value) : value;
}
std::optional<std::pair<std::string, Integer>>
SplitAddress(std::string_view text) {
  text = TrimView(text);
  if (text.empty())
    return {};
  if (auto number = ParseInteger(text))
    return {{std::string(text), 0}};
  std::size_t split = std::string_view::npos;
  for (std::size_t i = 1; i < text.size(); ++i)
    if (text[i] == '+' || text[i] == '-') {
      split = i;
      break;
    }
  if (split == std::string_view::npos) {
    if (text.find_first_of(" \t,[]{}") != std::string_view::npos)
      return {};
    return {{std::string(text), 0}};
  }
  auto base = TrimView(text.substr(0, split));
  std::string offset;
  for (unsigned char ch : text.substr(split))
    if (!std::isspace(ch))
      offset.push_back(ch);
  if (offset.starts_with("+-")) offset.erase(0,1);
  auto n = ParseInteger(offset);
  if (base.empty() || !n)
    return {};
  return {{std::string(base), *n}};
}
std::vector<std::string> RegistersIn(std::string_view text) {
  std::vector<std::string> result;
  text=TrimView(text);
  if(const auto separator=text.find('|');separator!=text.npos) {
    auto first=RegistersIn(text.substr(0,separator));
    auto second=RegistersIn(text.substr(separator+1));
    first.insert(first.end(),second.begin(),second.end());
    return first;
  }
  if (text.size()>=2 && text.front()=='{' && text.back()=='}') {
    text.remove_prefix(1); text.remove_suffix(1);
    for (;;) {
      const auto comma=text.find(',');
      auto part=RegistersIn(text.substr(0,comma)); result.insert(result.end(),part.begin(),part.end());
      if (comma==text.npos) break;
      text.remove_prefix(comma+1);
    }
    return result;
  }
  auto head = [](unsigned char c) {
    return std::isalpha(c) || c == '_' || c == '$';
  };
  auto ident = [&](unsigned char c) { return head(c) || std::isdigit(c); };
  for (std::size_t i = 0; i < text.size(); ++i)
    if (text[i] == '%' && i + 1 < text.size() && head(text[i + 1])) {
      auto begin = i++;
      while (i < text.size() && ident(text[i]))
        ++i;
      result.emplace_back(text.substr(begin, i - begin));
      if (i)
        --i;
    }
  auto token = TrimView(text);
  if (result.empty() && !token.empty() && token != "_" && head(token.front()) &&
      std::all_of(token.begin(), token.end(), ident))
    result.emplace_back(token);
  return result;
}
namespace {
ExecutionProgram SpecializeFunctionOnce(const Module &, const Function &, const FrontendConfig &,
                                        std::function<void()>, SpecializationPurpose);
// Representative control flow is a last resort: it applies only after ordinary execution failed on an
// unresolved branch, so it never changes a kernel the fixed input already determines.
template <class Run>
auto WithDataBranches(const FrontendConfig &c, bool defer_shared_control, Run run) {
  auto undetermined = [&](const ExecutionError &error) {
    // Shared-memory control is first retried by the caller as a whole cluster
    // (SpecializeCluster), which applies these passes itself if that also fails.
    return !exact_control_pass && !(defer_shared_control && error.shared_control) &&
           (error.code == "branch_not_static" || error.code == "missing_parameter_value");
  };
  auto pass = [&](bool data) {
    exact_control_pass = true;
    data_branch_pass = data;
    try { auto result = run(); exact_control_pass = data_branch_pass = false; return result; }
    catch (...) { exact_control_pass = data_branch_pass = false; throw; }
  };
  try { return run(); }
  catch (const ExecutionError &error) { if (!undetermined(error)) throw; }
  try { return pass(false); }
  catch (const ExecutionError &error) {
    if (!DataBranchesAssumed(c) ||
        (error.code != "branch_not_static" && error.code != "missing_parameter_value"))
      throw;
  }
  return pass(true);
}
} // namespace
ExecutionProgram SpecializeFunction(const Module &m, const Function &f,
                                    const FrontendConfig &c,
                                    std::function<void()> poll,
                                    SpecializationPurpose purpose) {
  return WithDataBranches(c, true, [&] { return SpecializeFunctionOnce(m, f, c, poll, purpose); });
}
namespace {
ExecutionProgram SpecializeFunctionOnce(const Module &m, const Function &f,
                                        const FrontendConfig &c,
                                        std::function<void()> poll,
                                        SpecializationPurpose purpose) {
  try {
    auto executor = std::make_shared<Executor>(m, f, c, poll, purpose);
    auto compressed = executor->RunTemplates();
    auto result = compressed ? std::move(*compressed) : executor->Run();
    if (purpose == SpecializationPurpose::Verification) RecoverTensorMaps(result);
    return result;
  }
  catch (const ExecutionError &error) {
    if (!error.shared_control) throw;
    const auto cluster=c.cluster_dim.value_or(Coordinates{1,1,1});
    if (cluster[0]*cluster[1]*cluster[2]!=1) throw;
    auto programs=Executor::RunCluster(m,f,{c},poll,purpose);
    if (purpose == SpecializationPurpose::Verification) RecoverTensorMaps(programs.front());
    return std::move(programs.front());
  }
}
} // namespace
std::vector<ExecutionProgram> SpecializeCluster(const Module &m, const Function &f,
    const std::vector<FrontendConfig> &configs, std::function<void()> poll,
    SpecializationPurpose purpose) {
  auto programs=WithDataBranches(configs.front(),false,[&]{return Executor::RunCluster(m,f,configs,poll,purpose);});
  if (purpose == SpecializationPurpose::Verification)
    for (auto &program : programs) RecoverTensorMaps(program);
  return programs;
}
Json ScalarJson(const Scalar &value) {
  if (const auto *bounded=std::get_if<BoundedInteger>(&value))
    return {{"kind","bounded_integer"},{"lower",ScalarJson(Scalar((*bounded)->lower))},
            {"upper",ScalarJson(Scalar((*bounded)->upper))},{"identity",(*bounded)->identity}};
  if (const auto *pointer=std::get_if<BoundedAllocationPointer>(&value))
    return {{"kind","bounded_allocation_pointer"},{"resource",(*pointer)->resource},
            {"lower",ScalarJson(Scalar((*pointer)->lower))},{"upper",ScalarJson(Scalar((*pointer)->upper))}};
  if (auto fragment=std::get_if<DescriptorFragment>(&value))
    return {{"kind","descriptor_fragment"},{"descriptor",fragment->descriptor},
            {"offset",fragment->offset},{"bytes",fragment->bytes}};
  if (const auto *bits=std::get_if<PointerBits>(&value))
    return {{"kind","pointer_bits"},{"pointer",std::visit([](const auto &p){return ScalarJson(Scalar(p));},*bits->pointer)},
        {"source_bit",bits->source_bit},{"width",bits->width},{"destination_bit",bits->destination_bit}};
  if (auto n = std::get_if<Integer>(&value)) {
    if (*n >= INT64_MIN && *n <= INT64_MAX)
      return n->convert_to<std::int64_t>();
    if (*n >= 0 && *n <= UINT64_MAX)
      return n->convert_to<std::uint64_t>();
    return {{"integer", n->convert_to<std::string>()}};
  }
  if (auto b = std::get_if<bool>(&value))
    return *b;
  if (auto p = std::get_if<ParameterPointer>(&value))
    return {{"kind", "parameter_pointer"},
            {"region", p->region},
            {"offset", ScalarJson(Scalar(p->offset))}};
  if (auto p = std::get_if<AllocationPointer>(&value))
    return {
        {"kind", "allocation_pointer"},
        {"resource", p->resource},
        {"offset", ScalarJson(Scalar(p->offset))},
        {"extent_bytes", p->extent_bytes ? Json(*p->extent_bytes) : Json()}};
  if (auto p = std::get_if<ClusterSharedPointer>(&value))
    return {{"kind", "cluster_shared_pointer"},
            {"local_address", ScalarJson(Scalar(p->local_address))},
            {"target_rank", p->target_rank}};
  return nullptr;
}
Json ExecutionRecordJson(const ExecutionProgram &program,
                         const ExecutedInstruction &r) {
  Json known = Json::object(), origins = Json::object();
  for (const auto &[id, v] : detail::ExecutionStorage::Bindings(r))
    known[program.register_names[id]] = ScalarJson(v);
  for (const auto &[id, v] : r.register_origins())
    origins[program.register_names[id]] = {v.instruction, v.occurrence};
  return {{"instruction", r.instruction_id()},
          {"occurrence", r.occurrence()},
          {"predicate_value",
           r.predicate_value() ? Json(*r.predicate_value()) : Json()},
          {"known_registers", known},
          {"register_origins", origins},
          {"collapsed_wait_loop", r.collapsed_wait_loop()},
          {"normalized_wait_result", r.normalized_wait_result() ? Json(*r.normalized_wait_result()) : Json()},
          {"normalized_clc_result", r.normalized_clc_result() ? Json(*r.normalized_clc_result()) : Json()},
          {"canonical_wait_probe", r.canonical_wait_probe()},
          {"wait_exit_label",
           r.wait_exit_label() ? Json(*r.wait_exit_label()) : Json()},
          {"schedule_seed_outcome",
           r.schedule_seed_outcome() ? Json(*r.schedule_seed_outcome()) : Json()},
          {"probe_fallback_key", r.probe_fallback_key()
                                     ? Json{r.probe_fallback_key()->instruction,
                                            r.probe_fallback_key()->occurrence}
                                     : Json()},
          {"opaque_compute_diamond_indices", r.opaque_compute_diamond_indices()},
          {"opaque_compute_join_label", r.opaque_compute_join_label()
                                            ? Json(*r.opaque_compute_join_label())
                                            : Json()},
          {"merged_arm_join", r.merged_arm_join() ? Json(*r.merged_arm_join()) : Json()},
          {"assumed_equality", r.assumed_equality()
              ? Json{r.assumed_equality()->first, ScalarJson(Scalar(r.assumed_equality()->second))} : Json()},
          {"opaque_compute_register_reads", r.opaque_compute_register_reads()},
          {"opaque_compute_register_writes", r.opaque_compute_register_writes()},
          {"collapsed_wait_indices", r.collapsed_wait_indices()}};
}
Json ExecutionJson(const ExecutionProgram &program) {
  Json lanes = Json::array();
  for (const auto &lane : program.lane_traces) {
    Json records = Json::array();
    for (std::size_t index=0;index<lane.instructions.size();++index)
      records.push_back(ExecutionRecordJson(program, detail::ExecutionStorage::Read(lane.instructions, index)));
    Json branches = Json::object();
    for (auto [k, v] : lane.branch_visits)
      branches[std::to_string(k)] = v;
    lanes.push_back({{"warp_id", lane.warp_id},
                     {"lane_id", lane.lane_id},
                     {"instructions", records},
                     {"branch_visits", branches},
                     {"loop_backedges", lane.loop_backedges}});
  }
  return {{"lane_traces", lanes},
          {"static_execution_counts", program.static_execution_counts}};
}
} // namespace sentryward::frontend::ptx
