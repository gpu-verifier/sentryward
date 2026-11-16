#pragma once
#include "importer.hpp"

namespace rg {

// Range premise (assumption floating_values_in_range, docs/assumptions.md): a finite ideal value of
// an IEEE type lies within that type's finite range. Hence, beside a finite operand, the type's most
// negative finite value never decides a Maximum (nor its largest finite value a Minimum); such
// operands are dropped. Without the premise nothing is dropped. O(operands).
inline auto Importer::in_range(std::vector<Id> x, const std::string &type, bool maximum) -> std::vector<Id> {
  const auto format = float_format(type);
  if (!range_premise || !format) return x;
  const Q largest = largest_finite(*format);
  const Id bound = a.constant(maximum ? -largest : largest);
  if (std::none_of(x.begin(), x.end(), [&](Id o) { return o != bound && a.nodes[o].facts.finite; })) return x;
  if (const auto dropped = std::erase(x, bound)) a.stats["range_premise_bounds_dropped"] += dropped;
  return x;
}

inline auto Importer::float_format(const std::string &t) -> std::optional<FloatFormat> {
  const auto ieee = sentryward::equivalence::ieeeFormat(t);
  if (!ieee) return std::nullopt;
  const unsigned max_exponent = (1u << (ieee->width - 2 - ieee->fraction)) - 1;
  return FloatFormat{ieee->fraction + 1, max_exponent, max_exponent - 1};
}

// Whether q is a finite value of format f: |q| <= MAX and q = odd 2^z with at most p significant bits
// and z at least the subnormal spacing exponent emin + 1 - p. O(digits).
inline auto Importer::representable(const Q &q, const FloatFormat &f) -> bool {
  if (q == 0) return true;
  const Z n = boost::multiprecision::abs(q.numerator()), d = q.denominator();
  if ((d & (d - 1)) != 0 || Q(n, d) > largest_finite(f)) return false;
  const unsigned zeros = unsigned(boost::multiprecision::lsb(n));
  const long exponent = long(zeros) - long(boost::multiprecision::msb(d));
  return boost::multiprecision::msb(Z(n >> zeros)) + 1 <= f.precision &&
         exponent >= 1 - long(f.min_normal_shift) - long(f.precision);
}

// The exact sum of two sets: NaN only from opposite infinities, which also widen the bound.
inline auto Importer::sum(const ValueSet &x, const ValueSet &y) -> ValueSet {
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
inline auto Importer::product(const ValueSet &x, const ValueSet &y) -> ValueSet {
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
inline auto Importer::flushed(ValueSet s, const FloatFormat &f) -> ValueSet {
  const Q normal(1, pow2(f.min_normal_shift));
  if (!s.lo.infinity && s.lo.value > 0 && s.lo.value < normal) s.lo.value = Q(0);
  if (!s.hi.infinity && s.hi.value < 0 && s.hi.value > -normal) s.hi.value = Q(0);
  return s;
}

// The value of format f at or below (`up`: at or above) finite q, on f's grid without range
// limits: q rounded down (up) to a multiple of 2^(e+1-p), e the exponent of q but at least emin. O(digits).
inline auto Importer::on_grid(const Q &q, const FloatFormat &f, bool up) -> Q {
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
inline auto Importer::rounded(const ValueSet &s, const FloatFormat &f, uint8_t modes, bool flush) -> ValueSet {
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
inline auto Importer::extremum(std::vector<ValueSet> sets, bool maximum, bool propagate) -> ValueSet {
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

inline auto Importer::bitcast_chain(std::string type, const std::string &conversion) -> std::optional<BitcastChain> {
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
inline auto Importer::parse_constant(const RawNode &n, const std::string &conversion) -> ValueSet {
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

inline auto Importer::converted(const ValueSet &s, const std::string &conversion) -> ValueSet {
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

// The operands of raw node `id` whose sets node_set reads (an add's contraction reads its mul's
// operands through the mul's own entry). O(operands).
inline auto Importer::set_operands(Id id, std::vector<Id> &out) const -> void {
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

// The hardware value set of raw edge e, in import scope `scope` (or in every scope when null): an
// imported infinity is that infinity, a Select that this scope decided is its chosen arm, a Negate
// negates its operand's set in this scope, an instruction image imported as a constant is evaluated
// from its operands in this scope; any other value has its scope-independent set.
inline auto Importer::edge_set(const Edge &e, const Env *scope, unsigned depth) -> ValueSet {
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
inline auto Importer::node_set(const RawNode &n, const Env *scope, unsigned depth) -> ValueSet {
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

// The data operands of raw node `raw` whose keys are tainted (a Select's arms, a Fold's element in
// scope `body`); empty when none is. O(operands).
inline auto Importer::tainted_operands(Id raw, const Env &env, const Env &body) -> std::vector<uint8_t> {
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

// Records the taint of import key k of raw node `raw`, imported as `result` from operands x of
// which `tainted` are tainted, once hardware_value let the import proceed. A comparison or other
// predicate carries none: it is exact on the infinity. A maximum (minimum) carries it from an
// operand that may decide it, that is anything but -inf (+inf), and when its result is an
// infinity. A floating instruction image whose value set hardware_value found finite carries none.
// Every other node carries it on: a Select arm, a negation or absolute value, an integer or bit
// operation (a Bitcast is an edge conversion of its operand), an empty Fold. O(operands).
inline auto Importer::carry_hardware_infinity(Id raw, const std::vector<Id> &k, const std::vector<Id> &x,
                             const std::vector<uint8_t> &tainted, Id result) -> void {
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

inline auto Importer::render_set(const ValueSet &s) -> std::string {
  const auto end = [](const Extended &e) {
    return e.infinity ? std::string(e.infinity < 0 ? "-inf" : "inf") : str(e.value);
  };
  return "[" + end(s.lo) + ", " + end(s.hi) + "]" + (s.nan ? " or NaN" : "");
}

// An instruction image with a singleton-infinity value set imports that infinity.
// Propagate taint only through operations that determine its hardware result;
// ambiguous arithmetic, Sum Folds and NaN-deciding extrema decline the family.
// Literal infinities retain extended-real semantics. Conversions use
// carry_hardware_infinity; ex2.approx(-Inf) is zero. Probe instruction images
// only when an operand is constant, infinite or already tainted.
inline auto Importer::hardware_value(Id raw, const std::vector<Id> &x, const std::vector<uint8_t> &tainted,
                                 const Env &env) -> std::optional<Id> {
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

} // namespace rg
