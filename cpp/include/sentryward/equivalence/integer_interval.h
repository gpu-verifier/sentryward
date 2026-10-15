#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

namespace sentryward::equivalence::integer_interval {

struct Type {
  bool is_signed = false;
  std::uint8_t bits = 64;
  bool operator==(const Type&) const = default;
};

// Fixed-size, inclusive bounds. The int64 representation is intentional:
// unsigned values above INT64_MAX are Unknown, not wrapped or approximated.
struct Interval {
  Type type;
  std::int64_t lower = 0;
  std::int64_t upper = 0;
  bool operator==(const Interval&) const = default;
};
using Result = std::optional<Interval>; // Empty means Unknown.
enum class Operation { Add, Subtract, Multiply, BitAnd, BitXor };
enum class Comparison { Less, LessEqual, Greater, GreaterEqual, Equal, NotEqual };
enum class Truth : std::uint8_t { Unknown = 0, True = 1, False = 2 };

std::optional<Type> parseType(std::string_view); // [exact:]u/s8/16/32/64 only.
Result interval(Type, std::int64_t lower, std::int64_t upper);
// The entire declared integer domain, only when it fits this representation.
// In particular u64 is Unknown: its upper half is not representable in int64.
Result typeRange(Type);
Result literal(Type, std::string_view decimal);
Result literal(std::string_view type, std::string_view decimal);

// The caller must supply an extent for this PARTICULAR external lexical level
// from a validated output family. This helper does not infer Fold binder bounds,
// traverse ancestors, or assume every use of an Index has the same domain.
Result externalIndex(Type, std::uint64_t extent);
Result externalIndex(std::string_view type, std::uint64_t extent);

Result apply(Operation, const Interval&, const Interval&);
Result divideByPositiveLiteral(const Interval&, std::int64_t divisor);
Result remainderByPositiveLiteral(const Interval&, std::int64_t divisor);
Result shiftRightByLiteral(const Interval&, std::int64_t shift);
Truth compare(Comparison, const Interval&, const Interval&);
Truth compare(Comparison, const Result&, const Result&);

// All operations are constant-sized (at most four checked endpoint products).
// A possible overflow, declared-type wrap, malformed interval/type, unsupported
// unsigned magnitude, nonpositive divisor, or missing bound returns Unknown.

} // namespace sentryward::equivalence::integer_interval
