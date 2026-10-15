#pragma once

#include <boost/multiprecision/cpp_int.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace sentryward::equivalence::polynomial_certificate {

// Optional caller-owned aggregate allowance. Invoked at existing primitive
// charge points BEFORE the charged work; false permanently exhausts Builder.
// The callback and immutable proof epoch must outlive Builder. No budget reset.
using WorkCharge = std::function<bool(std::size_t)>;

using Atom = std::size_t;
using Integer = boost::multiprecision::cpp_int;

struct Rational {
  Integer numerator = 0;
  Integer denominator = 1;
  bool operator==(const Rational&) const = default;
};

struct Monomial {
  // Sorted distinct atom IDs; strictly positive exponents.
  std::vector<std::pair<Atom,std::uint32_t>> powers;
  bool operator==(const Monomial&) const = default;
};

struct Term {
  Monomial monomial;
  Rational coefficient;
  bool operator==(const Term&) const = default;
};

class Builder;
class CanonicalPolynomial {
 public:
  CanonicalPolynomial() = default; // Exact zero.
  const std::vector<Term>& terms() const { return terms_; }
  bool operator==(const CanonicalPolynomial&) const = default;
  bool operator<(const CanonicalPolynomial&) const; // Complete-key ordered lookup.
 private:
  friend class Builder;
  explicit CanonicalPolynomial(std::vector<Term> terms) : terms_(std::move(terms)) {}
  std::vector<Term> terms_;
};

struct Limits {
  std::size_t max_terms = 64;
  std::uint32_t max_degree = 64;
  std::size_t max_coefficient_bits = 4096;
  std::size_t max_intermediate_bits = 8192;
  std::size_t max_decimal_chars = 1152;
  std::size_t max_work = 1000000;
};

enum class Operation { Add, Subtract, Negate, Multiply };
using Result = std::optional<CanonicalPolynomial>; // Empty means Unknown, never a proof.

// The caller admits only finite ideal-real arithmetic and stable atom identities.
// Fold, Select, opaque math, and nonliteral division must remain caller-supplied
// atoms. These certificates never construct distributed expression trees or
// cancel an arbitrary denominator. Exact canonical keys, not hashes, prove equal.
class Builder {
 public:
  explicit Builder(Limits = {}, WorkCharge = {});
  Result literal(std::string_view decimal);
  Result atom(Atom);
  Result apply(Operation, std::span<const CanonicalPolynomial* const> operands);
  Result divideByLiteral(const CanonicalPolynomial&, std::string_view divisor);
  std::size_t work() const { return work_; }
  bool exhausted() const { return exhausted_; }
 private:
  bool charge(std::size_t amount);
  bool admissible(const CanonicalPolynomial&);
  bool coefficientAdmissible(const Rational&) const;
  bool bounded(const Integer&, std::size_t bits) const;
  std::optional<Integer> multiply(const Integer&, const Integer&);
  std::optional<Integer> gcd(Integer, Integer);
  std::optional<Rational> normalize(Integer numerator, Integer denominator);
  std::optional<Rational> parse(std::string_view);
  std::optional<Rational> add(const Rational&, const Rational&, bool subtract);
  std::optional<Rational> product(const Rational&, const Rational&);
  std::optional<Monomial> product(const Monomial&, const Monomial&);
  Result combine(const CanonicalPolynomial&, const CanonicalPolynomial&, bool subtract);
  Result negate(const CanonicalPolynomial&);
  Result product(const CanonicalPolynomial&, const CanonicalPolynomial&);
  Limits limits_;
  std::size_t work_ = 0;
  bool exhausted_ = false;
  WorkCharge work_charge_;
};

} // namespace sentryward::equivalence::polynomial_certificate
