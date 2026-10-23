#pragma once
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/rational.hpp> // bad_rational, rational_detail::is_compatible_integer, integer::gcd
#include <boost/throw_exception.hpp>
#include <climits>
#include <concepts>
#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>
namespace rg {
using Z = boost::multiprecision::cpp_int;
// Single-limb arithmetic shared by Rational and the Z helpers. A canonical
// cpp_int is a sign plus limbs with no high zero limb and no negative zero, so
// a value has one limb exactly when its magnitude fits one. A product of two
// limbs fits a double limb; only a sum of two such products can overflow.
namespace rational_detail {
using Limb = boost::multiprecision::limb_type;
using Wide = boost::multiprecision::double_limb_type;
using SignedWide = boost::multiprecision::signed_double_limb_type;
static_assert(sizeof(Wide) == 2 * sizeof(Limb) && sizeof(SignedWide) == 2 * sizeof(Limb));
constexpr unsigned limb_bits = sizeof(Limb) * CHAR_BIT;
inline bool limb(const Z &z, Limb &m, bool &neg) noexcept {
  const auto &b = z.backend();
  if (b.size() != 1) return false;
  m = b.limbs()[0];
  neg = b.sign();
  return true;
}
// A positive Z (a denominator).
inline bool limb(const Z &z, Limb &m) noexcept {
  const auto &b = z.backend();
  if (b.size() != 1) return false;
  m = b.limbs()[0];
  return true;
}
// The canonical value: a destination that already has one limb takes the limb
// and sign directly (sign() keeps zero unsigned); otherwise Boost's own limb or
// double-limb assignment sizes it.
inline void store(Z &z, Limb m, bool neg) noexcept {
  auto &b = z.backend();
  if (b.size() == 1) {
    b.limbs()[0] = m;
    b.sign(neg);
    return;
  }
  b = m;
  if (neg) b.negate();
}
inline void store(Z &z, Wide m, bool neg) noexcept {
  if (Limb(m >> limb_bits) == 0) return store(z, Limb(m), neg);
  z.backend() = m;
  if (neg) z.backend().negate();
}
inline Z make(Wide m, bool neg) noexcept {
  Z z;
  store(z, m, neg);
  return z;
}
// One-limb values are rebuilt from the limb instead of cpp_int's resize and
// memcpy; wider values are copied or moved by cpp_int itself.
inline Z narrow(Limb m, bool neg) noexcept {
  Z z(m);
  if (neg) z.backend().negate();
  return z;
}
inline Z copied(const Z &z) {
  const auto &b = z.backend();
  return b.size() == 1 ? narrow(b.limbs()[0], b.sign()) : Z(z);
}
inline Z moved(Z &z) noexcept {
  const auto &b = z.backend();
  return b.size() == 1 ? narrow(b.limbs()[0], b.sign()) : Z(std::move(z));
}
inline void copy(Z &to, const Z &from) {
  Limb m;
  bool neg;
  if (limb(from, m, neg)) store(to, m, neg);
  else to = from;
}
inline void move(Z &to, Z &from) noexcept {
  Limb m;
  bool neg;
  if (limb(from, m, neg)) store(to, m, neg);
  else to = std::move(from);
}
inline bool same(const Z &x, const Z &y) {
  const auto &a = x.backend(), &b = y.backend();
  if (a.size() == 1 && b.size() == 1) return a.limbs()[0] == b.limbs()[0] && a.sign() == b.sign();
  return x == y;
}
// Sum of two sign-magnitude values; false when the magnitude overflows.
inline bool add(Wide p, bool np, Wide q, bool nq, Wide &m, bool &neg) noexcept {
  if (np == nq) {
    m = p + q;
    neg = np;
    return m >= p;
  }
  if (p >= q) m = p - q, neg = np;
  else m = q - p, neg = nq;
  return true;
}
// Order of two sign-magnitude values whose zeros are unsigned.
inline int order(Wide p, bool np, Wide q, bool nq) noexcept {
  if (np != nq) return np ? -1 : 1;
  const int c = (p > q) - (p < q);
  return np ? -c : c;
}
// Exact quotient and remainder by a limb; a gcd of 1 divides nothing and a
// double limb with a zero high limb divides in one limb.
inline Limb quotient(Limb x, Limb g) noexcept { return g == 1 ? x : x / g; }
inline Wide quotient(Wide x, Limb g) noexcept {
  return g == 1 ? x : Limb(x >> limb_bits) == 0 ? Wide(Limb(x) / g) : x / g;
}
inline Limb remainder(Wide x, Limb g) noexcept { return Limb(x >> limb_bits) == 0 ? Limb(x) % g : Limb(x % g); }
// Builtin integers no wider than a limb; wider builtins (__int128) and Z
// expressions take the Z path.
template <class T>
concept LimbInteger = std::is_integral_v<T> && std::numeric_limits<T>::digits <= std::numeric_limits<Limb>::digits;
template <class T> bool limb_of(const T &i, Limb &m, bool &neg) noexcept {
  if constexpr (LimbInteger<T>) {
    neg = false;
    if constexpr (std::is_signed_v<T>) neg = i < 0;
    m = neg ? Limb(0) - Limb(i) : Limb(i);
    return true;
  } else if constexpr (std::is_same_v<T, Z>) return limb(i, m, neg);
  else return false;
}
} // namespace rational_detail
// Exactly the integer arguments boost::rational<Z> accepts (builtin integers,
// bool, Z and Z expressions), so every overload set stays as it was.
template <class T> concept RationalInteger = boost::rational_detail::is_compatible_integer<T, Z>::value;
class Rational;
template <class T> concept RationalOperand = RationalInteger<T> || std::same_as<T, Rational>;
// boost::rational<Z> (Boost 1.90, BOOST_VERSION 109000): the same members, the
// same invariant (den > 0, gcd(|num|, den) = 1, zero = 0/1), Boost's public
// operator set with Boost's argument constraint, and Boost's exceptions in
// Boost's order. Only how a result is computed differs. When every part of the
// operands has one limb, Boost's own formulas run on limbs and double limbs;
// otherwise they run verbatim on Z. A reduced pair with a positive denominator
// is unique and cpp_int is canonical, so every result stores the same Z pair.
// The non-member operators are hidden friends: like Boost's namespace-boost
// templates seen from rg, they are found only by ADL on a Rational operand.
class Rational {
  using Limb = rational_detail::Limb;
  using Wide = rational_detail::Wide;
  Z num_ = 0, den_ = 1;
  // Boost's safe-bool conversion, so every implicit conversion stays as it was.
  struct helper {
    Z parts[2];
  };
  using bool_type = Z (helper::*)[2];
  bool integer() const noexcept {
    Limb d;
    return rational_detail::limb(den_, d) && d == 1;
  }
  bool limbs(const Rational &r, Limb &a, bool &na, Limb &b, Limb &c, bool &nc, Limb &d) const noexcept {
    return rational_detail::limb(num_, a, na) && rational_detail::limb(den_, b) &&
           rational_detail::limb(r.num_, c, nc) && rational_detail::limb(r.den_, d);
  }
  void normalize() { // rational.hpp:886-916
    if (den_.is_zero()) BOOST_THROW_EXCEPTION(boost::bad_rational());
    if (num_.is_zero()) {
      den_ = 1;
      return;
    }
    Limb a, b;
    bool na, nb;
    if (rational_detail::limb(num_, a, na) && rational_detail::limb(den_, b, nb) && !nb) {
      if (const Limb g = std::gcd(a, b); g != 1) {
        rational_detail::store(num_, rational_detail::quotient(a, g), na);
        rational_detail::store(den_, rational_detail::quotient(b, g), false);
      }
      return;
    }
    const Z g = boost::integer::gcd(num_, den_);
    num_ /= g;
    den_ /= g;
    if (den_ < -(std::numeric_limits<Z>::max)())
      BOOST_THROW_EXCEPTION(boost::bad_rational("bad rational: non-zero singular denominator"));
    if (den_ < 0) {
      num_ = -num_;
      den_ = -den_;
    }
  }
  Rational &plus(const Rational &r, bool minus) { // rational.hpp:516-568
    Limb a, b, c, d;
    bool na, nc;
    if (limbs(r, a, na, b, c, nc, d)) {
      if (minus) nc = c != 0 && !nc;
      Wide m;
      bool neg;
      if (d == 1) { // (a + c*b)/b is reduced: gcd(a + c*b, b) = gcd(a, b) = 1
        rational_detail::add(a, na, Wide(c) * b, nc, m, neg);
        rational_detail::store(num_, m, neg);
        return *this;
      }
      if (b == 1) { // (a*d + c)/d, likewise reduced
        rational_detail::add(Wide(a) * d, na, c, nc, m, neg);
        rational_detail::store(num_, m, neg);
        rational_detail::store(den_, d, false);
        return *this;
      }
      // Mladenov: g = gcd(b, d), n = a*(d/g) + c*(b/g), h = gcd(n, g)
      const Limb g = std::gcd(b, d), b1 = rational_detail::quotient(b, g);
      if (rational_detail::add(Wide(a) * rational_detail::quotient(d, g), na, Wide(c) * b1, nc, m, neg)) {
        const Limb h = g == 1 ? 1 : std::gcd(rational_detail::remainder(m, g), g);
        rational_detail::store(num_, rational_detail::quotient(m, h), neg);
        rational_detail::store(den_, Wide(b1) * rational_detail::quotient(d, h), false);
        return *this;
      }
    }
    // Protect against self-modification
    Z r_num = r.num_;
    Z r_den = r.den_;
    Z g = boost::integer::gcd(den_, r_den);
    den_ /= g;
    if (minus) num_ = num_ * (r_den / g) - r_num * den_;
    else num_ = num_ * (r_den / g) + r_num * den_;
    g = boost::integer::gcd(num_, g);
    num_ /= g;
    den_ *= r_den / g;
    return *this;
  }
  // The sign of x - y: a*d - c*b with b, d > 0.
  static int order(const Rational &x, const Rational &y) {
    Limb a, b, c, d;
    bool na, nc;
    if (x.limbs(y, a, na, b, c, nc, d)) return rational_detail::order(Wide(a) * d, na, Wide(c) * b, nc);
    const int sx = x.num_.sign(), sy = y.num_.sign();
    if (sx != sy) return sx < sy ? -1 : 1;
    if (x.integer() && y.integer()) return x.num_.compare(y.num_);
    return Z(x.num_ * y.den_).compare(Z(y.num_ * x.den_));
  }
  // The sign of *this - i: num - i*den with den > 0.
  template <class T> int order(const T &i) const {
    Limb a, b, m;
    bool na, n;
    if (rational_detail::limb(num_, a, na) && rational_detail::limb(den_, b) && rational_detail::limb_of(i, m, n))
      return rational_detail::order(a, na, Wide(m) * b, n);
    if constexpr (std::is_same_v<T, Z>) return integer() ? num_.compare(i) : num_.compare(Z(i * den_));
    else {
      const Z z(i);
      return integer() ? num_.compare(z) : num_.compare(Z(z * den_));
    }
  }
  template <class T> Rational &plus_integer(const T &i, bool minus) { // rational.hpp:292-303
    Limb a, b, m;
    bool na, n;
    if (rational_detail::limb(num_, a, na) && rational_detail::limb(den_, b) && rational_detail::limb_of(i, m, n)) {
      Wide s;
      bool neg;
      rational_detail::add(a, na, Wide(m) * b, minus ? m != 0 && !n : n, s, neg);
      rational_detail::store(num_, s, neg);
      return *this;
    }
    if (minus) num_ -= i * den_;
    else num_ += i * den_;
    return *this;
  }

public:
  using int_type = Z;
  Rational() = default;
  template <RationalInteger T> Rational(const T &n) : num_(n) {}
  Rational(const Z &n) : num_(rational_detail::copied(n)) {}
  Rational(Z &&n) noexcept : num_(rational_detail::moved(n)) {}
  template <RationalInteger T, RationalInteger U> Rational(const T &n, const U &d) : num_(n), den_(d) {
    normalize();
  }
  Rational(const Rational &o) : num_(rational_detail::copied(o.num_)), den_(rational_detail::copied(o.den_)) {}
  Rational(Rational &&o) noexcept : num_(rational_detail::moved(o.num_)), den_(rational_detail::moved(o.den_)) {}
  Rational &operator=(const Rational &o) {
    rational_detail::copy(num_, o.num_);
    rational_detail::copy(den_, o.den_);
    return *this;
  }
  Rational &operator=(Rational &&o) noexcept {
    rational_detail::move(num_, o.num_);
    rational_detail::move(den_, o.den_);
    return *this;
  }
  template <RationalInteger T> Rational &operator=(const T &n) { // assign(n, 1): n/1 is reduced
    if constexpr (std::is_same_v<T, Z>) rational_detail::copy(num_, n);
    else num_ = n;
    den_ = 1;
    return *this;
  }
  template <RationalInteger T, RationalInteger U> Rational &assign(const T &n, const U &d) {
    return *this = Rational(static_cast<Z>(n), static_cast<Z>(d));
  }
  const Z &numerator() const noexcept { return num_; }
  const Z &denominator() const noexcept { return den_; }
  Rational &operator+=(const Rational &r) { return plus(r, false); }
  Rational &operator-=(const Rational &r) { return plus(r, true); }
  Rational &operator*=(const Rational &r) { // rational.hpp:570-583
    Limb a, b, c, d;
    bool na, nc;
    if (limbs(r, a, na, b, c, nc, d)) {
      if (b == 1 && d == 1) {
        rational_detail::store(num_, Wide(a) * c, na != nc);
        return *this;
      }
      const Limb g1 = d == 1 ? 1 : std::gcd(a, d), g2 = b == 1 ? 1 : std::gcd(c, b);
      rational_detail::store(num_, Wide(rational_detail::quotient(a, g1)) * rational_detail::quotient(c, g2), na != nc);
      rational_detail::store(den_, Wide(rational_detail::quotient(b, g2)) * rational_detail::quotient(d, g1), false);
      return *this;
    }
    // Protect against self-modification
    Z r_num = r.num_;
    Z r_den = r.den_;
    Z gcd1 = boost::integer::gcd(num_, r_den);
    Z gcd2 = boost::integer::gcd(r_num, den_);
    num_ = (num_ / gcd1) * (r_num / gcd2);
    den_ = (den_ / gcd2) * (r_den / gcd1);
    return *this;
  }
  Rational &operator/=(const Rational &r) { // rational.hpp:585-612
    if (r.num_.is_zero()) BOOST_THROW_EXCEPTION(boost::bad_rational()); // the divisor first: 0/0 throws
    if (num_.is_zero()) return *this;
    Limb a, b, c, d;
    bool na, nc;
    if (limbs(r, a, na, b, c, nc, d)) { // Boost's final sign flip leaves the sign on num
      const Limb g1 = std::gcd(a, c), g2 = b == 1 || d == 1 ? 1 : std::gcd(d, b);
      rational_detail::store(num_, Wide(rational_detail::quotient(a, g1)) * rational_detail::quotient(d, g2), na != nc);
      rational_detail::store(den_, Wide(rational_detail::quotient(b, g2)) * rational_detail::quotient(c, g1), false);
      return *this;
    }
    // Protect against self-modification
    Z r_num = r.num_;
    Z r_den = r.den_;
    Z gcd1 = boost::integer::gcd(num_, r_num);
    Z gcd2 = boost::integer::gcd(r_den, den_);
    num_ = (num_ / gcd1) * (r_den / gcd2);
    den_ = (den_ / gcd2) * (r_num / gcd1);
    if (den_ < 0) {
      num_ = -num_;
      den_ = -den_;
    }
    return *this;
  }
  template <RationalInteger T> Rational &operator+=(const T &i) { return plus_integer(i, false); }
  template <RationalInteger T> Rational &operator-=(const T &i) { return plus_integer(i, true); }
  template <RationalInteger T> Rational &operator*=(const T &i) { // rational.hpp:304-312
    Limb a, b, m;
    bool na, n;
    if (rational_detail::limb(num_, a, na) && rational_detail::limb(den_, b) && rational_detail::limb_of(i, m, n)) {
      const Limb g = b == 1 ? 1 : std::gcd(m, b);
      rational_detail::store(num_, Wide(a) * rational_detail::quotient(m, g), na != n);
      rational_detail::store(den_, rational_detail::quotient(b, g), false);
      return *this;
    }
    Z gcd = boost::integer::gcd(static_cast<Z>(i), den_);
    num_ *= i / gcd;
    den_ /= gcd;
    return *this;
  }
  template <RationalInteger T> Rational &operator/=(const T &i) { // rational.hpp:313-333
    Limb a, b, m;
    bool na, n;
    // Boost re-reads i after dividing num, so q /= q.numerator() keeps Boost's own path.
    if (rational_detail::limb(num_, a, na) && rational_detail::limb(den_, b) && rational_detail::limb_of(i, m, n) &&
        static_cast<const void *>(&i) != static_cast<const void *>(&num_)) {
      if (m == 0) BOOST_THROW_EXCEPTION(boost::bad_rational());
      if (a == 0) return *this;
      const Limb g = std::gcd(a, m);
      rational_detail::store(num_, rational_detail::quotient(a, g), na != n);
      rational_detail::store(den_, Wide(b) * rational_detail::quotient(m, g), false);
      return *this;
    }
    Z const zero(0);
    if (i == zero) BOOST_THROW_EXCEPTION(boost::bad_rational());
    if (num_ == zero) return *this;
    Z const gcd = boost::integer::gcd(num_, static_cast<Z>(i));
    num_ /= gcd;
    den_ *= i / gcd;
    if (den_ < zero) {
      num_ = -num_;
      den_ = -den_;
    }
    return *this;
  }
  const Rational &operator++() {
    num_ += den_;
    return *this;
  }
  const Rational &operator--() {
    num_ -= den_;
    return *this;
  }
  Rational operator++(int) {
    Rational t(*this);
    ++(*this);
    return t;
  }
  Rational operator--(int) {
    Rational t(*this);
    --(*this);
    return t;
  }
  bool operator!() const { return !num_; }
  operator bool_type() const { return operator!() ? nullptr : &helper::parts; }
  bool operator<(const Rational &r) const { return order(*this, r) < 0; }
  bool operator>(const Rational &r) const { return order(*this, r) > 0; }
  bool operator==(const Rational &r) const {
    return rational_detail::same(num_, r.num_) && rational_detail::same(den_, r.den_);
  }
  template <RationalInteger T> bool operator<(const T &i) const { return order(i) < 0; }
  template <RationalInteger T> bool operator>(const T &i) const { return order(i) > 0; }
  template <RationalInteger T> bool operator==(const T &i) const { // den == 1 && num == i
    if (!integer()) return false;
    if constexpr (rational_detail::LimbInteger<T>) {
      Limb a, m;
      bool na, n;
      rational_detail::limb_of(i, m, n);
      return rational_detail::limb(num_, a, na) && a == m && na == n;
    } else if constexpr (std::is_same_v<T, Z>) return rational_detail::same(num_, i);
    else return num_ == i;
  }
  friend Rational operator+(const Rational &r) { return r; }
  friend Rational operator-(const Rational &r) {
    Rational t(r);
    t.num_.backend().negate(); // the negation of a reduced pair is reduced
    return t;
  }
  template <RationalOperand Arg> friend Rational operator+(const Rational &a, const Arg &b) {
    Rational t(a);
    t += b;
    return t;
  }
  template <RationalInteger Arg> friend Rational operator+(const Arg &b, const Rational &a) {
    Rational t(a);
    t += b;
    return t;
  }
  template <RationalOperand Arg> friend Rational operator-(const Rational &a, const Arg &b) {
    Rational t(a);
    t -= b;
    return t;
  }
  template <RationalInteger Arg> friend Rational operator-(const Arg &b, const Rational &a) {
    Rational t(a);
    t -= b;
    t.num_.backend().negate();
    return t;
  }
  template <RationalOperand Arg> friend Rational operator*(const Rational &a, const Arg &b) {
    Rational t(a);
    t *= b;
    return t;
  }
  template <RationalInteger Arg> friend Rational operator*(const Arg &b, const Rational &a) {
    Rational t(a);
    t *= b;
    return t;
  }
  template <RationalOperand Arg> friend Rational operator/(const Rational &a, const Arg &b) {
    Rational t(a);
    t /= b;
    return t;
  }
  template <RationalInteger Arg> friend Rational operator/(const Arg &b, const Rational &a) {
    Rational t(b);
    t /= a;
    return t;
  }
  template <RationalOperand Arg> friend bool operator<=(const Rational &a, const Arg &b) { return !a.operator>(b); }
  template <RationalInteger Arg> friend bool operator<=(const Arg &b, const Rational &a) { return !a.operator<(b); }
  template <RationalOperand Arg> friend bool operator>=(const Rational &a, const Arg &b) { return !a.operator<(b); }
  template <RationalInteger Arg> friend bool operator>=(const Arg &b, const Rational &a) { return !a.operator>(b); }
  template <RationalOperand Arg> friend bool operator!=(const Rational &a, const Arg &b) { return !a.operator==(b); }
  template <RationalInteger Arg> friend bool operator!=(const Arg &b, const Rational &a) { return !a.operator==(b); }
  template <RationalInteger Arg> friend bool operator<(const Arg &b, const Rational &a) { return a.operator>(b); }
  template <RationalInteger Arg> friend bool operator>(const Arg &b, const Rational &a) { return a.operator<(b); }
  template <RationalInteger Arg> friend bool operator==(const Arg &b, const Rational &a) { return a.operator==(b); }
};
static_assert(sizeof(Rational) == 2 * sizeof(Z));
} // namespace rg
