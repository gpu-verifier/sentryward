#pragma once
#include "core.hpp"
#include <array>
namespace rg {
// These are integer coordinates, not numerical-real expressions. Index variables
// carry their integer meaning in Arena::var; a bounded real read is insufficient.
inline bool coordinate_integer(const Arena &a, Id id, unsigned &budget) {
  if (budget == 0) return false;
  --budget;
  const auto &n = a.nodes[id];
  if (!n.facts.finite) return false;
  if (n.op == "const") return n.data[0].denominator() == 1;
  if (n.op == "var") return n.data.size() == 2 &&
      n.data[0].denominator() == 1 && n.data[1].denominator() == 1;
  if (n.op == "idiv") return n.data.size() == 1 &&
      n.data[0].denominator() == 1 && n.data[0] > 0;
  if (n.op == "imod") return n.args.size() == 1 && n.data.size() == 1 &&
      n.data[0].denominator() == 1 && n.data[0] > 0 &&
      coordinate_integer(a, n.args[0], budget);
  if ((n.op == "add" || n.op == "mul") && n.args.size() <= 7) {
    for (const Q &q : n.data) if (q.denominator() != 1) return false;
    for (Id child : n.args) if (!coordinate_integer(a, child, budget)) return false;
    return true;
  }
  return false;
}
inline bool bounded_coordinate_integer(const Arena &a, Id id, const Z &limit) {
  const auto &n = a.nodes[id];
  if (!n.bounds || n.bounds->first < 0 || n.bounds->first > n.bounds->second ||
      n.bounds->second >= Q(limit)) return false;
  unsigned budget = 32;
  return coordinate_integer(a, id, budget);
}
// A retained exact XOR operation is useful proof vocabulary: simplifying a
// particular constant XOR into arithmetic would discard its common permutation.
inline std::optional<Id> index_xor(Arena &a, Id index, Id mask, unsigned bits) {
  if ((bits != 6 && bits != 7) ||
      !bounded_coordinate_integer(a, index, pow2(bits)) ||
      !bounded_coordinate_integer(a, mask, pow2(bits))) return {};
  return a.intern("index_xor", {index, mask}, {Q(bits)}, "", {true, false, true},
                  std::pair(Q(0), Q(pow2(bits) - 1)));
}

// Recognize short floor/mod/add digit expressions by identities on binary digits,
// not by evaluating the input domain. A form has one integer axis of <=7 bits;
// each of its at most seven coefficients multiplies exactly one input bit.
struct XorCoordinateNormalizer {
  struct Digits {
    Id axis = invalid;
    unsigned width = 0;
    Z constant = 0;
    std::array<Z, 7> coefficients{};
    bool sliced = false;
  };
  Arena &a;
  std::unordered_map<Id, std::optional<Digits>> digits_cache;
  std::map<std::tuple<Id, unsigned, unsigned>, Id> address_cache;
  explicit XorCoordinateNormalizer(Arena &arena) : a(arena) {}
  static std::optional<unsigned> shift(const Q &q) {
    if (q.denominator() != 1 || q <= 0 || q > 128) return {};
    Z v = q.numerator();
    if ((v & (v - 1)) != 0) return {};
    return boost::multiprecision::msb(v);
  }
  static std::pair<Z, Z> range(const Digits &f) {
    Z low = f.constant, high = f.constant;
    for (const Z &c : f.coefficients) {
      if (c < 0) low += c; else high += c;
    }
    return {low, high};
  }
  std::optional<Digits> digits(Id id, unsigned &budget, unsigned depth = 0) {
    if (depth > 12 || budget == 0) return {};
    --budget;
    if (auto it = digits_cache.find(id); it != digits_cache.end()) return it->second;
    ++a.stats["coordinate_digit_node_visits"];
    const auto &node = a.nodes[id]; // digit parsing does not construct Arena nodes
    auto calculate = [&]() -> std::optional<Digits> {
      if (!node.facts.finite) return {};
      Digits result;
      if (node.op == "const") {
        if (node.data[0].denominator() != 1) return {};
        result.constant = node.data[0].numerator();
        return result;
      }
      if (node.op == "var") {
        if (node.data.size() != 2 || !node.bounds || node.data[0] != 0 ||
            node.bounds->first != 0 || node.bounds->second != node.data[1]) return {};
        auto width = shift(node.data[1] + 1);
        if (!width || *width == 0 || *width > 7) return {};
        result.axis = id; result.width = *width;
        for (unsigned i = 0; i < *width; ++i) result.coefficients[i] = pow2(i);
        return result;
      }
      if (node.op == "add" || (node.op == "mul" && node.args.size() == 1)) {
        if (node.args.size() > 8 || (node.op == "add" && node.data.size() != node.args.size())) return {};
        for (size_t j = 0; j < node.args.size(); ++j) {
          Q coefficient = node.data[node.op == "add" ? j : 0];
          if (coefficient.denominator() != 1) return {};
          auto child = digits(node.args[j], budget, depth + 1);
          if (!child) return {};
          if (child->axis != invalid) {
            if (result.axis != invalid && result.axis != child->axis) return {};
            result.axis = child->axis; result.width = child->width;
          }
          result.constant += coefficient.numerator() * child->constant;
          for (unsigned i = 0; i < 7; ++i)
            result.coefficients[i] += coefficient.numerator() * child->coefficients[i];
          result.sliced |= child->sliced;
        }
        return result;
      }
      if ((node.op == "idiv" || node.op == "imod") && node.args.size() == 1 && node.data.size() == 1) {
        auto amount = shift(node.data[0]);
        auto child = digits(node.args[0], budget, depth + 1);
        if (!amount || !child || child->constant != 0) return {};
        unsigned occupied = 0;
        // Dividing a carry-free positive bit slice drops its low bits. A signed
        // sum or an overlapping digit is deliberately not a supported slice.
        for (unsigned i = 0; i < 7; ++i) {
          Z c = child->coefficients[i];
          if (c == 0) continue;
          auto bit = shift(Q(c));
          if (!bit || (occupied & (1u << *bit))) return {};
          occupied |= 1u << *bit;
          if (node.op == "idiv") child->coefficients[i] = *bit < *amount ? Z(0) : pow2(*bit - *amount);
          else if (*bit >= *amount) child->coefficients[i] = 0;
        }
        child->sliced = true;
        return child;
      }
      if (node.op == "word" && node.args.size() == 1 && node.data.size() == 1 &&
          node.data[0].denominator() == 1 && node.data[0] > 0 && node.data[0] <= 64) {
        auto child = digits(node.args[0], budget, depth + 1);
        if (!child) return {};
        Z modulus = pow2(node.data[0].numerator().convert_to<unsigned>());
        // Unsigned source literals such as -8 enter as 2^64-8. First preserve
        // word congruence by reducing each integer bit coefficient; only then
        // may exact digit bounds prove a unique nonwrapping representative.
        child->constant = modz(child->constant, modulus);
        for (Z &c : child->coefficients) {
          bool negative = c < 0;
          c = modz(c, modulus);
          // Both signs represent the half-modulus residue. Preserve the
          // original sign at this tie so a negative digit need not acquire
          // an artificial carry; a positive half digit remains positive.
          if (c > modulus / 2 || (negative && c == modulus / 2)) c -= modulus;
        }
        auto [low, high] = range(*child);
        if (low < 0 || high >= modulus) return {};
        return child;
      }
      return {};
    };
    auto result = calculate();
    digits_cache.emplace(id, result);
    return result;
  }
  bool eligible(Id offset) const {
    const auto &free = a.nodes[offset].free;
    if (free.size() != 1) return false;
    const auto &axis = a.nodes[free[0]];
    if (axis.op != "var" || !axis.facts.finite || axis.data.size() != 2 ||
        axis.data[0] != 0 || !axis.bounds || axis.bounds->first != 0 ||
        axis.bounds->second != axis.data[1]) return false;
    auto bits = shift(axis.data[1] + 1);
    return bits && *bits >= 1 && *bits <= 7;
  }
  Id address(Id offset, unsigned bytes, unsigned bits) {
    Id result = offset;
    if (bytes != 2 || (bits != 6 && bits != 7) || !eligible(offset)) return result;
    // Almost all typed addresses are ordinary affine expressions or involve
    // multiple axes. Inspect the one candidate axis before allocating a cache
    // entry or walking any producer; this is an immutable integer-domain test.
    auto key = std::tuple(offset, bytes, bits);
    if (auto found = address_cache.find(key); found != address_cache.end()) return found->second;
    unsigned budget = 64;
    auto f = digits(offset, budget);
    if (f && f->axis != invalid && f->sliced && modz(f->constant, bytes) == 0) {
      Z elements = f->constant / bytes, domain = pow2(bits);
      Z seed = modz(elements, domain), base = elements - seed;
      std::array<int, 7> output_bits{}; output_bits.fill(-1);
      unsigned occupied = 0; bool valid = true, dynamic = false;
      for (unsigned i = 0; i < f->width; ++i) {
        Z c = f->coefficients[i];
        if (c == 0) continue;
        dynamic = true;
        if (modz(c, bytes) != 0) { valid = false; break; }
        Z magnitude = c / bytes;
        if (magnitude < 0) magnitude = -magnitude;
        auto bit = shift(Q(magnitude));
        if (!bit || *bit >= bits || (occupied & (1u << *bit)) ||
            (((seed >> *bit) & 1) != 0) != (c < 0)) { valid = false; break; }
        occupied |= 1u << *bit; output_bits[i] = int(*bit);
      }
      if (valid && dynamic) {
        std::vector<Id> mask_parts;
        for (unsigned i = 0; i < f->width;) {
          if (output_bits[i] < 0) { ++i; continue; }
          unsigned end = i + 1;
          while (end < f->width && output_bits[end] == output_bits[i] + int(end - i)) ++end;
          Id slice = a.idiv(f->axis, pow2(i));
          if (end < f->width) slice = a.imod(slice, pow2(end - i));
          mask_parts.push_back(a.mul({a.constant(Q(pow2(unsigned(output_bits[i])))), slice}));
          i = end;
        }
        Id mask = a.add(std::move(mask_parts));
        if (auto permuted = index_xor(a, a.constant(Q(seed)), mask, bits)) {
          result = a.mul({a.constant(bytes), a.add({a.constant(Q(base)), *permuted})});
          ++a.stats["coordinate_digit_xor_certificates"];
        }
      }
    }
    address_cache.emplace(key, result);
    return result;
  }
};
} // namespace rg
