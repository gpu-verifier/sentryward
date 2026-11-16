#pragma once
// Class mode's table atoms (frontend execution.h BoundedValue::exact): fixed read-only shape-array
// tables, whose elements address-model lookups name by their object and element type
// (TensorPropertySymbol) or by an element function over them (TensorPropertyFunctionSymbol). A
// lookup's argument is the element's byte offset in its table, so the exact valuation of a lookup is
// the table's element there (read as the symbol's type), through its function. A representative's
// lookups have its own arguments; a member's are the representative's shifted by the member's δ of
// the table object (the class log compared every element read up to that δ).
//
// Cost: construction O(table elements); Value O(1) for a plain lookup and O(function text) for an
// element function (parsed once per distinct function and cached).
#include "sentryward/algorithm/memory/address_solver.h"
#include "sentryward/algorithm/memory/symbolic_address_model.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace sentryward::algorithm {

class TableValuation final {
 public:
  using Int = address_solver::Int;
  // `tables`: a JSON array of {"object": O, "element_bytes": n, "values": [raw element bits]}.
  explicit TableValuation(std::string_view tables = "[]") {
    const auto parsed = nlohmann::json::parse(tables);
    for (const auto& table : parsed) {
      Table entry;
      entry.bytes = table.at("element_bytes").get<unsigned>();
      if (entry.bytes == 0 || entry.bytes > 8) throw std::invalid_argument("table atom element width");
      for (const auto& value : table.at("values")) entry.raw.push_back(value.get<std::uint64_t>());
      tables_.emplace(table.at("object").get<std::string>(), std::move(entry));
    }
  }
  [[nodiscard]] bool empty() const noexcept { return tables_.empty(); }
  [[nodiscard]] bool Has(const std::string& object) const { return tables_.contains(object); }
  // The objects with their element widths.
  [[nodiscard]] std::vector<std::pair<std::string, unsigned>> Objects() const {
    std::vector<std::pair<std::string, unsigned>> result;
    for (const auto& [object, table] : tables_) result.emplace_back(object, table.bytes);
    return result;
  }
  // Every element of `object` read as `sign`ed: the least and greatest, and whether all differ.
  // O(n log n) once per (object, sign).
  struct Range {
    Int lower = 0, upper = 0;
    bool injective = false;
  };
  [[nodiscard]] std::optional<Range> RangeOf(const std::string& object, bool sign) const {
    const auto found = tables_.find(object);
    if (found == tables_.end() || found->second.raw.empty()) return std::nullopt;
    std::vector<Int> values;
    values.reserve(found->second.raw.size());
    for (const auto raw : found->second.raw) values.push_back(Read(found->second, raw, sign));
    std::sort(values.begin(), values.end());
    Range range{values.front(), values.back(), std::adjacent_find(values.begin(), values.end()) == values.end()};
    return range;
  }
  // The object of a table lookup symbol, if it names a table.
  [[nodiscard]] std::optional<std::string> ObjectOf(std::string_view symbol) const {
    const auto parsed = Parse(symbol);
    if (!parsed || !tables_.contains(parsed->object)) return std::nullopt;
    return parsed->object;
  }
  // The exact value of lookup `symbol` at byte offset `arg`, or nullopt (not a table lookup, an
  // argument off the table or misaligned, a function this evaluator declines).
  [[nodiscard]] std::optional<Int> Value(std::string_view symbol, Int arg) const {
    const auto parsed = Parse(symbol);
    if (!parsed) return std::nullopt;
    const auto found = tables_.find(parsed->object);
    if (found == tables_.end()) return std::nullopt;
    const auto& table = found->second;
    if (parsed->bits != 8 * table.bytes || arg < 0 || arg % table.bytes != 0) return std::nullopt;
    const Int index = arg / table.bytes;
    if (index >= Int(table.raw.size())) return std::nullopt;
    const Int x = Read(table, table.raw[static_cast<std::size_t>(index)], parsed->sign);
    if (parsed->function.empty()) return x;
    return Apply(parsed->function, x);
  }
  // The exact offset of an abstract footprint in its allocation: `expression` an address on
  // `allocation` in `space` with a plain view whose offset is c + sum of k x lookup(symbol, arg)
  // (frontend Resources::AbstractExpression over slot_resolver.cpp AffineOffsetJson), every lookup a
  // table element here. Nullopt for any other expression (a guard, a TMA region, another operator),
  // a lookup the tables decline, or an offset outside [0, 2^64). O(expression size).
  [[nodiscard]] std::optional<std::uint64_t> FootprintOffset(std::string_view expression, std::string_view space,
                                                             std::string_view allocation) const {
    if (tables_.empty()) return std::nullopt;
    const auto parsed = nlohmann::json::parse(expression, nullptr, false);
    if (!parsed.is_object() || parsed.contains("guard") || parsed.contains("operator")) return std::nullopt;
    const auto address = parsed.find("address");
    if (address == parsed.end() || !address->is_object()) return std::nullopt;
    const auto base = address->find("base"), view = address->find("view"), form = address->find("offset");
    if (base == address->end() || !base->is_object() || form == address->end() || !form->is_object() ||
        base->value("identity", nlohmann::json()) != nlohmann::json(allocation) ||
        base->value("space", nlohmann::json()) != nlohmann::json(space) ||
        (view != address->end() && !view->is_null()) || form->value("operator", std::string()) != "add")
      return std::nullopt;
    const auto operands = form->value("operands", nlohmann::json());
    if (!operands.is_array() || operands.empty() || !operands[0].is_number_integer()) return std::nullopt;
    const auto integer = [](const nlohmann::json& x) -> std::optional<Int> {
      if (x.is_number_unsigned()) return Int(x.get<std::uint64_t>());
      if (x.is_number_integer()) return Int(x.get<std::int64_t>());
      return std::nullopt;
    };
    constexpr Int kBound = Int(1) << 100;
    Int total = *integer(operands[0]);
    for (std::size_t k = 1; k < operands.size(); ++k) {
      const auto& term = operands[k];
      if (!term.is_object() || term.value("operator", std::string()) != "mul") return std::nullopt;
      const auto parts = term.value("operands", nlohmann::json());
      if (!parts.is_array() || parts.size() != 2 || !parts[0].is_object() ||
          parts[0].value("operator", std::string()) != "gather")
        return std::nullopt;
      const auto coefficient = integer(parts[1]);
      const auto args = parts[0].value("args", nlohmann::json());
      const auto symbol = parts[0].value("symbol", nlohmann::json());
      if (!coefficient || !args.is_array() || args.size() != 1 || !symbol.is_string()) return std::nullopt;
      const auto arg = integer(args[0]);
      const auto element = arg ? Value(symbol.get<std::string>(), *arg) : std::nullopt;
      if (!element || *element < -kBound || *element > kBound || *coefficient < -kBound || *coefficient > kBound)
        return std::nullopt;
      total += *element * *coefficient;
      if (total < -kBound || total > kBound) return std::nullopt;
    }
    if (total < 0 || total > Int(std::numeric_limits<std::uint64_t>::max())) return std::nullopt;
    return static_cast<std::uint64_t>(total);
  }

 private:
  struct Table {
    unsigned bytes = 0;
    std::vector<std::uint64_t> raw;
  };
  struct Symbol {
    std::string object, function;
    unsigned bits = 0;
    bool sign = false;
  };
  static Int Read(const Table& table, std::uint64_t raw, bool sign) {
    const unsigned bits = 8 * table.bytes;
    const std::uint64_t mask = bits == 64 ? ~std::uint64_t(0) : (std::uint64_t(1) << bits) - 1;
    raw &= mask;
    Int value = Int(raw);
    if (sign && bits < 128 && (raw >> (bits - 1)) & 1) value -= Int(1) << bits;
    return value;
  }
  static std::optional<Symbol> Parse(std::string_view symbol) {
    Symbol result;
    std::string_view type;
    constexpr std::string_view function_prefix = "tensor_property_function:";
    if (symbol.starts_with(function_prefix)) {
      auto rest = symbol.substr(function_prefix.size());
      const auto colon = rest.find(':');
      if (colon == rest.npos) return std::nullopt;
      std::size_t length = 0;
      for (const char c : rest.substr(0, colon)) {
        if (c < '0' || c > '9') return std::nullopt;
        length = length * 10 + std::size_t(c - '0');
      }
      rest = rest.substr(colon + 1);
      if (rest.size() < length + 1 || rest[length] != ':') return std::nullopt;
      result.object = std::string(rest.substr(0, length));
      rest = rest.substr(length + 1);
      const auto split = rest.find(':');
      if (split == rest.npos) return std::nullopt;
      type = rest.substr(0, split);
      result.function = std::string(rest.substr(split + 1));
      if (result.function.empty()) return std::nullopt;
    } else {
      const auto parsed = ParseTensorPropertySymbol(symbol);
      if (!parsed) return std::nullopt;
      result.object = parsed->first;
      type = parsed->second;
    }
    if (type.size() < 2 || (type[0] != 's' && type[0] != 'u')) return std::nullopt;
    result.sign = type[0] == 's';
    unsigned bits = 0;
    for (const char c : type.substr(1)) {
      if (c < '0' || c > '9') return std::nullopt;
      bits = bits * 10 + unsigned(c - '0');
    }
    if (bits == 0 || bits > 64) return std::nullopt;
    result.bits = bits;
    return result;
  }
  // An element function (frontend execution.cpp WithAbstract): F := op(L,k) | op(k,L) with
  // L := lin(c0,c1,G), G := x | F; op in shr, div, rem, and, or, xor, min, max, applied to the
  // exact integers. Declined (nullopt) where the frontend's rule needs the operand's type: a
  // negative operand of shr, div, rem, and, or, xor.
  static std::optional<Int> Apply(std::string_view function, Int x) {
    std::size_t at = 0;
    const auto value = Function(function, at, x, 0);
    if (!value || at != function.size()) return std::nullopt;
    return value;
  }
  static std::optional<Int> Number(std::string_view text, std::size_t& at) {
    const bool negative = at < text.size() && text[at] == '-';
    if (negative) ++at;
    if (at >= text.size() || text[at] < '0' || text[at] > '9') return std::nullopt;
    Int value = 0;
    while (at < text.size() && text[at] >= '0' && text[at] <= '9') {
      if (__builtin_mul_overflow(value, Int(10), &value) || __builtin_add_overflow(value, Int(text[at] - '0'), &value))
        return std::nullopt;
      ++at;
    }
    return negative ? -value : value;
  }
  static bool Eat(std::string_view text, std::size_t& at, std::string_view token) {
    if (text.substr(at, token.size()) != token) return false;
    at += token.size();
    return true;
  }
  // G at `at`: "x" or a function.
  static std::optional<Int> Inner(std::string_view text, std::size_t& at, Int x, unsigned depth) {
    if (Eat(text, at, "x")) return x;
    return Function(text, at, x, depth + 1);
  }
  // L or k at `at`: lin(c0,c1,G) or an integer.
  static std::optional<Int> Argument(std::string_view text, std::size_t& at, Int x, unsigned depth, bool& linear) {
    if (Eat(text, at, "lin(")) {
      linear = true;
      const auto c0 = Number(text, at);
      if (!c0 || !Eat(text, at, ",")) return std::nullopt;
      const auto c1 = Number(text, at);
      if (!c1 || !Eat(text, at, ",")) return std::nullopt;
      const auto g = Inner(text, at, x, depth);
      if (!g || !Eat(text, at, ")")) return std::nullopt;
      Int product, sum;
      if (__builtin_mul_overflow(*c1, *g, &product) || __builtin_add_overflow(*c0, product, &sum)) return std::nullopt;
      return sum;
    }
    linear = false;
    return Number(text, at);
  }
  static std::optional<Int> Function(std::string_view text, std::size_t& at, Int x, unsigned depth) {
    if (depth > 64) return std::nullopt;
    const auto open = text.find('(', at);
    if (open == text.npos) return std::nullopt;
    const auto op = text.substr(at, open - at);
    at = open + 1;
    bool left_linear = false, right_linear = false;
    const auto a = Argument(text, at, x, depth, left_linear);
    if (!a || !Eat(text, at, ",")) return std::nullopt;
    const auto b = Argument(text, at, x, depth, right_linear);
    if (!b || !Eat(text, at, ")") || left_linear == right_linear) return std::nullopt;
    const Int u = *a, v = *b;
    if (op == "min") return u < v ? u : v;
    if (op == "max") return u < v ? v : u;
    if (u < 0 || v < 0) return std::nullopt;
    if (op == "and") return u & v;
    if (op == "or") return u | v;
    if (op == "xor") return u ^ v;
    if (op == "shr") return v >= 127 ? Int(0) : Int(u >> static_cast<int>(v));
    if (op == "div") return v == 0 ? std::nullopt : std::optional<Int>(u / v);
    if (op == "rem") return v == 0 ? std::nullopt : std::optional<Int>(u % v);
    return std::nullopt;
  }
  std::map<std::string, Table, std::less<>> tables_;
};

}  // namespace sentryward::algorithm
