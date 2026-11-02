#pragma once
#include <charconv>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include "sentryward/scalar_fold/graph.h"

namespace sentryward::scalar_fold {
// Index levels [0,shape.size()) bind output coordinates; inner Folds follow them.
struct OutputFamily {
  std::string allocation;
  std::uint64_t offset=0, element_bytes=0;
  std::vector<std::uint64_t> shape, strides_bytes;
  Ref body;
};
// The bytes an equivalence binding compares in one allocation: element_bytes
// wide elements at offset + sum_i index_i * strides_bytes[i], index_i < shape[i].
struct ComparedOutput {
  std::uint64_t offset=0, element_bytes=0;
  std::vector<std::uint64_t> shape, strides_bytes;
};

// An output at a symbolic position (index-function memory): its body is the
// Operation kPlacementOperation(value, atom_1, ..., atom_k) of the value's type, and
// its bytes lie at the family offset + sum(coefficient_j x atom_j). Each atom is an
// integer InitialCell of a read-only tensor_properties input whose elements lie in
// [lower_j, upper_j] (trusted premises). The attributes list "coefficient,lower,upper"
// per atom, ';'-separated. This is a placement annotation for the comparison
// adapter, which removes it; it is never a value of any other consumer.
inline constexpr std::string_view kPlacementOperation = "SymbolicPlacement";
struct PlacementTerm {
  std::int64_t coefficient = 0, lower = 0, upper = 0;
};
inline std::string placementAttributes(std::span<const PlacementTerm> terms) {
  std::string text;
  for (const auto& term : terms) {
    if (!text.empty()) text += ';';
    text += std::to_string(term.coefficient) + ',' + std::to_string(term.lower) + ',' + std::to_string(term.upper);
  }
  return text;
}
inline std::optional<std::vector<PlacementTerm>> parsePlacementAttributes(std::string_view text) {
  std::vector<PlacementTerm> terms;
  while (!text.empty()) {
    const auto end = text.find(';');
    auto item = text.substr(0, end);
    PlacementTerm term;
    std::int64_t* fields[] = {&term.coefficient, &term.lower, &term.upper};
    for (unsigned i = 0; i < 3; ++i) {
      const auto parsed = std::from_chars(item.data(), item.data() + item.size(), *fields[i]);
      if (parsed.ec != std::errc{}) return std::nullopt;
      item.remove_prefix(std::size_t(parsed.ptr - item.data()));
      if (i < 2) {
        if (item.empty() || item.front() != ',') return std::nullopt;
        item.remove_prefix(1);
      }
    }
    if (!item.empty() || !term.coefficient || term.lower > term.upper) return std::nullopt;
    terms.push_back(term);
    if (end == std::string_view::npos) break;
    text.remove_prefix(end + 1);
    if (text.empty()) return std::nullopt;
  }
  if (terms.empty()) return std::nullopt;
  return terms;
}
}
