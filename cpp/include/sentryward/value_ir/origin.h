#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "sentryward/algorithm/base/shared_string.h"

namespace sentryward::value_ir {

// Provenance only: none of these fields create graph or scheduling edges.
struct SourceOrigin final {
  std::optional<std::uint64_t> thread;
  std::optional<std::uint64_t> cta;
  std::optional<std::array<std::uint64_t, 3>> cta_coordinates;
  std::optional<std::uint64_t> warp;
  std::optional<std::uint64_t> lane;
  std::optional<std::uint64_t> instruction_index;
  std::optional<std::uint64_t> occurrence;
  std::optional<std::uint64_t> lowering_index;
  std::vector<std::pair<std::uint64_t, std::uint64_t>> unroll_context;
  std::optional<std::array<std::uint64_t, 3>> thread_coordinates;
  std::optional<std::uint64_t> coordinate_value;
  std::optional<algorithm::SharedString> function;
  std::optional<algorithm::SharedString> source_file;
  std::optional<std::uint64_t> source_line;
  std::optional<std::uint64_t> effect_identity;
  std::optional<algorithm::SharedString> source_text;
  std::optional<algorithm::SharedString> scope;
  std::optional<algorithm::SharedString> predicate;
  std::optional<bool> executed;

  friend bool operator==(const SourceOrigin&, const SourceOrigin&) = default;
};

}  // namespace sentryward::value_ir
