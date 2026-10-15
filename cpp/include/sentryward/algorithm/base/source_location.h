#pragma once

#include <cstdint>
#include <string>

#include "sentryward/algorithm/base/shared_string.h"

namespace sentryward::algorithm {

struct SourceLocation final {
  SharedString file;
  std::uint32_t line = 0;
  std::uint32_t column = 0;

  friend bool operator==(const SourceLocation&, const SourceLocation&) = default;
};

}  // namespace sentryward::algorithm
