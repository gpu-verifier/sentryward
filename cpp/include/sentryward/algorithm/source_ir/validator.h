#pragma once

#include <vector>

#include "sentryward/algorithm/source_ir/diagnostic.h"
#include "sentryward/algorithm/source_ir/program.h"

namespace sentryward::algorithm {

class SourceValidator final {
 public:
  [[nodiscard]] std::vector<Diagnostic> Validate(
      const SourceProgram& program) const;
};

}  // namespace sentryward::algorithm
