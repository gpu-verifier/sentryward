#pragma once

#include "sentryward/algorithm/base/result.h"
#include "sentryward/algorithm/event_ir/program.h"
#include "sentryward/algorithm/source_ir/diagnostic.h"
#include "sentryward/algorithm/source_ir/program.h"
#include "sentryward/algorithm/source_ir/validator.h"

namespace sentryward::algorithm {

// One lowering gives both relations of a program: its graph is the execution order, and the
// memory order is that graph less the execution-only edges (LoweredProgram::memoryOrderGraph).
class Lowerer final {
 public:
  [[nodiscard]] Result<LoweredProgram, Diagnostic> Run(
      const ValidatedSource& source, bool synchronization_only = false) const;
  // Validates the program first; an invalid program fails with its least diagnostic.
  [[nodiscard]] Result<LoweredProgram, Diagnostic> Run(
      const SourceProgram& source, bool synchronization_only = false) const;
};

}  // namespace sentryward::algorithm
