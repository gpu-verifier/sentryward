#pragma once

#include <vector>

#include "sentryward/algorithm/base/result.h"
#include "sentryward/algorithm/source_ir/diagnostic.h"
#include "sentryward/algorithm/source_ir/program.h"

namespace sentryward::algorithm {

// A program state that SourceValidator accepted; only validation creates one. It refers to the
// program, which keeps that state while no value of it changes (storage changes that keep every
// value, such as compaction, keep it). A transformation that may change the program takes the
// validation and returns it only when it changed nothing.
class ValidatedSource final {
 public:
  [[nodiscard]] const SourceProgram& program() const noexcept { return *program_; }

 private:
  friend class SourceValidator;
  explicit ValidatedSource(const SourceProgram& program) noexcept : program_(&program) {}
  const SourceProgram* program_;
};

class SourceValidator final {
 public:
  // The program's validation, or its diagnostics.
  [[nodiscard]] Result<ValidatedSource, std::vector<Diagnostic>> Validate(
      const SourceProgram& program) const;
};

}  // namespace sentryward::algorithm
