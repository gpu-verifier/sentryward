#pragma once

#include <cstdint>
#include <vector>

#include "sentryward/algorithm/source_ir/diagnostic.h"
#include "sentryward/algorithm/source_ir/program.h"

namespace sentryward::algorithm {

struct InitializationWarning final {
  BarrierLifetimeId barrier;
  InstId replaced_instruction;
  InstId retained_instruction;
  SourceLocation replaced_location;
  SourceLocation retained_location;
  std::uint32_t replaced_expected_arrivals;
  std::uint32_t retained_expected_arrivals;

  friend bool operator==(const InitializationWarning&,
                         const InitializationWarning&) = default;
};

struct InitializationRecovery final {
  std::vector<InitializationWarning> warnings;
  std::vector<Diagnostic> diagnostics;

  [[nodiscard]] bool applied() const noexcept { return !warnings.empty(); }
};

// Alternate-program recovery, not PTX equivalence. Requires only duplicate-init
// diagnostics, same-actor initializers, and no local barrier use before the last
// init in PO. Legacy AsyncLoad/Store completions targeting affected barriers are
// refused on any actor. Any failure leaves the whole program unchanged.
// Earlier inits become Await{}; edges and other uses stay unchanged. Explicit
// Issue refs to replaced inits become Complete refs, preserving one logical
// event. Callers must remap external refs similarly and check initialization
// ordering on the recovered program. Warnings retain IDs, locations and counts.
[[nodiscard]] InitializationRecovery RecoverDuplicateInitializers(
    SourceProgram& program);

}  // namespace sentryward::algorithm
