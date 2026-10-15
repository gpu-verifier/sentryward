#pragma once

#include "sentryward/algorithm/base/result.h"
#include "sentryward/algorithm/event_ir/program.h"
#include "sentryward/algorithm/source_ir/diagnostic.h"
#include "sentryward/algorithm/source_ir/program.h"

namespace sentryward::algorithm {

// Which dependencies of a warp data collective (shfl/elect/vote/match/redux
// .sync) the lowered graph carries. The two relations differ only in the
// completion -> done edge of these collectives; events and IDs are identical.
enum class CollectiveOrdering {
  // Memory order: the rendezvous publishes no memory effects, so a
  // participant's return is not ordered after the other lanes' issues (#251).
  kMemory,
  // Execution order: a participant resumes only after every participant has
  // issued the collective. The wait, phase and lifecycle checks reason about
  // when synchronization events can execute and need this relation.
  kExecution,
};

class Lowerer final {
 public:
  [[nodiscard]] Result<LoweredProgram, Diagnostic> Run(
      const SourceProgram& source, bool synchronization_only = false,
      CollectiveOrdering ordering = CollectiveOrdering::kMemory) const;
};

}  // namespace sentryward::algorithm
