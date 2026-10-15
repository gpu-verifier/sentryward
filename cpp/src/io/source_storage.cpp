#include "sentryward/algorithm/io/source_storage.h"

#include <type_traits>

#include "sentryward/algorithm/base/compact_vector.h"
#include "sentryward/algorithm/io/source_json.h"

namespace sentryward::algorithm {

void CompactSourceStorage(SourceArtifact& source) {
  for (std::size_t i=0;i<source.program.instructions.size();++i) {
    if (!source.program.instructions.isConcrete(i)) continue;
    auto& instruction=source.program.instructions.edit(i);
    std::visit([](auto& operation) {
      using T = std::decay_t<decltype(operation)>;
      if constexpr (std::is_same_v<T, AsyncOp>) {
        CompactVector(operation.effects);
        CompactVector(operation.completion_requires);
      } else if constexpr (std::is_same_v<T, Await>) {
        CompactVector(operation.requirements);
      }
    }, instruction.op);
  }
  source.program.instructions.shrink_to_fit();
  source.program.edges.shrink_to_fit();
  CompactVector(source.program.actors);
  CompactVector(source.program.collectives);
  CompactVector(source.program.dependencies);
  CompactVector(source.addresses);
  CompactVector(source.required_orders);
  CompactVector(source.verification_limitations);
  CompactVector(source.source_violations);
}

}  // namespace sentryward::algorithm
