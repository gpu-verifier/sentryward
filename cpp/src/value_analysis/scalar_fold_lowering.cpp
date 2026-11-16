#include "scalar_fold_lowering_impl.hpp"

namespace sentryward::value_analysis {
ScalarFoldResult lowerScalarFold(const value_ir::Graph& graph,const MemoryEliminationResult& memory,
                                const ScalarFoldOptions& options){
  return Lower(graph,memory,options).run();
}
} // namespace sentryward::value_analysis
