#pragma once
#include "sentryward/frontend/ptx/execution.h"
#include <map>

namespace sentryward::frontend::ptx {
using TensorMapFields = std::map<std::pair<std::string,std::optional<std::int64_t>>,Scalar>;
nlohmann::json DecodeTensorMap(const TensorMapFields &,const AllocationPointer &);
nlohmann::json ResolveTensorMap(const nlohmann::json &,unsigned rank);
void PublishTensorMap(nlohmann::json &,const nlohmann::json &);
void RecoverTensorMaps(ExecutionProgram &);
}
