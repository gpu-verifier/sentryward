#pragma once
#include "sentryward/frontend/ptx/execution.h"
#include <map>
#include <tuple>
namespace sentryward::frontend::ptx {
using NamedPhaseMap=std::map<std::tuple<std::uint32_t,std::uint32_t,DynamicKey>,std::uint32_t>;
NamedPhaseMap VariableCountPhases(const ExecutionProgram &);
}
