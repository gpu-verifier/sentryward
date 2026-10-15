#pragma once

#include <map>
#include <string>
#include <vector>

#include "sentryward/algorithm/source_ir/program.h"

namespace sentryward::algorithm {

using PtxAssumptions = std::map<std::string, bool>;
struct AssumptionDefinition { std::string label, key, text; };
[[nodiscard]] const std::vector<AssumptionDefinition>& AssumptionPool();

// Defaults are embedded from config/default-assumptions.json at build time.
[[nodiscard]] PtxAssumptions DefaultPtxAssumptions();
void OverridePtxAssumptions(PtxAssumptions& target, const PtxAssumptions& overrides);


}  // namespace sentryward::algorithm
