#include "sentryward/algorithm/source_ir/ptx_assumptions.h"
#include "sentryward/algorithm/source_ir/default_assumptions.h"
#include <nlohmann/json.hpp>

#include <set>
#include <stdexcept>

namespace sentryward::algorithm {

const std::vector<AssumptionDefinition>& AssumptionPool() {
  static const auto pool = [] {
    std::vector<AssumptionDefinition> result;
    std::set<std::string> labels, keys;
    for (const auto& item : nlohmann::json::parse(kAssumptionPoolJson)) {
      AssumptionDefinition entry{item.at("label"), item.at("key"), item.at("text")};
      if (!labels.insert(entry.label).second || !keys.insert(entry.key).second)
        throw std::invalid_argument("duplicate assumption label or key");
      result.push_back(std::move(entry));
    }
    return result;
  }();
  return pool;
}

PtxAssumptions DefaultPtxAssumptions() {
  static const PtxAssumptions defaults = [] {
    const auto value = nlohmann::json::parse(kDefaultAssumptionsJson);
    if (!value.is_object() || value.size() != AssumptionPool().size())
      throw std::invalid_argument("invalid default assumption keys");
    PtxAssumptions result;
    for (const auto& definition : AssumptionPool()) {
      const auto& name = definition.key;
      const auto& enabled = value.at(name);
      if (!enabled.is_boolean())
        throw std::invalid_argument("default assumption flags must be Boolean");
      result.emplace(name, enabled.get<bool>());
    }
    return result;
  }();
  return defaults;
}

void OverridePtxAssumptions(PtxAssumptions& target, const PtxAssumptions& overrides) {
  for (const auto& [name, enabled] : overrides) {
    if (!target.contains(name)) throw std::invalid_argument("unknown PTX semantic assumption: " + name);
    target.at(name) = enabled;
  }
}


}  // namespace sentryward::algorithm
