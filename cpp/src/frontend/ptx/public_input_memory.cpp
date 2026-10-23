#include "sentryward/frontend/ptx/public_input.h"
#include "sentryward/algorithm/io/source_json.h"
#include <type_traits>

namespace sentryward::frontend::ptx {
void ValidateConstantInputs(const FrontendConfig &config,
                            const algorithm::SourceArtifact &artifact) {
  using namespace algorithm;
  using Json = nlohmann::json;
  struct Region {
    std::string resource, allocation;
    Integer begin, end;
    bool fixed_read_only;
    std::string path = "$.shape_arrays";
  };
  std::vector<Region> regions;
  const auto contract = config.fixed_input.value("shape_input_contract", Json::object());
  const auto contracted_regions = contract.value("kind", "") == "fixed_read_only" &&
          contract.value("origin", "") == "shape_arrays"
      ? contract.value("regions", Json::array()) : Json::array();
  for (const auto &entry : config.fixed_input.value("global_memory_inputs", nlohmann::json::array())) {
    const auto name = entry.at("resource").get<std::string>();
    const auto resource = std::find_if(config.resources.begin(), config.resources.end(),
        [&](const auto &r) { return r.at("id") == name; });
    if (resource == config.resources.end() || resource->value("space", "") != "global" ||
        !resource->contains("allocation"))
      throw InputError("invalid_constant_input", "$.shape_arrays", "constant input has no global allocation");
    const auto &allocation = resource->at("allocation");
    const Integer begin = JsonInteger(allocation.value("base_byte_offset", nlohmann::json(0))) +
        JsonInteger(entry.value("byte_offset", nlohmann::json(0)));
    const Integer bytes = entry.at("values").size() * entry.at("element_bits").get<unsigned>() / 8;
    const auto object = allocation.at("object").get<std::string>();
    const bool fixed_read_only = std::any_of(contracted_regions.begin(), contracted_regions.end(),
        [&](const auto &region) {
          return region.value("resource", "") == name && region.value("allocation", "") == object &&
              region.contains("byte_offset") && JsonInteger(region.at("byte_offset")) == begin &&
              region.contains("bytes") && JsonInteger(region.at("bytes")) == bytes;
        });
    regions.push_back({name, object, begin, begin + bytes, fixed_read_only});
  }
  // Pointer tables are read-only input memory under their own caller contract.
  const auto pointer_contract = config.fixed_input.value("pointer_array_contract", Json::object());
  if (pointer_contract.value("kind", "") == "fixed_read_only" && pointer_contract.value("origin", "") == "pointer_arrays")
    for (const auto &region : pointer_contract.value("regions", Json::array())) {
      const Integer begin = JsonInteger(region.at("byte_offset"));
      regions.push_back({region.at("resource").get<std::string>(), region.at("allocation").get<std::string>(),
                         begin, begin + JsonInteger(region.at("bytes")), true, "$.pointer_arrays"});
    }
  // Objects with tensor properties are read-only input memory as a whole.
  const auto property_contract = config.fixed_input.value("tensor_property_contract", Json::object());
  if (property_contract.value("kind", "") == "fixed_read_only" && property_contract.value("origin", "") == "tensor_properties")
    for (const auto &region : property_contract.value("regions", Json::array()))
      regions.push_back({region.at("resource").get<std::string>(), region.at("allocation").get<std::string>(),
                         0, Integer(1) << 63, true, "$.tensor_properties"});
  if (regions.empty()) return;
  std::map<AddressId,const ConstantAddressBinding *> addresses;
  for (const auto &binding : artifact.addresses) addresses[binding.address] = &binding;
  auto check = [&](AddressId address, std::optional<std::uint64_t> bytes) {
    if (bytes == 0) return;
    const auto &binding = *addresses.at(address);
    if (binding.address_space != "global" && binding.address_space != "generic") return;
    std::optional<std::string> identity;
    if (!binding.expression.empty() && binding.address_space=="global") {
      const auto expression=nlohmann::json::parse(binding.expression);
      const auto base=expression.value("address",nlohmann::json::object()).value("base",nlohmann::json());
      if (base.is_object() && base.contains("identity")) {
        identity = base.at("identity").get<std::string>();
        if (std::none_of(regions.begin(),regions.end(),[&](const auto &r){return *identity==r.allocation;})) return;
      }
    }
    if (binding.expression.empty() && bytes)
      for (const auto &region : regions)
        if (binding.allocation == region.allocation && Integer(binding.offset) < region.end &&
            region.begin < Integer(binding.offset) + *bytes)
          throw InputError("constant_input_modified", region.path, "kernel writes fixed input bytes of " + region.resource);
    // An abstract write into an object with tensor_properties would void its element premise.
    if (!binding.expression.empty() && identity &&
        std::any_of(regions.begin(), regions.end(), [&](const auto &region) {
          return region.path == "$.tensor_properties" && region.allocation == *identity;
        }))
      throw InputError("constant_input_modified", "$.tensor_properties",
                       "kernel may write an object with tensor_properties");
    if (!binding.expression.empty() || !bytes || binding.address_space == "generic") {
      // Public shape regions are a caller contract, not a derived non-aliasing fact.
      if (std::any_of(regions.begin(), regions.end(), [&](const auto &region) {
            return !region.fixed_read_only && (!identity || *identity == region.allocation);
          }))
        throw InputError("constant_input_write_unresolved", "$.shape_arrays",
            "cannot prove that an abstract write preserves fixed shape input bytes", "ptx_execution_unsupported");
    }
  };
  for (const auto &instruction : artifact.program.instructions)
    std::visit([&](const auto &op) {
      using T = std::decay_t<decltype(op)>;
      if constexpr (std::is_same_v<T,ImmediateStore>) check(op.address, op.bytes);
      else if constexpr (std::is_same_v<T,AsyncOp>)
        for (const auto &effect : op.effects)
          for (const auto &access : effect.accesses)
            if (access.mode == AccessMode::kWrite)
              if (const auto *address = std::get_if<AddressId>(&access.endpoint)) check(*address, access.bytes);
    }, instruction.op);
}
} // namespace sentryward::frontend::ptx
