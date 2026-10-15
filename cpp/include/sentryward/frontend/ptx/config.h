#pragma once

#include <array>
#include <boost/multiprecision/cpp_int.hpp>
#include <cstdint>
#include <filesystem>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace sentryward::frontend::ptx {
struct Module;
struct Function;
using Integer = boost::multiprecision::cpp_int;
using Coordinates = std::array<std::uint32_t, 3>;
struct WarpContract {
  std::uint32_t warp_id{};
  std::string role, purpose;
  std::optional<std::string> entry_label;
  std::map<std::string, Integer> initial_registers;
};
struct LoopBound {
  std::string header;
  std::uint64_t trip_count{};
  std::optional<std::vector<std::uint32_t>> warp_ids;
  std::optional<std::uint32_t> backedge_line;
};
struct BranchOverride {
  std::uint32_t line{};
  std::vector<bool> outcomes;
  std::optional<std::uint32_t> warp_id, lane_id;
};
struct ParameterRegion {
  std::string name;
  std::uint64_t size_bytes{};
  std::optional<std::vector<std::uint8_t>> data;
  std::map<std::uint64_t, std::pair<std::uint32_t, Integer>> offset_values;
  nlohmann::json source;
  [[nodiscard]] std::optional<Integer>
  ReadInteger(std::int64_t offset, std::uint32_t width, bool is_signed) const;
};
struct PointerParameter {
  std::optional<std::string> parameter;
  std::optional<std::uint32_t> parameter_index;
  std::uint64_t byte_offset{};
  std::uint32_t width_bytes = 8;
  std::string resource;
  std::optional<std::uint64_t> extent_bytes;
};
struct FrontendConfig {
  nlohmann::json raw;
  std::string entry;
  Coordinates block_dim{};
  std::optional<Coordinates> cluster_dim;
  std::uint32_t warp_size = 32;
  std::vector<WarpContract> warps;
  nlohmann::json fixed_input, resources, instruction_contracts, tma_descriptors;
  std::map<std::string, Integer> parameters, special_registers;
  std::map<std::string, ParameterRegion> parameter_regions;
  std::vector<PointerParameter> pointer_parameters;
  std::vector<LoopBound> loop_bounds;
  std::vector<BranchOverride> branch_overrides;
  std::uint64_t max_steps_per_lane = 100000;
  std::optional<std::pair<std::uint32_t, std::uint32_t>> analysis_thread_range;
  bool compress_compute = true;
  bool compress_thread_records = true;
  std::optional<std::uint32_t> elect_lane;
  [[nodiscard]] std::uint64_t Threads() const;
  [[nodiscard]] const LoopBound *FindLoopBound(std::string_view header,
                                               std::uint32_t warp) const;
  [[nodiscard]] std::optional<bool> BranchOutcome(std::uint32_t line,
                                                  std::uint32_t warp,
                                                  std::uint32_t lane,
                                                  std::uint64_t visit) const;
  [[nodiscard]] const nlohmann::json &
  InstructionContract(std::uint32_t line, std::uint32_t instruction,
                      std::optional<std::uint32_t> warp = {},
                      std::optional<std::uint64_t> occurrence = {}) const;
};
[[nodiscard]] FrontendConfig
ParseConfig(nlohmann::json value, const std::filesystem::path &base_path = {});
[[nodiscard]] FrontendConfig BindLaunchObjects(const Module &, const Function &,
                                               const FrontendConfig &);
[[nodiscard]] std::pair<Coordinates, Coordinates>
LaunchDimensions(const FrontendConfig &config);
[[nodiscard]] FrontendConfig ConfigForCta(const FrontendConfig &config,
                                          Coordinates coordinates);
[[nodiscard]] Integer JsonInteger(const nlohmann::json &value);
} // namespace sentryward::frontend::ptx
