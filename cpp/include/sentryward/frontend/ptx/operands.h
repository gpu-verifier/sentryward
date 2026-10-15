#pragma once
#include "sentryward/frontend/ptx/module.h"
namespace sentryward::frontend::ptx {
[[nodiscard]] std::vector<std::string> Registers(std::string_view text);
[[nodiscard]] std::vector<std::string> Destinations(const Module &,
                                                    const Instruction &);
[[nodiscard]] std::optional<std::uint32_t> MemoryWidth(const Instruction &);
[[nodiscard]] std::optional<std::uint32_t> AddressLanes(const Instruction &);
} // namespace sentryward::frontend::ptx
