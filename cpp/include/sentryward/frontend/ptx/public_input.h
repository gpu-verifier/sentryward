#pragma once

#include "sentryward/frontend/ptx/config.h"
#include <stdexcept>
#include <string>

namespace sentryward::algorithm { struct SourceArtifact; }

namespace sentryward::frontend::ptx {
struct InputError : std::runtime_error {
  std::string code, path, status;
  InputError(std::string code, std::string path, std::string message,
             std::string status = "invalid_configuration");
};

struct KernelInput {
  nlohmann::json fields;
};

struct InputOptions {
  std::uint64_t max_steps_per_thread = 100000;
  bool compress_compute = true;
  bool compress_thread_records = true;
  std::uint32_t elect_lane = 0;
};

struct InferredPointer {
  std::string parameter;
  std::uint64_t offset = 0;
  unsigned width_bytes = 8;
};

[[nodiscard]] KernelInput ParseKernelInput(const nlohmann::json &);
// `contents`: the file's bytes, null when it cannot be opened.
[[nodiscard]] KernelInput LoadKernelInput(const std::filesystem::path &, const std::string *contents);
[[nodiscard]] FrontendConfig NormalizeKernelInput(
    const KernelInput &, const Module &, const InputOptions & = {},
    const std::vector<InferredPointer> & = {});
[[nodiscard]] std::vector<InferredPointer> InferPointerFields(
    const Module &, const Function &, const FrontendConfig &);
void ValidateConstantInputs(const FrontendConfig &, const algorithm::SourceArtifact &);
} // namespace sentryward::frontend::ptx
