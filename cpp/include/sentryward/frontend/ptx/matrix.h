#pragma once

#include <array>
#include <map>
#include <vector>

#include "sentryward/frontend/ptx/layouts.h"
#include "sentryward/frontend/ptx/resources.h"

namespace sentryward::frontend::ptx {
// One instance per specialized CTA. The ExecutionProgram and its lane storage
// must outlive the resolver and remain unchanged while cached occurrences
// exist.
class MatrixResolver {
public:
  explicit MatrixResolver(const ExecutionProgram &program);
  [[nodiscard]] std::vector<Fragment>
  Resolve(const LaneTrace &current, const ExecutedInstruction &execution,
          Resources &resources, const nlohmann::json &contract);

private:
  struct Key {
    std::uint32_t warp;
    DynamicKey dynamic;
    auto operator<=>(const Key &) const = default;
  };
  struct Occurrence {
    std::array<const LaneTrace *, 32> lanes{};
    std::array<const ExecutedInstruction *, 32> executions{};
    std::vector<std::uint64_t> rows;
  };
  const ExecutionProgram &program_;
  std::map<Key, Occurrence> occurrences_;
  std::map<std::uint32_t, MatrixShape> shapes_;
};
} // namespace sentryward::frontend::ptx
