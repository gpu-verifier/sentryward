#pragma once

#include <array>
#include <map>
#include <optional>
#include <vector>

#include "sentryward/frontend/ptx/layouts.h"
#include "sentryward/frontend/ptx/resources.h"
#include "sentryward/frontend/ptx/unit_facts.h"

namespace sentryward::frontend::ptx {
// One instance per specialized CTA. The facts, their program and its lane
// storage must outlive the resolver and remain unchanged while cached
// occurrences exist. For ldmatrix/stmatrix kernels the constructor expands
// every lane (T2).
class MatrixResolver {
public:
  explicit MatrixResolver(const CtaFacts &facts);
  [[nodiscard]] std::vector<Fragment>
  Resolve(LaneRef current, const RecordView &execution,
          Resources &resources, const nlohmann::json &contract);

private:
  struct Key {
    std::uint32_t warp;
    DynamicKey dynamic;
    auto operator<=>(const Key &) const = default;
  };
  // Per physical lane id: the recording lane and the position of its record
  // (an expanded record; positions[l] is meaningful iff lanes[l] is engaged).
  struct Occurrence {
    std::array<std::optional<LaneRef>, 32> lanes{};
    std::array<std::size_t, 32> positions{};
    std::vector<std::uint64_t> rows;
  };
  const CtaFacts &facts_;
  std::map<Key, Occurrence> occurrences_;
  std::map<std::uint32_t, MatrixShape> shapes_;
};
} // namespace sentryward::frontend::ptx
