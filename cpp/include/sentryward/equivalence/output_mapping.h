#pragma once

#include <span>
#include <string>
#include <vector>
#include "sentryward/scalar_fold/artifact.h"

namespace sentryward::equivalence {

struct OutputRangeMapping {
  std::string left_allocation, right_allocation;
  std::uint64_t left_offset=0, right_offset=0, bytes=0;
};
struct TensorOutputMapping {
  std::string left_allocation, right_allocation;
  std::uint64_t left_offset=0, right_offset=0, element_bytes=0;
  std::vector<std::uint64_t> shape, left_strides_bytes, right_strides_bytes;
};
struct OutputPairing {
  bool complete=false;
  std::string reason;
  std::vector<scalar_fold::Ref> left, right;
  std::uint64_t covered_bytes=0;
};
// Covers every declared byte interval exactly, without enumerating bytes.
// Input allocation and parameter identities are aligned separately.
OutputPairing pairOutputs(std::span<const scalar_fold::Output> left,
                          std::span<const scalar_fold::Output> right,
                          std::span<const OutputRangeMapping> mapping);

// Dense axis permutations; requires full tensor coverage using recorded roots.
OutputPairing pairTensorOutputs(std::span<const scalar_fold::Output> left,
                                std::span<const scalar_fold::Output> right,
                                const TensorOutputMapping&);

// Each allocation occurs once; all declared outputs must be covered.
OutputPairing pairTensorOutputs(std::span<const scalar_fold::Output> left,
                                std::span<const scalar_fold::Output> right,
                                std::span<const TensorOutputMapping>);

} // namespace sentryward::equivalence
