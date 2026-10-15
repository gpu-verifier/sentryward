#pragma once

#include <cstdint>
#include <iosfwd>
#include <span>
#include <string>
#include <vector>

#include "sentryward/scalar_fold/graph.h"
#include "sentryward/scalar_fold/output_family.h"

namespace sentryward::scalar_fold {

struct Output {
  std::string allocation;
  std::uint64_t offset = 0, bytes = 0;
  Ref value;
};

struct Artifact {
  Graph graph;
  std::vector<Output> outputs;
  std::vector<OutputFamily> output_families;
};

}  // namespace sentryward::scalar_fold
