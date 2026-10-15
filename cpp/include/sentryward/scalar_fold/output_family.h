#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "sentryward/scalar_fold/graph.h"

namespace sentryward::scalar_fold {
// Index levels [0,shape.size()) bind output coordinates; inner Folds follow them.
struct OutputFamily {
  std::string allocation;
  std::uint64_t offset=0, element_bytes=0;
  std::vector<std::uint64_t> shape, strides_bytes;
  Ref body;
};
}
