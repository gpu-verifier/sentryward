#pragma once

#include <cstdint>
#include <cstddef>
#include <unordered_map>
#include <memory>

#include "sentryward/algorithm/memory/access.h"

namespace sentryward::algorithm {

// Recognizes matching copy footprints, not equality of the values read.
class DuplicateTmaStores final {
 public:
  explicit DuplicateTmaStores(const SourceProgram& source);

  [[nodiscard]] bool contains(const MemoryAccess& first,
                              const MemoryAccess& second) const;
  // A matching copy in another lane exists; ordering is checked separately.
  [[nodiscard]] bool mayContain(const MemoryAccess& access) const;
  // Exact copy provenance class. Membership alone is not an ordering proof.
  [[nodiscard]] std::optional<std::size_t> copyClass(const MemoryAccess& access) const;

 private:
  struct Entry final {
    std::shared_ptr<const SourceInstruction> instruction;
    const ActorInfo* actor;
    std::size_t copy_class = 0;
    bool distinct_lanes = true;
  };
  std::unordered_map<std::uint32_t, Entry> entries_;
};

}  // namespace sentryward::algorithm
