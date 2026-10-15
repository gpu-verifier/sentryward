#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#include "sentryward/algorithm/base/dynamic_bitset.h"
#include "sentryward/algorithm/memory/access.h"
#include "sentryward/algorithm/memory/model.h"

namespace sentryward::algorithm {

struct AccessPair final {
  AccessId first;
  AccessId second;

  friend bool operator==(const AccessPair&, const AccessPair&) = default;
};

struct UnorderedAccessPairsOptions final {
  // Retained rows and FIFO entries; the linear event index is additional.
  std::size_t ancestor_budget_bytes = 64U * 1024U * 1024U;
};

// The graph must outlive the iterator and remain unchanged. Access IDs must
// equal their table indexes, and the table must cover every execution operand.
class UnorderedAccessPairs final {
 public:
  // Throws std::invalid_argument for an invalid graph or access table. The
  // access table need not outlive the iterator.
  UnorderedAccessPairs(const HappensBeforeGraph& graph,
                       MemoryAccessView accesses,
                       const UnorderedAccessPairsOptions& options = {},
                       std::unique_ptr<MemoryStorageFilter> storage_filter = {});

  // The first access belongs to the earlier topological event, not necessarily
  // the smaller AccessId. Throws std::logic_error if the graph has changed.
  [[nodiscard]] std::optional<AccessPair> next();
  // Reserved FIFO entries and retained rows; excludes linear working storage.
  [[nodiscard]] std::size_t retainedAncestorCacheBytes() const noexcept;

 private:
  struct AncestorCacheEntry final {
    EventId event;
    DynamicBitset ancestors;
  };

  void checkGraphUnchanged() const;
  void prepareEvent();
  void prepareAccess();
  void finishEvent();
  [[nodiscard]] const DynamicBitset& ancestorsOf(EventId event);
  [[nodiscard]] const DynamicBitset& currentCandidates() const;

  const HappensBeforeGraph& graph_;
  std::size_t graph_version_;
  std::size_t graph_event_count_;
  std::size_t graph_edge_count_;
  std::vector<std::vector<AccessId>> event_accesses_;
  std::vector<bool> writes_;
  std::vector<EventId> memory_order_;
  std::vector<DynamicBitset> ancestors_;
  bool lazy_ancestors_ = false;
  DynamicBitset current_ancestors_;
  DynamicBitset visited_;
  std::vector<EventId> traversal_;
  std::size_t ancestor_cache_capacity_ = 0;
  std::size_t ancestor_row_bytes_ = 0;
  std::size_t ancestor_cache_size_ = 0;
  std::size_t ancestor_cache_next_ = 0;
  std::vector<AncestorCacheEntry> ancestor_cache_;
  std::vector<std::size_t> ancestor_cache_slots_;
  DynamicBitset seen_all_;
  DynamicBitset seen_writes_;
  DynamicBitset write_candidates_;
  DynamicBitset read_candidates_;
  std::unique_ptr<MemoryStorageFilter> storage_filter_;
  DynamicBitset compatible_candidates_;
  const DynamicBitset* event_ancestors_ = nullptr;
  std::size_t event_cursor_ = 0;
  std::size_t access_cursor_ = 0;
  std::size_t next_candidate_ = DynamicBitset::npos;
  bool event_prepared_ = false;
};

}  // namespace sentryward::algorithm
