#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include "sentryward/algorithm/analysis/happens_before_graph.h"

namespace sentryward::algorithm {

// Shares equal adjacency rows, without identifying their original vertices.
// In particular, StateFor(a) == StateFor(b) does NOT imply that a reaches b.
// A search must test each literal EventId against its targets BEFORE using
// StateFor() to suppress expansion of an already visited state. Its reflexive
// root must likewise be tested as an EventId, not as a shared state.
//
// The referenced graph must outlive this index and only change through its
// append-only builder. Call SyncGraphChanges() between mutation and traversal;
// the graph must remain unchanged throughout a traversal.
class SharedReachabilityIndex final {
 public:
  using StateId = std::uint32_t;
  // Receives a sorted, deduplicated adjacency row. Collisions never affect
  // correctness: grouping always compares the complete rows as well.
  using RowHasher = std::uint64_t (*)(std::span<const EventId>);
  using NeighborProvider =
      std::function<EventNeighbors(EventId)>;

  explicit SharedReachabilityIndex(const HappensBeforeGraph& graph,
                                   bool reverse = false,
                                   RowHasher hasher = nullptr);
  // Builds an independent immutable index from dense EventIds [0, count).
  // The callback is used only during construction and is not retained.
  SharedReachabilityIndex(std::size_t count, NeighborProvider neighbors,
                          RowHasher hasher = nullptr);

  [[nodiscard]] StateId StateFor(EventId event) const;
  [[nodiscard]] std::size_t StateCount() const noexcept;
  // Original EventIds, rather than destination states, preserve exact query
  // endpoints even when vertices with different incoming edges share a row.
  [[nodiscard]] EventNeighbors Neighbors(StateId state) const;

  // Scans only appended event/edge records. A vertex with changed adjacency
  // leaves its immutable base group once and uses the graph's live row from
  // then on. A provenance-only edge record does not create a new state.
  void SyncGraphChanges();

  [[nodiscard]] bool reverse() const noexcept;
  [[nodiscard]] std::size_t baseStateCount() const noexcept;
  [[nodiscard]] std::size_t storedNeighborCount() const noexcept;
  // Owned vector capacity; excludes the shared graph, allocator overhead and
  // temporary construction storage. Empty base states can remain after splits.
  [[nodiscard]] std::size_t storageBytes() const noexcept;

 private:
  void build(std::size_t count, const NeighborProvider& neighbors,
             RowHasher hasher);
  [[nodiscard]] EventNeighbors adjacency(EventId event) const;
  [[nodiscard]] StateId appendSingleton(EventId event);

  const HappensBeforeGraph* graph_ = nullptr;
  bool reverse_ = false;
  std::size_t observed_version_ = 0;
  std::size_t observed_edge_count_ = 0;
  std::vector<StateId> state_of_;
  std::vector<std::size_t> row_offsets_;
  std::vector<EventId> row_neighbors_;
  // State IDs >= baseStateCount() refer to live singleton rows in this array.
  std::vector<EventId> singletons_;
};

}  // namespace sentryward::algorithm
