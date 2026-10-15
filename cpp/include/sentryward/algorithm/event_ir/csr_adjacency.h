#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "sentryward/algorithm/base/strong_id.h"
#include "sentryward/algorithm/event_ir/edge_records.h"
#include "sentryward/algorithm/event_ir/neighbors.h"

namespace sentryward::algorithm {

struct EventEdge;
class EventDag;
class MemoryCensus;

// Immutable CSR storage shared by a lowered DAG and its proof graphs. Keeping
// this owner separate from events/edge provenance lets graph views retain
// adjacency safely after the original LoweredProgram has been destroyed.
class CsrAdjacency final {
 public:
  [[nodiscard]] std::size_t eventCount() const noexcept {
    return count_;
  }
  [[nodiscard]] const EventRelations* relations() const noexcept {
    return relations_.get();
  }
  [[nodiscard]] const std::shared_ptr<const EventRelations>& relationStorage() const noexcept {return relations_;}
  [[nodiscard]] EventNeighbors predecessors(EventId id) const {
    if (relations_) return EventNeighbors(*relations_, id, true);
    const auto index = static_cast<std::size_t>(id.value());
    const auto end = predecessor_offsets_.at(index + 1);
    return std::span<const EventId>(predecessors_).subspan(
        predecessor_offsets_[index], end - predecessor_offsets_[index]);
  }
  [[nodiscard]] EventNeighbors successors(EventId id) const {
    if (relations_) return EventNeighbors(*relations_, id, false);
    const auto index = static_cast<std::size_t>(id.value());
    const auto end = successor_offsets_.at(index + 1);
    return std::span<const EventId>(successors_).subspan(
        successor_offsets_[index], end - successor_offsets_[index]);
  }

 private:
  friend class EventDag;
  friend class MemoryCensus;
  CsrAdjacency(const EventRecords& events, const EventEdges& edges);

  std::size_t count_ = 0;
  std::shared_ptr<const EventRelations> relations_;

  std::vector<std::size_t> predecessor_offsets_, successor_offsets_;
  std::vector<EventId> predecessors_, successors_;
};

}  // namespace sentryward::algorithm
