#pragma once
#include "sentryward/algorithm/event_ir/event_records.h"
#include "sentryward/algorithm/event_ir/edge_records.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "sentryward/algorithm/event_ir/event.h"
#include "sentryward/algorithm/event_ir/csr_adjacency.h"
#include "sentryward/algorithm/event_ir/event_order.h"

namespace sentryward::algorithm {

class EventDagBuilder;

class EventDag final {
 public:
  EventDag() = default;

  [[nodiscard]] std::size_t eventCount() const noexcept;
  [[nodiscard]] std::size_t edgeCount() const noexcept;
  [[nodiscard]] bool contains(EventId id) const noexcept;
  [[nodiscard]] Event event(EventId id) const;
  [[nodiscard]] const EventRecords& events() const noexcept;
  [[nodiscard]] const EventEdges& edges() const noexcept;
  [[nodiscard]] EventNeighbors predecessors(EventId id) const;
  [[nodiscard]] EventNeighbors successors(EventId id) const;
  [[nodiscard]] const std::shared_ptr<const CsrAdjacency>& adjacencyStorage()
      const noexcept { return adjacency_; }

  [[nodiscard]] std::optional<EventOrder> topologicalOrder() const;
  [[nodiscard]] bool isAcyclic() const;

 private:
  friend class EventDagBuilder;
  // Optional diagnostics inspect owned capacities without exposing mutable data.
  friend class MemoryCensus;

  EventDag(EventRecords events, EventEdges edges);
  [[nodiscard]] bool computeAcyclic() const;

  EventRecords events_;
  EventEdges edges_;
  std::shared_ptr<const CsrAdjacency> adjacency_;
  // Frozen graphs never change. Compute this once at construction so readers
  // can validate the graph without repeated O(V + E) work or mutable caches.
  bool acyclic_ = true;
  bool forward_ordered_ = true;
};

class EventDagBuilder final {
 public:
  EventDagBuilder()=default;
  explicit EventDagBuilder(EventRecords events);
  std::size_t beginThread(){return events_.beginThread();}
  bool compressThreadTail(std::size_t begin,std::uint64_t scope,std::uintptr_t history) {
    return events_.extendThreadFamily(begin,scope,history);
  }
  void reserve(std::size_t events, std::size_t edges);
  [[nodiscard]] EventId addEvent(EventSchedulingClass scheduling_class,
                                 std::optional<InstId> origin,
                                 std::optional<ActorId> actor,
                                 EventPayload payload);

  [[nodiscard]] bool addEdge(
      EventId from, EventId to, EventEdgeReason reason,
      std::optional<SourceEdgeProvenance> source_edge = std::nullopt);
  [[nodiscard]] bool addEdgeColumns(const std::vector<EventEdge>& columns,
      const std::vector<EventEdgeCodec::Delta>& deltas,std::size_t count);
  void beginEdgeGroup(std::uint64_t scope=0,std::uintptr_t history=0);
  void endEdgeGroup();

  [[nodiscard]] EventDag freeze() &&;

 private:
  EventRecords events_;
  EventEdges edges_;
  std::optional<std::size_t> edge_group_;
  std::pair<std::uint64_t,std::uintptr_t> edge_cohort_{};
};

}  // namespace sentryward::algorithm
