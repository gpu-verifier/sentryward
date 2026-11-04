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

  // This graph less the edges at `removed` (ascending indices into edges()): the same events and
  // IDs and the other edges in their order. The edges before the first removed one keep their
  // storage; the later ones are appended one by one, as EventDagBuilder::addEdge outside an edge
  // group appends them. Removing edges keeps an acyclic graph acyclic.
  // O(stored rows and blocks + (edges after the first removed one) log blocks), plus the adjacency.
  [[nodiscard]] EventDag without(std::span<const std::size_t> removed) const;

 private:
  friend class EventDagBuilder;
  // Optional diagnostics inspect owned capacities without exposing mutable data.
  friend class MemoryCensus;

  // `acyclic`: the edges are known to form no cycle (otherwise this is computed).
  EventDag(EventRecords events, EventEdges edges, bool acyclic = false);
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
  [[nodiscard]] std::size_t eventCount() const noexcept{return events_.size();}
  [[nodiscard]] std::size_t edgeCount() const noexcept{return edges_.size();}
  // Ordered iterations of the current thread's events (TemplateRecords::compressIterations).
  void compressIterations(std::size_t begin,const std::vector<std::pair<std::size_t,std::uint32_t>>& starts) {
    events_.compressIterations(begin,starts);
  }
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
