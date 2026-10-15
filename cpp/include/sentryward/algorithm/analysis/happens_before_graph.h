#pragma once
#include "sentryward/algorithm/event_ir/event_records.h"

#include <cstddef>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include "sentryward/algorithm/base/result.h"
#include "sentryward/algorithm/analysis/adjacency_overlay.h"
#include "sentryward/algorithm/analysis/trace_order.h"
#include "sentryward/algorithm/event_ir/dag.h"
#include "sentryward/algorithm/event_ir/program.h"
#include "sentryward/algorithm/semantics/barrier_state.h"

namespace sentryward::algorithm {

enum class HappensBeforeEdgeReason {
  kSourceProgramOrder,
  kSourceCompletionDependency,
  kInternalIssueToDone,
  kStateDefUse,
  kCollectiveArrival,
  kCollectiveRelease,
  kCertifiedPreviousPhase,
  kCertifiedPhaseCompletion,
  kWaitRelease,
  kAsyncEffectDependency,
  kAsyncAggregate,
  kAwaitRequirement,
  kEndpointDependency,
};

struct CertificateRef final {
  PhaseKey phase;

  friend bool operator==(const CertificateRef&,
                         const CertificateRef&) = default;
};

struct HappensBeforeEdge final {
  EventId from;
  EventId to;
  HappensBeforeEdgeReason reason;
  std::optional<SourceEdgeProvenance> source_edge;
  std::optional<CertificateRef> certificate;

  friend bool operator==(const HappensBeforeEdge&,
                         const HappensBeforeEdge&) = default;
};

struct HappensBeforeEdgeCodec final {
  using Binding = EventEdgeCodec::Binding;
  using Delta = EventEdgeCodec::Delta;
  static bool split(HappensBeforeEdge& edge, Binding& binding) {
    binding = {edge.from.value(), edge.to.value(),
        edge.source_edge ? edge.source_edge->from.value() : 0,
        edge.source_edge ? edge.source_edge->to.value() : 0};
    edge.from = EventId{}; edge.to = EventId{};
    if (edge.source_edge) edge.source_edge = SourceEdgeProvenance{};
    return true;
  }
  static void bind(HappensBeforeEdge& edge, const Binding& binding) {
    edge.from = EventId(binding[0]); edge.to = EventId(binding[1]);
    if (edge.source_edge) edge.source_edge = SourceEdgeProvenance{InstId(binding[2]), InstId(binding[3])};
  }
  static auto difference(const Binding& a, const Binding& b) { return EventEdgeCodec::difference(a, b); }
  static auto translate(Binding b, const Delta& d, std::size_t i) { return EventEdgeCodec::translate(b, d, i); }
  static std::size_t hash(const HappensBeforeEdge& edge) {
    std::size_t h = static_cast<std::size_t>(edge.reason);
    HashCombine(h, edge.source_edge.has_value());
    if (edge.certificate) {
      HashCombine(h, edge.certificate->phase.barrier.value());
      HashCombine(h, edge.certificate->phase.phase);
    }
    return h;
  }
};
using HappensBeforeEdges = TemplateRecords<HappensBeforeEdge, HappensBeforeEdgeCodec>;

struct DerivedEdgeJustification final {
  EventId from;
  EventId to;
  HappensBeforeEdgeReason reason;
  CertificateRef certificate;

  friend bool operator==(const DerivedEdgeJustification&,
                         const DerivedEdgeJustification&) = default;
};

struct DerivedEdgeCodec final {
  struct Binding { EventId from, to; friend bool operator==(const Binding&, const Binding&) = default; };
  using Delta = std::array<std::int64_t, 2>;
  static bool split(DerivedEdgeJustification& value, Binding& binding) {
    binding = {value.from, value.to}; value.from = EventId{}; value.to = EventId{}; return true;
  }
  static void bind(DerivedEdgeJustification& value, const Binding& binding) {
    value.from = binding.from; value.to = binding.to;
  }
  static std::optional<Delta> difference(const Binding& a, const Binding& b) {
    return Delta{std::int64_t(b.from.value()) - a.from.value(), std::int64_t(b.to.value()) - a.to.value()};
  }
  static std::optional<Binding> translate(const Binding& base, const Delta& delta, std::size_t i) {
    const auto from = EventIdCodec::translate(base.from, delta[0], i);
    const auto to = EventIdCodec::translate(base.to, delta[1], i);
    if (!from || !to) return {};
    return Binding{*from, *to};
  }
  static std::size_t hash(const DerivedEdgeJustification& value) {
    std::size_t result = static_cast<std::size_t>(value.reason);
    HashCombine(result, value.certificate.phase.barrier.value());
    HashCombine(result, value.certificate.phase.phase);
    return result;
  }
};
using DerivedEdgeJustifications = TemplateRecords<DerivedEdgeJustification, DerivedEdgeCodec>;

enum class HappensBeforeGraphMutationErrorCode {
  kFrozen,
  kMissingEvent,
  kInvalidReason,
  kCycle,
  kDuplicatePhaseCompletion,
  kEventIdOverflow,
  kTraceOrderViolation,
};

struct HappensBeforeGraphMutationError final {
  HappensBeforeGraphMutationErrorCode code;
  std::optional<EventId> from;
  std::optional<EventId> to;
  std::optional<PhaseKey> phase;

  friend bool operator==(const HappensBeforeGraphMutationError&,
                         const HappensBeforeGraphMutationError&) = default;
};

class HappensBeforeGraphBuilder;

class HappensBeforeGraph final {
 public:
  HappensBeforeGraph() = default;

  [[nodiscard]] std::size_t eventCount() const noexcept;
  [[nodiscard]] std::size_t edgeCount() const noexcept;
  [[nodiscard]] std::size_t version() const noexcept;
  [[nodiscard]] const TraceOrder* traceOrder() const noexcept {
    return trace_order_ ? &*trace_order_ : nullptr;
  }
  [[nodiscard]] bool contains(EventId id) const noexcept;
  [[nodiscard]] Event event(EventId id) const;
  [[nodiscard]] const EventRecords& events() const noexcept;
  [[nodiscard]] const std::vector<ActorInfo>& actors() const noexcept { return actors_; }
  [[nodiscard]] const HappensBeforeEdges& edges() const noexcept;
  [[nodiscard]] std::shared_ptr<const EventRelations> relationStorage() const;
  [[nodiscard]] EventNeighbors predecessors(EventId id) const;
  [[nodiscard]] EventNeighbors successors(EventId id) const;
  [[nodiscard]] std::optional<EventId> phaseCompletion(PhaseKey phase) const;
  [[nodiscard]] const std::map<PhaseKey, EventId>& phaseCompletions()
      const noexcept {
    return phase_completions_;
  }
  [[nodiscard]] const DerivedEdgeJustifications& justifications()
      const noexcept;

  [[nodiscard]] std::optional<EventOrder> topologicalOrder() const;
  [[nodiscard]] bool isAcyclic() const;

  // Returns a topology-compatible snapshot for generic graph analyses.
  [[nodiscard]] const EventDag& graph() const;

 private:
  friend class HappensBeforeGraphBuilder;

  void invalidateSnapshot() noexcept;
  void rebuildSnapshot() const;

  std::optional<TraceOrder> trace_order_;
  EventRecords events_;
  std::vector<ActorInfo> actors_;
  HappensBeforeEdges edges_;
  std::shared_ptr<const EventRelations> base_relations_;
  mutable std::shared_ptr<const EventRelations> relation_snapshot_;
  // Copies retain the immutable snapshot; each graph invalidates its own pointer.
  // Sharing the publication mutex also makes concurrent first readers safe.
  std::shared_ptr<std::mutex> relation_mutex_=std::make_shared<std::mutex>();
  AdjacencyOverlay predecessors_;
  AdjacencyOverlay successors_;
  std::map<PhaseKey, EventId> phase_completions_;
  DerivedEdgeJustifications justifications_;
  std::size_t version_ = 0;
  bool forward_ordered_ = true;
  mutable EventDag topology_snapshot_;
  mutable std::optional<std::size_t> topology_snapshot_version_;
  mutable std::optional<std::pair<std::size_t, bool>> acyclic_cache_;
};

class HappensBeforeGraphBuilder final {
 public:
  explicit HappensBeforeGraphBuilder(const LoweredProgram& program,
                                     const ReferenceTrace* trace = nullptr);

  [[nodiscard]] const HappensBeforeGraph& happensBeforeGraph() const noexcept;
  [[nodiscard]] bool Reaches(EventId from, EventId to) const;

  [[nodiscard]] Result<void, HappensBeforeGraphMutationError> AddCertifiedEdge(
      EventId from, EventId to, HappensBeforeEdgeReason reason,
      CertificateRef certificate);

  // A shared proof covers these distinct edges; it adds no ordering between members.
  [[nodiscard]] Result<void, HappensBeforeGraphMutationError> AddCertifiedEdges(
      std::span<const std::pair<EventId, EventId>> edges,
      HappensBeforeEdgeReason reason, CertificateRef certificate);

  [[nodiscard]] Result<EventId, HappensBeforeGraphMutationError>
  AddPhaseCompletion(PhaseKey phase);

  [[nodiscard]] const HappensBeforeGraph& Freeze() noexcept;
  // Consume the completed builder without copying the full proof graph.
  [[nodiscard]] HappensBeforeGraph TakeFrozen() &&;
  [[nodiscard]] bool frozen() const noexcept;

 private:
  [[nodiscard]] bool hasAdjacency(EventId from, EventId to) const;
  void appendAdjacency(EventId from, EventId to);

  HappensBeforeGraph happensBeforeGraph_;
  std::set<std::tuple<EventId, EventId, HappensBeforeEdgeReason, PhaseKey>>
      certified_edges_;
  bool frozen_ = false;
};

}  // namespace sentryward::algorithm
