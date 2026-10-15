#include "sentryward/algorithm/diagnostics/performance_profile.h"
#include "sentryward/algorithm/analysis/happens_before_graph.h"
#include "sentryward/algorithm/semantics/template_schedule.h"
#include "sentryward/algorithm/base/dynamic_bitset.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <queue>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace sentryward::algorithm {

namespace {

std::size_t indexOf(EventId id) { return static_cast<std::size_t>(id.value()); }

HappensBeforeEdgeReason toHappensBeforeReason(EventEdgeReason reason) {
  switch (reason) {
    case EventEdgeReason::kSourceProgramOrder:
      return HappensBeforeEdgeReason::kSourceProgramOrder;
    case EventEdgeReason::kSourceCompletionDependency:
      return HappensBeforeEdgeReason::kSourceCompletionDependency;
    case EventEdgeReason::kInternalIssueToDone:
      return HappensBeforeEdgeReason::kInternalIssueToDone;
    case EventEdgeReason::kStateDefUse:
      return HappensBeforeEdgeReason::kStateDefUse;
    case EventEdgeReason::kCollectiveArrival:
      return HappensBeforeEdgeReason::kCollectiveArrival;
    case EventEdgeReason::kCollectiveRelease:
      return HappensBeforeEdgeReason::kCollectiveRelease;
    case EventEdgeReason::kAsyncEffectDependency:
      return HappensBeforeEdgeReason::kAsyncEffectDependency;
    case EventEdgeReason::kAsyncAggregate:
      return HappensBeforeEdgeReason::kAsyncAggregate;
    case EventEdgeReason::kAwaitRequirement:
      return HappensBeforeEdgeReason::kAwaitRequirement;
    case EventEdgeReason::kEndpointDependency:
      return HappensBeforeEdgeReason::kEndpointDependency;
  }
  throw std::logic_error("unhandled event edge reason");
}

EventEdgeReason toTopologyReason(HappensBeforeEdgeReason reason) {
  switch (reason) {
    case HappensBeforeEdgeReason::kSourceProgramOrder:
      return EventEdgeReason::kSourceProgramOrder;
    case HappensBeforeEdgeReason::kSourceCompletionDependency:
      return EventEdgeReason::kSourceCompletionDependency;
    case HappensBeforeEdgeReason::kInternalIssueToDone:
      return EventEdgeReason::kInternalIssueToDone;
    case HappensBeforeEdgeReason::kStateDefUse:
      return EventEdgeReason::kStateDefUse;
    case HappensBeforeEdgeReason::kCollectiveArrival:
      return EventEdgeReason::kCollectiveArrival;
    case HappensBeforeEdgeReason::kCollectiveRelease:
      return EventEdgeReason::kCollectiveRelease;
    case HappensBeforeEdgeReason::kAsyncEffectDependency:
      return EventEdgeReason::kAsyncEffectDependency;
    case HappensBeforeEdgeReason::kAsyncAggregate:
      return EventEdgeReason::kAsyncAggregate;
    case HappensBeforeEdgeReason::kAwaitRequirement:
      return EventEdgeReason::kAwaitRequirement;
    case HappensBeforeEdgeReason::kEndpointDependency:
      return EventEdgeReason::kEndpointDependency;
    case HappensBeforeEdgeReason::kCertifiedPreviousPhase:
    case HappensBeforeEdgeReason::kCertifiedPhaseCompletion:
    case HappensBeforeEdgeReason::kWaitRelease:
      return EventEdgeReason::kInternalIssueToDone;
  }
  throw std::logic_error("unhandled happens-before edge reason");
}

bool isDerivedReason(HappensBeforeEdgeReason reason) {
  return reason == HappensBeforeEdgeReason::kCertifiedPreviousPhase ||
         reason == HappensBeforeEdgeReason::kCertifiedPhaseCompletion ||
         reason == HappensBeforeEdgeReason::kWaitRelease;
}

HappensBeforeGraphMutationError mutationError(
    HappensBeforeGraphMutationErrorCode code,
    std::optional<EventId> from = std::nullopt,
    std::optional<EventId> to = std::nullopt,
    std::optional<PhaseKey> phase = std::nullopt) {
  return HappensBeforeGraphMutationError{code, from, to, phase};
}

// Keep unique source rows shared. Only rows with parallel edges need an owned
// deduplicated representation; their first-occurrence order is unchanged.
void DeduplicateAdjacency(AdjacencyOverlay& rows, std::size_t index,
                          DynamicBitset& seen) {
  const auto input = rows.at(index);
  if (input.size() < 2) return;
  bool duplicate = false;
  for (EventId neighbor : input) {
    const auto neighbor_index = indexOf(neighbor);
    duplicate |= seen.test(neighbor_index);
    seen.set(neighbor_index);
  }
  for (EventId neighbor : input) seen.reset(indexOf(neighbor));
  if (!duplicate) return;

  std::vector<EventId> output;
  output.reserve(input.size());
  for (EventId neighbor : input) {
    const auto index = indexOf(neighbor);
    if (!seen.test(index)) {
      seen.set(index);
      output.push_back(neighbor);
    }
  }
  for (EventId neighbor : output) {
    seen.reset(indexOf(neighbor));
  }
  rows.replace(index, std::move(output));
}

}  // namespace

std::size_t HappensBeforeGraph::eventCount() const noexcept {
  return events_.size();
}

std::size_t HappensBeforeGraph::edgeCount() const noexcept {
  return edges_.size();
}

std::size_t HappensBeforeGraph::version() const noexcept { return version_; }

bool HappensBeforeGraph::contains(EventId id) const noexcept {
  return indexOf(id) < events_.size();
}

Event HappensBeforeGraph::event(EventId id) const {
  return events_.at(indexOf(id));
}

const EventRecords& HappensBeforeGraph::events() const noexcept {
  return events_;
}

const HappensBeforeEdges& HappensBeforeGraph::edges() const noexcept {
  return edges_;
}

std::shared_ptr<const EventRelations> HappensBeforeGraph::relationStorage() const {
  std::lock_guard lock(*relation_mutex_);
  if(!relation_snapshot_) {
    if(base_relations_)relation_snapshot_=std::make_shared<const EventRelations>(base_relations_,eventCount(),edges_);
    else relation_snapshot_=std::make_shared<const EventRelations>(events_,edges_);
  }
  return relation_snapshot_;
}

EventNeighbors HappensBeforeGraph::predecessors(EventId id) const {
  return predecessors_.at(indexOf(id));
}

EventNeighbors HappensBeforeGraph::successors(EventId id) const {
  return successors_.at(indexOf(id));
}

std::optional<EventId> HappensBeforeGraph::phaseCompletion(
    PhaseKey phase) const {
  const auto found = phase_completions_.find(phase);
  if (found == phase_completions_.end()) {
    return std::nullopt;
  }
  return found->second;
}

const DerivedEdgeJustifications& HappensBeforeGraph::justifications()
    const noexcept {
  return justifications_;
}

std::optional<EventOrder> HappensBeforeGraph::topologicalOrder()
    const {
  if(forward_ordered_)return EventOrder::identity(eventCount());
  if (events_.storedBindingCount() * 2 < eventCount())
    return TemplateTopologicalOrder(events_, edges_,relationStorage());
  std::vector<std::size_t> indegree(events_.size());
  for (std::size_t index = 0; index < events_.size(); ++index) {
    indegree[index] = predecessors_[index].size();
  }

  auto greaterId = [](EventId left, EventId right) {
    return left.value() > right.value();
  };
  std::priority_queue<EventId, std::vector<EventId>, decltype(greaterId)> ready(
      greaterId);
  for (std::size_t index = 0; index < indegree.size(); ++index) {
    if (indegree[index] == 0) {
      ready.push(EventId(static_cast<std::uint32_t>(index)));
    }
  }

  std::vector<EventId> order;
  order.reserve(events_.size());
  while (!ready.empty()) {
    const EventId current = ready.top();
    ready.pop();
    order.push_back(current);
    for (EventId successor : successors_[indexOf(current)]) {
      std::size_t& remaining = indegree[indexOf(successor)];
      --remaining;
      if (remaining == 0) {
        ready.push(successor);
      }
    }
  }

  if (order.size() != events_.size()) {
    return std::nullopt;
  }
  return order;
}

bool HappensBeforeGraph::isAcyclic() const {
  if (acyclic_cache_ && acyclic_cache_->first == version_)
    return acyclic_cache_->second;
  if (events_.storedBindingCount() * 2 < eventCount()) {
    const bool acyclic = TemplateTopologicalOrder(events_, edges_,relationStorage()).has_value();
    acyclic_cache_ = std::pair{version_, acyclic};
    return acyclic;
  }
  std::vector<std::size_t> pending(events_.size());
  std::vector<EventId> ready;
  ready.reserve(events_.size());
  for (std::size_t index = 0; index < events_.size(); ++index) {
    pending[index] = predecessors_[index].size();
    if (pending[index] == 0)
      ready.push_back(EventId(static_cast<std::uint32_t>(index)));
  }
  for (std::size_t cursor = 0; cursor < ready.size(); ++cursor) {
    for (EventId successor : successors_[indexOf(ready[cursor])]) {
      if (--pending[indexOf(successor)] == 0) ready.push_back(successor);
    }
  }
  const bool acyclic = ready.size() == events_.size();
  acyclic_cache_ = std::pair{version_, acyclic};
  return acyclic;
}

const EventDag& HappensBeforeGraph::graph() const {
  if (!topology_snapshot_version_.has_value() ||
      *topology_snapshot_version_ != version_) {
    rebuildSnapshot();
  }
  return topology_snapshot_;
}

void HappensBeforeGraph::invalidateSnapshot() noexcept {
  std::lock_guard lock(*relation_mutex_);
  relation_snapshot_.reset();
  topology_snapshot_version_.reset();
  acyclic_cache_.reset();
}

void HappensBeforeGraph::rebuildSnapshot() const {
  EventDagBuilder builder(events_);

  std::set<std::pair<EventId, EventId>> adjacency;
  for (const HappensBeforeEdge& edge : edges_) {
    if (!adjacency.emplace(edge.from, edge.to).second) {
      continue;
    }
    if (!builder.addEdge(edge.from, edge.to, toTopologyReason(edge.reason),
                         edge.source_edge)) {
      throw std::logic_error(
          "happens-before graph edge references a missing event");
    }
  }
  topology_snapshot_ = std::move(builder).freeze();
  topology_snapshot_version_ = version_;
}

HappensBeforeGraphBuilder::HappensBeforeGraphBuilder(
    const LoweredProgram& program, const ReferenceTrace* trace) {
  const EventDag& base = program.graph();
  happensBeforeGraph_.actors_ = program.actors();
  if (!base.isAcyclic()) {
    throw std::invalid_argument("lowered program graph is cyclic");
  }

  if (trace) happensBeforeGraph_.trace_order_.emplace(
      ProfileCall("sync.validate_trace_order", [&] { return TraceOrder(base, *trace); }));
  const auto completions = trace ? happensBeforeGraph_.trace_order_->completionCount() : 0;
  happensBeforeGraph_.events_=base.events();
  happensBeforeGraph_.base_relations_=base.adjacencyStorage()?base.adjacencyStorage()->relationStorage():nullptr;
  happensBeforeGraph_.events_.reserve(base.events().storedBindingCount() + completions);
  happensBeforeGraph_.predecessors_ = AdjacencyOverlay(
      base.adjacencyStorage(), AdjacencyOverlay::Direction::kPredecessors,
      base.eventCount() + completions);
  happensBeforeGraph_.successors_ = AdjacencyOverlay(
      base.adjacencyStorage(), AdjacencyOverlay::Direction::kSuccessors,
      base.eventCount() + completions);
  base.events().forEachAffineColumn([&](const Event& event, const auto& delta,
      std::size_t begin, std::size_t stride, std::size_t count) {
    if (indexOf(event.id) != begin || (count > 1 && delta[0] != std::int64_t(stride))) {
      throw std::invalid_argument("lowered program event IDs are not dense");
    }
    if (std::holds_alternative<MBarrierPhaseCompleteEvent>(event.payload)) {
      throw std::invalid_argument(
          "lowered program contains a derived phase-completion event");
    }
  });

  std::size_t derived_capacity = 0;
  const auto maximum = static_cast<std::size_t>(PTRDIFF_MAX);
  if (base.edgeCount() > maximum) throw std::length_error("HB edge count exceeds iterator limit");
  if (trace) {
    const auto effects = trace->effectPhases().size();
    const auto waits = trace->waitTargets().size();
    const auto available = maximum - base.edgeCount();
    if (effects > available / 2 || waits > available - 2 * effects)
      throw std::length_error("HB edge count exceeds iterator limit");
    derived_capacity = 2 * effects + waits;
  }
  happensBeforeGraph_.edges_.reserve(std::min(derived_capacity, base.edges().storedBindingCount()));
  base.edges().forEachAffineColumn([&](const EventEdge& edge, const auto& delta, auto, auto, std::size_t count) {
    happensBeforeGraph_.forward_ordered_ &= edge.from<edge.to;
    if (!base.contains(edge.from) || !base.contains(edge.to))
      throw std::invalid_argument("lowered program edge has an invalid ID");
    const auto last_from = std::int64_t(edge.from.value()) + delta[0] * std::int64_t(count - 1);
    const auto last_to = std::int64_t(edge.to.value()) + delta[1] * std::int64_t(count - 1);
    if (last_from < 0 || last_to < 0 || std::uint64_t(last_from) >= base.eventCount() ||
        std::uint64_t(last_to) >= base.eventCount()) throw std::invalid_argument("lowered program edge has an invalid ID");
    happensBeforeGraph_.forward_ordered_ &= last_from < last_to;
    happensBeforeGraph_.edges_.appendAffine(
        HappensBeforeEdge{edge.from, edge.to, toHappensBeforeReason(edge.reason), edge.source_edge, std::nullopt}, delta, count);
  });
  if (const auto* relations = base.adjacencyStorage() ? base.adjacencyStorage()->relations() : nullptr) {
    const auto [sources, targets] = relations->potentialParallelFamilies();
    if (!sources.empty() || !targets.empty()) {
      DynamicBitset seen(base.eventCount());
      const auto deduplicate = [&](const auto& families, auto& adjacency) {
        for (auto family : families) {
          const auto& members = relations->layout()->families()[family];
          for (std::size_t i = 0; i < members.count; ++i)
            DeduplicateAdjacency(adjacency, members.first.value() + members.stride * i, seen);
        }
      };
      deduplicate(sources, happensBeforeGraph_.successors_);
      deduplicate(targets, happensBeforeGraph_.predecessors_);
    }
  } else {
    DynamicBitset seen(base.eventCount());
    for (const Event& event : base.events()) {
      DeduplicateAdjacency(happensBeforeGraph_.predecessors_, indexOf(event.id), seen);
      DeduplicateAdjacency(happensBeforeGraph_.successors_, indexOf(event.id), seen);
    }
  }
  happensBeforeGraph_.invalidateSnapshot();
  happensBeforeGraph_.acyclic_cache_ =
      std::pair{happensBeforeGraph_.version_, true};
}

const HappensBeforeGraph& HappensBeforeGraphBuilder::happensBeforeGraph()
    const noexcept {
  return happensBeforeGraph_;
}

bool HappensBeforeGraphBuilder::Reaches(EventId from, EventId to) const {
  ProfileScope profile("builder.cycle_search");
  if (!happensBeforeGraph_.contains(from) ||
      !happensBeforeGraph_.contains(to)) {
    return false;
  }
  if (from == to) {
    return true;
  }

  std::vector<bool> visited(happensBeforeGraph_.eventCount(), false);
  std::vector<EventId> worklist{from};
  visited[indexOf(from)] = true;
  for (std::size_t cursor = 0; cursor < worklist.size(); ++cursor) {
    ++profile.nodes;
    for (EventId successor : happensBeforeGraph_.successors(worklist[cursor])) {
      ++profile.edges;
      if (successor == to) {
        return true;
      }
      const std::size_t index = indexOf(successor);
      if (!visited[index]) {
        visited[index] = true;
        worklist.push_back(successor);
      }
    }
  }
  return false;
}

Result<void, HappensBeforeGraphMutationError>
HappensBeforeGraphBuilder::AddCertifiedEdge(EventId from, EventId to,
                                            HappensBeforeEdgeReason reason,
                                            CertificateRef certificate) {
  ProfileScope profile("builder.add_edge");
  if (frozen_) {
    return Result<void, HappensBeforeGraphMutationError>::Failure(
        mutationError(HappensBeforeGraphMutationErrorCode::kFrozen, from, to,
                      certificate.phase));
  }
  if (!happensBeforeGraph_.contains(from) ||
      !happensBeforeGraph_.contains(to)) {
    return Result<void, HappensBeforeGraphMutationError>::Failure(
        mutationError(HappensBeforeGraphMutationErrorCode::kMissingEvent, from,
                      to, certificate.phase));
  }
  if (!isDerivedReason(reason)) {
    return Result<void, HappensBeforeGraphMutationError>::Failure(
        mutationError(HappensBeforeGraphMutationErrorCode::kInvalidReason, from,
                      to, certificate.phase));
  }
  const auto* order = happensBeforeGraph_.traceOrder();
  if (order && order->rank(from) >= order->rank(to)) {
    return Result<void, HappensBeforeGraphMutationError>::Failure(
        mutationError(HappensBeforeGraphMutationErrorCode::kTraceOrderViolation,
                      from, to, certificate.phase));
  }
  if (!order && (from == to || (!hasAdjacency(from, to) && Reaches(to, from)))) {
    return Result<void, HappensBeforeGraphMutationError>::Failure(
        mutationError(HappensBeforeGraphMutationErrorCode::kCycle, from, to,
                      certificate.phase));
  }

  const HappensBeforeEdge edge{from, to, reason, std::nullopt, certificate};
  const auto key = std::tuple{from, to, reason, certificate.phase};
  if (certified_edges_.contains(key)) {
    return Result<void, HappensBeforeGraphMutationError>::Success();
  }

  if (!hasAdjacency(from, to)) {
    appendAdjacency(from, to);
  }
  happensBeforeGraph_.edges_.pushBackAffine(edge);
  happensBeforeGraph_.forward_ordered_ &= from<to;
  happensBeforeGraph_.justifications_.push_back(
      DerivedEdgeJustification{from, to, reason, certificate});
  certified_edges_.insert(key);
  ++happensBeforeGraph_.version_;
  happensBeforeGraph_.invalidateSnapshot();
  happensBeforeGraph_.acyclic_cache_ =
      std::pair{happensBeforeGraph_.version_, true};
  return Result<void, HappensBeforeGraphMutationError>::Success();
}

Result<void, HappensBeforeGraphMutationError>
HappensBeforeGraphBuilder::AddCertifiedEdges(
    std::span<const std::pair<EventId, EventId>> edges,
    HappensBeforeEdgeReason reason, CertificateRef certificate) {
  ProfileScope profile("builder.add_edge_batch");
  const auto* order = happensBeforeGraph_.traceOrder();
  if (!order) {
    for (const auto& [from, to] : edges) {
      auto added = AddCertifiedEdge(from, to, reason, certificate);
      if (!added) return added;
    }
    return Result<void, HappensBeforeGraphMutationError>::Success();
  }
  for (const auto& [from, to] : edges) {
    const auto fail = [&](HappensBeforeGraphMutationErrorCode code) {
      return Result<void, HappensBeforeGraphMutationError>::Failure(
          mutationError(code, from, to, certificate.phase));
    };
    if (frozen_) return fail(HappensBeforeGraphMutationErrorCode::kFrozen);
    if (!happensBeforeGraph_.contains(from) || !happensBeforeGraph_.contains(to))
      return fail(HappensBeforeGraphMutationErrorCode::kMissingEvent);
    if (!isDerivedReason(reason)) return fail(HappensBeforeGraphMutationErrorCode::kInvalidReason);
    if (order->rank(from) >= order->rank(to))
      return fail(HappensBeforeGraphMutationErrorCode::kTraceOrderViolation);
  }
  DerivedEdgeJustifications additions;
  std::vector<std::pair<EventId, EventId>> fresh;
  fresh.reserve(edges.size());
  for (const auto& [from, to] : edges) {
    if (!certified_edges_.emplace(from, to, reason, certificate.phase).second) continue;
    if (!hasAdjacency(from, to)) appendAdjacency(from, to);
    happensBeforeGraph_.forward_ordered_ &= from < to;
    fresh.emplace_back(from, to);
  }
  if (fresh.empty()) return Result<void, HappensBeforeGraphMutationError>::Success();
  for (std::size_t begin = 0; begin < fresh.size();) {
    std::size_t end = begin + 1;
    DerivedEdgeCodec::Delta delta{};
    if (end < fresh.size()) {
      delta = {std::int64_t(fresh[end].first.value()) - fresh[begin].first.value(),
               std::int64_t(fresh[end].second.value()) - fresh[begin].second.value()};
      ++end;
      while (end < fresh.size() &&
          std::int64_t(fresh[end].first.value()) - fresh[end - 1].first.value() == delta[0] &&
          std::int64_t(fresh[end].second.value()) - fresh[end - 1].second.value() == delta[1]) ++end;
    }
    const DerivedEdgeJustification value{fresh[begin].first, fresh[begin].second, reason, certificate};
    additions.appendAffine(value, delta, end - begin);
    happensBeforeGraph_.justifications_.appendAffine(value, delta, end - begin);
    begin = end;
  }
  additions.forEachAffineColumn([&](const auto& edge,const auto& delta,auto,auto,std::size_t count) {
    happensBeforeGraph_.edges_.appendAffine(
        HappensBeforeEdge{edge.from,edge.to,edge.reason,std::nullopt,edge.certificate},
        HappensBeforeEdgeCodec::Delta{delta[0],delta[1],0,0},count);
  });
  profile.edges = fresh.size();
  ++happensBeforeGraph_.version_;
  happensBeforeGraph_.invalidateSnapshot();
  happensBeforeGraph_.acyclic_cache_ = std::pair{happensBeforeGraph_.version_, true};
  return Result<void, HappensBeforeGraphMutationError>::Success();
}

Result<EventId, HappensBeforeGraphMutationError>
HappensBeforeGraphBuilder::AddPhaseCompletion(PhaseKey phase) {
  if (frozen_) {
    return Result<EventId, HappensBeforeGraphMutationError>::Failure(
        mutationError(HappensBeforeGraphMutationErrorCode::kFrozen,
                      std::nullopt, std::nullopt, phase));
  }
  if (happensBeforeGraph_.phase_completions_.contains(phase)) {
    return Result<EventId, HappensBeforeGraphMutationError>::Failure(
        mutationError(
            HappensBeforeGraphMutationErrorCode::kDuplicatePhaseCompletion,
            std::nullopt, std::nullopt, phase));
  }
  if (happensBeforeGraph_.events_.size() >
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    return Result<EventId, HappensBeforeGraphMutationError>::Failure(
        mutationError(HappensBeforeGraphMutationErrorCode::kEventIdOverflow,
                      std::nullopt, std::nullopt, phase));
  }

  const EventId id(
      static_cast<std::uint32_t>(happensBeforeGraph_.events_.size()));
  if (happensBeforeGraph_.trace_order_ &&
      !happensBeforeGraph_.trace_order_->appendCompletion(id, phase)) {
    return Result<EventId, HappensBeforeGraphMutationError>::Failure(
        mutationError(HappensBeforeGraphMutationErrorCode::kTraceOrderViolation,
                      std::nullopt, std::nullopt, phase));
  }
  happensBeforeGraph_.events_.push_back(Event{
      id, EventSchedulingClass::kUrgentDerived, std::nullopt, std::nullopt,
      MBarrierPhaseCompleteEvent{phase.barrier, phase.phase}});
  happensBeforeGraph_.predecessors_.emplace_back();
  happensBeforeGraph_.successors_.emplace_back();
  happensBeforeGraph_.phase_completions_.emplace(phase, id);
  ++happensBeforeGraph_.version_;
  happensBeforeGraph_.invalidateSnapshot();
  happensBeforeGraph_.acyclic_cache_ =
      std::pair{happensBeforeGraph_.version_, true};
  return Result<EventId, HappensBeforeGraphMutationError>::Success(id);
}

const HappensBeforeGraph& HappensBeforeGraphBuilder::Freeze() noexcept {
  frozen_ = true;
  return happensBeforeGraph_;
}

bool HappensBeforeGraphBuilder::frozen() const noexcept { return frozen_; }

HappensBeforeGraph HappensBeforeGraphBuilder::TakeFrozen() && {
  if(!frozen_) throw std::logic_error("proof graph must be frozen before transfer");
  return std::move(happensBeforeGraph_);
}

bool HappensBeforeGraphBuilder::hasAdjacency(EventId from, EventId to) const {
  const EventNeighbors successors =
      happensBeforeGraph_.successors(from);
  return std::find(successors.begin(), successors.end(), to) !=
         successors.end();
}

void HappensBeforeGraphBuilder::appendAdjacency(EventId from, EventId to) {
  happensBeforeGraph_.successors_.append(indexOf(from), to);
  happensBeforeGraph_.predecessors_.append(indexOf(to), from);
}

}  // namespace sentryward::algorithm
