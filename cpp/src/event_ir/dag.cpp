#include "sentryward/algorithm/event_ir/dag.h"
#include "sentryward/algorithm/semantics/template_schedule.h"

#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>

namespace sentryward::algorithm {

namespace {

std::size_t indexOf(EventId id) {
  return static_cast<std::size_t>(id.value());
}

}  // namespace

CsrAdjacency::CsrAdjacency(const EventRecords& events, const EventEdges& edges) : count_(events.size()) {
  const auto count = count_;
  EventFamilies families(events);
  if (families.families().size() * 2 < count) {
    relations_=std::make_shared<const EventRelations>(events, edges);
    return;
  }
  predecessor_offsets_.resize(count + 1); successor_offsets_.resize(count + 1);
  predecessors_.resize(edges.size()); successors_.resize(edges.size());
  for(const auto& edge:edges) {
    ++predecessor_offsets_[indexOf(edge.to)+1];
    ++successor_offsets_[indexOf(edge.from)+1];
  }
  for(std::size_t i=1;i<predecessor_offsets_.size();++i) {
    predecessor_offsets_[i]+=predecessor_offsets_[i-1];
    successor_offsets_[i]+=successor_offsets_[i-1];
  }
  // Reuse the end offsets as cursors and fill backwards. This preserves edge
  // insertion order without allocating two additional O(V) cursor arrays.
  for (std::size_t i = edges.size(); i-- > 0;) {
    const auto edge = edges[i];
    predecessors_[--predecessor_offsets_[indexOf(edge.to)+1]]=edge.from;
    successors_[--successor_offsets_[indexOf(edge.from)+1]]=edge.to;
  }
  for(std::size_t i=0;i<count;++i) {
    predecessor_offsets_[i]=predecessor_offsets_[i+1];
    successor_offsets_[i]=successor_offsets_[i+1];
  }
  predecessor_offsets_.back()=successor_offsets_.back()=edges.size();
}

EventDag::EventDag(EventRecords events, EventEdges edges)
    : events_(std::move(events)), edges_(std::move(edges)),
      adjacency_(new CsrAdjacency(events_, edges_)) {
  forward_ordered_ = true;
  edges_.forEachAffineColumn([&](const EventEdge& edge, const auto& delta, auto, auto, std::size_t count) {
    forward_ordered_ &= edge.from < edge.to &&
        std::int64_t(edge.from.value()) + delta[0] * std::int64_t(count - 1) <
        std::int64_t(edge.to.value()) + delta[1] * std::int64_t(count - 1);
  });
  acyclic_ = forward_ordered_ || computeAcyclic();
}

std::size_t EventDag::eventCount() const noexcept { return events_.size(); }

std::size_t EventDag::edgeCount() const noexcept { return edges_.size(); }

bool EventDag::contains(EventId id) const noexcept {
  return indexOf(id) < events_.size();
}

Event EventDag::event(EventId id) const {
  return events_.at(indexOf(id));
}

const EventRecords& EventDag::events() const noexcept { return events_; }

const EventEdges& EventDag::edges() const noexcept { return edges_; }

EventNeighbors EventDag::predecessors(EventId id) const {
  if (!adjacency_) throw std::out_of_range("event ID is absent");
  return adjacency_->predecessors(id);
}

EventNeighbors EventDag::successors(EventId id) const {
  if (!adjacency_) throw std::out_of_range("event ID is absent");
  return adjacency_->successors(id);
}

std::optional<EventOrder> EventDag::topologicalOrder() const {
  if(forward_ordered_)return EventOrder::identity(eventCount());
  if (!acyclic_) {
    return std::nullopt;
  }
  if (events_.storedBindingCount() * 2 < eventCount())
    return TemplateTopologicalOrder(events_, edges_,adjacency_->relationStorage());
  std::vector<std::size_t> indegree(events_.size());
  for (std::size_t index = 0; index < events_.size(); ++index) {
    indegree[index] = predecessors(EventId(index)).size();
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

    for (EventId successor : successors(current)) {
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

bool EventDag::computeAcyclic() const {
  if (events_.storedBindingCount() * 2 < eventCount())
    return TemplateTopologicalOrder(events_, edges_,adjacency_->relationStorage()).has_value();
  std::vector<std::size_t> indegree(events_.size());
  std::vector<EventId> ready;
  for (std::size_t index = 0; index < events_.size(); ++index) {
    indegree[index] = predecessors(EventId(index)).size();
    if (indegree[index] == 0) {
      ready.emplace_back(static_cast<std::uint32_t>(index));
    }
  }
  // Only the active frontier is needed for the boolean result. The stable
  // topologicalOrder() API still uses its deterministic minimum-ID queue.
  std::size_t visited = 0;
  while (!ready.empty()) {
    const EventId current = ready.back();
    ready.pop_back();
    ++visited;
    for (EventId successor : successors(current)) {
      if (--indegree[indexOf(successor)] == 0) {
        ready.push_back(successor);
      }
    }
  }
  return visited == events_.size();
}

bool EventDag::isAcyclic() const { return acyclic_; }

EventDagBuilder::EventDagBuilder(EventRecords events):events_(std::move(events)) {
  for(std::size_t i=0;i<events_.size();++i)
    if(events_[i].id.value()!=i)throw std::invalid_argument("event IDs are not dense");
}

void EventDagBuilder::reserve(std::size_t events,std::size_t edges) {
  events_.reserve(events);edges_.reserve(std::min(edges, std::size_t{256}));
}

EventId EventDagBuilder::addEvent(EventSchedulingClass scheduling_class,
                                  std::optional<InstId> origin,
                                  std::optional<ActorId> actor,
                                  EventPayload payload) {
  if (events_.size() >
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    throw std::overflow_error("event ID space exhausted");
  }
  const EventId id(static_cast<std::uint32_t>(events_.size()));
  events_.push_back(Event{id, scheduling_class, origin, actor,
                          std::move(payload)});
  return id;
}

bool EventDagBuilder::addEdge(
    EventId from, EventId to, EventEdgeReason reason,
    std::optional<SourceEdgeProvenance> source_edge) {
  if (indexOf(from) >= events_.size() || indexOf(to) >= events_.size()) {
    return false;
  }
  EventEdge edge{from,to,reason,std::move(source_edge)};
  if(edge_group_)edges_.push_back(std::move(edge));else edges_.pushBackAffine(std::move(edge));
  return true;
}

EventDag EventDagBuilder::freeze() && {
  endEdgeGroup();
  events_.sealThreadFamilies();
  events_.shrink_to_fit();
  edges_.shrink_to_fit();
  return EventDag(std::move(events_), std::move(edges_));
}

bool EventDagBuilder::addEdgeColumns(const std::vector<EventEdge>& columns,
    const std::vector<EventEdgeCodec::Delta>& deltas,std::size_t count) {
  if(columns.empty() || columns.size()!=deltas.size())return false;
  for(std::size_t i=0;i<columns.size();++i) {
    auto edge=columns[i];EventEdgeCodec::Binding binding{};
    EventEdgeCodec::split(edge,binding);
    const auto last=EventEdgeCodec::translate(binding,deltas[i],count?count-1:0);
    if(!last || binding[0]>=events_.size() || binding[1]>=events_.size() ||
        (*last)[0]>=events_.size() || (*last)[1]>=events_.size())return false;
  }
  if(edge_group_) {
    for(std::size_t member=0;member<count;++member)for(std::size_t i=0;i<columns.size();++i) {
      auto edge=columns[i];EventEdgeCodec::Binding binding{};
      EventEdgeCodec::split(edge,binding);
      EventEdgeCodec::bind(edge,*EventEdgeCodec::translate(binding,deltas[i],member));
      edges_.push_back(std::move(edge));
    }
  } else {
    endEdgeGroup();edges_.appendAffineColumns(columns,deltas,count);
  }
  return true;
}

void EventDagBuilder::beginEdgeGroup(std::uint64_t scope,std::uintptr_t history) {
  if(edge_group_)throw std::logic_error("nested edge group");
  edge_group_=edges_.beginThread();
  edge_cohort_={scope,history};
}
void EventDagBuilder::endEdgeGroup() {
  if(!edge_group_)return;
  edges_.extendThreadFamily(*edge_group_,edge_cohort_.first,edge_cohort_.second);edge_group_.reset();
}

}  // namespace sentryward::algorithm
