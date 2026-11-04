#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "sentryward/algorithm/event_ir/dag.h"
#include "sentryward/algorithm/event_ir/origin_map.h"
#include "sentryward/algorithm/source_ir/program.h"

namespace sentryward::algorithm {

class StateDefMap final {
 public:
  [[nodiscard]] bool define(StateId state, EventId event);
  [[nodiscard]] std::optional<EventId> definition(StateId state) const;
  [[nodiscard]] const std::map<StateId, EventId>& entries() const noexcept;

 private:
  std::map<StateId, EventId> definitions_;
};

struct BarrierInitialization final {
  EventId event;
  std::uint32_t expected_arrivals;

  friend bool operator==(const BarrierInitialization&,
                         const BarrierInitialization&) = default;
};

class BarrierInitializationTable final {
 public:
  [[nodiscard]] bool record(BarrierLifetimeId barrier,
                            BarrierInitialization initialization);
  [[nodiscard]] std::optional<BarrierInitialization> find(
      BarrierLifetimeId barrier) const;
  [[nodiscard]] const std::map<BarrierLifetimeId, BarrierInitialization>&
  entries() const noexcept;

 private:
  std::map<BarrierLifetimeId, BarrierInitialization> entries_;
};

struct CollectiveParticipant final {
  ActorId actor;
  InstId instruction;
  Milestones milestones;

  friend bool operator==(const CollectiveParticipant&,
                         const CollectiveParticipant&) = default;
};

struct LoweredCollective final {
  SyncId sync_id;
  CollectiveKind kind;
  EventId completion;
  std::vector<ActorId> expected_participants;
  std::optional<std::uint32_t> lane_mask;
  std::vector<CollectiveParticipant> participants;

  friend bool operator==(const LoweredCollective&,
                         const LoweredCollective&) = default;
};

class CollectiveTable final {
 public:
  [[nodiscard]] bool record(LoweredCollective collective);
  [[nodiscard]] const LoweredCollective* find(SyncId sync_id) const;
  [[nodiscard]] const std::map<SyncId, LoweredCollective>& entries() const
      noexcept;

 private:
  std::map<SyncId, LoweredCollective> entries_;
};

struct EventIdCodec final {
  using Binding = EventId;
  using Delta = std::int64_t;
  static bool split(EventId& value, Binding& binding) {
    binding = value; value = EventId{}; return true;
  }
  static void bind(EventId& value, Binding binding) { value = binding; }
  static std::size_t hash(EventId) { return 0; }
  static std::optional<Delta> difference(Binding a, Binding b) {
    return std::int64_t(b.value()) - a.value();
  }
  static std::optional<Binding> translate(Binding base, Delta delta, std::size_t i) {
    if (delta >= 0) {
      const auto step = static_cast<std::uint64_t>(delta);
      if (step && i > (UINT32_MAX - base.value()) / step) return {};
      return EventId(base.value() + static_cast<std::uint32_t>(i * step));
    }
    const auto step = static_cast<std::uint64_t>(-(delta + 1)) + 1;
    if (i > base.value() / step) return {};
    return EventId(base.value() - static_cast<std::uint32_t>(i * step));
  }
};
using EventIdFamilies = TemplateRecords<EventId, EventIdCodec>;

struct SynchronizationEventIndex final {
  EventIdFamilies effects, wait_issues, wait_done, named_arrivals;
  std::size_t classified_templates = 0;
  explicit SynchronizationEventIndex(const EventRecords& events);
};

class LoweredProgram final {
 public:
  LoweredProgram(EventDag graph, OriginMap origins, StateDefMap state_defs,
                 BarrierInitializationTable barrier_initializations,
                 CollectiveTable collectives, std::vector<ActorInfo> actors,
                 std::vector<std::size_t> execution_only_edges);
  [[nodiscard]] const std::vector<ActorInfo>& actors() const noexcept { return actors_; }

  // The execution-order relation: a participant of a collective resumes only after every
  // participant has issued it. The wait, phase and lifecycle checks reason about when
  // synchronization events can execute and need this relation.
  [[nodiscard]] const EventDag& graph() const noexcept;
  // The edges of graph() that only the execution order carries (ascending indices into
  // graph().edges()): the completion -> done edges of warp data collectives
  // (shfl/elect/vote/match/redux .sync). Their rendezvous publishes no memory effects, so a
  // participant's return is not ordered after the other lanes' issues in memory order (#251).
  [[nodiscard]] const std::vector<std::size_t>& executionOnlyEdges() const noexcept {
    return execution_only_edges_;
  }
  // The memory-order relation: graph() less executionOnlyEdges(), with the same events and IDs.
  [[nodiscard]] EventDag memoryOrderGraph() const { return graph_.without(execution_only_edges_); }
  [[nodiscard]] const OriginMap& origins() const noexcept;
  // After synchronization, retain the source/event mapping without retaining
  // or copying the whole lowered graph.
  [[nodiscard]] OriginMap takeOrigins() && { return std::move(origins_); }
  [[nodiscard]] const StateDefMap& stateDefinitions() const noexcept;
  [[nodiscard]] const BarrierInitializationTable& barrierInitializations()
      const noexcept;
  [[nodiscard]] const CollectiveTable& collectives() const noexcept;
  [[nodiscard]] const SynchronizationEventIndex& synchronizationEvents() const noexcept {
    return synchronization_events_;
  }

 private:
  EventDag graph_;
  std::vector<std::size_t> execution_only_edges_;
  std::vector<ActorInfo> actors_;
  OriginMap origins_;
  StateDefMap state_defs_;
  BarrierInitializationTable barrier_initializations_;
  CollectiveTable collectives_;
  SynchronizationEventIndex synchronization_events_;
};

}  // namespace sentryward::algorithm
