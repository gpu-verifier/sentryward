#include "sentryward/algorithm/event_ir/program.h"

#include <utility>

namespace sentryward::algorithm {

SynchronizationEventIndex::SynchronizationEventIndex(const EventRecords& events) {
  events.forEachAffineColumn([&](const Event& event, const auto& delta,
                                 std::size_t, std::size_t, std::size_t count) {
    ++classified_templates;
    const auto add = [&](EventIdFamilies& family) {
      family.appendAffine(event.id, delta[0], count);
    };
    const auto& payload = event.payload;
    if (std::holds_alternative<MBarrierUpdateEvent>(payload) ||
        std::holds_alternative<MBarrierCompleteTxEvent>(payload) ||
        std::holds_alternative<MBarrierAddArrivalsEvent>(payload)) add(effects);
    if (const auto* op = std::get_if<AsyncMemoryExecEvent>(&payload); op && op->completion)
      add(effects);
    if (const auto* op = std::get_if<GenericAsyncEffectEvent>(&payload); op && op->completion)
      add(effects);
    if (const auto* op = std::get_if<NamedBarrierArriveEvent>(&payload)) {
      add(effects); add(named_arrivals);
      if (op->blocking) add(wait_issues);
    }
    if (std::holds_alternative<WaitStateIssueEvent>(payload) ||
        std::holds_alternative<WaitParityIssueEvent>(payload)) add(wait_issues);
    if (std::holds_alternative<WaitStateDoneEvent>(payload) ||
        std::holds_alternative<WaitParityDoneEvent>(payload) ||
        std::holds_alternative<NamedBarrierReturnEvent>(payload)) add(wait_done);
  });
}

bool StateDefMap::define(StateId state, EventId event) {
  return definitions_.emplace(state, event).second;
}

std::optional<EventId> StateDefMap::definition(StateId state) const {
  const auto found = definitions_.find(state);
  if (found == definitions_.end()) {
    return std::nullopt;
  }
  return found->second;
}

const std::map<StateId, EventId>& StateDefMap::entries() const noexcept {
  return definitions_;
}

bool BarrierInitializationTable::record(
    BarrierLifetimeId barrier, BarrierInitialization initialization) {
  return entries_.emplace(barrier, initialization).second;
}

std::optional<BarrierInitialization> BarrierInitializationTable::find(
    BarrierLifetimeId barrier) const {
  const auto found = entries_.find(barrier);
  if (found == entries_.end()) {
    return std::nullopt;
  }
  return found->second;
}

const std::map<BarrierLifetimeId, BarrierInitialization>&
BarrierInitializationTable::entries() const noexcept {
  return entries_;
}

bool CollectiveTable::record(LoweredCollective collective) {
  return entries_.emplace(collective.sync_id, std::move(collective)).second;
}

const LoweredCollective* CollectiveTable::find(SyncId sync_id) const {
  const auto found = entries_.find(sync_id);
  if (found == entries_.end()) {
    return nullptr;
  }
  return &found->second;
}

const std::map<SyncId, LoweredCollective>& CollectiveTable::entries() const
    noexcept {
  return entries_;
}

LoweredProgram::LoweredProgram(
    EventDag graph, OriginMap origins, StateDefMap state_defs,
    BarrierInitializationTable barrier_initializations,
    CollectiveTable collectives, std::vector<ActorInfo> actors)
    : graph_(std::move(graph)),
      actors_(std::move(actors)),
      origins_(std::move(origins)),
      state_defs_(std::move(state_defs)),
      barrier_initializations_(std::move(barrier_initializations)),
      collectives_(std::move(collectives)), synchronization_events_(graph_.events()) {}

const EventDag& LoweredProgram::graph() const noexcept { return graph_; }

const OriginMap& LoweredProgram::origins() const noexcept { return origins_; }

const StateDefMap& LoweredProgram::stateDefinitions() const noexcept {
  return state_defs_;
}

const BarrierInitializationTable& LoweredProgram::barrierInitializations()
    const noexcept {
  return barrier_initializations_;
}

const CollectiveTable& LoweredProgram::collectives() const noexcept {
  return collectives_;
}

}  // namespace sentryward::algorithm
