#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "sentryward/algorithm/event_ir/dag.h"
#include "sentryward/algorithm/event_ir/event_families.h"
#include "sentryward/algorithm/event_ir/event_relations.h"

namespace sentryward::algorithm {

// A member is an independent event; ranges share readiness bookkeeping only.
class TemplateSchedule final {
 public:
  struct Batch {
    std::size_t family, begin, count;
    EventId first;
    std::uint32_t stride;
  };
  explicit TemplateSchedule(const EventDag& graph) : TemplateSchedule(graph.events(), graph.edges(),graph.adjacencyStorage()?graph.adjacencyStorage()->relationStorage():nullptr) {}
  template<class Edges>
  TemplateSchedule(const EventRecords& events, const Edges& edges,std::shared_ptr<const EventRelations> topology={})
      :TemplateSchedule(events,topology?std::move(topology):std::make_shared<const EventRelations>(events,edges)) {}
  TemplateSchedule(const EventRecords& events,std::shared_ptr<const EventRelations> topology)
      :events_(events),topology_owner_(std::move(topology)),topology_(*topology_owner_),layout_(topology_.layout()) {
    for (const auto& member : layout_->families()) {
      const auto event = events_.at(member.first.value());
      Family family{event.id, member.stride, member.count,
                    event.scheduling_class == EventSchedulingClass::kUrgentDerived, {}, {}};
      family.runs.emplace(0, Run{member.count, 0, State::kPending});
      families_.push_back(std::move(family));
    }
    for (std::size_t i = 0; i < topology_.relations().size(); ++i) {
      const auto& r = topology_.relations()[i];
      change(r.target, r.first_target, r.first_target + (r.target_step ? r.count : 1),
             [&](Run& run) { run.pending += r.target_step ? 1 : r.count; });
    }
    for (std::size_t i = 0; i < families_.size(); ++i) refresh(i);
  }

  [[nodiscard]] std::size_t familyCount() const { return families_.size(); }
  [[nodiscard]] const std::shared_ptr<const EventFamilies>& layout() const { return layout_; }
  [[nodiscard]] std::size_t relationCount() const { return topology_.relations().size(); }
  [[nodiscard]] std::size_t executedCount() const { return executed_; }
  [[nodiscard]] bool executed(EventId id) const {
    const auto [family, member] = locate(id);
    return std::prev(families_[family].runs.upper_bound(member))->second.state == State::kDone;
  }
  [[nodiscard]] std::optional<Batch> next(bool urgent, bool prefer_high = false) const {
    const auto& queue = urgent ? urgent_ : normal_;
    if (queue.empty()) return {};
    const auto [id, family] = prefer_high ? *queue.rbegin() : *queue.begin();
    const auto& f = families_[family];
    const auto member = (id - f.first.value()) / f.stride;
    const auto& run = std::prev(f.runs.upper_bound(member))->second;
    return Batch{family, member, run.end - member, EventId(id), f.stride};
  }
  [[nodiscard]] Batch one(EventId id) const {
    auto [family, member] = locate(id);
    return {family, member, 1, id, families_[family].stride};
  }
  [[nodiscard]] bool hasUrgentSuccessor(const Batch& batch) const {
    for (auto id : topology_.outgoing(batch.family))
      if (families_[topology_.relations()[id].target].urgent) return true;
    return false;
  }
  [[nodiscard]] bool collectiveBatchSafe(const Batch& batch, SyncId sync) const {
    for (auto index : topology_.outgoing(batch.family)) {
      const auto& r = topology_.relations()[index];
      const auto& target = families_[r.target];
      if (!target.urgent) continue;
      const auto event = events_.at(target.first.value());
      const auto* complete = std::get_if<CollectiveCompleteEvent>(&event.payload);
      if (!complete || complete->sync_id != sync || r.target_step || !r.source_step ||
          r.first_source > batch.begin || r.first_source + r.count < batch.begin + batch.count) return false;
    }
    return true;
  }
  void park(const Batch& batch) {
    change(batch.family, batch.begin, batch.begin + batch.count,
           [](Run& run) { run.state = State::kParked; });
    refresh(batch.family);
  }
  void wake(EventId id) {
    wake(one(id));
  }
  void wake(const Batch& batch) {
    change(batch.family, batch.begin, batch.begin + batch.count,
           [](Run& run) { if (run.state == State::kParked) run.state = State::kPending; });
    refresh(batch.family);
  }
  [[nodiscard]] std::optional<Batch> correspondingIssue(const Batch& batch) const {
    for (auto index : topology_.incoming(batch.family)) {
      const auto& r = topology_.relations()[index];
      if (!r.source_step || !r.target_step || batch.begin < r.first_target ||
          batch.begin >= r.first_target + r.count) continue;
      const auto& source = families_[r.source];
      const auto start = r.first_source + batch.begin - r.first_target;
      const auto count = std::min(batch.count, r.first_target + r.count - batch.begin);
      const auto first = EventId(source.first.value() + source.stride * start);
      const auto issue = events_.at(first.value());
      if (!std::holds_alternative<WaitStateIssueEvent>(issue.payload) &&
          !std::holds_alternative<WaitParityIssueEvent>(issue.payload) &&
          !std::holds_alternative<NamedBarrierArriveEvent>(issue.payload)) continue;
      const auto done = events_.at(batch.first.value());
      if (!issue.origin || issue.origin != done.origin) continue;
      const auto last_issue = events_.at(first.value() + source.stride * (count - 1));
      const auto last_done = events_.at(batch.first.value() + batch.stride * (count - 1));
      if (last_issue.origin != last_done.origin) continue;
      return Batch{r.source, start, count, first, source.stride};
    }
    return {};
  }
  void finish(const Batch& batch) {
    change(batch.family, batch.begin, batch.begin + batch.count, [](Run& run) {
      if (run.pending || run.state == State::kDone) throw std::logic_error("event family not ready");
      run.state = State::kDone;
    });
    executed_ += batch.count;
    refresh(batch.family);
    for (auto id : topology_.outgoing(batch.family)) {
      const auto& r = topology_.relations()[id];
      std::size_t start = 0, count = r.count;
      if (r.source_step) {
        const auto lo = std::max(batch.begin, r.first_source);
        const auto hi = std::min(batch.begin + batch.count, r.first_source + r.count);
        if (lo >= hi) continue;
        start = lo - r.first_source; count = hi - lo;
      } else if (r.first_source < batch.begin || r.first_source >= batch.begin + batch.count) continue;
      const auto begin = r.first_target + (r.target_step ? start : 0);
      change(r.target, begin, begin + (r.target_step ? count : 1), [&](Run& run) {
        const auto decrement = r.target_step ? 1 : count;
        if (run.pending < decrement) throw std::logic_error("event dependency underflow");
        run.pending -= decrement;
      });
      refresh(r.target);
    }
  }

 private:
  enum class State { kPending, kParked, kDone };
  struct Run {
    std::size_t end, pending;
    State state;
    bool same(const Run& other) const { return pending == other.pending && state == other.state; }
  };
  struct Family {
    EventId first;
    std::uint32_t stride;
    std::size_t count;
    bool urgent;
    std::map<std::size_t, Run> runs;
    std::optional<std::uint32_t> ready;
  };
  std::pair<std::size_t, std::size_t> locate(EventId id) const {
    return layout_->locate(id);
  }
  template<class Update>
  void change(std::size_t family, std::size_t begin, std::size_t end, Update update) {
    auto& f = families_[family]; auto& runs = f.runs;
    const auto split = [&](std::size_t position) {
      if (position == f.count) return runs.end();
      auto it = std::prev(runs.upper_bound(position));
      if (it->first == position) return it;
      auto run = it->second; it->second.end = position;
      return runs.emplace(position, run).first;
    };
    auto last = split(end); auto first = split(begin);
    for (auto it = first; it != last; ++it) update(it->second);
    if (first != runs.begin()) --first;
    while (first != runs.end()) {
      auto next = std::next(first);
      if (next == runs.end() || next->first > end) break;
      if (first->second.same(next->second)) {
        first->second.end = next->second.end; runs.erase(next);
      } else first = next;
    }
  }
  void refresh(std::size_t family) {
    auto& f = families_[family]; auto& queue = f.urgent ? urgent_ : normal_;
    if (f.ready) queue.erase({*f.ready, family});
    f.ready.reset();
    for (const auto& [begin, run] : f.runs) {
      if (run.pending || run.state != State::kPending) continue;
      f.ready = f.first.value() + f.stride * begin;
      queue.emplace(*f.ready, family); break;
    }
  }
  const EventRecords& events_;
  std::shared_ptr<const EventRelations> topology_owner_;
  const EventRelations& topology_;
  std::shared_ptr<const EventFamilies> layout_;
  std::vector<Family> families_;
  std::set<std::pair<std::uint32_t, std::size_t>> normal_, urgent_;
  std::size_t executed_ = 0;
};

template<class Edges>
std::optional<EventOrder> TemplateTopologicalOrder(const EventRecords& events, const Edges& edges,std::shared_ptr<const EventRelations> topology={}) {
  TemplateSchedule ready(events, edges,std::move(topology));
  auto order = EventOrder::affine();
  while (true) {
    auto batch = ready.next(true);
    if (!batch) batch = ready.next(false);
    if (!batch) break;
    order.appendAffine(batch->first, batch->stride, batch->count);
    ready.finish(*batch);
  }
  if (ready.executedCount() != events.size()) return {};
  return order;
}

}  // namespace sentryward::algorithm
