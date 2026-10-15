#include "sentryward/algorithm/analysis/trace_order.h"

#include <stdexcept>

namespace sentryward::algorithm {
TraceOrder::TraceOrder(const EventDag& graph, const ReferenceTrace& trace)
    : base_count_(graph.eventCount()) {
  auto layout = std::make_shared<EventFamilies>(graph.events());
  ranks_.reset(layout);
  const auto assign = [&](EventId first, std::int64_t stride, std::size_t rank, std::size_t count) {
    std::size_t i = 0;
    while (i < count) {
      const auto value = std::int64_t(first.value()) + stride * std::int64_t(i);
      if (value < 0 || std::uint64_t(value) >= graph.eventCount())
        throw std::invalid_argument("trace order has an unknown event");
      const auto id = EventId(value);
      const auto [family, member] = layout->locate(id);
      const auto& group = layout->families()[family];
      const auto n = stride == group.stride ? std::min(count - i, group.count - member) : 1;
      try { ranks_.insert(id, rank + i, n); }
      catch (const std::logic_error&) { throw std::invalid_argument("trace order has a repeated event"); }
      i += n;
    }
  };
  for (const auto& run : trace.entries().runs()) {
    if (const auto* normal = std::get_if<NormalTraceEntry>(&run.first)) {
      assign(normal->event, run.delta, run.begin, run.count);
    } else if (const auto* collective = std::get_if<CollectiveCompletionTraceEntry>(&run.first)) {
      assign(collective->event, 0, run.begin, run.count);
    } else {
      const auto phase = std::get<BarrierPhaseCompletionTraceEntry>(run.first).phase;
      if (run.count != 1 || !completion_ranks_.emplace(phase, run.begin).second)
        throw std::invalid_argument("trace order has a repeated completion");
    }
  }
  if (ranks_.size() != graph.eventCount()) throw std::invalid_argument("trace order is incomplete");
  graph.edges().forEachAffineColumn([&](const EventEdge& edge, const auto& delta, auto, auto, std::size_t count) {
    std::size_t i = 0;
    while (i < count) {
      const auto from = EventId(std::int64_t(edge.from.value()) + delta[0] * std::int64_t(i));
      const auto to = EventId(std::int64_t(edge.to.value()) + delta[1] * std::int64_t(i));
      const auto a = ranks_.prefix(from, delta[0], count - i);
      const auto b = ranks_.prefix(to, delta[1], count - i);
      const auto n = std::min(a.count, b.count);
      if (a.rank >= b.rank || std::int64_t(a.rank) + a.step * std::int64_t(n - 1) >=
          std::int64_t(b.rank) + b.step * std::int64_t(n - 1))
        throw std::invalid_argument("original edge contradicts trace order");
      i += n;
    }
  });
}

bool TraceOrder::appendCompletion(EventId event, PhaseKey phase) {
  const auto found = completion_ranks_.find(phase);
  if (event.value() != base_count_ + derived_ranks_.size() || found == completion_ranks_.end())
    return false;
  derived_ranks_.push_back(found->second);
  return true;
}
}  // namespace sentryward::algorithm
