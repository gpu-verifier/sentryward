#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include "sentryward/algorithm/base/strong_id.h"
#include "sentryward/algorithm/base/template_records.h"

namespace sentryward::algorithm {

enum class EventEdgeReason {
  kSourceProgramOrder,
  kSourceCompletionDependency,
  kInternalIssueToDone,
  kStateDefUse,
  kCollectiveArrival,
  kCollectiveRelease,
  kAsyncEffectDependency,
  kAsyncAggregate,
  kAwaitRequirement,
  kEndpointDependency,
};

struct SourceEdgeProvenance final {
  InstId from;
  InstId to;

  friend bool operator==(const SourceEdgeProvenance&,
                         const SourceEdgeProvenance&) = default;
};

struct EventEdge final {
  EventId from;
  EventId to;
  EventEdgeReason reason;
  std::optional<SourceEdgeProvenance> source_edge;

  friend bool operator==(const EventEdge&, const EventEdge&) = default;
};

struct EventEdgeCodec final {
  using Binding = std::array<std::uint32_t, 4>;
  using Delta = std::array<std::int64_t, 4>;
  static bool split(EventEdge& edge, Binding& binding) {
    binding = {edge.from.value(), edge.to.value(),
        edge.source_edge ? edge.source_edge->from.value() : 0,
        edge.source_edge ? edge.source_edge->to.value() : 0};
    edge.from = EventId{}; edge.to = EventId{};
    if (edge.source_edge) edge.source_edge = SourceEdgeProvenance{};
    return true;
  }
  static void bind(EventEdge& edge, const Binding& binding) {
    edge.from = EventId(binding[0]); edge.to = EventId(binding[1]);
    if (edge.source_edge) edge.source_edge = SourceEdgeProvenance{InstId(binding[2]), InstId(binding[3])};
  }
  static std::size_t hash(const EventEdge& edge) {
    return static_cast<std::size_t>(edge.reason) * 2 + edge.source_edge.has_value();
  }
  static std::optional<Delta> difference(const Binding& a, const Binding& b) {
    Delta delta;
    for (std::size_t i = 0; i < delta.size(); ++i) delta[i] = std::int64_t(b[i]) - a[i];
    return delta;
  }
  static std::optional<Binding> translate(Binding base, const Delta& delta, std::size_t index) {
    for (std::size_t i = 0; i < base.size(); ++i) {
      if (delta[i] >= 0) {
        const auto step = static_cast<std::uint64_t>(delta[i]);
        if (step && index > (UINT32_MAX - base[i]) / step) return {};
        base[i] += static_cast<std::uint32_t>(step * index);
      } else {
        const auto step = static_cast<std::uint64_t>(-(delta[i] + 1)) + 1;
        if (index > base[i] / step) return {};
        base[i] -= static_cast<std::uint32_t>(step * index);
      }
    }
    return base;
  }
};

using EventEdges = TemplateRecords<EventEdge, EventEdgeCodec>;

} // namespace sentryward::algorithm
