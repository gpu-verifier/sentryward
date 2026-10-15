#pragma once

#include <map>
#include <optional>
#include <vector>

#include "sentryward/algorithm/base/compact_id_map.h"
#include "sentryward/algorithm/base/strong_id.h"
#include "sentryward/algorithm/source_ir/async.h"

namespace sentryward::algorithm {

struct Milestones final {
  EventId issue;
  EventId done;
  std::optional<EventId> reads_complete = std::nullopt;
  std::vector<EventId> effects = {};

  friend bool operator==(const Milestones&, const Milestones&) = default;
};

class OriginMap final {
 public:
  void reserveEvents(std::size_t count);
  [[nodiscard]] bool record(InstId instruction, Milestones milestones);
  [[nodiscard]] std::optional<InstId> origin(EventId event) const;
  [[nodiscard]] std::optional<Milestones> milestones(InstId instruction) const;
  [[nodiscard]] std::optional<Milestones> findMilestones(InstId instruction) const;
  // Scalar lookup must not copy an instruction's potentially large effects.
  [[nodiscard]] std::optional<EventId> resolve(MilestoneRef ref) const;

  [[nodiscard]] std::map<InstId, Milestones> byInstruction() const;
  [[nodiscard]] CompactIdMap<EventId, InstId> byEvent() const;
  [[nodiscard]] std::size_t runCount() const { return runs_.size(); }
  [[nodiscard]] std::size_t individualCount() const { return by_instruction_.size(); }

 private:
  friend class MemoryCensus;
  struct Run { EventId first; std::uint64_t count; std::uint32_t width; };
  std::map<InstId, Run> runs_;
  std::map<EventId, InstId> event_runs_;
  std::size_t reserved_events_ = 0;
  std::map<InstId, Milestones> by_instruction_;
  CompactIdMap<EventId, InstId> by_event_;
};

}  // namespace sentryward::algorithm
