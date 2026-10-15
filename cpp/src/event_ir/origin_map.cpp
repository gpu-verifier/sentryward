#include "sentryward/algorithm/event_ir/origin_map.h"

#include <algorithm>
#include <utility>

namespace sentryward::algorithm {

void OriginMap::reserveEvents(std::size_t count) {
  reserved_events_ = std::max(reserved_events_, count);
}

bool OriginMap::record(InstId instruction, Milestones milestones) {
  if (findMilestones(instruction)) return false;
  if (origin(milestones.issue) || origin(milestones.done) ||
      (milestones.reads_complete && origin(*milestones.reads_complete))) return false;
  for (const auto event : milestones.effects) if (origin(event)) return false;

  const auto first = std::uint64_t(milestones.issue.value());
  const auto last = std::uint64_t(milestones.done.value());
  if (!milestones.reads_complete && milestones.effects.empty() &&
      last >= first && last - first <= 1) {
    const auto width = static_cast<std::uint32_t>(last - first + 1);
    auto before = runs_.upper_bound(instruction);
    if (before != runs_.begin()) {
      --before;
      auto& run = before->second;
      if (std::uint64_t(before->first.value()) + run.count == instruction.value() &&
          std::uint64_t(run.first.value()) + run.count * run.width == first &&
          run.width == width) {
        ++run.count;
        return true;
      }
    }
    runs_.emplace(instruction, Run{milestones.issue, 1, width});
    event_runs_.emplace(milestones.issue, instruction);
    return true;
  }

  const auto entry = by_instruction_.emplace(instruction, std::move(milestones)).first;
  const auto& recorded = entry->second;
  by_event_.emplace(recorded.issue, instruction);
  by_event_.emplace(recorded.done, instruction);
  if (recorded.reads_complete) by_event_.emplace(*recorded.reads_complete, instruction);
  for (const auto event : recorded.effects) by_event_.emplace(event, instruction);
  return true;
}

std::optional<InstId> OriginMap::origin(EventId event) const {
  auto before = event_runs_.upper_bound(event);
  if (before != event_runs_.begin()) {
    --before;
    const auto& run = runs_.at(before->second);
    const auto delta = std::uint64_t(event.value()) - run.first.value();
    if (delta < run.count * run.width)
      return InstId(static_cast<std::uint32_t>(before->second.value() + delta / run.width));
  }
  const InstId* found = by_event_.get(event);
  return found ? std::optional(*found) : std::nullopt;
}

std::optional<Milestones> OriginMap::milestones(InstId instruction) const {
  auto before = runs_.upper_bound(instruction);
  if (before != runs_.begin()) {
    --before;
    const auto delta = std::uint64_t(instruction.value()) - before->first.value();
    const auto& run = before->second;
    if (delta < run.count) {
      const auto first = std::uint64_t(run.first.value()) + delta * run.width;
      return Milestones{EventId(static_cast<std::uint32_t>(first)),
                        EventId(static_cast<std::uint32_t>(first + run.width - 1))};
    }
  }
  const auto found = by_instruction_.find(instruction);
  return found == by_instruction_.end() ? std::nullopt : std::optional(found->second);
}

std::optional<Milestones> OriginMap::findMilestones(InstId instruction) const {
  return milestones(instruction);
}

std::optional<EventId> OriginMap::resolve(MilestoneRef ref) const {
  auto before = runs_.upper_bound(ref.instruction);
  if (before != runs_.begin()) {
    --before;
    const auto delta = std::uint64_t(ref.instruction.value()) - before->first.value();
    const auto& run = before->second;
    if (delta < run.count) {
      const auto first = run.first.value() + delta * run.width;
      if (ref.milestone == Milestone::kIssue) return EventId(first);
      if (ref.milestone == Milestone::kComplete) return EventId(first + run.width - 1);
      return {};
    }
  }
  const auto found = by_instruction_.find(ref.instruction);
  if (found == by_instruction_.end()) return {};
  const auto& m = found->second;
  if (ref.milestone == Milestone::kIssue) return m.issue;
  if (ref.milestone == Milestone::kComplete) return m.done;
  if (ref.milestone == Milestone::kReadsComplete) return m.reads_complete;
  if (ref.milestone == Milestone::kEffect && ref.effect_index < m.effects.size())
    return m.effects[ref.effect_index];
  return {};
}

// Explicit diagnostic snapshots; ordinary lookups retain interval storage.
std::map<InstId, Milestones> OriginMap::byInstruction() const {
  auto result = by_instruction_;
  for (const auto& [id, run] : runs_) {
    for (std::uint64_t i = 0; i < run.count; ++i) {
      const InstId instruction(static_cast<std::uint32_t>(id.value() + i));
      result.emplace(instruction, *milestones(instruction));
    }
  }
  return result;
}

CompactIdMap<EventId, InstId> OriginMap::byEvent() const {
  auto result = by_event_;
  result.reserveDense(reserved_events_);
  for (const auto& [id, run] : runs_)
    for (std::uint64_t i = 0; i < run.count * run.width; ++i)
      result.emplace(EventId(static_cast<std::uint32_t>(run.first.value() + i)),
                     InstId(static_cast<std::uint32_t>(id.value() + i / run.width)));
  return result;
}

}  // namespace sentryward::algorithm
