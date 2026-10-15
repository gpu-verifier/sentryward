#pragma once

#include "sentryward/algorithm/source_ir/async.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace sentryward::frontend::ptx {
using namespace sentryward::algorithm;

// A blocking wait makes these completions ancestors of all later issues by
// actor program order. Acknowledge only after the Await has been emitted.
class PendingCompletions {
public:
  void Issue(InstId id) {
    pending_.push_back(id);
    ++issued_;
  }
  const std::vector<InstId> &Pending() const { return pending_; }
  std::uint64_t Issued() const { return issued_; }
  void Acknowledge() { pending_.clear(); }

private:
  std::vector<InstId> pending_;
  std::uint64_t issued_ = 0;
};

// Waits may alternate between reads-complete and fully-complete, and their
// pending-group operand may increase or decrease. Keep separate monotone
// frontiers. Complete dominates reads-complete in the lowered AsyncOp graph;
// a read wait must never advance the full-completion frontier.
class CommittedGroups {
public:
  struct WaitPlan {
    std::vector<MilestoneRef> requirements;
    std::size_t end = 0;
    bool reads = false;
    std::uint64_t original_references = 0;
  };
  void Issue(InstId id) {
    open_.push_back(id);
    all_.push_back(id);
  }
  void Commit() {
    prefix_sizes_.push_back(prefix_sizes_.back() + open_.size());
    committed_.push_back(std::move(open_));
    open_.clear();
  }
  const std::vector<InstId> &All() const { return all_; }
  WaitPlan PrepareWait(std::uint64_t pending, bool reads = false) const {
    WaitPlan plan;
    plan.end = committed_.size() > pending ? committed_.size() - pending : 0;
    plan.reads = reads;
    plan.original_references = prefix_sizes_[plan.end];
    const auto begin = reads ? reads_ : complete_;
    if (plan.end <= begin)
      return plan;
    plan.requirements.reserve(prefix_sizes_[plan.end] - prefix_sizes_[begin]);
    for (auto group = begin; group < plan.end; ++group)
      for (auto id : committed_[group])
        plan.requirements.push_back(
            {id, reads ? Milestone::kReadsComplete : Milestone::kComplete});
    return plan;
  }
  void Acknowledge(const WaitPlan &plan) {
    reads_ = std::max(reads_, plan.end);
    if (!plan.reads)
      complete_ = std::max(complete_, plan.end);
  }

private:
  std::vector<InstId> open_, all_;
  std::vector<std::vector<InstId>> committed_;
  std::vector<std::size_t> prefix_sizes_{0};
  std::size_t reads_ = 0, complete_ = 0;
};

// A prefix is represented by effect 0 of a notification AsyncOp. That effect
// has no memory access or barrier action. It depends on the preceding *pure
// gate* and newly issued work; notification effects depend on this gate.
// Never use the preceding notification or AsyncOp.complete as the prefix:
// doing so would introduce an ordering between independent notifications.
class CompletionPrefix {
public:
  template <class Id>
  std::vector<MilestoneRef> Prepare(const std::vector<Id> &work) const {
    if (work.size() < covered_)
      throw std::logic_error("completion history shrank behind its prefix");
    std::vector<MilestoneRef> result;
    result.reserve(work.size() - covered_ + (previous_ ? 1 : 0));
    if (previous_)
      result.push_back(*previous_);
    for (auto i = covered_; i < work.size(); ++i) {
      if constexpr (std::is_same_v<Id, InstId>)
        result.push_back({work[i], Milestone::kComplete});
      else
        result.push_back({InstId(work[i]), Milestone::kComplete});
    }
    return result;
  }
  void Acknowledge(InstId notification, std::size_t work_count) {
    previous_ = MilestoneRef{notification, Milestone::kEffect, 0};
    covered_ = work_count;
  }

private:
  std::optional<MilestoneRef> previous_;
  std::size_t covered_ = 0;
};

struct DependencyCounts {
  std::uint64_t consumers = 0, original_references = 0, emitted_references = 0,
                prefix_effects = 0;
  void Record(std::size_t original, std::size_t emitted, bool prefix = false) {
    ++consumers;
    original_references += original;
    emitted_references += emitted;
    prefix_effects += prefix;
  }
};
} // namespace sentryward::frontend::ptx
