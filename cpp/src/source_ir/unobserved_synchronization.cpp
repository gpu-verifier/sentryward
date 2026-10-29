#include "sentryward/algorithm/source_ir/unobserved_synchronization.h"

#include <map>
#include <optional>
#include <set>
#include <type_traits>
#include <utility>
#include <variant>

namespace sentryward::algorithm {
namespace {

struct BarrierUses final {
  std::set<std::uint32_t> arrival_counts;
  // Initializer and expected arrivals of an mbarrier.
  std::optional<std::pair<InstId, std::uint32_t>> initializer;
  bool named = false;
  bool observed = false;
  // Only its arrivals change the pending count: no transaction count, no added arrivals.
  bool arrivals_only = true;
};

// The barriers an instruction's eliminable operations arrive on.
template <typename Visit>
void ForEachArrival(const SourceOp& operation, Visit&& visit) {
  if (const auto* op = std::get_if<NamedBarrierArrive>(&operation)) {
    visit(op->barrier);
  } else if (const auto* op = std::get_if<MBarrierArrive>(&operation)) {
    visit(op->barrier);
  } else if (const auto* op = std::get_if<AsyncOp>(&operation)) {
    for (const auto& effect : op->effects)
      if (const auto* arrival = effect.completion ? std::get_if<ArriveEffect>(&*effect.completion) : nullptr)
        visit(arrival->barrier);
  }
}

}  // namespace

UnobservedSynchronization EliminateUnobservedSynchronization(SourceProgram& program) {
  std::map<BarrierLifetimeId, BarrierUses> barriers;
  program.instructions.forEachAffineColumn([&](const SourceInstruction& first, const auto&, std::size_t,
                                              std::size_t, std::size_t) {
    std::visit([&](const auto& op) {
      using T = std::decay_t<decltype(op)>;
      if constexpr (std::is_same_v<T, NamedBarrierArrive>) {
        auto& uses = barriers[op.barrier];
        uses.named = true;
        uses.arrival_counts.insert(op.expected_arrivals);
      } else if constexpr (std::is_same_v<T, NamedBarrierSync> || std::is_same_v<T, MBarrierWaitState> ||
                           std::is_same_v<T, MBarrierWaitParity> || std::is_same_v<T, MBarrierTryWaitState> ||
                           std::is_same_v<T, MBarrierTryWaitParity>) {
        barriers[op.barrier].observed = true;
      } else if constexpr (std::is_same_v<T, MBarrierInit>) {
        barriers[op.barrier].initializer = {first.id, op.expected_arrivals};
      } else if constexpr (std::is_same_v<T, MBarrierArrive>) {
        auto& uses = barriers[op.barrier];
        uses.arrival_counts.insert(op.arrivals);
        uses.observed |= op.result.has_value();
      } else if constexpr (std::is_same_v<T, MBarrierArriveExpectTx>) {
        auto& uses = barriers[op.barrier];
        uses.arrivals_only = false;
        uses.observed |= op.result.has_value();
      } else if constexpr (std::is_same_v<T, MBarrierExpectTx> || std::is_same_v<T, MBarrierCompleteTx> ||
                           std::is_same_v<T, MBarrierAddArrivals>) {
        barriers[op.barrier].arrivals_only = false;
      } else if constexpr (std::is_same_v<T, AsyncLoad> || std::is_same_v<T, AsyncStore>) {
        if (op.completion) barriers[op.completion->barrier].arrivals_only = false;
      } else if constexpr (std::is_same_v<T, AsyncOp>) {
        for (const auto& effect : op.effects) {
          if (!effect.completion) continue;
          if (const auto* arrival = std::get_if<ArriveEffect>(&*effect.completion))
            barriers[arrival->barrier].arrival_counts.insert(arrival->arrivals);
          else
            barriers[std::get<CompleteTxEffect>(*effect.completion).barrier].arrivals_only = false;
        }
      }
    }, first.op);
  });
  std::set<BarrierLifetimeId> unobserved;
  for (const auto& [barrier, uses] : barriers) {
    if (uses.observed || uses.arrival_counts.size() != 1) continue;
    // Pending counts stay multiples of the arrival count, so no arrival underflows them.
    if (uses.named || (uses.arrivals_only && uses.initializer &&
                       uses.initializer->second % *uses.arrival_counts.begin() == 0))
      unobserved.insert(barrier);
  }
  UnobservedSynchronization result;
  if (unobserved.empty()) return result;
  std::map<BarrierLifetimeId, std::vector<MilestoneRef>> uses;
  program.instructions.forEachAffineColumn([&](const SourceInstruction& first, const auto& delta, std::size_t,
                                              std::size_t, std::size_t count) {
    std::set<BarrierLifetimeId> arrivals;
    ForEachArrival(first.op, [&](BarrierLifetimeId barrier) {
      if (unobserved.contains(barrier)) arrivals.insert(barrier);
    });
    for (const auto barrier : arrivals) {
      const bool named = barriers.at(barrier).named;
      result.eliminated.push_back({named ? EliminatedBarrierKind::kNamed : EliminatedBarrierKind::kMBarrier,
                                   barrier, first.id, first.location, count});
      if (named) continue;
      auto& refs = uses[barrier];
      for (std::size_t member = 0; member < count; ++member)
        refs.push_back({InstId(static_cast<std::uint32_t>(first.id.value() + delta[0] * std::int64_t(member))),
                        Milestone::kIssue});
    }
  });
  for (auto& [barrier, refs] : uses)
    result.initialization_orders.push_back({barriers.at(barrier).initializer->first, std::move(refs)});
  program.instructions.rewriteRecords([unobserved = std::move(unobserved)](SourceInstruction& instruction) {
    if (auto* op = std::get_if<AsyncOp>(&instruction.op)) {
      for (auto& effect : op->effects)
        if (const auto* arrival = effect.completion ? std::get_if<ArriveEffect>(&*effect.completion) : nullptr;
            arrival && unobserved.contains(arrival->barrier))
          effect.completion.reset();
      return;
    }
    bool eliminated = false;
    ForEachArrival(instruction.op, [&](BarrierLifetimeId barrier) { eliminated = unobserved.contains(barrier); });
    if (eliminated) instruction.op = Await{};
  });
  return result;
}

}  // namespace sentryward::algorithm
