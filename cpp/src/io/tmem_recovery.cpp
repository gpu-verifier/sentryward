#include "sentryward/algorithm/io/tmem_recovery.h"
#include "sentryward/algorithm/base/compact_vector.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace sentryward::algorithm {
namespace {
using RefKey = std::tuple<InstId, Milestone, std::uint32_t>;
struct InstHash {
  std::size_t operator()(InstId id) const noexcept { return id.value(); }
};
struct EffectSelectionHash {
  std::size_t operator()(const std::pair<InstId, const void*>& selection) const noexcept {
    return std::hash<const void*>{}(selection.second) ^
        (static_cast<std::size_t>(selection.first.value()) * 0x9e3779b9U);
  }
};
RefKey Key(const MilestoneRef& ref) {
  return {ref.instruction, ref.milestone, ref.effect_index};
}
bool AllocationOrder(const RequiredOrder& order) {
  return order.kind == "allocation_before_dealloc";
}
bool RetirementOrder(const RequiredOrder& order) {
  return AllocationOrder(order) || order.kind == "access_completion_before_dealloc";
}
}  // namespace

std::vector<TmemDeallocationRecovery> RecoverSsdDeallocations(SourceArtifact& input) {
  const auto violations = std::count_if(input.source_violations.begin(), input.source_violations.end(),
      [](const auto& item) { return item.code == "tmem_multiple_dealloc"; });
  if (violations == 0) return {};
  std::unordered_map<InstId, SourceInstructions::Handle, InstHash> instructions;
  std::map<ActorId, ActorInfo> actors;
  std::map<CtaId, std::set<std::tuple<WarpId, std::uint32_t, std::uint32_t>>> allocations;
  std::uint32_t next_id = 0, next_async = 0;
  for (const auto& actor : input.program.actors) actors.emplace(actor.id, actor);
  for (std::size_t i=0;i<input.program.instructions.size();++i) {
    const auto inst=input.program.instructions[i];
    if (!instructions.emplace(inst.id, input.program.instructions.ref(i)).second || !actors.contains(inst.actor)) return {};
    if (inst.id.value() == std::numeric_limits<std::uint32_t>::max()) return {};
    next_id = std::max(next_id, inst.id.value() + 1);
    std::visit([&](const auto& op) {
      if constexpr (requires { op.async_id; })
        next_async = std::max(next_async, op.async_id.value());
    }, inst.op);
    if (inst.ptx_origin && inst.ptx_origin->opcode.starts_with("tcgen05.alloc.")) {
      const auto& actor = actors.at(inst.actor);
      allocations[actor.cta].emplace(actor.warp, inst.ptx_origin->instruction_index,
                                    inst.ptx_origin->occurrence);
    }
  }
  struct Group {
    InstId allocation;
    CtaId cta;
    std::set<InstId> targets;
    std::set<RefKey> prerequisites;
    // Each shared selection contributes once to this retirement conjunction.
    // All original per-target RequiredOrders are still processed below.
    std::unordered_set<std::pair<InstId, const void*>, EffectSelectionHash> selections;
  };
  std::map<InstId, Group> groups;
  std::map<InstId, InstId> owner;
  for (const auto& order : input.required_orders) {
    if (!AllocationOrder(order)) continue;
    for(const auto& target_ref:order.targets()) {
    const auto a = instructions.find(order.from.instruction);
    const auto d = instructions.find(target_ref.instruction);
    if (a == instructions.end() || d == instructions.end()) return {};
    const auto& alloc = *a->second;
    const auto& dealloc = *d->second;
    if (!alloc.ptx_origin || !dealloc.ptx_origin) continue;
    if (alloc.ptx_origin->opcode != "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32" ||
        dealloc.ptx_origin->opcode != "tcgen05.dealloc.cta_group::1.sync.aligned.b32") continue;
    const auto& actor = actors.at(alloc.actor);
    if (actor.warp != WarpId(1) || actor.lane != 0 ||
        actors.at(dealloc.actor).cta != actor.cta ||
        !std::holds_alternative<AsyncOp>(alloc.op) || !std::holds_alternative<Await>(dealloc.op) ||
        allocations[actor.cta].size() != 1 || order.from.milestone != Milestone::kComplete ||
        target_ref.milestone != Milestone::kIssue) return {};
    auto [it, inserted] = groups.try_emplace(alloc.id, Group{alloc.id, actor.cta, {}, {}, {}});
    (void)inserted;
    it->second.targets.insert(dealloc.id);
    auto [target, fresh] = owner.emplace(dealloc.id, alloc.id);
    if (!fresh && target->second != alloc.id) return {};
    }
  }
  if (groups.size() != static_cast<std::size_t>(violations) ||
      groups.size() >= std::numeric_limits<std::uint32_t>::max() - next_id ||
      groups.size() >= std::numeric_limits<std::uint32_t>::max() - next_async) return {};
  std::set<CtaId> recovered_ctas;
  for (auto& [id, group] : groups) {
    if (!recovered_ctas.insert(group.cta).second || group.targets.size() != 128) return {};
    std::set<std::pair<WarpId, std::uint32_t>> participants;
    std::set<std::tuple<std::string, std::uint32_t, std::uint32_t>> sites;
    for (const auto target : group.targets) {
      const auto& inst = *instructions.at(target);
      const auto& actor = actors.at(inst.actor);
      if (actor.warp.value() < 8 || actor.warp.value() > 11 || actor.lane >= 32 ||
          !participants.emplace(actor.warp, actor.lane).second) return {};
      sites.emplace(inst.ptx_origin->file, inst.ptx_origin->instruction_index,
                    inst.ptx_origin->occurrence);
      group.prerequisites.emplace(target, Milestone::kComplete, 0);
    }
    if (sites.size() != 1) return {};
    group.prerequisites.emplace(id, Milestone::kComplete, 0);
  }
  std::map<const void*,std::set<InstId>> domain_owners;
  for (const auto& order : input.required_orders) {
    std::set<InstId> singleton;
    const std::set<InstId>* allocations=&singleton;
    if(order.target_domain) {
      auto [it,fresh]=domain_owners.try_emplace(order.target_domain.storage_identity());
      if(fresh)for(auto ref:order.targets()) {
        const auto target=owner.find(ref.instruction);
        if(target!=owner.end())it->second.insert(target->second);
      }
      allocations=&it->second;
    } else if(const auto target=owner.find(order.to.instruction);target!=owner.end())
      singleton.insert(target->second);
    if(!allocations->empty() && !RetirementOrder(order))return {};
    for(auto allocation:*allocations) {
    auto& group = groups.at(allocation);
    if (order.from_effects) {
      if (!group.selections.emplace(order.from.instruction,
                                    order.from_effects.storage_identity()).second)
        continue;
      for (const auto effect : *order.from_effects)
        group.prerequisites.emplace(order.from.instruction, Milestone::kEffect, effect);
    } else {
      group.prerequisites.insert(Key(order.from));
    }
    }
  }
  std::vector<TmemDeallocationRecovery> result;
  std::vector<SourceInstruction> sinks;
  std::map<InstId, InstId> retirement;
  for (const auto& [id, group] : groups) {
    const auto& first = *instructions.at(*group.targets.begin());
    const InstId sink(next_id++);
    std::vector<MilestoneRef> requirements;
    for (const auto& [instruction, milestone, effect] : group.prerequisites)
      requirements.push_back({instruction, milestone, effect});
    sinks.push_back({sink, first.actor, {"<recovery:deferred-tmem-deallocation>", 0, 0},
                     AsyncOp{AsyncOpId(++next_async), {}, std::move(requirements)}});
    input.program.edges.push_back({first.id, sink, SourceEdgeKind::kProgramOrder});
    retirement.emplace(id, sink);
    result.push_back({group.cta, id, sink, {group.targets.begin(), group.targets.end()},
                      instructions.at(id)->location, first.location});
  }
  for (auto& sink : sinks) input.program.instructions.push_back(std::move(sink));
  using OrderKey = std::tuple<SharedString, RefKey, std::vector<RefKey>,
                             SharedOptional<std::vector<std::uint32_t>>>;
  std::set<OrderKey> seen;
  std::size_t out = 0;
  std::map<const void*,SharedOptional<std::vector<MilestoneRef>>> rewritten_domains;
  for (auto& order : input.required_orders) {
    bool changed=false;
    if(order.target_domain) {
      auto [it,fresh]=rewritten_domains.try_emplace(order.target_domain.storage_identity());
      if(fresh) {
        std::vector<MilestoneRef> refs;
        std::set<RefKey> unique;
        for(auto ref:order.targets()) {
          const auto found=owner.find(ref.instruction);
          if(found!=owner.end())ref={retirement.at(found->second),Milestone::kComplete};
          if(unique.insert(Key(ref)).second)refs.push_back(ref);
        }
        it->second=std::move(refs);
      }
      order.target_domain=it->second;order.to=order.target_domain->front();changed=true;
    } else if(const auto found=owner.find(order.to.instruction);found!=owner.end()) {
      order.to={retirement.at(found->second),Milestone::kComplete};changed=true;
    }
    if(changed) {
      std::vector<RefKey> keys;for(auto ref:order.targets())keys.push_back(Key(ref));
      if(!seen.emplace(order.kind,Key(order.from),std::move(keys),order.from_effects).second)continue;
    }
    if (&order != &input.required_orders[out]) input.required_orders[out] = std::move(order);
    ++out;
  }
  input.required_orders.resize(out);
  CompactVector(input.required_orders);
  std::erase_if(input.source_violations, [](const auto& item) { return item.code == "tmem_multiple_dealloc"; });
  return result;
}

}  // namespace sentryward::algorithm
