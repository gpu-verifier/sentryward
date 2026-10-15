#include "sentryward/frontend/ptx/refinement.h"
#include <algorithm>
#include <limits>
#include <set>
#include <tuple>
namespace sentryward::frontend::ptx {
namespace {
using Link = std::tuple<std::uint32_t, std::uint32_t, std::uint32_t, Milestone,
                        std::uint32_t>;
Link Key(std::uint32_t i, std::uint32_t e, const MilestoneRef &r) {
  return {i, e, r.instruction.value(), r.milestone, r.effect_index};
}
constexpr auto absent = std::numeric_limits<std::uint32_t>::max();
bool Contains(const auto &values, const auto &value) {
  return std::find(values.begin(), values.end(), value) != values.end();
}
} // namespace
nlohmann::json Refine(const SourceProgram &program,
                      const std::vector<RequiredOrder> &orders,
                      const std::map<std::uint32_t, MmaPlan> &native_plans,
                      const std::vector<TcgenHistory> &native_histories,
                      const std::map<std::uint32_t,TcgenCopyPlan> &copies) {
  std::map<std::uint32_t, std::set<std::string>> blocked;
  for (const auto &[id, plan] : native_plans) {
    (void)plan;
    blocked[id];
  }
  std::set<std::uint32_t> retained;
  for (const auto &[id, plan] : native_plans) { (void)plan; retained.insert(id); }
  for (const auto &[id, plan] : copies) { (void)plan; retained.insert(id); }
  for (const auto &history : native_histories)
    for (const auto &record : history.records) retained.insert(record.id);
  // Pin repeatedly queried operations for this immutable refinement input only.
  std::map<std::uint32_t, SourceInstructions::Iterator::Arrow> instructions;
  auto instruction = [&](std::uint32_t id) -> SourceInstructions::Iterator::Arrow {
    if (id >= program.instructions.size()) return {};
    if (retained.contains(id)) {
      auto [entry, inserted] = instructions.try_emplace(id);
      if (inserted) entry->second = program.instructions.view(id);
      return {std::nullopt, entry->second.operator->()};
    }
    return program.instructions.view(id);
  };
  std::set<Link> pipeline;
  auto reject = [&](std::uint32_t id, const char *reason) {
    blocked[id].insert(reason);
  };
  for (const auto &[id, plan] : native_plans) {
    const auto current = instruction(id);
    for (const auto &[target, ref] : plan.copy_links) {
      const auto previous = instruction(ref.instruction.value());
      const auto prior = copies.find(ref.instruction.value());
      const auto *op = current ? std::get_if<AsyncOp>(&current->op) : nullptr;
      const auto *prev = previous ? std::get_if<AsyncOp>(&previous->op) : nullptr;
      const auto address = plan.operand_addresses.find(target);
      if (!current || !previous || current->actor != previous->actor || !op || !prev ||
          prior == copies.end() || (plan.cta_group != 1 && plan.cta_group != 2) ||
          plan.cta_group != prior->second.cta_group || ref.milestone != Milestone::kEffect ||
          !Contains(plan.a_tmem_reads, target) || address == plan.operand_addresses.end() ||
          address->second.first != prior->second.allocation ||
          (!prior->second.any_operand_address &&
           !CopyFeedsOperandBase(prior->second.address, address->second.second)) ||
          target >= op->effects.size() || ref.effect_index >= prev->effects.size() ||
          ref.effect_index % 2 != 1) {
        reject(id, "invalid_copy_pipeline_dependency"); continue;
      }
      const auto &old = prev->effects[ref.effect_index], &next = op->effects[target];
      if (!Contains(next.predecessors, ref) || old.accesses.size() != 1 || next.accesses.size() != 1 ||
          old.accesses.front().mode != AccessMode::kWrite || next.accesses.front().mode != AccessMode::kRead)
        reject(id, "invalid_copy_pipeline_dependency");
      else pipeline.insert(Key(id, target, ref));
    }
    for (const auto &[target, ref] : plan.operand_links) {
      const auto previous = instruction(ref.instruction.value());
      const auto prior=native_plans.find(ref.instruction.value());
      const auto *op=current?std::get_if<AsyncOp>(&current->op):nullptr;
      const auto *prev=previous?std::get_if<AsyncOp>(&previous->op):nullptr;
      if (!current || !previous || current->actor!=previous->actor || !op || !prev ||
          prior==native_plans.end() || (plan.cta_group!=1 && plan.cta_group!=2) ||
          plan.cta_group!=prior->second.cta_group || ref.milestone!=Milestone::kEffect ||
          !Contains(prior->second.a_tmem_reads,ref.effect_index) || !Contains(plan.writes,target) ||
          target>=op->effects.size() || ref.effect_index>=prev->effects.size()) {
        reject(id,"invalid_operand_pipeline_dependency"); continue;
      }
      const auto &old=prev->effects[ref.effect_index], &next=op->effects[target];
      if (!Contains(next.predecessors,ref) || old.accesses.empty() || next.accesses.empty() ||
          std::any_of(old.accesses.begin(),old.accesses.end(),[](const auto &a){return a.mode!=AccessMode::kRead;}) ||
          std::any_of(next.accesses.begin(),next.accesses.end(),[](const auto &a){return a.mode!=AccessMode::kWrite;}))
        reject(id,"invalid_operand_pipeline_dependency");
      else pipeline.insert(Key(id,target,ref));
    }
    std::map<std::uint32_t, MemoryEffect> targets;
    for (auto output : plan.writes) {
      const auto &effect = plan.effects.at(output);
      auto write = std::find_if(
          effect.accesses.begin(), effect.accesses.end(),
          [](const auto &a) { return a.mode == AccessMode::kWrite; });
      if (write == effect.accesses.end())
        throw std::invalid_argument("MMA output has no write");
      auto target = output;
      for (const auto &r : effect.predecessors)
        if (r.instruction.value() == id && r.milestone == Milestone::kEffect) {
          const auto &reads = plan.effects.at(r.effect_index).accesses;
          if (reads.size() == 1 && reads[0].mode == AccessMode::kRead &&
              reads[0].endpoint == write->endpoint &&
              reads[0].bytes == write->bytes) {
            target = r.effect_index;
            break;
          }
        }
      targets.insert_or_assign(target, *write);
    }
    for (const auto &[target, ref] : plan.links) {
      const auto previous = instruction(ref.instruction.value());
      const auto prior = native_plans.find(ref.instruction.value());
      const auto *op = current ? std::get_if<AsyncOp>(&current->op) : nullptr;
      const auto *prev =
          previous ? std::get_if<AsyncOp>(&previous->op) : nullptr;
      if (!current || !previous || current->actor != previous->actor || !op ||
          !prev || prior == native_plans.end() ||
          ref.milestone != Milestone::kEffect ||
          !Contains(prior->second.writes, ref.effect_index) ||
          !targets.contains(target) || target >= op->effects.size() ||
          ref.effect_index >= prev->effects.size() ||
          !Contains(op->effects[target].predecessors, ref) ||
          !Contains(prev->effects[ref.effect_index].accesses,
                    targets.at(target)))
        reject(id, "invalid_pipeline_dependency");
      else
        pipeline.insert(Key(id, target, ref));
    }
  }
  for (const auto &[id, plan] : copies) {
    for (const auto &[target, ref] : plan.operand_links) {
      const auto current = instruction(id), previous = instruction(ref.instruction.value());
      const auto prior = native_plans.find(ref.instruction.value());
      const auto *op = current ? std::get_if<AsyncOp>(&current->op) : nullptr;
      const auto *prev = previous ? std::get_if<AsyncOp>(&previous->op) : nullptr;
      if (!current || !previous || current->actor != previous->actor || !op || !prev ||
          prior == native_plans.end() || plan.cta_group != prior->second.cta_group ||
          (plan.cta_group != 1 && plan.cta_group != 2) || ref.milestone != Milestone::kEffect ||
          !Contains(prior->second.a_tmem_reads, ref.effect_index) || target % 2 != 1 ||
          target >= op->effects.size() || ref.effect_index >= prev->effects.size()) continue;
      const auto &old = prev->effects[ref.effect_index], &next = op->effects[target];
      if (Contains(next.predecessors, ref) && old.accesses.size() == 1 && next.accesses.size() == 1 &&
          old.accesses.front().mode == AccessMode::kRead && next.accesses.front().mode == AccessMode::kWrite)
        pipeline.insert(Key(id, target, ref));
    }
  }
  auto outgoing = [&](const MilestoneRef &ref, std::uint32_t consumer,
                      std::uint32_t effect = absent) {
    if (pipeline.contains(Key(consumer, effect, ref)))
      return;
    if (native_plans.contains(ref.instruction.value()) &&
        ref.instruction.value() != consumer &&
        ref.milestone != Milestone::kIssue &&
        ref.milestone != Milestone::kComplete)
      reject(ref.instruction.value(), "nonaggregate_milestone_consumer");
  };
  for (std::size_t position = 0; position < program.instructions.size(); ++position) {
    const auto view = instruction(static_cast<std::uint32_t>(position));
    const auto &inst = *view;
    const auto id = inst.id.value();
    const auto *op = std::get_if<AsyncOp>(&inst.op);
    if (auto found = native_plans.find(id); found != native_plans.end()) {
      const auto &plan = found->second;
      if (!op)
        reject(id, "not_async_mma");
      else {
        if (!op->completion_requires.empty())
          reject(id, "extra_completion_requirements");
        if (op->effects.size() != plan.effects.size())
          reject(id, "effect_coverage_changed");
        for (std::uint32_t i = 0; i < op->effects.size(); ++i) {
          const auto &effect = op->effects[i];
          if (i >= plan.effects.size() ||
              effect.accesses != plan.effects[i].accesses)
            reject(id, "effect_coverage_changed");
          if (effect.accesses.size() != 1) {
            if (effect.accesses.size() != 2 || !Contains(plan.accumulators, i))
              reject(id, "unproved_atomic_access_group");
            else {
              const auto &r = effect.accesses[0];
              const auto &w = effect.accesses[1];
              if (r.mode != AccessMode::kRead || w.mode != AccessMode::kWrite ||
                  r.endpoint != w.endpoint || r.bytes != w.bytes)
                reject(id, "unproved_atomic_access_group");
            }
          }
          if (effect.completion)
            reject(id, "mma_has_barrier_action");
          for (const auto &a : effect.accesses)
            if (!std::holds_alternative<AddressId>(a.endpoint))
              reject(id, "mma_has_nonmemory_endpoint");
          for (const auto &ref : effect.predecessors) {
            if (ref.milestone == Milestone::kIssue ||
                pipeline.contains(Key(id, i, ref)))
              continue;
            if (ref.instruction != inst.id ||
                ref.milestone != Milestone::kEffect || ref.effect_index >= i) {
              reject(id, "nonissue_effect_prerequisite");
              continue;
            }
            const auto &previous = op->effects[ref.effect_index];
            if (effect.accesses.size() != 1 || previous.accesses.size() != 1) {
              reject(id, "nonscalar_internal_dependency");
              continue;
            }
            const auto &r = previous.accesses[0];
            const auto &w = effect.accesses[0];
            if (r.mode != AccessMode::kRead || w.mode != AccessMode::kWrite ||
                r.endpoint != w.endpoint || r.bytes != w.bytes ||
                (r.bytes != 2 && r.bytes != 4 &&
                 !Contains(plan.split_accumulators,std::pair{ref.effect_index,i})) || i >= plan.effects.size() ||
                !Contains(plan.effects[i].predecessors, ref))
              reject(id, "nonscalar_internal_dependency");
          }
        }
      }
    }
    if (op) {
      for (std::uint32_t i = 0; i < op->effects.size(); ++i)
        for (const auto &ref : op->effects[i].predecessors)
          outgoing(ref, id, i);
      for (const auto &ref : op->completion_requires)
        outgoing(ref, id);
    } else if (const auto *wait = std::get_if<Await>(&inst.op))
      for (const auto &ref : wait->requirements)
        outgoing(ref, id);
  }
  for (const auto &edge : program.dependencies) {
    outgoing(edge.from, edge.to.instruction.value());
    if (native_plans.contains(edge.to.instruction.value()) &&
        edge.to.milestone != Milestone::kIssue)
      reject(edge.to.instruction.value(), "nonissue_endpoint_dependency");
  }
  std::set<const void*> checked_target_domains;
  for (const auto &order : orders) {
    if (native_plans.contains(order.from.instruction.value()) &&
        order.from_effects)
      reject(order.from.instruction.value(), "nonaggregate_required_order");
    const auto check_ref=[&](auto ref) {
      if (native_plans.contains(ref.instruction.value()) &&
          ref.milestone != Milestone::kIssue &&
          ref.milestone != Milestone::kComplete)
        reject(ref.instruction.value(), "nonaggregate_required_order");
    };
    check_ref(order.from);
    if(!order.target_domain || checked_target_domains.insert(order.target_domain.storage_identity()).second)
      for(auto ref:order.targets())check_ref(ref);
  }
  for (const auto &[id, plan] : native_plans) {
    (void)plan;
    if (!instruction(id))
      reject(id, "missing_mma_instruction");
  }
  std::set<std::uint32_t> proved;
  nlohmann::json mma_blocked = nlohmann::json::array();
  for (const auto &[id, reasons] : blocked) {
    if (reasons.empty())
      proved.insert(id);
    for (const auto &reason : reasons)
      mma_blocked.push_back({id, reason});
  }

  const auto mma_proved=proved.size();
  std::map<std::uint32_t,std::set<std::string>> copy_blocked;
  for (const auto &[id,plan] : copies) {
    const auto &expected = plan.effects;
    auto &reasons=copy_blocked[id];
    const auto inst = instruction(id);
    const auto *op=inst?std::get_if<AsyncOp>(&inst->op):nullptr;
    if (!op || op->effects!=expected || !op->completion_requires.empty()) {
      reasons.insert("copy_effects_changed"); continue;
    }
    if (expected.empty() || expected.size()%2) reasons.insert("incomplete_copy_pair");
    for (std::uint32_t index=0;index+1<expected.size();index+=2) {
      const auto &read=expected[index], &write=expected[index+1];
      if (read.accesses.size()!=1 || write.accesses.size()!=1 || read.completion || write.completion ||
          !read.predecessors.empty() ||
          !Contains(write.predecessors, MilestoneRef{InstId(id),Milestone::kEffect,index}) ||
          std::any_of(write.predecessors.begin(), write.predecessors.end(), [&,id=id](const auto &ref) {
            return ref != MilestoneRef{InstId(id),Milestone::kEffect,index} && !pipeline.contains(Key(id,index+1,ref));
          })) {
        reasons.insert("invalid_copy_pair"); continue;
      }
      const auto &a=read.accesses.front(), &b=write.accesses.front();
      if (a.mode!=AccessMode::kRead || b.mode!=AccessMode::kWrite ||
          !std::holds_alternative<AddressId>(a.endpoint) || !std::holds_alternative<AddressId>(b.endpoint) ||
          !a.bytes || !*a.bytes || a.bytes!=b.bytes) reasons.insert("invalid_copy_footprint");
    }
  }
  auto copy_consumer=[&](const MilestoneRef &ref,InstId consumer,std::uint32_t effect=absent) {
    if (pipeline.contains(Key(consumer.value(), effect, ref))) return;
    if (copies.contains(ref.instruction.value()) && ref.instruction!=consumer &&
        ref.milestone!=Milestone::kIssue && ref.milestone!=Milestone::kComplete)
      copy_blocked[ref.instruction.value()].insert("nonaggregate_copy_consumer");
  };
  for (std::size_t position = 0; position < program.instructions.size(); ++position) {
    if (!program.instructions.holdsOperation<AsyncOp>(position) &&
        !program.instructions.holdsOperation<Await>(position)) continue;
    const auto view = instruction(static_cast<std::uint32_t>(position));
    const auto &inst = *view;
    if (const auto *op=std::get_if<AsyncOp>(&inst.op)) {
      for (std::uint32_t effect=0; effect<op->effects.size(); ++effect)
        for (auto ref : op->effects[effect].predecessors) copy_consumer(ref,inst.id,effect);
      for (auto ref : op->completion_requires) copy_consumer(ref,inst.id);
    } else if (const auto *op=std::get_if<Await>(&inst.op))
      for (auto ref : op->requirements) copy_consumer(ref,inst.id);
  }
  for (const auto &edge : program.dependencies) {
    copy_consumer(edge.from,edge.to.instruction);
    if (copies.contains(edge.to.instruction.value()) && edge.to.milestone!=Milestone::kIssue)
      copy_blocked[edge.to.instruction.value()].insert("external_copy_effect_dependency");
  }
  nlohmann::json copy_reasons=nlohmann::json::array();
  for (const auto &[id,reasons] : copy_blocked) {
    if (reasons.empty()) proved.insert(id);
    for (const auto &reason : reasons) copy_reasons.push_back({id,reason});
  }
  std::set<std::uint32_t> requested;
  for (const auto &h : native_histories)
    for (const auto &r : h.records)
      requested.insert(r.id);
  std::set<ActorId> active_actors, incomplete;
  for (auto id : requested)
    if (id < program.instructions.size())
      active_actors.insert(program.instructions.actor(id));
  const auto n = program.instructions.size();
  std::vector<std::uint32_t> next(n, absent), prev(n, absent);
  std::vector<std::int64_t> positions(n, -1);
  std::map<ActorId, std::vector<std::uint32_t>> nodes;
  for (std::size_t position = 0; position < n; ++position) {
    const auto actor = program.instructions.actor(position);
    if (active_actors.contains(actor))
      nodes[actor].push_back(program.instructions.id(position).value());
  }
  for (const auto &edge : program.edges)
    if (edge.kind == SourceEdgeKind::kProgramOrder) {
      const bool s = edge.from.value() < n, t = edge.to.value() < n;
      const auto source_actor = s ? program.instructions.actor(edge.from.value()) : ActorId{};
      const auto target_actor = t ? program.instructions.actor(edge.to.value()) : ActorId{};
      bool a = s && active_actors.contains(source_actor),
           b = t && active_actors.contains(target_actor);
      if (!a && !b)
        continue;
      if (!a || !b || source_actor != target_actor) {
        if (a)
          incomplete.insert(source_actor);
        if (b)
          incomplete.insert(target_actor);
        continue;
      }
      const auto from = edge.from.value(), to = edge.to.value();
      if ((next[from] != absent && next[from] != to) ||
          (prev[to] != absent && prev[to] != from))
        incomplete.insert(source_actor);
      next[from] = to;
      prev[to] = from;
    }
  for (const auto &[actor, ids] : nodes) {
    std::uint32_t head = absent;
    std::size_t heads = 0;
    for (auto id : ids)
      if (prev[id] == absent) {
        head = id;
        ++heads;
      }
    if (heads != 1) {
      incomplete.insert(actor);
      continue;
    }
    std::int64_t position = 0;
    for (auto id = head; id != absent; id = next[id]) {
      if (positions[id] >= 0) {
        incomplete.insert(actor);
        break;
      }
      positions[id] = position++;
    }
    if (static_cast<std::size_t>(position) != ids.size())
      incomplete.insert(actor);
  }
  std::set<std::pair<std::int64_t, std::string>> history_blocked;
  std::set<std::uint32_t> seen;
  std::size_t operations = 0, notifications = 0, records = 0;
  for (const auto &history : native_histories) {
    std::map<int, std::vector<std::uint32_t>> work, commits;
    // A prefix refers only to a proved, pure effect-0 conjunction. In
    // particular, a prior commit's aggregate completion or arrival effects
    // cannot stand in for the prefix: either would order the notifications.
    struct Prefix {
      MilestoneRef gate;
      std::size_t covered;
    };
    std::map<int, Prefix> prefixes;
    std::optional<ActorId> actor;
    std::int64_t last = -1;
    for (const auto &record : history.records) {
      ++records;
      const auto id = record.id;
      bool record_valid = true;
      const auto bad = [&](const char *reason) {
        record_valid = false;
        history_blocked.emplace(id, reason);
      };
      if (!seen.insert(id).second)
        bad("duplicate_tcgen_history_record");
      const auto inst = instruction(id);
      if (!inst) {
        bad("missing_tcgen_instruction");
        continue;
      }
      if (!actor)
        actor = inst->actor;
      if (*actor != inst->actor)
        bad("tcgen_history_crosses_issuers");
      if (positions[id] <= last || incomplete.contains(inst->actor))
        bad("unproved_tcgen_issue_order");
      last = positions[id];
      if (record.group != 1 && record.group != 2)
        bad("invalid_tcgen_group");
      const auto *op = std::get_if<AsyncOp>(&inst->op);
      if (!op) {
        bad("tcgen_record_is_not_async");
        continue;
      }
      if (!op->completion_requires.empty())
        bad("tcgen_extra_completion_requirement");
      if (record.operation == "commit") {
        ++notifications;
        commits[record.group].push_back(id);
        const auto found = work.find(record.group);
        const std::size_t work_count =
            found == work.end() ? 0 : found->second.size();
        const auto matches_work = [&](const std::vector<MilestoneRef> &refs,
                                      std::size_t offset, std::size_t covered) {
          if (covered > work_count ||
              refs.size() != offset + work_count - covered)
            return false;
          for (std::size_t i = covered; i < work_count; ++i)
            if (refs[offset + i - covered] !=
                MilestoneRef{InstId(found->second[i]), Milestone::kComplete})
              return false;
          return true;
        };
        if (op->effects.empty())
          bad("empty_tcgen_notification");
        // A compact notification has one pure conjunction effect followed by
        // one or more ordinary arrivals. Validate the conjunction incrementally
        // against the already proved prefix, not by expanding all its leaves.
        const bool compact =
            !op->effects.empty() && !op->effects.front().completion;
        const MilestoneRef gate{inst->id, Milestone::kEffect, 0};
        if (compact) {
          const auto &effect = op->effects.front();
          if (!effect.accesses.empty())
            bad("tcgen_prefix_gate_has_memory_access");
          if (op->effects.size() < 2)
            bad("empty_tcgen_notification");
          const auto prior = prefixes.find(record.group);
          const std::size_t offset = prior == prefixes.end() ? 0 : 1;
          const std::size_t covered =
              prior == prefixes.end() ? 0 : prior->second.covered;
          if ((offset && (effect.predecessors.empty() ||
                          effect.predecessors.front() != prior->second.gate)) ||
              !matches_work(effect.predecessors, offset, covered))
            bad("tcgen_notification_tracking_mismatch");
        }
        for (std::size_t i = compact ? 1 : 0; i < op->effects.size(); ++i) {
          const auto &effect = op->effects[i];
          const auto *action =
              effect.completion ? std::get_if<ArriveEffect>(&*effect.completion)
                                : nullptr;
          if (!effect.accesses.empty() || !action || action->arrivals != 1)
            bad("tcgen_notification_not_ordinary_arrival");
          const bool tracked = compact
                                   ? effect.predecessors.size() == 1 &&
                                         effect.predecessors.front() == gate
                                   : matches_work(effect.predecessors, 0, 0);
          if (!tracked)
            bad("tcgen_notification_tracking_mismatch");
        }
        if (compact && record_valid)
          prefixes.insert_or_assign(record.group, Prefix{gate, work_count});
        else
          // A legacy flat commit has no reusable gate, even though its first
          // arrival also happens to have effect index zero.
          prefixes.erase(record.group);
      } else if (record.operation == "mma" || record.operation == "cp" ||
                 record.operation == "shift") {
        ++operations;
        work[record.group].push_back(id);
        if (!proved.contains(id))
          bad("unproved_tcgen_memory_refinement");
        for (std::uint32_t i = 0; i < op->effects.size(); ++i) {
          const auto &effect = op->effects[i];
          if (effect.completion)
            bad("tcgen_work_has_barrier_action");
          for (const auto &ref : effect.predecessors) {
            if (ref.instruction == inst->id &&
                (ref.milestone == Milestone::kIssue ||
                 ref.milestone == Milestone::kEffect))
              continue;
            const auto prior = instruction(ref.instruction.value());
            bool linked = pipeline.contains(Key(id, i, ref)) &&
                          proved.contains(ref.instruction.value());
            if ((ref.milestone != Milestone::kIssue && !linked) || !prior ||
                prior->actor != inst->actor ||
                positions[ref.instruction.value()] >= positions[id])
              bad("tcgen_work_needs_unissued_or_external_work");
          }
        }
      } else
        bad("unsupported_tcgen_history_operation");
    }
    if (work != history.work || commits != history.commits)
      history_blocked.emplace(
          history.records.empty()
              ? std::int64_t{-1}
              : static_cast<std::int64_t>(history.records.front().id),
          "unregistered_tcgen_history_entries");
  }
  for (const auto &edge : program.dependencies)
    if (requested.contains(edge.to.instruction.value()) &&
        edge.to.milestone != Milestone::kIssue)
      history_blocked.emplace(edge.to.instruction.value(),
                              "tcgen_external_work_dependency");
  nlohmann::json result;
  result["mma_refinement"] = {{"discharged", mma_proved},
                              {"blocked", mma_blocked}};
  nlohmann::json history_reasons = nlohmann::json::array();
  for (const auto &reason : history_blocked)
    history_reasons.push_back({reason.first, reason.second});
  result["tcgen_history_refinement"] = {{"operations", operations},
                                        {"notifications", notifications},
                                        {"blocked", history_reasons}};
  nlohmann::json discharged = nlohmann::json::array();
  if (!copies.empty()) {
    result["tcgen_copy_refinement"]={{"discharged",proved.size()-mma_proved},{"blocked",copy_reasons}};
    if (copy_reasons.empty()) discharged.push_back("tcgen_cp_memory_effect_refinement");
  }
  if (mma_blocked.empty())
    discharged.push_back("mma_memory_effect_refinement");
  if (records && history_blocked.empty()) {
    discharged.push_back("tcgen_pipeline_effect_refinement");
    discharged.push_back("tcgen_commit_notification_order_refinement");
  }
  result["discharged_codes"] = discharged;
  return result;
}
} // namespace sentryward::frontend::ptx
