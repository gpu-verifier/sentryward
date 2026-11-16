#pragma once
#include "lowering_impl.hpp"

namespace sentryward::frontend::ptx {
void ModelLowerer::Impl::CollectGroup2() {
  if (!tmem)
    return;
  std::map<TmemAllocationId, const TmemAllocation *> by_id;
  for (const auto &allocation : tmem->provenance.allocations)
    if (allocation.cta_group == 2)
      by_id.emplace(allocation.identifier, &allocation);
  if (by_id.empty())
    return;
  ProfileScope profile("frontend.model.group2_collect");
  std::map<TmemAllocationId, std::vector<MilestoneRef>> accesses;
  std::set<std::pair<TmemAllocationId,InstId>> visited;
  for (const auto &use : tmem->uses) {
    if (!by_id.contains(use.allocation))
      continue;
    auto found = sites.find(Key(use.site));
    if (found == sites.end())
      continue;
    for (auto id : found->second) {
      if (!visited.emplace(use.allocation,id).second) continue;
      const auto instruction = source.instruction(id);
      const auto *op = std::get_if<AsyncOp>(&instruction.op);
      if (!op)
        continue;
      auto &refs = accesses[use.allocation];
      if (plans.contains(id.value())) {
        refs.push_back(Ref(id));
        continue;
      }
      auto allocation = by_id.at(use.allocation);
      auto resource = allocation->columns == 512
                          ? std::string("physical.tmem.cta.0")
                          : use.allocation.ResourceName();
      const auto &frame = resources->frame;
      auto own_rank = frame.self().value;
      std::set<std::string> names;
      for (auto rank : {own_rank & ~1u, own_rank | 1u}) {
        auto name = frame.Format(AllocationKey::Tensor(frame.Rank(rank), resource)).allocation;
        names.insert(std::move(name));
      }
      bool has_tensor = false, matched = false;
      for (std::uint32_t index = 0; index < op->effects.size(); ++index) {
        bool match = false;
        for (const auto &access : op->effects[index].accesses)
          if (access.bytes)
            if (auto memory = std::get_if<AddressId>(&access.endpoint)) {
              const auto &binding = resources->Binding(*memory);
              if (binding.address_space == "tensor") {
                has_tensor = true;
                match |= names.contains(binding.allocation);
              }
            }
        if (match) {
          matched = true;
          refs.push_back(Ref(id, Milestone::kEffect, index));
        }
      }
      if (has_tensor && !matched)
        Diagnostic("unresolved_lifetime_footprint",
                   "TMEM retirement does not match the recovered allocation");
    }
  }
  std::vector<
      std::tuple<TmemAtomSite, std::string, std::optional<TmemAllocationId>>>
      records;
  for (const auto &allocation : tmem->provenance.allocations)
    if (allocation.cta_group == 2)
      records.emplace_back(allocation.site, "alloc", allocation.identifier);
  for (const auto &[site, identifier] : tmem->deallocations)
    if (by_id.contains(identifier))
      records.emplace_back(site, "dealloc", identifier);
  std::set<Site> seen;
  for (std::size_t k = 0; k < facts->LaneCount(); ++k) {
    const auto lane = facts->Lane(k);
    const auto warp = facts->Warp(lane);
    for (std::size_t index=0;index<facts->Size(lane);++index) {
      const auto e=facts->Meta({lane,index});
      const auto &instruction = function.instructions[e.instruction_id()];
      if (instruction.opcode == "tcgen05" && Has(instruction, "cta_group::2") &&
          Has(instruction, "relinquish_alloc_permit") &&
          seen.insert(Key(warp, e)).second)
        records.emplace_back(TmemAtomSite{warp, e.instruction_id(),
                                          e.occurrence(), instruction.line,
                                          opcode_texts[instruction.id]},
                             "relinquish_alloc_permit", std::nullopt);
    }
  }
  for (const auto &[site, operation, allocation_id] : records) {
    Group2LifetimeSite record;
    record.cta_coordinates = resources->frame.SelfCoordinates();
    record.operation = operation;
    record.instruction_index = site.instruction_index;
    record.occurrence = site.occurrence;
    record.warp = site.warp;
    record.allocation_key = allocation_id;
    std::map<ActorId, std::pair<InstId, InstId>> ends;
    auto found = sites.find(Key(site));
    if (found != sites.end())
      for (auto id : found->second) {
        auto actor = source.program.instructions.actor(id.value());
        auto [it, added] = ends.try_emplace(actor, std::pair{id, id});
        (void)added;
        it->second.second = id;
      }
    for (const auto &[actor, end] : ends) {
      record.participant_actors.push_back(actor);
      record.issue_refs.push_back(Ref(end.first, Milestone::kIssue));
      record.completion_refs.push_back(Ref(end.second));
    }
    if (allocation_id) {
      record.columns = by_id.at(*allocation_id)->columns;
      record.accesses = accesses[*allocation_id];
      std::sort(record.accesses.begin(), record.accesses.end(),
                [](const auto &a, const auto &b) {
                  return std::tuple{a.instruction, a.effect_index,
                                    a.milestone} <
                         std::tuple{b.instruction, b.effect_index, b.milestone};
                });
    }
    group2_sites[resources->frame.cluster_dim()]
        .push_back(std::move(record));
  }
}
void ModelLowerer::Impl::Matrix(Thread &t, const RecordView &e,
                                bool load) {
  ProfileScope profile("frontend.model.matrix");
  const auto &i = function.instructions[e.instruction_id()];
  const auto &contract = facts->launch().InstructionContract(
      i.line, i.id, t.actor.warp.value(), e.occurrence());
  auto fragments = matrix->Resolve(*current_lane, lane_position, e, *resources, contract);
  std::vector<std::string> regs;
  for (const auto &op : operands[i.id])
    if (op.starts_with('{'))
      for (const auto &r : RegistersIn(op))
        regs.push_back(Register(i, r));
  std::size_t per = Has(i, "trans") ? 2 : 1;
  if (fragments.size() != per * regs.size())
    throw SemanticError("matrix_register_map",
                        "matrix fragments disagree with registers");
  auto id = Next();
  std::vector<AsyncEffect> effects;
  for (std::size_t k = 0; k < regs.size(); ++k) {
    const auto &r = regs[k];
    if (registers.Width(r) != 4)
      throw SemanticError("matrix_register_map", "matrix needs b32 registers");
    MemoryEndpoint endpoint = RegisterAddress{t.actor.id, r};
    if (load) {
      std::vector<MilestoneRef> refs;
      for (std::size_t j = 0; j < per; ++j) {
        auto [a, n] = fragments[k * per + j];
        refs.push_back(Ref(id, Milestone::kEffect, effects.size()));
        effects.push_back({{MemoryEffect{AccessMode::kRead, a, n}}});
      }
      effects.push_back({{MemoryEffect{AccessMode::kWrite, endpoint, 4}},
                         {},
                         std::move(refs)});
      t.writes.erase(r);
    } else {
      auto index = static_cast<std::uint32_t>(effects.size());
      AsyncEffect read{{MemoryEffect{AccessMode::kRead, endpoint, 4}}};
      if (t.writes.contains(r))
        read.predecessors.push_back(t.writes.at(r));
      effects.push_back(std::move(read));
      for (std::size_t j = 0; j < per; ++j) {
        auto [a, n] = fragments[k * per + j];
        effects.push_back({{MemoryEffect{AccessMode::kWrite, a, n}},
                           {},
                           {Ref(id, Milestone::kEffect, index)}});
      }
    }
  }
  Async(t, e, std::move(effects));
  auto tail = Emit(t, e, Await{{Ref(id)}});
  aligned[Key(t.actor.warp.value(), e)].calls[t.actor.lane] = {id, tail};
}
// The terminal ret/exit record of the CTA's `lane_index`-th thread; none when its
// trace ends elsewhere.
std::optional<RecordView> ModelLowerer::Impl::ExitRecord(const CtaFacts &cta_facts,
                                                         std::size_t lane_index) const {
  const auto lane = cta_facts.Lane(lane_index);
  const auto size = cta_facts.Size(lane);
  if (size == 0) return std::nullopt;
  const auto last = cta_facts.Meta({lane, size - 1});
  const auto &control = function.instructions[last.instruction_id()].control;
  if (control != "return" && control != "exit") return std::nullopt;
  return last;
}
// The await standing for that thread's exit: emitted once, after its last instruction,
// at its exit record.
std::optional<InstId> ModelLowerer::Impl::Terminal(const CtaFacts &cta_facts, std::size_t lane_index,
                                                   const ActorInfo &actor) {
  if (const auto found = terminals.find(actor.id); found != terminals.end()) return found->second;
  const auto last = ExitRecord(cta_facts, lane_index);
  if (!last) return std::nullopt;
  Thread terminal;
  terminal.actor = actor;
  return terminals.emplace(actor.id, Emit(terminal, *last, Await{})).first->second;
}
// PTX ISA (bar, barrier; exit): without a thread count all threads of the CTA participate;
// bar.sync and bar.red wait for the non-exited threads, and barriers waiting only on exited
// threads are released. On a barrier whose every use is of that form, each use blocks until
// its phase completes, so thread u's k-th arrival belongs to phase k. With n_u arrivals per
// thread, phase k expects the threads with n_u > k; every other thread must exit before it
// completes. That obligation becomes one exit gate per thread at its first missed phase n_u
// (later phases complete after it), at the thread's terminal await, discharged by the
// well-synchronization checker. It is not an ordering edge: an exiting thread releases
// nothing, and its pending asynchronous operations stay unordered. If a thread with fewer
// arrivals does not end at ret/exit, the barrier keeps the full CTA count. Uniform counts
// change nothing. O(lanes + arrivals) per barrier; one edit per arrival whose count changes.
void ModelLowerer::Impl::CtaWideExits(const CtaFacts &cta_facts, std::size_t actor_begin) {
  ProfileScope profile("frontend.model.cta_wide_exits");
  const auto lanes = cta_facts.LaneCount();
  const auto threads = cta_facts.launch().Threads();
  for (auto &[barrier, use] : cta_wide_barriers) {
    if (!use.cta_wide || use.arrivals.empty() || lanes != threads) continue;
    std::vector<std::uint32_t> counts(lanes, 0);
    for (const auto &[actor, id] : use.arrivals) {
      (void)id;
      ++counts.at(actor.value() - actor_begin);
    }
    const auto most = *std::max_element(counts.begin(), counts.end());
    if (*std::min_element(counts.begin(), counts.end()) == most) continue;
    bool exits = true;
    for (std::size_t k = 0; k < lanes && exits; ++k)
      exits = counts[k] == most || ExitRecord(cta_facts, k);
    if (!exits) continue;
    // expected[k] = |{u : n_u > k}|, by suffix sums over the arrival-count histogram.
    std::vector<std::uint32_t> expected(most + 1, 0);
    for (const auto n : counts) ++expected[n];
    for (auto k = most; k-- > 0;) expected[k] += expected[k + 1];
    expected.erase(expected.begin());
    std::vector<std::uint32_t> seen(lanes, 0);
    for (const auto &[actor, id] : use.arrivals) {
      const auto k = seen[actor.value() - actor_begin]++;
      if (expected[k] != threads)
        std::get<NamedBarrierSync>(source.editInstruction(id).op).expected_arrivals = expected[k];
    }
    for (std::size_t k = 0; k < lanes; ++k)
      if (counts[k] != most)
        source.program.exit_gates.push_back(
            {barrier, counts[k], *Terminal(cta_facts, k, source.program.actors[actor_begin + k])});
  }
  cta_wide_barriers.clear();
}
// CUDA Programming Guide (Distributed Shared Memory): a CTA's shared memory exists until the CTA
// exits, so every access to it from another CTA of the cluster must complete before then. The CTA
// exits after its last thread does, so an event ordered before some thread's exit is ordered before
// the CTA's exit. One join of the thread exits stands for that: an event reaches the join exactly
// when it reaches some exit. A thread that does not end at ret/exit never exits; leaving it out of
// the join only strengthens the obligation. Cluster launches only. O(lanes).
void ModelLowerer::Impl::CtaExitJoin(const CtaFacts &cta_facts, std::size_t actor_begin) {
  const auto &frame = cta_facts.frame();
  const auto d = frame.cluster_dim();
  if (std::uint64_t(d[0]) * d[1] * d[2] == 1) return;
  cluster_launch = true;
  const auto coordinates = frame.SelfCoordinates();
  cluster_ctas.emplace(cta_facts.Linear(), coordinates);
  shared_owners.emplace(frame.Format(AllocationKey::Shared(frame.self())).allocation, coordinates);
  std::vector<MilestoneRef> exits;
  for (std::size_t k = 0; k < cta_facts.LaneCount(); ++k)
    if (const auto terminal = Terminal(cta_facts, k, source.program.actors[actor_begin + k]))
      exits.push_back(Ref(*terminal));
  if (exits.empty()) return;
  // Issued by the first such thread after its exit await (every instruction of a thread is in its program order);
  // its single effect waits for every exit.
  const auto first = source.instruction(exits.front().instruction);
  const auto join = Next();
  (void)source.append(first.actor, first.location,
                      AsyncOp{AsyncOpId(join.value()), {{{}, std::nullopt, std::move(exits)}}});
  cta_exits.emplace(coordinates, join);
}
// Every instruction or asynchronous effect that accesses another CTA's shared memory or acts on
// another CTA's mbarrier must happen before that CTA's exit join (CtaExitJoin). One pass over the
// instructions; each address's and barrier's owner is looked up once. One order per (instruction,
// owner): an asynchronous operation's order names its effects that reach that owner (the conjunction
// is checked only when the whole operation does not already precede the join), as the TMEM lifetime
// orders do; an MMA with a native plan keeps the aggregate milestone (RefineMmaPlans). A remote owner
// that was not lowered restarts class mode (unit closure) and is reported otherwise.
void ModelLowerer::Impl::DistributedSharedLifetimes() {
  if (!cluster_launch) return;
  ProfileScope profile("frontend.model.distributed_shared_lifetimes");
  using Effects = SharedOptional<std::vector<std::uint32_t>>;
  std::unordered_map<std::uint32_t, Coordinates> barrier_owners;
  for (const auto &[barrier, owner] : barriers.OwnedLifetimes()) barrier_owners.emplace(barrier.value(), owner);
  // Per address: -2 not looked up, -1 not a CTA's shared memory, else an index into `owners`.
  std::vector<Coordinates> owners;
  std::map<Coordinates, std::int32_t> owner_index;
  std::vector<std::int32_t> address_owners(addresses.Bindings().size(), -2);
  const auto address_owner = [&](AddressId id) -> std::int32_t {
    auto &slot = address_owners.at(id.value());
    if (slot == -2) {
      slot = -1;
      if (const auto found = shared_owners.find(addresses.Bindings()[id.value()].allocation); found != shared_owners.end()) {
        const auto [it, fresh] = owner_index.try_emplace(found->second, std::int32_t(owners.size()));
        if (fresh) owners.push_back(found->second);
        slot = it->second;
      }
    }
    return slot;
  };
  const auto barrier_owner = [&](BarrierLifetimeId id) -> std::int32_t {
    const auto found = barrier_owners.find(id.value());
    if (found == barrier_owners.end()) return -1;
    const auto [it, fresh] = owner_index.try_emplace(found->second, std::int32_t(owners.size()));
    if (fresh) owners.push_back(found->second);
    return it->second;
  };
  std::set<Coordinates> reported;
  std::size_t obligations = 0;
  const auto require = [&](InstId id, Effects effects, std::uint32_t cta, const Coordinates &owner) {
    if (const auto join = cta_exits.find(owner); join != cta_exits.end()) {
      orders.push_back({"distributed_shared_access_before_owner_exit", Ref(id), Ref(join->second),
                        "An access to another CTA's shared memory, or an operation on its mbarrier, "
                        "must complete before that CTA exits.",
                        std::move(effects)});
      ++obligations;
      return;
    }
    if (units && std::none_of(cluster_ctas.begin(), cluster_ctas.end(),
                              [&](const auto &item) { return item.second == owner; }))
      throw algorithm::UnitClassRestart{"unit_closure"};
    if (reported.insert(owner).second)
      Diagnostic("distributed_shared_owner_exit_unmodeled",
                 "A CTA whose shared memory another CTA accesses has no modeled exit.", nullptr, nullptr,
                 Json{{"owner", owner}}, units ? units->Weight(cta) : 1);
  };
  // Per owner index, the remote effects of the current asynchronous operation; or, for any other
  // instruction, whether it reaches that owner.
  std::vector<std::vector<std::uint32_t>> remote;
  for (const auto &inst : source.program.instructions) {
    const auto cta = source.program.actors[inst.actor.value()].cta.value();
    const auto issuer_it = cluster_ctas.find(cta);
    if (issuer_it == cluster_ctas.end()) continue;
    const auto issuer = issuer_it->second;
    remote.assign(owners.size(), {});
    const auto note = [&](std::int32_t owner, std::uint32_t effect) {
      if (owner < 0 || owners[owner] == issuer) return;
      if (std::size_t(owner) >= remote.size()) remote.resize(owners.size());
      auto &list = remote[owner];
      if (list.empty() || list.back() != effect) list.push_back(effect);
    };
    const auto *async = std::get_if<AsyncOp>(&inst.op);
    if (async) {
      for (std::uint32_t index = 0; index < async->effects.size(); ++index) {
        const auto &effect = async->effects[index];
        for (const auto &access : effect.accesses)
          if (const auto *address = std::get_if<AddressId>(&access.endpoint)) note(address_owner(*address), index);
        if (effect.completion)
          std::visit([&](const auto &action) { note(barrier_owner(action.barrier), index); }, *effect.completion);
      }
    } else {
      std::visit([&](const auto &op) {
        using T = std::decay_t<decltype(op)>;
        if constexpr (std::is_same_v<T, ImmediateLoad> || std::is_same_v<T, ImmediateStore>)
          note(address_owner(op.address), 0);
        if constexpr (std::is_same_v<T, AsyncLoad> || std::is_same_v<T, AsyncStore>) {
          note(address_owner(op.source), 0);
          note(address_owner(op.destination), 0);
          if (op.completion) note(barrier_owner(op.completion->barrier), 0);
        }
        if constexpr (requires { op.barrier; }) note(barrier_owner(op.barrier), 0);
      }, inst.op);
    }
    for (std::size_t owner = 0; owner < remote.size(); ++owner) {
      if (remote[owner].empty()) continue;
      Effects effects;
      if (async && !plans.contains(inst.id.value())) effects = std::move(remote[owner]);
      require(inst.id, std::move(effects), cta, owners[owner]);
    }
  }
  profile.nodes = obligations;
}
} // namespace sentryward::frontend::ptx
