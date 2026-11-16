#pragma once
#include "lowering_impl.hpp"

namespace sentryward::frontend::ptx {
// Lift only a memory instruction that intersects pending register accesses.
// Dependency edges target its individual operand reads, never the producer's
// aggregate completion or the entire outstanding TMEM load/store queue.
bool ModelLowerer::Impl::MemoryRegisters(Thread &t, const RecordView &e,
    AddressId address, std::uint32_t bytes, bool load) {
  if (t.writes.empty() && t.reads.empty()) return false;
  const auto &i=function.instructions[e.instruction_id()];
  const auto inputs=Sources(i), outputs=Destinations(i);
  bool pending=false;
  for (const auto *registers : {&inputs,&outputs})
    for (const auto &r:*registers)
      pending |= t.writes.contains(r) || t.reads.contains(r);
  if (!pending) return false;
  if (facts && facts->recording()) [[unlikely]]
    if (const auto at = e.ref()) {
      const auto pending_write = [&](const auto &r) { return t.writes.contains(r); };
      const bool overwrites = std::any_of(outputs.begin(), outputs.end(), pending_write);
      const bool reads = std::any_of(inputs.begin(), inputs.end(), pending_write);
      if (overwrites || reads) facts->Note(query::RegisterTouch{*at, overwrites, reads});
    }
  const auto id=Next();
  std::vector<AsyncEffect> effects;
  std::vector<MilestoneRef> operands;
  for (const auto &r:inputs) {
    AsyncEffect read{{MemoryEffect{AccessMode::kRead,RegisterAddress{t.actor.id,r},registers.Width(r)}}};
    if (t.writes.contains(r)) read.predecessors.push_back(t.writes.at(r));
    operands.push_back(Ref(id,Milestone::kEffect,effects.size()));
    effects.push_back(std::move(read));
  }
  const auto memory_index=static_cast<std::uint32_t>(effects.size());
  effects.push_back({{MemoryEffect{load?AccessMode::kRead:AccessMode::kWrite,address,bytes}}, {}, std::move(operands)});
  for (const auto &r:outputs)
    effects.push_back({{MemoryEffect{AccessMode::kWrite,RegisterAddress{t.actor.id,r},registers.Width(r)}},
        {},{Ref(id,Milestone::kEffect,memory_index)}});
  Async(t,e,std::move(effects));
  // Ordinary memory instructions complete their own effects. This does not
  // acknowledge any tcgen05 operation or wait for unused producer fragments.
  Emit(t,e,Await{{Ref(id)}});
  if(load && !shared_value_instructions.empty() && shared_value_instructions.at(e.instruction_id())) {
    const SharedValueSite site{current_coordinates,t.actor.warp.value()*32+t.actor.lane,e.Key()};
    if(shared_value_sites.contains(site))shared_value_reads.emplace(site,Ref(id,Milestone::kEffect,memory_index));
  }
  for(const auto &r:outputs)t.writes.erase(r);
  return true;
}
void ModelLowerer::Impl::Compute(Thread &t, const RecordView &e) {
  if (t.writes.empty() && t.reads.empty())
    return;
  ProfileScope profile("frontend.model.async_register_compute");
  const auto &i = function.instructions[e.instruction_id()];
  auto destinations = Destinations(i), sources = Sources(i);
  if (!e.opaque_compute_diamond_indices().empty()) {
    destinations = e.opaque_compute_register_writes();
    sources = e.opaque_compute_register_reads();
    if (i.predicate)
      UniquePush(sources, Register(i, *i.predicate));
  }
  bool pending = false;
  for (const auto *list : {&destinations, &sources})
    for (const auto &r : *list)
      pending |= t.writes.contains(r) || t.reads.contains(r);
  if (!pending)
    return;
  if (facts && facts->recording()) [[unlikely]]
    if (const auto at = e.ref()) {
      const auto pending_write = [&](const auto &r) { return t.writes.contains(r); };
      const bool overwrites = std::any_of(destinations.begin(), destinations.end(), pending_write);
      const bool reads = std::any_of(sources.begin(), sources.end(), pending_write);
      if (overwrites || reads) facts->Note(query::RegisterTouch{*at, overwrites, reads});
    }
  std::vector<MilestoneRef> predecessors;
  std::vector<MemoryEffect> accesses;
  for (const auto &r : sources) {
    if (t.writes.contains(r))
      UniquePush(predecessors, t.writes.at(r));
    accesses.push_back({AccessMode::kRead, RegisterAddress{t.actor.id, r},
                        registers.Width(r)});
  }
  for (const auto &r : destinations)
    accesses.push_back({AccessMode::kWrite, RegisterAddress{t.actor.id, r},
                        registers.Width(r)});
  auto id = Async(
      t, e, {AsyncEffect{std::move(accesses), {}, std::move(predecessors)}});
  Emit(t, e, Await{{Ref(id)}});
  for (const auto &r : destinations)
    t.writes.erase(r);
}
void ModelLowerer::Impl::WarpData(Thread &t, const RecordView &e) {
  ProfileScope profile("frontend.model.warp_data");
  const auto &i = function.instructions[e.instruction_id()];
  auto sources = Sources(i), destinations = Destinations(i);
  if (!sources.empty()) {
    std::vector<AsyncEffect> effects;
    for (const auto &r : sources) {
      AsyncEffect effect{
          {MemoryEffect{AccessMode::kRead, RegisterAddress{t.actor.id, r},
                        registers.Width(r)}}};
      if (t.writes.contains(r))
        effect.predecessors.push_back(t.writes.at(r));
      effects.push_back(std::move(effect));
    }
    auto id = Async(t, e, std::move(effects));
    Emit(t, e, Await{{Ref(id)}});
  }
  Collective(t, e, "warp_data");
  if (!destinations.empty()) {
    std::vector<AsyncEffect> effects;
    for (const auto &r : destinations) {
      effects.push_back(
          {{MemoryEffect{AccessMode::kWrite, RegisterAddress{t.actor.id, r},
                         registers.Width(r)}}});
      t.writes.erase(r);
    }
    auto id = Async(t, e, std::move(effects));
    Emit(t, e, Await{{Ref(id)}});
  }
}
void ModelLowerer::Impl::Collective(Thread &t, const RecordView &e,
                                    std::string_view operation) {
  ProfileScope profile("frontend.model.collective");
  const auto &ops = operands[e.instruction_id()];
  auto warp = t.actor.warp.value(), lane = t.actor.lane;
  if (operation == "warp_sync" || operation == "warp_data") {
    auto value = resources->ScalarValue(e, ops.back(), "warp mask");
    if (value < 0 && value >= -(Integer(1) << 31))
      value &= 0xffffffff;
    auto mask = Unsigned(value, "warp mask", UINT32_MAX);
    if (!mask || !(mask & (1u << lane)))
      throw SemanticError("invalid_warp_sync_mask",
                          "issuer not in nonempty u32 mask");
    auto key = std::tuple{warp, e.instruction_id(), e.occurrence(),
                          static_cast<std::uint32_t>(mask)};
    auto &specs = this->lane ? this->lane->collectives : source.program.collectives;
    auto [it, added] = warp_collectives.try_emplace(key, SyncId(specs.size()));
    if (added) {
      std::vector<ActorId> participants;
      for (std::uint32_t l = 0; l < 32; ++l)
        if (mask & (1u << l)) {
          auto found = actors.find({warp, l});
          if (found == actors.end())
            throw SemanticError("missing_collective_thread",
                                "mask names missing lane");
          participants.push_back(found->second);
        }
      specs.push_back({it->second, std::move(participants), static_cast<std::uint32_t>(mask)});
    }
    Emit(t, e, SyncWarp{it->second, static_cast<std::uint32_t>(mask), operation == "warp_sync"});
    return;
  }
  if (operation == "cta_sync" || operation == "named_arrive" ||
      operation == "named_reduce") {
    bool reduced = operation == "named_reduce";
    auto slot = Scalar(e, ops.at(reduced ? 1 : 0), "named slot", 15);
    const bool counted = ops.size() == (reduced ? 4u : 2u);
    auto expected =
        counted
            ? Scalar(e, ops.at(reduced ? 2 : 1), "named count", UINT32_MAX)
            : facts->launch().Threads();
    if (!expected || expected % 32 || expected > facts->launch().Threads())
      throw SemanticError("invalid_named_barrier",
                          "invalid named barrier thread count");
    const auto barrier = resources->interner().Apply(
        BarrierStep{{BarrierKey::Kind::Named, resources->frame.self(), std::to_string(slot)}});
    auto &use = cta_wide_barriers[barrier];
    InstId id;
    if (operation == "named_arrive")
      id = Emit(t, e, NamedBarrierArrive{barrier, static_cast<std::uint32_t>(expected)});
    else
      id = Emit(t, e, NamedBarrierSync{barrier, static_cast<std::uint32_t>(expected)});
    // bar.arrive always has a count (PTX ISA), so only blocking uses can be CTA-wide.
    if (counted || operation == "named_arrive") {
      use.cta_wide = false;
      use.arrivals.clear();
    } else if (use.cta_wide) {
      use.arrivals.emplace_back(t.actor.id, id);
    }
    return;
  }
  if (operation == "cluster_arrive" || operation == "cluster_wait") {
    const auto d = resources->frame.cluster_dim();
    const auto origin = resources->frame.ClusterOrigin();
    ClusterKey key{origin, d, cluster_phases[t.actor.id]};
    const auto phase=cluster_phases[t.actor.id];
    auto &arrivals = cluster_arrivals[key];
    if (operation == "cluster_arrive") {
      if (arrivals.contains(t.actor.id)) {
        violations.emplace("cluster_arrive_without_wait", "thread arrives twice at the cluster barrier before an intervening wait");
        throw SemanticError("cluster_arrive_without_wait",
                            "thread arrived twice before wait");
      }
      arrivals.emplace(t.actor.id, Emit(t, e, Await{}));
    } else {
      if (!arrivals.contains(t.actor.id))
        throw SemanticError("cluster_wait_without_arrive",
                            "cluster wait without arrival");
      auto id = Emit(t, e, Await{});
      cluster_waits.push_back(
          {key, id,
           std::uint64_t(d[0]) * d[1] * d[2] * facts->launch().Threads()});
      ++cluster_phases[t.actor.id];
    }
    auto id = sites.at(Key(warp, e)).back();
    if (RegistersAlignedLanes(function.instructions[e.instruction_id()])) aligned[Key(warp, e)].calls[lane] = {id, id};
    cluster_warp_rendezvous[{warp,std::string(operation),phase}][t.actor.id]=id;
    return;
  }
  throw SemanticError("unsupported_collective", "unknown collective");
}
void ModelLowerer::Impl::Mbarrier(Thread &t, const RecordView &e,
                                  const Json &contract) {
  ProfileScope profile("frontend.model.mbarrier");
  const auto &i = function.instructions[e.instruction_id()];
  const auto &ops = operands[i.id];
  auto brackets = Brackets(ops);
  if (brackets.size() != 1)
    throw SemanticError("mbarrier_operand", "needs one barrier address");
  auto barrier = resources->Barrier(e, brackets[0], contract);
  const auto &operation = i.semantics.operation;
  if (operation == "init")
    Emit(t, e,
         MBarrierInit{barrier,
                      static_cast<std::uint32_t>(Scalar(
                          e, ops.back(), "expected arrivals", UINT32_MAX))});
  else if (operation == "inval") {
    const auto address=resources->Memory(e,MemoryRequest(brackets[0],Json::object(),8,"shared"));
    barrier_invalidations[barrier].push_back(Emit(t,e,ImmediateStore{address.first,8}));
  }
  else if (operation == "expect_tx")
    Emit(t, e,
         MBarrierExpectTx{barrier, Scalar(e, ops.back(), "transaction bytes")});
  else if (operation == "complete_tx")
    Emit(t, e,
         MBarrierCompleteTx{barrier,
                            Scalar(e, ops.back(), "transaction bytes")});
  else if (operation == "arrive" || operation == "arrive_expect_tx") {
    auto count =
        operation == "arrive" && ops.size() >= 3
            ? Scalar(e, ops.back(), "arrival count", UINT32_MAX)
            : Unsigned(JsonInteger(contract.value("arrival_count", Json(1))),
                       "arrival count", UINT32_MAX);
    if (!count)
      throw SemanticError("invalid_arrival_count",
                          "arrival count must be positive");
    std::optional<StateId> result;
    if (ops[0] != "_")
      result = StateId(Next().value());
    if (operation == "arrive")
      Emit(t, e,
           MBarrierArrive{barrier, static_cast<std::uint32_t>(count), result});
    else
      Emit(t, e,
           MBarrierArriveExpectTx{barrier, static_cast<std::uint32_t>(count),
                                  Scalar(e, ops.back(), "expected bytes"),
                                  result});
  } else if (operation.starts_with("try_wait_") ||
             operation.starts_with("test_wait_")) {
    if (!e.collapsed_wait_loop() &&
        contract.value("wait_usage", "") != "logical_wait")
      throw SemanticError("one_shot_wait_probe",
                          "one shot wait lacks blocking wait proof");
    if (operation.ends_with("parity"))
      Emit(t, e,
           MBarrierWaitParity{barrier, static_cast<std::uint8_t>(Scalar(
                                           e, ops.at(2), "wait parity", 1))});
    else {
      auto origin = facts->Origin(e, ops.at(2));
      if (!origin || !t.origins.contains(*origin))
        throw SemanticError("unresolved_state_token",
                            "state wait has no retained arrival token");
      const auto op = Emitted(t.origins.at(*origin)).op;
      std::optional<StateId> state;
      if (auto p = std::get_if<MBarrierArrive>(&op))
        state = p->result;
      else if (auto p = std::get_if<MBarrierArriveExpectTx>(&op))
        state = p->result;
      if (!state)
        throw SemanticError("invalid_state_token",
                            "wait origin not state-producing arrival");
      Emit(t, e, MBarrierWaitState{barrier, *state});
    }
  } else
    throw SemanticError("mbarrier_operation",
                        "unrepresented mbarrier operation");
}
} // namespace sentryward::frontend::ptx
