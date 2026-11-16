#pragma once
#include "lowering_impl.hpp"

namespace sentryward::frontend::ptx {
void ModelLowerer::Impl::Tcgen(Thread &t, const RecordView &e,
                               const Json &contract) {
  const auto &i = function.instructions[e.instruction_id()];
  const auto &ops = operands[i.id];
  const auto &operation = i.semantics.operation;
  auto group = Has(i, "cta_group::2") ? 2 : 1;
  auto key = Key(t.actor.warp.value(), e);
  if (operation == "wait_load" || operation == "wait_store") {
    ProfileScope profile("frontend.model.wait_dependencies");
    auto &queue = operation == "wait_load" ? t.loads : t.stores;
    std::vector<MilestoneRef> refs;
    std::set<InstId> completed;
    refs.reserve(queue.Pending().size());
    for (auto id : queue.Pending()) {
      refs.push_back(Ref(id));
      completed.insert(id);
    }
    auto id = Emit(t, e, Await{std::move(refs)});
    dependency_counts[operation == "wait_load" ? "tcgen_load_wait"
                                               : "tcgen_store_wait"]
        .Record(queue.Issued(), queue.Pending().size());
    queue.Acknowledge();
    aligned[key].calls[t.actor.lane] = {id, id};
    aligned[key].merged_arm |= t.merged_arm.has_value();
    if (operation == "wait_load") {
      t.tmem_loads_since_store.clear();
      std::erase_if(t.writes, [&](const auto &item) {
        return completed.contains(item.second.instruction);
      });
    } else {
      t.tmem_last_store.reset();
      for (auto &[r, values] : t.reads) {
        (void)r;
        std::erase_if(values, [&](const auto &ref) {
          return completed.contains(ref.instruction);
        });
      }
      std::erase_if(t.reads,
                    [](const auto &item) { return item.second.empty(); });
    }
    return;
  }
  if (operation == "commit") {
    ProfileScope profile("frontend.model.completion_prefix");
    auto brackets = Brackets(ops);
    if (brackets.empty())
      throw SemanticError("tcgen_commit", "commit has no barrier");
    std::vector<BarrierLifetimeId> targets{
        resources->Barrier(e, brackets[0], contract)};
    if (std::any_of(i.modifiers.begin(), i.modifiers.end(),
                    [](const auto &m) { return m.starts_with("multicast"); })) {
      auto sh = resources->Shared(e, brackets[0]);
      if (!sh)
        throw SemanticError("tcgen_multicast_targets",
                            "requires concrete shared offset");
      auto mask = (resources->ScalarValue(e, ops.back(), "multicast mask") & Integer(65535)).convert_to<std::uint32_t>();
      if (!mask)
        throw SemanticError("tcgen_multicast_mask", "empty multicast mask");
      targets.clear();
      for (std::uint32_t rank = 0; rank < 16; ++rank)
        if (mask & (1u << rank)) {
          targets.push_back(resources->interner().Apply(
              BarrierStep{{BarrierKey::Kind::Physical, resources->frame.Rank(rank), std::to_string(sh->second)}}));
        }
    }
    auto id = Next();
    const auto &work = t.history.work[group];
    auto &prefix = t.tcgen_notifications[group];
    auto refs = prefix.Prepare(work);
    const auto emitted = refs.size() + targets.size();
    std::vector<AsyncEffect> effects;
    effects.push_back({{}, std::nullopt, std::move(refs)});
    for (auto target : targets)
      effects.push_back(
          {{}, ArriveEffect{target, 1}, {Ref(id, Milestone::kEffect, 0)}});
    if (!t.history.commits[group].empty())
      limitations["tcgen_commit_notification_order_refinement"] =
          "Successive commit notification ordering needs progress refinement.";
    t.history.commits[group].push_back(id.value());
    t.history.records.push_back({id.value(), group, "commit"});
    Async(t, e, std::move(effects));
    prefix.Acknowledge(id, work.size());
    dependency_counts["tcgen_commit"].Record(work.size() * targets.size(),
                                             emitted, true);
    return;
  }
  if (operation == "load" || operation == "store") {
    ProfileScope profile("frontend.model.tmem_ldst");
    auto brackets = Brackets(ops);
    if (brackets.empty())
      throw SemanticError("tcgen_ldst_operand", "missing TMEM address");
    auto address = ResolveTmem(t, e, brackets[0]);
    std::optional<std::uint64_t> half_split;
    if (Has(i, "16x32bx2")) {
      if (ops.size() != 3)
        throw SemanticError("tcgen_ldst_operand", "split TMEM transfer requires a half-split offset");
      const auto value = resources->ScalarValue(
          e, ops[operation == "load" ? 2 : 1], "TMEM half-split offset");
      if (value < 0 || value > UINT32_MAX)
        throw SemanticError("tcgen_ldst_operand", "TMEM half-split offset must fit u32");
      half_split = value.convert_to<std::uint64_t>();
    }
    auto intervals =
        DecodeLdstLaneRegisters(i, address.address, t.actor.lane,
                                t.actor.warp.value() % 4, address.columns, half_split);
    std::vector<std::string> regs;
    for (const auto &op : ops)
      if (op.starts_with('{'))
        for (const auto &r : RegistersIn(op))
          regs.push_back(Register(i, r));
    for (const auto &r : regs)
      if (!r.starts_with('%') || registers.Width(r) != 4)
        throw SemanticError(
            "tcgen_register_fragment_map",
            "TMEM transfer needs declared 32-bit scalar registers");
    if (Has(i, "red"))
      throw SemanticError("tcgen_register_fragment_map",
                          "packed/reduced register mapping unsupported");
    const unsigned count = Has(i, "pack::16b") || Has(i, "unpack::16b") ? 2 : 1;
    const unsigned size = count == 2 ? 2 : 4;
    if (intervals.size() != count * regs.size())
      throw SemanticError("tcgen_register_fragment_map",
                          "one four-byte TMEM fragment required per register");
    auto id = Next();
    // Pipeline predecessors of every TMEM access of this ld/st (one milestone
    // per earlier pipelined instruction, not one edge per earlier fragment).
    std::vector<MilestoneRef> pipeline;
    if (t.tmem_last_store) pipeline.push_back(Ref(*t.tmem_last_store));
    if (operation == "store")
      for (auto load : t.tmem_loads_since_store)
        pipeline.push_back(Ref(load, Milestone::kReadsComplete));
    std::vector<AsyncEffect> effects;
    effects.reserve(2 * regs.size());
    for (std::size_t k = 0; k < regs.size(); ++k) {
      MemoryEndpoint reg = RegisterAddress{t.actor.id, regs[k]};
      auto index = static_cast<std::uint32_t>(effects.size());
      if (operation == "store") {
        AsyncEffect read{{MemoryEffect{AccessMode::kRead, reg, 4}}};
        if (t.writes.contains(regs[k])) read.predecessors.push_back(t.writes.at(regs[k]));
        effects.push_back(std::move(read));
      }
      for (unsigned part = 0; part < count; ++part) {
        const auto &interval = intervals[count * k + part];
        if (interval.size != size)
          throw SemanticError("tcgen_register_fragment_map", "invalid TMEM register fragment size");
        auto memory = resources->Bind("tensor", address.resource, interval.offset, size);
        if (operation == "load") {
          effects.push_back({{MemoryEffect{AccessMode::kRead, memory.first, size}}, {}, pipeline});
        } else {
          std::vector<MilestoneRef> predecessors{Ref(id, Milestone::kEffect, index)};
          predecessors.insert(predecessors.end(), pipeline.begin(), pipeline.end());
          effects.push_back({{MemoryEffect{AccessMode::kWrite, memory.first, size}}, {},
                            std::move(predecessors)});
        }
      }
      if (operation == "load") {
        // A true register dependency yields the loaded value but does not order
        // the TMEM reads before the dependent instruction's memory accesses
        // (PTX ISA, tcgen05 register dependencies): the register write has no
        // TMEM-read predecessor, so the reads complete only at tcgen05.wait::ld.
        effects.push_back({{MemoryEffect{AccessMode::kWrite, reg, 4}}});
        t.writes.insert_or_assign(regs[k],
                                  Ref(id, Milestone::kEffect, index + count));
      } else
        t.reads[regs[k]].push_back(Ref(id, Milestone::kEffect, index));
    }
    Async(t, e, std::move(effects));
    (operation == "load" ? t.loads : t.stores).Issue(id);
    if (operation == "load")
      t.tmem_loads_since_store.push_back(id);
    else {
      t.tmem_last_store = id;
      t.tmem_loads_since_store.clear();
    }
    auto tail = Emit(t, e, Await{});
    aligned[key].calls[t.actor.lane] = {id, tail};
    aligned[key].merged_arm |= t.merged_arm.has_value();
    aligned[key].tmem_addresses[t.actor.lane] = {address.resource, address.address};
    return;
  }
  if (operation == "mma") {
    Mma(t, e, contract);
    return;
  }
  if (operation == "cp") {
    limitations["tcgen_cp_memory_effect_refinement"] =
        "Copy memory effects require refinement of tcgen pipeline ordering.";
    if (std::any_of(t.mma_operand_reads.begin(), t.mma_operand_reads.end(),
                    [&](const auto &item) { return std::get<0>(item.first) == static_cast<unsigned>(group); }))
      limitations["tcgen_pipeline_effect_refinement"] =
          "Copy effects following MMA operand reads require pipeline ordering refinement.";
    const auto shape = DecodeCopyShape(i.modifiers);
    if (ops.size() != 2 || !shape || Has(i, "b8x16") || Has(i, "b6x16_p32") || Has(i, "b4x16_p64"))
      throw SemanticError("tcgen_cp_variant", "copy format needs a region mapping");
    const auto target = ResolveTmem(t, e, ops[0]);
    const auto descriptor = Descriptor(e, ops[1], "copy shared descriptor");
    const auto &frame = resources->frame;
    std::vector<CtaRank> owners{frame.self()};
    if (group == 2) {
      if (target.columns != 512)
        throw SemanticError("tmem_group2_relative_peer_base", "copy peers require matched placement");
      const auto rank = frame.self().value;
      owners = {frame.Rank(rank & ~1U), frame.Rank(rank | 1U)};
    }
    std::optional<std::vector<std::pair<ByteInterval, ByteInterval>>> fragments;
    try { fragments = DecodeCopyFragments(descriptor, target.address, *shape, target.columns); }
    catch (const LayoutError &) {}
    std::vector<std::pair<Fragment, Fragment>> pairs;
    for (auto owner : owners) {
      if (fragments) {
        for (const auto &[read, write] : *fragments)
          pairs.push_back({resources->Bind("shared", "physical.shared.cta.0", read.offset, read.size, owner),
                           resources->Bind("tensor", target.resource, write.offset, write.size, owner)});
      } else {
        pairs.push_back({resources->Abstract("shared", "physical.shared.cta.0",
             {{"operator", "tcgen_cp_shared_region"}, {"operands", Json::array({descriptor, shape->name})}},
             shape->SourceBytes(), owner),
            resources->Abstract("tensor", target.resource,
             {{"operator", "tcgen_cp_tensor_region"}, {"operands", Json::array({target.address, shape->name})}},
             shape->TensorBytes(), owner)});
      }
    }
    const auto id = Next();
    std::vector<AsyncEffect> effects;
    for (const auto &[read, write] : pairs) {
      const auto index = static_cast<std::uint32_t>(effects.size());
      effects.push_back({{MemoryEffect{AccessMode::kRead, read.first, read.second}}});
      effects.push_back({{MemoryEffect{AccessMode::kWrite, write.first, write.second}},
                         std::nullopt, {Ref(id, Milestone::kEffect, index)}});
    }
    TcgenCopyPlan plan{effects, static_cast<unsigned>(group), target.resource, target.address,
                       shape->name == "4x256b", {}};
    for (std::uint32_t index = 1; index < effects.size(); index += 2) {
      const auto &write = effects[index].accesses.front();
      const auto &binding = resources->Binding(std::get<AddressId>(write.endpoint));
      const auto key = std::tuple{static_cast<unsigned>(group), binding.address_space,
                                 binding.alias_scope.empty() ? binding.allocation : binding.alias_scope};
      if (const auto previous = t.mma_operand_reads.find(key);
          previous != t.mma_operand_reads.end() && write.bytes && binding.expression.empty())
        for (const auto candidate : previous->second.Candidates(binding.allocation, binding.offset, *write.bytes, false)) {
          const auto &[ref, old, size] = previous->second.all[candidate];
          if (old.expression.empty() &&
              old.offset < binding.offset + *write.bytes && binding.offset < old.offset + *size) {
            UniquePush(effects[index].predecessors, ref);
            plan.operand_links.emplace_back(index, ref);
          }
        }
      if (write.bytes && *write.bytes && binding.expression.empty())
        t.copy_writes[{static_cast<unsigned>(group), binding.address_space, binding.allocation}].push_back(
            {Ref(id, Milestone::kEffect, index), binding, *write.bytes, target.address, plan.any_operand_address});
    }
    plan.effects = effects;
    copy_plans[id.value()] = std::move(plan);
    Async(t, e, std::move(effects));
    t.history.work[group].push_back(id.value());
    t.history.records.push_back({id.value(), group, "cp"});
    return;
  }
  if (operation == "shift")
    throw SemanticError("tcgen_matrix_footprint",
                        "requires certified operand and CTA footprints");
  throw SemanticError("unsupported_tcgen_operation",
                      "unknown TCGEN source semantics");
}
} // namespace sentryward::frontend::ptx
