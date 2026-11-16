#pragma once
#include "lowering_impl.hpp"

namespace sentryward::frontend::ptx {
void ModelLowerer::Impl::Instruction(Thread &t, const RecordView &e) {
  const auto &i = function.instructions[e.instruction_id()];
  const auto &s = i.semantics;
  const auto &ops = operands[i.id];
  const auto &operation = s.operation;
  const auto &contract = facts->launch().InstructionContract(
      i.line, i.id, t.actor.warp.value(), e.occurrence());
  if (t.merged_arm && (e.instruction_id() <= t.merged_arm->first || e.instruction_id() >= t.merged_arm->second))
    t.merged_arm.reset();
  // Merged one-armed diamond (execution.cpp): no tcgen05 ordering may be pending at the
  // branch, so no earlier operation may reach the arm through a tcgen05 pipeline, a wait in
  // the arm or a pending register write.
  if (const auto join = e.merged_arm_join()) {
    auto [entry, added] = merged_arm_facts.try_emplace(e.instruction_id());
    if (added) {
      std::set<std::string> written;
      for (auto index = e.instruction_id() + 1; index < *join && index < function.instructions.size(); ++index) {
        const auto &arm = function.instructions[index];
        entry->second.wait_load |= arm.opcode == "tcgen05" && arm.semantics.operation == "wait_load";
        for (const auto &r : Sources(arm))
          if (!written.contains(r)) UniquePush(entry->second.exposed, r);
        for (const auto &r : Destinations(arm)) written.insert(r);
      }
    }
    const auto &arm = entry->second;
    if (t.tmem_last_store || !t.tmem_loads_since_store.empty() ||
        (arm.wait_load && !t.loads.Pending().empty()) ||
        std::any_of(arm.exposed.begin(), arm.exposed.end(), [&](const auto &r) { return t.writes.contains(r); }))
      throw SemanticError("two_arm_merge_pending_tcgen05",
                          "tcgen05 operations are pending at a data-dependent branch whose conditionally skipped "
                          "block is analyzed on every thread; their order through the branch is not modeled");
    t.merged_arm = {e.instruction_id(), *join};
  }
  if (i.opcode=="tensormap" && s.modeled) {
    if (operation=="tensor_map_replace") {
      const auto target=resources->Memory(e,MemoryRequest(ops.at(0),Json::object(),128,"shared"));
      Emit(t,e,ImmediateLoad{target.first,128});
      Emit(t,e,ImmediateStore{target.first,128});
    } else {
      if (Scalar(e,ops.at(2),"tensor map copy size")!=128)
        throw SemanticError("tensor_map_size","tensor map copy requires 128 bytes");
      auto fragment=[&](std::string_view operand,std::string_view space) -> Fragment {
        const auto access=resources->Memory(e,MemoryRequest(operand,Json::object(),128,std::string(space)));
        const auto binding=resources->Binding(access.first);
        if (!binding.expression.empty()) throw SemanticError("tensor_map_fragment","tensor map copy requires constant region");
        if (space=="global") tensormap_destinations.emplace(binding.allocation,binding.offset);
        // The stored alias is not carried.
        return {resources->interner().Apply(ConstantStep{AllocationKey::Opaque(binding.address_space,binding.allocation,{}),
                                                         binding.offset+t.actor.lane*4}),4};
      };
      const auto head=Copy(t,e,{fragment(ops.at(1),"shared")},{fragment(ops.at(0),"global")});
      // A copy a runtime tensor map read reads from: its write effect (after the one read).
      if (const SharedValueSite site{current_coordinates,t.actor.warp.value()*32,e.Key()}; tensor_map_copies.contains(site))
        shared_value_writes[site].push_back(Ref(head,Milestone::kEffect,1));
      const auto tail=Emit(t,e,Await{{Ref(head)}});
      auto &entry=aligned[Key(t.actor.warp.value(),e)]; entry.calls[t.actor.lane]={head,tail}; entry.completions=true;
    }
    return;
  }
  if (s.category == "compute" || s.category == "control") {
    if (IsWarpDataCollective(i)) {
      if (i.predicate && !e.predicate_value())
        throw SemanticError("unresolved_instruction_predicate",
                            "warp data exchange has unknown predicate");
      WarpData(t, e);
    } else
      Compute(t, e);
    // A skipped region's st.local stores (LocalRegionStore, one array per region): the region
    // reads and writes the whole array once; the value IR lifts each element slot as one read and
    // one write of it.
    const LocalRegionStore *array = nullptr;
    for (const auto index : e.opaque_compute_diamond_indices()) {
      if (!LocalStoreInstruction(function.instructions.at(index))) continue;
      if (!local_region_stores) local_region_stores = std::make_unique<LocalRegionStores>(module, function);
      const auto *store = local_region_stores->Shape(index);
      if (!store || (array && (array->declaration.resource != store->declaration.resource ||
                               array->element_bytes != store->element_bytes)))
        throw SemanticError("lifted_region_local_store", "A skipped region's st.local has no element-slot shape of "
                            "the region's one .local array");
      array = store;
    }
    if (array) {
      const auto address = resources->Bind("local", array->declaration.resource, 0, array->declaration.bytes).first;
      Emit(t, e, ImmediateLoad{address, static_cast<std::uint32_t>(array->declaration.bytes)});
      Emit(t, e, ImmediateStore{address, static_cast<std::uint32_t>(array->declaration.bytes)});
    }
    return;
  }
  if (e.canonical_wait_probe())
    return;
  if (i.predicate && !e.predicate_value())
    throw SemanticError("unresolved_instruction_predicate",
                        "reached instruction predicate unresolved");
  const bool ordinary_memory = OrdinaryMemory(i);
  if (!ordinary_memory && i.opcode != "tcgen05" && i.opcode != "ldmatrix" &&
      i.opcode != "stmatrix" && (!t.writes.empty() || !t.reads.empty())) {
    for (std::size_t k = 0; k < ops.size(); ++k) {
      if (operation == "st_async_complete_tx" && ops.size() == 3 && k == 1)
        continue;
      for (const auto &reg : RegistersIn(ops[k])) {
        auto name = Register(i, reg);
        if (t.writes.contains(name))
          throw SemanticError("async_register_dependency_unsupported",
                              "Register dependency for this instruction is not modeled",
                              {{"register",name},{"consumer_opcode",Opcode(i)},
                               {"producer_line",Emitted(t.writes.at(name).instruction).location.line},
                               {"dependency","pending_register_write"}});
      }
    }
    for (const auto &reg : Destinations(i))
      if (t.reads.contains(reg))
        throw SemanticError("async_register_overwrite_unsupported",
                            "Register overwrite during an asynchronous read is not modeled for this instruction",
                            {{"register",reg},{"consumer_opcode",Opcode(i)},
                             {"reader_line",Emitted(t.reads.at(reg).front().instruction).location.line},
                             {"dependency","pending_register_read"}});
  }
  // atom/red that order_independent_atomics models: one synchronous read-modify-write
  // of a 2-, 4- or 8-byte element; a packed half pair (.f16x2, .bf16x2) is one 4-byte
  // element. Vector, 128-bit and .async forms keep the rejections below.
  const auto atomic_width = Has(i, "f16x2") || Has(i, "bf16x2") ? std::optional<std::uint32_t>(4)
                                                                 : i.memory_width;
  const bool atomic_rmw =
      s.category == "memory" && s.memory_mode == "atomic" && (i.opcode == "atom" || i.opcode == "red") &&
      !i.address_lanes && !Has(i, "async") && atomic_width &&
      (*atomic_width == 2 || *atomic_width == 4 || *atomic_width == 8) &&
      std::none_of(i.modifiers.begin(), i.modifiers.end(), [](const std::string &m) {
        return m.starts_with("v") && m.size() > 1 && std::isdigit(static_cast<unsigned char>(m[1]));
      });
  if (atomic_rmw && !atomics)
    throw SemanticError("assumption_required_order_independent_atomics",
        "Atomic read-modify-writes are unsupported. Enable order_independent_atomics "
        "to analyze them as indivisible accesses.");
  if (s.atomic_add && !atomic_rmw && (s.category == "memory" || !s.modeled))
    throw SemanticError("atomic_add_memory_semantics_unsupported",
                        "Atomic add memory semantics are unsupported");
  if (!s.modeled)
    throw SemanticError("unsupported_ptx_instruction",
                        "no source semantics for " + Opcode(i));
  if (s.category == "memory") {
    ProfileScope profile("frontend.model.immediate_memory");
    auto space = s.memory_space.value_or("generic");
    if (space == "param" || space == "const")
      return;
    if (atomic_rmw) {
      // An atomic write of the element, complete before the thread's next instruction.
      // Overlapping atomics in a scope that includes both threads do not race; an atomic
      // and an ordinary access do. Memory-order qualifiers add no ordering here, which
      // can only report more races.
      auto brackets = Brackets(ops);
      if (brackets.empty())
        throw SemanticError("missing_memory_operand", "missing address/width");
      auto [a, n] = resources->Memory(e, MemoryRequest(brackets[0], contract, *atomic_width, space));
      if (n != *atomic_width)
        throw SemanticError("immediate_access_size", "footprint extent disagrees with opcode");
      // .cluster is narrowed to .cta: conservative, since a cluster contains the CTA.
      const auto scope = Has(i, "sys") ? AtomicScope::kSys
                         : Has(i, "cta") || Has(i, "cluster") ? AtomicScope::kCta
                         : AtomicScope::kGpu;
      auto id = Async(t, e, {{{MemoryEffect{AccessMode::kWrite, a, *n, scope}}}});
      Emit(t, e, Await{{Ref(id, Milestone::kComplete)}});
      return;
    }
    if (s.memory_mode != "read" && s.memory_mode != "write")
      throw SemanticError("atomic_memory_semantics",
                          "atomic/RMW not ordinary access");
    if (i.address_lanes) {
      Matrix(t, e, *s.memory_mode == "read");
      return;
    }
    auto brackets = Brackets(ops);
    if (brackets.empty() || !i.memory_width)
      throw SemanticError("missing_memory_operand", "missing address/width");
    auto [a, n] =
        resources->Memory(e, MemoryRequest(brackets[0], contract, *i.memory_width, space));
    if (n != *i.memory_width)
      throw SemanticError("immediate_access_size",
                          "footprint extent disagrees with opcode");
    if (MemoryRegisters(t,e,a,static_cast<std::uint32_t>(*n),s.memory_mode=="read")) return;
    if (s.memory_mode == "read")
      Emit(t, e, ImmediateLoad{a, static_cast<std::uint32_t>(*n)});
    else
      Emit(t, e, ImmediateStore{a, static_cast<std::uint32_t>(*n)});
    return;
  }
  if (s.category == "mbarrier") {
    Mbarrier(t, e, contract);
    return;
  }
  if (s.category == "collective") {
    Collective(t, e, operation);
    return;
  }
  if (s.category == "fence") {
    Emit(t, e, Await{});
    return;
  }
  if (s.category == "tmem_lifetime") {
    Lifetime(t, e);
    return;
  }
  if (s.category == "external_control") {
    resources->AdmitGrid(i);
    return;
  }
  if (i.opcode == "tcgen05") {
    Tcgen(t, e, contract);
    return;
  }
  if (operation == "copy") {
    if (ops.size() != 3 && ops.size() != 4)
      throw SemanticError("cp_async_operands", "cp.async operand count");
    auto size = Scalar(e, ops[2], "cp.async size");
    if (size != 4 && size != 8 && size != 16)
      throw SemanticError("cp_async_size", "copy size must be 4,8,16");
    // The copy reads `least` to `source_size` source bytes and zero-fills the rest of
    // `size` (ISA: src-size > cp-size is undefined). A data-dependent operand reads the
    // counts its known bits allow: an ignore-src predicate 0 or `size` bytes.
    const std::string_view control = "source size/ignore source";
    auto source_size = size, least = size;
    if (ops.size() == 4) {
      auto reg = Register(i, ops[3]);
      bool immediate = ParseInteger(ops[3]).has_value();
      const bool predicate = !immediate && registers.Width(reg) == 1;
      const auto full = predicate ? 1ULL : 0xffffffffULL;
      const auto [fixed, known] = immediate ? Resources::KnownBits{0, full}
                                            : resources->FixedBits(e, ops[3], predicate ? 1 : 32);
      const auto value = known == full ? std::optional(Scalar(e, ops[3], control)) : std::nullopt;
      if (!immediate &&
          (!ops[3].starts_with('%') || ambiguous_registers.contains(ops[3])))
        throw SemanticError("cp_async_operand_scope",
                            "size register lacks unambiguous scope");
      if (predicate) {
        if (value > 1)
          throw SemanticError("cp_async_ignore_source",
                              "ignore source must be predicate");
        source_size = value == 1 ? 0 : size;
        least = value ? source_size : 0;
      } else {
        if (!immediate && registers.Width(reg) != 4)
          throw SemanticError("cp_async_operand_type",
                              "source size needs 32-bit register");
        source_size = value.value_or(fixed | (~known & full));
        least = value.value_or(fixed);
        if (!value && source_size > size)
          throw SemanticError("async_copy_source_control_not_static",
                              "Data-dependent source size is not provably at most the copy size",
                              {{"operand_role", control}, {"operand", ops[3]}});
      }
    }
    // A known source size above the copy size is reachable undefined behavior
    // (LoweringUndefinedBehavior); an unknown one is refused above.
    if (source_size > size)
      throw SemanticError("cp_async_source_size",
                          "source size exceeds copy size (undefined behavior)",
                          {{"source_size", source_size}, {"copy_size", size}});
    std::vector<Fragment> reads;
    // With a data-dependent count the read is possible, not certain: it is modeled as a
    // read of the largest count. A count of 0 reads nothing, so the source may then be
    // invalid; a source that does not resolve is a refusal, not an access error.
    if (source_size) {
      const auto source_contract = AccessContract(contract, "source_resource", "source_region");
      const MemoryRequest source(ops[1], source_contract, source_size, "global");
      if (least == source_size)
        reads.push_back(resources->Memory(e, source));
      else
        try {
          reads.push_back(resources->Memory(e, source));
        } catch (const SemanticError &error) {
          throw SemanticError("async_copy_source_control_not_static",
                              "Source of a data-dependent source size is not resolved: " +
                                  std::string(error.what()),
                              {{"operand_role", control}, {"operand", ops[3]}, {"source_error", error.code}});
        }
    }
    std::vector<Fragment> writes{resources->Memory(
        e, MemoryRequest(ops[0],
        AccessContract(contract, "destination_resource", "destination_region"),
        size, "shared"))};
    if ((!reads.empty() && reads[0].second != source_size) ||
        writes[0].second != size)
      throw SemanticError("cp_async_fragment_size",
                          "copy footprints disagree with opcode");
    t.classic.Issue(Copy(t, e, std::move(reads), std::move(writes)));
    return;
  }
  if (operation == "commit_group") {
    t.classic.Commit();
    Emit(t, e, Await{});
    return;
  }
  if (operation == "wait_all" || operation == "wait_group") {
    ProfileScope profile("frontend.model.wait_dependencies");
    if (operation == "wait_all")
      t.classic.Commit();
    auto wait = t.classic.PrepareWait(
        operation == "wait_all" ? 0 : Scalar(e, ops.back(), "pending groups"));
    const auto emitted = wait.requirements.size();
    Emit(t, e, Await{std::move(wait.requirements)});
    t.classic.Acknowledge(wait);
    dependency_counts["cp_async_wait"].Record(wait.original_references,
                                              emitted);
    return;
  }
  if (operation == "cp_async_mbarrier") {
    ProfileScope profile("frontend.model.completion_prefix");
    auto b = Brackets(ops);
    if (b.empty())
      throw SemanticError("cp_async_mbarrier", "missing barrier");
    auto barrier = resources->Barrier(e, b[0], contract);
    if (!Has(i, "noinc"))
      Emit(t, e, MBarrierAddArrivals{barrier, 1});
    const auto id = Next();
    auto refs = t.classic_notification.Prepare(t.classic.All());
    const auto emitted = refs.size() + 1;
    Async(t, e,
          {{{}, std::nullopt, std::move(refs)},
           {{}, ArriveEffect{barrier, 1}, {Ref(id, Milestone::kEffect, 0)}}});
    t.classic_notification.Acknowledge(id, t.classic.All().size());
    dependency_counts["cp_async_mbarrier"].Record(t.classic.All().size(),
                                                  emitted, true);
    return;
  }
  if (operation == "tma_load_complete_tx" || operation == "tma_bulk_tensor") {
    ProfileScope profile("frontend.model.tma");
    auto transfer = resources->Tma(e, TmaRequest(contract, operation == "tma_bulk_tensor"));
    // A runtime-written tensor map is read (tensormap proxy) between the TMA's issue and its
    // completion: an access of its own, issued with the TMA, that completes before the TMA's
    // complete_tx (load) or its bulk-group completion (store; not its reads, so not wait_group.read).
    std::optional<InstId> descriptor;
    if (transfer.descriptor) {
      AsyncEffect read{{MemoryEffect{AccessMode::kRead, transfer.descriptor->first, transfer.descriptor->second}}};
      descriptor = Async(t, e, {std::move(read)});
      shared_value_reads.emplace(SharedValueSite{current_coordinates, t.actor.warp.value() * 32 + t.actor.lane, e.Key()},
                                 Ref(*descriptor, Milestone::kEffect, 0));
    }
    auto id = Next();
    std::vector<AsyncEffect> effects;
    std::map<std::tuple<AddressId, std::optional<std::uint64_t>, std::uint64_t>, std::uint32_t>
        reads;
    std::map<std::uint32_t, std::vector<MilestoneRef>> destinations;
    std::vector<std::uint32_t> ranks;
    for (const auto &pair : transfer.pairs) {
      std::vector<MilestoneRef> refs;
      if (pair.source) {
        auto [a, n] = *pair.source;
        auto [it, added] =
            reads.try_emplace({a, n, pair.logical_offset}, effects.size());
        if (added)
          effects.push_back({{MemoryEffect{AccessMode::kRead, a, n}}});
        refs.push_back(Ref(id, Milestone::kEffect, it->second));
      }
      if (!pair.destination)
        continue;
      destinations[pair.target_rank].push_back(
          Ref(id, Milestone::kEffect, effects.size()));
      effects.push_back(
          {{MemoryEffect{AccessMode::kWrite, pair.destination->first,
                         pair.destination->second}},
           {},
           std::move(refs)});
      UniquePush(ranks, pair.target_rank);
    }
    if (transfer.completions.size() > ranks.size())
      throw SemanticError("tma_completion", "completion routing mismatch");
    for (std::size_t k = 0; k < transfer.completions.size(); ++k) {
      auto [barrier, size] = transfer.completions[k];
      auto refs = destinations.at(ranks[k]);
      if (descriptor) refs.push_back(Ref(*descriptor, Milestone::kEffect, 0));
      effects.push_back({{}, CompleteTxEffect{barrier, size}, std::move(refs)});
    }
    std::vector<MilestoneRef> completion;
    if (descriptor && transfer.completions.empty()) completion.push_back(Ref(*descriptor));
    Emit(t, e, AsyncOp{AsyncOpId(id.value()), std::move(effects), std::move(completion)});
    if (transfer.completions.empty())
      t.bulk.Issue(id);
    return;
  }
  if (operation == "bulk_reduce") {
    static const std::regex pattern(R"(^cp\.reduce\.async\.bulk(?:\.relaxed\.(cta|cluster|gpu|sys))?\.global\.shared::cta\.bulk_group(\.L2::cache_hint)?\.(add|min|max|inc|dec|and|or|xor)(\.noftz)?\.(\w+)$)");
    std::smatch match; const auto opcode=Opcode(i);
    if (!std::regex_match(opcode,match,pattern)) throw SemanticError("bulk_reduction_variant","unsupported global bulk reduction");
    std::pair<unsigned,unsigned> version{}; unsigned versions=0;
    for (const auto &directive : module.directives) if (directive.name==".version") {
      const auto text=Trim(module.view(directive.arguments)); const auto dot=text.find('.');
      if (dot==text.npos) throw SemanticError("bulk_reduction_version","invalid PTX version");
      version={std::stoul(text.substr(0,dot)),std::stoul(text.substr(dot+1))}; ++versions;
    }
    if (versions!=1 || version<std::pair{8U,0U}) throw SemanticError("bulk_reduction_version","bulk reduction requires PTX 8.0");
    const auto scope=match[1].matched?match[1].str():(version<std::pair{9U,3U}?"gpu":"sys");
    if (match[1].matched && version<std::pair{9U,3U}) throw SemanticError("bulk_reduction_version","explicit scope requires PTX 9.3");
    const auto reduction=match[3].str(), type=match[5].str(); const bool noftz=match[4].matched;
    static const std::map<std::string,std::set<std::string>> types={
      {"add",{"u32","s32","u64","f32","f64","f16","bf16"}},
      {"min",{"u32","s32","u64","s64","f16","bf16"}},
      {"max",{"u32","s32","u64","s64","f16","bf16"}},
      {"inc",{"u32"}},{"dec",{"u32"}}, {"and",{"b32","b64"}}, {"or",{"b32","b64"}}, {"xor",{"b32","b64"}}};
    if (!types.at(reduction).contains(type) ||
        (noftz && (reduction!="add" || (type!="f16" && type!="bf16" && type!="f32"))) ||
        (reduction=="add" && (type=="f16" || type=="bf16") && !noftz) ||
        (noftz && type=="f32" && version<std::pair{9U,4U}))
      throw SemanticError("bulk_reduction_variant","unsupported reduction operation or element type");
    if (scope=="cluster") throw SemanticError("bulk_reduction_scope","atomic scope lacks cluster membership");
    const auto brackets=Brackets(ops);
    if (ops.size()!=(match[2].matched?4U:3U) || brackets.size()!=2)
      throw SemanticError("bulk_reduction_variant","reduction operand count mismatch");
    const auto bytes=Scalar(e,ops[2],"bulk reduction size",UINT32_MAX);
    // PTX ISA cp.reduce.async.bulk: "size must be a multiple of 16. If the value is not a
    // multiple of 16, then the behavior is undefined."
    if (bytes%16) throw SemanticError("bulk_reduction_size","bulk reduction size is not a multiple of 16 (undefined behavior)",
                                      {{"size",bytes},{"isa_section","9.7.10.28.4.2 cp.reduce.async.bulk"}});
    const auto element=std::stoul(type.substr(type.find_first_of("0123456789")))/8;
    const auto [read, write] = BulkFootprints(e, MemoryRequest(brackets[1], Json::object(), bytes, "shared"),
                                              MemoryRequest(brackets[0], Json::object(), bytes, "global"),
                                              "bulk_reduction_fragmentation", "nonlinear reduction footprint");
    const auto read_base=resources->Binding(read.first), write_base=resources->Binding(write.first);
    auto fragment=[&](const ConstantAddressBinding &base,AddressId base_id,std::uint64_t offset) {
      if (base.expression.empty()) {
        resources->interner().RequireAligned(base,16,base_id);
        return resources->interner().Apply(
            ConstantStep{AllocationKey::Opaque(base.address_space,base.allocation,base.alias_scope),base.offset+offset});
      }
      auto expression=Json::parse(base.expression);
      auto old=expression["address"]["offset"];
      if (offset) expression["address"]["offset"]={{"operator","add"},{"operands",Json::array({old,offset})}};
      expression["byte_count"]=element;
      return resources->interner().Apply(
          SerializedExpressionStep(base.address_space,expression.dump(),base.alias_scope,base.allocation));
    };
    auto id=Next(); std::vector<AsyncEffect> effects;
    for (std::uint64_t offset=0;offset<bytes;offset+=element) {
      const auto index=effects.size();
      effects.push_back({{MemoryEffect{AccessMode::kRead,fragment(read_base,read.first,offset),element}}});
      effects.push_back({{MemoryEffect{AccessMode::kWrite,fragment(write_base,write.first,offset),element,
          scope=="cta"?AtomicScope::kCta:scope=="gpu"?AtomicScope::kGpu:AtomicScope::kSys}},
          {},{Ref(id,Milestone::kEffect,index)}});
    }
    Async(t,e,std::move(effects)); t.bulk.Issue(id); return;
  }
  if (operation == "bulk_complete_tx") {
    static const std::set<std::string> allowed{"async", "bulk", "weak", "shared::cta", "shared::cluster", "global",
                                               "mbarrier::complete_tx::bytes", "L2::cache_hint"};
    const auto spaces = BulkSpaces(i.modifiers, allowed);
    if ((spaces != std::vector<std::string>{"shared::cta", "global"} &&
         spaces != std::vector<std::string>{"shared::cluster", "global"} &&
         spaces != std::vector<std::string>{"shared::cluster", "shared::cta"}) ||
        (Has(i,"L2::cache_hint") && spaces[1]!="global") ||
        !Has(i, "async") || !Has(i, "bulk") ||
        !Has(i, "mbarrier::complete_tx::bytes") ||
        ops.size() != (Has(i, "L2::cache_hint") ? 5u : 4u))
      throw SemanticError("bulk_copy_variant", "unmodeled bulk copy variant");
    auto b = Brackets(ops);
    if (b.size() != 3)
      throw SemanticError("bulk_copy_variant", "bulk copy address count");
    const auto size = BulkSize(e, ops[2]);
    if (spaces[1]=="shared::cta") {
      auto destination=resources->Shared(e,b[0]);
      if (!destination || destination->first==resources->frame.self())
        throw SemanticError("bulk_copy_variant","shared bulk copy requires a resolved remote destination");
    }
    const auto footprints = BulkCopyFootprints(e, b, contract, size, spaces);
    RequireBulkCopyAlignment(footprints);
    const auto [read, write] = footprints;
    Copy(t, e, {read}, {write}, CompleteTxEffect{resources->Barrier(e, b[2], contract), size});
    return;
  }
  if (operation == "bulk_copy") {
    // cp.async.bulk.global.shared::cta.bulk_group: a linear copy from the executing CTA's
    // shared memory to global memory, completed through bulk async-groups like a TMA store.
    static const std::set<std::string> allowed{"async", "bulk", "global", "shared::cta", "bulk_group",
                                               "L2::cache_hint"};
    const auto spaces = BulkSpaces(i.modifiers, allowed);
    if (spaces != std::vector<std::string>{"global", "shared::cta"} ||
        !Has(i, "async") || !Has(i, "bulk") || !Has(i, "bulk_group") ||
        ops.size() != (Has(i, "L2::cache_hint") ? 4u : 3u))
      throw SemanticError("bulk_copy_variant", "unmodeled bulk copy variant");
    auto b = Brackets(ops);
    if (b.size() != 2)
      throw SemanticError("bulk_copy_variant", "bulk copy address count");
    const auto size = BulkSize(e, ops[2]);
    const auto footprints = BulkCopyFootprints(e, b, contract, size, spaces);
    RequireBulkCopyAlignment(footprints);
    const auto [read, write] = footprints;
    t.bulk.Issue(Copy(t, e, {read}, {write}));
    return;
  }
  if (operation == "bulk_commit_group") {
    t.bulk.Commit();
    Emit(t, e, Await{});
    return;
  }
  if (operation == "bulk_wait_group") {
    ProfileScope profile("frontend.model.wait_dependencies");
    const auto reads = Has(i, "read");
    auto wait =
        t.bulk.PrepareWait(Scalar(e, ops.back(), "pending bulk groups"), reads);
    const auto emitted = wait.requirements.size();
    Emit(t, e, Await{std::move(wait.requirements)});
    t.bulk.Acknowledge(wait);
    dependency_counts[reads ? "bulk_read_wait" : "bulk_full_wait"].Record(
        wait.original_references, emitted);
    return;
  }
  if (operation == "st_async_complete_tx") {
    std::set<std::string> types{"b32", "b64", "b128", "u32", "u64",
                                "s32", "s64", "f32",  "f64"},
        allowed = types, seen;
    for (const char *s : {"async", "weak", "shared::cluster",
                          "mbarrier::complete_tx::bytes", "v2", "v4"})
      allowed.insert(s);
    std::uint32_t element = 0, n = 0;
    for (const auto &m : i.modifiers) {
      if (!allowed.contains(m) || !seen.insert(m).second)
        throw SemanticError("st_async_variant",
                            "unsupported async store modifier");
      if (types.contains(m)) {
        ++n;
        element = std::stoul(m.substr(1)) / 8;
      }
    }
    auto b = Brackets(ops);
    if (n != 1 || (Has(i, "v2") && Has(i, "v4")) || !Has(i, "async") ||
        !Has(i, "shared::cluster") || !Has(i, "mbarrier::complete_tx::bytes") ||
        ops.size() != 3 || b.size() != 2 || ops[0] != b[0] || ops[2] != b[1] ||
        !i.memory_width)
      throw SemanticError("st_async_variant",
                          "async store operands/modifiers mismatch");
    auto dest = resources->Shared(e, b[0]),
         complete = resources->Shared(e, b[1]);
    const auto d = resources->frame.cluster_dim();
    if (!dest || !complete || std::uint64_t(d[0]) * d[1] * d[2] <= 1 ||
        dest->first != complete->first)
      throw SemanticError("st_async_owner",
                          "store needs same destination and barrier owner in "
                          "multi-CTA cluster");
    auto write = resources->Memory(e, MemoryRequest(b[0], contract, *i.memory_width,
                                   "shared::cluster"));
    if (write.second != *i.memory_width)
      throw SemanticError("st_async_fragmentation", "store extent mismatch");
    auto barrier = resources->Barrier(e, b[1], contract);
    auto id = Next();
    std::vector<AsyncEffect> effects;
    for (const auto &raw : RegistersIn(ops[1])) {
      auto r = Register(i, raw);
      if (registers.Width(r) < element)
        throw SemanticError("st_async_register_width",
                            "register narrower than element");
      AsyncEffect effect{{MemoryEffect{
          AccessMode::kRead, RegisterAddress{t.actor.id, r}, element}}};
      if (t.writes.contains(r))
        effect.predecessors.push_back(t.writes.at(r));
      effects.push_back(std::move(effect));
    }
    std::vector<MilestoneRef> refs;
    for (std::uint32_t k = 0; k < effects.size(); ++k)
      refs.push_back(Ref(id, Milestone::kEffect, k));
    effects.push_back(
        {{MemoryEffect{AccessMode::kWrite, write.first, write.second}},
         CompleteTxEffect{barrier, *i.memory_width},
         std::move(refs)});
    Async(t, e, std::move(effects));
    Emit(t, e, Await{{Ref(id, Milestone::kReadsComplete)}});
    return;
  }
  if (operation == "try_cancel_complete_tx") {
    // clc_requests spans the cluster's lanes: a worker leaves it to the consumer.
    FactSource::RequireSerial();
    if (facts->launch().ClusterLaunchCancellation())
      throw SemanticError("cluster_launch_control_outcome",
                          "CLC needs fixed no-cancellation scenario");
    const auto &frame = resources->frame;
    const auto d = frame.cluster_dim();
    const auto origin = frame.ClusterOrigin();
    if (clc_requests.contains(origin))
      throw SemanticError("repeated_failed_clc",
                          "another CLC after failed cancellation");
    auto b = Brackets(ops);
    if (b.size() != 2)
      throw SemanticError("clc_shared_addresses",
                          "CLC needs response and barrier");
    auto response = resources->Shared(e, b[0]),
         barrier = resources->Shared(e, b[1]);
    if (!response || !barrier || response->second % 16 || barrier->second % 8)
      throw SemanticError("clc_shared_addresses",
                          "CLC requires aligned shared addresses");
    std::vector<CtaRank> owners{frame.self()};
    if (Has(i, "multicast::cluster::all")) {
      owners.clear();
      for (std::uint32_t rank = 0; rank < d[0] * d[1] * d[2]; ++rank) {
        owners.push_back(frame.Rank(rank));
      }
    }
    auto id = Next();
    std::vector<AsyncEffect> effects;
    for (auto owner : owners) {
      auto a = resources->Bind("shared", "physical.shared.cta.0",
                               response->second, 16, owner);
      auto target = resources->interner().Apply(
          BarrierStep{{BarrierKey::Kind::Physical, owner, std::to_string(barrier->second)}});
      auto index = static_cast<std::uint32_t>(effects.size());
      effects.push_back({{MemoryEffect{AccessMode::kWrite, a.first, 16}}});
      effects.push_back({{},
                         CompleteTxEffect{target, 16},
                         {Ref(id, Milestone::kEffect, index)}});
    }
    Async(t, e, std::move(effects));
    clc_requests.insert(origin);
    return;
  }
  if (operation == "prefetch")
    return;
  throw SemanticError("unsupported_async_instruction",
                      "no source semantics for " + Opcode(i));
}
} // namespace sentryward::frontend::ptx
