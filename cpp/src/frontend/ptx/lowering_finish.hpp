#pragma once
#include "lowering_impl.hpp"

namespace sentryward::frontend::ptx {
LoweringResult ModelLowerer::Impl::Finish() {
  ProfileScope profile("frontend.model.finalize");
  // Class mode: the class size of an occurrence's unit (by its actor's CTA, or by a CTA's coordinates).
  const auto actor_weight = [&](ActorId actor) -> std::uint64_t {
    if (!units) return 1;
    return units->Weight(source.program.actors.at(actor.value()).cta.value());
  };
  const auto instruction_weight = [&](InstId id) -> std::uint64_t {
    return actor_weight(source.program.instructions.actor(id.value()));
  };
  if (units) {
    // A CTA whose lowering emitted a diagnostic closed its class (a ReportBearing event), so it
    // stands for itself.
    for (const auto cta : diagnostic_ctas)
      if (units->Weight(cta) != 1)
        throw std::logic_error(
            "internal error: CTA class bookkeeping (a CTA with a lowering diagnostic stands for other CTAs)");
    // Unit closure: every barrier owner and every cluster collective (keyed by its cluster
    // origin) belongs to a consumed unit. (Group2 lifetimes pair inside a cluster by construction:
    // PairGroup2Lifetimes buckets them by cluster origin.)
    const auto representative = [&](Coordinates cta) {
      for (int axis = 0; axis < 3; ++axis)
        if (cta[axis] >= units->grid[axis]) return false;
      const auto unit = units->UnitOf(units->Linear(cta));
      return units->classes.at(units->units.at(unit).klass).units.front() == unit;
    };
    for (const auto &owner : barriers.Owners())
      if (!representative(owner)) throw algorithm::UnitClassRestart{"unit_closure"};
    for (const auto &[key, arrivals] : cluster_arrivals)
      if (!representative(std::get<0>(key))) throw algorithm::UnitClassRestart{"unit_closure"};
    units->add_time_orders = orders.size();
  }
  std::vector<Group2LifetimeResult> group2_results;
  if (!group2_sites.empty()) {
    ProfileScope group2("frontend.model.group2_pairing");
    bool all = true;
    for (const auto &[dimensions, sites] : group2_sites) {
      auto result = PairGroup2Lifetimes(sites, dimensions);
      orders.insert(orders.end(), result.required_orders.begin(),
                    result.required_orders.end());
      for (const auto &limitation : result.verification_limitations)
        limitations[limitation.code] = limitation.message;
      all &= result.ParticipationProven();
      group2_results.push_back(std::move(result));
    }
    if (all)
      limitations.erase("tmem_group2_pairing");
    limitations["tmem_group2_optional_blocking"] =
        "Paired allocation management may block on peer participation; "
        "preserving termination needs an additional progress check.";
  }
  DistributedSharedLifetimes();
  source.finishEdges();
  if (!barrier_invalidations.empty()) {
    std::map<BarrierLifetimeId,std::vector<MilestoneRef>> uses;
    std::map<BarrierLifetimeId,unsigned> initializers;
    for (const auto &inst : source.program.instructions) {
      std::visit([&](const auto &op) {
        if constexpr (requires {op.barrier;}) {
          if (barrier_invalidations.contains(op.barrier)) uses[op.barrier].push_back(Ref(inst.id));
          if constexpr (std::is_same_v<std::decay_t<decltype(op)>,MBarrierInit>) ++initializers[op.barrier];
        }
      },inst.op);
      if (const auto *op=std::get_if<AsyncOp>(&inst.op))
        for (std::uint32_t index=0;index<op->effects.size();++index)
          if (op->effects[index].completion) std::visit([&](const auto &action) {
            if (barrier_invalidations.contains(action.barrier)) uses[action.barrier].push_back(Ref(inst.id,Milestone::kEffect,index));
          },*op->effects[index].completion);
    }
    for (const auto &[barrier,invalidations] : barrier_invalidations) {
      if (invalidations.size()!=1 || initializers[barrier]!=1) {
        Diagnostic("mbarrier_inval_lifetime","invalidation requires a single initialization and final invalidation",
                   nullptr, nullptr, Json::object(), instruction_weight(invalidations.front()));
        continue;
      }
      for (auto use : uses[barrier]) orders.push_back({"mbarrier_final_invalidation",use,Ref(invalidations.front(),Milestone::kIssue),
          "Every operation on the mbarrier must finish before its final invalidation."});
    }
  }
  std::map<ClusterKey, InstId> gates;
  std::map<InstId, InstId> replacements;
  for (const auto &wait : cluster_waits) {
    auto &arrivals = cluster_arrivals[wait.key];
    if (arrivals.size() != wait.expected) {
      Diagnostic("incomplete_cluster_collective",
                 "cluster phase lacks required thread arrivals", nullptr, nullptr, Json::object(),
                 instruction_weight(wait.id));
      continue;
    }
    InstId gate;
    if (!gates.contains(wait.key)) {
      std::vector<MilestoneRef> refs;
      for (auto [actor, id] : arrivals) {
        (void)actor;
        refs.push_back(Ref(id));
      }
      auto first = refs.front().instruction;
      const auto actor = source.instruction(first).actor;
      const auto location = source.instruction(first).location;
      gate = Next();
      (void)source.append(actor, location,
                          AsyncOp{AsyncOpId(gate.value()),
                                  {{{}, std::nullopt, std::move(refs)}}},
                          {}, false);
      gates.emplace(wait.key, gate);
      replacements.emplace(first, gate);
    } else
      gate = gates.at(wait.key);
    UniquePush(std::get<Await>(source.editInstruction(wait.id).op).requirements,
               Ref(gate));
  }
  if(!replacements.empty())for(std::size_t i=0;i<source.program.edges.size();++i) {
    const auto edge=source.program.edges[i];
    if(edge.kind==SourceEdgeKind::kProgramOrder && replacements.contains(edge.from))
      source.program.edges.edit(i).from=replacements.at(edge.from);
  }
  for (auto [from, to] : replacements)
    source.program.edges.push_back({from, to, SourceEdgeKind::kProgramOrder});
  for(const auto &dependency:shared_value_dependencies) {
    const auto write=shared_value_writes.find(dependency.write);
    const auto read=shared_value_reads.find(dependency.read);
    if(write==shared_value_writes.end() || write->second.empty() || read==shared_value_reads.end()) {
      Diagnostic("shared_constant_source_unmodeled","A shared constant's source write or consuming read has no modeled memory event.",
                 nullptr, nullptr, Json::object(),
                 units ? units->Weight(units->Linear(dependency.read.cta)) : 1);
      continue;
    }
    for(auto from:write->second)orders.push_back({"shared_constant_read_from",from,read->second,
        "The write supplying a frontend shared-memory constant must happen before its read; interfering writes are checked by memory analysis."});
  }
  // Each copy's lane writes happen before every descriptor read selected from it: one order per
  // lane write, its reads one shared target domain.
  std::map<SharedValueSite,std::vector<MilestoneRef>> tensor_map_targets;
  for(const auto &dependency:tensor_map_dependencies) {
    const auto read=shared_value_reads.find(dependency.read);
    if(read==shared_value_reads.end()) {
      Diagnostic("tensor_map_read_unmodeled","A runtime tensor map read has no modeled descriptor access.",
                 nullptr, nullptr, Json::object(), units ? units->Weight(units->Linear(dependency.read.cta)) : 1);
      continue;
    }
    tensor_map_targets[dependency.write].push_back(read->second);
  }
  for(auto &[site,targets]:tensor_map_targets) {
    const auto write=shared_value_writes.find(site);
    if(write==shared_value_writes.end() || write->second.size()!=32) {
      Diagnostic("tensor_map_source_unmodeled","A runtime tensor map's copy has no modeled write for every lane.",
                 nullptr, nullptr, Json::object(), units ? units->Weight(units->Linear(site.cta)) : 1);
      continue;
    }
    const SharedOptional<std::vector<MilestoneRef>> domain(std::move(targets));
    for(auto from:write->second)orders.push_back({"tensor_map_read_from",from,domain->front(),
        "A runtime tensor map's selected copy must happen before the TMA reads it; interfering writes are checked by memory analysis.",
        std::nullopt,domain});
  }
  ProfileCall("frontend.model.seal_families",[&]{source.program.instructions.sealThreadFamilies();});
  LoweringResult result;
  result.mma_layouts = std::move(mma_layouts);
  for (const auto &[object, begin] : tensormap_destinations) result.tensormap_destinations.push_back({object, begin});
  for (std::size_t i = 0; i < occurrence_counts.size(); ++i)
    if (occurrence_counts[i])
      counts[opcode_texts[i]] += occurrence_counts[i];
  // Class mode: every launch total weighted by class size.
  std::uint64_t cta_count = ctas, thread_count = source.program.actors.size();
  if (units) {
    cta_count = 0;
    for (const auto &[cta, sparse] : cta_occurrences) {
      const auto weight = units->Weight(cta);
      cta_count += weight;
      for (const auto &[id, count] : sparse) counts[opcode_texts[id]] += count * weight;
      // Containment members execute fewer records of some instructions (and more taken branches).
      const auto &adjust = units->ClassOf(cta).occurrence_adjust;
      if (const auto rank = units->RankOf(cta); rank < adjust.size())
        for (const auto &[id, change] : adjust[rank]) {
          if (id >= opcode_texts.size()) throw std::logic_error("internal error: CTA class bookkeeping (count change)");
          auto &total = counts[opcode_texts[id]];
          total = static_cast<std::uint64_t>(static_cast<std::int64_t>(total) + change);
        }
    }
    thread_count = 0;
    for (const auto &actor : source.program.actors) thread_count += units->Weight(actor.cta.value());
    // Members' per-CTA violation texts: the representative's CTA-free part, rendered with the
    // member's same-rank coordinates, into the same `violations` set.
    for (const auto &[cta, code, suffix] : cta_violations) {
      const auto &klass = units->ClassOf(cta);
      const auto rank = units->RankOf(cta);
      for (std::size_t i = 1; i < klass.units.size(); ++i) {
        const UnitFrame member(units->CtaCoordinates(units->units.at(klass.units[i]).ctas.at(rank)), std::nullopt);
        violations.emplace(code, "CTA " + member.Text(member.self()) + ": " + suffix);
      }
    }
  }
  result.metadata = {
      {"cta_count", cta_count},
      {"thread_count", thread_count},
      {"cluster_completion_joins", gates.size()},
      {"reached_opcodes", counts},
      {"lowered_instruction_count", source.program.instructions.size()},
      {"unsupported_occurrence_counts", errors},
      {"diagnostic_sample_count", diagnostics.size()},
      {"implementation", "cpp"}};
  result.metadata["shared_constant_propagation"]={{"read_from_dependencies",shared_value_dependencies.size()},
      {"validation","pending_backend"},{"representation","sparse_known_bytes"}};
  // Bounded integers round-tripped through shared memory (statistics; absent when none was stored).
  if(shared_bounded_writes || shared_bounded_reads) {
    result.metadata["shared_constant_propagation"]["bounded_write_sites"]=shared_bounded_writes;
    result.metadata["shared_constant_propagation"]["bounded_read_from_dependencies"]=shared_bounded_reads;
  }
  if(!source.program.exit_gates.empty()) {
    std::set<std::pair<BarrierLifetimeId,std::uint32_t>> phases;
    for(const auto &gate:source.program.exit_gates)phases.emplace(gate.barrier,gate.phase);
    result.metadata["cta_wide_exit_gates"]={{"gates",source.program.exit_gates.size()},{"phases",phases.size()}};
  }
  if(!tensor_map_dependencies.empty())
    result.metadata["runtime_tensor_maps"]={{"read_from_dependencies",tensor_map_dependencies.size()},
        {"copies",tensor_map_targets.size()},{"locations",tensormap_destinations.size()}};
  result.metadata["source_binding_storage"] = {
      {"logical_instructions", source.program.instructions.size()},
      {"stored_bindings", source.program.instructions.storedBindingCount()},
      {"binding_templates", source.program.instructions.bindingTemplateCount()},
      {"generated_memory_instances", generated_memory_instances},
      {"generated_address_rows", generated_address_rows},
      {"generated_body_rows", generated_body_rows}};
  result.metadata["source_edge_storage"] = {
      {"logical_edges",source.program.edges.size()},
      {"stored_bindings",source.program.edges.storedBindingCount()},
      {"binding_templates",source.program.edges.bindingTemplateCount()}};
  result.metadata["dependency_compression"] = Json::object();
  for (const auto &[kind, counts] : dependency_counts)
    result.metadata["dependency_compression"][kind] = {
        {"consumers", counts.consumers},
        {"original_references", counts.original_references},
        {"emitted_references", counts.emitted_references},
        {"prefix_effects", counts.prefix_effects}};
  if (!group2_results.empty()) {
    ProfileScope progress("frontend.model.group2_progress");
    bool all = true;
    std::size_t checked = 0;
    Json blocked = Json::array(), premises = Json::array();
    std::set<std::string> seen;
    for (const auto &group : group2_results) {
      auto check = CheckGroup2OptionalBlocking(source.program, group, poll);
      checked += check.checked_pairs;
      all &= check.discharged;
      for (const auto &reason : check.blocked)
        blocked.push_back(reason);
      for (const auto &premise : check.premises)
        if (seen.insert(premise).second)
          premises.push_back(premise);
    }
    result.metadata["group2_progress"] = {{"checked_pairs", checked},
                                          {"blocked", blocked},
                                          {"premises", premises}};
    if (all)
      limitations.erase("tmem_group2_optional_blocking");
  }
  if (!plans.empty() || !copy_plans.empty()) {
    ProfileScope refinement("frontend.model.refinement");
    auto outcome = Refine(source.program, orders, plans, histories, copy_plans);
    for (auto it = outcome.begin(); it != outcome.end(); ++it)
      result.metadata[it.key()] = it.value();
    for (const auto &code : outcome["discharged_codes"])
      limitations.erase(code.get<std::string>());
  }
  // Threads are lowered by cohort, not by ID: report each diagnostic's actors
  // as sorted, merged ranges.
  for (auto &diagnostic : diagnostics) {
    if (!diagnostic.contains("details") || !diagnostic["details"].contains("actor_ranges")) continue;
    auto &ranges = diagnostic["details"]["actor_ranges"];
    std::vector<std::pair<std::uint32_t, std::uint32_t>> sorted;
    for (const auto &range : ranges) sorted.emplace_back(range[0].get<std::uint32_t>(), range[1].get<std::uint32_t>());
    std::sort(sorted.begin(), sorted.end());
    Json merged = Json::array();
    for (const auto &[first, last] : sorted) {
      if (!merged.empty() && merged.back()[1].get<std::uint32_t>() + 1 >= first)
        merged.back()[1] = std::max(merged.back()[1].get<std::uint32_t>(), last);
      else merged.push_back({first, last});
    }
    ranges = std::move(merged);
  }
  result.diagnostics = std::move(diagnostics);
  result.artifact.program = std::move(source.program);
  result.artifact.addresses = UnitInterner::Release(std::move(addresses));
  result.artifact.required_orders = std::move(orders);
  for (auto &[code, message] : limitations)
    result.artifact.verification_limitations.push_back({code, message});
  for (const auto &[code, message] : violations)
    result.artifact.source_violations.push_back({code, message});
  return result;
}
ModelLowerer::ModelLowerer(const Module &m, const Function &f,
                           std::function<void()> poll, bool retain_matrix_layout,
                           bool atomics, std::size_t threads)
    : impl_(std::make_unique<Impl>(m, f, std::move(poll))) {
  impl_->retain_matrix_layout = retain_matrix_layout;
  impl_->atomics = atomics;
  impl_->threads = threads;
}
ModelLowerer::~ModelLowerer() = default;
} // namespace sentryward::frontend::ptx
