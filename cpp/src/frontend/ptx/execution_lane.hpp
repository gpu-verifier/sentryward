#pragma once
#include "execution_impl.hpp"

namespace sentryward::frontend::ptx {
namespace {
LaneTask Executor::Lane(const WarpContract &warp, std::uint32_t lane,
                        WarpContext &context) const {

  std::uint32_t pc = 0;
  if (warp.entry_label) {
    auto found = labels_.find(*warp.entry_label);
    if (found == labels_.end())
      throw std::runtime_error("unknown warp entry label");
    pc = found->second;
  }
  ProfileScope initialize("frontend.execute.lane.initialize");
  State state = Initial(warp, lane);
  initialize.nodes = state.registers.size();
  initialize.Stop();
  LaneTrace trace;
  trace.warp_id = warp.warp_id;
  trace.lane_id = lane;
  trace.instructions.reserve(2048);
  std::uint64_t record_cursor = 0;
  bool normalized_records = false;
  auto append = [&](ExecutedInstruction record) {
    if (c_.compress_thread_records)
      record_pool_.Intern(record, record_cursor);
    trace.instructions.push_back(std::move(record));
  };
  if (data_branch_pass) PrepareControl();
  std::uint32_t previous_block = kMissing;
  std::map<std::uint32_t, std::uint64_t> loop_iterations;
  // Branches and predicated terminal instructions must use identical control
  // classification. Treating @p ret as an ordinary guarded instruction can
  // otherwise force an unknown polling loop to terminate.
  auto representative_outcome = [&](std::uint32_t pc) -> std::optional<bool> {
    const auto &i = *decoded_[pc].instruction;
    std::optional<bool> taken;
    const auto choice = ClassifyControl(pc);
    if (choice.kind == ControlLoop::Polling)
      Error(i, warp.warp_id, lane,
            "Polling control requires a modeled wait; neither assumption can "
            "force completion.",
            "polling_wait_not_modeled");
    if (choice.kind == ControlLoop::Unknown)
      Error(i, warp.warp_id, lane,
            "Control is not a structured if/else or a classified "
            "computation-loop test; supply explicit control information.",
            "loop_control_not_classified");
    const bool ordinary = choice.kind == ControlLoop::None;
    const char *key =
        ordinary ? "representative_branches" : "bounded_computation_loops";
    if (Assumed(c_, key)) {
      if (ordinary)
        taken = false;
      else {
        const auto limit = c_.raw.value("data_branch_loop_iterations", 4u);
        const auto completed =
            loop_iterations.at(choice.loop) + (choice.post_test ? 1 : 0);
        taken = (completed < limit) == choice.taken_stays;
      }
      auto &decision = trace.data_branch_decisions[i.line];
      decision.assumption = key;
      decision.rule = ordinary ? "fallthrough" : "body_iteration_bound";
      ++(*taken ? decision.taken : decision.not_taken);
    }
    return taken;
  };
  std::map<std::uint32_t, ProbeValue> probe_values;
  std::map<DynamicKey, WaitProbe> wait_probes;
  std::map<std::uint32_t, std::map<std::uint32_t, ProbeValue>> probe_joins;
  for (std::uint64_t steps = 0;; ++steps) {
    if ((steps & 1023) == 0) {
      if (poll_)
        poll_();
    }
    if (steps >= c_.max_steps_per_lane)
      throw ExecutionError("execution_step_limit_exceeded", pc < f_.instructions.size() ? f_.instructions[pc].line : 0, "warp " + std::to_string(warp.warp_id) +
                               " lane " + std::to_string(lane) +
                               ": exceeded max_steps_per_lane at pc " +
                               std::to_string(pc));
    if (pc >= decoded_.size())
      throw std::runtime_error("PTX control fell outside instruction stream");
    if (data_branch_pass) {
      const auto block = control_block_for_pc_[pc];
      // Update only at block transitions (including a one-block loop backedge).
      if (block != previous_block || pc == control_blocks_[block].begin) {
        const auto &active = control_blocks_[block].loops;
        if (previous_block != kMissing)
          for (auto id : control_blocks_[previous_block].loops)
            if (std::find(active.begin(), active.end(), id) == active.end()) loop_iterations.erase(id);
        for (auto id : active) {
          const auto &loop = control_loops_[id];
          auto [it, entered] = loop_iterations.try_emplace(id, 0);
          if (!entered && block == loop.header && loop.latches.contains(previous_block)) ++it->second;
        }
        previous_block = block;
      }
    }
    const auto &d = decoded_[pc];
    const auto &i = *d.instruction;
    if (auto join = probe_joins.find(pc); join != probe_joins.end()) {
      for (const auto &[reg, value] : join->second)
        if (value.alternative != state.registers[reg])
          probe_values[reg] = value;
      probe_joins.erase(join);
    }
    const auto &contract = c_.InstructionContract(i.line, i.id, warp.warp_id,
                                                  state.occurrences[pc]);
    if (i.control == "direct_branch") {
      if (!i.branch_target)
        Error(i, warp.warp_id, lane, "unresolved direct branch");
      auto target = *i.branch_target;
      std::optional<std::pair<std::string, Integer>> assumed_equality;
      if (c_.fixed_input.value("assume_trap_assertions", false))
        if (auto assertion = Assertion(i, target)) {
          const auto &[reg, expected] = *assertion;
          if (!Unknown(state.registers[reg]) && state.registers[reg] != Scalar(expected))
            Error(i, warp.warp_id, lane, "assertion assumption contradicts a known value");
          state.registers[reg] = expected;
          state.registers[d.predicate] = true;
          assumed_equality = {names_[reg], expected};
          ++trace.assumed_assertions;
        }
      if (d.predicate != kMissing) {
        auto probe = probe_values.find(d.predicate);
        if (probe != probe_values.end()) {
          auto actual = Truth(state.registers[d.predicate]),
               alternative = Truth(probe->second.alternative);
          if (!actual || !alternative)
            Error(i, warp.warp_id, lane,
                  "probe-dependent branch has unresolved alternative");
          bool a = *actual, b = *alternative;
          if (i.predicate_negated) {
            a = !a;
            b = !b;
          }
          if (a != b) {
            auto fallback =
                !a && b
                    ? Fallback(d, state, target,
                               wait_probes.at(probe->second.key).identity,
                               warp.warp_id, lane, context)
                    : std::optional<
                          std::pair<std::uint32_t, std::set<std::uint32_t>>>{};
            if (!fallback || c_.BranchOutcome(i.line, warp.warp_id, lane,
                                              state.branch_visits[pc]))
              Error(i, warp.warp_id, lane,
                    "schedule-dependent probe lacks same-target "
                    "side-effect-free fallback wait", "schedule_dependent_wait_unsupported");
            auto &record = trace.instructions[wait_probes.at(probe->second.key)
                                                  .trace_index];
            normalized_records = true;
            record.mutableMetadata().canonical_wait_probe = true;
            record.mutableMetadata().schedule_seed_outcome.reset();
            record.mutableMetadata().probe_fallback_key =
                DynamicKey{fallback->first, state.occurrences[fallback->first]};
            auto &join = probe_joins[target];
            for (auto reg : fallback->second) {
              auto previous = probe_values.find(reg);
              join[reg] = {probe->second.key,
                           previous != probe_values.end() &&
                                   previous->second.key == probe->second.key
                               ? previous->second.alternative
                               : state.registers[reg]};
            }
          }
        }
      }
      // An unresolved forward branch over pure computation is skipped as an opaque region. In the
      // exact last-resort pass, the region may have several arms (if-else, nested ifs): every
      // path from the branch reaches one join without memory, synchronization or control
      // effects beyond forward branches inside it.
      const bool unresolved = d.predicate != kMissing && Unknown(state.registers[d.predicate]) &&
                              !c_.BranchOutcome(i.line, warp.warp_id, lane, state.branch_visits[pc]);
      // A data predicate (table atoms) prefers an opaque region, with several arms in any pass; without
      // one the branch follows its exact outcome.
      const bool data = d.predicate != kMissing && DataPredicate(state.registers[d.predicate]) &&
                        !c_.BranchOutcome(i.line, warp.warp_id, lane, state.branch_visits[pc]) &&
                        DataOpaqueAdmitted(pc);
      if (data) data_opaque_used = true;
      auto region = (unresolved || data) && target > pc + 1 ? OpaqueRegionJoin(pc, false) : std::nullopt;
      if ((unresolved || data) && !region && (unresolved ? exact_control_pass : target > pc + 1) &&
          (region = OpaqueRegionJoin(pc, true)))
        ++trace.opaque_regions[i.line];
      // Shared-memory routing: an undecided (not a data-predicate) branch whose predicate may read
      // shared memory is not skipped by a run without the shared-memory model.
      if (region && unresolved && !data) {
        if (SharedRouted(pc, context.shared != nullptr)) RouteToSharedModel(i);
        RoutedBranchUndecided(i, context.shared != nullptr);
      }
      const bool diamond = region.has_value();
      const std::uint32_t join = region.value_or(target);
      if (diamond) {
        auto occurrence = state.occurrences[pc]++;
        ++state.branch_visits[pc];
        for (std::uint32_t index = pc + 1; index < join; ++index)
          for (auto reg : decoded_[index].destinations) {
            state.registers[reg] = Scalar{};
            state.origins[reg].reset();
            probe_values.erase(reg);
          }
        ForgetRegionLocal(pc, join, state.local);
        auto record = Record(d, state, occurrence, {});
        OpaqueRegionNames(pc, join, record.mutableMetadata().opaque_compute_register_reads,
                          record.mutableMetadata().opaque_compute_register_writes);
        for (auto index = pc + 1; index < join; ++index)
          record.mutableMetadata().opaque_compute_diamond_indices.push_back(index);
        record.mutableMetadata().opaque_compute_join_label =
            join == target ? i.branch_label : "pc:" + std::to_string(join);
        append(std::move(record));
        if (convergence_) context.Diverge(lane);
        pc = join;
        continue;
      }
      auto visit = state.branch_visits[pc]++;
      std::optional<bool> bound;
      if (target <= pc)
        if (auto loop = [&]() {
              auto found = c_.FindLoopBound(i.branch_label, warp.warp_id);
              return found ? found
                           : c_.FindLoopBound(m_.view(i.operands[0]),
                                              warp.warp_id);
            }()) {
          if (loop->backedge_line && *loop->backedge_line != i.line)
            Error(i, warp.warp_id, lane,
                  "loop bound names a different backedge line");
          auto count = state.backedges[i.branch_label]++;
          bound = count < loop->trip_count - 1;
        }
      auto override = c_.BranchOutcome(i.line, warp.warp_id, lane, visit);
      auto predicate = d.predicate == kMissing
                           ? std::optional<bool>(true)
                           : Truth(state.registers[d.predicate]);
      if (predicate && i.predicate_negated)
        *predicate = !*predicate;
      if (!bound && override && predicate && *override != *predicate) {
        static_choice_lines_.insert(i.line);
        override.reset();
      }
      auto supplied = bound ? bound : override;
      if (supplied && predicate && *supplied != *predicate)
        Error(i, warp.warp_id, lane,
              "known predicate disagrees with fixed-input branch contract");
      auto taken = supplied ? supplied : predicate;
      if (!taken && d.predicate != kMissing && target > pc + 1)
        if (const auto *arm = MergedArmAt(pc); arm && (!arm->aligned || arm->uniform)) {
          if (SharedRouted(pc, context.shared != nullptr)) RouteToSharedModel(i);
          RoutedBranchUndecided(i, context.shared != nullptr);
          // Two-arm merge: this lane runs the fall-through arm (the superset arm); the arm's
          // registers are dead after the join: nothing after it reads a register the arm writes.
          auto record = Record(d, state, state.occurrences[pc]++, {});
          record.mutableMetadata().merged_arm_join = target;
          append(std::move(record));
          if (convergence_) context.Diverge(lane);
          ++pc;
          continue;
        }
      if (!taken && data_branch_pass)
        taken = representative_outcome(pc);
      // The last pass: a region that only an unproved st.local kept from being skipped.
      if (const auto refusal = region_store_refusals_.find(pc);
          !taken && exact_control_pass && (data_branch_pass || !DataBranchesAssumed(c_)) &&
          refusal != region_store_refusals_.end())
        throw ExecutionError("lifted_region_local_store", i.line,
                             "The data-dependent forward region of this branch holds a " + refusal->second);
      if (!taken)
        UnresolvedControl(i, state, trace, d.predicate);
      auto occurrence = state.occurrences[pc]++;
      const auto capture = override && !bound && d.predicate != kMissing
          ? CaptureDecision(i.line, *taken, visit) : std::optional<BranchScopeDecision>{};
      if (*taken || capture) {
        ExecutedInstruction record;
        if (capture) {
          record = Record(d, state, occurrence, true);
          record.mutableMetadata().branch_scope_decision = capture;
        } else {
          record.mutableMetadata().instruction_id = pc;
          record.mutableMetadata().occurrence = occurrence;
          record.mutableMetadata().predicate_value = true;
        }
        record.mutableMetadata().assumed_equality = std::move(assumed_equality);
        append(std::move(record));
      }
      if (convergence_ && d.predicate != kMissing) context.Decide(lane, pc, *taken ? target : pc + 1);
      pc = *taken ? target : pc + 1;
      continue;
    }
    if (d.predicate != kMissing && probe_values.contains(d.predicate))
      Error(i, warp.warp_id, lane,
            "schedule-dependent probe guards noncanonical instruction");
    bool active = true;
    if (d.predicate != kMissing) {
      auto known = Truth(state.registers[d.predicate]);
      if (known)
        active = *known;
      else if (contract.contains("active_lanes") &&
               contract["active_lanes"].is_array())
        active = std::find(contract["active_lanes"].begin(),
                           contract["active_lanes"].end(),
                           Json(lane)) != contract["active_lanes"].end();
      else if (contract.contains("predicate") &&
               contract["predicate"].is_boolean())
        active = contract["predicate"].get<bool>();
      else if (data_branch_pass && In(i.control, {"return", "exit", "trap"})) {
        const auto chosen = representative_outcome(pc);
        if (!chosen)
          UnresolvedControl(i, state, trace, d.predicate);
        // The common guard path below applies predicate negation once.
        active = i.predicate_negated ? !*chosen : *chosen;
      } else if (data_branch_pass && Assumed(c_, "representative_branches")) {
        // Representative control flow: an instruction whose guard is not
        // determined by the fixed input executes.
        active = !i.predicate_negated;
        auto &decision = trace.data_branch_decisions[i.line];
        decision.assumption = "representative_branches";
        decision.rule = "guard_executes";
        ++decision.taken;
      } else
        UnresolvedControl(i, state, trace, d.predicate);
      if (i.predicate_negated)
        active = !active;
    }
    auto occurrence = state.occurrences[pc]++;
    if (!active) {
      ++pc;
      continue;
    }
    const auto& assumptions = c_.raw.at("semantic_assumptions");
    if (i.opcode == "elect" && !assumptions.at("representative_election").get<bool>())
      throw ExecutionError("assumption_required_representative_election", i.line,
          "Election outcomes are unsupported. Enable Representative election (representative_election) to analyze one representative lane.");
    if (i.opcode == "clusterlaunchcontrol" && !assumptions.at("clc_always_fails").get<bool>())
      throw ExecutionError("assumption_required_clc_always_fails", i.line,
          "CLC outcomes are unsupported. Enable Failed launch cancellation (clc_always_fails) to analyze failed requests.");
    std::map<std::uint32_t, ProbeValue> updates;
    if (!probe_values.empty()) {
      std::map<std::uint32_t, ProbeValue> dependencies;
      std::set<DynamicKey> keys;
      for (std::size_t operand = d.destinations.empty() ? 0 : 1;
           operand < d.operands.size(); ++operand)
        for (auto reg : d.operands[operand].registers)
          if (auto found = probe_values.find(reg);
              found != probe_values.end()) {
            dependencies.emplace(*found);
            keys.insert(found->second.key);
          }
      if (!dependencies.empty()) {
        if (i.semantics.category != "compute" ||
            !In(i.opcode,
                {"mov", "cvt", "selp", "setp", "and", "or", "xor", "not"}) ||
            keys.size() != 1)
          Error(i, warp.warp_id, lane,
                "schedule-dependent probe escapes canonical fallback control");
        State shadow = state;
        for (const auto &[reg, value] : dependencies)
          shadow.registers[reg] = value.alternative;
        Data(d, shadow, warp.warp_id, lane, occurrence, contract, context);
        for (auto reg : d.destinations) {
          if (Unknown(shadow.registers[reg]))
            Error(i, warp.warp_id, lane, "unresolved probe-result computation");
          updates[reg] = {*keys.begin(), shadow.registers[reg]};
        }
      }
    }
    if (d.wait_loop) {
      auto record = Record(d, state, occurrence, true);
      if (context.shared) for (;;) {
        bool blocked=false;
        try { SharedData(d,state,warp.warp_id,lane,occurrence,contract,context,trace,true); }
        catch (const SharedBlocked &) { blocked=true; }
        if (!blocked) break;
        co_yield Deferred{lane,pc,occurrence,true};
      }
      record.mutableMetadata().collapsed_wait_loop = true;
      record.mutableMetadata().normalized_wait_result = true;
      record.mutableMetadata().wait_exit_label = d.wait_loop->first;
      for (auto index = pc; index < (d.replay_wait_result ? pc+1 : d.wait_loop->second); ++index)
        record.mutableMetadata().collapsed_wait_indices.push_back(index);
      append(std::move(record));
      Set(d, state, {true});
      Origins(d, state, occurrence);
      for (auto reg : d.destinations)
        probe_values.erase(reg);
      pc = d.replay_wait_result ? pc+1 : d.wait_loop->second;
      continue;
    }
    auto record = Record(d, state, occurrence, true);
    bool probe = IsWait(i);
    if (probe)
      record.mutableMetadata().schedule_seed_outcome = "WaitAgain";
    append(std::move(record));
    if (i.control == "indirect_branch") {
      const auto index=ScalarInteger(Value(d.operands.front(),state));
      if (!index) UnresolvedControl(i,state,trace,d.operands.front().reg);
      if (*index<0 || *index>=i.indirect_targets.size())
        Error(i,warp.warp_id,lane,"indirect branch index is outside its target table");
      if (convergence_) context.Decide(lane, pc, i.indirect_targets[index->convert_to<std::size_t>()]);
      pc=i.indirect_targets[index->convert_to<std::size_t>()];
      continue;
    }
    if (i.control == "call") {
      if (!contract.value("opaque_call", false) ||
          !contract.value("contains_no_sync_or_memory", false))
        Error(i, warp.warp_id, lane,
              "call lacks explicit no-sync/no-memory summary");
      Invalidate(d, state);
      state.local.clear();
      Origins(d, state, occurrence);
      ++pc;
      continue;
    }
    if (i.control == "return" || i.control == "exit")
      break;
    if (i.control == "trap")
      Error(i, warp.warp_id, lane, "fixed path reaches trap");
    for (;;) {
      std::optional<Deferred> dependency;
      try {
        if (!SharedData(d,state,warp.warp_id,lane,occurrence,contract,context,trace))
          Data(d, state, warp.warp_id, lane, occurrence, contract, context, probe);
      } catch (const Deferred &waiting) {
        dependency = waiting;
      } catch (const SharedBlocked &) {
        dependency = Deferred{lane,pc,occurrence,true};
      }
      if (!dependency) break;
      co_yield *dependency;
    }
    if (i.opcode == "clusterlaunchcontrol" && Has(i, "query_cancel") && Has(i, "is_canceled") &&
        !d.destinations.empty()) {
      normalized_records = true;
      trace.instructions.back().mutableMetadata().normalized_clc_result = Truth(state.registers[d.destinations.front()]);
    }
    Origins(d, state, occurrence);
    for (auto reg : d.destinations)
      probe_values.erase(reg);
    for (auto &[reg, value] : updates)
      if (value.alternative != state.registers[reg])
        probe_values[reg] = std::move(value);
    if (probe) {
      auto identity = Identity(d, state);
      if (!identity)
        Error(i, warp.warp_id, lane,
              "nonblocking wait probe has unresolved barrier or target");
      DynamicKey key{pc, occurrence};
      wait_probes[key] = {trace.instructions.size() - 1, *identity};
      for (auto reg : d.destinations)
        probe_values[reg] = {key, true};
    }
    ++pc;
  }
  for (const auto &[key, probe] : wait_probes) {
    normalized_records = true;
    auto &record = trace.instructions[probe.trace_index];
    record.mutableMetadata().canonical_wait_probe = true;
    record.mutableMetadata().normalized_wait_result = false;
    record.mutableMetadata().schedule_seed_outcome.reset();
  }
  if (c_.compress_thread_records && normalized_records) {
    record_cursor = 0;
    std::size_t position = 0;
    for (auto &record : trace.instructions) {
      if ((position++ & 1023) == 0 && poll_) poll_();
      record_pool_.Intern(record, record_cursor);
    }
  }
  for (std::uint32_t index = 0; index < state.branch_visits.size(); ++index)
    if (state.branch_visits[index])
      trace.branch_visits[index] = state.branch_visits[index];
  trace.loop_backedges = std::move(state.backedges);
  if (convergence_) context.Diverge(lane);
  co_return trace;
}

} // namespace
} // namespace sentryward::frontend::ptx
