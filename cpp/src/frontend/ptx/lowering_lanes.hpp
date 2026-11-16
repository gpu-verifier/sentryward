#pragma once
#include "lowering_impl.hpp"

namespace sentryward::frontend::ptx {
void ModelLowerer::Impl::LowerThread(const CtaFacts &cta_facts, LaneRef lane_ref, Thread &thread,
                                     const std::vector<std::size_t> *positions,
                                     std::span<const std::optional<std::vector<RegId>>> read_slices,
                                     std::uint64_t &visits, IterationStarts *iteration_starts) {
  const auto count=positions ? positions->size() : cta_facts.Size(lane_ref);
  const auto emitted=[&] { return lane ? lane->instructions.size() : source.program.instructions.size(); };
  const auto first=emitted();
  // A record whose PTX instruction precedes the previous one follows a back edge.
  std::optional<std::uint32_t> previous_instruction;
  for (std::size_t ordinal=0; ordinal<count; ++ordinal) {
    const auto index=positions ? (*positions)[ordinal] : ordinal;
    const auto metadata=cta_facts.Meta({lane_ref,index});
    const auto &semantics=function.instructions[metadata.instruction_id()].semantics;
    if (iteration_starts) {
      const auto instruction=static_cast<std::uint32_t>(metadata.instruction_id());
      if (previous_instruction && instruction < *previous_instruction)
        iteration_starts->emplace_back(emitted()-first, instruction);
      previous_instruction=instruction;
    }
    RecordHolder concrete; // a Slice over read_slices, or Full; the metadata record otherwise
    if (!cta_facts.Expanded(lane_ref)) {
      if(const auto& slice=read_slices[metadata.instruction_id()])
        concrete=cta_facts.ReadSlice({lane_ref,index},*slice);
      else if(semantics.category!="compute" && semantics.category!="control")
        concrete=cta_facts.ReadFull({lane_ref,index});
    }
    const auto e=concrete ? concrete.view() : metadata;
    if (e.predicate_value() == false)
      continue;
    if ((++visits & 1023) == 0)
      Poll();
    const auto &i = function.instructions[e.instruction_id()];
    if (occurrence_counts[i.id]++ == 0 && lane) lane->counted.push_back(i.id);
    // A lane whose lowering of an .aligned site fails executed it (RegistersAlignedLanes).
    const auto executed_aligned = [&] {
      if (RegistersAlignedLanes(i)) aligned[Key(thread.actor.warp.value(), e)].failed.insert(thread.actor.lane);
    };
    try {
      Instruction(thread, e);
    } catch (const SemanticError &error) {
      executed_aligned();
      Diagnostic(error.code, error.what(), &thread, &e, error.details);
    } catch (const LayoutError &error) {
      executed_aligned();
      Diagnostic(error.code, error.what(), &thread, &e, error.details);
    } catch (const std::bad_alloc &) {
      throw;
    } catch (const std::exception &error) {
      executed_aligned();
      Diagnostic("unexpected_exception", error.what(), &thread, &e,
                 {{"exception_type",typeid(error).name()}});
    }
  }
}
// A worker Impl's view of the consumer's current CTA: the read-only per-CTA state its lanes
// consult, a Resources of its own standing for `like` (same facts, frame, records and bridge, so
// the same resolution path) over its own tables, and the consumer's matrix resolver (it built
// every lane's expansion already; its row cache is shared by position, MatrixResolver). On the
// consuming thread.
void ModelLowerer::Impl::BeginLaneWork(const Impl &consumer, Resources &like, const ValueAddresses &bridge) {
  facts = consumer.facts;
  tmem = consumer.tmem;
  matrix = consumer.matrix;
  current_coordinates = consumer.current_coordinates;
  shared_value_sites = consumer.shared_value_sites;
  shared_value_instructions = consumer.shared_value_instructions;
  tensor_map_copies = consumer.tensor_map_copies;
  actors = consumer.actors;
  index_sites = consumer.index_sites;
  lane->resources.emplace(like, addresses, barriers, SlotConfig{Consumer::Lowering, bridge});
  resources = &*lane->resources;
}
// At the end of the CTA's lane phase: the lane state is released too (ClearLane), so that no
// worker Impl holds a lane's results, or buffers sized by its largest lane, past the phase.
void ModelLowerer::Impl::EndLaneWork() {
  poll = nullptr; // the CTA's LookAhead poll, which ends with it
  resources = nullptr;
  lane->resources.reset();
  matrix = nullptr;
  facts = nullptr;
  tmem = nullptr;
  current_lane.reset();
  ClearLane(true);
}
// Empties a worker Impl's lane state: the LaneWork and every per-CTA map holding one lane's
// entries. `release`: also frees the buffers a next lane would reuse. O(the state's size).
void ModelLowerer::Impl::ClearLane(bool release) {
  addresses = AddressTable{};
  barriers = BarrierTable{};
  sites.clear();
  aligned.clear();
  allocations.clear();
  warp_collectives.clear();
  cluster_arrivals.clear();
  cluster_waits.clear();
  cluster_phases.clear();
  cluster_warp_rendezvous.clear();
  cta_wide_barriers.clear();
  barrier_invalidations.clear();
  shared_value_writes.clear();
  shared_value_reads.clear();
  tensormap_destinations.clear();
  violations.clear();
  limitations.clear();
  dependency_counts.clear();
  plans.clear();
  copy_plans.clear();
  mma_layouts.clear();
  auto &work = *lane;
  for (const auto id : work.counted) occurrence_counts[id] = 0;
  if (release) {
    cluster_waits = {};
    mma_layouts = {};
    lane = std::make_unique<LaneWork>();
    return;
  }
  work.counted.clear();
  work.iteration_starts.clear();
  work.instructions.clear();
  work.prototypes.clear();
  work.operand_kinds.clear();
  work.addressed.clear();
  work.rows.clear();
  work.collectives.clear();
  work.diagnostics.clear();
  work.tape.Clear();
  work.history = {};
  work.lowered = false;
}
// On a worker: lowers `lane_ref`, at `position` in the CTA's lowering order, into fresh lane
// state. Its query events go to the lane's tape and its template values are private (the consumer
// and the other workers read the same facts). A lane that needs the consumer (SerialOnly), fails,
// or could not tape an event is left to the consumer, which lowers it again and so meets the same
// exception there.
void ModelLowerer::Impl::LowerLane(const CtaFacts &cta_facts, LaneRef lane_ref, std::size_t position,
                                   const ActorInfo &actor,
                                   std::span<const std::optional<std::vector<RegId>>> read_slices,
                                   const std::function<void()> &worker_poll) {
  ClearLane(false);
  addresses.KeepWrites();
  auto &work = *lane;
  poll = worker_poll;
  const FactSource::WorkerScope scope(cta_facts, {Consumer::Lowering, 0}, &work.tape);
  const PrivateTemplateValues values;
  current_lane = lane_ref;
  lane_position = position;
  // As the consumer's serial path does: the lane's notes do not depend on which lanes this shadow
  // lowered before (timing).
  resources->BeginLane(lane_ref);
  Thread thread;
  thread.actor = actor;
  std::uint64_t visits = 0;
  try {
    LowerThread(cta_facts, lane_ref, thread, nullptr, read_slices, visits,
                cta_facts.TemplateExecution() ? &work.iteration_starts : nullptr);
    work.history = std::move(thread.history);
    SplitLane();
    work.lowered = work.tape.Complete();
  } catch (...) {
    work.lowered = false;
  }
}
// On a worker, after its lane: the lane's instructions split as SourceInstructions stores them,
// the prototypes interned within the lane, so that Absorb interns each distinct prototype once
// and appends rows. O(instructions x their operands) plus a hash probe each.
void ModelLowerer::Impl::SplitLane() {
  auto &work = *lane;
  std::unordered_map<std::size_t, std::vector<std::uint32_t>> index;
  work.rows.reserve(work.instructions.size());
  for (auto &instruction : work.instructions) {
    SourceRecordCodec::Binding binding;
    SourceRecordCodec::split(instruction, binding);
    auto &candidates = index[SourceRecordCodec::hash(instruction)];
    std::optional<std::uint32_t> prototype;
    for (const auto candidate : candidates)
      if (work.prototypes[candidate] == instruction) {
        prototype = candidate;
        break;
      }
    if (!prototype) {
      prototype = static_cast<std::uint32_t>(work.prototypes.size());
      candidates.push_back(*prototype);
      work.addressed.push_back(std::holds_alternative<ImmediateLoad>(instruction.op) ||
                               std::holds_alternative<ImmediateStore>(instruction.op));
      work.operand_kinds.push_back(OperandKinds(instruction));
      work.prototypes.push_back(std::move(instruction));
    }
    work.rows.emplace_back(*prototype, std::move(binding));
  }
  work.instructions.clear();
}
// The CTA's lanes in `order`, lowered on `threads` workers (LookAhead) and applied here in that
// order. A worker lowers a lane without reading or writing this Impl, so its result does not depend
// on the other lanes; Absorb then performs the lane's writes here as lowering it here would have.
// A lane a worker leaves (or Absorb refuses) is lowered here, at its turn, while the workers go on:
// its lowering writes only this Impl, `r` and this thread's private values (no lane expansion or
// value-view build happens here: lane workers are off while the bridge could build one).
// Look-ahead: at most 2 x threads lanes beyond the one being applied, so their shadows
// (window + 1 Impls, kept across CTAs) bound the memory in flight to that many lanes' results.
void ModelLowerer::Impl::LowerLanes(const CtaFacts &cta_facts, Resources &r, const ValueAddresses &bridge,
                                    std::span<const std::size_t> order, std::span<const ActorInfo> lane_actors,
                                    std::span<const std::optional<std::vector<RegId>>> read_slices,
                                    std::span<const std::uintptr_t> member_cohorts, bool template_execution,
                                    std::uint64_t &visits) {
  const auto window = 2 * threads;
  while (shadows.size() < window + 1) {
    auto shadow = std::make_unique<Impl>(module, function, nullptr);
    shadow->retain_matrix_layout = retain_matrix_layout;
    shadow->atomics = atomics;
    shadow->lane = std::make_unique<LaneWork>();
    shadows.push_back(std::move(shadow));
  }
  struct End {
    std::vector<std::unique_ptr<Impl>> &shadows;
    MatrixResolver &matrix;
    ~End() {
      for (auto &shadow : shadows) shadow->EndLaneWork();
      matrix.EndConcurrent();
    }
  } end{shadows, *matrix};
  matrix->BeginConcurrent(order);
  for (auto &shadow : shadows) shadow->BeginLaneWork(*this, r, bridge);
  // The free shadows; a lane's handle returns its shadow when the lane is applied or dropped.
  std::mutex mutex;
  std::vector<Impl *> free;
  for (auto &shadow : shadows) free.push_back(shadow.get());
  struct Handle {
    Impl *shadow = nullptr;
    std::mutex *mutex = nullptr;
    std::vector<Impl *> *free = nullptr;
    Handle(Impl *s, std::mutex *m, std::vector<Impl *> *f) noexcept : shadow(s), mutex(m), free(f) {}
    Handle(Handle &&other) noexcept
        : shadow(std::exchange(other.shadow, nullptr)), mutex(other.mutex), free(other.free) {}
    Handle &operator=(Handle &&) = delete;
    ~Handle() {
      if (!shadow) return;
      const std::lock_guard lock(*mutex);
      free->push_back(shadow);
    }
  };
  const auto acquire = [&] {
    const std::lock_guard lock(mutex);
    if (free.empty()) throw std::logic_error("lane look-ahead exceeds its shadows");
    auto *shadow = free.back();
    free.pop_back();
    return shadow;
  };
  const PrivateTemplateValues values; // this thread's, while the workers read the same records
  LookAhead<Handle> lanes(order.size(), threads, window,
                          [&](std::size_t index, const std::function<void()> &worker_poll) {
                            Handle handle(acquire(), &mutex, &free);
                            const auto k = order[index];
                            // The lane's end, however it ends, for the matrix rows' waits.
                            struct Finished {
                              MatrixResolver &matrix;
                              std::size_t index;
                              ~Finished() { matrix.Finished(index); }
                            } finished{*matrix, index};
                            handle.shadow->LowerLane(cta_facts, cta_facts.Lane(k), index, lane_actors[k],
                                                     read_slices, worker_poll);
                            return handle;
                          },
                          poll);
  const auto cta = cta_facts.Linear();
  for (std::size_t index = 0; index < order.size(); ++index) {
    const auto handle = lanes.Take();
    const auto k = order[index];
    const auto lane_ref = cta_facts.Lane(k);
    const auto binding_begin = template_execution ? source.program.instructions.beginThread() : 0;
    IterationStarts consumer_starts;
    const IterationStarts *iteration_starts = &handle.shadow->lane->iteration_starts;
    if (!handle.shadow->lane->lowered || !Absorb(*handle.shadow)) {
      current_lane = lane_ref;
      lane_position = index;
      r.BeginLane(lane_ref);
      Thread thread;
      thread.actor = lane_actors[k];
      LowerThread(cta_facts, lane_ref, thread, nullptr, read_slices, visits,
                  template_execution ? &consumer_starts : nullptr);
      histories.push_back(std::move(thread.history));
      iteration_starts = &consumer_starts;
    }
    if (template_execution) {
      source.program.instructions.compressIterations(binding_begin, *iteration_starts);
      source.program.instructions.extendThreadFamily(binding_begin, cta, member_cohorts[k]);
    }
  }
}
// Applies a worker's lowered lane (`shadow`) here: the same writes, in the same order, as lowering
// the lane here, with the lane's ids mapped to ours (LaneIds). The lane's lowering read nothing of
// this Impl, and every input it read is the one lowering here would read: the facts (pure reads),
// the read-only per-CTA state, and its own bindings, whose fields are its own writes (an
// expression binding is written by the step that interns it, and its key names the lane's site)
// or come from its key. So its writes are lowering-here's, up to ids; per map below, merging the
// lane's own entries reproduces the in-order writes (appends append, assignments assign, |= ors,
// first-wins keeps the first). False, writing nothing, if lowering here could differ: an intern
// that would meet another lane's binding with another alias scope (alias_scope_conflict here), or
// the instruction id space. The query events replay in order (EventTape), so a recorder logs the
// one-thread log. O(the lane's instructions + entries) plus the table and builder costs.
bool ModelLowerer::Impl::Absorb(Impl &shadow) {
  auto &work = *shadow.lane;
  const auto base = source.program.instructions.size();
  if (base + work.rows.size() >= UINT32_MAX) return false;
  const auto lane_addresses = addresses.Absorb(shadow.addresses);
  if (!lane_addresses) return false;
  const auto lane_barriers = barriers.Absorb(shadow.barriers);
  // New collectives in the lane's creation order (a collective another lane created first is
  // that lane's).
  std::vector<SyncId> lane_syncs(work.collectives.size());
  {
    std::vector<const decltype(warp_collectives)::key_type *> keys(work.collectives.size());
    for (const auto &[key, id] : shadow.warp_collectives) keys.at(id.value()) = &key;
    for (std::size_t local = 0; local < keys.size(); ++local) {
      const auto [it, added] = warp_collectives.try_emplace(*keys[local], SyncId(source.program.collectives.size()));
      if (added) {
        auto spec = work.collectives[local];
        spec.sync_id = it->second;
        source.program.collectives.push_back(std::move(spec));
      }
      lane_syncs[local] = it->second;
    }
  }
  const LaneIds ids{static_cast<std::uint32_t>(base), *lane_addresses, lane_barriers, lane_syncs};
  // Each distinct prototype with its kept ids mapped, interned once; then each row with its
  // binding's ids mapped: what appending each mapped instruction would store.
  std::vector<std::uint32_t> prototypes;
  prototypes.reserve(work.prototypes.size());
  for (auto &prototype : work.prototypes) {
    ids.ApplyPrototype(prototype.op);
    prototypes.push_back(source.program.instructions.internSplit(std::move(prototype)));
  }
  for (const auto &[prototype, binding] : work.rows)
    source.appendSplit(prototypes[prototype],
                       ids.Apply(binding, work.addressed[prototype], work.operand_kinds[prototype]));
  for (const auto &[key, list] : shadow.sites) {
    auto &to = sites[key];
    for (const auto id : list) to.push_back(ids(id));
  }
  for (const auto &[key, entry] : shadow.aligned) {
    auto &to = aligned[key];
    for (const auto &[lane_id, call] : entry.calls) to.calls[lane_id] = {ids(call.first), ids(call.second)};
    to.failed.insert(entry.failed.begin(), entry.failed.end());
    for (const auto &[lane_id, value] : entry.tmem_addresses) to.tmem_addresses[lane_id] = value;
    to.completions |= entry.completions;
    to.merged_arm |= entry.merged_arm;
  }
  for (const auto &[key, id] : shadow.allocations) allocations[key] = ids(id);
  for (const auto &[key, arrivals] : shadow.cluster_arrivals)
    for (const auto &[actor, id] : arrivals) cluster_arrivals[key].emplace(actor, ids(id));
  for (const auto &wait : shadow.cluster_waits) cluster_waits.push_back({wait.key, ids(wait.id), wait.expected});
  for (const auto &[actor, phase] : shadow.cluster_phases) cluster_phases[actor] = phase;
  for (const auto &[key, entries] : shadow.cluster_warp_rendezvous)
    for (const auto &[actor, id] : entries) cluster_warp_rendezvous[key][actor] = ids(id);
  for (const auto &[barrier, use] : shadow.cta_wide_barriers) {
    // A counted use clears the arrivals and ends CTA-wide use; then nothing is appended.
    auto &to = cta_wide_barriers[ids(barrier)];
    if (!use.cta_wide) {
      to.cta_wide = false;
      to.arrivals.clear();
    } else if (to.cta_wide)
      for (const auto &[actor, id] : use.arrivals) to.arrivals.emplace_back(actor, ids(id));
  }
  for (const auto &[barrier, list] : shadow.barrier_invalidations) {
    auto &to = barrier_invalidations[ids(barrier)];
    for (const auto id : list) to.push_back(ids(id));
  }
  for (const auto &[site, refs] : shadow.shared_value_writes) {
    auto &to = shared_value_writes[site];
    for (const auto &ref : refs) to.push_back(ids(ref));
  }
  for (const auto &[site, ref] : shadow.shared_value_reads) shared_value_reads.emplace(site, ids(ref));
  tensormap_destinations.insert(shadow.tensormap_destinations.begin(), shadow.tensormap_destinations.end());
  violations.insert(shadow.violations.begin(), shadow.violations.end());
  for (const auto &[key, text] : shadow.limitations) limitations[key] = text;
  for (const auto &[key, lane_counts] : shadow.dependency_counts) dependency_counts[key].Merge(lane_counts);
  for (auto &[id, plan] : shadow.plans) {
    ids.Apply(plan);
    plans.emplace(ids(InstId(id)).value(), std::move(plan));
  }
  for (auto &[id, plan] : shadow.copy_plans) {
    ids.Apply(plan);
    copy_plans[ids(InstId(id)).value()] = std::move(plan);
  }
  for (auto &layout : shadow.mma_layouts) {
    layout.instruction = ids(layout.instruction);
    mma_layouts.push_back(std::move(layout));
  }
  for (const auto id : work.counted) occurrence_counts[id] += shadow.occurrence_counts[id];
  for (auto &d : work.diagnostics)
    AddDiagnostic(std::move(d.code), std::move(d.message), d.actor, d.line, std::move(d.details), d.weight);
  work.tape.Replay(*facts, *lane_addresses);
  ids.Apply(work.history);
  histories.push_back(std::move(work.history));
  return true;
}
} // namespace sentryward::frontend::ptx
