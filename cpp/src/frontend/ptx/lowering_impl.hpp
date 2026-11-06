#pragma once
#include "sentryward/algorithm/diagnostics/development_trace.h"
#include "sentryward/frontend/ptx/lowering.h"
#include "sentryward/algorithm/diagnostics/performance_profile.h"
#include "sentryward/algorithm/source_ir/builder.h"
#include "sentryward/algorithm/base/affine_index.h"
#include "sentryward/algorithm/checker/unit_quotient.h"
#include "sentryward/frontend/ptx/dependency_history.h"
#include "sentryward/frontend/ptx/group2_lifetime.h"
#include "sentryward/frontend/ptx/layouts.h"
#include "sentryward/frontend/ptx/look_ahead.h"
#include "sentryward/frontend/ptx/matrix.h"
#include "sentryward/frontend/ptx/refinement.h"
#include "sentryward/frontend/ptx/resources.h"
#include "sentryward/frontend/ptx/tmem_lifetime.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <mutex>
#include <numeric>
#include <regex>
#include <set>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <unordered_map>
namespace sentryward::frontend::ptx {
// The instructions whose lowering registers each lane at its .aligned site (aligned[...].calls):
// Collective (barrier.cluster), Tcgen (wait_load/wait_store, load/store), Lifetime, the
// tensormap.cp_fenceproxy copy of Instruction and Matrix.
bool RegistersAlignedLanes(const Instruction &i) {
  const auto &s = i.semantics;
  const auto &operation = s.operation;
  if (operation == "cluster_arrive" || operation == "cluster_wait") return Has(i, "aligned");
  if (s.category == "tmem_lifetime") return true;
  if (i.opcode == "tcgen05")
    return operation == "wait_load" || operation == "wait_store" || operation == "load" || operation == "store";
  if (i.opcode == "tensormap") return s.modeled && operation != "tensor_map_replace";
  return s.category == "memory" && i.address_lanes.has_value();
}
using Json = nlohmann::json;
namespace {
// An ordinary load or store: lowered by its address operand's Memory slot alone (Instruction's
// immediate memory path).
bool OrdinaryMemory(const Instruction& i) {
  const auto& s=i.semantics;
  return s.category=="memory" && s.modeled && !s.atomic_add && (s.memory_mode=="read" || s.memory_mode=="write") &&
         !i.address_lanes && s.memory_space!="param" && s.memory_space!="const";
}
// The state spaces a bidirectional generic-async proxy fence orders: fence.proxy.async (all
// spaces), .shared::cta and .shared::cluster (shared) or .global; membar.proxy is a synonym.
// Other fences, and the unidirectional fence.proxy.async::generic forms, order no proxies here.
std::uint8_t ProxyFenceSpaces(const Instruction& i) {
  const auto& m=i.modifiers;
  if ((i.opcode!="fence" && i.opcode!="membar") || m.size()<2 || m[0]!="proxy" || m[1]!="async") return 0;
  if (m.size()==2) return kProxyFenceShared | kProxyFenceGlobal;
  if (m.size()==3 && (m[2]=="shared::cta" || m[2]=="shared::cluster")) return kProxyFenceShared;
  if (m.size()==3 && m[2]=="global") return kProxyFenceGlobal;
  return 0;
}
bool IsWarpDataCollective(const Instruction& instruction) {
  return Has(instruction,"sync") &&
      (instruction.opcode=="elect" || instruction.opcode=="shfl" ||
       instruction.opcode=="vote" || instruction.opcode=="match" ||
       instruction.opcode=="redux");
}
struct AddressBindingCodec {
  using Binding=AddressId;
  using Delta=std::int64_t;
  static bool split(AddressId& value,Binding& binding){binding=value;value=AddressId{};return true;}
  static void bind(AddressId& value,Binding binding){value=binding;}
  static std::size_t hash(AddressId){return 0;}
  static std::optional<Delta> difference(Binding a,Binding b){return std::int64_t(b.value())-a.value();}
  static std::optional<Binding> translate(Binding base,Delta delta,std::size_t index) {
    auto result=SourceRecordCodec::translate({InstId{},ActorId{},base,0},{0,0,delta,0},index);
    if(!result)return {};return result->address;
  }
};
using AddressBindings=TemplateRecords<AddressId,AddressBindingCodec>;
MilestoneRef Ref(InstId id, Milestone m = Milestone::kComplete,
                 std::uint32_t e = 0) {
  return {id, m, e};
}
void UniquePush(auto &v, const auto &x) {
  if (std::find(v.begin(), v.end(), x) == v.end())
    v.push_back(x);
}
// One lane's MMA operand reads of one (tcgen group, space, allocation or alias
// scope), in issue order, with the sized ones indexed by allocation and start
// offset: a later write visits the reads that can overlap it (and, for an alias
// scope, those of the other allocations) instead of every earlier read.
// Candidates come back in issue order, so the links a caller adds are those of
// the full scan from index `from` on, in the same order. Add O(log n);
// Candidates O(offsets in the window × log n + result).
struct MmaOperandReads {
  using Read = std::tuple<MilestoneRef, ConstantAddressBinding, std::optional<std::uint64_t>>;
  std::vector<Read> all;
  struct Placed {
    // start offset -> indices into `all`, ascending (appended in issue order)
    std::map<std::uint64_t, std::vector<std::size_t>> by_offset;
    std::uint64_t max_size = 0;
  };
  std::map<std::string, Placed> by_allocation;
  void Add(MilestoneRef ref, ConstantAddressBinding binding, std::optional<std::uint64_t> size) {
    if (size) {
      auto &placed = by_allocation[binding.allocation];
      placed.by_offset[binding.offset].push_back(all.size());
      placed.max_size = std::max(placed.max_size, *size);
    }
    all.emplace_back(ref, std::move(binding), size);
  }
  // Sized reads with index >= `from` of `allocation` that may overlap
  // [offset, offset + bytes), plus (`other_allocations`) every such read of
  // another allocation, in issue order.
  std::vector<std::size_t> Candidates(const std::string &allocation, std::uint64_t offset,
                                      std::uint64_t bytes, bool other_allocations,
                                      std::size_t from = 0) const {
    std::vector<std::size_t> result;
    const auto take = [&](const std::vector<std::size_t> &indices) {
      result.insert(result.end(), std::lower_bound(indices.begin(), indices.end(), from), indices.end());
    };
    for (const auto &[name, placed] : by_allocation) {
      if (name != allocation) {
        if (other_allocations)
          for (const auto &[start, indices] : placed.by_offset) { (void)start; take(indices); }
        continue;
      }
      const auto lower = offset > placed.max_size ? offset - placed.max_size : 0;
      const auto upper = offset > UINT64_MAX - bytes ? UINT64_MAX : offset + bytes;
      for (auto it = placed.by_offset.lower_bound(lower); it != placed.by_offset.end() && it->first <= upper; ++it)
        take(it->second);
    }
    std::sort(result.begin(), result.end());
    return result;
  }
};
std::vector<std::string> Brackets(const std::vector<std::string> &v) {
  std::vector<std::string> r;
  for (const auto &x : v)
    if (x.starts_with('[') && x.ends_with(']'))
      r.push_back(x);
  return r;
}
// cp.async.bulk's modifiers (bulk_complete_tx, bulk_copy): each one of `allowed`, at most once.
// Returns the state-space modifiers among them, in order.
std::vector<std::string> BulkSpaces(const std::vector<std::string> &modifiers, const std::set<std::string> &allowed) {
  std::set<std::string> seen;
  std::vector<std::string> spaces;
  for (const auto &m : modifiers) {
    if (!allowed.contains(m) || !seen.insert(m).second)
      throw SemanticError("bulk_copy_variant", "unsupported bulk copy modifier");
    if (m == "shared::cta" || m == "shared::cluster" || m == "global") spaces.push_back(m);
  }
  return spaces;
}
Json AccessContract(const Json &c, const char *resource, const char *region) {
  Json r = Json::object();
  if (c.contains(resource))
    r["resource"] = c[resource];
  if (c.contains(region))
    r["region"] = c[region];
  return r;
}
struct Thread {
  ActorInfo actor;
  std::map<DynamicKey, InstId> origins;
  std::map<std::string, MilestoneRef> writes;
  std::map<std::string, std::vector<MilestoneRef>> reads;
  CommittedGroups classic, bulk;
  PendingCompletions loads, stores;
  CompletionPrefix classic_notification;
  std::map<int, CompletionPrefix> tcgen_notifications;
  // tcgen05 pipelines (PTX ISA, pipelined tcgen05 instructions): from the same
  // warp, ld -> st, st -> ld and st -> st access TMEM in issue order. Tracked
  // per thread until the matching tcgen05.wait, which orders everything anyway.
  std::vector<InstId> tmem_loads_since_store;
  std::optional<InstId> tmem_last_store;
  // (branch, join) instruction ids of the merged one-armed diamond this lane is in.
  std::optional<std::pair<std::uint32_t, std::uint32_t>> merged_arm;
  TcgenHistory history;
  std::map<std::string, InstId> pipeline;
  std::map<std::tuple<unsigned,std::string,std::string>, MmaOperandReads> mma_operand_reads;
  // (pipeline, D fragment address, bytes) -> (the MMA that last wrote it, the size
  // of its operand-read list when that MMA was linked). See Mma.
  std::map<std::tuple<std::string, std::uint32_t, std::uint64_t>, std::pair<InstId, std::size_t>>
      operand_watermarks;
  struct CopyWrite {
    MilestoneRef effect;
    ConstantAddressBinding binding;
    std::uint64_t bytes, operand_address;
    bool any_operand_address;
  };
  std::map<std::tuple<unsigned,std::string,std::string>, std::vector<CopyWrite>> copy_writes;
};
using Site = std::tuple<std::uint32_t, std::uint32_t, std::uint64_t>;
Site Key(std::uint32_t warp, const RecordView &e) {
  return {warp, e.instruction_id(), e.occurrence()};
}
Site Key(const TmemAtomSite &s) {
  return {s.warp, s.instruction_index, s.occurrence};
}
// The TMEM lifetime issue that stops lowering: a reachable undefined behavior first
// (LoweringUndefinedBehavior; the issues are ordered by code, so an unsupported one could
// otherwise hide it), else the first issue other than tmem_multiple_dealloc (reported as a
// lifecycle violation). O(issues).
const TmemLifetimeIssue *BlockingTmemIssue(const std::vector<TmemLifetimeIssue> &issues) {
  for (const auto &issue : issues)
    if (LoweringUndefinedBehavior(issue.code)) return &issue;
  for (const auto &issue : issues)
    if (issue.code != "tmem_multiple_dealloc") return &issue;
  return nullptr;
}
// Throws a blocking TMEM issue; a reachable undefined behavior names its own site (the
// instruction that violates the rule, which need not be the one being lowered). A static
// issue's site has a PTX line only, no executed warp or occurrence.
[[noreturn]] void ThrowTmemIssue(const TmemLifetimeIssue &issue) {
  if (!LoweringUndefinedBehavior(issue.code)) throw SemanticError(issue.code, issue.message);
  Json details{{"violation_ptx_line", issue.site.ptx_line}};
  if (!issue.static_site) {
    details["violation_warp"] = issue.site.warp;
    details["violation_occurrence"] = issue.site.occurrence;
  }
  throw SemanticError(issue.code, issue.message, std::move(details));
}
struct Aligned {
  std::map<std::uint32_t, std::pair<InstId, InstId>> calls;
  // Lanes that executed this occurrence but whose lowering of it failed (their own diagnostic
  // reports why): they did execute the instruction, so they are not missing from it.
  std::set<std::uint32_t> failed;
  // tcgen05.ld/st: each lane's resolved taddr (allocation resource, encoded address).
  std::map<std::uint32_t, std::pair<std::string, std::uint64_t>> tmem_addresses;
  bool completions = false;
  // Executed inside a merged arm: lanes still must all participate, but the site does
  // not publish memory across lanes (a skipping execution lacks the rendezvous).
  bool merged_arm = false;
};
using ClusterKey = std::tuple<Coordinates, Coordinates, std::uint32_t>;
struct ClusterWait {
  ClusterKey key;
  InstId id;
  std::uint64_t expected;
};
struct TmemAddress {
  std::uint64_t address;
  std::string resource;
  std::optional<std::uint32_t> columns;
};
// A worker's lane in the consumer's ids (ModelLowerer::Impl::Absorb): the ids derived from
// instruction ids (InstId, AsyncOpId, StateId) shift by the lane's first instruction id;
// addresses, barriers and collectives map through the absorbed tables. O(1) per id.
struct LaneIds {
  std::uint32_t base = 0;
  std::span<const AddressId> addresses;
  std::span<const BarrierLifetimeId> barriers;
  std::span<const SyncId> syncs;
  InstId operator()(InstId id) const { return InstId(base + id.value()); }
  AddressId operator()(AddressId id) const { return addresses[id.value()]; }
  BarrierLifetimeId operator()(BarrierLifetimeId id) const { return barriers[id.value()]; }
  MilestoneRef operator()(MilestoneRef ref) const {
    ref.instruction = (*this)(ref.instruction);
    return ref;
  }
  void Apply(std::vector<MilestoneRef> &refs) const {
    for (auto &ref : refs) ref = (*this)(ref);
  }
  void Apply(std::vector<std::pair<std::uint32_t, MilestoneRef>> &links) const {
    for (auto &link : links) link.second = (*this)(link.second);
  }
  void Apply(AsyncEffect &effect) const {
    if (std::any_of(effect.accesses.begin(), effect.accesses.end(),
                    [](const MemoryEffect &access) { return std::holds_alternative<AddressId>(access.endpoint); }))
      effect.accesses = effect.accesses.transformed([&](std::size_t, MemoryEffect access) {
        if (auto *address = std::get_if<AddressId>(&access.endpoint)) *address = (*this)(*address);
        return access;
      });
    if (effect.completion)
      std::visit([&](auto &action) { action.barrier = (*this)(action.barrier); }, *effect.completion);
    Apply(effect.predecessors);
  }
  void Apply(std::vector<AsyncEffect> &effects) const {
    for (auto &effect : effects) Apply(effect);
  }
  // The ids a split prototype keeps (SourceRecordCodec::split moves every other id into the
  // binding, a collective's SyncId included): barriers and completion barriers.
  void ApplyPrototype(SourceOp &op) const {
    std::visit([&](auto &o) {
      using T = std::decay_t<decltype(o)>;
      if constexpr (std::is_same_v<T, AsyncLoad> || std::is_same_v<T, AsyncStore>) {
        if (o.completion) o.completion->barrier = (*this)(o.completion->barrier);
      } else if constexpr (std::is_same_v<T, AsyncOp>) {
        for (auto &effect : o.effects)
          if (effect.completion)
            std::visit([&](auto &action) { action.barrier = (*this)(action.barrier); }, *effect.completion);
      } else if constexpr (requires { o.barrier; }) { // the mbarrier and named-barrier operations
        o.barrier = (*this)(o.barrier);
      }
    }, op);
  }
  // A split binding's ids: the instruction id; the address of a load or store prototype
  // (`addressed`); each operand by its kind (SplitLane), in split's visiting order.
  SourceRecordCodec::Binding Apply(const SourceRecordCodec::Binding &binding, bool addressed,
                                   const std::vector<std::uint8_t> &kinds) const;
  void Apply(MmaPlan &plan) const {
    Apply(plan.effects);
    Apply(plan.links);
    Apply(plan.operand_links);
    Apply(plan.copy_links);
    Apply(plan.contained_copy_links);
  }
  void Apply(TcgenCopyPlan &plan) const {
    Apply(plan.effects);
    Apply(plan.operand_links);
  }
  void Apply(TcgenHistory &history) const {
    for (auto &record : history.records) record.id += base;
    for (auto *ids : {&history.work, &history.commits})
      for (auto &[group, list] : *ids) {
        (void)group;
        for (auto &id : list) id += base;
      }
  }
};
// The kinds of a split prototype's binding operands, in split's visiting order: an id derived
// from an instruction id (InstId, AsyncOpId, StateId), an address, an actor, or a collective.
enum OperandKind : std::uint8_t { kInstructionOperand, kAddressOperand, kActorOperand, kSyncOperand };
std::vector<std::uint8_t> OperandKinds(SourceInstruction &prototype) {
  std::vector<std::uint8_t> kinds;
  SourceRecordCodec::visitOperands(prototype.op, [&](auto &id) {
    using T = std::decay_t<decltype(id)>;
    kinds.push_back(std::is_same_v<T, AddressId> ? kAddressOperand
                    : std::is_same_v<T, ActorId> ? kActorOperand
                    : std::is_same_v<T, SyncId> ? kSyncOperand : kInstructionOperand);
  });
  return kinds;
}
SourceRecordCodec::Binding LaneIds::Apply(const SourceRecordCodec::Binding &binding, bool addressed,
                                          const std::vector<std::uint8_t> &kinds) const {
  if (binding.operands.size() != kinds.size()) throw std::logic_error("split lane binding width");
  auto result = binding;
  result.id = (*this)(binding.id);
  if (addressed) result.address = (*this)(binding.address);
  if (!kinds.empty())
    result.operands = binding.operands.transformed([&](std::size_t index, std::uint32_t value) {
      switch (kinds[index]) {
      case kInstructionOperand: return base + value;
      case kAddressOperand: return addresses[value].value();
      case kSyncOperand: return syncs[value].value();
      default: return value;
      }
    });
  return result;
}
// The MemoHit/MemoFill event of the evaluated-address memo, out of line and cold so
// that the hot generated-memory loop keeps only the recording() branch (O(1) per
// hit/fill).
[[gnu::noinline, gnu::cold]] void NoteEvaluatedMemo(const CtaFacts &facts, bool hit, RecordRef at,
                                                    std::uint64_t key, RegId reg) {
  const MemoRef memo{at, key};
  if (hit) facts.Note(query::MemoHit{memo, reg});
  else facts.Note(query::MemoFill{memo, reg});
}
} // namespace
struct ModelLowerer::Impl {
  bool retain_matrix_layout = false;
  bool atomics = false;  // order_independent_atomics
  std::size_t generated_memory_instances=0,generated_address_rows=0,generated_body_rows=0;
  std::vector<MmaValueLayout> mma_layouts;
  const Module &module;
  const Function &function;
  std::function<void()> poll;
  SourceProgramBuilder source;
  AddressTable addresses;
  BarrierTable barriers;
  std::vector<RequiredOrder> orders;
  std::vector<SharedValueDependency> shared_value_dependencies;
  // Statistics only (CtaFacts::SharedBoundedCounts, summed over the lowered CTAs).
  std::uint64_t shared_bounded_writes = 0, shared_bounded_reads = 0;
  std::map<SharedValueSite,std::vector<MilestoneRef>> shared_value_writes;
  std::map<SharedValueSite,MilestoneRef> shared_value_reads;
  std::set<SharedValueSite> shared_value_sites;
  std::vector<bool> shared_value_instructions;
  // Reads of runtime-written tensor maps (RecoverTensorMaps): each TMA's descriptor read and the
  // tensormap.cp_fenceproxy it reads from (write site: lane 0 of the copying warp). Their endpoints
  // are kept in shared_value_writes (the copy's 32 lane writes) and shared_value_reads.
  std::vector<SharedValueDependency> tensor_map_dependencies;
  std::set<SharedValueSite> tensor_map_copies;
  Coordinates current_coordinates{};
  std::map<std::string, std::string> limitations;
  // Every consumed CTA's tensormap.cp_fenceproxy destinations (object, region base), checked
  // against input tensor maps in run.cpp on the per-unit path.
  std::set<std::pair<std::string, std::uint64_t>> tensormap_destinations;
  std::set<std::pair<std::string, std::string>> violations;
  std::map<std::uint32_t, MmaPlan> plans;
  std::map<std::uint32_t,TcgenCopyPlan> copy_plans;
  std::vector<TcgenHistory> histories;
  // Per merged branch: does its arm wait for loads, and which registers does it read
  // before writing them (upward-exposed reads).
  struct MergedArmFacts {
    bool wait_load = false;
    std::vector<std::string> exposed;
  };
  std::map<std::uint32_t, MergedArmFacts> merged_arm_facts;
  // st.local shapes (execution.h), built when a skipped region first holds one.
  std::unique_ptr<LocalRegionStores> local_region_stores;
  std::map<std::string, DependencyCounts> dependency_counts;
  std::vector<std::vector<std::string>> operands;
  std::vector<std::string> opcode_texts;
  SharedString source_name;
  std::map<std::uint32_t, SharedString> debug_files;
  std::vector<SharedString> origin_opcodes;
  std::vector<std::uint64_t> occurrence_counts;
  RegisterTable registers;
  Json diagnostics = Json::array();
  std::map<std::string,std::size_t> diagnostic_groups;
  std::map<std::string, std::uint64_t> errors, counts;
  std::map<ClusterKey, std::map<ActorId, InstId>> cluster_arrivals;
  std::vector<ClusterWait> cluster_waits;
  // The current CTA's named barriers (CtaWideExits): whether every use is a blocking one
  // without a thread count, and then its arrivals in lowering order, which lists each
  // thread's arrivals contiguously and in program order.
  struct CtaWideBarrier {
    bool cta_wide = true;
    std::vector<std::pair<ActorId, InstId>> arrivals;
  };
  std::map<BarrierLifetimeId, CtaWideBarrier> cta_wide_barriers;
  // Per thread of the current CTA, the await standing for its exit (Terminal).
  std::map<ActorId, InstId> terminals;
  // Distributed shared memory lifetimes (CtaExitJoin; cluster launches only): each lowered CTA's
  // coordinates by linear index, the allocation of its shared memory, and the join of its
  // threads' exits (absent when no thread ends at ret/exit).
  bool cluster_launch = false;
  std::map<std::uint32_t, Coordinates> cluster_ctas;
  std::unordered_map<std::string, Coordinates> shared_owners;
  std::map<Coordinates, InstId> cta_exits;
  std::set<Coordinates> clc_requests;
  std::map<ActorId, std::uint32_t> cluster_phases;
  std::map<BarrierLifetimeId,std::vector<InstId>> barrier_invalidations;
  std::map<Coordinates, std::vector<Group2LifetimeSite>> group2_sites;
  std::uint32_t ctas = 0;
  std::set<std::string> ambiguous_registers;
  // CTA-local indexes are cleared as soon as joins and lifetime orders are
  // built.
  std::map<Site, std::vector<InstId>> sites;
  bool index_sites = true;
  std::map<Site, Aligned> aligned;
  std::map<std::tuple<std::uint32_t,std::string,std::uint32_t>,
           std::map<ActorId,InstId>> cluster_warp_rendezvous;
  std::map<Site, InstId> allocations;
  std::map<
      std::tuple<std::uint32_t, std::uint32_t, std::uint64_t, std::uint32_t>,
      SyncId>
      warp_collectives;
  std::map<std::pair<std::uint32_t, std::uint32_t>, ActorId> actors;
  // Cross-CTA class mode (TrackUnits): per representative CTA its occurrence counts (sparse),
  // the CTA-free part of its rendered violations, and the CTAs with a lowering diagnostic (their
  // classes never admit, so their weight must be 1).
  algorithm::UnitQuotient *units = nullptr;
  std::vector<std::pair<std::uint32_t, std::vector<std::pair<std::uint32_t, std::uint64_t>>>> cta_occurrences;
  std::vector<std::tuple<std::uint32_t, std::string, std::string>> cta_violations; // (cta, code, suffix)
  std::set<std::uint32_t> diagnostic_ctas;
  const CtaFacts *facts = nullptr;
  Resources *resources = nullptr;
  TmemLifetimeAnalysis *tmem = nullptr;
  MatrixResolver *matrix = nullptr;
  std::optional<LaneRef> current_lane; // the lane being lowered (Add)
  std::size_t lane_position = 0;       // its position in the CTA's lowering order (MatrixResolver)
  // A CTA's lanes on worker threads (LowerLanes): the --sync-threads count; at most one lowers
  // every lane on the calling thread.
  std::size_t threads = 0;
  // A thread's loop iterations, in order: (position of the first instruction lowered from the
  // iteration, relative to the thread's first one; the loop header's PTX instruction).
  using IterationStarts = std::vector<std::pair<std::size_t, std::uint32_t>>;
  // A worker's Impl (`lane` set) lowers one lane of the consumer's current CTA into state of its
  // own: instruction ids from 0 in `instructions`, its own address and barrier tables (which keep
  // their writes) and collectives, and every per-CTA map of the Impl holding only that lane's
  // entries. The consumer's Absorb applies that state in the CTA's lowering order.
  struct LaneWork {
    std::vector<SourceInstruction> instructions; // emptied by SplitLane into the four below
    // The instructions as SourceInstructions stores them (SourceRecordCodec::split): the lane's
    // distinct prototypes with their binding operands' kinds (OperandKinds) and whether a load or
    // store holds the binding's address, and one (prototype, binding) row per instruction.
    std::vector<SourceInstruction> prototypes;
    std::vector<std::vector<std::uint8_t>> operand_kinds;
    std::vector<bool> addressed;
    std::vector<std::pair<std::uint32_t, SourceRecordCodec::Binding>> rows;
    std::vector<CollectiveSpec> collectives; // SyncIds from 0, in creation order
    struct Diagnosed {
      std::string code, message;
      std::optional<std::uint32_t> actor, line;
      Json details;
      std::uint64_t weight;
    };
    std::vector<Diagnosed> diagnostics; // AddDiagnostic's arguments, in call order
    EventTape tape;                     // the lane's query events while a recorder is attached
    std::optional<Resources> resources; // per CTA, over the Impl's tables
    std::vector<std::uint32_t> counted; // the instructions whose occurrence_counts entry is nonzero
    IterationStarts iteration_starts;   // with template execution
    TcgenHistory history;
    bool lowered = false; // false: the consumer lowers the lane itself
  };
  std::unique_ptr<LaneWork> lane;
  std::vector<std::unique_ptr<Impl>> shadows; // the consumer's worker Impls
  Impl(const Module &m, const Function &f, std::function<void()> p)
      : module(m), function(f), poll(std::move(p)), source_name(m.source_name), registers(m, f) {
    for (const auto& [id, path] : m.file_table) debug_files.emplace(id, path);
    for (const auto &i : f.instructions) {
      operands.push_back(OperandTexts(m, i));
      opcode_texts.push_back(Opcode(i));
      origin_opcodes.emplace_back(opcode_texts.back());
    }
    occurrence_counts.resize(f.instructions.size());
    std::set<std::string> names;
    static const std::regex declaration(
        R"(\.reg\s+\.(?:pred|(?:[busf]|bf)\d+)\s+([^;]+))"),
        reg(R"((%?[A-Za-z_$][A-Za-z0-9_$]*)(?:<(\d+)>)?)");
    for (const auto &directive : f.directives) {
      if (directive.name != ".reg")
        continue;
      std::string text(m.view(directive.source));
      std::smatch match;
      if (!std::regex_search(text, match, declaration))
        continue;
      auto list = match[1].str();
      for (std::sregex_iterator it(list.begin(), list.end(), reg), end;
           it != end; ++it) {
        auto name = (*it)[1].str();
        auto extent = (*it)[2].matched ? std::stoul((*it)[2]) : 1;
        for (std::size_t k = 0; k < extent; ++k) {
          auto expanded = (*it)[2].matched ? name + std::to_string(k) : name;
          if (!names.insert(expanded).second)
            ambiguous_registers.insert(expanded);
        }
      }
    }
  }
  void Poll() {
    if (poll)
      poll();
  }
  // `weight` (class mode, Finish only): the class size of the unit the occurrence belongs to.
  void Diagnostic(std::string code, std::string message,
                  const Thread *t = nullptr,
                  const RecordView *e = nullptr, Json details = Json::object(), std::uint64_t weight = 1) {
    // A ReportBearing event: per-CTA report evidence (closes the recording class); never after Add
    // (facts is null).
    if (facts && facts->recording()) [[unlikely]]
      facts->Note(query::ReportBearing{query::ReportKind::LoweringDiagnostic, code});
    std::optional<std::uint32_t> actor, line;
    if (t) actor = t->actor.id.value();
    if (e) line = function.instructions[e->instruction_id()].line;
    if (lane) {
      lane->diagnostics.push_back({std::move(code), std::move(message), actor, line, std::move(details), weight});
      return;
    }
    AddDiagnostic(std::move(code), std::move(message), actor, line, std::move(details), weight);
  }
  void AddDiagnostic(std::string code, std::string message, std::optional<std::uint32_t> actor,
                     std::optional<std::uint32_t> line, Json details, std::uint64_t weight) {
    errors[code] += weight;
    if (units && facts) diagnostic_ctas.insert(facts->Linear());
    Json d{{"code",code},{"message",message},{"stage","event_view_construction"},{"details",std::move(details)}};
    if(line)d["details"]["ptx_line"]=*line;
    const auto key=d.dump();
    const auto [it,added]=diagnostic_groups.try_emplace(key,diagnostics.size());
    if(added) {
      d["details"]["occurrence_count"]=0;
      d["details"]["actor_ranges"]=Json::array();
      diagnostics.push_back(std::move(d));
    }
    auto& evidence=diagnostics[it->second]["details"];
    evidence["occurrence_count"]=evidence["occurrence_count"].get<std::size_t>()+weight;
    if(actor) {
      auto& ranges=evidence["actor_ranges"];
      if(!ranges.empty() && ranges.back()[1].get<std::uint32_t>()+1==*actor)ranges.back()[1]=*actor;
      else if(ranges.empty() || ranges.back()[1]!=*actor)ranges.push_back({*actor,*actor});
    }
  }

  SourceLocation Location(const Instruction &i) {
    if (i.debug_location) {
      auto found = debug_files.find(i.debug_location->file);
      if (found != debug_files.end())
        return {found->second, i.debug_location->line,
                i.debug_location->column};
    }
    return {source_name, i.line, i.column};
  }
  InstId Next() const {
    const auto size = lane ? lane->instructions.size() : source.program.instructions.size();
    if (size >= UINT32_MAX)
      throw SemanticError("model_size", "source instruction IDs exhausted");
    return InstId(size);
  }
  // An instruction this Impl emitted (a worker's: by its lane-local id).
  SourceInstruction Emitted(InstId id) const {
    return lane ? lane->instructions.at(id.value()) : source.instruction(id);
  }
  void SharedValueEndpoint(const ActorInfo &actor,const DynamicKey &key,InstId id,const SourceOp &op) {
    if(shared_value_instructions.empty() || !shared_value_instructions.at(key.instruction))return;
    const SharedValueSite site{current_coordinates,actor.warp.value()*32+actor.lane,key};
    if(!shared_value_sites.contains(site))return;
    if(std::holds_alternative<ImmediateLoad>(op))shared_value_reads.emplace(site,Ref(id));
    if(std::holds_alternative<ImmediateStore>(op))shared_value_writes[site].push_back(Ref(id));
    if(const auto *async=std::get_if<AsyncOp>(&op))
      for(std::uint32_t index=0;index<async->effects.size();++index)
        for(const auto &access:async->effects[index].accesses)
          if(const auto *address=std::get_if<AddressId>(&access.endpoint);
              address && access.mode==AccessMode::kWrite && access.bytes &&
              resources->Binding(*address).address_space=="shared") {
            UniquePush(shared_value_writes[site],Ref(id,Milestone::kEffect,index));break;
          }
  }
  InstId Emit(Thread &t, const RecordView &e, SourceOp op) {
    const auto &i = function.instructions[e.instruction_id()];
    PtxOrigin origin{source_name, i.id, static_cast<std::uint32_t>(e.occurrence()), origin_opcodes[i.id]};
    InstId id;
    if (lane) {
      id = Next();
      lane->instructions.push_back({id, t.actor.id, Location(i), std::move(op), std::move(origin)});
    } else
      id = source.append(t.actor.id, Location(i), std::move(op), std::move(origin));
    if(!shared_value_instructions.empty() && shared_value_instructions.at(e.instruction_id()))
      SharedValueEndpoint(t.actor,e.Key(),id,Emitted(id).op);
    if (index_sites) {
      t.origins.try_emplace(e.Key(), id);
      sites[Key(t.actor.warp.value(), e)].push_back(id);
    }
    return id;
  }
  InstId Async(Thread &t, const RecordView &e,
               std::vector<AsyncEffect> effects) {
    return Emit(t, e, AsyncOp{AsyncOpId(Next().value()), std::move(effects)});
  }
  std::uint64_t Scalar(const RecordView &e, std::string_view s,
                       std::string_view field, std::uint64_t max = UINT64_MAX) {
    return Unsigned(resources->ScalarValue(e, s, field), field, max);
  }
  // cp.async.bulk's size operand. PTX ISA cp.async.bulk: "size must be a multiple of 16. If
  // the value is not a multiple of 16, then the behavior is undefined" (a known size that is
  // not one is undefined behavior, LoweringUndefinedBehavior). A zero-byte transfer is a
  // multiple of 16 but is not modeled.
  std::uint64_t BulkSize(const RecordView &e, std::string_view operand) {
    const auto size = Scalar(e, operand, "bulk transfer size", UINT32_MAX);
    if (size % 16)
      throw SemanticError("invalid_bulk_transfer_size", "bulk transfer size is not a multiple of 16 (undefined behavior)",
                          {{"size", size}, {"isa_section", "9.7.10.28.4.1 cp.async.bulk"}});
    if (!size) throw SemanticError("bulk_transfer_size_zero", "a zero-byte bulk transfer is not modeled");
    return size;
  }
  // PTX ISA cp.async.bulk: "The addresses dstMem and srcMem must be aligned to 16 bytes." A
  // constant shared binding's offset is its address in the shared window; its misalignment
  // under some placement that the shared objects' declared alignments permit is decided
  // (undefined behavior, with that placement: FrontendConfig::SharedMisalignment). A global
  // address is relative to an input object whose base alignment is not given, and a symbolic
  // one is not decided. O(shared objects).
  void RequireBulkCopyAlignment(const std::pair<Fragment, Fragment> &footprints) {
    for (const auto *fragment : {&footprints.first, &footprints.second}) {
      const auto binding = resources->Binding(fragment->first);
      if (!binding.expression.empty() || binding.address_space != "shared") continue;
      if (auto witness = facts->launch().SharedMisalignment(binding.offset, 16)) {
        Json details{{"address_space", binding.address_space}, {"offset", binding.offset}, {"alignment", 16},
                     {"operand", fragment == &footprints.first ? "srcMem" : "dstMem"},
                     {"isa_section", "9.7.10.28.4.1 cp.async.bulk"}};
        details.update(*witness);
        throw SemanticError("bulk_copy_alignment",
                            "bulk copy shared address is not 16-byte aligned under a legal placement (undefined behavior)",
                            std::move(details));
      }
    }
  }
  // A linear bulk transfer's footprints (copy and reduction): `source` read, then `destination`
  // written, each one fragment of its request's bytes; otherwise SemanticError(code, message).
  std::pair<Fragment, Fragment> BulkFootprints(const RecordView &e, const MemoryRequest &source,
                                               const MemoryRequest &destination, const char *code,
                                               const char *message) {
    const auto read = resources->Memory(e, source);
    const auto write = resources->Memory(e, destination);
    if (read.second != source.bytes || write.second != destination.bytes) throw SemanticError(code, message);
    return {read, write};
  }
  // cp.async.bulk's copy footprints (`spaces`: BulkSpaces): brackets[1] read in spaces[1], then
  // brackets[0] written in spaces[0], under the contract's source and destination entries.
  std::pair<Fragment, Fragment> BulkCopyFootprints(const RecordView &e, const std::vector<std::string> &brackets,
                                                   const Json &contract, std::uint64_t size,
                                                   const std::vector<std::string> &spaces) {
    const auto source = AccessContract(contract, "source_resource", "source_region");
    const auto destination = AccessContract(contract, "destination_resource", "destination_region");
    return BulkFootprints(e, MemoryRequest(brackets[1], source, size, spaces[1]),
                          MemoryRequest(brackets[0], destination, size, spaces[0]), "bulk_copy_fragment_size",
                          "bulk footprints disagree with size");
  }
  // A 64-bit descriptor operand, read as its bit pattern.
  std::uint64_t Descriptor(const RecordView &e, std::string_view s, std::string_view field) {
    return Bits(resources->ScalarValue(e, s, field), field);
  }
  std::string Register(const Instruction &i, const std::string &name) {
    return registers.Resolve(i.scope, name);
  }
  std::vector<std::string> Destinations(const Instruction &i) {
    std::vector<std::string> result;
    for (const auto &r : i.destinations)
      result.push_back(Register(i, r));
    return result;
  }
  std::vector<std::string> Sources(const Instruction &i) {
    std::vector<std::string> result;
    const auto &ops = operands[i.id];
    for (std::size_t k = 0; k < ops.size(); ++k) {
      bool destination = false;
      if (!i.destinations.empty()) {
        if (k == 0 && !(i.opcode == "tcgen05" && Has(i, "ld") && Has(i, "red")))
          destination = true;
        if (k == 1 && i.opcode == "tcgen05" && Has(i, "ld") && Has(i, "red"))
          destination = true;
      }
      if (!destination)
        for (const auto &r : RegistersIn(ops[k]))
          UniquePush(result, Register(i, r));
    }
    if (i.predicate)
      UniquePush(result, Register(i, *i.predicate));
    return result;
  }
  bool MemoryRegisters(Thread &, const RecordView &, AddressId, std::uint32_t, bool);
  void Compute(Thread &, const RecordView &);
  void WarpData(Thread &, const RecordView &);
  void Collective(Thread &, const RecordView &, std::string_view);
  void Mbarrier(Thread &, const RecordView &, const Json &);
  void Tcgen(Thread &, const RecordView &, const Json &);
  void Lifetime(Thread &, const RecordView &);
  std::optional<RecordView> ExitRecord(const CtaFacts &, std::size_t lane_index) const;
  std::optional<InstId> Terminal(const CtaFacts &, std::size_t lane_index, const ActorInfo &);
  void CtaWideExits(const CtaFacts &, std::size_t actor_begin);
  void CtaExitJoin(const CtaFacts &, std::size_t actor_begin);
  void DistributedSharedLifetimes();
  void LifetimeOrders();
  void CollectGroup2();
  void Matrix(Thread &, const RecordView &, bool);
  TmemAddress ResolveTmem(Thread &, const RecordView &,
                          std::string_view);
  void Mma(Thread &, const RecordView &, const Json &);
  InstId Copy(Thread &, const RecordView &, std::vector<Fragment>,
              std::vector<Fragment>,
              std::optional<CompletionAction> completion = {},
              MemoryProxy proxy = MemoryProxy::kGeneric);
  void Instruction(Thread &, const RecordView &);
  void LowerThread(const CtaFacts &, LaneRef, Thread &, const std::vector<std::size_t> *positions,
                   std::span<const std::optional<std::vector<RegId>>> read_slices, std::uint64_t &visits,
                   IterationStarts *iteration_starts = nullptr);
  void LowerLanes(const CtaFacts &, Resources &, const ValueAddresses &, std::span<const std::size_t> order,
                  std::span<const ActorInfo> lane_actors,
                  std::span<const std::optional<std::vector<RegId>>> read_slices,
                  std::span<const std::uintptr_t> member_cohorts, bool template_execution, std::uint64_t &visits);
  void BeginLaneWork(const Impl &consumer, Resources &like, const ValueAddresses &bridge);
  void EndLaneWork();
  void ClearLane(bool release);
  void LowerLane(const CtaFacts &, LaneRef, std::size_t position, const ActorInfo &,
                 std::span<const std::optional<std::vector<RegId>>> read_slices,
                 const std::function<void()> &worker_poll);
  void SplitLane();
  bool Absorb(Impl &shadow);
  void Add(const CtaFacts &, ValueAddresses, bool lane_workers);
  LoweringResult Finish();
};
} // namespace sentryward::frontend::ptx

#include "lowering_registers.hpp"
#include "lowering_matrix.hpp"
#include "lowering_tcgen.hpp"
#include "lowering_lifetimes.hpp"
#include "lowering_instruction.hpp"
#include "lowering_collectives.hpp"
#include "lowering_cta.hpp"
#include "lowering_lanes.hpp"
#include "lowering_finish.hpp"
