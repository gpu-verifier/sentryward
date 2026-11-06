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
  void LifetimeOrders();
  void CollectGroup2();
  void Matrix(Thread &, const RecordView &, bool);
  TmemAddress ResolveTmem(Thread &, const RecordView &,
                          std::string_view);
  void Mma(Thread &, const RecordView &, const Json &);
  InstId Copy(Thread &, const RecordView &, std::vector<Fragment>,
              std::vector<Fragment>,
              std::optional<CompletionAction> completion = {});
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
TmemAddress ModelLowerer::Impl::ResolveTmem(Thread &t,
                                            const RecordView &e,
                                            std::string_view operand) {
  if (!tmem)
    throw SemanticError("unresolved_tmem_pointer",
                        "TMEM provenance unavailable");
  if (const auto *issue = BlockingTmemIssue(tmem->issues))
    ThrowTmemIssue(*issue);
  auto scalar =
      tmem->provenance.Resolve(t.actor.warp.value(), e, operand, t.actor.lane);
  auto pointer = tmem->provenance.NormalizeTaddr(scalar);
  if (pointer) {
    auto allocation = tmem->provenance.AllocationById(pointer->allocation);
    if (!allocation)
      throw SemanticError("unresolved_tmem_pointer", "allocation missing");
    return {pointer->encoded_offset,
            allocation->columns == 512 ? "physical.tmem.cta.0"
                                       : pointer->ResourceName(),
            allocation->columns};
  }
  if (tmem->provenance.allocations.empty())
    if (auto n = std::get_if<Integer>(&scalar))
      return {
          Unsigned(*n, "TMEM address", UINT32_MAX), "physical.tmem.cta.0", {}};
  throw SemanticError("unresolved_tmem_pointer",
                      "TMEM operand lacks concrete allocation provenance");
}
void ModelLowerer::Impl::Mma(Thread &t, const RecordView &e,
                             const Json &contract) {
  (void)contract;
  ProfileScope profile("frontend.model.mma");
  const auto &i = function.instructions[e.instruction_id()];
  const auto &ops = operands[i.id];
  const bool sparse = Has(i, "sp"), scaled = Has(i, "block_scale");
  const bool nvfp4 = scaled && Has(i, "kind::mxf4nvf4");
  const bool mxf4 = scaled && Has(i, "kind::mxf4");
  const bool packed_fp4 = nvfp4 || mxf4;
  // PTX 9.2 .ws form: [d], a, b, idesc, enable-input-d {, zero-column-mask-desc}.
  const bool ws = Has(i, "ws");
  if ((ws ? (ops.size() != 5 && ops.size() != 6) || sparse || scaled || Has(i, "cta_group::2") ||
                ops[4].starts_with('{')
          : sparse ? (ops.size() != 7 && ops.size() != 8) : (ops.size() < 5 || ops.size() > 7)) ||
      Has(i, "ashift") ||
      std::any_of(i.modifiers.begin(), i.modifiers.end(),
                  [](const auto &m) { return m.starts_with("collector::"); }))
    throw SemanticError("tcgen_mma_variant",
                        "MMA variant has additional operand semantics");
  // A zero descriptor clears Non-Zero Mask and Column Shift: every B column is used.
  if (ws && ops.size() == 6 && Descriptor(e, ops[5], "MMA zero-column mask") != 0)
    throw SemanticError("tcgen_mma_variant", "nonzero zero-column mask masks or shifts B columns");
  unsigned group = Has(i, "cta_group::2") ? 2 : 1;
  std::vector<std::string> kinds;
  for (const auto &m : i.modifiers)
    if (m.starts_with("kind::"))
      kinds.push_back(m.substr(6));
  if (kinds.size() != 1)
    throw SemanticError("tcgen_mma_variant", "MMA requires one kind");
  if (scaled && (ops.size() != (sparse ? 8U : 7U) ||
      (mxf4 ? (ops[1].starts_with('[') || Has(i, "block16")) : nvfp4 ? (ops[1].starts_with('[') || Has(i, "block32") ||
                  (!Has(i, "block16") && !Has(i, "scale_vec::4X"))) :
                 (!Has(i, "kind::mxf8f6f4") || Has(i, "block16"))) ||
      std::any_of(i.modifiers.begin(), i.modifiers.end(), [&](const auto &m) {
        return m.starts_with("scale_vec::") && m != (mxf4 ? "scale_vec::2X" : nvfp4 ? "scale_vec::4X" : "scale_vec::1X");
      })))
    throw SemanticError("tcgen_mma_variant", "unsupported block-scaled operands or scale layout");
  const unsigned descriptor_index = sparse ? 4 : 3, mask_index = descriptor_index + 1;
  Resources::BitsResolver scale_bits = [&](const RecordView &record, std::string_view token,
                                           unsigned) -> std::optional<Resources::KnownBits> {
    if (!tmem || !scaled) return {};
    auto value = tmem->provenance.Resolve(t.actor.warp.value(), record, token, t.actor.lane);
    const auto *origin = facts->Origin(record, token);
    const auto *current = facts->Origin(e, token);
    if (std::holds_alternative<std::monostate>(value) && origin && current && *origin == *current)
      value = tmem->provenance.Resolve(t.actor.warp.value(), e, token, t.actor.lane);
    if (auto pointer = std::get_if<TmemPointer>(&value))
      return Resources::KnownBits{pointer->encoded_offset & 0xc0000000U, 0xc0000000U};
    return {};
  };
  const auto layout_mask = UINT32_MAX & ~((1U << 13) | (1U << 14) | (sparse ? 3U : 0U));
  const auto [fixed, known] = resources->FixedBits(e, ops[descriptor_index], 32, scale_bits);
  // PTX ISA instruction descriptor, .kind::mxf4 and .kind::mxf4nvf4: "Bit 12 for sparsity
  // version must map to v0 for architectures sm_100a, sm_103a, sm_110a and to v1 for
  // architecture sm_107a, otherwise behavior is undefined." Decided for those targets.
  if (packed_fp4 && ((known >> 12) & 1)) {
    const auto target = TargetArchitecture(module);
    const std::optional<unsigned> required =
        target == "sm_100a" || target == "sm_103a" || target == "sm_110a" ? std::optional<unsigned>(0)
        : target == "sm_107a" ? std::optional<unsigned>(1) : std::nullopt;
    const unsigned version = static_cast<unsigned>((fixed >> 12) & 1);
    if (required && version != *required)
      throw SemanticError("mma_sparsity_version",
                          "instruction descriptor SparsityVersion (bit 12) is v" + std::to_string(version) +
                              " on target " + target + ", which requires v" + std::to_string(*required) +
                              " (undefined behavior)",
                          {{"target", target}, {"sparsity_version", version}, {"required", *required},
                           {"isa_section", "9.7.18.4.2.1 Instruction descriptor, Target ISA Note"}});
  }
  const auto idesc = (known & layout_mask) == layout_mask ? fixed & layout_mask :
      Scalar(e, ops[descriptor_index], "MMA descriptor");
  auto desc = scaled ? DecodeBlockScaledDescriptor(idesc, kinds[0]) : DecodeMmaDescriptor(kinds[0], idesc);
  if (desc.sparse != sparse)
    throw SemanticError("tcgen_mma_descriptor", "MMA sparsity bit disagrees with opcode");
  if (ws && !WsMmaShape(desc.m, desc.n))
    throw SemanticError("tcgen_mma_variant", ".ws MMA shape is outside the PTX shape list");
  auto footprint_instruction = i;
  auto footprint_idesc = idesc;
  if (scaled) {
    std::erase(footprint_instruction.modifiers, "block_scale");
    for (auto &m : footprint_instruction.modifiers)
      if (m == "kind::mxf8f6f4" || m == "kind::mxf4nvf4" || m == "kind::mxf4") m = "kind::f8f6f4";
    footprint_idesc = (idesc & ~((3ULL << 4) | (1ULL << 23) | (3ULL << 29))) | (1ULL << 4);
    if (packed_fp4) footprint_idesc = (footprint_idesc & ~((7ULL << 7) | (7ULL << 10))) | (5ULL << 7) | (5ULL << 10);
  }
  std::vector<std::uint32_t> mask(4 * group, 0);
  const bool has_mask = !scaled && ops[mask_index].starts_with('{');
  if (has_mask) {
    if (ops.size() <= mask_index + 1 || !ops[mask_index].ends_with('}'))
      throw SemanticError("tcgen_mma_variant", "malformed MMA output mask operands");
    mask.clear();
    std::stringstream words(ops[mask_index].substr(1, ops[mask_index].size() - 2));
    for (std::string word; std::getline(words, word, ',');)
      mask.push_back((resources->ScalarValue(e, Trim(word), "MMA output mask") & Integer(UINT32_MAX)).convert_to<std::uint32_t>());
  }
  bool enabled = Scalar(e, ops[scaled ? descriptor_index + 3 : mask_index + has_mask], "MMA accumulator predicate", 1);
  auto peer_placement = [&](const TmemAddress &address) {
    if (address.columns == 512) return true;
    return tmem && std::any_of(tmem->provenance.zero_based_allocations.begin(),
        tmem->provenance.zero_based_allocations.end(), [&](const auto &id) {
          return id.ResourceName() == address.resource;
        });
  };
  std::vector<std::pair<std::string, std::vector<ByteInterval>>> scales;
  std::vector<TmemAddress> scale_addresses;
  if (scaled)
    for (unsigned operand = 0; operand < 2; ++operand) {
      const auto address = ResolveTmem(t, e, ops[descriptor_index + 1 + operand]);
      auto operand_address = address;
      // Use the same TMEM base as DecodeScaleFootprint, excluding selector bits.
      operand_address.address &= 0x3fffffff;
      scale_addresses.push_back(operand_address);
      if (group == 2 && address.columns != 512)
        throw SemanticError("tmem_group2_relative_peer_base", "scale peers require full allocation placement");
      scales.push_back({address.resource, DecodeScaleFootprint(address.address,
          (idesc >> (operand ? 4 : 29)) & 3, operand ? desc.n : desc.m / group,
          mxf4 ? 2 : nvfp4 ? 4 : 1, operand != 0, address.columns)});
    }
  auto d = ResolveTmem(t, e, ops[0]);
  if (group == 2 && !peer_placement(d))
    throw SemanticError("tmem_group2_relative_peer_base",
                        "group2 requires matched allocation placement");
  const auto &frame = resources->frame;
  std::vector<CtaRank> owners{frame.self()};
  if (group == 2) {
    auto rank = frame.self().value;
    owners = {frame.Rank(rank & ~1u), frame.Rank(rank | 1u)};
  }
  bool a_tmem = ops[1].starts_with('[');
  std::optional<TmemAddress> a;
  std::uint64_t shared_a = 0;
  if (a_tmem) {
    a = ResolveTmem(t, e, ops[1]);
    if (group == 2 && !peer_placement(*a))
      throw SemanticError("tmem_group2_relative_peer_base",
                          "group2 A needs matched placement");
  } else
    shared_a = Descriptor(e, ops[1], "MMA shared A descriptor");
  auto shared_b = Descriptor(e, ops[2], "MMA shared B descriptor");
  std::optional<TmemAddress> metadata;
  if (sparse) {
    metadata = ResolveTmem(t, e, ops[3]);
    if (group == 2 && !peer_placement(*metadata))
      throw SemanticError("tmem_group2_relative_peer_base", "metadata peers require matched placement");
  }
  // Layouts C and F use half the lanes; A, D and the sparsity metadata must share one
  // lane alignment (PTX ISA, "Data Path Layout Organization").
  if (const auto path = DecodeMmaDataPath(desc.m, group, ws, sparse); path && path->half_quarters) {
    const auto lane = [](const TmemAddress &address) { return (address.address >> 16) & 0x7f; };
    if ((a && lane(*a) != lane(d)) || (metadata && lane(*metadata) != lane(d)))
      throw SemanticError("tmem_lane_alignment_mismatch",
                          "layout C and F MMA operands need the lane alignment of D (undefined behavior)",
                          {{"d_lane", lane(d)}, {"a_lane", a ? Json(lane(*a)) : Json()},
                           {"metadata_lane", metadata ? Json(lane(*metadata)) : Json()}});
  }
  std::vector<Fragment> reads, writes;
  std::vector<std::pair<MmaOperandRole,std::uint32_t>> read_roles, write_roles;
  std::vector<std::uint32_t> a_tmem_reads;
  std::map<std::uint32_t, std::pair<std::string, std::uint64_t>> operand_addresses;
  std::map<std::size_t, std::size_t> pairs;
  for (std::size_t part = 0; part < owners.size(); ++part) {
    auto owner = owners[part];
    std::optional<std::uint32_t> portion;
    if (group == 2)
      portion = part;
    auto bind = [&](const auto &intervals, const std::string &resource,
                    const std::string &space, auto &out) {
      for (const auto &range : intervals)
        out.push_back(
            resources->Bind(space, resource, range.offset, range.size, owner));
    };
    auto shared = [&](std::uint64_t value, std::uint32_t mn, std::uint32_t k,
                      std::uint32_t bits, const std::string &major) {
      try {
        auto intervals = DecodeSharedMatrixFootprint(value, mn, packed_fp4 ? k / 2 : k,
            packed_fp4 || (scaled && bits < 8) ? 8 : bits, major);
        if (scaled && !packed_fp4 && bits < 8) {
          if (k != 32 || (bits != 4 && bits != 6) || major != "k")
            throw LayoutError("subbyte MXF8 shared packing requires dense K-major K=32");
          std::vector<ByteInterval> payload;
          for (auto range : intervals)
            for (auto offset = range.offset; offset < range.offset + range.size; offset += 16)
              payload.push_back({offset, 2ULL * bits});
          intervals = std::move(payload);
        }
        bind(intervals, "physical.shared.cta.0", "shared", reads);
      } catch (const LayoutError &) {
        reads.push_back(resources->Abstract("shared", "physical.shared.cta.0",
            {{"operator", "mma_shared_region"},
             {"operands", Json::array({value, mn, k, bits, major})}},
            std::uint64_t{mn} * k * bits / 8, owner));
      }
    };
    auto sparse_region = [&](const std::string &role, const TmemAddress &address) {
      Json full_descriptor;
      try { full_descriptor = Scalar(e, ops[descriptor_index], "MMA descriptor"); }
      catch (const SemanticError &) {
        full_descriptor = {{"operator", "ptx_scalar"}, {"operands", Json::array({
          frame.Json(frame.self()), t.actor.warp.value(), t.actor.lane,
          Json::array({e.instruction_id(), e.occurrence()}), ops[descriptor_index]})}};
      }
      return resources->Abstract("tensor", address.resource,
        {{"operator", "sparse_mma_tensor_region"}, {"operands", Json::array({
          address.address, desc.kind, full_descriptor, group, role})}}, std::nullopt, owner);
    };
    auto remember_operand = [&](std::size_t begin, const TmemAddress &address) {
      for (auto index = begin; index < reads.size(); ++index) {
        a_tmem_reads.push_back(index);
        operand_addresses.emplace(index, std::pair{address.resource, address.address});
      }
    };
    const auto a_begin = reads.size();
    if (a && sparse) {
      reads.push_back(sparse_region("a", *a));
    } else if (a) {
      auto f = DecodeMmaATmemFootprint(footprint_instruction, a->address, footprint_idesc, a->resource,
                                       a->columns, portion, true);
      bind(f.reads, a->resource, "tensor", reads);
    } else
      shared(shared_a, desc.m / group, desc.k / (sparse ? 2 : 1), desc.a_element_bits, desc.a_major);
    if (a) remember_operand(a_begin, *a);
    if (retain_matrix_layout) read_roles.resize(reads.size(),{MmaOperandRole::A,part});
    shared(shared_b, desc.n / group, desc.k, desc.b_element_bits, desc.b_major);
    if (retain_matrix_layout) read_roles.resize(reads.size(),{MmaOperandRole::B,part});
    if (metadata) {
      const auto begin = reads.size();
      Resources::BitsResolver allocation_bits = [&](const RecordView &record,
          std::string_view token, unsigned) -> std::optional<Resources::KnownBits> {
        const auto value = tmem->provenance.Resolve(t.actor.warp.value(), record, token, t.actor.lane);
        if (auto pointer = std::get_if<TmemPointer>(&value))
          return Resources::KnownBits{pointer->encoded_offset & 31U, 31U};
        return {};
      };
      const auto [value, selector_bits] = resources->FixedBits(e, ops[descriptor_index], 32, allocation_bits);
      try {
        if (!scaled && (selector_bits & 3) != 3) throw LayoutError("metadata selector is not constant");
        bind(DecodeSparseMetadataFootprint(footprint_idesc | (scaled ? 0 : value & 3),
              metadata->address, group, scaled ? "f8f6f4" : desc.kind, metadata->columns),
             metadata->resource, "tensor", reads);
      } catch (const LayoutError &) { reads.push_back(sparse_region("metadata", *metadata)); }
      remember_operand(begin, *metadata);
      if (retain_matrix_layout) read_roles.resize(reads.size(),{MmaOperandRole::SparseMetadata,part});
    }
    for (std::size_t index = 0; index < scales.size(); ++index) {
      const auto begin = reads.size();
      bind(scales[index].second, scales[index].first, "tensor", reads);
      remember_operand(begin, scale_addresses[index]);
      if (retain_matrix_layout) read_roles.resize(reads.size(),{index ? MmaOperandRole::ScaleB : MmaOperandRole::ScaleA,part});
    }
    auto output = DecodeMmaDFootprint(footprint_instruction, d.address, footprint_idesc, enabled, mask,
                                      d.resource, d.columns, portion);
    for (const auto &range : output.writes) {
      auto fragment = resources->Bind("tensor", d.resource, range.offset,
                                      range.size, owner);
      if (enabled) {
        pairs[reads.size()] = writes.size();
        reads.push_back(fragment);
        if (retain_matrix_layout) read_roles.emplace_back(MmaOperandRole::D,part);
      }
      writes.push_back(fragment);
      if (retain_matrix_layout) write_roles.emplace_back(MmaOperandRole::D,part);
    }
  }
  MmaPlan plan;
  plan.cta_group=group; plan.a_tmem_reads=std::move(a_tmem_reads);
  plan.operand_addresses = std::move(operand_addresses);
  const bool separate=std::any_of(t.mma_operand_reads.begin(),t.mma_operand_reads.end(),
      [&](const auto &entry){return std::get<0>(entry.first)==group;});
  auto id = Next();
  MmaValueLayout value_layout{id,idesc,a ? a->address : shared_a,shared_b,d.address,a.has_value(),group,{}};
  if (scaled) value_layout.scale_bases={scale_addresses[0].address,scale_addresses[1].address};
  value_layout.ws = ws;
  plan.writes.resize(writes.size());
  std::set<std::size_t> paired;
  for (std::size_t k = 0; k < reads.size(); ++k) {
    auto [address, size] = reads[k];
    AsyncEffect effect{{MemoryEffect{AccessMode::kRead, address, size}}};
    if (retain_matrix_layout) value_layout.accesses.push_back({static_cast<std::uint32_t>(plan.effects.size()),0,read_roles.at(k).first,read_roles.at(k).second});
    if (pairs.contains(k) && !separate) {
      auto w = pairs.at(k);
      if (writes[w] != reads[k])
        throw SemanticError("mma_accumulator", "nonidentical accumulator pair");
      plan.writes[w] = plan.effects.size();
      plan.accumulators.push_back(plan.effects.size());
      effect.accesses.push_back({AccessMode::kWrite, address, size});
      if (retain_matrix_layout) value_layout.accesses.push_back({static_cast<std::uint32_t>(plan.effects.size()),1,MmaOperandRole::D,write_roles.at(w).second});
      paired.insert(w);
    }
    plan.effects.push_back(std::move(effect));
  }
  for (std::size_t k = 0; k < writes.size(); ++k)
    if (!paired.contains(k)) {
      plan.writes[k] = plan.effects.size();
      if (retain_matrix_layout) value_layout.accesses.push_back({static_cast<std::uint32_t>(plan.effects.size()),0,MmaOperandRole::D,write_roles.at(k).second});
      plan.effects.push_back({{MemoryEffect{AccessMode::kWrite, writes[k].first,
                                            writes[k].second}}});
    }
  if (separate) for (const auto &[read,write] : pairs) {
    auto target=plan.writes[write];
    plan.effects[target].predecessors.push_back(Ref(id,Milestone::kEffect,read));
    plan.split_accumulators.emplace_back(read,target);
  }
  if (retain_matrix_layout) mma_layouts.push_back(std::move(value_layout));
  std::string pipeline = std::to_string(group) + ":" +
                         std::to_string(desc.m) + ":" + std::to_string(desc.n) +
                         ":" + d.resource + ":" +
                         std::to_string(d.address);
  // .ws and non-.ws MMAs use different D layouts; they never share a pipeline.
  if (ws) pipeline += ":ws";
  const auto previous_mma = t.pipeline.find(pipeline);
  // Operand reads before a D fragment's watermark already precede it: they precede
  // the previous MMA's write of the same fragment on this pipeline (directly, or
  // by induction along the chain), and the pipeline link below orders that write
  // before this one. Only the reads added since are linked: HB is that of linking
  // every earlier read, at a per-fragment cost of the new reads only.
  for (auto target : plan.writes) {
    const auto &write=plan.effects[target].accesses.back();
    if (!write.bytes || !*write.bytes) continue;
    const auto &binding=resources->Binding(std::get<AddressId>(write.endpoint));
    auto key=std::tuple{group,binding.address_space,binding.alias_scope.empty()?binding.allocation:binding.alias_scope};
    auto previous=t.mma_operand_reads.find(key);
    const auto fragment = std::tuple{pipeline, std::get<AddressId>(write.endpoint).value(), *write.bytes};
    const auto watermark = t.operand_watermarks.find(fragment);
    const std::size_t from = watermark != t.operand_watermarks.end() && previous_mma != t.pipeline.end() &&
            watermark->second.first == previous_mma->second
        ? watermark->second.second : 0;
    t.operand_watermarks.insert_or_assign(
        fragment, std::pair{id, previous == t.mma_operand_reads.end() ? std::size_t{0} : previous->second.all.size()});
    if (previous==t.mma_operand_reads.end()) continue;
    for (const auto index : previous->second.Candidates(binding.allocation,binding.offset,*write.bytes,true,from)) {
      const auto &[ref,old,size]=previous->second.all[index];
      if (*size &&
          (old.allocation!=binding.allocation ||
           std::max(old.offset,binding.offset)<std::min(old.offset+*size,binding.offset+*write.bytes))) {
        UniquePush(plan.effects[target].predecessors,ref); plan.operand_links.emplace_back(target,ref);
      }
    }
  }
  // PTX 9.4: tcgen05-memory-consistency-model-pipelined-instructions.
  for (auto target : plan.a_tmem_reads) {
    const auto &read = plan.effects[target].accesses.front();
    const auto &binding = resources->Binding(std::get<AddressId>(read.endpoint));
    const auto key = std::tuple{group, binding.address_space, binding.allocation};
    const auto previous = t.copy_writes.find(key);
    if (previous == t.copy_writes.end() || !binding.expression.empty() || !read.bytes) continue;
    for (const auto &copy : previous->second) {
      const bool feeds = copy.any_operand_address ||
          CopyFeedsOperandBase(copy.operand_address, plan.operand_addresses.at(target).second);
      const bool contained = copy.binding.offset <= binding.offset &&
          binding.offset + *read.bytes <= copy.binding.offset + copy.bytes;
      if (!feeds && !contained) continue;
      if (copy.binding.offset < binding.offset + *read.bytes && binding.offset < copy.binding.offset + copy.bytes) {
        UniquePush(plan.effects[target].predecessors, copy.effect);
        plan.copy_links.emplace_back(target, copy.effect);
        if (!feeds) plan.contained_copy_links.emplace_back(target, copy.effect);
      }
    }
  }
  if (auto previous = previous_mma; previous != t.pipeline.end()) {
    const auto &prior = plans.at(previous->second.value());
    std::map<std::pair<std::uint32_t, std::uint64_t>,
             std::vector<std::uint32_t>>
        outputs;
    for (auto index : prior.writes)
      for (const auto &access : prior.effects[index].accesses)
        if (access.mode == AccessMode::kWrite && access.bytes)
          outputs[{std::get<AddressId>(access.endpoint).value(), *access.bytes}]
              .push_back(index);
    for (auto output : plan.writes) {
      auto target=output;
      const auto &effect = plan.effects[output];
      auto write = std::find_if(
          effect.accesses.begin(), effect.accesses.end(),
          [](const auto &x) { return x.mode == AccessMode::kWrite; });
      if (write == effect.accesses.end())
        throw SemanticError("mma_output", "missing output access");
      if (!write->bytes)
        continue;
      for (const auto &[read,paired_write] : plan.split_accumulators)
        if (paired_write==output) { target=read; break; }
      for (auto index : outputs[{std::get<AddressId>(write->endpoint).value(),
                                 *write->bytes}]) {
        auto ref = Ref(previous->second, Milestone::kEffect, index);
        UniquePush(plan.effects[target].predecessors, ref);
        plan.links.emplace_back(target, ref);
      }
    }
    limitations["tcgen_pipeline_effect_refinement"] =
        "Implicit TCGEN pipeline ordering needs effect-level refinement.";
  }
  t.pipeline[pipeline] = id;
  for (auto index : plan.a_tmem_reads) {
    const auto &read=plan.effects[index].accesses.front();
    const auto &binding=resources->Binding(std::get<AddressId>(read.endpoint));
    auto key=std::tuple{group,binding.address_space,binding.alias_scope.empty()?binding.allocation:binding.alias_scope};
    t.mma_operand_reads[key].Add(Ref(id,Milestone::kEffect,index),binding,read.bytes);
  }
  t.history.work[group].push_back(id.value());
  t.history.records.push_back({id.value(), static_cast<int>(group), "mma"});
  limitations["mma_memory_effect_refinement"] =
      "Independent MMA effects require safety and progress refinement.";
  auto effects = plan.effects;
  plans.emplace(id.value(), std::move(plan));
  Async(t, e, std::move(effects));
  profile.nodes = reads.size() + writes.size();
}
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
void ModelLowerer::Impl::Lifetime(Thread &t, const RecordView &e) {
  ProfileScope profile("frontend.model.lifetime");
  if (!tmem)
    throw SemanticError("tmem_allocation_identity", "no lifetime analysis");
  if (const auto *issue = BlockingTmemIssue(tmem->issues))
    ThrowTmemIssue(*issue);
  const auto &i = function.instructions[e.instruction_id()];
  auto key = Key(t.actor.warp.value(), e);
  if (i.semantics.operation == "alloc") {
    const TmemAllocation *allocation = nullptr;
    for (const auto &a : tmem->provenance.allocations)
      if (Key(a.site) == key) {
        if (allocation)
          throw SemanticError("tmem_allocation_identity",
                              "multiple lifetimes match allocation");
        allocation = &a;
      }
    if (!allocation)
      throw SemanticError("tmem_allocation_identity",
                          "no matching logical lifetime");
    std::vector<AsyncEffect> effects;
    if (t.actor.lane == 0) {
      auto a = resources->Bind("shared", "physical.shared.cta.0",
                               Unsigned(allocation->shared_slot,
                                        "TMEM publication slot", UINT32_MAX),
                               4);
      effects.push_back({{MemoryEffect{AccessMode::kWrite, a.first, 4}}});
      allocations[key] = Next();
    }
    auto id = Async(t, e, std::move(effects));
    auto tail = Emit(t, e, Await{{Ref(id)}});
    auto &align = aligned[key];
    align.calls[t.actor.lane] = {id, tail};
    align.completions = true;
  } else {
    auto id = Emit(t, e, Await{});
    aligned[key].calls[t.actor.lane] = {id, id};
  }
}
void ModelLowerer::Impl::LifetimeOrders() {
  if (!tmem)
    return;
  ProfileScope profile("frontend.model.lifetime_orders");
  // One immutable selection per source/allocation, shared by every target.
  // The target conjunction is unchanged; only its repeated payload is shared.
  using Effects = SharedOptional<std::vector<std::uint32_t>>;
  std::map<std::pair<InstId, std::string>, Effects> selected;
  std::map<std::string, SharedString> strings;
  const auto intern = [&](const std::string& text) -> const SharedString& {
    const auto [entry, inserted] = strings.try_emplace(text);
    if (inserted) entry->second = SharedString(text);
    return entry->second;
  };
  for (const auto &obligation : tmem->ordering_obligations) {
    const auto& kind = intern(obligation.kind);
    const auto& message = intern(obligation.message);
    std::vector<InstId> from, to;
    auto source_key = Key(obligation.source),
         target_key = Key(obligation.target);
    if (allocations.contains(source_key))
      from.push_back(allocations.at(source_key));
    else if (sites.contains(source_key))
      for (auto id : sites.at(source_key))
        if (source.program.instructions.holdsOperation<AsyncOp>(id.value()))
          from.push_back(id);
    std::set<ActorId> seen;
    if (sites.contains(target_key))
      for (auto id : sites.at(target_key))
        if (seen.insert(source.program.instructions.actor(id.value())).second)
          to.push_back(id);
    if (from.empty() || to.empty()) {
      Diagnostic("lifecycle_order_endpoint_missing", obligation.message, nullptr, nullptr,
          {{"constraint",obligation.kind},{"source_missing",from.empty()},{"target_missing",to.empty()},
           {"allocation",obligation.allocation.ResourceName()},
           {"source",{{"ptx_line",obligation.source.ptx_line},{"warp",obligation.source.warp},{"occurrence",obligation.source.occurrence}}},
           {"target",{{"ptx_line",obligation.target.ptx_line},{"warp",obligation.target.warp},{"occurrence",obligation.target.occurrence}}}});
      continue;
    }
    std::vector<MilestoneRef> targets;
    targets.reserve(to.size());
    for (auto id : to) targets.push_back(Ref(id, Milestone::kIssue));
    const SharedOptional<std::vector<MilestoneRef>> target_domain(std::move(targets));
    for (auto id : from) {
      Effects effects;
      if (obligation.kind == "access_completion_before_dealloc" &&
          !plans.contains(id.value())) {
        auto allocation =
            tmem->provenance.AllocationById(obligation.allocation);
        if (!allocation)
          throw SemanticError("unresolved_lifetime_footprint",
                              "lifetime allocation missing");
        auto name = resources->frame.Format(AllocationKey::Tensor(resources->frame.self(),
            allocation->columns == 512 ? std::string("physical.tmem.cta.0")
                                       : obligation.allocation.ResourceName())).allocation;
        auto [it, added] = selected.try_emplace({id, name});
        if (added) {
          std::vector<std::uint32_t> indices;
          bool tensor = false;
          const auto instruction = source.instruction(id);
          const auto &op = std::get<AsyncOp>(instruction.op);
          for (std::uint32_t k = 0; k < op.effects.size(); ++k) {
            bool match = false;
            for (const auto &access : op.effects[k].accesses)
              if (access.bytes)
                if (auto p = std::get_if<AddressId>(&access.endpoint)) {
                  const auto &binding = resources->Binding(*p);
                  if (binding.address_space == "tensor") {
                    tensor = true;
                    match |= binding.allocation == name;
                  }
                }
            if (match)
              indices.push_back(k);
          }
          if (tensor && indices.empty())
            throw SemanticError("unresolved_lifetime_footprint",
                                "retirement does not match allocation");
          it->second = std::move(indices);
        }
        effects = it->second;
      }
      orders.push_back({kind, Ref(id), target_domain->front(), message,
                        effects, target_domain});
    }
    Poll();
  }
}
InstId ModelLowerer::Impl::Copy(Thread &t, const RecordView &e,
                                std::vector<Fragment> reads,
                                std::vector<Fragment> writes,
                                std::optional<CompletionAction> completion) {
  auto id = Next();
  std::vector<AsyncEffect> effects;
  std::vector<std::tuple<std::uint64_t, std::uint64_t, std::uint32_t>> ranges;
  std::uint64_t cursor = 0, total = 0;
  for (auto [a, n] : writes) {
    (void)a;
    if (!n) throw SemanticError("copy_fragment_mapping", "copy byte extent is unknown");
    total += *n;
  }
  for (auto [a, n] : reads) {
    if (!n) throw SemanticError("copy_fragment_mapping", "copy byte extent is unknown");
    ranges.emplace_back(cursor, cursor + *n, effects.size());
    effects.push_back({{MemoryEffect{AccessMode::kRead, a, n}}});
    cursor += *n;
  }
  if (cursor > total)
    throw SemanticError("copy_fragment_mapping",
                        "copy source larger than destination");
  cursor = 0;
  for (auto [a, n] : writes) {
    const auto binding = resources->Binding(a);
    auto end = cursor + *n;
    std::set<std::uint64_t> points{cursor, end};
    for (auto [begin, stop, index] : ranges) {
      (void)index;
      if (cursor < begin && begin < end)
        points.insert(begin);
      if (cursor < stop && stop < end)
        points.insert(stop);
    }
    for (auto it = points.begin(), next = std::next(it); next != points.end();
         ++it, ++next) {
      std::vector<MilestoneRef> refs;
      for (auto [begin, stop, index] : ranges)
        if (begin <= *it && *it < stop)
          refs.push_back(Ref(id, Milestone::kEffect, index));
      // Interned even for an expression binding.
      auto shifted = resources->interner().Apply(
          ConstantStep{AllocationKey::Opaque(binding.address_space, binding.allocation, binding.alias_scope),
                       binding.offset + *it - cursor});
      if (!binding.expression.empty()) {
        auto expression = Json::parse(binding.expression);
        expression["address"]["offset"] = {{"operator", "add"},
            {"operands", Json::array({expression["address"]["offset"], *it - cursor})}};
        expression["byte_count"] = *next - *it;
        shifted = resources->interner().Apply(SerializedExpressionStep(binding.address_space, expression.dump(),
                                                                       binding.alias_scope, binding.allocation));
      }
      effects.push_back(
          {{MemoryEffect{AccessMode::kWrite, shifted, *next - *it}},
           {},
           std::move(refs)});
    }
    cursor = end;
  }
  if (completion) {
    std::vector<MilestoneRef> refs;
    for (std::uint32_t k = 0; k < effects.size(); ++k)
      refs.push_back(Ref(id, Milestone::kEffect, k));
    effects.push_back({{}, completion, std::move(refs)});
  }
  return Async(t, e, std::move(effects));
}
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
void ModelLowerer::Impl::Add(const CtaFacts &cta_facts, ValueAddresses value_addresses, bool lane_workers) {
  ProfileScope profile("frontend.model.cta");
  const FactSource::ConsumerScope consumer(cta_facts, {Consumer::Lowering, 0});
  const auto cta = cta_facts.Linear();
  const auto coordinates = cta_facts.frame().SelfCoordinates();
  facts = &cta_facts;
  current_coordinates=coordinates;
  shared_value_sites.clear();
  shared_value_instructions.assign(function.instructions.size(),false);
  const auto shared_writes = cta_facts.SharedConstantWrites();
  const auto shared_dependencies = cta_facts.SharedValueDependencies();
  for(const auto &site:shared_writes)shared_value_sites.insert(site);
  for(const auto &dependency:shared_dependencies)shared_value_sites.insert(dependency.read);
  for(const auto &site:shared_value_sites)shared_value_instructions.at(site.instruction.instruction)=true;
  shared_value_dependencies.insert(shared_value_dependencies.end(),shared_dependencies.begin(),shared_dependencies.end());
  tensor_map_copies.clear();
  if(const auto &reads=cta_facts.TensorMapReads())
    for(const auto &[read,write]:*reads) {
      tensor_map_dependencies.push_back({{coordinates,write.writer_warp*32,write.write},{coordinates,read.first,read.second}});
      tensor_map_copies.insert(tensor_map_dependencies.back().write);
    }
  const auto bridge = value_addresses; // the workers' Resources resolve as `r` does
  Resources r(cta_facts, cta_facts.frame(), addresses, barriers,
              SlotConfig{Consumer::Lowering, std::move(value_addresses)});
  resources = &r;
  MatrixResolver matrix_resolver(cta_facts);
  matrix = &matrix_resolver;
  std::optional<TmemLifetimeAnalysis> analysis;
  bool has_tcgen =
      std::any_of(function.instructions.begin(), function.instructions.end(),
                  [](const auto &i) { return i.opcode == "tcgen05"; });
  if (has_tcgen) {
    ProfileScope scope("frontend.tmem_provenance");
    TmemAnalysisOptions options;
    options.allow_cta_group2 = true;
    options.poll = poll;
    analysis = AnalyzeTmemLifetimes(cta_facts, ExpansionSite::Tmem, options);
    tmem = &*analysis;
    bool lifetimes =
        std::any_of(function.instructions.begin(), function.instructions.end(),
                    [](const auto &i) {
                      return i.opcode == "tcgen05" &&
                             (Has(i, "alloc") || Has(i, "dealloc"));
                    });
    if (!lifetimes)
      std::erase_if(tmem->issues, [](const auto &issue) {
        return issue.code == "tmem_unresolved_use_pointer";
      });
    for (const auto &issue : tmem->issues)
      if (issue.code == "tmem_multiple_dealloc") {
        if (cta_facts.recording()) [[unlikely]]
          cta_facts.Note(query::ReportBearing{query::ReportKind::RenderedViolation, issue.code});
        auto suffix = issue.message + "; first deallocation at warp " +
                      std::to_string(issue.site.warp) + ", instruction " +
                      std::to_string(issue.site.instruction_index) +
                      ", occurrence " + std::to_string(issue.site.occurrence);
        violations.emplace(issue.code, "CTA " + r.frame.Text(r.frame.self()) + ": " + suffix);
        if (units) cta_violations.emplace_back(cta, issue.code, std::move(suffix));
      }
    if (tmem->cluster_pairing_required)
      limitations["tmem_group2_pairing"] =
          "Group2 lifetime requires a cluster participation and progress "
          "certificate.";
  }
  auto actor_begin = source.program.actors.size();
  const auto lane_count = cta_facts.LaneCount();
  const bool template_execution = cta_facts.TemplateExecution();
  std::vector<std::uintptr_t> member_cohorts(lane_count,0);
  if(template_execution) {
    const auto template_groups = cta_facts.TemplateGroups();
    for(std::size_t group=0;group<template_groups.size();++group)
      for(auto member:template_groups[group]) {
        if(member>=member_cohorts.size() || member_cohorts[member])
          throw SemanticError("invalid_thread_partition","template members must form a partition");
        member_cohorts[member]=group+1;
      }
    if(std::find(member_cohorts.begin(),member_cohorts.end(),0)!=member_cohorts.end())
      throw SemanticError("invalid_thread_partition","template partition does not cover every thread");
  }
  for (std::size_t k = 0; k < lane_count; ++k) {
    const auto lane = cta_facts.Lane(k);
    const auto warp_id = cta_facts.Warp(lane), lane_id = cta_facts.LaneId(lane);
    ActorInfo actor{ActorId(actor_begin + k), CtaId(cta), WarpId(warp_id),
                    lane_id};
    if (!actors.emplace(std::pair{warp_id, lane_id}, actor.id).second)
      throw SemanticError("duplicate_lane", "duplicate lane trace");
    source.program.actors.push_back(actor);
  }
  struct GeneratedFamily {
    std::size_t instruction_begin{},address_begin{},members{};
    ActorId actor;std::uintptr_t cohort{};
  };
  struct Projection {
    std::vector<std::size_t> positions;
    std::vector<std::uint64_t> skipped_counts;
    std::vector<std::uint64_t> memory_counts;
    std::shared_ptr<std::vector<std::pair<std::uint32_t,std::uint32_t>>> body;
    std::shared_ptr<AddressBindings> bindings;
    std::unordered_map<std::size_t,AddressId> uniform_global_addresses;
    std::shared_ptr<std::vector<AddressId>> uniform_body;
    std::shared_ptr<std::vector<std::size_t>> varying_slots;
    std::vector<std::size_t> varying_positions;
    std::shared_ptr<GeneratedFamily> family;
    std::shared_ptr<std::vector<std::size_t>> variable_columns;
  };
  std::map<ProjectionKey, Projection> projections;
  bool synchronous_template=template_execution &&
      std::all_of(function.instructions.begin(),function.instructions.end(),[](const auto &i) {
        const auto &s=i.semantics;
        if(IsWarpDataCollective(i))return false;
        return s.category=="compute" || s.category=="control" ||
            (s.category=="memory" && s.modeled && (s.memory_mode=="read" || s.memory_mode=="write"));
      });
  // That path takes memory events from memory records only; a skipped region's st.local is
  // lowered at the region's branch record (Instruction()). One scan per projection, and only
  // in functions with a st.local.
  if(synchronous_template &&
      std::any_of(function.instructions.begin(),function.instructions.end(),[](const auto &i) { return LocalStoreInstruction(i); })) {
    std::set<ProjectionKey> scanned;
    for(std::size_t k=0;k<lane_count && synchronous_template;++k) {
      const auto lane_ref=cta_facts.Lane(k);
      if(!scanned.insert(cta_facts.Projection(lane_ref)).second)continue;
      for(std::size_t index=0;index<cta_facts.Size(lane_ref) && synchronous_template;++index) {
        if((index&1023)==0)Poll();
        const auto record=cta_facts.Meta({lane_ref,index});
        for(const auto region:record.opaque_compute_diamond_indices())
          if(LocalStoreInstruction(function.instructions.at(region))){synchronous_template=false;break;}
      }
    }
  }
  index_sites = !synchronous_template;
  const bool generated_memory=synchronous_template &&
      std::all_of(function.instructions.begin(),function.instructions.end(),[](const auto& i) {
        if(i.semantics.category!="memory")return true;
        return (i.opcode=="ld" || i.opcode=="st") && !i.address_lanes &&
            i.memory_width && *i.memory_width<=UINT32_MAX;
      });
  auto prototypes=std::make_shared<std::vector<SourceInstruction>>(generated_memory?function.instructions.size():0);
  std::vector<std::string> memory_tokens(generated_memory?function.instructions.size():0);
  std::vector<std::optional<std::uint32_t>> memory_registers(memory_tokens.size());
  if(generated_memory)for(const auto& i:function.instructions) {
    if(i.semantics.category!="memory")continue;
    auto brackets=Brackets(operands[i.id]);
    if(brackets.empty())continue;
    memory_tokens[i.id]=brackets.front();
    const auto token=Trim(std::string_view(brackets.front()).substr(1,brackets.front().size()-2));
    const auto address=SplitAddress(token);
    const auto& base=address ? address->first : token;
    if(const auto reg=cta_facts.Reg(base))memory_registers[i.id]=*reg;
    SourceOp operation=i.semantics.memory_mode=="read"
      ?SourceOp(ImmediateLoad{AddressId{},static_cast<std::uint32_t>(*i.memory_width)})
      :SourceOp(ImmediateStore{AddressId{},static_cast<std::uint32_t>(*i.memory_width)});
    prototypes->at(i.id)={InstId{},ActorId{},Location(i),std::move(operation),
        PtxOrigin{source_name,i.id,0,origin_opcodes[i.id]}};
  }
  // Per static instruction, the registers lowering reads from its records, when they are known
  // (LowerThread reads only those; others are read in full): a warp collective's member mask; an
  // ordinary load or store's address operand (its slot reads only that register there,
  // AddressRegisters); a tcgen05.ld/st's TMEM address and half-split offset (not its data registers).
  std::vector<std::optional<std::vector<RegId>>> read_slices(function.instructions.size());
  for(const auto& i:function.instructions) {
    const auto& ops=operands[i.id];
    if(IsWarpDataCollective(i)) {
      if(ops.empty())continue;
      if(const auto reg=cta_facts.Reg(Trim(ops.back())))read_slices[i.id]=std::vector<RegId>{*reg};
    } else if(OrdinaryMemory(i)) {
      if(const auto brackets=Brackets(ops); !brackets.empty())read_slices[i.id]=AddressRegisters(cta_facts,brackets.front());
    } else if(i.opcode=="tcgen05" && (i.semantics.operation=="load" || i.semantics.operation=="store")) {
      std::vector<RegId> regs;
      for(const auto& op:ops) {
        if(Trim(op).starts_with('{'))continue;
        for(const auto reg:AddressRegisters(cta_facts,op))
          if(std::find(regs.begin(),regs.end(),reg)==regs.end())regs.push_back(reg);
      }
      std::sort(regs.begin(),regs.end());
      read_slices[i.id]=std::move(regs);
    }
  }
  std::uint64_t visits = 0;
  // A thread extends only the preceding thread's family block. Lower each
  // template cohort contiguously (stable by thread), so interleaved cohorts,
  // such as one elected lane per warp, form one block each. Cohorts keep the
  // order of their first thread, so non-interleaved cohorts keep thread order.
  std::vector<std::size_t> thread_order(lane_count);
  std::iota(thread_order.begin(), thread_order.end(), std::size_t{0});
  if (template_execution) {
    std::unordered_map<std::uintptr_t, std::size_t> first_thread;
    for (std::size_t k = 0; k < member_cohorts.size(); ++k) first_thread.try_emplace(member_cohorts[k], k);
    std::stable_sort(thread_order.begin(), thread_order.end(), [&](std::size_t a, std::size_t b) {
      return first_thread.at(member_cohorts[a]) < first_thread.at(member_cohorts[b]);
    });
  }
  // Lanes on worker threads, applied in this order: not with synchronous-template projections
  // (their address memo and generated families span lanes), nor while a bridge query may still
  // build the value view the workers' records come from.
  if (threads > 1 && lane_workers && !synchronous_template && lane_count > 1) {
    const std::vector<ActorInfo> lane_actors(source.program.actors.begin() + actor_begin, source.program.actors.end());
    LowerLanes(cta_facts, r, bridge, thread_order, lane_actors, read_slices, member_cohorts, template_execution,
               visits);
  } else
  for (std::size_t position = 0; position < thread_order.size(); ++position) {
    const auto k = thread_order[position];
    const auto lane_ref = cta_facts.Lane(k);
    const auto binding_begin=template_execution ? source.program.instructions.beginThread() : 0;
    current_lane = lane_ref;
    lane_position = position;
    ProfileCall("frontend.resources.index_lane", [&] { r.BeginLane(lane_ref); });
    Thread thread;
    thread.actor = source.program.actors[actor_begin + k];
    Projection* projection=nullptr;
    if (synchronous_template) {
      auto [entry,inserted]=projections.try_emplace(cta_facts.Projection(lane_ref));
      auto &view=entry->second;
      if (inserted) {
        view.skipped_counts.resize(function.instructions.size());
        view.memory_counts.resize(function.instructions.size());
        if(generated_memory) {
          view.body=std::make_shared<std::vector<std::pair<std::uint32_t,std::uint32_t>>>();
          view.bindings=std::make_shared<AddressBindings>();
        }
        for (std::size_t index=0;index<cta_facts.Size(lane_ref);++index) {
          if ((index&1023)==0) Poll();
          const auto record=cta_facts.Meta({lane_ref,index});
          if (record.predicate_value()==false) continue;
          const auto &s=function.instructions[record.instruction_id()].semantics;
          if (s.category=="compute" || s.category=="control" || s.memory_space=="param" || s.memory_space=="const")
            ++view.skipped_counts[record.instruction_id()];
          else {
            view.positions.push_back(index);
            if(generated_memory) {
              view.body->emplace_back(record.instruction_id(),static_cast<std::uint32_t>(record.occurrence()));
              ++view.memory_counts[record.instruction_id()];
            }
          }
        }
      }
      for (std::size_t id=0;id<view.skipped_counts.size();++id) occurrence_counts[id]+=view.skipped_counts[id];
      projection=&view;
    }
    if(generated_memory && projection) {
      ProfileScope generated_profile("frontend.model.template_memory_bindings");
      const auto& memory_positions=projection->uniform_body?projection->varying_positions:projection->positions;
      std::vector<AddressId> resolved;resolved.reserve(memory_positions.size());
      std::unordered_map<std::uint64_t,AddressId> evaluated_addresses;
      bool supported=true;
      for(auto index:memory_positions) {
        if((++visits&1023)==0)Poll();
        const auto instruction=cta_facts.Meta({lane_ref,index}).instruction_id();
        const auto& reg=memory_registers[instruction];
        const auto metadata=cta_facts.Meta({lane_ref,index});
        const auto& static_instruction=function.instructions[instruction];
        const auto& contract=cta_facts.launch().InstructionContract(static_instruction.line,instruction,cta_facts.Warp(lane_ref),metadata.occurrence());
        const auto evaluation=reg && contract.empty()?cta_facts.PathEvaluationKey({lane_ref,index},*reg):std::nullopt;
        const auto evaluation_key=evaluation && *evaluation<=UINT32_MAX?
            std::optional<std::uint64_t>((std::uint64_t(*evaluation)<<32)|instruction):std::nullopt;
        if(evaluation_key)if(const auto cached=evaluated_addresses.find(*evaluation_key);cached!=evaluated_addresses.end()) {
          // A MemoHit event: the hit reuses the fill's resolution (one branch; inert unless recording).
          if(cta_facts.recording()) [[unlikely]] NoteEvaluatedMemo(cta_facts,true,{lane_ref,index},*evaluation_key,*reg);
          resolved.push_back(cached->second);continue;
        }
        const bool uniform_global=static_instruction.semantics.memory_space==std::optional<std::string>("global") &&
            contract.empty() && reg && cta_facts.PathUniformBinding({lane_ref,index},*reg);
        if(uniform_global)if(const auto cached=projection->uniform_global_addresses.find(index);
            cached!=projection->uniform_global_addresses.end()) {
          resolved.push_back(cached->second);continue;
        }
        const auto held=reg ? cta_facts.ReadSlice({lane_ref,index},std::span(&*reg,1))
                            : cta_facts.ReadFull({lane_ref,index});
        const auto e=held.view();
        const auto& i=function.instructions[e.instruction_id()];
        try {
          if((i.predicate && !e.predicate_value()) || e.canonical_wait_probe() || memory_tokens[i.id].empty()) {
            supported=false;break;
          }
          MemoryRequest request(memory_tokens[i.id],contract,*i.memory_width,
              i.semantics.memory_space.value_or("generic"));
          const auto [address,bytes]=r.Memory(e,request);
          if(bytes!=i.memory_width){supported=false;break;}
          resolved.push_back(address);
          if(evaluation_key && evaluated_addresses.size()<100000 &&
              r.Binding(address).expression.empty()) {
            evaluated_addresses.emplace(*evaluation_key,address);
            // A MemoFill event (one branch; inert unless recording).
            if(cta_facts.recording()) [[unlikely]] NoteEvaluatedMemo(cta_facts,false,{lane_ref,index},*evaluation_key,*reg);
          }
          if(uniform_global && r.Binding(address).expression.empty())
            projection->uniform_global_addresses.emplace(index,address);
        } catch(const std::exception&) {supported=false;break;}
      }
      if(supported) {
        if(!projection->uniform_body && cta_facts.launch().instruction_contracts().empty()) {
          projection->uniform_body=std::make_shared<std::vector<AddressId>>(resolved);
          projection->varying_slots=std::make_shared<std::vector<std::size_t>>(resolved.size(),SIZE_MAX);
          std::vector<AddressId> varying;
          for(std::size_t i=0;i<projection->positions.size();++i) {
            if(projection->uniform_global_addresses.contains(projection->positions[i]))continue;
            projection->varying_slots->at(i)=varying.size();
            projection->varying_positions.push_back(projection->positions[i]);
            varying.push_back(resolved[i]);
          }
          resolved=std::move(varying);
        }
        auto bindings=projection->bindings;
        const auto begin=bindings->beginThread();
        for(auto address:resolved)bindings->push_back(address);
        bindings->compressThreadTail(begin,cta,member_cohorts[k]);
        auto body=projection->body;
        const auto uniform_body=projection->uniform_body;
        const auto varying_slots=projection->varying_slots;
        const auto first_instruction=Next();
        auto& family=projection->family;
        if(!family || family->instruction_begin+family->members*body->size()!=first_instruction.value() ||
            family->actor.value()+family->members!=thread.actor.id.value() || family->cohort!=member_cohorts[k])
          family=std::make_shared<GeneratedFamily>(GeneratedFamily{first_instruction.value(),begin,0,thread.actor.id,member_cohorts[k]});
        ++family->members;
        const auto varying_width=resolved.size();
        const auto make=[body,prototypes,family](std::size_t column,std::size_t member,AddressId address) {
          const auto [instruction,occurrence]=body->at(column);
          auto value=prototypes->at(instruction);
          if(auto* load=std::get_if<ImmediateLoad>(&value.op))load->address=address;
          else std::get<ImmediateStore>(value.op).address=address;
          value.ptx_origin->occurrence=occurrence;
          value.id=InstId(family->instruction_begin+member*body->size()+column);
          value.actor=ActorId(family->actor.value()+member);
          return value;
        };
        if(!projection->variable_columns) {
          projection->variable_columns=std::make_shared<std::vector<std::size_t>>(varying_width);
          for(std::size_t column=0;column<body->size();++column) {
            const auto slot=varying_slots?varying_slots->at(column):column;
            if(slot!=SIZE_MAX)projection->variable_columns->at(slot)=column;
          }
        }
        const auto variable_columns=projection->variable_columns;
        source.appendGeneratedThread(thread.actor.id,body->size(),
          [body,bindings,family,uniform_body,varying_slots,varying_width,make](std::size_t i) {
            const auto column=i%body->size(),member=i/body->size();
            const auto slot=varying_slots?varying_slots->at(column):column;
            return make(column,member,slot==SIZE_MAX?uniform_body->at(column):
                bindings->at(family->address_begin+member*varying_width+slot));
          },std::pair{std::uint64_t(cta),member_cohorts[k]},
          [body,bindings,family,uniform_body,varying_slots,varying_width,variable_columns,make](const auto& visit) {
            for(std::size_t column=0;column<body->size();++column) {
              if(!varying_slots || varying_slots->at(column)!=SIZE_MAX)continue;
              SourceRecordCodec::Delta delta{};delta.fields[0]=body->size();delta.fields[1]=1;
              visit(make(column,0,uniform_body->at(column)),delta,column,body->size(),family->members);
            }
            if(!varying_width)return;
            bindings->forEachAffineColumn([&](AddressId address,std::int64_t address_step,
                std::size_t first,std::size_t stride,std::size_t count) {
              const auto match=IntersectAffine({first,stride,count},
                  {family->address_begin,1,family->members*varying_width});
              if(!match.count)return;
              const auto logical=first+stride*match.left;
              const auto step=stride*match.left_step;
              const auto period=varying_width/std::gcd(step,varying_width);
              // Split only by body-column residue, never by spatial member.
              for(std::size_t residue=0;residue<std::min<std::size_t>(period,match.count);++residue) {
                const auto offset=logical+step*residue-family->address_begin;
                const auto member=offset/varying_width;const auto column=variable_columns->at(offset%varying_width);
                const auto n=1+(match.count-1-residue)/period;
                const auto member_step=step*period/varying_width;
                const auto start=AddressBindingCodec::translate(address,address_step,match.left+match.left_step*residue);
                if(!start)throw std::overflow_error("Generated address domain overflows");
                SourceRecordCodec::Delta delta{};delta.fields[0]=body->size()*member_step;delta.fields[1]=member_step;
                delta.fields[2]=address_step*std::int64_t(match.left_step*period);
                visit(make(column,member,*start),delta,member*body->size()+column,body->size()*member_step,n);
              }
            });
          },family.get());
        if(!shared_value_sites.empty())for(std::size_t i=0;i<body->size();++i) {
          const auto [instruction,occurrence]=body->at(i);
          if(!shared_value_instructions[instruction])continue;
          const auto id=InstId(first_instruction.value()+static_cast<std::uint32_t>(i));
          SharedValueEndpoint(thread.actor,{instruction,occurrence},id,source.instruction(id).op);
        }
        for(std::size_t id=0;id<projection->memory_counts.size();++id)
          occurrence_counts[id]+=projection->memory_counts[id];
        histories.push_back(std::move(thread.history));
        continue;
      }
    }
    IterationStarts iteration_starts;
    LowerThread(cta_facts, lane_ref, thread, projection ? &projection->positions : nullptr, read_slices, visits,
                template_execution ? &iteration_starts : nullptr);
    histories.push_back(std::move(thread.history));
    if(template_execution) {
      source.program.instructions.compressIterations(binding_begin,iteration_starts);
      source.program.instructions.extendThreadFamily(binding_begin,cta,member_cohorts[k]);
    }
  }
  for(const auto& [identity,projection]:projections) {
    (void)identity;
    if(!projection.bindings || projection.bindings->empty())continue;
    generated_memory_instances+=projection.bindings->size();
    generated_address_rows+=projection.bindings->storedBindingCount();
    generated_body_rows+=projection.body->size();
  }
  LifetimeOrders();
  CollectGroup2();
  for (const auto &[key,entries] : cluster_warp_rendezvous) {
    const auto warp=std::get<0>(key);
    std::vector<MilestoneRef> requirements;
    for (const auto &[actor,id] : entries) {
      (void)actor;
      requirements.push_back(Ref(id,Milestone::kIssue));
    }
    for (std::size_t k=0;k<lane_count;++k) {
      const auto lane=cta_facts.Lane(k);
      if (cta_facts.Warp(lane)!=warp) continue;
      const auto actor=source.program.actors[actor_begin+k];
      if (entries.contains(actor.id)) continue;
      const auto terminal=Terminal(cta_facts,k,actor);
      if (!terminal) {
        Diagnostic("incomplete_cluster_warp","warp peer has no cluster operation or terminal return"); continue;
      }
      requirements.push_back(Ref(*terminal));
    }
    for (const auto &[actor,id] : entries) {
      (void)actor;
      auto &target=std::get<Await>(source.editInstruction(id).op).requirements;
      for (auto ref : requirements) UniquePush(target,ref);
    }
  }
  CtaWideExits(cta_facts, actor_begin);
  terminals.clear();
  {
    ProfileScope joins("frontend.model.cta_joins");
    // Threads of each warp of the CTA (a partial last warp has fewer). O(lanes).
    std::map<std::uint32_t, std::uint32_t> warp_threads;
    for (std::size_t k = 0; k < lane_count; ++k) ++warp_threads[cta_facts.Warp(cta_facts.Lane(k))];
    for (auto &[site, entry] : aligned) {
      const bool full_warp = warp_threads[std::get<0>(site)] == 32;
      std::bitset<32> lowered, failed;
      for (const auto &[lane, call] : entry.calls) { (void)call; lowered.set(lane); }
      for (const auto lane : entry.failed) failed.set(lane);
      const auto participation = AlignedSiteParticipation(lowered, failed);
      // Every executing lane failed to lower it, and no lane of a full warp is missing: those
      // lanes' diagnostics report the site.
      if (entry.calls.empty() && (participation.missing.empty() || !full_warp)) continue;
      const auto &instruction = function.instructions[std::get<1>(site)];
      const bool tmem_access = instruction.opcode == "tcgen05" &&
          (instruction.semantics.operation == "load" || instruction.semantics.operation == "store");
      const auto location = [&](Json details) {
        details["line"] = instruction.line;
        details["warp"] = std::get<0>(site);
        details["occurrence"] = std::get<2>(site);
        details["instruction"] = std::string(module.view(instruction.source));
        return details;
      };
      if (entry.calls.size() != 32) {
        // PTX ISA tcgen05.ld/st: all threads of the warp must execute it (.aligned), and "the
        // behavior of tcgen05.ld [st] is undefined if ... any thread in the warp has exited".
        // In a full warp a lane without this occurrence exited or diverged: undefined behavior.
        // Every other .aligned instruction: "all threads in the warp must execute the same
        // ... instruction" (tcgen05.wait, barrier.cluster .aligned, tensormap.cp_fenceproxy,
        // ldmatrix, stmatrix, tcgen05 lifetime operations); a missing lane of a full warp
        // violates it. A lane that executed the occurrence but failed to lower it is not missing.
        const Json missing = participation.missing;
        if (!tmem_access && full_warp && !missing.empty()) {
          const auto &op = instruction.opcode;
          // A tcgen05 site other than ld/st is tcgen05.wait or a lifetime operation.
          const char *section = op == "tcgen05" ? (instruction.semantics.operation.starts_with("wait")
                                                       ? "9.7.18.8.5 tcgen05.wait" : "9.7.18.7.1 tcgen05.alloc") :
              op == "barrier" ? "9.7.15.3 barrier.cluster" : op == "tensormap" ? "9.7.15.17 tensormap.cp_fenceproxy" :
              op == "ldmatrix" ? "9.7.16.5.15 ldmatrix" : op == "stmatrix" ? "9.7.16.5.16 stmatrix" : "the instruction's .aligned qualifier";
          Diagnostic("aligned_lanes_missing",
                     op + " .aligned is not executed by every thread of its full warp (undefined behavior)",
                     nullptr, nullptr, location({{"missing_lanes", missing}, {"isa_section", section}}));
          continue;
        }
        if (tmem_access && full_warp && !missing.empty()) {
          Diagnostic("tmem_aligned_lanes_missing",
                     "tcgen05." + std::string(instruction.semantics.operation == "load" ? "ld" : "st") +
                         " is not executed by every thread of its full warp (undefined behavior)",
                     nullptr, nullptr, location({{"missing_lanes", missing},
                         {"isa_section", instruction.semantics.operation == "load" ? "9.7.18.8.3 tcgen05.ld" : "9.7.18.8.4 tcgen05.st"}}));
          continue;
        }
        Diagnostic("incomplete_aligned_warp",
                   "aligned instruction is not executed by all32 lanes");
        continue;
      }
      // PTX ISA tcgen05.ld/st: "The behavior of tcgen05.ld [st] is undefined if all threads do
      // not use the same values of taddr". O(32) per site.
      if (tmem_access && entry.tmem_addresses.size() == 32) {
        const auto &first = entry.tmem_addresses.begin()->second;
        for (const auto &[lane, value] : entry.tmem_addresses)
          if (value != first) {
            Diagnostic("tmem_address_not_uniform",
                       "tcgen05." + std::string(instruction.semantics.operation == "load" ? "ld" : "st") +
                           " lanes use different taddr values (undefined behavior)",
                       nullptr, nullptr, location({{"lane", lane}, {"taddr", value.second}, {"resource", value.first},
                           {"lane0_taddr", first.second}, {"lane0_resource", first.first},
                           {"isa_section", instruction.semantics.operation == "load" ? "9.7.18.8.3 tcgen05.ld" : "9.7.18.8.4 tcgen05.st"}}));
            break;
          }
      }
      std::vector<std::pair<InstId, InstId>> calls;
      for (std::uint32_t lane = 0; lane < 32; ++lane) {
        auto it = entry.calls.find(lane);
        if (it == entry.calls.end())
          throw SemanticError("incomplete_aligned_warp", "missing lane");
        calls.push_back(it->second);
      }
      if (!entry.merged_arm) source.alignWarp(calls, entry.completions);
    }
  }
  ++ctas;
  if (units) {
    // O(static instructions) per representative CTA.
    std::vector<std::pair<std::uint32_t, std::uint64_t>> counts;
    for (std::size_t id = 0; id < occurrence_counts.size(); ++id)
      if (occurrence_counts[id]) {
        counts.emplace_back(static_cast<std::uint32_t>(id), occurrence_counts[id]);
        occurrence_counts[id] = 0;
      }
    cta_occurrences.emplace_back(cta, std::move(counts));
  }
  profile.nodes = visits;
  SENTRYWARD_DEBUG_PRINT(lowering_0);
  SENTRYWARD_DEBUG_PRINT(lowering_1);
  facts = nullptr;
  resources = nullptr;
  tmem = nullptr;
  matrix = nullptr;
  current_lane.reset();
  sites.clear();
  aligned.clear();
  cluster_warp_rendezvous.clear();
  allocations.clear();
  actors.clear();
  warp_collectives.clear();
  Poll();
}
// One lane's records in order (`positions`: a synchronous-template projection's, else every
// record), lowered into this Impl's state (a worker's: the lane's own).
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
void ModelLowerer::AddCta(const CtaFacts &facts, ValueAddresses value_addresses, bool lane_workers) {
  impl_->Add(facts, std::move(value_addresses), lane_workers);
}
const algorithm::SourceProgram &ModelLowerer::currentProgram() const noexcept {
  return impl_->source.program;
}
void ModelLowerer::TrackUnits(algorithm::UnitQuotient *quotient) noexcept { impl_->units = quotient; }
LoweringResult ModelLowerer::Finish() { return impl_->Finish(); }
} // namespace sentryward::frontend::ptx
