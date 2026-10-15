#include "sentryward/algorithm/diagnostics/development_trace.h"
#include "sentryward/frontend/ptx/lowering.h"
#include "sentryward/algorithm/diagnostics/performance_profile.h"
#include "sentryward/algorithm/source_ir/builder.h"
#include "sentryward/algorithm/base/affine_index.h"
#include "sentryward/frontend/ptx/dependency_history.h"
#include "sentryward/frontend/ptx/group2_lifetime.h"
#include "sentryward/frontend/ptx/layouts.h"
#include "sentryward/frontend/ptx/matrix.h"
#include "sentryward/frontend/ptx/refinement.h"
#include "sentryward/frontend/ptx/resources.h"
#include "sentryward/frontend/ptx/tmem_lifetime.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
#include <regex>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>
namespace sentryward::frontend::ptx {
using Json = nlohmann::json;
namespace {
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
std::vector<std::string> Brackets(const std::vector<std::string> &v) {
  std::vector<std::string> r;
  for (const auto &x : v)
    if (x.starts_with('[') && x.ends_with(']'))
      r.push_back(x);
  return r;
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
  TcgenHistory history;
  std::map<std::string, InstId> pipeline;
  std::map<std::tuple<unsigned,std::string,std::string>,
      std::vector<std::tuple<MilestoneRef,ConstantAddressBinding,std::optional<std::uint64_t>>>> mma_operand_reads;
  struct CopyWrite {
    MilestoneRef effect;
    ConstantAddressBinding binding;
    std::uint64_t bytes, operand_address;
    bool any_operand_address;
  };
  std::map<std::tuple<unsigned,std::string,std::string>, std::vector<CopyWrite>> copy_writes;
};
using Site = std::tuple<std::uint32_t, std::uint32_t, std::uint64_t>;
Site Key(std::uint32_t warp, const ExecutedInstruction &e) {
  return {warp, e.instruction_id(), e.occurrence()};
}
Site Key(const TmemAtomSite &s) {
  return {s.warp, s.instruction_index, s.occurrence};
}
struct Aligned {
  std::map<std::uint32_t, std::pair<InstId, InstId>> calls;
  bool completions = false;
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
} // namespace
struct ModelLowerer::Impl {
  bool retain_matrix_layout = false;
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
  Coordinates current_coordinates{};
  std::map<std::string, std::string> limitations;
  std::set<std::pair<std::string, std::string>> violations;
  std::map<std::uint32_t, MmaPlan> plans;
  std::map<std::uint32_t,TcgenCopyPlan> copy_plans;
  std::vector<TcgenHistory> histories;
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
  const ExecutionProgram *execution = nullptr;
  Resources *resources = nullptr;
  TmemLifetimeAnalysis *tmem = nullptr;
  MatrixResolver *matrix = nullptr;
  const LaneTrace *current_lane = nullptr;
  std::map<std::tuple<std::uint32_t, DynamicKey>,
           std::vector<std::pair<std::uint32_t, const ExecutedInstruction *>>>
      warp_sites;
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
  void Diagnostic(std::string code, std::string message,
                  const Thread *t = nullptr,
                  const ExecutedInstruction *e = nullptr, Json details = Json::object()) {
    ++errors[code];
    Json d{{"code",code},{"message",message},{"stage","event_view_construction"},{"details",std::move(details)}};
    if(e)d["details"]["ptx_line"]=function.instructions[e->instruction_id()].line;
    const auto key=d.dump();
    const auto [it,added]=diagnostic_groups.try_emplace(key,diagnostics.size());
    if(added) {
      d["details"]["occurrence_count"]=0;
      d["details"]["actor_ranges"]=Json::array();
      diagnostics.push_back(std::move(d));
    }
    auto& evidence=diagnostics[it->second]["details"];
    evidence["occurrence_count"]=evidence["occurrence_count"].get<std::size_t>()+1;
    if(t) {
      const auto actor=t->actor.id.value();
      auto& ranges=evidence["actor_ranges"];
      if(!ranges.empty() && ranges.back()[1].get<std::uint32_t>()+1==actor)ranges.back()[1]=actor;
      else if(ranges.empty() || ranges.back()[1]!=actor)ranges.push_back({actor,actor});
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
    if (source.program.instructions.size() >= UINT32_MAX)
      throw SemanticError("model_size", "source instruction IDs exhausted");
    return InstId(source.program.instructions.size());
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
              addresses.bindings.at(address->value()).address_space=="shared") {
            UniquePush(shared_value_writes[site],Ref(id,Milestone::kEffect,index));break;
          }
  }
  InstId Emit(Thread &t, const ExecutedInstruction &e, SourceOp op) {
    const auto &i = function.instructions[e.instruction_id()];
    auto id = source.append(t.actor.id, Location(i), std::move(op),
                            PtxOrigin{source_name, i.id,
                                      static_cast<std::uint32_t>(e.occurrence()),
                                      origin_opcodes[i.id]});
    if(!shared_value_instructions.empty() && shared_value_instructions.at(e.instruction_id()))
      SharedValueEndpoint(t.actor,e.Key(),id,source.instruction(id).op);
    if (index_sites) {
      t.origins.try_emplace(e.Key(), id);
      sites[Key(t.actor.warp.value(), e)].push_back(id);
    }
    return id;
  }
  InstId Async(Thread &t, const ExecutedInstruction &e,
               std::vector<AsyncEffect> effects) {
    return Emit(t, e, AsyncOp{AsyncOpId(Next().value()), std::move(effects)});
  }
  std::uint64_t Scalar(const ExecutedInstruction &e, std::string_view s,
                       std::string_view field, std::uint64_t max = UINT64_MAX) {
    return Unsigned(resources->ScalarValue(e, s, field), field, max);
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
  bool MemoryRegisters(Thread &, const ExecutedInstruction &, AddressId, std::uint32_t, bool);
  void Compute(Thread &, const ExecutedInstruction &);
  void WarpData(Thread &, const ExecutedInstruction &);
  void Collective(Thread &, const ExecutedInstruction &, std::string_view);
  void Mbarrier(Thread &, const ExecutedInstruction &, const Json &);
  void Tcgen(Thread &, const ExecutedInstruction &, const Json &);
  void Lifetime(Thread &, const ExecutedInstruction &);
  void LifetimeOrders();
  void CollectGroup2();
  void Matrix(Thread &, const ExecutedInstruction &, bool);
  TmemAddress ResolveTmem(Thread &, const ExecutedInstruction &,
                          std::string_view);
  void Mma(Thread &, const ExecutedInstruction &, const Json &);
  InstId Copy(Thread &, const ExecutedInstruction &, std::vector<Fragment>,
              std::vector<Fragment>,
              std::optional<CompletionAction> completion = {});
  void Instruction(Thread &, const ExecutedInstruction &);
  void Add(std::uint32_t, Coordinates, const ExecutionProgram &, const ValueAddresses&);
  LoweringResult Finish();
};
// Lift only a memory instruction that intersects pending register accesses.
// Dependency edges target its individual operand reads, never the producer's
// aggregate completion or the entire outstanding TMEM load/store queue.
bool ModelLowerer::Impl::MemoryRegisters(Thread &t, const ExecutedInstruction &e,
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
void ModelLowerer::Impl::Compute(Thread &t, const ExecutedInstruction &e) {
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
void ModelLowerer::Impl::WarpData(Thread &t, const ExecutedInstruction &e) {
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
void ModelLowerer::Impl::Collective(Thread &t, const ExecutedInstruction &e,
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
    auto [it, added] = warp_collectives.try_emplace(
        key, SyncId(source.program.collectives.size()));
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
      source.program.collectives.push_back({it->second, std::move(participants),
                                            static_cast<std::uint32_t>(mask)});
    }
    Emit(t, e, SyncWarp{it->second, static_cast<std::uint32_t>(mask), operation == "warp_sync"});
    return;
  }
  if (operation == "cta_sync" || operation == "named_arrive" ||
      operation == "named_reduce") {
    bool reduced = operation == "named_reduce";
    auto slot = Scalar(e, ops.at(reduced ? 1 : 0), "named slot", 15);
    auto expected =
        ops.size() == (reduced ? 4u : 2u)
            ? Scalar(e, ops.at(reduced ? 2 : 1), "named count", UINT32_MAX)
            : execution->config.Threads();
    if (!expected || expected % 32 || expected > execution->config.Threads())
      throw SemanticError("invalid_named_barrier",
                          "invalid named barrier thread count");
    const auto barrier = barriers.Intern("named", resources->coordinates, std::to_string(slot));
    if (operation == "named_arrive")
      Emit(t, e, NamedBarrierArrive{barrier, static_cast<std::uint32_t>(expected)});
    else
      Emit(t, e, NamedBarrierSync{barrier, static_cast<std::uint32_t>(expected)});
    return;
  }
  if (operation == "cluster_arrive" || operation == "cluster_wait") {
    auto d = execution->config.cluster_dim.value_or(Coordinates{1, 1, 1});
    Coordinates origin;
    for (int k = 0; k < 3; ++k)
      origin[k] = resources->coordinates[k] - resources->coordinates[k] % d[k];
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
           std::uint64_t(d[0]) * d[1] * d[2] * execution->config.Threads()});
      ++cluster_phases[t.actor.id];
    }
    auto id = sites.at(Key(warp, e)).back();
    if (Has(function.instructions[e.instruction_id()],"aligned")) aligned[Key(warp, e)].calls[lane] = {id, id};
    cluster_warp_rendezvous[{warp,std::string(operation),phase}][t.actor.id]=id;
    return;
  }
  throw SemanticError("unsupported_collective", "unknown collective");
}
void ModelLowerer::Impl::Mbarrier(Thread &t, const ExecutedInstruction &e,
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
    const auto address=resources->Memory(e,brackets[0],Json::object(),8,"shared");
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
      auto origin = execution->RegisterOrigin(e, ops.at(2));
      if (!origin || !t.origins.contains(*origin))
        throw SemanticError("unresolved_state_token",
                            "state wait has no retained arrival token");
      const auto &op = source.instruction(t.origins.at(*origin)).op;
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
                                            const ExecutedInstruction &e,
                                            std::string_view operand) {
  if (!tmem)
    throw SemanticError("unresolved_tmem_pointer",
                        "TMEM provenance unavailable");
  for (const auto &issue : tmem->issues)
    if (issue.code != "tmem_multiple_dealloc")
      throw SemanticError(issue.code, issue.message);
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
void ModelLowerer::Impl::Mma(Thread &t, const ExecutedInstruction &e,
                             const Json &contract) {
  (void)contract;
  ProfileScope profile("frontend.model.mma");
  const auto &i = function.instructions[e.instruction_id()];
  const auto &ops = operands[i.id];
  const bool sparse = Has(i, "sp"), scaled = Has(i, "block_scale");
  const bool nvfp4 = scaled && Has(i, "kind::mxf4nvf4");
  const bool mxf4 = scaled && Has(i, "kind::mxf4");
  const bool packed_fp4 = nvfp4 || mxf4;
  if ((sparse ? (ops.size() != 7 && ops.size() != 8) : (ops.size() < 5 || ops.size() > 7)) || Has(i, "ws") ||
      Has(i, "ashift") ||
      std::any_of(i.modifiers.begin(), i.modifiers.end(),
                  [](const auto &m) { return m.starts_with("collector::"); }))
    throw SemanticError("tcgen_mma_variant",
                        "MMA variant has additional operand semantics");
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
  Resources::BitsResolver scale_bits = [&](const ExecutedInstruction &record, std::string_view token,
                                           unsigned) -> std::optional<Resources::KnownBits> {
    if (!tmem || !scaled) return {};
    auto value = tmem->provenance.Resolve(t.actor.warp.value(), record, token, t.actor.lane);
    const auto *origin = execution->RegisterOrigin(record, token);
    const auto *current = execution->RegisterOrigin(e, token);
    if (std::holds_alternative<std::monostate>(value) && origin && current && *origin == *current)
      value = tmem->provenance.Resolve(t.actor.warp.value(), e, token, t.actor.lane);
    if (auto pointer = std::get_if<TmemPointer>(&value))
      return Resources::KnownBits{pointer->encoded_offset & 0xc0000000U, 0xc0000000U};
    return {};
  };
  const auto layout_mask = UINT32_MAX & ~((1U << 13) | (1U << 14) | (sparse ? 3U : 0U));
  const auto [fixed, known] = resources->FixedBits(e, ops[descriptor_index], 32, scale_bits);
  const auto idesc = (known & layout_mask) == layout_mask ? fixed & layout_mask :
      Scalar(e, ops[descriptor_index], "MMA descriptor");
  auto desc = scaled ? DecodeBlockScaledDescriptor(idesc, kinds[0]) : DecodeMmaDescriptor(kinds[0], idesc);
  if (desc.sparse != sparse)
    throw SemanticError("tcgen_mma_descriptor", "MMA sparsity bit disagrees with opcode");
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
  std::vector<Coordinates> owners{resources->coordinates};
  if (group == 2) {
    auto rank = resources->Rank(resources->coordinates);
    owners = {resources->Target(rank & ~1u), resources->Target(rank | 1u)};
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
    shared_a = Scalar(e, ops[1], "MMA shared A descriptor");
  auto shared_b = Scalar(e, ops[2], "MMA shared B descriptor");
  std::optional<TmemAddress> metadata;
  if (sparse) {
    metadata = ResolveTmem(t, e, ops[3]);
    if (group == 2 && !peer_placement(*metadata))
      throw SemanticError("tmem_group2_relative_peer_base", "metadata peers require matched placement");
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
          resources->coordinates, t.actor.warp.value(), t.actor.lane,
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
      Resources::BitsResolver allocation_bits = [&](const ExecutedInstruction &record,
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
  for (auto target : plan.writes) {
    const auto &write=plan.effects[target].accesses.back();
    const auto &binding=addresses.bindings.at(std::get<AddressId>(write.endpoint).value());
    auto key=std::tuple{group,binding.address_space,binding.alias_scope.empty()?binding.allocation:binding.alias_scope};
    auto previous=t.mma_operand_reads.find(key);
    if (previous==t.mma_operand_reads.end()) continue;
    for (const auto &[ref,old,size] : previous->second)
      if (size && *size && write.bytes && *write.bytes &&
          (old.allocation!=binding.allocation ||
           std::max(old.offset,binding.offset)<std::min(old.offset+*size,binding.offset+*write.bytes))) {
        UniquePush(plan.effects[target].predecessors,ref); plan.operand_links.emplace_back(target,ref);
      }
  }
  // PTX 9.4: tcgen05-memory-consistency-model-pipelined-instructions.
  for (auto target : plan.a_tmem_reads) {
    const auto &read = plan.effects[target].accesses.front();
    const auto &binding = addresses.bindings.at(std::get<AddressId>(read.endpoint).value());
    const auto key = std::tuple{group, binding.address_space, binding.allocation};
    const auto previous = t.copy_writes.find(key);
    if (previous == t.copy_writes.end() || !binding.expression.empty() || !read.bytes) continue;
    for (const auto &copy : previous->second) {
      if (!copy.any_operand_address &&
          !CopyFeedsOperandBase(copy.operand_address, plan.operand_addresses.at(target).second)) continue;
      if (copy.binding.offset < binding.offset + *read.bytes && binding.offset < copy.binding.offset + copy.bytes) {
        UniquePush(plan.effects[target].predecessors, copy.effect);
        plan.copy_links.emplace_back(target, copy.effect);
      }
    }
  }
  std::string pipeline = std::to_string(group) + ":" +
                         std::to_string(desc.m) + ":" + std::to_string(desc.n) +
                         ":" + d.resource + ":" +
                         std::to_string(d.address);
  if (auto previous = t.pipeline.find(pipeline); previous != t.pipeline.end()) {
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
    const auto &binding=addresses.bindings.at(std::get<AddressId>(read.endpoint).value());
    auto key=std::tuple{group,binding.address_space,binding.alias_scope.empty()?binding.allocation:binding.alias_scope};
    t.mma_operand_reads[key].emplace_back(Ref(id,Milestone::kEffect,index),binding,read.bytes);
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
void ModelLowerer::Impl::Tcgen(Thread &t, const ExecutedInstruction &e,
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
    if (operation == "wait_load")
      std::erase_if(t.writes, [&](const auto &item) {
        return completed.contains(item.second.instruction);
      });
    else {
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
        if (mask & (1u << rank))
          targets.push_back(barriers.Intern("physical", resources->Target(rank),
                                            std::to_string(sh->second)));
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
      std::vector<MilestoneRef> reads;
      for (unsigned part = 0; part < count; ++part) {
        const auto &interval = intervals[count * k + part];
        if (interval.size != size)
          throw SemanticError("tcgen_register_fragment_map", "invalid TMEM register fragment size");
        auto memory = resources->Bind("tensor", address.resource, interval.offset, size);
        if (operation == "load") {
          reads.push_back(Ref(id, Milestone::kEffect, index + part));
          effects.push_back({{MemoryEffect{AccessMode::kRead, memory.first, size}}});
        } else {
          effects.push_back({{MemoryEffect{AccessMode::kWrite, memory.first, size}}, {},
                            {Ref(id, Milestone::kEffect, index)}});
        }
      }
      if (operation == "load") {
        effects.push_back({{MemoryEffect{AccessMode::kWrite, reg, 4}}, {}, std::move(reads)});
        t.writes.insert_or_assign(regs[k],
                                  Ref(id, Milestone::kEffect, index + count));
      } else
        t.reads[regs[k]].push_back(Ref(id, Milestone::kEffect, index));
    }
    Async(t, e, std::move(effects));
    (operation == "load" ? t.loads : t.stores).Issue(id);
    auto tail = Emit(t, e, Await{});
    aligned[key].calls[t.actor.lane] = {id, tail};
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
    std::vector<std::string> shapes;
    for (const auto &m : i.modifiers)
      if (m == "128x128b" || m == "128x256b" || m == "4x256b" || m == "32x128b")
        shapes.push_back(m);
    const bool multicast = Has(i, "warpx4");
    if (ops.size() != 2 || shapes.size() != 1 ||
        (shapes.front() == "32x128b") != multicast ||
        Has(i, "b8x16") || Has(i, "b6x16_p32") || Has(i, "b4x16_p64") ||
        std::any_of(i.modifiers.begin(), i.modifiers.end(), [](const auto &m) {
          return m.starts_with("warpx") && m != "warpx4";
        }))
      throw SemanticError("tcgen_cp_variant", "copy format needs a region mapping");
    auto shape = shapes.front();
    const std::uint64_t bytes = shape == "128x256b" ? 4096 :
        shape == "128x128b" ? 2048 : shape == "32x128b" ? 512 : 128;
    if (multicast) shape += ".warpx4";
    const auto target = ResolveTmem(t, e, ops[0]);
    const auto descriptor = Scalar(e, ops[1], "copy shared descriptor");
    std::vector<Coordinates> owners{resources->coordinates};
    if (group == 2) {
      if (target.columns != 512)
        throw SemanticError("tmem_group2_relative_peer_base", "copy peers require matched placement");
      const auto rank = resources->Rank(resources->coordinates);
      owners = {resources->Target(rank & ~1U), resources->Target(rank | 1U)};
    }
    std::optional<std::vector<std::pair<ByteInterval, ByteInterval>>> fragments;
    try { fragments = DecodeCopyFragments(descriptor, target.address, shape, target.columns); }
    catch (const LayoutError &) {}
    std::vector<std::pair<Fragment, Fragment>> pairs;
    for (auto owner : owners) {
      if (fragments) {
        for (const auto &[read, write] : *fragments)
          pairs.push_back({resources->Bind("shared", "physical.shared.cta.0", read.offset, read.size, owner),
                           resources->Bind("tensor", target.resource, write.offset, write.size, owner)});
      } else {
        pairs.push_back({resources->Abstract("shared", "physical.shared.cta.0",
             {{"operator", "tcgen_cp_shared_region"}, {"operands", Json::array({descriptor, shape})}}, bytes, owner),
            resources->Abstract("tensor", target.resource,
             {{"operator", "tcgen_cp_tensor_region"}, {"operands", Json::array({target.address, shape})}},
             bytes * (multicast ? 4 : 1), owner)});
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
                       shapes.front() == "4x256b", {}};
    for (std::uint32_t index = 1; index < effects.size(); index += 2) {
      const auto &write = effects[index].accesses.front();
      const auto &binding = addresses.bindings.at(std::get<AddressId>(write.endpoint).value());
      const auto key = std::tuple{static_cast<unsigned>(group), binding.address_space,
                                 binding.alias_scope.empty() ? binding.allocation : binding.alias_scope};
      if (const auto previous = t.mma_operand_reads.find(key); previous != t.mma_operand_reads.end())
        for (const auto &[ref, old, size] : previous->second)
          if (size && write.bytes && old.expression.empty() && binding.expression.empty() &&
              old.allocation == binding.allocation &&
              old.offset < binding.offset + *write.bytes && binding.offset < old.offset + *size) {
            UniquePush(effects[index].predecessors, ref);
            plan.operand_links.emplace_back(index, ref);
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
void ModelLowerer::Impl::Lifetime(Thread &t, const ExecutedInstruction &e) {
  ProfileScope profile("frontend.model.lifetime");
  if (!tmem)
    throw SemanticError("tmem_allocation_identity", "no lifetime analysis");
  for (const auto &issue : tmem->issues)
    if (issue.code != "tmem_multiple_dealloc")
      throw SemanticError(issue.code, issue.message);
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
        if (std::holds_alternative<AsyncOp>(source.instruction(id).op))
          from.push_back(id);
    std::set<ActorId> seen;
    if (sites.contains(target_key))
      for (auto id : sites.at(target_key))
        if (seen.insert(source.instruction(id).actor).second)
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
        auto name =
            "cta:" + CoordinateText(resources->coordinates) + ":tensor:" +
            (allocation->columns == 512 ? std::string("physical.tmem.cta.0")
                                        : obligation.allocation.ResourceName());
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
                  const auto &binding = addresses.bindings.at(p->value());
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
InstId ModelLowerer::Impl::Copy(Thread &t, const ExecutedInstruction &e,
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
    const auto binding = addresses.bindings.at(a.value());
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
      auto shifted =
          addresses.Intern(binding.address_space, binding.allocation,
                           binding.offset + *it - cursor, binding.alias_scope);
      if (!binding.expression.empty()) {
        auto expression = Json::parse(binding.expression);
        expression["address"]["offset"] = {{"operator", "add"},
            {"operands", Json::array({expression["address"]["offset"], *it - cursor})}};
        expression["byte_count"] = *next - *it;
        const auto encoded = expression.dump();
        shifted = addresses.Intern(binding.address_space, "expression:" + encoded, 0, binding.alias_scope);
        auto &target = addresses.bindings.at(shifted.value());
        target.allocation = binding.allocation;
        target.expression = encoded;
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
void ModelLowerer::Impl::Instruction(Thread &t, const ExecutedInstruction &e) {
  const auto &i = function.instructions[e.instruction_id()];
  const auto &s = i.semantics;
  const auto &ops = operands[i.id];
  const auto &operation = s.operation;
  const auto &contract = execution->config.InstructionContract(
      i.line, i.id, t.actor.warp.value(), e.occurrence());
  if (i.opcode=="tensormap" && s.modeled) {
    if (operation=="tensor_map_replace") {
      const auto target=resources->Memory(e,ops.at(0),Json::object(),128,"shared");
      Emit(t,e,ImmediateLoad{target.first,128});
      Emit(t,e,ImmediateStore{target.first,128});
    } else {
      if (Scalar(e,ops.at(2),"tensor map copy size")!=128)
        throw SemanticError("tensor_map_size","tensor map copy requires 128 bytes");
      auto fragment=[&](std::string_view operand,std::string_view space) -> Fragment {
        const auto access=resources->Memory(e,operand,Json::object(),128,std::string(space));
        const auto binding=addresses.bindings.at(access.first.value());
        if (!binding.expression.empty()) throw SemanticError("tensor_map_fragment","tensor map copy requires constant region");
        return {addresses.Intern(binding.address_space,binding.allocation,binding.offset+t.actor.lane*4),4};
      };
      const auto head=Copy(t,e,{fragment(ops.at(1),"shared")},{fragment(ops.at(0),"global")});
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
    return;
  }
  if (e.canonical_wait_probe())
    return;
  if (i.predicate && !e.predicate_value())
    throw SemanticError("unresolved_instruction_predicate",
                        "reached instruction predicate unresolved");
  const bool ordinary_memory = s.category=="memory" && s.modeled && !s.atomic_add &&
      (s.memory_mode=="read" || s.memory_mode=="write") && !i.address_lanes &&
      s.memory_space!="param" && s.memory_space!="const";
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
                               {"producer_line",source.instruction(t.writes.at(name).instruction).location.line},
                               {"dependency","pending_register_write"}});
      }
    }
    for (const auto &reg : Destinations(i))
      if (t.reads.contains(reg))
        throw SemanticError("async_register_overwrite_unsupported",
                            "Register overwrite during an asynchronous read is not modeled for this instruction",
                            {{"register",reg},{"consumer_opcode",Opcode(i)},
                             {"reader_line",source.instruction(t.reads.at(reg).front().instruction).location.line},
                             {"dependency","pending_register_read"}});
  }
  if (s.atomic_add && (s.category == "memory" || !s.modeled))
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
        resources->Memory(e, brackets[0], contract, *i.memory_width, space);
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
    auto source_size = size;
    if (ops.size() == 4) {
      auto value = Scalar(e, ops[3], "source size/ignore source");
      auto reg = Register(i, ops[3]);
      bool immediate = ParseInteger(ops[3]).has_value();
      if (!immediate &&
          (!ops[3].starts_with('%') || ambiguous_registers.contains(ops[3])))
        throw SemanticError("cp_async_operand_scope",
                            "size register lacks unambiguous scope");
      if (!immediate && registers.Width(reg) == 1) {
        if (value > 1)
          throw SemanticError("cp_async_ignore_source",
                              "ignore source must be predicate");
        source_size = value ? 0 : size;
      } else {
        if (!immediate && registers.Width(reg) != 4)
          throw SemanticError("cp_async_operand_type",
                              "source size needs 32-bit register");
        source_size = value;
      }
    }
    if (source_size > size)
      throw SemanticError("cp_async_source_size",
                          "source size exceeds copy size");
    std::vector<Fragment> reads;
    if (source_size)
      reads.push_back(resources->Memory(
          e, ops[1],
          AccessContract(contract, "source_resource", "source_region"),
          source_size, "global"));
    std::vector<Fragment> writes{resources->Memory(
        e, ops[0],
        AccessContract(contract, "destination_resource", "destination_region"),
        size, "shared")};
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
    auto transfer = resources->Tma(e, contract, operation == "tma_bulk_tensor");
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
      effects.push_back(
          {{}, CompleteTxEffect{barrier, size}, destinations.at(ranks[k])});
    }
    Async(t, e, std::move(effects));
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
    if (bytes%16) throw SemanticError("bulk_reduction_variant","reduction size must be a u32 multiple of 16");
    const auto element=std::stoul(type.substr(type.find_first_of("0123456789")))/8;
    const auto read=resources->Memory(e,brackets[1],Json::object(),bytes,"shared");
    const auto write=resources->Memory(e,brackets[0],Json::object(),bytes,"global");
    if (read.second!=bytes || write.second!=bytes) throw SemanticError("bulk_reduction_fragmentation","nonlinear reduction footprint");
    const auto read_base=addresses.bindings[read.first.value()], write_base=addresses.bindings[write.first.value()];
    auto fragment=[&](const ConstantAddressBinding &base,std::uint64_t offset) {
      if (base.expression.empty()) {
        if (base.offset%16) throw SemanticError("bulk_reduction_alignment","reduction base must be 16-byte aligned");
        return addresses.Intern(base.address_space,base.allocation,base.offset+offset,base.alias_scope);
      }
      auto expression=Json::parse(base.expression);
      auto old=expression["address"]["offset"];
      if (offset) expression["address"]["offset"]={{"operator","add"},{"operands",Json::array({old,offset})}};
      expression["byte_count"]=element;
      auto address=addresses.Intern(base.address_space,"expression:"+expression.dump(),0,base.alias_scope);
      addresses.bindings[address.value()].allocation=base.allocation;
      addresses.bindings[address.value()].expression=expression.dump(); return address;
    };
    auto id=Next(); std::vector<AsyncEffect> effects;
    for (std::uint64_t offset=0;offset<bytes;offset+=element) {
      const auto index=effects.size();
      effects.push_back({{MemoryEffect{AccessMode::kRead,fragment(read_base,offset),element}}});
      effects.push_back({{MemoryEffect{AccessMode::kWrite,fragment(write_base,offset),element,
          scope=="cta"?AtomicScope::kCta:scope=="gpu"?AtomicScope::kGpu:AtomicScope::kSys}},
          {},{Ref(id,Milestone::kEffect,index)}});
    }
    Async(t,e,std::move(effects)); t.bulk.Issue(id); return;
  }
  if (operation == "bulk_complete_tx") {
    std::set<std::string> allowed{"async",
                                  "bulk",
                                  "weak",
                                  "shared::cta",
                                  "shared::cluster",
                                  "global",
                                  "mbarrier::complete_tx::bytes",
                                  "L2::cache_hint"},
        seen;
    std::vector<std::string> spaces;
    for (const auto &m : i.modifiers) {
      if (!allowed.contains(m) || !seen.insert(m).second)
        throw SemanticError("bulk_copy_variant",
                            "unsupported bulk copy modifier");
      if (m == "shared::cta" || m == "shared::cluster" || m == "global")
        spaces.push_back(m);
    }
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
    auto size = Scalar(e, ops[2], "bulk transfer size");
    if (!size || size % 16 || size>UINT32_MAX)
      throw SemanticError("invalid_bulk_transfer_size",
                          "size must be positive multiple of16");
    if (spaces[1]=="shared::cta") {
      auto destination=resources->Shared(e,b[0]);
      if (!destination || destination->first==resources->coordinates)
        throw SemanticError("bulk_copy_variant","shared bulk copy requires a resolved remote destination");
    }
    std::vector<Fragment> reads{resources->Memory(
        e, b[1], AccessContract(contract, "source_resource", "source_region"),
        size, spaces[1])},
        writes{
            resources->Memory(e, b[0],
                              AccessContract(contract, "destination_resource",
                                             "destination_region"),
                              size, spaces[0])};
    if (reads[0].second != size || writes[0].second != size)
      throw SemanticError("bulk_copy_fragment_size",
                          "bulk footprints disagree with size");
    Copy(t, e, std::move(reads), std::move(writes),
         CompleteTxEffect{resources->Barrier(e, b[2], contract), size});
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
    auto d = execution->config.cluster_dim.value_or(Coordinates{1, 1, 1});
    if (!dest || !complete || std::uint64_t(d[0]) * d[1] * d[2] <= 1 ||
        dest->first != complete->first)
      throw SemanticError("st_async_owner",
                          "store needs same destination and barrier owner in "
                          "multi-CTA cluster");
    auto write = resources->Memory(e, b[0], contract, *i.memory_width,
                                   "shared::cluster");
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
    if (execution->config.fixed_input.value("cluster_launch_cancellation",
                                            true))
      throw SemanticError("cluster_launch_control_outcome",
                          "CLC needs fixed no-cancellation scenario");
    auto d = execution->config.cluster_dim.value_or(Coordinates{1, 1, 1});
    Coordinates origin;
    for (int k = 0; k < 3; ++k)
      origin[k] = resources->coordinates[k] - resources->coordinates[k] % d[k];
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
    std::vector<Coordinates> owners{resources->coordinates};
    if (Has(i, "multicast::cluster::all")) {
      owners.clear();
      for (std::uint32_t rank = 0; rank < d[0] * d[1] * d[2]; ++rank)
        owners.push_back(resources->Target(rank));
    }
    auto id = Next();
    std::vector<AsyncEffect> effects;
    for (auto owner : owners) {
      auto a = resources->Bind("shared", "physical.shared.cta.0",
                               response->second, 16, owner);
      auto target =
          barriers.Intern("physical", owner, std::to_string(barrier->second));
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
      auto own_rank = resources->Rank(resources->coordinates);
      std::set<std::string> names;
      for (auto rank : {own_rank & ~1u, own_rank | 1u})
        names.insert("cta:" + CoordinateText(resources->Target(rank)) +
                     ":tensor:" + resource);
      bool has_tensor = false, matched = false;
      for (std::uint32_t index = 0; index < op->effects.size(); ++index) {
        bool match = false;
        for (const auto &access : op->effects[index].accesses)
          if (access.bytes)
            if (auto memory = std::get_if<AddressId>(&access.endpoint)) {
              const auto &binding = addresses.bindings.at(memory->value());
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
  for (const auto &lane : execution->lane_traces)
    for (std::size_t index=0;index<lane.instructions.size();++index) {
      const auto &e=lane.instructions.metadata(index);
      const auto &instruction = function.instructions[e.instruction_id()];
      if (instruction.opcode == "tcgen05" && Has(instruction, "cta_group::2") &&
          Has(instruction, "relinquish_alloc_permit") &&
          seen.insert(Key(lane.warp_id, e)).second)
        records.emplace_back(TmemAtomSite{lane.warp_id, e.instruction_id(),
                                          e.occurrence(), instruction.line,
                                          opcode_texts[instruction.id]},
                             "relinquish_alloc_permit", std::nullopt);
    }
  for (const auto &[site, operation, allocation_id] : records) {
    Group2LifetimeSite record;
    record.cta_coordinates = resources->coordinates;
    record.operation = operation;
    record.instruction_index = site.instruction_index;
    record.occurrence = site.occurrence;
    record.warp = site.warp;
    record.allocation_key = allocation_id;
    std::map<ActorId, std::pair<InstId, InstId>> ends;
    auto found = sites.find(Key(site));
    if (found != sites.end())
      for (auto id : found->second) {
        auto actor = source.instruction(id).actor;
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
    group2_sites[execution->config.cluster_dim.value_or(Coordinates{1, 1, 1})]
        .push_back(std::move(record));
  }
}
void ModelLowerer::Impl::Matrix(Thread &t, const ExecutedInstruction &e,
                                bool load) {
  ProfileScope profile("frontend.model.matrix");
  const auto &i = function.instructions[e.instruction_id()];
  const auto &contract = execution->config.InstructionContract(
      i.line, i.id, t.actor.warp.value(), e.occurrence());
  auto fragments = matrix->Resolve(*current_lane, e, *resources, contract);
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
void ModelLowerer::Impl::Add(std::uint32_t cta, Coordinates coordinates,
                             const ExecutionProgram &p, const ValueAddresses& value_addresses) {
  ProfileScope profile("frontend.model.cta");
  execution = &p;
  current_coordinates=coordinates;
  shared_value_sites.clear();
  shared_value_instructions.assign(function.instructions.size(),false);
  for(const auto &site:p.shared_constant_writes)shared_value_sites.insert(site);
  for(const auto &dependency:p.shared_value_dependencies)shared_value_sites.insert(dependency.read);
  for(const auto &site:shared_value_sites)shared_value_instructions.at(site.instruction.instruction)=true;
  shared_value_dependencies.insert(shared_value_dependencies.end(),p.shared_value_dependencies.begin(),p.shared_value_dependencies.end());
  Resources r(p, coordinates, addresses, barriers);
  r.value_addresses = value_addresses;
  resources = &r;
  MatrixResolver matrix_resolver(p);
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
    analysis = AnalyzeTmemLifetimes(p, options);
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
      if (issue.code == "tmem_multiple_dealloc")
        violations.emplace(
            issue.code, "CTA " + CoordinateText(coordinates) + ": " +
                            issue.message + "; first deallocation at warp " +
                            std::to_string(issue.site.warp) + ", instruction " +
                            std::to_string(issue.site.instruction_index) +
                            ", occurrence " +
                            std::to_string(issue.site.occurrence));
    if (tmem->cluster_pairing_required)
      limitations["tmem_group2_pairing"] =
          "Group2 lifetime requires a cluster participation and progress "
          "certificate.";
  }
  auto actor_begin = source.program.actors.size();
  std::vector<std::uintptr_t> member_cohorts(p.lane_traces.size(),0);
  if(p.template_execution.used) {
    for(std::size_t group=0;group<p.thread_template_groups.size();++group)
      for(auto member:p.thread_template_groups[group]) {
        if(member>=member_cohorts.size() || member_cohorts[member])
          throw SemanticError("invalid_thread_partition","template members must form a partition");
        member_cohorts[member]=group+1;
      }
    if(std::find(member_cohorts.begin(),member_cohorts.end(),0)!=member_cohorts.end())
      throw SemanticError("invalid_thread_partition","template partition does not cover every thread");
  }
  for (std::size_t k = 0; k < p.lane_traces.size(); ++k) {
    const auto &lane = p.lane_traces[k];
    ActorInfo actor{ActorId(actor_begin + k), CtaId(cta), WarpId(lane.warp_id),
                    lane.lane_id};
    if (!actors.emplace(std::pair{lane.warp_id, lane.lane_id}, actor.id).second)
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
  std::map<const void*, Projection> projections;
  const bool synchronous_template=p.template_execution.used &&
      std::all_of(function.instructions.begin(),function.instructions.end(),[](const auto &i) {
        const auto &s=i.semantics;
        if(IsWarpDataCollective(i))return false;
        return s.category=="compute" || s.category=="control" ||
            (s.category=="memory" && s.modeled && (s.memory_mode=="read" || s.memory_mode=="write"));
      });
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
    const auto reg=std::find(p.register_names.begin(),p.register_names.end(),base);
    if(reg!=p.register_names.end())memory_registers[i.id]=reg-p.register_names.begin();
    SourceOp operation=i.semantics.memory_mode=="read"
      ?SourceOp(ImmediateLoad{AddressId{},static_cast<std::uint32_t>(*i.memory_width)})
      :SourceOp(ImmediateStore{AddressId{},static_cast<std::uint32_t>(*i.memory_width)});
    prototypes->at(i.id)={InstId{},ActorId{},Location(i),std::move(operation),
        PtxOrigin{source_name,i.id,0,origin_opcodes[i.id]}};
  }
  std::unordered_map<std::string_view,std::uint32_t> register_ids;
  register_ids.reserve(p.register_names.size());
  for(std::uint32_t id=0;id<p.register_names.size();++id)
    register_ids.emplace(p.register_names[id],id);
  std::vector<std::optional<std::uint32_t>> collective_masks(function.instructions.size());
  for(const auto& i:function.instructions) {
    if(!IsWarpDataCollective(i) || operands[i.id].empty())continue;
    const auto found=register_ids.find(Trim(operands[i.id].back()));
    if(found!=register_ids.end())collective_masks[i.id]=found->second;
  }
  std::uint64_t visits = 0;
  for (std::size_t k = 0; k < p.lane_traces.size(); ++k) {
    const auto &lane = p.lane_traces[k];
    const auto binding_begin=p.template_execution.used ? source.program.instructions.beginThread() : 0;
    current_lane = &lane;
    ProfileCall("frontend.resources.index_lane", [&] { r.SetLane(lane); });
    Thread thread;
    thread.actor = source.program.actors[actor_begin + k];
    Projection* projection=nullptr;
    if (synchronous_template) {
      auto [entry,inserted]=projections.try_emplace(lane.instructions.metadataIdentity());
      auto &view=entry->second;
      if (inserted) {
        view.skipped_counts.resize(function.instructions.size());
        view.memory_counts.resize(function.instructions.size());
        if(generated_memory) {
          view.body=std::make_shared<std::vector<std::pair<std::uint32_t,std::uint32_t>>>();
          view.bindings=std::make_shared<AddressBindings>();
        }
        for (std::size_t index=0;index<lane.instructions.size();++index) {
          if ((index&1023)==0) Poll();
          const auto &record=lane.instructions.metadata(index);
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
        const auto instruction=lane.instructions.metadata(index).instruction_id();
        const auto& reg=memory_registers[instruction];
        const auto& metadata=lane.instructions.metadata(index);
        const auto& static_instruction=function.instructions[instruction];
        const auto& contract=p.config.InstructionContract(static_instruction.line,instruction,lane.warp_id,metadata.occurrence());
        const auto evaluation=reg && contract.empty()?lane.instructions.registerEvaluationIdentity(index,*reg):std::nullopt;
        const auto evaluation_key=evaluation && *evaluation<=UINT32_MAX?
            std::optional<std::uint64_t>((std::uint64_t(*evaluation)<<32)|instruction):std::nullopt;
        if(evaluation_key)if(const auto cached=evaluated_addresses.find(*evaluation_key);cached!=evaluated_addresses.end()) {
          resolved.push_back(cached->second);continue;
        }
        const bool uniform_global=static_instruction.semantics.memory_space==std::optional<std::string>("global") &&
            contract.empty() && reg && std::any_of(metadata.known_registers().begin(),metadata.known_registers().end(),
                [&](const auto& binding){return binding.first==*reg;});
        if(uniform_global)if(const auto cached=projection->uniform_global_addresses.find(index);
            cached!=projection->uniform_global_addresses.end()) {
          resolved.push_back(cached->second);continue;
        }
        const auto e=reg ? lane.instructions.readRegisters(index,std::span(&*reg,1))
                         : lane.instructions.read(index);
        const auto& i=function.instructions[e.instruction_id()];
        try {
          if((i.predicate && !e.predicate_value()) || e.canonical_wait_probe() || memory_tokens[i.id].empty()) {
            supported=false;break;
          }
          const auto [address,bytes]=r.Memory(e,memory_tokens[i.id],contract,*i.memory_width,
              i.semantics.memory_space.value_or("generic"));
          if(bytes!=i.memory_width){supported=false;break;}
          resolved.push_back(address);
          if(evaluation_key && evaluated_addresses.size()<100000 &&
              addresses.bindings.at(address.value()).expression.empty())
            evaluated_addresses.emplace(*evaluation_key,address);
          if(uniform_global && addresses.bindings.at(address.value()).expression.empty())
            projection->uniform_global_addresses.emplace(index,address);
        } catch(const std::exception&) {supported=false;break;}
      }
      if(supported) {
        if(!projection->uniform_body && p.config.instruction_contracts.empty()) {
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
    const auto count=projection ? projection->positions.size() : lane.instructions.size();
    for (std::size_t ordinal=0; ordinal<count; ++ordinal) {
      const auto index=projection ? projection->positions[ordinal] : ordinal;
      const auto &metadata=lane.instructions.metadata(index);
      const auto &semantics=function.instructions[metadata.instruction_id()].semantics;
      std::optional<ExecutedInstruction> concrete;
      if (!lane.instructions.materialized()) {
        const auto& mask=collective_masks[metadata.instruction_id()];
        if(mask)
          concrete=lane.instructions.readRegisters(index,std::span(&*mask,1));
        else if(semantics.category!="compute" && semantics.category!="control")
          concrete=lane.instructions.read(index);
      }
      const auto &e=concrete ? *concrete : metadata;
      if (e.predicate_value() == false)
        continue;
      if ((++visits & 1023) == 0)
        Poll();
      const auto &i = function.instructions[e.instruction_id()];
      ++occurrence_counts[i.id];
      try {
        Instruction(thread, e);
      } catch (const SemanticError &error) {
        Diagnostic(error.code, error.what(), &thread, &e, error.details);
      } catch (const LayoutError &error) {
        Diagnostic(error.code, error.what(), &thread, &e, error.details);
      } catch (const std::bad_alloc &) {
        throw;
      } catch (const std::exception &error) {
        Diagnostic("unexpected_exception", error.what(), &thread, &e,
                   {{"exception_type",typeid(error).name()}});
      }
    }
    histories.push_back(std::move(thread.history));
    if(p.template_execution.used)
      source.program.instructions.extendThreadFamily(binding_begin,cta,member_cohorts[k]);
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
  std::map<ActorId,InstId> terminals;
  for (const auto &[key,entries] : cluster_warp_rendezvous) {
    const auto warp=std::get<0>(key);
    std::vector<MilestoneRef> requirements;
    for (const auto &[actor,id] : entries) {
      (void)actor;
      requirements.push_back(Ref(id,Milestone::kIssue));
    }
    for (std::size_t k=0;k<p.lane_traces.size();++k) {
      const auto &lane=p.lane_traces[k];
      if (lane.warp_id!=warp) continue;
      const auto actor=source.program.actors[actor_begin+k];
      if (entries.contains(actor.id)) continue;
      if (!terminals.contains(actor.id)) {
        if (lane.instructions.empty() ||
            (function.instructions[lane.instructions.metadata(lane.instructions.size()-1).instruction_id()].control!="return" &&
             function.instructions[lane.instructions.metadata(lane.instructions.size()-1).instruction_id()].control!="exit")) {
          Diagnostic("incomplete_cluster_warp","warp peer has no cluster operation or terminal return"); continue;
        }
        Thread terminal; terminal.actor=actor;
        terminals.emplace(actor.id,Emit(terminal,lane.instructions.metadata(lane.instructions.size()-1),Await{}));
      }
      requirements.push_back(Ref(terminals.at(actor.id)));
    }
    for (const auto &[actor,id] : entries) {
      (void)actor;
      auto &target=std::get<Await>(source.editInstruction(id).op).requirements;
      for (auto ref : requirements) UniquePush(target,ref);
    }
  }
  {
    ProfileScope joins("frontend.model.cta_joins");
    for (auto &[site, entry] : aligned) {
      (void)site;
      if (entry.calls.size() != 32) {
        Diagnostic("incomplete_aligned_warp",
                   "aligned instruction is not executed by all32 lanes");
        continue;
      }
      std::vector<std::pair<InstId, InstId>> calls;
      for (std::uint32_t lane = 0; lane < 32; ++lane) {
        auto it = entry.calls.find(lane);
        if (it == entry.calls.end())
          throw SemanticError("incomplete_aligned_warp", "missing lane");
        calls.push_back(it->second);
      }
      source.alignWarp(calls, entry.completions);
    }
  }
  ++ctas;
  profile.nodes = visits;
  SENTRYWARD_DEBUG_PRINT(lowering_0);
  SENTRYWARD_DEBUG_PRINT(lowering_1);
  execution = nullptr;
  resources = nullptr;
  tmem = nullptr;
  matrix = nullptr;
  current_lane = nullptr;
  sites.clear();
  aligned.clear();
  cluster_warp_rendezvous.clear();
  allocations.clear();
  actors.clear();
  warp_collectives.clear();
  Poll();
}
LoweringResult ModelLowerer::Impl::Finish() {
  ProfileScope profile("frontend.model.finalize");
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
        Diagnostic("mbarrier_inval_lifetime","invalidation requires a single initialization and final invalidation"); continue;
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
                 "cluster phase lacks required thread arrivals");
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
      Diagnostic("shared_constant_source_unmodeled","A shared constant's source write or consuming read has no modeled memory event.");
      continue;
    }
    for(auto from:write->second)orders.push_back({"shared_constant_read_from",from,read->second,
        "The write supplying a frontend shared-memory constant must happen before its read; interfering writes are checked by memory analysis."});
  }
  ProfileCall("frontend.model.seal_families",[&]{source.program.instructions.sealThreadFamilies();});
  LoweringResult result;
  result.mma_layouts = std::move(mma_layouts);
  for (std::size_t i = 0; i < occurrence_counts.size(); ++i)
    if (occurrence_counts[i])
      counts[opcode_texts[i]] += occurrence_counts[i];
  result.metadata = {
      {"cta_count", ctas},
      {"thread_count", source.program.actors.size()},
      {"cluster_completion_joins", gates.size()},
      {"reached_opcodes", counts},
      {"lowered_instruction_count", source.program.instructions.size()},
      {"unsupported_occurrence_counts", errors},
      {"diagnostic_sample_count", diagnostics.size()},
      {"implementation", "cpp"}};
  result.metadata["shared_constant_propagation"]={{"read_from_dependencies",shared_value_dependencies.size()},
      {"validation","pending_backend"},{"representation","sparse_known_bytes"}};
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
  result.diagnostics = std::move(diagnostics);
  result.artifact.program = std::move(source.program);
  result.artifact.addresses = std::move(addresses.bindings);
  result.artifact.required_orders = std::move(orders);
  for (auto &[code, message] : limitations)
    result.artifact.verification_limitations.push_back({code, message});
  for (const auto &[code, message] : violations)
    result.artifact.source_violations.push_back({code, message});
  return result;
}
ModelLowerer::ModelLowerer(const Module &m, const Function &f,
                           std::function<void()> poll, bool retain_matrix_layout)
    : impl_(std::make_unique<Impl>(m, f, std::move(poll))) {
  impl_->retain_matrix_layout = retain_matrix_layout;
}
ModelLowerer::~ModelLowerer() = default;
void ModelLowerer::AddCta(std::uint32_t cta, Coordinates coords,
                          const ExecutionProgram &program, ValueAddresses value_addresses) {
  impl_->Add(cta, coords, program, value_addresses);
}
const algorithm::SourceProgram &ModelLowerer::currentProgram() const noexcept {
  return impl_->source.program;
}
LoweringResult ModelLowerer::Finish() { return impl_->Finish(); }
} // namespace sentryward::frontend::ptx
