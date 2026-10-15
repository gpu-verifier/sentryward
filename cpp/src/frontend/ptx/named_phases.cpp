#include "sentryward/frontend/ptx/named_phases.h"
#include "sentryward/frontend/ptx/resources.h"
#include <set>
namespace sentryward::frontend::ptx {
NamedPhaseMap VariableCountPhases(const ExecutionProgram &program) {
  struct Call { std::uint32_t slot,count; DynamicKey key; bool blocking; };
  struct Actor { std::uint32_t warp,lane; std::vector<Call> calls; std::size_t position=0; bool waiting=false; };
  std::vector<Actor> actors;
  std::map<std::uint32_t,std::set<std::uint32_t>> counts;
  auto number=[&](const ExecutedInstruction &e,const std::string &text) {
    if (auto value=ParseInteger(text)) return Unsigned(*value,"named barrier operand",UINT32_MAX);
    if (const auto *value=program.KnownRegister(e,text))
      if (auto n=ScalarInteger(*value)) return Unsigned(*n,"named barrier operand",UINT32_MAX);
    if (auto it=program.config.parameters.find(text);it!=program.config.parameters.end())
      return Unsigned(it->second,"named barrier operand",UINT32_MAX);
    throw SemanticError("named_barrier_phase_matching","unresolved counted-barrier operand");
  };
  for (const auto &trace : program.lane_traces) {
    Actor actor{trace.warp_id,trace.lane_id,{}};
    for (std::size_t index=0; index<trace.instructions.size(); ++index) {
      const auto &metadata=trace.instructions.metadata(index);
      const auto &i=program.function->instructions[metadata.instruction_id()]; const auto &op=i.semantics.operation;
      if (op!="cta_sync" && op!="named_arrive" && op!="named_reduce") continue;
      const auto e=trace.instructions.read(index);
      const auto operands=OperandTexts(*program.module,i); const bool reduced=op=="named_reduce";
      const auto slot=number(e,operands.at(reduced?1:0));
      const auto count=operands.size()==(reduced?4U:2U)?number(e,operands.at(reduced?2:1)):program.config.Threads();
      if (!count || count>UINT32_MAX) throw SemanticError("named_barrier_phase_matching","invalid counted-barrier count");
      counts[slot].insert(count); actor.calls.push_back({std::uint32_t(slot),std::uint32_t(count),e.Key(),op!="named_arrive"});
    }
    actors.push_back(std::move(actor));
  }
  std::set<std::uint32_t> variable;
  for (const auto &[slot,values] : counts) if (values.size()>1) variable.insert(slot);
  if (variable.empty()) return {};
  using Member=std::pair<std::size_t,Call>;
  struct Phase { std::uint32_t count; std::vector<Member> members; };
  std::map<std::uint32_t,Phase> arrivals;
  std::map<std::uint32_t,std::uint32_t> generation;
  NamedPhaseMap result;
  for (;;) {
    std::map<std::pair<std::uint32_t,std::uint32_t>,std::vector<Member>> pending;
    std::map<std::uint32_t,std::set<std::uint32_t>> slot_counts;
    for (std::size_t a=0;a<actors.size();++a) {
      const auto &actor=actors[a];
      if (actor.waiting || actor.position>=actor.calls.size()) continue;
      const auto &call=actor.calls[actor.position];
      pending[{call.slot,call.count}].emplace_back(a,call); slot_counts[call.slot].insert(call.count);
    }
    if (pending.empty() && arrivals.empty()) return result;
    for (const auto &[slot,values] : slot_counts)
      if (!arrivals.contains(slot) && values.size()!=1)
        throw SemanticError("named_barrier_phase_matching","counted phases have ambiguous expected counts");
    bool advanced=false;
    for (const auto &[key,members] : pending) {
      const auto [slot,count]=key;
      if (arrivals.contains(slot) && arrivals.at(slot).count!=count) continue;
      auto &phase=arrivals.try_emplace(slot,Phase{count,{}}).first->second;
      phase.members.insert(phase.members.end(),members.begin(),members.end());
      for (const auto &[a,call] : members)
        if (call.blocking) actors[a].waiting=true; else ++actors[a].position;
      advanced=true;
      if (phase.members.size()>count)
        throw SemanticError("named_barrier_phase_matching","counted phase has excess arrivals");
      if (phase.members.size()==count) {
        for (const auto &[a,call] : phase.members) {
          auto &actor=actors[a];
          if (variable.contains(slot)) result[{actor.warp,actor.lane,call.key}]=generation[slot];
          if (call.blocking) { actor.waiting=false; ++actor.position; }
        }
        ++generation[slot]; arrivals.erase(slot);
      }
    }
    if (!advanced) throw SemanticError("named_barrier_phase_matching","counted phases have no unique rendezvous matching");
  }
}
}
