#pragma once
#include "execution_impl.hpp"

namespace sentryward::frontend::ptx {
namespace {
std::optional<ExecutionProgram> Executor::RunTemplates() {
  auto task=RunTemplateTask(nullptr);task.Resume();
  if(!task.Done())throw std::logic_error("noncooperative template task suspended");
  return task.TakeResult();
}

TemplateTask Executor::RunTemplateTask(SharedExecution* runtime) {
  using namespace detail;
  if (!c_.compress_thread_records ||
      !c_.loop_bounds.empty() ||
      !c_.instruction_contracts.empty()) co_return std::nullopt;
  const auto supports_template=[](const Decoded& d) {
    const auto &i=*d.instruction;
    if (!In(i.control,{"none","direct_branch","indirect_branch","return","exit"})) return false;
    if(i.opcode=="vote" && (i.modifiers.size()!=3 || i.modifiers[0]!="sync" ||
        !In(i.modifiers[1],{"ballot","all","any","uni"}) || d.operands.size()!=3))return false;
    if(i.opcode=="shfl" && (i.modifiers.size()<2 || i.modifiers[0]!="sync" ||
        !In(i.modifiers[1],{"idx","up","down","bfly"}) || d.operands.size()!=5))return false;
    if(i.opcode=="match" && (i.modifiers.size()!=3 || !In(i.modifiers[0],{"any","all"}) || i.modifiers[1]!="sync" ||
        d.operands.size()!=3))return false;
    return true;
  };

  std::vector<std::vector<std::uint32_t>> source_registers(decoded_.size());
  for (const auto &d:decoded_) {
    auto &reads=source_registers[d.instruction->id];
    auto add=[&](std::uint32_t reg) {
      if (reg!=kMissing && std::find(reads.begin(),reads.end(),reg)==reads.end()) reads.push_back(reg);
    };
    for (std::size_t k=1;k<d.operands.size();++k) {
      const auto &operand=d.operands[k];
      for (auto reg:operand.registers) add(reg);
      add(operand.reg);
      // Register packs can contain component-qualified special registers.
      for (std::uint32_t reg=0;reg<names_.size();++reg) if (names_[reg].find('.')!=std::string::npos) {
        auto at=operand.text.find(names_[reg]);
        if (at==std::string::npos) continue;
        auto end=at+names_[reg].size();
        if (end==operand.text.size() || (!std::isalnum(static_cast<unsigned char>(operand.text[end])) &&
            operand.text[end]!='_' && operand.text[end]!='.')) add(reg);
      }
    }
  }
  struct Member {std::uint32_t warp,lane;};
  struct Group {
    State state;
    std::vector<std::size_t> registers;
    std::shared_ptr<std::vector<std::size_t>> path=std::make_shared<std::vector<std::size_t>>();
    PathId path_id{}; // the identity of `path` (the root is PathId 0)
    algorithm::SharedArray<std::size_t> members;
    std::uint32_t pc{};
    std::uint64_t steps{};
    std::uint64_t assumed_assertions{};
    std::map<std::uint64_t,std::uint64_t> opaque_regions; // as LaneTrace::opaque_regions, per member
    std::map<std::uint32_t,ProbeValue> probe_values;
    std::map<DynamicKey,std::size_t> probes;
    std::map<std::uint32_t,std::map<std::uint32_t,ProbeValue>> probe_joins;
    struct SharedWork {
      std::vector<std::size_t> pending;
      std::vector<std::vector<Scalar>> outputs;
      bool handled=false;
    };
    std::optional<SharedWork> shared;
    std::vector<std::size_t> collective_results;
    // Join pc of the merged one-armed diamond being executed and the registers its arm
    // writes; they become unknown at the join (straight-line arms never nest).
    std::optional<std::pair<std::uint32_t,std::vector<std::uint32_t>>> merged_join;
    // Every member's .local slots (template values), filled by stores whose address all members share.
    LocalSlots<std::size_t> local;
    // Whether the path executed an opaque region or merged arm, inside which members may diverge.
    bool approximate_control=false;
  };
  auto arena=std::make_shared<TemplateArena>(); arena->poll=poll_;
  arena->Constant(Scalar{});
  std::vector<Member> members;
  auto member_coordinates=std::make_shared<std::vector<std::uint32_t>>();
  for (const auto &warp:c_.warps) {
    if (warp.entry_label) co_return std::nullopt;
    for (std::uint32_t lane=0;lane<c_.warp_size;++lane) {
      auto tid=warp.warp_id*c_.warp_size+lane;
      if (c_.analysis_thread_range && (tid<c_.analysis_thread_range->first || tid>=c_.analysis_thread_range->second)) continue;
      members.push_back({warp.warp_id,lane});member_coordinates->push_back(tid);
    }
  }
  if (members.empty()) co_return std::nullopt;
  TemplateValue thread_coordinate;
  thread_coordinate.member_value=[member_coordinates](std::size_t member)->Scalar{return Integer(member_coordinates->at(member));};
  const auto coordinate_value=arena->values.size();
  arena->values.push_back(std::move(thread_coordinate));
  arena->caches.resize(members.size());
  const auto first_warp=std::find_if(c_.warps.begin(),c_.warps.end(),[&](const auto& warp){return warp.warp_id==members.front().warp;});
  Group first; first.state=Initial(*first_warp,members.front().lane);
  auto default_warp=*first_warp;default_warp.initial_registers.clear();
  const auto default_initial=Initial(default_warp,members.front().lane);
  std::vector<std::size_t> first_members;
  auto member_indices=std::make_shared<std::map<std::uint32_t,std::size_t>>();
  for (std::size_t t=0;t<members.size();++t) {
    first_members.push_back(t);
    member_indices->emplace(members[t].warp*c_.warp_size+members[t].lane,t);
  }
  first.members=std::move(first_members);
  for (std::size_t reg=0;reg<names_.size();++reg) {
    const auto& name=names_[reg];
    unsigned kind=name=="%laneid"?1:name=="%warpid"?2:name=="%tid.x"?3:name=="%tid.y"?4:name=="%tid.z"?5:0;
    if(c_.special_registers.contains(name))kind=0;
    std::map<std::uint32_t,Integer> overrides;
    for(const auto& warp:c_.warps)if(auto found=warp.initial_registers.find(name);found!=warp.initial_registers.end())
      overrides.emplace(warp.warp_id,found->second);
    if(!kind && overrides.empty()) {
      first.registers.push_back(arena->Constant(first.state.registers[reg]));continue;
    }
    Scalar fallback=default_initial.registers[reg];
    TemplateValue node;node.operands={coordinate_value};
    node.evaluate=[kind,overrides=std::move(overrides),fallback,dimensions=c_.block_dim,warp_size=c_.warp_size](const std::vector<Scalar>& args)->Scalar {
      const auto thread=std::get<Integer>(args.at(0)).convert_to<std::uint32_t>();
      if(auto found=overrides.find(thread/warp_size);found!=overrides.end())return found->second;
      switch(kind) {
        case 1:return Integer(thread%warp_size);
        case 2:return Integer(thread/warp_size);
        case 3:return Integer(thread%dimensions[0]);
        case 4:return Integer(thread/dimensions[0]%dimensions[1]);
        case 5:return Integer(thread/(dimensions[0]*dimensions[1]));
        default:return fallback;
      }
    };
    arena->FoldCoordinates(node);
    first.state.registers[reg]=node.uniform.value_or(Scalar{});
    if(node.uniform)first.registers.push_back(arena->Constant(*node.uniform));
    else {first.registers.push_back(arena->values.size());arena->values.push_back(std::move(node));}
  }
  ExecutionProgram result; result.module=&m_; result.function=&f_; result.config=c_;
  result.register_names=names_; result.static_execution_counts.resize(decoded_.size());
  result.lane_traces.resize(members.size()); result.template_execution.used=true;
  // One split-tree node per path object, in creation order. `child` gets a copy
  // of `parent`'s path with the next PathId; the parent keeps its own.
  result.template_paths.emplace_back();
  const auto fork_path=[&result](Group& child,const Group& parent,SplitReason reason) {
    child.path=std::make_shared<std::vector<std::size_t>>(*parent.path);
    const PathId id{static_cast<std::uint32_t>(result.template_paths.size())};
    result.template_paths.push_back({parent.path_id,static_cast<std::uint32_t>(parent.path->size()),parent.pc,reason,{}});
    child.path_id=id;
  };
  AddressTable shared_addresses;BarrierTable shared_barriers;
  std::unique_ptr<ExecutorResources> shared_resources; // facts over the in-progress result
  if(runtime)shared_resources=std::make_unique<ExecutorResources>(result,c_.fixed_input.value("cta_coordinate",Coordinates{0,0,0}),shared_addresses,shared_barriers);
  std::map<std::uint32_t,WarpContext> shared_contexts;
  std::deque<Group> pending; pending.push_back(std::move(first));
  std::deque<Group> waiting;
  // A group's step is one step of every member: its budget is the per-thread one
  // times kTemplateStepFactor (practical shapes run long, cheap shared loops).
  const std::uint64_t template_step_budget =
      c_.max_steps_per_lane > UINT64_MAX / FrontendConfig::kTemplateStepFactor
          ? UINT64_MAX : c_.max_steps_per_lane * FrontendConfig::kTemplateStepFactor;
  std::map<DynamicKey,std::shared_ptr<TemplateExchange>> exchanges;
  std::map<DynamicKey,std::shared_ptr<TemplateVote>> votes;
  std::vector<std::pair<std::size_t,algorithm::SharedArray<std::size_t>>> vote_uses;
  struct ExchangeUse {
    std::shared_ptr<TemplateExchange> exchange;
    algorithm::SharedArray<std::size_t> members;
    std::size_t selection;
  };
  std::vector<ExchangeUse> exchange_uses;
  std::size_t exchange_revision=0,retried_revision=SIZE_MAX;
  std::uint64_t shared_revision=UINT64_MAX;
  bool split_collective_waits=false,stalled=false;
  auto owner=shared_from_this();
  struct EvaluationScratch {
    State state;
    std::vector<std::uint32_t> touched;
  };
  auto scratch=std::make_shared<EvaluationScratch>();
  scratch->state.registers.resize(names_.size());
  scratch->state.origins.resize(names_.size());
  struct ComputeDiamond {
    std::vector<std::uint32_t> writes,indices;
    std::vector<std::string> read_names,write_names;
    std::uint32_t join{};
    bool arms=false; // several arms (exact last-resort pass)
  };
  std::map<std::uint32_t,std::optional<ComputeDiamond>> diamonds;
  const auto compute_diamond=[&](std::uint32_t pc)->const ComputeDiamond* {
    auto [entry,inserted]=diamonds.try_emplace(pc);
    if(!inserted)return entry->second?&*entry->second:nullptr;
    auto join=OpaqueRegionJoin(pc,false);
    const bool arms=!join && exact_control_pass && (join=OpaqueRegionJoin(pc,true));
    if(!join)return nullptr;
    ComputeDiamond diamond;
    diamond.join=*join;diamond.arms=arms;
    std::set<std::uint32_t> writes;
    for(auto index=pc+1;index<*join;++index) {
      diamond.indices.push_back(index);
      writes.insert(decoded_[index].destinations.begin(),decoded_[index].destinations.end());
    }
    diamond.writes.assign(writes.begin(),writes.end());
    OpaqueRegionNames(pc,*join,diamond.read_names,diamond.write_names);
    entry->second=std::move(diamond);return &*entry->second;
  };
  // The opaque region a data predicate (table atoms) prefers: one join, several arms in any pass.
  std::map<std::uint32_t,std::optional<ComputeDiamond>> data_diamonds;
  const auto data_diamond=[&](std::uint32_t pc)->const ComputeDiamond* {
    if(const auto* plain=compute_diamond(pc))return plain;
    auto [entry,inserted]=data_diamonds.try_emplace(pc);
    if(!inserted)return entry->second?&*entry->second:nullptr;
    const auto &branch=*decoded_[pc].instruction;
    if(!branch.branch_target || *branch.branch_target<=pc+1)return nullptr;
    auto join=OpaqueRegionJoin(pc,true);
    if(!join)return nullptr;
    ComputeDiamond diamond;
    diamond.join=*join;diamond.arms=true;
    std::set<std::uint32_t> writes;
    for(auto index=pc+1;index<*join;++index) {
      diamond.indices.push_back(index);
      writes.insert(decoded_[index].destinations.begin(),decoded_[index].destinations.end());
    }
    diamond.writes.assign(writes.begin(),writes.end());
    OpaqueRegionNames(pc,*join,diamond.read_names,diamond.write_names);
    entry->second=std::move(diamond);return &*entry->second;
  };
  // Every warp with a member in the domain has all its CTA members in it: a warp split
  // across template groups could disagree on the predicate (aligned arms would be UB).
  std::map<std::uint32_t,std::size_t> warp_members;
  const auto whole_warps=[&](const std::vector<std::size_t>& domain) {
    if(warp_members.empty())for(const auto& member:members)++warp_members[member.warp];
    std::map<std::uint32_t,std::size_t> present;
    for(auto member:domain)++present[members[member].warp];
    return std::all_of(present.begin(),present.end(),[&](const auto& item){return item.second==warp_members.at(item.first);});
  };
  // The warp-uniform predicate for the template executor when the static proof
  // (MergedArm::uniform) fails: a provably warp-uniform template value (bounded walk of its
  // value DAG): a known uniform value, a full-warp vote.sync result, or a computation of such
  // values. A vote's mask node also reads the voted predicate, so an unfolded mask is
  // evaluated per member.
  const auto warp_uniform=[&](std::size_t root,const std::vector<std::size_t>& domain) {
    std::vector<std::size_t> stack{root};std::set<std::size_t> seen;
    while(!stack.empty()) {
      const auto id=stack.back();stack.pop_back();
      if(!seen.insert(id).second)continue;
      if(seen.size()>256)return false;
      const auto& node=arena->values[id];
      if(node.uniform){if(Unknown(*node.uniform))return false;continue;}
      if(node.vote) {
        // Each lane's match.any result names its own value class.
        if(node.operands.size()<2 || node.vote->mode=="match.any")return false;
        const auto mask=node.operands[1];
        if(const auto& uniform=arena->values[mask].uniform){if(!FullWarpMask(*uniform))return false;}
        else for(auto member:domain)if(!FullWarpMask(arena->Value(mask,member)))return false;
        continue;
      }
      if(!node.evaluate || node.exchange || node.member_value || !node.coordinates.empty())return false;
      stack.insert(stack.end(),node.operands.begin(),node.operands.end());
    }
    return true;
  };
  // A group's .local target: exact only when every member addresses the same slot; may_alias
  // when some member's address may reach .local memory. Evaluates before the step mutates the group.
  const auto local_target=[&](const Decoded& d,const Group& group)->LocalTarget {
    const auto& i=*d.instruction;
    const auto* o=AddressOperand(d);
    if(!LocalSpace(i) || !o || !i.memory_width)return {{},true};
    const auto& [base,displacement]=*o->address;
    const auto found=ids_.find(base);
    if(found==ids_.end())return LocalTargetOf(i,ValueText(base,group.state),displacement,*i.memory_width);
    const auto id=group.registers[found->second];
    if(const auto& uniform=arena->values[id].uniform)return LocalTargetOf(i,*uniform,displacement,*i.memory_width);
    std::optional<LocalTarget> agreed;
    for(auto member:group.members) {
      const auto target=LocalTargetOf(i,arena->Value(id,member),displacement,*i.memory_width);
      if(!agreed)agreed=target;
      else {
        if(agreed->exact!=target.exact)agreed->exact.reset();
        agreed->may_alias|=target.may_alias;
      }
      if(!agreed->exact && agreed->may_alias)break;
    }
    return agreed.value_or(LocalTarget{{},true});
  };
  // The distinct exact .local targets of a load's members (sorted), when every member's address is
  // an exact whole access through a register and at most kLocalSelectTargets differ.
  const auto local_targets=[&](const Decoded& d,const Group& group)->std::optional<std::vector<LocalAddress>> {
    const auto& i=*d.instruction;
    const auto* o=AddressOperand(d);
    if(!o || !i.memory_width || !ids_.contains(o->address->first))return {};
    const auto id=group.registers[ids_.at(o->address->first)];
    std::set<LocalAddress> targets;
    for(auto member:group.members) {
      const auto target=LocalTargetOf(i,arena->Value(id,member),o->address->second,*i.memory_width);
      if(!target.exact)return {};
      targets.insert(*target.exact);
      if(targets.size()>kLocalSelectTargets)return {};
    }
    return std::vector<LocalAddress>(targets.begin(),targets.end());
  };
  try {
    while (!pending.empty() || !waiting.empty()) {
      if(!waiting.empty() && (retried_revision!=exchange_revision ||
          (runtime && shared_revision!=runtime->revision))) {
        if(runtime){runtime->Progress();shared_revision=runtime->revision;}
        retried_revision=exchange_revision;
        pending.insert(pending.end(),std::make_move_iterator(waiting.begin()),std::make_move_iterator(waiting.end()));
        waiting.clear();
        // Progress resumed: waiting groups stay intact again.
        split_collective_waits=false;
      }
      if(pending.empty()) {
        if(retried_revision==exchange_revision && (!runtime || shared_revision==runtime->revision)) {
          // Only a stall that peer progress cannot resolve separates members
          // waiting for collective results (see the collective check below).
          // With a runtime, peers first get one chance to progress.
          if(!split_collective_waits && (!runtime || stalled))split_collective_waits=true;
          else {
            if(!runtime)throw TemplateFallback{};
            stalled=true;co_yield true;
          }
        }
        retried_revision=exchange_revision;
        if(runtime){runtime->Progress();shared_revision=runtime->revision;}
        pending=std::move(waiting);waiting.clear();
      }
      auto group=std::move(pending.front()); pending.pop_front();
      try {
      unsigned quantum=1;
      for (;;) {
        if(runtime && !quantum--) {pending.push_back(std::move(group));break;}
        if(!group.collective_results.empty()) {
          // Result availability follows peer groups' progress, not control
          // flow: members wait together and stay one thread cohort unless the
          // whole schedule has stalled. An intact group rethrows at its first
          // blocked member.
          std::vector<std::size_t> ready,blocked;
          for(auto member:group.members) {
            try {
              for(auto value:group.collective_results)(void)arena->Value(value,member);
              ready.push_back(member);
            } catch(const TemplatePending&) {
              if(!split_collective_waits)throw;
              blocked.push_back(member);
            }
          }
          if(!blocked.empty()) {
            if(!ready.empty()) {
              auto child=group;child.members=std::move(ready);child.collective_results.clear();
              fork_path(child,group,SplitReason::CollectiveWait);
              pending.push_back(std::move(child));++result.template_execution.splits;
              group.members=std::move(blocked);
            }
            throw TemplatePending{};
          }
          group.collective_results.clear();
        }
        if ((group.steps&1023)==0 && poll_) poll_();
        if (group.steps>=template_step_budget || group.pc>=decoded_.size()) throw TemplateFallback{};
        const auto &d=decoded_[group.pc]; const auto &i=*d.instruction;
        std::optional<std::pair<std::string,Integer>> assumed_equality;
        if(i.control=="direct_branch" && i.branch_target && c_.fixed_input.value("assume_trap_assertions",false))
          if(const auto assertion=Assertion(i,*i.branch_target)) {
            const auto& [reg,expected]=*assertion;
            for(auto member:group.members) {
              const auto value=arena->Value(group.registers[reg],member);
              if(!Unknown(value) && value!=Scalar(expected))
                Error(i,members[member].warp,members[member].lane,"assertion assumption contradicts a known value");
            }
            group.state.registers[reg]=expected;group.registers[reg]=arena->Constant(expected);
            group.state.registers[d.predicate]=true;group.registers[d.predicate]=arena->Constant(true);
            assumed_equality={names_[reg],expected};
          }
        if(group.merged_join && group.merged_join->first==group.pc) {
          for(auto reg:group.merged_join->second) {
            group.state.registers[reg]=Scalar{};group.state.origins[reg].reset();
            group.registers[reg]=arena->Constant(Scalar{});group.probe_values.erase(reg);
          }
          group.merged_join.reset();
        }
        if(auto join=group.probe_joins.find(group.pc);join!=group.probe_joins.end()) {
          for(const auto& [reg,value]:join->second)
            if(value.alternative!=group.state.registers[reg])group.probe_values[reg]=value;
          group.probe_joins.erase(join);
        }
        if(i.control=="direct_branch" && d.predicate!=kMissing) {
          if(auto probe=group.probe_values.find(d.predicate);probe!=group.probe_values.end()) {
            auto actual=Truth(group.state.registers[d.predicate]), alternative=Truth(probe->second.alternative);
            if(!actual || !alternative)throw TemplateFallback{};
            const bool a=*actual!=i.predicate_negated,b=*alternative!=i.predicate_negated;
            if(a!=b) {
              if(a || !b || !i.branch_target)throw TemplateFallback{};
              for(auto member:group.members)
                if(c_.BranchOutcome(i.line,members[member].warp,members[member].lane,group.state.branch_visits[group.pc]))
                  throw TemplateFallback{};
              const auto path_index=group.probes.at(probe->second.key);
              const auto record_index=group.path->at(path_index);
              const auto& original=decoded_[probe->second.key.instruction];
              std::set<std::uint32_t> needed;
              for(auto index=group.pc+1;index<std::min(*i.branch_target,group.pc+40);++index)
                needed.insert(decoded_[index].uses.begin(),decoded_[index].uses.end());
              std::optional<std::pair<std::uint32_t,std::set<std::uint32_t>>> fallback;
              State concrete=group.state, before=group.state;
              for(auto member_index:group.members) {
                for(auto reg:needed)concrete.registers[reg]=arena->Value(group.registers[reg],member_index);
                const auto record=arena->Read(record_index,member_index);
                for(auto reg:original.uses)before.registers[reg]=Scalar{};
                for(auto [reg,value]:detail::ExecutionStorage::Bindings(record))before.registers[reg]=value;
                const auto identity=Identity(original,before);
                if(!identity)throw TemplateFallback{};
                const auto member=members[member_index];WarpContext context;
                auto candidate=Fallback(d,concrete,*i.branch_target,*identity,member.warp,member.lane,context);
                if(!candidate || (fallback && fallback!=candidate))throw TemplateFallback{};
                fallback=std::move(candidate);
              }
              auto record=arena->records.at(record_index);
              record.record.mutableMetadata().probe_fallback_key=DynamicKey{fallback->first,group.state.occurrences[fallback->first]};
              group.path->at(path_index)=arena->records.size();arena->records.push_back(std::move(record));
              auto& join=group.probe_joins[*i.branch_target];
              for(auto reg:fallback->second) {
                auto previous=group.probe_values.find(reg);
                join[reg]={probe->second.key,previous!=group.probe_values.end() && previous->second.key==probe->second.key
                  ?previous->second.alternative:group.state.registers[reg]};
              }
            }
          }
        }
        // Partition before changing occurrence counters; children never rejoin.
        if(i.control!="direct_branch" && d.predicate!=kMissing && group.probe_values.contains(d.predicate))
          throw TemplateFallback{};
        bool active=true;
        const ComputeDiamond* opaque=nullptr;
        const MergedArm* merged=nullptr;
        if (d.predicate!=kMissing || i.control=="direct_branch") {
          std::vector<std::size_t> yes,no,unknown;
          // Members whose branch predicate is a data predicate (table atoms): an opaque region where
          // one is (data_diamond, several arms allowed), else their exact outcome.
          std::vector<std::size_t> data_yes,data_no;
          for (auto t:group.members) {
            const auto value=d.predicate==kMissing?Scalar(true):arena->Value(group.registers[d.predicate],t);
            auto truth=Truth(value);
            if(truth && i.predicate_negated)*truth=!*truth;
            bool supplied_choice=false;
            if(i.control=="direct_branch") {
              const auto supplied=c_.BranchOutcome(i.line,members[t].warp,members[t].lane,group.state.branch_visits[group.pc]);
              if(supplied && truth && *supplied!=*truth)static_choice_lines_.insert(i.line);
              else if(supplied)truth=supplied;
              supplied_choice=supplied.has_value();
            }
            if(!truth)unknown.push_back(t);
            else if(i.control=="direct_branch" && !supplied_choice && DataPredicate(value))(*truth?data_yes:data_no).push_back(t);
            else (*truth?yes:no).push_back(t);
          }
          const bool undecided=!unknown.empty(); // before data-predicate members join `unknown`
          if(!data_yes.empty() || !data_no.empty()) {
            if(unknown.empty() && DataOpaqueAdmitted(group.pc) && data_diamond(group.pc)) {
              data_opaque_used=true;
              unknown.insert(unknown.end(),data_yes.begin(),data_yes.end());
              unknown.insert(unknown.end(),data_no.begin(),data_no.end());
              std::sort(unknown.begin(),unknown.end());
            } else {
              yes.insert(yes.end(),data_yes.begin(),data_yes.end());
              no.insert(no.end(),data_no.begin(),data_no.end());
              std::sort(yes.begin(),yes.end());std::sort(no.begin(),no.end());
            }
          }
          if(!unknown.empty()) {
            if(i.control!="direct_branch")throw TemplateFallback{};
            if(!(opaque=compute_diamond(group.pc)) && !(opaque=(data_yes.empty() && data_no.empty()) ? nullptr : data_diamond(group.pc)) &&
                (!(merged=MergedArmAt(group.pc)) || (merged->aligned &&
                !merged->uniform && (!whole_warps(unknown) || !warp_uniform(group.registers[d.predicate],unknown)))))
              throw TemplateFallback{};
            // Shared-memory routing (as on the lane path): members whose predicate is undecided
            // (not a data predicate) are not skipped over without the shared-memory model.
            if(undecided && SharedRouted(group.pc,runtime!=nullptr))
              RouteToSharedModel(i);
            if(undecided)RoutedBranchUndecided(i,runtime!=nullptr);
            for(auto* domain:{&yes,&no})if(!domain->empty()) {
              auto child=group;child.members=std::move(*domain);
              fork_path(child,group,SplitReason::OpaqueDomain);
              pending.push_back(std::move(child));++result.template_execution.splits;
            }
            group.members=std::move(unknown);
          } else if (!yes.empty() && !no.empty()) {
            auto child=group; child.members=std::move(no); group.members=std::move(yes);
            fork_path(child,group,SplitReason::Predicate);
            pending.push_back(std::move(child)); ++result.template_execution.splits;
          } else active=!yes.empty();
        }
        if(active && !supports_template(d))throw TemplateFallback{};
        if(active && i.opcode=="elect" && !c_.raw.at("semantic_assumptions").at("representative_election").get<bool>())
          throw ExecutionError("assumption_required_representative_election",i.line,
              "Election outcomes are unsupported. Enable Representative election (representative_election) to analyze one representative lane.");
        if(active && i.opcode=="clusterlaunchcontrol" && !c_.raw.at("semantic_assumptions").at("clc_always_fails").get<bool>())
          throw ExecutionError("assumption_required_clc_always_fails",i.line,
              "CLC outcomes are unsupported. Enable Failed launch cancellation (clc_always_fails) to analyze failed requests.");
        std::map<std::uint32_t,ProbeValue> probe_updates;
        if(active && i.control!="direct_branch" && !group.probe_values.empty()) {
          State shadow=group.state;
          std::set<DynamicKey> keys;
          for(auto reg:source_registers[group.pc])if(auto dependency=group.probe_values.find(reg);dependency!=group.probe_values.end()) {
            keys.insert(dependency->second.key);shadow.registers[reg]=dependency->second.alternative;
          }
          if(!keys.empty()) {
            if(keys.size()!=1 || i.semantics.category!="compute" ||
                !In(i.opcode,{"mov","cvt","selp","setp","and","or","xor","not"}))throw TemplateFallback{};
            const auto member=members[group.members.front()];WarpContext context;
            Data(d,shadow,member.warp,member.lane,group.state.occurrences[group.pc],Json::object(),context);
            for(auto reg:d.destinations) {
              if(Unknown(shadow.registers[reg]))throw TemplateFallback{};
              probe_updates[reg]={*keys.begin(),shadow.registers[reg]};
            }
          }
        }
        if(active && (i.opcode=="elect" || i.opcode=="shfl")) {
          const auto& reads=source_registers[group.pc];
          State validation=group.state;
          std::uint32_t checked_lanes=0;
          const bool uniform_inputs=std::all_of(reads.begin(),reads.end(),[&](auto reg){return arena->values[group.registers[reg]].uniform.has_value();});
          for(auto t:group.members) {
            const auto member=members[t];
            const auto bit=std::uint32_t(1)<<member.lane;
            if(uniform_inputs && (checked_lanes&bit))continue;
            checked_lanes|=bit;
            for(auto reg:reads) {
              if(i.opcode=="shfl" && reg==d.operands[1].reg &&
                  std::none_of(d.operands.begin()+2,d.operands.end(),[&](const auto& operand){return operand.reg==reg;}))continue;
              validation.registers[reg]=arena->Value(group.registers[reg],t);
            }
            if(i.opcode=="shfl") (void)SelectShuffle(d,validation,member.warp,member.lane);
            else {
              WarpContext context;
              Data(d,validation,member.warp,member.lane,group.state.occurrences[group.pc],Json::object(),context);
            }
          }
        }
        if(active && IsWait(i) && !d.wait_loop) {
          State probe_state=group.state;
          for(auto member:group.members) {
            for(auto reg:source_registers[group.pc])probe_state.registers[reg]=arena->Value(group.registers[reg],member);
            if(!Identity(d,probe_state))throw TemplateFallback{};
          }
        }
        bool shared_handled=false;
        if(runtime && active && (In(i.opcode,{"bar","barrier","cp","tensormap"}) ||
            (i.opcode=="mbarrier" && (!IsWait(i) || d.wait_loop)) ||
            (i.opcode=="tcgen05" && Has(i,"commit")) ||
            (i.opcode=="clusterlaunchcontrol" && i.semantics.operation=="try_cancel_complete_tx") ||
            (In(i.opcode,{"ld","st"}) && (Has(i,"shared") || Has(i,"shared::cta") || Has(i,"shared::cluster"))) ||
            UntrackedShared(i))) {
          if(!group.shared) {
            group.shared=Group::SharedWork{};
            for(std::size_t slot=0;slot<group.members.size();++slot)group.shared->pending.push_back(slot);
            group.shared->outputs.resize(d.destinations.size(),std::vector<Scalar>(group.members.size()));
          }
          auto& work=*group.shared;
          std::vector<std::size_t> blocked;
          State concrete=group.state;
          for(auto slot:work.pending) {
            const auto member_index=group.members[slot];const auto member=members[member_index];
            try {for(auto reg:d.uses)concrete.registers[reg]=arena->Value(group.registers[reg],member_index);}
            catch(const TemplatePending&){blocked.push_back(slot);continue;}
            auto& context=shared_contexts[member.warp];context.shared=runtime;context.resources=shared_resources.get();
            auto& trace=result.lane_traces[member_index];trace.warp_id=member.warp;trace.lane_id=member.lane;
            trace.instructions=arena->Expand(arena,group.path,member_index,group.path_id);
            bool handled=false;
            try {handled=SharedData(d,concrete,member.warp,member.lane,group.state.occurrences[group.pc],Json::object(),context,trace,d.wait_loop.has_value());}
            catch(const SharedBlocked&){blocked.push_back(slot);continue;}
            work.handled|=handled;
            if(handled)for(std::size_t j=0;j<d.destinations.size();++j)
              work.outputs[j][slot]=concrete.registers[d.destinations[j]];
          }
          work.pending=std::move(blocked);
          if(!work.pending.empty())throw TemplatePending{};
          shared_handled=work.handled;
        }
        // With no slot filled, only an explicit .local access can change the slots.
        std::optional<LocalTarget> local_access;
        if(active && local_memory_ && (LocalSpace(i) || MayWriteLocal(i)) &&
            (!group.local.empty() || i.semantics.memory_space=="local"))local_access=local_target(d,group);
        ++group.steps; ++result.template_execution.steps;
        result.template_execution.logical_steps+=group.members.size();
        auto occurrence=group.state.occurrences[group.pc]++;
        const auto capture = i.control == "direct_branch" && !opaque && !merged && d.predicate != kMissing
            ? CaptureDecision(i.line, active, group.state.branch_visits[group.pc])
            : std::optional<BranchScopeDecision>{};
        if (i.control=="direct_branch") ++group.state.branch_visits[group.pc];
        if (!active && !capture) {++group.pc; continue;}
        TemplateRecord record;
        if(opaque) {
          for(auto reg:opaque->writes) {
            group.state.registers[reg]=Scalar{};group.state.origins[reg].reset();
            group.registers[reg]=arena->Constant(Scalar{});group.probe_values.erase(reg);
          }
          ForgetRegionLocal(group.pc,opaque->join,group.local);
        }
        if (i.control=="direct_branch" && !opaque && !merged && !capture) {
          auto &m=record.record.mutableMetadata(); m.instruction_id=group.pc; m.occurrence=occurrence; m.predicate_value=true;
          if(assumed_equality){m.assumed_equality=assumed_equality;++group.assumed_assertions;}
        } else {
          record.record=Record(d,group.state,occurrence,opaque || merged?std::optional<bool>{}:std::optional<bool>(true));
          if(merged)record.record.mutableMetadata().merged_arm_join=*i.branch_target;
          record.record.mutableMetadata().branch_scope_decision = capture;
          if (capture && active && assumed_equality) {
            record.record.mutableMetadata().assumed_equality = assumed_equality;
            ++group.assumed_assertions;
          }
          record.record.mutableKnownRegisters().clear();
          for (auto reg:d.uses) {
            auto id=group.registers[reg];
            if (!arena->values[id].uniform || !Unknown(*arena->values[id].uniform)) record.bindings.emplace_back(reg,id);
          }
        }
        if(opaque) {
          auto& metadata=record.record.mutableMetadata();
          metadata.opaque_compute_register_reads=opaque->read_names;
          metadata.opaque_compute_register_writes=opaque->write_names;
          metadata.opaque_compute_diamond_indices=opaque->indices;
          metadata.opaque_compute_join_label=opaque->join==*i.branch_target?i.branch_label:"pc:"+std::to_string(opaque->join);
          if(opaque->arms)++group.opaque_regions[i.line];
        }
        if (std::all_of(record.bindings.begin(),record.bindings.end(),[&](auto binding){return arena->values[binding.second].uniform.has_value();})) {
          auto &bindings=record.record.mutableKnownRegisters();
          for (auto [reg,id]:record.bindings) bindings.emplace_back(reg,*arena->values[id].uniform);
          record.bindings.clear();
        }
        if (d.wait_loop) {
          auto& metadata=record.record.mutableMetadata();
          metadata.collapsed_wait_loop=true;
          metadata.normalized_wait_result=true;
          metadata.wait_exit_label=d.wait_loop->first;
          for(auto index=group.pc;index<(d.replay_wait_result?group.pc+1:d.wait_loop->second);++index)
            metadata.collapsed_wait_indices.push_back(index);
        } else if(IsWait(i)) {
          auto& metadata=record.record.mutableMetadata();
          metadata.canonical_wait_probe=true;
          metadata.normalized_wait_result=false;
        }
        group.path->push_back(arena->records.size()); arena->records.push_back(std::move(record));
        result.static_execution_counts[group.pc]+=group.members.size();
        // False decisions are control observations only, never executed data
        // effects. This record exists solely in BranchScopeCapture mode.
        if (!active) { ++group.pc; continue; }
        if(opaque){group.approximate_control=true;group.pc=opaque->join;continue;}
        if(merged){group.approximate_control=true;group.merged_join={*i.branch_target,merged->writes};++group.pc;continue;}
        if(group.shared) {
          if(shared_handled)for(std::size_t j=0;j<d.destinations.size();++j) {
            const auto reg=d.destinations[j];const auto& outputs=group.shared->outputs[j];
            const auto value=arena->SharedResult({group.members.data(),group.members.size()},outputs,
                *member_coordinates,coordinate_value,c_.warp_size,c_.block_dim);
            group.registers[reg]=value;
            group.state.registers[reg]=arena->values[value].uniform.value_or(Scalar{});
          }
          group.shared.reset();
        }
        if (d.wait_loop) {
          Set(d,group.state,{true});
          for(auto reg:d.destinations)group.registers[reg]=arena->Constant(group.state.registers[reg]);
          Origins(d,group.state,occurrence);
          for(auto reg:d.destinations)group.probe_values.erase(reg);
          group.pc=d.replay_wait_result?group.pc+1:d.wait_loop->second;
          continue;
        }
        if(IsWait(i)) {
          const DynamicKey key{group.pc,occurrence};
          group.probes[key]=group.path->size()-1;
          Set(d,group.state,{false});
          for(auto reg:d.destinations) {
            group.registers[reg]=arena->Constant(group.state.registers[reg]);
            group.probe_values[reg]={key,true};
          }
          Origins(d,group.state,occurrence);++group.pc;continue;
        }
        if (i.control=="return" || i.control=="exit") {
          result.template_paths[group.path_id.value].group=static_cast<std::uint32_t>(result.thread_template_groups.size());
          result.group_paths.push_back(group.path_id);
          result.thread_template_groups.emplace_back(group.members.begin(),group.members.end());
          result.template_execution.path_entries+=group.path->size();
          result.template_execution.thread_memberships+=group.members.size();
          std::shared_ptr<const std::vector<std::size_t>> path=std::move(group.path);
          for (auto t:group.members) {
            auto &trace=result.lane_traces[t]; trace.warp_id=members[t].warp; trace.lane_id=members[t].lane;
            trace.assumed_assertions=group.assumed_assertions;
            trace.opaque_regions=group.opaque_regions;
            trace.instructions=arena->Expand(arena,path,t,group.path_id);
            for (std::uint32_t b=0;b<group.state.branch_visits.size();++b)
              if (group.state.branch_visits[b]) trace.branch_visits[b]=group.state.branch_visits[b];
          }
          break;
        }
        if (i.control=="direct_branch") {
          if (!i.branch_target) throw TemplateFallback{};
          group.pc=*i.branch_target; continue;
        }
        if (i.control=="indirect_branch") {
          std::map<std::uint32_t,std::vector<std::size_t>> targets;
          State concrete=group.state;
          for(auto member:group.members) {
            for(auto reg:d.uses)concrete.registers[reg]=arena->Value(group.registers[reg],member);
            const auto index=ScalarInteger(Value(d.operands.front(),concrete));
            if(!index)throw TemplateFallback{};
            if(*index<0 || *index>=i.indirect_targets.size())
              Error(i,members[member].warp,members[member].lane,"indirect branch index is outside its target table");
            targets[i.indirect_targets[index->convert_to<std::size_t>()]].push_back(member);
          }
          for(auto target=std::next(targets.begin());target!=targets.end();++target) {
            auto child=group;child.pc=target->first;child.members=std::move(target->second);
            fork_path(child,group,SplitReason::IndirectBranch);
            pending.push_back(std::move(child));++result.template_execution.splits;
          }
          group.pc=targets.begin()->first;group.members=std::move(targets.begin()->second);
          continue;
        }
        if(i.opcode=="vote" || i.opcode=="match") {
          auto& vote=votes[{group.pc,occurrence}];
          if(!vote) {
            vote=std::make_shared<TemplateVote>();vote->members=member_indices;
            vote->mode=i.opcode=="vote"?i.modifiers[1]:"match."+i.modifiers[0];
            vote->error=[owner,pc=group.pc](std::uint32_t actor,const std::string& message) {
              owner->Error(*owner->decoded_[pc].instruction,actor/owner->c_.warp_size,actor%owner->c_.warp_size,message);
            };
            if(i.opcode=="match")vote->equal=[owner,pc=group.pc](const Scalar& a,const Scalar& b) {
              return Truth(Compare(*owner->decoded_[pc].instruction,"eq",a,b,owner->c_));
            };
          }
          const auto reads=source_registers[group.pc];
          const auto operand=[&](std::size_t index) {
            TemplateValue node;
            for(auto reg:reads)node.operands.push_back(group.registers[reg]);
            node.evaluate=[owner,scratch,reads,pc=group.pc,index](const std::vector<Scalar>& args)->Scalar {
              auto& own=PrivateScratch(*scratch);
              auto& state=own.state;
              for(auto reg:own.touched){state.registers[reg]=Scalar{};state.origins[reg].reset();}
              own.touched=reads;
              for(std::size_t k=0;k<reads.size();++k)state.registers[reads[k]]=args[k];
              const auto& decoded=owner->decoded_[pc];
              auto text=TrimView(decoded.operands[index].text);
              if(index==1 && text.starts_with('!')) {
                auto value=ScalarInteger(owner->ValueText(text.substr(1),state));
                return value?Scalar(*value==0):Scalar{};
              }
              return owner->Value(decoded.operands[index],state);
            };
            arena->FoldCoordinates(node);
            if(node.uniform)return arena->Constant(*node.uniform);
            const auto id=arena->values.size();arena->values.push_back(std::move(node));return id;
          };
          const auto predicate=operand(1),mask=operand(2);
          vote->predicates->producers.push_back({group.members,predicate});
          vote->masks->producers.push_back({group.members,mask});++exchange_revision;
          for(std::size_t slot=0;slot<d.destination_slots.size();++slot) {
            // match.all's second destination is its predicate.
            auto result=vote;
            if(slot==1 && vote->mode=="match.all"){result=std::make_shared<TemplateVote>(*vote);result->mode="match.all.pred";}
            TemplateValue node;node.operands={coordinate_value,mask};node.vote=std::move(result);
            const auto id=arena->values.size();arena->values.push_back(std::move(node));
            vote_uses.emplace_back(id,group.members);
            group.collective_results.push_back(id);
            if(const auto reg=d.destination_slots[slot];reg!=kMissing){group.registers[reg]=id;group.state.registers[reg]=Scalar{};}
          }
          Origins(d,group.state,occurrence);++group.pc;continue;
        }
        // activemask: as in Data(), a warp whose lanes all share this group's path, which merged no
        // divergent region, executes convergently; other members stay unknown.
        if(i.opcode=="activemask") {
          std::map<std::uint32_t,std::size_t> present;
          for(auto member:group.members)++present[members[member].warp];
          std::vector<Scalar> outputs;outputs.reserve(group.members.size());
          for(auto member:group.members)
            outputs.push_back(!group.approximate_control && present[members[member].warp]==c_.warp_size
                ?Scalar(Integer(0xffffffffu)):Scalar{});
          const auto value=arena->SharedResult({group.members.data(),group.members.size()},outputs,
              *member_coordinates,coordinate_value,c_.warp_size,c_.block_dim);
          for(auto reg:d.destinations){group.registers[reg]=value;group.state.registers[reg]=arena->values[value].uniform.value_or(Scalar{});}
          Origins(d,group.state,occurrence);
          for(auto reg:d.destinations)group.probe_values.erase(reg);
          ++group.pc;continue;
        }
        if(i.opcode=="shfl") {
          const DynamicKey key{group.pc,occurrence};
          auto& exchange=exchanges[key];
          if(!exchange)exchange=std::make_shared<TemplateExchange>();
          const auto source=d.operands[1].reg==kMissing
              ?arena->Constant(Value(d.operands[1],group.state)):group.registers[d.operands[1].reg];
          exchange->producers.push_back({group.members,source});++exchange_revision;
          std::vector<std::uint32_t> selector_registers;
          for(std::size_t k=2;k<d.operands.size();++k)
            for(auto reg:d.operands[k].registers)
              if(std::find(selector_registers.begin(),selector_registers.end(),reg)==selector_registers.end())selector_registers.push_back(reg);
          auto selection_node=[&](bool index) {
            TemplateValue node;
            for(auto reg:selector_registers)node.operands.push_back(group.registers[reg]);
            node.operands.push_back(coordinate_value);
            node.evaluate=[owner,scratch,selector_registers,pc=group.pc,member_indices,index](const std::vector<Scalar>& args)->Scalar {
              auto& own=PrivateScratch(*scratch);
              auto& state=own.state;
              for(auto reg:own.touched){state.registers[reg]=Scalar{};state.origins[reg].reset();}
              own.touched=selector_registers;
              for(std::size_t k=0;k<selector_registers.size();++k)state.registers[selector_registers[k]]=args[k];
              const auto thread=std::get<Integer>(args.back()).convert_to<std::uint32_t>();
              const auto selected=owner->SelectShuffle(owner->decoded_[pc],state,thread/owner->c_.warp_size,thread%owner->c_.warp_size);
              if(!selected)return {};
              if(!index)return selected->valid;
              const auto actor=thread-thread%owner->c_.warp_size+selected->lane;
              auto found=member_indices->find(actor);
              if(found==member_indices->end())throw TemplateFallback{};
              return Integer(found->second);
            };
            const auto id=arena->values.size();arena->values.push_back(std::move(node));return id;
          };
          const auto selection=selection_node(true);
          exchange_uses.push_back({exchange,group.members,selection});
          TemplateValue completion;completion.operands={selection};completion.exchange=exchange;
          const auto completion_id=arena->values.size();
          group.collective_results.push_back(completion_id);
          arena->values.push_back(std::move(completion));
          if(!d.destination_slots.empty() && d.destination_slots[0]!=kMissing) {
            const auto reg=d.destination_slots[0];
            group.registers[reg]=completion_id;
            group.state.registers[reg]=Scalar{};
          }
          if(d.destination_slots.size()>1 && d.destination_slots[1]!=kMissing) {
            const auto reg=d.destination_slots[1];group.registers[reg]=selection_node(false);
            group.state.registers[reg]=Scalar{};
          }
          Origins(d,group.state,occurrence);++group.pc;continue;
        }
        // .local slots as in Data(): a load at an exact address (shared, or each member's own) reads
        // each member's stored value; other loads stay on the generic path (unknown for .local memory).
        if(local_access) {
          const auto width=i.memory_width.value_or(0);
          const auto count=d.destination_slots.size();
          if(i.opcode=="ld" && count && width%count==0 && (local_access->exact || LocalSpace(i))) {
            std::vector<std::size_t> loaded;
            bool known=false;
            const auto bytes=static_cast<unsigned>(width/count);
            const bool sign=Has(i,"s"+std::to_string(bytes*8));
            const auto add=[&](TemplateValue node) {
              arena->FoldCoordinates(node);
              if(node.uniform)loaded.push_back(arena->Constant(*node.uniform));
              else {loaded.push_back(arena->values.size());arena->values.push_back(std::move(node));}
            };
            if(local_access->exact) {
              for(std::size_t k=0;k<count;++k) {
                const auto slot=group.local.find(LocalAddress{local_access->exact->first,local_access->exact->second+std::int64_t(k*bytes)});
                if(slot==group.local.end() || slot->second.bytes!=bytes) {loaded.push_back(arena->Constant(Scalar{}));continue;}
                const auto stored=slot->second.value;
                if(const auto& uniform=arena->values[stored].uniform)loaded.push_back(arena->Constant(LocalLoaded(*uniform,bytes,sign)));
                else {
                  TemplateValue node;node.operands={stored};
                  node.evaluate=[bytes,sign](const std::vector<Scalar>& args){return LocalLoaded(args.at(0),bytes,sign);};
                  add(std::move(node));
                }
                known=true;
              }
            } else if(const auto targets=local_targets(d,group)) {
              // Exact addresses that differ between members: a per-member selection over the slots
              // the members address (operand 0 is the address, then one stored value per target).
              const auto* o=AddressOperand(d);
              const auto address=group.registers[ids_.at(o->address->first)];
              for(std::size_t k=0;k<count;++k) {
                TemplateValue node;node.operands={address};
                bool filled=false;
                for(const auto& [resource,offset]:*targets) {
                  const auto slot=group.local.find(LocalAddress{resource,offset+std::int64_t(k*bytes)});
                  const bool exact=slot!=group.local.end() && slot->second.bytes==bytes;
                  node.operands.push_back(exact?slot->second.value:arena->Constant(Scalar{}));
                  filled|=exact;
                }
                if(!filled) {loaded.push_back(arena->Constant(Scalar{}));continue;}
                node.evaluate=[instruction=&i,keys=*targets,displacement=o->address->second,width,bytes,sign](const std::vector<Scalar>& args)->Scalar {
                  const auto target=LocalTargetOf(*instruction,args.at(0),displacement,width);
                  const auto at=target.exact?std::lower_bound(keys.begin(),keys.end(),*target.exact):keys.end();
                  if(at==keys.end() || *at!=*target.exact)return {};
                  return LocalLoaded(args.at(1+(at-keys.begin())),bytes,sign);
                };
                add(std::move(node));
                known=true;
              }
            }
            if(known) {
              for(std::size_t k=0;k<count;++k)if(const auto reg=d.destination_slots[k];reg!=kMissing) {
                group.registers[reg]=loaded[k];
                group.state.registers[reg]=arena->values[loaded[k]].uniform.value_or(Scalar{});
              }
              Origins(d,group.state,occurrence);
              for(auto reg:d.destinations)group.probe_values.erase(reg);
              ++group.pc;continue;
            }
          } else if(MayWriteLocal(i) && local_access->may_alias) {
            const auto parts=i.opcode=="st" && d.operands.size()>=2?VectorElements(d.operands[1].text):std::vector<std::string_view>{};
            if(local_access->exact && width && !parts.empty() && width%parts.size()==0) {
              const auto bytes=static_cast<unsigned>(width/parts.size());
              ForgetLocal(group.local,*local_access->exact,width);
              for(std::size_t k=0;k<parts.size();++k) {
                const auto reg=ids_.find(std::string(parts[k]));
                const auto value=reg!=ids_.end()?group.registers[reg->second]:arena->Constant(ValueText(parts[k],group.state));
                if(const auto& uniform=arena->values[value].uniform;uniform && Unknown(*uniform))continue;
                group.local.insert_or_assign(LocalAddress{local_access->exact->first,local_access->exact->second+std::int64_t(k*bytes)},
                    LocalSlot<std::size_t>{bytes,value});
              }
            } else group.local.clear();
          }
        }
        if(shared_handled)Origins(d,group.state,occurrence);
        if (!shared_handled && !d.destinations.empty()) {
          const auto &reads=source_registers[group.pc];
          const bool uses_thread=i.opcode=="elect";
          const bool uniform_inputs=std::all_of(reads.begin(),reads.end(),[&](auto reg){return arena->values[group.registers[reg]].uniform.has_value();});
          bool uniform=!uses_thread && uniform_inputs;
          if(i.opcode=="clusterlaunchcontrol" && Has(i,"query_cancel") && Has(i,"is_canceled"))uniform=true;
          if (i.opcode=="ld" && !Has(i,"param") && constant_inputs_.empty() && pointer_tables_.empty() &&
              tensor_properties_.empty()) uniform=true;
          if (uniform) {
            WarpContext context; const auto member=members[group.members.front()];
            Data(d,group.state,member.warp,member.lane,occurrence,Json::object(),context);
            for (auto reg:d.destinations) group.registers[reg]=arena->Constant(group.state.registers[reg]);
          } else {
            std::vector<std::size_t> args; std::vector<std::optional<DynamicKey>> origins;
            for (auto reg:reads) {args.push_back(group.registers[reg]); origins.push_back(group.state.origins[reg]);}
            if(uses_thread)args.push_back(coordinate_value);
            for (auto dest:d.destinations) {
              const bool reusable=!Has(i,"cc") && In(i.opcode,{"add","sub","mul","mad","shl","shr","and","or","xor"}) &&
                  std::none_of(args.begin(),args.end(),[&](auto id) {
                    return arena->values[id].uniform && Unknown(*arena->values[id].uniform);
                  });
              TemplateArena::ComputationKey computation{group.pc,dest,args};
              if(reusable)if(const auto cached=arena->FindComputation(computation)) {
                group.registers[dest]=*cached;
                group.state.registers[dest]=arena->values[*cached].uniform.value_or(Scalar{});
                continue;
              }
              TemplateValue node; node.operands=args;
              node.evaluate=[owner,scratch,reads,origins,pc=group.pc,dest,occurrence,uses_thread](const std::vector<Scalar> &args) {
                auto &own=PrivateScratch(*scratch);
                auto &state=own.state;
                for (auto reg:own.touched) {
                  state.registers[reg]=Scalar{};
                  state.origins[reg].reset();
                }
                own.touched=reads;
                const auto &decoded=owner->decoded_[pc];
                own.touched.insert(own.touched.end(),decoded.destinations.begin(),decoded.destinations.end());
                for (std::size_t j=0;j<reads.size();++j) {state.registers[reads[j]]=args[j]; state.origins[reads[j]]=origins[j];}
                WarpContext context;
                const auto thread=uses_thread?std::get<Integer>(args.back()).convert_to<std::uint32_t>():0;
                owner->Data(decoded,state,thread/owner->c_.warp_size,thread%owner->c_.warp_size,occurrence,Json::object(),context);
                return state.registers[dest];
              };
              arena->FoldCoordinates(node);
              group.state.registers[dest]=node.uniform.value_or(Scalar{});
              if(reusable)arena->RememberComputation(std::move(computation),arena->values.size());
              group.registers[dest]=arena->values.size(); arena->values.push_back(std::move(node));
            }
          }
          Origins(d,group.state,occurrence);
        }
        for(auto reg:d.destinations)group.probe_values.erase(reg);
        if(i.opcode=="clusterlaunchcontrol" && Has(i,"query_cancel") && Has(i,"is_canceled") && !d.destinations.empty())
          arena->records.at(group.path->back()).record.mutableMetadata().normalized_clc_result=Truth(group.state.registers[d.destinations.front()]);
        for(auto& [reg,value]:probe_updates)
          if(value.alternative!=group.state.registers[reg])group.probe_values[reg]=std::move(value);
        ++group.pc;
      }
      } catch(const TemplatePending&) {waiting.push_back(std::move(group));}
    }
    for(const auto& [value,domain]:vote_uses)for(auto member:domain)(void)arena->Value(value,member);
    for(const auto& use:exchange_uses)for(auto member:use.members) {
      const auto value=PlainValue(arena->Value(use.selection,member));
      if(auto index=std::get_if<Integer>(&value))
        if(!use.exchange->Source(index->convert_to<std::size_t>()))throw TemplateFallback{};
    }
  } catch (const TemplateFallback &) {co_return std::nullopt;}
    catch (const TemplatePending &) {co_return std::nullopt;}
  result.template_execution.value_nodes=arena->values.size();
  result.template_execution.record_nodes=arena->records.size();
  for(const auto& value:arena->values)
    result.template_execution.coordinate_bindings+=value.coordinates.size();
  for(const auto& record:arena->records)
    result.template_execution.record_bindings+=record.bindings.size()+detail::ExecutionStorage::Bindings(record.record).size();
  result.release_value_caches=[weak=std::weak_ptr<TemplateArena>(arena)](std::optional<std::size_t> member) {
    const auto held=weak.lock();
    if(!held)return;
    if(!member)for(auto& cache:held->caches)std::unordered_map<std::size_t,Scalar>{}.swap(cache);
    else if(*member<held->caches.size())std::unordered_map<std::size_t,Scalar>{}.swap(held->caches[*member]);
  };
  result.release_value_caches(std::nullopt);
  if(runtime)for(auto it=runtime->tensor_maps.begin();it!=runtime->tensor_maps.end();++it)
    result.config.tma_descriptors["executed."+it.key()]=it.value();
  result.tensor_property_decisions=property_decisions_;
  result.static_branch_choice_lines=static_choice_lines_;
  co_return std::move(result);
}

} // namespace
} // namespace sentryward::frontend::ptx
