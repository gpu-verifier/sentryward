#pragma once
#include "execution_impl.hpp"

namespace sentryward::frontend::ptx {
namespace {
std::optional<WaitIdentity> Executor::Identity(const Decoded &d,
                                               const State &s) const {
  if (d.operands.size() < 3)
    return {};
  auto text = TrimView(d.operands[1].text);
  if (text.starts_with('[') && text.ends_with(']'))
    text = TrimView(text.substr(1, text.size() - 2));
  auto split = SplitAddress(text);
  if (!split)
    return {};
  auto address = ValueText(split->first, s);
  if (auto n = ScalarInteger(address))
    address = Integer(*n + split->second);
  else if (auto p = std::get_if<ClusterSharedPointer>(&address))
    p->local_address += split->second;
  else if (split->second != 0)
    return {};
  auto target = Value(d.operands[2], s);
  if (Unknown(address) || Unknown(target))
    return {};
  return WaitIdentity{d.instruction->semantics.operation.ends_with("parity")
                          ? "parity"
                          : "state",
                      std::move(address), std::move(target)};
}
std::optional<std::pair<std::uint32_t, std::set<std::uint32_t>>>
Executor::Fallback(const Decoded &branch, const State &state,
                   std::uint32_t target, const WaitIdentity &identity,
                   std::uint32_t warp, std::uint32_t lane,
                   WarpContext &context) const {
  auto pc = branch.instruction->id;
  if (target <= pc)
    return {};
  State shadow = state;
  std::set<std::uint32_t> changed;
  for (std::uint32_t index = pc + 1; index < std::min(target, pc + 40);
       ++index) {
    if (index >= decoded_.size())
      return {};
    const auto &d = decoded_[index];
    const auto &i = *d.instruction;
    if (i.predicate)
      return {};
    if (d.wait_loop) {
      changed.insert(d.destinations.begin(), d.destinations.end());
      if(d.replay_wait_result)
        for(auto replay=index+1;replay<d.wait_loop->second;++replay)
          changed.insert(decoded_[replay].destinations.begin(),decoded_[replay].destinations.end());
      auto exit = d.wait_loop->second;
      std::set<std::uint32_t> visited;
      while (exit != target && visited.insert(exit).second) {
        if (exit >= decoded_.size())
          return {};
        const auto &jump = *decoded_[exit].instruction;
        if (jump.control != "direct_branch" || jump.predicate ||
            !jump.branch_target)
          return {};
        exit = *jump.branch_target;
      }
      auto same = Identity(d, shadow);
      if (exit == target && same && *same == identity)
        return {{index, changed}};
      return {};
    }
    if (i.control != "none" || i.semantics.category != "compute" ||
        !In(i.opcode,
            {"mov", "cvt", "cvta", "add", "sub", "mul", "mad", "shl", "shr",
             "and", "or", "xor", "not", "neg", "setp", "selp"}))
      return {};
    Data(d, shadow, warp, lane, state.occurrences[index], Json::object(),
         context);
    changed.insert(d.destinations.begin(), d.destinations.end());
  }
  return {};
}
bool Executor::SharedData(const Decoded &d, State &state, std::uint32_t warp,
    std::uint32_t lane, std::uint64_t occurrence, const Json &contract,
    WarpContext &context, const LaneTrace &trace, bool wait) const {
  if (!context.shared) return false;
  const auto &i=*d.instruction;
  auto &runtime=*context.shared;
  const auto rank=c_.fixed_input.value("cluster_ctarank",0U);
  const SharedActor actor=(std::uint64_t(rank)<<32) | (warp*32+lane);
  const DynamicKey key{i.id,occurrence};
  const SharedValueSite site{c_.fixed_input.value("cta_coordinate",Coordinates{0,0,0}),warp*32+lane,key};
  auto value=[&](std::string_view text) { return ValueText(TrimView(text),state); };
  auto number=[&](std::string_view text) -> std::int64_t {
    auto n=ScalarInteger(value(text));
    if (!n) throw ExecutionError("shared_control_operand",i.line,"unknown shared-control operand: "+std::string(text));
    return Int64(*n,"shared-control operand");
  };
  const auto record=Record(d,state,occurrence,true);
  auto address=[&](std::string_view text) -> std::optional<std::uint64_t> {
    text=TrimView(text);
    if (text.starts_with('[') && text.ends_with(']')) text=TrimView(text.substr(1,text.size()-2));
    auto split=SplitAddress(text); if (!split) return {};
    auto v=value(split->first); auto owner=rank; std::optional<Integer> offset;
    if (auto p=std::get_if<ClusterSharedPointer>(&v)) { owner=p->target_rank; offset=Integer(p->local_address+split->second); }
    else if ((offset=ScalarInteger(v))) {
      // The record as the producer Resources read it: detached, on this trace's
      // lane, registered once where the lane is set.
      auto &executor=*context.resources;
      const auto record_lane=executor.facts.Register(trace);
      executor.resources.SetLane(record_lane);
      const auto detached=executor.facts.Detached(record,record_lane);
      if (auto inferred=executor.resources.SharedOwner(detached,split->first))
        owner=*inferred;
      *offset=executor.resources.AddressRegisterBits(detached,split->first,Integer(*offset+split->second));
    }
    if (!offset) return {};
    if (*offset<0 || *offset>UINT32_MAX)
      throw ExecutionError("shared_control_operand",i.line,"shared address outside u32");
    return (std::uint64_t(owner)<<32) | offset->convert_to<std::uint32_t>();
  };
  auto required_address=[&](std::string_view text) {
    auto a=address(text);
    if (!a) throw ExecutionError("shared_control_operand",i.line,"unknown shared address");
    return *a;
  };
  auto operand=[&](unsigned index) -> const std::string & { return d.operands.at(index).text; };
  auto encoded_values=[&](std::string_view text,unsigned width) {
    text=TrimView(text);
    if (text.starts_with('{') && text.ends_with('}')) text=text.substr(1,text.size()-2);
    std::vector<std::string_view> parts;
    for (;;) {
      auto comma=text.find(','); parts.push_back(TrimView(text.substr(0,comma)));
      if (comma==text.npos) break;
      text.remove_prefix(comma+1);
    }
    if (width%parts.size()) throw ExecutionError("shared_control_operand",i.line,"unaligned register vector width");
    std::vector<SharedByte> result;
    for (auto part : parts) {
      auto bytes=SharedBytes(value(part),width/parts.size()); result.insert(result.end(),bytes.begin(),bytes.end());
    }
    return result;
  };
  const auto cluster=c_.cluster_dim.value_or(Coordinates{1,1,1});
  const auto cluster_size=cluster[0]*cluster[1]*cluster[2];
  if (wait) {
    runtime.Wait(required_address(operand(1)),number(operand(2)),actor,Has(i,"parity"));
    return true;
  }
  if (i.opcode=="tensormap" && Has(i,"replace") && Has(i,"tile")) {
    // Table atoms (class mode) as their plain values: DecodeTensorMap reads integers and pointers.
    auto v=PlainValue(value(operand(d.operands.size()-1)));
    std::optional<std::int64_t> ordinal;
    if (d.operands.size()==3) ordinal=number(operand(1));
    const auto target=required_address(operand(0));
    if (!runtime.tensor_fields.contains(target)) {
      auto original=runtime.Read(site,target,128);
      if (auto fragment=std::get_if<DescriptorFragment>(&original);
          fragment && fragment->offset==0 && fragment->bytes==128 && c_.tma_descriptors.contains(fragment->descriptor))
        runtime.tensor_templates[target]=c_.tma_descriptors.at(fragment->descriptor);
    }
    runtime.tensor_fields[target][{i.modifiers.at(2),ordinal}]=std::move(v);
    runtime.constants.Invalidate(target,128);
    return true;
  }
  if (i.opcode=="tensormap" && Has(i,"cp_fenceproxy")) {
    auto text=TrimView(operand(0)); if (text.starts_with('[') && text.ends_with(']')) text=TrimView(text.substr(1,text.size()-2));
    const auto split=SplitAddress(text);
    auto v=split?PlainValue(value(split->first)):Scalar{};
    auto *destination=std::get_if<AllocationPointer>(&v);
    if (!destination && split) {
      auto reg=ids_.find(split->first);
      if (reg!=ids_.end()) {
        try { UnresolvedControl(i,state,trace,reg->second); }
        catch (ExecutionError &error) {
          if (error.code=="branch_not_static") error.code="unresolved_tma_descriptor_location";
          throw ExecutionError(error.code,i.line,
              "tensor map copy destination cannot be resolved",error.required_parameters);
        }
      }
    }
    if (!destination || number(operand(2))!=128) throw ExecutionError("unresolved_tma_descriptor_location",i.line,"tensor map copy destination or size unresolved");
    destination->offset+=split->second;
    auto fields=runtime.tensor_fields.find(required_address(operand(1)));
    if (fields!=runtime.tensor_fields.end()) {
      auto decoded=DecodeTensorMap(fields->second,*destination);
      if (auto initial=runtime.tensor_templates.find(fields->first);initial!=runtime.tensor_templates.end())
        decoded["initial_descriptor"]=initial->second;
      PublishTensorMap(runtime.tensor_maps,decoded);
    }
    return true;
  }
  if (i.opcode=="mbarrier") {
    if (Has(i,"init")) runtime.Init(required_address(operand(0)),number(operand(1)));
    else if (Has(i,"inval")) { runtime.barriers.erase(required_address(operand(0))); ++runtime.revision; }
    else if (Has(i,"arrive")) {
      auto arrivals=std::int64_t(1), tx=std::int64_t(0);
      if (Has(i,"expect_tx")) tx=number(operand(2));
      else if (d.operands.size()>2) arrivals=number(operand(2));
      Set(d,state,{Integer(runtime.Update(required_address(operand(1)),arrivals,tx,actor))});
    } else if (Has(i,"expect_tx") || Has(i,"complete_tx"))
      runtime.Update(required_address(operand(0)),0,
          (Has(i,"expect_tx")?1:-1)*number(operand(1)),actor);
    else if (IsWait(i)) return false;
    else throw ExecutionError("shared_control_mbarrier_variant",i.line,"unsupported shared-control mbarrier variant");
    return true;
  }
  if (i.opcode=="bar" || i.opcode=="barrier") {
    auto prefix="cta."+std::to_string(rank)+".";
    if (Has(i,"warp") && Has(i,"sync")) {
      const auto mask=std::uint32_t(number(operand(0)));
      if (!(mask&(1U<<lane))) throw ExecutionError("shared_control_invalid_sync",i.line,"executing lane absent from warp mask");
      runtime.Collective(prefix+"warp."+std::to_string(warp)+"."+std::to_string(mask),std::popcount(mask),actor,key);
    } else if (Has(i,"cluster")) {
      if (Has(i,"arrive")) {
        runtime.Collective("cluster",c_.Threads()*cluster_size,actor,key,false);
        runtime.cluster_tickets[actor]=key;
      } else if (Has(i,"wait")) {
        if (!runtime.cluster_tickets.contains(actor)) throw ExecutionError("shared_control_invalid_sync",i.line,"cluster wait has no arrival");
        runtime.Collective("cluster",c_.Threads()*cluster_size,actor,runtime.cluster_tickets.at(actor));
      } else throw ExecutionError("shared_control_collective_variant",i.line,"unsupported cluster collective");
    } else if (Has(i,"sync") || Has(i,"arrive"))
      runtime.Collective(prefix+"named."+std::to_string(number(operand(0))),
          d.operands.size()>1?number(operand(1)):std::int64_t(c_.Threads()),actor,key,Has(i,"sync"));
    else throw ExecutionError("shared_control_collective_variant",i.line,"unsupported shared-control collective");
    return true;
  }
  if (i.opcode=="clusterlaunchcontrol" && i.semantics.operation=="try_cancel_complete_tx") {
    const auto destination=required_address(operand(0));
    const auto barrier=required_address(operand(1));
    const bool multicast=Has(i,"multicast::cluster::all");
    const auto count=multicast?cluster_size:1U;
    for (unsigned target=0;target<count;++target) {
      const auto owner=std::uint64_t(multicast?target:rank)<<32;
      // The response is opaque even when cancellation is assumed to fail.
      // Publish unknown bytes, and complete the 16-byte transaction in each
      // destination CTA. Query normalization remains responsible for its value.
      runtime.Defer(owner|(barrier&UINT32_MAX),0,-16,
          {{site,owner|(destination&UINT32_MAX),std::vector<SharedByte>(16)}});
    }
    return true;
  }
  if (i.opcode=="st" && Has(i,"async") && Has(i,"mbarrier::complete_tx::bytes")) {
    if (!i.memory_width) throw ExecutionError("shared_control_operand",i.line,"unknown asynchronous store width");
    runtime.Defer(required_address(operand(2)),0,-std::int64_t(*i.memory_width),
        {{site,required_address(operand(0)),encoded_values(operand(1),*i.memory_width)}});
    return true;
  }
  if (In(i.opcode,{"ld","st"}) && (Has(i,"shared") || Has(i,"shared::cta") || Has(i,"shared::cluster"))) {
    const bool store=i.opcode=="st"; auto pointer=address(operand(store?0:1));
    if (store) {
      if (!pointer || !i.memory_width) runtime.Clear();
      else {
        auto bytes=encoded_values(operand(1),*i.memory_width);
        if (std::none_of(bytes.begin(),bytes.end(),[](auto b){return b.has_value();}))
          runtime.Invalidate(*pointer,bytes.size());
        else runtime.Write({site,*pointer,std::move(bytes)});
      }
      ++runtime.revision;
    } else {
      std::vector<Scalar> values;
      const auto count=d.destination_slots.size();
      const auto width=i.memory_width && count?*i.memory_width/count:0;
      const bool sign=width && Has(i,"s"+std::to_string(width*8));
      for (std::size_t k=0;k<count;++k) {
        // A bounded integer comes back whole (same store width, range fits the loaded type).
        auto loaded=pointer && width?runtime.Read(site,*pointer+k*width,width,sign):Scalar{};
        if (auto n=ScalarInteger(loaded); n && sign && ((*n>>(width*8-1))&1)!=0)
          loaded=Integer(*n-(Integer(1)<<(width*8)));
        values.push_back(std::move(loaded));
      }
      Set(d,state,std::move(values));
    }
    return true;
  }
  if (i.opcode=="cp" && Has(i,"tensor") && Has(i,"mbarrier::complete_tx::bytes")) {
    const auto &facts=context.resources->facts; const auto record_lane=facts.Register(trace);
    auto &adapter=context.resources->resources; adapter.SetLane(record_lane);
    TmaAccesses transfer;
    // The runtime maps are searched after the program's own.
    try { transfer=adapter.Tma(facts.Detached(record,record_lane),TmaRequest(contract,false,&runtime.tensor_maps)); }
    catch (const SemanticError &error) { throw ExecutionError(error.code,i.line,error.what()); }
    std::vector<std::pair<std::uint64_t,std::uint64_t>> completions;
    for (const auto &[id,bytes] : transfer.completions) {
      const auto [kind,owner,identity]=adapter.interner().Location(id);
      if (kind!="physical") throw ExecutionError("shared_control_operand",i.line,"nonphysical TMA completion address");
      const auto target=(std::uint64_t(adapter.frame.RankOf(owner).value)<<32) | std::stoull(identity);
      if (!runtime.barriers.contains(target)) throw SharedBlocked{};
      completions.emplace_back(target,bytes);
    }
    for (const auto &pair : transfer.pairs) if (pair.destination) {
      const auto &binding=adapter.Binding(pair.destination->first);
      if (!binding.expression.empty() || !pair.destination->second) runtime.Clear();
      else runtime.Invalidate((std::uint64_t(pair.target_rank)<<32)|binding.offset,*pair.destination->second);
    }
    for (const auto &[target,bytes] : completions) runtime.Update(target,0,-std::int64_t(bytes),actor);
    return true;
  }
  if (i.opcode=="cp" && Has(i,"async") && !Has(i,"bulk")) {
    auto &pending=runtime.pending_copies[actor]; auto &groups=runtime.copy_groups[actor];
    if (Has(i,"mbarrier") && Has(i,"arrive")) {
      const auto target=required_address(operand(0));
      if (!Has(i,"noinc")) runtime.Update(target,-1,0,actor);
      for (auto &group : groups) {
        for (const auto &copy : group) runtime.Write(copy);
        group.clear();
      }
      for (const auto &copy : pending) runtime.Write(copy);
      pending.clear();
      runtime.Defer(target,1,0); return true;
    }
    if (Has(i,"commit_group") || Has(i,"wait_all")) { groups.push_back(std::move(pending)); pending.clear(); }
    if (Has(i,"wait_group") || Has(i,"wait_all")) {
      const auto keep=Has(i,"wait_group")?number(operand(0)):0;
      if (keep<0 || keep>7) throw ExecutionError("shared_control_operand",i.line,"invalid cp.async wait count");
      while (groups.size()>std::uint64_t(keep)) {
        for (const auto &copy : groups.front()) runtime.Write(copy);
        groups.pop_front(); ++runtime.revision;
      }
      return true;
    }
    if (Has(i,"commit_group")) return true;
    if ((Has(i,"shared") || Has(i,"shared::cta")) && Has(i,"global") && (d.operands.size()==3 || d.operands.size()==4)) {
      // src-size bytes are copied and the rest zero-filled; a true ignore-src predicate copies
      // none. A data-dependent src-size or ignore-src leaves every destination byte unknown.
      const auto width=number(operand(2));
      std::optional<std::int64_t> source_width=width;
      if (d.operands.size()==4) {
        const auto control=value(operand(3));
        if (const auto *ignore=std::get_if<bool>(&control)) source_width=*ignore?0:width;
        else if (ScalarInteger(control)) source_width=number(operand(3));
        else source_width.reset();
      }
      if (!In(std::to_string(width),{"4","8","16"}) || (source_width && (*source_width<0 || *source_width>width)))
        throw ExecutionError("shared_control_operand",i.line,"invalid cp.async width");
      Scalar loaded;
      auto text=TrimView(operand(1)); text=text.substr(1,text.size()-2);
      if (auto split=SplitAddress(text)) {
        auto source=value(split->first);
        if (auto p=std::get_if<AllocationPointer>(&source);p && constant_inputs_.contains(p->resource) && source_width && *source_width) {
          auto n=constant_inputs_.at(p->resource).ReadInteger(Int64(p->offset+split->second,"source address"),*source_width,false);
          if (n) loaded=*n;
        }
      }
      auto bytes=SharedBytes(loaded,source_width.value_or(width));
      if (source_width) bytes.resize(width,std::uint8_t(0));
      pending.push_back({site,required_address(operand(0)),std::move(bytes)}); return true;
    }
  }
  if (i.opcode=="tcgen05" && Has(i,"commit")) {
    const auto target=required_address(operand(0));
    if (Has(i,"multicast::cluster")) {
      const auto mask=std::uint32_t(number(operand(1)))&65535;
      if (!mask || mask>=(1U<<std::min(cluster_size,16U)))
        throw ExecutionError("shared_control_async_completion",i.line,"commit multicast mask outside cluster");
      for (unsigned r=0;r<cluster_size;++r) if (mask&(1U<<r))
        runtime.Defer((std::uint64_t(r)<<32)|(target&UINT32_MAX),1,0);
    } else runtime.Defer(target,1,0);
    return true;
  }
  if (const auto untracked=UntrackedShared(i)) {
    // A write whose bytes this model does not track: the bytes it may write are forgotten (every
    // byte when its target is unknown), so no later read reassembles an older value across it.
    // Its results are computed as usual (Data).
    if (i.address_lanes && lane>=*i.address_lanes) return false; // stmatrix: no row address
    const auto &text=operand(untracked->operand);
    std::optional<std::uint64_t> target;
    if (untracked->generic) {
      // Only an executor shared address (an integer below 2^32, as cvta keeps it, or a cluster
      // shared pointer) is shared memory; a pointer into a global object or a parameter is not.
      auto inner=TrimView(text);
      if (inner.starts_with('[') && inner.ends_with(']')) inner=TrimView(inner.substr(1,inner.size()-2));
      const auto split=SplitAddress(inner);
      const auto base=split?PlainValue(value(split->first)):Scalar{};
      if (std::holds_alternative<AllocationPointer>(base) || std::holds_alternative<ParameterPointer>(base) ||
          std::holds_alternative<BoundedAllocationPointer>(base)) return false;
      if (const auto *n=std::get_if<Integer>(&base)) {
        const Integer at=*n+split->second;
        if (at<0 || at>UINT32_MAX) return false;
        target=address(text);
      } else if (std::holds_alternative<ClusterSharedPointer>(base)) target=address(text);
    } else target=address(text);
    if (!target || !untracked->bytes) runtime.Clear();
    else {
      runtime.Invalidate(*target,untracked->bytes);
      // cta_group::2: the pair's peer CTA (rank ^ 1) may receive the address at the same offset.
      if (i.opcode=="tcgen05" && Has(i,"cta_group::2") && cluster_size>1)
        runtime.Invalidate(*target^(std::uint64_t(1)<<32),untracked->bytes);
    }
    return false;
  }
  if (i.opcode=="cp" && Has(i,"async") && Has(i,"bulk") &&
      (Has(i,"commit_group") || Has(i,"wait_group") ||
       (Has(i,"bulk_group") && (Has(i,"tensor") || !Has(i,"reduce"))))) return false;
  if (In(i.opcode,{"cp","st"}) && Has(i,"async"))
    throw ExecutionError("shared_control_async_memory",i.line,"unsupported shared-control asynchronous transfer");
  return false;
}
} // namespace
} // namespace sentryward::frontend::ptx
