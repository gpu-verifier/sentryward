#pragma once
#include "lowering_impl.hpp"

namespace sentryward::frontend::ptx {
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
  const auto [bounded_writes,bounded_reads]=cta_facts.SharedBoundedCounts();
  shared_bounded_writes+=bounded_writes;shared_bounded_reads+=bounded_reads;
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
  CtaExitJoin(cta_facts, actor_begin);
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
} // namespace sentryward::frontend::ptx
