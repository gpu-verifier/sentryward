#include "sentryward/frontend/ptx/semantic_program.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "sentryward/value_ir/operations.h"
#include "sentryward/value_analysis/memory_instance_set.h"
#include "sentryward/value_ir/native_matrix.h"
#include "sentryward/frontend/ptx/layouts.h"
#include "sentryward/frontend/ptx/branch_scope_capture.h"
#include "sentryward/algorithm/diagnostics/memory_profile.h"
#include "sentryward/algorithm/diagnostics/performance_profile.h"
#include "sentryward/algorithm/base/affine_index.h"

namespace sentryward::frontend::ptx {
namespace {
namespace a = algorithm;
namespace v = value_ir;
namespace va = value_analysis;

struct SourceAccess {
  SourceMemoryKey key;
  a::AccessMode mode;
  a::AddressId address;
  std::optional<std::uint64_t> bytes;
  bool immediate;
  bool ordered_native;
};

std::vector<SourceAccess> Accesses(const a::SourceInstruction& instruction) {
  std::vector<SourceAccess> result;
  const auto add = [&](a::AccessOperand operand, a::AccessMode mode, a::AddressId address,
                       std::optional<std::uint64_t> bytes, bool immediate,
                       std::uint32_t effect = 0, std::uint32_t access = 0) {
    const auto opcode = instruction.ptx_origin ? instruction.ptx_origin->opcode.view() : std::string_view{};
    const bool ordered_native = opcode.starts_with("ldmatrix.") || opcode.starts_with("stmatrix.") ||
        opcode.starts_with("tcgen05.ld.") || opcode.starts_with("tcgen05.st.") ||
        opcode.starts_with("clusterlaunchcontrol.try_cancel.") || opcode.starts_with("cp.async.bulk.tensor.");
    result.push_back({{instruction.id, operand, effect, access}, mode, address, bytes, immediate, ordered_native});
  };
  std::visit([&](const auto& op) {
    using T = std::decay_t<decltype(op)>;
    if constexpr (std::is_same_v<T, a::ImmediateLoad> || std::is_same_v<T, a::ImmediateStore>) {
      add(a::AccessOperand::kImmediate, std::is_same_v<T, a::ImmediateLoad> ? a::AccessMode::kRead : a::AccessMode::kWrite,
          op.address, op.bytes, true);
    } else if constexpr (std::is_same_v<T, a::AsyncLoad> || std::is_same_v<T, a::AsyncStore>) {
      add(a::AccessOperand::kSource, a::AccessMode::kRead, op.source, op.bytes, false);
      add(a::AccessOperand::kDestination, a::AccessMode::kWrite, op.destination, op.bytes, false);
    } else if constexpr (std::is_same_v<T, a::AsyncOp>) {
      for (std::uint32_t effect = 0; effect < op.effects.size(); ++effect)
        for (std::uint32_t access = 0; access < op.effects[effect].accesses.size(); ++access) {
          const auto& memory = op.effects[effect].accesses[access];
          if (const auto* address = std::get_if<a::AddressId>(&memory.endpoint))
            add(a::AccessOperand::kEffect, memory.mode, *address, memory.bytes, false, effect, access);
        }
    }
  }, instruction.op);
  return result;
}

std::optional<SemanticOccurrence> Occurrence(const v::SourceOrigin& origin) {
  if (!origin.cta || !origin.warp || !origin.lane || !origin.instruction_index || !origin.occurrence)
    return std::nullopt;
  for (const auto id : {*origin.cta, *origin.warp, *origin.lane, *origin.instruction_index})
    if (id > UINT32_MAX) return std::nullopt;
  return SemanticOccurrence{static_cast<std::uint32_t>(*origin.cta), static_cast<std::uint32_t>(*origin.warp),
      static_cast<std::uint32_t>(*origin.lane), static_cast<std::uint32_t>(*origin.instruction_index), *origin.occurrence};
}

std::string AllocationIdentity(const a::ConstantAddressBinding& binding) {
  const auto field = [](const std::string& value) { return std::to_string(value.size()) + ':' + value; };
  return field(binding.address_space) + field(binding.allocation);
}

struct ValueAccess {
  va::MemoryInstance reference;
  a::AccessMode mode;
  std::optional<std::uint64_t> bytes;
  bool ordinary_scalar = false;
  bool ordered_payload = false;
};

std::optional<ValueAccess> Access(const v::Graph& graph, const v::Node& node, std::size_t instance) {
  const auto* load = dynamic_cast<const v::LoadOp*>(&node.operation());
  const auto* store = dynamic_cast<const v::StoreOp*>(&node.operation());
  if (!load && !store) return std::nullopt;
  ValueAccess result{{node.id(), instance}, load ? a::AccessMode::kRead : a::AccessMode::kWrite, {}, false};
  const auto& record = node.instances().at(instance);
  const auto& attributes = load ? load->attributes() : store->attributes();
  if(load && !record.operands.empty() && record.result_types.size()==1 && record.result_types[0].isMatrix()) {
    const auto* native=dynamic_cast<const v::NativeMatrixOp*>(&graph.definingNode(record.operands[0]).operation());
    result.ordered_payload=native && native->opcode()==v::NativeMatrixOpcode::TcgenRegisterAddress;
  }
  const v::ValueType* type = nullptr;
  if (load && record.result_types.size() == 1) type = &record.result_types[0];
  if (store && record.operands.size() >= 2) type = &graph.type(record.operands[1]);
  if (type && type->hasStaticShape() && !type->element().isParameterized()) {
    const auto bits = type->bitSize();
    if (bits && bits % 8 == 0) result.bytes = bits / 8;
    // A tensor-map store (tensormap.cp_fenceproxy) writes plain descriptor bytes.
    result.ordinary_scalar = !attributes.guarded && !attributes.layout && !attributes.descriptor &&
        (attributes.interpretation == v::MemoryAttributes::Interpretation::Ordinary ||
         (store && attributes.interpretation == v::MemoryAttributes::Interpretation::TensorMap));
  }
  return result;
}

void Diagnostic(SemanticProgram& result, std::string code, std::string message,
                std::optional<SemanticOccurrence> occurrence = {}, std::optional<a::InstId> instruction = {},
                std::optional<va::MemoryInstance> value = {}) {
  result.diagnostics.push_back({std::move(code), std::move(message), occurrence, instruction, value});
}

auto RecordIdentity(const RecordView& record) {
  return std::tie(record.instruction_id(), record.occurrence(), record.predicate_value(),
      record.branch_scope_decision(),
      record.collapsed_wait_loop(), record.normalized_wait_result(), record.normalized_clc_result(),
      record.canonical_wait_probe(), record.wait_exit_label(),
      record.schedule_seed_outcome(), record.probe_fallback_key(), record.opaque_compute_diamond_indices(),
      record.collapsed_wait_indices(), record.opaque_compute_join_label(), record.assumed_equality(),
      record.opaque_compute_register_reads(), record.opaque_compute_register_writes(),
      record.merged_arm_join());
}

// A lane's record at a cohort position, or none past its end.
using CohortRecord = std::optional<RecordView>;
struct RecordLess {
  bool operator()(const CohortRecord& first, const CohortRecord& second) const {
    if (!first || !second) return !first && second;
    return RecordIdentity(*first) < RecordIdentity(*second);
  }
};

struct DynamicKeyHash {
  std::size_t operator()(DynamicKey key) const noexcept {
    return std::hash<std::uint64_t>{}(key.occurrence) ^ (std::hash<std::uint32_t>{}(key.instruction) << 1);
  }
};

void BuildCohorts(std::uint32_t cta, const CtaFacts& facts,
    const a::SourceProgram& source, std::size_t instruction_begin, std::vector<ExecutionCohort>& cohorts,
    std::vector<std::optional<ExecutionCohortId>>& bindings, const std::function<void()>& poll) {
  const auto count = facts.LaneCount();
  if (!count) return;
  std::map<std::pair<std::uint32_t, std::uint32_t>, std::size_t> lanes;
  for (std::size_t lane = 0; lane < count; ++lane)
    lanes.emplace(std::pair{facts.Warp(facts.Lane(lane)), facts.LaneId(facts.Lane(lane))}, lane);
  const auto template_groups = facts.TemplateGroups();
  if (facts.TemplateExecution() && template_groups.size()==1 &&
      template_groups.front().size()==count) {
    const auto root=cohorts.size();
    cohorts.push_back({root,{},cta,count});
    std::unordered_set<DynamicKey,DynamicKeyHash> occurrences;
    const auto first_lane=facts.Lane(0);
    for(std::size_t i=0;i<facts.Size(first_lane);++i) {
      if((i&1023)==0&&poll)poll();
      occurrences.insert(facts.Meta({first_lane,i}).Key());
    }
    bindings.resize(source.instructions.size());
    for(std::size_t i=instruction_begin;i<source.instructions.size();++i) {
      if((i&1023)==0&&poll)poll();
      const auto instruction=source.instructions[i];
      if(!instruction.ptx_origin||instruction.actor.value()>=source.actors.size())continue;
      const auto& actor=source.actors[instruction.actor.value()];
      if(actor.id==instruction.actor&&actor.cta.value()==cta&&lanes.contains({actor.warp.value(),actor.lane})&&
          occurrences.contains({instruction.ptx_origin->instruction_index,instruction.ptx_origin->occurrence}))
        bindings.at(instruction.id.value())=root;
    }
    return;
  }
  std::vector<std::unordered_map<DynamicKey, std::vector<a::InstId>, DynamicKeyHash>> requests(count);
  bindings.resize(source.instructions.size());
  for (std::size_t index = instruction_begin; index < source.instructions.size(); ++index) {
    const auto& instruction = source.instructions[index];
    if (!instruction.ptx_origin || instruction.actor.value() >= source.actors.size()) continue;
    const auto& actor = source.actors[instruction.actor.value()];
    if (actor.id != instruction.actor || actor.cta.value() != cta) continue;
    const auto lane = lanes.find({actor.warp.value(), actor.lane});
    if (lane == lanes.end()) continue;
    requests[lane->second][{instruction.ptx_origin->instruction_index, instruction.ptx_origin->occurrence}].push_back(instruction.id);
  }
  struct Work {
    ExecutionCohortId group;
    std::size_t position;
    std::vector<std::size_t> lanes;
  };
  const auto root = cohorts.size();
  cohorts.push_back({root, {}, cta, count});
  std::vector<std::size_t> members(count);
  std::iota(members.begin(), members.end(), 0);
  std::vector<Work> pending{{root, 0, std::move(members)}};
  std::size_t visits = 0;
  while (!pending.empty()) {
    auto work = std::move(pending.back());
    pending.pop_back();
    for (;;) {
      const auto record = [&](std::size_t lane) -> CohortRecord {
        const auto ref = facts.Lane(lane);
        return work.position < facts.Size(ref) ? CohortRecord(facts.Meta({ref, work.position})) : std::nullopt;
      };
      const auto first = record(work.lanes.front());
      const bool identical = std::all_of(work.lanes.begin() + 1, work.lanes.end(), [&](auto lane) {
        const auto other = record(lane);
        // Absent records compare by presence; present ones by their metadata,
        // which is trivially equal when both lanes share one metadata object.
        return (!first || !other) ? first.has_value() == other.has_value()
                                  : FactSource::SameMetadata(*first, *other) ||
                                        RecordIdentity(*first) == RecordIdentity(*other);
      });
      if (!identical) {
        std::map<CohortRecord, std::vector<std::size_t>, RecordLess> partitions;
        for (auto lane : work.lanes) partitions[record(lane)].push_back(lane);
        for (auto& [_, partition] : partitions) {
          const auto child = cohorts.size();
          cohorts.push_back({child, work.group, cta, partition.size()});
          pending.push_back({child, work.position, std::move(partition)});
        }
        break;
      }
      if (!first) break;
      for (auto lane : work.lanes) {
        if ((++visits & 1023) == 0 && poll) poll();
        auto& queries = requests[lane];
        const auto found = queries.find(first->Key());
        if (found == queries.end()) continue;
        for (auto instruction : found->second) bindings.at(instruction.value()) = work.group;
        queries.erase(found);
      }
      ++work.position;
    }
  }
}

void BuildSyncGroups(SemanticProgram& result, const std::vector<std::optional<ExecutionCohortId>>& cohorts) {
  std::map<a::ActorId, const a::ActorInfo*> actors;
  for (const auto& actor : result.lowering.artifact.program.actors) actors.emplace(actor.id, &actor);
  using GroupKey = std::tuple<ExecutionCohortId, std::uint32_t, std::uint64_t, std::size_t, std::size_t>;
  std::map<GroupKey, std::size_t> groups;
  std::map<SemanticOccurrence, std::size_t> lowerings;
  for (const auto& instruction : result.lowering.artifact.program.instructions) {
    const auto actor = actors.find(instruction.actor);
    if (actor != actors.end() && !instruction.ptx_origin && Accesses(instruction).empty()) {
      result.synthetic_synchronization_nodes.push_back(instruction.id);
      continue;
    }
    if (actor == actors.end() || !instruction.ptx_origin) {
      Diagnostic(result, "missing_semantic_occurrence", "Source instruction has no complete thread/PTX occurrence identity", {}, instruction.id);
      continue;
    }
    const auto& thread = *actor->second;
    const auto& origin = *instruction.ptx_origin;
    const SemanticOccurrence occurrence{thread.cta.value(), thread.warp.value(), thread.lane,
        origin.instruction_index, origin.occurrence};
    const auto lowering_index = lowerings[occurrence]++;
    const auto group = instruction.id.value() < cohorts.size() ? cohorts[instruction.id.value()] : std::nullopt;
    if (!group) {
      Diagnostic(result, "missing_synchronization_group", "Source occurrence has no exact execution-prefix cohort", occurrence, instruction.id);
      continue;
    }
    const GroupKey key{*group, origin.instruction_index, origin.occurrence, instruction.op.index(), lowering_index};
    auto [entry, inserted] = groups.try_emplace(key, result.synchronization_groups.size());
    if (inserted)
      result.synchronization_groups.push_back({*group, origin.instruction_index, origin.occurrence, instruction.op.index(), lowering_index, {}});
    result.synchronization_groups[entry->second].members.push_back(instruction.id);
  }
}

void AppendMemoryBinding(SemanticMemoryBindings& bindings,SemanticMemoryBinding binding) {
  // Explicit singleton producers retain contiguous row storage.
  // Never reconstruct the last 128 logical records: earlier records may now
  // already be parameterized matrix-fragment runs.
  bindings.push_back(std::move(binding));
}

void LinkMmaMemory(SemanticProgram& result, const MmaValueLayout& layout,
    const SemanticOccurrence& occurrence, const std::vector<SourceAccess>& source,
    const std::vector<ValueAccess>& values,
    const std::map<a::AddressId,const a::ConstantAddressBinding*>& addresses) {
  SemanticMemoryBindings bindings;
  std::optional<SemanticMemoryBinding> pending_binding;
  const auto flush_binding=[&] {
    if(pending_binding) {
      bindings.push_back(*pending_binding);
      pending_binding.reset();
    }
  };
  const auto append_binding=[&](SemanticMemoryBinding binding,
      const SemanticMemoryBindingCodec::Delta& delta,std::size_t count) {
    // Coalesce adjacent contiguous physical pieces, not concrete instances.
    if(count==1) {
      if(pending_binding) {
        auto& p=*pending_binding;
        if(p.source==binding.source && p.value==binding.value && p.region && binding.region &&
            p.value_byte_offset && binding.value_byte_offset &&
            p.region->allocation==binding.region->allocation &&
            p.region->offset+p.region->bytes==binding.region->offset &&
            *p.value_byte_offset+p.region->bytes==*binding.value_byte_offset) {
          p.region->bytes+=binding.region->bytes;return;
        }
      }
      flush_binding();pending_binding=std::move(binding);
    } else {
      flush_binding();bindings.appendAffineColumns({binding},{delta},count);
    }
  };
  try {
    using Key = std::pair<MmaOperandRole,a::AccessMode>;
    std::map<Key,const ValueAccess*> operands;
    std::string kind;
    for (const auto& value : values) {
      const auto& instance = result.values.node(value.reference.node).instances().at(value.reference.instance);
      const auto* address = dynamic_cast<const v::NativeMatrixOp*>(&result.values.definingNode(instance.operands.at(0)).operation());
      if (!address || address->opcode()!=v::NativeMatrixOpcode::TcgenMmaAddress || !value.bytes)
        throw LayoutError("MMA value needs a fixed shape and a native matrix address");
      MmaOperandRole role;
      switch (address->attributes().role) {
        case v::MatrixRole::A: role=MmaOperandRole::A; break;
        case v::MatrixRole::B: role=MmaOperandRole::B; break;
        case v::MatrixRole::D: role=MmaOperandRole::D; break;
        case v::MatrixRole::ScaleA: role=MmaOperandRole::ScaleA; break;
        case v::MatrixRole::ScaleB: role=MmaOperandRole::ScaleB; break;
        default: throw LayoutError("Scaled or sparse MMA value fragments are not decoded");
      }
      for (const auto& modifier : address->attributes().modifiers)
        if (modifier.starts_with("kind::")) kind=modifier.substr(6);
      if (!operands.emplace(Key{role,value.mode},&value).second)
        throw LayoutError("MMA has more than one value for the same operand role");
    }
    const bool scaled=operands.contains({MmaOperandRole::ScaleA,a::AccessMode::kRead});
    const auto descriptor=scaled ? DecodeBlockScaledDescriptor(layout.descriptor,kind) : DecodeMmaDescriptor(kind,layout.descriptor);
    // Two-CTA M=128 D uses the 2x2 datapath: each CTA's row r keeps columns
    // [0,N/2) in lane r and [N/2,N) in lane r+64 (CUTLASS tmem_frg_2sm,
    // M_MMA=64). TMEM-A and scale factors use duplicated layouts not decoded here.
    const bool two_by_two=layout.cta_group==2 && descriptor.m==128;
    if (two_by_two && (scaled || layout.a_in_tmem))
      throw LayoutError("Two-CTA M=128 MMA duplicated TMEM operand layout is not decoded");
    std::map<std::pair<std::uint32_t,std::uint32_t>,const MmaValueAccess*> tags;
    for (const auto& tag : layout.accesses) tags.emplace(std::pair{tag.effect,tag.access},&tag);
    using PartKey=std::tuple<MmaOperandRole,a::AccessMode,std::uint32_t>;
    std::map<PartKey,std::map<std::uint64_t,const SourceAccess*>> regions;
    for (const auto& access : source) {
      const auto tag=tags.find({access.key.effect_index,access.key.access_index});
      if (access.key.instruction!=layout.instruction || tag==tags.end() || !access.bytes)
        throw LayoutError("MMA source access has no matching operand-layout annotation");
      const auto& address=*addresses.at(access.address);
      if (!address.expression.empty()) throw LayoutError("MMA physical memory address is not constant");
      if (!regions[{tag->second->role,access.mode,tag->second->cta_part}].emplace(address.offset,&access).second)
        throw LayoutError("MMA role contains duplicate physical intervals");
    }
    std::map<SourceMemoryKey,std::vector<ByteInterval>> coverage;
    for (const auto& [key,intervals] : regions) {
      const auto [role,mode,part]=key;
      const auto operand=operands.find({role,mode});
      if (operand==operands.end()) throw LayoutError("MMA physical operand has no numerical value");
      const auto& value=*operand->second;
      const bool scale=role==MmaOperandRole::ScaleA || role==MmaOperandRole::ScaleB;
      const bool packed=scaled && (role==MmaOperandRole::A ? descriptor.a_element_bits==4 :
          role==MmaOperandRole::B && descriptor.b_element_bits==4);
      const auto rows=role==MmaOperandRole::ScaleB ? descriptor.n :
          (role==MmaOperandRole::B ? descriptor.n : descriptor.m)/layout.cta_group;
      const unsigned vector=kind=="mxf4"?2:kind=="mxf4nvf4"?4:1;
      const auto columns=scale ? vector : role==MmaOperandRole::D ? descriptor.n : packed ? descriptor.k/2 : descriptor.k;
      const auto bits=scale || (packed && role!=MmaOperandRole::D) ? 8 :
          role==MmaOperandRole::A ? descriptor.a_element_bits : role==MmaOperandRole::B ? descriptor.b_element_bits : descriptor.d_element_bits;
      if (bits%8) throw LayoutError("Subbyte MMA elements require packed element mapping");
      // Intersect physical runs with actual source fragments. The output is
      // an affine binding column, never an element vector to compress later.
      const auto emit=[&](const MatrixByteMapping& map) {
        if(!map.count || !map.element_bytes || map.count>UINT64_MAX/map.element_bytes ||
            map.physical_offset>UINT64_MAX-map.count*map.element_bytes)
          throw LayoutError("Invalid matrix mapping extent");
        auto physical=map.physical_offset;
        std::uint64_t member=0;
        while(member<map.count) {
          auto found=intervals.upper_bound(physical);
          if(found==intervals.begin())throw LayoutError("MMA logical element is outside its physical footprint");
          const auto& access=*std::prev(found)->second;
          const auto& address=*addresses.at(access.address);
          const auto offset=physical-address.offset;
          if(offset>=*access.bytes)throw LayoutError("MMA mapping crosses a physical gap");
          const auto count=std::min(map.count-member,(*access.bytes-offset)/map.element_bytes);
          if(!count)throw LayoutError("MMA logical element crosses a physical fragment");
          const auto logical=static_cast<unsigned __int128>(map.payload_offset)+
              static_cast<unsigned __int128>(member)*map.payload_stride;
          if(logical>UINT64_MAX || map.payload_stride>INT64_MAX || map.element_bytes>INT64_MAX)
            throw LayoutError("MMA payload coordinate overflow");
          const bool contiguous=map.payload_stride==map.element_bytes;
          const auto span=count*map.element_bytes;
          SemanticMemoryBinding binding{access.key,occurrence,value.reference,mode,access.address,*access.bytes,
              va::ConstantRegion{AllocationIdentity(address),physical,contiguous?span:map.element_bytes},
              {},offset,static_cast<std::uint64_t>(logical)};
          if(!address.alias_scope.empty() && address.address_space!="tensor") {
            binding.region.reset();binding.unsupported_reason="Matrix allocation may alias another allocation";
          }
          SemanticMemoryBindingCodec::Delta delta{};
          delta[9]=delta[10]=map.element_bytes;delta[11]=map.payload_stride;
          append_binding(std::move(binding),delta,contiguous?1:count);
          coverage[access.key].push_back({offset,span});
          physical+=span;member+=count;
        }
      };
      if (scale) {
        const bool b=role==MmaOperandRole::ScaleB;
        // Existing scale decoder validates selector, lane and TMEM bounds.
        // Each scale interval is a real physical fragment (including padding
        // between byte-sized scales), not an expanded thread domain.
        const auto elements=DecodeScaleFootprint(layout.scale_bases[b],(layout.descriptor>>(b?4:29))&3,rows,vector,b);
        for(std::size_t index=0;index<elements.size();++index) {
          const auto lane=index/(rows/32);
          const auto logical=((std::uint64_t{part}*4+lane/32)*rows+
              (index%(rows/32))*32+lane%32)*vector;
          emit({elements[index].offset,logical,vector,1,vector});
        }
      } else if (role==MmaOperandRole::B || (role==MmaOperandRole::A && !layout.a_in_tmem)) {
        const auto major=role==MmaOperandRole::A ? descriptor.a_major : descriptor.b_major;
        const bool padded=packed && kind=="mxf8f6f4";
        if(padded && (descriptor.k!=32 || major!="k"))
          throw LayoutError("FP4 padded shared mapping requires dense K-major K=32");
        ForEachSharedMatrixMapping(role==MmaOperandRole::A ? layout.a_base : layout.b_descriptor,
            rows,padded?descriptor.k:columns,bits,major,[&](MatrixByteMapping map) {
          if(padded) {
            // Each 16-byte physical chunk contains 8 payload bytes followed
            // by 8 padding bytes. Preserve the existing packed layout exactly.
            map.count=8;map.payload_offset/=2;
          }
          map.payload_offset+=std::uint64_t{part}*rows*columns*bits/8;
          emit(map);
        });
      }
      else {
        if(packed)throw LayoutError("FP4 TMEM-A logical packing is not implemented");
        const auto [lane,column]=DecodeTmemAddress(role==MmaOperandRole::A ? layout.a_base : layout.d_base);
        const unsigned halves=two_by_two ? 2 : 1, width=columns/halves;
        for (unsigned row=0;row<rows;++row)
          for (unsigned half=0;half<halves;++half) {
            const auto physical_row=descriptor.m==64 ? lane+32*(row/16)+row%16 : lane+row+64*half;
            const auto physical=EncodeCellByteOffset(physical_row,column);
            const auto logical=((std::uint64_t{part}*rows+row)*columns+std::uint64_t{half}*width)*bits/8;
            if(role==MmaOperandRole::D && bits!=32) {
              // The existing D16 footprint has a two-byte hole between cells;
              // these are distinct physical fragments, not contiguous bytes.
              for(unsigned k=0;k<width;++k)emit({physical+std::uint64_t{k}*4,
                  logical+std::uint64_t{k}*bits/8,bits/8,1,bits/8});
            } else emit({physical,logical,bits/8,width,bits/8});
          }
      }
    }
    flush_binding();
    for (const auto& access : source)
      if (MergeIntervals(coverage[access.key])!=std::vector<ByteInterval>{{0,*access.bytes}})
        throw LayoutError("MMA element mapping does not cover its complete physical access");
  } catch (const std::exception& error) {
    Diagnostic(result,"unsupported_mma_value_mapping",error.what(),occurrence,layout.instruction,
        values.empty()?std::nullopt:std::optional(values[0].reference));
    return;
  }
  bindings.forEachAffineColumn([&](const auto& first,const auto& delta,auto,auto,std::size_t count) {
    if(count==1)result.memory_bindings.push_back(first);
    else result.memory_bindings.appendAffineColumns({first},{delta},count);
  });
}

SemanticMemoryBinding BindMemoryValue(const SourceAccess& access, const ValueAccess& value,
    SemanticOccurrence occurrence, std::uint64_t offset,
    const a::ConstantAddressBinding* address) {
  SemanticMemoryBinding binding{access.key, occurrence, value.reference, access.mode,
      access.address, *access.bytes, {}, {}, offset};
  if (!address) binding.unsupported_reason = "Source access has no address binding";
  else if (!address->expression.empty()) binding.unsupported_reason = "Memory address remains symbolic";
  else if (!address->alias_scope.empty() && address->address_space!="tensor")
    binding.unsupported_reason = "Allocation may alias another allocation in its alias scope";
  else if (address->address_space.empty() || address->allocation.empty())
    binding.unsupported_reason = "Memory address has no concrete allocation identity";
  else if (offset > std::numeric_limits<std::uint64_t>::max() - address->offset)
    binding.unsupported_reason = "Vector component address overflows";
  else binding.region = va::ConstantRegion{AllocationIdentity(*address), address->offset + offset, *value.bytes};
  return binding;
}

bool LinkCopyMemory(SemanticProgram& result,const SemanticOccurrence& occurrence,
    const std::vector<SourceAccess>& source,const std::vector<ValueAccess>& values,
    const std::map<a::AddressId,const a::ConstantAddressBinding*>& addresses) {
  if(values.size()!=2)return false;
  const ValueAccess* read=nullptr;const ValueAccess* write=nullptr;
  for(const auto& value:values) {
    const auto& node=result.values.node(value.reference.node);
    const auto& instance=node.instances().at(value.reference.instance);
    const auto* address=dynamic_cast<const v::NativeMatrixOp*>(
        &result.values.definingNode(instance.operands.at(0)).operation());
    if(!address || address->opcode()!=v::NativeMatrixOpcode::TcgenCopyAddress)return false;
    (value.mode==a::AccessMode::kRead?read:write)=&value;
  }
  try {
    if(!read || !write || !read->bytes || read->bytes!=write->bytes || source.size()%2)
      throw LayoutError("TMEM copy needs matching fixed byte matrices");
    using Region=std::tuple<std::string,std::uint64_t,std::uint64_t>;
    std::map<Region,std::uint64_t> offsets;
    const auto& read_node=result.values.node(read->reference.node);
    const auto& read_instance=read_node.instances().at(read->reference.instance);
    const auto& copy=dynamic_cast<const v::NativeMatrixOp&>(
        result.values.definingNode(read_instance.operands.at(0)).operation());
    const bool replicated=std::ranges::find(copy.attributes().modifiers,"warpx4")!=copy.attributes().modifiers.end();
    std::uint64_t replicated_size=0;
    for(std::size_t i=0;i<source.size();i+=2) {
      const auto& input=source[i];const auto& output=source[i+1];
      if(input.mode!=a::AccessMode::kRead || output.mode!=a::AccessMode::kWrite || !input.bytes || input.bytes!=output.bytes ||
          input.key.instruction!=output.key.instruction || input.key.effect_index+1!=output.key.effect_index)
        throw LayoutError("TMEM copy fragment read/write pairing is invalid");
      const auto& address=*addresses.at(input.address);
      offsets.emplace(Region{AllocationIdentity(address),address.offset,*input.bytes},0);
      if(*input.bytes>UINT64_MAX-replicated_size)throw LayoutError("TMEM copy size overflows");
      replicated_size+=*input.bytes;
    }
    std::uint64_t size=0;
    std::optional<Region> previous;
    for(auto& [region,offset]:offsets) {
      const auto& [allocation,start,bytes]=region;
      if(bytes>UINT64_MAX-size || bytes>UINT64_MAX-start ||
          (previous && std::get<0>(*previous)==allocation &&
           std::get<1>(*previous)+std::get<2>(*previous)>start))
        throw LayoutError("TMEM copy source fragments overlap or overflow");
      offset=size;size+=bytes;previous=region;
    }
    if((replicated?replicated_size:size)!=*read->bytes)throw LayoutError("TMEM copy fragments do not cover its byte matrix");
    std::vector<SemanticMemoryBinding> bindings;
    std::uint64_t replica_offset=0;
    for(std::size_t i=0;i<source.size();i+=2) {
      const auto& input=*addresses.at(source[i].address);
      const auto offset=replicated?replica_offset:offsets.at(Region{AllocationIdentity(input),input.offset,*source[i].bytes});
      replica_offset+=*source[i].bytes;
      for(unsigned side=0;side<2;++side) {
        const auto& access=source[i+side];const auto& address=*addresses.at(access.address);
        if(!address.expression.empty() || (!address.alias_scope.empty() && address.address_space!="tensor"))
          throw LayoutError("TMEM copy needs concrete, non-aliased fragment addresses");
        bindings.push_back({access.key,occurrence,(side?write:read)->reference,access.mode,
            access.address,*access.bytes,va::ConstantRegion{AllocationIdentity(address),address.offset,*access.bytes},
            {},0,offset});
      }
    }
    for(auto& binding:bindings)AppendMemoryBinding(result.memory_bindings,std::move(binding));
  } catch(const std::exception& error) {
    Diagnostic(result,"unsupported_copy_value_mapping",error.what(),occurrence,
        source.empty()?std::nullopt:std::optional(source[0].key.instruction));
  }
  return true;
}

#include "semantic_memory_link.inc"

void LinkMemory(SemanticProgram& result, bool compact,
                const std::function<void(std::string_view)>& stage) {
  if (compact) {
    if (stage) stage("frontend.semantic_program.memory_link.templates");
    if (LinkMemoryDomains(result)) return;
  }
  if (stage) stage("frontend.semantic_program.memory_link.general");
  a::ProfileScope profile("frontend.semantic_program.memory_link.general");
  std::map<SemanticOccurrence, std::vector<SourceAccess>> source_accesses;
  std::map<SemanticOccurrence, std::vector<ValueAccess>> value_accesses;
  std::map<a::ActorId, const a::ActorInfo*> actors;
  for (const auto& actor : result.lowering.artifact.program.actors) actors.emplace(actor.id, &actor);
  for (const auto& instruction : result.lowering.artifact.program.instructions) {
    auto accesses = Accesses(instruction);
    if (accesses.empty()) continue;
    const auto actor = actors.find(instruction.actor);
    if (actor == actors.end() || !instruction.ptx_origin) continue;
    const auto& thread = *actor->second;
    const auto& origin = *instruction.ptx_origin;
    const SemanticOccurrence occurrence{thread.cta.value(), thread.warp.value(), thread.lane,
        origin.instruction_index, origin.occurrence};
    auto& target = source_accesses[occurrence];
    target.insert(target.end(), accesses.begin(), accesses.end());
  }
  for (const auto& node : result.values.nodes()) {
    if (!dynamic_cast<const v::LoadOp*>(&node.operation()) &&
        !dynamic_cast<const v::StoreOp*>(&node.operation())) continue;
    for (std::size_t instance = 0; instance < node.instanceCount(); ++instance) {
      const auto access = Access(result.values, node, instance);
      if (!access) continue;
      const auto occurrence = Occurrence(node.instanceOrigin(instance));
      if (!occurrence) {
        Diagnostic(result, "missing_value_occurrence", "Memory expression has no complete thread/PTX occurrence identity", {}, {}, access->reference);
        continue;
      }
      value_accesses[*occurrence].push_back(*access);
    }
  }
  std::map<a::AddressId, const a::ConstantAddressBinding*> addresses;
  for (const auto& binding : result.lowering.artifact.addresses) addresses.emplace(binding.address, &binding);
  std::map<a::InstId,const MmaValueLayout*> matrix_layouts;
  for (const auto& layout : result.lowering.mma_layouts) matrix_layouts.emplace(layout.instruction,&layout);
  std::set<SemanticOccurrence> occurrences;
  for (const auto& [key, _] : source_accesses) occurrences.insert(key);
  for (const auto& [key, _] : value_accesses) occurrences.insert(key);
  for (const auto& occurrence : occurrences) {
    const auto& source = source_accesses[occurrence];
    auto& values = value_accesses[occurrence];
    std::stable_sort(values.begin(), values.end(), [&](const auto& left, const auto& right) {
      return result.values.node(left.reference.node).instanceOrigin(left.reference.instance).lowering_index <
          result.values.node(right.reference.node).instanceOrigin(right.reference.instance).lowering_index;
    });
    if (source.empty() && values.empty()) continue;
    if(values.size()==1 && values.front().ordered_payload) {
      const auto& value=values.front();std::uint64_t payload=0;
      for(const auto& access:source) {
        if(!access.ordered_native || access.mode!=value.mode || !access.bytes ||
            !value.bytes || *access.bytes>*value.bytes-payload)
          throw LayoutError("Register payload does not match source fragments");
        auto part=value;part.bytes=access.bytes;
        const auto address=addresses.find(access.address);
        auto binding=BindMemoryValue(access,part,occurrence,0,address==addresses.end()?nullptr:address->second);
        binding.value_byte_offset=payload;payload+=*access.bytes;
        result.memory_bindings.push_back(std::move(binding));
      }
      if(value.bytes!=payload)throw LayoutError("Incomplete register payload mapping");
      continue;
    }
    if (LinkCopyMemory(result,occurrence,source,values,addresses)) continue;
    if (!source.empty())
      if (const auto layout=matrix_layouts.find(source[0].key.instruction); layout!=matrix_layouts.end()) {
        LinkMmaMemory(result,*layout->second,occurrence,source,values,addresses);
        continue;
      }
    bool ordered_native = source.size() == values.size() && !source.empty();
    for (std::size_t index = 0; ordered_native && index < source.size(); ++index)
      ordered_native = source[index].ordered_native && values[index].ordinary_scalar &&
          source[index].mode == values[index].mode && source[index].bytes &&
          source[index].bytes == values[index].bytes;
    std::map<a::AccessMode, const SourceAccess*> endpoints;
    std::map<a::AccessMode, std::uint64_t> totals;
    bool contiguous = !source.empty() && !values.empty();
    for (const auto& endpoint : source)
      if (!endpoint.bytes || !endpoints.emplace(endpoint.mode, &endpoint).second) contiguous = false;
    for (const auto& value : values) {
      auto& total = totals[value.mode];
      if (!contiguous || !value.ordinary_scalar || !value.bytes || !endpoints.contains(value.mode) ||
          *value.bytes > std::numeric_limits<std::uint64_t>::max() - total) { contiguous = false; break; }
      total += *value.bytes;
    }
    for (const auto& [mode, endpoint] : endpoints)
      if (!endpoint->bytes || totals[mode] != *endpoint->bytes) contiguous = false;
    if (!contiguous && !ordered_native) {
      Diagnostic(result, "unsupported_memory_value_mapping",
          "Exact memory/value mapping requires a unique endpoint per direction and matching scalar byte extents (" +
              std::to_string(source.size()) + " checked accesses, " + std::to_string(values.size()) + " value accesses)",
          occurrence, source.empty() ? std::nullopt : std::optional(source[0].key.instruction),
          values.empty() ? std::nullopt : std::optional(values[0].reference));
      continue;
    }
    std::map<a::AccessMode, std::uint64_t> offsets;
    std::size_t value_index = 0;
    for (const auto& value : values) {
    const auto& access = ordered_native ? source[value_index++] : *endpoints.at(value.mode);
    auto& offset = offsets[value.mode];
    if (ordered_native) offset = 0;
    const auto address = addresses.find(access.address);
    AppendMemoryBinding(result.memory_bindings,BindMemoryValue(access,value,occurrence,offset,
        address==addresses.end()?nullptr:address->second));
    offset += *value.bytes;
    }
  }
}
}  // namespace

struct SemanticProgramBuilder::Impl {
  const Module& module;
  const Function& function;
  v::Graph graph;
  std::unique_ptr<BranchScopeCapture> branch_capture;
  ValueIrExtractor values;
  ModelLowerer model;
  std::function<void()> poll;
  std::set<std::uint32_t> ctas;
  std::vector<ExecutionCohort> cohorts;
  std::vector<std::optional<ExecutionCohortId>> source_cohorts;
  bool finished = false;
  std::size_t address_queries = 0, address_resolved = 0;
  std::string address_unavailable;
  SemanticProgramOptions options;
  Impl(const Module& m, const Function& f, std::function<void()> poll, SemanticProgramOptions options)
      : module(m), function(f),
        branch_capture(options.capture_branch_scope ? std::make_unique<BranchScopeCapture>(graph) : nullptr),
        values(graph, true, true, branch_capture.get()), model(m, f, poll, true, options.atomics), poll(std::move(poll)), options(options) {}
};

SemanticProgramBuilder::SemanticProgramBuilder(const Module& module, const Function& function,
    std::function<void()> poll, SemanticProgramOptions options)
    : impl_(std::make_unique<Impl>(module, function, std::move(poll), options)) {}
SemanticProgramBuilder::~SemanticProgramBuilder() = default;
const BranchScopeCapture& SemanticProgramBuilder::BranchScopeForExport() const {
  if(impl_->finished || !impl_->branch_capture || !impl_->values.complete())
    throw std::invalid_argument("branch scope requires complete original emission");
  if(!impl_->branch_capture->failure().empty())
    throw std::invalid_argument("branch scope unavailable: "+impl_->branch_capture->failure());
  return *impl_->branch_capture;
}
void SemanticProgramBuilder::AddCta(const CtaFacts& facts) {
  const auto cta = facts.Linear();
  if (impl_->finished) throw std::logic_error("SemanticProgramBuilder is finished");
  if (!facts.Specialized() || &facts.module() != &impl_->module || &facts.function() != &impl_->function)
    throw std::invalid_argument("SemanticProgramBuilder requires its own module/function execution");
  if (impl_->ctas.contains(cta)) throw std::invalid_argument("Duplicate CTA in SemanticProgramBuilder");
  std::set<PathKey> validated_paths;
  for (std::size_t lane_index = 0; lane_index < facts.LaneCount(); ++lane_index) {
    const auto lane = facts.Lane(lane_index);
    if (validated_paths.insert(facts.Path(lane)).second)
      for (std::size_t index = 0; index < facts.Size(lane); ++index)
        if (facts.Meta({lane, index}).occurrence() > UINT32_MAX) throw std::invalid_argument("Source occurrence ID exceeds the source language range");
  }
  impl_->ctas.insert(cta);
  const auto first_value_node = impl_->graph.size();
  impl_->values.AddCta(facts);
  ValueAddressIndex address_values(impl_->graph, facts, first_value_node,
                                  impl_->options.defer_address_serialization);
  if (a::MemoryProfileEnabled()) {
    a::MemoryReport report("frontend.values", "value_graph", &impl_->graph);
    report.Metric("cta", cta);
    report.Add("nodes", impl_->graph.size()*sizeof(v::Node), impl_->graph.size()*sizeof(v::Node), impl_->graph.size(), 0, true);
    report.Add("groups", impl_->graph.groups().size()*sizeof(v::ThreadGroup), impl_->graph.groups().size()*sizeof(v::ThreadGroup), impl_->graph.groups().size(), 0, true);
    std::set<const void*> member_storage;
    for (const auto& group : impl_->graph.groups())
      if(member_storage.insert(group.threads.storageIdentity()).second)
        report.Vector("group_members", group.threads.values());
    std::set<const v::ValueType*> type_storage;
    for (const auto& node : impl_->graph.nodes()) {
      report.Metric("stored_instances", node.storedInstanceCount());
      report.Add("instances", node.instanceCount()*sizeof(v::Instance), node.instanceCount()*sizeof(v::Instance), node.instanceCount(), 0, true);
      for (const auto& instance : node.instances()) {
        report.Vector("operands", instance.operands);
        if (!instance.result_types.empty() && type_storage.insert(instance.result_types.data()).second)
          report.Add("result_types", instance.result_types.size()*sizeof(v::ValueType), instance.result_types.size()*sizeof(v::ValueType), instance.result_types.size());
        if (!instance.element_types.empty() && type_storage.insert(instance.element_types.data()).second)
          report.Add("element_types", instance.element_types.size()*sizeof(v::ValueType), instance.element_types.size()*sizeof(v::ValueType), instance.element_types.size());
        report.Vector("unroll_context", instance.origin.unroll_context);
        if (instance.origin.function) report.String("function_names", *instance.origin.function);
        if (instance.origin.source_file) report.String("source_files", *instance.origin.source_file);
        if (instance.origin.source_text) report.String("source_text", *instance.origin.source_text);
        if (instance.origin.scope) report.String("scopes", *instance.origin.scope);
      }
    }
    report.Emit();
  }
  const auto instruction_begin = impl_->model.currentProgram().instructions.size();
  impl_->model.AddCta(facts,
      [&](LaneRef lane, const RecordView& record, std::string_view text) {
        return address_values.Extract(lane, record, text);
      });
  impl_->address_queries += address_values.queries();
  impl_->address_resolved += address_values.resolved();
  if (impl_->address_unavailable.empty())
    impl_->address_unavailable = address_values.firstUnavailableReason();
  if (impl_->options.collect_synchronization_groups) {
    const FactSource::ConsumerScope consumer(facts, {Consumer::Cohorts, 0});
    BuildCohorts(cta, facts, impl_->model.currentProgram(), instruction_begin,
      impl_->cohorts, impl_->source_cohorts, impl_->poll);
  }
}
void SemanticProgramBuilder::TrackUnits(a::UnitQuotient* quotient) noexcept { impl_->model.TrackUnits(quotient); }
const v::Graph& SemanticProgramBuilder::graph() const noexcept { return impl_->graph; }
const std::vector<ValueIrDiagnostic>& SemanticProgramBuilder::valueDiagnostics() const noexcept { return impl_->values.diagnostics(); }
SemanticProgram SemanticProgramBuilder::Finish(std::function<void(std::string_view)> stage) {
  if (impl_->finished) throw std::logic_error("SemanticProgramBuilder is finished");
  impl_->finished = true;
  if (stage) stage("frontend.model.finalize");
  SemanticProgram result{impl_->model.Finish(), std::move(impl_->graph), impl_->values.diagnostics(), {}, {}, {}, {}, std::move(impl_->cohorts)};
  result.source_complete = result.lowering.diagnostics.empty();
  result.lowering.metadata["address_value_bridge"] = {
      {"source", "value_lowering"}, {"value_view_complete", result.valuesComplete()},
      {"queries", impl_->address_queries}, {"resolved", impl_->address_resolved},
      {"unavailable", impl_->address_queries - impl_->address_resolved},
      {"first_unavailable_reason", impl_->address_unavailable}};
  // Preserve value-lowering causes even if memory checking blocks correctness.
  auto& causes = result.lowering.metadata["address_value_bridge"]["diagnostics"];
  causes = nlohmann::json::array();
  for (const auto& diagnostic : result.value_diagnostics) {
    nlohmann::json details = nlohmann::json::object();
    const auto& origin = diagnostic.origin;
    if (origin.source_file) details["location"]["file"] = *origin.source_file;
    if (origin.source_line) details["location"]["line"] = *origin.source_line;
    if (origin.source_text) details["instruction"] = *origin.source_text;
    if (origin.cta) details["cta"] = *origin.cta;
    if (origin.thread) details["thread"] = *origin.thread;
    if (origin.instruction_index) details["instruction_index"] = *origin.instruction_index;
    if (origin.occurrence) details["occurrence"] = *origin.occurrence;
    causes.push_back({{"code", diagnostic.code}, {"stage", "value_lowering"},
                      {"message", diagnostic.message}, {"details", std::move(details)}});
  }
  result.synchronization_groups_materialized=impl_->options.collect_synchronization_groups;
  if(result.synchronization_groups_materialized) {
    if (stage) stage("frontend.semantic_program.sync_groups");
    a::ProfileScope profile("frontend.semantic_program.sync_groups");
    BuildSyncGroups(result, impl_->source_cohorts);
  }
  if (!result.valuesComplete()) {
    const auto& diagnostic = result.value_diagnostics.front();
    Diagnostic(result, "incomplete_value_semantics", diagnostic.code + ": " + diagnostic.message,
        Occurrence(diagnostic.origin));
    return result;
  }
  if (impl_->options.link_numerical_memory)
    LinkMemory(result, impl_->options.compact_memory_linking, stage);
  result.numerical_mma_layouts=std::move(result.lowering.mma_layouts);
  return result;
}

CheckedMemoryActions BuildTrustedReferenceMemoryActions(
    const SemanticProgram& program, const a::SourceArtifact& artifact) {
  CheckedMemoryActions result;
  result.contract.basis = va::MemoryOrderBasis::TrustedReference;
  const auto fail = [&](std::string code, std::string message) {
    result.diagnostics.push_back({std::move(code), std::move(message), {}, {}, {}});
  };
  const auto& source = artifact.program;
  if (!program.valuesComplete() || !program.mappingComplete() || !program.source_complete ||
      !program.lowering.diagnostics.empty() || !artifact.verification_limitations.empty() ||
      !artifact.source_violations.empty()) {
    fail("incomplete_reference_semantics", "Trusted reference still requires complete source and memory/value translation");
    return result;
  }
  if (!source.dependencies.empty() || !source.collectives.empty()) {
    fail("unsupported_reference_order", "Trusted-reference order currently supports synchronous memory without collectives");
    return result;
  }
  const auto id_hash=[](auto id) { return std::hash<std::uint32_t>{}(id.value()); };
  std::unordered_map<a::InstId,std::size_t,decltype(id_hash)> positions;
  const auto count = source.instructions.size();
  const auto first_id = count ? std::uint64_t(source.instructions.id(0).value()) : 0;
  bool consecutive = true;
  const auto build_positions=[&] {
    consecutive=false;
    for (std::size_t i=0;i<count;++i)
      if (!positions.emplace(source.instructions.id(i),i).second) {
        fail("unsupported_reference_order", "Trusted-reference order requires unique synchronous memory instructions");
        return false;
      }
    return true;
  };
  const auto positionOf = [&](a::InstId id) -> std::optional<std::size_t> {
    if (consecutive) {
      const auto value = std::uint64_t(id.value());
      if (value >= first_id && value - first_id < count)
        return static_cast<std::size_t>(value - first_id);
      if(!build_positions())return {};
    }
    const auto found = positions.find(id);
    return found == positions.end() ? std::nullopt : std::optional(found->second);
  };
  std::unordered_map<a::AddressId,const a::ConstantAddressBinding*,decltype(id_hash)> addresses;
  std::unordered_map<a::ActorId,const a::ActorInfo*,decltype(id_hash)> actors;
  for (const auto& address:artifact.addresses) addresses.emplace(address.address,&address);
  for (const auto& actor:source.actors) actors.emplace(actor.id,&actor);
  va::MemoryInstanceSet covered;
  using OrderKey = std::pair<std::size_t,std::uint64_t>;
  std::optional<OrderKey> previous_key;
  bool ordered = true;
  std::optional<a::ActorId> action_actor;
  std::size_t action_begin=0;
  std::pair<std::uint64_t,std::uintptr_t> action_cohort{};
  const auto finish_actions=[&] {
    if(action_actor)result.actions.endThread(action_begin,action_cohort.first,action_cohort.second);
  };
  for (std::size_t binding_index=0;binding_index<program.memory_bindings.size();++binding_index) {
    const auto binding=program.memory_bindings[binding_index];
    auto position=positionOf(binding.source.instruction);
    if (!position || binding.source.operand!=a::AccessOperand::kImmediate) {
      fail("invalid_reference_mapping", "Reference memory binding has no synchronous source instruction");
      return result;
    }
    auto inst=source.instructions[*position];
    if(inst.id!=binding.source.instruction) {
      if(!consecutive) {
        fail("invalid_reference_mapping", "Reference memory binding has an inconsistent source instruction ID");
        return result;
      }
      if(!build_positions())return result;
      position=positionOf(binding.source.instruction);
      if(!position) {
        fail("invalid_reference_mapping", "Reference memory binding has no synchronous source instruction");
        return result;
      }
      inst=source.instructions[*position];
    }
    const auto* load=std::get_if<a::ImmediateLoad>(&inst.op);
    const auto* store=std::get_if<a::ImmediateStore>(&inst.op);
    if(!load && !store) {
      fail("unsupported_reference_order", "Trusted-reference order requires unique synchronous memory instructions");
      return result;
    }
    if (binding.mode!=(load?a::AccessMode::kRead:a::AccessMode::kWrite) ||
        binding.address!=(load?load->address:store->address) ||
        binding.bytes!=(load?load->bytes:store->bytes)) {
      fail("invalid_reference_mapping", "Reference memory binding differs from its source instruction");
      return result;
    }
    const auto actor=actors.find(inst.actor);
    const auto address=addresses.find(binding.address);
    if (actor==actors.end() || !inst.ptx_origin ||
        SemanticOccurrence{actor->second->cta.value(),actor->second->warp.value(),actor->second->lane,
            inst.ptx_origin->instruction_index,inst.ptx_origin->occurrence}!=binding.occurrence) {
      fail("invalid_reference_mapping", "Reference memory binding has a different execution occurrence");
      return result;
    }
    if (binding.region && (address==addresses.end() || !address->second->expression.empty() ||
        !address->second->alias_scope.empty() ||
        binding.component_offset>UINT64_MAX-address->second->offset ||
        va::ConstantRegion{AllocationIdentity(*address->second),address->second->offset+binding.component_offset,
                           binding.region->bytes}!=*binding.region)) {
      fail("invalid_reference_mapping", "Reference memory binding has a different allocation or byte range");
      return result;
    }
    covered.insert({0,*position});
    const OrderKey key{*position,binding.component_offset};
    if (previous_key && key<*previous_key) ordered=false;
    previous_key=key;
    if (ordered) {
      const auto cohort=source.instructions.threadCohort(*position).value_or(
          std::pair<std::uint64_t,std::uintptr_t>{actor->second->cta.value(),std::uintptr_t(inst.actor.value())});
      if(action_actor!=inst.actor || cohort!=action_cohort) {
        finish_actions();action_actor=inst.actor;action_cohort=cohort;
        action_begin=result.actions.beginThread();
      }
      result.actions.push_back({binding.value,binding.region,binding.unsupported_reason,binding.value_byte_offset,{}});
    }
  }
  finish_actions();
  if (covered.size()!=count) {
    fail("incomplete_reference_mapping", "A reference memory instruction has no value mapping");
    return result;
  }
  const auto valid_edge=[&](const a::SourceEdge& edge) {
    const auto from=positionOf(edge.from),to=positionOf(edge.to);
    return from && to && *from<*to;
  };
  bool valid_edges=true;
  if(consecutive) {
    source.edges.forEachAffineColumn([&](const a::SourceEdge& first,const auto& delta,
        std::size_t,std::size_t,std::size_t count) {
      if(!valid_edges)return;
      auto last=first;
      a::SourceEdgeCodec::Binding binding{};
      a::SourceEdgeCodec::split(last,binding);
      const auto translated=a::SourceEdgeCodec::translate(binding,delta,count-1);
      if(!translated){valid_edges=false;return;}
      a::SourceEdgeCodec::bind(last,*translated);
      valid_edges=valid_edge(first) && valid_edge(last);
    });
  } else {
    for(const auto& edge:source.edges)if(!valid_edge(edge)) {
      valid_edges=false;break;
    }
  }
  if(!valid_edges) {
    fail("unsupported_reference_order", "Serialized reference instructions do not respect source dependencies");
    return result;
  }
  if (!ordered) {
    result.actions=va::MemoryActions{};
    std::vector<std::pair<OrderKey,std::size_t>> order;
    order.reserve(program.memory_bindings.size());
    for (std::size_t i=0;i<program.memory_bindings.size();++i) {
      const auto binding=program.memory_bindings[i];
      order.push_back({{*positionOf(binding.source.instruction),binding.component_offset},i});
    }
    std::stable_sort(order.begin(),order.end(),[](const auto& a,const auto& b){return a.first<b.first;});
    for (const auto& [key,index]:order) {
      const auto binding=program.memory_bindings[index];
      result.actions.push_back({binding.value,binding.region,binding.unsupported_reason,binding.value_byte_offset,{}});
    }
  }
  std::map<std::string,va::AllocationInfo> allocations;
  for (const auto& address:artifact.addresses) {
    if (!address.expression.empty() || address.allocation.empty() || address.address_space.empty()) continue;
    const auto identity=AllocationIdentity(address);
    const bool global=address.address_space=="global";
    allocations.try_emplace(identity,va::AllocationInfo{identity,global || address.address_space=="const",global});
  }
  for (auto& [_,allocation]:allocations)result.contract.allocations.push_back(std::move(allocation));
  return result;
}

}  // namespace sentryward::frontend::ptx
