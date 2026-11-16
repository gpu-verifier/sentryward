#pragma once
#include "lowering_impl.hpp"

namespace sentryward::frontend::ptx {
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
} // namespace sentryward::frontend::ptx
