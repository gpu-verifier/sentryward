#pragma once
#include "value_ir_impl.hpp"

namespace sentryward::frontend::ptx {
namespace {
void ValueTemplateExtractor::TcgenMma() {
  const auto& i = *instruction_;
  const bool sparse = Has(i, "sp"), scaled = Has(i, "block_scale");
  const unsigned descriptor_index = sparse ? 4 : 3;
  if (i.operands.size() <= descriptor_index + 1) throw std::invalid_argument("MMA operands missing");
  for (const auto& mod : i.modifiers)
    if (!In(mod, {"mma", "sp", "ws", "cta_group::1", "cta_group::2", "kind::f16", "kind::tf32", "kind::i8",
        "kind::f8f6f4", "kind::mxf8f6f4", "kind::mxf4", "kind::mxf4nvf4", "block_scale", "block16", "block32",
        "scale_vec::1X", "scale_vec::2X", "scale_vec::4X"}))
      throw std::invalid_argument("Unsupported MMA value modifier: " + mod);
  const bool ws = Has(i, "ws");
  if (ws && (sparse || scaled || Has(i, "cta_group::2")))
    throw std::invalid_argument("Unsupported .ws MMA value variant");
  const auto text = [&](unsigned n) { return Trim(module_.view(i.operands.at(n))); };
  auto d = Address(text(0)), a = text(1).starts_with('[') ? Address(text(1)) : Operand(text(1), U64());
  auto b = Operand(text(2), U64()), descriptor = Operand(text(descriptor_index), U32());
  if (a) a = Bits(*a);
  if (b) b = Bits(*b);
  if (descriptor) descriptor = Bits(*descriptor);
  if (!d || !a || !b || !descriptor) return;
  unsigned next = descriptor_index + 1;
  std::optional<Expression> sa, sb, meta;
  if (scaled) {
    if (i.operands.size() <= next + 2) throw std::invalid_argument("MMA scale operands missing");
    sa = Address(text(next++)); sb = Address(text(next++)); if (!sa || !sb) return;
  }
  if (sparse) { meta = Address(text(3)); if (!meta) return; }
  std::vector<v::Value> masks;
  bool masks_are_zero = true;
  if (text(next).starts_with('{')) {
    if (ws) throw std::invalid_argument(".ws MMA has no output lane mask operand");
    const auto mask_text = text(next++);
    if (mask_text.back() != '}') throw std::invalid_argument("MMA output mask is malformed");
    for (const auto& item : Split(std::string_view(mask_text).substr(1, mask_text.size()-2), ',')) {
      auto value = Operand(item, U32()); if (!value) return; masks.push_back(value->value);
      const auto known = Known(item);
      masks_are_zero = masks_are_zero && known && *known == 0;
    }
    if (masks.size() != (Has(i,"cta_group::2") ? 8u : 4u)) throw std::invalid_argument("MMA output mask width mismatch");
  }
  if (next >= i.operands.size()) throw std::invalid_argument("MMA enable-input-d missing");
  const auto enabled_text = text(next++);
  const auto enabled_known = Known(enabled_text);
  auto enabled = Operand(enabled_text, Pred()); if (!enabled) return;
  if (enabled->type.value.element().kind() != v::ElementKind::Predicate) throw std::invalid_argument("MMA enable-input-d must be a predicate");
  std::optional<Expression> exponent;
  if (ws && next < i.operands.size()) {
    // The .ws trailing operand is the zero-column mask descriptor, not scale-input-d.
    // Only a known zero (no masked or shifted B columns) keeps D = A*B(+D).
    const auto zero_column_mask = Known(text(next++));
    if (!zero_column_mask || *zero_column_mask != 0)
      throw std::invalid_argument("Unsupported .ws zero-column mask");
  } else if (next < i.operands.size()) { exponent = Operand(text(next++), U32()); if (!exponent) return; }
  if (next != i.operands.size()) throw std::invalid_argument("Unexpected MMA operands");
  const auto issuer = ClusterRank();
  std::string mode = "tcgen05";
  for (const auto& mod : i.modifiers) mode += '.' + mod;
  std::optional<MmaDescriptor> fixed_descriptor;
  auto known_descriptor=Known(text(descriptor_index));
  // Sparse: the descriptor without its selector (0-1) and negate (13-14) bits, as the
  // lowering's MmaValueLayout keeps it, when those bits are not all known.
  std::optional<std::uint64_t> sparse_layout;
  if (!known_descriptor && ((scaled && tmem_provenance) || sparse) && resources) {
    resources->SetLane(lane_);
    const auto resolver=[&](const RecordView& record,std::string_view token,unsigned)
        ->std::optional<Resources::KnownBits> {
      if(!tmem_provenance)return {};
      auto value=tmem_provenance->Resolve(facts_.Warp(lane_),record,token,facts_.LaneId(lane_));
      const auto* origin=facts_.Origin(record,token);
      const auto* current=facts_.Origin(*record_,token);
      if(std::holds_alternative<std::monostate>(value) && origin && current && *origin==*current)
        value=tmem_provenance->Resolve(facts_.Warp(lane_),*record_,token,facts_.LaneId(lane_));
      if(const auto* pointer=std::get_if<TmemPointer>(&value))
        return Resources::KnownBits{pointer->encoded_offset&0xc0000000U,0xc0000000U};
      return {};
    };
    const auto [value,mask]=resources->FixedBits(*record_,text(descriptor_index),32,resolver);
    const std::uint64_t layout=UINT32_MAX & ~(3ULL | (1ULL<<13) | (1ULL<<14));
    if(mask==UINT32_MAX)known_descriptor=Integer(value);
    else if(sparse && (mask & layout)==layout)sparse_layout=value & layout;
  }
  // A sparse descriptor's selector bits (0-1) only move the metadata column; the shapes
  // below do not depend on them.
  if (const auto known = known_descriptor; known && *known >= 0 && *known <= UINT32_MAX) {
    descriptor = Constant(U32(), known->convert_to<std::string>());
    for (const auto& modifier : i.modifiers)
      if (modifier.starts_with("kind::"))
        try { fixed_descriptor = scaled ? DecodeBlockScaledDescriptor(known->convert_to<std::uint64_t>(),modifier.substr(6)) :
            DecodeMmaDescriptor(modifier.substr(6),known->convert_to<std::uint64_t>()); }
        catch (const LayoutError&) {}
    if (fixed_descriptor && fixed_descriptor->sparse != sparse) fixed_descriptor.reset();
  } else if (sparse_layout) {
    // The operand shapes only (the descriptor stays a value): they do not depend on those bits.
    for (const auto& modifier : i.modifiers)
      if (modifier.starts_with("kind::"))
        try { fixed_descriptor = scaled ? DecodeBlockScaledDescriptor(*sparse_layout,modifier.substr(6)) :
            DecodeMmaDescriptor(modifier.substr(6),*sparse_layout); }
        catch (const LayoutError&) {}
    if (fixed_descriptor && !fixed_descriptor->sparse) fixed_descriptor.reset();
  }
  const auto type = [&](std::string role, bool address = false) {
    const auto view = mode + ':' + role;
    if (fixed_descriptor && (role == "A" || role == "B" || role == "D")) {
      const auto& d = *fixed_descriptor;
      auto bits = role == "A" ? d.a_element_bits : role == "B" ? d.b_element_bits : d.d_element_bits;
      const bool packed = scaled && bits==4 && role!="D";
      if (packed) bits=8;
      // A sparse A operand is stored packed: M x K/2 elements (ISA "Sparse Matrices").
      std::vector<std::uint64_t> shape{role == "A" ? d.m : role == "B" ? d.n : d.m,
          role == "D" ? d.n : (packed ? d.k/2 : d.k) / (sparse && role == "A" ? 2 : 1)};
      // A Tensor Memory A operand holds one copy per data-path lane group.
      if (role == "A" && text(1).starts_with('['))
        if (const auto path = DecodeMmaDataPath(d.m, Has(i,"cta_group::2") ? 2 : 1, ws, d.sparse);
            path && path->slices > 1)
          shape.insert(shape.begin(), path->slices);
      return v::ValueType::matrix(address ? v::ElementType::unsignedInteger(64) : v::ElementType::bit(bits),
          std::move(shape));
    }
    if (fixed_descriptor && scaled && (role=="scale_A" || role=="scale_B")) {
      const auto& d=*fixed_descriptor;
      const unsigned group=Has(i,"cta_group::2")?2:1;
      const unsigned vector=d.kind=="mxf4"?2:d.kind=="mxf4nvf4"?4:1;
      return v::ValueType::matrix(address ? v::ElementType::unsignedInteger(64) : v::ElementType::bit(8),
          {group,4,role=="scale_A"?d.m/group:d.n,vector});
    }
    // Sparse metadata (M = 128 x cta_group): the bytes of the Tensor Memory footprint
    // DecodeSparseMetadataFootprint reads, one row per data-path lane in lane order (lanes of
    // CTA part p are rows 128p..128p+127), 4 bytes per 32-bit column: one column for
    // kind::f16, two for f8f6f4 and the block-scaled kinds. Every 4-bit field of the footprint
    // is metadata of this MMA; their assignment to A rows and K chunks (ISA Fig. 288/292) is
    // not decoded here.
    if (fixed_descriptor && sparse && role=="metadata") {
      const auto& d=*fixed_descriptor;
      const unsigned group=Has(i,"cta_group::2")?2:1;
      const bool known_kind=d.kind=="f16" || d.kind=="f8f6f4" || d.kind=="mxf8f6f4" ||
          d.kind=="mxf4" || d.kind=="mxf4nvf4";
      if (known_kind && d.m==128*group)
        return v::ValueType::matrix(address ? v::ElementType::unsignedInteger(64) : v::ElementType::bit(8),
            {d.m, d.kind=="f16" ? 4u : 8u});
    }
    return v::ValueType::parameterizedMatrix(address ? v::ElementType::unsignedInteger(64) :
        v::ElementType::fromDescriptor(descriptor->value, view), descriptor->value, view);
  };
  const auto address = [&](Expression base, v::MatrixRole role, const char* view, const char* space) {
    v::NativeMatrixAttributes attrs{i.modifiers}; attrs.role = role;
    base.value = graph_.asPointer(base.value, {"", space, cta_, {}});
    auto result = Native(v::NativeMatrixOpcode::TcgenMmaAddress, type(view, true), {base.value, descriptor->value, issuer}, attrs);
    return graph_.asPointer(result, {"", space, Has(i,"cta_group::2") ? std::optional<std::uint64_t>{} : cta_, {}});
  };
  auto read = [&](v::Value addr, const char* role) {
    return Add(std::make_unique<v::LoadOp>(type(role)), {addr}, origin_).result();
  };
  auto av = read(address(*a, v::MatrixRole::A, "A", text(1).starts_with('[') ? "tmem" : "shared-matrix-descriptor"), "A");
  auto bv = read(address(*b, v::MatrixRole::B, "B", "shared-matrix-descriptor"), "B");
  auto dst = address(*d, v::MatrixRole::D, "D", "tmem");
  v::TcgenMmaAttributes mma{i.modifiers, !enabled_known || *enabled_known != 0, scaled, sparse, exponent.has_value()};
  std::vector<v::Value> args{av, bv, descriptor->value};
  if (mma.accumulate) {
    v::MemoryAttributes memory; memory.guarded = !enabled_known;
    std::vector<v::Value> load_args{dst}; if (memory.guarded) load_args.push_back(enabled->value);
    args.push_back(Add(std::make_unique<v::LoadOp>(type("D"), memory), std::move(load_args), origin_).result());
  }
  if (scaled) {
    args.push_back(read(address(*sa,v::MatrixRole::ScaleA,"scale_A","tmem"),"scale_A"));
    args.push_back(read(address(*sb,v::MatrixRole::ScaleB,"scale_B","tmem"),"scale_B"));
  }
  if (sparse) args.push_back(read(address(*meta,v::MatrixRole::SparseMetadata,"metadata","tmem"),"metadata"));
  if (exponent) args.push_back(exponent->value);
  auto result = Add(std::make_unique<v::TcgenMmaOp>(type("D"), mma), std::move(args), origin_).result();
  std::vector<v::Value> store_args{dst, result}; v::MemoryAttributes store;
  if (!masks.empty() && !masks_are_zero) {
    std::vector<v::Value> mask_args{descriptor->value, d->value}; mask_args.insert(mask_args.end(), masks.begin(), masks.end());
    v::NativeMatrixAttributes attrs{i.modifiers}; attrs.mask_words = masks.size();
    auto mask_type = v::ValueType::parameterizedMatrix(v::ElementType::predicate(), descriptor->value, mode + ":D");
    store_args.push_back(Native(v::NativeMatrixOpcode::TcgenMmaOutputMask, mask_type, mask_args, attrs));
    store.guarded = true;
  }
  Add(std::make_unique<v::StoreOp>(store), std::move(store_args), origin_);
}

void ValueTemplateExtractor::Tcgen() {
  const auto& i = *instruction_;
  if (Has(i, "mma")) { TcgenMma(); return; }
  const auto text = [&](unsigned n) { return Trim(module_.view(i.operands.at(n))); };
  if (Has(i, "ld") || Has(i, "st")) {
    const bool load = Has(i, "ld");
    unsigned registers = 0, repeats = 0;
    for (const auto& mod : i.modifiers) {
      if (mod == "32x32b" || mod == "16x64b" || mod == "16x32bx2") registers = 1;
      if (mod == "16x128b") registers = 2;
      if (mod == "16x256b") registers = 4;
      if (std::regex_match(mod, std::regex("x(1|2|4|8|16|32|64|128)"))) repeats = std::stoul(mod.substr(1));
      if (!(In(mod,{"ld","st","sync","aligned","b32","32x32b","16x64b","16x128b","16x256b","16x32bx2","pack::16b","unpack::16b"}) ||
          std::regex_match(mod,std::regex("x(1|2|4|8|16|32|64|128)"))))
        throw std::invalid_argument("Unsupported TMEM register modifier: " + mod);
    }
    if (!registers || !repeats || !Has(i,"sync") || !Has(i,"aligned") || !Has(i,"b32"))
      throw std::invalid_argument("Incomplete TMEM register instruction");
    registers *= repeats;
    if (registers > 128 || (load && Has(i,"unpack::16b")) || (!load && Has(i,"pack::16b")))
      throw std::invalid_argument("Invalid TMEM register packing/width");
    if (i.operands.size() != (Has(i,"16x32bx2") ? 3u : 2u)) throw std::invalid_argument("TMEM register operand count mismatch");
    auto vector = text(load ? 0 : Has(i,"16x32bx2") ? 2 : 1);
    if (!vector.starts_with('{') || !vector.ends_with('}')) throw std::invalid_argument("TMEM register vector is missing");
    auto components = Split(std::string_view(vector).substr(1,vector.size()-2), ',');
    if (components.size() != registers) throw std::invalid_argument("TMEM register vector width mismatch");
    auto base = Address(text(load ? 1 : 0)); if (!base) return;
    base->value = graph_.asPointer(base->value,{"","tmem",cta_,{}});
    auto offset = Has(i,"16x32bx2") ? Operand(text(load ? 2 : 1),U32()) : std::optional<Expression>(Constant(U32(),"0"));
    if (!offset) return;
    auto thread = LinearThread();
    const auto data_type = *PtxType("b32");
    const bool packed=Has(i,"pack::16b") || Has(i,"unpack::16b");
    const auto element=packed ? *PtxType("b16") : data_type;
    // Inside a merged arm: the cells (address operands, shape, the address register's value) and,
    // for a store, their registers from the arm's tcgen05.ld of them since its last store.
    std::optional<std::pair<std::string, v::Value>> cells;
    std::vector<Expression> prior;
    if (merged_ && !packed && !Has(i,"16x32bx2")) {
      const auto address = text(load ? 1 : 0);
      std::string spelling = address;
      for (const auto& modifier : i.modifiers) if (modifier != "ld" && modifier != "st") spelling += '.' + modifier;
      const auto split = SplitAddress(std::string_view(address).substr(1, address.size() - 2));
      if (const auto found = split ? registers_.find(Resolve(split->first, i.scope)) : registers_.end(); found != registers_.end())
        cells = {spelling, found->second.value};
      if (cells && !load)
        for (const auto& loaded : merged_->loads)
          if (loaded.address == cells->first && loaded.base == cells->second) prior = loaded.registers;
    }
    if (merged_ && !load && prior.size() != registers) {
      Fail("two_arm_merge_values", "A tcgen05.st in a merged arm writes TMEM cells that the arm did not load since its last "
           "store; their values without the arm are not known");
      return;
    }
    std::vector<Expression> loaded_registers;
    std::optional<v::Value> loaded_components;
    if(load) {
      // One native instruction produces an indexed register payload, not a
      // collection of unrelated Load roots. Spatial members still have their
      // own instances; each physical fragment retains its source event binding.
      const auto count=registers*(packed?2u:1u);
      v::NativeMatrixAttributes attrs{i.modifiers};
      const auto address_type=v::ValueType::matrix(v::ElementType::unsignedInteger(64),{count});
      auto addr=Native(v::NativeMatrixOpcode::TcgenRegisterAddress,address_type,
          {base->value,thread.value,offset->value},attrs);
      addr=graph_.asPointer(addr,{"","tmem",cta_,{}});
      v::MemoryAttributes memory;
      memory.native_modifiers=i.modifiers;
      loaded_components=Add(std::make_unique<v::LoadOp>(
          v::ValueType::matrix(element.value.element(),{count}),memory),{addr},origin_).result();
    }
    for (unsigned n = 0; n < registers; ++n) {
      std::optional<Expression> source, combined;
      if (!load) {
        source=Operand(components[n],data_type); if (source) source=Reinterpret(*source,data_type);
        // The arm does not run: the cells keep their values.
        if (source && merged_) source=Choose(merged_->runs,*source,prior[n]);
        if (!source) continue;
      }
      for (unsigned half=0; half < (packed ? 2u : 1u); ++half) {
        v::NativeMatrixAttributes attrs{i.modifiers}; attrs.register_index = n;
        if (packed) attrs.packed_half=half;
        if (load) {
          const auto index=Constant(U32(),std::to_string(n*(packed?2u:1u)+half));
          Expression value{graph_.component(*loaded_components,{index.value}),element};
          if (packed) {
            auto integer=Reinterpret(value,*PtxType("u16")); if (!integer) return;
            value={Add(std::make_unique<v::ConvertOp>(U32().value),{integer->value},origin_).result(),U32()};
            if (half) {
              auto shift=Constant(U32(),"16");
              auto upper=Add(std::make_unique<v::BinaryOp>(v::BinaryOpcode::ShiftLeft,U32().value),{value.value,shift.value},origin_).result();
              value={Add(std::make_unique<v::BinaryOp>(v::BinaryOpcode::BitOr,U32().value),{combined->value,upper},origin_).result(),U32()};
            }
          }
          combined=value;
        } else {
          auto addr = Native(v::NativeMatrixOpcode::TcgenRegisterAddress,U64().value,{base->value,thread.value,offset->value},attrs);
          addr = graph_.asPointer(addr,{"","tmem",cta_,{}});
          auto value=packed ? Part(*source,element,16*half) : source;
          if (!value) return;
          Add(std::make_unique<v::StoreOp>(),{addr,value->value},origin_);
        }
      }
      if (load && combined) {
        auto value=Reinterpret(*combined,data_type);
        if (value) { Bind(components[n],*value); if (cells) loaded_registers.push_back(*value); }
      }
    }
    if (merged_ && !load) merged_->loads.clear();
    if (merged_ && load && cells && loaded_registers.size() == registers)
      merged_->loads.push_back({cells->first, cells->second, std::move(loaded_registers)});
    return;
  }
  if (Has(i,"cp")) {
    for (const auto& mod : i.modifiers)
      if (!std::regex_match(mod,std::regex("(128|64|32|4)x(128|256)b")) &&
          !In(mod,{"cp","cta_group::1","cta_group::2","warpx4","warpx2::02_13","warpx2::01_23"}))
        throw std::invalid_argument("Unsupported tcgen05.cp value modifier: " + mod);
    const auto shape=DecodeCopyShape(i.modifiers);
    if (!shape || i.operands.size()!=2)
      throw std::invalid_argument("Invalid tcgen05.cp shape/operands");
    auto dst=Address(text(0)), src=Operand(text(1),U64());
    if (src) src=Reinterpret(*src,U64());
    if (!dst || !src) return;
    src->value=graph_.asPointer(src->value,{"","shared-matrix-descriptor",cta_,{}});
    dst->value=graph_.asPointer(dst->value,{"","tmem",cta_,{}});
    const auto issuer=ClusterRank();
    std::string view="tcgen05.copy"; for (const auto& mod : i.modifiers) view+='.'+mod;
    const auto type=[&](const char* role, bool address=false) {
      if (!address && shape->LaneMapped())
        return v::ValueType::matrix(v::ElementType::bit(8),{shape->TensorBytes()*(Has(i,"cta_group::2")?2:1)});
      return v::ValueType::parameterizedMatrix(address ? v::ElementType::unsignedInteger(64) : v::ElementType::bit(8),
          src->value,view+':'+role);
    };
    v::NativeMatrixAttributes attrs{i.modifiers};
    const auto owner=Has(i,"cta_group::2") ? std::optional<std::uint64_t>{} : cta_;
    auto source=Native(v::NativeMatrixOpcode::TcgenCopyAddress,type("source",true),{src->value,issuer},attrs);
    source=graph_.asPointer(source,{"","shared-matrix-descriptor",owner,{}});
    auto value=Add(std::make_unique<v::LoadOp>(type("source")),{source},origin_).result();
    attrs.role=v::MatrixRole::Destination;
    value=Native(v::NativeMatrixOpcode::TcgenCopyValue,type("destination"),{value},attrs);
    auto destination=Native(v::NativeMatrixOpcode::TcgenCopyAddress,type("destination",true),{dst->value,issuer},attrs);
    destination=graph_.asPointer(destination,{"","tmem-copy-view",owner,{}});
    Add(std::make_unique<v::StoreOp>(),{destination,value},origin_);
    return;
  }
  if (Has(i,"alloc")) {
    if (i.operands.size()!=2) throw std::invalid_argument("Invalid TMEM allocation operands");
    if (facts_.LaneId(lane_) != 0) return;
    auto dst=Address(text(0)), columns=Operand(text(1),U32()); if (!dst || !columns) return;
    dst->value=graph_.asPointer(dst->value,{"","shared",cta_,{}});
    auto pointer=Native(v::NativeMatrixOpcode::TmemAllocation,U32().value,{columns->value},v::NativeMatrixAttributes{i.modifiers});
    pointer=graph_.asPointer(pointer,{"","tmem",cta_,{}});
    Add(std::make_unique<v::StoreOp>(),{dst->value,pointer},origin_);
    return;
  }
  if (In(i.semantics.operation,{"commit","wait_load","wait_store","fence","dealloc","relinquish_alloc_permit"})) return;
  Fail("unsupported_tcgen_value","No numerical model for this tcgen05 variant"); Invalidate();
}

} // namespace
} // namespace sentryward::frontend::ptx
