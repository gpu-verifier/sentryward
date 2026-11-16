#pragma once
#include "value_ir_impl.hpp"

namespace sentryward::frontend::ptx {
namespace {
v::Value ValueTemplateExtractor::Native(v::NativeMatrixOpcode opcode, v::ValueType type,
                              std::vector<v::Value> operands, v::NativeMatrixAttributes attributes) {
  return Add(std::make_unique<v::NativeMatrixOp>(opcode, std::move(type), std::move(attributes)),
             std::move(operands), origin_).result();
}
v::Value ValueTemplateExtractor::ClusterRank() {
  return Add(std::make_unique<v::CoordinateOp>(v::CoordinateKind::ClusterRank), {}, origin_).result();
}
v::Value ValueTemplateExtractor::SlotAddressValue(const algorithm::ConstantAddressBinding& binding) {
  auto offset = Constant(U64(), std::to_string(binding.offset));
  return graph_.asPointer(offset.value, {binding.allocation, binding.address_space, {}, {}});
}

void ValueTemplateExtractor::Tma() {
  if (ResolvedTma()) return;
  const auto& i = *instruction_;
  // Only constant byte copies model a gather; the descriptor-shaped native load
  // below would describe one map box, not the four gathered rows.
  if (Has(i, "tile::gather4")) {
    Fail("tma_gather4_dynamic_rows", "TMA tile::gather4 coordinates are not statically known "
         "(data-dependent rows); the gathered values are not modeled");
    Invalidate(); return;
  }
  const bool load = Has(i, "mbarrier::complete_tx::bytes");
  const bool multicast = Has(i, "multicast::cluster");
  unsigned rank = 0;
  for (unsigned n = 1; n <= 5; ++n) if (Has(i, std::to_string(n) + "d")) rank = n;
  for (const auto& mod : i.modifiers)
    if (!(In(mod, {"async", "bulk", "tensor", "global", "shared::cta", "shared::cluster", "tile", "tile::gather4",
          "mbarrier::complete_tx::bytes", "bulk_group", "multicast::cluster", "L2::cache_hint", "cta_group::1", "cta_group::2"}) ||
          mod == std::to_string(rank) + "d"))
      throw std::invalid_argument("Unsupported TMA value modifier: " + mod);
  if (!rank || !Has(i, "global") || (!load && !Has(i, "bulk_group")) || (!load && multicast))
    throw std::invalid_argument("Unsupported TMA direction/completion form");
  const auto count = (load ? 3u : 2u) + multicast + Has(i, "L2::cache_hint");
  if (i.operands.size() != count) throw std::invalid_argument("TMA operand count mismatch");
  const auto text = [&](unsigned n) { return Trim(module_.view(i.operands.at(n))); };
  const auto tensor = text(load ? 1 : 0);
  static const std::regex form(R"(^\[\s*([^,]+),\s*\{([^}]+)\}\s*\]$)");
  std::smatch match;
  if (!std::regex_match(tensor, match, form)) throw std::invalid_argument("Unsupported tensor-map operand form");
  const auto coordinates = Split(match[2].str(), ',');
  if (coordinates.size() != (Has(i,"tile::gather4") ? 5u : rank) || (Has(i,"tile::gather4") && (!load || rank != 2)))
    throw std::invalid_argument("TMA coordinate count mismatch");
  auto descriptor_pointer = Operand(Trim(match[1].str()), U64());
  auto shared = Address(text(load ? 0 : 1));
  if (!descriptor_pointer || !shared) return;
  shared->value = graph_.asPointer(shared->value, {"", Has(i,"shared::cluster") ? "cluster-shared-address" : "shared",
      Has(i,"shared::cluster") ? std::optional<std::uint64_t>{} : cta_, {}});
  std::vector<v::Value> coords;
  for (const auto& coordinate : coordinates) {
    auto value = Operand(coordinate, *PtxType("s32"));
    if (value) value = Reinterpret(*value, *PtxType("s32"));
    if (!value) return;
    coords.push_back(value->value);
  }
  v::MemoryAttributes descriptor_memory;
  descriptor_memory.interpretation = v::MemoryAttributes::Interpretation::TensorMap;
  auto descriptor = Add(std::make_unique<v::LoadOp>(v::ValueType::matrix(v::ElementType::bit(64), {16}), descriptor_memory),
      {descriptor_pointer->value}, origin_).result();
  std::string view = "tma.tensor";
  for (const auto& mod : i.modifiers) view += '.' + mod;
  const auto matrix = v::ValueType::parameterizedMatrix(v::ElementType::fromDescriptor(descriptor, view), descriptor, view);
  const auto address_type = v::ValueType::parameterizedMatrix(v::ElementType::unsignedInteger(64), descriptor, view);
  v::NativeMatrixAttributes attrs{i.modifiers}; attrs.coordinate_count = coordinates.size();
  std::vector<v::Value> global_args{descriptor}; global_args.insert(global_args.end(), coords.begin(), coords.end());
  auto global = Native(v::NativeMatrixOpcode::TmaGlobalAddress, address_type, global_args, attrs);
  global = graph_.asPointer(global, {"", "global", {}, {}});
  const auto issuer = ClusterRank();
  v::MemoryAttributes tensor_memory;
  tensor_memory.interpretation = v::MemoryAttributes::Interpretation::TmaTensor;
  tensor_memory.native_modifiers = i.modifiers;
  if (!load) {
    attrs.role = v::MatrixRole::Source;
    auto src = Native(v::NativeMatrixOpcode::TmaSharedAddress, address_type,
        {shared->value, descriptor, issuer, issuer}, attrs);
    src = graph_.asPointer(src, {"", "shared", cta_, {}});
    const auto data = Add(std::make_unique<v::LoadOp>(matrix), {src}, origin_).result();
    Add(std::make_unique<v::StoreOp>(tensor_memory), {global, data}, origin_);
    return;
  }
  const auto data = Add(std::make_unique<v::LoadOp>(matrix, tensor_memory), {global}, origin_).result();
  auto shape = frame_.cluster_dim();
  const auto cluster_size = shape[0] * shape[1] * shape[2];
  const auto current = frame_.self().value;
  std::optional<Expression> mask;
  if (multicast) {
    mask = Operand(text(3), U32());
    if (mask && mask->type.bits == 16) {
      mask = Reinterpret(*mask,*PtxType("u16"));
      if (mask) mask=Expression{Add(std::make_unique<v::ConvertOp>(U32().value),{mask->value},origin_).result(),U32()};
    }
    if (mask) mask = Reinterpret(*mask, U32());
    if (!mask) return;
  }
  if (multicast && cluster_size > 16) throw std::invalid_argument("TMA multicast mask supports at most 16 CTA ranks");
  const auto targets = multicast ? cluster_size : 1u;
  for (unsigned n = 0; n < targets; ++n) {
    const auto target_rank = multicast ? n : current;
    auto target = Constant(U32(), std::to_string(target_rank));
    attrs.role = v::MatrixRole::Destination;
    auto dst = Native(v::NativeMatrixOpcode::TmaSharedAddress, address_type,
        {shared->value, descriptor, issuer, target.value}, attrs);
    // Non-multicast ownership follows dstMem; cta_group only selects completion targets.
    dst = graph_.asPointer(dst, {"", "cluster-shared-view", {}, {}});
    v::MemoryAttributes store;
    std::vector<v::Value> args{dst, data};
    if (mask) {
      auto bit = Constant(U32(), std::to_string(std::uint64_t(1) << target_rank));
      auto enabled = Add(std::make_unique<v::BinaryOp>(v::BinaryOpcode::BitAnd, U32().value), {mask->value, bit.value}, origin_).result();
      auto zero = Constant(U32(), "0");
      auto predicate = Add(std::make_unique<v::CompareOp>(v::ComparePredicate::NotEqual, Pred().value), {enabled, zero.value}, origin_).result();
      args.push_back(predicate); store.guarded = true;
    }
    Add(std::make_unique<v::StoreOp>(store), std::move(args), origin_);
  }
}

bool ValueTemplateExtractor::ResolvedTma() {
  if (!resources) return false;
  resources->SetLane(lane_);
  TmaAccesses transfer;
  try {
    const auto& contract = facts_.launch().InstructionContract(instruction_->line, instruction_->id,
        facts_.Warp(lane_), record_->occurrence());
    transfer = resources->Tma(*record_, TmaRequest(contract, Has(*instruction_, "bulk_group")));
  } catch (const SemanticError&) { return false; }
  catch (const LayoutError& error) {
    // An unsupported gather layout stays a value diagnostic; the event view
    // reports the same layout error for the instruction.
    if (!Has(*instruction_, "tile::gather4")) throw;
    Fail(error.code, error.what()); Invalidate(); return true;
  }
  std::optional<algorithm::AddressId> region;
  for (const auto& pair : transfer.pairs)
    for (const auto* endpoint : {&pair.source, &pair.destination})
      if (*endpoint) {
        const auto& [id, bytes] = **endpoint;
        const auto& binding = resources->Binding(id);
        if (!bytes || !*bytes || !binding.alias_scope.empty()) return false;
        if (!region && !binding.expression.empty()) region = id;
      }
  // Lowering gives a tile at a non-constant global region an abstract address, the region
  // expression of slot_resolver.cpp (TmaBody). Each abstract fragment is then read or written
  // at its allocation-relative address, its region's first byte (descriptor base + sum of
  // coordinate x stride) plus its place in the region, when the fragment is contiguous in the
  // tensor and its region is proven inside the tensor: every byte of it is then the tensor's
  // byte at that address. A tile::gather4 row not proven inside (index -1 rows) is the
  // guarded Load of the row clamped into the tensor, guarded by 0 <= row < rows: PTX ISA,
  // Tensor "Out of Boundary Access", zero fill mode: "Elements in the Bounding Box which fall
  // outside of the tensor boundary are set to 0", and a gather4 row is its own one-row
  // bounding box ("four request coordinates will form four Bounding Boxes"); a disabled Load
  // yields zero. Otherwise its values are not modeled: name the cause from the expression
  // itself (no further fact queries).
  std::vector<std::optional<TmaRegionOrigin>> origins;
  // The region binding's expression, parsed once (it embeds the whole descriptor).
  const auto expression = region ? nlohmann::json::parse(resources->Binding(*region).expression, nullptr, false)
                                 : nlohmann::json();
  if (region && expression.is_object() && expression.value("operator", std::string()) == "tma_region")
    if (auto found = TmaRegionOrigins(transfer, expression.value("descriptor", nlohmann::json())))
      origins = std::move(*found);
  if (region && origins.empty()) {
    const auto tensor = Trim(module_.view(instruction_->operands.at(Has(*instruction_, "bulk_group") ? 0 : 1)));
    const auto open = tensor.find('{'), close = tensor.find('}');
    const auto tokens = open != std::string::npos && close != std::string::npos && open < close
        ? Split(std::string_view(tensor).substr(open + 1, close - open - 1), ',') : std::vector<std::string>{};
    const auto coordinates = expression.is_object() ? expression.value("coordinates", nlohmann::json()) : nlohmann::json();
    // A gather4 region is one row {column, row}: the row of the first abstract pair.
    const auto row = std::find_if(transfer.pairs.begin(), transfer.pairs.end(), [](const auto& p) { return p.place; });
    const bool gather = Has(*instruction_, "tile::gather4") && row != transfer.pairs.end() && tokens.size() == 5;
    std::string dynamic;
    if (coordinates.is_array() && coordinates.size() == (gather ? 2 : tokens.size()))
      for (std::size_t k = 0; k < coordinates.size(); ++k)
        if (!coordinates[k].is_number_integer())
          dynamic += (dynamic.empty() ? "" : ", ") + tokens[gather && k == 1 ? 1 + row->place->region : k];
    if (!dynamic.empty())
      Fail("tma_data_dependent_coordinate", "TMA tile coordinate " + dynamic + " is data-dependent (not a "
           "constant of the execution) and its range is not proven to keep the tile inside the tensor "
           "(0 <= c, c + box <= global), or a fragment is not contiguous in the tensor; the values of its "
           "tile region are not modeled");
    else
      Fail("tma_nonconstant_tensor_map", "TMA tile region is not a constant of the execution (tensor-map global "
           "address, shape or stride); the values of its tile region are not modeled");
    Invalidate(); return true;
  }
  const auto address = [&](const Fragment& fragment) {
    return SlotAddressValue(resources->Binding(fragment.first));
  };
  // The read of a runtime-written tensor map (the descriptor itself comes from its resolution).
  if (transfer.descriptor)
    Add(std::make_unique<v::LoadOp>(v::ValueType::matrix(v::ElementType::bit(8), {*transfer.descriptor->second})),
        {address(*transfer.descriptor)}, origin_);
  // An abstract fragment's address and attributes (TmaRegionOrigins): its region's offset
  // plus its place, a pointer naming its binding's allocation.
  const auto placed = [&](const Fragment& fragment, const TmaRegionPlace& place, v::MemoryAttributes& attributes) {
    const auto& origin = *origins.at(place.region);
    const auto s64 = *PtxType("s64");
    const auto offset = Add(std::make_unique<v::BinaryOp>(v::BinaryOpcode::Add, s64.value),
        {origin.offset.value, Constant(s64, std::to_string(*place.offset)).value}, origin_).result();
    const auto unsigned_offset = Add(std::make_unique<v::ConvertOp>(U64().value), {offset}, origin_).result();
    const auto& binding = resources->Binding(fragment.first);
    attributes.region_offset = *place.offset;
    attributes.region_exact = origin.exact;
    return graph_.asPointer(unsigned_offset, {binding.allocation, binding.address_space, {}, {}});
  };
  std::map<std::tuple<algorithm::AddressId, std::optional<std::uint64_t>, std::uint64_t>, v::Value> reads;
  const auto abstract_endpoint = [&](const TmaPair& pair, const std::optional<Fragment>& endpoint) {
    return pair.place && !origins.empty() && endpoint && !resources->Binding(endpoint->first).expression.empty();
  };
  // Source rows: a load's consecutive pairs whose source bytes are contiguous in the
  // tensor (one box row; one tile::gather4 row) and in the tile's logical order form one run. A
  // run is one Load of all its bytes and one Store of that value; the memory linking binds the
  // run's read and write fragments in order at their value byte offsets (LinkTmaRuns). A
  // swizzled destination thus keeps the row's logical byte order in one value instead of one
  // Load per 16-byte unit. Exact: every fragment keeps its own address and bytes; only the
  // value each fragment belongs to changes. A transfer with a runtime-written tensor map, a
  // concretely addressed source, a repeated source fragment (multicast) or an endpoint-less
  // pair keeps one Load per pair.
  // O(pairs) per transfer.
  struct Run { std::size_t begin, end; std::uint64_t bytes; };
  std::vector<Run> runs;
  {
    bool rows = !Has(*instruction_, "bulk_group") && !transfer.descriptor;
    std::set<std::tuple<algorithm::AddressId, std::optional<std::uint64_t>, std::uint64_t>> keys;
    // Only a symbolically placed source (a tensor-map region read, memory elimination's symbolic
    // initial reads): a concretely addressed source keeps one Load per pair, whose whole-value
    // initial reads form the affine initial-read families that source templates rely on.
    for (const auto& pair : transfer.pairs)
      if (!pair.source || !pair.destination || !pair.source->second || pair.source->second != pair.destination->second ||
          !abstract_endpoint(pair, pair.source) ||
          !keys.insert({pair.source->first, pair.source->second, pair.logical_offset}).second) { rows = false; break; }
    const auto continues = [&](const TmaPair& a, const TmaPair& b) {
      const auto bytes = *a.source->second;
      if (a.target_rank != b.target_rank || b.logical_offset != a.logical_offset + bytes) return false;
      const bool abstract = abstract_endpoint(a, a.source);
      if (abstract != abstract_endpoint(b, b.source)) return false;
      if (abstract)
        return a.place->region == b.place->region && a.place->in_bounds == b.place->in_bounds &&
            a.place->offset && b.place->offset && *b.place->offset == *a.place->offset + bytes;
      const auto& x = resources->Binding(a.source->first);
      const auto& y = resources->Binding(b.source->first);
      return x.expression.empty() && y.expression.empty() && x.alias_scope.empty() && y.alias_scope.empty() &&
          x.allocation == y.allocation && x.address_space == y.address_space && y.offset == x.offset + bytes;
    };
    for (std::size_t i = 0; rows && i < transfer.pairs.size();) {
      auto end = i + 1;
      std::uint64_t bytes = *transfer.pairs[i].source->second;
      while (end < transfer.pairs.size() && continues(transfer.pairs[end - 1], transfer.pairs[end]))
        bytes += *transfer.pairs[end++].source->second;
      runs.push_back({i, end, bytes});
      i = end;
    }
    if (!rows || std::all_of(runs.begin(), runs.end(), [](const Run& r) { return r.end == r.begin + 1; }))
      runs.clear();
  }
  for (const auto& run : runs) {
    const auto& first = transfer.pairs[run.begin];
    const auto type = v::ValueType::matrix(v::ElementType::bit(8), {run.bytes});
    v::Value data;
    if (abstract_endpoint(first, first.source)) {
      v::MemoryAttributes attributes;
      std::vector<v::Value> operands{placed(*first.source, *first.place, attributes)};
      if (const auto& guard = origins.at(first.place->region)->guard) {
        attributes.guarded = true;
        operands.push_back(guard->value);
      }
      data = Add(std::make_unique<v::LoadOp>(type, std::move(attributes)), std::move(operands), origin_).result();
    } else data = Add(std::make_unique<v::LoadOp>(type), {address(*first.source)}, origin_).result();
    if (abstract_endpoint(first, first.destination)) {
      v::MemoryAttributes attributes;
      const auto destination = placed(*first.destination, *first.place, attributes);
      Add(std::make_unique<v::StoreOp>(std::move(attributes)), {destination, data}, origin_);
    } else Add(std::make_unique<v::StoreOp>(), {address(*first.destination), data}, origin_);
  }
  if (!runs.empty()) return true;
  for (const auto& pair : transfer.pairs) {
    const bool abstract_source = abstract_endpoint(pair, pair.source);
    const bool abstract_destination = abstract_endpoint(pair, pair.destination);
    std::optional<v::Value> data;
    if (pair.source) {
      const auto key = std::tuple{pair.source->first, pair.source->second, pair.logical_offset};
      if (const auto found = reads.find(key); found != reads.end()) data = found->second;
      else {
        const auto type = v::ValueType::matrix(v::ElementType::bit(8), {*pair.source->second});
        if (abstract_source) {
          v::MemoryAttributes attributes;
          std::vector<v::Value> operands{placed(*pair.source, *pair.place, attributes)};
          if (const auto& guard = origins.at(pair.place->region)->guard) {
            attributes.guarded = true;
            operands.push_back(guard->value);
          }
          data = Add(std::make_unique<v::LoadOp>(type, std::move(attributes)), std::move(operands), origin_).result();
        } else data = Add(std::make_unique<v::LoadOp>(type), {address(*pair.source)}, origin_).result();
        reads.emplace(key, *data);
      }
    }
    if (!pair.destination) continue;
    const auto type = v::ValueType::matrix(v::ElementType::bit(8), {*pair.destination->second});
    if (!data) data = Add(std::make_unique<v::ConstantOp>(type, "0"), {}, origin_).result();
    if (graph_.type(*data) != type) throw std::invalid_argument("Resolved TMA fragment byte extents differ");
    if (abstract_destination) {
      v::MemoryAttributes attributes;
      const auto destination = placed(*pair.destination, *pair.place, attributes);
      Add(std::make_unique<v::StoreOp>(std::move(attributes)), {destination, *data}, origin_);
    } else Add(std::make_unique<v::StoreOp>(), {address(*pair.destination), *data}, origin_);
  }
  return true;
}

std::optional<std::vector<std::optional<ValueTemplateExtractor::TmaRegionOrigin>>>
ValueTemplateExtractor::TmaRegionOrigins(const TmaAccesses& transfer, const nlohmann::json& descriptor) {
  const auto& i = *instruction_;
  const bool gather = Has(i, "tile::gather4"), load = !Has(i, "bulk_group");
  const auto strides = descriptor.value("coordinate_strides_bytes", nlohmann::json());
  const auto base = descriptor.value("base_byte_offset", nlohmann::json(0));
  const auto global = descriptor.value("global_shape", nlohmann::json());
  const auto integral = [](const nlohmann::json& x) { return x.is_number_integer() && !x.is_boolean(); };
  if (!strides.is_array() || strides.empty() || (gather && strides.size() != 2) || !integral(base) ||
      !std::all_of(strides.begin(), strides.end(), integral))
    return std::nullopt;
  const auto tensor = Trim(module_.view(i.operands.at(load ? 1 : 0)));
  const auto open = tensor.find('{'), close = tensor.find('}');
  if (open == std::string::npos || close == std::string::npos || close < open) return std::nullopt;
  const auto tokens = Split(std::string_view(tensor).substr(open + 1, close - open - 1), ',');
  const std::size_t dims = strides.size();
  if (tokens.size() != (gather ? 5 : dims)) return std::nullopt;
  const auto s32 = *PtxType("s32"), s64 = *PtxType("s64");
  const auto binary = [&](v::BinaryOpcode opcode, const Type& type, v::Value x, v::Value y) {
    return Expression{Add(std::make_unique<v::BinaryOp>(opcode, type.value), {x, y}, origin_).result(), type};
  };
  const auto compare = [&](v::ComparePredicate predicate, v::Value x, v::Value y) {
    return Expression{Add(std::make_unique<v::CompareOp>(predicate, Pred().value), {x, y}, origin_).result(), Pred()};
  };
  std::vector<std::optional<TmaRegionOrigin>> origins(gather ? 4 : 1);
  for (const auto& pair : transfer.pairs) {
    // Every endpoint at a non-constant region is an abstract pair's global endpoint.
    for (const auto* endpoint : {&pair.source, &pair.destination})
      if (*endpoint && !resources->Binding((*endpoint)->first).expression.empty() && !pair.place)
        return std::nullopt;
    if (!pair.place) continue;
    const auto& place = *pair.place;
    if (!place.offset || place.region >= origins.size()) return std::nullopt;
    // A region not proven inside the tensor: only a zero-filled gather4 row (its column is
    // inside, or it has no offset).
    const bool guarded = !place.in_bounds;
    if (guarded && (!gather || !load || descriptor.value("oob_fill", std::string()) != "zero" ||
                    !global.is_array() || global.size() != 2 || !integral(global[1]) ||
                    global[1].get<std::int64_t>() < 1 || global[1].get<std::int64_t>() > INT32_MAX))
      return std::nullopt;
    auto& origin = origins[place.region];
    if (origin) continue;
    // A region proven inside the tensor whose first byte is exact in this CTA and compared by class
    // admission up to δ (ResolvedTma::region_valued): that literal is its origin. The value view is
    // then the per-unit one of this CTA at this region, and a class member reads the same cells
    // moved by its δ of the allocation, with no coordinate arithmetic left in the address.
    if (!guarded && place.region < transfer.region_valued.size() && transfer.region_valued[place.region]) {
      origin = TmaRegionOrigin{Constant(s64, std::to_string(*transfer.region_valued[place.region])), std::nullopt, true};
      continue;
    }
    auto offset = Constant(s64, std::to_string(base.get<std::int64_t>()));
    std::optional<Expression> guard;
    for (std::size_t dim = 0; dim < dims; ++dim) {
      auto coordinate = Operand(Trim(tokens.at(gather && dim == 1 ? 1 + place.region : dim)), s32);
      if (coordinate) coordinate = Reinterpret(*coordinate, s32);
      if (!coordinate) return std::nullopt;
      if (guarded && dim == 1) {
        const auto rows = global[1].get<std::int64_t>();
        const auto zero = Constant(s32, "0"), last = Constant(s32, std::to_string(rows - 1));
        guard = binary(v::BinaryOpcode::BitAnd, Pred(),
            compare(v::ComparePredicate::GreaterEqual, coordinate->value, zero.value).value,
            compare(v::ComparePredicate::LessEqual, coordinate->value, last.value).value);
        coordinate = binary(v::BinaryOpcode::Maximum, s32,
            binary(v::BinaryOpcode::Minimum, s32, coordinate->value, last.value).value, zero.value);
      }
      const auto wide = Add(std::make_unique<v::ConvertOp>(s64.value), {coordinate->value}, origin_).result();
      const auto stride = Constant(s64, std::to_string(strides[dim].get<std::int64_t>()));
      offset = binary(v::BinaryOpcode::Add, s64, offset.value,
                      binary(v::BinaryOpcode::Multiply, s64, wide, stride.value).value);
    }
    origin = TmaRegionOrigin{std::move(offset), std::move(guard)};
  }
  return origins;
}

} // namespace
} // namespace sentryward::frontend::ptx
