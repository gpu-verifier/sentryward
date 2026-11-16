#pragma once
#include "value_ir_impl.hpp"

namespace sentryward::frontend::ptx {
namespace {
void ValueTemplateExtractor::Memory() {
  const auto &i = *instruction_;
  // The optional cache policy is a performance hint, not a data operand.
  const bool cache_hint = Has(i, "L2::cache_hint");
  if (i.operands.size() != (cache_hint ? 3u : 2u)) {
    Fail("unsupported_memory_operands", "Memory operand form is not a scalar address plus value/register vector"); Invalidate(); return;
  }
  std::optional<Type> type;
  std::uint32_t count = 1;
  for (const auto &modifier : i.modifiers) {
    if (auto parsed = PtxType(modifier)) type = *parsed;
    if (modifier == "v2") count = 2;
    if (modifier == "v4") count = 4;
    if (modifier == "v8") count = 8;
  }
  if (!type || type->elements != 1) {
    Fail("unsupported_memory_type", "Memory element type is unsupported"); Invalidate(); return;
  }
  const bool load = i.opcode != "st";
  std::optional<std::pair<std::string, Integer>> parameter;
  if (load && Has(i, "param")) {
    const auto text = Trim(module_.view(i.operands[1]));
    if (text.size() >= 2 && text.front() == '[' && text.back() == ']')
      parameter = SplitAddress(std::string_view(text).substr(1, text.size() - 2));
    if (!parameter) {
      Fail("unresolved_parameter_field", "Parameter field requires a static base and constant byte offset"); Invalidate(); return;
    }
    if (const auto* known = facts_.Value(*record_, parameter->first))
      if (const auto* pointer = std::get_if<ParameterPointer>(known)) {
        parameter->first = pointer->region;
        parameter->second += pointer->offset;
      }
    const auto declaration = std::find_if(function_.parameters.begin(), function_.parameters.end(),
        [&](const auto& declared) { return declared.name == parameter->first; });
    if (declaration == function_.parameters.end()) {
      Fail("unresolved_parameter_field", "Parameter field address has no statically resolved kernel-parameter identity"); Invalidate(); return;
    }
    std::optional<Integer> extent;
    for (const auto& spelling : declaration->types) {
      const auto element = PtxType(spelling.starts_with('.') ? std::string_view(spelling).substr(1) : spelling);
      if (element && element->bits % 8 == 0) extent = element->bits / 8 * element->elements;
    }
    if (extent && declaration->array) {
      const auto& array = *declaration->array;
      const auto elements = array.size() >= 2 && array.front() == '[' && array.back() == ']'
          ? ParseInteger(std::string_view(array).substr(1, array.size() - 2)) : std::optional<Integer>{};
      if (elements && *elements >= 0) *extent *= *elements;
      else extent.reset();
    }
    const Integer bytes = Integer(count) * type->bits / 8;
    if (!extent || parameter->second < 0 || parameter->second + bytes > *extent) {
      Fail("parameter_field_out_of_bounds", "Parameter field access exceeds its declared PTX parameter storage"); Invalidate(); return;
    }
  }
  auto address = Address(std::string(module_.view(i.operands[load ? 1 : 0])));
  if (!address) { Invalidate(); return; }
  auto text = Trim(module_.view(i.operands[load ? 0 : 1]));
  if (text.starts_with('{') && text.ends_with('}')) text = text.substr(1, text.size() - 2);
  const auto components = Split(text, ',');
  if (components.size() != count) {
    Fail("unsupported_memory_vector", "Memory vector width disagrees with explicit register components"); Invalidate(); return;
  }
  // Scalar memory components share the native instruction/occurrence origin.
  // The component label records their precise byte/register association; this
  // is not a source-language tensor or a cross-thread assembled value.
  for (std::uint32_t n = 0; n < count; ++n) {
    auto component_address = *address;
    if (n) {
      auto offset = Constant(address->type, std::to_string(n * type->bits / 8));
      component_address = {Operation("ptx.address_add", "vector component byte offset", {address->type}, {*address, offset}).result(), address->type};
    }
    std::string mnemonic = i.opcode;
    for (const auto &modifier : i.modifiers) mnemonic += '.' + modifier;
    v::MemoryAttributes attributes;
    if (load && Has(i, "param")) {
      const Integer offset = parameter->second + n * type->bits / 8;
      std::optional<v::PointerProperties> pointer;
      for (const auto& binding : facts_.launch().pointer_parameters())
        if ((binding.parameter == parameter->first || (binding.parameter_index &&
              *binding.parameter_index < function_.parameters.size() &&
              function_.parameters[*binding.parameter_index].name == parameter->first)) &&
            offset == binding.byte_offset && type->bits / 8 == binding.width_bytes) {
          pointer = v::PointerProperties{binding.resource, "global", {}, {}};
          // A resource handle (input.argN.offset) is not necessarily the
          // allocation identity used by the checked memory model. Preserve
          // the declared object identity for an actual allocation-base input.
          // A nonzero suballocation base is not an allocation-base pointer.
          for(const auto& resource:facts_.launch().resources())
            if(resource.value("id",std::string{})==binding.resource && resource.contains("allocation")) {
              const auto& allocation=resource.at("allocation");
              if(allocation.value("resolution",std::string{})=="concrete" &&
                  allocation.value("base_byte_offset",nlohmann::json(0))==nlohmann::json(0)) {
                pointer->identity=allocation.at("object").get<std::string>();
                pointer->addressing=resource.value("space",std::string("global"));
              }
              break;
            }
        }
      std::optional<std::string> specialized;
      // A floating field is specialized by its exact bits, read as b<width>.
      const bool floating = !type->value.element().isBitLike() && type->value.isScalar() &&
          type->value.element().kind() == v::ElementKind::Floating;
      const auto read_type = floating ? *PtxType("b" + std::to_string(type->bits)) : *type;
      if (!pointer && (type->value.element().isBitLike() || floating) && type->bits <= 64) {
        // Carry the existing public-input specialization across the value
        // boundary. Do not use simulated memory values or register contracts.
        if (auto region = facts_.launch().parameter_regions().find(parameter->first);
            region != facts_.launch().parameter_regions().end() && offset <= INT64_MAX) {
          if (auto value = region->second.ReadInteger(offset.convert_to<std::int64_t>(), type->bits / 8,
                  type->value.element().kind() == v::ElementKind::SignedInteger))
            specialized = value->convert_to<std::string>();
        } else if (count == 1 && !floating) {
          auto spelling = Trim(module_.view(i.operands[1]));
          if (spelling.starts_with('[') && spelling.ends_with(']'))
            spelling = Trim(std::string_view(spelling).substr(1, spelling.size() - 2));
          if (auto value = facts_.launch().parameters().find(spelling); value != facts_.launch().parameters().end())
            specialized = value->second.convert_to<std::string>();
        }
      }
      if (floating && specialized) {
        const auto &node = Add(std::make_unique<v::InputParameterOp>(read_type.value,
            parameter->first + "+" + offset.convert_to<std::string>(), pointer, specialized), {}, origin_);
        auto value = Reinterpret(Expression{node.result(), read_type}, *type);
        if (!value) { Invalidate(); return; }
        Bind(components[n], *value);
        continue;
      }
      const auto &node = Add(std::make_unique<v::InputParameterOp>(type->value,
          parameter->first + "+" + offset.convert_to<std::string>(), pointer, specialized), {}, origin_);
      Bind(components[n], {node.result(), *type});
      continue;
    }
    if (load) {
      std::vector<v::Value> operands{component_address.value};
      const auto &node = Add(std::make_unique<v::LoadOp>(type->value, attributes), std::move(operands), origin_);
      Bind(components[n], {node.result(), *type});
    } else {
      auto value = Operand(components[n], *type);
      if (value && type->value.isScalar() && type->value.element().isBitLike() &&
          value->type.value.bitSize() > type->value.bitSize()) {
        // Narrow stores use the low bits, including packed register values.
        value = Part(*value, *type, 0);
      }
      if (value) value = Reinterpret(*value, *type);
      if (!value) continue;
      std::vector<v::Value> operands{component_address.value, value->value};
      Add(std::make_unique<v::StoreOp>(attributes), std::move(operands), origin_);
    }
  }
}

void ValueTemplateExtractor::BulkCopy() {
  const auto& i = *instruction_;
  std::vector<std::string> spaces;
  std::set<std::string> seen;
  // A load completes on an mbarrier; a store (shared::cta to global) through bulk async-groups.
  const bool store = i.semantics.operation == "bulk_copy";
  for (const auto& modifier : i.modifiers) {
    if (!In(modifier, {"async", "bulk", "weak", "shared::cta", "shared::cluster", "global",
                       store ? "bulk_group" : "mbarrier::complete_tx::bytes", "L2::cache_hint"}) ||
        !seen.insert(modifier).second) {
      Fail("unsupported_bulk_copy", "Unsupported linear bulk-copy modifier"); return;
    }
    if (In(modifier, {"shared::cta", "shared::cluster", "global"})) spaces.push_back(modifier);
  }
  if (spaces.size() != 2 ||
      !(store ? spaces[0] == "global" && spaces[1] == "shared::cta" && Has(i, "bulk_group")
              : (In(spaces[0], {"shared::cta", "shared::cluster"}) && spaces[1] == "global") ||
                (spaces[0] == "shared::cluster" && spaces[1] == "shared::cta")) ||
      (Has(i, "L2::cache_hint") && !store && spaces[1] != "global") ||
      i.operands.size() != (Has(i, "L2::cache_hint") ? 1u : 0u) + (store ? 3u : 4u)) {
    Fail("unsupported_bulk_copy", "Unsupported linear bulk-copy operands or spaces"); return;
  }
  const auto bytes = Known(Trim(module_.view(i.operands[2])));
  if (!bytes || *bytes <= 0 || *bytes > UINT32_MAX || *bytes % 16 != 0) {
    Fail("unsupported_bulk_copy", "Bulk-copy byte count must be a positive constant multiple of 16"); return;
  }
  const auto size = bytes->convert_to<std::uint64_t>();
  const bool cluster = spaces[0] == "shared::cluster" && spaces[1] == "shared::cta";
  const auto destination = LinearBulkEndpoint(0, size, spaces[0], cluster),
             source = LinearBulkEndpoint(1, size, spaces[1], cluster);
  if (!destination || !source) {
    Fail("unsupported_bulk_copy", "Bulk-copy endpoint requires a resolved allocation and byte range"); return;
  }
  const auto type = v::ValueType::matrix(v::ElementType::bit(8), {size});
  auto data = Add(std::make_unique<v::LoadOp>(type), {*source}, origin_).result();
  Add(std::make_unique<v::StoreOp>(), {*destination, data}, origin_);
}

// The address of a linear bulk operand (`[...]` of `size` bytes in `space`): the slot value
// of its resolved constant binding, else the operand as a pointer of that space. A shared::cluster
// to shared::cta copy has no pointer fallback.
std::optional<v::Value> ValueTemplateExtractor::LinearBulkEndpoint(unsigned operand, std::uint64_t size,
                                                                   const std::string& space, bool cluster_source) {
  const auto& i = *instruction_;
  auto text = Trim(module_.view(i.operands[operand]));
  if (!text.starts_with('[') || !text.ends_with(']')) return {};
  if (resources) {
    resources->SetLane(lane_);
    const auto fragment = resources->Memory(*record_, MemoryRequest(text.substr(1, text.size()-2),
                                           nlohmann::json::object(), size, space));
    const auto& binding = resources->Binding(fragment.first);
    if (fragment.second == size && binding.expression.empty() && binding.alias_scope.empty())
      return SlotAddressValue(binding);
  }
  if (cluster_source) return {};
  auto address = Address(text);
  if (!address) return {};
  return space == "global" ? graph_.asPointer(address->value, {"", "global", {}, {}}) :
                             graph_.asPointer(address->value, {"", "shared", cta_, {}});
}

// cp.reduce.async.bulk[.relaxed.scope].global.shared::cta.bulk_group[.L2::cache_hint].{add,min,max}.type
// [dst], [src], size: each element of the shared payload combines into the global element at the same
// offset; completion through bulk async-groups (PTX ISA, cp.reduce.async.bulk). The synchronization side
// models each element as an atomic read-modify-write (lowering.cpp bulk_reduce). Value: a Load of the
// payload as elements of the reduction type, then a reduction Store (MemoryAttributes::reduction).
// add on f32 (ideal-real, order-free) and on u32/s32/u64 (modulo 2^w); min/max on u32/s32/u64/s64.
void ValueTemplateExtractor::BulkReduce() {
  const auto& i = *instruction_;
  std::optional<std::string> operation, element;
  std::vector<std::string> spaces;
  std::set<std::string> seen;
  for (const auto& modifier : i.modifiers) {
    if (!seen.insert(modifier).second) { Fail("reduction_operation_unsupported", "Repeated bulk-reduction modifier"); return; }
    if (In(modifier, {"reduce", "async", "bulk", "bulk_group", "L2::cache_hint", "relaxed", "gpu", "sys"})) continue;
    if (In(modifier, {"global", "shared::cta"})) { spaces.push_back(modifier); continue; }
    if (!operation && In(modifier, {"add", "min", "max"})) { operation = modifier; continue; }
    if (!element && In(modifier, {"f32", "u32", "s32", "u64", "s64"})) { element = modifier; continue; }
    Fail("reduction_operation_unsupported", "Unsupported bulk-reduction operation, type or modifier"); return;
  }
  if (!operation || !element || spaces != std::vector<std::string>{"global", "shared::cta"} ||
      (*operation == "add" && *element == "s64") || (*operation != "add" && *element == "f32") ||
      i.operands.size() != (Has(i, "L2::cache_hint") ? 4u : 3u)) {
    Fail("reduction_operation_unsupported", "Unsupported bulk-reduction operation, type or operands"); return;
  }
  const auto type = PtxType(*element);
  const auto bytes = Known(Trim(module_.view(i.operands[2])));
  if (!type || !bytes || *bytes <= 0 || *bytes > UINT32_MAX || *bytes % 16 != 0) {
    Fail("reduction_operation_unsupported", "Bulk-reduction byte count must be a positive constant multiple of 16"); return;
  }
  const auto size = bytes->convert_to<std::uint64_t>();
  const auto destination = LinearBulkEndpoint(0, size, "global", false),
             source = LinearBulkEndpoint(1, size, "shared::cta", false);
  if (!destination || !source) {
    Fail("reduction_operation_unsupported", "Bulk-reduction endpoint requires a resolved allocation and byte range"); return;
  }
  const auto payload = v::ValueType::matrix(type->value.element(), {size / (type->bits / 8)});
  auto data = Add(std::make_unique<v::LoadOp>(payload), {*source}, origin_).result();
  v::MemoryAttributes attributes;
  attributes.reduction = v::MemoryAttributes::Reduction{*operation, type->value.element()};
  Add(std::make_unique<v::StoreOp>(std::move(attributes)), {*destination, data}, origin_);
}

std::optional<v::Value> ValueTemplateExtractor::SpaceAddress(std::string_view operand, std::string_view space,
                                                            std::optional<Expression> extra) {
  const std::string text = Trim(operand);
  if (text.size() < 2 || text.front() != '[' || text.back() != ']') return {};
  const auto split = SplitAddress(std::string_view(text).substr(1, text.size() - 2));
  if (!split) return {};
  auto base = Operand(split->first, U64());
  if (!base || base->type.elements != 1 || base->type.spelling == "pred" ||
      base->type.spelling.starts_with('f') || base->type.spelling.starts_with("bf")) return {};
  if (base->type.value.element().kind() == v::ElementKind::Bit)
    base = Reinterpret(*base, *PtxType("u" + std::to_string(base->type.bits)));
  if (!base) return {};
  const auto add = [&](const Expression& offset) {
    base = Expression{Operation("ptx.address_add", "byte offset", {base->type}, {*base, offset}).result(), base->type};
  };
  if (split->second != 0) add(Constant(base->type, split->second.convert_to<std::string>()));
  if (extra) {
    auto offset = *extra;
    if (offset.type.value != base->type.value)
      offset = Expression{Add(std::make_unique<v::ConvertOp>(base->type.value), {offset.value}, origin_).result(), base->type};
    add(offset);
  }
  v::PointerProperties pointer{"", std::string(space), {}, {}};
  if (space == "shared") pointer.owner_cta = cta_;
  return graph_.asPointer(base->value, pointer);
}

// Tensor-map edits, with the memory effects lowering gives them. TMA consumers
// take their descriptor from the executor's resolution (ResolvedTma); these
// effects keep the descriptor bytes themselves sound for any other reader.
// - tensormap.replace reads and rewrites the 128-byte object at its address.
//   CUtensorMap's layout is opaque, so the rewritten object is an OpaqueValue.
// - tensormap.cp_fenceproxy (warp-aligned) copies it to global memory, each lane
//   its own 4 bytes at 4 * lane.
// An edit this cannot place is unsupported, never a no-op.
void ValueTemplateExtractor::TensorMap() {
  const auto& i = *instruction_;
  constexpr std::uint64_t kBytes = 128;
  const auto text = [&](std::size_t n) { return Trim(module_.view(i.operands.at(n))); };
  const auto bytes = [](std::uint64_t n) { return v::ValueType::matrix(v::ElementType::bit(8), {n}); };
  if (Has(i, "replace")) {
    std::optional<std::string> space;
    if (Has(i, "shared::cta")) space = "shared";
    else if (Has(i, "global")) space = "global";
    if (!Has(i, "tile") || !space || i.operands.size() < 2 || i.operands.size() > 3) {
      Fail("unsupported_tensormap", "Unsupported tensormap.replace form"); Invalidate(); return;
    }
    const auto target = SpaceAddress(text(0), *space, std::nullopt);
    if (!target) {
      Fail("unsupported_tensormap", "tensormap.replace target is not an addressable object"); Invalidate(); return;
    }
    Add(std::make_unique<v::LoadOp>(bytes(kBytes)), {*target}, origin_);
    const auto opaque = Add(std::make_unique<v::OpaqueValueOp>(bytes(kBytes), "tensormap.replace"), {}, origin_).result();
    Add(std::make_unique<v::StoreOp>(), {*target, opaque}, origin_);
    return;
  }
  if (Has(i, "cp_fenceproxy")) {
    const auto size = i.operands.size() == 3 ? Known(text(2)) : std::nullopt;
    if (!Has(i, "global") || !Has(i, "shared::cta") || !Has(i, "aligned") || !size || *size != Integer(kBytes)) {
      Fail("unsupported_tensormap", "Unsupported tensormap.cp_fenceproxy form"); Invalidate(); return;
    }
    const auto binary = [&](v::BinaryOpcode op, v::Value a, v::Value b) {
      return Add(std::make_unique<v::BinaryOp>(op, U32().value), {a, b}, origin_).result();
    };
    const auto lane = binary(v::BinaryOpcode::BitAnd, LinearThread().value, Constant(U32(), "31").value);
    const Expression offset{binary(v::BinaryOpcode::Multiply, lane, Constant(U32(), "4").value), U32()};
    const auto destination = SpaceAddress(text(0), "global", offset), source = SpaceAddress(text(1), "shared", offset);
    if (!destination || !source) {
      Fail("unsupported_tensormap", "tensormap.cp_fenceproxy endpoints are not addressable"); Invalidate(); return;
    }
    const auto data = Add(std::make_unique<v::LoadOp>(bytes(4)), {*source}, origin_).result();
    v::MemoryAttributes descriptor_memory;
    descriptor_memory.interpretation = v::MemoryAttributes::Interpretation::TensorMap;
    Add(std::make_unique<v::StoreOp>(descriptor_memory), {*destination, data}, origin_);
    return;
  }
  Fail("unsupported_tensormap", "Unsupported tensormap operation"); Invalidate();
}

// mbarrier.inval ends the object's lifetime; its location may then serve another purpose (PTX ISA,
// mbarrier.inval), so its 8 bytes are unspecified afterwards. Lowering records the same 8-byte
// shared write (ImmediateStore); here the written value is opaque. Generic addressing is not placed.
void ValueTemplateExtractor::MbarrierInvalidate() {
  const auto& i = *instruction_;
  constexpr std::uint64_t kBytes = 8;
  const auto type = v::ValueType::matrix(v::ElementType::bit(8), {kBytes});
  const bool shared = i.modifiers == std::vector<std::string>{"inval", "shared", "b64"} ||
      i.modifiers == std::vector<std::string>{"inval", "shared::cta", "b64"};
  const auto target = shared && i.operands.size() == 1 ?
      SpaceAddress(module_.view(i.operands[0]), "shared", std::nullopt) : std::nullopt;
  if (!target) {
    Fail("unsupported_mbarrier_invalidation", "mbarrier.inval needs a .shared::cta address of the object");
    Invalidate(); return;
  }
  const auto opaque = Add(std::make_unique<v::OpaqueValueOp>(type, "mbarrier.inval"), {}, origin_).result();
  Add(std::make_unique<v::StoreOp>(), {*target, opaque}, origin_);
}

// The condition under which a cp.async with a data-dependent control copies its whole source and
// whether that is its true value. PTX ISA cp.async: src-size bytes are copied and the rest of
// cp-size zero-filled; a true ignore-src predicate copies none. A source size qualifies when it is a
// selp of two constants each 0 or `bytes`; anything else is nullopt.
std::optional<std::pair<Expression, bool>> ValueTemplateExtractor::WholeCopyCondition(
    std::string operand, bool predicate, const Integer &bytes) {
  if (predicate) {
    auto ignore = Operand(operand, Pred());
    if (!ignore || ignore->missing_definition) return std::nullopt;
    return std::pair{*ignore, false};
  }
  auto size = Operand(operand, U32());
  if (!size || size->missing_definition || size->value.conversion() || size->value.threadSelector() ||
      !size->value.indices().empty() || !dynamic_cast<const v::SelectOp*>(&graph_.definingNode(size->value).operation()))
    return std::nullopt;
  const auto arms = graph_.instanceOperands(size->value);
  if (arms.size() != 3) return std::nullopt;
  for (std::size_t k = 1; k < 3; ++k) {
    const auto& arm = arms[k];
    if (arm.conversion() || arm.threadSelector() || !arm.indices().empty()) return std::nullopt;
    const auto* constant = dynamic_cast<const v::ConstantOp*>(&graph_.definingNode(arm).operation());
    const auto value = constant && !constant->pointer() ? ParseInteger(constant->literal()) : std::nullopt;
    if (!value || (*value != 0 && *value != bytes)) return std::nullopt;
  }
  const auto zero = Constant(size->type, "0");
  const Expression copies{Add(std::make_unique<v::CompareOp>(v::ComparePredicate::NotEqual, Pred().value),
                              {size->value, zero.value}, origin_).result(), Pred()};
  return std::pair{copies, true};
}

void ValueTemplateExtractor::AsyncCopy() {
  const auto &i = *instruction_;
  if (i.operands.size() < 3 || i.operands.size() > 4 ||
      !Has(i, "global") || !(Has(i, "shared") || Has(i, "shared::cta"))) {
    Fail("unsupported_async_copy", "Only direct global-to-shared cp.async with fixed byte counts is supported"); return;
  }
  const auto bytes = ParseInteger(Trim(module_.view(i.operands[2])));
  auto source_bytes = bytes;
  // A data-dependent control that copies all or none of the source: the condition and whether
  // its true value copies.
  std::optional<std::pair<Expression, bool>> whole_copy;
  if (i.operands.size() == 4) {
    const auto operand = Trim(module_.view(i.operands[3]));
    source_bytes = Known(operand);
    if (!ParseInteger(operand)) {
      const auto declaration = declarations_.find(Resolve(operand, i.scope));
      if (declaration == declarations_.end() ||
          (declaration->second.bits != 1 && declaration->second.bits != 32)) {
        Fail("unsupported_async_copy", "cp.async source size requires a 32-bit register or predicate"); return;
      }
      if (source_bytes && declaration->second.bits == 1) {
        if (*source_bytes != 0 && *source_bytes != 1) {
          Fail("unsupported_async_copy", "cp.async ignore-source predicate is not Boolean"); return;
        }
        source_bytes = *source_bytes != 0 ? std::optional<Integer>(Integer(0)) : bytes;
      }
      // A lifted region knows no per-thread value (Known), so its controls stay refused.
      if (!source_bytes && bytes && !lifting_) {
        whole_copy = WholeCopyCondition(std::string(operand), declaration->second.bits == 1, *bytes);
        if (whole_copy) source_bytes = bytes;
      }
      // Any other data-dependent size would need a per-byte select(k < src-size, loaded, 0).
      if (!source_bytes) {
        Fail("unsupported_async_copy", "cp.async source size is data-dependent and not either 0 or the copy size; "
             "its zero-filled destination is not modeled in the value analysis"); return;
      }
    }
  }
  if (!bytes || !source_bytes || (*bytes != 4 && *bytes != 8 && *bytes != 16) || *source_bytes < 0 || *source_bytes > *bytes) {
    Fail("unsupported_async_copy", "cp.async size/source-size must be statically determined legal byte counts"); return;
  }
  auto destination = Address(std::string(module_.view(i.operands[0])));
  auto source = *source_bytes ? Address(std::string(module_.view(i.operands[1]))) : std::optional<Expression>{};
  if (!destination || (*source_bytes && !source)) return;
  destination->value = graph_.asPointer(destination->value, {"", "shared", cta_, {}});
  if (source) source->value = graph_.asPointer(source->value, {"", "global", {}, {}});
  const auto count = bytes->convert_to<std::uint32_t>();
  const auto valid_count = source_bytes->convert_to<std::uint32_t>();
  // All or none of the source by a data-dependent control: select(copies, loaded, 0) of the whole
  // payload (the byte payload below, and the zero literal of the suffix). The read is possible, not
  // certain, as lowering models it.
  if (whole_copy) {
    const auto type = v::ValueType::matrix(v::ElementType::bit(8), {count});
    const auto data = Add(std::make_unique<v::LoadOp>(type), {source->value}, origin_).result();
    const auto zero = Add(std::make_unique<v::ConstantOp>(type, "0"), {}, origin_).result();
    const auto& [condition, copies_when_true] = *whole_copy;
    const auto selected = Add(std::make_unique<v::SelectOp>(type),
        {condition.value, copies_when_true ? data : zero, copies_when_true ? zero : data}, origin_).result();
    Add(std::make_unique<v::StoreOp>(), {destination->value, selected}, origin_);
    return;
  }
  // A cp.async has one contiguous source payload, not unrelated byte reads.
  // Keep that payload identity so wider numerical consumers can select its
  // components even when the global address contains a dynamic page index.
  if (valid_count) {
    const auto type=v::ValueType::matrix(v::ElementType::bit(8),{valid_count});
    const auto data=Add(std::make_unique<v::LoadOp>(type),{source->value},origin_).result();
    Add(std::make_unique<v::StoreOp>(),{destination->value,data},origin_);
  }
  // The ISA source-size operand specifies an exact zero-filled suffix, for which no source read is
  // issued: one whole-byte zero literal of its length after the copied prefix, the representation of
  // a TMA tile's zero fill, so that a consumer may read any element range of it.
  if (valid_count < count) {
    auto suffix = *destination;
    if (valid_count) {
      const auto offset = Constant(suffix.type, std::to_string(valid_count));
      suffix = Expression{Operation("ptx.address_add", "", {suffix.type}, {suffix, offset}).result(), suffix.type};
    }
    const auto zero = Add(std::make_unique<v::ConstantOp>(
        v::ValueType::matrix(v::ElementType::bit(8), {count - valid_count}), "0"), {}, origin_).result();
    Add(std::make_unique<v::StoreOp>(), {suffix.value, zero}, origin_);
  }
}

} // namespace
} // namespace sentryward::frontend::ptx
