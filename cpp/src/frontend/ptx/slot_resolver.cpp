#include "sentryward/frontend/ptx/slot_resolver.h"
#include "sentryward/frontend/ptx/layouts.h"
#include "sentryward/frontend/ptx/tensor_maps.h"
#include "sentryward/frontend/ptx/value_ir.h"
#include "sentryward/algorithm/memory/symbolic_address_model.h"
#include <algorithm>
#include <sstream>
namespace sentryward::frontend::ptx {

using Json = nlohmann::json;
namespace {
// `n` as int64, or nothing when it does not fit.
std::optional<std::int64_t> Int64(const Integer &n) {
  if (n < INT64_MIN || n > INT64_MAX) return std::nullopt;
  return n.convert_to<std::int64_t>();
}
// The offset of a bounded pointer (tensor_properties), or nothing when a value
// does not fit int64.
std::optional<AbstractAffine> AffineOffset(const AffineForm &form, const Integer &displacement) {
  const auto constant = Int64(form.constant + displacement);
  if (!constant) return std::nullopt;
  AbstractAffine result{*constant, {}};
  for (const auto &[atom, coefficient] : form.terms) {
    const auto factor = Int64(coefficient), offset = Int64(atom.byte_offset);
    if (!factor || !offset) return std::nullopt;
    result.terms.push_back({atom.object, atom.element_type, *offset, *factor});
  }
  return result;
}
// The address-model expression of that offset: constant + coefficient x element lookups.
Json AffineOffsetJson(const AbstractAffine &affine) {
  Json operands = Json::array({affine.constant});
  for (const auto &term : affine.terms)
    operands.push_back({{"operator", "mul"}, {"operands", Json::array({
        Json{{"operator", "gather"}, {"symbol", algorithm::TensorPropertySymbol(term.object, term.element_type)},
             {"args", Json::array({term.offset})}}, term.coefficient})}});
  return Json{{"operator", "add"}, {"operands", std::move(operands)}};
}
} // namespace
SlotResolver::SlotResolver(Resources &core, SlotConfig config)
    : core_(core), config_(std::move(config)) {}
SlotResolver::~SlotResolver() = default;
namespace {
// The outcome of one resolution: `body` runs inside one try; any exception is
// stored with the steps appended so far as the prefix.
// `body` returns the terminal as a prvalue, which initializes the outcome's
// terminal in place (the fallback is read after it, in list order).
template <class T, class Body>
SlotOutcome<T> Resolve(std::vector<SlotStep> &steps, FallbackUse &fallback, Body &&body) {
  try {
    return {{}, body(), fallback};
  } catch (const SemanticError &error) {
    return {std::move(steps), SlotError{error.code, std::current_exception()}, fallback};
  } catch (...) {
    return {std::move(steps), SlotError{{}, std::current_exception()}, fallback};
  }
}
// A footprint terminal built in place: ResolvedFootprint's members from `args`.
template <class... Args> SlotTerminal<ResolvedFootprint> Footprint(Args &&...args) {
  return SlotTerminal<ResolvedFootprint>(std::in_place_index<0>, std::forward<Args>(args)...);
}
} // namespace
SlotOutcome<ResolvedFootprint> SlotResolver::Memory(const RecordView &e, const MemoryRequest &request) {
  const FactSource::SlotScope slot(core_.facts);
  std::vector<SlotStep> prefix;
  auto fallback = FallbackUse::None;
  auto outcome = Resolve<ResolvedFootprint>(prefix, fallback,
                                            [&] { return MemoryBody(e, request, prefix, fallback); });
  // A successful resolution keeps its prefix (the fallback's base binding).
  if (!prefix.empty()) outcome.prefix = std::move(prefix);
  if (core_.facts.recording()) [[unlikely]] {
    const SlotQuery slot_query{{config_.consumer, core_.id_}, e, &request, nullptr, core_.active_lane_};
    core_.facts.Note(query::SlotMemory{&slot_query, &outcome});
  }
  return outcome;
}
SlotOutcome<ResolvedTma> SlotResolver::Tma(const RecordView &e, const TmaRequest &request) {
  const FactSource::SlotScope slot(core_.facts);
  std::vector<SlotStep> steps;
  auto fallback = FallbackUse::None;
  auto outcome = Resolve<ResolvedTma>(steps, fallback, [&] {
    auto resolved = TmaBody(e, request, steps, fallback);
    resolved.steps = std::move(steps);
    return SlotTerminal<ResolvedTma>(std::move(resolved));
  });
  if (core_.facts.recording()) [[unlikely]] {
    const SlotQuery slot_query{{config_.consumer, core_.id_}, e, nullptr, &request, core_.active_lane_};
    core_.facts.Note(query::SlotTma{&slot_query, &outcome});
  }
  return outcome;
}
ConstantStep SlotResolver::MemoryResource(std::string id, std::uint64_t offset,
                                          std::uint64_t bytes) {
  if(const auto cached=memory_resources_.find(id);cached!=memory_resources_.end()) {
    const auto& info=cached->second;
    if(info.extent && (offset>*info.extent || bytes>*info.extent-offset))
      throw SemanticError("pointer_extent", "access exceeds captured pointer allocation");
    if(offset>UINT64_MAX-info.base)
      throw SemanticError("invalid_memory_range", "allocation offset overflows");
    return core_.BindStep(info.space,info.object,offset+info.base,bytes,{});
  }
  const auto &facts = core_.facts;
  auto &records = core_.records_;
  auto it = records.find(id);
  if(it==records.end()) {
    // Value address provenance uses allocation identity; execution contracts
    // may use a distinct resource handle. Resolve that alias once, retaining
    // the original resource's extent validation. Never choose ambiguously or
    // reinterpret a nonzero suballocation pointer as the allocation base.
    std::optional<std::string> alias;
    for(const auto& pointer:facts.launch().pointer_parameters()) {
      const auto candidate=records.find(pointer.resource);
      if(candidate==records.end() || !candidate->second.contains("allocation"))continue;
      const auto& allocation=candidate->second.at("allocation");
      if(allocation.value("resolution",std::string{})!="concrete" ||
          allocation.value("object",std::string{})!=id ||
          allocation.value("base_byte_offset",Json(0))!=Json(0))continue;
      if(alias && *alias!=pointer.resource)
        throw SemanticError("ambiguous_memory_resource","Multiple pointer resources name allocation "+id);
      alias=pointer.resource;
    }
    if(alias) {
      auto result=MemoryResource(*alias,offset,bytes);
      memory_resources_.emplace(id,memory_resources_.at(*alias));
      return result;
    }
  }
  if (it == records.end() || it->second.value("kind", "") != "memory")
    throw SemanticError("unknown_memory_resource",
                        "unknown memory resource " + id);
  const auto &r = it->second;
  bool pointer = false;
  std::optional<std::uint64_t> extent;
  for (const auto &p : facts.launch().pointer_parameters())
    if (p.resource == id) {
      pointer = true;
      if(p.extent_bytes)extent=extent?std::min(*extent,*p.extent_bytes):*p.extent_bytes;
      if (p.extent_bytes &&
          (offset > *p.extent_bytes || bytes > *p.extent_bytes - offset))
        throw SemanticError("pointer_extent",
                            "access exceeds captured pointer allocation");
    }
  auto allocation = r.value("allocation", Json::object());
  std::string object = id;
  std::uint64_t base=0;
  if (!allocation.empty()) {
    if (allocation.value("resolution", "") != "concrete" ||
        !allocation.contains("base_byte_offset") ||
        !allocation.contains("object"))
      throw SemanticError("unresolved_footprint",
                          "resource has no concrete allocation");
    object = allocation["object"].get<std::string>();
    base = Unsigned(JsonInteger(allocation["base_byte_offset"]),
                         "allocation base");
    if (offset > UINT64_MAX - base)
      throw SemanticError("invalid_memory_range",
                          "allocation offset overflows");
    offset += base;
  } else if (!pointer)
    throw SemanticError("unresolved_footprint",
                        "resource has symbolic allocation");
  const auto space=r.value("space", "generic");
  auto result=core_.BindStep(space, object, offset, bytes, {});
  memory_resources_.emplace(std::move(id),MemoryResourceInfo{space,std::move(object),base,extent});
  return result;
}
std::optional<Json> SlotResolver::ExtractAddress(const RecordView &e, std::string_view operand) {
  if (!address_values_) address_values_ = std::make_unique<AddressValueExtractor>(core_.facts, core_.frame);
  return address_values_->Extract(*core_.active_lane_, e, operand, [this](LaneRef lane, DynamicKey key) {
    const auto previous = core_.active_lane_;
    core_.SetLane(lane);
    const auto result = core_.Definition(key, true);
    core_.SetLane(*previous);
    return result;
  });
}
SlotTerminal<ResolvedFootprint> SlotResolver::MemoryBody(const RecordView &e, const MemoryRequest &request,
                                                         std::vector<SlotStep> &prefix, FallbackUse &fallback) {
  const auto &facts = core_.facts;
  const auto &frame = core_.frame;
  const auto text = request.operand;
  const auto &contract = request.contract;
  const auto bytes = request.bytes;
  auto constant = [](ConstantStep &&step, std::uint64_t size) {
    return Footprint(FootprintKind::Constant, std::move(step), std::nullopt, size);
  };
  auto space = CanonicalSpace(request.space);
  if (space == "shared" || space == "cluster_shared")
    if (auto a = core_.Shared(e, text)) {
      if (space == "shared" && a->first != frame.self())
        throw SemanticError("shared_cta_owner",
                            "CTA local address points to another CTA");
      if (a->second + bytes > (1ull << 32))
        throw SemanticError("invalid_shared_range", "shared range outside u32");
      return constant(core_.BindStep("shared", "physical.shared.cta.0", a->second, bytes, a->first),
                      bytes);
    }
  auto token = Trim(text);
  if (token.starts_with('[') && token.ends_with(']'))
    token = Trim(std::string_view(token).substr(1, token.size() - 2));
  auto split = SplitAddress(token);
  std::string base = split ? split->first : token;
  Integer displacement = split ? split->second : Integer(0);
  auto v = core_.Value(e, base);
  if (space.empty() || space=="generic")
    if (auto owner=core_.GenericOwner(e,base))
      if (auto value=ScalarInteger(v)) {
        const auto offset=Unsigned(*value+displacement,"shared generic address",UINT32_MAX);
        if (bytes>(1ULL<<32)-offset) throw SemanticError("invalid_shared_range","shared range outside u32");
        return constant(core_.BindStep("shared","physical.shared.cta.0",offset,bytes,frame.Rank(*owner)),
                        bytes);
      }
  auto region = contract.value("region", Json::object());
  std::string id = contract.value("resource", "");
  // The step's own space (its key's), not the table: the binding interned for
  // it has that space (a stored binding equals its step). A mismatch leaves the
  // step interned.
  auto checked = [&](ConstantStep &&step, std::uint64_t size) {
    auto want = space == "cluster_shared" ? "shared" : space;
    if (!space.empty() && space != "generic" && step.key.space != want) {
      prefix.push_back({std::move(step)});
      throw SemanticError("memory_space_mismatch",
                          "resource space disagrees with instruction");
    }
    return constant(std::move(step), size);
  };
  if (!region.contains("byte_offset")) {
    if (auto p = std::get_if<AllocationPointer>(&v)) {
      if (!id.empty() && id != p->resource)
        throw SemanticError("pointer_resource_mismatch",
                            "contract disagrees with pointer provenance");
      auto offset = Unsigned(p->offset + displacement, "pointer offset");
      if (p->extent_bytes &&
          (offset > *p->extent_bytes || bytes > *p->extent_bytes - offset))
        throw SemanticError("pointer_extent", "access beyond captured extent");
      return checked(MemoryResource(p->resource, offset, bytes), bytes);
    }
    // A bounded pointer (tensor_properties) is an abstract footprint in its allocation
    // when its whole offset range lies inside the captured extent.
    if (const auto *pointer = std::get_if<BoundedAllocationPointer>(&v);
        pointer && (space.empty() || space == "global" || space == "generic")) {
      const auto &bounded = *pointer->value;
      if (!id.empty() && id != bounded.resource)
        throw SemanticError("pointer_resource_mismatch",
                            "contract disagrees with pointer provenance");
      const Integer lower = bounded.lower + displacement, upper = bounded.upper + displacement;
      // A non-affine offset (offset_identity) has no abstract footprint: the fallback below.
      const bool inside = lower >= 0 && bounded.offset_identity.empty() &&
          (!bounded.extent_bytes || upper + Integer(bytes) <= Integer(*bounded.extent_bytes));
      if (auto affine = inside ? AffineOffset(bounded.offset, displacement) : std::nullopt) {
        auto placement = MemoryResource(bounded.resource, 0, 0);
        if (placement.offset == 0) {
          const auto binding = frame.Format(placement.key);
          prefix.push_back({std::move(placement)});
          auto step = core_.BindStep(binding.space, binding.allocation, 0, bytes, {});
          const auto stored = frame.Format(step.key);
          auto expression = Resources::AbstractExpression(stored.space, stored.allocation, stored.alias,
                                                          AffineOffsetJson(*affine), bytes);
          return Footprint(FootprintKind::Abstract, std::move(step), std::move(expression), bytes,
                           std::move(*affine));
        }
      }
    }
    if (auto n = std::get_if<Integer>(&v); n && !ParseInteger(base)) {
      auto offset = Unsigned(*n + displacement, "numeric memory address");
      std::vector<std::pair<std::string, std::uint64_t>> matches;
      for (const auto &[key, r] : core_.records_) {
        if (r.value("kind", "") != "memory" || (!id.empty() && id != key))
          continue;
        auto s = CanonicalSpace(r.value("space", ""));
        if (!space.empty() && space != "generic" && s != space)
          continue;
        auto ad = r.value("address", Json::object());
        if (!ad.contains("value") || !ad.contains("bytes"))
          continue;
        auto begin = Unsigned(JsonInteger(ad["value"]), "range begin"),
             size = Unsigned(JsonInteger(ad["bytes"]), "range bytes");
        if (offset >= begin && offset - begin <= size &&
            bytes <= size - (offset - begin))
          matches.emplace_back(key, begin);
      }
      if (!matches.empty()) {
        std::set<std::string> spaces;
        for (const auto &candidate : matches)
          spaces.insert(
              CanonicalSpace(core_.records_.at(candidate.first).value("space", "")));
        if (std::all_of(spaces.begin(), spaces.end(), [](const auto &s) {
              return s == "shared" || s == "cluster_shared";
            })) {
          if (spaces.size() != 1)
            throw SemanticError(
                "ambiguous_memory_address",
                "numeric operand ambiguous between CTA and cluster shared");
          return checked(core_.BindStep(*spaces.begin(), "physical.shared.cta.0", offset, bytes, {}),
                         bytes);
        }
      }
      if (matches.size() > 1)
        throw SemanticError("ambiguous_memory_address",
                            "numeric operand matches multiple allocations");
      if (matches.size() == 1) {
        const auto &[key, begin] = matches[0];
        const auto &r = core_.records_.at(key);
        if (r.contains("allocation") ||
            std::any_of(facts.launch().pointer_parameters().begin(),
                        facts.launch().pointer_parameters().end(),
                        [&key = key](const auto &p) { return p.resource == key; }))
          return checked(MemoryResource(key, offset - begin, bytes), bytes);
        return checked(core_.BindStep(r.value("space", "generic"), key, offset - begin, bytes, {}),
                       bytes);
      }
    }
  }
  if (id.empty()) {
    for (const auto &[key, r] : core_.records_) {
      if (r.value("kind", "") != "memory")
        continue;
      auto ad = r.value("address", Json::object());
      if (ad.value("symbol", "") == base) {
        if (!id.empty())
          throw SemanticError("ambiguous_memory_operand",
                              "operand names multiple resources");
        id = key;
      }
    }
  }
  if (id.empty()) {
    if (!core_.active_lane_) throw SemanticError("active_thread_context_missing", "memory operand has no active thread");
    std::optional<Json> value_expression;
    // Every bridge or extractor call precedes every step (see slot_resolver.h).
    if (config_.bridge) {
      fallback = FallbackUse::Bridge;
      value_expression = config_.bridge(*core_.active_lane_, e, text);
    } else {
      fallback = FallbackUse::Extractor;
      value_expression = ExtractAddress(e, text);
    }
    if (value_expression && value_expression->contains("allocation") && space == "global") {
      const auto resource = value_expression->at("allocation").get<std::string>();
      // Resolve aliases and explicit allocation placement using the same resource table.
      auto placement = MemoryResource(resource, 0, 0);
      // The base binding's fields, read from its step (which the stored binding equals).
      const auto binding = frame.Format(placement.key);
      const auto binding_offset = placement.offset;
      prefix.push_back({std::move(placement)});
      if (binding_offset == 0) {
        // Abstract over the base binding's formatted strings: the step binds the
        // formatted space and allocation name again, as a resource at offset 0.
        const auto &offset = value_expression->at("offset");
        auto step = core_.BindStep(binding.space, binding.allocation, 0, bytes, {});
        const auto stored = frame.Format(step.key);
        auto expression = Resources::AbstractExpression(stored.space, stored.allocation, stored.alias,
                                                        offset, bytes);
        return Footprint(FootprintKind::Abstract, std::move(step), std::move(expression), bytes);
      }
    }
    Json expression = {{"address", {{"base", nullptr}, {"view", nullptr},
        {"offset", {{"operator", "ptx_memory_operand"},
                    {"operands", Json::array({space, frame.Json(frame.self()), facts.Warp(*core_.active_lane_),
                        facts.LaneId(*core_.active_lane_), Json::array({e.instruction_id(),e.occurrence()}), text})}}}}},
        {"byte_count", bytes}};
    if (value_expression) expression["address"]["offset"] = value_expression->at("offset");
    auto serialized = expression.dump();
    // Braced: the name copy is taken before the expression is moved.
    return Footprint(FootprintKind::Unresolved, std::nullopt,
                     ExpressionStep{space.empty() ? "generic" : space, serialized, {},
                                    std::string("unresolved"), std::move(serialized)},
                     bytes);
  }
  if (!region.contains("byte_offset")) {
    auto ad = core_.records_.at(id).value("address", Json::object());
    if (ad.value("symbol", "") != base)
      throw SemanticError("unresolved_footprint",
                          "no exact resource byte offset");
    region["byte_offset"] = Unsigned(displacement, "symbol displacement");
  }
  Json extent = region.contains("bytes") ? region["bytes"]
                                         : contract.value("bytes", Json(bytes));
  if (!extent.is_number_integer() || extent.is_boolean())
    throw SemanticError("invalid_memory_extent",
                        "memory extent must be a positive integer");
  auto size = Unsigned(JsonInteger(extent), "memory extent");
  if (!size)
    throw SemanticError("invalid_memory_extent",
                        "memory extent must be positive");
  return checked(MemoryResource(
      id, Unsigned(JsonInteger(region["byte_offset"]), "region offset"), size), size);
}
const Json &SlotResolver::TensorMap(const RecordView &e,
                                    std::string_view operand,
                                    const Json &contract, const Json *dynamic_tensor_maps,
                                    std::string &descriptor) const {
  const auto &facts = core_.facts;
  auto text = Trim(operand);
  if (text.starts_with('['))
    text = Trim(std::string_view(text).substr(1));
  auto comma = text.find(',');
  if (comma != text.npos)
    text = Trim(std::string_view(text).substr(0, comma));
  if (text.ends_with(']'))
    text.pop_back();
  auto split = SplitAddress(text);
  auto base = split ? split->first : text;
  Integer displacement = split ? split->second : Integer(0);
  Scalar value = facts.launch().parameter_regions().contains(base)
                     ? Scalar(ParameterPointer{base, 0})
                     : core_.Value(e, base);
  const Json *match = nullptr;
  std::string matched;
  auto explicit_name = contract.value("descriptor", "");
  // The executor's runtime maps follow the program's own; other consumers have none.
  for (const auto *maps : {&facts.TensorMaps(), dynamic_tensor_maps}) {
  if (!maps) continue;
  for (auto it = maps->begin(); it != maps->end(); ++it) {
    const auto &d = it.value();
    bool yes = false;
    if (auto p = std::get_if<ParameterPointer>(&value)) {
      if (d.contains("parameter_location")) {
        auto location = d["parameter_location"];
        yes = location.is_array() && location.size() == 2 &&
              location[0] == p->region &&
              JsonInteger(location[1]) == p->offset + displacement;
      }
    } else if (auto p=std::get_if<AllocationPointer>(&value)) {
      if (d.contains("allocation_location")) {
        const auto &location=d["allocation_location"];
        yes=location.is_array() && location.size()==2 && location[0]==p->resource &&
            JsonInteger(location[1])==p->offset+displacement;
      }
    } else if (auto n = std::get_if<Integer>(&value))
      yes = d.contains("address") &&
            JsonInteger(d["address"]) == *n + displacement;
    if (yes) {
      if (match)
        throw SemanticError("missing_tma_descriptor",
                            "tensor operand matches multiple maps");
      match = &d;
      matched = it.key();
    }
  }
  }
  if (match) {
    if (!explicit_name.empty() && explicit_name != matched)
      throw SemanticError("missing_tma_descriptor",
                          "tensor operand conflicts with explicit descriptor");
    descriptor = std::move(matched);
    return *match;
  }
  if (!explicit_name.empty() &&
      facts.TensorMaps().contains(explicit_name)) {
    const auto &d = facts.TensorMaps().at(explicit_name);
    if (!d.contains("parameter_location") && !d.contains("address")) {
      descriptor = "explicit:" + explicit_name;
      return d;
    }
  }
  throw SemanticError("missing_tma_descriptor",
                      "tensor operand has no matching captured map");
}
ResolvedTma SlotResolver::TmaBody(const RecordView &e, const TmaRequest &request,
                                  std::vector<SlotStep> &steps, FallbackUse &fallback) {
  const auto &facts = core_.facts;
  const auto &frame = core_.frame;
  const auto &contract = request.contract;
  const auto store = request.store;
  const auto &i = facts.function().instructions[e.instruction_id()];
  const auto ops = OperandTexts(facts.module(), i);
  static const std::set<std::string> allowed{"async",
                                             "bulk",
                                             "tensor",
                                             "1d",
                                             "2d",
                                             "3d",
                                             "4d",
                                             "5d",
                                             "tile",
                                             "shared::cta",
                                             "shared::cluster",
                                             "global",
                                             "bulk_group",
                                             "mbarrier::complete_tx::bytes",
                                             "cta_group::1",
                                             "cta_group::2",
                                             "multicast::cluster",
                                             "L2::cache_hint"};
  std::set<std::string> seen;
  std::vector<std::string> spaces;
  std::uint32_t rank = 0, ranks = 0;
  for (const auto &p : i.modifiers) {
    if (!allowed.contains(p) || !seen.insert(p).second)
      throw SemanticError("tma_opcode_variant", "unsupported TMA modifier");
    if (p == "global" || p == "shared::cta" || p == "shared::cluster")
      spaces.push_back(p);
    if (p.size() == 2 && p[1] == 'd' && p[0] >= '1' && p[0] <= '5') {
      rank = p[0] - '0';
      ++ranks;
    }
  }
  const auto completion = store ? "bulk_group" : "mbarrier::complete_tx::bytes";
  const auto other = store ? "mbarrier::complete_tx::bytes" : "bulk_group";
  bool space_ok =
      store ? spaces == std::vector<std::string>{"global", "shared::cta"}
            : (spaces == std::vector<std::string>{"shared::cta", "global"} ||
               spaces == std::vector<std::string>{"shared::cluster", "global"});
  if (!space_ok || ranks != 1 || !Has(i, completion) || Has(i, other) ||
      !Has(i, "async") || !Has(i, "bulk") || !Has(i, "tensor") ||
      (Has(i, "cta_group::1") && Has(i, "cta_group::2")) ||
      (Has(i, "multicast::cluster") &&
       (store || spaces[0] != "shared::cluster")))
    throw SemanticError("tma_opcode_variant",
                        "unsupported TMA tiled copy variant");
  const std::size_t n = store ? 2 : 3;
  if (ops.size() != n + Has(i, "L2::cache_hint") + Has(i, "multicast::cluster"))
    throw SemanticError("tma_opcode_variant", "TMA operand count mismatch");
  for (std::size_t k = 0; k < n; ++k)
    if (!ops[k].starts_with('[') || !ops[k].ends_with(']'))
      throw SemanticError("tma_opcode_variant", "TMA needs bracket operands");
  const auto &tensor = ops[store ? 0 : 1];
  std::string descriptor_key;
  const auto descriptor = ResolveTensorMap(
      TensorMap(e, tensor, contract, request.dynamic_tensor_maps, descriptor_key), rank);
  auto shared = core_.Shared(e, ops[store ? 1 : 0]);
  if (!shared)
    throw SemanticError("unresolved_tma_shared_owner",
                        "TMA shared pointer is not concrete");
  if ((store || spaces[0] == "shared::cta") && shared->first != frame.self())
    throw SemanticError("tma_shared_cta_owner",
                        "local TMA shared pointer resolves to peer");
  std::vector<std::int64_t> origins;
  Json coordinate_expressions = Json::array();
  std::vector<bool> symbolic_coordinates;
  // Per coordinate, its affine form in tensor-property elements (bounded coordinates).
  std::vector<std::optional<AbstractAffine>> affine_coordinates;
  bool opaque_coordinate = false;
  auto a = tensor.find('{'), b = tensor.find('}', a);
  if (a == tensor.npos || b == tensor.npos)
    throw SemanticError("tma_opcode_rank", "tensor operand has no coordinates");
  std::istringstream values(tensor.substr(a + 1, b - a - 1));
  std::string coordinate;
  while (std::getline(values, coordinate, ',')) {
    const auto token = Trim(coordinate);
    const auto value = core_.Value(e, token);
    const auto known = ScalarInteger(value);
    symbolic_coordinates.push_back(!known);
    origins.push_back(known ? Signed(*known, "tensor coordinate") : 0);
    affine_coordinates.emplace_back();
    if (known) { coordinate_expressions.push_back(origins.back()); continue; }
    // A bounded coordinate (tensor_properties) in the s32 range is its exact affine form in
    // element lookups (the address model's gathers): no extraction.
    if (const auto *bounded = std::get_if<BoundedInteger>(&value);
        bounded && (*bounded)->affine && (*bounded)->lower >= INT32_MIN && (*bounded)->upper <= INT32_MAX)
      if (auto affine = AffineOffset(*(*bounded)->affine, 0)) {
        coordinate_expressions.push_back(AffineOffsetJson(*affine));
        affine_coordinates.back() = std::move(*affine);
        continue;
      }
    opaque_coordinate = true;
    if (!core_.active_lane_) throw SemanticError("active_thread_context_missing", "TMA coordinate has no thread");
    // Every extractor call precedes every step (see slot_resolver.h).
    fallback = FallbackUse::Extractor;
    auto expression = ExtractAddress(e, "[" + token + "]");
    // If extraction is unavailable, a fresh occurrence-scoped atom is a sound
    // relaxation. Never conflate independent threads or repeated dynamic reads.
    // Coordinates interpret register bits as signed s32. Keep that identity
    // separate from an unsigned/pointer use of the same underlying value.
    coordinate_expressions.push_back(expression ? Json{{"operator","opaque"},
        {"provenance",Json::array({"tma_signed_s32",expression->at("offset")})}} : Json{
        {"operator","opaque"}, {"provenance",Json::array({"tma_coordinate",frame.Json(frame.self()),
          facts.Warp(*core_.active_lane_),facts.LaneId(*core_.active_lane_),e.instruction_id(),e.occurrence(),token})}});
  }
  if (origins.size() != rank)
    throw SemanticError("tma_opcode_rank",
                        "opcode rank disagrees with coordinates");
  auto dims = frame.cluster_dim();
  TmaTransferOptions options;
  options.symbolic_coordinates = symbolic_coordinates;
  options.shared_base = shared->second;
  options.destination_rank = shared->first.value;
  options.cluster_size = dims[0] * dims[1] * dims[2];
  options.cta_group = Has(i, "cta_group::2") ? 2 : 1;
  options.store = store;
  std::optional<std::uint64_t> barrier_offset;
  if (!store) {
    auto bar = core_.Shared(e, ops[2]);
    if (!bar || bar->second % 8)
      throw SemanticError("invalid_barrier_address",
                          "TMA completion needs aligned concrete barrier");
    options.completion_rank = bar->first.value;
    barrier_offset = bar->second;
    if (options.cta_group == 2) {
      auto token = Trim(std::string_view(ops[2]).substr(1, ops[2].size() - 2));
      auto split = SplitAddress(token);
      auto value = core_.Value(e, split ? split->first : token);
      if (contract.contains("completion_cta_rank")) {
        auto ev = contract.value("completion_owner_evidence", Json::array());
        if (ev.empty() || !ev.is_array() ||
            std::any_of(ev.begin(), ev.end(), [](const auto &x) {
              return !x.is_string() ||
                     Trim(x.template get<std::string>()).empty();
            }))
          throw SemanticError("tma_group2_barrier_owner",
                              "completion owner requires source evidence");
        auto rank = Unsigned(JsonInteger(contract["completion_cta_rank"]),
                             "completion rank", UINT32_MAX);
        if (std::holds_alternative<ClusterSharedPointer>(value) &&
            *options.completion_rank != rank)
          throw SemanticError("tma_completion_owner_mismatch",
                              "owner contract conflicts with pointer");
        options.completion_rank = rank;
      } else if (!std::holds_alternative<ClusterSharedPointer>(value)) {
        const auto owner = core_.Owner(e, split ? split->first : token);
        if (!owner) throw SemanticError("tma_group2_barrier_owner",
            "group2 completion address lacks shared-pointer provenance");
        options.completion_rank = *owner;
      }
    }
    if (spaces[0] == "shared::cta" &&
        options.completion_rank != frame.self().value)
      throw SemanticError("tma_shared_cta_owner",
                          "local TMA barrier resolves to peer");
  }
  if (Has(i, "multicast::cluster"))
    options.multicast_mask = Unsigned(core_.ScalarValue(e, ops[3], "multicast mask") & 65535,
                                      "multicast mask", UINT32_MAX);
  if (contract.contains("transfer_bytes"))
    options.transfer_bytes =
        Unsigned(JsonInteger(contract["transfer_bytes"]), "transfer bytes");
  auto mapping = DecodeTmaTransfer(descriptor, origins, options);
  auto resource = descriptor.at("resource").get<std::string>();
  ResolvedTma result;
  result.store = store;
  result.descriptor = std::move(descriptor_key);
  // Every symbolic coordinate affine: the region's first byte, base + sum of coordinate x
  // stride, as one affine form in element lookups (the class log's δ form, query_log.cpp).
  if (!opaque_coordinate && std::any_of(affine_coordinates.begin(), affine_coordinates.end(),
                                        [](const auto &c) { return c.has_value(); })) {
    const auto strides = descriptor.value("coordinate_strides_bytes", Json());
    const auto base = descriptor.value("base_byte_offset", Json(0));
    if (strides.is_array() && strides.size() == origins.size() && base.is_number_integer()) {
      Integer constant = JsonInteger(base);
      std::vector<AbstractAffine::Term> terms;
      bool exact = true;
      for (std::size_t dim = 0; exact && dim < origins.size(); ++dim) {
        if (!strides[dim].is_number_integer()) { exact = false; break; }
        const Integer stride = JsonInteger(strides[dim]);
        if (!affine_coordinates[dim]) { constant += Integer(origins[dim]) * stride; continue; }
        constant += Integer(affine_coordinates[dim]->constant) * stride;
        for (const auto &term : affine_coordinates[dim]->terms) {
          const auto coefficient = Int64(Integer(term.coefficient) * stride);
          if (!coefficient) { exact = false; break; }
          terms.push_back({term.object, term.element_type, term.offset, *coefficient});
        }
      }
      if (const auto fitted = exact ? Int64(constant) : std::nullopt)
        result.region_affine = AbstractAffine{*fitted, std::move(terms)};
    }
  }
  result.pairs.reserve(mapping.segments.size());
  const bool known_resource = resource != "unresolved";
  // The exact step count: an abstract_global segment appends its expression,
  // the offset-0 placement when the resource is known, and its patch; a global_offset
  // segment its placement; every segment its shared binding; then the completions.
  std::size_t step_count = mapping.completions.size();
  for (const auto &s : mapping.segments)
    step_count += 1 + (s.abstract_global ? 2 + std::size_t(known_resource) : s.global_offset ? 1 : 0);
  steps.reserve(step_count);
  for (const auto &s : mapping.segments) {
    std::optional<std::size_t> global;
    if (s.abstract_global) {
      Json expression={{"operator","tma_region"},{"descriptor",descriptor},
          {"coordinates",coordinate_expressions},{"logical_offset",s.logical_offset},{"bytes",s.bytes},
          {"occurrence",Json::array({frame.Json(frame.self()),facts.Warp(*core_.active_lane_),facts.LaneId(*core_.active_lane_),e.instruction_id(),e.occurrence()})}};
      auto serialized=expression.dump();
      const auto address=steps.size();
      steps.push_back({ExpressionStep{"global",serialized,{},std::string("unresolved"),{}}});
      // A dynamic origin does not erase a descriptor's known global allocation.
      // MemoryResource may throw here and leave the binding half-patched.
      PatchStep patch{address, {}, {}, std::move(serialized)};
      if (known_resource) {
        auto placement = MemoryResource(resource, 0, 0);
        // The placement binding's allocation and offset, from its step (which the
        // stored binding equals).
        patch.allocation = placement.key;
        patch.offset = placement.offset;
        steps.push_back({std::move(placement)});
      }
      steps.push_back({std::move(patch)});
      global=address;
    } else if (s.global_offset) {
      auto placement = MemoryResource(resource, *s.global_offset, s.bytes);
      global = steps.size();
      steps.push_back({std::move(placement)});
    }
    auto sh = core_.BindStep("shared", "physical.shared.cta.0", s.shared_offset, s.bytes,
                             frame.Rank(s.target_rank));
    result.pairs.push_back({global, steps.size(), s.bytes, s.logical_offset, s.target_rank});
    steps.push_back({std::move(sh)});
  }
  for (const auto &c : mapping.completions) {
    if (!barrier_offset)
      throw SemanticError("tma_completion", "completion has no barrier");
    BarrierStep step{{BarrierKey::Kind::Physical, frame.Rank(c.rank), std::to_string(*barrier_offset)}};
    result.completions.emplace_back(steps.size(), c.bytes);
    steps.push_back({std::move(step)});
  }
  return result;
}
} // namespace sentryward::frontend::ptx
