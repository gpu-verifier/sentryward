#include "sentryward/frontend/ptx/resources.h"
#include "sentryward/frontend/ptx/layouts.h"
#include "sentryward/frontend/ptx/tensor_maps.h"
#include "sentryward/frontend/ptx/value_ir.h"
#include <algorithm>
#include <regex>
#include <sstream>
namespace sentryward::frontend::ptx {
using Json = nlohmann::json;
std::string Trim(std::string_view x) {
  auto a = x.find_first_not_of(" \t\r\n"), b = x.find_last_not_of(" \t\r\n");
  return a == x.npos ? "" : std::string(x.substr(a, b - a + 1));
}
std::string CoordinateText(Coordinates c) {
  return "(" + std::to_string(c[0]) + ", " + std::to_string(c[1]) + ", " +
         std::to_string(c[2]) + ")";
}
std::string Opcode(const Instruction &i) {
  std::string x = i.opcode;
  for (const auto &m : i.modifiers)
    x += '.' + m;
  return x;
}
std::vector<std::string> OperandTexts(const Module &m, const Instruction &i) {
  std::vector<std::string> r;
  for (auto v : i.operands)
    r.emplace_back(m.view(v));
  return r;
}
bool Has(const Instruction &i, std::string_view m) {
  return std::find(i.modifiers.begin(), i.modifiers.end(), m) !=
         i.modifiers.end();
}
std::string CanonicalSpace(std::string s) {
  if (s == "shared::cta" || s == "shared.cta")
    return "shared";
  if (s == "shared::cluster" || s == "shared.cluster")
    return "cluster_shared";
  return s;
}
std::uint64_t Unsigned(const Integer &x, std::string_view field,
                       std::uint64_t max) {
  if (x < 0 || x > max)
    throw SemanticError("integer_range",
                        std::string(field) + " outside unsigned range");
  return x.convert_to<std::uint64_t>();
}
std::int64_t Signed(const Integer &x, std::string_view field) {
  if (x < INT64_MIN || x > INT64_MAX)
    throw SemanticError("integer_range",
                        std::string(field) + " outside signed range");
  return x.convert_to<std::int64_t>();
}
AddressId AddressTable::Intern(std::string space, std::string allocation,
                               std::uint64_t offset, std::string alias_scope) {
  auto [name, inserted] =
      names_.try_emplace({space, allocation}, names_.size());
  (void)inserted;
  auto [it, added] =
      ids_.try_emplace({name->second, offset}, AddressId(bindings.size()));
  if (added)
    bindings.push_back({it->second, std::move(space), std::move(allocation),
                        offset, std::move(alias_scope)});
  else if (bindings[it->second.value()].alias_scope != alias_scope)
    throw SemanticError("alias_scope_conflict",
                        "inconsistent allocation alias domain");
  return it->second;
}
BarrierLifetimeId BarrierTable::Intern(std::string kind, Coordinates owner,
                                       std::string id) {
  return ids_
      .try_emplace({std::move(kind), owner, std::move(id)},
                   BarrierLifetimeId(ids_.size()))
      .first->second;
}
Resources::Resources(const ExecutionProgram &p, Coordinates c, AddressTable &a,
                     BarrierTable &b)
    : program(p), coordinates(c), addresses(a), barriers(b) {
  for (const auto &r : p.config.resources) {
    auto id = r.at("id").get<std::string>();
    records_.emplace(id, r);
    auto ad = r.value("address", Json::object());
    auto space = CanonicalSpace(r.value("space", ""));
    if ((r.value("kind", "") == "mbarrier" || space == "shared" ||
         space == "cluster_shared") &&
        ad.contains("symbol"))
      shared_symbols_.insert(ad["symbol"].get<std::string>());
  }
  static const std::regex pattern(
      R"(\.(?:[busf]\d+|bf16)\s+([A-Za-z_$][A-Za-z0-9_$]*))");
  auto add = [&](const auto &ds) {
    for (const auto &d : ds) {
      auto text = std::string(p.module->view(d.source));
      if (text.find(".shared") == text.npos)
        continue;
      std::smatch match;
      if (std::regex_search(text, match, pattern))
        shared_symbols_.insert(match[1]);
    }
  };
  add(p.function->directives);
  add(p.module->directives);
  for (const auto& local : LocalMemoryDeclarations(*p.module, *p.function))
    records_.emplace(local.resource, Json{{"id",local.resource},{"kind","memory"},{"space","local"},
        {"allocation",{{"resolution","concrete"},{"base_byte_offset",0},{"object",local.resource}}}});
}
Resources::~Resources() = default;
std::tuple<std::string, Coordinates, std::string> BarrierTable::Location(BarrierLifetimeId id) const {
  for (const auto &[key,value] : ids_) if (value==id) return key;
  throw SemanticError("shared_control_operand", "unknown barrier identity");
}
void Resources::SetLane(const LaneTrace &lane) {
  active_lane_ = &lane;
  active_index_ = &lane_indices_[&lane];
}
const ExecutedInstruction *Resources::Definition(DynamicKey key, bool address_slice) const {
  if (!active_lane_) return {};
  auto &index = *active_index_;
  const auto &records = active_lane_->instructions;
  auto &positions=occurrence_indices_[records.metadataIdentity()];
  if (positions.indexed > records.size()) {positions = {};index = {};}
  for (; positions.indexed < records.size(); ++positions.indexed)
    positions.executions[records.metadata(positions.indexed).Key()] = positions.indexed;
  const auto found = positions.executions.find(key);
  if (found == positions.executions.end()) return {};
  if (records.materialized()) return &records.at(found->second);
  if (address_slice) {
    auto [entry, inserted] = index.address_requested.try_emplace(key);
    if (inserted) {
      if (address_register_ids_.empty())
        for (std::uint32_t id = 0; id < program.register_names.size(); ++id)
          address_register_ids_.emplace(program.register_names[id], id);
      auto [registers, added] = address_instruction_registers_.try_emplace(key.instruction);
      if (added)
        for (const auto& name : program.function->instructions.at(key.instruction).registers)
          if (const auto reg = address_register_ids_.find(name); reg != address_register_ids_.end())
            registers->second.push_back(reg->second);
      entry->second = records.readRegisters(found->second, registers->second);
    }
    return &entry->second;
  }
  auto [entry, inserted] = index.requested.try_emplace(key);
  if (inserted) entry->second = records.read(found->second);
  return &entry->second;
}
Scalar Resources::Value(const ExecutedInstruction &e,
                        std::string_view name) const {
  if (auto n = ParseInteger(name))
    return *n;
  if (auto p = program.KnownRegister(e, name))
    return *p;
  auto it = program.config.parameters.find(std::string(name));
  if (it != program.config.parameters.end())
    return it->second;
  return {};
}
Integer Resources::ScalarValue(const ExecutedInstruction &e,
                               std::string_view text,
                               std::string_view field) const {
  auto v = ScalarInteger(Value(e, Trim(text)));
  if (!v)
    throw SemanticError(field == "tensor coordinate" ? "tma_coordinate_not_static" :
                        field == "source size/ignore source" ? "async_copy_source_control_not_static" : "scalar_operand_not_static",
                        "Operand must have a statically determined integer value",
                        {{"operand_role",field},{"operand",text}});
  return *v;
}
Coordinates Resources::Target(std::uint32_t rank) const {
  auto d = program.config.cluster_dim.value_or(Coordinates{1, 1, 1});
  if (rank >= std::uint64_t(d[0]) * d[1] * d[2])
    throw SemanticError("invalid_cluster_rank", "cluster rank outside launch");
  Coordinates local{rank % d[0], (rank / d[0]) % d[1], rank / (d[0] * d[1])},
      result;
  for (int i = 0; i < 3; ++i)
    result[i] = coordinates[i] - coordinates[i] % d[i] + local[i];
  return result;
}
std::uint32_t Resources::Rank(Coordinates c) const {
  auto d = program.config.cluster_dim.value_or(Coordinates{1, 1, 1});
  Coordinates local;
  for (int i = 0; i < 3; ++i) {
    auto origin = coordinates[i] - coordinates[i] % d[i];
    if (c[i] < origin || c[i] >= origin + d[i])
      throw SemanticError("cluster_owner", "owner outside cluster");
    local[i] = c[i] - origin;
  }
  return local[0] + d[0] * (local[1] + d[1] * local[2]);
}
std::optional<std::uint32_t> Resources::Owner(const ExecutedInstruction &e,
                                              std::string_view operand,
                                              std::set<DynamicKey> visiting) {
  auto name = Trim(operand);
  auto v = Value(e, name);
  if (auto p = std::get_if<ClusterSharedPointer>(&v))
    return p->target_rank;
  if (shared_symbols_.contains(name))
    return Rank(coordinates);
  auto origin = program.RegisterOrigin(e, name);
  if (!origin || visiting.contains(*origin))
    return {};
  if (!active_index_)
    return {};
  auto &owners = active_index_->owners;
  if (auto it = owners.find(*origin); it != owners.end())
    return it->second;
  auto definition = Definition(*origin);
  if (!definition)
    return {};
  visiting.insert(*origin);
  const auto &def = *definition;
  const auto &i = program.function->instructions[def.instruction_id()];
  auto ops = OperandTexts(*program.module, i);
  std::optional<std::uint32_t> result;
  if ((i.opcode == "mov" || i.opcode == "cvt" || i.opcode == "cvta") &&
      ops.size() == 2)
    result = Owner(def, ops[1], visiting);
  else if ((i.opcode == "add" || i.opcode == "sub") && ops.size() == 3) {
    auto l = Owner(def, ops[1], visiting), r = Owner(def, ops[2], visiting);
    if (l && !r)
      result = l;
    else if (i.opcode == "add" && !l)
      result = r;
  } else if (i.opcode == "selp" && ops.size() == 4) {
    auto pred = ScalarInteger(Value(def, ops[3]));
    if (pred)
      result = Owner(def, ops[*pred != 0 ? 1 : 2], visiting);
  } else if (i.opcode == "and" &&
             i.modifiers == std::vector<std::string>{"b32"} &&
             ops.size() == 3) {
    for (int k = 1; k <= 2; ++k) {
      auto mask = ScalarInteger(Value(def, ops[3 - k]));
      if (!mask)
        continue;
      auto owner = Owner(def, ops[k], visiting);
      if (!owner)
        continue;
      auto n = (*mask & 0xffffffff).convert_to<std::uint32_t>();
      if ((n & 0xff000000) == 0xff000000)
        result = owner;
      else if ((n & 0xff000000) == 0xfe000000)
        result = *owner & ~1u;
    }
  } else if (i.opcode=="shfl" && i.modifiers==std::vector<std::string>{"sync","idx","b32"} && ops.size()==5) {
    auto lane=ScalarInteger(Value(def,ops[2])), clamp=ScalarInteger(Value(def,ops[3])), mask=ScalarInteger(Value(def,ops[4]));
    if (lane && *lane>=0 && *lane<32 && clamp==Integer(31) && mask && ((*mask&0xffffffff)==0xffffffff)) {
      if (active_lane_ && lane == Integer(active_lane_->lane_id)) {
        result = Owner(def, ops[1], visiting);
        owners[*origin] = result;
        return result;
      }
      const auto *saved_lane=active_lane_; auto *saved_index=active_index_;
      for (const auto &trace : program.lane_traces)
        if (trace.warp_id==saved_lane->warp_id && trace.lane_id==lane->convert_to<std::uint32_t>()) {
          SetLane(trace);
          if (auto source=Definition(*origin))
            result=Owner(*source,ops[1],visiting);
          break;
        }
      active_lane_=saved_lane; active_index_=saved_index;
    }
  }
  owners[*origin] = result;
  return result;
}
std::optional<std::uint32_t> Resources::GenericOwner(const ExecutedInstruction &e,
    std::string_view operand, std::set<DynamicKey> visiting) {
  auto origin=program.RegisterOrigin(e,Trim(operand));
  if (!origin || !active_index_ || !visiting.insert(*origin).second) return {};
  auto &cache=active_index_->generic_owners;
  if (auto known=cache.find(*origin);known!=cache.end()) return known->second;
  auto found=Definition(*origin); if (!found) return {};
  const auto &def=*found; const auto &i=program.function->instructions[def.instruction_id()];
  const auto ops=OperandTexts(*program.module,i);
  auto owner=[&](std::string_view token){return GenericOwner(def,token,visiting);};
  auto scalar=[&](std::string_view token){return ScalarInteger(Value(def,token));};
  std::optional<std::uint32_t> result;
  if (i.opcode=="cvta" && Has(i,"shared") && !Has(i,"to") && ops.size()==2) {
    result=Owner(def,ops[1]); if (!result && scalar(ops[1])) result=Rank(coordinates);
  } else if ((i.opcode=="mov" || i.opcode=="cvt") && ops.size()==2 &&
      (Has(i,"b64") || Has(i,"u64") || Has(i,"s64"))) result=owner(ops[1]);
  else if ((i.opcode=="add" || i.opcode=="sub" || i.opcode=="xor" || i.opcode=="or") && ops.size()==3) {
    const auto left=owner(ops[1]), right=owner(ops[2]);
    if (left.has_value()!=right.has_value()) {
      auto displacement=scalar(ops[left?2:1]);
      if (displacement && (left || i.opcode!="sub") &&
          ((i.opcode!="xor" && i.opcode!="or") || (*displacement>=0 && *displacement<(1<<24)))) result=left?left:right;
    }
  } else if (i.opcode=="selp" && ops.size()==4) {
    if (auto predicate=scalar(ops[3])) result=owner(ops[*predicate!=0?1:2]);
  }
  cache[*origin]=result; return result;
}
Integer Resources::AddressRegisterBits(const ExecutedInstruction &e,
                                      std::string_view token, const Integer &value) {
  // Execution keeps congruent unbounded representatives. Both event lowering
  // and shared-control execution must interpret declared address register bits
  // after displacement, before range checks, without truncating literal addresses.
  Integer bits = value;
  if (bits > UINT32_MAX && !ParseInteger(token)) {
    if (!address_register_types_)
      address_register_types_ = std::make_unique<RegisterTable>(*program.module, *program.function);
    const auto &instruction = program.function->instructions.at(e.instruction_id());
    const auto identity = address_register_types_->Resolve(instruction.scope, token);
    if (auto bytes = address_register_types_->DeclaredWidth(identity); bytes && (*bytes == 4 || *bytes == 8))
      bits &= (Integer(1) << (8 * *bytes)) - 1;
  }
  return bits;
}
std::optional<SharedAddress> Resources::Shared(const ExecutedInstruction &e,
                                               std::string_view text) {
  auto token = Trim(text);
  if (token.starts_with('[') && token.ends_with(']'))
    token = Trim(std::string_view(token).substr(1, token.size() - 2));
  Integer displacement = 0;
  Scalar value;
  auto parsed = ParseInteger(token);
  if (parsed)
    value = *parsed;
  else {
    auto split = SplitAddress(token);
    if (!split)
      return {};
    token = split->first;
    displacement = split->second;
    value = Value(e, token);
  }
  if (std::holds_alternative<std::monostate>(value)) {
    std::set<Integer> placements;
    for (const auto &[id, r] : records_) {
      (void)id;
      auto ad = r.value("address", Json::object());
      auto s = CanonicalSpace(r.value("space", ""));
      if ((r.value("kind", "") == "mbarrier" || s == "shared" ||
           s == "cluster_shared") &&
          ad.value("symbol", "") == token && ad.contains("value"))
        placements.insert(JsonInteger(ad["value"]));
    }
    if (placements.size() > 1)
      throw SemanticError("ambiguous_shared_symbol",
                          "conflicting shared placements");
    if (!placements.empty())
      value = *placements.begin();
  }
  if (auto n = std::get_if<Integer>(&value)) {
    const auto bits = AddressRegisterBits(e, token, *n + displacement);
    auto owner = Owner(e, token);
    return SharedAddress{
        owner ? Target(*owner) : coordinates,
        Unsigned(bits, "shared address", UINT32_MAX)};
  }
  if (auto p = std::get_if<ClusterSharedPointer>(&value))
    return SharedAddress{Target(p->target_rank),
                         Unsigned(p->local_address + displacement,
                                  "shared address", UINT32_MAX)};
  return {};
}
BarrierLifetimeId Resources::Barrier(const ExecutedInstruction &e,
                                     std::string_view operand,
                                     const Json &contract) {
  if (auto shared = Shared(e, operand)) {
    if (shared->second % 8)
      throw SemanticError("invalid_barrier_address",
                          "mbarrier address must be aligned to eight bytes");
    return barriers.Intern("physical", shared->first,
                           std::to_string(shared->second));
  }
  std::set<std::string> matches;
  auto configured = contract.value("resource", "");
  for (const auto &[id, r] : records_)
    if (r.value("kind", "") == "mbarrier") {
      if (configured == id) {
        matches = {id};
        break;
      }
      auto ad = r.value("address", Json::object());
      if (ad.contains("symbol")) {
        auto split = SplitAddress(Trim(operand.substr(1, operand.size() - 2)));
        if (split && split->first == ad["symbol"].get<std::string>())
          matches.insert(id);
      }
    }
  if (matches.size() != 1)
    throw SemanticError("unresolved_barrier",
                        "cannot resolve mbarrier operand " +
                            std::string(operand));
  return barriers.Intern("symbolic", coordinates, *matches.begin());
}
Fragment Resources::Bind(std::string space, std::string resource,
                         std::uint64_t offset, std::uint64_t bytes,
                         std::optional<Coordinates> owner) {
  space = CanonicalSpace(space);
  std::string alias;
  auto c = CoordinateText(owner.value_or(coordinates));
  if (space == "shared" || space == "cluster_shared") {
    if (resource != "physical.shared.cta.0")
      throw SemanticError("unresolved_shared_placement",
                          "shared allocation has no physical placement");
    resource = "cta:" + c + ":shared:" + resource;
    space = "shared";
  } else if (space == "tensor") {
    resource = "cta:" + c + ":tensor:" + resource;
    alias = "cta:" + c + ":tensor";
  } else if (space == "local") {
    if (!active_lane_) throw SemanticError("active_thread_context_missing", "local access has no active thread");
    resource = "cta:" + c + ":thread:" + std::to_string(active_lane_->warp_id * 32 + active_lane_->lane_id) + ':' + resource;
  } else if (space == "generic")
    throw SemanticError("unresolved_footprint",
                        "memory allocation has no exact owner");
  if (bytes > UINT64_MAX - offset)
    throw SemanticError("invalid_memory_range", "memory range overflows u64");
  return {addresses.Intern(std::move(space), std::move(resource), offset,
                           std::move(alias)),
          bytes};
}
Fragment Resources::Abstract(std::string space, std::string resource,
                             const Json &offset, std::optional<std::uint64_t> bytes,
                             std::optional<Coordinates> owner) {
  auto base = Bind(space, resource, 0, bytes.value_or(0), owner);
  const auto binding = addresses.bindings.at(base.first.value());
  Json expression = {{"address", {{"base", {{"identity", binding.allocation},
      {"space", binding.address_space}}}, {"offset", offset}, {"view", nullptr}}},
      {"byte_count", bytes ? Json(*bytes) : Json(nullptr)}};
  const auto serialized = expression.dump();
  auto address = addresses.Intern(binding.address_space, "expression:" + serialized,
                                   0, binding.alias_scope);
  auto &result = addresses.bindings.at(address.value());
  result.allocation = binding.allocation;
  result.expression = serialized;
  return {address, bytes};
}
Fragment Resources::MemoryResource(std::string id, std::uint64_t offset,
                                   std::uint64_t bytes) {
  if(const auto cached=memory_resources_.find(id);cached!=memory_resources_.end()) {
    const auto& info=cached->second;
    if(info.extent && (offset>*info.extent || bytes>*info.extent-offset))
      throw SemanticError("pointer_extent", "access exceeds captured pointer allocation");
    if(offset>UINT64_MAX-info.base)
      throw SemanticError("invalid_memory_range", "allocation offset overflows");
    return Bind(info.space,info.object,offset+info.base,bytes);
  }
  auto it = records_.find(id);
  if(it==records_.end()) {
    // Value address provenance uses allocation identity; execution contracts
    // may use a distinct resource handle. Resolve that alias once, retaining
    // the original resource's extent validation. Never choose ambiguously or
    // reinterpret a nonzero suballocation pointer as the allocation base.
    std::optional<std::string> alias;
    for(const auto& pointer:program.config.pointer_parameters) {
      const auto candidate=records_.find(pointer.resource);
      if(candidate==records_.end() || !candidate->second.contains("allocation"))continue;
      const auto& allocation=candidate->second.at("allocation");
      if(allocation.value("resolution",std::string{})!="concrete" ||
          allocation.value("object",std::string{})!=id ||
          allocation.value("base_byte_offset",Json(0))!=Json(0))continue;
      if(alias && *alias!=pointer.resource)
        throw SemanticError("ambiguous_memory_resource","Multiple pointer resources name allocation "+id);
      alias=pointer.resource;
    }
    if(alias) {
      const auto result=MemoryResource(*alias,offset,bytes);
      memory_resources_.emplace(id,memory_resources_.at(*alias));
      return result;
    }
  }
  if (it == records_.end() || it->second.value("kind", "") != "memory")
    throw SemanticError("unknown_memory_resource",
                        "unknown memory resource " + id);
  const auto &r = it->second;
  bool pointer = false;
  std::optional<std::uint64_t> extent;
  for (const auto &p : program.config.pointer_parameters)
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
  const auto result=Bind(space, object, offset, bytes);
  memory_resources_.emplace(std::move(id),MemoryResourceInfo{space,std::move(object),base,extent});
  return result;
}
Fragment Resources::Memory(const ExecutedInstruction &e, std::string_view text,
                           const Json &contract, std::uint64_t bytes,
                           std::string space) {
  space = CanonicalSpace(space);
  if (space == "shared" || space == "cluster_shared")
    if (auto a = Shared(e, text)) {
      if (space == "shared" && a->first != coordinates)
        throw SemanticError("shared_cta_owner",
                            "CTA local address points to another CTA");
      if (a->second + bytes > (1ull << 32))
        throw SemanticError("invalid_shared_range", "shared range outside u32");
      return Bind("shared", "physical.shared.cta.0", a->second, bytes,
                  a->first);
    }
  auto token = Trim(text);
  if (token.starts_with('[') && token.ends_with(']'))
    token = Trim(std::string_view(token).substr(1, token.size() - 2));
  auto split = SplitAddress(token);
  std::string base = split ? split->first : token;
  Integer displacement = split ? split->second : Integer(0);
  auto v = Value(e, base);
  if (space.empty() || space=="generic")
    if (auto owner=GenericOwner(e,base))
      if (auto value=ScalarInteger(v)) {
        const auto offset=Unsigned(*value+displacement,"shared generic address",UINT32_MAX);
        if (bytes>(1ULL<<32)-offset) throw SemanticError("invalid_shared_range","shared range outside u32");
        return Bind("shared","physical.shared.cta.0",offset,bytes,Target(*owner));
      }
  auto region = contract.value("region", Json::object());
  std::string id = contract.value("resource", "");
  auto checked = [&](Fragment f) {
    auto got = addresses.bindings[f.first.value()].address_space;
    auto want = space == "cluster_shared" ? "shared" : space;
    if (!space.empty() && space != "generic" && got != want)
      throw SemanticError("memory_space_mismatch",
                          "resource space disagrees with instruction");
    return f;
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
      return checked(MemoryResource(p->resource, offset, bytes));
    }
    if (auto n = std::get_if<Integer>(&v); n && !ParseInteger(base)) {
      auto offset = Unsigned(*n + displacement, "numeric memory address");
      std::vector<std::pair<std::string, std::uint64_t>> matches;
      for (const auto &[key, r] : records_) {
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
              CanonicalSpace(records_.at(candidate.first).value("space", "")));
        if (std::all_of(spaces.begin(), spaces.end(), [](const auto &s) {
              return s == "shared" || s == "cluster_shared";
            })) {
          if (spaces.size() != 1)
            throw SemanticError(
                "ambiguous_memory_address",
                "numeric operand ambiguous between CTA and cluster shared");
          return checked(
              Bind(*spaces.begin(), "physical.shared.cta.0", offset, bytes));
        }
      }
      if (matches.size() > 1)
        throw SemanticError("ambiguous_memory_address",
                            "numeric operand matches multiple allocations");
      if (matches.size() == 1) {
        const auto &[key, begin] = matches[0];
        const auto &r = records_.at(key);
        if (r.contains("allocation") ||
            std::any_of(program.config.pointer_parameters.begin(),
                        program.config.pointer_parameters.end(),
                        [&key = key](const auto &p) { return p.resource == key; }))
          return checked(MemoryResource(key, offset - begin, bytes));
        return checked(
            Bind(r.value("space", "generic"), key, offset - begin, bytes));
      }
    }
  }
  if (id.empty()) {
    for (const auto &[key, r] : records_) {
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
    if (!active_lane_) throw SemanticError("active_thread_context_missing", "memory operand has no active thread");
    std::optional<Json> value_expression;
    if (value_addresses) value_expression = value_addresses(*active_lane_, e, text);
    else {
    if (!address_values_) address_values_ = std::make_unique<AddressValueExtractor>(program, coordinates);
    value_expression = address_values_->Extract(*active_lane_, e, text,
        [this](const LaneTrace& lane, DynamicKey key) {
          const auto* previous = active_lane_;
          SetLane(lane);
          const auto* result = Definition(key, true);
          SetLane(*previous);
          return result;
        });
    }
    if (value_expression && value_expression->contains("allocation") && space == "global") {
      const auto resource = value_expression->at("allocation").get<std::string>();
      // Resolve aliases and explicit allocation placement using the same resource table.
      const auto base_fragment = MemoryResource(resource, 0, 0);
      const auto binding = addresses.bindings.at(base_fragment.first.value());
      if (binding.offset == 0)
        return Abstract(binding.address_space, binding.allocation, value_expression->at("offset"), bytes);
    }
    Json expression = {{"address", {{"base", nullptr}, {"view", nullptr},
        {"offset", {{"operator", "ptx_memory_operand"},
                    {"operands", Json::array({space, coordinates, active_lane_->warp_id,
                        active_lane_->lane_id, Json::array({e.instruction_id(),e.occurrence()}), text})}}}}},
        {"byte_count", bytes}};
    if (value_expression) expression["address"]["offset"] = value_expression->at("offset");
    const auto serialized = expression.dump();
    auto address = addresses.Intern(space.empty() ? "generic" : space, "expression:" + serialized, 0);
    addresses.bindings[address.value()].allocation = "unresolved";
    addresses.bindings[address.value()].expression = serialized;
    return {address, bytes};
  }
  if (!region.contains("byte_offset")) {
    auto ad = records_.at(id).value("address", Json::object());
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
      id, Unsigned(JsonInteger(region["byte_offset"]), "region offset"), size));
}
void Resources::AdmitGrid(const Instruction &i) const {
  const auto &f = program.config.fixed_input;
  auto launch = f.value("launch_contract", Json::object());
  auto ev = launch.value("source_evidence", Json::array());
  if (i.opcode != "griddepcontrol" || !i.operands.empty() ||
      (i.modifiers != std::vector<std::string>{"wait"} &&
       i.modifiers != std::vector<std::string>{"launch_dependents"}))
    throw SemanticError("grid_dependency_variant",
                        "unknown grid dependency instruction");
  if (f.value("public_input_schema", "") == "spverify.kernel-input/v1" &&
      launch.value("external_grids", "") == "no_overlap") return;
  if (launch.value("kind", "") != "ordinary" ||
      launch.value("programmatic_dependent_launch", true) ||
      launch.value("external_grids", "") != "no_overlap" || !ev.is_array() ||
      ev.empty() || std::any_of(ev.begin(), ev.end(), [](const auto &v) {
        return !v.is_string() || Trim(v.template get<std::string>()).empty();
      }))
    throw SemanticError(
        "grid_dependency_launch_contract",
        "requires explicit ordinary launch with no overlapping external grids");
}
const Json &Resources::TensorMap(const ExecutedInstruction &e,
                                 std::string_view operand,
                                 const Json &contract) const {
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
  Scalar value = program.config.parameter_regions.contains(base)
                     ? Scalar(ParameterPointer{base, 0})
                     : Value(e, base);
  const Json *match = nullptr;
  std::string matched;
  auto explicit_name = contract.value("descriptor", "");
  for (const auto *maps : {&program.config.tma_descriptors,static_cast<const Json *>(&dynamic_tensor_maps)})
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
  if (match) {
    if (!explicit_name.empty() && explicit_name != matched)
      throw SemanticError("missing_tma_descriptor",
                          "tensor operand conflicts with explicit descriptor");
    return *match;
  }
  if (!explicit_name.empty() &&
      program.config.tma_descriptors.contains(explicit_name)) {
    const auto &d = program.config.tma_descriptors.at(explicit_name);
    if (!d.contains("parameter_location") && !d.contains("address"))
      return d;
  }
  throw SemanticError("missing_tma_descriptor",
                      "tensor operand has no matching captured map");
}
TmaAccesses Resources::Tma(const ExecutedInstruction &e, const Json &contract,
                           bool store) {
  const auto &i = program.function->instructions[e.instruction_id()];
  const auto ops = OperandTexts(*program.module, i);
  std::set<std::string> allowed{"async",
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
  const auto descriptor = ResolveTensorMap(TensorMap(e, tensor, contract),rank);
  auto shared = Shared(e, ops[store ? 1 : 0]);
  if (!shared)
    throw SemanticError("unresolved_tma_shared_owner",
                        "TMA shared pointer is not concrete");
  if ((store || spaces[0] == "shared::cta") && shared->first != coordinates)
    throw SemanticError("tma_shared_cta_owner",
                        "local TMA shared pointer resolves to peer");
  std::vector<std::int64_t> origins;
  Json coordinate_expressions = Json::array();
  std::vector<bool> symbolic_coordinates;
  auto a = tensor.find('{'), b = tensor.find('}', a);
  if (a == tensor.npos || b == tensor.npos)
    throw SemanticError("tma_opcode_rank", "tensor operand has no coordinates");
  std::istringstream values(tensor.substr(a + 1, b - a - 1));
  std::string coordinate;
  while (std::getline(values, coordinate, ',')) {
    const auto token = Trim(coordinate);
    const auto known = ScalarInteger(Value(e, token));
    symbolic_coordinates.push_back(!known);
    origins.push_back(known ? Signed(*known, "tensor coordinate") : 0);
    if (known) { coordinate_expressions.push_back(origins.back()); continue; }
    if (!active_lane_) throw SemanticError("active_thread_context_missing", "TMA coordinate has no thread");
    if (!address_values_) address_values_ = std::make_unique<AddressValueExtractor>(program, coordinates);
    auto expression = address_values_->Extract(*active_lane_, e, "[" + token + "]",
        [this](const LaneTrace& lane, DynamicKey key) {
          const auto* previous = active_lane_; SetLane(lane);
          const auto* result = Definition(key, true); SetLane(*previous); return result;
        });
    // If extraction is unavailable, a fresh occurrence-scoped atom is a sound
    // relaxation. Never conflate independent threads or repeated dynamic reads.
    // Coordinates interpret register bits as signed s32. Keep that identity
    // separate from an unsigned/pointer use of the same underlying value.
    coordinate_expressions.push_back(expression ? Json{{"operator","opaque"},
        {"provenance",Json::array({"tma_signed_s32",expression->at("offset")})}} : Json{
        {"operator","opaque"}, {"provenance",Json::array({"tma_coordinate",coordinates,
          active_lane_->warp_id,active_lane_->lane_id,e.instruction_id(),e.occurrence(),token})}});
  }
  if (origins.size() != rank)
    throw SemanticError("tma_opcode_rank",
                        "opcode rank disagrees with coordinates");
  auto dims = program.config.cluster_dim.value_or(Coordinates{1, 1, 1});
  TmaTransferOptions options;
  options.symbolic_coordinates = symbolic_coordinates;
  options.shared_base = shared->second;
  options.destination_rank = Rank(shared->first);
  options.cluster_size = dims[0] * dims[1] * dims[2];
  options.cta_group = Has(i, "cta_group::2") ? 2 : 1;
  options.store = store;
  std::optional<std::uint64_t> barrier_offset;
  if (!store) {
    auto bar = Shared(e, ops[2]);
    if (!bar || bar->second % 8)
      throw SemanticError("invalid_barrier_address",
                          "TMA completion needs aligned concrete barrier");
    options.completion_rank = Rank(bar->first);
    barrier_offset = bar->second;
    if (options.cta_group == 2) {
      auto token = Trim(std::string_view(ops[2]).substr(1, ops[2].size() - 2));
      auto split = SplitAddress(token);
      auto value = Value(e, split ? split->first : token);
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
        const auto owner = Owner(e, split ? split->first : token);
        if (!owner) throw SemanticError("tma_group2_barrier_owner",
            "group2 completion address lacks shared-pointer provenance");
        options.completion_rank = *owner;
      }
    }
    if (spaces[0] == "shared::cta" &&
        options.completion_rank != Rank(coordinates))
      throw SemanticError("tma_shared_cta_owner",
                          "local TMA barrier resolves to peer");
  }
  if (Has(i, "multicast::cluster"))
    options.multicast_mask = Unsigned(ScalarValue(e, ops[3], "multicast mask") & 65535,
                                      "multicast mask", UINT32_MAX);
  if (contract.contains("transfer_bytes"))
    options.transfer_bytes =
        Unsigned(JsonInteger(contract["transfer_bytes"]), "transfer bytes");
  auto mapping = DecodeTmaTransfer(descriptor, origins, options);
  auto resource = descriptor.at("resource").get<std::string>();
  TmaAccesses result;
  result.pairs.reserve(mapping.segments.size());
  for (const auto &s : mapping.segments) {
    std::optional<Fragment> global;
    if (s.abstract_global) {
      Json expression={{"operator","tma_region"},{"descriptor",descriptor},
          {"coordinates",coordinate_expressions},{"logical_offset",s.logical_offset},{"bytes",s.bytes},
          {"occurrence",Json::array({coordinates,active_lane_->warp_id,active_lane_->lane_id,e.instruction_id(),e.occurrence()})}};
      const auto serialized=expression.dump();
      auto address=addresses.Intern("global","expression:"+serialized,0);
      auto &binding=addresses.bindings.at(address.value());
      binding.allocation="unresolved";
      // A dynamic origin does not erase a descriptor's known global allocation.
      if (resource != "unresolved") {
        const auto base = MemoryResource(resource, 0, 0);
        const auto placement = addresses.bindings.at(base.first.value());
        // MemoryResource may append and invalidate the reference above.
        auto& placed = addresses.bindings.at(address.value());
        placed.allocation = placement.allocation;
        placed.offset = placement.offset;
      }
      addresses.bindings.at(address.value()).expression=serialized;
      global=Fragment{address,s.bytes};
    } else if (s.global_offset)
      global = MemoryResource(resource, *s.global_offset, s.bytes);
    auto sh = Bind("shared", "physical.shared.cta.0", s.shared_offset, s.bytes,
                   Target(s.target_rank));
    if (store)
      result.pairs.push_back({sh, global, s.logical_offset, s.target_rank});
    else
      result.pairs.push_back({global, sh, s.logical_offset, s.target_rank});
  }
  for (const auto &c : mapping.completions) {
    if (!barrier_offset)
      throw SemanticError("tma_completion", "completion has no barrier");
    result.completions.emplace_back(
        barriers.Intern("physical", Target(c.rank),
                        std::to_string(*barrier_offset)),
        c.bytes);
  }
  return result;
}
} // namespace sentryward::frontend::ptx
