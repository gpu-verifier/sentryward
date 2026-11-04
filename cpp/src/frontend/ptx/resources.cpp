#include "sentryward/frontend/ptx/resources.h"
#include "sentryward/frontend/ptx/layouts.h"
#include "sentryward/frontend/ptx/slot_resolver.h"
#include "sentryward/frontend/ptx/value_ir.h"
#include <algorithm>
#include <regex>
#include <stdexcept>
#include <type_traits>
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
std::vector<RegId> AddressRegisters(const FactSource &facts, std::string_view operand) {
  auto token = Trim(operand);
  if (token.starts_with('[') && token.ends_with(']')) token = Trim(std::string_view(token).substr(1, token.size() - 2));
  std::vector<RegId> regs;
  if (const auto reg = facts.Reg(token)) regs.push_back(*reg);
  if (const auto split = SplitAddress(token))
    if (const auto reg = facts.Reg(split->first); reg && std::find(regs.begin(), regs.end(), *reg) == regs.end())
      regs.push_back(*reg);
  std::sort(regs.begin(), regs.end());
  return regs;
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
std::uint64_t Bits(const Integer &x, std::string_view field, unsigned width) {
  const Integer span = Integer(1) << width;
  if (x < -(span >> 1) || x >= span)
    throw SemanticError("integer_range",
                        std::string(field) + " outside its " + std::to_string(width) + "-bit pattern");
  return (x < 0 ? x + span : x).convert_to<std::uint64_t>();
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
  if (inserted && keep_writes_) name_keys_.push_back(&name->first);
  auto [it, added] =
      ids_.try_emplace({name->second, offset}, AddressId(bindings.size()));
  if (added) {
    bindings.push_back({it->second, std::move(space), std::move(allocation),
                        offset, std::move(alias_scope)});
    if (keep_writes_) {
      keys_.push_back(it->first);
      written_.push_back(0);
    }
  } else if (bindings[it->second.value()].alias_scope != alias_scope)
    throw SemanticError("alias_scope_conflict",
                        "inconsistent allocation alias domain");
  return it->second;
}
std::optional<std::vector<AddressId>> AddressTable::Absorb(const AddressTable &local) {
  if (!local.keep_writes_) throw std::logic_error("AddressTable::Absorb needs a KeepWrites table");
  const auto count = local.bindings.size();
  // Each local name's index here, if it has one (one name lookup per local name).
  std::vector<std::optional<std::uint32_t>> names(local.name_keys_.size());
  for (std::size_t index = 0; index < names.size(); ++index)
    if (const auto found = names_.find(*local.name_keys_[index]); found != names_.end()) names[index] = found->second;
  // First every lookup (an existing binding with another alias scope: the conflict), then the writes.
  std::vector<std::optional<AddressId>> existing(count);
  for (std::size_t index = 0; index < count; ++index) {
    const auto &key = local.keys_[index];
    const auto &name = names.at(key.name);
    if (!name) continue;
    const auto found = ids_.find({*name, key.offset});
    if (found == ids_.end()) continue;
    if (bindings[found->second.value()].alias_scope != local.bindings[index].alias_scope) return std::nullopt;
    existing[index] = found->second;
  }
  std::vector<AddressId> ids;
  ids.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const auto &from = local.bindings[index];
    const auto &[space, allocation] = *local.name_keys_[local.keys_[index].name];
    const auto id = existing[index] ? *existing[index]
                                    : Intern(space, allocation, local.keys_[index].offset, from.alias_scope);
    auto &to = bindings[id.value()];
    const auto fields = local.written_[index];
    if (fields & kWroteAllocation) to.allocation = from.allocation;
    if (fields & kWroteOffset) to.offset = from.offset;
    if (fields & kWroteExpression) to.expression = from.expression;
    Wrote(id, fields);
    ids.push_back(id);
  }
  return ids;
}
BarrierLifetimeId BarrierTable::Intern(std::string kind, Coordinates owner,
                                       std::string id) {
  const auto [it, added] = ids_.try_emplace({std::move(kind), owner, std::move(id)},
                                            BarrierLifetimeId(ids_.size()));
  return it->second;
}
std::vector<BarrierLifetimeId> BarrierTable::Absorb(const BarrierTable &local) {
  std::vector<const std::tuple<std::string, Coordinates, std::string> *> keys(local.ids_.size());
  for (const auto &[key, id] : local.ids_) keys.at(id.value()) = &key;
  std::vector<BarrierLifetimeId> ids;
  ids.reserve(keys.size());
  for (const auto *key : keys) ids.push_back(Intern(std::get<0>(*key), std::get<1>(*key), std::get<2>(*key)));
  return ids;
}
AddressId UnitInterner::Constant(FormattedAllocation &&formatted, std::uint64_t offset) {
  return addresses_.Intern(std::move(formatted.space), std::move(formatted.allocation), offset,
                           std::move(formatted.alias));
}
AddressId UnitInterner::Apply(ConstantStep &&step) {
  return Constant(frame_.Format(std::move(step.key)), step.offset);
}
PreparedKey UnitInterner::Prepare(AllocationKey key) const {
  auto formatted = frame_.Format(key);
  return {std::move(key), std::move(formatted)};
}
AddressId UnitInterner::Apply(const PreparedKey &key, std::uint64_t offset) {
  return Constant(FormattedAllocation(key.formatted), offset);
}
std::string UnitInterner::Allocation(StoredAllocation &&allocation) const {
  if (auto *key = std::get_if<AllocationKey>(&allocation))
    return frame_.Format(std::move(*key)).allocation;
  return std::move(std::get<std::string>(allocation));
}
AddressId UnitInterner::Apply(ExpressionStep &&step) {
  const auto id = addresses_.Intern(std::move(step.space), "expression:" + step.serialized, 0,
                                    std::move(step.alias));
  auto &binding = addresses_.bindings.at(id.value());
  binding.allocation = Allocation(std::move(step.allocation));
  if (step.expression) binding.expression = std::move(*step.expression);
  addresses_.Wrote(id, static_cast<std::uint8_t>(AddressTable::kWroteAllocation |
                                                 (step.expression ? AddressTable::kWroteExpression : 0)));
  return id;
}
void UnitInterner::Apply(PatchStep &&step, AddressId target) {
  if (step.allocation) {
    // Formatted first: the binding is looked up only after any formatting.
    auto allocation = Allocation(std::move(*step.allocation));
    addresses_.bindings.at(target.value()).allocation = std::move(allocation);
  }
  if (step.offset) addresses_.bindings.at(target.value()).offset = *step.offset;
  addresses_.bindings.at(target.value()).expression = std::move(step.expression);
  addresses_.Wrote(target, static_cast<std::uint8_t>((step.allocation ? AddressTable::kWroteAllocation : 0) |
                                                     (step.offset ? AddressTable::kWroteOffset : 0) |
                                                     AddressTable::kWroteExpression));
}
BarrierLifetimeId UnitInterner::Apply(BarrierStep &&step) {
  return barriers_.Intern(std::string(KindText(step.key.kind)), frame_.Cta(step.key.owner),
                          std::move(step.key.identity));
}
void UnitInterner::RequireAligned(const ConstantAddressBinding &binding, std::uint64_t modulus,
                                  AddressId id) const {
  if (facts_.recording()) [[unlikely]]
    facts_.Note(query::AbsoluteCheck{id, modulus, &binding});
  if (binding.offset % modulus)
    throw SemanticError("bulk_reduction_alignment", "reduction base must be 16-byte aligned");
}
std::uint32_t UnitInterner::ApplyStep(SlotStep &&slot, const std::vector<std::uint32_t> *ids) {
  auto &step = slot.step;
  if (auto *constant = std::get_if<ConstantStep>(&step)) return Apply(std::move(*constant)).value();
  if (auto *expression = std::get_if<ExpressionStep>(&step))
    return Apply(std::move(*expression)).value();
  if (auto *patch = std::get_if<PatchStep>(&step)) {
    if (!ids) throw std::logic_error("slot prefix PatchStep has no target ids");
    const AddressId target(ids->at(patch->target));
    Apply(std::move(*patch), target);
    return 0;
  }
  return Apply(std::move(std::get<BarrierStep>(step))).value();
}
std::vector<std::uint32_t> UnitInterner::Apply(std::vector<SlotStep> &&steps) {
  std::vector<std::uint32_t> ids;
  ids.reserve(steps.size());
  for (auto &step : steps) {
    const auto id = ApplyStep(std::move(step), &ids);
    ids.push_back(id);
  }
  return ids;
}
void UnitInterner::ApplyPrefix(std::vector<SlotStep> &&steps) {
  for (auto &step : steps) ApplyStep(std::move(step), nullptr);
}
Fragment UnitInterner::Commit(SlotOutcome<ResolvedFootprint> &&outcome) {
  // A footprint prefix holds only placement bindings (no PatchStep): no ids.
  if (!outcome.prefix.empty()) ApplyPrefix(std::move(outcome.prefix));
  if (auto *error = std::get_if<SlotError>(&outcome.terminal)) std::rethrow_exception(error->exception);
  auto &resolved = *std::get_if<ResolvedFootprint>(&outcome.terminal);
  std::optional<AddressId> base;
  if (resolved.base) base = Apply(std::move(*resolved.base));
  if (resolved.expression) return {Apply(std::move(*resolved.expression)), resolved.bytes};
  return {*base, resolved.bytes};
}
TmaAccesses UnitInterner::Commit(SlotOutcome<ResolvedTma> &&outcome) {
  if (!outcome.prefix.empty()) Apply(std::move(outcome.prefix));
  if (auto *error = std::get_if<SlotError>(&outcome.terminal)) std::rethrow_exception(error->exception);
  auto &resolved = *std::get_if<ResolvedTma>(&outcome.terminal);
  const auto ids = Apply(std::move(resolved.steps));
  TmaAccesses result;
  result.pairs.reserve(resolved.pairs.size());
  for (const auto &pair : resolved.pairs) {
    std::optional<Fragment> global;
    if (pair.global) global = Fragment{AddressId(ids[*pair.global]), pair.bytes};
    const Fragment shared{AddressId(ids[pair.shared]), pair.bytes};
    if (resolved.store)
      result.pairs.push_back({shared, global, pair.logical_offset, pair.target_rank});
    else
      result.pairs.push_back({global, shared, pair.logical_offset, pair.target_rank});
  }
  for (const auto &[step, bytes] : resolved.completions)
    result.completions.emplace_back(BarrierLifetimeId(ids[step]), bytes);
  return result;
}
Resources::Resources(const FactSource &f, const UnitFrame &u, AddressTable &a,
                     BarrierTable &b, SlotConfig slots)
    : facts(f), frame(u), interner_(f, u, a, b), id_(f.NewResourcesId()),
      slots_(std::make_unique<SlotResolver>(*this, std::move(slots))) {
  for (const auto &r : facts.launch().resources()) {
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
      auto text = std::string(facts.module().view(d.source));
      if (text.find(".shared") == text.npos)
        continue;
      std::smatch match;
      if (std::regex_search(text, match, pattern))
        shared_symbols_.insert(match[1]);
    }
  };
  add(facts.function().directives);
  add(facts.module().directives);
  for (const auto& local : LocalMemoryDeclarations(facts.module(), facts.function()))
    records_.emplace(local.resource, Json{{"id",local.resource},{"kind","memory"},{"space","local"},
        {"allocation",{{"resolution","concrete"},{"base_byte_offset",0},{"object",local.resource}}}});
}
Resources::Resources(const Resources &like, AddressTable &a, BarrierTable &b, SlotConfig slots)
    : facts(like.facts), frame(like.frame), interner_(like.facts, like.frame, a, b), id_(like.id_),
      slots_(std::make_unique<SlotResolver>(*this, std::move(slots))), records_(like.records_),
      shared_symbols_(like.shared_symbols_) {}
Resources::~Resources() = default;
std::vector<Coordinates> BarrierTable::Owners() const {
  std::vector<Coordinates> owners;
  owners.reserve(ids_.size());
  for (const auto &[key, id] : ids_) owners.push_back(std::get<1>(key));
  return owners;
}
std::tuple<std::string, Coordinates, std::string> BarrierTable::Location(BarrierLifetimeId id) const {
  for (const auto &[key,value] : ids_) if (value==id) return key;
  throw SemanticError("shared_control_operand", "unknown barrier identity");
}
void Resources::SetLane(LaneRef lane) {
  active_lane_ = lane;
  active_index_ = &lane_indices_[lane];
}
void Resources::BeginLane(LaneRef lane) {
  lane_indices_.clear();
  SetLane(lane); // a new active index: the old one was cleared
}
std::optional<RecordView> Resources::Definition(DynamicKey key, bool address_slice) const {
  if (!active_lane_) return {};
  auto &index = *active_index_;
  const auto lane = *active_lane_;
  auto &positions=occurrence_indices_[facts.Path(lane)];
  if (positions.indexed > facts.Size(lane)) {positions = {};index = {};}
  for (; positions.indexed < facts.Size(lane); ++positions.indexed)
    positions.executions[facts.Meta({lane, positions.indexed}).Key()] = positions.indexed;
  const auto found = positions.executions.find(key);
  if (found == positions.executions.end()) return {};
  const RecordRef at{lane, found->second};
  if (facts.Expanded(lane)) return facts.Meta(at);
  if (address_slice) {
    auto [entry, inserted] = index.address_requested.try_emplace(key);
    if (inserted) {
      auto [registers, added] = address_instruction_registers_.try_emplace(key.instruction);
      if (added)
        for (const auto& name : facts.function().instructions.at(key.instruction).registers)
          if (const auto reg = facts.Reg(name))
            registers->second.push_back(*reg);
      entry->second = facts.Keep(facts.ReadSlice(at, registers->second));
      return facts.ViewSlice(entry->second, at, registers->second);
    }
    // The entry was read over this instruction's registers, built once and final.
    return facts.ViewSlice(entry->second, at, address_instruction_registers_.find(key.instruction)->second);
  }
  auto [entry, inserted] = index.requested.try_emplace(key);
  if (inserted) entry->second = facts.Keep(facts.ReadFull(at));
  return facts.ViewFull(entry->second, at);
}
Scalar Resources::Value(const RecordView &e,
                        std::string_view name) const {
  if (auto n = ParseInteger(name))
    return *n;
  if (auto p = facts.Value(e, name))
    return *p;
  const auto &parameters = facts.launch().parameters();
  auto it = parameters.find(std::string(name));
  if (it != parameters.end())
    return it->second;
  return {};
}
Integer Resources::ScalarValue(const RecordView &e,
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
std::optional<std::uint32_t> Resources::Owner(const RecordView &e,
                                              std::string_view operand,
                                              std::set<DynamicKey> visiting) {
  auto name = Trim(operand);
  auto v = Value(e, name);
  if (auto p = std::get_if<ClusterSharedPointer>(&v))
    return p->target_rank;
  if (shared_symbols_.contains(name)) {
    return frame.self().value;
  }
  auto origin = facts.Origin(e, name);
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
  const auto &i = facts.function().instructions[def.instruction_id()];
  auto ops = OperandTexts(facts.module(), i);
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
      if (active_lane_ && lane == Integer(facts.LaneId(*active_lane_))) {
        result = Owner(def, ops[1], visiting);
        owners[*origin] = result;
        return result;
      }
      const auto saved_lane=active_lane_; auto *saved_index=active_index_;
      if (const auto peer=facts.FindLane(facts.Warp(*saved_lane),lane->convert_to<std::uint32_t>())) {
        SetLane(*peer);
        if (auto source=Definition(*origin))
          result=Owner(*source,ops[1],visiting);
      }
      active_lane_=saved_lane; active_index_=saved_index;
    }
  }
  owners[*origin] = result;
  return result;
}
std::optional<std::uint32_t> Resources::GenericOwner(const RecordView &e,
    std::string_view operand, std::set<DynamicKey> visiting) {
  auto origin=facts.Origin(e,Trim(operand));
  if (!origin || !active_index_ || !visiting.insert(*origin).second) return {};
  auto &cache=active_index_->generic_owners;
  if (auto known=cache.find(*origin);known!=cache.end()) return known->second;
  auto found=Definition(*origin); if (!found) return {};
  const auto &def=*found; const auto &i=facts.function().instructions[def.instruction_id()];
  const auto ops=OperandTexts(facts.module(),i);
  auto owner=[&](std::string_view token){return GenericOwner(def,token,visiting);};
  auto scalar=[&](std::string_view token){return ScalarInteger(Value(def,token));};
  std::optional<std::uint32_t> result;
  if (i.opcode=="cvta" && Has(i,"shared") && !Has(i,"to") && ops.size()==2) {
    result=Owner(def,ops[1]);
    if (!result && scalar(ops[1])) {
      result=frame.self().value;
    }
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
Integer Resources::AddressRegisterBits(const RecordView &e,
                                      std::string_view token, const Integer &value) {
  // Execution keeps congruent unbounded representatives. Both event lowering
  // and shared-control execution must interpret declared address register bits
  // after displacement, before range checks, without truncating literal addresses.
  Integer bits = value;
  if (bits > UINT32_MAX && !ParseInteger(token)) {
    if (!address_register_types_)
      address_register_types_ = std::make_unique<RegisterTable>(facts.module(), facts.function());
    const auto &instruction = facts.function().instructions.at(e.instruction_id());
    const auto identity = address_register_types_->Resolve(instruction.scope, token);
    if (auto bytes = address_register_types_->DeclaredWidth(identity); bytes && (*bytes == 4 || *bytes == 8))
      bits &= (Integer(1) << (8 * *bytes)) - 1;
  }
  return bits;
}
std::optional<SharedAddress> Resources::Shared(const RecordView &e,
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
    // Braced: Rank's invalid_cluster_rank is thrown before Unsigned's integer_range.
    return SharedAddress{
        owner ? frame.Rank(*owner) : frame.self(),
        Unsigned(bits, "shared address", UINT32_MAX)};
  }
  if (auto p = std::get_if<ClusterSharedPointer>(&value)) {
    return SharedAddress{frame.Rank(p->target_rank),
                         Unsigned(p->local_address + displacement,
                                  "shared address", UINT32_MAX)};
  }
  return {};
}
BarrierLifetimeId Resources::Barrier(const RecordView &e,
                                     std::string_view operand,
                                     const Json &contract) {
  if (auto shared = Shared(e, operand)) {
    if (shared->second % 8)
      throw SemanticError("invalid_barrier_address",
                          "mbarrier address must be aligned to eight bytes");
    return interner_.Apply(BarrierStep{{BarrierKey::Kind::Physical, shared->first,
                                        std::to_string(shared->second)}});
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
  return interner_.Apply(BarrierStep{{BarrierKey::Kind::Symbolic, frame.self(), *matches.begin()}});
}
ConstantStep Resources::BindStep(std::string space, std::string resource, std::uint64_t offset,
                                 std::uint64_t bytes, std::optional<CtaRank> owner) const {
  // The key is built in the returned step (BindKey, then CheckRange).
  ConstantStep step{BindKey(std::move(space), std::move(resource), owner ? *owner : frame.self(), ActiveThread()),
                    offset};
  CheckRange(offset, bytes);
  return step;
}
Fragment Resources::Bind(std::string space, std::string resource,
                         std::uint64_t offset, std::uint64_t bytes,
                         std::optional<CtaRank> owner) {
  auto step = BindStep(std::move(space), std::move(resource), offset, bytes, owner);
  return {interner_.Apply(std::move(step)), bytes};
}
ExpressionStep Resources::AbstractExpression(const std::string &space, const std::string &allocation,
                                             const std::string &alias, const Json &offset,
                                             std::optional<std::uint64_t> bytes) {
  Json expression = {{"address", {{"base", {{"identity", allocation},
      {"space", space}}}, {"offset", offset}, {"view", nullptr}}},
      {"byte_count", bytes ? Json(*bytes) : Json(nullptr)}};
  return SerializedExpressionStep(space, expression.dump(), alias, allocation);
}
Fragment Resources::Abstract(std::string space, std::string resource,
                             const Json &offset, std::optional<std::uint64_t> bytes,
                             std::optional<CtaRank> owner) {
  auto base = Bind(space, resource, 0, bytes.value_or(0), owner);
  const auto binding = interner_.Stored(base.first);
  auto address = interner_.Apply(AbstractExpression(binding.address_space, binding.allocation,
                                                    binding.alias_scope, offset, bytes));
  return {address, bytes};
}
Fragment Resources::Memory(const RecordView &e, const MemoryRequest &request) {
  return interner_.Commit(slots_->Memory(e, request));
}
TmaAccesses Resources::Tma(const RecordView &e, const TmaRequest &request) {
  auto result = interner_.Commit(slots_->Tma(e, request));
  const auto &reads = facts.TensorMapReads();
  if (const auto thread = ActiveThread(); reads && thread && reads->contains({*thread, e.Key()})) {
    auto tensor = Trim(OperandTexts(facts.module(), facts.function().instructions[e.instruction_id()])
                           .at(request.store ? 0 : 1));
    tensor = "[" + Trim(std::string_view(tensor).substr(1, tensor.find(',') - 1)) + "]";
    const auto &binding = Binding(Memory(e, MemoryRequest(tensor, Json::object(), 128, "global")).first);
    if (!binding.expression.empty())
      throw SemanticError("tensor_map_fragment", "a runtime tensor map read requires a constant region");
    // The stored alias is not carried.
    result.descriptor = Fragment{interner_.Apply(ConstantStep{
        AllocationKey::Opaque(binding.address_space, binding.allocation, {}), binding.offset}), 128};
  }
  return result;
}
SlotOutcome<ResolvedFootprint> Resources::ReplayMemory(const RecordView &e, const MemoryRequest &request) {
  return slots_->Memory(e, request);
}
SlotOutcome<ResolvedTma> Resources::ReplayTma(const RecordView &e, const TmaRequest &request) {
  return slots_->Tma(e, request);
}
EventTape::View EventTape::Copy(const RecordView &view) {
  const auto ref = view.ref();
  View copy{ref.value_or(RecordRef{}), {}, view.kind(), ref.has_value()};
  copy.slice.assign(view.slice().begin(), view.slice().end());
  return copy;
}
RecordView EventTape::Rebuild(const FactSource &facts, const View &view) {
  static const ExecutedInstruction unpositioned;
  if (!view.positioned) return RecordView(&unpositioned, LaneRef{}, view.kind);
  return RecordView(facts.Meta(view.at).record_, view.at, view.kind, view.slice);
}
void EventTape::On(const QueryEvent &event, ConsumerTag consumer, bool in_slot, bool in_partition) noexcept {
  if (!complete_) return;
  try {
    const auto copy = [](const auto &e) -> Event {
      using T = std::decay_t<decltype(e)>;
      if constexpr (std::is_same_v<T, query::ValueRead>)
        return ValueRead{Copy(e.view), e.reg, e.answer ? std::optional<Scalar>(*e.answer) : std::nullopt};
      else if constexpr (std::is_same_v<T, query::LegacyUniform>)
        return LegacyUniform{e.leader, e.expanded, std::string(e.name), e.answer,
                             std::vector<LaneRef>(e.members.begin(), e.members.end())};
      else if constexpr (std::is_same_v<T, query::SpecialRead>)
        return SpecialRead{std::string(e.name), e.answer};
      else if constexpr (std::is_same_v<T, query::SlotMemory> || std::is_same_v<T, query::SlotTma>) {
        const auto &q = *e.query;
        Slot slot{q.consumer, Copy(q.site), std::nullopt, std::nullopt, q.active_lane, *e.outcome};
        if (q.memory)
          slot.memory = MemoryCopy{std::string(q.memory->operand), q.memory->contract, q.memory->bytes,
                                   q.memory->space};
        if (q.tma) slot.tma = TmaCopy{q.tma->contract, q.tma->store, q.tma->dynamic_tensor_maps};
        return std::make_unique<const Slot>(std::move(slot));
      } else if constexpr (std::is_same_v<T, query::UniformBodySkip>)
        return UniformBodySkip{e.lane, std::vector<std::size_t>(e.positions.begin(), e.positions.end()),
                               std::vector<std::size_t>(e.varying_slots.begin(), e.varying_slots.end())};
      else if constexpr (std::is_same_v<T, query::AbsoluteCheck>)
        return AbsoluteCheck{e.id, e.modulus,
                             e.binding ? std::optional<ConstantAddressBinding>(*e.binding) : std::nullopt};
      else if constexpr (std::is_same_v<T, query::PartitionRead>)
        return PartitionRead{e.leader, e.evaluated, e.instruction, e.classes, e.end,
                             std::vector<LaneRef>(e.members.begin(), e.members.end()), e.spec, e.shuffles};
      else if constexpr (std::is_same_v<T, query::ReportBearing>)
        return ReportBearing{e.kind, std::string(e.code)};
      else
        return e; // self-contained
    };
    entries_.push_back({consumer, in_slot, in_partition, std::visit(copy, event)});
  } catch (...) {
    complete_ = false;
  }
}
void EventTape::Replay(const FactSource &facts, std::span<const AddressId> addresses) const {
  if (!complete_) throw std::logic_error("replay of an incomplete event tape");
  if (!facts.sink_) return;
  auto &sink = *facts.sink_;
  for (const auto &entry : entries_) {
    const auto note = [&](const QueryEvent &event) { sink.On(event, entry.consumer, entry.in_slot, entry.in_partition); };
    std::visit([&](const auto &e) {
      using T = std::decay_t<decltype(e)>;
      if constexpr (std::is_same_v<T, ValueRead>)
        note(query::ValueRead{Rebuild(facts, e.view), e.reg, e.answer ? &*e.answer : nullptr});
      else if constexpr (std::is_same_v<T, LegacyUniform>)
        note(query::LegacyUniform{e.leader, e.expanded, e.name, e.answer, e.members});
      else if constexpr (std::is_same_v<T, SpecialRead>)
        note(query::SpecialRead{e.name, e.answer});
      else if constexpr (std::is_same_v<T, std::unique_ptr<const Slot>>) {
        const auto &slot = *e;
        std::optional<MemoryRequest> memory;
        std::optional<TmaRequest> tma;
        if (slot.memory)
          memory.emplace(slot.memory->operand, slot.memory->contract, slot.memory->bytes, slot.memory->space);
        if (slot.tma) tma.emplace(slot.tma->contract, slot.tma->store, slot.tma->dynamic_tensor_maps);
        const SlotQuery query{slot.consumer, Rebuild(facts, slot.site), memory ? &*memory : nullptr,
                              tma ? &*tma : nullptr, slot.active_lane};
        if (const auto *outcome = std::get_if<SlotOutcome<ResolvedFootprint>>(&slot.outcome))
          note(query::SlotMemory{&query, outcome});
        else
          note(query::SlotTma{&query, &std::get<SlotOutcome<ResolvedTma>>(slot.outcome)});
      } else if constexpr (std::is_same_v<T, UniformBodySkip>)
        note(query::UniformBodySkip{e.lane, e.positions, e.varying_slots});
      else if constexpr (std::is_same_v<T, query::BindingRead>)
        note(query::BindingRead{addresses[e.id.value()]});
      else if constexpr (std::is_same_v<T, AbsoluteCheck>) {
        std::optional<ConstantAddressBinding> binding = e.binding;
        if (binding) binding->address = addresses[binding->address.value()];
        note(query::AbsoluteCheck{addresses[e.id.value()], e.modulus, binding ? &*binding : nullptr});
      } else if constexpr (std::is_same_v<T, PartitionRead>)
        note(query::PartitionRead{e.leader, e.evaluated, e.instruction, e.classes, e.end, e.members, e.spec,
                                  e.shuffles});
      else if constexpr (std::is_same_v<T, ReportBearing>)
        note(query::ReportBearing{e.kind, e.code});
      else
        note(e);
    }, entry.event);
  }
}
void Resources::AdmitGrid(const Instruction &i) const {
  const auto &f = facts.launch();
  auto launch = f.LaunchContract();
  auto ev = launch.value("source_evidence", Json::array());
  // modifiers == {m}, without a temporary vector (GCC 15 reports a false
  // -Wfree-nonheap-object for the temporaries once this TU is smaller).
  const auto only = [&](std::string_view m) { return i.modifiers.size() == 1 && i.modifiers[0] == m; };
  if (i.opcode != "griddepcontrol" || !i.operands.empty() ||
      (!only("wait") && !only("launch_dependents")))
    throw SemanticError("grid_dependency_variant",
                        "unknown grid dependency instruction");
  if (f.PublicInputSchema() == "spverify.kernel-input/v1" &&
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
} // namespace sentryward::frontend::ptx
