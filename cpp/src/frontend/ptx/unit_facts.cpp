#include "sentryward/frontend/ptx/unit_facts.h"

#include <algorithm>
#include <array>
#include <bit>

namespace sentryward::frontend::ptx {

std::optional<RegId> FactSource::Reg(std::string_view exact_name) const {
  if (!regs_) {
    auto &table = regs_.emplace();
    const auto &names = program_->register_names;
    table.reserve(names.size());
    for (RegId id = 0; id < names.size(); ++id)
      table.emplace_back(names[id], id);
    // (name, id) order: a name's first entry carries its first id.
    std::sort(table.begin(), table.end());
  }
  const auto found = std::lower_bound(regs_->begin(), regs_->end(), exact_name,
      [](const auto &entry, std::string_view name) { return entry.first < name; });
  if (found == regs_->end() || found->first != exact_name) return {};
  return found->second;
}

std::vector<LaneRef> FactSource::ProgramLanes() const {
  std::vector<LaneRef> lanes;
  lanes.reserve(program_->lane_traces.size());
  for (const auto &trace : program_->lane_traces)
    lanes.push_back(LaneOf(trace));
  return lanes;
}

std::optional<LaneRef> FactSource::FindLane(std::uint32_t warp, std::uint32_t lane) const {
  for (const auto &trace : program_->lane_traces)
    if (trace.warp_id == warp && trace.lane_id == lane) return LaneOf(trace);
  return {};
}

RecordHolder FactSource::CopyMeta(RecordRef at) const {
  return RecordHolder(TraceOf(at.lane).instructions.metadata(at.pos), at, RecordKind::PrototypeCopy, {});
}

RecordHolder FactSource::ReadFull(RecordRef at) const {
  return RecordHolder(TraceOf(at.lane).instructions.read(at.pos), at, RecordKind::Full, {});
}

RecordHolder FactSource::ReadSlice(RecordRef at, std::span<const RegId> regs) const {
  return RecordHolder(TraceOf(at.lane).instructions.readRegisters(at.pos, regs), at, RecordKind::Slice, regs);
}

std::optional<Scalar> FactSource::ValueAt(RecordRef at, RegId reg) const {
  const std::array<RegId, 1> regs{reg};
  const auto record = TraceOf(at.lane).instructions.readRegisters(at.pos, regs);
  std::optional<Scalar> answer;
  if (const auto *value = ById(record, reg)) answer = *value;
  if (recording()) [[unlikely]]
    Note(query::ValueAtRead{at, reg, answer});
  return answer;
}

std::optional<Integer> FactSource::Special(const std::string &name) const {
  std::optional<Integer> answer;
  const auto &specials = program_->config.special_registers;
  if (const auto found = specials.find(name); found != specials.end()) answer = found->second;
  if (recording()) [[unlikely]]
    Note(query::SpecialRead{name, answer});
  return answer;
}

std::optional<Integer> FactSource::LegacyUniformOnlyValue(RecordRef leader, std::span<const LaneRef> members,
                                                          std::string_view name) const {
  std::optional<Integer> answer;
  // A cohort-wide value must hold for every member. metadata() returns one
  // lane's concrete record once any consumer materializes the lane, so read
  // the shared template.
  if (const auto *shared = PrototypeRecord(leader)) {
    if (const auto *value = ByName(*shared, name)) answer = ScalarInteger(*value);
  } else {
    // Without a template, every member's concrete value must agree.
    std::optional<Integer> agreed;
    const auto agrees = [&](LaneRef member) {
      const auto *value = ByName(TraceOf(member).instructions.metadata(leader.pos), name);
      const auto known = value ? ScalarInteger(*value) : std::optional<Integer>{};
      if (!known || (agreed && *agreed != *known)) return false;
      agreed = known;
      return true;
    };
    bool uniform = true;
    if (members.empty())
      uniform = agrees(leader.lane);
    else
      for (const auto member : members)
        if (!(uniform = agrees(member))) break;
    if (uniform) answer = agreed;
  }
  if (recording()) [[unlikely]]
    Note(query::LegacyUniform{leader, Expanded(leader.lane), name, answer, members});
  return answer;
}

void FactSource::Note(const QueryEvent &event) const {
  // Every call site is inside `if (recording())`; the check keeps a stray call inert.
  if (sink_) sink_->On(event, current_, slot_depth_ != 0, partition_depth_ != 0);
}

std::pair<unsigned, bool> ShuffleSource(const FactSource &facts, RecordRef at) {
  const auto meta = facts.Meta(at);
  const auto &i = facts.function().instructions[meta.instruction_id()];
  const auto known = [&](unsigned index) -> Integer {
    const auto text = Trim(facts.module().view(i.operands.at(index)));
    if (auto literal = ParseInteger(text)) return *literal;
    if (const auto reg = facts.Reg(text))
      if (const auto value = facts.ValueAt(at, *reg))
        if (auto integer = ScalarInteger(*value)) return *integer;
    throw std::invalid_argument("Shuffle selector/clamp/membermask is dynamic");
  };
  const auto b = known(2), c = known(3), mask = known(4);
  const auto lane_id = facts.LaneId(at.lane);
  const auto amount = (b & 31).convert_to<unsigned>();
  const auto segment = ((c >> 8) & 31).convert_to<unsigned>();
  const auto maximum = (lane_id & segment) | ((c & 31).convert_to<unsigned>() & ~segment);
  int source = (lane_id & segment) | (amount & ~segment);
  if (Has(i, "bfly")) source = lane_id ^ amount;
  if (Has(i, "up")) source = int(lane_id) - int(amount);
  if (Has(i, "down")) source = lane_id + amount;
  const bool valid = Has(i, "up") ? source >= int(maximum) : source <= int(maximum);
  if (!valid) source = lane_id;
  const auto bits = (mask & 0xffffffff).convert_to<std::uint32_t>();
  if (source < 0 || source >= 32 || !(bits & (1u << lane_id)) || !(bits & (1u << source)))
    throw std::invalid_argument("Shuffle issuer or source is outside membermask");
  return {unsigned(source), valid};
}

PartitionCursor FactSource::BeginPartition(std::span<const LaneRef> members, std::size_t pos,
                                           const PartitionSpec &spec) const {
  return PartitionCursor(*this, members, pos, spec);
}

PartitionCursor::PartitionCursor(const FactSource &facts, std::span<const LaneRef> members, std::size_t pos,
                                 const PartitionSpec &spec)
    : facts_(facts), members_(members), pos_(pos), spec_(spec),
      instruction_(facts.function().instructions[spec.instruction]) {
  if (members.empty()) throw std::logic_error("Partition without members");
  // Register ids are per program: resolved once per call, not per member.
  regs_.reserve(spec.fragments.size());
  for (const auto &fragment : spec.fragments) {
    const bool named = fragment.kind == PartitionFragment::Token ||
                       (fragment.kind == PartitionFragment::Elected && !fragment.literal);
    regs_.push_back(named ? facts.Reg(fragment.name) : std::nullopt);
  }
  classes_.reserve(members.size());
}

std::string_view PartitionCursor::Next(MemberFacts &facts) {
  const auto member = Member();
  const FactSource::PartitionScope scope(facts_);
  facts = {};
  text_.clear();
  const auto &fragments = spec_.fragments;
  std::optional<RecordView> record;
  for (std::size_t k = 0;; ++k) {
    // The key builder read the member's metadata after its warp and issuer parts.
    if (k == spec_.record_at) record = facts_.Meta({member, pos_});
    if (k == fragments.size()) break;
    const auto &fragment = fragments[k];
    switch (fragment.kind) {
    case PartitionFragment::Warp:
      text_ += "warp:";
      text_ += std::to_string(facts_.Warp(member));
      break;
    case PartitionFragment::Issuer:
      text_ += "issuer:";
      text_ += facts_.LaneId(member) == 0 ? '1' : '0';
      break;
    case PartitionFragment::Token:
      if (regs_[k])
        if (const auto value = facts_.ValueAt({member, pos_}, *regs_[k])) {
          text_ += fragment.name;
          text_ += '=';
          text_ += ScalarJson(*value).dump();
          text_ += ';';
        }
      break;
    case PartitionFragment::Contract:
      text_ += facts_.launch()
                   .InstructionContract(instruction_.line, instruction_.id, facts_.Warp(member), record->occurrence())
                   .dump();
      break;
    case PartitionFragment::Elected: {
      const Integer *raw = fragment.literal ? &*fragment.literal : nullptr;
      std::optional<Integer> known;
      if (!raw && regs_[k])
        if (const auto value = facts_.ValueAt({member, pos_}, *regs_[k]))
          if ((known = ScalarInteger(*value))) raw = &*known;
      if (raw) {
        const auto mask = (*raw & 0xffffffff).convert_to<std::uint32_t>();
        auto chosen = facts_.launch().elect_lane().value_or(32);
        if (chosen >= 32 || !(mask & (1u << chosen))) chosen = std::countr_zero(mask);
        text_ += "elected:";
        text_ += facts_.LaneId(member) == chosen ? '1' : '0';
      }
      break;
    }
    case PartitionFragment::Shuffle:
      try {
        facts.shuffle = ShuffleSource(facts_, {member, pos_});
        text_ += "valid:";
        text_ += facts.shuffle->second ? '1' : '0';
      } catch (const std::invalid_argument &) {
        text_ += "invalid:";
        text_ += std::to_string(facts_.LaneId(member));
      }
      break;
    case PartitionFragment::Missing:
      (void)instruction_.operands.at(fragment.operand); // throws std::out_of_range
      break;
    }
  }
  const auto found = numbering_.try_emplace(text_, static_cast<std::uint32_t>(numbering_.size())).first;
  classes_.push_back(found->second);
  if (facts_.recording()) [[unlikely]]
    shuffles_.push_back(facts.shuffle);
  ++next_;
  return text_;
}

void PartitionCursor::Finish(query::PartitionEnd end) {
  if (finished_) return;
  finished_ = true;
  if (facts_.recording()) [[unlikely]]
    facts_.Note(query::PartitionRead{{members_.front(), pos_}, next_, spec_.instruction, classes_, end, members_,
                                     &spec_, std::move(shuffles_)});
}

void FactSource::NoteValue(const RecordView &view, std::string_view name, const Scalar *answer) const {
  if (const auto reg = Reg(name)) Note(query::ValueRead{view, *reg, answer});
}

LaneRef FactSource::RegisterLane(const LaneTrace &trace) const {
  // registered_ is set only by ExecutorFacts.
  return static_cast<const ExecutorFacts *>(this)->Register(trace);
}

ExpandedRecords CtaFacts::ExpandLane(LaneRef lane, ExpansionSite site) const {
  (void)site;
  const auto &records = TraceOf(lane).instructions.records();
  return ExpandedRecords(records.data(), records.size(), lane);
}

bool CtaFacts::PathUniformBinding(RecordRef at, RegId reg) const {
  const auto &sequence = TraceOf(at.lane).instructions;
  const auto *shared = sequence.sharedMetadata(at.pos);
  const auto &record = shared ? *shared : sequence.metadata(at.pos);
  const bool present = std::any_of(record.known_registers().begin(), record.known_registers().end(),
                                   [&](const auto &binding) { return binding.first == reg; });
  if (recording()) [[unlikely]]
    Note(query::UniformBinding{at, reg, present});
  return present;
}

void CtaFacts::PrototypeRegisterIds(RecordRef at, std::vector<RegId> &out) const {
  out.clear();
  if (const auto *shared = TraceOf(at.lane).instructions.sharedMetadata(at.pos))
    for (const auto &[id, value] : shared->known_registers()) {
      (void)value;
      out.push_back(id);
    }
  std::sort(out.begin(), out.end());
}

std::vector<SharedSiteRef> CtaFacts::SharedSitesRelative() const {
  std::vector<SharedSiteRef> sites;
  sites.reserve(program_->shared_constant_writes.size() + 2 * program_->shared_value_dependencies.size());
  const auto add = [&](const SharedValueSite &site) {
    sites.push_back({frame_.TryRankOf(site.cta), site.cta, site.thread, site.instruction});
  };
  for (const auto &site : program_->shared_constant_writes) add(site);
  for (const auto &dependency : program_->shared_value_dependencies) {
    add(dependency.write);
    add(dependency.read);
  }
  return sites;
}

std::vector<std::vector<LaneRef>> CtaFacts::LanePathCohorts() const {
  std::vector<std::vector<LaneRef>> groups;
  std::unordered_map<PathKey, std::size_t, PathKeyHash> group_of;
  for (std::size_t index = 0; index < LaneCount(); ++index) {
    const auto [found, inserted] = group_of.try_emplace(Path(Lane(index)), groups.size());
    if (inserted) groups.emplace_back();
    groups[found->second].push_back(Lane(index));
  }
  return groups;
}

LaneRef ExecutorFacts::Register(const LaneTrace &trace) const {
  const auto [found, inserted] = ids_.try_emplace(&trace, LaneRef{static_cast<std::uint32_t>(lanes_.size())});
  if (inserted) lanes_.push_back(&trace);
  return found->second;
}
} // namespace sentryward::frontend::ptx
