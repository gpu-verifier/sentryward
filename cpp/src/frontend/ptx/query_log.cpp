#include "sentryward/frontend/ptx/query_log.h"

#include "sentryward/frontend/ptx/resources.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace sentryward::frontend::ptx {
namespace {
using Json = nlohmann::json;
constexpr std::uint32_t kNone = UINT32_MAX;

constexpr std::uint64_t Mix(std::uint64_t z) noexcept {
  z += 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
std::uint64_t HashText(std::string_view text) noexcept {
  std::uint64_t h = Mix(text.size());
  std::size_t i = 0;
  for (; i + 8 <= text.size(); i += 8) {
    std::uint64_t word;
    std::memcpy(&word, text.data() + i, 8);
    h = Mix(h ^ word);
  }
  std::uint64_t tail = 0;
  if (i < text.size()) std::memcpy(&tail, text.data() + i, text.size() - i);
  return Mix(h ^ tail);
}
std::uint64_t HashInteger(const Integer &x) noexcept {
  const auto &backend = x.backend();
  std::uint64_t h = Mix(std::uint64_t(x.sign() + 2));
  for (unsigned i = 0; i < backend.size(); ++i) h = Mix(h ^ std::uint64_t(backend.limbs()[i]));
  return h;
}
// Lookup hash of an answer; equality is the variant's exact operator==.
std::uint64_t HashScalar(const Scalar &value) noexcept {
  std::uint64_t h = Mix(value.index() + 1);
  std::visit(
      [&](const auto &v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, Integer>) h = Mix(h ^ HashInteger(v));
        else if constexpr (std::is_same_v<T, bool>) h = Mix(h ^ (v ? 3 : 5));
        else if constexpr (std::is_same_v<T, ParameterPointer>) h = Mix(h ^ HashText(v.region) ^ Mix(HashInteger(v.offset)));
        else if constexpr (std::is_same_v<T, AllocationPointer>)
          h = Mix(h ^ HashText(v.resource) ^ Mix(HashInteger(v.offset)) ^ Mix(v.extent_bytes.value_or(7)));
        else if constexpr (std::is_same_v<T, ClusterSharedPointer>) h = Mix(h ^ HashInteger(v.local_address) ^ v.target_rank);
        else if constexpr (std::is_same_v<T, PointerBits>)
          h = Mix(h ^ v.source_bit ^ (std::uint64_t(v.width) << 16) ^ (std::uint64_t(v.destination_bit) << 32));
        else if constexpr (std::is_same_v<T, DescriptorFragment>)
          h = Mix(h ^ HashText(v.descriptor) ^ v.offset ^ (std::uint64_t(v.bytes) << 32));
        else if constexpr (std::is_same_v<T, BoundedInteger>)
          h = Mix(h ^ (v.value ? HashText(v.value->identity) : 0));
        else if constexpr (std::is_same_v<T, BoundedAllocationPointer>)
          h = Mix(h ^ (v.value ? HashText(v.value->resource) ^ Mix(HashInteger(v.value->offset.constant)) ^
                                     HashText(v.value->offset_identity) : 0));
      },
      value);
  return h;
}

// ---- per-unit pools ---------------------------------------------------------
class Strings {
public:
  std::uint32_t Id(std::string_view text) {
    const auto [it, added] = ids_.try_emplace(std::string(text), static_cast<std::uint32_t>(values_.size()));
    if (added) {
      values_.push_back(&it->first);
      bytes_ += text.size() + 48;
    }
    return it->second;
  }
  [[nodiscard]] const std::string &At(std::uint32_t id) const { return *values_.at(id); }
  [[nodiscard]] std::uint64_t Bytes() const noexcept { return bytes_; }

private:
  std::unordered_map<std::string, std::uint32_t> ids_;
  std::vector<const std::string *> values_;
  std::uint64_t bytes_ = 0;
};
// Answers interned once per unit (exact equality; the hash selects candidates).
class Answers {
public:
  std::uint32_t Id(const Scalar *value) {
    if (!value) return kNone;
    const auto h = HashScalar(*value);
    const auto [first, last] = index_.equal_range(h);
    for (auto it = first; it != last; ++it)
      if (values_[it->second] == *value) return it->second;
    const auto id = static_cast<std::uint32_t>(values_.size());
    values_.push_back(*value);
    index_.emplace(h, id);
    return id;
  }
  std::uint32_t Id(const std::optional<Integer> &value) {
    if (!value) return kNone;
    const Scalar scalar(*value);
    return Id(&scalar);
  }
  [[nodiscard]] bool Same(std::uint32_t id, const Scalar *value) const {
    if (id == kNone || !value) return id == kNone && !value;
    return values_.at(id) == *value;
  }
  [[nodiscard]] bool Same(std::uint32_t id, const std::optional<Integer> &value) const {
    if (id == kNone || !value) return id == kNone && !value;
    const auto *integer = std::get_if<Integer>(&values_.at(id));
    return integer && *integer == *value;
  }
  [[nodiscard]] std::uint64_t Bytes() const noexcept { return values_.size() * (sizeof(Scalar) + 32); }

private:
  std::vector<Scalar> values_;
  std::unordered_multimap<std::uint64_t, std::uint32_t> index_;
};

// ---- log entries ------------------------------------------------------------
// How a member re-reads a logged record: Meta, the shared
// prototype on an unexpanded lane (path-uniform); Full, a full read (also an
// expanded record's bindings); Slice, ReadSlice over the logged set; At,
// FactSource::ValueAt.
enum class ReadKind : std::uint8_t { Meta, Full, Slice, At };
struct ValueEntry {
  std::uint32_t lane, pos;
  RegId reg;
  std::uint32_t slice; // Slice only
  ReadKind kind;
  std::uint32_t answer;
};
struct UniformEntry {
  std::uint32_t lane, pos;
  RegId reg;
  bool present;
};
struct SpecialEntry {
  std::uint32_t name, answer;
};
struct LegacyEntry {
  std::uint32_t lane, pos, members, name, answer;
};
struct MemoEntry {
  std::uint32_t lane, fill, hit;
  RegId reg;
};
struct AlignEntry {
  std::uint32_t object;
  std::uint64_t modulus;
};
struct PartitionEntry {
  std::uint32_t lane, pos, members;
  const PartitionSpec *spec;
  std::uint32_t evaluated, line;
  std::vector<std::uint32_t> classes;
  std::vector<std::optional<std::pair<unsigned, bool>>> shuffles;
};
struct SlotEntry {
  Consumer consumer;
  std::uint16_t resources;
  bool memory;
  ReadKind site_kind;
  std::uint32_t active_lane, site_lane, site_pos, slice, request, shape, offsets_begin, offsets_count, line;
  std::uint32_t shifts_begin, shifts_count;
};
struct GlobalOffset {
  std::uint32_t object;
  std::uint64_t offset;
};
// A bounded-pointer footprint's signed offset term (Canon::shifts).
struct SignedOffset {
  std::uint32_t object;
  std::int64_t value;
};
struct RequestCopy {
  bool tma = false;
  std::string operand;
  Json contract;
  std::uint64_t bytes = 0;
  std::string space;
  bool store = false;
};
// One CTA's log (one rank of a unit).
struct RankLog {
  std::vector<ValueEntry> values;
  std::vector<UniformEntry> uniforms;
  std::vector<SpecialEntry> specials;
  std::vector<LegacyEntry> legacy;
  std::vector<MemoEntry> memos;
  std::vector<AlignEntry> aligns;
  std::vector<PartitionEntry> partitions;
  std::vector<SlotEntry> slots;
  std::vector<GlobalOffset> offsets;
  std::vector<SignedOffset> shifts;
  // (I.5a): this CTA's tensormap.cp_fenceproxy global destinations.
  std::vector<GlobalOffset> fences;
  bool fence_nonconstant = false;
  bool recorded = false;
};
// The logs of one unit (all its ranks) and their pools. A class keeps its
// representative's; a member's is dropped once it is admitted.
struct UnitLog {
  explicit UnitLog(std::size_t ranks) : ranks(ranks) {}
  std::vector<RankLog> ranks;
  Strings strings;
  Answers answers;
  std::vector<std::vector<RegId>> slices;
  std::map<std::vector<RegId>, std::uint32_t> slice_ids;
  std::vector<std::vector<LaneRef>> member_lists;
  std::map<std::vector<std::uint32_t>, std::uint32_t> member_ids;
  std::vector<RequestCopy> requests;
  std::unordered_map<std::string, std::uint32_t> request_ids;
  std::uint64_t request_bytes = 0;
  // Why this log can never admit a member (the first reason).
  std::string closed;
  std::uint32_t Slice(std::span<const RegId> regs) {
    std::vector<RegId> key(regs.begin(), regs.end());
    const auto [it, added] = slice_ids.try_emplace(std::move(key), static_cast<std::uint32_t>(slices.size()));
    if (added) slices.push_back(it->first);
    return it->second;
  }
  std::uint32_t Members(std::span<const LaneRef> lanes) {
    std::vector<std::uint32_t> key;
    key.reserve(lanes.size());
    for (const auto lane : lanes) key.push_back(lane.index);
    const auto [it, added] = member_ids.try_emplace(std::move(key), static_cast<std::uint32_t>(member_lists.size()));
    if (added) member_lists.emplace_back(lanes.begin(), lanes.end());
    return it->second;
  }
  [[nodiscard]] std::uint64_t Bytes() const {
    std::uint64_t bytes = strings.Bytes() + answers.Bytes() + request_bytes;
    for (const auto &s : slices) bytes += 2 * s.size() * sizeof(RegId) + 64;
    for (const auto &m : member_lists) bytes += 2 * m.size() * sizeof(std::uint32_t) + 64;
    for (const auto &r : ranks) {
      bytes += r.values.size() * sizeof(ValueEntry) + r.uniforms.size() * sizeof(UniformEntry) +
               r.specials.size() * sizeof(SpecialEntry) + r.legacy.size() * sizeof(LegacyEntry) +
               r.memos.size() * sizeof(MemoEntry) + r.aligns.size() * sizeof(AlignEntry) +
               r.slots.size() * sizeof(SlotEntry) + r.offsets.size() * sizeof(GlobalOffset) +
               r.shifts.size() * sizeof(SignedOffset) + r.fences.size() * sizeof(GlobalOffset);
      for (const auto &p : r.partitions)
        bytes += sizeof(PartitionEntry) + p.classes.size() * 4 + p.shuffles.size() * sizeof(p.shuffles[0]);
    }
    return bytes;
  }
};

// ---- canonical slot outcomes ------------------------------------------------
// The outcome without the representative-absolute global offsets: `shape`
// holds everything that must be equal; `globals` the Plain global steps'
// (allocation, offset) in step order, whose differences are δ; `shifts` a
// bounded-pointer footprint's signed offset terms (object, value), also up to δ.
struct Canon {
  std::string shape;
  std::vector<std::pair<std::string, std::uint64_t>> globals;
  std::vector<std::pair<std::string, std::int64_t>> shifts;
  std::string unliftable; // non-empty: the outcome can never be shared
};
void Put(std::string &out, std::string_view text) {
  out += std::to_string(text.size());
  out += ':';
  out += text;
}
void Put(std::string &out, std::uint64_t value) {
  out += std::to_string(value);
  out += ',';
}
void PutStep(Canon &canon, const InternStep &step) {
  auto &out = canon.shape;
  std::visit(
      [&](const auto &s) {
        using T = std::decay_t<decltype(s)>;
        if constexpr (std::is_same_v<T, ConstantStep>) {
          out += 'C';
          Put(out, std::uint64_t(s.key.kind));
          Put(out, s.key.space);
          Put(out, s.key.owner.value);
          Put(out, s.key.thread);
          Put(out, s.key.object);
          Put(out, s.key.alias);
          if (s.key.kind == AllocationKey::Kind::Plain && s.key.space == "global") {
            out += 'G';
            canon.globals.emplace_back(s.key.object, s.offset);
          } else Put(out, s.offset);
        } else if constexpr (std::is_same_v<T, BarrierStep>) {
          out += 'B';
          Put(out, std::uint64_t(s.key.kind));
          Put(out, s.key.owner.value);
          Put(out, s.key.identity);
        } else if constexpr (std::is_same_v<T, ExpressionStep>) {
          if (canon.unliftable.empty()) canon.unliftable = "slot_expression";
        } else {
          if (canon.unliftable.empty()) canon.unliftable = "slot_patch";
        }
      },
      step);
}
void PutError(Canon &canon, const SlotError &error) {
  auto &out = canon.shape;
  out += 'E';
  Put(out, error.code);
  try {
    std::rethrow_exception(error.exception);
  } catch (const SemanticError &e) {
    Put(out, typeid(e).name());
    Put(out, e.what());
    Put(out, e.details.dump(-1, ' ', false, Json::error_handler_t::replace));
  } catch (const std::exception &e) {
    Put(out, typeid(e).name());
    Put(out, e.what());
  } catch (...) {
    Put(out, "unknown");
  }
}
// A bounded-pointer footprint (tensor_properties): its placement and base, both at offset 0 of the
// target allocation, are anchors in the shape; the structure (element objects, types, coefficients)
// must be equal; the offset's constant (the allocation's δ) and each element read's byte offset (its
// input object's δ, shared with the object's plain reads) are compared up to δ. False: not this form.
bool CanonicalizeAffine(const std::vector<SlotStep> &prefix, const ResolvedFootprint &resolved, Canon &canon) {
  if (prefix.size() != 1 || !resolved.base || !resolved.affine) return false;
  const auto *placement = std::get_if<ConstantStep>(&prefix.front().step);
  const auto &base = *resolved.base;
  const auto anchor = [](const ConstantStep &s) {
    return s.key.kind == AllocationKey::Kind::Plain && s.key.space == "global" && s.offset == 0;
  };
  if (!placement || !anchor(*placement) || !anchor(base) || placement->key.object != base.key.object) return false;
  auto &out = canon.shape;
  for (const auto *s : {placement, &base}) {
    out += 'Z';
    Put(out, std::uint64_t(s->key.kind));
    Put(out, s->key.space);
    Put(out, s->key.owner.value);
    Put(out, s->key.thread);
    Put(out, s->key.object);
    Put(out, s->key.alias);
  }
  out += 'F';
  Put(out, std::uint64_t(resolved.kind));
  Put(out, resolved.bytes);
  out += 'A';
  Put(out, std::uint64_t(resolved.affine->terms.size()));
  canon.shifts.emplace_back(base.key.object, resolved.affine->constant);
  for (const auto &term : resolved.affine->terms) {
    Put(out, term.object);
    Put(out, term.element_type);
    Put(out, std::to_string(term.coefficient));
    canon.shifts.emplace_back(term.object, term.offset);
  }
  return true;
}
// A TMA region whose symbolic coordinates are all affine in tensor-property elements
// (ResolvedTma::region_affine): each abstract segment's expression and patch are the region
// form, its placement (offset 0 of the target allocation) an anchor; the other steps (shared
// bindings, completions) as usual. The structure (element objects, types, byte coefficients)
// must be equal; the region's constant (the allocation's δ) and each element read's byte
// offset (its object's δ) are compared up to δ. False (canon unchanged): not this form.
bool CanonicalizeAffineTma(const ResolvedTma &resolved, Canon &canon) {
  if (!resolved.region_affine) return false;
  Canon local = canon;
  auto &out = local.shape;
  out += 'Y';
  Put(out, resolved.store ? 1 : 0);
  Put(out, resolved.descriptor);
  std::string object; // the target allocation
  const auto &steps = resolved.steps;
  for (std::size_t k = 0; k < steps.size(); ++k) {
    const auto &step = steps[k].step;
    if (const auto *expression = std::get_if<ExpressionStep>(&step)) {
      if (expression->space != "global" || k + 2 >= steps.size()) return false;
      const auto *placement = std::get_if<ConstantStep>(&steps[k + 1].step);
      const auto *patch = std::get_if<PatchStep>(&steps[k + 2].step);
      if (!placement || !patch || patch->target != k || placement->offset != 0 ||
          placement->key.kind != AllocationKey::Kind::Plain || placement->key.space != "global" ||
          (!object.empty() && object != placement->key.object))
        return false;
      object = placement->key.object;
      out += 'X';
      Put(out, placement->key.space);
      Put(out, placement->key.owner.value);
      Put(out, placement->key.thread);
      Put(out, placement->key.object);
      Put(out, placement->key.alias);
      k += 2;
      continue;
    }
    if (std::holds_alternative<PatchStep>(step)) return false;
    PutStep(local, step);
  }
  if (object.empty() || !local.unliftable.empty()) return false;
  const auto &affine = *resolved.region_affine;
  out += 'A';
  Put(out, std::uint64_t(affine.terms.size()));
  local.shifts.emplace_back(object, affine.constant);
  for (const auto &term : affine.terms) {
    Put(out, term.object);
    Put(out, term.element_type);
    Put(out, std::to_string(term.coefficient));
    local.shifts.emplace_back(term.object, term.offset);
  }
  out += '|';
  for (const auto &pair : resolved.pairs) {
    Put(out, pair.global ? *pair.global + 1 : 0);
    Put(out, pair.shared);
    Put(out, pair.bytes);
    Put(out, pair.logical_offset);
    Put(out, pair.target_rank);
  }
  out += '|';
  for (const auto &[step, bytes] : resolved.completions) {
    Put(out, step);
    Put(out, bytes);
  }
  canon = std::move(local);
  return true;
}
template <class T> void Canonicalize(const SlotOutcome<T> &outcome, Canon &canon) {
  if (outcome.fallback != FallbackUse::None) canon.unliftable = "slot_fallback";
  canon.shape += 'P';
  if constexpr (std::is_same_v<T, ResolvedFootprint>) {
    if (const auto *resolved = std::get_if<0>(&outcome.terminal);
        resolved && resolved->affine && CanonicalizeAffine(outcome.prefix, *resolved, canon))
      return;
  }
  for (const auto &step : outcome.prefix) PutStep(canon, step.step);
  if (const auto *error = std::get_if<SlotError>(&outcome.terminal)) {
    PutError(canon, *error);
    return;
  }
  const auto &resolved = std::get<0>(outcome.terminal);
  if constexpr (std::is_same_v<T, ResolvedFootprint>) {
    canon.shape += 'F';
    Put(canon.shape, std::uint64_t(resolved.kind));
    Put(canon.shape, resolved.bytes);
    if (resolved.base) PutStep(canon, InternStep(*resolved.base));
    if (resolved.expression && canon.unliftable.empty()) canon.unliftable = "slot_expression";
  } else {
    if (CanonicalizeAffineTma(resolved, canon)) return;
    canon.shape += 'T';
    Put(canon.shape, resolved.store ? 1 : 0);
    Put(canon.shape, resolved.descriptor);
    for (const auto &step : resolved.steps) PutStep(canon, step.step);
    canon.shape += '|';
    for (const auto &pair : resolved.pairs) {
      Put(canon.shape, pair.global ? *pair.global + 1 : 0);
      Put(canon.shape, pair.shared);
      Put(canon.shape, pair.bytes);
      Put(canon.shape, pair.logical_offset);
      Put(canon.shape, pair.target_rank);
    }
    canon.shape += '|';
    for (const auto &[step, bytes] : resolved.completions) {
      Put(canon.shape, step);
      Put(canon.shape, bytes);
    }
  }
}

// ---- the recorder (one per recorded CTA) ------------------------------------
struct UniformKey {
  PathKey path;
  std::uint32_t pos;
  RegId reg;
  std::uint8_t kind;
  bool operator==(const UniformKey &) const = default;
};
struct UniformKeyHash {
  std::size_t operator()(const UniformKey &k) const noexcept {
    return Mix(k.path.Hash() ^ Mix((std::uint64_t(k.pos) << 32) | k.reg) ^ k.kind);
  }
};
struct LaneKey {
  std::uint64_t a, b; // (lane, pos), (reg, slice, kind)
  bool operator==(const LaneKey &) const = default;
};
struct LaneKeyHash {
  std::size_t operator()(const LaneKey &k) const noexcept { return Mix(k.a ^ Mix(k.b)); }
};
struct PairHash {
  std::size_t operator()(const std::pair<std::uint64_t, std::uint64_t> &k) const noexcept {
    return Mix(k.first ^ Mix(k.second));
  }
};

class Recorder final : public QuerySink {
public:
  Recorder(UnitLog &log, std::uint32_t rank, const CtaFacts &facts) : log_(log), rank_(log.ranks.at(rank)), facts_(facts) {
    rank_.recorded = true;
  }
  void On(const QueryEvent &event, ConsumerTag consumer, bool in_slot, bool in_partition) noexcept override {
    try {
      if (consumer.kind == Consumer::Driver || consumer.kind == Consumer::Executor) return;
      std::visit([&](const auto &e) { Handle(e, consumer, in_slot, in_partition); }, event);
    } catch (...) {
      Close("recorder_failure");
    }
  }
  // The log is complete: entries in replay order, dedup state released.
  void Seal() {
    std::sort(rank_.values.begin(), rank_.values.end(), [](const ValueEntry &a, const ValueEntry &b) {
      return std::tie(a.lane, a.pos, a.kind, a.slice, a.reg) < std::tie(b.lane, b.pos, b.kind, b.slice, b.reg);
    });
    uniform_seen_ = {};
    lane_seen_ = {};
    special_seen_ = {};
    legacy_seen_ = {};
    fills_ = {};
    align_seen_ = {};
  }

private:
  void Close(std::string reason) {
    if (log_.closed.empty()) log_.closed = std::move(reason);
  }
  bool Closed() const noexcept { return !log_.closed.empty(); }
  bool Position(std::size_t pos, std::uint32_t &out) {
    if (pos > UINT32_MAX - 1) {
      Close("position_overflow");
      return false;
    }
    out = static_cast<std::uint32_t>(pos);
    return true;
  }
  std::uint32_t Line(std::uint32_t instruction) const {
    const auto &instructions = facts_.function().instructions;
    return instruction < instructions.size() ? instructions[instruction].line : 0;
  }
  void Value(RecordRef at, RegId reg, ReadKind kind, std::uint32_t slice, std::uint32_t answer) {
    std::uint32_t pos;
    if (!Position(at.pos, pos)) return;
    if (kind == ReadKind::Meta) {
      // Path-uniform: one lane of the path answers for all of them.
      const auto [it, added] = uniform_seen_.try_emplace({facts_.Path(at.lane), pos, reg, 0}, answer);
      if (!added) {
        if (it->second != answer) Close("inconsistent_answer");
        return;
      }
    } else {
      const LaneKey key{(std::uint64_t(at.lane.index) << 32) | pos,
                        (std::uint64_t(reg) << 32) ^ (std::uint64_t(slice) << 8) ^ std::uint64_t(kind)};
      const auto [it, added] = lane_seen_.try_emplace(key, answer);
      if (!added) {
        if (it->second != answer) Close("inconsistent_answer");
        return;
      }
    }
    rank_.values.push_back({at.lane.index, pos, reg, slice, kind, answer});
  }
  void Handle(const query::ValueRead &e, ConsumerTag, bool in_slot, bool in_partition) {
    if (in_slot || in_partition || Closed()) return; // covered by the slot re-run / the H9 classes
    const auto ref = e.view.ref();
    if (!ref) {
      // An unpositioned view is an empty holder's default record, which binds
      // nothing; any answer from one cannot be re-asked on a member.
      if (e.answer) Close("unpositioned_read");
      return;
    }
    ReadKind kind = ReadKind::Full;
    std::uint32_t slice = 0;
    switch (e.view.kind()) {
    case RecordKind::Prototype: kind = ReadKind::Meta; break;
    case RecordKind::PrototypeCopy: kind = facts_.Expanded(ref->lane) ? ReadKind::Full : ReadKind::Meta; break;
    case RecordKind::Expanded:
    case RecordKind::Full: kind = ReadKind::Full; break;
    case RecordKind::Slice:
      kind = ReadKind::Slice;
      slice = log_.Slice(e.view.slice());
      break;
    case RecordKind::Detached: Close("detached_read"); return;
    }
    Value(*ref, e.reg, kind, slice, log_.answers.Id(e.answer));
  }
  void Handle(const query::ValueAtRead &e, ConsumerTag, bool in_slot, bool in_partition) {
    if (in_slot || in_partition || Closed()) return;
    Value(e.at, e.reg, ReadKind::At, 0, e.answer ? log_.answers.Id(&*e.answer) : kNone);
  }
  void Handle(const query::LegacyUniform &e, ConsumerTag, bool in_slot, bool in_partition) {
    if (in_slot || in_partition || Closed()) return;
    std::uint32_t pos;
    if (!Position(e.leader.pos, pos)) return;
    const LegacyEntry entry{e.leader.lane.index, pos, log_.Members(e.members), log_.strings.Id(e.name),
                            log_.answers.Id(e.answer)};
    const auto [it, added] =
        legacy_seen_.try_emplace({(std::uint64_t(entry.lane) << 32) | pos, (std::uint64_t(entry.members) << 32) | entry.name},
                                 entry.answer);
    if (!added) {
      if (it->second != entry.answer) Close("inconsistent_answer");
      return;
    }
    rank_.legacy.push_back(entry);
  }
  void Handle(const query::UniformBinding &e, ConsumerTag, bool in_slot, bool in_partition) {
    if (in_slot || in_partition || Closed()) return;
    std::uint32_t pos;
    if (!Position(e.at.pos, pos)) return;
    const auto [it, added] = uniform_seen_.try_emplace({facts_.Path(e.at.lane), pos, e.reg, 1}, e.present ? 1u : 0u);
    if (!added) {
      if (it->second != (e.present ? 1u : 0u)) Close("inconsistent_answer");
      return;
    }
    rank_.uniforms.push_back({e.at.lane.index, pos, e.reg, e.present});
  }
  void Handle(const query::SpecialRead &e, ConsumerTag, bool in_slot, bool in_partition) {
    if (in_slot || in_partition || Closed()) return;
    const auto name = log_.strings.Id(e.name);
    const auto answer = log_.answers.Id(e.answer);
    const auto [it, added] = special_seen_.try_emplace(name, answer);
    if (!added) {
      if (it->second != answer) Close("inconsistent_answer");
      return;
    }
    rank_.specials.push_back({name, answer});
  }
  void Handle(const query::SlotBegin &, ConsumerTag, bool, bool) {}
  void Handle(const query::SlotEnd &, ConsumerTag, bool, bool) {}
  template <class T> void Slot(const SlotQuery &q, const SlotOutcome<T> &outcome) {
    const bool memory = std::is_same_v<T, ResolvedFootprint>;
    const auto instruction = q.site.instruction_id();
    const auto &i = facts_.function().instructions.at(instruction);
    // (I.5a): a tensormap.cp_fenceproxy destination (lowering's 128-byte global Memory slot).
    const bool fence = memory && q.memory && q.memory->space == "global" && q.memory->bytes == 128 &&
                       i.opcode == "tensormap" && !i.modifiers.empty() && i.modifiers[0] == "cp_fenceproxy";
    if (Closed() && !fence) return;
    Canon canon;
    Canonicalize(outcome, canon);
    if (fence) {
      if (!canon.unliftable.empty() || canon.globals.size() != 1 || std::holds_alternative<SlotError>(outcome.terminal))
        rank_.fence_nonconstant = true;
      else rank_.fences.push_back({log_.strings.Id(canon.globals.front().first), canon.globals.front().second});
    }
    if (Closed()) return;
    if (!canon.unliftable.empty()) return Close(canon.unliftable);
    const auto ref = q.site.ref();
    if (!ref || q.site.kind() == RecordKind::Detached) return Close("slot_site_unpositioned");
    if (q.tma && q.tma->dynamic_tensor_maps) return Close("slot_dynamic_tensor_maps");
    SlotEntry entry{};
    entry.consumer = q.consumer.kind;
    entry.resources = q.consumer.resources;
    entry.memory = memory;
    entry.slice = 0;
    switch (q.site.kind()) {
    case RecordKind::Prototype: entry.site_kind = ReadKind::Meta; break;
    case RecordKind::PrototypeCopy:
      entry.site_kind = facts_.Expanded(ref->lane) ? ReadKind::Full : ReadKind::Meta;
      break;
    case RecordKind::Slice:
      entry.site_kind = ReadKind::Slice;
      entry.slice = log_.Slice(q.site.slice());
      break;
    default: entry.site_kind = ReadKind::Full; break;
    }
    entry.active_lane = q.active_lane ? q.active_lane->index : kNone;
    entry.site_lane = ref->lane.index;
    if (!Position(ref->pos, entry.site_pos)) return;
    // The request, deep-copied and interned (contracts copied only when non-empty).
    std::string key;
    if (q.memory) {
      key = "M";
      Put(key, q.memory->operand);
      Put(key, q.memory->contract.empty() ? std::string() : q.memory->contract.dump());
      Put(key, q.memory->bytes);
      Put(key, q.memory->space);
    } else if (q.tma) {
      key = "T";
      Put(key, q.tma->contract.empty() ? std::string() : q.tma->contract.dump());
      Put(key, q.tma->store ? 1 : 0);
    } else return Close("slot_without_request");
    const auto [found, added] = log_.request_ids.try_emplace(key, static_cast<std::uint32_t>(log_.requests.size()));
    if (added) {
      RequestCopy copy;
      copy.tma = q.tma != nullptr;
      if (q.memory) {
        copy.operand = std::string(q.memory->operand);
        copy.contract = q.memory->contract;
        copy.bytes = q.memory->bytes;
        copy.space = q.memory->space;
      } else {
        copy.contract = q.tma->contract;
        copy.store = q.tma->store;
      }
      log_.request_bytes += 2 * key.size() + 96;
      log_.requests.push_back(std::move(copy));
    }
    entry.request = found->second;
    entry.shape = log_.strings.Id(canon.shape);
    entry.offsets_begin = static_cast<std::uint32_t>(rank_.offsets.size());
    entry.offsets_count = static_cast<std::uint32_t>(canon.globals.size());
    for (const auto &[object, offset] : canon.globals) rank_.offsets.push_back({log_.strings.Id(object), offset});
    entry.shifts_begin = static_cast<std::uint32_t>(rank_.shifts.size());
    entry.shifts_count = static_cast<std::uint32_t>(canon.shifts.size());
    for (const auto &[object, value] : canon.shifts) rank_.shifts.push_back({log_.strings.Id(object), value});
    entry.line = i.line;
    rank_.slots.push_back(entry);
  }
  void Handle(const query::SlotMemory &e, ConsumerTag, bool, bool) { Slot(*e.query, *e.outcome); }
  void Handle(const query::SlotTma &e, ConsumerTag, bool, bool) { Slot(*e.query, *e.outcome); }
  void Handle(const query::MemoFill &e, ConsumerTag consumer, bool, bool) {
    if (Closed() || e.memo.kind != MemoKind::EvaluatedAddress) return;
    std::uint32_t pos;
    if (!Position(e.memo.at.pos, pos)) return;
    const auto key = std::get<0>(e.memo.key);
    fills_[{(std::uint64_t(consumer.resources) << 32) | e.memo.at.lane.index, key}] = pos;
  }
  void Handle(const query::MemoHit &e, ConsumerTag consumer, bool, bool) {
    if (Closed() || e.memo.kind != MemoKind::EvaluatedAddress) return;
    std::uint32_t pos;
    if (!Position(e.memo.at.pos, pos)) return;
    const auto key = std::get<0>(e.memo.key);
    const auto fill = fills_.find({(std::uint64_t(consumer.resources) << 32) | e.memo.at.lane.index, key});
    if (fill == fills_.end()) return Close("memo_hit_without_fill");
    rank_.memos.push_back({e.memo.at.lane.index, fill->second, pos, e.reg});
  }
  void Handle(const query::UniformBodySkip &, ConsumerTag, bool, bool) {}
  void Handle(const query::BindingRead &, ConsumerTag, bool, bool) {}
  void Handle(const query::AbsoluteCheck &e, ConsumerTag, bool, bool) {
    if (Closed()) return;
    if (!e.binding) return Close("absolute_check_without_binding");
    if (e.binding->address_space != "global") return; // δ = 0 outside global memory
    const auto object = log_.strings.Id(e.binding->allocation);
    if (align_seen_.insert({object, e.modulus}).second) rank_.aligns.push_back({object, e.modulus});
  }
  void Handle(const query::PartitionRead &e, ConsumerTag, bool, bool) {
    if (Closed()) return;
    if (e.end == query::PartitionEnd::Threw) return Close("partition_threw");
    if (!e.spec) return Close("partition_without_rule");
    std::uint32_t pos;
    if (!Position(e.leader.pos, pos)) return;
    if (e.evaluated > e.members.size() || e.classes.size() != e.evaluated || e.shuffles.size() != e.evaluated)
      return Close("partition_payload");
    rank_.partitions.push_back({e.leader.lane.index, pos, log_.Members(e.members), e.spec,
                                static_cast<std::uint32_t>(e.evaluated), Line(e.instruction), e.classes, e.shuffles});
  }
  void Handle(const query::ReportBearing &e, ConsumerTag, bool, bool) {
    if (e.kind == query::ReportKind::RenderedViolation) return; // re-rendered per member; does not close
    Close("lowering_diagnostic:" + std::string(e.code));
  }

  UnitLog &log_;
  RankLog &rank_;
  const CtaFacts &facts_;
  std::unordered_map<UniformKey, std::uint32_t, UniformKeyHash> uniform_seen_;
  std::unordered_map<LaneKey, std::uint32_t, LaneKeyHash> lane_seen_;
  std::unordered_map<std::uint32_t, std::uint32_t> special_seen_;
  std::unordered_map<std::pair<std::uint64_t, std::uint64_t>, std::uint32_t, PairHash> legacy_seen_;
  std::unordered_map<std::pair<std::uint64_t, std::uint64_t>, std::uint32_t, PairHash> fills_;
  std::unordered_set<std::pair<std::uint64_t, std::uint64_t>, PairHash> align_seen_;
};

// ---- member replay ----------------------------------------------------------
// The scratch Resources of one member CTA, one per logged (consumer, Resources
// id) as on the representative. The tables are never written (nothing is
// committed); the bridge fails, so a member resolution that would consult the
// bridge or the extractor reports a fallback.
class ReplayContexts {
public:
  explicit ReplayContexts(const CtaFacts &facts) : facts_(facts) {}
  Resources &At(Consumer consumer, std::uint16_t id) {
    auto &slot = contexts_[{static_cast<std::uint8_t>(consumer), id}];
    if (!slot) {
      slot = std::make_unique<Context>();
      slot->resources = std::make_unique<Resources>(
          facts_, facts_.frame(), slot->addresses, slot->barriers,
          SlotConfig{consumer, [](LaneRef, const RecordView &, std::string_view) -> std::optional<Json> {
                       return std::nullopt;
                     }});
    }
    return *slot->resources;
  }

private:
  struct Context {
    AddressTable addresses;
    BarrierTable barriers;
    std::unique_ptr<Resources> resources;
  };
  const CtaFacts &facts_;
  std::map<std::pair<std::uint8_t, std::uint16_t>, std::unique_ptr<Context>> contexts_;
};
// The registers a memory slot's resolution reads from its site record: the address
// operand's own register (MemoryBody, Shared and GenericOwner read only it there;
// definitions are other records).
std::vector<RegId> AddressRegisters(const CtaFacts &facts, std::string_view operand) {
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
// The member's record as the representative's consumer saw its own.
RecordView MemberRecord(const CtaFacts &facts, RecordRef at, ReadKind kind, std::span<const RegId> slice,
                        RecordHolder &holder) {
  switch (kind) {
  case ReadKind::Meta: return facts.Meta(at);
  case ReadKind::Slice: holder = facts.ReadSlice(at, slice); return holder.view();
  default: holder = facts.ReadFull(at); return holder.view();
  }
}
// δ = member - rep as a signed 64-bit value (false when it does not fit).
bool AddDelta(UnitDelta &delta, const std::string &object, std::uint64_t member, std::uint64_t rep) {
  constexpr auto kMax = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  std::int64_t d;
  if (member >= rep) {
    if (member - rep > kMax) return false;
    d = static_cast<std::int64_t>(member - rep);
  } else {
    if (rep - member > kMax) return false;
    d = -static_cast<std::int64_t>(rep - member);
  }
  const auto [it, added] = delta.try_emplace(object, d);
  return added || it->second == d;
}
// The same for signed offsets (a bounded-pointer footprint's terms).
bool AddSignedDelta(UnitDelta &delta, const std::string &object, std::int64_t member, std::int64_t rep) {
  std::int64_t d;
  if (__builtin_sub_overflow(member, rep, &d)) return false;
  const auto [it, added] = delta.try_emplace(object, d);
  return added || it->second == d;
}
// C2 (H1-H4, H6, H9) then C3 (H5) of one rank's log on the member CTA's own
// facts; δ accumulates across ranks. False with the first failing question.
bool Replay(const UnitLog &log, std::uint32_t rank, const CtaFacts &facts, UnitDelta &delta) {
  const auto &r = log.ranks.at(rank);
  try {
    for (const auto &e : r.specials) {
      if (!log.answers.Same(e.answer, facts.Special(log.strings.At(e.name)))) return false;
    }
    for (const auto &e : r.uniforms) {
      if (facts.PathUniformBinding({{e.lane}, e.pos}, e.reg) != e.present) return false;
    }
    for (const auto &e : r.legacy) {
      const auto &members = log.member_lists.at(e.members);
      if (!log.answers.Same(e.answer, facts.LegacyUniformOnlyValue({{e.lane}, e.pos}, members, log.strings.At(e.name))))
        return false;
    }
    for (const auto &e : r.memos) {
      const auto hit = facts.PathEvaluationKey({{e.lane}, e.hit}, e.reg);
      if (!hit || *hit > UINT32_MAX || hit != facts.PathEvaluationKey({{e.lane}, e.fill}, e.reg)) return false;
    }
    for (const auto &e : r.partitions) {
      const auto &members = log.member_lists.at(e.members);
      auto cursor = facts.BeginPartition(members, e.pos, *e.spec);
      for (std::uint32_t k = 0; k < e.evaluated; ++k) {
        MemberFacts derived;
        try {
          (void)cursor.Next(derived);
        } catch (...) {
          return false;
        }
        if (cursor.Class(k) != e.classes[k] || derived.shuffle != e.shuffles[k]) return false;
      }
    }
    // H1, one record read per (lane, pos, kind, slice) group (the log is sorted).
    // The log is lane-major: a lane's value caches are dropped once its reads
    // are done, so a replay holds one lane's memoized values at a time.
    // A Full group is read as a Slice over its own registers, the only ones compared:
    // those answers are the Full read's, without evaluating the record's other values.
    RecordHolder holder;
    std::optional<RecordView> view;
    const ValueEntry *group = nullptr;
    std::vector<RegId> group_regs;
    for (std::size_t k = 0; k < r.values.size(); ++k) {
      const auto &e = r.values[k];
      if (group && group->lane != e.lane) facts.ReleaseValues(LaneRef{group->lane});
      if (e.kind == ReadKind::At) {
        const auto answer = facts.ValueAt({{e.lane}, e.pos}, e.reg);
        if (!log.answers.Same(e.answer, answer ? &*answer : nullptr)) return false;
        continue;
      }
      if (!group || group->lane != e.lane || group->pos != e.pos || group->kind != e.kind || group->slice != e.slice) {
        group = &e;
        if (e.kind == ReadKind::Full) {
          group_regs.clear();
          for (std::size_t j = k; j < r.values.size() && r.values[j].lane == e.lane && r.values[j].pos == e.pos &&
                                  r.values[j].kind == e.kind && r.values[j].slice == e.slice; ++j)
            group_regs.push_back(r.values[j].reg);
          view = MemberRecord(facts, {{e.lane}, e.pos}, ReadKind::Slice, group_regs, holder);
        } else {
          const std::span<const RegId> slice =
              e.kind == ReadKind::Slice ? std::span<const RegId>(log.slices.at(e.slice)) : std::span<const RegId>();
          view = MemberRecord(facts, {{e.lane}, e.pos}, e.kind, slice, holder);
        }
      }
      if (!log.answers.Same(e.answer, facts.Value(*view, e.reg))) return false;
    }
  } catch (const std::exception &) {
    return false;
  }
  // C3: every logged slot re-run on the member, then compared up to δ.
  bool ok = true;
  try {
    ReplayContexts contexts(facts);
    // Slots follow lowering's lane order: as in H1, a lane's value caches are
    // dropped when the sites move to another lane (a return recomputes them).
    std::optional<std::uint32_t> site_lane;
    std::vector<RegId> site_regs;
    for (const auto &e : r.slots) {
      if (site_lane && *site_lane != e.site_lane) facts.ReleaseValues(LaneRef{*site_lane});
      site_lane = e.site_lane;
      auto &resources = contexts.At(e.consumer, e.resources);
      if (e.active_lane != kNone) resources.SetLane({e.active_lane});
      RecordHolder holder;
      const auto &request = log.requests.at(e.request);
      std::span<const RegId> slice =
          e.site_kind == ReadKind::Slice ? std::span<const RegId>(log.slices.at(e.slice)) : std::span<const RegId>();
      auto kind = e.site_kind;
      // A memory slot read Full is re-run on a Slice over its address registers:
      // the reads it makes there get the Full read's answers; any other read would
      // find nothing and change the outcome, failing the comparison, never passing it.
      if (e.memory && kind == ReadKind::Full) {
        site_regs = AddressRegisters(facts, request.operand);
        slice = site_regs;
        kind = ReadKind::Slice;
      }
      const auto site = MemberRecord(facts, {{e.site_lane}, e.site_pos}, kind, slice, holder);
      Canon canon;
      if (e.memory) {
        const MemoryRequest copy(request.operand, request.contract, request.bytes, request.space);
        Canonicalize(resources.ReplayMemory(site, copy), canon);
      } else {
        const TmaRequest copy(request.contract, request.store, nullptr);
        Canonicalize(resources.ReplayTma(site, copy), canon);
      }
      if (!canon.unliftable.empty() || canon.shape != log.strings.At(e.shape) || canon.globals.size() != e.offsets_count ||
          canon.shifts.size() != e.shifts_count) {
        ok = false;
        break;
      }
      for (std::uint32_t k = 0; k < e.offsets_count && ok; ++k) {
        const auto &rep = r.offsets[e.offsets_begin + k];
        const auto &object = log.strings.At(rep.object);
        if (canon.globals[k].first != object || !AddDelta(delta, object, canon.globals[k].second, rep.offset))
          ok = false;
      }
      for (std::uint32_t k = 0; k < e.shifts_count && ok; ++k) {
        const auto &rep = r.shifts[e.shifts_begin + k];
        const auto &object = log.strings.At(rep.object);
        if (canon.shifts[k].first != object || !AddSignedDelta(delta, object, canon.shifts[k].second, rep.value))
          ok = false;
      }
      if (!ok) break;
    }
  } catch (const std::exception &) {
    ok = false;
  }
  return ok;
}
bool AlignOk(const UnitLog &log, const UnitDelta &delta) {
  for (const auto &r : log.ranks)
    for (const auto &a : r.aligns) {
      if (!a.modulus) continue;
      const auto found = delta.find(log.strings.At(a.object));
      const std::int64_t d = found == delta.end() ? 0 : found->second;
      if (d % static_cast<std::int64_t>(a.modulus) != 0) return false;
    }
  return true;
}
} // namespace

// ---- the classes of one launch ----------------------------------------------
struct UnitClasses::Impl {
  const UnitPlan &plan;
  std::uint64_t budget;
  struct Candidate {
    std::uint32_t klass;
    UnitDelta delta;
  };
  struct Pending {
    std::unique_ptr<UnitLog> log;
    std::vector<std::unique_ptr<Recorder>> recorders;
    std::vector<StructureKey> keys;
    std::vector<Candidate> viable;
  };
  struct ClassInfo {
    std::uint32_t rep;
    std::vector<StructureKey> keys;
    std::shared_ptr<UnitLog> log; // null once closed (no admission needs it); never changes once set
    std::string closed;
    std::uint64_t log_bytes = 0;
    std::vector<std::uint32_t> members; // admitted units, in order
    std::vector<UnitDelta> deltas;
    std::vector<std::pair<std::string, std::uint64_t>> fences; // (I.5a) representative regions
    bool fence_nonconstant = false;
  };
  std::vector<ClassInfo> classes;
  // Per unit: its class and index in it (0 = representative), once decided.
  std::vector<std::optional<std::pair<std::uint32_t, std::uint32_t>>> membership;
  std::unordered_map<std::uint32_t, Pending> pending;
  std::uint64_t admitted = 0, class_bytes = 0;
  bool futility = false;
  // The classes with logs as of the last Found, for PreAdmit off the main thread.
  struct SnapshotClass {
    std::uint32_t klass;
    std::vector<StructureKey> keys;
    std::shared_ptr<const UnitLog> log;
  };
  struct Snapshot {
    std::uint32_t classes = 0;
    std::vector<SnapshotClass> logged;
  };
  mutable std::mutex snapshot_mutex;
  std::shared_ptr<const Snapshot> snapshot = std::make_shared<const Snapshot>();

  Impl(const UnitPlan &p, std::uint64_t b) : plan(p), budget(b), membership(p.units.size()) {}

  // The unit founds a class with its (sealed) log, unless the log is closed.
  void Found(std::uint32_t unit, Pending &p) {
    ClassInfo info;
    info.rep = unit;
    info.keys = std::move(p.keys);
    if (p.log) {
      for (const auto &r : p.log->ranks) {
        for (const auto &f : r.fences) info.fences.emplace_back(p.log->strings.At(f.object), f.offset);
        info.fence_nonconstant |= r.fence_nonconstant;
      }
      info.log_bytes = p.log->Bytes();
    }
    if (futility) info.closed = "futility";
    else if (!p.log) info.closed = "not_recorded";
    else if (class_bytes + info.log_bytes > budget) info.closed = "log_budget";
    else if (!p.log->closed.empty()) info.closed = p.log->closed;
    else {
      info.log = std::move(p.log);
      class_bytes += info.log_bytes;
    }
    membership.at(unit) = {static_cast<std::uint32_t>(classes.size()), 0};
    classes.push_back(std::move(info));
    if (admitted == 0 && classes.size() >= 16) futility = true;
    // Publish the class (its keys copied once per founded class).
    std::shared_ptr<const Snapshot> previous;
    {
      std::lock_guard lock(snapshot_mutex);
      previous = snapshot;
    }
    auto next = std::make_shared<Snapshot>(*previous);
    next->classes = static_cast<std::uint32_t>(classes.size());
    if (classes.back().log)
      next->logged.push_back({next->classes - 1, classes.back().keys, classes.back().log});
    std::lock_guard lock(snapshot_mutex);
    snapshot = std::move(next);
  }
  void Join(std::uint32_t unit, std::uint32_t klass, UnitDelta delta) {
    ++admitted;
    classes[klass].members.push_back(unit);
    classes[klass].deltas.push_back(std::move(delta));
    membership.at(unit) = {klass, static_cast<std::uint32_t>(classes[klass].members.size())};
  }
  // (I.5) over every unit's cp_fenceproxy regions (representative + δ for members).
  struct Region {
    std::string object;
    std::uint64_t begin;
    std::uint32_t unit;
  };
  struct LocalityDetail {
    bool holds = true, constant = true, unknown = false, cross_overlap = false, input_overlap = false;
  };
  LocalityDetail Locality() const {
    LocalityDetail result;
    if (!plan.precondition.tensormap_copies) return result;
    std::vector<Region> regions;
    for (const auto &c : classes) {
      if (c.fence_nonconstant) result.constant = false;
      for (const auto &[object, offset] : c.fences) {
        regions.push_back({object, offset, c.rep});
        for (std::size_t m = 0; m < c.members.size(); ++m) {
          const auto found = c.deltas[m].find(object);
          if (found == c.deltas[m].end()) {
            result.unknown = true; // never expected: the member's own slot fixes δ
            continue;
          }
          regions.push_back({object, std::uint64_t(std::int64_t(offset) + found->second), c.members[m]});
        }
      }
      if (c.closed == "futility" || c.closed == "log_budget" || c.closed == "not_recorded") result.unknown = true;
    }
    // (I.5b): no two units' 128-byte tensormap regions overlap; (I.5c): none
    // overlaps an input map's location (an absolute address cannot be placed).
    std::sort(regions.begin(), regions.end(), [](const Region &a, const Region &b) {
      return std::tie(a.object, a.begin, a.unit) < std::tie(b.object, b.begin, b.unit);
    });
    for (std::size_t i = 0; i + 1 < regions.size(); ++i)
      for (std::size_t j = i + 1; j < regions.size() && regions[j].object == regions[i].object &&
                                  regions[j].begin < regions[i].begin + 128;
           ++j)
        if (regions[j].unit != regions[i].unit) result.cross_overlap = true;
    if (!regions.empty() && !plan.input_maps.empty()) {
      std::vector<TensormapRegion> placed;
      placed.reserve(regions.size());
      for (const auto &region : regions) placed.push_back({region.object, region.begin});
      result.input_overlap = TensormapInputOverlap(placed, plan.input_maps);
    }
    result.holds = result.constant && !result.unknown && !result.cross_overlap && !result.input_overlap;
    return result;
  }
};

UnitClasses::UnitClasses(const UnitPlan &plan, std::uint64_t log_budget_bytes)
    : impl_(std::make_unique<Impl>(plan, log_budget_bytes)) {}
UnitClasses::~UnitClasses() = default;

// ---- class mode -------------------------------------------------------------
UnitClasses::PreAdmission UnitClasses::PreAdmit(std::span<const CtaFacts *const> facts,
                                                const std::vector<StructureKey> &keys) const {
  const auto &s = *impl_;
  std::shared_ptr<const Impl::Snapshot> snapshot;
  {
    std::lock_guard lock(s.snapshot_mutex);
    snapshot = s.snapshot;
  }
  PreAdmission result;
  result.classes = snapshot->classes;
  const auto size = s.plan.UnitSize();
  if (facts.size() != size || keys.size() != size) throw std::logic_error("pre-admission needs every rank");
  for (const auto &candidate : snapshot->logged) {
    bool ok = candidate.keys.size() == size;
    for (std::uint32_t k = 0; ok && k < size; ++k)
      ok = candidate.keys[k].hash == keys[k].hash && candidate.keys[k].bytes == keys[k].bytes;
    if (!ok) continue;
    UnitDelta delta;
    for (std::uint32_t rank = 0; ok && rank < size; ++rank) ok = Replay(*candidate.log, rank, *facts[rank], delta);
    if (ok && !AlignOk(*candidate.log, delta)) ok = false;
    if (ok) {
      result.joined = candidate.klass;
      result.delta = std::move(delta);
      return result;
    }
  }
  return result;
}

std::optional<std::uint32_t> UnitClasses::Admit(std::uint32_t unit, std::span<const CtaFacts *const> facts,
                                                std::vector<StructureKey> keys, std::optional<PreAdmission> pre) {
  auto &s = *impl_;
  const auto size = s.plan.UnitSize();
  if (facts.size() != size) throw std::logic_error("class-mode admission needs every rank of the unit");
  if (!keys.empty() && keys.size() != size) throw std::logic_error("class-mode admission needs every rank's key");
  auto &p = s.pending[unit];
  p.keys = std::move(keys);
  if (p.keys.empty()) {
    p.keys.reserve(size);
    for (const auto *f : facts) p.keys.push_back(ComputeStructureKey(*f));
  }
  // Classes [0, pre->classes) are decided by the look-ahead: the first of them that admits
  // the unit is the sequential first (earlier ones fail on these facts either way).
  std::uint32_t first = 0;
  if (pre) {
    if (pre->classes > s.classes.size()) throw std::logic_error("pre-admission saw a later class");
    if (pre->joined) {
      s.Join(unit, *pre->joined, std::move(pre->delta));
      s.pending.erase(unit);
      return pre->joined;
    }
    first = pre->classes;
  }
  // C1: candidate classes in creation order (hash first, then bytes, every rank).
  for (std::uint32_t c = first; c < s.classes.size(); ++c) {
    const auto &keys = s.classes[c].keys;
    bool same = keys.size() == size;
    for (std::uint32_t k = 0; same && k < size; ++k)
      same = keys[k].hash == p.keys[k].hash && keys[k].bytes == p.keys[k].bytes;
    if (!same) continue;
    if (s.classes[c].log) p.viable.push_back({c, {}});
  }
  std::optional<std::uint32_t> joined;
  if (!p.viable.empty()) {
    for (auto &candidate : p.viable) {
      bool ok = true;
      for (std::uint32_t rank = 0; ok && rank < size; ++rank)
        ok = Replay(*s.classes[candidate.klass].log, rank, *facts[rank], candidate.delta);
      if (ok && !AlignOk(*s.classes[candidate.klass].log, candidate.delta)) ok = false; // H8 (AbsoluteCheck)
      // Class mode with values: value lifting adds no admission condition.
      // The only modulus a value stage applies to an absolute global offset
      // is an audited row width R (a row map's row width), known only after
      // value analysis; run.cpp then
      // re-checks every member's δ against it and restarts per unit on a
      // violation. The offset bounds are checked where δ is applied.
      if (ok) {
        joined = candidate.klass;
        s.Join(unit, candidate.klass, std::move(candidate.delta));
        break;
      }
    }
  }
  if (joined) s.pending.erase(unit);
  else p.viable.clear();
  return joined;
}

QuerySink *UnitClasses::Record(std::uint32_t unit, std::uint32_t rank, const CtaFacts &facts) {
  auto &s = *impl_;
  auto &p = s.pending.at(unit);
  const auto size = s.plan.UnitSize();
  if (p.recorders.empty()) p.recorders.resize(size);
  if (s.futility) return nullptr; // futile: no new log
  if (!p.log) p.log = std::make_unique<UnitLog>(size);
  p.recorders[rank] = std::make_unique<Recorder>(*p.log, rank, facts);
  return p.recorders[rank].get();
}

void UnitClasses::Recorded(std::uint32_t unit, std::uint32_t rank) {
  auto &s = *impl_;
  auto &p = s.pending.at(unit);
  if (rank < p.recorders.size() && p.recorders[rank]) {
    p.recorders[rank]->Seal();
    p.recorders[rank].reset();
  }
  if (rank + 1 == s.plan.UnitSize()) {
    s.Found(unit, p);
    s.pending.erase(unit);
  }
}

algorithm::UnitQuotient UnitClasses::Quotient() const {
  const auto &s = *impl_;
  algorithm::UnitQuotient q;
  q.grid = s.plan.grid;
  q.unit_shape = s.plan.unit_shape;
  q.units.reserve(s.plan.units.size());
  for (std::uint32_t u = 0; u < s.plan.units.size(); ++u) {
    const auto &unit = s.plan.units[u];
    algorithm::UnitQuotient::Unit entry;
    entry.kappa = unit.kappa;
    const auto &membership = s.membership.at(u);
    if (!membership) throw std::logic_error("unit without a class");
    entry.klass = membership->first;
    for (const auto &cta : unit.ctas) entry.ctas.push_back(q.Linear(cta));
    q.units.push_back(std::move(entry));
  }
  q.classes.reserve(s.classes.size());
  for (const auto &c : s.classes) {
    algorithm::UnitQuotient::Class entry;
    entry.units.push_back(c.rep);
    entry.units.insert(entry.units.end(), c.members.begin(), c.members.end());
    entry.delta.emplace_back();
    entry.delta.insert(entry.delta.end(), c.deltas.begin(), c.deltas.end());
    q.classes.push_back(std::move(entry));
  }
  return q;
}

UnitClasses::Locality UnitClasses::TensormapLocality() const {
  const auto detail = impl_->Locality();
  return {detail.holds, detail.input_overlap};
}
} // namespace sentryward::frontend::ptx
