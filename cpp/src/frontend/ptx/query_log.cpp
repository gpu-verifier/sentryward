#include "sentryward/frontend/ptx/query_log.h"

#include "sentryward/frontend/ptx/resources.h"

#include <algorithm>
#include <atomic>
#include <set>
#include <cstring>
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
  std::uint32_t Id(const Scalar *answer) {
    if (!answer) return kNone;
    // A table atom answers as its plain value: every consumer reads that one (ScalarInteger).
    const Scalar plain = PlainValue(*answer);
    const Scalar *value = &plain;
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
    return values_.at(id) == PlainValue(*value);
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
  std::uint32_t valued_begin, valued_count;
  // A TMA load's read-relaxed form (CanonicalizeTmaRead; a string id), kNone otherwise.
  std::uint32_t relaxed;
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
  std::vector<SignedOffset> valued; // Canon::valued
  // For the tensormap locality rule: this CTA's tensormap.cp_fenceproxy global destinations.
  std::vector<GlobalOffset> fences;
  bool fence_nonconstant = false;
  // Containment: the (lane, pos) of every query::RegisterTouch that overwrote a pending register
  // (`touches`) or read one (`read_touches`), and whether lowering rendered a per-CTA violation
  // (re-rendered for members: a containment member need not have it).
  std::vector<std::pair<std::uint32_t, std::uint32_t>> touches, read_touches;
  bool rendered = false;
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
               (r.shifts.size() + r.valued.size()) * sizeof(SignedOffset) + r.fences.size() * sizeof(GlobalOffset);
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
  // A table-atom footprint's or region's exact offset (object, value): equal to the representative's
  // up to one δ per allocation (with `globals`) for value lifting only, never for admission.
  std::vector<std::pair<std::string, std::int64_t>> valued;
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
// An allocation key after its kind (a writer that encodes the kind puts it first).
void PutKey(std::string &out, const AllocationKey &key) {
  Put(out, key.space);
  Put(out, key.owner.value);
  Put(out, key.thread);
  Put(out, key.object);
  Put(out, key.alias);
}
void PutStep(Canon &canon, const InternStep &step) {
  auto &out = canon.shape;
  std::visit(
      [&](const auto &s) {
        using T = std::decay_t<decltype(s)>;
        if constexpr (std::is_same_v<T, ConstantStep>) {
          out += 'C';
          Put(out, std::uint64_t(s.key.kind));
          PutKey(out, s.key);
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
// An affine form in tensor-property elements: its structure (element objects, types, coefficients)
// in the shape; its constant (the δ of `object`) and each element read's byte offset (its input
// object's δ) as shifts.
void PutAffine(Canon &canon, const std::string &object, const AbstractAffine &affine) {
  auto &out = canon.shape;
  out += 'A';
  Put(out, std::uint64_t(affine.terms.size()));
  canon.shifts.emplace_back(object, affine.constant);
  for (const auto &term : affine.terms) {
    Put(out, term.object);
    Put(out, term.element_type);
    Put(out, std::to_string(term.coefficient));
    // An element function: its operator tree and bounds (functions of shifted elements).
    Put(out, term.function);
    Put(out, std::to_string(term.lower));
    Put(out, std::to_string(term.upper));
    canon.shifts.emplace_back(term.object, term.offset);
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
    PutKey(out, s->key);
  }
  out += 'F';
  Put(out, std::uint64_t(resolved.kind));
  Put(out, resolved.bytes);
  PutAffine(canon, base.key.object, *resolved.affine);
  if (resolved.valued) canon.valued.emplace_back(base.key.object, *resolved.valued);
  return true;
}
// A TMA terminal's pairs and completions.
void PutTmaTail(std::string &out, const ResolvedTma &resolved) {
  out += '|';
  for (const auto &pair : resolved.pairs) {
    Put(out, pair.global ? *pair.global + 1 : 0);
    Put(out, pair.shared);
    Put(out, pair.bytes);
    Put(out, pair.logical_offset);
    Put(out, pair.target_rank);
    // An abstract endpoint's place, tagged by its presence (pairs are not counted): its
    // offset and the region's bounds proof (TmaBody).
    Put(out, pair.place ? 1 : 0);
    if (pair.place) {
      Put(out, pair.place->region);
      Put(out, pair.place->offset ? *pair.place->offset + 1 : 0);
      Put(out, pair.place->in_bounds ? 1 : 0);
    }
  }
  out += '|';
  for (const auto &[step, bytes] : resolved.completions) {
    Put(out, step);
    Put(out, bytes);
  }
}
// A TMA region whose symbolic coordinates are all affine in tensor-property elements
// (ResolvedTma::region_affine): each region's expression and patch are the region form, its
// placement (offset 0 of the target allocation) an anchor; the other steps (shared bindings,
// completions) as usual. The structure (element objects, types, byte coefficients)
// must be equal; the region's constant (the allocation's δ) and each element read's byte
// offset (its object's δ) are compared up to δ. A gather has one form per row: a segment's
// row is fixed by its logical offset (PutTmaTail). False (canon unchanged): not this form.
bool CanonicalizeAffineTma(const ResolvedTma &resolved, Canon &canon) {
  if (resolved.region_affine.empty()) return false;
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
      PutKey(out, placement->key);
      k += 2;
      continue;
    }
    if (std::holds_alternative<PatchStep>(step)) return false;
    PutStep(local, step);
  }
  if (object.empty() || !local.unliftable.empty()) return false;
  for (const auto &form : resolved.region_affine) PutAffine(local, object, form);
  for (const auto &value : resolved.region_valued)
    if (value) local.valued.emplace_back(object, *value);
  PutTmaTail(out, resolved);
  canon = std::move(local);
  return true;
}
// An extractor-resolved region expression made relative to the executing CTA `self` (its
// coordinates as the frame writes them, [x, y, z]). Exactly three fields of such an expression name
// a CTA, each the executing one: the region's occurrence (TmaBody: frame.Json(frame.self())), an
// unextracted coordinate's provenance (the same), and a value-DAG Load's occurrence (the
// extractor's BeginOrigin: its frame's SelfCoordinates; the extractor reads only that CTA's
// facts). Each must name `self` and becomes "self"; any other CTA, field or form: false. Every
// other field stays as it is, so equal results are equal expressions up to the executing CTA: a
// BlockIndex binding keeps its κ, a Load its address, a lookup its element. O(expression size).
bool SelfRelativeRegion(Json &region, const Json &self) {
  if (!region.is_object() || region.value("operator", std::string()) != "tma_region") return false;
  for (const auto &item : region.items())
    if (item.key() != "operator" && item.key() != "descriptor" && item.key() != "coordinates" &&
        item.key() != "occurrence")
      return false;
  const auto occurrence = region.find("occurrence");
  const auto coordinates = region.find("coordinates");
  if (occurrence == region.end() || !occurrence->is_array() || occurrence->empty() || (*occurrence)[0] != self ||
      coordinates == region.end() || !coordinates->is_array())
    return false;
  (*occurrence)[0] = "self";
  for (auto &coordinate : *coordinates) {
    // A number or an affine form in lookups (AffineOffsetJson): no CTA field, compared as it is.
    if (!coordinate.is_object() || coordinate.value("operator", std::string()) != "opaque") continue;
    const auto provenance = coordinate.find("provenance");
    if (provenance == coordinate.end() || !provenance->is_array() || provenance->size() < 2 ||
        !(*provenance)[0].is_string())
      return false;
    const auto kind = (*provenance)[0].get<std::string>();
    if (kind == "tma_coordinate") {
      if ((*provenance)[1] != self) return false;
      (*provenance)[1] = "self";
      continue;
    }
    auto &dag = (*provenance)[1];
    if (kind != "tma_signed_s32" || !dag.is_object() || dag.value("operator", std::string()) != "value_dag" ||
        !dag.contains("nodes") || !dag["nodes"].is_array())
      return false;
    for (auto &node : dag["nodes"]) {
      if (!node.is_object()) return false;
      const auto load = node.find("occurrence");
      if (load == node.end()) continue;
      if (!load->is_array() || load->size() < 3 || (*load)[2] != self) return false;
      (*load)[2] = "self";
    }
  }
  return true;
}
// A TMA region whose symbolic coordinates the address-value extractor resolved (TmaBody's only
// fallback): an opaque region, a box of its allocation at offsets that only data determine. Each
// region's expression, placement and patch are the region form: the expression (descriptor,
// coordinate expressions, occurrence) is compared exactly up to the executing CTA
// (SelfRelativeRegion), the placement must be offset 0 of a Plain global allocation and is that
// allocation's δ answer (0: the expression has no representative-absolute term a δ would move).
// The other steps (shared bindings, completions) are as usual. Between units only the data that
// the expressions read may differ; the cross-unit pass places such a region by its storage alone
// (cross_unit_races.h). False (canon unchanged): not this form. O(steps + expression size).
bool CanonicalizeOpaqueTma(const ResolvedTma &resolved, const Json &self, Canon &canon) {
  if (!resolved.region_affine.empty()) return false;
  Canon local = canon;
  auto &out = local.shape;
  out += 'Q';
  Put(out, resolved.store ? 1 : 0);
  Put(out, resolved.descriptor);
  bool region = false;
  const auto &steps = resolved.steps;
  for (std::size_t k = 0; k < steps.size(); ++k) {
    const auto &step = steps[k].step;
    if (const auto *expression = std::get_if<ExpressionStep>(&step)) {
      if (expression->space != "global" || !expression->alias.empty() || expression->expression ||
          k + 2 >= steps.size())
        return false;
      const auto *literal = std::get_if<std::string>(&expression->allocation);
      const auto *placement = std::get_if<ConstantStep>(&steps[k + 1].step);
      const auto *patch = std::get_if<PatchStep>(&steps[k + 2].step);
      if (!literal || *literal != "unresolved" || !placement || !patch || patch->target != k ||
          patch->offset != placement->offset || patch->expression != expression->serialized ||
          placement->offset != 0 || placement->key.kind != AllocationKey::Kind::Plain ||
          placement->key.space != "global")
        return false;
      auto form = Json::parse(expression->serialized, nullptr, false);
      if (form.is_discarded() || !SelfRelativeRegion(form, self)) return false;
      out += 'X';
      PutKey(out, placement->key);
      Put(out, form.dump());
      local.globals.emplace_back(placement->key.object, placement->offset);
      region = true;
      k += 2;
      continue;
    }
    if (std::holds_alternative<PatchStep>(step)) return false;
    PutStep(local, step);
  }
  if (!region || !local.unliftable.empty()) return false;
  PutTmaTail(out, resolved);
  canon = std::move(local);
  return true;
}
// A TMA load's outcome without where it reads in global memory (containment admission of reads of
// allocations that no unit writes, ContainMap): the request (descriptor), every non-global step
// (shared destinations with their offsets, barriers) exactly, the global allocations it reads by name
// only, each pair's shared destination step, byte count, logical offset and target rank, and the
// completions (barrier step, bytes). nullopt for a store, an error terminal, a fallback other than the
// extractor's, or a step list this form does not describe. O(steps + pairs).
struct TmaRead {
  std::string shape;
  std::set<std::string> objects; // the global allocations read
};
std::optional<TmaRead> CanonicalizeTmaRead(const SlotOutcome<ResolvedTma> &outcome) {
  if (outcome.fallback != FallbackUse::None && outcome.fallback != FallbackUse::Extractor) return std::nullopt;
  const auto *resolved = std::get_if<0>(&outcome.terminal);
  if (!resolved || resolved->store) return std::nullopt;
  TmaRead read;
  Canon local;
  auto &out = local.shape;
  out += 'R';
  Put(out, resolved->descriptor);
  const auto &steps = resolved->steps;
  std::vector<std::string> step_form(steps.size()); // a non-global step's canonical text
  std::vector<bool> global(steps.size());
  for (std::size_t k = 0; k < steps.size(); ++k) {
    const auto &step = steps[k].step;
    if (const auto *expression = std::get_if<ExpressionStep>(&step)) {
      if (expression->space != "global" || k + 2 >= steps.size()) return std::nullopt;
      const auto *placement = std::get_if<ConstantStep>(&steps[k + 1].step);
      const auto *patch = std::get_if<PatchStep>(&steps[k + 2].step);
      if (!placement || !patch || patch->target != k || placement->key.kind != AllocationKey::Kind::Plain ||
          placement->key.space != "global")
        return std::nullopt;
      read.objects.insert(placement->key.object);
      global[k] = global[k + 1] = global[k + 2] = true;
      out += 'X';
      Put(out, placement->key.object);
      k += 2;
      continue;
    }
    if (std::holds_alternative<PatchStep>(step)) return std::nullopt;
    if (const auto *constant = std::get_if<ConstantStep>(&step);
        constant && constant->key.kind == AllocationKey::Kind::Plain && constant->key.space == "global") {
      read.objects.insert(constant->key.object);
      global[k] = true;
      out += 'X';
      Put(out, constant->key.object);
      continue;
    }
    Canon one;
    PutStep(one, step);
    if (!one.unliftable.empty() || !one.globals.empty()) return std::nullopt;
    step_form[k] = std::move(one.shape);
    out += step_form[k];
  }
  if (read.objects.empty()) return std::nullopt;
  out += '|';
  for (const auto &pair : resolved->pairs) {
    if (pair.shared >= steps.size() || global[pair.shared] || (pair.global && *pair.global >= steps.size()) ||
        (pair.global && !global[*pair.global]))
      return std::nullopt;
    Put(out, pair.global ? 1 : 0);
    Put(out, step_form[pair.shared]);
    Put(out, pair.bytes);
    Put(out, pair.logical_offset);
    Put(out, pair.target_rank);
  }
  out += '|';
  for (const auto &[step, bytes] : resolved->completions) {
    if (step >= steps.size() || global[step]) return std::nullopt;
    Put(out, step_form[step]);
    Put(out, bytes);
  }
  read.shape = std::move(out);
  return read;
}
// `self`: the executing CTA's coordinates ([x, y, z]), for the opaque TMA form.
template <class T> void Canonicalize(const SlotOutcome<T> &outcome, const Json &self, Canon &canon) {
  if constexpr (std::is_same_v<T, ResolvedTma>) {
    if (const auto *resolved = std::get_if<0>(&outcome.terminal);
        resolved && outcome.fallback == FallbackUse::Extractor) {
      canon.shape += 'P';
      if (CanonicalizeOpaqueTma(*resolved, self, canon)) return;
    }
  }
  if (outcome.fallback != FallbackUse::None) canon.unliftable = "slot_fallback";
  canon.shape += 'P';
  if constexpr (std::is_same_v<T, ResolvedFootprint>) {
    if (const auto *resolved = std::get_if<0>(&outcome.terminal);
        resolved && CanonicalizeAffine(outcome.prefix, *resolved, canon))
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
    PutTmaTail(canon.shape, resolved);
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
  std::uint64_t a, b; // (lane, pos), (reg, slice)
  std::uint8_t kind;
  bool operator==(const LaneKey &) const = default;
};
struct LaneKeyHash {
  std::size_t operator()(const LaneKey &k) const noexcept { return Mix(k.a ^ Mix(k.b ^ Mix(k.kind))); }
};
struct PairHash {
  std::size_t operator()(const std::pair<std::uint64_t, std::uint64_t> &k) const noexcept {
    return Mix(k.first ^ Mix(k.second));
  }
};

class Recorder final : public QuerySink {
public:
  Recorder(UnitLog &log, std::uint32_t rank, const CtaFacts &facts)
      : log_(log), rank_(log.ranks.at(rank)), facts_(facts) {}
  void On(const QueryEvent &event, ConsumerTag consumer, bool in_slot, bool in_partition) noexcept override {
    try {
      if (consumer.kind == Consumer::Driver || consumer.kind == Consumer::Executor) return;
      std::visit([&](const auto &e) { Handle(e, consumer, in_slot, in_partition); }, event);
    } catch (...) {
      Close("recorder_failure");
    }
  }
  // The log is complete: entries in replay order.
  void Seal() {
    std::sort(rank_.values.begin(), rank_.values.end(), [](const ValueEntry &a, const ValueEntry &b) {
      return std::tie(a.lane, a.pos, a.kind, a.slice, a.reg) < std::tie(b.lane, b.pos, b.kind, b.slice, b.reg);
    });
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
  // Whether `key` is asked for the first time; a repeated question with another
  // answer closes the log.
  template <class Seen> bool First(Seen &seen, const typename Seen::key_type &key, std::uint32_t answer) {
    const auto [it, added] = seen.try_emplace(key, answer);
    if (!added && it->second != answer) Close("inconsistent_answer");
    return added;
  }
  // How a member re-reads `view`, a positioned record of `lane` that is not
  // Detached; a Slice's registers are interned into `slice`.
  ReadKind KindOf(const RecordView &view, LaneRef lane, std::uint32_t &slice) {
    switch (view.kind()) {
    case RecordKind::Prototype: return ReadKind::Meta;
    case RecordKind::PrototypeCopy: return facts_.Expanded(lane) ? ReadKind::Full : ReadKind::Meta;
    case RecordKind::Slice: slice = log_.Slice(view.slice()); return ReadKind::Slice;
    default: return ReadKind::Full;
    }
  }
  void Value(RecordRef at, RegId reg, ReadKind kind, std::uint32_t slice, std::uint32_t answer) {
    std::uint32_t pos;
    if (!Position(at.pos, pos)) return;
    if (kind == ReadKind::Meta) {
      // Path-uniform: one lane of the path answers for all of them.
      if (!First(uniform_seen_, {facts_.Path(at.lane), pos, reg, 0}, answer)) return;
    } else {
      const LaneKey key{(std::uint64_t(at.lane.index) << 32) | pos, (std::uint64_t(reg) << 32) | slice,
                        static_cast<std::uint8_t>(kind)};
      if (!First(lane_seen_, key, answer)) return;
    }
    rank_.values.push_back({at.lane.index, pos, reg, slice, kind, answer});
  }
  void Handle(const query::ValueRead &e, ConsumerTag, bool in_slot, bool in_partition) {
    if (in_slot || in_partition || Closed()) return; // covered by the slot re-run / the PartitionRead classes
    const auto ref = e.view.ref();
    if (!ref) {
      // An unpositioned view is an empty holder's default record, which binds
      // nothing; any answer from one cannot be re-asked on a member.
      if (e.answer) Close("unpositioned_read");
      return;
    }
    if (e.view.kind() == RecordKind::Detached) return Close("detached_read");
    std::uint32_t slice = 0;
    const auto kind = KindOf(e.view, ref->lane, slice);
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
    if (!First(legacy_seen_, {(std::uint64_t(entry.lane) << 32) | pos, (std::uint64_t(entry.members) << 32) | entry.name},
               entry.answer))
      return;
    rank_.legacy.push_back(entry);
  }
  void Handle(const query::UniformBinding &e, ConsumerTag, bool in_slot, bool in_partition) {
    if (in_slot || in_partition || Closed()) return;
    std::uint32_t pos;
    if (!Position(e.at.pos, pos)) return;
    if (!First(uniform_seen_, {facts_.Path(e.at.lane), pos, e.reg, 1}, e.present ? 1u : 0u)) return;
    rank_.uniforms.push_back({e.at.lane.index, pos, e.reg, e.present});
  }
  void Handle(const query::SpecialRead &e, ConsumerTag, bool in_slot, bool in_partition) {
    if (in_slot || in_partition || Closed()) return;
    const auto name = log_.strings.Id(e.name);
    const auto answer = log_.answers.Id(e.answer);
    if (!First(special_seen_, name, answer)) return;
    rank_.specials.push_back({name, answer});
  }
  void Handle(const query::SlotBegin &, ConsumerTag, bool, bool) {}
  void Handle(const query::SlotEnd &, ConsumerTag, bool, bool) {}
  template <class T> void Slot(const SlotQuery &q, const SlotOutcome<T> &outcome) {
    const bool memory = std::is_same_v<T, ResolvedFootprint>;
    const auto instruction = q.site.instruction_id();
    const auto &i = facts_.function().instructions.at(instruction);
    // A tensormap.cp_fenceproxy destination (lowering's 128-byte global Memory slot), kept for
    // the tensormap locality rule.
    const bool fence = memory && q.memory && q.memory->space == "global" && q.memory->bytes == 128 &&
                       i.opcode == "tensormap" && !i.modifiers.empty() && i.modifiers[0] == "cp_fenceproxy";
    if (Closed() && !fence) return;
    Canon canon;
    Canonicalize(outcome, Json(facts_.frame().SelfCoordinates()), canon);
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
    entry.site_kind = KindOf(q.site, ref->lane, entry.slice);
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
    entry.valued_begin = static_cast<std::uint32_t>(rank_.valued.size());
    entry.valued_count = static_cast<std::uint32_t>(canon.valued.size());
    for (const auto &[object, value] : canon.valued) rank_.valued.push_back({log_.strings.Id(object), value});
    entry.line = i.line;
    entry.relaxed = kNone;
    if constexpr (std::is_same_v<T, ResolvedTma>)
      if (const auto read = CanonicalizeTmaRead(outcome)) entry.relaxed = log_.strings.Id(read->shape);
    rank_.slots.push_back(entry);
  }
  void Handle(const query::SlotMemory &e, ConsumerTag, bool, bool) { Slot(*e.query, *e.outcome); }
  void Handle(const query::SlotTma &e, ConsumerTag, bool, bool) { Slot(*e.query, *e.outcome); }
  void Handle(const query::MemoFill &e, ConsumerTag consumer, bool, bool) {
    if (Closed()) return;
    std::uint32_t pos;
    if (!Position(e.memo.at.pos, pos)) return;
    const auto key = e.memo.key;
    fills_[{(std::uint64_t(consumer.resources) << 32) | e.memo.at.lane.index, key}] = pos;
  }
  void Handle(const query::MemoHit &e, ConsumerTag consumer, bool, bool) {
    if (Closed()) return;
    std::uint32_t pos;
    if (!Position(e.memo.at.pos, pos)) return;
    const auto key = e.memo.key;
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
    if (e.kind == query::ReportKind::RenderedViolation) { // re-rendered per member; does not close
      rank_.rendered = true;
      return;
    }
    Close("lowering_diagnostic:" + std::string(e.code));
  }
  void Handle(const query::RegisterTouch &e, ConsumerTag, bool, bool) {
    std::uint32_t pos;
    if (Closed() || !Position(e.at.pos, pos)) return;
    if (e.writes) rank_.touches.emplace_back(e.at.lane.index, pos);
    if (e.reads) rank_.read_touches.emplace_back(e.at.lane.index, pos);
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
// committed); the bridge fails, so a member memory resolution that would consult
// the bridge or the extractor reports a fallback (never liftable). A TMA's
// symbolic coordinates go to the member's own extractor (TmaBody), whose opaque
// region form is compared exactly (CanonicalizeOpaqueTma).
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
// The member's record as the representative's consumer saw its own.
RecordView MemberRecord(const CtaFacts &facts, RecordRef at, ReadKind kind, std::span<const RegId> slice,
                        RecordHolder &holder) {
  switch (kind) {
  case ReadKind::Meta: return facts.Meta(at);
  case ReadKind::Slice: holder = facts.ReadSlice(at, slice); return holder.view();
  default: holder = facts.ReadFull(at); return holder.view();
  }
}
// The δ answers of one unit's replay, per global allocation. Without `piecewise`, two answers for
// one allocation must agree (the first disagreement fails the replay). With it, the allocation's
// (site, δ) entries in replay order, without consecutive repeats (the globals of one slot share its
// site). O(log allocations) per answer; O(allocations) memory, plus O(distinct runs) with `piecewise`.
class DeltaAnswers {
public:
  explicit DeltaAnswers(bool piecewise) : piecewise_(piecewise) {}
  // δ = member - rep as a signed 64-bit value at `site` (false when it does not fit, or disagrees).
  bool Add(const std::string &object, const algorithm::UnitSite &site, std::uint64_t member, std::uint64_t rep) {
    constexpr auto kMax = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    std::int64_t d;
    if (member >= rep) {
      if (member - rep > kMax) return false;
      d = static_cast<std::int64_t>(member - rep);
    } else {
      if (rep - member > kMax) return false;
      d = -static_cast<std::int64_t>(rep - member);
    }
    return Put(object, site, d);
  }
  // Without `piecewise`: the δ this replay already holds for `object`, if any. O(log allocations).
  [[nodiscard]] std::optional<std::int64_t> Held(const std::string &object) const {
    const auto found = answers_.find(object);
    if (found == answers_.end() || found->second.empty()) return std::nullopt;
    return found->second.front().second;
  }
  // The same for signed offsets (a bounded-pointer footprint's terms).
  bool AddSigned(const std::string &object, const algorithm::UnitSite &site, std::int64_t member, std::int64_t rep) {
    std::int64_t d;
    if (__builtin_sub_overflow(member, rep, &d)) return false;
    return Put(object, site, d);
  }
  // The unit's δ table: an allocation whose answers all agree has one δ; otherwise (`piecewise`
  // only) its sorted (site, δ) list, each site with one δ. Nullopt when a site has two δ.
  // O(entries log entries).
  [[nodiscard]] std::optional<UnitDelta> Table() && {
    UnitDelta table;
    for (auto &[object, entries] : answers_) {
      const auto d = entries.front().second;
      if (std::all_of(entries.begin(), entries.end(), [&](const auto &e) { return e.second == d; })) {
        table.uniform.emplace(object, d);
        continue;
      }
      std::sort(entries.begin(), entries.end());
      entries.erase(std::unique(entries.begin(), entries.end()), entries.end());
      for (std::size_t k = 1; k < entries.size(); ++k)
        if (entries[k].first == entries[k - 1].first) return std::nullopt;
      table.pieces.emplace(object, std::move(entries));
    }
    return table;
  }

private:
  bool Put(const std::string &object, const algorithm::UnitSite &site, std::int64_t d) {
    auto &entries = answers_[object];
    if (!entries.empty() && !piecewise_) return entries.front().second == d;
    if (entries.empty() || entries.back().first != site || entries.back().second != d) entries.emplace_back(site, d);
    return true;
  }
  bool piecewise_;
  std::map<std::string, std::vector<std::pair<algorithm::UnitSite, std::int64_t>>> answers_;
};
// The valued δ of one unit's replay, for value lifting only: every slot's plain global offsets and
// every table-atom footprint's or region's exact offset (Canon::valued) against the
// representative's, one δ per allocation. It never fails a replay: a disagreement (or a δ outside
// the signed 64-bit range) only marks the unit as not liftable. O(log allocations) per answer.
class ValuedDelta {
public:
  void Add(const std::string &object, std::uint64_t member, std::uint64_t rep) {
    const __int128 d = __int128(member) - __int128(rep);
    if (d < std::numeric_limits<std::int64_t>::min() || d > std::numeric_limits<std::int64_t>::max()) {
      liftable_ = false;
      return;
    }
    Put(object, static_cast<std::int64_t>(d));
  }
  void AddSigned(const std::string &object, std::int64_t member, std::int64_t rep) {
    std::int64_t d;
    if (__builtin_sub_overflow(member, rep, &d)) liftable_ = false;
    else Put(object, d);
  }
  void Fail() noexcept { liftable_ = false; }
  // The unit's valued δ per allocation, or nullopt when it is not one δ per allocation.
  [[nodiscard]] std::optional<std::map<std::string, std::int64_t>> Table() && {
    if (!liftable_) return std::nullopt;
    return std::move(delta_);
  }

private:
  void Put(const std::string &object, std::int64_t d) {
    if (!liftable_) return;
    const auto [found, added] = delta_.try_emplace(object, d);
    if (!added && found->second != d) liftable_ = false;
  }
  bool liftable_ = true;
  std::map<std::string, std::int64_t> delta_;
};
// The logged answers (special, uniform, memo, Partition and value reads), then
// the logged slots, of one rank's log re-asked on the member CTA's own facts; δ
// accumulates across ranks, and so does the valued δ (`valued`). False with the first failing
// question.
bool Replay(const UnitLog &log, std::uint32_t rank, const CtaFacts &facts, DeltaAnswers &delta, ValuedDelta &valued) {
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
    {
      // Value reads, one record read per (lane, pos, kind, slice) group (the log is sorted).
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
    }
    // Slot replay: every logged slot re-run on the member, then compared up to δ.
    ReplayContexts contexts(facts);
    const Json self(facts.frame().SelfCoordinates());
    // Slots follow lowering's lane order: as for value reads, a lane's value caches are
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
        Canonicalize(resources.ReplayMemory(site, copy), self, canon);
      } else {
        const TmaRequest copy(request.contract, request.store, nullptr);
        Canonicalize(resources.ReplayTma(site, copy), self, canon);
      }
      if (!canon.unliftable.empty() || canon.shape != log.strings.At(e.shape) || canon.globals.size() != e.offsets_count ||
          canon.shifts.size() != e.shifts_count)
        return false;
      const algorithm::UnitSite at{rank, static_cast<std::uint32_t>(site.instruction_id()),
                                   static_cast<std::uint32_t>(site.occurrence())};
      for (std::uint32_t k = 0; k < e.offsets_count; ++k) {
        const auto &rep = r.offsets[e.offsets_begin + k];
        const auto &object = log.strings.At(rep.object);
        if (canon.globals[k].first != object || !delta.Add(object, at, canon.globals[k].second, rep.offset))
          return false;
        valued.Add(object, canon.globals[k].second, rep.offset);
      }
      for (std::uint32_t k = 0; k < e.shifts_count; ++k) {
        const auto &rep = r.shifts[e.shifts_begin + k];
        const auto &object = log.strings.At(rep.object);
        if (canon.shifts[k].first != object || !delta.AddSigned(object, at, canon.shifts[k].second, rep.value))
          return false;
      }
      if (canon.valued.size() != e.valued_count) valued.Fail();
      else
        for (std::uint32_t k = 0; k < e.valued_count; ++k) {
          const auto &rep = r.valued[e.valued_begin + k];
          const auto &object = log.strings.At(rep.object);
          if (canon.valued[k].first != object) valued.Fail();
          else valued.AddSigned(object, canon.valued[k].second, rep.value);
        }
    }
  } catch (const std::exception &) {
    return false;
  }
  return true;
}
// Every δ of an aligned allocation (each of its pieces) a multiple of the modulus.
bool AlignOk(const UnitLog &log, const UnitDelta &delta) {
  for (const auto &r : log.ranks)
    for (const auto &a : r.aligns) {
      if (!a.modulus) continue;
      const auto &object = log.strings.At(a.object);
      const auto modulus = static_cast<std::int64_t>(a.modulus);
      if (const auto pieces = delta.pieces.find(object); pieces != delta.pieces.end()) {
        for (const auto &[site, d] : pieces->second)
          if (d % modulus != 0) return false;
        continue;
      }
      const auto found = delta.uniform.find(object);
      const std::int64_t d = found == delta.uniform.end() ? 0 : found->second;
      if (d % modulus != 0) return false;
    }
  return true;
}
// Equal structure keys on every rank (hash first, then bytes).
bool SameStructure(const std::vector<StructureKey> &a, const std::vector<StructureKey> &b) {
  if (a.size() != b.size()) return false;
  for (std::size_t k = 0; k < a.size(); ++k)
    if (a[k].hash != b[k].hash || a[k].bytes != b[k].bytes) return false;
  return true;
}
// Equal answers and slot outcomes (every rank's log replayed, δ accumulating)
// and the alignment checks (AbsoluteCheck) of a unit against one class log; `piecewise`: the class
// admits a piecewise δ (UnitClasses::AdmitPiecewise); `form_only`: it admits a unit whose exact
// offsets are not one δ per allocation from the representative's (UnitClasses::AdmitFormOnly). The
// unit's δ table, or nullopt.
std::optional<UnitDelta> Admits(const UnitLog &log, std::span<const CtaFacts *const> facts, bool piecewise,
                                bool form_only) {
  DeltaAnswers answers(piecewise);
  ValuedDelta valued;
  for (std::uint32_t rank = 0; rank < facts.size(); ++rank)
    if (!Replay(log, rank, *facts[rank], answers, valued)) return std::nullopt;
  auto delta = std::move(answers).Table();
  if (!delta || !AlignOk(log, *delta)) return std::nullopt;
  delta->valued = std::move(valued).Table();
  // A piecewise δ is never lifted (a value stage refuses a piecewise quotient), so its valued δ
  // decides nothing.
  if (!delta->valued && !form_only && delta->pieces.empty()) return std::nullopt;
  return delta;
}

// ---- containment admission --------------------------------------------------
// One rank's lane-by-lane containment of a member in its representative (unit_classes.h
// ContainmentProfile): per pair (representative path P, member path Q) that some lane follows, the
// member position of each representative record (kRemoved when the member has no such record), and
// the member's executed-record count change per instruction. Records are matched by (instruction,
// occurrence), which is unique within one path, so the matching is the only one: matched records
// keep their order and are equal (metadata; predicates of kFixed records); a representative record
// without a match, or active only in the representative, is removable (kAccess, kCompute, kBranch or
// kOpaque; a kOpaque may match a member's kBranch); a member record without a match is a kBranch.
// Lowering's register dependency state stays the representative's except for registers a removed
// record overwrote or read while pending, which no member record may write before they complete; the
// origins lowering reads are the representative's. O(records + registers) per pair, O(lanes) per
// rank.
class ContainMap {
public:
  static constexpr std::uint32_t kRemoved = UINT32_MAX;
  bool Build(const ContainmentProfile &rep, const ContainmentProfile &member) {
    if (rep.lane_path.empty() || rep.key != member.key || rep.lane_path.size() != member.lane_path.size()) return false;
    rep_ = &rep;
    member_ = &member;
    overlaps_.assign(rep.paths.size(), {});
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint64_t> lanes;
    for (std::uint32_t lane = 0; lane < rep.lane_path.size(); ++lane) {
      const auto pq = std::pair{rep.lane_path[lane], member.lane_path[lane]};
      if (pq.first >= rep.paths.size() || pq.second >= member.paths.size()) return false;
      if (lanes[pq]++ == 0) {
        overlaps_[pq.first].emplace_back(pq.second, lane);
        auto &alignment = pairs_[pq];
        if (!Align(rep.paths[pq.first], rep.origins.at(pq.first), rep.registers.at(pq.first), member.paths[pq.second],
                   member.registers.at(pq.second), alignment))
          return false;
      }
    }
    for (const auto &[pq, count] : lanes)
      for (const auto &[instruction, d] : pairs_.at(pq).occurrences)
        occurrences_[instruction] += d * static_cast<std::int64_t>(count);
    std::erase_if(occurrences_, [](const auto &entry) { return entry.second == 0; });
    return true;
  }
  // The member (lane, pos) of a question the representative asked at (lane, pos) for its lane only.
  // nullopt when the member removed that record; false when the position is not the path's.
  bool Lane(std::uint32_t lane, std::uint32_t pos, std::optional<RecordRef> &out) const {
    out.reset();
    if (lane >= rep_->lane_path.size()) return false;
    const auto &to = pairs_.at({rep_->lane_path[lane], member_->lane_path[lane]}).to_member;
    if (pos >= to.size()) return false;
    if (to[pos] != kRemoved) out = RecordRef{{lane}, to[pos]};
    return true;
  }
  // The member (lane, pos) of a question the representative asked at (lane, pos) for every lane of
  // the lane's path: one per member path those lanes follow (the lane itself on its own member path),
  // without the member paths that removed the record. False when the position is not the path's.
  bool Path(std::uint32_t lane, std::uint32_t pos, std::vector<RecordRef> &out) const {
    out.clear();
    if (lane >= rep_->lane_path.size()) return false;
    const auto p = rep_->lane_path[lane];
    for (const auto &[q, first] : overlaps_[p]) {
      const auto &to = pairs_.at({p, q}).to_member;
      if (pos >= to.size()) return false;
      if (to[pos] == kRemoved) continue;
      out.push_back({{member_->lane_path[lane] == q ? lane : first}, to[pos]});
    }
    return true;
  }
  // The member path a lane follows.
  [[nodiscard]] std::uint32_t MemberPath(std::uint32_t lane) const { return member_->lane_path.at(lane); }
  [[nodiscard]] const std::map<std::uint32_t, std::int64_t> &Occurrences() const noexcept { return occurrences_; }
  // Whether some lane of the member is marked (UnitDelta::hazard).
  [[nodiscard]] bool Hazard() const {
    return std::any_of(pairs_.begin(), pairs_.end(), [](const auto &entry) { return entry.second.hazard; });
  }
  using RemovedAccesses = std::vector<std::pair<std::uint32_t, std::uint32_t>>;
  // Per lane: the representative's active kAccess records that the lane's member path removed or has
  // predicated off (UnitDelta::removed). O(lanes).
  [[nodiscard]] std::vector<std::shared_ptr<const RemovedAccesses>> RemovedAccessesByLane() const {
    std::vector<std::shared_ptr<const RemovedAccesses>> result(rep_->lane_path.size());
    for (std::uint32_t lane = 0; lane < result.size(); ++lane)
      result[lane] = pairs_.at({rep_->lane_path[lane], member_->lane_path[lane]}).removed;
    return result;
  }

  struct Alignment {
    std::vector<std::uint32_t> to_member;
    std::map<std::uint32_t, std::int64_t> occurrences;
    std::shared_ptr<const RemovedAccesses> removed;
    bool hazard = false; // UnitDelta::hazard, for this pair of paths
  };
  static bool Active(const ContainmentRecord &r) noexcept { return r.predicate != 2; }
  // (kOpaque is removable; a member's kOpaque record must match one of the representative's.)
  static bool Align(const std::vector<ContainmentRecord> &p,
                    const std::vector<std::pair<std::uint32_t, std::uint32_t>> &p_origins,
                    const std::vector<std::uint32_t> &p_registers, const std::vector<ContainmentRecord> &q,
                    const std::vector<std::uint32_t> &q_registers, Alignment &out) {
    std::unordered_map<std::uint64_t, std::uint32_t> at;
    at.reserve(p.size() * 2);
    for (std::uint32_t i = 0; i < p.size(); ++i)
      if (!at.try_emplace((std::uint64_t(p[i].instruction) << 32) | p[i].occurrence, i).second) return false;
    out.to_member.assign(p.size(), kRemoved);
    std::int64_t last = -1;
    // Registers whose pending asynchronous write a removed record overwrote or read in the
    // representative's lowering (ContainmentRecord::touched, touched_reads): pending in the member
    // until the next tcgen05.wait::ld. The representative's record ordered that write before the
    // thread's later events (its own write replaced it, or its read waited for it); the member has no
    // such order, so a later write of the register would be unordered with the pending one only in
    // the member. While pending, a member's kAccess, kCompute or kOpaque record may read them
    // (lowering orders such a read after the write) and no kFixed record reads them; a tcgen05.ld
    // that writes one marks the member (below), any other record that writes one is refused. (A
    // record that read a pending register leaves every register it reads pending: the ones that were
    // not are a stricter condition, or an unneeded mark, only.)
    std::unordered_set<std::uint32_t> pending;
    const auto remove = [&](std::int64_t k) {
      const auto &r = p[k];
      if (r.kind == ContainmentKind::kFixed) return false;
      if (!Active(r)) return true;
      --out.occurrences[r.instruction];
      if (r.touched)
        for (auto x = r.writes_begin; x < r.writes_begin + r.writes_count; ++x) {
          if (p_registers.at(x) == kUnknownRegister) return false;
          pending.insert(p_registers[x]);
        }
      if (r.touched_reads)
        for (auto x = r.reads_begin; x < r.reads_begin + r.reads_count; ++x) {
          if (p_registers.at(x) == kUnknownRegister) return false;
          pending.insert(p_registers[x]);
        }
      return true;
    };
    // A member record lowered (active) while `pending` holds registers.
    const auto lowered = [&](const ContainmentRecord &m) {
      if (pending.empty()) return true;
      for (auto x = m.writes_begin; x < m.writes_begin + m.writes_count; ++x) {
        if (q_registers.at(x) == kUnknownRegister) return false;
        if (!pending.contains(q_registers[x])) continue;
        // A member's tcgen05.ld writes a register left pending: its write is an asynchronous effect
        // without a predecessor, as the pending one, so the two are unordered in the member only
        // (the representative's removed record ordered them). The member is marked
        // (UnitDelta::hazard) and the register stays pending, now on the new write, as in the
        // member's own lowering: later records are checked against it as before. Any other record
        // that writes a pending register is refused: its own lowering may or may not order the
        // write, which the alignment does not decide.
        if (!m.async_load) return false;
        out.hazard = true;
      }
      if (m.kind == ContainmentKind::kFixed)
        for (auto x = m.reads_begin; x < m.reads_begin + m.reads_count; ++x)
          if (q_registers.at(x) == kUnknownRegister || pending.contains(q_registers[x])) return false;
      if (m.clears) pending.clear();
      return true;
    };
    for (std::uint32_t j = 0; j < q.size(); ++j) {
      const auto &m = q[j];
      const auto found = at.find((std::uint64_t(m.instruction) << 32) | m.occurrence);
      if (found == at.end()) {
        if (m.kind != ContainmentKind::kBranch) return false;
        if (Active(m)) ++out.occurrences[m.instruction];
        continue;
      }
      const std::int64_t i = found->second;
      if (i <= last) return false;
      for (auto k = last + 1; k < i; ++k)
        if (!remove(k)) return false;
      last = i;
      const auto &r = p[i];
      if (r.kind == ContainmentKind::kOpaque && m.kind == ContainmentKind::kBranch) {
        // The representative's open region, the member's taken branch: the region removed, the
        // branch added.
        if (!remove(i)) return false;
        if (Active(m)) ++out.occurrences[m.instruction];
        continue;
      }
      if (r.kind != m.kind || r.meta != m.meta) return false;
      if (r.predicate != m.predicate) {
        // Only a record active in the representative and predicated off in the member differs.
        if (r.kind == ContainmentKind::kFixed || !Active(r) || Active(m) || !remove(i)) return false;
      } else if (Active(m) && !lowered(m)) return false;
      out.to_member[i] = j;
    }
    for (auto k = last + 1; k < static_cast<std::int64_t>(p.size()); ++k)
      if (!remove(k)) return false;
    // A register still pending at the end of the path: in the representative the removed record
    // ordered its asynchronous write before the thread's end; in the member nothing does.
    if (!pending.empty()) return false;
    // Origins: at every active record whose origins lowering reads, the member's are the
    // representative's; where lowering follows definition chains, so along the representative's
    // chains (every record of them matched, with equal origins: the member's chains are the same
    // records). O(chain records) per root, each record once.
    std::vector<bool> seen(p.size());
    std::vector<std::uint32_t> stack;
    for (std::uint32_t root = 0; root < p.size(); ++root) {
      if (!p[root].chases || !Active(p[root])) continue;
      if (p[root].chases == 1) {
        if (out.to_member[root] == kRemoved || p[root].origins != q[out.to_member[root]].origins) return false;
        continue;
      }
      if (seen[root]) continue;
      stack.assign(1, root);
      seen[root] = true;
      while (!stack.empty()) {
        const auto c = stack.back();
        stack.pop_back();
        if (out.to_member[c] == kRemoved || p[c].origins != q[out.to_member[c]].origins) return false;
        for (auto k = p[c].origin_begin; k < p[c].origin_begin + p[c].origin_count; ++k) {
          const auto found = at.find((std::uint64_t(p_origins.at(k).first) << 32) | p_origins.at(k).second);
          if (found == at.end() || seen[found->second]) continue;
          seen[found->second] = true;
          stack.push_back(found->second);
        }
      }
    }
    // The accesses the member does not perform: the representative's active kAccess records without
    // an active member record. O(records).
    RemovedAccesses removed;
    for (std::uint32_t i = 0; i < p.size(); ++i)
      if (p[i].kind == ContainmentKind::kAccess && Active(p[i]) &&
          (out.to_member[i] == kRemoved || !Active(q[out.to_member[i]])))
        removed.emplace_back(p[i].instruction, p[i].occurrence);
    std::sort(removed.begin(), removed.end());
    out.removed = std::make_shared<const RemovedAccesses>(std::move(removed));
    return true;
  }

private:
  const ContainmentProfile *rep_ = nullptr, *member_ = nullptr;
  std::map<std::pair<std::uint32_t, std::uint32_t>, Alignment> pairs_;
  std::vector<std::vector<std::pair<std::uint32_t, std::uint32_t>>> overlaps_; // per rep path: (member path, first lane)
  std::map<std::uint32_t, std::int64_t> occurrences_;
};

// Replay of one rank's log on a containment member: as Replay, with every question re-asked at the
// member's matching records (ContainMap::Lane for a lane's own read, ContainMap::Path for a
// path-uniform one, ContainMap::Members for a member list), skipped where the member removed the
// record. False with the first failing question.
bool ReplayContained(const UnitLog &log, std::uint32_t rank, const CtaFacts &facts, const ContainMap &map,
                     DeltaAnswers &delta, ValuedDelta &valued, std::set<std::string> &relaxed) {
  const auto &r = log.ranks.at(rank);
  try {
    for (const auto &e : r.specials) {
      if (!log.answers.Same(e.answer, facts.Special(log.strings.At(e.name)))) return false;
    }
    std::vector<RecordRef> targets;
    for (const auto &e : r.uniforms) {
      if (!map.Path(e.lane, e.pos, targets)) return false;
      for (const auto at : targets)
        if (facts.PathUniformBinding(at, e.reg) != e.present) return false;
    }
    // A value uniform over a cohort: uniform, with the representative's value, over the cohort's lanes
    // on each member path that keeps the record (asked there), the lanes that removed it aside.
    std::map<std::uint32_t, std::pair<std::size_t, std::vector<LaneRef>>> cohorts;
    for (const auto &e : r.legacy) {
      const auto &members = log.member_lists.at(e.members);
      cohorts.clear();
      const auto add = [&](LaneRef lane) {
        std::optional<RecordRef> at;
        if (!map.Lane(lane.index, e.pos, at)) return false;
        if (!at) return true;
        auto &cohort = cohorts[map.MemberPath(lane.index)];
        if (cohort.second.empty()) cohort.first = at->pos;
        else if (cohort.first != at->pos) return false;
        cohort.second.push_back(lane);
        return true;
      };
      if (members.empty() ? !add(LaneRef{e.lane}) : !std::all_of(members.begin(), members.end(), add)) return false;
      for (const auto &[path, cohort] : cohorts) {
        const auto lanes = members.empty() ? std::span<const LaneRef>() : std::span<const LaneRef>(cohort.second);
        if (!log.answers.Same(e.answer, facts.LegacyUniformOnlyValue({cohort.second.front(), cohort.first}, lanes,
                                                                     log.strings.At(e.name))))
          return false;
      }
    }
    std::vector<RecordRef> hits;
    for (const auto &e : r.memos) {
      if (!map.Path(e.lane, e.fill, targets) || !map.Path(e.lane, e.hit, hits)) return false;
      if (targets.size() != hits.size()) return false;
      for (std::size_t k = 0; k < targets.size(); ++k) {
        if (targets[k].lane != hits[k].lane) return false;
        const auto hit = facts.PathEvaluationKey(hits[k], e.reg);
        if (!hit || *hit > UINT32_MAX || hit != facts.PathEvaluationKey(targets[k], e.reg)) return false;
      }
    }
    // A Partition stage: each evaluated member's key at its own matching record (a stage-1 key reads
    // only its member), numbered by first occurrence in the representative's member order.
    std::map<std::string, std::uint32_t, std::less<>> numbering;
    for (const auto &e : r.partitions) {
      const auto &members = log.member_lists.at(e.members);
      if (e.evaluated > members.size()) return false;
      numbering.clear();
      for (std::uint32_t k = 0; k < e.evaluated; ++k) {
        std::optional<RecordRef> at;
        if (!map.Lane(members[k].index, e.pos, at) || !at) return false;
        const LaneRef one[1] = {members[k]};
        auto cursor = facts.BeginPartition(one, at->pos, *e.spec);
        MemberFacts derived;
        std::string key;
        try {
          key = std::string(cursor.Next(derived));
        } catch (...) {
          return false;
        }
        const auto found = numbering.try_emplace(std::move(key), static_cast<std::uint32_t>(numbering.size())).first;
        if (found->second != e.classes[k] || derived.shuffle != e.shuffles[k]) return false;
      }
    }
    {
      // Value reads, one record read per (lane, pos, kind, slice) group and target; the values of
      // the lanes a group read are dropped when the representative's lane changes.
      std::vector<std::uint32_t> read_lanes;
      const auto release = [&] {
        for (const auto lane : read_lanes) facts.ReleaseValues(LaneRef{lane});
        read_lanes.clear();
      };
      std::vector<RegId> group_regs;
      std::optional<std::uint32_t> current_lane;
      for (std::size_t k = 0; k < r.values.size();) {
        const auto &e = r.values[k];
        if (current_lane && *current_lane != e.lane) release();
        current_lane = e.lane;
        if (e.kind == ReadKind::At) {
          std::optional<RecordRef> at;
          if (!map.Lane(e.lane, e.pos, at)) return false;
          if (at) {
            read_lanes.push_back(at->lane.index);
            const auto answer = facts.ValueAt(*at, e.reg);
            if (!log.answers.Same(e.answer, answer ? &*answer : nullptr)) return false;
          }
          ++k;
          continue;
        }
        auto end = k;
        while (end < r.values.size() && r.values[end].lane == e.lane && r.values[end].pos == e.pos &&
               r.values[end].kind == e.kind && r.values[end].slice == e.slice)
          ++end;
        targets.clear();
        if (e.kind == ReadKind::Meta) {
          if (!map.Path(e.lane, e.pos, targets)) return false;
        } else {
          std::optional<RecordRef> at;
          if (!map.Lane(e.lane, e.pos, at)) return false;
          if (at) targets.push_back(*at);
        }
        for (const auto at : targets) {
          read_lanes.push_back(at.lane.index);
          RecordHolder holder;
          RecordView view = facts.Meta(at);
          if (e.kind == ReadKind::Full) {
            group_regs.clear();
            for (auto j = k; j < end; ++j) group_regs.push_back(r.values[j].reg);
            view = MemberRecord(facts, at, ReadKind::Slice, group_regs, holder);
          } else {
            const std::span<const RegId> slice =
                e.kind == ReadKind::Slice ? std::span<const RegId>(log.slices.at(e.slice)) : std::span<const RegId>();
            view = MemberRecord(facts, at, e.kind, slice, holder);
          }
          for (auto j = k; j < end; ++j)
            if (!log.answers.Same(r.values[j].answer, facts.Value(view, r.values[j].reg))) return false;
        }
        k = end;
      }
      release();
    }
    // Slot replay at every target, compared up to δ (the member's site: its matching record's
    // instruction and occurrence, which are the representative's).
    ReplayContexts contexts(facts);
    const Json self(facts.frame().SelfCoordinates());
    std::vector<RegId> site_regs;
    // The alignment moduli the representative's lowering checked, per allocation (every rank's).
    std::map<std::string, std::vector<std::uint64_t>> aligned;
    for (const auto &rank_log : log.ranks)
      for (const auto &a : rank_log.aligns)
        if (a.modulus) aligned[log.strings.At(a.object)].push_back(a.modulus);
    // As for value reads: the lanes a representative lane's sites were re-asked on are released when
    // the sites move to another representative lane.
    std::vector<std::uint32_t> site_lanes;
    std::optional<std::uint32_t> site_lane;
    for (const auto &e : r.slots) {
      if (site_lane && *site_lane != e.site_lane) {
        for (const auto lane : site_lanes) facts.ReleaseValues(LaneRef{lane});
        site_lanes.clear();
      }
      site_lane = e.site_lane;
      targets.clear();
      if (e.site_kind == ReadKind::Meta) {
        if (!map.Path(e.site_lane, e.site_pos, targets)) return false;
      } else {
        std::optional<RecordRef> at;
        if (!map.Lane(e.site_lane, e.site_pos, at)) return false;
        if (at) targets.push_back(*at);
      }
      const auto &request = log.requests.at(e.request);
      for (const auto target : targets) {
        auto &resources = contexts.At(e.consumer, e.resources);
        if (e.active_lane != kNone) {
          if (e.active_lane == e.site_lane) resources.SetLane(target.lane);
          else if (target.lane.index == e.site_lane) resources.SetLane({e.active_lane});
          else return false;
        }
        RecordHolder holder;
        std::span<const RegId> slice =
            e.site_kind == ReadKind::Slice ? std::span<const RegId>(log.slices.at(e.slice)) : std::span<const RegId>();
        auto kind = e.site_kind;
        if (e.memory && kind == ReadKind::Full) {
          site_regs = AddressRegisters(facts, request.operand);
          slice = site_regs;
          kind = ReadKind::Slice;
        }
        const auto site = MemberRecord(facts, target, kind, slice, holder);
        Canon canon;
        std::optional<SlotOutcome<ResolvedTma>> tma;
        if (e.memory) {
          const MemoryRequest copy(request.operand, request.contract, request.bytes, request.space);
          Canonicalize(resources.ReplayMemory(site, copy), self, canon);
        } else {
          const TmaRequest copy(request.contract, request.store, nullptr);
          tma.emplace(resources.ReplayTma(site, copy));
          Canonicalize(*tma, self, canon);
        }
        site_lanes.push_back(target.lane.index);
        // Exactly the representative's outcome up to δ (checked before any δ is recorded)...
        bool exact = canon.unliftable.empty() && canon.shape == log.strings.At(e.shape) &&
                     canon.globals.size() == e.offsets_count && canon.shifts.size() == e.shifts_count;
        if (exact) {
          std::map<std::string, std::int64_t> seen;
          const auto agrees = [&](const std::string &object, std::int64_t d) {
            const auto held = delta.Held(object);
            const auto [it, added] = seen.try_emplace(object, d);
            return (!held || *held == d) && (added || it->second == d);
          };
          for (std::uint32_t k = 0; exact && k < e.offsets_count; ++k) {
            const auto &rep = r.offsets[e.offsets_begin + k];
            const __int128 d = __int128(canon.globals[k].second) - __int128(rep.offset);
            exact = canon.globals[k].first == log.strings.At(rep.object) && d >= INT64_MIN && d <= INT64_MAX &&
                    agrees(canon.globals[k].first, std::int64_t(d));
          }
          for (std::uint32_t k = 0; exact && k < e.shifts_count; ++k) {
            const auto &rep = r.shifts[e.shifts_begin + k];
            std::int64_t d;
            exact = canon.shifts[k].first == log.strings.At(rep.object) &&
                    !__builtin_sub_overflow(canon.shifts[k].second, rep.value, &d) && agrees(canon.shifts[k].first, d);
          }
        }
        // ... or a load that reads elsewhere in the same allocations: a plain load with the
        // representative's outcome but other offsets, or a TMA load with the representative's outcome
        // without where it reads (CanonicalizeTmaRead). The allocations it reads must then be written
        // by no unit (UnitDelta::relaxed, checked by the memory check), its δ is not recorded, and
        // its own offsets keep every alignment the representative's lowering checked on them.
        if (!exact) {
          if (tma) {
            if (e.relaxed == kNone) return false;
            const auto read = CanonicalizeTmaRead(*tma);
            if (!read || read->shape != log.strings.At(e.relaxed)) return false;
            for (const auto &object : read->objects)
              if (aligned.contains(object)) return false;
            relaxed.insert(read->objects.begin(), read->objects.end());
          } else {
            if (facts.function().instructions.at(site.instruction_id()).opcode != "ld" || !canon.unliftable.empty() ||
                canon.shape != log.strings.At(e.shape) || canon.globals.size() != e.offsets_count ||
                canon.shifts.size() != e.shifts_count)
              return false;
            const auto keeps_alignment = [&](const std::string &object, __int128 d) {
              const auto found = aligned.find(object);
              if (found == aligned.end()) return true;
              return std::all_of(found->second.begin(), found->second.end(),
                                 [&](std::uint64_t modulus) { return d % __int128(modulus) == 0; });
            };
            for (std::uint32_t k = 0; k < e.offsets_count; ++k) {
              const auto &object = log.strings.At(r.offsets[e.offsets_begin + k].object);
              if (canon.globals[k].first != object ||
                  !keeps_alignment(object, __int128(canon.globals[k].second) - __int128(r.offsets[e.offsets_begin + k].offset)))
                return false;
              relaxed.insert(object);
            }
            for (std::uint32_t k = 0; k < e.shifts_count; ++k) {
              const auto &object = log.strings.At(r.shifts[e.shifts_begin + k].object);
              if (canon.shifts[k].first != object ||
                  !keeps_alignment(object, __int128(canon.shifts[k].second) - __int128(r.shifts[e.shifts_begin + k].value)))
                return false;
              relaxed.insert(object);
            }
          }
          valued.Fail();
          continue;
        }
        const algorithm::UnitSite at{rank, static_cast<std::uint32_t>(site.instruction_id()),
                                     static_cast<std::uint32_t>(site.occurrence())};
        for (std::uint32_t k = 0; k < e.offsets_count; ++k) {
          const auto &rep = r.offsets[e.offsets_begin + k];
          const auto &object = log.strings.At(rep.object);
          if (canon.globals[k].first != object || !delta.Add(object, at, canon.globals[k].second, rep.offset))
            return false;
          valued.Add(object, canon.globals[k].second, rep.offset);
        }
        for (std::uint32_t k = 0; k < e.shifts_count; ++k) {
          const auto &rep = r.shifts[e.shifts_begin + k];
          const auto &object = log.strings.At(rep.object);
          if (canon.shifts[k].first != object || !delta.AddSigned(object, at, canon.shifts[k].second, rep.value))
            return false;
        }
        if (canon.valued.size() != e.valued_count) valued.Fail();
        else
          for (std::uint32_t k = 0; k < e.valued_count; ++k) {
            const auto &rep = r.valued[e.valued_begin + k];
            const auto &object = log.strings.At(rep.object);
            if (canon.valued[k].first != object) valued.Fail();
            else valued.AddSigned(object, canon.valued[k].second, rep.value);
          }
      }
    }
  } catch (const std::exception &) {
    return false;
  }
  return true;
}
// Containment admission of a unit against one class log with its founder's profiles: every rank's
// ContainMap, then ReplayContained, the alignment checks, one δ per allocation (no piecewise δ) for
// every allocation the log places, and the valued δ as in Admits. The unit's δ table
// (UnitDelta::contained), or nullopt.
std::optional<UnitDelta> AdmitsContained(const UnitLog &log, const std::vector<ContainmentProfile> &rep,
                                         std::span<const CtaFacts *const> facts,
                                         std::span<const ContainmentProfile> member, bool form_only) {
  if (rep.size() != facts.size() || member.size() != facts.size()) return std::nullopt;
  for (std::size_t rank = 0; rank < facts.size(); ++rank)
    if (rep[rank].key != member[rank].key || log.ranks.at(rank).rendered) return std::nullopt;
  DeltaAnswers answers(false);
  ValuedDelta valued;
  std::set<std::string> relaxed;
  std::vector<std::map<std::uint32_t, std::int64_t>> occurrences(facts.size());
  std::vector<std::vector<std::shared_ptr<const ContainMap::RemovedAccesses>>> removed(facts.size());
  bool hazard = false;
  for (std::uint32_t rank = 0; rank < facts.size(); ++rank) {
    ContainMap map;
    if (!map.Build(rep[rank], member[rank])) return std::nullopt;
    if (!ReplayContained(log, rank, *facts[rank], map, answers, valued, relaxed)) return std::nullopt;
    occurrences[rank] = map.Occurrences();
    removed[rank] = map.RemovedAccessesByLane();
    hazard = hazard || map.Hazard();
  }
  auto delta = std::move(answers).Table();
  if (!delta || !delta->pieces.empty() || !AlignOk(log, *delta)) return std::nullopt;
  // Every allocation the log places has the member's δ, except one only read where it is relaxed.
  for (const auto &r : log.ranks) {
    for (const auto &o : r.offsets)
      if (const auto &object = log.strings.At(o.object); !delta->uniform.contains(object) && !relaxed.contains(object))
        return std::nullopt;
    for (const auto &o : r.shifts)
      if (const auto &object = log.strings.At(o.object); !delta->uniform.contains(object) && !relaxed.contains(object))
        return std::nullopt;
  }
  delta->valued = std::move(valued).Table();
  if (!delta->valued && !form_only) return std::nullopt;
  delta->contained = true;
  delta->occurrences = std::move(occurrences);
  delta->removed = std::move(removed);
  delta->hazard = hazard;
  delta->relaxed = std::move(relaxed);
  return delta;
}
} // namespace

ContainmentAlignment AlignContainmentPaths(const std::vector<ContainmentRecord> &rep,
                                           const std::vector<std::pair<std::uint32_t, std::uint32_t>> &rep_origins,
                                           const std::vector<std::uint32_t> &rep_registers,
                                           const std::vector<ContainmentRecord> &member,
                                           const std::vector<std::uint32_t> &member_registers) {
  ContainMap::Alignment alignment;
  ContainmentAlignment result;
  result.admitted = ContainMap::Align(rep, rep_origins, rep_registers, member, member_registers, alignment);
  if (!result.admitted) return result;
  result.hazard = alignment.hazard;
  if (alignment.removed) result.removed = *alignment.removed;
  return result;
}

// ---- containment founder planning -------------------------------------------
struct ContainmentPlan::Impl {
  struct Path {
    std::vector<ContainmentRecord> records;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> origins;
    std::vector<std::uint32_t> registers;
    std::uint64_t active = 0;
  };
  struct Unit {
    std::uint32_t unit = 0;
    std::vector<std::uint64_t> keys;                  // per rank
    std::vector<std::vector<std::uint32_t>> lane_path; // per rank: an interned path per lane
    std::uint64_t active = 0;
  };
  std::vector<Path> paths;
  std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> by_hash;
  std::vector<Unit> units;
  mutable std::map<std::pair<std::uint32_t, std::uint32_t>, bool> contains; // (founder path, member path)

  std::uint32_t Intern(Path path) {
    std::uint64_t h = Mix(path.records.size());
    for (const auto &r : path.records) {
      h = Mix(h ^ Mix((std::uint64_t(r.instruction) << 32) | r.occurrence));
      h = Mix(h ^ r.meta ^ Mix(r.origins) ^ (std::uint64_t(r.predicate) << 8) ^ std::uint64_t(r.kind));
    }
    auto &bucket = by_hash[h];
    for (const auto id : bucket)
      if (paths[id].records == path.records && paths[id].origins == path.origins && paths[id].registers == path.registers)
        return id;
    for (const auto &r : path.records) path.active += r.predicate != 2;
    bucket.push_back(static_cast<std::uint32_t>(paths.size()));
    paths.push_back(std::move(path));
    return bucket.back();
  }
  // Whether path `p` contains path `q` (ContainMap's record matching; no record is touched yet).
  bool Contains(std::uint32_t p, std::uint32_t q) const {
    if (p == q) return true;
    const auto [found, fresh] = contains.try_emplace({p, q}, false);
    if (fresh) {
      ContainMap::Alignment alignment;
      found->second = ContainMap::Align(paths[p].records, paths[p].origins, paths[p].registers, paths[q].records,
                                        paths[q].registers, alignment);
    }
    return found->second;
  }
  bool Contains(const Unit &f, const Unit &u) const {
    for (std::size_t rank = 0; rank < f.lane_path.size(); ++rank) {
      if (f.lane_path[rank].size() != u.lane_path[rank].size()) return false;
      for (std::size_t lane = 0; lane < f.lane_path[rank].size(); ++lane)
        if (!Contains(f.lane_path[rank][lane], u.lane_path[rank][lane])) return false;
    }
    return true;
  }
};
ContainmentPlan::ContainmentPlan() : impl_(std::make_unique<Impl>()) {}
ContainmentPlan::~ContainmentPlan() = default;
void ContainmentPlan::Add(std::uint32_t unit, std::vector<ContainmentProfile> profiles) {
  auto &s = *impl_;
  Impl::Unit entry;
  entry.unit = unit;
  for (auto &profile : profiles) {
    if (profile.lane_path.empty()) return; // no profile (unit_classes.h): the unit is not planned
    std::vector<std::uint32_t> interned(profile.paths.size());
    for (std::size_t p = 0; p < profile.paths.size(); ++p)
      interned[p] = s.Intern({std::move(profile.paths[p]), std::move(profile.origins.at(p)),
                              std::move(profile.registers.at(p)), 0});
    auto &lanes = entry.lane_path.emplace_back();
    lanes.reserve(profile.lane_path.size());
    for (const auto path : profile.lane_path) {
      lanes.push_back(interned.at(path));
      entry.active += s.paths[lanes.back()].active;
    }
    entry.keys.push_back(profile.key);
  }
  if (!entry.keys.empty()) s.units.push_back(std::move(entry));
}
ContainmentPlan::Order ContainmentPlan::Plan(std::uint32_t units) const {
  const auto &s = *impl_;
  std::map<std::vector<std::uint64_t>, std::vector<const Impl::Unit *>> groups;
  for (const auto &unit : s.units) groups[unit.keys].push_back(&unit);
  std::vector<bool> founder(units, false);
  bool contains = false;
  for (auto &[keys, group] : groups) {
    std::stable_sort(group.begin(), group.end(), [](const Impl::Unit *a, const Impl::Unit *b) {
      return a->active != b->active ? a->active > b->active : a->unit < b->unit;
    });
    std::vector<const Impl::Unit *> chosen;
    for (const auto *unit : group) {
      const auto by = std::find_if(chosen.begin(), chosen.end(), [&](const Impl::Unit *f) { return s.Contains(*f, *unit); });
      if (by != chosen.end()) {
        contains = contains || (*by)->lane_path != unit->lane_path;
        continue;
      }
      chosen.push_back(unit);
      if (unit->unit < units) founder[unit->unit] = true;
    }
  }
  Order order;
  order.contains = contains;
  order.units.reserve(units);
  for (std::uint32_t u = 0; u < units; ++u)
    if (founder[u]) order.units.push_back(u);
  order.founders = order.units.size();
  for (std::uint32_t u = 0; u < units; ++u)
    if (!founder[u]) order.units.push_back(u);
  return order;
}

// ---- the classes of one launch ----------------------------------------------
struct UnitClasses::Impl {
  const UnitPlan &plan;
  std::uint64_t budget;
  using Keys = std::shared_ptr<const std::vector<StructureKey>>;
  struct Pending {
    std::unique_ptr<UnitLog> log;
    std::unique_ptr<Recorder> recorder; // the recorded rank's, from Record to Recorded
    std::vector<StructureKey> keys;
    std::vector<ContainmentProfile> profiles; // empty: none computed
  };
  struct ClassInfo {
    std::uint32_t rep;
    Keys keys;                    // null for a class without a log
    std::shared_ptr<UnitLog> log; // null once closed (no admission needs it); never changes once set
    std::string closed;
    std::vector<std::uint32_t> members; // admitted units, in order
    std::vector<UnitDelta> deltas;
    std::vector<std::pair<std::string, std::uint64_t>> fences; // the representative's tensormap regions
    bool fence_nonconstant = false;
    bool piecewise = false; // admits a piecewise δ (AdmitPiecewise before the class was founded)
    bool form_only = false; // admits a valued-unliftable unit (AdmitFormOnly before the class was founded)
    // The founder's profiles with its register touches (AdmitContainment before the class was
    // founded, and the founder had profiles); null: no containment member.
    std::shared_ptr<const std::vector<ContainmentProfile>> profiles;
  };
  std::vector<ClassInfo> classes;
  // Per unit: its class and index in it (0 = representative), once decided.
  std::vector<std::optional<std::pair<std::uint32_t, std::uint32_t>>> membership;
  std::unordered_map<std::uint32_t, Pending> pending;
  std::uint64_t admitted = 0, class_bytes = 0;
  bool valued_unliftable = false; // some member's UnitDelta::valued is nullopt
  bool piecewise_admitted = false; // some member's UnitDelta::pieces is non-empty
  bool containment_admitted = false; // some member's UnitDelta::contained
  bool marked_admitted = false;      // some member's UnitDelta::hazard
  // PlanUseful: a containment member was admitted, or two founders have equal containment keys.
  bool plan_useful = false;
  std::set<std::vector<std::uint64_t>> founder_keys; // per founder with profiles: its ranks' keys
  bool futility = false, piecewise = false, form_only = false;
  std::atomic<bool> containment{false}; // AdmitContainment was called (read by the driver's workers)
  std::size_t expected_founders = 0;     // ExpectFounders
  // The classes with logs as of the last Found, for PreAdmit off the main thread.
  struct SnapshotClass {
    std::uint32_t klass;
    Keys keys;
    std::shared_ptr<const UnitLog> log;
    bool piecewise, form_only;
    std::shared_ptr<const std::vector<ContainmentProfile>> profiles;
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
    info.piecewise = piecewise;
    info.form_only = form_only;
    std::uint64_t log_bytes = 0;
    if (p.log) {
      for (const auto &r : p.log->ranks) {
        for (const auto &f : r.fences) info.fences.emplace_back(p.log->strings.At(f.object), f.offset);
        info.fence_nonconstant |= r.fence_nonconstant;
      }
      log_bytes = p.log->Bytes();
    }
    // Two founders with one containment key (the same lanes and fixed records) differ only in records
    // a containment member may lack: one may contain the other, or a third unit both.
    if (containment && !p.profiles.empty() &&
        std::none_of(p.profiles.begin(), p.profiles.end(), [](const auto &profile) { return profile.lane_path.empty(); })) {
      std::vector<std::uint64_t> keys;
      for (const auto &profile : p.profiles) keys.push_back(profile.key);
      if (!founder_keys.insert(std::move(keys)).second) plan_useful = true;
    }
    if (futility) info.closed = "futility";
    else if (!p.log) info.closed = "not_recorded";
    else if (class_bytes + log_bytes > budget) info.closed = "log_budget";
    else if (!p.log->closed.empty()) info.closed = p.log->closed;
    else {
      // Only a class with a log can admit units, so only it keeps the structure keys (and, for
      // containment members, its profiles with the founder's register touches).
      if (containment && p.profiles.size() == p.log->ranks.size()) {
        bool rendered = false;
        for (std::size_t rank = 0; rank < p.profiles.size(); ++rank) {
          auto &profile = p.profiles[rank];
          const auto &r = p.log->ranks[rank];
          rendered |= r.rendered;
          for (const auto &[lane, pos] : r.touches)
            if (lane < profile.lane_path.size() && pos < profile.paths[profile.lane_path[lane]].size())
              profile.paths[profile.lane_path[lane]][pos].touched = true;
          for (const auto &[lane, pos] : r.read_touches)
            if (lane < profile.lane_path.size() && pos < profile.paths[profile.lane_path[lane]].size())
              profile.paths[profile.lane_path[lane]][pos].touched_reads = true;
        }
        if (!rendered) info.profiles = std::make_shared<const std::vector<ContainmentProfile>>(std::move(p.profiles));
      }
      info.log = std::move(p.log);
      info.keys = std::make_shared<const std::vector<StructureKey>>(std::move(p.keys));
      class_bytes += log_bytes;
    }
    membership.at(unit) = {static_cast<std::uint32_t>(classes.size()), 0};
    classes.push_back(std::move(info));
    if (admitted == 0 && classes.size() >= 16 + expected_founders) futility = true;
    // Publish the class (the snapshot shares its keys and log).
    std::shared_ptr<const Snapshot> previous;
    {
      std::lock_guard lock(snapshot_mutex);
      previous = snapshot;
    }
    auto next = std::make_shared<Snapshot>(*previous);
    next->classes = static_cast<std::uint32_t>(classes.size());
    if (classes.back().log)
      next->logged.push_back({next->classes - 1, classes.back().keys, classes.back().log, classes.back().piecewise,
                              classes.back().form_only, classes.back().profiles});
    std::lock_guard lock(snapshot_mutex);
    snapshot = std::move(next);
  }
  void Join(std::uint32_t unit, std::uint32_t klass, UnitDelta delta) {
    ++admitted;
    if (!delta.valued && delta.pieces.empty()) valued_unliftable = true;
    if (!delta.pieces.empty()) piecewise_admitted = true;
    if (delta.contained) containment_admitted = plan_useful = true;
    if (delta.hazard) marked_admitted = true;
    classes[klass].members.push_back(unit);
    classes[klass].deltas.push_back(std::move(delta));
    membership.at(unit) = {klass, static_cast<std::uint32_t>(classes[klass].members.size())};
  }
  // The class's allocations with a piecewise δ in some member: their sites (every member replayed the
  // same log, so each such member lists the same sites; a member with one δ has it at every site),
  // grouped into pieces by their δ over the class's units. O(members x sites log) per allocation.
  static std::map<std::string, algorithm::PiecewiseDelta> Pieces(const ClassInfo &c) {
    std::map<std::string, algorithm::PiecewiseDelta> result;
    for (const auto &member : c.deltas)
      for (const auto &[object, entries] : member.pieces) {
        auto &table = result[object];
        if (!table.sites.empty()) continue;
        for (const auto &[site, d] : entries) table.sites.push_back(site);
      }
    for (auto &[object, table] : result) {
      std::vector<std::vector<std::int64_t>> rows(table.sites.size(), std::vector<std::int64_t>(c.deltas.size() + 1, 0));
      for (std::size_t m = 0; m < c.deltas.size(); ++m) {
        const auto &member = c.deltas[m];
        if (const auto pieced = member.pieces.find(object); pieced != member.pieces.end()) {
          if (pieced->second.size() != table.sites.size())
            throw std::logic_error("internal error: CTA class bookkeeping (members with different slot sites)");
          for (std::size_t k = 0; k < table.sites.size(); ++k) {
            if (pieced->second[k].first != table.sites[k])
              throw std::logic_error("internal error: CTA class bookkeeping (members with different slot sites)");
            rows[k][m + 1] = pieced->second[k].second;
          }
          continue;
        }
        const auto uniform = member.uniform.find(object);
        if (uniform == member.uniform.end()) {
          // A containment member that only reads the allocation through relaxed TMA loads: no unit
          // writes it, so its δ places nothing.
          if (member.relaxed.contains(object)) continue;
          throw std::logic_error("internal error: CTA class bookkeeping (a member without the allocation's δ)");
        }
        for (auto &row : rows) row[m + 1] = uniform->second;
      }
      std::map<std::vector<std::int64_t>, std::uint32_t> ids;
      table.piece.reserve(rows.size());
      for (auto &row : rows) {
        const auto [it, added] = ids.try_emplace(row, static_cast<std::uint32_t>(table.delta.size()));
        if (added) table.delta.push_back(std::move(row));
        table.piece.push_back(it->second);
      }
    }
    return result;
  }
  // The tensormap locality rule over every unit's cp_fenceproxy regions (representative + δ for members).
  struct Region {
    std::string object;
    std::uint64_t begin;
    std::uint32_t unit;
  };
  bool Locality() const {
    if (!plan.precondition.tensormap_copies) return true;
    std::vector<Region> regions;
    for (const auto &c : classes) {
      if (c.fence_nonconstant) return false; // a region that is not constant
      for (const auto &[object, offset] : c.fences) {
        regions.push_back({object, offset, c.rep});
        for (std::size_t m = 0; m < c.members.size(); ++m) {
          // A piecewise δ is not placed here: unknown.
          const auto found = c.deltas[m].uniform.find(object);
          if (found == c.deltas[m].uniform.end()) return false;
          std::int64_t moved; // the member's region; one outside the int64 offset range is unknown
          if (offset > std::uint64_t(INT64_MAX) || __builtin_add_overflow(std::int64_t(offset), found->second, &moved) ||
              moved < 0)
            return false;
          regions.push_back({object, std::uint64_t(moved), c.members[m]});
        }
      }
      if (c.closed == "futility" || c.closed == "log_budget" || c.closed == "not_recorded") return false; // unknown
    }
    // No two units' 128-byte tensormap regions overlap, and none overlaps an
    // input map's location (an absolute address cannot be placed).
    std::sort(regions.begin(), regions.end(), [](const Region &a, const Region &b) {
      return std::tie(a.object, a.begin, a.unit) < std::tie(b.object, b.begin, b.unit);
    });
    for (std::size_t i = 0; i + 1 < regions.size(); ++i)
      for (std::size_t j = i + 1; j < regions.size() && regions[j].object == regions[i].object &&
                                  regions[j].begin - regions[i].begin < 128; // sorted: no wrap
           ++j)
        if (regions[j].unit != regions[i].unit) return false;
    if (regions.empty() || plan.input_maps.empty()) return true;
    std::vector<TensormapRegion> placed;
    placed.reserve(regions.size());
    for (const auto &region : regions) placed.push_back({region.object, region.begin});
    return !TensormapInputOverlap(placed, plan.input_maps);
  }
};

UnitClasses::UnitClasses(const UnitPlan &plan, std::uint64_t log_budget_bytes)
    : impl_(std::make_unique<Impl>(plan, log_budget_bytes)) {}
UnitClasses::~UnitClasses() = default;

// ---- class mode -------------------------------------------------------------
UnitClasses::PreAdmission UnitClasses::PreAdmit(std::span<const CtaFacts *const> facts,
                                                const std::vector<StructureKey> &keys, PreAdmission previous,
                                                std::span<const ContainmentProfile> profiles) const {
  if (previous.joined) return previous;
  const auto &s = *impl_;
  std::shared_ptr<const Impl::Snapshot> snapshot;
  {
    std::lock_guard lock(s.snapshot_mutex);
    snapshot = s.snapshot;
  }
  if (snapshot->classes < previous.classes)
    throw std::logic_error("internal error: CTA class bookkeeping (look-ahead admission saw fewer classes)");
  PreAdmission result;
  result.classes = snapshot->classes;
  const auto size = s.plan.UnitSize();
  if (facts.size() != size || keys.size() != size)
    throw std::logic_error("internal error: CTA class bookkeeping (look-ahead admission without every CTA of the unit)");
  for (const auto &candidate : snapshot->logged) {
    if (candidate.klass < previous.classes) continue;
    std::optional<UnitDelta> delta;
    if (SameStructure(*candidate.keys, keys)) delta = Admits(*candidate.log, facts, candidate.piecewise, candidate.form_only);
    else if (candidate.profiles && !profiles.empty())
      delta = AdmitsContained(*candidate.log, *candidate.profiles, facts, profiles, candidate.form_only);
    if (delta) {
      result.joined = candidate.klass;
      result.delta = std::move(*delta);
      return result;
    }
  }
  return result;
}

std::optional<std::uint32_t> UnitClasses::Admit(std::uint32_t unit, std::span<const CtaFacts *const> facts,
                                                std::vector<StructureKey> keys, PreAdmission pre,
                                                std::vector<ContainmentProfile> profiles) {
  auto &s = *impl_;
  const auto size = s.plan.UnitSize();
  if (facts.size() != size)
    throw std::logic_error("internal error: CTA class bookkeeping (admission without every CTA of the unit)");
  if (keys.size() != size)
    throw std::logic_error("internal error: CTA class bookkeeping (admission without every CTA's structure key)");
  if (pre.joined) {
    const auto joined = *pre.joined;
    Join(unit, std::move(pre));
    return joined;
  }
  // Classes [0, pre.classes) are decided by the look-ahead: the first of them that admits
  // the unit is the sequential first (earlier ones fail on these facts either way).
  if (pre.classes > s.classes.size())
    throw std::logic_error("internal error: CTA class bookkeeping (look-ahead admission saw a later class)");
  // Structure keys, then answers, slot outcomes and alignment checks: the later
  // candidate classes in creation order.
  // Class mode with values: value lifting adds no admission condition.
  // The only modulus a value stage applies to an absolute global offset
  // is an audited row width R (a row map's row width), known only after
  // value analysis; run.cpp then
  // re-checks every member's δ against it and restarts per unit on a
  // violation. The offset bounds are checked where δ is applied.
  // A class whose structure keys differ may admit the unit as a containment member.
  for (std::uint32_t c = pre.classes; c < s.classes.size(); ++c) {
    const auto &candidate = s.classes[c];
    if (!candidate.log) continue;
    std::optional<UnitDelta> delta;
    if (SameStructure(*candidate.keys, keys)) delta = Admits(*candidate.log, facts, candidate.piecewise, candidate.form_only);
    else if (candidate.profiles && !profiles.empty())
      delta = AdmitsContained(*candidate.log, *candidate.profiles, facts, profiles, candidate.form_only);
    if (delta) {
      s.Join(unit, c, std::move(*delta));
      return c;
    }
  }
  auto &pending = s.pending[unit];
  pending.keys = std::move(keys);
  pending.profiles = std::move(profiles);
  return std::nullopt;
}

void UnitClasses::Join(std::uint32_t unit, PreAdmission pre) {
  auto &s = *impl_;
  if (!pre.joined)
    throw std::logic_error("internal error: CTA class bookkeeping (a join without a look-ahead admission)");
  if (pre.classes > s.classes.size())
    throw std::logic_error("internal error: CTA class bookkeeping (look-ahead admission saw a later class)");
  s.Join(unit, *pre.joined, std::move(pre.delta));
}

void UnitClasses::AdmitPiecewise() noexcept { impl_->piecewise = true; }
void UnitClasses::AdmitFormOnly() noexcept { impl_->form_only = true; }
bool UnitClasses::ValuedUnliftable() const noexcept { return impl_->valued_unliftable; }
bool UnitClasses::PiecewiseAdmitted() const noexcept { return impl_->piecewise_admitted; }
void UnitClasses::AdmitContainment() noexcept { impl_->containment = true; }
bool UnitClasses::ContainmentOn() const noexcept { return impl_->containment; }
void UnitClasses::ExpectFounders(std::size_t founders) noexcept { impl_->expected_founders = founders; }
bool UnitClasses::ContainmentAdmitted() const noexcept { return impl_->containment_admitted; }
bool UnitClasses::PlanUseful() const noexcept { return impl_->plan_useful; }
bool UnitClasses::MarkedAdmitted() const noexcept { return impl_->marked_admitted; }

QuerySink *UnitClasses::Record(std::uint32_t unit, std::uint32_t rank, const CtaFacts &facts) {
  auto &s = *impl_;
  auto &p = s.pending.at(unit);
  if (s.futility) return nullptr; // futile: no new log
  if (!p.log) p.log = std::make_unique<UnitLog>(s.plan.UnitSize());
  p.recorder = std::make_unique<Recorder>(*p.log, rank, facts);
  return p.recorder.get();
}

void UnitClasses::Recorded(std::uint32_t unit, std::uint32_t rank) {
  auto &s = *impl_;
  auto &p = s.pending.at(unit);
  if (p.recorder) {
    p.recorder->Seal();
    p.recorder.reset();
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
    if (!membership) throw std::logic_error("internal error: CTA class bookkeeping (a launch unit has no class)");
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
    for (const auto &d : c.deltas) entry.delta.push_back(d.uniform);
    entry.valued_delta.emplace_back(std::map<std::string, std::int64_t>{});
    for (const auto &d : c.deltas) entry.valued_delta.push_back(d.valued);
    // (Members are in admission order: unit order, unless the loop took planned founders first.)
    entry.contained.push_back(false);
    entry.removed.emplace_back();
    for (const auto &d : c.deltas) {
      entry.contained.push_back(d.contained);
      entry.removed.push_back(d.removed);
      entry.hazard = entry.hazard || d.hazard;
      if (!d.contained) continue;
      entry.relaxed_reads.insert(d.relaxed.begin(), d.relaxed.end());
      entry.occurrence_adjust.resize(std::max(entry.occurrence_adjust.size(), d.occurrences.size()));
      for (std::size_t rank = 0; rank < d.occurrences.size(); ++rank)
        for (const auto &[instruction, change] : d.occurrences[rank]) entry.occurrence_adjust[rank][instruction] += change;
    }
    entry.pieces = Impl::Pieces(c);
    for (const auto &[object, pieces] : entry.pieces)
      for (auto &d : entry.delta) d.erase(object);
    q.classes.push_back(std::move(entry));
  }
  return q;
}

bool UnitClasses::TensormapLocality() const { return impl_->Locality(); }
} // namespace sentryward::frontend::ptx
