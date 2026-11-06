#pragma once
#include "sentryward/algorithm/io/source_json.h"
#include "sentryward/frontend/ptx/execution.h"
#include "sentryward/frontend/ptx/unit_facts.h"
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <variant>
#include <vector>
namespace sentryward::frontend::ptx {
class RegisterTable;
class SlotResolver;
using namespace sentryward::algorithm;
class SemanticError : public std::runtime_error {
public:
  std::string code;
  nlohmann::json details = nlohmann::json::object();
  SemanticError(std::string code, std::string message, nlohmann::json evidence = nlohmann::json::object())
      : std::runtime_error(std::move(message)), code(std::move(code)), details(std::move(evidence)) {}
};
using Fragment = std::pair<AddressId, std::optional<std::uint64_t>>;
using SharedAddress = std::pair<CtaRank, std::uint64_t>;
std::string CoordinateText(Coordinates);
std::string Opcode(const Instruction &);
std::vector<std::string> OperandTexts(const Module &, const Instruction &);
std::string CanonicalSpace(std::string);
std::uint64_t Unsigned(const Integer &, std::string_view,
                       std::uint64_t maximum = UINT64_MAX);
std::int64_t Signed(const Integer &, std::string_view);
// A bit-pattern operand (a matrix or copy descriptor): the value modulo 2^width.
// Registers hold integers unreduced, so a .b64 `or` with a negative immediate
// is negative; integer_range outside both the signed and unsigned readings.
std::uint64_t Bits(const Integer &, std::string_view, unsigned width = 64);
class UnitInterner;
// Written only through UnitInterner; consumers read a
// binding through UnitInterner::Binding / Resources::Binding.
class AddressTable {
public:
  // A worker's table (ModelLowerer's lanes on worker threads, which intern into tables of their
  // own): it also keeps each binding's intern key and which fields later steps overwrote.
  void KeepWrites() noexcept { keep_writes_ = true; }
  // The effect on this table of the steps that built `local` (a KeepWrites table), applied in
  // their order: local bindings are interned in local id order (the order their keys first
  // occurred, so new ids come out as one in-order application would assign them), then each
  // field a step overwrote gets the local value (its last write). Returns the local -> own id
  // map, or nullopt, writing nothing, when one of those interns would have thrown
  // alias_scope_conflict here. O(local bindings x lookup).
  [[nodiscard]] std::optional<std::vector<AddressId>> Absorb(const AddressTable &local);

private:
  friend class UnitInterner;
  std::vector<ConstantAddressBinding> bindings;
  AddressId Intern(std::string space, std::string allocation,
                   std::uint64_t offset, std::string alias_scope = {});
  struct Key {
    std::uint32_t name;
    std::uint64_t offset;
    bool operator==(const Key &) const = default;
  };
  struct Hash {
    std::size_t operator()(const Key &k) const {
      return k.offset ^ (std::size_t(k.name) * 0x9e3779b97f4a7c15ULL);
    }
  };
  std::map<std::pair<std::string, std::string>, std::uint32_t> names_;
  std::unordered_map<Key, AddressId, Hash> ids_;
  // KeepWrites only: per binding its intern key and overwritten fields (kWrote*); per name its key.
  enum : std::uint8_t { kWroteAllocation = 1, kWroteOffset = 2, kWroteExpression = 4 };
  void Wrote(AddressId id, std::uint8_t fields) {
    if (keep_writes_) written_.at(id.value()) |= fields;
  }
  bool keep_writes_ = false;
  std::vector<Key> keys_;
  std::vector<std::uint8_t> written_;
  std::vector<const std::pair<std::string, std::string> *> name_keys_;
};
class BarrierTable {
public:
  std::tuple<std::string, Coordinates, std::string> Location(BarrierLifetimeId) const;
  // Every lifetime's owner CTA (class mode's unit-closure check); O(lifetimes).
  [[nodiscard]] std::vector<Coordinates> Owners() const;
  // The effect on this table of the interns that built `local`, in their order (local id order):
  // the local -> own id map. O(local lifetimes x lookup).
  [[nodiscard]] std::vector<BarrierLifetimeId> Absorb(const BarrierTable &local);

private:
  friend class UnitInterner;
  BarrierLifetimeId Intern(std::string kind, Coordinates owner,
                           std::string identity);
  std::map<std::tuple<std::string, Coordinates, std::string>, BarrierLifetimeId>
      ids_;
};
// Intern steps: one alternative per kind of table write.
// ConstantStep: AddressTable::Intern(fmt.space, fmt.allocation, offset, fmt.alias)
// with fmt = frame.Format(key).
struct ConstantStep {
  AllocationKey key;
  std::uint64_t offset = 0;
};
// A binding's stored allocation: a key formatted by the frame, or a literal
// ("unresolved", or a copy of a binding's allocation).
using StoredAllocation = std::variant<AllocationKey, std::string>;
// Intern(space, "expression:" + serialized, 0, alias), then the unconditional
// .allocation = allocation, and .expression = *expression if present.
struct ExpressionStep {
  std::string space, serialized, alias;
  StoredAllocation allocation;
  std::optional<std::string> expression;
};
// The ExpressionStep whose .expression is its `serialized` text.
inline ExpressionStep SerializedExpressionStep(std::string space, std::string serialized, std::string alias,
                                               StoredAllocation allocation) {
  // Braced: the name copy is taken before the expression is moved.
  return ExpressionStep{std::move(space), serialized, std::move(alias), std::move(allocation),
                        std::move(serialized)};
}
// The TMA region patch (resources.cpp Tma): .allocation/.offset if present,
// then .expression. In a slot outcome `target` is the patched ExpressionStep's
// index in the same ordered step list; UnitInterner::Apply(PatchStep, id)
// applies it through that step's id.
struct PatchStep {
  std::size_t target = 0;
  std::optional<StoredAllocation> allocation;
  std::optional<std::uint64_t> offset;
  std::string expression;
};
// BarrierTable::Intern(KindText(kind), frame.Cta(owner), identity).
struct BarrierStep {
  BarrierKey key;
};
using InternStep = std::variant<ConstantStep, ExpressionStep, PatchStep, BarrierStep>;
// A constant key formatted once, for a run of ConstantSteps {key, offset}
// that share it (matrix fragments).
struct PreparedKey {
  AllocationKey key;
  FormattedAllocation formatted;
};
// An abstract global endpoint's place in its tile region (TmaBody): the region (a
// tile::gather4 row, else 0), the endpoint's first byte from the region's first byte when its
// bytes are contiguous in the tensor, and whether every symbolic coordinate of the region is a
// bounded integer whose range keeps the region's box inside the tensor.
struct TmaRegionPlace {
  std::uint32_t region = 0;
  std::optional<std::uint64_t> offset;
  bool in_bounds = false;
};
struct TmaPair {
  std::optional<Fragment> source;
  std::optional<Fragment> destination;
  std::uint64_t logical_offset;
  std::uint32_t target_rank;
  std::optional<TmaRegionPlace> place = std::nullopt;  // an abstract global endpoint
};
struct TmaAccesses {
  std::vector<TmaPair> pairs;
  std::vector<std::pair<BarrierLifetimeId, std::uint64_t>> completions;
  // A read of a runtime-written tensor map (RecoverTensorMaps): its 128 bytes, keyed as the
  // tensormap.cp_fenceproxy lane writes are.
  std::optional<Fragment> descriptor;
};
// ---- the slot interface -----------------------------------------------------
// The address bridge: (active lane, record, operand text) -> address JSON.
using AddressBridge = std::function<std::optional<nlohmann::json>(LaneRef, const RecordView &, std::string_view)>;
// The registers a memory slot's resolution reads from its site record, for its address operand
// (`[reg]`, `[reg+imm]` or a plain register): that operand's own register (MemoryBody, Shared and
// GenericOwner read only it there; definitions are other records). Sorted.
std::vector<RegId> AddressRegisters(const FactSource &facts, std::string_view operand);
// The consumer a Resources serves, and its bridge: an empty bridge means the
// slot resolver consults its own address-value extractor (`if (bridge)`).
struct SlotConfig {
  Consumer consumer = Consumer::Driver;
  AddressBridge bridge;
};
// Resources::Memory's arguments. `operand` and `contract` are the caller's and
// must outlive the call.
struct MemoryRequest {
  MemoryRequest(std::string_view operand, const nlohmann::json &contract, std::uint64_t bytes,
                std::string space) noexcept
      : operand(operand), contract(contract), bytes(bytes), space(std::move(space)) {}
  std::string_view operand;
  const nlohmann::json &contract;
  std::uint64_t bytes;
  std::string space;
};
// Resources::Tma's arguments. The executor's runtime tensor maps are searched
// after the program's own (null: none).
struct TmaRequest {
  TmaRequest(const nlohmann::json &contract, bool store,
             const nlohmann::json *dynamic_tensor_maps = nullptr) noexcept
      : contract(contract), store(store), dynamic_tensor_maps(dynamic_tensor_maps) {}
  const nlohmann::json &contract;
  bool store;
  const nlohmann::json *dynamic_tensor_maps;
};
// The original exception object of a failed resolution: Commit rethrows it
// (same dynamic type, code, what() and details). `code`: a SemanticError's code.
struct SlotError {
  std::string code;
  std::exception_ptr exception;
};
enum class FallbackUse : std::uint8_t { None, Bridge, Extractor };
// One intern step of a slot outcome.
struct SlotStep {
  InternStep step;
};
// A slot's terminal. The resolver returns it as a prvalue built in place
// (std::in_place_index<0>), so the outcome holds it without a move.
template <class T> using SlotTerminal = std::variant<T, SlotError>;
template <class T> struct SlotOutcome {
  std::vector<SlotStep> prefix;  // steps applied before the terminal point; empty on the hot path
  SlotTerminal<T> terminal;
  FallbackUse fallback = FallbackUse::None;  // the bridge or extractor was consulted, whatever it returned
};
enum class FootprintKind : std::uint8_t { Constant, Abstract, Unresolved };
// The offset of a bounded-pointer footprint (tensor_properties) in its allocation:
// constant + sum(coefficient x element of `object` at byte `offset`, read as `element_type`,
// or an element function of it: a nonempty `function`, within [lower, upper]).
struct AbstractAffine {
  struct Term {
    std::string object, element_type;
    std::int64_t offset = 0, coefficient = 0;
    std::string function = {};
    std::int64_t lower = 0, upper = 0;
  };
  std::int64_t constant = 0;
  std::vector<Term> terms;
};
// Memory's terminal: `base` (Constant, Abstract) then `expression` (Abstract,
// Unresolved) are applied; the fragment is the last one's id with `bytes`.
struct ResolvedFootprint {
  FootprintKind kind;
  std::optional<ConstantStep> base;
  std::optional<ExpressionStep> expression;
  std::uint64_t bytes;
  // A bounded-pointer footprint: `expression` serializes this offset (class mode compares it).
  std::optional<AbstractAffine> affine = std::nullopt;
};
// Tma's terminal: per segment [global steps][shared step], then the
// completions' BarrierSteps; pairs and completions name steps by index. An
// abstract region's global steps (expression, placement, patch) come once, at
// the first segment of the region; its other segments' pairs name the same step.
struct ResolvedTma {
  std::vector<SlotStep> steps;
  struct Pair {
    std::optional<std::size_t> global;
    std::size_t shared;
    std::uint64_t bytes, logical_offset;
    std::uint32_t target_rank;
    std::optional<TmaRegionPlace> place = std::nullopt;  // an abstract global endpoint
  };
  std::vector<Pair> pairs;
  std::vector<std::pair<std::size_t, std::uint64_t>> completions;
  bool store = false;
  std::string descriptor;  // the matched map key, or "explicit:" + the contract's descriptor
  // Symbolic coordinates that are all affine in tensor-property elements (and element
  // functions): the region's first byte in its global allocation, base + sum of coordinate x
  // stride (bytes); one form per gather row (a tile::gather4), else one. Empty otherwise.
  std::vector<AbstractAffine> region_affine;
};
// The recorder's query of one slot (a SlotMemory or SlotTma event); built only while recording.
// `active_lane`: the Resources' lane at the call (Definition and local owners read it).
struct SlotQuery {
  ConsumerTag consumer;
  RecordView site;
  const MemoryRequest *memory;
  const TmaRequest *tma;
  std::optional<LaneRef> active_lane;
};
// The only writer of AddressTable and BarrierTable. It
// owns no state: each Apply performs its table write at the caller's program
// point, immediately, and names owners through the frame (table keys stay
// absolute).
class UnitInterner {
public:
  UnitInterner(const FactSource &facts, const UnitFrame &frame, AddressTable &addresses,
               BarrierTable &barriers) noexcept
      : facts_(facts), frame_(frame), addresses_(addresses), barriers_(barriers) {}
  // Steps are taken by rvalue reference: their strings move straight into the
  // table, with no by-value parameter in between (hot on Memory's path).
  AddressId Apply(ConstantStep &&step);
  AddressId Apply(const PreparedKey &key, std::uint64_t offset);
  AddressId Apply(ExpressionStep &&step);
  void Apply(PatchStep &&step, AddressId target);
  BarrierLifetimeId Apply(BarrierStep &&step);
  [[nodiscard]] PreparedKey Prepare(AllocationKey key) const;
  // A consumer read of a binding: .at() semantics; a BindingRead event.
  const ConstantAddressBinding &Binding(AddressId id) const {
    if (facts_.recording()) [[unlikely]]
      facts_.Note(query::BindingRead{id});
    return addresses_.bindings.at(id.value());
  }
  // bulk_reduce's base alignment (its only caller): an AbsoluteCheck event, then the
  // bulk_reduction_alignment throw for a shared offset that some legal placement leaves
  // misaligned (FrontendConfig::SharedMisalignment), or bulk_reduction_base_alignment_unknown
  // for another address space's offset with offset % modulus != 0.
  void RequireAligned(const ConstantAddressBinding &binding, std::uint64_t modulus, AddressId id) const;
  // BarrierTable::Location (executor only).
  std::tuple<std::string, Coordinates, std::string> Location(BarrierLifetimeId id) const {
    return barriers_.Location(id);
  }
  // Apply a slot outcome: the prefix steps in order, then rethrow
  // the stored exception, or apply the terminal's steps and return its result.
  Fragment Commit(SlotOutcome<ResolvedFootprint> &&outcome);
  TmaAccesses Commit(SlotOutcome<ResolvedTma> &&outcome);
  // The launch bindings, moved out at the end of lowering (ModelLowerer Finish).
  static std::vector<ConstantAddressBinding> Release(AddressTable &&addresses) {
    return std::move(addresses.bindings);
  }

private:
  friend class Resources;
  // The read-back of a binding interned earlier in the same call (Abstract's
  // base): not a consumer read, so no event. The slot resolver reads such a
  // binding from its step instead (a stored binding equals its step).
  const ConstantAddressBinding &Stored(AddressId id) const { return addresses_.bindings.at(id.value()); }
  AddressId Constant(FormattedAllocation &&formatted, std::uint64_t offset);
  // Applies `steps` in order; the result holds each step's AddressId or
  // BarrierLifetimeId value (0 for a PatchStep), by step index.
  std::vector<std::uint32_t> Apply(std::vector<SlotStep> &&steps);
  // Applies `steps` in order without collecting ids: a footprint prefix, which
  // holds no PatchStep (logic_error otherwise).
  void ApplyPrefix(std::vector<SlotStep> &&steps);
  // One step; `ids` holds the earlier steps' values, null when no PatchStep may occur.
  std::uint32_t ApplyStep(SlotStep &&step, const std::vector<std::uint32_t> *ids);
  std::string Allocation(StoredAllocation &&allocation) const;
  const FactSource &facts_;
  const UnitFrame &frame_;
  AddressTable &addresses_;
  BarrierTable &barriers_;
};
class Resources {
public:
  using KnownBits = std::pair<std::uint64_t, std::uint64_t>;
  using BitsResolver = std::function<std::optional<KnownBits>(
      const RecordView &, std::string_view, unsigned)>;
  // Values are read through `facts`; names and owners through `frame`; the
  // tables are written and read only through interner(). `slots`: the
  // consumer and its address bridge.
  Resources(const FactSource &, const UnitFrame &, AddressTable &,
            BarrierTable &, SlotConfig slots);
  // A worker's Resources for `like`'s facts, frame and resource records over its own tables
  // (built on the consuming thread): its events carry `like`'s id, so a recorder logs the
  // worker's slots as `like`'s, as if `like` had resolved them.
  Resources(const Resources &like, AddressTable &, BarrierTable &, SlotConfig slots);
  ~Resources();
  const FactSource &facts;
  const UnitFrame &frame;
  UnitInterner &interner() noexcept { return interner_; }
  // A consumer read of a binding (a BindingRead event).
  const ConstantAddressBinding &Binding(AddressId id) const { return interner_.Binding(id); }
  void SetLane(LaneRef);
  // SetLane at the start of a lane's lowering. It first drops every lane's cached answers and
  // copies (lane_indices_): another lane's shfl or matrix rows fill a peer's entries, and a hit
  // skips the reads the lane would note. So a lane notes the same reads whichever lanes this
  // Resources served before. It clears also when nothing records: that keeps one code path and
  // bounds the caches to about one lane's entries, at the cost of rebuilding a peer's entries
  // that a later lane reads again. O(the dropped entries).
  void BeginLane(LaneRef);
  std::optional<std::uint32_t> SharedOwner(const RecordView &e, std::string_view token) {
    return Owner(e, token);
  }
  Integer ScalarValue(const RecordView &, std::string_view,
                      std::string_view field) const;
  KnownBits FixedBits(const RecordView &, std::string_view, unsigned = 32,
                      const BitsResolver & = {}) const;
  std::optional<SharedAddress> Shared(const RecordView &,
                                      std::string_view);
  BarrierLifetimeId Barrier(const RecordView &, std::string_view,
                            const nlohmann::json &contract);
  Fragment Bind(std::string space, std::string resource, std::uint64_t offset,
                std::uint64_t bytes, std::optional<CtaRank> owner = {});
  Fragment Abstract(std::string space, std::string resource,
                    const nlohmann::json &offset, std::optional<std::uint64_t> bytes,
                    std::optional<CtaRank> owner = {});
  // The slots: SlotResolver resolves without writing either
  // table; the interner then commits the outcome here, at the call site.
  Fragment Memory(const RecordView &, const MemoryRequest &);
  TmaAccesses Tma(const RecordView &, const TmaRequest &);
  // Class-mode slot replay only (query_log.cpp, which compares a member's slot
  // outcomes with the logged ones): the slot outcome without its commit, on a
  // Resources over a member's facts and scratch tables. Nothing is interned.
  [[nodiscard]] SlotOutcome<ResolvedFootprint> ReplayMemory(const RecordView &, const MemoryRequest &);
  [[nodiscard]] SlotOutcome<ResolvedTma> ReplayTma(const RecordView &, const TmaRequest &);
  void AdmitGrid(const Instruction &) const;
  Integer AddressRegisterBits(const RecordView &, std::string_view, const Integer &);

private:
  friend class SlotResolver;
  UnitInterner interner_;
  std::uint16_t id_; // FactSource::NewResourcesId(); events only
  std::unique_ptr<SlotResolver> slots_;
  std::unique_ptr<RegisterTable> address_register_types_;
  mutable std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> address_instruction_registers_;
  std::map<std::string, nlohmann::json> records_;
  std::set<std::string> shared_symbols_;
  struct DynamicHash {
    std::size_t operator()(const DynamicKey &key) const noexcept {
      return std::hash<std::uint64_t>{}(key.occurrence) ^
             (std::size_t(key.instruction) * 0x9e3779b97f4a7c15ULL);
    }
  };
  // The facts' lane records are immutable for this CTA. Matrix accesses
  // repeatedly switch supplying lanes; build each index once and keep each
  // lane's provenance answers in its own cache (until BeginLane). unordered_map rehash preserves
  // references to elements, so active_index_ remains valid as lanes are
  // added.
  struct OccurrenceIndex {
    std::size_t indexed = 0;
    std::unordered_map<DynamicKey, std::size_t, DynamicHash>
        executions;
  };
  mutable std::unordered_map<PathKey,OccurrenceIndex,PathKeyHash> occurrence_indices_;
  // The cached copies have an ExecutedInstruction's size (RecordCopy): an
  // entry's key fixes the lane, position, kind and slice of its read, and
  // Definition views the copy with them.
  struct LaneIndex {
    std::map<DynamicKey, RecordCopy> requested;         // ReadFull
    std::map<DynamicKey, RecordCopy> address_requested; // ReadSlice over address_instruction_registers_
    std::unordered_map<DynamicKey, std::optional<std::uint32_t>, DynamicHash>
        owners;
    std::unordered_map<DynamicKey, std::optional<std::uint32_t>, DynamicHash> generic_owners;
  };
  std::unordered_map<LaneRef, LaneIndex> lane_indices_;
  std::optional<LaneRef> active_lane_;
  LaneIndex *active_index_ = nullptr;
  // warp_id * 32 + lane_id of the active lane: Bind's thread-local owner.
  std::optional<std::uint32_t> ActiveThread() const {
    if (!active_lane_) return std::nullopt;
    return facts.Warp(*active_lane_) * 32 + facts.LaneId(*active_lane_);
  }
  // The defining record of `key` on the active lane: Expanded on an expanded
  // lane, otherwise a view of the cached Slice (address registers) or Full copy.
  std::optional<RecordView> Definition(DynamicKey, bool address_slice = false) const;
  std::optional<std::uint32_t> Owner(const RecordView &,
                                     std::string_view,
                                     std::set<DynamicKey> visiting = {});
  std::optional<std::uint32_t> GenericOwner(const RecordView &, std::string_view,
                                           std::set<DynamicKey> visiting = {});
  Scalar Value(const RecordView &, std::string_view) const;
  // Bind's validation and key (BindKey, then CheckRange) without the intern:
  // Bind applies the step at once, the slot resolver appends it.
  ConstantStep BindStep(std::string space, std::string resource, std::uint64_t offset,
                        std::uint64_t bytes, std::optional<CtaRank> owner) const;
  // Abstract's expression over its base binding's (space, allocation, alias).
  static ExpressionStep AbstractExpression(const std::string &space, const std::string &allocation,
                                           const std::string &alias, const nlohmann::json &offset,
                                           std::optional<std::uint64_t> bytes);
};
// A worker's query events (FactSource::WorkerScope), deep-copied in the order it noted them, with
// their tag and depths. Replay hands them to the consumer's sink on the consuming thread, so the
// sink sees the events the worker's unit of work would have noted there. Replayed views carry the
// noted position, kind and slice; their record is the position's metadata (a sink reads views'
// positions, kinds, slices and instructions, never their values: a value read's answer is copied).
// AddressIds in BindingRead and AbsoluteCheck are the worker's table's: Replay maps them.
// O(event size) per event both ways.
class EventTape final : public QuerySink {
public:
  void On(const QueryEvent &event, ConsumerTag consumer, bool in_slot, bool in_partition) noexcept override;
  // Whether every event was copied (a failed copy leaves the tape incomplete).
  [[nodiscard]] bool Complete() const noexcept { return complete_; }
  // Into `facts`' sink (none: nothing), on the consuming thread.
  void Replay(const FactSource &facts, std::span<const AddressId> addresses) const;
  void Clear() noexcept {
    entries_.clear();
    complete_ = true;
  }

private:
  struct View {
    RecordRef at;
    std::vector<RegId> slice;
    RecordKind kind = RecordKind::Full;
    bool positioned = false;
  };
  struct ValueRead { View view; RegId reg; std::optional<Scalar> answer; };
  struct LegacyUniform { RecordRef leader; bool expanded; std::string name; std::optional<Integer> answer;
                         std::vector<LaneRef> members; };
  struct SpecialRead { std::string name; std::optional<Integer> answer; };
  struct MemoryCopy { std::string operand; nlohmann::json contract; std::uint64_t bytes; std::string space; };
  struct TmaCopy { nlohmann::json contract; bool store; const nlohmann::json *dynamic_tensor_maps; };
  struct Slot {
    ConsumerTag consumer;
    View site;
    std::optional<MemoryCopy> memory;
    std::optional<TmaCopy> tma;
    std::optional<LaneRef> active_lane;
    std::variant<SlotOutcome<ResolvedFootprint>, SlotOutcome<ResolvedTma>> outcome;
  };
  struct UniformBodySkip { LaneRef lane; std::vector<std::size_t> positions, varying_slots; };
  struct AbsoluteCheck { AddressId id; std::uint64_t modulus; std::optional<ConstantAddressBinding> binding; };
  struct PartitionRead { RecordRef leader; std::size_t evaluated; std::uint32_t instruction;
                         std::vector<std::uint32_t> classes; query::PartitionEnd end; std::vector<LaneRef> members;
                         const PartitionSpec *spec; std::vector<std::optional<std::pair<unsigned, bool>>> shuffles; };
  struct ReportBearing { query::ReportKind kind; std::string code; };
  // A Slot, several times larger than the rest, is held out of line: an entry is then about the
  // size of a ValueRead, the most frequent event.
  using Event = std::variant<ValueRead, query::ValueAtRead, LegacyUniform, query::UniformBinding, SpecialRead,
                             query::SlotBegin, query::SlotEnd, std::unique_ptr<const Slot>, query::MemoFill,
                             query::MemoHit, UniformBodySkip, query::BindingRead, AbsoluteCheck, PartitionRead,
                             ReportBearing>;
  struct Entry {
    ConsumerTag consumer;
    bool in_slot, in_partition;
    Event event;
  };
  static View Copy(const RecordView &view);
  static RecordView Rebuild(const FactSource &facts, const View &view);
  std::vector<Entry> entries_;
  bool complete_ = true;
};
} // namespace sentryward::frontend::ptx
