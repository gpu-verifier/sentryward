#pragma once
// The single typed boundary through which frontend consumers read one CTA's
// execution facts. Storage stays ExecutionProgram: every accessor below is one
// InstructionSequence/ExecutionProgram expression, so a read through it returns
// the same object or value as that expression.
//
//  - FactSource: the shared, concrete and non-virtual read core. Resources, the
//    address-value extractor and ValueTemplateExtractor read through it.
//  - CtaFacts: one per consumed ExecutionProgram (run.cpp builds it).
//  - ExecutorFacts: the producer's view of its in-progress program; never recorded.
//
// The class-mode event funnel (recording(), Note, ConsumerScope, SlotScope) is
// inert unless a recorder is attached, and event payloads are built only inside
// `if (recording())`. Hot accessors are inline.
//
// Consumers name no storage type: lanes are LaneRefs, records are
// RecordViews (or RecordHolders/RecordCopies where a copy is kept), launch
// configuration is a LaunchView, and the storage reads of execution.h are
// private to FactSource/CtaFacts (and detail::ExecutionStorage for producers).
//
// Path identity is the executor's PathId or the lane, never an address, and
// lowering's path-uniform reads read the shared prototype, so they do not
// depend on expansion.
#include "sentryward/algorithm/base/strong_id.h"
#include "sentryward/frontend/ptx/execution.h"
#include "sentryward/frontend/ptx/unit_frame.h"

#include <compare>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace sentryward::algorithm {
struct ConstantAddressBinding;
} // namespace sentryward::algorithm

namespace sentryward::frontend::ptx {
// Index into ExecutionProgram::register_names (names are unique).
using RegId = std::uint32_t;
// CtaFacts: the lane_traces index. ExecutorFacts: a registration index.
struct LaneRef {
  std::uint32_t index{};
  friend auto operator<=>(const LaneRef &, const LaneRef &) = default;
};
struct RecordRef {
  LaneRef lane;
  std::size_t pos{};
  friend auto operator<=>(const RecordRef &, const RecordRef &) = default;
};
} // namespace sentryward::frontend::ptx

template <> struct std::hash<sentryward::frontend::ptx::LaneRef> {
  std::size_t operator()(const sentryward::frontend::ptx::LaneRef &lane) const noexcept {
    return std::hash<std::uint32_t>{}(lane.index);
  }
};
template <> struct std::hash<sentryward::frontend::ptx::RecordRef> {
  std::size_t operator()(const sentryward::frontend::ptx::RecordRef &record) const noexcept {
    return std::hash<std::size_t>{}(record.pos) ^
           (std::size_t(record.lane.index) * 0x9e3779b97f4a7c15ULL);
  }
};

namespace sentryward::frontend::ptx {
class FactSource;
class CtaFacts;
class ExecutorFacts;
class RecordHolder;
class ExpandedRecords;

// Which object a view reads at a site. Value reads depend only on this
// object; metadata is shared by every representation of one record.
enum class RecordKind : std::uint8_t {
  Prototype,     // metadata(pos) on an unexpanded lane: the arena's shared record
  PrototypeCopy, // a by-value copy of metadata(pos) (the value view's dead positions)
  Full,          // read(pos): every known binding evaluated
  Slice,         // readRegisters(pos, regs): only `regs`; slice() names them
  Expanded,      // metadata(pos) == records().at(pos) on an expanded lane
  Detached       // producer only: a fresh Executor::Record at no trace position
};

// Non-owning and trivially copyable. Metadata only: values go through FactSource.
// A positioned view's ref() is its (lane, position); a Detached view, and the
// view of an empty RecordHolder, have none.
class RecordView {
public:
  [[nodiscard]] const auto &instruction_id() const noexcept { return record_->instruction_id(); }
  [[nodiscard]] const auto &occurrence() const noexcept { return record_->occurrence(); }
  [[nodiscard]] const auto &predicate_value() const noexcept { return record_->predicate_value(); }
  [[nodiscard]] const auto &branch_scope_decision() const noexcept { return record_->branch_scope_decision(); }
  [[nodiscard]] const auto &register_origins() const noexcept { return record_->register_origins(); }
  [[nodiscard]] const auto &collapsed_wait_loop() const noexcept { return record_->collapsed_wait_loop(); }
  [[nodiscard]] const auto &canonical_wait_probe() const noexcept { return record_->canonical_wait_probe(); }
  [[nodiscard]] const auto &normalized_wait_result() const noexcept { return record_->normalized_wait_result(); }
  [[nodiscard]] const auto &normalized_clc_result() const noexcept { return record_->normalized_clc_result(); }
  [[nodiscard]] const auto &wait_exit_label() const noexcept { return record_->wait_exit_label(); }
  [[nodiscard]] const auto &schedule_seed_outcome() const noexcept { return record_->schedule_seed_outcome(); }
  [[nodiscard]] const auto &probe_fallback_key() const noexcept { return record_->probe_fallback_key(); }
  [[nodiscard]] const auto &opaque_compute_diamond_indices() const noexcept { return record_->opaque_compute_diamond_indices(); }
  [[nodiscard]] const auto &collapsed_wait_indices() const noexcept { return record_->collapsed_wait_indices(); }
  [[nodiscard]] const auto &opaque_compute_join_label() const noexcept { return record_->opaque_compute_join_label(); }
  [[nodiscard]] const auto &assumed_equality() const noexcept { return record_->assumed_equality(); }
  [[nodiscard]] const auto &opaque_compute_register_reads() const noexcept { return record_->opaque_compute_register_reads(); }
  [[nodiscard]] const auto &opaque_compute_register_writes() const noexcept { return record_->opaque_compute_register_writes(); }
  [[nodiscard]] const auto &merged_arm_join() const noexcept { return record_->merged_arm_join(); }
  [[nodiscard]] DynamicKey Key() const { return record_->Key(); }
  [[nodiscard]] std::optional<RecordRef> ref() const noexcept {
    return has_ref_ ? std::optional<RecordRef>(RecordRef{lane_, pos_}) : std::nullopt;
  }
  [[nodiscard]] RecordKind kind() const noexcept { return kind_; }
  [[nodiscard]] std::span<const RegId> slice() const noexcept { return slice_; }

private:
  friend class FactSource;
  friend class CtaFacts;
  friend class ExecutorFacts;
  friend class RecordHolder;
  friend class ExpandedRecords;
  RecordView(const ExecutedInstruction *record, RecordRef at, RecordKind kind,
             std::span<const RegId> slice = {}) noexcept
      : record_(record), pos_(at.pos), slice_(slice), lane_(at.lane), kind_(kind), has_ref_(true) {}
  RecordView(const ExecutedInstruction *record, LaneRef lane, RecordKind kind) noexcept
      : record_(record), lane_(lane), kind_(kind) {}
  // 40 bytes: a view is built per record in the consumers' per-record loops.
  const ExecutedInstruction *record_;
  std::size_t pos_ = 0;
  std::span<const RegId> slice_;
  LaneRef lane_;
  RecordKind kind_;
  bool has_ref_ = false;
};

// Copyable owner of a PrototypeCopy / Full / Slice copy, for a consumer that
// keeps a copy. Copying it copies the record's two shared_ptrs.
// An empty holder views a default ExecutedInstruction, which is what a
// default-constructed cache entry holds if its read threw; that view is
// unpositioned (ref() is empty).
class RecordHolder {
public:
  RecordHolder() = default;
  [[nodiscard]] RecordView view() const noexcept {
    if (!record_) return RecordView(&Empty(), LaneRef{}, kind_);
    return RecordView(&*record_, at_, kind_, slice_);
  }
  explicit operator bool() const noexcept { return record_.has_value(); }

private:
  friend class FactSource;
  static const ExecutedInstruction &Empty() noexcept {
    static const ExecutedInstruction empty;
    return empty;
  }
  RecordHolder(ExecutedInstruction record, RecordRef at, RecordKind kind, std::span<const RegId> slice)
      : record_(std::move(record)), at_(at), kind_(kind), slice_(slice) {}
  std::optional<ExecutedInstruction> record_;
  RecordRef at_{};
  RecordKind kind_ = RecordKind::Full;
  std::span<const RegId> slice_;
};

// A holder's copy without its view metadata (FactSource::Keep), for objects
// that keep many copies (Resources' Definition caches) or one per extractor
// (the value view's current record). It has the size of an ExecutedInstruction,
// so keeping it costs no more than keeping the record itself; the keeper
// supplies the read's position, kind and slice again (FactSource::ViewFull,
// ViewSlice, ViewPrototypeCopy).
class RecordCopy {
public:
  RecordCopy() = default;

private:
  friend class FactSource;
  // FactSource::Keep: move-constructs from the holder's record.
  explicit RecordCopy(ExecutedInstruction &&record) noexcept : record_(std::move(record)) {}
  ExecutedInstruction record_;
};

// The records of one lane CtaFacts::ExpandLane expanded, in order, as Expanded
// views, for a range-for over that lane's records.
class ExpandedRecords {
public:
  class Iterator {
  public:
    using value_type = RecordView;
    using difference_type = std::ptrdiff_t;
    [[nodiscard]] RecordView operator*() const noexcept { return View(records_ + pos_, {lane_, pos_}); }
    Iterator &operator++() noexcept {
      ++pos_;
      return *this;
    }
    [[nodiscard]] bool operator==(const Iterator &other) const noexcept { return pos_ == other.pos_; }

  private:
    friend class ExpandedRecords;
    Iterator(const ExecutedInstruction *records, LaneRef lane, std::size_t pos) noexcept
        : records_(records), lane_(lane), pos_(pos) {}
    const ExecutedInstruction *records_;
    LaneRef lane_;
    std::size_t pos_;
  };
  [[nodiscard]] Iterator begin() const noexcept { return {records_, lane_, 0}; }
  [[nodiscard]] Iterator end() const noexcept { return {records_, lane_, size_}; }

private:
  friend class CtaFacts;
  ExpandedRecords(const ExecutedInstruction *records, std::size_t size, LaneRef lane) noexcept
      : records_(records), size_(size), lane_(lane) {}
  static RecordView View(const ExecutedInstruction *record, RecordRef at) noexcept {
    return RecordView(record, at, RecordKind::Expanded);
  }
  const ExecutedInstruction *records_;
  std::size_t size_;
  LaneRef lane_;
};

// The launch-invariant part of one CTA's FrontendConfig, as references into
// that object, so every value is the one its readers read from it directly.
// cluster_dim is the frame's; special registers and tensor maps are reached
// only through FactSource::Special and the private TensorMaps; the per-CTA keys
// of fixed_input have no accessor.
class LaunchView {
public:
  explicit LaunchView(const FrontendConfig &config) noexcept : config_(&config) {}
  [[nodiscard]] const Coordinates &block_dim() const noexcept { return config_->block_dim; }
  [[nodiscard]] const std::uint32_t &warp_size() const noexcept { return config_->warp_size; }
  [[nodiscard]] std::uint64_t Threads() const { return config_->Threads(); }
  [[nodiscard]] const std::optional<std::uint32_t> &elect_lane() const noexcept { return config_->elect_lane; }
  [[nodiscard]] const std::vector<WarpContract> &warps() const noexcept { return config_->warps; }
  [[nodiscard]] const std::map<std::string, Integer> &parameters() const noexcept { return config_->parameters; }
  [[nodiscard]] const std::map<std::string, ParameterRegion> &parameter_regions() const noexcept {
    return config_->parameter_regions;
  }
  [[nodiscard]] const std::vector<PointerParameter> &pointer_parameters() const noexcept {
    return config_->pointer_parameters;
  }
  [[nodiscard]] const nlohmann::json &resources() const noexcept { return config_->resources; }
  [[nodiscard]] const nlohmann::json &instruction_contracts() const noexcept { return config_->instruction_contracts; }
  [[nodiscard]] const nlohmann::json &InstructionContract(std::uint32_t line, std::uint32_t instruction,
                                                         std::optional<std::uint32_t> warp = {},
                                                         std::optional<std::uint64_t> occurrence = {}) const {
    return config_->InstructionContract(line, instruction, warp, occurrence);
  }
  // fixed_input.value(key, default) for the three launch keys, one accessor
  // each so that every key keeps the conversion and default its readers
  // applied.
  [[nodiscard]] bool ClusterLaunchCancellation() const {
    return config_->fixed_input.value("cluster_launch_cancellation", true);
  }
  [[nodiscard]] nlohmann::json LaunchContract() const {
    return config_->fixed_input.value("launch_contract", nlohmann::json::object());
  }
  [[nodiscard]] std::string PublicInputSchema() const { return config_->fixed_input.value("public_input_schema", ""); }

private:
  const FrontendConfig *config_;
};

// Path identity: the executor's template path (PathId) of a lane's
// sequence, or the lane itself on the lane path and for a non-template
// sequence. Read live; never expands and never depends on expansion or
// on addresses. Equal key implies an identical ExecutionMetadata
// sequence, not identical records: an expanded lane's records carry its
// own bindings. Pure cache key.
class PathKey {
public:
  PathKey() = default;
  friend bool operator==(const PathKey &, const PathKey &) = default;
  friend std::strong_ordering operator<=>(const PathKey &, const PathKey &) = default;
  [[nodiscard]] std::size_t Hash() const noexcept { return std::hash<std::uint64_t>{}(value_); }

private:
  friend class FactSource;
  static constexpr std::uint64_t kLane = std::uint64_t(1) << 32; // | LaneRef::index; else the PathId
  explicit PathKey(std::uint64_t value) noexcept : value_(value) {}
  std::uint64_t value_ = 0;
};
struct PathKeyHash {
  std::size_t operator()(const PathKey &key) const noexcept { return key.Hash(); }
};
// SEMANTIC: lanes sharing a key share lowering's synchronous-template
// projection, whose generated-memory sums are serialized (generated_* counts).
// The lane's PathKey, under its own type so the contract stays visible.
class ProjectionKey {
public:
  ProjectionKey() = default;
  friend bool operator==(const ProjectionKey &, const ProjectionKey &) = default;
  friend std::strong_ordering operator<=>(const ProjectionKey &, const ProjectionKey &) = default;

private:
  friend class CtaFacts;
  explicit ProjectionKey(PathKey path) noexcept : path_(path) {}
  PathKey path_;
};
// SEMANTIC: the address-value extractor's demand_peer gate. The
// lane, its path and whether it was expanded, captured at extractor creation;
// FactSource::LegacyPeerAligned compares them with the peer's live state.
class PeerToken {
public:
  PeerToken() = default;

private:
  friend class FactSource;
  PathKey path_;
  LaneRef lane_;
  bool expanded_ = false;
};

// The physical whole-lane expansion triggers.
enum class ExpansionSite : std::uint8_t {
  BlockScaleTmem, // the value view's TMEM analysis (block-scaled tcgen05)
  Matrix,         // MatrixResolver (ldmatrix/stmatrix)
  Tmem            // lowering's TMEM analysis (every tcgen05 kernel)
};

// ---- the event funnel (inert unless a sink is attached) ---------------------
// In class mode the driver attaches a recorder (query_log.h) to a
// representative CTA's facts; the query events below are the questions it logs.
class QuerySink;
enum class Consumer : std::uint8_t { Driver, ValueView, AddressIndex, Lowering, Cohorts, Executor };
struct ConsumerTag {
  Consumer kind = Consumer::Driver;
  std::uint16_t resources = 0; // Resources instance id; 0 = none
};
// An entry of lowering's evaluated-address memo.
struct MemoRef {
  RecordRef at;      // position of the fill or of the hit
  std::uint64_t key; // the evaluation key
};
struct SlotQuery;         // resources.h
struct ResolvedFootprint; // resources.h
struct ResolvedTma;       // resources.h
template <class T> struct SlotOutcome;
struct PartitionSpec;     // below
namespace query {
struct ValueRead { RecordView view; RegId reg; const Scalar *answer; };                    // Value
struct ValueAtRead { RecordRef at; RegId reg; std::optional<Scalar> answer; };            // ValueAt
// LegacyUniformOnlyValue. `members`: the span it received.
struct LegacyUniform { RecordRef leader; bool expanded; std::string_view name;
                       std::optional<Integer> answer; std::span<const LaneRef> members; };
struct UniformBinding { RecordRef at; RegId reg; bool present; };                         // PathUniformBinding
struct SpecialRead { std::string_view name; std::optional<Integer> answer; };             // Special
struct SlotBegin {};
struct SlotEnd {};                                                                        // bracket one slot
struct SlotMemory { const SlotQuery *query; const SlotOutcome<ResolvedFootprint> *outcome; }; // a slot outcome
struct SlotTma { const SlotQuery *query; const SlotOutcome<ResolvedTma> *outcome; };
// Lowering's evaluated-address memo. `reg`: the
// address register whose PathEvaluationKey the memo key holds.
struct MemoFill { MemoRef memo; RegId reg; };
struct MemoHit { MemoRef memo; RegId reg; };
struct UniformBodySkip { LaneRef lane; std::span<const std::size_t> positions;
                         std::span<const std::size_t> varying_slots; };                   // lowering's skipped uniform body
struct BindingRead { algorithm::AddressId id; };                                          // Resources::Binding
// An alignment check of an absolute offset. `binding`: the checked binding; its
// allocation names the δ the check constrains.
struct AbsoluteCheck { algorithm::AddressId id; std::uint64_t modulus;
                       const algorithm::ConstantAddressBinding *binding; };
// How one Partition fact stage ended. Threw: an exception cut the call short,
// so its in-partition reads have no complete answer (never mergeable).
enum class PartitionEnd : std::uint8_t { Complete, Vetoed, Threw };
// One Partition fact stage. Derived answers only: `classes` numbers the
// evaluated members' stage-1 keys by first occurrence (never the raw values).
// Self-contained: the classes are copied out of the cursor. It also carries
// the cursor's member span and rule (to re-run the stage on a member) and,
// per evaluated member, the shuffle source the consumer stage looks up
// (MemberFacts::shuffle).
struct PartitionRead { RecordRef leader; std::size_t evaluated; std::uint32_t instruction;
                       std::vector<std::uint32_t> classes; PartitionEnd end;
                       std::span<const LaneRef> members; const PartitionSpec *spec;
                       std::vector<std::optional<std::pair<unsigned, bool>>> shuffles; };
// An event that puts per-CTA evidence into the report. A lowering
// Diagnostic closes the class; a rendered violation is re-rendered per
// member and does not.
enum class ReportKind : std::uint8_t { LoweringDiagnostic, RenderedViolation };
struct ReportBearing { ReportKind kind; std::string_view code; };
} // namespace query
using QueryEvent = std::variant<query::ValueRead, query::ValueAtRead, query::LegacyUniform, query::UniformBinding,
                                query::SpecialRead, query::SlotBegin, query::SlotEnd, query::SlotMemory,
                                query::SlotTma, query::MemoFill, query::MemoHit, query::UniformBodySkip,
                                query::BindingRead, query::AbsoluteCheck, query::PartitionRead,
                                query::ReportBearing>;
// The recorder interface. noexcept: SlotScope's
// destructor notes SlotEnd; a recorder failure only closes its class.
class QuerySink {
public:
  virtual void On(const QueryEvent &event, ConsumerTag consumer, bool in_slot, bool in_partition) noexcept = 0;

protected:
  ~QuerySink() = default;
};

// ---- Partition facts --------------------------------------------------------
// The static rule of one instruction the value view's Partition whitelists:
// the fragments of the per-member key, in key order (warp, issuer, bulk/param
// tokens, tcgen05.mma tokens, tensor contract, copy tokens, election,
// shuffle). Function text only: the value view builds it once per function and
// shares it with every CTA; never consumer state.
struct PartitionFragment {
  enum Kind : std::uint8_t {
    Warp,     // "warp:" Warp(member)
    Issuer,   // "issuer:" (LaneId(member) == 0)
    Token,    // name "=" ScalarJson(value) ";" when the member's value of `name` is known
    Contract, // the tensor InstructionContract(line, id, Warp(member), occurrence) as JSON
    Elected,  // "elected:" (LaneId(member) == chosen) when the mask is a literal or known
    Shuffle,  // ShuffleSource: "valid:" b, or "invalid:" LaneId(member) if it throws std::invalid_argument
    Missing   // operands.at(operand) throws here, where the key builder read the operand
  };
  Kind kind = Token;
  std::string name;               // Token: one match of the key builder's regex; Elected: the trimmed operand
  std::optional<Integer> literal; // Elected: ParseInteger(name)
  std::uint32_t operand = 0;      // Elected, Missing: the operand index
};
struct PartitionSpec {
  std::uint32_t instruction = 0;
  std::uint32_t record_at = 0; // the member's metadata is read before fragments[record_at]
  std::vector<PartitionFragment> fragments;
};
// What the fact stage derived for one member, for the consumer stage.
struct MemberFacts {
  std::optional<std::pair<unsigned, bool>> shuffle; // ShuffleSource: (source lane, valid); none if it threw
};
class PartitionCursor;
// Instruction-text helpers (defined in resources.cpp), declared here so that
// the fact layer does not depend on Resources.
std::string Trim(std::string_view);
bool Has(const Instruction &, std::string_view);
// The source lane of shfl.sync for `at.lane` at `at.pos` and whether it is
// valid (the shuffle rule of the value view's key). Throws
// std::invalid_argument when the selector, clamp or member mask is not known
// or the lane or its source is outside the member mask.
[[nodiscard]] std::pair<unsigned, bool> ShuffleSource(const FactSource &facts, RecordRef at);

// A shared-value site relative to the consuming CTA's cluster.
// `owner` is TryRankOf of the site's CTA: absent for a CTA outside the cluster.
struct SharedSiteRef {
  std::optional<CtaRank> owner;
  std::uint32_t thread{};
  DynamicKey instruction;
};

// ---- the read core ----------------------------------------------------------
class FactSource {
public:
  FactSource(const FactSource &) = delete;
  FactSource &operator=(const FactSource &) = delete;

  // static / launch (never an event)
  [[nodiscard]] const Module &module() const noexcept { return *program_->module; }
  [[nodiscard]] const Function &function() const noexcept { return *program_->function; }
  // Whether the program names its module and function (every specialized one does).
  [[nodiscard]] bool Specialized() const noexcept { return program_->module && program_->function; }
  [[nodiscard]] const LaunchView &launch() const noexcept { return launch_; }
  // One name -> id table, built lazily on first use (first id wins). Views
  // into register_names, final before any read.
  // Stored as one sorted array (binary search), not one node per register, so
  // the table lives in a single allocation of 24 bytes per register.
  [[nodiscard]] std::optional<RegId> Reg(std::string_view exact_name) const;
  // RecoverTensorMaps' reads of runtime-written tensor maps (structure; never an event).
  [[nodiscard]] const auto &TensorMapReads() const noexcept { return program_->tensor_map_reads; }

  // lanes (structure; never an event). Always read live.
  [[nodiscard]] std::vector<LaneRef> ProgramLanes() const;
  [[nodiscard]] std::optional<LaneRef> FindLane(std::uint32_t warp, std::uint32_t lane) const;
  [[nodiscard]] std::uint32_t Warp(LaneRef lane) const { return TraceOf(lane).warp_id; }
  [[nodiscard]] std::uint32_t LaneId(LaneRef lane) const { return TraceOf(lane).lane_id; }
  [[nodiscard]] std::size_t Size(LaneRef lane) const { return TraceOf(lane).instructions.size(); }
  [[nodiscard]] bool Expanded(LaneRef lane) const { return TraceOf(lane).instructions.materialized(); }
  // The executor's template path of the lane's sequence; none on the lane path.
  [[nodiscard]] std::optional<PathId> TemplatePath(LaneRef lane) const {
    return TraceOf(lane).instructions.templatePath();
  }
  // CtaFacts: kLane | the lane_traces index; ExecutorFacts: | the registration index.
  [[nodiscard]] PathKey Path(LaneRef lane) const;
  [[nodiscard]] PeerToken PeerPathToken(LaneRef lane) const;
  // The address-value extractor's peer gate, without addresses: an unexpanded
  // token aligns with an unexpanded peer on the same path; an expanded token
  // only with itself, still expanded. So after expansion every peer demand
  // fails except the self-peer.
  [[nodiscard]] bool LegacyPeerAligned(const PeerToken &token, LaneRef peer) const;

  // records
  [[nodiscard]] RecordView Meta(RecordRef at) const; // == metadata(pos): Prototype, or Expanded
  [[nodiscard]] RecordHolder CopyMeta(RecordRef at) const; // == a copy of metadata(pos)
  [[nodiscard]] RecordHolder ReadFull(RecordRef at) const; // == read(pos)
  // == readRegisters(pos, regs); `regs` must outlive the holder.
  [[nodiscard]] RecordHolder ReadSlice(RecordRef at, std::span<const RegId> regs) const;
  // The holder's copy, moved out (a default record for an empty holder), for
  // an object that keeps the copy beyond the holder.
  [[nodiscard]] RecordCopy Keep(RecordHolder &&holder) const noexcept {
    return holder.record_ ? RecordCopy(std::move(*holder.record_)) : RecordCopy{};
  }
  // A view of a kept copy, read at `at` by CopyMeta, ReadFull or ReadSlice
  // (over `slice`, whose storage must outlive the view). Only the copy kinds
  // are expressible: a copy never poses as a Prototype or Expanded element,
  // whose addresses the TMEM and Matrix RecordRef keys stand for.
  [[nodiscard]] RecordView ViewPrototypeCopy(const RecordCopy &copy, RecordRef at) const noexcept {
    return RecordView(&copy.record_, at, RecordKind::PrototypeCopy);
  }
  [[nodiscard]] RecordView ViewFull(const RecordCopy &copy, RecordRef at) const noexcept {
    return RecordView(&copy.record_, at, RecordKind::Full);
  }
  [[nodiscard]] RecordView ViewSlice(const RecordCopy &copy, RecordRef at,
                                     std::span<const RegId> slice) const noexcept {
    return RecordView(&copy.record_, at, RecordKind::Slice, slice);
  }
  // Whether both views read one stored metadata object; then every metadata
  // accessor above agrees. A fast path before a field-wise comparison only
  // (BuildCohorts' lanes share their template's metadata): never a key, and a
  // false answer says nothing.
  [[nodiscard]] static bool SameMetadata(const RecordView &a, const RecordView &b) noexcept {
    return a.record_->metadata_ == b.record_->metadata_;
  }

  // values (ValueRead and ValueAtRead events)
  [[nodiscard]] const Scalar *Value(const RecordView &view, RegId reg) const; // first binding with id == reg
  // The first binding whose register name is `name` (a string compare per binding).
  [[nodiscard]] const Scalar *Value(const RecordView &view, std::string_view name) const;
  // == readRegisters(pos, {reg}) + the first binding with id == reg.
  [[nodiscard]] std::optional<Scalar> ValueAt(RecordRef at, RegId reg) const;
  // The first register origin whose register name is `name`. Structure, never an event.
  [[nodiscard]] const DynamicKey *Origin(const RecordView &view, std::string_view name) const;
  // == config.special_registers lookup (a SpecialRead event). Takes the caller's string:
  // the map is not transparent, so a string_view key would allocate per read.
  [[nodiscard]] std::optional<Integer> Special(const std::string &name) const;

  // For ValueTemplateExtractor::Known(text, uniform_only=true) only, reached
  // from selp in cohort-template mode: the value in the leader's shared
  // template record (sharedMetadata, which expansion never replaces) when its
  // sequence has one; otherwise every member's metadata() value must be known
  // and agree. Empty `members` means {leader}. A LegacyUniform event.
  [[nodiscard]] std::optional<Integer> LegacyUniformOnlyValue(RecordRef leader, std::span<const LaneRef> members,
                                                            std::string_view name) const;

  // The fact stage of the value view's Partition at `pos`:
  // the members' keys under `spec`, one member per PartitionCursor::Next.
  // `members` and `spec` must outlive the cursor.
  [[nodiscard]] PartitionCursor BeginPartition(std::span<const LaneRef> members, std::size_t pos,
                                               const PartitionSpec &spec) const;

  // Class-mode plumbing (inert unless recording)
  [[nodiscard]] bool recording() const noexcept { return sink_ != nullptr; }
  void Note(const QueryEvent &event) const; // every call site is `if (recording())`
  class ConsumerScope;                      // sets/restores the consumer tag only
  class SlotScope;                          // ++/-- the slot depth only
  class PartitionScope;                     // ++/-- the partition depth only
  [[nodiscard]] std::uint16_t NewResourcesId() const noexcept { return ++resources_ids_; }

protected:
  // `registered`: ExecutorFacts' lane registry; null for CtaFacts (lane_traces indices).
  FactSource(const ExecutionProgram &program, const std::deque<const LaneTrace *> *registered) noexcept
      : program_(&program), registered_(registered), launch_(program.config) {}
  ~FactSource() = default;
  // The trace a LaneRef names: CtaFacts, a lane_traces index (bounds-checked);
  // ExecutorFacts, a registration.
  [[nodiscard]] const LaneTrace &TraceOf(LaneRef lane) const;
  // The sequence's shared template record regardless of expansion; null for a
  // lane-path sequence (InstructionSequence::sharedMetadata).
  [[nodiscard]] const ExecutedInstruction *PrototypeRecord(RecordRef at) const {
    return TraceOf(at.lane).instructions.sharedMetadata(at.pos);
  }
  [[nodiscard]] const Scalar *ById(const ExecutedInstruction &record, RegId reg) const {
    for (const auto &[id, value] : record.known_registers())
      if (id == reg) return &value;
    return nullptr;
  }
  const ExecutionProgram *program_;
  const std::deque<const LaneTrace *> *registered_;
  LaunchView launch_;
  QuerySink *sink_ = nullptr;
  mutable ConsumerTag current_;
  mutable unsigned slot_depth_ = 0;
  mutable unsigned partition_depth_ = 0;
  mutable std::uint16_t resources_ids_ = 0;
  // Sorted by (name, id); views into program_->register_names.
  mutable std::optional<std::vector<std::pair<std::string_view, RegId>>> regs_;

private:
  friend class SlotResolver; // TensorMaps: SlotResolver::TensorMap only (inside Tma)
  // == config.tma_descriptors, including the executed.* and runtime.* maps.
  [[nodiscard]] const nlohmann::json &TensorMaps() const noexcept { return program_->config.tma_descriptors; }
  // The LaneRef of a trace: its lane_traces index, or ExecutorFacts::Register.
  [[nodiscard]] LaneRef LaneOf(const LaneTrace &trace) const;
  [[nodiscard]] const Scalar *ByName(const ExecutedInstruction &record, std::string_view name) const {
    for (const auto &[id, value] : record.known_registers())
      if (program_->register_names[id] == name) return &value;
    return nullptr;
  }
  LaneRef RegisterLane(const LaneTrace &trace) const; // ExecutorFacts::Register
  void NoteValue(const RecordView &view, std::string_view name, const Scalar *answer) const;
};

class FactSource::ConsumerScope {
public:
  ConsumerScope(const FactSource &facts, ConsumerTag tag) noexcept : facts_(facts), saved_(facts.current_) {
    facts.current_ = tag;
  }
  ~ConsumerScope() { facts_.current_ = saved_; }
  ConsumerScope(const ConsumerScope &) = delete;
  ConsumerScope &operator=(const ConsumerScope &) = delete;

private:
  const FactSource &facts_;
  ConsumerTag saved_;
};
// The bracket of one slot resolution (the SlotBegin/SlotEnd events).
class FactSource::SlotScope {
public:
  // Not noexcept: Note may throw once a sink is attached. The destructor is
  // implicitly noexcept, so a sink must not throw from SlotEnd. SlotBegin is
  // noted inside the slot (depth already raised); if Note throws, the destructor
  // does not run, so the increment is undone before rethrowing.
  explicit SlotScope(const FactSource &facts) : facts_(facts) {
    ++facts.slot_depth_;
    if (facts.recording()) [[unlikely]] {
      try {
        facts.Note(query::SlotBegin{});
      } catch (...) {
        --facts.slot_depth_;
        throw;
      }
    }
  }
  ~SlotScope() {
    if (facts_.recording()) [[unlikely]]
      facts_.Note(query::SlotEnd{});
    --facts_.slot_depth_;
  }
  SlotScope(const SlotScope &) = delete;
  SlotScope &operator=(const SlotScope &) = delete;

private:
  const FactSource &facts_;
};
// The fact reads of one Partition member (PartitionCursor::Next): value reads
// inside it are in_partition (replay compares the stage's PartitionRead classes,
// not these reads). The consumer stage between two members is outside it.
class FactSource::PartitionScope {
public:
  explicit PartitionScope(const FactSource &facts) noexcept : facts_(facts) { ++facts.partition_depth_; }
  ~PartitionScope() { --facts_.partition_depth_; }
  PartitionScope(const PartitionScope &) = delete;
  PartitionScope &operator=(const PartitionScope &) = delete;

private:
  const FactSource &facts_;
};

// The fact stage of one Partition call, evaluated member by
// member so that the consumer stage of a member runs right after its fact
// stage and before the next member's: vetoes and exceptions happen at the
// member and at the fragment that raises them.
// Next() is the member's stage-1 key: the part of its key before the first
// consumer-state part. Class(k) is the answer:
// first-occurrence numbering of the evaluated members' stage-1 keys, computed
// for every member (the value view groups by it).
class PartitionCursor {
public:
  PartitionCursor(const PartitionCursor &) = delete;
  PartitionCursor &operator=(const PartitionCursor &) = delete;
  [[nodiscard]] bool Done() const noexcept { return next_ >= members_.size(); }
  [[nodiscard]] LaneRef Member() const {
    if (Done()) throw std::out_of_range("Partition cursor past its last member");
    return members_[next_];
  }
  // Evaluates the next member: its stage-1 key (valid until the next call) and
  // the facts the consumer stage needs.
  std::string_view Next(MemberFacts &facts);
  [[nodiscard]] std::size_t Evaluated() const noexcept { return next_; }
  [[nodiscard]] std::uint32_t Class(std::size_t index) const { return classes_.at(index); }
  // The PartitionRead event (only if recording()), once: the first call wins, so the caller's
  // exception path may end a stage that a veto or completion already ended.
  void Finish(query::PartitionEnd end);

private:
  friend class FactSource;
  PartitionCursor(const FactSource &facts, std::span<const LaneRef> members, std::size_t pos,
                  const PartitionSpec &spec);
  const FactSource &facts_;
  std::span<const LaneRef> members_;
  std::size_t pos_;
  const PartitionSpec &spec_;
  const Instruction &instruction_;
  std::vector<std::optional<RegId>> regs_; // per fragment: Reg(name) in this program
  std::string text_;
  std::vector<std::uint32_t> classes_;
  std::unordered_map<std::string, std::uint32_t> numbering_;
  // Per evaluated member, MemberFacts::shuffle; filled only while recording (PartitionRead).
  std::vector<std::optional<std::pair<unsigned, bool>>> shuffles_;
  std::size_t next_ = 0;
  bool finished_ = false;
};

// ---- the consumer facade: one per consumed program --------------------------
class CtaFacts final : public FactSource {
public:
  // `linear`: run.cpp's linear CTA id, narrowed to uint32 for every consumer.
  CtaFacts(const ExecutionProgram &program, const UnitFrame &frame, std::uint32_t linear) noexcept
      : FactSource(program, nullptr), frame_(frame), linear_(linear) {}
  [[nodiscard]] const UnitFrame &frame() const noexcept { return frame_; }
  [[nodiscard]] std::uint32_t Linear() const noexcept { return linear_; }

  // structure (never an event)
  [[nodiscard]] std::size_t LaneCount() const noexcept { return program_->lane_traces.size(); }
  [[nodiscard]] LaneRef Lane(std::size_t index) const noexcept { return {static_cast<std::uint32_t>(index)}; }
  // Template execution: drops the value caches of `lane` (of every lane
  // without one), the memoization behind lazy reads; a later read recomputes
  // the same answers. Memory only; no event.
  void ReleaseValues(std::optional<LaneRef> lane = {}) const {
    if (program_->release_value_caches)
      program_->release_value_caches(lane ? std::optional<std::size_t>(lane->index) : std::nullopt);
  }
  [[nodiscard]] bool TemplateExecution() const noexcept { return program_->template_execution.used; }
  [[nodiscard]] std::span<const std::vector<std::size_t>> TemplateGroups() const noexcept {
    return program_->thread_template_groups;
  }
  // Structure-key input: the split tree (index == PathId::value) and each
  // TemplateGroups() entry's path. Both empty on the lane path.
  [[nodiscard]] std::span<const TemplatePathNode> TemplatePaths() const noexcept { return program_->template_paths; }
  [[nodiscard]] std::span<const PathId> GroupPaths() const noexcept { return program_->group_paths; }
  [[nodiscard]] std::span<const SharedValueSite> SharedConstantWrites() const noexcept {
    return program_->shared_constant_writes;
  }
  [[nodiscard]] std::span<const SharedValueDependency> SharedValueDependencies() const noexcept {
    return program_->shared_value_dependencies;
  }
  // Structure-key input only. Every SharedConstantWrites() site, then each
  // SharedValueDependencies() entry's write and read site, then each TensorMapReads()
  // entry's write (lane 0 of its warp) and read site, in order.
  [[nodiscard]] std::vector<SharedSiteRef> SharedSitesRelative() const;

  // driver accounting (references: no copies)
  [[nodiscard]] const TemplateExecutionStatistics &TemplateStatistics() const noexcept {
    return program_->template_execution;
  }
  [[nodiscard]] const ThreadRecordStatistics &RecordCompression() const noexcept {
    return program_->record_compression;
  }
  [[nodiscard]] const std::map<std::string, std::set<std::uint32_t>> &TensorPropertySites() const noexcept {
    return program_->tensor_property_sites;
  }
  struct LaneAggregates {
    const std::uint64_t &assumed_assertions;
    const std::map<std::uint64_t, std::uint64_t> &opaque_regions;
    const std::map<std::uint64_t, DataBranchDecision> &data_branch_decisions;
  };
  [[nodiscard]] LaneAggregates Aggregates(LaneRef lane) const {
    const auto &trace = TraceOf(lane);
    return {trace.assumed_assertions, trace.opaque_regions, trace.data_branch_decisions};
  }
  [[nodiscard]] std::span<const std::uint64_t> StaticCounts() const noexcept {
    return program_->static_execution_counts;
  }

  // The only consumer path to records(): expands the whole lane when called
  // and returns its records in order (each Meta({lane, pos}): the same
  // element, kind Expanded).
  ExpandedRecords ExpandLane(LaneRef lane, ExpansionSite site) const;

  // Path-uniform reads for lowering's generated memory, one call
  // site each; independent of expansion. Reached only on unexpanded lanes,
  // where they equal the lane's own metadata reads.
  // uniform_global: whether the template record at `at` binds `reg` uniformly
  // (the shared prototype, which expansion never replaces); without a
  // template, whether metadata(pos) binds it. A UniformBinding event.
  [[nodiscard]] bool PathUniformBinding(RecordRef at, RegId reg) const;
  // The memo key: the arena node of the template binding of `reg` at `at`; none without a template.
  [[nodiscard]] std::optional<std::size_t> PathEvaluationKey(RecordRef at, RegId reg) const {
    return TraceOf(at.lane).instructions.pathEvaluationIdentity(at.pos, reg);
  }
  // lowering's projections key (semantic; see ProjectionKey).
  [[nodiscard]] ProjectionKey Projection(LaneRef lane) const { return ProjectionKey(Path(lane)); }
  // The value view's lane-path grouping: lanes by Path(), groups in order of
  // their first lane. On the lane path every lane is its own group, in lane order.
  [[nodiscard]] std::vector<std::vector<LaneRef>> LanePathCohorts() const;
  // Structure-key input: the ids of the registers bound on the
  // shared prototype at `at`, ascending; never their values. Empty without a
  // template (the lane path), where member replay compares every value instead.
  void PrototypeRegisterIds(RecordRef at, std::vector<RegId> &out) const;

  // Class mode: the driver attaches a recorder before a representative CTA's
  // consumers run and detaches it (nullptr) after them.
  void Attach(QuerySink *sink) noexcept { sink_ = sink; }

private:
  UnitFrame frame_;
  std::uint32_t linear_;
};

// ---- the producer's view of its in-progress program (never recorded) --------
class ExecutorFacts final : public FactSource {
public:
  // Declared before the Resources built over it, which keeps a reference to
  // it (Resources::facts).
  explicit ExecutorFacts(const ExecutionProgram &in_progress) noexcept : FactSource(in_progress, &lanes_) {}
  // Any trace, including RunCluster's coroutine-local ones; dedup by address.
  [[nodiscard]] LaneRef Register(const LaneTrace &trace) const;
  // Executor::Record: a fresh record at no trace position.
  [[nodiscard]] RecordView Detached(const ExecutedInstruction &record, LaneRef lane) const noexcept {
    return RecordView(&record, lane, RecordKind::Detached);
  }

private:
  mutable std::deque<const LaneTrace *> lanes_; // node-stable; grows lazily
  mutable std::unordered_map<const LaneTrace *, LaneRef> ids_;
};


// ---- inline definitions -----------------------------------------------------
inline LaneRef FactSource::LaneOf(const LaneTrace &trace) const {
  if (registered_) [[unlikely]]
    return RegisterLane(trace);
  const auto &traces = program_->lane_traces;
  const auto *first = traces.data();
  if (std::less<const LaneTrace *>{}(&trace, first) || !std::less<const LaneTrace *>{}(&trace, first + traces.size()))
    throw std::out_of_range("lane trace outside the CTA program");
  return {static_cast<std::uint32_t>(&trace - first)};
}
inline const LaneTrace &FactSource::TraceOf(LaneRef lane) const {
  if (registered_) [[unlikely]]
    return *registered_->at(lane.index);
  return program_->lane_traces.at(lane.index);
}
inline PathKey FactSource::Path(LaneRef lane) const {
  const auto path = TemplatePath(lane);
  return PathKey(path ? std::uint64_t(path->value) : PathKey::kLane | lane.index);
}
inline PeerToken FactSource::PeerPathToken(LaneRef lane) const {
  PeerToken token;
  token.path_ = Path(lane);
  token.lane_ = lane;
  token.expanded_ = Expanded(lane);
  return token;
}
inline bool FactSource::LegacyPeerAligned(const PeerToken &token, LaneRef peer) const {
  return Expanded(peer) ? token.expanded_ && peer == token.lane_ : !token.expanded_ && Path(peer) == token.path_;
}
inline RecordView FactSource::Meta(RecordRef at) const {
  const auto &sequence = TraceOf(at.lane).instructions;
  const auto *record = &sequence.metadata(at.pos);
  return RecordView(record, at, sequence.materialized() ? RecordKind::Expanded : RecordKind::Prototype);
}
inline const Scalar *FactSource::Value(const RecordView &view, RegId reg) const {
  const auto *answer = ById(*view.record_, reg);
  if (recording()) [[unlikely]]
    Note(query::ValueRead{view, reg, answer});
  return answer;
}
inline const Scalar *FactSource::Value(const RecordView &view, std::string_view name) const {
  const auto *answer = ByName(*view.record_, name);
  if (recording()) [[unlikely]]
    NoteValue(view, name, answer); // Reg(name) is computed only here
  return answer;
}
inline const DynamicKey *FactSource::Origin(const RecordView &view, std::string_view name) const {
  for (const auto &[id, value] : view.record_->register_origins())
    if (program_->register_names[id] == name) return &value;
  return nullptr;
}
} // namespace sentryward::frontend::ptx
