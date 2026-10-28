#pragma once
#include "sentryward/frontend/ptx/config.h"
#include "sentryward/frontend/ptx/module.h"
#include <functional>
#include <algorithm>
#include <map>
#include <set>
#include <span>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace sentryward::frontend::ptx {
struct LocalMemoryDeclaration {
  std::string name, resource;
  std::vector<std::uint32_t> scope;
  std::uint64_t bytes;
};
std::vector<LocalMemoryDeclaration> LocalMemoryDeclarations(const Module&, const Function&);
struct ExecutionError : std::runtime_error {
  std::string code;
  std::uint32_t line;
  nlohmann::json required_parameters;
  bool shared_control = false;
  ExecutionError(std::string code, std::uint32_t line, std::string message,
                 nlohmann::json required = nlohmann::json::array())
      : std::runtime_error(std::move(message)), code(std::move(code)), line(line),
        required_parameters(std::move(required)) {}
};
struct ParameterPointer {
  std::string region;
  Integer offset = 0;
  bool operator==(const ParameterPointer &) const = default;
};
struct AllocationPointer {
  std::string resource;
  Integer offset = 0;
  std::optional<std::uint64_t> extent_bytes;
  bool operator==(const AllocationPointer &) const = default;
};
struct ClusterSharedPointer {
  Integer local_address = 0;
  std::uint32_t target_rank{};
  bool operator==(const ClusterSharedPointer &) const = default;
};
using PointerValue=std::variant<ParameterPointer,AllocationPointer,ClusterSharedPointer>;
struct PointerBits {
  std::shared_ptr<const PointerValue> pointer;
  unsigned source_bit{}, width{}, destination_bit{};
  bool operator==(const PointerBits &other) const {
    return pointer && other.pointer && *pointer==*other.pointer && source_bit==other.source_bit &&
        width==other.width && destination_bit==other.destination_bit;
  }
};
struct DescriptorFragment {
  std::string descriptor;
  unsigned offset{}, bytes{};
  bool operator==(const DescriptorFragment &) const = default;
};
// One element of an input object with tensor_properties, as an address-model atom.
struct ElementAtom {
  std::string object, element_type;
  Integer byte_offset;
  bool operator==(const ElementAtom &) const = default;
  bool operator<(const ElementAtom &other) const {
    if (object != other.object) return object < other.object;
    if (element_type != other.element_type) return element_type < other.element_type;
    return byte_offset < other.byte_offset;
  }
};
// constant + sum of coefficient x atom; terms sorted by atom, coefficients nonzero.
struct AffineForm {
  Integer constant;
  std::vector<std::pair<ElementAtom, Integer>> terms;
  bool operator==(const AffineForm &) const = default;
};
// A function of one small-domain bounded key: values[k - key_lower] when the bounded
// value named `key` is k (the key's range has values.size() elements).
struct BoundedTable {
  std::string key;
  Integer key_lower;
  std::vector<Integer> values;
};
// An unknown integer known to lie in [lower, upper], from an element of an input object
// with trusted tensor_properties (or simple arithmetic on one). Exact, like Integer.
// Equal identities denote equal values in every execution; `sources` names the objects;
// `affine` is the exact value in element atoms when it is affine in them; `table`, the
// exact value as a function of a small-domain key (non-affine values only).
struct BoundedValue {
  Integer lower, upper;
  std::string identity;
  std::vector<std::string> sources;
  std::optional<AffineForm> affine;
  std::optional<BoundedTable> table = std::nullopt;
};
struct BoundedInteger {
  std::shared_ptr<const BoundedValue> value;
  const BoundedValue *operator->() const noexcept { return value.get(); }
  bool operator==(const BoundedInteger &other) const {
    return value && other.value && value->identity == other.value->identity &&
        value->lower == other.value->lower && value->upper == other.value->upper;
  }
};
// A pointer into `resource` at a byte offset affine in element atoms, with the offset's
// exact range [lower, upper] within [0, 2^63). A non-affine offset has a nonempty
// `offset_identity` (`offset` is then 0) and its exact byte offsets over a small-domain key.
struct BoundedPointerValue {
  std::string resource;
  std::optional<std::uint64_t> extent_bytes;
  AffineForm offset;
  Integer lower, upper;
  std::vector<std::string> sources;
  std::string offset_identity = {};
  std::optional<BoundedTable> table = std::nullopt;
};
struct BoundedAllocationPointer {
  std::shared_ptr<const BoundedPointerValue> value;
  const BoundedPointerValue *operator->() const noexcept { return value.get(); }
  bool operator==(const BoundedAllocationPointer &other) const {
    return value && other.value && value->resource == other.value->resource && value->offset == other.value->offset &&
        value->offset_identity == other.value->offset_identity;
  }
};
// monostate is UNKNOWN. Integer has exact unbounded Python integer semantics.
using Scalar = std::variant<std::monostate, Integer, bool, ParameterPointer,
                            AllocationPointer, ClusterSharedPointer, PointerBits, DescriptorFragment,
                            BoundedInteger, BoundedAllocationPointer>;
std::optional<PointerBits> PointerFragment(const Scalar &,unsigned source,unsigned width);
Scalar RestorePointer(const PointerBits &);
struct DynamicKey {
  std::uint32_t instruction{};
  std::uint64_t occurrence{};
  auto operator<=>(const DynamicKey &) const = default;
};
struct SharedValueSite {
  Coordinates cta{};
  std::uint32_t thread{};
  DynamicKey instruction;
  auto operator<=>(const SharedValueSite &) const = default;
};
struct SharedValueDependency {
  SharedValueSite write, read;
  auto operator<=>(const SharedValueDependency &) const = default;
};
// A TMA read of a tensor map that a tensormap.cp_fenceproxy writes: the copy it reads from
// (a warp occurrence) and that copy's version, a key of config.tma_descriptors.
struct TensorMapRead {
  std::uint32_t writer_warp{};
  DynamicKey write;
  std::string version;
};
// An observation, not a proved predicate. Present only during explicitly
// requested, expression-only capture of the unchanged input's user choice.
struct BranchScopeDecision {
  bool taken{};
  std::uint64_t visit{};
  std::size_t user_decision{};
  auto operator<=>(const BranchScopeDecision&) const = default;
};
struct ExecutionMetadata {
  std::uint32_t instruction_id{};
  std::uint64_t occurrence{};
  std::optional<bool> predicate_value;
  std::optional<BranchScopeDecision> branch_scope_decision;
  std::vector<std::pair<std::uint32_t, DynamicKey>> register_origins;
  bool collapsed_wait_loop = false, canonical_wait_probe = false;
  std::optional<bool> normalized_wait_result;
  std::optional<bool> normalized_clc_result;
  std::optional<std::string> wait_exit_label, schedule_seed_outcome;
  std::optional<DynamicKey> probe_fallback_key;
  std::vector<std::uint32_t> opaque_compute_diamond_indices,
      collapsed_wait_indices;
  std::optional<std::string> opaque_compute_join_label;
  std::optional<std::pair<std::string, Integer>> assumed_equality;
  std::vector<std::string> opaque_compute_register_reads, opaque_compute_register_writes;
  // Branch of a merged one-armed diamond (two-arm merge): its predicate is
  // unknown and the fall-through arm up to this join pc is executed by all.
  std::optional<std::uint32_t> merged_arm_join;
  bool operator==(const ExecutionMetadata &) const = default;
};
using ExecutionBindings = std::vector<std::pair<std::uint32_t, Scalar>>;
// Executor-assigned identity of one template path object: dense per
// program, in creation order. Stable under expansion;
// never derived from addresses or content.
struct PathId {
  std::uint32_t value{};
  friend auto operator<=>(const PathId &, const PathId &) = default;
};
// Why a template path object was copied from its parent.
enum class SplitReason : std::uint8_t { Root, Predicate, OpaqueDomain, IndirectBranch, CollectiveWait };
// One node of a template program's split tree (index == PathId::value).
struct TemplatePathNode {
  std::optional<PathId> parent;       // the path this one was copied from (none for the root)
  std::uint32_t prefix = 0;           // the parent's path length at the copy (entries copied)
  std::uint32_t pc = 0;               // the instruction at which the fork happened
  SplitReason reason = SplitReason::Root;
  std::optional<std::uint32_t> group; // index into thread_template_groups once finished
};
class ThreadRecordPool;
// The only readers of record values and lane storage besides the producer
// (unit_facts.h); detail::ExecutionStorage serves producer code that cannot be
// befriended by name.
class FactSource;
class CtaFacts;
namespace detail { struct ExecutionStorage; }
class ExecutedInstruction {
public:
  ExecutedInstruction();
  [[nodiscard]] const auto &instruction_id() const { return metadata_->instruction_id; }
  [[nodiscard]] const auto &occurrence() const { return metadata_->occurrence; }
  [[nodiscard]] const auto &predicate_value() const { return metadata_->predicate_value; }
  [[nodiscard]] const auto &branch_scope_decision() const { return metadata_->branch_scope_decision; }
  [[nodiscard]] const auto &register_origins() const { return metadata_->register_origins; }
  [[nodiscard]] const auto &collapsed_wait_loop() const { return metadata_->collapsed_wait_loop; }
  [[nodiscard]] const auto &canonical_wait_probe() const { return metadata_->canonical_wait_probe; }
  [[nodiscard]] const auto &normalized_wait_result() const { return metadata_->normalized_wait_result; }
  [[nodiscard]] const auto &normalized_clc_result() const { return metadata_->normalized_clc_result; }
  [[nodiscard]] const auto &wait_exit_label() const { return metadata_->wait_exit_label; }
  [[nodiscard]] const auto &schedule_seed_outcome() const { return metadata_->schedule_seed_outcome; }
  [[nodiscard]] const auto &probe_fallback_key() const { return metadata_->probe_fallback_key; }
  [[nodiscard]] const auto &opaque_compute_diamond_indices() const { return metadata_->opaque_compute_diamond_indices; }
  [[nodiscard]] const auto &collapsed_wait_indices() const { return metadata_->collapsed_wait_indices; }
  [[nodiscard]] const auto &opaque_compute_join_label() const { return metadata_->opaque_compute_join_label; }
  [[nodiscard]] const auto &assumed_equality() const { return metadata_->assumed_equality; }
  [[nodiscard]] const auto &opaque_compute_register_reads() const { return metadata_->opaque_compute_register_reads; }
  [[nodiscard]] const auto &opaque_compute_register_writes() const { return metadata_->opaque_compute_register_writes; }
  [[nodiscard]] const auto &merged_arm_join() const { return metadata_->merged_arm_join; }
  [[nodiscard]] DynamicKey Key() const { return {instruction_id(), occurrence()}; }
  ExecutionMetadata &mutableMetadata();
  ExecutionBindings &mutableKnownRegisters();

private:
  friend class ThreadRecordPool;
  friend class FactSource;
  friend class CtaFacts;
  friend struct detail::ExecutionStorage;
  // Values: consumers read them through FactSource::Value.
  [[nodiscard]] const auto &known_registers() const { return *bindings_; }
  std::shared_ptr<ExecutionMetadata> metadata_;
  std::shared_ptr<ExecutionBindings> bindings_;
};
struct ThreadRecordStatistics {
  std::uint64_t record_count{}, template_count{}, unique_binding_count{};
  std::uint64_t estimated_owned_bytes{};
};
class ThreadRecordPool {
public:
  ThreadRecordPool();
  ~ThreadRecordPool();
  ThreadRecordPool(ThreadRecordPool &&) noexcept;
  ThreadRecordPool &operator=(ThreadRecordPool &&) noexcept;
  ThreadRecordPool(const ThreadRecordPool &) = delete;
  ThreadRecordPool &operator=(const ThreadRecordPool &) = delete;
  // Start each lane at cursor zero; divergent prefixes never merge.
  void Intern(ExecutedInstruction &, std::uint64_t &cursor);
  [[nodiscard]] ThreadRecordStatistics Statistics() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
// Materialize concrete thread occurrences only when a consumer requests them.
// Reads are private: consumers go through FactSource (unit_facts.h), producer
// code that cannot be befriended by name through detail::ExecutionStorage. The
// non-const element access below is the producer's mutation API; consumers
// hold only const facts.
class InstructionSequence {
public:
  using Storage = std::vector<ExecutedInstruction>;
  InstructionSequence() = default;
  InstructionSequence(std::size_t size, std::function<Storage()> expand,
      std::function<const ExecutedInstruction &(std::size_t)> metadata = {},
      std::function<ExecutedInstruction(std::size_t)> read = {}, std::optional<PathId> path = {},
      std::function<ExecutedInstruction(std::size_t,std::span<const std::uint32_t>)> read_registers = {},
      std::function<std::optional<std::size_t>(std::size_t,std::uint32_t)> register_identity = {})
      : size_(size), expand_(std::move(expand)), metadata_(std::move(metadata)), read_(std::move(read)), path_(path ? path->value : kNoPath), read_registers_(std::move(read_registers)), register_identity_(std::move(register_identity)) {}
  std::size_t size() const { return expand_ ? size_ : records_.size(); }
  bool empty() const { return size() == 0; }
  bool materialized() const { return !expand_; }
  auto begin() { materialize(); return records_.begin(); }
  auto end() { materialize(); return records_.end(); }
  auto &operator[](std::size_t i) { materialize(); return records_[i]; }
  auto &at(std::size_t i) { materialize(); return records_.at(i); }
  auto &back() { materialize(); return records_.back(); }
  auto &front() { materialize(); return records_.front(); }
  template<class... Args> auto &emplace_back(Args&&... args) { materialize(); return records_.emplace_back(std::forward<Args>(args)...); }
  void reserve(std::size_t n) { materialize(); records_.reserve(n); }
  void push_back(ExecutedInstruction r) { materialize(); records_.push_back(std::move(r)); }
private:
  friend class FactSource;
  friend class CtaFacts;
  friend struct detail::ExecutionStorage;
  const Storage &records() const {
    if (expand_) { records_ = expand_(); expand_ = {}; }
    return records_;
  }
  // The executor's template path; none on the lane path and for a
  // default-constructed sequence. Independent of expansion.
  std::optional<PathId> templatePath() const {
    return path_ == kNoPath ? std::nullopt : std::optional<PathId>(PathId{path_});
  }
  // Metadata never evaluates thread-dependent bindings.
  const ExecutedInstruction &metadata(std::size_t i) const {
    if (i >= size()) throw std::out_of_range("instruction index");
    return expand_ && metadata_ ? metadata_(i) : records().at(i);
  }
  // The shared template record, or null without a template. Unlike metadata(),
  // materialization never replaces it with one thread's concrete record.
  const ExecutedInstruction *sharedMetadata(std::size_t i) const {
    if (i >= size()) throw std::out_of_range("instruction index");
    return metadata_ ? &metadata_(i) : nullptr;
  }
  // A concrete record has caller-owned lifetime; no whole-thread expansion.
  ExecutedInstruction read(std::size_t i) const {
    if (i >= size()) throw std::out_of_range("instruction index");
    return expand_ && read_ ? read_(i) : records().at(i);
  }
  ExecutedInstruction readRegisters(std::size_t i,std::span<const std::uint32_t> registers) const {
    if(i>=size())throw std::out_of_range("instruction index");
    if(expand_ && read_registers_)return read_registers_(i,registers);
    auto result=read(i);
    std::erase_if(result.mutableKnownRegisters(),[&](const auto& binding) {
      return std::find(registers.begin(),registers.end(),binding.first)==registers.end();
    });
    return result;
  }
  // A pure evaluator identity of the template binding of `reg` at `index`,
  // valid only within this template path; regardless of expansion.
  std::optional<std::size_t> pathEvaluationIdentity(std::size_t index,std::uint32_t reg) const {
    if(index>=size())throw std::out_of_range("instruction index");
    return register_identity_?register_identity_(index,reg):std::nullopt;
  }
  void materialize() { (void)records(); }
  std::size_t size_{};
  mutable Storage records_;
  mutable std::function<Storage()> expand_;
  std::function<const ExecutedInstruction &(std::size_t)> metadata_;
  std::function<ExecutedInstruction(std::size_t)> read_;
  static constexpr std::uint32_t kNoPath = UINT32_MAX;
  std::uint32_t path_ = kNoPath;
  std::function<ExecutedInstruction(std::size_t,std::span<const std::uint32_t>)> read_registers_;
  std::function<std::optional<std::size_t>(std::size_t,std::uint32_t)> register_identity_;
};
struct TemplateExecutionStatistics {
  bool used = false;
  std::uint64_t steps{}, logical_steps{}, splits{}, value_nodes{}, record_nodes{};
  std::uint64_t record_bindings{}, coordinate_bindings{}, path_entries{}, thread_memberships{};
};
// Representative control flow outcome chosen for one PTX line whose control was not determined by the fixed input.
struct DataBranchDecision {
  std::string assumption;
  std::string rule;  // fallthrough, body_iteration_bound or guard_executes
  std::uint64_t taken{}, not_taken{};
};
struct LaneTrace {
  std::uint32_t warp_id{}, lane_id{};
  InstructionSequence instructions;
  std::map<std::uint32_t, std::uint64_t> branch_visits;
  std::map<std::string, std::uint64_t> loop_backedges;
  std::uint64_t assumed_assertions = 0;
  std::map<std::uint64_t, DataBranchDecision> data_branch_decisions;  // by PTX line
  std::map<std::uint64_t, std::uint64_t> opaque_regions;  // multi-arm compute-only regions skipped

};
struct ExecutionProgram {
  const Module *module = nullptr;
  const Function *function = nullptr;
  FrontendConfig config;
  std::vector<std::string> register_names;
  std::vector<LaneTrace> lane_traces;
  std::vector<std::uint64_t> static_execution_counts;
  std::vector<SharedValueSite> shared_constant_writes;
  std::vector<SharedValueDependency> shared_value_dependencies;
  // RecoverTensorMaps (verification only): every TMA read of a runtime-written tensor map,
  // by (warp * 32 + lane, instruction occurrence). Absent when the maps were not recovered.
  std::optional<std::map<std::pair<std::uint32_t, DynamicKey>, TensorMapRead>> tensor_map_reads;
  ThreadRecordStatistics record_compression;
  TemplateExecutionStatistics template_execution;
  // Indices into lane_traces; groups belong to this CTA and never rejoin.
  std::vector<std::vector<std::size_t>> thread_template_groups;
  // Every template path object in creation order
  // (index == PathId::value), and each finished group's path, parallel to
  // thread_template_groups. Both empty on the lane path.
  std::vector<TemplatePathNode> template_paths;
  std::vector<PathId> group_paths;
  // tensor_properties: per object, the instructions whose control value was decided
  // by an element bound.
  std::map<std::string, std::set<std::uint32_t>> tensor_property_sites;
  // Template execution: drops the arena's value caches of one thread (a
  // lane_traces index) or, for nullopt, of every thread: the memoization of
  // lazy record reads (a later read recomputes). Empty on the lane path.
  std::function<void(std::optional<std::size_t>)> release_value_caches;
};
namespace detail {
// The storage reads for producer code that cannot be befriended by name: the
// executor (internal linkage in execution.cpp), RecoverTensorMaps and the
// debug trace JSON. Static forwarders only; consumers read through FactSource.
struct ExecutionStorage {
  static const ExecutionBindings &Bindings(const ExecutedInstruction &record) { return record.known_registers(); }
  static const InstructionSequence::Storage &Records(const InstructionSequence &sequence) { return sequence.records(); }
  static const ExecutedInstruction &Metadata(const InstructionSequence &sequence, std::size_t i) { return sequence.metadata(i); }
  static ExecutedInstruction Read(const InstructionSequence &sequence, std::size_t i) { return sequence.read(i); }
  // The first binding named `name`.
  static const Scalar *Value(const ExecutionProgram &program, const ExecutedInstruction &record,
                             std::string_view name) {
    for (const auto &[id, value] : record.known_registers())
      if (program.register_names[id] == name)
        return &value;
    return nullptr;
  }
};
} // namespace detail
enum class SpecializationPurpose { Verification, ValueExtraction, BranchScopeCapture, VerificationWithBranchScope };
[[nodiscard]] ExecutionProgram
SpecializeFunction(const Module &, const Function &, const FrontendConfig &,
                   std::function<void()> poll = {},
                   SpecializationPurpose purpose = SpecializationPurpose::Verification);
[[nodiscard]] std::vector<ExecutionProgram>
SpecializeCluster(const Module &, const Function &, const std::vector<FrontendConfig> &,
                  std::function<void()> poll = {},
                  SpecializationPurpose purpose = SpecializationPurpose::Verification);
[[nodiscard]] std::optional<Integer> ScalarInteger(const Scalar &value);
// isspacep.<space> of a generic address from its provenance: a pointer into a global object
// lies in the global window, outside the shared, local and const windows. Otherwise unknown.
[[nodiscard]] std::optional<bool> GenericAddressInSpace(std::string_view space, const Scalar &address,
                                                        const nlohmann::json &resources);
// Warp-level operations: their results depend on other lanes or on which lanes execute them,
// so an opaque compute region never holds one.
[[nodiscard]] bool WarpLevelOpcode(std::string_view opcode);
[[nodiscard]] std::optional<std::pair<std::string, Integer>>
SplitAddress(std::string_view text);
[[nodiscard]] std::vector<std::string> RegistersIn(std::string_view text);
[[nodiscard]] std::optional<Integer> ParseInteger(std::string_view text);
[[nodiscard]] nlohmann::json ScalarJson(const Scalar &value);
// Detailed trace JSON is for small differential fixtures only, not production.
[[nodiscard]] nlohmann::json ExecutionRecordJson(const ExecutionProgram &,
                                                 const ExecutedInstruction &);
[[nodiscard]] nlohmann::json ExecutionJson(const ExecutionProgram &program);
} // namespace sentryward::frontend::ptx
