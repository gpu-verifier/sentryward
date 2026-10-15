#pragma once
#include "sentryward/frontend/ptx/config.h"
#include "sentryward/frontend/ptx/module.h"
#include <functional>
#include <algorithm>
#include <span>
#include <memory>
#include <optional>
#include <string>
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
// monostate is UNKNOWN. Integer has exact unbounded Python integer semantics.
using Scalar = std::variant<std::monostate, Integer, bool, ParameterPointer,
                            AllocationPointer, ClusterSharedPointer, PointerBits, DescriptorFragment>;
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
  bool operator==(const ExecutionMetadata &) const = default;
};
using ExecutionBindings = std::vector<std::pair<std::uint32_t, Scalar>>;
class ThreadRecordPool;
class ExecutedInstruction {
public:
  ExecutedInstruction();
  [[nodiscard]] const auto &instruction_id() const { return metadata_->instruction_id; }
  [[nodiscard]] const auto &occurrence() const { return metadata_->occurrence; }
  [[nodiscard]] const auto &predicate_value() const { return metadata_->predicate_value; }
  [[nodiscard]] const auto &branch_scope_decision() const { return metadata_->branch_scope_decision; }
  [[nodiscard]] const auto &known_registers() const { return *bindings_; }
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
  [[nodiscard]] DynamicKey Key() const { return {instruction_id(), occurrence()}; }
  ExecutionMetadata &mutableMetadata();
  ExecutionBindings &mutableKnownRegisters();

private:
  friend class ThreadRecordPool;
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
class InstructionSequence {
public:
  using Storage = std::vector<ExecutedInstruction>;
  InstructionSequence() = default;
  InstructionSequence(std::size_t size, std::function<Storage()> expand,
      std::function<const ExecutedInstruction &(std::size_t)> metadata = {},
      std::function<ExecutedInstruction(std::size_t)> read = {}, const void *identity = nullptr,
      std::function<ExecutedInstruction(std::size_t,std::span<const std::uint32_t>)> read_registers = {},
      std::function<std::optional<std::size_t>(std::size_t,std::uint32_t)> register_identity = {})
      : size_(size), expand_(std::move(expand)), metadata_(std::move(metadata)), read_(std::move(read)), identity_(identity), read_registers_(std::move(read_registers)), register_identity_(std::move(register_identity)) {}
  std::size_t size() const { return expand_ ? size_ : records_.size(); }
  bool empty() const { return size() == 0; }
  bool materialized() const { return !expand_; }
  const Storage &records() const {
    if (expand_) { records_ = expand_(); expand_ = {}; }
    return records_;
  }
  const void *metadataIdentity() const { return expand_ && identity_ ? identity_ : this; }
  // Metadata never evaluates thread-dependent bindings.
  const ExecutedInstruction &metadata(std::size_t i) const {
    if (i >= size()) throw std::out_of_range("instruction index");
    return expand_ && metadata_ ? metadata_(i) : records().at(i);
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
  auto data() const { return records().data(); }
  // A pure evaluator identity, valid only within this unmodified thread sequence.
  std::optional<std::size_t> registerEvaluationIdentity(std::size_t index,std::uint32_t reg) const {
    if(index>=size())throw std::out_of_range("instruction index");
    return expand_ && register_identity_?register_identity_(index,reg):std::nullopt;
  }
  auto begin() const { return records().begin(); }
  auto end() const { return records().end(); }
  auto begin() { materialize(); return records_.begin(); }
  auto end() { materialize(); return records_.end(); }
  const auto &operator[](std::size_t i) const { return records()[i]; }
  auto &operator[](std::size_t i) { materialize(); return records_[i]; }
  const auto &at(std::size_t i) const { return records().at(i); }
  auto &at(std::size_t i) { materialize(); return records_.at(i); }
  const auto &back() const { return records().back(); }
  auto &back() { materialize(); return records_.back(); }
  const auto &front() const { return records().front(); }
  auto &front() { materialize(); return records_.front(); }
  template<class... Args> auto &emplace_back(Args&&... args) { materialize(); return records_.emplace_back(std::forward<Args>(args)...); }
  void reserve(std::size_t n) { materialize(); records_.reserve(n); }
  void push_back(ExecutedInstruction r) { materialize(); records_.push_back(std::move(r)); }
private:
  void materialize() { (void)records(); }
  std::size_t size_{};
  mutable Storage records_;
  mutable std::function<Storage()> expand_;
  std::function<const ExecutedInstruction &(std::size_t)> metadata_;
  std::function<ExecutedInstruction(std::size_t)> read_;
  const void *identity_ = nullptr;
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
  ThreadRecordStatistics record_compression;
  TemplateExecutionStatistics template_execution;
  // Indices into lane_traces; groups belong to this CTA and never rejoin.
  std::vector<std::vector<std::size_t>> thread_template_groups;
  [[nodiscard]] const Scalar *KnownRegister(const ExecutedInstruction &,
                                            std::string_view name) const;
  [[nodiscard]] const DynamicKey *RegisterOrigin(const ExecutedInstruction &,
                                                 std::string_view name) const;
};
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
