#pragma once

#include <compare>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "sentryward/frontend/ptx/lowering.h"
#include "sentryward/frontend/ptx/value_ir.h"
#include "sentryward/value_analysis/memory_elimination.h"

namespace sentryward::frontend::ptx {
class BranchScopeCapture;

struct SemanticOccurrence final {
  std::uint32_t cta = 0, warp = 0, lane = 0, instruction = 0;
  std::uint64_t occurrence = 0;
  friend auto operator<=>(const SemanticOccurrence&, const SemanticOccurrence&) = default;
};

struct SourceMemoryKey final {
  algorithm::InstId instruction;
  algorithm::AccessOperand operand = algorithm::AccessOperand::kImmediate;
  std::uint32_t effect_index = 0, access_index = 0;
  friend auto operator<=>(const SourceMemoryKey&, const SourceMemoryKey&) = default;
};

struct SemanticDiagnostic final {
  std::string code, message;
  std::optional<SemanticOccurrence> occurrence;
  std::optional<algorithm::InstId> instruction;
  std::optional<value_analysis::MemoryInstance> value;
};

struct SemanticMemoryBinding final {
  SourceMemoryKey source;
  SemanticOccurrence occurrence;
  value_analysis::MemoryInstance value;
  algorithm::AccessMode mode = algorithm::AccessMode::kRead;
  algorithm::AddressId address;
  std::uint64_t bytes = 0;
  std::optional<value_analysis::ConstantRegion> region;
  std::string unsupported_reason;
  std::uint64_t component_offset = 0;
  std::optional<std::uint64_t> value_byte_offset = std::nullopt;
  friend bool operator==(const SemanticMemoryBinding&,const SemanticMemoryBinding&)=default;
};

struct SemanticMemoryBindingCodec final {
  using Binding=std::array<std::uint64_t,12>;
  using Delta=std::array<std::int64_t,12>;
  static bool split(SemanticMemoryBinding& value,Binding& fields) {
    fields={value.source.instruction.value(),value.occurrence.cta,value.occurrence.warp,
        value.occurrence.lane,value.occurrence.instruction,value.occurrence.occurrence,
        value.value.node,value.value.instance,value.address.value(),
        value.region?value.region->offset:0,value.component_offset,value.value_byte_offset.value_or(0)};
    value.source.instruction=algorithm::InstId{};value.occurrence={};value.value={};value.address=algorithm::AddressId{};
    if(value.region)value.region->offset=0;
    value.component_offset=0;if(value.value_byte_offset)value.value_byte_offset=0;
    return true;
  }
  static void bind(SemanticMemoryBinding& value,const Binding& fields) {
    value.source.instruction=algorithm::InstId(static_cast<std::uint32_t>(fields[0]));
    value.occurrence={static_cast<std::uint32_t>(fields[1]),static_cast<std::uint32_t>(fields[2]),
        static_cast<std::uint32_t>(fields[3]),static_cast<std::uint32_t>(fields[4]),fields[5]};
    value.value={static_cast<std::size_t>(fields[6]),static_cast<std::size_t>(fields[7])};
    value.address=algorithm::AddressId(static_cast<std::uint32_t>(fields[8]));
    if(value.region)value.region->offset=fields[9];
    value.component_offset=fields[10];if(value.value_byte_offset)value.value_byte_offset=fields[11];
  }
  static std::size_t hash(const SemanticMemoryBinding& value) {
    std::size_t h=static_cast<std::size_t>(value.mode);
    algorithm::HashCombine(h,static_cast<std::size_t>(value.source.operand));
    algorithm::HashCombine(h,value.source.effect_index);algorithm::HashCombine(h,value.source.access_index);
    algorithm::HashCombine(h,value.bytes);algorithm::HashCombine(h,value.region.has_value());
    if(value.region) {
      algorithm::HashCombine(h,std::hash<std::string>{}(value.region->allocation));
      algorithm::HashCombine(h,value.region->bytes);
    }
    algorithm::HashCombine(h,std::hash<std::string>{}(value.unsupported_reason));
    algorithm::HashCombine(h,value.value_byte_offset.has_value());return h;
  }
  static std::optional<Delta> difference(const Binding& first,const Binding& next) {
    Delta delta;
    for(std::size_t i=0;i<first.size();++i) {
      const auto distance=first[i]<next[i]?next[i]-first[i]:first[i]-next[i];
      if(distance>INT64_MAX)return {};
      delta[i]=first[i]<=next[i]?static_cast<std::int64_t>(distance):-static_cast<std::int64_t>(distance);
    }
    return delta;
  }
  static std::optional<Binding> translate(Binding base,const Delta& delta,std::size_t index) {
    for(std::size_t i=0;i<base.size();++i) {
      const std::uint64_t maximum=(i<=4 || i==8)?UINT32_MAX:(i==6 || i==7)?SIZE_MAX:UINT64_MAX;
      if(base[i]>maximum)return {};
      if(delta[i]>=0) {
        const auto step=static_cast<std::uint64_t>(delta[i]);
        if(step && index>(maximum-base[i])/step)return {};
        base[i]+=step*index;
      } else {
        const auto step=static_cast<std::uint64_t>(-(delta[i]+1))+1;
        if(index>base[i]/step)return {};
        base[i]-=step*index;
      }
    }
    return base;
  }
};
using SemanticMemoryBindings=algorithm::TemplateRecords<SemanticMemoryBinding,SemanticMemoryBindingCodec>;

using ExecutionCohortId = std::size_t;
// A cohort splits on execution-history differences and never rejoins.
struct ExecutionCohort final {
  ExecutionCohortId id = 0;
  std::optional<ExecutionCohortId> parent;
  std::uint32_t cta = 0;
  std::size_t thread_count = 0;
};

// Members retain their individual operations, attributes, and incident edges.
struct SynchronizationGroup final {
  ExecutionCohortId execution_cohort = 0;
  std::uint32_t instruction_index = 0;
  std::uint64_t occurrence = 0;
  std::size_t operation_kind = 0;
  std::size_t lowering_index = 0;
  std::vector<algorithm::InstId> members;
};

struct SemanticProgram final {
  LoweringResult lowering;
  value_ir::Graph values;
  std::vector<ValueIrDiagnostic> value_diagnostics;
  std::vector<SemanticDiagnostic> diagnostics;
  SemanticMemoryBindings memory_bindings;
  std::vector<SynchronizationGroup> synchronization_groups;
  std::vector<algorithm::InstId> synthetic_synchronization_nodes;
  std::vector<ExecutionCohort> execution_cohorts;
  bool source_complete = false;
  std::vector<MmaValueLayout> numerical_mma_layouts = {};
  bool synchronization_groups_materialized = true;
  bool memory_linking_templates_used = false;

  [[nodiscard]] bool valuesComplete() const noexcept { return value_diagnostics.empty(); }
  [[nodiscard]] bool mappingComplete() const noexcept { return diagnostics.empty(); }
  [[nodiscard]] const algorithm::SourceProgram& commonProgram() const noexcept {
    return lowering.artifact.program;
  }
};

struct SemanticProgramOptions final {
  bool collect_synchronization_groups = true;
  bool compact_memory_linking = true;
  bool link_numerical_memory = true;
  bool defer_address_serialization = false;
  bool capture_branch_scope = false;
};

// Both views consume the same specialized instruction occurrences.
class SemanticProgramBuilder final {
 public:
  SemanticProgramBuilder(const Module&, const Function&, std::function<void()> poll = {},
                         SemanticProgramOptions = {});
  ~SemanticProgramBuilder();
  SemanticProgramBuilder(const SemanticProgramBuilder&) = delete;
  SemanticProgramBuilder& operator=(const SemanticProgramBuilder&) = delete;
  void AddCta(std::uint32_t cta, Coordinates, const ExecutionProgram&);
  [[nodiscard]] const value_ir::Graph& graph() const noexcept;
  [[nodiscard]] const std::vector<ValueIrDiagnostic>& valueDiagnostics() const noexcept;
  // Publish observations from THIS emission before moving/mutating its Value
  // graph. No second PTX execution or numerical assumption is introduced.
  const BranchScopeCapture& BranchScopeForExport() const;
  [[nodiscard]] SemanticProgram Finish(std::function<void(std::string_view)> stage = {});
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct CheckedMemoryActions final {
  struct Statistics {
    std::size_t event_families=0, value_families=0, relations=0;
    std::size_t scheduled_batches=0, logical_events=0, binding_patterns=0;
    std::size_t action_patterns=0, logical_actions=0;
    std::size_t expanded_event_records=0, reachability_queries=0;
    std::size_t domain_refinements=0, singleton_batches=0, largest_batch_members=0;
  } statistics;
  value_analysis::CheckedOrderContract contract;
  value_analysis::MemoryActions actions;
  std::vector<SemanticDiagnostic> diagnostics;
  [[nodiscard]] bool complete() const noexcept { return diagnostics.empty(); }
};

// Trusted reference with synchronous memory only; preserves source dependency order.
[[nodiscard]] CheckedMemoryActions BuildTrustedReferenceMemoryActions(
    const SemanticProgram&, const algorithm::SourceArtifact&);

}  // namespace sentryward::frontend::ptx
