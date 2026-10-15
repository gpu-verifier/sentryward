#pragma once
#include "sentryward/algorithm/base/template_records.h"

#include <cstdint>
#include <array>
#include <optional>
#include <string>
#include <vector>

#include "sentryward/algorithm/base/source_location.h"
#include "sentryward/algorithm/base/strong_id.h"
#include "sentryward/algorithm/source_ir/operation.h"

namespace sentryward::algorithm {

struct ActorInfo final {
  ActorId id;
  CtaId cta;
  WarpId warp;
  std::uint32_t lane;

  friend bool operator==(const ActorInfo&, const ActorInfo&) = default;
};

struct PtxOrigin final {
  // PTX site identity, independent of a potentially shared .loc debug mapping.
  SharedString file;
  std::uint32_t instruction_index;
  // Per-thread visits to this static site, including predicated-off visits.
  std::uint32_t occurrence;
  SharedString opcode;

  friend bool operator==(const PtxOrigin&, const PtxOrigin&) = default;
};

struct SourceInstruction final {
  InstId id;
  ActorId actor;
  SourceLocation location;
  SourceOp op;
  std::optional<PtxOrigin> ptx_origin = std::nullopt;

  friend bool operator==(const SourceInstruction&,
                         const SourceInstruction&) = default;
};


struct SourceRecordCodec {
  struct Binding {
    InstId id; ActorId actor; AddressId address; std::uint32_t occurrence;
    SharedArray<std::uint32_t> operands{};
    friend bool operator==(const Binding&,const Binding&)=default;
  };
  struct Delta {
    std::array<std::int64_t, 4> fields{};
    std::vector<std::int64_t> operands;
    Delta() = default;
    Delta(std::initializer_list<std::int64_t> values) {
      if (values.size() != fields.size()) throw std::invalid_argument("source binding delta width");
      std::copy(values.begin(), values.end(), fields.begin());
    }
    std::int64_t operator[](std::size_t i) const { return fields[i]; }
    friend bool operator==(const Delta&, const Delta&) = default;
  };
  static std::optional<Delta> difference(const Binding& a,const Binding& b) {
    Delta result{std::int64_t(b.id.value())-a.id.value(),std::int64_t(b.actor.value())-a.actor.value(),
                 std::int64_t(b.address.value())-a.address.value(),std::int64_t(b.occurrence)-a.occurrence};
    if (a.operands.size() != b.operands.size()) return {};
    for (std::size_t i = 0; i < a.operands.size(); ++i)
      result.operands.push_back(std::int64_t(b.operands[i]) - a.operands[i]);
    return result;
  }
  static std::optional<Binding> translate(const Binding& base,const Delta& delta,std::size_t index) {
    std::array<std::uint32_t,4> fields{base.id.value(),base.actor.value(),base.address.value(),base.occurrence};
    for(std::size_t i=0;i<fields.size();++i) {
      if(delta[i]>=0) {
        auto step=static_cast<std::uint64_t>(delta[i]);
        if(step && index>(UINT32_MAX-fields[i])/step)return {};
        fields[i]+=static_cast<std::uint32_t>(step*index);
      } else {
        auto step=static_cast<std::uint64_t>(-(delta[i]+1))+1;
        if(index>fields[i]/step)return {};
        fields[i]-=static_cast<std::uint32_t>(step*index);
      }
    }
    auto operands = base.operands;
    if (!delta.operands.empty()) {
      if (delta.operands.size() != operands.size()) return {};
      for (std::size_t i = 0; i < operands.size(); ++i) {
        const auto step = delta.operands[i];
        if (!step || !index) continue;
        const auto magnitude = step > 0 ? std::uint64_t(step) : std::uint64_t(-(step+1))+1;
        if (index > (step > 0 ? UINT32_MAX-operands[i] : operands[i]) / magnitude) return {};
      }
      if (index && std::any_of(delta.operands.begin(), delta.operands.end(),
                              [](auto step) { return step != 0; }))
        operands = operands.transformed([&](std::size_t i, std::uint32_t value) {
          return static_cast<std::uint32_t>(value + delta.operands[i]*std::int64_t(index));
        });
    }
    return Binding{InstId(fields[0]),ActorId(fields[1]),AddressId(fields[2]),fields[3],std::move(operands)};
  }
  template<class Visit>
  static void visitOperands(SourceOp& operation, Visit visit) {
    const auto references = [&](auto& refs) { for (auto& ref : refs) visit(ref.instruction); };
    std::visit([&](auto& op) {
      if constexpr (requires { op.async_id; }) visit(op.async_id);
      if constexpr (requires { op.source; op.destination; }) { visit(op.source); visit(op.destination); }
      if constexpr (requires { op.state; }) visit(op.state);
      if constexpr (requires { op.result; }) { if (op.result) visit(*op.result); }
      if constexpr (requires { op.requirements; }) references(op.requirements);
      if constexpr (requires { op.completion_requires; }) references(op.completion_requires);
      if constexpr (requires { op.effects; }) for (auto& effect : op.effects) {
        references(effect.predecessors);
        effect.accesses = effect.accesses.transformed([&](std::size_t, MemoryEffect access) {
          std::visit([&](auto& endpoint) {
            if constexpr (requires { endpoint.actor; }) visit(endpoint.actor);
            else visit(endpoint);
          }, access.endpoint);
          return access;
        });
      }
    }, operation);
  }
  static bool split(SourceInstruction& inst,Binding& binding){
    auto* load=std::get_if<ImmediateLoad>(&inst.op);
    auto* store=std::get_if<ImmediateStore>(&inst.op);
    binding={inst.id,inst.actor,load?load->address:store?store->address:AddressId{},inst.ptx_origin?inst.ptx_origin->occurrence:0};
    std::vector<std::uint32_t> operands;
    visitOperands(inst.op, [&](auto& id) { operands.push_back(id.value()); id = std::remove_reference_t<decltype(id)>{}; });
    binding.operands = std::move(operands);
    inst.id=InstId{};inst.actor=ActorId{};
    if(load)load->address=AddressId{};else if(store)store->address=AddressId{};
    if(inst.ptx_origin)inst.ptx_origin->occurrence=0;
    return true;
  }
  static void bind(SourceInstruction& inst,const Binding& binding){
    inst.id=binding.id;inst.actor=binding.actor;
    if(inst.ptx_origin)inst.ptx_origin->occurrence=binding.occurrence;
    if(auto* load=std::get_if<ImmediateLoad>(&inst.op))load->address=binding.address;
    else if(auto* store=std::get_if<ImmediateStore>(&inst.op))store->address=binding.address;
    std::size_t i = 0;
    visitOperands(inst.op, [&](auto& id) { id = std::remove_reference_t<decltype(id)>(binding.operands.at(i++)); });
  }
  static SharedString opcode(const SourceInstruction& inst) {
    return inst.ptx_origin ? inst.ptx_origin->opcode : SharedString{};
  }
  static std::size_t effectCount(const SourceInstruction& inst) {
    if (const auto* op = std::get_if<AsyncOp>(&inst.op)) return op->effects.size();
    return std::holds_alternative<ImmediateLoad>(inst.op) ||
           std::holds_alternative<ImmediateStore>(inst.op) ||
           std::holds_alternative<AsyncLoad>(inst.op) ||
           std::holds_alternative<AsyncStore>(inst.op) ? 1 : 0;
  }
  static std::size_t hash(const SourceInstruction& inst){
    auto h=inst.op.index();
    HashCombine(h,std::hash<std::string_view>{}(inst.location.file.view()));
    HashCombine(h,inst.location.line);HashCombine(h,inst.location.column);
    std::visit([&](const auto& op) {
      if constexpr(requires {op.barrier.value();})HashCombine(h,op.barrier.value());
      if constexpr(requires {op.state.value();})HashCombine(h,op.state.value());
      if constexpr(requires {op.sync_id.value();})HashCombine(h,op.sync_id.value());
      if constexpr(requires {op.bytes;})HashCombine(h,op.bytes);
      if constexpr(requires {op.arrivals;})HashCombine(h,op.arrivals);
      if constexpr(requires {op.expected_arrivals;})HashCombine(h,op.expected_arrivals);
      if constexpr(requires {op.parity;})HashCombine(h,op.parity);
      if constexpr(requires {op.lane_mask;})HashCombine(h,op.lane_mask);
      if constexpr(requires {op.memory_ordering;})HashCombine(h,op.memory_ordering);
      if constexpr(requires {op.result.has_value();}) {
        HashCombine(h,op.result.has_value());if(op.result)HashCombine(h,op.result->value());
      }
      if constexpr(requires {op.time_hint_ns.has_value();}) {
        HashCombine(h,op.time_hint_ns.has_value());if(op.time_hint_ns)HashCombine(h,*op.time_hint_ns);
      }
    },inst.op);
    if(inst.ptx_origin){
      HashCombine(h,std::hash<std::string_view>{}(inst.ptx_origin->file.view()));
      HashCombine(h,inst.ptx_origin->instruction_index);
      HashCombine(h,std::hash<std::string_view>{}(inst.ptx_origin->opcode.view()));
    }
    return h;
  }
};
using SourceInstructions=TemplateRecords<SourceInstruction,SourceRecordCodec>;

enum class SourceEdgeKind {
  kProgramOrder,
  kCompleteBeforeIssue,
};

struct SourceEdge final {
  InstId from;
  InstId to;
  SourceEdgeKind kind;

  friend bool operator==(const SourceEdge&, const SourceEdge&) = default;
};

struct SourceEdgeCodec {
  using Binding=std::array<InstId,2>;
  using Delta=std::array<std::int64_t,2>;
  static bool split(SourceEdge& edge,Binding& binding) {
    binding={edge.from,edge.to};edge.from=InstId{};edge.to=InstId{};return true;
  }
  static void bind(SourceEdge& edge,const Binding& binding){edge.from=binding[0];edge.to=binding[1];}
  static std::size_t hash(const SourceEdge& edge){return static_cast<std::size_t>(edge.kind);}
  static std::optional<Delta> difference(const Binding& a,const Binding& b) {
    return Delta{std::int64_t(b[0].value())-a[0].value(),std::int64_t(b[1].value())-a[1].value()};
  }
  static std::optional<Binding> translate(const Binding& base,const Delta& delta,std::size_t i) {
    auto translated=SourceRecordCodec::translate({base[0],ActorId(base[1].value()),AddressId{},0},
        {delta[0],delta[1],0,0},i);
    if(!translated)return {};
    return Binding{translated->id,InstId(translated->actor.value())};
  }
};
using SourceEdges=TemplateRecords<SourceEdge,SourceEdgeCodec>;

struct CollectiveSpec final {
  SyncId sync_id;
  std::vector<ActorId> expected_participants;
  // A mask denotes one local warp; no mask permits a fixed cross-CTA group.
  std::optional<std::uint32_t> lane_mask;

  friend bool operator==(const CollectiveSpec&,
                         const CollectiveSpec&) = default;
};

struct SourceProgram final {
  std::vector<ActorInfo> actors = {};
  SourceInstructions instructions = {};
  SourceEdges edges = {};
  std::vector<CollectiveSpec> collectives = {};
  std::vector<EndpointDependency> dependencies = {};

  friend bool operator==(const SourceProgram&, const SourceProgram&) = default;
};

}  // namespace sentryward::algorithm
