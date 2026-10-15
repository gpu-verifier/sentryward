#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "sentryward/value_ir/graph.h"
#include "sentryward/value_analysis/memory_actions.h"

namespace sentryward::value_analysis {

struct AllocationInfo final {
  std::string identity;
  bool has_initial_contents = false;
  bool output = false;
};

enum class MemoryOrderBasis { Verified, TrustedReference };

// The caller supplies a complete execution order and its correctness basis.
struct CheckedOrderContract final {
  bool synchronization_checked = false;
  bool race_free = false;
  bool lifetimes_checked = false;
  std::vector<AllocationInfo> allocations;
  MemoryOrderBasis basis = MemoryOrderBasis::Verified;
  [[nodiscard]] bool permitsElimination() const noexcept {
    return basis == MemoryOrderBasis::TrustedReference ||
        (synchronization_checked && race_free && lifetimes_checked);
  }
};

struct InitialMemorySymbol final {
  std::size_t id;
  ConstantRegion region;
  value_ir::ValueType type;
};

struct ReinterpretedRead final {
  // Extract type.bitSize() bits at byte_offset in PTX little-endian order.
  std::variant<value_ir::Value, InitialMemorySymbol> source;
  value_ir::ValueType type;
  std::uint64_t byte_offset = 0;
};
struct AssembledRead final {
  value_ir::ValueType type;
  // Logical value byte order; the first piece supplies the low bits.
  std::vector<ReinterpretedRead> pieces;
};
using ReadReplacement = std::variant<value_ir::Value, InitialMemorySymbol, ReinterpretedRead, AssembledRead>;

// A block copies bytes, repeated over axes; axis zero is innermost.
struct ViewAxis final {
  std::uint64_t count, source_stride, result_stride;
  friend bool operator==(const ViewAxis&, const ViewAxis&) = default;
};
using ViewSource = std::variant<value_ir::Value, InitialMemorySymbol>;
struct IndexedValueBinding final {
  value_ir::Value leaf;
  std::int64_t base = 0, stride = 0;
  std::optional<InitialMemorySymbol> initial;
  bool fixed = false;
  // A slice of one already-computed value (including its exact occurrence).
  // The index map yields a byte offset, not a scalar value. The enclosing
  // IndexedSource's bindings must not capture this producer's own leaves.
  std::optional<value_ir::Value> producer = {};
  // True only after structural validation of all finite members: the producer
  // is a scalar expression template whose leaves use this same binding scope.
  // False keeps an exact computed occurrence opaque (e.g. one MMA result).
  bool producer_uses_bindings = false;
  // A finite, validated two-axis index map. A zero period is the affine case;
  // otherwise use base + stride*(i % period) + outer_stride*(i / period).
  // These coefficients never encode a table of concrete member sources.
  std::uint64_t period = 0;
  std::int64_t outer_stride = 0;
  // Decode a narrower floating field before interpreting this scalar use.
  // Example: a bf16 field promoted to f32 by shifting its bits left by 16.
  std::optional<value_ir::ElementType> numerical_field = {};
  // Bounded additional radix carries: sum(coefficient * floor(i/divisor)).
  // Each extension is zero throughout the already-validated prefix. The
  // matcher validates subsequent records; this is not extrapolation by samples.
  struct QuotientTerm {
    std::uint64_t divisor;
    std::int64_t coefficient;
    bool operator==(const QuotientTerm&) const = default;
  };
  std::vector<QuotientTerm> quotient_terms = {};
  // Scalar literal map only: coefficients denote the exact bit index, and
  // the value is 1 << index. Never applies to an address or producer offset.
  bool power_of_two = false;
};
// Evaluate the coordinate map (bit index, not decoded value, for power_of_two).
std::optional<std::int64_t> indexedBindingValue(const IndexedValueBinding&,std::uint64_t);
// An immutable use DAG. Two uses of one Value may have different maps; equal
// uses share this node. Children follow operand positions, not variable names.
// Refinement replaces only changed ancestors, never expands a shared subtree.
struct IndexedSourceUse final {
  enum class Kind { Exact, Binding, Operands, Conversion, Selected, Forward,
                    NonzeroSelect, OneHotLiteral, BitTestSelect, MemberTemplate, CoordinateMap,
                    Component };
  // Component children are [matrix use, index use for each axis]. All use
  // maps share the reader coordinate, but each has its own producer/member
  // mapping. In particular the matrix may be fixed while its indices vary.
  // Payload projection is composed with these indices, never with a sampled
  // representative's component. Temporal producer identity stays in prototype.
  // CoordinateMap composes base + stride*i with one existing use DAG. It
  // changes a use coordinate, never a producer occurrence or memory binding.
  // MemberTemplate is an existing Value template, not an inferred common
  // expression. binding.base/stride map the enclosing index to its instance.
  // Children are obtained from Graph's existing operand bindings on demand.
  // OneHotLiteral is an exact literal in a bit-mask operand position; its
  // refinement may bind the bit index. It has the same value as prototype.
  // NonzeroSelect children are [integer test value, nonzero arm, zero arm].
  // It canonicalizes eq/ne-to-zero with exchanged arms without choosing one.
  value_ir::Value prototype;
  Kind kind = Kind::Exact;
  // Binding nodes use this payload; BitTestSelect reuses its coordinate map
  // for the tested bit index (not a substitution for the result Value).
  // BitTestSelect children are [mask, bit-set arm, bit-clear arm].
  // Do not reserve the full producer,
  // type and initial-memory record in every pure-operation use node.
  std::shared_ptr<const IndexedValueBinding> binding;
  std::vector<std::shared_ptr<const IndexedSourceUse>> children;
};
// One source expression with affine/mixed-radix use bindings; no source list.
struct IndexedSource final {
  value_ir::Value prototype;
  std::uint64_t count = 0;
  // Evaluation follows this DAG. A printable binding summary is generated
  // on demand, never copied and retained alongside every source expression.
  std::shared_ptr<const IndexedSourceUse> root;
  // An exactly validated finite grid mask, not a table of source fragments.
  // Keep the source when (row-column >= diagonal) == lower; otherwise zero.
  // row=i/columns and column=i%columns. Only constant zero is elided.
  struct ZeroMask final {
    std::uint64_t columns;
    std::int64_t diagonal;
    bool lower = true;
  };
  std::optional<ZeroMask> zero_mask = {};
};
std::vector<IndexedValueBinding> indexedSourceBindings(const IndexedSource&);
struct ViewBlock final {
  std::size_t source;
  std::uint64_t source_offset, result_offset, bytes;
  std::vector<ViewAxis> axes;
  friend bool operator==(const ViewBlock&,const ViewBlock&) = default;
};
enum class MemoryViewKind { Element, Slice, Layout, Segmented };
struct MemoryView final {
  value_ir::ValueType type;
  MemoryViewKind kind;
  std::vector<ViewSource> sources;
  std::vector<ViewBlock> blocks;
  // Row-major coordinates for an element of a statically shaped producer.
  std::vector<std::uint64_t> indices;
  std::shared_ptr<const IndexedSource> indexed_source = {};
};
struct ReadRegion final {
  std::uint64_t begin, bytes;
  MemoryView view;
};
// Disjoint result domains with independent source rules.
struct PartitionedRead final {
  value_ir::ValueType type;
  std::vector<ReadRegion> regions;
  // These domains retain existing Value templates. Consumers must not undo
  // the partition by trying to match their mathematical expression trees.
  bool template_domains = false;
};
using NormalizedRead = std::variant<value_ir::Value, InitialMemorySymbol, MemoryView, PartitionedRead>;
using NormalizedReadReference = std::variant<const value_ir::Value*, const InitialMemorySymbol*, const MemoryView*, const PartitionedRead*>;

struct OutputWrite final {
  ConstantRegion region;
  value_ir::Value value;
  MemoryInstance writer;
  std::uint64_t value_byte_offset = 0;
  // Numerical interpretation recovered from a byte-storage view. Keep the
  // range and its source mapping intact instead of exporting scalar pieces.
  std::optional<value_ir::ElementType> element_type = {};
};

// A numerical output domain planned from existing finite read-substitution
// pieces. This is a query view, not a new write family or execution order.
// Original OutputWrites remain available for provenance and diagnostics.
struct OutputViewDomain final {
  // offset is the physical base; bytes is the number of logical payload
  // bytes. shape/strides describe any holes, never a covering memory range.
  ConstantRegion region;
  value_ir::ElementType element_type;
  MemoryView view;
  std::vector<std::uint64_t> shape, strides_bytes;
  std::uint64_t source_byte_offset = 0;
};
struct OutputViewGroup final {
  std::size_t end;  // Upper bound on member records, not a contiguous range.
  std::vector<OutputViewDomain> domains;
};

struct InitialReadFamily final {
  MemoryInstance first;
  std::size_t count;
  std::int64_t byte_stride;
  InitialMemorySymbol initial;
  std::int64_t symbol_stride = 1;
};

// A constant source mapping over a contiguous domain of spatial instances.
struct SharedReadFamily final {
  MemoryInstance first;
  std::size_t count;
  MemoryInstance prototype;
};

// A read-from certificate, produced while ordered reads are resolved. Every
// member has been checked against the current memory state, not inferred from
// endpoints. One producer node (temporal occurrence) supplies a spatial domain:
// source byte offset = source_offset + byte_stride * (instance-first.instance).
// It never merges different execution occurrences or hides a source table.
struct ReadSlicePattern final {
  MemoryInstance first;
  std::size_t count;
  value_ir::Value source;
  std::uint64_t bytes, source_offset;
  std::int64_t byte_stride = 0;
  std::uint64_t result_offset = 0;
  std::int64_t source_instance_stride = 0;
  // An exact Value can carry interpretation annotations. Only plain Values
  // may vary their instance here; never silently rebind annotation operands.
  [[nodiscard]] bool sameSourceTemplate(const value_ir::Value& other) const {
    if(source==other)return true;
    const auto plain=[](const value_ir::Value& v) {
      return v.indices().empty() && !v.conversion() && !v.threadSelector() && !v.pointer();
    };
    return plain(source) && plain(other) && source.graphIdentity()==other.graphIdentity() &&
        source.nodeId()==other.nodeId() && source.resultIndex()==other.resultIndex();
  }
  [[nodiscard]] bool mapsTo(std::size_t member,const value_ir::Value& value,
      std::uint64_t offset) const {
    if(member<first.instance || member-first.instance>=count || !sameSourceTemplate(value))return false;
    const auto i=static_cast<__int128>(member-first.instance);
    return static_cast<__int128>(source.instanceIndex())+source_instance_stride*i==value.instanceIndex() &&
        static_cast<__int128>(source_offset)+byte_stride*i==offset;
  }
};

struct OutputWriteFamily final {
  MemoryInstance first;
  std::size_t count;
  ConstantRegion region;
  std::int64_t byte_stride;
  [[nodiscard]] OutputWrite at(const value_ir::Graph& graph,std::size_t index) const {
    if(index>=count)throw std::out_of_range("output family instance");
    auto location=region;
    const auto offset=static_cast<__int128>(region.offset)+static_cast<__int128>(byte_stride)*index;
    if(offset<0 || offset>UINT64_MAX || first.instance>SIZE_MAX-index)
      throw std::logic_error("invalid output family");
    location.offset=static_cast<std::uint64_t>(offset);
    const MemoryInstance writer{first.node,first.instance+index};
    return {location,graph.node(writer.node).instances().at(writer.instance).operands.at(1),writer,0};
  }
};

struct MemoryEliminationDiagnostic final {
  std::string code;
  std::string message;
  std::optional<MemoryInstance> operation;
};

struct MemoryEliminationResult final {
  // Original Load leaves, retained only after the complete action census has
  // excluded writes to their allocation. Addresses and occurrences stay in IR.
  std::map<MemoryInstance, ConstantRegion> symbolic_initial_reads;
  bool complete = false;
  std::map<MemoryInstance, ReadReplacement> reads;
  std::map<MemoryInstance, SharedReadFamily> shared_read_families;
  // Result-fragment domains, then spatial domains. Native fragments are
  // certified at resolution too, not reconstructed from AssembledRead later.
  std::map<std::pair<std::size_t,std::uint64_t>,
      std::map<std::size_t,ReadSlicePattern>> read_slice_patterns;
  std::size_t shared_read_instances = 0;
  std::vector<OutputWrite> outputs;
  std::vector<OutputWriteFamily> output_families;
  std::vector<InitialMemorySymbol> initial_memory;
  std::map<MemoryInstance,InitialReadFamily> initial_read_families;
  std::size_t processed_families = 0;
  std::optional<MemoryEliminationDiagnostic> diagnostic;
  std::size_t processed_actions = 0;
  std::size_t duplicate_write_value_checks = 0;
  bool views_normalized = false;
  std::map<MemoryInstance, NormalizedRead> normalized_reads;
  std::map<MemoryInstance,MemoryInstance> normalized_read_aliases;
  std::map<std::size_t,OutputViewGroup> normalized_output_groups;
  // Membership in the existing finite output record list. This does not
  // enumerate members of a compressed family or encode source expressions.
  std::vector<bool> normalized_output_members;
  [[nodiscard]] const ReadSlicePattern* readSlicePattern(MemoryInstance key,
      std::uint64_t offset=0,std::uint64_t width=0) const {
    auto it=read_slice_patterns.upper_bound({key.node,offset});
    if(it==read_slice_patterns.begin())return nullptr;
    const auto& [domain,patterns]=*std::prev(it);
    if(domain.first!=key.node)return nullptr;
    auto member=patterns.upper_bound(key.instance);
    if(member==patterns.begin())return nullptr;
    const auto& pattern=std::prev(member)->second;
    return key.instance>=pattern.first.instance && key.instance-pattern.first.instance<pattern.count &&
        offset>=pattern.result_offset && offset-pattern.result_offset<=pattern.bytes &&
        width<=pattern.bytes-(offset-pattern.result_offset) ? &pattern : nullptr;
  }
  // Online proof construction, O(log stored domains), with O(1) local work.
  // Call only when the read has been resolved; unsupported sources simply
  // have no certificate. The raw replacement remains available for provenance.
  void recordInitialRead(MemoryInstance key,const InitialMemorySymbol& input) {
    const auto* initial=&input;
    auto next=initial_read_families.lower_bound(key);
    if(next!=initial_read_families.begin()) {
      auto previous=std::prev(next);auto& p=previous->second;
      if(p.first.node==key.node && key.instance>=p.first.instance &&
          key.instance-p.first.instance==p.count && p.initial.type==initial->type &&
          p.initial.region.allocation==initial->region.allocation &&
          p.initial.region.bytes==initial->region.bytes) {
        const auto bytes=static_cast<__int128>(initial->region.offset)-p.initial.region.offset;
        const auto ids=static_cast<__int128>(initial->id)-p.initial.id;
        if((p.count==1 && bytes>=INT64_MIN && bytes<=INT64_MAX && ids>=INT64_MIN && ids<=INT64_MAX) ||
            (bytes==static_cast<__int128>(p.byte_stride)*p.count &&
             ids==static_cast<__int128>(p.symbol_stride)*p.count)) {
          if(p.count==1){p.byte_stride=static_cast<std::int64_t>(bytes);p.symbol_stride=static_cast<std::int64_t>(ids);}
          ++p.count;return;
        }
      }
    }
    initial_read_families.emplace(key,InitialReadFamily{key,1,0,*initial,0});return;
  }
  void recordReadSlice(MemoryInstance key,const ReadReplacement& replacement,
      std::uint64_t result_offset=0) {
    if(const auto* assembly=std::get_if<AssembledRead>(&replacement)) {
      // These are already resolved read-from fragments, not member samples.
      // Retain each exact result-byte domain, including holes occupied by
      // initial-memory sources. Never infer relationships between writers.
      auto cursor=result_offset;
      for(const auto& piece:assembly->pieces) {
        const auto bits=piece.type.bitSize();
        if(bits%8 || bits/8>UINT64_MAX-cursor)
          throw std::length_error("read slice fragment extent overflow");
        recordReadSlice(key,ReadReplacement{piece},cursor);
        cursor+=bits/8;
      }
      return;
    }
    const value_ir::Value* source=nullptr;
    std::uint64_t offset=0,bytes=0;
    if(const auto* value=std::get_if<value_ir::Value>(&replacement))source=value;
    else if(const auto* slice=std::get_if<ReinterpretedRead>(&replacement)) {
      source=std::get_if<value_ir::Value>(&slice->source);
      offset=slice->byte_offset;bytes=slice->type.bitSize()/8;
    }
    if(!source || !bytes)return; // Whole-value reads need a typed slice first.
    auto& patterns=read_slice_patterns[{key.node,result_offset}];
    auto next=patterns.lower_bound(key.instance);
    const auto join=[&](auto current) {
      auto& previous=current->second;
      while(next!=patterns.end() && next->first>=previous.first.instance &&
            next->first-previous.first.instance==previous.count &&
            previous.sameSourceTemplate(next->second.source) && next->second.bytes==previous.bytes &&
            (next->second.count==1 || (next->second.byte_stride==previous.byte_stride &&
                next->second.source_instance_stride==previous.source_instance_stride)) &&
            static_cast<__int128>(next->second.source.instanceIndex())==
                static_cast<__int128>(previous.source.instanceIndex())+
                static_cast<__int128>(previous.source_instance_stride)*previous.count &&
            static_cast<__int128>(next->second.source_offset)==static_cast<__int128>(previous.source_offset)+
                static_cast<__int128>(previous.byte_stride)*previous.count) {
        if(next->second.count>SIZE_MAX-previous.count)throw std::length_error("read slice domain overflow");
        previous.count+=next->second.count;next=patterns.erase(next);
      }
    };
    if(next!=patterns.begin()) {
      auto& previous=std::prev(next)->second;
      if(previous.first.node==key.node && previous.sameSourceTemplate(*source) &&
         previous.bytes==bytes && key.instance>=previous.first.instance &&
         key.instance-previous.first.instance==previous.count) {
        const auto delta=static_cast<__int128>(offset)-previous.source_offset;
        const auto instance_delta=static_cast<__int128>(source->instanceIndex())-previous.source.instanceIndex();
        if(previous.count==1 && delta>=INT64_MIN && delta<=INT64_MAX &&
            instance_delta>=INT64_MIN && instance_delta<=INT64_MAX) {
          previous.byte_stride=static_cast<std::int64_t>(delta);
          previous.source_instance_stride=static_cast<std::int64_t>(instance_delta);
          ++previous.count;join(std::prev(next));return;
        }
        if(delta==static_cast<__int128>(previous.byte_stride)*previous.count &&
            instance_delta==static_cast<__int128>(previous.source_instance_stride)*previous.count) {
          ++previous.count;join(std::prev(next));return;
        }
      }
    }
    auto [inserted,added]=patterns.emplace(key.instance,ReadSlicePattern{key,1,*source,bytes,offset,0,result_offset});
    if(!added)
      throw std::logic_error("read-from slice certificate recorded twice");
    // A late earlier member can extend an already certified affine domain.
    // Every removed domain is merged once, so total joining is amortized linear.
    if(next!=patterns.end() && next->first-key.instance==1) {
      const auto delta=static_cast<__int128>(next->second.source_offset)-offset;
      const auto instance_delta=static_cast<__int128>(next->second.source.instanceIndex())-source->instanceIndex();
      if(delta>=INT64_MIN && delta<=INT64_MAX && instance_delta>=INT64_MIN && instance_delta<=INT64_MAX) {
        inserted->second.byte_stride=static_cast<std::int64_t>(delta);
        inserted->second.source_instance_stride=static_cast<std::int64_t>(instance_delta);join(inserted);
      }
    }
  }
  [[nodiscard]] MemoryInstance canonicalRead(MemoryInstance key) const {
    auto it=shared_read_families.upper_bound(key);
    if(it!=shared_read_families.begin()) {
      const auto& family=std::prev(it)->second;
      if(key.node==family.first.node && key.instance>=family.first.instance &&
         key.instance-family.first.instance<family.count)return family.prototype;
    }
    return key;
  }
  [[nodiscard]] const ReadReplacement* readReplacement(MemoryInstance key) const {
    const auto it=reads.find(canonicalRead(key));
    return it==reads.end()?nullptr:&it->second;
  }
  void shareRead(MemoryInstance key,MemoryInstance prototype) {
    prototype=canonicalRead(prototype);
    if(key==prototype || key.node!=prototype.node || !reads.contains(prototype) ||
       reads.contains(key) || canonicalRead(key)!=key)
      throw std::invalid_argument("invalid shared read mapping");
    auto next=shared_read_families.lower_bound(key);
    auto current=shared_read_families.end();
    if(next!=shared_read_families.begin()) {
      auto previous=std::prev(next);
      auto& family=previous->second;
      if(family.first.node==key.node && family.prototype==prototype &&
         key.instance-family.first.instance==family.count) {
        ++family.count;current=previous;
      }
    }
    if(current==shared_read_families.end())
      current=shared_read_families.emplace(key,SharedReadFamily{key,1,prototype}).first;
    auto& family=current->second;
    if(next!=shared_read_families.end() && next->first.node==key.node &&
       next->second.prototype==prototype &&
       next->first.instance-family.first.instance==family.count) {
      family.count+=next->second.count;shared_read_families.erase(next);
    }
    ++shared_read_instances;
  }
  [[nodiscard]] std::size_t outputCount() const {
    auto count=outputs.size();
    for(const auto& family:output_families) {
      if(family.count>SIZE_MAX-count)throw std::length_error("output count overflow");
      count+=family.count;
    }
    return count;
  }
  template<class Visitor> void forEachOutput(const value_ir::Graph& graph,Visitor&& visitor) const {
    for(const auto& output:outputs)visitor(output);
    for(const auto& family:output_families)
      for(std::size_t i=0;i<family.count;++i)visitor(family.at(graph,i));
  }
  [[nodiscard]] std::optional<InitialMemorySymbol> initialRead(MemoryInstance key) const {
    auto it=initial_read_families.upper_bound(key);
    if(it==initial_read_families.begin())return {};
    const auto& family=std::prev(it)->second;
    if(key.node!=family.first.node || key.instance<family.first.instance ||
       key.instance-family.first.instance>=family.count)return {};
    const auto index=key.instance-family.first.instance;
    auto value=family.initial;
    const auto offset=static_cast<__int128>(value.region.offset)+static_cast<__int128>(family.byte_stride)*index;
    if(offset<0 || offset>UINT64_MAX)throw std::logic_error("invalid initial-read family");
    const auto id=static_cast<__int128>(value.id)+static_cast<__int128>(family.symbol_stride)*index;
    if(id<0 || id>UINT64_MAX)throw std::logic_error("invalid initial-read symbol family");
    value.region.offset=static_cast<std::uint64_t>(offset);value.id=static_cast<std::uint64_t>(id);
    return value;
  }
  // References remain valid until the corresponding read maps are modified.
  [[nodiscard]] std::optional<NormalizedReadReference> normalizedRead(MemoryInstance key) const {
    if (!views_normalized) return {};
    key=canonicalRead(key);
    if(const auto alias=normalized_read_aliases.find(key);alias!=normalized_read_aliases.end())
      key=alias->second;
    if (const auto found=normalized_reads.find(key); found!=normalized_reads.end())
      return std::visit([](const auto& value)->NormalizedReadReference { return &value; },found->second);
    if (const auto found=reads.find(key); found!=reads.end())
      if (const auto* initial=std::get_if<InitialMemorySymbol>(&found->second))
        return NormalizedReadReference{initial};
    return {};
  }
};

[[nodiscard]] MemoryEliminationResult eliminateMemory(
    const value_ir::Graph& graph, MemoryActionRange actions,
    const CheckedOrderContract& contract);

}  // namespace sentryward::value_analysis
