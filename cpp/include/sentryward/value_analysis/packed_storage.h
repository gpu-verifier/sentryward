#pragma once
#include "sentryward/value_ir/graph.h"
#include "sentryward/value_ir/operations.h"
#include "sentryward/value_ir/type_conversion.h"
#include "sentryward/algorithm/diagnostics/compression_audit.h"
#include <algorithm>
#include <bit>
#include <functional>
#include <map>
#include <tuple>
#include <unordered_map>
#include <span>
#include <type_traits>

namespace sentryward::value_analysis {
struct Bf16PromotionField final { value_ir::Value word; std::uint64_t byte_offset; };
// f32 bits are a bf16 field followed by sixteen zero mantissa bits. Recognize
// only the complete 32-bit idioms: low << 16 or word & 0xffff0000, then bitcast
// to f32. A different shift/mask, numerical conversion, or width is not this
// representation. This is field decoding, not dropping arbitrary bit ops.
template<class Known>
std::optional<Bf16PromotionField> bf16PromotionField(
    const value_ir::Graph& graph,value_ir::Value value,Known&& known) {
  namespace v=value_ir;
  value=graph.canonicalInterpretation(value);
  if(graph.type(value)!=v::ValueType::scalar(v::ElementType::floating(32)))return {};
  bool bits=false;
  while(value.conversion()) {
    const auto& c=*value.conversion();
    if(c.spec->kind!=v::TypeConversionKind::Bitcast || graph.type(c.source).bitSize()!=32)return {};
    value=graph.canonicalInterpretation(c.source);bits=true;
  }
  if(!bits || !value.indices().empty() || value.threadSelector() || value.pointer() ||
     !graph.type(value).element().isBitLike())return {};
  const auto* op=dynamic_cast<const v::BinaryOp*>(&graph.definingNode(value).operation());
  if(!op)return {};
  const auto& attr=op->attributes();
  if(attr.rounding!=v::RoundingMode::None || attr.multiply_mode!=v::IntegerMultiplyMode::Low ||
     attr.saturate || attr.flush_to_zero || attr.propagate_nan)return {};
  const auto args=graph.instance(value).operands;
  if(op->opcode()==v::BinaryOpcode::ShiftLeft && known(args[1])==16)
    return Bf16PromotionField{args[0],0};
  if(op->opcode()==v::BinaryOpcode::BitAnd) {
    if(known(args[1])==4294901760LL)return Bf16PromotionField{args[0],2};
    if(known(args[0])==4294901760LL)return Bf16PromotionField{args[1],2};
  }
  return {};
}
// Exact bit-storage provenance, not a floating-point numerical rewrite. Missing
// bits are not invented, and disjoint Add fields are required to exclude carry.
struct PackedStorageField final {
  value_ir::Value value;
  unsigned start, width;
};
class PackedStorageDecoder final {
 public:
  using Value=value_ir::Value;
  using Fields=std::vector<PackedStorageField>;
  using Known=std::function<std::optional<std::int64_t>(Value)>;
  explicit PackedStorageDecoder(const value_ir::Graph& graph,Known known={})
      :graph_(graph),known_(std::move(known)) {}
  // Source-mapping clients only need fields different from the original
  // handle. Do not allocate/cache an opaque singleton for every numeric op,
  // Load or numeric conversion; decode() still retains its leaf semantics.
  std::optional<std::span<const PackedStorageField>> borrowComposed(Value value) {
    value=graph_.resolve(std::move(value));
    const auto& type=graph_.type(value);
    if(!type.isScalar() || type.element().isParameterized() ||
       value.threadSelector() || !value.indices().empty() || value.pointer())return {};
    if(value.conversion()) {
      const auto& c=*value.conversion();const auto& source=graph_.type(c.source);
      if(!source.isScalar() || source.element().isParameterized())return {};
      const bool bits=c.spec->kind==value_ir::TypeConversionKind::Bitcast &&
          source.element().width()==type.element().width();
      const bool widen=c.spec->kind==value_ir::TypeConversionKind::Convert &&
          c.spec->attributes==value_ir::ConversionAttributes{} && source.element().isBitLike() &&
          source.element().kind()!=value_ir::ElementKind::SignedInteger &&
          type.element().kind()==value_ir::ElementKind::UnsignedInteger &&
          source.element().width()<type.element().width();
      if(!bits && !widen)return {};
    } else {
      const auto* binary=dynamic_cast<const value_ir::BinaryOp*>(&graph_.definingNode(value).operation());
      if(!binary || !type.element().isBitLike())return {};
      const auto opcode=binary->opcode();
      if(opcode!=value_ir::BinaryOpcode::Add && opcode!=value_ir::BinaryOpcode::BitOr &&
         opcode!=value_ir::BinaryOpcode::Multiply && opcode!=value_ir::BinaryOpcode::ShiftLeft)return {};
    }
    return borrow(std::move(value));
  }
  // One immutable graph and one immutable interpretation environment per
  // decoder. Cached entries retain their typed handle, including conversions.
  // At most 64 disjoint fields per scalar; expected O(V) cache indexing for
  // V distinct typed handles, plus their direct field operations. Hits borrow
  // immutable fields rather than copying/allocating an array of Value handles.
  std::optional<Fields> decode(Value value,unsigned depth=0) {
    // A selected scalar is one opaque, complete storage field. Its selection
    // is retained in the Value handle, not indexed by the unselected cache
    // key. This lets ordinary widening/packing compose matrix components.
    const auto& type=graph_.type(value);
    if(!value.indices().empty() && !value.conversion() && !value.threadSelector() &&
        !value.pointer() && type.isScalar() && !type.element().isParameterized() &&
        type.element().width() && type.element().width()<=64)
      return Fields{{value,0,type.element().width()}};
    const auto result=borrow(value,depth);
    return result?std::optional<Fields>(Fields(result->begin(),result->end())):std::nullopt;
  }
  std::optional<std::span<const PackedStorageField>> borrow(Value value,unsigned depth=0) {
    if(depth>=32)return {};
    const Key key{value.graphIdentity(),value.nodeId(),value.instanceIndex(),value.resultIndex(),
        reinterpret_cast<std::uintptr_t>(value.conversion().get())};
    // Selection/pointer annotations are deliberately not in the scalar key.
    // They cannot hit a scalar entry; a miss still performs normal validation.
    if(!value.threadSelector() && value.indices().empty() && !value.pointer())
      if(auto found=cache_.find(key);found!=cache_.end()) {
        SENTRYWARD_AUDIT(audit->add(AuditCounter::packed_storage_cache_hits));
        return view(found->second);
      }
    const auto retained=value;
    value=graph_.resolve(value);
    const auto& type=graph_.type(value);
    if(!type.isScalar() || !type.element().width() || type.element().width()>64 ||
       value.threadSelector() || !value.indices().empty() || value.pointer())return {};
    SENTRYWARD_AUDIT(audit->add(AuditCounter::packed_storage_values));
    auto result=inspect(value,depth);
    if(result) {
      std::sort(result->begin(),result->end(),[](const auto& a,const auto& b){return a.start<b.start;});
      unsigned end=0;
      for(const auto& field:*result) {
        if(!field.width || field.start<end || field.start+field.width>type.element().width())return {};
        end=field.start+field.width;
      }
      SENTRYWARD_AUDIT(audit->add(AuditCounter::packed_storage_fields,result->size()));
    }
    auto [found,inserted]=cache_.emplace(key,Entry{retained,std::move(result)});
    return view(found->second);
  }
 private:
  using Key=std::tuple<std::uint64_t,std::size_t,std::size_t,std::size_t,std::uintptr_t>;
  struct KeyHash {
    std::size_t operator()(const Key& key) const {
      std::size_t hash=0;
      std::apply([&](const auto&... parts) {
        ((hash^=std::hash<std::decay_t<decltype(parts)>>{}(parts)+
            std::size_t(0x9e3779b97f4a7c15ULL)+(hash<<6)+(hash>>2)),...);
      },key);
      return hash;
    }
  };
  struct Entry { Value value;std::optional<Fields> fields; };
  const value_ir::Graph& graph_;
  Known known_;
  std::unordered_map<Key,Entry,KeyHash> cache_;
  static std::optional<std::span<const PackedStorageField>> view(const Entry& entry) {
    return entry.fields?std::optional<std::span<const PackedStorageField>>(*entry.fields):std::nullopt;
  }
  std::optional<std::int64_t> known(Value value) const {
    if(known_)return known_(value);
    value=graph_.resolve(value);
    const auto& type=graph_.type(value);
    if(value.conversion() || value.pointer() || value.threadSelector() ||
       !value.indices().empty() || !type.isScalar() || !type.element().isBitLike())return {};
    const auto* c=dynamic_cast<const value_ir::ConstantOp*>(&graph_.definingNode(value).operation());
    if(!c || !type.element().width() || type.element().width()>64)return {};
    try {
      std::size_t used=0;auto n=std::stoll(c->literal(),&used,0);
      if(n<0 || used!=c->literal().size())return {};
      auto width=type.element().width();
      if(type.element().kind()==value_ir::ElementKind::SignedInteger)--width;
      if(width<63 && std::uint64_t(n)>=(std::uint64_t{1}<<width))return {};
      return n;
    } catch(const std::exception&) {return {};}
  }
  std::optional<Fields> inspect(Value value,unsigned depth) {
    const auto& type=graph_.type(value);const auto width=type.element().width();
    if(value.conversion()) {
      const auto& c=*value.conversion();const auto& source=graph_.type(c.source);
      if(!source.isScalar())return {};
      if(c.spec->kind==value_ir::TypeConversionKind::Bitcast && source.element().width()==width)
        return decode(c.source,depth+1);
      if(c.spec->kind==value_ir::TypeConversionKind::Convert &&
         c.spec->attributes==value_ir::ConversionAttributes{} && source.element().isBitLike() &&
         source.element().kind()!=value_ir::ElementKind::SignedInteger &&
         type.element().kind()==value_ir::ElementKind::UnsignedInteger && source.element().width()<width)
        return decode(c.source,depth+1);
      return Fields{{value,0,width}};
    }
    const auto* binary=dynamic_cast<const value_ir::BinaryOp*>(&graph_.definingNode(value).operation());
    if(!binary || !type.element().isBitLike() ||
       binary->attributes().rounding!=value_ir::RoundingMode::None ||
       binary->attributes().multiply_mode!=value_ir::IntegerMultiplyMode::Low ||
       binary->attributes().saturate || binary->attributes().flush_to_zero || binary->attributes().propagate_nan)
      return Fields{{value,0,width}};
    const auto args=graph_.instance(value).operands;
    if(binary->opcode()==value_ir::BinaryOpcode::Add || binary->opcode()==value_ir::BinaryOpcode::BitOr) {
      auto a=decode(args[0],depth+1),b=decode(args[1],depth+1);
      if(!a || !b)return {};
      for(const auto& x:*a)for(const auto& y:*b)
        if(x.start<y.start+y.width && y.start<x.start+x.width)return {};
      a->insert(a->end(),b->begin(),b->end());return a;
    }
    if(binary->opcode()==value_ir::BinaryOpcode::Multiply || binary->opcode()==value_ir::BinaryOpcode::ShiftLeft) {
      auto payload=args[0];auto amount=known(args[1]);
      if(!amount && binary->opcode()==value_ir::BinaryOpcode::Multiply) {amount=known(args[0]);payload=args[1];}
      if(!amount || *amount<0)return {};
      unsigned shift;
      if(binary->opcode()==value_ir::BinaryOpcode::ShiftLeft) {
        if(std::uint64_t(*amount)>=width)return {};
        shift=static_cast<unsigned>(*amount);
      } else {
        if(!std::has_single_bit(std::uint64_t(*amount)))return {};
        shift=std::countr_zero(std::uint64_t(*amount));
      }
      auto result=decode(payload,depth+1);if(!result)return {};
      for(auto& field:*result) {
        if(field.start+field.width+shift>width)return {};
        field.start+=shift;
      }
      return result;
    }
    return Fields{{value,0,width}};
  }
};
} // namespace sentryward::value_analysis
