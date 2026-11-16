#pragma once
#include "scalar_fold_lowering_impl.hpp"

namespace sentryward::value_analysis {
namespace {

// An address built as cvta.to.global(root parameter) + terms by Add/Subtract: the terms,
// which are exactly address - cvta(root). The conversion never has to be evaluated or
// assumed to be the identity. `use` is the address's use DAG in a template projection
// (its operand children are followed; any other use kind is not decomposed).
// nullopt for any other form. O(depth), depth <= kCvtaTermDepth.
inline auto Lower::cvtaRelative(v::Value address,const v::Value& root,
    const std::shared_ptr<const IndexedSourceUse>& use,unsigned depth) -> std::optional<s::Ref> {
  using UseKind=IndexedSourceUse::Kind;
  if(depth>kCvtaTermDepth || !address.indices().empty() || address.threadSelector())return std::nullopt;
  if(use && use->prototype!=input.canonicalInterpretation(address))return std::nullopt;
  // The use of an operand or conversion source: the stored child, or (member template)
  // its own member mapping, bound exactly as the template projection binds a uniform
  // template's operands. Null without a use; nullopt when no mapping is certified.
  const auto sourceUse=[&](const v::Value& source,std::size_t k)->std::optional<std::shared_ptr<const IndexedSourceUse>> {
    if(!use)return std::shared_ptr<const IndexedSourceUse>{};
    if(use->kind==UseKind::Operands || use->kind==UseKind::Conversion) {
      if(k>=use->children.size())return std::nullopt;
      return use->children[k];
    }
    if(use->kind!=UseKind::MemberTemplate || !use->binding)return std::nullopt;
    auto bound=bindSourceUse(input,input.canonicalInterpretation(source),address,use->binding->base,use->binding->stride);
    if(!bound)return std::nullopt;
    return bound;
  };
  // A 64-bit scalar integer bitcast (s64/u64/b64 views of one address) changes no bits.
  if(const auto& conversion=address.conversion()) {
    if(conversion->spec->kind!=v::TypeConversionKind::Bitcast || !input.type(address).isScalar() ||
       input.type(address).bitSize()!=64 || input.type(conversion->source).bitSize()!=64)return std::nullopt;
    if(use && use->kind!=UseKind::Conversion && use->kind!=UseKind::MemberTemplate)return std::nullopt;
    const auto inner=sourceUse(conversion->source,0);
    if(!inner)return std::nullopt;
    return cvtaRelative(conversion->source,root,*inner,depth+1);
  }
  address=input.resolve(address);
  if(use && use->kind!=UseKind::Operands &&
     !(use->kind==UseKind::MemberTemplate && input.definingNode(address).hasUniformThreadTemplate()))return std::nullopt;
  const auto& op=input.definingNode(address).operation();
  const auto operands=input.instanceOperands(address);
  const auto bits=[&](v::Value x) {
    while(x.conversion() && x.conversion()->spec->kind==v::TypeConversionKind::Bitcast &&
          x.indices().empty() && !x.threadSelector())x=x.conversion()->source;
    return input.resolve(x);
  };
  if(const auto* convert=dynamic_cast<const v::PtxScalarOp*>(&op);convert &&
     convert->opcode()==v::PtxScalarOpcode::AddressSpaceConvert && !convert->attributes().carry_in &&
     !convert->attributes().carry_out && operands.size()==1 && samePointer(bits(operands[0]),bits(root)) &&
     (convert->attributes().modifiers==std::vector<std::string>{"to","global","u64"} ||
      convert->attributes().modifiers==std::vector<std::string>{"global","u64"}))
    return result.graph.constant("u64","0");
  const auto* binary=dynamic_cast<const v::BinaryOp*>(&op);
  if(!binary || operands.size()!=2 || input.type(address).bitSize()!=64 ||
     (binary->opcode()!=v::BinaryOpcode::Add && binary->opcode()!=v::BinaryOpcode::Subtract))return std::nullopt;
  const auto first=sourceUse(operands[0],0),second=sourceUse(operands[1],1);
  if(!first || !second)return std::nullopt;
  const auto term=[&](std::size_t k){return inUse(k?*second:*first,[&]{return scalar(operands[k]);});};
  if(auto left=cvtaRelative(operands[0],root,*first,depth+1))
    return emitOperation(binary->opcode()==v::BinaryOpcode::Add?"Add":"Subtract","u64",{*left,term(1)});
  if(binary->opcode()==v::BinaryOpcode::Add)
    if(auto right=cvtaRelative(operands[1],root,*second,depth+1))return emitOperation("Add","u64",{term(0),*right});
  return std::nullopt;
}

// One pointer value: the same Value, or global pointer parameters of one allocation (pointer_aliases).
inline auto Lower::samePointer(const v::Value& a,const v::Value& b) const -> bool {
  if(a==b)return true;
  const auto symbol=[&](const v::Value& x)->const std::string* {
    const auto* p=dynamic_cast<const v::InputParameterOp*>(&input.definingNode(x).operation());
    if(!p || !p->pointer())return nullptr;
    const auto alias=pointer_aliases.find(p->symbol());
    return alias!=pointer_aliases.end()?&alias->second:&p->symbol();
  };
  const auto* x=symbol(a);const auto* y=symbol(b);
  return x && y && *x==*y;
}

// The positioned read run covering members [first, last] of a node, if any.
inline auto Lower::positionRun(std::size_t node,std::size_t first,std::size_t last) const -> const std::pair<const MemoryInstance,PositionRun>* {
  auto found=position_runs.upper_bound({node,first});
  if(found==position_runs.begin())return nullptr;
  --found;
  if(found->first.node!=node || first<found->first.instance || last-found->first.instance>=found->second.count)return nullptr;
  return &*found;
}

// A positioned initial read at its exact position (memory elimination): the cell at
// `constant` + sum(coefficient x atom), each atom its atomCell (as placements are). No
// address value graph, hence no unit parameter: lifting moves the constant by
// δ(allocation) and each atom by δ(its object), the members' own positions. `constant`
// is an Offset over the projection's binder when the read is a template member.
inline auto Lower::positionedCell(const SymbolicPosition& position,const Offset& constant,const std::string& type) -> s::Ref {
  auto offset=offsetValue(constant);
  for(const auto& term:position.base) {
    const auto atom=result.graph.convert(atomCell(term),"s64","Convert");
    offset=emitOperation("Add","u64",{offset,emitOperation("Multiply","s64",
        {atom,result.graph.constant("s64",std::to_string(term.coefficient))})});
  }
  return repAbsoluteCell(position.constant.allocation,offset,type);
}

// A member of a TMA region fragment family (IndexedValueBinding::region_fragment) at its place
// `place` (the family's coordinate map), `offset` bytes into it: the fragment read of read() below
// (the cell at its allocation-relative address, zero where the region's guard is false), with the
// place symbolic. The origin and the guard are the leaf's, which every member shares (the
// template signature). Exact: the fragment's address is u64(origin + place) for the s64 origin and
// its literal place, and u64(origin + place) = u64(origin) + place modulo 2^64. O(1) plus the
// projections of the origin and the guard, memoized across members and places.
// The value of a mapped integer coordinate in `type` (a grid family's literal or induction
// step). A coordinate that can be negative is its non-negative shift plus the (negative) lower
// bound, added in `type`; Bit kinds take the unsigned value of their width. Exact within the
// type's range, which is checked.
inline auto Lower::integerValue(const Offset& position,const v::ElementType& type) -> s::Ref {
  const auto [lo,hi]=bounds(position);
  const auto width=type.width();
  if(!width || width>64 || hi<lo)fail("scalar_fold_invalid_index","mapped integer coordinate has no integer type");
  const bool is_signed=type.kind()==v::ElementKind::SignedInteger;
  const auto top=width>=63?INT64_MAX:(std::int64_t(1)<<(is_signed?width-1:width))-1;
  if(hi>top || (is_signed?(width<64 && lo<-(std::int64_t(1)<<(width-1))):lo<0))
    fail("scalar_fold_invalid_index","mapped integer coordinate exceeds its type");
  const bool bits=type.kind()==v::ElementKind::Bit;
  const auto carrier=bits?"u"+std::to_string(width):type.toString();
  if(lo>=0) {
    auto value=result.graph.convert(offsetValue(position),carrier,"Convert");
    return bits?result.graph.convert(value,type.toString(),"Bitcast"):value;
  }
  auto shifted=result.graph.convert(offsetValue(plus(position,Offset(-lo))),carrier,"Convert");
  return emitOperation("Add",type.toString(),{shifted,result.graph.constant(type.toString(),std::to_string(lo))},
      attributes(v::BinaryOp(v::BinaryOpcode::Add,v::ValueType::scalar(type))));
}

// The literal origin of an exact region fragment Load (MemoryAttributes::region_exact): the
// region's first byte in its allocation, which the frontend took from the slot's exact value and
// class admission compares up to δ(allocation). Nullopt for any other fragment. O(1).
inline auto Lower::exactRegionOrigin(const v::Value& leaf,const RegionFragmentLoad& fragment) const -> std::optional<std::int64_t> {
  const auto* load=dynamic_cast<const v::LoadOp*>(&input.definingNode(leaf).operation());
  if(!load || !load->attributes().region_exact || fragment.guard)return std::nullopt;
  const auto origin=input.resolve(fragment.origin);
  if(origin.pointer() || origin.conversion() || !origin.indices().empty() || origin.threadSelector())return std::nullopt;
  const auto* literal=dynamic_cast<const v::ConstantOp*>(&input.definingNode(origin).operation());
  if(!literal || literal->pointer())return std::nullopt;
  const auto value=exactInteger<std::int64_t>(literal->literal());
  if(!value || *value<0)return std::nullopt;
  return value;
}

// With origin_use / guard_use (a grid family whose members are fragments of different regions,
// e.g. gathered rows): the origin and the guard are projected under those uses at the current
// member index, so they are the member's own. `leaf` is then the first member's fragment Load.
inline auto Lower::regionFragmentCell(const v::Value& leaf,const Offset& place,const Offset& offset,
    const v::ElementType& type,const v::ElementType* numerical,
    std::shared_ptr<const IndexedSourceUse> origin_use,
    std::shared_ptr<const IndexedSourceUse> guard_use) -> s::Ref {
  const auto fragment=regionFragmentLoad(input,memory,leaf);
  if(!fragment)fail("scalar_fold_prerequisite","region fragment family lost its fragment form");
  // As read(): the cell moves by δ(allocation) at instantiation. An exact region (exactRegionOrigin)
  // is the cell at its literal first byte, and class admission compared that byte up to δ; any
  // other region's origin is rebuilt per member, so its δ must be 0, which admission keeps.
  const auto exact=exactRegionOrigin(leaf,*fragment);
  if(exact && !origin_use)
    return exactRegionCell(fragment->allocation,*exact,plus(place,offset),type,numerical);
  if(options.unit_lifting && !exact && !anchoredRegion(fragment->allocation))
    liftRefuse("TMA region fragment read at its allocation-relative address");
  static const std::string step="Convert "+attributes(v::ConvertOp(
      v::ValueType::scalar(v::ElementType::unsignedInteger(64)),v::ConversionAttributes{}));
  auto origin=inUse(origin_use,[&]{return scalar(fragment->origin);});
  origin.pointer_provenance.clear();
  auto cell=repAbsoluteCell(fragment->allocation,emitOperation("Add","u64",
      {result.graph.convert(origin,"u64",step),offsetValue(plus(place,offset))}),type.toString());
  if(fragment->guard) {
    const auto guard=inUse(guard_use,[&]{return scalar(*fragment->guard);});
    cell=emitOperation("Select",type.toString(),{guard,cell,result.graph.constant(type.toString(),"0")});
  }
  return numerical?result.graph.convert(cell,numerical->toString(),"NumericInterpretation"):cell;
}

inline auto Lower::read(v::Value value,const Offset& offset,const v::ElementType& type) -> s::Ref {
  if(auto found=memory.symbolic_initial_reads.find({value.nodeId(),value.instanceIndex()});
      found!=memory.symbolic_initial_reads.end()) {
    const auto& region=found->second;
    auto [lo,hi]=bounds(offset);
    const auto bytes=(type.width()+7)/8;
    if(lo<0 || hi<0 || std::uint64_t(hi)>region.bytes || bytes>region.bytes-std::uint64_t(hi))
      fail("scalar_fold_invalid_index","symbolic initial-memory slice exceeds its recorded payload");
    // Outside a template projection the instance is this one: its exact position.
    if(!indexed_context)
      if(const auto position=memory.symbolic_read_positions.find({value.nodeId(),value.instanceIndex()});
         position!=memory.symbolic_read_positions.end())
        return positionedCell(position->second,plus(Offset(integer(position->second.constant.offset)),offset),type.toString());
    // A TMA region fragment (MemoryAttributes::region_offset): its address operand is its
    // allocation-relative byte offset, with no pointer root, and its guard is its zero fill.
    const auto& load=dynamic_cast<const v::LoadOp&>(input.definingNode(value).operation());
    if(load.attributes().region_offset) {
      // Lifting: the cell's offset is the region's origin (relative, rebuilt per member with
      // its unit parameters and its own index cells) plus the fragment's place and the region
      // binding's offset, the region's placement anchor, which admission keeps at offset 0 of
      // the allocation in every unit (query_log.cpp: opaque and affine TMA forms). The cell
      // moves by δ(allocation) at instantiation, so that δ must be 0 in every member: the anchor
      // does not move. Any other δ refuses.
      // An exact region (MemoryAttributes::region_exact) is the cell at its literal first byte,
      // which admission compared up to δ(allocation): no origin to rebuild.
      if(load.attributes().region_exact && !load.attributes().guarded && !region.offset)
        if(const auto fragment=regionFragmentLoad(input,memory,value))
          if(const auto exact=exactRegionOrigin(value,*fragment))
            return exactRegionCell(region.allocation,*exact,plus(Offset(integer(fragment->place)),offset),type,nullptr);
      if(options.unit_lifting && !anchoredRegion(region.allocation))
        liftRefuse("TMA region fragment read at its allocation-relative address");
      const auto operands=input.instanceOperands(value);
      const auto use=[&](std::size_t k)->std::shared_ptr<const IndexedSourceUse> {
        return indexed_context && indexed_context->use &&
            indexed_context->use->kind==IndexedSourceUse::Kind::Operands?indexed_context->use->children.at(k):nullptr;
      };
      auto relative=inUse(use(0),[&]{return scalar(operands[0]);});
      relative.pointer_provenance.clear();
      auto cell=repAbsoluteCell(region.allocation,emitOperation("Add","u64",
          {relative,offsetValue(plus(offset,Offset(integer(region.offset))))}),type.toString());
      if(!load.attributes().guarded)return cell;
      const auto guard=inUse(use(1),[&]{return scalar(operands[1]);});
      return emitOperation("Select",type.toString(),{guard,cell,result.graph.constant(type.toString(),"0")});
    }
    // Preserve the exact integer address, including page-table loads. This is
    // a subtraction of the real object base, not substitution of base=0.
    const auto base=allocation_bases.find(region.allocation);
    if(base==allocation_bases.end())
      fail("symbolic_initial_base_unsupported","No pointer root for symbolic initial-memory allocation "+region.allocation);
    const auto address=input.instanceOperands(value).at(0);
    const auto use=indexed_context && indexed_context->use &&
        indexed_context->use->kind==IndexedSourceUse::Kind::Operands?
        indexed_context->use->children.at(0):nullptr;
    auto base_value=inUse(nullptr,[&]{return scalar(base->second);});
    // A cvta.to.global root: (root parameter + terms) - root parameter keeps the
    // parameter in the offset's cone (a κ-covariant cell for lifting) without the cvta.
    const auto terms=cvtaRelative(address,base->second,use,0);
    auto address_value=terms?emitOperation("Add","u64",{base_value,*terms}):
        inUse(use,[&]{return scalar(address);});
    const auto relative=emitOperation("Subtract","u64",{address_value,base_value});
    // Lifting: the payload offset is relative to the address (serialization
    // sets it to 0 for Loads); a nonzero one would mix a representative-
    // absolute displacement into a covariant offset.
    if(options.unit_lifting && region.offset)liftRefuse("symbolic initial read with an absolute payload offset");
    return covariantCell(region.allocation,emitOperation("Add","u64",
        {relative,offsetValue(plus(offset,Offset(integer(region.offset))))}),type.toString());
  }
  if(spatial_count) {
    const auto stride=spatial_instance_strides.find(valueKey(value));
    if(stride!=spatial_instance_strides.end()) {
      const MemoryInstance key{value.nodeId(),value.instanceIndex()};
      auto found=memory.initial_read_families.upper_bound(key);
      if(found!=memory.initial_read_families.begin()) {
        const auto& family=std::prev(found)->second;
        const auto last=static_cast<__int128>(key.instance)+static_cast<__int128>(stride->second)*(spatial_count-1);
        if(family.first.node==key.node && key.instance>=family.first.instance &&
           key.instance-family.first.instance<family.count && last>=family.first.instance &&
           last-static_cast<__int128>(family.first.instance)<family.count) {
          const auto first=memory.initialRead(key);
          const auto step=static_cast<__int128>(stride->second)*family.byte_stride;
          if(first && first->region.offset<=INT64_MAX && step>=INT64_MIN && step<=INT64_MAX) {
            const auto [lo,hi]=bounds(offset);
            if(lo>=0 && hi>=0 && std::uint64_t(hi)<=first->region.bytes &&
               type.width()/8<=first->region.bytes-std::uint64_t(hi)) {
              Offset address(static_cast<std::int64_t>(first->region.offset));
              if(step)address.terms[0]=static_cast<std::int64_t>(step);
              return repAbsoluteCell(first->region.allocation,offsetValue(plus(address,offset)),type.toString());
            }
          }
        }
      }
    }
    std::vector<v::Value> replacements;
    std::vector<std::uint64_t> offsets;
    std::vector<InitialMemorySymbol> initials;
    const auto& members=spatialValues(value);
    for(std::size_t member=0;member<members.count;++member) {
      const auto x=members.at(member);
      auto found=memory.normalizedRead({x.nodeId(),x.instanceIndex()});
      auto family_initial=memory.initialRead({x.nodeId(),x.instanceIndex()});
      if(family_initial)found=NormalizedReadReference{&*family_initial};
      if(!found)spatialReject();
      const v::Value* replacement=nullptr;
      const InitialMemorySymbol* initial=nullptr;
      std::uint64_t source_offset=0,source_bytes=0;
      if(const auto* item=std::get_if<const v::Value*>(&*found))replacement=*item;
      else if(const auto* item=std::get_if<const InitialMemorySymbol*>(&*found))initial=*item;
      else {
        if(!std::holds_alternative<const MemoryView*>(*found))spatialReject();
        const auto& view=**std::get_if<const MemoryView*>(&*found);
        if(view.indexed_source)spatialReject();
        if(view.blocks.size()!=1)spatialReject();
        const auto& block=view.blocks[0];
        if(block.result_offset || !block.axes.empty() || block.bytes<view.type.bitSize()/8 ||
           block.source>=view.sources.size())spatialReject();
        const auto& source=view.sources[block.source];
        source_offset=block.source_offset;source_bytes=block.bytes;
        replacement=std::get_if<v::Value>(&source);
        initial=std::get_if<InitialMemorySymbol>(&source);
      }
      if(replacement) {
        if(source_offset)spatialReject();
        replacements.push_back(*replacement);
      } else if(initial) {
        auto item=*initial;
        if(source_offset>item.region.bytes || source_bytes>item.region.bytes-source_offset ||
           source_offset>UINT64_MAX-item.region.offset)spatialReject();
        item.region.offset+=source_offset;
        if(source_bytes)item.region.bytes=source_bytes;
        if(!initials.empty() && (item.type!=initials[0].type ||
           item.region.allocation!=initials[0].region.allocation || item.region.bytes!=initials[0].region.bytes))spatialReject();
        offsets.push_back(item.region.offset);initials.push_back(std::move(item));
      } else spatialReject();
    }
    if(replacements.size()==spatial_count) {
      const auto source=replacements.front();registerSpatial(std::move(replacements));
      return project(source,offset,type);
    }
    if(initials.empty() || offsets.size()!=spatial_count)spatialReject();
    const auto* first=&initials[0];
    const auto [lo,hi]=bounds(offset);
    if(lo<0 || hi<0 || std::uint64_t(hi)>first->region.bytes || type.width()/8>first->region.bytes-std::uint64_t(hi))spatialReject();
    return repAbsoluteCell(first->region.allocation,
        offsetValue(plus(offset,spatialOffsets(offsets))),type.toString());
  }
  auto found=memory.normalizedRead({value.nodeId(),value.instanceIndex()});
  auto family_initial=memory.initialRead({value.nodeId(),value.instanceIndex()});
  if(family_initial)found=NormalizedReadReference{&*family_initial};
  if(!found)fail("scalar_fold_missing_read","read has no normalized memory substitution");
  if(const auto* r=std::get_if<const v::Value*>(&*found))return project(**r,offset,type);
  if(const auto* r=std::get_if<const InitialMemorySymbol*>(&*found))return initial(**r,offset,type);
  if(const auto* r=std::get_if<const PartitionedRead*>(&*found))return partition(**r,value,offset,type);
  return view(*std::get<const MemoryView*>(*found),offset,type);
}

} // namespace
} // namespace sentryward::value_analysis
