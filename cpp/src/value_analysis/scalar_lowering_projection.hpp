#pragma once
#include "scalar_fold_lowering_impl.hpp"

namespace sentryward::value_analysis {
namespace {

// Decode an already projected integer index into the shared range algebra.
// One worklist over this index DAG, not over matrix/math ancestors or member
// values. Every arithmetic/conversion step must preserve its exact value on
// the current domain; bit operations are accepted only with proven bounds.
inline auto Lower::componentOffset(s::Ref root) -> Offset {
  std::unordered_map<s::Ref,Offset,s::RefHash> done;
  struct Frame {s::Ref ref;std::size_t child=0;};
  std::vector<Frame> work{{root}};
  const auto fits=[&](const Offset& x,const std::string& type) {
    if(type.size()<2 || std::string("ubs").find(type[0])==std::string::npos)return false;
    unsigned width=0;const auto [end,error]=std::from_chars(type.data()+1,type.data()+type.size(),width);
    if(error!=std::errc{} || end!=type.data()+type.size() || !width || width>64)return false;
    const auto [lo,hi]=bounds(x);
    const __int128 low=type[0]=='s'?-(__int128(1)<<(width-1)):0;
    const __int128 high=type[0]=='s'?(__int128(1)<<(width-1))-1:(__int128(1)<<width)-1;
    return lo>=low && hi<=high;
  };
  while(!work.empty()) {
    auto& frame=work.back();const auto& n=result.graph.node(frame.ref);
    if(n.kind!=s::Kind::Constant && n.kind!=s::Kind::Index &&
        !(n.kind==s::Kind::Operation && n.operands.size()==2 &&
          (n.name=="Add" || n.name=="Subtract" || n.name=="Multiply" ||
           n.name=="Divide" || n.name=="Remainder" || n.name=="ShiftLeft" ||
           n.name=="ShiftRight" || n.name=="BitAnd")))
      fail("compact_view_unsupported","component index requires an unsupported coordinate operation: "+n.name);
    if(frame.child<n.operands.size()) {
      auto child=n.operands[frame.child++];
      if(!done.contains(child))work.push_back({std::move(child)});
      continue;
    }
    std::optional<Offset> value;
    if(n.kind==s::Kind::Constant) {if(auto x=literal(n.name))value=Offset(*x);}
    else if(n.kind==s::Kind::Index && n.index_level<domains.size()) {
      value=Offset{};value->terms[n.index_level]=1;
    } else if(n.kind==s::Kind::Operation && n.operands.size()==2) {
      const auto& a=done.at(n.operands[0]);const auto& b=done.at(n.operands[1]);
      const auto [al,ah]=bounds(a);const auto [bl,bh]=bounds(b);
      if(n.name=="Add")value=plus(a,b);
      else if(n.name=="Subtract")value=plus(a,times(b,-1));
      else if(n.name=="Multiply") {
        if(bl==bh)value=times(a,bl);else if(al==ah)value=times(b,al);
      } else if(bl==bh && bl>0 && al>=0 && n.name=="Divide")value=quotient(a,std::uint64_t(bl));
      else if(bl==bh && bl>0 && al>=0 && n.name=="Remainder")value=remainder(a,std::uint64_t(bl));
      else if(bl==bh && bl>=0 && bl<63 && al>=0 && n.name=="ShiftLeft")value=times(a,std::int64_t{1}<<bl);
      else if(bl==bh && bl>=0 && bl<63 && al>=0 && n.name=="ShiftRight")value=quotient(a,std::uint64_t{1}<<bl);
      else if(bl==bh && bl>=0 && bl<INT64_MAX && al>=0 && n.name=="BitAnd" &&
          std::has_single_bit(std::uint64_t(bl)+1))value=remainder(a,std::uint64_t(bl)+1);
    }
    if(!value || !fits(*value,n.type))
      fail("compact_view_unsupported","component index is outside exact affine/radix range algebra: "+n.name);
    // Validate each conversion edge, including intermediate narrowing, not
    // merely the final type. Never erase an FP or saturating conversion.
    std::string_view steps=frame.ref.conversion;
    while(!steps.empty()) {
      const auto split=steps.find("; ");const auto step=steps.substr(0,split);
      const auto open=step.find('('),arrow=step.find(" -> ");
      if(open==std::string_view::npos || arrow==std::string_view::npos || step.back()!=')' ||
          (step.substr(0,open)!="Convert" && step.substr(0,open)!="Bitcast") ||
          !fits(*value,std::string(step.substr(open+1,arrow-open-1))) ||
          !fits(*value,std::string(step.substr(arrow+4,step.size()-arrow-5))))
        fail("compact_view_unsupported","component index conversion is not range-preserving");
      if(split==std::string_view::npos)break;
      steps.remove_prefix(split+2);
    }
    done.emplace(frame.ref,std::move(*value));work.pop_back();
  }
  return done.at(root);
}

// The byte-slice step of a same-width bitcast, an attribute-free integer narrowing, or a right
// shift by a constant multiple of 8, as a value conversion or an operation node. An attribute-free
// Convert between integer scalars keeps the low-order bits, i.e. the low bytes in PTX's
// little-endian order; a right shift by 8k moves byte i+k to byte i; a bitcast keeps every byte.
// So a constant byte slice inside the result is the same-width slice of the source at the same
// (resp. +k) offset. `conversion_source` replaces a value conversion's source (its use child).
// O(1).
inline auto Lower::sliceStep(const v::Value& value,const Offset& offset,const v::ElementType& desired,
    std::optional<v::Value> conversion_source) const -> std::optional<SliceStep> {
  if(value.threadSelector() || value.pointer() || !value.indices().empty())return {};
  const auto& type=input.type(value);
  if(!type.isScalar() || !desired.width() || desired.width()%8 || type.element().width()%8)return {};
  const auto [lo,hi]=bounds(offset);
  const auto bytes=type.element().width()/8;
  if(lo<0 || lo!=hi || std::uint64_t(lo)+desired.width()/8>bytes)return {};
  const auto fits=[&](const v::Value& source,bool convert) {
    const auto& source_type=input.type(source);
    if(!source_type.isScalar() || source_type.element().width()%8)return false;
    if(!convert)return source_type.bitSize()==type.bitSize();
    return integerKind(type.element()) && integerKind(source_type.element()) &&
        source_type.element().width()>type.element().width();
  };
  if(const auto& c=value.conversion()) {
    const auto source=conversion_source.value_or(c->source);
    if(c->spec->kind==v::TypeConversionKind::Bitcast && fits(source,false))return SliceStep{source,offset};
    if(c->spec->kind==v::TypeConversionKind::Convert && c->spec->attributes==v::ConversionAttributes{} &&
       fits(source,true))return SliceStep{source,offset};
    return {};
  }
  const auto& op=input.definingNode(value).operation();
  const auto operands=input.instanceOperands(value);
  if(const auto* convert=dynamic_cast<const v::ConvertOp*>(&op);
     convert && convert->attributes()==v::ConversionAttributes{} && operands.size()==1 && fits(operands[0],true))
    return SliceStep{operands[0],offset};
  if(dynamic_cast<const v::BitcastOp*>(&op) && operands.size()==1 && fits(operands[0],false))
    return SliceStep{operands[0],offset};
  if(const auto* shift=dynamic_cast<const v::BinaryOp*>(&op);
     shift && shift->opcode()==v::BinaryOpcode::ShiftRight && integerKind(type.element()) &&
     operands.size()==2 && input.type(operands[0])==type)
    // Only a literal amount: it is the same for every member under a class-mode use.
    if(const auto amount=dynamic_cast<const v::ConstantOp*>(&input.definingNode(operands[1]).operation())?
           knownAt(operands[1],0,nullptr):std::nullopt;
       amount && *amount>=0 && *amount%8==0 &&
       std::uint64_t(*amount/8)<bytes && std::uint64_t(lo)+desired.width()/8<=bytes-std::uint64_t(*amount/8))
      return SliceStep{operands[0],Offset(lo+*amount/8),0};
  return {};
}

// Whether slice steps from `x` reach a loaded scalar (possibly a component of a vector load,
// ld.v2.b64 {a, b}). Only reads can assemble their bytes from several stores or leave some
// unwritten; any other narrowing keeps its Convert/ExtractBits form. O(1): at most 6 steps.
inline auto Lower::reachesLoad(v::Value x) const -> bool {
  for(unsigned depth=0;depth<6;++depth) {
    if(x.threadSelector() || x.pointer())return false;
    if(!x.conversion() && dynamic_cast<const v::LoadOp*>(&input.definingNode(x).operation()))return true;
    // The low byte of x: a chain step that exists for some byte exists for this one.
    const auto step=sliceStep(x,Offset{},v::ElementType::bit(8));
    if(!step)return false;
    x=step->source;
  }
  return false;
}

// Projects a byte slice through one slice step whose chain reaches a load, keeping the source's
// use: plain, a Conversion use's child, or an Operands use's operand child. Neither the shifted
// out nor the truncated bytes of the source are requested.
inline auto Lower::projectSlice(const v::Value& value,const Offset& offset,const v::ElementType& desired) -> std::optional<s::Ref> {
  std::shared_ptr<const IndexedSourceUse> child;
  std::optional<v::Value> conversion_source;
  const bool indexed=bool(indexed_context);
  if(indexed) {
    const auto& use=indexed_context->use;
    if(!use)return std::nullopt;
    if(value.conversion()) {
      if(use->kind!=IndexedSourceUse::Kind::Conversion || use->children.size()!=1)return std::nullopt;
      child=use->children[0];conversion_source=child->prototype;
    } else if(use->kind!=IndexedSourceUse::Kind::Operands || use->prototype!=value)return std::nullopt;
  }
  const auto step=sliceStep(value,offset,desired,conversion_source);
  if(!step || !reachesLoad(step->source))return std::nullopt;
  if(indexed && !value.conversion()) {
    if(step->operand>=indexed_context->use->children.size())return std::nullopt;
    child=indexed_context->use->children[step->operand];
  }
  ++result.narrowing_slice_projections;
  if(!indexed)return project(step->source,step->offset,desired);
  return inUse(child,[&]{return project(step->source,step->offset,desired);});
}

inline auto Lower::projectImpl(v::Value value,Offset offset,const v::ElementType& desired) -> s::Ref {
  if(indexed_context && indexed_context->use) {
    const auto use=indexed_context->use;
    if(use->prototype!=input.canonicalInterpretation(value))
      fail("scalar_fold_prerequisite","indexed projection lost its use-position binding");
    if(use->kind==IndexedSourceUse::Kind::MemberTemplate)
      return projectMemberTemplate(value,offset,desired);
    if(use->kind==IndexedSourceUse::Kind::Component) {
      const auto matrix=use->children.at(0);
      const auto& shape=input.type(matrix->prototype).shape();
      if(use->children.size()!=shape.size()+1)
        fail("scalar_fold_component_rank","component use has inconsistent index mappings");
      Offset element;
      for(std::size_t axis=0;axis<shape.size();++axis) {
        const auto child=use->children[axis+1];
        auto coordinate=componentOffset(inUse(child,[&]{return scalar(child->prototype);}));
        const auto [lo,hi]=bounds(coordinate);
        if(lo<0 || hi<lo || std::uint64_t(hi)>=shape[axis])
          fail("scalar_fold_component_index","mapped component index exceeds matrix shape");
        element=plus(times(element,integer(shape[axis])),coordinate);
      }
      const auto width=input.type(matrix->prototype).element().width();
      if(!width || width%8)fail("scalar_fold_subbyte_value","component payload must be byte aligned");
      return inUse(matrix,[&]{return project(matrix->prototype,
          plus(offset,times(element,integer(width/8))),desired);});
    }
    if(use->kind==IndexedSourceUse::Kind::Exact || use->kind==IndexedSourceUse::Kind::OneHotLiteral)
      return inUse({},[&]{return project(value,offset,desired);});
    if(use->kind==IndexedSourceUse::Kind::Forward || use->kind==IndexedSourceUse::Kind::Selected)
      return inUse(use->children.at(0),[&]{return project(use->children.at(0)->prototype,offset,desired);});
    if(use->kind==IndexedSourceUse::Kind::NonzeroSelect || use->kind==IndexedSourceUse::Kind::BitTestSelect) {
      const auto& type=input.type(value);
      if(!type.isScalar())
        fail("scalar_fold_conversion_projection","conditional source requires its complete scalar");
      // A byte slice of the selected scalar: the slice of the complete selection.
      if(bounds(offset)!=std::pair<std::int64_t,std::int64_t>{0,0} || type.element().width()!=desired.width()) {
        const auto [lo,hi]=bounds(offset);
        if(lo<0 || hi<lo || !type.element().width() || type.element().width()%8 || !desired.width() || desired.width()%8 ||
           std::uint64_t(hi)>type.element().width()/8 || desired.width()/8>type.element().width()/8-std::uint64_t(hi))
          fail("scalar_fold_conversion_projection","conditional source requires its complete scalar");
        const auto whole=inContext(indexed_context,[&]{return project(value,Offset{},type.element());});
        return emitOperation("ExtractBits",desired.toString(),{whole,offsetValue(times(offset,8))},
            "width="+std::to_string(desired.width()));
      }
      const auto child=[&](std::size_t i) {
        const auto source=use->children.at(i);
        return inUse(source,[&]{return scalar(source->prototype);});
      };
      auto test=child(0),yes=child(1),no=child(2);
      auto test_type=input.type(use->children.at(0)->prototype).element().toString();
      if(use->kind==IndexedSourceUse::Kind::BitTestSelect) {
        const auto& map=*use->binding;
        auto bit=plus(Offset(map.base),plus(times(remainder(indexed_context->index,map.period),map.stride),
            times(quotient(indexed_context->index,map.period),map.outer_stride)));
        const auto [first,last]=bounds(bit);
        const auto width=input.type(use->children.at(0)->prototype).element().width();
        if(first<0 || last>=width)fail("scalar_fold_invalid_index","bit-test index exceeds mask width");
        test=result.graph.convert(test,"b"+std::to_string(width),"Bitcast");
        test=emitOperation("ExtractBits","u32",{test,offsetValue(bit)},"width=1");
        test_type="u32";
      }
      auto predicate=emitOperation("Compare","pred",
          {test,result.graph.constant(test_type,"0")},
          attributes(v::CompareOp(v::ComparePredicate::NotEqual,
              v::ValueType::scalar(v::ElementType::predicate()))));
      return cast(emitOperation("Select",type.element().toString(),{predicate,yes,no}),desired);
    }
  }
    if(const auto* found=indexedBinding(value)) {
      const auto& binding=*found;
      if(binding.fixed) {
        return inUse({},[&]{return project(value,offset,desired);});
      }
      auto position=mappedCoordinate(binding,indexed_context->index);
      const auto& t=input.type(value);
      const auto [lo,hi]=bounds(offset);
      if(lo<0 || hi<lo || std::uint64_t(hi)>t.bitSize()/8 || desired.width()/8>t.bitSize()/8-std::uint64_t(hi))
        fail("compact_view_unsupported","indexed source projection exceeds its value");
      if(binding.numerical_field && (lo!=0 || hi!=0 || desired!=t.element()))
        fail("scalar_fold_conversion_projection","decoded floating field requires its complete numerical scalar");
      const auto field=binding.numerical_field.value_or(desired);
      if(binding.initial) {
        auto r=repAbsoluteCell(binding.initial->region.allocation,
            offsetValue(plus(position,offset)),field.toString());
        return binding.numerical_field?result.graph.convert(r,desired.toString(),"NumericInterpretation"):r;
      }
      if(binding.region_fragment) {
        // A grid family's fragment: its children are the uses of the origin and of the guard;
        // a slice names its fragment Load as the producer.
        const auto& children=indexed_context->use->children;
        return regionFragmentCell(binding.producer?*binding.producer:value,position,offset,field,
            binding.numerical_field?&desired:nullptr,
            children.size()>0?children[0]:nullptr,children.size()>1?children[1]:nullptr);
      }
      if(binding.producer) {
        // A grid family's slice of a wider scalar: the producer is projected under its own use
        // (the binding's child) at the mapped byte offset.
        if(!indexed_context->use->children.empty()) {
          auto r=inUse(indexed_context->use->children[0],
              [&]{return project(*binding.producer,plus(position,offset),field);});
          return binding.numerical_field?result.graph.convert(r,desired.toString(),"NumericInterpretation"):r;
        }
        // The affine offset belongs to this indexed use, but the shared
        // producer retains its own leaves and occurrence identity.
        auto r=inContext(binding.producer_uses_bindings?indexed_context:nullptr,
            [&]{return project(*binding.producer,plus(position,offset),field);});
        return binding.numerical_field?result.graph.convert(r,desired.toString(),"NumericInterpretation"):r;
      }
      if(binding.addend) {
        // A grid family's induction value: its base (the binding's child use) plus the mapped
        // step, added in the value's integer type.
        const auto& children=indexed_context->use->children;
        if(children.size()!=1 || !t.isScalar())
          fail("scalar_fold_prerequisite","induction binding lost its base");
        const auto base=inUse(children[0],[&]{return this->scalar(children[0]->prototype);});
        const auto sum=emitOperation("Add",t.element().toString(),{base,integerValue(position,t.element())},
            attributes(v::BinaryOp(v::BinaryOpcode::Add,t)));
        if(lo==0 && hi==0 && desired.width()==t.element().width())return cast(sum,desired);
        return emitOperation("ExtractBits",desired.toString(),{sum,offsetValue(times(offset,8))},
            "width="+std::to_string(desired.width()));
      }
      const auto* lifted=liftedBindingCoordinate(binding,position);
      // A Bit-kind or signed literal of a grid family: integerValue.
      const bool bits=!lifted && t.element().kind()==v::ElementKind::Bit;
      const bool grid_literal=!lifted && !binding.power_of_two &&
          (bits || t.element().kind()==v::ElementKind::SignedInteger);
      auto scalar=lifted?
          result.graph.convert(unitCoordinate(lifted->axis(),std::uint64_t(position.base),"u64"),
              t.element().toString(),"Convert"):
          grid_literal?integerValue(position,t.element()):
          result.graph.convert(offsetValue(position),
              bits?"u"+std::to_string(t.element().width()):t.element().toString(),"Convert");
      if(binding.power_of_two) {
        const auto [first,last]=bounds(position);
        if(first<0 || last>=t.element().width() || binding.initial || binding.producer)
          fail("scalar_fold_invalid_index","one-hot scalar bit index exceeds its exact width");
        if(bits) {
          const auto unsigned_type="u"+std::to_string(t.element().width());
          scalar=result.graph.convert(emitOperation("ShiftLeft",unsigned_type,
              {result.graph.constant(unsigned_type,"1"),scalar}),t.element().toString(),"Bitcast");
        } else
        scalar=emitOperation("ShiftLeft",t.element().toString(),
            {result.graph.constant(t.element().toString(),"1"),scalar});
      }
      if(lo==0 && hi==0 && desired.width()==t.element().width())return cast(scalar,desired);
      return emitOperation("ExtractBits",desired.toString(),{scalar,offsetValue(times(offset,8))},
          "width="+std::to_string(desired.width()));
    }
  if(spatial_count)prepareSpatial(value);
  // A conversion use already binds its source (including any cross-thread
  // selection). Resolve that child under its own map below. Evaluating the
  // selector under the conversion's map confuses two different use sites.
  const bool mapped_conversion=value.conversion() && indexed_context && indexed_context->use &&
      indexed_context->use->kind==IndexedSourceUse::Kind::Conversion;
  if(!mapped_conversion)value=selected(value);
  const auto& type=input.type(value);
  if(type.element().isParameterized() || !type.hasStaticShape())
    fail("scalar_fold_parameterized_type","value type or shape depends on an unresolved descriptor");
  if(!value.conversion() && type.isScalar() && type.element().kind()==v::ElementKind::Predicate &&
     desired.kind()==v::ElementKind::Predicate && !offset.expression && offset.base==0 && offset.terms.empty()){
    return whole(value,Offset{});
  }
  const bool subbyte = !desired.width() || desired.width()%8 ||
      !type.element().width() || type.element().width()%8;
  if(subbyte) {
    // A complete numeric scalar (e.g. FP4 conversion or its u4 bitcast)
    // has no byte index. Only storage projection requires byte alignment.
    // Never route a Load through whole(): memory still requires whole bytes.
    // A conversion view of a Load (the low e2m1 field of a loaded word:
    // Convert -> u8, Convert -> u4, Bitcast -> e2m1) is not: its source is.
    const auto [lo,hi]=bounds(offset);
    if(!type.isScalar() || !value.indices().empty() || lo!=0 || hi!=0 ||
        !desired.width() || desired.width()!=type.element().width() ||
        (!value.conversion() && dynamic_cast<const v::LoadOp*>(&input.definingNode(value).operation())))
      fail("scalar_fold_subbyte_value","projection requires byte-aligned scalar elements");
    if(!value.conversion())return cast(whole(value,Offset{}),desired);
    // Conversion handling below retains its exact conversion attributes and
    // recursively requests the entire source scalar, not a byte slice.
  }
  if(!indexed_context && type.isScalar() && desired.width()<type.element().width()) {
    const auto [lo,hi]=bounds(offset);
    if(lo>=0 && lo==hi && std::uint64_t(lo)<type.element().width()/8)
      if(auto fields=packedFields(value);fields && fields->size()>1)
        for(const auto& field:*fields)
          if(field.start==std::uint64_t(lo)*8 && field.width==desired.width())
            return project(field.value,Offset{},desired);
  }
  // Exact slice rules: a byte slice of a same-width bitcast is the source's
  // slice, and a slice of a Select (selp chooses a whole value) is the
  // Select of the alternatives' slices. Neither reads the other bits.
  const auto select_slice=[&](const v::Operation& op)->std::optional<s::Ref> {
    const auto operands=input.instanceOperands(value);
    if(operands.size()!=3 || !input.type(operands[0]).isScalar())return std::nullopt;
    // Under an indexed operand use (the matrix rule below), each operand keeps its member
    // mapping, as pure() projects it.
    const auto within=[&](std::size_t k,auto&& fn) {
      return indexed_context?inUse(indexed_context->use->children.at(k),fn):fn();
    };
    auto condition=within(0,[&]{return scalar(operands[0]);});
    // In the unspecified cone, the selection-context rule of pure(): an arm the context
    // or a constant rules out is not lowered; each lowered arm gets its literals.
    const bool cone=!spatial_count && inUnspecifiedCone(value);
    if(cone)if(const auto runs=decidedPredicate(condition))
      return within(*runs?1:2,[&]{return project(operands[*runs?1:2],offset,desired);});
    const auto arm=[&](std::size_t i) {
      SelectScope scope(*this,cone?selectContextFor(condition,i==1):select_context);
      return within(i,[&]{return project(operands[i],offset,desired);});
    };
    auto yes=arm(1);
    auto no=arm(2);
    return emitOperation("Select",desired.toString(),{std::move(condition),std::move(yes),std::move(no)},
        attributes(op));
  };
  if(!indexed_context && type.isScalar() && desired.width()%8==0 && desired.width()<type.element().width() &&
     value.indices().empty()) {
    const auto [lo,hi]=bounds(offset);
    if(lo>=0 && lo==hi && std::uint64_t(lo)+desired.width()/8<=type.element().width()/8) {
      if(const auto& c=value.conversion()) {
        if(c->spec->kind==v::TypeConversionKind::Bitcast && input.type(c->source).isScalar() &&
           input.type(c->source).bitSize()==type.bitSize())
          return project(c->source,offset,desired);
      } else if(const auto& op=input.definingNode(value).operation();dynamic_cast<const v::SelectOp*>(&op)) {
        if(auto sliced=select_slice(op))return *sliced;
      }
    }
  }
  // Exact byte-slice rules for narrowing, byte shifts and bitcasts of loaded scalars (projectSlice).
  if(auto sliced=projectSlice(value,offset,desired))return *sliced;
  // The same Select rule for whole matrices chosen by a scalar condition (a cp.async's zero-filled
  // payload), for a slice that crosses their elements, where the elementwise rule cannot apply.
  const bool operand_use=!indexed_context || (indexed_context->use &&
      indexed_context->use->kind==IndexedSourceUse::Kind::Operands && indexed_context->use->prototype==value &&
      indexed_context->use->children.size()==3);
  if(operand_use && type.isMatrix() && !value.conversion() && value.indices().empty() &&
     desired.width()%8==0) {
    // Any offset inside the value; the crossing test is the elementwise rule's own (below).
    const auto [lo,hi]=bounds(offset);
    const std::uint64_t element=type.element().width()/8,width=desired.width()/8;
    const auto [il,ih]=bounds(remainder(offset,element));
    if(lo>=0 && hi>=lo && std::uint64_t(hi)+width<=type.bitSize()/8 &&
       (il<0 || ih<0 || std::uint64_t(ih)+width>element))
      if(const auto& op=input.definingNode(value).operation();dynamic_cast<const v::SelectOp*>(&op))
        if(auto sliced=select_slice(op))return *sliced;
  }
  if(value.conversion()){
    if(!value.indices().empty())fail("scalar_fold_conversion_projection","component selection after conversion is not lowered");
    if(!type.isScalar())fail("scalar_fold_conversion_projection","matrix conversion requires per-element rules");
    const auto& c=*value.conversion();
    const auto child=indexed_context && indexed_context->use?indexed_context->use->children.at(0):nullptr;
    auto base=inUse(child,[&]{return scalar(child?child->prototype:c.source);});
    std::string step=c.spec->kind==v::TypeConversionKind::Bitcast?"Bitcast":"Convert";
    if(c.spec->kind==v::TypeConversionKind::Convert)step+=" "+attributes(v::ConvertOp(c.spec->target,c.spec->attributes));
    // e2m1 reads its four storage bits as b4 (as the MMA operand path does); a u4 value
    // reinterpreted as e2m1 holds the same four bits.
    if(step=="Bitcast" && type.element().kind()==v::ElementKind::Floating && type.element().width()==4 &&
       result.graph.type(base)=="u4")
      base=result.graph.convert(base,"b4","Bitcast");
    auto converted=result.graph.convert(base,type.element().toString(),step);
    if(c.spec->kind==v::TypeConversionKind::Convert)
      if(auto exact=constantConversion(base,c.spec->attributes,type.element()))converted=*exact;
    auto [lo,hi]=bounds(offset);
    if(lo==0 && hi==0 && desired.width()==type.element().width())return cast(converted,desired);
    if(lo<0 || hi<0 || add(hi,desired.width()/8)>integer(type.bitSize()/8))
      fail("scalar_fold_invalid_index","converted scalar slice exceeds value");
    return emitOperation("ExtractBits",desired.toString(),{converted,offsetValue(times(offset,8))},
      "width="+std::to_string(desired.width()));
  }
  if(!value.indices().empty()){
    auto base=value.withoutSelection();const auto& shape=input.type(base).shape();
    if(shape.size()!=value.indices().size())fail("scalar_fold_component_rank","component index rank mismatch");
    Offset index;
    for(std::size_t j=0;j<shape.size();++j){
      auto k=known(value.indices()[j]);
      if(!k || *k<0 || std::uint64_t(*k)>=shape[j])fail("scalar_fold_component_index","component index is not a fixed valid coordinate");
      index=plus(times(index,integer(shape[j])),Offset(*k));
    }
    return project(base,plus(times(index,integer(type.element().width()/8)),offset),desired);
  }
  const auto& op=input.definingNode(value).operation();
  if(dynamic_cast<const v::LoadOp*>(&op))return read(value,offset,desired);
  if(const auto* copy=dynamic_cast<const v::NativeMatrixOp*>(&op);
      copy && copy->opcode()==v::NativeMatrixOpcode::TcgenCopyValue && type.hasStaticShape()) {
    const auto source=input.instanceOperands(value).at(0);
    if(input.type(source)==type)return project(source,offset,desired);
  }
  const auto width=type.element().width()/8;
  auto [lo,hi]=bounds(offset);
  if(lo<0 || hi<0 || std::uint64_t(hi)>type.bitSize()/8 || desired.width()/8>type.bitSize()/8-std::uint64_t(hi))
    fail("scalar_fold_invalid_index","scalar projection exceeds producer value");
  // Inside an all-zero matrix every requested byte is zero, whatever the element boundaries.
  if(desired.width()%8==0 && desired.width()<=64 && zeroMatrixLiteral(value))
    return cast(result.graph.constant("b"+std::to_string(desired.width()),"0"),desired);
  auto index=quotient(offset,width),inner=remainder(offset,width);
  auto [il,ih]=bounds(inner);
  if(il<0 || ih<0 || std::uint64_t(ih)+desired.width()/8>width)
    fail("scalar_fold_cross_element","scalar projection crosses producer element boundaries");
  auto result=whole(value,index);
  if(il==0 && ih==0 && desired.width()==type.element().width())return cast(result,desired);
  return this->emitOperation("ExtractBits",desired.toString(),{result,offsetValue(times(inner,8))},
    "width="+std::to_string(desired.width()));
}

inline auto Lower::evaluateProjection(Projection& task) -> s::Ref {
  SENTRYWARD_LEDGER(ledger->evaluate(keys.text(task.key)));
  // A task owns its direct requested results, not every request made by
  // inlined descendants. A completed child's value already retains its
  // semantic operands. Rooting the child's temporary construction history
  // prevents Fold replacement from releasing concrete intermediate terms.
  struct RestoreRequests {
    KeySet*& target;
    KeySet* saved;
    ~RestoreRequests(){target=saved;}
  } restore_requests{requested,requested};
  // Explicitly suspended child results remain roots until this task has
  // completed. Clearing them on retry can repeatedly collect the very child
  // that the worklist just finished before an inline parent consumes it.
  requested=&task.dependencies;
  struct RestoreTask {
    const Projection*& current;const Projection* saved;
    ~RestoreTask(){current=saved;}
  } restore_task{active_projection,active_projection};
  active_projection=&task;
  struct RestoreFold {
    Lower& owner;std::shared_ptr<const FoldOwner> saved;
    ~RestoreFold(){owner.fold_owner=std::move(saved);}
  } restore_fold{*this,fold_owner};
  fold_owner=task.fold;
  SENTRYWARD_AUDIT(audit->add(AuditCounter::projection_evaluations);
    audit->maximum(AuditCounter::max_projection_depth,projection_depth));
  struct Restore {
    std::shared_ptr<const IndexedContext>& target;
    std::shared_ptr<const IndexedContext> saved;
    ~Restore(){target=std::move(saved);}
  } restore{indexed_context,indexed_context};
  struct RestoreKey {
    const ProjectionKey*& target;
    const ProjectionKey* saved;
    ~RestoreKey(){target=saved;}
  } restore_key{projection_key,projection_key};
  SelectScope restore_select{*this,task.select_context};
  projection_key=&task.key;
  indexed_context=task.indexed;
  current=MemoryInstance{task.value.nodeId(),task.value.instanceIndex()};
  limit();
  auto r=projectImpl(task.value,task.offset,task.type);
  const auto& op=input.definingNode(task.value).operation();
  const auto* pointer=task.value.pointer().get();
  if(!pointer) {
    if(const auto* c=dynamic_cast<const v::ConstantOp*>(&op);c && c->pointer())pointer=&*c->pointer();
    else if(const auto* p=dynamic_cast<const v::InputParameterOp*>(&op);p && p->pointer())pointer=&*p->pointer();
  }
  if(pointer)r=result.graph.withPointer(std::move(r),pointerLabel(*pointer));
  limit();
  keys.emplace(task.key,r);
  SENTRYWARD_LEDGER(ledger->complete(keys.text(task.key)));
  finishReduction(task.key);
  return r;
}

inline auto Lower::project(v::Value value,Offset offset,const v::ElementType& type) -> s::Ref {
  SENTRYWARD_AUDIT(audit->add(AuditCounter::projection_requests));
  offset=specializeSingletons(std::move(offset));
  value=input.canonicalInterpretation(value);
  struct RestoreContext {
    std::shared_ptr<const IndexedContext>& target;
    std::shared_ptr<const IndexedContext> saved;
    ~RestoreContext(){target=std::move(saved);}
  } restore_context{indexed_context,indexed_context};
  // Coordinate maps change the query, not the value. Compose them before
  // cache lookup/worklist insertion; equivalent mapped uses share one task.
  while(indexed_context && indexed_context->use &&
        indexed_context->use->kind==IndexedSourceUse::Kind::CoordinateMap) {
    const auto& use=indexed_context->use;
    if(use->prototype!=value)
      fail("scalar_fold_prerequisite","indexed projection lost its use-position binding");
    const auto& map=*use->binding;const auto child=use->children.at(0);
    auto coordinate=mappedCoordinate(map,indexed_context->index);
    indexed_context=std::make_shared<IndexedContext>(IndexedContext{std::move(coordinate),child});
    value=child->prototype;
  }
  // Indexed/spatial prototypes must use the already-validated family map,
  // not the concrete predicate of its first member. Outside those scopes,
  // fixed input facts can select an operand before constructing either arm.
  if(!indexed_context && !spatial_count)value=specialization.selected(value);
  bool shared_view=false;
  if(!indexed_context && !spatial_count && !offset.expression && offset.terms.empty() &&
      offset.base>=0 && type.width() && type.width()%8==0) {
    if(auto field=pure_source_templates.storageSlice(value,std::uint64_t(offset.base),type.width()/8)) {
      // The direct storage-slice path already canonicalizes a normalized
      // read to its source. Count a subsequent cache hit as view reuse too;
      // do not require the old Load spelling to survive that optimization.
      if(dynamic_cast<const v::LoadOp*>(&input.definingNode(value).operation()))
        shared_view=memory.normalizedRead({value.nodeId(),value.instanceIndex()}).has_value();
      value=field->value;offset=Offset(integer(field->offset));
    }
  }
  // The key text is spelled component by component into a reused buffer,
  // then interned once. A view identity replaces the value's key; valueKey
  // is pure, so not spelling it in that case is exact.
  auto& text=key_text;
  text.clear();
  if(spatial_count){text+="spatial:";appendDecimal(text,spatial_serial);text+=':';}
  bool view=false;
  if(!value.pointer() && !value.conversion() && value.indices().empty() && !value.threadSelector() &&
     dynamic_cast<const v::LoadOp*>(&input.definingNode(value).operation())) {
    auto read=memory.canonicalRead({value.nodeId(),value.instanceIndex()});
    if(const auto alias=memory.normalized_read_aliases.find(read);alias!=memory.normalized_read_aliases.end())
      read=alias->second;
    if(const auto canonical=memory.normalized_reads.find(read);
        canonical!=memory.normalized_reads.end()) {
      text+="view:";appendDecimal(text,read.node);text+=':';appendDecimal(text,read.instance);
      view=shared_view=true;
    }
  }
  if(!view)appendValueKey(text,value);
  text+='@';offset.appendKey(text);text+='/';text+=type.toString();text+='#';
  auto used=offset.binderMask();
  if(indexed_context) {
    used|=indexed_context->index.binderMask();
    index_text.clear();indexed_context->index.appendKey(index_text);
    text+="indexed:";indexed_use_identity.appendContext(text,indexed_context->use,index_text);
  }
  // A projection depends on its selected Value/use and index bindings, not
  // the extents of unrelated enclosing Folds. Keep binder positions/rank
  // (inner Fold levels rely on them), and all used extents. Spatial legacy
  // bindings retain their full environment. No operand ancestors are read.
  for(std::size_t i=0;i<domains.size();++i)
    if(spatial_count || i>=64 || (used&(std::uint64_t{1}<<i))){appendDecimal(text,domains[i]);text+=',';}
    else text+="_,";
  // Only a projection in the unspecified cone reads its selection context; any other one is
  // context-free and keeps its plain key and memo entry.
  const auto context=select_context && !spatial_count && inUnspecifiedCone(value)?select_context:0;
  if(context){text+="select:";appendDecimal(text,context);}
  const auto key=keys.intern(text);
  SENTRYWARD_LEDGER(ledger->request(keys.text(key),projection_key?&keys.text(*projection_key):nullptr,
      std::string(input.definingNode(value).operation().name())+(indexed_context?"/indexed":"/plain"));
    if(ledger->checkpoint())ledger->snapshot(result.graph,keys.size(),result.outputs.size()));
  SENTRYWARD_AUDIT(if(requested) {
    audit->add(AuditCounter::dependency_requests);
  });
  // Retain cache hits too: an unfinished parent needs these roots across GC.
  // One probe of the interned key serves the retention and the memo test.
  if(requested)requested->insert(key);
  if(const auto* hit=keys.find(key)){
    SENTRYWARD_LEDGER(ledger->hit(keys.text(key)));
    SENTRYWARD_AUDIT(audit->add(AuditCounter::projection_cache_hits));
    if(shared_view)++result.shared_view_projection_hits;
    return *hit;
  }
  // `value` is not read below; the request owns it.
  Projection request{std::move(value),std::move(offset),type,domains,key};
  request.indexed=indexed_context;
  request.fold=fold_owner;
  request.select_context=context;
  if(projecting) {
    if(projection_depth>=options.max_pending_projections)
      fail("projection_worklist_limit_exceeded","projection worklist budget exceeded");
    if(inline_projection_depth>=32) {
      SENTRYWARD_LEDGER(ledger->add("dependency_suspensions"));
      SENTRYWARD_AUDIT(audit->event("DEPENDENCY_SUSPEND","projection_exception_worklist",
        projection_depth,inline_projection_depth,request.value.nodeId()));
      // One unwind to the driver. While it unwinds, every inline ancestor's
      // InlineFrame moves its request into suspended_projections; no handler
      // in between may catch Projection or `...` (the driver checks the
      // count). Reserved first, so those nothrow pushes cannot throw.
      static_assert(std::is_nothrow_move_constructible_v<Projection>);
      suspended_projections.reserve(suspended_projections.size()+inline_projection_depth);
      suspension_base=suspended_projections.size();suspension_frames=inline_projection_depth;
      suspending=true;
      throw request;
    }
    if(!visiting.insert(request.key))fail("scalar_fold_cycle","cyclic value or read substitution; requested="+
        keys.text(request.key)+"; parent="+(active_projection?keys.text(active_projection->key):"none"));
    struct InlineFrame {
      Lower& owner;
      Projection& request;
      std::vector<std::uint64_t> saved_domains;
      std::optional<MemoryInstance> saved_current;
      bool completed=false;
      ~InlineFrame() {
        --owner.projection_depth;--owner.inline_projection_depth;
        owner.visiting.erase(request.key);owner.domains=std::move(saved_domains);
        if(completed)owner.current=saved_current;
        // After this frame's cleanup and before request's own destructor;
        // unwinding records the inline frames innermost first.
        else if(owner.suspending)owner.suspended_projections.push_back(std::move(request));
      }
    } frame{*this,request,domains,current};
    ++projection_depth;++inline_projection_depth;
    // A dependency suspension is handled once by the outer worklist. Do
    // not catch/rethrow it at every inline ancestor: that restarts exception
    // search repeatedly along the same path. Cleanup alone restores the
    // caller's domain, while failures retain their diagnostic location.
    auto r=evaluateProjection(request);
    frame.completed=true;
    return r;
  }
  const auto outer_domains=domains;
  projecting=true;
  std::vector<Projection> work;
  std::optional<std::pair<ProjectionKey,std::size_t>> pressure_retry;
  work.push_back(std::move(request));
  visiting.insert(key);
  try {
    while(!work.empty()) {
      auto& task=work.back();
      SENTRYWARD_AUDIT(if(task.audit_attempts++) {
        audit->add(AuditCounter::parent_restarts);
        audit->event("DEPENDENCY_REWALK","parent_restarts_after_dependency",
          task.audit_attempts,task.dependencies.size(),task.value.nodeId());
      });
      requested=&task.dependencies;
      domains=task.domains;
      projection_depth=work.size();
      try {
        evaluateProjection(task);
        pressure_retry.reset();
        visiting.erase(task.key);
        work.pop_back();
        requested=nullptr;
        if(options.construction_collection_interval && result.graph.size()>=next_collection)
          collectFrontier(work,key);
      } catch(const CollectProjectionFrontier&) {
        SENTRYWARD_LEDGER(ledger->add("node_pressure_retries"));
        requested=nullptr;
        collectFrontier(work,key);
        const auto live=result.graph.size();
        // A task that cannot advance after reclaiming the same frontier
        // cannot be retried indefinitely under this construction budget.
        if(live>=options.max_nodes || (pressure_retry &&
            pressure_retry->first==task.key && live>=pressure_retry->second))
          fail("expression_node_limit_exceeded","scalar graph node limit exceeded");
        pressure_retry=std::pair{task.key,live};
      } catch(const RetryFoldDomain& retry) {
        SENTRYWARD_LEDGER(ledger->add("fold_domain_retries"));
        // Only the current, uncommitted Fold piece changed. Accumulators
        // retain completed pieces; cached completed projections remain valid.
        KeySet keep;
        for(const auto& kept:retry.keep)keep.insert(kept);
        for(auto p=reduction_progress.begin();p!=reduction_progress.end();) {
          if(keep.contains(p->first)){p->second.pending.clear();++p;continue;}
          const auto abandoned=p->first;++p;finishReduction(abandoned);
        }
        if(!refinement_resume)throw std::logic_error("missing Fold continuation");
        auto resume=std::move(*refinement_resume);refinement_resume.reset();
        // Retain the enclosing explicit tasks. An inline owner gets its own
        // task here, rather than re-entering the entire output expression
        // for every discovered source boundary.
        std::size_t prefix=1;
        for(std::size_t i=0;i<work.size();++i)
          if(keep.contains(work[i].key))prefix=i+1;
        work.erase(work.begin()+prefix,work.end());visiting.clear();
        for(auto& task:work)visiting.insert(task.key);
        if(work.back().key!=resume.key) {
          work.back().dependencies.insert(resume.key);
          visiting.insert(resume.key);work.push_back(std::move(resume));
        }
        requested=nullptr;
      } catch(Projection& child) {
        suspending=false;
        // Every inline frame unwound by this suspension recorded itself.
        if(suspended_projections.size()-suspension_base!=suspension_frames)
          throw std::logic_error("suspended inline projection frames were not recorded");
        requested=nullptr;
        const auto enqueue=[&](Projection&& next) {
          work.back().dependencies.insert(next.key);
          if(!visiting.insert(next.key))fail("scalar_fold_cycle","cyclic value or read substitution");
          if(work.size()>=options.max_pending_projections)
            fail("projection_worklist_limit_exceeded","projection worklist budget exceeded");
          work.push_back(std::move(next));
        };
        // Parent-before-child stack order. Each suspended frame is retried
        // only after its direct child completes; it no longer reconstructs
        // every discarded inline ancestor. These records contain no scalar
        // graph refs, and collectFrontier roots/remaps their cached results.
        for(auto i=suspended_projections.rbegin();i!=suspended_projections.rend();++i)
          enqueue(std::move(*i));
        suspended_projections.clear();
        enqueue(std::move(child));
      }
    }
    domains=outer_domains;
    requested=nullptr;
    projecting=false;
    discardReductions();
    return keys.at(key);
  } catch(...) {
    suspending=false;
    suspended_projections.clear();
    // Output-domain refinement intentionally unwinds this driver and later
    // retries another domain. The explicit work frames are being destroyed,
    // so none of their keys remains an active ancestor. Inline frame keys
    // have already been removed by InlineFrame's destructor. Remove only
    // the keys owned by this driver; ordinary in-driver cycle checks stay.
    for(const auto& task:work)visiting.erase(task.key);
    domains=outer_domains;
    requested=nullptr;
    projecting=false;
    discardReductions();
    throw;
  }
}

} // namespace
} // namespace sentryward::value_analysis
