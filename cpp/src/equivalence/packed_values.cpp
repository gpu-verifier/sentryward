#include "sentryward/equivalence/packed_values.h"
#include "sentryward/equivalence/constant_values.h"
#include "sentryward/equivalence/conversion_uses.h"
#include "sentryward/equivalence/egraph.h"
#include "sentryward/equivalence/representation_rewriting.h"
#include <algorithm>
#include <array>
#include <charconv>
#include <set>
#include <string_view>

namespace sentryward::equivalence {
namespace {
namespace s=scalar_fold;
using Origin=SymbolicBitOrigin;
std::optional<std::uint64_t> integer(const s::Graph& g,s::Ref r) {
  const auto& n=g.node(r);
  if(n.kind!=s::Kind::Constant || !r.pointer_provenance.empty() ||
      r.conversion.size()>128 || n.name.size()>128)return {};
  auto normalized=normalizeConversions(n.type,g.type(r),r.conversion);if(!normalized)return {};
  auto text=*normalized;
  while(!text.empty()) {
    const auto end=text.find("; "),begin=text.find('(');if(begin==std::string::npos)return {};
    const auto op=text.substr(0,begin);
    if(op!="Bitcast" && op!="Convert" && op!="NumericInterpretation")return {};
    if(end==std::string::npos)break;text.erase(0,end+2);
  }
  auto v=constantValue(n.type,n.name,r.conversion,r.converted_type);if(!v)return {};
  std::uint64_t x=0;auto p=std::from_chars(v->literal.data(),v->literal.data()+v->literal.size(),x);
  if(p.ec!=std::errc{} || p.ptr!=v->literal.data()+v->literal.size())return {};return x;
}
bool supported(const s::Ref& ref) {
  return ref.pointer_provenance.empty() && ref.conversion.size()<=128 && ref.converted_type.size()<=128;
}
bool rawStorageType(const std::string& type) {
  const auto bits=bitWidth(type);
  return bits && !type.empty() && std::string("bsu").find(type.front())!=std::string::npos &&
      type==std::string(1,type.front())+std::to_string(bits);
}
bool unsignedStorageType(const std::string& type) {
  return rawStorageType(type) && type.front()=='u';
}
struct LocalMemoryField {
  s::Ref source;
  unsigned bit_offset=0;
  std::string type,remainder;
};
std::optional<LocalMemoryField> localFloatingMemoryField(const s::Graph& graph,const s::Ref& ref) {
  if(!supported(ref) || ref.conversion.empty())return {};
  struct Cast {unsigned source,target;};
  unsigned work=0,low=0,bits=0;std::string type,remainder;
  // The terminal floating bitcast is a typed use boundary, not a raw spine
  // step. Only a fixed array of <=4 raw cast records can precede it.
  const auto casts=[&](const s::Ref& value,bool terminal)->bool {
    if(!supported(value))return false;
    const auto& source_type=graph.node(value).type;
    if(!rawStorageType(source_type))return false;
    auto current=source_type,text=value.conversion;
    std::array<Cast,4> records{};unsigned count=0;bool found=false;
    while(!text.empty()) {
      auto step=leadingConversion(current,text);if(!step)return false;
      const auto from=bitWidth(current),to=bitWidth(step->target);
      if(terminal && step->operation=="Bitcast" &&
          exactFloatingWidening(step->target,step->target)) {
        if(!to || from!=to || (step->remainder.empty() && graph.type(value)!=step->target))return false;
        bits=to;type=std::move(step->target);remainder=std::move(step->remainder);found=true;break;
      }
      if(work==4 || !rawStorageType(step->target) ||
          !((step->operation=="Bitcast" && from==to) ||
            (step->operation=="Convert" && unsignedStorageType(current) &&
             unsignedStorageType(step->target) && to<=from)))return false;
      records[count++]={from,to};++work;current=std::move(step->target);text=std::move(step->remainder);
    }
    if(terminal?!found:current!=graph.type(value))return false;
    while(count) {
      const auto step=records[--count];
      if(low+bits>step.target || low+bits>step.source)return false;
    }
    return true;
  };
  if(!casts(ref,true))return {};
  auto value=graph.ref(ref.node);
  // At most four raw steps plus the final cell are inspected. A selected
  // interval may never include shift zero-fill or discarded/nonnative bits.
  for(unsigned visited=0;visited<5;++visited) {
    if(!casts(value,false))return {};
    const auto& n=graph.node(value);const auto width=bitWidth(n.type);
    if(!rawStorageType(n.type) || low+bits>width || !n.attributes.empty() || n.fold)return {};
    if(n.kind==s::Kind::InitialCell) {
      if(low%8 || n.name.empty() || n.name.size()>4096 || n.operands.size()!=1)return {};
      const auto& address=n.operands[0];
      if(!supported(address) || graph.type(address)!="u64")return {};
      return LocalMemoryField{graph.ref(value.node),low,std::move(type),std::move(remainder)};
    }
    if(work==4 || n.kind!=s::Kind::Operation || n.name!="ShiftRight" ||
        !unsignedStorageType(n.type) || n.operands.size()!=2 ||
        graph.type(n.operands[0])!=n.type || !supported(n.operands[0]))return {};
    const auto& amount=n.operands[1];const auto& literal=graph.node(amount);
    if(!supported(amount) || !amount.conversion.empty() || literal.kind!=s::Kind::Constant ||
        !literal.attributes.empty() || literal.fold || !literal.operands.empty() ||
        !unsignedStorageType(literal.type))return {};
    const auto shift=integer(graph,amount);
    if(!shift || *shift>=width || low+bits>width-*shift)return {};
    low+=unsigned(*shift);++work;value=n.operands[0];
  }
  return {};
}
s::Ref tail(s::Ref result,const s::Ref& ref) {
  if(!ref.conversion.empty()) {
    if(!result.conversion.empty())result.conversion+="; ";
    result.conversion+=ref.conversion;result.converted_type=ref.converted_type;
  }
  return result;
}
bool shifted(const SymbolicBitOffset& a,const SymbolicBitOffset& b,std::uint64_t amount) {
  return a.index==b.index && a.coefficient==b.coefficient &&
      b.constant>=a.constant && b.constant-a.constant==amount;
}
s::Ref byteOffset(s::Graph& graph,s::Ref base,s::Ref delta) {
  // The input mapper emits constant-first u64 affine addresses. Selecting a
  // packed subfield must preserve that template, rather than append another
  // Add whose shape depends on which original word contained the field.
  // Only one already-stored header is inspected. All arithmetic is modulo
  // 2^64, exactly like the original byte offset (not a no-wrap inference).
  const auto d=integer(graph,delta);
  if(!d || !delta.conversion.empty() || !delta.pointer_provenance.empty() ||
      graph.type(delta)!="u64")return graph.operation("Add","u64",{base,delta});
  if(!*d)return base;
  const auto node=graph.node(base);
  if(base.conversion.empty() && base.pointer_provenance.empty() && node.type=="u64" &&
      node.attributes.empty()) {
    if(auto b=integer(graph,base))return graph.constant("u64",std::to_string(*b+*d));
    if(node.kind==s::Kind::Operation && node.name=="Add" && node.operands.size()==2)
      for(unsigned side=0;side<2;++side) {
        const auto& value=node.operands[side];
        if(!value.conversion.empty() || !value.pointer_provenance.empty() || graph.type(value)!="u64")continue;
        if(auto b=integer(graph,value)) {
          auto constant=graph.constant("u64",std::to_string(*b+*d));
          return graph.operation("Add","u64",{constant,node.operands[1-side]});
        }
      }
  }
  return graph.operation("Add","u64",{base,delta});
}
std::optional<s::Ref> memoryOffset(s::Graph& graph,const BitSource& source,
    const SymbolicBitOffset& bit,BitOriginAnalysis& analysis) {
  const auto& cell=graph.node(source.value);
  if(source.kind!=BitSource::Kind::Memory || cell.kind!=s::Kind::InitialCell ||
      !source.value.conversion.empty() || !source.value.pointer_provenance.empty() || cell.operands.size()!=1 ||
      graph.type(cell.operands[0])!="u64" || !cell.operands[0].pointer_provenance.empty() ||
      cell.operands[0].conversion.size()>128 || cell.operands[0].converted_type.size()>128)return {};
  const auto original_offset=cell.operands[0];
  auto delta=analysis.emit(graph,bit,8);if(!delta)return {};
  if(!bit.index && !bit.constant)return original_offset;
  // Memory offsets use the same u64 modulo-address arithmetic as the source.
  // The selected field was proved inside that fixed-width source read.
  return byteOffset(graph,original_offset,*delta);
}
}
scalar_fold::Ref rewriteLocalFloatingMemoryRead(scalar_fold::Graph& graph,scalar_fold::Ref ref) {
  const auto field=localFloatingMemoryField(graph,ref);if(!field)return ref;
  // Copy the bounded source fields before any graph append. u64 byte-offset
  // arithmetic is the same modulo address arithmetic as recoverBitField.
  auto allocation=graph.node(field->source).name;
  auto address=graph.node(field->source).operands[0];
  if(field->bit_offset) {
    auto delta=graph.constant("u64",std::to_string(field->bit_offset/8));
    address=byteOffset(graph,address,delta);
  }
  auto result=graph.initialCell(std::move(allocation),std::move(address),field->type);
  if(!field->remainder.empty()) {
    result.conversion=field->remainder;result.converted_type=ref.converted_type;
  }
  return result;
}
std::optional<scalar_fold::Ref> recoverBitField(scalar_fold::Graph& graph,
    const SymbolicBitOrigins& bits,const std::string& type,BitOriginAnalysis& analysis) {
  if(bits.empty() || bits.size()!=bitWidth(type) || bits.size()>64)return {};
  if(!analysis.chargeWork(bits.size()))return {};
  const auto first=bits.front();
  if(first.kind!=Origin::Kind::Source || first.inverted)return {};
  for(unsigned i=0;i<bits.size();++i)if(bits[i].kind!=Origin::Kind::Source || bits[i].source!=first.source ||
      bits[i].inverted || !shifted(first.offset,bits[i].offset,i))return {};
  const auto* record=analysis.source(first.source);if(!record)return {};
  const auto source=*record;
  auto range=analysis.bounds(first.offset);
  if(!range || range->second>source.width || bits.size()>source.width-range->second)return {};
  if(source.kind==BitSource::Kind::OpaqueValue) {
    if(source.width!=bits.size() || first.offset.index || first.offset.constant ||
        graph.type(source.value)!=type)return {};
    return source.value;
  }
  auto offset=memoryOffset(graph,source,first.offset,analysis);if(!offset)return {};
  return graph.initialCell(graph.node(source.value).name,*offset,type);
}
bool contiguousReadCandidate(const scalar_fold::Graph& graph,const scalar_fold::Ref& ref) {
  if(!supported(ref))return false;
  const auto& n=graph.node(ref);
  if(!bitWidth(n.type) || !bitWidth(graph.type(ref)))return false;
  const bool field=n.kind==s::Kind::Operation && n.name=="ExtractBits";
  if(field)return true;
  // A complete unchanged literal/index/parameter/cell is either constant bits
  // or its own source. Recovering that full native field can only return self.
  if(ref.conversion.empty() && n.kind!=s::Kind::Operation)return false;
  if(!exactFloatingWidening(n.type,n.type))return true; // Preserve integer packing.
  // Ordinary floating producers and pure floating numeric conversions are
  // opaque values, never byte-field aliases. They matter when a later packed
  // consumer traces their ORIGINAL converted value, not as top-level uses.
  // Retain every explicit floating Bitcast field, including memory subfields.
  for(std::size_t position=0;position<ref.conversion.size();) {
    const auto end=ref.conversion.find("; ",position);
    const auto length=end==std::string::npos?ref.conversion.size()-position:end-position;
    const std::string_view step(ref.conversion.data()+position,length);
    const auto arrow=step.find(" -> ");
    if(step.starts_with("Bitcast(") && arrow!=std::string_view::npos && step.back()==')') {
      const auto target=step.substr(arrow+4,step.size()-arrow-5);
      if(target=="e4m3" || target=="e5m2" || target=="f16" || target=="bf16" ||
          target=="f32" || target=="f64")return true;
    }
    if(end==std::string::npos)break;position=end+2;
  }
  return false;
}
bool packedExtractCandidate(const scalar_fold::Graph& graph,const scalar_fold::Ref& ref) {
  if(!supported(ref))return false;
  const auto& n=graph.node(ref);
  return n.kind==s::Kind::Operation && n.name=="ExtractBits" && n.operands.size()==2 &&
      n.attributes=="width=16" && (n.type=="bf16" || n.type=="f16");
}
bool signedNibbleCandidate(const scalar_fold::Graph& graph,const scalar_fold::Ref& ref) {
  if(!supported(ref))return false;
  const auto& n=graph.node(ref);
  return n.kind==s::Kind::Operation && (n.name=="Subtract" || n.name=="Add") && n.type=="bf16" &&
      (n.attributes.empty() || n.attributes=="ideal_arithmetic") && n.operands.size()==2;
}
scalar_fold::Ref rewriteContiguousRead(scalar_fold::Graph& graph,scalar_fold::Ref ref,
    BitOriginAnalysis& analysis,const BitDomains& domains) {
  if(!contiguousReadCandidate(graph,ref))return ref;
  // Preserve all conversions after the first exact floating bitcast.
  // Probe that field before the complete use. A later numeric conversion is
  // deliberately opaque: recovering its own canonical Ref must not mask the
  // earlier exact field and prevent this typed memory/packed-value rewrite.
  auto text=ref.conversion;std::string prefix;
  while(!text.empty()) {
    const auto end=text.find("; ");const auto step=text.substr(0,end);
    if(!prefix.empty())prefix+="; ";prefix+=step;
    const auto arrow=step.find(" -> ");if(arrow==std::string::npos)break;
    const auto target=step.substr(arrow+4,step.size()-arrow-5);
    if(step.starts_with("Bitcast(") && (target=="e4m3" || target=="e5m2" ||
        target=="f16" || target=="bf16" || target=="f32" || target=="f64")) {
      auto field=ref;field.conversion=prefix;field.converted_type=target;
      if(auto bits=analysis.trace(field,domains))if(auto result=recoverBitField(graph,*bits,target,analysis)) {
        if(end!=std::string::npos) {
          if(!result->conversion.empty())result->conversion+="; ";
          result->conversion+=text.substr(end+2);result->converted_type=ref.converted_type;
        }
        return *result;
      }
      break;
    }
    if(end==std::string::npos)break;text.erase(0,end+2);
  }
  if(auto bits=analysis.trace(ref,domains))if(auto field=recoverBitField(graph,*bits,graph.type(ref),analysis);field && *field!=ref) {
    // A numeric opaque source may be this exact use with canonical conversion
    // spelling. That is not field recovery; retain the caller's unchanged Ref.
    // Both metadata parsers are bounded by supported(ref)'s 128-byte cap.
    if(field->node==ref.node && field->pointer_provenance==ref.pointer_provenance &&
        graph.type(*field)==graph.type(ref)) {
      auto before=normalizeConversions(graph.node(ref).type,graph.type(ref),ref.conversion);
      auto after=normalizeConversions(graph.node(*field).type,graph.type(*field),field->conversion);
      if(before && after && *before==*after)return ref;
    }
    return *field;
  }
  return ref;
}
scalar_fold::Ref rewritePackedExtract(scalar_fold::Graph& graph,scalar_fold::Ref ref,
    BitOriginAnalysis& analysis,const BitDomains& domains) {
  if(!packedExtractCandidate(graph,ref))return ref;
  const auto& node=graph.node(ref);
  const auto n=node; // Only the certified two-operand record is copied.
  const auto type=graph.type(n.operands[0]);if(type!="b32" && type!="u32")return ref;
  if(auto amount=integer(graph,n.operands[1]);amount && (*amount==0 || *amount==16)) {
    auto bits=n.operands[0];if(type=="b32")bits=graph.convert(bits,"u32","Bitcast");
    bits=normalizeExactUse(graph,bits);
    if(*amount)bits=graph.operation("ShiftRight","u32",{bits,graph.constant("u32","16")});
    bits=graph.convert(graph.convert(bits,"u16","Convert"),n.type,"Bitcast");
    auto result=rewriteContiguousRead(graph,normalizeExactUse(graph,bits),analysis,domains);
    return tail(result,ref);
  }
  const auto selector=analysis.integer(n.operands[1],domains);
  // The proof describes a discrete {0,16} selector, not merely an interval.
  if(!selector || !selector->index || selector->coefficient!=16 || selector->constant ||
      selector->lower!=0 || selector->upper!=16)return ref;
  auto packed=analysis.trace(n.operands[0],domains);if(!packed || packed->size()!=32)return ref;
  auto low=recoverBitField(graph,SymbolicBitOrigins(packed->begin(),packed->begin()+16),n.type,analysis);
  auto high=recoverBitField(graph,SymbolicBitOrigins(packed->begin()+16,packed->end()),n.type,analysis);
  if(!low || !high)return ref;
  auto zero=graph.constant(graph.type(n.operands[1]),"0");
  if(graph.type(zero).starts_with("b"))return ref;
  auto predicate=graph.operation("Equal","pred",{n.operands[1],zero});
  // This fixed new predicate is not an original stored use visited by the
  // input adapter. Reuse the exact registry/caps for this ONE generated head,
  // not a child walker. Its pred type cannot reenter packed-field matching.
  predicate=rewriteRepresentation(graph,std::move(predicate),analysis,domains);
  return tail(graph.operation("Select",n.type,{predicate,*low,*high}),ref);
}
scalar_fold::Ref rewriteSignedNibble(scalar_fold::Graph& graph,scalar_fold::Ref ref,
    BitOriginAnalysis& analysis,const BitDomains& domains) {
  if(!signedNibbleCandidate(graph,ref))return ref;
  const auto& node=graph.node(ref);
  const auto n=node; // Keep only this bounded record across graph appends.
  if(graph.type(n.operands[0])!="bf16" || graph.type(n.operands[1])!="bf16")return ref;
  // Lowering may retain subtraction or express it as addition of a negated
  // constant. Accept that bounded local form, without following arbitrary
  // arithmetic ancestors or changing the bit-level certificate below.
  auto bias_ref=n.operands[1];
  if(n.name=="Add") {
    if(!bias_ref.conversion.empty() || !bias_ref.pointer_provenance.empty())return ref;
    const auto& negative=graph.node(bias_ref);
    if(negative.kind!=s::Kind::Operation || negative.name!="Negate" || negative.type!="bf16" ||
        negative.operands.size()!=1 ||
        (!negative.attributes.empty() && negative.attributes!="ideal_arithmetic") ||
        graph.type(negative.operands[0])!="bf16")return ref;
    bias_ref=negative.operands[0];
  }
  auto bias=integer(graph,bias_ref);if(!bias || *bias!=136)return ref;
  const auto bits=analysis.trace(n.operands[0],domains);if(!bits || bits->size()!=16)return ref;
  std::optional<std::size_t> source;std::optional<SymbolicBitOffset> first;
  std::set<std::uint64_t> variables;
  for(const auto& bit:*bits) {
    if(bit.kind==Origin::Kind::Unknown)return ref;
    if(bit.kind!=Origin::Kind::Source)continue;
    if(source && *source!=bit.source)return ref;source=bit.source;
    if(first && (first->index!=bit.offset.index || first->coefficient!=bit.offset.coefficient))return ref;
    if(!first || bit.offset.constant<first->constant)first=bit.offset;
    variables.insert(bit.offset.constant);
  }
  if(!source || !first || variables.size()!=4 || first->constant%4 ||
      *variables.rbegin()-first->constant!=3 || first->coefficient%8)return ref;
  const auto* record=analysis.source(*source);
  if(!record || record->kind!=BitSource::Kind::Memory)return ref;
  const auto origin=*record;
  const auto range=analysis.bounds(*first);
  if(!range || range->second>origin.width || 4>origin.width-range->second ||
      !analysis.chargeWork(16*16))return ref;
  // Same 16-value certificate as the literal case: symbolic location does
  // not change the four-bit truth table, rounding or BF16 representability.
  for(unsigned value=0;value<16;++value) {
    unsigned raw=0;
    for(unsigned i=0;i<16;++i) {
      const auto& bit=(*bits)[i];
      const bool v=bit.kind==Origin::Kind::Source?bool((value>>(bit.offset.constant-first->constant))&1):false;
      raw|=unsigned(v^bit.inverted)<<i;
    }
    auto decoded=constantValue("b16",std::to_string(raw),"Bitcast(b16 -> bf16)","bf16");
    if(!decoded || std::stod(decoded->literal)-136!=(value<8?int(value):int(value)-16))return ref;
  }
  auto byte=*first;byte.constant-=first->constant%8;
  auto offset=memoryOffset(graph,origin,byte,analysis);if(!offset)return ref;
  auto raw=graph.convert(graph.initialCell(graph.node(origin.value).name,*offset,"u8"),"s32","Convert");
  if(first->constant%8)raw=graph.operation("ShiftRight","s32",{raw,graph.constant("s32","4")});
  auto nibble=graph.operation("BitAnd","s32",{raw,graph.constant("s32","15")});
  auto signed_value=graph.operation("Subtract","s32",{
      graph.operation("BitXor","s32",{nibble,graph.constant("s32","8")}),graph.constant("s32","8")});
  // The exhaustive nibble proof above establishes [-8,7]. Every value is
  // exactly representable in f32 and bf16, so choosing an explicit rounding
  // mode loses nothing and avoids an underspecified integer conversion use.
  auto result=graph.convert(graph.convert(signed_value,"f32","Convert rounding=nearest-even"),"bf16","Convert");
  return tail(result,ref);
}
scalar_fold::Ref normalizePackedUse(scalar_fold::Graph& graph,scalar_fold::Ref ref,
    BitOriginAnalysis& analysis,const BitDomains& domains) {
  return rewriteSignedNibble(graph,rewritePackedExtract(graph,rewriteContiguousRead(graph,ref,analysis,domains),analysis,domains),analysis,domains);
}
scalar_fold::Ref rewriteContiguousRead(scalar_fold::Graph& graph,scalar_fold::Ref ref) {
  BitOriginAnalysis analysis(graph);return rewriteContiguousRead(graph,ref,analysis,{});
}
scalar_fold::Ref rewritePackedExtract(scalar_fold::Graph& graph,scalar_fold::Ref ref) {
  BitOriginAnalysis analysis(graph);return rewritePackedExtract(graph,ref,analysis,{});
}
scalar_fold::Ref rewriteSignedNibble(scalar_fold::Graph& graph,scalar_fold::Ref ref) {
  BitOriginAnalysis analysis(graph);return rewriteSignedNibble(graph,ref,analysis,{});
}
scalar_fold::Ref normalizePackedUse(scalar_fold::Graph& graph,scalar_fold::Ref ref) {
  BitOriginAnalysis analysis(graph);return normalizePackedUse(graph,ref,analysis,{});
}
}
