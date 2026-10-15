#include "sentryward/equivalence/conversion_uses.h"
#include "sentryward/equivalence/float_from_chars.h"
#include <charconv>
#include <cmath>
#include <vector>

namespace sentryward::equivalence {
namespace {
struct Step { std::string operation, source, target; };
unsigned width(const std::string& type) {
  if(type=="bf16")return 16;
  if(type=="e2m1")return 4;
  if(type=="ue8m0" || type=="ue4m3")return 8;
  if(type=="e4m3" || type=="e5m2")return 8;
  if(type.empty() || std::string("bsuf").find(type[0])==std::string::npos)return 0;
  unsigned bits=0;auto result=std::from_chars(type.data()+1,type.data()+type.size(),bits);
  return result.ec==std::errc{} && result.ptr==type.data()+type.size() &&
      (bits==4 || bits==8 || bits==16 || bits==32 || bits==64)?bits:0;
}
bool bitcast(const Step& step) {
  return step.operation=="Bitcast" && width(step.source) &&
      width(step.source)==width(step.target);
}
bool unsignedConvert(const Step& step) {
  return step.operation=="Convert" && step.source.starts_with("u") &&
      step.target.starts_with("u") && width(step.source) && width(step.target);
}
}
bool exactFloatingWidening(const std::string& source,const std::string& target) {
  struct Format { int fraction, min_subnormal, max_exponent; };
  const auto format=[](const std::string& type)->std::optional<Format> {
    // CUDA fp8 finite formats: cuda_fp8.hpp, __nv_cvt_double_to_fp8.
    if(type=="e4m3")return Format{3,-9,8};
    if(type=="e2m1")return Format{1,-1,2};
    if(type=="ue8m0")return Format{0,-127,127};
    if(type=="ue4m3")return Format{3,-9,8};
    if(type=="e5m2")return Format{2,-16,15};
    if(type=="f16")return Format{10,-24,15};
    if(type=="bf16")return Format{7,-133,127};
    if(type=="f32")return Format{23,-149,127};
    if(type=="f64")return Format{52,-1074,1023};
    return {};
  };
  auto a=format(source),b=format(target);
  return a && b && b->fraction>=a->fraction &&
      b->min_subnormal<=a->min_subnormal && b->max_exponent>=a->max_exponent;
}
integer_interval::Result integerConversionRange(
    const std::string& source,const std::string& target,
    const std::string& conversion,const integer_interval::Interval& value) {
  namespace i=integer_interval;
  const auto from=i::parseType(source),to=i::parseType(target);
  if(!from || !to || value.type!=*from ||
      !i::interval(*from,value.lower,value.upper) || conversion.size()>128)return {};
  const auto edge="("+source+" -> "+target+")";
  const bool numeric=conversion=="Convert"+edge;
  const bool bit_preserving=conversion=="Bitcast"+edge && from->bits==to->bits;
  const bool zero_extend=conversion=="ZeroExtend"+edge && !from->is_signed &&
      to->bits>=from->bits;
  if(!numeric && !bit_preserving && !zero_extend)return {};
  // This excludes negative signed→unsigned reinterprets, unsigned values above
  // the signed maximum and every possible narrowing/wrap. It also works for
  // a non-singleton interval without replacing the conversion itself.
  return i::interval(*to,value.lower,value.upper);
}
std::optional<ConstantValue> exactIntegerFloatingValue(
    const std::string& source,const std::string& target,
    const std::string& conversion,std::int64_t value) {
  if(!integer_interval::literal(source,std::to_string(value)) ||
      (target!="f32" && target!="f64") || conversion.size()>128)return {};
  const auto edge="("+source+" -> "+target+")";
  if(conversion!="Convert"+edge &&
      conversion!="Convert rounding=nearest-even"+edge)return {};
  // constantValue's encode succeeds only for exact finite representation;
  // it does not silently round an inexact integer. In particular 2^24+1 is
  // refused for f32, and all overflow/FTZ/saturation paths remain explicit.
  return constantValue(source,std::to_string(value),conversion,target);
}
std::optional<FloatingClampRange> exactFloatingClampRange(
    const scalar_fold::Graph& graph,const scalar_fold::Ref& ref) {
  namespace s=scalar_fold;
  const auto plain=[](const s::Ref& use) {
    return use.conversion.empty() && use.converted_type.empty() && use.pointer_provenance.empty();
  };
  if(!plain(ref))return {};
  const auto literal=[&](const s::Ref& use)->std::optional<double> {
    if(!plain(use))return {};
    const auto& n=graph.node(use);
    if(n.kind!=s::Kind::Constant || n.type!="f32" || !n.attributes.empty() || n.fold ||
        !n.operands.empty() || n.name.size()>128)return {};
    // Decode the recorded f32, rather than compare decimal source spellings.
    // Every finite f32 is exactly representable as double.
    auto value=constantValue(n.type,n.name);if(!value)return {};
    double result=0;
    const auto parsed=floatFromChars(value->literal.data(),value->literal.data()+value->literal.size(),result);
    if(parsed.ec!=std::errc{} || parsed.ptr!=value->literal.data()+value->literal.size() ||
        !std::isfinite(result))return {};
    return result;
  };
  const auto typed_float_use=[&](const s::Ref& use) {
    if(!use.pointer_provenance.empty() || use.conversion.size()>128 ||
        use.converted_type.size()>128)return false;
    auto current=graph.node(use).type,remaining=use.conversion;
    if(remaining.empty())return use.converted_type.empty() && current=="f32";
    while(!remaining.empty()) {
      auto step=leadingConversion(current,remaining);if(!step)return false;
      current=std::move(step->target);remaining=std::move(step->remainder);
    }
    return current=="f32" && use.converted_type==current;
  };
  const auto clamp_head=[](const s::Node& n,const char* operation) {
    return n.kind==s::Kind::Operation && n.type=="f32" && n.name==operation &&
        n.attributes.empty() && !n.fold && n.operands.size()==2;
  };
  const auto& outer=graph.node(ref);
  if(!clamp_head(outer,"Maximum") && !clamp_head(outer,"Minimum"))return {};
  // This exact contract is preserved by frontend/ptx/value_ir.cpp:
  // .NaN is serialized as propagate_nan, and other min/max modifiers are
  // checked explicitly. Default PTX min/max return the non-NaN operand.
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-min
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-max
  for(unsigned side=0;side<2;++side) {
    const auto a=literal(outer.operands[side]);
    const auto& inside=outer.operands[1-side];
    if(!a || !plain(inside))continue;
    const auto& inner=graph.node(inside);
    if(!clamp_head(inner,outer.name=="Maximum"?"Minimum":"Maximum"))continue;
    for(unsigned endpoint=0;endpoint<2;++endpoint) {
      const auto b=literal(inner.operands[endpoint]);
      const auto& value=inner.operands[1-endpoint];
      if(!b || !typed_float_use(value))continue;
      const auto lower=outer.name=="Maximum"?*a:*b;
      const auto upper=outer.name=="Maximum"?*b:*a;
      if(lower<=upper)return FloatingClampRange{lower,upper};
    }
  }
  return {};
}
std::optional<scalar_fold::Ref> normalizeInactiveFiniteSaturation(
    const scalar_fold::Graph& graph,const scalar_fold::Ref& ref) {
  if(!ref.pointer_provenance.empty() || graph.node(ref).type!="f32" ||
      ref.converted_type.empty() || ref.conversion.size()>128 || ref.converted_type.size()>128)return {};
  const auto prefix=leadingConversion("f32",ref.conversion);
  if(!prefix || prefix->operation!="Convert rounding=nearest-even saturate_finite" ||
      prefix->target!="e4m3")return {};
  // Validate the entire bounded type chain; unchanged tail operations need no
  // semantic admission. Unknown modifiers on the FIRST conversion are denied.
  auto current=prefix->target,remaining=prefix->remainder;
  while(!remaining.empty()) {
    auto step=leadingConversion(current,remaining);if(!step)return {};
    current=std::move(step->target);remaining=std::move(step->remainder);
  }
  if(current!=ref.converted_type)return {};
  const auto bounds=exactFloatingClampRange(graph,graph.ref(ref.node));
  if(!bounds || bounds->lower < -448 || bounds->upper > 448)return {};
  // CUDA/PTX e4m3 maximum finite magnitude is exactly 448. For a finite
  // source already inside that closed interval, RN cannot overflow; SAT has
  // no effect on result bits, signed zero, subnormal behavior or rounding.
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt
  auto result=ref;
  result.conversion="Convert rounding=nearest-even(f32 -> e4m3)";
  if(!prefix->remainder.empty())result.conversion+="; "+prefix->remainder;
  return result;
}
std::optional<ConversionPrefix> leadingConversion(
    const std::string& source,const std::string& conversion) {
  if(source.empty() || source.size()>128 || conversion.empty() || conversion.size()>128)return {};
  const auto begin=conversion.find('(');if(begin==std::string::npos)return {};
  auto operation=conversion.substr(0,begin);
  while(!operation.empty() && operation.back()==' ')operation.pop_back();
  const auto prefix="("+source+" -> ";
  if(operation.empty() || conversion.compare(begin,prefix.size(),prefix)!=0)return {};
  const auto end=conversion.find(')',begin+prefix.size());
  if(end==std::string::npos)return {};
  auto target=conversion.substr(begin+prefix.size(),end-begin-prefix.size());
  if(target.empty())return {};
  if(end+1==conversion.size())return ConversionPrefix{std::move(operation),std::move(target),{}};
  if(conversion.substr(end+1,2)!="; " || end+3==conversion.size())return {};
  return ConversionPrefix{std::move(operation),std::move(target),conversion.substr(end+3)};
}
std::optional<BitcastPrefix> leadingBitcast(
    const std::string& source,const std::string& conversion) {
  if(source.size()>128 || conversion.size()>128)return {};
  if(!conversion.starts_with("Bitcast("+source+" -> "))return {};
  auto prefix=leadingConversion(source,conversion);
  if(!prefix || prefix->operation!="Bitcast" || !width(source) || width(source)!=width(prefix->target))return {};
  return BitcastPrefix{std::move(prefix->target),std::move(prefix->remainder)};
}
std::optional<BitcastPrefix> floatingStorageReadPrefix(
    const scalar_fold::Graph& graph,const scalar_fold::Ref& ref) {
  const auto& n=graph.node(ref);
  // bf16 is a floating format, never a b-prefixed bit-vector tag. Require an
  // exact canonical storage spelling rather than just testing its first byte.
  const bool storage=n.type=="b8" || n.type=="b16" || n.type=="b32" || n.type=="b64" ||
      n.type=="u8" || n.type=="u16" || n.type=="u32" || n.type=="u64" ||
      n.type=="s8" || n.type=="s16" || n.type=="s32" || n.type=="s64";
  if(!storage || n.kind!=scalar_fold::Kind::InitialCell || !n.attributes.empty() ||
      n.fold || n.name.empty() || n.name.size()>4096 || n.operands.size()!=1 ||
      !ref.pointer_provenance.empty() || ref.conversion.size()>128 ||
      ref.converted_type.empty() || ref.converted_type.size()>128)return {};
  const auto& address=n.operands[0];
  if(!address.pointer_provenance.empty() || address.conversion.size()>128 ||
      address.converted_type.size()>128 || graph.type(address)!="u64")return {};
  auto prefix=leadingBitcast(n.type,ref.conversion);
  if(!prefix || !exactFloatingWidening(prefix->target,prefix->target) ||
      (prefix->remainder.empty() && (graph.type(ref)!=prefix->target ||
       ref.conversion!="Bitcast("+n.type+" -> "+prefix->target+")")))return {};
  return prefix;
}
bool idealFloatingStorageReadUse(
    const scalar_fold::Graph& graph,const scalar_fold::Ref& ref,bool storage_tail) {
  const auto prefix=floatingStorageReadPrefix(graph,ref);
  return prefix && idealNumericConversion(prefix->target,graph.type(ref),
      prefix->remainder,storage_tail);
}
namespace {
bool idealConversion(const std::string& source,const std::string& target,
                     const std::string& conversion,bool storage_tail,bool output_saturation) {
  const auto floating=[](const std::string& t){return t=="real" || exactFloatingWidening(t,t);};
  if(!floating(source))return false;
  auto current=source;
  for(std::size_t pos=0;pos<conversion.size();) {
    const auto end=conversion.find("; ",pos);
    const auto step=conversion.substr(pos,end==std::string::npos?end:end-pos);
    const auto begin=step.find('('),arrow=step.find(" -> ");
    if(begin==std::string::npos || arrow==std::string::npos || arrow<=begin || step.back()!=')')return false;
    auto op=step.substr(0,begin);while(!op.empty() && op.back()==' ')op.pop_back();
    const auto from=step.substr(begin+1,arrow-begin-1);
    const auto to=step.substr(arrow+4,step.size()-arrow-5);
    if(from!=current)return false;
    if(storage_tail && end==std::string::npos && op=="Bitcast" && floating(from) &&
       !to.empty() && std::string("bsu").find(to[0])!=std::string::npos &&
       width(from) && width(from)==width(to))return to==target;
    if(output_saturation) {
      // Admit only the existing, explicitly spelled floating conversion modes.
      // Unknown modifiers and integer conversions are never numerical aliases.
      for(const auto* suffix:{" saturate_finite"," saturate"})
        if(op.ends_with(suffix)) {op.resize(op.size()-std::string(suffix).size());break;}
    }
    if(!floating(from) || !floating(to) ||
       (op!="Convert" && op!="NumericInterpretation" &&
        op!="Convert rounding=nearest-even" && op!="Convert rounding=zero" &&
        op!="Convert rounding=negative-infinity" && op!="Convert rounding=positive-infinity"))
      return false;
    current=to;
    if(end==std::string::npos)break;
    pos=end+2;
  }
  return current==target;
}
}
bool idealNumericConversion(const std::string& source,const std::string& target,
                            const std::string& conversion,bool storage_tail) {
  return idealConversion(source,target,conversion,storage_tail,false);
}
bool idealNumericOutputConversion(const std::string& source,const std::string& target,
                                 const std::string& conversion) {
  return conversion.size()<=128 && source.size()<=128 && target.size()<=128 &&
      idealConversion(source,target,conversion,true,true);
}
std::optional<std::string> normalizeConversions(
    const std::string& source, const std::string& target, const std::string& conversion) {
  std::vector<Step> steps;
  auto current=source;
  for(std::size_t pos=0;pos<conversion.size();) {
    const auto end=conversion.find("; ",pos);
    const auto text=conversion.substr(pos,end==std::string::npos?end:end-pos);
    const auto begin=text.find('('),arrow=text.find(" -> ");
    if(begin==std::string::npos || arrow==std::string::npos || arrow<=begin ||
        text.back()!=')')return {};
    Step step{text.substr(0,begin),text.substr(begin+1,arrow-begin-1),
        text.substr(arrow+4,text.size()-arrow-5)};
    while(!step.operation.empty() && step.operation.back()==' ')step.operation.pop_back();
    if(step.source!=current || step.operation.empty() || step.target.empty())return {};
    current=step.target;
    steps.push_back(std::move(step));
    bool changed=true;
    while(changed && !steps.empty()) {
      changed=false;
      auto& last=steps.back();
      if((bitcast(last) || unsignedConvert(last)) && last.source==last.target) {
        steps.pop_back();changed=true;continue;
      }
      if(steps.size()<2)break;
      const auto& previous=steps[steps.size()-2];
      if((bitcast(previous) && bitcast(last)) ||
          (unsignedConvert(previous) && unsignedConvert(last) &&
           width(previous.target)>=width(previous.source))) {
        auto merged=last;merged.source=previous.source;
        steps.pop_back();steps.back()=std::move(merged);changed=true;
      }
    }
    if(end==std::string::npos)break;
    pos=end+2;
  }
  if(!target.empty() && current!=target)return {};
  // Canonicalize only a final, purely numeric widening suffix.
  const auto numeric=[](const Step& step) {
    return exactFloatingWidening(step.source,step.target) &&
        (step.operation=="Convert" || step.operation=="NumericInterpretation" ||
         step.operation=="Convert rounding=nearest-even" ||
         step.operation=="Convert rounding=zero" ||
         step.operation=="Convert rounding=negative-infinity" ||
         step.operation=="Convert rounding=positive-infinity");
  };
  if(!steps.empty() && numeric(steps.back())) {
    auto first=steps.size()-1;
    while(first && numeric(steps[first-1]))--first;
    Step merged{"NumericInterpretation",steps[first].source,steps.back().target};
    steps.resize(first);
    if(merged.source!=merged.target)steps.push_back(std::move(merged));
  }
  std::string result;
  for(const auto& step:steps) {
    if(!result.empty())result+="; ";
    result+=step.operation+"("+step.source+" -> "+step.target+")";
  }
  return result;
}
}
