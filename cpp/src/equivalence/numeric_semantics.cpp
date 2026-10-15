#include "sentryward/equivalence/numeric_semantics.h"

namespace sentryward::equivalence {
bool idealScalarOperation(std::string_view operation,std::size_t arity) {
  return ((operation=="Add" || operation=="Multiply" || operation=="Maximum" ||
           operation=="Minimum" || operation=="Subtract" || operation=="Divide" ||
           operation=="Power") && arity==2) ||
      (operation=="Fma" && arity==3) ||
      ((operation=="Exp" || operation=="Exp2" || operation=="Sqrt" || operation=="Rsqrt" ||
        operation=="ReciprocalSqrt" || operation=="Negate" || operation=="Absolute" || operation=="Reciprocal" ||
        operation=="Sin" || operation=="Cos" || operation=="Log" || operation=="Log2") && arity==1);
}
std::optional<std::string_view> legacyIdealPrimitive(
    std::string_view operation,std::string_view type,
    std::string_view attributes,std::string_view operand_type) {
  std::string_view name;
  if(operation=="PtxReciprocal")name="Reciprocal";
  else if(operation=="PtxSqrt")name="Sqrt";
  else if(operation=="PtxReciprocalSqrt")name="ReciprocalSqrt";
  else if(operation=="PtxSin")name="Sin";
  else if(operation=="PtxCos")name="Cos";
  else if(operation=="PtxLog2")name="Log2";
  else return {};
  if((type!="f32" && type!="f64") || operand_type!=type)return {};
  const auto separator=attributes.find(' ');
  if(separator==std::string_view::npos || !attributes.starts_with("source="))return {};
  const auto tail=attributes.substr(separator+1);
  if(!tail.starts_with("source_type=") || tail.substr(12)!=type)return {};
  auto source=attributes.substr(7,separator-7);const auto first=source.find('.');
  if(source.substr(0,first)!=type)return {};
  if(operation=="PtxSin" || operation=="PtxCos" || operation=="PtxLog2") {
    if(type!="f32" || (source!="f32.approx" && source!="f32.approx.f32" &&
        source!="f32.approx.ftz" && source!="f32.approx.ftz.f32"))return {};
    return name;
  }
  bool mode=false,ftz=false,dtype=false;
  while(first!=std::string_view::npos && source.find('.')!=std::string_view::npos) {
    source.remove_prefix(source.find('.')+1);
    const auto end=source.find('.');const auto modifier=source.substr(0,end);
    if(modifier=="approx" || modifier=="rn" || modifier=="rz" || modifier=="rm" || modifier=="rp") {
      if(mode || (modifier=="approx" && type!="f32"))return {};
      mode=true;
    } else if(modifier=="ftz") {
      if(ftz || type!="f32")return {};ftz=true;
    } else if(modifier==type) {
      if(dtype)return {};dtype=true;
    } else return {};
    if(end==std::string_view::npos)break;
  }
  return name;
}

bool finiteIntegerFloatConversion(std::string_view source,
    std::string_view target,std::string_view conversion) {
  const auto integer=[](std::string_view type) {
    return type=="s8" || type=="u8" || type=="s16" || type=="u16" ||
        type=="s32" || type=="u32" || type=="s64" || type=="u64";
  };
  const auto width=[&](std::string_view type)->unsigned {
    if(!integer(type) && type!="b8" && type!="b16" && type!="b32" && type!="b64")return 0;
    return type.substr(1)=="8"?8:type.substr(1)=="16"?16:type.substr(1)=="32"?32:64;
  };
  if((target!="f32" && target!="f64") || !width(source) || conversion.empty())return false;
  auto current=source;
  for(std::size_t count=0;count<128;++count) {
    const auto end=conversion.find("; ");
    const auto step=conversion.substr(0,end);
    const auto begin=step.find('('),arrow=step.find(" -> ");
    if(begin==std::string_view::npos || arrow==std::string_view::npos ||
        arrow<=begin || step.empty() || step.back()!=')')return false;
    auto operation=step.substr(0,begin);
    while(!operation.empty() && operation.back()==' ')operation.remove_suffix(1);
    const auto from=step.substr(begin+1,arrow-begin-1);
    const auto to=step.substr(arrow+4,step.size()-arrow-5);
    if(from!=current)return false;
    if(end==std::string_view::npos) {
      return integer(from) && to==target &&
          (operation=="Convert" || operation=="Convert rounding=nearest-even" ||
           operation=="Convert rounding=zero" || operation=="Convert rounding=negative-infinity" ||
           operation=="Convert rounding=positive-infinity");
    }
    if(operation!="Bitcast" || !width(from) || width(from)!=width(to))return false;
    current=to;conversion.remove_prefix(end+2);
    if(conversion.empty())return false;
  }
  return false;
}
}
