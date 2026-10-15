#include "sentryward/equivalence/constant_values.h"
#include "sentryward/equivalence/float_from_chars.h"
#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <string_view>
#include <vector>

namespace sentryward::equivalence {
namespace {
enum class Encoding { Ieee, E2M1, UE8M0, UE4M3 };
struct Type { unsigned width, fraction=0, exponent=0; bool sign=false; Encoding encoding=Encoding::Ieee; };
std::optional<Type> typeOf(const std::string& type) {
  if(type=="f16")return Type{16,10,5};
  if(type=="bf16")return Type{16,7,8};
  if(type=="f32")return Type{32,23,8};
  if(type=="f64")return Type{64,52,11};
  if(type=="e2m1")return Type{4,1,2,false,Encoding::E2M1};
  if(type=="ue8m0")return Type{8,0,8,false,Encoding::UE8M0};
  if(type=="ue4m3")return Type{8,3,4,false,Encoding::UE4M3};
  if(type.empty() || (type[0]!='b' && type[0]!='u' && type[0]!='s'))return {};
  unsigned width=0;auto p=std::from_chars(type.data()+1,type.data()+type.size(),width);
  if(p.ec!=std::errc{} || p.ptr!=type.data()+type.size() ||
     (width!=4 && width!=8 && width!=16 && width!=32 && width!=64))return {};
  return Type{width,0,0,type[0]=='s'};
}
std::uint64_t mask(unsigned width) {return width==64?UINT64_MAX:(std::uint64_t(1)<<width)-1;}
struct Bits { Type type; std::uint64_t bits; };
struct Binary { bool negative; std::uint64_t significand; int exponent; };
Binary normalized(Binary value) {
  if(value.significand)while(!(value.significand&1)){value.significand>>=1;++value.exponent;}
  return value;
}
std::optional<Binary> binary(Bits value) {
  const auto t=value.type;const auto raw=value.bits&mask(t.width);
  // CUDA FP4 E2M1 is finite; UE8M0 is exponent-only (255 is NaN).
  if(t.encoding==Encoding::UE8M0)return raw==255?std::nullopt:std::optional<Binary>({false,1,int(raw)-127});
  if(t.encoding==Encoding::UE4M3 && (raw&128 || raw==127))return {};
  if(!t.exponent) {
    const bool negative=t.sign && (raw>>(t.width-1));
    return normalized({negative,negative?((~raw+1)&mask(t.width)):raw,0});
  }
  const auto exponent=(raw>>t.fraction)&mask(t.exponent);
  if(t.encoding==Encoding::Ieee && exponent==mask(t.exponent))return {};
  const int bias=(1<<(t.exponent-1))-1;
  auto significand=raw&mask(t.fraction);
  if(exponent)significand|=std::uint64_t(1)<<t.fraction;
  return normalized({bool(raw>>(t.width-1)),significand,
      (exponent?int(exponent)-bias:1-bias)-int(t.fraction)});
}
std::optional<Bits> encode(Binary value,Type target) {
  value=normalized(value);
  if(!target.exponent)return {};
  if(target.encoding!=Encoding::Ieee) {
    for(std::uint64_t bits=0;bits<=mask(target.width);++bits) {
      const auto decoded=binary({target,bits});
      if(decoded && decoded->negative==value.negative && decoded->significand==value.significand &&
          (!value.significand || decoded->exponent==value.exponent))return Bits{target,bits};
    }
    return {};
  }
  const auto sign=value.negative?(std::uint64_t(1)<<(target.width-1)):0;
  if(!value.significand)return Bits{target,sign};
  const unsigned length=std::bit_width(value.significand);
  const int highest=int(length)-1+value.exponent;
  const int bias=(1<<(target.exponent-1))-1, minimum=1-bias;
  if(highest>bias)return {};
  if(highest>=minimum) {
    if(length>target.fraction+1)return {};
    const auto fraction=(value.significand<<(target.fraction+1-length))&mask(target.fraction);
    return Bits{target,sign|(std::uint64_t(highest+bias)<<target.fraction)|fraction};
  }
  const int shift=value.exponent-(minimum-int(target.fraction));
  if(shift<0 || unsigned(shift)+length>target.fraction)return {};
  return Bits{target,sign|(value.significand<<shift)};
}
std::optional<Bits> parse(const std::string& type,const std::string& literal) {
  const auto spec=typeOf(type);if(!spec || literal.empty())return {};
  if(literal.starts_with("0f") || literal.starts_with("0d") || literal.starts_with("0x")) {
    const auto expected=literal[1]=='f'?32u:literal[1]=='d'?64u:spec->width;
    if(expected!=spec->width)return {};
    std::uint64_t raw=0;auto p=std::from_chars(literal.data()+2,literal.data()+literal.size(),raw,16);
    if(p.ec!=std::errc{} || p.ptr!=literal.data()+literal.size() || raw>mask(spec->width))return {};
    return Bits{*spec,raw};
  }
  if(spec->exponent) {
    if(type=="f32") {
      float value=0;auto p=floatFromChars(literal.data(),literal.data()+literal.size(),value);
      if(p.ec!=std::errc{} || p.ptr!=literal.data()+literal.size() || !std::isfinite(value))return {};
      return Bits{*spec,std::bit_cast<std::uint32_t>(value)};
    }
    double value=0;auto p=floatFromChars(literal.data(),literal.data()+literal.size(),value);
    if(p.ec!=std::errc{} || p.ptr!=literal.data()+literal.size() || !std::isfinite(value))return {};
    auto raw=Bits{*typeOf("f64"),std::bit_cast<std::uint64_t>(value)};
    return encode(*binary(raw),*spec);
  }
  if(spec->sign || (type[0]=='b' && literal[0]=='-')) {
    std::int64_t value=0;auto p=std::from_chars(literal.data(),literal.data()+literal.size(),value);
    if(p.ec!=std::errc{} || p.ptr!=literal.data()+literal.size())return {};
    if(spec->width<64 && (value<-(std::int64_t(1)<<(spec->width-1)) || value>=(std::int64_t(1)<<(spec->width-1))))return {};
    return Bits{*spec,std::uint64_t(value)&mask(spec->width)};
  }
  std::uint64_t value=0;auto p=std::from_chars(literal.data(),literal.data()+literal.size(),value);
  if(p.ec!=std::errc{} || p.ptr!=literal.data()+literal.size() || value>mask(spec->width))return {};
  return Bits{*spec,value};
}
std::string multiplyDecimal(std::string text,unsigned factor) {
  unsigned carry=0;
  for(auto i=text.size();i--;) {auto digit=unsigned(text[i]-'0')*factor+carry;text[i]=char('0'+digit%10);carry=digit/10;}
  while(carry){text.insert(text.begin(),char('0'+carry%10));carry/=10;}
  return text;
}
std::optional<ConstantValue> literal(Bits bits,std::string type) {
  if(!bits.type.exponent) {
    const auto width=bits.type.width;const auto raw=bits.bits&mask(width);
    if(bits.type.sign && (raw>>(width-1)))
      return ConstantValue{std::move(type),"-"+std::to_string((~raw+1)&mask(width))};
    return ConstantValue{std::move(type),std::to_string(raw)};
  }
  auto value=binary(bits);if(!value)return {};
  std::string text=std::to_string(value->significand);
  if(value->significand) {
    for(int i=0;i<value->exponent;++i)text=multiplyDecimal(std::move(text),2);
    if(value->exponent<0) {
      const auto places=unsigned(-value->exponent);
      for(unsigned i=0;i<places;++i)text=multiplyDecimal(std::move(text),5);
      if(text.size()<=places)text.insert(0,places+1-text.size(),'0');
      text.insert(text.size()-places,1,'.');
      while(text.back()=='0')text.pop_back();
      if(text.back()=='.')text.pop_back();
    }
  }
  if(value->negative)text="-"+text;
  return ConstantValue{std::move(type),std::move(text)};
}
std::string trim(std::string value) {
  while(!value.empty() && value.back()==' ')value.pop_back();
  const auto start=value.find_first_not_of(' ');return start==std::string::npos?"":value.substr(start);
}
std::optional<Bits> convert(Bits value,Type target,const std::string& op) {
  if(op=="Bitcast")return value.type.width==target.width?std::optional<Bits>(Bits{target,value.bits}):std::nullopt;
  if(op=="ZeroExtend") {
    if(value.type.exponent || target.exponent || target.width<value.type.width)return {};
    return Bits{target,value.bits};
  }
  if(op=="NumericInterpretation" && (!value.type.exponent || !target.exponent))return {};
  if(op!="Convert" && op!="NumericInterpretation" && op!="Convert rounding=nearest-even" &&
     op!="Convert rounding=zero" && op!="Convert rounding=negative-infinity" &&
     op!="Convert rounding=positive-infinity")return {};
  if(!value.type.exponent && !target.exponent) {
    auto raw=value.bits;
    if(value.type.sign && (raw>>(value.type.width-1)))raw|=~mask(value.type.width);
    return Bits{target,raw&mask(target.width)};
  }
  auto b=binary(value);if(!b)return {};
  if(target.exponent)return encode(*b,target);
  if(b->significand && (b->exponent<0 || std::bit_width(b->significand)+b->exponent>64))return {};
  auto magnitude=b->significand?(b->significand<<b->exponent):0;
  if(target.sign) {
    const auto edge=std::uint64_t(1)<<(target.width-1);
    if(magnitude>edge || (!b->negative && magnitude==edge))return {};
  } else if((b->negative && magnitude) || magnitude>mask(target.width))return {};
  return Bits{target,(b->negative?0-magnitude:magnitude)&mask(target.width)};
}
}
std::optional<ConstantValue> constantValue(std::string type,std::string text,
                                         std::string conversion,std::string converted_type) {
  auto value=parse(type,text);if(!value)return {};
  while(!conversion.empty()) {
    const auto end=conversion.find("; ");
    const auto step=conversion.substr(0,end);
    const auto begin=step.find('('),arrow=step.find(" -> ");
    if(begin==std::string::npos || arrow==std::string::npos || step.back()!=')' || arrow<=begin)return {};
    const auto source=step.substr(begin+1,arrow-begin-1);
    const auto target=step.substr(arrow+4,step.size()-arrow-5);
    if(source!=type)return {};
    const auto spec=typeOf(target);if(!spec)return {};
    value=convert(*value,*spec,trim(step.substr(0,begin)));if(!value)return {};
    type=target;
    if(end==std::string::npos){conversion.clear();break;}
    conversion.erase(0,end+2);
  }
  if(!converted_type.empty() && type!=converted_type)return {};
  return literal(*value,std::move(type));
}
std::optional<ConstantValue> constantOperation(const std::string& name,const std::string& type,
    const std::string& attributes,std::span<const ConstantValue> operands) {
  if(name=="BitNot") {
    if(!attributes.empty() || operands.size()!=1 || operands[0].type!=type)return {};
    if(type=="pred") {
      if(operands[0].literal=="0")return ConstantValue{type,"1"};
      if(operands[0].literal=="1")return ConstantValue{type,"0"};
      return {};
    }
    auto value=parse(type,operands[0].literal);
    if(!value || value->type.exponent)return {};
    value->bits=(~value->bits)&mask(value->type.width);
    return literal(*value,type);
  }
  if(type=="pred" && attributes.empty() && operands.size()==2 &&
     std::all_of(operands.begin(),operands.end(),[](const auto& x){
       return x.type=="pred" && (x.literal=="0" || x.literal=="1");})) {
    const bool a=operands[0].literal=="1",b=operands[1].literal=="1";
    if(name=="BitAnd")return ConstantValue{"pred",a&&b?"1":"0"};
    if(name=="BitOr")return ConstantValue{"pred",a||b?"1":"0"};
    if(name=="BitXor")return ConstantValue{"pred",a!=b?"1":"0"};
    return {};
  }
  if(name=="Multiply" && trim(attributes)=="multiply=high" && operands.size()==2 &&
     operands[0].type==type && operands[1].type==type) {
    auto a=parse(type,operands[0].literal),b=parse(type,operands[1].literal);
    if(!a || !b || a->type.exponent || b->type.exponent)return {};
    const auto width=a->type.width;
    unsigned __int128 product;
    if(a->type.sign) {
      const auto signed_value=[&](std::uint64_t x)->__int128 {
        return (x>>(width-1))?static_cast<__int128>(x)-(static_cast<__int128>(1)<<width):x;
      };
      product=static_cast<unsigned __int128>(signed_value(a->bits)*signed_value(b->bits));
    } else product=static_cast<unsigned __int128>(a->bits)*b->bits;
    return literal({a->type,static_cast<std::uint64_t>(product>>width)&mask(width)},type);
  }
  if(name=="Compare" && type=="pred" && operands.size()==2 && operands[0].type==operands[1].type) {
    auto a=parse(operands[0].type,operands[0].literal),b=parse(operands[1].type,operands[1].literal);
    // Exact integer comparisons only. Floating NaN/FTZ rules are not inferred.
    if(!a || !b || a->type.exponent || b->type.exponent)return {};
    const auto sign=a->type.sign?(std::uint64_t(1)<<(a->type.width-1)):0;
    const auto x=a->bits^sign,y=b->bits^sign;
    bool result;
    if(attributes=="predicate=eq")result=x==y;
    else if(attributes=="predicate=ne")result=x!=y;
    else if(attributes=="predicate=lt")result=x<y;
    else if(attributes=="predicate=le")result=x<=y;
    else if(attributes=="predicate=gt")result=x>y;
    else if(attributes=="predicate=ge")result=x>=y;
    else return {};
    return ConstantValue{"pred",result?"1":"0"};
  }
  if(name=="ExtractBits" && operands.size()==2) {
    const auto target=typeOf(type),source=typeOf(operands[0].type);
    if(!target || !source || attributes!="width="+std::to_string(target->width))return {};
    auto a=parse(operands[0].type,operands[0].literal),b=parse(operands[1].type,operands[1].literal);
    if(!a || !b || b->type.exponent || b->bits>source->width || target->width>source->width-b->bits)return {};
    return literal({*target,(a->bits>>b->bits)&mask(target->width)},type);
  }
  if(name=="PtxFunnelShift") {
    if(type!="b32" || operands.size()!=3)return {};
    bool left=false,wrap=false,recognized=false;
    for(const auto* direction:{"l","r"})for(const auto* mode:{"wrap","clamp"})
      if(attributes=="source=b32."+std::string(direction)+"."+mode+".b32 source_type=b32") {
        left=std::string_view(direction)=="l";wrap=std::string_view(mode)=="wrap";recognized=true;
      }
    if(!recognized)return {};
    auto a=parse(operands[0].type,operands[0].literal);
    auto b=parse(operands[1].type,operands[1].literal);
    auto c=parse(operands[2].type,operands[2].literal);
    if(!a || !b || !c || a->type.width!=32 || b->type.width!=32 || c->type.width!=32 ||
        a->type.exponent || b->type.exponent || c->type.exponent)return {};
    // PTX shf: https://docs.nvidia.com/cuda/parallel-thread-execution/#logic-and-shift-instructions-shf
    const auto shift=wrap?(c->bits&31):std::min<std::uint64_t>(c->bits,32);
    const auto packed=(b->bits<<32)|a->bits;
    const auto result=left?((packed<<shift)>>32):(packed>>shift);
    return literal(Bits{*typeOf("b32"),result&UINT32_MAX},type);
  }
  if(!attributes.empty() || operands.size()!=2)return {};
  auto target=typeOf(type);if(!target || target->exponent)return {};
  auto a=parse(operands[0].type,operands[0].literal),b=parse(operands[1].type,operands[1].literal);
  if(!a || !b || a->type.exponent || b->type.exponent || operands[0].type!=type)return {};
  const auto x=a->bits,y=b->bits;const auto width=target->width;
  std::uint64_t result;
  if(name=="ShiftLeft")result=y>=width?0:(x<<y);
  else if(name=="ShiftRight") {
    const bool negative=target->sign && (x>>(width-1));
    result=y>=width?(negative?mask(width):0):(x>>y);
    if(negative && y && y<width)result|=mask(width)^mask(width-unsigned(y));
  } else {
    if(operands[1].type!=type)return {};
    if(name=="Add")result=x+y;
    else if(name=="Subtract")result=x-y;
    else if(name=="Multiply")result=x*y;
    else if(name=="BitAnd")result=x&y;
    else if(name=="BitOr")result=x|y;
    else if(name=="BitXor")result=x^y;
    else if(name=="Maximum" || name=="Minimum") {
      const auto sign=target->sign?(std::uint64_t(1)<<(width-1)):0;
      const auto less=(x^sign)<(y^sign);
      result=(name=="Maximum"?less:!less)?y:x;
    }
    else if(name=="Divide" && y && !target->sign)result=x/y;
    else if(name=="Remainder" && y && !target->sign)result=x%y;
    else if((name=="Divide" || name=="Remainder") && y && target->sign) {
      const auto signed_value=[&](std::uint64_t raw)->__int128 {
        return (raw>>(width-1))?static_cast<__int128>(raw)-(static_cast<__int128>(1)<<width):raw;
      };
      const auto sx=signed_value(x),sy=signed_value(y);
      if(sx==-(static_cast<__int128>(1)<<(width-1)) && sy==-1)return {};
      result=static_cast<std::uint64_t>(name=="Divide"?sx/sy:sx%sy);
    }
    else return {};
  }
  return literal(Bits{*target,result&mask(width)},type);
}
}
