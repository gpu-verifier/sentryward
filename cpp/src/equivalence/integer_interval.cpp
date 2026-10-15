#include "sentryward/equivalence/integer_interval.h"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <limits>
#include <utility>

namespace sentryward::equivalence::integer_interval {
namespace {
using Bounds = std::pair<std::int64_t,std::int64_t>;
std::optional<Bounds> bounds(Type type) {
  if(type.bits!=8 && type.bits!=16 && type.bits!=32 && type.bits!=64)return {};
  if(type.is_signed) {
    if(type.bits==64)return Bounds{std::numeric_limits<std::int64_t>::min(),std::numeric_limits<std::int64_t>::max()};
    const auto sign=std::int64_t{1}<<(type.bits-1);
    return Bounds{-sign,sign-1};
  }
  if(type.bits==64)return Bounds{0,std::numeric_limits<std::int64_t>::max()};
  return Bounds{0,(std::int64_t{1}<<type.bits)-1};
}
bool admissible(const Interval& value) {
  const auto range=bounds(value.type);
  return range && value.lower<=value.upper && value.lower>=range->first && value.upper<=range->second;
}
bool divisorAdmissible(const Interval& value,std::int64_t divisor) {
  if(!admissible(value) || divisor<=0)return false;
  const auto range=bounds(value.type);
  return divisor<=range->second;
}
Truth invert(Truth value) {
  if(value==Truth::True)return Truth::False;
  if(value==Truth::False)return Truth::True;
  return Truth::Unknown;
}
} // namespace

std::optional<Type> parseType(std::string_view text) {
  if(text.starts_with("exact:"))text.remove_prefix(6);
  if(text.empty() || (text.front()!='u' && text.front()!='s'))return {};
  Type result;result.is_signed=text.front()=='s';text.remove_prefix(1);
  if(text=="8")result.bits=8;
  else if(text=="16")result.bits=16;
  else if(text=="32")result.bits=32;
  else if(text=="64")result.bits=64;
  else return {};
  return result;
}
Result interval(Type type,std::int64_t lower,std::int64_t upper) {
  Interval result{type,lower,upper};
  if(!admissible(result))return {};
  return result;
}
Result typeRange(Type type) {
  if(!type.is_signed && type.bits==64)return {};
  auto range=bounds(type);if(!range)return {};
  return interval(type,range->first,range->second);
}
Result literal(Type type,std::string_view text) {
  if(text.empty() || text.size()>21)return {};
  if(text.front()=='+'){
    text.remove_prefix(1);
    if(text.empty() || text.front()=='+' || text.front()=='-')return {};
  }
  std::int64_t value=0;
  const auto parsed=std::from_chars(text.data(),text.data()+text.size(),value);
  if(parsed.ec!=std::errc{} || parsed.ptr!=text.data()+text.size())return {};
  return interval(type,value,value);
}
Result literal(std::string_view type,std::string_view text) {
  auto parsed=parseType(type);if(!parsed)return {};
  return literal(*parsed,text);
}
Result externalIndex(Type type,std::uint64_t extent) {
  if(!extent || extent-1>static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))return {};
  return interval(type,0,static_cast<std::int64_t>(extent-1));
}
Result externalIndex(std::string_view type,std::uint64_t extent) {
  auto parsed=parseType(type);if(!parsed)return {};
  return externalIndex(*parsed,extent);
}
Result apply(Operation operation,const Interval& a,const Interval& b) {
  if(a.type!=b.type || !admissible(a) || !admissible(b))return {};
  std::int64_t lower=0,upper=0;
  switch(operation) {
    case Operation::Add:
      if(__builtin_add_overflow(a.lower,b.lower,&lower) ||
          __builtin_add_overflow(a.upper,b.upper,&upper))return {};
      break;
    case Operation::Subtract:
      if(__builtin_sub_overflow(a.lower,b.upper,&lower) ||
          __builtin_sub_overflow(a.upper,b.lower,&upper))return {};
      break;
    case Operation::Multiply: {
      std::array<std::int64_t,4> endpoints{};
      if(__builtin_mul_overflow(a.lower,b.lower,&endpoints[0]) ||
          __builtin_mul_overflow(a.lower,b.upper,&endpoints[1]) ||
          __builtin_mul_overflow(a.upper,b.lower,&endpoints[2]) ||
          __builtin_mul_overflow(a.upper,b.upper,&endpoints[3]))return {};
      const auto [minimum,maximum]=std::minmax_element(endpoints.begin(),endpoints.end());
      lower=*minimum;upper=*maximum;break;
    }
    case Operation::BitAnd:
      // Nonnegative two's-complement values have no sign bit. AND can set no
      // bit absent from either value, hence is bounded by each operand.
      if(a.lower<0 || b.lower<0)return {};
      lower=0;upper=std::min(a.upper,b.upper);break;
    case Operation::BitXor: {
      if(a.lower<0 || b.lower<0)return {};
      const auto maximum=static_cast<std::uint64_t>(std::max(a.upper,b.upper));
      const auto bits=std::bit_width(maximum);
      lower=0;upper=bits?static_cast<std::int64_t>((std::uint64_t{1}<<bits)-1):0;
      break;
    }
    default:return {};
  }
  // Reject possible typed wrap even if the int64 endpoint operation succeeded.
  return interval(a.type,lower,upper);
}
Result divideByPositiveLiteral(const Interval& value,std::int64_t divisor) {
  if(!divisorAdmissible(value,divisor))return {};
  // Truncation toward zero is monotone for a strictly positive divisor. The
  // only signed division overflow, INT64_MIN/-1, cannot occur here.
  return interval(value.type,value.lower/divisor,value.upper/divisor);
}
Result remainderByPositiveLiteral(const Interval& value,std::int64_t divisor) {
  if(!divisorAdmissible(value,divisor))return {};
  if(value.lower/divisor==value.upper/divisor)
    return interval(value.type,value.lower%divisor,value.upper%divisor);
  const auto magnitude=divisor-1;
  const auto lower=value.lower<0?std::max(value.lower,-magnitude):std::int64_t{0};
  const auto upper=value.upper>0?std::min(value.upper,magnitude):std::int64_t{0};
  return interval(value.type,lower,upper);
}
Result shiftRightByLiteral(const Interval& value,std::int64_t shift) {
  if(!admissible(value) || value.lower<0 || shift<0 || shift>=value.type.bits)return {};
  return interval(value.type,value.lower>>shift,value.upper>>shift);
}
Truth compare(Comparison operation,const Interval& a,const Interval& b) {
  if(a.type!=b.type || !admissible(a) || !admissible(b))return Truth::Unknown;
  switch(operation) {
    case Comparison::Less:
      if(a.upper<b.lower)return Truth::True;
      if(a.lower>=b.upper)return Truth::False;
      break;
    case Comparison::LessEqual:
      if(a.upper<=b.lower)return Truth::True;
      if(a.lower>b.upper)return Truth::False;
      break;
    case Comparison::Greater:return compare(Comparison::Less,b,a);
    case Comparison::GreaterEqual:return compare(Comparison::LessEqual,b,a);
    case Comparison::Equal:
      if(a.lower==a.upper && b.lower==b.upper && a.lower==b.lower)return Truth::True;
      if(a.upper<b.lower || b.upper<a.lower)return Truth::False;
      break;
    case Comparison::NotEqual:return invert(compare(Comparison::Equal,a,b));
    default:return Truth::Unknown;
  }
  return Truth::Unknown;
}
Truth compare(Comparison operation,const Result& a,const Result& b) {
  if(!a || !b)return Truth::Unknown;
  return compare(operation,*a,*b);
}

} // namespace sentryward::equivalence::integer_interval
