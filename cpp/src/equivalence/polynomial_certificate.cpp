#include "sentryward/equivalence/polynomial_certificate.h"

#include <algorithm>
#include <limits>
#include <map>

namespace sentryward::equivalence::polynomial_certificate {
namespace {
Integer absolute(const Integer& value) { return value < 0 ? -value : value; }
std::size_t bits(const Integer& value) {
  return value == 0 ? 0 : boost::multiprecision::msb(absolute(value)) + 1;
}
std::size_t words(const Integer& value) { return (bits(value)+63)/64; }
bool less(const Monomial& a,const Monomial& b) { return a.powers < b.powers; }
struct Less {
  bool operator()(const Monomial& a,const Monomial& b) const { return less(a,b); }
};
std::uint32_t degree(const Monomial& monomial) {
  std::uint32_t result=0;
  for(auto [atom,power]:monomial.powers) { (void)atom;result+=power; }
  return result;
}
} // namespace

bool CanonicalPolynomial::operator<(const CanonicalPolynomial& other) const {
  return std::lexicographical_compare(terms_.begin(),terms_.end(),other.terms_.begin(),other.terms_.end(),
      [](const Term& a,const Term& b) {
        if(a.monomial!=b.monomial)return less(a.monomial,b.monomial);
        if(a.coefficient.numerator!=b.coefficient.numerator)
          return a.coefficient.numerator<b.coefficient.numerator;
        return a.coefficient.denominator<b.coefficient.denominator;
      });
}

Builder::Builder(Limits limits,WorkCharge work_charge)
    : limits_(limits),work_charge_(std::move(work_charge)) {
  // Configuration may tighten, not remove, the allocation bounds.
  limits_.max_terms=std::min<std::size_t>(limits_.max_terms,64);
  limits_.max_degree=std::min<std::uint32_t>(limits_.max_degree,64);
  limits_.max_coefficient_bits=std::min<std::size_t>(limits_.max_coefficient_bits,4096);
  limits_.max_intermediate_bits=std::min<std::size_t>(limits_.max_intermediate_bits,8192);
  limits_.max_coefficient_bits=std::min(limits_.max_coefficient_bits,limits_.max_intermediate_bits);
  limits_.max_decimal_chars=std::min<std::size_t>(limits_.max_decimal_chars,1152);
}
bool Builder::charge(std::size_t amount) {
  if(exhausted_)return false;
  if(amount>limits_.max_work-work_ || (work_charge_ && !work_charge_(amount))) {
    exhausted_=true;return false;
  }
  work_+=amount;return true;
}
bool Builder::bounded(const Integer& value,std::size_t limit) const { return bits(value)<=limit; }
bool Builder::coefficientAdmissible(const Rational& coefficient) const {
  return coefficient.denominator>0 && bounded(coefficient.numerator,limits_.max_coefficient_bits) &&
      bounded(coefficient.denominator,limits_.max_coefficient_bits);
}
bool Builder::admissible(const CanonicalPolynomial& polynomial) {
  if(polynomial.terms_.size()>limits_.max_terms)return false;
  for(const auto& term:polynomial.terms_) {
    if(!charge(1+term.monomial.powers.size()+words(term.coefficient.numerator)+
        words(term.coefficient.denominator)))return false;
    if(!coefficientAdmissible(term.coefficient) || degree(term.monomial)>limits_.max_degree)return false;
  }
  return true;
}
std::optional<Integer> Builder::multiply(const Integer& a,const Integer& b) {
  // Reject before allocating an unbounded intermediate. For nonzero values,
  // product bits are either bits(a)+bits(b) or one less; be conservative.
  if(a!=0 && b!=0 && bits(a)+bits(b)>limits_.max_intermediate_bits)return {};
  if(!charge(1+words(a)*words(b)))return {};
  Integer result=a*b;
  if(!bounded(result,limits_.max_intermediate_bits))return {};
  return result;
}
std::optional<Integer> Builder::gcd(Integer a,Integer b) {
  a=absolute(a);b=absolute(b);
  while(b!=0) {
    if(!charge(1+words(a)*words(b)))return {};
    Integer remainder=a%b;a=std::move(b);b=std::move(remainder);
  }
  return a;
}
std::optional<Rational> Builder::normalize(Integer numerator,Integer denominator) {
  if(denominator==0 || !bounded(numerator,limits_.max_intermediate_bits) ||
      !bounded(denominator,limits_.max_intermediate_bits))return {};
  if(numerator==0)return Rational{};
  if(denominator<0){numerator=-numerator;denominator=-denominator;}
  auto factor=gcd(numerator,denominator);if(!factor)return {};
  if(!charge(1+words(numerator)*words(*factor)+words(denominator)*words(*factor)))return {};
  Rational result{numerator/ *factor,denominator/ *factor};
  if(!coefficientAdmissible(result))return {};
  return result;
}
std::optional<Rational> Builder::parse(std::string_view text) {
  if(text.empty() || text.size()>limits_.max_decimal_chars || !charge(text.size()))return {};
  std::size_t cursor=0;bool negative=false;
  if(text[cursor]=='+' || text[cursor]=='-'){negative=text[cursor]=='-';if(++cursor==text.size())return {};}
  Integer numerator=0;std::size_t digits=0,fractional=0;bool point=false;
  for(;cursor<text.size();++cursor) {
    const auto c=text[cursor];
    if(c=='.' && !point){point=true;continue;}
    if(c<'0' || c>'9')break;
    if(!charge(1+words(numerator)))return {};
    numerator*=10;numerator+=c-'0';
    if(!bounded(numerator,limits_.max_intermediate_bits))return {};
    ++digits;if(point)++fractional;
  }
  if(!digits)return {};
  std::int64_t exponent=0;
  if(cursor<text.size() && (text[cursor]=='e' || text[cursor]=='E')) {
    ++cursor;bool exponent_negative=false;
    if(cursor<text.size() && (text[cursor]=='+' || text[cursor]=='-')){
      exponent_negative=text[cursor]=='-';++cursor;
    }
    const auto start=cursor;
    for(;cursor<text.size() && text[cursor]>='0' && text[cursor]<='9';++cursor) {
      if(exponent>8192)return {};
      exponent=exponent*10+(text[cursor]-'0');
    }
    if(cursor==start || exponent>8192)return {};
    if(exponent_negative)exponent=-exponent;
  }
  if(cursor!=text.size())return {};
  exponent-=static_cast<std::int64_t>(fractional);
  if(numerator==0)return Rational{};
  if(exponent>8192 || exponent < -8192)return {};
  Integer denominator=1;
  // Cancel decimal trailing zeros before constructing a power of ten.
  while(exponent<0 && numerator%10==0) {
    if(!charge(1+words(numerator)))return {};
    numerator/=10;++exponent;
  }
  const auto power=exponent<0?-exponent:exponent;
  // 10^power has at least 3*power+1 bits, allowing an early bounded failure.
  if(power && static_cast<std::uint64_t>(power)>limits_.max_intermediate_bits/3)return {};
  Integer scale=1;
  for(std::int64_t i=0;i<power;++i) {
    if(!charge(1+words(scale)))return {};
    scale*=10;if(!bounded(scale,limits_.max_intermediate_bits))return {};
  }
  if(exponent>=0){auto scaled=multiply(numerator,scale);if(!scaled)return {};numerator=std::move(*scaled);}
  else denominator=std::move(scale);
  if(negative)numerator=-numerator;
  return normalize(std::move(numerator),std::move(denominator));
}
std::optional<Rational> Builder::add(const Rational& a,const Rational& b,bool subtract) {
  auto common=gcd(a.denominator,b.denominator);if(!common)return {};
  const Integer left_scale=b.denominator/ *common,right_scale=a.denominator/ *common;
  if(!charge(1+words(a.denominator)*words(*common)+words(b.denominator)*words(*common)))return {};
  auto left=multiply(a.numerator,left_scale),right=multiply(b.numerator,right_scale);
  auto denominator=multiply(a.denominator,left_scale);
  if(!left || !right || !denominator || !charge(1+words(*left)+words(*right)))return {};
  Integer numerator=*left;if(subtract)numerator-=*right;else numerator+=*right;
  return normalize(std::move(numerator),std::move(*denominator));
}
std::optional<Rational> Builder::product(const Rational& a,const Rational& b) {
  auto first=gcd(a.numerator,b.denominator),second=gcd(b.numerator,a.denominator);
  if(!first || !second)return {};
  if(!charge(1+words(a.numerator)*words(*first)+words(b.denominator)*words(*first)+
      words(b.numerator)*words(*second)+words(a.denominator)*words(*second)))return {};
  auto numerator=multiply(a.numerator/ *first,b.numerator/ *second);
  auto denominator=multiply(a.denominator/ *second,b.denominator/ *first);
  if(!numerator || !denominator)return {};
  return normalize(std::move(*numerator),std::move(*denominator));
}
std::optional<Monomial> Builder::product(const Monomial& a,const Monomial& b) {
  if(degree(a)>limits_.max_degree-degree(b) || !charge(1+a.powers.size()+b.powers.size()))return {};
  Monomial result;result.powers.reserve(a.powers.size()+b.powers.size());
  std::size_t i=0,j=0;
  while(i<a.powers.size() || j<b.powers.size()) {
    if(j==b.powers.size() || (i<a.powers.size() && a.powers[i].first<b.powers[j].first))result.powers.push_back(a.powers[i++]);
    else if(i==a.powers.size() || b.powers[j].first<a.powers[i].first)result.powers.push_back(b.powers[j++]);
    else {result.powers.emplace_back(a.powers[i].first,a.powers[i].second+b.powers[j].second);++i;++j;}
  }
  return result;
}
Result Builder::literal(std::string_view text) {
  auto coefficient=parse(text);if(!coefficient)return {};
  if(coefficient->numerator==0)return CanonicalPolynomial{};
  if(!limits_.max_terms)return {};
  std::vector<Term> terms;terms.push_back({{},std::move(*coefficient)});
  return CanonicalPolynomial{std::move(terms)};
}
Result Builder::atom(Atom id) {
  if(!limits_.max_terms || !limits_.max_degree || !limits_.max_coefficient_bits ||
      !limits_.max_intermediate_bits || !charge(2))return {};
  Monomial monomial;monomial.powers.emplace_back(id,1);
  std::vector<Term> terms;terms.push_back({std::move(monomial),{1,1}});
  return CanonicalPolynomial{std::move(terms)};
}
Result Builder::combine(const CanonicalPolynomial& a,const CanonicalPolynomial& b,bool subtract) {
  std::vector<Term> terms;terms.reserve(std::min(limits_.max_terms,a.terms_.size()+b.terms_.size()));
  std::size_t i=0,j=0;
  while(i<a.terms_.size() || j<b.terms_.size()) {
    if(!charge(1))return {};
    if(j==b.terms_.size() || (i<a.terms_.size() && less(a.terms_[i].monomial,b.terms_[j].monomial)))terms.push_back(a.terms_[i++]);
    else if(i==a.terms_.size() || less(b.terms_[j].monomial,a.terms_[i].monomial)) {
      terms.push_back(b.terms_[j++]);if(subtract)terms.back().coefficient.numerator=-terms.back().coefficient.numerator;
    } else {
      auto coefficient=add(a.terms_[i].coefficient,b.terms_[j].coefficient,subtract);if(!coefficient)return {};
      if(coefficient->numerator!=0)terms.push_back({a.terms_[i].monomial,std::move(*coefficient)});
      ++i;++j;
    }
    if(terms.size()>limits_.max_terms)return {};
  }
  return CanonicalPolynomial{std::move(terms)};
}
Result Builder::negate(const CanonicalPolynomial& value) {
  auto terms=value.terms_;
  for(auto& term:terms)term.coefficient.numerator=-term.coefficient.numerator;
  return CanonicalPolynomial{std::move(terms)};
}
Result Builder::product(const CanonicalPolynomial& a,const CanonicalPolynomial& b) {
  if(a.terms_.empty() || b.terms_.empty())return CanonicalPolynomial{};
  std::map<Monomial,Rational,Less> accumulated;
  for(const auto& x:a.terms_)for(const auto& y:b.terms_) {
    if(!charge(1))return {};
    auto monomial=product(x.monomial,y.monomial);if(!monomial)return {};
    auto coefficient=product(x.coefficient,y.coefficient);if(!coefficient)return {};
    // std::map comparison can scan the full bounded monomial at each level.
    // A red-black tree of at most 64 terms has fewer than 14 such levels.
    if(!charge(14*(1+monomial->powers.size())))return {};
    auto found=accumulated.find(*monomial);
    if(found==accumulated.end()) {
      if(accumulated.size()==limits_.max_terms)return {};
      accumulated.emplace(std::move(*monomial),std::move(*coefficient));
    } else {
      auto combined=add(found->second,*coefficient,false);if(!combined)return {};
      if(combined->numerator==0)accumulated.erase(found);else found->second=std::move(*combined);
    }
  }
  std::vector<Term> terms;terms.reserve(accumulated.size());
  for(auto& [monomial,coefficient]:accumulated) {
    if(!charge(1+monomial.powers.size()))return {};
    terms.push_back({monomial,std::move(coefficient)});
  }
  return CanonicalPolynomial{std::move(terms)};
}
Result Builder::apply(Operation operation,std::span<const CanonicalPolynomial* const> operands) {
  if(!charge(1))return {};
  const auto arity=operation==Operation::Negate?1u:2u;
  if(operands.size()!=arity)return {};
  for(auto operand:operands)if(!operand || !admissible(*operand))return {};
  switch(operation) {
    case Operation::Add:return combine(*operands[0],*operands[1],false);
    case Operation::Subtract:return combine(*operands[0],*operands[1],true);
    case Operation::Negate:return negate(*operands[0]);
    case Operation::Multiply:return product(*operands[0],*operands[1]);
  }
  return {};
}
Result Builder::divideByLiteral(const CanonicalPolynomial& numerator,std::string_view divisor) {
  if(!charge(1))return {};
  if(!admissible(numerator))return {};
  auto value=parse(divisor);if(!value || value->numerator==0)return {};
  Rational inverse{value->denominator,value->numerator};
  if(inverse.denominator<0){inverse.numerator=-inverse.numerator;inverse.denominator=-inverse.denominator;}
  std::vector<Term> terms;terms.reserve(numerator.terms_.size());
  for(const auto& term:numerator.terms_) {
    auto coefficient=product(term.coefficient,inverse);if(!coefficient)return {};
    terms.push_back({term.monomial,std::move(*coefficient)});
  }
  return CanonicalPolynomial{std::move(terms)};
}

} // namespace sentryward::equivalence::polynomial_certificate
