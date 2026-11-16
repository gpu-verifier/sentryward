#pragma once
#include "egraph_engine.hpp"

namespace sentryward::equivalence {
namespace {

inline auto Engine::find(Class id) -> Class {
  while (parents_[id] != id) {
    parents_[id] = parents_[parents_[id]];
    id = parents_[id];
  }
  return id;
}

inline auto Engine::scopeProperties(const Key& k) -> std::pair<std::int64_t,bool> {
  std::int64_t maximum=k.kind==static_cast<int>(s::Kind::Index)?static_cast<std::int64_t>(k.index_level):-1;
  bool fold=k.kind==static_cast<int>(s::Kind::Fold);
  for(std::size_t i=0;i<k.children.size();++i) {
    if(branchWrapper(k) && i==1)continue; // Proof anchor is not a semantic operand.
    auto child=k.children[i];
    child=find(child);maximum=std::max(maximum,max_index_[child]);fold=fold || has_fold_[child];
  }
  return {maximum,fold};
}

inline auto Engine::positivity(const Key& k) -> std::pair<bool,bool> {
  if(!realType(k.type))return {false,false};
  if(k.kind==static_cast<int>(s::Kind::Parameter) && positive_parameters_.contains(k.name))
    return {true,true};
  if(k.kind==static_cast<int>(s::Kind::Constant)) {
    if(!k.attributes.empty() || !k.provenance.empty())return {false,false};
    double value=0;
    const auto parsed=floatFromChars(k.name.data(),k.name.data()+k.name.size(),value);
    if(parsed.ec==std::errc{} && parsed.ptr==k.name.data()+k.name.size() && std::isfinite(value))
      return {value>=0,value>0};
    return {false,false};
  }
  const auto nonnegative=[&](std::size_t i){return nonnegative_[find(k.children.at(i))];};
  const auto positive=[&](std::size_t i){return positive_[find(k.children.at(i))];};
  if(idealSelect(k)) {
    const auto condition=predicate_[find(k.children[0])];
    if(condition)return {nonnegative(condition==1?1:2),positive(condition==1?1:2)};
    return {nonnegative(1)&&nonnegative(2),positive(1)&&positive(2)};
  }
  if(k.kind==static_cast<int>(s::Kind::Fold)) {
    if(k.fold->count==0)return {nonnegative(0),positive(0)};
    if(k.fold->reducer==s::Reducer::Sum && nonnegative(0) && nonnegative(1))
      return {true,positive(0)||positive(1)};
    return {false,false};
  }
  if(!idealOperation(k))return {false,false};
  if(k.name=="Absolute" && finite_[find(k.children[0])])
    return {true,nonzero_[find(k.children[0])]};
  if((k.name=="Exp" || k.name=="Exp2") && finite_[find(k.children[0])])return {true,true};
  if(k.name=="Power" && positive(0) && finite_[find(k.children[0])] &&
      finite_[find(k.children[1])])return {true,true};
  if(k.name=="Sqrt" && finite_[find(k.children[0])] && nonnegative(0))
    return {true,positive(0)};
  if((k.name=="Rsqrt" || k.name=="ReciprocalSqrt") &&
      finite_[find(k.children[0])] && positive(0))return {true,true};
  if(k.name=="Multiply" && find(k.children[0])==find(k.children[1]) &&
      finite_[find(k.children[0])])return {true,nonzero_[find(k.children[0])]};
  if(k.name=="Add" && nonnegative(0) && nonnegative(1))return {true,positive(0)||positive(1)};
  if(k.name=="Maximum" && finite_[find(k.children[0])] && finite_[find(k.children[1])])
    return {nonnegative(0)||nonnegative(1),positive(0)||positive(1)};
  if(k.name=="Minimum" && finite_[find(k.children[0])] && finite_[find(k.children[1])])
    return {nonnegative(0)&&nonnegative(1),positive(0)&&positive(1)};
  if(k.name=="Multiply" && nonnegative(0) && nonnegative(1))return {true,positive(0)&&positive(1)};
  if(k.name=="Divide" && nonnegative(0) && positive(1))return {true,positive(0)};
  if(k.name=="Reciprocal" && positive(0))return {true,true};
  return {false,false};
}

inline auto Engine::provesFinite(const Key& k) -> bool {
  if(!realType(k.type))return false;
  if(k.kind==7 && k.name=="ExactNumericValue" && k.children.size()==1) {
    const auto& conversion=nodes_[find(k.children[0])];
    if(conversion.kind==6 && conversion.children.size()==1 && conversion.provenance.empty() &&
        conversion.attributes.empty()) {
      const auto& input=nodes_[find(conversion.children[0])];
      if(input.type.starts_with("exact:") && conversion.type.starts_with("exact:") &&
          finiteIntegerFloatConversion(std::string_view(input.type).substr(6),
              std::string_view(conversion.type).substr(6),conversion.name))return true;
    }
  }
  if(k.kind==static_cast<int>(s::Kind::Parameter) || k.kind==static_cast<int>(s::Kind::InitialCell))return true;
  if(k.kind==static_cast<int>(s::Kind::Constant)) {
    if(!k.attributes.empty() || !k.provenance.empty())return false;
    double value=0;
    const auto parsed=floatFromChars(k.name.data(),k.name.data()+k.name.size(),value);
    return parsed.ec==std::errc{} && parsed.ptr==k.name.data()+k.name.size() && std::isfinite(value);
  }
  if(k.kind==6 && k.provenance.empty() && k.children.size()==1) {
    const auto child=find(k.children[0]);
    const auto& source=nodes_[child].type;
    return finite_[child] && exactFloatingWidening(source,k.type) &&
        k.name=="NumericInterpretation("+source+" -> "+k.type+")";
  }
  if(idealSelect(k)) {
    const auto condition=predicate_[find(k.children[0])];
    if(condition)return finite_[find(k.children[condition==1?1:2])];
    return finite_[find(k.children[1])] && finite_[find(k.children[2])];
  }
  if(k.kind!=static_cast<int>(s::Kind::Fold) && !idealOperation(k))return false;
  if((k.name=="Exp2" || k.name=="Exp") && negative_infinity_[find(k.children[0])])return true;
  if(k.kind==static_cast<int>(s::Kind::Fold)) {
    if(k.fold->count==0)return finite_[find(k.children.at(0))];
    if((k.fold->reducer==s::Reducer::Max || k.fold->reducer==s::Reducer::Min) &&
        (k.fold->reducer==s::Reducer::Max?negative_infinity_[find(k.children.at(0))]:
            positive_infinity_[find(k.children.at(0))]) &&
        finite_[find(k.children.at(1))])return true;
  } else if(k.name=="Maximum" || k.name=="Minimum") {
    for(unsigned side=0;side<2;++side)
      if((k.name=="Maximum"?negative_infinity_[find(k.children[side])]:
          positive_infinity_[find(k.children[side])]) &&
          finite_[find(k.children[1-side])])return true;
  }
  if(k.name=="Divide" && !nonzero_[find(k.children.at(1))])return false;
  if(k.name=="Reciprocal" && !nonzero_[find(k.children.at(0))])return false;
  if(k.name=="Sqrt" && !nonnegative_[find(k.children.at(0))])return false;
  if((k.name=="Rsqrt" || k.name=="ReciprocalSqrt") &&
      !positive_[find(k.children.at(0))])return false;
  // Real logarithms need a strictly positive finite argument. General
  // real powers require a strictly positive finite base and finite exponent;
  // zero/negative bases are not admitted by an integer-exponent guess.
  if((k.name=="Log" || k.name=="Log2" || k.name=="Power") &&
      !positive_[find(k.children.at(0))])return false;
  return std::all_of(k.children.begin(),k.children.end(),[&](auto c){return finite_[find(c)];});
}

inline auto Engine::provesPredicate(const Key& k,Class id) -> std::uint8_t {
  if(k.kind==static_cast<int>(s::Kind::Constant) && (k.type=="pred" || k.type=="exact:pred")) {
    if(k.name=="1" || k.name=="true")return 1;
    if(k.name=="0" || k.name=="false")return 2;
  }
  if(branchWrapper(k))return branch_proved_wrappers_.contains(id)?1:predicate_[find(k.children[0])];
  if(k.kind==7 && k.type=="pred" && k.name=="ScopedPredicateAnd" &&
      k.attributes.empty() && k.provenance.empty() && k.children.size()==2) {
    const auto a=predicate_[find(k.children[0])],b=predicate_[find(k.children[1])];
    if(a==2 || b==2)return 2;
    if(a==1 && b==1)return 1;
    return 0;
  }
  const auto cmp=comparison(k);if(!cmp)return 0;
  const auto a=find(k.children[0]),b=find(k.children[1]);
  namespace i=integer_interval;
  const auto range_comparison=*cmp=="lt"?i::Comparison::Less:*cmp=="le"?i::Comparison::LessEqual:
      *cmp=="gt"?i::Comparison::Greater:*cmp=="ge"?i::Comparison::GreaterEqual:
      *cmp=="eq"?i::Comparison::Equal:i::Comparison::NotEqual;
  const auto integer_proof=i::compare(range_comparison,integer_ranges_[a],integer_ranges_[b]);
  if(integer_proof!=i::Truth::Unknown)return static_cast<std::uint8_t>(integer_proof);
  const auto answer=[](bool value)->std::uint8_t{return value?1:2;};
  if(finite_[a] && finite_[b]) {
    if(a==b)return answer(*cmp=="eq" || *cmp=="le" || *cmp=="ge");
    if((zero_[b] && nonzero_[a]) || (zero_[a] && nonzero_[b])) {
      if(*cmp=="eq" || *cmp=="ne")return answer(*cmp=="ne");
    }
    if(zero_[b] && positive_[a])return answer(*cmp=="ne" || *cmp=="gt" || *cmp=="ge");
    if(zero_[a] && positive_[b])return answer(*cmp=="ne" || *cmp=="lt" || *cmp=="le");
    if(zero_[b] && nonnegative_[a]) {
      if(*cmp=="ge")return 1;if(*cmp=="lt")return 2;
    }
    if(zero_[a] && nonnegative_[b]) {
      if(*cmp=="le")return 1;if(*cmp=="gt")return 2;
    }
  }
  // Ordered and unordered PTX comparisons agree on finite-vs-infinity.
  if(finite_[a] && (negative_infinity_[b] || positive_infinity_[b])) {
    const bool greater=negative_infinity_[b];
    return answer(*cmp=="ne" || (greater?(*cmp=="gt" || *cmp=="ge"):(*cmp=="lt" || *cmp=="le")));
  }
  if(finite_[b] && (negative_infinity_[a] || positive_infinity_[a])) {
    const bool greater=positive_infinity_[a];
    return answer(*cmp=="ne" || (greater?(*cmp=="gt" || *cmp=="ge"):(*cmp=="lt" || *cmp=="le")));
  }
  return 0;
}

inline auto Engine::integerRange(const Key& k) -> integer_interval::Result {
  namespace i=integer_interval;
  auto type=i::parseType(k.type);if(!type || !k.provenance.empty() || !k.attributes.empty())return {};
  if(k.kind==static_cast<int>(s::Kind::Constant))return i::literal(*type,k.name);
  if(k.kind==static_cast<int>(s::Kind::Index) && k.index_level<options_.outer_index_extents.size())
    return i::externalIndex(*type,options_.outer_index_extents[k.index_level]);
  if(k.kind==6 && k.children.size()==1) {
    const auto child=find(k.children[0]);const auto& range=integer_ranges_[child];
    if(!range)return {};
    const auto raw=[](const std::string& t){return t.starts_with("exact:")?t.substr(6):t;};
    return integerConversionRange(raw(nodes_[child].type),raw(k.type),k.name,*range);
  }
  if(k.kind!=static_cast<int>(s::Kind::Operation) || k.children.size()!=2)return {};
  auto a=integer_ranges_[find(k.children[0])],b=integer_ranges_[find(k.children[1])];
  if(!a || !b || a->type!=*type || b->type!=*type)return {};
  if(k.name=="Add")return i::apply(i::Operation::Add,*a,*b);
  if(k.name=="Subtract")return i::apply(i::Operation::Subtract,*a,*b);
  if(k.name=="Multiply")return i::apply(i::Operation::Multiply,*a,*b);
  if(b->lower==b->upper) {
    if(k.name=="Divide")return i::divideByPositiveLiteral(*a,b->lower);
    if(k.name=="Remainder")return i::remainderByPositiveLiteral(*a,b->lower);
  }
  return {};
}

inline auto Engine::rewriteUnsignedShift(Class id,const Key& k) -> void {
  if(k.kind!=6 || k.children.size()!=1 || !k.provenance.empty() || !k.attributes.empty())return;
  const auto type=k.type=="exact:u32"?"u32":k.type=="exact:u64"?"u64":"";
  if(!*type)return;const auto bits=std::string(type)=="u32"?32u:64u;
  const auto bit_type="b"+std::to_string(bits);
  if(k.name!="Bitcast("+bit_type+" -> "+type+")")return;
  const auto shift=nodes_[find(k.children[0])];
  if(shift.kind!=static_cast<int>(s::Kind::Operation) || shift.name!="ShiftLeft" ||
      shift.type!="exact:"+bit_type || !shift.attributes.empty() || !shift.provenance.empty() ||
      shift.children.size()!=2)return;
  const auto cast=nodes_[find(shift.children[0])];
  const auto amount=integer_ranges_[find(shift.children[1])];
  if(cast.kind!=6 || cast.children.size()!=1 || !cast.provenance.empty() || !cast.attributes.empty() ||
      cast.type!=shift.type || cast.name!="Bitcast("+std::string(type)+" -> "+bit_type+")" ||
      !amount || amount->lower!=amount->upper || amount->lower<0 || amount->upper>=bits || amount->upper>=63)return;
  const auto input=find(cast.children[0]);const auto range=integer_ranges_[input];
  if(nodes_[input].type!=k.type || !range)return;
  const auto factor=integer_interval::literal(type,std::to_string(UINT64_C(1)<<amount->lower));
  if(!factor || !integer_interval::apply(integer_interval::Operation::Multiply,*range,*factor))return;
  Key constant;constant.kind=static_cast<int>(s::Kind::Constant);constant.type=k.type;
  constant.name=std::to_string(factor->lower);
  Key result;result.kind=static_cast<int>(s::Kind::Operation);result.type=k.type;
  result.name="Multiply";result.children={input,intern(std::move(constant))};
  rewrite(id,intern(std::move(result)));
}

inline auto Engine::rewriteExactIntegerNumeric(Class id,const Key& k,
    const std::vector<std::vector<Class>>& members) -> void {
  // A local homomorphism only where the cached integer interval proves
  // both no integer overflow and exact floating representation. Unknown
  // tensor integers retain their original rounding conversion.
  if(k.kind!=7 || k.name!="ExactNumericValue" || k.type!="real" ||
      !k.provenance.empty() || k.children.size()!=1)return;
  const auto encoded=nodes_[find(k.children[0])];
  if(encoded.kind!=6 || !encoded.provenance.empty() || !encoded.attributes.empty() ||
      encoded.children.size()!=1)return;
  const auto raw=[](const std::string& t){return t.starts_with("exact:")?t.substr(6):t;};
  const auto target=raw(encoded.type);
  if((target!="f32" && target!="f64") || k.attributes!="encoding="+target)return;
  const auto source_id=find(encoded.children[0]);auto source=nodes_[source_id];
  // An exact identity can add the usable arithmetic head while leaving a
  // bitcast as the union leader. Select an admitted local head, not the
  // arbitrary representative. Cap alternatives; never walk ancestors.
  if(source_id<members.size()) {
    unsigned inspected=0;
    for(auto head:members[source_id]) {
      if(++inspected>8)break;
      const auto& candidate=nodes_[head];
      if(!candidate.attributes.empty() || !candidate.provenance.empty())continue;
      if(candidate.kind==static_cast<int>(s::Kind::Operation) && candidate.children.size()==2 &&
          (candidate.name=="Add" || candidate.name=="Subtract" || candidate.name=="Multiply"))
        {source=candidate;break;}
      if(candidate.kind==6 && candidate.children.size()==1) {
        const auto child=find(candidate.children[0]);const auto& r=integer_ranges_[child];
        if(r && integerConversionRange(raw(nodes_[child].type),raw(candidate.type),candidate.name,*r))
          {source=candidate;break;}
      }
    }
  }
  const auto range=integer_ranges_[source_id];
  const std::int64_t bound=target=="f32"?(INT64_C(1)<<24):(INT64_C(1)<<53);
  if(!range || range->lower< -bound || range->upper>bound ||
      !exactIntegerFloatingValue(raw(source.type),target,encoded.name,range->lower) ||
      !exactIntegerFloatingValue(raw(source.type),target,encoded.name,range->upper))return;
  const auto lift=[&](Class child) {
    child=find(child);Key cast;cast.kind=6;cast.type="exact:f64";
    cast.name="Convert rounding=nearest-even("+raw(nodes_[child].type)+" -> f64)";
    cast.children={child};
    Key value;value.kind=7;value.type="real";value.name="ExactNumericValue";
    value.attributes="encoding=f64";value.children={intern(std::move(cast))};
    return intern(std::move(value));
  };
  if(source.kind==static_cast<int>(s::Kind::Operation) && source.children.size()==2 &&
      source.attributes.empty() && source.provenance.empty() &&
      (source.name=="Add" || source.name=="Subtract" || source.name=="Multiply")) {
    for(auto child:source.children) {
      const auto r=integer_ranges_[find(child)];
      if(!r || r->lower< -(INT64_C(1)<<53) || r->upper>(INT64_C(1)<<53))return;
    }
    Key result;result.kind=static_cast<int>(s::Kind::Operation);result.type="real";
    result.name=source.name;result.attributes="ideal_arithmetic";
    result.children={lift(source.children[0]),lift(source.children[1])};
    rewrite(id,intern(std::move(result)));
  } else if(source.kind==6 && source.children.size()==1 &&
      source.attributes.empty() && source.provenance.empty()) {
    const auto child=find(source.children[0]);const auto original=integer_ranges_[child];
    // integerConversionRange certifies value preservation, not bitcast
    // identity. Matching ranges alone would not certify a conversion.
    if(original && integerConversionRange(raw(nodes_[child].type),raw(source.type),
        source.name,*original) && original->lower==range->lower && original->upper==range->upper)
      rewrite(id,lift(child));
  } else if(source.kind==static_cast<int>(s::Kind::Index))rewrite(id,lift(source_id));
}

inline auto Engine::singletonConstant(const Key& k) -> std::optional<Key> {
  const auto constant=[&](std::string type,std::string name) {
    Key result;result.kind=static_cast<int>(s::Kind::Constant);
    result.type=std::move(type);result.name=std::move(name);return result;
  };
  const auto raw=[](const std::string& t){return t.starts_with("exact:")?t.substr(6):t;};
  // Reference byte offsets are plain u64 ring Add/Multiply, including
  // same-width signed gather literals. Signedness does not change these
  // two fixed-width bit operations. Admit only nonnegative exact singleton
  // operands/results, converting LOCAL evaluator inputs, never the original
  // signed constant class. Overflow, other widths/modifiers remain Unknown.
  if(k.kind==static_cast<int>(s::Kind::Operation) && (k.type=="u64" || k.type=="exact:u64") &&
      k.attributes.empty() && k.provenance.empty() && k.children.size()==2 &&
      (k.name=="Add" || k.name=="Multiply")) {
    std::array<ConstantValue,2> values;
    std::array<integer_interval::Interval,2> ranges;
    bool valid=true,mixed=false;
    for(unsigned n=0;n<2 && valid;++n) {
      const auto& c=nodes_[find(k.children[n])];
      if(c.kind!=static_cast<int>(s::Kind::Constant) || !c.attributes.empty() ||
          !c.provenance.empty() || c.name.size()>32 ||
          (c.type!="u64" && c.type!="exact:u64" && c.type!="s64" && c.type!="exact:s64")){valid=false;break;}
      const auto type=raw(c.type);
      mixed=mixed || type=="s64";
      auto range=integer_interval::literal(type,c.name);
      if(!range || range->lower!=range->upper || range->lower<0){valid=false;break;}
      if(type=="s64")range=integerConversionRange("s64","u64","Convert(s64 -> u64)",*range);
      if(!range){valid=false;break;}
      ranges[n]=*range;values[n]={"u64",std::to_string(range->lower)};
    }
    if(valid && mixed) {
      const auto range=integer_interval::apply(k.name=="Add"?integer_interval::Operation::Add:
          integer_interval::Operation::Multiply,ranges[0],ranges[1]);
      if(range && range->lower==range->upper)
        if(auto value=constantOperation(k.name,"u64",{},values))
          return constant(k.type,std::move(value->literal));
    }
  }
  if(k.kind!=static_cast<int>(s::Kind::Constant))
    if(auto range=integerRange(k);range && range->lower==range->upper)
      return constant(k.type,std::to_string(range->lower));
  if(k.kind==6 && k.provenance.empty() && k.attributes.empty() && k.children.size()==1) {
    const auto child=find(k.children[0]);const auto& range=integer_ranges_[child];
    const auto& source=nodes_[child];
    if(source.kind==static_cast<int>(s::Kind::Constant) && source.provenance.empty() &&
        source.attributes.empty() &&
        k.name=="Bitcast("+raw(source.type)+" -> "+raw(k.type)+")")
      if(auto value=constantValue(raw(source.type),source.name,k.name,raw(k.type)))
        return constant(k.type,std::move(value->literal));
    if(range && range->lower==range->upper)
      if(auto value=exactIntegerFloatingValue(raw(nodes_[child].type),raw(k.type),
                                             k.name,range->lower))
        return constant(k.type,std::move(value->literal));
  }
  // A projected opaque conversion can become an exact constant after a
  // range proof. Consult only its immediate head and cached integer facts;
  // never search arbitrary class members/ancestors or erase a symbolic cast.
  if(k.kind==7 && k.type=="real" && k.name=="ExactNumericValue" &&
      k.provenance.empty() && k.children.size()==1) {
    const auto& encoded=nodes_[find(k.children[0])];const auto type=raw(encoded.type);
    if(encoded.type.starts_with("exact:") && (type=="f32" || type=="f64") &&
        k.attributes=="encoding="+type && encoded.provenance.empty() && encoded.attributes.empty()) {
      if(encoded.kind==static_cast<int>(s::Kind::Constant)) {
        if(auto value=constantValue(type,encoded.name))return constant("real",std::move(value->literal));
      } else if(encoded.kind==6)
        if(auto value=singletonConstant(encoded))return constant("real",std::move(value->name));
    }
  }
  return {};
}

inline auto Engine::refineIntegerRange(Class root,const integer_interval::Result& proof) -> bool {
  if(!proof)return false;
  auto& current=integer_ranges_[root];
  auto narrowed=*proof;
  if(current) {
    if(current->type!=proof->type)throw std::logic_error("inconsistent integer range type");
    narrowed.lower=std::max(current->lower,proof->lower);
    narrowed.upper=std::min(current->upper,proof->upper);
    if(narrowed.lower>narrowed.upper)throw std::logic_error("contradictory integer range proof");
    if(narrowed==*current)return false;
  }
  current=narrowed;return true;
}

inline auto Engine::provesZero(const Key& k) -> bool {
  if(!realType(k.type) || !k.provenance.empty())return false;
  if(k.kind==static_cast<int>(s::Kind::Constant))
    return k.name=="0" && k.attributes.empty();
  // Computed zero is an ideal-numeric fact, not a raw signed-zero or FTZ
  // bit-pattern fact. Constants retain the existing typed-zero behavior.
  if(k.type!="real")return false;
  if(k.kind==static_cast<int>(s::Kind::Fold) && k.fold && k.children.size()==2 &&
      k.attributes.empty() && nodes_[find(k.children[0])].type==k.type) {
    const auto seed=find(k.children[0]);
    // Empty Fold is its ACTUAL seed, never an invented neutral value.
    if(!k.fold->count)return zero_[seed];
    const auto body=find(k.children[1]);
    return k.fold->reducer==s::Reducer::Sum && nodes_[body].type==k.type &&
        zero_[seed] && zero_[body] && finite_[seed] && finite_[body];
  }
  if(arithmetic(k) && k.name=="Multiply" &&
      nodes_[find(k.children[0])].type==k.type && nodes_[find(k.children[1])].type==k.type) {
    const auto a=find(k.children[0]),b=find(k.children[1]);
    return (zero_[a] && finite_[b]) || (zero_[b] && finite_[a]);
  }
  return false;
}

inline auto Engine::provesNonzero(const Key& k,bool positive) -> bool {
  if(positive)return true;
  if(!realType(k.type))return false;
  if(k.kind==static_cast<int>(s::Kind::Constant)) {
    if(!k.attributes.empty() || !k.provenance.empty())return false;
    double value=0;const auto parsed=floatFromChars(k.name.data(),k.name.data()+k.name.size(),value);
    return parsed.ec==std::errc{} && parsed.ptr==k.name.data()+k.name.size() && std::isfinite(value) && value!=0;
  }
  if(idealSelect(k)) {
    const auto condition=predicate_[find(k.children[0])];
    if(condition)return nonzero_[find(k.children[condition==1?1:2])];
    return nonzero_[find(k.children[1])] && nonzero_[find(k.children[2])];
  }
  if(!idealOperation(k))return false;
  if(k.name=="Sqrt") {
    const auto child=find(k.children[0]);
    return finite_[child] && nonnegative_[child] && nonzero_[child];
  }
  if(k.name=="Negate" || k.name=="Reciprocal")return nonzero_[find(k.children[0])];
  if(k.name=="Multiply" || k.name=="Divide")
    return nonzero_[find(k.children[0])] && nonzero_[find(k.children[1])];
  return false;
}

inline auto Engine::updateProperties(Class id) -> bool {
  ++statistics_.property_evaluations;
  const auto root=find(id);const auto& k=nodes_[id];bool changed=false;
  changed=refineIntegerRange(root,integerRange(k));
  const auto add=[&](std::vector<bool>& values,bool proof) {
    if(proof && !values[root]){values[root]=true;changed=true;}
  };
  add(finite_,provesFinite(k));
  const auto [nonnegative,positive]=positivity(k);
  add(nonnegative_,nonnegative);add(positive_,positive);
  add(nonzero_,provesNonzero(k,positive));
  add(zero_,provesZero(k));
  add(negative_infinity_,infinityConstant(k,true));add(positive_infinity_,infinityConstant(k,false));
  const auto predicate=provesPredicate(k,id);
  if(predicate && !predicate_[root]){predicate_[root]=predicate;changed=true;}
  else if(predicate && predicate_[root]!=predicate)throw std::logic_error("contradictory predicate proof");
  const auto [maximum,fold]=scopeProperties(k);
  if(maximum>max_index_[root]){max_index_[root]=maximum;changed=true;}
  add(has_fold_,fold);
  std::vector<std::uint32_t> free;
  if(k.kind==static_cast<int>(s::Kind::Index))free.push_back(k.index_level);
  for(std::size_t i=0;i<k.children.size();++i) {
    if(branchWrapper(k) && i==1)continue; // Captured proof metadata has its own index namespace.
    for(auto level:free_indices_[find(k.children[i])]) {
      if(k.kind==static_cast<int>(s::Kind::Fold) && i==1 && level==k.fold->binder_level)continue;
      free.push_back(level);
    }
  }
  free.insert(free.end(),free_indices_[root].begin(),free_indices_[root].end());
  std::sort(free.begin(),free.end());free.erase(std::unique(free.begin(),free.end()),free.end());
  if(free!=free_indices_[root]){free_indices_[root]=std::move(free);changed=true;}
  if(changed){++statistics_.property_updates;notifyDependents(root);}
  return changed;
}

inline auto Engine::canonicalizeCommutative(Key& key) -> void {
  const bool integer_operation=integerCommutative(key) &&
      nodes_[find(key.children[0])].type==key.type && nodes_[find(key.children[1])].type==key.type;
  const bool real_operation=arithmetic(key) && finite_[find(key.children[0])] && finite_[find(key.children[1])];
  if((integer_operation || real_operation) && minimum_[key.children[1]]<minimum_[key.children[0]])
    std::swap(key.children[0],key.children[1]);
}

inline auto Engine::exactIntegerIdentity(const Key& k) const -> std::optional<Class> {
  // Shared fixed-head bit identities. No floating/rounded conversion is
  // discarded, no descendants are inspected and no interval is invented.
  if(k.kind!=static_cast<int>(s::Kind::Operation) ||
      (k.type!="u64" && k.type!="exact:u64") || !k.attributes.empty() ||
      !k.provenance.empty() || k.fold || k.children.size()!=2 ||
      nodes_[k.children[0]].type!=k.type || nodes_[k.children[1]].type!=k.type ||
      !nodes_[k.children[0]].provenance.empty() || !nodes_[k.children[1]].provenance.empty())return {};
  const auto literal=[&](Class id,std::string_view value) {
    const auto& c=nodes_[id];
    return c.kind==static_cast<int>(s::Kind::Constant) && c.type==k.type && c.name==value &&
        c.attributes.empty() && c.provenance.empty() && c.children.empty() && !c.fold;
  };
  if(k.name=="Divide" && literal(k.children[1],"1"))return k.children[0];
  if(k.name=="Add" || k.name=="Multiply")for(unsigned side=0;side<2;++side)
    if(literal(k.children[side],k.name=="Add"?"0":"1"))return k.children[1-side];
  return {};
}

inline auto Engine::intern(Key k) -> Class {
  poll();
  for (auto& child : k.children) child = find(child);
  if(k.name!="Compare")if(auto predicate=comparison(k)) {
    k.name="Compare";k.attributes="predicate="+*predicate;
  }
  canonicalizeCommutative(k);
  if (idealOperation(k) || idealSelect(k)) k.attributes = "ideal_arithmetic";
  // Exact interval singletons are canonical before projection too, so their
  // numeric/bit/address uses all retain the same typed constant semantics.
  if(auto singleton=singletonConstant(k))k=std::move(*singleton);
  if(auto identity=exactIntegerIdentity(k))return *identity;
  // Bounded local identities are canonical at construction time. Keeping
  // x+0 as another member of x's class creates self-referential alternatives
  // which reassociation/sign/exponential rules could expand indefinitely.
  // Consult only cached facts and one immediate head, never ancestors.
  if(idealOperation(k)) {
    const auto literal=[&](Class id,const char* value) {
      Key key;key.kind=static_cast<int>(s::Kind::Constant);key.type=k.type;key.name=value;
      auto found=memo_.find(key);return found!=memo_.end() && find(found->second)==id;
    };
    if(k.children.size()==2)for(unsigned side=0;side<2;++side) {
      auto a=k.children[side],b=k.children[1-side];
      if(finite_[b] && ((k.name=="Add" && zero_[a]) ||
                       (k.name=="Multiply" && literal(a,"1"))))return b;
      if(k.name=="Multiply" && zero_[a] && finite_[b])return a;
    }
    if(k.name=="Negate" && finite_[k.children[0]]) {
      auto child=k.children[0];const auto& head=nodes_[child];
      if(zero_[child])return child;
      if(idealOperation(head) && head.name=="Negate")return find(head.children[0]);
    }
    if(k.name=="Add")for(unsigned side=0;side<2;++side) {
      const auto& head=nodes_[k.children[side]];
      if(idealOperation(head) && head.name=="Negate" &&
          find(head.children[0])==k.children[1-side] && finite_[k.children[1-side]]) {
        Key zero;zero.kind=static_cast<int>(s::Kind::Constant);zero.type=k.type;zero.name="0";
        return intern(std::move(zero));
      }
    }
    if((k.name=="Exp2" || k.name=="Exp") && zero_[k.children[0]]) {
      Key one;one.kind=static_cast<int>(s::Kind::Constant);one.type=k.type;one.name="1";
      return intern(std::move(one));
    }
    if(k.name=="Power" && positive_[k.children[0]] && finite_[k.children[0]] &&
        zero_[k.children[1]]) {
      Key one;one.kind=static_cast<int>(s::Kind::Constant);one.type=k.type;one.name="1";
      return intern(std::move(one));
    }
  }
  if (const auto it = memo_.find(k); it != memo_.end()) {
    return find(it->second);
  }
  if (nodes_.size() >= options_.max_nodes) throw Limit{"node_budget"};
  const auto id = nodes_.size();
  // Only canonical memo MISSES register a physical source. A partially
  // denied dormant bootstrap may lag the arena; its next preparation will
  // resume that prefix rather than fabricating slots or resetting work.
  if(complete_headers_ && complete_headers_->sources.size()==id)
    complete_headers_->registerSource(id,k);
  if(sentryward::debug::environment("SENTRYWARD_EGRAPH_TRACE_NODES") && id<256) {
    SENTRYWARD_DEBUG_PRINT(egraph_151);
    for(auto child:k.children)SENTRYWARD_DEBUG_PRINT(egraph_152);
    SENTRYWARD_DEBUG_PRINT(egraph_153);
  }
  const auto [maximum,fold]=scopeProperties(k);
  bool sat_sensitive=scopedSatHead(k,nodes_);
  for(auto child:k.children)sat_sensitive=sat_sensitive || projection_sat_sensitive_[child];
  projection_sat_sensitive_.push_back(sat_sensitive);
  bool integer_sensitive=scopedIndexRemainderHead(k,nodes_);
  for(auto child:k.children)integer_sensitive=integer_sensitive || projection_integer_sensitive_[child];
  projection_integer_sensitive_.push_back(integer_sensitive);
  bool branch_sensitive=false;
  if(options_.certified_branch_scope) {
    branch_sensitive=branchPredicateSource(k,nodes_);
    for(auto child:k.children)branch_sensitive=branch_sensitive || projection_branch_sensitive_[child];
    const bool candidate=branchWrapper(k).has_value() ||
        (k.kind==7 && k.type=="pred" && k.name=="ScopedPredicateAnd") ||
        (k.kind==static_cast<int>(s::Kind::Fold) && k.type=="real" && k.fold && k.fold->reducer==s::Reducer::Sum);
    if(candidate && branch_candidates_.size()<2048 && branchCharge())branch_candidates_.push_back(id);
  }
  projection_branch_sensitive_.push_back(branch_sensitive);
  if(projection_scopes_)projection_source_ids_.push_back(id);
  nodes_.push_back(k); parents_.push_back(id); sizes_.push_back(1);minimum_.push_back(id);finite_.push_back(false);
  class_members_.append(id);
  // Structural indexes only. Every finite/attribute/provenance/rank/seed
  // admission remains epoch-local; an opaque real head cannot become a fact
  // merely by appearing in this ascending append-only ID catalog.
  if(realType(k.type))real_head_ids_.push_back(id);
  if(k.kind==static_cast<int>(s::Kind::Fold))fold_head_ids_.push_back(id);
  max_index_.push_back(maximum);has_fold_.push_back(fold);
  nonnegative_.push_back(false);positive_.push_back(false);nonzero_.push_back(false);predicate_.push_back(0);
  zero_.push_back(false);negative_infinity_.push_back(false);positive_infinity_.push_back(false);
  integer_ranges_.emplace_back();
  free_indices_.emplace_back();dependents_.emplace_back();property_queued_.push_back(false);
  for(auto child:k.children)dependents_[child].push_back(id);
  memo_.emplace(std::move(k), id);
  updateProperties(id);
  return id;
}

inline auto Engine::unite(Class a, Class b) -> bool {
  a = find(a); b = find(b);
  if (a == b) return false;
  if (nodes_[a].type != nodes_[b].type) throw std::logic_error("e-class result types differ");
  if (sizes_[a] < sizes_[b]) std::swap(a, b);
  parents_[b] = a; sizes_[a] += sizes_[b];finite_[a]=finite_[a] || finite_[b];
  minimum_[a]=std::min(minimum_[a],minimum_[b]);
  class_members_.splice(a,b);
  max_index_[a]=std::max(max_index_[a],max_index_[b]);has_fold_[a]=has_fold_[a] || has_fold_[b];
  nonnegative_[a]=nonnegative_[a] || nonnegative_[b];
  positive_[a]=positive_[a] || positive_[b];
  nonzero_[a]=nonzero_[a] || nonzero_[b];zero_[a]=zero_[a] || zero_[b];
  negative_infinity_[a]=negative_infinity_[a] || negative_infinity_[b];
  positive_infinity_[a]=positive_infinity_[a] || positive_infinity_[b];
  refineIntegerRange(a,integer_ranges_[b]);
  if(predicate_[a] && predicate_[b] && predicate_[a]!=predicate_[b])
    throw std::logic_error("contradictory predicate merge");
  if(!predicate_[a])predicate_[a]=predicate_[b];
  auto& indices=free_indices_[a];indices.insert(indices.end(),free_indices_[b].begin(),free_indices_[b].end());
  std::sort(indices.begin(),indices.end());indices.erase(std::unique(indices.begin(),indices.end()),indices.end());
  dependents_[a].splice(dependents_[a].end(),dependents_[b]);
  notifyDependents(a);
  ++unions_;
  return true;
}

inline auto Engine::rebuild() -> void {
  bool changed;
  do {
    changed = false;
    memo_.clear();
    for (std::size_t id = 0; id < nodes_.size(); ++id) {
      poll();
      auto& key = nodes_[id];
      for (auto& child : key.children) child = find(child);
      canonicalizeCommutative(key);
      const auto [it, inserted] = memo_.emplace(key,id);
      if (!inserted && unite(id,it->second)) {
        ++statistics_.congruence_unions; changed = true;
      }
    }
    propagateProperties();
  } while (changed);
}

} // namespace
} // namespace sentryward::equivalence
