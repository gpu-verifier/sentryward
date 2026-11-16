#pragma once
#include "sentryward/algorithm/base/analysis_scope.h"
#include "sentryward/algorithm/diagnostics/development_trace.h"
#include "sentryward/equivalence/egraph.h"
#include "sentryward/equivalence/float_from_chars.h"
#include "sentryward/equivalence/constant_values.h"
#include "sentryward/equivalence/conversion_uses.h"
#include "sentryward/equivalence/polynomial_certificate.h"
#include "sentryward/equivalence/integer_interval.h"
#include "sentryward/equivalence/bounded_linear_reduction.h"
#include "sentryward/equivalence/bounded_index_translation.h"
#include "sentryward/equivalence/moment_certificate.h"
#include "sentryward/equivalence/numeric_semantics.h"
#include "sentryward/equivalence/bit_origins.h"
#include "sentryward/equivalence/lexical_scope.h"
#include "sentryward/equivalence/scoped_body_relation.h"
#include "sentryward/equivalence/joined_body_relation.h"
#include "sentryward/equivalence/scoped_product_relation.h"
#include "sentryward/equivalence/finite_prefix_certificate.h"
#include "sentryward/equivalence/epoch_work_allowance.h"
#include "sentryward/equivalence/causal_region_certificate.h"
#include "sentryward/equivalence/certified_branch_scope.h"
#include "sentryward/equivalence/stored_class_catalog.h"
#include "sentryward/equivalence/scoped_work_budget.h"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <functional>
#include <limits>
#include <list>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <type_traits>
#include <variant>
#include <vector>

namespace sentryward::equivalence {
namespace {
namespace s = scalar_fold;
using Class = std::size_t;

bool boundedScopedRef(const s::Ref& ref) {
  return ref.conversion.size()<=4096 && ref.converted_type.size()<=128 && ref.pointer_provenance.size()<=4096;
}
struct ExtractedScopedBody {s::Ref body;std::vector<s::Id> folds;};
ExtractedScopedBody extractScopedBody(const s::Graph& graph,const ScopedBodyPath& path) {
  if(path.edges.size()>32)throw std::invalid_argument("scoped_path_cap");
  ExtractedScopedBody out{path.wrapper,{}};
  for(auto edge:path.edges) {
    const auto& node=graph.node(out.body);
    if(node.operands.size()>64 || edge>=node.operands.size())throw std::invalid_argument("invalid_scoped_path");
    if(node.kind==s::Kind::Fold && edge==1)out.folds.push_back(out.body.node);
    if(!boundedScopedRef(node.operands[edge]))throw std::invalid_argument("scoped_metadata_cap");
    out.body=node.operands[edge];
  }
  return out;
}

struct Limit { const char* reason; };

// Exact affine arithmetic in the u64 ring; each lexical index is independent.
struct IndexAffine {
  std::uint64_t constant = 0;
  std::map<std::uint32_t,std::uint64_t> coefficients;
};
std::optional<IndexAffine> combineAffine(const std::string& operation,
                                       IndexAffine a, IndexAffine b) {
  if(operation=="Multiply") {
    if(!a.coefficients.empty() && !b.coefficients.empty())return {};
    if(a.coefficients.empty())std::swap(a,b);
    a.constant*=b.constant;
    for(auto& [index,coefficient]:a.coefficients)coefficient*=b.constant;
  } else if(operation=="Add" || operation=="Subtract") {
    const auto sign=operation=="Add"?std::uint64_t{1}:UINT64_MAX;
    a.constant+=sign*b.constant;
    for(auto [index,coefficient]:b.coefficients)a.coefficients[index]+=sign*coefficient;
  } else return {};
  std::erase_if(a.coefficients,[](const auto& entry){return entry.second==0;});
  return a;
}
bool shiftedAffine(const IndexAffine& first,const IndexAffine& next,
                   std::uint32_t binder,std::uint64_t shift) {
  if(first.coefficients!=next.coefficients)return false;
  const auto found=first.coefficients.find(binder);
  const auto coefficient=found==first.coefficients.end()?0:found->second;
  return first.constant+coefficient*shift==next.constant;
}

struct Key {
  int kind = 0;
  std::string type, name, attributes, provenance;
  std::optional<s::FoldInfo> fold;
  std::uint32_t index_level = 0;
  std::vector<Class> children;
  bool operator==(const Key&) const = default;
};

struct Hash {
  std::size_t operator()(const Key& k) const {
    std::size_t h = k.kind;
    const auto add = [&](std::size_t v) { h ^= v + 0x9e3779b9 + (h << 6) + (h >> 2); };
    for (const auto* v : {&k.type, &k.name, &k.attributes, &k.provenance})
      add(std::hash<std::string>{}(*v));
    add(k.index_level);
    add(k.fold.has_value());
    if (k.fold) {
      add(static_cast<std::size_t>(k.fold->reducer));
      add(k.fold->binder_level);
      add(std::hash<std::uint64_t>{}(k.fold->count));
    }
    for (auto child : k.children) add(child);
    return h;
  }
};

bool realType(const std::string& t) {
  return t == "real" || exactFloatingWidening(t,t);
}
std::optional<bool> floatingInfinity(const std::string& type,const std::string& literal) {
  // Some real-valued encodings (e2m1, ue8m0, ue4m3, e4m3) have no infinity.
  // A spelling unsupported by its declared format is not a mathematical fact.
  if(type!="real" && type!="f16" && type!="bf16" && type!="f32" &&
      type!="f64" && type!="e5m2")return {};
  if(literal.starts_with("0f") || literal.starts_with("0d") || literal.starts_with("0x")) {
    // Decode while the original floating type is available. Inferring a type
    // from an untyped real hex literal would accept a wrong-width source after
    // numerical projection erased its f32/f64 type. Bitcast preserves all bits
    // (including nonfinite values); ordinary constantValue remains finite-only.
    if(type!="f32" && type!="f64")return {};
    const auto width=type=="f32"?32U:64U;
    if(literal.size()!=2+width/4)return {};
    const std::string target=width==32?"u32":"u64";
    auto encoded=constantValue(type,literal,"Bitcast("+type+" -> "+target+")",target);
    if(!encoded)return {};
    std::uint64_t bits=0;
    const auto parsed=std::from_chars(encoded->literal.data(),
        encoded->literal.data()+encoded->literal.size(),bits);
    if(parsed.ec!=std::errc{} || parsed.ptr!=encoded->literal.data()+encoded->literal.size())return {};
    const auto sign=std::uint64_t{1}<<(width-1);
    const auto infinity=width==32?std::uint64_t{0x7f800000}:std::uint64_t{0x7ff0000000000000};
    // Fraction bits must ALL be zero: NaN encodings are not infinities.
    if((bits&~sign)!=infinity)return {};
    return bool(bits&sign);
  }
  double value=0;
  const auto parsed=floatFromChars(literal.data(),literal.data()+literal.size(),value);
  if(parsed.ec!=std::errc{} || parsed.ptr!=literal.data()+literal.size() || !std::isinf(value))return {};
  return std::signbit(value);
}
bool infinityConstant(const Key& key,bool negative) {
  if(key.kind!=static_cast<int>(s::Kind::Constant) || !key.provenance.empty() ||
      !key.attributes.empty())return false;
  const auto decoded=floatingInfinity(key.type,key.name);
  return decoded && *decoded==negative;
}
bool arithmetic(const Key& k) {
  return k.kind == static_cast<int>(s::Kind::Operation) && realType(k.type) &&
      (k.attributes.empty() || k.attributes == "ideal_arithmetic") &&
      (k.name == "Add" || k.name == "Multiply" || k.name == "Maximum" || k.name == "Minimum") &&
      k.children.size() == 2;
}

bool integerCommutative(const Key& k) {
  const auto type=k.type.starts_with("exact:")?k.type.substr(6):k.type;
  const bool integer_type=type=="u8" || type=="u16" || type=="u32" || type=="u64" ||
      type=="s8" || type=="s16" || type=="s32" || type=="s64" ||
      type=="b8" || type=="b16" || type=="b32" || type=="b64";
  return integer_type && k.kind==static_cast<int>(s::Kind::Operation) &&
      k.attributes.empty() && k.provenance.empty() && k.children.size()==2 &&
      (k.name=="Add" || k.name=="Multiply" || k.name=="BitAnd" ||
       k.name=="BitOr" || k.name=="BitXor");
}

// Admit only ordinary primitive PTX math in the explicit ideal-real projection.
// These are the same arithmetic modifiers admitted for rcp by scalar lowering;
// saturation, NaN controls, carry state and unknown modifiers remain exact.
// This does not recognize software polynomials or numerical bit reinterpretation.
std::optional<std::string_view> legacyIdealPrimitive(const Key& k,
                                                  std::string_view operand_type) {
  if(k.kind!=static_cast<int>(s::Kind::Operation) || k.children.size()!=1 ||
      !k.provenance.empty())return {};
  return sentryward::equivalence::legacyIdealPrimitive(k.name,k.type,k.attributes,operand_type);
}

bool idealOperation(const Key& k,bool numeric_abstraction=false) {
  if(k.kind!=static_cast<int>(s::Kind::Operation) || !realType(k.type))return false;
  if(k.attributes!="" && k.attributes!="ideal_arithmetic" &&
      !(k.name=="Exp2" && k.attributes=="semantics=ideal")) {
    if(!numeric_abstraction)return false;
    auto text=k.attributes;
    while(!text.empty()) {
      const auto end=text.find(' ');const auto token=text.substr(0,end);
      if(token!="rounding=nearest-even" && token!="rounding=zero" &&
          token!="rounding=negative-infinity" && token!="rounding=positive-infinity" &&
          token!="flush_to_zero")return false;
      if(end==std::string::npos)break;text.erase(0,end+1);
    }
  }
  return idealScalarOperation(k.name,k.children.size());
}

// Predicates retain their PTX comparison attribute. Unknown attributes do not
// license real-number reasoning (notably unordered/NaN behavior).
std::optional<std::string> comparison(const Key& k) {
  if(k.kind!=static_cast<int>(s::Kind::Operation) || k.children.size()!=2 ||
      (k.type!="pred" && k.type!="exact:pred") || !k.provenance.empty())return {};
  if(k.name!="Compare") {
    if(!k.attributes.empty())return {};
    for(const auto& [name,predicate]:std::initializer_list<std::pair<std::string_view,std::string_view>>{
        {"Less","lt"},{"LessEqual","le"},{"Greater","gt"},{"GreaterEqual","ge"},
        {"Equal","eq"},{"NotEqual","ne"}})
      if(k.name==name)return std::string(predicate);
    return {};
  }
  std::optional<std::string> result;
  auto text=k.attributes;
  while(!text.empty()) {
    const auto end=text.find(' ');const auto token=text.substr(0,end);
    if(token.starts_with("predicate=") && !result)result=token.substr(10);
    else if(token!="unordered_true")return {};
    if(end==std::string::npos)break;text.erase(0,end+1);
  }
  if(result && (*result=="eq" || *result=="ne" || *result=="lt" ||
                *result=="le" || *result=="gt" || *result=="ge"))return result;
  return {};
}
bool idealSelect(const Key& k) {
  return k.kind==static_cast<int>(s::Kind::Operation) && k.name=="Select" &&
    realType(k.type) && k.children.size()==3 && k.provenance.empty() &&
    (k.attributes.empty() || k.attributes=="ideal_arithmetic");
}

// Truncating to n bits discards any additive multiple of 2^n.
std::optional<s::Ref> packedLowUse(const s::Graph& graph, const s::Ref& ref) {
  if (!ref.pointer_provenance.empty()) return {};
  const auto& node = graph.node(ref);
  if (node.kind != s::Kind::Operation || node.name != "Add" ||
      !node.attributes.empty() || node.operands.size() != 2) return {};
  unsigned width = 0, low = 0;
  for (unsigned candidate : {8u, 16u, 32u, 64u})
    if (node.type == "u" + std::to_string(candidate)) width = candidate;
  if (!width) return {};
  for (unsigned candidate : {8u, 16u, 32u}) {
    if (candidate >= width) continue;
    for(const auto* operation : {"Convert (","Convert("}) {
      const auto prefix = operation + node.type + " -> u" + std::to_string(candidate) + ")";
      if (ref.conversion == prefix || ref.conversion.starts_with(prefix + "; "))
        low = candidate;
    }
  }
  if (!low) return {};
  for (unsigned side = 0; side < 2; ++side) {
    const auto& high = node.operands[side];
    const auto& retained = node.operands[1-side];
    if (!high.conversion.empty() || !high.pointer_provenance.empty() ||
        !retained.pointer_provenance.empty() || graph.type(retained) != node.type) continue;
    const auto& product = graph.node(high);
    if (product.kind != s::Kind::Operation || product.name != "Multiply" ||
        product.type != node.type || !product.attributes.empty() ||
        product.operands.size() != 2) continue;
    for (unsigned factor = 0; factor < 2; ++factor) {
      const auto& operand = product.operands[factor];
      const auto& other = product.operands[1-factor];
      if (!operand.pointer_provenance.empty() || !other.pointer_provenance.empty() ||
          graph.type(other) != node.type) continue;
      const auto& constant = graph.node(operand);
      if (constant.kind != s::Kind::Constant) continue;
      const auto value = constantValue(constant.type, constant.name,
                                       operand.conversion, operand.converted_type);
      if (!value || value->type != node.type) continue;
      std::uint64_t bits = 0;
      const auto [end, error] = std::from_chars(
          value->literal.data(), value->literal.data()+value->literal.size(), bits);
      if (error != std::errc{} || end != value->literal.data()+value->literal.size() ||
          (bits & ((std::uint64_t{1} << low)-1))) continue;
      auto result = retained;
      if (!result.conversion.empty()) result.conversion += "; ";
      result.conversion += ref.conversion;
      result.converted_type = ref.converted_type;
      return result;
    }
  }
  return {};
}


// A zero-extended field cannot carry into the adjacent packed field.
bool unsignedFits(const s::Graph& graph,const s::Ref& ref,unsigned bits) {
  if(!ref.pointer_provenance.empty() || !bits || bits>=64)return false;
  const auto type=graph.type(ref);
  if(type.empty() || type.front()!='u')return false;
  const auto& node=graph.node(ref);
  if(node.kind==s::Kind::Constant) {
    if(auto value=constantValue(node.type,node.name,ref.conversion,ref.converted_type)) {
      std::uint64_t integer=0;
      const auto parsed=std::from_chars(value->literal.data(),value->literal.data()+value->literal.size(),integer);
      if(parsed.ec==std::errc{} && parsed.ptr==value->literal.data()+value->literal.size())
        return integer<(UINT64_C(1)<<bits);
    }
  }
  auto normalized=normalizeConversions(node.type,type,ref.conversion);
  if(!normalized)return false;
  for(unsigned width:{8u,16u,32u})if(width<=bits) {
    const auto source="u"+std::to_string(width);
    if(type==source)return true;
    const auto step="Convert("+source+" -> "+type+")";
    if(*normalized==step || normalized->ends_with("; "+step))return true;
  }
  return false;
}
std::optional<s::Ref> packedHighUse(const s::Graph& graph,const s::Ref& ref) {
  if(!ref.pointer_provenance.empty())return {};
  const auto& shift=graph.node(ref);
  if(shift.kind!=s::Kind::Operation || shift.name!="ShiftRight" ||
      !shift.attributes.empty() || shift.operands.size()!=2)return {};
  unsigned width=0;
  for(unsigned candidate:{16u,32u,64u})if(shift.type=="u"+std::to_string(candidate))width=candidate;
  if(!width)return {};
  const auto constant=[&](const s::Ref& value)->std::optional<std::uint64_t> {
    const auto& node=graph.node(value);
    if(!value.pointer_provenance.empty() || node.kind!=s::Kind::Constant)return {};
    auto decoded=constantValue(node.type,node.name,value.conversion,value.converted_type);
    if(!decoded)return {};
    std::uint64_t result=0;
    const auto parsed=std::from_chars(decoded->literal.data(),decoded->literal.data()+decoded->literal.size(),result);
    if(parsed.ec!=std::errc{} || parsed.ptr!=decoded->literal.data()+decoded->literal.size())return {};
    return result;
  };
  auto amount=constant(shift.operands[1]);
  if(!amount || !*amount || *amount>=width)return {};
  const auto& packed=shift.operands[0];
  if(!packed.conversion.empty() || !packed.pointer_provenance.empty())return {};
  const auto& add=graph.node(packed);
  if(add.kind!=s::Kind::Operation || add.name!="Add" || add.type!=shift.type ||
      !add.attributes.empty() || add.operands.size()!=2)return {};
  for(unsigned side=0;side<2;++side) {
    const auto& low=add.operands[1-side];const auto& product=add.operands[side];
    if(graph.type(low)!=shift.type || !unsignedFits(graph,low,*amount) ||
        !product.conversion.empty() || !product.pointer_provenance.empty())continue;
    const auto& mul=graph.node(product);
    if(mul.kind!=s::Kind::Operation || mul.name!="Multiply" || mul.type!=shift.type ||
        !mul.attributes.empty() || mul.operands.size()!=2)continue;
    for(unsigned factor=0;factor<2;++factor) {
      const auto power=constant(mul.operands[factor]);
      const auto& high=mul.operands[1-factor];
      if(!power || *power!=(UINT64_C(1)<<*amount) ||
          graph.type(mul.operands[factor])!=shift.type || graph.type(high)!=shift.type ||
          !unsignedFits(graph,high,width-*amount))continue;
      auto result=high;
      if(!ref.conversion.empty()) {
        if(!result.conversion.empty())result.conversion+="; ";
        result.conversion+=ref.conversion;result.converted_type=ref.converted_type;
      }
      return result;
    }
  }
  return {};
}

// Syntax/type continuity only: opaque tail operations keep their semantics.
// The byte cap makes every substring copy and head scan bounded metadata work.
bool completeBoundedConversionUse(const std::string& source,const s::Ref& ref) {
  if(source.empty() || source.size()>128 || ref.conversion.empty() ||
      ref.conversion.size()>128 || ref.converted_type.empty() || ref.converted_type.size()>128)return false;
  auto current=source,remaining=ref.conversion;
  while(!remaining.empty()) {
    auto step=leadingConversion(current,remaining);if(!step)return false;
    current=std::move(step->target);remaining=std::move(step->remainder);
  }
  return current==ref.converted_type;
}

// A final unsigned narrowing observes only [shift,shift+bits). Work is over
// at most four packing/widening spine heads, never a producer/bit-origin DAG.
std::optional<s::Ref> packedShiftedFieldUse(s::Graph& graph,const s::Ref& ref) {
  const auto clean=[&](const s::Ref& r) {
    if(!r.pointer_provenance.empty() || r.conversion.size()>128 || r.converted_type.size()>128)return false;
    if(r.conversion.empty())return r.converted_type.empty();
    return completeBoundedConversionUse(graph.node(r).type,r);
  };
  const auto unsignedWidth=[](const std::string& type)->unsigned {
    for(unsigned bits:{8u,16u,32u,64u})if(type=="u"+std::to_string(bits))return bits;
    return 0;
  };
  const auto constant=[&](const s::Ref& r)->std::optional<std::uint64_t> {
    if(!clean(r) || !r.conversion.empty())return {};
    const auto& n=graph.node(r);
    if(n.kind!=s::Kind::Constant || !n.attributes.empty() || n.fold || !n.operands.empty() ||
        n.name.size()>128 || (!unsignedWidth(n.type) && n.type!="b32" && n.type!="b64"))return {};
    auto decoded=constantValue(n.type,n.name);if(!decoded)return {};
    std::uint64_t value=0;
    const auto result=std::from_chars(decoded->literal.data(),decoded->literal.data()+decoded->literal.size(),value);
    if(result.ec!=std::errc{} || result.ptr!=decoded->literal.data()+decoded->literal.size())return {};
    return value;
  };
  if(!clean(ref) || ref.conversion.empty())return {};
  const auto& root=graph.node(ref);auto width=unsignedWidth(root.type);
  if(!width || root.kind!=s::Kind::Operation || root.name!="ShiftRight" || root.fold ||
      !root.attributes.empty() || root.operands.size()!=2 || !clean(root.operands[0]) ||
      graph.type(root.operands[0])!=root.type)return {};
  auto type=root.type;
  auto narrowing=leadingConversion(type,ref.conversion);
  if(!narrowing || narrowing->operation!="Convert")return {};
  const auto bits=unsignedWidth(narrowing->target);auto amount=constant(root.operands[1]);
  if(!bits || bits>=width || !amount || *amount>=width || bits>width-*amount ||
      (narrowing->remainder.empty() && graph.type(ref)!=narrowing->target))return {};
  // Copy all source/use fields before appending anything to the graph.
  auto value=root.operands[0];const auto amount_type=graph.node(root.operands[1]).type;
  const auto target=ref.converted_type;const auto suffix=narrowing->remainder;
  unsigned shift=static_cast<unsigned>(*amount);bool changed=false;
  for(unsigned visited=0;visited<4;++visited) {
    if(!clean(value) || graph.type(value)!=type)break;
    const auto& n=graph.node(value);
    if(!n.attributes.empty() || n.fold)break;
    if(!value.conversion.empty()) {
      const auto source_width=unsignedWidth(n.type);
      auto normalized=normalizeConversions(n.type,type,value.conversion);
      if(!source_width || source_width>=width || shift+bits>source_width || !normalized ||
          *normalized!="Convert("+n.type+" -> "+type+")")break;
      // The observed window excludes every zero-fill bit of this widening.
      value=graph.ref(value.node);type=n.type;width=source_width;changed=true;continue;
    }
    if(n.kind!=s::Kind::Operation || n.name!="Add" || n.type!=type || n.operands.size()!=2)break;
    bool peeled=false;
    for(unsigned side=0;side<2 && !peeled;++side) {
      const auto& retained=n.operands[1-side];const auto& product=n.operands[side];
      if(!clean(retained) || !clean(product) || !product.conversion.empty() ||
          graph.type(retained)!=type || graph.type(product)!=type)continue;
      const auto& mul=graph.node(product);
      if(mul.kind!=s::Kind::Operation || mul.name!="Multiply" || mul.type!=type ||
          !mul.attributes.empty() || mul.fold || mul.operands.size()!=2)continue;
      for(unsigned factor=0;factor<2 && !peeled;++factor) {
        const auto& literal=mul.operands[factor];const auto& high=mul.operands[1-factor];
        if(!clean(literal) || !clean(high) || graph.type(literal)!=type || graph.type(high)!=type)continue;
        auto power=constant(literal);if(!power || !std::has_single_bit(*power))continue;
        const auto k=static_cast<unsigned>(std::countr_zero(*power));
        if(k>=shift+bits) {
          // Adding a multiple of 2^(shift+bits) cannot alter this exact field,
          // even if the unsigned Add/Multiply wraps above the observed bits.
          value=retained;peeled=true;
        } else if(k && k<=shift && graph.node(retained).attributes.empty() &&
                  !graph.node(retained).fold && unsignedFits(graph,retained,k)) {
          // low<2^k and high*2^k has k zero low bits: no carry. The observed
          // window is inside width, so discarded product overflow is excluded.
          value=high;shift-=k;peeled=true;
        }
      }
    }
    if(!peeled)break;changed=true;
  }
  if(!changed)return {};
  if(shift) {
    auto literal=graph.constant(amount_type,std::to_string(shift));
    value=graph.operation("ShiftRight",type,{value,literal});
  }
  if(!value.conversion.empty())value.conversion+="; ";
  value.conversion+="Convert("+type+" -> "+narrowing->target+")";
  if(!suffix.empty())value.conversion+="; "+suffix;
  value.converted_type=target;return value;
}

struct FrozenIndex {
  std::unordered_map<Key,Class,Hash> memo;
  std::vector<bool> finite;
  std::vector<Class> imported_roots;
};
class Engine {
 public:
  explicit Engine(Options options,std::span<const std::string> positive_parameters = {})
    : options_(options), started_(std::chrono::steady_clock::now()),
      positive_parameters_(positive_parameters.begin(),positive_parameters.end()) {}

  FrozenIndex freeze(std::span<const Class> roots={});

  std::vector<Class> import(const s::Graph& graph, std::span<const s::Ref> roots,
                            std::vector<unsigned char>* storage_roots=nullptr,
                            std::span<const s::Ref> witnesses={},std::vector<Class>* witness_outputs=nullptr,
                            bool captured_coordinate=false);

  Result compare(std::span<const Class> a, std::span<const Class> b,bool compile_only=false,
                 std::span<const unsigned char> storage_a={},std::span<const unsigned char> storage_b={});

  Result result(Status status, std::string reason) const;

  ScopedBodyResult scopedBodyProof(Class reference_wrapper,Class actual_wrapper,
      Class reference,Class actual,const s::Graph& reference_graph,std::span<const s::Id> reference_folds,
      const s::Graph& actual_graph,std::span<const s::Id> actual_folds,
      std::span<const ScopedAxisBinding> bindings,std::span<const Class> coordinates,
      std::span<const std::optional<Class>> image_coordinates={});

  JoinedBodyResult joinedBodyProof(Class reference_wrapper,Class reference,
      const s::Graph& reference_graph,std::span<const s::Id> reference_folds,
      std::span<const std::uint32_t> reference_axes,
      const s::Graph& actual_graph,std::span<const Class> actual_wrappers,
      std::span<const Class> bodies,std::span<const std::vector<s::Id>> folds,
      std::span<const std::array<Class,2>> coordinates,std::span<const JoinedBodyPiece> requests);

  ScopedProductResult productBodyProof(const s::Graph& reference_graph,
      std::span<const Class> reference_wrappers,std::span<const Class> reference_bodies,
      std::span<const std::vector<s::Id>> reference_folds,const s::Graph& actual_graph,
      std::span<const Class> actual_wrappers,std::span<const Class> actual_bodies,
      std::span<const std::vector<s::Id>> actual_folds,std::span<const ScopedAxisBinding> bindings,
      std::span<const Class> coordinates,std::span<const std::optional<Class>> image_coordinates);

 private:
  struct StorageCopy {
    // Store one immutable enode ID, not a copied allocation string/record.
    std::optional<Class> cell;
    Class offset=0;
    bool ambiguous=false;
  };
  void indexStoredOutputs();
  bool sameStoredOutput(Class a,Class b);
  std::vector<StorageCopy> storage_copies_;
  std::size_t storage_copy_nodes_=std::numeric_limits<std::size_t>::max();
  std::size_t storage_copy_unions_=std::numeric_limits<std::size_t>::max();

  struct PhysicalWorkPlan {
    std::size_t initial=epoch_work_allowance::default_work,remaining=initial,sizing=0;
    epoch_work_allowance::Size physical;
    bool exhausted=false;
  };
  PhysicalWorkPlan physicalWorkPlan(const std::vector<Key>& snapshot,
      std::size_t floor,std::size_t already_spent,bool account_global_work=false);
  void mergeLinearReductionNormalForms(const std::vector<Key>& snapshot,
      const std::vector<bool>&,std::span<const Class> left,
      std::span<const Class> right);
  std::optional<Class> emitCommonAddFactor(Class id,const Key& k,Class factor,
                                           Class left,Class right);
  void factorAdditiveForests(const std::vector<Key>& snapshot,
                              const std::vector<bool>& nonrecursive);
  void mergePolynomialNormalForms(const std::vector<Key>& snapshot,
                                  const std::vector<bool>& nonrecursive);
  void mergeAssociativeNormalForms(const std::vector<Key>& snapshot,
                                   const std::vector<bool>& nonrecursive);
  std::vector<bool> nonrecursiveHeads(const std::vector<Key>& snapshot);
  void mergeAlphaEquivalentFolds();
  struct SelectHeads {
    // One certified representative per local pattern. Different predicates
    // remain distinct; the index is rebuilt after congruence, never memoizes a
    // failed match across an e-class merge.
    std::unordered_map<Class,Class> conditions;
    std::unordered_map<std::string,Class> unary;
    std::map<std::pair<std::string,Class>,Class> factors;
    std::optional<std::pair<Class,Class>> product,difference,quotient;
  };
  struct SelectIndex {
    struct U64Constant { std::string type;std::uint64_t value; };
    struct U64AddConstant { std::string type;Class source,value;std::uint64_t constant; };
    std::unordered_map<Class,SelectHeads> heads;
    std::unordered_set<Class> constants,ones;
    std::unordered_map<Class,U64Constant> u64_constants;
    std::unordered_map<Class,U64AddConstant> u64_add_constants;
    const SelectHeads& get(Class id) const {
      static const SelectHeads empty;
      auto found=heads.find(id);return found==heads.end()?empty:found->second;
    }
  };
  SelectIndex indexSelectPatterns(const std::vector<Key>& snapshot,const std::vector<bool>& nonrecursive);
  void rewriteExactU64Add(Class id,const Key& k,const std::vector<Key>& snapshot,
                          const SelectIndex& index);
  bool rewriteKnownPredicateSelect(Class id,const Key& k,
      const std::vector<Key>& snapshot);
  bool rewriteFiniteCompareSelectExtremum(Class id,const Key& k,
      const std::vector<Key>& snapshot);
  void rewriteSelectPatterns(Class id,const Key& k,const std::vector<Class>& original,
      const std::vector<Key>& snapshot,const SelectIndex& index);
  // Opt-in bounded diagnostics, never part of proof or property propagation.
#if defined(SENTRYWARD_DEVELOPMENT_DIAGNOSTICS) || defined(SENTRYWARD_ENABLE_COMPRESSION_AUDIT)
#include "sentryward/development/diagnose.inc"
#else
  void diagnose(std::span<const Class> ,std::span<const Class> ) {}
#endif
  void compactImported(std::vector<Class>& a,std::vector<Class>& b);
  Class normalizeExponentialRatio(Class id,std::string_view base);

  bool numericOutput(Class id) const;

  Class numericOutputProducer(Class id) const;

  using AffineCache=std::unordered_map<Class,std::optional<IndexAffine>>;
  // Private scheduling witnesses, never equality facts. Header and fixed-radius
  // signatures have fixed local size, with no eager ancestor-DAG admission;
  // an ambiguous signature is not searched through or retried after failure.
  struct BodyHeadIndex {
    static constexpr std::size_t max_entries=4096,max_members=64,max_arity=64;
    // Public/local consumers retain their original default. Only an explicit
    // immutable physical-epoch caller opts into larger aggregate memo tables.
    std::size_t retained_entries=max_entries;
    Engine& engine;
    const std::vector<Key>& snapshot;
    const std::vector<std::vector<Class>>* members=nullptr;
    const std::vector<Class>* singleton_members=nullptr;
    std::function<std::span<const Class>(Class)> member_provider;
    std::size_t provider_size=0;
    std::size_t memberCount() const {
      return members?members->size():member_provider?provider_size:singleton_members->size();
    }
    std::span<const Class> classMembers(Class cls) const {
      if(cls>=memberCount())return {};
      return members?std::span<const Class>((*members)[cls]):member_provider?member_provider(cls):
          std::span<const Class>(&(*singleton_members)[cls],1);
    }
    struct Adapter {
      std::function<Class(Class)> owner;
      std::function<std::optional<IndexAffine>(Class,AffineCache&,std::size_t&)> affine;
      std::function<bool(Class,std::uint32_t)> depends;
    };
    std::optional<Adapter> adapter;
    Class owner(Class id) { return adapter?adapter->owner(id):engine.find(id); }
    bool depends(Class id,std::uint32_t binder) {
      return adapter?adapter->depends(id,binder):engine.dependsOn(id,binder);
    }
    AffineCache& affine_cache;
    std::size_t* remaining;
    scoped_work_budget::Guards query_guards;
    std::size_t*& query_remaining=query_guards.aggregate;
    std::size_t*& mapper_remaining=query_guards.local;
    bool coverage_active=false;
    std::size_t* fallback_trace_remaining=nullptr;
    std::size_t* denial_trace_remaining=nullptr;
    bool exhausted=false;
    struct Roots {Class preferred;Class tag;};
    struct Choice {Class head;bool unique=true;};
    using Signature=std::pair<Class,std::vector<Class>>;
    // Alpha shape signatures deliberately erase Fold binder labels. Keep
    // actual labels as separate variants so already-proven alpha aliases do
    // not make an otherwise unique compatible head ambiguous.
    using ChoiceKey=std::pair<Class,std::uint32_t>;
    using Choices=std::map<ChoiceKey,Choice>;
    // A probe borrows a source header only during one table operation. Table
    // keys and header_values still OWN all strings/zero children, including
    // append-only projection where nodes_ can reallocate after this call.
    struct HeaderProbe {
      int kind=0;
      std::string_view type,name,attributes,provenance;
      std::optional<s::FoldInfo> fold;
      std::uint32_t index_level=0;
      std::size_t arity=0;
      std::size_t hash=0;
    };
    struct HeaderKey {Key value;std::size_t hash;};
    struct HeaderBudget {};
    static HeaderProbe headerProbe(const Key& key) {
      return {key.kind,key.type,key.name,key.attributes,key.provenance,
          key.fold,key.index_level,key.children.size()};
    }
    struct HeaderHash {
      using is_transparent=void;
      std::size_t operator()(const HeaderProbe& k) const noexcept {return k.hash;}
      std::size_t operator()(const HeaderKey& k) const noexcept {return k.hash;}
      static std::size_t compute(const HeaderProbe& k) {
        std::size_t h=k.kind;
        const auto add=[&](std::size_t v){h^=v+0x9e3779b9+(h<<6)+(h>>2);};
        for(auto text:{k.type,k.name,k.attributes,k.provenance})add(std::hash<std::string_view>{}(text));
        add(k.index_level);add(k.fold.has_value());
        if(k.fold) {
          add(static_cast<std::size_t>(k.fold->reducer));add(k.fold->binder_level);
          add(std::hash<std::uint64_t>{}(k.fold->count));
        }
        // Every owned private header has zero-valued children. Its arity,
        // not any actual class child, is the complete header field here.
        add(k.arity);return h;
      }
    };
    template<class Charge>
    static bool equalHeaders(const HeaderProbe& a,const HeaderProbe& b,Charge&& charge) {
        charge(8);
        if(a.kind!=b.kind || a.index_level!=b.index_level || a.fold!=b.fold || a.arity!=b.arity)return false;
        const auto text=[&](std::string_view x,std::string_view y) {
          if(x.size()!=y.size())return false;
          if(x.empty())return true;
          charge(x.size());return x==y;
        };
        return text(a.type,b.type) && text(a.name,b.name) &&
            text(a.attributes,b.attributes) && text(a.provenance,b.provenance);
    }
    struct HeaderEqual {
      using is_transparent=void;
      BodyHeadIndex* index;
      bool operator()(const HeaderProbe& a,const HeaderProbe& b) const {
        return equalHeaders(a,b,[&](std::size_t amount) {
          if(!index->charge(amount))throw HeaderBudget{};
        });
      }
      bool operator()(const HeaderKey& a,const HeaderKey& b) const {return (*this)(headerProbe(a.value),headerProbe(b.value));}
      bool operator()(const HeaderKey& a,const HeaderProbe& b) const {return (*this)(headerProbe(a.value),b);}
      bool operator()(const HeaderProbe& a,const HeaderKey& b) const {return (*this)(a,headerProbe(b.value));}
    };
    std::unordered_map<HeaderKey,Class,HeaderHash,HeaderEqual> headers;
    std::size_t header_probes=0,header_probe_hits=0,header_copied_bytes=0;
    struct HeaderStatistics {
      std::size_t probes=0,hits=0,keys=0,copied_bytes=0;
      std::size_t sole_root_hits=0,sole_shallow_hits=0,sole_root_tokens=0,sole_shallow_tokens=0;
      std::size_t source_header_hits=0,source_header_tokens=0;
      std::size_t descriptor_hits=0,descriptor_translations=0;
    };
    HeaderStatistics* header_statistics=nullptr;
    // Sole actual snapshot witnesses have no alternative to disambiguate.
    // Retain only the already-interned scalar scheduling IDs, NOT proof rows,
    // copied headers, lexical contexts or expanded ancestor signatures. Raw
    // scalar arrays are default-initialized (unwritten slots are never read);
    // only O(snapshot/64) validity words require eager initialization.
    std::size_t sole_bound=0,sole_epoch=0;
    std::unique_ptr<Class[]> sole_root_tags,sole_shallow_tags;
    std::vector<std::uint64_t> sole_root_valid,sole_shallow_valid;
    std::size_t sole_root_hits=0,sole_shallow_hits=0,sole_root_tokens=0,sole_shallow_tokens=0;
    // Existing source-id header cache, represented as scalar tags for this
    // opted-in immutable prefix only. Full and affine-filter headers occupy
    // separate slots; no header metadata or failed proof result is copied.
    std::unique_ptr<Class[]> source_header_tags;
    std::vector<std::uint64_t> source_header_valid;
    std::size_t source_header_hits=0,source_header_tokens=0;
    // Exact physical metadata IDs are only a scheduling-header cache. The
    // local tag remains first-encounter ordered, and every actual child/scope
    // obligation still runs through the full paired walk.
    std::size_t descriptor_generation=0,descriptor_hits=0,descriptor_translations=0;
    std::unordered_map<Class,Class> descriptor_tags;
    std::unordered_map<Class,Class> header_cache;
    std::vector<Key> header_values;
    std::unordered_map<Class,Roots> roots_cache;
    std::unordered_map<Class,Class> compact_cache;
    std::unordered_map<Class,Class> shallow_cache;
    std::map<Signature,Class> shallow_keys;
    std::map<Signature,Class> choice_keys;
    std::unordered_map<Class,Choices> choices_cache;
    std::unordered_map<Class,Choices> root_choices_cache;
    std::size_t choice_entries=0;
    std::size_t root_first_attempts=0,root_first_hits=0;
    std::size_t root_orientation_attempts=0,root_orientation_hits=0;
    BodyHeadIndex(Engine& e,const std::vector<Key>& s,
        const std::vector<std::vector<Class>>& m,AffineCache& a,std::size_t* r)
        :engine(e),snapshot(s),members(&m),affine_cache(a),remaining(r),
         headers(0,HeaderHash{},HeaderEqual{this}) {}
    // A comparison-local owning lazy catalog; no borrowed span is retained by
    // this callback. Its provider and immutable ID epoch outlive this index.
    template<class Members> BodyHeadIndex(Engine& e,const std::vector<Key>& s,
        const Members& m,AffineCache& a,std::size_t* r)
        :engine(e),snapshot(s),member_provider([&m](Class id){return m[id];}),
         provider_size(m.size()),affine_cache(a),remaining(r),
         headers(0,HeaderHash{},HeaderEqual{this}) {}
    // Append-only projection has one header per class. Borrow its ID catalog;
    // never reconstruct a vector-of-members or copy the source DAG per SAT.
    BodyHeadIndex(Engine& e,const std::vector<Key>& s,
        const std::vector<Class>& m,AffineCache& a,std::size_t* r,Adapter callbacks)
        :engine(e),snapshot(s),singleton_members(&m),adapter(std::move(callbacks)),affine_cache(a),remaining(r),
         headers(0,HeaderHash{},HeaderEqual{this}) {}
    BodyHeadIndex(const BodyHeadIndex&)=delete;
    BodyHeadIndex& operator=(const BodyHeadIndex&)=delete;
    ~BodyHeadIndex() {
      if(header_statistics)*header_statistics={header_probes,header_probe_hits,headers.size(),header_copied_bytes,
          sole_root_hits,sole_shallow_hits,sole_root_tokens,sole_shallow_tokens,
          source_header_hits,source_header_tokens,descriptor_hits,descriptor_translations};
    }
    struct QueryBudget {
      BodyHeadIndex& index;std::size_t* previous;std::size_t* previous_mapper;
      bool previous_coverage;
      // Existing local/prefix callers keep their original constructor and
      // limits. If reached inside scoped coverage, they too retain its
      // aggregate instead of overwriting it with a fresh local allowance.
      QueryBudget(BodyHeadIndex& i,std::size_t* local):QueryBudget(i,local,local,false) {}
      QueryBudget(BodyHeadIndex& i,std::size_t* body,std::size_t* coverage,bool scoped)
          :index(i),previous(i.query_remaining),previous_mapper(i.mapper_remaining),
           previous_coverage(i.coverage_active) {
        if(scoped || previous_coverage) {
          // The helper relay precharges guards, then sync invoices epoch.
          // These PRIVATE scoped counters must therefore stay distinct from
          // the epoch pointer. Fail closed on an invalid future caller; the
          // standalone Guards utility still permits aggregate/local aliasing.
          if(i.remaining && (body==i.remaining || previous==i.remaining ||
              (!previous && coverage==i.remaining))){i.cap();return;}
          // Only the outermost scoped goal borrows a new counter from the
          // ALREADY remaining epoch. Every nested mapper retains that SAME
          // aggregate (or any pre-existing tighter caller allowance).
          i.coverage_active=true;i.query_remaining=previous?previous:coverage;
          i.mapper_remaining=body;
        } else if(body!=i.remaining)i.query_remaining=body;
      }
      ~QueryBudget(){index.query_remaining=previous;index.mapper_remaining=previous_mapper;
        index.coverage_active=previous_coverage;}
    };
    struct MapperPause {
      BodyHeadIndex& index;std::size_t* previous;
      explicit MapperPause(BodyHeadIndex& i):index(i),previous(i.mapper_remaining) {
        if(i.coverage_active)i.mapper_remaining=nullptr;
      }
      ~MapperPause(){index.mapper_remaining=previous;}
    };
    struct FallbackTrace {
      BodyHeadIndex& index;std::size_t* previous;std::size_t* previous_denial;
      FallbackTrace(BodyHeadIndex& i,std::size_t* r,std::size_t* d=nullptr)
          :index(i),previous(i.fallback_trace_remaining),previous_denial(i.denial_trace_remaining) {
        index.fallback_trace_remaining=r;index.denial_trace_remaining=d;
      }
      ~FallbackTrace(){index.fallback_trace_remaining=previous;index.denial_trace_remaining=previous_denial;}
    };
    bool charge(std::size_t amount=1);
    bool cap(){exhausted=true;return false;}
    void initializeSoleHints(std::size_t retained_capacity=max_entries,
        bool complete_descriptors=false);
    std::optional<Class> sourceHeaderHint(Class id);
    bool retainSourceHeader(Class id,Class tag);
    std::optional<Class> soleHint(Class cls,bool shallow);
    bool rememberSoleHint(Class cls,Class tag,bool shallow);
    static bool bounded(const Key& k) {
      return k.type.size()<=128 && k.name.size()<=4096 && k.attributes.size()<=128 &&
          k.provenance.size()<=4096 && k.children.size()<=max_arity;
    }
    std::optional<IndexAffine> affine(Class cls);
    std::optional<Class> header(Class source_id,bool affine_address=false);
    static unsigned priority(const Key& k);
    bool preferredTag(Class a,Class b) const {
      const auto& x=header_values[a];const auto& y=header_values[b];
      return std::tuple{priority(x),x.kind,std::string_view{x.type},std::string_view{x.name},a}<
             std::tuple{priority(y),y.kind,std::string_view{y.type},std::string_view{y.name},b};
    }
    std::optional<Roots> roots(Class cls);
    std::optional<Class> compact(Class cls);
    std::optional<Class> preferred(Class cls) {
      return compact(cls);
    }
    std::optional<Class> shallow(Class cls);
    const Choices* choices(Class cls);
    std::optional<std::pair<Class,Class>> matchChoices(const Choices& x,const Choices& y,
        bool unique_root=false);
    const Choices* rootChoices(Class cls);
    std::optional<std::pair<Class,Class>> canonicalRootPair(const Choices& x,const Choices& y);
    std::optional<std::pair<Class,Class>> fixedFoldPair(const Choices& x,const Choices& y);
    std::optional<std::pair<Class,Class>> rootMatched(Class a,Class b);
    bool localRootCompatible(Class a,Class b);
    std::optional<std::pair<Class,Class>> matched(Class a,Class b,bool complete_root_first=false);
  };
  // One dormant, Engine-owned COMPLETE scheduling-header catalog. A source
  // slot indexes its actual physical node, never its changing e-class owner.
  // Descriptor equality deliberately matches BodyHeadIndex::header(false),
  // including normalized Fold binder0; this is NOT a value/scope certificate.
  struct CompleteHeaderCatalog {
    using Probe=BodyHeadIndex::HeaderProbe;
    using Entry=BodyHeadIndex::HeaderKey;
    static constexpr Class missing=std::numeric_limits<Class>::max();
    Engine& engine;
    BodyHeadIndex* preparing=nullptr; // Borrowed only during synchronous bootstrap.
    struct Equal {
      using is_transparent=void;
      CompleteHeaderCatalog* catalog;
      bool operator()(const Probe& a,const Probe& b) const {
        return BodyHeadIndex::equalHeaders(a,b,[&](std::size_t amount){catalog->charge(amount);});
      }
      bool operator()(const Entry& a,const Entry& b) const {
        return (*this)(BodyHeadIndex::headerProbe(a.value),BodyHeadIndex::headerProbe(b.value));
      }
      bool operator()(const Entry& a,const Probe& b) const {return (*this)(BodyHeadIndex::headerProbe(a.value),b);}
      bool operator()(const Probe& a,const Entry& b) const {return (*this)(a,BodyHeadIndex::headerProbe(b.value));}
    };
    std::unordered_map<Entry,Class,BodyHeadIndex::HeaderHash,Equal> headers;
    // unordered_map rehash preserves references to its OWNED keys. No Key is
    // duplicated per source or in this scalar descriptor-to-key directory.
    std::vector<const Entry*> values;
    std::vector<Class> sources;
    explicit CompleteHeaderCatalog(Engine& e):engine(e),headers(0,BodyHeadIndex::HeaderHash{},Equal{this}) {}
    void charge(std::size_t amount) {
      if(preparing) {
        if(!preparing->charge(amount))throw BodyHeadIndex::HeaderBudget{};
      } else engine.completeHeaderCharge(amount);
    }
    template<class T> void append(std::vector<T>& entries,T value) {
      if(entries.size()==entries.capacity()) {
        const auto old=entries.size();
        if(old>(std::numeric_limits<std::size_t>::max()-1)/3)throw Limit{"node_budget"};
        // Precharge physical capacity allocation and scalar relocation. No
        // growing immutable array is rebuilt on every single append.
        charge(3*old+1);entries.reserve(2*old+1);
      }
      charge(1);entries.push_back(value);
    }
    Class registerHeader(const Key& source) {
      charge(1);
      if(!BodyHeadIndex::bounded(source))return missing;
      auto probe=BodyHeadIndex::headerProbe(source);
      if(probe.fold)probe.fold->binder_level=0;
      const auto bytes=probe.type.size()+probe.name.size()+probe.attributes.size()+probe.provenance.size();
      charge(8+bytes);probe.hash=BodyHeadIndex::HeaderHash::compute(probe);
      if(auto found=headers.find(probe);found!=headers.end())return found->second;
      if(headers.size()>=epoch_work_allowance::maximum_retained_headers)return missing;
      charge(bytes+probe.arity+1);
      Key key;key.kind=probe.kind;key.type=probe.type;key.name=probe.name;
      key.attributes=probe.attributes;key.provenance=probe.provenance;
      key.fold=probe.fold;key.index_level=probe.index_level;key.children.resize(probe.arity,0);
      // Cached hashes make rehash scalar-only; debit the actual retained
      // directory if this insertion can exceed the default unit load factor.
      if(headers.size()+1>headers.bucket_count())charge(headers.size()+1);
      const auto tag=values.size();
      const auto entry=headers.emplace(Entry{std::move(key),probe.hash},tag);
      if(!entry.second)return entry.first->second;
      try {append(values,&entry.first->first);}
      catch(...) {headers.erase(entry.first);throw;}
      return tag;
    }
    void registerSource(Class id,const Key& source) {
      if(id!=sources.size())throw std::logic_error("noncontiguous complete header source");
      const auto tag=registerHeader(source);append(sources,tag);
    }
  };
  void completeHeaderCharge(std::size_t amount);
  bool prepareCompleteHeaders(std::size_t bound,BodyHeadIndex& heads);
  std::optional<Class> completeHeaderId(Class source) const {
    if(!complete_headers_ || source>=complete_headers_->sources.size())return {};
    const auto tag=complete_headers_->sources[source];
    if(tag==CompleteHeaderCatalog::missing)return {};return tag;
  }
  const BodyHeadIndex::HeaderKey* completeHeader(Class tag) const {
    if(!complete_headers_ || tag>=complete_headers_->values.size())return nullptr;
    return complete_headers_->values[tag];
  }
  struct RectangleAxis {std::uint32_t binder;std::uint64_t count;};
  struct RectangleMap {
    std::uint64_t begin=0;
    std::vector<std::uint64_t> coefficients;
    bool operator==(const RectangleMap&) const = default;
  };
  static constexpr std::size_t rectangle_work_limit=16384;
  std::size_t rectangle_limit_=rectangle_work_limit;
  bool rectanglePoll(std::size_t& work,std::size_t amount=1);
  template<class Members>
  std::optional<IndexAffine> rectangleAffine(Class root,const std::vector<Key>& snapshot,
      const Members& members,AffineCache& cache,std::size_t& work) {
    // Iterative postorder over ONE admitted nonrecursive representative. No
    // recursive evaluator or search through alternative affine expressions.
    struct Task {Class id;bool ready=false;};
    std::vector<Task> pending{{find(root)}};std::unordered_set<Class> visiting;
    while(!pending.empty()) {
      if(!rectanglePoll(work))return {};
      auto t=pending.back();pending.pop_back();t.id=find(t.id);
      if(cache.contains(t.id))continue;
      if(t.id>=members.size() || members[t.id].empty())return {};
      const auto& k=snapshot[members[t.id].front()];
      if(k.provenance.empty() && k.attributes.empty() &&
          (k.type=="u64" || k.type=="exact:u64")) {
        if(k.kind==static_cast<int>(s::Kind::Index)) {
          cache.emplace(t.id,IndexAffine{0,{{k.index_level,1}}});continue;
        }
        if(k.kind==static_cast<int>(s::Kind::Constant)) {
          std::uint64_t value=0;
          const auto parsed=std::from_chars(k.name.data(),k.name.data()+k.name.size(),value);
          if(parsed.ec==std::errc{} && parsed.ptr==k.name.data()+k.name.size())
            cache.emplace(t.id,IndexAffine{value,{}});
          else cache.emplace(t.id,std::nullopt);
          continue;
        }
        if(k.kind==static_cast<int>(s::Kind::Operation) && k.children.size()==2 &&
            (k.name=="Add" || k.name=="Subtract" || k.name=="Multiply") &&
            snapshot[k.children[0]].type==k.type && snapshot[k.children[1]].type==k.type) {
          if(!t.ready) {
            if(!visiting.insert(t.id).second)return {};
            pending.push_back({t.id,true});
            for(auto child:k.children)if(!cache.contains(find(child)))pending.push_back({child,false});
            continue;
          }
          visiting.erase(t.id);
          const auto a=cache.find(find(k.children[0])),b=cache.find(find(k.children[1]));
          if(a==cache.end() || b==cache.end())return {};
          if(a->second && b->second) {
            const auto first=a->second->coefficients.size(),second=b->second->coefficients.size();
            if(first>rectangle_limit_ || second>rectangle_limit_-first) {
              work=rectangle_limit_+1;return {};
            }
            const auto entries=first+second;
            // combineAffine takes both maps by value, merges/erases entries,
            // and may grow a balanced map. Charge BEFORE those copies/work.
            const auto merge_work=4*(entries+1)*(1+std::bit_width(entries+1));
            if(!rectanglePoll(work,merge_work))return {};
            auto combined=combineAffine(k.name,*a->second,*b->second);
            // Move the result into the shared cache instead of cloning it.
            cache.emplace(t.id,std::move(combined));
          } else cache.emplace(t.id,std::nullopt);
          continue;
        }
      }
      cache.emplace(t.id,std::nullopt);
    }
    const auto found=cache.find(find(root));if(found==cache.end())return {};
    if(found->second && !rectanglePoll(work,1+found->second->coefficients.size()))return {};
    return found->second;
  }
  template<class Members>
  std::optional<RectangleMap> rectangleBodyMap(Class target,Class block,std::uint32_t binder,
      const std::vector<RectangleAxis>& axes,const std::vector<Key>& snapshot,
      const Members& members,AffineCache& cache,std::size_t& work,
      BodyHeadIndex* witness_heads=nullptr) {
    struct Task {Class a,b;bool ready=false;unsigned scope=0;};
    std::vector<Task> pending{{target,block}};
    // Inner reductions remain reductions. Only their lexical coordinates are
    // alpha-renamed while the surrounding rectangle is matched; no members
    // of either reduction are materialized. Seeds use the enclosing scope.
    using Renames=std::map<std::uint32_t,std::uint32_t>;
    std::vector<Renames> renames(1);
    std::map<std::tuple<Class,Class,unsigned>,unsigned char> visited;
    std::optional<RectangleMap> substitution;
    const auto alphaFoldHead=[](const Key& x,const Key& y) {
      return x.fold && y.fold && x.kind==y.kind && x.type==y.type &&
          x.name==y.name && x.attributes==y.attributes && x.provenance==y.provenance &&
          x.children.size()==2 && y.children.size()==2 &&
          x.fold->reducer==y.fold->reducer && x.fold->count==y.fold->count;
    };
    const auto sameHead=[](const Key& a,const Key& b) {
      return a.kind==b.kind && a.type==b.type && a.name==b.name && a.attributes==b.attributes &&
          a.provenance==b.provenance && a.fold==b.fold && a.index_level==b.index_level &&
          a.children.size()==b.children.size();
    };
    const auto compatible=[&](Class a,Class b) {
      a=find(a);b=find(b);
      if(a>=members.size() || b>=members.size())return false;
      for(auto p:members[a])for(auto q:members[b]) {
        if(!rectanglePoll(work))return false;
        if(sameHead(snapshot[p],snapshot[q]) || alphaFoldHead(snapshot[p],snapshot[q]))return true;
      }
      return false;
    };
    // Fixed-radius scheduling hint only. Complete operand/address matching
    // below remains mandatory; matching headers never establishes equality.
    const auto childCompatible=[&](Class a,Class b) {
      a=find(a);b=find(b);
      for(auto x:members[a])for(auto y:members[b]) {
        if(!rectanglePoll(work))return false;
        const auto& p=snapshot[x];const auto& q=snapshot[y];
        if(!sameHead(p,q) || p.children.size()>2)continue;
        bool direct=true,reverse=p.children.size()==2 &&
            (arithmetic(p) || integerCommutative(p));
        for(std::size_t i=0;i<p.children.size();++i) {
          direct &= compatible(p.children[i],q.children[i]);
          if(reverse)reverse &= compatible(p.children[i],q.children[1-i]);
        }
        if(direct || reverse)return true;
      }
      return false;
    };
    while(!pending.empty()) {
      if(!rectanglePoll(work))return {};
      auto t=pending.back();pending.pop_back();t.a=find(t.a);t.b=find(t.b);
      if(t.a>=members.size() || t.b>=members.size())return {};
      if(snapshot[t.a].type!=snapshot[t.b].type)return {};
      const auto pair=std::tuple{t.a,t.b,t.scope};
      if(t.ready){visited[pair]=2;continue;}
      if(auto found=visited.find(pair);found!=visited.end()) {
        if(found->second==1)return {};continue;
      }
      bool independent=!dependsOn(t.a,binder);
      for(const auto& axis:axes)independent &= !dependsOn(t.b,axis.binder);
      for(auto [from,to]:renames[t.scope])
        independent &= !dependsOn(t.a,from) && !dependsOn(t.b,to);
      if(t.a==t.b && independent){visited[pair]=2;continue;}
      auto a=rectangleAffine(t.a,snapshot,members,cache,work);
      auto b=rectangleAffine(t.b,snapshot,members,cache,work);
      if(work>rectangle_limit_)return {};
      if(a && b) {
        if(!rectanglePoll(work,1+a->coefficients.size()+renames[t.scope].size()))return {};
        // Simultaneous rename, not an in-place sequence of substitutions.
        std::map<std::uint32_t,std::uint64_t> renamed;
        for(auto [level,value]:a->coefficients) {
          auto found=renames[t.scope].find(level);
          const auto next=found==renames[t.scope].end()?level:found->second;
          if(renamed.contains(next))return {}; // No capture or aliasing.
          renamed.emplace(next,value);
        }
        a->coefficients=std::move(renamed);
        // The returned affine maps are bounded, but their comparison/erase
        // work is not a scalar read. No wide cancelled-index map is free.
        const auto entries=a->coefficients.size()+b->coefficients.size();
        if(!rectanglePoll(work,(entries+axes.size()+1)*(1+std::bit_width(entries+1))))return {};
        const auto coefficient=a->coefficients.contains(binder)?a->coefficients.at(binder):0;
        RectangleMap proposed;proposed.coefficients.resize(axes.size());
        if(coefficient) {
          if(b->constant<a->constant || (b->constant-a->constant)%coefficient)return {};
          proposed.begin=(b->constant-a->constant)/coefficient;
          for(std::size_t i=0;i<axes.size();++i) {
            const auto found=b->coefficients.find(axes[i].binder);
            const auto value=found==b->coefficients.end()?0:found->second;
            if(value%coefficient)return {};proposed.coefficients[i]=value/coefficient;
          }
          if(substitution && *substitution!=proposed)return {};substitution=std::move(proposed);
        } else if(a->constant!=b->constant)return {};
        a->coefficients.erase(binder);
        for(const auto& axis:axes) {
          if(!coefficient && b->coefficients.contains(axis.binder))return {};
          b->coefficients.erase(axis.binder);
        }
        if(a->coefficients!=b->coefficients)return {};
        visited[pair]=2;continue;
      }
      std::optional<std::pair<Class,Class>> selected;
      if(witness_heads) {
        if(work>=rectangle_limit_)return {};
        const auto allowance=rectangle_limit_-work;
        *witness_heads->remaining=allowance;
        selected=witness_heads->matched(t.a,t.b);
        if(!rectanglePoll(work,allowance-*witness_heads->remaining))return {};
        if(selected && !sameHead(snapshot[selected->first],snapshot[selected->second]) &&
            !alphaFoldHead(snapshot[selected->first],snapshot[selected->second]))return {};
      } else for(auto p:members[t.a]) {
        for(auto q:members[t.b]) {
          if(!rectanglePoll(work))return {};
          if(sameHead(snapshot[p],snapshot[q]) || alphaFoldHead(snapshot[p],snapshot[q]))
            {selected=std::pair{p,q};break;}
        }
        if(selected)break;
      }
      if(!selected)return {};
      const auto& p=snapshot[selected->first];const auto& q=snapshot[selected->second];
      if(p.fold) {
        if(renames[t.scope].size()>=8 || renames.size()>=128 ||
            p.fold->binder_level<=binder)return {};
        for(const auto& axis:axes)if(q.fold->binder_level<=axis.binder)return {};
        auto next=renames[t.scope];
        if(next.contains(p.fold->binder_level))return {};
        for(auto [from,to]:next)if(to==q.fold->binder_level)return {};
        next.emplace(p.fold->binder_level,q.fold->binder_level);
        const auto childScope=static_cast<unsigned>(renames.size());
        renames.push_back(std::move(next));
        visited[pair]=1;pending.push_back({t.a,t.b,true,t.scope});
        pending.push_back({p.children[0],q.children[0],false,t.scope});
        pending.push_back({p.children[1],q.children[1],false,childScope});
        continue;
      }
      if(p.kind==static_cast<int>(s::Kind::Index))return {};
      bool swapped=false;
      if((arithmetic(p) || integerCommutative(p)) && p.children.size()==2) {
        bool direct=compatible(p.children[0],q.children[0]) && compatible(p.children[1],q.children[1]);
        bool reverse=compatible(p.children[0],q.children[1]) && compatible(p.children[1],q.children[0]);
        if(direct && reverse) {
          const bool sameDirect=find(p.children[0])==find(q.children[0]) && find(p.children[1])==find(q.children[1]);
          const bool sameReverse=find(p.children[0])==find(q.children[1]) && find(p.children[1])==find(q.children[0]);
          if(sameDirect || sameReverse){direct=sameDirect;reverse=!sameDirect && sameReverse;}
          else {
            direct=childCompatible(p.children[0],q.children[0]) && childCompatible(p.children[1],q.children[1]);
            reverse=childCompatible(p.children[0],q.children[1]) && childCompatible(p.children[1],q.children[0]);
          }
        }
        if(!direct && !reverse)return {};
        if(direct && reverse && (find(p.children[0])!=find(p.children[1]) ||
                                find(q.children[0])!=find(q.children[1])))return {};
        swapped=!direct && reverse;
      }
      if(work>rectangle_limit_)return {};
      visited[pair]=1;pending.push_back({t.a,t.b,true,t.scope});
      for(std::size_t i=0;i<p.children.size();++i) {
        if(!rectanglePoll(work))return {};
        pending.push_back({p.children[i],q.children[swapped?1-i:i],false,t.scope});
      }
    }
    return substitution;
  }
  void flattenAffineFold(Class id,const Key& root,const std::vector<Key>& snapshot,
      const std::vector<std::vector<Class>>& members,
      std::unordered_map<Class,Class>* import_replacements=nullptr);
  void coalesceRectangularFolds(Class id,const Key& root,const std::vector<Key>& snapshot,
      const std::vector<std::vector<Class>>& members,
      const std::map<std::tuple<std::string,std::uint32_t,std::uint64_t>,std::vector<Class>>& targets);
  // Domain tokens are minted from actual comparison metadata/Fold BODY edges.
  // Distinct lexical binders are not identified merely by a level/count pair.
  // This small catalog is shared by one immutable scoped-certificate epoch.
  struct TypedDispatch;
  struct BodyScopeRegistry {
    using Id=std::uint32_t;
    struct Scope {
      lexical_scope::Id bounds;std::vector<Id> domains;
      std::optional<Id> parent={}; // Actual lexical parent, not another count algorithm.
    };
    struct ImageId {
      bool operator==(const ImageId&) const=default;
      std::size_t hash() const {return std::hash<const void*>{}(owner)^std::hash<std::size_t>{}(id);}
     private:const BodyScopeRegistry* owner=nullptr;std::size_t id=0;
      friend struct BodyScopeRegistry;friend struct TypedDispatch;
    };
    // One actual outer assignment may use different exact unit coordinate
    // origins. This token changes ONLY a private geometric time role; it
    // never equates the two graph expressions or changes their lexical bounds.
    struct CommonTranslationId {
      bool operator==(const CommonTranslationId&) const=default;
     private:
      const BodyScopeRegistry* owner=nullptr;std::uint64_t epoch=0;std::size_t id=0;
      friend struct BodyScopeRegistry;
    };
    // A scalar assignment minted only by the actual typed selector cursor
    // after its remaining checked image is a singleton. This is NOT an
    // enode or an assertion that the coordinate producer equals a literal.
    struct DispatchPointId {
      DispatchPointId()=default;
      DispatchPointId(const DispatchPointId& p) noexcept:owner(p.owner),epoch(p.epoch),image(p.image),
          root(p.root),branch(p.branch),value(p.value) {}
      DispatchPointId& operator=(const DispatchPointId& p) noexcept {
        owner=p.owner;epoch=p.epoch;image=p.image;root=p.root;branch=p.branch;value=p.value;return *this;
      }
      bool operator==(const DispatchPointId&) const=default;
     private:
      const BodyScopeRegistry* owner=nullptr;std::uint64_t epoch=0;
      std::size_t image=0;Class root=0,branch=0;std::uint64_t value=0;
      friend struct BodyScopeRegistry;friend struct TypedDispatch;
    };
    // DispatchPointId's nontrivial copy makes variant copies construct ONLY
    // the active alternative; ordinary retained-Class keys do not copy its
    // larger inactive storage on every sampling cache operation.
    struct TypedPointId {
      TypedPointId()=default;
      TypedPointId(const TypedPointId& p) noexcept:owner(p.owner),epoch(p.epoch),image(p.image),
          cell(p.cell),address(p.address),scope(p.scope),value(p.value) {}
      TypedPointId& operator=(const TypedPointId& p) noexcept {
        owner=p.owner;epoch=p.epoch;image=p.image;cell=p.cell;address=p.address;scope=p.scope;value=p.value;return *this;
      }
      bool operator==(const TypedPointId&) const=default;
     private:
      const BodyScopeRegistry* owner=nullptr;std::uint64_t epoch=0;
      std::size_t image=0;Class cell=0,address=0;Id scope=0;std::uint64_t value=0;
      friend struct BodyScopeRegistry;
    };
    using ScalarPoint=std::variant<Class,DispatchPointId,TypedPointId>;
    struct PointId {
      bool operator==(const PointId&) const=default;
     private:
      const BodyScopeRegistry* owner=nullptr;
      std::size_t image=0;Class coordinate=0;Id actual_scope=0;std::uint64_t local=0;
      bool dispatch=false;
      friend struct BodyScopeRegistry;
    };
    struct Binding {
      std::uint32_t reference_level;
      std::optional<std::uint32_t> actual_level;
      std::uint64_t offset,extent;
      Class actual_coordinate;
      // false: reference i=actual j+offset; true: actual j=reference i+offset.
      // Point mappings remain reference-only. No lexical scope is widened.
      bool actual_shift=false;
      std::optional<PointId> point={};
      std::optional<ImageId> coordinate_image={};
      bool operator==(const Binding&) const=default;
    };
    struct Context {
      Id reference_scope,actual_scope;
      std::array<Binding,2> axes{};
      unsigned axis_count=0;
      bool operator==(const Context&) const=default;
    };
    // This handle admits only immutable assignment geometry, not expression
    // equality. Original scope/point/image identities remain in the record.
    struct ContextId {
      bool operator==(const ContextId&) const=default;
      std::size_t hash() const {
        auto h=std::hash<std::uint64_t>{}(epoch);
        h^=id+0x9e3779b9u+(h<<6)+(h>>2);return h;
      }
     private:
      const BodyScopeRegistry* owner=nullptr;std::uint64_t epoch=0;std::size_t id=0;
      friend struct BodyScopeRegistry;
    };
    struct ContextCap {};
    struct ContextLess {
      BodyScopeRegistry* owner;
      bool operator()(const Context& a,const Context& b) const {return owner->lessContext(a,b);}
    };
    static std::uint64_t nextContextEpoch() {
      return algorithm::AnalysisScope::Current().Next(algorithm::AnalysisScope::Ids::kEGraphContext);
    }
    static constexpr std::size_t default_retained_records=4096,max_depth=8;
    BodyHeadIndex& heads;
    // One storage allowance for append-only owned records and subordinate
    // compiled plans. It does not change any rank, original-domain, local
    // proof or shared-work guard. One-shot callers retain their4096 default.
    const std::size_t max_retained_records;
    // Copied scalar domain identities use the SAME physical record allowance.
    // Default4096 records still permits16384 fields; explicit epoch capacity
    // is bounded by65536 records, hence at most262144 retained scalar fields.
    // This changes storage only, never domain identities or proof/work guards.
    const std::size_t max_identity_slots;
    std::size_t retained_record_rejections=0;
    const char* retained_record_cap="none";
    std::size_t context_plan_rejections=0;
    const char* context_plan_cap="none";
    std::size_t image_rejections=0;
    const char* image_cap="none";
    const std::uint64_t context_epoch=nextContextEpoch();
    lexical_scope::Catalog bounds;
    // deque keeps borrowed parent records stable while derived contexts are
    // appended. No growing Context array is cloned or invalidates references.
    std::deque<Context> contexts;
    std::map<Context,ContextId,ContextLess> context_keys;
    std::size_t context_reads=0,context_prefix_hits=0,original_body_views=0,transported_body_uses=0;
    std::vector<Scope> scopes;
    std::map<std::pair<Id,Class>,Id> body_scopes;
    std::map<std::tuple<Id,unsigned,s::Id>,Id> source_scopes;
    Id next_domain=0;std::size_t identity_slots=0,coordinate_checks=0,affine_checks=0;
    bool identity_cap_reported=false;
    struct Image {Id scope;std::uint32_t level;Class coordinate;std::uint64_t begin,end;bool derived=false;};
    std::vector<Image> images;
    std::map<std::tuple<Id,std::uint32_t,Class>,ImageId> image_keys;
    struct TypedImageKey {
      Class piece,full;Context context;std::uint32_t piece_axis,full_axis;
      bool operator==(const TypedImageKey&) const=default;
    };
    struct TypedImageHash {
      std::size_t operator()(const TypedImageKey& key) const {
        // Hash is only an index; equality retains the full owned context,
        // including every coordinate/point/image and original scope token.
        std::size_t h=key.piece;
        const auto mix=[&](std::size_t value){h^=value+0x9e3779b9u+(h<<6)+(h>>2);};
        mix(key.full);mix(key.context.reference_scope);mix(key.context.actual_scope);
        mix(key.piece_axis);mix(key.full_axis);mix(key.context.axis_count);
        // Context has two fixed slots, not a logical member/axis array. Include
        // its varying assignments to avoid a systematic same-scope hash bucket.
        for(const auto& axis:key.context.axes) {
          mix(axis.reference_level);mix(axis.actual_level.value_or(UINT32_MAX));
          mix(axis.offset);mix(axis.extent);mix(axis.actual_coordinate);
          mix(axis.actual_shift);mix(bool(axis.point));mix(bool(axis.coordinate_image));
        }
        return h;
      }
    };
    std::unordered_map<TypedImageKey,ImageId,TypedImageHash> typed_image_keys;
    struct TypedImageSource {Class full;Id scope;std::uint32_t axis;};
    std::unordered_map<std::size_t,TypedImageSource> typed_image_sources;
    struct TypedTimeSource {
      Class piece,anchor;ContextId context;
      std::uint64_t stride,anchor_constant;
    };
    std::map<std::tuple<Class,Class,std::size_t,std::uint32_t>,ImageId> typed_time_keys;
    std::map<std::size_t,TypedTimeSource> typed_time_sources;
    std::map<std::pair<std::size_t,std::size_t>,ImageId> translated_typed_images;
    std::map<std::pair<std::size_t,std::size_t>,ImageId> relative_image_keys;
    std::map<std::pair<std::size_t,Id>,ImageId> original_scope_image_keys;
    struct CommonTranslation {ImageId source,actual;bool add;std::uint64_t difference;};
    std::deque<CommonTranslation> common_translations;
    std::map<std::pair<std::size_t,std::size_t>,CommonTranslationId> common_translation_keys;
    std::map<std::pair<std::size_t,std::size_t>,ImageId> common_image_keys;
    std::map<std::size_t,std::size_t> common_image_translations;
    struct SampleKey {
      ImageId time;std::optional<ImageId> state;
      Id actual_scope;ScalarPoint point;std::uint32_t state_axis;
      bool operator==(const SampleKey&) const=default;
    };
    struct SampleHash {
      std::size_t operator()(const SampleKey& k) const {
        auto h=k.time.hash();const auto mix=[&](std::size_t n){h^=n+0x9e3779b9u+(h<<6)+(h>>2);};
        mix(k.state?k.state->hash():0);mix(bool(k.state));mix(k.actual_scope);mix(k.state_axis);
        mix(k.point.index());
        if(auto point=std::get_if<Class>(&k.point))mix(*point);
        else if(auto p=std::get_if<DispatchPointId>(&k.point)) {
          mix(p->epoch);mix(p->image);mix(p->root);mix(p->branch);mix(p->value);
        } else {
          const auto& typed=std::get<TypedPointId>(k.point);
          mix(typed.epoch);mix(typed.image);mix(typed.cell);mix(typed.address);mix(typed.scope);mix(typed.value);
        }
        return h;
      }
    };
    std::unordered_map<SampleKey,ContextId,SampleHash> sample_contexts;
    std::size_t sample_context_hits=0;
    explicit BodyScopeRegistry(BodyHeadIndex& h,std::size_t retained_capacity=default_retained_records):heads(h),
        max_retained_records(std::min(retained_capacity,epoch_work_allowance::maximum_retained_headers)),
        max_identity_slots(4*max_retained_records),
        bounds(lexical_scope::Limits{128,2048,16384,131072,max_depth},
            [&h](std::size_t amount){return h.charge(amount);}),context_keys(ContextLess{this}) {}
    // Keys/comparator/handles retain THIS immutable epoch's owner address.
    BodyScopeRegistry(const BodyScopeRegistry&)=delete;
    BodyScopeRegistry& operator=(const BodyScopeRegistry&)=delete;
    BodyScopeRegistry(BodyScopeRegistry&&)=delete;
    BodyScopeRegistry& operator=(BodyScopeRegistry&&)=delete;
    bool retainedCapacity(std::size_t stored,const char* label) {
      if(stored<max_retained_records)return true;
      ++retained_record_rejections;
      if(retained_record_rejections==1)retained_record_cap=label;
      return false;
    }
    bool contextCapacity(std::size_t stored,const char* label) {
      if(retainedCapacity(stored,label))return true;
      ++context_plan_rejections;
      if(context_plan_rejections==1)context_plan_cap=label;
      return false;
    }
    bool imageCapacity(const char* label) {
      if(retainedCapacity(images.size(),label))return true;
      ++image_rejections;
      if(image_rejections==1)image_cap=label;
      return false;
    }
    struct ScopeView {
     private:
      const BodyScopeRegistry* owner=nullptr;std::uint64_t epoch=0;
      lexical_scope::Id bounds=lexical_scope::unknown;Id parent=0;
      std::optional<Class> body_fold;
      friend struct BodyScopeRegistry;
    };
    struct OccurrenceUse {
     private:
      ScopeView scope;Class body=0;
      std::array<unsigned,8> body_path{};unsigned path_size=0;
      friend struct BodyScopeRegistry;
    };
    std::optional<ScopeView> scopeView(Id scope) {
      if(scope>=scopes.size() || !heads.charge())return {};
      ScopeView v;v.owner=this;v.epoch=context_epoch;v.bounds=scopes[scope].bounds;v.parent=scope;return v;
    }
    std::optional<ScopeView> bodyView(Id parent,Class fold);
    bool valid(const ScopeView& v) const {
      return v.owner==this && v.epoch==context_epoch && v.parent<scopes.size() &&
          (!v.body_fold || *v.body_fold<heads.snapshot.size());
    }
    std::optional<std::uint64_t> count(const ScopeView& v,std::uint32_t level) const {
      return valid(v)?bounds.count(v.bounds,level):std::nullopt;
    }
    std::optional<std::pair<unsigned,std::uint64_t>> domain(const ScopeView& v,std::uint32_t level) const;
    bool sameAssignment(Class cls,const ScopeView& reference,const ScopeView& actual,
        std::span<const std::uint32_t> covered);
    std::optional<OccurrenceUse> bodyUse(const ScopeView& v,std::span<const unsigned> path);
    std::optional<Class> transport(const OccurrenceUse& use,Id canonical,std::uint32_t bound_axis);
    std::optional<std::uint64_t> count(Id scope,std::uint32_t level) const {
      return scope<scopes.size()?bounds.count(scopes[scope].bounds,level):std::nullopt;
    }
    std::optional<Id> output();
    std::optional<Id> insideFold(Id parent,Class head);
    std::optional<Scope> bind(Id parent,std::uint32_t binder,std::uint64_t count);
    std::optional<Id> prefixScope(Id scope,std::uint32_t rank);
    std::optional<Id> insideSourceFold(Id parent,const s::Graph& graph,unsigned graph_tag,s::Id head);
    bool boundedAffine(const IndexAffine& affine,Id scope);
    std::optional<ImageId> image(Id scope,std::uint32_t level,Class coordinate);
    std::optional<CommonTranslationId> commonTranslation(ImageId source,ImageId actual);
    const CommonTranslation* commonTranslation(CommonTranslationId id) {
      if(!heads.charge() || id.owner!=this || id.epoch!=context_epoch || id.id>=common_translations.size())return nullptr;
      return &common_translations[id.id];
    }
    std::optional<ImageId> commonImage(CommonTranslationId translation,ImageId local);
    bool commonImagePair(CommonTranslationId translation,ImageId a,ImageId b);
    bool commonOutputAssignment(CommonTranslationId translation,Class source,Class actual,Id scope) {
      const auto* t=commonTranslation(translation);if(!t || !heads.charge(4))return false;
      const auto& s=images[t->source.id];const auto& a=images[t->actual.id];
      return s.scope==scope && a.scope==scope && s.coordinate==heads.owner(source) && a.coordinate==heads.owner(actual);
    }
    std::optional<PointId> point(ImageId image,Class coordinate,Id actual_scope);
    std::optional<std::uint64_t> globalValue(const ScalarPoint& point);
    std::optional<Class> pointWitness(const ScalarPoint& point);
    std::optional<PointId> point(ImageId image,const ScalarPoint& coordinate,Id actual_scope);
    std::optional<Image> view(ImageId image) {
      if(image.owner!=this || image.id>=images.size() || !heads.charge())return {};
      return images[image.id]; // Fixed scalar copy, never a reconstructed scope.
    }
    bool prefix(Id parent,Id child);
    std::optional<PointId> pointIn(PointId p,Id reference_scope,Id actual_scope);
    bool validate(const Context& c);
    bool lessContext(const Context& a,const Context& b);
    bool orderContextAxes(Context& context);
    std::optional<ContextId> admitContext(const Context& source);
    const Context* context(ContextId id) {
      if(!heads.charge())return nullptr;
      ++context_reads;
      if(id.owner!=this || id.epoch!=context_epoch || id.id>=contexts.size())return nullptr;
      return &contexts[id.id]; // Borrowed immutable record, O(1), no copy/hash.
    }
    std::optional<std::size_t> contextOrdinal(ContextId id) {
      return context(id)?std::optional{id.id}:std::nullopt;
    }
    std::optional<ContextId> sampleContext(ImageId time,std::optional<ImageId> state,
        Id actual_scope,ScalarPoint global_point,std::uint32_t state_axis=UINT32_MAX);
    bool projectLeaf(IndexAffine& reference,IndexAffine& actual,const Context& initial,
        unsigned active,Id reference_scope,Id actual_scope,
        std::span<const std::pair<std::uint32_t,std::uint32_t>> lexical);
    bool finishLeaf(const IndexAffine& reference,const IndexAffine& actual,
        Id reference_scope,Id actual_scope);
    bool leaf(IndexAffine& reference,IndexAffine& actual,const Context& initial,
        unsigned active,Id reference_scope,Id actual_scope,
        std::span<const std::pair<std::uint32_t,std::uint32_t>> lexical) {
      return projectLeaf(reference,actual,initial,active,reference_scope,actual_scope,lexical) &&
          finishLeaf(reference,actual,reference_scope,actual_scope);
    }
    std::optional<std::pair<Class,Class>> typedCellPair(Class piece,Class full);
    std::optional<ImageId> typedTimeImage(Context known,Class piece,Class anchor,std::uint32_t time_axis);
    std::optional<TypedPointId> typedPoint(ImageId role,Class actual,Id actual_scope);
    std::optional<ImageId> imageFromTypedCells(Context known,Class piece,Class full,
        std::uint32_t piece_axis,std::uint32_t full_axis);
    std::optional<ImageId> globalTypedImage(ImageId local,ImageId reference,Class full);
    std::optional<ImageId> relativeImage(ImageId actual,ImageId reference);
    std::optional<ImageId> imageInOriginalScope(ImageId source,Id scope,Class source_use);
  };
  // A scoped Sum proof borrows the SAME selected-DAG resolver and paired body
  // walker used by global reduction coverage. Its geometry is private: a
  // success under i=s*j+o never equates the unspecialized expression classes.
  struct ScopedSumProof {
    using Registry=BodyScopeRegistry;using Scope=Registry::Id;
    using Environment=std::vector<std::pair<std::uint32_t,std::uint32_t>>;
    using Linear=bounded_linear_reduction::Result;
    struct Axis {
      Scope reference_scope=0,actual_scope=0;
      std::uint32_t reference=0;std::optional<std::uint32_t> actual;
      std::uint64_t reference_count=0,actual_count=0;
      std::optional<std::uint64_t> stride,offset;
      bool operator==(const Axis&) const=default;
      bool operator<(const Axis& b) const {
        return std::tie(reference_scope,actual_scope,reference,actual,reference_count,actual_count,stride,offset)<
            std::tie(b.reference_scope,b.actual_scope,b.reference,b.actual,b.reference_count,b.actual_count,b.stride,b.offset);
      }
    };
    // Only this factory mints records; borrowed records never outlive the
    // sealed registry/index epoch. Two nonidentity axes is an admission bound,
    // not an inferred tensor-domain or a relaxation of Context's unit policy.
    struct Assignment {
      const ScopedSumProof* owner=nullptr;std::uint64_t epoch=0;
      std::vector<Axis> axes;
    };
    struct Equation {std::array<std::uint64_t,2> coefficients{};std::uint64_t value=0;};
    struct Geometry {Assignment assignment;std::vector<Equation> equations;};
    Registry& registry;BodyHeadIndex& heads;
    bounded_linear_reduction::Builder& builder;
    std::function<Linear(Class)> summarize;
    std::function<std::optional<Class>(bounded_linear_reduction::Atom)> reduction;
    std::function<void()> synchronize;
    scoped_work_budget::Guards*& planner_query;
    const std::size_t physical_nodes;
    const std::size_t selection_epoch;
    std::size_t queries=0,hits=0,completed=0;
    std::size_t identity_assignment_hits=0;
    // Scheduling only: complete deterministic witness selection depends on
    // the immutable owner and ORIGINAL lexical rank, not domain geometry.
    std::map<std::pair<Class,std::uint32_t>,std::optional<Class>> fold_selections;
    std::size_t fold_selection_hits=0,fold_selection_scans=0,fold_selection_absences=0,fold_selection_no_fold=0;
    const bool trace_work=sentryward::debug::environment("SENTRYWARD_EGRAPH_TRACE")!=nullptr;
    std::size_t piece_reports=0,failed_goal_reports=0,lattice_failure_reports=0,mapper_visits=0,mapper_tasks=0;
    using GoalKey=std::tuple<Class,Class,Scope,Scope,std::vector<Axis>,Environment>;
    struct Record {
      bool gray=true;std::optional<Assignment> result;
      std::optional<Assignment> projected_result;
    };
    std::map<GoalKey,Record> records;
    // Complete WHOLE numeric-function relations may ignore unused enclosing
    // binders only after checking every original domain and free capture.
    using Captures=std::vector<std::tuple<std::uint32_t,unsigned,std::uint64_t,std::uint64_t>>;
    using QualifiedKey=std::tuple<Class,Class,std::uint32_t,std::uint32_t,
        std::vector<Axis>,Captures,Captures,Environment>;
    struct CompletedRelation {const GoalKey* key;const Record* record;};
    std::map<QualifiedKey,CompletedRelation> qualified;
    std::size_t qualified_hits=0;
    ScopedSumProof(Registry& r,bounded_linear_reduction::Builder& b,
        std::function<Linear(Class)> s,
        std::function<std::optional<Class>(bounded_linear_reduction::Atom)> f,
        std::function<void()> sync,scoped_work_budget::Guards*& query)
        :registry(r),heads(r.heads),builder(b),summarize(std::move(s)),
         reduction(std::move(f)),synchronize(std::move(sync)),planner_query(query),physical_nodes(heads.snapshot.size()),
         selection_epoch(heads.engine.unions_) {}
    Assignment root() const {return {this,registry.context_epoch,{}};}
    struct QueryRelay {
      scoped_work_budget::Guards*& slot;scoped_work_budget::Guards* previous;
      QueryRelay(scoped_work_budget::Guards*& s,scoped_work_budget::Guards* q):slot(s),previous(s){slot=q;}
      ~QueryRelay(){slot=previous;}
    };
    // First sixteen goals plus eight later failures PER ScopedSumProof/reducer
    // cohort/epoch). Scalar samples only; nested work is inclusive in its
    // caller's phase, and no helper invoice or proof ledger is modified.
    struct GoalTrace {
      ScopedSumProof& proof;bool enabled;
      Class target,actual;Scope reference_scope,actual_scope;
      std::size_t goal,axis_count;
      std::size_t* query;
      std::size_t last_query=0,last_helper=0,phase=0;
      std::array<std::size_t,6> query_work{},helper_work{};
      std::size_t seed_terms=0,source_terms=0,pieces=0;
      bool success=false;
      GoalTrace(ScopedSumProof& p,Class t,Class a,Scope r,Scope s,std::size_t axes)
          :proof(p),enabled(p.trace_work),
           target(t),actual(a),reference_scope(r),actual_scope(s),goal(p.queries),axis_count(axes),
           query(p.heads.query_remaining) {
        if(enabled){last_query=query?*query:0;last_helper=proof.builder.work();}
      }
      void sample() {
        if(!enabled)return;
        const auto remaining=query?*query:0,helper=proof.builder.work();
        query_work[phase]+=last_query-remaining;helper_work[phase]+=helper-last_helper;
        last_query=remaining;last_helper=helper;
      }
      void next(std::size_t stage){if(enabled){sample();phase=stage;}}
      ~GoalTrace() noexcept {
        if(!enabled)return;
        sample();
        if(goal>16) {
          if(success || proof.failed_goal_reports>=8)return;
          ++proof.failed_goal_reports;
        }
        // Diagnostics must not replace a pending proof/cap exception.
        try {
          static constexpr std::array<const char*,6> labels{"setup","seed_summary","source_summary","body","lattice","result_merge"};
          SENTRYWARD_DEBUG_PRINT(egraph_82);
          for(std::size_t i=0;i<labels.size();++i)
            SENTRYWARD_DEBUG_PRINT(egraph_83);
          SENTRYWARD_DEBUG_PRINT(egraph_84);
        } catch(...) {}
      }
    };
    struct PieceTrace {
      ScopedSumProof& proof;bool enabled;
      Class target,actual;Scope reference_scope,actual_scope;
      std::uint64_t count;std::size_t ordinal=0,goal;
      std::size_t* query;std::size_t start_query=0,probes=0,hits=0;
      std::size_t choices=0,root_choices=0,visits=0,tasks=0,root_attempts=0,root_hits=0;
      std::size_t orientation_attempts=0,orientation_hits=0;
      bool complete=false;
      PieceTrace(ScopedSumProof& p,Class t,Class a,Scope r,Scope s,std::uint64_t n)
          :proof(p),enabled(p.trace_work && p.piece_reports<16),target(t),actual(a),
           reference_scope(r),actual_scope(s),count(n),goal(p.queries),query(p.heads.query_remaining) {
        if(!enabled)return;
        ordinal=++proof.piece_reports;start_query=query?*query:0;
        probes=proof.heads.header_probes;hits=proof.heads.header_probe_hits;
        choices=proof.heads.choices_cache.size();root_choices=proof.heads.root_choices_cache.size();
        visits=proof.mapper_visits;tasks=proof.mapper_tasks;
        root_attempts=proof.heads.root_first_attempts;root_hits=proof.heads.root_first_hits;
        orientation_attempts=proof.heads.root_orientation_attempts;orientation_hits=proof.heads.root_orientation_hits;
      }
      ~PieceTrace() noexcept {
        if(!enabled)return;
        try {
          SENTRYWARD_DEBUG_PRINT(egraph_85);
        } catch(...) {}
      }
    };
    bool valid(const Assignment& a) const {
      return a.owner==this && a.epoch==registry.context_epoch && a.axes.size()<=2;
    }
    bool plainFold(Class id);
    std::optional<Class> fold(Class cls,Scope parent);
    std::optional<Class> forest(Class cls);
    bool compatible(Class a,Class b);
    struct Selection {
      std::optional<std::pair<Class,Class>> ordinary;
      bool coverage=false;
    };
    Selection select(Class a,Class b,Scope r,Scope actual);
    bool domain(const Axis& axis,Scope r,Scope a);
    bool independent(Class cls,const Assignment& assignment,Scope r,Scope a,
        std::span<const std::pair<std::uint32_t,std::uint32_t>> lexical);
    bool sameOwnerIdentity(Class cls,Geometry& geometry,Scope r,Scope a,
        std::span<const std::pair<std::uint32_t,std::uint32_t>> lexical);
    bool leaf(IndexAffine x,IndexAffine y,Geometry& geometry,Scope r,Scope a,
        std::span<const std::pair<std::uint32_t,std::uint32_t>> lexical);
    bool solve(Geometry& geometry);
    struct Projection {
      Assignment local;
      std::array<std::size_t,2> original{};
    };
    std::optional<Projection> project(Class target,Class actual,Scope r,Scope a,
        const Assignment& incoming);
    bool merge(const Projection& projection,const Assignment& proved,
        const Assignment& incoming,Assignment& result);
    std::optional<QualifiedKey> qualify(Class witness,Class actual,Scope r,Scope a,
        const Assignment& active,const Environment& enclosing);
    bool rebind(const CompletedRelation& completed,const Projection& projection,
        Class witness,Class actual,Scope r,Scope a,const Assignment& incoming,Assignment& result);
    void remember(QualifiedKey key,const GoalKey& original,const Record& completed);
    bool sameCoefficient(const bounded_linear_reduction::Coefficient& x,
        const bounded_linear_reduction::Coefficient& y);
    bool nestedPointCoverage(Class outer,Scope r,Scope a,const Environment& enclosing,
        const Linear& source,const Assignment& incoming);
    bool related(Class target,Class actual,Scope r,Scope a,const Environment& enclosing,
        Assignment& incoming);
  };
  // A family is an owner-bound finite piecewise definition over ACTUAL original
  // scopes. Its private logical domain is never used as a graph lexical scope.
  // One catalog/interval Builder shares the epoch allowance across all joins.
  struct JoinedFamilies {
    using Registry=BodyScopeRegistry;
    struct Id {
      bool operator==(const Id&) const=default;
     private:const JoinedFamilies* owner=nullptr;std::size_t id=0;friend struct JoinedFamilies;
    };
    struct Piece {Class body=0;Registry::Id scope=0;std::array<Registry::ImageId,2> images{};};
    using Box=std::array<std::pair<std::uint64_t,std::uint64_t>,2>;
    using Capture=std::pair<std::uint32_t,Registry::Id>;
    struct Family {
      unsigned rank;Box domain;std::string_view type;
      std::vector<Piece> pieces;std::vector<Box> boxes;std::vector<Capture> captures;
    };
    Registry& registry;bounded_linear_reduction::Builder intervals;
    std::vector<std::unique_ptr<Family>> families;
    using PieceKey=std::tuple<Class,Registry::Id,std::uint32_t,Class,std::uint32_t,Class,Box>;
    using Key=std::tuple<unsigned,Box,std::vector<PieceKey>>;
    std::map<Key,Id> keys;
    static bounded_linear_reduction::Limits intervalLimits(std::size_t work);
    explicit JoinedFamilies(Registry& r,std::size_t work=epoch_work_allowance::default_work):registry(r),
        intervals(intervalLimits(work),[&r](std::size_t amount){return r.heads.charge(amount);}) {}
    bool sync() {
      // Primitive WorkCharge already debits the shared allowance before work.
      // Delta counters are diagnostics only; never charge nested work twice.
      return !intervals.exhausted() && !registry.heads.exhausted;
    }
    bool exhausted() const {return intervals.exhausted() || registry.heads.exhausted;}
    bool intervalExhausted() const {return intervals.exhausted();}
    std::size_t coverageWork() const {return intervals.work();}
    bool cover(std::span<const std::pair<std::uint64_t,std::uint64_t>> pieces,
        std::pair<std::uint64_t,std::uint64_t> domain);
    std::optional<Id> join(std::span<const Piece> pieces,unsigned rank,Box domain);
    const Family* view(Id id) const {
      return id.owner==this && id.id<families.size()?families[id.id].get():nullptr;
    }
    bool uniform(Id id,Class reference,Registry::Id scope,std::span<const std::uint32_t> axes,
        std::span<const Registry::ImageId> reference_images={});
  };
  // Proof-only bounded product view. Flatten only selected actual Multiply
  // heads, preserving every factor occurrence/original scope. No alternate
  // search or graph construction; chosen cycles and ambiguous pairings deny.
  struct BoundedProducts {
    using Registry=BodyScopeRegistry;
    struct Use {Class body;Registry::Id scope;bool operator==(const Use&) const=default;};
    Registry& registry;
    struct RelationKey {
      Use reference,actual;std::variant<Registry::Context,Registry::ContextId> context;
      RelationKey(Use r,Use a,const Registry::Context& c):reference(r),actual(a),
          context(std::in_place_type<Registry::Context>,c) {}
      RelationKey(Use r,Use a,Registry::ContextId c):reference(r),actual(a),
          context(std::in_place_type<Registry::ContextId>,c) {}
      // std::variant may have a Context-sized trivial copy representation.
      // These nontrivial copies construct ONLY the active alternative; token
      // list/map inserts never clone the inactive Context storage.
      RelationKey(const RelationKey& k):reference(k.reference),actual(k.actual),
          context(std::in_place_type<Registry::ContextId>) {copyContext(k);}
      RelationKey(RelationKey&& k):RelationKey(static_cast<const RelationKey&>(k)) {}
      RelationKey& operator=(const RelationKey& k) {
        if(this!=&k){reference=k.reference;actual=k.actual;copyContext(k);}return *this;
      }
      RelationKey& operator=(RelationKey&& k) {return *this=static_cast<const RelationKey&>(k);}
      bool operator==(const RelationKey&) const=default;
     private:
      void copyContext(const RelationKey& k) {
        if(const auto* id=std::get_if<Registry::ContextId>(&k.context))
          context.emplace<Registry::ContextId>(*id);
        else context.emplace<Registry::Context>(std::get<Registry::Context>(k.context));
      }
    };
    static_assert(!std::is_trivially_copy_constructible_v<RelationKey>);
    struct RelationHash {
      std::size_t operator()(const RelationKey& k) const {
        std::size_t h=k.reference.body;
        const auto mix=[&](std::size_t n){h^=n+0x9e3779b9u+(h<<6)+(h>>2);};
        mix(k.actual.body);mix(k.reference.scope);mix(k.actual.scope);
        if(const auto* id=std::get_if<Registry::ContextId>(&k.context)) {
          mix(id->hash());return h;
        }
        const auto& context=std::get<Registry::Context>(k.context);
        mix(context.reference_scope);mix(context.actual_scope);mix(context.axis_count);
        for(const auto& axis:context.axes) {
          mix(axis.reference_level);mix(axis.actual_level.value_or(UINT32_MAX));
          mix(axis.offset);mix(axis.extent);mix(axis.actual_coordinate);mix(axis.actual_shift);
          mix(bool(axis.point));mix(bool(axis.coordinate_image));
        }
        return h;
      }
    };
    struct CachedRelation {bool proved;std::list<RelationKey>::iterator age;};
    std::unordered_map<RelationKey,CachedRelation,RelationHash> relations;
    std::list<RelationKey> relation_age;
    std::map<std::tuple<std::size_t,Registry::Id,Registry::Id,unsigned>,Registry::ContextId> prefix_contexts;
    explicit BoundedProducts(Registry& r):registry(r) {}
    BoundedProducts(const BoundedProducts&)=delete;
    BoundedProducts& operator=(const BoundedProducts&)=delete;
    BoundedProducts(BoundedProducts&&)=delete;
    BoundedProducts& operator=(BoundedProducts&&)=delete;
    template<class Prove> bool memoRelation(const RelationKey& key,Prove prove) {
      if(auto it=relations.find(key);it!=relations.end()) {
        relation_age.splice(relation_age.end(),relation_age,it->second.age);return it->second.proved;
      }
      const bool proved=prove();
      if(registry.heads.exhausted || (registry.heads.remaining && !*registry.heads.remaining))return false;
      if(relations.size()>=4096) {
        if(!registry.heads.charge())return false;
        relations.erase(relation_age.front());relation_age.pop_front();
      }
      relation_age.push_back(key);relations.emplace(key,CachedRelation{proved,std::prev(relation_age.end())});return proved;
    }
    bool related(Use r,Use a,const Registry::Context& context);
    bool related(Use r,Use a,Registry::ContextId id);
    std::optional<Registry::Id> common(std::span<const Use> uses);
    bool admitted(Use use);
    std::optional<std::vector<Use>> factors(std::span<const Use> uses,bool strict_real_products=false);
    std::optional<unsigned> activeAxes(Use r,Use a,const Registry::Context& context);
    std::optional<Registry::Context> contextFor(Use r,Use a,const Registry::Context& context);
    std::optional<Registry::ContextId> contextFor(Use r,Use a,Registry::ContextId parent);
    struct AssignedUse {
     private:
      const BoundedProducts* owner=nullptr;std::uint64_t epoch=0;
      Use original{},canonical{};Registry::ContextId context;
      friend struct BoundedProducts;
    };
    std::optional<AssignedUse> canonicalUse(Use original,Registry::ContextId assignment,bool coordinate=false);
    std::optional<Use> assigned(const AssignedUse& use) {
      if(use.owner!=this || use.epoch!=registry.context_epoch || !registry.heads.charge() ||
          !registry.context(use.context))return {};
      return use.canonical; // Fixed scalar copy after the full original-use proof.
    }
    struct ReferenceId {
     private:const BoundedProducts* owner=nullptr;std::uint64_t epoch=0;std::size_t id=0;
      friend struct BoundedProducts;
    };
    struct Group {Class tag;Use use;std::size_t copies;};
    struct Groups {
      // Only [0,size) is initialized/read/copied. Avoid an eight-slot record
      // clone on every sample of a one/three-factor immutable reference.
      std::array<Group,8> data;std::size_t size=0,factor_count=0;
      Groups()=default;
      Groups(const Groups& g):size(g.size),factor_count(g.factor_count) {
        for(std::size_t i=0;i<size;++i)data[i]=g.data[i];
      }
      Groups(Groups&& g):Groups(static_cast<const Groups&>(g)) {}
      Groups& operator=(const Groups&)=delete;
      Groups& operator=(Groups&&)=delete;
    };
    static_assert(!std::is_trivially_copy_constructible_v<Groups>);
    std::deque<std::unique_ptr<const Groups>> reference_products;
    std::size_t reference_reads=0;
    std::optional<Groups> group(std::span<const Use> input,bool strict_real_products=false);
    std::optional<ReferenceId> referenceProduct(std::span<const Use> uses) {
      if(!registry.retainedCapacity(reference_products.size(),"reference_product_plan"))return {};
      auto groups=group(uses);if(!groups || !registry.heads.charge(4*groups->size+4))return {};
      ReferenceId id;id.owner=this;id.epoch=registry.context_epoch;id.id=reference_products.size();
      reference_products.push_back(std::make_unique<const Groups>(std::move(*groups)));return id;
    }
    const Groups* referenceProduct(ReferenceId id) {
      if(!registry.heads.charge() || id.owner!=this || id.epoch!=registry.context_epoch ||
          id.id>=reference_products.size())return nullptr;
      ++reference_reads;return reference_products[id.id].get(); // O(1), never reconstructs a plan.
    }
    template<class Related>
    std::optional<std::pair<std::size_t,std::size_t>> pairGroups(const Groups& r,
        const Groups& a,Related relate) {
      if(r.factor_count!=a.factor_count || r.size!=a.size)return {};
      for(std::size_t i=0;i<r.size;++i) {
        const auto& expected=r.data[i];const Group* found=nullptr;
        for(std::size_t j=0;j<a.size;++j) {
          if(!registry.heads.charge())return {};
          if(a.data[j].tag==expected.tag){found=&a.data[j];break;}
        }
        // Ambiguous distinct same-shape uses were already denied by group().
        // Occurrence multiplicity is retained; no candidate-pair product/search.
        if(!found || found->copies!=expected.copies || !relate(expected.use,found->use))return {};
      }
      return std::pair{r.factor_count,a.factor_count};
    }
    template<class Related>
    std::optional<std::pair<std::size_t,std::size_t>> equalReference(ReferenceId reference,
        std::span<const Use> actual,Related relate) {
      const auto* r=referenceProduct(reference);auto a=group(actual);
      return r && a?pairGroups(*r,*a,relate):std::nullopt;
    }
    using FactorPair=std::pair<Use,Use>;
    std::optional<std::vector<FactorPair>> pairedFactors(Use reference,Use actual);
    std::optional<std::pair<std::size_t,std::size_t>> equal(std::span<const Use> reference,
        std::span<const Use> actual,const Registry::Context& context);
    std::optional<std::pair<std::size_t,std::size_t>> equal(std::span<const Use> reference,
        std::span<const Use> actual,Registry::ContextId context) {
      if(!registry.context(context))return {};
      auto plan=referenceProduct(reference);return plan?equal(*plan,actual,context):std::nullopt;
    }
    std::optional<std::pair<std::size_t,std::size_t>> equal(ReferenceId reference,
        std::span<const Use> actual,Registry::ContextId context);
  };
  struct JoinedProducts {
    using Registry=BodyScopeRegistry;
    using Use=BoundedProducts::Use;
    struct Id {
      bool operator==(const Id&) const=default;
     private:const JoinedProducts* owner=nullptr;std::size_t id=0;friend struct JoinedProducts;
    };
    struct Piece {
      std::array<Use,8> factors{};unsigned count=0;
      std::array<Registry::ImageId,2> images{};
      std::optional<BoundedProducts::ReferenceId> reference_product={};
    };
    struct Family {
      JoinedFamilies::Id geometry;
      std::vector<Piece> pieces; // Same order as the certified geometry boxes.
    };
    Registry& registry;BoundedProducts products;JoinedFamilies geometry;
    std::vector<std::unique_ptr<Family>> families;
    using PieceKey=std::tuple<Class,Registry::Id,std::uint32_t,Class,std::uint32_t,Class,JoinedFamilies::Box>;
    explicit JoinedProducts(Registry& r,std::size_t work=epoch_work_allowance::default_work)
        :registry(r),products(r),geometry(r,work) {}
    const Family* view(Id id) const {
      return id.owner==this && id.id<families.size()?families[id.id].get():nullptr;
    }
    std::optional<std::vector<JoinedFamilies::Capture>> captures(
        std::span<const Use> factors,Registry::Id scope,
        std::span<const std::uint32_t> axes);
    std::optional<Id> join(std::span<const Piece> input,JoinedFamilies::Box domain);
    bool sample(Id id,std::span<const Use> actual,const Registry::ScalarPoint& global_point,std::uint32_t state_axis);
  };
  // The EXISTING stored Select cursor, factored into a private admission
  // boundary. It visits each actual selector once, validates exact typed Eq
  // and ORIGINAL coordinate image, and retains only stored in-image cases.
  // No alternate parser, ancestor walk, logical-domain iteration or enode.
  struct TypedDispatch {
    using Registry=BodyScopeRegistry;using Point=Registry::ScalarPoint;
    struct Step {std::uint64_t endpoint;Point point;Class branch;bool inside,done;};
    Registry& registry;JoinedFamilies& coverage;Registry::Id scope;
    Class root,current,coordinate=std::numeric_limits<Class>::max();
    std::optional<Registry::ImageId> image;
    std::unordered_set<Class> active;
    std::vector<std::pair<std::uint64_t,std::uint64_t>> excluded;
    bool closed=false;
    TypedDispatch(Registry& r,JoinedFamilies& c,Registry::Id s,Class start)
        :registry(r),coverage(c),scope(s),root(r.heads.owner(start)),current(root) {}
    std::optional<Step> next();
  };
  bool proveFirstFinitePrefixOutput(std::span<const Class> left,std::span<const Class> right);
  // Trace-only actual-cohort extraction. No prefix/dispatch/region facts or unions.
#if defined(SENTRYWARD_DEVELOPMENT_DIAGNOSTICS) || defined(SENTRYWARD_ENABLE_COMPRESSION_AUDIT)
#include "sentryward/development/diagnoseFinitePrefixSourceHeads.inc"
#else
  void diagnoseFinitePrefixSourceHeads(std::span<const Class> ,std::span<const Class> ) {}
#endif

  template<class Members>
  std::optional<std::pair<std::uint64_t,std::uint64_t>> affineBodyMap(
      Class target,Class block,std::uint32_t binder,const std::vector<Key>& snapshot,
      const Members& members,bool singleton=false,
      std::size_t* remaining=nullptr,AffineCache* shared_affine=nullptr,
      BodyHeadIndex* shared_heads=nullptr) {
    // A supplied sealed-epoch index already owns these hints. Do not allocate
    // or initialize a second dense scalar memo for every body-map query.
    if(shared_heads)return affineBodyMapWithHeads(target,block,binder,snapshot,
        singleton,remaining,*shared_heads);
    std::size_t local_remaining=131072;
    AffineCache local_affine;
    auto& affine_cache=shared_affine?*shared_affine:local_affine;
    BodyHeadIndex local_heads(*this,snapshot,members,affine_cache,
        remaining?remaining:&local_remaining);
    return affineBodyMapWithHeads(target,block,binder,snapshot,singleton,remaining,local_heads);
  }
  std::optional<std::pair<std::uint64_t,std::uint64_t>> affineBodyMapWithHeads(
      Class target,Class block,std::uint32_t binder,const std::vector<Key>& snapshot,
      bool singleton,std::size_t* remaining,BodyHeadIndex& heads,
      std::uint32_t block_binder=UINT32_MAX,BodyScopeRegistry* registry=nullptr,
      const BodyScopeRegistry::Context* initial=nullptr,
      const BodyScopeRegistry::ContextId* admitted=nullptr,ScopedSumProof* sums=nullptr,
      ScopedSumProof::Assignment* assignment=nullptr,
      BodyScopeRegistry::Id reference_scope=0,BodyScopeRegistry::Id actual_scope=0,
      const ScopedSumProof::Environment* enclosing=nullptr);
  template<class Pieces>
  bool bitPermutedFoldCover(const Pieces& pieces,Class target,std::uint32_t binder,
      const std::vector<Key>& snapshot,BodyHeadIndex& heads) {
    // A bounded forest of stored blocks, NOT a loop over reduction elements.
    // Prove start(mask,t) = block_size * (mask XOR bits(t)). Every mask must
    // occur once; XOR by a common bit vector is a bijection for every t.
    if(!std::has_single_bit(pieces.size()) || pieces.size()>64)return false;
    const auto deny=[&](const char* reason,Class a=0,Class b=0) {
      if(sentryward::debug::environment("SENTRYWARD_EGRAPH_TRACE"))SENTRYWARD_DEBUG_PRINT(egraph_141);
      return false;
    };
    const auto count=pieces.front().count;
    if(!count)return false;
    for(const auto& p:pieces)if(p.singleton || p.count!=count)return false;
    using Digit=std::pair<std::uint32_t,unsigned>;
    struct Form {
      std::uint64_t constant=0,axis=0;
      std::map<Digit,std::uint64_t> digits;
      bool operator==(const Form&) const = default;
    };
    std::map<Class,std::optional<Form>> cache;
    std::function<std::optional<Form>(Class,unsigned)> parse;
    const auto literal=[&](const Key& n)->std::optional<std::uint64_t> {
      if(n.kind!=static_cast<int>(s::Kind::Constant))return {};
      std::uint64_t u=0;auto r=std::from_chars(n.name.data(),n.name.data()+n.name.size(),u);
      if(r.ec==std::errc{} && r.ptr==n.name.data()+n.name.size())return u;
      std::int64_t v=0;r=std::from_chars(n.name.data(),n.name.data()+n.name.size(),v);
      if(r.ec==std::errc{} && r.ptr==n.name.data()+n.name.size())return static_cast<std::uint64_t>(v);
      return {};
    };
    parse=[&](Class cls,unsigned depth)->std::optional<Form> {
      if(!heads.charge() || depth>128)return {};cls=heads.owner(cls);
      if(auto i=cache.find(cls);i!=cache.end()) {
        if(i->second && !heads.charge(i->second->digits.size()+1))return {};
        return i->second;
      }
      if(cache.size()>=4096)return {};
      auto selected=heads.preferred(cls);if(!selected)return {};
      const auto& n=snapshot[*selected];std::optional<Form> out;
      if((n.type!="u64" && n.type!="exact:u64") || !n.provenance.empty() || !n.attributes.empty())return {};
      if(auto c=literal(n))out=Form{*c,0,{}};
      else if(n.kind==static_cast<int>(s::Kind::Index) && n.index_level==binder)out=Form{0,1,{}};
      else if(n.kind==static_cast<int>(s::Kind::Index) || n.name=="Divide" || n.name=="Remainder") {
        // Decode a power-of-two slice of ONE bounded unsigned coordinate.
        auto current=*selected;unsigned low=0,high=64;bool valid=true;
        for(unsigned steps=0;steps<64;++steps) {
          if(!heads.charge()){valid=false;break;}const auto& k=snapshot[current];
          if(k.kind==static_cast<int>(s::Kind::Index))break;
          if(k.kind!=static_cast<int>(s::Kind::Operation) || !k.attributes.empty() ||
              !k.provenance.empty() || k.type!=n.type || k.children.size()!=2 ||
              (k.name!="Divide" && k.name!="Remainder")){valid=false;break;}
          auto divisor=heads.preferred(k.children[1]);if(!divisor){valid=false;break;}
          auto c=literal(snapshot[*divisor]);
          if(!c || !std::has_single_bit(*c)){valid=false;break;}
          auto bits=static_cast<unsigned>(std::countr_zero(*c));
          if(k.name=="Remainder")high=std::min(high,bits);
          else {if(low+bits>=64){valid=false;break;}low+=bits;high=std::min(64u,high+bits);}
          auto child=heads.preferred(k.children[0]);if(!child){valid=false;break;}current=*child;
        }
        const auto& base=snapshot[current];
        if(valid && base.kind==static_cast<int>(s::Kind::Index) && base.index_level!=binder &&
            base.index_level<options_.outer_index_extents.size()) {
          auto extent=options_.outer_index_extents[base.index_level];
          if(extent) {
            high=std::min(high,static_cast<unsigned>(std::bit_width(extent-1)));
            Form f;
            for(unsigned bit=low;bit<high;++bit) {
              if(!heads.charge())return {};
              f.digits[{base.index_level,bit}]=std::uint64_t{1}<<(bit-low);
            }
            out=std::move(f);
          }
        }
      } else if(n.kind==static_cast<int>(s::Kind::Operation) && n.children.size()==2 &&
          (n.name=="Add" || n.name=="Subtract" || n.name=="Multiply")) {
        auto a=parse(n.children[0],depth+1),b=parse(n.children[1],depth+1);
        if(a && b) {
          if(!heads.charge(a->digits.size()+b->digits.size()+1))return {};
          if(n.name=="Multiply") {
            if(!a->axis && a->digits.empty())std::swap(a,b);
            if(!b->axis && b->digits.empty()) {
              a->constant*=b->constant;a->axis*=b->constant;
              for(auto& [d,c]:a->digits)c*=b->constant;out=std::move(a);
            }
          } else {
            auto sign=n.name=="Add"?std::uint64_t{1}:UINT64_MAX;
            a->constant+=sign*b->constant;a->axis+=sign*b->axis;
            for(auto [d,c]:b->digits)a->digits[d]+=sign*c;out=std::move(a);
          }
          if(out)std::erase_if(out->digits,[](auto& e){return !e.second;});
        }
      }
      if(out && (out->digits.size()>64 || !heads.charge(out->digits.size()+1)))return {};
      cache.emplace(cls,out);return out;
    };
    std::map<Digit,std::uint64_t> common_digits;
    std::set<std::uint64_t> masks;
    bool first=true;
    for(const auto& piece:pieces) {
      std::vector<std::pair<Class,Class>> work{{target,piece.body}};
      std::set<std::pair<Class,Class>> visited;
      std::optional<Form> image;
      while(!work.empty()) {
        if(!heads.charge() || visited.size()>=4096)return false;
        auto [a,b]=work.back();work.pop_back();a=heads.owner(a);b=heads.owner(b);
        if(!visited.emplace(a,b).second)continue;
        if(a==b && !heads.depends(a,binder))continue;
        auto x=parse(a,0),y=parse(b,0);
        if(x && y && x->axis) {
          if(!x->digits.empty() || x->axis!=y->axis || x->axis>INT64_MAX)return false;
          Form map=*y;map.constant-=x->constant;map.axis=0;
          // The block offset is interpreted in the u64 ring; divisibility
          // is checked in signed coefficients before the nonwrapping cover.
          const auto divide=[&](std::uint64_t v)->std::optional<std::uint64_t> {
            auto signed_v=std::bit_cast<std::int64_t>(v);
            if(signed_v%static_cast<std::int64_t>(x->axis))return {};
            return static_cast<std::uint64_t>(signed_v/static_cast<std::int64_t>(x->axis));
          };
          auto c=divide(map.constant);if(!c)return false;map.constant=*c;
          for(auto& [d,v]:map.digits){auto z=divide(v);if(!z)return false;v=*z;}
          if(image && *image!=map)return false;image=std::move(map);continue;
        }
        auto pair=heads.matched(a,b);if(!pair)return deny("body_head",a,b);
        const auto& p=snapshot[pair->first];const auto& q=snapshot[pair->second];
        if(p.fold || q.fold || p.kind==static_cast<int>(s::Kind::Index) ||
            p.children.size()!=q.children.size() || p.children.empty())return deny("body_leaf",a,b);
        for(std::size_t j=0;j<p.children.size();++j)work.emplace_back(p.children[j],q.children[j]);
      }
      if(!image || image->constant%count)return deny("missing_image");
      auto mask=image->constant/count;if(mask>=pieces.size() || !masks.insert(mask).second)return deny("mask",mask);
      std::map<Digit,std::uint64_t> digits;std::uint64_t positions=0;
      for(auto [d,v]:image->digits) {
        if(!heads.charge())return false;
        const bool negative=std::bit_cast<std::int64_t>(v)<0;auto magnitude=negative?0-v:v;
        if(magnitude%count)return false;auto weight=magnitude/count;
        if(!std::has_single_bit(weight) || weight>=pieces.size() || (positions&weight) ||
            bool(mask&weight)!=negative)return deny("digit",mask,weight);
        positions|=weight;digits[d]=weight;
      }
      if(first){common_digits=std::move(digits);first=false;}
      else if(digits!=common_digits)return deny("digit_basis");
    }
    return masks.size()==pieces.size();
  }
  void coalesceInterleavedFolds(Class id,const Key& root,const std::vector<Key>& snapshot,
      const std::vector<std::vector<Class>>& members,
      const std::map<std::tuple<std::string,std::uint32_t,std::uint64_t>,std::vector<Class>>& targets);
  std::optional<IndexAffine> affineValue(Class id,AffineCache& cache,unsigned depth=0,
                                        std::size_t* remaining=nullptr,
                                        std::size_t max_entries=0,std::size_t max_terms=0);
  bool shiftedProjectedBody(Class first,Class next,std::uint32_t binder,std::uint64_t shift,
                            std::uint64_t count);
  void concatenateProjectedFold(Key& key);

  using ScopeId=lexical_scope::Id;
  static constexpr std::uint32_t branch_fact_binder=UINT32_MAX-1;
  struct BranchScopeRecord {std::vector<std::uint64_t> counts;};
  std::vector<BranchScopeRecord> branch_scopes_;
  std::map<ScopeId,std::size_t> branch_scope_ids_;
  std::vector<unsigned char> projection_branch_sensitive_;
  std::unordered_set<Class> branch_proved_wrappers_;
  std::vector<Class> branch_candidates_;
  std::optional<Class> branch_fact_anchor_;
  std::size_t branch_remaining_=262144;
  std::size_t branch_trace_remaining_=16;
  bool branch_scope_used_=false;
  std::size_t branch_requests_=0,branch_admitted_=0,branch_indicators_=0;
  bool branchCharge(std::size_t amount=1);
  static bool branchPredicateSource(const Key& k,const std::vector<Key>& arena);
  static bool predicateFloatConversion(const Key& k,const std::vector<Key>& arena);
  static bool predicateConjunction(const Key& k,const std::vector<Key>& arena) {
    return k.kind==static_cast<int>(s::Kind::Operation) && k.name=="BitAnd" &&
        k.type=="pred" && k.attributes.empty() && k.provenance.empty() && k.children.size()==2 &&
        arena[k.children[0]].type=="pred" && arena[k.children[1]].type=="pred";
  }
  void prepareBranchFacts();
  std::optional<std::size_t> branchScope(ScopeId id);
  static std::optional<std::pair<std::size_t,std::string>> branchWrapper(const Key& k);
  std::optional<std::vector<std::pair<std::uint64_t,std::uint64_t>>> branchMask(
      Class mask,std::uint32_t binder,std::uint64_t extent,const std::vector<Key>& snapshot,
      BodyHeadIndex& heads);
  void rewriteScopedBranchPredicates(const std::vector<Key>& snapshot,
      const std::vector<std::vector<Class>>& members);
  struct ProjectionKey {
    Class source;ScopeId scope;
    bool operator==(const ProjectionKey&) const=default;
  };
  struct ProjectionHash {
    std::size_t operator()(const ProjectionKey& k) const {
      const auto h=std::hash<Class>{}(k.source),s=std::hash<ScopeId>{}(k.scope);
      return h^(s+0x9e3779b9+(h<<6)+(h>>2));
    }
  };
  using ProjectionMap=std::unordered_map<ProjectionKey,Class,ProjectionHash>;
  std::unique_ptr<lexical_scope::Catalog> projection_scopes_;
  ScopeId projection_root_scope_=lexical_scope::unknown;
  std::vector<unsigned char> projection_sat_sensitive_;
  std::vector<unsigned char> projection_integer_sensitive_;
  std::vector<Class> projection_source_ids_;
  AffineCache projection_affine_cache_;
  std::unique_ptr<BodyHeadIndex> projection_body_heads_;
  std::size_t projection_sat_remaining_=2000000,projection_sat_requests_=0;
  struct MaxSample {Class body;std::uint32_t binder;std::uint64_t count;bool singleton;};
  struct MaxSamples {bool valid=false;std::vector<MaxSample> samples;};
  std::unordered_map<Class,MaxSamples> projection_max_samples_;
  std::size_t projection_sample_count_=0;
  std::map<std::tuple<Class,Class,ScopeId>,bool> projection_max_membership_;
  std::unordered_map<ProjectionKey,bool,ProjectionHash> projection_sat_certificates_;
  bool projectionCharge(std::size_t amount=1);
  static bool scopedSatHead(const Key& k,const std::vector<Key>& arena);
  template<class Op>
  static std::optional<std::pair<Class,Class>> localDifference(
      Class id,const std::vector<Key>& arena,const Op& op) {
    if(op(id,"Subtract",2))return std::pair{arena[id].children[0],arena[id].children[1]};
    if(op(id,"Add",2))for(unsigned side=0;side<2;++side) {
      const auto neg=arena[id].children[side];
      if(op(neg,"Negate",1))return std::pair{arena[id].children[1-side],arena[neg].children[0]};
    }
    return {};
  }
  bool scopedExponentialSource(Class exponential) const;
  void initializeProjectionScope();
  bool projectionFinite(Class id) const {
    // Never transfer a finite fact from exact/raw/opaque representations.
    return id<nodes_.size() && parents_[id]==id && nodes_[id].type=="real" &&
        nodes_[id].provenance.empty() && nodes_[id].name!="ExactNumericValue" && finite_[id];
  }
  bool projectedOperation(Class id,std::string_view name,std::size_t arity) const {
    if(id>=nodes_.size())return false;
    const auto& k=nodes_[id];
    return k.type=="real" && k.provenance.empty() && idealOperation(k) &&
        k.name==name && k.children.size()==arity;
  }
  const MaxSamples* projectionMaxSamples(Class maximum);
  bool projectedMaxContains(Class maximum,Class score,ScopeId scope);
  bool projectedInactiveSaturation(Class conversion,Class exponential,ScopeId scope);

  static bool scopedIndexRemainderHead(const Key& k,const std::vector<Key>& arena);
  bool scopedIndexRemainder(const Key& source,ScopeId scope);

  Class project(Class root,bool numeric);
  ProjectionMap exact_projection_,numeric_projection_,predicate_projection_;
  void poll() const {
    if(options_.max_work && ++work_>options_.max_work)throw Limit{"work_budget"};
    if (std::chrono::steady_clock::now() - started_ >= options_.timeout)
      throw Limit{"time_budget"};
  }
  Class find(Class id);
  std::pair<std::int64_t,bool> scopeProperties(const Key& k);
  std::pair<bool,bool> positivity(const Key& k);
  bool provesFinite(const Key& k);
  std::uint8_t provesPredicate(const Key& k,Class id);
  integer_interval::Result integerRange(const Key& k);
  void rewriteUnsignedShift(Class id,const Key& k);
  void rewriteExactIntegerNumeric(Class id,const Key& k,
      const std::vector<std::vector<Class>>& members);
  std::optional<Key> singletonConstant(const Key& k);
  bool refineIntegerRange(Class root,const integer_interval::Result& proof);
  bool provesZero(const Key& k);
  bool provesNonzero(const Key& k,bool positive);
  // Only e-class summaries are consulted by transfer functions. Every stored
  // child edge is registered once; union splices use lists instead of copying
  // an ever-growing array. On a fixed DAG scalar facts change a bounded number
  // of times; binder-set work also depends on lexical depth, never Fold count.
  void enqueueProperty(Class id) {
    if(!property_queued_[id]){property_queued_[id]=true;property_queue_.push_back(id);}
  }
  void notifyDependents(Class root) {
    for(auto user:dependents_[find(root)])enqueueProperty(user);
  }
  bool updateProperties(Class id);
  void propagateProperties() {
    for(std::size_t cursor=0;cursor<property_queue_.size();++cursor) {
      poll();const auto id=property_queue_[cursor];property_queued_[id]=false;updateProperties(id);
    }
    property_queue_.clear();
  }
  bool dependsOn(Class id,std::uint32_t binder) {
    const auto& levels=free_indices_[find(id)];
    return std::binary_search(levels.begin(),levels.end(),binder);
  }
  void canonicalizeCommutative(Key& key);
  std::optional<Class> exactIntegerIdentity(const Key& k) const;
  Class intern(Key k);
  bool unite(Class a, Class b);
  void rewrite(Class a, Class b) {
    if (unite(a,b)) ++statistics_.rewrite_unions;
  }
  void rebuild();

  Options options_;
  std::chrono::steady_clock::time_point started_;
  std::vector<Key> nodes_;
  std::unique_ptr<CompleteHeaderCatalog> complete_headers_;
  std::size_t complete_header_generation_=1;
  std::vector<Class> parents_;
  std::vector<Class> minimum_;
  stored_class_catalog::Members class_members_;
  std::vector<Class> real_head_ids_,fold_head_ids_;
  std::vector<std::size_t> sizes_;
  std::vector<bool> finite_,nonnegative_,positive_,nonzero_,zero_,negative_infinity_,positive_infinity_;
  std::vector<integer_interval::Result> integer_ranges_;
  std::vector<std::uint8_t> predicate_;
  std::vector<std::vector<std::uint32_t>> free_indices_;
  std::vector<std::list<Class>> dependents_;
  std::vector<Class> property_queue_;
  std::vector<bool> property_queued_;
  std::unordered_set<Class> factored_folds_,factored_sums_,joined_sums_;
  std::unordered_set<Class> factored_selects_,merged_selects_;
  std::unordered_set<Class> negated_products_;
  std::map<std::tuple<Class,Class,std::uint32_t,std::uint64_t,std::uint64_t>,bool> shifted_body_matches_;
  bool shifted_body_cap_reported_=false;
  std::unordered_set<Class> split_seed_folds_;
  std::unordered_set<Class> expanded_exponentials_;
  std::unordered_set<Class> combined_exponentials_;
  std::unordered_set<Class> coalesced_forests_;
  std::unordered_set<Class> flattened_folds_;
  std::size_t imported_node_count_=0;
  std::vector<std::int64_t> max_index_;
  std::vector<bool> has_fold_;
  std::unordered_map<Key,Class,Hash> memo_;
  std::size_t unions_ = 0;
  Statistics statistics_;
  mutable std::size_t work_=0;
  std::unordered_set<std::string> positive_parameters_;
};
}  // namespace


} // namespace sentryward::equivalence

#include "egraph_import.hpp"
#include "egraph_compare.hpp"
#include "egraph_scoped_proofs.hpp"
#include "egraph_linear_reduction.hpp"
#include "egraph_normal_forms.hpp"
#include "egraph_selects.hpp"
#include "egraph_rectangles.hpp"
#include "egraph_finite_prefix.hpp"
#include "egraph_fold_domains.hpp"
#include "egraph_branch_scopes.hpp"
#include "egraph_projection.hpp"
#include "egraph_properties.hpp"
#include "egraph_body_heads.hpp"
#include "egraph_body_scopes.hpp"
#include "egraph_scoped_sums.hpp"
#include "egraph_joined_families.hpp"
#include "egraph_bounded_products.hpp"
#include "egraph_joined_products.hpp"
#include "egraph_typed_dispatch.hpp"
