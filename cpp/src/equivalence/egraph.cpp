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
#include <atomic>
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

  FrozenIndex freeze(std::span<const Class> roots={}) {
    FrozenIndex result;result.finite.resize(nodes_.size());
    for(auto root:roots)result.imported_roots.push_back(find(root));
    for(Class id=0;id<nodes_.size();++id)result.finite[find(id)]=finite_[find(id)];
    for(Class id=0;id<nodes_.size();++id) {
      auto key=nodes_[id];for(auto& child:key.children)child=find(child);
      canonicalizeCommutative(key);
      if(idealOperation(key))key.attributes="ideal_arithmetic";
      result.memo.emplace(std::move(key),find(id));
    }
    return result;
  }

  std::vector<Class> import(const s::Graph& graph, std::span<const s::Ref> roots,
                            std::vector<unsigned char>* storage_roots=nullptr,
                            std::span<const s::Ref> witnesses={},std::vector<Class>* witness_outputs=nullptr,
                            bool captured_coordinate=false) {
    graph.validateRoots(roots,captured_coordinate?1:options_.outer_binders);
    const auto normalizeUse = [&](s::Ref ref) {
      for(;;) {
        poll();
        if(!ref.conversion.empty()) {
          if(auto inactive=normalizeInactiveFiniteSaturation(graph,ref))ref=std::move(*inactive);
          if(auto normalized=normalizeConversions(graph.node(ref).type,graph.type(ref),ref.conversion)) {
            ref.conversion=*normalized;
            if(ref.conversion.empty())ref.converted_type.clear();
          }
        }
        auto reduced=packedLowUse(graph,ref);
        if(!reduced)reduced=packedHighUse(graph,ref);
        if(!reduced)return ref;
        ref=std::move(*reduced);
      }
    };
    std::vector<bool> live(graph.size(), false);
    std::vector<s::Id> pending, order;
    for (const auto& root : roots) { poll(); pending.push_back(normalizeUse(root).node); }
    while (!pending.empty()) {
      poll();
      auto id = pending.back(); pending.pop_back();
      if (live[id]) continue;
      live[id] = true; order.push_back(id);
      for (const auto& operand : graph.node(id).operands) pending.push_back(normalizeUse(operand).node);
    }
    std::sort(order.begin(), order.end());
    const bool certify_storage=storage_roots && options_.storage_output_bytes &&
        options_.storage_output_bytes<=8;
    std::vector<unsigned char> storage_nodes(certify_storage?graph.size():0);
    const auto storageUse=[&](const s::Ref& ref) {
      if(!certify_storage || !storage_nodes.at(ref.node) || !ref.pointer_provenance.empty())return false;
      const auto expected=8*options_.storage_output_bytes;
      auto current=graph.node(ref).type;
      if(bitWidth(current)!=expected)return false;
      // Inspect the original edge, before ideal/numeric canonicalization.
      // Only exact, same-width bitcasts may carry a bit-copy certificate.
      for(std::size_t position=0;position<ref.conversion.size();) {
        poll();const auto end=ref.conversion.find("; ",position);
        const auto step=ref.conversion.substr(position,end==std::string::npos?end:end-position);
        const auto arrow=step.find(" -> ");
        if(!step.starts_with("Bitcast(") || arrow==std::string::npos || step.back()!=')')return false;
        const auto from=step.substr(8,arrow-8),to=step.substr(arrow+4,step.size()-arrow-5);
        if(from!=current || bitWidth(from)!=expected || bitWidth(to)!=expected)return false;
        current=to;
        if(end==std::string::npos)break;
        position=end+2;
      }
      return current==graph.type(ref);
    };
    struct FoldPrefix { std::uint64_t count; Class initial, imported_body; };
    std::unordered_map<s::Id, FoldPrefix> prefixes;
    std::unordered_map<s::Id, Class> mapping;
    std::unordered_map<s::Id, std::optional<ConstantValue>> constants;
    const auto knownUse = [&](const s::Ref& ref)->std::optional<ConstantValue> {
      if(!ref.pointer_provenance.empty())return {};
      const auto found=constants.find(ref.node);if(found==constants.end())return {};
      const auto& value=found->second;if(!value)return {};
      return constantValue(value->type,value->literal,ref.conversion,ref.converted_type);
    };
    const auto use = [&](s::Ref ref) {
      ref=normalizeUse(std::move(ref));
      if(auto value=knownUse(ref)) {
        Key constant;constant.kind=static_cast<int>(s::Kind::Constant);
        constant.type=value->type;constant.name=value->literal;return intern(std::move(constant));
      }
      Class base = mapping.at(ref.node);
      if(ref.pointer_provenance.empty() && graph.node(ref).kind==s::Kind::InitialCell) {
        if(auto cast=leadingBitcast(graph.node(ref).type,ref.conversion)) {
          auto read=nodes_[base];read.type=cast->target;
          base=intern(std::move(read));ref.conversion=cast->remainder;
        }
      }
      if (ref.conversion.empty() && ref.pointer_provenance.empty()) return base;
      if(ref.pointer_provenance.empty() && ref.conversion.find("; ")!=std::string::npos) {
        auto text=ref.conversion;
        while(!text.empty()) {
          const auto end=text.find("; ");const auto step=text.substr(0,end);
          const auto arrow=step.find(" -> ");
          if(arrow==std::string::npos || step.back()!=')')throw std::invalid_argument("invalid_conversion_chain");
          Key k;k.kind=6;k.name=step;k.type=step.substr(arrow+4,step.size()-arrow-5);k.children={base};
          base=intern(std::move(k));
          if(end==std::string::npos)break;text.erase(0,end+2);
        }
        return base;
      }
      Key k;
      k.kind = 6;
      k.type = graph.type(ref);
      k.name = ref.conversion;
      k.provenance = ref.pointer_provenance;
      k.children = {base};
      return intern(std::move(k));
    };
    std::unordered_map<s::Id,IndexAffine> index_affines;
    for (auto id : order) {
      poll();
      const auto& node = graph.node(id);
      if(certify_storage) {
        bool copy=node.kind==s::Kind::InitialCell && !node.name.empty() &&
            node.attributes.empty() && !node.fold && node.operands.size()==1 &&
            node.operands[0].pointer_provenance.empty() && graph.type(node.operands[0])=="u64" &&
            bitWidth(node.type)==8*options_.storage_output_bytes;
        if(node.kind==s::Kind::Operation && node.name=="Select" && !node.fold && node.operands.size()==3 &&
            (node.attributes.empty() || node.attributes=="ideal_arithmetic") &&
            graph.type(node.operands[0])=="pred" && node.operands[0].conversion.empty() &&
            node.operands[0].pointer_provenance.empty() &&
            graph.type(node.operands[1])==node.type && graph.type(node.operands[2])==node.type)
          copy=storageUse(node.operands[1]) && storageUse(node.operands[2]);
        storage_nodes[id]=copy;
      }
      Key k{static_cast<int>(node.kind), node.type, node.name, node.attributes, {},
            node.fold, node.index_level, {}};
      if(captured_coordinate && node.kind==s::Kind::Index) {
        if(node.index_level)throw std::invalid_argument("captured_coordinate_rank");
        k.index_level=branch_fact_binder;
      }
      for (const auto& operand : node.operands) k.children.push_back(use(operand));
      std::optional<ConstantValue> evaluated;
      if(node.kind==s::Kind::Constant)evaluated=constantValue(node.type,node.name);
      else if(node.kind==s::Kind::Operation) {
        std::vector<ConstantValue> args;bool known=true;
        for(const auto& operand:node.operands) {
          auto value=knownUse(operand);if(!value){known=false;break;}
          args.push_back(std::move(*value));
        }
        if(known)evaluated=constantOperation(node.name,node.type,node.attributes,args);
      }
      constants.emplace(id,evaluated);
      if(evaluated) {
        k=Key{};k.kind=static_cast<int>(s::Kind::Constant);
        k.type=evaluated->type;k.name=evaluated->literal;
      }
      if(node.kind==s::Kind::Constant && !evaluated && k.type!="real" && realType(k.type) && k.attributes.empty() &&
          !floatingInfinity(k.type,k.name))
        k.attributes="encoding="+k.type;
      std::optional<IndexAffine> form;
      if(node.type=="u64" && node.attributes.empty()) {
        if(node.kind==s::Kind::Index)form=IndexAffine{0,{{k.index_level,1}}};
        else if(evaluated && evaluated->type=="u64") {
          std::uint64_t value=0;
          const auto& text=evaluated->literal;
          auto parsed=std::from_chars(text.data(),text.data()+text.size(),value);
          if(parsed.ec==std::errc{} && parsed.ptr==text.data()+text.size())form=IndexAffine{value,{}};
        } else if(node.kind==s::Kind::Operation && node.operands.size()==2) {
          const auto& x=node.operands[0];const auto& y=node.operands[1];
          if(x.conversion.empty() && y.conversion.empty() && x.pointer_provenance.empty() &&
              y.pointer_provenance.empty() && index_affines.contains(x.node) && index_affines.contains(y.node))
            form=combineAffine(node.name,index_affines.at(x.node),index_affines.at(y.node));
        }
      }
      if(form) {
        index_affines.emplace(id,*form);
        const auto constant=[&](std::uint64_t value) {
          Key key;key.kind=static_cast<int>(s::Kind::Constant);key.type="u64";
          key.name=std::to_string(value);return intern(std::move(key));
        };
        std::optional<Class> canonical;
        if(form->constant || form->coefficients.empty())canonical=constant(form->constant);
        for(auto [level,coefficient]:form->coefficients) {
          Key index;index.kind=static_cast<int>(s::Kind::Index);index.type="u64";index.index_level=level;
          auto term=intern(std::move(index));
          if(coefficient!=1) {
            Key scale;scale.kind=static_cast<int>(s::Kind::Operation);scale.type="u64";
            scale.name="Multiply";scale.children={term,constant(coefficient)};term=intern(std::move(scale));
          }
          if(canonical) {
            Key sum;sum.kind=static_cast<int>(s::Kind::Operation);sum.type="u64";
            sum.name="Add";sum.children={*canonical,term};canonical=intern(std::move(sum));
          } else canonical=term;
        }
        mapping.emplace(id,*canonical);continue;
      }
      // Type validation is independent of whether typed arithmetic permits
      // dropping rounding/FTZ. Numeric projection must not repair invalid IR.
      const auto legacy_primitive=node.operands.size()==1?
          legacyIdealPrimitive(k,graph.type(node.operands[0])):std::nullopt;
      if (idealOperation(k,true) || legacy_primitive)
        for (const auto& operand : node.operands)
          if (graph.type(operand) != k.type)
            throw std::invalid_argument("arithmetic operand/result types differ");
      if(comparison(k) && graph.type(node.operands[0])!=graph.type(node.operands[1]))
        throw std::invalid_argument("comparison operand types differ");
      if(idealSelect(k) && (graph.type(node.operands[0])!="pred" ||
          graph.type(node.operands[1])!=k.type || graph.type(node.operands[2])!=k.type))
        throw std::invalid_argument("Select predicate/branch types differ");
      if(node.kind==s::Kind::Fold && realType(node.type)) {
        const auto& initial=node.operands[0];
        auto prefix=prefixes.find(initial.node);
        if(initial.conversion.empty() && initial.pointer_provenance.empty() &&
            prefix!=prefixes.end()) {
          const auto& previous=graph.node(initial);
          if(previous.type==node.type && previous.fold->reducer==node.fold->reducer &&
              previous.fold->binder_level==node.fold->binder_level &&
              node.fold->count<=std::numeric_limits<std::uint64_t>::max()-prefix->second.count &&
              shiftedProjectedBody(prefix->second.imported_body,k.children[1],
                                   node.fold->binder_level,prefix->second.count,node.fold->count)) {
            k.fold->count+=prefix->second.count;
            k.children={prefix->second.initial,prefix->second.imported_body};
            prefixes.emplace(id,FoldPrefix{k.fold->count,k.children[0],k.children[1]});
          }
        }
        if(!prefixes.contains(id))
          prefixes.emplace(id,FoldPrefix{node.fold->count,k.children[0],k.children[1]});
      }
      const auto imported=intern(std::move(k));
      mapping.emplace(id, imported);
    }
    std::vector<Class> result;
    result.reserve(roots.size());
    for (const auto& root : roots) { poll(); result.push_back(use(root)); }
    if(storage_roots) {
      storage_roots->clear();storage_roots->reserve(roots.size());
      for(const auto& root:roots) {poll();storage_roots->push_back(storageUse(root));}
    }
    if(witness_outputs) {
      witness_outputs->clear();witness_outputs->reserve(witnesses.size());
      for(const auto& ref:witnesses) {
        poll();if(ref.node>=live.size() || !live[ref.node])throw std::invalid_argument("scoped_witness_not_reachable");
        witness_outputs->push_back(use(ref));
      }
    }
    return result;
  }

  Result compare(std::span<const Class> a, std::span<const Class> b,bool compile_only=false,
                 std::span<const unsigned char> storage_a={},std::span<const unsigned char> storage_b={}) {
    if(compile_only && options_.certified_branch_scope)
      return result(Status::Unsupported,"conditional_scope_requires_output_comparison");
    if(options_.ideal_numeric_outputs && options_.certified_branch_scope)prepareBranchFacts();
    std::vector<Class> numeric_a,numeric_b;
    if(options_.ideal_numeric_outputs) {
      numeric_a.reserve(a.size());numeric_b.reserve(b.size());
      for(std::size_t i=0;i<a.size();++i) {
        poll();
        if(numericOutput(a[i]) && numericOutput(b[i])) {
          numeric_a.push_back(project(numericOutputProducer(a[i]),true));
          numeric_b.push_back(project(numericOutputProducer(b[i]),true));
        } else {
          numeric_a.push_back(project(a[i],false));
          numeric_b.push_back(project(b[i],false));
        }
      }
      a=numeric_a;b=numeric_b;
    } else {
      numeric_a.assign(a.begin(),a.end());numeric_b.assign(b.begin(),b.end());
    }
    for(auto& root:numeric_a)for(const auto* base:{"Exp2","Exp"})root=normalizeExponentialRatio(root,base);
    for(auto& root:numeric_b)for(const auto* base:{"Exp2","Exp"})root=normalizeExponentialRatio(root,base);
    compactImported(numeric_a,numeric_b);
    imported_node_count_=nodes_.size();
    // Normalize affine nested reductions before polynomial certificates add
    // cyclic alternatives such as zero=x-x. Reuse the same bounded rule;
    // there is no element expansion or second normalization implementation.
    {
      rebuild();const auto snapshot=nodes_;
      const auto admitted=nonrecursiveHeads(snapshot);
      std::vector<std::vector<Class>> members(nodes_.size());
      for(Class id=0;id<snapshot.size();++id)if(admitted[id])members[find(id)].push_back(id);
      std::unordered_map<Class,Class> replacements;
      for(Class id=0;id<snapshot.size();++id)if(admitted[id])
        flattenAffineFold(id,snapshot[id],snapshot,members,&replacements);
      // Publish the normalized acyclic import witness, not only an e-class
      // alias that the minimum-ID linear catalog cannot select. One traversal
      // of stored nodes/edges; neither Fold members nor output elements expand.
      if(!replacements.empty()) {
        const auto normalization_size=nodes_.size();std::unordered_map<Class,Class> mapped;
        std::unordered_set<Class> active;
        const auto normalize=[&](Class root) {
          std::vector<std::pair<Class,bool>> todo{{root,false}};
          while(!todo.empty()) {
            poll();auto [id,ready]=todo.back();todo.pop_back();
            if(mapped.contains(id))continue;
            const auto source=replacements.contains(id)?replacements.at(id):id;
            if(source>=normalization_size)throw std::logic_error("invalid import Fold replacement");
            if(!ready) {
              if(!active.insert(id).second)throw std::logic_error("cyclic import Fold normalization");
              todo.emplace_back(id,true);
              for(auto child:nodes_[source].children)if(!mapped.contains(child))todo.emplace_back(child,false);
            } else {
              auto key=nodes_[source];for(auto& child:key.children)child=mapped.at(child);
              mapped.emplace(id,intern(std::move(key)));active.erase(id);
            }
          }
          return mapped.at(root);
        };
        for(auto& root:numeric_a)root=normalize(root);
        for(auto& root:numeric_b)root=normalize(root);
        compactImported(numeric_a,numeric_b);
        imported_node_count_=nodes_.size();
      }
    }
    mergeAlphaEquivalentFolds();
    a=numeric_a;b=numeric_b;
    if(sentryward::debug::environment("SENTRYWARD_EGRAPH_PREFIX_TRACE"))diagnoseFinitePrefixSourceHeads(a,b);
    if(!compile_only)proveFirstFinitePrefixOutput(a,b);
    bool storage_equality_used=false;
    const auto equal = [&] {
      storage_equality_used=false;
      for (std::size_t i=0; i<a.size(); ++i) {
        poll();
        if (find(a[i]) != find(b[i])) {
          if(i>=storage_a.size() || i>=storage_b.size() || !storage_a[i] || !storage_b[i] ||
              !sameStoredOutput(a[i],b[i]))return false;
          storage_equality_used=true;
        }
      }
      return true;
    };
    for (std::size_t iteration = 0;; ++iteration) {
      rebuild();
      if(sentryward::debug::environment("SENTRYWARD_EGRAPH_TRACE")) {
#if defined(SENTRYWARD_DEVELOPMENT_DIAGNOSTICS) || defined(SENTRYWARD_ENABLE_COMPRESSION_AUDIT)
#include "sentryward/development/blocks/egraph_trace_0.inc"
#endif
}
      if (!compile_only && equal()) return result(Status::Equivalent,
          storage_equality_used?"same_stored_bytes":"same_eclass");
      if (iteration == options_.max_iterations) {
        diagnose(a,b);
        return result(Status::NotProven, "iteration_budget");
      }
      statistics_.iterations = iteration + 1;
      const auto before = unions_;
      shifted_body_matches_.clear();
      const auto snapshot = nodes_;
      auto nonrecursive=nonrecursiveHeads(snapshot);
      mergeAssociativeNormalForms(snapshot,nonrecursive);
      mergePolynomialNormalForms(snapshot,nonrecursive);
      // Unions above merge each class's facts immediately, but an ancestor's
      // conservative dependency union is updated through the incremental use
      // queue. Drain it before interpreting an empty dependency set as closed.
      propagateProperties();
      // Algebraic equality may introduce recursive class alternatives. Keep
      // certificates, but do not unfold those alternatives in later matching.
      nonrecursive=nonrecursiveHeads(snapshot);
      mergeLinearReductionNormalForms(snapshot,nonrecursive,a,b);
      nonrecursive=nonrecursiveHeads(snapshot);
      factorAdditiveForests(snapshot,nonrecursive);
      propagateProperties(); // Newly emitted residues retain their exact dependencies.
      nonrecursive=nonrecursiveHeads(snapshot);
      std::vector<std::vector<std::size_t>> members(nodes_.size());
      for (std::size_t i = 0; i < snapshot.size(); ++i)if(nonrecursive[i])members[find(i)].push_back(i);
      rewriteScopedBranchPredicates(snapshot,members);
      const auto selects=indexSelectPatterns(snapshot,nonrecursive);
      std::map<std::tuple<std::string,std::uint32_t,std::uint64_t>,std::vector<Class>> sum_targets;
      for(Class id=0;id<snapshot.size();++id) {
        const auto& k=snapshot[id];
        if(nonrecursive[id] && k.kind==static_cast<int>(s::Kind::Fold) && k.fold &&
            k.fold->reducer==s::Reducer::Sum && k.fold->count &&
            k.attributes.empty() && k.provenance.empty() &&
            k.children.size()==2 && zero_[find(k.children[0])] && finite_[find(id)])
          sum_targets[{k.type,k.fold->binder_level,k.fold->count}].push_back(id);
      }
      for (std::size_t id = 0; id < snapshot.size(); ++id) {
        poll();
        // This fixed-head identity neither unfolds an eclass alternative nor
        // traverses children. Proven cyclic aliases must not hide the stored
        // compare-select witness; expansive patterns remain SCC-guarded below.
        if(rewriteKnownPredicateSelect(id,snapshot[id],snapshot))continue;
        if(rewriteFiniteCompareSelectExtremum(id,snapshot[id],snapshot))continue;
        // These inspect fixed local heads and cached interval facts only.
        // A harmless x%N=x alias must not suppress their exact conversion proof.
        rewriteUnsignedShift(id,snapshot[id]);
        rewriteExactIntegerNumeric(id,snapshot[id],members);
        if(!nonrecursive[id])continue;
        Key k = snapshot[id];
        const auto original_children = k.children;
        for (auto& c : k.children) c = find(c);
        if(auto singleton=singletonConstant(k)) {
          rewrite(id,intern(std::move(*singleton)));continue;
        }
        flattenAffineFold(id,k,snapshot,members);
        if(arithmetic(k) && k.name=="Add" && !coalesced_forests_.contains(id)) {
          coalesceInterleavedFolds(id,k,snapshot,members,sum_targets);
          if(!coalesced_forests_.contains(id) && finite_[find(id)] && has_fold_[find(id)])
            coalesceRectangularFolds(id,k,snapshot,members,sum_targets);
        }
        if(k.kind==static_cast<int>(s::Kind::Fold) && k.fold->reducer==s::Reducer::Sum &&
            k.children.size()==2 && realType(k.type) && finite_[find(id)] &&
            has_fold_[find(k.children[1])] && !coalesced_forests_.contains(id))
          coalesceRectangularFolds(id,k,snapshot,members,sum_targets);
        rewriteSelectPatterns(id,k,original_children,snapshot,selects);
        rewriteExactU64Add(id,k,snapshot,selects);
        if(k.kind==static_cast<int>(s::Kind::Operation) && k.children.size()==2 &&
            k.provenance.empty() && k.attributes.empty() && integer_interval::parseType(k.type)) {
          const auto a=integer_ranges_[find(k.children[0])],b=integer_ranges_[find(k.children[1])];
          if(a && b && a->type==b->type && snapshot[k.children[0]].type==k.type &&
              snapshot[k.children[1]].type==k.type && a->lower>=0 &&
              b->lower==b->upper && b->lower>0 && a->lower/b->lower==a->upper/b->lower) {
            const auto quotient=a->lower/b->lower;
            if(k.name=="Remainder") {
              if(!quotient)rewrite(id,find(k.children[0]));
              else {
                Key offset;offset.kind=static_cast<int>(s::Kind::Constant);
                offset.type=k.type;offset.name=std::to_string(quotient*b->lower);
                auto subtract=k;subtract.name="Subtract";
                subtract.children={find(k.children[0]),intern(std::move(offset))};
                rewrite(id,intern(std::move(subtract)));
              }
            }
            if(k.name=="Divide") {
              Key zero;zero.kind=static_cast<int>(s::Kind::Constant);zero.type=k.type;zero.name=std::to_string(quotient);
              rewrite(id,intern(std::move(zero)));
            }
          }
        }
        if (k.kind == static_cast<int>(s::Kind::Fold) && k.fold->count == 0) {
          rewrite(id, k.children.at(0));
          continue;
        }
        if(k.kind==static_cast<int>(s::Kind::Fold) && k.fold->reducer==s::Reducer::Sum &&
            zero_[find(k.children[1])] && finite_[find(k.children[0])]) {
          rewrite(id,k.children[0]);continue;
        }
        if(k.kind==static_cast<int>(s::Kind::Fold) && k.fold->reducer==s::Reducer::Sum &&
            k.type=="real" && k.attributes.empty() && k.provenance.empty() && k.children.size()==2 &&
            !factored_folds_.contains(id) && k.children[0]<snapshot.size() && k.children[1]<snapshot.size() &&
            snapshot[k.children[0]].type==k.type &&
            snapshot[k.children[1]].type==k.type && snapshot[k.children[0]].provenance.empty() &&
            snapshot[k.children[1]].provenance.empty() && finite_[find(k.children[0])] &&
            finite_[find(k.children[1])] && !dependsOn(k.children[1],k.fold->binder_level)) {
          // ONE ideal-real invariant body, not count logical samples. The
          // conservative cached dependency union certifies absence of this
          // binder; other captured axes and the seed are retained unchanged.
          // Count0 and zero-body semantics were handled above. Reuse the
          // existing once-per-Fold factor marker and exact count literal.
          Key count;count.kind=static_cast<int>(s::Kind::Constant);count.type=k.type;
          count.name=std::to_string(k.fold->count);
          Key product;product.kind=static_cast<int>(s::Kind::Operation);product.type=k.type;
          product.name="Multiply";product.attributes="ideal_arithmetic";
          product.children={intern(std::move(count)),k.children[1]};
          auto total=intern(std::move(product));
          if(!zero_[find(k.children[0])]) {
            Key add;add.kind=static_cast<int>(s::Kind::Operation);add.type=k.type;
            add.name="Add";add.attributes="ideal_arithmetic";add.children={k.children[0],total};
            total=intern(std::move(add));
          }
          rewrite(id,total);factored_folds_.insert(id);continue;
        }
        if(idealOperation(k) && (k.name=="Maximum" || k.name=="Minimum")) {
          for(unsigned side=0;side<2;++side)
            if((k.name=="Maximum"?negative_infinity_[find(k.children[side])]:
                positive_infinity_[find(k.children[side])]) &&
                finite_[find(k.children[1-side])])rewrite(id,k.children[1-side]);
          for(auto a:members[original_children[0]])for(auto b:members[original_children[1]]) {
            const auto& left=snapshot[a];const auto& right=snapshot[b];
            if(left.kind!=static_cast<int>(s::Kind::Constant) ||
                right.kind!=static_cast<int>(s::Kind::Constant) || !left.provenance.empty() ||
                !right.provenance.empty() || !left.attributes.empty() || !right.attributes.empty())continue;
            double x=0,y=0;
            const auto xp=floatFromChars(left.name.data(),left.name.data()+left.name.size(),x);
            const auto yp=floatFromChars(right.name.data(),right.name.data()+right.name.size(),y);
            if(xp.ec!=std::errc{} || yp.ec!=std::errc{} || xp.ptr!=left.name.data()+left.name.size() ||
                yp.ptr!=right.name.data()+right.name.size() || !std::isfinite(x) || !std::isfinite(y))continue;
            // Parsing is monotone: strict inequality certifies the ordering.
            // Equal rounded values with distinct exact literals stay unknown.
            if(x<y)rewrite(id,k.name=="Maximum"?b:a);
            if(x>y)rewrite(id,k.name=="Maximum"?a:b);
          }
        }
        if(k.kind==static_cast<int>(s::Kind::Fold) && k.fold->reducer==s::Reducer::Sum &&
            realType(k.type) && finite_[find(id)] && !split_seed_folds_.contains(id)) {
          // Initial is outside the binder: expose it once, not once per term.
          bool zero_seed=false;
          for(auto seed:members[original_children[0]]) {
            const auto& n=snapshot[seed];
            zero_seed |= n.kind==static_cast<int>(s::Kind::Constant) && n.type==k.type && n.name=="0";
          }
          if(!zero_seed) {
            Key zero;zero.kind=static_cast<int>(s::Kind::Constant);zero.type=k.type;zero.name="0";
            auto sum=k;sum.children[0]=intern(std::move(zero));
            Key add;add.kind=static_cast<int>(s::Kind::Operation);add.type=k.type;
            add.name="Add";add.attributes="ideal_arithmetic";
            add.children={k.children[0],intern(std::move(sum))};
            rewrite(id,intern(std::move(add)));split_seed_folds_.insert(id);
          }
        }
        if(k.kind==static_cast<int>(s::Kind::Fold) && k.fold->reducer==s::Reducer::Sum &&
            realType(k.type) && finite_[find(id)] &&
            !factored_folds_.contains(id)) {
          const auto extract=[&](Class factor,Class body,bool division=false) {
            factor=find(factor);
            if(!finite_[factor] || dependsOn(factor,k.fold->binder_level) ||
                (division && !nonzero_[factor]))return;
            const auto emit=[&](Class initial) {
              auto sum=k;sum.children={find(initial),find(body)};
              const auto reduced=intern(std::move(sum));
              Key outer;outer.kind=static_cast<int>(s::Kind::Operation);outer.type=k.type;
              outer.name=division?"Divide":"Multiply";outer.attributes="ideal_arithmetic";
              outer.children=division?std::vector<Class>{reduced,factor}:std::vector<Class>{factor,reduced};
              rewrite(id,intern(std::move(outer)));factored_folds_.insert(id);
            };
            if(division) {
              // The initial value remains OUTSIDE division. Only a proven
              // zero seed may use this form; the existing seed-split rule
              // retains every nonzero seed separately before factoring.
              if(zero_[find(k.children[0])] && finite_[find(k.children[0])])emit(k.children[0]);
              return;
            }
            for(auto initial:members[original_children[0]]) {
              poll();const auto& init=snapshot[initial];
              if(init.kind==static_cast<int>(s::Kind::Constant) && init.type==k.type && init.name=="0")
                emit(initial);
              else if(arithmetic(init) && init.name=="Multiply" && init.type==k.type)
                for(unsigned j=0;j<2;++j)
                  if(find(init.children[j])==factor)emit(init.children[1-j]);
            }
          };
          // ONE already-indexed complete local Divide witness, not a search
          // through body aliases or a parallel indicator arithmetic walk.
          // This new identity is confined to the explicit ideal-real view.
          if(k.type=="real" && k.attributes.empty() && k.provenance.empty() && k.children.size()==2) {
            if(auto quotient=selects.get(original_children[1]).quotient) {
              const auto numerator=find(quotient->first),denominator=find(quotient->second);
              if(numerator<snapshot.size() && denominator<snapshot.size() &&
                  snapshot[numerator].type==k.type && snapshot[denominator].type==k.type &&
                  finite_[numerator] && finite_[denominator] && nonzero_[denominator])
                extract(denominator,numerator,true);
            }
          }
          if(!factored_folds_.contains(id))for(auto term:members[original_children[1]]) {
            poll();const auto& product=snapshot[term];
            if(!arithmetic(product) || product.name!="Multiply" || product.type!=k.type)continue;
            for(unsigned side=0;side<2;++side) {
              extract(product.children[side],product.children[1-side]);
              if(factored_folds_.contains(id))break;
              // The equally local associative pattern u*(c*v). Do not search
              // arbitrary ancestors to find c; each successful match removes
              // one invariant factor and leaves the remaining Fold compact.
              for(auto member:members[product.children[side]]) {
                const auto& nested=snapshot[member];
                if(!arithmetic(nested) || nested.name!="Multiply" || nested.type!=k.type)continue;
                for(unsigned inner=0;inner<2;++inner) {
                  const auto factor=find(nested.children[inner]);
                  if(!finite_[factor] || dependsOn(factor,k.fold->binder_level))continue;
                  auto body=product;body.children={find(product.children[1-side]),find(nested.children[1-inner])};
                  extract(factor,intern(std::move(body)));
                  if(factored_folds_.contains(id))break;
                }
                if(factored_folds_.contains(id))break;
              }
              if(factored_folds_.contains(id))break;
            }
            if(factored_folds_.contains(id))break;
          }
        }
        if(idealOperation(k)) {
          const auto make=[&](std::string name,std::vector<Class> children) {
            Key operation;operation.kind=static_cast<int>(s::Kind::Operation);
            operation.type=k.type;operation.name=std::move(name);
            operation.attributes="ideal_arithmetic";operation.children=std::move(children);
            return intern(std::move(operation));
          };
          const auto zero=[&] {
            Key constant;constant.kind=static_cast<int>(s::Kind::Constant);
            constant.type=k.type;constant.name="0";return intern(std::move(constant));
          };
          const auto one=[&] {
            Key constant;constant.kind=static_cast<int>(s::Kind::Constant);
            constant.type=k.type;constant.name="1";return intern(std::move(constant));
          };
          // Canonical two-sided clamp in the NUMERICAL projection only.
          // min(U,max(L,x)) == max(L,min(U,x)) for finite ideal-real x and
          // finite L<=U; the exact min_num/max_num orders differ on NaN.
          // Inspect only two immediate heads and cached finite facts. Emit
          // one canonical form, never a min/max distribution/member search.
          if(k.type=="real" && k.name=="Maximum" && k.children.size()==2 &&
              k.provenance.empty()) {
            const auto bound=[&](Class cls)->std::optional<double> {
              cls=find(cls);if(cls>=snapshot.size())return {};
              const auto& n=snapshot[cls];
              if(n.kind!=static_cast<int>(s::Kind::Constant) || n.type!="real" ||
                  !n.attributes.empty() || !n.provenance.empty() || n.name.size()>128 ||
                  !finite_[cls])return {};
              double result=0;
              const auto parsed=floatFromChars(n.name.data(),n.name.data()+n.name.size(),result);
              if(parsed.ec!=std::errc{} || parsed.ptr!=n.name.data()+n.name.size() ||
                  !std::isfinite(result))return {};
              return result;
            };
            bool clamp_emitted=false;
            for(unsigned outside=0;outside<2 && !clamp_emitted;++outside) {
              const auto lower=find(k.children[outside]);
              const auto lo=bound(lower);if(!lo)continue;
              const auto inner_id=find(k.children[1-outside]);
              if(inner_id>=snapshot.size())continue;
              const auto& inner=snapshot[inner_id];
              if(!idealOperation(inner) || inner.type!="real" || inner.name!="Minimum" ||
                  !inner.provenance.empty() || inner.children.size()!=2)continue;
              for(unsigned endpoint=0;endpoint<2;++endpoint) {
                const auto upper=find(inner.children[endpoint]),value=find(inner.children[1-endpoint]);
                const auto hi=bound(upper);
                if(!hi || value>=snapshot.size() || snapshot[value].type!="real" ||
                    !finite_[value] || !(lower==upper || *lo<*hi))continue;
                // Strict ordering of monotone rounded parses proves ordering
                // of exact real literals. Distinct equal parses stay Unknown.
                auto inside=make("Maximum",{lower,value});
                rewrite(id,make("Minimum",{upper,inside}));
                clamp_emitted=true;break; // One witness emits at most two enodes.
              }
            }
          }
          const auto literal=[&](Class id,const char* text) {
            const auto& n=snapshot[id];
            return n.kind==static_cast<int>(s::Kind::Constant) && n.type==k.type &&
              n.name==text && n.attributes.empty() && n.provenance.empty();
          };
          if(k.name=="Fma" && std::all_of(k.children.begin(),k.children.end(),[&](auto c){return finite_[find(c)];}))
            rewrite(id,make("Add",{make("Multiply",{k.children[0],k.children[1]}),k.children[2]}));
          if(k.name=="Subtract" && finite_[k.children[0]] && finite_[k.children[1]])
            rewrite(id,make("Add",{k.children[0],make("Negate",{k.children[1]})}));
          if(k.name=="Subtract" && negative_infinity_[find(k.children[0])] && finite_[find(k.children[1])])
            rewrite(id,k.children[0]);
          if((k.name=="Exp2" || k.name=="Exp") && negative_infinity_[find(k.children[0])])rewrite(id,zero());
          if(k.name=="Negate" && (negative_infinity_[find(k.children[0])] ||
              positive_infinity_[find(k.children[0])])) {
            // Exact sign change of a recognized extended-real sentinel. Do not
            // require finiteness or confer it on either infinity; unknown or
            // opaque values cannot enter this rule through decimal spelling.
            Key sentinel;sentinel.kind=static_cast<int>(s::Kind::Constant);
            sentinel.type=k.type;sentinel.name=positive_infinity_[find(k.children[0])]?"-inf":"inf";
            rewrite(id,intern(std::move(sentinel)));
          }
          if(k.name=="Divide" && finite_[find(k.children[0])])
            for(auto divisor:members[original_children[1]])if(literal(divisor,"1"))rewrite(id,k.children[0]);
          // One immediate exact unit witness, not Mul/Divide expansion or
          // denominator cancellation. Both sides retain the same finite,
          // nonzero real domain. The existing indexed unit set avoids scanning
          // numerator alternatives and emits at most one unary head per Divide.
          // Only the explicitly mathematical real view is admitted. Raw
          // unprojected f32/f64 producers may also have exact bit consumers;
          // their existing import path does not mint a separate Numeric view.
          if(k.type=="real" && k.name=="Divide" && k.provenance.empty() && k.children.size()==2 &&
              k.children[0]<snapshot.size() && k.children[1]<snapshot.size() &&
              snapshot[k.children[0]].type==k.type && snapshot[k.children[1]].type==k.type &&
              selects.ones.contains(k.children[0]) && finite_[k.children[1]] && nonzero_[k.children[1]])
            rewrite(id,make("Reciprocal",{k.children[1]}));
          if(k.name=="Divide" && find(k.children[0])==find(k.children[1]) &&
              finite_[find(k.children[0])] && nonzero_[find(k.children[0])])rewrite(id,one());
          // Primitive identities retain their real-domain guards. In particular,
          // sqrt(x*x) is not x for arbitrary x and rsqrt(0) is not finite.
          if(k.name=="Exp" && finite_[find(k.children[0])])
            for(auto operand:members[original_children[0]])
              if(literal(operand,"0"))rewrite(id,one());
          if(k.name=="Power" && positive_[find(k.children[0])] &&
              finite_[find(k.children[0])] && zero_[find(k.children[1])])
            rewrite(id,one());
          if(k.name=="Exp" || k.name=="Exp2" || k.name=="Log" || k.name=="Log2") {
            const auto inverse=k.name=="Exp"?"Log":k.name=="Exp2"?"Log2":
                k.name=="Log"?"Exp":"Exp2";
            const auto& heads=selects.get(original_children[0]).unary;
            if(auto found=heads.find(inverse);found!=heads.end()) {
              const auto argument=find(snapshot[found->second].children[0]);
              const bool exponential=k.name=="Exp" || k.name=="Exp2";
              if(finite_[argument] && (!exponential || positive_[argument]))
                rewrite(id,argument);
            }
            if(k.name=="Exp" || k.name=="Exp2") {
              // One more immediate head, not a search through the argument:
              // exp(-log(a))=1/a only inside the Log domain. A merely nonzero
              // a would incorrectly admit negative real logarithm arguments.
              if(auto negated=heads.find("Negate");negated!=heads.end()) {
                const auto negated_argument=find(snapshot[negated->second].children[0]);
                const auto& inner=selects.get(negated_argument).unary;
                if(auto logarithm=inner.find(inverse);logarithm!=inner.end()) {
                  const auto argument=find(snapshot[logarithm->second].children[0]);
                  if(finite_[argument] && positive_[argument])
                    rewrite(id,make("Reciprocal",{argument}));
                }
              }
            }
          }
          if(k.name=="Sqrt" && finite_[find(k.children[0])] &&
              nonnegative_[find(k.children[0])])
            for(auto operand:members[original_children[0]]) {
              if(literal(operand,"0"))rewrite(id,zero());
              if(literal(operand,"1"))rewrite(id,one());
            }
          if((k.name=="Rsqrt" || k.name=="ReciprocalSqrt") &&
              finite_[find(k.children[0])] && positive_[find(k.children[0])])
            rewrite(id,make("Reciprocal",{make("Sqrt",{k.children[0]})}));
          if(k.name=="Absolute" && finite_[find(k.children[0])]) {
            if(nonnegative_[find(k.children[0])])rewrite(id,k.children[0]);
            // One certified unary head from the shared epoch index; no
            // operand-ancestor search or logical Fold-element enumeration.
            const auto& unary=selects.get(original_children[0]).unary;
            if(auto negation=unary.find("Negate");negation!=unary.end()) {
              const auto value=find(snapshot[negation->second].children[0]);
              if(finite_[value])rewrite(id,make("Absolute",{value}));
            }
          }
          // A fixed-depth ideal-real vocabulary identity, not a global
          // coefficient rewrite or floating-point accuracy claim. Reuse the
          // epoch-local head index: inspect one product and its two unary
          // heads, never ancestors, alternative pairs, or Fold members.
          if(options_.ideal_numeric_outputs && k.type=="real" && k.name=="Exp2") {
            const auto& product=selects.get(original_children[0]).product;
            if(product)for(unsigned side=0;side<2;++side) {
              poll();const auto factor=find(side?product->second:product->first);
              const auto exponent=find(side?product->first:product->second);
              const auto& unary=selects.get(factor).unary;
              const auto logarithm=unary.find("Log2");
              if(logarithm==unary.end())continue;
              const auto base=find(snapshot[logarithm->second].children[0]);
              if(positive_[base] && finite_[base] && finite_[exponent])
                rewrite(id,make("Power",{base,exponent}));
            }
          }
          if(k.name=="Exp2" || k.name=="Exp")for(auto operand:members[original_children[0]]) {
            if(literal(operand,"0"))rewrite(id,one());
            const auto& difference=snapshot[operand];
            if(id<imported_node_count_ && !expanded_exponentials_.contains(id) && idealOperation(difference) &&
                difference.name=="Subtract" && finite_[find(difference.children[0])] &&
                finite_[find(difference.children[1])]) {
              auto left=make(k.name,{find(difference.children[0])});
              auto right=make(k.name,{make("Negate",{find(difference.children[1])})});
              rewrite(id,make("Multiply",{left,right}));
              expanded_exponentials_.insert(id);
            }
            if(id<imported_node_count_ && !expanded_exponentials_.contains(id) && arithmetic(difference) &&
                difference.name=="Add" && finite_[find(difference.children[0])] &&
                finite_[find(difference.children[1])]) {
              // Separate a finite shift independent of at least one coordinate
              // of the other operand. This is the same exp(a+b) identity,
              // including a negative literal times a maximum; no syntactic
              // Negate wrapper is required. Cached coordinate sets suffice.
              const auto& left_indices=free_indices_[find(difference.children[0])];
              const auto& right_indices=free_indices_[find(difference.children[1])];
              const bool independent_shift=
                  (left_indices.size()<right_indices.size() &&
                   std::includes(right_indices.begin(),right_indices.end(),left_indices.begin(),left_indices.end())) ||
                  (right_indices.size()<left_indices.size() &&
                   std::includes(left_indices.begin(),left_indices.end(),right_indices.begin(),right_indices.end()));
              if(k.type=="real" && independent_shift) {
                rewrite(id,make("Multiply",{make(k.name,{find(difference.children[0])}),
                                             make(k.name,{find(difference.children[1])})}));
                expanded_exponentials_.insert(id);continue;
              }
              for(unsigned side=0;side<2;++side)for(auto candidate:members[difference.children[side]]) {
                const auto& negative=snapshot[candidate];
                // Also retain a complete ideal-real subtractive offset as
                // ONE exponential factor: exp(x+(p-y))=exp(x)*exp(p-y).
                // This is a local finite identity, not distribution into a
                // Fold or a global rounded-coefficient rewrite. Exact/typed
                // views do not gain this new admission.
                const bool subtractive_offset=k.type=="real" && negative.type==k.type &&
                    negative.name=="Subtract" && negative.children.size()==2 &&
                    negative.provenance.empty() && finite_[find(negative.children[0])] &&
                    finite_[find(negative.children[1])];
                // Ratio recentering can leave a literal offset instead of a
                // subtractive one. Keep that exact real literal local too:
                // exp(x+p)=exp(x)*exp(p), without rewriting p itself.
                const bool literal_offset=k.type=="real" && negative.type==k.type &&
                    negative.kind==static_cast<int>(s::Kind::Constant) &&
                    negative.children.empty() && negative.attributes.empty() &&
                    negative.provenance.empty() && finite_[find(candidate)];
                if(!literal_offset && (!idealOperation(negative) ||
                    (negative.name!="Negate" && !subtractive_offset)))continue;
                rewrite(id,make("Multiply",{make(k.name,{find(difference.children[0])}),
                                             make(k.name,{find(difference.children[1])})}));
                expanded_exponentials_.insert(id);break;
              }
            }
          }
          if(k.name=="Negate" && finite_[k.children[0]]) {
            if(!negated_products_.contains(id)) {
              const auto& product=selects.get(original_children[0]).product;
              if(product && finite_[find(product->first)] && finite_[find(product->second)]) {
                rewrite(id,make("Multiply",{find(product->first),make("Negate",{find(product->second)})}));
                negated_products_.insert(id);
              }
            }
            for(auto operand:members[original_children[0]]) {
              const auto& n=snapshot[operand];
              if(literal(operand,"0"))rewrite(id,zero());
              if(!idealOperation(n))continue;
              if(n.name=="Negate")rewrite(id,find(n.children[0]));
              if(n.name=="Add" && find(n.children[0])!=find(operand) && find(n.children[1])!=find(operand))
                rewrite(id,make("Add",{make("Negate",{find(n.children[0])}),make("Negate",{find(n.children[1])})}));
            }
          }
          if(k.name=="Reciprocal" && finite_[k.children[0]] && nonzero_[k.children[0]]) {
            for(auto operand:members[original_children[0]]) {
              const auto& n=snapshot[operand];
              if(literal(operand,"1"))rewrite(id,one());
              if(idealOperation(n) && (n.name=="Exp2" || n.name=="Exp"))
                rewrite(id,make(n.name,{make("Negate",{find(n.children[0])})}));
              if(idealOperation(n) && n.name=="Multiply" &&
                  positive_[find(n.children[0])] && positive_[find(n.children[1])])
                rewrite(id,make("Multiply",{make("Reciprocal",{find(n.children[0])}),make("Reciprocal",{find(n.children[1])})}));
            }
          }
          // Divide is canonical. The opposite Mul/Reciprocal -> Divide rule
          // below supplies equality without endlessly cycling between forms.
          if(k.name=="Multiply")for(unsigned side=0;side<2;++side) {
            const auto& quotient=selects.get(original_children[side]).quotient;
            auto other=find(k.children[1-side]);
            if(quotient && finite_[other] && finite_[find(quotient->first)] &&
                finite_[find(quotient->second)])
              rewrite(id,make("Divide",{make("Multiply",{other,find(quotient->first)}),find(quotient->second)}));
          }
          if(k.name=="Multiply" && !combined_exponentials_.contains(id))for(const auto* base:{"Exp2","Exp"}) {
            // Combine exponentials through one local product layer without
            // pushing anything into Fold. Work is over e-class alternatives
            // of this fixed-size pattern, not ancestors or Fold elements.
            struct ExponentialFactor {Class exponent;std::optional<Class> rest;};
            const auto factors=[&](Class child)->std::optional<ExponentialFactor> {
              const auto root=find(child);const auto& heads=selects.get(root);
              const auto direct=heads.unary.find(base);
              if(direct!=heads.unary.end() && finite_[root])
                return ExponentialFactor{find(snapshot[direct->second].children[0]),{}};
              if(!heads.product)return {};
              auto [a,b]=*heads.product;
              if(a==root || b==root)return {};
              for(unsigned side=0;side<2;++side) {
                auto exp_class=side?b:a,rest=side?a:b;
                const auto& unary=selects.get(exp_class).unary;
                const auto exp=unary.find(base);
                if(exp!=unary.end() && finite_[exp_class] && finite_[rest])
                  return ExponentialFactor{find(snapshot[exp->second].children[0]),rest};
              }
              return {};
            };
            auto left=factors(original_children[0]),right=factors(original_children[1]);
            if(left && right) {
              const auto& l=*left;const auto& r=*right;
              auto combined=make(base,{make("Add",{l.exponent,r.exponent})});
              std::optional<Class> rest=l.rest;
              if(r.rest)rest=rest?make("Multiply",{*rest,*r.rest}):r.rest;
              rewrite(id,rest?make("Multiply",{combined,*rest}):combined);
              combined_exponentials_.insert(id);
            }
          }
          if(k.name=="Add" || k.name=="Multiply") {
            for(unsigned side=0;side<2;++side)for(auto operand:members[original_children[side]]) {
              poll();const auto& n=snapshot[operand];
              const auto other=k.children[1-side];
              if(!idealOperation(n))continue;
              if(k.name=="Add" && n.name=="Negate" && find(n.children[0])==other && finite_[other])
                rewrite(id,zero());
              if(k.name=="Multiply" && n.name=="Reciprocal" &&
                  finite_[find(n.children[0])] && finite_[other]) {
                // This definitional form preserves the SAME possible zero
                // denominator on both sides; it asserts no nonzero/finite
                // result. Cancelling a denominator still needs nonzero below.
                rewrite(id,make("Divide",{other,find(n.children[0])}));
                if(find(n.children[0])==other && nonzero_[other])rewrite(id,one());
              }
            }
          }
          if(k.name=="Add") {
            // Factoring, not the reverse distributive expansion. Inspect the
            // two immediate product/Fold e-classes; never enumerate a domain.
            for(auto left:members[original_children[0]])for(auto right:members[original_children[1]]) {
              poll();const auto& l=snapshot[left];const auto& r=snapshot[right];
              if(arithmetic(l) && arithmetic(r) && l.name=="Multiply" && r.name=="Multiply" &&
                  l.type==k.type && r.type==k.type && finite_[find(left)] && finite_[find(right)] &&
                  !factored_sums_.contains(id)) {
                for(unsigned i=0;i<2;++i)for(unsigned j=0;j<2;++j)
                  if(find(l.children[i])==find(r.children[j])) {
                    emitCommonAddFactor(id,k,find(l.children[i]),
                                        find(l.children[1-i]),find(r.children[1-j]));
                  }
              }
              if(joined_sums_.contains(id))continue;
              const auto sumFold=[&](const Key& f,Class root) {
                return f.kind==static_cast<int>(s::Kind::Fold) && f.type==k.type &&
                    f.attributes.empty() && f.provenance.empty() && f.children.size()==2 &&
                    f.fold->reducer==s::Reducer::Sum && f.fold->count && finite_[find(root)] &&
                    zero_[find(f.children[0])];
              };
              if(!sumFold(l,left) || !sumFold(r,right) ||
                  l.fold->binder_level!=r.fold->binder_level || l.fold->count>UINT64_MAX-r.fold->count)continue;
              for(unsigned order=0;order<2;++order) {
                const auto& a=order?r:l;const auto& b=order?l:r;
                if(!shiftedProjectedBody(a.children[1],b.children[1],a.fold->binder_level,
                                         a.fold->count,b.fold->count))continue;
                auto joined=a;joined.fold->count+=b.fold->count;
                rewrite(id,intern(std::move(joined)));joined_sums_.insert(id);break;
              }
            }
          }
          if(k.name=="Divide" && finite_[find(k.children[0])] && finite_[find(k.children[1])] &&
              nonzero_[find(k.children[1])]) {
            for(auto numerator:members[original_children[0]]) {
              const auto& n=snapshot[numerator];
              if(!arithmetic(n) || n.name!="Multiply")continue;
              for(auto denominator:members[original_children[1]]) {
                poll();const auto& d=snapshot[denominator];
                if(!arithmetic(d) || d.name!="Multiply")continue;
                for(unsigned i=0;i<2;++i)for(unsigned j=0;j<2;++j) {
                  auto factor=find(n.children[i]), divisor=find(d.children[1-j]);
                  if(factor==find(d.children[j]) && nonzero_[factor] &&
                      finite_[factor] && nonzero_[divisor] && finite_[divisor])
                    rewrite(id,make("Divide",{find(n.children[1-i]),divisor}));
                }
              }
            }
          }
          if(k.name=="Subtract" && k.children[0]==k.children[1] && finite_[find(k.children[0])])
            rewrite(id,zero());
          if(k.name=="Add" || k.name=="Multiply") {
            for(unsigned side=0;side<2;++side) {
              for(auto left:members[original_children[side]]) {
                poll();
                auto l=snapshot[left];
                for(auto& child:l.children)child=find(child);
                auto other=find(k.children[1-side]);
                if(k.name=="Add" && arithmetic(l) && l.name=="Add" &&
                    finite_[find(left)] && finite_[other]) {
                  for(unsigned inner=0;inner<2;++inner) {
                    for(auto candidate:members[snapshot[left].children[inner]]) {
                      const auto& neg=snapshot[candidate];
                      if(idealOperation(neg) && neg.name=="Negate" && find(neg.children[0])==other)
                        rewrite(id,l.children[1-inner]);
                    }
                    for(auto candidate:members[original_children[1-side]]) {
                      const auto& neg=snapshot[candidate];
                      if(idealOperation(neg) && neg.name=="Negate" && find(neg.children[0])==l.children[inner])
                        rewrite(id,l.children[1-inner]);
                    }
                  }
                }
                if(k.name=="Add" && idealOperation(l) && l.name=="Subtract" &&
                    finite_[find(left)] && finite_[other]) {
                  if(l.children[1]==other)rewrite(id,l.children[0]);
                  for(auto right:members[original_children[1-side]]) {
                    poll();auto r=snapshot[right];
                    for(auto& child:r.children)child=find(child);
                    if(idealOperation(r) && r.name=="Subtract" &&
                        l.children[1]==r.children[0] && finite_[find(right)])
                      rewrite(id,make("Subtract",{l.children[0],r.children[1]}));
                  }
                }
              }
            }
          }
        }
        if (!arithmetic(k) || !finite_[find(k.children[0])] || !finite_[find(k.children[1])]) continue;
        auto x = k.children[0], y = k.children[1];
        Key swapped = k; std::swap(swapped.children[0], swapped.children[1]);
        rewrite(id, intern(std::move(swapped)));
        for (unsigned side = 0; side < 2; ++side) {
          auto other = side ? x : y;
          // Snapshot membership remains valid even when a rewrite joins classes.
          for (auto member : members[original_children[side]]) {
            poll();
            auto inner = snapshot[member];
            for (auto& c : inner.children) c = find(c);
            if (inner.kind == static_cast<int>(s::Kind::Constant) && inner.type == k.type) {
              if ((k.name == "Add" && inner.name == "0") ||
                  (k.name == "Multiply" && inner.name == "1")) rewrite(id, other);
              if(k.name=="Multiply" && inner.name=="0" && finite_[find(other)])rewrite(id,find(member));
            }
            if (!arithmetic(inner) || inner.type != k.type) continue;
            // AC equality is certified by frozen multiset signatures below,
            // not by generating alternative binary bracketings.
            // Never emit the reverse distributive expansion: the common-
            // factor and invariant-Fold rules already canonicalize compactly.
          }
        }
        if ((k.name == "Maximum" || k.name == "Minimum") && find(x) == find(y))
          rewrite(id, x);
      }
      rebuild();
      if (!compile_only && equal()) return result(Status::Equivalent,
          storage_equality_used?"same_stored_bytes":"same_eclass");
      if (before == unions_) {
        diagnose(a,b);
        return result(Status::NotProven, "saturated");
      }
    }
  }

  Result result(Status status, std::string reason) const {
    auto stats = statistics_;
    stats.nodes = nodes_.size();
    if(options_.certified_branch_scope && sentryward::debug::environment("SENTRYWARD_EGRAPH_TRACE"))
      SENTRYWARD_DEBUG_PRINT(egraph_3);
    return {status, std::move(reason), stats,branch_scope_used_ && options_.certified_branch_scope?
        options_.certified_branch_scope->digest():std::string{}};
  }

  ScopedBodyResult scopedBodyProof(Class reference_wrapper,Class actual_wrapper,
      Class reference,Class actual,const s::Graph& reference_graph,std::span<const s::Id> reference_folds,
      const s::Graph& actual_graph,std::span<const s::Id> actual_folds,
      std::span<const ScopedAxisBinding> bindings,std::span<const Class> coordinates,
      std::span<const std::optional<Class>> image_coordinates={}) {
    ScopedBodyResult result;
    if(!options_.ideal_numeric_outputs || bindings.size()>2 || coordinates.size()!=bindings.size() ||
        (!image_coordinates.empty() && image_coordinates.size()!=bindings.size())) {
      result.reason="unsupported_scoped_context";return result;
    }
    if(!numericOutput(reference_wrapper) || !numericOutput(actual_wrapper)) {
      result.reason="unsupported_numeric_root";return result;
    }
    reference_wrapper=project(reference_wrapper,true);actual_wrapper=project(actual_wrapper,true);
    if(!realType(nodes_[find(reference_wrapper)].type) || !realType(nodes_[find(actual_wrapper)].type)) {
      result.reason="unsupported_root_policy";return result;
    }
    reference=project(reference,true);actual=project(actual,true);propagateProperties();
    if(!realType(nodes_[find(reference)].type) || nodes_[find(reference)].type!=nodes_[find(actual)].type ||
        !finite_[find(reference)] || !finite_[find(actual)]) {
      result.reason="nonfinite_or_exact_body";return result;
    }
    // One epoch catalog over stored heads, not per query root reachability sets.
    std::vector<std::vector<Class>> members(nodes_.size());
    for(Class id=0;id<nodes_.size();++id){poll();members[find(id)].push_back(id);}
    std::size_t remaining=131072;AffineCache affine;
    BodyHeadIndex heads(*this,nodes_,members,affine,&remaining);BodyScopeRegistry registry(heads);
    auto root=registry.output();
    if(!root){result.reason="outer_domain_unknown";return result;}
    auto reference_scope=*root,actual_scope=*root;
    for(auto head:reference_folds) {
      auto next=registry.insideSourceFold(reference_scope,reference_graph,0,head);
      if(!next){result.reason="reference_scope_unknown";return result;}reference_scope=*next;
    }
    for(auto head:actual_folds) {
      auto next=registry.insideSourceFold(actual_scope,actual_graph,1,head);
      if(!next){result.reason="actual_scope_unknown";return result;}actual_scope=*next;
    }
    BodyScopeRegistry::Context context{reference_scope,actual_scope};
    context.axis_count=static_cast<unsigned>(bindings.size());
    for(unsigned i=0;i<context.axis_count;++i) {
      const auto& b=bindings[i];
      const auto extent=b.actual_level?registry.count(b.actual_shift?reference_scope:actual_scope,
          b.actual_shift?b.reference_level:*b.actual_level):std::optional<std::uint64_t>{1};
      if(!extent){result.reason="binding_domain_unknown";return result;}
      context.axes[i]={b.reference_level,b.actual_level,b.offset,*extent,coordinates[i],b.actual_shift};
      if(!image_coordinates.empty() && image_coordinates[i]) {
        if(b.actual_level || b.actual_shift){result.reason="invalid_derived_point";return result;}
        auto image=registry.image(reference_scope,b.reference_level,*image_coordinates[i]);
        auto point=image?registry.point(*image,coordinates[i],actual_scope):std::nullopt;
        if(!point){result.reason="derived_point_unknown";return result;}
        context.axes[i].point=*point;
      }
    }
    const auto matched=affineBodyMapWithHeads(reference,actual,UINT32_MAX,nodes_,false,&remaining,heads,
        UINT32_MAX,&registry,&context);
    result.proved=bool(matched);result.reason=result.proved?"scoped_body_certificate":
        (!remaining || heads.exhausted?"scoped_body_local_cap":"scoped_body_unknown");
    result.body_work=131072-remaining;result.scope_identities=registry.scopes.size();
    result.coordinate_checks=registry.coordinate_checks;result.affine_checks=registry.affine_checks;return result;
  }

  JoinedBodyResult joinedBodyProof(Class reference_wrapper,Class reference,
      const s::Graph& reference_graph,std::span<const s::Id> reference_folds,
      std::span<const std::uint32_t> reference_axes,
      const s::Graph& actual_graph,std::span<const Class> actual_wrappers,
      std::span<const Class> bodies,std::span<const std::vector<s::Id>> folds,
      std::span<const std::array<Class,2>> coordinates,std::span<const JoinedBodyPiece> requests) {
    JoinedBodyResult result;
    if(!options_.ideal_numeric_outputs || reference_axes.empty() || reference_axes.size()>2 ||
        requests.empty() || requests.size()>1024 || bodies.size()!=requests.size() ||
        actual_wrappers.size()!=requests.size() || folds.size()!=requests.size() ||
        coordinates.size()!=requests.size()) {result.reason="invalid_join_configuration";return result;}
    if(!numericOutput(reference_wrapper)) {result.reason="unsupported_numeric_root";return result;}
    reference_wrapper=project(reference_wrapper,true);reference=project(reference,true);
    std::vector<Class> numeric_bodies;numeric_bodies.reserve(bodies.size());
    for(std::size_t i=0;i<bodies.size();++i) {
      poll();if(!numericOutput(actual_wrappers[i])) {result.reason="unsupported_numeric_root";return result;}
      auto wrapper=project(actual_wrappers[i],true);
      if(!realType(nodes_[find(wrapper)].type)){result.reason="unsupported_root_policy";return result;}
      numeric_bodies.push_back(project(bodies[i],true));
    }
    // Coordinate witnesses and numeric cell addresses must use the SAME exact
    // projection before the immutable certificate epoch is constructed.
    std::vector<std::array<Class,2>> exact_coordinates(coordinates.begin(),coordinates.end());
    for(std::size_t i=0;i<requests.size();++i)
      for(unsigned j=0;j<requests[i].axis_count && j<2;++j)
        exact_coordinates[i][j]=project(exact_coordinates[i][j],false);
    propagateProperties();
    if(!realType(nodes_[find(reference_wrapper)].type) || !realType(nodes_[find(reference)].type) ||
        !finite_[find(reference)]) {result.reason="nonfinite_or_exact_body";return result;}
    std::vector<std::vector<Class>> members(nodes_.size());
    for(Class id=0;id<nodes_.size();++id){poll();members[find(id)].push_back(id);}
    std::size_t remaining=131072;AffineCache affine;
    BodyHeadIndex heads(*this,nodes_,members,affine,&remaining);BodyScopeRegistry registry(heads);
    auto root=registry.output();if(!root){result.reason="outer_domain_unknown";return result;}
    auto reference_scope=*root;
    for(auto head:reference_folds) {
      auto next=registry.insideSourceFold(reference_scope,reference_graph,0,head);
      if(!next){result.reason="reference_scope_unknown";return result;}reference_scope=*next;
    }
    std::array<std::pair<std::uint64_t,std::uint64_t>,2> domain{};
    for(std::size_t i=0;i<reference_axes.size();++i) {
      auto extent=registry.count(reference_scope,reference_axes[i]);
      if(!extent || (i && reference_axes[i]==reference_axes[i-1])) {
        result.reason="join_domain_unknown";return result;
      }
      domain[i]={0,*extent};
    }
    JoinedFamilies catalog(registry);
    std::vector<JoinedFamilies::Piece> pieces;pieces.reserve(requests.size());
    for(std::size_t i=0;i<requests.size();++i) {
      if(!heads.charge()){result.reason="join_local_cap";return result;}
      const auto& request=requests[i];auto scope=*root;
      if(request.axis_count!=reference_axes.size()){result.reason="join_axis_mismatch";return result;}
      for(auto head:folds[i]) {
        auto next=registry.insideSourceFold(scope,actual_graph,1,head);
        if(!next){result.reason="actual_scope_unknown";return result;}scope=*next;
      }
      JoinedFamilies::Piece piece;piece.body=numeric_bodies[i];piece.scope=scope;
      for(unsigned j=0;j<request.axis_count;++j) {
        auto image=registry.image(scope,request.axes[j].level,exact_coordinates[i][j]);
        if(!image && request.axis_count==1) {
          // Rank-one typed-cell fallback: the caller still supplies an actual
          // reachable byte-offset witness. Geometry cannot bypass the later
          // full joined-body certificate or introduce a graph coordinate.
          auto head=heads.compact(piece.body);
          if(head && heads.snapshot[*head].kind==static_cast<int>(s::Kind::InitialCell) &&
              heads.snapshot[*head].children.size()==1 &&
              heads.owner(heads.snapshot[*head].children[0])==heads.owner(exact_coordinates[i][j]))
            image=registry.imageFromTypedCells({scope,reference_scope},piece.body,reference,
                request.axes[j].level,reference_axes[j]);
        }
        if(!image){result.reason="join_image_unknown";return result;}piece.images[j]=*image;
      }
      pieces.push_back(std::move(piece));
    }
    auto family=catalog.join(pieces,static_cast<unsigned>(reference_axes.size()),domain);
    if(family && catalog.uniform(*family,reference,reference_scope,reference_axes)) {
      result.proved=true;result.reason="joined_body_certificate";result.pieces=pieces.size();
    } else result.reason=!remaining || heads.exhausted || catalog.exhausted()?"join_local_cap":"join_unknown";
    result.body_work=131072-remaining;result.coverage_work=catalog.coverageWork();
    result.scope_identities=registry.scopes.size();return result;
  }

  ScopedProductResult productBodyProof(const s::Graph& reference_graph,
      std::span<const Class> reference_wrappers,std::span<const Class> reference_bodies,
      std::span<const std::vector<s::Id>> reference_folds,const s::Graph& actual_graph,
      std::span<const Class> actual_wrappers,std::span<const Class> actual_bodies,
      std::span<const std::vector<s::Id>> actual_folds,std::span<const ScopedAxisBinding> bindings,
      std::span<const Class> coordinates,std::span<const std::optional<Class>> image_coordinates) {
    ScopedProductResult result;
    if(!options_.ideal_numeric_outputs || reference_bodies.empty() || actual_bodies.empty() ||
        reference_bodies.size()>8 || actual_bodies.size()>8 || bindings.size()>2 ||
        reference_wrappers.size()!=reference_bodies.size() || reference_folds.size()!=reference_bodies.size() ||
        actual_wrappers.size()!=actual_bodies.size() || actual_folds.size()!=actual_bodies.size() ||
        coordinates.size()!=bindings.size() || image_coordinates.size()!=bindings.size()) {
      result.reason="invalid_product_configuration";return result;
    }
    const auto projectBodies=[&](std::span<const Class> wrappers,std::span<const Class> bodies) {
      std::vector<Class> result;result.reserve(bodies.size());
      for(std::size_t i=0;i<bodies.size();++i) {
        poll();if(!numericOutput(wrappers[i]))return std::optional<std::vector<Class>>{};
        auto wrapper=project(wrappers[i],true);
        if(!realType(nodes_[find(wrapper)].type))return std::optional<std::vector<Class>>{};
        result.push_back(project(bodies[i],true));
      }
      return std::optional{std::move(result)};
    };
    auto rb=projectBodies(reference_wrappers,reference_bodies),ab=projectBodies(actual_wrappers,actual_bodies);
    if(!rb || !ab){result.reason="unsupported_numeric_root";return result;}propagateProperties();
    std::vector<std::vector<Class>> members(nodes_.size());
    for(Class id=0;id<nodes_.size();++id){poll();members[find(id)].push_back(id);}
    std::size_t remaining=131072;AffineCache affine;
    BodyHeadIndex heads(*this,nodes_,members,affine,&remaining);BodyScopeRegistry registry(heads);
    auto root=registry.output();if(!root){result.reason="outer_domain_unknown";return result;}
    const auto uses=[&](const s::Graph& graph,unsigned tag,std::span<const Class> bodies,
        std::span<const std::vector<s::Id>> folds) {
      std::vector<BoundedProducts::Use> result;result.reserve(bodies.size());
      for(std::size_t i=0;i<bodies.size();++i) {
        auto scope=*root;
        for(auto head:folds[i]) {
          auto next=registry.insideSourceFold(scope,graph,tag,head);
          if(!next)return std::optional<std::vector<BoundedProducts::Use>>{};scope=*next;
        }
        result.push_back({bodies[i],scope});
      }
      return std::optional{std::move(result)};
    };
    auto r=uses(reference_graph,0,*rb,reference_folds),a=uses(actual_graph,1,*ab,actual_folds);
    if(!r || !a){result.reason="product_scope_unknown";return result;}
    BoundedProducts factory(registry);auto rs=factory.common(*r),as=factory.common(*a);
    if(!rs || !as){result.reason="incompatible_product_scopes";return result;}
    BodyScopeRegistry::Context context{*rs,*as};context.axis_count=static_cast<unsigned>(bindings.size());
    for(unsigned i=0;i<context.axis_count;++i) {
      const auto& b=bindings[i];
      const auto extent=b.actual_level?registry.count(b.actual_shift?*rs:*as,
          b.actual_shift?b.reference_level:*b.actual_level):std::optional<std::uint64_t>{1};
      if(!extent){result.reason="binding_domain_unknown";return result;}
      context.axes[i]={b.reference_level,b.actual_level,b.offset,*extent,coordinates[i],b.actual_shift};
      if(image_coordinates[i]) {
        if(b.actual_level || b.actual_shift){result.reason="invalid_derived_point";return result;}
        auto image=registry.image(*rs,b.reference_level,*image_coordinates[i]);
        auto point=image?registry.point(*image,coordinates[i],*as):std::nullopt;
        if(!point){result.reason="derived_point_unknown";return result;}context.axes[i].point=*point;
      }
    }
    // Compile this ORIGINAL assignment once. Product factors retain their
    // own source paths/scopes and use only owned prefix-derived IDs below.
    auto admitted=registry.admitContext(context);
    auto proof=admitted?factory.equal(*r,*a,*admitted):std::nullopt;
    result.proved=bool(proof);result.reason=result.proved?"scoped_product_certificate":
        (!remaining || heads.exhausted?"product_local_cap":"product_unknown");
    if(proof){result.reference_factors=proof->first;result.actual_factors=proof->second;}
    result.body_work=131072-remaining;result.scope_identities=registry.scopes.size();
    result.context_entries=registry.contexts.size();result.context_id_reads=registry.context_reads;
    result.context_prefix_hits=registry.context_prefix_hits;return result;
  }

 private:
  struct StorageCopy {
    // Store one immutable enode ID, not a copied allocation string/record.
    std::optional<Class> cell;
    Class offset=0;
    bool ambiguous=false;
  };
  void indexStoredOutputs() {
    if(storage_copy_nodes_==nodes_.size() && storage_copy_unions_==unions_)return;
    storage_copies_.assign(nodes_.size(),{});
    // One bounded scan per congruence epoch, independent of output count.
    // Only immediate, exact InitialCell heads are inspected: no ancestor walk,
    // numeric conversion removal, or internal cross-type union is performed.
    for(Class id=0;id<nodes_.size();++id) {
      poll();const auto& k=nodes_[id];
      if(k.kind!=static_cast<int>(s::Kind::InitialCell) || k.name.empty() ||
          !k.attributes.empty() || !k.provenance.empty() || k.fold || k.children.size()!=1)continue;
      const auto type=k.type.starts_with("exact:")?k.type.substr(6):k.type;
      if(bitWidth(type)!=8*options_.storage_output_bytes)continue;
      const auto offset=find(k.children[0]);const auto& address=nodes_[offset];
      if(address.type!="u64" && address.type!="exact:u64")continue;
      auto& entry=storage_copies_[find(id)];
      if(!entry.cell) {entry.cell=id;entry.offset=offset;}
      else if(nodes_[*entry.cell].name!=k.name || entry.offset!=offset)entry.ambiguous=true;
    }
    storage_copy_nodes_=nodes_.size();storage_copy_unions_=unions_;
  }
  bool sameStoredOutput(Class a,Class b) {
    // Known scalar storage types are at most eight bytes; reject overflow,
    // unknown/unbounded formats, and the default-disabled contract up front.
    if(!options_.storage_output_bytes || options_.storage_output_bytes>8)return false;
    indexStoredOutputs();
    const auto& x=storage_copies_[find(a)];const auto& y=storage_copies_[find(b)];
    return x.cell && y.cell && !x.ambiguous && !y.ambiguous &&
        x.offset==y.offset && nodes_[*x.cell].name==nodes_[*y.cell].name;
  }
  std::vector<StorageCopy> storage_copies_;
  std::size_t storage_copy_nodes_=std::numeric_limits<std::size_t>::max();
  std::size_t storage_copy_unions_=std::numeric_limits<std::size_t>::max();

  struct PhysicalWorkPlan {
    std::size_t initial=epoch_work_allowance::default_work,remaining=initial,sizing=0;
    epoch_work_allowance::Size physical;
    bool exhausted=false;
  };
  // RESOURCE policy only. Shared by candidate-gated optional certificate
  // epochs, never a source/domain/finite admission. Count stored headers and
  // edges plus bounded scalar metadata lengths once; no string/record copies,
  // no logical extent and no per-root replenishment. Preserve all prior work.
  PhysicalWorkPlan physicalWorkPlan(const std::vector<Key>& snapshot,
      std::size_t floor,std::size_t already_spent,bool account_global_work=false) {
    namespace w=epoch_work_allowance;
    PhysicalWorkPlan out;w::PhysicalEpoch size;
    if(already_spent>w::maximum_work) {
      out.initial=w::maximum_work;out.remaining=0;out.exhausted=true;return out;
    }
    for(const auto& k:snapshot) {
      poll();constexpr std::size_t cost=6; // Header/edges/four scalar lengths.
      if(cost>w::maximum_work-already_spent-out.sizing) {
        out.initial=w::maximum_work;out.remaining=out.initial-already_spent-out.sizing;
        out.physical=size.allowance(floor).physical;out.exhausted=true;return out;
      }
      if(account_global_work && options_.max_work) {
        if(work_>options_.max_work || cost>options_.max_work-work_)throw Limit{"work_budget"};
        work_+=cost;
      }
      out.sizing+=cost; // Precharge before actual scalar header inspection.
      const std::array<std::size_t,4> lengths{
          k.type.size(),k.name.size(),k.attributes.size(),k.provenance.size()};
      size.addHeader(k.children.size(),lengths);
    }
    const auto allowance=size.allowance(floor);out.initial=allowance.max_work;out.physical=allowance.physical;
    if(already_spent>out.initial || out.sizing>out.initial-already_spent) {
      out.remaining=0;out.exhausted=true;return out;
    }
    out.remaining=out.initial-already_spent-out.sizing;return out;
  }
  void mergeLinearReductionNormalForms(const std::vector<Key>& snapshot,
      const std::vector<bool>&,std::span<const Class> left,
      std::span<const Class> right) {
    namespace l=bounded_linear_reduction;
    // One frozen RANKED selected-head DAG and one summary cache per epoch. Candidate
    // roots are indexed by type/context/body skeleton, then exact weighted
    // interval keys: no root-pair product, no generated enodes or Fold elements.
    constexpr std::size_t max_candidates=1024,max_active_nodes=4096,max_cached_terms=32768;
    constexpr std::size_t max_root_candidates=max_candidates/2;
    constexpr std::size_t max_signatures=1024,max_interval_entries=32768,max_groups=256,max_mappings=8192;
    // ONE physical stored-header capacity for retained planner metadata.
    // Active walks, candidate fanout and complete mapper geometry keep their
    // separate fixed limits: more memo slots are never an equality fact.
    epoch_work_allowance::Size summary_basis;summary_basis.headers=snapshot.size();
    const auto max_retained_summary_nodes=epoch_work_allowance::retainedHeaderCapacity(summary_basis);
    struct LocalCap {};
    std::size_t initial_work=2000000,remaining=initial_work,policy_work=0,charged_helper=0;
    epoch_work_allowance::Size physical_epoch;
    const char* phase="linear_setup";const char* cap_ledger="optional_work";
    std::size_t cap_entries=0;bool shape_capped=false;
    const auto localCap=[&](const char* ledger,std::size_t entries=0) {
      cap_ledger=ledger;cap_entries=entries;throw LocalCap{};
    };
    scoped_work_budget::Guards* scoped_sum_query=nullptr;bool synchronizing_helper=false;
    const auto charge=[&](std::size_t amount=1) {
      poll();
      // Resolver/catalog work uses this same pre-work path. The sync delta
      // has already passed the primitive hook and may not debit query twice.
      if(scoped_sum_query && !synchronizing_helper) {
        if(!scoped_sum_query->charge(amount))localCap("scoped_sum_query",0);
      }
      if(amount>remaining)localCap("optional_work",initial_work-remaining);remaining-=amount;
      if(options_.max_work) {
        if(amount>options_.max_work-work_)throw Limit{"work_budget"};work_+=amount;
      }
    };
    const auto traceCap=[&](std::size_t helper_work=0) {
      if(sentryward::debug::environment("SENTRYWARD_EGRAPH_TRACE"))
        SENTRYWARD_DEBUG_PRINT(egraph_4);
    };
    std::vector<std::pair<Class,Class>> equalities;
    // Reducer contexts share the original optional ledgers. Extrema use
    // interval-first selected-DAG summaries, never a max/min -> real sum rule.
    std::size_t candidate_entries=0,summary_entries=0,cached_terms=0;
    std::size_t interval_entries=0,residual_entries=0,mapping_entries=0;
    std::size_t signature_entries=0,group_entries=0,shape_entries=0,lexical_entries=0,coverage_entries=0,constant_entries=0;
    std::array<bool,3> modes{};
    try {
    for(auto id:fold_head_ids_) {
      if(id>=snapshot.size())break;
      const auto& k=snapshot[id];
      charge();if(k.kind!=static_cast<int>(s::Kind::Fold) || !k.fold || !k.fold->count)continue;
      if(k.fold->reducer==s::Reducer::Max)modes[0]=true;
      if(k.fold->reducer==s::Reducer::Min)modes[1]=true;
      if(k.fold->reducer==s::Reducer::Sum)modes[2]=true;
    }
    } catch(const LocalCap&) {traceCap();return;}
    if(std::none_of(modes.begin(),modes.end(),[](bool value){return value;}))return;
    const auto plan=physicalWorkPlan(snapshot,initial_work,initial_work-remaining,true);
    initial_work=plan.initial;remaining=plan.remaining;policy_work=plan.sizing;physical_epoch=plan.physical;
    if(plan.exhausted){traceCap();return;}
    // PLAN FIRST: neither helper has done work before the mode gate/sizing.
    // One lifetime allowance borrows the SAME physically derived remainder;
    // reducer/query/root transitions never reconstruct or replenish it.
    l::Limits limits;limits.max_nodes=max_retained_summary_nodes;limits.max_cached_terms=max_cached_terms;
    limits.max_work=remaining;
    if(options_.max_work)limits.max_work=std::min(limits.max_work,options_.max_work-work_);
    limits.coefficients.max_work=limits.max_work;
    // Linear and exact-coefficient hooks precharge the SAME coverage aggregate
    // and currently active body guard. The existing sync() still owns the ONE
    // epoch/global debit. Legacy planner queries keep a null relay.
    l::Builder builder(limits,[&](std::size_t amount) {
      if(!scoped_sum_query)return true;
      return scoped_sum_query->charge(amount);
    });
    const auto sync=[&] {
      const auto delta=builder.work()-charged_helper;
      struct SyncGuard {bool& active;bool previous;SyncGuard(bool& a):active(a),previous(a){active=true;}
        ~SyncGuard(){active=previous;}} guard(synchronizing_helper);
      charge(delta);
      charged_helper=builder.work();
      if(builder.exhausted())localCap(scoped_sum_query && scoped_sum_query->exhausted()?
          "scoped_sum_query":"linear_builder",builder.work());
    };
    const auto settleHelperFailure=[&] {
      // Primitive work already passed the active-query precharge, but a
      // resolver/cache LocalCap may bypass sync(). Invoice its pending delta
      // once even when the optional epoch has no room left. No proof activity,
      // fresh allowance, query debit, or throwing unwinding destructor occurs.
      const auto delta=builder.work()-charged_helper;
      if(!delta)return;
      if(options_.max_work) {
        if(delta>options_.max_work-work_)throw Limit{"work_budget"};
        work_+=delta;
      }
      remaining=delta>remaining?0:remaining-delta;
      charged_helper=builder.work();
    };
    // All modes borrow ONE immutable epoch catalog. Preserve the full stored
    // class minimum-rank theorem and all address/body witnesses; only later
    // numeric scheduling scans use sparse ascending stored-head ID lists.
    const auto n=snapshot.size();
    struct ModeCatalog {
      std::unordered_map<Class,Class> selected,ordinary_folds,seeded_folds,sum_folds;
      std::vector<Class> eligible_ids,fold_ids,schedule_ids;
    };
    std::array<ModeCatalog,3> mode_catalogs;
    // No whole-node owner/rank/member reconstruction. Union maintains minimum_
    // and actual class membership; this immutable proof epoch borrows them.
    const auto owner_epoch=unions_;
    struct Owners {
      std::function<Class(Class)> lookup;
      Class operator[](Class id) const {return lookup(id);}
    } owners{[&](Class id) {
      if(unions_!=owner_epoch)throw std::logic_error("linear catalog epoch changed");
      if(id>=n)localCap("frozen_owner",id);
      const auto owner=find(id);if(owner>=n)localCap("frozen_owner",owner);
      return owner;
    }};
    struct RankView {
      std::function<bool(Class)> lookup;
      bool operator[](Class id) const {return lookup(id);}
    };
    std::unordered_map<Class,bool> rank_cache;
    RankView ranked{[&](Class id) {
      if(unions_!=owner_epoch)throw std::logic_error("linear catalog epoch changed");
      if(auto found=rank_cache.find(id);found!=rank_cache.end())return found->second;
      charge();const auto owner=owners[id];const auto rank=minimum_[owner];
      bool result=true;
      for(auto child:snapshot[id].children) {
        charge();if(minimum_[owners[child]]>=rank)result=false;
      }
      rank_cache.emplace(id,result);return result;
    }};
    struct SparseFlags {
      std::unordered_set<Class> values;
      bool operator[](Class id) const {return values.contains(id);}
    } admitted;
    struct MemberView {
      std::size_t boundary;
      std::function<std::span<const Class>(Class)> lookup;
      std::size_t size() const {return boundary;}
      std::span<const Class> operator[](Class id) const {return lookup(id);}
    };
    // Map nodes retain their vectors across rehash; only this epoch owns views.
    // Full mapper views cap actual PREFIX members at64, not appended members.
    // Ranked lexical/rectangle views retain every old ranked alternative.
    std::unordered_map<Class,std::optional<std::vector<Class>>> ranked_views,body_views;
    MemberView members{n,[&](Class cls)->std::span<const Class> {
      cls=owners[cls];
      auto found=ranked_views.find(cls);
      if(found==ranked_views.end()) {
        auto view=class_members_.prefix(cls,n,n,[&](std::size_t amount){charge(amount);return true;});
        if(view) {
          auto out=view->begin();
          for(auto id:*view){charge();if(ranked[id])*out++=id;}
          view->erase(out,view->end());
        }
        found=ranked_views.emplace(cls,std::move(view)).first;
      }
      return found->second?std::span<const Class>{*found->second}:std::span<const Class>{};
    }};
    MemberView body_members{n,[&](Class cls)->std::span<const Class> {
      cls=owners[cls];
      auto found=body_views.find(cls);
      if(found==body_views.end()) {
        auto view=class_members_.prefix(cls,n,BodyHeadIndex::max_members,
            [&](std::size_t amount){charge(amount);return true;});
        found=body_views.emplace(cls,std::move(view)).first;
      }
      return found->second?std::span<const Class>{*found->second}:std::span<const Class>{};
    }};
    std::vector<Class> owner_ids,numeric_candidates;
    std::unordered_map<Class,std::string> literals;
    bool catalog_ready=false,catalog_failed=false;
    // Optional scheduling is demanded by the SAME paired-root half pool, not
    // every numerical head ever produced by saturation. Each reached eclass
    // discovers dependencies through ALL physical snapshot aliases. Numerical
    // admission still uses the original minimum-rank theorem. Exact addresses
    // are opaque to this numeric scheduling walk;
    // the full scoped mapper still borrows ALL original body/address members.
    std::vector<Class> demanded_heads;
    std::size_t demanded_owners=0,demanded_root_entries=0,demanded_aliases=0,demanded_unranked=0;
    bool demand_ready=false;
    const auto ensureDemand=[&] {
      if(demand_ready)return;
      std::unordered_set<Class> reached,root_window;
      std::vector<Class> pending;
      const auto reach=[&](Class cls) {
        charge();cls=owners[cls];
        if(realType(snapshot[cls].type) && reached.insert(cls).second) {
          charge(3);pending.push_back(cls);
        }
      };
      const auto root=[&](Class cls) {
        cls=owners[cls];
        if(root_window.insert(cls).second){++demanded_root_entries;reach(cls);}
      };
      for(std::size_t i=0;i<std::max(left.size(),right.size());++i) {
        charge();
        const bool certified=i<left.size() && i<right.size() && owners[left[i]]==owners[right[i]];
        if(certified) {
          // Explicit Fold output targets preserve their existing nullable body
          // group/collision index, even when the mate is already certified.
          // Do not recursively promote equal non-Fold output wrappers/noise.
          if(snapshot[left[i]].kind==static_cast<int>(s::Kind::Fold) ||
              snapshot[right[i]].kind==static_cast<int>(s::Kind::Fold))reach(left[i]);
          continue;
        }
        // Reserve the two possible mate slots exactly as appendRoot does.
        // Missing mates remain ordinary optional roots, never an equality.
        if(demanded_root_entries+2>max_root_candidates)continue;
        if(i<left.size())root(left[i]);
        if(i<right.size())root(right[i]);
      }
      while(!pending.empty()) {
        charge();const auto cls=pending.back();pending.pop_back();++demanded_owners;
        charge(3);owner_ids.push_back(cls);
        // Discovery is NOT an acyclic proof witness. A later compact Fold may
        // fail the min-owner rank test yet be chosen by the full scoped mapper;
        // its actual children still need numerical prerequisite catalogs. Scan
        // all physical aliases once, then compact the SAME owned prefix vector
        // into the original ranked view. Reached owners break alias cycles.
        auto view=class_members_.prefix(cls,n,n,
            [&](std::size_t amount){charge(amount);return true;});
        if(!view)localCap("demand_members",cls);
        // Existing literal precedence is also a discovery atom boundary.
        // Otherwise zero's unrelated x-x/zero-Fold alternatives reconnect
        // every numerical ancestor despite the demanded-root window. This
        // is the SAME finite/plain real literal guard as ensureCatalog(),
        // not an intersection dependency fact or a new equality.
        bool literal_boundary=false;
        for(auto head:*view) {
          charge();const auto& k=snapshot[head];
          if(k.kind==static_cast<int>(s::Kind::Constant) && realType(k.type) &&
              k.attributes.empty() && k.provenance.empty() && finite_[cls] && ranked[head]) {
            literal_boundary=true;break;
          }
        }
        auto ranked_out=view->begin();
        for(auto head:*view) {
          charge();const auto& k=snapshot[head];++demanded_aliases;
          const bool head_ranked=ranked[head];
          if(head_ranked) {
            *ranked_out++=head;
            if(realType(k.type)){charge(3);demanded_heads.push_back(head);}
          } else ++demanded_unranked;
          // InitialCell reads preserve exact address/provenance semantics on
          // their untouched body-map view. No numeric rank/catalog is required
          // for their u64 address, raw bit or opaque exact producer ancestors.
          if(!literal_boundary)for(auto child:k.children) {
            charge();if(realType(snapshot[owners[child]].type))reach(child);
          }
        }
        view->erase(ranked_out,view->end());
        charge();ranked_views.try_emplace(cls,std::move(view));
      }
      if(!stored_class_catalog::Members::orderIds(demanded_heads,n,
          [&](std::size_t amount){charge(amount);return true;}))localCap("demand_order",demanded_heads.size());
      if(!stored_class_catalog::Members::orderIds(owner_ids,n,
          [&](std::size_t amount){charge(amount);return true;}))localCap("demand_owner_order",owner_ids.size());
      demand_ready=true;
    };
    const auto modeIndex=[](s::Reducer reducer) {
      return reducer==s::Reducer::Max?0U:reducer==s::Reducer::Min?1U:2U;
    };
    const auto eligibleFor=[&](const Key& k,s::Reducer reducer) {
      if(!idealOperation(k) || !k.provenance.empty())return false;
      if(reducer!=s::Reducer::Sum) {
        if(k.type!="real" || k.children.size()!=2 ||
            k.name!=(reducer==s::Reducer::Max?"Maximum":"Minimum"))return false;
      } else if(k.name!="Add" && k.name!="Subtract" && k.name!="Negate" && k.name!="Multiply" &&
          k.name!="Divide" && k.name!="Reciprocal")return false;
      return std::all_of(k.children.begin(),k.children.end(),[&](auto c) {
        return snapshot[c].type==k.type && finite_[owners[c]];
      });
    };
    const auto ensureCatalog=[&] {
      if(catalog_ready)return;
      if(n>std::numeric_limits<l::Atom>::max()/2)localCap("private_atom_ids",n);
      ensureDemand();
      for(auto id:demanded_heads) {
        charge();const auto& k=snapshot[id];const auto owner=owners[id];
        // The separately ordered owner list includes a cyclic/opaque minimum
        // owner too, when its LATER ranked alias is the actual chosen witness.
        // Requiring the minimum head itself to rank would discard that alias.
        if(!realType(k.type) || !k.provenance.empty() || !finite_[owner] || !ranked[id])continue;
        admitted.values.insert(owner);
        if(k.kind==static_cast<int>(s::Kind::Constant) && k.attributes.empty())literals.try_emplace(owner,k.name);
        if(k.kind==static_cast<int>(s::Kind::Fold) || k.kind==static_cast<int>(s::Kind::Operation))
          numeric_candidates.push_back(id);
      }
      catalog_ready=true;
    };
    const auto run_mode=[&](s::Reducer reducer) {
    if(catalog_failed)return;
    const bool extrema=reducer!=s::Reducer::Sum;
    l::Statistics linear_statistics;
    const bool trace=sentryward::debug::environment("SENTRYWARD_EGRAPH_TRACE")!=nullptr;
    const auto preparation_start=remaining;
    std::optional<std::size_t> setup_end,groups_end,scheduling_end;
    struct MapTrace {std::size_t calls=0,work=0,start=0;bool active=false;};
    MapTrace origin_trace,domain_trace;
    BodyHeadIndex::HeaderStatistics header_statistics;
    std::size_t origin_identity=0,domain_queries=0,domain_hits=0,domain_identity=0;
    std::size_t frontier_nodes=0,frontier_edges=0,frontier_targets=0;
    std::size_t constant_head_calls=0,constant_head_tasks=0;
    std::size_t constant_cached_false_hits=0,constant_child_cached_false=0;
    std::size_t domain_fallback_reports=16;
    std::size_t domain_denial_reports=16;
    std::size_t coverage_denial_reports=16;
    const auto tracedMap=[&](MapTrace& stats,auto&& invoke) {
      if(trace){++stats.calls;stats.start=remaining;stats.active=true;}
      auto result=invoke();
      if(trace){stats.work+=stats.start-remaining;stats.active=false;}
      return result;
    };
    try {
      phase="linear_setup";
      ensureCatalog();
      auto& catalog=mode_catalogs[modeIndex(reducer)];
      const auto& selected=catalog.selected;
      const auto& ordinary_folds=catalog.ordinary_folds;
      const auto& seeded_folds=catalog.seeded_folds;
      const auto& sum_folds=catalog.sum_folds;
      // Use this strict rank mask exclusively, never OR with the global SCC
      // mask. sum=2*mean can create a proven class SCC even though the original
      // stored Divide(Add(samples),2) is an acyclic lower-ranked witness.
      const auto eligible=[&](const Key& k) {return eligibleFor(k,reducer);};
      // Mode admission remains lazy and ordered: a future Sum catalog cannot
      // spend optional work before an earlier Max/Min proof. The demand catalog
      // retains EVERY finite ranked Operation/Fold alternative of each reached
      // owner, not only a selected head or a recursively expanded shape hint.
      for(auto id:numeric_candidates) {
        charge();const auto& k=snapshot[id];const auto owner=owners[id];
        if(k.kind==static_cast<int>(s::Kind::Fold) && k.fold && k.fold->reducer==reducer &&
            k.children.size()==2 && k.attributes.empty() &&
            snapshot[k.children[0]].type==k.type && snapshot[k.children[1]].type==k.type &&
            (!extrema || k.type=="real") && finite_[owners[k.children[1]]] &&
            (finite_[owners[k.children[0]]] ||
             (reducer==s::Reducer::Max && negative_infinity_[owners[k.children[0]]]) ||
             (reducer==s::Reducer::Min && positive_infinity_[owners[k.children[0]]]))) {
          catalog.sum_folds.try_emplace(owner,id);
          const bool neutral=reducer==s::Reducer::Sum?zero_[owners[k.children[0]]]:
              reducer==s::Reducer::Max?(negative_infinity_[owners[k.children[0]]] ||
                  (k.fold->count && zero_[owners[k.children[0]]] && nonnegative_[owners[k.children[1]]])):
              positive_infinity_[owners[k.children[0]]];
          auto& index=neutral?catalog.ordinary_folds:catalog.seeded_folds;
          if(index.try_emplace(owner,id).second) {
            catalog.fold_ids.push_back(id);catalog.schedule_ids.push_back(id);
          }
        }
        if(eligible(k)) {
          // Keep EVERY eligible alternate for moment primitive indexing.
          catalog.eligible_ids.push_back(id);
          if(catalog.selected.try_emplace(owner,id).second)catalog.schedule_ids.push_back(id);
        }
      }
      // Exact dependencies of ONE selected expression, not the conservative
      // union of dependencies across all eclass alternatives (x-x can pollute
      // a zero class with a binder that this chosen constant does not use).
      std::unordered_map<Class,std::optional<std::vector<std::uint32_t>>> lexical;
      const auto lexicalFree=[&](Class root)->std::optional<std::vector<std::uint32_t>> {
        struct Task {Class cls,head;bool ready;};
        std::vector<Task> pending{{owners[root],n,false}};std::unordered_set<Class> active;
        while(!pending.empty()) {
          charge();auto t=pending.back();pending.pop_back();t.cls=owners[t.cls];
          if(lexical.contains(t.cls))continue;
          if(lexical_entries+active.size()>=max_retained_summary_nodes)
            localCap("lexical_nodes",lexical_entries+active.size());
          if(active.size()>=max_active_nodes)localCap("lexical_active",active.size());
          // The public dependency property is the conservative UNION of all
          // alternatives after the epoch's worklist reaches a fixed point.
          // Empty therefore certifies every chosen witness is closed; it is
          // not an Unknown initialized to empty or an intersection guess.
          if(free_indices_[t.cls].empty()) {
            lexical.emplace(t.cls,std::vector<std::uint32_t>{});++lexical_entries;active.erase(t.cls);continue;
          }
          if(!t.ready) {
            if(!active.insert(t.cls).second)localCap("lexical_cycle",active.size());
            if(literals.contains(t.cls)) {
              lexical.emplace(t.cls,std::vector<std::uint32_t>{});++lexical_entries;active.erase(t.cls);continue;
            }
            if(auto h=selected.find(t.cls);h!=selected.end())t.head=h->second;
            else if(auto h=ordinary_folds.find(t.cls);h!=ordinary_folds.end())t.head=h->second;
            else if(!members[t.cls].empty())t.head=members[t.cls].front();
            if(t.head==n) {lexical.emplace(t.cls,std::nullopt);++lexical_entries;active.erase(t.cls);continue;}
            pending.push_back({t.cls,t.head,true});
            for(auto c:snapshot[t.head].children)if(!lexical.contains(owners[c]))pending.push_back({owners[c],n,false});
            continue;
          }
          const auto& k=snapshot[t.head];std::vector<std::uint32_t> indices;
          bool valid=k.provenance.empty();
          if(k.kind==static_cast<int>(s::Kind::Index))indices.push_back(k.index_level);
          for(std::size_t i=0;i<k.children.size() && valid;++i) {
            const auto c=lexical.find(owners[k.children[i]]);if(c==lexical.end() || !c->second){valid=false;break;}
            for(auto level:*c->second)if(!(k.fold && i==1 && level==k.fold->binder_level))indices.push_back(level);
          }
          charge(indices.size());std::sort(indices.begin(),indices.end());
          indices.erase(std::unique(indices.begin(),indices.end()),indices.end());
          if(indices.size()>8)valid=false;
          lexical.emplace(t.cls,valid?std::optional{std::move(indices)}:std::nullopt);++lexical_entries;active.erase(t.cls);
        }
        return lexical.at(owners[root]);
      };
      // Recursive shapes partition only the bounded stored Fold targets.
      // Candidate atoms use the fixed-local body index below, never this
      // ancestor walk. affineBodyMap still checks every actual leaf/context.
      AffineCache affine_cache;
      BodyHeadIndex body_heads_index(*this,snapshot,body_members,affine_cache,&remaining);
      // ONE opt-in token memo per immutable planner epoch, not per local Add
      // candidate or standalone body query on the same large snapshot.
      body_heads_index.initializeSoleHints(max_retained_summary_nodes,true);
      if(trace)body_heads_index.header_statistics=&header_statistics;
      std::unordered_map<Class,std::optional<Class>> shapes;
      std::unordered_map<Key,Class,Hash> shape_keys;
      std::unordered_set<Class> active_shapes;
      // Large closed factors are exact opaque CLASS identities in this private
      // candidate filter. A small scalar point still receives its normal shape
      // (so constants can match a Fold address); a closed subtree gets at most
      // 64 newly visited heads before retaining exact identity, never a wildcard.
      // Full body/context/affine checks remain the equality certificate.
      constexpr std::size_t closed_shape_visits=64;
      std::function<std::optional<Class>(Class,unsigned,std::size_t*)> shapeImpl;
      shapeImpl=[&](Class cls,unsigned depth,std::size_t* closed_remaining)->std::optional<Class> {
        charge();cls=owners[cls];
        if(auto found=shapes.find(cls);found!=shapes.end())return found->second;
        if(!closed_remaining && free_indices_[cls].empty()) {
          std::size_t local_remaining=closed_shape_visits;
          return shapeImpl(cls,depth,&local_remaining);
        }
        if(depth>128)return {};
        if(shape_entries+active_shapes.size()>=max_retained_summary_nodes){shape_capped=true;return {};}
        if(!active_shapes.insert(cls).second)return {};
        const auto finish=[&](std::optional<Class> result) {
          active_shapes.erase(cls);shapes.emplace(cls,result);++shape_entries;return result;
        };
        Key key;
        if(closed_remaining && !*closed_remaining) {
          if(!free_indices_[cls].empty())return finish({});
          key.kind=10;key.type="closed-eclass-shape-filter";key.children={cls};
        } else if(affineValue(cls,affine_cache,0,&remaining)) {
          if(closed_remaining)--*closed_remaining;
          key.kind=8;key.type="affine-u64-filter";
        }
        else {
          if(closed_remaining)--*closed_remaining;
          Class head=n;
          if(auto choice=body_heads_index.preferred(cls))head=*choice;
          // Local compact heads are scheduling hints only. The shape's own
          // active/depth/closed-factor limits bound this walk; the actual
          // paired body certificate below checks its chosen witness cycles.
          if(head==n)return finish({});
          const auto& source=snapshot[head];
          if(source.type.size()>4096 || source.name.size()>4096 || source.attributes.size()>4096 ||
              source.provenance.size()>4096 || source.children.size()>4096)return finish({});
          charge(source.type.size()+source.name.size()+source.attributes.size()+source.provenance.size()+source.children.size());
          key=snapshot[head];key.children.clear();
          // Candidate-only alpha shape. Every actual binder/count/reducer and
          // captured leaf is subsequently checked in the paired lexical map.
          if(key.fold)key.fold->binder_level=0;
          for(auto child:snapshot[head].children) {
            auto child_shape=shapeImpl(child,depth+1,closed_remaining);if(!child_shape)return finish({});
            key.children.push_back(*child_shape);
          }
          // Class IDs can place a common closed factor on opposite sides of
          // a template and a scalar point. Canonicalize only this candidate
          // filter; affineBodyMap still validates every actual operand/scope.
          if((arithmetic(source) || integerCommutative(source)) && key.children.size()==2 &&
              key.children[1]<key.children[0])std::swap(key.children[0],key.children[1]);
        }
        if(!remaining)return finish({});
        charge(key.children.size());
        auto [entry,inserted]=shape_keys.try_emplace(std::move(key),shape_keys.size());(void)inserted;
        return finish(entry->second);
      };
      const auto shape=[&](Class cls,unsigned depth) {return shapeImpl(cls,depth,nullptr);};
      const auto scopeFor=[&](Class cls)->std::optional<std::vector<std::uint32_t>> {
        cls=owners[cls];
        if(literals.contains(cls))return std::vector<std::uint32_t>{};
        // Fixed-point dependency UNION is a conservative lexical upper bound.
        // Missing binder proves absence; extra captured levels only reject a
        // match. This PRIVATE filter never establishes expression equality.
        // A chosen Fold's literal seed/body has certified empty dependencies,
        // even when x-x alternatives polluted that literal's public union.
        if(!extrema)if(auto f=sum_folds.find(cls);f!=sum_folds.end()) {
          const auto& k=snapshot[f->second];std::vector<std::uint32_t> levels;
          for(unsigned side=0;side<2;++side) {
            const auto child=owners[k.children[side]];if(literals.contains(child))continue;
            const auto& bound=free_indices_[child];charge(bound.size());
            if(bound.size()>8)return {};
            for(auto level:bound)if(side==0 || level!=k.fold->binder_level)levels.push_back(level);
          }
          charge(levels.size());std::sort(levels.begin(),levels.end());
          levels.erase(std::unique(levels.begin(),levels.end()),levels.end());
          if(levels.size()>8)return {};return levels;
        }
        // The full-body map below still checks every actual index operand,
        // seed and capture. Moments use a separate EXACT lexical index below.
        const auto& levels=free_indices_[cls];charge(levels.size());
        if(levels.size()>8)return {};return levels;
      };
      using BucketKey=std::tuple<std::string,std::vector<std::uint32_t>,Class,s::Reducer>;
      // The last token names a SCHEDULING component, not an equality fact.
      // One wildcard body skeleton can contain functions with incompatible
      // relative affine origins (e.g. x(i)*state(i+page)). Full maps below are
      // still the only body/coordinate certificates.
      using GroupKey=std::tuple<std::string,std::vector<std::uint32_t>,Class,s::Reducer,Class>;
      using Captures=std::vector<std::pair<std::uint32_t,std::uint64_t>>;
      using AnchorFamily=std::tuple<Class,std::uint64_t,std::uint32_t,Captures,std::uint64_t>;
      struct AnchorInterval {const AnchorFamily* family;std::uint64_t begin,end;};
      struct Group {
        Class fold;
        std::array<std::unordered_map<Class,std::optional<std::pair<std::uint64_t,std::uint64_t>>>,2> mapped;
        std::optional<AnchorInterval> anchor;
      };
      struct ReadAnchor {Class tag;IndexAffine address;};
      const auto uniqueReadAnchor=[&](Class body)->std::optional<ReadAnchor> {
        // Exactly THREE stored header levels, at most 1+2+4 heads. No address
        // ancestor walk, Fold BODY descent, candidate Cartesian search or
        // recursive signature expansion. The existing affine cache is shared.
        std::array<std::pair<Class,unsigned>,7> pending{};std::size_t size=1;
        pending[0]={owners[body],0};std::optional<Class> read;
        while(size) {
          charge();const auto [cls,depth]=pending[--size];
          auto head=body_heads_index.preferred(cls);if(!head)return {};
          const auto& k=snapshot[*head];
          const bool memory=k.kind==static_cast<int>(s::Kind::InitialCell) ||
              k.kind==static_cast<int>(s::Kind::InputValue);
          if(memory && realType(k.type) && k.children.size()==1 && finite_[owners[*head]]) {
            if(read && owners[*read]!=owners[*head])return {};
            read=*head;continue;
          }
          if(depth==2 || k.kind!=static_cast<int>(s::Kind::Operation) ||
              !idealOperation(k) || k.children.size()>2)continue;
          for(auto child:k.children)if(realType(snapshot[owners[child]].type)) {
            charge();if(size==pending.size())return {};
            pending[size++]={owners[child],depth+1};
          }
        }
        if(!read)return {};
        auto tag=body_heads_index.header(*read);if(!tag)return {};
        auto address=body_heads_index.affine(snapshot[*read].children[0]);
        if(!address || address->coefficients.size()>64)return {};
        charge(address->coefficients.size());return ReadAnchor{*tag,std::move(*address)};
      };
      struct Prototype {Class fold;const GroupKey* key;};
      struct Partition {
        // Each actual target contributes at most one immutable anchor record.
        // A collision at a coordinate is nullable, not alternative search.
        std::map<AnchorFamily,std::map<std::uint64_t,std::optional<Prototype>>> prototypes;
        std::size_t components=0;
        bool anchored=false,unsupported=false;
      };
      std::map<BucketKey,Partition> partitions;
      std::size_t prototype_entries=0;
      const auto targetAnchor=[&](const Key& k)->std::optional<std::pair<AnchorFamily,AnchorInterval>> {
        auto read=uniqueReadAnchor(k.children[1]);if(!read || !k.fold || !k.fold->count)return {};
        const auto binder=k.fold->binder_level;
        auto coefficient=read->address.coefficients.find(binder);
        if(coefficient==read->address.coefficients.end() || !coefficient->second)return {};
        const auto stride=coefficient->second,constant=read->address.constant;
        // Check the CONSTANT coordinate lattice, not an invented no-wrap
        // property for captured addresses. Full u64-ring/capture checks remain
        // mandatory. A wrapping lattice conservatively has no scheduling key.
        if(k.fold->count-1>(UINT64_MAX-constant)/stride)return {};
        const auto begin=constant/stride;
        if(k.fold->count>UINT64_MAX-begin)return {};
        Captures captures;captures.reserve(read->address.coefficients.size()-1);
        for(auto [level,value]:read->address.coefficients)if(level!=binder) {
          charge();if(level>=binder)return {};captures.push_back({level,value});
        }
        charge(captures.size()+8);
        // An absolute bound label is not part of this scheduling family.
        // Distinct depths can denote the same sample function. Keep EVERY
        // actual label in the full origin proof and canonical Fold below;
        // captured affine coefficients stay exact, never alpha-erased here.
        AnchorFamily family{read->tag,stride,0,std::move(captures),constant%stride};
        return std::pair{std::move(family),AnchorInterval{nullptr,begin,begin+k.fold->count}};
      };
      const auto sameBoundFoldBody=[&](const Key& a,const Key& b) {
        charge(8);
        // Pointwise equality of the SAME proven body, not a scheduling/header
        // hint. Callers separately retain the captured query scope and seed.
        // A different binder must still use the complete lexical body map.
        return a.kind==static_cast<int>(s::Kind::Fold) && b.kind==a.kind &&
            a.fold && b.fold && a.children.size()==2 && b.children.size()==2 &&
            a.type==b.type && a.fold->reducer==b.fold->reducer &&
            a.fold->binder_level==b.fold->binder_level &&
            a.attributes.empty() && b.attributes.empty() &&
            a.provenance.empty() && b.provenance.empty() &&
            snapshot[a.children[0]].type==a.type && snapshot[a.children[1]].type==a.type &&
            snapshot[b.children[0]].type==b.type && snapshot[b.children[1]].type==b.type &&
            owners[a.children[1]]==owners[b.children[1]];
      };
      std::map<GroupKey,Group> groups;
      // An admitted ACTUAL Fold header can retain its existing target group
      // without asking a lossy scalar-point hint to rediscover that group.
      // std::map keys are stable for this immutable epoch; no body or domain
      // equality is asserted by the borrowed key. Conflicting heads are Unknown.
      struct FoldMembership {Class head;const GroupKey* key;std::optional<AnchorInterval> anchor;};
      std::unordered_map<Class,std::optional<FoldMembership>> fold_memberships;
      std::vector<Class> membership_owners;
      std::set<std::pair<int,std::string>> body_heads;
      // Fixed-local metadata only. Do not recursively shape/map every stored
      // Fold before an actual certificate candidate has been selected.
      std::vector<Class> potential_target_ids;
      std::set<std::pair<std::string,std::vector<std::uint32_t>>> group_scopes;
      for(auto id:catalog.fold_ids) {
        charge();const auto owner=owners[id];
        const bool ordinary=ordinary_folds.contains(owner) && ordinary_folds.at(owner)==id;
        const bool seeded=extrema && seeded_folds.contains(owner) && seeded_folds.at(owner)==id;
        if((!ordinary && !seeded) || !snapshot[id].fold->count)continue;
        const auto body=owners[snapshot[id].children[1]];
        if(!extrema && (literals.contains(body) ||
            !dependsOn(body,snapshot[id].fold->binder_level)))continue;
        auto scope=scopeFor(owner);if(!scope)continue;
        const auto body_head=selected.contains(body)?selected.at(body):
            members[body].empty()?body:members[body].front();
        if(snapshot[body_head].name.size()>4096)continue;
        charge(8+snapshot[id].type.size()+scope->size()+snapshot[body_head].name.size());
        potential_target_ids.push_back(id);
        group_scopes.emplace(snapshot[id].type,std::move(*scope));
        body_heads.emplace(snapshot[body_head].kind,snapshot[body_head].name);
      }

      // Only a necessary scheduling discriminator, never a body certificate.
      // Construct it lazily, once per reducer/epoch, after the root-only filter
      // below. Unrelated output Divide/Multiply heads must not fill the shared
      // body matcher's retained-entry caches before any coverage query can run.
      bool body_shallow_ready=false;
      std::unordered_set<Class> body_shallow;
      const auto pointBodyHead=[&](Class owner) {
        const auto& alternatives=body_members[owner];
        if(alternatives.empty() || alternatives.size()>BodyHeadIndex::max_members)return false;
        bool possible=false;
        for(auto id:alternatives) {
          const auto& k=snapshot[id];
          if(k.name.size()>4096)continue;
          charge(1+k.name.size());
          if(body_heads.contains({k.kind,k.name})){possible=true;break;}
        }
        if(!possible)return false;
        if(!body_shallow_ready) {
          for(auto id:potential_target_ids) {
            charge();
            if(auto signature=body_heads_index.shallow(snapshot[id].children[1]))
              body_shallow.insert(*signature);
          }
          body_shallow_ready=true;
        }
        if(auto signature=body_heads_index.shallow(owner))return body_shallow.contains(*signature);
        return false;
      };
      // Coverage filters may carry conservative extra dependencies. Moment
      // semantics must not use those keys as an exact lexical context. Build
      // one lazy index of the existing (at most max_groups) stored targets,
      // retaining their public tokens only after an exact selected-DAG scope
      // check. Colliding target contexts are Unknown, never alternative search.
      bool exact_groups_ready=false;
      std::map<GroupKey,std::optional<GroupKey>> exact_groups;
      std::set<std::pair<std::string,std::vector<std::uint32_t>>> exact_group_scopes;
      const auto exactGroups=[&] {
        if(exact_groups_ready)return;
        for(const auto& [key,group]:groups) {
          charge();auto scope=lexicalFree(owners[group.fold]);if(!scope)continue;
          charge(std::get<0>(key).size()+scope->size());
          GroupKey exact{std::get<0>(key),*scope,std::get<2>(key),reducer,std::get<4>(key)};
          auto [entry,inserted]=exact_groups.try_emplace(exact,key);
          if(!inserted && entry->second && *entry->second!=key)entry->second.reset();
          exact_group_scopes.emplace(std::get<0>(key),std::move(*scope));
        }
        exact_groups_ready=true;
      };
      const auto admittedFoldGroup=[&](Class head,const std::string& type,
          const std::vector<std::uint32_t>& scope,bool semantic=false)->std::optional<GroupKey> {
        charge();if(head>=n)return {};
        const auto& source=snapshot[head];
        if(source.kind!=static_cast<int>(s::Kind::Fold) || !source.fold ||
            source.type!=type || source.fold->reducer!=reducer)return {};
        auto member=fold_memberships.find(owners[head]);
        if(member==fold_memberships.end() || !member->second || member->second->head!=head)return {};
        const auto& key=*member->second->key;
        if(std::get<0>(key)!=type || std::get<3>(key)!=reducer)return {};
        if(!semantic)return std::get<1>(key)==scope?std::optional{key}:std::nullopt;
        // Family mean queries remove an outer axis. Do not silently substitute
        // that different query scope for this Fold's actual enclosing context.
        auto actual=lexicalFree(owners[head]);if(!actual || *actual!=scope)return {};
        exactGroups();charge(type.size()+scope.size());
        auto found=exact_groups.find(GroupKey{type,scope,std::get<2>(key),reducer,std::get<4>(key)});
        return found==exact_groups.end()?std::nullopt:found->second;
      };
      // Index at most max_groups existing target groups by their shared public
      // fixed-radius scheduling signature. A shallow collision NEVER chooses a
      // target or merges groups: only a unique existing GroupKey is returned.
      // This is a necessary filter, not a body/coordinate equality fact.
      std::array<std::map<BucketKey,std::optional<GroupKey>>,2> body_groups;
      using AnchorSchema=std::pair<Class,Captures>;
      struct AnchorSegment {std::uint64_t begin,end;const GroupKey* key;};
      struct AnchorEvent {std::uint64_t coordinate;const GroupKey* key;bool begin;};
      struct AnchorLattice {
        std::uint64_t stride;std::uint32_t binder;
        std::map<std::uint64_t,std::vector<AnchorEvent>> events;
        std::map<std::uint64_t,std::vector<AnchorSegment>> segments;
      };
      struct AnchorBucket {
        bool partitioned=false;
        // The point has no derivative from which to guess a target stride.
        // Different strides/binders for one read/capture schema are Unknown.
        std::map<AnchorSchema,std::optional<AnchorLattice>> lattices;
      };
      std::array<std::map<BucketKey,AnchorBucket>,2> anchor_buckets;
      std::array<bool,2> body_groups_ready{};
      const auto bodyGroups=[&](bool semantic) {
        const unsigned mode=semantic?1U:0U;
        if(body_groups_ready[mode])return;
        if(semantic)exactGroups();
        std::map<const GroupKey*,BucketKey> group_queries;
        for(const auto& [key,group]:groups) {
          charge();auto scope=semantic?lexicalFree(owners[group.fold]):
              std::optional{std::get<1>(key)};
          if(!scope)continue;
          auto signature=body_heads_index.shallow(snapshot[group.fold].children[1]);
          if(!signature)continue;
          std::optional<GroupKey> value=key;
          if(semantic) {
            GroupKey exact{std::get<0>(key),*scope,std::get<2>(key),reducer,std::get<4>(key)};
            auto found=exact_groups.find(exact);
            value=found==exact_groups.end()?std::nullopt:found->second;
          }
          charge(std::get<0>(key).size()+scope->size());
          BucketKey query{std::get<0>(key),std::move(*scope),*signature,reducer};
          charge(32+std::get<0>(query).size()+std::get<1>(query).size());
          // Semantic anchor lookup must not bypass the existing nullable
          // exact lexical index, even though it remains scheduling only.
          if(value)group_queries.emplace(&key,query);
          auto& anchors=anchor_buckets[mode][query];
          const auto& partition=partitions.at(BucketKey{std::get<0>(key),std::get<1>(key),std::get<2>(key),reducer});
          anchors.partitioned|=partition.anchored && (partition.components>1 || partition.unsupported);
          auto [entry,inserted]=body_groups[mode].try_emplace(std::move(query),value);
          if(!inserted && entry->second!=value) {
            // Print only the FIRST distinct-key collision in a bucket. Once
            // nullable, that bucket remains Unknown without more diagnostics.
            if(sentryward::debug::environment("SENTRYWARD_EGRAPH_TRACE") && entry->second && value) {
              SENTRYWARD_DEBUG_PRINT(egraph_5);
              for(auto level:std::get<1>(entry->first))SENTRYWARD_DEBUG_PRINT(egraph_6);
              SENTRYWARD_DEBUG_PRINT(egraph_7);
            }
            entry->second.reset();
          }
        }
        // Sweep actual stored target intervals ONCE, not a group scan per
        // scalar point. Retained targets yield at most twice as many events;
        // the separate aggregate interval-entry cap still applies to segments.
        // Overlap between distinct components is Unknown; gaps
        // stay gaps. Neither a hull nor an anchor is an equality certificate.
        for(auto owner:membership_owners) {
          charge();const auto& member=fold_memberships.at(owner);
          if(!member || !member->anchor)continue;
          auto query=group_queries.find(member->key);if(query==group_queries.end())continue;
          auto& bucket=anchor_buckets[mode].at(query->second);if(!bucket.partitioned)continue;
          const auto& interval=*member->anchor;const auto& family=*interval.family;
          // Only full origin certificates join this component. Its ACTUAL
          // canonical binder is shared by its scheduling records, even when
          // an equivalent original target uses another absolute bound label.
          const auto canonical_binder=snapshot[groups.at(*member->key).fold].fold->binder_level;
          AnchorSchema schema{std::get<0>(family),std::get<3>(family)};
          charge(4*(schema.second.size()+8)*(1+std::bit_width(bucket.lattices.size()+1)));
          auto [entry,inserted]=bucket.lattices.try_emplace(std::move(schema),
              AnchorLattice{std::get<1>(family),canonical_binder,{}, {}});
          if(!inserted && entry->second && (entry->second->stride!=std::get<1>(family) ||
              entry->second->binder!=canonical_binder))entry->second.reset();
          if(!entry->second)continue;
          auto& events=entry->second->events[std::get<4>(family)];
          charge(8);events.push_back({interval.begin,member->key,true});
          events.push_back({interval.end,member->key,false});
        }
        for(auto& [query,bucket]:anchor_buckets[mode]) {
          (void)query;if(!bucket.partitioned)continue;
          for(auto& [schema,lattice]:bucket.lattices) {
            (void)schema;if(!lattice)continue;
            for(auto& [residue,events]:lattice->events) {
              std::sort(events.begin(),events.end(),[&](const auto& a,const auto& b) {
                charge();return a.coordinate<b.coordinate;
              });
              std::map<const GroupKey*,std::size_t> active;
              auto& segments=lattice->segments[residue];
              for(std::size_t i=0;i<events.size();) {
                const auto coordinate=events[i].coordinate;
                do {
                  charge(4*(1+std::bit_width(active.size()+1)));
                  const auto& event=events[i++];
                  if(event.begin)++active[event.key];
                  else if(auto found=active.find(event.key);found!=active.end()) {
                    if(!--found->second)active.erase(found);
                  } else throw std::logic_error("anchor interval imbalance");
                } while(i<events.size() && events[i].coordinate==coordinate);
                if(i==events.size() || active.size()!=1)continue;
                const auto end=events[i].coordinate;const auto key=active.begin()->first;
                charge(4);
                if(!segments.empty() && segments.back().end==coordinate && segments.back().key==key)
                  segments.back().end=end;
                else {
                  if(interval_entries>=max_interval_entries)localCap("coverage_intervals",interval_entries);
                  ++interval_entries;segments.push_back({coordinate,end,key});
                }
              }
              if(!active.empty())throw std::logic_error("anchor interval imbalance");
            }
            // No borrowed vector/span survives this release. The owned final
            // segments are immutable for the remainder of the reducer epoch.
            lattice->events.clear();
          }
        }
        body_groups_ready[mode]=true;
      };
      const auto candidateGroup=[&](Class body,const std::string& type,
          const std::vector<std::uint32_t>& scope,bool semantic=false,
          bool* ambiguous=nullptr)->std::optional<GroupKey> {
        charge();body=owners[body];
        // Existing bounded root/header discriminator avoids indexing every
        // unrelated finite residual atom. No recursive private shape is read.
        if(!pointBodyHead(body))return {};
        bodyGroups(semantic);
        auto signature=body_heads_index.shallow(body);if(!signature)return {};
        charge(type.size()+scope.size());
        const auto& index=body_groups[semantic?1:0];
        const BucketKey query{type,scope,*signature,reducer};
        auto bucket=anchor_buckets[semantic?1:0].find(query);
        if(bucket!=anchor_buckets[semantic?1:0].end() && bucket->second.partitioned) {
          // No legacy/cross-component fallback in a partitioned bucket. A
          // denied possible sample must not be demoted to an opaque residual.
          if(ambiguous)*ambiguous=true;
          auto read=uniqueReadAnchor(body);if(!read)return {};
          Captures captures;captures.reserve(read->address.coefficients.size());
          for(auto entry:read->address.coefficients){charge();captures.push_back(entry);}
          charge(4*(captures.size()+8)*(1+std::bit_width(bucket->second.lattices.size()+1)));
          auto lattice=bucket->second.lattices.find(AnchorSchema{read->tag,std::move(captures)});
          if(lattice==bucket->second.lattices.end() || !lattice->second)return {};
          const auto& value=*lattice->second;
          auto residue=value.segments.find(read->address.constant%value.stride);
          if(residue==value.segments.end())return {};
          const auto point=read->address.constant/value.stride;
          charge(4*(1+std::bit_width(residue->second.size()+1)));
          auto segment=std::upper_bound(residue->second.begin(),residue->second.end(),point,
              [](std::uint64_t p,const auto& s){return p<s.begin;});
          if(segment==residue->second.begin())return {};--segment;
          if(point>=segment->end)return {};
          if(ambiguous)*ambiguous=false;
          return *segment->key;
        }
        auto found=index.find(query);
        if(found!=index.end() && !found->second && ambiguous)*ambiguous=true;
        return found==index.end()?std::nullopt:found->second;
      };
      // Constant-only classification is a separate, memoized selected-DAG
      // property. Do it before asking for linear forms: proving that neither
      // factor is a scalar must not materialize two large variance/mean forms.
      std::unordered_map<Class,bool> constant_heads;
      const auto trivialConstantHead=[&](Class cls)->std::optional<bool> {
        cls=owners[cls];
        // Literal alternatives take precedence over conservative dependency
        // contamination from equivalent nonliteral witnesses. Otherwise a
        // literal-only arithmetic witness must be closed and have a selected
        // arithmetic head. These necessary negatives never create facts.
        if(literals.contains(cls))return true;
        if(!selected.contains(cls) || !free_indices_[cls].empty())return false;
        return {};
      };
      const auto immediateConstantHead=[&](Class cls)->std::optional<bool> {
        if(auto known=trivialConstantHead(cls))return known;
        if(auto cached=constant_heads.find(cls);cached!=constant_heads.end()) {
          if(trace && !cached->second)++constant_cached_false_hits;
          return cached->second;
        }
        // At most two immediate operands. An opaque/free-index operand makes
        // this literal-only classification false without storing another
        // negative entry for every ordinary variable product or sum.
        bool all=true;
        for(auto child:snapshot[selected.at(cls)].children) {
          charge();const auto known=trivialConstantHead(owners[child]);
          // Diagnostic only: count an existing negative child memo that this
          // current classifier does not consult. Do not change its result,
          // recurse, materialize a form, or create another memo entry.
          if(trace && !known) {
            auto cached=constant_heads.find(owners[child]);
            if(cached!=constant_heads.end() && !cached->second)++constant_child_cached_false;
          }
          if(known && !*known)return false;
          if(!known)all=false;
        }
        return all?std::optional{true}:std::nullopt;
      };
      const auto constantHead=[&](Class root) {
        if(trace)++constant_head_calls;
        struct Task {Class cls;bool ready;};
        std::vector<Task> pending{{owners[root],false}};std::unordered_set<Class> active_constants;
        while(!pending.empty()) {
          if(trace)++constant_head_tasks;
          charge();auto t=pending.back();pending.pop_back();t.cls=owners[t.cls];
          if(immediateConstantHead(t.cls))continue;
          if(constant_entries+active_constants.size()>=max_retained_summary_nodes)
            localCap("constant_nodes",constant_entries+active_constants.size());
          if(active_constants.size()>=max_active_nodes)localCap("constant_active",active_constants.size());
          auto head=selected.find(t.cls);
          const auto& k=snapshot[head->second];
          if(!t.ready) {
            if(!active_constants.insert(t.cls).second)localCap("constant_cycle",active_constants.size());
            pending.push_back({t.cls,true});
            for(auto child:k.children)if(!immediateConstantHead(owners[child]))pending.push_back({owners[child],false});
          } else {
            const bool all=std::all_of(k.children.begin(),k.children.end(),
                [&](auto child){return immediateConstantHead(owners[child]).value();});
            constant_heads.emplace(t.cls,all);++constant_entries;active_constants.erase(t.cls);
          }
        }
        return immediateConstantHead(owners[root]).value();
      };
      const auto linearHead=[&](const Key& k) {
        if(k.name=="Multiply")return constantHead(k.children[0]) || constantHead(k.children[1]);
        if(k.name=="Divide")return constantHead(k.children[1]);
        if(k.name=="Reciprocal")return constantHead(k.children[0]);
        return true;
      };
      // Candidate indexing is optional scheduling, not a constant/equality
      // fact. Unknown immediate coefficients leave their descendants eligible
      // as separate components; only an actual summary asks the bounded deep
      // classifier. Do not fill its memo by walking every unrelated head.
      const auto indexedLinearHead=[&](const Key& k) {
        const auto knownConstant=[&](Class child) {
          charge();return immediateConstantHead(owners[child]).value_or(false);
        };
        if(k.name=="Multiply")return knownConstant(k.children[0]) || knownConstant(k.children[1]);
        if(k.name=="Divide")return knownConstant(k.children[1]);
        if(k.name=="Reciprocal")return knownConstant(k.children[0]);
        return true;
      };
      // One public compact catalog, not a private cache of every expanded
      // prefix form. The shared session aggregates one exact scalar adjoint
      // per reached stored node; only the caller-owned final form is expanded.
      // Virtual IDs are bounded scalar headers, never logical Fold elements.
      auto session_limits=limits;
      session_limits.max_nodes=max_retained_summary_nodes-summary_entries;
      session_limits.max_cached_terms=max_cached_terms-cached_terms;
      l::DagSession linear_session(builder,[&](l::NodeId id)->std::optional<l::Node> {
        charge();l::Node node;node.admitted=true;
        if(id>=2*n) {
          const auto cls=id-2*n;
          if(cls>=n || !selected.contains(cls) || snapshot[selected.at(cls)].name!="Reciprocal")return {};
          node.kind=l::NodeKind::Literal;node.literal="1";node.type=snapshot[cls].type;return node;
        }
        if(id>=n) {
          const auto cls=id-n;
          if(cls>=n || !seeded_folds.contains(cls) || !snapshot[seeded_folds.at(cls)].fold->count)return {};
          node.kind=l::NodeKind::Atom;node.atom=static_cast<l::Atom>(id);node.type=snapshot[cls].type;return node;
        }
        const auto cls=static_cast<Class>(id);
        if(owners[cls]!=cls || !admitted[cls])return {};
        node.type=snapshot[cls].type;
        if(literals.contains(cls)) {node.kind=l::NodeKind::Literal;node.literal=literals.at(cls);return node;}
        if(!ordinary_folds.contains(cls) && seeded_folds.contains(cls)) {
          const auto& k=snapshot[seeded_folds.at(cls)];
          // The seed stays OUTSIDE the binder, once. The private region atom
          // retains the existing reductionFold()/mappedDomain semantic key.
          node.kind=k.fold->count?l::NodeKind::Add:l::NodeKind::Identity;
          node.children={owners[k.children[0]],n+cls};node.child_count=k.fold->count?2:1;return node;
        }
        if(auto head=selected.find(cls);head!=selected.end() && linearHead(snapshot[head->second])) {
          const auto& k=snapshot[head->second];node.child_count=k.children.size();
          if(k.name=="Reciprocal") {node.kind=l::NodeKind::Divide;node.children={2*n+cls,owners[k.children[0]]};node.child_count=2;return node;}
          if(node.child_count>2)return {};
          for(std::size_t i=0;i<node.child_count;++i)node.children[i]=owners[k.children[i]];
          node.kind=l::NodeKind::Add;
          if(k.name=="Subtract")node.kind=l::NodeKind::Subtract;
          if(k.name=="Negate")node.kind=l::NodeKind::Negate;
          if(k.name=="Multiply")node.kind=l::NodeKind::Multiply;
          if(k.name=="Divide")node.kind=l::NodeKind::Divide;
          return node;
        }
        // Unselected/nonlinear admitted heads are the SAME finite atom. No
        // nonconstant denominator or operand is cancelled or opened here.
        node.kind=l::NodeKind::Atom;node.atom=cls;return node;
      },session_limits);
      std::size_t session_nodes=0,session_terms=0;
      const auto summarize=[&](Class root)->l::Result {
        auto result=linear_session.summarize(owners[root]);
        const auto stats=linear_session.statistics();linear_statistics=stats;
        if(stats.nodes-session_nodes>max_retained_summary_nodes-summary_entries)localCap("summary_nodes",summary_entries);
        if(stats.cached_terms-session_terms>max_cached_terms-cached_terms)localCap("summary_terms",cached_terms);
        summary_entries+=stats.nodes-session_nodes;cached_terms+=stats.cached_terms-session_terms;
        session_nodes=stats.nodes;session_terms=stats.cached_terms;sync();
        if(!result) {
          switch(linear_session.cap()) {
            case l::DagSession::Cap::Nodes:localCap("summary_nodes",summary_entries);break;
            case l::DagSession::Cap::CachedTerms:localCap("summary_terms",cached_terms);break;
            case l::DagSession::Cap::Atoms:localCap("summary_atoms",limits.max_atoms);break;
            case l::DagSession::Cap::Work:localCap("linear_builder",builder.work());break;
            case l::DagSession::Cap::Coefficients:localCap("summary_coefficients",session_nodes);break;
            case l::DagSession::Cap::None:localCap("summary_unknown",session_nodes);break;
          }
        }
        return result;
      };
      struct CoverageKey {
        GroupKey group;
        l::Coefficient constant;
        std::vector<l::WeightedInterval> intervals;
        // Exact finite atoms outside the one certified reduction-body group.
        // A nonliteral seed is retained once, outside the Fold binder, rather
        // than discarded or treated as a uniform value at every sample.
        std::vector<l::Term> residual;
      };
      const auto less=[&](const CoverageKey& a,const CoverageKey& b) {
        charge(8);
        if(a.group!=b.group)return a.group<b.group;
        // Compare the cheap interval geometry before exact coefficient
        // payloads. Distinct short Fold domains should not spend the bounded
        // rational-comparison allowance merely to order scalar endpoints.
        const auto size=std::min(a.intervals.size(),b.intervals.size());
        for(std::size_t i=0;i<size;++i) {
          charge();const auto& x=a.intervals[i];const auto& y=b.intervals[i];
          if(x.begin!=y.begin)return x.begin<y.begin;
          if(x.end!=y.end)return x.end<y.end;
          charge(128);
          if(x.weight!=y.weight)return x.weight<y.weight;
        }
        if(a.intervals.size()!=b.intervals.size())return a.intervals.size()<b.intervals.size();
        charge(128);if(a.constant!=b.constant)return a.constant<b.constant;
        const auto residual_size=std::min(a.residual.size(),b.residual.size());
        for(std::size_t i=0;i<residual_size;++i) {
          charge();const auto& x=a.residual[i];const auto& y=b.residual[i];
          if(x.atom!=y.atom)return x.atom<y.atom;
          charge(128);
          if(x.weight!=y.weight)return x.weight<y.weight;
        }
        return a.residual.size()<b.residual.size();
      };
      std::map<CoverageKey,Class,decltype(less)> signatures(less);
      setup_end=remaining;
      std::vector<Class> candidates;std::unordered_set<Class> candidate_set;
      std::unordered_set<Class> linear_parented;
      const auto append=[&](Class id) {
        const auto owner=owners[id];
        if(candidate_entries<max_candidates && admitted[owner] && candidate_set.insert(owner).second) {
          candidates.push_back(owner);++candidate_entries;
        }
      };
      const auto appendRoot=[&](Class id,bool room) {
        charge();const auto owner=owners[id];
        // Whole outputs already have their own bounded priority pool. They
        // must not refill the prerequisite pool as "maximal components" when
        // many outputs depend on the SAME interior Fold/linear certificate.
        charge();linear_parented.insert(owner);
        if(!room || candidate_entries>=max_root_candidates)return;
        // Complete linear roots and reduction regions are normal candidates.
        // A nonlinear root is one opaque sample only when its bounded local
        // head matches an indexed Fold BODY head; do not scan its descendants.
        if(ordinary_folds.contains(owner) || seeded_folds.contains(owner) ||
            (selected.contains(owner) && indexedLinearHead(snapshot[selected.at(owner)]))) {
          append(id);return;
        }
        if(candidate_entries>=max_candidates || !admitted[owner] || candidate_set.contains(owner))return;
        // This lookup schedules only. Coverage still checks the complete
        // selected witness, lexical context, affine address and weighted domain.
        if(pointBodyHead(owner))append(id);
      };
      // Index each complete output pair before unrelated later roots. A large
      // unsupported root must not exhaust a local certificate ledger between
      // two small mates which would already have a complete equality key.
      for(std::size_t i=0;i<std::max(left.size(),right.size());++i) {
        // Reserve both possible mate slots before admitting this root pair.
        // The remaining half stays available for certificate prerequisites.
        // A frozen e-class equality is already a complete mate certificate,
        // not a head/shape hint. Keep both parent markers via appendRoot, but
        // do not spend optional root slots or promote their interior boundaries.
        const bool certified=i<left.size() && i<right.size() && owners[left[i]]==owners[right[i]];
        const bool room=!certified && candidate_entries+2<=max_root_candidates;
        if(i<left.size())appendRoot(left[i],room);
        if(i<right.size())appendRoot(right[i],room);
      }
      const auto root_candidates=candidates.size();
      phase="maximal_linear";
      // Fold-containing maximal components precede optional moments and the
      // many individual Fold candidates. Scalar-only components are deferred:
      // eagerly summarizing every partial quadratic-merge mean/correction can
      // consume the shared work allowance before a complete moment proof.
      // Both phases share the original 1024 candidate/work/summary limits and
      // the one half-pool component cap; no growing prefix can fill that pool.
      for(const auto& [cls,head]:selected) {
        charge();if(literals.contains(cls) || !indexedLinearHead(snapshot[head]))continue;
        for(auto child:snapshot[head].children){charge();linear_parented.insert(owners[child]);}
      }
      // A maximal heterogeneous Add/Subtract need not itself have one coverage
      // group. Keep its immediate linear region as a prerequisite instead of
      // hiding that region behind an unrelated opaque scalar/reduction arm.
      // Inspect only maximal heads (or already admitted whole roots), never
      // every Add prefix or an ancestor path. The bounded scope/header tests
      // schedule a candidate; its complete weighted-body certificate is final.
      std::unordered_set<Class> linear_boundaries;
      // Keep only one bounded stored-head schedule, not another whole-DAG
      // traversal. Earliest IDs are deterministic even though selected is an
      // unordered index; omitted boundaries remain ordinary later candidates.
      std::set<Class> boundary_heads;
      for(const auto& [cls,head]:selected) {
        charge();const auto& k=snapshot[head];
        if((linear_parented.contains(cls) && !candidate_set.contains(cls)) ||
            (k.name!="Add" && k.name!="Subtract") || k.children.size()!=2)continue;
        const std::array<Class,2> children{owners[k.children[0]],owners[k.children[1]]};
        const auto linearChild=[&](Class child) {
          charge();return admitted[child] && !literals.contains(child) &&
              (ordinary_folds.contains(child) || seeded_folds.contains(child) ||
               (selected.contains(child) && indexedLinearHead(snapshot[selected.at(child)]))) &&
              !immediateConstantHead(child).value_or(false);
        };
        const bool a=linearChild(children[0]),b=linearChild(children[1]);
        if(a==b)continue;
        const auto region=children[a?0:1],other=children[a?1:0];
        if(literals.contains(other) || immediateConstantHead(other).value_or(false))continue;
        auto scope=scopeFor(region);if(!scope)continue;
        charge();if(!group_scopes.contains({snapshot[region].type,*scope}) ||
            !linear_boundaries.insert(region).second)continue;
        const auto region_head=ordinary_folds.contains(region)?ordinary_folds.at(region):
            seeded_folds.contains(region)?seeded_folds.at(region):selected.at(region);
        // set operations and their bounded metadata copies are charged. This
        // shares the existing candidate cap, not a second admission allowance.
        charge(32);
        if(boundary_heads.size()<max_candidates || region_head<*boundary_heads.rbegin()) {
          boundary_heads.insert(region_head);
          if(boundary_heads.size()>max_candidates)boundary_heads.erase(std::prev(boundary_heads.end()));
        }
      }
      constexpr std::size_t max_component_candidates=max_candidates/2;
      std::size_t component_candidates=0;
      // Reserve component room within the SAME fixed pool. Only an opaque
      // boundary region precedes complete components; parented/intermediate
      // Fold aliases are optional later candidates, not prerequisites merely
      // because a rewrite produced another stored Fold head.
      const auto early_fold_cap=(max_candidates-candidate_entries)/2;
      std::size_t early_fold_candidates=0;
      const auto appendBoundaries=[&](bool folded) {
        for(auto id:boundary_heads) {
          charge();const auto owner=owners[id];
          if(candidate_entries>=max_candidates || component_candidates>=max_component_candidates)break;
          if(has_fold_[owner]!=folded)continue;
          const bool fold_region=(ordinary_folds.contains(owner) && ordinary_folds.at(owner)==id) ||
              (seeded_folds.contains(owner) && seeded_folds.at(owner)==id);
          if(fold_region) {
            if(!folded || early_fold_candidates>=early_fold_cap)continue;
            const auto before=candidates.size();append(id);
            if(candidates.size()!=before)++early_fold_candidates;
          } else if(selected.contains(owner) && selected.at(owner)==id &&
              indexedLinearHead(snapshot[id]) && !immediateConstantHead(owner).value_or(false)) {
            const auto before=candidates.size();append(id);
            if(candidates.size()!=before)++component_candidates;
          }
        }
      };
      const auto appendComponents=[&](bool folded) {
        for(auto id:catalog.schedule_ids) {
          charge();const auto owner=owners[id];
          if(candidate_entries>=max_candidates || component_candidates>=max_component_candidates)break;
          // A selected linear parent can summarize this region as one atom.
          // Its opaque/nonlinear boundary, however, may need a whole-region
          // equality before a complete output can match by congruence.
          if(folded && ((ordinary_folds.contains(owner) && ordinary_folds.at(owner)==id) ||
              (seeded_folds.contains(owner) && seeded_folds.at(owner)==id)) &&
              (!linear_parented.contains(owner) || linear_boundaries.contains(owner)) &&
              early_fold_candidates<early_fold_cap) {
            const auto before=candidates.size();append(id);
            if(candidates.size()!=before)++early_fold_candidates;
            continue;
          }
          if(selected.contains(owner) && selected.at(owner)==id &&
              has_fold_[owner]==folded &&
              (!linear_parented.contains(owner) || linear_boundaries.contains(owner)) &&
              indexedLinearHead(snapshot[id]) && !immediateConstantHead(owner).value_or(false)) {
            const auto before=candidates.size();append(id);
            if(candidates.size()!=before)++component_candidates;
          }
        }
      };
      appendBoundaries(true);
      appendComponents(true);
      if(extrema){appendBoundaries(false);appendComponents(false);}
      const auto priority_candidates=candidates.size();
      // Collect the SAME complete candidate window before publishing any
      // coordinate key. Only proof execution is deferred past moments.
      phase="scalar_linear_components";
      if(!extrema){appendBoundaries(false);appendComponents(false);}
      phase="fold_alias_fallback";
      // Retain the original region alternatives after complete components
      // (and Sum's optional moments/scalar components). Heterogeneous parents
      // can still need an interior Fold equality; it is never dropped as a
      // semantic assertion. All categories share the original total pool.
      for(auto id:catalog.fold_ids) {
        if(candidate_entries>=max_candidates)break;
        charge();const auto owner=owners[id];
        if((ordinary_folds.contains(owner) && ordinary_folds.at(owner)==id) ||
            (seeded_folds.contains(owner) && seeded_folds.at(owner)==id))append(id);
      }
      phase="linear_prefix_fallback";
      // Heterogeneous parents need not have a coverage key even when an
      // interior component does. Retain the old bounded prefix fallback LAST;
      // neither this scan nor a cap adds an equality.
      if(!extrema)for(auto id:catalog.schedule_ids) {
        charge();if(selected.contains(owners[id]) && selected.at(owners[id])==id &&
            !immediateConstantHead(owners[id]).value_or(false))append(id);
      }

      scheduling_end=remaining;
      if(trace) {
        // Scalar census of the already sealed, capped candidate window. Do
        // not select another witness, render expressions or traverse bodies.
        SENTRYWARD_DEBUG_PRINT(egraph_8);
        if(!left.empty())SENTRYWARD_DEBUG_PRINT(egraph_9);else SENTRYWARD_DEBUG_PRINT(egraph_10);
        SENTRYWARD_DEBUG_PRINT(egraph_11);if(!right.empty())SENTRYWARD_DEBUG_PRINT(egraph_12);else SENTRYWARD_DEBUG_PRINT(egraph_13);
        SENTRYWARD_DEBUG_PRINT(egraph_14);
        for(auto cls:candidates) {
          cls=owners[cls];Class head=cls;
          if(auto f=ordinary_folds.find(cls);f!=ordinary_folds.end())head=f->second;
          else if(auto f=seeded_folds.find(cls);f!=seeded_folds.end())head=f->second;
          else if(auto h=selected.find(cls);h!=selected.end())head=h->second;
          SENTRYWARD_DEBUG_PRINT(egraph_15);
          if(head<n && snapshot[head].fold)SENTRYWARD_DEBUG_PRINT(egraph_16);
          SENTRYWARD_DEBUG_PRINT(egraph_17);
        }
        SENTRYWARD_DEBUG_PRINT(egraph_18);
      }
      phase="target_frontier";
      // ONE union-of-candidate selected-DAG frontier. An ordinary Fold retains
      // its ACTUAL template even if its class also has a selected linear alias;
      // this does not choose the summary's atom or assert a body equality.
      // A seeded Fold retains its original seed once and its opaque body-only
      // boundary. No Fold BODY/logical instance is traversed.
      // Nonlinear atoms stop here. Their independently queued Fold/linear
      // children remain roots in the unchanged boundary/component schedule.
      std::unordered_set<Class> target_frontier_seen,target_frontier_owners;
      std::vector<Class> target_frontier_pending;
      charge(candidates.size());target_frontier_pending=candidates;
      const auto frontierPush=[&](Class cls) {
        charge();cls=owners[cls];
        if(!target_frontier_seen.contains(cls)) {
          if(target_frontier_pending.size()>=max_retained_summary_nodes)
            localCap("target_frontier_active",target_frontier_pending.size());
          charge(3);target_frontier_pending.push_back(cls);
        }
      };
      while(!target_frontier_pending.empty()) {
        charge();const auto cls=owners[target_frontier_pending.back()];target_frontier_pending.pop_back();
        if(target_frontier_seen.contains(cls))continue;
        if(target_frontier_seen.size()>=max_retained_summary_nodes)
          localCap("target_frontier_nodes",target_frontier_seen.size());
        charge(3);target_frontier_seen.insert(cls);++frontier_nodes;
        if(!admitted[cls] || literals.contains(cls))continue;
        if(auto f=ordinary_folds.find(cls);f!=ordinary_folds.end()) {
          charge(3);target_frontier_owners.insert(cls);
        }
        if(auto f=seeded_folds.find(cls);f!=seeded_folds.end() && !ordinary_folds.contains(cls)) {
          const auto& k=snapshot[f->second];
          // Extrema retain an actual non-neutral seed in the complete form.
          // Sum's seed/body split is exactly the existing DagSession resolver.
          if(extrema){charge(3);target_frontier_owners.insert(cls);}
          frontierPush(k.children[0]);
          // A certified Add(seed, neutral-Fold) alias can provide the ACTUAL
          // body-only target template. The summary may prefer its private
          // seeded atom, but target discovery must retain that stored alias.
          // Follow only ranked selected linear children, never the Fold BODY.
          if(auto alias=selected.find(cls);alias!=selected.end() && linearHead(snapshot[alias->second]))
            for(auto child:snapshot[alias->second].children){++frontier_edges;frontierPush(child);}
          continue;
        }
        if(auto f=selected.find(cls);f!=selected.end() && linearHead(snapshot[f->second])) {
          for(auto child:snapshot[f->second].children){++frontier_edges;frontierPush(child);}
        }
      }
      // IDs were already admitted in deterministic ascending snapshot order.
      // This scalar filter never reopens a denied target or whole Fold body.
      std::vector<Class> frontier_fold_ids;
      for(auto id:potential_target_ids) {
        charge();if(target_frontier_owners.contains(owners[id])) {
          charge(3);frontier_fold_ids.push_back(id);
        }
      }
      frontier_targets=frontier_fold_ids.size();
      phase="group_shapes";
      // Metadata-sized target scan; the selected Fold body remains unchanged.
      for(auto id:frontier_fold_ids) {
        charge();const auto owner=owners[id];
        const bool ordinary=ordinary_folds.contains(owner) && ordinary_folds.at(owner)==id;
        const bool seeded=extrema && seeded_folds.contains(owner) && seeded_folds.at(owner)==id;
        if((!ordinary && !seeded) || !snapshot[id].fold->count)continue;
        const auto body=owners[snapshot[id].children[1]];
        // A coordinate-coverage target needs an observable reduction axis.
        // Fixed-point absence (or a certified literal) proves this body is
        // invariant; it cannot supply an affine interval origin. This is only
        // target admission: the Fold and its normal algebra remain unchanged.
        if(!extrema && (literals.contains(body) ||
            !dependsOn(body,snapshot[id].fold->binder_level)))continue;
        auto scope=scopeFor(owner);if(!scope)continue;
        auto body_shape=shape(snapshot[id].children[1],0);if(!body_shape)continue;
        const auto body_head=selected.contains(body)?selected.at(body):
            members[body].empty()?body:members[body].front();
        body_heads.emplace(snapshot[body_head].kind,snapshot[body_head].name);
        const BucketKey bucket{snapshot[id].type,*scope,*body_shape,reducer};
        charge(snapshot[id].type.size()+scope->size()+32);
        if(group_entries>=max_groups && !partitions.contains(bucket))continue;
        auto& partition=partitions[bucket];
        auto anchor=targetAnchor(snapshot[id]);
        if(!anchor)partition.unsupported=true;
        std::optional<AnchorInterval> interval;
        const GroupKey* component=nullptr;
        std::map<std::uint64_t,std::optional<Prototype>>* prototypes=nullptr;
        if(anchor) {
          charge(4*(std::get<3>(anchor->first).size()+8)*(1+std::bit_width(partition.prototypes.size()+1)));
          if(group_entries>=max_groups && !partition.prototypes.contains(anchor->first))continue;
          auto [family,new_family]=partition.prototypes.try_emplace(std::move(anchor->first));
          (void)new_family;interval=anchor->second;interval->family=&family->first;
          prototypes=&family->second;partition.anchored=true;
          // ONE deterministic nearest compatible prototype and ONE oriented
          // full proof. Failed/ambiguous/exhausted proofs create no equality;
          // no second prototype or opposite orientation is tried.
          charge(4*(1+std::bit_width(prototypes->size()+1)));
          auto nearest=prototypes->lower_bound(interval->begin);
          if(nearest!=prototypes->begin()) {
            auto lower=std::prev(nearest);
            if(nearest==prototypes->end() || interval->begin-lower->first<=nearest->first-interval->begin)nearest=lower;
          }
          if(nearest!=prototypes->end() && nearest->second && !body_heads_index.exhausted) {
            const auto& prototype=*nearest->second;
            const auto& p=snapshot[prototype.fold];const auto& c=snapshot[id];
            const bool candidate_lower=interval->begin<nearest->first;
            const auto expected=candidate_lower?nearest->first-interval->begin:interval->begin-nearest->first;
            bool related=false;
            if(!expected && sameBoundFoldBody(p,c)) {related=true;if(trace)++origin_identity;}
            else {
              auto map=tracedMap(origin_trace,[&] {
                return affineBodyMapWithHeads(candidate_lower?c.children[1]:p.children[1],
                    candidate_lower?p.children[1]:c.children[1],
                    candidate_lower?c.fold->binder_level:p.fold->binder_level,
                    snapshot,false,&remaining,body_heads_index,
                    candidate_lower?p.fold->binder_level:c.fold->binder_level);
              });
              related=map && map->first==1 && map->second==expected;
            }
            if(!remaining)localCap("optional_work",initial_work);
            if(related)component=prototype.key;
          }
        }
        GroupKey key=component?*component:GroupKey{snapshot[id].type,*scope,*body_shape,reducer,
            (anchor || partition.anchored)?id+1:0};
        if(group_entries>=max_groups && !groups.contains(key))continue;
        auto [entry,inserted]=groups.try_emplace(std::move(key),Group{id,{},interval});
        if(inserted)++group_entries;
        if(inserted)++partition.components;
        if(!inserted && interval) {
          if(entry->second.anchor && interval->family==entry->second.anchor->family &&
              interval->begin<entry->second.anchor->begin) {
            // The one complete origin proof above certified this component.
            // Changing its canonical origin is scheduling only; EVERY atom
            // will still be fully mapped to the final canonical body below.
            entry->second.fold=id;entry->second.anchor=interval;
          }
        } else if(!inserted && !partition.anchored) {
          // A WHOLLY unsupported anchor bucket retains the original path.
          // Its token0 never joins an anchored component by a wildcard hint.
          const auto& current=snapshot[entry->second.fold];const auto& candidate=snapshot[id];
          if(candidate.fold->binder_level==current.fold->binder_level) {
            if(sameBoundFoldBody(candidate,current)) {
              if(trace)++origin_identity;
            } else {
            // Choose a lower coordinate origin only after proving the ENTIRE
            // current body is candidate(i+b), with unit stride and b>0.
            // Node/visit order, allocation names and a single address leaf do
            // not establish this direction. One check per stored Fold body;
            // shared shape/affine caches and the aggregate work cap bound it.
            auto lower=tracedMap(origin_trace,[&] {
              return affineBodyMap(candidate.children[1],current.children[1],
                  candidate.fold->binder_level,snapshot,members,false,&remaining,&affine_cache,&body_heads_index);
            });
            if(!remaining)localCap("optional_work",initial_work);
            if(lower && lower->first==1 && lower->second>0)entry->second.fold=id;
            }
          }
        }
        if(prototypes && (fold_memberships.contains(owner) || fold_memberships.size()<max_retained_summary_nodes)) {
          charge(8*(1+std::bit_width(prototypes->size()+1)));
          if(prototypes->contains(interval->begin) || prototype_entries<max_retained_summary_nodes) {
            auto [record,new_record]=prototypes->try_emplace(interval->begin,Prototype{id,&entry->first});
            if(new_record)++prototype_entries;
            else if(record->second && record->second->key!=&entry->first)record->second.reset();
          }
        }
        // O(1) owner/header records use the SAME physical retained capacity,
        // not another copied group key or ancestor walk.
        if(fold_memberships.contains(owner) || fold_memberships.size()<max_retained_summary_nodes) {
          charge(32);
          auto [member,new_member]=fold_memberships.try_emplace(owner,FoldMembership{id,&entry->first,interval});
          if(new_member){charge(3);membership_owners.push_back(owner);}
          if(!new_member && member->second &&
              (member->second->head!=id || member->second->key!=&entry->first))member->second.reset();
        }
      }
      groups_end=remaining;
      if(sentryward::debug::environment("SENTRYWARD_EGRAPH_TRACE")) {
#if defined(SENTRYWARD_DEVELOPMENT_DIAGNOSTICS) || defined(SENTRYWARD_ENABLE_COMPRESSION_AUDIT)
#include "sentryward/development/blocks/egraph_trace_1.inc"
#endif
}

      if(groups.empty())return;
      std::array<std::unordered_map<Class,std::optional<CoverageKey>>,2> coverage_caches;
      const auto reductionFold=[&](l::Atom atom)->std::optional<Class> {
        if(atom>=n) {
          const auto cls=static_cast<Class>(atom-n);
          if(auto found=seeded_folds.find(cls);found!=seeded_folds.end())return found->second;
          return {};
        }
        if(auto found=ordinary_folds.find(static_cast<Class>(atom));found!=ordinary_folds.end())return found->second;
        return {};
      };
      // The same full-body/scoped mapping factory serves every reducer. Shape
      // identity merely schedules this exact certificate; it is not a proof.
      const auto tracedDomainMap=[&](auto&& invoke) {
        BodyHeadIndex::FallbackTrace reports(body_heads_index,trace?&domain_fallback_reports:nullptr,
            trace?&domain_denial_reports:nullptr);
        return tracedMap(domain_trace,std::forward<decltype(invoke)>(invoke));
      };
      const auto mappedDomain=[&](Group& group,l::Atom atom,
          const std::vector<std::uint32_t>& scope,bool semantic=false) {
        if(trace)++domain_queries;
        auto& cache=group.mapped[semantic?1:0];
        auto mapped=cache.find(atom);
        if(mapped==cache.end()) {
          const auto& target=snapshot[group.fold];const auto binder=target.fold->binder_level;
          std::optional<std::pair<std::uint64_t,std::uint64_t>> domain;
          if(auto fold=reductionFold(atom)) {
            const auto& k=snapshot[*fold];auto atom_scope=semantic?lexicalFree(owners[*fold]):scopeFor(owners[*fold]);
            if(k.type==target.type && k.fold->reducer==target.fold->reducer && atom_scope && *atom_scope==scope) {
              if(k.fold->binder_level==binder) {
                if(sameBoundFoldBody(target,k)) {
                  if(trace)++domain_identity;
                  // Count is the ACTUAL source domain, never the target count.
                  // Seeded atoms represent body-only sums; their exact seed
                  // remains once in the existing residual/constant form.
                  domain=std::pair{std::uint64_t{0},k.fold->count};
                } else {
                  auto map=tracedDomainMap([&] {
                    return affineBodyMap(target.children[1],k.children[1],binder,snapshot,members,false,&remaining,&affine_cache,&body_heads_index);
                  });
                  if(map && map->first==1 && k.fold->count<=UINT64_MAX-map->second)domain=std::pair{map->second,map->second+k.fold->count};
                }
              } else {
                std::size_t rectangle_work=0;
                const std::vector<RectangleAxis> axes{{k.fold->binder_level,k.fold->count}};
                auto map=tracedDomainMap([&] {
                  auto result=rectangleBodyMap(target.children[1],k.children[1],binder,
                      axes,snapshot,members,affine_cache,rectangle_work);charge(rectangle_work);return result;
                });
                if(map && map->coefficients.size()==1 && map->coefficients[0]==1 &&
                    k.fold->count<=UINT64_MAX-map->begin)domain=std::pair{map->begin,map->begin+k.fold->count};
              }
            }
          } else if(atom<n) {
            auto atom_scope=semantic?lexicalFree(static_cast<Class>(atom)):scopeFor(static_cast<Class>(atom));
            if(atom_scope && !std::binary_search(atom_scope->begin(),atom_scope->end(),binder) &&
                snapshot[atom].type==target.type && *atom_scope==scope) {
              auto map=tracedDomainMap([&] {
                return affineBodyMap(target.children[1],static_cast<Class>(atom),binder,snapshot,members,true,&remaining,&affine_cache,&body_heads_index);
              });
              if(map && map->second!=UINT64_MAX)domain=std::pair{map->second,map->second+1};
            }
          }
          if(!remaining)localCap("optional_work",initial_work);
          if(mapping_entries>=max_mappings)localCap("affine_mappings",mapping_entries);++mapping_entries;
          mapped=cache.emplace(atom,domain).first;
        } else if(trace)++domain_hits;
        return mapped->second;
      };
      struct ExtremaForm {
        std::optional<GroupKey> group;
        std::vector<l::WeightedInterval> intervals;
        std::vector<l::Term> residual;
      };
      std::unordered_map<Class,std::optional<ExtremaForm>> extrema_summaries;
      std::unordered_map<Class,unsigned char> extrema_state;
      std::optional<l::Coefficient> unit,private_zero;
      if(extrema) {
        unit=builder.coefficient("1");private_zero=builder.coefficient("0");sync();
        if(!unit || !private_zero)localCap("private_coefficients");
      }
      const auto summarizeExtrema=[&](Class root)->const ExtremaForm* {
        struct Task {Class cls;bool ready;};std::vector<Task> pending{{owners[root],false}};
        while(!pending.empty()) {
          charge();auto [cls,ready]=pending.back();pending.pop_back();
          if(extrema_summaries.contains(cls))continue;
          if(!extrema_state.contains(cls)) {
            if(summary_entries>=max_retained_summary_nodes)localCap("summary_nodes",summary_entries);
            extrema_state.emplace(cls,0);++summary_entries;
          }
          std::optional<ExtremaForm> result;const auto fold_head=reductionFold(
              seeded_folds.contains(cls)?static_cast<l::Atom>(n+cls):cls);
          const auto head=selected.find(cls);
          // A complete matching body is one sample even if that body's head
          // itself is Maximum/Minimum. Never distribute the body operation.
          auto scope=scopeFor(cls);std::optional<GroupKey> body_group;
          const auto chosen=head!=selected.end()?head->second:members[cls].empty()?cls:members[cls].front();
          // Only bounded local signatures classify a complete body sample;
          // the shared typed/scoped mapping remains the equality certificate.
          const auto& chosen_name=snapshot[chosen].name;
          if(chosen_name.size()>4096)localCap("sample_head_metadata",chosen_name.size());
          charge(chosen_name.size());
          bool ambiguous=false;
          if(scope && fold_head)body_group=admittedFoldGroup(*fold_head,snapshot[cls].type,*scope);
          if(!body_group && scope && (fold_head || body_heads.contains({snapshot[chosen].kind,chosen_name})))
            body_group=candidateGroup(fold_head?snapshot[*fold_head].children[1]:cls,
                snapshot[cls].type,*scope,false,&ambiguous);
          if(ambiguous) {extrema_summaries.emplace(cls,std::nullopt);continue;}
          if(scope && body_group && admitted[cls]) {
            if(auto group=groups.find(*body_group);group!=groups.end()) {
              auto domain=mappedDomain(group->second,fold_head && seeded_folds.contains(cls)?n+cls:cls,*scope);
              if(domain)result=ExtremaForm{*body_group,{{domain->first,domain->second,*unit}},{}};
              else {extrema_summaries.emplace(cls,std::nullopt);continue;}
            }
          }
          const bool seeded=fold_head && seeded_folds.contains(cls);
          const bool combine=!result && head!=selected.end();
          if(seeded || combine) {
            const auto& k=snapshot[seeded?*fold_head:head->second];
            if(!ready) {
              if(extrema_state.at(cls)==1)localCap("summary_cycle",summary_entries);
              extrema_state[cls]=1;pending.push_back({cls,true});
              if(seeded)pending.push_back({owners[k.children[0]],false});
              else for(auto c:k.children)pending.push_back({owners[c],false});
              continue;
            }
            if(seeded && !result) {extrema_summaries.emplace(cls,std::nullopt);continue;}
            const auto merge=[&](const ExtremaForm& a,const ExtremaForm& b)->std::optional<ExtremaForm> {
              if(a.group && b.group && *a.group!=*b.group)return {};
              const auto pieces=a.intervals.size()+b.intervals.size();
              const auto atoms=a.residual.size()+b.residual.size();charge(pieces+atoms);
              if(pieces>limits.max_intervals || atoms>max_cached_terms-cached_terms)
                localCap("summary_terms",cached_terms);
              std::vector<l::WeightedInterval> intervals=a.intervals;
              intervals.insert(intervals.end(),b.intervals.begin(),b.intervals.end());
              auto canonical=builder.canonicalIntervals(intervals);sync();if(!canonical)return {};
              std::vector<l::Term> residual;residual.reserve(atoms);
              std::merge(a.residual.begin(),a.residual.end(),b.residual.begin(),b.residual.end(),
                  std::back_inserter(residual),[](const auto& x,const auto& y){return x.atom<y.atom;});
              return ExtremaForm{a.group?a.group:b.group,std::move(*canonical),std::move(residual)};
            };
            auto a=extrema_summaries.find(owners[k.children[0]]);
            if(seeded) {
              if(a!=extrema_summaries.end() && a->second)result=merge(*result,*a->second);else result.reset();
            } else {
              auto b=extrema_summaries.find(owners[k.children[1]]);
              if(a!=extrema_summaries.end() && a->second && b!=extrema_summaries.end() && b->second) {
                // A finite nonnegative real makes Max(0,x)'s zero inactive.
                // This is not a NaN/IEEE rule or an arbitrary extrema weight.
                if(reducer==s::Reducer::Max && zero_[owners[k.children[0]]] && nonnegative_[owners[k.children[1]]])result=*b->second;
                else if(reducer==s::Reducer::Max && zero_[owners[k.children[1]]] && nonnegative_[owners[k.children[0]]])result=*a->second;
                else result=merge(*a->second,*b->second);
              }
            }
          } else if(!result && admitted[cls])result=ExtremaForm{{},{},{{cls,*unit}}};
          if(result) {
            const auto pieces=result->intervals.size(),atoms=result->residual.size();
            if(pieces+atoms>max_cached_terms-cached_terms)localCap("summary_terms",cached_terms);
            if(pieces>max_interval_entries-interval_entries)localCap("coverage_intervals",interval_entries);
            if(atoms>max_cached_terms-residual_entries)localCap("coverage_residuals",residual_entries);
            cached_terms+=pieces+atoms;interval_entries+=pieces;residual_entries+=atoms;
          }
          extrema_summaries.emplace(cls,std::move(result));extrema_state[cls]=2;
        }
        const auto& result=extrema_summaries.at(owners[root]);return result?&*result:nullptr;
      };
      const auto coverage=[&](Class candidate,bool semantic=false)->const CoverageKey* {
        candidate=owners[candidate];charge();
        auto& coverage_cache=coverage_caches[semantic?1:0];
        if(auto cached=coverage_cache.find(candidate);cached!=coverage_cache.end())return cached->second?&*cached->second:nullptr;
        const std::vector<std::uint32_t>* diagnostic_scope=nullptr;
        const auto fail=[&](const char* reason,l::Atom atom=std::numeric_limits<l::Atom>::max(),
            std::size_t terms=0)->const CoverageKey* {
          if(trace && coverage_denial_reports) {
            --coverage_denial_reports;
            SENTRYWARD_DEBUG_PRINT(egraph_24);
            if(!left.empty())SENTRYWARD_DEBUG_PRINT(egraph_25);else SENTRYWARD_DEBUG_PRINT(egraph_26);
            SENTRYWARD_DEBUG_PRINT(egraph_27);if(!right.empty())SENTRYWARD_DEBUG_PRINT(egraph_28);else SENTRYWARD_DEBUG_PRINT(egraph_29);
            SENTRYWARD_DEBUG_PRINT(egraph_30);
            if(atom!=std::numeric_limits<l::Atom>::max())SENTRYWARD_DEBUG_PRINT(egraph_31);else SENTRYWARD_DEBUG_PRINT(egraph_32);
            SENTRYWARD_DEBUG_PRINT(egraph_33);
            if(diagnostic_scope) {
              for(std::size_t i=0;i<std::min<std::size_t>(diagnostic_scope->size(),8);++i)
                SENTRYWARD_DEBUG_PRINT(egraph_34);
              if(diagnostic_scope->size()>8)SENTRYWARD_DEBUG_PRINT(egraph_35);
            } else SENTRYWARD_DEBUG_PRINT(egraph_36);
            const auto text=[&](std::string_view value) {
              SENTRYWARD_DEBUG_PRINT(egraph_37);constexpr char hex[]="0123456789abcdef";
              const auto size=std::min<std::size_t>(value.size(),128);
              for(std::size_t i=0;i<size;++i) {
                const auto c=static_cast<unsigned char>(value[i]);
                if(c=='"' || c=='\\')SENTRYWARD_DEBUG_PRINT(egraph_38);
                else if(c<32 || c==127)SENTRYWARD_DEBUG_PRINT(egraph_39);
                else SENTRYWARD_DEBUG_PRINT(egraph_40);
              }
              SENTRYWARD_DEBUG_PRINT(egraph_41);if(size<value.size())SENTRYWARD_DEBUG_PRINT(egraph_42);
            };
            const auto catalogHead=[&](Class id) {
              if(id>=n)return id;const auto cls=owners[id];
              if(auto f=ordinary_folds.find(cls);f!=ordinary_folds.end())return f->second;
              if(auto f=seeded_folds.find(cls);f!=seeded_folds.end())return f->second;
              if(auto h=selected.find(cls);h!=selected.end())return h->second;
              return cls;
            };
            const auto plain=[&](Class id) {
              SENTRYWARD_DEBUG_PRINT(egraph_43);if(id>=n){SENTRYWARD_DEBUG_PRINT(egraph_44);return;}
              const auto& k=snapshot[id];SENTRYWARD_DEBUG_PRINT(egraph_45);text(k.name);
              SENTRYWARD_DEBUG_PRINT(egraph_46);text(k.type);SENTRYWARD_DEBUG_PRINT(egraph_47);text(k.attributes);
              SENTRYWARD_DEBUG_PRINT(egraph_48);text(k.provenance);SENTRYWARD_DEBUG_PRINT(egraph_49);
              for(std::size_t i=0;i<std::min<std::size_t>(k.children.size(),4);++i)SENTRYWARD_DEBUG_PRINT(egraph_50);
              if(k.children.size()>4)SENTRYWARD_DEBUG_PRINT(egraph_51);
              if(k.fold)SENTRYWARD_DEBUG_PRINT(egraph_52);
              SENTRYWARD_DEBUG_PRINT(egraph_53);
            };
            // At most FOUR fixed-local headers. Catalog hints are labelled
            // diagnostics only; they are not selected by the full body proof.
            auto fold=atom!=std::numeric_limits<l::Atom>::max()?reductionFold(atom):std::nullopt;
            const auto head=fold?*fold:catalogHead(atom<n?static_cast<Class>(atom):candidate);
            SENTRYWARD_DEBUG_PRINT(egraph_54);plain(head);
            if(head<n && snapshot[head].fold && snapshot[head].children.size()==2) {
              const auto body=catalogHead(snapshot[head].children[1]);plain(body);
              if(body<n)for(std::size_t i=0;i<std::min<std::size_t>(snapshot[body].children.size(),2);++i)
                plain(catalogHead(snapshot[body].children[i]));
            } else if(head<n)for(std::size_t i=0;i<std::min<std::size_t>(snapshot[head].children.size(),3);++i)
              plain(catalogHead(snapshot[head].children[i]));
            SENTRYWARD_DEBUG_PRINT(egraph_55);
          }
          coverage_cache.emplace(candidate,std::nullopt);++coverage_entries;return nullptr;
        };
        if(coverage_entries>=max_retained_summary_nodes)localCap("coverage_nodes",coverage_entries);
        if(!admitted[candidate])return fail("not_admitted");
        auto scope=semantic?lexicalFree(candidate):scopeFor(candidate);if(!scope)return fail("scope_unknown");
        diagnostic_scope=&*scope;
        if(semantic)exactGroups();
        const auto& scopes=semantic?exact_group_scopes:group_scopes;
        charge();if(!scopes.contains({snapshot[candidate].type,*scope}))return fail("scope_unindexed");
        if(extrema) {
          const auto* form=summarizeExtrema(candidate);
          if(!form || !form->group || form->intervals.empty() ||
              std::get<0>(*form->group)!=snapshot[candidate].type || std::get<1>(*form->group)!=*scope)return fail("extrema_form");
          if(form->intervals.size()>max_interval_entries-interval_entries)localCap("coverage_intervals",interval_entries);
          if(form->residual.size()>max_cached_terms-residual_entries)localCap("coverage_residuals",residual_entries);
          interval_entries+=form->intervals.size();residual_entries+=form->residual.size();
          CoverageKey key{*form->group,*private_zero,form->intervals,form->residual};
          charge(form->intervals.size()+form->residual.size());++coverage_entries;
          auto entry=coverage_cache.emplace(candidate,std::move(key)).first;return &*entry->second;
        }
        auto result=summarize(candidate);if(!result || result->terms().empty())return fail("empty_summary");
        const auto& form=*result;
        std::optional<GroupKey> group_key;bool valid=true;
        const char* denial="no_group";l::Atom denied_atom=std::numeric_limits<l::Atom>::max();
        std::vector<bool> reduction_terms;reduction_terms.reserve(form.terms().size());
        for(const auto& term:form.terms()) {
          charge();
          const auto atom=static_cast<Class>(term.atom);
          const auto fold=reductionFold(term.atom);
          denied_atom=term.atom;
          if(!fold && (atom>=n || !admitted[atom])){denial="invalid_atom";valid=false;break;}
          bool ambiguous=false;
          auto key=fold?admittedFoldGroup(*fold,snapshot[candidate].type,*scope,semantic):std::nullopt;
          if(!key)key=candidateGroup(fold?snapshot[*fold].children[1]:atom,
              snapshot[candidate].type,*scope,semantic,&ambiguous);
          if(ambiguous){denial="ambiguous_group";valid=false;break;}
          const bool reduction=key.has_value();
          reduction_terms.push_back(reduction);
          if(!reduction)continue;
          // Two different admitted Fold-body groups are ambiguous here. Do
          // not guess an origin, demote a conflicting mapped group to an atom,
          // or enumerate root/target combinations.
          if(group_key && *group_key!=*key){denial="conflicting_groups";valid=false;break;}group_key=std::move(*key);
        }
        if(!valid || !group_key)return fail(denial,denied_atom,form.terms().size());
        auto& group=groups.at(*group_key);
        std::vector<l::WeightedInterval> intervals;intervals.reserve(form.terms().size());
        std::vector<l::Term> residual;residual.reserve(form.terms().size());
        for(std::size_t i=0;i<form.terms().size();++i) {
          charge();const auto& term=form.terms()[i];
          if(!reduction_terms[i]) {
            // Builder terms already have sorted distinct atom IDs, exact
            // bounded rational weights, and certified finite source domains.
            // Equality compares this entire residual multiset verbatim.
            charge(128);residual.push_back(term);continue;
          }
          auto domain=mappedDomain(group,term.atom,*scope,semantic);
          if(!domain){denied_atom=term.atom;valid=false;break;}
          intervals.push_back({domain->first,domain->second,term.weight});
        }
        if(!valid)return fail("domain_unknown",denied_atom,form.terms().size());
        auto canonical=builder.canonicalIntervals(intervals);sync();if(!canonical)return fail("interval_canonicalization",denied_atom,form.terms().size());
        if(canonical->size()>max_interval_entries-interval_entries)localCap("coverage_intervals",interval_entries);
        interval_entries+=canonical->size();
        if(residual.size()>max_cached_terms-residual_entries)localCap("coverage_residuals",residual_entries);
        residual_entries+=residual.size();
        CoverageKey key{*group_key,form.constant(),std::move(*canonical),std::move(residual)};
        ++coverage_entries;auto entry=coverage_cache.emplace(candidate,std::move(key)).first;return &*entry->second;
      };
      const auto indexCandidate=[&](Class candidate) {
        auto key=coverage(candidate);if(!key)return;
        if(signature_entries>=max_signatures)localCap("coverage_signatures",signature_entries);
        charge(128*(key->intervals.size()+key->residual.size()+1));
        auto [entry,inserted]=signatures.emplace(*key,candidate);
        if(inserted)++signature_entries;
        if(!inserted)equalities.emplace_back(candidate,entry->second);
      };
      phase="linear_prerequisites";
      // Complete interior certificates unblock many outputs in the next
      // congruence epoch. Give them one existing half-pool share before roots,
      // not an unbounded ancestor query or a fresh proof-work allowance.
      // When there are no admitted whole roots, there is no root work to defer.
      const auto prerequisite_end=root_candidates?std::min(priority_candidates,
          root_candidates+max_component_candidates):priority_candidates;
      for(std::size_t i=root_candidates;i<prerequisite_end;++i)indexCandidate(candidates[i]);
      phase="linear_candidates";
      for(std::size_t i=0;i<root_candidates;++i)indexCandidate(candidates[i]);
      for(std::size_t i=prerequisite_end;i<priority_candidates;++i)indexCandidate(candidates[i]);
      if(!extrema)try {
      phase="moment_setup";
      // Exact moments use the SAME affine linear-mean coverage keys above.
      // Local quadratic merge identities telescope over those private keys;
      // no sample cross-products, Fold members or new enodes are constructed.
      namespace m=moment_certificate;
      m::Limits moment_limits;moment_limits.max_work=std::min<std::size_t>(1000000,remaining);
      m::Builder moments(moment_limits);std::size_t charged_moments=0;
      const auto momentSync=[&] {
        const auto delta=moments.work()-charged_moments;charged_moments=moments.work();charge(delta);
        if(moments.exhausted())localCap("moment_builder",moments.work());
      };
      std::map<GroupKey,std::size_t> group_tokens;
      for(const auto& [key,value]:groups){(void)value;group_tokens.emplace(key,group_tokens.size());}
      struct MeanRecord {m::MeanId id;const Group* group;const CoverageKey* coverage;};
      std::unordered_map<Class,std::optional<MeanRecord>> mean_records;
      const auto mean=[&](Class cls)->std::optional<MeanRecord> {
        cls=owners[cls];charge();if(auto found=mean_records.find(cls);found!=mean_records.end())return found->second;
        if(mean_records.size()>=max_retained_summary_nodes)localCap("moment_means",mean_records.size());
        std::optional<MeanRecord> result;
        if(auto key=coverage(cls,true);key && key->residual.empty()) {
          auto id=moments.mean({group_tokens.at(key->group),0,key->constant,key->intervals,finite_[cls]});
          momentSync();if(id)result=MeanRecord{*id,&groups.at(key->group),key};
        }
        mean_records.emplace(cls,result);return result;
      };
      struct FamilyMeanRecord {m::MeanId id;const Group* group;std::uint64_t step;};
      using FamilyQuery=std::tuple<Class,std::uint32_t,std::uint64_t>;
      std::map<FamilyQuery,std::optional<FamilyMeanRecord>> family_means;
      struct FamilyDomain {std::uint64_t begin,end,step;};
      using FamilyMapQuery=std::tuple<std::size_t,l::Atom,std::uint32_t,std::uint64_t>;
      std::map<FamilyMapQuery,std::optional<FamilyDomain>> family_mappings;
      const auto familyMean=[&](Class cls,std::uint32_t axis,std::uint64_t extent)->std::optional<FamilyMeanRecord> {
        cls=owners[cls];charge(48);const FamilyQuery query{cls,axis,extent};
        if(auto found=family_means.find(query);found!=family_means.end())return found->second;
        const auto fail=[&]()->std::optional<FamilyMeanRecord>{family_means.emplace(query,std::nullopt);return {};};
        if(family_means.size()>=max_retained_summary_nodes)localCap("moment_family_means",family_means.size());
        if(!extent || !admitted[cls])return fail();
        auto scope=lexicalFree(cls);if(!scope)return fail();
        scope->erase(std::remove(scope->begin(),scope->end(),axis),scope->end());
        exactGroups();charge();if(!exact_group_scopes.contains({snapshot[cls].type,*scope}))return fail();
        auto result=summarize(cls);if(!result || result->terms().empty())return fail();
        const auto& form=*result;std::optional<GroupKey> group_key;
        for(const auto& term:form.terms()) {
          charge();const auto atom=term.atom;
          const auto fold=reductionFold(atom);
          if(!fold && (atom>=n || !admitted[atom]))return fail();
          auto key=fold?admittedFoldGroup(*fold,snapshot[cls].type,*scope,true):std::nullopt;
          if(!key)key=candidateGroup(fold?snapshot[*fold].children[1]:static_cast<Class>(atom),
              snapshot[cls].type,*scope,true);
          if(!key || (group_key && *group_key!=*key))return fail();group_key=std::move(*key);
        }
        if(!group_key)return fail();
        auto& group=groups.at(*group_key);const auto& target=snapshot[group.fold];
        const auto token=group_tokens.at(*group_key);std::optional<std::uint64_t> step;
        std::vector<l::WeightedInterval> intervals;intervals.reserve(form.terms().size());
        for(const auto& term:form.terms()) {
          charge(48);const auto atom=term.atom;const FamilyMapQuery key{token,atom,axis,extent};
          auto mapped=family_mappings.find(key);
          if(mapped==family_mappings.end()) {
            std::optional<FamilyDomain> domain;const auto fold=reductionFold(atom);
            const auto source=fold?*fold:static_cast<Class>(atom);
            if(source>=n)return fail();
            // A private seeded-region atom denotes its body ONLY. Its seed is
            // already a separate actual linear term; never index n+owner as a
            // source node or import the complete seeded Fold as a body point.
            const auto body=fold?snapshot[*fold].children[1]:source;
            auto atom_scope=lexicalFree(body);
            if(atom_scope && fold) {
              const auto inner=snapshot[*fold].fold->binder_level;
              atom_scope->erase(std::remove(atom_scope->begin(),atom_scope->end(),inner),atom_scope->end());
            }
            if(atom_scope)atom_scope->erase(std::remove(atom_scope->begin(),atom_scope->end(),axis),atom_scope->end());
            if(atom_scope && *atom_scope==*scope && snapshot[source].type==target.type) {
              std::vector<RectangleAxis> axes{{axis,extent}};std::uint64_t count=1;
              bool valid=true;
              if(fold) {
                const auto& f=snapshot[*fold];count=f.fold->count;
                if(!count || f.fold->binder_level<=axis)valid=false;
                else axes.push_back({f.fold->binder_level,count});
              }
              std::size_t rectangle_work=0;
              auto map=valid?rectangleBodyMap(target.children[1],body,target.fold->binder_level,
                  axes,snapshot,members,affine_cache,rectangle_work):std::nullopt;
              charge(rectangle_work);
              if(map && map->coefficients.size()==axes.size() &&
                  (axes.size()==1 || map->coefficients[1]==1) && count<=UINT64_MAX-map->begin &&
                  (!map->coefficients[0] || extent-1<=(UINT64_MAX-map->begin-count)/map->coefficients[0]))
                domain=FamilyDomain{map->begin,map->begin+count,map->coefficients[0]};
            }
            if(mapping_entries>=max_mappings)localCap("moment_family_mappings",mapping_entries);++mapping_entries;
            mapped=family_mappings.emplace(key,domain).first;
          }
          if(!mapped->second || (step && *step!=mapped->second->step))return fail();step=mapped->second->step;
          intervals.push_back({mapped->second->begin,mapped->second->end,term.weight});
        }
        auto canonical=builder.canonicalIntervals(intervals);sync();if(!canonical || !step)return fail();
        auto id=moments.mean({token,0,form.constant(),*canonical,finite_[cls]});momentSync();if(!id)return fail();
        FamilyMeanRecord record{*id,&group,*step};family_means.emplace(query,record);return record;
      };
      const auto constantForm=[&](Class cls)->std::optional<l::LinearForm> {
        cls=owners[cls];if(!constantHead(cls))return {};
        auto result=summarize(cls);
        if(result && result->terms().empty())return result;return {};
      };
      const auto combine=[&](l::Operation op,const l::LinearForm& a,const l::LinearForm& b) {
        auto result=builder.apply(op,std::array{&a,&b});sync();return result;
      };
      auto one=builder.literal("1");sync();if(!one)localCap("moment_one",builder.work());
      // Fixed epoch head index: one nonrecursive admitted representative per
      // pattern/owner, then O(1) primitive lookups. Add+Negate uses this unary
      // index; no member-pair search or repeated eclass member traversal.
      std::unordered_map<Class,Class> negates,squares;
      std::unordered_map<Class,std::pair<Class,Class>> differences;
      for(auto id:catalog.eligible_ids) {
        charge();const auto& k=snapshot[id];
        if(!ranked[id] || !finite_[owners[id]] || !eligible(k))continue;
        if(k.name=="Negate" && k.children.size()==1)negates.try_emplace(owners[id],owners[k.children[0]]);
        if(k.name=="Subtract" && k.children.size()==2)
          differences.try_emplace(owners[id],owners[k.children[0]],owners[k.children[1]]);
        if(k.name=="Multiply" && k.children.size()==2 && owners[k.children[0]]==owners[k.children[1]])
          squares.try_emplace(owners[id],owners[k.children[0]]);
      }
      for(auto id:catalog.eligible_ids) {
        charge();const auto& k=snapshot[id];
        if(!ranked[id] || !finite_[owners[id]] || !eligible(k) || k.name!="Add" || k.children.size()!=2)continue;
        for(unsigned side=0;side<2;++side)if(auto neg=negates.find(owners[k.children[side]]);neg!=negates.end()) {
          differences.try_emplace(owners[id],owners[k.children[1-side]],neg->second);break;
        }
      }
      const auto difference=[&](Class cls)->std::optional<std::pair<Class,Class>> {
        charge();auto found=differences.find(owners[cls]);if(found==differences.end())return {};return found->second;
      };
      const auto square=[&](Class cls)->std::optional<Class> {
        charge();auto found=squares.find(owners[cls]);if(found==squares.end())return {};return found->second;
      };
      enum class MomentKind {Zero,Correction,IndexedCorrection,Centered,Add,Subtract,Negate,Scale};
      struct MomentNode {
        MomentKind kind=MomentKind::Zero;std::vector<Class> children;
        std::vector<Class> consumed;
        l::LinearForm factor;std::optional<m::Builder::Correction> correction;
        m::Builder::Translation translation;
        std::optional<m::MeanId> center;std::vector<std::pair<std::uint64_t,std::uint64_t>> intervals;
        bool has_moment=false;
      };
      const auto primitive=[&](Class cls)->std::optional<MomentNode> {
        cls=owners[cls];if(!admitted[cls])return {};
        if(zero_[cls])return MomentNode{};
        if(auto fold=ordinary_folds.find(cls);fold!=ordinary_folds.end()) {
          const auto& f=snapshot[fold->second];if(!f.fold->count)return {};
          // A rectangular centered sum can cover separated blocks. Prove its
          // support against the independently certified mean's STORED ranges;
          // never expand the outer or inner reduction elements.
          if(zero_[owners[f.children[0]]]) {
            auto nested=ordinary_folds.find(owners[f.children[1]]);
            if(nested!=ordinary_folds.end()) {
              const auto& h=snapshot[nested->second];
              if(h.fold->count && h.fold->binder_level>f.fold->binder_level && zero_[owners[h.children[0]]])
                if(auto squared=square(h.children[1]))if(auto delta=difference(*squared))
                  for(unsigned side=0;side<2;++side) {
                    const auto center=side?delta->first:delta->second;
                    const auto sample=side?delta->second:delta->first;
                    auto c=mean(center);auto deps=lexicalFree(center);
                    if(!c || !deps || !finite_[sample] ||
                        std::binary_search(deps->begin(),deps->end(),f.fold->binder_level) ||
                        std::binary_search(deps->begin(),deps->end(),h.fold->binder_level))continue;
                    const auto& target=snapshot[c->group->fold];std::size_t rectangle_work=0;
                    const std::vector<RectangleAxis> axes{{f.fold->binder_level,f.fold->count},
                        {h.fold->binder_level,h.fold->count}};
                    auto map=rectangleBodyMap(target.children[1],sample,target.fold->binder_level,
                        axes,snapshot,members,affine_cache,rectangle_work);charge(rectangle_work);
                    if(!map || map->coefficients.size()!=2 || map->coefficients[1]!=1 ||
                        map->coefficients[0]<h.fold->count)continue;
                    const auto step=map->coefficients[0],width=h.fold->count;
                    const auto& support=c->coverage->intervals;
                    bool valid=true;std::uint64_t rows=0;
                    MomentNode out;out.kind=MomentKind::Centered;out.center=c->id;out.has_moment=true;
                    for(const auto& interval:support) {
                      charge();const auto begin=interval.begin,end=interval.end;
                      if(begin<map->begin || end<begin || (begin-map->begin)%step ||
                          (begin-map->begin)/step!=rows){valid=false;break;}
                      const auto length=end-begin;
                      if(!length || length%width || (step!=width && length!=width) ||
                          length/width>f.fold->count-rows){valid=false;break;}
                      rows+=length/width;out.intervals.emplace_back(begin,end);
                    }
                    if(valid && rows==f.fold->count)return out;
                  }
            }
          }
          if(auto squared=square(f.children[1]))if(auto delta=difference(*squared))
            for(unsigned side=0;side<2;++side) {
              auto c=mean(side?delta->first:delta->second);if(!c)continue;
              const auto sample=side?delta->second:delta->first;
              auto deps=lexicalFree(side?delta->first:delta->second);
              if(!deps || std::binary_search(deps->begin(),deps->end(),f.fold->binder_level) || !finite_[sample])continue;
              const auto& target=snapshot[c->group->fold];std::size_t rectangle_work=0;
              const std::vector<RectangleAxis> axes{{f.fold->binder_level,f.fold->count}};
              auto map=rectangleBodyMap(target.children[1],sample,target.fold->binder_level,
                  axes,snapshot,members,affine_cache,rectangle_work);charge(rectangle_work);
              if(!map || map->coefficients.size()!=1 || map->coefficients[0]!=1 ||
                  f.fold->count>UINT64_MAX-map->begin)continue;
              MomentNode out;out.kind=MomentKind::Centered;out.center=c->id;
              out.intervals.emplace_back(map->begin,map->begin+f.fold->count);out.has_moment=true;return out;
            }
        }
        if(auto fold=sum_folds.find(cls);fold!=sum_folds.end()) {
          const auto& f=snapshot[fold->second];const auto binder=f.fold->binder_level;
          auto seed_scope=lexicalFree(f.children[0]);
          if(f.fold->count && seed_scope && !std::binary_search(seed_scope->begin(),seed_scope->end(),binder)) {
            auto body=owners[f.children[1]];l::LinearForm coefficient=*one;
            for(unsigned depth=0;depth<8;++depth) {
              charge();if(auto squared=square(body))if(auto delta=difference(*squared)) {
                auto a=familyMean(delta->first,binder,f.fold->count),b=familyMean(delta->second,binder,f.fold->count);
                if(!a || !b || a->group!=b->group || a->step!=b->step)break;
                auto correction=moments.correction(a->id,b->id,coefficient.constant());momentSync();if(!correction)break;
                MomentNode out;out.kind=MomentKind::IndexedCorrection;out.correction=std::move(correction);
                out.translation={0,a->step,f.fold->count};out.children.push_back(owners[f.children[0]]);
                out.has_moment=true;return out;
              }
              auto head=selected.find(body);if(head==selected.end())break;
              const auto& k=snapshot[head->second];std::optional<l::LinearForm> factor;Class next=n;
              if(k.name=="Multiply")for(unsigned side=0;side<2 && !factor;++side)
                if(auto c=constantForm(k.children[side])){factor=std::move(c);next=owners[k.children[1-side]];}
              if(k.name=="Divide")if(auto c=constantForm(k.children[1])) {
                factor=combine(l::Operation::Divide,*one,*c);next=owners[k.children[0]];
              }
              if(!factor || next==n)break;
              auto product=combine(l::Operation::Multiply,coefficient,*factor);if(!product)break;
              coefficient=std::move(*product);body=next;
            }
          }
        }
        // Complete scalar coefficient chain, capped at eight immediate heads.
        // A bare or partially scaled square is NOT a Welford correction unless
        // its exact coefficient already satisfies the local count identity.
        auto current=cls;l::LinearForm coefficient=*one;std::vector<Class> consumed;
        for(unsigned depth=0;depth<8;++depth) {
          charge();if(auto squared=square(current))if(auto delta=difference(*squared)) {
            auto a=mean(delta->first),b=mean(delta->second);if(!a || !b)return {};
            auto correction=moments.correction(a->id,b->id,coefficient.constant());momentSync();
            if(!correction)return {};
            MomentNode out;out.kind=MomentKind::Correction;out.correction=std::move(correction);
            out.consumed=std::move(consumed);out.has_moment=true;return out;
          }
          auto head=selected.find(current);if(head==selected.end())break;
          const auto& k=snapshot[head->second];std::optional<l::LinearForm> factor;Class next=n;
          if(k.name=="Multiply")for(unsigned side=0;side<2 && !factor;++side)
            if(auto c=constantForm(k.children[side])){factor=std::move(c);next=owners[k.children[1-side]];}
          if(k.name=="Divide")if(auto c=constantForm(k.children[1])) {
            factor=combine(l::Operation::Divide,*one,*c);next=owners[k.children[0]];
          }
          if(!factor || next==n)break;
          auto product=combine(l::Operation::Multiply,coefficient,*factor);if(!product)break;
          coefficient=std::move(*product);current=next;consumed.push_back(current);
        }
        return {};
      };
      std::unordered_map<Class,std::optional<MomentNode>> plans;
      std::unordered_map<Class,MomentNode> drafts;
      std::vector<Class> order;std::unordered_set<Class> active;
      const auto plan=[&](Class root) {
        struct Task {Class cls;bool ready;};std::vector<Task> pending{{owners[root],false}};
        while(!pending.empty()) {
          charge();auto [cls,ready]=pending.back();pending.pop_back();cls=owners[cls];
          if(plans.contains(cls))continue;
          if(plans.size()+active.size()>=max_retained_summary_nodes)
            localCap("moment_plans",plans.size()+active.size());
          if(active.size()>=max_active_nodes)localCap("moment_active",active.size());
          if(!ready) {
            if(!active.insert(cls).second)localCap("moment_cycle",active.size());
            if(auto p=primitive(cls)) {
              if(p->children.empty()){plans.emplace(cls,std::move(p));order.push_back(cls);active.erase(cls);continue;}
              auto draft=drafts.emplace(cls,std::move(*p)).first;
              pending.push_back({cls,true});
              for(auto child:draft->second.children){charge();if(!plans.contains(child))pending.push_back({child,false});}
              continue;
            }
            auto head=selected.find(cls);
            if(!admitted[cls] || head==selected.end()) {plans.emplace(cls,std::nullopt);active.erase(cls);continue;}
            const auto& k=snapshot[head->second];MomentNode p;bool valid=true;
            if(k.name=="Add" || k.name=="Subtract" || k.name=="Negate") {
              p.kind=k.name=="Add"?MomentKind::Add:k.name=="Subtract"?MomentKind::Subtract:MomentKind::Negate;
              for(auto c:k.children)p.children.push_back(owners[c]);
            } else if(k.name=="Multiply" || k.name=="Divide") {
              std::optional<l::LinearForm> factor;Class child=n;
              if(k.name=="Multiply")for(unsigned side=0;side<2 && !factor;++side)
                if(auto c=constantForm(k.children[side])){factor=std::move(c);child=owners[k.children[1-side]];}
              if(k.name=="Divide")if(auto c=constantForm(k.children[1])) {
                factor=combine(l::Operation::Divide,*one,*c);child=owners[k.children[0]];
              }
              if(!factor || child==n)valid=false;
              else {p.kind=MomentKind::Scale;p.factor=std::move(*factor);p.children.push_back(child);}
            } else valid=false;
            if(!valid) {plans.emplace(cls,std::nullopt);active.erase(cls);continue;}
            // Keep pending internal records separate until all children are
            // validated; no partially completed component is a certificate.
            pending.push_back({cls,true});
            for(auto child:p.children)if(!plans.contains(child))pending.push_back({child,false});
            drafts.emplace(cls,std::move(p));
            continue;
          }
          MomentNode p=std::move(drafts.at(cls));drafts.erase(cls);bool valid=true;
          for(auto child:p.children) {
            const auto q=plans.find(child);if(q==plans.end() || !q->second){valid=false;break;}
            p.has_moment|=q->second->has_moment;
          }
          plans.emplace(cls,valid?std::optional{std::move(p)}:std::nullopt);order.push_back(cls);active.erase(cls);
        }
      };
      phase="moment_plans";
      for(auto id:owner_ids)if(admitted[id])plan(id);
      std::unordered_set<Class> parented;
      for(const auto& [cls,p]:plans)if(p && p->has_moment) {
        for(auto child:p->children)if(plans.at(child)->has_moment)parented.insert(child);
        // Complete corrections consume at most eight immediate scalar heads.
        // Those internal partial factors are not independent candidate roots;
        // otherwise a balanced tree could spend the fixed root cap on them.
        for(auto child:p->consumed){charge();parented.insert(child);}
      }
      constexpr std::size_t moment_candidate_cap=16;
      std::vector<Class> moment_candidates;std::unordered_set<Class> moment_candidate_set;
      const auto appendMoment=[&](Class cls) {
        charge();if(moment_candidate_set.contains(cls))return true;
        if(moment_candidates.size()>=moment_candidate_cap)return false;
        moment_candidate_set.insert(cls);moment_candidates.push_back(cls);return true;
      };
      const auto appendMomentTail=[&](Class cls) {
        if(!appendMoment(cls))return;auto child=cls;
        // Also certify the unscaled M2 beneath a bounded final scalar tail.
        for(unsigned depth=0;depth<8 && plans.at(child)->kind==MomentKind::Scale;++depth) {
          child=plans.at(child)->children[0];if(!appendMoment(child))break;
        }
      };
      // Output-reachable complete components precede optional interior roots.
      // Traverse one ranked witness DAG once; when a whole plan is found its
      // descendants need not compete for the fixed sixteen-certificate cap.
      std::vector<Class> frontier;
      std::unordered_set<Class> frontier_seen;
      const auto appendFrontier=[&](Class cls) {
        charge();cls=owners[cls];if(frontier_seen.contains(cls) || frontier.size()>=max_active_nodes)return;
        frontier_seen.insert(cls);frontier.push_back(cls);
      };
      for(auto cls:left)appendFrontier(cls);for(auto cls:right)appendFrontier(cls);
      for(std::size_t i=0;i<frontier.size();++i) {
        charge();auto cls=frontier[i];
        if(auto p=plans.find(cls);p!=plans.end() && p->second && p->second->has_moment) {
          appendMomentTail(cls);continue;
        }
        Class head=n;
        if(literals.contains(cls))continue;
        if(auto h=selected.find(cls);h!=selected.end())head=h->second;
        else if(auto h=ordinary_folds.find(cls);h!=ordinary_folds.end())head=h->second;
        else if(!members[cls].empty())head=members[cls].front();
        if(head==n)continue;
        for(auto child:snapshot[head].children)appendFrontier(child);
      }
      for(const auto& [cls,p]:plans) {
        charge();if(moment_candidates.size()>=moment_candidate_cap)break;
        if(p && p->has_moment && !parented.contains(cls))appendMomentTail(cls);
      }
      std::vector<std::pair<Class,m::Builder::FamilySignature>> moment_signatures;
      for(auto root:moment_candidates) {
        // Reverse selected-DAG adjoints accumulate one scalar coefficient per
        // node. Never cache/copy a growing polynomial/term vector per prefix.
        std::unordered_map<Class,l::LinearForm> weights;weights.emplace(root,*one);
        m::Builder::FamilyForm form;std::map<m::MeanId,std::vector<m::Interval>> centered_pieces;
        std::size_t centered_entries=0;bool valid=true;
        for(auto it=order.rbegin();it!=order.rend() && valid;++it) {
          charge();const auto cls=*it;auto found=weights.find(cls);if(found==weights.end())continue;
          const auto& p=*plans.at(cls);const auto weight=found->second;
          if(p.kind==MomentKind::Zero)continue;
          if(p.kind==MomentKind::Correction) {
            if(!moments.addFixed(form,*p.correction,weight.constant()))valid=false;
            momentSync();continue;
          }
          if(p.kind==MomentKind::IndexedCorrection) {
            if(!moments.addFamily(form,*p.correction,p.translation,weight.constant(),true))valid=false;
            momentSync();if(!valid)break;
            // Fall through to the seed child with the SAME outer coefficient;
            // the seed is outside the Fold binder and is never repeated.
          }
          if(p.kind==MomentKind::Centered) {
            if(p.intervals.size()>1024-centered_entries){valid=false;break;}
            centered_entries+=p.intervals.size();
            for(auto [begin,end]:p.intervals) {
              charge();centered_pieces[*p.center].push_back({begin,end,weight.constant()});
            }
            continue;
          }
          for(std::size_t i=0;i<p.children.size();++i) {
            auto next=std::optional{weight};
            if((p.kind==MomentKind::Subtract && i==1) || p.kind==MomentKind::Negate) {
              next=builder.apply(l::Operation::Negate,std::array{&weight});sync();
            }
            if(p.kind==MomentKind::Scale)next=combine(l::Operation::Multiply,weight,p.factor);
            if(!next){valid=false;break;}
            auto previous=weights.find(p.children[i]);
            if(previous!=weights.end())next=combine(l::Operation::Add,previous->second,*next);
            if(!next){valid=false;break;}weights.insert_or_assign(p.children[i],std::move(*next));
          }
        }
        if(!valid)continue;
        for(const auto& [center,pieces]:centered_pieces) {
          if(!moments.addCentered(form,center,pieces,true)){valid=false;break;}momentSync();
        }
        if(!valid)continue;
        auto signature=moments.finishFamily(form);momentSync();
        if(!signature)continue;
        for(const auto& [other,key]:moment_signatures) {
          charge();auto equal=moments.equalFamilies(*signature,key);momentSync();
          if(equal && *equal)equalities.emplace_back(root,other);
        }
        moment_signatures.emplace_back(root,std::move(*signature));
      }
      } catch(const LocalCap&) {
        // Optional moment planning cannot discard complete linear certificates
        // or prevent the final bounded prefix fallback from being attempted.
        settleHelperFailure();
        traceCap(builder.work());
      }
      phase="scalar_linear_components";
      for(std::size_t i=priority_candidates;i<candidates.size();++i)indexCandidate(candidates[i]);
      sync();
      if(!extrema) {
        phase="scoped_nested_sums";
        if(trace){domain_fallback_reports=16;domain_denial_reports=16;}
        // Preserve every old linear/moment opportunity before this optional
        // complete-root proof. It borrows the SAME unrefreshed epoch ledger.
        // This is an existing paired-root window, not a per-output ancestor
        // scheduler; context-local Sum facts never create internal unions.
        BodyScopeRegistry scopes(body_heads_index,max_retained_summary_nodes);
        ScopedSumProof sums(scopes,builder,summarize,reductionFold,sync,scoped_sum_query);
        auto root_scope=scopes.output();
        if(root_scope) {
          std::size_t root_slots=0;
          const auto pairs=std::min(left.size(),right.size());
          for(std::size_t i=0;i<pairs && root_slots+2<=max_root_candidates;++i) {
            charge();const auto a=owners[left[i]],b=owners[right[i]];
            if(a==b)continue;root_slots+=2;
            if(a>=n || b>=n || snapshot[a].type!="real" || snapshot[b].type!="real" ||
                !finite_[a] || !finite_[b])continue;
            auto assignment=sums.root();
            BodyHeadIndex::FallbackTrace reports(body_heads_index,trace?&domain_fallback_reports:nullptr,
                trace?&domain_denial_reports:nullptr);
            if(affineBodyMapWithHeads(a,b,UINT32_MAX,snapshot,false,&remaining,body_heads_index,
                UINT32_MAX,nullptr,nullptr,nullptr,&sums,&assignment,*root_scope,*root_scope))
              equalities.emplace_back(a,b);
          }
        }
        sync();
        if(trace)SENTRYWARD_DEBUG_PRINT(egraph_56);
      }
    } catch(const LocalCap&) {
      if(!catalog_ready)catalog_failed=true;
      // Optional proof cap, never a partial certificate. Earlier completed
      // whole-root certificates are still sound and can be retained.
      settleHelperFailure();
      traceCap(builder.work());
    }
    if(linear_statistics.queries && sentryward::debug::environment("SENTRYWARD_EGRAPH_TRACE"))
      SENTRYWARD_DEBUG_PRINT(egraph_57);
    if(trace) {
      // One compact line per reducer epoch, including an interrupted mapper's
      // consumed work. Map components overlap group/proof costs; no per-node
      // trace, new walk, reset allowance or proof assumption is introduced.
      SENTRYWARD_DEBUG_PRINT(egraph_58);
    }
    if(shape_capped && sentryward::debug::environment("SENTRYWARD_EGRAPH_TRACE"))
      SENTRYWARD_DEBUG_PRINT(egraph_59);
    };
    // Extrema first: they can discharge a shared max/min scale before Sum
    // moments/prefix candidates consume the same fixed aggregate pool.
    if(modes[0])run_mode(s::Reducer::Max);
    if(modes[1])run_mode(s::Reducer::Min);
    if(modes[2])run_mode(s::Reducer::Sum);
    for(auto [a,b]:equalities)rewrite(a,b);
  }
  std::optional<Class> emitCommonAddFactor(Class id,const Key& k,Class factor,
                                           Class left,Class right) {
    // Both callers certify this SAME local finite ideal identity. Return its
    // compact residue so a frozen additive DAG can carry the proof upward
    // without awaiting another global rewrite epoch. Never distribute back.
    if(factored_sums_.contains(id))return {};
    Key residue;residue.kind=static_cast<int>(s::Kind::Operation);residue.type=k.type;
    residue.name="Add";residue.attributes="ideal_arithmetic";residue.children={left,right};
    const auto body=intern(std::move(residue));
    Key product;product.kind=static_cast<int>(s::Kind::Operation);product.type=k.type;
    product.name="Multiply";product.attributes="ideal_arithmetic";product.children={factor,body};
    rewrite(id,intern(std::move(product)));factored_sums_.insert(id);return body;
  }
  void factorAdditiveForests(const std::vector<Key>& snapshot,
                              const std::vector<bool>& nonrecursive) {
    namespace l=bounded_linear_reduction;
    // Optional bounded scheduling of the existing common-factor identity.
    // Only selected Add edges are followed; Fold, products and factor/residue
    // values remain opaque. Every stored class is summarized once per epoch.
    constexpr std::size_t max_records=65536,max_product_heads=4,max_factors=8;
    std::size_t remaining=2000000;
    const auto charge=[&](std::size_t amount=1) {
      poll();if(amount>remaining)return false;remaining-=amount;
      if(options_.max_work) {
        if(amount>options_.max_work-work_)throw Limit{"work_budget"};
        work_+=amount;
      }
      return true;
    };
    const auto n=snapshot.size();if(n>1000000)return;
    std::vector<Class> owners(n);
    std::vector<bool> unit(n);
    for(Class id=0;id<n;++id) {
      if(!charge())return;
      owners[id]=find(id);if(owners[id]>=n)return; // Appended owner: Unknown for this frozen factory.
      const auto& k=snapshot[id];
      if(k.kind==static_cast<int>(s::Kind::Constant) && k.type=="real" && k.name=="1" &&
          k.attributes.empty() && k.provenance.empty())unit[owners[id]]=true;
    }
    l::RankedDagStatistics statistics;
    auto ranked=l::rankedDagHeads(owners,[&](Class id)->std::span<const Class> {
      return snapshot[id].children;
    },n,remaining,&statistics);
    if(!charge(statistics.work) || !ranked)return;
    struct Descriptor {Class factor,residue;};
    struct Record {
      Class sum=0;bool has_sum=false,unknown=false;unsigned char state=0;
      std::vector<Class> products;
      std::vector<Descriptor> factors;
    };
    std::unordered_map<Class,Record> records;
    const auto eligible=[&](const Key& k) {
      return k.type=="real" && idealOperation(k) && k.provenance.empty() &&
        k.children.size()==2 && (k.name=="Add" || k.name=="Multiply") &&
        snapshot[k.children[0]].type==k.type && snapshot[k.children[1]].type==k.type &&
        finite_[owners[k.children[0]]] && finite_[owners[k.children[1]]];
    };
    for(Class id=0;id<n;++id) {
      if(!charge())return;const auto& k=snapshot[id];
      if(!nonrecursive[id] || !eligible(k))continue;
      const auto cls=owners[id];
      if(!records.contains(cls) && records.size()==max_records)return;
      auto& record=records[cls];
      if(k.name=="Add" && (*ranked)[id] && !record.has_sum) {
        record.sum=id;record.has_sum=true;
      }
      if(k.name=="Multiply") {
        // Late proven product aliases need not precede their owner rank.
        // They are inspected ONLY as local opaque identities, never followed
        // recursively; the SCC mask and explicit self-use guard reject cycles.
        if(owners[k.children[0]]==cls || owners[k.children[1]]==cls)continue;
        if(record.products.size()==max_product_heads)record.unknown=true;
        else record.products.push_back(id);
      }
    }
    const auto insert=[&](Record& r,Descriptor d) {
      auto at=std::lower_bound(r.factors.begin(),r.factors.end(),d.factor,
          [](const Descriptor& a,Class b){return a.factor<b;});
      if(at!=r.factors.end() && at->factor==d.factor)return true;
      if(r.factors.size()==max_factors)return false;
      r.factors.insert(at,d);return true;
    };
    struct Task {Class cls;bool ready;};
    for(auto& [root,unused]:records) {
      (void)unused;if(records.at(root).state==2)continue;
      std::vector<Task> pending{{root,false}};
      while(!pending.empty()) {
        if(!charge())return;
        auto t=pending.back();pending.pop_back();auto& r=records.at(t.cls);
        if(r.state==2)continue;
        if(!t.ready) {
          if(r.state==1){r.unknown=true;r.state=2;continue;}
          r.state=1;pending.push_back({t.cls,true});
          if(r.has_sum)for(auto child:snapshot[r.sum].children) {
            const auto cls=owners[child];auto c=records.find(cls);
            if(c!=records.end() && c->second.state!=2)pending.push_back({cls,false});
          }
          continue;
        }
        r.state=2;if(r.unknown)continue;
        for(auto head:r.products)for(unsigned side=0;side<2;++side) {
          if(!charge())return;const auto& k=snapshot[head];
          const auto factor=owners[k.children[side]];
          if(unit[factor])continue; // Never grow a self-alias by factoring one.
          if(!insert(r,{factor,owners[k.children[1-side]]})){r.unknown=true;break;}
        }
        if(r.unknown || !r.has_sum || factored_sums_.contains(r.sum))continue;
        const auto k=snapshot[r.sum];
        auto a=records.find(owners[k.children[0]]),b=records.find(owners[k.children[1]]);
        if(a==records.end() || b==records.end() || a->second.unknown || b->second.unknown)continue;
        const auto& left=a->second.factors;const auto& right=b->second.factors;
        std::size_t i=0,j=0;
        while(i<left.size() && j<right.size()) {
          if(!charge())return;
          if(left[i].factor<right[j].factor){++i;continue;}
          if(right[j].factor<left[i].factor){++j;continue;}
          const auto factor=left[i].factor;
          auto existing=std::lower_bound(r.factors.begin(),r.factors.end(),factor,
              [](const Descriptor& a,Class f){return a.factor<f;});
          if(existing!=r.factors.end() && existing->factor==factor)break;
          if(r.factors.size()==max_factors){r.unknown=true;break;}
          // The two descriptors are complete proven use identities, including
          // type/capture. Their opaque residues may be newly interned IDs;
          // no vector element reference survives the shared emitter's append.
          const auto residue=emitCommonAddFactor(r.sum,k,factor,left[i].residue,right[j].residue);
          if(residue)insert(r,{factor,*residue});
          break; // One common witness, at most two enodes per stored Add head.
        }
      }
    }
  }
  void mergePolynomialNormalForms(const std::vector<Key>& snapshot,
                                  const std::vector<bool>& nonrecursive) {
    namespace p=polynomial_certificate;
    // Exact bounded certificates for finite ideal-real scalar algebra. Fold,
    // Select and other math stay atomic. Only literal nonzero denominators can
    // be expanded; arbitrary division/cancellation is never licensed here.
    // No distributed expression trees are interned. Memoization is per frozen
    // class/type and epoch; hard limits bound terms, degree and integer bits.
    std::map<std::string,std::vector<Class>> groups;
    const auto n=snapshot.size();std::vector<Class> owners(n);
    for(Class id=0;id<n;++id) {
      poll();
      owners[id]=find(id);
      if(nonrecursive[id] && realType(snapshot[id].type) &&
          snapshot[id].provenance.empty() && finite_[owners[id]])
        groups[snapshot[id].type].push_back(id);
    }
    std::vector<std::pair<Class,Class>> equalities;
    for(const auto& [type,heads]:groups) {
      p::Limits limits;
      if(options_.max_work)limits.max_work=std::min(limits.max_work,options_.max_work-work_);
      p::Builder builder(limits);
      std::size_t charged=0;
      const auto charge=[&] {
        const auto delta=builder.work()-charged;charged=builder.work();
        if(options_.max_work && delta>options_.max_work-work_)throw Limit{"work_budget"};
        if(options_.max_work)work_+=delta;
      };
      std::unordered_map<Class,Class> selected;
      std::unordered_map<Class,std::string> literals;
      for(auto id:heads)if(snapshot[id].kind==static_cast<int>(s::Kind::Constant) && snapshot[id].attributes.empty())
        literals.try_emplace(owners[id],snapshot[id].name);
      const auto eligible=[&,&type=type](const Key& k) {
        if(!idealOperation(k))return false;
        if(!std::all_of(k.children.begin(),k.children.end(),[&](auto c) {
          return snapshot[c].type==type && finite_[owners[c]];
        }))return false;
        if(k.name=="Add" || k.name=="Subtract" || k.name=="Multiply" || k.name=="Negate")return true;
        if(k.name=="Divide")return literals.contains(owners[k.children[1]]);
        if(k.name=="Reciprocal")return literals.contains(owners[k.children[0]]);
        return false;
      };
      for(auto id:heads){poll();if(eligible(snapshot[id]))selected.try_emplace(owners[id],id);}
      std::unordered_map<Class,p::Result> summaries;
      std::unordered_map<Class,unsigned char> state;
      const auto evaluate=[&](const Key& k)->p::Result {
        std::vector<const p::CanonicalPolynomial*> operands;
        for(auto child:k.children) {
          const auto& summary=summaries[owners[child]];if(!summary)return {};
          operands.push_back(&*summary);
        }
        if(k.name=="Reciprocal") {
          auto one=builder.literal("1");if(!one)return {};
          return builder.divideByLiteral(*one,literals.at(owners[k.children[0]]));
        }
        if(k.name=="Divide")return builder.divideByLiteral(*operands[0],literals.at(owners[k.children[1]]));
        auto op=p::Operation::Add;
        if(k.name=="Subtract")op=p::Operation::Subtract;
        if(k.name=="Multiply")op=p::Operation::Multiply;
        if(k.name=="Negate")op=p::Operation::Negate;
        return builder.apply(op,operands);
      };
      struct Task {Class cls;bool ready;};
      for(auto id:heads) {
        if(state[owners[id]]==2)continue;
        std::vector<Task> work{{owners[id],false}};
        while(!work.empty() && !builder.exhausted()) {
          poll();auto [cls,ready]=work.back();work.pop_back();if(state[cls]==2)continue;
          if(auto literal=literals.find(cls);literal!=literals.end()) {
            summaries[cls]=builder.literal(literal->second);state[cls]=2;continue;
          }
          auto head=selected.find(cls);
          if(head==selected.end()){summaries[cls]=builder.atom(cls);state[cls]=2;continue;}
          const auto& k=snapshot[head->second];
          if(ready) {
            summaries[cls]=evaluate(k);
            if(!summaries[cls])summaries[cls]=builder.atom(cls);
            state[cls]=2;continue;
          }
          if(state[cls]==1){summaries[cls]=std::nullopt;state[cls]=2;continue;}
          state[cls]=1;work.push_back({cls,true});
          for(auto child:k.children)if(state[owners[child]]!=2)work.push_back({owners[child],false});
        }
        charge();if(builder.exhausted())break;
      }
      std::map<p::CanonicalPolynomial,Class> signatures;
      for(auto id:heads) {
        poll();
        const auto& k=snapshot[id];p::Result signature;
        const auto owner=owners[id];
        const auto choice=selected.find(owner);
        if((choice!=selected.end() && choice->second==id) ||
            (choice==selected.end() && (!eligible(k) || literals.contains(owner)))) {
          if(auto found=summaries.find(owner);found!=summaries.end())signature=found->second;
        } else if(!builder.exhausted()) {
          if(k.kind==static_cast<int>(s::Kind::Constant) && k.attributes.empty())signature=builder.literal(k.name);
          else if(eligible(k))signature=evaluate(k);
          else signature=builder.atom(owner);
        }
        charge();
        if(!signature)continue;
        auto [entry,inserted]=signatures.emplace(std::move(*signature),id);
        if(!inserted)equalities.emplace_back(id,entry->second);
      }
    }
    for(auto [a,b]:equalities)rewrite(a,b);
  }
  void mergeAssociativeNormalForms(const std::vector<Key>& snapshot,
                                   const std::vector<bool>& nonrecursive) {
    // A bounded congruence analysis, not an expansive rewrite. Freeze owners
    // and one nonrecursive decomposition per class/operator for this epoch;
    // only then union equal complete signatures. No binary enodes are emitted.
    // B=256 DISTINCT scalar atoms per summary. Counts preserve shared DAGs
    // such as Add(x,x) without materializing their logical occurrences. A cap
    // or count overflow makes this optional certificate unavailable, not true.
    // Cost O(B*(N+E)+B*N*log N) per epoch, space O(B*N), with no Fold expansion.
    constexpr std::size_t max_atoms=256;
    using Terms=std::vector<std::pair<Class,std::uint64_t>>;
    const auto n=snapshot.size();
    std::vector<Class> owners(n);
    for(Class id=0;id<n;++id)owners[id]=find(id);
    std::map<std::pair<std::string,std::string>,std::vector<Class>> groups;
    for(Class id=0;id<n;++id) {
      const auto& k=snapshot[id];
      if(nonrecursive[id] && arithmetic(k) && k.provenance.empty() &&
          (k.name=="Add" || k.name=="Multiply") &&
          finite_[owners[k.children[0]]] && finite_[owners[k.children[1]]])
        groups[{k.name,k.type}].push_back(id);
    }
    std::vector<std::pair<Class,Class>> equalities;
    const auto merge=[&](const Terms& a,const Terms& b)->std::optional<Terms> {
      Terms result;result.reserve(std::min(max_atoms,a.size()+b.size()));
      std::size_t i=0,j=0;
      while(i<a.size() || j<b.size()) {
        poll();std::pair<Class,std::uint64_t> term;
        if(j==b.size() || (i<a.size() && a[i].first<b[j].first))term=a[i++];
        else if(i==a.size() || b[j].first<a[i].first)term=b[j++];
        else {
          if(b[j].second>UINT64_MAX-a[i].second)return {};
          term={a[i].first,a[i].second+b[j].second};++i;++j;
        }
        if(result.size()==max_atoms)return {};result.push_back(term);
      }
      return result;
    };
    for(const auto& [spec,heads]:groups) {
      std::unordered_map<Class,Class> selected;
      for(auto id:heads)selected.try_emplace(owners[id],id);
      std::unordered_map<Class,std::optional<Terms>> summaries;
      std::unordered_map<Class,unsigned char> state;
      struct Task {Class cls;bool ready;};
      for(auto id:heads) {
        std::vector<Task> work{{owners[id],false}};
        while(!work.empty()) {
          poll();auto [cls,ready]=work.back();work.pop_back();
          if(state[cls]==2)continue;
          auto head=selected.find(cls);
          if(head==selected.end()) {
            summaries[cls]=Terms{{cls,1}};state[cls]=2;continue;
          }
          const auto& k=snapshot[head->second];
          auto a=owners[k.children[0]],b=owners[k.children[1]];
          if(ready) {
            auto& left=summaries[a];auto& right=summaries[b];
            summaries[cls]=left && right?merge(*left,*right):std::nullopt;
            state[cls]=2;continue;
          }
          if(state[cls]==1) {summaries[cls]=std::nullopt;state[cls]=2;continue;}
          state[cls]=1;work.push_back({cls,true});
          if(state[b]!=2)work.push_back({b,false});
          if(state[a]!=2)work.push_back({a,false});
        }
      }
      std::map<Terms,Class> signatures;
      for(auto id:heads) {
        poll();const auto& k=snapshot[id];
        const auto& a=summaries[owners[k.children[0]]];
        const auto& b=summaries[owners[k.children[1]]];
        if(!a || !b)continue;
        auto signature=merge(*a,*b);if(!signature)continue;
        auto [entry,inserted]=signatures.emplace(std::move(*signature),id);
        if(!inserted)equalities.emplace_back(id,entry->second);
      }
    }
    for(auto [a,b]:equalities)rewrite(a,b);
  }
  std::vector<bool> nonrecursiveHeads(const std::vector<Key>& snapshot) {
    // Equality creates recursive alternatives such as x=(x+y)-y. They are
    // useful proven unions, not finite trees to unfold repeatedly. Compute
    // class SCCs once per epoch (O(enodes+operand edges)); expansive matching
    // uses only heads which leave their result's SCC. Never discard a union.
    // Rewriters can append and unite new product/residue nodes while this
    // epoch's source snapshot stays frozen. SCC owners and paths therefore
    // range over the CURRENT stored graph, not merely its snapshot prefix.
    const auto n=nodes_.size();
    std::vector<std::vector<Class>> out(n),in(n);
    for(Class id=0;id<n;++id)for(auto child:nodes_[id].children) {
      poll();auto a=find(id),b=find(child);out[a].push_back(b);in[b].push_back(a);
    }
    std::vector<bool> seen(n);std::vector<Class> order;order.reserve(n);
    for(Class root=0;root<n;++root)if(find(root)==root && !seen[root]) {
      seen[root]=true;std::vector<std::pair<Class,std::size_t>> work{{root,0}};
      while(!work.empty()) {
        poll();auto& [id,cursor]=work.back();
        if(cursor==out[id].size()){order.push_back(id);work.pop_back();continue;}
        auto child=out[id][cursor++];if(seen[child])continue;
        seen[child]=true;work.emplace_back(child,0);
      }
    }
    std::vector<Class> component(n,n);
    for(auto cursor=order.rbegin();cursor!=order.rend();++cursor)if(component[*cursor]==n) {
      component[*cursor]=*cursor;std::vector<Class> work{*cursor};
      while(!work.empty()) {
        poll();auto id=work.back();work.pop_back();
        for(auto parent:in[id])if(component[parent]==n){component[parent]=*cursor;work.push_back(parent);}
      }
    }
    std::vector<bool> result(snapshot.size(),true);
    for(Class id=0;id<snapshot.size();++id)for(auto child:snapshot[id].children)
      if(component[find(id)]==component[find(child)]){result[id]=false;break;}
    return result;
  }
  void mergeAlphaEquivalentFolds() {
    // One import normalization, before any unions can introduce cyclic
    // e-class alternatives. Private signatures use de Bruijn bound indices;
    // free indices remain absolute. ONLY whole Fold results signed under the
    // empty environment are unioned, never context-dependent bodies/indices.
    struct PairHash {
      std::size_t operator()(const std::pair<Class,Class>& p) const {
        return p.first^(p.second+0x9e3779b9+(p.first<<6)+(p.first>>2));
      }
    };
    struct Scope {Class parent;std::uint32_t level;};
    std::vector<Scope> scopes{{0,0}};
    std::unordered_map<std::pair<Class,Class>,Class,PairHash> scope_ids,values;
    std::map<std::vector<std::pair<std::uint32_t,std::uint32_t>>,Class> context_ids;
    context_ids.emplace(std::vector<std::pair<std::uint32_t,std::uint32_t>>{},0);
    std::unordered_map<Key,Class,Hash> signatures;
    std::vector<std::pair<Class,Class>> folds;
    const auto bound=[&](std::uint32_t level,Class scope)->std::optional<std::uint32_t> {
      std::uint32_t distance=0;
      for(auto current=scope;current;current=scopes[current].parent,++distance) {
        poll();if(scopes[current].level==level)return distance;
      }
      return {};
    };
    const auto context=[&](Class node,Class scope) {
      // Unused enclosing binders cannot multiply the cost of a closed DAG.
      std::vector<std::pair<std::uint32_t,std::uint32_t>> used;
      for(auto level:free_indices_[node])if(auto position=bound(level,scope))used.emplace_back(level,*position);
      auto found=context_ids.find(used);
      if(found==context_ids.end()) {
        if(context_ids.size()>=options_.max_nodes)throw Limit{"alpha_normalization_context_budget"};
        found=context_ids.emplace(std::move(used),context_ids.size()).first;
      }
      return std::pair{node,found->second};
    };
    const auto bodyScope=[&](Class parent,std::uint32_t level) {
      auto key=std::pair{parent,Class(level)};
      if(auto found=scope_ids.find(key);found!=scope_ids.end())return found->second;
      if(scopes.size()>=options_.max_nodes)throw Limit{"alpha_normalization_context_budget"};
      auto id=scopes.size();scopes.push_back({parent,level});scope_ids.emplace(key,id);return id;
    };
    struct Task {Class node,scope;bool ready;};
    for(Class root=0;root<nodes_.size();++root) {
      poll();
      if(nodes_[root].kind!=static_cast<int>(s::Kind::Fold))continue;
      std::vector<Task> work{{root,0,false}};
      while(!work.empty()) {
        poll();const auto task=work.back();work.pop_back();
        const auto ck=context(task.node,task.scope);
        if(values.contains(ck))continue;
        auto k=nodes_[task.node];
        const auto is_fold=k.kind==static_cast<int>(s::Kind::Fold);
        const auto nested=is_fold?bodyScope(task.scope,k.fold->binder_level):task.scope;
        if(!task.ready) {
          work.push_back({task.node,task.scope,true});
          for(std::size_t i=0;i<k.children.size();++i) {
            const auto scope=is_fold && i==1?nested:task.scope;
            if(!values.contains(context(k.children[i],scope)))work.push_back({k.children[i],scope,false});
          }
          continue;
        }
        const bool commutative=(arithmetic(k) && finite_[k.children[0]] && finite_[k.children[1]]) ||
            (integerCommutative(k) && nodes_[k.children[0]].type==k.type && nodes_[k.children[1]].type==k.type);
        for(std::size_t i=0;i<k.children.size();++i)
          k.children[i]=values.at(context(k.children[i],is_fold && i==1?nested:task.scope));
        if(commutative && k.children[1]<k.children[0])std::swap(k.children[0],k.children[1]);
        if(k.kind==static_cast<int>(s::Kind::Index)) {
          if(auto position=bound(k.index_level,task.scope)){k.kind=9;k.index_level=*position;}
          else k.kind=8;
        }
        if(is_fold)k.fold->binder_level=0;
        auto found=signatures.find(k);
        if(found==signatures.end()) {
          if(signatures.size()>=options_.max_nodes)throw Limit{"alpha_normalization_node_budget"};
          found=signatures.emplace(std::move(k),signatures.size()).first;
        }
        if(values.size()>=options_.max_nodes)throw Limit{"alpha_normalization_context_budget"};
        values.emplace(ck,found->second);
      }
      folds.emplace_back(values.at(context(root,0)),root);
    }
    std::unordered_map<Class,Class> first;
    const auto before=unions_;
    for(auto [signature,root]:folds) {
      auto [found,inserted]=first.try_emplace(signature,root);
      if(!inserted)rewrite(root,found->second);
    }
    if(sentryward::debug::environment("SENTRYWARD_EGRAPH_TRACE"))SENTRYWARD_DEBUG_PRINT(egraph_60);
  }
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
  SelectIndex indexSelectPatterns(const std::vector<Key>& snapshot,const std::vector<bool>& nonrecursive) {
    SelectIndex result;
    for(Class id=0;id<snapshot.size();++id) {
      if(!nonrecursive[id])continue;
      poll();const auto& k=snapshot[id];const auto root=find(id);
      if(k.kind==static_cast<int>(s::Kind::Constant) && realType(k.type)) {
        result.constants.insert(root);
        if(k.name=="1" && k.attributes.empty() && k.provenance.empty())result.ones.insert(root);
      }
      if(k.kind==static_cast<int>(s::Kind::Constant) &&
          (k.type=="u64" || k.type=="exact:u64") && k.attributes.empty() &&
          k.provenance.empty() && k.name.size()<=20) {
        std::uint64_t value=0;
        const auto parsed=std::from_chars(k.name.data(),k.name.data()+k.name.size(),value);
        if(parsed.ec==std::errc{} && parsed.ptr==k.name.data()+k.name.size())
          result.u64_constants.try_emplace(root,SelectIndex::U64Constant{k.type,value});
      }
      if(idealSelect(k))result.heads[root].conditions.try_emplace(find(k.children[0]),id);
      if(idealOperation(k) && k.provenance.empty() && k.children.size()==1)
        result.heads[root].unary.try_emplace(k.name,id);
    }
    for(Class id=0;id<snapshot.size();++id) {
      if(!nonrecursive[id])continue;
      poll();const auto& k=snapshot[id];
      if(k.kind==static_cast<int>(s::Kind::Operation) && k.name=="Add" &&
          (k.type=="u64" || k.type=="exact:u64") && k.children.size()==2 &&
          k.attributes.empty() && k.provenance.empty()) {
        const auto a=find(k.children[0]),b=find(k.children[1]);
        if(a<snapshot.size() && b<snapshot.size() && snapshot[a].type==k.type && snapshot[b].type==k.type)
          for(unsigned side=0;side<2;++side) {
            const auto constant=result.u64_constants.find(side?b:a);
            if(constant==result.u64_constants.end() || constant->second.type!=k.type)continue;
            result.u64_add_constants.try_emplace(find(id),
                SelectIndex::U64AddConstant{k.type,id,side?a:b,constant->second.value});
            break; // One acyclic immediate witness per class, never products of alternatives.
          }
      }
      if(!idealOperation(k) || !k.provenance.empty() || k.children.size()!=2 ||
          (k.name!="Add" && k.name!="Multiply" && k.name!="Subtract" && k.name!="Divide"))continue;
      auto& heads=result.heads[find(id)];
      auto a=find(k.children[0]),b=find(k.children[1]);
      if(k.name=="Add" || k.name=="Multiply") {
        heads.factors.try_emplace(std::pair{k.name,a},b);
        heads.factors.try_emplace(std::pair{k.name,b},a);
      }
      if(k.name=="Multiply") {
        // Zero-arm factoring prefers a constant invariant factor. This is a
        // local canonical choice, not a search through a numerical subtree.
        if(!result.constants.contains(a) && result.constants.contains(b))std::swap(a,b);
        if(!heads.product || (!result.constants.contains(heads.product->first) && result.constants.contains(a)))
          heads.product=std::pair{a,b};
      }
      if(k.name=="Subtract" && !heads.difference)heads.difference=std::pair{a,b};
      if(k.name=="Divide" && !heads.quotient)heads.quotient=std::pair{a,b};
      if(k.name=="Add" && !heads.difference)for(unsigned side=0;side<2;++side) {
        const auto& unary=result.get(find(k.children[side])).unary;
        const auto neg=unary.find("Negate");
        if(neg==unary.end())continue;
        heads.difference=std::pair{find(k.children[1-side]),find(snapshot[neg->second].children[0])};
        break;
      }
    }
    return result;
  }
  void rewriteExactU64Add(Class id,const Key& k,const std::vector<Key>& snapshot,
                          const SelectIndex& index) {
    if(k.kind!=static_cast<int>(s::Kind::Operation) || k.name!="Add" ||
        (k.type!="u64" && k.type!="exact:u64") || k.children.size()!=2 ||
        !k.attributes.empty() || !k.provenance.empty())return;
    const auto a=find(k.children[0]),b=find(k.children[1]);
    if(a>=snapshot.size() || b>=snapshot.size() || snapshot[a].type!=k.type || snapshot[b].type!=k.type)return;
    const auto literal=[&](Class root)->std::optional<std::uint64_t> {
      const auto found=index.u64_constants.find(root);
      if(found==index.u64_constants.end() || found->second.type!=k.type)return {};
      return found->second.value;
    };
    const auto constant=[&](std::uint64_t value) {
      Key result;result.kind=static_cast<int>(s::Kind::Constant);
      result.type=k.type;result.name=std::to_string(value);return intern(std::move(result));
    };
    const auto sum=[](std::uint64_t x,std::uint64_t y) {
      // Reuse the exact u64 affine ring: overflow is modulo 2^64, not an
      // assumed absence of wrap. Both coefficient maps are empty, O(1).
      return combineAffine("Add",IndexAffine{x,{}},IndexAffine{y,{}})->constant;
    };
    const auto first=literal(a),second=literal(b);
    if(first && second){rewrite(id,constant(sum(*first,*second)));return;}
    for(unsigned side=0;side<2;++side) {
      poll();const auto outer=side?second:first;
      if(!outer)continue;
      const auto inner=side?a:b;
      if(*outer==0){rewrite(id,inner);return;}
      const auto head=index.u64_add_constants.find(inner);
      if(head==index.u64_add_constants.end() || head->second.type!=k.type)continue;
      const auto value=find(head->second.value);
      if(value>=snapshot.size() || snapshot[value].type!=k.type ||
          find(head->second.source)==find(id) || value==find(id))continue;
      const auto folded=sum(*outer,head->second.constant);
      if(!folded){rewrite(id,value);return;}
      Key result;result.kind=static_cast<int>(s::Kind::Operation);
      result.name="Add";result.type=k.type;result.children={value,constant(folded)};
      rewrite(id,intern(std::move(result)));return; // At most one replacement head.
    }
  }
  bool rewriteKnownPredicateSelect(Class id,const Key& k,
      const std::vector<Key>& snapshot) {
    if(k.kind!=static_cast<int>(s::Kind::Operation) || k.name!="Select" ||
        k.children.size()!=3 || !k.provenance.empty() ||
        (!k.attributes.empty() && k.attributes!="ideal_arithmetic"))return false;
    const auto p=find(k.children[0]),a=find(k.children[1]),b=find(k.children[2]);
    if(p>=snapshot.size() || a>=snapshot.size() || b>=snapshot.size() ||
        (snapshot[p].type!="pred" && snapshot[p].type!="exact:pred") ||
        snapshot[a].type!=k.type || snapshot[b].type!=k.type)return false;
    // Existing three-valued proof: Unknown is never interpreted as False.
    // Selection is exact for typed storage/integer/opaque arms too. No finite
    // fact, arithmetic interpretation, new head or alternative scan is needed.
    const auto known=predicate_[p];if(!known)return false;
    rewrite(id,known==1?a:b);return true;
  }
  bool rewriteFiniteCompareSelectExtremum(Class id,const Key& k,
      const std::vector<Key>& snapshot) {
    if(!idealSelect(k) || k.type!="real" || !k.provenance.empty())return false;
    // Keep this stored predicate head: an earlier local union in this epoch
    // may have made its current owner a constant without removing the witness.
    const auto p=k.children[0],a=find(k.children[1]),b=find(k.children[2]);
    if(p>=snapshot.size())return false;
    const auto& condition=snapshot[p];const auto predicate=comparison(condition);
    if(!predicate || (*predicate!="lt" && *predicate!="le" &&
                      *predicate!="gt" && *predicate!="ge"))return false;
    const auto x=find(condition.children[0]),y=find(condition.children[1]);
    if(x>=snapshot.size() || y>=snapshot.size() || snapshot[x].type!="real" ||
        snapshot[y].type!="real" || !finite_[x] || !finite_[y] ||
        !((a==x && b==y) || (a==y && b==x)))return false;
    // Both finite numerical arms must be exactly the comparands. The shared
    // comparison certificate also admits unordered_true on these finite inputs.
    // Exact/bit/opaque heads and unknown predicates retain their original form.
    const bool true_is_x=a==x,true_is_smaller=*predicate=="lt" || *predicate=="le";
    Key extrema;extrema.kind=static_cast<int>(s::Kind::Operation);extrema.type=k.type;
    extrema.name=true_is_x==true_is_smaller?"Minimum":"Maximum";
    extrema.attributes="ideal_arithmetic";extrema.children={x,y};
    rewrite(id,intern(std::move(extrema)));return true; // At most one new head.
  }
  void rewriteSelectPatterns(Class id,const Key& k,const std::vector<Class>& original,
      const std::vector<Key>& snapshot,const SelectIndex& index) {
    const auto make=[&](std::string name,std::vector<Class> children) {
      Key operation;operation.kind=static_cast<int>(s::Kind::Operation);
      operation.type=k.type;operation.name=std::move(name);
      operation.attributes="ideal_arithmetic";operation.children=std::move(children);
      return intern(std::move(operation));
    };
    const auto choose=[&](Class p,Class a,Class b) {
      return make("Select",{find(p),find(a),find(b)});
    };
    const auto zero=[&] {
      Key value;value.kind=static_cast<int>(s::Kind::Constant);value.type=k.type;value.name="0";
      return intern(std::move(value));
    };
    const auto finite=[&](Class a){return finite_[find(a)];};
    const auto nonzero=[&](Class a){return nonzero_[find(a)];};
    if(idealSelect(k)) {
      const auto p=k.children[0],a=k.children[1],b=k.children[2];
      if(a==b){rewrite(id,a);return;}
      for(unsigned arm=0;arm<2;++arm) {
        const auto& heads=index.get(original[arm+1]).conditions;
        if(auto found=heads.find(p);found!=heads.end()) {
          const auto& nested=snapshot[found->second];
          if(nested.type!=k.type)continue;
          auto branches=k.children;branches[arm+1]=find(nested.children[arm+1]);
          rewrite(id,make("Select",std::move(branches)));
        }
      }
      if(factored_selects_.contains(id) || !finite(a) || !finite(b))return;
      // Unary factoring is oriented out of Select, not both ways. Exp2(0)=1
      // supplies the identity arm without an unrestricted inverse rewrite.
      for(const auto* name:{"Exp2","Exp","Negate","Reciprocal"}) {
        std::optional<Class> args[2];
        for(unsigned arm=0;arm<2;++arm) {
          const auto& heads=index.get(original[arm+1]).unary;
          if(auto found=heads.find(name);found!=heads.end()) {
            const auto& unary=snapshot[found->second];
            if(unary.type==k.type)args[arm]=find(unary.children[0]);
          } else if((std::string_view(name)=="Exp2" || std::string_view(name)=="Exp") && index.ones.contains(original[arm+1]))args[arm]=zero();
          else if(std::string_view(name)=="Negate" && zero_[find(original[arm+1])])args[arm]=zero();
        }
        if(!args[0] || !args[1] || !finite(*args[0]) || !finite(*args[1]) ||
            (std::string_view(name)=="Reciprocal" && (!nonzero(*args[0]) || !nonzero(*args[1]))))continue;
        rewrite(id,make(name,{choose(p,*args[0],*args[1])}));
        factored_selects_.insert(id);return;
      }
      // Intersect local factor heads rather than cross-product all e-class
      // alternatives. Each successful source enode emits only one factoring.
      const auto& left=index.get(original[1]).factors;
      const auto& right=index.get(original[2]).factors;
      const auto& small=left.size()<right.size()?left:right;
      const auto& other=left.size()<right.size()?right:left;
      for(const auto& [head,variable]:small) {
        poll();auto found=other.find(head);if(found==other.end())continue;
        const auto [name,common]=head;
        const auto x=left.size()<right.size()?variable:found->second;
        const auto y=left.size()<right.size()?found->second:variable;
        if(!finite(common) || !finite(x) || !finite(y))continue;
        rewrite(id,make(name,{find(common),choose(p,x,y)}));
        factored_selects_.insert(id);return;
      }
      for(unsigned arm=0;arm<2;++arm) {
        if(!zero_[find(original[arm+1])])continue;
        const auto& heads=index.get(original[2-arm]);
        if(heads.product) {
          auto [factor,value]=*heads.product;
          if(finite(factor) && finite(value)) {
            auto selected=arm==0?choose(p,zero(),value):choose(p,value,zero());
            rewrite(id,make("Multiply",{find(factor),selected}));
            factored_selects_.insert(id);return;
          }
        }
        // Select(p,0,x-y)=x-Select(p,x,y), and the opposite arm.
        // This fixed Add/Negate/Subtract pattern exposes correlated recentering
        // without distributing arbitrary arithmetic across one Select.
        if(heads.difference) {
          auto [x,y]=*heads.difference;
          if(finite(x) && finite(y)) {
            auto selected=arm==0?choose(p,x,y):choose(p,y,x);
            rewrite(id,make("Add",{find(x),make("Negate",{selected})}));
            factored_selects_.insert(id);return;
          }
        }
      }
      return;
    }
    if(!idealOperation(k) || !k.provenance.empty() || k.children.size()!=2 ||
        merged_selects_.contains(id))return;
    const auto& left=index.get(original[0]).conditions;
    const auto& right=index.get(original[1]).conditions;
    const auto& small=left.size()<right.size()?left:right;
    const auto& other=left.size()<right.size()?right:left;
    for(auto [condition,member]:small) {
      poll();auto found=other.find(condition);if(found==other.end())continue;
      const auto& x=snapshot[left.size()<right.size()?member:found->second];
      const auto& y=snapshot[left.size()<right.size()?found->second:member];
      if(x.type!=k.type || y.type!=k.type ||
          !finite(x.children[1]) || !finite(x.children[2]) ||
          !finite(y.children[1]) || !finite(y.children[2]) ||
          (k.name=="Divide" && (!nonzero(y.children[1]) || !nonzero(y.children[2]))))continue;
      auto a=make(k.name,{find(x.children[1]),find(y.children[1])});
      auto b=make(k.name,{find(x.children[2]),find(y.children[2])});
      rewrite(id,choose(condition,a,b));merged_selects_.insert(id);return;
    }
  }
  // Opt-in bounded diagnostics, never part of proof or property propagation.
#if defined(SENTRYWARD_DEVELOPMENT_DIAGNOSTICS) || defined(SENTRYWARD_ENABLE_COMPRESSION_AUDIT)
#include "sentryward/development/diagnose.inc"
#else
  void diagnose(std::span<const Class> ,std::span<const Class> ) {}
#endif
  void compactImported(std::vector<Class>& a,std::vector<Class>& b) {
    std::vector<bool> live(nodes_.size());
    std::vector<Class> todo=a;todo.insert(todo.end(),b.begin(),b.end());
    while(!todo.empty()) {
      poll();auto id=todo.back();todo.pop_back();
      if(live[id])continue;
      live[id]=true;todo.insert(todo.end(),nodes_[id].children.begin(),nodes_[id].children.end());
    }
    auto old=std::move(nodes_);
    nodes_.clear();parents_.clear();sizes_.clear();minimum_.clear();finite_.clear();
    class_members_.clear();real_head_ids_.clear();fold_head_ids_.clear();
    nonnegative_.clear();positive_.clear();nonzero_.clear();predicate_.clear();
    zero_.clear();negative_infinity_.clear();positive_infinity_.clear();
    integer_ranges_.clear();
    free_indices_.clear();dependents_.clear();property_queue_.clear();property_queued_.clear();
    max_index_.clear();has_fold_.clear();
    memo_.clear();factored_folds_.clear();
    // Import/projection matcher keys contain class IDs, which this compaction
    // remaps. No structural certificate survives a different ID epoch.
    shifted_body_matches_.clear();
    factored_selects_.clear();merged_selects_.clear();
    negated_products_.clear();
    combined_exponentials_.clear();
    coalesced_forests_.clear();
    flattened_folds_.clear();
    numeric_projection_.clear();exact_projection_.clear();predicate_projection_.clear();
    if(projection_scopes_ && sentryward::debug::environment("SENTRYWARD_EGRAPH_TRACE")) {
      std::size_t admitted=0;
      for(const auto& entry:projection_sat_certificates_)admitted+=entry.second;
      const auto scopes=projection_scopes_->statistics();
      SENTRYWARD_DEBUG_PRINT(egraph_76);
    }
    projection_body_heads_.reset();projection_affine_cache_.clear();
    // All source IDs are reassigned below. No descriptor source slot or
    // borrowed metadata view can survive compaction; the next physical
    // planner bootstrap remints exact descriptors under a new generation.
    complete_headers_.reset();
    if(++complete_header_generation_==0)throw std::overflow_error("complete header epoch overflow");
    projection_source_ids_.clear();projection_sat_sensitive_.clear();projection_branch_sensitive_.clear();
    projection_integer_sensitive_.clear();
    branch_candidates_.clear();
    // A compaction epoch reassigns node IDs. Only newly certified wrappers
    // may acquire a conditional predicate property in the new arena.
    branch_proved_wrappers_.clear();
    projection_max_samples_.clear();projection_max_membership_.clear();
    projection_sat_certificates_.clear();projection_sample_count_=0;
    projection_scopes_.reset();projection_root_scope_=lexical_scope::unknown;
    projection_sat_remaining_=2000000;projection_sat_requests_=0;
    std::vector<Class> mapped(old.size());
    for(std::size_t id=0;id<old.size();++id)if(live[id]) {
      poll();auto key=std::move(old[id]);
      for(auto& child:key.children)child=mapped[child];
      mapped[id]=intern(std::move(key));
    }
    for(auto& id:a)id=mapped[id];
    for(auto& id:b)id=mapped[id];
    if(branch_fact_anchor_)branch_fact_anchor_=live[*branch_fact_anchor_]?
        std::optional<Class>{mapped[*branch_fact_anchor_]}:std::nullopt;
    branch_scope_ids_.clear(); // Catalog IDs are epoch-local; retained wrappers own their old records.
  }
  Class normalizeExponentialRatio(Class id,std::string_view base) {
    const auto root=nodes_[id];
    if(!idealOperation(root) || root.name!="Divide" ||
        !finite_[find(root.children[0])] || !positive_[find(root.children[1])] ||
        !finite_[find(root.children[1])])return id;
    using Powers=std::map<Class,std::int64_t>;
    struct Term {Powers powers;Class weight;};
    std::vector<Term> numerator,denominator;
    std::size_t budget=4096;
    const auto operation=[&](std::string name,std::vector<Class> children) {
      Key key;key.kind=static_cast<int>(s::Kind::Operation);key.type=root.type;
      key.name=std::move(name);key.attributes="ideal_arithmetic";key.children=std::move(children);
      return intern(std::move(key));
    };
    const auto constant=[&](std::string text) {
      Key key;key.kind=static_cast<int>(s::Kind::Constant);key.type=root.type;key.name=std::move(text);
      return intern(std::move(key));
    };
    const auto literal=[&](Class value,const char* text) {
      const auto& key=nodes_[value];
      return key.kind==static_cast<int>(s::Kind::Constant) && key.type==root.type &&
          key.provenance.empty() && key.attributes.empty() && key.name==text;
    };
    const auto one=constant("1");
    std::function<bool(Class,std::int64_t,Powers&)> exponent;
    exponent=[&](Class value,std::int64_t sign,Powers& powers) {
      poll();if(!budget)return false;--budget;
      const auto key=nodes_[value];
      if(!finite_[find(value)] || key.type!=root.type || !key.provenance.empty())return false;
      if(literal(value,"0"))return true;
      if(idealOperation(key)) {
        if(key.name=="Add" || key.name=="Subtract")
          return exponent(key.children[0],sign,powers) &&
              exponent(key.children[1],key.name=="Add"?sign:-sign,powers);
        if(key.name=="Negate")return exponent(key.children[0],-sign,powers);
      }
      std::int64_t coefficient;
      if(__builtin_add_overflow(powers[value],sign,&coefficient))return false;
      if(coefficient)powers[value]=coefficient;else powers.erase(value);
      return true;
    };
    std::function<bool(Class,Powers,std::vector<Term>&)> collect;
    collect=[&](Class value,Powers powers,std::vector<Term>& terms) {
      poll();if(!budget || terms.size()>=128)return false;--budget;
      const auto key=nodes_[value];
      if(!finite_[find(value)] || key.type!=root.type || !key.provenance.empty())return false;
      if(literal(value,"0"))return true;
      if(idealOperation(key)) {
        if(key.name=="Add")return collect(key.children[0],powers,terms) && collect(key.children[1],powers,terms);
        if(key.name=="Multiply") {
          for(unsigned side=0;side<2;++side) {
            const auto factor=nodes_[key.children[side]];
            if(idealOperation(factor) && factor.name==base) {
              if(!exponent(factor.children[0],1,powers))return false;
              return collect(key.children[1-side],std::move(powers),terms);
            }
          }
        }
        if(key.name==base) {
          if(!exponent(key.children[0],1,powers))return false;
          terms.push_back({std::move(powers),one});return true;
        }
      }
      terms.push_back({std::move(powers),value});return true;
    };
    if(!collect(root.children[0],{},numerator) || !collect(root.children[1],{},denominator) ||
        numerator.empty() || denominator.empty())return id;
    Powers common;
    for(const auto* terms:{&numerator,&denominator})for(const auto& term:*terms)
      for(auto [atom,coefficient]:term.powers)common.emplace(atom,coefficient);
    bool changed=false;
    for(auto& [atom,coefficient]:common) {
      for(const auto* terms:{&numerator,&denominator})for(const auto& term:*terms) {
        auto found=term.powers.find(atom);
        coefficient=std::min(coefficient,found==term.powers.end()?0:found->second);
      }
      changed|=coefficient!=0;
    }
    if(!changed)return id;
    // Both sides are divided by the same strictly positive finite Exp2 factor.
    for(auto* terms:{&numerator,&denominator})for(auto& term:*terms) {
      for(auto [atom,coefficient]:common) {
        auto& current=term.powers[atom];
        if(__builtin_sub_overflow(current,coefficient,&current))return id;
      }
      std::erase_if(term.powers,[](const auto& item){return item.second==0;});
    }
    const auto build=[&](std::vector<Term>& terms) {
      std::sort(terms.begin(),terms.end(),[](const auto& a,const auto& b) {
        return std::tie(a.powers,a.weight)<std::tie(b.powers,b.weight);
      });
      auto sum=constant("0");
      for(const auto& term:terms) {
        auto power=constant("0");
        for(auto [atom,coefficient]:term.powers) {
          auto value=coefficient==1?atom:operation("Multiply",{constant(std::to_string(coefficient)),atom});
          power=literal(power,"0")?value:operation("Add",{power,value});
        }
        auto value=term.weight;
        if(!term.powers.empty()) {
          auto factor=operation(std::string(base),{power});
          value=literal(value,"1")?factor:operation("Multiply",{value,factor});
        }
        sum=literal(sum,"0")?value:operation("Add",{sum,value});
      }
      return sum;
    };
    auto top=build(numerator),bottom=build(denominator);
    return operation("Divide",{top,bottom});
  }

  bool numericOutput(Class id) const {
    const auto& k=nodes_[id];
    if(!k.provenance.empty())return false;
    if(realType(k.type))return true;
    return k.kind==6 && k.provenance.empty() && k.children.size()==1 &&
        idealNumericOutputConversion(nodes_[k.children[0]].type,k.type,k.name);
  }

  Class numericOutputProducer(Class id) const {
    // Erase an approved conversion USE at the comparison root, not its e-class.
    // A shared producer's internal/predicate/exact consumers still project from
    // their original typed nodes. No global union or ancestor search is needed.
    // Import represents the bounded annotation chain as successive use nodes.
    // Only its outermost edge may be the same-width storage bitcast; never walk
    // through an integer producer, a mathematical op, or an unrecognized cast.
    std::size_t metadata=0;
    for(unsigned step=0;step<16;++step) {
      const auto& k=nodes_[id];
      if(k.kind!=6 || !k.provenance.empty() || k.children.size()!=1 ||
          (step && !realType(k.type)) || k.name.size()>128-metadata ||
          !idealNumericOutputConversion(nodes_[k.children[0]].type,k.type,k.name))break;
      metadata+=k.name.size();id=k.children[0];
    }
    return id;
  }

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
    bool charge(std::size_t amount=1) {
      engine.poll();
      if(exhausted || (remaining && amount>*remaining)) {
        // Exhausting a per-query budget does not invalidate earlier epoch
        // witnesses. The outer ledger, when exhausted, stays exhausted.
        if(remaining && amount>*remaining)*remaining=0;
        return false;
      }
      if(!query_guards.charge(amount))return false;
      if(remaining && remaining!=query_remaining && remaining!=mapper_remaining)*remaining-=amount;
      if(engine.options_.max_work && amount>1) {
        if(amount-1>engine.options_.max_work-engine.work_)throw Limit{"work_budget"};
        engine.work_+=amount-1;
      }
      return true;
    }
    bool cap(){exhausted=true;return false;}
    void initializeSoleHints(std::size_t retained_capacity=max_entries,
        bool complete_descriptors=false) {
      retained_entries=std::clamp(retained_capacity,max_entries,
          epoch_work_allowance::maximum_retained_headers);
      const auto bound=snapshot.size(),words=bound/64+(bound%64!=0);
      if(bound>std::numeric_limits<std::size_t>::max()/2)return;
      const auto header_slots=2*bound,header_words=header_slots/64+(header_slots%64!=0);
      if(words>(std::numeric_limits<std::size_t>::max()-header_words-3)/2 ||
          !charge(2*words+header_words+3))return;
      // No parentheses: fundamental Class slots are NOT value-initialized.
      // A validity bit is always tested before reading its corresponding slot.
      auto roots=std::unique_ptr<Class[]>(new Class[bound]);
      auto shallow=std::unique_ptr<Class[]>(new Class[bound]);
      auto headers=std::unique_ptr<Class[]>(new Class[header_slots]);
      sole_root_valid.resize(words,0);sole_shallow_valid.resize(words,0);
      source_header_valid.resize(header_words,0);
      sole_root_tags=std::move(roots);sole_shallow_tags=std::move(shallow);
      source_header_tags=std::move(headers);
      sole_bound=bound;sole_epoch=engine.unions_;
      // Full descriptor registration is an explicit preparation opt-in, not
      // a prerequisite for the original cheap local/sole-source token cache.
      // Prefix/local consumers must not bootstrap an unrelated metadata arena.
      // When requested, bootstrap charges THIS epoch/global work before a
      // query; later intern misses charge ordinary Engine work once.
      if(complete_descriptors && !adapter && engine.prepareCompleteHeaders(bound,*this))
        descriptor_generation=engine.complete_header_generation_;
    }
    std::optional<Class> sourceHeaderHint(Class id) {
      if(!source_header_tags || id/2>=sole_bound)return {};
      if(engine.unions_!=sole_epoch){cap();return {};}
      if(!charge(2))return {};
      if(!(source_header_valid[id/64]&(std::uint64_t{1}<<(id%64))))return {};
      ++source_header_hits;return source_header_tags[id];
    }
    bool retainSourceHeader(Class id,Class tag) {
      if(source_header_tags && id/2<sole_bound) {
        if(engine.unions_!=sole_epoch)return cap();
        if(!charge(2))return false;
        const auto bit=std::uint64_t{1}<<(id%64);
        source_header_tags[id]=tag;
        if(!(source_header_valid[id/64]&bit))++source_header_tokens;
        source_header_valid[id/64]|=bit;return true;
      }
      // Local/append-only callers keep the old bounded sparse representation.
      if(header_cache.size()<retained_entries)header_cache.emplace(id,tag);
      return true;
    }
    std::optional<Class> soleHint(Class cls,bool shallow) {
      if(cls>=sole_bound)return {};
      if(engine.unions_!=sole_epoch){cap();return {};}
      if(!charge(2))return {};
      const auto& valid=shallow?sole_shallow_valid:sole_root_valid;
      if(!(valid[cls/64]&(std::uint64_t{1}<<(cls%64))))return {};
      if(shallow){++sole_shallow_hits;return sole_shallow_tags[cls];}
      ++sole_root_hits;return sole_root_tags[cls];
    }
    bool rememberSoleHint(Class cls,Class tag,bool shallow) {
      if(cls>=sole_bound)return true; // Append-only projected IDs stay uncached.
      if(engine.unions_!=sole_epoch)return cap();
      if(!charge(2))return false;
      auto& valid=shallow?sole_shallow_valid:sole_root_valid;
      const auto bit=std::uint64_t{1}<<(cls%64);
      if(shallow) {
        sole_shallow_tags[cls]=tag;if(!(valid[cls/64]&bit))++sole_shallow_tokens;
      } else {
        sole_root_tags[cls]=tag;if(!(valid[cls/64]&bit))++sole_root_tokens;
      }
      valid[cls/64]|=bit;return true;
    }
    static bool bounded(const Key& k) {
      return k.type.size()<=128 && k.name.size()<=4096 && k.attributes.size()<=128 &&
          k.provenance.size()<=4096 && k.children.size()<=max_arity;
    }
    std::optional<IndexAffine> affine(Class cls) {
      cls=owner(cls);
      if(cls>=snapshot.size() ||
          (snapshot[cls].type!="u64" && snapshot[cls].type!="exact:u64"))return {};
      std::size_t available=remaining?*remaining:131072;
      available=query_guards.available(available);
      const auto before=available;
      auto result=adapter?adapter->affine(cls,affine_cache,available):
          engine.affineValue(cls,affine_cache,0,&available,0,64);
      const auto used=before-available;
      // The callee used at most the prechecked minimum. No global double
      // charge: affineValue already owns its actual primitive invoice.
      if(!query_guards.charge(used))return {};
      if(remaining && remaining!=query_remaining && remaining!=mapper_remaining)*remaining-=used;
      return result;
    }
    std::optional<Class> header(Class source_id,bool affine_address=false) {
      if(source_id>=snapshot.size())return {};
      // Explicit stable IDs: no pointer arithmetic, retained node reference or
      // header span may survive an append to the borrowed arena.
      const auto& source=snapshot[source_id];
      const auto id=source_id*2+affine_address;
      if(source_header_tags && source_id<sole_bound) {
        if(auto tag=sourceHeaderHint(id))return tag;
        if(exhausted || (remaining && !*remaining) || query_guards.exhausted())return {};
      } else {
        if(auto found=header_cache.find(id);found!=header_cache.end()) {
          if(!charge())return {};return found->second;
        }
      }
      if(!bounded(source))return {};
      std::optional<Class> descriptor;
      const HeaderKey* registered=nullptr;
      if(!affine_address && !adapter && descriptor_generation) {
        if(descriptor_generation!=engine.complete_header_generation_) {cap();return {};}
        if(!charge(2))return {};
        descriptor=engine.completeHeaderId(source_id);
        if(descriptor) {
          if(auto found=descriptor_tags.find(*descriptor);found!=descriptor_tags.end()) {
            ++descriptor_hits;
            if(!retainSourceHeader(id,found->second))return {};return found->second;
          }
          registered=engine.completeHeader(*descriptor);
          if(!registered)return {};
        }
      }
      auto probe=headerProbe(source);
      if(affine_address)probe={8,source.type,"affine-address-body-filter",{},{},{},0,0};
      else if(probe.fold)probe.fold->binder_level=0; // Scheduling only; full scope remains mandatory.
      const auto bytes=probe.type.size()+probe.name.size()+probe.attributes.size()+probe.provenance.size();
      // Hash the transient fields once. Owning entries retain this scalar
      // fingerprint, so rehash never rescans strings or throws mid-insertion.
      if(registered) {
        // The immutable Engine-owned descriptor already performed exact full
        // collision equality. Borrow it only for this operation, never retain
        // a view into nodes_ or substitute its birth ID for local tag order.
        if(!charge(8))return {};probe=headerProbe(registered->value);probe.hash=registered->hash;
      } else {
        if(!charge(8+bytes))return {};probe.hash=HeaderHash::compute(probe);
      }
      ++header_probes;
      try {
        // Equality charges every ACTUAL collision comparison. No string/vector
        // is copied on a hit; hash equality is never an equality proof.
        if(auto found=headers.find(probe);found!=headers.end()) {
          ++header_probe_hits;
          if(descriptor) {
            if(!charge(2))return {};descriptor_tags.emplace(*descriptor,found->second);++descriptor_translations;
          }
          if(!retainSourceHeader(id,found->second))return {};
          return found->second;
        }
        if(headers.size()>=retained_entries){cap();return {};}
        // One owned key plus the existing owned header_values copy, only on
        // a distinct-header miss. Child zeros are charged before construction.
        if(!charge(2*bytes+2*probe.arity+1))return {};
        Key key;key.kind=probe.kind;key.type=probe.type;key.name=probe.name;
        key.attributes=probe.attributes;key.provenance=probe.provenance;
        key.fold=probe.fold;key.index_level=probe.index_level;key.children.resize(probe.arity,0);
        const auto tag=header_values.size();header_values.push_back(key);
        bool inserted=false;Class retained_tag=tag;
        try {
          const auto entry=headers.emplace(HeaderKey{std::move(key),probe.hash},tag);
          inserted=entry.second;retained_tag=entry.first->second;
        } catch(...) {
          // Cached hash is noexcept; insertion has no effect if equality or
          // allocation throws. Keep the owned table/arena tag invariant even
          // on a local allowance denial or a propagated global limit.
          header_values.pop_back();throw;
        }
        if(!inserted) {
          header_values.pop_back();
          if(descriptor) {
            if(!charge(2))return {};descriptor_tags.emplace(*descriptor,retained_tag);++descriptor_translations;
          }
          if(!retainSourceHeader(id,retained_tag))return {};return retained_tag;
        }
        header_copied_bytes+=2*bytes;
        if(descriptor) {
          if(!charge(2))return {};descriptor_tags.emplace(*descriptor,tag);++descriptor_translations;
        }
        if(!retainSourceHeader(id,tag))return {};return tag;
      } catch(const HeaderBudget&) {return {};}
    }
    static unsigned priority(const Key& k) {
      if(k.kind==static_cast<int>(s::Kind::Constant))return 0;
      if(k.kind==static_cast<int>(s::Kind::Fold))return 1;
      if(k.kind==static_cast<int>(s::Kind::Operation)) {
        if(k.name=="Reciprocal" || k.name=="Negate")return 2;
        if(k.name=="Multiply")return 3;
        if(k.name=="Add")return 4;
        if(k.name=="Divide")return 5;
      }
      return 6;
    }
    bool preferredTag(Class a,Class b) const {
      const auto& x=header_values[a];const auto& y=header_values[b];
      return std::tuple{priority(x),x.kind,std::string_view{x.type},std::string_view{x.name},a}<
             std::tuple{priority(y),y.kind,std::string_view{y.type},std::string_view{y.name},b};
    }
    std::optional<Roots> roots(Class cls) {
      cls=owner(cls);
      if(auto found=roots_cache.find(cls);found!=roots_cache.end())return found->second;
      if(cls>=memberCount() || classMembers(cls).empty() || classMembers(cls).size()>max_members)return {};
      const bool sole=classMembers(cls).size()==1;
      if(sole) {
        if(auto tag=soleHint(cls,false))return Roots{classMembers(cls).front(),*tag};
        if(exhausted || (remaining && !*remaining) || query_guards.exhausted())return {};
      }
      // An affine address is a wildcard only in the filter. Its complete ring
      // coefficients, including captured indices, are checked by affineBodyMap.
      if((snapshot[cls].type=="u64" || snapshot[cls].type=="exact:u64") && affine(cls)) {
        auto tag=header(classMembers(cls).front(),true);if(!tag)return {};
        // The shared exact affine cache already owns this complete summary.
        // Repeated address headers need no second retained per-class hint.
        if(sole && !rememberSoleHint(cls,*tag,false))return {};
        return Roots{classMembers(cls).front(),*tag};
      }
      if((remaining && !*remaining) || query_guards.exhausted())return {};
      if(sole) {
        // One ACTUAL snapshot-prefix witness has no header choice to memoize.
        // Preserve roots()'s metadata/direct-self checks; an indirect cycle is
        // still rejected by the complete paired walk, never by this hint.
        const auto id=classMembers(cls).front();const auto& k=snapshot[id];
        if(!bounded(k) || !charge(k.children.size()))return {};
        if(std::any_of(k.children.begin(),k.children.end(),
            [&](Class child){return owner(child)==cls;}))return {};
        auto tag=header(id);if(!tag || !rememberSoleHint(cls,*tag,false))return {};
        return Roots{id,*tag};
      }
      if(roots_cache.size()>=retained_entries){cap();return {};}
      std::optional<Roots> result;
      for(auto id:classMembers(cls)) {
        if(!charge(snapshot[id].children.size()))return {};
        // Do not let a direct cyclic alias (notably Fold0(seed,...)=seed)
        // hide a normal scheduling shape. This is not an admission guard:
        // choices() still indexes these heads, and a chosen cyclic pair is
        // rejected by the actual per-side witness walk, with no retry.
        if(std::any_of(snapshot[id].children.begin(),snapshot[id].children.end(),
            [&](Class child){return owner(child)==cls;}))continue;
        auto tag=header(id);if(!tag)return {};
        if(!result || preferredTag(*tag,result->tag))result=Roots{id,*tag};
      }
      if(!result)return {};
      roots_cache.emplace(cls,*result);return result;
    }
    std::optional<Class> compact(Class cls) {
      cls=owner(cls);
      if(auto found=compact_cache.find(cls);found!=compact_cache.end())return found->second;
      if(cls<memberCount() && classMembers(cls).size()==1) {
        // There is NO scheduling alternative or child-tag tie to resolve.
        // Preserve exactly roots()/compact()'s local bounded/self guard, but
        // avoid constructing/caching irrelevant headers for a sole witness.
        // Indirect selected cycles remain the actual paired walker's duty.
        const auto id=classMembers(cls).front();
        if(id>=snapshot.size() || !bounded(snapshot[id]) ||
            !charge(1+snapshot[id].children.size()))return {};
        if(std::any_of(snapshot[id].children.begin(),snapshot[id].children.end(),
            [&](Class child){return owner(child)==cls;}))return {};
        return id;
      }
      if(compact_cache.size()>=retained_entries){cap();return {};}
      auto root=roots(cls);if(!root)return {};
      if(header_values[root->tag].kind==8) {
        compact_cache.emplace(cls,root->preferred);return root->preferred;
      }
      // Tie-break one common root header by immediate child headers only.
      // In particular Multiply(x,1/d) chooses an already stored Reciprocal
      // witness when available. No recursive compact()/choices() descent.
      Class result=root->preferred;std::optional<std::vector<Class>> best;
      for(auto id:classMembers(cls)) {
        const auto& k=snapshot[id];auto tag=header(id);if(!tag)return {};
        if(*tag!=root->tag)continue;
        if(!charge(k.children.size()))return {};
        if(std::any_of(k.children.begin(),k.children.end(),
            [&](Class child){return owner(child)==cls;}))continue;
        std::vector<Class> tags;tags.reserve(k.children.size());
        for(auto child:k.children) {auto part=roots(child);if(!part)return {};tags.push_back(part->tag);}
        if((arithmetic(k) || integerCommutative(k)) && tags.size()==2 && preferredTag(tags[1],tags[0]))
          std::swap(tags[0],tags[1]);
        bool less=!best;
        if(best)for(std::size_t i=0;i<tags.size();++i) {
          const auto& a=header_values[tags[i]];const auto& b=header_values[(*best)[i]];
          if(!charge(1+std::min(a.type.size(),b.type.size())+std::min(a.name.size(),b.name.size())))return {};
          if(tags[i]!=(*best)[i]){less=preferredTag(tags[i],(*best)[i]);break;}
        }
        if(less){result=id;best=std::move(tags);}
      }
      compact_cache.emplace(cls,result);return result;
    }
    std::optional<Class> preferred(Class cls) {
      return compact(cls);
    }
    std::optional<Class> shallow(Class cls) {
      cls=owner(cls);
      if(auto found=shallow_cache.find(cls);found!=shallow_cache.end())return found->second;
      const bool sole=cls<memberCount() && classMembers(cls).size()==1;
      if(sole) {
        if(auto tag=soleHint(cls,true))return tag;
        if(exhausted || (remaining && !*remaining) || query_guards.exhausted())return {};
      }
      if(!sole && shallow_cache.size()>=retained_entries){cap();return {};}
      auto root=roots(cls);if(!root)return {};
      auto head=preferred(cls);if(!head)return {};
      const auto& k=snapshot[*head];Signature signature{root->tag,{}};
      const bool address=header_values[root->tag].kind==8;
      const auto arity=address?std::size_t{0}:k.children.size();
      if(!charge(arity))return {};signature.second.reserve(arity<=2?3*arity:arity);
      if(arity<=2) {
        // Exactly two local edges, not a recursive shallow()/choices() walk.
        // Binary/unary roots contribute at most four terminal headers. A
        // child's header carries its arity, so these flattened blocks remain
        // unambiguous. Affine addresses stop at their arity-zero wildcard.
        std::array<std::vector<Class>,2> parts;
        for(std::size_t i=0;i<arity;++i) {
          const auto child=k.children[i];auto tag=roots(child);if(!tag)return {};
          parts[i].push_back(tag->tag);
          if(header_values[tag->tag].kind==8)continue;
          const auto child_arity=header_values[tag->tag].children.size();
          // Large-arity heads retain the old immediate hint. Never fan out
          // another layer beneath one, or recursively expand a grandchild.
          if(child_arity>2)continue;
          auto child_head=preferred(child);if(!child_head)return {};
          const auto& child_key=snapshot[*child_head];
          if(!charge(child_arity))return {};
          for(auto grandchild:child_key.children) {
            auto leaf=roots(grandchild);if(!leaf)return {};parts[i].push_back(leaf->tag);
          }
          if((arithmetic(child_key) || integerCommutative(child_key)) && child_arity==2 &&
              parts[i][2]<parts[i][1])std::swap(parts[i][1],parts[i][2]);
        }
        if((arithmetic(k) || integerCommutative(k)) && arity==2 && parts[1]<parts[0])
          std::swap(parts[0],parts[1]);
        if(!charge(3*arity))return {};
        for(std::size_t i=0;i<arity;++i)
          signature.second.insert(signature.second.end(),parts[i].begin(),parts[i].end());
      } else {
        // The existing bounded header-only behavior for larger arities is a
        // conservative scheduling hint, not a different proof procedure.
        for(auto child:k.children) {
          auto tag=roots(child);if(!tag)return {};signature.second.push_back(tag->tag);
        }
      }
      if(!charge((signature.second.size()+1)*(1+std::bit_width(shallow_keys.size()+1))))return {};
      auto found=shallow_keys.find(signature);
      if(found==shallow_keys.end()) {
        if(shallow_keys.size()>=retained_entries){cap();return {};}
        found=shallow_keys.emplace(std::move(signature),shallow_keys.size()).first;
      }
      // Both kinds retain the SAME complete bounded interned signature. Sole
      // classes need only a scalar token; multi-witness/ambiguous cache rows
      // and all distinct-signature/header/choice capacities stay unchanged.
      if(sole) {
        if(!rememberSoleHint(cls,found->second,true))return {};
      } else shallow_cache.emplace(cls,found->second);
      return found->second;
    }
    const Choices* choices(Class cls) {
      cls=owner(cls);
      if(auto found=choices_cache.find(cls);found!=choices_cache.end())return &found->second;
      if(cls>=memberCount() || classMembers(cls).empty() || classMembers(cls).size()>max_members)return nullptr;
      if(choices_cache.size()+root_choices_cache.size()>=retained_entries){cap();return nullptr;}
      Choices result;
      for(auto id:classMembers(cls)) {
        const auto& k=snapshot[id];auto tag=header(id);if(!tag)return nullptr;
        Signature signature{*tag,{}};
        if(!charge(k.children.size()))return nullptr;signature.second.reserve(k.children.size());
        for(auto child:k.children) {
          auto part=shallow(child);if(!part)return nullptr;signature.second.push_back(*part);
        }
        if((arithmetic(k) || integerCommutative(k)) && signature.second.size()==2 &&
            signature.second[1]<signature.second[0])std::swap(signature.second[0],signature.second[1]);
        if(!charge((signature.second.size()+1)*(1+std::bit_width(choice_keys.size()+1))))return nullptr;
        auto found=choice_keys.find(signature);
        if(found==choice_keys.end()) {
          if(choice_keys.size()>=retained_entries){cap();return nullptr;}
          found=choice_keys.emplace(std::move(signature),choice_keys.size()).first;
        }
        const auto binder=k.fold?k.fold->binder_level:std::numeric_limits<std::uint32_t>::max();
        auto [entry,inserted]=result.try_emplace(ChoiceKey{found->second,binder},Choice{id});
        if(inserted) {
          if(choice_entries>=retained_entries){cap();return nullptr;}++choice_entries;
        } else {
          const auto& previous=snapshot[entry->second.head];
          bool same=previous.fold==k.fold && previous.index_level==k.index_level;
          if(!charge(k.children.size()))return nullptr;
          for(std::size_t i=0;i<k.children.size();++i)same &= owner(previous.children[i])==owner(k.children[i]);
          entry->second.unique &= same;
        }
      }
      return &choices_cache.emplace(cls,std::move(result)).first->second;
    }
    std::optional<std::pair<Class,Class>> matchChoices(const Choices& x,const Choices& y,
        bool unique_root=false) {
      std::optional<std::pair<Class,Class>> result;std::optional<Class> best;
      auto p=x.begin(),q=y.begin();
      while(p!=x.end() && q!=y.end()) {
        if(!charge())return {};
        if(p->first.first<q->first.first){++p;continue;}
        if(q->first.first<p->first.first){++q;continue;}
        const auto signature=p->first.first;
        auto p_end=p,q_end=q;
        while(p_end!=x.end() && p_end->first.first==signature) {
          if(!charge())return {};++p_end;
        }
        while(q_end!=y.end() && q_end->first.first==signature) {
          if(!charge())return {};++q_end;
        }
        // A linear merge, not a variant-pair product. Prefer one unique exact
        // binder match. Only singleton unique buckets may pair alpha labels;
        // the actual chosen witness below still validates every lexical edge,
        // seed, capture and cycle. Never retry another head after it fails.
        std::optional<std::pair<Class,Class>> candidate;
        auto a=p,b=q;
        while(a!=p_end && b!=q_end) {
          if(!charge())return {};
          if(a->first.second<b->first.second){++a;continue;}
          if(b->first.second<a->first.second){++b;continue;}
          if(a->second.unique && b->second.unique) {
            if(unique_root && candidate)return {};
            candidate=std::pair{a->second.head,b->second.head};
            if(!unique_root)break;
          }
          ++a;++b;
        }
        if(!candidate && std::next(p)==p_end && std::next(q)==q_end &&
            p->second.unique && q->second.unique)
          candidate=std::pair{p->second.head,q->second.head};
        if(candidate) {
          if(unique_root && result)return {};
          auto tag=header(candidate->first);if(!tag)return {};
          if(!best || preferredTag(*tag,*best)) {
            result=*candidate;best=*tag;
          }
        }
        p=p_end;q=q_end;
      }
      return result;
    }
    const Choices* rootChoices(Class cls) {
      cls=owner(cls);
      if(auto found=root_choices_cache.find(cls);found!=root_choices_cache.end()) {
        if(!charge())return nullptr;return &found->second;
      }
      if(cls>=memberCount() || classMembers(cls).empty() || classMembers(cls).size()>max_members)return nullptr;
      if(choices_cache.size()+root_choices_cache.size()>=retained_entries){cap();return nullptr;}
      // At most max_members ACTUAL complete root headers. Child shallow
      // signatures are optional scheduling hints, not semantic premises.
      // A shared invariant may have no nonrecursive child header after a
      // certified Add/Subtract alias creates an SCC; the paired walk can
      // still certify that child as an independent same-class opaque leaf.
      // This cache shares the existing resident/entry/work allowances.
      Choices result;
      for(auto id:classMembers(cls)) {
        if(!charge(2+std::bit_width(result.size()+1)))return nullptr;
        const auto& k=snapshot[id];auto tag=header(id);if(!tag)return nullptr;
        const auto binder=k.fold?k.fold->binder_level:std::numeric_limits<std::uint32_t>::max();
        auto [entry,inserted]=result.try_emplace(ChoiceKey{*tag,binder},Choice{id});
        if(inserted) {
          if(choice_entries>=retained_entries){cap();return nullptr;}++choice_entries;
          continue;
        }
        const auto& previous=snapshot[entry->second.head];
        bool same=previous.fold==k.fold && previous.index_level==k.index_level;
        if(!charge(k.children.size()))return nullptr;
        for(std::size_t i=0;i<k.children.size();++i)
          same &= owner(previous.children[i])==owner(k.children[i]);
        entry->second.unique &= same;
      }
      return &root_choices_cache.emplace(cls,std::move(result)).first->second;
    }
    std::optional<std::pair<Class,Class>> canonicalRootPair(const Choices& x,const Choices& y) {
      // Alternate certified operators in ONE eclass are not alternate
      // meanings. Select one common complete root header by the existing
      // fixed priority, BEFORE examining child-witness uniqueness. A chosen
      // ambiguous header stays Unknown; never retry a less preferred header.
      // Both indexes have at most max_members entries. This is a linear
      // ordered merge, not a candidate-pair product or an ancestor walk.
      auto p=x.begin(),q=y.begin();
      auto p_begin=x.end(),p_end=x.end(),q_begin=y.end(),q_end=y.end();
      std::optional<Class> best;
      while(p!=x.end() && q!=y.end()) {
        if(!charge())return {};
        if(p->first.first<q->first.first){++p;continue;}
        if(q->first.first<p->first.first){++q;continue;}
        const auto tag=p->first.first;
        auto a=p,b=q;
        while(a!=x.end() && a->first.first==tag) {if(!charge())return {};++a;}
        while(b!=y.end() && b->first.first==tag) {if(!charge())return {};++b;}
        if(!best || preferredTag(tag,*best)) {
          best=tag;p_begin=p;p_end=a;q_begin=q;q_end=b;
        }
        p=a;q=b;
      }
      if(!best)return {};
      std::optional<std::pair<Class,Class>> result;
      auto a=p_begin,b=q_begin;
      while(a!=p_end && b!=q_end) {
        if(!charge())return {};
        if(a->first.second<b->first.second){++a;continue;}
        if(b->first.second<a->first.second){++b;continue;}
        if(a->second.unique && b->second.unique) {
          if(result)return {}; // Within-header binder variants are ambiguous.
          result=std::pair{a->second.head,b->second.head};
        }
        ++a;++b;
      }
      if(!result && std::next(p_begin)==p_end && std::next(q_begin)==q_end &&
          p_begin->second.unique && q_begin->second.unique)
        result=std::pair{p_begin->second.head,q_begin->second.head};
      return result;
    }
    std::optional<std::pair<Class,Class>> fixedFoldPair(const Choices& x,const Choices& y) {
      // Binder labels distinguish actual witnesses, not the meanings of
      // already-certified aliases in one eclass. If the old unique-bucket
      // route cannot choose, select ONE stored Fold pair deterministically.
      // No candidate product or retry: every seed/body/lexical edge below is
      // still proved by the ordinary paired walk. Adapter-owned IDs have no
      // Engine finite-property certificate and conservatively keep Unknown.
      if(adapter)return {};
      auto p=x.begin(),q=y.begin();std::optional<Class> best;
      std::optional<std::pair<Class,Class>> result;
      while(p!=x.end() && q!=y.end()) {
        if(!charge())return {};
        if(p->first.first<q->first.first){++p;continue;}
        if(q->first.first<p->first.first){++q;continue;}
        const auto tag=p->first.first;auto a=p,b=q;
        auto left=p->second.head,right=q->second.head;
        while(a!=x.end() && a->first.first==tag) {
          if(!charge())return {};left=std::min(left,a->second.head);++a;
        }
        while(b!=y.end() && b->first.first==tag) {
          if(!charge())return {};right=std::min(right,b->second.head);++b;
        }
        const auto& h=header_values[tag];
        if(h.kind==static_cast<int>(s::Kind::Fold) && h.fold && h.type=="real" &&
            h.attributes.empty() && h.provenance.empty() && h.children.size()==2 &&
            (!best || preferredTag(tag,*best))) {
          best=tag;result=std::pair{left,right};
        }
        p=a;q=b;
      }
      if(!result)return {};
      // Do not search another alias if this ONE actual witness is ineligible.
      for(auto id:{result->first,result->second}) {
        if(!charge(7) || id>=snapshot.size())return {};
        const auto& k=snapshot[id];
        if(!engine.finite_[owner(id)])return {};
        for(auto child:k.children)if(child>=snapshot.size() || snapshot[child].type!="real" ||
            !engine.finite_[owner(child)])return {};
      }
      return result;
    }
    std::optional<std::pair<Class,Class>> rootMatched(Class a,Class b) {
      if(exhausted || (remaining && !*remaining) || query_guards.exhausted())return {};
      const auto* rx=rootChoices(a);if(!rx)return {};const auto* ry=rootChoices(b);if(!ry)return {};
      // ONE canonical common root pair, never retry/Cartesian alternatives.
      // Uniqueness is required WITHIN the chosen complete root header, not
      // across already-certified Multiply/Divide or Rsqrt/Reciprocal aliases.
      // Actual children, affine leaves, binder captures and selected cycles
      // still pass through the SAME complete scoped witness walk below.
      auto result=canonicalRootPair(*rx,*ry);
      if(!result)result=fixedFoldPair(*rx,*ry);
      if(fallback_trace_remaining && *fallback_trace_remaining) {
        --*fallback_trace_remaining;
        SENTRYWARD_DEBUG_PRINT(egraph_77);
        if(result)SENTRYWARD_DEBUG_PRINT(egraph_78);
        else SENTRYWARD_DEBUG_PRINT(egraph_79);
        SENTRYWARD_DEBUG_PRINT(egraph_80);
      }
      return result;
    }
    bool localRootCompatible(Class a,Class b) {
      auto pair=rootMatched(a,b);if(!pair)return false;
      const auto& p=snapshot[pair->first];const auto& q=snapshot[pair->second];
      // Fixed ONE extra local edge (<=2 child headers), never recursive
      // signatures or a trial full proof. A Fold preserves seed/body order.
      if(p.children.size()>2 || p.children.size()!=q.children.size())return false;
      const auto compatible=[&](std::size_t i,std::size_t j) {
        // Address shapes stay wildcards at this scheduling boundary. Their
        // exact coefficients and captured binders are checked by the walk.
        auto x=roots(p.children[i]),y=roots(q.children[j]);
        if(!x || !y)return false;
        if(header_values[x->tag].kind==8 && header_values[y->tag].kind==8)
          return x->tag==y->tag;
        return bool(rootMatched(p.children[i],q.children[j]));
      };
      if(p.children.size()==2 && (arithmetic(p) || integerCommutative(p))) {
        const bool direct=compatible(0,0) && compatible(1,1);
        const bool reverse=compatible(0,1) && compatible(1,0);
        // Repeating either operand makes the two orientations the SAME
        // multiset of child-pair obligations, not two possible proofs. All
        // obligations still pass through the full scoped walk below.
        return direct!=reverse || (direct &&
            (owner(p.children[0])==owner(p.children[1]) ||
             owner(q.children[0])==owner(q.children[1])));
      }
      for(std::size_t i=0;i<p.children.size();++i)if(!compatible(i,i))return false;
      return true;
    }
    std::optional<std::pair<Class,Class>> matched(Class a,Class b,bool complete_root_first=false) {
      a=owner(a);b=owner(b);
      if(exhausted || (remaining && !*remaining) || query_guards.exhausted())return {};
      if(a<memberCount() && b<memberCount() &&
          classMembers(a).size()==1 && classMembers(b).size()==1) {
        // No alternative or within-header ambiguity exists on either side.
        // Equal COMPLETE headers select the SAME sole actual witnesses; this
        // is not equality by shape. Every child, original Fold seed/binder,
        // capture, affine coefficient and indirect cycle is still checked by
        // the ordinary paired walk below. Different headers keep the old path.
        const auto p=classMembers(a).front(),q=classMembers(b).front();
        if(!charge(2))return {};
        auto x=header(p),y=header(q);if(!x || !y)return {};
        if(*x==*y)return std::pair{p,q};
      }
      if(!adapter) {
        // Preserve a common unary wrapper before consulting child-shape
        // hints. Its child may be a Fold on one side and a stored Sum forest
        // on the other; that is for the scoped reduction proof, not this
        // fixed-radius witness selector. Expanded binary aliases must not
        // obscure an already available matching wrapper.
        if(auto direct=rootMatched(a,b)) {
          const auto& p=snapshot[direct->first];
          if(idealOperation(p) && p.children.size()==1)return direct;
        }
        if(exhausted || (remaining && !*remaining) || query_guards.exhausted())return {};
      }
      if(complete_root_first && !adapter) {
        // Original-rank scoped Sum dispatch has ALREADY run. Use the existing
        // canonical complete-header priority to select ONE actual pair before
        // preparing descendant hints. Ambiguity within the chosen header is
        // still Unknown; other certified aliases are not alternate meanings.
        // A Fold stays on its existing lexical route. No selected child proof
        // is retried, and no header match itself certifies a value or geometry.
        ++root_first_attempts;
        if(auto direct=rootMatched(a,b)) {
          const auto& p=snapshot[direct->first];
          if(p.kind!=static_cast<int>(s::Kind::Fold) && !p.fold) {
            ++root_first_hits;return direct;
          }
        }
        if(exhausted || (remaining && !*remaining) || query_guards.exhausted())return {};
      }
      const auto* x=choices(a);const auto* y=choices(b);
      if(x && y)if(auto direct=matchChoices(*x,*y))return direct;
      // Unknown child hints are not a failed full witness. Select ONE bounded
      // complete root pair, then run the SAME scoped proof exactly once.
      return rootMatched(a,b);
    }
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
  void completeHeaderCharge(std::size_t amount) {
    poll();
    if(options_.max_work && amount>1) {
      if(amount-1>options_.max_work-work_)throw Limit{"work_budget"};
      work_+=amount-1;
    }
  }
  bool prepareCompleteHeaders(std::size_t bound,BodyHeadIndex& heads) {
    if(heads.adapter || bound>nodes_.size() || heads.query_remaining)return false;
    if(!complete_headers_) {
      if(!heads.charge(1))return false;
      complete_headers_=std::make_unique<CompleteHeaderCatalog>(*this);
    }
    auto& catalog=*complete_headers_;
    struct RegistrationScope {
      CompleteHeaderCatalog& catalog;
      explicit RegistrationScope(CompleteHeaderCatalog& c,BodyHeadIndex& h):catalog(c){catalog.preparing=&h;}
      ~RegistrationScope(){catalog.preparing=nullptr;}
    } scope(catalog,heads);
    try {
      if(catalog.sources.capacity()<bound) {
        // One bulk scalar allocation for the original physical prefix. The
        // later append path retains amortized geometric growth accounting.
        catalog.charge(bound+catalog.sources.size()+1);catalog.sources.reserve(bound);
      }
      while(catalog.sources.size()<bound) {
        const auto id=catalog.sources.size();catalog.registerSource(id,nodes_[id]);
      }
      return true;
    } catch(const BodyHeadIndex::HeaderBudget&) {return false;}
  }
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
  bool rectanglePoll(std::size_t& work,std::size_t amount=1) {
    poll();
    if(work>rectangle_limit_ || amount>rectangle_limit_-work) {
      work=rectangle_limit_+1;return false;
    }
    // poll() already accounts for one unit. Other rectangle consumers may
    // have no outer aggregate ledger, so retain global work-budget coverage.
    if(options_.max_work && amount>1) {
      if(amount-1>options_.max_work-work_)throw Limit{"work_budget"};
      work_+=amount-1;
    }
    work+=amount;return true;
  }
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
      std::unordered_map<Class,Class>* import_replacements=nullptr) {
    // Sum_i Sum_j f(N*i+j) -> Sum_k f(k), only after EVERY dependent
    // index leaf proves this same affine substitution. No logical elements.
    if(root.kind!=static_cast<int>(s::Kind::Fold) || !root.fold ||
        root.fold->reducer!=s::Reducer::Sum || root.fold->count<=1 ||
        !root.attributes.empty() || !root.provenance.empty() ||
        !realType(root.type) || !finite_[find(id)] ||
        !zero_[find(root.children[0])] || flattened_folds_.contains(id))return;
    const auto body=find(root.children[1]);if(body>=members.size())return;
    std::size_t work=0;
    std::optional<Class> inner;
    for(auto candidate:members[body]) {
      if(!rectanglePoll(work))return;
      const auto& h=snapshot[candidate];
      if(h.fold && h.fold->reducer==s::Reducer::Sum && h.fold->count &&
          h.kind==root.kind && h.type==root.type && h.attributes.empty() && h.provenance.empty() &&
          h.fold->binder_level>root.fold->binder_level && zero_[find(h.children[0])]) {
        inner=candidate;break;
      }
    }
    if(!inner)return;const auto nested=snapshot[*inner];
    // Singleton binders already have their own lexical-substitution rule.
    // Do not change that representation while combining nontrivial blocks.
    if(nested.fold->count<=1)return;
    std::uint64_t count;
    if(__builtin_mul_overflow(root.fold->count,nested.fold->count,&count))return;
    const auto outer_axis=root.fold->binder_level,inner_axis=nested.fold->binder_level;
    AffineCache affine;
    struct Plan {Class source;Class head;std::optional<IndexAffine> address;};
    std::vector<Plan> plans;std::map<Class,unsigned char> state;
    std::vector<std::pair<Class,bool>> pending{{find(nested.children[1]),false}};
    bool mapped=false,dense=true;
    while(!pending.empty()) {
      if(!rectanglePoll(work) || state.size()>4096)return;
      auto [node,ready]=pending.back();pending.pop_back();node=find(node);
      if(node>=snapshot.size())return;
      if(node>=members.size() || members[node].empty())return;
      const auto head=members[node].front();
      if(ready){state[node]=2;plans.push_back({node,head,{}});continue;}
      if(auto found=state.find(node);found!=state.end()) {
        if(found->second==1)return;continue;
      }
      if(!dependsOn(node,outer_axis) && !dependsOn(node,inner_axis)){state[node]=2;continue;}
      // A union leader can be a recursive alternative (e.g. 0=x-x).
      // Rebuild the admitted finite head, never unfold the union leader.
      const auto& h=snapshot[head];
      if(h.kind==static_cast<int>(s::Kind::Constant)){state[node]=2;continue;}
      auto index=rectangleAffine(node,snapshot,members,affine,work);
      if(index) {
        const auto a=index->coefficients.contains(outer_axis)?index->coefficients.at(outer_axis):0;
        const auto b=index->coefficients.contains(inner_axis)?index->coefficients.at(inner_axis):0;
        std::uint64_t expected;
        if(__builtin_mul_overflow(b,nested.fold->count,&expected) || a!=expected)dense=false;
        if(!dense && root.fold->count!=2)return;
        mapped|=a!=0 || b!=0;state[node]=2;plans.push_back({node,head,std::move(index)});continue;
      }
      if(h.kind==static_cast<int>(s::Kind::Index) || h.children.empty())return;
      if(h.fold && (h.fold->binder_level==outer_axis || h.fold->binder_level==inner_axis))return;
      if(!rectanglePoll(work,h.children.size()))return;
      state[node]=1;pending.push_back({node,true});
      for(auto child:h.children)pending.push_back({find(child),false});
    }
    if(!mapped)return;
    const auto integer=[&](std::uint64_t n) {Key k;k.kind=static_cast<int>(s::Kind::Constant);
      k.type="exact:u64";k.name=std::to_string(n);return intern(std::move(k));};
    const auto operation=[&](const char* name,Class a,Class b) {
      Key k;k.kind=static_cast<int>(s::Kind::Operation);k.type="exact:u64";
      k.name=name;k.children={a,b};return intern(std::move(k));};
    const auto emit=[&](unsigned block)->std::optional<Class> {
    std::map<Class,Class> replacement;
    for(const auto& plan:plans) {
      if(!rectanglePoll(work,4+(plan.address?plan.address->coefficients.size()*8:
          snapshot[plan.head].children.size()*4)))return {};
      if(plan.address) {
        auto address=*plan.address;
        const auto outer=address.coefficients.contains(outer_axis)?address.coefficients.at(outer_axis):0;
        const auto inner=address.coefficients.contains(inner_axis)?address.coefficients.at(inner_axis):0;
        address.coefficients.erase(outer_axis);
        if(dense) {
          address.coefficients.erase(inner_axis);
          if(inner)address.coefficients.emplace(outer_axis,inner);
        } else address.constant+=block?outer:0; // exact u64 ring arithmetic
        Class value=integer(address.constant);
        for(auto [axis,coefficient]:address.coefficients) {
          Key k;k.kind=static_cast<int>(s::Kind::Index);k.type="exact:u64";k.index_level=axis;
          auto term=intern(std::move(k));
          if(coefficient!=1)term=operation("Multiply",term,integer(coefficient));
          value=operation("Add",value,term);
        }
        replacement.emplace(plan.source,value);
      } else {
        auto k=snapshot[plan.head];
        for(auto& child:k.children) {
          child=find(child);if(auto found=replacement.find(child);found!=replacement.end())child=found->second;
        }
        replacement.emplace(plan.source,intern(std::move(k)));
      }
    }
    auto result=dense?root:nested;if(dense)result.fold->count=count;
    result.children[1]=replacement.at(find(nested.children[1]));
    return intern(std::move(result));
    };
    auto first=emit(0);if(!first)return;
    if(dense) {
      if(import_replacements)import_replacements->emplace(id,*first);
      else rewrite(id,*first);
    }
    else {
      // Fixed binary partition law: two separated BLOCKS remain two Folds.
      // Never iterate over either block's elements or arbitrary outer counts.
      auto second=emit(1);if(!second)return;
      Key sum;sum.kind=static_cast<int>(s::Kind::Operation);sum.type=root.type;
      sum.name="Add";sum.attributes="ideal_arithmetic";
      sum.children={*first,*second};
      auto result=intern(std::move(sum));
      if(import_replacements)import_replacements->emplace(id,result);
      else rewrite(id,result);
    }
    flattened_folds_.insert(id);
  }
  void coalesceRectangularFolds(Class id,const Key& root,const std::vector<Key>& snapshot,
      const std::vector<std::vector<Class>>& members,
      const std::map<std::tuple<std::string,std::uint32_t,std::uint64_t>,std::vector<Class>>& targets) {
    // Bounded template certificate: scalar seeds retain their enclosing axes,
    // while bodies acquire exactly one more axis. No new enodes or elements.
    if(!root.provenance.empty() ||
        (!root.attributes.empty() && !(arithmetic(root) && root.name=="Add")))return;
    struct Scope {unsigned parent;RectangleAxis axis;std::uint64_t volume;unsigned depth;};
    struct Task {Class id;unsigned scope;};
    struct Piece {Class body;unsigned scope;};
    std::vector<Scope> scopes{{0,{0,1},1,0}};
    std::map<std::tuple<unsigned,std::uint32_t,std::uint64_t>,unsigned> interned;
    std::vector<Task> pending{{id,0}};std::vector<Piece> pieces;
    std::set<std::pair<Class,unsigned>> visited;
    std::optional<std::uint32_t> binder;
    std::uint64_t total=0;std::size_t work=0;
    while(!pending.empty()) {
      if(!rectanglePoll(work))return;
      auto t=pending.back();pending.pop_back();t.id=find(t.id);
      if(t.id>=members.size() || !finite_[t.id])return;
      if(zero_[t.id])continue;
      if(!visited.emplace(t.id,t.scope).second || visited.size()>4096)return;
      std::optional<Class> fold,add,leaf;
      for(auto member:members[t.id]) {
        if(!rectanglePoll(work))return;
        const auto& k=snapshot[member];
        if(k.type!=root.type || !k.provenance.empty())continue;
        if(k.kind==static_cast<int>(s::Kind::Fold) && k.attributes.empty() &&
            k.fold->reducer==s::Reducer::Sum && k.children.size()==2){fold=member;break;}
        if(arithmetic(k) && k.name=="Add" && !add)add=member;
        if(!leaf)leaf=member;
      }
      if(fold) {
        const auto& k=snapshot[*fold];const auto count=k.fold->count;
        pending.push_back({k.children[0],t.scope});
        if(!count)continue;
        if(scopes[t.scope].depth>=8 || (t.scope && k.fold->binder_level<=scopes[t.scope].axis.binder))return;
        if(count>UINT64_MAX/scopes[t.scope].volume)return;
        binder=binder?std::min(*binder,k.fold->binder_level):k.fold->binder_level;
        const auto key=std::tuple{t.scope,k.fold->binder_level,count};
        auto found=interned.find(key);unsigned child;
        if(found==interned.end()) {
          if(scopes.size()>=128)return;
          child=static_cast<unsigned>(scopes.size());interned.emplace(key,child);
          scopes.push_back({t.scope,{k.fold->binder_level,count},scopes[t.scope].volume*count,scopes[t.scope].depth+1});
        } else child=found->second;
        pending.push_back({k.children[1],child});
      } else if(add)for(auto child:snapshot[*add].children)pending.push_back({child,t.scope});
      else {
        if(!leaf || pieces.size()>=1024 || scopes[t.scope].volume>UINT64_MAX-total)return;
        total+=scopes[t.scope].volume;pieces.push_back({t.id,t.scope});
      }
    }
    if(!binder || !total || pieces.empty())return;
    const auto candidates=targets.find({root.type,*binder,total});if(candidates==targets.end())return;
    // Only a complete matching-volume stored forest receives this ledger.
    // No logical Fold element is visited. Keep small unrelated proofs at the
    // ordinary budget; this allowance is local to this coverage certificate.
    struct BudgetScope {
      std::size_t& slot;std::size_t previous;
      ~BudgetScope(){slot=previous;}
    } budget_scope{rectangle_limit_,rectangle_limit_};
    rectangle_limit_=std::min<std::size_t>(8388608,
        rectangle_work_limit+pieces.size()*8192);
    AffineCache cache;
    std::size_t head_remaining=0;
    BodyHeadIndex witness_heads(*this,snapshot,members,cache,&head_remaining);
    for(auto target:candidates->second) {
      if(!rectanglePoll(work))return;
      if(find(target)==find(id) || free_indices_[find(target)]!=free_indices_[find(id)])continue;
      struct Domain {
        std::uint64_t begin,stride,count;
        std::vector<std::pair<std::uint64_t,std::uint64_t>> digits;
      };
      std::vector<Domain> domains;bool valid=true;std::uint64_t lattice=1;
      for(const auto& piece:pieces) {
        std::vector<RectangleAxis> axes;
        for(auto scope=piece.scope;scope;scope=scopes[scope].parent)axes.push_back(scopes[scope].axis);
        std::reverse(axes.begin(),axes.end());
        auto map=rectangleBodyMap(snapshot[target].children[1],piece.body,*binder,axes,snapshot,members,cache,work,&witness_heads);
        if(!map || map->begin>=total){
          valid=false;break;
        }
        std::vector<std::pair<std::uint64_t,std::uint64_t>> strides;
        for(std::size_t i=0;i<axes.size();++i)if(axes[i].count>1)strides.emplace_back(map->coefficients[i],axes[i].count);
        std::sort(strides.begin(),strides.end());
        const auto step=strides.empty()?std::uint64_t{1}:strides.back().first;
        if(!step){valid=false;break;}
        // A rectangle can have holes: e.g. 256*i + 32*j + k,
        // i,j in [0,3), k in [0,32). Do not require dense AP flattening.
        // Superincreasing byte/index strides give a unique bounded digit
        // decomposition. The largest axis stays an interval; smaller axes
        // are tested symbolically at a bounded residue, never enumerated.
        std::uint64_t span=0;
        for(auto [stride,count]:strides) {
          std::uint64_t part;
          if(stride<=span || __builtin_mul_overflow(stride,count-1,&part) ||
              __builtin_add_overflow(span,part,&span)){valid=false;break;}
        }
        if(!valid)break;
        const auto count=strides.empty()?std::uint64_t{1}:strides.back().second;
        std::uint64_t next;
        if(span>=total-map->begin ||
            __builtin_mul_overflow(lattice/std::gcd(lattice,step),step,&next) || next>4096)
          {valid=false;break;}
        if(!strides.empty())strides.pop_back();
        lattice=next;domains.push_back({map->begin,step,count,std::move(strides)});
      }
      if(work>rectangle_limit_)return;
      if(!valid)continue;
      // A rectangular block may collapse to a strided AP, not only a dense
      // interval. Compare all APs in a bounded common residue lattice using
      // the existing exact weighted-interval engine. Work depends on stored
      // pieces and <=4096 residues, never their logical iteration counts.
      bounded_linear_reduction::Limits limits;
      limits.max_work=rectangle_limit_;
      bounded_linear_reduction::Builder coverage(limits,[&](std::size_t amount){return rectanglePoll(work,amount);});
      auto one=coverage.coefficient("1");if(!one)return;
      using IntervalShape=std::vector<std::pair<std::uint64_t,std::uint64_t>>;
      std::set<std::pair<IntervalShape,IntervalShape>> certified_residues;
      for(std::uint64_t residue=0;residue<lattice && valid;++residue) {
        std::vector<bounded_linear_reduction::WeightedInterval> actual,required;
        if(residue<total)required.push_back({0,1+(total-1-residue)/lattice,*one});
        for(const auto& d:domains) {
          if(!rectanglePoll(work,2+2*d.digits.size()))return;
          const auto delta=(residue+lattice-d.begin%lattice)%lattice;
          auto remainder=delta%d.stride;bool covered=true;
          for(auto it=d.digits.rbegin();it!=d.digits.rend();++it) {
            const auto digit=remainder/it->first;
            if(digit>=it->second){covered=false;break;}
            remainder%=it->first;
          }
          if(!covered || remainder)continue;
          const auto first=delta/d.stride;if(first>=d.count)continue;
          const auto start=(d.begin+delta)/lattice;
          const auto count=1+(d.count-1-first)/(lattice/d.stride);
          actual.push_back({start,start+count,*one});
        }
        // Adjacent residues frequently ask exactly the same interval proof.
        // Reuse that proof, not a representative sample of a different set.
        IntervalShape actual_shape,required_shape;
        for(const auto& interval:actual)actual_shape.emplace_back(interval.begin,interval.end);
        for(const auto& interval:required)required_shape.emplace_back(interval.begin,interval.end);
        auto signature=std::pair{std::move(actual_shape),std::move(required_shape)};
        if(!rectanglePoll(work,(1+actual.size()+required.size())*
            (1+std::bit_width(certified_residues.size()+1))))return;
        if(certified_residues.contains(signature))continue;
        auto equal=coverage.intervalsEqual(actual,required);
        if(!equal || !*equal)valid=false;
        else certified_residues.insert(std::move(signature));
      }
      if(valid){rewrite(id,target);coalesced_forests_.insert(id);return;}
    }
  }
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
      static std::atomic<std::uint64_t> next{1};auto value=next.load(std::memory_order_relaxed);
      for(;;) {
        if(!value)throw std::overflow_error("scoped context epoch overflow");
        if(next.compare_exchange_weak(value,value+1,std::memory_order_relaxed))return value;
      }
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
    // Thin actual-occurrence metadata only; Catalog deduplicates the exact
    // bounds. No persistent source-domain identity is minted per case.
    std::optional<ScopeView> bodyView(Id parent,Class fold) {
      if(parent>=scopes.size() || fold>UINT32_MAX || fold>=heads.snapshot.size() || !heads.charge(4))return {};
      const auto& k=heads.snapshot[fold];const auto cls=heads.owner(fold);
      if(k.kind!=static_cast<int>(s::Kind::Fold) || !k.fold || k.fold->reducer!=s::Reducer::Sum ||
          !k.fold->count || k.children.size()!=2 || k.type!="real" ||
          !k.attributes.empty() || !k.provenance.empty() ||
          !heads.engine.finite_[cls] || !heads.engine.zero_[heads.owner(k.children[0])])return {};
      auto b=bounds.bindBody(scopes[parent].bounds,k.fold->binder_level,k.fold->count);if(!b)return {};
      ScopeView v;v.owner=this;v.epoch=context_epoch;v.bounds=*b;v.parent=parent;v.body_fold=fold;
      ++original_body_views;return v;
    }
    bool valid(const ScopeView& v) const {
      return v.owner==this && v.epoch==context_epoch && v.parent<scopes.size() &&
          (!v.body_fold || *v.body_fold<heads.snapshot.size());
    }
    std::optional<std::uint64_t> count(const ScopeView& v,std::uint32_t level) const {
      return valid(v)?bounds.count(v.bounds,level):std::nullopt;
    }
    std::optional<std::pair<unsigned,std::uint64_t>> domain(const ScopeView& v,std::uint32_t level) const {
      if(!valid(v))return {};
      if(level<scopes[v.parent].domains.size())return std::pair{0u,std::uint64_t(scopes[v.parent].domains[level])};
      if(v.body_fold && level==scopes[v.parent].domains.size()) {
        // Distinct ORIGINAL Fold BODY occurrences never compare equal merely
        // by binder/count. This value token cannot alias a resident domain ID.
        return std::pair{1u,(std::uint64_t(v.parent)<<32)|*v.body_fold};
      }
      return {};
    }
    bool sameAssignment(Class cls,const ScopeView& reference,const ScopeView& actual,
        std::span<const std::uint32_t> covered) {
      if(!valid(reference) || !valid(actual))return false;
      cls=heads.owner(cls);const auto& free=heads.engine.free_indices_[cls];
      if(free.size()>64 || !heads.charge(free.size()+1))return false;
      for(auto level:free) {
        if(std::find(covered.begin(),covered.end(),level)!=covered.end())continue;
        auto r=domain(reference,level),a=domain(actual,level);
        auto rc=count(reference,level),ac=count(actual,level);
        if(!r || !a || r!=a || !rc || rc!=ac)return false;
      }
      return true;
    }
    std::optional<OccurrenceUse> bodyUse(const ScopeView& v,std::span<const unsigned> path) {
      if(!valid(v) || !v.body_fold || path.size()>8 || !heads.charge(path.size()+2))return {};
      const auto& fold=heads.snapshot[*v.body_fold];Class current=heads.owner(fold.children[1]);
      std::array<Class,8> active{};unsigned used=0;
      for(auto edge:path) {
        if(std::find(active.begin(),active.begin()+used,current)!=active.begin()+used)return {};
        active[used++]=current;
        auto selected=heads.compact(current);if(!selected)return {};
        const auto& k=heads.snapshot[*selected];
        // A further actual Fold BODY requires its OWN binding/view. No path
        // may silently cross a binder or reuse an ancestor's occurrence token.
        if(k.kind==static_cast<int>(s::Kind::Fold) || edge>=k.children.size())return {};
        current=heads.owner(k.children[edge]);
      }
      if(std::find(active.begin(),active.begin()+used,current)!=active.begin()+used)return {};
      const auto& k=heads.snapshot[current];
      if(k.type!="real" || !k.provenance.empty() || !heads.engine.finite_[current])return {};
      OccurrenceUse use;use.scope=v;use.body=current;use.path_size=path.size();
      std::copy(path.begin(),path.end(),use.body_path.begin());return use;
    }
    std::optional<Class> transport(const OccurrenceUse& use,Id canonical,std::uint32_t bound_axis) {
      if(!valid(use.scope) || !use.scope.body_fold || !heads.charge(3))return {};
      auto target=scopeView(canonical);if(!target)return {};
      const auto& fold=heads.snapshot[*use.scope.body_fold];
      const auto a=count(use.scope,bound_axis),r=count(*target,bound_axis);
      if(fold.fold->binder_level!=bound_axis || !a || a!=r ||
          bounds.rank(use.scope.bounds)!=bounds.rank(target->bounds) ||
          heads.engine.free_indices_[use.body].size()>8)return {};
      const std::array<std::uint32_t,1> mapped{bound_axis};
      // The SAME full matcher's identity-assignment guard. Only the actual
      // bound axis is explicitly unit/same-count mapped; EVERY captured axis
      // must retain its exact ORIGINAL occurrence and count.
      if(!sameAssignment(use.body,*target,use.scope,mapped))return {};
      ++transported_body_uses;return use.body;
    }
    std::optional<std::uint64_t> count(Id scope,std::uint32_t level) const {
      return scope<scopes.size()?bounds.count(scopes[scope].bounds,level):std::nullopt;
    }
    std::optional<Id> output() {
      const auto& o=heads.engine.options_;
      if(o.outer_binders>max_depth || o.outer_index_extents.size()!=o.outer_binders ||
          !heads.charge(o.outer_binders+1))return {};
      if(!scopes.empty())return Id{0};
      auto root_bounds=bounds.root(o.outer_binders,o.outer_index_extents);if(!root_bounds)return {};
      Scope root{*root_bounds,{}};
      for(auto count:o.outer_index_extents) {
        if(!count)return {};root.domains.push_back(next_domain++);
      }
      identity_slots+=root.domains.size()+1;
      scopes.push_back(std::move(root));return Id{0};
    }
    std::optional<Id> insideFold(Id parent,Class head) {
      if(parent>=scopes.size() || head>=heads.snapshot.size() || !heads.charge())return {};
      const auto& k=heads.snapshot[head];
      if(k.kind!=static_cast<int>(s::Kind::Fold) || !k.fold ||
          k.children.size()!=2 || !k.fold->count ||
          !k.attributes.empty() || !k.provenance.empty())return {};
      const auto key=std::pair{parent,head};
      if(!heads.charge(1+std::bit_width(body_scopes.size()+1)))return {};
      if(auto found=body_scopes.find(key);found!=body_scopes.end())return found->second;
      auto nested=bind(parent,k.fold->binder_level,k.fold->count);if(!nested)return {};
      const auto id=static_cast<Id>(scopes.size());scopes.push_back(std::move(*nested));
      body_scopes.emplace(key,id);return id;
    }
    std::optional<Scope> bind(Id parent,std::uint32_t binder,std::uint64_t count) {
      if(parent>=scopes.size() || !count || !retainedCapacity(scopes.size(),"occurrence_scope"))return {};
      const auto rank=bounds.rank(scopes[parent].bounds);
      if(!rank || *rank>max_depth)return {};
      if(identity_slots>max_identity_slots || *rank+2>max_identity_slots-identity_slots) {
        // TRACE-only first retained identity rejection. Read scalar metadata;
        // no expression/domain walk, proof charge, or admission change.
        if(!identity_cap_reported && sentryward::debug::environment("SENTRYWARD_EGRAPH_TRACE")) {
          identity_cap_reported=true;const auto stats=bounds.statistics();
          SENTRYWARD_DEBUG_PRINT(egraph_81);
        }
        return {};
      }
      if(!heads.charge(*rank+1))return {};
      auto next=bounds.bindBody(scopes[parent].bounds,binder,count);if(!next)return {};
      // Only <=8 source identity fields are copied; count storage is Catalog's.
      Scope nested{*next,scopes[parent].domains,parent};nested.domains.push_back(next_domain++);
      identity_slots+=nested.domains.size()+1;return nested;
    }
    std::optional<Id> prefixScope(Id scope,std::uint32_t rank) {
      // Follow <=8 ACTUAL scope metadata edges. Never move a use into a
      // deeper scope: a future internal Fold binder could otherwise capture.
      for(unsigned depth=0;depth<=max_depth;++depth) {
        if(scope>=scopes.size() || !heads.charge())return {};
        const auto current=bounds.rank(scopes[scope].bounds);if(!current || *current<rank)return {};
        if(*current==rank)return scope;
        if(!scopes[scope].parent)return {};scope=*scopes[scope].parent;
      }
      return {};
    }
    std::optional<Id> insideSourceFold(Id parent,const s::Graph& graph,unsigned graph_tag,s::Id head) {
      if(parent>=scopes.size() || head>=graph.size() || !heads.charge())return {};
      const auto& k=graph.node(head);
      if(k.kind!=s::Kind::Fold || !k.fold || k.operands.size()!=2 || !k.fold->count ||
          !k.attributes.empty())return {};
      const auto key=std::tuple{parent,graph_tag,head};
      if(!heads.charge(1+std::bit_width(source_scopes.size()+1)))return {};
      if(auto found=source_scopes.find(key);found!=source_scopes.end())return found->second;
      auto nested=bind(parent,k.fold->binder_level,k.fold->count);if(!nested)return {};
      const auto id=static_cast<Id>(scopes.size());scopes.push_back(std::move(*nested));source_scopes.emplace(key,id);return id;
    }
    bool boundedAffine(const IndexAffine& affine,Id scope) {
      if(scope>=scopes.size() || affine.coefficients.size()>64 ||
          !heads.charge(affine.coefficients.size()+1))return false;
      auto maximum=affine.constant;
      for(auto [level,coefficient]:affine.coefficients) {
        const auto extent=count(scope,level);if(!extent)return false;
        std::uint64_t term;
        if(__builtin_mul_overflow(coefficient,*extent-1,&term) ||
            __builtin_add_overflow(maximum,term,&maximum))return false;
      }
      return true;
    }
    // Private geometry handle: the interval is read from an actual original
    // BODY domain and a reachable pure exact-u64 unit-coordinate witness.
    // No caller interval/extent or selected expression at widened bounds.
    std::optional<ImageId> image(Id scope,std::uint32_t level,Class coordinate) {
      if(scope>=scopes.size() ||
          !heads.charge(1+std::bit_width(image_keys.size()+1)))return {};
      coordinate=heads.owner(coordinate);const auto key=std::tuple{scope,level,coordinate};
      if(auto it=image_keys.find(key);it!=image_keys.end())return it->second;
      if(!imageCapacity("unit_image"))return {};
      auto affine=heads.affine(coordinate);auto extent=count(scope,level);
      if(!extent || !affine || affine->coefficients.size()!=1 ||
          !affine->coefficients.contains(level) || affine->coefficients.at(level)!=1 ||
          !boundedAffine(*affine,scope))return {};
      std::uint64_t end;if(__builtin_add_overflow(affine->constant,*extent,&end))return {};
      ImageId id;id.owner=this;id.id=images.size();
      images.push_back({scope,level,coordinate,affine->constant,end});image_keys.emplace(key,id);return id;
    }
    // Mint only from TWO admitted original outer unit images under the SAME
    // actual lexical assignment. The difference is read from their checked
    // affine constants, not guessed from a frame's declared begin. Neither
    // coordinate has another free/captured coefficient, and both full ranges
    // were checked for exact-u64 wrap by image().
    std::optional<CommonTranslationId> commonTranslation(ImageId source,ImageId actual) {
      auto s=view(source),a=view(actual);
      if(!s || !a || s->derived || a->derived || s->scope!=a->scope || s->level!=a->level ||
          scopes[s->scope].parent || s->end-s->begin!=a->end-a->begin ||
          s->end-s->begin!=count(s->scope,s->level) || !heads.charge(12))return {};
      const auto key=std::pair{source.id,actual.id};
      if(!heads.charge(1+std::bit_width(common_translation_keys.size()+1)))return {};
      if(auto f=common_translation_keys.find(key);f!=common_translation_keys.end())return f->second;
      if(!retainedCapacity(common_translations.size(),"common_translation") || !heads.charge(12))return {};
      CommonTranslationId id;id.owner=this;id.epoch=context_epoch;id.id=common_translations.size();
      common_translations.push_back({source,actual,s->begin>=a->begin,
          s->begin>=a->begin?s->begin-a->begin:a->begin-s->begin});
      common_translation_keys.emplace(key,id);return id;
    }
    const CommonTranslation* commonTranslation(CommonTranslationId id) {
      if(!heads.charge() || id.owner!=this || id.epoch!=context_epoch || id.id>=common_translations.size())return nullptr;
      return &common_translations[id.id];
    }
    // Keep the actual LOCAL unit witness. Only its private image interval is
    // translated, with the SAME original outer assignment/captures, checked
    // containment and no-wrap arithmetic. Future sampling still proves ALL
    // P/Q/C numeric leaves through their original scopes and local assignment.
    std::optional<ImageId> commonImage(CommonTranslationId translation,ImageId local) {
      const auto* t=commonTranslation(translation);auto a=view(local);
      if(!t || !a || a->derived || !heads.charge(12))return {};
      // Copy before images append, so no borrowed vector element is retained.
      const auto output=images[t->actual.id],source=images[t->source.id];
      if(!prefix(output.scope,a->scope) || a->begin<output.begin || a->end>output.end ||
          a->end-a->begin!=count(a->scope,a->level))return {};
      // Identity changes neither the actual witness nor its bounds. All
      // owner/epoch/original-scope/domain/containment checks above still run;
      // do not allocate derived metadata or reroute existing point transport.
      if(!t->difference)return local;
      const auto key=std::pair{translation.id,local.id};
      if(!heads.charge(1+std::bit_width(common_image_keys.size()+1)))return {};
      if(auto f=common_image_keys.find(key);f!=common_image_keys.end())return f->second;
      if(!imageCapacity("common_image"))return {};
      std::uint64_t begin,end;
      if(t->add) {
        if(__builtin_add_overflow(a->begin,t->difference,&begin) ||
            __builtin_add_overflow(a->end,t->difference,&end))return {};
      } else {
        if(a->begin<t->difference || a->end<t->difference)return {};
        begin=a->begin-t->difference;end=a->end-t->difference;
      }
      if(begin>=end || begin<source.begin || end>source.end || !heads.charge(18))return {};
      ImageId id;id.owner=this;id.id=images.size();
      images.push_back({a->scope,a->level,a->coordinate,begin,end,true});
      common_image_keys.emplace(key,id);common_image_translations.emplace(id.id,translation.id);return id;
    }
    bool commonImagePair(CommonTranslationId translation,ImageId a,ImageId b) {
      const auto* t=commonTranslation(translation);if(!t)return false;
      if(!t->difference) {
        // No derived record exists for identity images. Validate BOTH original
        // images through the same owned token; a foreign scope/shifted image
        // cannot become a common order-preserving assignment by equal length.
        auto x=commonImage(translation,a),y=commonImage(translation,b);
        return x && y && *x==a && *y==b;
      }
      if(a.owner!=this || b.owner!=this ||
          !heads.charge(3+2*std::bit_width(common_image_translations.size()+1)))return false;
      const auto x=common_image_translations.find(a.id),y=common_image_translations.find(b.id);
      return x!=common_image_translations.end() && y!=common_image_translations.end() &&
          x->second==translation.id && y->second==translation.id;
    }
    bool commonOutputAssignment(CommonTranslationId translation,Class source,Class actual,Id scope) {
      const auto* t=commonTranslation(translation);if(!t || !heads.charge(4))return false;
      const auto& s=images[t->source.id];const auto& a=images[t->actual.id];
      return s.scope==scope && a.scope==scope && s.coordinate==heads.owner(source) && a.coordinate==heads.owner(actual);
    }
    std::optional<PointId> point(ImageId image,Class coordinate,Id actual_scope) {
      if(image.owner!=this || image.id>=images.size() || actual_scope>=scopes.size() ||
          !heads.charge())return {};
      coordinate=heads.owner(coordinate);
      const auto piece=images[image.id];auto global=heads.affine(coordinate);
      if(!global || !global->coefficients.empty() || !boundedAffine(*global,actual_scope) ||
          global->constant<piece.begin || global->constant>=piece.end)return {};
      // Both endpoints are checked above; subtraction cannot wrap. This
      // value handle is owned by the same immutable registry epoch. It creates
      // no retained per-update point table and copies only fixed scalar fields.
      PointId id;id.owner=this;id.image=image.id;id.coordinate=coordinate;
      id.actual_scope=actual_scope;id.local=global->constant-piece.begin;return id;
    }
    std::optional<std::uint64_t> globalValue(const ScalarPoint& point) {
      if(auto literal=std::get_if<Class>(&point)) {
        if(*literal>=heads.snapshot.size())return {};
        auto value=heads.affine(*literal);
        return value && value->coefficients.empty()?std::optional{value->constant}:std::nullopt;
      }
      if(auto p=std::get_if<TypedPointId>(&point)) {
        if(p->owner!=this || p->epoch!=context_epoch || p->image>=images.size() ||
            p->cell>=heads.snapshot.size() || p->address>=heads.snapshot.size() ||
            p->scope>=scopes.size() || !heads.charge(8+std::bit_width(typed_time_sources.size()+1)))return {};
        auto source=typed_time_sources.find(p->image);if(source==typed_time_sources.end())return {};
        auto assignment=context(source->second.context);
        return assignment && assignment->actual_scope==p->scope?std::optional{p->value}:std::nullopt;
      }
      const auto& p=std::get<DispatchPointId>(point);
      if(p.owner!=this || p.epoch!=context_epoch || p.image>=images.size() ||
          p.root>=heads.snapshot.size() || p.branch>=heads.snapshot.size() || !heads.charge(5))return {};
      const auto& source=images[p.image];
      if(source.derived || p.value<source.begin || p.value>=source.end)return {};
      return p.value;
    }
    std::optional<Class> pointWitness(const ScalarPoint& point) {
      if(auto literal=std::get_if<Class>(&point))
        return *literal<heads.snapshot.size()?std::optional{heads.owner(*literal)}:std::nullopt;
      if(!globalValue(point))return {};
      if(auto p=std::get_if<TypedPointId>(&point))return p->address;
      return images[std::get<DispatchPointId>(point).image].coordinate;
    }
    std::optional<PointId> point(ImageId image,const ScalarPoint& coordinate,Id actual_scope) {
      if(auto literal=std::get_if<Class>(&coordinate))return point(image,*literal,actual_scope);
      if(auto source=std::get_if<TypedPointId>(&coordinate)) {
        auto global=globalValue(coordinate);
        if(!global || image.owner!=this || image.id>=images.size() || actual_scope>=scopes.size() ||
            !prefix(source->scope,actual_scope) || !heads.charge(6))return {};
        for(std::uint32_t axis=scopes[source->scope].domains.size();axis<scopes[actual_scope].domains.size();++axis)
          if(!heads.charge(2) || heads.depends(source->cell,axis) || heads.depends(source->address,axis))return {};
        if(!heads.charge(4+2*std::bit_width(typed_time_sources.size()+1)))return {};
        const auto piece=images[image.id];auto role=typed_time_sources.find(image.id);
        auto original=typed_time_sources.find(source->image);
        // Equal numeric point values alone never identify two time roles.
        // Every target tile must use the SAME actual anchor/header lattice.
        if(role==typed_time_sources.end() || original==typed_time_sources.end() ||
            role->second.anchor!=original->second.anchor || role->second.stride!=original->second.stride ||
            role->second.anchor_constant!=original->second.anchor_constant ||
            *global<piece.begin || *global>=piece.end)return {};
        auto a=context(role->second.context),b=context(original->second.context);
        if(!a || !b || a->actual_scope!=b->actual_scope)return {};
        PointId id;id.owner=this;id.image=image.id;id.coordinate=source->address;
        id.actual_scope=actual_scope;id.local=*global-piece.begin;id.dispatch=true;return id;
      }
      auto global=globalValue(coordinate);const auto& source=std::get<DispatchPointId>(coordinate);
      if(!global || image.owner!=this || image.id>=images.size() || actual_scope>=scopes.size() ||
          !prefix(images[source.image].scope,actual_scope) || !heads.charge(6))return {};
      const auto& piece=images[image.id];
      if(*global<piece.begin || *global>=piece.end)return {};
      PointId id;id.owner=this;id.image=image.id;id.coordinate=images[source.image].coordinate;
      id.actual_scope=actual_scope;id.local=*global-piece.begin;id.dispatch=true;return id;
    }
    std::optional<Image> view(ImageId image) {
      if(image.owner!=this || image.id>=images.size() || !heads.charge())return {};
      return images[image.id]; // Fixed scalar copy, never a reconstructed scope.
    }
    bool prefix(Id parent,Id child) {
      if(parent>=scopes.size() || child>=scopes.size() ||
          scopes[parent].domains.size()>scopes[child].domains.size() ||
          !heads.charge(scopes[parent].domains.size()+1))return false;
      for(std::uint32_t level=0;level<scopes[parent].domains.size();++level)
        if(scopes[parent].domains[level]!=scopes[child].domains[level] ||
            count(parent,level)!=count(child,level))return false;
      return true;
    }
    std::optional<PointId> pointIn(PointId p,Id reference_scope,Id actual_scope) {
      if(p.owner!=this || p.image>=images.size() ||
          !prefix(reference_scope,images[p.image].scope) || !prefix(actual_scope,p.actual_scope))return {};
      const auto original=images[p.image];
      // These derived images, unlike a byte-address image, retain a proved
      // UNIT coordinate witness. They may rebase ONLY to original scope
      // prefixes via the same checked common assignment factory. Typed-cell
      // images below keep their stricter unchanged no-reinterpretation rule.
      auto translation=common_image_translations.end();
      if(original.derived && !common_image_translations.empty()) {
        if(!heads.charge(1+std::bit_width(common_image_translations.size()+1)))return {};
        translation=common_image_translations.find(p.image);
      }
      if(translation!=common_image_translations.end()) {
        CommonTranslationId token;token.owner=this;token.epoch=context_epoch;token.id=translation->second;
        auto local=image(reference_scope,original.level,original.coordinate);
        auto image_here=local?commonImage(token,*local):std::nullopt;
        if(!image_here)return {};
        if(!p.dispatch)return point(*image_here,p.coordinate,actual_scope);
        auto target=view(*image_here);std::uint64_t global;
        if(!target || p.local>=original.end-original.begin ||
            __builtin_add_overflow(original.begin,p.local,&global) || global<target->begin || global>=target->end ||
            !heads.charge(8))return {};
        PointId result=p;result.image=image_here->id;result.actual_scope=actual_scope;
        result.local=global-target->begin;return result;
      }
      if(p.dispatch) {
        if(!heads.charge(8))return {};
        // The private point already proves this scalar assignment, not the
        // value of its reachable source-coordinate witness. Rebase only the
        // owned reference image and actual ORIGINAL scope prefixes.
        std::optional<ImageId> image_here;
        if(original.derived) {
          if(reference_scope!=original.scope)return {};
          ImageId here;here.owner=this;here.id=p.image;image_here=here;
        } else image_here=image(reference_scope,original.level,original.coordinate);
        auto target=image_here?view(*image_here):std::nullopt;
        std::uint64_t global;
        if(!target || p.local>=original.end-original.begin ||
            __builtin_add_overflow(original.begin,p.local,&global) || global<target->begin || global>=target->end)return {};
        PointId result=p;result.image=image_here->id;result.actual_scope=actual_scope;
        result.local=global-target->begin;return result;
      }
      // Derived images retain their actual typed-cell proof. Never reinterpret
      // its byte-address witness as a unit graph coordinate in a new scope.
      if(original.derived) {
        if(reference_scope!=original.scope)return {};
        ImageId image_here;image_here.owner=this;image_here.id=p.image;
        return point(image_here,p.coordinate,actual_scope);
      }
      auto image_here=image(reference_scope,original.level,original.coordinate);
      return image_here?point(*image_here,p.coordinate,actual_scope):std::nullopt;
    }
    bool validate(const Context& c) {
      if(heads.adapter || c.reference_scope>=scopes.size() || c.actual_scope>=scopes.size() ||
          c.axis_count>c.axes.size() || !heads.charge(c.axis_count+1))return false;
      const auto& reference=scopes[c.reference_scope];const auto& actual=scopes[c.actual_scope];
      std::optional<std::uint32_t> previous_reference;
      for(unsigned i=0;i<c.axis_count;++i) {
        ++coordinate_checks;
        const auto& a=c.axes[i];
        if(a.reference_level>=reference.domains.size() || !a.extent ||
            (previous_reference && a.reference_level<=*previous_reference))return false;
        previous_reference=a.reference_level;
        if(a.actual_shift && !a.actual_level)return false;
        const auto extent=count(a.actual_shift?c.actual_scope:c.reference_scope,
            a.actual_shift?*a.actual_level:a.reference_level);
        if(!extent || a.offset>=*extent || a.extent>*extent-a.offset)return false;
        std::optional<IndexAffine> coordinate;
        if(a.point) {
          if(a.coordinate_image || a.actual_level || a.actual_shift || a.point->owner!=this || a.point->image>=images.size() ||
              !heads.charge())return false;
          const auto& p=*a.point;const auto& image=images[p.image];
          if(image.scope!=c.reference_scope || image.level!=a.reference_level ||
              p.actual_scope!=c.actual_scope || p.coordinate!=heads.owner(a.actual_coordinate) ||
              p.local>=image.end-image.begin)return false;
          coordinate.emplace();coordinate->constant=p.local;
        } else if(a.coordinate_image) {
          auto image=view(*a.coordinate_image);
          if(!image || !a.actual_level ||
              image->scope!=(a.actual_shift?c.reference_scope:c.actual_scope) ||
              image->level!=(a.actual_shift?a.reference_level:*a.actual_level) ||
              image->coordinate!=heads.owner(a.actual_coordinate))return false;
          // Borrow a checked one-term logical image; no graph node is created
          // and the original narrow BODY scope is never widened.
          coordinate.emplace();coordinate->constant=image->begin;
          coordinate->coefficients.emplace(image->level,1);
        } else coordinate=heads.affine(a.actual_coordinate);
        if(!coordinate || !boundedAffine(*coordinate,a.actual_shift?c.reference_scope:c.actual_scope) ||
            coordinate->constant!=a.offset)return false;
        if(a.actual_level) {
          // Both axes are already live in their ORIGINAL lexical scopes.
          // Their semantic roles may be permuted (e.g. finite Sum interchange),
          // but the assignment must remain injective. Scope BODY extension
          // still requires a fresh binder beyond every existing live axis.
          for(unsigned j=0;j<i;++j)
            if(c.axes[j].actual_level==a.actual_level)return false;
          if(*a.actual_level>=actual.domains.size() ||
              a.extent!=count(a.actual_shift?c.reference_scope:c.actual_scope,
                  a.actual_shift?a.reference_level:*a.actual_level) ||
              coordinate->coefficients.size()!=1 ||
              !coordinate->coefficients.contains(a.actual_shift?a.reference_level:*a.actual_level) ||
              coordinate->coefficients.at(a.actual_shift?a.reference_level:*a.actual_level)!=1)return false;
        } else if(a.extent!=1 || !coordinate->coefficients.empty())return false;
      }
      return true;
    }
    bool lessContext(const Context& a,const Context& b) {
      // Intern construction is O(log stored contexts), not an uncharged hash
      // collision-chain walk. Every comparison is precharged and bounded.
      if(!heads.charge(4))throw ContextCap{};
      const auto ah=std::tie(a.reference_scope,a.actual_scope,a.axis_count);
      const auto bh=std::tie(b.reference_scope,b.actual_scope,b.axis_count);
      if(ah!=bh)return ah<bh;
      for(unsigned i=0;i<a.axis_count;++i) {
        if(!heads.charge(20))throw ContextCap{};
        const auto& x=a.axes[i];const auto& y=b.axes[i];
        const auto xh=std::tuple{x.reference_level,x.actual_level,x.offset,x.extent,x.actual_coordinate,
            x.actual_shift,bool(x.point),bool(x.coordinate_image)};
        const auto yh=std::tuple{y.reference_level,y.actual_level,y.offset,y.extent,y.actual_coordinate,
            y.actual_shift,bool(y.point),bool(y.coordinate_image)};
        if(xh!=yh)return xh<yh;
        // validate() admitted all private owners before interning. They are
        // this registry, so these complete varying owned fields finish the key.
        if(x.point) {
          if(!heads.charge(12))throw ContextCap{};
          const auto xp=std::tie(x.point->image,x.point->coordinate,x.point->actual_scope,x.point->local,x.point->dispatch);
          const auto yp=std::tie(y.point->image,y.point->coordinate,y.point->actual_scope,y.point->local,y.point->dispatch);
          if(xp!=yp)return xp<yp;
        }
        if(x.coordinate_image) {
          if(!heads.charge(2))throw ContextCap{};
          if(x.coordinate_image->id!=y.coordinate_image->id)return x.coordinate_image->id<y.coordinate_image->id;
        }
      }
      return false;
    }
    // Role order (time,state) is independent of lexical binder order. Keep
    // semantic image/box arrays unchanged; canonicalize only the at-most-two
    // assignment records so validation and memo keys retain one ordering.
    bool orderContextAxes(Context& context) {
      if(context.axis_count>context.axes.size())return false;
      if(context.axis_count==2 && context.axes[0].reference_level>context.axes[1].reference_level) {
        constexpr auto swap_work=3*((sizeof(Binding)+sizeof(Class)-1)/sizeof(Class));
        if(!heads.charge(swap_work))return false;
        std::swap(context.axes[0],context.axes[1]);
      }
      return true;
    }
    std::optional<ContextId> admitContext(const Context& source) {
      if(source.axis_count>source.axes.size())return {};
      // Three bounded record copies and insertion metadata. Comparator work
      // below is charged separately, once during explicit construction.
      constexpr auto construction_work=3*((sizeof(Context)+sizeof(Class)-1)/sizeof(Class))+8;
      if(!heads.charge(construction_work))return {};
      Context key=source;
      for(unsigned i=key.axis_count;i<key.axes.size();++i)key.axes[i]=Binding{};
      if(!orderContextAxes(key) || !validate(key))return {};
      try {
        auto found=context_keys.lower_bound(key);ContextLess less{this};
        if(found!=context_keys.end() && !less(key,found->first))return found->second;
        if(!contextCapacity(contexts.size(),"owned_assignment"))return {};
        ContextId id;id.owner=this;id.epoch=context_epoch;id.id=contexts.size();
        // Insert first: a comparator cap cannot leave an unindexed record.
        context_keys.emplace_hint(found,key,id);contexts.push_back(std::move(key));return id;
      } catch(const ContextCap&) {return {};}
    }
    const Context* context(ContextId id) {
      if(!heads.charge())return nullptr;
      ++context_reads;
      if(id.owner!=this || id.epoch!=context_epoch || id.id>=contexts.size())return nullptr;
      return &contexts[id.id]; // Borrowed immutable record, O(1), no copy/hash.
    }
    std::optional<std::size_t> contextOrdinal(ContextId id) {
      return context(id)?std::optional{id.id}:std::nullopt;
    }
    // Compile ONLY an exact original-image assignment. A memo hit reads a
    // private ID, never reconstructs/copies/hashes/validates a full Context.
    // Expression/leaf/capture checks still occur in the full body relation.
    std::optional<ContextId> sampleContext(ImageId time,std::optional<ImageId> state,
        Id actual_scope,ScalarPoint global_point,std::uint32_t state_axis=UINT32_MAX) {
      if(time.owner!=this || time.id>=images.size() ||
          (state && (state->owner!=this || state->id>=images.size())) ||
          actual_scope>=scopes.size() || !heads.charge(9))return {};
      if(auto literal=std::get_if<Class>(&global_point)) {
        if(*literal>=heads.snapshot.size())return {};*literal=heads.owner(*literal);
      } else {
        const auto copies=std::visit([](const auto& p){return 2*((sizeof(p)+sizeof(Class)-1)/sizeof(Class));},global_point);
        if(!globalValue(global_point) || !heads.charge(copies))return {};
      }
      if(!state)state_axis=UINT32_MAX;
      const SampleKey key{time,state,actual_scope,global_point,state_axis};
      if(auto it=sample_contexts.find(key);it!=sample_contexts.end()) {++sample_context_hits;return it->second;}
      if(!contextCapacity(sample_contexts.size(),"sample_plan"))return {};
      auto t=view(time);auto local=point(time,global_point,actual_scope);if(!t || !local)return {};
      Context result{t->scope,actual_scope};result.axis_count=1;
      auto witness=pointWitness(global_point);if(!witness)return {};
      result.axes[0]={t->level,{},local->local,1,*witness,false,*local};
      if(state) {
        auto s=view(*state);
        if(!s || s->scope!=t->scope || s->level==t->level)return {};
        result.axis_count=2;
        result.axes[1]={s->level,state_axis,s->begin,s->end-s->begin,s->coordinate,true};
        if(s->derived)result.axes[1].coordinate_image=*state;
      }
      auto admitted=admitContext(result);if(!admitted || !heads.charge(8))return {};
      sample_contexts.emplace(key,*admitted);return admitted;
    }
    // Both maps are small checked exact-u64 summaries from the shared factory.
    // Reference/actual namespaces remain separate, even when axis labels match.
    bool projectLeaf(IndexAffine& reference,IndexAffine& actual,const Context& initial,
        unsigned active,Id reference_scope,Id actual_scope,
        std::span<const std::pair<std::uint32_t,std::uint32_t>> lexical) {
      ++affine_checks;
      if(!boundedAffine(reference,reference_scope) || !boundedAffine(actual,actual_scope) ||
          !heads.charge((lexical.size()+initial.axis_count+1)*
              (1+std::bit_width(reference.coefficients.size()+actual.coefficients.size()+1))))return false;
      for(unsigned i=0;i<initial.axis_count;++i)if(active&(1u<<i)) {
        const auto& a=initial.axes[i];
        const auto found=reference.coefficients.find(a.reference_level);
        const auto coefficient=found==reference.coefficients.end()?0:found->second;
        std::uint64_t shift;
        if(__builtin_mul_overflow(coefficient,a.offset,&shift))return false;
        auto& shifted=a.actual_shift?actual.constant:reference.constant;
        if(__builtin_add_overflow(shifted,shift,&shifted))return false;
        reference.coefficients.erase(a.reference_level);
        if(a.actual_level) {
          const auto sample=actual.coefficients.find(*a.actual_level);
          if(coefficient!=(sample==actual.coefficients.end()?0:sample->second))return false;
          actual.coefficients.erase(*a.actual_level);
        }
      }
      for(auto [a,b]:lexical) {
        const auto ca=reference.coefficients.contains(a)?reference.coefficients.at(a):0;
        const auto cb=actual.coefficients.contains(b)?actual.coefficients.at(b):0;
        if(ca!=cb)return false;reference.coefficients.erase(a);actual.coefficients.erase(b);
      }
      return true;
    }
    bool finishLeaf(const IndexAffine& reference,const IndexAffine& actual,
        Id reference_scope,Id actual_scope) {
      if(reference.constant!=actual.constant || reference.coefficients!=actual.coefficients)return false;
      for(auto [level,coefficient]:reference.coefficients) {
        if(!heads.charge() || level>=scopes[reference_scope].domains.size() ||
            level>=scopes[actual_scope].domains.size() ||
            scopes[reference_scope].domains[level]!=scopes[actual_scope].domains[level] ||
            count(reference_scope,level)!=count(actual_scope,level))return false;
      }
      return true;
    }
    bool leaf(IndexAffine& reference,IndexAffine& actual,const Context& initial,
        unsigned active,Id reference_scope,Id actual_scope,
        std::span<const std::pair<std::uint32_t,std::uint32_t>> lexical) {
      return projectLeaf(reference,actual,initial,active,reference_scope,actual_scope,lexical) &&
          finishLeaf(reference,actual,reference_scope,actual_scope);
    }
    std::optional<std::pair<Class,Class>> typedCellPair(Class piece,Class full) {
      if(piece>=heads.snapshot.size() || full>=heads.snapshot.size())return {};
      piece=heads.owner(piece);full=heads.owner(full);
      if(piece>=heads.snapshot.size() || full>=heads.snapshot.size() || !heads.charge())return {};
      auto p=heads.compact(piece),f=heads.compact(full);if(!p || !f)return {};
      const auto& a=heads.snapshot[*p];const auto& b=heads.snapshot[*f];
      std::size_t metadata=0;
      for(std::string_view field:{a.type,a.name,a.attributes,a.provenance,b.type,b.name,b.attributes,b.provenance}) {
        if(field.size()>4096)return {};metadata+=field.size();
      }
      if(!heads.charge(metadata+2))return {};
      if(a.kind!=static_cast<int>(s::Kind::InitialCell) || b.kind!=a.kind ||
          !realType(a.type) || a.type!=b.type || a.name!=b.name || a.attributes!=b.attributes ||
          !a.provenance.empty() || !b.provenance.empty() || a.children.size()!=1 || b.children.size()!=1 ||
          !heads.engine.finite_[piece] || !heads.engine.finite_[full])return {};
      return std::pair{*p,*f};
    }
    // Relative time geometry anchored to TWO actual typed potential cells.
    // The first source cell names zero only inside this private address
    // lattice. No graph coordinate, widened time scope, or branch fact exists.
    std::optional<ImageId> typedTimeImage(Context known,Class piece,Class anchor,std::uint32_t time_axis) {
      if(piece>=heads.snapshot.size() || anchor>=heads.snapshot.size())return {};
      piece=heads.owner(piece);anchor=heads.owner(anchor);
      if(!validate(known))return {};
      for(unsigned i=0;i<known.axis_count;++i)
        if(known.axes[i].reference_level==time_axis)return {};
      auto ids=typedCellPair(piece,anchor);auto n=count(known.reference_scope,time_axis);
      if(!ids || !n)return {};
      auto context_id=admitContext(known);auto ordinal=context_id?contextOrdinal(*context_id):std::nullopt;
      if(!ordinal || !heads.charge(12+4*std::bit_width(typed_time_keys.size()+1)))return {};
      const auto key=std::tuple{piece,anchor,*ordinal,time_axis};
      if(auto found=typed_time_keys.find(key);found!=typed_time_keys.end())return found->second;
      if(!imageCapacity("typed_time_image"))return {};
      const auto& a=heads.snapshot[ids->first];const auto& b=heads.snapshot[ids->second];
      auto pa=heads.affine(a.children[0]),fa=heads.affine(b.children[0]);
      if(!pa || !fa || !projectLeaf(*pa,*fa,known,(1u<<known.axis_count)-1,
          known.reference_scope,known.actual_scope,{}) || !pa->coefficients.contains(time_axis))return {};
      const auto stride=pa->coefficients.at(time_axis);
      if(!stride || pa->constant<fa->constant || (pa->constant-fa->constant)%stride)return {};
      const auto begin=(pa->constant-fa->constant)/stride;std::uint64_t end;
      if(__builtin_add_overflow(begin,*n,&end))return {};
      const auto anchor_constant=fa->constant;
      pa->coefficients.erase(time_axis);pa->constant=fa->constant;
      if(!finishLeaf(*pa,*fa,known.reference_scope,known.actual_scope) || !heads.charge(24))return {};
      ImageId image;image.owner=this;image.id=images.size();
      images.push_back({known.reference_scope,time_axis,heads.owner(a.children[0]),begin,end,true});
      typed_time_sources.emplace(image.id,TypedTimeSource{piece,anchor,*context_id,stride,anchor_constant});
      typed_time_keys.emplace(key,image);return image;
    }
    std::optional<TypedPointId> typedPoint(ImageId role,Class actual,Id actual_scope) {
      if(role.owner!=this || role.id>=images.size() || actual_scope>=scopes.size() ||
          !heads.charge(10+std::bit_width(typed_time_sources.size()+1)))return {};
      auto found=typed_time_sources.find(role.id);if(found==typed_time_sources.end())return {};
      const auto source=found->second;auto known=context(source.context);
      if(!known || known->actual_scope!=actual_scope)return {};
      if(actual>=heads.snapshot.size())return {};
      actual=heads.owner(actual);auto ids=typedCellPair(source.piece,actual);if(!ids)return {};
      const auto& a=heads.snapshot[ids->first];const auto& b=heads.snapshot[ids->second];
      auto pa=heads.affine(a.children[0]),fa=heads.affine(b.children[0]);
      if(!pa || !fa || !projectLeaf(*pa,*fa,*known,(1u<<known->axis_count)-1,
          known->reference_scope,known->actual_scope,{}) ||
          !pa->coefficients.contains(images[role.id].level) ||
          pa->coefficients.at(images[role.id].level)!=source.stride ||
          fa->constant<source.anchor_constant || (fa->constant-source.anchor_constant)%source.stride)return {};
      const auto value=(fa->constant-source.anchor_constant)/source.stride;
      pa->coefficients.erase(images[role.id].level);pa->constant=fa->constant;
      if(!finishLeaf(*pa,*fa,known->reference_scope,known->actual_scope) || !heads.charge(14))return {};
      TypedPointId point;point.owner=this;point.epoch=context_epoch;point.image=role.id;
      point.cell=actual;point.address=heads.owner(b.children[0]);point.scope=actual_scope;point.value=value;
      return point;
    }
    // A geometry-only image from ACTUAL typed cells, not an invented reachable
    // Index+origin node. Known axes are validated by the SAME scoped affine
    // substitution factory; remaining captures retain occurrence identities.
    // Full typed body relation remains a separate mandatory proof obligation.
    std::optional<ImageId> imageFromTypedCells(Context known,Class piece,Class full,
        std::uint32_t piece_axis,std::uint32_t full_axis) {
      piece=heads.owner(piece);full=heads.owner(full);
      if(!validate(known) || piece>=heads.snapshot.size() || full>=heads.snapshot.size() ||
          !heads.charge())return {};
      const TypedImageKey key{piece,full,known,piece_axis,full_axis};
      if(auto it=typed_image_keys.find(key);it!=typed_image_keys.end())return it->second;
      if(!imageCapacity("typed_cell_image"))return {};
      for(unsigned i=0;i<known.axis_count;++i)
        if(known.axes[i].reference_level==piece_axis ||
            known.axes[i].actual_level==full_axis)return {};
      auto pair=typedCellPair(piece,full);if(!pair)return {};
      const auto& a=heads.snapshot[pair->first];const auto& b=heads.snapshot[pair->second];
      auto pa=heads.affine(a.children[0]),fa=heads.affine(b.children[0]);
      if(!pa || !fa || !projectLeaf(*pa,*fa,known,(1u<<known.axis_count)-1,
          known.reference_scope,known.actual_scope,{}))return {};
      auto pn=count(known.reference_scope,piece_axis),fn=count(known.actual_scope,full_axis);
      if(!pn || !fn || !pa->coefficients.contains(piece_axis) || !fa->coefficients.contains(full_axis))return {};
      const auto coefficient=pa->coefficients.at(piece_axis);
      if(!coefficient || coefficient!=fa->coefficients.at(full_axis) || pa->constant<fa->constant)return {};
      const auto difference=pa->constant-fa->constant;
      if(difference%coefficient)return {};
      const auto origin=difference/coefficient;
      if(origin>=*fn || *pn>*fn-origin)return {};
      pa->coefficients.erase(piece_axis);fa->coefficients.erase(full_axis);
      pa->constant=fa->constant;
      if(!finishLeaf(*pa,*fa,known.reference_scope,known.actual_scope))return {};
      ImageId id;id.owner=this;id.id=images.size();
      images.push_back({known.reference_scope,piece_axis,heads.owner(a.children[0]),origin,origin+*pn,true});
      if(!heads.charge(8))return {};
      typed_image_sources.emplace(id.id,TypedImageSource{full,known.actual_scope,full_axis});
      typed_image_keys.emplace(key,id);return id;
    }
    // Compose a proved local typed-cell image with the FULL cell's owned
    // global coordinate image. Original BODY domains remain unchanged.
    // Geometry alone still admits no P/Q or state-body equality.
    std::optional<ImageId> globalTypedImage(ImageId local,ImageId reference,Class full) {
      if(local.owner!=this || reference.owner!=this || local.id>=images.size() || reference.id>=images.size() ||
          !heads.charge(12+2*std::bit_width(translated_typed_images.size()+1)))return {};
      const auto key=std::pair{local.id,reference.id};
      auto source=typed_image_sources.find(local.id);if(source==typed_image_sources.end())return {};
      const auto& a=images[local.id];const auto& b=images[reference.id];
      if(!a.derived || source->second.full!=heads.owner(full) || source->second.scope!=b.scope ||
          source->second.axis!=b.level || b.end-b.begin!=count(b.scope,b.level) || a.end>b.end-b.begin)return {};
      if(auto it=translated_typed_images.find(key);it!=translated_typed_images.end())return it->second;
      if(!imageCapacity("global_typed_image"))return {};
      std::uint64_t begin,end;
      if(__builtin_add_overflow(a.begin,b.begin,&begin) || __builtin_add_overflow(a.end,b.begin,&end) ||
          begin>=end || end>b.end || !heads.charge(16))return {};
      ImageId id;id.owner=this;id.id=images.size();images.push_back({a.scope,a.level,a.coordinate,begin,end,true});
      translated_typed_images.emplace(key,id);return id;
    }
    std::optional<ImageId> relativeImage(ImageId actual,ImageId reference) {
      if(actual.owner!=this || reference.owner!=this || actual.id>=images.size() || reference.id>=images.size() ||
          !heads.charge(10+2*std::bit_width(relative_image_keys.size()+1)))return {};
      const auto& a=images[actual.id];const auto& b=images[reference.id];
      if(a.begin<b.begin || a.end>b.end || a.end-a.begin!=count(a.scope,a.level) ||
          b.end-b.begin!=count(b.scope,b.level))return {};
      const auto key=std::pair{actual.id,reference.id};
      if(auto it=relative_image_keys.find(key);it!=relative_image_keys.end())return it->second;
      if(!imageCapacity("relative_image") || !heads.charge(12))return {};
      ImageId id;id.owner=this;id.id=images.size();
      images.push_back({a.scope,a.level,a.coordinate,a.begin-b.begin,a.end-b.begin,true});
      relative_image_keys.emplace(key,id);return id;
    }
    // Reuse owned geometry ONLY in an ACTUAL original descendant scope.
    // The retained axis/count/capture identities cannot change. Both the
    // physical coordinate and its original typed source use must be proved
    // independent of EVERY added axis before a cache hit can be returned.
    // This creates no graph coordinate or lexical scope and proves no value
    // equality; the full scoped body relation still checks all numeric leaves.
    std::optional<ImageId> imageInOriginalScope(ImageId source,Id scope,Class source_use) {
      if(source_use>=heads.snapshot.size())return {};
      auto image=view(source);source_use=heads.owner(source_use);
      if(!image || source_use>=heads.snapshot.size() ||
          !prefix(image->scope,scope) || count(image->scope,image->level)!=count(scope,image->level) ||
          !heads.charge(6))return {};
      for(std::uint32_t axis=scopes[image->scope].domains.size();axis<scopes[scope].domains.size();++axis)
        if(!heads.charge(2) || heads.depends(image->coordinate,axis) || heads.depends(source_use,axis))return {};
      if(scope==image->scope)return source;
      const auto key=std::pair{source.id,scope};
      if(!heads.charge(1+std::bit_width(original_scope_image_keys.size()+1)))return {};
      if(auto found=original_scope_image_keys.find(key);found!=original_scope_image_keys.end())return found->second;
      if(!imageCapacity("original_scope_image") || !heads.charge(12))return {};
      ImageId id;id.owner=this;id.id=images.size();
      images.push_back({scope,image->level,image->coordinate,image->begin,image->end,image->derived});
      original_scope_image_keys.emplace(key,id);return id;
    }
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
    bool plainFold(Class id) {
      if(id>=physical_nodes || !heads.charge(5))return false;
      const auto& k=heads.snapshot[id];
      return k.kind==static_cast<int>(s::Kind::Fold) && k.fold &&
          k.fold->reducer==s::Reducer::Sum && k.fold->count && k.type=="real" &&
          k.attributes.empty() && k.provenance.empty() && k.children.size()==2 &&
          k.children[0]<physical_nodes && k.children[1]<physical_nodes &&
          heads.engine.finite_[heads.owner(id)] &&
          heads.snapshot[k.children[0]].type==k.type && heads.snapshot[k.children[1]].type==k.type &&
          heads.engine.finite_[heads.owner(k.children[0])] && heads.engine.finite_[heads.owner(k.children[1])];
    }
    std::optional<Class> fold(Class cls,Scope parent) {
      if(parent>=registry.scopes.size())return {};
      const auto rank=registry.bounds.rank(registry.scopes[parent].bounds);if(!rank)return {};
      if(heads.engine.unions_!=selection_epoch){heads.cap();return {};}
      cls=heads.owner(cls);
      if(!heads.charge(2))return {};
      // This existing conservative OR property is eagerly true for every
      // physical Fold and merged alias; a false value therefore excludes a
      // Fold witness without a new catalog. Adapter IDs are another namespace.
      if(!heads.adapter && cls<heads.engine.has_fold_.size() && !heads.engine.has_fold_[cls]) {
        if(trace_work)++fold_selection_no_fold;
        return {};
      }
      const auto selection_key=std::pair{cls,*rank};
      if(!heads.charge(3+std::bit_width(fold_selections.size()+1)))return {};
      if(auto cached=fold_selections.find(selection_key);cached!=fold_selections.end()) {
        if(trace_work)++fold_selection_hits;
        return cached->second;
      }
      const auto* choices=heads.rootChoices(cls);if(!choices)return {};
      if(trace_work)++fold_selection_scans;
      std::optional<Class> result;
      // ONE minimum actual witness at the original lexical depth. Certified
      // alpha aliases are not independently retried on proof failure.
      for(const auto& [key,choice]:*choices) {
        (void)key;if(!heads.charge())return {};
        const bool eligible=plainFold(choice.head);
        // plainFold() may deny the LAST choice on a per-query charge without
        // making the index sticky-exhausted. Never cache that partial scan,
        // including a prior candidate or an apparent absence. Exact quota
        // depletion conservatively remains uncached too.
        if(heads.exhausted || (heads.remaining && !*heads.remaining) ||
            heads.query_guards.exhausted())return {};
        if(eligible && heads.snapshot[choice.head].fold->binder_level==*rank &&
            (!result || choice.head<*result))result=choice.head;
      }
      if(!registry.contextCapacity(fold_selections.size(),"scoped_sum_fold_selection") ||
          !heads.charge(3+std::bit_width(fold_selections.size()+1)))return {};
      fold_selections.emplace(selection_key,result);
      if(!result && trace_work)++fold_selection_absences;
      return result;
    }
    std::optional<Class> forest(Class cls) {
      cls=heads.owner(cls);
      if(heads.engine.unions_!=selection_epoch || !heads.engine.finite_[cls])return {};
      const auto eligible=[&](Class id) {
        if(id>=physical_nodes || !heads.charge(4))return false;
        const auto& k=heads.snapshot[id];
        return k.kind==static_cast<int>(s::Kind::Operation) && !k.fold &&
            k.children.size()==2 && k.type=="real" && k.provenance.empty() &&
            idealOperation(k) && (k.name=="Add" || k.name=="Subtract");
      };
      auto preferred=heads.compact(cls);if(!preferred)return {};
      if(eligible(*preferred))return preferred;
      // A certified Fold alias may have the wrong ORIGINAL lexical depth for
      // this use. Select ONE existing linear forest header before the proof;
      // never rebind that Fold, search ancestors, or retry a failed body walk.
      const auto* choices=heads.rootChoices(cls);if(!choices)return {};
      std::optional<Class> selected;
      for(const auto& [key,choice]:*choices) {
        (void)key;const bool admitted=eligible(choice.head);
        if(heads.exhausted || (heads.remaining && !*heads.remaining) ||
            heads.query_guards.exhausted())return {};
        if(admitted && (!selected || choice.head<*selected))selected=choice.head;
      }
      return selected;
    }
    bool compatible(Class a,Class b) {
      // O(64) root-header scheduling only. It does not inspect an ancestor,
      // invent a Fold, or certify any child/seed/address relation.
      const auto* choices=heads.rootChoices(heads.owner(a));if(!choices)return false;
      bool sum=false;for(const auto& [key,c]:*choices){(void)key;if(!heads.charge())return false;sum|=plainFold(c.head);}
      if(!sum || !heads.engine.finite_[heads.owner(b)])return false;
      auto selected=heads.compact(heads.owner(b));if(!selected)return false;
      return plainFold(*selected) || bool(forest(b));
    }
    struct Selection {
      std::optional<std::pair<Class,Class>> ordinary;
      bool coverage=false;
    };
    Selection select(Class a,Class b,Scope r,Scope actual) {
      // A seeded Sum may share a certified Add alias with its mate. Choose
      // its ONE original-depth actual witness BEFORE structural matching;
      // otherwise that Add can pair a seed with a partial Fold. This is only
      // header dispatch: related() or the unchanged ordinary paired walk
      // must prove every actual seed/body/address below, without a retry.
      auto p=fold(a,r);if(!p)return {};
      auto q=fold(b,actual);
      if(q) {
        if(heads.snapshot[*p].fold->count!=heads.snapshot[*q].fold->count)
          return {{},true};
        auto x=heads.header(*p),y=heads.header(*q);
        if(x && y && *x==*y)return {std::pair{*p,*q},false};
        return {};
      }
      if(forest(b))return {{},true};
      return {};
    }
    bool domain(const Axis& axis,Scope r,Scope a) {
      if(r>=registry.scopes.size() || a>=registry.scopes.size() ||
          axis.reference_scope>=registry.scopes.size() || axis.actual_scope>=registry.scopes.size() ||
          !heads.charge(6))return false;
      const auto& rs=registry.scopes[r];const auto& origin=registry.scopes[axis.reference_scope];
      if(axis.reference>=rs.domains.size() || axis.reference>=origin.domains.size() ||
          rs.domains[axis.reference]!=origin.domains[axis.reference] ||
          registry.count(r,axis.reference)!=axis.reference_count)return false;
      if(axis.actual) {
        const auto& as=registry.scopes[a];const auto& ao=registry.scopes[axis.actual_scope];
        if(*axis.actual>=as.domains.size() || *axis.actual>=ao.domains.size() ||
            as.domains[*axis.actual]!=ao.domains[*axis.actual] ||
            registry.count(a,*axis.actual)!=axis.actual_count)return false;
      } else {
        // A point has no actual binder. Nested equal-count BODY scopes may
        // extend its original parent, but may not replace a captured domain.
        const auto& parent=registry.scopes[axis.actual_scope];const auto& current=registry.scopes[a];
        if(axis.actual_count!=1 || parent.domains.size()>current.domains.size())return false;
        for(std::size_t i=0;i<parent.domains.size();++i) {
          if(!heads.charge() || parent.domains[i]!=current.domains[i] ||
              registry.count(axis.actual_scope,i)!=registry.count(a,i))return false;
        }
      }
      return true;
    }
    bool independent(Class cls,const Assignment& assignment,Scope r,Scope a,
        std::span<const std::pair<std::uint32_t,std::uint32_t>> lexical) {
      if(!valid(assignment))return false;
      std::array<std::uint32_t,10> covered{};std::size_t n=0;
      for(const auto& axis:assignment.axes) {
        if(!domain(axis,r,a))return false;
        const bool identity=axis.actual && *axis.actual==axis.reference &&
            axis.stride==1 && axis.offset==0;
        if(!identity && (heads.depends(cls,axis.reference) ||
            (axis.actual && heads.depends(cls,*axis.actual))))return false;
        if(identity)covered[n++]=axis.reference;
      }
      for(auto [x,y]:lexical) {
        if(!heads.charge() || (x!=y && (heads.depends(cls,x) || heads.depends(cls,y))))return false;
        if(x==y)covered[n++]=x;
      }
      auto rv=registry.scopeView(r),av=registry.scopeView(a);
      return rv && av && registry.sameAssignment(cls,*rv,*av,std::span(covered.data(),n));
    }
    bool sameOwnerIdentity(Class cls,Geometry& geometry,Scope r,Scope a,
        std::span<const std::pair<std::uint32_t,std::uint32_t>> lexical) {
      // Congruence of the SAME complete function at identity coordinates. This
      // is a completed task boundary, not an orientation/header hint and not
      // an inference that this function is injective. No inner Fold is rebound.
      if(heads.engine.options_.certified_branch_scope || heads.adapter ||
          heads.engine.unions_!=selection_epoch || !valid(geometry.assignment) ||
          lexical.size()>8 || r>=registry.scopes.size() || a>=registry.scopes.size() ||
          cls>=physical_nodes)return false;
      cls=heads.owner(cls);
      const auto free_size=heads.engine.free_indices_[cls].size();
      if(free_size>64 || !heads.charge(16+8*(geometry.assignment.axes.size()+lexical.size())+
          4*(geometry.assignment.axes.size()+lexical.size()+1)*std::bit_width(free_size+1)))return false;
      auto candidate=geometry.assignment; // At most two fixed-size Axis records.
      for(auto& axis:candidate.axes) {
        // Even unused axes must retain their checked ORIGINAL domains.
        if(!domain(axis,r,a))return false;
        const bool used=heads.depends(cls,axis.reference) ||
            (axis.actual && heads.depends(cls,*axis.actual));
        if(!used)continue;
        if(!axis.actual || *axis.actual!=axis.reference || !axis.actual_count ||
            axis.actual_count>axis.reference_count ||
            (axis.stride && *axis.stride!=1) || (axis.offset && *axis.offset!=0))return false;
        axis.stride=1;axis.offset=0;
      }
      for(auto [x,y]:lexical) {
        if(!heads.depends(cls,x) && !heads.depends(cls,y))continue;
        const auto rc=registry.count(r,x),ac=registry.count(a,y);
        if(x!=y || !rc || !*rc || rc!=ac)return false;
      }
      // Existing whole-owner cached absence/capture guards include seeds.
      // All unmapped free coordinates retain exact occurrence tokens/counts;
      // nonidentity lexical bindings cannot capture a coordinate of this owner.
      if(!independent(cls,candidate,r,a,lexical))return false;
      geometry.assignment=std::move(candidate);++identity_assignment_hits;return true;
    }
    bool leaf(IndexAffine x,IndexAffine y,Geometry& geometry,Scope r,Scope a,
        std::span<const std::pair<std::uint32_t,std::uint32_t>> lexical) {
      if(!valid(geometry.assignment) || geometry.equations.size()>=4096 ||
          !registry.boundedAffine(x,r) || !registry.boundedAffine(y,a) ||
          !heads.charge(4*(x.coefficients.size()+y.coefficients.size()+lexical.size()+3)*
              (1+std::bit_width(x.coefficients.size()+y.coefficients.size()+1))))return false;
      Equation equation;
      for(std::size_t i=0;i<geometry.assignment.axes.size();++i) {
        auto& axis=geometry.assignment.axes[i];if(!domain(axis,r,a))return false;
        const auto ca=x.coefficients.contains(axis.reference)?x.coefficients.at(axis.reference):0;
        const auto cb=axis.actual && y.coefficients.contains(*axis.actual)?y.coefficients.at(*axis.actual):0;
        if(ca) {
          if((axis.actual && (!cb || cb%ca)) || (!axis.actual && cb))return false;
          const auto stride=axis.actual?cb/ca:1;
          if(!stride || (axis.stride && *axis.stride!=stride))return false;
          axis.stride=stride;
        } else if(cb)return false;
        equation.coefficients[i]=ca;x.coefficients.erase(axis.reference);
        if(axis.actual)y.coefficients.erase(*axis.actual);
      }
      for(auto [xlevel,ylevel]:lexical) {
        const auto ca=x.coefficients.contains(xlevel)?x.coefficients.at(xlevel):0;
        const auto cb=y.coefficients.contains(ylevel)?y.coefficients.at(ylevel):0;
        if(ca!=cb)return false;x.coefficients.erase(xlevel);y.coefficients.erase(ylevel);
      }
      if(y.constant<x.constant)return false;equation.value=y.constant-x.constant;
      x.constant=y.constant=0;
      if(!registry.finishLeaf(x,y,r,a))return false;
      geometry.equations.push_back(equation);return true;
    }
    bool solve(Geometry& geometry) {
      if(!valid(geometry.assignment) || !heads.charge(32*geometry.equations.size()+8))return false;
      auto& axes=geometry.assignment.axes;
      // Unit pivots are common typed-memory anchors. At most two passes can
      // resolve two axes; unresolved coupled offsets are Unknown, not zero.
      for(unsigned pass=0;pass<2;++pass)for(const auto& e:geometry.equations) {
        std::uint64_t known=0;std::optional<std::size_t> unknown;bool multiple=false;
        for(std::size_t i=0;i<axes.size();++i)if(e.coefficients[i]) {
          if(axes[i].offset) {
            std::uint64_t part;if(__builtin_mul_overflow(e.coefficients[i],*axes[i].offset,&part) ||
                __builtin_add_overflow(known,part,&known))return false;
          } else if(unknown)multiple=true;else unknown=i;
        }
        if(known>e.value)return false;
        if(unknown && !multiple) {
          const auto value=e.value-known,coefficient=e.coefficients[*unknown];
          if(value%coefficient)return false;axes[*unknown].offset=value/coefficient;
        } else if(!unknown && known!=e.value)return false;
      }
      // A mixed-radix address can determine two point coordinates without a
      // second address equation: H*x + L*y = offset, 0 <= y < N, H > L*(N-1).
      // Use the ORIGINAL finite domains, not guessed layout dimensions. The
      // gap check and final substitution below also reject addresses in holes.
      // This is constant work per stored equation; no domain is enumerated.
      if(axes.size()==2 && !axes[0].offset && !axes[1].offset) {
        for(const auto& e:geometry.equations) {
          if(!e.coefficients[0] || !e.coefficients[1])continue;
          const auto high=e.coefficients[0]>=e.coefficients[1]?0u:1u,low=1u-high;
          const auto h=e.coefficients[high],l=e.coefficients[low];
          std::uint64_t span;
          if(!axes[low].reference_count ||
              __builtin_mul_overflow(l,axes[low].reference_count-1,&span) || h<=span)continue;
          const auto x=e.value/h,rest=e.value%h;
          if(rest%l)return false;
          axes[high].offset=x;axes[low].offset=rest/l;break;
        }
      }
      for(const auto& axis:axes) {
        if(!axis.stride || !axis.offset || !axis.reference_count || !axis.actual_count)return false;
        std::uint64_t last;
        if(__builtin_mul_overflow(axis.actual_count-1,*axis.stride,&last) ||
            __builtin_add_overflow(last,*axis.offset,&last) || last>=axis.reference_count)return false;
      }
      for(const auto& e:geometry.equations) {
        std::uint64_t value=0;
        for(std::size_t i=0;i<axes.size();++i) {
          std::uint64_t part;if(__builtin_mul_overflow(e.coefficients[i],*axes[i].offset,&part) ||
              __builtin_add_overflow(value,part,&value))return false;
        }
        if(value!=e.value)return false;
      }
      return true;
    }
    struct Projection {
      Assignment local;
      std::array<std::size_t,2> original{};
    };
    std::optional<Projection> project(Class target,Class actual,Scope r,Scope a,
        const Assignment& incoming) {
      if(!valid(incoming) || target>=physical_nodes || actual>=physical_nodes ||
          !heads.charge(8+4*incoming.axes.size()))return {};
      const auto left=heads.owner(target),right=heads.owner(actual);
      Projection result{root(),{}};
      for(std::size_t i=0;i<incoming.axes.size();++i) {
        const auto& axis=incoming.axes[i];
        // An omitted axis retains its ORIGINAL lexical domains. Absence from
        // the conservative whole-expression union includes the actual seeds;
        // presence is Unknown, never a request for another dependency walk.
        if(!domain(axis,r,a) || !heads.charge(2+
            std::bit_width(heads.engine.free_indices_[left].size()+1)+
            std::bit_width(heads.engine.free_indices_[right].size()+1)))return {};
        const bool reference_used=heads.depends(left,axis.reference);
        const bool actual_used=axis.actual && heads.depends(right,*axis.actual);
        if(!reference_used && !actual_used)continue;
        result.original[result.local.axes.size()]=i;
        result.local.axes.push_back(axis);
      }
      return result;
    }
    bool merge(const Projection& projection,const Assignment& proved,
        const Assignment& incoming,Assignment& result) {
      if(!valid(projection.local) || !valid(proved) || !valid(incoming) ||
          proved.axes.size()!=projection.local.axes.size() ||
          !heads.charge(8+8*incoming.axes.size()))return false;
      result=incoming;
      for(std::size_t i=0;i<proved.axes.size();++i) {
        const auto original=projection.original[i];if(original>=incoming.axes.size())return false;
        const auto& before=incoming.axes[original];const auto& after=proved.axes[i];
        if(projection.local.axes[i]!=before ||
            std::tie(before.reference_scope,before.actual_scope,before.reference,before.actual,
                before.reference_count,before.actual_count)!=
            std::tie(after.reference_scope,after.actual_scope,after.reference,after.actual,
                after.reference_count,after.actual_count) ||
            (before.stride && before.stride!=after.stride) ||
            (before.offset && before.offset!=after.offset))return false;
        // Only genuinely derived retained-axis facts are returned. Omitted
        // axes, including unknown geometry, are copied unchanged. The caller's
        // equations remain in their ORIGINAL index order and must still pass
        // its final solve(); this local proof invents no zero offset/stride.
        if(after.stride)result.axes[original].stride=after.stride;
        if(after.offset)result.axes[original].offset=after.offset;
      }
      return true;
    }
    std::optional<QualifiedKey> qualify(Class witness,Class actual,Scope r,Scope a,
        const Assignment& active,const Environment& enclosing) {
      // Certified branch anchors have a separate proof-metadata namespace
      // deliberately absent from free_indices_. Keep their old full-key path.
      if(heads.engine.options_.certified_branch_scope || !valid(active) || active.axes.size()>=2 ||
          r>=registry.scopes.size() || a>=registry.scopes.size() ||
          heads.engine.unions_!=selection_epoch ||
          !heads.charge(8+8*(active.axes.size()+enclosing.size())))return {};
      auto rr=registry.bounds.rank(registry.scopes[r].bounds);
      auto ar=registry.bounds.rank(registry.scopes[a].bounds);
      if(!rr || !ar)return {};
      for(const auto& axis:active.axes)if(!domain(axis,r,a))return {};
      auto rv=registry.scopeView(r),av=registry.scopeView(a);if(!rv || !av)return {};
      const auto capture=[&](Class cls,const auto& view,bool reference)->std::optional<Captures> {
        const auto& free=heads.engine.free_indices_[heads.owner(cls)];
        if(free.size()>64 || !heads.charge(4*free.size()+1))return {};
        Captures result;result.reserve(free.size());
        for(auto level:free) {
          bool mapped=false;
          for(const auto& axis:active.axes)
            mapped|=reference?axis.reference==level:axis.actual==level;
          if(mapped)continue;
          auto token=registry.domain(view,level);auto count=registry.count(view,level);
          if(!token || !count || !*count)return {};
          result.emplace_back(level,token->first,token->second,*count);
        }
        return result;
      };
      auto left=capture(witness,*rv,true),right=capture(actual,*av,false);
      if(!left || !right)return {};
      return QualifiedKey{witness,heads.owner(actual),*rr,*ar,active.axes,
          std::move(*left),std::move(*right),enclosing};
    }
    bool rebind(const CompletedRelation& completed,const Projection& projection,
        Class witness,Class actual,Scope r,Scope a,const Assignment& incoming,Assignment& result) {
      if(!completed.key || !completed.record || completed.record->gray ||
          !completed.record->projected_result || heads.engine.unions_!=selection_epoch)return false;
      const auto& key=*completed.key;
      if(std::get<0>(key)!=witness || std::get<1>(key)!=heads.owner(actual))return false;
      auto old_r=registry.scopeView(std::get<2>(key)),new_r=registry.scopeView(r);
      auto old_a=registry.scopeView(std::get<3>(key)),new_a=registry.scopeView(a);
      if(!old_r || !new_r || !old_a || !new_a)return false;
      std::array<std::uint32_t,2> reference{},actual_levels{};std::size_t rn=0,an=0;
      for(const auto& axis:projection.local.axes) {
        // These are the ONLY skipped capture tokens; both requests must use
        // the SAME immutable original axis definitions and their domains.
        if(!domain(axis,std::get<2>(key),std::get<3>(key)) || !domain(axis,r,a))return false;
        reference[rn++]=axis.reference;
        if(axis.actual)actual_levels[an++]=*axis.actual;
      }
      if(!registry.sameAssignment(witness,*old_r,*new_r,std::span(reference.data(),rn)) ||
          !registry.sameAssignment(actual,*old_a,*new_a,std::span(actual_levels.data(),an)))return false;
      return merge(projection,*completed.record->projected_result,incoming,result);
    }
    void remember(QualifiedKey key,const GoalKey& original,const Record& completed) {
      if(!heads.charge(16*(std::get<4>(key).size()+std::get<5>(key).size()+
          std::get<6>(key).size()+std::get<7>(key).size()+8)*(1+std::bit_width(qualified.size()+1))))return;
      if(qualified.find(key)!=qualified.end())return;
      if(!registry.contextCapacity(qualified.size(),"scoped_sum_completed_relation"))return;
      // map-owned keys/records remain stable throughout this sealed epoch.
      // No graph node, growing vector element or callback is borrowed here.
      qualified.emplace(std::move(key),CompletedRelation{&original,&completed});
    }
    bool sameCoefficient(const bounded_linear_reduction::Coefficient& x,
        const bounded_linear_reduction::Coefficient& y) {
      if(x.terms().size()>1 || y.terms().size()>1 ||
          (!x.terms().empty() && !x.terms()[0].monomial.powers.empty()) ||
          (!y.terms().empty() && !y.terms()[0].monomial.powers.empty()))return false;
      std::size_t work=2;
      for(const auto* coefficient:{&x,&y})for(const auto& term:coefficient->terms())
        work+=term.coefficient.numerator.backend().size()+term.coefficient.denominator.backend().size()+1;
      return heads.charge(work) && x==y;
    }
    // Compare a two-axis zero-seeded Sum with a stored additive forest.
    // Each source atom is a point or an inner-axis Fold. Retain its parameter
    // range and prove the complete scalar body under a two-axis assignment.
    // Rows are indexed by STORED pieces, never generated from the target count.
    bool nestedPointCoverage(Class outer,Scope r,Scope a,const Environment& enclosing,
        const Linear& source,const Assignment& incoming) {
      if(!source || !incoming.axes.empty())return false;
      const auto& f=heads.snapshot[outer];
      if(!plainFold(outer) || !heads.engine.zero_[heads.owner(f.children[0])] ||
          !source->constant().terms().empty())return false;
      auto rs=registry.insideFold(r,outer);if(!rs)return false;
      auto inner=fold(f.children[1],*rs);if(!inner || !plainFold(*inner))return false;
      const auto& g=heads.snapshot[*inner];
      if(g.type!=f.type || g.fold->binder_level!=f.fold->binder_level+1 ||
          !heads.engine.zero_[heads.owner(g.children[0])] ||
          !f.fold->count || !g.fold->count)return false;
      auto body_scope=registry.insideFold(*rs,*inner);if(!body_scope)return false;
      const auto one=builder.coefficient("1");synchronize();if(!one)return false;
      struct Piece {std::uint64_t stride,offset,count;};
      std::map<std::uint64_t,std::vector<Piece>> rows;
      if(source->terms().empty() || source->terms().size()>512)return false;
      for(const auto& term:source->terms()) {
        if(!heads.charge(32) || !sameCoefficient(term.weight,*one))return false;
        auto sf=reduction(term.atom);Class body;Scope ab=a;
        std::uint64_t count=1;std::optional<std::uint32_t> axis;
        if(sf) {
          if(!plainFold(*sf))return false;const auto& h=heads.snapshot[*sf];
          if(h.type!=f.type || (term.atom<physical_nodes &&
              !heads.engine.zero_[heads.owner(h.children[0])]))return false;
          auto scope=registry.insideFold(a,*sf);if(!scope)return false;
          ab=*scope;body=h.children[1];count=h.fold->count;axis=h.fold->binder_level;
        } else {
          if(term.atom>=physical_nodes)return false;body=static_cast<Class>(term.atom);
        }
        Assignment map=incoming;
        map.axes.push_back({*rs,a,f.fold->binder_level,{},f.fold->count,1,{},{}});
        map.axes.push_back({*body_scope,ab,g.fold->binder_level,axis,g.fold->count,count,{},{}});
        if(!heads.engine.affineBodyMapWithHeads(g.children[1],body,UINT32_MAX,
            heads.snapshot,false,heads.remaining,heads,UINT32_MAX,nullptr,nullptr,nullptr,this,
            &map,*body_scope,ab,&enclosing))return false;
        if(!map.axes[0].offset || !map.axes[1].offset || !map.axes[1].stride)return false;
        rows[*map.axes[0].offset].push_back({*map.axes[1].stride,*map.axes[1].offset,count});
      }
      // Every row must occur, but checking the map size and adjacent stored
      // keys establishes this without iterating a potentially huge extent.
      if(rows.size()!=f.fold->count)return false;
      std::uint64_t expected=0;
      for(const auto& [row,pieces]:rows) {
        if(row!=expected++)return false;
        std::uint64_t lattice=1;
        for(const auto& p:pieces) {
          if(!heads.charge(4))return false;
          std::uint64_t next;
          if(!p.stride || __builtin_mul_overflow(lattice/std::gcd(lattice,p.stride),p.stride,&next) ||
              next>64)return false;
          lattice=next;
        }
        for(std::uint64_t residue=0;residue<lattice;++residue) {
          std::vector<bounded_linear_reduction::WeightedInterval> left,right;
          if(residue<g.fold->count)right.push_back({0,1+(g.fold->count-1-residue)/lattice,*one});
          for(const auto& p:pieces) {
            if(!heads.charge(8))return false;
            const auto delta=(residue+lattice-p.offset%lattice)%lattice;
            if(delta%p.stride)continue;
            const auto first=delta/p.stride;if(first>=p.count)continue;
            std::uint64_t coordinate,end;
            if(__builtin_mul_overflow(first,p.stride,&coordinate) ||
                __builtin_add_overflow(p.offset,coordinate,&coordinate))return false;
            const auto begin=coordinate/lattice,n=1+(p.count-1-first)/(lattice/p.stride);
            if(__builtin_add_overflow(begin,n,&end))return false;
            left.push_back({begin,end,*one});
          }
          const auto equal=builder.intervalsEqual(left,right);synchronize();
          if(!equal || !*equal)return false;
        }
      }
      return true;
    }
    bool related(Class target,Class actual,Scope r,Scope a,const Environment& enclosing,
        Assignment& incoming) {
      // This operation proves ONE stored forest, not one individual mapper
      // body. Suspend only the paused caller's local pathology guard; every
      // child body installs its own guard and keeps the SAME aggregate.
      BodyHeadIndex::MapperPause aggregation(heads);
      if(!valid(incoming) || enclosing.size()>8 ||
          !heads.charge(24+8*(incoming.axes.size()+enclosing.size())))return false;
      auto witness=fold(target,r);if(!witness)return false;
      auto projection=project(target,actual,r,a,incoming);
      if(!projection || projection->local.axes.size()>=2)return false;
      // Memoize the FULL original assignment, not the shortened local view.
      // Original scopes/enclosing bindings also remain complete key fields.
      const auto key=GoalKey{*witness,heads.owner(actual),r,a,incoming.axes,enclosing};
      if(!heads.charge(4*(incoming.axes.size()+enclosing.size()+8)*(1+std::bit_width(records.size()+1))))return false;
      if(auto found=records.find(key);found!=records.end()) {
        ++hits;if(found->second.gray || !found->second.result)return false;
        incoming=*found->second.result;return true;
      }
      if(!registry.contextCapacity(records.size(),"scoped_sum_goal"))return false;
      auto [entry,inserted]=records.emplace(key,Record{});(void)inserted;++queries;
      GoalTrace report(*this,*witness,heads.owner(actual),r,a,incoming.axes.size());
      // Every exit seals this exact goal as success or Unknown. No alternative
      // prototype, budget reset, or failed full-proof retry is attempted.
      struct Seal {Record& r;~Seal(){r.gray=false;}} seal{entry->second};
      auto qualified_key=qualify(*witness,actual,r,a,projection->local,enclosing);
      if(qualified_key) {
        if(!heads.charge(16*(projection->local.axes.size()+std::get<5>(*qualified_key).size()+
            std::get<6>(*qualified_key).size()+enclosing.size()+8)*
            (1+std::bit_width(qualified.size()+1))))return false;
        if(auto found=qualified.find(*qualified_key);found!=qualified.end()) {
          Assignment rebound;
          // ONE completed prototype, never a gray/Unknown record or retry of
          // another scope after a failed transport. Full request stays keyed.
          if(!rebind(found->second,*projection,*witness,actual,r,a,incoming,rebound))return false;
          entry->second.result=rebound;
          entry->second.projected_result=found->second.record->projected_result;
          incoming=std::move(rebound);++qualified_hits;++completed;
          if(report.enabled)report.success=true;
          return true;
        }
      }
      const auto& target_fold=heads.snapshot[*witness];
      auto rb=registry.insideFold(r,*witness);if(!rb)return false;
      report.next(1);
      auto seed=summarize(heads.owner(target_fold.children[0]));
      if(report.enabled && seed)report.seed_terms=seed->terms().size();
      report.next(2);
      auto source=summarize(heads.owner(actual));
      if(report.enabled && source)report.source_terms=source->terms().size();
      if(!seed || !source)return false;
      report.next(3);
      if(nestedPointCoverage(*witness,r,a,enclosing,source,projection->local)) {
        // No enclosing nonidentity axis was inferred; preserve the complete
        // incoming assignment and the ordinary goal memo's cycle protection.
        entry->second.result=incoming;entry->second.projected_result=projection->local;
        ++completed;if(report.enabled)report.success=true;return true;
      }
      // Linear coefficients are exact constant rationals. Scalar backend
      // limb counts account for equality WITHOUT copying a bigint/record.
      if(!sameCoefficient(seed->constant(),source->constant()))return false;
      const auto& residual=seed->terms();std::vector<bool> consumed(residual.size());
      if(!heads.charge(8*(residual.size()+source->terms().size()+1)))return false;
      struct Piece {std::uint64_t stride,offset,count;};std::vector<Piece> pieces;
      const auto one=builder.coefficient("1");synchronize();if(!one)return false;
      Assignment committed=projection->local;
      for(const auto& term:source->terms()) {
        // Preserve exact outside-binder seed atoms once. A captured atom may
        // not cross a nonidentity/unknown outer assignment just by class ID.
        if(!heads.charge(1+std::bit_width(residual.size()+1)))return false;
        auto found=std::lower_bound(residual.begin(),residual.end(),term.atom,
            [](const auto& s,auto atom){return s.atom<atom;});
        const auto position=static_cast<std::size_t>(found-residual.begin());
        if(found!=residual.end() && found->atom==term.atom && !consumed[position] &&
            sameCoefficient(found->weight,term.weight) && term.atom<physical_nodes &&
            independent(static_cast<Class>(term.atom),committed,r,a,enclosing)) {
          consumed[position]=true;continue;
        }
        if(!sameCoefficient(term.weight,*one) || pieces.size()>=512)return false;
        auto source_fold=reduction(term.atom);
        Class body;Scope ab=a;std::uint64_t count=1;std::optional<std::uint32_t> actual_axis;
        if(source_fold) {
          if(!plainFold(*source_fold))return false;const auto& f=heads.snapshot[*source_fold];
          auto nested=registry.insideFold(a,*source_fold);if(!nested)return false;
          // A private body-only atom excludes the actual seed; a physical
          // ordinary Fold atom is admissible here ONLY with certified zero.
          if(term.atom<physical_nodes && !heads.engine.zero_[heads.owner(f.children[0])])return false;
          body=f.children[1];ab=*nested;count=f.fold->count;actual_axis=f.fold->binder_level;
        } else {
          if(term.atom>=physical_nodes)return false;body=static_cast<Class>(term.atom);
        }
        Geometry local{committed,{}};
        const auto axis=target_fold.fold->binder_level;
        for(const auto& outer:local.assignment.axes)if(outer.reference==axis ||
            (actual_axis && outer.actual==actual_axis))return false;
        local.assignment.axes.push_back({*rb,ab,axis,actual_axis,target_fold.fold->count,count,{},{}});
        PieceTrace piece_report(*this,target_fold.children[1],body,*rb,ab,count);
        if(!heads.engine.affineBodyMapWithHeads(target_fold.children[1],body,UINT32_MAX,
            heads.snapshot,false,heads.remaining,heads,UINT32_MAX,nullptr,nullptr,nullptr,this,
            &local.assignment,*rb,ab,&enclosing))return false;
        if(piece_report.enabled)piece_report.complete=true;
        const auto image=local.assignment.axes.back();
        if(!image.stride || !image.offset)return false;
        pieces.push_back({*image.stride,*image.offset,count});
        if(report.enabled)report.pieces=pieces.size();
        local.assignment.axes.pop_back();
        committed=std::move(local.assignment);
      }
      if(std::find(consumed.begin(),consumed.end(),false)!=consumed.end() || pieces.empty())return false;
      report.next(4);
      // Finite lattice quotient, not Fold enumeration. At most64 residue
      // classes and1024 symbolic intervals retain exact additive multiplicity.
      std::uint64_t lattice=1;
      for(const auto& p:pieces) {
        if(!heads.charge(1+std::bit_width(lattice)+std::bit_width(p.stride)))return false;
        const auto g=std::gcd(lattice,p.stride);std::uint64_t next;
        if(__builtin_mul_overflow(lattice/g,p.stride,&next) || next>64)return false;lattice=next;
      }
      std::vector<bounded_linear_reduction::WeightedInterval> left,right;
      for(std::uint64_t residue=0;residue<lattice;++residue) {
        const auto total=target_fold.fold->count;
        if(residue<total)right.push_back({0,1+(total-1-residue)/lattice,*one});
        for(const auto& p:pieces) {
          if(!heads.charge(8))return false;
          const auto initial=p.offset%lattice;
          const auto delta=(residue+lattice-initial)%lattice;
          if(delta%p.stride)continue;const auto first=delta/p.stride;
          if(first>=p.count)continue;
          std::uint64_t coordinate;
          if(__builtin_mul_overflow(first,p.stride,&coordinate) ||
              __builtin_add_overflow(p.offset,coordinate,&coordinate))return false;
          const auto step=lattice/p.stride,count=1+(p.count-1-first)/step;
          const auto begin=coordinate/lattice;std::uint64_t end;
          if(__builtin_add_overflow(begin,count,&end) || left.size()+right.size()>=1024)return false;
          left.push_back({begin,end,*one});
        }
        const auto equal=builder.intervalsEqual(left,right);synchronize();
        if(!equal || !*equal) {
          // Bounded TRACE sample of intervals already constructed by the proof;
          // no new ancestor, member, or logical reduction traversal.
          if(trace_work && lattice_failure_reports++<8) {
            SENTRYWARD_DEBUG_PRINT(egraph_86);
            for(std::size_t i=0;i<std::min<std::size_t>(left.size(),16);++i)
              SENTRYWARD_DEBUG_PRINT(egraph_87);
            SENTRYWARD_DEBUG_PRINT(egraph_88);
          }
          return false;
        }
        left.clear();right.clear();
      }
      report.next(5);
      Assignment merged;
      if(!merge(*projection,committed,incoming,merged))return false;
      entry->second.result=merged;entry->second.projected_result=committed;
      entry->second.gray=false;
      // Exact incoming and completed geometry have separate keys, not wildcard
      // facts. Two slots avoid repeating an initially-Unknown offset proof
      // once that proof certified it; known conflicts never hit either key.
      if(qualified_key)remember(std::move(*qualified_key),entry->first,entry->second);
      if(auto solved=qualify(*witness,actual,r,a,committed,enclosing))
        remember(std::move(*solved),entry->first,entry->second);
      incoming=std::move(merged);++completed;
      if(report.enabled)report.success=true;
      return true;
    }
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
    static bounded_linear_reduction::Limits intervalLimits(std::size_t work) {
      // Opt-in uses the SAME frozen physical-epoch allowance; each primitive
      // still pre-debits heads.charge. This is not a fresh per-family budget.
      bounded_linear_reduction::Limits limits;
      limits.max_work=std::min(work,epoch_work_allowance::maximum_work);
      limits.coefficients.max_work=limits.max_work;return limits;
    }
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
    // Canonical interval equality rejects multiplicity, gaps, overlaps and
    // duplicates; no logical coordinate is inspected. Builder work is charged
    // through primitive hooks to the SAME epoch allowance before any work.
    bool cover(std::span<const std::pair<std::uint64_t,std::uint64_t>> pieces,
        std::pair<std::uint64_t,std::uint64_t> domain) {
      if(pieces.size()>=1024 || !registry.heads.charge(pieces.size()+1))return false;
      auto one=intervals.coefficient("1");if(!sync() || !one)return false;
      std::vector<bounded_linear_reduction::WeightedInterval> values;values.reserve(pieces.size());
      for(auto [begin,end]:pieces) {
        if(begin>=end || !registry.heads.charge(2))return false;
        values.push_back({begin,end,*one});
      }
      const std::array<bounded_linear_reduction::WeightedInterval,1> target{{{domain.first,domain.second,*one}}};
      auto result=intervals.intervalsEqual(values,target);return sync() && result && *result;
    }
    std::optional<Id> join(std::span<const Piece> pieces,unsigned rank,Box domain) {
      if(rank==0 || rank>2 || pieces.empty() || pieces.size()>=1024 ||
          !registry.heads.charge(pieces.size()*(20+std::bit_width(pieces.size()+1))))return {};
      for(unsigned axis=0;axis<rank;++axis)if(domain[axis].first>=domain[axis].second)return {};
      std::vector<Box> boxes;boxes.reserve(pieces.size());
      std::vector<PieceKey> piece_keys;piece_keys.reserve(pieces.size());
      std::optional<std::vector<Capture>> captures;std::string_view type;
      for(const auto& piece:pieces) {
        auto body=registry.heads.owner(piece.body);
        if(body>=registry.heads.snapshot.size())return {};
        const auto& k=registry.heads.snapshot[body];
        if(piece.scope>=registry.scopes.size() || !realType(k.type) || !k.provenance.empty() ||
            !registry.heads.engine.finite_[body] || !registry.heads.charge())return {};
        if(type.empty())type=k.type;else if(type!=k.type)return {};
        Box box{};std::array<std::uint32_t,2> levels{};std::array<Class,2> coordinates{};
        for(unsigned axis=0;axis<rank;++axis) {
          auto image=registry.view(piece.images[axis]);
          if(!image || image->scope!=piece.scope || (axis && image->level==levels[axis-1]) ||
              image->begin<domain[axis].first || image->end>domain[axis].second)return {};
          levels[axis]=image->level;coordinates[axis]=image->coordinate;box[axis]={image->begin,image->end};
        }
        const auto& free=registry.heads.engine.free_indices_[body];
        if(free.size()>8 || !registry.heads.charge(free.size()+1))return {};
        std::vector<Capture> actual;
        for(auto level:free) {
          bool reduced=false;for(unsigned axis=0;axis<rank;++axis)reduced|=levels[axis]==level;
          if(reduced)continue;
          if(level>=registry.scopes[piece.scope].domains.size())return {};
          actual.emplace_back(level,registry.scopes[piece.scope].domains[level]);
        }
        if(captures && *captures!=actual)return {};captures=std::move(actual);
        boxes.push_back(box);piece_keys.emplace_back(body,piece.scope,levels[0],coordinates[0],levels[1],coordinates[1],box);
      }
      // Exact complete keys, not shape/name equality. Immutable owner/epoch,
      // scopes/occurrences and actual coordinate IDs are retained in the key.
      if(!registry.heads.charge(pieces.size()*(6+6*std::bit_width(keys.size()+1))))return {};
      Key key{rank,domain,piece_keys};
      if(auto it=keys.find(key);it!=keys.end())return it->second;
      if(families.size()>=256)return {};
      std::vector<std::size_t> order(pieces.size());std::iota(order.begin(),order.end(),0);
      std::sort(order.begin(),order.end(),[&](auto a,auto b){return boxes[a]<boxes[b];});
      std::vector<std::pair<std::uint64_t,std::uint64_t>> bands;
      if(rank==1) {
        bands.reserve(order.size());for(auto i:order)bands.push_back(boxes[i][0]);
      } else {
        for(std::size_t i=0;i<order.size();) {
          const auto band=boxes[order[i]][0];bands.push_back(band);
          std::vector<std::pair<std::uint64_t,std::uint64_t>> inner;
          do {inner.push_back(boxes[order[i]][1]);++i;}
            while(i<order.size() && boxes[order[i]][0]==band);
          if(!cover(inner,domain[1]))return {};
        }
      }
      if(!cover(bands,domain[0]))return {};
      if(!registry.heads.charge(pieces.size()*10+captures->size()*2))return {};
      auto family=std::make_unique<Family>(Family{rank,domain,type,{}, {},std::move(*captures)});
      family->pieces.reserve(pieces.size());family->boxes.reserve(pieces.size());
      for(auto i:order){family->pieces.push_back(pieces[i]);family->boxes.push_back(boxes[i]);}
      Id id;id.owner=this;id.id=families.size();families.push_back(std::move(family));keys.emplace(std::move(key),id);return id;
    }
    const Family* view(Id id) const {
      return id.owner==this && id.id<families.size()?families[id.id].get():nullptr;
    }
    bool uniform(Id id,Class reference,Registry::Id scope,std::span<const std::uint32_t> axes,
        std::span<const Registry::ImageId> reference_images={}) {
      const auto* family=view(id);if(!family || axes.size()!=family->rank ||
          (!reference_images.empty() && reference_images.size()!=axes.size()))return false;
      for(const auto& piece:family->pieces) {
        if(!registry.heads.charge())return false;
        Registry::Context context{scope,piece.scope};context.axis_count=family->rank;
        for(unsigned axis=0;axis<family->rank;++axis) {
          auto image=registry.view(piece.images[axis]);if(!image)return false;
          context.axes[axis]={axes[axis],image->level,image->begin,image->end-image->begin,image->coordinate,false};
          if(image->derived)context.axes[axis].coordinate_image=piece.images[axis];
          if(!reference_images.empty()) {
            auto ref=registry.view(reference_images[axis]);
            if(!ref || ref->scope!=scope || ref->level!=axes[axis])return false;
            auto relative=registry.relativeImage(piece.images[axis],reference_images[axis]);
            auto r=relative?registry.view(*relative):std::nullopt;if(!r)return false;
            context.axes[axis].offset=r->begin;context.axes[axis].actual_coordinate=r->coordinate;
            context.axes[axis].coordinate_image=*relative;
          }
        }
        if(!registry.orderContextAxes(context) ||
            !registry.heads.engine.affineBodyMapWithHeads(reference,piece.body,UINT32_MAX,
            registry.heads.snapshot,false,registry.heads.remaining,registry.heads,UINT32_MAX,&registry,&context))return false;
      }
      return true;
    }
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
    bool related(Use r,Use a,const Registry::Context& context) {
      // Resident, complete-key proof memo; eviction never resets the shared
      // allowance. Original scopes and private point/image handles remain in
      // equality even though the hash is only a cheap indexing discriminator.
      if(!registry.heads.charge(24))return false; // Two fixed complete-context hash slots.
      const RelationKey key{r,a,context};
      return memoRelation(key,[&] {
        return bool(registry.heads.engine.affineBodyMapWithHeads(r.body,a.body,UINT32_MAX,
            registry.heads.snapshot,false,registry.heads.remaining,registry.heads,UINT32_MAX,&registry,&context));
      });
    }
    bool related(Use r,Use a,Registry::ContextId id) {
      const auto* context=registry.context(id);
      if(!context || r.scope!=context->reference_scope || a.scope!=context->actual_scope ||
          !registry.heads.charge(12))return false;
      // Two actual Uses plus an opaque owned ID: no Context record copy,
      // complete-field hash or initial coordinate validation on a memo hit.
      const RelationKey key{r,a,id};
      return memoRelation(key,[&] {
        return bool(registry.heads.engine.affineBodyMapWithHeads(r.body,a.body,UINT32_MAX,
            registry.heads.snapshot,false,registry.heads.remaining,registry.heads,UINT32_MAX,&registry,context,&id));
      });
    }
    std::optional<Registry::Id> common(std::span<const Use> uses) {
      if(uses.empty() || uses.size()>8 || uses.front().scope>=registry.scopes.size())return {};
      auto deepest=uses.front().scope;
      for(const auto& use:uses) {
        if(use.scope>=registry.scopes.size() || !registry.heads.charge())return {};
        if(registry.scopes[use.scope].domains.size()>registry.scopes[deepest].domains.size())deepest=use.scope;
      }
      for(const auto& use:uses)if(!registry.prefix(use.scope,deepest))return {};
      return deepest;
    }
    bool admitted(Use use) {
      use.body=registry.heads.owner(use.body);
      if(use.scope>=registry.scopes.size() || use.body>=registry.heads.snapshot.size() ||
          !registry.heads.charge())return false;
      const auto& k=registry.heads.snapshot[use.body];
      if(!realType(k.type) || !k.provenance.empty() || !registry.heads.engine.finite_[use.body])return false;
      const auto& free=registry.heads.engine.free_indices_[use.body];
      if(free.size()>8 || !registry.heads.charge(free.size()+1))return false;
      for(auto level:free)if(level>=registry.scopes[use.scope].domains.size())return false;
      return true;
    }
    std::optional<std::vector<Use>> factors(std::span<const Use> uses,bool strict_real_products=false) {
      if(uses.empty() || uses.size()>8 || !registry.heads.charge(uses.size()))return {};
      struct Task {Use use;bool ready=false;};std::vector<Task> tasks;
      for(auto use:uses)tasks.push_back({use});
      std::unordered_set<Class> active;std::vector<Use> result;std::size_t steps=0;
      while(!tasks.empty()) {
        if(++steps>48 || !registry.heads.charge())return {};
        auto task=tasks.back();tasks.pop_back();task.use.body=registry.heads.owner(task.use.body);
        if(task.ready){active.erase(task.use.body);continue;}
        if(active.contains(task.use.body) || !admitted(task.use))return {};
        auto head=registry.heads.compact(task.use.body);if(!head)return {};
        const auto& k=registry.heads.snapshot[*head];
        if(k.type!=registry.heads.snapshot[task.use.body].type || !k.provenance.empty())return {};
        if(k.kind==static_cast<int>(s::Kind::Operation) && k.name=="Multiply") {
          // Numeric projection's existing marker is an admitted ideal
          // arithmetic head, not an unknown modifier. Other modifiers remain
          // opaque/Unknown; leaf encodings are compared by the full mapper.
          if(!arithmetic(k) || k.children.size()!=2 || (strict_real_products && k.type!="real"))return {};
          active.insert(task.use.body);tasks.push_back({task.use,true});
          tasks.push_back({{k.children[0],task.use.scope}});tasks.push_back({{k.children[1],task.use.scope}});
        } else {
          // Existing exact ideal unit identity. No rounded coefficient, SAT or
          // arbitrary conversion is discarded; unknown constants stay factors.
          if(k.kind==static_cast<int>(s::Kind::Constant) && k.name=="1" && k.attributes.empty())continue;
          if(result.size()==8)return {};result.push_back(task.use);
        }
      }
      return result;
    }
    std::optional<unsigned> activeAxes(Use r,Use a,const Registry::Context& context) {
      if(!registry.prefix(r.scope,context.reference_scope) || !registry.prefix(a.scope,context.actual_scope))return {};
      if(!registry.heads.charge(2+12*context.axis_count))return {};
      unsigned active=0;
      for(unsigned i=0;i<context.axis_count;++i) {
        const auto& axis=context.axes[i];
        const bool ref_present=bool(registry.count(r.scope,axis.reference_level));
        const bool actual_present=!axis.actual_level || bool(registry.count(a.scope,*axis.actual_level));
        const bool used=registry.heads.depends(r.body,axis.reference_level) ||
            (axis.actual_level && registry.heads.depends(a.body,*axis.actual_level));
        // A proved dependency ABSENCE on both original factor uses permits
        // removing an irrelevant assignment, even when that axis is present.
        // This is checked per expression, never inferred from cached geometry.
        if(!used)continue;
        if(!ref_present || !actual_present)return {};
        active|=1u<<i;
      }
      return active;
    }
    std::optional<Registry::Context> contextFor(Use r,Use a,const Registry::Context& context) {
      auto active=activeAxes(r,a,context);if(!active)return {};
      Registry::Context result{r.scope,a.scope};
      for(unsigned i=0;i<context.axis_count;++i)if(*active&(1u<<i)) {
        auto axis=context.axes[i];
        if(axis.point) {
          auto point=registry.pointIn(*axis.point,r.scope,a.scope);if(!point)return {};axis.point=*point;
        }
        result.axes[result.axis_count++]=axis;
      }
      return result;
    }
    std::optional<Registry::ContextId> contextFor(Use r,Use a,Registry::ContextId parent) {
      const auto* context=registry.context(parent);
      if(!context || !registry.prefix(r.scope,context->reference_scope) ||
          !registry.prefix(a.scope,context->actual_scope))return {};
      // These are EXPRESSION checks, not cached geometry. An already minted
      // subcontext cannot authorize a different factor to cross a live axis.
      auto active=activeAxes(r,a,*context);if(!active)return {};
      if(r.scope==context->reference_scope && a.scope==context->actual_scope &&
          *active==((1u<<context->axis_count)-1))return parent;
      auto ordinal=registry.contextOrdinal(parent);if(!ordinal ||
          !registry.heads.charge(6+4*std::bit_width(prefix_contexts.size()+1)))return {};
      const auto key=std::tuple{*ordinal,r.scope,a.scope,*active};
      if(auto found=prefix_contexts.find(key);found!=prefix_contexts.end()) {
        ++registry.context_prefix_hits;return found->second;
      }
      if(!registry.contextCapacity(prefix_contexts.size(),"factor_prefix_plan"))return {};
      // Parent records have stable deque storage through this append. The
      // existing contextFor validates actual prefixes and rebases owned points;
      // admission checks the complete newly derived original-scope geometry.
      auto derived=contextFor(r,a,*context);if(!derived)return {};
      auto admitted=registry.admitContext(*derived);if(!admitted)return {};
      prefix_contexts.emplace(key,*admitted);return admitted;
    }
    struct AssignedUse {
     private:
      const BoundedProducts* owner=nullptr;std::uint64_t epoch=0;
      Use original{},canonical{};Registry::ContextId context;
      friend struct BoundedProducts;
    };
    // Functional assignment transport, not body/family identity by shape.
    // The returned value retains the exact ORIGINAL use and the full scoped
    // proof context. Internal future Fold binders keep the ORIGINAL rank.
    std::optional<AssignedUse> canonicalUse(Use original,Registry::ContextId assignment,bool coordinate=false) {
      const auto* context=registry.context(assignment);
      if(!context || original.scope>=registry.scopes.size() || original.body>=registry.heads.snapshot.size() ||
          !registry.heads.charge(context->axis_count+2))return {};
      for(unsigned i=0;i<context->axis_count;++i) {
        const auto& axis=context->axes[i];
        if(!axis.actual_level || *axis.actual_level!=axis.reference_level || axis.offset || axis.actual_shift ||
            axis.point || axis.coordinate_image ||
            registry.count(context->reference_scope,axis.reference_level)!=std::optional{axis.extent} ||
            registry.count(context->actual_scope,*axis.actual_level)!=std::optional{axis.extent})return {};
      }
      original.body=registry.heads.owner(original.body);
      if(coordinate) {
        const auto& k=registry.heads.snapshot[original.body];
        if((k.type!="u64" && k.type!="exact:u64") || !k.provenance.empty())return {};
        auto affine=registry.heads.affine(original.body);
        if(!affine || !registry.boundedAffine(*affine,original.scope))return {};
      } else if(!admitted(original))return {};
      const auto rank=registry.bounds.rank(registry.scopes[original.scope].bounds);if(!rank)return {};
      auto target=registry.prefixScope(context->reference_scope,*rank);if(!target)return {};
      const Use canonical{original.body,*target};
      auto local=contextFor(canonical,original,assignment);
      if(!local || !related(canonical,original,*local) || !registry.heads.charge(6))return {};
      AssignedUse result;result.owner=this;result.epoch=registry.context_epoch;
      result.original=original;result.canonical=canonical;result.context=*local;return result;
    }
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
    std::optional<Groups> group(std::span<const Use> input,bool strict_real_products=false) {
      auto normalized=factors(input,strict_real_products);if(!normalized || !registry.heads.charge(2))return {};
      Groups result;result.factor_count=normalized->size();
      for(auto use:*normalized) {
        auto tag=registry.heads.shallow(use.body);if(!tag)return {};
        std::size_t i=0;
        for(;i<result.size;++i) {
          if(!registry.heads.charge())return {};
          if(result.data[i].tag==*tag)break;
        }
        if(i<result.size) {
          if(result.data[i].use!=use || !registry.heads.charge())return {};
          ++result.data[i].copies;
        } else {
          if(result.size==result.data.size() || !registry.heads.charge(4))return {};
          result.data[result.size++]={*tag,use,1};
        }
      }
      if(!registry.heads.charge(4*result.size))return {}; // Active-only optional return copy.
      return result;
    }
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
    std::optional<std::vector<FactorPair>> pairedFactors(Use reference,Use actual) {
      // SAME selected <=8-factor factory and multiplicity-preserving matching,
      // exposed as actual Uses, not a relation proof or fabricated Context.
      // Physical Engine properties may not admit an adapter-owned namespace.
      if(registry.heads.adapter)return {};
      // This new ideal-real paired route may not reassociate a raw rounded
      // f32/f64 product nested beneath an explicit real parent. Keep the SAME
      // selected factor parser; existing callers retain their original policy.
      auto r=group(std::span(&reference,1),true),a=group(std::span(&actual,1),true);
      if(!r || !a || !registry.heads.charge(4*r->size+1))return {};
      std::vector<FactorPair> result;result.reserve(r->size);
      if(!pairGroups(*r,*a,[&](Use p,Use q){result.emplace_back(p,q);return true;}))return {};
      return result;
    }
    std::optional<std::pair<std::size_t,std::size_t>> equal(std::span<const Use> reference,
        std::span<const Use> actual,const Registry::Context& context) {
      if(!registry.validate(context))return {};
      auto plan=referenceProduct(reference);if(!plan)return {};
      return equalReference(*plan,actual,[&](Use r,Use a) {
        auto local=contextFor(r,a,context);return local && related(r,a,*local);
      });
    }
    std::optional<std::pair<std::size_t,std::size_t>> equal(std::span<const Use> reference,
        std::span<const Use> actual,Registry::ContextId context) {
      if(!registry.context(context))return {};
      auto plan=referenceProduct(reference);return plan?equal(*plan,actual,context):std::nullopt;
    }
    std::optional<std::pair<std::size_t,std::size_t>> equal(ReferenceId reference,
        std::span<const Use> actual,Registry::ContextId context) {
      if(!registry.context(context))return {};
      return equalReference(reference,actual,[&](Use r,Use a) {
        auto local=contextFor(r,a,context);return local && related(r,a,*local);
      });
    }
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
    // Necessary complete capture facts only. A conservative dependency union
    // may reject a valid piece; it never permits a factor to cross a binder.
    std::optional<std::vector<JoinedFamilies::Capture>> captures(
        std::span<const Use> factors,Registry::Id scope,
        std::span<const std::uint32_t> axes) {
      std::vector<JoinedFamilies::Capture> result;
      for(auto use:factors) {
        if(!products.admitted(use) || !registry.prefix(use.scope,scope))return {};
        const auto body=registry.heads.owner(use.body);
        const auto& free=registry.heads.engine.free_indices_[body];
        if(!registry.heads.charge(free.size()+1))return {};
        for(auto level:free) {
          if(std::find(axes.begin(),axes.end(),level)!=axes.end())continue;
          if(level>=registry.scopes[use.scope].domains.size())return {};
          result.emplace_back(level,registry.scopes[use.scope].domains[level]);
        }
      }
      if(!registry.heads.charge(result.size()*(1+std::bit_width(result.size()+1))))return {};
      std::sort(result.begin(),result.end());result.erase(std::unique(result.begin(),result.end()),result.end());return result;
    }
    std::optional<Id> join(std::span<const Piece> input,JoinedFamilies::Box domain) {
      if(input.empty() || input.size()>=1024 || families.size()>=256 ||
          !registry.heads.charge(input.size()*(28+std::bit_width(input.size()+1))))return {};
      std::vector<Piece> checked;std::vector<JoinedFamilies::Piece> geometric;
      checked.reserve(input.size());geometric.reserve(input.size());
      std::map<PieceKey,std::size_t> positions;
      for(const auto& piece:input) {
        if(!piece.count || piece.count>piece.factors.size())return {};
        const auto original=std::span(piece.factors.data(),piece.count);
        auto factors=products.factors(original);if(!factors || factors->empty())return {};
        auto scope=products.common(*factors);if(!scope)return {};
        std::array<Registry::Image,2> images;
        for(unsigned axis=0;axis<2;++axis) {
          auto image=registry.view(piece.images[axis]);
          if(!image || image->scope!=*scope || (axis && image->level==images[axis-1].level))return {};
          images[axis]=*image;
        }
        const std::array<std::uint32_t,2> axes{images[0].level,images[1].level};
        auto all_captures=captures(*factors,*scope,axes);if(!all_captures)return {};
        // Geometry's body anchor must carry exactly ALL product captures. Its
        // existing family proof is used only for intervals/scopes, never as an
        // asserted equality of the omitted product factors.
        std::optional<Use> anchor;std::string_view type;
        for(auto use:*factors) {
          const auto body=registry.heads.owner(use.body);
          if(type.empty())type=registry.heads.snapshot[body].type;
          else if(type!=registry.heads.snapshot[body].type)return {};
          if(use.scope==*scope) {
            auto own=captures(std::span(&use,1),*scope,axes);
            if(!own)return {};if(*own==*all_captures && !anchor)anchor=use;
          }
        }
        if(!anchor)return {};
        JoinedFamilies::Piece gp{anchor->body,*scope,piece.images};
        Piece cp;cp.count=static_cast<unsigned>(factors->size());cp.images=piece.images;
        std::copy(factors->begin(),factors->end(),cp.factors.begin());
        cp.reference_product=products.referenceProduct(std::span(cp.factors.data(),cp.count));
        if(!cp.reference_product)return {};
        const PieceKey key{registry.heads.owner(gp.body),gp.scope,
            images[0].level,images[0].coordinate,images[1].level,images[1].coordinate,
            JoinedFamilies::Box{{{images[0].begin,images[0].end},{images[1].begin,images[1].end}}}};
        if(!registry.heads.charge(12+6*std::bit_width(positions.size()+1)) ||
            !positions.emplace(key,checked.size()).second)return {};
        checked.push_back(cp);geometric.push_back(gp);
      }
      auto joined=geometry.join(geometric,2,domain);if(!joined)return {};
      const auto* layout=geometry.view(*joined);if(!layout || layout->pieces.size()!=checked.size())return {};
      auto family=std::make_unique<Family>(Family{*joined,{}});family->pieces.reserve(checked.size());
      for(const auto& piece:layout->pieces) {
        auto t=registry.view(piece.images[0]),s=registry.view(piece.images[1]);if(!t || !s)return {};
        const PieceKey key{registry.heads.owner(piece.body),piece.scope,t->level,t->coordinate,s->level,s->coordinate,
            JoinedFamilies::Box{{{t->begin,t->end},{s->begin,s->end}}}};
        if(!registry.heads.charge(12+6*std::bit_width(positions.size()+1)))return {};
        auto pos=positions.find(key);if(pos==positions.end())return {};family->pieces.push_back(checked[pos->second]);
      }
      Id result;result.owner=this;result.id=families.size();families.push_back(std::move(family));return result;
    }
    // Prove a complete state-vector sample at a retained exact-u64 scalar or
    // private actual-dispatch singleton. Both use the SAME original-image
    // assignment and complete typed/scoped factor relation below.
    bool sample(Id id,std::span<const Use> actual,const Registry::ScalarPoint& global_point,std::uint32_t state_axis) {
      const auto* family=view(id);if(!family || actual.empty())return false;
      const auto* layout=geometry.view(family->geometry);if(!layout)return false;
      auto actual_scope=products.common(actual);if(!actual_scope)return false;
      auto global=registry.globalValue(global_point);
      auto extent=registry.count(*actual_scope,state_axis);
      if(!global || !extent ||
          layout->domain[1]!=std::pair<std::uint64_t,std::uint64_t>{0,*extent})return false;
      const auto point=*global;
      if(point<layout->domain[0].first || point>=layout->domain[0].second)return false;
      if(!registry.heads.charge(1+std::bit_width(layout->boxes.size()+1)))return false;
      const auto first=std::lower_bound(layout->boxes.begin(),layout->boxes.end(),point,
          [](const auto& box,std::uint64_t p){return box[0].second<=p;});
      if(first==layout->boxes.end() || point<(*first)[0].first)return false;
      const auto band=(*first)[0];
      for(std::size_t i=first-layout->boxes.begin();i<layout->boxes.size() && layout->boxes[i][0]==band;++i) {
        if(!registry.heads.charge())return false;
        const auto& piece=family->pieces[i];const auto& original=layout->pieces[i];
        if(!piece.reference_product)return false;
        auto context=registry.sampleContext(piece.images[0],piece.images[1],*actual_scope,global_point,state_axis);
        const auto* assignment=context?registry.context(*context):nullptr;
        if(!assignment || assignment->reference_scope!=original.scope ||
            !products.equal(*piece.reference_product,actual,*context))return false;
      }
      return true;
    }
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
    std::optional<Step> next() {
      auto& heads=registry.heads;const auto& snapshot=heads.snapshot;
      constexpr Class none=std::numeric_limits<Class>::max();
      if(closed || scope>=registry.scopes.size() || !heads.charge() || !active.insert(current).second)return {};
      auto selected=heads.compact(current);if(!selected)return {};
      const auto& select=snapshot[*selected];
      if(!idealSelect(select)) {
        auto domain=image?registry.view(*image):std::nullopt;
        if(!domain || domain->derived || domain->end<=domain->begin ||
            excluded.size()+1!=domain->end-domain->begin ||
            !coverage.cover(excluded,{domain->begin,domain->end-1}) ||
            !heads.charge(8+3*((sizeof(Registry::DispatchPointId)+sizeof(Class)-1)/sizeof(Class))))return {};
        // All earlier actual Eq exclusions are unit-weight, disjoint and
        // complete below end-1. The actual false-path remainder is exactly
        // this last singleton; this token is a scalar proof, never a Class.
        Registry::DispatchPointId point;point.owner=&registry;point.epoch=registry.context_epoch;
        point.image=image->id;point.root=root;point.branch=current;point.value=domain->end-1;
        closed=true;return Step{point.value,point,current,true,true};
      }
      auto predicate_head=heads.compact(select.children[0]);if(!predicate_head)return {};
      const auto& predicate=snapshot[*predicate_head];
      if(predicate.children.size()!=2 || comparison(predicate)!=std::optional<std::string>{"eq"})return {};
      Class point=none,axis=none;std::uint64_t endpoint=0;
      for(unsigned side=0;side<2;++side) {
        const auto a=heads.owner(predicate.children[side]),b=heads.owner(predicate.children[1-side]);
        if(snapshot[a].type!=snapshot[b].type || (snapshot[a].type!="u64" && snapshot[a].type!="exact:u64"))return {};
        auto pa=heads.affine(a),ba=heads.affine(b);
        if(pa && ba && pa->coefficients.empty() && ba->coefficients.size()==1 &&
            ba->coefficients.begin()->second==1 &&
            ba->coefficients.begin()->first<heads.engine.options_.outer_binders) {
          if(point!=none)return {};point=a;axis=b;endpoint=pa->constant;
        }
      }
      if(point==none)return {};
      if(coordinate==none) {
        coordinate=axis;auto a=heads.affine(axis);
        image=registry.image(scope,a->coefficients.begin()->first,axis);if(!image)return {};
      } else if(axis!=coordinate)return {};
      auto domain=registry.view(*image);if(!domain)return {};
      const bool inside=endpoint>=domain->begin && endpoint<domain->end;
      if(inside) {
        if(endpoint==UINT64_MAX || excluded.size()>=512 || !heads.charge(2))return {};
        excluded.emplace_back(endpoint,endpoint+1);
      }
      const auto branch=heads.owner(select.children[1]);current=heads.owner(select.children[2]);
      return Step{endpoint,point,branch,inside,false};
    }
  };
  // Bounded actual-source cohort bridge. A zero-entry frame output is
  // certified only after the recurrence,
  // dispatch, projection and causal region all share a complete scoped proof.
  // One immutable cohort epoch/aggregate allowance; no allowance per output.
  bool proveFirstFinitePrefixOutput(std::span<const Class> left,std::span<const Class> right) {
    namespace fp=finite_prefix_certificate;
    namespace cr=causal_region_certificate;
    using Registry=BodyScopeRegistry;
    using Use=BoundedProducts::Use;
    constexpr Class none=std::numeric_limits<Class>::max();
    constexpr Registry::Id virtual_scope=UINT32_MAX;
    std::size_t initial_work=epoch_work_allowance::default_work,remaining=initial_work,policy_work=0;
    epoch_work_allowance::Size physical_epoch;
    std::vector<std::pair<Class,Class>> proved_roots;
    std::size_t roots_attempted=0,completed_frames=0;
    std::size_t source_cases=0,attached=0,products_count=0,p_samples=0,q_samples=0;
    std::size_t proved_updates=0,scope_count=0,coverage_work=0;
    std::size_t potential_coverage_work=0,product_coverage_work=0;
    bool potential_coverage_cap=false,product_coverage_cap=false;
    unsigned candidate_flags=0,candidate_factors=0,source_factors_count=0;
    unsigned selected_projection_count=0,selected_weight_count=0;
    std::size_t bounds_scopes=0,header_entries=0,relation_entries=0,original_views=0,transported_uses=0;
    std::size_t context_entries=0,sample_context_entries=0,sample_context_hits=0,reference_plans=0,reference_plan_reads=0;
    std::size_t context_capacity=0,context_plan_rejections=0;
    std::size_t retained_record_rejections=0,image_entries=0,identity_slots=0;
    const char* retained_record_cap="none";
    const char* context_plan_cap="none";
    unsigned product_calls=0,last_product_kind=0,target_calls=0,mask_calls=0;
    std::string_view entry_stage="none";
    bool region_complete=false,dispatch_complete=false,region_cap=false;
    cr::Statistics region_statistics;
    bool body_cap=false,scope_cap=false;
    const char* stage="configuration";bool component=false,output_proved=false;
    const bool prefix_trace=sentryward::debug::environment("SENTRYWARD_EGRAPH_PREFIX_TRACE")!=nullptr;
    enum class PrefixPhase : unsigned {Source,Templates,Sampling,Output,Carry,Region,Dispatch,Count};
    std::array<std::size_t,static_cast<unsigned>(PrefixPhase::Count)> phase_work{};
    struct PhaseTrace {
      bool enabled;std::size_t& remaining;decltype(phase_work)& work;
      std::size_t previous;PrefixPhase phase=PrefixPhase::Source;
      void finish() {
        if(enabled){work[static_cast<unsigned>(phase)]+=previous-remaining;previous=remaining;}
      }
      void next(PrefixPhase value) {finish();phase=value;}
      ~PhaseTrace(){finish();}
    };
    struct RootTrace {
      std::size_t ordinal,work,cases,attachments,updates,frames,proved;
      const char* stage;std::string_view entry;
      struct Coordinate {
        Class predicate=UINT32_MAX,time=UINT32_MAX,output=UINT32_MAX;
        std::array<Class,2> children{UINT32_MAX,UINT32_MAX};
        std::array<std::uint64_t,2> constants{},coefficients{};
        std::array<std::uint32_t,2> axes{UINT32_MAX,UINT32_MAX};
        std::array<std::size_t,2> terms{};
        std::uint64_t source_begin=0,source_end=0,time_begin=0,time_end=0;
        Registry::Id original_scope=UINT32_MAX,piece_scope=UINT32_MAX;
        std::uint32_t binder=UINT32_MAX;unsigned candidate=0;
      } coordinate;
    };
    RootTrace::Coordinate coordinate_trace;
    std::array<RootTrace,64> root_trace{};std::size_t root_trace_size=0;
    unsigned carry_denial_trace=0;
    std::size_t member_queries=0,member_hits=0,member_entries=0;
    fp::Statistics prefix_statistics{};
    const auto run=[&]() {
      if(left.empty() || left.size()!=right.size() || !options_.ideal_numeric_outputs ||
          !options_.outer_binders || options_.outer_binders+1>=Registry::max_depth ||
          options_.outer_index_extents.size()!=options_.outer_binders)return;
      // Necessary O(stored roots) syntax filter before any epoch catalog.
      // This schedules only; every actual head/path is still re-read below.
      bool candidate=false;
      for(std::size_t i=0;i<left.size();++i) {
        poll();
        if(!remaining){stage="configuration_cap";return;}--remaining;
        const auto& l=nodes_[find(left[i])];const auto& r=nodes_[find(right[i])];
        if((idealSelect(l) || (l.type=="real" && arithmetic(l) && l.name=="Add")) &&
            ((arithmetic(r) && r.name=="Add") ||
            (r.kind==static_cast<int>(s::Kind::Fold) && r.fold && r.fold->reducer==s::Reducer::Sum)))
          {candidate=true;break;}
      }
      if(!candidate)return;
      const auto plan=physicalWorkPlan(nodes_,initial_work,initial_work-remaining);
      initial_work=plan.initial;remaining=plan.remaining;policy_work=plan.sizing;physical_epoch=plan.physical;
      if(plan.exhausted){stage="configuration_cap";return;}
      propagateProperties();
      // Borrow the maintained immutable membership epoch. A requested owner
      // copies only its capped physical prefix, once; never reconstruct all
      // stored classes or walk the graph to discover two exact Index heads.
      const auto boundary=nodes_.size(),owner_epoch=unions_;
      struct MemberView {
        std::size_t boundary;std::function<std::span<const Class>(Class)> lookup;
        std::size_t size() const {return boundary;}
        std::span<const Class> operator[](Class id) const {return lookup(id);}
      };
      std::unordered_map<Class,std::vector<Class>> member_views;
      BodyHeadIndex* member_budget=nullptr;
      MemberView members{boundary,[&](Class cls)->std::span<const Class> {
        if(unions_!=owner_epoch || nodes_.size()!=boundary)
          throw std::logic_error("prefix catalog epoch changed");
        if(cls>=boundary || !member_budget)return {};
        ++member_queries;cls=find(cls);
        if(auto found=member_views.find(cls);found!=member_views.end()) {
          // Like the existing linear body_members view, a cached span stays
          // stable for repeated size/front reads within ONE head operation.
          // Do not turn its second read into an empty span on local-cap exit;
          // the unchanged actual head/matcher charge points deny that query.
          ++member_hits;return found->second;
        }
        if(!member_budget->charge())return {};
        auto view=class_members_.prefix(cls,boundary,BodyHeadIndex::max_members,
            [&](std::size_t amount){return member_budget->charge(amount);});
        // A failed query-local allowance must not poison later cohort uses.
        if(!view || !member_budget->charge(2))return {};
        auto found=member_views.emplace(cls,std::move(*view)).first;
        ++member_entries;return found->second;
      }};
      AffineCache affine;BodyHeadIndex heads(*this,nodes_,members,affine,&remaining);
      member_budget=&heads;
      // ONE immutable cohort index. The existing local header/sole tokens are
      // shared across all roots; no new allowance or per-root array is created.
      heads.initializeSoleHints();
      const auto indexCoordinate=[&](std::uint32_t level) {
        Class selected=none;
        for(const auto* type:{"u64","exact:u64"}) {
          if(!heads.charge(16+3*std::char_traits<char>::length(type)))return none;
          Key probe;probe.kind=static_cast<int>(s::Kind::Index);probe.type=type;probe.index_level=level;
          auto found=memo_.find(probe);
          if(found!=memo_.end())selected=std::min(selected,found->second);
        }
        return selected==none?none:find(selected);
      };
      const Class state_coordinate=indexCoordinate(options_.outer_binders);
      const Class nested_state_coordinate=indexCoordinate(options_.outer_binders+1);
      Registry registry(heads,epoch_work_allowance::retainedHeaderCapacity(physical_epoch));
      JoinedProducts products(registry,initial_work);JoinedFamilies potentials(registry,initial_work);
      struct Counters {
        Registry& registry;JoinedProducts& products;JoinedFamilies& potentials;
        std::size_t& scopes;std::size_t& bounds;std::size_t& headers;std::size_t& relations;std::size_t& coverage;
        bool& body_cap;bool& scope_cap;std::size_t& originals;std::size_t& transports;
        std::size_t& potential_work;std::size_t& product_work;bool& potential_cap;bool& product_cap;
        std::size_t& contexts;std::size_t& samples;std::size_t& sample_hits;std::size_t& plans;std::size_t& plan_reads;
        std::size_t& context_capacity;std::size_t& context_rejections;const char*& context_cap;
        std::size_t& record_rejections;const char*& record_cap;std::size_t& images;std::size_t& identities;
        ~Counters() {
          scopes=registry.scopes.size();const auto s=registry.bounds.statistics();bounds=s.scopes;scope_cap=s.exhausted;
          headers=registry.heads.roots_cache.size();relations=products.products.relations.size();body_cap=registry.heads.exhausted;
          coverage=potentials.coverageWork()+products.geometry.coverageWork();
          potential_work=potentials.coverageWork();product_work=products.geometry.coverageWork();
          potential_cap=potentials.intervalExhausted();product_cap=products.geometry.intervalExhausted();
          originals=registry.original_body_views;transports=registry.transported_body_uses;
          contexts=registry.contexts.size();samples=registry.sample_contexts.size();sample_hits=registry.sample_context_hits;
          plans=products.products.reference_products.size();plan_reads=products.products.reference_reads;
          context_capacity=registry.max_retained_records;context_rejections=registry.context_plan_rejections;
          context_cap=registry.context_plan_cap;
          record_rejections=registry.retained_record_rejections;record_cap=registry.retained_record_cap;
          images=registry.images.size();identities=registry.identity_slots;
        }
      } counters{registry,products,potentials,scope_count,bounds_scopes,header_entries,relation_entries,
          coverage_work,body_cap,scope_cap,original_views,transported_uses,
          potential_coverage_work,product_coverage_work,potential_coverage_cap,product_coverage_cap,
          context_entries,sample_context_entries,sample_context_hits,reference_plans,reference_plan_reads,
          context_capacity,context_plan_rejections,context_plan_cap,
          retained_record_rejections,retained_record_cap,image_entries,identity_slots};
      auto outer=registry.output();if(!outer || state_coordinate==none){stage="outer_scope";return;}
      const auto head=[&](Class cls)->const Key* {
        if(cls>=nodes_.size())return nullptr;
        auto id=heads.compact(cls);return id?&nodes_[*id]:nullptr;
      };
      const auto prefix_node=[&](fp::ExprId expression,fp::ScopeId scope)->std::optional<fp::Node> {
        if(expression>=nodes_.size() || scope>=registry.scopes.size())return {};
        const auto cls=find(static_cast<Class>(expression));auto k=head(cls);
        if(!k || !realType(k->type) || !k->provenance.empty() || !finite_[cls] || !heads.charge())return {};
        fp::Node node;node.type=k->type;node.attributes=k->attributes;node.provenance=k->provenance;
        const auto& free=free_indices_[cls];
        if(free.size()>node.free_indices.size() || !heads.charge(free.size()+1))return {};
        for(auto level:free)if(!registry.count(scope,level))return {};
        std::copy(free.begin(),free.end(),node.free_indices.begin());node.free_count=free.size();
        // A conservative certified dependency upper bound is sufficient for
        // absence checks; it can reject, but cannot invent independence.
        node.exact_dependencies=true;node.finite_ideal=true;
        // This is an existing certified value fact of this e-class.
        // Selected Fold/Multiply aliases must not hide it from the SAME parser.
        if(zero_[cls]) {node.kind=fp::Kind::Zero;node.attributes={};return node;}
        if(k->kind==static_cast<int>(s::Kind::Constant) && k->attributes.empty()) {
          if(k->name=="0")node.kind=fp::Kind::Zero;
          else if(k->name=="1")node.kind=fp::Kind::One;
        } else if(k->kind==static_cast<int>(s::Kind::InitialCell)) {
          if(k->children.size()!=1)return {};
        } else if(arithmetic(*k) || idealOperation(*k)) {
          if(k->name=="Add")node.kind=fp::Kind::Add;
          else if(k->name=="Subtract")node.kind=fp::Kind::Subtract;
          else if(k->name=="Negate")node.kind=fp::Kind::Negate;
          else if(k->name=="Multiply")node.kind=fp::Kind::Multiply;
          else if(k->name=="Exp")node.kind=fp::Kind::NaturalExp;
          else return {};
          if(k->children.size()>node.children.size())return {};
          node.arity=k->children.size();
          for(unsigned i=0;i<node.arity;++i)node.children[i]=find(k->children[i]);
        } else return {};
        return node;
      };
      // Private semantic descriptors live for this ONE immutable proof epoch.
      // Every descriptor is populated only from its completed joined-family
      // admission; different frames never share a virtual expression identity.
      std::unordered_map<fp::ExprId,fp::Node> virtual_prefix_nodes;
      std::unordered_map<cr::ExprId,cr::Node> joined_projection_nodes;
      const auto region_node=[&](cr::Use use)->std::optional<cr::Node> {
        if(use.scope==virtual_scope) {
          auto found=joined_projection_nodes.find(use.expression);
          return found==joined_projection_nodes.end()?std::nullopt:std::optional{found->second};
        }
        auto p=prefix_node(use.expression,use.scope);auto k=head(use.expression);
        if(!k || use.scope>=registry.scopes.size())return {};
        cr::Node n;
        if(p) {
          n.type=p->type;n.attributes=p->attributes;n.provenance=p->provenance;
          n.finite_ideal=p->finite_ideal;n.exact_dependencies=p->exact_dependencies;
          n.free_indices=p->free_indices;n.free_count=p->free_count;n.arity=p->arity;
          for(unsigned i=0;i<n.arity;++i)n.children[i]=p->children[i];
          switch(p->kind) {
            case fp::Kind::Zero:n.kind=cr::Kind::Zero;break;
            case fp::Kind::Add:n.kind=cr::Kind::Add;break;
            case fp::Kind::Subtract:n.kind=cr::Kind::Subtract;break;
            case fp::Kind::Negate:n.kind=cr::Kind::Negate;break;
            case fp::Kind::Multiply:n.kind=cr::Kind::Multiply;break;
            case fp::Kind::NaturalExp:n.kind=cr::Kind::NaturalExp;break;
            default:break;
          }
          return n;
        }
        const auto cls=find(use.expression);
        if(!realType(k->type) || !k->provenance.empty() || !finite_[cls] || !heads.charge())return {};
        if(k->kind==static_cast<int>(s::Kind::Fold) && k->fold && k->fold->reducer==s::Reducer::Sum &&
            k->children.size()==2 && k->attributes.empty()) {
          n.kind=cr::Kind::Sum;n.binder=k->fold->binder_level;n.count=k->fold->count;
        } else if(idealSelect(*k))n.kind=cr::Kind::Select;else return {};
        n.type=k->type;n.attributes=k->attributes;n.provenance=k->provenance;
        n.finite_ideal=true;n.exact_dependencies=true;n.arity=k->children.size();
        for(unsigned i=0;i<n.arity;++i)n.children[i]=find(k->children[i]);
        const auto& free=free_indices_[cls];if(free.size()>8 || !heads.charge(free.size()+1))return {};
        for(auto level:free)if(!registry.count(use.scope,level))return {};
        std::copy(free.begin(),free.end(),n.free_indices.begin());n.free_count=free.size();return n;
      };
      // Source syntax is selected locally; every actual update is later
      // rechecked by the existing core, and every P/Q leaf by the full mapper.
      struct Update {Class previous,current_p,previous_p,injection;};
      const auto update=[&](Class state,Class zero_baseline=std::numeric_limits<Class>::max())->std::optional<Update> {
        auto k=head(state);if(!k || !arithmetic(*k) || k->name!="Add")return {};
        std::optional<Update> result;
        for(unsigned arm=0;arm<2;++arm) {
          auto m=head(k->children[arm]);if(!m || !arithmetic(*m) || m->name!="Multiply")continue;
          for(unsigned side=0;side<2;++side) {
            auto exp=head(m->children[side]);
            if(!exp || !idealOperation(*exp) || exp->name!="Exp" || exp->children.size()!=1)continue;
            auto difference=head(exp->children[0]);if(!difference)return {};
            std::optional<std::pair<Class,Class>> pair;
            if(idealOperation(*difference) && difference->name=="Subtract" && difference->children.size()==2)
              pair=std::pair{find(difference->children[0]),find(difference->children[1])};
            else if(arithmetic(*difference) && difference->name=="Add")
              for(unsigned sign=0;sign<2;++sign) {
                auto negative=head(difference->children[sign]);
                if(negative && idealOperation(*negative) && negative->name=="Negate" && negative->children.size()==1) {
                  if(pair)return {};pair=std::pair{find(difference->children[1-sign]),find(negative->children[0])};
                }
              }
            // The existing core admits Exp(P) as Exp(P-E) only when this
            // ACTUAL baseline is a certified ideal-real zero. State carry is
            // a separate nonzero operand and is never erased by this hint.
            if(!pair && zero_baseline<nodes_.size() && nodes_[find(zero_baseline)].type=="real" &&
                finite_[find(zero_baseline)] && zero_[find(zero_baseline)])
              pair=std::pair{find(exp->children[0]),find(zero_baseline)};
            if(pair) {
              if(result)return {};
              result=Update{find(m->children[1-side]),pair->first,pair->second,find(k->children[1-arm])};
            }
          }
        }
        return result;
      };
      // ONE core/region builder and ONE aggregate ledger, not a replay or
      // fresh allowance per root. Actual query callbacks are installed only
      // while their complete original-root attachment records are alive.
      fp::Relate core_relate;
      cr::RelateProduct region_product;cr::ProveMask region_mask;cr::ProveOutputTarget region_target;
      cr::ProveStateTarget region_state_target;
      fp::Limits prefix_limits;prefix_limits.max_work=initial_work;
      prefix_limits.max_states=epoch_work_allowance::retainedHeaderCapacity(physical_epoch);
      cr::Limits region_limits;region_limits.max_work=initial_work;
      region_limits.max_nodes=std::max(region_limits.max_nodes,
          epoch_work_allowance::retainedHeaderCapacity(physical_epoch));
      fp::Builder core([&](auto e,auto scope)->std::optional<fp::Node> {
            if(scope==virtual_scope) {
              auto found=virtual_prefix_nodes.find(e);
              return found==virtual_prefix_nodes.end()?std::nullopt:std::optional{found->second};
            }
            return prefix_node(e,scope);
          },[&](const auto& request){return core_relate?core_relate(request):fp::RelationResult{};},
          prefix_limits,[&](auto n){return heads.charge(n);});
      struct PrefixStatisticsSnapshot {
        fp::Builder& core;fp::Statistics& stored;
        ~PrefixStatisticsSnapshot(){stored=core.statistics();}
      } prefix_statistics_snapshot{core,prefix_statistics};
      cr::Builder regions(core,region_node,
          [&](cr::Use use,std::size_t allowance) {
            const auto before=remaining;std::size_t local=allowance;BodyHeadIndex::QueryBudget guard(heads,&local);
            auto selected=heads.compact(use.expression);
            auto scope=selected?registry.insideFold(use.scope,*selected):std::nullopt;
            return cr::ScopeResult{scope,before-remaining};
          },[&](const auto& request){return region_product?region_product(request):cr::ProductResult{};},
          [&](const auto& request){return region_mask?region_mask(request):fp::RelationResult{};},
          [&](auto n){return heads.charge(n);},region_limits,
          [&](const auto& request){return region_target?region_target(request):cr::OutputTargetResult{};},
          [&](const auto& request){return region_state_target?region_state_target(request):cr::StateTargetResult{};});
      struct RegionStatisticsSnapshot {
        cr::Builder& regions;cr::Statistics& stored;bool& stopped;bool enabled;
        ~RegionStatisticsSnapshot(){if(enabled){stored=regions.statistics();stopped=regions.exhausted();}}
      } region_statistics_snapshot{regions,region_statistics,region_cap,prefix_trace};
      // This index schedules a real same-count original BODY scope. It is
      // never an equality fact: EVERY other occurrence is attached by the
      // existing complete same-assignment guard, including all captures.
      std::map<std::uint64_t,Registry::Id> canonical_sources;
      std::map<std::uint64_t,Registry::Id> canonical_carry_sources;
      std::map<std::pair<std::uint64_t,std::uint64_t>,Registry::Id> canonical_rhs_scopes;
      const auto samplePotential=[&](JoinedFamilies::Id definition,Use actual,const Registry::ScalarPoint& global_point) {
        const auto* family=potentials.view(definition);auto point=registry.globalValue(global_point);
        if(!family || !point ||
            !heads.charge(1+std::bit_width(family->boxes.size()+1)))return false;
        auto it=std::lower_bound(family->boxes.begin(),family->boxes.end(),*point,
            [](const auto& box,std::uint64_t n){return box[0].second<=n;});
        if(it==family->boxes.end() || *point<(*it)[0].first)return false;
        const auto& piece=family->pieces[it-family->boxes.begin()];
        auto context=registry.sampleContext(piece.images[0],{},actual.scope,global_point);
        const auto* assignment=context?registry.context(*context):nullptr;
        return assignment && assignment->reference_scope==piece.scope &&
            products.products.related({piece.body,piece.scope},actual,*context);
      };
      struct Terminal {
        fp::Builder::StateId state;fp::Builder::FrameId frame;
        fp::Builder::CompletedPrefixFamilyId completed;
        Registry::Id scope;std::uint32_t axis;std::uint64_t extent;
        fp::Interval time;Class expression,potential;Registry::ScalarPoint point;Class zero;
        JoinedFamilies::Id p;JoinedProducts::Id q;
        fp::ExprId virtual_p,virtual_q;
      };
      using TerminalKey=std::tuple<Class,Registry::Id,std::uint32_t,std::uint64_t,std::uint64_t>;
      std::deque<Terminal> terminals;
      std::map<TerminalKey,std::optional<std::size_t>> terminal_index;
      struct Case {
        std::uint64_t endpoint;Registry::ScalarPoint point;Class state,projection,seed;
        Registry::ScopeView body_scope;
        Registry::OccurrenceUse state_use,projection_use;
        // The ACTUAL selected Fold/member and parent remain in body_scope;
        // both original BODY operand paths remain in the two use handles.
      };
      struct SourceFrameRequest {
        std::span<const Case> cases;std::span<const Update> updates;
        Registry::Id state_scope;
        std::uint32_t actual_state_axis,template_time_axis,template_state_axis;
        std::optional<std::uint32_t> output_time_axis;
        std::uint64_t state_extent;fp::Interval time;
        Class entry_state,entry_potential;
        const Terminal* carry;
        JoinedFamilies::Id potential;JoinedProducts::Id injection;
        fp::ExprId virtual_p,virtual_q;
      };
      struct SourceFrame {
        fp::Builder::FrameRequest definition;
        fp::Builder::FrameId frame;fp::Builder::StateId terminal;
        fp::Builder::CompletedPrefixFamilyId completed;
        std::vector<fp::Builder::StateId> states;
      };
      // ONE existing exact source-frame proof. Producers keep their ORIGINAL
      // output/coordinate/path attachments; this callable does not certify a
      // producer's whole output, fabricate a projection, or admit a point.
      // Every stored update still passes the SAME core and full P/Q samplers.
      const auto certifySourceFrame=[&](const SourceFrameRequest& source,PhaseTrace& phase)
          ->std::optional<SourceFrame> {
        if(!heads.charge(16) || source.cases.empty() || source.cases.size()!=source.updates.size())return {};
        if(source.time.begin>=source.time.end ||
            source.time.end-source.time.begin!=source.cases.size())return {};
        auto endpoint=source.time.begin;
        for(const auto& c:source.cases) {
          if(!heads.charge(3) || c.endpoint!=endpoint)return {};
          auto point=registry.globalValue(c.point);if(!point || *point!=endpoint)return {};
          ++endpoint;
        }
        if(endpoint!=source.time.end)return {};
        const auto* q_family=products.view(source.injection);
        const auto* p_family=potentials.view(source.potential);
        const auto* q_geometry=q_family?products.geometry.view(q_family->geometry):nullptr;
        if(!p_family || !q_geometry || p_family->rank!=1 || q_geometry->rank!=2 ||
            p_family->type!=q_geometry->type || source.template_time_axis==source.template_state_axis ||
            p_family->domain[0]!=std::pair{source.time.begin,source.time.end} ||
            q_geometry->domain[0]!=std::pair{source.time.begin,source.time.end} ||
            q_geometry->domain[1]!=std::pair<std::uint64_t,std::uint64_t>{0,source.state_extent} ||
            registry.count(source.state_scope,source.actual_state_axis)!=std::optional{source.state_extent})return {};
        // The explicit virtual labels name the ACTUAL admitted template roles,
        // not arbitrary new free axes. Original captures retain their labels.
        for(const auto& piece:p_family->pieces) {
          if(!heads.charge(3))return {};
          auto time=registry.view(piece.images[0]);
          if(!time || time->scope!=piece.scope || time->level!=source.template_time_axis)return {};
        }
        for(const auto& piece:q_geometry->pieces) {
          if(!heads.charge(6))return {};
          auto time=registry.view(piece.images[0]),state=registry.view(piece.images[1]);
          if(!time || !state || time->scope!=piece.scope || state->scope!=piece.scope ||
              time->level!=source.template_time_axis || state->level!=source.template_state_axis)return {};
        }
        const auto virtual_node=[&](fp::ExprId expression,fp::ScopeId scope)->std::optional<fp::Node> {
          if(scope!=virtual_scope || (expression!=source.virtual_p && expression!=source.virtual_q))return {};
          fp::Node n;n.type=p_family->type;n.finite_ideal=true;n.exact_dependencies=true;
          std::vector<std::uint32_t> free{source.template_time_axis};
          if(expression==source.virtual_q)free.push_back(source.template_state_axis);
          const auto& captures=expression==source.virtual_q?q_geometry->captures:p_family->captures;
          for(auto [level,identity]:captures) { (void)identity;free.push_back(level); }
          std::sort(free.begin(),free.end());free.erase(std::unique(free.begin(),free.end()),free.end());
          if(free.size()>n.free_indices.size() || !heads.charge(free.size()+1))return {};
          std::copy(free.begin(),free.end(),n.free_indices.begin());n.free_count=free.size();return n;
        };
        auto pn=virtual_node(source.virtual_p,virtual_scope),qn=virtual_node(source.virtual_q,virtual_scope);
        if(!pn || !qn || !heads.charge(2+pn->free_count+qn->free_count))return {};
        if(virtual_prefix_nodes.contains(source.virtual_p) || virtual_prefix_nodes.contains(source.virtual_q))return {};
        virtual_prefix_nodes.emplace(source.virtual_p,*pn);virtual_prefix_nodes.emplace(source.virtual_q,*qn);
        // The callback borrows only this invocation's source span. Clear it
        // before that request/span can expire, including every failed return.
        struct ResetRelation {fp::Relate& slot;~ResetRelation(){slot={};}} reset_relation{core_relate};
        core_relate=[&](const fp::RelationRequest& request) {
          const auto before=remaining;fp::RelationResult result;
          const auto finish=[&]() {result.work=before-remaining;return result;};
          if(request.reference_scope!=virtual_scope || request.actual_scope!=source.state_scope || request.axis_count!=2 ||
              request.axes[0].reference_axis!=source.template_time_axis || request.axes[0].actual_axis || request.axes[0].extent!=1 ||
              request.axes[1].reference_axis!=source.template_state_axis || request.axes[1].actual_axis!=source.actual_state_axis ||
              request.axes[1].offset || request.axes[1].extent!=source.state_extent || request.actual>=nodes_.size())return result;
          std::size_t local=request.max_work;BodyHeadIndex::QueryBudget guard(heads,&local);
          // The factory certified an exact consecutive endpoint/owned-point
          // span once. Reuse that invariant; do not search it on each P/Q use.
          if(!heads.charge(3) || request.axes[0].offset<source.time.begin ||
              request.axes[0].offset>=source.time.end)return finish();
          const auto position=request.axes[0].offset-source.time.begin;
          if(position>=source.cases.size())return finish();
          const auto* c=&source.cases[position];
          if(c->endpoint!=request.axes[0].offset)return finish();
          Use actual{find(request.actual),source.state_scope};bool proved=false;
          if(request.reference==source.virtual_p){++p_samples;proved=samplePotential(source.potential,actual,c->point);}
          if(request.reference==source.virtual_q){++q_samples;proved=products.sample(source.injection,std::span(&actual,1),c->point,source.actual_state_axis);}
          if(proved)result.status=fp::RelationStatus::Proved;return finish();
        };
        fp::Builder::FrameRequest frame;frame.potential_template=source.virtual_p;frame.injection_template=source.virtual_q;
        frame.template_scope=virtual_scope;frame.state_scope=source.state_scope;
        frame.time=source.time;frame.template_time_axis=source.template_time_axis;
        frame.template_state_axis=source.template_state_axis;frame.actual_state_axis=source.actual_state_axis;
        frame.state_extent=source.state_extent;frame.output_time_axis=source.output_time_axis;
        frame.entry_state=source.entry_state;frame.entry_potential=source.entry_potential;
        if(source.carry)frame.carry=source.carry->state;
        phase.next(PrefixPhase::Sampling);
        stage="core_begin";auto f=core.begin(frame);if(!f)return {};
        std::optional<fp::Builder::StateId> previous;std::vector<fp::Builder::StateId> certified_states;
        if(!heads.charge(source.cases.size()))return {};
        certified_states.reserve(source.cases.size());
        stage="core_advance";
        for(std::size_t i=0;i<source.cases.size();++i) {
          const auto& c=source.cases[i];const auto& u=source.updates[i];
          auto next=core.advance(*f,previous,{c.state,u.current_p,u.injection,c.endpoint});
          if(!next)return {};previous=next;certified_states.push_back(*next);++proved_updates;
        }
        auto completed=previous?core.complete(*previous):std::nullopt;
        component=component || bool(completed);stage=completed?"completed_first_frame":"complete_unknown";
        if(!completed)return {};
        ++completed_frames;
        phase.next(PrefixPhase::Output);
        if(!heads.charge(72+5*std::bit_width(terminal_index.size()+1)))return {};
        const auto& last=source.cases.back();
        const TerminalKey terminal_key{last.state,source.state_scope,source.actual_state_axis,source.state_extent,last.endpoint};
        auto [position,inserted]=terminal_index.try_emplace(terminal_key,terminals.size());
        if(inserted)terminals.push_back({*previous,*f,*completed,source.state_scope,source.actual_state_axis,
            source.state_extent,frame.time,last.state,source.updates.back().current_p,last.point,source.cases.front().seed,
            source.potential,source.injection,source.virtual_p,source.virtual_q});
        else if(position->second && terminals[*position->second].frame!=*f)
          position->second.reset(); // Ambiguous semantic frame is not a shape-selected carry.
        constexpr auto returned_fields=(sizeof(fp::Builder::FrameRequest)+sizeof(Class)-1)/sizeof(Class)+8;
        if(!heads.charge(returned_fields))return {};
        return SourceFrame{frame,*f,*previous,*completed,std::move(certified_states)};
      };
      struct StateFunctionProof {cr::Builder::Certificate source,region;fp::Interval state;};
      // One complete Identity implementation serves actual entry restrictions
      // and direct terminal roots. Every numeric sample still uses the shared
      // typed/scoped matcher; geometry does not union a partial source function.
      using StateFunction=std::function<std::optional<StateFunctionProof>(const Terminal&,Use,Class,Class,
          std::uint32_t,std::uint32_t,Registry::ImageId,Class,Registry::ContextId,bool)>;
      if(!heads.charge(24))return;
      StateFunction proveStateFunction;
      proveStateFunction=[&](const Terminal& terminal,Use actual_use,
          Class source_coordinate,Class rhs_coordinate,std::uint32_t actual_state_axis,
          std::uint32_t actual_time_axis,Registry::ImageId component_image,
          Class dependency_witness,Registry::ContextId assignment,bool restricted)
          ->std::optional<StateFunctionProof> {
        stage="carry_state_header";
        const auto* carried=&terminal;const auto* actual=&actual_use;
        const auto canonical_scope=terminal.scope,carry_scope=actual_use.scope;
        const auto state_axis=terminal.axis;
        const auto state_coordinate=source_coordinate;
        const std::optional<std::uint64_t> state_extent=terminal.extent;
        const std::optional<Registry::ImageId> image=component_image;
        const std::optional<Registry::ContextId> component_context=assignment;
        auto view=registry.view(component_image);
        const auto* old_p=potentials.view(terminal.p);const auto* old_q=products.view(terminal.q);
        const auto* old_geometry=old_q?products.geometry.view(old_q->geometry):nullptr;
        auto old_frame=core.view(terminal.frame);
        if(!view || !old_p || !old_q || !old_geometry || !old_frame ||
            view->scope!=carry_scope || view->level!=actual_state_axis ||
            view->begin>=view->end || view->end>terminal.extent ||
            registry.count(carry_scope,actual_state_axis)!=std::optional{view->end-view->begin} ||
            !registry.context(assignment))return {};
        const auto entry_count=view->end-view->begin;
        stage="carry_state_assignment";
        const auto* assigned=registry.context(assignment);
        if(!assigned || assigned->reference_scope!=canonical_scope || assigned->actual_scope!=carry_scope ||
            assigned->axis_count!=1 || assigned->axes[0].reference_level!=state_axis ||
            assigned->axes[0].actual_level!=std::optional{actual_state_axis} ||
            assigned->axes[0].offset!=view->begin || assigned->axes[0].extent!=entry_count ||
            assigned->axes[0].actual_shift || assigned->axes[0].point ||
            assigned->axes[0].coordinate_image!=std::optional{component_image})return {};
        const fp::Interval component_state{view->begin,view->end};
        const cr::StateTarget state_target{1,1,carry_scope,component_state};
        const cr::StateTarget source_target{1,1,canonical_scope,component_state};
        struct IdentityPiece {
          const cr::Candidate* syntax;Registry::ImageId time;
          Use p;Registry::Context p_context,q_context;
          BoundedProducts::ReferenceId q;
        };
        std::vector<IdentityPiece> identity_pieces;
        using IdentityKey=std::tuple<cr::ExprId,Registry::Id,Registry::Id>;
        std::map<IdentityKey,std::size_t> identity_index;
        stage="carry_state_syntax";
        auto syntax=regions.candidates({actual->body,actual->scope},cr::CandidateMode::Identity);
        if(!syntax)return {};
        struct IdentityEntry {
          const cr::Candidate* syntax;cr::Weight weight;Use actual;
          StateFunctionProof previous;
        };
        std::optional<IdentityEntry> identity_entry;
        for(const auto& piece:*syntax) {
          if(!piece.time_fold) {
            stage="carry_state_entry";
            if(identity_entry || piece.state_fold || piece.predicate || !old_frame->carry ||
                !heads.charge(8+2*piece.factor_count))return {};
            if(!carried->time.begin || !heads.charge(8+5*std::bit_width(terminal_index.size()+1)))return {};
            auto previous_terminal=terminal_index.find({find(old_frame->entry_state),canonical_scope,
                state_axis,*state_extent,carried->time.begin-1});
            if(previous_terminal==terminal_index.end() || !previous_terminal->second)return {};
            const auto* prior=&terminals[*previous_terminal->second];
            if(prior->state!=*old_frame->carry || prior->time.end!=carried->time.begin || prior->time.end>=carried->time.end ||
                prior->scope!=canonical_scope || prior->axis!=state_axis || prior->extent!=*state_extent ||
                prior->expression!=find(old_frame->entry_state))return {};
            std::optional<cr::Weight> weight;std::optional<Use> entry;
            for(unsigned i=0;i<piece.factor_count;++i) {
              const auto factor=piece.factors[i];if(factor.expression>=nodes_.size())return {};
              if(piece.weight && factor==piece.weight->factor) {
                if(weight)return {};weight=*piece.weight;
              } else {
                const auto k=head(factor.expression);
                if(!piece.weight && k && k->type=="real" && idealOperation(*k) &&
                    k->name=="Exp" && k->children.size()==1 &&
                    find(old_frame->entry_potential)<zero_.size() && zero_[find(old_frame->entry_potential)]) {
                  if(weight)return {};
                  weight=cr::Weight{factor,{k->children[0],factor.scope},
                      {old_frame->entry_potential,canonical_scope}};
                } else {if(entry)return {};entry=Use{find(factor.expression),factor.scope};}
              }
            }
            if(!weight || !entry || entry->scope!=carry_scope ||
                registry.count(entry->scope,actual_state_axis)!=std::optional{entry_count})return {};
            auto prior_image=registry.imageInOriginalScope(component_image,entry->scope,entry->body);
            if(!prior_image)return {};
            // The private carry's source scope/count are unchanged. Only its
            // actual component function is restricted by this SAME assignment.
            auto previous=proveStateFunction(*prior,*entry,source_coordinate,rhs_coordinate,
                actual_state_axis,actual_time_axis,*prior_image,entry->body,assignment,restricted);
            if(!previous || previous->state!=component_state || !heads.charge(24))return {};
            identity_entry=IdentityEntry{&piece,*weight,*entry,*previous};continue;
          }
          stage="carry_state_piece_header";
          if(!piece.time_fold || piece.state_fold || !piece.weight || identity_pieces.size()>=1023)return {};
          const auto& tf=*piece.time_fold;
          if(tf.binder!=actual_time_axis || piece.weight->source_potential.scope!=tf.body_scope)return {};
          struct PotentialPiece {Use use;Registry::ImageId time;std::size_t index;};
          std::optional<PotentialPiece> p_definition;
          stage="carry_state_p_image";
          for(std::size_t i=0;i<old_p->pieces.size();++i) {
            if(!heads.charge())return {};
            const auto& reference=old_p->pieces[i];
            auto reference_time=registry.view(reference.images[0]);if(!reference_time)return {};
            auto local=registry.imageFromTypedCells({piece.weight->source_potential.scope,reference.scope},
                find(piece.weight->source_potential.expression),reference.body,tf.binder,reference_time->level);
            auto global=local?registry.globalTypedImage(*local,reference.images[0],reference.body):std::nullopt;
            if(!global)continue;
            auto range=registry.view(*global);
            if(!range || range->begin<carried->time.begin || range->end>carried->time.end)continue;
            if(p_definition)return {};p_definition=PotentialPiece{{reference.body,reference.scope},*global,i};
          }
          if(!p_definition)return {};
          const auto time=p_definition->time;auto time_view=registry.view(time);if(!time_view)return {};
          stage="carry_state_q_box";
          std::optional<std::size_t> q_index;
          for(std::size_t i=0;i<old_geometry->boxes.size();++i) {
            if(!heads.charge())return {};const auto& box=old_geometry->boxes[i];
            if(box[0].first<=time_view->begin && box[0].second>=time_view->end &&
                box[1].first<=component_state.begin && box[1].second>=component_state.end) {
              if(q_index)return {};q_index=i;
            }
          }
          if(!q_index)return {};const auto& qr=old_q->pieces[*q_index];
          if(!qr.reference_product || !qr.count)return {};
          stage="carry_state_relative_images";
          auto relative_q=registry.relativeImage(time,qr.images[0]);
          auto qv=relative_q?registry.view(*relative_q):std::nullopt;if(!qv)return {};
          // Keep the selected physical piece from the charged scan above;
          // do not perform an uncharged second walk over the family.
          const auto& pr=old_p->pieces[p_definition->index];
          auto pr_time=registry.view(pr.images[0]),qr_time=registry.view(qr.images[0]),qr_state=registry.view(qr.images[1]);
          if(!pr_time || !qr_time || !qr_state)return {};
          auto relative_p=registry.relativeImage(time,pr.images[0]);
          auto pv=relative_p?registry.view(*relative_p):std::nullopt;if(!pv)return {};
          auto state_here=registry.imageInOriginalScope(*image,tf.body_scope,dependency_witness);
          auto relative_state=state_here?registry.relativeImage(*state_here,qr.images[1]):std::nullopt;
          auto sv=relative_state?registry.view(*relative_state):std::nullopt;if(!sv)return {};
          stage="carry_state_contexts";
          Registry::Context pc{p_definition->use.scope,tf.body_scope};pc.axis_count=1;
          pc.axes[0]={pr_time->level,tf.binder,pv->begin,tf.count,pv->coordinate,false,{},*relative_p};
          // The product's admitted geometry owns the complete original
          // BODY scope. A first scalar factor may live in a shorter prefix;
          // equalReference/contextFor trims only proved unused axes per use.
          Registry::Context qc{old_geometry->pieces[*q_index].scope,tf.body_scope};qc.axis_count=2;
          qc.axes[0]={qr_time->level,tf.binder,qv->begin,tf.count,qv->coordinate,false,{},*relative_q};
          qc.axes[1]={qr_state->level,actual_state_axis,sv->begin,entry_count,sv->coordinate,false,{},*relative_state};
          constexpr auto context_fields=(sizeof(Registry::Context)+sizeof(Class)-1)/sizeof(Class);
          if(!heads.charge(3*context_fields+12+4*std::bit_width(identity_index.size()+1)))return {};
          const IdentityKey key{tf.head.expression,tf.head.scope,tf.body_scope};
          if(!identity_index.emplace(key,identity_pieces.size()).second)return {};
          identity_pieces.push_back({&piece,time,p_definition->use,pc,qc,*qr.reference_product});
        }
        // Recursive old-frame proofs finish BEFORE installing these callbacks;
        // their strictly decreasing time endpoints prohibit a certificate cycle.
        struct ClearStateCallbacks {
          cr::ProveStateTarget& target;cr::RelateProduct& product;cr::ProveMask& mask;
          ~ClearStateCallbacks(){target={};product={};mask={};}
        } clear_callbacks{region_state_target,region_product,region_mask};
        region_state_target=[&](const cr::StateTargetRequest& request) {
          const auto before=remaining;cr::StateTargetResult result;
          const auto finish=[&]() {result.work=before-remaining;return result;};
          std::size_t local=request.max_work;BodyHeadIndex::QueryBudget guard(heads,&local);
          const cr::Use source{carried->expression,canonical_scope},coord{state_coordinate,canonical_scope};
          const cr::Use narrow_coordinate{rhs_coordinate,carry_scope};
          if(!heads.charge() || request.terminal!=carried->state || request.terminal_state.frame!=carried->frame ||
              request.terminal_state.expression!=carried->expression || request.terminal_state.endpoint!=carried->time.end-1 ||
              request.source_state!=source || request.source_coordinate!=coord ||
              (request.region_root!=source && request.region_root!=cr::Use{actual->body,actual->scope}) ||
              request.frame.state_scope!=canonical_scope || request.frame.actual_state_axis!=state_axis ||
              request.frame.state_extent!=*state_extent || request.frame.time!=carried->time)return finish();
          if(restricted) {
            // Geometry and source-terminal attachment ONLY. Numeric region
            // equality is still proved by the complete causal callbacks.
            if(request.mode!=cr::StateTargetMode::PairedRestriction ||
                request.region_root!=cr::Use{actual->body,actual->scope} ||
                request.state_coordinate!=narrow_coordinate ||
                registry.count(carry_scope,actual_state_axis)!=std::optional{entry_count} ||
                !registry.context(*component_context))return finish();
            auto full_coordinate=registry.image(canonical_scope,state_axis,state_coordinate);
            auto local_coordinate=registry.image(carry_scope,actual_state_axis,rhs_coordinate);
            auto full=full_coordinate?registry.view(*full_coordinate):std::nullopt;
            auto narrow=local_coordinate?registry.view(*local_coordinate):std::nullopt;
            if(!full || !narrow || full->begin || full->end!=*state_extent ||
                narrow->begin || narrow->end!=entry_count ||
                !registry.prefix(*outer,canonical_scope) || !registry.prefix(*outer,carry_scope))return finish();
            result.source_target=source_target;
          } else if(request.mode!=cr::StateTargetMode::Identity || request.state_coordinate!=coord)return finish();
          result.status=fp::RelationStatus::Proved;result.target=state_target;return finish();
        };
        region_product=[&](const cr::ProductRequest& request) {
          const auto before=remaining;cr::ProductResult result;
          const auto finish=[&]() {result.work=before-remaining;return result;};
          std::size_t local=request.max_work;BodyHeadIndex::QueryBudget guard(heads,&local);
          if(!heads.charge() || !request.endpoint ||
              request.endpoint->frame!=carried->frame || request.endpoint->expression!=carried->expression ||
              request.endpoint->endpoint!=carried->time.end-1 || request.state_output!=state_target ||
              request.completed || request.output || request.projection || request.state_fold || !request.weight ||
              request.frame.potential_template!=carried->virtual_p || request.frame.injection_template!=carried->virtual_q ||
              request.frame.template_scope!=virtual_scope || request.frame.state_scope!=canonical_scope ||
              request.frame.entry_state!=old_frame->entry_state || request.frame.entry_potential!=old_frame->entry_potential ||
              request.frame.carry!=old_frame->carry || request.frame.time!=carried->time ||
              request.frame.state_extent!=*state_extent || request.factor_count>8)return finish();
          if(request.kind==cr::ProductKind::EntryIdentity) {
            if(request.time_fold || !identity_entry || request.factor_count!=1 ||
                request.factors[0]!=cr::Use{identity_entry->actual.body,identity_entry->actual.scope} ||
                request.weight->factor!=identity_entry->weight.factor ||
                request.weight->target_potential!=identity_entry->weight.target_potential ||
                request.weight->source_potential!=identity_entry->weight.source_potential ||
                !regions.equal(identity_entry->previous.source,identity_entry->previous.region).value_or(false) ||
                !samplePotential(carried->p,{find(request.weight->target_potential.expression),request.weight->target_potential.scope},carried->point))return finish();
            const Use expected{old_frame->entry_potential,canonical_scope};
            const Use actual_e{find(request.weight->source_potential.expression),request.weight->source_potential.scope};
            auto baseline=products.products.contextFor(expected,actual_e,assignment);
            if(!baseline || !products.products.related(expected,actual_e,*baseline))return finish();
            result.status=fp::RelationStatus::Proved;result.state=component_state;
            result.geometry_token=identity_pieces.size()+1;return finish();
          }
          if(request.kind!=cr::ProductKind::InjectionIdentity || !request.time_fold)return finish();
          const auto& tf=*request.time_fold;
          if(!heads.charge(4+4*std::bit_width(identity_index.size()+1)))return finish();
          auto found=identity_index.find({tf.head.expression,tf.head.scope,tf.body_scope});if(found==identity_index.end())return finish();
          const auto& record=identity_pieces[found->second];const auto& original=*record.syntax;
          if(tf.binder!=original.time_fold->binder || tf.count!=original.time_fold->count ||
              request.weight->factor!=original.weight->factor || request.weight->target_potential!=original.weight->target_potential ||
              request.weight->source_potential!=original.weight->source_potential ||
              !samplePotential(carried->p,{find(request.weight->target_potential.expression),request.weight->target_potential.scope},carried->point) ||
              !products.products.related(record.p,{find(request.weight->source_potential.expression),request.weight->source_potential.scope},record.p_context))return finish();
          std::array<Use,8> actual_factors{};
          if(!heads.charge(2*request.factor_count))return finish();
          for(unsigned i=0;i<request.factor_count;++i) {
            if(request.factors[i].expression>=nodes_.size())return finish();
            actual_factors[i]={find(request.factors[i].expression),request.factors[i].scope};
          }
          auto context=registry.admitContext(record.q_context);
          if(!context || !products.products.equal(record.q,std::span(actual_factors.data(),request.factor_count),*context))return finish();
          auto range=registry.view(record.time);if(!range)return finish();
          result.status=fp::RelationStatus::Proved;result.time={range->begin,range->end};result.state=component_state;
          result.geometry_token=found->second+1;return finish();
        };
        region_mask=[&](const cr::MaskRequest& request) {
          const auto before=remaining;fp::RelationResult result;
          const auto finish=[&]() {result.work=before-remaining;return result;};
          std::size_t local=request.max_work;BodyHeadIndex::QueryBudget guard(heads,&local);
          if(!heads.charge() || !request.endpoint || *request.endpoint!=carried->time.end-1 || request.state_output!=state_target ||
              !request.geometry_token || request.geometry_token>identity_pieces.size())return finish();
          const auto& record=identity_pieces[request.geometry_token-1];const auto& original=*record.syntax;
          if(!original.predicate || request.predicate!=*original.predicate || request.selected_when_true!=original.selected_when_true)return finish();
          auto k=head(request.predicate.expression);auto relation=k?comparison(*k):std::nullopt;
          if(!k || !relation || k->children.size()!=2 || !k->provenance.empty())return finish();
          const auto range=[&](Class cls)->std::optional<std::pair<std::uint64_t,std::uint64_t>> {
            const auto& n=nodes_[find(cls)];if(n.type!="u64" && n.type!="exact:u64")return {};
            auto a=heads.affine(cls);if(!a || !registry.boundedAffine(*a,request.predicate.scope))return {};
            auto end=a->constant;
            for(auto [axis,coefficient]:a->coefficients) {
              auto count=registry.count(request.predicate.scope,axis);std::uint64_t term;
              if(!count || !heads.charge() || __builtin_mul_overflow(coefficient,*count-1,&term) ||
                  __builtin_add_overflow(end,term,&end))return {};
            }
            return std::pair{a->constant,end};
          };
          auto a=range(k->children[0]),b=range(k->children[1]);if(!a || !b)return finish();
          std::optional<bool> predicate;
          if(*relation=="le") {if(a->second<=b->first)predicate=true;else if(a->first>b->second)predicate=false;}
          if(*relation=="lt") {if(a->second<b->first)predicate=true;else if(a->first>=b->second)predicate=false;}
          if(*relation=="ge") {if(a->first>=b->second)predicate=true;else if(a->second<b->first)predicate=false;}
          if(*relation=="gt") {if(a->first>b->second)predicate=true;else if(a->second<=b->first)predicate=false;}
          if(predicate && *predicate==request.selected_when_true)result.status=fp::RelationStatus::Proved;return finish();
        };
        stage="carry_identity_region";
        const cr::Use src{carried->expression,canonical_scope},coord{state_coordinate,canonical_scope};
        const cr::Use actual_root{actual->body,actual->scope},actual_coordinate{rhs_coordinate,actual->scope};
        std::optional<cr::Builder::StateRestrictionId> restriction;
        if(restricted) {
          restriction=regions.stateRestriction({actual_root,actual_coordinate,src,coord,carried->state});
          if(!restriction){stage="carry_restriction_attachment_unknown";return {};}
        }
        auto source=regions.identityState({src,coord,src,coord,carried->state,restriction});
        auto region=regions.identityRegion({actual_root,actual_coordinate,src,coord,carried->state,restriction});
        if(!source || !region || !regions.equal(*source,*region).value_or(false) || !heads.charge(18))return {};
        return StateFunctionProof{*source,*region,component_state};
      };
      // Private expression IDs do not enter the graph or overlap normal
      // producers' reserved three IDs per physically stored root pair.
      fp::ExprId next_terminal_virtual=UINT64_MAX;
      if(left.size()>(UINT64_MAX-3)/3)return;
      next_terminal_virtual-=3*left.size();
      const auto proveTerminalRoot=[&](Class left_root,Class right_root) {
        PhaseTrace phase{prefix_trace,remaining,phase_work,remaining};
        entry_stage="none";coordinate_trace={};
        stage="terminal_source_header";++roots_attempted;
        const auto& free=free_indices_[find(left_root)];
        if(free.size()!=1 || !heads.charge(4))return;
        const auto state_axis=*free.begin(),time_axis=options_.outer_binders;
        const auto source_coordinate=indexCoordinate(state_axis);
        auto extent=registry.count(*outer,state_axis);
        if(source_coordinate==none || !extent || state_axis>=time_axis)return;
        auto state_image=registry.image(*outer,state_axis,source_coordinate);
        auto state_view=state_image?registry.view(*state_image):std::nullopt;
        if(!state_view || state_view->derived || state_view->begin || state_view->end!=*extent)return;

        struct StoredUpdate {Class state;Update update;};
        std::vector<StoredUpdate> backwards;std::unordered_set<Class> active;
        Class initial=find(left_root);
        while(backwards.size()<1024) {
          if(!heads.charge(24) || !active.insert(initial).second)return;
          auto selected=update(initial);
          if(!selected)break;
          backwards.push_back({initial,*selected});initial=selected->previous;
        }
        // A first injection has no P operand in its source expression. Its
        // actual P is the retained previous-P operand of the first update,
        // never a manufactured ordinal, constant or load.
        if(backwards.empty() || backwards.size()==1024 || !heads.charge(backwards.size()))return;
        std::reverse(backwards.begin(),backwards.end());
        const auto anchor=backwards.front().update.previous_p;
        if(!prefix_node(initial,*outer) || !prefix_node(anchor,*outer))return;

        struct TemplatePiece {const cr::Candidate* syntax;Registry::ImageId time;JoinedProducts::Piece q;};
        std::vector<TemplatePiece> templates;
        std::vector<Use> syntax_roots{{find(right_root),*outer}};
        std::set<std::pair<Class,Registry::Id>> syntax_active;
        std::optional<Registry::ImageId> point_role;Class zero_seed=none;
        while(!syntax_roots.empty()) {
          auto root=syntax_roots.back();syntax_roots.pop_back();
          if(syntax_active.size()>=256 || !heads.charge(16+3*std::bit_width(syntax_active.size()+1)) ||
              !syntax_active.emplace(root.body,root.scope).second)return;
          const auto syntax=regions.candidates({root.body,root.scope},cr::CandidateMode::Identity);
          if(!syntax)return;
          for(const auto& piece:*syntax) {
            if(!heads.charge(12+2*piece.factor_count))return;
            if(!piece.time_fold) {
              // Untrusted parser hints only. The complete Identity proof
              // later checks the actual weight/E and prior private carry.
              std::optional<Use> old;unsigned weights=0;
              for(unsigned i=0;i<piece.factor_count;++i) {
                const auto use=piece.factors[i];auto k=head(use.expression);
                if(!k)return;
                if((piece.weight && use==piece.weight->factor) ||
                    (!piece.weight && k->type=="real" && idealOperation(*k) && k->name=="Exp" && k->children.size()==1))++weights;
                else {if(old)return;old=Use{find(use.expression),use.scope};}
              }
              if(piece.state_fold || piece.predicate || weights!=1 || !old || old->scope!=*outer)return;
              syntax_roots.push_back(*old);continue;
            }
            const auto& tf=*piece.time_fold;
            if(piece.state_fold || !piece.weight || tf.binder!=time_axis || templates.size()>=1023 ||
                piece.weight->source_potential.scope!=tf.body_scope)return;
            auto selected=head(tf.head.expression);
            if(!selected || selected->kind!=static_cast<int>(s::Kind::Fold) || !selected->fold ||
                selected->fold->reducer!=s::Reducer::Sum || selected->children.size()!=2 ||
                !selected->attributes.empty() || !selected->provenance.empty())return;
            // The SAME parser retains every actual outside-binder seed as
            // another contribution. A later seeded tile may carry an earlier
            // tile; it is not silently zeroed by this template hint.
            const auto seed=find(selected->children[0]);
            if(zero_seed==none && zero_[seed] && finite_[seed])zero_seed=seed;
            Registry::Context known{tf.body_scope,*outer};known.axis_count=1;
            known.axes[0]={state_axis,state_axis,0,*extent,source_coordinate,false};
            auto image=registry.typedTimeImage(known,find(piece.weight->source_potential.expression),anchor,time_axis);
            auto range=image?registry.view(*image):std::nullopt;
            auto state=registry.image(tf.body_scope,state_axis,source_coordinate);
            if(!range || !state)return;
            if(!range->begin && !point_role)point_role=*image;
            JoinedProducts::Piece q;q.images={*image,*state};
            for(unsigned i=0;i<piece.factor_count;++i)if(piece.factors[i]!=piece.weight->factor) {
              if(q.count==q.factors.size())return;
              q.factors[q.count++]={find(piece.factors[i].expression),piece.factors[i].scope};
            }
            constexpr auto template_fields=(sizeof(TemplatePiece)+sizeof(Class)-1)/sizeof(Class);
            // Vector growth is geometric; precharge its bounded fixed record
            // allocation/copies, not just the number of active factors.
            if(!q.count || !heads.charge(3*template_fields+2*q.count))return;
            templates.push_back({&piece,*image,q});
          }
        }
        if(!point_role || zero_seed==none || templates.empty())return;
        auto first_point=registry.typedPoint(*point_role,anchor,*outer);
        auto first_value=first_point?registry.globalValue(Registry::ScalarPoint{*first_point}):std::nullopt;
        if(!first_value || *first_value)return;
        constexpr auto case_fields=(sizeof(Case)+sizeof(Class)-1)/sizeof(Class);
        if(!heads.charge((backwards.size()+1)*(case_fields+8)))return;
        std::vector<Case> cases;std::vector<Update> updates;std::vector<std::size_t> begins{0};
        cases.reserve(backwards.size()+1);updates.reserve(backwards.size()+1);begins.reserve(backwards.size()+1);
        cases.push_back({0,Registry::ScalarPoint{*first_point},initial,none,zero_seed,{},{},{}});
        updates.push_back({zero_seed,anchor,zero_seed,initial});
        for(const auto& stored:backwards) {
          auto point=registry.typedPoint(*point_role,stored.update.current_p,*outer);
          auto value=point?registry.globalValue(Registry::ScalarPoint{*point}):std::nullopt;
          if(!value || cases.back().endpoint==UINT64_MAX || *value!=cases.back().endpoint+1 ||
              stored.update.previous!=cases.back().state)return;
          auto prior=registry.typedPoint(*point_role,stored.update.previous_p,*outer);
          auto prior_value=prior?registry.globalValue(Registry::ScalarPoint{*prior}):std::nullopt;
          if(!prior_value || *prior_value!=cases.back().endpoint)begins.push_back(cases.size());
          cases.push_back({*value,Registry::ScalarPoint{*point},stored.state,none,zero_seed,{},{},{}});
          updates.push_back(stored.update);++source_cases;
        }
        begins.push_back(cases.size());
        const Terminal* carried=nullptr;
        for(std::size_t frame_index=0;frame_index+1<begins.size();++frame_index) {
          const auto begin=begins[frame_index],end=begins[frame_index+1];
          if(begin>=end || cases[end-1].endpoint==UINT64_MAX)return;
          const fp::Interval time{cases[begin].endpoint,cases[end-1].endpoint+1};
          std::vector<JoinedFamilies::Piece> ps;std::vector<JoinedProducts::Piece> qs;
          constexpr auto family_fields=(sizeof(JoinedFamilies::Piece)+sizeof(JoinedProducts::Piece)+sizeof(Class)-1)/sizeof(Class);
          if(!heads.charge(templates.size()*2*family_fields))return;
          ps.reserve(templates.size());qs.reserve(templates.size());
          for(const auto& piece:templates) {
            if(!heads.charge())return;auto range=registry.view(piece.time);if(!range)return;
            if(range->begin<time.begin || range->end>time.end)continue;
            const auto& p=*piece.syntax->weight;
            ps.push_back({find(p.source_potential.expression),p.source_potential.scope,{piece.time,{}}});qs.push_back(piece.q);
          }
          JoinedFamilies::Box domain{{{time.begin,time.end},{0,*extent}}};
          auto p=potentials.join(ps,1,domain);auto q=products.join(qs,domain);
          if(!p || !q || next_terminal_virtual<2 || next_terminal_virtual-2<=nodes_.size() || !heads.charge(8))return;
          const auto virtual_p=next_terminal_virtual--,virtual_q=next_terminal_virtual--;
          const Class entry=begin?updates[begin].previous:zero_seed;
          const Class baseline=begin?updates[begin].previous_p:zero_seed;
          SourceFrameRequest request{std::span(cases).subspan(begin,end-begin),std::span(updates).subspan(begin,end-begin),
              *outer,state_axis,time_axis,state_axis,{},*extent,time,entry,baseline,carried,*p,*q,virtual_p,virtual_q};
          auto certified=certifySourceFrame(request,phase);if(!certified)return;
          const TerminalKey key{cases[end-1].state,*outer,state_axis,*extent,cases[end-1].endpoint};
          if(!heads.charge(12+5*std::bit_width(terminal_index.size()+1)))return;
          auto position=terminal_index.find(key);
          if(position==terminal_index.end() || !position->second)return;
          carried=&terminals[*position->second];
          if(carried->state!=certified->terminal || carried->frame!=certified->frame)return;
        }
        if(!carried || carried->expression!=find(left_root))return;
        Registry::Context state_assignment{*outer,*outer};state_assignment.axis_count=1;
        state_assignment.axes[0]={state_axis,state_axis,0,*extent,source_coordinate,false,{},*state_image};
        auto assignment=registry.admitContext(state_assignment);if(!assignment)return;
        phase.next(PrefixPhase::Region);stage="terminal_identity_region";
        auto identity=proveStateFunction(*carried,{find(right_root),*outer},source_coordinate,source_coordinate,
            state_axis,time_axis,*state_image,right_root,*assignment,false);
        if(!identity || identity->state!=fp::Interval{0,*extent} || !heads.charge(2))return;
        proved_roots.emplace_back(left_root,right_root);output_proved=true;stage="terminal_output_proved";
      };
      const auto proveRoot=[&](Class left_root,Class right_root,std::size_t ordinal) {
      if(ordinal>(UINT64_MAX-3)/3 || !heads.charge())return;
      PhaseTrace phase{prefix_trace,remaining,phase_work,remaining};
      entry_stage="none";
      coordinate_trace={};
      const fp::ExprId virtual_p=UINT64_MAX-3*ordinal;
      const fp::ExprId virtual_q=virtual_p-1,virtual_c=virtual_p-2;
      ++roots_attempted;
      std::vector<Case> cases;
      std::optional<Registry::Id> canonical_source;
      TypedDispatch dispatch_cursor(registry,potentials,*outer,left_root);
      Class coordinate=none;std::optional<Registry::ImageId> source_image;
      stage="source_dispatch";
      const auto appendCase=[&](std::uint64_t endpoint,const Registry::ScalarPoint& point,Class branch) {
          if(std::holds_alternative<Registry::DispatchPointId>(point) &&
              !heads.charge(2*((sizeof(Registry::DispatchPointId)+sizeof(Class)-1)/sizeof(Class))))return false;
          auto sum_id=heads.compact(branch);
          if(!sum_id)return false;const auto& sum=nodes_[*sum_id];
          if(sum.kind!=static_cast<int>(s::Kind::Fold) || !sum.fold || sum.fold->reducer!=s::Reducer::Sum ||
              sum.fold->binder_level!=options_.outer_binders || !sum.fold->count ||
              sum.children.size()!=2 || !sum.attributes.empty() || !sum.provenance.empty() ||
              !zero_[find(sum.children[0])])return false;
          auto original=registry.bodyView(*outer,*sum_id);if(!original)return false;
          if(!canonical_source) {
            if(!heads.charge(2+std::bit_width(canonical_sources.size()+1)))return false;
            if(auto found=canonical_sources.find(sum.fold->count);found!=canonical_sources.end())
              canonical_source=found->second;
            else {
              canonical_source=registry.insideFold(*outer,*sum_id);if(!canonical_source)return false;
              canonical_sources.emplace(sum.fold->count,*canonical_source);
            }
          }
          if(!canonical_source)return false;
          auto product=head(sum.children[1]);if(!product || !arithmetic(*product) || product->name!="Multiply")return false;
          std::optional<std::pair<Class,Class>> split;unsigned projection_side=0;
          for(unsigned side=0;side<2;++side) {
            auto c=head(product->children[side]);
            if(c && c->kind==static_cast<int>(s::Kind::InitialCell)) {
              if(split)return false;split=std::pair{find(product->children[1-side]),find(product->children[side])};
              projection_side=side;
            }
          }
          if(!split)return false;
          const std::array<unsigned,1> sp{1-projection_side},cp{projection_side};
          auto state_use=registry.bodyUse(*original,sp),projection_use=registry.bodyUse(*original,cp);
          if(!state_use || !projection_use ||
              registry.transport(*state_use,*canonical_source,sum.fold->binder_level)!=split->first ||
              registry.transport(*projection_use,*canonical_source,sum.fold->binder_level)!=split->second)return false;
          if(!heads.charge(18))return false;
          cases.push_back({endpoint,point,split->first,split->second,find(sum.children[0]),
              *original,*state_use,*projection_use});++source_cases;return true;
      };
      while(cases.size()<512) {
        auto next=dispatch_cursor.next();if(!next) {
          stage=potentials.intervalExhausted()?"source_dispatch_interval_cap":"source_dispatch_unknown";return;
        }
        coordinate=dispatch_cursor.coordinate;source_image=dispatch_cursor.image;
        if(next->inside && !appendCase(next->endpoint,next->point,next->branch))return;
        if(next->done)break;
        auto domain=source_image?registry.view(*source_image):std::nullopt;if(!domain)return;
        if(cases.size()==domain->end-domain->begin)break; // Exact coverage checked below before use.
      }
      if(cases.size()<2 || !source_image)return;
      if(!heads.charge(cases.size()*(1+std::bit_width(cases.size()+1))))return;
      std::sort(cases.begin(),cases.end(),[](const auto& a,const auto& b){return a.endpoint<b.endpoint;});
      auto source_domain=registry.view(*source_image);if(!source_domain)return;
      std::vector<std::pair<std::uint64_t,std::uint64_t>> intervals;
      for(const auto& c:cases) {if(c.endpoint==UINT64_MAX)return;intervals.emplace_back(c.endpoint,c.endpoint+1);}
      if(!potentials.cover(intervals,{source_domain->begin,source_domain->end}))return;
      if(!canonical_source)return;
      const auto canonical_scope=*canonical_source;
      const auto state_axis=options_.outer_binders;
      auto state_extent=registry.count(canonical_scope,state_axis);if(!state_extent)return;
      // Attach only ACTUAL reachable source BODY uses to this canonical state
      // scope. The shared full mapper certifies identical index assignment;
      // it does not traverse or infer a recurrence from shared P/Q headers.
      for(const auto& c:cases) {
        if(registry.count(c.body_scope,state_axis)!=state_extent)return;
        // Both ACTUAL original operand attachments were certified once during
        // Case construction by the SAME identity-assignment factory above.
        ++attached;
      }
      const Terminal* carried=nullptr;
      Class entry_state=cases[0].seed,entry_potential=cases[0].seed;
      std::vector<Update> updates;updates.reserve(cases.size());
      if(!source_domain->begin) {
        auto first=update(cases[1].state);
        if(!first || first->previous!=cases[0].state){stage="initial_update_unknown";return;}
        updates.push_back({entry_state,first->previous_p,entry_potential,cases[0].state});
      } else {
        auto first=update(cases[0].state,cases[0].seed);
        if(!first || !heads.charge(8+5*std::bit_width(terminal_index.size()+1)))return;
        auto found=terminal_index.find({first->previous,canonical_scope,state_axis,*state_extent,source_domain->begin-1});
        if(found==terminal_index.end() || !found->second){stage="carry_attachment_unknown";return;}
        carried=&terminals[*found->second];
        if(carried->time.end!=source_domain->begin || carried->expression!=first->previous ||
            carried->scope!=canonical_scope || carried->axis!=state_axis || carried->extent!=*state_extent)return;
        entry_state=first->previous;entry_potential=first->previous_p;updates.push_back(*first);
      }
      for(std::size_t i=1;i<cases.size();++i) {
        auto u=update(cases[i].state);
        if(!u || u->previous!=cases[i-1].state || u->previous_p!=updates.back().current_p)return;
        updates.push_back(*u);
      }
      stage="actual_region_candidates";
      phase.next(PrefixPhase::Templates);
      auto candidates=regions.candidates({find(right_root),*outer});if(!candidates)return;
      std::vector<JoinedProducts::Piece> q_pieces;std::vector<JoinedFamilies::Piece> p_pieces;
      // Original uses and the SAME full relation contexts are retained until
      // this root's complete dispatch/region attachment has been certified.
      std::deque<BoundedProducts::AssignedUse> assignment_attachments;
      struct ProductRecord {
        const cr::Candidate* syntax;
        Use projection;
        Registry::ImageId time,state;
      };
      using FoldKey=std::tuple<cr::ExprId,Registry::Id,cr::ExprId,Registry::Id>;
      std::map<FoldKey,std::size_t> product_records;
      std::vector<ProductRecord> records;
      std::vector<JoinedProducts::Piece> c_pieces;
      std::optional<std::pair<std::uint64_t,std::uint64_t>> projection_band;
      Class output_coordinate=none;
      const auto output_axis=source_domain->level;
      std::optional<Registry::CommonTranslationId> common_time;
      const auto globalTimeImage=[&](Registry::Id scope,std::uint32_t axis,Class coordinate)
          ->std::optional<Registry::ImageId> {
        auto local=registry.image(scope,axis,coordinate);
        return local && common_time?registry.commonImage(*common_time,*local):std::nullopt;
      };
      std::unordered_set<Class> potential_folds;
      const auto same_cell_header=[](const Key& a,const Key& b) {
        return a.kind==static_cast<int>(s::Kind::InitialCell) && b.kind==a.kind && a.name==b.name &&
            a.type==b.type && a.attributes==b.attributes && a.provenance==b.provenance && a.children.size()==1 && b.children.size()==1;
      };
      const auto* projection_head=head(cases.front().projection);if(!projection_head)return;
      stage="checked_joined_templates";
      for(const auto& candidate:*candidates) {
        // The SAME parser also retains the projected entry contribution.
        // It is not a P/Q injection-template piece: preserve it unchanged for
        // the subsequent whole-region certificate, which must validate the
        // actual entry/carry and baseline. This component admits no equality.
        if(!candidate.time_fold)continue;
        if(prefix_trace) {
          ++coordinate_trace.candidate;
          coordinate_trace.source_begin=source_domain->begin;
          coordinate_trace.source_end=source_domain->end;
          coordinate_trace.binder=candidate.time_fold->binder;
        }
        if(q_pieces.size()>=1023)return; // Existing joined-family admission cap.
        candidate_flags=bool(candidate.time_fold)+2*bool(candidate.state_fold)+
            4*bool(candidate.weight)+8*bool(candidate.predicate);
        candidate_factors=candidate.factor_count;
        stage="candidate_headers";
        if(!candidate.time_fold || !candidate.state_fold || !candidate.weight || !candidate.predicate ||
            candidate.time_fold->binder!=state_axis || candidate.state_fold->binder!=state_axis+1)return;
        if(nested_state_coordinate==none)return;
        const auto original_piece_scope=candidate.state_fold->body_scope;
        const auto scope_key=std::pair{candidate.time_fold->count,candidate.state_fold->count};
        if(!heads.charge(3+2*std::bit_width(canonical_rhs_scopes.size()+1)))return;
        auto position=canonical_rhs_scopes.find(scope_key);
        const auto piece_scope=position==canonical_rhs_scopes.end()?original_piece_scope:position->second;
        if(prefix_trace) {
          coordinate_trace.original_scope=original_piece_scope;
          coordinate_trace.piece_scope=piece_scope;
        }
        if(position==canonical_rhs_scopes.end())canonical_rhs_scopes.emplace(scope_key,piece_scope);
        Registry::Context original_assignment{piece_scope,original_piece_scope};original_assignment.axis_count=2;
        original_assignment.axes[0]={state_axis,state_axis,0,scope_key.first,state_coordinate,false};
        original_assignment.axes[1]={state_axis+1,state_axis+1,0,scope_key.second,nested_state_coordinate,false};
        auto assignment=registry.admitContext(original_assignment);if(!assignment)return;
        const auto assigned=[&](Use original,bool coordinate_use=false)->std::optional<Use> {
          auto proof=products.products.canonicalUse(original,*assignment,coordinate_use);
          auto value=proof?products.products.assigned(*proof):std::nullopt;
          if(!value || !heads.charge(10))return {};
          assignment_attachments.push_back(*proof);return value;
        };
        auto predicate=head(candidate.predicate->expression);if(!predicate || !comparison(*predicate))return;
        Class time_coordinate=none;
        stage="candidate_time_coordinate";
        if(prefix_trace) {
          coordinate_trace.predicate=find(candidate.predicate->expression);
          coordinate_trace.time=none;coordinate_trace.output=output_coordinate;
          coordinate_trace.children={none,none};coordinate_trace.constants={};
          coordinate_trace.coefficients={};coordinate_trace.axes={UINT32_MAX,UINT32_MAX};
          coordinate_trace.terms={};coordinate_trace.time_begin=coordinate_trace.time_end=0;
        }
        unsigned predicate_child=0;
        for(auto child:predicate->children) {
          auto a=heads.affine(child);
          if(prefix_trace && predicate_child<2) {
            coordinate_trace.children[predicate_child]=find(child);
            if(a) {
              coordinate_trace.constants[predicate_child]=a->constant;
              coordinate_trace.terms[predicate_child]=a->coefficients.size();
              if(!a->coefficients.empty()) {
                coordinate_trace.axes[predicate_child]=a->coefficients.begin()->first;
                coordinate_trace.coefficients[predicate_child]=a->coefficients.begin()->second;
              }
            } else coordinate_trace.terms[predicate_child]=SIZE_MAX;
          }
          ++predicate_child;
          if(a && a->coefficients.size()==1 && a->coefficients.begin()->first==candidate.time_fold->binder &&
              a->coefficients.begin()->second==1) {
            if(time_coordinate!=none){stage="candidate_time_duplicate";return;}time_coordinate=find(child);
            if(prefix_trace)coordinate_trace.time=time_coordinate;
          }
          if(a && a->coefficients.size()==1 && a->coefficients.begin()->first==output_axis &&
              a->coefficients.begin()->second==1) {
            if(output_coordinate!=none && output_coordinate!=find(child)){stage="candidate_output_conflict";return;}
            output_coordinate=find(child);
            if(prefix_trace)coordinate_trace.output=output_coordinate;
          }
        }
        if(time_coordinate==none || output_coordinate==none){stage="candidate_time_axes_missing";return;}
        if(!common_time) {
          // The source selector and RHS mask are actual unit coordinates of
          // the SAME outer assignment. Their origins may differ; this private
          // geometric token never asserts their raw graph values are equal.
          auto actual_output=registry.image(*outer,output_axis,output_coordinate);
          if(!actual_output)return;
          common_time=registry.commonTranslation(*source_image,*actual_output);if(!common_time)return;
        }
        stage="candidate_time_transport";
        auto actual_time=assigned({time_coordinate,original_piece_scope},true);if(!actual_time || actual_time->scope!=piece_scope)return;
        time_coordinate=actual_time->body;
        if(prefix_trace)coordinate_trace.time=time_coordinate;
        stage="candidate_time_image";
        auto time=globalTimeImage(piece_scope,candidate.time_fold->binder,time_coordinate);if(!time)return;
        auto time_view=registry.view(*time);if(!time_view)return;
        if(prefix_trace){coordinate_trace.time_begin=time_view->begin;coordinate_trace.time_end=time_view->end;}
        stage="candidate_time_anchor";
        auto anchor=std::lower_bound(cases.begin(),cases.end(),time_view->begin,
            [](const auto& c,std::uint64_t p){return c.endpoint<p;});
        if(anchor==cases.end() || anchor->endpoint!=time_view->begin)return;
        const auto index=anchor-cases.begin();
        Use injection{updates[index].injection,canonical_scope};
        stage="source_injection_factors";
        auto source_factors=products.products.factors(std::span(&injection,1));if(!source_factors)return;
        source_factors_count=source_factors->size();
        std::optional<Use> source_cell;
        for(auto use:*source_factors) {
          auto k=head(use.body);
          if(k && k->kind==static_cast<int>(s::Kind::InitialCell) && heads.depends(use.body,state_axis)) {
            if(source_cell)return;source_cell=use;
          }
        }
        if(!source_cell)return;
        const auto* source_cell_head=head(source_cell->body);if(!source_cell_head)return;
        JoinedProducts::Piece piece;std::optional<Use> piece_cell,piece_projection,original_projection;
        unsigned projection_count=0,weight_count=0;
        stage="candidate_factor_selection";
        for(unsigned j=0;j<candidate.factor_count;++j) {
          const auto use=candidate.factors[j];auto k=head(use.expression);if(!k)return;
          if(find(use.expression)==find(candidate.weight->factor.expression)) {++weight_count;continue;}
          auto value=assigned({find(use.expression),use.scope});if(!value)return;
          if(same_cell_header(*k,*projection_head)) {
            ++projection_count;piece_projection=*value;original_projection=Use{find(use.expression),use.scope};continue;
          }
          if(piece.count==piece.factors.size())return;
          piece.factors[piece.count++]=*value;
          if(same_cell_header(*k,*source_cell_head) && heads.depends(use.expression,candidate.state_fold->binder)) {
            if(piece_cell)return;piece_cell=*value;
          }
        }
        selected_projection_count=projection_count;selected_weight_count=weight_count;
        if(projection_count!=1 || weight_count!=1 || !piece_cell || !piece_projection ||
            piece_cell->scope!=piece_scope || piece_projection->scope!=piece_scope)return;
        auto point=registry.point(*time,anchor->point,canonical_scope);if(!point)return;
        Registry::Context known{piece_scope,canonical_scope};known.axis_count=1;
        stage="candidate_state_image";
        auto witness=registry.pointWitness(anchor->point);if(!witness)return;
        known.axes[0]={candidate.time_fold->binder,{},anchor->endpoint-time_view->begin,1,*witness,false,*point};
        auto state=registry.imageFromTypedCells(known,piece_cell->body,source_cell->body,
            candidate.state_fold->binder,state_axis);if(!state)return;
        piece.images={*time,*state};q_pieces.push_back(piece);++products_count;
        const auto& tf=*candidate.time_fold;const auto& sf=*candidate.state_fold;
        const FoldKey fold_key{tf.head.expression,tf.head.scope,sf.head.expression,sf.head.scope};
        if(!heads.charge(4+std::bit_width(product_records.size()+1)) ||
            !product_records.emplace(fold_key,records.size()).second)return;
        // Final root products still consume the ORIGINAL parser occurrence.
        // Its canonical value is only a proved definition for sampling.
        if(!original_projection)return;
        records.push_back({&candidate,*original_projection,*time,*state});
        // One actual time band suffices to define C(T,s), but EVERY later use
        // is still mapped to this definition before any region certificate.
        // Its output-time image is read in the original BODY scope; no scope
        // is widened and the inner U binder is never silently captured.
        const auto band=std::pair{time_view->begin,time_view->end};
        if(!projection_band)projection_band=band;
        if(*projection_band==band) {
          stage="projection_state_image";
          auto ct=globalTimeImage(piece_scope,output_axis,output_coordinate);if(!ct)return;
          auto cp=registry.point(*ct,cases.front().point,canonical_scope);if(!cp)return;
          Registry::Context ck{piece_scope,canonical_scope};ck.axis_count=1;
          auto witness=registry.pointWitness(cases.front().point);if(!witness)return;
          ck.axes[0]={output_axis,{},0,1,*witness,false,*cp};
          auto cs=registry.imageFromTypedCells(ck,piece_projection->body,cases.front().projection,
              sf.binder,state_axis);if(!cs)return;
          JoinedProducts::Piece c;c.count=1;c.factors[0]=*piece_projection;c.images={*ct,*cs};c_pieces.push_back(c);
        }
        if(potential_folds.insert(find(candidate.time_fold->head.expression)).second) {
          const auto use=candidate.weight->source_potential;
          auto value=assigned({find(use.expression),use.scope});if(!value)return;
          auto image=globalTimeImage(value->scope,candidate.time_fold->binder,time_coordinate);if(!image)return;
          p_pieces.push_back({value->body,value->scope,{*image,{}}});
        }
      }
      JoinedFamilies::Box domain{{{source_domain->begin,source_domain->end},{0,*state_extent}}};
      auto q=products.join(q_pieces,domain);
      auto p=potentials.join(p_pieces,1,domain);
      stage="joined_projection";
      auto c=products.join(c_pieces,domain);
      if(!q || !p || !c || output_coordinate==none || nested_state_coordinate==none)return;
      auto time_role=registry.view(*source_image);if(!time_role)return;
      SourceFrameRequest frame_source{cases,updates,canonical_scope,state_axis,state_axis,state_axis+1,
          time_role->level,*state_extent,{source_domain->begin,source_domain->end},
          entry_state,entry_potential,carried,*p,*q,virtual_p,virtual_q};
      auto certified=certifySourceFrame(frame_source,phase);if(!certified)return;
      const auto& frame=certified->definition;
      const auto f=std::optional{certified->frame};
      const auto completed=std::optional{certified->completed};
      const auto& certified_states=certified->states;
      const auto* c_family=products.view(*c);
      const auto* c_geometry=c_family?products.geometry.view(c_family->geometry):nullptr;
      if(!c_family || !c_geometry)return;
      // An owner-bound C descriptor, NEVER a graph expression at a widened
      // lexical scope. Only region_node resolves this private virtual use.
      cr::Node cn;cn.type=c_geometry->type;cn.finite_ideal=true;cn.exact_dependencies=true;
      std::vector<std::uint32_t> cfree{output_axis,state_axis+1};
      for(auto [level,identity]:c_geometry->captures){(void)identity;cfree.push_back(level);}
      if(!heads.charge(cfree.size()*(1+std::bit_width(cfree.size()+1))))return;
      std::sort(cfree.begin(),cfree.end());cfree.erase(std::unique(cfree.begin(),cfree.end()),cfree.end());
      if(cfree.size()>cn.free_indices.size())return;
      std::copy(cfree.begin(),cfree.end(),cn.free_indices.begin());cn.free_count=cfree.size();
      if(!heads.charge(cfree.size()+1) || joined_projection_nodes.contains(virtual_c))return;
      joined_projection_nodes.emplace(virtual_c,cn);
      const cr::ProjectionRequest projection_definition{{virtual_c,virtual_scope},output_axis,state_axis+1,*state_extent};
      stage="output_projection";auto projection=regions.projection(projection_definition);if(!projection)return;

      // Exact output-coordinate equality is independent of P being injective.
      // A constant potential never proves an output time assignment.
      auto output_image=globalTimeImage(*outer,output_axis,output_coordinate);if(!output_image || !common_time)return;
      auto output_domain=registry.view(*output_image);if(!output_domain ||
          output_domain->begin!=source_domain->begin || output_domain->end!=source_domain->end)return;
      const auto localOutput=[&](Registry::Id scope)->std::optional<Registry::ImageId> {
        auto actual=globalTimeImage(scope,output_axis,output_coordinate);
        return actual?registry.relativeImage(*actual,*output_image):std::nullopt;
      };
      auto output_assignment=localOutput(*outer);if(!output_assignment)return;
      Registry::Context output_context{*outer,*outer};output_context.axis_count=1;
      output_context.axes[0]={output_axis,output_axis,0,source_domain->end-source_domain->begin,
          output_coordinate,false,{},*output_assignment};
      if(!registry.commonOutputAssignment(*common_time,coordinate,output_coordinate,*outer))return;
      const cr::OutputTarget checked_output{1,1,*outer,{output_domain->begin,output_domain->end}};
      bool dispatch_attached=false;
      region_target=[&](const cr::OutputTargetRequest& request) {
        ++target_calls;
        const auto before=remaining;cr::OutputTargetResult result;
        const auto finish=[&]() {result.work=before-remaining;return result;};
        std::size_t local=request.max_work;BodyHeadIndex::QueryBudget guard(heads,&local);
        if(!heads.charge() || request.completed!=*completed || request.completed_prefix.frame!=*f ||
            request.source_dispatch!=cr::Use{find(left_root),*outer} ||
            request.source_coordinate!=cr::Use{coordinate,*outer} ||
            request.projection!=projection_definition ||
            (request.region_root!=request.source_dispatch && request.region_root!=cr::Use{find(right_root),*outer}) ||
            request.output_coordinate.scope!=*outer)return finish();
        // Source/source or actual/actual coordinate attachment stays a full
        // scoped proof. The DIFFERENT raw origins are related only by the
        // owned common assignment token above, never by an e-class equality.
        const auto requested_coordinate=request.region_root==request.source_dispatch?coordinate:output_coordinate;
        if(request.output_coordinate.expression>=nodes_.size() ||
            !registry.commonOutputAssignment(*common_time,coordinate,output_coordinate,*outer) ||
            !products.products.related({requested_coordinate,*outer},
              {find(request.output_coordinate.expression),*outer},output_context))return finish();
        if(!dispatch_attached) {
          // The ACTUAL selector path, all exact case predicates and complete
          // domain coverage were checked once above. Here attach EVERY branch
          // to its private StateId and the same complete C definition, retaining
          // its actual state Sum count/zero seed and original BODY occurrence.
          for(std::size_t i=0;i<cases.size();++i) {
            if(!heads.charge())return finish();auto state=core.view(certified_states[i]);
            if(!state || state->frame!=*f || state->expression!=cases[i].state || state->endpoint!=cases[i].endpoint ||
                !zero_[cases[i].seed] || registry.count(cases[i].body_scope,state_axis)!=state_extent)return finish();
            // The retained original projection_use attachment, including its
            // actual Fold/BODY path, authorizes this canonical value use.
            Use cp{cases[i].projection,canonical_scope};
            if(!products.sample(*c,std::span(&cp,1),cases[i].point,state_axis))return finish();
          }
          dispatch_attached=true;
        }
        result.status=fp::RelationStatus::Proved;result.target=checked_output;return finish();
      };

      std::map<std::pair<Class,Registry::Id>,bool> target_potential_proofs,projection_proofs;
      const auto checked_c=[&](const ProductRecord& record) {
        const auto key=std::pair{record.projection.body,record.projection.scope};
        if(!heads.charge(1+std::bit_width(projection_proofs.size()+1)))return false;
        if(auto it=projection_proofs.find(key);it!=projection_proofs.end())return it->second;
        auto state=registry.view(record.state);if(!state)return false;
        if(!heads.charge(1+std::bit_width(c_geometry->boxes.size()+1)))return false;
        auto it=std::lower_bound(c_geometry->boxes.begin(),c_geometry->boxes.end(),state->begin,
            [](const auto& box,std::uint64_t point){return box[1].second<=point;});
        if(it==c_geometry->boxes.end() || (*it)[1]!=std::pair{state->begin,state->end})return false;
        const auto& piece=c_family->pieces[it-c_geometry->boxes.begin()];
        if(piece.count!=1)return false;
        auto local=localOutput(record.projection.scope);if(!local)return false;
        Registry::Context context{piece.factors[0].scope,record.projection.scope};context.axis_count=2;
        context.axes[0]={output_axis,output_axis,0,output_domain->end-output_domain->begin,
            output_coordinate,false,{},*local};
        context.axes[1]={state->level,state->level,0,state->end-state->begin,nested_state_coordinate,false};
        const bool proved=products.products.related(piece.factors[0],record.projection,context);
        if(!heads.exhausted && remaining)projection_proofs.emplace(key,proved);return proved;
      };
      const auto checked_target_p=[&](cr::Use target) {
        if(target.expression>=nodes_.size())return false;
        const auto key=std::pair{find(target.expression),target.scope};
        if(!heads.charge(1+std::bit_width(target_potential_proofs.size()+1)))return false;
        if(!target_potential_proofs.contains(key)) {
          Use full{key.first,*outer},original{key.first,target.scope};
          auto local=localOutput(original.scope);if(!local)return false;
          Registry::Context lift{*outer,original.scope};lift.axis_count=1;
          lift.axes[0]={output_axis,output_axis,0,output_domain->end-output_domain->begin,
              output_coordinate,false,{},*local};
          const std::array<std::uint32_t,1> axis{output_axis};
          const std::array<Registry::ImageId,1> images{*output_image};
          const bool proved=products.products.related(full,original,lift) && potentials.uniform(*p,full.body,full.scope,axis,images);
          if(heads.exhausted || !remaining)return false;target_potential_proofs.emplace(key,proved);
        }
        return target_potential_proofs.at(key);
      };
      const auto checked_p=[&](const cr::Weight& weight,const ProductRecord& record) {
        if(!checked_target_p(weight.target_potential))return false;
        // Source P(U) is related to the exact admitted time-piece definition,
        // not merely selected by an allocation/header or equal interval length.
        const auto& syntax=*record.syntax;
        Registry::Context context{syntax.weight->source_potential.scope,weight.source_potential.scope};context.axis_count=1;
        context.axes[0]={syntax.time_fold->binder,syntax.time_fold->binder,0,syntax.time_fold->count,state_coordinate,false};
        return products.products.related({find(syntax.weight->source_potential.expression),context.reference_scope},
            {find(weight.source_potential.expression),context.actual_scope},context);
      };
      struct EntryRecord {
        const cr::Candidate* syntax;Use projection,carry,weight;
        cr::Builder::Certificate source,region;
        fp::Interval state;
      };
      std::vector<EntryRecord> entry_records;
      if(carried) {
        // Preprove the ACTUAL entry State function before entering the main
        // EntryProjection callback. No nested query allowance is replaced.
        for(const auto& entry:*candidates)if(!entry.time_fold) {
          phase.next(PrefixPhase::Carry);
          stage="carry_entry_header";
          if(!entry.state_fold || entry.predicate || entry.state_fold->binder!=state_axis ||
              !entry.state_fold->count || entry.state_fold->count>*state_extent)return;
          const auto entry_count=entry.state_fold->count;
          stage="carry_entry_factors";
          std::array<Use,8> raw{};for(unsigned i=0;i<entry.factor_count;++i)
            raw[i]={find(entry.factors[i].expression),entry.factors[i].scope};
          auto factors=products.products.factors(std::span(raw.data(),entry.factor_count));if(!factors)return;
          std::optional<Use> entry_c,entry_s,entry_weight;unsigned weights=0;
          for(auto use:*factors) {
            stage="carry_entry_factor_head";
            auto k=head(use.body);if(!k)return;
            stage="carry_entry_factor_roles";
            if(idealOperation(*k) && k->name=="Exp" && k->children.size()==1){++weights;entry_weight=use;continue;}
            if(same_cell_header(*k,*projection_head)) {if(entry_c)return;entry_c=use;}
            else {if(entry_s)return;entry_s=use;}
          }
          stage="carry_entry_factor_scope";
          if(weights!=1 || !entry_c || !entry_s || entry_s->scope!=entry.state_fold->body_scope ||
              entry_c->scope!=entry.state_fold->body_scope)return;
          const auto original_entry_c=*entry_c,original_entry_s=*entry_s;
          stage="carry_entry_transport_context";
          // Same-count original BODY transport only. A narrow16-state use
          // never acquires the canonical source's128-state lexical domain.
          if(!heads.charge(3+2*std::bit_width(canonical_carry_sources.size()+1)))return;
          auto carry_position=canonical_carry_sources.find(entry_count);
          const auto carry_scope=entry_count==*state_extent?canonical_scope:
              carry_position==canonical_carry_sources.end()?entry_s->scope:carry_position->second;
          Registry::Context transport{carry_scope,entry_s->scope};transport.axis_count=1;
          transport.axes[0]={state_axis,state_axis,0,entry_count,state_coordinate,false};
          auto transport_id=registry.admitContext(transport);if(!transport_id)return;
          stage="carry_entry_state_transport";
          auto attachment=products.products.canonicalUse(*entry_s,*transport_id);
          auto actual=attachment?products.products.assigned(*attachment):std::nullopt;
          if(!actual || actual->scope!=carry_scope || !heads.charge(10))return;
          assignment_attachments.push_back(*attachment);
          stage="carry_entry_projection_transport";
          auto c_attachment=products.products.canonicalUse(*entry_c,*transport_id);
          auto actual_c=c_attachment?products.products.assigned(*c_attachment):std::nullopt;
          if(!actual_c || actual_c->scope!=carry_scope || !heads.charge(10))return;
          assignment_attachments.push_back(*c_attachment);
          if(entry_count!=*state_extent && carry_position==canonical_carry_sources.end())
            canonical_carry_sources.emplace(entry_count,carry_scope);
          entry_c=actual_c;
          stage="carry_entry_time_image";
          auto ct=globalTimeImage(entry_c->scope,output_axis,output_coordinate);
          auto cp=ct?registry.point(*ct,cases.front().point,canonical_scope):std::nullopt;if(!cp)return;
          Registry::Context known{entry_c->scope,canonical_scope};known.axis_count=1;
          auto witness=registry.pointWitness(cases.front().point);if(!witness)return;
          known.axes[0]={output_axis,{},0,1,*witness,false,*cp};
          stage="carry_entry_state_image";
          auto image=registry.imageFromTypedCells(known,entry_c->body,cases.front().projection,state_axis,state_axis);
          auto view=image?registry.view(*image):std::nullopt;
          if(!view || view->scope!=carry_scope || view->level!=state_axis ||
              view->begin>=view->end || view->end>*state_extent || view->end-view->begin!=entry_count)return;
          const bool restricted=entry_count!=*state_extent || view->begin!=0 || view->end!=*state_extent;
          if(!restricted) {
            stage="carry_entry_full_projection";
            // Preserve the existing full-state route, including families
            // whose complete C definition comprises several state boxes.
            const std::array<std::uint32_t,2> c_axes{output_axis,state_axis};
            const std::array<Registry::ImageId,2> c_images{*ct,*image};
            if(!products.geometry.uniform(c_family->geometry,entry_c->body,entry_c->scope,c_axes,c_images))return;
          } else {
          stage="carry_entry_projection_box";
          // One uniquely containing ACTUAL admitted C box, then the SAME full
          // typed scoped relation. Geometry is not body equality and does not
          // ask a16-state representative to be uniform over all128 states.
          std::optional<std::size_t> c_index;
          for(std::size_t i=0;i<c_geometry->boxes.size();++i) {
            if(!heads.charge())return;const auto& box=c_geometry->boxes[i];
            if(box[0]==std::pair{output_domain->begin,output_domain->end} &&
                box[1].first<=view->begin && box[1].second>=view->end) {
              if(c_index)return;c_index=i;
            }
          }
          if(!c_index)return;const auto& expected_c=c_family->pieces[*c_index];
          if(expected_c.count!=1)return;
          stage="carry_entry_projection_image";
          auto c_time=registry.relativeImage(*ct,expected_c.images[0]);
          auto c_state=registry.relativeImage(*image,expected_c.images[1]);
          auto ctv=c_time?registry.view(*c_time):std::nullopt,csv=c_state?registry.view(*c_state):std::nullopt;
          if(!ctv || !csv)return;
          Registry::Context c_relation{expected_c.factors[0].scope,entry_c->scope};c_relation.axis_count=2;
          c_relation.axes[0]={output_axis,output_axis,ctv->begin,output_domain->end-output_domain->begin,
              ctv->coordinate,false,{},*c_time};
          auto expected_state=registry.view(expected_c.images[1]);if(!expected_state)return;
          c_relation.axes[1]={expected_state->level,state_axis,csv->begin,entry_count,
              csv->coordinate,false,{},*c_state};
          stage="carry_entry_projection_relation";
          if(!products.products.related(expected_c.factors[0],*entry_c,c_relation))return;
          }
          stage="carry_entry_component_context";
          Registry::Context component_assignment{canonical_scope,carry_scope};component_assignment.axis_count=1;
          component_assignment.axes[0]={state_axis,state_axis,view->begin,entry_count,view->coordinate,false,{},*image};
          auto component_context=registry.admitContext(component_assignment);if(!component_context)return;
          stage="carry_entry_state_function";
          auto identity=proveStateFunction(*carried,*actual,state_coordinate,state_coordinate,
              state_axis,state_axis+1,*image,entry_c->body,*component_context,restricted);
          if(!identity)return;
          entry_records.push_back({&entry,original_entry_c,original_entry_s,*entry_weight,
              identity->source,identity->region,identity->state});
        }
        stage="carry_entry_records";
        if(entry_records.empty())return;
      }
      region_product=[&](const cr::ProductRequest& request) {
        ++product_calls;last_product_kind=1+static_cast<unsigned>(request.kind);
        if(request.kind==cr::ProductKind::EntryProjection)entry_stage="header";
        const auto before=remaining;cr::ProductResult result;
        const auto finish=[&]() {result.work=before-remaining;return result;};
        std::size_t local=request.max_work;BodyHeadIndex::QueryBudget guard(heads,&local);
        if(!heads.charge() ||
            !request.completed || *request.completed!=*completed || !request.output || *request.output!=checked_output ||
            !request.projection || *request.projection!=projection_definition ||
            !request.state_fold || !request.weight ||
            request.frame.potential_template!=virtual_p || request.frame.injection_template!=virtual_q ||
            request.frame.template_scope!=virtual_scope || request.frame.state_scope!=canonical_scope ||
            request.frame.entry_state!=entry_state || request.frame.entry_potential!=entry_potential ||
            request.frame.carry!=frame.carry || request.frame.time!=fp::Interval{source_domain->begin,source_domain->end} ||
            request.frame.state_extent!=*state_extent || request.factor_count>8)return finish();
        if(request.kind==cr::ProductKind::EntryProjection) {
          entry_stage="entry_header";
          if(!carried || request.time_fold || request.endpoint || request.state_output ||
              request.weight->source_potential.expression>=nodes_.size())return finish();
          const auto& sf=*request.state_fold;const EntryRecord* found=nullptr;std::size_t index=0;
          entry_stage="record";
          for(std::size_t i=0;i<entry_records.size();++i) {
            if(!heads.charge())return finish();const auto& record=entry_records[i];
            const auto& original=*record.syntax->state_fold;
            if(sf.head==original.head && sf.body_scope==original.body_scope && sf.binder==original.binder && sf.count==original.count) {
              if(found)return finish();found=&record;index=i;
            }
          }
          if(!found)return finish();entry_stage="weight_occurrence";
          if(request.weight->factor!=cr::Use{found->weight.body,found->weight.scope})return finish();
          entry_stage="identity_certificate";
          if(!regions.equal(found->source,found->region).value_or(false))return finish();
          entry_stage="target_potential";
          if(!checked_target_p(request.weight->target_potential))return finish();
          // The private Identity certificate relates the original carry
          // function to the actual prior terminal. The baseline remains the
          // actual new frame E, independently of that prior terminal P.
          entry_stage="baseline_image";
          auto local=localOutput(request.weight->source_potential.scope);if(!local)return finish();
          Registry::Context baseline{canonical_scope,request.weight->source_potential.scope};baseline.axis_count=2;
          baseline.axes[0]={output_axis,output_axis,0,output_domain->end-output_domain->begin,output_coordinate,false,{},*local};
          baseline.axes[1]={state_axis,state_axis,found->state.begin,sf.count,state_coordinate,false};
          entry_stage="baseline_relation";
          const Use expected_e{entry_potential,canonical_scope};
          const Use actual_e{find(request.weight->source_potential.expression),request.weight->source_potential.scope};
          // E may be outside the state Fold. Derive its original-use context
          // through the SAME conservative dependency-absence factory; never
          // move that scalar into a wider scope or omit a used/captured axis.
          auto scalar_context=products.products.contextFor(expected_e,actual_e,baseline);
          if(!scalar_context || !products.products.related(expected_e,actual_e,*scalar_context))return finish();
          const std::array<Use,2> expected{found->projection,found->carry};std::array<Use,8> actual{};
          entry_stage="factor_occurrences";
          if(request.factor_count!=expected.size() || !heads.charge(2*request.factor_count))return finish();
          for(unsigned i=0;i<request.factor_count;++i) {
            if(request.factors[i].expression>=nodes_.size())return finish();
            actual[i]={find(request.factors[i].expression),request.factors[i].scope};
          }
          entry_stage="factor_image";
          auto identity_output=localOutput(sf.body_scope);if(!identity_output)return finish();
          Registry::Context identity{sf.body_scope,sf.body_scope};identity.axis_count=2;
          identity.axes[0]={output_axis,output_axis,0,output_domain->end-output_domain->begin,output_coordinate,false,{},*identity_output};
          identity.axes[1]={state_axis,state_axis,0,sf.count,state_coordinate,false};
          entry_stage="factor_relation";
          if(!products.products.equal(expected,std::span(actual.data(),request.factor_count),identity))return finish();
          result.status=fp::RelationStatus::Proved;result.state=found->state;
          entry_stage="complete";
          result.geometry_token=records.size()+index+1;return finish();
        }
        if(request.kind!=cr::ProductKind::InjectionProjection || !request.time_fold)return finish();
        const auto& tf=*request.time_fold;const auto& sf=*request.state_fold;
        if(!heads.charge(4+std::bit_width(product_records.size()+1)))return finish();
        auto found=product_records.find({tf.head.expression,tf.head.scope,sf.head.expression,sf.head.scope});
        if(found==product_records.end())return finish();
        const auto& record=records[found->second];const auto& syntax=*record.syntax;
        if(tf.binder!=syntax.time_fold->binder || tf.count!=syntax.time_fold->count || tf.body_scope!=syntax.time_fold->body_scope ||
            sf.binder!=syntax.state_fold->binder || sf.count!=syntax.state_fold->count || sf.body_scope!=syntax.state_fold->body_scope ||
            request.weight->factor!=syntax.weight->factor || request.weight->target_potential!=syntax.weight->target_potential ||
            request.weight->source_potential!=syntax.weight->source_potential ||
            !checked_c(record) || !checked_p(*request.weight,record))return finish();
        std::array<Use,8> expected{},actual{};unsigned count=0;
        for(unsigned i=0;i<syntax.factor_count;++i)if(syntax.factors[i]!=syntax.weight->factor) {
          if(count>=8)return finish();expected[count++]={find(syntax.factors[i].expression),syntax.factors[i].scope};
        }
        if(count!=request.factor_count || !count || !heads.charge(count))return finish();
        for(unsigned i=0;i<count;++i) {
          if(request.factors[i].expression>=nodes_.size())return finish();
          actual[i]={find(request.factors[i].expression),request.factors[i].scope};
        }
        Registry::Context context{sf.body_scope,sf.body_scope};context.axis_count=2;
        context.axes[0]={tf.binder,tf.binder,0,tf.count,state_coordinate,false};
        context.axes[1]={sf.binder,sf.binder,0,sf.count,nested_state_coordinate,false};
        if(!products.products.equal(std::span(expected.data(),count),std::span(actual.data(),count),context))return finish();
        auto time=registry.view(record.time),state=registry.view(record.state);if(!time || !state)return finish();
        result.status=fp::RelationStatus::Proved;result.time={time->begin,time->end};result.state={state->begin,state->end};
        result.geometry_token=found->second+1;return finish();
      };
      region_mask=[&](const cr::MaskRequest& request) {
        ++mask_calls;
        const auto before=remaining;fp::RelationResult result;
        const auto finish=[&]() {result.work=before-remaining;return result;};
        std::size_t local=request.max_work;BodyHeadIndex::QueryBudget guard(heads,&local);
        if(!heads.charge() || !request.output || *request.output!=checked_output || request.endpoint ||
            !request.geometry_token || request.geometry_token>records.size())return finish();
        const auto& record=records[request.geometry_token-1];const auto& syntax=*record.syntax;
        if(!syntax.predicate || request.predicate!=*syntax.predicate || request.selected_when_true!=syntax.selected_when_true)return finish();
        auto k=head(request.predicate.expression);if(!k || k->children.size()!=2 ||
            !k->provenance.empty())return finish();
        auto relation=comparison(*k);if(!relation)return finish();
        auto time=registry.view(record.time);if(!time)return finish();
        // The actual predicate compares LOCAL unit witnesses. Its two
        // semantic images were translated through this SAME proved assignment
        // (with checked no-wrap bounds), so <=/>= order is preserved on BOTH
        // sides. A one-sided shift or another frame's token cannot certify it.
        if(!registry.commonImagePair(*common_time,record.time,*output_image))return finish();
        const auto side=[&](Class child,Class wanted) {
          if(child>=nodes_.size() || wanted>=nodes_.size())return false;
          const auto raw=[](const std::string& t){return t.starts_with("exact:")?t.substr(6):t;};
          if(raw(nodes_[find(child)].type)!="u64" || raw(nodes_[find(wanted)].type)!="u64")return false;
          auto a=heads.affine(child),b=heads.affine(wanted);
          return a && b && registry.boundedAffine(*a,request.predicate.scope) &&
              registry.boundedAffine(*b,request.predicate.scope) && a->constant==b->constant && a->coefficients==b->coefficients;
        };
        const bool ut=side(k->children[0],time->coordinate) && side(k->children[1],output_coordinate);
        const bool tu=side(k->children[1],time->coordinate) && side(k->children[0],output_coordinate);
        const bool proved=request.selected_when_true?((*relation=="le" && ut)||(*relation=="ge" && tu)):
            ((*relation=="gt" && ut)||(*relation=="lt" && tu));
        if(proved)result.status=fp::RelationStatus::Proved;return finish();
      };
      phase.next(PrefixPhase::Region);
      stage="whole_output_region";
      const cr::Builder::ParameterizedRequest target{{find(right_root),*outer},{output_coordinate,*outer},
          {find(left_root),*outer},{coordinate,*outer},*completed,*projection};
      auto region=regions.parameterizedRegion(target);
      region_complete=bool(region);region_cap=regions.exhausted();
      const cr::Builder::ParameterizedRequest source{{find(left_root),*outer},{coordinate,*outer},
          {find(left_root),*outer},{coordinate,*outer},*completed,*projection};
      phase.next(PrefixPhase::Dispatch);
      auto dispatch=regions.parameterizedDispatch(source);
      dispatch_complete=bool(dispatch);region_cap=region_cap || regions.exhausted();
      const bool proved=region && dispatch && regions.equal(*region,*dispatch).value_or(false);
      if(proved) {
        if(!heads.charge(3))return;
        proved_roots.emplace_back(left_root,right_root);output_proved=true;
      }
      stage=proved?"whole_output_proved":"whole_output_unknown";
      scope_count=registry.scopes.size();coverage_work=potentials.coverageWork()+products.geometry.coverageWork();
      };
      for(std::size_t i=0;i<left.size();++i) {
        if(!remaining || heads.exhausted || regions.exhausted() || core.exhausted())break;
        if(!heads.charge())break;
        const auto& l=nodes_[find(left[i])];
        const bool dispatch=idealSelect(l);
        const bool terminal=l.type=="real" && arithmetic(l) && l.name=="Add";
        if(!dispatch && !terminal)continue;
        const auto root_attempts=roots_attempted,root_work=remaining,root_cases=source_cases,root_attached=attached,
            root_updates=proved_updates,root_frames=completed_frames,root_proved=proved_roots.size();
        if(dispatch)proveRoot(find(left[i]),find(right[i]),i);
        else proveTerminalRoot(find(left[i]),find(right[i]));
        // Report only a failed existing carry stage. No new witness selection,
        // traversal, rendering, semantic fact or proof-work debit is involved.
        if(prefix_trace && roots_attempted>root_attempts && proved_roots.size()==root_proved &&
            carry_denial_trace<16) {
          ++carry_denial_trace;
          SENTRYWARD_DEBUG_PRINT(egraph_89);
        }
        if(prefix_trace && roots_attempted>root_attempts && root_trace_size<root_trace.size())
          root_trace[root_trace_size++]={i,root_work-remaining,
            source_cases-root_cases,attached-root_attached,proved_updates-root_updates,
            completed_frames-root_frames,proved_roots.size()-root_proved,stage,entry_stage,coordinate_trace};
        // A callback must never outlive the root-specific original-path
        // records it borrows. Earlier private certificates remain valid; no
        // later query can accidentally call a previous root's attachment.
        core_relate={};region_product={};region_mask={};region_target={};region_state_target={};
      }
    };
    run();
    if(prefix_trace)SENTRYWARD_DEBUG_PRINT(egraph_90);
    if(prefix_trace) {
      constexpr std::array names{"source","templates","sampling","output","carry","region","dispatch"};
      SENTRYWARD_DEBUG_PRINT(egraph_91);
      SENTRYWARD_DEBUG_PRINT(egraph_92);
      std::size_t accounted=0;SENTRYWARD_DEBUG_PRINT(egraph_93);
      for(unsigned i=0;i<phase_work.size();++i){accounted+=phase_work[i];SENTRYWARD_DEBUG_PRINT(egraph_94);}
      SENTRYWARD_DEBUG_PRINT(egraph_95);
      for(std::size_t i=0;i<root_trace_size;++i) {
        const auto& r=root_trace[i];SENTRYWARD_DEBUG_PRINT(egraph_96);
        if(r.coordinate.candidate) {
          const auto& c=r.coordinate;
          SENTRYWARD_DEBUG_PRINT(egraph_97);
          for(unsigned j=0;j<2;++j)SENTRYWARD_DEBUG_PRINT(egraph_98);
          SENTRYWARD_DEBUG_PRINT(egraph_99);
        }
      }
    }
    // All borrowed scopes/heads/core/region certificates have now died. Only
    // their completed whole-root equality is transferred to the live Engine.
    // No private template/component handle creates an internal union.
    for(auto [a,b]:proved_roots)rewrite(a,b);
    return output_proved;
  }
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
      const ScopedSumProof::Environment* enclosing=nullptr) {
    // ONE existing paired proof, shared by reduction coverage and Numeric SAT.
    // The optional source binder supports a sample Fold at another lexical depth.
    if(block_binder==UINT32_MAX)block_binder=binder;
    constexpr std::size_t max_pairs=4096,max_environments=4096,max_depth=8;
    const bool sum_context=sums && assignment;
    constexpr std::size_t mapper_limit=131072;
    std::size_t mapper_remaining=mapper_limit;
    // Aggregate borrowing does not enlarge or refresh the physical epoch.
    // Nested coverage and body calls reuse the original pointer. Legacy
    // callers retain their old131k allowance unchanged.
    std::size_t coverage_remaining=heads.remaining?*heads.remaining:mapper_limit;
    const bool scoped_coverage=sum_context || heads.coverage_active;
    BodyHeadIndex::QueryBudget budget(heads,&mapper_remaining,&coverage_remaining,sum_context);
    if(bool(sums)!=bool(assignment) || (sum_context && (registry || initial || admitted ||
        &sums->heads!=&heads || !sums->valid(*assignment))))return {};
    std::optional<ScopedSumProof::QueryRelay> query_relay;
    if(sum_context)query_relay.emplace(sums->planner_query,&heads.query_guards);
    auto* geometry_registry=sum_context?&sums->registry:registry;
    if(bool(registry)!=bool(initial) || (admitted && !registry) ||
        (registry && &registry->heads!=&heads))return {};
    if(admitted) {
      // No caller bool: require this EXACT immutable record, owned by the
      // current registry/epoch. All actual body/leaf/scope checks below remain.
      if(registry->context(*admitted)!=initial)return {};
    } else if(registry && !registry->validate(*initial))return {};
    const bool explicit_context=initial!=nullptr;
    using Environment=std::vector<std::pair<std::uint32_t,std::uint32_t>>;
    std::vector<Environment> environments(1);
    if(enclosing){if(!sum_context || enclosing->size()>max_depth)return {};environments[0]=*enclosing;}
    std::map<Environment,unsigned> environment_keys{{environments[0],0}};
    ScopedSumProof::Geometry geometry;
    if(sum_context)geometry.assignment=*assignment;
    // This selected product view only supplies factor tasks. All tasks retain
    // the SAME unfinished assignment, original scopes, gray/cycle sets and
    // final affine-equation solve; no independent scoped relation is minted.
    std::optional<BoundedProducts> scoped_products;
    struct Task {
      Class a,b;unsigned environment=0;bool substitute=true,ready=false;
      unsigned fixed_mask=0;
      BodyScopeRegistry::Id reference_scope=0,actual_scope=0;
    };
    std::vector<Task> work{{target,block}};
    if(explicit_context) {
      work.front().substitute=false;work.front().fixed_mask=(1u<<initial->axis_count)-1;
      work.front().reference_scope=initial->reference_scope;work.front().actual_scope=initial->actual_scope;
    }
    if(sum_context) {
      if(reference_scope>=geometry_registry->scopes.size() || actual_scope>=geometry_registry->scopes.size())return {};
      work.front().substitute=false;work.front().reference_scope=reference_scope;work.front().actual_scope=actual_scope;
    }
    std::map<std::tuple<Class,Class,unsigned,bool,unsigned,
        BodyScopeRegistry::Id,BodyScopeRegistry::Id>,unsigned char> visited;
    struct MapperTrace {
      ScopedSumProof* sums;const decltype(visited)& pairs;std::size_t tasks=0;
      ~MapperTrace(){if(sums){sums->mapper_visits+=pairs.size();sums->mapper_tasks+=tasks;}}
    } mapper_report{sum_context && sums->trace_work?sums:nullptr,visited};
    // Validate only the heads actually chosen by this paired proof. No eager
    // ancestor walk: shared independent eclasses and exact affine summaries
    // are already certified opaque boundaries. A selected side may not reenter
    // an active class under a different counterpart or lexical environment.
    std::unordered_set<Class> active_left,active_right;
    std::optional<std::pair<std::uint64_t,std::uint64_t>> map;
    const auto deny=[&](const char* reason,const Task& task,
        std::optional<std::pair<Class,Class>> chosen=std::nullopt,
        std::optional<std::pair<std::uint64_t,std::uint64_t>> observed=std::nullopt)
        ->std::optional<std::pair<std::uint64_t,std::uint64_t>> {
      if(heads.denial_trace_remaining && *heads.denial_trace_remaining) {
        --*heads.denial_trace_remaining;
        SENTRYWARD_DEBUG_PRINT(egraph_102);
        if(chosen)SENTRYWARD_DEBUG_PRINT(egraph_103);else SENTRYWARD_DEBUG_PRINT(egraph_104);
        SENTRYWARD_DEBUG_PRINT(egraph_105);
        // This walk already admits at most8 lexical bindings; no ancestor or
        // expression rendering is used, and orientation probes do not spend
        // this separate first16-denials-per-reducer/epoch ledger.
        for(auto [a,b]:environments[task.environment])SENTRYWARD_DEBUG_PRINT(egraph_106);
        SENTRYWARD_DEBUG_PRINT(egraph_107);
        if(map)SENTRYWARD_DEBUG_PRINT(egraph_108);else SENTRYWARD_DEBUG_PRINT(egraph_109);
        if(observed)SENTRYWARD_DEBUG_PRINT(egraph_110);
        if(sum_context) {
          SENTRYWARD_DEBUG_PRINT(egraph_111);
          for(const auto& axis:geometry.assignment.axes) {
            SENTRYWARD_DEBUG_PRINT(egraph_112);if(axis.actual)SENTRYWARD_DEBUG_PRINT(egraph_113);else SENTRYWARD_DEBUG_PRINT(egraph_114);
            SENTRYWARD_DEBUG_PRINT(egraph_115);if(axis.stride)SENTRYWARD_DEBUG_PRINT(egraph_116);else SENTRYWARD_DEBUG_PRINT(egraph_117);
            SENTRYWARD_DEBUG_PRINT(egraph_118);if(axis.offset)SENTRYWARD_DEBUG_PRINT(egraph_119);else SENTRYWARD_DEBUG_PRINT(egraph_120);
            SENTRYWARD_DEBUG_PRINT(egraph_121);
          }
        }
        const auto active_remaining=scoped_coverage?mapper_remaining:
            (heads.query_remaining?*heads.query_remaining:mapper_remaining);
        SENTRYWARD_DEBUG_PRINT(egraph_122);
        // Read only these immutable stored headers. Never select a witness,
        // inspect class members, render ancestors, or debit proof work merely
        // to explain one of the existing first16 TRACE denials.
        const auto text=[&](std::string_view value) {
          SENTRYWARD_DEBUG_PRINT(egraph_123);
          const auto size=std::min<std::size_t>(value.size(),128);
          constexpr char hex[]="0123456789abcdef";
          for(std::size_t i=0;i<size;++i) {
            const auto c=static_cast<unsigned char>(value[i]);
            if(c=='"' || c=='\\')SENTRYWARD_DEBUG_PRINT(egraph_124);
            else if(c<32 || c==127)SENTRYWARD_DEBUG_PRINT(egraph_125);
            else SENTRYWARD_DEBUG_PRINT(egraph_126);
          }
          SENTRYWARD_DEBUG_PRINT(egraph_127);
          if(size<value.size())SENTRYWARD_DEBUG_PRINT(egraph_128);
        };
        const auto plain=[&](const char* label,Class id) {
          SENTRYWARD_DEBUG_PRINT(egraph_129);
          if(id>=snapshot.size()){SENTRYWARD_DEBUG_PRINT(egraph_130);return;}
          const auto& k=snapshot[id];
          SENTRYWARD_DEBUG_PRINT(egraph_131);text(k.name);
          SENTRYWARD_DEBUG_PRINT(egraph_132);text(k.type);
          SENTRYWARD_DEBUG_PRINT(egraph_133);text(k.attributes);
          SENTRYWARD_DEBUG_PRINT(egraph_134);text(k.provenance);
          SENTRYWARD_DEBUG_PRINT(egraph_135);
          for(std::size_t i=0;i<std::min<std::size_t>(k.children.size(),4);++i)
            SENTRYWARD_DEBUG_PRINT(egraph_136);
          if(k.children.size()>4)SENTRYWARD_DEBUG_PRINT(egraph_137);
          if(k.fold)SENTRYWARD_DEBUG_PRINT(egraph_138);
          SENTRYWARD_DEBUG_PRINT(egraph_139);
        };
        plain("plain_left",heads.owner(task.a));
        plain("plain_right",heads.owner(task.b));
        if(chosen) {
          plain("chosen_left",chosen->first);plain("chosen_right",chosen->second);
        }
        SENTRYWARD_DEBUG_PRINT(egraph_140);
      }
      return {};
    };
    const auto independent=[&](Class cls,unsigned environment,bool substitute,unsigned fixed_mask,
        BodyScopeRegistry::Id reference_scope,BodyScopeRegistry::Id actual_scope) {
      if(substitute && (heads.depends(cls,binder) || heads.depends(cls,block_binder)))return false;
      for(auto [a,b]:environments[environment]) {
        if(!heads.charge())return false;
        if(a!=b && (heads.depends(cls,a) || heads.depends(cls,b)))return false;
      }
      if(explicit_context) {
        for(unsigned i=0;i<initial->axis_count;++i)if(fixed_mask&(1u<<i)) {
          if(!heads.charge())return false;const auto& a=initial->axes[i];
          const bool identity=a.actual_level && *a.actual_level==a.reference_level && !a.offset;
          if(!identity && (heads.depends(cls,a.reference_level) ||
              (a.actual_level && heads.depends(cls,*a.actual_level))))return false;
        }
        std::array<std::uint32_t,max_depth+2> covered{};std::size_t n=0;
        for(unsigned i=0;i<initial->axis_count;++i)if(fixed_mask&(1u<<i)) {
          const auto& a=initial->axes[i];
          if(a.actual_level && a.reference_level==*a.actual_level && !a.offset)covered[n++]=a.reference_level;
        }
        for(auto [a,b]:environments[environment])if(a==b)covered[n++]=a;
        auto r=registry->scopeView(reference_scope),a=registry->scopeView(actual_scope);
        if(!r || !a || !registry->sameAssignment(cls,*r,*a,std::span(covered.data(),n)))return false;
      }
      if(sum_context && !sums->independent(cls,geometry.assignment,reference_scope,actual_scope,
          environments[environment]))return false;
      return true;
    };
    while(!work.empty()) {
      if(mapper_report.sums)++mapper_report.tasks;
      if(!heads.charge())return deny("work",work.back());auto t=work.back();work.pop_back();t.a=heads.owner(t.a);t.b=heads.owner(t.b);
      if(t.a>=heads.memberCount() || t.b>=heads.memberCount())return deny("member_bounds",t);
      if(snapshot[t.a].type!=snapshot[t.b].type)return deny("type",t);
      const auto pair=std::tuple{t.a,t.b,t.environment,t.substitute,t.fixed_mask,t.reference_scope,t.actual_scope};
      if(t.ready) {
        active_left.erase(t.a);active_right.erase(t.b);visited[pair]=2;continue;
      }
      if(visited.size()>=max_pairs && !visited.contains(pair))return deny("pair_cap",t);
      const auto invariant=[&](Class cls) {
        return independent(cls,t.environment,t.substitute,t.fixed_mask,t.reference_scope,t.actual_scope);
      };
      if(t.a==t.b && (invariant(t.a) || (sum_context &&
          sums->sameOwnerIdentity(t.a,geometry,t.reference_scope,t.actual_scope,
              environments[t.environment])))){visited[pair]=2;continue;}
      auto x=heads.affine(t.a),y=heads.affine(t.b);
      if(heads.query_guards.exhausted() ||
          (remaining && !*remaining))return deny("affine_work",t);
      if(x && y) {
        if(sum_context) {
          if(!sums->leaf(std::move(*x),std::move(*y),geometry,t.reference_scope,t.actual_scope,
              environments[t.environment]))return deny("sum_scoped_affine_leaf",t);
          visited[pair]=2;continue;
        }
        if(explicit_context) {
          if(!registry->leaf(*x,*y,*initial,t.fixed_mask,t.reference_scope,t.actual_scope,
              environments[t.environment]))return deny("scoped_affine_leaf",t);
          visited[pair]=2;continue;
        }
        if(!heads.charge((x->coefficients.size()+y->coefficients.size()+environments[t.environment].size()+1)*
            (1+std::bit_width(x->coefficients.size()+y->coefficients.size()+1))))return deny("affine_coeff_work",t);
        for(auto [a,b]:environments[t.environment]) {
          const auto ca=x->coefficients.contains(a)?x->coefficients.at(a):0;
          const auto cb=y->coefficients.contains(b)?y->coefficients.at(b):0;
          if(ca!=cb)return deny("bound_coefficient",t);x->coefficients.erase(a);y->coefficients.erase(b);
        }
        auto c=x->coefficients.contains(binder)?x->coefficients.at(binder):0;
        auto d=y->coefficients.contains(block_binder)?y->coefficients.at(block_binder):0;
        if(!t.substitute)c=d=0;
        if(c) {
          if((!d && !singleton) || (d && singleton) || d%c ||
              y->constant<x->constant || (y->constant-x->constant)%c)return deny("affine_stride_offset",t);
          // A scalar piece covers exactly one point. Its address contains no
          // reduction binder; only the offset is observable, not the stride.
          const auto candidate=std::pair{singleton?std::uint64_t{1}:d/c,(y->constant-x->constant)/c};
          if(map && *map!=candidate)return deny("affine_image_conflict",t,std::nullopt,candidate);map=candidate;
        } else if(d || x->constant!=y->constant)return deny("affine_constant",t);
        if(t.substitute){x->coefficients.erase(binder);y->coefficients.erase(block_binder);}
        if(x->coefficients!=y->coefficients)return deny("captured_coefficients",t);
        visited[pair]=2;continue;
      }
      if(active_left.contains(t.a) || active_right.contains(t.b))return deny("active_cycle",t);
      if(auto v=visited.find(pair);v!=visited.end()){if(v->second==1)return deny("paired_cycle",t);continue;}
      std::optional<std::pair<Class,Class>> matched;
      if(sum_context) {
        const auto selected=sums->select(t.a,t.b,t.reference_scope,t.actual_scope);
        if(selected.coverage) {
          if(!sums->related(t.a,t.b,t.reference_scope,t.actual_scope,
              environments[t.environment],geometry.assignment))return deny("sum_coverage",t);
          visited[pair]=2;continue;
        }
        matched=selected.ordinary;
      }
      // Reuse the ordinary fixed-radius witness selector before its existing
      // root-only fallback. A certified alias may move a Fold's seed into an
      // Add; selecting the first common root header misses that valid witness.
      if(!matched)matched=heads.matched(t.a,t.b);
      if(!matched) {
        if(sum_context && sums->compatible(t.a,t.b) &&
            sums->related(t.a,t.b,t.reference_scope,t.actual_scope,environments[t.environment],geometry.assignment)) {
          visited[pair]=2;continue;
        }
        return deny("head_match",t);
      }
      const auto& p=snapshot[matched->first];const auto& q=snapshot[matched->second];
      if(p.kind==static_cast<int>(s::Kind::Index) || p.children.size()!=q.children.size())return deny("index_or_arity",t,matched);
      unsigned body_environment=t.environment;
      auto reference_body_scope=t.reference_scope,actual_body_scope=t.actual_scope;
      auto body_mask=t.fixed_mask;
      if(p.fold) {
        if(!q.fold || p.fold->reducer!=q.fold->reducer || p.fold->count!=q.fold->count ||
            p.children.size()!=2 || p.kind!=static_cast<int>(s::Kind::Fold))return deny("fold_metadata",t,matched);
        const auto& enclosing=environments[t.environment];
        if(enclosing.size()>=max_depth || (!enclosing.empty() &&
            (p.fold->binder_level<=enclosing.back().first || q.fold->binder_level<=enclosing.back().second)))return deny("fold_lexical_order",t,matched);
        // Reject capture/rebinding, including an apparently harmless identical
        // eclass reused under different lexical pairings.
        for(auto [a,b]:enclosing)if(a==p.fold->binder_level || b==q.fold->binder_level)return deny("fold_capture",t,matched);
        if(!heads.charge(enclosing.size()+1))return deny("fold_work",t,matched);
        if(explicit_context || sum_context) {
          auto a=geometry_registry->insideFold(t.reference_scope,matched->first);
          auto b=geometry_registry->insideFold(t.actual_scope,matched->second);
          if(!a || !b)return deny("scoped_fold_body",t,matched);reference_body_scope=*a;actual_body_scope=*b;
          // Registry requires fresh, strictly deeper actual BODY binders.
          // Keep the mask explicit in memo keys even for conservative shadowing.
          for(unsigned i=0;explicit_context && i<initial->axis_count;++i)if(body_mask&(1u<<i)) {
            const auto& axis=initial->axes[i];
            if(axis.reference_level==p.fold->binder_level ||
                (axis.actual_level && *axis.actual_level==q.fold->binder_level))return deny("scoped_fold_capture",t,matched);
          }
          if(sum_context)for(const auto& axis:geometry.assignment.axes)
            if(axis.reference==p.fold->binder_level || axis.actual==q.fold->binder_level)
              return deny("sum_scoped_fold_capture",t,matched);
        }
        auto nested=enclosing;nested.emplace_back(p.fold->binder_level,q.fold->binder_level);
        if(!heads.charge((nested.size()+1)*(1+std::bit_width(environment_keys.size()+1))))return deny("environment_work",t,matched);
        auto found=environment_keys.find(nested);
        if(found==environment_keys.end()) {
          if(environments.size()>=max_environments)return deny("environment_cap",t,matched);
          body_environment=static_cast<unsigned>(environments.size());
          environments.push_back(nested);environment_keys.emplace(std::move(nested),body_environment);
        } else body_environment=found->second;
      }
      bool swap=false;
      std::optional<std::vector<BoundedProducts::FactorPair>> factor_pairs;
      if((arithmetic(p) || integerCommutative(p)) && p.children.size()==2) {
        auto p0=heads.owner(p.children[0]),p1=heads.owner(p.children[1]);
        auto q0=heads.owner(q.children[0]),q1=heads.owner(q.children[1]);
        const bool common_direct=(p0==q0 && invariant(p0)) || (p1==q1 && invariant(p1));
        const bool common_reverse=(p0==q1 && invariant(p0)) || (p1==q0 && invariant(p1));
        const auto root_orientation=[&]()->std::optional<bool> {
          if(!sum_context || heads.adapter)return {};
          // At most FOUR existing complete-root selections. They schedule
          // one orientation only; no child equality, affine assignment or
          // invariant is inferred from a root header. Ambiguity retains the
          // old hints BEFORE full child proof, never retries a failed walk.
          ++heads.root_orientation_attempts;
          const bool direct=bool(heads.rootMatched(p0,q0)) && bool(heads.rootMatched(p1,q1));
          const bool reverse=bool(heads.rootMatched(p0,q1)) && bool(heads.rootMatched(p1,q0));
          if(heads.exhausted || (remaining && !*remaining) ||
              heads.query_guards.exhausted())return {};
          if(direct!=reverse || (direct && (p0==p1 || q0==q1))) {
            ++heads.root_orientation_hits;return !direct;
          }
          return {};
        };
        if(common_direct)swap=false;
        else if(common_reverse)swap=true;
        else if(auto orientation=root_orientation())swap=*orientation;
        else {
          if(sum_context && (heads.exhausted || (remaining && !*remaining) ||
              heads.query_guards.exhausted()))return deny("orientation_root_work",t,matched);
          auto a=heads.shallow(p0),b=heads.shallow(p1),c=heads.shallow(q0),d=heads.shallow(q1);
          if((!a || !b || !c || !d) && !sum_context)return deny("orientation_hints",t,matched);
          bool direct=a && b && c && d && *a==*c && *b==*d;
          bool reverse=a && b && c && d && *a==*d && *b==*c;
          if(!direct && !reverse) {
            // Preferred-child shallow shapes can differ only because an
            // equivalent child prefers another stored alias. A fixed-local
            // root-header discriminator may orient ONE compatible pairing;
            // the SAME scoped walk still proves every chosen operand below.
            const auto compatible=[&](Class a,Class b) {
              return heads.localRootCompatible(a,b) || (sum_context && sums->compatible(a,b));
            };
            direct=compatible(p0,q0) && compatible(p1,q1);
            reverse=compatible(p0,q1) && compatible(p1,q0);
          }
          const auto ambiguous=[&] {return direct==reverse && (!direct || (p0!=p1 && q0!=q1));};
          if(ambiguous() && sum_context) {
            // Complete immediate root headers are a weaker scheduling hint,
            // not a proof veto based on the association of their children.
            const auto compatible=[&](Class a,Class b) {
              return bool(heads.rootMatched(a,b)) || sums->compatible(a,b);
            };
            direct=compatible(p0,q0) && compatible(p1,q1);
            reverse=compatible(p0,q1) && compatible(p1,q0);
          }
          if(ambiguous() && sum_context && p.name=="Add") {
            // An exact Negate alias can distinguish v + (-sum) even when
            // both children also have Multiply aliases. This only chooses
            // an orientation: the ordinary scoped walk proves BOTH terms.
            const auto negated=[&](Class cls) {
              const auto* choices=heads.rootChoices(cls);if(!choices)return false;
              for(const auto& [key,choice]:*choices) {
                (void)key;if(!heads.charge())return false;
                const auto& head=snapshot[choice.head];
                if(idealOperation(head) && head.name=="Negate" && head.children.size()==1)return true;
              }
              return false;
            };
            const bool a=negated(p0),b=negated(p1),c=negated(q0),d=negated(q1);
            if(a!=b && c!=d){direct=a==c;reverse=!direct;}
          }
          if(ambiguous()) {
            if(sum_context && !heads.adapter && p.name=="Multiply" && q.name=="Multiply" &&
                p.type=="real" && q.type=="real" && p.provenance.empty() && q.provenance.empty() &&
                arithmetic(p) && arithmetic(q) && finite_[heads.owner(t.a)] && finite_[heads.owner(t.b)]) {
              // Construct only when used: the factory owns deque storage,
              // whose empty construction otherwise allocates on every query.
              if(!scoped_products)scoped_products.emplace(*geometry_registry);
              factor_pairs=scoped_products->pairedFactors({t.a,t.reference_scope},{t.b,t.actual_scope});
            }
            if(!factor_pairs)return deny("orientation_root_ambiguous",t,matched);
          }
          swap=!direct;
        }
      }
      active_left.insert(t.a);active_right.insert(t.b);
      visited[pair]=1;work.push_back({t.a,t.b,t.environment,t.substitute,true,
          t.fixed_mask,t.reference_scope,t.actual_scope});
      if(factor_pairs) {
        // Repeated equal-shape factors have identical actual Uses and checked
        // occurrence counts; one proof per distinct pair suffices. Different
        // same-tag uses were denied by the EXISTING product group factory.
        for(const auto& [a,b]:*factor_pairs) {
          if(!heads.charge())return deny("product_child_work",t,matched);
          work.push_back({a.body,b.body,t.environment,t.substitute,false,
              t.fixed_mask,a.scope,b.scope});
        }
        continue;
      }
      for(std::size_t i=0;i<p.children.size();++i) {
        const bool shadow=p.fold && i==1 && p.fold->binder_level==binder;
        if(!heads.charge())return deny("child_work",t,matched);
        work.push_back({p.children[i],q.children[swap?1-i:i],
            p.fold && i==1?body_environment:t.environment,t.substitute && !shadow,false,
            p.fold && i==1?body_mask:t.fixed_mask,
            p.fold && i==1?reference_body_scope:t.reference_scope,
            p.fold && i==1?actual_body_scope:t.actual_scope});
      }
    }
    if(sum_context) {
      if(!sums->solve(geometry))return deny("sum_affine_assignment",Task{target,block});
      *assignment=std::move(geometry.assignment);
      return std::pair{std::uint64_t{1},std::uint64_t{0}};
    }
    if(!explicit_context && !map)return deny("no_observable_axis",Task{target,block});
    return explicit_context?std::optional{std::pair{std::uint64_t{1},std::uint64_t{0}}}:map;
  }
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
      const std::map<std::tuple<std::string,std::uint32_t,std::uint64_t>,std::vector<Class>>& targets) {
    // At most 64 compact Fold pieces in this bounded local Add pattern. The
    // bound limits matching work, NOT reduction count (possibly billions).
    struct Piece {Class body;std::uint64_t count;bool singleton;};
    // One local ledger and immutable selected-head/affine caches for the
    // ENTIRE stored forest/target loop. A failed target must not reset the
    // body-match budget or recopy the same signature metadata. No union is
    // made until a complete certificate succeeds and this call returns.
    std::size_t remaining=rectangle_work_limit;
    AffineCache affine_cache;
    BodyHeadIndex heads(*this,snapshot,members,affine_cache,&remaining);
    if(!heads.charge(root.children.size()))return;
    std::vector<Piece> pieces;std::vector<Class> work=root.children;std::unordered_set<Class> seen;
    std::optional<std::uint32_t> binder;std::uint64_t total=0;
    while(!work.empty()) {
      if(!heads.charge())return;auto cls=find(work.back());work.pop_back();
      if(cls>=members.size())return;
      if(!seen.insert(cls).second || seen.size()>127)return;
      std::optional<Class> fold,add;
      for(auto member:members[cls]) {
        const auto& k=snapshot[member];
        if(!BodyHeadIndex::bounded(k) ||
            !heads.charge(1+k.type.size()+k.name.size()+k.attributes.size()+k.provenance.size()))return;
        if(k.kind==static_cast<int>(s::Kind::Fold) && k.type==root.type && k.attributes.empty() &&
            k.provenance.empty() && k.fold->reducer==s::Reducer::Sum && k.fold->count &&
            k.children.size()==2 && zero_[find(k.children[0])] && finite_[cls]){fold=member;break;}
        if(arithmetic(k) && k.type==root.type && k.name=="Add" && !add)add=member;
      }
      if(fold) {
        const auto& k=snapshot[*fold];
        if(binder && *binder!=k.fold->binder_level)return;binder=k.fold->binder_level;
        if(k.fold->count>UINT64_MAX-total)return;total+=k.fold->count;
        pieces.push_back({k.children[1],k.fold->count,false});
        if(pieces.size()>64)return;
      } else if(add)for(auto child:snapshot[*add].children) {
        if(!heads.charge())return;work.push_back(child);
      }
      else {
        if(!finite_[cls] || snapshot[cls].type!=root.type || !snapshot[cls].provenance.empty() || total==UINT64_MAX)return;
        ++total;pieces.push_back({cls,1,true});if(pieces.size()>64)return;
      }
    }
    if(pieces.size()<2 || !binder)return;
    if(!heads.charge(1+root.type.size()+std::bit_width(targets.size()+1)))return;
    auto candidates=targets.find({root.type,*binder,total});if(candidates==targets.end())return;
    for(auto target:candidates->second) {
      if(!heads.charge())return;
      if(find(target)==find(id))continue;
      if(bitPermutedFoldCover(pieces,snapshot[target].children[1],*binder,snapshot,heads)) {
        rewrite(id,target);coalesced_forests_.insert(id);return;
      }
      if(!remaining || heads.exhausted)return;
      std::map<std::uint64_t,std::vector<std::pair<std::uint64_t,std::uint64_t>>> domains;
      std::optional<std::uint64_t> stride;bool valid=true;
      struct Domain {std::uint64_t begin,count;bool singleton;};
      std::vector<Domain> mapped;
      for(const auto& piece:pieces) {
        if(!heads.charge())return;
        if(piece.singleton && dependsOn(piece.body,*binder)){valid=false;break;}
        auto map=affineBodyMap(snapshot[target].children[1],piece.body,*binder,snapshot,members,
            piece.singleton,&remaining,&affine_cache,&heads);
        if(!remaining || heads.exhausted)return;
        if(!map || (!piece.singleton && stride && *stride!=map->first) || !map->first || map->first>pieces.size()){valid=false;break;}
        if(!piece.singleton)stride=map->first;
        mapped.push_back({map->second,piece.count,piece.singleton});
      }
      if(!valid || !stride)continue;
      for(auto domain:mapped) {
        if(!heads.charge(1+std::bit_width(domains.size()+1)))return;
        const auto start=domain.begin/ *stride,count=domain.count;
        if(count>UINT64_MAX-start){valid=false;break;}
        domains[domain.begin% *stride].emplace_back(start,start+count);
      }
      if(!valid || !stride || domains.size()!=*stride)continue;
      for(auto& [residue,ranges]:domains) {
        if(!heads.charge(1+ranges.size()*(1+std::bit_width(ranges.size()+1))))return;
        std::sort(ranges.begin(),ranges.end());std::uint64_t end=0;
        for(auto [begin,next]:ranges) {
          if(!heads.charge())return;if(begin!=end){valid=false;break;}end=next;
        }
        const auto required=total/ *stride+(residue<total% *stride);
        if(end!=required)valid=false;
      }
      if(valid) {
        if(!heads.charge())return;
        rewrite(id,target);coalesced_forests_.insert(id);return;
      }
    }
  }
  std::optional<IndexAffine> affineValue(Class id,AffineCache& cache,unsigned depth=0,
                                        std::size_t* remaining=nullptr,
                                        std::size_t max_entries=0,std::size_t max_terms=0) {
      poll();if(remaining){if(!*remaining)return {};--*remaining;}
      id=find(id);if(depth>256)return {};
      if(auto found=cache.find(id);found!=cache.end()) {
        // A shared cache may have been populated by a less restrictive
        // caller. Enforce this query's term bound BEFORE copying its map.
        if(max_terms && found->second && found->second->coefficients.size()>max_terms) {
          if(remaining)*remaining=0;return {};
        }
        if(remaining && found->second) {
          const auto copied=found->second->coefficients.size();
          if(copied>*remaining){*remaining=0;return {};}*remaining-=copied;
        }
        return found->second;
      }
      if(max_entries && cache.size()>=max_entries) {
        if(remaining)*remaining=0;return {};
      }
      const auto& node=nodes_[id];std::optional<IndexAffine> result;
      // Zero cache-entry limit does not disable bounded-query metadata checks.
      if((max_entries || max_terms) && (node.type.size()>128 || node.name.size()>4096 ||
          node.attributes.size()>128 || node.provenance.size()>4096 || node.children.size()>64)) {
        if(remaining)*remaining=0;return {};
      }
      if((node.type=="exact:u64" || node.type=="u64") && node.provenance.empty()) {
        if(node.kind==static_cast<int>(s::Kind::Index))
          result=IndexAffine{0,{{node.index_level,1}}};
        else if(node.kind==static_cast<int>(s::Kind::Constant)) {
          std::uint64_t value=0;
          auto parsed=std::from_chars(node.name.data(),node.name.data()+node.name.size(),value);
          if(parsed.ec==std::errc{} && parsed.ptr==node.name.data()+node.name.size())
            result=IndexAffine{value,{}};
        } else if(node.kind==static_cast<int>(s::Kind::Operation) &&
                  node.attributes.empty() && node.children.size()==2) {
          auto a=affineValue(node.children[0],cache,depth+1,remaining,max_entries,max_terms),
               b=affineValue(node.children[1],cache,depth+1,remaining,max_entries,max_terms);
          if(a && b) {
            if(remaining) {
              const auto copied=a->coefficients.size()+b->coefficients.size();
              if(copied>*remaining){*remaining=0;return {};}*remaining-=copied;
            }
            result=combineAffine(node.name,*a,*b);
          }
        }
      }
      if(max_terms && result && result->coefficients.size()>max_terms) {
        if(remaining)*remaining=0;return {};
      }
      if(remaining && result) {
        const auto copied=result->coefficients.size();
        if(copied>*remaining){*remaining=0;return {};}*remaining-=copied;
      }
      if(max_entries && cache.size()>=max_entries) {
        if(remaining)*remaining=0;return {};
      }
      cache.emplace(id,result);return result;
  }
  bool shiftedProjectedBody(Class first,Class next,std::uint32_t binder,std::uint64_t shift,
                            std::uint64_t count) {
    first=find(first);next=find(next);
    const auto query=std::tuple{first,next,binder,shift,count};
    if(auto found=shifted_body_matches_.find(query);found!=shifted_body_matches_.end())return found->second;
    constexpr std::size_t max_queries=4096,max_pairs=4096,max_affine_nodes=4096,max_terms=64;
    constexpr std::size_t max_local_work=131072;
    const auto capped=[&](const char* ledger) {
      if(!shifted_body_cap_reported_ && sentryward::debug::environment("SENTRYWARD_EGRAPH_TRACE")) {
        shifted_body_cap_reported_=true;
        SENTRYWARD_DEBUG_PRINT(egraph_142);
      }
      return false;
    };
    if(shifted_body_matches_.size()>=max_queries)return capped("queries");
    // A body match visits template pairs, not reduction members. Memoization
    // lasts for one congruence epoch; a new union can expose a new match.
    const auto finish=[&](bool result){shifted_body_matches_.emplace(query,result);return result;};
    std::size_t remaining=max_local_work;
    const auto charge=[&](std::size_t amount) {
      if(amount>remaining){remaining=0;return false;}
      remaining-=amount;poll();
      if(options_.max_work && amount>1) {
        if(amount-1>options_.max_work-work_)throw Limit{"work_budget"};
        work_+=amount-1;
      }
      return true;
    };
    const auto boundedHead=[](const Key& key) {
      return key.type.size()<=128 && key.name.size()<=4096 && key.attributes.size()<=128 &&
             key.provenance.size()<=4096 && key.children.size()<=64;
    };
    AffineCache cache;
    namespace bit=bounded_index_translation;
    std::optional<bit::Session> euclidean;
    if(options_.outer_index_extents.size()<64) {
    std::vector<bit::IndexExtent> extents;
    for(std::size_t level=0;level<options_.outer_index_extents.size();++level) {
      if(!charge(1))return finish(capped("work"));
      extents.push_back({static_cast<std::uint32_t>(level),options_.outer_index_extents[level]});
    }
    // One scoped dual-DAG session per body match. Only stored exact u64 heads
    // are resolved; no e-class-member, ancestor, or logical Fold-member scan.
    euclidean.emplace([&](Class id)->std::optional<bit::Node> {
      const auto& k=nodes_[find(id)];
      if(!boundedHead(k) || !k.attributes.empty() || !k.provenance.empty() || k.fold || k.children.size()>2)return {};
      bit::Node result;result.type=k.type;result.admitted=true;
      if(k.kind==static_cast<int>(s::Kind::Constant)) {
        result.kind=bit::Kind::Literal;result.literal=k.name;
      } else if(k.kind==static_cast<int>(s::Kind::Index)) {
        result.kind=bit::Kind::Index;result.index_level=k.index_level;
      } else if(k.kind==static_cast<int>(s::Kind::Operation)) {
        if(k.name=="Add")result.kind=bit::Kind::Add;
        else if(k.name=="Multiply")result.kind=bit::Kind::Multiply;
        else if(k.name=="Divide")result.kind=bit::Kind::Divide;
        else if(k.name=="Remainder")result.kind=bit::Kind::Remainder;
        else return {};
      } else return {};
      result.child_count=k.children.size();
      std::copy(k.children.begin(),k.children.end(),result.children.begin());return result;
    },extents,binder,shift,count,bit::Limits{},charge);
    }
    struct Task {Class first,next;bool ready=false;};
    std::vector<Task> work{{first,next}};
    // A cycle introduced by e-class equality is not a structural proof. Only
    // completed pairs may be reused; recursive pairs conservatively fail.
    std::map<std::pair<Class,Class>,unsigned char> visited;
    while(!work.empty()) {
      if(!charge(1))return finish(capped("work"));
      auto [a,b,ready]=work.back();work.pop_back();a=find(a);b=find(b);
      const auto pair=std::pair{a,b};
      if(ready){visited[pair]=2;continue;}
      if(auto found=visited.find(pair);found!=visited.end()) {
        if(found->second==1)return finish(false);
        continue;
      }
      if(visited.size()>=max_pairs)return finish(capped("pairs"));
      if(a==b && !dependsOn(a,binder)){visited[pair]=2;continue;}
      if(euclidean && euclidean->equivalent(a,b)){visited[pair]=2;continue;}
      auto x=affineValue(a,cache,0,&remaining,max_affine_nodes,max_terms),
           y=affineValue(b,cache,0,&remaining,max_affine_nodes,max_terms);
      if(!remaining)return finish(capped("affine_or_work"));
      if(x && y) {
        if(!shiftedAffine(*x,*y,binder,shift))return finish(false);
        visited[pair]=2;
        continue;
      }
      const auto& lhs=nodes_[a];const auto& rhs=nodes_[b];
      if(!boundedHead(lhs) || !boundedHead(rhs))return finish(capped("metadata"));
      if(lhs.kind!=rhs.kind || lhs.type!=rhs.type || lhs.name!=rhs.name ||
          lhs.attributes!=rhs.attributes || lhs.provenance!=rhs.provenance ||
          lhs.fold!=rhs.fold || lhs.index_level!=rhs.index_level ||
          lhs.children.size()!=rhs.children.size())return finish(false);
      if(lhs.kind==static_cast<int>(s::Kind::Index) && lhs.index_level==binder && shift)return finish(false);
      visited[pair]=1;work.push_back({a,b,true});
      const auto sameHead=[&](const Key& x,const Key& y) {
        return boundedHead(x) && boundedHead(y) && x.kind==y.kind && x.type==y.type && x.name==y.name && x.attributes==y.attributes &&
            x.provenance==y.provenance && x.fold==y.fold && x.index_level==y.index_level &&
            x.children.size()==y.children.size();
      };
      // E-class IDs need not sort corresponding commutative operands alike
      // after an index shift. Select a uniquely determined head permutation;
      // ambiguous heads stay conservative, with no exponential backtracking.
      bool swapped=false;
      if((arithmetic(lhs) && arithmetic(rhs)) ||
          (integerCommutative(lhs) && integerCommutative(rhs))) {
        const auto& x=nodes_[find(lhs.children[0])];const auto& y=nodes_[find(lhs.children[1])];
        const auto& u=nodes_[find(rhs.children[0])];const auto& v=nodes_[find(rhs.children[1])];
        swapped=(!sameHead(x,u) || !sameHead(y,v)) && sameHead(x,v) && sameHead(y,u);
      }
      for(std::size_t i=0;i<lhs.children.size();++i) {
        if(lhs.kind==static_cast<int>(s::Kind::Fold) && i==1 && lhs.fold->binder_level==binder) {
          if(find(lhs.children[i])!=find(rhs.children[i]))return finish(false);
          continue;
        }
        work.push_back({lhs.children[i],rhs.children[swapped?1-i:i],false});
      }
    }
    return finish(true);
  }
  void concatenateProjectedFold(Key& key) {
    if(key.kind!=static_cast<int>(s::Kind::Fold) || key.type!="real" ||
        key.children.size()!=2 || !key.provenance.empty())return;
    const auto& prefix=nodes_[key.children[0]];
    if(prefix.kind!=key.kind || prefix.type!=key.type || prefix.children.size()!=2 ||
        !prefix.provenance.empty() || prefix.fold->reducer!=key.fold->reducer ||
        prefix.fold->binder_level!=key.fold->binder_level ||
        key.fold->count>UINT64_MAX-prefix.fold->count)return;
    if(!shiftedProjectedBody(prefix.children[1],key.children[1],
                             key.fold->binder_level,prefix.fold->count,key.fold->count))return;
    key.fold->count+=prefix.fold->count;
    key.children=prefix.children;
  }

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
  bool branchCharge(std::size_t amount=1) {
    poll();if(amount>branch_remaining_){branch_remaining_=0;return false;}
    branch_remaining_-=amount;
    if(options_.max_work && amount>1) {
      if(amount-1>options_.max_work-work_)throw Limit{"work_budget"};
      work_+=amount-1;
    }
    return true;
  }
  static bool branchPredicateSource(const Key& k,const std::vector<Key>& arena) {
    if(comparison(k)!=std::optional<std::string>{"gt"} ||
        (k.attributes!="predicate=gt" && !k.attributes.empty()) ||
        !k.provenance.empty() || k.children.size()!=2)return false;
    const auto& argument=arena[k.children[0]];const auto& zero=arena[k.children[1]];
    const auto plain=[&](const Key& h,std::string_view name) {
      return h.kind==static_cast<int>(s::Kind::Operation) && h.type==argument.type &&
          h.name==name && h.children.size()==2 && h.provenance.empty() &&
          (h.attributes.empty() || h.attributes=="ideal_arithmetic");
    };
    if((argument.type!="f32" && argument.type!="f64") || !plain(argument,"Add") ||
        zero.kind!=static_cast<int>(s::Kind::Constant) || zero.type!=argument.type ||
        zero.name.size()>32 || !zero.attributes.empty() || !zero.provenance.empty())return false;
    // Import already decoded finite typed constants; do not parse a literal
    // again at every node-intern/sensitivity transfer.
    if(zero.name!="0")return false;
    for(auto child:argument.children)if(plain(arena[child],"Subtract"))return true;
    return false;
  }
  static bool predicateFloatConversion(const Key& k,const std::vector<Key>& arena) {
    if(k.kind!=6 || !k.provenance.empty() || !k.attributes.empty() ||
        k.children.size()!=1 || arena[k.children[0]].type!="pred" ||
        (k.type!="f32" && k.type!="f64"))return false;
    const auto step=leadingConversion("pred",k.name);
    return step && step->target==k.type && step->remainder.empty() &&
        (step->operation=="Convert" || step->operation=="Convert rounding=nearest-even");
  }
  static bool predicateConjunction(const Key& k,const std::vector<Key>& arena) {
    return k.kind==static_cast<int>(s::Kind::Operation) && k.name=="BitAnd" &&
        k.type=="pred" && k.attributes.empty() && k.provenance.empty() && k.children.size()==2 &&
        arena[k.children[0]].type=="pred" && arena[k.children[1]].type=="pred";
  }
  void prepareBranchFacts() {
    if(branch_fact_anchor_ || !options_.certified_branch_scope)return;
    const auto& evidence=*options_.certified_branch_scope;
    if(evidence.pieces().empty() || evidence.pieces().size()>64 || !branchCharge(evidence.pieces().size()*4))return;
    std::vector<s::Ref> refs;
    for(const auto& piece:evidence.pieces()) {
      refs.push_back(piece.expression);
      refs.insert(refs.end(),piece.encoded_operands.begin(),piece.encoded_operands.end());
    }
    const auto encoded=import(evidence.graph(),refs,nullptr,{},nullptr,true);
    Key anchor;anchor.kind=7;anchor.type="branch-facts";anchor.name="CapturedConditionalFacts";
    anchor.attributes=evidence.digest();
    for(auto ref:encoded)anchor.children.push_back(project(ref,true));
    branch_fact_anchor_=intern(std::move(anchor));
  }
  std::optional<std::size_t> branchScope(ScopeId id) {
    if(auto it=branch_scope_ids_.find(id);it!=branch_scope_ids_.end())return it->second;
    if(!projection_scopes_ || id==lexical_scope::unknown || branch_scopes_.size()>=512)return {};
    const auto counts=projection_scopes_->counts(id);
    if(!counts || counts->size()>64 || !branchCharge(counts->size()+1))return {};
    BranchScopeRecord scope;
    for(auto count:*counts){if(!count || !*count)return {};scope.counts.push_back(*count);}
    const auto result=branch_scopes_.size();branch_scopes_.push_back(std::move(scope));
    branch_scope_ids_.emplace(id,result);return result;
  }
  static std::optional<std::pair<std::size_t,std::string>> branchWrapper(const Key& k) {
    if(k.kind!=7 || k.type!="pred" || k.name!="ScopedPositivePredicate" ||
        !k.provenance.empty() || k.children.size()!=2 || k.attributes.size()>32)return {};
    const auto colon=k.attributes.find(':');if(colon==std::string::npos)return {};
    std::size_t scope=0;const auto parsed=std::from_chars(k.attributes.data(),k.attributes.data()+colon,scope);
    const auto format=k.attributes.substr(colon+1);
    if(parsed.ec!=std::errc{} || parsed.ptr!=k.attributes.data()+colon ||
        (format!="f32" && format!="f64"))return {};
    return std::pair{scope,format};
  }
  // A mask describes only a subset of an already validated ACTUAL binder.
  // This is fixed-head exact-u64 geometry, not a branch-note interpretation.
  std::optional<std::vector<std::pair<std::uint64_t,std::uint64_t>>> branchMask(
      Class mask,std::uint32_t binder,std::uint64_t extent,const std::vector<Key>& snapshot,
      BodyHeadIndex& heads) {
    mask=find(mask);if(mask>=snapshot.size() || !branchCharge())return {};
    if(predicate_[mask]==1)return std::vector<std::pair<std::uint64_t,std::uint64_t>>{{0,extent}};
    if(predicate_[mask]==2)return std::vector<std::pair<std::uint64_t,std::uint64_t>>{};
    // A certified conjunction can share an e-class with its exact mask while
    // the canonical owner still names the now-cyclic private conjunction.
    // Reuse the bounded existing selected-head catalog, not that owner head.
    const auto witness=heads.compact(mask);if(!witness || *witness>=snapshot.size())return {};
    const auto& k=snapshot[*witness];const auto cmp=comparison(k);
    if(cmp!=std::optional<std::string>{"ne"} || k.attributes!="predicate=ne" ||
        k.children.size()!=2 || !k.provenance.empty())return {};
    for(unsigned side=0;side<2;++side) {
      const auto coordinate=find(k.children[side]),point=find(k.children[1-side]);
      if(coordinate>=snapshot.size() || point>=snapshot.size())continue;
      if(snapshot[coordinate].type!=snapshot[point].type)continue;
      std::optional<IndexAffine> a,b;
      if(snapshot[coordinate].type=="u64" || snapshot[coordinate].type=="exact:u64") {
        a=heads.affine(coordinate);b=heads.affine(point);
      } else if(snapshot[coordinate].type=="s64" || snapshot[coordinate].type=="exact:s64") {
        // ONE exact value-preserving integer conversion is only a geometry
        // view. The original typed comparison/bit consumer is never aliased.
        // At most TWO same-type exact Add(0,x)/Multiply(1,x) envelopes.
        // This fixed local identity view does not follow arbitrary ancestry.
        Class terminal=coordinate;std::optional<Class> selected;
        for(unsigned depth=0;depth<3;++depth) {
          selected=heads.compact(terminal);if(!selected || !branchCharge(4))break;
          const auto& h=snapshot[*selected];
          if(h.kind==6)break;
          if(depth==2 || h.kind!=static_cast<int>(s::Kind::Operation) ||
              h.type!=snapshot[coordinate].type || !h.attributes.empty() || !h.provenance.empty() ||
              h.children.size()!=2 || (h.name!="Add" && h.name!="Multiply")){selected.reset();break;}
          bool identity=false;
          for(unsigned side=0;side<2;++side) {
            const auto c=find(h.children[side]),x=find(h.children[1-side]);
            if(c>=snapshot.size() || x>=snapshot.size())continue;
            const auto& constant=snapshot[c];
            if(constant.kind==static_cast<int>(s::Kind::Constant) && constant.type==h.type &&
                constant.attributes.empty() && constant.provenance.empty() &&
                constant.name==(h.name=="Add"?"0":"1") && snapshot[x].type==h.type) {
              terminal=x;identity=true;break;
            }
          }
          if(!identity){selected.reset();break;}
        }
        if(!selected)continue;
        const auto& cast=snapshot[*selected];const auto& literal=snapshot[point];
        if(cast.kind!=6 || cast.type!=snapshot[coordinate].type || !cast.attributes.empty() || !cast.provenance.empty() ||
            cast.children.size()!=1 || literal.kind!=static_cast<int>(s::Kind::Constant) ||
            !literal.attributes.empty() || !literal.provenance.empty() || literal.name.size()>32 ||
            cast.name!="Convert(u64 -> s64)" || !branchCharge(8))continue;
        const auto child=find(cast.children[0]);if(child>=snapshot.size() ||
            (snapshot[child].type!="u64" && snapshot[child].type!="exact:u64"))continue;
        a=heads.affine(child);
        if(!a || a->coefficients.size()!=1 || !a->coefficients.contains(binder) ||
            a->coefficients.at(binder)!=1 || !extent ||
            a->constant>std::uint64_t(INT64_MAX) || extent-1>std::uint64_t(INT64_MAX)-a->constant)continue;
        auto range=integer_interval::interval({false,64},static_cast<std::int64_t>(a->constant),
            static_cast<std::int64_t>(a->constant+extent-1));
        auto point_range=integer_interval::literal("s64",literal.name);
        if(!range || !integerConversionRange("u64","s64",cast.name,*range) ||
            !point_range || point_range->lower!=point_range->upper || point_range->lower<0)continue;
        b=IndexAffine{static_cast<std::uint64_t>(point_range->lower),{}};
      } else continue;
      if(!a || !b || a->coefficients.size()!=1 || !a->coefficients.contains(binder) ||
          a->coefficients.at(binder)!=1 || !b->coefficients.empty() ||
          a->constant>UINT64_MAX-extent)continue;
      const auto end=a->constant+extent;
      if(b->constant<a->constant || b->constant>=end)
        return std::vector<std::pair<std::uint64_t,std::uint64_t>>{{0,extent}};
      const auto omitted=b->constant-a->constant;
      std::vector<std::pair<std::uint64_t,std::uint64_t>> intervals;
      if(omitted)intervals.emplace_back(0,omitted);
      if(omitted+1<extent)intervals.emplace_back(omitted+1,extent);
      return intervals;
    }
    return {};
  }
  void rewriteScopedBranchPredicates(const std::vector<Key>& snapshot,
      const std::vector<std::vector<Class>>& members) {
    if(!options_.certified_branch_scope || !branch_remaining_)return;
    AffineCache affine;BodyHeadIndex heads(*this,snapshot,members,affine,&branch_remaining_);
    // Share one comparison-wide TRACE allowance with the existing paired-map
    // denial printer. No extra proof query, rendering, or ancestor traversal.
    if(sentryward::debug::environment("SENTRYWARD_EGRAPH_TRACE"))heads.denial_trace_remaining=&branch_trace_remaining_;
    namespace l=bounded_linear_reduction;
    l::Limits limits;limits.max_intervals=128;limits.max_work=262144;
    l::Builder intervals(limits,[&](std::size_t amount){return branchCharge(amount);});
    const auto one=intervals.coefficient("1");if(!one)return;
    std::map<std::pair<Class,Class>,bool> memo;
    const auto certificate=[&](Class wrapper,Class mask)->bool {
      wrapper=find(wrapper);mask=mask==SIZE_MAX?mask:find(mask);
      const auto key=std::pair{wrapper,mask};
      if(auto it=memo.find(key);it!=memo.end())return it->second;
      const auto finish=[&](bool proof){if(memo.size()<512)memo.emplace(key,proof);return proof;};
      if(memo.size()>=512 || branch_requests_>=2048 || !branchCharge() || wrapper>=snapshot.size())return false;
      ++branch_requests_;
      const auto parsed=branchWrapper(snapshot[wrapper]);
      if(!parsed || parsed->first>=branch_scopes_.size())return finish(false);
      const auto& wrapper_head=snapshot[wrapper];
      const auto cmp_id=find(wrapper_head.children[0]),anchor_id=find(wrapper_head.children[1]);
      if(cmp_id>=snapshot.size() || anchor_id>=snapshot.size())return finish(false);
      const auto& cmp=snapshot[cmp_id];const auto& anchor=snapshot[anchor_id];
      if(comparison(cmp)!=std::optional<std::string>{"gt"} || cmp.attributes!="predicate=gt" ||
          anchor.kind!=7 || anchor.type!="branch-facts" || anchor.name!="CapturedConditionalFacts" ||
          anchor.attributes!=options_.certified_branch_scope->digest() || !anchor.provenance.empty() ||
          anchor.children.size()!=options_.certified_branch_scope->pieces().size()*4 ||
          !finite_[find(cmp.children[0])] || !zero_[find(cmp.children[1])] ||
          !finite_[find(cmp.children[1])])return finish(false);
      const auto expression=find(cmp.children[0]);
      if(expression>=snapshot.size() || has_fold_[expression])return finish(false);
      const auto& free=free_indices_[expression];
      if(free.size()!=1)return finish(false);
      const auto binder=free.front();const auto& scope=branch_scopes_[parsed->first];
      if(binder>=scope.counts.size())return finish(false);
      const auto extent=scope.counts[binder];
      auto wanted=mask==SIZE_MAX?
          std::optional{std::vector<std::pair<std::uint64_t,std::uint64_t>>{{0,extent}}}:
          branchMask(mask,binder,extent,snapshot,heads);
      if(heads.denial_trace_remaining && *heads.denial_trace_remaining) {
        --*heads.denial_trace_remaining;
        SENTRYWARD_DEBUG_PRINT(egraph_143);
        if(mask==SIZE_MAX)SENTRYWARD_DEBUG_PRINT(egraph_144);else SENTRYWARD_DEBUG_PRINT(egraph_145);
        SENTRYWARD_DEBUG_PRINT(egraph_146);
        if(!wanted)SENTRYWARD_DEBUG_PRINT(egraph_147);
        else if(wanted->empty())SENTRYWARD_DEBUG_PRINT(egraph_148);
        else for(const auto& [begin,end]:*wanted)SENTRYWARD_DEBUG_PRINT(egraph_149);
        SENTRYWARD_DEBUG_PRINT(egraph_150);
      }
      if(!wanted)return finish(false);
      if(wanted->empty())return finish(true);
      std::vector<l::WeightedInterval> covered,required;
      for(auto [begin,end]:*wanted)required.push_back({begin,end,*one});
      const auto pieces=options_.certified_branch_scope->pieces();
      for(std::size_t n=0;n<pieces.size();++n) {
        if(!branchCharge(5))return finish(false);
        if(pieces[n].format!=parsed->second || !pieces[n].count)continue;
        const auto fact=find(anchor.children[4*n]);bool finite=true;
        if(fact>=snapshot.size() || has_fold_[fact])continue;
        for(unsigned j=0;j<4;++j)finite=finite && finite_[find(anchor.children[4*n+j])];
        if(!finite)continue;
        const bool point=pieces[n].count==1 &&
            !std::binary_search(free_indices_[fact].begin(),free_indices_[fact].end(),branch_fact_binder);
        const auto image=affineBodyMapWithHeads(expression,fact,binder,snapshot,point,&branch_remaining_,heads,branch_fact_binder);
        if(!image) {if(heads.exhausted || !branch_remaining_)return finish(false);continue;}
        if(image->first!=1 || image->second>UINT64_MAX-pieces[n].count)continue;
        const auto begin=image->second,end=std::min(extent,begin+pieces[n].count);
        for(auto [first,last]:*wanted) {
          const auto a=std::max(first,begin),b=std::min(last,end);
          if(a<b)covered.push_back({a,b,*one});
        }
      }
      // Every input weight is +1: canonical nonzero runs are exactly the
      // union of certified reached images, even if observations overlap.
      auto canonical=intervals.canonicalIntervals(covered);if(!canonical)return finish(false);
      for(auto& piece:*canonical)piece.weight=*one;
      const auto equal=intervals.intervalsEqual(*canonical,required);
      const bool proved=equal && *equal;
      branch_admitted_+=proved;
      branch_scope_used_=branch_scope_used_ || proved;
      return finish(proved);
    };
    const auto candidate_count=branch_candidates_.size();
    for(std::size_t candidate=0;candidate<candidate_count;++candidate) {
      if(!branchCharge())break;
      const auto id=branch_candidates_[candidate];if(id>=snapshot.size())continue;
      const auto& k=snapshot[id];
      if(branchWrapper(k)) {
        if(!predicate_[find(id)] && certificate(id,SIZE_MAX) && branch_proved_wrappers_.insert(id).second)
          enqueueProperty(id); // Private scoped fact, NEVER a global GT/true union.
      } else if(k.kind==7 && k.type=="pred" && k.name=="ScopedPredicateAnd" &&
          k.attributes.empty() && k.provenance.empty() && k.children.size()==2) {
        for(unsigned side=0;side<2;++side) {
          const auto a=find(k.children[side]),b=find(k.children[1-side]);
          if(predicate_[a]==1 || certificate(a,b)){rewrite(id,b);break;}
          if(predicate_[a]==2){rewrite(id,a);break;}
        }
      } else if(k.kind==static_cast<int>(s::Kind::Fold) && k.type=="real" && k.fold &&
          k.fold->reducer==s::Reducer::Sum && k.fold->count && k.attributes.empty() &&
          k.provenance.empty() && k.children.size()==2 && zero_[find(k.children[0])]) {
        const auto body=find(k.children[1]);if(body>=snapshot.size())continue;
        const auto& select=snapshot[body];
        if(!idealSelect(select) || select.type!="real" || !zero_[find(select.children[2])])continue;
        const auto value_id=find(select.children[1]);if(value_id>=snapshot.size())continue;
        const auto witness=heads.compact(value_id);if(!witness || *witness>=snapshot.size())continue;
        const auto& value=snapshot[*witness];
        if(value.kind!=static_cast<int>(s::Kind::Constant) || value.type!="real" || value.name!="1" ||
            !value.attributes.empty() || !value.provenance.empty())continue;
        const auto mask=branchMask(select.children[0],k.fold->binder_level,k.fold->count,snapshot,heads);
        if(!mask)continue;
        std::uint64_t count=0;for(auto [begin,end]:*mask)count+=end-begin;
        Key constant;constant.kind=static_cast<int>(s::Kind::Constant);constant.type="real";constant.name=std::to_string(count);
        rewrite(id,intern(std::move(constant)));++branch_indicators_;
      }
    }
  }
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
  bool projectionCharge(std::size_t amount=1) {
    poll();
    if(amount>projection_sat_remaining_){projection_sat_remaining_=0;return false;}
    projection_sat_remaining_-=amount;
    if(options_.max_work && amount>1) {
      if(amount-1>options_.max_work-work_)throw Limit{"work_budget"};
      work_+=amount-1;
    }
    return true;
  }
  static bool scopedSatHead(const Key& k,const std::vector<Key>& arena) {
    if(k.kind!=6 || k.type!="e4m3" || !k.attributes.empty() || !k.provenance.empty() ||
        k.children.size()!=1 || arena[k.children[0]].type!="f32")return false;
    const auto step=leadingConversion("f32",k.name);
    return step && step->operation=="Convert rounding=nearest-even saturate_finite" &&
        step->target=="e4m3" && step->remainder.empty();
  }
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
  bool scopedExponentialSource(Class exponential) const {
    // Keep all complete local typing/modifier/provenance guards BEFORE type
    // erasure. The body proof later consumes only the Numeric views.
    const auto op=[&](Class id,std::string_view name,std::size_t arity) {
      const auto& k=nodes_[id];
      return k.type=="f32" && k.provenance.empty() && idealOperation(k,true) &&
          k.name==name && k.children.size()==arity &&
          std::all_of(k.children.begin(),k.children.end(),
              [&](Class child){return nodes_[child].type=="f32";});
    };
    if(!op(exponential,"Exp2",1))return false;
    const auto argument=nodes_[exponential].children[0];
    if(!op(argument,"Add",2))return false;
    for(unsigned side=0;side<2;++side) {
      const auto a=nodes_[argument].children[side],b=nodes_[argument].children[1-side];
      if(auto difference=localDifference(a,nodes_,op); difference && op(b,"Multiply",2)) {
        const auto product=difference->second;if(!op(product,"Multiply",2))continue;
        for(unsigned x=0;x<2;++x)for(unsigned y=0;y<2;++y)
          if(nodes_[product].children[x]==nodes_[b].children[y])return true;
      } else if(op(b,"Multiply",2)) {
        for(auto child:nodes_[b].children)if(localDifference(child,nodes_,op))return true;
      }
    }
    return false;
  }
  void initializeProjectionScope() {
    if(projection_scopes_)return;
    projection_scopes_=std::make_unique<lexical_scope::Catalog>(lexical_scope::Limits{},
        [this](std::size_t amount){return projectionCharge(amount);});
    projection_root_scope_=projection_scopes_->root(options_.outer_binders,
        options_.outer_index_extents).value_or(lexical_scope::unknown);
    projection_source_ids_.resize(nodes_.size());
    std::iota(projection_source_ids_.begin(),projection_source_ids_.end(),Class{0});
    // Explicit arena callbacks, not IDs from a second private graph. This is
    // one Engine's append-only projection epoch, BEFORE unions/rebuild. Child
    // properties are complete before their parent is interned. All borrowed
    // headers/spans live only within a query which itself does not intern.
    BodyHeadIndex::Adapter adapter;
    adapter.owner=[this](Class id) {
      return id<nodes_.size() && parents_[id]==id?id:std::numeric_limits<Class>::max();
    };
    adapter.affine=[this](Class id,AffineCache& cache,std::size_t& remaining) {
      if(id>=nodes_.size() || parents_[id]!=id)return std::optional<IndexAffine>{};
      return affineValue(id,cache,0,&remaining,0,64);
    };
    adapter.depends=[this](Class id,std::uint32_t binder) {
      if(id>=nodes_.size() || parents_[id]!=id)return true;
      const auto& levels=free_indices_[id];
      return std::binary_search(levels.begin(),levels.end(),binder);
    };
    projection_body_heads_=std::make_unique<BodyHeadIndex>(*this,nodes_,projection_source_ids_,
        projection_affine_cache_,&projection_sat_remaining_,std::move(adapter));
  }
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
  const MaxSamples* projectionMaxSamples(Class maximum) {
    if(auto found=projection_max_samples_.find(maximum);found!=projection_max_samples_.end())
      return &found->second;
    if(projection_max_samples_.size()>=128)return nullptr;
    MaxSamples result;bool valid=true;
    std::vector<MaxSample> todo{{maximum,0,1,true}};
    std::set<std::tuple<Class,std::uint32_t,std::uint64_t,bool>> seen;
    while(!todo.empty()) {
      if(!projectionCharge()){valid=false;break;}
      const auto sample=todo.back();todo.pop_back();const auto id=sample.body;
      if(!seen.emplace(id,sample.binder,sample.count,sample.singleton).second)continue;
      if(seen.size()>1024 || id>=nodes_.size()){valid=false;break;}
      const auto& k=nodes_[id];
      if(k.type=="real" && k.provenance.empty() && negative_infinity_[id])continue;
      if(!projectionFinite(id)){valid=false;break;}
      if(projectedOperation(id,"Maximum",2)) {
        for(auto child:k.children)todo.push_back({child,sample.binder,sample.count,sample.singleton});
        continue;
      }
      if(k.kind==static_cast<int>(s::Kind::Fold) && k.fold &&
          k.fold->reducer==s::Reducer::Max && k.children.size()==2 &&
          (k.attributes.empty() || k.attributes=="ideal_arithmetic")) {
        if(!sample.singleton || !k.fold->count || !projectionFinite(k.children[1]) ||
            std::any_of(free_indices_[k.children[0]].begin(),free_indices_[k.children[0]].end(),
                [&](auto level){return level>=k.fold->binder_level;}) ||
            std::any_of(free_indices_[k.children[1]].begin(),free_indices_[k.children[1]].end(),
                [&](auto level){return level>k.fold->binder_level;})){valid=false;break;}
        todo.push_back({k.children[1],k.fold->binder_level,k.fold->count,false});
        if(!negative_infinity_[k.children[0]])todo.push_back({k.children[0],0,1,true});
      } else {
        if(result.samples.size()>=1024 || projection_sample_count_>=4096){valid=false;break;}
        result.samples.push_back(sample);++projection_sample_count_;
      }
    }
    result.valid=valid;
    return &projection_max_samples_.emplace(maximum,std::move(result)).first->second;
  }
  bool projectedMaxContains(Class maximum,Class score,ScopeId scope) {
    const auto key=std::tuple{maximum,score,scope};
    if(auto found=projection_max_membership_.find(key);found!=projection_max_membership_.end())
      return found->second;
    if(projection_max_membership_.size()>=2048)return false;
    const auto finish=[&](bool answer) {
      projection_max_membership_.emplace(key,answer);return answer;
    };
    if(!projection_scopes_ || scope==lexical_scope::unknown ||
        !projectionFinite(maximum) || !projectionFinite(score))return finish(false);
    auto counts=projection_scopes_->counts(scope);if(!counts)return finish(false);
    // Unknown and empty domains never supply point coverage. A rank-zero
    // known scope is valid and can certify an identical scalar Max sample.
    for(auto count:*counts)if(!count || !*count)return finish(false);
    std::optional<std::uint32_t> binder;
    for(auto level:free_indices_[score]) {
      if(!projectionCharge() || level>=counts->size())return finish(false);
      if(!std::binary_search(free_indices_[maximum].begin(),free_indices_[maximum].end(),level)) {
        if(binder)return finish(false);binder=level;
      }
    }
    const auto* collection=projectionMaxSamples(maximum);
    if(!collection || !collection->valid)return finish(false);
    if(!binder) {
      for(const auto& sample:collection->samples) {
        if(!projectionCharge())return finish(false);
        if(sample.singleton && sample.body==score)return finish(true);
      }
      return finish(false);
    }
    const auto count=projection_scopes_->count(scope,*binder);
    if(!count || !*count)return finish(false);
    struct Progression {std::uint64_t begin,stride,count;};
    std::vector<Progression> progressions;std::uint64_t lattice=1;
    for(const auto& sample:collection->samples) {
      if(!projectionCharge())return finish(false);
      // A Fold binds its BODY, not its seed. Mismatched lexical depth is
      // Unknown, and inferred mapped j must stay in an actual stored interval.
      if(!sample.singleton) {
        const auto parent=projection_scopes_->prefix(scope,sample.binder);
        if(!parent || !projection_scopes_->bindBody(*parent,sample.binder,sample.count))continue;
      }
      const auto map=affineBodyMapWithHeads(score,sample.body,*binder,nodes_,
          sample.singleton,&projection_sat_remaining_,*projection_body_heads_,
          sample.singleton?*binder:sample.binder);
      if(!map) {
        if(!projection_sat_remaining_ || projection_body_heads_->exhausted)return finish(false);
        continue; // A Maximum dominates any certified subset of its samples.
      }
      if(!map->first)continue;
      if(map->second>=*count)continue;
      std::uint64_t next;
      if(__builtin_mul_overflow(lattice/std::gcd(lattice,map->first),map->first,&next) || next>1024)
        continue; // This subset cannot be represented within the proof budget.
      lattice=next;
      progressions.push_back({map->second,map->first,
          std::min(sample.count,1+(*count-1-map->second)/map->first)});
    }
    // A Max dominates the UNION of its parameterized samples, including
    // interleaved blocks. Unlike Sum coverage, overlap is harmless here.
    for(std::uint64_t residue=0;residue<lattice && residue<*count;++residue) {
      std::vector<std::pair<std::uint64_t,std::uint64_t>> intervals;
      for(const auto& p:progressions) {
        if(!projectionCharge(8))return finish(false);
        const auto delta=(residue+lattice-p.begin%lattice)%lattice;
        if(delta%p.stride)continue;
        const auto first=delta/p.stride;if(first>=p.count)continue;
        const auto begin=(p.begin+delta)/lattice;
        intervals.emplace_back(begin,begin+1+(p.count-1-first)/(lattice/p.stride));
      }
      if(!projectionCharge(intervals.size()*(1+std::bit_width(intervals.size()+1))))return finish(false);
      std::sort(intervals.begin(),intervals.end());std::uint64_t end=0;
      for(auto [begin,next]:intervals){if(begin>end)return finish(false);end=std::max(end,next);}
      if(end<1+(*count-1-residue)/lattice)return finish(false);
    }
    return finish(true);
  }
  bool projectedInactiveSaturation(Class conversion,Class exponential,ScopeId scope) {
    const ProjectionKey key{conversion,scope};
    if(auto found=projection_sat_certificates_.find(key);found!=projection_sat_certificates_.end())
      return found->second;
    if(projection_sat_certificates_.size()>=2048 || projection_sat_requests_>=256)return false;
    ++projection_sat_requests_;
    const auto finish=[&](bool answer) {
      projection_sat_certificates_.emplace(key,answer);return answer;
    };
    if(!projectionCharge() || !projectedOperation(exponential,"Exp2",1))return finish(false);
    const auto argument=nodes_[exponential].children[0];
    if(!projectedOperation(argument,"Add",2))return finish(false);
    std::optional<Class> score,maximum,coefficient,offset;
    const auto op=[&](Class id,std::string_view name,std::size_t arity) {
      return projectedOperation(id,name,arity);
    };
    // COMPLETE bounded local forms (p-c*M)+c*S and p+c*(S-M), sharing the
    // same positive finite coefficient. No descendant search/distribution.
    for(unsigned side=0;side<2 && !score;++side) {
      const auto a=nodes_[argument].children[side],b=nodes_[argument].children[1-side];
      if(auto difference=localDifference(a,nodes_,op); difference && projectedOperation(b,"Multiply",2)) {
        const auto p=difference->first,product=difference->second;
        if(!projectedOperation(product,"Multiply",2))continue;
        for(unsigned x=0;x<2 && !score;++x)for(unsigned y=0;y<2 && !score;++y) {
          const auto c=nodes_[product].children[x],d=nodes_[b].children[y];
          if(c==d && projectionFinite(c) && positive_[c]) {
            offset=p;coefficient=c;maximum=nodes_[product].children[1-x];score=nodes_[b].children[1-y];
          }
        }
      } else if(projectedOperation(b,"Multiply",2)) {
        for(unsigned x=0;x<2 && !score;++x) {
          const auto c=nodes_[b].children[x],difference=nodes_[b].children[1-x];
          const auto parts=localDifference(difference,nodes_,op);
          if(!projectionFinite(c) || !positive_[c] || !parts)continue;
          offset=a;coefficient=c;score=parts->first;maximum=parts->second;
        }
      }
    }
    if(!score || !maximum || !coefficient || !offset)return finish(false);
    const auto& p=nodes_[*offset];double bound=0;
    if(p.name.size()>128)return finish(false);
    const auto parsed=floatFromChars(p.name.data(),p.name.data()+p.name.size(),bound);
    if(p.kind!=static_cast<int>(s::Kind::Constant) || p.type!="real" ||
        !p.attributes.empty() || !p.provenance.empty() || parsed.ec!=std::errc{} ||
        parsed.ptr!=p.name.data()+p.name.size() || !std::isfinite(bound) || bound>8)return finish(false);
    if(idealSelect(nodes_[*maximum])) {
      const auto& select=nodes_[*maximum];const auto& pred=nodes_[select.children[0]];
      const auto cmp=comparison(pred);
      if(!cmp || *cmp!="ne" || !zero_[select.children[2]] ||
          pred.children.size()!=2 || pred.children[0]!=select.children[1] ||
          !negative_infinity_[pred.children[1]])return finish(false);
      maximum=select.children[1];
    }
    if(!projectionFinite(*maximum) || !projectionFinite(*score))return finish(false);
    // Numeric ONLY: S<=M,c>0,p<=8 imply ideal Exp2<=256<448. This is NOT an
    // IEEE RN/FTZ/approx theorem or a bit alias for quantized/NaN exact values.
    // PTX RN/SAT/NaN storage semantics remain on the untouched Exact view:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt
    return finish(projectedMaxContains(*maximum,*score,scope));
  }

  static bool scopedIndexRemainderHead(const Key& k,const std::vector<Key>& arena) {
    if(k.kind!=static_cast<int>(s::Kind::Operation) || k.name!="Remainder" ||
        (k.type!="u64" && k.type!="exact:u64") || !k.attributes.empty() ||
        !k.provenance.empty() || k.fold || k.children.size()!=2)return false;
    const auto& index=arena[k.children[0]];const auto& divisor=arena[k.children[1]];
    return index.kind==static_cast<int>(s::Kind::Index) && index.type==k.type &&
        index.attributes.empty() && index.provenance.empty() && index.children.empty() && !index.fold &&
        divisor.kind==static_cast<int>(s::Kind::Constant) && divisor.type==k.type &&
        divisor.attributes.empty() && divisor.provenance.empty() && divisor.children.empty() &&
        !divisor.fold && divisor.name.size()<=32;
  }
  bool scopedIndexRemainder(const Key& source,ScopeId scope) {
    if(!projection_scopes_ || scope==lexical_scope::unknown ||
        !scopedIndexRemainderHead(source,nodes_) ||
        !projectionCharge(8+nodes_[source.children[1]].name.size()))return false;
    const auto& index=nodes_[source.children[0]];
    const auto count=projection_scopes_->count(scope,index.index_level);
    if(!count || !*count)return false;
    const auto range=integer_interval::externalIndex(source.type,*count);
    const auto divisor=integer_interval::literal(source.type,nodes_[source.children[1]].name);
    if(!range || !divisor || divisor->lower!=divisor->upper || divisor->lower<=0 ||
        range->upper>=divisor->lower)return false;
    // Exact LOCAL use identity, not an e-class union or a global binder fact.
    // The only bound is this actual output/Fold BODY scope. Empty, unknown,
    // out-of-range and values outside the shared interval model are denied.
    return integer_interval::remainderByPositiveLiteral(*range,divisor->lower)==range;
  }

  Class project(Class root,bool numeric) {
    // Predicate ancestry is a third memoized use context, not a repeated
    // ancestor search. Existing numerical comparisons retain their vocabulary;
    // newly admitted approximate trig/log instructions stay exact beneath
    // predicates, even when their producer also has ordinary numeric users.
    enum class Context {Exact,Numeric,NumericPredicate};
    if(numeric && (projection_sat_sensitive_[root] || projection_integer_sensitive_[root] ||
        (options_.certified_branch_scope && projection_branch_sensitive_[root])))initializeProjectionScope();
    const auto projection=[&](Context context)->ProjectionMap& {
      return context==Context::Exact?exact_projection_:
          context==Context::Numeric?numeric_projection_:predicate_projection_;
    };
    const auto keyFor=[&](Class id,Context context,ScopeId scope) {
      const bool scoped=projection_integer_sensitive_[id] ||
          (context!=Context::Exact && options_.certified_branch_scope && projection_branch_sensitive_[id]);
      return ProjectionKey{id,scoped || (context==Context::Numeric && projection_sat_sensitive_[id])?scope:0};
    };
    const auto projected=[&](Class id,Context context,ScopeId scope) {
      return projection(context).at(keyFor(id,context,scope));
    };
    struct Task {Class id;Context context;ScopeId scope;bool ready;};
    const auto root_context=numeric?Context::Numeric:Context::Exact;
    // Exact ROOTS never acquire a Numeric consumer's lexical assumptions.
    // Exact address/mask children below Numeric retain the enclosing scope,
    // with full scope keys on the sensitive integer node AND its ancestors.
    const auto root_scope=numeric?projection_root_scope_:lexical_scope::unknown;
    std::vector<Task> work{{root,root_context,root_scope,false}};
    while(!work.empty()) {
      poll();const auto task=work.back();work.pop_back();
      auto& cache=projection(task.context);
      const auto cache_key=keyFor(task.id,task.context,task.scope);
      if(cache.contains(cache_key))continue;
      const auto source=nodes_[task.id];
      const bool numerical=task.context!=Context::Exact;
      const bool boolean_conversion=numerical && options_.certified_branch_scope && predicateFloatConversion(source,nodes_);
      const bool conjunction=numerical && options_.certified_branch_scope && predicateConjunction(source,nodes_);
      const bool boolean_constant=numerical && options_.certified_branch_scope &&
          source.kind==static_cast<int>(s::Kind::Constant) && source.type=="pred" &&
          source.attributes.empty() && source.provenance.empty() &&
          (source.name=="0" || source.name=="1" || source.name=="true" || source.name=="false");
      const bool sat_candidate=task.context==Context::Numeric &&
          task.scope!=lexical_scope::unknown && scopedSatHead(source,nodes_) &&
          scopedExponentialSource(source.children[0]);
      bool unwrap=numerical && source.kind==6 && source.provenance.empty() &&
          source.children.size()==1 &&
          idealNumericConversion(nodes_[source.children[0]].type,source.type,source.name,true);
      if(task.ready && sat_candidate)unwrap=projectedInactiveSaturation(task.id,
          projected(source.children[0],Context::Numeric,task.scope),task.scope);
      const bool predicate_approximate=task.context==Context::NumericPredicate &&
          (source.name=="PtxSin" || source.name=="PtxCos" || source.name=="PtxLog2");
      const auto legacy_primitive=!predicate_approximate && source.children.size()==1?
          legacyIdealPrimitive(source,nodes_[source.children[0]].type):std::nullopt;
      const bool math=numerical &&
          (boolean_conversion || conjunction || boolean_constant ||
           idealOperation(source,true) || legacy_primitive || idealSelect(source) || comparison(source) ||
           (source.kind==static_cast<int>(s::Kind::Fold) && realType(source.type)));
      const bool leaf=numerical && realType(source.type) &&
          (source.kind==static_cast<int>(s::Kind::InitialCell) ||
           source.kind==static_cast<int>(s::Kind::Parameter) ||
           source.kind==static_cast<int>(s::Kind::Constant));
      const bool opaque=numerical && !unwrap && !math && !leaf;
      const auto childContext=[&](std::size_t i) {
        if(!math)return Context::Exact;
        if(boolean_conversion || conjunction)return task.context;
        if(comparison(source))return realType(nodes_[source.children[i]].type)?
            Context::NumericPredicate:Context::Exact;
        if(source.name=="Select" && i==0)return comparison(nodes_[source.children[i]])?
            Context::NumericPredicate:Context::Exact;
        return task.context;
      };
      const auto childScope=[&](std::size_t i) {
        if(task.scope!=lexical_scope::unknown && source.kind==static_cast<int>(s::Kind::Fold) &&
            source.fold && i==1 && projection_scopes_ &&
            (projection_sat_sensitive_[source.children[i]] || projection_branch_sensitive_[source.children[i]] ||
             projection_integer_sensitive_[source.children[i]]))
          return projection_scopes_->bindBody(task.scope,source.fold->binder_level,
              source.fold->count).value_or(lexical_scope::unknown);
        return task.scope;
      };
      if(!task.ready) {
        work.push_back({task.id,task.context,task.scope,true});
        if(sat_candidate) {
          // One ideal child for the local proof plus the unchanged exact fallback.
          // No temporary SAT alias escapes to another use/context.
          work.push_back({task.id,Context::Exact,task.scope,false});
          work.push_back({source.children[0],Context::Numeric,task.scope,false});
        } else if(opaque)work.push_back({task.id,Context::Exact,task.scope,false});
        else if(unwrap)work.push_back({source.children[0],task.context,task.scope,false});
        else for(std::size_t i=0;i<source.children.size();++i)
          work.push_back({source.children[i],childContext(i),childScope(i),false});
        continue;
      }
      Class value;
      if(task.context==Context::Exact && scopedIndexRemainder(source,task.scope)) {
        value=projected(source.children[0],Context::Exact,task.scope);
      } else if(boolean_conversion) {
        Key one;one.kind=static_cast<int>(s::Kind::Constant);one.type="real";one.name="1";
        auto zero=one;zero.name="0";
        Key select;select.kind=static_cast<int>(s::Kind::Operation);select.type="real";select.name="Select";
        select.children={projected(source.children[0],task.context,task.scope),intern(std::move(one)),intern(std::move(zero))};
        value=intern(std::move(select));
      } else if(unwrap)value=projected(source.children[0],task.context,task.scope);
      else if(opaque) {
        Key k;k.kind=7;k.type="real";k.name="ExactNumericValue";
        k.attributes="encoding="+source.type;
        k.children={projected(task.id,Context::Exact,task.scope)};
        value=intern(std::move(k));
      } else {
        auto k=source;k.children.clear();
        for(std::size_t i=0;i<source.children.size();++i)
          k.children.push_back(projected(source.children[i],childContext(i),childScope(i)));
        if(numerical) {
          if(source.kind==static_cast<int>(s::Kind::InitialCell) ||
              source.kind==static_cast<int>(s::Kind::Parameter))
            k.attributes="encoding="+source.type+";"+source.attributes;
          k.type=comparison(source) || conjunction || boolean_constant?"pred":"real";
          if(conjunction){k.kind=7;k.name="ScopedPredicateAnd";k.attributes.clear();}
          if(source.kind==static_cast<int>(s::Kind::Constant) && !boolean_constant) {
            if(auto decoded=constantValue(source.type,source.name))k.name=decoded->literal;
            else {
              const auto infinity=source.provenance.empty() && source.attributes.empty()?
                  floatingInfinity(source.type,source.name):std::nullopt;
              if(infinity)k.name=*infinity?"-inf":"inf";
              // Undecodable constants must retain their declared encoding. A
              // finite-only format's invalid "inf" literal is not real infinity
              // merely because projection erased its result type. This also
              // keeps malformed widths and NaN payloads typed and opaque.
              else if(source.type!="real" && source.attributes.empty())k.attributes="encoding="+source.type;
            }
          }
        } else k.type="exact:"+source.type;
        if(numerical)concatenateProjectedFold(k);
        if(math && source.kind==static_cast<int>(s::Kind::Operation)) {
          if(legacy_primitive)k.name=*legacy_primitive;
          else if(k.name=="Rsqrt")k.name="ReciprocalSqrt";
          if(k.name=="Fma") {
            auto product=k;product.name="Multiply";product.attributes="ideal_arithmetic";
            product.children.resize(2);
            k.name="Add";k.children={intern(std::move(product)),k.children[2]};
          }
          if(!comparison(k) && k.name!="Select" && !conjunction)k.attributes="ideal_arithmetic";
        }
        value=intern(std::move(k));
        if(numerical && branch_fact_anchor_ && branchPredicateSource(source,nodes_))
          if(const auto scope=branchScope(task.scope)) {
            Key wrapper;wrapper.kind=7;wrapper.type="pred";wrapper.name="ScopedPositivePredicate";
            wrapper.attributes=std::to_string(*scope)+":"+nodes_[source.children[0]].type;
            wrapper.children={value,*branch_fact_anchor_};value=intern(std::move(wrapper));
          }
      }
      if(cache.size()>=options_.max_nodes)throw Limit{"projection_memo_budget"};
      cache.emplace(cache_key,value);
    }
    return projected(root,root_context,root_scope);
  }
  ProjectionMap exact_projection_,numeric_projection_,predicate_projection_;
  void poll() const {
    if(options_.max_work && ++work_>options_.max_work)throw Limit{"work_budget"};
    if (std::chrono::steady_clock::now() - started_ >= options_.timeout)
      throw Limit{"time_budget"};
  }
  Class find(Class id) {
    while (parents_[id] != id) {
      parents_[id] = parents_[parents_[id]];
      id = parents_[id];
    }
    return id;
  }
  std::pair<std::int64_t,bool> scopeProperties(const Key& k) {
    std::int64_t maximum=k.kind==static_cast<int>(s::Kind::Index)?static_cast<std::int64_t>(k.index_level):-1;
    bool fold=k.kind==static_cast<int>(s::Kind::Fold);
    for(std::size_t i=0;i<k.children.size();++i) {
      if(branchWrapper(k) && i==1)continue; // Proof anchor is not a semantic operand.
      auto child=k.children[i];
      child=find(child);maximum=std::max(maximum,max_index_[child]);fold=fold || has_fold_[child];
    }
    return {maximum,fold};
  }
  std::pair<bool,bool> positivity(const Key& k) {
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
  bool provesFinite(const Key& k) {
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
  std::uint8_t provesPredicate(const Key& k,Class id) {
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
  integer_interval::Result integerRange(const Key& k) {
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
  void rewriteUnsignedShift(Class id,const Key& k) {
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
  void rewriteExactIntegerNumeric(Class id,const Key& k,
      const std::vector<std::vector<Class>>& members) {
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
  std::optional<Key> singletonConstant(const Key& k) {
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
  bool refineIntegerRange(Class root,const integer_interval::Result& proof) {
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
  bool provesZero(const Key& k) {
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
  bool provesNonzero(const Key& k,bool positive) {
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
  bool updateProperties(Class id) {
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
  void canonicalizeCommutative(Key& key) {
    const bool integer_operation=integerCommutative(key) &&
        nodes_[find(key.children[0])].type==key.type && nodes_[find(key.children[1])].type==key.type;
    const bool real_operation=arithmetic(key) && finite_[find(key.children[0])] && finite_[find(key.children[1])];
    if((integer_operation || real_operation) && minimum_[key.children[1]]<minimum_[key.children[0]])
      std::swap(key.children[0],key.children[1]);
  }
  std::optional<Class> exactIntegerIdentity(const Key& k) const {
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
  Class intern(Key k) {
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
  bool unite(Class a, Class b) {
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
  void rewrite(Class a, Class b) {
    if (unite(a,b)) ++statistics_.rewrite_unions;
  }
  void rebuild() {
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

struct FrozenUpdateEgraph::Impl {
  FrozenIndex index;
  Result result;
};
FrozenUpdateEgraph::FrozenUpdateEgraph(const s::Graph& graph,std::span<const s::Ref> roots,
    Options options,std::span<const std::string> positives):impl_(std::make_unique<Impl>()) {
  Engine engine(options,positives);
  std::vector<Class> imported_roots;
  if(!options.max_nodes || !options.max_work || options.timeout.count()<=0 || roots.empty()) {
    impl_->result=engine.result(Status::Unsupported,"invalid_update_budget");return;
  }
  try {
    imported_roots=engine.import(graph,roots);
    s::Graph other;auto sentinel=other.parameter(graph.type(roots[0]),"$update_never_equal");
    auto y=engine.import(other,std::span(&sentinel,1));
    // Full bounded saturation, not an equality query that could return early.
    std::vector<Class> unmatched(imported_roots.size(),y[0]);
    impl_->result=engine.compare(imported_roots,unmatched,true);
  } catch(const Limit& limit) {
    impl_->result=engine.result(Status::NotProven,limit.reason);
  } catch(const std::invalid_argument& error) {
    impl_->result=engine.result(Status::Unsupported,error.what());
  }
  // Even a budget stop retains only proven unions. Queries can miss, never
  // infer inequivalence or introduce a new union from an unproved candidate.
  impl_->index=engine.freeze(imported_roots);
}
FrozenUpdateEgraph::~FrozenUpdateEgraph()=default;
FrozenUpdateEgraph::FrozenUpdateEgraph(FrozenUpdateEgraph&&) noexcept=default;
FrozenUpdateEgraph& FrozenUpdateEgraph::operator=(FrozenUpdateEgraph&&) noexcept=default;
const Result& FrozenUpdateEgraph::compilation() const{return impl_->result;}
FrozenUpdateEgraph::Lookup FrozenUpdateEgraph::importedRoot(std::size_t index) const {
  if(index>=impl_->index.imported_roots.size())return {};
  return {impl_->index.imported_roots[index],1};
}
FrozenUpdateEgraph::Lookup FrozenUpdateEgraph::lookup(const s::Graph& graph,s::Ref root,std::size_t cap) const {
  Lookup result;
  std::unordered_map<s::Id,Class> known;
  std::vector<s::Ref> pending{root};
  while(!pending.empty()) {
    if(++result.work>cap)return result;
    const auto value=pending.back();
    if(!value.conversion.empty() || !value.pointer_provenance.empty())return result;
    if(known.contains(value.node)){pending.pop_back();continue;}
    const auto& n=graph.node(value);
    if(n.kind!=s::Kind::Parameter && n.kind!=s::Kind::Constant && n.kind!=s::Kind::Operation)return result;
    bool ready=true;
    for(const auto& child:n.operands) {
      if(!child.conversion.empty() || !child.pointer_provenance.empty())return result;
      if(!known.contains(child.node)) {pending.push_back(child);ready=false;}
    }
    if(!ready)continue;
    Key key{static_cast<int>(n.kind),n.type,n.name,n.attributes,{},n.fold,n.index_level,{}};
    for(const auto& child:n.operands)key.children.push_back(known.at(child.node));
    if(idealOperation(key) || idealSelect(key))key.attributes="ideal_arithmetic";
    if(arithmetic(key) && impl_->index.finite[key.children[0]] && impl_->index.finite[key.children[1]] &&
        key.children[1]<key.children[0])std::swap(key.children[0],key.children[1]);
    const auto found=impl_->index.memo.find(key);
    if(found==impl_->index.memo.end())return result;
    known.emplace(value.node,found->second);pending.pop_back();
  }
  result.eclass=known.at(root.node);return result;
}

scalar_fold::Ref normalizeExactUse(const scalar_fold::Graph& graph,scalar_fold::Ref ref) {
  for(;;) {
    if(!ref.conversion.empty()) {
      if(auto inactive=normalizeInactiveFiniteSaturation(graph,ref))ref=std::move(*inactive);
      if(auto normalized=normalizeConversions(graph.node(ref).type,graph.type(ref),ref.conversion)) {
        ref.conversion=*normalized;
        if(ref.conversion.empty())ref.converted_type.clear();
      }
    }
    auto reduced=packedLowUse(graph,ref);
    if(!reduced)reduced=packedHighUse(graph,ref);
    if(!reduced)return ref;
    ref=std::move(*reduced);
  }
}

scalar_fold::Ref normalizeExactUse(scalar_fold::Graph& graph,scalar_fold::Ref ref) {
  // Validate bounded metadata before the const path can canonicalize it.
  // In particular, a dangling delimiter must not become a valid field tail.
  if(!ref.conversion.empty() && ref.conversion.size()<=128 && ref.converted_type.size()<=128 &&
      !completeBoundedConversionUse(graph.node(ref).type,ref))return ref;
  ref=normalizeExactUse(static_cast<const scalar_fold::Graph&>(graph),std::move(ref));
  if(auto field=packedShiftedFieldUse(graph,ref))
    return normalizeExactUse(static_cast<const scalar_fold::Graph&>(graph),std::move(*field));
  return ref;
}

Result compareExpressionSets(const s::Graph& left, std::span<const s::Ref> a,
                             const s::Graph& right, std::span<const s::Ref> b, Options options) {
  Engine engine(options);
  if (!options.max_nodes || options.timeout.count() <= 0)
    return engine.result(Status::Unsupported, "invalid_budget");
  if(options.outer_index_extents.size()>options.outer_binders ||
      std::any_of(options.outer_index_extents.begin(),options.outer_index_extents.end(),[](auto n){return !n;}))
    return engine.result(Status::Unsupported,"invalid_outer_index_domain");
  if (a.size()!=b.size()) return engine.result(Status::Unsupported, "output_mapping_size_mismatch");
  if (a.empty()) return engine.result(Status::Unsupported, "empty_output_mapping");
  try {
    std::vector<unsigned char> storage_a,storage_b;
    const auto x=engine.import(left,a,options.storage_output_bytes?&storage_a:nullptr);
    const auto y=engine.import(right,b,options.storage_output_bytes?&storage_b:nullptr);
    for (std::size_t i=0; i<a.size(); ++i)
      if (!options.ideal_numeric_outputs && !options.storage_output_bytes &&
          left.type(a[i]) != right.type(b[i]))
        return engine.result(Status::NotProven,"different_result_types");
    return engine.compare(x,y,false,storage_a,storage_b);
  } catch (const Limit& limit) {
    return engine.result(Status::NotProven,limit.reason);
  } catch (const std::invalid_argument& error) {
    return engine.result(Status::Unsupported,error.what());
  } catch (const std::out_of_range& error) {
    return engine.result(Status::Unsupported,error.what());
  }
}

Result compareExpressions(const s::Graph& left, s::Ref a,
                          const s::Graph& right, s::Ref b, Options options) {
  return compareExpressionSets(left,std::span(&a,1),right,std::span(&b,1),options);
}

ScopedBodyResult proveScopedBodyRelation(const s::Graph& reference,const ScopedBodyPath& reference_path,
    const s::Graph& actual,const ScopedBodyPath& actual_path,
    std::span<const ScopedAxisBinding> bindings,Options options) {
  ScopedBodyResult result;
  if(!options.max_nodes || options.timeout.count()<=0 || bindings.size()>2 ||
      !options.ideal_numeric_outputs || options.outer_index_extents.size()!=options.outer_binders ||
      std::any_of(options.outer_index_extents.begin(),options.outer_index_extents.end(),[](auto n){return !n;})) {
    result.reason="invalid_scoped_configuration";return result;
  }
  if(!boundedScopedRef(reference_path.wrapper) || !boundedScopedRef(actual_path.wrapper) ||
      std::any_of(bindings.begin(),bindings.end(),[&](const auto& b){
        return !boundedScopedRef(b.coordinate) || (b.image_coordinate && !boundedScopedRef(*b.image_coordinate));})) {
    result.reason="scoped_metadata_cap";return result;
  }
  try {
    // Actual graph-root validation in import() precedes all domain minting.
    const auto r=extractScopedBody(reference,reference_path),a=extractScopedBody(actual,actual_path);
    std::vector<s::Ref> reference_witnesses{r.body},actual_witnesses{a.body};
    std::array<std::pair<bool,std::size_t>,2> locations{};
    std::array<std::optional<std::size_t>,2> image_locations{};
    for(std::size_t i=0;i<bindings.size();++i) {
      auto& witnesses=bindings[i].actual_shift?reference_witnesses:actual_witnesses;
      locations[i]={bindings[i].actual_shift,witnesses.size()};witnesses.push_back(bindings[i].coordinate);
      if(bindings[i].image_coordinate) {
        image_locations[i]=reference_witnesses.size();reference_witnesses.push_back(*bindings[i].image_coordinate);
      }
    }
    Engine engine(options);std::vector<Class> rw,aw;
    const auto rr=engine.import(reference,std::span(&reference_path.wrapper,1),nullptr,reference_witnesses,&rw);
    const auto ar=engine.import(actual,std::span(&actual_path.wrapper,1),nullptr,actual_witnesses,&aw);
    std::array<Class,2> coordinates{};
    std::array<std::optional<Class>,2> image_coordinates{};
    for(std::size_t i=0;i<bindings.size();++i)
      coordinates[i]=locations[i].first?rw[locations[i].second]:aw[locations[i].second];
    for(std::size_t i=0;i<bindings.size();++i)
      if(image_locations[i])image_coordinates[i]=rw[*image_locations[i]];
    return engine.scopedBodyProof(rr.front(),ar.front(),rw.front(),aw.front(),reference,r.folds,actual,a.folds,
        bindings,std::span(coordinates.data(),bindings.size()),std::span(image_coordinates.data(),bindings.size()));
  } catch(const Limit& limit) {result.reason=limit.reason;}
    catch(const std::invalid_argument& error) {result.reason=error.what();}
    catch(const std::out_of_range& error) {result.reason=error.what();}
  return result;
}

JoinedBodyResult proveJoinedBodyRelation(const s::Graph& reference,const ScopedBodyPath& reference_path,
    std::span<const std::uint32_t> reference_axes,const s::Graph& actual,
    std::span<const JoinedBodyPiece> pieces,Options options) {
  JoinedBodyResult result;
  if(!options.max_nodes || options.timeout.count()<=0 || !options.ideal_numeric_outputs ||
      options.outer_index_extents.size()!=options.outer_binders || reference_axes.empty() ||
      reference_axes.size()>2 || pieces.empty() || pieces.size()>=1024 ||
      std::any_of(options.outer_index_extents.begin(),options.outer_index_extents.end(),[](auto n){return !n;})) {
    result.reason="invalid_join_configuration";return result;
  }
  if(!boundedScopedRef(reference_path.wrapper)) {result.reason="scoped_metadata_cap";return result;}
  try {
    auto target=extractScopedBody(reference,reference_path);
    std::vector<s::Ref> actual_roots,actual_witnesses;
    std::vector<std::vector<s::Id>> folds;
    std::vector<std::array<std::size_t,3>> locations;
    actual_roots.reserve(pieces.size());folds.reserve(pieces.size());locations.reserve(pieces.size());
    for(const auto& piece:pieces) {
      if(piece.axis_count!=reference_axes.size() || !boundedScopedRef(piece.body.wrapper))
        throw std::invalid_argument("invalid_join_piece");
      auto use=extractScopedBody(actual,piece.body);std::array<std::size_t,3> location{};
      location[0]=actual_witnesses.size();actual_witnesses.push_back(use.body);
      for(unsigned axis=0;axis<piece.axis_count;++axis) {
        if(!boundedScopedRef(piece.axes[axis].coordinate))throw std::invalid_argument("scoped_metadata_cap");
        location[axis+1]=actual_witnesses.size();actual_witnesses.push_back(piece.axes[axis].coordinate);
      }
      actual_roots.push_back(piece.body.wrapper);folds.push_back(std::move(use.folds));locations.push_back(location);
    }
    Engine engine(options);std::vector<Class> rw,aw;
    const std::array<s::Ref,1> reference_witnesses{target.body};
    auto rr=engine.import(reference,std::span(&reference_path.wrapper,1),nullptr,reference_witnesses,&rw);
    auto ar=engine.import(actual,actual_roots,nullptr,actual_witnesses,&aw);
    std::vector<Class> bodies;std::vector<std::array<Class,2>> coordinates;
    bodies.reserve(pieces.size());coordinates.reserve(pieces.size());
    for(std::size_t i=0;i<pieces.size();++i) {
      bodies.push_back(aw[locations[i][0]]);std::array<Class,2> c{};
      for(unsigned axis=0;axis<pieces[i].axis_count;++axis)c[axis]=aw[locations[i][axis+1]];
      coordinates.push_back(c);
    }
    return engine.joinedBodyProof(rr.front(),rw.front(),reference,target.folds,reference_axes,
        actual,ar,bodies,folds,coordinates,pieces);
  } catch(const Limit& limit) {result.reason=limit.reason;}
    catch(const std::invalid_argument& error) {result.reason=error.what();}
    catch(const std::out_of_range& error) {result.reason=error.what();}
  return result;
}

ScopedProductResult proveScopedProductRelation(const s::Graph& reference,
    std::span<const ScopedBodyPath> reference_paths,const s::Graph& actual,
    std::span<const ScopedBodyPath> actual_paths,std::span<const ScopedAxisBinding> bindings,Options options) {
  ScopedProductResult result;
  if(!options.max_nodes || options.timeout.count()<=0 || !options.ideal_numeric_outputs ||
      options.outer_index_extents.size()!=options.outer_binders || reference_paths.empty() ||
      actual_paths.empty() || reference_paths.size()>8 || actual_paths.size()>8 || bindings.size()>2 ||
      std::any_of(options.outer_index_extents.begin(),options.outer_index_extents.end(),[](auto n){return !n;})) {
    result.reason="invalid_product_configuration";return result;
  }
  try {
    struct Input {
      std::vector<s::Ref> roots,witnesses;std::vector<std::vector<s::Id>> folds;
      std::vector<std::size_t> body_locations;
    };
    const auto input=[&](const s::Graph& graph,std::span<const ScopedBodyPath> paths) {
      Input result;
      for(const auto& path:paths) {
        if(!boundedScopedRef(path.wrapper))throw std::invalid_argument("scoped_metadata_cap");
        auto use=extractScopedBody(graph,path);result.roots.push_back(path.wrapper);
        result.body_locations.push_back(result.witnesses.size());result.witnesses.push_back(use.body);
        result.folds.push_back(std::move(use.folds));
      }
      return result;
    };
    auto r=input(reference,reference_paths),a=input(actual,actual_paths);
    std::array<std::pair<bool,std::size_t>,2> locations{};
    std::array<std::optional<std::size_t>,2> image_locations{};
    for(std::size_t i=0;i<bindings.size();++i) {
      const auto& b=bindings[i];
      if(!boundedScopedRef(b.coordinate) || (b.image_coordinate && !boundedScopedRef(*b.image_coordinate)))
        throw std::invalid_argument("scoped_metadata_cap");
      auto& witnesses=b.actual_shift?r.witnesses:a.witnesses;
      locations[i]={b.actual_shift,witnesses.size()};witnesses.push_back(b.coordinate);
      if(b.image_coordinate){image_locations[i]=r.witnesses.size();r.witnesses.push_back(*b.image_coordinate);}
    }
    Engine engine(options);std::vector<Class> rw,aw;
    auto rr=engine.import(reference,r.roots,nullptr,r.witnesses,&rw);
    auto ar=engine.import(actual,a.roots,nullptr,a.witnesses,&aw);
    std::vector<Class> rb,ab;
    for(auto location:r.body_locations)rb.push_back(rw[location]);
    for(auto location:a.body_locations)ab.push_back(aw[location]);
    std::array<Class,2> coordinates{};std::array<std::optional<Class>,2> images{};
    for(std::size_t i=0;i<bindings.size();++i) {
      coordinates[i]=locations[i].first?rw[locations[i].second]:aw[locations[i].second];
      if(image_locations[i])images[i]=rw[*image_locations[i]];
    }
    return engine.productBodyProof(reference,rr,rb,r.folds,actual,ar,ab,a.folds,bindings,
        std::span(coordinates.data(),bindings.size()),std::span(images.data(),bindings.size()));
  } catch(const Limit& limit) {result.reason=limit.reason;}
    catch(const std::invalid_argument& error) {result.reason=error.what();}
    catch(const std::out_of_range& error) {result.reason=error.what();}
  return result;
}
}  // namespace sentryward::equivalence
