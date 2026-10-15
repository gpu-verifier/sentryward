#include "sentryward/equivalence/math_intrinsics.h"
#include "sentryward/equivalence/float_from_chars.h"
#include "sentryward/equivalence/constant_values.h"
#include "sentryward/equivalence/conversion_uses.h"
#include "sentryward/equivalence/numeric_semantics.h"
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace sentryward::equivalence {
namespace {
namespace s = scalar_fold;

bool floating(const std::string& type) {
  return type == "real" || exactFloatingWidening(type,type);
}
bool bare(const s::Ref& ref) {
  return ref.conversion.empty() && ref.converted_type.empty() &&
      ref.pointer_provenance.empty();
}

// Empty/ideal attributes are the already-lowered cached mathematical shape.
// Explicit source rounding, when still present, must match the complete idiom.
bool arithmeticAttributes(const std::string& attributes,
                          std::string_view expected_rounding = "nearest-even",
                          bool lowered_only = false) {
  bool ideal = false, rounding = false;
  std::istringstream words(attributes);std::string token;
  while (words >> token) {
    if (token == "ideal_arithmetic") {
      if (ideal) return false;
      ideal = true;
    } else if (!lowered_only && token == "rounding=" + std::string(expected_rounding)) {
      if (rounding) return false;
      rounding = true;
    } else return false;
  }
  return !(ideal && rounding);
}
bool idealExp2Attributes(const std::string& attributes) {
  return attributes.empty() || attributes == "ideal_arithmetic" ||
      attributes == "semantics=ideal";
}
// The direct fast-exp multiply and its ordinary numeric consumers may retain
// RN/FTZ source modifiers. Unsupported modes/controls remain exact. This does
// not relax the stricter range-reduction matcher's own rounding certificates.
bool numericArithmeticAttributes(const std::string& attributes) {
  bool ideal=false,rounding=false,ftz=false;
  std::istringstream words(attributes);std::string token;
  while(words>>token) {
    if(token=="ideal_arithmetic") {if(ideal)return false;ideal=true;}
    else if(token=="rounding=nearest-even") {if(rounding)return false;rounding=true;}
    else if(token=="flush_to_zero") {if(ftz)return false;ftz=true;}
    else return false;
  }
  return !(ideal && (rounding || ftz));
}

class MathIntrinsicMatcher {
 public:
  MathIntrinsicMatcher(const s::Graph& graph,std::size_t cap)
      :graph_(graph),cap_(std::min<std::size_t>(cap,128)) {}
  std::size_t inspected() const { return inspected_; }
  std::optional<NaturalExpIntrinsicMatch> match(s::Ref root) {
    const auto* result = operation(root,"Multiply",2);
    bool fused = false;
    if (!result) { result = operation(root,"Fma",3);fused = true; }
    if (!result || !arithmeticAttributes(result->attributes)) return {};
    if (fused && !hasType(result->operands[2],"f32")) return {};
    for (unsigned side=0;side<2;++side) {
      auto argument = product(result->operands[side],result->operands[1-side]);
      if (argument) return NaturalExpIntrinsicMatch{*argument,
          fused ? std::optional<s::Ref>(result->operands[2]) : std::nullopt};
    }
    return {};
  }
  std::optional<NaturalExpIntrinsicMatch> matchFast(s::Ref root) {
    const auto* exponential=operation(root,"Exp2",1);
    if(!exponential || !idealExp2Attributes(exponential->attributes))return {};
    struct SignedBinding {s::Ref value;bool negative=false;};
    std::array<SignedBinding,4> pending{};
    std::size_t pending_count=1,spine=0,scaled=0;
    pending[0]={exponential->operands[0],false};
    NaturalExpIntrinsicMatch result;result.fast_direct=true;
    const auto binding=[&](const s::Ref& value) {
      if(!numericBinding(value))return false;
      const auto* source=node(value);
      return source && (source->kind!=s::Kind::Constant || finiteConstant(value,*source));
    };
    while(pending_count) {
      const auto term=pending[--pending_count];
      const auto* source=node(term.value);
      if(!source)return {};
      if(bare(term.value) && source->kind==s::Kind::Operation &&
          (source->name=="Add" || source->name=="Subtract" || source->name=="Negate")) {
        const auto arity=source->name=="Negate"?1u:2u;
        if(source->type!="f32" || source->operands.size()!=arity ||
            !numericArithmeticAttributes(source->attributes) || ++spine>3 ||
            pending_count+arity>pending.size())return {};
        // Reverse pushes preserve source operand order. Only this fixed-size
        // spine is opened; scaled arguments and the residual are never walked.
        for(unsigned i=arity;i-->0;)pending[pending_count++]={source->operands[i],
            term.negative != (source->name=="Negate" || (source->name=="Subtract" && i==1))};
        continue;
      }
      const auto* product=operation(term.value,"Multiply",2);
      if(bare(term.value) && source->kind==s::Kind::Operation && source->name=="Multiply" &&
          !product)return {};
      if(product) {
        if(!numericArithmeticAttributes(product->attributes))return {};
        bool matched=false;
        for(unsigned side=0;side<2;++side) {
          const bool positive=literal(product->operands[side],0x3fb8aa3b);
          const bool negative=!positive && literal(product->operands[side],0xbfb8aa3b);
          if(!positive && !negative)continue;
          const auto& argument=product->operands[1-side];
          if(!binding(argument) || scaled==2)return {};
          if(scaled++==0) {
            result.argument=argument;result.negate_argument=term.negative ^ negative;
          } else {
            result.second_argument=argument;result.negate_second_argument=term.negative ^ negative;
          }
          matched=true;break;
        }
        if(matched)continue;
        // One additional product head accommodates ordinary reassociation of
        // a complete three-factor fast-exp argument. This is not an ancestor
        // search and does not rewrite the rounded coefficient independently.
        if(!scaled && !pending_count && !result.base_two_residual) {
          for(unsigned side=0;side<2;++side) {
            const auto* nested=operation(product->operands[side],"Multiply",2);
            if(!nested || !numericArithmeticAttributes(nested->attributes))continue;
            for(unsigned factor=0;factor<2;++factor) {
              const bool positive=literal(nested->operands[factor],0x3fb8aa3b);
              const bool negative=!positive && literal(nested->operands[factor],0xbfb8aa3b);
              if((!positive && !negative) || !binding(nested->operands[1-factor]) ||
                  !binding(product->operands[1-side]))continue;
              result.argument=nested->operands[1-factor];
              result.argument_factor=product->operands[1-side];
              result.negate_argument=term.negative ^ negative;
              return result;
            }
          }
        }
        // Do not reinterpret a damaged/wrong-width coefficient as an unscaled
        // offset. Nonliteral products may still be opaque residual bindings.
        for(const auto& factor:product->operands) {
          const auto* leaf=node(factor);
          if(!leaf || leaf->kind==s::Kind::Constant)return {};
        }
      }
      if(result.base_two_residual || !binding(term.value))return {};
      result.base_two_residual=term.value;result.negate_base_two_residual=term.negative;
    }
    if(!scaled)return {};
    return result;
  }
  std::optional<NaturalLogIntrinsicMatch> matchFastLog(s::Ref root) {
    const auto* product=operation(root,"Multiply",2);
    if(!product || !numericArithmeticAttributes(product->attributes))return {};
    for(unsigned side=0;side<2;++side) {
      const auto coefficient=product->operands[side];
      if(!literal(coefficient,0x3f317218))continue;
      const auto value=product->operands[1-side];
      const auto logarithm=[&](s::Ref ref)->std::optional<s::Ref> {
        const auto* log=operation(ref,"PtxLog2",1);
        if(!log || legacyIdealPrimitive(log->name,log->type,log->attributes,
            graph_.type(log->operands[0]))!="Log2")return {};
        const auto argument=log->operands[0];
        if(!numericBinding(argument))return {};
        const auto* source=node(argument);
        if(source->kind==s::Kind::Constant && !positiveFiniteConstant(argument,*source))return {};
        return argument;
      };
      if(auto argument=logarithm(value))return NaturalLogIntrinsicMatch{*argument,coefficient,{}};
      const auto* sum=operation(value,"Add",2);
      if(!sum || !numericArithmeticAttributes(sum->attributes))continue;
      for(unsigned term=0;term<2;++term) {
        auto argument=logarithm(sum->operands[term]);
        const auto offset=sum->operands[1-term];
        if(!argument || !numericBinding(offset))continue;
        const auto* source=node(offset);
        if(source->kind==s::Kind::Constant && !finiteConstant(offset,*source))continue;
        return NaturalLogIntrinsicMatch{*argument,coefficient,offset};
      }
    }
    return {};
  }
  std::optional<SoftwareLog2IntrinsicMatch> matchSoftwareLog2(s::Ref root) {
    const auto* outer=select(root);
    if(!outer || !literal(outer->operands[1],0xff800000))return {};
    const auto* zero_test=operation(outer->operands[0],"Compare",2,"pred");
    if(!zero_test || zero_test->attributes!="predicate=eq")return {};
    std::optional<s::Ref> scaled;
    for(unsigned side=0;side<2;++side)
      if(literal(zero_test->operands[side],0) && hasType(zero_test->operands[1-side],"f32"))
        scaled=zero_test->operands[1-side];
    if(!scaled)return {};
    const auto* exceptional=select(outer->operands[2]);
    if(!exceptional)return {};
    const auto* nonfinite=operation(exceptional->operands[0],"Compare",2,"pred");
    if(!nonfinite || nonfinite->attributes!="predicate=gt" ||
        !integerLiteral(nonfinite->operands[1],"u32",0x7f7fffff))return {};
    auto nonfinite_input=softwareCastSource(nonfinite->operands[0],"f32","u32");
    if(!nonfinite_input || *nonfinite_input!=*scaled)return {};
    auto exceptional_value=affine(exceptional->operands[1],0x7f800000);
    if(!exceptional_value || exceptional_value->argument!=*scaled ||
        !literal(exceptional_value->addend,0x7f800000))return {};
    const auto* sum=operation(exceptional->operands[2],"Add",2);
    if(!sum || !arithmeticAttributes(sum->attributes))return {};
    for(unsigned side=0;side<2;++side) {
      auto exponent=affine(sum->operands[side],0x34000000);
      auto t=softwareLogPolynomial(sum->operands[1-side]);
      if(!exponent || !t)continue;
      auto masked=roundedSignedExponentSource(exponent->argument);
      const auto* correction=select(exponent->addend);
      if(!masked || !correction || !literal(correction->operands[1],0xc1b80000) ||
          !literal(correction->operands[2],0))continue;
      const auto* rescale=select(*scaled);
      if(!rescale || rescale->operands[0]!=correction->operands[0])continue;
      const auto x=rescale->operands[2];
      if(x.conversion.size()>128 || !numericBinding(x))continue;
      const auto* source=node(x);
      if(source->kind==s::Kind::Constant && !positiveFiniteConstant(x,*source))continue;
      const auto* subnormal=operation(rescale->operands[0],"Compare",2,"pred");
      if(!subnormal || subnormal->attributes!="predicate=lt" || subnormal->operands[0]!=x ||
          !literal(subnormal->operands[1],0x00800000))continue;
      const auto* product=operation(rescale->operands[1],"Multiply",2);
      if(!product || !arithmeticAttributes(product->attributes))continue;
      bool same_x=false;
      for(unsigned factor=0;factor<2;++factor)
        same_x|=product->operands[factor]==x && literal(product->operands[1-factor],0x4b000000);
      if(!same_x || !softwareLogExtraction(*scaled,*masked,*t))continue;
      return SoftwareLog2IntrinsicMatch{x};
    }
    return {};
  }
 private:
  const s::Graph& graph_;std::size_t cap_,inspected_ = 0;
  std::array<s::Id,128> seen_{};
  const s::Node* node(const s::Ref& ref) {
    if (!ref.pointer_provenance.empty()) return nullptr;
    if (std::find(seen_.begin(),seen_.begin()+inspected_,ref.node) ==
        seen_.begin()+inspected_) {
      if (inspected_ == cap_) return nullptr;
      seen_[inspected_++] = ref.node;
    }
    return &graph_.node(ref);
  }
  bool hasType(const s::Ref& ref,std::string_view expected) {
    return node(ref) && graph_.type(ref) == expected;
  }
  const s::Node* operation(const s::Ref& ref,std::string_view name,std::size_t arity,
                          std::string_view type = "f32") {
    if (!bare(ref)) return nullptr;
    const auto* n = node(ref);
    if (!n || n->kind != s::Kind::Operation || n->name != name ||
        n->type != type || n->operands.size() != arity) return nullptr;
    for (const auto& operand : n->operands)
      if (!operand.pointer_provenance.empty()) return nullptr;
    return n;
  }
  bool literal(const s::Ref& ref,std::uint32_t bits) {
    if (!bare(ref)) return false;
    const auto* n = node(ref);
    if (!n || n->kind != s::Kind::Constant || n->type != "f32" ||
        !n->attributes.empty() || !n->operands.empty()) return false;
    const auto decoded = constantValue("f32",n->name,"Bitcast(f32 -> u32)","u32");
    if (!decoded) {
      // constantValue is exact-finite-only for decimal floating literals.
      // The exception wrapper also contains typed infinities, whose numeric
      // meaning is unambiguous. No NaN payload spelling is generalized.
      if(bits!=0x7f800000 && bits!=0xff800000)return false;
      float value=0;const auto parsed=floatFromChars(n->name.data(),n->name.data()+n->name.size(),value);
      return parsed.ec==std::errc{} && parsed.ptr==n->name.data()+n->name.size() &&
          std::isinf(value) && std::bit_cast<std::uint32_t>(value)==bits;
    }
    std::uint32_t value = 0;
    const auto parsed = std::from_chars(decoded->literal.data(),
        decoded->literal.data()+decoded->literal.size(),value);
    return parsed.ec == std::errc{} &&
        parsed.ptr == decoded->literal.data()+decoded->literal.size() && value == bits;
  }
  bool finiteConstant(const s::Ref& ref,const s::Node& n) const {
    if(!n.attributes.empty() || !n.operands.empty())return false;
    const auto value=constantValue(n.type,n.name,ref.conversion,ref.converted_type);
    if(!value || value->type!="f32")return false;
    const auto decoded=constantValue("f32",value->literal,"Bitcast(f32 -> u32)","u32");
    if(!decoded)return false;
    std::uint32_t bits=0;const auto parsed=std::from_chars(decoded->literal.data(),
        decoded->literal.data()+decoded->literal.size(),bits);
    return parsed.ec==std::errc{} && parsed.ptr==decoded->literal.data()+decoded->literal.size() &&
        (bits&0x7f800000)!=0x7f800000;
  }
  bool positiveFiniteConstant(const s::Ref& ref,const s::Node& n) const {
    if(!finiteConstant(ref,n))return false;
    const auto value=constantValue(n.type,n.name,ref.conversion,ref.converted_type);
    const auto decoded=constantValue("f32",value->literal,"Bitcast(f32 -> u32)","u32");
    if(!decoded)return false;
    std::uint32_t bits=0;const auto parsed=std::from_chars(decoded->literal.data(),
        decoded->literal.data()+decoded->literal.size(),bits);
    return parsed.ec==std::errc{} && parsed.ptr==decoded->literal.data()+decoded->literal.size() &&
        !(bits&0x80000000) && bits!=0;
  }
  bool numericBinding(const s::Ref& ref) {
    if(!hasType(ref,"f32"))return false;
    const auto* source=node(ref);
    return source && ((floating(source->type) && (ref.conversion.empty() ||
        idealNumericConversion(source->type,"f32",ref.conversion))) ||
        idealFloatingStorageReadUse(graph_,ref));
  }
  const s::Node* select(const s::Ref& ref) {
    const auto* result=operation(ref,"Select",3);
    return result && arithmeticAttributes(result->attributes,"nearest-even",true)?result:nullptr;
  }
  std::optional<s::Ref> softwareCastSource(s::Ref ref,std::string_view from,std::string_view to) {
    if(ref.conversion.size()>128)return {};
    return castSource(std::move(ref),from,to);
  }
  bool integerLiteral(const s::Ref& ref,std::string_view type,std::uint32_t bits) {
    if(!bare(ref))return false;
    const auto* n=node(ref);
    if(!n || n->kind!=s::Kind::Constant || n->type!=type || !n->attributes.empty() ||
        !n->operands.empty())return false;
    const auto decoded=constantValue(n->type,n->name,"Bitcast("+n->type+" -> u32)","u32");
    if(!decoded)return false;
    std::uint32_t value=0;const auto p=std::from_chars(decoded->literal.data(),
        decoded->literal.data()+decoded->literal.size(),value);
    return p.ec==std::errc{} && p.ptr==decoded->literal.data()+decoded->literal.size() && value==bits;
  }
  std::optional<s::Ref> roundedSignedExponentSource(const s::Ref& ref) {
    if(ref.conversion.size()>128 || !hasType(ref,"f32"))return {};
    const auto* n=node(ref);
    if(!n || n->type!="b32" || ref.conversion.empty())return {};
    const auto normalized=normalizeConversions(n->type,graph_.type(ref),ref.conversion);
    if(normalized && *normalized==
        "Bitcast(b32 -> s32); Convert rounding=nearest-even(s32 -> f32)")
      return graph_.ref(ref.node);
    return {};
  }
  std::optional<s::Ref> sharedProduct(const s::Ref& value,const s::Ref& factor) {
    const auto* product=operation(value,"Multiply",2);
    if(!product || !arithmeticAttributes(product->attributes))return {};
    for(unsigned side=0;side<2;++side)
      if(product->operands[side]==factor && hasType(product->operands[1-side],"f32"))
        return product->operands[1-side];
    return {};
  }
  std::optional<s::Ref> hornerStep(const s::Ref& value,const s::Ref& t,std::uint32_t coefficient) {
    if(const auto* fma=operation(value,"Fma",3)) {
      if(!arithmeticAttributes(fma->attributes) || !literal(fma->operands[2],coefficient))return {};
      for(unsigned side=0;side<2;++side)
        if(fma->operands[side]==t && hasType(fma->operands[1-side],"f32"))
          return fma->operands[1-side];
      return {};
    }
    const auto* sum=operation(value,"Add",2);
    if(!sum || !arithmeticAttributes(sum->attributes))return {};
    for(unsigned side=0;side<2;++side)
      if(literal(sum->operands[side],coefficient))
        if(auto next=sharedProduct(sum->operands[1-side],t))return next;
    return {};
  }
  std::optional<s::Ref> softwareLogPolynomial(const s::Ref& value) {
    constexpr std::array<std::uint32_t,11> coefficient{
      0x3fb8aa3b,0xbf38aa3b,0x3ef6384a,0xbeb8aa49,0x3e93bf99,0xbe763c8b,
      0x3e520bf4,0xbe374e43,0x3e2fcf2a,0xbe2c7f30,0x3dc6b27f};
    auto linear=affine(value,coefficient[0]);
    if(!linear)return {};
    const auto t=linear->argument;
    auto first=sharedProduct(linear->addend,t);
    if(!first)return {};
    auto h=sharedProduct(*first,t);
    if(!h)return {};
    for(std::size_t index=1;index<10;++index) {
      h=hornerStep(*h,t,coefficient[index]);
      if(!h)return {};
    }
    if(!literal(*h,coefficient[10]))return {};
    return t;
  }
  bool softwareLogExtraction(const s::Ref& z,const s::Ref& masked,const s::Ref& t) {
    const auto* delta=operation(t,"Add",2);
    if(!delta || !arithmeticAttributes(delta->attributes))return false;
    std::optional<s::Ref> mantissa;
    for(unsigned side=0;side<2;++side)
      if(literal(delta->operands[side],0xbf800000))mantissa=delta->operands[1-side];
    if(!mantissa)return false;
    auto mantissa_bits=softwareCastSource(*mantissa,"s32","f32");
    if(!mantissa_bits)return false;
    const auto* subtraction=operation(*mantissa_bits,"Subtract",2,"s32");
    if(!subtraction || !subtraction->attributes.empty())return false;
    auto z_bits=softwareCastSource(subtraction->operands[0],"f32","s32");
    auto exponent_bits=softwareCastSource(subtraction->operands[1],"b32","s32");
    if(!z_bits || !exponent_bits || *z_bits!=z || *exponent_bits!=masked)return false;
    const auto* bit_and=operation(masked,"BitAnd",2,"b32");
    if(!bit_and || !bit_and->attributes.empty())return false;
    for(unsigned side=0;side<2;++side) {
      if(!integerLiteral(bit_and->operands[side],"b32",0xff800000))continue;
      auto shifted=softwareCastSource(bit_and->operands[1-side],"s32","b32");
      if(!shifted)continue;
      const auto* sum=operation(*shifted,"Add",2,"s32");
      if(!sum || !sum->attributes.empty())continue;
      for(unsigned term=0;term<2;++term) {
        if(!integerLiteral(sum->operands[term],"s32",0xc0cafb0d))continue;
        auto source=softwareCastSource(sum->operands[1-term],"f32","s32");
        if(source && *source==z)return true;
      }
    }
    return false;
  }
  bool shiftAmount(const s::Ref& ref) {
    if (!bare(ref)) return false;
    const auto* n = node(ref);
    if (!n || n->kind != s::Kind::Constant || n->type != "u32" ||
        !n->attributes.empty() || !n->operands.empty()) return false;
    const auto decoded = constantValue(n->type,n->name);
    return decoded && decoded->literal == "23";
  }
  std::optional<s::Ref> castSource(s::Ref ref,std::string_view from,std::string_view to) {
    if (!hasType(ref,to)) return {};
    const auto* n = node(ref);
    if (!n) return {};
    if (n->type == from && !ref.conversion.empty()) {
      const auto normalized = normalizeConversions(n->type,graph_.type(ref),ref.conversion);
      if (normalized && *normalized == "Bitcast("+std::string(from)+" -> "+std::string(to)+")")
        return graph_.ref(ref.node);
    }
    if (bare(ref) && n->kind == s::Kind::Operation && n->name == "Bitcast" &&
        n->type == to && n->attributes.empty() && n->operands.size() == 1 &&
        bare(n->operands[0]) && hasType(n->operands[0],from))
      return n->operands[0];
    return {};
  }
  struct AffineTerm { s::Ref argument,addend; };
  // Recognize either retained Fma or its cached ideal Add/Multiply expansion.
  std::optional<AffineTerm> affine(const s::Ref& value,std::uint32_t coefficient,
                                  std::string_view rounding = "nearest-even") {
    if (const auto* n = operation(value,"Fma",3)) {
      if (!arithmeticAttributes(n->attributes,rounding)) return {};
      for (unsigned side=0;side<2;++side)
        if (literal(n->operands[side],coefficient) &&
            hasType(n->operands[1-side],"f32") &&
            hasType(n->operands[2],"f32"))
          return AffineTerm{n->operands[1-side],n->operands[2]};
      return {};
    }
    const auto* sum = operation(value,"Add",2);
    if (!sum || !arithmeticAttributes(sum->attributes,rounding,rounding != "nearest-even")) return {};
    for (unsigned side=0;side<2;++side) {
      const auto* product = operation(sum->operands[side],"Multiply",2);
      if (!product || !arithmeticAttributes(product->attributes,"nearest-even",
                                            rounding != "nearest-even")) continue;
      for (unsigned factor=0;factor<2;++factor)
        if (literal(product->operands[factor],coefficient) &&
            hasType(product->operands[1-factor],"f32") &&
            hasType(sum->operands[1-side],"f32"))
          return AffineTerm{product->operands[1-factor],sum->operands[1-side]};
    }
    return {};
  }
  std::optional<s::Ref> saturation(s::Ref ref) {
    if (!hasType(ref,"f32")) return {};
    const auto* n = node(ref);
    if (!n || n->type != "f32" || ref.conversion.empty()) return {};
    const auto normalized = normalizeConversions(n->type,graph_.type(ref),ref.conversion);
    if (!normalized || *normalized != "Convert saturate(f32 -> f32)") return {};
    return graph_.ref(ref.node);
  }
  std::optional<s::Ref> product(s::Ref exponential,s::Ref scale) {
    const auto* exp = operation(exponential,"Exp2",1);
    if (!exp || !idealExp2Attributes(exp->attributes)) return {};
    auto shift_ref = castSource(scale,"b32","f32");
    if (!shift_ref) return {};
    const auto* shift = operation(*shift_ref,"ShiftLeft",2,"b32");
    if (!shift || !shift->attributes.empty() || !shiftAmount(shift->operands[1])) return {};
    auto q = castSource(shift->operands[0],"f32","b32");
    if (!q) return {};
    auto low = affine(exp->operands[0],0x32a57060);
    if (!low) return {};
    auto high = affine(low->addend,0x3fb8aa3b);
    if (!high || high->argument != low->argument ||
        !low->argument.pointer_provenance.empty()) return {};
    const auto* neg = operation(high->addend,"Negate",1);
    if (!neg || !arithmeticAttributes(neg->attributes)) return {};
    const auto* n = operation(neg->operands[0],"Add",2);
    if (!n || !arithmeticAttributes(n->attributes)) return {};
    bool same_q = false;
    for (unsigned side=0;side<2;++side)
      same_q |= n->operands[side] == *q && literal(n->operands[1-side],0xcb40007f);
    if (!same_q) return {};
    auto rounded = affine(*q,0x437c0000,"toward-negative");
    if (!rounded || !literal(rounded->addend,0x4b400001)) return {};
    auto t = saturation(rounded->argument);
    if (!t) return {};
    auto clamp_input = affine(*t,0x3bbb989d);
    if (!clamp_input || !literal(clamp_input->addend,0x3f000000) ||
        clamp_input->argument != low->argument) return {};
    return low->argument;
  }
};

// Exact/opaque consumers must not receive rewritten floating bit patterns.
bool numericNode(const s::Graph& graph,const s::Node& n) {
  if (n.kind == s::Kind::Constant || n.kind == s::Kind::Parameter ||
      n.kind == s::Kind::InitialCell) return floating(n.type);
  if (n.kind == s::Kind::Fold) return floating(n.type) && n.operands.size() == 2;
  if (n.kind != s::Kind::Operation) return false;
  if (!floating(n.type)) return false;
  if (n.name == "Exp2") return n.operands.size() == 1 &&
      graph.type(n.operands[0])==n.type && idealExp2Attributes(n.attributes);
  const bool legacy=n.operands.size()==1 &&
      legacyIdealPrimitive(n.name,n.type,n.attributes,graph.type(n.operands[0])).has_value();
  if (!legacy && !numericArithmeticAttributes(n.attributes)) return false;
  if (n.name == "Select") return n.operands.size() == 3 &&
      graph.type(n.operands[0]) == "pred" &&
      graph.type(n.operands[1]) == n.type && graph.type(n.operands[2]) == n.type;
  if(!legacy && !idealScalarOperation(n.name,n.operands.size()))return false;
  return std::all_of(n.operands.begin(),n.operands.end(),[&](const auto& operand){
    return graph.type(operand)==n.type;
  });
}

struct Transform {
  const s::Graph& input;
  const std::unordered_map<s::Id,NaturalExpIntrinsicMatch>& matches;
  const std::unordered_map<s::Id,NaturalLogIntrinsicMatch>& log_matches;
  const std::unordered_map<s::Id,SoftwareLog2IntrinsicMatch>& software_log2_matches;
  s::Graph output;
  // Cached base values are unannotated: IDs suffice, avoiding two empty Ref
  // string records per input node. Edge annotations are copied only on use.
  std::vector<std::array<std::optional<s::Id>,2>> cache;
  std::size_t limit,work=0;
  std::chrono::steady_clock::time_point deadline;
  MathIntrinsicStatistics& statistics;
  void check() {
    if (output.size() > limit) throw std::invalid_argument("math_intrinsics_node_budget");
    if ((++work & 255) == 0 && std::chrono::steady_clock::now() >= deadline)
      throw std::invalid_argument("math_intrinsics_time_budget");
  }
  bool context(const s::Ref& ref,bool numeric) const {
    if (!numeric || !ref.pointer_provenance.empty()) return false;
    if (ref.conversion.empty()) return true;
    return idealNumericConversion(input.node(ref).type,input.type(ref),ref.conversion) ||
        idealFloatingStorageReadUse(input,ref);
  }
  s::Ref use(const s::Ref& ref,bool numeric) const {
    auto result = output.ref(*cache.at(ref.node)[context(ref,numeric)]);
    result.conversion=ref.conversion;result.converted_type=ref.converted_type;
    result.pointer_provenance=ref.pointer_provenance;return result;
  }
  s::Ref copy(const s::Node& n,std::vector<s::Ref> args) {
    switch (n.kind) {
      case s::Kind::Constant:return output.constant(n.type,n.name);
      case s::Kind::Parameter:return output.parameter(n.type,n.name);
      case s::Kind::InitialCell:return output.initialCell(n.name,args.at(0),n.type);
      case s::Kind::Index:return output.index(n.index_level);
      case s::Kind::Operation:return output.operation(n.name,n.type,std::move(args),n.attributes);
      case s::Kind::Fold:return output.fold(n.fold->reducer,args.at(0),args.at(1),
          n.fold->binder_level,n.fold->count,n.type);
    }
    throw std::logic_error("invalid math intrinsic source node");
  }
  s::Ref root(const s::Ref& ref) {
    // Called only under the ideal-float output contract. The approved numeric
    // output quantization/saturation policy applies to this boundary only;
    // context() still keeps internal, predicate and bit uses exact. Retain the
    // complete output Ref annotation instead of globally rewriting its cast.
    const auto& source_type=input.node(ref).type;
    const auto& result_type=input.type(ref);
    const bool numeric=ref.pointer_provenance.empty() &&
        ((floating(source_type) && idealNumericOutputConversion(source_type,result_type,ref.conversion)) ||
         idealFloatingStorageReadUse(input,ref,true));
    struct Task {s::Id id;bool numeric,ready;};
    std::vector<Task> pending{{ref.node,numeric,false}};
    while (!pending.empty()) {
      check();const auto task=pending.back();pending.pop_back();
      if (cache[task.id][task.numeric]) continue;
      const auto& n=input.node(task.id);
      const auto found=matches.find(task.id);
      const auto* match=found==matches.end()?nullptr:&found->second;
      const auto log_found=log_matches.find(task.id);
      const auto* log_match=log_found==log_matches.end()?nullptr:&log_found->second;
      const auto software_log2_found=software_log2_matches.find(task.id);
      const auto* software_log2_match=software_log2_found==software_log2_matches.end()?
          nullptr:&software_log2_found->second;
      const bool mathematical=task.numeric && numericNode(input,n);
      const auto dependencies=[&] {
        std::vector<std::pair<s::Ref,bool>> result;
        if (task.numeric && match) {
          result.emplace_back(match->argument,true);
          if (match->second_argument) result.emplace_back(*match->second_argument,true);
          if (match->argument_factor) result.emplace_back(*match->argument_factor,true);
          if (match->base_two_residual) result.emplace_back(*match->base_two_residual,true);
          if (match->accumulator) result.emplace_back(*match->accumulator,true);
        } else if(task.numeric && log_match) {
          result.emplace_back(log_match->argument,true);
          if(log_match->offset) {
            result.emplace_back(*log_match->offset,true);
          }
        } else if(task.numeric && software_log2_match) {
          result.emplace_back(software_log2_match->argument,true);
        } else for (std::size_t i=0;i<n.operands.size();++i)
          result.emplace_back(n.operands[i],mathematical && !(n.name=="Select" && i==0));
        return result;
      }();
      if (!task.ready) {
        pending.push_back({task.id,task.numeric,true});
        for (const auto& [operand,child_numeric] : dependencies)
          if (!cache[operand.node][context(operand,child_numeric)])
            pending.push_back({operand.node,context(operand,child_numeric),false});
        continue;
      }
      s::Ref value;
      if (task.numeric && match) {
        auto argument=use(match->argument,true);
        if(match->argument_factor)argument=output.operation("Multiply","f32",
            {argument,use(*match->argument_factor,true)},"ideal_arithmetic");
        if(match->negate_argument)argument=output.operation("Negate","f32",{argument},"ideal_arithmetic");
        if(match->second_argument) {
          auto second=use(*match->second_argument,true);
          if(match->negate_second_argument)second=output.operation("Negate","f32",{second},"ideal_arithmetic");
          argument=output.operation("Add","f32",{argument,second},"ideal_arithmetic");
        }
        value=output.operation("Exp","f32",{argument},"ideal_arithmetic");
        if(match->base_two_residual) {
          auto residual=use(*match->base_two_residual,true);
          if(match->negate_base_two_residual)residual=output.operation("Negate","f32",{residual},"ideal_arithmetic");
          auto base_two=output.operation("Exp2","f32",{residual},"ideal_arithmetic");
          value=output.operation("Multiply","f32",{value,base_two},"ideal_arithmetic");
        }
        if (match->accumulator) {
          value=output.operation("Add","f32",{value,use(*match->accumulator,true)},"ideal_arithmetic");
          ++statistics.natural_exp_accumulators;
        }
        ++statistics.natural_exp_nodes;
        if(match->fast_direct)++statistics.fast_natural_exp_nodes;
      } else if(task.numeric && log_match) {
        auto argument=use(log_match->argument,true);
        if(log_match->offset) {
          // Idealize ONLY the complete typed software base-change idiom
          // RNln2*(PtxLog2(a)+b), never the standalone rounded coefficient.
          // Log(a*Exp2(b)) preserves its ideal domain a>0 for finite b;
          // positivity/finite certificates still belong to the egraph. Exact,
          // bit and quantized consumers use the unchanged nonnumeric clone.
          auto offset=output.operation("Exp2","f32",{use(*log_match->offset,true)},"ideal_arithmetic");
          argument=output.operation("Multiply","f32",{argument,offset},"ideal_arithmetic");
          ++statistics.natural_log_offsets;
        }
        value=output.operation("Log","f32",{argument},"ideal_arithmetic");
        ++statistics.natural_log_nodes;
      } else if(task.numeric && software_log2_match) {
        value=output.operation("Log2","f32",{use(software_log2_match->argument,true)},"ideal_arithmetic");
        ++statistics.software_log2_nodes;
      } else {
        std::vector<s::Ref> args;args.reserve(dependencies.size());
        for (const auto& [operand,child_numeric] : dependencies)args.push_back(use(operand,child_numeric));
        value=copy(n,std::move(args));
      }
      cache[task.id][task.numeric]=value.node;++statistics.transformed_nodes;check();
    }
    auto result=output.ref(*cache[ref.node][numeric]);
    result.conversion=ref.conversion;result.converted_type=ref.converted_type;
    result.pointer_provenance=ref.pointer_provenance;return result;
  }
};
}  // namespace

std::optional<NaturalExpIntrinsicMatch> matchNaturalExpIntrinsic(
    const s::Graph& graph,s::Ref root,std::size_t cap,std::size_t* inspected_nodes) {
  MathIntrinsicMatcher matcher(graph,cap);auto result=matcher.match(std::move(root));
  if (inspected_nodes) *inspected_nodes=matcher.inspected();return result;
}
std::optional<NaturalExpIntrinsicMatch> matchFastNaturalExpIntrinsic(
    const s::Graph& graph,s::Ref root,std::size_t cap,std::size_t* inspected_nodes) {
  MathIntrinsicMatcher matcher(graph,cap);auto result=matcher.matchFast(std::move(root));
  if(inspected_nodes)*inspected_nodes=matcher.inspected();return result;
}
std::optional<NaturalLogIntrinsicMatch> matchFastNaturalLogIntrinsic(
    const s::Graph& graph,s::Ref root,std::size_t cap,std::size_t* inspected_nodes) {
  MathIntrinsicMatcher matcher(graph,cap);auto result=matcher.matchFastLog(std::move(root));
  if(inspected_nodes)*inspected_nodes=matcher.inspected();return result;
}
std::optional<SoftwareLog2IntrinsicMatch> matchSoftwareLog2Intrinsic(
    const s::Graph& graph,s::Ref root,std::size_t cap,std::size_t* inspected_nodes) {
  MathIntrinsicMatcher matcher(graph,cap);auto result=matcher.matchSoftwareLog2(std::move(root));
  if(inspected_nodes)*inspected_nodes=matcher.inspected();return result;
}

s::Artifact abstractMathIntrinsics(s::Artifact&& input,std::size_t max_nodes,
    std::chrono::steady_clock::time_point deadline,MathIntrinsicStatistics* observed) {
  if (!max_nodes) throw std::invalid_argument("math_intrinsics_node_budget");
  if (std::chrono::steady_clock::now() >= deadline)
    throw std::invalid_argument("math_intrinsics_time_budget");
  MathIntrinsicStatistics local;auto& statistics=observed?*observed:local;
  std::unordered_map<s::Id,NaturalExpIntrinsicMatch> matches;
  std::unordered_map<s::Id,NaturalLogIntrinsicMatch> log_matches;
  std::unordered_map<s::Id,SoftwareLog2IntrinsicMatch> software_log2_matches;
  for (s::Id id=0;id<input.graph.size();++id) {
    if ((id & 255) == 0 && std::chrono::steady_clock::now() >= deadline)
      throw std::invalid_argument("math_intrinsics_time_budget");
    const auto& n=input.graph.node(id);
    if (n.kind != s::Kind::Operation || n.type != "f32" ||
        (n.name != "Multiply" && n.name != "Fma" && n.name != "Exp2" && n.name!="Select")) continue;
    std::size_t inspected=0;
    if(n.name=="Select") {
      auto logarithm=matchSoftwareLog2Intrinsic(input.graph,input.graph.ref(id),128,&inspected);
      ++statistics.match_attempts;statistics.inspected_nodes+=inspected;
      statistics.max_match_nodes=std::max(statistics.max_match_nodes,inspected);
      if(logarithm)software_log2_matches.emplace(id,std::move(*logarithm));
      continue;
    }
    auto matched=n.name=="Exp2"?
        matchFastNaturalExpIntrinsic(input.graph,input.graph.ref(id),128,&inspected):
        matchNaturalExpIntrinsic(input.graph,input.graph.ref(id),128,&inspected);
    ++statistics.match_attempts;statistics.inspected_nodes+=inspected;
    statistics.max_match_nodes=std::max(statistics.max_match_nodes,inspected);
    if(matched)matches.emplace(id,std::move(*matched));
    else if(n.name=="Multiply") {
      auto logarithm=matchFastNaturalLogIntrinsic(input.graph,input.graph.ref(id),128,&inspected);
      ++statistics.match_attempts;statistics.inspected_nodes+=inspected;
      statistics.max_match_nodes=std::max(statistics.max_match_nodes,inspected);
      if(logarithm)log_matches.emplace(id,std::move(*logarithm));
    }
  }
  if (matches.empty() && log_matches.empty() && software_log2_matches.empty()) return std::move(input);
  Transform transform{input.graph,matches,log_matches,software_log2_matches,{},
      std::vector<std::array<std::optional<s::Id>,2>>(input.graph.size()),
      max_nodes,0,deadline,statistics};
  s::Artifact result;result.outputs=std::move(input.outputs);
  result.output_families=std::move(input.output_families);
  for (auto& out : result.outputs)out.value=transform.root(out.value);
  for (auto& family : result.output_families)family.body=transform.root(family.body);
  std::vector<s::Ref> roots;
  for (const auto& out : result.outputs)roots.push_back(out.value);
  for (const auto& family : result.output_families)roots.push_back(family.body);
  if(std::chrono::steady_clock::now()>=deadline)
    throw std::invalid_argument("math_intrinsics_time_budget");
  transform.output.collect(roots);std::size_t index=0;
  if(std::chrono::steady_clock::now()>=deadline)
    throw std::invalid_argument("math_intrinsics_time_budget");
  for (auto& out : result.outputs)out.value=roots[index++];
  for (auto& family : result.output_families)family.body=roots[index++];
  result.graph=std::move(transform.output);return result;
}
}  // namespace sentryward::equivalence
