#pragma once
#include "input.hpp"
namespace rg {
// Structural analysis before import; no kernel-name or magic-constant-only rules.
// The producer must retain explicit fma.rm semantics and a saturating input edge.
struct FloatingIdiomPlan {
  std::vector<bool> rounded;
  std::vector<Id> power_source;
  std::vector<Id> log2_source;
  size_t producers=0, powers=0;
  FloatingIdiomPlan() = default;
  FloatingIdiomPlan(const std::vector<RawNode> &nodes, Arena &arena)
      : rounded(nodes.size(),false),power_source(nodes.size(),invalid),log2_source(nodes.size(),invalid) {
    auto constant=[&](const Edge &e,const char *value,const char *type) {
      if(!e.conversion.empty() || e.node>=nodes.size())return false;
      const auto &n=nodes[e.node];
      return n.kind=="Constant" && n.name==value && n.type==type && n.operands.empty() && n.attributes.empty();
    };
    for(size_t i=0;i<nodes.size();++i) {
      if((i&4095)==0)arena.check();
      const auto &shift=nodes[i];
      if(shift.kind!="Operation" || shift.name!="ShiftLeft" || shift.type!="b32" ||
         !shift.attributes.empty() || shift.operands.size()!=2 ||
         shift.operands[0].conversion!="Bitcast(f32 -> b32)" ||
         !constant(shift.operands[1],"23","u32"))continue;
      Id source=shift.operands[0].node;if(source>=i)continue;
      const auto &fma=nodes[source];
      if(fma.kind!="Operation" || fma.name!="Fma" || fma.type!="f32" ||
         (fma.attributes!="rounding=toward-negative" && fma.attributes!="rounding=toward-negative ") ||
         fma.operands.size()!=3 || !constant(fma.operands[2],"12582913","f32"))continue;
      bool multiply=false;
      for(unsigned j=0;j<2;++j)
        multiply |= constant(fma.operands[1-j],"252","f32") &&
          fma.operands[j].conversion=="Convert saturate (f32 -> f32)" &&
          fma.operands[j].node<source && nodes[fma.operands[j].node].type=="f32";
      if(!multiply)continue;
      if(!rounded[source]) {rounded[source]=true;++producers;}
      power_source[i]=source;++powers;
    }
    for(size_t i=0;i<nodes.size();++i) {
      if((i&4095)==0)arena.check();
      log2_source[i]=match_log2(nodes,Id(i));
    }
  }
  // Complete binary32 log2 polynomial, including subnormal normalization,
  // exponent extraction, exact RN-FMA coefficients and exceptional guards.
  // Fixed-size pattern only: do not inspect the mathematical input's ancestors.
  static Id match_log2(const std::vector<RawNode>& nodes,Id root) {
    auto operation=[&](Id id,const char* name,const char* type,size_t arity,const char* attributes="") {
      if(id>=nodes.size())return false;
      const auto& n=nodes[id];return n.kind=="Operation" && n.name==name && n.type==type &&
        n.operands.size()==arity && n.attributes==attributes;
    };
    auto plain=[](const Edge& e,Id id) {return e.node==id && e.conversion.empty();};
    auto constant=[&](const Edge& e,const char* literal,const char* type="f32") {
      if(!e.conversion.empty() || e.node>=nodes.size())return false;
      const auto& n=nodes[e.node];return n.kind=="Constant" && n.type==type && n.name==literal &&
        n.operands.empty() && n.attributes.empty();
    };
    auto fma=[&](Id id) {return operation(id,"Fma","f32",3,"rounding=nearest-even") ||
      operation(id,"Fma","f32",3,"rounding=nearest-even ");};
    auto child=[&](const Edge& e,const char* name,const char* type,size_t arity,const char* attributes="") {
      return e.conversion.empty() && operation(e.node,name,type,arity,attributes);
    };
    if(!operation(root,"Select","f32",3))return invalid;
    const auto& zero=nodes[root].operands;
    if(!constant(zero[1],"-inf") || !child(zero[0],"Compare","pred",2,"predicate=eq") ||
       !child(zero[2],"Select","f32",3))return invalid;
    const auto& equality=nodes[zero[0].node].operands;Id normalized=equality[0].node;
    if(!plain(equality[0],normalized) || !constant(equality[1],"0") || !operation(normalized,"Select","f32",3))return invalid;
    const auto& normal=nodes[normalized].operands;Id input=normal[2].node,predicate=normal[0].node;
    if(input>=nodes.size() || nodes[input].type!="f32" || !plain(normal[2],input) ||
       !child(normal[0],"Compare","pred",2,"predicate=lt") || !child(normal[1],"Multiply","f32",2))return invalid;
    const auto& small=nodes[predicate].operands;const auto& scaled=nodes[normal[1].node].operands;
    if(!plain(small[0],input) || !constant(small[1],"0.000000000000000000000000000000000000011754943508222875079687365372222456778186655567720875215087517062784172594547271728515625") ||
       !plain(scaled[0],input) || !constant(scaled[1],"8388608"))return invalid;
    const auto& exceptional=nodes[zero[2].node].operands;
    if(!child(exceptional[0],"Compare","pred",2,"predicate=gt") ||
       !exceptional[1].conversion.empty() || !fma(exceptional[1].node) ||
       !child(exceptional[2],"Add","f32",2))return invalid;
    const auto& finite=nodes[exceptional[0].node].operands;
    if(finite[0].node!=normalized || finite[0].conversion!="Bitcast(f32 -> u32)" ||
       !constant(finite[1],"2139095039","u32"))return invalid;
    const auto& special=nodes[exceptional[1].node].operands;
    if(!plain(special[0],normalized) || !constant(special[1],"inf") || !constant(special[2],"inf"))return invalid;
    const auto& sum=nodes[exceptional[2].node].operands;
    if(!sum[0].conversion.empty() || !fma(sum[0].node) || !sum[1].conversion.empty() || !fma(sum[1].node))return invalid;
    const auto& exponent=nodes[sum[0].node].operands;Id masked=exponent[0].node;
    if(exponent[0].conversion!="Bitcast(b32 -> s32); Convert rounding=nearest-even (s32 -> f32)" ||
       !operation(masked,"BitAnd","b32",2) || !constant(exponent[1],"0.00000011920928955078125") ||
       !child(exponent[2],"Select","f32",3))return invalid;
    const auto& adjustment=nodes[exponent[2].node].operands;
    if(!plain(adjustment[0],predicate) || !constant(adjustment[1],"-23") || !constant(adjustment[2],"0"))return invalid;
    const auto& mask=nodes[masked].operands;
    if(mask[0].conversion!="Bitcast(s32 -> b32)" || !operation(mask[0].node,"Add","s32",2) ||
       !constant(mask[1],"4286578688","b32"))return invalid;
    const auto& biased=nodes[mask[0].node].operands;
    if(biased[0].node!=normalized || biased[0].conversion!="Bitcast(f32 -> s32)" ||
       !constant(biased[1],"-1060439283","s32"))return invalid;
    const auto& polynomial=nodes[sum[1].node].operands;Id delta=polynomial[0].node;
    if(!plain(polynomial[0],delta) || !operation(delta,"Add","f32",2) ||
       !constant(polynomial[1],"1.44269502162933349609375") || !child(polynomial[2],"Multiply","f32",2))return invalid;
    const auto& reduced=nodes[delta].operands;
    if(reduced[0].conversion!="Bitcast(s32 -> f32)" || !operation(reduced[0].node,"Subtract","s32",2) ||
       !constant(reduced[1],"-1"))return invalid;
    const auto& mantissa=nodes[reduced[0].node].operands;
    if(mantissa[0].node!=normalized || mantissa[0].conversion!="Bitcast(f32 -> s32)" ||
       mantissa[1].node!=masked || mantissa[1].conversion!="Bitcast(b32 -> s32)")return invalid;
    const auto& square=nodes[polynomial[2].node].operands;
    if(!plain(square[0],delta) || !child(square[1],"Multiply","f32",2))return invalid;
    const auto& product=nodes[square[1].node].operands;
    if(!plain(product[0],delta) || !product[1].conversion.empty())return invalid;
    static constexpr const char* coefficients[]={
      "0.097020141780376434326171875","-0.1684539318084716796875","0.1716887056827545166015625",
      "-0.17900948226451873779296875","0.205123722553253173828125","-0.24046532809734344482421875",
      "0.2885711491107940673828125","-0.3606741726398468017578125","0.480898201465606689453125",
      "-0.721347510814666748046875"};
    Id horner=product[1].node;
    for(unsigned i=9;i>1;--i) {
      if(!fma(horner))return invalid;const auto& step=nodes[horner].operands;
      if(!step[0].conversion.empty() || !plain(step[1],delta) || !constant(step[2],coefficients[i]))return invalid;
      horner=step[0].node;
    }
    if(!fma(horner))return invalid;const auto& first=nodes[horner].operands;
    if(!plain(first[0],delta) || !constant(first[1],coefficients[0]) || !constant(first[2],coefficients[1]))return invalid;
    return input;
  }
  Id log2_input(Id id) const {return id<log2_source.size()?log2_source[id]:invalid;}
  bool is_rounding(Id id) const {return id<rounded.size() && rounded[id];}
  bool is_power(Id id) const {return id<power_source.size() && power_source[id]!=invalid;}
  Id lower_rounding(Arena &a,const std::vector<Id> &args) const {
    // At C=12582913 binary32 spacing is exactly one. The exact FMA result
    // C+252*s is in [C,C+252], so rounding toward -infinity equals C+floor(252*s).
    Id product=a.mul({args.at(0),args.at(1)});
    if(!a.nodes[product].facts.finite)throw Unsupported("FP exponent idiom requires finite saturated input");
    Id integral=a.floor_value(product,std::pair(Q(0),Q(252)));
    return a.add({a.constant(12582913),integral});
  }
};
} // namespace rg
