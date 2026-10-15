#include "sentryward/frontend/ptx/rounded_sign.h"
#include "sentryward/value_ir/operations.h"
#include "sentryward/equivalence/constant_values.h"

namespace sentryward::frontend::ptx {
namespace {
namespace v=value_ir;
bool bare(const v::Graph& g,v::Value x) {
  return g.contains(x) && !x.conversion() && !x.pointer() &&
      !x.threadSelector() && x.indices().empty() && g.type(x).isScalar();
}
bool ieee(const v::ArithmeticAttributes& a) {
  return !a.flush_to_zero && !a.saturate && !a.propagate_nan &&
      a.multiply_mode==v::IntegerMultiplyMode::Low &&
      (a.rounding==v::RoundingMode::None || a.rounding==v::RoundingMode::NearestEven ||
       a.rounding==v::RoundingMode::TowardZero || a.rounding==v::RoundingMode::TowardNegative ||
       a.rounding==v::RoundingMode::TowardPositive);
}
bool floating(const v::Graph& g,v::Value x,const std::string& format) {
  return bare(g,x) && g.type(x).toString()==format;
}
bool encodedLeaf(const v::Graph& g,v::Value x) {
  const auto& op=g.definingNode(x).operation();
  // The consumer must map these exact encoded values, not idealize arbitrary
  // computed operands whose prior rounding would change a/b/c themselves.
  return dynamic_cast<const v::ConstantOp*>(&op) ||
      dynamic_cast<const v::InputParameterOp*>(&op) || dynamic_cast<const v::LoadOp*>(&op);
}
const v::BinaryOp* binary(const v::Graph& g,v::Value x,v::BinaryOpcode opcode,const std::string& format) {
  if(!floating(g,x,format))return nullptr;
  const auto& n=g.definingNode(x);
  const auto* op=dynamic_cast<const v::BinaryOp*>(&n.operation());
  return op && op->opcode()==opcode && ieee(op->attributes()) && n.operands().size()==2?op:nullptr;
}
}
std::optional<RoundedPositiveSubAddShape> roundedPositiveSubAddShape(const v::Graph& g,v::Value p) {
  if(!bare(g,p) || g.type(p).element().kind()!=v::ElementKind::Predicate)return {};
  const auto& node=g.definingNode(p);
  const auto* cmp=dynamic_cast<const v::CompareOp*>(&node.operation());
  if(!cmp || cmp->predicate()!=v::ComparePredicate::Greater || cmp->flushToZero() ||
      cmp->unorderedTrue() || node.operands().size()!=2)return {};
  const auto args=g.instance(p).operands;
  if(!bare(g,args[0]) || !bare(g,args[1]))return {};
  const auto format=g.type(args[0]).toString();
  if((format!="f32" && format!="f64") || !floating(g,args[1],format))return {};
  const auto* zero=dynamic_cast<const v::ConstantOp*>(&g.definingNode(args[1]).operation());
  if(!zero || zero->pointer() || zero->literal().size()>32)return {};
  const auto z=equivalence::constantValue(format,zero->literal());
  if(!z || (z->literal!="0" && z->literal!="-0"))return {};
  if(!binary(g,args[0],v::BinaryOpcode::Add,format))return {};
  const auto add=g.instance(args[0]).operands;
  if(!floating(g,add[0],format) || !floating(g,add[1],format))return {};
  for(unsigned side=0;side<2;++side) {
    if(!binary(g,add[side],v::BinaryOpcode::Subtract,format))continue;
    const auto sub=g.instance(add[side]).operands;
    if(!floating(g,sub[0],format) || !floating(g,sub[1],format) ||
        !encodedLeaf(g,sub[0]) || !encodedLeaf(g,sub[1]) || !encodedLeaf(g,add[1-side]))continue;
    return RoundedPositiveSubAddShape{p,args[0],add[side],{sub[0],sub[1],add[1-side]},format};
  }
  return {};
}
} // namespace sentryward::frontend::ptx
