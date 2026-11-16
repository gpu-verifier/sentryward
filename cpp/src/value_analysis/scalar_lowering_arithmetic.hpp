#pragma once
#include "scalar_fold_lowering_impl.hpp"

namespace sentryward::value_analysis {
namespace {

inline auto Lower::plannedScalarReduction(v::Value value,const v::ElementType& type) -> std::optional<s::Ref> {
  // Plan existing scalar instruction chains before projecting their terms.
  // No logical member materialization and no comparison of scalar DAGs.
  if(indexed_context || spatial_count || type.kind()!=v::ElementKind::Floating)return {};
  const auto* binary=dynamic_cast<const v::BinaryOp*>(&input.definingNode(value).operation());
  if(!binary || binary->attributes().saturate || binary->attributes().propagate_nan)return {};
  std::optional<s::Reducer> reducer;
  switch(binary->opcode()) {
    case v::BinaryOpcode::Add:reducer=s::Reducer::Sum;break;
    case v::BinaryOpcode::Maximum:reducer=s::Reducer::Max;break;
    case v::BinaryOpcode::Minimum:reducer=s::Reducer::Min;break;
    default:return {};
  }
  const auto key=valueKey(value);
  auto found=scalar_reduction_plans.find(key);
  if(found==scalar_reduction_plans.end()) {
    ScalarReductionPlan plan{*reducer,{},{}};
    auto cursor=value;
    for(;;) {
      cursor=specialization.selected(input.canonicalInterpretation(cursor));
      const auto* op=dynamic_cast<const v::BinaryOp*>(&input.definingNode(cursor).operation());
      if(cursor.conversion() || !cursor.indices().empty() || cursor.threadSelector() ||
         !op || op->opcode()!=binary->opcode() || op->attributes().saturate || op->attributes().propagate_nan ||
         input.type(cursor)!=v::ValueType::scalar(type))break;
      const auto args=input.instanceOperands(cursor);
      if(args.size()!=2)break;
      plan.terms.push_back(args[1]);cursor=args[0];
    }
    plan.terms.push_back(cursor);std::reverse(plan.terms.begin(),plan.terms.end());
    if(plan.terms.size()>=3)plan.pieces=pure_source_templates.map(plan.terms);
    const bool compact=std::any_of(plan.pieces.begin(),plan.pieces.end(),
        [](const auto& d){return d.source && d.count>=3;});
    found=scalar_reduction_plans.emplace(key,compact?std::optional(std::move(plan)):std::nullopt).first;
  }
  if(!found->second)return {};
  const auto& plan=*found->second;
  const auto level=domains.size();
  const auto identity=*reducer==s::Reducer::Sum?"0":*reducer==s::Reducer::Max?"-inf":"inf";
  auto& progress=resumeReduction([&]{return result.graph.constant(type.toString(),identity);});
  ReductionRequests requests(requested,progress);
  auto accumulated=keys.at(progress.accumulator_key);
  for(auto i=progress.next;i<plan.pieces.size();++i) {
    const auto& piece=plan.pieces[i];
    const FoldPieceKey piece_key{*projection_key,i};
    auto [parts,inserted]=fold_pieces.try_emplace(piece_key,
        std::vector<std::pair<std::uint64_t,std::uint64_t>>{{0,piece.count}});
    for(auto sub=progress.sub_next;sub<parts->second.size();++sub) {
    const auto [first,count]=parts->second[sub];
    FoldScope scope(*this,piece_key,level,sub,first,count);
    s::Ref term;
    {
      struct RestoreDomains {
        std::vector<std::uint64_t>& current;std::vector<std::uint64_t> saved;
        ~RestoreDomains(){current=std::move(saved);}
      } restore{domains,domains};
      if(piece.source && piece.count>1) {
        Offset coordinate(integer(first));
        if(count>1){domains.push_back(count);coordinate.terms[level]=1;}
        term=inContext(std::make_shared<IndexedContext>(IndexedContext{coordinate,piece.source->root}),
            [&]{return project(piece.source->prototype,Offset(0),type);});
      } else term=scalar(plan.terms[piece.begin]);
    }
    if(piece.source && count>1)
      accumulated=result.graph.fold(*reducer,accumulated,term,level,count,type.toString());
    else accumulated=emitOperation(*reducer==s::Reducer::Sum?"Add":
        *reducer==s::Reducer::Max?"Maximum":"Minimum",type.toString(),{accumulated,term},"ideal_arithmetic");
    keys.at(progress.accumulator_key)=accumulated;progress.sub_next=sub+1;progress.pending.clear();
    }
    commitReduction(progress,i+1,accumulated);limit();
  }
  return accumulated;
}

// A term at the thread it is read from. Evaluated for the concrete value itself, outside any
// indexed use (whose bindings describe the members, not this value).
inline auto Lower::laneTerm(const v::Value& term) -> std::optional<v::Value> {
  if(!term.threadSelector())return term;
  return inUse({},[&]()->std::optional<v::Value> {
    const auto thread=known(selected(*term.threadSelector()));
    if(!thread || *thread<0)return {};
    return selected(term);
  });
}

inline auto Lower::laneRoot(const v::Value& value,const v::ElementType& type) const -> const v::BinaryOp* {
  if(type.kind()!=v::ElementKind::Predicate || value.conversion() || !value.indices().empty() ||
     value.threadSelector() || value.pointer())return nullptr;
  const auto* root=dynamic_cast<const v::BinaryOp*>(&input.definingNode(value).operation());
  return root && root->attributes().lane_own &&
      (root->opcode()==v::BinaryOpcode::BitOr || root->opcode()==v::BinaryOpcode::BitAnd)?root:nullptr;
}

inline auto Lower::laneReduction(v::Value value,const v::ElementType& type) -> LaneReductionPlan* {
  const auto* root=laneRoot(value,type);
  if(!root)return nullptr;
  const auto key=valueKey(value);
  auto found=lane_reduction_plans.find(key);
  if(found==lane_reduction_plans.end()) {
    std::optional<LaneReductionPlan> plan;
    const auto args=input.instanceOperands(value);
    if(args.size()==2) {
      LaneReductionPlan made{root->opcode()==v::BinaryOpcode::BitOr?s::Reducer::Max:s::Reducer::Min,
          args[0],args[1],{},false,{}};
      // The chain's terms in lane order: the left spine of the same unflagged operator.
      std::vector<v::Value> spine;
      auto cursor=args[1];
      for(;;) {
        cursor=specialization.selected(input.canonicalInterpretation(cursor));
        const auto* op=dynamic_cast<const v::BinaryOp*>(&input.definingNode(cursor).operation());
        if(cursor.conversion() || !cursor.indices().empty() || cursor.threadSelector() || cursor.pointer() ||
           !op || op->opcode()!=root->opcode() || op->attributes().lane_own ||
           input.type(cursor)!=v::ValueType::scalar(type))break;
        const auto operands=input.instanceOperands(cursor);
        if(operands.size()!=2)break;
        spine.push_back(operands[1]);cursor=operands[0];
      }
      spine.push_back(cursor);std::reverse(spine.begin(),spine.end());
      bool resolved=true;
      for(const auto& term:spine) {
        auto lane=laneTerm(term);
        if(!lane){resolved=false;break;}
        made.terms.push_back(std::move(*lane));
      }
      const auto own=resolved?laneTerm(args[0]):std::nullopt;
      if(own && std::find(made.terms.begin(),made.terms.end(),*own)!=made.terms.end())plan=std::move(made);
    }
    found=lane_reduction_plans.emplace(key,std::move(plan)).first;
  }
  return found->second?&*found->second:nullptr;
}

inline auto Lower::plannedLaneReduction(LaneReductionPlan& plan,const v::ElementType& type) -> std::optional<s::Ref> {
  // Like a scalar chain: planned outside indexed and spatial projections only.
  if(indexed_context || spatial_count)return {};
  if(!plan.planned) {
    plan.planned=true;
    if(plan.terms.size()>=3) {
      auto pieces=pure_source_templates.map(plan.terms);
      if(std::any_of(pieces.begin(),pieces.end(),[](const auto& d){return d.source && d.count>=3;}))
        plan.pieces=std::move(pieces);
    }
  }
  if(plan.pieces.empty())return {};
  // The pieces in lane order, each a run of lanes that are one family or a single lane, joined
  // through the accumulator: Fold(initial = what came before) for a family, the operator for
  // a single lane. The first initial is the issuing lane's own term.
  const auto level=domains.size();
  const auto name=plan.reducer==s::Reducer::Max?"BitOr":"BitAnd";
  auto& progress=resumeReduction([&]{return scalar(plan.own);});
  ReductionRequests requests(requested,progress);
  auto accumulated=keys.at(progress.accumulator_key);
  for(auto i=progress.next;i<plan.pieces.size();++i) {
    const auto& piece=plan.pieces[i];
    const FoldPieceKey piece_key{*projection_key,i};
    auto [parts,inserted]=fold_pieces.try_emplace(piece_key,
        std::vector<std::pair<std::uint64_t,std::uint64_t>>{{0,piece.count}});
    for(auto sub=progress.sub_next;sub<parts->second.size();++sub) {
      const auto [first,count]=parts->second[sub];
      FoldScope scope(*this,piece_key,level,sub,first,count);
      s::Ref term;
      {
        struct RestoreDomains {
          std::vector<std::uint64_t>& current;std::vector<std::uint64_t> saved;
          ~RestoreDomains(){current=std::move(saved);}
        } restore{domains,domains};
        if(piece.source && piece.count>1) {
          Offset coordinate(integer(first));
          if(count>1){domains.push_back(count);coordinate.terms[level]=1;}
          term=inContext(std::make_shared<IndexedContext>(IndexedContext{coordinate,piece.source->root}),
              [&]{return project(piece.source->prototype,Offset(0),type);});
        } else term=scalar(plan.terms[piece.begin]);
      }
      accumulated=piece.source && count>1
          ?result.graph.fold(plan.reducer,accumulated,term,level,count,type.toString())
          :emitOperation(name,type.toString(),{accumulated,term});
      keys.at(progress.accumulator_key)=accumulated;progress.sub_next=sub+1;progress.pending.clear();
    }
    commitReduction(progress,i+1,accumulated);limit();
  }
  return accumulated;
}

// Whether x is the result of a floating mul without rounding modifier, read without conversion:
// an add or sub without rounding modifier that reads it may be contracted with it into an fma
// (PTX ISA, mul). O(1).
inline auto Lower::contractibleProduct(v::Value x) const -> bool {
  if(x.conversion() || !x.indices().empty() || x.threadSelector())return false;
  const auto* product=dynamic_cast<const v::BinaryOp*>(&input.definingNode(x).operation());
  return product && product->opcode()==v::BinaryOpcode::Multiply &&
      product->attributes().rounding==v::RoundingMode::None &&
      input.type(x).element().kind()==v::ElementKind::Floating;
}

// Exact constants (docs/reference-binding-json.md, numerical model): an IEEE binary32/binary64
// add, sub, mul, div or fma whose operands are all exact constants has its IEEE round-to-nearest-
// even result, not the ideal real one. PTX rounds a default-rounding add/sub/mul to nearest even,
// but may contract a default mul into a default add/sub (PTX ISA, mul/add): such an add/sub keeps
// the ideal-real semantics (and is no instruction image when the mul folds: markArithmeticInstruction).
// Other rounding modes, saturation and non-finite results do as well. O(operands) per operation.
inline auto Lower::constantArithmetic(const v::Operation& op,std::span<const v::Value> args,
    const std::vector<s::Ref>& operands,const v::ElementType& type) -> std::optional<s::Ref> {
  const auto name=type.toString();
  if(type.kind()!=v::ElementKind::Floating || (name!="f32" && name!="f64"))return {};
  const auto contractible=[&](v::Value x){return contractibleProduct(x);};
  std::string operation;bool flush=false;
  if(const auto* binary=dynamic_cast<const v::BinaryOp*>(&op)) {
    const auto& a=binary->attributes();
    const bool nearest=a.rounding==v::RoundingMode::NearestEven,plain=a.rounding==v::RoundingMode::None;
    if(a.saturate || a.propagate_nan || !(nearest || plain))return {};
    switch(binary->opcode()) {
      case v::BinaryOpcode::Add:operation="Add";break;
      case v::BinaryOpcode::Subtract:operation="Subtract";break;
      case v::BinaryOpcode::Multiply:operation="Multiply";break;
      case v::BinaryOpcode::Divide:if(!nearest)return {};operation="Divide";break;
      default:return {};
    }
    if(plain && operation!="Multiply" && std::any_of(args.begin(),args.end(),contractible))return {};
    flush=a.flush_to_zero;
  } else if(const auto* fma=dynamic_cast<const v::FmaOp*>(&op)) {
    const auto& a=fma->attributes();
    if(a.saturate || a.propagate_nan || a.rounding!=v::RoundingMode::NearestEven)return {};
    operation="Fma";flush=a.flush_to_zero;
  } else return {};
  std::vector<equivalence::ConstantValue> values;
  for(const auto& r:operands) {
    auto value=exactConstant(r);
    if(!value || value->type!=name)return {};
    values.push_back(std::move(*value));
  }
  const auto folded=equivalence::ieeeArithmetic(operation,name,values,flush);
  if(!folded)return {};
  return real_expressions.constant(name,folded->literal);
}

// A conversion of an exact constant to a floating type with an explicit rounding mode is that
// constant rounded as the conversion rounds (cvt.{rn,rz,rm,rp}); saturating or flushing
// conversions keep the existing conversion edge. O(1).
inline auto Lower::constantConversion(const s::Ref& base,const v::ConversionAttributes& attributes,
    const v::ElementType& target) -> std::optional<s::Ref> {
  if(target.kind()!=v::ElementKind::Floating || attributes.saturate || attributes.flush_to_zero ||
     attributes.saturate_finite)return {};
  std::optional<equivalence::Rounding> rounding;
  switch(attributes.rounding) {
    case v::RoundingMode::NearestEven:rounding=equivalence::Rounding::NearestEven;break;
    case v::RoundingMode::TowardZero:rounding=equivalence::Rounding::TowardZero;break;
    case v::RoundingMode::TowardNegative:rounding=equivalence::Rounding::TowardNegative;break;
    case v::RoundingMode::TowardPositive:rounding=equivalence::Rounding::TowardPositive;break;
    default:return {};
  }
  const auto value=exactConstant(base);
  if(!value)return {};
  const auto type=target.toString();
  const auto folded=equivalence::roundedConversion(*value,type,*rounding);
  if(!folded)return {};
  return real_expressions.constant(type,folded->literal);
}

inline auto Lower::pure(v::Value value,const Offset& index,const v::ElementType& type) -> s::Ref {
  if(auto compact=plannedScalarReduction(value,type))return *compact;
  auto* const lane=laneReduction(value,type);
  if(lane)if(auto compact=plannedLaneReduction(*lane,type))return *compact;
  const auto& node=input.definingNode(value);const auto& op=node.operation();
  const auto input_operands=input.instanceOperands(value);
  std::vector<s::Ref> operands(input_operands.size());
  const auto projectOperand=[&](std::size_t i) {
    auto operand=input_operands[i];
    const auto& t=input.type(operand);
    const auto projectChild=[&] {
    if(t.isScalar())return scalar(operand);
    else {
      if(!t.hasStaticShape())fail("scalar_fold_parameterized_shape","matrix operand shape is descriptor-dependent");
      return project(operand,times(index,integer(t.element().width()/8)),t.element());
    }
    };
    if(indexed_context && indexed_context->use && indexed_context->use->kind==IndexedSourceUse::Kind::Operands)
      operands[i]=inUse(indexed_context->use->children.at(i),projectChild);
    else operands[i]=projectChild();
  };
  // A lane reduction that is not one family: the chain's value, as before the own term existed.
  if(lane){projectOperand(1);return operands[1];}
  if(dynamic_cast<const v::FmaOp*>(&op) && input_operands.size()==3) {
    projectOperand(2);projectOperand(0);projectOperand(1);
  } else if(const auto* binary=dynamic_cast<const v::BinaryOp*>(&op);
      binary && binary->opcode()==v::BinaryOpcode::Multiply && input_operands.size()==2 &&
      input_operands[1].nodeId()<input_operands[0].nodeId()) {
    projectOperand(1);projectOperand(0);
  } else if(dynamic_cast<const v::SelectOp*>(&op) && input_operands.size()==3 && !spatial_count &&
            inUnspecifiedCone(value)) {
    // A Select in the unspecified cone (selection contexts, above): the condition first; an arm
    // that the context or a constant rules out is not lowered; each lowered arm gets the literals
    // its condition value implies.
    projectOperand(0);
    if(const auto runs=decidedPredicate(operands[0])) {
      const std::size_t arm=*runs?1:2;
      projectOperand(arm);
      if(result.graph.type(operands[arm])==type.toString())return operands[arm];
      projectOperand(3-arm);
    } else for(const std::size_t arm:{std::size_t{1},std::size_t{2}}) {
      SelectScope scope(*this,selectContextFor(operands[0],arm==1));
      projectOperand(arm);
    }
  } else for(std::size_t i=0;i<input_operands.size();++i)projectOperand(i);
  if(auto folded=constantArithmetic(op,input_operands,operands,type))return *folded;
  if (type.kind()==v::ElementKind::Floating) {
    const auto eligible=[](const v::ArithmeticAttributes& a) {
      // This path constructs ideal-real mathematics, consistently with the
      // equivalence normalizer: rounding and FTZ are not scalar operations.
      // The original Value IR still retains the hardware attributes. Do not
      // discard saturation or special NaN behavior under this abstraction.
      return !a.saturate && !a.propagate_nan;
    };
    if (const auto* binary=dynamic_cast<const v::BinaryOp*>(&op);
        binary && eligible(binary->attributes())) {
      std::optional<s::Reducer> reducer;
      switch(binary->opcode()) {
        case v::BinaryOpcode::Add: reducer=s::Reducer::Sum; break;
        case v::BinaryOpcode::Multiply: reducer=s::Reducer::Product; break;
        case v::BinaryOpcode::Maximum: reducer=s::Reducer::Max; break;
        case v::BinaryOpcode::Minimum: reducer=s::Reducer::Min; break;
        default: break;
      }
      if(reducer || binary->opcode()==v::BinaryOpcode::Subtract || binary->opcode()==v::BinaryOpcode::Divide) {
        const auto inputs=operands;
        auto made=real_expressions.operation(std::string(op.name()),type.toString(),std::move(operands),"ideal_arithmetic",domains.size());
        markArithmeticInstruction(made,*binary,inputs,input_operands);
        return made;
      }
    }
    // Explicit PTX rounding is needed by bit-level numerical idioms (for
    // example exponent construction). Preserve the fused operation until a
    // consumer proves a complete abstraction, rather than guessing it from
    // constants after the metadata has been erased. Unspecified rounding is
    // the existing ideal-mathematics reduction contract.
    if (const auto* fma=dynamic_cast<const v::FmaOp*>(&op);fma && eligible(fma->attributes()) &&
        fma->attributes().rounding==v::RoundingMode::None) {
      auto product=real_expressions.operation("Multiply",type.toString(),
                                          {operands.at(0),operands.at(1)},"ideal_arithmetic",domains.size());
      // The fused value takes the fma's place: never an add instruction's image.
      auto fused=real_expressions.operation("Add",type.toString(),{operands.at(2),product},"ideal_arithmetic",domains.size());
      markInstruction(fused,"Add",{operands.at(2),product},s::instruction_mode::synthesized);
      markInstruction(fused,"Add",{product,operands.at(2)},s::instruction_mode::synthesized);
      return fused;
    }
    if(const auto* unary=dynamic_cast<const v::UnaryOp*>(&op);
        unary && unary->opcode()==v::UnaryOpcode::Negate && eligible(unary->attributes()))
      return real_expressions.operation("Negate",type.toString(),std::move(operands),"ideal_arithmetic",domains.size());
    if(isIdealExp2(op)) {
      const auto inputs=operands;
      auto made=real_expressions.operation("Exp2",type.toString(),std::move(operands),"ideal_arithmetic",domains.size());
      if(const auto* ptx=dynamic_cast<const v::PtxScalarOp*>(&op)) {
        const auto& modifiers=ptx->attributes().modifiers;
        const bool ftz=std::find(modifiers.begin(),modifiers.end(),"ftz")!=modifiers.end();
        markInstruction(made,"Exp2",inputs,ftz?s::instruction_mode::flush:s::instruction_mode::no_flush);
      }
      return made;
    }
    if(const auto* ptx=dynamic_cast<const v::PtxScalarOp*>(&op);
        ptx && ptx->opcode()==v::PtxScalarOpcode::Reciprocal) {
      // rcp's standard approximation belongs to the admitted ideal-real
      // numerical abstraction, like native/software Exp2 and FTZ arithmetic.
      // The original Value IR still preserves mnemonic and modifiers.
      const auto& modifiers=ptx->attributes().modifiers;
      const bool admitted=std::all_of(modifiers.begin(),modifiers.end(),[](const auto& modifier) {
        return modifier=="approx" || modifier=="ftz" || modifier=="rn" ||
          modifier=="rz" || modifier=="rm" || modifier=="rp" ||
          modifier=="f32" || modifier=="f64";
      });
      if(admitted)
        return real_expressions.operation("Reciprocal",type.toString(),std::move(operands),"ideal_arithmetic",domains.size());
    }
  }
  auto attr=attributes(op);
  // lane_own is a fact about the value view's operands, not an attribute of the scalar operator.
  if(const auto at=attr.find("lane_own ");at!=std::string::npos)attr.erase(at,9);
  if(node.resultCount()>1)attr+=" result="+std::to_string(value.resultIndex());
  if(const auto* p=dynamic_cast<const v::PtxScalarOp*>(&op))attr+=" source_type="+p->sourceType().toString();
  const auto* fma=type.kind()==v::ElementKind::Floating?dynamic_cast<const v::FmaOp*>(&op):nullptr;
  const auto inputs=fma?operands:std::vector<s::Ref>{};
  auto made=emitOperation(std::string(op.name()),type.toString(),std::move(operands),attr);
  if(fma && !fma->attributes().saturate && !fma->attributes().propagate_nan)
    markInstruction(made,"Fma",inputs,arithmeticModes(fma->attributes()));
  return made;
}

// The instruction_mode bits of a floating PTX add, sub, mul, fma, max or min: its rounding (none:
// round to nearest even, contractible into an fma) and .ftz. 0 for a rounding the equivalence
// import does not model (.rna).
inline auto Lower::arithmeticModes(const v::ArithmeticAttributes& attributes) -> std::uint8_t {
  namespace m=s::instruction_mode;
  std::uint8_t modes=attributes.flush_to_zero?m::flush:m::no_flush;
  switch(attributes.rounding) {
    case v::RoundingMode::None:return modes|m::contractible;
    case v::RoundingMode::NearestEven:return modes|m::nearest;
    case v::RoundingMode::TowardZero:return modes|m::toward_zero;
    case v::RoundingMode::TowardNegative:return modes|m::toward_negative;
    case v::RoundingMode::TowardPositive:return modes|m::toward_positive;
    default:return 0;
  }
}

// A floating PTX add, mul, max or min is the builder's Add, Multiply, Maximum or Minimum of its
// operands; a sub is Add(x, Negate(y)) (exact: x - y = x + (-y) in IEEE arithmetic). A max or min
// has no rounding: only its .ftz is recorded. A div is a Multiply by a Reciprocal, no rounded
// product, and stays unmarked. An add or sub without rounding modifier whose mul operand without
// rounding modifier was folded to a constant (constantArithmetic) is marked synthesized, never an
// image: the fold hides the product the hardware may fuse with it. O(operands).
inline auto Lower::markArithmeticInstruction(const s::Ref& made,const v::BinaryOp& binary,const std::vector<s::Ref>& operands,
    std::span<const v::Value> values) -> void {
  namespace m=s::instruction_mode;
  auto modes=arithmeticModes(binary.attributes());
  if(binary.opcode()==v::BinaryOpcode::Maximum || binary.opcode()==v::BinaryOpcode::Minimum)
    modes&=m::flush|m::no_flush;
  if((binary.opcode()==v::BinaryOpcode::Add || binary.opcode()==v::BinaryOpcode::Subtract) && (modes&m::contractible))
    for(std::size_t i=0;i<values.size() && i<operands.size();++i)
      if(contractibleProduct(values[i]) && result.graph.node(operands[i]).kind==s::Kind::Constant)modes=m::synthesized;
  switch(binary.opcode()) {
    case v::BinaryOpcode::Add:markInstruction(made,"Add",operands,modes);break;
    case v::BinaryOpcode::Multiply:markInstruction(made,"Multiply",operands,modes);break;
    case v::BinaryOpcode::Maximum:markInstruction(made,"Maximum",operands,modes);break;
    case v::BinaryOpcode::Minimum:markInstruction(made,"Minimum",operands,modes);break;
    case v::BinaryOpcode::Subtract: {
      const auto& node=result.graph.node(made);
      if(operands.size()!=2 || node.kind!=s::Kind::Operation || node.name!="Add" || node.operands.size()!=2)break;
      const auto negation=node.operands[1];
      const auto& negate=result.graph.node(negation);
      if(!negation.conversion.empty() || negate.kind!=s::Kind::Operation || negate.name!="Negate" ||
         negate.operands.size()!=1 || !sameOperand(negate.operands[0],operands[1]))break;
      markInstruction(made,"Add",{operands[0],negation},modes);
      break;
    }
    default:break;
  }
}

} // namespace
} // namespace sentryward::value_analysis
