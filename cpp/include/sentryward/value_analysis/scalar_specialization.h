#pragma once
#include "sentryward/value_ir/graph.h"
#include "sentryward/value_ir/operations.h"
#include "sentryward/value_ir/native_matrix.h"
#include "sentryward/value_ir/type_conversion.h"
#include "sentryward/equivalence/constant_values.h"
#include "sentryward/value_analysis/unit_lifting.h"
#include <map>
#include <sstream>
#include <tuple>

namespace sentryward::value_analysis {
// Immutable, exact facts for fixed scalar launch parameters and coordinates.
// No memory loads, tensor values, branch overrides or algebraic guesses enter
// this cache. Each encountered typed value/edge is evaluated at most once per
// instance; this is not a traversal of every thread in the graph.
class ScalarSpecialization final {
 public:
  using Value=value_ir::Value;
  using Constant=equivalence::ConstantValue;
  // `lift_block_index`: the scalar-fold lowering's instance in lifting mode
  // (unit_lifting.h) knows no BlockIndex literal; every other instance does.
  explicit ScalarSpecialization(const value_ir::Graph& graph,bool lift_block_index=false)
      :g_(graph),lift_block_index_(lift_block_index) {}
  std::optional<Constant> known(Value value,unsigned depth=0) {
    if(depth>=128){++truncated_evaluations_;return {};}
    value=g_.canonicalInterpretation(value);
    if(value.threadSelector()) {
      // Resolve the exact requested member before reading operands or caching
      // a fact. A selector may itself be a selected value (nested shuffle).
      // Never cache that annotated query under the unselected node key.
      const auto resolved=selected(value,depth+1);
      if(resolved.threadSelector())return {};
      return known(resolved,depth+1);
    }
    const auto& type=g_.type(value);
    if(!type.isScalar() || !type.element().isBitLike() || type.element().isParameterized() ||
       type.element().width()>64 || value.pointer() || value.threadSelector() || !value.indices().empty())return {};
    const Key key{value.nodeId(),value.instanceIndex(),value.resultIndex(),value.conversion().get()};
    if(auto it=cache_.find(key);it!=cache_.end())return it->second.constant;
#if defined(SENTRYWARD_ENABLE_COMPRESSION_AUDIT)
    ++evaluations_;
#endif
    const auto truncated_before=truncated_evaluations_;
    auto result=inspect(value,depth);
    // Budget exhaustion is not an immutable fact about this value. A later
    // root query has a fresh depth budget and may prove the same expression.
    // Intrinsic unknowns remain cached; only context-dependent failure and
    // its ancestors are excluded. No additional recursive work is permitted.
    if(result || truncated_before==truncated_evaluations_)
      cache_.emplace(key,Entry{value,result});
    return result;
  }
  Value selected(Value value,unsigned depth=0,bool preserve_data_select=false) {
    // This returns an existing selected operand, never builds/remerges nodes.
    for(;depth<128;++depth) {
      value=g_.canonicalInterpretation(value);
      if(value.threadSelector()) {
        const auto thread=known(*value.threadSelector(),depth+1);
        if(!thread)break;
        std::uint64_t member;
        try {member=std::stoull(thread->literal);}catch(const std::exception&) {break;}
        if(value.conversion()) {
          const auto c=*value.conversion();
          auto source=selected(g_.fromThread(c.source,*value.threadSelector()),depth+1,preserve_data_select);
          if(source.threadSelector())break;
          auto result=c.spec->kind==value_ir::TypeConversionKind::Bitcast
              ?g_.bitcast(source,c.spec->target):g_.convert(source,c.spec->target,c.spec->attributes);
          if(!value.indices().empty())result=g_.component(result,{value.indices().begin(),value.indices().end()});
          if(value.pointer())result=g_.asPointer(result,*value.pointer());
          value=result;continue;
        }
        const auto& node=g_.definingNode(value);
        auto at=node.instanceThreads().indexOf(member);
        if(node.instanceThreads().empty() && node.origin().thread==member)at=0;
        if(!at)break;
        auto result=node.result(value.resultIndex(),*at);
        // A ThreadForward instance is that source thread's own value.
        if(dynamic_cast<const value_ir::ThreadForwardOp*>(&node.operation())) {
          if(!value.indices().empty() || value.pointer())break;
          value=g_.forwarded(result);continue;
        }
        if(!value.indices().empty())result=g_.component(result,{value.indices().begin(),value.indices().end()});
        if(value.pointer())result=g_.asPointer(result,*value.pointer());
        value=result;continue;
      }
      // A WarpMma collective has one numerical matrix and many provenance
      // members. Only a stored, non-rebinding operand pattern certifies this;
      // ordinary per-thread operations and different collectives stay distinct.
      const auto& node=g_.definingNode(value);
      if(!value.conversion() && !value.pointer() && value.instanceIndex() &&
         dynamic_cast<const value_ir::WarpMmaOp*>(&node.operation())) {
        auto [entry,added]=collective_.try_emplace(node.id(),false);
        if(added && node.hasUniformThreadTemplate()) {
          node.forEachInstancePattern([&,entry=entry](const auto&,std::size_t begin,std::size_t count,bool rebind) {
            entry->second=!begin && count==node.instanceCount() && !rebind;
          });
        }
        if(entry->second) {
          auto result=node.result(value.resultIndex());
          if(!value.indices().empty())result=g_.component(result,{value.indices().begin(),value.indices().end()});
          value=result;
        }
      }
      if(preserve_data_select || value.conversion() || value.pointer() || !value.indices().empty() ||
         !dynamic_cast<const value_ir::SelectOp*>(&g_.definingNode(value).operation()))break;
      const auto operands=g_.instanceOperands(value);
      auto condition=known(operands[0],depth+1);
      if(!condition || condition->type!="pred" || (condition->literal!="0" && condition->literal!="1"))break;
      value=operands[condition->literal=="1"?1:2];
    }
    if(depth>=128)++truncated_evaluations_;
    return value;
  }
  std::size_t evaluations() const noexcept {
#if defined(SENTRYWARD_ENABLE_COMPRESSION_AUDIT)
    return evaluations_;
#else
    return 0;
#endif
  }
 private:
  using Key=std::tuple<std::size_t,std::size_t,std::size_t,const value_ir::TypeConversion*>;
  struct Entry {Value retained;std::optional<Constant> constant;};
  const value_ir::Graph& g_;
  bool lift_block_index_=false;
  std::map<Key,Entry> cache_;
  std::size_t truncated_evaluations_=0;
  std::map<std::size_t,bool> collective_;
#if defined(SENTRYWARD_ENABLE_COMPRESSION_AUDIT)
  std::size_t evaluations_=0;
#endif
  std::optional<Constant> inspect(Value value,unsigned depth) {
    const auto type=g_.type(value).element().toString();
    if(value.conversion()) {
      const auto& conversion=*value.conversion();
      auto source=known(conversion.source,depth+1);if(!source)return {};
      std::string operation=conversion.spec->kind==value_ir::TypeConversionKind::Bitcast?"Bitcast":"Convert";
      if(conversion.spec->kind==value_ir::TypeConversionKind::Convert) {
        std::ostringstream attr;value_ir::ConvertOp(conversion.spec->target,conversion.spec->attributes).printAttributes(attr);
        if(!attr.str().empty())operation+=' '+attr.str();
      }
      return equivalence::constantValue(source->type,source->literal,
          operation+'('+source->type+" -> "+type+')',type);
    }
    const auto& node=g_.definingNode(value);const auto& op=node.operation();
    std::optional<std::string> literal;
    if(const auto* parameter=dynamic_cast<const value_ir::InputParameterOp*>(&op))literal=parameter->specializedLiteral();
    else if(const auto* constant=dynamic_cast<const value_ir::ConstantOp*>(&op))literal=constant->literal();
    else if(const auto* coordinate_op=dynamic_cast<const value_ir::CoordinateOp*>(&op)) {
      const bool block=coordinate_op->kind()==value_ir::CoordinateKind::BlockIndex;
      if(block)NoteUnitSite(lift_block_index_?UnitSite::FoldSpecialization:UnitSite::ValueSpecialization);
      if(block && lift_block_index_)return {};
      if(auto coordinate=g_.instanceOrigin(value).coordinate_value)literal=std::to_string(*coordinate);
    }
    if(literal) {
      if(type=="pred") {
        if(*literal=="true" || *literal=="1")return Constant{"pred","1"};
        if(*literal=="false" || *literal=="0")return Constant{"pred","0"};
        return {};
      }
      return equivalence::constantValue(type,*literal);
    }
    if(dynamic_cast<const value_ir::SelectOp*>(&op)) {
      auto choice=selected(value,depth+1);
      return choice==value?std::nullopt:known(choice,depth+1);
    }
    if(!dynamic_cast<const value_ir::BinaryOp*>(&op) && !dynamic_cast<const value_ir::CompareOp*>(&op))return {};
    std::vector<Constant> operands;
    for(auto operand:g_.instanceOperands(value)) {
      auto constant=known(operand,depth+1);if(!constant)return {};
      operands.push_back(std::move(*constant));
    }
    std::ostringstream attr;op.printAttributes(attr);
    return equivalence::constantOperation(std::string(op.name()),type,attr.str(),operands);
  }
};
} // namespace sentryward::value_analysis
