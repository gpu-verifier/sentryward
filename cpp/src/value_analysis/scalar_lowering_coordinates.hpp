#pragma once
#include "scalar_fold_lowering_impl.hpp"

namespace sentryward::value_analysis {
namespace {

inline auto Lower::limit() -> void {
  SENTRYWARD_AUDIT(audit->maximum(AuditCounter::max_graph_nodes,result.graph.size()));
  result.peak_construction_nodes=std::max(result.peak_construction_nodes,result.graph.size());
  if(!collecting && result.graph.size()>options.max_nodes) {
    // Raw Ref locals must unwind before a moving collection. The explicit
    // worklist and reduction cache own every value needed to resume.
    if(projecting && options.construction_collection_interval)
      throw CollectProjectionFrontier{};
    fail("expression_node_limit_exceeded","scalar graph node limit exceeded");
  }
  if(domains.size()>options.max_recursion)
    fail("expression_nesting_limit_exceeded","scalar/Fold binder nesting budget exceeded");
}

inline auto Lower::bounds(const Offset& x,std::size_t depth) const -> std::pair<std::int64_t,std::int64_t> {
  if(depth>options.max_recursion)fail("layout_nesting_limit_exceeded","layout expression nesting budget exceeded");
  if(x.expression) {
    const auto& e=*x.expression;auto [lo,hi]=bounds(e.left,depth+1);
    if(e.kind==OffsetExpression::Add) {
      auto [rl,rh]=bounds(e.right,depth+1);return {add(lo,rl),add(hi,rh)};
    }
    if(e.kind==OffsetExpression::Scale) {
      auto l=mul(lo,e.factor),h=mul(hi,e.factor);return {std::min(l,h),std::max(l,h)};
    }
    if(lo<0 || e.factor<=0)fail("scalar_fold_invalid_index","invalid mixed-radix layout coordinate");
    if(e.kind==OffsetExpression::Quotient)return {lo/e.factor,hi/e.factor};
    if(lo/e.factor==hi/e.factor)return {lo%e.factor,hi%e.factor};
    const auto step=std::gcd(alignment(e.left),static_cast<std::uint64_t>(e.factor));
    return {0,e.factor-static_cast<std::int64_t>(step)};
  }
  auto lo=x.base,hi=x.base;
  for(auto [level,c]:x.terms){
    if(level>=domains.size())fail("scalar_fold_internal_binding","index is outside its Fold");
    auto amount=mul(c,integer(domains[level]?domains[level]-1:0));
    if(amount<0)lo=add(lo,amount);else hi=add(hi,amount);
  }
  return {lo,hi};
}

// Split an affine mixed-radix coordinate only when its residual is proved
// to stay within one digit. E.g. (128*r+c)/128=r for 0<=c<128.
// Visits the stored index expression, never the members of its domains.
// Compound indices may require subtree bounds; the nesting budget bounds
// this work independently of the number of tensor elements.
inline auto Lower::digit(const Offset& x,std::int64_t d,std::size_t depth) const -> std::optional<std::pair<Offset,Offset>> {
  if(depth>options.max_recursion)fail("layout_nesting_limit_exceeded","layout digit nesting budget exceeded");
  const auto normalize=[&](Offset q,Offset r)->std::optional<std::pair<Offset,Offset>> {
    const auto [lo,hi]=bounds(r);
    const auto floor=[&](std::int64_t v){return v/d-(v%d<0);};
    const auto carry=floor(lo);
    if(carry!=floor(hi))return {};
    const auto shift=-static_cast<__int128>(carry)*d;
    if(shift<INT64_MIN || shift>INT64_MAX)return {};
    return std::pair{plus(q,Offset(carry)),plus(r,Offset(static_cast<std::int64_t>(shift)))};
  };
  if(x.expression) {
    const auto& e=*x.expression;
    if(e.kind==OffsetExpression::Scale && e.factor>0) {
      if(e.factor%d==0)return std::pair{times(e.left,e.factor/d),Offset{}};
      if(d%e.factor==0)if(auto inner=digit(e.left,d/e.factor,depth+1))
        return std::pair{inner->first,times(inner->second,e.factor)};
    }
    if(e.kind==OffsetExpression::Add) {
      auto a=digit(e.left,d,depth+1),b=digit(e.right,d,depth+1);
      if(a && b) {
        auto r=plus(a->second,b->second);
        return normalize(plus(a->first,b->first),std::move(r));
      }
    }
    const auto [lo,hi]=bounds(x);
    if(lo>=0 && hi<d)return std::pair{Offset{},x};
    return {};
  }
  Offset q(x.base/d),r(x.base%d);
  for(auto [level,c]:x.terms) {
    if(c/d)q.terms[level]=c/d;
    if(c%d)r.terms[level]=c%d;
  }
  return normalize(std::move(q),std::move(r));
}

inline auto Lower::quotient(const Offset& x,std::uint64_t width) const -> Offset {
  if(!width)fail("scalar_fold_invalid_index","zero index divisor");
  if(width==1)return x;
  const auto d=integer(width);
  auto [lo,hi]=bounds(x);
  if(lo<0)fail("scalar_fold_invalid_index","negative view coordinate");
  if(auto split=digit(x,d))return std::move(split->first);
  if(lo/d==hi/d)return Offset(lo/d);
  if(!x.expression && x.base>=d) {
    auto residual=x;residual.base%=d;
    if(bounds(residual).first>=0)
      return plus(Offset(x.base/d),quotient(residual,width));
  }
  if(x.expression && x.expression->kind==OffsetExpression::Add) {
    const auto& e=*x.expression;
    if(!e.left.expression && e.left.terms.empty() && e.left.base>=d) {
      auto residual=plus(Offset(e.left.base%d),e.right);
      if(bounds(residual).first>=0)
        return plus(Offset(e.left.base/d),quotient(residual,width));
    }
  }
  return compound(OffsetExpression::Quotient,x,{},d);
}

inline auto Lower::remainder(const Offset& x,std::uint64_t width) const -> Offset {
  if(!width)fail("scalar_fold_invalid_index","zero index divisor");
  const auto [lo,hi]=bounds(x);
  if(lo<0)fail("scalar_fold_invalid_index","negative view coordinate");
  if(std::uint64_t(hi)<width)return x;
  if(x.expression && x.expression->kind==OffsetExpression::Remainder &&
     std::uint64_t(x.expression->factor)%width==0)
    return remainder(x.expression->left,width);
  if(auto split=digit(x,integer(width)))return std::move(split->second);
  if(!x.expression && x.base>=integer(width)) {
    auto residual=x;residual.base%=integer(width);
    if(bounds(residual).first>=0)return remainder(residual,width);
  }
  if(x.expression && x.expression->kind==OffsetExpression::Add) {
    const auto& e=*x.expression;
    // Removing an exact nonnegative multiple preserves the residue. Bounds
    // reject signed/wrapping cases; no replica equality is assumed.
    if(!e.left.expression && e.left.terms.empty() && e.left.base>=0) {
      auto reduced=plus(Offset(std::uint64_t(e.left.base)%width),e.right);
      const auto [rl,rh]=bounds(reduced);
      if(rl>=0)return std::uint64_t(rh)<width?reduced:
          compound(OffsetExpression::Remainder,std::move(reduced),{},integer(width));
    }
  }
  auto q=quotient(x,width);
  if(alignment(x)%width==0)return Offset{};
  if(!x.expression && !q.expression)return plus(x,times(q,-integer(width)));
  return compound(OffsetExpression::Remainder,x,{},integer(width));
}

// One exact construction path for both PTX arithmetic and operations added
// by projection (notably ExtractBits). Inspect only direct literal operands
// and their conversion edges; never search operand ancestors. The shared
// constant evaluator preserves bit width, signedness and admitted modifiers.
inline auto Lower::emitOperation(std::string name,std::string type,std::vector<s::Ref> operands,
    std::string attr) -> s::Ref {
  const auto decoded=[&](const s::Ref& value)->std::optional<equivalence::ConstantValue> {
    const auto& expression=result.graph.node(value);
    if(expression.kind!=s::Kind::Constant || !value.pointer_provenance.empty())return {};
    // Predicate literals are not integer bit encodings in constantValue.
    // Only accept the unconverted canonical Boolean literals here.
    if(expression.type=="pred" && value.conversion.empty() &&
        (value.converted_type.empty() || value.converted_type=="pred") &&
        (expression.name=="0" || expression.name=="1"))
      return equivalence::ConstantValue{"pred",expression.name};
    return equivalence::constantValue(expression.type,expression.name,value.conversion,value.converted_type);
  };
  if(name=="Select" && attr.empty() && operands.size()==3) {
    if(auto predicate=decoded(operands[0]);predicate && predicate->type=="pred" &&
        (predicate->literal=="0" || predicate->literal=="1")) {
      const auto& selected=operands[predicate->literal=="1"?1:2];
      if(result.graph.type(selected)==type)return selected;
    }
  }
  // The usual dynamic expression must not allocate a temporary literal
  // array. This pass reads only the fixed-arity operand headers twice.
  for(const auto& operand:operands)
    if(result.graph.node(operand).kind!=s::Kind::Constant || !operand.pointer_provenance.empty())
      return result.graph.operation(std::move(name),std::move(type),std::move(operands),std::move(attr));
  std::vector<equivalence::ConstantValue> constants;
  constants.reserve(operands.size());
  for(const auto& operand:operands) {
    auto value=decoded(operand);if(!value)break;
    constants.push_back(std::move(*value));
  }
  if(constants.size()==operands.size())
    if(auto value=equivalence::constantOperation(name,type,attr,constants))
      return result.graph.constant(std::move(value->type),std::move(value->literal));
  return result.graph.operation(std::move(name),std::move(type),std::move(operands),std::move(attr));
}

inline auto Lower::offsetValue(const Offset& x) -> s::Ref {
  if(x.expression) {
    const auto& e=*x.expression;
    auto left=offsetValue(e.left);
    if(e.kind==OffsetExpression::Add)return emitOperation("Add","u64",{left,offsetValue(e.right)});
    const auto name=e.kind==OffsetExpression::Scale?"Multiply":e.kind==OffsetExpression::Quotient?"Divide":"Remainder";
    return emitOperation(name,"u64",{left,result.graph.constant("u64",std::to_string(e.factor))});
  }
  auto r=result.graph.constant("u64",std::to_string(x.base));
  for(auto [level,c]:x.terms){
    // Exact local specialization; do not recursively recompute bounds here.
    if(level>=domains.size())fail("scalar_fold_internal_binding","offset binder is absent");
    if(domains[level]==1)continue;
    auto t=result.graph.index(level);
    if(c!=1)t=emitOperation("Multiply","u64",{t,result.graph.constant("u64",std::to_string(c))});
    r=emitOperation("Add","u64",{r,t});
  }
  return r;
}

// The unit parameter of a BlockIndex value: `coordinate` is its CTA's ctaid
// on `axis`, so the name carries the CTA's offset from its unit's origin
// (its %cluster_ctaid). Instantiation substitutes the literal the unlifted
// lowering emits, of the type its site uses.
inline auto Lower::unitCoordinate(v::Axis axis,std::uint64_t coordinate,std::string type) -> s::Ref {
  const auto a=static_cast<unsigned>(axis);
  const auto shape=options.unit_lifting->unit_shape.at(a);
  if(!shape)liftRefuse("empty unit shape");
  ++result.lifting.parameters;
  return result.graph.parameter(std::move(type),UnitCoordinateParameter(a,coordinate%shape));
}

// An indexed binding whose leaf is BlockIndex maps every position to that
// leaf's own coordinate (stride 0): without lifting, one u64 literal.
// Returns the leaf's BlockIndex operation when the binding is lifted; else null.
inline auto Lower::liftedBindingCoordinate(const IndexedValueBinding& binding,const Offset& position) -> const v::CoordinateOp* {
  const auto* lifted=options.unit_lifting?liftedCoordinate(input.definingNode(binding.leaf).operation()):nullptr;
  if(!lifted)return nullptr;
  const auto coordinate=input.instanceOrigin(binding.leaf).coordinate_value;
  if(position.expression || !position.terms.empty() || !coordinate || position.base<0 ||
     std::uint64_t(position.base)!=*coordinate)
    liftRefuse("BlockIndex binding is not its leaf's coordinate");
  return lifted;
}

inline auto Lower::parametricCone(const s::Ref& root) const -> bool {
  std::vector<s::Id> work{root.node};std::unordered_set<s::Id> seen;
  while(!work.empty()) {
    const auto id=work.back();work.pop_back();
    if(!seen.insert(id).second)continue;
    const auto& node=result.graph.node(id);
    if(node.kind==s::Kind::InputParameter && !node.name.starts_with(kUnitCoordinatePrefix))return true;
    for(const auto& operand:node.operands)work.push_back(operand.node);
  }
  return false;
}

// Bytes of an intermediate allocation that no store covered are opaque initial contents
// (eliminateMemory). After the final collection every graph node is reachable from an output,
// so an InitialCell of such an allocation means an output depends on bytes the kernel never
// wrote: refuse. A read whose value is discarded (e.g. by a selp on a known predicate) leaves
// no cell. Such a cell has no finiteness fact, so 0 * cell keeps it (the bytes may be NaN).
// O(graph nodes).
inline auto Lower::refuseUnwrittenOutputs() -> void {
  if(memory.unwritten_read_allocations.empty())return;
  for(s::Id id=0;id<result.graph.size();++id) {
    const auto& node=result.graph.node(id);
    if(node.kind==s::Kind::InitialCell && memory.unwritten_read_allocations.contains(node.name))
      fail("uninitialized_memory_read","An output depends on bytes of intermediate allocation "+node.name+
          " that are read before a covering store.");
  }
}

// Every completed output enters here, with its CTA in lifting mode (its
// offset is representative-absolute).
inline auto Lower::pushOutput(ConstantRegion region,s::Ref body) -> void {
  if(!output_base.empty())body=placed(std::move(body));
  result.outputs.push_back({std::move(region),std::move(body)});
  if(!options.unit_lifting)return;
  output_ctas.resize(result.outputs.size());
  output_ctas.back()=output_cta;
}

// The body annotated with its symbolic position (output_family.h), one atom cell per term.
inline auto Lower::placed(s::Ref body) -> s::Ref {
  std::vector<s::Ref> operands{body};std::vector<s::PlacementTerm> terms;
  for(const auto& term:output_base) {
    const auto* fact=findTensorProperty(memory.tensor_properties,term);
    if(!fact || !fact->lower || !fact->upper)
      fail("symbolic_output_position_unbounded","a symbolic output position's element has no bounds");
    operands.push_back(atomCell(term));
    terms.push_back({term.coefficient,*fact->lower,*fact->upper});
  }
  const auto type=result.graph.type(body);
  return result.graph.operation(std::string(s::kPlacementOperation),type,std::move(operands),s::placementAttributes(terms));
}

// Class mode: an output built from `records` of memory.outputs belongs to
// one unit only if every record's store does; otherwise its unit is
// unknown and lifting refuses at instantiation. Callers bound the records
// by memory.outputs.size(). O(records), class mode only.
inline auto Lower::includeRecordUnits(std::ranges::input_range auto&& records) -> void {
  if(!options.unit_lifting || !output_cta.writer)return;
  const auto unit=unitOrigin(*output_cta.writer);
  for(const std::size_t k:records) {
    const auto& writer=memory.outputs[k].writer;
    const auto cta=input.node(writer.node).instanceOrigin(writer.instance).cta_coordinates;
    if(!cta || unitOrigin(*cta)!=unit){output_cta.writer.reset();return;}
  }
}

// Whether a use's decoration has a step other than a bitcast from or to a non-integral
// type. The unlifted lowering evaluates such a conversion of a literal as the instruction
// rounds (constantConversion), so lifting cannot rebuild one over a unit parameter. An
// unparsed chain counts as such a conversion. O(conversion steps).
inline auto Lower::realConversion(const s::Ref& use) const -> bool {
  auto current=result.graph.node(use).type;
  auto rest=use.conversion;
  while(!rest.empty()) {
    const auto step=equivalence::leadingConversion(current,rest);
    if(!step)return true;
    if(step->operation!="Bitcast" && (!integralType(current) || !integralType(step->target)))return true;
    current=step->target;rest=step->remainder;
  }
  return false;
}

// Lifting mode: whether every member unit has δ(allocation) = 0 (a class without members has
// none to check). O(members) once per allocation (memoized).
inline auto Lower::anchoredRegion(const std::string& allocation) -> bool {
  const auto [found,added]=anchored_regions.try_emplace(allocation,true);
  if(added)
    for(const auto& [origin,members]:options.unit_lifting->members)
      for(const auto& member:members) {
        const auto delta=member.delta.find(allocation);
        if(delta==member.delta.end() || delta->second)found->second=false;
      }
  return found->second;
}

inline auto Lower::provenanceFields(std::string_view label) -> std::optional<ProvenanceFields> {
  ProvenanceFields f;std::size_t at=0;
  const auto field=[&](std::string_view key,std::string_view& out) {
    if(label.substr(at,key.size())!=key)return false;
    at+=key.size();std::size_t length=0;
    const auto parsed=std::from_chars(label.data()+at,label.data()+label.size(),length);
    if(parsed.ec!=std::errc{} || parsed.ptr>=label.data()+label.size() || *parsed.ptr!=':')return false;
    at=std::size_t(parsed.ptr-label.data())+1;
    if(length>label.size()-at)return false;
    out=label.substr(at,length);at+=length;return true;
  };
  if(!field("pointer={identity=",f.identity) || !field(",addressing=",f.addressing) ||
     label.substr(at,5)!=",cta=")return std::nullopt;
  f.cta_begin=at+5;f.cta_end=label.find(',',f.cta_begin);
  if(f.cta_end==std::string_view::npos)return std::nullopt;
  return f;
}

// A slot constant moved by `delta`, its decorations kept.
inline auto Lower::moveSlot(s::Ref use,std::int64_t delta) -> s::Ref {
  const auto& node=result.graph.node(use);
  const auto value=exactInteger<std::int64_t>(node.name);
  if(!value)liftRefuse("unparsed slot constant");
  if(!delta)return use;
  auto moved=result.graph.constant(node.type,std::to_string(add(*value,delta)));
  moved.conversion=std::move(use.conversion);moved.converted_type=std::move(use.converted_type);
  moved.pointer_provenance=std::move(use.pointer_provenance);
  return moved;
}

// A use's own changes for `unit`: its pointer provenance renamed to the
// unit's CTA, and a slot constant moved by δ.
inline auto Lower::instantiateUse(s::Ref use,const UnitInstance& unit) -> s::Ref {
  if(use.pointer_provenance.empty() || (!unit.delta && !unit.ctas))return use;
  const auto fields=provenanceFields(use.pointer_provenance);
  if(!fields)liftRefuse("unparsed pointer provenance");
  // Read the fields before the rename below edits the label they view.
  const auto slot=unit.delta?slotAllocation(use,*fields):std::nullopt;
  const auto cta=std::string_view(use.pointer_provenance).substr(fields->cta_begin,fields->cta_end-fields->cta_begin);
  if(unit.ctas && cta!="none") {
    const auto id=exactInteger<std::uint64_t>(cta);
    if(!id)liftRefuse("unparsed provenance CTA");
    const auto renamed=unit.ctas->find(*id);
    if(renamed==unit.ctas->end())liftRefuse("provenance CTA outside the unit");
    use.pointer_provenance.replace(fields->cta_begin,cta.size(),std::to_string(renamed->second));
  }
  if(slot)use=moveSlot(std::move(use),deltaOf(unit,*slot));
  return use;
}

// plus(offset, Offset(δ)) on offsetValue's emitted form, for a
// representative-absolute InitialCell offset: a plain Offset is a left-deep
// Add chain from its (always emitted) constant base over Index and
// Multiply(Index, c) terms; an expression Offset keeps a nonzero constant
// displacement as Add(Constant c, R) and drops a zero one. Any other form
// refuses (the rule was checked against members' own bodies).
inline auto Lower::shiftOffset(const s::Ref& offset,std::int64_t delta) -> s::Ref {
  if(!delta)return offset;
  if(!offset.conversion.empty() || !offset.converted_type.empty() || !offset.pointer_provenance.empty())
    liftRefuse("decorated representative-absolute offset");
  const auto node=result.graph.node(offset);
  const auto literal=[&](const s::Node& n)->std::optional<std::int64_t> {
    if(n.kind!=s::Kind::Constant || n.type!="u64")return std::nullopt;
    return exactInteger<std::int64_t>(n.name);
  };
  const auto plain=[](const s::Ref& r){return r.conversion.empty() && r.converted_type.empty() && r.pointer_provenance.empty();};
  if(const auto value=literal(node))return result.graph.constant("u64",std::to_string(add(*value,delta)));
  if(node.kind==s::Kind::Operation && node.name=="Add" && node.type=="u64" && node.attributes.empty() &&
     node.operands.size()==2 && plain(node.operands[0]) && plain(node.operands[1])) {
    const auto& right=result.graph.node(node.operands[1]);
    const bool term=right.kind==s::Kind::Index || (right.kind==s::Kind::Operation && right.name=="Multiply" &&
        right.operands.size()==2 && result.graph.node(right.operands[0]).kind==s::Kind::Index &&
        literal(result.graph.node(right.operands[1])));
    if(term)return emitOperation("Add","u64",{shiftOffset(node.operands[0],delta),node.operands[1]});
    if(const auto value=literal(result.graph.node(node.operands[0]))) {
      const auto moved=add(*value,delta);
      if(!moved)return node.operands[1];
      return emitOperation("Add","u64",{result.graph.constant("u64",std::to_string(moved)),node.operands[1]});
    }
  }
  if(node.kind==s::Kind::Operation)
    return emitOperation("Add","u64",{result.graph.constant("u64",std::to_string(delta)),offset});
  liftRefuse("unrecognized representative-absolute offset");
}

} // namespace
} // namespace sentryward::value_analysis
