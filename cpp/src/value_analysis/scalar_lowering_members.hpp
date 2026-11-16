#pragma once
#include "scalar_fold_lowering_impl.hpp"

namespace sentryward::value_analysis {
namespace {

// Rebuild one body for `unit`. Only nodes marked
// in `depends` (one ascending pass: node IDs are topological) are rebuilt,
// bottom-up through emitOperation, the lowering's own constant-folding path;
// `memo` shares them within the unit and hash-consing shares everything
// else. A rebuild that the unlifted construction would not produce the
// same way refuses (refuseUnliftableNode, refuseUnliftableUse). δ alone moves
// cells, never decisions, so a Fold over representative-absolute cells is
// rebuilt as it stands.
inline auto Lower::instantiate(const s::Ref& root,const UnitInstance& unit,const std::vector<std::uint8_t>& depends,
                   std::unordered_map<s::Id,s::Ref>& memo) -> s::Ref {
  // A use's decorations apply on top of the rebuilt node, as graph.convert
  // and withPointer applied them to the parametric node. A folded Select
  // stands for its selected operand, which may carry its own conversion; a
  // use converting it again is not rebuilt (the unlifted convert would compose).
  const auto rebuilt=[&](const s::Ref& old) {
    if(!depends[old.node])return instantiateUse(old,unit);
    if(depends[old.node]&kKappaDependent)refuseUnliftableUse(old);
    auto value=memo.at(old.node);
    if(!old.conversion.empty() || !old.converted_type.empty()) {
      if(!value.conversion.empty() || !value.converted_type.empty())
        liftRefuse("a converted use of a folded selection");
      value.conversion=old.conversion;value.converted_type=old.converted_type;
    }
    if(!old.pointer_provenance.empty())value.pointer_provenance=old.pointer_provenance;
    return instantiateUse(value,unit);
  };
  std::vector<std::pair<s::Id,bool>> work{{root.node,false}};
  while(!work.empty()) {
    const auto [id,ready]=work.back();work.pop_back();
    if(!depends[id] || memo.contains(id))continue;
    const auto node=result.graph.node(id); // a copy: rebuilding appends nodes
    if(!ready) {
      work.emplace_back(id,true);
      for(const auto& operand:node.operands)
        if(depends[operand.node] && !memo.contains(operand.node))work.emplace_back(operand.node,false);
      continue;
    }
    const bool kappa=depends[id]&kKappaDependent;
    if(kappa)refuseUnliftableNode(node);
    s::Ref value;
    switch(node.kind) {
      case s::Kind::InputParameter: {
        const auto [axis,offset]=unitParameter(node.name);
        value=result.graph.constant(node.type,std::to_string(unit.origin[axis]+offset));
        break;
      }
      case s::Kind::InputValue: {
        // Representative-absolute (an Offset-built offset, free of input
        // parameters) moves by δ; a κ-covariant offset (symbolic read: the
        // address value graph) is only rebuilt.
        auto offset=rebuilt(node.operands.at(0));
        if(!(depends[id]&kCovariantCell))offset=shiftOffset(offset,deltaOf(unit,node.name));
        value=result.graph.initialCell(node.name,offset,node.type);
        break;
      }
      case s::Kind::Operation: {
        std::vector<s::Ref> operands;operands.reserve(node.operands.size());
        for(const auto& operand:node.operands)operands.push_back(rebuilt(operand));
        const auto rebuilt_operands=operands;
        value=kappa?emitOperation(node.name,node.type,std::move(operands),node.attributes):
            result.graph.operation(node.name,node.type,std::move(operands),node.attributes);
        // The instruction image moves to the rebuilt node, unless emission simplified it away.
        if(const auto& made=result.graph.node(value);made.name==node.name && made.operands==rebuilt_operands)
          result.graph.markInstruction(value,result.graph.instructionModes(id));
        // Uses were converted from this type; a folded literal of another
        // type would change their conversion chains.
        if(result.graph.type(value)!=node.type)liftRefuse("instantiated "+node.name+" changes its type");
        break;
      }
      case s::Kind::Fold:
        value=result.graph.fold(node.fold->reducer,rebuilt(node.operands.at(0)),rebuilt(node.operands.at(1)),
            node.fold->binder_level,node.fold->count,node.type);
        break;
      default:liftRefuse("unexpected parameter-dependent "+s::kindName(node.kind));
    }
    memo.emplace(id,value);
  }
  return rebuilt(root);
}

inline auto Lower::unitOrigin(const CtaCoordinates& cta) const -> CtaCoordinates {
  CtaCoordinates origin{};
  for(unsigned axis=0;axis<3;++axis) {
    const auto shape=options.unit_lifting->unit_shape[axis];
    origin[axis]=shape?cta[axis]-cta[axis]%shape:cta[axis];
  }
  return origin;
}

inline auto Lower::instantiationMarks(bool moves) const -> std::vector<std::uint8_t> {
  const auto& graph=result.graph;
  std::vector<std::uint8_t> marks(graph.size());
  std::vector<bool> parametric(graph.size()); // an input parameter (not a unit's) below
  for(s::Id id=0;id<graph.size();++id) {
    const auto& node=graph.node(id);
    std::uint8_t m=0;bool p=false;
    if(node.kind==s::Kind::InputParameter) {
      if(node.name.starts_with(kUnitCoordinatePrefix))m|=kKappaDependent;else p=true;
    }
    for(const auto& operand:node.operands) {
      m|=marks[operand.node]&(kKappaDependent|kCellDependent);p=p || parametric[operand.node];
      if(moves && !operand.pointer_provenance.empty())m|=kCellDependent;
    }
    if(node.kind==s::Kind::InputValue) {
      if(p)m|=kCovariantCell;else if(moves)m|=kCellDependent;
    }
    marks[id]=(m&(kKappaDependent|kCellDependent))?m:0;
    parametric[id]=p;
  }
  return marks;
}

inline auto Lower::memberBox(const CtaCoordinates& rep,const std::vector<UnitLifting::Member>& members) const -> std::optional<MemberBox> {
  const auto& shape=options.unit_lifting->unit_shape;
  std::array<std::uint64_t,3> extent{1,1,1};
  std::vector<std::array<std::uint64_t,3>> positions;positions.reserve(members.size());
  for(const auto& member:members) {
    std::array<std::uint64_t,3> at{};
    for(unsigned axis=0;axis<3;++axis) {
      if(!shape[axis] || member.origin[axis]<rep[axis] || (member.origin[axis]-rep[axis])%shape[axis])return std::nullopt;
      at[axis]=(member.origin[axis]-rep[axis])/shape[axis];
      if(at[axis]>=(std::uint64_t(1)<<32))return std::nullopt;
      extent[axis]=std::max(extent[axis],at[axis]+1);
    }
    positions.push_back(at);
  }
  if(members.empty() || static_cast<unsigned __int128>(extent[0])*extent[1]*extent[2]!=members.size()+1)return std::nullopt;
  const std::array<std::uint64_t,3> stride{1,extent[0],extent[0]*extent[1]};
  MemberBox box;std::vector<bool> seen(members.size()+1);
  std::vector<std::string> keys;
  for(const auto& [key,delta]:members.front().delta)keys.push_back(key);
  std::vector<std::vector<std::int64_t>> table(members.size()+1,std::vector<std::int64_t>(keys.size()));
  for(std::size_t i=0;i<members.size();++i) {
    const auto& at=positions[i];
    const auto linear=at[0]+stride[1]*at[1]+stride[2]*at[2];
    if(!linear || seen[linear] || members[i].delta.size()!=keys.size())return std::nullopt;
    seen[linear]=true;
    std::size_t k=0;
    for(const auto& [key,delta]:members[i].delta) {
      if(key!=keys[k])return std::nullopt;
      table[linear][k++]=delta;
    }
  }
  for(std::size_t linear=1;linear<table.size();++linear)
    for(std::size_t k=0;k<keys.size();++k) {
      __int128 sum=0;
      for(unsigned axis=0;axis<3;++axis)sum+=table[linear/stride[axis]%extent[axis]*stride[axis]][k];
      if(sum!=table[linear][k])return std::nullopt;
    }
  for(unsigned axis=0;axis<3;++axis) {
    std::uint64_t period=1,n=extent[axis];
    const auto at=[&](std::uint64_t j)->const std::vector<std::int64_t>&{return table[j*period*stride[axis]];};
    while(n>1) {
      // The longest prefix on which every δ is linear in the digit.
      std::uint64_t p=n;
      for(std::uint64_t j=2;j<n && p==n;++j)
        for(std::size_t k=0;k<keys.size() && p==n;++k)
          if(static_cast<__int128>(at(1)[k])*j!=at(j)[k])p=j;
      if(n%p)return std::nullopt;
      for(std::uint64_t j=p;j<n;++j)
        for(std::size_t k=0;k<keys.size();++k)
          if(static_cast<__int128>(at(j-j%p)[k])+at(j%p)[k]!=at(j)[k])return std::nullopt;
      MemberDigit digit{axis,p,shape[axis]*period,{}};
      for(std::size_t k=0;k<keys.size();++k)digit.delta.emplace(keys[k],at(1)[k]);
      box.digits.push_back(std::move(digit));
      period*=p;n/=p;
    }
  }
  return box;
}

// `base` plus sum_d (digit value or Index level) x coefficient_d in `type` (the arm
// digits' part folded into the literal `constant`, or into a u64 offset by shiftOffset when
// `constant` is null); an Index is converted exactly to `type`. A negative coefficient
// subtracts: the value is a member's exact offset or coordinate, never negative.
inline auto Lower::addMemberTerms(s::Ref base,std::optional<std::int64_t> constant,const std::string& type,
    const MemberBox& box,const MemberPoint& p,const std::function<std::int64_t(const MemberDigit&)>& coefficient) -> s::Ref {
  std::int64_t fixed=constant.value_or(0);
  std::vector<std::pair<std::uint32_t,std::int64_t>> terms;
  for(std::size_t d=0;d<box.digits.size();++d) {
    const auto c=coefficient(box.digits[d]);
    if(!c)continue;
    if(p.level[d])terms.emplace_back(*p.level[d],c);
    else fixed=add(fixed,mul(c,integer(p.value[d])));
  }
  if(constant)base=result.graph.constant(type,std::to_string(fixed));
  else if(fixed)base=shiftOffset(base,fixed);
  for(const auto& [level,c]:terms) {
    auto term=result.graph.index(level);
    if(type!="u64")term=result.graph.convert(term,type,"Convert");
    const auto magnitude=c<0?-static_cast<unsigned __int128>(c):static_cast<unsigned __int128>(c);
    if(magnitude!=1)term=result.graph.operation("Multiply",type,{term,result.graph.constant(type,std::to_string(std::uint64_t(magnitude)))});
    base=result.graph.operation(c<0?"Subtract":"Add",type,{base,term});
  }
  return base;
}

inline auto Lower::familyMarks() const -> std::vector<std::uint8_t> {
  const auto& graph=result.graph;
  std::vector<std::uint8_t> marks(graph.size());
  for(s::Id id=0;id<graph.size();++id) {
    const auto& node=graph.node(id);
    std::uint8_t m=0;
    if(node.kind==s::Kind::InputParameter)
      m=node.name.starts_with(kUnitCoordinatePrefix)?kFamilyUnitParameter:kFamilyParametric;
    if(node.kind==s::Kind::Index || node.kind==s::Kind::Fold || node.kind==s::Kind::InputValue)m|=kFamilyRebuild;
    for(const auto& operand:node.operands) {
      m|=marks[operand.node];
      if(!operand.pointer_provenance.empty())m|=kFamilyRebuild;
    }
    marks[id]=m;
  }
  return marks;
}

// `root` as one body over a member point: Index levels and Fold binders move up by
// p.shift, unit parameters become the point's unit coordinates, representative-absolute
// cells and slot constants move by the point's δ, κ-covariant cells (an input parameter
// in the offset) are only rebuilt. This is the per-member instantiation with the members'
// differences as Index terms, and it refuses lifting where instantiate does, through the
// same checks (refuseUnliftableNode, refuseUnliftableUse). A unit coordinate (an arm
// digit's part a literal) therefore reaches integral operations and conversions only,
// whose exact fixed-width values do not depend on whether they are folded, so the body is
// the representative's parametric body evaluated at each member. A use whose provenance
// names a CTA, or a slot constant moving along a dense digit, has no such form:
// MemberFamilyUnavailable (the caller copies per member). Marked nodes only, memoized per
// point.
inline auto Lower::memberBody(const s::Ref& root,const MemberBox& box,const MemberPoint& p,const CtaCoordinates& rep,
                  const std::vector<std::uint8_t>& marks,std::unordered_map<s::Id,s::Ref>& memo) -> s::Ref {
  const auto decorate=[&](const s::Ref& old) {
    if(marks[old.node]&kFamilyKappa)refuseUnliftableUse(old);
    auto value=marks[old.node]&kFamilyRebuild?memo.at(old.node):old;
    if(marks[old.node]&kFamilyRebuild) {
      value.conversion=old.conversion;value.converted_type=old.converted_type;
      value.pointer_provenance=old.pointer_provenance;
    }
    if(value.pointer_provenance.empty())return value;
    const auto fields=provenanceFields(value.pointer_provenance);
    if(!fields)liftRefuse("unparsed pointer provenance");
    if(std::string_view(value.pointer_provenance).substr(fields->cta_begin,fields->cta_end-fields->cta_begin)!="none")
      throw MemberFamilyUnavailable{};
    if(const auto slot=slotAllocation(value,*fields)) {
      std::int64_t shift=0;
      for(std::size_t d=0;d<box.digits.size();++d) {
        const auto c=deltaIn(box.digits[d].delta,*slot);
        if(!c)continue;
        if(p.level[d])throw MemberFamilyUnavailable{};
        shift=add(shift,mul(c,integer(p.value[d])));
      }
      value=moveSlot(std::move(value),shift);
    }
    return value;
  };
  std::vector<std::pair<s::Id,bool>> work{{root.node,false}};
  while(!work.empty()) {
    const auto [id,ready]=work.back();work.pop_back();
    if(!(marks[id]&kFamilyRebuild) || memo.contains(id))continue;
    const auto node=result.graph.node(id); // a copy: rebuilding appends nodes
    if(!ready) {
      work.emplace_back(id,true);
      for(const auto& operand:node.operands)
        if((marks[operand.node]&kFamilyRebuild) && !memo.contains(operand.node))work.emplace_back(operand.node,false);
      continue;
    }
    std::vector<s::Ref> operands;operands.reserve(node.operands.size());
    for(const auto& operand:node.operands)operands.push_back(decorate(operand));
    if(marks[id]&kFamilyKappa)refuseUnliftableNode(node);
    s::Ref value;
    switch(node.kind) {
      case s::Kind::Index:value=result.graph.index(node.index_level+p.shift);break;
      case s::Kind::InputParameter: { // a unit parameter (the only marked ones)
        const auto [axis,offset]=unitParameter(node.name);
        value=addMemberTerms({},add(integer(rep[axis]),integer(offset)),node.type,box,p,
            [axis=axis](const MemberDigit& digit){return digit.axis==axis?integer(digit.step):std::int64_t{0};});
        break;
      }
      case s::Kind::InputValue: {
        auto offset=operands.at(0);
        if(!(marks[node.operands.at(0).node]&kFamilyParametric))
          offset=addMemberTerms(offset,{},"u64",box,p,
              [&](const MemberDigit& digit){return deltaIn(digit.delta,node.name);});
        value=result.graph.initialCell(node.name,offset,node.type);
        break;
      }
      case s::Kind::Operation:
        value=result.graph.operation(node.name,node.type,std::move(operands),node.attributes);
        result.graph.markInstruction(value,result.graph.instructionModes(id));
        break;
      case s::Kind::Fold:
        value=result.graph.fold(node.fold->reducer,operands.at(0),operands.at(1),node.fold->binder_level+p.shift,
            node.fold->count,node.type);
        break;
      default:liftRefuse("unexpected member-dependent "+s::kindName(node.kind));
    }
    memo.emplace(id,value);
  }
  return decorate(root);
}

// Class mode, for the classes lifted as member families: their representative outputs
// that are families merge first, by consolidation's exact shifted-body merge, so every
// lifted family stands for a merged one and the lifted family graph fits the node budget.
// Two more exact forms are what make such outputs merge, and only this merge uses them:
// canonical sums (CanonicalForms::SumsAndBf16Fields: sums associated differently, as by
// the lanes of a warp butterfly) and contiguous equal rows as one two-dimensional family
// (tensor_families). Ordinary consolidation does without them, since there they enable no
// merge that pays for them: a flattened sum loses the shared partial sums the equivalence
// importer reuses, and a two-dimensional family can exceed the tensor's declared extent,
// which the comparison refuses. Both are exact identities, so the choice affects only
// representation and budgets, never values. A one-element family's stride addresses
// nothing; it is set to the element width so that adjacent ones can merge. Only outputs of
// one unit (store and value) merge with each other. O(graph) for the merge's affine forms,
// plus the canonical forms and the pending families and their compared bodies.
// Under node pressure (collectCompletedOutputs, pending=true) the same merge runs for the pending
// outputs of every unit, with the forms of an ordinary consolidation (typed bf16 fields, no
// canonical sums, no contiguous two-dimensional families) plus strided rows and strided runs of
// one-element outputs (consolidate_outputs.h): the representation a per-unit lowering reaches by
// consolidating its completed outputs. Returns whether any family merged.
inline auto Lower::mergeRepresentativeFamilies(const std::map<CtaCoordinates,MemberBox>* classes,bool pending) -> bool {
  const auto end=result.outputs.size();
  std::map<std::pair<CtaCoordinates,CtaCoordinates>,std::vector<std::size_t>> groups;
  for(auto i=instantiated_end;i<end;++i) {
    const auto origin=i<output_ctas.size()?output_ctas[i]:OutputOrigin{};
    if(!origin.writer || !origin.value || !output_domains.contains(i) ||
       (classes && !classes->contains(unitOrigin(*origin.writer))))continue;
    groups[{unitOrigin(*origin.writer),unitOrigin(*origin.value)}].push_back(i);
  }
  std::vector<std::size_t> owners;
  for(const auto& [units,indices]:groups)owners.insert(owners.end(),indices.begin(),indices.end());
  canonicalizeOutputs(owners,pending?s::CanonicalForms::Bf16Fields:s::CanonicalForms::SumsAndBf16Fields);
  std::vector<bool> merged(end-instantiated_end);
  std::vector<std::pair<std::size_t,std::vector<s::OutputFamily>>> replacements;
  for(const auto& [units,indices]:groups) {
    if(indices.size()<2)continue;
    std::vector<s::OutputFamily> families;families.reserve(indices.size());
    for(const auto i:indices) {
      auto family=output_domains.at(i).family(result.outputs[i]);
      if(family.shape==std::vector<std::uint64_t>{1})family.strides_bytes={family.element_bytes};
      families.push_back(std::move(family));
    }
    if(!s::mergeOutputFamilies(result.graph,families,/*tensor_families=*/!pending,/*strided_rows=*/pending,
        /*strided_runs=*/pending))continue;
    for(const auto i:indices)merged[i-instantiated_end]=true;
    replacements.emplace_back(indices.front(),std::move(families));
  }
  if(replacements.empty())return false;
  std::vector<ScalarFoldOutput> outputs;std::vector<OutputOrigin> origins;
  std::vector<std::optional<OutputDomain>> domains;
  for(auto i=instantiated_end;i<end;++i) {
    if(merged[i-instantiated_end])continue;
    outputs.push_back(result.outputs[i]);origins.push_back(i<output_ctas.size()?output_ctas[i]:OutputOrigin{});
    if(const auto found=output_domains.find(i);found!=output_domains.end())domains.emplace_back(found->second);
    else domains.emplace_back();
  }
  for(const auto& [first,families]:replacements)
    for(const auto& family:families) {
      OutputDomain domain(family.shape,family.strides_bytes);
      if(!family.element_bytes || domain.count>UINT64_MAX/family.element_bytes)
        fail("scalar_fold_prerequisite","merged output payload overflow");
      outputs.push_back({{family.allocation,family.offset,domain.count*family.element_bytes},family.body});
      origins.push_back(output_ctas[first]);domains.emplace_back(std::move(domain));
    }
  result.outputs.resize(instantiated_end);output_ctas.resize(instantiated_end);
  output_domains.erase(output_domains.lower_bound(instantiated_end),output_domains.end());
  for(std::size_t k=0;k<outputs.size();++k) {
    if(domains[k])output_domains.emplace(result.outputs.size(),std::move(*domains[k]));
    result.outputs.push_back(std::move(outputs[k]));output_ctas.push_back(origins[k]);
  }
  return true;
}

// Whether per-member copies of a class's pending outputs (values `roots`) fit the node
// budget. A copy rebuilds the moving nodes below its outputs; a rebuilt node is new only
// where something below it changes for the member: a unit parameter, a cell of an
// allocation the member moves (δ ≠ 0) or a decorated (renamed or moved) use. These are
// counted per member over the marked part in topological order, an upper bound of the
// copies' new nodes, stopping once `total` + count exceeds the budget. `total` is the graph
// these copies are added to (with the copies counted before); advanced only when they fit.
// O(marked part x log + min(members x marked part, budget)).
inline auto Lower::copiesFit(const std::vector<UnitLifting::Member>& members,const std::vector<s::Ref>& roots,
               const std::vector<std::uint8_t>& moving,std::uint64_t& planned) const -> bool {
  const auto& graph=result.graph;
  std::vector<std::uint8_t> state(graph.size()); // bit 0: in the marked part; bit 1: changed
  std::vector<s::Id> part,work;
  for(const auto& root:roots)if(moving[root.node] && !state[root.node]){state[root.node]=1;work.push_back(root.node);}
  while(!work.empty()) {
    const auto id=work.back();work.pop_back();part.push_back(id);
    for(const auto& operand:graph.node(id).operands)
      if(moving[operand.node] && !state[operand.node]){state[operand.node]=1;work.push_back(operand.node);}
  }
  std::sort(part.begin(),part.end());
  std::uint64_t total=planned;
  for(const auto& member:members) {
    for(const auto id:part) {
      const auto& node=graph.node(id);
      bool changed=node.kind==s::Kind::InputParameter;
      if(node.kind==s::Kind::InputValue && !(moving[id]&kCovariantCell)) {
        const auto delta=member.delta.find(node.name);
        changed=delta==member.delta.end() || delta->second;
      }
      for(const auto& operand:node.operands)
        changed=changed || (state[operand.node]&2) || !operand.pointer_provenance.empty();
      if(changed) {
        state[id]|=2;
        if(++total>options.max_nodes)return false;
      }
    }
    for(const auto id:part)state[id]&=1;
  }
  planned=total;
  return true;
}

// Class mode: every pending output is a representative's. Each member gets a copy,
// instantiated at its origin, δ and CTA ids, its region moved by δ(allocation), and the
// representative's own body is instantiated for its coordinate. A class whose copies do
// not fit the node budget (copiesFit) and whose units fill a member box is lifted as
// member families instead: its representative families merge first, then each output
// becomes one family per arm point: digits that move its bytes by a positive δ, and leave a
// placement's atoms in place, are the family's outer dimensions (outermost first, their δ
// the strides); the others are arms, enumerated in unit order, each arm family standing
// for all members at that point (index-function arms become pull arms, the last
// outermost); an output with no dense digit, or a body without a family form, is copied.
// The output's unit is its store's (its records must lie in one unit). Cost: two marking
// passes, the copies' count per class, and for member-family classes one member box, the
// merge and one more marking pass; then per family the marked part of its body per arm
// point (memoized per point), or, per copy, the moving part (κ, cells, renamed uses)
// memoized per unit. The lifted graph is bounded by the node budget (liftBudget).
inline auto Lower::instantiateClasses() -> void {
  const auto& lifting=*options.unit_lifting;
  auto kappa=instantiationMarks(false),moving=instantiationMarks(true);
  std::map<CtaCoordinates,std::vector<s::Ref>> roots;
  for(auto i=instantiated_end;i<result.outputs.size();++i)
    if(i<output_ctas.size() && output_ctas[i].writer)roots[unitOrigin(*output_ctas[i].writer)].push_back(result.outputs[i].value);
  std::map<CtaCoordinates,MemberBox> boxes;
  // A class whose copies do not fit on their own takes its member box. The copies of the other
  // classes share the one budget: when they do not fit together, each of them takes its box too
  // (copying some and not others would depend on the order of the classes).
  std::vector<CtaCoordinates> copied;
  std::uint64_t together=result.graph.size();bool fit=true;
  for(const auto& [rep,values]:roots) {
    const auto found=lifting.members.find(rep);
    if(found==lifting.members.end())continue;
    auto alone=std::uint64_t(result.graph.size());
    if(!copiesFit(found->second,values,moving,alone)) {
      if(auto box=memberBox(rep,found->second))boxes.emplace(rep,std::move(*box));
      continue;
    }
    copied.push_back(rep);
    fit=fit && copiesFit(found->second,values,moving,together);
  }
  if(!fit)for(const auto& rep:copied)
    if(auto box=memberBox(rep,lifting.members.at(rep)))boxes.emplace(rep,std::move(*box));
  if(!boxes.empty()) {
    mergeRepresentativeFamilies(&boxes);
    kappa=instantiationMarks(false);moving=instantiationMarks(true);
  }
  const auto marks=boxes.empty()?std::vector<std::uint8_t>{}:familyMarks();
  std::map<CtaCoordinates,std::unordered_map<s::Id,s::Ref>> own,copies;
  std::map<std::tuple<CtaCoordinates,std::vector<bool>,std::size_t>,std::unordered_map<s::Id,s::Ref>> points;
  const auto end=result.outputs.size();
  for(auto i=instantiated_end;i<end;++i) {
    const auto origin=i<output_ctas.size()?output_ctas[i]:OutputOrigin{};
    if(!origin.writer)liftRefuse("an output has no unit, or its records lie in more than one unit");
    const auto rep=unitOrigin(*origin.writer);
    const auto parametric=result.outputs[i].value;
    const auto region=result.outputs[i].region;
    if(kappa.at(parametric.node) && (!origin.value || unitOrigin(*origin.value)!=rep))
      liftRefuse("an output over unit parameters is stored by another unit");
    std::optional<OutputDomain> domain;
    if(const auto found=output_domains.find(i);found!=output_domains.end())domain=found->second;
    if(const auto found=lifting.members.find(rep);found!=lifting.members.end()) {
      if(const auto box=boxes.find(rep);box!=boxes.end() && memberFamilies(i,box->second,rep,domain,marks,points))continue;
      for(const auto& member:found->second) {
        const UnitInstance unit{member.origin,&member.delta,&member.ctas};
        auto body=instantiate(parametric,unit,moving,copies[member.origin]);
        auto moved=region;
        moved.offset=shiftRegion(region,domain,deltaOf(unit,region.allocation));
        if(domain)output_domains.emplace(result.outputs.size(),*domain);
        result.outputs.push_back({std::move(moved),std::move(body)});
        liftBudget();
      }
    }
    auto body=parametric;
    if(kappa.at(parametric.node))body=instantiate(parametric,{rep,nullptr,nullptr},kappa,own[rep]);
    result.outputs[i].value=std::move(body);
  }
  instantiated_end=result.outputs.size();output_ctas.clear();
}

// Output i as member families over `box` (instantiateClasses); false when no digit is dense
// or a body has no family form. The arm points are enumerated with the innermost arm digit
// fastest (unit order); the first replaces output i, the others are appended.
inline auto Lower::memberFamilies(std::size_t i,const MemberBox& box,const CtaCoordinates& rep,const std::optional<OutputDomain>& domain,
    const std::vector<std::uint8_t>& marks,
    std::map<std::tuple<CtaCoordinates,std::vector<bool>,std::size_t>,std::unordered_map<s::Id,s::Ref>>& points) -> bool {
  const auto parametric=result.outputs[i].value;
  const auto region=result.outputs[i].region;
  std::vector<std::string> atoms;
  if(const auto& root=result.graph.node(parametric);root.kind==s::Kind::Operation && root.name==s::kPlacementOperation)
    for(std::size_t k=1;k<root.operands.size();++k)atoms.push_back(result.graph.node(root.operands[k]).name);
  std::vector<bool> dense(box.digits.size());
  MemberPoint point;point.level.resize(box.digits.size());point.value.resize(box.digits.size());
  std::vector<std::uint64_t> shape,strides;
  for(std::size_t d=box.digits.size();d-->0;) {
    const auto& digit=box.digits[d];
    dense[d]=deltaIn(digit.delta,region.allocation)>0 &&
        std::ranges::all_of(atoms,[&](const std::string& atom){return !deltaIn(digit.delta,atom);});
    if(!dense[d])continue;
    point.level[d]=static_cast<std::uint32_t>(shape.size());
    shape.push_back(digit.count);strides.push_back(std::uint64_t(deltaIn(digit.delta,region.allocation)));
  }
  if(shape.empty())return false;
  point.shift=static_cast<std::uint32_t>(shape.size());
  std::uint64_t width=region.bytes;
  if(domain) {
    width=region.bytes/domain->count;
    shape.insert(shape.end(),domain->shape.begin(),domain->shape.end());
    if(domain->strides.empty())strides.push_back(width);
    else strides.insert(strides.end(),domain->strides.begin(),domain->strides.end());
  }
  const OutputDomain layout(shape,strides);
  if(layout.count>UINT64_MAX/width)fail("index_representation_overflow","member family payload overflow");
  std::uint64_t arms=1;
  for(std::size_t d=0;d<box.digits.size();++d)if(!dense[d])arms*=box.digits[d].count;
  std::vector<std::pair<ConstantRegion,s::Ref>> families;families.reserve(arms);
  try {
    for(std::size_t arm=0;arm<arms;++arm) {
      std::int64_t moved=0;
      for(std::size_t d=0,rest=arm;d<box.digits.size();++d) {
        if(dense[d])continue;
        point.value[d]=rest%box.digits[d].count;rest/=box.digits[d].count;
        moved=add(moved,mul(deltaIn(box.digits[d].delta,region.allocation),integer(point.value[d])));
      }
      auto family=region;family.bytes=layout.count*width;
      family.offset=shiftRegion(family,layout,moved);
      families.emplace_back(family,memberBody(parametric,box,point,rep,marks,points[{rep,dense,arm}]));
      liftBudget();
    }
  } catch(const MemberFamilyUnavailable&) {return false;}
  for(std::size_t arm=0;arm<arms;++arm) {
    auto& [family,body]=families[arm];
    const auto at=arm?result.outputs.size():i;
    output_domains.insert_or_assign(at,layout);
    if(arm)result.outputs.push_back({family,body});
    else result.outputs[i]={family,body};
  }
  return true;
}

} // namespace
} // namespace sentryward::value_analysis
