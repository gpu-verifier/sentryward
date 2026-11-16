#pragma once
#include "scalar_fold_lowering_impl.hpp"

namespace sentryward::value_analysis {
namespace {

inline auto Lower::buildUnspecifiedCone() -> void {
  unspecified_cone_built=true;
  std::size_t end=0;bool opaque=false;
  for(const auto& node:input.nodes()) {
    end=std::max(end,node.id()+1);
    opaque=opaque || dynamic_cast<const v::OpaqueValueOp*>(&node.operation());
  }
  if(!opaque)return;
  std::vector<std::size_t> referenced;
  const auto operandNodes=[&](const v::Node& node) {
    referenced.clear();
    const auto add=[&](auto&& self,const v::Value& x)->void {
      referenced.push_back(input.resolve(x).nodeId());
      if(x.conversion())self(self,x.conversion()->source);
      for(const auto& index:x.indices())self(self,index);
      if(x.threadSelector())self(self,*x.threadSelector());
    };
    node.forEachInstancePattern([&](const v::Instance& pattern,std::size_t,std::size_t,bool) {
      for(const auto& operand:pattern.operands)add(add,operand);
    });
    std::sort(referenced.begin(),referenced.end());
    referenced.erase(std::unique(referenced.begin(),referenced.end()),referenced.end());
  };
  std::vector<std::size_t> first(end+1,0);
  for(const auto& node:input.nodes()) {
    operandNodes(node);
    for(const auto id:referenced)if(id<end)++first[id+1];
  }
  for(std::size_t i=0;i<end;++i)first[i+1]+=first[i];
  std::vector<std::size_t> users(first[end]),next(first.begin(),first.end()-1);
  for(const auto& node:input.nodes()) {
    operandNodes(node);
    for(const auto id:referenced)if(id<end)users[next[id]++]=node.id();
  }
  unspecified_cone.assign(end,0);
  std::vector<std::size_t> pending;
  for(const auto& node:input.nodes())
    if(dynamic_cast<const v::OpaqueValueOp*>(&node.operation())){unspecified_cone[node.id()]=1;pending.push_back(node.id());}
  while(!pending.empty()) {
    const auto id=pending.back();pending.pop_back();
    for(auto k=first[id];k<first[id+1];++k)
      if(!unspecified_cone[users[k]]){unspecified_cone[users[k]]=1;pending.push_back(users[k]);}
  }
}

inline auto Lower::inUnspecifiedCone(const v::Value& value) -> bool {
  if(!unspecified_cone_built)buildUnspecifiedCone();
  if(unspecified_cone.empty())return false;
  const auto in=[&](auto&& self,const v::Value& x)->bool {
    const auto id=input.resolve(x).nodeId();
    if(id<unspecified_cone.size() && unspecified_cone[id])return true;
    if(x.conversion() && self(self,x.conversion()->source))return true;
    for(const auto& index:x.indices())if(self(self,index))return true;
    return x.threadSelector() && self(self,*x.threadSelector());
  };
  return in(in,value);
}

// A predicate constant: PTX reads an integer constant used as a predicate as C does, zero
// false and nonzero true (ISA, Predicate Constants).
inline auto Lower::predicateConstant(const s::Ref& x) const -> std::optional<bool> {
  if(!x.conversion.empty() || !x.pointer_provenance.empty() || (!x.converted_type.empty() && x.converted_type!="pred"))
    return std::nullopt;
  const auto& node=result.graph.node(x);
  if(node.kind!=s::Kind::Constant || node.type!="pred")return std::nullopt;
  if(node.name=="true" || node.name=="false")return node.name=="true";
  const auto integer=exactInteger<std::int64_t>(node.name);
  if(!integer)return std::nullopt;
  return *integer!=0;
}

// The value predicate x has where the current selection context holds: a constant, a context
// literal, or one of those behind BitNot or a Select of two constant arms.
inline auto Lower::decidedPredicate(s::Ref x) const -> std::optional<bool> {
  bool negated=false;
  const auto context=select_context?select_contexts.find(select_context):select_contexts.end();
  for(std::size_t step=0;step<=kPredicateSteps;++step) {
    if(const auto constant=predicateConstant(x))return *constant!=negated;
    if(context!=select_contexts.end())
      for(const auto& [atom,holds]:context->second.literals)
        if(atom==x)return holds!=negated;
    const auto* node=predicateOperation(x);
    if(!node)return std::nullopt;
    if(node->name=="BitNot" && node->operands.size()==1){x=node->operands[0];negated=!negated;continue;}
    if(node->name!="Select" || node->operands.size()!=3)return std::nullopt;
    const auto yes=predicateConstant(node->operands[1]),no=predicateConstant(node->operands[2]);
    if(!yes || !no)return std::nullopt;
    if(*yes==*no)return *yes!=negated;
    if(!*yes)negated=!negated;
    x=node->operands[0];
  }
  return std::nullopt;
}

// The context of an arm on which `condition` is `value`: that literal and the literals it implies
// through BitNot, BitAnd (true), BitOr (false) and a Select with a constant arm (an arm whose
// constant differs from the Select's value did not run); at most kSelectContextLiterals.
inline auto Lower::selectContextFor(const s::Ref& condition,bool value) -> std::uint32_t {
  std::vector<std::pair<s::Ref,bool>> literals,pending{{condition,value}};
  while(!pending.empty() && literals.size()<kSelectContextLiterals) {
    const auto entry=pending.back();
    pending.pop_back();
    const auto& x=entry.first;
    const bool holds=entry.second;
    if(predicateConstant(x) ||
       std::any_of(literals.begin(),literals.end(),[&](const auto& literal){return literal.first==x;}))continue;
    literals.emplace_back(x,holds);
    const auto* node=predicateOperation(x);
    if(!node)continue;
    const auto& o=node->operands;
    if(node->name=="BitNot" && o.size()==1)pending.emplace_back(o[0],!holds);
    else if(o.size()==2 && ((node->name=="BitAnd" && holds) || (node->name=="BitOr" && !holds))) {
      pending.emplace_back(o[0],holds);pending.emplace_back(o[1],holds);
    } else if(node->name=="Select" && o.size()==3) {
      const auto yes=predicateConstant(o[1]),no=predicateConstant(o[2]);
      if(yes && *yes!=holds){pending.emplace_back(o[0],false);pending.emplace_back(o[2],holds);}
      else if(no && *no!=holds){pending.emplace_back(o[0],true);pending.emplace_back(o[1],holds);}
    }
  }
  if(literals.empty())return 0;
  const auto text=selectContextText(literals);
  const auto [entry,inserted]=select_context_ids.try_emplace(text,next_select_context);
  if(inserted)select_contexts.emplace(next_select_context++,SelectContext{std::move(literals)});
  return entry->second;
}

// The interning text of a literal set: its literals' Refs and values, sorted.
inline auto Lower::selectContextText(const std::vector<std::pair<s::Ref,bool>>& literals) -> std::string {
  std::vector<std::string> spelled;
  for(const auto& [x,holds]:literals) {
    std::string out;appendDecimal(out,x.node);out+=':';appendDecimal(out,x.owner);
    for(const auto* part:{&x.conversion,&x.converted_type,&x.pointer_provenance}) {
      out+=':';appendDecimal(out,part->size());out+=':';out+=*part;
    }
    out+=holds?"=1;":"=0;";
    spelled.push_back(std::move(out));
  }
  std::sort(spelled.begin(),spelled.end());
  std::string text;
  for(const auto& literal:spelled)text+=literal;
  return text;
}

inline auto Lower::whole(v::Value value,const Offset& index) -> s::Ref {
  const auto& op=input.definingNode(value).operation();
  const auto& type=input.type(value);
  if(const auto* c=dynamic_cast<const v::ConstantOp*>(&op)){
    if(!type.isScalar())fail("scalar_fold_matrix_constant","matrix literal needs component decoding");
    return real_expressions.constant(type.element().toString(),c->literal());
  }
  if(const auto* p=dynamic_cast<const v::InputParameterOp*>(&op)){
    // A supplied scalar launch value is a concrete specialization fact,
    // including float bits loaded through b32/b64 registers. Do not turn
    // pointer/object identity or tensor memory contents into numeric constants.
    if(type.isScalar() && p->specializedLiteral() && !p->pointer() && !value.pointer())
      return real_expressions.constant(type.element().toString(),*p->specializedLiteral());
    const auto alias=p->pointer()?pointer_aliases.find(p->symbol()):pointer_aliases.end();
    auto r=result.graph.parameter(type.element().toString(),alias!=pointer_aliases.end()?alias->second:p->symbol());
    if(!type.isScalar())r=emitOperation("ParameterElement",type.element().toString(),{r,offsetValue(index)});
    return r;
  }
  if(const auto* coordinate_op=dynamic_cast<const v::CoordinateOp*>(&op)){
    const bool lift=liftedCoordinate(op);
    if(spatial_count) {
      std::vector<std::uint64_t> values;
      const auto& members=spatialValues(value);
      for(std::size_t i=0;i<members.count;++i) {
        auto coordinate=input.instanceOrigin(members.at(i)).coordinate_value;
        if(!coordinate)spatialReject();
        values.push_back(*coordinate);
      }
      auto offsets=spatialOffsets(values);
      if(lift) {
        // BlockIndex is constant across one CTA's members: the parameter plus
        // the zero-stride spatial offset (without lifting: one u64 literal, converted).
        if(offsets.expression || !offsets.terms.empty())liftRefuse("spatial BlockIndex members differ");
        return result.graph.convert(unitCoordinate(coordinate_op->axis(),values.front(),"u64"),
            type.element().toString(),"Convert");
      }
      return result.graph.convert(offsetValue(offsets),type.element().toString(),"Convert");
    }
    const auto origin=input.instanceOrigin(value);
    if(!origin.coordinate_value)fail("scalar_fold_coordinate","execution coordinate is not bound");
    if(lift)return unitCoordinate(coordinate_op->axis(),*origin.coordinate_value,type.element().toString());
    return result.graph.constant(type.element().toString(),std::to_string(*origin.coordinate_value));
  }
  if(dynamic_cast<const v::WarpMmaOp*>(&op))return warpMma(value,index);
  if(dynamic_cast<const v::TcgenMmaOp*>(&op) || dynamic_cast<const v::MatMulOp*>(&op))return mma(value,index);
  if(scalarOperation(op))return pure(value,index,type.element());
  if(const auto* opaque=dynamic_cast<const v::OpaqueValueOp*>(&op))unspecified(value,*opaque);
  fail("scalar_fold_operation_unsupported","scalar semantics unavailable for "+std::string(op.name()));
}

// The numeric fields of a wide packed output record: a scalar wider than 64 bits
// built by Add/BitOr/ShiftLeft from narrower fields (e.g. st.shared.b128 of four bf16x2 words
// that a TMA store copies out), decoded by the shared storage decoder at 128 bits. Some only
// when every field is a byte-aligned numeric scalar of one floating element type and the
// fields cover the record. O(record DAG).
inline auto Lower::wideRecordFields(v::Value value) const -> std::optional<std::pair<PackedStorageDecoder::Fields,v::ElementType>> {
  const auto& type=input.type(value);
  if(!type.isScalar() || type.element().width()<=64 || type.element().width()>128)return std::nullopt;
  auto fields=PackedStorageDecoder(input,[this](auto x){return known(x);},128).decode(value);
  if(!fields || fields->size()<2)return std::nullopt;
  std::optional<v::ElementType> element;unsigned end=0;
  for(const auto& field:*fields) {
    const auto& t=input.type(field.value);
    if(!t.isScalar() || t.element().kind()!=v::ElementKind::Floating || t.element().isParameterized() ||
       t.element().width()!=field.width || field.width%8 || field.start!=end)return std::nullopt;
    if(element && *element!=t.element())return std::nullopt;
    element=t.element();end+=field.width;
  }
  if(end!=type.element().width())return std::nullopt;
  return std::pair{std::move(*fields),*element};
}

// Consecutive wide packed records [begin, end) of one allocation, CTA and field layout as their
// numeric field outputs, in address order (one OutputWrite per field), for spatialOutput. The
// records stay what they are; each field is the stored value's own slice. O(records x fields).
inline auto Lower::wideRecordRun(std::size_t begin,std::size_t& end) -> std::optional<std::vector<OutputWrite>> {
  const auto& first=memory.outputs[begin];
  const auto layout=wideRecordFields(first.value);
  if(!layout || first.reduced || !first.base.empty() || first.value_byte_offset || first.element_type)return std::nullopt;
  const auto cta=input.instanceOrigin(first.value).cta;
  std::vector<OutputWrite> fields;
  end=begin;
  for(;end<memory.outputs.size();++end) {
    const auto& record=memory.outputs[end];
    if(end>begin) {
      if(record.region.allocation!=first.region.allocation || record.region.bytes!=first.region.bytes ||
         record.region.offset!=memory.outputs[end-1].region.offset+first.region.bytes || record.reduced ||
         !record.base.empty() || record.value_byte_offset || record.element_type ||
         input.instanceOrigin(record.value).cta!=cta ||
         comparedIn(record.region.allocation,record.region.offset,record.region.bytes)!=Compared::All)break;
    }
    const auto own=end==begin?layout:wideRecordFields(record.value);
    if(!own || own->second!=layout->second || own->first.size()!=layout->first.size())break;
    for(const auto& field:own->first) {
      auto region=record.region;region.offset+=field.start/8;region.bytes=field.width/8;
      fields.push_back({region,field.value,record.writer,0,std::nullopt,{},{}});
    }
  }
  if(end-begin<1 || fields.size()<2)return std::nullopt;
  return fields;
}

inline auto Lower::memberValue(v::Value value,std::size_t member) const -> v::Value {
  if(value.threadSelector()) {
    auto selector=*value.threadSelector();
    const auto map=input.threadBindingMap(selector,value);
    if(!map)fail("compact_view_unsupported","nested selector requires a receiver-domain member mapping");
    if(!map->fixed) {
      const auto at=__int128(map->base)+__int128(map->stride)*member;
      if(at<0 || at>SIZE_MAX)fail("scalar_fold_invalid_index","nested selector member is outside its domain");
      selector=memberValue(selector,static_cast<std::size_t>(at));
    }
    auto base=value.withoutSelection();
    if(!value.indices().empty())base=input.component(base,{value.indices().begin(),value.indices().end()});
    return input.fromThread(memberValue(base,member),selector);
  }
  if(!value.indices().empty())return input.component(memberValue(value.withoutSelection(),member),
      {value.indices().begin(),value.indices().end()});
  if(value.conversion()) {
    const auto& c=*value.conversion();auto source=memberValue(c.source,member);
    return c.spec->kind==v::TypeConversionKind::Bitcast?input.bitcast(source,c.spec->target):
        input.convert(source,c.spec->target,c.spec->attributes);
  }
  return input.node(value.nodeId()).result(value.resultIndex(),member);
}

// Whether `a` and `b` are instances of one value template, so that a member index names the
// node instance under both (memberUse): the same node and result and equal annotations, a type
// conversion of equal kind over its own unannotated instance (memberValue rebuilds the same
// conversion at every member). No component, selection or pointer annotation. O(conversion depth).
inline auto Lower::sameMemberTemplate(const v::Value& a,const v::Value& b) -> bool {
  if(a.graphIdentity()!=b.graphIdentity() || a.nodeId()!=b.nodeId() || a.resultIndex()!=b.resultIndex() ||
     !a.indices().empty() || !b.indices().empty() || a.threadSelector() || b.threadSelector() ||
     a.pointer() || b.pointer() || bool(a.conversion())!=bool(b.conversion()))return false;
  if(!a.conversion())return true;
  const auto& x=*a.conversion();const auto& y=*b.conversion();
  return (x.spec==y.spec || *x.spec==*y.spec) && x.source.instanceIndex()==a.instanceIndex() &&
      y.source.instanceIndex()==b.instanceIndex() && sameMemberTemplate(x.source,y.source);
}

inline auto Lower::memberAffineRead(const v::Value& value,const Offset& position,std::int64_t lo,
    std::uint64_t width,const v::ElementType& desired) -> std::optional<s::Ref> {
  if(position.expression || position.terms.size()!=1 || lo<0)return std::nullopt;
  const auto [level,step]=*position.terms.begin();
  if(level>=domains.size() || step<=0 || domains[level]<2)return std::nullopt;
  const auto n=domains[level];
  struct Member {const ReadSlicePattern* slice;__int128 source,bytes;};
  std::vector<Member> members;members.reserve(n);
  for(std::uint64_t t=0;t<n;++t) {
    const auto at=static_cast<__int128>(position.base)+static_cast<__int128>(step)*t;
    if(at<0 || at>static_cast<__int128>(SIZE_MAX))return std::nullopt;
    const auto* slice=memory.readSlicePattern({value.nodeId(),std::size_t(at)},std::uint64_t(lo),width);
    if(!slice)return std::nullopt;
    const auto i=at-static_cast<__int128>(slice->first.instance);
    members.push_back({slice,static_cast<__int128>(slice->source.instanceIndex())+slice->source_instance_stride*i,
        static_cast<__int128>(slice->source_offset)-static_cast<__int128>(slice->result_offset)+lo+slice->byte_stride*i});
  }
  const auto same=[&](std::uint64_t a,std::uint64_t b){return sameMemberTemplate(members[a].slice->source,members[b].slice->source);};
  std::vector<std::pair<std::uint64_t,std::uint64_t>> runs; // [begin, end) binder values
  for(std::uint64_t begin=0;begin<n;) {
    auto end=begin+1;
    if(end<n && same(begin,end)) {
      const auto dm=members[end].source-members[begin].source,db=members[end].bytes-members[begin].bytes;
      for(++end;end<n && same(begin,end) && members[end].source-members[end-1].source==dm &&
          members[end].bytes-members[end-1].bytes==db;++end) {}
      // Any two members are affine; a run needs a third member that confirms the step.
      if(end-begin<kMinAffineRun)end=begin+1;
    }
    runs.emplace_back(begin,end);begin=end;
  }
  if(runs.size()==1) {
    const auto& first=members.front();
    const auto dm=members[1].source-first.source,db=members[1].bytes-first.bytes;
    const auto fits=[](__int128 x){return x>=INT64_MIN && x<=INT64_MAX;};
    if(first.source<0 || first.bytes<0 || !fits(first.source) || !fits(first.bytes) || !fits(dm) || !fits(db))
      return std::nullopt;
    Offset source_members(static_cast<std::int64_t>(first.source));
    if(dm)source_members.terms[level]=static_cast<std::int64_t>(dm);
    Offset source_bytes(static_cast<std::int64_t>(first.bytes));
    if(db)source_bytes.terms[level]=static_cast<std::int64_t>(db);
    const auto source=first.slice->source;
    return inContext(std::make_shared<IndexedContext>(IndexedContext{source_members,memberUse(source,0,1)}),
        [&]{return project(source,source_bytes,desired);});
  }
  if(std::all_of(runs.begin(),runs.end(),[](const auto& run){return run.second-run.first==1;}))return std::nullopt;
  PartitionedRead partition{v::ValueType::matrix(bit(8),{input.definingNode(value).instanceCount()}),{},true};
  for(const auto& [begin,end]:runs) {
    const auto first=std::uint64_t(position.base)+std::uint64_t(step)*begin;
    const auto last=std::uint64_t(position.base)+std::uint64_t(step)*(end-1);
    partition.regions.push_back({first,last-first+1,MemoryView{partition.type,MemoryViewKind::Slice,{},
        {{0,0,first,last-first+1,{}}},{}}});
  }
  refinePartitionDomain(partition,position,bit(8));
  return std::nullopt;
}

inline auto Lower::projectMemberTemplate(v::Value value,const Offset& offset,const v::ElementType& desired) -> s::Ref {
  const auto original=indexed_context->use;
  if(auto found=expanded_member_uses.find(original.get());found!=expanded_member_uses.end())
    // Expanding a stored member binding is the implementation of THIS
    // request, not another dependency on the same value and byte range.
    // Keep one continuation/cache identity per request. The expanded use
    // is Conversion/Operands; only its actual operand queries enqueue work.
    return inUse(found->second,[&]{return projectImpl(value,offset,desired);});
  const auto map=*indexed_context->use->binding;
  const auto position=plus(Offset(map.base),times(indexed_context->index,map.stride));
  const auto [first,last]=bounds(position);
  const auto& node=input.definingNode(value);
  if(!value.indices().empty() && !value.threadSelector()) {
    auto use=bindSourceUse(input,value,value,map.base,map.stride);
    if(!use)fail("compact_view_unsupported","component requires independent producer and index mappings");
    expanded_member_uses.emplace(original.get(),use);
    return inUse(std::move(use),[&]{return projectImpl(value,offset,desired);});
  }
  if(node.isCollectiveOccurrence() && dynamic_cast<const v::WarpMmaOp*>(&node.operation()) &&
      !value.conversion() && !value.threadSelector() && !value.pointer()) {
    if(first<0 || last<first || std::uint64_t(last)>=node.instanceCount())
      fail("scalar_fold_invalid_index","collective use exceeds its member domain");
    return inUse({},[&]{return project(value,offset,desired);});
  }
  // A selection is indexed in the receiver's domain, not the selected
  // producer's domain (a warp broadcast commonly selects a singleton).
  if(value.threadSelector()) {
    const auto selector=input.canonicalInterpretation(*value.threadSelector());
    const auto count=input.definingNode(selector).instanceCount();
    if(first<0 || last<first || (count!=1 && std::uint64_t(last)>=count))
      fail("scalar_fold_invalid_index","selector member map exceeds its domain");
    auto [entry,inserted]=member_selectors.try_emplace(valueKey(selector));
    if(inserted) {
      // Validate existing finite selector bindings once. This does enumerate
      // the selector's members, but never source values, payloads or outputs.
      std::size_t begin=0;std::optional<std::int64_t> previous;
      for(std::size_t i=0;i<count;++i) {
        const auto current=inUse({},[&]{return known(selected(memberValue(selector,i)));});
        if(!current || *current<0) {entry->second.clear();break;}
        if(previous && *current!=*previous) {
          entry->second.emplace(begin,SelectorRun{i,*previous});begin=i;
        }
        previous=current;
        if(i+1==count)entry->second.emplace(begin,SelectorRun{count,*previous});
      }
    }
    const auto at=count==1?0:std::size_t(first);
    auto run=entry->second.upper_bound(at);
    if(run==entry->second.begin())fail("compact_view_unsupported","thread selector has no static member domain");
    --run;
    if(count!=1 && std::uint64_t(last)>=run->second.end) {
      // Selection facts already define disjoint index domains. Reuse the
      // same domain refinement as reads; do not project each receiver or
      // assume that distinct selected producers carry equal values.
      PartitionedRead selection{v::ValueType::matrix(bit(8),{count}),{},true};
      for(const auto& [begin,item]:entry->second) {
        if(integer(item.end)<=first || integer(begin)>last)continue;
        selection.regions.push_back({begin,item.end-begin,
            MemoryView{selection.type,MemoryViewKind::Slice,{},
                {{0,0,begin,item.end-begin,{}}},{}}});
      }
      refinePartitionDomain(selection,position,bit(8));
      fail("compact_view_unsupported","thread selection crosses distinct producer domains");
    }
    auto source=value.withoutSelection();
    if(!value.indices().empty())source=input.component(source,{value.indices().begin(),value.indices().end()});
    const auto selected_value=inUse({},[&]{return selected(input.fromThread(source,memberValue(selector,at)));});
    return inUse({},[&]{return project(selected_value,offset,desired);});
  }
  if(value.conversion()) {
    auto child=memberUse(value.conversion()->source,map.base,map.stride);
    auto use=std::make_shared<IndexedSourceUse>(IndexedSourceUse{value,
        IndexedSourceUse::Kind::Conversion,{}, {std::move(child)}});
    expanded_member_uses.emplace(original.get(),use);
    return inUse(std::move(use),[&]{return projectImpl(value,offset,desired);});
  }
  if(first<0 || last<first || std::uint64_t(last)>=node.instanceCount())
    fail("scalar_fold_invalid_index","source template member map exceeds its domain");
  if(first==last)return inUse({},[&]{return project(memberValue(value,std::size_t(first)),offset,desired);});
  // A pointer annotation identifies an allocation; it does not change this
  // scalar address computation or select a different spatial member.
  if(value.threadSelector())
    fail("compact_view_unsupported","template source requires an explicit selector mapping");
  const auto& op=node.operation();
  if(dynamic_cast<const v::CoordinateOp*>(&op)) {
    auto [entry,inserted]=member_coordinates.try_emplace(node.id());
    if(inserted) {
      // The launch binding is an existing finite table. Validate it once
      // per coordinate node, not once per projected output or ancestor.
      const auto count=node.instanceCount();
      for(std::size_t i=0;i<count;++i) {
        const auto x=node.instanceOrigin(i).coordinate_value;
        if(!x || *x>INT64_MAX){entry->second=v::ThreadMembers{};break;}
        entry->second.push_back(*x);
      }
    }
    const auto& coordinates=entry->second;
    const auto segment=coordinates.affineSegment(std::size_t(first));
    if(!segment)fail("compact_view_unsupported","coordinate member domain has no known values");
    if(std::uint64_t(last)>=segment->begin+segment->count) {
      PartitionedRead selection{v::ValueType::matrix(bit(8),{coordinates.size()}),{},true};
      for(std::size_t begin=std::size_t(first);begin<=std::size_t(last);) {
        const auto run=*coordinates.affineSegment(begin);
        selection.regions.push_back({run.begin,run.count,
            MemoryView{selection.type,MemoryViewKind::Slice,{},{{0,0,run.begin,run.count,{}}},{}}});
        begin=run.begin+run.count;
      }
      refinePartitionDomain(selection,position,bit(8));
      fail("compact_view_unsupported","coordinate query crosses affine domains");
    }
    if(bounds(offset)!=std::pair<std::int64_t,std::int64_t>{0,0} ||
        desired.width()!=input.type(value).element().width())
      fail("scalar_fold_invalid_index","coordinate projection requires the full scalar");
    if(const auto* lifted=liftedCoordinate(op)) {
      // One CTA's BlockIndex members are equal (stride 0): without lifting, one u64 literal.
      if(segment->stride)liftRefuse("BlockIndex member domain is not constant");
      return result.graph.convert(unitCoordinate(lifted->axis(),
          coordinates.at(segment->begin),"u64"),desired.toString(),"Convert");
    }
    return result.graph.convert(offsetValue(plus(Offset(integer(coordinates.at(segment->begin))),
        times(plus(position,Offset(-integer(segment->begin))),segment->stride))),desired.toString(),"Convert");
  }
  if(dynamic_cast<const v::LoadOp*>(&op)) {
    const auto [lo,hi]=bounds(offset);const auto width=desired.width()/8;
    if(lo<0 || hi<lo || !width)fail("scalar_fold_invalid_index","invalid template read slice");
    if(const auto* initial=initialByteMap(value.nodeId(),std::size_t(first),std::size_t(last))) {
      if(std::uint64_t(hi)<=initial->initial.bytes && width<=initial->initial.bytes-std::uint64_t(hi)) {
        auto delta=plus(position,Offset(-integer(initial->first)));
        auto relative=initial->period?
            plus(times(remainder(delta,initial->period),initial->stride),
                times(quotient(delta,initial->period),initial->outer_stride)):
            times(delta,initial->stride);
        auto address=plus(Offset(integer(initial->initial.offset)),plus(offset,relative));
        return repAbsoluteCell(initial->initial.allocation,offsetValue(address),desired.toString());
      }
    }
    auto domain=symbolic_read_domains.upper_bound({value.nodeId(),std::size_t(first)});
    if(domain!=symbolic_read_domains.begin()) {
      --domain;
      const auto prototype=memory.symbolic_initial_reads.find({value.nodeId(),value.instanceIndex()});
      const auto region=memory.symbolic_initial_reads.find(domain->first);
      if(domain->first.node==value.nodeId() && std::size_t(last)<domain->second &&
          prototype!=memory.symbolic_initial_reads.end() && region!=memory.symbolic_initial_reads.end() &&
          prototype->second.allocation==region->second.allocation &&
          prototype->second.offset==region->second.offset && prototype->second.bytes==region->second.bytes &&
          node.hasUniformThreadTemplate()) {
        // Positioned members of one run: constant(member) = first constant + step x
        // (member - run start), at member = map.base + map.stride x binder.
        if(const auto* run=positionRun(value.nodeId(),std::size_t(first),std::size_t(last))) {
          const auto& [start,positions]=*run;
          if(std::uint64_t(hi)+width>positions.first->constant.bytes)
            fail("scalar_fold_invalid_index","positioned read slice exceeds its payload");
          return positionedCell(*positions.first,plus(plus(Offset(integer(positions.first->constant.offset)),
              times(plus(position,Offset(-integer(start.instance))),positions.step)),offset),
              desired.toString());
        }
        std::vector<std::shared_ptr<const IndexedSourceUse>> children;
        for(auto operand:input.instanceOperands(value)) {
          operand=input.canonicalInterpretation(operand);
          auto use=bindSourceUse(input,operand,value,map.base,map.stride);
          if(!use)fail("compact_view_unsupported","symbolic read address requires a cross-domain member mapping");
          children.push_back(std::move(use));
        }
        auto use=std::make_shared<IndexedSourceUse>(IndexedSourceUse{value,
            IndexedSourceUse::Kind::Operands,{},std::move(children)});
        // Coverage depends on this query's binder bounds. Unlike a pure
        // operand expansion, do not cache it as valid for every later domain.
        return inUse(std::move(use),[&]{return read(value,offset,desired);});
      }
    }
    const auto* slice=memory.readSlicePattern({value.nodeId(),std::size_t(first)},
        std::uint64_t(lo),std::uint64_t(hi-lo)+width);
    if(slice && std::uint64_t(last)-slice->first.instance<slice->count) {
      const auto delta=plus(position,Offset(-integer(slice->first.instance)));
      auto source_offset=plus(Offset(integer(slice->source_offset)-integer(slice->result_offset)),
          plus(offset,times(delta,slice->byte_stride)));
      const auto base=add(integer(slice->source.instanceIndex()),
          mul(map.base-integer(slice->first.instance),slice->source_instance_stride));
      auto use=memberUse(slice->source,base,mul(map.stride,slice->source_instance_stride));
      return inUse(std::move(use),[&]{return project(slice->source,source_offset,desired);});
    }
    if(const auto* periodic=periodicReadMap(value.nodeId(),std::size_t(first),std::size_t(last),
        std::uint64_t(lo),std::uint64_t(hi-lo)+width)) {
      const auto& p=*periodic->first;
      const auto delta=plus(position,Offset(-integer(p.first.instance)));
      const auto outer=quotient(delta,periodic->period),inner=remainder(delta,periodic->period);
      const auto source_offset=plus(offset,plus(Offset(integer(p.source_offset)-integer(p.result_offset)),
          plus(times(outer,periodic->byte_outer),times(inner,p.byte_stride))));
      const auto members=plus(Offset(integer(p.source.instanceIndex())),
          plus(times(outer,periodic->member_outer),times(inner,p.source_instance_stride)));
      return inContext(std::make_shared<IndexedContext>(IndexedContext{members,memberUse(p.source,0,1)}),
          [&]{return project(p.source,source_offset,desired);});
    }
    auto found=memory.initial_read_families.upper_bound({value.nodeId(),std::size_t(first)});
    if(found!=memory.initial_read_families.begin()) {
      const auto& family=std::prev(found)->second;
      if(family.first.node==value.nodeId() && std::uint64_t(first)>=family.first.instance &&
          std::uint64_t(last)-family.first.instance<family.count) {
        const auto initial=memory.initialRead({value.nodeId(),family.first.instance});
        if(initial && std::uint64_t(hi)<=initial->region.bytes && width<=initial->region.bytes-std::uint64_t(hi)) {
          auto address=plus(Offset(integer(initial->region.offset)),plus(offset,
              times(plus(position,Offset(-integer(family.first.instance))),family.byte_stride)));
          return repAbsoluteCell(initial->region.allocation,offsetValue(address),desired.toString());
        }
      }
    }
    // A read assembled from several stores (AssembledRead) has one slice pattern per piece.
    // A constant byte range across pieces is projected one byte at a time and packed in
    // PTX's little-endian memory order, as indexed packing is. O(width) projections.
    if(lo==hi && width>1 && width<=8 && desired.width()==8*width) {
      bool sliced=true;
      for(std::uint64_t j=0;sliced && j<width;++j) {
        const auto* byte=memory.readSlicePattern({value.nodeId(),std::size_t(first)},std::uint64_t(lo)+j,1);
        sliced=byte && std::uint64_t(last)-byte->first.instance<byte->count;
      }
      if(sliced) {
        const auto target="b"+std::to_string(desired.width());
        auto packed=result.graph.constant(target,"0");
        for(std::uint64_t j=0;j<width;++j) {
          auto byte=result.graph.convert(project(value,plus(offset,Offset(integer(j))),bit(8)),target,"Convert");
          if(j)byte=emitOperation("ShiftLeft",target,{byte,result.graph.constant("u32",std::to_string(8*j))});
          packed=emitOperation("BitOr",target,{packed,byte});
        }
        return cast(packed,desired);
      }
    }
    if(lo==hi)if(auto read=memberAffineRead(value,position,lo,width,desired))return *read;
    // Existing read-from patterns may have distinct writers on adjacent
    // member ranges. Refine the output coordinate at those stored boundaries;
    // do not compare writer expressions or enumerate members to find them.
    SENTRYWARD_AUDIT((void)audit;
      static unsigned witnesses=0;
      if(sentryward::debug::environment("SENTRYWARD_READ_DOMAIN_WITNESS") && witnesses++<4) {
        SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_38);
        if(auto initial=initial_byte_maps.find(value.nodeId());initial!=initial_byte_maps.end()) {
          unsigned shown=0;
          for(const auto& map:initial->second) {
            if(shown++==12)break;
            SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_39);
          }
        }
        auto bucket=memory.read_slice_patterns.upper_bound({value.nodeId(),std::uint64_t(lo)});
        if(bucket!=memory.read_slice_patterns.begin()) {
          --bucket;
          if(bucket->first.first==value.nodeId()) {
            auto p=bucket->second.upper_bound(std::size_t(first));
            if(p!=bucket->second.begin())--p;
            for(unsigned shown=0;p!=bucket->second.end() && shown<12;++p,++shown) {
              const auto& x=p->second;
              SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_40);
            }
          }
        }
      });
    if(lo==hi) {
        PartitionedRead partition{v::ValueType::matrix(bit(8),{node.instanceCount()}),{},true};
        const auto addDomain=[&](std::uint64_t begin,std::uint64_t count) {
          if(begin>std::uint64_t(last) || begin+count<=std::uint64_t(first))return;
          partition.regions.push_back({begin,count,MemoryView{partition.type,MemoryViewKind::Slice,{},
              {{0,0,begin,count,{}}},{}}});
        };
        // Initial-memory reads have the same member-domain contract as
        // writer-backed slices. Refine at stored family boundaries; never
        // enumerate members or extrapolate one family's address stride.
        auto initial=memory.initial_read_families.upper_bound({value.nodeId(),std::size_t(first)});
        if(initial!=memory.initial_read_families.begin())--initial;
        for(;initial!=memory.initial_read_families.end();++initial) {
          const auto& p=initial->second;
          if(p.first.node<value.nodeId())continue;
          if(p.first.node!=value.nodeId() || p.first.instance>std::uint64_t(last))break;
          if(p.first.instance+p.count<=std::uint64_t(first))continue;
          addDomain(p.first.instance,p.count);
        }
        auto domain=memory.read_slice_patterns.upper_bound({value.nodeId(),std::uint64_t(lo)});
        if(domain!=memory.read_slice_patterns.begin()) {
          --domain;
          if(domain->first.first==value.nodeId())for(const auto& [start,p]:domain->second) {
            (void)start;
            if(std::uint64_t(lo)<p.result_offset || std::uint64_t(lo)-p.result_offset>p.bytes ||
                width>p.bytes-(std::uint64_t(lo)-p.result_offset))continue;
            addDomain(p.first.instance,p.count);
          }
        }
        refinePartitionDomain(partition,position,bit(8));
    }
    SENTRYWARD_AUDIT((void)audit;SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_41));
    fail("compact_view_unsupported","template read crosses read-from member or payload domains");
  }
  if(dynamic_cast<const v::ConstantOp*>(&op) || dynamic_cast<const v::InputParameterOp*>(&op))
    return inUse({},[&]{return project(value,offset,desired);});
  if(const auto* opaque=dynamic_cast<const v::OpaqueValueOp*>(&op))unspecified(value,*opaque);
  if(!scalarOperation(op) || !node.hasUniformThreadTemplate())
    fail("compact_view_unsupported","operation requires an explicit template member mapping");
  std::vector<std::shared_ptr<const IndexedSourceUse>> children;
  // One stored operand list. Graph owns the SameDomain/Fixed distinction;
  // do not enumerate members to rediscover it or search operand ancestry.
  for(auto operand:input.instanceOperands(value)) {
    operand=input.canonicalInterpretation(operand);
    auto use=bindSourceUse(input,operand,value,map.base,map.stride);
    if(use)children.push_back(std::move(use));
    else {
      const auto& producer=input.definingNode(operand);
      const auto& pm=producer.instanceThreads();const auto& cm=node.instanceThreads();
      fail("compact_view_unsupported","operand requires an explicit cross-domain member mapping: producer="+
          std::to_string(operand.nodeId())+" members="+std::to_string(pm.size())+
          " step="+(pm.affineStep()?std::to_string(*pm.affineStep()):"nonaffine")+
          " consumer="+std::to_string(value.nodeId())+" members="+std::to_string(cm.size())+
          " step="+(cm.affineStep()?std::to_string(*cm.affineStep()):"nonaffine")+
          " annotations="+std::to_string(bool(operand.conversion()))+","+
          std::to_string(bool(operand.threadSelector()))+","+std::to_string(operand.indices().size()));
    }
  }
  auto use=std::make_shared<IndexedSourceUse>(IndexedSourceUse{value,
      IndexedSourceUse::Kind::Operands,{},std::move(children)});
  expanded_member_uses.emplace(original.get(),use);
  return inUse(std::move(use),[&]{return projectImpl(value,offset,desired);});
}

} // namespace
} // namespace sentryward::value_analysis
