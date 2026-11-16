#pragma once
#include "scalar_fold_lowering_impl.hpp"

namespace sentryward::value_analysis {
namespace {

// A member's output region: the representative's moved by δ. Every byte it
// covers (a strided domain: the last element too) must stay in the signed
// 64-bit offset domain the member's own lowering works in; otherwise
// lifting refuses.
inline auto Lower::shiftRegion(const ConstantRegion& region,const std::optional<OutputDomain>& domain,std::int64_t delta) -> std::uint64_t {
  const auto refuse=[&]{liftRefuse("a member's output region leaves the signed 64-bit offset domain");};
  std::int64_t begin=0,end=0;
  if(region.offset>std::uint64_t(INT64_MAX) || region.bytes>std::uint64_t(INT64_MAX) ||
     __builtin_add_overflow(std::int64_t(region.offset),delta,&begin) || begin<0 ||
     __builtin_add_overflow(begin,std::int64_t(region.bytes),&end))refuse();
  if(domain && !domain->strides.empty() && domain->count) {
    std::int64_t last=0;
    if(__builtin_add_overflow(begin,std::int64_t(region.bytes/domain->count),&last))refuse();
    for(std::size_t axis=0;axis<domain->shape.size();++axis) {
      std::int64_t step=0;
      if(domain->strides[axis]>std::uint64_t(INT64_MAX) ||
         __builtin_mul_overflow(std::int64_t(domain->shape[axis]-1),std::int64_t(domain->strides[axis]),&step) ||
         __builtin_add_overflow(last,step,&last))refuse();
    }
  }
  return static_cast<std::uint64_t>(begin);
}

inline auto Lower::knownAt(v::Value x,unsigned depth,const IndexedSourceUse* use) const -> std::optional<std::int64_t> {
  if(depth>32)return {};
  x=input.resolve(x);
  if(use && use->prototype!=input.canonicalInterpretation(x))return {};
  if(use && (use->kind==IndexedSourceUse::Kind::Exact || use->kind==IndexedSourceUse::Kind::OneHotLiteral))use=nullptr;
  if(use && (use->kind==IndexedSourceUse::Kind::Forward || use->kind==IndexedSourceUse::Kind::Selected))
    return knownAt(use->children.at(0)->prototype,depth+1,use->children.at(0).get());
  if(use && (use->kind==IndexedSourceUse::Kind::BitTestSelect ||
             use->kind==IndexedSourceUse::Kind::MemberTemplate ||
             use->kind==IndexedSourceUse::Kind::Component ||
             use->kind==IndexedSourceUse::Kind::CoordinateMap))return {};
  if(use && use->binding) {
      const auto& binding=*use->binding;
      if(binding.fixed)return {};
      if(binding.initial || binding.producer || binding.region_fragment || binding.addend)return {};
      if(options.unit_lifting && liftedCoordinate(input.definingNode(binding.leaf).operation()))
        NoteUnitSite(UnitSite::FoldBindingKnown);
      const auto decode=[&](std::optional<std::int64_t> coordinate)->std::optional<std::int64_t> {
        if(!coordinate || !binding.power_of_two)return coordinate;
        const auto width=input.type(x).element().width();
        if(*coordinate<0 || *coordinate>=63 || *coordinate>=width)return {};
        return knownInteger(std::int64_t{1}<<*coordinate,input.type(x).element());
      };
      if(!binding.stride && !binding.outer_stride && binding.quotient_terms.empty())return decode(binding.base);
      const auto [lo,hi]=bounds(indexed_context->index);
      if(lo==hi && lo>=0) {
        return decode(indexedBindingValue(binding,static_cast<std::uint64_t>(lo)));
      }
      return {};
    }
  if(!x.indices().empty() || x.threadSelector() || !input.type(x).isScalar())return {};
  if(x.conversion()){
    const auto& conversion=*x.conversion();
    const auto& source_type=input.type(conversion.source);
    const auto& target_type=input.type(x);
    if(!source_type.isScalar() || !source_type.element().isBitLike() ||
       !target_type.element().isBitLike())return {};
    if(conversion.spec->kind==v::TypeConversionKind::Convert &&
       conversion.spec->attributes!=v::ConversionAttributes{})return {};
    const auto* child=use?use->children.at(0).get():nullptr;
    auto value=knownAt(child?child->prototype:conversion.source,depth+1,child);
    return value?knownInteger(*value,target_type.element()):std::nullopt;
  }
  const auto& n=input.definingNode(x);
  if(const auto* c=dynamic_cast<const v::ConstantOp*>(&n.operation())){
    auto value=literal(c->literal());
    return value?knownInteger(*value,input.type(x).element()):std::nullopt;
  }
  if(dynamic_cast<const v::CoordinateOp*>(&n.operation())){
    if(liftedCoordinate(n.operation())){NoteUnitSite(UnitSite::FoldKnown);return {};}
    if(auto val=input.instanceOrigin(x).coordinate_value)
      if(*val<=std::uint64_t(INT64_MAX))return knownInteger(std::int64_t(*val),input.type(x).element());
  }
  if(dynamic_cast<const v::BinaryOp*>(&n.operation()) && input.type(x).element().isBitLike()) {
    std::vector<equivalence::ConstantValue> operands;
    std::size_t operand_index=0;
    for(auto operand:input.instanceOperands(x)) {
      auto value=knownAt(operand,depth+1,use?use->children.at(operand_index).get():nullptr);
      ++operand_index;
      if(!value)return {};
      operands.push_back({input.type(operand).element().toString(),std::to_string(*value)});
    }
    if(auto value=equivalence::constantOperation(std::string(n.operation().name()),
        input.type(x).element().toString(),attributes(n.operation()),operands)) {
      auto parsed=literal(value->literal);
      return parsed?knownInteger(*parsed,input.type(x).element()):std::nullopt;
    }
  }
  return {};
}

// A whole-byte matrix constant whose bits are all zero (the frontend's TMA
// out-of-bounds zero fill): every byte of it reads as zero.
inline auto Lower::zeroMatrixLiteral(const v::Value& value) const -> bool {
  if(value.conversion() || !value.indices().empty() || value.threadSelector() || value.pointer())return false;
  const auto& type=input.type(value);
  if(!type.isMatrix() || !type.hasStaticShape() || type.element().isParameterized() || type.bitSize()%8)return false;
  const auto* constant=dynamic_cast<const v::ConstantOp*>(&input.definingNode(value).operation());
  return constant && !constant->pointer() && constant->literal()=="0";
}

inline auto Lower::zeroRead(v::Value value,unsigned depth) -> bool {
  value=input.canonicalInterpretation(value);
  if(zeroMatrixLiteral(value))return true;
  if(depth>=8 || value.conversion() || !value.indices().empty() || value.threadSelector() || value.pointer() ||
     !dynamic_cast<const v::LoadOp*>(&input.definingNode(value).operation()))return false;
  const MemoryInstance key{value.nodeId(),value.instanceIndex()};
  if(const auto found=zero_reads.find(key);!depth && found!=zero_reads.end())return found->second;
  // An indexed region projects its use's prototype; only an Exact use keeps
  // that prototype for every member (CoordinateMap only remaps coordinates).
  const auto zeroView=[&](const MemoryView& view) {
    if(const auto& source=view.indexed_source) {
      auto use=source->root;
      while(use && use->kind==IndexedSourceUse::Kind::CoordinateMap && use->children.size()==1)use=use->children[0];
      return !source->zero_mask && use && use->kind==IndexedSourceUse::Kind::Exact && zeroMatrixLiteral(use->prototype);
    }
    return !view.blocks.empty() && std::all_of(view.blocks.begin(),view.blocks.end(),[&](const ViewBlock& block) {
      const auto* source=block.source<view.sources.size()?std::get_if<v::Value>(&view.sources[block.source]):nullptr;
      return source && zeroRead(*source,depth+1);
    });
  };
  bool zero=false;
  if(!memory.symbolic_initial_reads.contains(key) && !memory.initialRead(key))
    if(const auto found=memory.normalizedRead(key)) {
      if(const auto* r=std::get_if<const v::Value*>(&*found))zero=zeroRead(**r,depth+1);
      else if(const auto* r=std::get_if<const MemoryView*>(&*found))zero=zeroView(**r);
      else if(const auto* r=std::get_if<const PartitionedRead*>(&*found))
        zero=!(*r)->regions.empty() && std::all_of((*r)->regions.begin(),(*r)->regions.end(),
            [&](const ReadRegion& region){return zeroView(region.view);});
    }
  if(!depth)zero_reads.emplace(key,zero);
  return zero;
}

// D_in of a tcgen05 MMA whose product terms are all +0: the accumulator when
// enable-input-d is a known nonzero (an unguarded D load), +0 when it is a
// known zero. A guarded D load (unknown predicate) keeps the reduction path.
// `cell` is the byte stride of D_in elements, as in each caller's reduction.
inline auto Lower::zeroProductMma(const v::OperandsRef& args,bool accumulate,std::size_t accumulator,
    const Offset& index,std::int64_t cell,const v::ElementType& dtype,const v::ElementType& element) -> std::optional<s::Ref> {
  if(spatial_count || !(zeroRead(args.at(0)) || zeroRead(args.at(1))))return std::nullopt;
  if(!accumulate)return cast(result.graph.constant(dtype.toString(),"0"),element);
  const auto* load=dynamic_cast<const v::LoadOp*>(&input.definingNode(args.at(accumulator)).operation());
  if(!load || load->attributes().guarded)return std::nullopt;
  return cast(project(args.at(accumulator),times(index,cell),dtype),element);
}

inline auto Lower::selected(v::Value value) -> v::Value {
  if(!value.threadSelector())return value;
  const auto thread=known(selected(*value.threadSelector()));
  if(!thread || *thread<0) {
    const auto selector=*value.threadSelector();
    const auto fixed=specialization.known(selector);
    fail("scalar_fold_thread_selector","thread selector is not constant; selector_node="+
        std::to_string(selector.nodeId())+", selector_instance="+std::to_string(selector.instanceIndex())+
        ", operation="+std::string(input.definingNode(selector).operation().name())+
        ", use_kind="+(indexed_context && indexed_context->use?
            std::to_string(static_cast<int>(indexed_context->use->kind)):"none")+
        ", fixed_instance_value="+(fixed?fixed->literal:"unknown"));
  }
  if(value.conversion()) {
    const auto& conversion=*value.conversion();
    auto source=selected(input.fromThread(conversion.source,*value.threadSelector()));
    auto result=conversion.spec->kind==v::TypeConversionKind::Bitcast
        ?input.bitcast(source,conversion.spec->target)
        :input.convert(source,conversion.spec->target,conversion.spec->attributes);
    if(!value.indices().empty())result=input.component(result,{value.indices().begin(),value.indices().end()});
    if(value.pointer())result=input.asPointer(result,*value.pointer());
    return result;
  }
  const auto& n=input.definingNode(value);
  const auto cta=input.instanceOrigin(value).cta;
  if(n.origin().cta==cta) {
    const auto j=n.memberIndex(static_cast<std::uint64_t>(*thread));
    if(j) {
      auto result=n.result(value.resultIndex(),*j);
      // A ThreadForward instance is that source thread's own value.
      if(dynamic_cast<const v::ThreadForwardOp*>(&n.operation())) {
        if(!value.indices().empty() || value.pointer())
          fail("scalar_fold_thread_selector","a value forwarded from another thread carries component indices or a pointer annotation");
        return selected(input.forwarded(result));
      }
      if(!value.indices().empty())result=input.component(result,{value.indices().begin(),value.indices().end()});
      if(value.pointer())result=input.asPointer(result,*value.pointer());
      return result;
    }
  }
  fail("scalar_fold_thread_selector","selected thread "+std::to_string(*thread)+
      " has no matching value instance; source thread="+
      (n.origin().thread?std::to_string(*n.origin().thread):"none")+
      ", members="+std::to_string(n.instanceCount()));
}

inline auto Lower::initial(const InitialMemorySymbol& x,const Offset& offset,const v::ElementType& type) -> s::Ref {
  auto [lo,hi]=bounds(offset);
  const auto bytes=type.width()/8;
  if(lo<0 || hi<0 || std::uint64_t(hi)>x.region.bytes || bytes>x.region.bytes-std::uint64_t(hi))
    fail("scalar_fold_invalid_index","initial-memory slice exceeds its recorded region");
  return repAbsoluteCell(x.region.allocation,offsetValue(plus(offset,Offset(integer(x.region.offset)))),type.toString());
}

inline auto Lower::intervalsFor(const MemoryView& view) -> const std::vector<ViewInterval>& {
  auto [entry,inserted]=view_intervals.try_emplace(&view);
  if(inserted) {
    for(std::size_t i=0;i<view.blocks.size();++i) {
      const auto& block=view.blocks[i];
      unsigned __int128 end=static_cast<unsigned __int128>(block.result_offset)+block.bytes;
      for(const auto& axis:block.axes)if(axis.count) {
        end+=static_cast<unsigned __int128>(axis.count-1)*axis.result_stride;
        end=std::min(end,static_cast<unsigned __int128>(UINT64_MAX));
      }
      entry->second.push_back({block.result_offset,end>UINT64_MAX?UINT64_MAX:static_cast<std::uint64_t>(end),0,i});
    }
    auto& intervals=entry->second;
    std::sort(intervals.begin(),intervals.end(),[](const auto& a,const auto& b){return a.begin<b.begin;});
    std::uint64_t end=0;
    for(auto& interval:intervals)interval.prefix_end=end=std::max(end,interval.end);
  }
  return entry->second;
}

inline auto Lower::coveringBlocks(const MemoryView& view,std::uint64_t begin,std::uint64_t end) -> std::vector<std::size_t> {
  const auto& intervals=intervalsFor(view);
  std::vector<std::size_t> candidates;
  auto it=std::upper_bound(intervals.begin(),intervals.end(),begin,
      [](auto value,const auto& interval){return value<interval.begin;});
  while(it!=intervals.begin()) {
    --it;
    if(it->prefix_end<end)break;
    if(it->end>=end)candidates.push_back(it->block);
  }
  std::sort(candidates.begin(),candidates.end());
  return candidates;
}

// Plan an affine read interval without constructing expression nodes.
inline auto Lower::readSpan(v::Value value,const Offset& offset,std::uint64_t stride,
                       const v::ElementType& type,std::uint64_t count,unsigned depth) -> std::uint64_t {
  SENTRYWARD_LEDGER(ledger->add("read_span_calls"));
  SENTRYWARD_AUDIT(if(count>1 && (depth>=32 || spatial_count))
    audit->event("SPAN_DEGRADED",spatial_count?"spatial_projection_disables_span":"read_span_depth_limit",
        count,1,value.nodeId()));
  if(count<=1 || depth>=32 || spatial_count)return 1;
  value=input.canonicalInterpretation(value);
  const auto single=[&]([[maybe_unused]] const char* reason) {
    SENTRYWARD_AUDIT(audit->event("SPAN_DEGRADED",reason,count,1,value.nodeId()));
    SENTRYWARD_AUDIT({
      (void)audit;
      static thread_local std::set<std::string> reported;
      const auto& op=input.definingNode(value).operation();
      const auto label=std::string(reason)+":"+std::string(op.name());
      if(reported.size()<16 && reported.insert(label).second)
        SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_0);
    });
    return std::uint64_t{1};
  };
  if(value.threadSelector() || !value.indices().empty())return single("read_span_value_selection");
  const auto [lo,hi]=bounds(offset);
  if(lo<0 || hi<lo)return 1;
  const auto width=type.width()/8;
  const auto fit=[&](std::uint64_t bytes,std::uint64_t start) {
    if(start>bytes || width>bytes-start)return std::uint64_t{1};
    return stride?std::min(count,1+(bytes-start-width)/stride):count;
  };
  const auto& value_type=input.type(value);
  if(!value_type.hasStaticShape() || value_type.element().isParameterized() ||
     value_type.bitSize()%8)return single("read_span_unresolved_type");
  // A computed matrix has a known numerical element domain. A packed scalar
  // only has a byte domain: its independent numeric fields need their own
  // source mapping before treating them as one parametric numerical domain.
  const auto& operation=input.definingNode(value).operation();
  // Use the same identity-transfer rule as projectImpl. A layout-preserving
  // native copy must not turn a known matrix domain into K scalar queries.
  if(const auto* copy=dynamic_cast<const v::NativeMatrixOp*>(&operation);
     copy && copy->opcode()==v::NativeMatrixOpcode::TcgenCopyValue) {
    const auto source=input.instanceOperands(value).at(0);
    if(input.type(source)==value_type)
      return readSpan(source,offset,stride,type,count,depth+1);
  }
  if(value_type.isMatrix() && (value.conversion() || scalarOperation(operation) ||
      dynamic_cast<const v::MatMulOp*>(&operation) ||
      dynamic_cast<const v::TcgenMmaOp*>(&operation)))
    return fit(value_type.bitSize()/8,std::uint64_t(hi));
  // An all-zero matrix constant is one span: its pieces project to zero (projectImpl).
  if(zeroMatrixLiteral(value))return fit(value_type.bitSize()/8,std::uint64_t(hi));
  if(value.conversion() || !dynamic_cast<const v::LoadOp*>(&operation))return single("read_span_not_matrix_or_load");
  if(auto initial=memory.initialRead({value.nodeId(),value.instanceIndex()}))
    return fit(initial->region.bytes,std::uint64_t(hi));
  const auto found=memory.normalizedRead({value.nodeId(),value.instanceIndex()});
  if(!found)return single("read_span_missing_substitution");
  if(const auto* ref=std::get_if<const v::Value*>(&*found))
    return readSpan(**ref,offset,stride,type,count,depth+1);
  if(const auto* initial=std::get_if<const InitialMemorySymbol*>(&*found))
    return fit((*initial)->region.bytes,std::uint64_t(hi));
  const MemoryView* selected_view=nullptr;
  if(const auto* direct=std::get_if<const MemoryView*>(&*found))selected_view=*direct;
  else if(const auto* split=std::get_if<const PartitionedRead*>(&*found)) {
    for(const auto& region:(*split)->regions)
      if(std::uint64_t(lo)>=region.begin && std::uint64_t(hi)+width<=region.begin+region.bytes) {
        selected_view=&region.view;break;
      }
  }
  if(!selected_view) {
    if(const auto* split=std::get_if<const PartitionedRead*>(&*found))
      if(const auto* grid=partitionGrid(**split,value,type))
        return fit(grid->blocks.front().bytes*grid->indexed_source->count,std::uint64_t(hi));
    // A residue map certifies this column, not adjacent reduction columns.
    // Reuse/precompute its proof but do not manufacture a wider read span.
    if(const auto* split=std::get_if<const PartitionedRead*>(&*found))
      if(!(*split)->template_domains)(void)partitionLattice(**split,value,offset,type);
    return single("read_span_crosses_source_domains");
  }
  const auto& view=*selected_view;
  if(view.indexed_source) {
    const auto& block=view.blocks.at(0);
    if(block.axes.size()>1) {
      auto range=offset;Offset iteration;iteration.terms[domains.size()]=integer(stride);
      range=plus(range,iteration);domains.push_back(count);
      const auto mapped=viewOffset(block,range,type);domains.pop_back();
      return mapped?count:single("read_span_crosses_source_domains");
    }
    const auto total=block.bytes*view.indexed_source->count;
    const auto unit=std::min<std::uint64_t>(width,block.bytes);
    if(!unit || std::max<std::uint64_t>(width,block.bytes)%unit ||
       alignment(plus(offset,Offset(-integer(block.result_offset))))%unit || stride%unit)return 1;
    return fit(total,std::uint64_t(hi)-block.result_offset);
  }
  const auto candidates=coveringBlocks(view,std::uint64_t(lo),std::uint64_t(hi)+width);
  if(candidates.empty())return single("read_span_no_covering_block");
  const auto& block=view.blocks[candidates.front()];
  if(!block.axes.empty()) {
    if(view.blocks.size()!=1)return 1;
    const auto& source=view.sources.at(block.source);
    const auto* initial=std::get_if<InitialMemorySymbol>(&source);
    const auto extent=initial?initial->region.bytes:input.type(std::get<v::Value>(source)).bitSize()/8;
    auto range=offset;
    Offset iteration;iteration.terms[domains.size()]=integer(stride);
    range=plus(range,iteration);
    domains.push_back(count);
    const auto mapped=viewOffset(block,range,type);
    bool valid=false;
    if(mapped) {
      const auto [sl,sh]=bounds(*mapped);
      valid=sl>=0 && sh>=sl && std::uint64_t(sh)<=extent &&
          width<=extent-std::uint64_t(sh);
    }
    domains.pop_back();
    return valid?count:single("read_span_axis_mapping");
  }
  auto span=fit(block.bytes,std::uint64_t(hi)-block.result_offset);
  // A new overlapping block may change source precedence.
  const auto& intervals=intervalsFor(view);
  auto next=std::upper_bound(intervals.begin(),intervals.end(),std::uint64_t(lo),
      [](auto x,const auto& interval){return x<interval.begin;});
  if(stride && next!=intervals.end()) {
    if(next->begin<std::uint64_t(hi)+width)return 1;
    span=std::min(span,1+(next->begin-std::uint64_t(hi)-width)/stride);
  }
  auto source_offset=plus(offset,Offset(add(integer(block.source_offset),-integer(block.result_offset))));
  if(const auto* initial=std::get_if<InitialMemorySymbol>(&view.sources.at(block.source))) {
    const auto [sl,sh]=bounds(source_offset);
    if(sl<0 || sh<sl)return 1;
    return std::min(span,fit(initial->region.bytes,std::uint64_t(sh)));
  }
  return std::min(span,readSpan(std::get<v::Value>(view.sources.at(block.source)),
      source_offset,stride,type,span,depth+1));
}

inline auto Lower::viewOffset(const ViewBlock& block,const Offset& offset,const v::ElementType& type) -> std::optional<Offset> {
  auto pos=plus(offset,Offset(-integer(block.result_offset)));
  auto [lo,hi]=bounds(pos);if(lo<0 || hi<0)return {};
  auto src=Offset(integer(block.source_offset));
  std::vector<ViewAxis> axes=block.axes;
  // Outermost first. A one-element axis (the explicit inner axis of a row grid whose block
  // spans its whole row: result stride = the row stride) comes after the other axes of its
  // stride, where its coordinate is 0. Placed first it would read the row coordinate as its
  // own; for queries inside its single element both orders agree.
  std::stable_sort(axes.begin(),axes.end(),[](const auto& a,const auto& b){
    return a.result_stride!=b.result_stride?a.result_stride>b.result_stride:(a.count!=1 && b.count==1);});
  try{
    for(const auto& axis:axes){
      auto coord=quotient(pos,axis.result_stride);
      auto [cl,ch]=bounds(coord);
      if(cl<0 || std::uint64_t(ch)>=axis.count)return {};
      pos=remainder(pos,axis.result_stride);
      src=plus(src,times(coord,integer(axis.source_stride)));
    }
  }catch(const Failure& e){
    if(e.code=="scalar_fold_non_affine_projection")return {};
    throw;
  }
  auto [pl,ph]=bounds(pos);auto width=type.width()/8;
  if(pl<0 || ph<0 || std::uint64_t(ph)>block.bytes || width>block.bytes-std::uint64_t(ph))return {};
  return plus(src,pos);
}

inline auto Lower::view(const MemoryView& view,const Offset& offset,const v::ElementType& type) -> s::Ref {
  if(view.sources.size()!=1 || view.blocks.size()!=1 || view.blocks[0].source!=0)
    fail("compact_view_unsupported","a view must contain one source rule; use disjoint result domains");
  if(view.indexed_source)return indexedView(view,offset,type);
  if(view.blocks.size()<=4) {
    for(const auto& b:view.blocks)if(auto r=viewBlock(view,b,offset,type))return *r;
  } else {
    const auto [lo,hi]=bounds(offset);
    std::vector<std::size_t> candidates;
    if(lo>=0 && hi>=lo)candidates=coveringBlocks(view,std::uint64_t(lo),std::uint64_t(hi)+type.width()/8);
    for(auto index:candidates)if(auto r=viewBlock(view,view.blocks[index],offset,type))return *r;
  }
  fail("compact_view_unsupported","read spans independent source domains or has unsupported alignment");
}

inline auto Lower::mappedCoordinate(const IndexedValueBinding& binding,const Offset& index) -> Offset {
  auto position=Offset(binding.base);
  if(binding.period) {
    position=plus(position,times(remainder(index,binding.period),binding.stride));
    position=plus(position,times(quotient(index,binding.period),binding.outer_stride));
  } else position=plus(position,times(index,binding.stride));
  for(const auto& term:binding.quotient_terms)
    position=plus(position,times(quotient(index,term.divisor),term.coefficient));
  if(binding.quotient_terms.empty())return position;
  // Use independent radix digits to keep bounds precise. Work is over the
  // stored coordinate coefficients, never over represented source members.
  auto base=binding.base;
  std::vector<std::pair<Offset,std::int64_t>> digits;
  const auto append=[&](Offset digit,std::int64_t coefficient,std::int64_t upper) {
    if(!coefficient)return;
    if(coefficient<0) {
      base=add(base,mul(coefficient,upper));
      digit=plus(Offset(upper),times(std::move(digit),-1));coefficient=mul(coefficient,-1);
    }
    digits.emplace_back(std::move(digit),coefficient);
  };
  auto divisor=binding.period?binding.period:1;
  auto coefficient=binding.period?binding.outer_stride:binding.stride;
  if(binding.period)append(remainder(index,binding.period),binding.stride,integer(binding.period-1));
  for(const auto& term:binding.quotient_terms) {
    if(!term.divisor || term.divisor%divisor)return position;
    const auto ratio=term.divisor/divisor;
    append(remainder(quotient(index,divisor),ratio),coefficient,integer(ratio-1));
    coefficient=add(mul(coefficient,integer(ratio)),term.coefficient);divisor=term.divisor;
  }
  const auto [lo,hi]=bounds(index);
  if(lo<0)return position;
  append(quotient(index,divisor),coefficient,hi/integer(divisor));
  if(base<0)return position;
  position=Offset(base);
  for(const auto& [digit,scale]:digits)position=plus(position,times(digit,scale));
  return position;
}

inline auto Lower::indexedView(const MemoryView& view,const Offset& offset,const v::ElementType& type) -> s::Ref {
  if(spatial_count)fail("compact_view_unsupported","simultaneous spatial and indexed-source binding is not supported");
  const auto& pattern=*view.indexed_source;
  const auto& block=view.blocks.at(0);
  const auto result_step=block.axes.empty()?block.bytes:block.axes.front().result_stride;
  const auto width=type.width()/8;
  const auto unit=std::min<std::uint64_t>(width,block.bytes);
  auto relative=plus(offset,Offset(-integer(block.result_offset)));
  const auto [lo,hi]=bounds(relative);
  const bool grid=block.axes.size()>1;
  std::uint64_t extent=block.bytes;
  for(const auto& axis:block.axes)extent+=(axis.count-1)*axis.result_stride;
  if(!unit || std::max<std::uint64_t>(width,block.bytes)%unit || alignment(relative)%unit ||
     lo<0 || hi<lo || !result_step ||
     std::uint64_t(hi)>extent || width>extent-std::uint64_t(hi))
    fail("compact_view_unsupported","indexed read requires aligned, integer-multiple access widths within its domain");
  if(!grid && result_step!=block.bytes) {
    const auto [pl,ph]=bounds(remainder(relative,result_step));
    if(pl<0 || ph<pl || std::uint64_t(ph)>block.bytes || width>block.bytes-std::uint64_t(ph))
      fail("compact_view_unsupported","indexed read crosses disjoint template residues");
  }
  const auto part=[&](const Offset& at,const v::ElementType& desired) {
    auto index=quotient(at,result_step);
    auto source_offset=plus(Offset(integer(block.source_offset)),remainder(at,result_step));
    if(grid) {
      auto pos=at;index=Offset(0);
      source_offset=Offset(integer(block.source_offset));
      for(auto it=block.axes.rbegin();it!=block.axes.rend();++it) {
        const auto coordinate=quotient(pos,it->result_stride);
        const auto [low,high]=bounds(coordinate);
        if(low<0 || high<low || std::uint64_t(high)>=it->count)
          fail("compact_view_unsupported","indexed row coordinate exceeds its domain");
        index=plus(times(index,integer(it->count)),coordinate);
        source_offset=plus(source_offset,times(coordinate,integer(it->source_stride)));
        pos=remainder(pos,it->result_stride);
      }
      const auto [low,high]=bounds(pos);
      if(low<0 || high<low || std::uint64_t(high)>block.bytes ||
          desired.width()/8>block.bytes-std::uint64_t(high))
        fail("compact_view_unsupported","indexed read crosses row residues");
      source_offset=plus(source_offset,pos);
    }
    auto r=inContext(std::make_shared<const IndexedContext>(IndexedContext{index,pattern.root}),
        [&]{return project(pattern.prototype,source_offset,desired);});
      if(pattern.zero_mask) {
        const auto& mask=*pattern.zero_mask;
        const auto row=quotient(index,mask.columns),col=remainder(index,mask.columns);
        // Nonnegative index arithmetic avoids unsigned subtraction below
        // the diagonal. Projection still checks every source's full bounds;
        // unsupported extrapolations are not silently hidden by the mask.
        auto lhs=row,rhs=col;
        if(mask.diagonal>=0)rhs=plus(rhs,Offset(mask.diagonal));
        else lhs=plus(lhs,Offset(-mask.diagonal));
        const auto pred=emitOperation("Compare","pred",{offsetValue(lhs),offsetValue(rhs)},
            mask.lower?"predicate=ge":"predicate=lt");
        r=emitOperation("Select",desired.toString(),
            {pred,r,result.graph.constant(desired.toString(),"0")});
      }
    return r;
  };
  if(width<=block.bytes)return part(relative,type);
  if(width>8)fail("compact_view_unsupported","indexed scalar packing exceeds 64 bits");
  const auto target="b"+std::to_string(type.width());
  auto packed=result.graph.constant(target,"0");
  for(std::uint64_t j=0;j<width/block.bytes;++j) {
    auto value=part(plus(relative,Offset(integer(j*block.bytes))),bit(unsigned(8*block.bytes)));
    value=result.graph.convert(value,target,"ZeroExtend");
    if(j)value=emitOperation("ShiftLeft",target,
        {value,result.graph.constant("u32",std::to_string(8*j*block.bytes))});
    packed=emitOperation("BitOr",target,{packed,value});
  }
  return cast(packed,type);
}

// The region of `split` holding every byte the query can read, if any.
inline auto Lower::coveringRegion(const PartitionedRead& split,const Offset& offset,const v::ElementType& type) -> const ReadRegion* {
  const auto [lo,hi]=bounds(offset);
  if(lo>=0 && hi>=lo)for(const auto& region:split.regions)
    if(std::uint64_t(lo)>=region.begin && std::uint64_t(hi)<=region.begin+region.bytes &&
       type.width()/8<=region.begin+region.bytes-std::uint64_t(hi) &&
       (!split.template_domains || viewOffset(region.view.blocks.at(0),offset,type)))return &region;
  return nullptr;
}

} // namespace
} // namespace sentryward::value_analysis
