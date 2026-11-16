#pragma once
#include "scalar_fold_lowering_impl.hpp"

namespace sentryward::value_analysis {
namespace {

// The K domains of a template-domain partition: one interval or AP per
// stored region the contraction crosses; none unless they cover K.
inline auto Lower::regionDomains(const PartitionedRead& split,
    const Offset& offset,std::uint64_t stride,const v::ElementType& type,std::uint64_t count) -> std::optional<std::vector<algorithm::AffineDomain>> {
  const auto [lo,hi]=bounds(offset);
  std::vector<algorithm::AffineDomain> result;
  std::uint64_t covered=0;
  const auto width=type.width()/8;
  for(const auto& region:split.regions) {
    // Filter whole stored row groups before interpreting their local axes.
    // Different source templates may own disjoint row intervals.
    const auto query_end=static_cast<unsigned __int128>(hi)+
        static_cast<unsigned __int128>(count-1)*stride+width;
    if(query_end<=region.begin || std::uint64_t(lo)>=region.begin+region.bytes)continue;
    auto b=region.view.blocks.at(0);
    // This is a coverage plan, not a rewrite of the source mapping. Adjacent
    // payload repetitions cover one interval even when their source stride
    // differs. Keep the original view for projection of the Fold body.
    // Collapse only inside this one read-result region/occurrence.
    std::erase_if(b.axes,[](const auto& axis){return axis.count==1;});
    while(!b.axes.empty() && b.axes.front().result_stride==b.bytes) {
      const auto count=b.axes.front().count;
      if(!count || b.bytes>UINT64_MAX/count)return {};
      b.bytes*=count;b.axes.erase(b.axes.begin());
    }
    auto local_offset=offset;auto local_lo=lo,local_hi=hi;
    while(b.axes.size()>1) {
      const auto outer=b.axes.back();
      if(!outer.result_stride || !outer.count)return {};
      const auto origin=b.result_offset-b.result_offset%outer.result_stride;
      const auto relative=plus(local_offset,Offset(-integer(origin)));
      // The overlap filter admits a query that begins in an earlier
      // region's rows. No row of this region then holds every binder value,
      // the same no-plan outcome as a row past the last one.
      if(bounds(relative).first<0)return {};
      const auto row=bounds(quotient(relative,outer.result_stride));
      if(row.first<0 || row.second<row.first || std::uint64_t(row.second)>=outer.count)return {};
      local_offset=remainder(relative,outer.result_stride);
      b.result_offset-=origin;
      std::tie(local_lo,local_hi)=bounds(local_offset);
      if(local_lo<0 || static_cast<unsigned __int128>(local_hi)+
          static_cast<unsigned __int128>(count-1)*stride+width>outer.result_stride)return {};
      b.axes.pop_back();
    }
    if(b.axes.size()>1 || b.bytes<width)return {};
    const auto lo=local_lo,hi=local_hi;
    if(b.axes.empty()) {
      // A single contiguous source region can cover a consecutive K range.
      const auto lower=lo>=integer(b.result_offset)?0:
          (b.result_offset-std::uint64_t(lo)+stride-1)/stride;
      const auto end=b.result_offset+b.bytes-width;
      if(std::uint64_t(hi)>end || lower>=count)continue;
      const auto upper=std::min(count-1,(end-std::uint64_t(hi))/stride);
      if(lower<=upper){result.push_back({lower,1,upper-lower+1});covered+=upper-lower+1;}
      continue;
    }
    const auto& axis=b.axes[0];
    if(!axis.count || !axis.result_stride)return {};
    const auto residue=bounds(remainder(local_offset,axis.result_stride));
    if(residue.first!=residue.second)return {};
    // One contraction remains within one repetition period, even while the
    // output binder selects many periods. Intersect the contiguous payload
    // with K as one interval, not one AP per payload byte/element.
    const auto r=std::uint64_t(residue.first);
    const auto start=b.result_offset%axis.result_stride;
    if(b.bytes<=axis.result_stride-start &&
       static_cast<unsigned __int128>(r)+static_cast<unsigned __int128>(count-1)*stride+width<=axis.result_stride) {
      const auto lower=r>=start?0:(start-r+stride-1)/stride;
      if(start+b.bytes>=r+width) {
        const auto upper=std::min(count-1,(start+b.bytes-width-r)/stride);
        const auto first=static_cast<unsigned __int128>(lo)+static_cast<unsigned __int128>(lower)*stride;
        const auto last=static_cast<unsigned __int128>(hi)+static_cast<unsigned __int128>(upper)*stride+width;
        const auto end=static_cast<unsigned __int128>(b.result_offset)+
            static_cast<unsigned __int128>(axis.count-1)*axis.result_stride+b.bytes;
        if(lower<=upper && first>=b.result_offset && last<=end) {
          result.push_back({lower,1,upper-lower+1});covered+=upper-lower+1;continue;
        }
      } else continue;
    }
    const auto inner=(std::uint64_t(residue.first)+axis.result_stride-
        b.result_offset%axis.result_stride)%axis.result_stride;
    // If K changes the residue, intersect its exact progression with each
    // payload position. The loop is over bytes of one stored fragment, not
    // over threads, matrix elements, or reduction members.
    const bool varying_residue=stride%axis.result_stride!=0;
    const auto positions=varying_residue?b.bytes-width+1:1;
    for(std::uint64_t p=0;p<positions;++p) {
      const auto within=varying_residue?p:inner;
      if(within>b.bytes-width)continue;
      const algorithm::AffineDomain addresses{b.result_offset+within,axis.result_stride,axis.count};
      const auto intersection=algorithm::IntersectAffine(
          {std::uint64_t(lo),stride,count},addresses);
      if(!intersection.count)continue;
      const auto end=addresses.last()+b.bytes-within-width;
      if(std::uint64_t(hi)>end)continue;
      const auto upper=std::min(count-1,(end-std::uint64_t(hi))/stride);
      if(intersection.left>upper)continue;
      const auto step=intersection.count>1?intersection.left_step:1;
      const auto n=std::min(intersection.count,1+(upper-intersection.left)/step);
      result.push_back({intersection.left,step,n});covered+=n;
    }
  }
  if(covered!=count)return {};
  return result;
}

// Pull a reduction's index progression through stored read domains. This
// never asks whether the source expressions have a common mathematical form.
inline auto Lower::reductionDomains(
    v::Value value,const Offset& offset,std::uint64_t stride,
    const v::ElementType& type,std::uint64_t count) -> std::optional<std::vector<algorithm::AffineDomain>> {
  SENTRYWARD_LEDGER(ledger->add("reduction_domain_queries"));
  const auto [lo,hi]=bounds(offset);
  if(lo<0 || hi<lo || !stride || !count || !type.width() || type.width()%8)return {};
  const auto read=memory.normalizedRead({value.nodeId(),value.instanceIndex()});
  const auto* split=read?std::get_if<const PartitionedRead*>(&*read):nullptr;
  if(split)if(const auto* grid=partitionGrid(**split,value,type)) {
    const auto end=static_cast<unsigned __int128>(hi)+
        static_cast<unsigned __int128>(count-1)*stride+type.width()/8;
    if(end<=grid->type.bitSize()/8)
      return std::vector<algorithm::AffineDomain>{{0,1,count}};
  }
  if(!split || !(*split)->template_domains) {
    if(readSpan(value,offset,stride,type,count)==count)
      return std::vector<algorithm::AffineDomain>{{0,1,count}};
    if(split)if(const auto* periodic=periodicPartition(**split))return regionDomains(*periodic,offset,stride,type,count);
    return {};
  }
  if(auto planned=regionDomains(**split,offset,stride,type,count))return planned;
  if(const auto* periodic=periodicPartition(**split))return regionDomains(*periodic,offset,stride,type,count);
  return {};
}

// The exact bits of a value when they are a launch-time constant, after its
// lane selection is resolved and same-width bitcasts are crossed.
inline auto Lower::constantBits(v::Value value) -> std::optional<equivalence::ConstantValue> {
  for(unsigned depth=0;depth<kConstantBitsDepth;++depth) {
    value=specialization.selected(input.canonicalInterpretation(value));
    if(value.threadSelector() || value.pointer() || !value.indices().empty())return {};
    if(!value.conversion())return specialization.known(value);
    const auto& c=*value.conversion();
    if(c.spec->kind!=v::TypeConversionKind::Bitcast || input.type(c.source).bitSize()!=input.type(value).bitSize())
      return {};
    value=c.source;
  }
  return {};
}

// A representative slice when every slice is a whole value of one type whose
// bits are the same constant; equal bits of equal types are equal values.
inline auto Lower::uniformConstant(std::span<const IndexedSourceSlice> slices,std::uint64_t width) -> std::optional<v::Value> {
  if(slices.empty())return {};
  const auto& type=input.type(slices.front().value);
  if(!type.isScalar() || type.bitSize()!=8*width)return {};
  std::optional<equivalence::ConstantValue> bits;
  for(const auto& slice:slices) {
    if(slice.offset || slice.bytes!=width || input.type(slice.value)!=type)return {};
    auto c=constantBits(slice.value);
    if(!c || (bits && (c->type!=bits->type || c->literal!=bits->literal)))return {};
    bits=std::move(c);
  }
  return slices.front().value;
}

inline auto Lower::warpMma(v::Value value,const Offset& index) -> s::Ref {
  auto [lo,hi]=bounds(index);
  if(lo<0 || hi>=128)fail("scalar_fold_invalid_index","warp MMA output exceeds 16x8 matrix");
  const auto& op=dynamic_cast<const v::WarpMmaOp&>(input.definingNode(value).operation());
  const unsigned K=op.k(),bytes=op.elementBytes(),per=4/bytes;
  const auto type=v::ElementType::floating(8*bytes,op.element());
  const MemoryInstance key{value.nodeId(),value.instanceIndex()};
  auto found=warp_operand_views.find(key);
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16832
  // Element k of a row sits in lane group (k%(K/2))/per, register half k/(K/2), byte bytes*(k%per).
  if(found==warp_operand_views.end()) {
    const auto args=input.instanceOperands(value);
    if(args.size()!=320)fail("scalar_fold_mma_shape","warp MMA fragment operand count is not 320");
    std::array<std::vector<IndexedSourceSlice>,3> slices;
    slices[0].reserve(16*K);slices[1].reserve(8*K);slices[2].reserve(128);
    for(unsigned row=0;row<16;++row)for(unsigned k=0;k<K;++k)
      slices[0].push_back({args.at(((row%8)*4+(k%(K/2))/per)*4+(row/8)+2*(k/(K/2))),bytes*(k%per),bytes});
    for(unsigned col=0;col<8;++col)for(unsigned k=0;k<K;++k)
      slices[1].push_back({args.at(128+(col*4+(k%(K/2))/per)*2+k/(K/2)),bytes*(k%per),bytes});
    for(unsigned row=0;row<16;++row)for(unsigned col=0;col<8;++col)
      slices[2].push_back({args.at(192+((row%8)*4+col/2)*4+(row/8)*2+col%2),0,4});
    const auto make=[&](unsigned role) {
      auto pattern=output_source_matcher.matchSlices(slices[role]);
      const auto width=role==2?4u:bytes;
      const auto matrix_type=v::ValueType::matrix(role==2?v::ElementType::floating(32):type,{slices[role].size()});
      PartitionedRead partition{matrix_type,{},true};
      if(pattern) {
        auto source=std::make_shared<const IndexedSource>(std::move(*pattern));
        partition.regions.push_back({0,source->count*width,MemoryView{matrix_type,
            MemoryViewKind::Layout,{source->prototype},{{0,0,0,width,{{source->count,0,width}}}},{},source}});
        return partition;
      }
      // Every element one exact constant (a zero-initialized accumulator in
      // per-lane registers): one region whose elements all read that value
      // (source stride 0), not one region per register, which would split
      // every consumer domain at each element. O(slices) cached lookups.
      if(const auto uniform=uniformConstant(slices[role],width)) {
        partition.regions.push_back({0,slices[role].size()*width,MemoryView{matrix_type,MemoryViewKind::Layout,
            {*uniform},{{0,0,0,width,{{slices[role].size(),0,width}}}},{},{}}});
        return partition;
      }
      // Preserve existing register/component domains instead of requiring
      // unrelated C registers to match one expression. This finite layout
      // contains at most 512 records, independently of output/kernel size.
      std::vector<v::Value> values;std::vector<std::uint64_t> offsets;
      for(const auto& slice:slices[role]){values.push_back(slice.value);offsets.push_back(slice.offset);}
      for(const auto& domain:mapSourceTemplates(input,values,true,offsets)) {
        const auto begin=domain.begin*width;
        const auto bytes=((domain.count-1)*domain.result_stride+1)*width;
        const auto prototype=domain.source?domain.source->prototype:values[domain.begin];
        partition.regions.push_back({begin,bytes,MemoryView{matrix_type,MemoryViewKind::Layout,
            {prototype},{{0,offsets[domain.begin],begin,width,{{domain.count,0,domain.result_stride*width}}}},{},domain.source}});
      }
      return partition;
    };
    found=warp_operand_views.emplace(key,std::array<PartitionedRead,3>{make(0),make(1),make(2)}).first;
  }
  const auto& views=found->second;
  const auto operand=[&](unsigned role,const Offset& offset,const v::ElementType& element) {
    if(views[role].regions.size()==1)return view(views[role].regions.front().view,offset,element);
    return partition(views[role],value,offset,element);
  };
  const auto row=quotient(index,8),col=remainder(index,8);
  const auto level=static_cast<unsigned>(domains.size());
  struct Restore { std::vector<std::uint64_t>& domains;std::size_t size;
    ~Restore(){domains.resize(size);} } restore{domains,level};
  auto& progress=resumeReduction([&]{
    return operand(2,times(index,4),v::ElementType::floating(32));
  });
  ReductionRequests requests(requested,progress);
  auto sum=keys.at(progress.accumulator_key);
  if(progress.next)return sum;
  // Native warp contractions own their reduction axis just like other MMA
  // contractions. Read partitions may refine only the current uncommitted
  // piece; dependency suspension and collection retain completed sums.
  const FoldPieceKey piece_key{*projection_key,0};
  auto [parts,inserted]=fold_pieces.try_emplace(piece_key,
      std::vector<std::pair<std::uint64_t,std::uint64_t>>{{0,K}});
  for(auto sub=progress.sub_next;sub<parts->second.size();++sub) {
    const auto [first,count]=parts->second[sub];
    domains.resize(level);
    FoldScope scope(*this,piece_key,level,sub,first,count);
    Offset k(integer(first));
    if(count>1){domains.push_back(count);k.terms[level]=1;}
    auto a=operand(0,times(plus(times(row,K),k),bytes),type);
    auto b=operand(1,times(plus(times(col,K),k),bytes),type);
    a=result.graph.convert(a,"f32","NumericInterpretation");
    b=result.graph.convert(b,"f32","NumericInterpretation");
    const auto body=emitOperation("Multiply","f32",{a,b},"ideal_arithmetic");
    domains.resize(level);
    sum=count==1?reductions.append(s::Reducer::Sum,sum,body,"f32",level):
        reductions.appendFold(s::Reducer::Sum,sum,body,level,count,"f32");
    result.graph.markInstruction(sum,s::instruction_mode::synthesized);
    keys.at(progress.accumulator_key)=sum;
    progress.sub_next=sub+1;progress.pending.clear();
    limit();
  }
  commitReduction(progress,1,sum);
  return sum;
}

// The global input object named by an allocation identity (AllocationIdentity), else none.
inline auto Lower::globalObject(std::string_view identity) -> std::optional<std::string> {
  const auto colon=identity.find(':');
  if(colon==std::string_view::npos)return {};
  std::size_t length=0;
  try{length=std::stoul(std::string(identity.substr(0,colon)));}catch(const std::exception&){return {};}
  const auto rest=identity.substr(colon+1);
  if(rest.size()<length || rest.substr(0,length)!="global")return {};
  const auto name=rest.substr(length);
  const auto second=name.find(':');
  if(second==std::string_view::npos)return {};
  try{length=std::stoul(std::string(name.substr(0,second)));}catch(const std::exception&){return {};}
  if(name.size()!=second+1+length)return {};
  return std::string(name.substr(second+1));
}

// The values a 4-bit field at bit position `position` (a multiple of 4) of an object can
// hold under its field_values declaration: whole 1-, 2- or 4-bit fields combine; an
// 8-bit field contributes its low or high half. Exact marginal domain.
inline auto Lower::nibbleDomain(const TensorPropertyFact* fact,unsigned position) -> std::bitset<16> {
  std::bitset<16> domain;
  if(!fact){domain.set();return domain;}
  const auto bits=fact->field_bits;
  if(bits==8) {
    for(unsigned v=0;v<256;++v)if(fact->field_values.test(v))domain.set((v>>(position%8))&15);
    return domain;
  }
  const unsigned fields=4/bits;
  for(unsigned nibble=0;nibble<16;++nibble) {
    bool ok=true;
    for(unsigned f=0;f<fields && ok;++f)ok=fact->field_values.test((nibble>>(f*bits))&((1u<<bits)-1));
    if(ok)domain.set(nibble);
  }
  return domain;
}

inline auto Lower::metadataByteSource(const s::Ref& byte) -> MetadataByteSource {
  MetadataByteSource out;
  const auto plain=[](const s::Ref& ref){return ref.conversion.empty() || ref.conversion=="Bitcast";};
  if(!plain(byte)){out.reason="the metadata byte is a converted value";return out;}
  const auto* node=&result.graph.node(byte);
  if(node->kind==s::Kind::Constant) {
    const auto value=literal(node->name);
    if(!value){out.reason="the metadata byte is a non-integer literal";return out;}
    const auto bits=std::uint64_t(*value)&0xff;
    out.known=true;out.source="constant";
    out.fields[0].set(bits&15);out.fields[1].set(bits>>4);
    return out;
  }
  unsigned shift=0;
  if(node->kind==s::Kind::Operation && node->name=="ExtractBits" && node->attributes=="width=8" &&
     node->operands.size()==2) {
    const auto& amount=result.graph.node(node->operands[1]);
    const auto value=amount.kind==s::Kind::Constant?literal(amount.name):std::nullopt;
    if(!value || *value<0 || *value%4 || !plain(node->operands[0])) {
      out.reason="the metadata byte is a bit field at an unknown or unaligned position";return out;
    }
    shift=unsigned(*value);node=&result.graph.node(node->operands[0]);
  }
  // Bytes of an allocation without initial contents that no store covered before this read
  // (memory elimination's initial symbol of it): their contents are unspecified, so either
  // field can hold any value.
  if(node->kind==s::Kind::InitialCell && memory.unwritten_read_allocations.contains(node->name)) {
    out.known=true;out.source="unwritten_tensor_memory";
    out.fields[0].set();out.fields[1].set();
    return out;
  }
  if(node->kind!=s::Kind::InputValue) {
    out.reason="the metadata byte is computed by the kernel ("+(node->name.empty()?std::string("expression"):node->name)+")";
    return out;
  }
  const auto object=globalObject(node->name);
  if(!object){out.reason="the metadata byte is initial memory outside a global input object";return out;}
  out.known=true;out.source="input_object";out.object=*object;
  out.fact=findFieldValues(memory.tensor_properties,*object);
  // Fields are aligned to the object's bytes (field_bits divides 8), so a byte at any
  // offset holds whole fields; an 8-bit field may be split by an odd 4-bit shift.
  out.fields[0]=nibbleDomain(out.fact,shift);out.fields[1]=nibbleDomain(out.fact,shift+4);
  if(!node->operands.empty())
    if(const auto& offset=result.graph.node(node->operands[0]);offset.kind==s::Kind::Constant)
      if(const auto value=literal(offset.name);value && *value>=0)out.object_byte_offset=std::uint64_t(*value)+shift/8;
  return out;
}

// The PTX ISA sentence for the MMA's kind, quoted. 9.7.18.10.9.2 (.kind::f16, .kind::f8f6f4,
// .kind::mxf8f6f4, .kind::i8, .kind::ti16) only lists the meaningful values; 9.7.18.10.9.3
// (.kind::mxf4, .kind::mxf4nvf4) also names the rest undefined. Values outside the list have
// no meaning in PTX either way (owner rule 2026-11-05: follow PTX strictly).
inline auto Lower::sparseMetadataWording(const std::string& kind,UndefinedBehaviorWitness& w) -> void {
  w.isa_reference="PTX ISA 9.7.18.10.9 Sparse Matrices";
  if(kind=="mxf4" || kind=="mxf4nvf4") {
    w.isa_section="9.7.18.10.9.3 Sparse tcgen05.mma.sp with .kind::mxf4 and .kind::mxf4nvf4";
    w.isa_wording="undefined";
    w.condition="\"The only meaningful values of the metadata index are: 0b0100, 0b1000, 0b1100, 0b1001, 0b1101, 0b0110, 0b1110. Rest of the values result in undefined behavior.\"";
  } else {
    w.isa_section="9.7.18.10.9.2 Sparse tcgen05.mma.sp with .kind::f16, .kind::f8f6f4, .kind::mxf8f6f4, .kind::i8, .kind::ti16";
    w.isa_wording="only_meaningful";
    w.condition="\"The only meaningful values of the index are: 0b0100, 0b1000, 0b1100, 0b1001, 0b1101, 0b0110, 0b1110\" (the section gives no meaning to other values)";
  }
}

inline auto Lower::checkSparseMetadata() -> void {
  const auto meaningful=meaningfulSparseMetadata();
  const auto byte=v::ElementType::bit(8);
  std::optional<std::string> unproven;
  std::optional<MemoryInstance> unproven_at;
  const auto note=[&](std::string reason,const MemoryInstance& at) {
    if(!unproven){unproven=std::move(reason);unproven_at=at;}
  };
  for(const auto& node:input.nodes()) {
    const auto* mma=dynamic_cast<const v::TcgenMmaOp*>(&node.operation());
    if(!mma || !mma->attributes().sparse)continue;
    const auto& attr=mma->attributes();
    // kSparseMetadataMeaningful is the list of 9.7.18.10.9.2 and 9.7.18.10.9.3; another kind
    // (.kind::tf32, 9.7.18.10.9.1, lists other values) is not decided here.
    std::string kind;
    for(const auto& mod:attr.modifiers)if(mod.starts_with("kind::"))kind=mod.substr(6);
    if(kind!="f16" && kind!="f8f6f4" && kind!="mxf8f6f4" && kind!="i8" && kind!="ti16" &&
       kind!="mxf4" && kind!="mxf4nvf4") {
      if(node.instanceCount())
        note("the sparse MMA kind ("+(kind.empty()?std::string("none"):kind)+") has no modeled metadata value list",
             MemoryInstance{node.id(),0});
      continue;
    }
    for(std::size_t instance=0;instance<node.instanceCount();++instance) {
      const MemoryInstance at{node.id(),instance};
      const auto value=node.result(0,instance);
      const auto args=input.instanceOperands(value);
      const auto index=3u+unsigned(attr.accumulate)+(attr.scaled?2u:0u);
      if(index>=args.size()){note("the sparse MMA has no metadata operand",at);continue;}
      const auto metadata=args[index];
      const auto& type=input.type(metadata);
      if(!type.hasStaticShape() || type.element().isParameterized() || type.bitSize()%8 || !type.bitSize()) {
        note("the sparse MMA metadata has no fixed layout",at);continue;
      }
      const auto total=type.bitSize()/8;
      const auto level=static_cast<unsigned>(domains.size());
      for(std::uint64_t begin=0;begin<total;) {
        domains.resize(level);
        const auto count=std::max<std::uint64_t>(1,readSpan(metadata,Offset(integer(begin)),1,byte,total-begin));
        Offset position(integer(begin));
        if(count>1){domains.push_back(count);position.terms[level]=1;}
        // A byte whose projection fails is undecided; the scan goes on, since another
        // byte may still witness undefined behavior (project restores its state).
        MetadataByteSource source;
        try{source=metadataByteSource(project(metadata,position,byte));}
        catch(const Failure& e) {
          if(e.code.ends_with("_limit_exceeded"))throw;
          source.reason="a metadata byte does not project ("+e.code+": "+e.what()+")";
        }
        domains.resize(level);
        if(!source.known){note(source.reason,at);begin+=count;continue;}
        for(unsigned field=0;field<2;++field) {
          const auto invalid=source.fields[field]&~meaningful;
          if(invalid.none())continue;
          std::uint64_t witness=0;while(!invalid.test(witness))++witness;
          UndefinedBehaviorWitness w;
          sparseMetadataWording(kind,w);
          w.source=source.source;w.object=source.object;
          w.property=source.fact?"field_values":"none";
          w.operand_byte=begin;w.field=field;w.value=witness;
          if(source.source=="input_object") {
            // The witness byte's own cell offset (the span's first byte).
            if(count>1) {
              try{w.object_byte_offset=metadataByteSource(project(metadata,Offset(integer(begin)),byte)).object_byte_offset;}
              catch(const Failure& e){if(e.code.ends_with("_limit_exceeded"))throw;}
            } else w.object_byte_offset=source.object_byte_offset;
            if(source.fact)for(unsigned v=0;v<(1u<<source.fact->field_bits);++v)
              if(source.fact->field_values.test(v))w.declared_values.push_back(v);
          }
          for(const auto v:kSparseMetadataMeaningful)w.defined_values.push_back(v);
          std::sort(w.defined_values.begin(),w.defined_values.end());
          sparse_metadata_witness=std::move(w);
          current=at;
          fail("sparse_metadata_invalid","Sparse MMA metadata can hold an index value outside the PTX ISA's meaningful values");
        }
        begin+=count;
      }
    }
  }
  if(unproven) {
    current=unproven_at;
    fail("sparse_metadata_validity_not_proven","Sparse MMA metadata validity is not proven: "+*unproven);
  }
}

inline auto Lower::scaledMma(v::Value value,const Offset& index,const v::TcgenMmaAttributes& attr) -> s::Ref {
  const auto args=input.instanceOperands(value);
  const auto& dst=input.type(value);
  std::string kind;
  unsigned group=1;
  for(const auto& mod:attr.modifiers) {
    if(mod.starts_with("kind::"))kind=mod.substr(6);
    if(mod=="cta_group::2")group=2;
  }
  if(attr.sparse || attr.scale_input_d || (kind!="mxf4" && kind!="mxf4nvf4" && kind!="mxf8f6f4"))
    fail("scalar_fold_mma_variant","block-scaled scalar lowering requires dense MMA without scaled D");
  auto desc=known(args.at(2));
  if(!desc || *desc<0 || std::uint64_t(*desc)>UINT32_MAX)
    fail("scalar_fold_mma_descriptor","MMA instruction descriptor is not constant");
  const auto bits=std::uint64_t(*desc);
  const auto d=frontend::ptx::DecodeBlockScaledDescriptor(bits,kind);
  if((d.a_element_bits!=4 && d.a_element_bits!=8) ||
     (d.b_element_bits!=4 && d.b_element_bits!=8))
    fail("scalar_fold_mma_variant","block-scaled scalar lowering requires FP4 or FP8 operands");
  if(bits&(3ULL<<13))fail("scalar_fold_mma_flags","MMA operand negation is not lowered");
  const auto vector=kind=="mxf4"?2u:kind=="mxf4nvf4"?4u:1u, block=d.k/vector;
  const auto scale_index=3u+unsigned(attr.accumulate);
  const auto scale_type=v::ElementType::floating(8,(bits&(1ULL<<23))?"ue8m0":"ue4m3");
  const auto byte=v::ElementType::bit(8);
  const auto checkShape=[&](v::Value operand,std::vector<std::uint64_t> shape) {
    const auto& type=input.type(operand);
    if(!type.hasStaticShape() || !std::ranges::equal(type.shape(),shape) || type.element().width()!=8)
      fail("scalar_fold_mma_shape","packed FP4 or replicated scale shape disagrees with descriptor");
  };
  const unsigned unit=(d.a_element_bits==4 || d.b_element_bits==4)?2:1;
  const unsigned astride=unit*d.a_element_bits/8,bstride=unit*d.b_element_bits/8;
  checkShape(args.at(0),{d.m,d.k*d.a_element_bits/8});
  checkShape(args.at(1),{d.n,d.k*d.b_element_bits/8});
  checkShape(args.at(scale_index),{group,4,d.m/group,vector});
  checkShape(args.at(scale_index+1),{group,4,d.n,vector});
  if(!std::ranges::equal(dst.shape(),std::vector<std::uint64_t>{d.m,d.n}))
    fail("scalar_fold_mma_shape","FP4 output shape disagrees with descriptor");
  const auto row=quotient(index,d.n),col=remainder(index,d.n);
  const auto dtype=v::ElementType::floating(32);
  // A zero-fill A or B makes every scaled product +0, whatever the scales.
  if(auto zero=zeroProductMma(args,attr.accumulate,3,index,4,dtype,dst.element()))return *zero;
  auto& progress=resumeReduction([&]{return attr.accumulate?
      project(args.at(3),times(index,4),dtype):result.graph.constant("f32","0");});
  ReductionRequests requests(requested,progress);
  auto sum=keys.at(progress.accumulator_key);
  const auto level=static_cast<unsigned>(domains.size());
  // PTX scale replicas: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-2x
  const auto scale=[&](bool b,unsigned chunk) {
    const auto rows=b?d.n:d.m/group;
    const auto part=b?Offset(0):quotient(row,rows);
    const auto mn=b?col:remainder(row,rows);
    std::optional<s::Ref> canonical;
    for(unsigned peer=0;peer<(b?group:1);++peer)for(unsigned replica=0;replica<4;++replica) {
      const auto p=b?Offset(peer):part;
      const auto position=plus(times(plus(times(plus(times(p,4),Offset(replica)),rows),mn),vector),Offset(chunk));
      auto value=project(args.at(scale_index+unsigned(b)),position,scale_type);
      if(canonical && *canonical!=value) {
#ifdef SENTRYWARD_ENABLE_COMPRESSION_AUDIT
        if(sentryward::debug::environment("SENTRYWARD_SCALE_REPLICA_DIAGNOSTIC")) {
#if defined(SENTRYWARD_DEVELOPMENT_DIAGNOSTICS) || defined(SENTRYWARD_ENABLE_COMPRESSION_AUDIT)
#include "sentryward/development/blocks/scalar_fold_lowering_trace_2.inc"
#endif
}
#endif
        fail("scalar_fold_scale_replicas","scale replicas do not have the same proven value");
      }
      canonical=value;
    }
    return result.graph.convert(*canonical,"f32","NumericInterpretation");
  };
  const auto fp4=[&](s::Ref packed,unsigned half) {
    auto field=emitOperation("ExtractBits","b4",
        {packed,result.graph.constant("u64",std::to_string(half*4))},"width=4");
    return result.graph.convert(result.graph.convert(field,"e2m1","Bitcast"),"f32","NumericInterpretation");
  };
  const auto element=[&](bool b,const Offset& position,unsigned component) {
    const auto width=b?d.b_element_bits:d.a_element_bits;
    auto packed=project(args[unsigned(b)],plus(position,Offset(width==8?component:0)),byte);
    if(width==4)return fp4(packed,component);
    const auto code=(bits>>(b?10:7))&7;
    return result.graph.convert(result.graph.convert(packed,code?"e5m2":"e4m3","Bitcast"),"f32","NumericInterpretation");
  };
  for(unsigned chunk=static_cast<unsigned>(progress.next/(block/unit));chunk<vector;++chunk) {
    domains.resize(level);
    auto sa=scale(false,chunk),sb=scale(true,chunk);
    for(std::uint64_t begin=progress.next;begin<(chunk+1)*block/unit;) {
      domains.resize(level);
      auto ai=plus(times(row,d.k*d.a_element_bits/8),Offset(integer(begin*astride)));
      auto bi=plus(times(col,d.k*d.b_element_bits/8),Offset(integer(begin*bstride)));
      const auto remaining=(chunk+1)*block/unit-begin;
      auto count=std::min(readSpan(args[0],ai,astride,byte,remaining),readSpan(args[1],bi,bstride,byte,remaining));
      // Mixed FP4/FP8 contractions consume two FP8 bytes per packed unit.
      // Check both source spans without expanding the reduction domain.
      if(astride==2)count=std::min(count,readSpan(args[0],plus(ai,Offset(1)),astride,byte,remaining));
      if(bstride==2)count=std::min(count,readSpan(args[1],plus(bi,Offset(1)),bstride,byte,remaining));
      // Extent comes from the instruction/source map, not repetition
      // inference. A certified two-element domain is already a legal Fold.
      if(count>1){
        domains.push_back(count);Offset iterator;iterator.terms[level]=1;
        // ai/bi may be compound quotient/remainder expressions. Compose the
        // new binder through the Offset algebra; writing their affine terms
        // directly would leave those terms hidden behind expression.
        ai=plus(ai,times(iterator,astride));bi=plus(bi,times(iterator,bstride));
      }
      std::array<s::Ref,2> terms;
      for(unsigned half=0;half<unit;++half) {
        auto a=emitOperation("Multiply","f32",{element(false,ai,half),sa},"ideal_arithmetic");
        auto b=emitOperation("Multiply","f32",{element(true,bi,half),sb},"ideal_arithmetic");
        terms[half]=emitOperation("Multiply","f32",{a,b},"ideal_arithmetic");
      }
      auto body=unit==1?terms[0]:emitOperation("Add","f32",{terms[0],terms[1]},"ideal_arithmetic");
      domains.resize(level);
      sum=count==1?reductions.append(s::Reducer::Sum,sum,body,"f32",level):
          reductions.appendFold(s::Reducer::Sum,sum,body,level,count,"f32");
      result.graph.markInstruction(sum,s::instruction_mode::synthesized);
      begin+=count;commitReduction(progress,begin,sum);limit();
    }
  }
  return cast(sum,dst.element());
}

inline auto Lower::mma(v::Value value,const Offset& index) -> s::Ref {
  const auto& node=input.definingNode(value);const auto& op=node.operation();
  const auto args=input.instanceOperands(value);
  const auto& dst=input.type(value);
  if(!dst.hasStaticShape() || dst.shape().size()!=2)
    fail("scalar_fold_parameterized_mma","MMA result shape is not a fixed rank-two matrix");
  const auto rows=dst.shape()[0],cols=dst.shape()[1];
  auto row=quotient(index,cols),col=remainder(index,cols);
  auto [rl,rh]=bounds(row);if(rl<0 || std::uint64_t(rh)>=rows)fail("scalar_fold_invalid_index","MMA output row exceeds shape");
  const bool native=dynamic_cast<const v::TcgenMmaOp*>(&op);
  bool accumulate=false;
  std::size_t accumulator=2;
  auto atype=input.type(args[0]).element(),btype=input.type(args[1]).element(),dtype=dst.element();
  if(native){
    const auto& attr=dynamic_cast<const v::TcgenMmaOp&>(op).attributes();
    // Sparse MMA arithmetic is not lowered (checkSparseMetadata ran before any output).
    if(attr.sparse)fail("scalar_fold_mma_variant","sparse MMA arithmetic (ISA metadata placement) is not lowered");
    if(attr.scaled)return scaledMma(value,index,attr);
    if(attr.scale_input_d)
      fail("scalar_fold_mma_variant","scaled, sparse or scaled-D MMA needs scalar semantics");
    auto desc=known(args[2]);
    if(!desc || *desc<0 || std::uint64_t(*desc)>UINT32_MAX)
      fail("scalar_fold_mma_descriptor","MMA instruction descriptor is not constant");
    const auto bits=std::uint64_t(*desc);
    // PTX instruction descriptor: flags 0..3 and negate bits need separate rules.
    if(bits & (15ULL | (3ULL<<13)))
      fail("scalar_fold_mma_flags","MMA sparsity, saturation or negate flags are not lowered");
    std::string kind;
    for(const auto& mod:attr.modifiers)if(mod.starts_with("kind::"))kind=mod.substr(6);
    if(kind!="f16" && kind!="tf32" && kind!="f8f6f4")
      fail("scalar_fold_mma_variant","MMA arithmetic kind is not supported");
    auto d=frontend::ptx::DecodeMmaDescriptor(kind,bits);
    const auto interpret=[&](unsigned code,unsigned width){
      if(kind=="f16")return v::ElementType::floating(16,code?"bf16":"f16");
      if(kind=="tf32")return v::ElementType::floating(32,"tf32");
      if(width!=8)fail("scalar_fold_subbyte_mma","sub-byte MMA elements need bit-index iterators");
      return v::ElementType::floating(8,code?"e5m2":"e4m3");
    };
    atype=interpret((bits>>7)&7,d.a_element_bits);
    btype=interpret((bits>>10)&7,d.b_element_bits);
    dtype=v::ElementType::floating(d.d_element_bits);
    accumulate=attr.accumulate;accumulator=3;
  }else{
    const auto* mat=dynamic_cast<const v::MatMulOp*>(&op);
    if(!mat)fail("scalar_fold_mma_variant","native fragment/scaled MMA is not lowered");
    accumulate=mat->accumulate();
  }
  const auto& a=input.type(args[0]);const auto& b=input.type(args[1]);
  // A native Tensor Memory A of shape (slices, rows, K) holds one copy per
  // data-path lane group; lane group s multiplies its own copy into D columns
  // [s*cols/slices, (s+1)*cols/slices).
  const bool sliced=native && a.hasStaticShape() && a.shape().size()==3;
  if(!a.hasStaticShape() || !b.hasStaticShape() || a.shape().size()!=(sliced?3U:2U) || b.shape().size()!=2)
    fail("scalar_fold_parameterized_mma","MMA operand shape is not fixed rank two");
  const auto slices=sliced?a.shape()[0]:1;
  const auto k=a.shape().back();
  if(a.shape()[sliced?1:0]!=rows || !slices || cols%slices ||
      (native?(b.shape()[0]!=cols || b.shape()[1]!=k):(b.shape()[0]!=k || b.shape()[1]!=cols)))
    fail("scalar_fold_mma_shape","MMA operand shapes disagree with scalar contraction");
  const auto arow=sliced?plus(times(quotient(col,cols/slices),integer(rows)),row):row;
  // A zero-fill A or B makes every product term +0: no plan, pieces or cuts.
  if(native)if(auto zero=zeroProductMma(args,accumulate,accumulator,index,
      integer(dst.element().width()/8),dtype,dst.element()))return *zero;
  const auto level=static_cast<unsigned>(domains.size());
  // Publish only a complete plan. Domain refinement may throw; leaving an
  // empty cache entry then would silently disable planning on the retry.
  auto plan_it=reduction_domain_plans.find(*projection_key);
  if(plan_it==reduction_domain_plans.end()) {
    std::vector<algorithm::AffineDomain> computed_plan;
    const auto ao=times(times(arow,integer(k)),integer(atype.width()/8));
    const auto bo=times(native?times(col,integer(k)):col,integer(btype.width()/8));
    const auto ad=reductionDomains(args[0],ao,atype.width()/8,atype,k);
    const auto bd=reductionDomains(args[1],bo,(native?1:cols)*(btype.width()/8),btype,k);
    const auto preflight=[&](v::Value operand,const Offset& at,const v::ElementType& type) {
      const auto read=memory.normalizedRead({operand.nodeId(),operand.instanceIndex()});
      if(read)if(const auto* split=std::get_if<const PartitionedRead*>(&*read)) {
        if(const auto* periodic=periodicPartition(**split))refinePartitionDomain(*periodic,at,type);
        refinePartitionDomain(**split,at,type);
      }
    };
    if(!ad)preflight(args[0],ao,atype);
    if(!bd)preflight(args[1],bo,btype);
    SENTRYWARD_AUDIT({
      (void)audit;
      static thread_local std::set<std::uint64_t> reported;
      const bool fragmented=(ad && ad->size()>1) || (bd && bd->size()>1);
      if((!ad || !bd || fragmented) && reported.size()<16 && reported.insert(value.nodeId()).second) {
        SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_30);
        for(auto operand:{args[0],args[1]}) {
          const auto read=memory.normalizedRead({operand.nodeId(),operand.instanceIndex()});
          SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_31);
          if(read)if(const auto* split=std::get_if<const PartitionedRead*>(&*read)) {
            SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_32);
            for(std::size_t i=0;i<std::min<std::size_t>(4,(*split)->regions.size());++i) {
              const auto& block=(*split)->regions[i].view.blocks.at(0);
              SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_33);
              for(const auto& axis:block.axes)SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_34);
              SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_35);
              const auto& region=(*split)->regions[i].view;
              if(const auto* source=std::get_if<v::Value>(&region.sources.at(block.source))) {
                SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_36);
                for(const auto& arg:input.instance(*source).operands)
                  SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_37);
              }
            }
          }
        }
      }
    });
    if(ad && bd) {
      SENTRYWARD_LEDGER(ledger->add("reduction_domain_candidate_pairs",ad->size()*bd->size()));
      for(const auto& a_domain:*ad)for(const auto& b_domain:*bd) {
        const auto common=algorithm::IntersectAffine(a_domain,b_domain);
        if(common.count)computed_plan.push_back({a_domain.first+a_domain.step*common.left,
            common.count>1?a_domain.step*common.left_step:1,common.count});
      }
      std::uint64_t covered=0;for(const auto& d:computed_plan)covered+=d.count;
      if(covered!=k)fail("scalar_fold_internal_binding","reduction source domains do not cover K");
    }
    plan_it=reduction_domain_plans.emplace(*projection_key,std::move(computed_plan)).first;
    SENTRYWARD_LEDGER(ledger->add(plan_it->second.empty()?"mma_unplanned":"mma_planned");
      ledger->add("mma_plan_pieces",plan_it->second.size()));
  }
  auto& progress=resumeReduction([&]{return accumulate?
      project(args.at(accumulator),times(index,integer(dst.element().width()/8)),dtype):
      result.graph.constant(dtype.toString(),"0");});
  ReductionRequests requests(requested,progress);
  auto folded=keys.at(progress.accumulator_key);
  const auto& plan=plan_it->second;
  const bool planned=!plan.empty();
  for(std::uint64_t cursor=progress.next;cursor<(planned?plan.size():k);) {
    const auto begin=planned?plan[cursor].first:cursor;
    const auto step=planned?plan[cursor].step:1;
    domains.resize(level);
    const auto a_start=times(plus(times(arow,integer(k)),Offset(integer(begin))),integer(atype.width()/8));
    const auto b_start=times(native?plus(times(col,integer(k)),Offset(integer(begin))):
        plus(Offset(mul(integer(begin),integer(cols))),col),integer(btype.width()/8));
    auto count=planned?plan[cursor].count:std::min(readSpan(args[0],a_start,atype.width()/8,atype,k-begin),
        readSpan(args[1],b_start,mul(native?1:integer(cols),integer(btype.width()/8)),btype,k-begin));
    const auto whole_count=count;
    const FoldPieceKey piece_key{*projection_key,cursor};
    auto [parts,inserted]=fold_pieces.try_emplace(piece_key,
        std::vector<std::pair<std::uint64_t,std::uint64_t>>{{0,whole_count}});
    for(auto sub=progress.sub_next;sub<parts->second.size();++sub) {
    const auto [first,part_count]=parts->second[sub];count=part_count;
    SENTRYWARD_LEDGER(ledger->add("mma_piece_attempts");ledger->add(count==1?"mma_singleton_attempts":"mma_fold_attempts"));
    domains.resize(level);
    FoldScope scope(*this,piece_key,level,sub,first,count);
    // The >=3 threshold belongs to recognizing repetition in scalar chains;
    // a native contraction's certified extent does not need rediscovery.
    SENTRYWARD_AUDIT(if(count==1 && k-begin>1)
      audit->event("INSTANCE_ENUMERATION","mma_scalar_terms_then_fold_detection",k-begin,1,value.nodeId(),true);
      if(count==1)audit->add(AuditCounter::scalar_reduction_terms));
    if(count>1)domains.push_back(count);
    Offset iv(add(integer(begin),mul(integer(step),integer(first))));if(count>1)iv.terms[level]=integer(step);
    auto ai=plus(times(arow,integer(k)),iv);
    auto bi=native?plus(times(col,integer(k)),iv):plus(times(iv,integer(cols)),col);
    auto av=project(args[0],times(ai,integer(atype.width()/8)),atype);
    auto bv=project(args[1],times(bi,integer(btype.width()/8)),btype);
    av=result.graph.convert(av,dtype.toString(),"NumericInterpretation");
    bv=result.graph.convert(bv,dtype.toString(),"NumericInterpretation");
    auto body=emitOperation("Multiply",dtype.toString(),{av,bv},"ideal_arithmetic");
    domains.resize(level);
    // Storage-domain boundaries do not imply different mathematical term
    // templates. Submit each emitted singleton to the same incremental
    // builder, preserving its exact operands and existing domain order.
    // Recognition starts after three terms; a mismatch remains an Add.
    if(count==1)folded=reductions.append(s::Reducer::Sum,folded,body,dtype.toString(),level);
    else folded=reductions.appendFold(s::Reducer::Sum,folded,body,level,count,dtype.toString());
    result.graph.markInstruction(folded,s::instruction_mode::synthesized);
    keys.at(progress.accumulator_key)=folded;progress.sub_next=sub+1;progress.pending.clear();
    SENTRYWARD_LEDGER(ledger->add("mma_pieces_committed"));
    }
    cursor+=planned?1:whole_count;
    commitReduction(progress,cursor,folded);
    limit();
  }
  return cast(folded,dst.element());
}

} // namespace
} // namespace sentryward::value_analysis
