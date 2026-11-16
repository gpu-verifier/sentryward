#pragma once
#include "scalar_fold_lowering_impl.hpp"

namespace sentryward::value_analysis {
namespace {

inline auto Lower::periodicPartition(const PartitionedRead& split) -> const PartitionedRead* {
  auto [entry,inserted]=periodic_partitions.try_emplace(&split);
  if(!inserted)return entry->second?&*entry->second:nullptr;
  const auto& regions=split.regions;
  // A phase reads initial memory at an address, a zero literal, or one existing Value at a
  // payload offset (e.g. a tcgen05.cp image read through a permuted operand layout). Value
  // phases are re-encoded only for a permutation, i.e. a period of at least two regions: a
  // one-region period of one Value is a plain strided slice (e.g. a scale-factor copy read
  // across K blocks), whose stored regions already plan its reductions block by block.
  struct Phase {const InitialMemorySymbol* initial;std::uint64_t address;const v::Value* value=nullptr;};
  const auto phase=[&](const ReadRegion& r)->std::optional<Phase> {
    const auto& view=r.view;
    if(view.indexed_source || view.sources.size()!=1 || view.blocks.size()!=1)return {};
    const auto& b=view.blocks[0];
    if(b.source || b.result_offset!=r.begin || !r.bytes)return {};
    if(const auto* initial=std::get_if<InitialMemorySymbol>(&view.sources[0])) {
      const auto address=static_cast<unsigned __int128>(initial->region.offset)+b.source_offset;
      if(address>std::uint64_t(INT64_MAX))return {};
      return Phase{initial,static_cast<std::uint64_t>(address)};
    }
    const auto& value=std::get<v::Value>(view.sources[0]);
    if(zeroMatrixLiteral(value))return Phase{nullptr,0};
    return Phase{nullptr,b.source_offset,&value};
  };
  const auto shape=[&](const ReadRegion& a,const ReadRegion& b) {
    const auto pa=phase(a),pb=phase(b);
    return pa && pb && a.bytes==b.bytes && a.view.blocks[0].bytes==b.view.blocks[0].bytes &&
        a.view.blocks[0].axes==b.view.blocks[0].axes && !pa->initial==!pb->initial &&
        (!pa->initial || pa->initial->region.allocation==pb->initial->region.allocation) &&
        !pa->value==!pb->value && (!pa->value || *pa->value==*pb->value);
  };
  // Next region of the same shape: keyed by shape, scanned from the back.
  std::vector<std::size_t> next(regions.size(),regions.size());
  {
    std::map<std::tuple<std::uint64_t,std::uint64_t,std::vector<std::uint64_t>,std::string>,std::size_t> later;
    for(auto i=regions.size();i-->0;) {
      const auto p=phase(regions[i]);
      if(!p)continue;
      const auto& b=regions[i].view.blocks[0];
      std::vector<std::uint64_t> axes;
      for(const auto& axis:b.axes){axes.push_back(axis.count);axes.push_back(axis.source_stride);axes.push_back(axis.result_stride);}
      auto [found,fresh]=later.try_emplace({regions[i].bytes,b.bytes,std::move(axes),
          p->initial?p->initial->region.allocation:p->value?"value:"+valueKey(*p->value):std::string()},i);
      if(!fresh){next[i]=found->second;found->second=i;}
    }
  }
  PartitionedRead result{split.type,{},true};
  bool changed=false;
  for(std::size_t i=0;i<regions.size();) {
    const auto q=next[i]-i;
    std::size_t periods=0;
    std::vector<std::uint64_t> steps;
    if(next[i]<regions.size() && regions[next[i]].begin>regions[i].begin) {
      const auto period=regions[next[i]].begin-regions[i].begin;
      const auto matches=[&](std::size_t p) {
        if(i+(p+1)*q>regions.size())return false;
        for(std::size_t j=0;j<q;++j) {
          const auto& first=regions[i+j];const auto& later=regions[i+p*q+j];
          if(!shape(first,later) || __int128(later.begin)!=__int128(first.begin)+__int128(p)*period)return false;
          const auto a=phase(first),b=phase(later);
          if(!a->initial && !a->value)continue;
          if(p==1){if(b->address<a->address)return false;steps[j]=b->address-a->address;}
          else if(__int128(b->address)!=__int128(a->address)+__int128(p)*steps[j])return false;
        }
        return true;
      };
      steps.assign(q,0);
      periods=1;
      while(matches(periods))++periods;
    }
    if(periods>=2 && q<2)if(const auto p=phase(regions[i]);p && p->value)periods=0;
    const auto period=periods<2?0:regions[next[i]].begin-regions[i].begin;
    std::vector<ReadRegion> merged;
    for(std::size_t j=0;j<q && periods>=2;++j) {
      auto region=regions[i+j];
      // The period must enclose the phase, so that it is the outermost radix.
      if(region.bytes>period)break;
      auto& b=region.view.blocks[0];
      if(auto* s=std::get_if<InitialMemorySymbol>(&region.view.sources[0])) {
        const auto origin=phase(region)->address;
        auto end=static_cast<unsigned __int128>(s->region.offset)+s->region.bytes;
        for(std::size_t p=1;p<periods;++p) {
          const auto& later=std::get<InitialMemorySymbol>(regions[i+p*q+j].view.sources[0]);
          end=std::max(end,static_cast<unsigned __int128>(later.region.offset)+later.region.bytes);
        }
        s->region.offset=origin;s->region.bytes=static_cast<std::uint64_t>(end-origin);
        s->type=v::ValueType::matrix(v::ElementType::bit(8),{s->region.bytes});
        b.source_offset=0;
      }
      const auto bytes=static_cast<unsigned __int128>(periods-1)*period+region.bytes;
      if(bytes>std::uint64_t(INT64_MAX) || static_cast<unsigned __int128>(region.begin)+bytes>std::uint64_t(INT64_MAX))break;
      b.axes.push_back({periods,steps[j],period});
      region.bytes=static_cast<std::uint64_t>(bytes);region.view.kind=MemoryViewKind::Layout;
      merged.push_back(std::move(region));
    }
    if(merged.size()!=q || periods<2){result.regions.push_back(regions[i++]);continue;}
    for(auto& region:merged)result.regions.push_back(std::move(region));
    i+=periods*q;changed=true;
  }
  if(changed)entry->second=std::move(result);
  return entry->second?&*entry->second:nullptr;
}

inline auto Lower::partitionGrid(const PartitionedRead& split,v::Value reader,
                                const v::ElementType& type) -> const MemoryView* {
  const auto& reader_type=input.type(reader);
  if(type.isParameterized() || !type.width() || type.width()%8 ||
     !reader_type.isMatrix() || !reader_type.hasStaticShape() ||
     reader_type.element().isParameterized() || reader_type.element().width()!=type.width() ||
     reader_type.shape().size()!=2)return nullptr;
  const auto width=std::uint64_t(type.width()/8);
  const auto columns=reader_type.shape().back();
  const PartitionGridKey key{&split,valueKey(reader),type.toString()};
  auto [entry,inserted]=partition_grids.try_emplace(key);
  if(inserted) {
    const auto* replacement=memory.readReplacement({reader.nodeId(),reader.instanceIndex()});
    const auto* raw=replacement?std::get_if<AssembledRead>(replacement):nullptr;
    if(!raw)return nullptr;
    SENTRYWARD_AUDIT({
      (void)audit;
      std::map<std::uint64_t,std::size_t> histogram;
      for(const auto& piece:raw->pieces)if(piece.type.hasStaticShape() && !piece.type.element().isParameterized())
        ++histogram[piece.type.bitSize()];
      SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_1);
      for(auto [bits,n]:histogram)SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_2);
      SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_3);
    });
    std::uint64_t fields=0;
    for(const auto& piece:raw->pieces) {
      // piece.type describes the stored byte fragment, not the producer's
      // type: elimination publishes matrix<Nxb8> even for packed registers.
      // Bound the stored fragment to64bits; every resulting slice must still
      // resolve to an exact whole numeric field through the shared decoder.
      if(!piece.type.hasStaticShape() ||
         piece.type.element().isParameterized() || !piece.type.bitSize() ||
         piece.type.bitSize()>64 || piece.type.bitSize()%type.width())return nullptr;
      const auto count=piece.type.bitSize()/type.width();
      if(count>1000000-fields)return nullptr;
      fields+=count;
    }
    if(fields<3 || columns<2 || fields%columns || fields/columns<2 ||
       reader_type.bitSize()%8 || fields*width!=reader_type.bitSize()/8)return nullptr;
    std::vector<IndexedSourceSlice> slices;slices.reserve(fields);
    std::uint64_t offset=0;
    // At most8 byte-aligned fields per stored fragment, independent of source
    // matrix extent or thread count. Wider compact pieces were rejected above.
    for(const auto& piece:raw->pieces) {
      for(std::uint64_t field=0;field<piece.type.bitSize()/type.width();++field) {
        slices.push_back({reader,offset,width});offset+=width;
      }
    }
    if(auto pattern=output_source_matcher.matchGridSlices(slices,columns)) {
      if(pattern->count!=slices.size())return nullptr;
      auto source=std::make_shared<const IndexedSource>(std::move(*pattern));
      entry->second=MemoryView{reader_type,MemoryViewKind::Layout,{source->prototype},
          {{0,0,0,width,{{source->count,0,width}}}},{},std::move(source)};
    }
  }
  return entry->second?&*entry->second:nullptr;
}

inline auto Lower::partitionLattice(
    const PartitionedRead& split,v::Value reader,const Offset& offset,const v::ElementType& type) -> std::optional<PartitionLattice> {
  if(type.isParameterized() || !type.width() || type.width()%8)return {};
  const auto width=std::uint64_t(type.width()/8);
  const auto [lo,hi]=bounds(offset);
  if(lo<0 || hi<=lo)return {};
  // Exact congruence summary, not a fit to representative evaluations.
  // Quotient/remainder values are integral but otherwise unconstrained.
  // Outer scaling therefore retains e.g. 32*floor((512*i+j)/128).
  using Congruence=std::pair<std::int64_t,std::uint64_t>;
  std::unordered_map<const OffsetExpression*,Congruence> congruences;
  const auto congruence=[&](auto&& self,const Offset& x,unsigned depth)->Congruence {
    if(!x.expression) {
      std::uint64_t step=0;
      for(auto [axis,c]:x.terms) {
        (void)axis;
        const auto magnitude=c<0?std::uint64_t{0}-std::uint64_t(c):std::uint64_t(c);
        step=std::gcd(step,magnitude);
      }
      return {x.base,step};
    }
    if(depth>=options.max_recursion)return {0,1};
    if(auto found=congruences.find(x.expression.get());found!=congruences.end())return found->second;
    const auto& e=*x.expression;
    if(e.kind==OffsetExpression::Quotient || e.kind==OffsetExpression::Remainder)return {0,1};
    const auto a=self(self,e.left,depth+1);
    if(e.kind==OffsetExpression::Add) {
      const auto b=self(self,e.right,depth+1);
      const Congruence result{add(a.first,b.first),std::gcd(a.second,b.second)};
      congruences.emplace(x.expression.get(),result);return result;
    }
    const auto magnitude=e.factor<0?std::uint64_t{0}-std::uint64_t(e.factor):std::uint64_t(e.factor);
    if(magnitude && a.second>std::uint64_t(INT64_MAX)/magnitude)return {0,1};
    const Congruence result{mul(a.first,e.factor),a.second*magnitude};
    congruences.emplace(x.expression.get(),result);return result;
  };
  const auto [base,step]=congruence(congruence,offset,0);
  if(step<width || step>std::uint64_t(INT64_MAX))return {};
  const auto modulus=std::int64_t(step);
  const auto raw_residue=base%modulus;
  const auto residue=raw_residue<0?raw_residue+modulus:raw_residue;
  const auto first=std::uint64_t(lo)+
      (std::uint64_t(residue)+step-std::uint64_t(lo)%step)%step;
  if(first>std::uint64_t(hi))return {};
  const auto count=(std::uint64_t(hi)-first)/step+1;
  // At most one selected position per stored region is admitted here. This
  // bound forbids constructing a logical-member vector from a compact region.
  if(count<2 || count>split.regions.size() || count>UINT64_MAX/width)return {};
  const PartitionLatticeKey key{&split,valueKey(reader),type.toString(),first,step,count};
  auto [entry,inserted]=partition_lattices.try_emplace(key);
  if(inserted) {
    const bool raw_reader=memory.readReplacement({reader.nodeId(),reader.instanceIndex()})!=nullptr;
    const auto reader_type=input.type(reader);
    if(raw_reader && (!reader_type.hasStaticShape() || reader_type.element().isParameterized()))return {};
    const auto reader_bits=raw_reader?reader_type.bitSize():0;
    std::vector<std::pair<std::uint64_t,IndexedSourceSlice>> pieces;
    pieces.reserve(std::min<std::size_t>(split.regions.size(),count));
    bool valid=true;
    std::uint64_t previous_end=0;
    for(const auto& region:split.regions) {
      if(!region.bytes || region.begin<previous_end || region.begin>UINT64_MAX-region.bytes){valid=false;break;}
      const auto end=region.begin+region.bytes;
      previous_end=end;
      if(end<=first || region.begin>std::uint64_t(hi))continue;
      const auto begin=std::max(first,region.begin);
      const auto q=(begin-first)/step+((begin-first)%step!=0);
      if(q>=count)continue;
      const auto at=first+q*step;
      if(at>=end)continue;
      if(width>end-at || (q+1<count && step<end-at)) {valid=false;break;}
      // Prefer the authoritative immutable read substitution. slice() in
      // the shared matcher already resolves assembled pieces and read aliases
      // with cached piece-end indices, preserving the actual writer/use.
      // In particular never strip an indexed region down to its prototype.
      if(raw_reader) {
        const auto bits=reader_bits;
        if(bits%8 || at>bits/8 || width>bits/8-at){valid=false;break;}
        pieces.push_back({q,{reader,at,width}});
        continue;
      }
      const auto& source_view=region.view;
      // Existing indexed regions are already handled by the direct covering
      // path. Never flatten one here to satisfy matchSlices' finite API.
      if(source_view.indexed_source || source_view.blocks.size()!=1 ||
         source_view.sources.size()!=1 || source_view.blocks[0].source!=0) {valid=false;break;}
      const auto* value=std::get_if<v::Value>(&source_view.sources[0]);
      if(!value){valid=false;break;}
      const auto mapped=viewOffset(source_view.blocks[0],Offset(integer(at)),type);
      if(!mapped || mapped->expression || !mapped->terms.empty() || mapped->base<0) {valid=false;break;}
      const auto bits=input.type(*value).bitSize();
      if(bits%8 || std::uint64_t(mapped->base)>bits/8 || width>bits/8-std::uint64_t(mapped->base)) {valid=false;break;}
      pieces.push_back({q,{*value,std::uint64_t(mapped->base),width}});
    }
    if(valid && pieces.size()==count) {
      std::sort(pieces.begin(),pieces.end(),[](const auto& a,const auto& b){return a.first<b.first;});
      std::vector<IndexedSourceSlice> slices;slices.reserve(pieces.size());
      for(std::size_t i=0;i<pieces.size();++i) {
        if(pieces[i].first!=i){valid=false;break;}
        slices.push_back(pieces[i].second);
      }
      if(valid)if(auto pattern=output_source_matcher.matchSlices(slices)) {
        auto source=std::make_shared<const IndexedSource>(std::move(*pattern));
        if(source->count==count)
          entry->second=MemoryView{v::ValueType::matrix(type,{count}),MemoryViewKind::Layout,
              {source->prototype},{{0,0,0,width,{{count,0,width}}}},{},std::move(source)};
      }
    }
  }
  if(!entry->second)return {};
  auto index=quotient(plus(offset,Offset(-integer(first))),integer(step));
  return PartitionLattice{&*entry->second,times(index,integer(width))};
}

inline auto Lower::partition(const PartitionedRead& split,v::Value reader,const Offset& offset,const v::ElementType& type) -> s::Ref {
  SENTRYWARD_LEDGER(ledger->add("partition_queries");ledger->add("partition_regions_available",split.regions.size()));
  if(const auto* region=coveringRegion(split,offset,type))return view(region->view,offset,type);
  // A template-domain partition is not necessarily a complete rectangular
  // formula: independent row/column operands and causal zero masks can span
  // many such regions. Reuse the cached grid proof over retained raw pieces
  // before splitting the caller's output domain. partitionGrid never expands
  // compact source members; its finite input is the stored read fragments.
  if(const auto* grid=partitionGrid(split,reader,type))return view(*grid,offset,type);
  if(!split.template_domains)
    if(auto lattice=partitionLattice(split,reader,offset,type))return view(*lattice->view,lattice->offset,type);
  const auto* periodic=periodicPartition(split);
  if(periodic)
    if(const auto* region=coveringRegion(*periodic,offset,type))return view(region->view,offset,type);
  // Refine at the period (radix), not at every stored repetition of it; where the periodic
  // encoding gives no refinement, at the stored regions themselves.
  if(periodic)refinePartitionDomain(*periodic,offset,type);
  refinePartitionDomain(split,offset,type);
  SENTRYWARD_AUDIT({
    (void)audit;
    SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_4);
    for(auto d:domains)SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_5);
    SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_6);
    for(std::size_t i=0;i<std::min<std::size_t>(8,split.regions.size());++i) {
      const auto& r=split.regions[i];SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_7);
      for(const auto& b:r.view.blocks) {SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_8);for(const auto& a:b.axes)SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_9);}
      SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_10);
    }
  });
  fail("compact_view_unsupported","projection crosses independent read-result domains");
}

// Shared domain-only preflight: no Value projection and no scalar nodes.
// The same exact boundaries serve ordinary reads and MMA operand planning.
inline auto Lower::refinePartitionDomain(const PartitionedRead& split,const Offset& offset,const v::ElementType& type) -> void {
  SENTRYWARD_LEDGER(ledger->add("partition_refinement_queries");ledger->add("partition_refinement_regions",split.regions.size()));
  if(!split.regions.empty() && !split.regions[0].view.blocks.at(0).axes.empty()) {
    const auto outer=split.regions[0].view.blocks[0].axes.back();
    const bool grid=outer.result_stride && std::all_of(split.regions.begin(),split.regions.end(),
        [&](const auto& r){return r.view.blocks.size()==1 && !r.view.blocks[0].axes.empty() &&
          r.view.blocks[0].axes.back().result_stride==outer.result_stride;});
    if(grid) {
      // A single strided axis also has row gaps (e.g. interleaved columns).
      // Introduce the row radix once, then split only the within-row domain.
      // Never turn repeated row boundaries into one cut per logical row.
      if(output_refinement_depth && !offset.expression && offset.terms.size()==1) {
        const auto [axis,step]=*offset.terms.begin();
        if(axis<output_domain_rank && step>0 && outer.result_stride%std::uint64_t(step)==0) {
          const auto period=outer.result_stride/std::uint64_t(step);
          if(period>1 && domains[axis]>period && domains[axis]%period==0)
            throw OutputDomainRadix{axis,period};
          // A domain that is not a whole number of periods: split off the remainder, so that
          // the leading part takes the radix. Exact: a cut only splits the output domain.
          if(period>1 && domains[axis]>period) {
            const auto whole=domains[axis]-domains[axis]%period;
            throw OutputDomainCuts{{whole},axis};
          }
        }
      }
      const auto row_index=quotient(offset,outer.result_stride);
      const auto [row_lo,row_hi]=bounds(row_index);
      auto row=split;row.regions.clear();
      auto bands=split;bands.regions.clear();
      bool crosses=false;
      for(const auto& region:split.regions) {
        const auto& block=region.view.blocks[0];
        const auto first=region.begin/outer.result_stride;
        const auto count=block.axes.back().count;
        auto band=region;band.begin=first;band.bytes=count;
        band.view.blocks[0].axes.clear();bands.regions.push_back(std::move(band));
        if(row_hi<integer(first) || row_lo>=integer(first+count))continue;
        if(row_lo<integer(first) || row_hi>=integer(first+count)) {crosses=true;continue;}
        auto local=region;local.begin%=outer.result_stride;
        local.bytes-=(count-1)*outer.result_stride;
        local.view.blocks[0].result_offset=local.begin;
        local.view.blocks[0].axes.pop_back();row.regions.push_back(std::move(local));
      }
      // Split only at stored source-band boundaries, never at every row.
      // Once inside one band, all columns are checked in the local row domain.
      if(crosses){refinePartitionDomain(bands,row_index,bit(8));return;}
      if(!row.regions.empty()) {
        refinePartitionDomain(row,remainder(offset,outer.result_stride),type);return;
      }
    }
  }
  const auto [lo,hi]=bounds(offset);
  // Affine/add/scale/quotient maps (positive factors) are monotone in every binder.
  // Invert source boundaries using lower/upper envelopes for the remaining
  // coordinates: O(index axes * regions * log(domain) * index expression),
  // not Cartesian members. Varying remainder maps still need a radix domain.
  // A binder may also be non-increasing (a member index that counts down
  // over a fold axis); each binder needs one direction across all its terms.
  std::map<unsigned,int> varying_axes;
  const auto monotone=[&](auto&& self,const Offset& x)->bool {
    if(!x.expression) {
      for(auto [level,coefficient]:x.terms) {
        if(domains.at(level)==1 || !coefficient)continue;
        const int direction=coefficient>0?1:-1;
        if(const auto [it,added]=varying_axes.emplace(level,direction);!added && it->second!=direction)return false;
      }
      return true;
    }
    const auto& e=*x.expression;
    return self(self,e.left) &&
      (e.kind==OffsetExpression::Add ? self(self,e.right) :
       (e.kind==OffsetExpression::Scale || e.kind==OffsetExpression::Quotient) && e.factor>0);
  };
  if((output_refinement_depth || fold_owner) && lo>=0 && monotone(monotone,offset))
  for(const auto [axis_value,direction]:varying_axes) {
    const std::optional<unsigned> axis=axis_value;
    const auto at=[&](auto&& self,const Offset& x,std::uint64_t i,bool upper)->std::int64_t {
      if(!x.expression) {
        auto n=x.base;
        for(auto [level,c]:x.terms)
          n=add(n,mul(c,integer(level==*axis?i:(upper?domains[level]-1:0))));
        return n;
      }
      const auto& e=*x.expression;const auto n=self(self,e.left,i,upper);
      if(e.kind==OffsetExpression::Add)return add(n,self(self,e.right,i,upper));
      if(e.kind==OffsetExpression::Scale)return mul(n,e.factor);
      return n/e.factor;
    };
    OutputDomainCuts refinement;
    refinement.axis=*axis;
    const auto inverse=[&](std::uint64_t boundary) {
      // Bound the other index axes at both corners. These monotone envelopes
      // isolate source-boundary crossings without enumerating Cartesian
      // members; a subsequent query can refine another remaining axis.
      for(const bool upper:{false,true}) {
      std::uint64_t first=0,last=domains[*axis];
      while(first<last) {
        const auto middle=first+(last-first)/2;
        const auto value=at(at,offset,middle,upper);
        // The first binder value on the other side of the boundary.
        if(direction>0?value<integer(boundary):value>=integer(boundary))first=middle+1;else last=middle;
      }
      if(first && first<domains[*axis])refinement.cuts.push_back(first);
      }
    };
    for(const auto& region:split.regions) {
      inverse(region.begin);
      // Split also before an element would straddle a source boundary.
      if(region.bytes>=type.width()/8)
        inverse(region.begin+region.bytes-type.width()/8+1);
    }
    std::sort(refinement.cuts.begin(),refinement.cuts.end());
    refinement.cuts.erase(std::unique(refinement.cuts.begin(),refinement.cuts.end()),refinement.cuts.end());
    if(!refinement.cuts.empty()) {
      if(*axis>=output_domain_rank)refineFold(*axis,refinement.cuts);
      SENTRYWARD_AUDIT({
        (void)audit;static unsigned witnesses=0;
        if(sentryward::debug::environment("SENTRYWARD_OUTPUT_DOMAIN_WITNESS") && witnesses++<8) {
          SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_11);
          for(std::size_t i=0;i<std::min<std::size_t>(4,split.regions.size());++i) {
            const auto& r=split.regions[i];SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_12);
            SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_13);
            if(r.view.indexed_source) {
              const auto& source=*r.view.indexed_source;
              SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_14);
              if(source.root) {
                SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_15);
                if(source.root->binding)SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_16);
                if(!source.root->children.empty())SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_17);
              }
            } else for(const auto& source:r.view.sources)
              if(const auto* value=std::get_if<v::Value>(&source))
                SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_18);
            for(const auto& b:r.view.blocks) {SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_19);
              for(const auto& a:b.axes)SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_20);}
            SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_21);
          }
        }
      });
      throw refinement;
    }
  }
  // Periodic source selection stays rectangular in (quotient,remainder)
  // coordinates. Reshape the *output domain*, not every repeated row.
  const auto radix=[&](auto&& self,const Offset& x)->std::optional<OutputDomainRadix> {
    if(!x.expression)return {};
    const auto& e=*x.expression;
    if(e.kind==OffsetExpression::Remainder && e.factor>1 && !e.left.expression &&
       e.left.base==0 && e.left.terms.size()==1) {
      const auto [level,c]=*e.left.terms.begin();
      if(level<output_domain_rank && c==1 && domains[level]>std::uint64_t(e.factor) &&
         domains[level]%e.factor==0)return OutputDomainRadix{level,std::uint64_t(e.factor)};
    }
    if(auto found=self(self,e.left))return found;
    return self(self,e.right);
  };
  if(output_refinement_depth)if(auto split=radix(radix,offset))throw *split;
  SENTRYWARD_AUDIT({
    audit->event("DOMAIN_SPLIT_REQUIRED","read_projection_crosses_source_partition",split.regions.size(),1);
    SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_22);
    for(auto n:domains)SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_23);
    SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_24);
    for(std::size_t i=0;i<std::min<std::size_t>(split.regions.size(),12);++i) {
      const auto& r=split.regions[i];SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_25);
      if(r.view.indexed_source)SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_26);
      SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_27);
    }
    SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_28);
  });
}

} // namespace
} // namespace sentryward::value_analysis
