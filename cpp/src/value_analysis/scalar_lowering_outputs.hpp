#pragma once
#include "scalar_fold_lowering_impl.hpp"

namespace sentryward::value_analysis {
namespace {

// A reduced element (ReducedContents): base (op) contribution_1 (op) ..., one output
// element. The base is the prior stored value or, absent, the initial cell at the element.
// Floating add is ideal-real (order-free); integer add, min and max are exact. O(contributions).
inline auto Lower::reducedOutput(const OutputWrite& output) -> void {
  const auto& contents=*output.reduced;
  const auto width=contents.element.width()/8;
  if(!width || contents.element.width()%8 || output.region.bytes!=width)
    fail("scalar_fold_prerequisite","invalid reduced output extent");
  const bool floating=contents.element.kind()==v::ElementKind::Floating;
  const std::string name=contents.operation=="add"?"Add":contents.operation=="min"?"Minimum":
      contents.operation=="max"?"Maximum":"";
  if(name.empty() || (floating && name!="Add"))fail("scalar_fold_operation_unsupported","unsupported reduction operator");
  const auto type=contents.element.toString();
  emitOutputDomain(output.region,OutputDomain(1),Offset(0),[&](const Offset& at) {
    // A projection may collect the graph: the running result is pinned, so it stays current.
    PinnedRoot sum(*this,contents.base?
        project(contents.base->first,plus(Offset(integer(contents.base->second)),at),contents.element):
        repAbsoluteCell(output.region.allocation,offsetValue(plus(Offset(integer(output.region.offset)),at)),type));
    for(const auto& [value,offset]:contents.contributions) {
      auto next=project(value,plus(Offset(integer(offset)),at),contents.element);
      auto combined=floating?
          real_expressions.operation(name,type,{sum.value(),next},"ideal_arithmetic",domains.size()):
          emitOperation(name,type,{sum.value(),next});
      if(floating && !(combined==sum.value()) && !(combined==next))
        result.graph.markInstruction(combined,s::instruction_mode::synthesized);
      sum.value()=std::move(combined);
    }
    return sum.value();
  });
  if(options.collection_interval)collectCompletedOutputs();
}

// Whether the elements at offset + sum_i n_i*strides[i] (n_i < shape[i]),
// `width` bytes each, are compared entirely, not at all or in part. The hull
// first, O(axes); a hull that crosses uncompared bytes visits each row of
// contiguous elements, O(rows * axes), only for such hulls.
inline auto Lower::comparedIn(const std::string& allocation,std::uint64_t offset,std::uint64_t width,
                    const std::vector<std::uint64_t>& shape,const std::vector<std::uint64_t>& strides) const -> Compared {
  const auto found=compared_bytes.find(allocation);
  if(found==compared_bytes.end() || !width || shape.size()!=strides.size() ||
     std::find(shape.begin(),shape.end(),0)!=shape.end())return Compared::All;
  const auto& bytes=found->second;
  unsigned __int128 last=offset;
  for(std::size_t i=0;i<shape.size();++i)last+=static_cast<unsigned __int128>(strides[i])*(shape[i]-1);
  if(last+width>UINT64_MAX)return Compared::All;
  const auto end=static_cast<std::uint64_t>(last+width);
  if(bytes.covers(offset,end))return Compared::All;
  if(!bytes.meets(offset,end))return Compared::None;
  std::size_t row=shape.size();
  for(std::size_t i=0;i<shape.size() && row==shape.size();++i)if(shape[i]>1 && strides[i]==width)row=i;
  const auto row_bytes=row<shape.size()?width*shape[row]:width;
  std::vector<std::uint64_t> index(shape.size());
  bool all=true,any=false;
  for(;;) {
    auto at=offset;
    for(std::size_t i=0;i<shape.size();++i)at+=index[i]*strides[i];
    if(bytes.covers(at,at+row_bytes))any=true;
    else {all=false;any|=bytes.meets(at,at+row_bytes);}
    if(any && !all)return Compared::Some;
    auto i=shape.size();
    for(;i-->0;) {
      if(i==row)continue;
      if(++index[i]<shape[i])break;
      index[i]=0;
    }
    if(i==std::numeric_limits<std::size_t>::max())break;
  }
  return all?Compared::All:Compared::None;
}

inline auto Lower::comparedIn(const OutputWriteFamily& family) const -> Compared {
  const auto step=family.byte_stride<0?-static_cast<__int128>(family.byte_stride):family.byte_stride;
  const auto low=static_cast<__int128>(family.region.offset)-(family.byte_stride<0?step*(family.count-1):0);
  if(!family.count || low<0 || step>UINT64_MAX)return Compared::All;
  return comparedIn(family.region.allocation,static_cast<std::uint64_t>(low),family.region.bytes,
      {step?family.count:1},{static_cast<std::uint64_t>(step)});
}

// A store the binding compares only in part: project just those bytes, as
// runs of whole `element` values where elements are compared entirely and as
// aligned power-of-two bit slices of at most 8 bytes elsewhere, each an
// output family. Exact: every compared byte is projected once, no other
// byte is. O(compared pieces) families.
inline auto Lower::projectComparedBytes(const OutputWrite& output,const v::ElementType& element) -> void {
  const auto width=element.width()/8;
  if(!width)fail("scalar_fold_subbyte_value","sub-byte output requires bit projection");
  const auto& region=output.region;
  compared_bytes.at(region.allocation).pieces(region.offset,region.offset+region.bytes,
      [&](std::uint64_t lo,std::uint64_t hi) {
    const auto end=hi-region.offset;
    for(auto at=lo-region.offset;at<end;) {
      const auto inner=at%width;
      if(!inner && width<=8 && end-at>=width) {
        const auto count=(end-at)/width;
        outputFamily(output,at,count,element,width);at+=count*width;continue;
      }
      std::uint64_t bytes=8;
      while(bytes>end-at || bytes>width-inner || inner%bytes)bytes/=2;
      outputFamily(output,at,1,bit(unsigned(8*bytes)),bytes);at+=bytes;
    }
  });
}

inline auto Lower::outputFamily(const OutputWrite& output,std::uint64_t begin,std::uint64_t count,
                  const v::ElementType& type,std::uint64_t bytes) -> void {
  SENTRYWARD_AUDIT(audit->add(AuditCounter::output_elements_requested,count));
  if(result.outputs.size()>=options.max_outputs) {
    collectCompletedOutputs(true);
    if(result.outputs.size()>=options.max_outputs)
      fail("output_family_limit_exceeded","output family budget exceeded");
  }
    Offset offset(integer(output.value_byte_offset+begin));
    if(count>1)offset.terms[0]=integer(bytes);
    auto region=output.region;region.offset+=begin;region.bytes=count*bytes;
    emitOutputDomain(region,OutputDomain(count),offset,[&](const Offset& at){return project(output.value,at,type);});
    // The growth threshold is O(1); a family is not one scalar output.
    // Use the same boundary as normalized output domains, rather than
    // retaining intermediate projections until 128 families accumulate.
    if(options.collection_interval)collectCompletedOutputs();
    return;
}

inline auto Lower::spatialOutput(std::span<const OutputWrite> outputs) -> bool {
  if(outputs.empty())return false;
  std::vector<v::Value> values;values.reserve(outputs.size());
  for(const auto& output:outputs)values.push_back(output.value);
  // The shared-subtree policy where the ordinary plan leaves single records: records of one
  // output row that differ only in leaves under subtrees every record shares (the row's own
  // state) are then one run. A call that the ordinary plan already groups into runs is untouched,
  // and only calls with single records pay the extra planning pass. The shared plan
  // is taken only when each of its runs has at least kSharedRunMinRecords records: short runs
  // of single records are cheaper on the per-record path, whose outputs the consolidation
  // merges (merge_state: runs of 4 gave 64x the output families and 6x the time). 16 is the
  // consolidation's own minimum for merging rows (kTensorMinRows, consolidate_outputs.cpp).
  static constexpr std::size_t kSharedRunMinRecords=16;
  auto plan=pure_source_templates.map(values,true);
  bool shared_plan=false;
  if(std::any_of(plan.begin(),plan.end(),[](const auto& domain){return domain.count==1;})) {
    auto shared=pure_source_templates.map(values,true,{},true);
    if(pure_source_templates.last_map_shared &&
       std::all_of(shared.begin(),shared.end(),[](const auto& domain){
         return domain.count==1 || domain.count>=kSharedRunMinRecords;})) {
      plan=std::move(shared);shared_plan=true;
    }
  }
#if defined(SENTRYWARD_DEVELOPMENT_DIAGNOSTICS) || defined(SENTRYWARD_ENABLE_COMPRESSION_AUDIT)
#include "sentryward/development/blocks/scalar_fold_lowering_trace_3.inc"
#endif
  // Validate addresses of the existing records, never synthesize members.
  // The resulting output stride and source-member stride are independent.
  for(const auto& domain:plan) {
    const auto& first=outputs[domain.begin];
    if(!first.region.bytes || domain.count>UINT64_MAX/first.region.bytes)return false;
    const auto step=domain.count>1?
        outputs[domain.begin+domain.result_stride].region.offset-first.region.offset:first.region.bytes;
    if(!step)return false;
    for(std::size_t i=0;i<domain.count;++i) {
      const auto& item=outputs[domain.begin+i*domain.result_stride];
      if(item.region.allocation!=first.region.allocation || item.region.bytes!=first.region.bytes ||
         item.value_byte_offset!=first.value_byte_offset ||
         static_cast<unsigned __int128>(first.region.offset)+
           static_cast<unsigned __int128>(step)*i!=item.region.offset)return false;
    }
  }
  struct Grid {std::size_t first,rows=1;std::uint64_t outer_stride=0;};
  std::vector<Grid> grids;
  std::map<std::tuple<std::size_t,std::size_t,std::size_t,std::int64_t,std::uint64_t>,std::size_t> open;
  for(std::size_t i=0;i<plan.size();++i) {
    const auto& d=plan[i];const auto& first=outputs[d.begin];
    // A shared plan stays one-axis: its rows are merged later by the consolidation, under
    // its own conditions, as view normalization does for the same policy.
    if(shared_plan){grids.push_back({i});continue;}
    const auto root=d.source?d.source->root:nullptr;
    if(!root || root->kind!=IndexedSourceUse::Kind::CoordinateMap || !root->binding ||
       root->binding->period || !root->binding->quotient_terms.empty() ||
       root->children.size()!=1 || d.source->zero_mask || d.count<2) {
      grids.push_back({i});continue;
    }
    const auto& map=*root->binding;
    const auto key=std::tuple{indexed_use_identity.use(root->children[0]),d.count,d.result_stride,
        map.stride,first.value_byte_offset};
    auto found=open.find(key);bool extended=false;
    if(found!=open.end()) {
      auto& grid=grids[found->second];const auto& base=plan[grid.first];
      const auto& origin=outputs[base.begin];
      const auto column_stride=outputs[base.begin+base.result_stride].region.offset-origin.region.offset;
      const auto current_stride=outputs[d.begin+d.result_stride].region.offset-first.region.offset;
      const auto outer=grid.rows==1?first.region.offset-origin.region.offset:grid.outer_stride;
      const auto expected=static_cast<__int128>(base.source->root->binding->base)+
          static_cast<__int128>(grid.rows)*d.count*map.stride;
      if(expected==map.base && current_stride==column_stride && origin.region.allocation==first.region.allocation &&
         origin.region.bytes==first.region.bytes &&
         static_cast<unsigned __int128>(outer)>=(static_cast<unsigned __int128>(d.count)-1)*column_stride+first.region.bytes &&
         static_cast<unsigned __int128>(origin.region.offset)+static_cast<unsigned __int128>(grid.rows)*outer==first.region.offset) {
        ++grid.rows;grid.outer_stride=outer;extended=true;
      }
    }
    if(!extended){open.insert_or_assign(key,grids.size());grids.push_back({i});}
  }
  SENTRYWARD_AUDIT((void)audit;if(sentryward::debug::environment("SENTRYWARD_OUTPUT_DOMAIN_WITNESS"))
    SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_51));
  // If a strided family is refined all the way to individual records,
  // continuing whole columns separates reuse of row-local subexpressions.
  // The records already exist: after observing that refinement, project
  // the remaining records in their original (address) order. Keep the
  // template path for grids that actually retain compact output domains.
  const bool strided_single_axis=grids.size()>1 &&
    std::all_of(grids.begin(),grids.end(),[&](const auto& grid) {
      const auto& d=plan[grid.first];
      return grid.rows==1 && d.count>1 && d.result_stride>1;
    });
  std::vector<unsigned char> completed_records;
  if(strided_single_axis)completed_records.resize(outputs.size());
  for(const auto& grid:grids) {
    const auto& domain=plan[grid.first];
    const auto& first=outputs[domain.begin];
    if(!domain.source) {
      outputFamily(first,0,1,input.type(first.value).element(),first.region.bytes);continue;
    }
    const auto step=outputs[domain.begin+domain.result_stride].region.offset-first.region.offset;
    if(grid.rows>UINT64_MAX/domain.count || grid.rows*domain.count>UINT64_MAX/first.region.bytes)
      fail("index_representation_overflow","output grid payload overflow");
    const auto count=grid.rows*domain.count;
    auto region=first.region;region.bytes*=count;
    auto source=std::make_shared<IndexedSource>(*domain.source);source->count=count;
    output_source_patterns.push_back(source);
    Offset index;index.terms[0]=1;
    OutputDomain layout({domain.count},{step});
    if(grid.rows>1){index.terms[0]=integer(domain.count);index.terms[1]=1;
      layout=OutputDomain({grid.rows,domain.count},{grid.outer_stride,step});}
    const auto outputs_before=result.outputs.size();
    emitOutputDomain(region,std::move(layout),index,[&](const Offset& at) {
      return inContext(std::make_shared<IndexedContext>(IndexedContext{at,source->root}),[&] {
        // The source plan owns the representative and its coordinate map;
        // the output record supplies only destination/payload information.
        return project(source->prototype,Offset(integer(first.value_byte_offset)),input.type(first.value).element());
      });
    });
    ++result.spatial_families_projected;result.spatial_outputs_projected+=count;
    const bool fully_refined=result.outputs.size()-outputs_before==count;
    if(strided_single_axis) {
      for(std::size_t i=0;i<domain.count;++i)
        completed_records[domain.begin+i*domain.result_stride]=1;
    }
    if(options.collection_interval)collectCompletedOutputs();
    if(strided_single_axis && fully_refined) {
      for(std::size_t i=0;i<outputs.size();++i)if(!completed_records[i]) {
        const auto& item=outputs[i];
        outputFamily(item,0,1,input.type(item.value).element(),item.region.bytes);
        if(options.collection_interval)collectCompletedOutputs();
      }
      return true;
    }
  }
  return true;
}

inline auto Lower::spatialStoreFamily(const OutputWriteFamily& family) -> bool {
  if(!options.compact_output_families || family.count<2 ||
     !family.region.bytes || family.count>UINT64_MAX/family.region.bytes ||
     family.region.offset>UINT64_MAX-family.count*family.region.bytes ||
     family.byte_stride<0 || std::uint64_t(family.byte_stride)!=family.region.bytes)return false;
  const auto& store=input.node(family.first.node);
  if(!store.hasUniformThreadTemplate())return false;
  const auto operand=store.operands()[1];
  if(input.resolve(operand)!=operand || operand.conversion() || operand.pointer() ||
     !operand.indices().empty() || operand.threadSelector() || !input.type(operand).isScalar())return false;
  const auto& child=input.definingNode(operand);
  if(child.storedInstanceCount()!=1)return false;
  auto base=operand;std::int64_t stride=0;
  const auto& threads=child.instanceThreads();
  if(child.origin().cta==store.origin().cta && !threads.empty() &&
     threads.at(operand.instanceIndex())==store.origin().thread) {
    if(threads.storageIdentity()!=store.instanceThreads().storageIdentity())return false;
    base=child.result(operand.resultIndex(),family.first.instance);stride=1;
  }
  spatial_count=family.count;++spatial_serial;
  spatial_values.clear();spatial_instance_strides.clear();domains={spatial_count};
  if(options.unit_lifting)setOutputCta(family.at(input,0));
  try {
    registerSpatialFamily({&input,base,spatial_count,stride,{}});
    const auto& type=input.type(base);
    auto body=project(base,Offset{},type.element());
    auto region=family.region;region.bytes*=family.count;
    output_domains.emplace(result.outputs.size(),family.count);
    pushOutput(region,body);
    ++result.spatial_families_projected;result.spatial_outputs_projected+=family.count;
  } catch(const Failure& error) {
    spatial_count=0;spatial_values.clear();spatial_instance_strides.clear();domains.clear();
    if(error.code=="scalar_fold_spatial_projection" || error.code=="scalar_fold_non_affine_projection" ||
       (error.code.ends_with("_limit_exceeded") || error.code=="index_representation_overflow" || error.code=="invalid_analysis_budget")) {
      visiting.clear();collectCompletedOutputs((error.code.ends_with("_limit_exceeded") || error.code=="index_representation_overflow" || error.code=="invalid_analysis_budget"));return false;
    }
    throw;
  }
  spatial_count=0;spatial_values.clear();spatial_instance_strides.clear();domains.clear();
  return true;
}

// A store family whose compared members are all compared entirely: one spatial family, else
// the spatial domains recovered from its members, else one output per member.
inline auto Lower::storeFamily(const OutputWriteFamily& family) -> void {
  if(spatialStoreFamily(family))return;
  if(options.unit_lifting)setOutputCta(family.at(input,0));
  if(options.compact_output_families && family.count>1 && family.byte_stride>=0 &&
     std::uint64_t(family.byte_stride)==family.region.bytes) {
    std::vector<OutputWrite> outputs;outputs.reserve(family.count);
    SENTRYWARD_AUDIT(audit->event("INSTANCE_ENUMERATION","output_family_materialized_for_spatial_recovery",
      family.count,1,family.first.node,true));
    for(std::size_t i=0;i<family.count;++i)outputs.push_back(family.at(input,i));
    if(spatialOutput(outputs))return;
  }
  SENTRYWARD_AUDIT(if(family.count>1)audit->event("INSTANCE_ENUMERATION","output_family_scalar_fallback",
    family.count,1,family.first.node,true));
  for(std::size_t i=0;i<family.count;++i) {
    const auto output=family.at(input,i);
    const auto& type=input.type(output.value);
    setOutputCta(output);
    outputFamily(output,0,1,type.element(),output.region.bytes);
  }
}

// A store family the binding compares in part (comparedIn(family) is Some), in member
// order: every maximal run of members compared entirely as one sub-family (storeFamily),
// every member compared in part at its compared bytes (projectComparedBytes), no member
// without a compared byte. Members lie at least their width apart (memory elimination), so
// one is compared entirely exactly when it lies inside one maximal compared piece, and in
// part when it extends past an end of a piece it meets. Whole members of consecutive
// pieces join one run (possible only when members have gaps between them: the byte that
// separates two pieces otherwise lies in a member). O(compared pieces of the family's hull
// + runs), never per member.
inline auto Lower::comparedStoreFamily(const OutputWriteFamily& family) -> void {
  const auto width=family.region.bytes,count=std::uint64_t(family.count);
  const auto magnitude=family.byte_stride<0?0-std::uint64_t(family.byte_stride):
      std::uint64_t(family.byte_stride);
  const auto step=count>1?magnitude:width;
  if(!width || step<width)fail("scalar_fold_prerequisite","overlapping output family members");
  // Positions count members in address order; comparedIn bounded the hull [low, high).
  const auto low=family.byte_stride<0?family.region.offset-step*(count-1):family.region.offset;
  const auto high=low+step*(count-1)+width;
  struct Run {std::uint64_t begin,end;bool whole;};
  std::vector<Run> runs;
  std::uint64_t next=0; // the first position no run holds
  const auto take=[&](std::uint64_t begin,std::uint64_t end,bool whole) {
    begin=std::max(begin,next);
    if(begin>=end)return;
    if(whole && !runs.empty() && runs.back().whole && runs.back().end==begin)runs.back().end=end;
    else runs.push_back({begin,end,whole});
    next=end;
  };
  compared_bytes.at(family.region.allocation).pieces(low,high,[&](std::uint64_t lo,std::uint64_t hi) {
    const auto first=(lo-low)/step,last=(hi-1-low)/step;
    const auto into_first=(lo-low)%step,into_last=(hi-1-low)%step;
    // Positions [inside, end) lie within [lo, hi).
    const auto inside=first+(into_first!=0),end=hi-low>=width?(hi-low-width)/step+1:0;
    if(into_first && into_first<width)take(first,first+1,false); // also holds bytes before lo
    if(inside<end)take(inside,end,true);
    if(last>=std::max(inside,end) && into_last<width)take(last,last+1,false); // also holds bytes from hi
  });
  const bool reversed=family.byte_stride<0;
  for(std::size_t k=0;k<runs.size();++k) {
    const auto& run=runs[reversed?runs.size()-1-k:k];
    const auto member=reversed?count-run.end:run.begin; // the run's first member
    if(run.whole) {
      auto part=family;
      part.first.instance+=member;part.count=run.end-run.begin;part.region=family.at(input,member).region;
      storeFamily(part);
      continue;
    }
    const auto output=family.at(input,member);
    setOutputCta(output);
    projectComparedBytes(output,input.type(output.value).element());
  }
}

inline auto Lower::validateRadixPartition(const PartitionedRead& split) -> void {
  // First reduce coverage-only adjacent axes to a progression of intervals.
  // This does not alter the retained member/payload maps. Mixed row strides
  // need not share a radix (e.g. a missing source at one row breaks an AP).
  struct Span {std::uint64_t first,width,step,count;};
  std::vector<Span> spans;bool strided=true;
  unsigned __int128 span_coverage=0;
  const auto extent=split.type.bitSize()/8;
  for(const auto& region:split.regions) {
    if(region.view.blocks.size()!=1){strided=false;break;}
    const auto& b=region.view.blocks[0];
    if(!b.bytes || b.result_offset!=region.begin)fail("compact_view_unsupported","invalid strided domain origin");
    Span span{b.result_offset,b.bytes,0,1};
    for(const auto& axis:b.axes) {
      if(!axis.count)fail("compact_view_unsupported","empty strided domain axis");
      if(axis.count==1)continue;
      if(span.count==1 && axis.result_stride==span.width) {
        if(span.width>UINT64_MAX/axis.count)fail("compact_view_unsupported","strided domain width overflow");
        span.width*=axis.count;
      } else if(span.count==1 && axis.result_stride>=span.width) {
        span.step=axis.result_stride;span.count=axis.count;
      } else if(static_cast<unsigned __int128>(span.step)*span.count==axis.result_stride &&
                span.count<=UINT64_MAX/axis.count)span.count*=axis.count;
      else {strided=false;break;}
    }
    if(!strided)break;
    const auto bounding=static_cast<unsigned __int128>(span.count-1)*span.step+span.width;
    if(bounding!=region.bytes || span.first+bounding>extent)
      fail("compact_view_unsupported","strided domain bounding extent mismatch");
    span_coverage+=static_cast<unsigned __int128>(span.width)*span.count;
    spans.push_back(span);
  }
  if(strided) {
    const auto overlaps=[](const Span& a,const Span& b) {
      using Wide=__int128;
      const auto end=[](const Span& x){return Wide(x.first)+Wide(x.count-1)*x.step+x.width;};
      if(end(a)<=b.first || end(b)<=a.first)return false;
      const auto interval=[](const Span& one,const Span& many) {
        const auto lo=Wide(one.first)-Wide(many.width-1),hi=Wide(one.first)+one.width-1;
        if(many.count==1)return Wide(many.first)>=lo && Wide(many.first)<=hi;
        const auto first=lo<=many.first?Wide(0):(lo-many.first+many.step-1)/many.step;
        const auto last=hi< many.first?Wide(-1):std::min(Wide(many.count-1),(hi-many.first)/many.step);
        return first<=last;
      };
      if(a.count==1)return interval(a,b);
      if(b.count==1)return interval(b,a);
      // Only shifts congruent to the start difference modulo gcd can
      // intersect. Usually one or two tests for aligned fragment strides;
      // worst case depends on stored payload width/gcd, not member count.
      const Wide g=std::gcd(a.step,b.step);
      const Wide low=-Wide(a.width-1),high=Wide(b.width-1);
      auto correction=(Wide(a.first)-b.first-low)%g;if(correction<0)correction+=g;
      for(auto delta=low+correction;delta<=high;delta+=g) {
        const Wide af=Wide(a.first)+(delta<0?-delta:0),bf=Wide(b.first)+(delta>0?delta:0);
        if(af>UINT64_MAX || bf>UINT64_MAX)continue;
        const auto ac=std::min(Wide(a.count),1+(Wide(UINT64_MAX)-af)/a.step);
        const auto bc=std::min(Wide(b.count),1+(Wide(UINT64_MAX)-bf)/b.step);
        if(algorithm::IntersectAffine({std::uint64_t(af),a.step,std::uint64_t(ac)},
            {std::uint64_t(bf),b.step,std::uint64_t(bc)}).count)return true;
      }
      return false;
    };
    for(std::size_t i=0;i<spans.size();++i)for(std::size_t j=0;j<i;++j)
      if(overlaps(spans[i],spans[j]))fail("compact_view_unsupported","overlapping strided result domains");
    if(span_coverage!=extent)fail("compact_view_unsupported","incomplete strided result domains");
    return;
  }
  // A stored mixed-radix box has one interval per digit. Coverage and
  // disjointness cost O(R^2 * D + A log A), not its represented cardinality.
  // R = source rules, D = distinct axis strides, A = stored axes.
  const auto total=split.type.bitSize()/8;
  const auto unit=split.regions.front().view.blocks.at(0).bytes;
  if(!unit || total%unit)fail("compact_view_unsupported","invalid radix view unit");
  std::vector<std::uint64_t> strides{unit,total};
  for(const auto& region:split.regions) {
    if(region.view.blocks.size()!=1)fail("compact_view_unsupported","radix domain needs one source rule");
    const auto& block=region.view.blocks[0];
    if(block.bytes!=unit || block.result_offset!=region.begin || region.begin%unit)
      fail("compact_view_unsupported","inconsistent radix domain origin or unit");
    for(const auto& axis:block.axes) {
      if(!axis.count || !axis.result_stride || axis.result_stride%unit || axis.result_stride>total)
        fail("compact_view_unsupported","invalid radix domain axis");
      strides.push_back(axis.result_stride);
    }
  }
  std::sort(strides.begin(),strides.end());strides.erase(std::unique(strides.begin(),strides.end()),strides.end());
  for(std::size_t i=1;i<strides.size();++i)if(strides[i]%strides[i-1])
    fail("compact_view_unsupported","view axes do not share a radix grid");
  using Interval=std::pair<std::uint64_t,std::uint64_t>;
  std::vector<std::vector<Interval>> boxes;unsigned __int128 covered=0;
  for(const auto& region:split.regions) {
    const auto& block=region.view.blocks[0];
    if(region.begin>=total)fail("compact_view_unsupported","radix domain exceeds read");
    std::vector<Interval> box;
    for(std::size_t d=0;d+1<strides.size();++d) {
      const auto first=(region.begin/strides[d])%(strides[d+1]/strides[d]);
      box.emplace_back(first,first+1);
    }
    unsigned __int128 extent=unit,volume=1;
    std::set<std::uint64_t> varying;
    for(const auto& axis:block.axes) {
      if(axis.count==1)continue;
      const auto d=std::size_t(std::lower_bound(strides.begin(),strides.end(),axis.result_stride)-strides.begin());
      if(d+1>=strides.size() || !varying.insert(axis.result_stride).second ||
          axis.count>strides[d+1]/strides[d]-box[d].first)
        fail("compact_view_unsupported","radix domain crosses a digit boundary");
      box[d].second=box[d].first+axis.count;
      extent+=static_cast<unsigned __int128>(axis.count-1)*axis.result_stride;
      volume*=axis.count;
    }
    if(extent!=region.bytes || static_cast<unsigned __int128>(region.begin)+extent>total)
      fail("compact_view_unsupported","radix domain bounding extent mismatch");
    for(const auto& prior:boxes) {
      bool overlaps=true;
      for(std::size_t d=0;d<box.size();++d)
        if(box[d].second<=prior[d].first || prior[d].second<=box[d].first){overlaps=false;break;}
      if(overlaps)fail("compact_view_unsupported","overlapping radix result domains");
    }
    covered+=volume;boxes.push_back(std::move(box));
  }
  if(covered!=total/unit)fail("compact_view_unsupported","incomplete radix result domains");
}

} // namespace
} // namespace sentryward::value_analysis
