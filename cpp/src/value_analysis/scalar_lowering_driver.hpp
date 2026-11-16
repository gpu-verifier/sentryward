#pragma once
#include "scalar_fold_lowering_impl.hpp"

namespace sentryward::value_analysis {
namespace {

inline auto Lower::run() -> ScalarFoldResult {
  SENTRYWARD_AUDIT_STAGE("scalar_fold");
  SENTRYWARD_AUDIT(audit->add(AuditCounter::input_output_records,memory.outputs.size());
    audit->add(AuditCounter::input_output_families,memory.output_families.size());
    audit->event("INPUT_REPRESENTATION","scalar_fold_output_sources",
      memory.outputCount(),memory.output_families.size()));
  result.source_outputs=memory.outputCount();
  SENTRYWARD_AUDIT({(void)audit;if(sentryward::debug::environment("SENTRYWARD_OUTPUT_DOMAIN_WITNESS"))
    for(std::size_t i=0;i<std::min<std::size_t>(8,memory.outputs.size());++i) {
      const auto& x=memory.outputs[i];
      SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_52);
    }
  });
  auto& stop=result.failure_site; // first-item rule bookkeeping (phases: ScalarFoldResult)
  try{
    stop.phase='a';
    if(!memory.complete || !memory.views_normalized)fail("scalar_fold_prerequisite","complete memory elimination and normalized views are required");
    stop.phase='b';
    for(const auto& [key,read]:memory.normalized_reads) {
      (void)key;
      if(const auto* split=std::get_if<PartitionedRead>(&read)) {
        if(split->template_domains && !split->regions.empty()) {
          const auto& first_block=split->regions.front().view.blocks.at(0);
          const auto outer=first_block.axes.empty()?0:first_block.axes.back().result_stride;
          if(std::any_of(split->regions.begin(),split->regions.end(),[&](const auto& r) {
              return r.view.blocks.size()==1 && (r.view.blocks[0].axes.size()>2 ||
                  (!r.view.blocks[0].axes.empty() && r.view.blocks[0].axes.back().result_stride!=outer));
            })) {
            try {validateRadixPartition(*split);}catch(const Failure&) {
              SENTRYWARD_AUDIT((void)audit;SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_53);
                for(const auto& r:split->regions) {SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_54);
                  for(const auto& axis:r.view.blocks[0].axes)SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_55);
                  SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_56);});
              throw;
            }
            continue;
          }
          // Check stored AP domains, not their members. A common byte grid
          // makes overlap exactly an AP intersection of cell indices.
          const auto unit=split->regions.front().view.blocks.at(0).bytes;
          const auto& first_axes=split->regions.front().view.blocks.at(0).axes;
          const bool row_grid=first_axes.size()==2;
          const auto row_stride=row_grid?first_axes.back().result_stride:split->type.bitSize()/8;
          const auto row_count=row_grid && row_stride?split->type.bitSize()/8/row_stride:1;
          const bool uniform=unit && std::all_of(split->regions.begin(),split->regions.end(),
            [&](const auto& r){return r.view.blocks.size()==1 &&
              (row_grid?(r.view.blocks[0].axes.size()==2 &&
                  r.view.blocks[0].axes.back().result_stride==row_stride &&
                  r.view.blocks[0].axes.back().count>0):r.view.blocks[0].axes.size()<=1) &&
              r.view.blocks[0].bytes==unit && r.begin%unit==0;});
          if(uniform) {
            if(!row_stride || !row_count || row_stride%unit ||
                static_cast<unsigned __int128>(row_stride)*row_count!=split->type.bitSize()/8)
              fail("compact_view_unsupported","invalid template row coverage");
            std::vector<algorithm::AffineDomain> sets;
            std::vector<std::pair<std::uint64_t,std::uint64_t>> row_ranges;
            unsigned __int128 covered=0;
            for(const auto& region:split->regions) {
              const auto& b=region.view.blocks[0];
              if((!b.axes.empty() &&
                  (!b.axes[0].count || b.axes[0].result_stride%unit)))
                fail("compact_view_unsupported","invalid template result progression");
              const auto first_row=row_grid?region.begin/row_stride:0;
              const auto rows=row_grid?b.axes.back().count:1;
              if(first_row>=row_count || rows>row_count-first_row)
                fail("compact_view_unsupported","template row range exceeds read");
              algorithm::AffineDomain d{(region.begin%row_stride)/unit,b.axes.empty()?0:b.axes[0].result_stride/unit,
                  b.axes.empty()?1:b.axes[0].count};
              if(d.last()>=row_stride/unit || (!d.step && d.count>1))
                fail("compact_view_unsupported","template result progression exceeds read");
              covered+=static_cast<unsigned __int128>(d.count)*rows;
              sets.push_back(d);
              row_ranges.emplace_back(first_row,first_row+rows);
            }
            // Row intervals may be disjoint even when column APs coincide;
            // AffineIndex requires disjoint one-dimensional keys. Sweep
            // stored row intervals and test only row-overlapping candidates.
            // O(R log R + C * AP-intersection), C <= R*(R-1)/2, no members.
            std::vector<std::size_t> order(sets.size());
            std::iota(order.begin(),order.end(),0);
            std::sort(order.begin(),order.end(),[&](auto a,auto b){return row_ranges[a]<row_ranges[b];});
            std::multimap<std::uint64_t,std::size_t> active;
            for(auto i:order) {
              active.erase(active.begin(),active.upper_bound(row_ranges[i].first));
              for(auto [end,j]:active) {
                (void)end;
                if(algorithm::IntersectAffine(sets[i],sets[j]).count)
                  fail("compact_view_unsupported","overlapping template result domains");
              }
              active.emplace(row_ranges[i].second,i);
            }
            if(covered!=split->type.bitSize()/8/unit || split->type.bitSize()/8%unit)
              fail("compact_view_unsupported","incomplete template result domains");
            continue;
          }
        }
        std::uint64_t cursor=0;
        for(const auto& region:split->regions) {
          if(!region.bytes || region.begin!=cursor || region.bytes>UINT64_MAX-cursor)
            fail("compact_view_unsupported","read-result domains must form a disjoint complete partition");
          cursor+=region.bytes;
        }
        if(cursor!=split->type.bitSize()/8)fail("compact_view_unsupported","incomplete read-result domains");
      }
    }
    stop.phase='c';
    if(!options.max_nodes || !options.max_outputs || !options.max_recursion || !options.max_pending_projections)fail("invalid_analysis_budget","scalar/Fold budgets must be positive");
    next_collection=std::min(options.construction_collection_interval,
                             std::max<std::size_t>(1,options.max_nodes/4));
    next_completed_collection=next_collection;
    // Reachable undefined behavior precedes every output (and every value failure).
    checkSparseMetadata();
    stop.phase='d';stop.output_families=!memory.output_families.empty();
    for(const auto& family:memory.output_families) {
      const auto coverage=comparedIn(family);
      if(coverage==Compared::None)continue;
      if(coverage==Compared::Some)comparedStoreFamily(family);
      else storeFamily(family);
    }
    std::size_t spatial_failed_end=0,wide_failed_end=0;
    stop.phase='e';
    if((!memory.normalized_output_groups.empty() || !memory.normalized_output_members.empty()) &&
       memory.normalized_output_members.size()!=memory.outputs.size())
      fail("scalar_fold_prerequisite","invalid normalized output membership");
    for(std::size_t output_index=0;output_index<memory.outputs.size();++output_index){
      setOutputCta(memory.outputs[output_index]);
      if(options.compact_output_families) {
        if(const auto found=memory.normalized_output_groups.find(output_index);
           found!=memory.normalized_output_groups.end()) {
          const auto& group=found->second;
          beginItem(output_index);
          output_base=memory.outputs[output_index].base;
          if(group.records.empty() || group.records.front()!=output_index ||
             group.records.back()>=memory.outputs.size())
            fail("scalar_fold_prerequisite","invalid normalized output record range");
          includeRecordUnits(group.records);
          for(const auto& output:group.domains) {
            const auto width=output.element_type.width()/8;
            if(!width || !output.region.bytes || output.region.bytes%width)
              fail("scalar_fold_prerequisite","invalid numerical output view extent");
            const auto coverage=comparedIn(output.region.allocation,output.region.offset,width,
                output.shape,output.strides_bytes);
            if(coverage==Compared::None)continue;
            if(coverage==Compared::Some)
              fail("output_comparison_partial_view","a numerical output view stores bytes the binding compares "
                   "and bytes it does not compare; only whole stored records are projected at compared bytes");
            if(result.outputs.size()>=options.max_outputs) {
              collectCompletedOutputs(true);
              if(result.outputs.size()>=options.max_outputs)
                fail("output_family_limit_exceeded","output family budget exceeded");
            }
            const auto count=output.region.bytes/width;
            OutputDomain layout(output.shape,output.strides_bytes);
            if(layout.count!=count)fail("scalar_fold_prerequisite","output view shape does not cover payload");
            SENTRYWARD_AUDIT(audit->add(AuditCounter::output_elements_requested,count));
            domains=output.shape;Offset offset(integer(output.source_byte_offset));
            std::uint64_t step=width;
            for(std::size_t axis=domains.size();axis-->0;) {
              if(domains[axis]>1)offset.terms[axis]=integer(step);
              if(step>UINT64_MAX/domains[axis])fail("scalar_fold_prerequisite","output projection extent overflow");
              step*=domains[axis];
            }
            emitOutputDomain(output.region,std::move(layout),offset,
                [&](const Offset& at){return view(output.view,at,output.element_type);});
            ++result.spatial_families_projected;result.spatial_outputs_projected+=count;
            // A template can cover thousands of outputs. Poll the O(1)
            // node-growth guard per completed template, not per 128 templates.
            if(options.collection_interval)collectCompletedOutputs();
          }
          continue;
        }
        if(!memory.normalized_output_members.empty() && memory.normalized_output_members[output_index])continue;
      }
      const auto& output=memory.outputs[output_index];
      const auto coverage=comparedIn(output.region.allocation,output.region.offset,output.region.bytes);
      if(coverage==Compared::None)continue;
      beginItem(output_index);
      output_base=output.base;
      if(output.reduced){setOutputCta(output);reducedOutput(output);continue;}
      const auto& type=input.type(output.value);
      const auto element=output.element_type.value_or(type.element());
      if(element.isParameterized())fail("scalar_fold_parameterized_type","output element type is unresolved");
      const auto element_bytes=element.width()/8;
      if(!element_bytes)fail("scalar_fold_subbyte_value","sub-byte output requires bit projection");
      if(element_bytes>kWidestScalarBytes && element.kind()!=v::ElementKind::Floating &&
         element_bytes%kWidestScalarBytes==0 && output.region.bytes%kWidestScalarBytes==0) {
        // A run of packed numeric records: their fields as one numerical spatial domain plan.
        // Otherwise unsigned 64-bit fields per record.
        if(options.compact_output_families && coverage==Compared::All && output_index>=wide_failed_end) {
          std::size_t end=output_index;
          if(auto fields=wideRecordRun(output_index,end)) {
            includeRecordUnits(std::views::iota(output_index,end));
            if(spatialOutput(*fields)){output_index=end-1;continue;}
            wide_failed_end=end;
            setOutputCta(output);
          }
        }
        wideOutput(output,coverage);continue;
      }
      if(coverage==Compared::Some){projectComparedBytes(output,element);continue;}
      if(options.compact_output_families && output_index>=spatial_failed_end && type.isScalar() &&
         element==type.element() && output.region.bytes==element_bytes && element_bytes<=8) {
        auto end=output_index+1;
        const auto cta=input.instanceOrigin(output.value).cta;
        while(end<memory.outputs.size()) {
          const auto& next=memory.outputs[end];
          const auto offset=static_cast<__int128>(output.region.offset)+static_cast<__int128>(end-output_index)*element_bytes;
          if(next.region.allocation!=output.region.allocation || next.region.bytes!=element_bytes ||
             offset!=next.region.offset || input.type(next.value)!=type ||
             next.element_type.value_or(type.element())!=element ||
             next.value_byte_offset!=output.value_byte_offset || input.instanceOrigin(next.value).cta!=cta ||
             next.base!=output.base || next.reduced ||
             comparedIn(next.region.allocation,next.region.offset,next.region.bytes)!=Compared::All)break;
          ++end;
        }
        // Node identity is not a source-pattern boundary after memory
        // elimination. Distinct scalar producers may implement one affine
        // output domain. Plan/validate that domain before constructing any
        // scalar expressions; do not build N outputs and recover it later.
        if(end-output_index>1) {
          includeRecordUnits(std::views::iota(output_index,end));
          if(spatialOutput(std::span(memory.outputs).subspan(output_index,end-output_index))) {
            output_index=end-1;continue;
          }
          spatial_failed_end=end;
          setOutputCta(output); // the single record's own unit again
        }
      }
      if(options.compact_output_families && output.region.bytes &&
          output.region.bytes%element_bytes==0 && element_bytes<=8) {
        outputFamily(output,0,output.region.bytes/element_bytes,element,element_bytes);
        continue;
      }
      std::uint64_t done=0;
      while(done<output.region.bytes){
        if(result.outputs.size()>=options.max_outputs)fail("output_count_limit_exceeded","scalar output budget exceeded");
        auto bytes=std::min<std::uint64_t>(element_bytes,output.region.bytes-done);
        if(bytes>8)fail("scalar_fold_wide_scalar","output scalar wider than 64 bits requires a numeric type");
        auto target=bytes==element_bytes?element:bit(unsigned(8*bytes));
        auto projected=project(output.value,Offset(integer(output.value_byte_offset+done)),target);
        auto region=output.region;region.offset+=done;region.bytes=bytes;
        pushOutput(region,projected);done+=bytes;
        if(options.collection_interval && result.outputs.size()%options.collection_interval==0)
          collectCompletedOutputs();
      }
    }
    // Lifting: every body for its unit, then the output tail the unlifted lowering runs.
    stop.phase='f';
    collectBeforeFinalLifting();
    instantiatePending();
    if(output_domains.empty())coalesceFragments(result.graph,result.outputs);
    else canonicalizeOutputs(familyOutputs(),s::CanonicalForms::Bf16Fields); // as at every consolidation
    std::vector<s::Ref> roots;for(const auto& out:result.outputs)roots.push_back(out.value);
    std::map<std::size_t,std::vector<s::Ref>> validation_roots;
    for(std::size_t i=0;i<roots.size();++i) {
      const auto domain=output_domains.find(i);
      const auto rank=domain==output_domains.end()?0:domain->second.shape.size();
      validation_roots[rank].push_back(roots[i]);
    }
    for(const auto& [rank,values]:validation_roots)result.graph.validateRoots(values,rank);
    result.nodes_before_final_collection=result.graph.size();
    result.nodes_reclaimed+=result.graph.collect(roots);
    for(std::size_t i=0;i<roots.size();++i)result.outputs[i].value=roots[i];
    if(!output_domains.empty()) {
      std::vector<ScalarFoldOutput> scalars;
      for(std::size_t i=0;i<result.outputs.size();++i) {
        const auto& out=result.outputs[i];
        if(auto domain=output_domains.find(i);domain!=output_domains.end()) {
          result.output_families.push_back(domain->second.family(out));
        } else scalars.push_back(out);
      }
      result.outputs=std::move(scalars);
      s::Artifact artifact;
      artifact.graph=std::move(result.graph);
      artifact.output_families=std::move(result.output_families);
      for(const auto& output:result.outputs)
        artifact.outputs.push_back({output.region.allocation,output.region.offset,output.region.bytes,output.value});
      const auto before=artifact.graph.size();
      // Outside class mode, rows of one output tensor become one family over an outer row binder
      // (consolidateOutputs, strided rows), as class mode's member families already are: one
      // body, every row checked against it. Class mode keeps its own member-family merge.
      result.output_families_merged+=s::consolidateOutputs(artifact,false,{},options.unit_lifting==nullptr);
      if(before>artifact.graph.size())result.nodes_reclaimed+=before-artifact.graph.size();
      result.graph=std::move(artifact.graph);
      result.output_families=std::move(artifact.output_families);
      for(std::size_t i=0;i<result.outputs.size();++i)result.outputs[i].value=artifact.outputs[i].value;
    }
    refuseUnwrittenOutputs();
    result.complete=true;
  }catch(const Failure& e){
    result.diagnostic=ScalarFoldDiagnostic{e.code,e.what(),current};stop.literal_message=e.literal;
    if(e.code=="sparse_metadata_invalid")result.diagnostic->undefined_behavior=std::move(sparse_metadata_witness);
  }
   catch(const IndexedSourceLimit& e){result.diagnostic=ScalarFoldDiagnostic{e.code,e.what(),current};}
   catch(const frontend::ptx::LayoutError& e){result.diagnostic=ScalarFoldDiagnostic{"scalar_fold_mma_descriptor",e.what(),current};}
  if(result.diagnostic)stop.completed_outputs=result.outputs.size();
  if(options.unit_lifting) {
    const auto sites=UnitSiteCounts();
    const auto delta=[&](UnitSite site){return sites[std::size_t(site)]-sites_at_start[std::size_t(site)];};
    result.lifting.declined=delta(UnitSite::FoldKnown)+delta(UnitSite::FoldSpecialization)+
        delta(UnitSite::FoldTemplateLeaf);
    // A lowering that never met a BlockIndex value ran exactly the unlifted
    // lowering's course, so its failure is the unlifted lowering's. Otherwise
    // the failure may be the lifting's own (an unknown coordinate): refuse,
    // and the driver restarts per unit.
    if(result.diagnostic && (result.diagnostic->code=="scalar_fold_unit_parameter_unsupported" ||
                             result.lifting.parameters || result.lifting.declined))
      result.lifting.refused=true;
    // Class mode: the bodies also stand for members that were never
    // analysed, so any failure may be one the members' own lowering would
    // report differently (or not at all), and a BlockIndex value read
    // numerically by any instance in the fold baked the representative's
    // coordinate into a body: refuse.
    if(!result.lifting.refused) {
      const auto numeric=delta(UnitSite::ValueSpecialization)+delta(UnitSite::ValueTemplateLeaf)+
          delta(UnitSite::FoldBindingKnown);
      if(result.diagnostic || numeric)result.lifting.refused=true;
    }
  }
  if(sentryward::debug::environment("SENTRYWARD_SCALAR_NODE_HISTOGRAM")) {
#if defined(SENTRYWARD_DEVELOPMENT_DIAGNOSTICS) || defined(SENTRYWARD_ENABLE_COMPRESSION_AUDIT)
#include "sentryward/development/blocks/scalar_fold_lowering_trace_4.inc"
#endif
}
  SENTRYWARD_LEDGER(ledger->snapshot(result.graph,keys.size(),result.outputs.size()));
  return std::move(result);
}

} // namespace
} // namespace sentryward::value_analysis
