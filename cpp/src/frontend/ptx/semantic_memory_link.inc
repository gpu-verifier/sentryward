// Included in semantic_program.cpp's private namespace. All joins below are
// over stored source columns, instance patterns and exact progression pieces.
// Explicit actor/address/member metadata is indexed once, not once per access.
// Dispatch through already-explicit metadata, then consume whole AP pieces.
// No per-query scan of other strides or overlapping bounding intervals.
class MemoryAddressRunIndex {
  struct Entry {a::AffineDomain domain;std::size_t payload;};
  std::vector<Entry> entries_;
  std::unordered_map<std::uint64_t,std::size_t> owners_;
 public:
  void reserve(std::size_t size) {owners_.reserve(size);}
  void bind(std::uint64_t id,std::size_t run) {owners_.emplace(id,run);}
  void add(a::AffineDomain domain,std::size_t payload) {
    entries_.push_back({domain,payload});
  }
  template<class Visit> void intersect(a::AffineDomain query,Visit visit) const {
    if(!query.count || entries_.empty())return;
    if(!query.step)query.count=1;
    std::vector<a::AffineDomain> pending{query};
    while(!pending.empty()) {
      const auto part=pending.back();pending.pop_back();
      const auto owner=owners_.find(part.first);
      if(owner==owners_.end())throw LayoutError("Source domain has missing address metadata");
      const auto& entry=entries_[owner->second];
      auto match=a::IntersectAffine(part,entry.domain);
      if(!match.count)throw LayoutError("Address metadata run does not contain its member");
      const a::AffineDomain covered{part.first+part.step*match.left,part.step*match.left_step,match.count};
      if(query.step) {
        match.left=(covered.first-query.first)/query.step;
        match.left_step=covered.step/query.step;
      }
      visit(entry.payload,match);
      auto remainder=a::SubtractAffine(part,covered);
      pending.insert(pending.end(),remainder.begin(),remainder.end());
    }
  }
};
bool LinkMemoryDomains(SemanticProgram& result) {
  a::ProfileScope profile("frontend.semantic_program.memory_link.domains");
  using Domain=a::AffineDomain;
  using Key=std::tuple<std::uint32_t,std::uint32_t,std::uint64_t>;
  const auto require=[](bool yes,const char* why) {if(!yes)throw LayoutError(why);};
  const auto advance=[&](std::uint64_t base,std::int64_t step,std::uint64_t i) {
    const auto value=__int128(base)+__int128(step)*i;
    require(value>=0 && value<=UINT64_MAX,"Binding coordinate overflow");
    return static_cast<std::uint64_t>(value);
  };
  struct ActorRun {const a::ActorInfo* first;std::size_t count=1;std::int64_t lane_step=0;};
  struct AddressRun {const a::ConstantAddressBinding* first;std::size_t count=1;std::int64_t step=0,id_step=0;const a::ConstantAddressBinding* last=nullptr;};
  struct SourceColumn {
    SourceAccess first;Domain threads;std::int64_t instruction_step,address_step,offset_step;
    const a::ConstantAddressBinding* address;
    std::uint64_t address_offset,order;std::int64_t order_step;
  };
  struct ValueColumn {ValueAccess first;Domain threads;std::uint64_t instance_step,lowering;};
  struct Group {std::vector<SourceColumn> sources;std::vector<ValueColumn> values;};
  std::map<Key,Group> groups;
  std::vector<ActorRun> actors;
  a::AffineIndex actor_index;
  struct AddressIndex {std::vector<AddressRun> runs;MemoryAddressRunIndex index;};
  AddressIndex indexed_addresses;
  const auto& artifact=result.lowering.artifact;
  try {
    for(const auto& actor:artifact.program.actors) {
      if(!actors.empty()) {
        auto& run=actors.back();const auto& first=*run.first;
        if(actor.id.value()==std::uint64_t(first.id.value())+run.count && actor.cta==first.cta && actor.warp==first.warp) {
          const auto step=std::int64_t(actor.lane)-first.lane;
          if(run.count==1)run.lane_step=step;
          if(step==run.lane_step*std::int64_t(run.count)){++run.count;continue;}
        }
      }
      actors.push_back({&actor});
    }
    for(std::size_t i=0;i<actors.size();++i)actor_index.add({actors[i].first->id.value(),1,actors[i].count},i);
    actor_index.seal();
    // Build one index of the explicit metadata, independent of query stride.
    // Interleaved allocations have separate tails. Each address is checked
    // against both the ID progression and byte progression before extending
    // a run; no endpoint sampling or inferred expression equality is used.
    // Storage is O(A), not O(A * number_of_query_strides).
    {
      a::ProfileScope index_profile("frontend.semantic_program.memory_link.address_index");
      index_profile.nodes=artifact.addresses.size();
      auto& addresses=indexed_addresses.runs;
      indexed_addresses.index.reserve(artifact.addresses.size());
      using AddressKey=std::tuple<std::string_view,std::string_view,std::string_view,std::string_view>;
      const auto addressKey=[](const auto& address)->AddressKey {
        return {address.address_space,address.allocation,address.alias_scope,address.expression};
      };
      // Sort pointers, not bindings/expressions. Ordering by byte coordinate
      // separates interleaved accesses to distinct regions of one allocation.
      // This is O(A log A) comparisons and O(A) auxiliary storage.
      std::vector<const a::ConstantAddressBinding*> ordered;
      ordered.reserve(artifact.addresses.size());
      for(const auto& address:artifact.addresses)ordered.push_back(&address);
      std::sort(ordered.begin(),ordered.end(),[&](const auto* x,const auto* y) {
        const auto xk=addressKey(*x),yk=addressKey(*y);
        if(xk!=yk)return xk<yk;
        return std::tie(x->offset,x->address)<std::tie(y->offset,y->address);
      });
      for(const auto* entry:ordered) {
        const auto& address=*entry;
        if(!addresses.empty() && addressKey(*addresses.back().first)==addressKey(address)) {
          auto& run=addresses.back();const auto& first=*run.first;
          const auto distance=__int128(address.offset)-first.offset;
          if(address.address!=first.address && distance>=INT64_MIN && distance<=INT64_MAX) {
            if(run.count==1) {
              run.id_step=std::int64_t(address.address.value())-first.address.value();
              run.step=static_cast<std::int64_t>(distance);
            }
            if(__int128(address.address.value())==__int128(first.address.value())+__int128(run.id_step)*run.count &&
                distance==__int128(run.step)*run.count){
              indexed_addresses.index.bind(address.address.value(),addresses.size()-1);
              ++run.count;run.last=&address;continue;
            }
          }
        }
        indexed_addresses.index.bind(address.address.value(),addresses.size());
        addresses.push_back({&address});
      }
      for(std::size_t i=0;i<addresses.size();++i) {
        auto& run=addresses[i];
        if(run.id_step<0){run.first=run.last;run.id_step=-run.id_step;run.step=-run.step;}
        indexed_addresses.index.add({run.first->address.value(),std::uint64_t(run.id_step),run.count},i);
      }
      index_profile.edges=addresses.size();
    }
    // Return indices in original query order, including negative/broadcast
    // progressions. No evaluation or inference from endpoint samples occurs.
    const auto intersect=[&](std::uint64_t first,std::int64_t step,std::size_t count,
        const auto& index,auto consume) {
      const bool reverse=step<0;
      const auto magnitude=reverse?std::uint64_t(-(step+1))+1:std::uint64_t(step);
      std::size_t covered=0;
      index.intersect({reverse?advance(first,step,count-1):first,magnitude,count},[&](auto id,auto m) {
        if(!step){m.count=count;m.left_step=1;m.right_step=0;}
        const auto left=reverse?count-1-(m.left+m.left_step*(m.count-1)):m.left;
        const auto right=reverse?m.right+m.right_step*(m.count-1):m.right;
        consume(id,left,m.left_step,right,reverse?-std::int64_t(m.right_step):std::int64_t(m.right_step),m.count);
        covered+=m.count;
      });
      require(covered==count,"Source domain has missing actor/address metadata");
    };
    artifact.program.instructions.forEachAffineColumn([&](const auto& first,const auto& delta,
        auto order,auto order_step,std::size_t count) {
      const auto accesses=Accesses(first);if(accesses.empty())return;
      ++profile.nodes;profile.edges+=count;
      require(first.ptx_origin.has_value(),"Memory source lacks PTX occurrence");
      // Temporal occurrences and multiple components of one actor remain
      // distinct. Only a column with changing actors is a spatial domain.
      const auto temporal=(delta[3] || !delta[1])?count:1;
      for(std::size_t time=0;time<temporal;++time) {
        auto base=first;a::SourceRecordCodec::Binding fields;
        a::SourceRecordCodec::split(base,fields);
        auto translated=a::SourceRecordCodec::translate(fields,delta,time);
        require(bool(translated),"Source column translation overflow");a::SourceRecordCodec::bind(base,*translated);
        const auto extent=temporal>1?1:count;
        auto next=base;
        if(extent>1) {
          auto next_fields=a::SourceRecordCodec::translate(fields,delta,1);
          require(bool(next_fields),"Source column translation overflow");a::SourceRecordCodec::bind(next,*next_fields);
        }
        const auto endpoints=Accesses(base),next_endpoints=Accesses(next);
        require(endpoints.size()==next_endpoints.size(),"Source column changes access arity");
        intersect(base.actor.value(),extent>1?delta[1]:0,extent,actor_index,
          [&](auto actor_id,auto actor_begin,auto actor_stride,auto row,auto row_stride,auto actor_count) {
          const auto& actor=actors[actor_id];const auto& info=*actor.first;
          const auto thread=std::uint64_t(info.warp.value())*32+advance(info.lane,actor.lane_step,row);
          const auto thread_step=actor.lane_step*row_stride;
          require(actor_count==1 || thread_step>0,"Source members are not distinct increasing threads");
          for(std::size_t e=0;e<endpoints.size();++e) {
            const auto& endpoint=endpoints[e];require(endpoint.bytes.has_value(),"Unknown source payload size");
            const auto address_step=std::int64_t(next_endpoints[e].address.value())-endpoint.address.value();
            const auto address_first=advance(endpoint.address.value(),address_step,actor_begin);
            const auto& indexed=indexed_addresses;
            intersect(address_first,address_step*std::int64_t(actor_stride),actor_count,indexed.index,
              [&](auto address_id,auto begin,auto stride,auto address_row,auto address_stride,auto n) {
              const auto& address=indexed.runs[address_id];
              auto access=endpoint;
              access.key.instruction=a::InstId(advance(base.id.value(),delta[0],actor_begin+actor_stride*begin));
              access.address=a::AddressId(advance(address_first,address_step*std::int64_t(actor_stride),begin));
              const Key key{info.cta.value(),base.ptx_origin->instruction_index,base.ptx_origin->occurrence};
              groups[key].sources.push_back({access,
                  {advance(thread,thread_step,begin),std::uint64_t(thread_step)*stride,n},
                  delta[0]*std::int64_t(actor_stride*stride),address_step*std::int64_t(actor_stride*stride),
                  address.step*address_stride,address.first,advance(address.first->offset,address.step,address_row),
                  order+order_step*(time+actor_begin+actor_stride*begin),std::int64_t(order_step*actor_stride*stride)});
            });
          }
        });
      }
    });
    struct Members {std::size_t begin;Domain threads;};
    std::unordered_map<const void*,std::vector<Members>> member_cache;
    for(const auto& node:result.values.nodes()) {
      if(!dynamic_cast<const v::LoadOp*>(&node.operation()) && !dynamic_cast<const v::StoreOp*>(&node.operation()))continue;
      const auto& members=node.instanceThreads();
      auto [entry,inserted]=member_cache.try_emplace(members.storageIdentity());
      if(inserted && !members.empty())for(std::size_t i=0;i<members.size();) {
        auto end=i+1;std::uint64_t step=0;
        if(end<members.size() && members[end]>members[i]) {
          step=members[end]-members[i];++end;
          while(end<members.size() && members[end]==members[i]+step*(end-i))++end;
        }
        entry->second.push_back({i,{members[i],step,end-i}});i=end;
      }
      node.forEachInstancePattern([&](const v::Instance& pattern,std::size_t begin,std::size_t count,bool) {
        const auto occurrence=Occurrence(pattern.origin);
        require(bool(occurrence) && pattern.origin.lowering_index.has_value(),"Value lacks occurrence/component identity");
        const Key key{occurrence->cta,occurrence->instruction,occurrence->occurrence};
        const auto access=Access(result.values,node,begin);require(bool(access),"Missing memory value");
        if(members.empty()) {
          require(count==1,"Non-template Value has multiple unbound members");
          groups[key].values.push_back({*access,{std::uint64_t(occurrence->warp)*32+occurrence->lane,0,1},0,*pattern.origin.lowering_index});
        } else for(const auto& run:entry->second) {
          const auto m=a::IntersectAffine({begin,1,count},{run.begin,1,run.threads.count});if(!m.count)continue;
          auto value=*access;value.reference.instance=begin+m.left;
          groups[key].values.push_back({value,{run.threads.first+run.threads.step*m.right,
              run.threads.step*m.right_step,m.count},m.left_step,*pattern.origin.lowering_index});
        }
      });
    }
    std::map<a::InstId,const MmaValueLayout*> matrix_layouts;
    for(const auto& layout:result.lowering.mma_layouts)matrix_layouts.emplace(layout.instruction,&layout);
    std::map<a::AddressId,const a::ConstantAddressBinding*> address_lookup;
    for(const auto& address:artifact.addresses)address_lookup.emplace(address.address,&address);
    for(auto& [key,group]:groups) {
      std::vector<Domain> cuts;
      std::uint64_t lo=UINT64_MAX,hi=0;
      const auto add=[&](Domain d){if(!d.count)return;cuts.push_back(d);lo=std::min(lo,d.first);hi=std::max(hi,d.last());};
      for(const auto& s:group.sources)add(s.threads);
      for(const auto& v:group.values)add(v.threads);
      std::vector<Domain> pieces{{lo,1,hi-lo+1}};
      // Membership refinement is in actual input domain boundaries. It never
      // compares expressions or manufactures per-member source lists.
      for(auto cut:cuts) {
        std::vector<Domain> next;
        for(auto d:pieces) {
          const auto m=a::IntersectAffine(d,cut);
          if(!m.count || m.count==d.count){next.push_back(d);continue;}
          Domain inside{d.first+d.step*m.left,d.step*m.left_step,m.count};
          next.push_back(inside);auto outside=a::SubtractAffine(d,inside);
          next.insert(next.end(),outside.begin(),outside.end());
        }
        pieces=std::move(next);
      }
      for(auto domain:pieces) {
        struct S {SourceAccess access;const SourceColumn* column;std::uint64_t begin,step,order;};
        struct V {ValueAccess access;const ValueColumn* column;std::uint64_t begin,step;};
        std::vector<S> sources;std::vector<V> values;
        for(const auto& s:group.sources) {
          const auto m=a::IntersectAffine(domain,s.threads);if(!m.count)continue;
          require(m.count==domain.count,"Source domain refinement gap");
          auto access=s.first;access.key.instruction=a::InstId(advance(access.key.instruction.value(),s.instruction_step,m.right));
          access.address=a::AddressId(advance(access.address.value(),s.address_step,m.right));
          sources.push_back({access,&s,m.right,m.right_step,advance(s.order,s.order_step,m.right)});
        }
        for(const auto& v:group.values) {
          const auto m=a::IntersectAffine(domain,v.threads);if(!m.count)continue;
          require(m.count==domain.count,"Value domain refinement gap");
          auto access=v.first;access.reference.instance+=v.instance_step*m.right;
          values.push_back({access,&v,m.right,m.right_step});
        }
        if(sources.empty() && values.empty())continue;
        require(!sources.empty() && !values.empty(),"Source and Value domains have different coverage");
        std::stable_sort(sources.begin(),sources.end(),[](auto& x,auto& y){return x.order<y.order;});
        for(std::size_t i=1;i<sources.size();++i)require(
            __int128(sources[i-1].order)+__int128(sources[i-1].column->order_step)*sources[i-1].step*(domain.count-1)<=
            __int128(sources[i].order)+__int128(sources[i].column->order_step)*sources[i].step*(domain.count-1),
            "Source component order changes within domain");
        std::stable_sort(values.begin(),values.end(),[](auto& x,auto& y){return x.column->lowering<y.column->lowering;});
        const auto [cta,instruction,occ]=key;
        const SemanticOccurrence occurrence{cta,std::uint32_t(domain.first/32),std::uint32_t(domain.first%32),instruction,occ};
        // Actor domains were cut at warp boundaries, so lane is affine here.
        require(domain.first/32==domain.last()/32,"Binding domain crosses a warp boundary");
        if(values.size()==1 && values.front().access.ordered_payload) {
          const auto& value=values.front();std::uint64_t payload=0;
          for(const auto& source:sources) {
            require(source.access.ordered_native && source.access.mode==value.access.mode &&
                source.access.bytes && value.access.bytes && payload<=*value.access.bytes &&
                *source.access.bytes<=*value.access.bytes-payload,"Register payload does not match source fragments");
            auto address=*source.column->address;address.address=source.access.address;
            address.offset=advance(source.column->address_offset,source.column->offset_step,source.begin);
            auto part=value.access;part.bytes=source.access.bytes;
            auto binding=BindMemoryValue(source.access,part,occurrence,0,&address);
            binding.value_byte_offset=payload;payload+=*source.access.bytes;
            SemanticMemoryBindingCodec::Delta delta{};
            delta[0]=source.column->instruction_step*std::int64_t(source.step);
            delta[3]=domain.step;delta[7]=value.column->instance_step*value.step;
            delta[8]=source.column->address_step*std::int64_t(source.step);
            delta[9]=source.column->offset_step*std::int64_t(source.step);
            if(domain.count==1)result.memory_bindings.push_back(std::move(binding));
            else result.memory_bindings.appendAffineColumns({binding},{delta},domain.count);
          }
          require(value.access.bytes==payload,"Incomplete register payload mapping");
          continue;
        }
        bool ordered=sources.size()==values.size();
        for(std::size_t i=0;ordered && i<sources.size();++i)
          ordered=sources[i].access.ordered_native && values[i].access.ordinary_scalar &&
              sources[i].access.mode==values[i].access.mode && sources[i].access.bytes==values[i].access.bytes;
        std::map<a::AccessMode,std::size_t> endpoints;std::map<a::AccessMode,std::uint64_t> totals;
        bool contiguous=true;
        for(std::size_t i=0;i<sources.size();++i)if(!endpoints.emplace(sources[i].access.mode,i).second)contiguous=false;
        for(const auto& value:values) {
          if(!value.access.ordinary_scalar || !value.access.bytes || !endpoints.contains(value.access.mode))contiguous=false;
          else totals[value.access.mode]+=*value.access.bytes;
        }
        for(auto [mode,i]:endpoints)if(totals[mode]!=sources[i].access.bytes)contiguous=false;
        if(!ordered && !contiguous) {
          a::ProfileScope native("frontend.semantic_program.memory_link.native_payload");
          native.nodes=sources.size();native.edges=values.size();
          const auto stored_before=result.memory_bindings.storedBindingCount();
          require(domain.count==1,"Native matrix/copy issuer domain requires range lifting");
          std::vector<SourceAccess> ss;std::vector<ValueAccess> vv;
          for(const auto& s:sources)ss.push_back(s.access);for(const auto& v:values)vv.push_back(v.access);
          if(LinkCopyMemory(result,occurrence,ss,vv,address_lookup))continue;
          const auto layout=matrix_layouts.find(ss.front().key.instruction);
          require(layout!=matrix_layouts.end(),"No exact native matrix payload mapping");
          LinkMmaMemory(result,*layout->second,occurrence,ss,vv,address_lookup);
          native.pruned=result.memory_bindings.storedBindingCount()-stored_before;continue;
        }
        std::map<a::AccessMode,std::uint64_t> offsets;
        for(std::size_t i=0;i<values.size();++i) {
          const auto& value=values[i];const auto& source=sources[ordered?i:endpoints.at(value.access.mode)];
          const auto offset=ordered?0:offsets[value.access.mode];offsets[value.access.mode]+=*value.access.bytes;
          auto address=*source.column->address;address.address=source.access.address;
          address.offset=advance(source.column->address_offset,source.column->offset_step,source.begin);
          auto binding=BindMemoryValue(source.access,value.access,occurrence,offset,&address);
          SemanticMemoryBindingCodec::Delta delta{};
          delta[0]=source.column->instruction_step*std::int64_t(source.step);
          delta[3]=domain.step;delta[7]=value.column->instance_step*value.step;
          delta[8]=source.column->address_step*std::int64_t(source.step);
          delta[9]=source.column->offset_step*std::int64_t(source.step);
          if(domain.count==1)result.memory_bindings.push_back(std::move(binding));
          else result.memory_bindings.appendAffineColumns({binding},{delta},domain.count);
        }
      }
    }
    result.memory_linking_templates_used=true;
    profile.pruned=result.memory_bindings.storedBindingCount();
  } catch(const std::exception& error) {
    Diagnostic(result,"unsupported_memory_binding_domain",error.what());
  }
  return true; // Never silently retry the whole kernel through expanded linking.
}
