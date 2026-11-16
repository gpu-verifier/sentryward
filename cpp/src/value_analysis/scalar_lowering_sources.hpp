#pragma once
#include "scalar_fold_lowering_impl.hpp"

namespace sentryward::value_analysis {
namespace {

inline auto Lower::initialByteMap(std::size_t node,std::size_t first,std::size_t last) -> const InitialByteMap* {
  auto [entry,inserted]=initial_byte_maps.try_emplace(node);
  if(inserted) {
    // Initial-symbol IDs describe retained read records, not different
    // mathematical arrays. Coalesce their certified byte maps without
    // changing IDs/provenance in MemoryEliminationResult. Every stored run
    // is checked, never just representative logical members.
    for(auto it=memory.initial_read_families.lower_bound({node,0});
        it!=memory.initial_read_families.end() && it->first.node==node;++it) {
      const auto& f=it->second;
      if(!entry->second.empty()) {
        auto& previous=entry->second.back();
        const auto delta=__int128(f.initial.region.offset)-previous.initial.offset;
        const auto stride=previous.count==1?delta:__int128(previous.stride);
        if(previous.first<=SIZE_MAX-previous.count && previous.first+previous.count==f.first.instance &&
            previous.initial.allocation==f.initial.region.allocation && previous.initial.bytes==f.initial.region.bytes &&
            stride>=INT64_MIN && stride<=INT64_MAX &&
            delta==stride*previous.count && (f.count==1 || f.byte_stride==stride) &&
            f.count<=SIZE_MAX-previous.count) {
          previous.stride=static_cast<std::int64_t>(stride);previous.count+=f.count;continue;
        }
      }
      entry->second.push_back({f.first.instance,f.count,f.initial.region,f.byte_stride});
    }
    // Compile exact first-difference runs. A periodic carry can straddle
    // the arbitrary boundaries of stored affine records (e.g. an elected
    // lane removed every warp). Validate every run and junction; do not
    // enumerate members or require the storage segmentation to align.
    std::vector<InitialByteMap> periodic;
    const auto& runs=entry->second;
    for(std::size_t begin=0;begin<runs.size();) {
      auto end=begin+1,anchor=begin;
      while(end<runs.size() && runs[end-1].first<=SIZE_MAX-runs[end-1].count &&
          runs[end-1].first+runs[end-1].count==runs[end].first &&
          runs[end].initial.allocation==runs[begin].initial.allocation &&
          runs[end].initial.bytes==runs[begin].initial.bytes) {
        if(runs[end].count>runs[anchor].count)anchor=end;
        ++end;
      }
      const auto baseline=runs[anchor].stride;
      std::size_t period=0,anomalies=0;
      __int128 carry=0;
      const auto edge=[&](std::size_t at,std::size_t count,__int128 slope) {
        if(!count || slope==baseline)return true;
        if(count!=1)return false;
        const auto jump=at-runs[begin].first+1;
        if(!period){period=jump;carry=slope-baseline;}
        ++anomalies;
        return __int128(jump)==__int128(period)*anomalies && slope-baseline==carry;
      };
      bool valid=true;
      for(std::size_t i=begin;valid && i<end;++i) {
        const auto& r=runs[i];
        valid=edge(r.first,r.count-1,r.stride);
        if(valid && i+1<end)
          valid=edge(r.first+r.count-1,1,__int128(runs[i+1].initial.offset)-
              (__int128(r.initial.offset)+__int128(r.stride)*(r.count-1)));
      }
      const auto count=runs[end-1].first-runs[begin].first+runs[end-1].count;
      const auto outer=__int128(baseline)*period+carry;
      if(valid && period && anomalies==(count-1)/period && outer>=INT64_MIN && outer<=INT64_MAX) {
        auto map=runs[begin];map.count=count;map.stride=baseline;
        map.period=period;map.outer_stride=static_cast<std::int64_t>(outer);
        periodic.push_back(std::move(map));
      } else for(auto i=begin;i<end;++i)periodic.push_back(runs[i]);
      begin=end;
    }
    entry->second=std::move(periodic);
  }
  const auto& maps=entry->second;
  auto it=std::upper_bound(maps.begin(),maps.end(),first,[](auto i,const auto& map){return i<map.first;});
  if(it==maps.begin())return nullptr;
  --it;
  return first<=last && last-it->first<it->count?&*it:nullptr;
}

inline auto Lower::periodicReadMap(std::size_t node,std::size_t first,
    std::size_t last,std::uint64_t offset,std::uint64_t width) -> const PeriodicReadMap* {
  auto bucket=memory.read_slice_patterns.upper_bound({node,offset});
  if(bucket==memory.read_slice_patterns.begin())return nullptr;
  --bucket;if(bucket->first.first!=node)return nullptr;
  auto [cached,inserted]=periodic_read_maps.try_emplace(bucket->first);
  if(inserted) {
    // Compile the already stored affine segments once. This compares only
    // source identity and numeric maps, never operands or logical members.
    const auto& patterns=bucket->second;
    for(auto begin=patterns.begin();begin!=patterns.end();) {
      const auto& p=begin->second;auto next=std::next(begin);
      std::int64_t byte_outer=0,member_outer=0;std::size_t blocks=1;
      while(next!=patterns.end()) {
        const auto& q=next->second;
        if(!p.count || q.count!=p.count || !p.sameSourceTemplate(q.source) ||
            q.bytes!=p.bytes || q.result_offset!=p.result_offset ||
            q.byte_stride!=p.byte_stride || q.source_instance_stride!=p.source_instance_stride ||
            static_cast<unsigned __int128>(p.first.instance)+
                static_cast<unsigned __int128>(blocks)*p.count!=q.first.instance)break;
        const auto bytes=static_cast<__int128>(q.source_offset)-p.source_offset;
        const auto members=static_cast<__int128>(q.source.instanceIndex())-p.source.instanceIndex();
        if(blocks==1) {
          if(bytes<INT64_MIN || bytes>INT64_MAX || members<INT64_MIN || members>INT64_MAX)break;
          byte_outer=static_cast<std::int64_t>(bytes);member_outer=static_cast<std::int64_t>(members);
        } else if(bytes!=static_cast<__int128>(blocks)*byte_outer ||
                  members!=static_cast<__int128>(blocks)*member_outer)break;
        ++blocks;++next;
      }
      if(blocks>1 && p.count<=SIZE_MAX/blocks)
        cached->second.push_back({&p,p.count*blocks,p.count,byte_outer,member_outer});
      begin=next;
    }
  }
  const auto& maps=cached->second;
  auto found=std::upper_bound(maps.begin(),maps.end(),first,
      [](auto member,const auto& map){return member<map.first->first.instance;});
  if(found==maps.begin())return nullptr;
  const auto& map=*std::prev(found);const auto& p=*map.first;
  if(last<first || first<p.first.instance || last-p.first.instance>=map.count ||
      offset<p.result_offset || offset-p.result_offset>p.bytes ||
      width>p.bytes-(offset-p.result_offset))return nullptr;
  return &map;
}

inline auto Lower::memberUse(v::Value value,
    std::int64_t base,std::int64_t stride) -> std::shared_ptr<const IndexedSourceUse> {
  value=input.canonicalInterpretation(value);
  auto key=std::tuple{valueKey(value),base,stride};
  if(auto found=member_uses.find(key);found!=member_uses.end())return found->second;
  auto binding=std::make_shared<IndexedValueBinding>(IndexedValueBinding{value,base,stride,{}});
  auto use=std::make_shared<IndexedSourceUse>(IndexedSourceUse{value,
      IndexedSourceUse::Kind::MemberTemplate,std::move(binding),{}});
  member_uses.emplace(std::move(key),use);return use;
}

// A refinement may turn an existing binder into the singleton [0,1).
// Its value is then exactly zero. Retaining its old coefficient makes
// equivalent queries have different cache keys, and repeated coordinate
// composition can overflow coefficients that contribute nothing at all.
// Inspect only affected index sub-DAGs, never source/math ancestors.
inline auto Lower::specializeSingletons(Offset x) const -> Offset {
  std::uint64_t fixed=0;
  for(unsigned i=0;i<domains.size() && i<64;++i)
    if(domains[i]==1)fixed|=std::uint64_t{1}<<i;
  if(!(x.binderMask()&fixed))return x;
  std::unordered_map<const OffsetExpression*,Offset> memo;
  const auto visit=[&](auto&& self,Offset at)->Offset {
    if(!(at.binderMask()&fixed))return at;
    if(!at.expression) {
      for(auto it=at.terms.begin();it!=at.terms.end();)
        if(it->first<domains.size() && domains[it->first]==1)it=at.terms.erase(it);
        else ++it;
      return at;
    }
    const auto* e=at.expression.get();
    if(auto found=memo.find(e);found!=memo.end())return found->second;
    auto left=self(self,e->left);
    Offset out;
    switch(e->kind) {
      case OffsetExpression::Add:out=plus(std::move(left),self(self,e->right));break;
      case OffsetExpression::Scale:out=times(std::move(left),e->factor);break;
      case OffsetExpression::Quotient:out=quotient(left,e->factor);break;
      case OffsetExpression::Remainder:out=remainder(left,e->factor);break;
    }
    memo.emplace(e,out);return out;
  };
  return visit(visit,std::move(x));
}

inline auto Lower::indexedBinding(v::Value value) -> const IndexedValueBinding* {
  if(!indexed_context)return nullptr;
  const auto& use=indexed_context->use;
  if(use && use->kind==IndexedSourceUse::Kind::Binding &&
     use->prototype==input.canonicalInterpretation(value) && use->binding)return &*use->binding;
  return nullptr;
}

inline auto Lower::refineFold(unsigned axis,const std::vector<std::uint64_t>& cuts) -> void {
  SENTRYWARD_LEDGER(ledger->add("fold_refinements");ledger->add("fold_refinement_cuts",cuts.size()));
  auto owner=fold_owner;while(owner && owner->axis!=axis)owner=owner->parent;
  if(!owner)fail("compact_view_unsupported","read partition requires an unowned reduction-domain split");
  auto& pieces=fold_pieces.at(owner->key);
  if(owner->segment>=pieces.size() || pieces[owner->segment]!=std::pair{owner->first,owner->count})
    fail("scalar_fold_internal_binding","stale reduction-domain refinement");
  std::vector<std::pair<std::uint64_t,std::uint64_t>> replacement;
  std::uint64_t start=0;
  for(auto cut:cuts)if(cut>start && cut<owner->count) {
    replacement.emplace_back(owner->first+start,cut-start);start=cut;
  }
  if(!start)fail("compact_view_unsupported","read partition supplied no reduction boundary");
  replacement.emplace_back(owner->first+start,owner->count-start);
  pieces.erase(pieces.begin()+owner->segment);
  pieces.insert(pieces.begin()+owner->segment,replacement.begin(),replacement.end());
  RetryFoldDomain retry;
  refinement_resume=*owner->resume;
  for(auto p=owner;p;p=p->parent)retry.keep.push_back(p->key.first);
  throw retry;
}

inline auto Lower::commitReduction(ReductionProgress& progress,std::uint64_t next,s::Ref accumulator) -> void {
  if(next<=progress.next)throw std::logic_error("reduction continuation did not advance");
  keys.at(progress.accumulator_key)=accumulator;
  progress.next=next;
  progress.sub_next=0;
  // The accumulator now contains the semantic dependencies. Operand cache
  // entries replaced by a Fold are no longer strong roots of this task.
  progress.pending.clear();
  SENTRYWARD_AUDIT(audit->add(AuditCounter::reduction_domains_committed));
}

inline auto Lower::finishReduction(const ProjectionKey& key) -> void {
  if(auto found=reduction_progress.find(key);found!=reduction_progress.end()) {
    keys.erase(found->second.accumulator_key);reduction_progress.erase(found);
  }
  auto piece=fold_pieces.lower_bound({key,0});
  while(piece!=fold_pieces.end() && piece->first.first==key)piece=fold_pieces.erase(piece);
}

inline auto Lower::registerSpatialFamily(SpatialValues family) -> void {
  const auto key=valueKey(family.base);
  if(auto found=spatial_values.find(key);found!=spatial_values.end()) {
    if(!found->second.equals(family))spatialReject();
    return;
  }
  if(family.affine())spatial_instance_strides.insert_or_assign(key,family.stride);
  spatial_values.emplace(key,std::move(family));
}

inline auto Lower::registerSpatial(std::vector<v::Value> values) -> void {
  SENTRYWARD_AUDIT(audit->add(AuditCounter::spatial_members_enumerated,values.size());
    if(values.size()>1)audit->event("INSTANCE_ENUMERATION","rediscover_spatial_template_from_members",
        values.size(),1,values.front().nodeId(),true));
  if(values.size()!=spatial_count)spatialReject();
  for(auto& x:values)x=input.resolve(x);
  const auto& first=values.front();
  for(const auto& x:values) {
    if(x.nodeId()!=first.nodeId() || x.resultIndex()!=first.resultIndex() ||
        input.type(x)!=input.type(first) || !input.type(x).isScalar() ||
        !x.indices().empty() || x.threadSelector() || x.pointer() ||
        bool(x.conversion())!=bool(first.conversion()) ||
        (x.conversion() && v::typeConversionLabel(x)!=v::typeConversionLabel(first)))spatialReject();
  }
  auto key=valueKey(first);
  if(values.size()>1) {
    const auto stride=static_cast<__int128>(values[1].instanceIndex())-first.instanceIndex();
    bool affine=stride>=INT64_MIN && stride<=INT64_MAX;
    for(std::size_t i=0;affine && i<values.size();++i)
      affine=static_cast<__int128>(first.instanceIndex())+stride*i==values[i].instanceIndex();
    if(affine)spatial_instance_strides.insert_or_assign(key,static_cast<std::int64_t>(stride));
    else spatial_instance_strides.erase(key);
  }
  const auto stride=spatial_instance_strides.find(key);
  if(!first.conversion() && stride!=spatial_instance_strides.end())
    registerSpatialFamily({&input,first,spatial_count,stride->second,{}});
  else {
    const auto base=first;
    registerSpatialFamily({&input,base,spatial_count,0,std::move(values)});
  }
}

inline auto Lower::spatialOffsets(const std::vector<std::uint64_t>& offsets) -> Offset {
  if(offsets.size()!=spatial_count || offsets.front()>std::uint64_t(INT64_MAX))spatialReject();
  const auto step=static_cast<__int128>(offsets[1])-offsets[0];
  if(step<INT64_MIN || step>INT64_MAX)spatialReject();
  for(std::size_t i=0;i<offsets.size();++i)
    if(static_cast<__int128>(offsets[0])+step*i!=offsets[i] || offsets[i]>std::uint64_t(INT64_MAX))spatialReject();
  Offset result(static_cast<std::int64_t>(offsets[0]));
  if(step)result.terms[0]=static_cast<std::int64_t>(step);
  return result;
}

inline auto Lower::prepareSpatial(v::Value value) -> void {
  const auto& members=spatialValues(value);
  if(value.conversion()) {
    std::vector<v::Value> sources;
    SENTRYWARD_AUDIT(if(members.count>1)audit->event("INSTANCE_ENUMERATION",
      "spatial_conversion_sources",members.count,1,value.nodeId(),true));
    for(std::size_t i=0;i<members.count;++i)sources.push_back(members.at(i).conversion()->source);
    registerSpatial(std::move(sources));return;
  }
  const auto& node=input.definingNode(value);
  const auto& op=node.operation();
  if(scalarOperation(op)) {
    if(members.affine() && node.hasUniformThreadTemplate()) {
      std::vector<SpatialValues> children;
      bool supported=true;
      for(const auto& operand:node.operands()) {
        const auto resolved=input.resolve(operand);
        if(resolved!=operand || !operand.indices().empty() || operand.threadSelector() ||
           operand.pointer() || operand.conversion() || !input.type(operand).isScalar()) {supported=false;break;}
        const auto& child=input.definingNode(operand);
        if(child.storedInstanceCount()!=1){supported=false;break;}
        auto base=operand;std::int64_t stride=0;
        const auto& threads=child.instanceThreads();
        if(child.origin().cta==node.origin().cta && !threads.empty() &&
           threads.at(operand.instanceIndex())==node.origin().thread) {
          if(threads.storageIdentity()!=node.instanceThreads().storageIdentity()) {supported=false;break;}
          base=child.result(operand.resultIndex(),value.instanceIndex());stride=members.stride;
        }
        children.push_back({&input,base,spatial_count,stride,{}});
      }
      if(supported) {
        for(auto& child:children)registerSpatialFamily(std::move(child));
        ++result.spatial_template_operations;
        return;
      }
    }
    const auto first=input.instanceOperands(value);
    std::vector<std::vector<v::Value>> operands(first.size());
    SENTRYWARD_AUDIT(if(members.count>1)audit->event("INSTANCE_ENUMERATION",
      "spatial_operand_fallback",members.count,1,value.nodeId(),true));
    for(std::size_t member=0;member<members.count;++member) {
      const auto item=input.instanceOperands(members.at(member));
      if(item.size()!=operands.size())spatialReject();
      for(std::size_t i=0;i<operands.size();++i)operands[i].push_back(item[i]);
    }
    for(auto& items:operands)registerSpatial(std::move(items));
  } else if(const auto* opaque=dynamic_cast<const v::OpaqueValueOp*>(&op))unspecified(value,*opaque);
  else if(!dynamic_cast<const v::LoadOp*>(&op) && !dynamic_cast<const v::ConstantOp*>(&op) &&
            !dynamic_cast<const v::InputParameterOp*>(&op) && !dynamic_cast<const v::CoordinateOp*>(&op))spatialReject();
}

} // namespace
} // namespace sentryward::value_analysis
