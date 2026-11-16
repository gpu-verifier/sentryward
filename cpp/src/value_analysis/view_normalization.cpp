#include "sentryward/algorithm/diagnostics/development_trace.h"
#include "sentryward/value_analysis/view_normalization.h"
#include "sentryward/algorithm/diagnostics/compression_audit.h"
#include "sentryward/algorithm/diagnostics/performance_profile.h"
#include "sentryward/value_analysis/indexed_source.h"
#include "sentryward/value_analysis/pure_source_templates.h"
#include "sentryward/value_analysis/packed_storage.h"
#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <iostream>
#include <set>
#include <stdexcept>
#include <utility>
#include <unordered_map>
#include "sentryward/value_ir/operations.h"
#include "sentryward/value_ir/native_matrix.h"
#include "sentryward/value_ir/type_conversion.h"

namespace sentryward::value_analysis {
namespace {
namespace v = value_ir;
std::uint64_t add(std::uint64_t a, std::uint64_t b) {
  if (b > std::numeric_limits<std::uint64_t>::max()-a)
    throw std::invalid_argument("memory view offset overflow");
  return a+b;
}
std::uint64_t bytes(const v::ValueType& type) {
  if (type.bitSize()%8) throw std::invalid_argument("memory view requires byte-aligned storage");
  return type.bitSize()/8;
}
bool same(const ViewSource& a, const ViewSource& b) {
  if (a.index()!=b.index()) return false;
  if (auto* x=std::get_if<v::Value>(&a)) return *x==std::get<v::Value>(b);
  const auto& x=std::get<InitialMemorySymbol>(a);
  const auto& y=std::get<InitialMemorySymbol>(b);
  return x.id==y.id && x.region==y.region && x.type==y.type;
}
struct Piece {
  ViewSource source;
  std::uint64_t offset, size;
};
class Normalizer {
 public:
  Normalizer(const v::Graph& graph, MemoryEliminationResult& result)
      : g(graph), r(result), storage_decoder(graph), pure_templates(graph,result) {
    for (const auto& [key, read]:r.reads) {
      (void)key;
      if (auto* initial=std::get_if<InitialMemorySymbol>(&read)) collect(*initial);
      else if (auto* slice=std::get_if<ReinterpretedRead>(&read)) collect(slice->source);
      else if (auto* assembly=std::get_if<AssembledRead>(&read))
        for (const auto& piece:assembly->pieces) collect(piece.source);
    }
    for (auto& [name,symbol]:initials) {
      (void)name;
      symbol.type=v::ValueType::matrix(v::ElementType::bit(8),{symbol.region.bytes});
    }
  }
  ViewNormalizationStatistics run() {
    SENTRYWARD_AUDIT_STAGE("view_normalization");
    SENTRYWARD_AUDIT(audit->add(AuditCounter::input_output_records,r.outputs.size());
      audit->add(AuditCounter::input_output_families,r.output_families.size()));
    if (!r.complete) throw std::invalid_argument("cannot normalize incomplete memory elimination");
    if (r.views_normalized) return stats;
    sentryward::algorithm::ProfileScope reads_profile("value_analysis.view_normalization.reads");
    // The reads output records store from (a TMA store's staging read): their partitions are the
    // outputs' projection domains, and keep the previous template planning (no shared-subtree
    // policy). The stored records only (store families are register stores, never expanded):
    // O(records log records).
    for(const auto& output:r.outputs) {
      const auto value=g.resolve(output.value);
      if(!value.conversion() && value.indices().empty() && !value.threadSelector() &&
         dynamic_cast<const v::LoadOp*>(&g.definingNode(value).operation()))
        output_reads.insert(r.canonicalRead({value.nodeId(),value.instanceIndex()}));
    }
    for (const auto& [key, raw]:r.reads) {
      const auto target=g.type(g.node(key.node).result(0,key.instance));
      if (const auto* initial=std::get_if<InitialMemorySymbol>(&raw);
          initial && initial->type==target && initial->region.bytes==bytes(target)) {
        ++stats.reused_initial_reads;
        continue;
      }
      const auto* assembly=std::get_if<AssembledRead>(&raw);
      const auto raw_hash=assembly && assembly->pieces.size()>=3?
          std::optional<std::size_t>(rawPlanHash(target,*assembly)):std::nullopt;
      if(raw_hash && reuseRawPlan(key,target,*assembly,*raw_hash))continue;
      const auto& flat=read(key);
      if (flat.size()==1) {
        if (auto* value=std::get_if<v::Value>(&flat[0].source);
            value && flat[0].offset==0 && bytes(g.type(*value))==bytes(target) &&
            (g.type(*value)==target ||
             (g.type(*value).isScalar() && target.isScalar()))) {
          r.normalized_reads.emplace(key,g.type(*value)==target ? *value : g.bitcast(*value,target));
          ++stats.identities;
          if(raw_hash)rememberRawPlan(key,target,*raw_hash);
          continue;
        }
      }
      const auto plan_hash=flat.size()>=3?std::optional<std::size_t>(readPlanHash(target,flat)):std::nullopt;
      if(plan_hash && reuseReadPlan(key,target,flat,*plan_hash)) {
        if(raw_hash)rememberRawPlan(key,target,*raw_hash);continue;
      }
      if(flat.size()>=3 && indexedRead(key,target,flat)) {
        rememberReadPlan(key,target,*plan_hash);
        if(raw_hash)rememberRawPlan(key,target,*raw_hash);continue;
      }
      MemoryView view{target,MemoryViewKind::Segmented,{},{},{}};
      std::uint64_t dest=0;
      for (const auto& piece:flat) {
        auto found=std::find_if(view.sources.begin(),view.sources.end(),
            [&](const auto& source){return same(source,piece.source);});
        auto index=static_cast<std::size_t>(found-view.sources.begin());
        if (found==view.sources.end()) view.sources.push_back(piece.source);
        view.blocks.push_back({index,piece.offset,dest,piece.size,{}});
        dest=add(dest,piece.size);
      }
      if (dest!=bytes(target)) throw std::invalid_argument("memory view does not cover its result");
      compact(view.blocks);
      if (flat.size()==1) {
        view.kind=MemoryViewKind::Slice;
        if (auto* value=std::get_if<v::Value>(&flat[0].source)) {
          const auto& source_type=g.type(*value);
          if (target.isScalar() && source_type.isMatrix() && source_type.hasStaticShape() &&
              !source_type.element().isParameterized() && source_type.element().width()%8==0 &&
              source_type.element().width()==target.bitSize() &&
              flat[0].offset%(source_type.element().width()/8)==0) {
            view.kind=MemoryViewKind::Element;
            auto index=flat[0].offset/(source_type.element().width()/8);
            view.indices.resize(source_type.shape().size());
            for (std::size_t j=view.indices.size();j-->0;) {
              view.indices[j]=index%source_type.shape()[j];
              index/=source_type.shape()[j];
            }
          }
        }
      } else if (view.blocks.size()==1) view.kind=MemoryViewKind::Layout;
      switch(view.kind) {
        case MemoryViewKind::Element: ++stats.elements; break;
        case MemoryViewKind::Slice: ++stats.slices; break;
        case MemoryViewKind::Layout: ++stats.layouts; break;
        case MemoryViewKind::Segmented: ++stats.segmented; break;
      }
      stats.mapping_blocks+=view.blocks.size();
      if(view.blocks.size()==1)storeView(key,std::move(view));
      else {
        SENTRYWARD_AUDIT(audit->event("DOMAIN_SPLIT","read_view_source_partitions",
            view.blocks.size(),1,key.node));
        PartitionedRead partition{target,{}};
        std::uint64_t cursor=0;
        for(const auto& block:view.blocks) {
          auto extent=block.bytes;
          for(const auto& axis:block.axes) {
            if(axis.result_stride!=extent || !axis.count || extent>UINT64_MAX/axis.count) {
              r.complete=false;r.diagnostic=MemoryEliminationDiagnostic{
                "compact_view_unsupported","view regions are not contiguous affine domains",key};
              return stats;
            }
            extent*=axis.count;
          }
          if(block.result_offset!=cursor)throw std::invalid_argument("noncontiguous read partition");
          auto one=block;one.source=0;
          partition.regions.push_back({cursor,extent,MemoryView{target,MemoryViewKind::Slice,
              {view.sources.at(block.source)},{std::move(one)},{}}});
          cursor=add(cursor,extent);
        }
        stats.stored_mapping_blocks+=partition.regions.size();
        r.normalized_reads.emplace(key,std::move(partition));
      }
      if(plan_hash)rememberReadPlan(key,target,*plan_hash);
      if(raw_hash)rememberRawPlan(key,target,*raw_hash);
    }
    r.views_normalized=true;
    reads_profile.Stop();
    {
      sentryward::algorithm::ProfileScope plans("value_analysis.view_normalization.read_plans");
      plans.nodes=read_plan_hits+raw_plan_hits;plans.edges=read_plan_misses+raw_plan_misses;
      plans.pruned=raw_plan_hits;
    }
    sentryward::algorithm::ProfileCall("value_analysis.view_normalization.outputs",[&]{normalizeOutputs();});
    sentryward::algorithm::ProfileCall("value_analysis.view_normalization.output_domains",[&]{normalizeOutputDomains();});
    return stats;
  }
 private:
  struct ReadPlan {MemoryInstance key,owner;v::ValueType type;};
  std::size_t sourceHash(const ViewSource& source) const {
    std::size_t hash=0;
    const auto mix=[&](std::size_t x){hash^=x+0x9e3779b9+(hash<<6)+(hash>>2);};
    const auto valueHash=[&](auto&& self,const v::Value& value)->void {
      mix(value.graphIdentity());mix(value.nodeId());mix(value.instanceIndex());mix(value.resultIndex());
      mix(value.indices().size());for(const auto& index:value.indices())self(self,index);
      mix(value.threadSelector()!=nullptr);if(value.threadSelector())self(self,*value.threadSelector());
      mix(bool(value.pointer()));if(value.pointer()) {
        const auto& p=*value.pointer();mix(std::hash<std::string>{}(p.identity));
        mix(std::hash<std::string>{}(p.addressing));
        mix(p.owner_cta.has_value());if(p.owner_cta)mix(*p.owner_cta);
        mix(p.owner_thread.has_value());if(p.owner_thread)mix(*p.owner_thread);
      }
      mix(bool(value.conversion()));if(value.conversion()) {
        const auto& c=*value.conversion();mix(static_cast<std::size_t>(c.spec->kind));
        mix(c.spec->target.bitSize());mix(c.spec->element.bitSize());
        mix(static_cast<std::size_t>(c.spec->attributes.rounding));mix(c.spec->attributes.saturate);
        mix(c.spec->attributes.flush_to_zero);mix(c.spec->attributes.saturate_finite);
        self(self,c.source);
      }
    };
    mix(source.index());
    if(const auto* value=std::get_if<v::Value>(&source))valueHash(valueHash,*value);
    else {
      const auto& initial=std::get<InitialMemorySymbol>(source);
      mix(initial.id);mix(std::hash<std::string>{}(initial.region.allocation));
      mix(initial.region.offset);mix(initial.region.bytes);mix(initial.type.bitSize());
    }
    return hash;
  }
  std::size_t readPlanHash(const v::ValueType& target,const std::vector<Piece>& pieces) const {
    std::size_t hash=std::hash<std::string>{}(target.toString());
    const auto mix=[&](std::size_t x){hash^=x+0x9e3779b9+(hash<<6)+(hash>>2);};
    mix(pieces.size());
    for(const auto& piece:pieces){mix(piece.offset);mix(piece.size);mix(sourceHash(piece.source));}
    return hash;
  }
  std::size_t rawPlanHash(const v::ValueType& target,const AssembledRead& read) const {
    std::size_t hash=std::hash<std::string>{}(target.toString());
    const auto mix=[&](std::size_t x){hash^=x+0x9e3779b9+(hash<<6)+(hash>>2);};
    mix(read.type.bitSize());mix(read.pieces.size());
    for(const auto& piece:read.pieces){mix(piece.byte_offset);mix(piece.type.bitSize());mix(sourceHash(piece.source));}
    return hash;
  }
  void aliasPlan(MemoryInstance key,MemoryInstance owner) {
    r.normalized_read_aliases.emplace(key,owner);++stats.reused_views;
    const auto& normalized=r.normalized_reads.at(owner);
    if(const auto* view=std::get_if<MemoryView>(&normalized)) {
      stats.mapping_blocks+=view->blocks.size();
      switch(view->kind) {
        case MemoryViewKind::Element:++stats.elements;break;
        case MemoryViewKind::Slice:++stats.slices;break;
        case MemoryViewKind::Layout:++stats.layouts;break;
        case MemoryViewKind::Segmented:++stats.segmented;break;
      }
    } else if(const auto* partition=std::get_if<PartitionedRead>(&normalized)) {
      ++stats.segmented;stats.mapping_blocks+=partition->regions.size();
    } else if(std::holds_alternative<v::Value>(normalized))++stats.identities;
  }
  bool reuseRawPlan(MemoryInstance key,const v::ValueType& target,const AssembledRead& raw,std::size_t hash) {
    if(auto found=raw_plans.find(hash);found!=raw_plans.end())for(const auto& plan:found->second) {
      const auto& previous=std::get<AssembledRead>(r.reads.at(plan.key));
      if(plan.type!=target || previous.type!=raw.type || previous.pieces.size()!=raw.pieces.size())continue;
      if(!std::equal(previous.pieces.begin(),previous.pieces.end(),raw.pieces.begin(),[](const auto& a,const auto& b) {
        return a.byte_offset==b.byte_offset && a.type==b.type && same(a.source,b.source);
      }))continue;
      // This equality is checked BEFORE forwarding/flattening. The immutable
      // raw environment and canonical initial allocations are the same, so
      // ancestor substitution and finite coordinate proofs need not be redone.
      aliasPlan(key,plan.owner);++raw_plan_hits;return true;
    }
    ++raw_plan_misses;return false;
  }
  bool reuseReadPlan(MemoryInstance key,const v::ValueType& target,
      const std::vector<Piece>& pieces,std::size_t hash) {
    if(auto found=read_plans.find(hash);found!=read_plans.end())for(const auto& plan:found->second) {
      if(plan.type!=target)continue;
      const auto& previous=read(plan.key);
      if(previous.size()!=pieces.size() || !std::equal(previous.begin(),previous.end(),pieces.begin(),
          [](const auto& a,const auto& b){return a.offset==b.offset && a.size==b.size && same(a.source,b.source);}))continue;
      // Equal resolved pieces in the immutable read environment reuse a proven
      // source map, not the identity or execution occurrence of the read.
      aliasPlan(key,plan.owner);++read_plan_hits;return true;
    }
    ++read_plan_misses;return false;
  }
  void rememberReadPlan(MemoryInstance key,const v::ValueType& target,std::size_t hash) {
    auto owner=key;
    if(auto alias=r.normalized_read_aliases.find(key);alias!=r.normalized_read_aliases.end())owner=alias->second;
    if(r.normalized_reads.contains(owner))read_plans[hash].push_back({key,owner,target});
  }
  void rememberRawPlan(MemoryInstance key,const v::ValueType& target,std::size_t hash) {
    auto owner=key;
    if(auto alias=r.normalized_read_aliases.find(key);alias!=r.normalized_read_aliases.end())owner=alias->second;
    if(r.normalized_reads.contains(owner))raw_plans[hash].push_back({key,owner,target});
  }
  // Lift repeated initial-memory rows into one affine outer axis. Every
  // stored row is checked; no tensor member or expression ancestry is visited.
  static void coalesceInitialRows(PartitionedRead& partition) {
    // Mixed/interleaved Value domains retain their AP partition contract.
    // Initial-only partitions are in contiguous destination order.
    if(!std::all_of(partition.regions.begin(),partition.regions.end(),[](const auto& r) {
      return !r.view.indexed_source && r.view.sources.size()==1 &&
          std::holds_alternative<InitialMemorySymbol>(r.view.sources[0]);
    }))return;
    std::vector<ReadRegion> result;
    for(std::size_t i=0;i<partition.regions.size();) {
      const auto& first=partition.regions[i];const auto& view=first.view;
      const auto eligible=[](const ReadRegion& r) {
        if(r.view.indexed_source || r.view.sources.size()!=1 || r.view.blocks.size()!=1 ||
            !std::holds_alternative<InitialMemorySymbol>(r.view.sources[0]))return false;
        const auto& b=r.view.blocks[0];auto extent=b.bytes;
        if(b.source || b.result_offset!=r.begin || !extent)return false;
        for(const auto& axis:b.axes) {
          if(!axis.count || axis.result_stride!=extent || extent>UINT64_MAX/axis.count)return false;
          extent*=axis.count;
        }
        return extent==r.bytes;
      };
      if(!eligible(first) || i+1==partition.regions.size()) {
        result.push_back(std::move(partition.regions[i++]));continue;
      }
      const auto& source=std::get<InitialMemorySymbol>(view.sources[0]);const auto& block=view.blocks[0];
      const auto origin=add(source.region.offset,block.source_offset);
      std::optional<std::uint64_t> step;auto end=i+1;
      auto source_end=add(source.region.offset,source.region.bytes);
      for(;end<partition.regions.size();++end) {
        const auto& row=partition.regions[end];
        if(!eligible(row) || row.bytes!=first.bytes ||
            static_cast<unsigned __int128>(first.begin)+static_cast<unsigned __int128>(end-i)*first.bytes!=row.begin)break;
        const auto& b=row.view.blocks[0];const auto& s=std::get<InitialMemorySymbol>(row.view.sources[0]);
        if(s.region.allocation!=source.region.allocation || b.bytes!=block.bytes || b.axes!=block.axes)break;
        const auto at=add(s.region.offset,b.source_offset);
        if(at<origin)break;
        if(!step)step=at-origin;
        if(static_cast<unsigned __int128>(origin)+static_cast<unsigned __int128>(end-i)* *step!=at)break;
        source_end=std::max(source_end,add(s.region.offset,s.region.bytes));
      }
      if(end==i+1){result.push_back(std::move(partition.regions[i++]));continue;}
      auto merged=first;auto& s=std::get<InitialMemorySymbol>(merged.view.sources[0]);
      s.region.offset=origin;s.region.bytes=source_end-origin;
      s.type=v::ValueType::matrix(v::ElementType::bit(8),{s.region.bytes});
      auto& b=merged.view.blocks[0];b.source_offset=0;
      b.axes.push_back({end-i,*step,first.bytes});
      if(first.bytes>UINT64_MAX/(end-i))throw std::invalid_argument("initial row extent overflow");
      merged.bytes=first.bytes*(end-i);merged.view.kind=MemoryViewKind::Layout;
      result.push_back(std::move(merged));i=end;
    }
    partition.regions=std::move(result);
  }
  // Join already certified row domains by shared use identity and exact
  // coordinate increments. No producer-tree matching or member expansion.
  static void coalesceIndexedRows(PartitionedRead& destination) {
    // Normalize use spelling, not mathematical expressions. Work on stored
    // regions only and commit atomically when the whole row layout is valid.
    if(destination.type.shape().size()!=2 || destination.type.element().width()%8)return;
    auto partition=destination;
    const auto reject=[&](const char* reason) {
      SENTRYWARD_AUDIT((void)audit;static unsigned reports=0;
        if(sentryward::debug::environment("SENTRYWARD_OUTPUT_DOMAIN_WITNESS") && partition.regions.size()>100 && reports++<8)
          SENTRYWARD_DEBUG_PRINT(view_normalization_0));
    };
    auto rows=partition.type.shape()[0];
    auto row_bytes=partition.type.shape()[1]*(partition.type.element().width()/8);
    if(rows<2 || !row_bytes)return;
    // A role may repeat every several physical rows. Use that stored period
    // as the row-block radix rather than enumerating its missing rows. This
    // changes only index coordinates, never source values or occurrences.
    std::uint64_t period=0;
    for(const auto& region:partition.regions)if(region.view.blocks.size()==1) {
      const auto& block=region.view.blocks[0];
      if(block.axes.size()==1 && block.axes[0].count>1) {
        const auto stride=block.axes[0].result_stride;
        if(stride>row_bytes && stride%row_bytes==0)period=std::gcd(period,stride);
      }
    }
    const auto total=partition.type.bitSize()/8;
    if(period>row_bytes && total%period==0){row_bytes=period;rows=total/period;}
    using UseKey=std::tuple<int,std::uint64_t,std::size_t,std::size_t,std::size_t>;
    std::map<UseKey,std::vector<std::shared_ptr<const IndexedSourceUse>>> uses;
    const auto intern=[&](v::Value value,IndexedSourceUse::Kind kind) {
      auto& bucket=uses[{int(kind),value.graphIdentity(),value.nodeId(),value.resultIndex(),value.instanceIndex()}];
      for(const auto& use:bucket)if(use->prototype==value)return use;
      auto use=std::make_shared<IndexedSourceUse>(IndexedSourceUse{value,kind,{},{}});
      if(kind==IndexedSourceUse::Kind::MemberTemplate) {
        auto binding=std::make_shared<IndexedValueBinding>();binding->leaf=value;binding->stride=1;
        use->binding=std::move(binding);
      }
      bucket.push_back(use);return std::shared_ptr<const IndexedSourceUse>(std::move(use));
    };
    for(auto& region:partition.regions) {
      auto& view=region.view;
      if(view.blocks.size()!=1)return reject("block_count");
      auto& block=view.blocks[0];
      std::shared_ptr<const IndexedSourceUse> child;
      std::int64_t base=0,stride=0;
      std::uint64_t count=1;
      v::Value prototype;
      if(!view.indexed_source) {
        if(!block.axes.empty() || block.source>=view.sources.size())return reject("plain_axes_or_source");
        const auto* value=std::get_if<v::Value>(&view.sources[block.source]);
        if(!value)return reject("plain_initial_memory");
        prototype=*value;child=intern(prototype,IndexedSourceUse::Kind::Exact);
        block.axes.push_back({1,0,block.bytes});
      } else {
        const auto& source=*view.indexed_source;
        if(source.zero_mask || !source.root)return reject("masked_or_missing_root");
        if(source.root->kind==IndexedSourceUse::Kind::CoordinateMap)continue;
        const auto& root=*source.root;
        if(root.kind!=IndexedSourceUse::Kind::MemberTemplate || !root.binding ||
            root.binding->period || !root.binding->quotient_terms.empty() ||
            root.binding->power_of_two || root.binding->numerical_field)return reject("nonlinear_member_or_other_root");
        prototype=source.prototype;count=source.count;
        base=root.binding->base;stride=root.binding->stride;
        child=intern(root.prototype,IndexedSourceUse::Kind::MemberTemplate);
      }
      auto binding=std::make_shared<IndexedValueBinding>();
      binding->leaf=prototype;binding->base=base;binding->stride=stride;
      auto root=std::make_shared<IndexedSourceUse>(IndexedSourceUse{
          prototype,IndexedSourceUse::Kind::CoordinateMap,std::move(binding),{std::move(child)}});
      view.indexed_source=std::make_shared<IndexedSource>(IndexedSource{prototype,count,std::move(root)});
    }
    using Key=std::tuple<std::uintptr_t,std::uint64_t,std::uint64_t,std::uint64_t,
                         std::uint64_t,std::uint64_t,std::int64_t>;
    std::map<Key,std::vector<std::size_t>> groups;
    std::vector<ReadRegion> carried;
    for(std::size_t i=0;i<partition.regions.size();++i) {
      const auto& r=partition.regions[i];const auto& v=r.view;
      if(!v.indexed_source || v.blocks.size()!=1 || v.indexed_source->zero_mask)return;
      const auto& b=v.blocks[0];const auto& root=v.indexed_source->root;
      if(b.axes.size()==1 && b.axes[0].result_stride &&
          b.axes[0].result_stride<row_bytes && row_bytes%b.axes[0].result_stride==0) {
        const auto inner=row_bytes/b.axes[0].result_stride;
        const auto& axis=b.axes[0];
        if(axis.count>inner && axis.count%inner==0 &&
           axis.source_stride<=UINT64_MAX/inner &&
           r.begin%row_bytes+(inner-1)*axis.result_stride+b.bytes<=row_bytes &&
           r.bytes==(axis.count-1)*axis.result_stride+b.bytes) {
          auto column=r;auto& axes=column.view.blocks[0].axes;
          axes={{inner,axis.source_stride,axis.result_stride},
                {axis.count/inner,axis.source_stride*inner,row_bytes}};
          carried.push_back(std::move(column));continue;
        }
      }
      // The interleaved planner may already have compressed a column across
      // every row. Preserve it, expressing the singleton inner axis explicitly
      // so all consumers see the same two-axis row-grid contract.
      if(b.axes.size()==1 && b.axes[0].result_stride==row_bytes &&
          b.axes[0].count>0 && r.begin/row_bytes<rows &&
          b.axes[0].count<=rows-r.begin/row_bytes &&
          b.bytes<=row_bytes-r.begin%row_bytes && b.result_offset==r.begin &&
          r.bytes==(b.axes[0].count-1)*row_bytes+b.bytes) {
        auto column=r;
        column.view.blocks[0].axes.insert(column.view.blocks[0].axes.begin(),{1,0,b.bytes});
        carried.push_back(std::move(column));continue;
      }
      // A grid family's region (indexedRead) is already a two-axis row grid when its outer
      // axis steps whole rows and its inner axis stays inside one row.
      if(b.axes.size()==2 && b.axes[1].result_stride%row_bytes==0 && b.result_offset==r.begin &&
          r.begin%row_bytes+(b.axes[0].count-1)*b.axes[0].result_stride+b.bytes<=row_bytes) {
        carried.push_back(r);continue;
      }
      if(b.axes.size()!=1 || !root || root->kind!=IndexedSourceUse::Kind::CoordinateMap ||
          root->children.size()!=1 || !root->binding || root->binding->period ||
          !root->binding->quotient_terms.empty() || root->binding->power_of_two || root->binding->numerical_field ||
          r.begin%row_bytes+r.bytes>row_bytes || b.result_offset!=r.begin) {
        SENTRYWARD_AUDIT((void)audit;static unsigned details=0;
          if(sentryward::debug::environment("SENTRYWARD_OUTPUT_DOMAIN_WITNESS") && partition.regions.size()>100 && details++<4) {
            SENTRYWARD_DEBUG_PRINT(view_normalization_1);
            for(const auto& a:b.axes)SENTRYWARD_DEBUG_PRINT(view_normalization_2);
            SENTRYWARD_DEBUG_PRINT(view_normalization_3);
          });
        return reject("row_geometry");
      }
      groups[{reinterpret_cast<std::uintptr_t>(root->children[0].get()),r.begin%row_bytes,
          b.bytes,b.source_offset,b.axes[0].count,b.axes[0].result_stride,
          root->binding->stride}].push_back(i);
    }
    std::vector<ReadRegion> merged=std::move(carried);merged.reserve(merged.size()+groups.size());
    // Rows left single by their group (one payload offset per group) whose payload offset
    // instead progresses along the rows, e.g. the rows of a transposed operand that read one
    // source template at the next element of each row: keyed like `groups` without the payload
    // offset, merged below with that offset as the row axis's source stride.
    std::map<std::tuple<std::uintptr_t,std::uint64_t,std::uint64_t,std::uint64_t,std::uint64_t,std::int64_t>,
        std::vector<std::size_t>> singles;
    // Runs of rows in begin order: equal row steps, exact template-base and payload-offset
    // progressions. A run of one row is returned to `singles` (first pass) or kept (second pass).
    const auto runs=[&](std::vector<std::size_t>& indices,bool payload) {
      std::sort(indices.begin(),indices.end(),[&](auto a,auto b){
        return partition.regions[a].begin<partition.regions[b].begin;});
      for(std::size_t start=0;start<indices.size();) {
      const auto& first=partition.regions[indices[start]];
      const auto& source=*first.view.indexed_source;
      const auto& binding=*source.root->binding;
      std::int64_t delta=0;
      std::uint64_t payload_step=0;
      auto row_step=row_bytes;
      auto end=start+1;
      if(end<indices.size()) {
        const auto& second=partition.regions[indices[end]];
        const auto next=second.view.indexed_source->root->binding->base;
        if(__builtin_sub_overflow(next,binding.base,&delta))delta=0;
        const auto next_begin=second.begin;
        if(next_begin>first.begin && (next_begin-first.begin)%row_bytes==0)
          row_step=next_begin-first.begin;
        if(payload && second.view.blocks[0].source_offset>first.view.blocks[0].source_offset)
          payload_step=second.view.blocks[0].source_offset-first.view.blocks[0].source_offset;
      }
      for(;end<indices.size();++end) {
        const auto row=end-start;const auto& r=partition.regions[indices[end]];
        if(static_cast<unsigned __int128>(r.begin)!=static_cast<unsigned __int128>(first.begin)+
                static_cast<unsigned __int128>(row)*row_step || r.bytes!=first.bytes ||
            static_cast<__int128>(binding.base)+static_cast<__int128>(row)*delta!=
                r.view.indexed_source->root->binding->base ||
            static_cast<unsigned __int128>(r.view.blocks[0].source_offset)!=
                static_cast<unsigned __int128>(first.view.blocks[0].source_offset)+
                static_cast<unsigned __int128>(row)*payload_step)break;
      }
      const auto count=end-start;
      // Each group owns its exact row interval and column progression. Different
      // groups need not have identical/aligned heights: consumers validate their
      // AP intersections on overlapping row intervals, not bounding rectangles.
      if(count==1){
        const auto& b=first.view.blocks[0];const auto& root=first.view.indexed_source->root;
        if(!payload) {
          singles[{reinterpret_cast<std::uintptr_t>(root->children[0].get()),first.begin%row_bytes,
              b.bytes,b.axes[0].count,b.axes[0].result_stride,root->binding->stride}].push_back(indices[start]);
          start=end;continue;
        }
        auto r=first;r.view.blocks[0].axes.push_back({1,0,row_bytes});
        merged.push_back(std::move(r));start=end;continue;
      }
      auto r=first;auto pattern=std::make_shared<IndexedSource>(source);
      auto map=std::make_shared<IndexedValueBinding>(binding);
      map->period=source.count;map->outer_stride=delta;
      auto root=std::make_shared<IndexedSourceUse>(*source.root);root->binding=std::move(map);
      pattern->root=std::move(root);pattern->count*=count;r.view.indexed_source=std::move(pattern);
      r.view.blocks[0].axes.push_back({count,payload_step,row_step});
      r.bytes=add(r.bytes,(count-1)*row_step);merged.push_back(std::move(r));start=end;
      }
    };
    for(auto& [key,indices]:groups){(void)key;runs(indices,false);}
    for(auto& [key,indices]:singles){(void)key;runs(indices,true);}
    std::sort(merged.begin(),merged.end(),[](const auto& a,const auto& b){return a.begin<b.begin;});
    partition.regions=std::move(merged);
    destination=std::move(partition);
  }
  static void coalescePayloadRows(PartitionedRead& partition) {
    // The member map and the payload map are independent. Adjacent register
    // components of the same source template are an axis, not new templates.
    // Group stored two-axis rules; do not enumerate any represented members.
    using Key=std::tuple<std::uintptr_t,std::int64_t,std::int64_t,std::uint64_t,std::int64_t,
        std::uint64_t,std::uint64_t,std::uint64_t,std::uint64_t,std::uint64_t,
        std::uint64_t,std::uint64_t,std::uint64_t,std::uint64_t,std::uint64_t>;
    std::map<Key,std::vector<std::size_t>> groups;
    std::vector<ReadRegion> merged;
    for(std::size_t i=0;i<partition.regions.size();++i) {
      const auto& r=partition.regions[i];const auto& view=r.view;
      if(!view.indexed_source || view.indexed_source->zero_mask || view.blocks.size()!=1) {
        merged.push_back(r);continue;
      }
      const auto& p=*view.indexed_source;const auto& root=p.root;const auto& b=view.blocks[0];
      if(!root || root->kind!=IndexedSourceUse::Kind::CoordinateMap || !root->binding ||
          root->children.size()!=1 || b.axes.size()!=2 || !b.axes[0].count || !b.axes[1].count ||
          b.result_offset!=r.begin ||
          !b.axes[1].result_stride || b.axes[0].count>UINT64_MAX/b.axes[1].count ||
          p.count!=b.axes[0].count*b.axes[1].count) {merged.push_back(r);continue;}
      const auto& m=*root->binding;
      if(m.fixed || m.initial || m.producer || m.producer_uses_bindings || m.numerical_field ||
          m.power_of_two || !m.quotient_terms.empty() || m.leaf!=p.prototype ||
          root->prototype!=p.prototype) {merged.push_back(r);continue;}
      groups[{reinterpret_cast<std::uintptr_t>(root->children[0].get()),m.base,m.stride,m.period,m.outer_stride,
          p.count,b.bytes,b.axes[0].count,b.axes[0].source_stride,b.axes[0].result_stride,
          b.axes[1].count,b.axes[1].source_stride,b.axes[1].result_stride,
          r.begin/b.axes[1].result_stride,r.bytes}].push_back(i);
    }
    for(auto& [key,indices]:groups) {
      (void)key;
      std::sort(indices.begin(),indices.end(),[&](auto a,auto b){return partition.regions[a].begin<partition.regions[b].begin;});
      for(std::size_t first=0;first<indices.size();) {
        const auto& origin=partition.regions[indices[first]];const auto& block=origin.view.blocks[0];
        const auto& pattern=*origin.view.indexed_source;
        auto end=first+1;std::uint64_t result_step=0,payload_step=0;
        if(end<indices.size()) {
          const auto& next=partition.regions[indices[end]];
          if(next.begin>origin.begin && next.view.blocks[0].source_offset>=block.source_offset) {
            result_step=next.begin-origin.begin;
            payload_step=next.view.blocks[0].source_offset-block.source_offset;
          }
        }
        const auto inner_extent=static_cast<unsigned __int128>(block.axes[0].count-1)*block.axes[0].result_stride+block.bytes;
        if(result_step>=inner_extent && payload_step && result_step<block.axes[1].result_stride)
          for(;end<indices.size();++end) {
            const auto& r=partition.regions[indices[end]];const auto n=end-first;
            if(r.view.indexed_source->prototype!=pattern.prototype ||
                r.view.sources.size()!=origin.view.sources.size() ||
                !std::equal(r.view.sources.begin(),r.view.sources.end(),origin.view.sources.begin(),same) ||
                r.view.blocks[0].source!=block.source ||
                r.view.blocks[0].result_offset!=r.begin ||
                static_cast<unsigned __int128>(origin.begin)+n*static_cast<unsigned __int128>(result_step)!=r.begin ||
                static_cast<unsigned __int128>(block.source_offset)+n*static_cast<unsigned __int128>(payload_step)!=r.view.blocks[0].source_offset ||
                origin.begin%block.axes[1].result_stride+n*static_cast<unsigned __int128>(result_step)+inner_extent>
                    block.axes[1].result_stride)break;
          }
        const auto count=end-first;
        auto out=origin;
        if(count>1 && pattern.count<=UINT64_MAX/count && block.axes[0].count<=UINT64_MAX/count &&
            block.axes[0].count<=INT64_MAX) {
          const auto inner=block.axes[0].count;
          auto map=std::make_shared<IndexedValueBinding>();map->leaf=pattern.prototype;
          map->period=inner;map->stride=1;
          map->quotient_terms.push_back({inner*count,std::int64_t(inner)});
          auto source=std::make_shared<IndexedSource>(pattern);source->count*=count;
          source->root=std::make_shared<IndexedSourceUse>(IndexedSourceUse{pattern.prototype,
              IndexedSourceUse::Kind::CoordinateMap,std::move(map),{pattern.root}});
          out.view.indexed_source=std::move(source);
          out.view.blocks[0].axes.insert(out.view.blocks[0].axes.begin()+1,{count,payload_step,result_step});
          out.bytes=add(out.bytes,(count-1)*result_step);
        } else end=first+1;
        merged.push_back(std::move(out));first=end;
      }
    }
    std::sort(merged.begin(),merged.end(),[](const auto& a,const auto& b){return a.begin<b.begin;});
    partition.regions=std::move(merged);
  }
  // A transposed read: pieces of equal width whose sources are few existing values,
  // each source read at one constant payload step and one constant result step (a column-major
  // MMA operand over row values, e.g. V tokens written row by row). One region per source with
  // that axis: the result domains interleave and stay disjoint (distinct residues of one common
  // result step). Exact: every piece keeps its source and payload byte; each region is checked at
  // every piece. Applies only when every source contributes at least two pieces and all share
  // one result step, the uniform interleaved contract of the fold's partition check, and only
  // in place of a partition that stays fragmented (indexedRead). O(pieces log sources); no
  // member enumeration beyond the stored pieces.
  std::optional<PartitionedRead> transposedRead(const v::ValueType& target,const std::vector<Piece>& pieces) {
    if(pieces.size()<4)return std::nullopt;
    const auto width=pieces.front().size;
    struct Group {ViewSource source;std::uint64_t begin,source_offset,result_step=0,source_step=0,count=0,last;};
    using Key=std::tuple<std::size_t,std::size_t,std::size_t,std::size_t>;
    std::map<Key,std::size_t> index;std::vector<Group> groups;
    std::uint64_t dest=0;
    for(const auto& piece:pieces) {
      const auto* value=std::get_if<v::Value>(&piece.source);
      if(!value || piece.size!=width || !width)return std::nullopt;
      const Key k{value->graphIdentity(),value->nodeId(),value->resultIndex(),value->instanceIndex()};
      auto [found,fresh]=index.try_emplace(k,groups.size());
      if(fresh)groups.push_back({piece.source,dest,piece.offset,0,0,1,dest});
      else {
        auto& g=groups[found->second];
        if(!same(g.source,piece.source))return std::nullopt;
        const auto result_step=dest-g.begin,source_offset=piece.offset;
        if(g.count==1) {
          if(source_offset<g.source_offset)return std::nullopt;
          g.result_step=result_step;g.source_step=source_offset-g.source_offset;
        } else if(result_step!=g.result_step*g.count ||
                  source_offset!=g.source_offset+g.source_step*g.count)return std::nullopt;
        ++g.count;g.last=dest;
      }
      dest=add(dest,piece.size);
    }
    if(dest!=bytes(target) || groups.size()*2>pieces.size())return std::nullopt;
    const auto step=groups.front().result_step;
    if(!step || step<=width)return std::nullopt;
    for(const auto& g:groups)if(g.count<2 || g.result_step!=step)return std::nullopt;
    PartitionedRead partition{target,{},true};
    for(const auto& g:groups) {
      MemoryView view{target,MemoryViewKind::Layout,{g.source},
          {{0,g.source_offset,g.begin,width,{{g.count,g.source_step,step}}}},{}};
      partition.regions.push_back({g.begin,(g.count-1)*step+width,std::move(view)});
    }
    std::sort(partition.regions.begin(),partition.regions.end(),[](const auto& a,const auto& b){return a.begin<b.begin;});
    return partition;
  }
  bool indexedRead(MemoryInstance key,const v::ValueType& target,const std::vector<Piece>& pieces) {
    PartitionedRead partition{target,{},true};
    const bool uniform_width=std::all_of(pieces.begin(),pieces.end(),[&](const auto& p){
      return p.size==pieces.front().size;
    });
    bool compressed=false;
    std::uint64_t dest=0;
    std::vector<v::Value> values;
    std::vector<std::uint64_t> payload_offsets;
    for(std::size_t i=0;i<pieces.size();) {
      const auto& first=pieces[i];
      if(const auto* initial=std::get_if<InitialMemorySymbol>(&first.source);
         initial && first.size && i+1<pieces.size()) {
        const auto start=add(initial->region.offset,first.offset);
        const auto* next=std::get_if<InitialMemorySymbol>(&pieces[i+1].source);
        if(next && next->region.allocation==initial->region.allocation &&
            pieces[i+1].size==first.size && add(next->region.offset,pieces[i+1].offset)>=start) {
          const auto step=add(next->region.offset,pieces[i+1].offset)-start;
          auto end=i+2;
          while(end<pieces.size()) {
            const auto* candidate=std::get_if<InitialMemorySymbol>(&pieces[end].source);
            if(!candidate || candidate->region.allocation!=initial->region.allocation ||
                pieces[end].size!=first.size ||
                static_cast<unsigned __int128>(start)+static_cast<unsigned __int128>(end-i)*step!=
                    static_cast<unsigned __int128>(candidate->region.offset)+pieces[end].offset)break;
            ++end;
          }
          const auto span=static_cast<unsigned __int128>(end-i-1)*step+first.size;
          if(span>UINT64_MAX)throw std::invalid_argument("initial source range overflow");
          auto source=*initial;source.region.offset=start;source.region.bytes=std::uint64_t(span);
          source.type=v::ValueType::matrix(v::ElementType::bit(8),{source.region.bytes});
          const auto extent=(end-i)*first.size;
          MemoryView view{target,MemoryViewKind::Layout,{std::move(source)},
              {{0,0,dest,first.size,{{end-i,step,first.size}}}},{}};
          partition.regions.push_back({dest,extent,std::move(view)});
          dest=add(dest,extent);i=end;compressed=true;continue;
        }
      }
      // A read can slice one existing matrix producer at affine byte offsets.
      // Preserve that mapping directly; fixed component offsets are not new
      // expressions to match. Scan the already stored fragments once.
      if(std::holds_alternative<v::Value>(first.source) && first.size && i+1<pieces.size() &&
          same(first.source,pieces[i+1].source) && first.size==pieces[i+1].size &&
          pieces[i+1].offset>=first.offset) {
        const auto step=pieces[i+1].offset-first.offset;
        auto end=i+2;
        while(end<pieces.size() && same(first.source,pieces[end].source) &&
              pieces[end].size==first.size &&
              static_cast<unsigned __int128>(first.offset)+
                  static_cast<unsigned __int128>(end-i)*step==pieces[end].offset)++end;
        const auto extent=(end-i)*first.size;
        MemoryView view{target,MemoryViewKind::Layout,{first.source},
            {{0,first.offset,dest,first.size,{{end-i,step,first.size}}}},{}};
        partition.regions.push_back({dest,extent,std::move(view)});
        dest=add(dest,extent);i=end;compressed=true;continue;
      }
      if(std::holds_alternative<v::Value>(first.source) && first.size) {
        auto end=i;values.clear();payload_offsets.clear();
        while(end<pieces.size() && pieces[end].size==first.size &&
              std::holds_alternative<v::Value>(pieces[end].source)) {
          values.push_back(std::get<v::Value>(pieces[end].source));
          payload_offsets.push_back(pieces[end].offset);++end;
        }
        // The same source-domain plan is consumed by output projection.
        // Existing finite records are validated, never expanded a second time.
        // Grid families (two-axis domains) only for equal-width pieces of a read that is not an
        // output staging read: their regions are two-axis views over the family's member index.
        const bool staging=output_reads.contains(key);
        auto plan=pure_templates.map(values,uniform_width,payload_offsets,!staging,uniform_width && !staging,
            /*grid_runs=*/false);
        // A two-axis region must be a row grid of the read's matrix: its outer axis steps one row,
        // its inner axis stays inside a row. Other layouts (every second row, with another
        // family's rows between) keep the one-axis plan, which the fold's partitions refine.
        const auto rowGrid=[&](const IndexedSourceDomain& d) {
          if(d.outer_count==1)return true;
          const auto& shape=target.shape();
          if(shape.size()!=2 || target.element().width()%8)return false;
          const auto row_bytes=shape[1]*(target.element().width()/8);
          const auto size=pieces[i+d.begin].size;
          const auto begin=add(dest,d.begin*size);
          return row_bytes && d.outer_stride*size==row_bytes &&
              begin%row_bytes+(d.count-1)*d.result_stride*size+size<=row_bytes;
        };
        if(!std::all_of(plan.begin(),plan.end(),rowGrid))
          plan=pure_templates.map(values,uniform_width,payload_offsets,!staging);
        for(const auto& domain:plan) {
          const auto& piece=pieces[i+domain.begin];
          if(domain.count>UINT64_MAX/piece.size || domain.outer_count>UINT64_MAX/piece.size/std::max<std::size_t>(domain.outer_stride,1))
            throw std::invalid_argument("indexed view size overflow");
          const auto start=add(dest,domain.begin*piece.size);
          auto extent=add((domain.count-1)*domain.result_stride*piece.size,piece.size);
          MemoryView view{target,domain.source?MemoryViewKind::Layout:MemoryViewKind::Slice,
              {piece.source},{{0,piece.offset,start,piece.size,{}}},{},domain.source};
          if(domain.source) {
            view.blocks[0].axes.push_back({domain.count,0,domain.result_stride*piece.size});compressed=true;
            if(domain.outer_count>1) {
              view.blocks[0].axes.push_back({domain.outer_count,0,domain.outer_stride*piece.size});
              extent=add(extent,(domain.outer_count-1)*domain.outer_stride*piece.size);
            }
          }
          partition.regions.push_back({start,extent,std::move(view)});
        }
        dest=add(dest,(end-i)*first.size);
        i=end;
      } else {
        partition.regions.push_back({dest,first.size,MemoryView{target,MemoryViewKind::Slice,
            {first.source},{{0,first.offset,dest,first.size,{}}},{}}});
        dest=add(dest,first.size);++i;
      }
    }
    (void)compressed;
    if(dest!=bytes(target))throw std::invalid_argument("indexed view does not cover result");
    coalesceInitialRows(partition);
    coalesceIndexedRows(partition);
    coalescePayloadRows(partition);
    // A partition that stays fragmented (about one region per piece): the transposed form when
    // it applies.
    if(partition.regions.size()*4>=pieces.size()*3)
      if(auto transposed=transposedRead(target,pieces);transposed && transposed->regions.size()<partition.regions.size())
        partition=std::move(*transposed);
    stats.mapping_blocks+=partition.regions.size();stats.stored_mapping_blocks+=partition.regions.size();
    if(partition.regions.size()==1) {
      ++stats.layouts;r.normalized_reads.emplace(key,std::move(partition.regions[0].view));
    } else {++stats.segmented;r.normalized_reads.emplace(key,std::move(partition));}
    return true;
  }
  std::optional<v::Value> numericStorage(v::Value value) const {
    value=g.resolve(value);
    while(g.type(value).element().kind()==v::ElementKind::Bit && value.conversion()) {
      const auto& conversion=*value.conversion();
      if(conversion.spec->kind!=v::TypeConversionKind::Bitcast ||
          g.type(conversion.source).bitSize()!=g.type(value).bitSize())break;
      value=g.resolve(conversion.source);
    }
    const auto& type=g.type(value);
    if(value.pointer() || !type.isScalar() || type.element().isParameterized() ||
        !type.element().isNumeric() || !type.bitSize() || type.bitSize()%8)return {};
    return value;
  }
  std::vector<OutputWrite> typedOutput(const OutputWrite& output) const {
    const auto value=g.resolve(output.value);
    const auto& type=g.type(value);
    if(!output.region.bytes || !type.isMatrix() || !type.hasStaticShape() ||
        type.element()!=v::ElementType::bit(8))return {};
    const auto& node=g.definingNode(value);
    if(value!=node.result(value.resultIndex(),value.instanceIndex()) ||
        !dynamic_cast<const v::LoadOp*>(&node.operation()))return {};
    const auto replacement=r.normalizedRead({value.nodeId(),value.instanceIndex()});
    if(!replacement)return {};
    std::vector<const MemoryView*> views;
    if(const auto* stored_view=std::get_if<const MemoryView*>(&*replacement))views.push_back(*stored_view);
    else if(const auto* partition=std::get_if<const PartitionedRead*>(&*replacement))
      for(const auto& region:(*partition)->regions)views.push_back(&region.view);
    else return {};
    const auto begin=output.value_byte_offset,end=add(begin,output.region.bytes);
    std::vector<OutputWrite> pieces;
    for(const auto* item:views) {
      const auto& view=*item;
      if(view.indexed_source)return {};
      for(const auto& block:view.blocks) {
      // Repeated axes remain compact in the original view.
      if(!block.axes.empty())return {};
      const auto block_end=add(block.result_offset,block.bytes);
      if(block_end<=begin || block.result_offset>=end)continue;
      if(block.result_offset<begin || block_end>end || block.source_offset ||
          !block.bytes)return {};
      const auto* source=std::get_if<v::Value>(&view.sources.at(block.source));
      if(!source)return {};
      const auto numeric=numericStorage(*source);
      if(!numeric || bytes(g.type(*numeric))!=block.bytes)return {};
      auto region=output.region;
      region.offset=add(region.offset,block.result_offset-begin);
      region.bytes=block.bytes;
      // Storage bitcasts preserve these bytes; numeric conversions stay on the value.
      pieces.push_back({std::move(region),*numeric,output.writer,0,{},output.base});
      }
    }
    std::sort(pieces.begin(),pieces.end(),[](const auto& a,const auto& b) {
      return a.region.offset<b.region.offset;
    });
    auto cursor=output.region.offset;
    for(const auto& piece:pieces) {
      if(piece.region.offset!=cursor)return {};
      cursor=add(cursor,piece.region.bytes);
    }
    if(cursor!=add(output.region.offset,output.region.bytes))return {};
    return pieces;
  }
  std::optional<v::ElementType> compactOutputType(const OutputWrite& output) const {
    const auto value=g.resolve(output.value);
    const auto& type=g.type(value);
    if(!output.region.bytes || !type.isMatrix() || !type.hasStaticShape() ||
       type.element()!=v::ElementType::bit(8) || value.conversion() ||
       !value.indices().empty() || value.threadSelector() ||
       !dynamic_cast<const v::LoadOp*>(&g.definingNode(value).operation()))return {};
    auto read=r.normalizedRead({value.nodeId(),value.instanceIndex()});
    if(!read)return {};
    std::vector<const MemoryView*> views;
    if(auto direct=std::get_if<const MemoryView*>(&*read))views.push_back(*direct);
    else if(auto partition=std::get_if<const PartitionedRead*>(&*read))
      for(const auto& region:(*partition)->regions)views.push_back(&region.view);
    else return {};
    const auto begin=output.value_byte_offset,end=add(begin,output.region.bytes);
    std::optional<v::ElementType> element;
    std::uint64_t covered=0;
    for(const auto* view:views) {
      if(view->sources.size()!=1 || view->blocks.size()!=1)return {};
      const auto& block=view->blocks[0];
      if(block.axes.size()==1 && block.axes[0].result_stride!=block.bytes) {
        const auto& axis=block.axes[0];
        if(!axis.count || !axis.result_stride || begin%block.bytes || end%block.bytes)return {};
        const auto first=begin<=block.result_offset?0:(begin-block.result_offset+axis.result_stride-1)/axis.result_stride;
        const auto last=end<add(block.result_offset,block.bytes)?0:
            std::min(axis.count,1+(end-block.result_offset-block.bytes)/axis.result_stride);
        if(first>=last)continue;
        const auto* source=std::get_if<v::Value>(&view->sources[0]);
        const auto numeric=source?numericStorage(*source):std::nullopt;
        if(!numeric || block.source_offset || bytes(g.type(*numeric))!=block.bytes)return {};
        const auto candidate=g.type(*numeric).element();
        if(candidate.kind()!=v::ElementKind::Floating || (element && *element!=candidate))return {};
        element=candidate;covered=add(covered,(last-first)*block.bytes);continue;
      }
      auto extent=block.bytes;
      for(const auto& axis:block.axes) {
        if(!axis.count || axis.result_stride!=extent || extent>UINT64_MAX/axis.count)return {};
        extent*=axis.count;
      }
      const auto lo=std::max(begin,block.result_offset),hi=std::min(end,add(block.result_offset,extent));
      if(lo>=hi)continue;
      const auto* source=std::get_if<v::Value>(&view->sources[0]);
      if(!source)return {};
      auto numeric=numericStorage(*source);
      if(!numeric || block.source_offset || bytes(g.type(*numeric))!=block.bytes)return {};
      const auto candidate=g.type(*numeric).element();
      // Integer/bit views can intentionally expose individual storage bytes.
      // Recover a numerical tensor domain only from floating-point producers.
      if(candidate.kind()!=v::ElementKind::Floating)return {};
      if(element && *element!=candidate)return {};
      if((lo-block.result_offset)%block.bytes || (hi-lo)%block.bytes)return {};
      // Keep a typed output region even when its existing source domains are
      // distinct. Projection consumes those domains; do not materialize a new
      // scalar OutputWrite list solely because the source nodes differ.
      element=candidate;covered=add(covered,hi-lo);
    }
    if(covered!=output.region.bytes)return {};
    return element;
  }
  void normalizeOutputs() {
    std::vector<OutputWrite> outputs;
    outputs.reserve(r.outputs.size());
    for(const auto& output:r.outputs) {
      // A reduced element keeps its record; the fold combines its contents.
      if(output.reduced){outputs.push_back(output);continue;}
      if(auto element=compactOutputType(output)) {
        auto typed=output;typed.element_type=*element;
        outputs.push_back(std::move(typed));
        continue;
      }
      auto pieces=typedOutput(output);
      if(pieces.empty())outputs.push_back(output);
      else {
        SENTRYWARD_AUDIT(if(pieces.size()>1)
          audit->event("DOMAIN_SPLIT","typed_output_region_to_numeric_pieces",
              pieces.size(),1,output.writer.node));
        ++stats.typed_output_regions;
        stats.typed_output_pieces+=pieces.size();
        outputs.insert(outputs.end(),pieces.begin(),pieces.end());
      }
    }
    r.outputs=std::move(outputs);
    SENTRYWARD_AUDIT(audit->add(AuditCounter::output_records,r.outputs.size());
      audit->add(AuditCounter::output_families,r.output_families.size()));
  }
  // Reuse the finite pieces already resolved by read(). Do not enumerate a
  // compressed source's elements, producer instances or expression ancestry.
  // Rollback here only discards source references, never constructed math.
  bool outputSourceRejected([[maybe_unused]] const OutputWrite& output,
                            [[maybe_unused]] const char* reason) {
    SENTRYWARD_AUDIT(const auto& type=g.type(output.value);
      audit->event("OUTPUT_SOURCE_BOUNDARY",reason,output.region.bytes,
        type.hasStaticShape() && !type.element().isParameterized()?type.bitSize():0,output.value.nodeId()));
    return false;
  }
  bool appendOutputSources(const OutputWrite& output,std::vector<v::Value>& values,
                           std::optional<v::ElementType>& element) {
    const auto value=g.resolve(output.value);
    const auto& type=g.type(value);
    if(!output.region.bytes || !type.isMatrix() || !type.hasStaticShape())
      return outputSourceRejected(output,"not_a_static_matrix");
    if(type.element()!=v::ElementType::bit(8))
      return outputSourceRejected(output,"not_byte_matrix_storage");
    if(value.conversion() ||
       !value.indices().empty() || value.threadSelector() ||
       !dynamic_cast<const v::LoadOp*>(&g.definingNode(value).operation()))
      return outputSourceRejected(output,"not_plain_read");
    const auto key=r.canonicalRead({value.nodeId(),value.instanceIndex()});
    // Initial reads have no computed output template to recover.
    if(!r.readReplacement(key))return outputSourceRejected(output,"no_read_substitution");
    const auto& flat=read(key);
    const auto& ends=piece_ends.at(key);
    const auto begin=output.value_byte_offset,end=add(begin,output.region.bytes);
    const auto saved_size=values.size();const auto saved_element=element;
    auto i=static_cast<std::size_t>(std::upper_bound(ends.begin(),ends.end(),begin)-ends.begin());
    auto cursor=i?ends[i-1]:0;
    const auto reject=[&](const char* reason) {
      values.resize(saved_size);element=saved_element;return outputSourceRejected(output,reason);
    };
    for(;i<flat.size() && cursor<end;++i) {
      const auto& piece=flat[i];
      if(cursor<begin || ends[i]>end || piece.offset)return reject("not_whole_storage_field");
      const auto* source=std::get_if<v::Value>(&piece.source);
      if(!source)return reject("initial_memory_source");
      const auto numeric=numericStorage(*source);
      if(!numeric || bytes(g.type(*numeric))!=piece.size)return reject("not_whole_numeric_scalar");
      const auto candidate=g.type(*numeric).element();
      if(candidate.kind()!=v::ElementKind::Floating ||
         (element && *element!=candidate) ||
         (output.element_type && *output.element_type!=candidate))return reject("element_type_boundary");
      element=candidate;values.push_back(*numeric);cursor=ends[i];
    }
    return cursor==end || reject("incomplete_source_coverage");
  }
  // Reuse an already certified read-result coordinate system. The physical
  // run and the source byte interval must both be contiguous; sharing a
  // producer alone does not establish this correspondence.
  bool reuseNormalizedOutputRun(const std::vector<std::size_t>& records,
                                std::size_t begin,std::size_t end) {
    if(begin==end)return false;
    const auto& first=r.outputs[records[begin]];
    if(!first.element_type || first.element_type->kind()!=v::ElementKind::Floating ||
       !first.element_type->width() || first.element_type->width()%8)return false;
    const auto width=first.element_type->width()/8;
    const auto read=[&](const OutputWrite& output)->std::optional<NormalizedReadReference> {
      const auto value=g.resolve(output.value);
      const auto& type=g.type(value);
      if(!type.isMatrix() || type.element()!=v::ElementType::bit(8) ||
         value.conversion() || value.threadSelector() || !value.indices().empty() ||
         !dynamic_cast<const v::LoadOp*>(&g.definingNode(value).operation()))return {};
      return r.normalizedRead({value.nodeId(),value.instanceIndex()});
    };
    const auto source=read(first);
    if(!source || (!std::holds_alternative<const MemoryView*>(*source) &&
                   !std::holds_alternative<const PartitionedRead*>(*source)))return false;
    auto source_end=first.value_byte_offset;
    auto physical_end=first.region.offset;
    for(auto i=begin;i<end;++i) {
      const auto& output=r.outputs[records[i]];
      if(output.element_type!=first.element_type || output.region.allocation!=first.region.allocation ||
         output.region.offset!=physical_end || output.value_byte_offset!=source_end ||
         !output.region.bytes || output.region.bytes%width || read(output)!=source)return false;
      source_end=add(source_end,output.region.bytes);physical_end=add(physical_end,output.region.bytes);
    }
    if(first.value_byte_offset%width || source_end-first.value_byte_offset<=width)return false;
    OutputViewGroup group{{records.begin()+begin,records.begin()+end},{}};
    auto cursor=first.value_byte_offset;
    const auto append=[&](const MemoryView& view,std::uint64_t lo,std::uint64_t hi) {
      lo=std::max(lo,first.value_byte_offset);hi=std::min(hi,source_end);
      if(lo>=hi)return true;
      if(lo!=cursor || lo%width || (hi-lo)%width)return false;
      // compactOutputType established whole numeric fields and exact coverage
      // before OutputWrite.element_type was populated. Keep the certified map
      // unchanged, including producer occurrence, mask and absolute offsets.
      group.domains.push_back({{first.region.allocation,
          add(first.region.offset,lo-first.value_byte_offset),hi-lo},*first.element_type,
          view,{(hi-lo)/width},{width},lo});
      cursor=hi;return true;
    };
    if(const auto* view=std::get_if<const MemoryView*>(&*source)) {
      if(!append(**view,0,bytes((*view)->type)))return false;
    } else {
      const auto* split=std::get<const PartitionedRead*>(*source);
      for(const auto& region:split->regions) {
        const auto& block=region.view.blocks.at(0);
        if(!block.axes.empty() && block.axes[0].result_stride!=block.bytes)return false;
        if(!append(region.view,region.begin,add(region.begin,region.bytes)))return false;
      }
    }
    if(cursor!=source_end || group.domains.empty())return false;
    for(auto i=begin;i<end;++i)r.normalized_output_members[records[i]]=true;
    SENTRYWARD_AUDIT(audit->event("DOMAIN_REUSED","certified_output_read_coordinates",
        (source_end-first.value_byte_offset)/width,group.domains.size(),first.value.nodeId()));
    r.normalized_output_groups.emplace(records[begin],std::move(group));
    return true;
  }
  void normalizeOutputDomains() {
    [[maybe_unused]] unsigned geometry_witnesses=0;
    // Physical address order can interleave CTAs (e.g. [row,head,column]).
    // Bucket only the already explicit records; never merge CTA semantics or
    // enumerate an indexed view. Within a bucket, addresses remain sorted.
    // A symbolic base is part of the bucket: positions of different bases never form one run.
    using Bucket=std::tuple<std::string,std::vector<SymbolicAddressTerm>,std::optional<std::size_t>>;
    std::map<Bucket,std::vector<std::size_t>> buckets;
    for(std::size_t i=0;i<r.outputs.size();++i) {
      const auto& output=r.outputs[i];
      if(output.reduced)continue;
      buckets[{output.region.allocation,output.base,g.definingNode(output.value).origin().cta}].push_back(i);
    }
    r.normalized_output_members.assign(r.outputs.size(),false);
    struct Run {
      std::size_t begin,end; // Positions in this bucket's existing-record list.
      std::uint64_t offset,bytes;
      v::ElementType element;
      std::vector<v::Value> values;
    };
    for(const auto& [bucket,records]:buckets) {
      std::vector<Run> runs;
      // One attempt per physical run: a rejected source map must not cause
      // quadratic rescanning of every remaining suffix.
      std::size_t normalized_attempt_end=0;
      for(std::size_t begin=0;begin<records.size();) {
        const auto& first=r.outputs[records[begin]];
        if(begin>=normalized_attempt_end) {
          auto end=begin+1;auto cursor=add(first.region.offset,first.region.bytes);
          while(end<records.size() && r.outputs[records[end]].region.offset==cursor) {
            cursor=add(cursor,r.outputs[records[end]].region.bytes);++end;
          }
          normalized_attempt_end=end;
          if(reuseNormalizedOutputRun(records,begin,end)){begin=end;continue;}
        }
        std::vector<v::Value> values;std::optional<v::ElementType> element;
        if(!appendOutputSources(first,values,element)) {++begin;continue;}
        auto cursor=add(first.region.offset,first.region.bytes);
        auto end=begin+1;
        for(;end<records.size();++end) {
          const auto& next=r.outputs[records[end]];
          if(next.region.offset!=cursor || !appendOutputSources(next,values,element))break;
          cursor=add(cursor,next.region.bytes);
        }
        runs.push_back({begin,end,first.region.offset,cursor-first.region.offset,*element,std::move(values)});
        begin=end;
      }
      for(std::size_t begin=0;begin<runs.size();) {
        const auto& first=runs[begin];auto end=begin+1;
        auto stride=first.bytes;
        if(end<runs.size() && runs[end].begin==first.end &&
           runs[end].element==first.element && runs[end].bytes==first.bytes &&
           runs[end].offset>=add(first.offset,first.bytes)) {
          stride=runs[end].offset-first.offset;++end;
          while(end<runs.size() && runs[end].begin==runs[end-1].end &&
                runs[end].element==first.element && runs[end].bytes==first.bytes &&
                runs[end].offset>=runs[end-1].offset && runs[end].offset-runs[end-1].offset==stride)++end;
        }
        const auto record_begin=first.begin,record_end=runs[end-1].end;
        const std::uint64_t width=first.element.width()/8;
        const auto columns=first.bytes/width,rows=static_cast<std::uint64_t>(end-begin);
        if(first.bytes%width || columns>UINT64_MAX/rows)
          throw std::logic_error("invalid output source rectangle");
        if(record_end-record_begin<2 || columns*rows<3) {begin=end;continue;}
        std::vector<v::Value> values;values.reserve(columns*rows);
        for(auto row=begin;row<end;++row) {
          if(runs[row].values.size()!=columns)throw std::logic_error("output row source coverage");
          values.insert(values.end(),runs[row].values.begin(),runs[row].values.end());
        }
        // A finite proof may end inside a row. Split its physical image into
        // prefix/full-rows/suffix (at most three rectangles), not scalar cells.
        // Grid families (two-axis domains): rows of equal source domains at one record step, e.g.
        // the heads of a [head, column] tensor whose columns come in several blocks. Their byte
        // layout must be two-axis too: one physical run, or whole physical rows per grid row.
        auto line=pure_templates.mapLine(values,false,{},true);
        auto plan=pure_templates.gridDomains(values,{},line);
        const auto gridLayout=[&](const IndexedSourceDomain& d) {
          if(rows==1 || (d.outer_count==1 && d.result_stride==1))return true;
          if(d.begin%columns+(d.count-1)*d.result_stride>=columns)return false;
          return d.outer_count==1 || (d.outer_stride%columns==0 &&
              d.begin/columns+(d.outer_count-1)*(d.outer_stride/columns)<rows);
        };
        // A grid plan is taken where the strided-row merge of the final consolidation would merge
        // the rows too (consolidate_outputs.cpp, tensorizeRows): families of at least
        // kGridOutputRows rows in a regular tiling, i.e. every domain of the rectangle has one
        // length. Rows cut into pieces of different lengths (column ranges computed differently)
        // keep their one-axis domains, whose per-row comparison those pieces need. O(domains).
        constexpr std::uint64_t kGridOutputRows=16;
        const auto regular=[&] {
          for(const auto& d:plan)
            if(d.count!=plan.front().count || (d.outer_count>1 && d.outer_count<kGridOutputRows))return false;
          return true;
        };
        if(plan.size()!=line.size() && !(std::all_of(plan.begin(),plan.end(),gridLayout) && regular()))plan=std::move(line);
        // A shared-subtree plan (one template per row whose shared parts are that row's own
        // Values) is projected row by row: its domains are not widened to several rows here.
        // (mapLine's answer; gridDomains does not change it.)
        const bool row_domains=pure_templates.last_map_shared;
        SENTRYWARD_AUDIT((void)audit;for(const auto& domain:plan) {
          if(geometry_witnesses>=2)break;
          ++geometry_witnesses;
          const auto& output=r.outputs[records[record_begin]];
          SENTRYWARD_DEBUG_PRINT(view_normalization_4);
          std::vector<std::shared_ptr<const IndexedSourceUse>> pending;
          if(domain.source)pending.push_back(domain.source->root);
          std::set<const IndexedSourceUse*> seen;
          // Bounded diagnostic, not a proof or a scan of all logical members.
          for(std::size_t cursor=0;cursor<pending.size() && seen.size()<64;++cursor) {
            const auto& use=pending[cursor];
            if(!use || !seen.insert(use.get()).second)continue;
            SENTRYWARD_DEBUG_PRINT(view_normalization_5);
            if(use->binding) {
              const auto& b=*use->binding;
              SENTRYWARD_DEBUG_PRINT(view_normalization_6);
              if(b.producer)SENTRYWARD_DEBUG_PRINT(view_normalization_7);
              if(b.initial)SENTRYWARD_DEBUG_PRINT(view_normalization_8);
              for(const auto& q:b.quotient_terms)SENTRYWARD_DEBUG_PRINT(view_normalization_9);
            }
            SENTRYWARD_DEBUG_PRINT(view_normalization_10);
            // The use graph is immutable; queue stored child references only.
            for(const auto& child:use->children)if(pending.size()<256)pending.push_back(child);
          }
          SENTRYWARD_DEBUG_PRINT(view_normalization_11);
        });
        OutputViewGroup group{{records.begin()+record_begin,records.begin()+record_end},{}};
        for(const auto& domain:plan) {
          if(domain.outer_count>1 || domain.result_stride!=1) {
            // A grid family: one domain of shape [outer, inner] over the family's member index.
            const auto members=domain.count*domain.outer_count;
            MemoryView view{v::ValueType::matrix(first.element,{members}),MemoryViewKind::Layout,
                {values[domain.begin]},{{0,0,0,width,{{members,0,width}}}},{},domain.source};
            const auto row_step=rows==1?domain.outer_stride*width:(domain.outer_stride/columns)*stride;
            const auto offset=add(first.offset,add((domain.begin/columns)*stride,(domain.begin%columns)*width));
            std::vector<std::uint64_t> shape{domain.count},strides{domain.result_stride*width};
            if(domain.outer_count>1){shape={domain.outer_count,domain.count};strides={row_step,domain.result_stride*width};}
            group.domains.push_back({{std::get<0>(bucket),offset,members*width},first.element,std::move(view),
                std::move(shape),std::move(strides),0});
            continue;
          }
          const auto type=v::ValueType::matrix(first.element,{domain.count});
          auto at=domain.begin,left=domain.count;
          while(left) {
            const auto column=at%columns;
            const auto full_rows=column==0?(row_domains?std::min<std::uint64_t>(1,left/columns):left/columns):0;
            const auto count=full_rows?full_rows*columns:std::min<std::uint64_t>(left,columns-column);
            MemoryView view{type,domain.source?MemoryViewKind::Layout:MemoryViewKind::Slice,
                {values[domain.begin]},{{0,0,0,width,{}}},{},domain.source};
            if(domain.source)view.blocks[0].axes.push_back({domain.count,0,width});
            std::vector<std::uint64_t> shape{count},strides{width};
            if(full_rows>1 && stride!=first.bytes) {shape={full_rows,columns};strides={stride,width};}
            const auto offset=add(first.offset,add((at/columns)*stride,column*width));
            group.domains.push_back({{std::get<0>(bucket),offset,count*width},first.element,std::move(view),
                std::move(shape),std::move(strides),(at-domain.begin)*width});
            at+=count;left-=count;
          }
        }
        for(auto i=record_begin;i<record_end;++i) {
          if(r.normalized_output_members[records[i]])throw std::logic_error("output view has duplicate record membership");
          r.normalized_output_members[records[i]]=true;
        }
        SENTRYWARD_AUDIT(audit->event("DOMAIN_PLANNED","strided_cta_output_source_map",
            values.size(),group.domains.size(),r.outputs[records[record_begin]].value.nodeId()));
        r.normalized_output_groups.emplace(records[record_begin],std::move(group));
        begin=end;
      }
    }
    if(r.normalized_output_groups.empty())r.normalized_output_members.clear();
  }
  void storeView(MemoryInstance key,MemoryView view) {
    std::size_t hash=0;
    const auto mix=[&](std::size_t value) {hash^=value+0x9e3779b9+(hash<<6)+(hash>>2);};
    const auto text=[&](const std::string& value){mix(std::hash<std::string>{}(value));};
    text(view.type.toString());mix(static_cast<std::size_t>(view.kind));
    for(const auto& source:view.sources) {
      mix(source.index());
      if(const auto* value=std::get_if<v::Value>(&source)) {
        mix(value->nodeId());mix(value->instanceIndex());mix(value->resultIndex());
      } else {
        const auto& initial=std::get<InitialMemorySymbol>(source);
        mix(initial.id);text(initial.region.allocation);mix(initial.region.offset);
        mix(initial.region.bytes);text(initial.type.toString());
      }
    }
    for(const auto& block:view.blocks) {
      mix(block.source);mix(block.source_offset);mix(block.result_offset);mix(block.bytes);
      mix(block.axes.size());
      for(const auto& axis:block.axes){mix(axis.count);mix(axis.source_stride);mix(axis.result_stride);}
    }
    for(auto index:view.indices)mix(index);
    auto& bucket=views[hash];
    for(auto prior:bucket) {
      const auto& candidate=std::get<MemoryView>(r.normalized_reads.at(prior));
      if(candidate.type==view.type && candidate.kind==view.kind && candidate.blocks==view.blocks &&
         candidate.indices==view.indices && candidate.sources.size()==view.sources.size() &&
         std::equal(candidate.sources.begin(),candidate.sources.end(),view.sources.begin(),same)) {
        r.normalized_read_aliases.emplace(key,prior);++stats.reused_views;return;
      }
    }
    stats.stored_mapping_blocks+=view.blocks.size();
    r.normalized_reads.emplace(key,std::move(view));bucket.push_back(key);
  }
  void collect(const ViewSource& source) {
    if (auto* initial=std::get_if<InitialMemorySymbol>(&source)) collect(*initial);
  }
  void collect(const InitialMemorySymbol& symbol) {
    auto [it,inserted]=initials.emplace(symbol.region.allocation,symbol);
    if (!inserted) {
      auto& prior=it->second;
      const auto begin=std::min(prior.region.offset,symbol.region.offset);
      const auto end=std::max(add(prior.region.offset,prior.region.bytes),
                              add(symbol.region.offset,symbol.region.bytes));
      prior.region={symbol.region.allocation,begin,end-begin};
      prior.id=std::min(prior.id,symbol.id);
    }
  }
  void append(std::vector<Piece>& out, Piece piece) {
    if (!piece.size) return;
    if (auto* symbol=std::get_if<InitialMemorySymbol>(&piece.source)) {
      const auto& canonical=initials.at(symbol->region.allocation);
      piece.offset=add(piece.offset,symbol->region.offset-canonical.region.offset);
      piece.source=canonical;
    }
    if (!out.empty() && same(out.back().source,piece.source) &&
        add(out.back().offset,out.back().size)==piece.offset)
      out.back().size=add(out.back().size,piece.size);
    else out.push_back(std::move(piece));
  }
  void expand(ViewSource source, std::uint64_t offset, std::uint64_t size,
              std::vector<Piece>& out) {
    if (auto* value=std::get_if<v::Value>(&source)) {
      *value=g.resolve(*value);
      if (add(offset,size)>bytes(g.type(*value)))
        throw std::invalid_argument("memory view exceeds producer");
      // Component selection is an existing payload map, not a distinct
      // producer. Preserve its matrix/occurrence identity before planning
      // source domains. Numeric conversions are deliberately not crossed.
      if(!value->conversion() && !value->pointer() && !value->threadSelector() &&
         !value->indices().empty()) {
        const auto base=value->withoutSelection();
        const auto& type=g.type(base);
        const auto indices=value->indices();
        bool exact=type.hasStaticShape() && type.shape().size()==indices.size() &&
            !type.element().isParameterized() && type.element().width()%8==0;
        std::uint64_t linear=0;
        for(std::size_t axis=0;exact && axis<indices.size();++axis) {
          const auto index=g.resolve(indices[axis]);
          const auto* constant=dynamic_cast<const v::ConstantOp*>(&g.definingNode(index).operation());
          if(!constant || index.conversion() || index.pointer() || index.threadSelector() ||
             !index.indices().empty() || !g.type(index).element().isBitLike()) {exact=false;break;}
          std::size_t used=0;std::uint64_t n=0;
          try {
            if(constant->literal().empty() || constant->literal().front()=='-') {exact=false;break;}
            n=std::stoull(constant->literal(),&used,0);
          } catch(const std::exception&) {exact=false;break;}
          if(used!=constant->literal().size() || n>=type.shape()[axis] ||
             linear>(UINT64_MAX-n)/type.shape()[axis]) {exact=false;break;}
          linear=linear*type.shape()[axis]+n;
        }
        const auto width=type.element().isParameterized()?0:type.element().width()/8;
        if(exact && width && linear<=UINT64_MAX/width) {
          expand(base,add(offset,linear*width),size,out);
          return;
        }
      }
      // Resolve storage fields before matching adjacent numerical sources. The
      // scalar-width bound limits this to at most eight byte-aligned fields;
      // it never enumerates tensor elements or constructs a scalar expression.
      // A partially covered field or gap keeps the original exact byte view.
      if(g.type(*value).isScalar()) {
        if(auto fields=storage_decoder.borrowComposed(*value);fields && fields->size()>1) {
          std::uint64_t covered=0;
          bool exact=true;
          for(const auto& field:*fields) {
            if(field.start%8 || field.width%8) {exact=false;break;}
            const auto begin=field.start/8,end=begin+field.width/8;
            if(end<=offset || begin>=offset+size)continue;
            if(begin!=offset+covered || end>offset+size) {exact=false;break;}
            covered+=field.width/8;
          }
          if(exact && covered==size) {
            for(const auto& field:*fields)if(field.start/8>=offset && field.start/8<offset+size)
              expand(field.value,0,field.width/8,out);
            return;
          }
        }
      }
      const auto& node=g.definingNode(*value);
      if(r.symbolic_initial_reads.contains({value->nodeId(),value->instanceIndex()})) {
        append(out,{std::move(source),offset,size});
        return;
      }
      // A tcgen05.cp value of its source's type is that source byte for byte (the identity
      // transfer the scalar fold projects through): its pieces are the source's. A consumer
      // that reads the copy in another layout (an MMA operand over a swizzled tile) then meets
      // the tile's own sources, where the two layouts compose. O(1) here; the source read is
      // expanded once (cache).
      if (*value==node.result(value->resultIndex(),value->instanceIndex()))
        if (const auto* copy=dynamic_cast<const v::NativeMatrixOp*>(&node.operation());
            copy && copy->opcode()==v::NativeMatrixOpcode::TcgenCopyValue) {
          const auto operands=g.instanceOperands(*value);
          if (operands.size()==1 && g.type(operands[0])==g.type(*value)) {
            expand(operands[0],offset,size,out);
            return;
          }
        }
      if (*value==node.result(value->resultIndex(),value->instanceIndex()) &&
          dynamic_cast<const v::LoadOp*>(&node.operation())) {
        const auto key=r.canonicalRead({value->nodeId(),value->instanceIndex()});
        const auto& pieces=read(key);
        const auto& ends=piece_ends.at(key);
        const auto first=static_cast<std::size_t>(std::upper_bound(ends.begin(),ends.end(),offset)-ends.begin());
        std::uint64_t cursor=first?ends[first-1]:0, covered=0;
        for (auto i=first;i<pieces.size() && cursor<add(offset,size);++i) {
          const auto& piece=pieces[i];
          const auto end=ends[i];
          const auto lo=std::max(cursor,offset), hi=std::min(end,add(offset,size));
          if (lo<hi) {
            append(out,{piece.source,add(piece.offset,lo-cursor),hi-lo});
            covered+=hi-lo;
          }
          cursor=end;
        }
        if (covered!=size) throw std::invalid_argument("incomplete view forwarding");
        return;
      }
    } else {
      if (add(offset,size)>std::get<InitialMemorySymbol>(source).region.bytes)
        throw std::invalid_argument("memory view exceeds initial region");
    }
    append(out,{std::move(source),offset,size});
  }
  const std::vector<Piece>& read(MemoryInstance key) {
    key=r.canonicalRead(key);
    if (auto it=cache.find(key);it!=cache.end()) return it->second;
    if (!active.insert(key).second) throw std::invalid_argument("cyclic view substitution");
    const auto found=r.reads.find(key);
    if (found==r.reads.end()) throw std::invalid_argument("missing view read substitution");
    const auto& raw=found->second;
    std::vector<Piece> out;
    if (auto* value=std::get_if<v::Value>(&raw))
      expand(*value,0,bytes(g.type(g.node(key.node).result(0,key.instance))),out);
    else if (auto* initial=std::get_if<InitialMemorySymbol>(&raw))
      expand(*initial,0,bytes(initial->type),out);
    else if (auto* slice=std::get_if<ReinterpretedRead>(&raw))
      expand(slice->source,slice->byte_offset,bytes(slice->type),out);
    else {
      for (const auto& piece:std::get<AssembledRead>(raw).pieces) {
        ++stats.original_pieces;
        expand(piece.source,piece.byte_offset,bytes(piece.type),out);
      }
    }
    active.erase(key);
    auto& ends=piece_ends[key];
    ends.reserve(out.size());
    std::uint64_t end=0;
    for(const auto& piece:out){end=add(end,piece.size);ends.push_back(end);}
    return cache.emplace(key,std::move(out)).first->second;
  }
  static bool matching(const ViewBlock& a,const ViewBlock& b) {
    return a.source==b.source && a.bytes==b.bytes && a.axes==b.axes;
  }
  static void compact(std::vector<ViewBlock>& blocks) {
    // Fold repeated runs, then repeated groups of runs, preserving exact offsets.
    for (;;) {
      std::vector<ViewBlock> next;
      bool changed=false;
      for (std::size_t i=0;i<blocks.size();) {
        auto b=blocks[i];
        std::size_t end=i+1;
        if (end<blocks.size() && matching(b,blocks[end]) &&
            blocks[end].source_offset>=b.source_offset &&
            blocks[end].result_offset>b.result_offset) {
          const auto ss=blocks[end].source_offset-b.source_offset;
          const auto ds=blocks[end].result_offset-b.result_offset;
          ++end;
          while(end<blocks.size() && matching(b,blocks[end]) &&
                blocks[end].source_offset>=blocks[end-1].source_offset &&
                blocks[end].source_offset-blocks[end-1].source_offset==ss &&
                blocks[end].result_offset>=blocks[end-1].result_offset &&
                blocks[end].result_offset-blocks[end-1].result_offset==ds) ++end;
          b.axes.push_back({end-i,ss,ds});
          changed=true;
        }
        next.push_back(std::move(b)); i=end;
      }
      blocks=std::move(next);
      if (!changed) break;
    }
  }
  const v::Graph& g;
  MemoryEliminationResult& r;
  PackedStorageDecoder storage_decoder;
  PureSourceTemplates pure_templates;
  ViewNormalizationStatistics stats;
  std::unordered_map<std::size_t,std::vector<MemoryInstance>> views;
  std::unordered_map<std::size_t,std::vector<ReadPlan>> read_plans;
  std::unordered_map<std::size_t,std::vector<ReadPlan>> raw_plans;
  std::size_t read_plan_hits=0,read_plan_misses=0;
  std::size_t raw_plan_hits=0,raw_plan_misses=0;
  std::map<std::string,InitialMemorySymbol> initials;
  std::map<MemoryInstance,std::vector<Piece>> cache;
  std::map<MemoryInstance,std::vector<std::uint64_t>> piece_ends;
  std::set<MemoryInstance> active;
  // Canonical reads of output records (run()).
  std::set<MemoryInstance> output_reads;
};
}
ViewNormalizationStatistics normalizeMemoryViews(const value_ir::Graph& graph,
                                                 MemoryEliminationResult& result) {
  try {return Normalizer(graph,result).run();}
  catch(const IndexedSourceLimit& error) {
    result.complete=false;
    result.diagnostic=MemoryEliminationDiagnostic{"compact_view_resource_limit",error.what(),{}};
    return {};
  }
}
}  // namespace sentryward::value_analysis
