#pragma once
#include "scalar_fold_lowering_impl.hpp"

namespace sentryward::value_analysis {
namespace {

inline auto Lower::collectFrontier(const std::vector<Projection>& work,const ProjectionKey& root) -> void {
#if defined(SENTRYWARD_ENABLE_COMPRESSION_AUDIT)
  auto measurement=ledger?std::make_unique<debug::ProjectionLedger::Collection>(*ledger,result.graph,keys.size()):nullptr;
#endif
  struct CollectionScope {bool& flag;bool saved;~CollectionScope(){flag=saved;}}
    scope{collecting,collecting};
  collecting=true;
  SENTRYWARD_AUDIT(audit->add(AuditCounter::gc_calls);
    audit->add(AuditCounter::gc_input_nodes,result.graph.size());
    audit->event("GRAPH_COLLECTION","projection_frontier_collection",result.graph.size(),keys.size()));
  // Completed projections are memoized computations, including results
  // whose *first* reuse is in a later output. Keep the existing cache while
  // it fits within graph headroom; a hit-only policy loses those results
  // before their first reuse. At pressure retain only actual dependencies.
  // Nothing below adds nodes before this decision takes effect.
  const auto reserve=std::max<std::size_t>(1,options.max_nodes/4);
  const bool headroom=result.graph.size()<options.max_nodes-reserve;
  // With headroom every cache entry is a root, so none is evicted whatever
  // the dependency set holds: build it only under pressure (audit builds
  // also report its size).
  KeySet needed;
  if(!headroom || audit_build) {
    needed.insert(root);
    for(const auto& task:work)needed.insert(task.dependencies);
    for(const auto& [key,progress]:reduction_progress) {
      (void)key;
      needed.insert(progress.accumulator_key);
      needed.insert(progress.pending);
    }
  }
  std::vector<s::Ref> roots;
  for(const auto& output:result.outputs)roots.push_back(output.value);
  // Dense marks over this epoch's IDs: the membership answers of a node set,
  // without a heap node and rehash per live node.
  // 1: live and traced; 2: live, its operands still to mark. Node IDs are topological, so
  // a descending sweep traces every pending descendant, reading live nodes in storage order.
  std::vector<char> live(result.graph.size());std::size_t live_count=0;
  const auto isLive=[&](s::Id id){return id<live.size() && live[id];};
  std::vector<s::Id> pending;
  for(const auto& value:roots)pending.push_back(value.node);
  const auto trace=[&] {
    for(const auto id:pending){result.graph.node(id);if(!live[id])live[id]=2;}  // an invalid ID throws here
    pending.clear();
    const auto& nodes=result.graph.nodes();
    for(s::Id id=live.size();id-->0;)if(live[id]==2) {
      live[id]=1;++live_count;
      for(const auto& child:nodes[id].operands)if(!live[child.node])live[child.node]=2;
    }
  };
  trace();
  SENTRYWARD_AUDIT(audit->event("LIVE_GRAPH","completed_output_nodes",live_count,result.outputs.size());
    if(sentryward::debug::environment("SENTRYWARD_OUTPUT_DOMAIN_WITNESS") && live_count) {
      std::map<std::string,std::size_t> kinds,shapes;
      for(s::Id id=0;id<live.size();++id)if(live[id]) {
        const auto& n=result.graph.node(id);
        ++kinds[std::to_string(int(n.kind))+":"+(n.kind==s::Kind::Operation?n.name:"")];
      }
      for(const auto& [i,d]:output_domains) {
        (void)i;std::string shape;
        for(auto size:d.shape)shape+=std::to_string(size)+",";
        ++shapes[shape];
      }
      SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_42);
      for(const auto& [kind,count]:kinds)SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_43);
      SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_44);for(const auto& [shape,count]:shapes)SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_45);
      SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_46);
    });
  [[maybe_unused]] const auto output_live=live_count;
  needed.forEachId([&](std::uint32_t id) {  // empty under headroom outside audit builds
    if(const auto* value=keys.findId(id))pending.push_back(value->node);
  });
  if(headroom) {
    for(std::size_t i=0;i<keys.size();++i)pending.push_back(keys.valueAt(i).node);
    SENTRYWARD_LEDGER(ledger->add("headroom_projection_roots",keys.size()));
  } else {
    SENTRYWARD_LEDGER(ledger->add("pressure_weak_cache_collections"));
  }
  trace();
  SENTRYWARD_AUDIT(audit->event("LIVE_GRAPH","additional_frontier_nodes",live_count-output_live,needed.size());
    audit->event("LIVE_GRAPH","unreachable_before_collection",result.graph.size()-live_count,keys.size());
    if(sentryward::debug::environment("SENTRYWARD_COMPRESSION_AUDIT"))
      SENTRYWARD_DEBUG_PRINT(scalar_fold_lowering_47));
  keys.retainIf([&](std::uint32_t id,const s::Ref& value) {
    if(isLive(value.node) || needed.contains(id))return true;
    SENTRYWARD_LEDGER(ledger->evict(keys.textOf(id)));
    return false;
  });
  // Copy, not move: collect() first refreshes real-expression facts, and
  // that budget poll may raise a Failure the spatial-family loop recovers.
  // Nothing adds or erases a memo entry below, so positions stay valid.
  roots.reserve(roots.size()+keys.size());
  for(std::size_t i=0;i<keys.size();++i)roots.push_back(keys.valueAt(i));
  // Pending requests keep their selection contexts: the literals are roots and are remapped,
  // as are the partial results held across a projection (PinnedRoot).
  for(const auto& [serial,context]:select_contexts)
    for(const auto& literal:context.literals)roots.push_back(literal.first);
  roots.insert(roots.end(),pinned_roots.begin(),pinned_roots.end());
  result.nodes_reclaimed+=real_expressions.collect(roots,
      std::span<const s::Ref>(roots).subspan(result.outputs.size(),keys.size()));
  ++result.construction_collections;
  for(std::size_t i=0;i<result.outputs.size();++i)result.outputs[i].value=std::move(roots[i]);
  for(std::size_t i=0;i<keys.size();++i)keys.valueAt(i)=std::move(roots[result.outputs.size()+i]);
  auto remapped=roots.begin()+std::ptrdiff_t(result.outputs.size()+keys.size());
  // The same iteration order as above: nothing inserted or erased a context since.
  select_context_ids.clear();
  for(auto& [serial,context]:select_contexts) {
    for(auto& literal:context.literals)literal.first=std::move(*remapped++);
    select_context_ids.emplace(selectContextText(context.literals),serial);
  }
  for(auto& pinned:pinned_roots)pinned=std::move(*remapped++);
  if(result.graph.size()>options.max_nodes)
    fail("expression_node_limit_exceeded","scalar graph node limit exceeded");
  scheduleCollection();
}

// Observe, never root, cached projections. Consolidation must be allowed to
// discard concrete terms replaced by a Fold. Both collections remap only
// survivors, in O(cache entries) extra work; no ancestry search per entry.
inline auto Lower::observedCache() -> std::vector<std::optional<s::Id>> {
  std::vector<std::optional<s::Id>> weak_cache;
  weak_cache.reserve(keys.size());
  for(std::size_t i=0;i<keys.size();++i) {
    const auto& value=keys.valueAt(i);
    result.graph.node(value); // validate owner before retaining only the ID
    // The following collect deliberately drops completed Fold matcher seeds.
    // Do not cache an accumulator whose compact extension needs those seeds.
    if(reductions.hasMatchingState(value))weak_cache.emplace_back(std::nullopt);
    else weak_cache.emplace_back(value.node);
  }
  return weak_cache;
}

// The cache entries whose observed nodes survived (observedCache, then collections), remapped;
// the others are dropped. Returns how many were dropped.
inline auto Lower::retainObserved(const std::vector<std::optional<s::Id>>& weak_cache) -> std::size_t {
  std::size_t i=0,discarded=0;
  // No memo entry was added or erased since the observers were collected,
  // so the stable compaction visits them in the same order.
  keys.retainIf([&](std::uint32_t id,s::Ref& value) {
    const auto& weak=weak_cache[i++];
    if(weak) {
      const auto retained=result.graph.ref(*weak);
      value.node=retained.node;value.owner=retained.owner;
      return true;
    }
    SENTRYWARD_LEDGER(ledger->evict(keys.textOf(id)));
    (void)id;++discarded;
    return false;
  });
  return discarded;
}

// Phase f in lifting mode: with outputs pending and the graph at the pressure point (where a
// completed-output collection would run), the construction's unreachable nodes are collected
// before the pending outputs are lifted, which adds the members' bodies to the graph and its
// node budget (liftBudget). The roots and observed cache entries of a completed-output
// collection; no consolidation (phase f's own follows the lifting). O(graph).
inline auto Lower::collectBeforeFinalLifting() -> void {
  if(!options.unit_lifting || instantiated_end>=result.outputs.size() || result.graph.size()<pressureNodes())return;
  if(!pinned_roots.empty())throw std::logic_error("a partial result is pinned across an output collection");
  auto weak_cache=observedCache();
  std::vector<s::Ref> live;live.reserve(result.outputs.size());
  for(const auto& out:result.outputs)live.push_back(out.value);
  result.nodes_reclaimed+=real_expressions.collect(live, {}, weak_cache);
  for(std::size_t i=0;i<live.size();++i)result.outputs[i].value=std::move(live[i]);
  // No request is pending between outputs (as in collectCompletedOutputs).
  select_contexts.clear();
  select_context_ids.clear();
  retainObserved(weak_cache);
  ++result.construction_collections;
}

inline auto Lower::collectCompletedOutputs(bool force) -> void {
  if(!force && result.graph.size()<next_completed_collection)return;
  // Output compaction uses weak observers and would undo construction
  // cache retention. In the automatic policy it is a pressure operation,
  // not an operation at every newly completed family. Explicit intervals
  // continue to exercise aggressive moving collection in component tests.
  if(!force && options.construction_collection_interval==std::numeric_limits<std::size_t>::max() &&
      result.graph.size()<pressureNodes())return;
#if defined(SENTRYWARD_ENABLE_COMPRESSION_AUDIT)
  auto measurement=ledger?std::make_unique<debug::ProjectionLedger::Collection>(*ledger,result.graph,keys.size()):nullptr;
#endif
  SENTRYWARD_AUDIT(audit->add(AuditCounter::gc_calls);
    audit->add(AuditCounter::gc_input_nodes,result.graph.size());
    audit->event("CACHE_COLLECTION","completed_output_weak_projections",keys.size(),result.outputs.size()));
  // Only collectFrontier remaps pinned partial results (PinnedRoot).
  if(!pinned_roots.empty())throw std::logic_error("a partial result is pinned across an output collection");
  auto weak_cache=observedCache();
  // Lifting mode, a collection under node pressure (not forced): pending outputs are neither
  // lifted nor consolidated; only the construction's unreachable nodes are collected. Lifting
  // replaces a representative's output bodies by its members' bodies, so the projections cached
  // in them (observed, never rooted) would be dropped and rebuilt by every later output that
  // shares them; consolidation takes instantiated outputs. Pending outputs are lifted, and then
  // consolidated, by a forced collection and at the end of the lowering (phase f), as in a
  // lowering that never reaches the pressure point. Exact: lifting and consolidation choose
  // representations, never values.
  const bool deferred=options.unit_lifting && !force && instantiated_end<result.outputs.size();
  // Lifting: consolidation takes instantiated outputs, so instantiate them first. The
  // instantiation adds the members' bodies (or member families) to the graph and its node
  // budget (liftBudget), so the construction's unreachable nodes are collected before it, as
  // the collection below would: the same roots and observed cache entries. O(graph), lifting
  // only, when outputs are pending.
  if(!deferred && options.unit_lifting && instantiated_end<result.outputs.size()) {
    std::vector<s::Ref> live;live.reserve(result.outputs.size());
    for(const auto& out:result.outputs)live.push_back(out.value);
    result.nodes_reclaimed+=real_expressions.collect(live, {}, weak_cache);
    for(std::size_t i=0;i<live.size();++i)result.outputs[i].value=std::move(live[i]);
  }
  if(!deferred)instantiatePending();
  std::vector<s::Ref> roots;
  for(const auto& out:result.outputs)roots.push_back(out.value);
  result.nodes_reclaimed+=real_expressions.collect(roots, {}, weak_cache);
  for(std::size_t i=0;i<roots.size();++i)result.outputs[i].value=roots[i];
  // Lifting mode, deferred: when the pending outputs alone still hold half of the pressure point,
  // the lowering would run into the node budget before they are lifted: the next output's
  // construction can need as much again (one output tile of flat MSA rebuilds the softmax rows of
  // 31 lanes, about 0.8M nodes). They are then merged among themselves, per unit, as a per-unit
  // lowering's consolidation merges its completed outputs (mergeRepresentativeFamilies, pending),
  // and the bodies the merged families replace are collected. They stay pending (parametric);
  // lifting takes the merged families. Exact: a merge chooses a representation, never a value.
  // Only a lowering whose pending outputs hold that much after a collection takes this path.
  // O(graph) for the merge and the collection.
  if(deferred && result.graph.size()>=pressureNodes()/2 && mergeRepresentativeFamilies(nullptr,true)) {
    ++result.pending_output_merges;
    // The merge builds its generalized and shifted bodies beside the ones they replace, as a
    // consolidation does on its artifact graph, so the graph may stand above the node budget until
    // this collection drops the replaced bodies; the budget is the collected graph's (the next
    // construction checks it). The builder's own budget poll is therefore off for this collection.
    // peak_construction_nodes stays the construction's peak, as with a consolidation.
    struct CollectionScope {bool& flag;bool saved;~CollectionScope(){flag=saved;}} scope{collecting,collecting};
    collecting=true;
    const auto peak=result.peak_construction_nodes;
    roots.clear();
    for(const auto& out:result.outputs)roots.push_back(out.value);
    result.nodes_reclaimed+=real_expressions.collect(roots, {}, weak_cache);
    for(std::size_t i=0;i<roots.size();++i)result.outputs[i].value=roots[i];
    result.peak_construction_nodes=std::max(peak,result.graph.size());
  }
  if(!deferred && !output_domains.empty()) {
    // Typed bf16 reads only, and no two-dimensional families (mergeRepresentativeFamilies).
    canonicalizeOutputs(familyOutputs(),s::CanonicalForms::Bf16Fields);
    s::Artifact artifact;artifact.graph=std::move(result.graph);
    for(std::size_t i=0;i<result.outputs.size();++i) {
      const auto& output=result.outputs[i];
      if(auto domain=output_domains.find(i);domain!=output_domains.end()) {
        artifact.output_families.push_back(domain->second.family(output));
      } else artifact.outputs.push_back({output.region.allocation,output.region.offset,output.region.bytes,output.value});
    }
    const auto before=artifact.graph.size();
    result.output_families_merged+=s::consolidateOutputs(artifact,false,weak_cache,false,real_expressions.summaryObserver());
    if(before>artifact.graph.size())result.nodes_reclaimed+=before-artifact.graph.size();
    result.graph=std::move(artifact.graph);result.outputs.clear();output_domains.clear();
    for(const auto& output:artifact.outputs)
      result.outputs.push_back({{output.allocation,output.offset,output.bytes},output.value});
    for(const auto& family:artifact.output_families) {
      OutputDomain domain(family.shape,family.strides_bytes);
      if(!family.element_bytes || domain.count>UINT64_MAX/family.element_bytes)
        fail("scalar_fold_prerequisite","collected output payload overflow");
      const auto bytes=domain.count*family.element_bytes;
      output_domains.emplace(result.outputs.size(),std::move(domain));
      result.outputs.push_back({{family.allocation,family.offset,bytes},family.body});
    }
    // Lifting: every rebuilt output was instantiated before the consolidation;
    // outputs completed from here on start at the new end.
    instantiated_end=result.outputs.size();
  }
  // No request is pending between outputs: retire the selection contexts, whose literals this
  // collection did not keep. Their serials stay unused (next_select_context).
  select_contexts.clear();
  select_context_ids.clear();
  [[maybe_unused]] const auto discarded=retainObserved(weak_cache);
  SENTRYWARD_AUDIT(audit->add(AuditCounter::cache_entries_discarded,discarded);
    audit->event("CACHE_RETENTION","live_projections_after_output_consolidation",keys.size(),discarded));
  ++result.construction_collections;
  scheduleCollection();
  const auto cap=std::max<std::size_t>(1,options.max_nodes/4);
  const auto growth=std::min(cap,std::max(options.construction_collection_interval,result.graph.size()/2));
  next_completed_collection=std::max(result.graph.size()+growth,
      std::min(std::max<std::size_t>(1,options.max_nodes/2),next_completed_collection+growth));
}

} // namespace
} // namespace sentryward::value_analysis
