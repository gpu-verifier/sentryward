#include "sentryward/algorithm/checker/family_memory_checker.h"
#include "sentryward/algorithm/checker/duplicate_tma_stores.h"
#include "sentryward/algorithm/analysis/family_relation_query.h"
#include "sentryward/algorithm/memory/family_conflict_order.h"
#include "sentryward/algorithm/diagnostics/performance_profile.h"
#include <chrono>
#include <map>
#include <numeric>
#include <unordered_map>

namespace sentryward::algorithm {
namespace {
using Coordinate = address_solver::Int;
struct FamilyIntervalIndex {
  struct Interval {std::size_t index;Coordinate lo,hi;};
  struct Group {
    std::vector<Interval> rows;
    std::vector<Coordinate> maximum;
    std::size_t leaves=1;
    void build() {
      std::sort(rows.begin(),rows.end(),[](const auto& a,const auto& b) {
        return std::tie(a.lo,a.index)<std::tie(b.lo,b.index);
      });
      while(leaves<rows.size())leaves*=2;
      maximum.assign(2*leaves,-(Coordinate(1)<<126));
      for(std::size_t i=0;i<rows.size();++i)maximum[leaves+i]=rows[i].hi;
      for(std::size_t i=leaves;i-->1;)maximum[i]=std::max(maximum[2*i],maximum[2*i+1]);
    }
    template<class Visitor> void query(Coordinate lo,Coordinate hi,Visitor&& visit) const {
      const auto walk=[&](auto&& self,std::size_t node,std::size_t begin,std::size_t end)->void {
        if(begin>=rows.size() || rows[begin].lo>=hi || maximum[node]<=lo)return;
        if(end-begin==1) {visit(rows[begin].index);return;}
        const auto middle=begin+(end-begin)/2;
        self(self,2*node,begin,middle);self(self,2*node+1,middle,end);
      };
      walk(walk,1,0,leaves);
    }
  };
  struct Space {
    std::map<std::string,Group> allocations;
    std::map<std::string,std::map<std::string,Group>> aliases;
  };
  std::map<std::string,Space> spaces;
  void add(const FamilyFootprint& footprint,Interval interval) {
    auto& space=spaces[footprint.space];
    space.allocations[footprint.allocation].rows.push_back(interval);
    if(!footprint.alias.empty() && !footprint.allocation.empty()) {
      interval.lo=-(Coordinate(1)<<125);interval.hi=Coordinate(1)<<125;
      space.aliases[footprint.alias][footprint.allocation].rows.push_back(interval);
    }
  }
  void build() {
    for(auto& [name,space]:spaces) {
      for(auto& [allocation,group]:space.allocations)group.build();
      for(auto& [alias,allocations]:space.aliases)
        for(auto& [allocation,group]:allocations)group.build();
    }
  }
  template<class Visitor> void query(const FamilyFootprint& footprint,Coordinate lo,Coordinate hi,
      Visitor&& visit) const {
    const auto all=[&](const Space& space) {
      for(const auto& [name,group]:space.allocations)group.query(lo,hi,visit);
    };
    if(footprint.space.empty()) {
      for(const auto& [name,space]:spaces)all(space);
      return;
    }
    if(const auto unknown=spaces.find("");unknown!=spaces.end())all(unknown->second);
    const auto found=spaces.find(footprint.space);
    if(found==spaces.end())return;
    const auto& space=found->second;
    if(footprint.allocation.empty()) {all(space);return;}
    for(const auto& name:{footprint.allocation,std::string{}})
      if(const auto group=space.allocations.find(name);group!=space.allocations.end())
        group->second.query(lo,hi,visit);
    if(!footprint.alias.empty())
      if(const auto alias=space.aliases.find(footprint.alias);alias!=space.aliases.end())
        for(const auto& [allocation,group]:alias->second)
          if(allocation!=footprint.allocation)group.query(lo,hi,visit);
  }
};
} // namespace
std::optional<Result<MemoryRaceResult,MemoryAnalysisError>> TryFamilyMemoryCheck(
    const SourceProgram& source,const PassCertificate& certificate,MemoryModel& memory,
    const MemoryRaceOptions& options,
    std::optional<std::span<const MemoryAccessFamily>> supplied_families,StaticHbQueries* shared_hb) {
  using Output=Result<MemoryRaceResult,MemoryAnalysisError>;
  using I=address_solver::Int;
  if(options.search_strategy!=MemorySearchStrategy::kAuto || options.exhaustive || options.max_races!=1 ||
      options.continue_after_race || options.reachability_factory ||
      options.reachability.algorithm!=HbAlgorithm::kAuto)return {};
  if(!options.reachability.max_batch_sources || options.reachability.index_budget_bytes<1024*1024)return {};
  ProfileScope profile("memory.family.total");
  const auto start=std::chrono::steady_clock::now();
  const auto& graph=certificate.happensBeforeGraph;
  DuplicateTmaStores duplicate(source);
  std::vector<MemoryAccessFamily> extracted_families;
  if (!supplied_families) {
    auto extracted=ProfileCall("memory.family.extract_accesses",[&]{return ExtractMemoryAccessFamilies(graph,source);});
    if(!extracted)return Output::Failure(extracted.error());
    extracted_families=std::move(extracted).value();
  }
  const auto source_families=supplied_families.value_or(std::span<const MemoryAccessFamily>(extracted_families));
  std::vector<MemoryAccessFamily> families;
  std::vector<FamilyFootprint> footprints;
  std::vector<std::optional<std::size_t>> copy_classes;
  std::vector<std::uint8_t> exempt_families;
  bool declined=false;std::size_t logical_accesses=0;
  ProfileCall("memory.family.accesses",[&] {
    const auto append=[&](MemoryAccessFamily family,FamilyFootprint footprint) {
      const auto safe=[](I value){return value>-(I(1)<<90) && value<(I(1)<<90);};
      if(!safe(footprint.offset) || !safe(footprint.step)) {declined=true;return;}
      const auto emit=[&](MemoryAccessFamily part,std::optional<std::size_t> copy) {
        families.push_back(std::move(part));footprints.push_back(footprint);
        copy_classes.push_back(copy);exempt_families.push_back(copy.has_value());
      };
      if(!options.warn_duplicate_tma_stores || family.first.mode!=AccessMode::kWrite ||
          family.first.operand!=AccessOperand::kEffect) {emit(std::move(family),{});return;}
      // Validate provenance once per eligible member, not once per pair.
      // Split only at actual copy-class/address boundaries, retaining coordinates.
      auto first=family.first_member;
      auto access=family.first;
      auto copy=duplicate.copyClass(access);
      if(!copy && duplicate.mayContain(access)) {declined=true;return;}
      for(auto member=first+1;member<family.first_member+family.count;++member) {
        const auto next=family.at(member);const auto next_copy=duplicate.copyClass(next);
        if(!next_copy && duplicate.mayContain(next)) {declined=true;return;}
        if(next_copy==copy && (!copy || next.address==access.address))continue;
        auto part=family;part.first=access;part.first_member=first;part.count=member-first;
        emit(std::move(part),copy);first=member;access=next;copy=next_copy;
      }
      auto part=family;part.first=access;part.first_member=first;
      part.count=family.first_member+family.count-first;emit(std::move(part),copy);
    };
    for(const auto& family:source_families) {
        logical_accesses+=family.count;
        if(family.first.atomic_scope) {
          declined=true;return;
        }
        auto footprint=memory.familyFootprint(family);
        if(footprint) {append(family,std::move(*footprint));if(declined)return;continue;}
        // Partition a flat address table once; no guessed affine interpolation.
        std::optional<MemoryAccessFamily> run;FamilyFootprint previous;
        for(std::size_t member=family.first_member;member<family.first_member+family.count;++member) {
          auto single=family;single.first=family.at(member);single.first_member=member;single.count=1;
          auto current=memory.familyFootprint(single);
          if(!current) {declined=true;return;}
          current->offset+=current->step*I(member);
          current->step=0;
          if(current->offset<=-(I(1)<<90) || current->offset>=(I(1)<<90)) {declined=true;return;}
          bool compatible=run && previous.space==current->space && previous.allocation==current->allocation &&
              previous.alias==current->alias && previous.symbolic==current->symbolic && previous.bytes==current->bytes &&
              previous.affine==current->affine && previous.empty==current->empty;
          if(compatible && run->count==1) {
            previous.step=current->offset-previous.offset;
            previous.offset-=previous.step*I(run->first_member);
          }
          if(compatible && current->offset==previous.offset+previous.step*I(member)) {++run->count;continue;}
          if(run)append(*run,previous);
          if(declined)return;
          run=single;previous=*current;
        }
        if(run)append(*run,previous);
        if(declined)return;
    }
  });
  if(declined) {profile.pruned=1;return {};}
  ProfileCall("memory.family.input_counts",[&] {
    ProfileScope counts("memory.family.input_size");
    counts.nodes=logical_accesses;counts.edges=source_families.size();
  });
  profile.nodes=families.size();
  MemoryRaceResult result;
  result.search_strategy="conflict-family";result.reachability_algorithm="member-relations";
  if(options.on_search_strategy)options.on_search_strategy(result.search_strategy,false);
  FamilyIntervalIndex candidates_index;
  const auto bounds=[&](std::size_t i) {
    const auto& f=families[i];const auto& p=footprints[i];
    // Nonconstant common terms cannot be compared across distinct expressions.
    if(p.space.empty() || p.allocation.empty() || !p.symbolic.empty() || !p.affine)
      return std::pair<I,I>{-(I(1)<<125),I(1)<<125};
    auto a=p.offset+p.step*I(f.first_member),b=p.offset+p.step*I(f.first_member+f.count-1);
    return std::pair{std::min(a,b),std::max(a,b)+I(p.bytes)};
  };
  ProfileCall("memory.family.candidate_index",[&] {
    for(std::size_t i=0;i<families.size();++i)if(!footprints[i].empty) {
      const auto [lo,hi]=bounds(i);candidates_index.add(footprints[i],{i,lo,hi});
    }
    candidates_index.build();
  });
  StaticHbQueries local_hb(graph,options.reachability);
  auto& hb=(shared_hb?*shared_hb:local_hb).relations();
  const auto initial_hb_stats=hb.stats();
  auto order=ProfileCall("memory.family.conflict_order",[&] {
    auto result=FamilyConflictOrder(families,footprints,graph.traceOrder(),exempt_families);
    ProfileScope counts("memory.family.conflict_order_counts");
    counts.nodes=result.reducedFamilies();counts.edges=result.memberChecks();
    return result;
  });
  std::size_t relation_count=0,row_enumerations=0;
  std::size_t duplicate_rows=0,self_rows=0;
  ProfileScope expansion("memory.family.expansion_sites");
  // Union complete duplicate-write obligations by relation support, never by
  // enumerating its Cartesian product. Materialize AccessIds only for the
  // existing public obligation interface (once per newly connected member).
  std::unordered_map<std::uint32_t,std::uint32_t> parents;
  const auto root=[&](std::uint32_t id) {
    auto [it,inserted]=parents.try_emplace(id,id);
    while(it->second!=id) {
      it->second=parents.at(it->second);id=it->second;it=parents.find(id);
    }
    return id;
  };
  const auto unite=[&](std::uint32_t a,std::uint32_t b) {
    a=root(a);b=root(b);parents[std::max(a,b)]=std::min(a,b);
  };
  struct ConnectedDomain {MemberSet members;std::uint32_t anchor;};
  std::unordered_map<std::size_t,std::vector<ConnectedDomain>> connected;
  const auto connect=[&](std::size_t family,const MemberSet& members,std::uint32_t anchor) {
    auto& domains=connected[family];MemberSet covered(members.extent());
    for(auto& domain:domains) {
      if(!domain.members.intersect(members).empty())unite(anchor,domain.anchor);
    }
    for(const auto& domain:domains)
      if(root(anchor)==root(domain.anchor))covered.unite(domain.members);
    members.subtract(covered).forEach([&](auto member){unite(anchor,families[family].at(member).id.value());});
    for(auto& domain:domains)if(root(anchor)==root(domain.anchor)) {
      domain.members.unite(members);return;
    }
    domains.push_back({members,anchor});
  };
  const auto cardinality=[](const MemberRelation& relation) {
    const auto a=relation.left.count();
    if(relation.diagonal)return a;
    const auto b=relation.right.count();
    if(b && a>SIZE_MAX/b)throw std::overflow_error("family pair count overflow");
    return a*b;
  };
  // Select each self-family pair exactly once. This costs member rows/bitsets,
  // not member pairs; other families are already deduplicated by index.
  const auto unique=[&](const MemberRelation& relation,const MemoryAccessFamily& family) {
    std::vector<MemberRelation> rows;
    if(relation.diagonal) {
      if(__int128(family.delta[0])*relation.shift>0)rows.push_back(relation);
    } else relation.left.forEach([&](auto member) {
      ++self_rows;expansion.edges=self_rows;
      auto right=relation.right.intersect(family.delta[0]>0 ?
          MemberRange(family.extent,member+1,family.extent):MemberRange(family.extent,0,member));
      if(!right.empty())rows.push_back(MemberRelation::Rectangle(MemberRange(family.extent,member,member+1),std::move(right)));
    });
    return rows;
  };
  const auto finish=[&]() -> Output {
    result.duplicate_write_groups_complete=options.collect_duplicate_write_groups && result.search_complete;
    if(options.collect_duplicate_write_groups) {
      std::map<std::uint32_t,std::vector<AccessId>> groups;
      for(const auto& [id,parent]:parents)groups[root(id)].push_back(AccessId(id));
      for(auto& [id,members]:groups) {
        std::sort(members.begin(),members.end());
        result.duplicate_write_groups.push_back(std::move(members));
      }
    }
    profile.nodes=families.size();profile.edges=relation_count;
    profile.pruned=row_enumerations+hb.stats().member_enumerations-initial_hb_stats.member_enumerations;
    result.family_stats=MemoryRaceResult::FamilyStats{logical_accesses,families.size(),relation_count,
        row_enumerations,hb.stats().member_enumerations-initial_hb_stats.member_enumerations,hb.stats().frontier_pieces-initial_hb_stats.frontier_pieces};
    result.reachability_stats.batches=hb.stats().traversals-initial_hb_stats.traversals;
    result.reachability_stats.queries=result.address_candidates;
    result.reachability_stats.edge_visits=hb.stats().relation_visits-initial_hb_stats.relation_visits;
    return Output::Success(std::move(result));
  };
  std::vector<FamilyRelationBatch> batch;
  std::vector<std::vector<std::pair<std::size_t,std::size_t>>> metadata;
  const auto process=[&]() -> std::optional<Output> {
    const auto hb_start=std::chrono::steady_clock::now();
    hb.checkOrderBatch(batch);
    result.reachability_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-hb_start).count();
    for(std::size_t block=0;block<batch.size();++block) {
    const auto& requests=batch[block].requests;const auto& targets=metadata[block];
    for(std::size_t q=0;q<requests.size();++q)for(std::size_t domain=0;domain<requests[q].requested.size();++domain) {
      const auto i=targets[q].first,j=targets[q].second;
      const bool duplicate_candidate=copy_classes[i] && copy_classes[i]==copy_classes[j];
      std::vector<MemberRelation> duplicate_parts;
      std::span<const MemberRelation> parts(&requests[q].requested[domain],1);
      if(duplicate_candidate) {
        duplicate_parts=requests[q].duplicateResidual(domain,duplicate_rows);
        expansion.pruned=duplicate_rows;parts=duplicate_parts;
      }
      for(const auto& relation:parts) {
      const auto witness=duplicate_candidate?relation.witness():requests[q].witness(domain);
      if(!witness)continue;
      const auto& first=families[i];const auto& second=families[j];
      auto a=first.at(witness->first),b=second.at(witness->second);
      if(a.event==b.event)throw std::logic_error("reflexive HB left an identical-event candidate");
      ++result.candidates_examined;++result.overlap_queries;
      const auto overlap=memory.mayOverlap(a,b);
      if(overlap.status==OverlapStatus::kUnknown) {
        result.status=MemoryRaceStatus::kUnresolved;result.unresolved_count=1;
        result.unresolved.push_back({a,b,overlap.reason,overlap.code});return finish();
      }
      if(overlap.status==OverlapStatus::kDisjoint) {
        // Family Unknowns require further partitioning; retain the exact existing path.
        profile.pruned=2;declined=true;return {};
      }
      if(!overlap.evidence)throw std::logic_error("family overlap has no evidence");
      if(copy_classes[i] && copy_classes[i]==copy_classes[j] &&
          (first.count==1 || !first.delta[4]) && (second.count==1 || !second.delta[4]) &&
          duplicate.contains(a,b)) {
        // Same copy class, field and endpoint certify every different-actor
        // pair. A same-actor pair must already be ordered; validate that
        // invariant, rather than using a witness to assume it for the domain.
        const auto actor_base=[](const auto& f) {
          return std::int64_t(f.first.actor.value())-f.delta[3]*std::int64_t(f.first_member);
        };
        const auto abase=actor_base(first),bbase=actor_base(second);
        bool same_actor=false;
        relation.left.forEach([&](auto x) {
          const auto actor=abase+first.delta[3]*std::int64_t(x);
          if(relation.diagonal) {
            same_actor|=actor==bbase+second.delta[3]*(std::int64_t(x)+relation.shift);
          } else if(!second.delta[3])same_actor|=actor==bbase;
          else {
            const auto difference=actor-bbase;
            if(difference%second.delta[3]==0) {
              const auto y=difference/second.delta[3];
              same_actor|=y>=0 && std::uint64_t(y)<second.extent && relation.right.contains(y);
            }
          }
        });
        if(same_actor) {declined=true;return {};}
        const auto pairs=cardinality(relation);
        if(pairs>SIZE_MAX-result.warning_count)throw std::overflow_error("warning count overflow");
        result.warning_count+=pairs;
        if(options.collect_duplicate_write_groups) {
          if(relation.diagonal)relation.left.forEach([&](auto x) {
            unite(first.at(x).id.value(),second.at(std::size_t(std::int64_t(x)+relation.shift)).id.value());
          });
          else {
            connect(i,relation.left,a.id.value());connect(j,relation.right,a.id.value());
          }
        }
        if(result.duplicate_tma_store_warnings.size()<options.max_warning_samples) {
          relation.left.forEach([&](auto x) {
            const auto sample=[&](auto y) {
              if(result.duplicate_tma_store_warnings.size()>=options.max_warning_samples)return;
              const auto left=first.at(x),right=second.at(y);
              if(!duplicate.contains(left,right))throw std::logic_error("invalid duplicate family proof");
              result.duplicate_tma_store_warnings.push_back({left,right,*overlap.evidence});
            };
            if(relation.diagonal)sample(std::size_t(std::int64_t(x)+relation.shift));
            else if(result.duplicate_tma_store_warnings.size()<options.max_warning_samples)relation.right.forEach(sample);
          });
        }
        continue;
      }
      result.status=MemoryRaceStatus::kRaceFound;result.race_count=1;
      result.races.push_back({a,b,*overlap.evidence});
      result.first_race_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
      if(options.on_race)options.on_race(result.races.front());
      return finish();
    }
    }
    }
    batch.clear();metadata.clear();return {};
  };
  // Query ordering is independent of program scheduling. Group source
  // domains by physical mask width before forming batches; alternating small
  // and large domains must not repeatedly flush nearly-empty traversals.
  std::map<std::pair<std::size_t,std::size_t>,std::vector<std::size_t>> write_sources;
  for(std::size_t i=0;i<families.size();++i)
    if(families[i].first.mode==AccessMode::kWrite && !footprints[i].empty) {
      const auto source=families[i].event_family;
      const auto width=MemberSet::storageWords(hb.families().families()[source].count);
      write_sources[{width,source}].push_back(i);
    }
  for(const auto& [domain_source,source_accesses]:write_sources) {
    const auto source_family=domain_source.second;
    std::vector<FamilyRelationRequest> requests;
    std::vector<std::pair<std::size_t,std::size_t>> targets;
    ProfileScope candidates("memory.family.generate_relations");
    for(auto i:source_accesses) {
    const auto& first=families[i];
    if(first.first.mode!=AccessMode::kWrite || footprints[i].empty)continue;
    const auto [lo,hi]=bounds(i);
    const auto* reduced=order.candidates(i);
    const auto candidate=[&](std::size_t j) {
      ++candidates.edges;
      const auto& second=families[j];
      if(second.first.mode==AccessMode::kWrite && j<i)return;
      auto overlap=ClassifyFamilyOverlap(first,footprints[i],second,footprints[j]);
      row_enumerations+=overlap.member_rows;
      expansion.nodes=row_enumerations;
      for(auto& r:overlap.unknown)overlap.overlap.push_back(std::move(r));
      // Boolean race queries need no triangular member-row expansion.
      // Reflexive pairs are discharged by the HB identity proof. Retain the
      // legacy unique-pair contract only for A002 warning counts/groups.
      if(i==j && copy_classes[i]) {
        std::vector<MemberRelation> rows;
        for(const auto& relation:overlap.overlap) {
          auto parts=unique(relation,first);
          rows.insert(rows.end(),std::make_move_iterator(parts.begin()),std::make_move_iterator(parts.end()));
        }
        overlap.overlap=std::move(rows);
      }
      if(overlap.overlap.empty())return;
      for(const auto& relation:overlap.overlap)result.address_candidates+=cardinality(relation);
      relation_count+=overlap.overlap.size();
      profile.edges=relation_count;
      requests.push_back({second.event_family,std::move(overlap.overlap)});targets.push_back({i,j});
    };
    if(reduced)for(auto j:*reduced)candidate(j);
    else candidates_index.query(footprints[i],lo,hi,candidate);
    }
    candidates.nodes=requests.size();candidates.Stop();
    if(requests.empty())continue;
    const auto extent=hb.families().families()[source_family].count;
    // Inline masks of 1..256 members have the same physical cost. Mix their
    // labels; for larger masks retain equal storage widths, so padding cannot
    // turn singleton queries into large dense-array work.
    if(!batch.empty() && (batch.size()==std::min(std::size_t(64),options.reachability.max_batch_sources) ||
        MemberSet::storageWords(extent)!=MemberSet::storageWords(hb.families().families()[batch.front().source].count))) {
      if(auto answer=process())return answer;
      if(declined)return {};
    }
    batch.push_back({source_family,std::move(requests)});metadata.push_back(std::move(targets));
    ProfileSnapshot();
  }
  profile.nodes=families.size();profile.edges=relation_count;profile.pruned=row_enumerations+hb.stats().member_enumerations;
  if(!batch.empty()) {
    if(auto answer=process())return answer;
    if(declined)return {};
  }
  result.search_complete=true;
  return finish();
}
} // namespace sentryward::algorithm
