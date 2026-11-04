#include "sentryward/algorithm/checker/family_memory_checker.h"
#include "sentryward/algorithm/checker/duplicate_tma_stores.h"
#include "sentryward/algorithm/checker/same_value_writes.h"
#include "sentryward/algorithm/checker/unit_quotient.h"
#include "sentryward/algorithm/analysis/family_relation_query.h"
#include "sentryward/algorithm/memory/family_conflict_order.h"
#include "sentryward/algorithm/diagnostics/performance_profile.h"
#include <chrono>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <string>
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
// Union complete write-obligation groups (duplicate TMA stores, same-value writes) by relation
// support, never by enumerating a relation's Cartesian product. AccessIds are materialized only
// for the public obligation interface, once per newly connected member.
struct ObligationGroups {
  WriteObligationGroups accesses;
  struct ConnectedDomain {MemberSet members;AccessId anchor;};
  std::unordered_map<std::size_t,std::vector<ConnectedDomain>> connected;
  // `members` of `family` join the group of `anchor`, each paired with `partner`.
  void connect(const std::vector<MemoryAccessFamily>& families,std::size_t family,const MemberSet& members,
      AccessId anchor,AccessId partner) {
    auto& domains=connected[family];MemberSet covered(members.extent());
    for(auto& domain:domains) {
      if(!domain.members.intersect(members).empty())accesses.unite(anchor,domain.anchor);
    }
    for(const auto& domain:domains)
      if(accesses.connected(anchor,domain.anchor))covered.unite(domain.members);
    members.subtract(covered).forEach([&](auto member){accesses.join(families[family].at(member).id,partner,anchor);});
    for(auto& domain:domains)if(accesses.connected(anchor,domain.anchor)) {
      domain.members.unite(members);return;
    }
    domains.push_back({members,anchor});
  }
  // Every pair of `relation` between families i and j; `anchor` is one of its accesses. Each
  // member of a rectangle is paired with the other side's first member.
  void add(const std::vector<MemoryAccessFamily>& families,std::size_t i,std::size_t j,
      const MemberRelation& relation,AccessId anchor) {
    if(relation.diagonal)relation.left.forEach([&](auto x) {
      accesses.pair(families[i].at(x).id,families[j].at(std::size_t(std::int64_t(x)+relation.shift)).id);
    });
    else {
      const auto left=families[i].at(relation.left.first()).id,right=families[j].at(relation.right.first()).id;
      connect(families,i,relation.left,anchor,right);connect(families,j,relation.right,anchor,left);
    }
  }
};
} // namespace
void WriteObligationGroups::setPartner(AccessId access,AccessId partner) {
  auto& entry=entries_.try_emplace(access.value(),Entry{access.value()}).first->second;
  if(!entry.paired) {entry.partner=partner.value();entry.paired=true;}
}
// A root is the least id of its group (unite links the larger root below the smaller).
std::uint32_t WriteObligationGroups::root(std::uint32_t id) {
  const auto found=entries_.find(id);
  if(found==entries_.end())return id;
  auto* entry=&found->second;
  while(entry->parent!=id) {
    entry->parent=entries_.at(entry->parent).parent;
    id=entry->parent;entry=&entries_.at(id);
  }
  return id;
}
void WriteObligationGroups::unite(AccessId first,AccessId second) {
  for(const auto id:{first.value(),second.value()})entries_.try_emplace(id,Entry{id});
  const auto a=root(first.value()),b=root(second.value());
  entries_.at(std::max(a,b)).parent=std::min(a,b);
}
void WriteObligationGroups::join(AccessId access,AccessId partner,AccessId anchor) {
  setPartner(access,partner);unite(access,anchor);
}
void WriteObligationGroups::pair(AccessId first,AccessId second) {
  setPartner(first,second);setPartner(second,first);unite(first,second);
}
bool WriteObligationGroups::connected(AccessId first,AccessId second) {
  return root(first.value())==root(second.value());
}
std::vector<std::vector<AccessId>> WriteObligationGroups::groups() {
  std::map<std::uint32_t,std::vector<AccessId>> by_root;
  for(const auto& [id,entry]:entries_)by_root[root(id)].push_back(AccessId(id));
  std::vector<std::vector<AccessId>> result;result.reserve(by_root.size());
  for(auto& [id,members]:by_root) {
    std::sort(members.begin(),members.end());result.push_back(std::move(members));
  }
  return result;
}
std::vector<std::pair<AccessId,AccessId>> WriteObligationGroups::partners() const {
  std::vector<std::pair<AccessId,AccessId>> result;
  for(const auto& [id,entry]:entries_)if(entry.paired)result.emplace_back(AccessId(id),AccessId(entry.partner));
  std::sort(result.begin(),result.end());
  return result;
}
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
  // Class mode: a duplicate pair lies in one CTA and stands for that pair in every unit of the
  // CTA's class.
  std::optional<ActorWeights> weights;
  if(options.units)weights.emplace(*options.units,source.actors);
  const ActorCtas ctas(source.actors);
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
  // Families whose storage FamilyConflictOrder leaves unreduced: duplicate-store exemptions.
  std::vector<std::uint8_t> blocked_families;
  bool declined=false;std::size_t logical_accesses=0;
  ProfileCall("memory.family.accesses",[&] {
    const auto append=[&](MemoryAccessFamily family,FamilyFootprint footprint) {
      const auto safe=[](I value){return value>-(I(1)<<90) && value<(I(1)<<90);};
      if(!safe(footprint.offset) || !safe(footprint.step)) {declined=true;return;}
      const auto emit=[&](MemoryAccessFamily part,std::optional<std::size_t> copy) {
        families.push_back(std::move(part));footprints.push_back(footprint);
        copy_classes.push_back(copy);blocked_families.push_back(copy.has_value());
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
    // Reads with a member that has no footprint: dropped below when no write can reach their storage.
    std::vector<const MemoryAccessFamily*> unplaced_reads;
    for(const auto& family:source_families) {
        logical_accesses+=family.count;
        if(family.first.atomic_scope) {
          declined=true;return;
        }
        auto footprint=memory.familyFootprint(family);
        if(footprint) {append(family,std::move(*footprint));if(declined)return;continue;}
        // Partition a flat address table once; no guessed affine interpolation.
        const auto mark=families.size();
        std::optional<MemoryAccessFamily> run;FamilyFootprint previous;
        for(std::size_t member=family.first_member;member<family.first_member+family.count;++member) {
          auto single=family;single.first=family.at(member);single.first_member=member;single.count=1;
          auto current=memory.familyFootprint(single);
          if(!current && family.first.mode==AccessMode::kRead) {
            families.resize(mark);footprints.resize(mark);copy_classes.resize(mark);blocked_families.resize(mark);
            unplaced_reads.push_back(&family);run.reset();break;
          }
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
    // A race needs a write. Every write family has footprints (else declined above); a read is
    // disjoint from all of them when its storage is known and each nonempty write is in another
    // known space, or on another allocation of its space without a shared alias scope (the address
    // models' allocation separation). Such reads are dropped; any other unplaced read declines.
    // O(F log F) for F families, plus each unplaced read's members once.
    if(unplaced_reads.empty())return;
    std::set<std::pair<std::string,std::string>> written,written_aliases; // (space, allocation / alias)
    for(std::size_t k=0;k<families.size();++k) {
      const auto& f=footprints[k];
      if(families[k].first.mode==AccessMode::kRead || f.empty)continue;
      if(!KnownSpace(f.space)) {declined=true;return;}
      written.emplace(f.space,f.allocation);
      if(!f.alias.empty())written_aliases.emplace(f.space,f.alias);
    }
    for(const auto* read:unplaced_reads) {
      const auto storage=memory.familyStorage(*read);
      if(!storage || !KnownSpace(storage->space) || written.contains({storage->space,storage->allocation}) ||
          (!storage->alias.empty() && written_aliases.contains({storage->space,storage->alias}))) {declined=true;return;}
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
    return FootprintSpan(f,p);
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
  // Same-value write obligations (same_value_writes.h). Whether every member of a family is an
  // eligible write is decided once per family, only for families of a candidate write pair.
  SameValueWrites same_value(source,&ctas,weights?&*weights:nullptr,options.max_warning_samples);
  std::vector<std::uint8_t> plain_writes(families.size()); // 0 unknown, 1 eligible, 2 not
  const auto same_value_candidate=[&](std::size_t i,std::size_t j) {
    const auto& x=footprints[i];const auto& y=footprints[j];
    if(families[i].first.mode!=AccessMode::kWrite || families[j].first.mode!=AccessMode::kWrite ||
        !x.affine || !y.affine || !x.symbolic.empty() || !y.symbolic.empty() || x.bytes!=y.bytes ||
        x.space!=y.space || x.allocation!=y.allocation)return false;
    for(auto k:{i,j})if(!plain_writes[k])plain_writes[k]=same_value.eligible(families[k])?1:2;
    return plain_writes[i]==1 && plain_writes[j]==1;
  };
  // A pair of the relation whose ranges start at different bytes, a partial overlap (every pair
  // of a relation overlaps and the candidate test fixed equal widths), or none when every pair
  // covers the same bytes. Starts are affine in the members, so the corner pairs decide.
  const auto partial_pair=[&](std::size_t i,std::size_t j,const MemberRelation& relation)
      -> std::optional<std::pair<std::size_t,std::size_t>> {
    const auto start=[&](std::size_t k,std::size_t member){return footprints[k].offset+footprints[k].step*I(member);};
    const auto last=[](const MemberSet& set){std::size_t end=0;set.forEachRun([&](auto,auto stop){end=stop;});return end-1;};
    const auto l0=relation.left.first(),l1=last(relation.left);
    std::vector<std::pair<std::size_t,std::size_t>> corners;
    if(relation.diagonal)corners={{l0,std::size_t(std::int64_t(l0)+relation.shift)},{l1,std::size_t(std::int64_t(l1)+relation.shift)}};
    else {
      const auto r0=relation.right.first(),r1=last(relation.right);
      corners={{l0,r0},{l0,r1},{l1,r0},{l1,r1}};
    }
    for(const auto& [p,q]:corners)if(start(i,p)!=start(j,q))return std::pair{p,q};
    return std::nullopt;
  };
  auto order=ProfileCall("memory.family.conflict_order",[&] {
    auto result=FamilyConflictOrder(families,footprints,graph.traceOrder(),blocked_families);
    ProfileScope counts("memory.family.conflict_order_counts");
    counts.nodes=result.reducedFamilies();counts.edges=result.memberChecks();
    return result;
  });
  std::size_t relation_count=0,row_enumerations=0;
  std::size_t obligation_rows=0,self_rows=0;
  ProfileScope expansion("memory.family.expansion_sites");
  ObligationGroups duplicates,same_values;
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
  // An accepted obligation certifies every different-actor pair of its relation. A same-actor
  // pair must already be ordered; validate that invariant, rather than using a witness to
  // assume it for the domain.
  const auto same_actor=[](const MemberRelation& relation,const MemoryAccessFamily& first,
                           const MemoryAccessFamily& second) {
    const auto step=second.delta[MemoryAccessFamily::kActorField];
    bool found=false;
    relation.left.forEach([&](auto x) {
      const auto actor=first.actor(x);
      if(relation.diagonal)found|=actor==second.actor(std::size_t(std::int64_t(x)+relation.shift));
      else if(!step)found|=actor==second.first.actor;
      else {
        // The member of `second` with this actor, if any (its actor is affine in the member).
        const auto difference=std::int64_t(actor.value())-std::int64_t(second.first.actor.value());
        if(difference%step==0) {
          const auto y=std::int64_t(second.first_member)+difference/step;
          found|=y>=0 && std::uint64_t(y)<second.extent && relation.right.contains(std::size_t(y));
        }
      }
    });
    return found;
  };
  const auto finish=[&]() -> Output {
    result.duplicate_write_groups_complete=options.collect_duplicate_write_groups && result.search_complete;
    if(options.collect_duplicate_write_groups)result.duplicate_write_groups=duplicates.accesses.groups();
    result.same_value_write_groups=same_values.accesses.groups();
    result.same_value_write_partners=same_values.accesses.partners();
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
      const bool same_candidate=!duplicate_candidate && same_value_candidate(i,j);
      // Obligations take the complete unordered residual, not one witness.
      bool residual=duplicate_candidate;
      std::vector<MemberRelation> residual_parts;
      std::span<const MemberRelation> parts(&requests[q].requested[domain],1);
      if(duplicate_candidate) {
        residual_parts=requests[q].obligationResidual(domain,obligation_rows);
        expansion.pruned=obligation_rows;parts=residual_parts;
      } else if(same_candidate && requests[q].witness(domain)) {
        residual_parts=requests[q].obligationResidual(domain,obligation_rows);
        expansion.pruned=obligation_rows;parts=residual_parts;residual=true;
      }
      for(const auto& relation:parts) {
      const auto witness=residual?relation.witness():requests[q].witness(domain);
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
        // Same copy class, field and endpoint certify every different-actor pair.
        if(same_actor(relation,first,second)) {declined=true;return {};}
        auto pairs=cardinality(relation);
        if(weights) {
          // One weight when the first family lies in one CTA (its actors are affine in the member);
          // otherwise per row, as the same-actor check above.
          const auto& w=*weights;
          const auto row=relation.diagonal?std::size_t(1):relation.right.count();
          const auto front=first.first.actor,back=first.actor(first.first_member+first.count-1);
          std::size_t weighted=0;
          const auto add=[&](std::uint64_t weight,std::size_t count) {
            if(count && weight>SIZE_MAX/count)throw std::overflow_error("warning count overflow");
            if(weight*count>SIZE_MAX-weighted)throw std::overflow_error("warning count overflow");
            weighted+=weight*count;
          };
          if(ctas(front)==ctas(back))add(w(front),pairs);
          else relation.left.forEach([&](auto x){add(w(first.actor(x)),row);});
          pairs=weighted;
        }
        if(pairs>SIZE_MAX-result.warning_count)throw std::overflow_error("warning count overflow");
        result.warning_count+=pairs;
        if(options.collect_duplicate_write_groups)duplicates.add(families,i,j,relation,a.id);
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
      auto evidence=*overlap.evidence;
      if(residual && same_candidate) {
        if(const auto partial=partial_pair(i,j,relation)) {
          // A mixed-size race: report that pair rather than the witness.
          a=first.at(partial->first);b=second.at(partial->second);
          ++result.overlap_queries;
          auto exact=memory.mayOverlap(a,b);
          if(exact.status==OverlapStatus::kUnknown) {
            result.status=MemoryRaceStatus::kUnresolved;result.unresolved_count=1;
            result.unresolved.push_back({a,b,exact.reason,exact.code});return finish();
          }
          if(exact.status==OverlapStatus::kDisjoint || !exact.evidence) {profile.pruned=2;declined=true;return {};}
          evidence=std::move(*exact.evidence);
        } else if(same_value.pair(a,b,evidence)) {
          if(same_actor(relation,first,second)) {declined=true;return {};}
          same_values.add(families,i,j,relation,a.id);
          // An unordered pair of reduced families: their constraints no longer certify the pairs
          // around it, so the order requires further pairs (FamilyConflictOrder::extension).
          if((order.candidates(i) || order.candidates(j)) && !order.addObligation(i,j)) {declined=true;return {};}
          // Every pair of the relation covers the same bytes, at the first family's member offset:
          // a further sample's overlap is the witness's, its region moved with that member.
          const auto moved=[&](std::size_t x,std::size_t) {
            auto shifted=evidence;
            if(shifted.constant_region) {
              const auto offset=I(shifted.constant_region->offset)+footprints[i].step*(I(x)-I(witness->first));
              if(offset<0 || offset>I(UINT64_MAX))shifted.constant_region.reset();
              else shifted.constant_region->offset=std::uint64_t(offset);
            }
            return shifted;
          };
          // Unknown CTA ranges (ActorCtas::contiguous): the access path counts pair by pair.
          if(!same_value.sampleRelation(first,second,relation,moved,result)) {declined=true;return {};}
          continue;
        }
      }
      result.status=MemoryRaceStatus::kRaceFound;result.race_count=1;
      result.races.push_back({a,b,evidence});
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
  using Sources=std::map<std::pair<std::size_t,std::size_t>,std::vector<std::size_t>>;
  const auto add_source=[&](Sources& sources,std::size_t i) {
    const auto source=families[i].event_family;
    const auto width=MemberSet::storageWords(hb.families().families()[source].count);
    sources[{width,source}].push_back(i);
  };
  // One relation search: each write family i of `sources` against every family that
  // `candidates_of(i, visit)` visits; a write/write pair once, from its lesser family.
  const auto search=[&](const Sources& sources,const auto& candidates_of) -> std::optional<Output> {
    for(const auto& [domain_source,source_accesses]:sources) {
      const auto source_family=domain_source.second;
      std::vector<FamilyRelationRequest> requests;
      std::vector<std::pair<std::size_t,std::size_t>> targets;
      ProfileScope candidates("memory.family.generate_relations");
      for(auto i:source_accesses) {
        const auto& first=families[i];
        if(first.first.mode!=AccessMode::kWrite || footprints[i].empty)continue;
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
        candidates_of(i,candidate);
      }
      candidates.nodes=requests.size();candidates.Stop();
      if(requests.empty())continue;
      const auto extent=hb.families().families()[source_family].count;
      // Inline masks of 1..256 members have the same physical cost. Mix their
      // labels; for larger masks retain equal storage widths, so padding cannot
      // turn singleton queries into large dense-array work.
      if(!batch.empty() && (batch.size()==std::min(kMaxRelationBatchSources,options.reachability.max_batch_sources) ||
          MemberSet::storageWords(extent)!=MemberSet::storageWords(hb.families().families()[batch.front().source].count))) {
        if(auto answer=process())return answer;
        if(declined)return {};
      }
      batch.push_back({source_family,std::move(requests)});metadata.push_back(std::move(targets));
      ProfileSnapshot();
    }
    profile.nodes=families.size();profile.edges=relation_count;profile.pruned=row_enumerations+hb.stats().member_enumerations;
    if(!batch.empty())return process();
    return {};
  };
  Sources write_sources;
  for(std::size_t i=0;i<families.size();++i)
    if(families[i].first.mode==AccessMode::kWrite && !footprints[i].empty)add_source(write_sources,i);
  if(auto answer=search(write_sources,[&](std::size_t i,const auto& visit) {
        if(const auto* reduced=order.candidates(i))for(const auto j:*reduced)visit(j);
        else {const auto [lo,hi]=bounds(i);candidates_index.query(footprints[i],lo,hi,visit);}
      }))return answer;
  if(declined)return {};
  // The pairs that the obligations of reduced families require beyond their constraints; a pass
  // over them may record further obligations, so repeat until none is new.
  for(;;) {
    const auto extension=order.extension([&](std::size_t family,const std::function<void(std::size_t)>& visit) {
      const auto [lo,hi]=bounds(family);candidates_index.query(footprints[family],lo,hi,visit);
    });
    if(extension.empty())break;
    Sources sources;
    for(const auto& entry:extension)
      if(families[entry.first].first.mode==AccessMode::kWrite)add_source(sources,entry.first);
    if(auto answer=search(sources,[&](std::size_t i,const auto& visit) {
          const auto found=std::lower_bound(extension.begin(),extension.end(),i,
              [](const auto& entry,std::size_t family){return entry.first<family;});
          for(const auto j:found->second)visit(j);
        }))return answer;
    if(declined)return {};
  }
  result.search_complete=true;
  return finish();
}
} // namespace sentryward::algorithm
