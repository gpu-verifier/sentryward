#include "sentryward/algorithm/memory/family_conflict_order.h"
#include "sentryward/algorithm/memory/conflict_order.h"
#include <algorithm>
#include <map>
#include <numeric>
#include <set>
#include <tuple>
#include <unordered_map>

namespace sentryward::algorithm {
FamilyConflictOrder::FamilyConflictOrder(const std::vector<MemoryAccessFamily>& families,
    const std::vector<FamilyFootprint>& footprints,const TraceOrder* order,
    std::span<const std::uint8_t> exempt_families)
    :buckets_(families.size(),SIZE_MAX),peers_(families.size()) {
  if(!order)return;
  using I=address_solver::Int;
  using Storage=std::pair<std::string,std::string>;
  using Pattern=std::tuple<I,I,std::size_t,std::uint64_t>;
  struct Bucket { I lo,hi;bool valid=true;std::vector<std::size_t> families; };
  std::map<Storage,std::vector<std::size_t>> spaces;
  std::map<Storage,bool> blocked;
  std::set<std::string> blocked_spaces;
  // Different allocations interact only through a shared alias scope. A scope
  // naming one allocation aliases nothing, so its storage is reduced as usual.
  std::map<Storage,std::set<std::string>> alias_allocations;
  for(const auto& p:footprints)
    if(!p.empty && !p.alias.empty())alias_allocations[{p.space,p.alias}].insert(p.allocation);
  for(std::size_t i=0;i<families.size();++i) {
    const auto& p=footprints[i];if(p.empty)continue;
    const Storage key{p.space,p.allocation};spaces[key].push_back(i);
    const bool aliased=!p.alias.empty() && alias_allocations.at({p.space,p.alias}).size()>1;
    // A tolerated write/write gap cannot certify adjacent conflict orders.
    // Disable reduction only for its storage; other allocations stay reduced.
    // Members of one family must be disjoint: a descending progression is
    // as exact as an ascending one, but self-overlapping members are not.
    // Every allocation of a shared alias scope is blocked by its own footprints.
    if((!exempt_families.empty() && exempt_families[i]) ||
        !p.affine || aliased || !p.symbolic.empty() ||
        (families[i].count>1 && (p.step<0?-p.step:p.step)<I(p.bytes) &&
         !(families[i].first.mode==AccessMode::kRead && p.step==0)))blocked[key]=true;
    // An unbounded footprint may overlap any allocation of its space.
    if(p.allocation.empty() || p.space.empty())blocked_spaces.insert(p.space);
  }
  if(blocked_spaces.contains(""))return;
  std::size_t next_bucket=0;
  for(const auto& [storage,items]:spaces) {
    if(blocked[storage] || blocked_spaces.contains(storage.first))continue;
    if(std::all_of(items.begin(),items.end(),[&](auto i){return families[i].count==1;})) {
      reduceMembers(items,families,footprints,*order,next_bucket++);
      continue;
    }
    std::map<Pattern,std::vector<std::size_t>> patterns;
    for(auto i:items) {
      const auto& f=families[i];const auto& p=footprints[i];
      patterns[{p.offset+p.step*I(f.first_member),f.count==1?I(0):p.step,f.count,p.bytes}].push_back(i);
    }
    std::vector<Bucket> buckets;
    for(auto& [pattern,members]:patterns) {
      // A pattern names its first member's address; a descending progression
      // starts at its highest member. Its envelope spans both ends.
      const auto [start,step,count,bytes]=pattern;
      const auto last=start+step*I(count-1);
      buckets.push_back({std::min(start,last),std::max(start,last)+I(bytes),true,std::move(members)});
    }
    std::sort(buckets.begin(),buckets.end(),[](const auto& a,const auto& b){return a.lo<b.lo;});
    // Envelope order is address order. Mark every envelope involved in overlap.
    I maximum=-(I(1)<<125);std::size_t owner=0;
    for(std::size_t i=0;i<buckets.size();++i) {
      if(buckets[i].lo<maximum) {buckets[i].valid=false;buckets[owner].valid=false;}
      if(buckets[i].hi>maximum) {maximum=buckets[i].hi;owner=i;}
    }
    // A bucket without one family-level order (overlapping envelopes or
    // member-wise rank disagreement) overlaps only other such buckets, never a
    // certified one. Order those families exactly by one member-interval sweep.
    std::vector<std::size_t> sweep;
    for(auto& bucket:buckets) {
      if(!bucket.valid) {sweep.insert(sweep.end(),bucket.families.begin(),bucket.families.end());continue;}
      if(bucket.families.size()<2)continue;
      auto& members=bucket.families;
      std::vector<std::pair<std::size_t,std::size_t>> ranks;
      for(auto i:members)ranks.emplace_back(order->rank(families[i].first.event),i);
      std::sort(ranks.begin(),ranks.end());
      members.clear();for(const auto& item:ranks)members.push_back(item.second);
      // A trace establishes orientation, not HB; every retained constraint is checked.
      for(std::size_t j=1;j<members.size() && bucket.valid;++j) {
        const auto& a=families[members[j-1]];const auto& b=families[members[j]];
        const auto x=order->rankBounds(a.first.event,a.delta[1],a.count);
        const auto y=order->rankBounds(b.first.event,b.delta[1],b.count);
        if(x.second<y.first)continue;
        for(std::size_t k=0;k<a.count;++k) {
          ++member_checks_;
          if(order->rank(a.at(a.first_member+k).event)>=order->rank(b.at(b.first_member+k).event)) {
            bucket.valid=false;break;
          }
        }
      }
      if(!bucket.valid) {sweep.insert(sweep.end(),members.begin(),members.end());continue;}
      const auto id=next_bucket++;
      for(auto i:members)buckets_[i]=id;
      reduced_families_+=members.size();
      std::size_t writer=SIZE_MAX;std::vector<std::size_t> readers;
      const auto emit=[&](auto a,auto b){peers_[a].push_back(b);peers_[b].push_back(a);};
      for(auto i:members) {
        if(families[i].first.mode==AccessMode::kRead) {
          if(writer!=SIZE_MAX)emit(writer,i);
          readers.push_back(i);
        } else {
          if(readers.empty() && writer!=SIZE_MAX)emit(writer,i);
          for(auto reader:readers)emit(reader,i);
          readers.clear();writer=i;
        }
      }
    }
    if(!sweep.empty())reduceMembers(sweep,families,footprints,*order,next_bucket++);
  }
}
void FamilyConflictOrder::reduceMembers(const std::vector<std::size_t>& items,
    const std::vector<MemoryAccessFamily>& families,const std::vector<FamilyFootprint>& footprints,
    const TraceOrder& order,std::size_t bucket) {
  using I=address_solver::Int;
  // Reuse the exact interval sweep over member accesses: O(N log N + H) for N
  // members and H access/slab incidences, instead of a relation per pair of
  // overlapping families. Operands of one event share a rank slot, so they are
  // never obligated against each other. Families left unreduced keep the
  // exact interval-index query: without a write they have no conflict here,
  // and beyond the member budget the sweep would not pay for itself.
  constexpr std::size_t kMemberBudget=std::size_t(1)<<26;
  std::size_t members=0;bool writes=false;
  for(auto i:items){members+=families[i].count;writes|=families[i].first.mode==AccessMode::kWrite;}
  if(!writes || members>kMemberBudget)return;
  std::vector<ConstantConflictRegion> regions;std::vector<std::size_t> owners,ranks;
  regions.reserve(members);owners.reserve(members);
  std::unordered_map<std::uint32_t,std::size_t> slots;
  for(auto i:items) {
    const auto& f=families[i];const auto& p=footprints[i];
    for(std::size_t k=0;k<f.count;++k) {
      const auto offset=p.offset+p.step*I(f.first_member+k);
      const auto event=std::int64_t(f.first.event.value())+f.delta[1]*std::int64_t(k);
      if(offset<0 || offset>I(UINT64_MAX) || event<0 || event>std::int64_t(UINT32_MAX))return;
      const auto [slot,added]=slots.try_emplace(std::uint32_t(event),ranks.size());
      if(added)ranks.push_back(order.rank(EventId(std::uint32_t(event))));
      regions.push_back({AccessId(std::uint32_t(regions.size())),EventId(std::uint32_t(slot->second)),
          f.first.mode==AccessMode::kWrite,0,std::uint64_t(offset),p.bytes});
      owners.push_back(i);
    }
  }
  auto stream=MakeConflictOrder(std::move(regions),ranks);
  while(auto pair=stream->next()) {
    const auto a=owners[pair->first.value()],b=owners[pair->second.value()];
    peers_[a].push_back(b);peers_[b].push_back(a);
  }
  for(auto i:items) {
    buckets_[i]=bucket;
    auto& peers=peers_[i];std::sort(peers.begin(),peers.end());
    peers.erase(std::unique(peers.begin(),peers.end()),peers.end());
  }
  reduced_families_+=items.size();
}
const std::vector<std::size_t>* FamilyConflictOrder::candidates(std::size_t family) const {
  return buckets_[family]==SIZE_MAX?nullptr:&peers_[family];
}
} // namespace sentryward::algorithm
