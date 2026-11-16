#include "sentryward/algorithm/memory/family_conflict_order.h"
#include "sentryward/algorithm/memory/conflict_order.h"
#include <algorithm>
#include <map>
#include <numeric>
#include <set>
#include <string_view>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace sentryward::algorithm {
FamilyConflictOrder::FamilyConflictOrder(const FamilyTable& families,
    const std::vector<FamilyFootprint>& footprints,const TraceOrder* order,
    std::span<const std::uint8_t> blocked_families)
    :buckets_(families.size(),SIZE_MAX),peers_(families.size()),shares_event_(families.size()) {
  if(!order)return;
  using I=address_solver::Int;
  // Names are views of `footprints`, which outlive the construction.
  using Name=std::pair<std::string_view,std::string_view>;
  using Pattern=std::tuple<I,I,std::size_t,std::uint64_t>;
  struct Bucket { I lo,hi;bool valid=true;std::size_t begin,end; };
  // Different allocations interact only through a shared alias scope. A scope
  // naming one allocation aliases nothing, so its storage is reduced as usual.
  std::map<Name,std::set<std::string_view>> alias_allocations;
  for(const auto& p:footprints)
    if(!p.empty && !p.alias.empty())alias_allocations[{p.space,p.alias}].insert(p.allocation);
  // Each storage (space, allocation) by a compact id, and the storage of each placed family:
  // one name lookup per family (O(F log S) for F families on S storages), no per-family lists.
  constexpr auto kUnplaced=UINT32_MAX;
  std::map<Name,std::uint32_t> storage_ids;
  std::vector<std::uint32_t> storage_of(families.size(),kUnplaced);
  std::vector<std::uint8_t> blocked;
  std::set<std::string_view> blocked_spaces;
  for(std::size_t i=0;i<families.size();++i) {
    const auto& p=footprints[i];if(p.empty)continue;
    const auto [entry,added]=storage_ids.try_emplace(Name{p.space,p.allocation},std::uint32_t(blocked.size()));
    if(added)blocked.push_back(0);
    const auto storage=storage_of[i]=entry->second;
    const bool aliased=!p.alias.empty() && alias_allocations.at({p.space,p.alias}).size()>1;
    // A tolerated write/write gap (a blocked family) cannot certify adjacent conflict orders.
    // Disable reduction only for its storage; other allocations stay reduced.
    // Members of one family must be disjoint: a descending progression is
    // as exact as an ascending one, but self-overlapping members are not.
    // Every allocation of a shared alias scope is blocked by its own footprints.
    if((!blocked_families.empty() && blocked_families[i]) ||
        !p.affine || aliased || !p.symbolic.empty() ||
        (families[i].count>1 && (p.step<0?-p.step:p.step)<I(p.bytes) &&
         !(families[i].first.mode==AccessMode::kRead && p.step==0)))blocked[storage]=1;
    // An unbounded footprint may overlap any allocation of its space.
    if(p.allocation.empty() || p.space.empty())blocked_spaces.insert(p.space);
  }
  if(blocked_spaces.contains(""))return;
  // The families of each storage in family order, storages in name order: one counting sort into
  // `items` (storage s holds items[start[s], start[s + 1])). O(F + S).
  std::vector<std::uint32_t> position(storage_ids.size());
  std::vector<Name> names;names.reserve(storage_ids.size());
  for(const auto& [name,id]:storage_ids) {position[id]=std::uint32_t(names.size());names.push_back(name);}
  std::vector<std::size_t> start(names.size()+1,0),items;
  for(const auto storage:storage_of)if(storage!=kUnplaced)++start[position[storage]+1];
  for(std::size_t s=0;s<names.size();++s)start[s+1]+=start[s];
  items.resize(start.back());
  {
    auto next=start;
    for(std::size_t i=0;i<families.size();++i)
      if(storage_of[i]!=kUnplaced)items[next[position[storage_of[i]]]++]=i;
  }
  std::vector<std::uint32_t>().swap(storage_of);
  std::size_t next_bucket=0;
  std::vector<Bucket> buckets;
  std::vector<std::pair<std::size_t,std::size_t>> ranks;
  for(std::size_t s=0;s<names.size();++s) {
    if(blocked[storage_ids.at(names[s])] || blocked_spaces.contains(names[s].first))continue;
    const std::span<std::size_t> storage(items.data()+start[s],start[s+1]-start[s]);
    if(std::all_of(storage.begin(),storage.end(),[&](auto i){return families[i].count==1;})) {
      reduceMembers(storage,families,footprints,*order,next_bucket++);
      continue;
    }
    // The families of each pattern (first member's address, stride, count, width), patterns in
    // order and families in family order: one sort of (pattern, family) keys, O(n log n) for the
    // storage's n families, then each bucket is a range of `storage`; no list per pattern.
    std::vector<std::pair<Pattern,std::size_t>> keyed;keyed.reserve(storage.size());
    for(const auto i:storage) {
      const auto& f=families[i];const auto& p=footprints[i];
      keyed.push_back({{p.offset+p.step*I(f.first_member),f.count==1?I(0):p.step,f.count,p.bytes},i});
    }
    std::sort(keyed.begin(),keyed.end());
    buckets.clear();
    for(std::size_t begin=0,end=0;begin<keyed.size();begin=end) {
      const auto& key=keyed[begin].first;
      for(end=begin+1;end<keyed.size() && keyed[end].first==key;)++end;
      // A pattern names its first member's address; a descending progression
      // starts at its highest member. Its envelope spans both ends.
      const auto [first,step,count,bytes]=key;
      const auto last=first+step*I(count-1);
      buckets.push_back({std::min(first,last),std::max(first,last)+I(bytes),true,begin,end});
    }
    for(std::size_t k=0;k<keyed.size();++k)storage[k]=keyed[k].second;
    std::vector<std::pair<Pattern,std::size_t>>().swap(keyed);
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
      const auto members=storage.subspan(bucket.begin,bucket.end-bucket.begin);
      if(!bucket.valid) {sweep.insert(sweep.end(),members.begin(),members.end());continue;}
      if(members.size()<2)continue;
      ranks.clear();
      for(auto i:members)ranks.emplace_back(order->rank(families[i].first.event),i);
      std::sort(ranks.begin(),ranks.end());
      for(std::size_t k=0;k<members.size();++k)members[k]=ranks[k].second;
      // A trace establishes orientation, not HB; every retained constraint is checked.
      for(std::size_t j=1;j<members.size() && bucket.valid;++j) {
        const auto& a=families[members[j-1]];const auto& b=families[members[j]];
        const auto x=order->rankBounds(a.first.event,a.delta[1],a.count);
        const auto y=order->rankBounds(b.first.event,b.delta[1],b.count);
        if(x.second<y.first)continue;
        for(std::size_t k=0;k<a.count;++k) {
          ++member_checks_;
          if(order->rank(a.event(a.first_member+k))>=order->rank(b.event(b.first_member+k))) {
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
void FamilyConflictOrder::reduceMembers(std::span<const std::size_t> items,
    const FamilyTable& families,const std::vector<FamilyFootprint>& footprints,
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
  std::vector<std::size_t> slot_families; // the first family of each slot
  for(auto i:items) {
    const auto& f=families[i];const auto& p=footprints[i];
    for(std::size_t k=0;k<f.count;++k) {
      const auto offset=p.offset+p.step*I(f.first_member+k);
      const auto event=std::int64_t(f.first.event.value())+f.delta[1]*std::int64_t(k);
      if(offset<0 || offset>I(UINT64_MAX) || event<0 || event>std::int64_t(UINT32_MAX))return;
      const auto [slot,added]=slots.try_emplace(std::uint32_t(event),ranks.size());
      if(added) {
        ranks.push_back(order.rank(EventId(std::uint32_t(event))));
        slot_families.push_back(i);
      } else if(slot_families[slot->second]!=i)
        shares_event_[i]=shares_event_[slot_families[slot->second]]=1;
      regions.push_back({AccessId(std::uint32_t(regions.size())),EventId(std::uint32_t(slot->second)),
          f.first.mode==AccessMode::kWrite,0,std::uint64_t(offset),p.bytes});
      owners.push_back(i);
    }
  }
  auto stream=MakeConflictOrder(std::move(regions),ranks);
  // The stream repeats a family pair once per member and slab; a repeat of a family's last partner
  // is not stored (the sort below removes every other repeat), and each list keeps no spare capacity.
  const auto add=[&](std::size_t a,std::size_t b) {
    auto& peers=peers_[a];
    if(peers.empty() || peers.back()!=b)peers.push_back(b);
  };
  while(auto pair=stream->next()) {
    const auto a=owners[pair->first.value()],b=owners[pair->second.value()];
    add(a,b);add(b,a);
  }
  for(auto i:items) {
    buckets_[i]=bucket;
    auto& peers=peers_[i];std::sort(peers.begin(),peers.end());
    peers.erase(std::unique(peers.begin(),peers.end()),peers.end());
    peers.shrink_to_fit();
  }
  reduced_families_+=items.size();
}
const std::vector<std::size_t>* FamilyConflictOrder::candidates(std::size_t family) const {
  return buckets_[family]==SIZE_MAX?nullptr:&peers_[family];
}
std::size_t FamilyConflictOrder::group(std::size_t family) {
  if(parents_.empty())return family;
  while(parents_[family]!=family) {
    parents_[family]=parents_[parents_[family]];
    family=parents_[family];
  }
  return family;
}
bool FamilyConflictOrder::addObligation(std::size_t a,std::size_t b) {
  // Constraints join only families of one reduced bucket, and its families overlap no others.
  if(buckets_[a]==SIZE_MAX || buckets_[a]!=buckets_[b])
    throw std::logic_error("a write obligation joins families outside one reduced bucket");
  if(shares_event_[a] || shares_event_[b])return false;
  if(parents_.empty()) {
    parents_.resize(buckets_.size());
    std::iota(parents_.begin(),parents_.end(),std::size_t(0));
  }
  a=group(a);b=group(b);
  if(a==b)return true;
  const auto members=[&](std::size_t root)->std::vector<std::size_t>& {
    auto& list=groups_[root];
    if(list.empty())list.push_back(root);
    return list;
  };
  // Union by size: each family moves O(log families) times.
  auto* x=&members(a);auto* y=&members(b);
  if(x->size()<y->size()) {std::swap(a,b);std::swap(x,y);}
  x->insert(x->end(),y->begin(),y->end());
  groups_.erase(b);parents_[b]=a;
  joined_.push_back(a);
  return true;
}
std::vector<std::pair<std::size_t,std::vector<std::size_t>>> FamilyConflictOrder::extension(
    const Overlapping& overlapping) {
  std::vector<std::size_t> roots;roots.reserve(joined_.size());
  for(const auto root:joined_)roots.push_back(group(root));
  joined_.clear();
  std::sort(roots.begin(),roots.end());roots.erase(std::unique(roots.begin(),roots.end()),roots.end());
  std::map<std::size_t,std::vector<std::size_t>> added;
  for(const auto root:roots) {
    const auto& members=groups_.at(root);
    std::unordered_set<std::size_t> linked{root};
    for(const auto f:members)for(const auto peer:peers_[f])linked.insert(group(peer));
    for(const auto f:members) {
      auto constraints=peers_[f];std::sort(constraints.begin(),constraints.end());
      overlapping(f,[&](std::size_t g) {
        if(g==f || !linked.contains(group(g)) || std::binary_search(constraints.begin(),constraints.end(),g))return;
        if(!required_.emplace(std::min(f,g),std::max(f,g)).second)return;
        added[f].push_back(g);added[g].push_back(f);
      });
    }
  }
  return {std::make_move_iterator(added.begin()),std::make_move_iterator(added.end())};
}
} // namespace sentryward::algorithm
