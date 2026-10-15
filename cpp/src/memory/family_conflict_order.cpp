#include "sentryward/algorithm/memory/family_conflict_order.h"
#include "sentryward/algorithm/memory/conflict_order.h"
#include <algorithm>
#include <map>
#include <numeric>
#include <set>
#include <tuple>

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
  for(std::size_t i=0;i<families.size();++i) {
    const auto& p=footprints[i];if(p.empty)continue;
    const Storage key{p.space,p.allocation};spaces[key].push_back(i);
    // A tolerated write/write gap cannot certify adjacent conflict orders.
    // Disable reduction only for its storage; other allocations stay reduced.
    if((!exempt_families.empty() && exempt_families[i]) ||
        !p.affine || !p.alias.empty() || !p.symbolic.empty() ||
        (families[i].count>1 && p.step<I(p.bytes) &&
         !(families[i].first.mode==AccessMode::kRead && p.step==0)))blocked[key]=true;
    if(p.allocation.empty() || p.space.empty() || !p.alias.empty()) {
      // Potential cross-allocation aliasing disables reduction for this space.
      blocked_spaces.insert(p.space);
    }
  }
  if(blocked_spaces.contains(""))return;
  std::size_t next_bucket=0;
  for(const auto& [storage,items]:spaces) {
    if(blocked[storage] || blocked_spaces.contains(storage.first))continue;
    const bool scalar=std::all_of(items.begin(),items.end(),[&](auto i) {
      const auto& f=families[i];const auto& p=footprints[i];
      const auto offset=p.offset+p.step*I(f.first_member);
      return f.count==1 && offset>=0 && offset<=I(UINT64_MAX);
    });
    if(scalar) {
      // Reuse the exact interval sweep without expanding any family members.
      std::vector<ConstantConflictRegion> regions;
      std::vector<std::size_t> ranks;
      regions.reserve(items.size());ranks.reserve(items.size());
      for(std::size_t j=0;j<items.size();++j) {
        const auto i=items[j];const auto& f=families[i];const auto& p=footprints[i];
        regions.push_back({AccessId(j),EventId(j),f.first.mode==AccessMode::kWrite,0,
            std::uint64_t(p.offset+p.step*I(f.first_member)),p.bytes});
        ranks.push_back(order->rank(f.first.event));
      }
      auto stream=MakeConflictOrder(std::move(regions),ranks);
      while(auto pair=stream->next()) {
        const auto a=items[pair->first.value()],b=items[pair->second.value()];
        peers_[a].push_back(b);peers_[b].push_back(a);
      }
      const auto id=next_bucket++;
      for(auto i:items) {
        buckets_[i]=id;
        auto& peers=peers_[i];std::sort(peers.begin(),peers.end());
        peers.erase(std::unique(peers.begin(),peers.end()),peers.end());
      }
      reduced_families_+=items.size();
      continue;
    }
    std::map<Pattern,std::vector<std::size_t>> patterns;
    for(auto i:items) {
      const auto& f=families[i];const auto& p=footprints[i];
      patterns[{p.offset+p.step*I(f.first_member),f.count==1?I(0):p.step,f.count,p.bytes}].push_back(i);
    }
    std::vector<Bucket> buckets;
    for(auto& [pattern,members]:patterns) {
      const auto [lo,step,count,bytes]=pattern;
      buckets.push_back({lo,lo+step*I(count-1)+I(bytes),true,std::move(members)});
    }
    // Pattern order is address order. Mark every envelope involved in overlap.
    I maximum=-(I(1)<<125);std::size_t owner=0;
    for(std::size_t i=0;i<buckets.size();++i) {
      if(buckets[i].lo<maximum) {buckets[i].valid=false;buckets[owner].valid=false;}
      if(buckets[i].hi>maximum) {maximum=buckets[i].hi;owner=i;}
    }
    for(auto& bucket:buckets) {
      if(!bucket.valid || bucket.families.size()<2)continue;
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
      if(!bucket.valid)continue;
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
  }
}
const std::vector<std::size_t>* FamilyConflictOrder::candidates(std::size_t family) const {
  return buckets_[family]==SIZE_MAX?nullptr:&peers_[family];
}
} // namespace sentryward::algorithm
