#pragma once
#include <algorithm>
#include <cstdint>
#include <map>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace sentryward::algorithm {
// Finite arithmetic progressions; indices returned by intersection refer to
// the original progressions. No element enumeration or solver is involved.
struct AffineDomain {
  std::uint64_t first=0, step=0, count=0;
  std::uint64_t last() const { return first+step*(count-1); }
};
struct AffineIntersection {
  std::uint64_t left=0,right=0,left_step=0,right_step=0,count=0;
};
inline AffineIntersection IntersectAffine(AffineDomain a,AffineDomain b) {
  if(!a.count || !b.count)return {};
  if(!a.step || !b.step) {
    if(!a.step && !b.step)return a.first==b.first?AffineIntersection{0,0,0,0,1}:AffineIntersection{};
    const bool swap=!b.step; if(swap)std::swap(a,b);
    if(a.first<b.first || a.first>b.last() || (a.first-b.first)%b.step)return {};
    const auto index=(a.first-b.first)/b.step;
    return swap?AffineIntersection{index,0,0,0,1}:AffineIntersection{0,index,0,0,1};
  }
  const auto low=std::max(a.first,b.first),high=std::min(a.last(),b.last());
  if(low>high)return {};
  using I=__int128;
  const auto gcd=std::gcd(a.step,b.step);
  const I difference=I(b.first)-a.first;
  if(difference%gcd)return {};
  const auto modulus=b.step/gcd;
  I old_r=a.step/gcd,r=modulus,old_x=1,x=0;
  while(r) { const I q=old_r/r; const I nr=old_r-q*r;old_r=r;r=nr;
    const I nx=old_x-q*x;old_x=x;x=nx; }
  I k=modulus==1?0:((difference/gcd*old_x)%modulus+modulus)%modulus;
  const I period=I(a.step/gcd)*b.step;
  I value=I(a.first)+I(a.step)*k;
  if(value<low)value+=((I(low)-value+period-1)/period)*period;
  if(value>high)return {};
  const auto count=static_cast<std::uint64_t>((I(high)-value)/period+1);
  return {static_cast<std::uint64_t>((value-a.first)/a.step),
      static_cast<std::uint64_t>((value-b.first)/b.step),
      count>1?static_cast<std::uint64_t>(period/a.step):0,
      count>1?static_cast<std::uint64_t>(period/b.step):0,count};
}

// Exact difference, expressed as progression patterns rather than members.
// A strided hole introduces residue classes, never a list of surviving IDs.
inline std::vector<AffineDomain> SubtractAffine(AffineDomain a,AffineDomain b) {
  const auto m=IntersectAffine(a,b);
  if(!m.count)return {a};
  if(!a.step)return {};
  std::vector<AffineDomain> result;
  const auto append=[&](std::uint64_t first,std::uint64_t step,std::uint64_t count) {
    if(count)result.push_back({a.first+a.step*first,a.step*step,count});
  };
  append(0,1,m.left);
  const auto last=m.left+m.left_step*(m.count-1);
  if(m.count>1 && m.left_step>1) {
    // Choose the smaller exact partition: intervals between sparse holes,
    // or periodic residue classes for densely repeated holes. Two far-apart
    // removed members must not materialize all the surviving members.
    if(m.count<m.left_step) {
      for(std::uint64_t hole=0;hole<m.count-1;++hole)
        append(m.left+hole*m.left_step+1,1,m.left_step-1);
    } else {
      for(std::uint64_t residue=1;residue<m.left_step;++residue)
        append(m.left+residue,m.left_step,m.count-1);
    }
  }
  append(last+1,1,a.count-last-1);
  return result;
}

// Index disjoint progressions. Cost is in stored stride/residue patterns, not
// logical members. The caller supplies an immutable payload index for each run.
class AffineIndex {
  struct Run { AffineDomain domain;std::size_t payload; };
  using Runs=std::vector<Run>;
  std::map<std::uint64_t,std::map<std::uint64_t,Runs>> buckets_;
 public:
  void add(AffineDomain domain,std::size_t payload) {
    if(!domain.count)return;
    if(domain.count==1)domain.step=0;
    buckets_[domain.step][domain.step?domain.first%domain.step:domain.first].push_back({domain,payload});
  }
  void seal() {
    for(auto& [step,residues]:buckets_)for(auto& [residue,runs]:residues) {
      (void)step;(void)residue;
      std::sort(runs.begin(),runs.end(),[](auto a,auto b){return a.domain.first<b.domain.first;});
      for(std::size_t i=1;i<runs.size();++i)
        if(runs[i-1].domain.last()>=runs[i].domain.first)
          throw std::logic_error("overlapping affine index keys");
    }
  }
  template<class Visit> void intersect(AffineDomain query,Visit visit) const {
    if(!query.count)return;
    for(const auto& [step,residues]:buckets_) {
      const auto scan=[&](const Runs& runs) {
        auto it=std::lower_bound(runs.begin(),runs.end(),query.first,
            [](const Run& run,auto value){return run.domain.last()<value;});
        for(;it!=runs.end() && it->domain.first<=query.last();++it)
          if(auto match=IntersectAffine(query,it->domain);match.count)visit(it->payload,match);
      };
      if(!step) {
        for(auto it=residues.lower_bound(query.first);it!=residues.end() && it->first<=query.last();++it)
          if(!query.step || (it->first-query.first)%query.step==0)scan(it->second);
      } else if(query.count==1 || !query.step || query.step%step==0) {
        const auto it=residues.find(query.first%step);if(it!=residues.end())scan(it->second);
      } else {
        const auto gcd=std::gcd(query.step,step);
        for(const auto& [residue,runs]:residues)if(residue%gcd==query.first%gcd)scan(runs);
      }
    }
  }
};
}
