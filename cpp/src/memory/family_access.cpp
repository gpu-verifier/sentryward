#include "sentryward/algorithm/memory/family_access.h"
#include <algorithm>
#include <limits>

namespace sentryward::algorithm {
using I=address_solver::Int;
namespace {
I Floor(I a,I b) { auto q=a/b,r=a%b;return q-(r<0); }
I Ceil(I a,I b) { return -Floor(-a,b); }
MemberSet InBand(std::size_t extent,std::size_t begin,std::size_t end,I coefficient,I low,I high) {
  if(coefficient<0) {coefficient=-coefficient;const auto old=low;low=-high;high=-old;}
  if(!coefficient)return low<=0 && high>=0?MemberRange(extent,begin,end):MemberSet(extent);
  const auto lo=std::max(I(begin),Ceil(low,coefficient));
  const auto hi=std::min(I(end)-1,Floor(high,coefficient));
  return lo<=hi?MemberRange(extent,std::size_t(lo),std::size_t(hi)+1):MemberSet(extent);
}
}
FamilyOverlap ClassifyFamilyOverlap(const MemoryAccessFamily& a,const FamilyFootprint& x,
    const MemoryAccessFamily& b,const FamilyFootprint& y) {
  FamilyOverlap result;
  if(x.empty || y.empty)return result;
  if(!x.space.empty() && !y.space.empty() && x.space!=y.space)return result;
  if(!x.space.empty() && x.space==y.space && !x.allocation.empty() && !y.allocation.empty() &&
      x.allocation!=y.allocation && !(x.alias!="" && x.alias==y.alias))return result;
  auto all=MemberRelation::Rectangle(a.members(),b.members());
  if(!x.affine || !y.affine || x.space!=y.space || x.allocation!=y.allocation || x.symbolic!=y.symbolic) {
    result.unknown.push_back(std::move(all));return result;
  }
  // Bounds are inclusive integer-byte overlap conditions.
  const I low=-I(x.bytes)+1,high=I(y.bytes)-1;
  I difference;
  if(__builtin_sub_overflow(x.offset,y.offset,&difference)) {result.unknown.push_back(std::move(all));return result;}
  // Reject unrepresentable arithmetic before interval or stride simplification.
  const I bound=I(1)<<90;
  const auto safe=[&](I v){return v>-bound && v<bound;};
  if(!safe(difference) || !safe(x.step) || !safe(y.step) || a.extent>UINT32_MAX || b.extent>UINT32_MAX) {
    result.unknown.push_back(std::move(all));return result;
  }
  if(!x.step) {
    auto right=InBand(b.extent,b.first_member,b.first_member+b.count,-y.step,low-difference,high-difference);
    if(!right.empty())result.overlap.push_back(MemberRelation::Rectangle(a.members(),std::move(right)));
    return result;
  }
  if(!y.step) {
    auto left=InBand(a.extent,a.first_member,a.first_member+a.count,x.step,low-difference,high-difference);
    if(!left.empty())result.overlap.push_back(MemberRelation::Rectangle(std::move(left),b.members()));
    return result;
  }
  const auto xmin=x.offset+x.step*I(a.first_member),xmax=x.offset+x.step*I(a.first_member+a.count-1);
  const auto ymin=y.offset+y.step*I(b.first_member),ymax=y.offset+y.step*I(b.first_member+b.count-1);
  const auto mind=std::min(xmin,xmax)-std::max(ymin,ymax),maxd=std::max(xmin,xmax)-std::min(ymin,ymax);
  if(maxd<low || mind>high)return result;
  if(mind>=low && maxd<=high) {result.overlap.push_back(std::move(all));return result;}
  if(x.step==y.step) {
    I l=low-difference,h=high-difference,c=-x.step;
    if(c<0) {c=-c;const auto old=l;l=-h;h=-old;}
    const auto first=std::max(Ceil(l,c),I(b.first_member)-I(a.first_member+a.count-1));
    const auto last=std::min(Floor(h,c),I(b.first_member+b.count-1)-I(a.first_member));
    // A narrow band is represented by diagonals, not by member pairs.
    if(last-first<64) {
      for(auto shift=first;shift<=last;++shift) {
        auto members=a.members().intersect(b.members().shifted(-std::int64_t(shift),a.extent));
        if(!members.empty())result.overlap.push_back(MemberRelation::Diagonal(std::move(members),b.extent,std::int64_t(shift)));
      }
      return result;
    }
  }
  // Exact row intervals for general unequal strides; never invoke a solver per pair.
  for(std::size_t i=a.first_member;i<a.first_member+a.count;++i) {
    ++result.member_rows;
    auto right=InBand(b.extent,b.first_member,b.first_member+b.count,-y.step,
        low-difference-x.step*I(i),high-difference-x.step*I(i));
    if(!right.empty())result.overlap.push_back(MemberRelation::Rectangle(MemberRange(a.extent,i,i+1),std::move(right)));
  }
  return result;
}
} // namespace sentryward::algorithm
