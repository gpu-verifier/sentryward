#pragma once
#include "output_alignment_builder.hpp"
#include <numeric>
#include <queue>

namespace rg::output_alignment_detail {
inline uint64_t checkedExtent(Z n) {
  if(n<0 || n>UINT64_MAX)throw Unsupported("output extent overflow");
  return n.convert_to<uint64_t>();
}
inline uint64_t elementCount(const std::vector<uint64_t>& xs) {
  Z n=1;for(auto x:xs){if(!x)throw Unsupported("empty output axis");n*=x;}return checkedExtent(n);
}
struct Axis {uint64_t stride,size;unsigned index;};
struct Range {uint64_t first,last,step;};
// Exact finite AP intersection. Work depends on stored component pairs, not members.
inline bool intersects(Range a,Range b) {
  if(std::max(a.first,b.first)>std::min(a.last,b.last))return false;
  Z g=std::gcd(a.step,b.step),delta=Z(b.first)-a.first;
  if(delta%g!=0)return false;
  Z modulus=Z(b.step)/g,aa=Z(a.step)/g;
  Z x=0,y=1,r=modulus,s=aa;
  while(s!=0){Z q=r/s;Z nr=r-q*s;r=s;s=nr;Z nx=x-q*y;x=y;y=nx;}
  Z i=modulus==1?Z(0):Z((delta/g*x)%modulus);if(i<0)i+=modulus;
  Z first=Z(a.first)+Z(a.step)*i,period=Z(a.step)*modulus,lo=std::max(a.first,b.first);
  if(first<lo)first+=((lo-first+period-1)/period)*period;
  return first<=std::min(a.last,b.last);
}
// Pairwise disjointness of n items, swept along one axis on which item i spans
// [lo(i), hi(i)]: only an active item p whose span meets the current item b's
// is tested with overlap(p,b), not all pairs.
template<class Lo,class Hi,class Overlap> void requireDisjoint(std::size_t n,Lo lo,Hi hi,Overlap overlap) {
  std::vector<std::size_t> order(n);std::iota(order.begin(),order.end(),std::size_t(0));
  std::sort(order.begin(),order.end(),[&](std::size_t x,std::size_t y){return lo(x)<lo(y);});
  std::vector<std::size_t> active;
  for(auto b:order) {
    std::erase_if(active,[&](std::size_t p){return hi(p)<lo(b);});
    for(auto p:active)if(overlap(p,b))throw Unsupported("overlapping outputs");
    active.push_back(b);
  }
}
// For a common nontrivial step, group finite APs by residue and check adjacent
// sorted hulls: O(n log n). Mixed steps use requireDisjoint's overlap sweep.
inline void requireDisjointRanges(const std::vector<Range>& ranges) {
  std::optional<uint64_t> step;bool common=true;
  for(const auto& r:ranges)if(r.first!=r.last){if(!step)step=r.step;else if(*step!=r.step){common=false;break;}}
  if(common && step && *step) {
    const auto s=*step;
    std::vector<std::size_t> order(ranges.size());std::iota(order.begin(),order.end(),std::size_t(0));
    std::sort(order.begin(),order.end(),[&](std::size_t x,std::size_t y){
      const auto cx=ranges[x].first%s,cy=ranges[y].first%s;
      return cx!=cy?cx<cy:ranges[x].first<ranges[y].first;});
    for(std::size_t i=1;i<order.size();++i) {
      const auto& p=ranges[order[i-1]];const auto& q=ranges[order[i]];
      if(p.first%s==q.first%s && p.last>=q.first)throw Unsupported("overlapping outputs");
    }
    return;
  }
  requireDisjoint(ranges.size(),[&](std::size_t i){return ranges[i].first;},[&](std::size_t i){return ranges[i].last;},
    [&](std::size_t p,std::size_t b){return intersects(ranges[p],ranges[b]);});
}
// The pairs a sweep along `axis` tests (requireDisjoint): for each box in order of its first
// coordinate on the axis, the earlier boxes whose last coordinate is not below it. O(n log n).
inline std::size_t sweepPairs(const std::vector<std::vector<Range>>& boxes,std::size_t axis) {
  std::vector<std::pair<uint64_t,uint64_t>> spans;spans.reserve(boxes.size());
  for(const auto& box:boxes)spans.emplace_back(box[axis].first,box[axis].last);
  std::sort(spans.begin(),spans.end());
  std::priority_queue<uint64_t,std::vector<uint64_t>,std::greater<uint64_t>> lasts;
  std::size_t pairs=0;
  for(const auto& [first,last]:spans) {
    while(!lasts.empty() && lasts.top()<first)lasts.pop();
    pairs+=lasts.size();lasts.push(last);
  }
  return pairs;
}
// Choose the sweep axis with the fewest overlapping spans, preserving `sweep`
// on ties; check candidate pairs on every axis. Axis selection costs O(d n log n).
// All boxes must have equal rank; with multiple boxes, rank must exceed `sweep`.
inline void requireDisjointBoxes(const std::vector<std::vector<Range>>& boxes,std::size_t sweep) {
  if(boxes.size()>2) {
    auto best=sweepPairs(boxes,sweep);
    for(std::size_t axis=0;axis<boxes.front().size() && best;++axis)
      if(axis!=sweep)if(const auto pairs=sweepPairs(boxes,axis);pairs<best){best=pairs;sweep=axis;}
  }
  requireDisjoint(boxes.size(),[&](std::size_t x){return boxes[x][sweep].first;},
      [&](std::size_t x){return boxes[x][sweep].last;},
      [&](std::size_t p,std::size_t q){
        for(std::size_t i=0;i<boxes[p].size();++i)if(!intersects(boxes[p][i],boxes[q][i]))return false;
        return true;});
}
// A family dimension's digit on one (tensor or grid) axis: the axis coordinate gains
// slope x ((j_dim / divisor) % radix) over `count` positions; radix 0: a single term (no Remainder).
struct Term {unsigned dim;uint64_t divisor,radix,slope,count;};
// Why a family box does not bind to the axes (bindDimension, tileAxis); each caller words the diagnostic
// in a table with one message per value, checked against box_error_count (Extent stays last).
enum class BoxError {Stride,ExcludedAxis,FusedStride,FusedCrossing,FusedAlignment,FusedExtent,Holes,Extent};
constexpr std::size_t box_error_count=std::size_t(BoxError::Extent)+1;
// Binds family dimension `dim` (extent n > 1, byte stride `stride`) of a box at `base` to `axes`
// (ascending strides; terms are appended by axis index). The largest axis stride dividing `stride`
// takes one term when the dimension fits that axis from `base`; otherwise, at unit slope, the
// dimension is fused over consecutive axes as an exact mixed-radix chain: each digit min(rest, axis
// size), aligned (a digit spanning a whole axis starts at 0) and dividing the rest. No term may use
// the axis `excluded`. O(axes).
inline std::optional<BoxError> bindDimension(const std::vector<Axis>& axes,const std::vector<uint64_t>& base,
    std::optional<unsigned> excluded,unsigned dim,uint64_t n,uint64_t stride,std::vector<std::vector<Term>>& terms) {
  const auto blocked=[&](const Axis& x){return excluded && x.index==*excluded;};
  auto reverse=std::find_if(axes.rbegin(),axes.rend(),[&](Axis x){return stride>0 && stride%x.stride==0;});
  if(reverse==axes.rend())return BoxError::Stride;
  auto it=std::prev(reverse.base());
  if(blocked(*it))return BoxError::ExcludedAxis;
  const auto slope=stride/it->stride;
  if(Z(slope)*(n-1)<Z(it->size)-Z(base[it->index])){terms[it->index].push_back({dim,1,0,slope,n});return std::nullopt;}
  if(slope!=1)return BoxError::FusedStride;
  uint64_t remaining=n,divisor=1,physical=stride;
  while(remaining>1) {
    if(it==axes.end() || it->stride!=physical || blocked(*it))return BoxError::FusedCrossing;
    const auto digit=std::min(remaining,it->size);
    if(remaining>=it->size && base[it->index])return BoxError::FusedAlignment;
    if(remaining%digit)return BoxError::FusedExtent;
    terms[it->index].push_back({dim,divisor,digit,1,digit});remaining/=digit;
    divisor=checkedExtent(Z(divisor)*digit);physical=checkedExtent(Z(physical)*digit);++it;
  }
  return std::nullopt;
}
// Sorts one axis's terms by slope and checks that they tile base + step x [0, count) exactly (each
// slope the product of the smaller terms' counts) inside an axis of `size`. O(terms log terms).
inline std::optional<BoxError> tileAxis(std::vector<Term>& terms,uint64_t base,uint64_t size,uint64_t& step,uint64_t& count) {
  std::sort(terms.begin(),terms.end(),[](const Term& x,const Term& y){return x.slope<y.slope;});
  step=terms.empty()?1:terms.front().slope;count=1;
  for(const auto& term:terms) {
    if(Z(term.slope)!=Z(step)*count)return BoxError::Holes;
    count=checkedExtent(Z(count)*term.count);
  }
  if(Z(base)+Z(step)*(count-1)>=Z(size))return BoxError::Extent;
  return std::nullopt;
}
// The most segments splitPiece cuts one dimension into, per axis. A run along a dense mixed-radix
// chain of axes is a union of at most two maximal aligned segments per axis (one rising to an
// aligned start, one falling to the run's end); a run that needs more is left to the refusal.
constexpr std::size_t kSegmentsPerAxis=2;
// A part of a piece: the family coordinates first + [0, extent), relative to the piece's own, whose
// first element is at byte `offset`.
struct Part {uint64_t offset;std::vector<uint64_t> first,extent;};
// The parts of a piece (first element at byte `offset`; dimensions `extent` at byte `strides`) that
// is no box on `axes` (ascending strides) because a dimension crosses an axis boundary. Each
// dimension is cut, along its own run from the piece's first element, into segments: from each
// start, the longest run that stays inside the axis it moves along (the largest stride dividing its
// stride, as bindDimension), or, at unit slope from a start aligned to the whole rows of the dense
// axes below some axis, those rows times what remains of that axis. The parts are the products of
// the segments; the caller places each as a box and refuses unless every one is. Empty when no
// dimension is cut, a segment would start outside the axes, or a dimension needs more than
// kSegmentsPerAxis segments per axis. O(dims x axes^2), and O(product of segment counts) parts.
inline std::vector<Part> splitPiece(const std::vector<Axis>& axes,uint64_t origin,uint64_t offset,
    const std::vector<uint64_t>& extent,const std::vector<uint64_t>& strides) {
  std::vector<std::vector<std::pair<uint64_t,uint64_t>>> segments(extent.size());
  bool cut=false;
  for(std::size_t d=0;d<extent.size();++d) {
    const auto n=extent[d],stride=strides[d];
    auto& list=segments[d];
    const auto home=std::find_if(axes.rbegin(),axes.rend(),[&](const Axis& x){return stride>0 && stride%x.stride==0;});
    if(n<=1 || home==axes.rend()){list.push_back({0,n});continue;}
    const auto a=std::size_t(std::prev(home.base())-axes.begin());
    const auto slope=stride/axes[a].stride;
    for(uint64_t start=0;start<n;) {
      if(list.size()>=kSegmentsPerAxis*axes.size())return {};
      // The axis coordinates of the segment's first element.
      std::vector<uint64_t> digit(axes.size());
      auto rem=checkedExtent(Z(offset)+Z(start)*stride-origin);
      for(std::size_t i=axes.size();i-->0;){digit[i]=rem/axes[i].stride;rem%=axes[i].stride;if(digit[i]>=axes[i].size)return {};}
      const auto rest=n-start;
      auto length=std::min<uint64_t>(rest,(axes[a].size-1-digit[a])/slope+1);
      if(slope==1) {
        Z row=1;
        for(std::size_t i=a;i<axes.size() && row<=rest;++i) {
          if(i>a && Z(axes[i].stride)!=Z(axes[i-1].stride)*axes[i-1].size)break;
          length=std::max(length,checkedExtent(Z(row*std::min(Z(axes[i].size-digit[i]),Z(Z(rest)/row)))));
          if(digit[i])break;
          row*=axes[i].size;
        }
      }
      list.push_back({start,length});start+=length;
    }
    cut|=list.size()>1;
  }
  if(!cut)return {};
  std::vector<Part> parts{{offset,std::vector<uint64_t>(extent.size()),extent}};
  for(std::size_t d=0;d<extent.size();++d) {
    if(segments[d].size()==1)continue;
    std::vector<Part> next;
    for(const auto& part:parts)for(const auto& [start,length]:segments[d]) {
      auto p=part;p.offset=checkedExtent(Z(part.offset)+Z(start)*strides[d]);p.first[d]=start;p.extent[d]=length;
      next.push_back(std::move(p));
    }
    parts=std::move(next);
  }
  return parts;
}
// An axis coordinate: base + the digits `terms` of the family coordinates index(first + dim);
// `remainder(t)` says whether digit t takes its Remainder.
template<class Remainder> Edge axisCoordinate(OutputAlignmentBuilder& x,uint64_t base,const std::vector<Term>& terms,unsigned first,
    Remainder remainder) {
  Edge c=x.constant(base);
  for(const auto& t:terms) {
    Edge j=x.index(first+t.dim);
    if(t.divisor!=1)j=x.op("Divide",j,x.constant(t.divisor));
    if(remainder(t))j=x.op("Remainder",j,x.constant(t.radix));
    if(t.slope!=1)j=x.op("Multiply",j,x.constant(t.slope));
    c=x.op("Add",c,j);
  }
  return c;
}
// Lane `slot` (width bytes) of a packed value of `packed` bytes and type `type`: reinterpreted as
// unsigned bits, shifted right by slot x width bytes and truncated to the lane. False (value
// unchanged) when the type is not b or u bits of the packed width.
} // namespace rg::output_alignment_detail
