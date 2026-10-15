#pragma once
#include <optional>
#include <vector>
#include "sentryward/algorithm/analysis/member_set.h"

namespace sentryward::algorithm {
// A rectangle or a translated diagonal. Members remain independent events.
struct MemberRelation {
  MemberSet left, right;
  bool diagonal = false;
  std::int64_t shift = 0;
  static MemberRelation Rectangle(MemberSet a, MemberSet b) {
    return {std::move(a),std::move(b),false,0};
  }
  static MemberRelation Diagonal(MemberSet a,std::size_t right_extent,std::int64_t shift=0) {
    auto b=a.shifted(shift,right_extent);
    a=a.intersect(b.shifted(-shift,a.extent()));
    return {std::move(a),std::move(b),true,shift};
  }
  bool empty() const { return left.empty() || right.empty(); }
  bool contains(std::size_t i,std::size_t j) const {
    return left.contains(i) && right.contains(j) &&
      (!diagonal || std::int64_t(j)-std::int64_t(i)==shift);
  }
  MemberRelation transpose() const {
    return diagonal?Diagonal(right,left.extent(),-shift):Rectangle(right,left);
  }
  std::optional<std::pair<std::size_t,std::size_t>> witness() const {
    if(empty())return {};
    const auto a=left.first();std::size_t b=SIZE_MAX;
    if(diagonal)b=std::size_t(std::int64_t(a)+shift);
    else b=right.first();
    return std::pair{a,b};
  }
};
inline MemberSet MemberRange(std::size_t extent,std::size_t begin,std::size_t end) {
  MemberSet result(extent);result.addRange(begin,end);return result;
}
inline bool CoversRelation(const MemberRelation& a,const MemberRelation& b) {
  return (!a.diagonal || (b.diagonal && a.shift==b.shift)) &&
      b.left.isSubsetOf(a.left) && b.right.isSubsetOf(a.right);
}
inline bool IntersectsRelation(const MemberRelation& a,const MemberRelation& b) {
  if(a.diagonal && b.diagonal && a.shift!=b.shift)return false;
  // For product/diagonal mixtures this is a conservative candidate test.
  // False positives are allowed, false negatives are not.
  return a.left.intersects(b.left) && a.right.intersects(b.right);
}
inline std::size_t RelationHash(const MemberRelation& r) {
  auto h=r.left.hash();
  const auto mix=[&](std::size_t x){h^=x+0x9e3779b97f4a7c15ULL+(h<<6)+(h>>2);};
  mix(r.right.hash());mix(r.diagonal);mix(std::size_t(r.shift));return h;
}
inline std::vector<MemberRelation> SubtractRelation(const MemberRelation& a,
    const MemberRelation& b,std::size_t& enumerated) {
  std::vector<MemberRelation> out;
  if(!IntersectsRelation(a,b)) {if(!a.empty())out.push_back(a);return out;}
  if(CoversRelation(b,a))return out;
  const auto emit=[&](MemberRelation r){if(!r.empty())out.push_back(std::move(r));};
  if(a.diagonal) {
    auto removed=b.diagonal?(a.shift==b.shift?b.left:MemberSet(a.left.extent())):
      b.left.intersect(b.right.shifted(-a.shift,a.left.extent()));
    emit(MemberRelation::Diagonal(a.left.subtract(removed),a.right.extent(),a.shift));
  } else if(!b.diagonal) {
    emit(MemberRelation::Rectangle(a.left.subtract(b.left),a.right));
    emit(MemberRelation::Rectangle(a.left.intersect(b.left),a.right.subtract(b.right)));
  } else {
    auto shared=a.left.intersect(b.left).intersect(a.right.shifted(-b.shift,a.left.extent()));
    emit(MemberRelation::Rectangle(a.left.subtract(shared),a.right));
    shared.forEach([&](auto i){
      ++enumerated;
      auto row=MemberRange(a.left.extent(),i,i+1);
      auto removed=MemberRange(a.right.extent(),std::size_t(std::int64_t(i)+b.shift),std::size_t(std::int64_t(i)+b.shift)+1);
      emit(MemberRelation::Rectangle(std::move(row),a.right.subtract(removed)));
    });
  }
  return out;
}
inline void RemoveRelation(std::vector<MemberRelation>& rows,const MemberRelation& r,std::size_t& enumerated) {
  // The common HB result covers an entire family relation. Answer directly
  // in its bitset representation, without allocating subtraction fragments.
  if(rows.empty() || r.empty())return;
  std::erase_if(rows,[&](const auto& row){return CoversRelation(r,row);});
  if(rows.empty())return;
  if(std::none_of(rows.begin(),rows.end(),[&](const auto& row) {
    return row.left.intersects(r.left) && row.right.intersects(r.right);
  }))return;
  std::vector<MemberRelation> result;
  for(const auto& row:rows) {
    if(!IntersectsRelation(row,r)) {result.push_back(row);continue;}
    auto pieces=SubtractRelation(row,r,enumerated);
    for(auto& piece:pieces)result.push_back(std::move(piece));
  }
  rows=std::move(result);
}
} // namespace sentryward::algorithm
