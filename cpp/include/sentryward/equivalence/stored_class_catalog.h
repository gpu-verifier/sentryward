#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace sentryward::equivalence::stored_class_catalog {

// Bookkeeping only: IDs denote actual stored heads, not arithmetic facts.
// Engine owns union/find, immutable headers and the current epoch boundary.
// Append is amortized O(1); splice is O(1). No member array is copied on union.
class Members {
 public:
  using Id=std::size_t;
  static constexpr Id absent=std::numeric_limits<Id>::max();
  void clear() {next_.clear();head_.clear();tail_.clear();count_.clear();}
  std::size_t size() const {return next_.size();}
  void append(Id id) {
    if(id!=size())throw std::logic_error("noncontiguous class catalog append");
    next_.push_back(absent);head_.push_back(id);tail_.push_back(id);count_.push_back(1);
  }
  void splice(Id owner,Id absorbed) {
    if(owner>=size() || absorbed>=size() || owner==absorbed ||
        !count_[owner] || !count_[absorbed] || count_[absorbed]>size()-count_[owner])
      throw std::logic_error("invalid class catalog splice");
    next_[tail_[owner]]=head_[absorbed];tail_[owner]=tail_[absorbed];
    count_[owner]+=count_[absorbed];head_[absorbed]=tail_[absorbed]=absent;count_[absorbed]=0;
  }
  // A prefix cap is about actual snapshot members, NOT later appended heads.
  // All retained views own their IDs. Every visited link/copy and sort is
  // charged before work; a failed admission returns Unknown, never a prefix.
  template<class Charge> std::optional<std::vector<Id>> prefix(
      Id owner,std::size_t boundary,std::size_t max_members,Charge&& charge) const {
    if(owner>=size() || boundary>size() || !count_[owner] || !charge(1))return {};
    std::vector<Id> result;
    const auto capacity=std::min({count_[owner],max_members,boundary});
    if(!charge(capacity))return {};result.reserve(capacity);
    for(auto id=head_[owner];id!=absent;id=next_[id]) {
      if(!charge(1))return {};
      if(id>=boundary)continue;
      if(result.size()>=max_members)return {};
      if(!charge(1))return {};result.push_back(id);
    }
    if(!orderIds(result,boundary,charge))return {};
    return result;
  }
  // Reusable physical-ID ordering, never logical Fold/output materialization.
  // Callers own the vector and charge its construction separately. The same
  // machine-width radix bound serves class prefixes and sparse epoch catalogs.
  template<class Charge> static bool orderIds(std::vector<Id>& result,
      std::size_t boundary,Charge&& charge) {
    if(result.size()<2)return true;
    if(result.size()<=64) {
      // Fixed-size comparison sort. No uncharged wide member ordering.
      if(!charge(result.size()*(1+std::bit_width(result.size()))))return false;
      std::sort(result.begin(),result.end());return true;
    }
    // Wider ranked views preserve the same ascending-ID order in linear work.
    // Eight-bit radix passes bound metadata by physical ID width, not arity,
    // Fold count, output extent, or the number of class merges.
    const unsigned passes=static_cast<unsigned>((std::bit_width(boundary?boundary-1:0)+7)/8);
    if(result.size()>(std::numeric_limits<std::size_t>::max()-512)/2)return false;
    if(!charge(result.size()))return false;
    std::vector<Id> scratch(result.size());
    for(unsigned pass=0;pass<passes;++pass) {
      if(!charge(2*result.size()+512))return false;
      std::array<std::size_t,256> offsets{};
      const unsigned shift=8*pass;
      for(auto id:result)++offsets[(id>>shift)&255];
      std::size_t total=0;
      for(auto& offset:offsets){const auto count=offset;offset=total;total+=count;}
      for(auto id:result)scratch[offsets[(id>>shift)&255]++]=id;
      result.swap(scratch);
    }
    return true;
  }
 private:
  std::vector<Id> next_,head_,tail_;
  std::vector<std::size_t> count_;
};

} // namespace sentryward::equivalence::stored_class_catalog
