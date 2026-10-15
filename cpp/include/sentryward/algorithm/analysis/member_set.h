#pragma once
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace sentryward::algorithm {
// Coordinate positions are preserved when the active subset changes.
class MemberSet final {
 public:
  explicit MemberSet(std::size_t extent = 0) : extent_(extent) {
    if (extent > 256) extra_.resize((extent+63)/64);
  }
  std::size_t extent() const { return extent_; }
  static std::size_t storageWords(std::size_t extent) { return extent<=256?4:(extent+63)/64; }
  std::size_t words() const { return storageWords(extent_); }
  // A lookup accelerator only: callers must still compare sets on a hash hit.
  std::size_t hash() const {
    std::size_t h=extent_;
    for(std::size_t i=0;i<words();++i)
      h^=word(i)+0x9e3779b97f4a7c15ULL+(h<<6)+(h>>2);
    return h;
  }
  bool empty() const { for(std::size_t i=0;i<words();++i)if(word(i))return false; return true; }
  std::size_t first() const {
    for(std::size_t i=0;i<words();++i)if(word(i))return i*64+std::countr_zero(word(i));
    return SIZE_MAX;
  }
  std::size_t count() const {
    std::size_t n=0;for(std::size_t i=0;i<words();++i)n+=std::popcount(word(i));return n;
  }
  // Visit maximal set-bit runs, not individual members. Cross-word runs join.
  template<class Visit> void forEachRun(Visit visit) const {
    std::size_t begin=SIZE_MAX,end=0;
    for(std::size_t w=0;w<words();++w) {
      auto bits=word(w);
      while(bits) {
        const unsigned start=std::countr_zero(bits);
        const unsigned length=std::countr_one(bits>>start);
        const auto a=w*64+start,b=a+length;
        if(begin!=SIZE_MAX && a==end)end=b;
        else {if(begin!=SIZE_MAX)visit(begin,end);begin=a;end=b;}
        bits&=~((length==64?UINT64_MAX:((std::uint64_t(1)<<length)-1))<<start);
      }
    }
    if(begin!=SIZE_MAX)visit(begin,end);
  }
  bool contains(std::size_t bit) const { return bit<extent_ && ((word(bit/64)>>(bit%64))&1); }
  void set(std::size_t bit) {
    if(bit>=extent_)throw std::out_of_range("member coordinate");
    word(bit/64)|=std::uint64_t(1)<<(bit%64);
  }
  void addRange(std::size_t begin,std::size_t end) {
    if(begin>end || end>extent_)throw std::out_of_range("member coordinate range");
    while(begin<end) {
      const auto n=std::min(end-begin,64-begin%64);
      word(begin/64)|=(n==64?UINT64_MAX:((std::uint64_t(1)<<n)-1))<<(begin%64);
      begin+=n;
    }
  }
  MemberSet& unite(const MemberSet& other) {
    compatible(other);for(std::size_t i=0;i<words();++i)word(i)|=other.word(i);return *this;
  }
  MemberSet intersect(const MemberSet& other) const {
    compatible(other);auto result=*this;
    for(std::size_t i=0;i<words();++i)result.word(i)&=other.word(i);return result;
  }
  MemberSet subtract(const MemberSet& other) const {
    compatible(other);auto result=*this;
    for(std::size_t i=0;i<words();++i)result.word(i)&=~other.word(i);return result;
  }
  bool intersects(const MemberSet& other) const {
    compatible(other);
    for(std::size_t i=0;i<words();++i)if(word(i)&other.word(i))return true;
    return false;
  }
  bool isSubsetOf(const MemberSet& other) const {
    compatible(other);
    for(std::size_t i=0;i<words();++i)if(word(i)&~other.word(i))return false;
    return true;
  }
  MemberSet shifted(std::int64_t offset,std::size_t destination_extent) const {
    MemberSet result(destination_extent);
    const auto magnitude=offset<0?std::uint64_t(-(offset+1))+1:std::uint64_t(offset);
    if((offset<0 && magnitude>=extent_) || (offset>=0 && magnitude>=destination_extent))return result;
    const auto whole=std::size_t(magnitude/64);
    const auto part=unsigned(magnitude%64);
    for(std::size_t i=0;i<words();++i) {
      const auto value=word(i);
      if(offset>=0) {
        if(i+whole<result.words())result.word(i+whole)|=value<<part;
        if(part && i+whole+1<result.words())result.word(i+whole+1)|=value>>(64-part);
      } else {
        if(i>=whole && i-whole<result.words())result.word(i-whole)|=value>>part;
        if(part && i>whole && i-whole-1<result.words())result.word(i-whole-1)|=value<<(64-part);
      }
    }
    const auto used=(destination_extent+63)/64;
    for(auto i=used;i<result.words();++i)result.word(i)=0;
    if(used && destination_extent%64)result.word(used-1)&=(std::uint64_t(1)<<(destination_extent%64))-1;
    return result;
  }
  template<class Visit> void forEach(Visit visit) const {
    for(std::size_t i=0;i<words();++i)for(auto bits=word(i);bits;bits&=bits-1)
      visit(i*64+std::countr_zero(bits));
  }
  bool operator==(const MemberSet& other) const {
    if(extent_!=other.extent_)return false;
    for(std::size_t i=0;i<words();++i)if(word(i)!=other.word(i))return false;return true;
  }
 private:
  void compatible(const MemberSet& other) const {
    if(extent_!=other.extent_)throw std::invalid_argument("member coordinate domain mismatch");
  }
  std::uint64_t word(std::size_t i) const { return extent_<=256?inline_[i]:extra_[i]; }
  std::uint64_t& word(std::size_t i) { return extent_<=256?inline_[i]:extra_[i]; }
  std::size_t extent_;
  std::array<std::uint64_t,4> inline_{};
  std::vector<std::uint64_t> extra_;
};
}  // namespace sentryward::algorithm
