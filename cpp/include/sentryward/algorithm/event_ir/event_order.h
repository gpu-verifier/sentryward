#pragma once
#include <algorithm>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <vector>
#include <optional>
#include "sentryward/algorithm/base/affine_sequence.h"
#include "sentryward/algorithm/base/strong_id.h"

namespace sentryward::algorithm {

// Identity and affine orders retain logical IDs without materializing members.
class EventOrder final {
  struct IdCodec {
    using Delta = std::int64_t;
    static std::optional<Delta> difference(EventId a, EventId b) {
      return std::int64_t(b.value()) - a.value();
    }
    static EventId advance(EventId a, Delta delta, std::size_t index) {
      return EventId(static_cast<std::uint32_t>(a.value() + delta * std::int64_t(index)));
    }
  };
 public:
  class Iterator final {
   public:
    using value_type=EventId;
    using difference_type=std::ptrdiff_t;
    using reference=EventId;
    using pointer=void;
    using iterator_category=std::random_access_iterator_tag;
    Iterator()=default;
    EventId operator*() const{return (*order_)[position_];}
    EventId operator[](difference_type n) const{return *(*this+n);}
    Iterator& operator++(){++position_;return *this;}
    Iterator operator++(int){auto old=*this;++*this;return old;}
    Iterator& operator--(){--position_;return *this;}
    Iterator operator--(int){auto old=*this;--*this;return old;}
    Iterator& operator+=(difference_type n){position_+=n;return *this;}
    Iterator& operator-=(difference_type n){return *this+=-n;}
    friend Iterator operator+(Iterator it,difference_type n){return it+=n;}
    friend Iterator operator+(difference_type n,Iterator it){return it+=n;}
    friend Iterator operator-(Iterator it,difference_type n){return it-=n;}
    friend difference_type operator-(Iterator a,Iterator b){return a.position_-b.position_;}
    friend bool operator==(Iterator,Iterator)=default;
    friend auto operator<=>(Iterator a,Iterator b){return a.position_<=>b.position_;}
   private:
    friend class EventOrder;
    Iterator(const EventOrder* order,difference_type position):order_(order),position_(position){}
    const EventOrder* order_=nullptr;
    difference_type position_=0;
  };
  EventOrder()=default;
  EventOrder(std::vector<EventId> ids):ids_(std::move(ids)){}
  static EventOrder identity(std::size_t count) {
    if(count>std::uint64_t{UINT32_MAX}+1 || count>static_cast<std::size_t>(PTRDIFF_MAX))
      throw std::length_error("event order exceeds ID domain");
    EventOrder result;result.identity_=true;result.count_=count;return result;
  }
  bool isIdentity() const noexcept{return identity_;}
  static EventOrder affine() { EventOrder result; result.runs_.emplace(); return result; }
  void appendAffine(EventId first, std::uint32_t stride, std::size_t count) {
    if (!runs_) throw std::logic_error("event order is not affine");
    if (count && (count-1 > (UINT32_MAX-first.value()) / std::uint64_t(stride ? stride : 1)))
      throw std::length_error("event order exceeds ID domain");
    runs_->append(first, stride, count);
  }
  std::size_t storedCount() const noexcept{return runs_?runs_->runs().size():ids_.size();}
  std::size_t size() const noexcept{return identity_?count_:runs_?runs_->size():ids_.size();}
  bool empty() const noexcept{return size()==0;}
  EventId operator[](std::size_t i) const{return identity_?EventId(static_cast<std::uint32_t>(i)):runs_?(*runs_)[i]:ids_[i];}
  EventId at(std::size_t i) const{if(i>=size())throw std::out_of_range("event order index");return (*this)[i];}
  EventId front() const{return at(0);}
  EventId back() const{return at(size()-1);}
  void push_back(EventId id) {
    if(identity_)throw std::logic_error("cannot append to an identity event order");
    if(runs_){runs_->push_back(id);return;}
    ids_.push_back(id);
  }
  Iterator begin() const{return Iterator(this,0);}
  Iterator end() const{return Iterator(this,size());}
  auto rbegin() const{return std::make_reverse_iterator(end());}
  auto rend() const{return std::make_reverse_iterator(begin());}
  friend bool operator==(const EventOrder& a,const std::vector<EventId>& b){return std::ranges::equal(a,b);}
 private:
  bool identity_=false;
  std::size_t count_=0;
  std::vector<EventId> ids_;
  std::optional<AffineSequence<EventId, IdCodec>> runs_;
};
}  // namespace sentryward::algorithm
