#pragma once

#include <cstdint>
#include <cstddef>
#include <utility>
#include <initializer_list>
#include <memory>
#include <vector>
#include <optional>
#include <unordered_map>
#include <algorithm>

namespace sentryward::value_ir {

// Copy-on-write membership; templates share storage without coupling their evolution.
class ThreadMembers {
 public:
  struct AffineSegment {std::size_t begin,count;std::int64_t stride;};
  using value_type=std::uint64_t;
  using const_iterator=std::vector<value_type>::const_iterator;
  ThreadMembers()=default;
  ThreadMembers(std::initializer_list<value_type> values):ThreadMembers(std::vector<value_type>(values)){}
  ThreadMembers(std::vector<value_type> values):values_(std::make_shared<Storage>(std::move(values))){}
  const std::vector<value_type>& values() const {
    static const std::vector<value_type> empty;
    return values_?values_->members:empty;
  }
  const_iterator begin() const{return values().begin();}
  const_iterator end() const{return values().end();}
  std::size_t size() const{return values().size();}
  bool empty() const{return values().empty();}
  const value_type& operator[](std::size_t index) const{return values()[index];}
  const value_type& at(std::size_t index) const{return values().at(index);}
  const value_type& front() const{return values().front();}
  [[nodiscard]] std::optional<std::size_t> indexOf(value_type value) const {
    if(!values_)return {};
    const auto found=values_->positions.find(value);
    if(found==values_->positions.end())return {};
    return found->second;
  }
  void push_back(value_type value){insert(end(),value);}
  const void* storageIdentity() const{return values_.get();}
  // Cached at construction/mutation; consumers never rescan the member table.
  std::optional<std::int64_t> affineStep() const {
    return values_?values_->affine_step:std::nullopt;
  }
  std::optional<AffineSegment> affineSegment(std::size_t index) const {
    if(!values_ || index>=size())return {};
    const auto& runs=values_->segments;
    auto found=std::upper_bound(runs.begin(),runs.end(),index,
        [](auto i,const auto& run){return i<run.begin;});
    return found==runs.begin()?std::nullopt:std::optional<AffineSegment>(*std::prev(found));
  }
  void insert(const_iterator position,value_type value) {
    auto offset=position-begin();
    if(!values_)values_=std::make_shared<Storage>(std::vector<value_type>{});
    else if(values_.use_count()!=1)values_=std::make_shared<Storage>(*values_);
    if(static_cast<std::size_t>(offset)==values_->members.size()) {
      values_->observe(value);
      values_->positions.try_emplace(value,offset);
      values_->members.push_back(value);
      return;
    }
    values_->members.insert(values_->members.begin()+offset,value);
    for(auto& [member,index]:values_->positions)if(index>=static_cast<std::size_t>(offset))++index;
    auto [found,inserted]=values_->positions.try_emplace(value,offset);
    if(!inserted && found->second>static_cast<std::size_t>(offset))found->second=offset;
    values_->recomputeAffine();
  }
 private:
  struct Storage {
    explicit Storage(std::vector<value_type> values):members(std::move(values)) {
      positions.reserve(members.size());
      for(std::size_t i=0;i<members.size();++i)positions.try_emplace(members[i],i);
      recomputeAffine();
    }
    std::optional<std::int64_t> affine_step=0;
    std::vector<AffineSegment> segments;
    void observeSegment(value_type value,std::size_t index) {
      if(!segments.empty()) {
        auto& run=segments.back();
        const auto difference=__int128(value)-members[index-1];
        if(difference>=INT64_MIN && difference<=INT64_MAX &&
            (run.count==1 || difference==run.stride)) {
          run.stride=static_cast<std::int64_t>(difference);++run.count;return;
        }
      }
      segments.push_back({index,1,0});
    }
    void observe(value_type value) {
      observeSegment(value,members.size());
      if(members.empty()){affine_step=0;return;}
      const auto difference=__int128(value)-members.back();
      if(difference<INT64_MIN || difference>INT64_MAX){affine_step.reset();return;}
      if(members.size()==1)affine_step=static_cast<std::int64_t>(difference);
      else if(affine_step && *affine_step!=difference)affine_step.reset();
    }
    void recomputeAffine() {
      segments.clear();
      for(std::size_t i=0;i<members.size();++i)observeSegment(members[i],i);
      affine_step=0;
      if(members.size()<2)return;
      const auto difference=__int128(members[1])-members[0];
      if(difference<INT64_MIN || difference>INT64_MAX){affine_step.reset();return;}
      affine_step=static_cast<std::int64_t>(difference);
      for(std::size_t i=2;i<members.size();++i)
        if(__int128(members[i])-members[i-1]!=difference){affine_step.reset();return;}
    }
    std::vector<value_type> members;
    std::unordered_map<value_type,std::size_t> positions;
  };
  std::shared_ptr<Storage> values_;
};
} // namespace sentryward::value_ir
