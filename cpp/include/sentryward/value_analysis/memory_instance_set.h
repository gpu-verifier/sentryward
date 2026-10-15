#pragma once

#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

#include "sentryward/value_analysis/memory_actions.h"

namespace sentryward::value_analysis {

// Exact membership; adjacent inserted instances share a storage interval.
class MemoryInstanceSet final {
 public:
  MemoryInstanceSet() = default;
  MemoryInstanceSet(const MemoryInstanceSet&) = default;
  MemoryInstanceSet& operator=(const MemoryInstanceSet&) = default;
  MemoryInstanceSet(MemoryInstanceSet&& other) noexcept
      : ranges_(std::move(other.ranges_)), size_(std::exchange(other.size_,0)) {
    other.ranges_.clear();
  }
  MemoryInstanceSet& operator=(MemoryInstanceSet&& other) noexcept {
    if (this!=&other) {
      ranges_=std::move(other.ranges_);
      size_=std::exchange(other.size_,0);
      other.ranges_.clear();
    }
    return *this;
  }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] bool empty() const noexcept { return size_==0; }
  [[nodiscard]] std::size_t storedRangeCount() const noexcept { return ranges_.size(); }
  [[nodiscard]] bool contains(MemoryInstance key) const {
    auto right=ranges_.upper_bound(key);
    if (right==ranges_.begin()) return false;
    const auto left=std::prev(right);
    return left->first.node==key.node && key.instance<=left->second;
  }
  std::pair<MemoryInstance,bool> insert(MemoryInstance key) {
    auto right=ranges_.upper_bound(key);
    auto left=right;
    bool adjacent_left=false;
    if (right!=ranges_.begin()) {
      --left;
      if (left->first.node==key.node) {
        if (key.instance<=left->second) return {key,false};
        adjacent_left=left->second!=std::numeric_limits<std::size_t>::max() &&
                      left->second+1==key.instance;
      }
    }
    if (size_==std::numeric_limits<std::size_t>::max())
      throw std::length_error("memory instance membership count overflow");
    const bool adjacent_right=right!=ranges_.end() && right->first.node==key.node &&
        key.instance!=std::numeric_limits<std::size_t>::max() &&
        key.instance+1==right->first.instance;
    if (adjacent_left) {
      left->second=adjacent_right ? right->second : key.instance;
      if (adjacent_right) ranges_.erase(right);
    } else {
      const auto end=adjacent_right ? right->second : key.instance;
      ranges_.emplace(key,end);
      if (adjacent_right) ranges_.erase(right);
    }
    ++size_;
    return {key,true};
  }
  template<class Iterator> void insert(Iterator first,Iterator last) {
    for (;first!=last;++first) insert(*first);
  }
 private:
  std::map<MemoryInstance,std::size_t> ranges_;
  std::size_t size_=0;
};

}  // namespace sentryward::value_analysis
