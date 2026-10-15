#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace sentryward::algorithm {

// Ordered logical records, stored as exact affine runs.
template<class T, class Codec>
class AffineSequence final {
 public:
  using value_type = T;
  using Delta = typename Codec::Delta;
  struct Run { std::size_t begin, count; T first; Delta delta; };
  class Iterator {
   public:
    using value_type = T;
    using reference = T;
    using pointer = void;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;
    Iterator() = default;
    Iterator(const AffineSequence* owner, std::size_t index) : owner_(owner), index_(index) {}
    T operator*() const { return (*owner_)[index_]; }
    Iterator& operator++() { ++index_; return *this; }
    Iterator operator++(int) { auto old = *this; ++*this; return old; }
    friend bool operator==(const Iterator&, const Iterator&) = default;
   private:
    const AffineSequence* owner_ = nullptr;
    std::size_t index_ = 0;
  };
  [[nodiscard]] std::size_t size() const { return size_; }
  [[nodiscard]] bool empty() const { return size_ == 0; }
  [[nodiscard]] Iterator begin() const { return {this, 0}; }
  [[nodiscard]] Iterator end() const { return {this, size_}; }
  [[nodiscard]] T operator[](std::size_t index) const {
    if (index >= size_) throw std::out_of_range("affine sequence index");
    auto it = std::upper_bound(runs_.begin(), runs_.end(), index,
        [](auto i, const Run& run) { return i < run.begin; });
    const auto& run = *std::prev(it);
    return Codec::advance(run.first, run.delta, index - run.begin);
  }
  [[nodiscard]] T back() const { return (*this)[size_ - 1]; }
  [[nodiscard]] const std::vector<Run>& runs() const { return runs_; }
  void push_back(T value) { append(std::move(value), Delta{}, 1); }
  void append(T first, Delta delta, std::size_t count) {
    if (!count) return;
    if (!runs_.empty()) {
      auto& run = runs_.back();
      if (run.count == 1) {
        const auto candidate = Codec::difference(run.first, first);
        if (candidate && (count == 1 || *candidate == delta)) {
          run.delta = *candidate; run.count += count; size_ += count; return;
        }
      } else if ((count == 1 || run.delta == delta) &&
                 Codec::advance(run.first, run.delta, run.count) == first) {
        run.count += count; size_ += count; return;
      }
    }
    runs_.push_back({size_, count, std::move(first), delta});
    size_ += count;
  }
  void pop_back() {
    if (empty()) throw std::out_of_range("empty affine sequence");
    if (--runs_.back().count == 0) runs_.pop_back();
    --size_;
  }
 private:
  std::vector<Run> runs_;
  std::size_t size_ = 0;
};

}  // namespace sentryward::algorithm
