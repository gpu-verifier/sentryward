#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace sentryward::algorithm {

// A mostly dense strong-ID map. A separate presence bitmap keeps every Value
// representable (including zero and UINT32_MAX) and distinguishes holes from
// stored values. Isolated large IDs use a tree instead of allocating up to that
// ID. Call reserveDense() when the complete contiguous ID domain is known.
//
// Read-only iteration remains ordered by ID, but produces value pairs rather
// than references to std::map nodes. Mutations invalidate iterators/references.
template <class Id, class Value>
class CompactIdMap final {
 public:
  static_assert(std::is_trivially_copyable_v<Value>,
                "compact ID indexes store scalar values without owned buffers");
  using value_type = std::pair<const Id, Value>;

  CompactIdMap() = default;
  CompactIdMap(const CompactIdMap&) = default;
  CompactIdMap& operator=(const CompactIdMap&) = default;
  CompactIdMap(CompactIdMap&& other) noexcept
      : values_(std::move(other.values_)), present_(std::move(other.present_)),
        sparse_(std::move(other.sparse_)), size_(std::exchange(other.size_, 0)) {}
  CompactIdMap& operator=(CompactIdMap&& other) noexcept {
    if (this != &other) {
      values_ = std::move(other.values_);
      present_ = std::move(other.present_);
      sparse_ = std::move(other.sparse_);
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }

  class const_iterator final {
   public:
    using iterator_category = std::input_iterator_tag;
    using value_type = typename CompactIdMap::value_type;
    using difference_type = std::ptrdiff_t;
    using reference = value_type;
    struct Arrow final {
      value_type value;
      const value_type* operator->() const noexcept { return &value; }
    };
    using pointer = Arrow;

    const_iterator() = default;
    [[nodiscard]] value_type operator*() const {
      if (position_ < owner_->values_.size()) {
        return {Id(static_cast<std::uint32_t>(position_)),
                owner_->values_[position_]};
      }
      return *sparse_;
    }
    [[nodiscard]] Arrow operator->() const { return {**this}; }
    const_iterator& operator++() {
      if (position_ < owner_->values_.size()) {
        ++position_;
        skipHoles();
      } else {
        ++sparse_;
      }
      return *this;
    }
    const_iterator operator++(int) {
      const_iterator previous = *this;
      ++*this;
      return previous;
    }
    friend bool operator==(const const_iterator& left,
                           const const_iterator& right) {
      if (left.owner_ != right.owner_) return false;
      if (!left.owner_) return true;
      return left.position_ == right.position_ && left.sparse_ == right.sparse_;
    }

   private:
    friend class CompactIdMap;
    using SparseIterator = typename std::map<Id, Value>::const_iterator;
    const_iterator(const CompactIdMap* owner, std::size_t position,
                   SparseIterator sparse)
        : owner_(owner), position_(position), sparse_(sparse) {}
    void skipHoles() {
      while (position_ < owner_->values_.size() &&
             !owner_->present(position_)) {
        ++position_;
      }
    }
    const CompactIdMap* owner_ = nullptr;
    std::size_t position_ = 0;
    SparseIterator sparse_;
  };

  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

  // Allocates once for callers with a known dense domain. New slots remain
  // absent; existing sparse entries entering the domain are migrated exactly.
  void reserveDense(std::size_t count) {
    if (count <= values_.size()) return;
    // Allocate presence first: if value allocation throws, the extra zero
    // words are harmless and every pre-existing lookup remains valid.
    present_.resize(count / 64 + (count % 64 != 0), 0);
    values_.resize(count);
    auto entry = sparse_.begin();
    while (entry != sparse_.end() && entry->first.value() < count) {
      const auto position = entry->first.value();
      values_[position] = entry->second;
      setPresent(position);
      entry = sparse_.erase(entry);
    }
  }

  [[nodiscard]] const Value* get(Id id) const noexcept {
    const std::size_t position = id.value();
    if (position < values_.size()) {
      return present(position) ? &values_[position] : nullptr;
    }
    const auto entry = sparse_.find(id);
    return entry == sparse_.end() ? nullptr : &entry->second;
  }
  [[nodiscard]] bool contains(Id id) const noexcept { return get(id) != nullptr; }
  [[nodiscard]] const Value& at(Id id) const {
    if (const auto* value = get(id)) return *value;
    throw std::out_of_range("compact ID map key is absent");
  }

  std::pair<const_iterator, bool> emplace(Id id, Value value) {
    const std::size_t position = id.value();
    if (position >= values_.size()) {
      // Geometric growth is bounded by density; one arbitrary sparse ID must
      // never create a multi-gigabyte array. Known graph domains are pre-sized.
      const std::size_t dense_budget = std::max<std::size_t>(64, 2 * (size_ + 1));
      if (position < dense_budget) {
        reserveDense(std::min(dense_budget,
                             std::max(position + 1, 2 * values_.size())));
      } else {
        const auto [entry, inserted] = sparse_.emplace(id, value);
        size_ += inserted;
        return {{this, values_.size(), entry}, inserted};
      }
    }
    const bool inserted = !present(position);
    if (inserted) {
      values_[position] = value;
      setPresent(position);
      ++size_;
    }
    return {{this, position, sparse_.begin()}, inserted};
  }

  std::size_t erase(Id id) {
    const std::size_t position = id.value();
    if (position < values_.size()) {
      if (!present(position)) return 0;
      present_[position / 64] &= ~(std::uint64_t{1} << (position % 64));
      --size_;
      return 1;
    }
    const auto removed = sparse_.erase(id);
    size_ -= removed;
    return removed;
  }

  [[nodiscard]] const_iterator begin() const {
    const_iterator result(this, 0, sparse_.begin());
    result.skipHoles();
    return result;
  }
  [[nodiscard]] const_iterator end() const {
    return {this, values_.size(), sparse_.end()};
  }
  [[nodiscard]] const_iterator find(Id id) const {
    const std::size_t position = id.value();
    if (position < values_.size()) {
      return present(position) ? const_iterator(this, position, sparse_.begin())
                               : end();
    }
    return {this, values_.size(), sparse_.find(id)};
  }

  // Read-only storage views for opt-in owned-memory accounting.
  [[nodiscard]] const std::vector<Value>& denseValues() const noexcept {
    return values_;
  }
  [[nodiscard]] const std::vector<std::uint64_t>& presenceWords() const noexcept {
    return present_;
  }
  [[nodiscard]] const std::map<Id, Value>& sparseValues() const noexcept {
    return sparse_;
  }

 private:
  [[nodiscard]] bool present(std::size_t position) const noexcept {
    return (present_[position / 64] >> (position % 64)) & 1;
  }
  void setPresent(std::size_t position) noexcept {
    present_[position / 64] |= std::uint64_t{1} << (position % 64);
  }
  std::vector<Value> values_;
  std::vector<std::uint64_t> present_;
  std::map<Id, Value> sparse_;
  std::size_t size_ = 0;
};

}  // namespace sentryward::algorithm
