#pragma once

#include "sentryward/algorithm/event_ir/event_families.h"

namespace sentryward::algorithm {

// Piecewise-constant values on independent members of event families.
template<class T>
class FamilyValues final {
  struct Run { std::size_t end; T value; };
  using Rows = std::map<std::size_t, Run>;
  using Groups = std::map<std::size_t, Rows>;
 public:
  void reset(std::shared_ptr<const EventFamilies> layout) { layout_ = std::move(layout); }
  [[nodiscard]] std::size_t size() const { return size_; }
  [[nodiscard]] bool empty() const { return size_ == 0; }
  [[nodiscard]] std::size_t runCount() const {
    std::size_t n = 0; for (const auto& [family, rows] : groups_) n += rows.size(); return n;
  }
  [[nodiscard]] std::optional<T> get(EventId id) const {
    if (!layout_ || id.value() >= layout_->eventCount()) return {};
    const auto [family, member] = layout_->locate(id);
    auto group = groups_.find(family);
    if (group == groups_.end()) return {};
    auto it = group->second.upper_bound(member);
    if (it == group->second.begin()) return {};
    --it;
    if (it->second.end <= member) return {};
    return it->second.value;
  }
  [[nodiscard]] std::size_t uniformPrefix(EventId id, std::size_t count) const {
    const auto [family, member] = layout_->locate(id);
    const auto group = groups_.find(family);
    if (group == groups_.end()) return 0;
    auto it = group->second.upper_bound(member);
    if (it == group->second.begin()) return 0;
    --it;
    return it->second.end <= member ? 0 : std::min(count, it->second.end - member);
  }
  bool emplace(EventId id, T value) {
    if (get(id)) return false;
    insert(id, 1, std::move(value)); return true;
  }
  void insert(EventId first, std::size_t count, T value) {
    if (!count) return;
    const auto [family, member] = layout_->locate(first);
    if (member + count > layout_->families()[family].count)
      throw std::out_of_range("family value domain");
    auto& rows = groups_[family];
    auto next = rows.lower_bound(member);
    if ((next != rows.end() && next->first < member + count) ||
        (next != rows.begin() && std::prev(next)->second.end > member))
      throw std::logic_error("overlapping family value definitions");
    auto it = rows.emplace_hint(next, member, Run{member + count, std::move(value)});
    if (it != rows.begin()) {
      auto previous = std::prev(it);
      if (previous->second.end == member && previous->second.value == it->second.value) {
        previous->second.end = it->second.end; rows.erase(it); it = previous;
      }
    }
    next = std::next(it);
    if (next != rows.end() && it->second.end == next->first && it->second.value == next->second.value) {
      it->second.end = next->second.end; rows.erase(next);
    }
    size_ += count;
  }
  class Iterator {
   public:
    using value_type = std::pair<EventId, T>;
    using reference = value_type;
    using pointer = void;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;
    Iterator() = default;
    Iterator(const FamilyValues* owner, bool end) : owner_(owner), group_(end ? owner->groups_.end() : owner->groups_.begin()) {
      if (group_ != owner_->groups_.end()) { row_ = group_->second.begin(); member_ = row_->first; }
    }
    value_type operator*() const {
      const auto& family = owner_->layout_->families()[group_->first];
      return {EventId(family.first.value() + family.stride * member_), row_->second.value};
    }
    Iterator& operator++() {
      if (++member_ != row_->second.end) return *this;
      if (++row_ == group_->second.end()) {
        if (++group_ == owner_->groups_.end()) return *this;
        row_ = group_->second.begin();
      }
      member_ = row_->first; return *this;
    }
    Iterator operator++(int) { auto old = *this; ++*this; return old; }
    friend bool operator==(const Iterator& a, const Iterator& b) {
      return a.owner_ == b.owner_ && (!a.owner_ || (a.group_ == b.group_ &&
          (a.atEnd() || (a.row_ == b.row_ && a.member_ == b.member_))));
    }
   private:
    const FamilyValues* owner_ = nullptr;
    typename Groups::const_iterator group_;
    typename Rows::const_iterator row_;
    std::size_t member_ = 0;
    bool atEnd() const { return group_ == owner_->groups_.end(); }
  };
  Iterator begin() const { return {this, false}; }
  Iterator end() const { return {this, true}; }
  friend bool operator==(const FamilyValues& a, const FamilyValues& b) {
    if (a.size() != b.size()) return false;
    for (const auto& [id, value] : a) if (b.get(id) != value) return false;
    return true;
  }
 private:
  std::shared_ptr<const EventFamilies> layout_;
  Groups groups_;
  std::size_t size_ = 0;
};

}  // namespace sentryward::algorithm
