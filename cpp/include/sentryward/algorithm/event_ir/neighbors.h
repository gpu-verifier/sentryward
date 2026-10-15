#pragma once

#include <span>
#include "sentryward/algorithm/event_ir/event_relations.h"

namespace sentryward::algorithm {

// A borrowed adjacency view; affine neighbors are generated only on iteration.
class EventNeighbors final {
 public:
  EventNeighbors() = default;
  EventNeighbors(std::span<const EventId> row) : row_(row) {}
  EventNeighbors(const std::vector<EventId>& row) : row_(row) {}
  EventNeighbors(const EventRelations& topology, EventId event, bool reverse)
      : topology_(&topology), reverse_(reverse) {
    const auto [family, member] = topology.layout()->locate(event);
    member_ = member;
    relations_ = reverse ? topology.incoming(family) : topology.outgoing(family);
  }
  class Iterator {
   public:
    using value_type = EventId;
    using reference = EventId;
    using pointer = void;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;
    Iterator() = default;
    Iterator(EventNeighbors view, bool end)
        : topology_(view.topology_), relations_(view.relations_), row_(view.row_),
          member_(view.member_), reverse_(view.reverse_) {
      relation_ = end ? (topology_ ? relations_.size() : row_.size()) : 0;
      if (topology_ && !end) seek();
    }
    EventId operator*() const {
      return topology_ ? EventId(first_ + stride_ * offset_) : row_[relation_];
    }
    Iterator& operator++() {
      if (!topology_) { ++relation_; return *this; }
      if (++offset_ == count_) { ++relation_; seek(); }
      return *this;
    }
    Iterator operator++(int) { auto old = *this; ++*this; return old; }
    friend bool operator==(const Iterator& a, const Iterator& b) {
      return a.topology_ == b.topology_ && a.relations_.sameStorage(b.relations_) &&
          a.row_.data() == b.row_.data() && a.relation_ == b.relation_ &&
          (!a.topology_ || a.relation_ == a.relations_.size() || a.offset_ == b.offset_);
    }
   private:
    void seek() {
      offset_ = 0;
      while (relation_ < relations_.size()) {
        const auto& r = topology_->relations()[relations_[relation_]];
        const auto from = reverse_ ? r.first_target : r.first_source;
        const auto to = reverse_ ? r.first_source : r.first_target;
        const bool own_step = reverse_ ? r.target_step : r.source_step;
        const bool neighbor_step = reverse_ ? r.source_step : r.target_step;
        if (member_ >= from && member_ < from + (own_step ? r.count : 1)) {
          const auto& family = topology_->layout()->families()[reverse_ ? r.source : r.target];
          first_ = family.first.value() + family.stride * (to + (own_step && neighbor_step ? member_ - from : 0));
          stride_ = neighbor_step ? family.stride : 0;
          count_ = own_step ? 1 : r.count;
          return;
        }
        ++relation_;
      }
    }
    const EventRelations* topology_ = nullptr;
    RelationSpan<std::size_t> relations_;
    std::span<const EventId> row_;
    std::size_t member_ = 0, relation_ = 0, offset_ = 0, count_ = 0;
    std::uint32_t first_ = 0, stride_ = 0;
    bool reverse_ = false;
  };
  Iterator begin() const { return {*this, false}; }
  Iterator end() const { return {*this, true}; }
  bool sharesStorageWith(const EventNeighbors& other) const {
    return topology_ == other.topology_ && relations_.sameStorage(other.relations_) &&
        row_.data() == other.row_.data();
  }
  [[nodiscard]] bool empty() const { return begin() == end(); }
  [[nodiscard]] std::size_t size() const {
    if (!topology_) return row_.size();
    std::size_t size = 0;
    for (auto index : relations_) {
      const auto& r = topology_->relations()[index];
      const auto first = reverse_ ? r.first_target : r.first_source;
      const bool step = reverse_ ? r.target_step : r.source_step;
      if (member_ >= first && member_ < first + (step ? r.count : 1)) size += step ? 1 : r.count;
    }
    return size;
  }
  EventId operator[](std::size_t index) const {
    if (!topology_) return row_[index];
    for (auto relation : relations_) {
      const auto& r = topology_->relations()[relation];
      const auto from = reverse_ ? r.first_target : r.first_source;
      const auto to = reverse_ ? r.first_source : r.first_target;
      const bool own_step = reverse_ ? r.target_step : r.source_step;
      const bool neighbor_step = reverse_ ? r.source_step : r.target_step;
      if (member_ < from || member_ >= from + (own_step ? r.count : 1)) continue;
      const auto count = own_step ? 1 : r.count;
      if (index >= count) { index -= count; continue; }
      const auto& family = topology_->layout()->families()[reverse_ ? r.source : r.target];
      const auto offset = neighbor_step ? (own_step ? member_ - from : index) : 0;
      return EventId(family.first.value() + family.stride * (to + offset));
    }
    throw std::out_of_range("event neighbor index");
  }
  EventId front() const { return *begin(); }
 private:
  const EventRelations* topology_ = nullptr;
  RelationSpan<std::size_t> relations_;
  std::span<const EventId> row_;
  std::size_t member_ = 0;
  bool reverse_ = false;
};

}  // namespace sentryward::algorithm

template<>
inline constexpr bool std::ranges::enable_borrowed_range<sentryward::algorithm::EventNeighbors> = true;
