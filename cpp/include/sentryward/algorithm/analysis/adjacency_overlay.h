#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sentryward/algorithm/event_ir/csr_adjacency.h"

namespace sentryward::algorithm {

// Unchanged rows share the source CSR; changed rows belong to each copy.
class AdjacencyOverlay final {
 public:
  enum class Direction { kPredecessors, kSuccessors };

  AdjacencyOverlay() = default;
  AdjacencyOverlay(std::shared_ptr<const CsrAdjacency> base,
                   Direction direction, std::size_t /*capacity*/)
      : base_(std::move(base)), direction_(direction),
        size_(base_ ? base_->eventCount() : 0) {}

  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] std::size_t ownedRowCount() const noexcept { return rows_.size(); }
  [[nodiscard]] EventNeighbors at(std::size_t index) const {
    checkIndex(index);
    const auto overlay = rows_.find(index);
    if (overlay != rows_.end()) return overlay->second;
    if (!base_ || index >= base_->eventCount()) return {};
    const EventId id(static_cast<std::uint32_t>(index));
    return direction_ == Direction::kPredecessors ? base_->predecessors(id)
                                                 : base_->successors(id);
  }
  [[nodiscard]] EventNeighbors operator[](std::size_t index) const {
    return at(index);
  }
  void emplace_back() {
    if (size_ == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("HB adjacency size exhausted");
    ++size_;
  }

  void replace(std::size_t index, std::vector<EventId> row) {
    checkIndex(index);
    rows_.insert_or_assign(index, std::move(row));
  }

  void append(std::size_t index, EventId neighbor) {
    checkIndex(index);
    const auto overlay = rows_.find(index);
    if (overlay != rows_.end()) {
      overlay->second.push_back(neighbor);
      return;
    }
    const auto base = at(index);
    std::vector<EventId> row;
    row.reserve(base.size() + 1);
    row.insert(row.end(), base.begin(), base.end());
    row.push_back(neighbor);
    replace(index, std::move(row));
  }

 private:
  friend class MemoryCensus;
  void checkIndex(std::size_t index) const {
    if (index >= size_) throw std::out_of_range("HB adjacency row");
  }
  std::shared_ptr<const CsrAdjacency> base_;
  Direction direction_ = Direction::kPredecessors;
  std::size_t size_ = 0;
  std::unordered_map<std::size_t, std::vector<EventId>> rows_;
};

}  // namespace sentryward::algorithm
