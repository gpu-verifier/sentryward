#include "sentryward/algorithm/memory/conflict_order.h"

#include <algorithm>
#include <queue>
#include <set>
#include <tuple>

namespace sentryward::algorithm {
namespace {

// Sweep spatial endpoints, then traverse each nonempty slab in event order.
// For each slab: consecutive writes form a chain; every read must follow its
// preceding write and precede its following write. Transitivity covers all other
// conflicting pairs. Read/read ordering is deliberately not required.
// Cost is O(n log n + H), where H counts access/slab incidences, plus HB queries.
// Partial overlaps can increase H; equal intervals require only one traversal.
class ConflictOrder final : public MemoryConflictCandidates {
  using Coordinate = std::pair<bool, std::uint64_t>;  // Carry + low 64 bits.
  struct ByEvent {
    const std::vector<ConstantConflictRegion>* rows;
    std::span<const std::size_t> ranks;
    bool operator()(std::size_t a, std::size_t b) const {
      const auto& x = (*rows)[a];
      const auto& y = (*rows)[b];
      // A write represents all accesses at the same event in this slab.
      return std::tuple(ranks[x.event.value()], !x.write, x.access) <
             std::tuple(ranks[y.event.value()], !y.write, y.access);
    }
  };
  using Active = std::set<std::size_t, ByEvent>;

 public:
  ConflictOrder(std::vector<ConstantConflictRegion> rows,
                std::span<const std::size_t> ranks)
      : rows_(std::move(rows)), active_(ByEvent{&rows_, ranks}),
        writes_(ByEvent{&rows_, ranks}), ranks_(ranks) {
    std::sort(rows_.begin(), rows_.end(), [](const auto& a, const auto& b) {
      return std::tie(a.group, a.offset, a.access) < std::tie(b.group, b.offset, b.access);
    });
  }

  std::optional<MemoryConflictPair> next() override {
    if (pending_) return std::exchange(pending_, std::nullopt);
    for (;;) {
      if (!in_slab_ || cursor_ == active_.end()) {
        if (!advanceSlab()) return std::nullopt;
      }
      const auto& row = rows_[*cursor_];
      // Multiple operands of one event are not races with each other.
      do { ++cursor_; }
      while (cursor_ != active_.end() && rows_[*cursor_].event == row.event);
      while (next_write_ != writes_.end() &&
             ranks_[rows_[*next_write_].event.value()] <= ranks_[row.event.value()])
        ++next_write_;

      const auto previous = last_write_;
      if (row.write) {
        last_write_ = row.access;
      } else if (next_write_ != writes_.end()) {
        pending_ = MemoryConflictPair{row.access, rows_[*next_write_].access};
      }
      if (previous) return MemoryConflictPair{*previous, row.access};
      if (pending_) return std::exchange(pending_, std::nullopt);
    }
  }

 private:
  bool advanceSlab() {
    for (;;) {
      if (active_.empty() && ends_.empty()) {
        if (position_ == rows_.size()) return false;
        group_ = rows_[position_].group;
      }
      std::optional<Coordinate> point;
      if (position_ < rows_.size() && rows_[position_].group == group_)
        point = Coordinate{false, rows_[position_].offset};
      if (!ends_.empty() && (!point || ends_.top().first < *point))
        point = ends_.top().first;
      if (!point) return false;
      while (!ends_.empty() && ends_.top().first == *point) {
        const auto index = ends_.top().second;
        active_.erase(index);
        writes_.erase(index);
        ends_.pop();
      }
      while (position_ < rows_.size() && rows_[position_].group == group_ &&
             Coordinate{false, rows_[position_].offset} == *point) {
        const auto& row = rows_[position_];
        active_.insert(position_);
        if (row.write) writes_.insert(position_);
        const auto end = row.offset + row.bytes;
        ends_.emplace(Coordinate{end < row.offset, end}, position_);
        ++position_;
      }
      if (active_.empty() || writes_.empty()) continue;
      cursor_ = active_.begin();
      next_write_ = writes_.begin();
      last_write_.reset();
      in_slab_ = true;
      return true;
    }
  }

  std::vector<ConstantConflictRegion> rows_;
  Active active_, writes_;
  std::span<const std::size_t> ranks_;
  Active::const_iterator cursor_, next_write_;
  using End = std::pair<Coordinate, std::size_t>;
  std::priority_queue<End, std::vector<End>, std::greater<End>> ends_;
  std::size_t position_ = 0, group_ = 0;
  bool in_slab_ = false;
  std::optional<AccessId> last_write_;
  std::optional<MemoryConflictPair> pending_;
};
}  // namespace

std::unique_ptr<MemoryConflictCandidates> MakeConflictOrder(
    std::vector<ConstantConflictRegion> regions,
    std::span<const std::size_t> ranks) {
  return std::make_unique<ConflictOrder>(std::move(regions), ranks);
}
}  // namespace sentryward::algorithm
