#include "sentryward/algorithm/analysis/shared_reachability_index.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace sentryward::algorithm {
namespace {

using IndexState = SharedReachabilityIndex::StateId;
constexpr IndexState kEmptySlot = std::numeric_limits<IndexState>::max();

std::uint64_t hashRow(std::span<const EventId> row) {
  std::uint64_t hash = 14695981039346656037ULL;
  for (EventId event : row) {
    hash ^= event.value();
    hash *= 1099511628211ULL;
  }
  hash ^= row.size();
  // Avalanche the low bits used by the power-of-two table.
  hash ^= hash >> 30;
  hash *= 0xbf58476d1ce4e5b9ULL;
  hash ^= hash >> 27;
  hash *= 0x94d049bb133111ebULL;
  return hash ^ (hash >> 31);
}

IndexState checkedState(std::size_t value) {
  if (value >= kEmptySlot) {
    throw std::length_error("shared reachability state ID overflow");
  }
  return static_cast<IndexState>(value);
}

}  // namespace

SharedReachabilityIndex::SharedReachabilityIndex(
    const HappensBeforeGraph& graph, bool reverse, RowHasher hasher)
    : graph_(&graph),
      reverse_(reverse),
      observed_version_(graph.version()),
      observed_edge_count_(graph.edgeCount()) {
  build(graph.eventCount(), [this](EventId event) { return adjacency(event); },
        hasher);
}

SharedReachabilityIndex::SharedReachabilityIndex(
    std::size_t count, NeighborProvider neighbors, RowHasher hasher) {
  build(count, neighbors, hasher);
}

void SharedReachabilityIndex::build(std::size_t count,
                                    const NeighborProvider& neighbors,
                                    RowHasher hasher) {
  if (count > std::numeric_limits<std::uint32_t>::max()) {
    throw std::length_error("shared reachability event ID overflow");
  }
  state_of_.resize(count);
  row_offsets_.push_back(0);
  if (hasher == nullptr) hasher = hashRow;
  // Only one canonical row is copied per group. The hash table and its hash
  // cache are construction-only, keeping lookup memory out of every query.
  {
    std::vector<IndexState> table(16, kEmptySlot);
    std::vector<std::uint64_t> hashes;
    std::vector<EventId> row;
    for (std::size_t index = 0; index < count; ++index) {
      const auto adjacent = neighbors(EventId(static_cast<std::uint32_t>(index)));
      row.assign(adjacent.begin(), adjacent.end());
      std::sort(row.begin(), row.end());
      row.erase(std::unique(row.begin(), row.end()), row.end());
      if (!row.empty() && row.back().value() >= count) {
        throw std::invalid_argument("shared reachability neighbor is invalid");
      }
      const auto hash = hasher(row);
      std::size_t slot = hash & (table.size() - 1);
      while (table[slot] != kEmptySlot) {
        const auto state = table[slot];
        if (hashes[state] == hash &&
            std::ranges::equal(row, Neighbors(state))) {
          break;
        }
        slot = (slot + 1) & (table.size() - 1);
      }
      if (table[slot] == kEmptySlot) {
        const auto state = checkedState(baseStateCount());
        table[slot] = state;
        row_neighbors_.insert(row_neighbors_.end(), row.begin(), row.end());
        row_offsets_.push_back(row_neighbors_.size());
        hashes.push_back(hash);
        state_of_[index] = state;

        // Rehash at 75% occupancy. Cached hashes make this linear in states,
        // independent of the degree of the representative rows.
        if (hashes.size() >= table.size() - table.size() / 4) {
          if (table.size() > table.max_size() / 2) {
            throw std::length_error("shared reachability hash table overflow");
          }
          table.assign(table.size() * 2, kEmptySlot);
          for (std::size_t index = 0; index < hashes.size(); ++index) {
            std::size_t destination = hashes[index] & (table.size() - 1);
            while (table[destination] != kEmptySlot) {
              destination = (destination + 1) & (table.size() - 1);
            }
            table[destination] = static_cast<IndexState>(index);
          }
        }
      } else {
        state_of_[index] = table[slot];
      }
    }
  }
  row_offsets_.shrink_to_fit();
  row_neighbors_.shrink_to_fit();
}

SharedReachabilityIndex::StateId SharedReachabilityIndex::StateFor(
    EventId event) const {
  return state_of_.at(event.value());
}

std::size_t SharedReachabilityIndex::StateCount() const noexcept {
  return baseStateCount() + singletons_.size();
}

EventNeighbors SharedReachabilityIndex::Neighbors(StateId state) const {
  if (state >= baseStateCount()) {
    return adjacency(singletons_.at(state - baseStateCount()));
  }
  const std::size_t begin = row_offsets_[state];
  return std::span<const EventId>(row_neighbors_).subspan(
      begin, row_offsets_[state + 1] - begin);
}

void SharedReachabilityIndex::SyncGraphChanges() {
  if (graph_ == nullptr || observed_version_ == graph_->version()) return;
  if (graph_->eventCount() < state_of_.size() ||
      graph_->edgeCount() < observed_edge_count_) {
    throw std::logic_error("shared reachability requires an append-only graph");
  }
  while (state_of_.size() < graph_->eventCount()) {
    const auto event = EventId(static_cast<std::uint32_t>(state_of_.size()));
    const auto state = appendSingleton(event);
    state_of_.push_back(state);
  }
  for (std::size_t i=observed_edge_count_;i<graph_->edgeCount();++i) {
    const auto edge=graph_->edges()[i];
    const EventId changed = reverse_ ? edge.to : edge.from;
    auto& state = state_of_[changed.value()];
    // Adjacencies are deduplicated and append-only. Equal degree therefore
    // proves that a new provenance record did not change this original row.
    if (state < baseStateCount() &&
        adjacency(changed).size() != row_offsets_[state + 1] - row_offsets_[state]) {
      state = appendSingleton(changed);
    }
  }
  observed_edge_count_ = graph_->edgeCount();
  observed_version_ = graph_->version();
}

bool SharedReachabilityIndex::reverse() const noexcept { return reverse_; }

std::size_t SharedReachabilityIndex::baseStateCount() const noexcept {
  return row_offsets_.size() - 1;
}

std::size_t SharedReachabilityIndex::storedNeighborCount() const noexcept {
  return row_neighbors_.size();
}

std::size_t SharedReachabilityIndex::storageBytes() const noexcept {
  return state_of_.capacity() * sizeof(StateId) +
         row_offsets_.capacity() * sizeof(std::size_t) +
         row_neighbors_.capacity() * sizeof(EventId) +
         singletons_.capacity() * sizeof(EventId);
}

EventNeighbors SharedReachabilityIndex::adjacency(EventId event) const {
  return reverse_ ? graph_->predecessors(event) : graph_->successors(event);
}

SharedReachabilityIndex::StateId SharedReachabilityIndex::appendSingleton(
    EventId event) {
  const auto state = checkedState(StateCount());
  singletons_.push_back(event);
  return state;
}

}  // namespace sentryward::algorithm
