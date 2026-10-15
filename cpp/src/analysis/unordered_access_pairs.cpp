#include "sentryward/algorithm/analysis/unordered_access_pairs.h"

#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <variant>

namespace sentryward::algorithm {
namespace {

[[nodiscard]] std::vector<EventId> topologicalOrder(
    const HappensBeforeGraph& graph) {
  std::vector<std::size_t> pending(graph.eventCount());
  std::vector<EventId> order;
  order.reserve(graph.eventCount());
  for (const Event& event : graph.events()) {
    pending[event.id.value()] = graph.predecessors(event.id).size();
    if (pending[event.id.value()] == 0) {
      order.push_back(event.id);
    }
  }
  // FIFO Kahn traversal keeps the preprocessing linear in graph size.
  for (std::size_t cursor = 0; cursor < order.size(); ++cursor) {
    for (EventId successor : graph.successors(order[cursor])) {
      if (--pending[successor.value()] == 0) {
        order.push_back(successor);
      }
    }
  }
  if (order.size() != graph.eventCount()) {
    throw std::invalid_argument(
        "unordered access pairs require an acyclic graph");
  }
  return order;
}

[[nodiscard]] bool validDirection(MemoryDirection direction) {
  return direction == MemoryDirection::kLoad ||
         direction == MemoryDirection::kStore;
}

[[nodiscard]] std::size_t validateAccess(const HappensBeforeGraph& graph,
                                         const MemoryAccess& access) {
  if (!graph.contains(access.event)) {
    throw std::invalid_argument("memory access references an unknown event");
  }
  if (access.mode != AccessMode::kRead && access.mode != AccessMode::kWrite) {
    throw std::invalid_argument("memory access has an invalid mode");
  }
  const Event& event = graph.event(access.event);
  if (!event.origin.has_value() || *event.origin != access.instruction ||
      !event.actor.has_value() || *event.actor != access.actor) {
    throw std::invalid_argument("memory access provenance differs from event");
  }
  if (event.scheduling_class != EventSchedulingClass::kNormal) {
    throw std::invalid_argument("memory execution must be a normal event");
  }

  if (std::holds_alternative<OperationCompleteEvent>(event.payload)) {
    if (access.operand != AccessOperand::kImmediate || access.effect_index != 0 || access.access_index != 0)
      throw std::invalid_argument("compute access differs from completion endpoint");
    return 0;
  }

  if (const auto* immediate =
          std::get_if<ImmediateMemoryExecEvent>(&event.payload)) {
    if (!validDirection(immediate->direction) ||
        access.operand != AccessOperand::kImmediate ||
        access.mode != (immediate->direction == MemoryDirection::kLoad
                            ? AccessMode::kRead
                            : AccessMode::kWrite) ||
        access.address != MemoryEndpoint{immediate->address} ||
        access.bytes != immediate->bytes || access.effect_index != 0 ||
        access.access_index != 0) {
      throw std::invalid_argument(
          "memory access differs from immediate operand");
    }
    return 0;
  }
  if (const auto* asynchronous =
          std::get_if<AsyncMemoryExecEvent>(&event.payload)) {
    if (!validDirection(asynchronous->direction) ||
        access.bytes != asynchronous->bytes || access.effect_index != 0 ||
        access.access_index != 0) {
      throw std::invalid_argument("memory access differs from async operand");
    }
    if (access.operand == AccessOperand::kSource &&
        access.mode == AccessMode::kRead &&
        access.address == MemoryEndpoint{asynchronous->source}) {
      return 0;
    }
    if (access.operand == AccessOperand::kDestination &&
        access.mode == AccessMode::kWrite &&
        access.address == MemoryEndpoint{asynchronous->destination}) {
      return 1;
    }
    throw std::invalid_argument("memory access differs from async operand");
  }
  if (const auto* generic =
          std::get_if<GenericAsyncEffectEvent>(&event.payload)) {
    if (generic->detached_accesses) {
      if (access.effect_index != generic->effect_index)
        throw std::invalid_argument("compute access differs from effect endpoint");
      if (generic->detached_transfer) {
        if (access.access_index != 0) throw std::invalid_argument("invalid transfer access index");
        if (access.operand == AccessOperand::kSource && access.mode == AccessMode::kRead) return 0;
        if (access.operand == AccessOperand::kDestination && access.mode == AccessMode::kWrite) return 1;
        throw std::invalid_argument("compute access differs from transfer endpoint");
      }
      if (access.operand != AccessOperand::kEffect || access.access_index >= generic->detached_access_count)
        throw std::invalid_argument("compute access differs from effect operand index");
      return access.access_index;
    }
    if (access.operand != AccessOperand::kEffect ||
        access.effect_index != generic->effect_index ||
        access.access_index >= generic->accesses.size()) {
      throw std::invalid_argument(
          "memory access differs from generic effect index");
    }
    const MemoryEffect& operand = generic->accesses[access.access_index];
    if (access.mode != operand.mode || access.address != operand.endpoint ||
        access.bytes != operand.bytes || access.atomic_scope != operand.atomic_scope) {
      throw std::invalid_argument(
          "memory access differs from generic effect operand");
    }
    if (const auto* reg = std::get_if<RegisterAddress>(&access.address);
        reg && reg->name.empty()) {
      throw std::invalid_argument("register endpoint has an empty name");
    }
    return access.access_index;
  }
  throw std::invalid_argument("memory access does not reference an execution");
}

}  // namespace

UnorderedAccessPairs::UnorderedAccessPairs(
    const HappensBeforeGraph& graph, MemoryAccessView accesses,
    const UnorderedAccessPairsOptions& options,
    std::unique_ptr<MemoryStorageFilter> storage_filter)
    : graph_(graph),
      graph_version_(graph.version()),
      graph_event_count_(graph.eventCount()),
      graph_edge_count_(graph.edgeCount()),
      event_accesses_(graph.eventCount()),
      writes_(accesses.size()),
      seen_all_(accesses.size()),
      seen_writes_(accesses.size()),
      write_candidates_(accesses.size()),
      read_candidates_(accesses.size()),
      storage_filter_(std::move(storage_filter)),
      compatible_candidates_(accesses.size()) {
  const auto order = topologicalOrder(graph);

  std::vector<std::vector<bool>> operands(graph.eventCount());
  for (const Event& event : graph.events()) {
    std::size_t count = 0;
    if (std::holds_alternative<ImmediateMemoryExecEvent>(event.payload) ||
        std::holds_alternative<OperationCompleteEvent>(event.payload))
      count = 1;
    else if (std::holds_alternative<AsyncMemoryExecEvent>(event.payload))
      count = 2;
    else if (const auto* generic =
                 std::get_if<GenericAsyncEffectEvent>(&event.payload))
      count = generic->detached_accesses ? generic->detached_access_count : generic->accesses.size();
    operands[event.id.value()].resize(count, false);
  }
  for (std::size_t index = 0; index < accesses.size(); ++index) {
    const MemoryAccess& access = accesses[index];
    if (access.id.value() != index) {
      throw std::invalid_argument("memory access IDs must equal table indexes");
    }
    const std::size_t operand = validateAccess(graph, access);
    auto& seen = operands[access.event.value()];
    if (seen[operand]) {
      throw std::invalid_argument("duplicate memory execution operand");
    }
    seen[operand] = true;
    event_accesses_[access.event.value()].push_back(access.id);
    writes_[index] = access.mode == AccessMode::kWrite;
  }
  for (const Event& event : graph.events()) {
    for (bool seen : operands[event.id.value()]) {
      if (!seen)
        throw std::invalid_argument(
            "memory access table omits execution operands");
    }
  }

  const std::size_t row_bytes =
      (accesses.size() / 64 + (accesses.size() % 64 != 0)) * sizeof(std::uint64_t);
  const std::size_t row_storage = row_bytes + sizeof(DynamicBitset);
  lazy_ancestors_ = graph.eventCount() > options.ancestor_budget_bytes / row_storage;
  if (lazy_ancestors_) {
    current_ancestors_.resize(accesses.size());
    visited_.resize(graph.eventCount());
    ancestor_row_bytes_ = row_bytes;
    const std::size_t container_bytes = sizeof(ancestor_cache_);
    const std::size_t entry_bytes = row_bytes + sizeof(AncestorCacheEntry);
    if (options.ancestor_budget_bytes > container_bytes) {
      ancestor_cache_capacity_ = std::min(
          graph.eventCount(),
          (options.ancestor_budget_bytes - container_bytes) / entry_bytes);
    }
    if (ancestor_cache_capacity_ != 0) {
      ancestor_cache_.resize(ancestor_cache_capacity_);
      ancestor_cache_slots_.assign(graph.eventCount(), DynamicBitset::npos);
    }
  } else {
    ancestors_.reserve(graph.eventCount());
    for (std::size_t index = 0; index < graph.eventCount(); ++index) {
      ancestors_.emplace_back(accesses.size());
    }
  }
  for (EventId event : order) {
    if (!lazy_ancestors_) {
      DynamicBitset& ancestors = ancestors_[event.value()];
      // Nonmemory vertices carry paths between memory executions too.
      for (EventId predecessor : graph.predecessors(event)) {
        ancestors |= ancestors_[predecessor.value()];
      }
      for (AccessId access : event_accesses_[event.value()]) {
        ancestors.set(access.value());
      }
    }
    if (!event_accesses_[event.value()].empty()) {
      memory_order_.push_back(event);
    }
  }
  checkGraphUnchanged();
}

std::size_t UnorderedAccessPairs::retainedAncestorCacheBytes() const noexcept {
  if (ancestor_cache_capacity_ == 0) return 0;
  return sizeof(ancestor_cache_) +
         ancestor_cache_.capacity() * sizeof(AncestorCacheEntry) +
         ancestor_cache_size_ * ancestor_row_bytes_;
}

void UnorderedAccessPairs::checkGraphUnchanged() const {
  if (graph_.version() != graph_version_ ||
      graph_.eventCount() != graph_event_count_ ||
      graph_.edgeCount() != graph_edge_count_) {
    throw std::logic_error("graph changed during unordered access enumeration");
  }
}

void UnorderedAccessPairs::prepareEvent() {
  const EventId event = memory_order_[event_cursor_];
  event_ancestors_ = nullptr;
  if (!storage_filter_) {
    const DynamicBitset& ancestors = ancestorsOf(event);
    write_candidates_ = seen_all_;
    write_candidates_.subtract(ancestors);
    read_candidates_ = seen_writes_;
    read_candidates_.subtract(ancestors);
  }
  access_cursor_ = 0;
  prepareAccess();
  event_prepared_ = true;
}

void UnorderedAccessPairs::prepareAccess() {
  if (storage_filter_) {
    const auto access = event_accesses_[memory_order_[event_cursor_].value()][access_cursor_];
    compatible_candidates_ = writes_[access.value()] ? seen_all_ : seen_writes_;
    storage_filter_->retainCompatible(access, compatible_candidates_);
    if (compatible_candidates_.any()) {
      if (!event_ancestors_) event_ancestors_ = &ancestorsOf(memory_order_[event_cursor_]);
      compatible_candidates_.subtract(*event_ancestors_);
    }
  }
  next_candidate_ = currentCandidates().find_first();
}

const DynamicBitset& UnorderedAccessPairs::ancestorsOf(EventId event) {
  if (!lazy_ancestors_) {
    return ancestors_[event.value()];
  }
  current_ancestors_.reset();
  traversal_.clear();
  traversal_.push_back(event);
  visited_.set(event.value());
  for (std::size_t cursor = 0; cursor < traversal_.size(); ++cursor) {
    const EventId current = traversal_[cursor];
    if (ancestor_cache_capacity_ != 0) {
      const auto cached = ancestor_cache_slots_[current.value()];
      if (cached != DynamicBitset::npos) {
        current_ancestors_ |= ancestor_cache_[cached].ancestors;
        continue;
      }
    }
    for (AccessId access : event_accesses_[current.value()]) {
      current_ancestors_.set(access.value());
    }
    for (EventId predecessor : graph_.predecessors(current)) {
      if (!visited_.test(predecessor.value())) {
        visited_.set(predecessor.value());
        traversal_.push_back(predecessor);
      }
    }
  }
  for (EventId visited : traversal_) {
    visited_.reset(visited.value());
  }
  if (ancestor_cache_capacity_ != 0) {
    auto& entry = ancestor_cache_[ancestor_cache_next_];
    if (ancestor_cache_size_ == ancestor_cache_capacity_)
      ancestor_cache_slots_[entry.event.value()] = DynamicBitset::npos;
    else
      ++ancestor_cache_size_;
    entry.event = event;
    entry.ancestors = current_ancestors_;
    ancestor_cache_slots_[event.value()] = ancestor_cache_next_;
    ancestor_cache_next_ =
        (ancestor_cache_next_ + 1) % ancestor_cache_capacity_;
  }
  return current_ancestors_;
}

void UnorderedAccessPairs::finishEvent() {
  const EventId event = memory_order_[event_cursor_];
  // Insert a whole execution at once, excluding its own operand pairs.
  for (AccessId access : event_accesses_[event.value()]) {
    seen_all_.set(access.value());
    if (writes_[access.value()]) {
      seen_writes_.set(access.value());
    }
  }
  ++event_cursor_;
  event_prepared_ = false;
}

const DynamicBitset& UnorderedAccessPairs::currentCandidates() const {
  if (storage_filter_) return compatible_candidates_;
  const EventId event = memory_order_[event_cursor_];
  const AccessId access = event_accesses_[event.value()][access_cursor_];
  return writes_[access.value()] ? write_candidates_ : read_candidates_;
}

std::optional<AccessPair> UnorderedAccessPairs::next() {
  checkGraphUnchanged();
  while (event_cursor_ < memory_order_.size()) {
    if (!event_prepared_) {
      prepareEvent();
    }
    const auto& accesses =
        event_accesses_[memory_order_[event_cursor_].value()];
    while (access_cursor_ < accesses.size()) {
      if (next_candidate_ != DynamicBitset::npos) {
        const auto first = static_cast<std::uint32_t>(next_candidate_);
        next_candidate_ = currentCandidates().find_next(next_candidate_);
        return AccessPair{AccessId(first), accesses[access_cursor_]};
      }
      ++access_cursor_;
      if (access_cursor_ < accesses.size()) {
        prepareAccess();
      }
    }
    finishEvent();
  }
  return std::nullopt;
}

}  // namespace sentryward::algorithm
