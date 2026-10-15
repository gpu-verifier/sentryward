#include "sentryward/algorithm/diagnostics/performance_profile.h"
#include "sentryward/algorithm/analysis/reachability.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace sentryward::algorithm {
namespace {

std::size_t indexOf(EventId id) {
  return static_cast<std::size_t>(id.value());
}

std::size_t bitsetBytes(std::size_t count) {
  return (count / 64 + (count % 64 != 0)) * sizeof(std::uint64_t);
}

}  // namespace

ReachabilityOracle::ReachabilityOracle(const HappensBeforeGraph& graph,
                                       ReachabilityOptions options)
    : graph_(graph), options_(options) {
  refreshForGraphVersion();
}

bool ReachabilityOracle::Reaches(EventId from, EventId to) {
  ProfileScope query_profile("oracle.query");
  refreshForGraphVersion();
  if (!graph_.contains(from) || !graph_.contains(to)) return false;
  if (from == to) return true;
  if (auto* families = familyQueries()) return families->reaches(from, to);
  if (ancestor_sets_) {
    return (*ancestor_sets_)[indexOf(to)].test(indexOf(from));
  }
  if (hasOrder() && rank(from) >=
                     rank(to)) return false;
  const std::array<EventId, 1> target{from};
  const std::array<EventId, 1> destination{to};
  if (const auto* ancestors = cached(to, true, target)) {
    return ancestors->test(indexOf(from));
  }
  if (const auto* successors = cached(from, false, destination)) {
    return successors->test(indexOf(to));
  }
  return search(to, true, target, false).has_value();
}

std::vector<std::uint8_t> ReachabilityOracle::ReachesBatch(std::span<const HbQuery> queries) {
  ProfileScope profile("oracle.batch_pairs");
  refreshForGraphVersion();
  std::vector<std::uint8_t> answers(queries.size());
  if (auto* families = familyQueries()) {
    std::vector<HbQuery> valid;
    std::vector<std::size_t> indices;
    for (std::size_t i = 0; i < queries.size(); ++i) {
      if (!graph_.contains(queries[i].from) || !graph_.contains(queries[i].to)) continue;
      valid.push_back(queries[i]); indices.push_back(i);
    }
    const auto result = families->reachesBatch(valid);
    for (std::size_t i = 0; i < result.size(); ++i) answers[indices[i]] = result[i];
  } else {
    for (std::size_t i = 0; i < queries.size(); ++i)
      answers[i] = Reaches(queries[i].from, queries[i].to);
  }
  return answers;
}

bool ReachabilityOracle::AllReach(std::span<const EventId> from, EventId to) {
  ProfileScope query_profile("oracle.query");
  refreshForGraphVersion();
  if (!graph_.contains(to)) return false;
  if (from.empty()) return true;
  if (from.size() == 1) return Reaches(from.front(), to);
  if (auto* families = familyQueries()) return families->allReachMany(from, std::span(&to, 1)).front();
  for (EventId event : from) {
    if (!graph_.contains(event)) return false;
    if (hasOrder() && event != to && rank(event) >=
                                    rank(to)) return false;
  }
  const DynamicBitset* ancestors = ancestor_sets_
      ? &(*ancestor_sets_)[indexOf(to)] : cached(to, true, from);
  if (ancestors) {
    for (EventId event : from) {
      if (!ancestors->test(indexOf(event))) return false;
    }
    return true;
  }
  return search(to, true, from, true).has_value();
}

bool ReachabilityOracle::ReachesAny(EventId from,
                                    std::span<const EventId> targets) {
  ProfileScope query_profile("oracle.query");
  return FindDescendant(from, targets).has_value();
}

std::vector<std::uint8_t> ReachabilityOracle::AllReachMany(
    std::span<const EventId> from, std::span<const EventId> targets) {
  ProfileScope profile("oracle.batch_all_ancestors");
  refreshForGraphVersion();
  std::vector<std::uint8_t> result(targets.size(), 0);
  if (targets.empty()) return result;
  if (auto* families = familyQueries()) return families->allReachMany(from, targets);
  for (const auto source : from)
    if (!graph_.contains(source)) return result;
  if (from.empty()) {
    for (std::size_t i = 0; i < targets.size(); ++i)
      result[i] = graph_.contains(targets[i]);
    return result;
  }
  if (from.size() == 1) {
    const auto answers = FindAncestors(from, targets);
    for (std::size_t i = 0; i < targets.size(); ++i)
      result[i] = answers[i].has_value();
    return result;
  }
  std::vector<EventId> sources(from.begin(), from.end());
  std::sort(sources.begin(), sources.end());
  sources.erase(std::unique(sources.begin(), sources.end()), sources.end());
  const bool bounded = hasOrder();
  std::size_t lower = std::numeric_limits<std::size_t>::max(), last_source = 0;
  if (bounded) for (auto source : sources) {
    lower = std::min(lower, rank(source));
    last_source = std::max(last_source, rank(source));
  }
  // A predecessor postorder is shared by all queries and all source words.
  DynamicBitset visited(graph_.eventCount());
  struct Frame { EventId event; std::size_t next = 0; };
  std::vector<Frame> stack;
  std::vector<EventId> order;
  for (std::size_t i = 0; i < targets.size(); ++i) {
    const auto target = targets[i];
    if (!graph_.contains(target) || (bounded && rank(target) < last_source)) continue;
    result[i] = 1;
    if (visited.test(indexOf(target))) continue;
    visited.set(indexOf(target));
    stack.push_back({target});
    while (!stack.empty()) {
      auto& frame = stack.back();
      const auto predecessors = graph_.predecessors(frame.event);
      if (frame.next == predecessors.size()) {
        order.push_back(frame.event);
        stack.pop_back();
        continue;
      }
      const auto predecessor = predecessors[frame.next++];
      if ((bounded && rank(predecessor) < lower) || visited.test(indexOf(predecessor))) continue;
      visited.set(indexOf(predecessor));
      stack.push_back({predecessor});
      if ((order.size() + stack.size()) % 65536 == 0) ProfileSnapshot();
    }
  }
  if (order.empty()) return result;
  // One word per visited event, independent of source/target-set cardinality.
  std::vector<std::uint64_t> masks(graph_.eventCount(), 0);
  if (targets.size() < sources.size()) {
    for (std::size_t begin = 0; begin < targets.size(); begin += 64) {
      const auto count = std::min<std::size_t>(64, targets.size() - begin);
      std::uint64_t accepted = 0;
      for (auto event : order) masks[indexOf(event)] = 0;
      for (std::size_t bit = 0; bit < count; ++bit) {
        if (!result[begin + bit]) continue;
        const auto flag = std::uint64_t{1} << bit;
        accepted |= flag;
        masks[indexOf(targets[begin + bit])] |= flag;
      }
      if (!accepted) continue;
      for (auto it = order.rbegin(); it != order.rend(); ++it) {
        const auto mask = masks[indexOf(*it)];
        for (auto predecessor : graph_.predecessors(*it)) {
          if (visited.test(indexOf(predecessor))) masks[indexOf(predecessor)] |= mask;
          ++profile.edges;
        }
        ++profile.nodes;
        if ((profile.nodes & 65535U) == 0) ProfileSnapshot();
      }
      for (auto source : sources) {
        accepted &= visited.test(indexOf(source)) ? masks[indexOf(source)] : 0;
        if (!accepted) break;
      }
      for (std::size_t bit = 0; bit < count; ++bit)
        result[begin + bit] = (accepted >> bit) & 1;
    }
    return result;
  }
  for (std::size_t begin = 0; begin < sources.size(); begin += 64) {
    const auto count = std::min<std::size_t>(64, sources.size() - begin);
    const auto full = count == 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << count) - 1;
    for (auto event : order) masks[indexOf(event)] = 0;
    for (std::size_t bit = 0; bit < count; ++bit)
      masks[indexOf(sources[begin + bit])] |= std::uint64_t{1} << bit;
    for (auto event : order) {
      auto& mask = masks[indexOf(event)];
      for (auto predecessor : graph_.predecessors(event)) {
        if (visited.test(indexOf(predecessor))) mask |= masks[indexOf(predecessor)];
        ++profile.edges;
      }
      ++profile.nodes;
      if ((profile.nodes & 65535U) == 0) ProfileSnapshot();
    }
    bool any = false;
    for (std::size_t i = 0; i < targets.size(); ++i) {
      if (result[i]) result[i] = masks[indexOf(targets[i])] == full;
      any |= result[i] != 0;
    }
    if (!any) break;
  }
  return result;
}

bool ReachabilityOracle::ReachesAll(EventId from,
                                   std::span<const EventId> targets) {
  ProfileScope query_profile("oracle.query");
  refreshForGraphVersion();
  if (!graph_.contains(from)) return false;
  if (targets.empty()) return true;
  if (auto* families = familyQueries(true))
    return families->allReachMany(targets, std::span(&from, 1)).front();
  for (EventId target : targets) {
    if (!graph_.contains(target)) return false;
    if (hasOrder() && target != from && rank(target) <=
                                       rank(from)) return false;
  }
  if (ancestor_sets_) {
    for (EventId target : targets) {
      if (!(*ancestor_sets_)[indexOf(target)].test(indexOf(from))) return false;
    }
    return true;
  }
  if (const auto* successors = cached(from, false, targets)) {
    for (EventId target : targets) {
      if (!successors->test(indexOf(target))) return false;
    }
    return true;
  }
  return search(from, false, targets, true).has_value();
}

std::optional<EventId> ReachabilityOracle::FindDescendant(
    EventId from, std::span<const EventId> targets) {
  ProfileScope query_profile("oracle.query");
  refreshForGraphVersion();
  if (!graph_.contains(from)) return std::nullopt;
  if (auto* families = familyQueries(true)) return families->findAncestors(targets, std::span(&from, 1)).front();
  if (ancestor_sets_) {
    for (EventId target : targets) {
      if (graph_.contains(target) &&
          (*ancestor_sets_)[indexOf(target)].test(indexOf(from))) return target;
    }
    return std::nullopt;
  }
  bool has_candidate = false;
  for (EventId target : targets) {
    if (!graph_.contains(target)) continue;
    if (target == from) return target;
    if (!hasOrder() || rank(from) <
                        rank(target)) has_candidate = true;
  }
  if (!has_candidate) return std::nullopt;
  if (const auto* successors = cached(from, false, targets)) {
    for (EventId target : targets) {
      if (graph_.contains(target) && successors->test(indexOf(target))) return target;
    }
    return std::nullopt;
  }
  return search(from, false, targets, false);
}

std::optional<EventId> ReachabilityOracle::FindAncestor(
    std::span<const EventId> candidates, EventId target) {
  ProfileScope query_profile("oracle.query");
  refreshForGraphVersion();
  if (!graph_.contains(target)) return std::nullopt;
  if (auto* families = familyQueries()) return families->findAncestors(candidates, std::span(&target, 1)).front();
  const DynamicBitset* ancestors = ancestor_sets_
      ? &(*ancestor_sets_)[indexOf(target)] : cached(target, true, candidates);
  bool has_candidate = false;
  for (EventId candidate : candidates) {
    if (!graph_.contains(candidate)) continue;
    if (candidate == target) return candidate;
    if (hasOrder() && candidate != target && rank(candidate) >=
                       rank(target)) continue;
    if (ancestors && ancestors->test(indexOf(candidate))) return candidate;
    has_candidate = true;
  }
  if (ancestors || !has_candidate) return std::nullopt;
  return search(target, true, candidates, false);
}

std::vector<std::optional<EventId>> ReachabilityOracle::FindAncestors(
    std::span<const EventId> candidates, std::span<const EventId> targets) {
  ProfileScope profile("oracle.batch_ancestors");
  refreshForGraphVersion();
  std::vector<std::optional<EventId>> result(targets.size());
  if (candidates.empty() || targets.empty()) return result;
  if (auto* families = familyQueries()) return families->findAncestors(candidates, targets);
  if (targets.size() == 1 || ancestor_sets_) {
    for (std::size_t i = 0; i < targets.size(); ++i)
      result[i] = FindAncestor(candidates, targets[i]);
    return result;
  }
  for (const auto event : batch_worklist_) batch_witnesses_[indexOf(event)].reset();
  batch_worklist_.clear();
  for (const auto event : batch_target_ids_) batch_targets_.reset(indexOf(event));
  batch_target_ids_.clear();
  batch_witnesses_.resize(graph_.eventCount());
  batch_targets_.resize(graph_.eventCount());
  const bool bounded = options_.use_interval_pruning && hasOrder();
  std::size_t upper = 0;
  for (const auto target : targets) {
    if (!graph_.contains(target)) continue;
    if (!batch_targets_.test(indexOf(target))) {
      batch_targets_.set(indexOf(target));
      batch_target_ids_.push_back(target);
    }
    if (bounded) upper = std::max(upper, rank(target));
  }
  std::size_t remaining = batch_target_ids_.size();
  if (remaining == 0) return result;
  const auto discover = [&](EventId event, EventId witness) {
    if (batch_witnesses_[indexOf(event)]) return;
    batch_witnesses_[indexOf(event)] = witness;
    batch_worklist_.push_back(event);
    if (batch_targets_.test(indexOf(event))) {
      batch_targets_.reset(indexOf(event));
      --remaining;
    }
  };
  for (const auto candidate : candidates) {
    if (!graph_.contains(candidate) || (bounded && rank(candidate) > upper)) continue;
    discover(candidate, candidate);
  }
  for (std::size_t cursor = 0; remaining && cursor < batch_worklist_.size(); ++cursor) {
    const auto current = batch_worklist_[cursor];
    ++profile.nodes;
    for (const auto successor : graph_.successors(current)) {
      ++profile.edges;
      if (bounded && rank(successor) > upper) { ++profile.pruned; continue; }
      discover(successor, *batch_witnesses_[indexOf(current)]);
    }
    if ((cursor & 65535U) == 0) ProfileSnapshot();
  }
  for (std::size_t i = 0; i < targets.size(); ++i)
    if (graph_.contains(targets[i])) result[i] = batch_witnesses_[indexOf(targets[i])];
  return result;
}

bool ReachabilityOracle::hasOrder() const noexcept {
  return frozen_ || (options_.use_interval_pruning && graph_.traceOrder());
}

std::size_t ReachabilityOracle::rank(EventId event) const noexcept {
  if (const auto* order = graph_.traceOrder()) return order->rank(event);
  return topological_ranks_[indexOf(event)];
}

void ReachabilityOracle::NotifyGraphMutation() noexcept {
  ProfileScope profile("oracle.invalidate");
  observed_version_ = kUnknownVersion;
  frozen_ = false;
  cached_bytes_ = 0;
  cache_.clear();
  scratch_ = DynamicBitset{};
  query_targets_ = DynamicBitset{};
  worklist_.clear();
  query_candidates_.clear();
  batch_witnesses_.clear();
  batch_worklist_.clear();
  batch_targets_ = DynamicBitset{};
  batch_target_ids_.clear();
  topological_ranks_.clear();
  ancestor_sets_.reset();
  family_queries_checked_ = bool(family_queries_);
  reverse_family_queries_checked_ = bool(reverse_family_queries_);
  shared_visited_ = DynamicBitset{};
  shared_worklist_.clear();
  scratch_contains_shared_search_ = false;
  // Preserve base row sharing. Each index lazily consumes appended edge/event
  // records and splits only the vertices whose directional adjacency changed.
}

bool ReachabilityOracle::Freeze() {
  ProfileScope profile("oracle.freeze");
  refreshForGraphVersion();
  if (frozen_) return true;
  const std::size_t row_size = bitsetBytes(graph_.eventCount()) + sizeof(DynamicBitset);
  if (graph_.traceOrder() && graph_.eventCount() > options_.dense_memory_limit_bytes / row_size) {
    frozen_ = true;
    return true;
  }
  const auto order = graph_.topologicalOrder();
  if (!order) return false;

  const std::size_t count = graph_.eventCount();
  topological_ranks_.resize(count);
  for (std::size_t index = 0; index < order->size(); ++index) {
    topological_ranks_[indexOf((*order)[index])] = index;
  }
  const std::size_t row_bytes = bitsetBytes(count) + sizeof(DynamicBitset);
  if (count <= options_.dense_memory_limit_bytes / row_bytes) {
    std::vector<DynamicBitset> ancestors;
    ancestors.reserve(count);
    for (std::size_t index = 0; index < count; ++index) ancestors.emplace_back(count);
    for (EventId event : *order) {
      DynamicBitset& current = ancestors[indexOf(event)];
      current.set(indexOf(event));
      for (EventId predecessor : graph_.predecessors(event)) {
        current |= ancestors[indexOf(predecessor)];
      }
    }
    ancestor_sets_ = std::move(ancestors);
    cache_.clear();
    cached_bytes_ = 0;
    scratch_ = DynamicBitset{};
    query_targets_ = DynamicBitset{};
    worklist_.clear();
    query_candidates_.clear();
  }
  frozen_ = true;
  return true;
}

bool ReachabilityOracle::frozen() const noexcept {
  return frozen_ && observed_version_ == graph_.version();
}

bool ReachabilityOracle::usesDenseIndex() const noexcept {
  return frozen() && ancestor_sets_.has_value();
}

std::size_t ReachabilityOracle::cachedBitsetBytes() const noexcept {
  return observed_version_ == graph_.version() ? cached_bytes_ : 0;
}

std::size_t ReachabilityOracle::sharedStateCount(bool reverse) const noexcept {
  const auto& index = reverse ? reverse_shared_ : forward_shared_;
  return index ? index->StateCount() : 0;
}

std::size_t ReachabilityOracle::sharedIndexBytes() const noexcept {
  return (forward_shared_ ? forward_shared_->storageBytes() : 0) +
         (reverse_shared_ ? reverse_shared_->storageBytes() : 0);
}

void ReachabilityOracle::refreshForGraphVersion() {
  if (observed_version_ == graph_.version()) return;
  NotifyGraphMutation();
  observed_version_ = graph_.version();
}

FamilyHbReachability* ReachabilityOracle::familyQueries(bool reverse) {
  auto& checked = reverse ? reverse_family_queries_checked_ : family_queries_checked_;
  auto& index = reverse ? reverse_family_queries_ : family_queries_;
  if (!checked) {
    checked = true;
    HbReachabilityOptions options;
    options.index_budget_bytes = options_.cache_memory_limit_bytes;
    index = TryFamilyHbReachability(graph_, options, reverse);
  }
  if(index)index->RefreshAppendOnly();
  return index.get();
}

const DynamicBitset* ReachabilityOracle::cached(
    EventId root, bool reverse, std::span<const EventId> targets) {
  const auto found = std::find_if(cache_.begin(), cache_.end(),
      [&](const CacheEntry& entry) {
        if (entry.root != root || entry.reverse != reverse) return false;
        if (entry.lower == 0 && entry.upper == std::numeric_limits<std::size_t>::max()) return true;
        for (EventId target : targets) {
          if (!graph_.contains(target)) continue;
          if (rank(target) < entry.lower || rank(target) > entry.upper) return false;
        }
        return true;
      });
  if (found != cache_.end()) {
    cache_.splice(cache_.begin(), cache_, found);
    return &cache_.front().vertices;
  }
  return nullptr;
}

std::optional<EventId> ReachabilityOracle::search(
    EventId root, bool reverse, std::span<const EventId> targets,
    bool require_all) {
  ProfileScope profile("oracle.search");
  if (scratch_.size() != graph_.eventCount()) {
    scratch_.resize(graph_.eventCount());
    query_targets_.resize(graph_.eventCount());
  }
  if (scratch_contains_shared_search_) scratch_.reset();
  else for (EventId event : worklist_) scratch_.reset(indexOf(event));
  scratch_contains_shared_search_ = false;
  worklist_.clear();
  for (EventId event : query_candidates_) query_targets_.reset(indexOf(event));
  query_candidates_.clear();
  for (EventId target : targets) {
    if (!graph_.contains(target)) continue;
    if (hasOrder() && target != root &&
        (reverse ? rank(target) >= rank(root)
                 : rank(target) <= rank(root))) {
      continue;
    }
    if (!query_targets_.test(indexOf(target))) {
      query_targets_.set(indexOf(target));
      query_candidates_.push_back(target);
    }
  }
  if (query_candidates_.empty()) return std::nullopt;
  bounded_search_ = options_.use_interval_pruning && hasOrder();
  search_lower_ = 0;
  search_upper_ = std::numeric_limits<std::size_t>::max();
  if (bounded_search_) {
    search_lower_ = rank(root);
    search_upper_ = rank(root);
    for (EventId target : query_candidates_) {
      search_lower_ = std::min(search_lower_, rank(target));
      search_upper_ = std::max(search_upper_, rank(target));
    }
  }
  std::size_t remaining = query_candidates_.size();
  if (options_.use_shared_states &&
      graph_.eventCount() >= options_.shared_state_min_vertices) {
    return searchShared(root, reverse, require_all, remaining);
  }
  scratch_.set(indexOf(root));
  worklist_.push_back(root);
  for (std::size_t cursor = 0; cursor < worklist_.size(); ++cursor) {
    const EventId current = worklist_[cursor];
    ++profile.nodes;
    if (query_targets_.test(indexOf(current))) {
      query_targets_.reset(indexOf(current));
      --remaining;
      if (!require_all || remaining == 0) return current;
    }
    const auto neighbors = reverse ? graph_.predecessors(current)
                                   : graph_.successors(current);
    for (EventId neighbor : neighbors) {
      ++profile.edges;
      if (bounded_search_ &&
          (rank(neighbor) < search_lower_ || rank(neighbor) > search_upper_)) { ++profile.pruned; continue; }
      if (!scratch_.test(indexOf(neighbor))) {
        scratch_.set(indexOf(neighbor));
        worklist_.push_back(neighbor);
      }
    }
  }
  // Only exhausted searches produce complete cache rows.
  cacheComplete(root, reverse);
  return std::nullopt;
}

std::optional<EventId> ReachabilityOracle::searchShared(
    EventId root, bool reverse, bool require_all, std::size_t remaining) {
  ProfileScope profile("oracle.shared_search");
  if (query_targets_.test(indexOf(root))) {
    query_targets_.reset(indexOf(root));
    if (!require_all || --remaining == 0) return root;
  }
  auto& index = reverse ? reverse_shared_ : forward_shared_;
  if (!index) index.emplace(graph_, reverse);
  else index->SyncGraphChanges();
  // Short queries should not clear a graph-sized bitset. Dense traversals
  // clear whole words instead of revisiting every individual state.
  if (shared_worklist_.size() < shared_visited_.size() / 64) {
    for (const auto state : shared_worklist_) shared_visited_.reset(state);
  } else {
    shared_visited_.reset();
  }
  shared_visited_.resize(index->StateCount());
  shared_worklist_.clear();
  const auto initial = index->StateFor(root);
  shared_visited_.set(initial);
  shared_worklist_.push_back(initial);
  for (std::size_t cursor = 0; cursor < shared_worklist_.size(); ++cursor) {
    ++profile.nodes;
    for (EventId neighbor : index->Neighbors(shared_worklist_[cursor])) {
      ++profile.edges;
      if (bounded_search_ &&
          (rank(neighbor) < search_lower_ || rank(neighbor) > search_upper_)) { ++profile.pruned; continue; }
      // Test the ORIGINAL endpoint before suppressing a shared expansion.
      if (query_targets_.test(indexOf(neighbor))) {
        query_targets_.reset(indexOf(neighbor));
        if (!require_all || --remaining == 0) return neighbor;
      }
      const auto state = index->StateFor(neighbor);
      if (!shared_visited_.test(state)) {
        shared_visited_.set(state);
        shared_worklist_.push_back(state);
      }
    }
  }
  // Only an exhausted, cacheable search needs original-vertex membership.
  // Reconstruct it from the expanded rows, including literal endpoints that
  // shared a visited state. Early successes and cache-disabled queries avoid
  // maintaining and clearing a second visited set over all original vertices.
  if (bitsetBytes(graph_.eventCount()) <= options_.cache_memory_limit_bytes) {
    scratch_contains_shared_search_ = true;
    scratch_.set(indexOf(root));
    for (const auto state : shared_worklist_) {
      for (EventId neighbor : index->Neighbors(state)) {
        scratch_.set(indexOf(neighbor));
      }
    }
    cacheComplete(root, reverse);
  }
  return std::nullopt;
}

void ReachabilityOracle::cacheComplete(EventId root, bool reverse) {
  // Exhausted intervals are cached with their validity bounds, not as full rows.
  const std::size_t row_bytes = bitsetBytes(graph_.eventCount());
  if (row_bytes == 0 || row_bytes > options_.cache_memory_limit_bytes) return;
  while (cache_.size() >= 64 ||
         cached_bytes_ > options_.cache_memory_limit_bytes - row_bytes) {
    cache_.pop_back();
    cached_bytes_ -= row_bytes;
  }
  cache_.push_front(CacheEntry{root, reverse, scratch_,
      bounded_search_ ? search_lower_ : 0,
      bounded_search_ ? search_upper_ : std::numeric_limits<std::size_t>::max()});
  cached_bytes_ += row_bytes;
}

}  // namespace sentryward::algorithm
