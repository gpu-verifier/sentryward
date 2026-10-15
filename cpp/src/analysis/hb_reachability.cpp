#include "sentryward/algorithm/analysis/hb_reachability.h"
#include "sentryward/algorithm/analysis/family_relation_query.h"
#include "sentryward/algorithm/analysis/compact_hb_reachability.h"
#include "sentryward/algorithm/analysis/family_hb_reachability.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "sentryward/algorithm/diagnostics/performance_profile.h"

namespace sentryward::algorithm {

FamilyRelationQuery& StaticHbQueries::relations() {
  if(graph_.version()!=version_)throw std::logic_error("HB graph changed");
  if(!relations_)relations_=std::make_shared<FamilyRelationQuery>(graph_);
  return *relations_;
}


std::vector<std::uint8_t> HbReachability::reachesBatch(std::span<const HbQuery> queries) {
  std::vector<std::uint8_t> answer;
  answer.reserve(queries.size());
  for (const auto& query : queries) answer.push_back(reaches(query.from, query.to));
  return answer;
}

std::vector<std::uint8_t> HbReachability::allReachMany(
    std::span<const EventId> sources, std::span<const EventId> targets) {
  for (auto source : sources)
    if (!graph().contains(source)) throw std::invalid_argument("unknown HB source");
  std::vector<std::uint8_t> result(targets.size(), 1);
  for (std::size_t i = 0; i < targets.size(); ++i) {
    if (!graph().contains(targets[i])) throw std::invalid_argument("unknown HB target");
    for (auto source : sources)
      if (!reaches(source, targets[i])) { result[i] = 0; break; }
  }
  return result;
}

namespace {

class DfsIndex : public HbReachability {
 public:
  explicit DfsIndex(const HappensBeforeGraph& graph)
      : graph_(graph), version_(graph.version()),
        depth_(graph.eventCount()), visited_(graph.eventCount()) {
    auto order = graph.topologicalOrder();
    if (!order) throw std::invalid_argument("HB index requires an acyclic graph");
    order_ = std::move(*order);
    if (!order_.isIdentity()) rank_.resize(graph.eventCount());
    for (std::size_t i = 0; i < order_.size(); ++i) {
      const auto id = order_[i];
      if (!order_.isIdentity()) rank_[id.value()] = i;
      for (const auto pred : graph_.predecessors(id))
        depth_[id.value()] = std::max(depth_[id.value()], depth_[pred.value()] + 1);
    }
  }
  const HappensBeforeGraph& graph() const noexcept override { return graph_; }
  std::string_view name() const noexcept override { return "dfs"; }
  HbReachabilityStats stats() const noexcept override { return stats_; }
  bool reaches(EventId from, EventId to) override {
    unchanged();
    ++stats_.queries;
    if (const auto answer = trivial({from, to}); answer.has_value()) return *answer;
    return search({from, to});
  }

 protected:
  std::size_t rank(EventId id) const noexcept {
    return order_.isIdentity() ? id.value() : rank_[id.value()];
  }
  void unchanged() const {
    if (graph_.version() != version_ || graph_.eventCount() != order_.size())
      throw std::logic_error("graph changed during HB index use");
  }
  std::optional<bool> trivial(const HbQuery& query) {
    if (!graph_.contains(query.from) || !graph_.contains(query.to))
      throw std::invalid_argument("HB query references an unknown event");
    if (query.from == query.to) { ++stats_.filtered_queries; return true; }
    if (rank(query.from) >= rank(query.to) ||
        depth_[query.from.value()] >= depth_[query.to.value()]) {
      ++stats_.filtered_queries;
      return false;
    }
    return std::nullopt;
  }
  bool search(const HbQuery& query) {
    ++stats_.dfs_queries;
    if (++generation_ == 0) {
      std::fill(visited_.begin(), visited_.end(), 0);
      generation_ = 1;
    }
    work_.clear();
    work_.push_back(query.to);
    visited_[query.to.value()] = generation_;
    while (!work_.empty()) {
      const auto current = work_.back();
      work_.pop_back();
      for (const auto pred : graph_.predecessors(current)) {
        ++stats_.edge_visits;
        if (pred == query.from) return true;
        if (rank(pred) <= rank(query.from) ||
            depth_[pred.value()] <= depth_[query.from.value()] ||
            visited_[pred.value()] == generation_) continue;
        visited_[pred.value()] = generation_;
        work_.push_back(pred);
      }
    }
    return false;
  }

  const HappensBeforeGraph& graph_;
  std::size_t version_;
  EventOrder order_;
  std::vector<std::size_t> rank_;
  std::vector<std::size_t> depth_;
  std::vector<std::uint32_t> visited_;
  std::uint32_t generation_ = 0;
  std::vector<EventId> work_;
  HbReachabilityStats stats_;
};

class BitsetIndex final : public DfsIndex {
 public:
  BitsetIndex(const HappensBeforeGraph& graph, HbReachabilityOptions options)
      : DfsIndex(graph), options_(options), cache_(options.cached_pairs) {
    if (options_.max_batch_sources == 0)
      throw std::invalid_argument("max_batch_sources must be positive");
  }
  std::string_view name() const noexcept override { return "batched-bitset"; }
  bool reaches(EventId from, EventId to) override {
    unchanged();
    ++stats_.queries;
    const HbQuery query{from, to};
    if (const auto answer = trivial(query)) return *answer;
    if (const auto answer = lookup(query)) {
      ++stats_.cache_hits;
      return *answer;
    }
    const auto answer = search(query);
    remember(query, answer);
    return answer;
  }
  std::vector<std::uint8_t> reachesBatch(std::span<const HbQuery> queries) override {
    unchanged();
    stats_.queries += queries.size();
    std::vector<std::uint8_t> answer(queries.size());
    std::vector<std::size_t> pending;
    for (std::size_t i = 0; i < queries.size(); ++i) {
      const auto known = trivial(queries[i]);
      if (known) answer[i] = *known;
      else if (const auto cached = lookup(queries[i])) {
        answer[i] = *cached;
        ++stats_.cache_hits;
      } else pending.push_back(i);
    }
    std::sort(pending.begin(), pending.end(), [&](auto a, auto b) {
      return rank(queries[a].from) < rank(queries[b].from);
    });
    for (std::size_t begin = 0; begin < pending.size();) {
      std::size_t end = begin, sources = 0;
      const auto low = rank(queries[pending[begin]].from);
      std::size_t high = low;
      std::vector<EventId> source_ids;
      while (end < pending.size()) {
        const auto source = queries[pending[end]].from;
        if (sources == options_.max_batch_sources) break;
        source_ids.push_back(source);
        ++sources;
        do {
          high = std::max(high, rank(queries[pending[end]].to));
          ++end;
        } while (end < pending.size() && queries[pending[end]].from == source);
      }
      auto words = (sources + 63) / 64;
      while (high - low + 1 > options_.index_budget_bytes / sizeof(std::uint64_t) / words && sources > 1) {
        sources = (sources + 1) / 2;
        source_ids.resize(sources);
        end = begin;
        high = low;
        while (end < pending.size() &&
               rank(queries[pending[end]].from) <= rank(source_ids.back())) {
          high = std::max(high, rank(queries[pending[end]].to));
          ++end;
        }
        words = (sources + 63) / 64;
      }
      const auto rows = high - low + 1;
      if (rows > options_.index_budget_bytes / sizeof(std::uint64_t) / words) {
        for (auto i = begin; i < end; ++i) answer[pending[i]] = search(queries[pending[i]]);
        begin = end;
        continue;
      }
      ++stats_.batches;
      stats_.peak_index_bytes = std::max(stats_.peak_index_bytes, rows * words * sizeof(std::uint64_t));
      std::vector<std::uint64_t> bits(rows * words);
      std::vector<std::uint8_t> active(rows);
      for (std::size_t i = 0; i < sources; ++i) {
        const auto row = rank(source_ids[i]) - low;
        bits[row * words + i / 64] |= std::uint64_t{1} << (i % 64);
        active[row] = 1;
      }
      for (std::size_t rank = low; rank <= high; ++rank) {
        auto* row = bits.data() + (rank - low) * words;
        for (const auto pred : graph_.predecessors(order_[rank])) {
          ++stats_.edge_visits;
          const auto p = this->rank(pred);
          if (p < low || !active[p - low]) continue;
          const auto* parent = bits.data() + (p - low) * words;
          for (std::size_t w = 0; w < words; ++w) row[w] |= parent[w];
          active[rank - low] = 1;
        }
      }
      std::size_t source_index = 0;
      for (auto i = begin; i < end; ++i) {
        const auto& query = queries[pending[i]];
        while (source_ids[source_index] != query.from) ++source_index;
        const auto row = rank(query.to) - low;
        answer[pending[i]] = (bits[row * words + source_index / 64] >> (source_index % 64)) & 1;
      }
      begin = end;
    }
    for (const auto i : pending) remember(queries[i], answer[i]);
    return answer;
  }

 private:
  struct CachedPair {
    std::uint64_t key = 0;
    std::uint8_t answer = 0;
    bool valid = false;
  };
  static std::uint64_t key(const HbQuery& query) {
    return (std::uint64_t{query.from.value()} << 32) | query.to.value();
  }
  std::size_t slot(std::uint64_t value) const {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    return (value ^ (value >> 31)) % cache_.size();
  }
  std::optional<std::uint8_t> lookup(const HbQuery& query) const {
    if (cache_.empty()) return std::nullopt;
    const auto id = key(query);
    const auto& entry = cache_[slot(id)];
    if (entry.valid && entry.key == id) return entry.answer;
    return std::nullopt;
  }
  void remember(const HbQuery& query, std::uint8_t answer) {
    if (cache_.empty()) return;
    const auto id = key(query);
    cache_[slot(id)] = {id, answer, true};
  }
  HbReachabilityOptions options_;
  std::vector<CachedPair> cache_;
};
}  // namespace

std::unique_ptr<HbReachability> MakeHbReachability(
    const HappensBeforeGraph& graph, const HbReachabilityOptions& options) {
  switch (options.algorithm) {
    case HbAlgorithm::kAuto:
      if(options.max_batch_sources==0)
        throw std::invalid_argument("max_batch_sources must be positive");
      if(auto compact=TryCompactHbReachability(graph,options))return compact;
      if(auto family=TryFamilyHbReachability(graph,options))return family;
      return std::make_unique<BitsetIndex>(graph,options);

    case HbAlgorithm::kBatchedBitset: return std::make_unique<BitsetIndex>(graph, options);
    case HbAlgorithm::kDfs: return std::make_unique<DfsIndex>(graph);
  }
  throw std::invalid_argument("unknown HB reachability algorithm");
}

StaticHbQueries::StaticHbQueries(const HappensBeforeGraph& graph,
    HbReachabilityOptions options, HbReachabilityFactory factory)
    : graph_(graph), version_(graph.version()), options_(options), factory_(std::move(factory)) {}

HbReachability& StaticHbQueries::get() {
  if (graph_.version() != version_)
    throw std::invalid_argument("static HB queries require an unchanged graph");
  if (!index_) {
    auto index = ProfileCall("hb.static_index", [&] {
      return factory_ ? factory_(graph_, options_) : MakeHbReachability(graph_, options_);
    });
    if (!index || &index->graph() != &graph_)
      throw std::invalid_argument("HB index must refer to the certificate graph");
    index_ = std::move(index);
  }
  return *index_;
}

HbReachabilityStats StaticHbQueries::stats() const noexcept {
  return index_ ? index_->stats() : HbReachabilityStats{};
}

std::string_view StaticHbQueries::name() const noexcept {
  return index_ ? index_->name() : "not-built";
}

HbReachabilityStats HbStatsSince(const HbReachabilityStats& current,
                               const HbReachabilityStats& baseline) noexcept {
  return {current.queries - baseline.queries,
      current.filtered_queries - baseline.filtered_queries,
      current.batches - baseline.batches,
      current.edge_visits - baseline.edge_visits,
      current.peak_index_bytes,
      current.dfs_queries - baseline.dfs_queries,
      current.cache_hits - baseline.cache_hits};
}

}  // namespace sentryward::algorithm
