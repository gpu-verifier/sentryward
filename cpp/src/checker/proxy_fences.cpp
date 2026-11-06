#include "sentryward/algorithm/checker/proxy_fences.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <span>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "sentryward/algorithm/analysis/happens_before_graph.h"
#include "sentryward/algorithm/base/dynamic_bitset.h"

namespace sentryward::algorithm {
namespace {

constexpr std::uint8_t kBothSpaces = kProxyFenceShared | kProxyFenceGlobal;
constexpr std::size_t kNone = std::numeric_limits<std::size_t>::max();

// The fence spaces a pair needs: its state space, or both when the overlap has no constant region
// (only a fence covering every space is known to cover it). Zero: a space without an async proxy.
std::uint8_t NeededSpaces(const std::optional<OverlapEvidence>& evidence) {
  if (!evidence || !evidence->constant_region) return kBothSpaces;
  const auto& space = evidence->constant_region->address_space;
  if (space.starts_with("shared")) return kProxyFenceShared;
  if (space == "global") return kProxyFenceGlobal;
  return 0;
}

// Answers in batches: the index's batch path shares work across queries, a single query searches.
std::vector<std::uint8_t> Reaches(HbReachability& hb, const std::vector<HbQuery>& queries) {
  constexpr std::size_t kBatch = 65536;
  std::vector<std::uint8_t> answers;
  answers.reserve(queries.size());
  for (std::size_t begin = 0; begin < queries.size(); begin += kBatch) {
    const auto part = hb.reachesBatch(std::span(queries).subspan(begin, std::min(kBatch, queries.size() - begin)));
    answers.insert(answers.end(), part.begin(), part.end());
  }
  return answers;
}

// Each thread's proxy fences in program order, by the spaces they must cover (index 1-3): fences of
// one thread are ordered by happens-before, so whether g reaches a fence is monotone along a list
// and whether a fence reaches a is its reverse.
class FenceIndex {
 public:
  FenceIndex(const SourceProgram& source, const HappensBeforeGraph& graph) {
    std::unordered_map<std::uint32_t, std::uint8_t> spaces; // fence instruction -> its spaces
    for (std::size_t i = 0; i < source.instructions.size(); ++i) {
      if (!source.instructions.holdsOperation<Await>(i)) continue;
      const auto instruction = source.instructions[i];
      if (const auto fence = std::get<Await>(instruction.op).proxy_fence)
        spaces.emplace(instruction.id.value(), fence);
    }
    if (spaces.empty()) return;
    graph.events().forEachAffineColumn([&](const Event& first, const auto& delta, std::size_t,
                                           std::size_t, std::size_t count) {
      if (!std::holds_alternative<AwaitIssueEvent>(first.payload) || !first.origin || !first.actor) return;
      for (std::size_t k = 0; k < count; ++k) {
        const auto step = static_cast<std::int64_t>(k);
        const auto origin = static_cast<std::uint32_t>(first.origin->value() + delta[1] * step);
        const auto found = spaces.find(origin);
        if (found == spaces.end()) continue;
        const auto actor = static_cast<std::uint32_t>(first.actor->value() + delta[2] * step);
        const auto event = EventId(static_cast<std::uint32_t>(first.id.value() + delta[0] * step));
        auto& lists = threads_[actor];
        for (std::uint8_t needed = 1; needed <= kBothSpaces; ++needed)
          if ((found->second & needed) == needed) lists[needed].push_back({InstId(origin), event});
        all_.push_back({event, found->second});
        ++count_;
      }
    });
    for (auto& [actor, lists] : threads_)
      for (auto& list : lists)
        std::sort(list.begin(), list.end(), [](const auto& a, const auto& b) {
          return a.first.value() < b.first.value();
        });
  }
  [[nodiscard]] std::size_t count() const noexcept { return count_; }
  // Every fence: its issue event and spaces.
  [[nodiscard]] const std::vector<std::pair<EventId, std::uint8_t>>& all() const noexcept { return all_; }
  // Whether each list is a happens-before chain, as the binary searches below require.
  [[nodiscard]] bool chained(HbReachability& hb) const {
    std::vector<HbQuery> links;
    for (const auto& [actor, lists] : threads_)
      for (const auto& list : lists)
        for (std::size_t k = 1; k < list.size(); ++k) links.push_back({list[k - 1].second, list[k].second});
    const auto answers = Reaches(hb, links);
    return std::all_of(answers.begin(), answers.end(), [](auto answer) { return answer != 0; });
  }
  [[nodiscard]] const std::vector<std::pair<InstId, EventId>>& of(ActorId actor, std::uint8_t needed) const {
    static const std::vector<std::pair<InstId, EventId>> none;
    const auto found = threads_.find(actor.value());
    return found == threads_.end() ? none : found->second[needed];
  }

 private:
  std::unordered_map<std::uint32_t, std::array<std::vector<std::pair<InstId, EventId>>, 4>> threads_;
  std::vector<std::pair<EventId, std::uint8_t>> all_;
  std::size_t count_ = 0;
};

// Whether some fence of `fences` covering pair k's spaces lies between its two events (from -> F ->
// to), for the pairs with `pending` set; it clears the pairs it finds fenced. Paths between them stay
// inside the topological rank window [rank(from), rank(to)], so the window of all pending pairs is
// scanned, 64 fences at a time: a reverse pass marks the fences each event reaches, a forward pass the
// fences reaching it. O(F / 64 x (V + E)) over the window's V events and E edges.
struct EndpointPair {
  EventId from, to;
  std::uint8_t needed;
};
void ClearFenced(const HappensBeforeGraph& graph, const EventOrder& order, const std::vector<std::size_t>& rank,
                 const std::vector<std::pair<EventId, std::uint8_t>>& fences, const std::vector<EndpointPair>& pairs,
                 std::vector<std::uint8_t>& pending) {
  if (std::none_of(pending.begin(), pending.end(), [](auto p) { return p != 0; })) return;
  std::size_t low = SIZE_MAX, high = 0;
  for (std::size_t k = 0; k < pairs.size(); ++k)
    if (pending[k]) low = std::min(low, rank[pairs[k].from.value()]), high = std::max(high, rank[pairs[k].to.value()]);
  std::vector<std::pair<std::size_t, std::uint8_t>> inside; // window-relative fence positions
  for (const auto& [event, spaces] : fences)
    if (const auto r = rank[event.value()]; r >= low && r <= high) inside.push_back({r - low, spaces});
  if (inside.empty() || high < low) return;
  // Successors inside the window, in window-relative ranks.
  const std::size_t size = high - low + 1;
  std::vector<std::size_t> offsets(size + 1);
  std::vector<std::uint32_t> successors;
  for (std::size_t v = 0; v < size; ++v) {
    offsets[v] = successors.size();
    for (const auto w : graph.successors(order[low + v]))
      if (const auto r = rank[w.value()]; r <= high) successors.push_back(static_cast<std::uint32_t>(r - low));
  }
  offsets[size] = successors.size();
  std::vector<std::uint64_t> reaches(size), reached(size);
  for (std::size_t begin = 0; begin < inside.size(); begin += 64) {
    const auto end = std::min(inside.size(), begin + 64);
    std::fill(reaches.begin(), reaches.end(), 0);
    std::fill(reached.begin(), reached.end(), 0);
    std::array<std::uint64_t, 4> covers{}; // fences of this chunk covering the needed spaces
    for (auto f = begin; f < end; ++f) {
      const auto bit = std::uint64_t{1} << (f - begin);
      reaches[inside[f].first] |= bit;
      reached[inside[f].first] |= bit;
      for (std::uint8_t needed = 1; needed <= kBothSpaces; ++needed)
        if ((inside[f].second & needed) == needed) covers[needed] |= bit;
    }
    for (std::size_t v = size; v-- > 0;)
      for (auto e = offsets[v]; e < offsets[v + 1]; ++e) reaches[v] |= reaches[successors[e]];
    for (std::size_t v = 0; v < size; ++v)
      for (auto e = offsets[v]; e < offsets[v + 1]; ++e) reached[successors[e]] |= reached[v];
    bool left = false;
    for (std::size_t k = 0; k < pairs.size(); ++k) {
      if (!pending[k]) continue;
      if (reaches[rank[pairs[k].from.value()] - low] & reached[rank[pairs[k].to.value()] - low] & covers[pairs[k].needed])
        pending[k] = 0;
      else left = true;
    }
    if (!left) return;
  }
}

}  // namespace

Result<ProxyFenceResult, MemoryAnalysisError> CheckProxyFences(
    const SourceProgram& source, const HappensBeforeGraph& graph, StaticHbQueries& queries,
    MemoryModel& memory, MemoryAccessView accesses) {
  using Output = Result<ProxyFenceResult, MemoryAnalysisError>;
  ProxyFenceResult result;
  // Memory accesses of the generic and async proxies (registers and tensormap reads excluded).
  std::vector<std::size_t> async_reads, async_writes;
  DynamicBitset generic(accesses.size()), generic_writes(accesses.size());
  for (std::size_t i = 0; i < accesses.size(); ++i) {
    const auto access = accesses[i];
    if (!std::holds_alternative<AddressId>(access.address) || (access.bytes && *access.bytes == 0)) continue;
    if (access.proxy == MemoryProxy::kAsync) {
      (access.mode == AccessMode::kRead ? async_reads : async_writes).push_back(i);
    } else if (access.proxy == MemoryProxy::kGeneric) {
      generic.set(i);
      if (access.mode == AccessMode::kWrite) generic_writes.set(i);
    }
  }
  result.stats.async_accesses = async_reads.size() + async_writes.size();
  if (!result.stats.async_accesses) return Output::Success(std::move(result));
  // The generic accesses on storage an async access may share: generic writes for async reads, and
  // every generic access for async writes. One filter query per distinct address.
  DynamicBitset before_reads(accesses.size()), before_writes(accesses.size()), shared_storage(accesses.size());
  if (auto filter = memory.storageFilter(accesses)) {
    const auto keep = [&](const std::vector<std::size_t>& async, const DynamicBitset& candidates, DynamicBitset& kept) {
      std::unordered_set<std::uint32_t> seen;
      for (const auto i : async) {
        if (!seen.insert(std::get<AddressId>(accesses[i].address).value()).second) continue;
        auto compatible = candidates;
        filter->retainCompatible(AccessId(static_cast<std::uint32_t>(i)), compatible);
        kept |= compatible;
      }
    };
    keep(async_reads, generic_writes, before_reads);
    keep(async_writes, generic, before_writes);
    keep(async_reads, generic, shared_storage);
    shared_storage |= before_writes;
  } else {
    before_reads = generic_writes;
    before_writes = generic;
    shared_storage = generic;
  }
  // Candidate (generic, async) pairs: the model's stream over each async set and its generic
  // accesses, so that no async/async pair is enumerated.
  struct Pair {
    std::uint32_t generic, async; // indices into `pool`
    std::uint8_t needed;          // fence spaces
    bool known;                   // a known overlap with a constant region
  };
  std::vector<MemoryAccess> pool;
  std::unordered_map<std::size_t, std::uint32_t> pooled;
  const auto pooled_index = [&](std::size_t index, const MemoryAccess& access) {
    const auto [slot, added] = pooled.try_emplace(index, static_cast<std::uint32_t>(pool.size()));
    if (added) pool.push_back(access);
    return slot->second;
  };
  std::vector<Pair> pairs;
  const auto collect = [&](const std::vector<std::size_t>& async, const DynamicBitset& kept) {
    std::vector<MemoryAccess> subset;
    std::vector<std::size_t> origin;
    for (const auto i : async) subset.push_back(accesses[i]), origin.push_back(i);
    for (std::size_t i = 0; i < accesses.size(); ++i)
      if (kept.test(i)) subset.push_back(accesses[i]), origin.push_back(i);
    if (subset.size() == async.size()) return true;
    for (std::size_t i = 0; i < subset.size(); ++i) subset[i].id = AccessId(static_cast<std::uint32_t>(i));
    const auto add = [&](std::size_t g, std::size_t a) {
      if (subset[g].mode == AccessMode::kRead && subset[a].mode == AccessMode::kRead) return;
      const auto overlap = memory.mayOverlap(subset[g], subset[a]);
      if (overlap.status == OverlapStatus::kDisjoint) return;
      const bool known = overlap.status == OverlapStatus::kOverlap;
      if (const auto needed = NeededSpaces(known ? overlap.evidence : std::nullopt))
        pairs.push_back({pooled_index(origin[g], subset[g]), pooled_index(origin[a], subset[a]), needed, known});
    };
    const auto count = async.size();
    if (auto stream = memory.conflictCandidates(MemoryAccessView(subset))) {
      while (const auto pair = stream->next()) {
        if (pair->first.value() >= subset.size() || pair->second.value() >= subset.size()) return false;
        const bool first_async = pair->first.value() < count, second_async = pair->second.value() < count;
        if (first_async == second_async) continue;
        if (first_async) add(pair->second.value(), pair->first.value());
        else add(pair->first.value(), pair->second.value());
      }
    } else {
      for (std::size_t a = 0; a < count; ++a)
        for (std::size_t g = count; g < subset.size(); ++g) add(g, a);
    }
    return true;
  };
  auto& hb = queries.get();
  const auto order = graph.topologicalOrder();
  if (!order) return Output::Failure({"proxy fence check requires an acyclic graph", {}, {}});
  std::vector<std::size_t> rank(graph.eventCount());
  for (std::size_t i = 0; i < order->size(); ++i) rank[(*order)[i].value()] = i;
  // The model's conflict-order stream over the async accesses and the generic accesses on their storage:
  // a sufficient set of conflicting pairs, oriented by rank. When happens-before orders every pair of
  // it, every conflicting pair is joined by a chain of them, and a chain from a generic access to a
  // later async access has a generic-then-async link: a fence on that link lies on the pair's path.
  // So its generic-then-async pairs are the ones to check. Otherwise (an exempt unordered pair, such as
  // duplicate TMA stores), the complete candidate streams are used.
  const auto reduced = [&]() -> bool {
    std::vector<MemoryAccess> subset;
    std::vector<std::size_t> origin;
    for (const auto* async : {&async_reads, &async_writes})
      for (const auto i : *async) subset.push_back(accesses[i]), origin.push_back(i);
    const auto count = subset.size();
    for (std::size_t i = 0; i < accesses.size(); ++i)
      if (shared_storage.test(i)) subset.push_back(accesses[i]), origin.push_back(i);
    for (std::size_t i = 0; i < subset.size(); ++i) subset[i].id = AccessId(static_cast<std::uint32_t>(i));
    auto stream = memory.conflictOrder(MemoryAccessView(subset), rank);
    if (!stream) return false;
    std::vector<HbQuery> links;
    std::vector<Pair> cross;
    while (const auto pair = stream->next()) {
      if (pair->first.value() >= subset.size() || pair->second.value() >= subset.size()) return false;
      auto first = pair->first.value(), second = pair->second.value();
      if (rank[subset[first].event.value()] > rank[subset[second].event.value()]) std::swap(first, second);
      links.push_back({subset[first].event, subset[second].event});
      if (first < count || second >= count) continue; // not generic then async
      const auto overlap = memory.mayOverlap(subset[first], subset[second]);
      if (overlap.status == OverlapStatus::kDisjoint) continue;
      const bool known = overlap.status == OverlapStatus::kOverlap;
      if (const auto needed = NeededSpaces(known ? overlap.evidence : std::nullopt))
        cross.push_back({pooled_index(origin[first], subset[first]), pooled_index(origin[second], subset[second]),
                         needed, known});
    }
    const auto ordered = Reaches(hb, links);
    if (!std::all_of(ordered.begin(), ordered.end(), [](auto answer) { return answer != 0; })) return false;
    pairs = std::move(cross);
    return true;
  }();
  if (!reduced) {
    pairs.clear();
    if (!collect(async_reads, before_reads) || !collect(async_writes, before_writes))
      return Output::Failure({"memory candidate references an unknown access", {}, {}});
  }
  // The pairs that happens-before orders generic first (all of them in the reduced stream).
  if (!reduced) {
    std::vector<HbQuery> generic_first;
    generic_first.reserve(pairs.size());
    for (const auto& pair : pairs) generic_first.push_back({pool[pair.generic].event, pool[pair.async].event});
    const auto ordered = Reaches(hb, generic_first);
    std::size_t kept = 0;
    for (std::size_t i = 0; i < pairs.size(); ++i)
      if (ordered[i]) pairs[kept++] = pairs[i];
    pairs.resize(kept);
  }
  result.stats.ordered_pairs = pairs.size();
  const FenceIndex fences(source, graph);
  result.stats.fences = fences.count();
  if (pairs.empty()) return Output::Success(std::move(result));
  if (!fences.chained(hb)) return Output::Failure({"a thread's proxy fences are not in program order", {}, {}});
  // For each generic access, the first fence of its thread that it reaches; for each async access, the
  // last fence of its issuing thread that reaches it. Binary searches over each thread's fences
  // (a happens-before chain), advanced together with one batch of queries per step.
  struct Search {
    EventId event;
    const std::vector<std::pair<InstId, EventId>>* list;
    std::size_t low, high;
    bool after; // the first fence the event reaches; else the last fence reaching the event
    [[nodiscard]] std::size_t found() const {
      return after ? (low < list->size() ? low : kNone) : (low > 0 ? low - 1 : kNone);
    }
  };
  std::vector<Search> searches;
  std::unordered_map<std::uint64_t, std::size_t> after_search, before_search;
  const auto key = [](EventId event, std::uint8_t needed) { return (std::uint64_t(event.value()) << 2) | needed; };
  std::vector<std::pair<std::size_t, std::size_t>> pair_searches; // (after, before) per pair
  for (const auto& pair : pairs) {
    const auto& g = pool[pair.generic];
    const auto& a = pool[pair.async];
    const auto [after, added_after] = after_search.try_emplace(key(g.event, pair.needed), searches.size());
    if (added_after) {
      const auto& list = fences.of(g.actor, pair.needed);
      searches.push_back({g.event, &list, 0, list.size(), true});
    }
    const auto [before, added_before] = before_search.try_emplace(key(a.event, pair.needed), searches.size());
    if (added_before) {
      const auto& list = fences.of(a.actor, pair.needed);
      searches.push_back({a.event, &list, 0, list.size(), false});
    }
    pair_searches.emplace_back(after->second, before->second);
  }
  for (std::vector<HbQuery> step;; step.clear()) {
    std::vector<std::size_t> active;
    for (std::size_t k = 0; k < searches.size(); ++k) {
      const auto& search = searches[k];
      if (search.low >= search.high) continue;
      const auto fence = (*search.list)[search.low + (search.high - search.low) / 2].second;
      step.push_back(search.after ? HbQuery{search.event, fence} : HbQuery{fence, search.event});
      active.push_back(k);
    }
    if (step.empty()) break;
    const auto answers = Reaches(hb, step);
    for (std::size_t j = 0; j < active.size(); ++j) {
      auto& search = searches[active[j]];
      const auto middle = search.low + (search.high - search.low) / 2;
      if (search.after == bool(answers[j])) search.high = middle;
      else search.low = middle + 1;
    }
  }
  // A pair is fenced when g's fence reaches a, or g reaches a's fence.
  std::vector<std::uint8_t> fenced(pairs.size());
  for (int side = 0; side < 2; ++side) {
    std::vector<HbQuery> step;
    std::vector<std::size_t> which;
    for (std::size_t i = 0; i < pairs.size(); ++i) {
      if (fenced[i]) continue;
      const auto& search = searches[side ? pair_searches[i].second : pair_searches[i].first];
      const auto k = search.found();
      if (k == kNone) continue;
      const auto fence = (*search.list)[k].second;
      step.push_back(side ? HbQuery{pool[pairs[i].generic].event, fence} : HbQuery{fence, pool[pairs[i].async].event});
      which.push_back(i);
    }
    const auto answers = Reaches(hb, step);
    for (std::size_t j = 0; j < which.size(); ++j) fenced[which[j]] = answers[j];
  }
  // The rest: a fence in any thread on a path between them.
  {
    std::vector<EndpointPair> endpoints;
    std::vector<std::uint8_t> pending;
    for (std::size_t i = 0; i < pairs.size(); ++i) {
      endpoints.push_back({pool[pairs[i].generic].event, pool[pairs[i].async].event, pairs[i].needed});
      pending.push_back(!fenced[i]);
    }
    ClearFenced(graph, *order, rank, fences.all(), endpoints, pending);
    for (std::size_t i = 0; i < pairs.size(); ++i) fenced[i] = !pending[i];
  }
  // The pairs without a fence, by kind and PTX sites.
  std::map<std::tuple<const char*, std::uint32_t, std::uint32_t>, std::size_t> sites;
  std::optional<std::size_t> witness;
  bool witness_is_raw = false;
  const auto ptx = [&](const MemoryAccess& access) -> std::optional<std::uint32_t> {
    const auto origin = HeaderOf(source.instructions, access.instruction.value()).origin;
    return origin ? std::optional(origin->first) : std::nullopt;
  };
  for (std::size_t i = 0; i < pairs.size(); ++i) {
    if (fenced[i]) continue;
    const auto& g = pool[pairs[i].generic];
    const auto& a = pool[pairs[i].async];
    if (!pairs[i].known || pairs[i].needed == kBothSpaces) {
      // Unknown overlap or state space: no fence covering every space orders the pair.
      ++result.stats.unresolved;
      if (!result.first_unresolved) result.first_unresolved.emplace(g, a);
      continue;
    }
    const bool raw = g.mode == AccessMode::kWrite && a.mode == AccessMode::kRead;
    const char* kind = g.mode == AccessMode::kRead ? "write_after_read" : raw ? "read_after_write" : "write_after_write";
    ++(g.mode == AccessMode::kRead ? result.stats.write_after_read
       : raw                       ? result.stats.read_after_write
                                   : result.stats.write_after_write);
    const auto generic_ptx = ptx(g), async_ptx = ptx(a);
    const auto [site, added] = sites.try_emplace({kind, generic_ptx.value_or(UINT32_MAX), async_ptx.value_or(UINT32_MAX)},
                                                 result.sites.size());
    if (added && result.sites.size() < kMaxProxyFenceSites)
      result.sites.push_back({kind, g, a, generic_ptx, async_ptx, 0});
    if (site->second < result.sites.size()) ++result.sites[site->second].count;
    // The witness: the first read after write when there is one (a stale read), else the first pair.
    if (!witness || (!witness_is_raw && raw)) witness = i, witness_is_raw = raw;
  }
  if (witness) {
    const auto& g = pool[pairs[*witness].generic];
    const auto& a = pool[pairs[*witness].async];
    auto overlap = memory.mayOverlap(g, a);
    if (overlap.status != OverlapStatus::kOverlap || !overlap.evidence)
      return Output::Failure({"proxy fence witness lost its overlap", {}, {}});
    result.first = ProxyFenceFinding{g, a, std::move(*overlap.evidence)};
  }
  return Output::Success(std::move(result));
}

}  // namespace sentryward::algorithm
