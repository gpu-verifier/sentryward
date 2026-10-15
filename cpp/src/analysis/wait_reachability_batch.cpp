#include "sentryward/algorithm/analysis/wait_reachability_batch.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>

#include "sentryward/algorithm/diagnostics/memory_profile.h"
#include "sentryward/algorithm/diagnostics/performance_profile.h"
#include "sentryward/algorithm/analysis/family_hb_reachability.h"

namespace sentryward::algorithm {
namespace {

constexpr auto kNone = std::numeric_limits<std::uint32_t>::max();

struct QuerySlot final {
  std::uint32_t query;
  std::uint32_t component = 0;
};

struct Component final {
  std::uint32_t root;
  std::uint32_t begin = 0;
  std::uint32_t size = 0;
  std::uint32_t cursor = 0;
};

struct Seed final {
  std::uint32_t position;
  std::uint32_t phase;
  std::uint32_t bit;
};

template <typename T>
std::size_t AllocatedBytes(const std::vector<T>& values) {
  return values.capacity() * sizeof(T);
}

// The same array becomes event->packed component position once union/find ends.
// Negative roots encode component size, avoiding a separate rank/size array.
std::uint32_t Find(std::vector<std::int32_t>& parents, std::uint32_t vertex) {
  auto root = vertex;
  while (parents[root] >= 0) root = static_cast<std::uint32_t>(parents[root]);
  while (vertex != root) {
    const auto next = static_cast<std::uint32_t>(parents[vertex]);
    parents[vertex] = static_cast<std::int32_t>(root);
    vertex = next;
  }
  return root;
}

void Unite(std::vector<std::int32_t>& parents, EventId from, EventId to) {
  auto a = Find(parents, from.value());
  auto b = Find(parents, to.value());
  if (a == b) return;
  if (parents[a] > parents[b]) std::swap(a, b);
  parents[a] += parents[b];
  parents[b] = static_cast<std::int32_t>(a);
}

bool AddBytes(std::size_t& total, std::size_t count, std::size_t width,
              std::size_t limit) {
  if (total > limit || (width && count > (limit - total) / width)) return false;
  total += count * width;
  return true;
}

bool Full(const std::uint64_t* bits, std::size_t count) {
  const auto words = (count + 63) / 64;
  for (std::size_t word = 0; word < words; ++word) {
    const auto remaining = count - 64 * word;
    const auto expected = remaining >= 64
        ? std::numeric_limits<std::uint64_t>::max()
        : (std::uint64_t{1} << remaining) - 1;
    if (bits[word] != expected) return false;
  }
  return true;
}

}  // namespace

const WaitReachabilityAnswer* WaitReachabilityBatch::find(EventId issue) const noexcept {
  const auto found = std::lower_bound(answers_.begin(), answers_.end(), issue,
      [](const auto& answer, EventId event) { return answer.issue < event; });
  return found != answers_.end() && found->issue == issue ? &*found : nullptr;
}

WaitReachabilityBatch WaitReachabilityBatch::Build(
    const HappensBeforeGraph& graph, std::span<const PhaseInfo> phases,
    std::span<const WaitReachabilityQuery> queries,
    WaitReachabilityBatchOptions options) {
  ProfileScope total_profile("wait.batch.build");
  WaitReachabilityBatch result;
  const auto* order = graph.traceOrder();
  const auto count = graph.eventCount();
  const auto limit = options.scratch_limit_bytes;
  if (!options.certified_phase_ordering || !order || queries.empty() ||
      !count || count > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
      queries.size() > kNone) {
    total_profile.pruned = queries.size();
    return result;
  }

  // Include retained answers and all small indexing vectors in the budget.
  if (queries.size() <= limit / (4 * sizeof(WaitReachabilityAnswer))) {
    HbReachabilityOptions family_options;
    family_options.index_budget_bytes = limit / 4;
    auto forward = TryFamilyHbReachability(graph, family_options);
    if (forward) {
      ProfileScope family_profile("wait.batch.family_relations");
      auto reverse = TryFamilyHbReachability(graph, family_options, true);
      std::map<PhaseKey, const PhaseInfo*> phase_map;
      std::map<BarrierLifetimeId, std::set<EventId>> effects;
      for (const auto& phase : phases) {
        if (!phase_map.emplace(PhaseKey{phase.barrier, phase.phase}, &phase).second) return {};
        if (phase.completed && phase.effects.empty()) return {};
        for (auto effect : phase.effects)
          if (!graph.contains(effect) || !effects[phase.barrier].insert(effect).second) return {};
      }
      std::map<BarrierLifetimeId, std::pair<std::uint32_t, bool>> sequence;
      for (const auto& [key, phase] : phase_map) {
        auto& [next, open] = sequence[key.barrier];
        if (key.phase != next++ || open) return {};
        open = !phase->completed;
      }
      using Key = std::tuple<BarrierLifetimeId, EventId, std::optional<std::uint32_t>, std::optional<std::uint32_t>>;
      std::map<Key, std::vector<std::size_t>> groups;
      std::set<EventId> seen;
      result.answers_.resize(queries.size());
      for (std::size_t i = 0; i < queries.size(); ++i) {
        const auto& q = queries[i];
        result.answers_[i].issue = q.issue;
        if (!seen.insert(q.issue).second) return {};
        if (graph.contains(q.issue) && graph.contains(q.initialization))
          groups[{q.barrier, q.initialization, q.previous_phase, q.next_phase}].push_back(i);
      }
      for (const auto& [key, indices] : groups) {
        const auto& [barrier, init, previous, next] = key;
        std::vector<EventId> targets;
        targets.reserve(indices.size());
        for (auto i : indices) targets.push_back(queries[i].issue);
        const auto initialized = forward->allReachMany(std::span(&init, 1), targets);
        for (std::size_t i = 0; i < indices.size(); ++i) result.answers_[indices[i]].initialization = bool(initialized[i]);
        if (previous) {
          const auto phase = phase_map.find({barrier, *previous});
          if (phase != phase_map.end() && phase->second->completed) {
            const auto lower = forward->allReachMany(phase->second->effects, targets);
            for (std::size_t i = 0; i < indices.size(); ++i) result.answers_[indices[i]].lower = bool(lower[i]);
          }
        }
        if (next) {
          const auto phase = phase_map.find({barrier, *next});
          if (phase != phase_map.end() && phase->second->completed) {
            const auto upper = reverse->findAncestors(phase->second->effects, targets);
            for (std::size_t i = 0; i < indices.size(); ++i) {
              result.answers_[indices[i]].upper_indexed = true;
              result.answers_[indices[i]].upper_witness = upper[i];
            }
          }
        }
      }
      std::sort(result.answers_.begin(), result.answers_.end(), [](const auto& a, const auto& b) { return a.issue < b.issue; });
      return result;
    }
  }

  // Include retained answers and all small indexing vectors in the budget.
  // Component/root reserves below never exceed the query count charged here.
  std::size_t base_bytes = 0;
  if (!AddBytes(base_bytes, count, sizeof(std::int32_t), limit) ||
      !AddBytes(base_bytes, queries.size(), sizeof(WaitReachabilityAnswer) +
          sizeof(QuerySlot) + sizeof(std::uint32_t) + sizeof(Component), limit) ||
      !AddBytes(base_bytes, phases.size(), sizeof(const PhaseInfo*), limit)) {
    total_profile.pruned = queries.size();
    return result;
  }

  result.answers_.resize(queries.size());
  std::vector<QuerySlot> slots;
  slots.reserve(queries.size());
  for (std::size_t i = 0; i < queries.size(); ++i) {
    result.answers_[i].issue = queries[i].issue;
    slots.push_back({static_cast<std::uint32_t>(i), 0});
  }
  std::sort(slots.begin(), slots.end(), [&](auto a, auto b) {
    return queries[a.query].issue < queries[b.query].issue;
  });
  for (std::size_t i = 1; i < slots.size(); ++i) {
    if (queries[slots[i - 1].query].issue == queries[slots[i].query].issue)
      return {};  // An issue cannot name two different query descriptors.
  }
  slots.erase(std::remove_if(slots.begin(), slots.end(), [&](auto slot) {
    return !graph.contains(queries[slot.query].issue) ||
           !graph.contains(queries[slot.query].initialization);
  }), slots.end());
  if (slots.empty()) return {};

  std::vector<const PhaseInfo*> phase_order;
  phase_order.reserve(phases.size());
  for (const auto& phase : phases) phase_order.push_back(&phase);
  std::sort(phase_order.begin(), phase_order.end(), [](auto a, auto b) {
    return std::tie(a->barrier, a->phase) < std::tie(b->barrier, b->phase);
  });

  std::vector<std::int32_t> positions(count, -1);
  std::vector<Component> components;
  std::vector<EventId> ordered;
  {
    ProfileScope component_profile("wait.batch.components");
    component_profile.nodes = count;
    // Actual current connectivity, including every derived edge; CTA identity
    // is deliberately not used as a partitioning assumption.
    for (const auto& event : graph.events()) {
      for (auto next : graph.successors(event.id)) {
        Unite(positions, event.id, next);
        ++component_profile.edges;
      }
    }
    std::vector<std::uint32_t> roots;
    roots.reserve(slots.size());
    for (auto slot : slots) roots.push_back(Find(positions, queries[slot.query].issue.value()));
    std::sort(roots.begin(), roots.end());
    roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
    components.reserve(roots.size());
    for (auto root : roots) components.push_back({root});
    for (auto& slot : slots) {
      const auto root = Find(positions, queries[slot.query].issue.value());
      slot.component = static_cast<std::uint32_t>(
          std::lower_bound(roots.begin(), roots.end(), root) - roots.begin());
    }
    // Flatten completely before overwriting any parent with a component index.
    for (std::uint32_t v = 0; v < count; ++v) (void)Find(positions, v);
    std::size_t max_rank = 0;
    for (std::uint32_t v = 0; v < count; ++v) {
      const auto root = positions[v] < 0 ? v : static_cast<std::uint32_t>(positions[v]);
      const auto component = std::lower_bound(roots.begin(), roots.end(), root);
      if (component == roots.end() || *component != root) {
        positions[v] = -1;
      } else {
        const auto index = static_cast<std::uint32_t>(component - roots.begin());
        positions[v] = static_cast<std::int32_t>(index);
        ++components[index].size;
      }
      max_rank = std::max(max_rank, order->rank(EventId(v)));
    }
    std::uint32_t packed_count = 0;
    for (auto& component : components) {
      component.begin = component.cursor = packed_count;
      packed_count += component.size;
    }
    std::size_t topology_bytes = base_bytes;
    if (max_rank == std::numeric_limits<std::size_t>::max() ||
        !AddBytes(topology_bytes, packed_count, sizeof(EventId), limit) ||
        !AddBytes(topology_bytes, max_rank + 1, sizeof(std::uint32_t), limit))
      return {};
    ordered.resize(packed_count);
    // Rank is the already validated trace-entry position, not an EventId.
    // Its inverse avoids sorting tens of millions of events by rank.
    std::vector<std::uint32_t> inverse(max_rank + 1, kNone);
    for (std::uint32_t v = 0; v < count; ++v) {
      const auto rank = order->rank(EventId(v));
      if (inverse[rank] != kNone) return {};
      inverse[rank] = v;
    }
    for (auto v : inverse) {
      if (v == kNone || positions[v] < 0) continue;
      auto& component = components[static_cast<std::uint32_t>(positions[v])];
      const auto position = component.cursor++;
      ordered[position] = EventId(v);
      positions[v] = static_cast<std::int32_t>(position);
    }
    if (MemoryProfileEnabled()) {
      MemoryReport report("wait.batch.topology", "WaitReachabilityBatch", &result);
      report.Vector("event_positions", positions);
      report.Vector("component_vertices", ordered);
      report.Vector("inverse_trace_order", inverse);
      report.Vector("components", components);
      report.Vector("component_roots", roots);
      report.Vector("queries", slots);
      report.Vector("phase_pointers", phase_order);
      report.Vector("answers", result.answers_);
      report.Emit();
      MemoryCheckpoint("wait.batch.topology.ready");
    }
  }

  AddBytes(base_bytes, ordered.size(), sizeof(EventId), limit);
  std::sort(slots.begin(), slots.end(), [&](auto a, auto b) {
    const auto& x = queries[a.query];
    const auto& y = queries[b.query];
    return std::tie(x.barrier, x.initialization, a.component) <
           std::tie(y.barrier, y.initialization, b.component);
  });
  std::size_t peak_group_bytes = 0;
  for (std::size_t begin = 0; begin < slots.size();) {
    const auto& first = queries[slots[begin].query];
    const auto component_id = slots[begin].component;
    std::size_t end = begin + 1;
    while (end < slots.size()) {
      const auto& next = queries[slots[end].query];
      if (next.barrier != first.barrier || next.initialization != first.initialization ||
          slots[end].component != component_id) break;
      ++end;
    }
    ProfileScope group_profile("wait.batch.group");
    group_profile.nodes = end - begin;
    const auto phase_begin = std::lower_bound(phase_order.begin(), phase_order.end(), first.barrier,
        [](const auto* phase, auto barrier) { return phase->barrier < barrier; });
    const auto phase_end = std::upper_bound(phase_begin, phase_order.end(), first.barrier,
        [](auto barrier, const auto* phase) { return barrier < phase->barrier; });
    std::size_t completed = 0, max_effects = 0, effect_count = 0;
    bool chain_valid = phase_begin != phase_end;
    bool saw_open = false;
    for (auto it = phase_begin; it != phase_end; ++it) {
      const auto& phase = **it;
      if (phase.phase != completed || saw_open) { chain_valid = false; break; }
      if (!phase.completed) { saw_open = true; continue; }
      if (phase.effects.empty()) { chain_valid = false; break; }
      ++completed;
      max_effects = std::max(max_effects, phase.effects.size());
      if (!AddBytes(effect_count, phase.effects.size(), 1, limit)) {
        chain_valid = false; break;
      }
      for (auto event : phase.effects)
        if (!graph.contains(event)) { chain_valid = false; break; }
      if (!chain_valid) break;
    }
    if (!chain_valid || completed > kNone) {
      group_profile.pruned = end - begin;
      begin = end;
      continue;
    }
    bool need_lower = false, need_upper = false;
    for (auto i = begin; i < end; ++i) {
      const auto& query = queries[slots[i].query];
      need_lower |= query.previous_phase && *query.previous_phase < completed;
      need_upper |= query.next_phase && *query.next_phase < completed;
    }
    const auto& component = components[component_id];
    const std::size_t size = component.size;
    const auto stop = component.begin + component.size;
    const auto init_position = positions[first.initialization.value()];
    const bool init_in_component = init_position >= 0 &&
        static_cast<std::uint32_t>(init_position) >= component.begin &&
        static_cast<std::uint32_t>(init_position) < stop;
    const auto words = need_lower ? (max_effects + 63) / 64 : 0;
    const std::size_t forward_width = sizeof(std::uint8_t) +
        (need_lower ? sizeof(std::uint32_t) + words * sizeof(std::uint64_t) : 0);
    const std::size_t reverse_width = need_upper ? 2 * sizeof(std::uint32_t) : 0;
    std::size_t group_bytes = base_bytes;
    if (!AddBytes(group_bytes, effect_count, sizeof(Seed) + sizeof(EventId), limit) ||
        !AddBytes(group_bytes, size, std::max(forward_width, reverse_width), limit)) {
      group_profile.pruned = end - begin;
      begin = end;
      continue;
    }
    std::vector<Seed> seeds;
    seeds.reserve(effect_count);
    {
      // Metadata is normally validated by PhaseDatabase, but this modular
      // entry point also accepts public PhaseInfo spans. Reject duplicate IDs
      // anywhere in the barrier, including components with no query vertices.
      std::vector<EventId> effect_ids;
      effect_ids.reserve(effect_count);
      for (std::size_t p = 0; p < completed; ++p) {
        const auto& effects = (*(phase_begin + p))->effects;
        effect_ids.insert(effect_ids.end(), effects.begin(), effects.end());
      }
      std::sort(effect_ids.begin(), effect_ids.end());
      if (std::adjacent_find(effect_ids.begin(), effect_ids.end()) != effect_ids.end())
        chain_valid = false;
    }
    if (need_lower || need_upper) {
      for (std::size_t p = 0; p < completed; ++p) {
        const auto& effects = (*(phase_begin + p))->effects;
        if (effects.size() > kNone) { chain_valid = false; break; }
        for (std::size_t bit = 0; bit < effects.size(); ++bit) {
          const auto position = positions[effects[bit].value()];
          if (position >= 0 && static_cast<std::uint32_t>(position) >= component.begin &&
              static_cast<std::uint32_t>(position) < stop)
            seeds.push_back({static_cast<std::uint32_t>(position),
                             static_cast<std::uint32_t>(p), static_cast<std::uint32_t>(bit)});
        }
      }
      std::sort(seeds.begin(), seeds.end(), [](auto a, auto b) {
        return std::tie(a.position, a.phase, a.bit) < std::tie(b.position, b.phase, b.bit);
      });
      for (std::size_t i = 1; i < seeds.size(); ++i)
        if (seeds[i - 1].position == seeds[i].position) chain_valid = false;
    }
    if (!chain_valid) {
      group_profile.pruned = end - begin;
      begin = end;
      continue;
    }
    const auto memory_checkpoint = [&](auto& first_vector, auto& second_vector,
                                       auto& third_vector, const char* stage) {
      if (!MemoryProfileEnabled()) return;
      const auto reserved = AllocatedBytes(positions) + AllocatedBytes(ordered) +
          AllocatedBytes(components) + AllocatedBytes(slots) + AllocatedBytes(phase_order) +
          AllocatedBytes(result.answers_) + AllocatedBytes(seeds) +
          AllocatedBytes(first_vector) + AllocatedBytes(second_vector) +
          AllocatedBytes(third_vector);
      if (reserved <= peak_group_bytes) return;
      peak_group_bytes = reserved;
      MemoryReport report(stage, "WaitReachabilityBatch", &result);
      report.Vector("event_positions", positions);
      report.Vector("component_vertices", ordered);
      report.Vector("components", components);
      report.Vector("queries", slots);
      report.Vector("phase_pointers", phase_order);
      report.Vector("answers", result.answers_);
      report.Vector("effect_seeds", seeds);
      report.Vector("labels", first_vector);
      report.Vector("payload", second_vector);
      report.Vector("flags", third_vector);
      report.Emit();
      const auto detail = "barrier=" + std::to_string(first.barrier.value()) +
          " component_vertices=" + std::to_string(size) +
          " scratch_reserved_bytes=" + std::to_string(reserved) +
          " bounded_bytes=" + std::to_string(group_bytes);
      MemoryCheckpoint(stage, detail.c_str());
    };
    if (init_in_component || need_lower) {
      ProfileScope forward_profile("wait.batch.forward");
      std::vector<std::uint32_t> latest(need_lower ? size : 0, kNone);
      std::vector<std::uint64_t> masks(need_lower ? size * words : 0, 0);
      // Bit 0 is initialization reachability; bit 1 is full latest-phase coverage.
      std::vector<std::uint8_t> flags(size, 0);
      memory_checkpoint(latest, masks, flags, "wait.batch.forward.ready");
      std::size_t seed = 0;
      for (std::size_t local = 0; local < size; ++local) {
        const auto position = component.begin + local;
        const auto event = ordered[position];
        auto* mask = need_lower ? masks.data() + local * words : nullptr;
        if (event == first.initialization) flags[local] = 1;
        for (auto predecessor : graph.predecessors(event)) {
          const auto previous = static_cast<std::uint32_t>(positions[predecessor.value()]);
          if (previous < component.begin || previous >= position) {
            chain_valid = false; break;
          }
          const auto from = previous - component.begin;
          flags[local] |= flags[from] & 1;
          ++forward_profile.edges;
          if (!need_lower || latest[from] == kNone) continue;
          const auto* incoming = masks.data() + from * words;
          if (latest[local] == kNone || latest[from] > latest[local]) {
            latest[local] = latest[from];
            std::copy_n(incoming, words, mask);
          } else if (latest[from] == latest[local]) {
            for (std::size_t word = 0; word < words; ++word) mask[word] |= incoming[word];
          }
        }
        if (!chain_valid) break;
        while (seed < seeds.size() && seeds[seed].position == position) {
          const auto value = seeds[seed++];
          if (!need_lower) continue;
          if (latest[local] == kNone || value.phase > latest[local]) {
            latest[local] = value.phase;
            std::fill_n(mask, words, 0);
          }
          if (value.phase == latest[local]) mask[value.bit / 64] |= std::uint64_t{1} << (value.bit % 64);
        }
        if (need_lower && latest[local] != kNone &&
            Full(mask, (*(phase_begin + latest[local]))->effects.size())) flags[local] |= 2;
        ++forward_profile.nodes;
      }
      if (chain_valid) {
        for (auto i = begin; i < end; ++i) {
          const auto index = slots[i].query;
          const auto local = static_cast<std::uint32_t>(positions[queries[index].issue.value()]) - component.begin;
          auto& answer = result.answers_[index];
          answer.initialization = (flags[local] & 1) != 0;
          const auto previous = queries[index].previous_phase;
          if (previous && *previous < completed)
            answer.lower = latest[local] != kNone &&
                (*previous < latest[local] || (*previous == latest[local] && (flags[local] & 2)));
        }
      }
    } else {
      // Different weak components cannot reach one another.
      for (auto i = begin; i < end; ++i) result.answers_[slots[i].query].initialization = false;
    }
    if (chain_valid && need_upper) {
      ProfileScope reverse_profile("wait.batch.reverse");
      std::vector<std::uint32_t> earliest(size, kNone);
      std::vector<EventId> witnesses(size);
      std::vector<std::uint8_t> no_flags;
      memory_checkpoint(earliest, witnesses, no_flags, "wait.batch.reverse.ready");
      std::size_t seed = seeds.size();
      for (std::size_t local = size; local-- > 0;) {
        const auto position = component.begin + local;
        const auto event = ordered[position];
        while (seed > 0 && seeds[seed - 1].position == position) {
          const auto value = seeds[--seed];
          if (value.phase < earliest[local]) {
            earliest[local] = value.phase;
            witnesses[local] = event;
          }
        }
        for (auto successor : graph.successors(event)) {
          const auto following = static_cast<std::uint32_t>(positions[successor.value()]);
          if (following <= position || following >= stop) {
            chain_valid = false; break;
          }
          const auto to = following - component.begin;
          if (earliest[to] < earliest[local]) {
            earliest[local] = earliest[to];
            witnesses[local] = witnesses[to];
          }
          ++reverse_profile.edges;
        }
        if (!chain_valid) break;
        ++reverse_profile.nodes;
      }
      if (chain_valid) {
        for (auto i = begin; i < end; ++i) {
          const auto index = slots[i].query;
          const auto next = queries[index].next_phase;
          if (!next || *next >= completed) continue;
          auto& answer = result.answers_[index];
          answer.upper_indexed = true;
          const auto local = static_cast<std::uint32_t>(positions[queries[index].issue.value()]) - component.begin;
          if (earliest[local] > *next) continue;
          answer.upper_witness = earliest[local] == *next
              ? witnesses[local] : (*(phase_begin + *next))->effects.front();
        }
      }
    }
    if (!chain_valid) {
      for (auto i = begin; i < end; ++i) {
        auto& answer = result.answers_[slots[i].query];
        const auto issue = answer.issue;
        answer = {};
        answer.issue = issue;
      }
      group_profile.pruned = end - begin;
    }
    begin = end;
  }
  std::sort(result.answers_.begin(), result.answers_.end(), [](const auto& a, const auto& b) {
    return a.issue < b.issue;
  });
  total_profile.nodes = result.answers_.size();
  return result;
}

}  // namespace sentryward::algorithm
