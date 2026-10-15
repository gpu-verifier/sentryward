#include "sentryward/frontend/ptx/group2_lifetime.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace sentryward::frontend::ptx {
namespace {
using namespace algorithm;
using RefKey = std::tuple<std::uint32_t, Milestone, std::uint32_t>;
RefKey Key(const MilestoneRef &ref) {
  return {ref.instruction.value(), ref.milestone, ref.effect_index};
}
std::string CoordinatesText(Coordinates c) {
  return "(" + std::to_string(c[0]) + ", " + std::to_string(c[1]) + ", " +
         std::to_string(c[2]) + ")";
}
std::string AllocationText(TmemAllocationId id) {
  return "(" + std::to_string(id.warp) + ", " +
         std::to_string(id.instruction_index) + ", " +
         std::to_string(id.occurrence) + ")";
}
std::string Describe(const Group2LifetimeSite &site) {
  return "CTA " + CoordinatesText(site.cta_coordinates) + " warp " +
         std::to_string(site.warp) + " I" +
         std::to_string(site.instruction_index) + ".O" +
         std::to_string(site.occurrence);
}
bool ValidReference(const MilestoneRef &ref) {
  switch (ref.milestone) {
  case Milestone::kEffect:
    return true;
  case Milestone::kIssue:
  case Milestone::kComplete:
  case Milestone::kReadsComplete:
    return ref.effect_index == 0;
  }
  return false;
}
std::vector<MilestoneRef>
UniqueReferences(const std::vector<MilestoneRef> &values) {
  std::vector<MilestoneRef> result;
  std::set<RefKey> seen;
  for (const auto &value : values)
    if (seen.insert(Key(value)).second)
      result.push_back(value);
  return result;
}
const std::vector<std::string> &Premises() {
  static const std::vector<std::string> values{
      "Baseline synchronization and asynchronous work must independently make "
      "progress.",
      "Resident peer CTAs must be scheduled fairly and have admitted TMEM "
      "capacity.",
      "Allocation placement and retirement RequiredOrders are separate "
      "obligations.",
      "Source references, actor ordering, and collective participation must be "
      "validated."};
  return values;
}

// Commit/cp.async notification compression shares only this pure conjunction,
// never the enclosing operation's notification completion. Other effects retain
// the conservative whole-operation abstraction used by the progress proof.
bool HasNotificationPrefix(const SourceInstruction &instruction) {
  const auto *op = std::get_if<AsyncOp>(&instruction.op);
  if (!op || op->effects.size() < 2 ||
      !op->effects[0].accesses.empty() || op->effects[0].completion)
    return false;
  const MilestoneRef gate{instruction.id, Milestone::kEffect, 0};
  return std::all_of(op->effects.begin() + 1, op->effects.end(),
                     [&](const auto &effect) {
                       const auto *arrival = effect.completion
                           ? std::get_if<ArriveEffect>(&*effect.completion)
                           : nullptr;
                       return effect.accesses.empty() && arrival &&
                              arrival->arrivals == 1 &&
                              effect.predecessors.size() == 1 &&
                              effect.predecessors[0] == gate;
                     });
}

// A temporary progress dependency graph, not a happens-before graph. Flat edge
// assembly plus CSR avoids one heap allocation per event before iterative SCC.
struct ProgressGraph {
  std::size_t nodes = 0;
  std::vector<std::pair<std::size_t, std::size_t>> edges;
  std::size_t Node() { return nodes++; }
  void Edge(std::size_t from, std::size_t to) {
    if (from >= nodes || to >= nodes)
      throw std::invalid_argument("progress graph endpoint is invalid");
    edges.emplace_back(from, to);
  }
  std::vector<std::size_t> Components(const std::function<void()> &poll) const {
    std::vector<std::size_t> offsets(nodes + 1, 0);
    for (const auto &[from, to] : edges) {
      (void)to;
      ++offsets[from + 1];
    }
    for (std::size_t i = 1; i < offsets.size(); ++i)
      offsets[i] += offsets[i - 1];
    auto cursors = offsets;
    std::vector<std::size_t> targets(edges.size());
    for (const auto &[from, to] : edges)
      targets[cursors[from]++] = to;
    const auto missing = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> discovery(nodes, missing), low(nodes), stack,
        component(nodes, missing);
    std::vector<std::uint8_t> active(nodes, 0);
    struct Frame {
      std::size_t node, index;
    };
    std::vector<Frame> frames;
    std::size_t serial = 0, count = 0, iterations = 0;
    for (std::size_t start = 0; start < nodes; ++start) {
      if (discovery[start] != missing)
        continue;
      discovery[start] = low[start] = serial++;
      stack.push_back(start);
      active[start] = 1;
      frames.push_back({start, offsets[start]});
      while (!frames.empty()) {
        if ((++iterations & 4095) == 0 && poll)
          poll();
        auto &frame = frames.back();
        const auto node = frame.node;
        if (frame.index < offsets[node + 1]) {
          auto target = targets[frame.index++];
          if (discovery[target] == missing) {
            discovery[target] = low[target] = serial++;
            stack.push_back(target);
            active[target] = 1;
            frames.push_back({target, offsets[target]});
          } else if (active[target])
            low[node] = std::min(low[node], discovery[target]);
          continue;
        }
        frames.pop_back();
        if (!frames.empty())
          low[frames.back().node] =
              std::min(low[frames.back().node], low[node]);
        if (low[node] == discovery[node]) {
          while (true) {
            auto member = stack.back();
            stack.pop_back();
            active[member] = 0;
            component[member] = count;
            if (member == node)
              break;
          }
          ++count;
        }
      }
    }
    return component;
  }
};
} // namespace

Group2LifetimeResult
PairGroup2Lifetimes(std::span<const Group2LifetimeSite> sites,
                    Coordinates dimensions) {
  using namespace algorithm;
  if (std::any_of(dimensions.begin(), dimensions.end(),
                  [](auto x) { return x == 0; }))
    throw std::invalid_argument(
        "cluster_dim must contain three positive integers");
  const auto cluster_size =
      std::uint64_t(dimensions[0]) * dimensions[1] * dimensions[2];
  Group2LifetimeResult result;
  std::set<std::pair<std::string, std::string>> seen_limitations;
  const auto limit = [&](std::string code, std::string message) {
    if (seen_limitations.emplace(code, message).second)
      result.verification_limitations.push_back(
          {std::move(code), std::move(message)});
  };
  using BucketKey = std::tuple<Coordinates, std::uint64_t, std::string,
                               std::uint32_t, std::uint64_t>;
  std::map<BucketKey, std::size_t> bucket_ids;
  std::vector<std::vector<std::pair<unsigned, const Group2LifetimeSite *>>>
      buckets;
  std::vector<const Group2LifetimeSite *> valid;
  for (const auto &site : sites) {
    std::set<ActorId> participants(site.participant_actors.begin(),
                                   site.participant_actors.end());
    std::set<RefKey> issues;
    for (const auto &ref : site.issue_refs)
      issues.insert(Key(ref));
    bool well_formed =
        (site.operation == "alloc" || site.operation == "dealloc" ||
         site.operation == "relinquish_alloc_permit") &&
        site.participant_actors.size() == 32 && participants.size() == 32 &&
        site.issue_refs.size() == 32 && issues.size() == 32 &&
        std::all_of(site.issue_refs.begin(), site.issue_refs.end(),
                    [](const auto &ref) {
                      return ref.milestone == Milestone::kIssue;
                    }) &&
        (site.completion_refs.empty() || site.completion_refs.size() == 32);
    for (const auto *refs : {&site.issue_refs, &site.completion_refs,
                             &site.publication_refs, &site.accesses})
      well_formed &= std::all_of(refs->begin(), refs->end(), ValidReference);
    if (site.operation == "alloc" || site.operation == "dealloc")
      well_formed &= site.columns && *site.columns >= 32 &&
                     *site.columns <= 512 &&
                     ((*site.columns & (*site.columns - 1)) == 0) &&
                     site.allocation_key.has_value();
    if (site.operation == "alloc")
      well_formed &=
          site.allocation_key ==
          TmemAllocationId{site.warp, site.instruction_index, site.occurrence};
    if (!well_formed) {
      limit("tmem_group2_invalid_participation",
            "malformed group2 lifetime record at " + Describe(site));
      continue;
    }
    Coordinates local{}, origin{};
    for (unsigned i = 0; i < 3; ++i) {
      local[i] = site.cta_coordinates[i] % dimensions[i];
      origin[i] = site.cta_coordinates[i] - local[i];
    }
    const auto rank = std::uint64_t(local[0]) +
                      std::uint64_t(dimensions[0]) *
                          (local[1] + std::uint64_t(dimensions[1]) * local[2]);
    const auto pair_rank = rank & ~std::uint64_t{1};
    if (pair_rank + 1 >= cluster_size) {
      limit("tmem_group2_missing_peer",
            Describe(site) + " has no peer CTA in its cluster");
      continue;
    }
    valid.push_back(&site);
    const BucketKey key{origin, pair_rank, site.operation,
                        site.instruction_index, site.occurrence};
    auto [bucket, added] = bucket_ids.emplace(key, buckets.size());
    if (added)
      buckets.emplace_back();
    buckets[bucket->second].emplace_back(rank & 1, &site);
  }
  using AllocationKey = std::pair<Coordinates, TmemAllocationId>;
  std::map<AllocationKey, std::vector<const Group2LifetimeSite *>> allocations,
      frees;
  std::vector<AllocationKey> allocation_order, free_order;
  for (const auto *site : valid) {
    if (site->operation != "alloc" && site->operation != "dealloc")
      continue;
    auto key = AllocationKey{site->cta_coordinates, *site->allocation_key};
    auto &table = site->operation == "alloc" ? allocations : frees;
    auto &order = site->operation == "alloc" ? allocation_order : free_order;
    auto [found, added] = table.try_emplace(key);
    if (added)
      order.push_back(key);
    found->second.push_back(site);
  }
  const auto allocation_description = [](const AllocationKey &key) {
    return "(" + CoordinatesText(key.first) + ", " +
           AllocationText(key.second) + ")";
  };
  for (const auto &key : allocation_order) {
    if (allocations[key].size() != 1)
      limit("tmem_group2_duplicate_allocation",
            "group2 allocation identity " + allocation_description(key) +
                " occurs more than once");
    if (frees[key].empty())
      limit("tmem_group2_missing_deallocation",
            "group2 allocation " + allocation_description(key) +
                " has no reached deallocation");
    else if (frees[key].size() != 1)
      limit("tmem_group2_multiple_dealloc",
            "group2 allocation " + allocation_description(key) +
                " is deallocated by more than one warp/site");
  }
  for (const auto &key : free_order)
    if (allocations[key].size() != 1)
      limit("tmem_group2_unknown_allocation",
            "deallocation at " + Describe(*frees[key].front()) +
                " has no unique allocation");
  using OrderKey =
      std::tuple<RefKey, std::vector<RefKey>, SharedOptional<std::vector<std::uint32_t>>>;
  std::set<OrderKey> seen_orders;
  const SharedString order_kind("tmem_group2_access_before_dealloc");
  const SharedString order_message(
      "all accesses to both paired TMEM allocations must complete "
      "before either warp attempts deallocation");
  for (const auto &bucket : buckets) {
    if (bucket.size() != 2 || bucket[0].first == bucket[1].first) {
      const auto &first = *bucket.front().second;
      limit("tmem_group2_unmatched_site",
            "group2 " + first.operation + " at " + Describe(first) +
                " lacks exactly one matching peer warp");
      continue;
    }
    const auto &even = *bucket[bucket[0].first ? 1 : 0].second;
    const auto &odd = *bucket[bucket[0].first ? 0 : 1].second;
    std::set<ActorId> participants(even.participant_actors.begin(),
                                   even.participant_actors.end());
    if (std::any_of(odd.participant_actors.begin(),
                    odd.participant_actors.end(),
                    [&](auto actor) { return participants.contains(actor); })) {
      limit("tmem_group2_invalid_participation",
            "paired CTA records at " + Describe(even) + " reuse actor IDs");
      continue;
    }
    if (even.columns != odd.columns) {
      limit("tmem_group2_column_mismatch", "peer lifetime operations at " +
                                               Describe(even) +
                                               " disagree on column count");
      continue;
    }
    if ((even.operation == "alloc" || even.operation == "dealloc") &&
        (even.allocation_key->instruction_index !=
             odd.allocation_key->instruction_index ||
         even.allocation_key->occurrence != odd.allocation_key->occurrence)) {
      limit("tmem_group2_allocation_mismatch",
            "peer lifetime operations at " + Describe(even) +
                " refer to different allocation occurrences");
      continue;
    }
    if (even.operation == "dealloc") {
      const AllocationKey even_key{even.cta_coordinates, *even.allocation_key},
          odd_key{odd.cta_coordinates, *odd.allocation_key};
      const auto &left = allocations[even_key];
      const auto &right = allocations[odd_key];
      if (left.size() != 1 || right.size() != 1)
        continue;
      if (left.front()->columns != even.columns ||
          right.front()->columns != odd.columns) {
        limit("tmem_group2_column_mismatch",
              "deallocation at " + Describe(even) +
                  " disagrees with its allocation size");
        continue;
      }
      if (frees[even_key].size() != 1 || frees[odd_key].size() != 1)
        continue;
      auto references = left.front()->accesses;
      references.insert(references.end(), right.front()->accesses.begin(),
                        right.front()->accesses.end());
      references = UniqueReferences(references);
      std::map<InstId, std::set<std::uint32_t>> grouped;
      std::vector<InstId> group_order;
      std::vector<
          std::pair<MilestoneRef, SharedOptional<std::vector<std::uint32_t>>>>
          sources;
      for (const auto &ref : references)
        if (ref.milestone == Milestone::kEffect) {
          auto [it, added] = grouped.try_emplace(ref.instruction);
          if (added)
            group_order.push_back(ref.instruction);
          it->second.insert(ref.effect_index);
        } else
          sources.emplace_back(ref, std::nullopt);
      for (auto instruction : group_order) {
        const auto &indices = grouped[instruction];
        sources.emplace_back(
            MilestoneRef{instruction, Milestone::kComplete},
            std::vector<std::uint32_t>(indices.begin(), indices.end()));
      }
      auto targets=even.issue_refs;
      targets.insert(targets.end(),odd.issue_refs.begin(),odd.issue_refs.end());
      targets=UniqueReferences(targets);
      std::vector<RefKey> target_keys;
      for(const auto& ref:targets)target_keys.push_back(Key(ref));
      const SharedOptional<std::vector<MilestoneRef>> target_domain(std::move(targets));
      if(!target_domain->empty())for (const auto &[source, effects] : sources)
        if(seen_orders.emplace(Key(source),target_keys,effects).second)
          result.required_orders.push_back({order_kind,source,target_domain->front(),
                                            order_message,effects,target_domain});
    }
    result.pairs.push_back({even, odd});
  }
  return result;
}

Group2ProgressResult
CheckGroup2OptionalBlocking(const algorithm::SourceProgram &program,
                            const Group2LifetimeResult &lifetimes,
                            std::function<void()> poll) {
  using namespace algorithm;
  const auto failed = [&](std::size_t checked, std::string message) {
    return Group2ProgressResult{
        false, checked, {std::move(message)}, Premises()};
  };
  if (!lifetimes.ParticipationProven())
    return failed(0, "paired participation is not proved");
  std::unordered_map<std::uint32_t, std::size_t> positions;
  positions.reserve(program.instructions.size());
  for (std::size_t i = 0; i < program.instructions.size(); ++i)
    if (!positions.emplace(program.instructions.id(i).value(), i).second)
      throw std::invalid_argument("duplicate progress source instruction");
  std::unordered_map<std::uint32_t, const ActorInfo *> actors;
  actors.reserve(program.actors.size());
  for (const auto &actor : program.actors)
    actors.emplace(actor.id.value(), &actor);
  ProgressGraph graph;
  std::vector<std::array<std::size_t, 2>> milestones;
  milestones.reserve(program.instructions.size());
  std::unordered_map<std::uint32_t, std::size_t> prefix_gates;
  const auto instructionPosition = [&](InstId id) {
    auto found = positions.find(id.value());
    if (found == positions.end())
      throw std::invalid_argument(
          "progress reference names an unknown instruction");
    return found->second;
  };
  const auto instruction = [&](InstId id) {
    return program.instructions.view(instructionPosition(id));
  };
  const auto wait = [](const SourceOp &op) {
    return std::holds_alternative<MBarrierWaitState>(op) ||
           std::holds_alternative<MBarrierWaitParity>(op);
  };
  for (const auto &item : program.instructions) {
    const auto issue = graph.Node();
    const auto barrier = std::visit(
        [](const auto &op) { return requires { op.barrier; }; }, item.op);
    const auto done = barrier && !wait(item.op) ? issue : graph.Node();
    milestones.push_back({issue, done});
    if (issue != done)
      graph.Edge(issue, done);
    if (HasNotificationPrefix(item)) {
      const auto gate = graph.Node();
      prefix_gates.emplace(item.id.value(), gate);
      graph.Edge(issue, gate);
      graph.Edge(gate, done);
    }
  }
  const auto resolve = [&](const MilestoneRef &ref) {
    const auto found = positions.find(ref.instruction.value());
    if (found == positions.end())
      throw std::invalid_argument("progress reference names an unknown instruction");
    const auto position = found->second;
    if (ref.milestone == Milestone::kEffect ||
        ref.milestone == Milestone::kReadsComplete) {
      if (!program.instructions.holdsOperation<AsyncOp>(position))
        throw std::invalid_argument(
            "progress reference requires an async instruction");
      if (ref.milestone == Milestone::kEffect &&
          ref.effect_index >= program.instructions.effectCount(position))
        throw std::invalid_argument(
            "progress effect reference is out of range");
      if (ref.milestone == Milestone::kEffect && ref.effect_index == 0)
        if (const auto gate = prefix_gates.find(ref.instruction.value());
            gate != prefix_gates.end())
          return gate->second;
    }
    return milestones.at(position)[ref.milestone != Milestone::kIssue];
  };
  std::unordered_map<std::uint32_t, std::size_t> barriers, collectives, states;
  const auto hub = [&](auto &table, std::uint32_t id) {
    auto [found, added] = table.try_emplace(id, graph.nodes);
    if (added)
      graph.Node();
    return found->second;
  };
  std::size_t count = 0;
  for (const auto &item : program.instructions) {
    if ((++count & 4095) == 0 && poll)
      poll();
    // Plain locals: clang before 16 cannot capture structured bindings.
    const auto &milestone = milestones.at(positions.at(item.id.value()));
    const std::size_t issue = milestone[0], done = milestone[1];
    std::visit(
        [&](const auto &op) {
          using T = std::decay_t<decltype(op)>;
          if constexpr (std::is_same_v<T, AsyncOp>) {
            for (const auto &ref : op.completion_requires)
              graph.Edge(resolve(ref), done);
            const auto gate = prefix_gates.find(item.id.value());
            for (std::size_t i = 0; i < op.effects.size(); ++i) {
              const auto &effect = op.effects[i];
              const auto target = i == 0 && gate != prefix_gates.end()
                                      ? gate->second : done;
              for (const auto &ref : effect.predecessors)
                graph.Edge(resolve(ref), target);
              if (effect.completion)
                std::visit(
                    [&](const auto &action) {
                      graph.Edge(done, hub(barriers, action.barrier.value()));
                    },
                    *effect.completion);
            }
          } else if constexpr (std::is_same_v<T, Await>) {
            for (const auto &ref : op.requirements)
              graph.Edge(resolve(ref), done);
          } else if constexpr (std::is_same_v<T, MBarrierWaitState> ||
                               std::is_same_v<T, MBarrierWaitParity>)
            graph.Edge(hub(barriers, op.barrier.value()), done);
          else if constexpr (std::is_same_v<T, SyncThreads> ||
                             std::is_same_v<T, SyncWarp>) {
            const auto gate = hub(collectives, op.sync_id.value());
            // Progress graph: a data collective also blocks until every lane
            // has issued it, whether or not it orders memory.
            graph.Edge(issue, gate);
            graph.Edge(gate, done);
          } else if constexpr (requires { op.barrier; })
            graph.Edge(done, hub(barriers, op.barrier.value()));
          if constexpr (requires { op.result; })
            if (op.result)
              states[op.result->value()] = done;
        },
        item.op);
  }
  for (std::size_t position = 0; position < program.instructions.size(); ++position) {
    if (!program.instructions.holdsOperation<MBarrierWaitState>(position)) continue;
    const auto item = program.instructions[position];
    if (const auto *op = std::get_if<MBarrierWaitState>(&item.op)) {
      if (!states.contains(op->state.value()))
        return failed(0, "barrier state definition is missing");
      graph.Edge(states.at(op->state.value()),
                 milestones.at(positions.at(item.id.value()))[0]);
    }
  }
  for (const auto &dependency : program.dependencies)
    graph.Edge(resolve(dependency.from), resolve(dependency.to));
  for (const auto &dependency : program.edges) {
    const auto source_position = instructionPosition(dependency.from);
    const auto target_position = instructionPosition(dependency.to);
    const auto index =
        dependency.kind == SourceEdgeKind::kCompleteBeforeIssue ||
        !program.instructions.holdsOperation<AsyncOp>(source_position);
    graph.Edge(milestones.at(source_position)[index],
               milestones.at(target_position)[0]);
  }
  std::vector<std::tuple<std::size_t, std::size_t, std::size_t>> optional_edges;
  for (std::size_t pair_index = 0; pair_index < lifetimes.pairs.size();
       ++pair_index) {
    const auto &pair = lifetimes.pairs[pair_index];
    for (bool reverse : {false, true}) {
      const auto &current = reverse ? pair.odd : pair.even;
      const auto &peer = reverse ? pair.even : pair.odd;
      if (current.completion_refs.size() != 32)
        return failed(pair_index, "all 32 lifetime returns must be recorded");
      try {
        std::set<std::uint32_t> lanes, warps, ctas;
        if (current.participant_actors.size() != 32 ||
            current.issue_refs.size() != 32)
          throw std::invalid_argument(
              "lifetime participants disagree with source actors");
        for (std::size_t i = 0; i < 32; ++i) {
          auto found = actors.find(current.participant_actors[i].value());
          if (found == actors.end())
            throw std::invalid_argument(
                "lifetime participants disagree with source actors");
          const auto &actor = *found->second;
          lanes.insert(actor.lane);
          warps.insert(actor.warp.value());
          ctas.insert(actor.cta.value());
          if (instruction(current.issue_refs[i].instruction)->actor !=
                  actor.id ||
              current.completion_refs[i].milestone != Milestone::kComplete ||
              instruction(current.completion_refs[i].instruction)->actor !=
                  actor.id)
            throw std::invalid_argument(
                "lifetime participants disagree with source actors");
        }
        if (lanes.size() != 32 || *lanes.begin() != 0 ||
            *lanes.rbegin() != 31 ||
            warps != std::set<std::uint32_t>{current.warp} || ctas.size() != 1)
          throw std::invalid_argument(
              "lifetime participants disagree with source actors");
        const auto gate = graph.Node();
        for (const auto &ref : peer.issue_refs) {
          const auto source = resolve(ref);
          graph.Edge(source, gate);
          optional_edges.emplace_back(source, gate, pair_index);
        }
        auto targets = current.completion_refs;
        targets.insert(targets.end(), current.publication_refs.begin(),
                       current.publication_refs.end());
        for (const auto &ref : current.issue_refs)
          if (std::holds_alternative<AsyncOp>(instruction(ref.instruction)->op))
            targets.push_back({ref.instruction, Milestone::kComplete});
        for (const auto &ref : UniqueReferences(targets)) {
          const auto target = resolve(ref);
          graph.Edge(gate, target);
          optional_edges.emplace_back(gate, target, pair_index);
        }
      } catch (const std::exception &error) {
        return failed(pair_index, error.what());
      }
    }
  }
  const auto components = graph.Components(poll);
  std::set<std::size_t> cyclic;
  for (const auto &[source, target, index] : optional_edges)
    if (components[source] == components[target])
      cyclic.insert(index);
  Group2ProgressResult result{
      cyclic.empty(), lifetimes.pairs.size(), {}, Premises()};
  for (auto index : cyclic)
    result.blocked.push_back(
        "optional peer blocking may close a dependency cycle at paired site " +
        std::to_string(index));
  return result;
}
} // namespace sentryward::frontend::ptx
