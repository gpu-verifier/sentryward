#include "sentryward/algorithm/checker/memory_race_checker.h"

#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <utility>
#include <unordered_map>
#include <numeric>
#include <map>

#include "sentryward/algorithm/analysis/unordered_access_pairs.h"
#include "sentryward/algorithm/checker/duplicate_tma_stores.h"
#include "sentryward/algorithm/checker/family_memory_checker.h"
#include "sentryward/algorithm/diagnostics/performance_profile.h"

namespace sentryward::algorithm {
namespace {

bool validResult(const OverlapResult& result) {
  switch (result.status) {
    case OverlapStatus::kDisjoint:
      return !result.evidence && result.reason.empty();
    case OverlapStatus::kUnknown:
      return !result.evidence && !result.reason.empty();
    case OverlapStatus::kOverlap:
      return result.evidence && result.reason.empty() &&
             (!result.evidence->description.empty() ||
              (result.evidence->constant_region &&
               result.evidence->constant_region->bytes != 0));
  }
  return false;
}

}  // namespace

static Result<MemoryRaceResult, MemoryAnalysisError> RunMemoryCheck(
    const SourceProgram& source, const PassCertificate& certificate,
    MemoryModel& memory, const MemoryRaceOptions& options, StaticHbQueries* shared_hb,
    const MemoryAccessRecords* supplied_accesses = nullptr,
    std::optional<std::span<const MemoryAccessFamily>> supplied_families = std::nullopt) {
  using Output = Result<MemoryRaceResult, MemoryAnalysisError>;
  if (shared_hb && &shared_hb->graph() != &certificate.happensBeforeGraph)
    return Output::Failure({"HB index must refer to the certificate graph", {}, {}});
  const auto scan_start = std::chrono::steady_clock::now();
  if (options.max_races == 0) {
    return Output::Failure({"max_races must be positive", {}, {}});
  }
  if (options.conflict_batch_size == 0) {
    return Output::Failure({"conflict_batch_size must be positive", {}, {}});
  }
  if (options.search_strategy != MemorySearchStrategy::kAuto &&
      options.search_strategy != MemorySearchStrategy::kHappensBeforeFirst) {
    return Output::Failure({"unknown memory search strategy", {}, {}});
  }
  if (options.continue_after_race &&
      (options.exhaustive || options.max_races != 1 || options.warn_duplicate_tma_stores ||
       options.search_strategy != MemorySearchStrategy::kAuto)) {
    return Output::Failure(
        {"continue-after-race requires strict conflict-order search with one witness "
         "and no exhaustive counting", {}, {}});
  }
  if (auto error = memory.validate()) {
    return Output::Failure(std::move(*error));
  }
  try {
    if(auto family=TryFamilyMemoryCheck(source,certificate,memory,options,supplied_families,shared_hb))return std::move(*family);
  } catch(const std::logic_error& error) {
    return Output::Failure({error.what(),{}, {}});
  }
  MemoryAccessRecords extracted_accesses;
  if (!supplied_accesses) {
    auto extracted = ProfileCall("memory.extract_accesses", [&] {
      return ExtractMemoryAccesses(certificate.happensBeforeGraph, source);
    });
    if (!extracted) return Output::Failure(std::move(extracted).error());
    extracted_accesses = std::move(extracted).value();
  }
  const auto& accesses = supplied_accesses ? *supplied_accesses : extracted_accesses;
  const bool has_atomics = std::any_of(accesses.begin(), accesses.end(),
      [](const auto& access) { return access.atomic_scope.has_value(); });
  if (options.continue_after_race && has_atomics)
    return Output::Failure({"continued conflict-order search does not support atomic exemptions", {}, {}});
  const DuplicateTmaStores duplicate_stores(source);
  std::unordered_map<std::uint32_t, CtaId> actor_ctas;
  for (const auto& actor : source.actors) actor_ctas.emplace(actor.id.value(), actor.cta);
  MemoryRaceResult result;
  std::vector<std::size_t> duplicate_parents;
  std::vector<bool> duplicate_members;
  if (options.collect_duplicate_write_groups) {
    duplicate_parents.resize(accesses.size());
    std::iota(duplicate_parents.begin(),duplicate_parents.end(),0);
    duplicate_members.resize(accesses.size());
  }
  const auto duplicate_root=[&](std::size_t id) {
    while (duplicate_parents[id]!=id) {
      duplicate_parents[id]=duplicate_parents[duplicate_parents[id]];
      id=duplicate_parents[id];
    }
    return id;
  };
  ProfileScope scan("memory.scan");
  try {
    bool order_only = false;
    bool hybrid = false;
    auto evaluate = [&](const MemoryAccess& first, const MemoryAccess& second) {
      ++result.candidates_examined;
      if (first.bytes == 0 || second.bytes == 0) {
        return false;
      }
      ++result.overlap_queries;
      OverlapResult overlap = memory.mayOverlap(first, second);
      if (!validResult(overlap)) {
        throw std::invalid_argument("memory model returned an invalid overlap result");
      }
      if (order_only && overlap.status != OverlapStatus::kOverlap)
        throw std::invalid_argument("memory order constraint must have known overlap");
      switch (overlap.status) {
        case OverlapStatus::kDisjoint:
          break;
        case OverlapStatus::kOverlap:
          if (first.atomic_scope && second.atomic_scope && first.bytes == second.bytes &&
              ((*first.atomic_scope != AtomicScope::kCta && *second.atomic_scope != AtomicScope::kCta) ||
               actor_ctas.at(first.actor.value()) == actor_ctas.at(second.actor.value()))) {
            const auto& region = overlap.evidence->constant_region;
            if (!region) {
              ++result.unresolved_count;
              result.unresolved.push_back({first, second, "atomic element identity/alignment is unresolved", "atomic_alignment_unresolved"});
              result.status = MemoryRaceStatus::kUnresolved;
              return true;
            }
            if (first.bytes && *first.bytes != 0 && region->bytes == *first.bytes &&
                region->offset % *first.bytes == 0)
              break;
          }
          if (options.warn_duplicate_tma_stores && duplicate_stores.contains(first, second)) {
            ++result.warning_count;
            if (options.collect_duplicate_write_groups) {
              const auto x=first.id.value(),y=second.id.value();
              duplicate_members[x]=duplicate_members[y]=true;
              const auto a=duplicate_root(x),b=duplicate_root(y);
              duplicate_parents[std::max(a,b)]=std::min(a,b);
            }
            if (result.duplicate_tma_store_warnings.size() < options.max_warning_samples) {
              result.duplicate_tma_store_warnings.push_back(
                  {first, second, std::move(*overlap.evidence)});
            }
            break;
          }
          ++result.race_count;
          if (options.on_race) options.on_race({first, second, *overlap.evidence});
          if (result.race_count == 1) {
            result.first_race_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - scan_start).count();
          }
          scan.pruned = result.race_count;  // Diagnostic counter: findings, not pruned pairs.
          if (result.races.size() < options.max_races) {
            result.races.push_back({first, second, std::move(*overlap.evidence)});
          }
          if (!options.exhaustive && !options.continue_after_race &&
              result.race_count >= options.max_races) {
            result.status = MemoryRaceStatus::kRaceFound;
            return true;
          }
          break;
        case OverlapStatus::kUnknown:
          ++result.unresolved_count;
          result.unresolved.push_back({first, second, std::move(overlap.reason), std::move(overlap.code)});
          result.status = MemoryRaceStatus::kUnresolved;
          return true;
      }
      return false;
    };
    const auto& graph = certificate.happensBeforeGraph;
    std::vector<std::size_t> ranks;
    auto prepare_ranks = [&] {
      if (!ranks.empty()) return;
      ProfileScope rank_profile("memory.topological_ranks");
      const auto order = graph.topologicalOrder();
      if (!order) throw std::invalid_argument("memory analysis requires an acyclic graph");
      ranks.resize(graph.eventCount());
      for (std::size_t i = 0; i < order->size(); ++i) ranks[(*order)[i].value()] = i;
    };
    auto stream = ProfileCall("memory.candidate_index", [&]() -> std::unique_ptr<MemoryConflictCandidates> {
      if (options.search_strategy != MemorySearchStrategy::kAuto) return nullptr;
      if (!options.exhaustive && options.max_races == 1 && !options.warn_duplicate_tma_stores && !has_atomics) {
        prepare_ranks();
        auto constraints = memory.conflictOrder(accesses, ranks);
        if (constraints) { order_only = true; return constraints; }
        if (options.continue_after_race)
          throw std::invalid_argument(
              "memory model cannot provide exact conflict-order constraints for continued scanning");
      }
      if (!options.exhaustive && options.max_races == 1 &&
          options.warn_duplicate_tma_stores && !has_atomics) {
        prepare_ranks();
        std::vector<AccessId> exempt_writes;
        for (const auto& access : accesses)
          if (duplicate_stores.mayContain(access)) exempt_writes.push_back(access.id);
        auto candidates = memory.conflictCandidatesWithExemptions(accesses, ranks, exempt_writes);
        if (candidates) { hybrid = true; return candidates; }
      }
      return memory.conflictCandidates(accesses);
    });
    if (stream) {
      result.search_strategy = order_only ? "conflict-order" :
                               hybrid ? "conflict-hybrid" : "conflict-first";
      if (options.on_search_strategy) options.on_search_strategy(result.search_strategy, false);
      prepare_ranks();
      StaticHbQueries local_hb(graph, options.reachability, options.reachability_factory);
      auto* index = &ProfileCall("memory.hb_index", [&]() -> HbReachability& {
        return (shared_hb ? *shared_hb : local_hb).get();
      });
      const auto initial_stats = index->stats();
      result.reachability_algorithm = index->name();
      std::vector<MemoryConflictPair> pairs;
      std::vector<HbQuery> queries;
      pairs.reserve(options.conflict_batch_size);
      queries.reserve(options.conflict_batch_size);
      bool exhausted = false;
      while (!exhausted) {
        pairs.clear(); queries.clear();
        ProfileScope candidates_profile("memory.generate_constraints");
        while (pairs.size() < options.conflict_batch_size) {
          auto pair = stream->next();
          if (!pair) { exhausted = true; break; }
          if (pair->first.value() >= accesses.size() || pair->second.value() >= accesses.size())
            throw std::invalid_argument("memory candidate references an unknown access");
          const auto& first = accesses[pair->first.value()];
          const auto& second = accesses[pair->second.value()];
          if (first.event == second.event || first.bytes == 0 || second.bytes == 0 ||
              (first.mode == AccessMode::kRead && second.mode == AccessMode::kRead))
            throw std::invalid_argument("memory candidate violates the pair contract");
          if (ranks[first.event.value()] > ranks[second.event.value()])
            std::swap(pair->first, pair->second);
          pairs.push_back(*pair);
          queries.push_back({accesses[pair->first.value()].event, accesses[pair->second.value()].event});
        }
        candidates_profile.Stop();
        result.address_candidates += pairs.size();
        if (pairs.empty()) break;
        const auto start = std::chrono::steady_clock::now();
        const auto ordered = ProfileCall("memory.hb_batch", [&] {
          return index->reachesBatch(queries);
        });
        result.reachability_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        result.reachability_stats = HbStatsSince(index->stats(), initial_stats);
        scan.nodes = result.address_candidates;  // Candidate pairs submitted to HB.
        scan.edges = result.reachability_stats.edge_visits;
        if (ordered.size() != pairs.size())
          throw std::invalid_argument("HB index returned the wrong answer count");
        for (std::size_t i = 0; i < pairs.size(); ++i) {
          if (ordered[i] > 1) throw std::invalid_argument("HB index returned a nonboolean answer");
          if (!ordered[i]) {
            if (order_only) ++result.violated_order_constraints;
            // The exact-order stream guarantees overlap for every obligation.
            // After saving the witness, continue all HB work but do not pretend
            // repeated constraints across slabs are distinct race pairs.
            if (options.continue_after_race && result.race_count != 0) continue;
            if (evaluate(accesses[pairs[i].first.value()], accesses[pairs[i].second.value()]))
              return Output::Success(std::move(result));
          }
        }
        ProfileSnapshot();
      }
    } else {
      result.search_strategy = "hb-first";
      result.candidate_generation_fallback = options.search_strategy == MemorySearchStrategy::kAuto;
      if (options.on_search_strategy)
        options.on_search_strategy(result.search_strategy, result.candidate_generation_fallback);
      result.reachability_algorithm = "ancestor-iterator";
      auto storage_filter = ProfileCall("memory.storage_index", [&] {
        return memory.storageFilter(accesses);
      });
      UnorderedAccessPairs candidates(certificate.happensBeforeGraph, accesses, {},
                                      std::move(storage_filter));
      while (auto pair = candidates.next()) {
        if (evaluate(accesses.at(pair->first.value()), accesses.at(pair->second.value())))
          return Output::Success(std::move(result));
        scan.nodes = result.candidates_examined;
        if ((result.candidates_examined & 65535U) == 0) ProfileSnapshot();
      }
    }
  } catch (const std::logic_error& error) {
    return Output::Failure({error.what(), {}, {}});
  }
  result.search_complete = true;
  if (options.collect_duplicate_write_groups) {
    std::map<std::size_t,std::vector<AccessId>> groups;
    for (std::size_t index=0;index<accesses.size();++index)
      if (duplicate_members[index]) groups[duplicate_root(index)].push_back(accesses[index].id);
    for (auto& [root,members]:groups) result.duplicate_write_groups.push_back(std::move(members));
    result.duplicate_write_groups_complete=true;
  }
  result.status = result.race_count != 0 ? MemoryRaceStatus::kRaceFound
                  : result.unresolved_count != 0 ? MemoryRaceStatus::kUnresolved
                                                 : MemoryRaceStatus::kRaceFree;
  return Output::Success(std::move(result));
}

Result<MemoryRaceResult, MemoryAnalysisError> MemoryRaceChecker::Run(
    const SourceProgram& source, const PassCertificate& certificate,
    MemoryModel& memory, const MemoryRaceOptions& options, StaticHbQueries* shared_hb) const {
  return RunMemoryCheck(source, certificate, memory, options, shared_hb);
}

Result<MemoryRaceResult, MemoryAnalysisError> MemoryRaceChecker::RunWithAccesses(
    const SourceProgram& source, const PassCertificate& certificate,
    const MemoryAccessRecords& accesses, std::span<const MemoryAccessFamily> families,
    MemoryModel& memory, const MemoryRaceOptions& options, StaticHbQueries* shared_hb) const {
  return RunMemoryCheck(source, certificate, memory, options, shared_hb, &accesses, families);
}

}  // namespace sentryward::algorithm
