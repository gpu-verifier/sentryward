#include "sentryward/algorithm/checker/memory_race_checker.h"

#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <utility>
#include <optional>
#include <span>
#include <vector>

#include "sentryward/algorithm/analysis/unordered_access_pairs.h"
#include "sentryward/algorithm/checker/cross_unit_races.h"
#include "sentryward/algorithm/checker/duplicate_tma_stores.h"
#include "sentryward/algorithm/checker/family_memory_checker.h"
#include "sentryward/algorithm/checker/same_value_writes.h"
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

// `complete_candidates`: the rescan after a same-value write pair in a reduced candidate stream
// (order constraints or exemptions), with every conflicting pair; the family check declined.
static Result<MemoryRaceResult, MemoryAnalysisError> RunQuotientMemoryCheck(
    const SourceProgram& source, const PassCertificate& certificate,
    MemoryModel& memory, const MemoryRaceOptions& options, StaticHbQueries* shared_hb,
    const MemoryAccessRecords* supplied_accesses = nullptr,
    std::optional<std::span<const MemoryAccessFamily>> supplied_families = std::nullopt,
    bool complete_candidates = false) {
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
  if (!complete_candidates) try {
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
  // Class mode: a duplicate pair lies in one CTA (its site key holds the CTA) and stands for that
  // pair in every unit of the CTA's class.
  std::optional<ActorWeights> weights;
  if (options.units) weights.emplace(*options.units, source.actors);
  const ActorCtas ctas(source.actors);
  MemoryRaceResult result;
  WriteObligationGroups duplicates, same_values;
  SameValueWrites same_value(source, &ctas, weights ? &*weights : nullptr, options.max_warning_samples);
  // The continued conflict-order search, a diagnostic mode that needs strict order
  // constraints, records no same-value obligations: such pairs are races there.
  const bool same_value_rule = !options.continue_after_race;
  bool rescan = false;
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
               ctas.at(first.actor) == ctas.at(second.actor))) {
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
            const std::size_t weight = weights ? (*weights)(first.actor) : 1;
            if (weight > SIZE_MAX - result.warning_count) throw std::overflow_error("warning count overflow");
            result.warning_count += weight;
            if (options.collect_duplicate_write_groups) duplicates.pair(first.id, second.id);
            if (result.duplicate_tma_store_warnings.size() < options.max_warning_samples) {
              result.duplicate_tma_store_warnings.push_back(
                  {first, second, std::move(*overlap.evidence)});
            }
            break;
          }
          if (same_value_rule && same_value.pair(first, second, *overlap.evidence)) {
            // A reduced stream certifies orders through write/write gaps; with this pair
            // unordered its other pairs on the storage may be missing. Rescan them all.
            if (order_only || hybrid) { rescan = true; return true; }
            same_values.pair(first.id, second.id);
            same_value.sample(first, second, *overlap.evidence, result);
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
      if (complete_candidates) return memory.conflictCandidates(accesses);
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
            if (evaluate(accesses[pairs[i].first.value()], accesses[pairs[i].second.value()])) {
              if (rescan)
                return RunQuotientMemoryCheck(source, certificate, memory, options, shared_hb,
                                              &accesses, supplied_families, true);
              return Output::Success(std::move(result));
            }
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
    result.duplicate_write_groups=duplicates.groups();
    result.duplicate_write_groups_complete=true;
  }
  result.same_value_write_groups=same_values.groups();
  result.same_value_write_partners=same_values.partners();
  result.status = result.race_count != 0 ? MemoryRaceStatus::kRaceFound
                  : result.unresolved_count != 0 ? MemoryRaceStatus::kUnresolved
                                                 : MemoryRaceStatus::kRaceFree;
  return Output::Success(std::move(result));
}

// Cross-CTA class mode: after the quotient search ended without a race, every global pair with an
// access in a member unit (cross_unit_races.h). A race found there is reported as the first race; an
// Unknown, a work cap, a race behind an unresolved quotient search (whose first finding on the
// per-unit path could be either), or a write pair that the same-value rule could accept (an
// obligation on the per-unit path, which only its value analysis decides) restarts the launch per
// unit.
static Result<MemoryRaceResult, MemoryAnalysisError> RunMemoryCheck(
    const SourceProgram& source, const PassCertificate& certificate,
    MemoryModel& memory, const MemoryRaceOptions& options, StaticHbQueries* shared_hb,
    const MemoryAccessRecords* supplied_accesses = nullptr,
    std::optional<std::span<const MemoryAccessFamily>> supplied_families = std::nullopt) {
  auto output = RunQuotientMemoryCheck(source, certificate, memory, options, shared_hb, supplied_accesses,
                                       supplied_families);
  if (!options.units || !output || output.value().status == MemoryRaceStatus::kRaceFound) return output;
  ProfileScope profile("memory.cross_unit");
  std::vector<MemoryAccessFamily> extracted;
  if (!supplied_families) {
    auto families = ExtractMemoryAccessFamilies(certificate.happensBeforeGraph, source);
    if (!families) return Result<MemoryRaceResult, MemoryAnalysisError>::Failure(std::move(families).error());
    extracted = std::move(families).value();
  }
  const auto families = supplied_families.value_or(std::span<const MemoryAccessFamily>(extracted));
  // Representative family parts, split at CTA boundaries (a family's actor is affine in its member
  // and the actor ids of one CTA are contiguous, so each CTA holds one member range).
  const ActorCtas ctas(source.actors);
  if (!ctas.contiguous()) throw std::logic_error("cross-unit pass: the actor ids of a CTA are not contiguous");
  std::vector<CrossUnitPart> parts;
  for (const auto& family : families) {
    if (!family.count) continue;
    const bool write = family.first.mode == AccessMode::kWrite;
    const auto cta_of = [&](std::size_t member) { return ctas(family.actor(member)); };
    std::vector<std::pair<std::size_t, std::size_t>> ranges; // [begin, end) member ranges, one CTA each
    ForEachCtaRange(family.first_member, family.first_member + family.count, cta_of,
                    [&](std::size_t begin, std::size_t end) { ranges.emplace_back(begin, end); });
    for (const auto& [first, stop] : ranges) {
      auto part = family;
      part.first = family.at(first);
      part.first_member = first;
      part.count = stop - first;
      const auto cta = cta_of(first);
      if (auto footprint = memory.crossUnitFootprint(part)) {
        parts.push_back({std::move(part), std::move(*footprint), write, cta});
        continue;
      }
      // One footprint per member (its own offset, stride 0), unmerged.
      for (std::size_t member = first; member < stop; ++member) {
        auto single = family;
        single.first = family.at(member);
        single.first_member = member;
        single.count = 1;
        auto footprint = memory.crossUnitFootprint(single);
        if (!footprint) {
          FamilyFootprint none; // not placeable
          parts.push_back({std::move(single), std::move(none), write, cta});
          continue;
        }
        footprint->offset += footprint->step * address_solver::Int(member);
        footprint->step = 0;
        parts.push_back({std::move(single), std::move(*footprint), write, cta});
      }
    }
  }
  const auto outcome = CheckCrossUnit(parts, *options.units, memory);
  using Status = CrossUnitOutcome::Status;
  if (outcome.status == Status::kNone) return output;
  if (outcome.status == Status::kUnknown) throw UnitClassRestart{"cross_unit_unknown:" + outcome.reason};
  if (outcome.status == Status::kCap) throw UnitClassRestart{"cross_unit_cap:" + outcome.reason};
  auto& result = output.value();
  if (result.status == MemoryRaceStatus::kUnresolved) throw UnitClassRestart{"cross_unit_tie_break"};
  const auto& a = parts.at(outcome.first_part);
  const auto& b = parts.at(outcome.second_part);
  auto first = a.family.at(outcome.first_member), second = b.family.at(outcome.second_member);
  // A write pair that the same-value rule could accept is an obligation of the per-unit search. The
  // representatives may be one write; its instances in two units are distinct writes, as
  // SameValueWrites::pair requires of the pair, so only the event test is left out.
  if (!options.continue_after_race && a.write && b.write &&
      SameValueWrites(source).sameBytes(first, second, outcome.region_bytes))
    throw UnitClassRestart{"cross_unit_same_value"};
  const auto& q = *options.units;
  OverlapEvidence evidence;
  // Units are clusters, or single CTAs when the launch has no clusters (kappa is then the CTA id).
  const auto coordinates = [](const UnitQuotient::Coordinates& c) {
    return "(" + std::to_string(c[0]) + "," + std::to_string(c[1]) + "," + std::to_string(c[2]) + ")";
  };
  const bool single_cta_units = q.unit_shape == UnitQuotient::Coordinates{1, 1, 1};
  evidence.description = std::string("global byte ranges of ") + (single_cta_units ? "CTAs " : "clusters ") +
                         coordinates(q.units.at(outcome.first_unit).kappa) + " and " +
                         coordinates(q.units.at(outcome.second_unit).kappa) + " intersect";
  evidence.constant_region = ByteOverlap{"global", outcome.allocation, outcome.region_offset, outcome.region_bytes};
  const auto rank_a = q.RankOf(a.cta), rank_b = q.RankOf(b.cta);
  result.cross_unit_race = CrossUnitEvidence{q.units.at(outcome.first_unit).kappa,
                                          q.CtaCoordinates(q.units.at(outcome.first_unit).ctas.at(rank_a)),
                                          q.units.at(outcome.second_unit).kappa,
                                          q.CtaCoordinates(q.units.at(outcome.second_unit).ctas.at(rank_b)),
                                          outcome.allocation, outcome.first_delta, outcome.second_delta};
  RaceFinding race{std::move(first), std::move(second), std::move(evidence)};
  result.status = MemoryRaceStatus::kRaceFound;
  result.race_count = 1;
  result.races.assign(1, race);
  // As the per-unit search, which stops at its first race (class mode never
  // runs an exhaustive or continued search).
  result.search_complete = false;
  if (options.on_race) options.on_race(result.races.front());
  return output;
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
