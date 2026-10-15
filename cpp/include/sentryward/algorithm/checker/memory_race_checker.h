#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "sentryward/algorithm/checker/certificate.h"
#include "sentryward/algorithm/analysis/hb_reachability.h"
#include "sentryward/algorithm/memory/model.h"
#include "sentryward/algorithm/memory/family_access.h"

namespace sentryward::algorithm {

enum class MemoryRaceStatus { kRaceFound, kRaceFree, kUnresolved };

struct RaceFinding final {
  MemoryAccess first;
  MemoryAccess second;
  OverlapEvidence overlap;
};

struct UnresolvedMemoryPair final {
  MemoryAccess first;
  MemoryAccess second;
  std::string reason;
  std::string_view code = "address_overlap_unresolved";
};

enum class MemorySearchStrategy { kAuto, kHappensBeforeFirst };
inline constexpr const char* kMemoryCandidateFallback =
    "The memory model could not generate a complete set of potentially conflicting "
    "memory-access pairs. Analysis instead enumerates HB-unordered access pairs "
    "and checks address overlap for each pair.";

struct MemoryRaceOptions final {
  // Maximum retained findings; also the stopping threshold unless exhaustive
  // or continue_after_race is enabled.
  std::size_t max_races = 1;
  // Compatibility field; the first unsupported pair is always retained.
  std::size_t max_unresolved_samples = 16;
  std::size_t max_warning_samples = 16;
  bool warn_duplicate_tma_stores = false;
  bool collect_duplicate_write_groups = false;
  MemorySearchStrategy search_strategy = MemorySearchStrategy::kAuto;
  HbReachabilityOptions reachability = {};
  std::size_t conflict_batch_size = 65536;
  HbReachabilityFactory reachability_factory = {};
  bool exhaustive = false;
  // Observe each established race before bounded sample retention.
  std::function<void(const RaceFinding&)> on_race = {};
  std::function<void(const std::string&, bool)> on_search_strategy = {};
  // Diagnostic: consume the optimized order-constraint stream after the first
  // finding. Keep one witness; do not enumerate/count all conflicting pairs.
  // Requires strict conflict-order support; incompatible modes fail explicitly.
  bool continue_after_race = false;
};

struct MemoryRaceResult final {
  struct FamilyStats {
    std::size_t logical_accesses=0,access_families=0,candidate_relations=0;
    std::size_t affine_member_rows=0,hb_member_enumerations=0,hb_frontier_pieces=0;
  };
  std::optional<FamilyStats> family_stats;
  MemoryRaceStatus status = MemoryRaceStatus::kRaceFree;
  std::vector<RaceFinding> races;
  std::vector<RaceFinding> duplicate_tma_store_warnings;
  std::size_t warning_count = 0;
  // Complete equivalence obligations for accepted write/write pairs, not samples.
  std::vector<std::vector<AccessId>> duplicate_write_groups;
  bool duplicate_write_groups_complete = false;
  std::vector<UnresolvedMemoryPair> unresolved;
  std::size_t unresolved_count = 0;
  std::size_t candidates_examined = 0;
  std::size_t overlap_queries = 0;
  bool search_complete = false;
  std::string search_strategy;
  bool candidate_generation_fallback = false;
  std::string reachability_algorithm;
  std::size_t address_candidates = 0;
  HbReachabilityStats reachability_stats;
  double reachability_seconds = 0;
  // In continue_after_race mode this counts retained witnesses, not all pairs.
  std::size_t race_count = 0;
  std::optional<double> first_race_seconds;
  // May repeat the same access pair across spatial slabs; not a race-pair count.
  std::size_t violated_order_constraints = 0;
};

class MemoryRaceChecker final {
 public:
  // The certificate and source must describe the same successful check.
  // A shared service supplies its own index options/factory and is borrowed.
  [[nodiscard]] Result<MemoryRaceResult, MemoryAnalysisError> Run(
      const SourceProgram& source, const PassCertificate& certificate,
      MemoryModel& memory, const MemoryRaceOptions& options = {},
      StaticHbQueries* shared_hb = nullptr) const;

  // Access records are compressed independently of synchronization payloads.
  // Caller supplies two exact views of the same complete access set. Event IDs
  // refer to this certificate; source supplies warning provenance only.
  [[nodiscard]] Result<MemoryRaceResult, MemoryAnalysisError> RunWithAccesses(
      const SourceProgram& source, const PassCertificate& certificate,
      const MemoryAccessRecords& accesses,
      std::span<const MemoryAccessFamily> families,
      MemoryModel& memory, const MemoryRaceOptions& options = {},
      StaticHbQueries* shared_hb = nullptr) const;
};

}  // namespace sentryward::algorithm
