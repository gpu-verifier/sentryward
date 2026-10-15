#pragma once
#include "sentryward/algorithm/checker/memory_race_checker.h"
namespace sentryward::algorithm {
std::optional<Result<MemoryRaceResult,MemoryAnalysisError>> TryFamilyMemoryCheck(
    const SourceProgram&,const PassCertificate&,MemoryModel&,const MemoryRaceOptions&,
    std::optional<std::span<const MemoryAccessFamily>> families = std::nullopt,
    StaticHbQueries* shared_hb = nullptr);
}
