#pragma once
#include "sentryward/frontend/ptx/semantic_program.h"

namespace sentryward::frontend::ptx {
// Consumes the same validated access records used by the completed race check.
// No source re-extraction, reachability query, or expanded event order is allowed.
CheckedMemoryActions SerializeCheckedMemory(const SemanticProgram&,
    const algorithm::SourceArtifact&, const algorithm::PassCertificate&,
    const algorithm::MemoryRaceResult&, const algorithm::MemoryAccessRecords&,
    bool lifetimes_checked);
}
