#pragma once

namespace sentryward::algorithm {
struct SourceArtifact;

// Call after recovery/refinement has finished appending source records. No
// source instruction/effect pointers may survive this storage-only operation.
void CompactSourceStorage(SourceArtifact& source);
}  // namespace sentryward::algorithm
