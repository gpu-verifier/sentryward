#pragma once
#include "sentryward/scalar_fold/graph.h"
#include "sentryward/equivalence/bit_origins.h"
namespace sentryward::equivalence {
// Conservative fixed-header candidate predicates, shared by registry dispatch
// and each implementation. False means this rule cannot change the use;
// true still requires the complete bounded exact certificate.
bool contiguousReadCandidate(const scalar_fold::Graph&,const scalar_fold::Ref&);
bool packedExtractCandidate(const scalar_fold::Graph&,const scalar_fold::Ref&);
bool signedNibbleCandidate(const scalar_fold::Graph&,const scalar_fold::Ref&);
scalar_fold::Ref rewriteContiguousRead(scalar_fold::Graph&,scalar_fold::Ref);
scalar_fold::Ref rewritePackedExtract(scalar_fold::Graph&,scalar_fold::Ref);
scalar_fold::Ref rewriteSignedNibble(scalar_fold::Graph&,scalar_fold::Ref);
scalar_fold::Ref normalizePackedUse(scalar_fold::Graph&,scalar_fold::Ref);
// Budget-independent exact memory-field fast path: <=4 unsigned casts or
// constant logical shifts, width <=64, a byte-aligned field inside one raw
// cell, then a floating Bitcast. No arbitrary producer/bit walk is performed;
// every remaining conversion is preserved, including unsupported modifiers.
scalar_fold::Ref rewriteLocalFloatingMemoryRead(scalar_fold::Graph&,scalar_fold::Ref);
// Shared, scope-specific analysis; every rewrite remains exact.
scalar_fold::Ref rewriteContiguousRead(scalar_fold::Graph&,scalar_fold::Ref,
    BitOriginAnalysis&,const BitDomains&);
scalar_fold::Ref rewritePackedExtract(scalar_fold::Graph&,scalar_fold::Ref,
    BitOriginAnalysis&,const BitDomains&);
scalar_fold::Ref rewriteSignedNibble(scalar_fold::Graph&,scalar_fold::Ref,
    BitOriginAnalysis&,const BitDomains&);
scalar_fold::Ref normalizePackedUse(scalar_fold::Graph&,scalar_fold::Ref,
    BitOriginAnalysis&,const BitDomains&);
// Recover one complete field. Opaque value sources can yield ONLY the original
// typed value (whole width, unchanged bit order); partial reinterpretations
// remain Unknown. Memory fields additionally permit byte-aligned offsets.
std::optional<scalar_fold::Ref> recoverBitField(scalar_fold::Graph&,
    const SymbolicBitOrigins&,const std::string&,BitOriginAnalysis&);
}
