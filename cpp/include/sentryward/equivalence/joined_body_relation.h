#pragma once

#include "sentryward/equivalence/scoped_body_relation.h"
#include <array>

namespace sentryward::equivalence {
struct JoinedAxisUse {
  std::uint32_t level=0;
  // Reachable exact-u64 Index(level)+global_origin in the ACTUAL piece scope.
  // Rank-one InitialCell pieces may instead name their actual exact-u64 byte
  // offset: a private factory checks equal positive typed-cell state strides,
  // exact origin divisibility, original scopes/captures and full-domain bounds.
  // This derives geometry only; the same full body matcher remains mandatory.
  scalar_fold::Ref coordinate;
};
struct JoinedBodyPiece {
  ScopedBodyPath body;
  std::array<JoinedAxisUse,2> axes{};
  unsigned axis_count=0;
};
struct JoinedBodyResult {
  bool proved=false;
  std::string reason;
  std::size_t pieces=0,body_work=0,coverage_work=0,scope_identities=0;
};
// Proof-only equality between one ACTUAL full-domain body and a partition of
// ACTUAL original-domain piece bodies. The returned certificate is NOT Sum/
// wrapper/output equality: seeds, residuals, weights, masks and output dispatch
// remain separate obligations. No joined scope is submitted to graph resolution.
// This one-shot factory imports stored wrappers once; an Engine adapter reuses
// its private joined-family catalog/matcher, not the importer per piece/sample.
JoinedBodyResult proveJoinedBodyRelation(const scalar_fold::Graph&,const ScopedBodyPath&,
    std::span<const std::uint32_t> reference_axes,const scalar_fold::Graph&,
    std::span<const JoinedBodyPiece>,Options={});
} // namespace sentryward::equivalence
