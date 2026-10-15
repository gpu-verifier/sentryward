#pragma once

#include "sentryward/equivalence/egraph.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace sentryward::equivalence {
// A bounded actual source-graph path. Every Fold BODY edge extends the scope;
// its seed never does. The whole wrapper root is validated before extraction.
struct ScopedBodyPath {
  scalar_fold::Ref wrapper;
  std::vector<unsigned> edges;
};
struct ScopedAxisBinding {
  std::uint32_t reference_level=0;
  std::optional<std::uint32_t> actual_level;
  std::uint64_t offset=0;
  // false: reference i=actual j+offset. true: actual j=reference i+offset.
  // A point (actual_level=null) is reference-only, with extent exactly one.
  bool actual_shift=false;
  // An actual pure exact-u64 coordinate witness, reachable from the wrapper.
  // The witness belongs to reference graph for actual_shift, otherwise actual.
  scalar_fold::Ref coordinate;
  // Optional point translation request. For a reference-only point, this is
  // the reachable REFERENCE tile coordinate Index(reference_level)+origin.
  // The factory validates its ORIGINAL domain/image and derives coordinate-
  // origin from the reachable ACTUAL exact-u64 global point. No graph constant
  // is synthesized. offset must agree with that checked derived local point.
  std::optional<scalar_fold::Ref> image_coordinate={};
};
struct ScopedBodyResult {
  bool proved=false;
  std::string reason;
  std::size_t body_work=0,scope_identities=0,coordinate_checks=0,affine_checks=0;
};
// Proof-only body equality under the actual registered finite lexical domains.
// This does NOT prove whole-wrapper/output equality or authorize numerical
// abstraction. Enclosing-wrapper seeds are outside this body goal and must be
// accounted for by any output certificate; nested Fold seeds are compared in
// their parent scope. Ideal policy must already be explicit; unsupported
// exact/bit-sensitive/nonfinite roots remain Unknown. No logical Fold elements
// are enumerated and no equality-saturation run is performed.
// This one-shot public factory imports stored wrapper DAGs once. An Engine
// cohort reuses the same internal paired matcher/catalog, not this importer
// once per recurrence update.
ScopedBodyResult proveScopedBodyRelation(const scalar_fold::Graph&,const ScopedBodyPath&,
    const scalar_fold::Graph&,const ScopedBodyPath&,std::span<const ScopedAxisBinding>,Options={});
} // namespace sentryward::equivalence
