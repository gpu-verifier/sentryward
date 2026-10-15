#pragma once

#include <vector>

#include "sentryward/algorithm/analysis/lifecycle_checker.h"
#include "sentryward/algorithm/analysis/phase_boundary.h"
#include "sentryward/algorithm/analysis/happens_before_graph.h"
#include "sentryward/algorithm/analysis/wait_checker.h"
#include "sentryward/algorithm/semantics/reference_trace.h"

namespace sentryward::algorithm {

struct PassCertificate final {
  ReferenceTrace reference;
  std::vector<PhaseCertificate> completed_phases;
  std::vector<OpenPhaseCertificate> open_phases;
  std::vector<WaitCertificate> waits;
  LifecycleCertificate lifecycle;
  HappensBeforeGraph happensBeforeGraph;
  DerivedEdgeJustifications justifications;
  bool has_frozen_reachability = false;
};

}  // namespace sentryward::algorithm
