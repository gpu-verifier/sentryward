#pragma once

#include <cstdint>
#include <vector>

#include "sentryward/algorithm/source_ir/program.h"

namespace sentryward::algorithm {

enum class EliminatedBarrierKind { kNamed, kMBarrier };

// One template column of eliminated arrivals: `count` operations of one site on one barrier.
struct EliminatedArrivals final {
  EliminatedBarrierKind kind;
  BarrierLifetimeId barrier;
  InstId first;
  SourceLocation location;
  std::uint64_t count;
};

// An eliminated mbarrier's initializer must still precede the issue of every eliminated use.
struct RetainedInitializationOrder final {
  InstId initializer;
  std::vector<MilestoneRef> uses;
};

struct UnobservedSynchronization final {
  std::vector<EliminatedArrivals> eliminated;
  std::vector<RetainedInitializationOrder> initialization_orders;
};

// Requires a validated program, whose barrier operands each name one exact lifetime.
// A barrier is unobserved when no operation of the launch can wait on it: no blocking
// participant (bar.sync, bar.red) of a named barrier, no wait on an mbarrier, and no
// state token from its arrivals (tokens are not followed to their waits). Its arrivals
// order nothing and complete no phase anyone sees: they become Await{}, and arrive-on
// actions leave their async effects, which keep accesses and dependencies. Instruction
// identities stay, so invalidation orders still cover the eliminated operations.
// Checks that do not depend on phases are retained by eliminating only when they hold
// in every order: one thread count per named barrier; for an mbarrier, no transaction
// count, no added arrivals and one arrival count dividing its expected count, so the
// pending count never underflows. Its initialization order is returned.
// O(templates) to decide and rewrite; O(eliminated mbarrier operations) for the orders.
[[nodiscard]] UnobservedSynchronization EliminateUnobservedSynchronization(
    SourceProgram& program);

}  // namespace sentryward::algorithm
