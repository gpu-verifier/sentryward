#include "sentryward/algorithm/memory/address_solver.h"

namespace sentryward::algorithm::address_solver {

// Built without an SMT backend: every solver-dependent pair stays Unknown,
// recovering the universal-relation fallback (dynamic.pdf Section 5).
std::unique_ptr<Solver> MakeSolver() { return nullptr; }
const char* BackendName() noexcept { return "unavailable"; }

}  // namespace sentryward::algorithm::address_solver
