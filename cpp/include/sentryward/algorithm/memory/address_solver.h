#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace sentryward::algorithm::address_solver {

// Checked 128-bit integers hold every u64/s64 offset and small coefficient
// products; producers treat overflow as an unsupported (non-scalar) offset.
__extension__ typedef __int128 Int;

[[nodiscard]] std::string ToString(Int value);

// Affine form over query-local integer variables: constant + sum(c_k * v_k).
struct Linear final {
  Int constant = 0;
  std::vector<std::pair<std::size_t, Int>> terms;
};

// Ground quantifier-free linear integer query (dynamic.pdf Section 4.3).
// Gather applications are already instantiated as variables, so functional
// consistency holds by interning and injectivity arrives as disequalities.
struct Query final {
  // Names are used only in diagnostic witnesses.
  std::vector<std::string> variables;
  struct Bound final {
    Linear term;
    std::optional<Int> lower;  // inclusive
    std::optional<Int> upper;  // inclusive
  };
  std::vector<Bound> bounds;
  std::vector<std::pair<Linear, Int>> multiples;  // term == 0 (mod positive m)
  // Each clause is a disjunction of variable disequalities.
  std::vector<std::vector<std::pair<std::size_t, std::size_t>>> distinct;
  // Overlap goal: goal_lower < goal < goal_upper (both exclusive).
  Linear goal;
  Int goal_lower = 0;
  Int goal_upper = 0;
};

enum class Verdict { kUnsat, kSat, kUnknown };

struct Answer final {
  // Satisfiability of the environment alone, then of environment AND goal.
  // The goal is checked only when the environment is kSat.
  Verdict environment = Verdict::kUnknown;
  Verdict goal = Verdict::kUnknown;
  // Solver reason for kUnknown, or a candidate valuation for kSat.
  std::string detail;
};

class Solver {
 public:
  virtual ~Solver() = default;
  // Both checks share the budget. Implementations never throw.
  [[nodiscard]] virtual Answer solve(const Query& query, unsigned timeout_ms) = 0;
  // Environment-only satisfiability, used to reject inconsistent invariants.
  [[nodiscard]] virtual Verdict consistent(const Query& query, unsigned timeout_ms) = 0;
};

// Null when the build has no SMT backend; callers then return Unknown.
[[nodiscard]] std::unique_ptr<Solver> MakeSolver();
[[nodiscard]] const char* BackendName() noexcept;

}  // namespace sentryward::algorithm::address_solver
