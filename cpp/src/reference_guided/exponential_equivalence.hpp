#pragma once
#include "linear_reduction_pattern.hpp"
namespace rg {
// Optional complete comparison attempt. Reuse the owning Checker vocabulary:
// call IDs are local spec indices, so a second Checker on the same Arena could
// alias unrelated reduction definitions. Move/restore only the active context.
struct ExponentialEquivalence {
  Checker &checker;
  Arena &a;
  J receipt;
  std::string last_failure;
  explicit ExponentialEquivalence(Checker &owner) : checker(owner), a(owner.a) {}
  bool prove(Id reference, Id optimized) {
    ++a.stats["ep_comparison_attempts"];
    if (checker.exponential_mode) { last_failure="nested_ep_attempt"; return false; }
    struct Restore {
      Checker &c;
      decltype(Checker::active) active, version;
      decltype(Checker::reference_binders) binders;
      decltype(Checker::active_byroot) roots;
      decltype(Checker::opt_cache) cache;
      decltype(Checker::match_cache) matches;
      explicit Restore(Checker &checker) : c(checker), active(std::move(c.active)),
          version(std::move(c.opt_version)), binders(std::move(c.reference_binders)),
          roots(std::move(c.active_byroot)), cache(std::move(c.opt_cache)), matches(std::move(c.match_cache)) {
        c.exponential_mode=true; c.scalar.reset();
        c.active.clear(); c.opt_version.clear(); c.reference_binders.clear();
        c.active_byroot.clear(); c.opt_cache.clear(); c.match_cache.clear();
      }
      ~Restore() {
        c.exponential_mode=false; c.scalar.reset();
        c.active=std::move(active); c.opt_version=std::move(version);
        c.reference_binders=std::move(binders); c.active_byroot=std::move(roots);
        c.opt_cache=std::move(cache); c.match_cache=std::move(matches);
      }
    } restore(checker);
    try {
      Id left = checker.normalize(reference, true);
      Id right = checker.normalize(optimized);
      bool equal = checker.scalar.equal(left, right);
      J reduction;
      if (!equal) {
        LinearReductionPattern linear(checker);
        if (auto result = linear.prove(left, right)) {
          equal = true; reduction = result->receipt;
        } else last_failure = linear.last_failure;
      }
      if (!equal) {
        if (!checker.scalar.last_failure.empty())
          last_failure += "; scalar: " + checker.scalar.last_failure;
        return false;
      }
      receipt = {{"method", "exponential_polynomial_with_reduction_coverage"},
                 {"scalar_work", checker.scalar.work},
                 {"body_and_outer", "shared_bounded_exponential_polynomial"},
                 {"reduction", reduction}};
      ++a.stats["ep_comparison_proofs"];
      return true;
    } catch (const Unsupported &e) {
      last_failure = e.what(); return false;
    }
  }
};
} // namespace rg
