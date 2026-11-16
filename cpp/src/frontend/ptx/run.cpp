#include "run_pipeline.hpp"

// The attempts: a class-mode attempt that may restart with table atoms (TableAtomRestart), with
// founder planning (FounderPlanRestart) and then in unit order (FounderOrderRestart), then one that
// restarts per unit, which receives the call's inputs, its start time and its verdict.
int main(int argc, char **argv) {
  CallInputs inputs;
  std::optional<PerUnitRestart> unit_restart;
  std::optional<TableAtomRestart> atom_restart;
  std::optional<FounderOrderRestart> order_restart;
  std::optional<FounderPlanRestart> plan_restart;
  for (;;) {
    try {
      return RunAttempt(argc, argv, inputs, unit_restart, atom_restart, order_restart, plan_restart);
    } catch (FounderPlanRestart &restart) {
      plan_restart = restart;
#if defined(__GLIBC__)
      malloc_trim(0);
#endif
    } catch (TableAtomRestart &restart) {
      atom_restart = restart;
#if defined(__GLIBC__)
      malloc_trim(0);
#endif
    } catch (FounderOrderRestart &restart) {
      order_restart = restart;
#if defined(__GLIBC__)
      malloc_trim(0);
#endif
    } catch (PerUnitRestart &restart) {
      unit_restart = std::move(restart);
#if defined(__GLIBC__)
      // The class-mode attempt's pages, freed by its unwinding, go back before
      // the per-unit attempt.
      malloc_trim(0);
#endif
    }
  }
}
