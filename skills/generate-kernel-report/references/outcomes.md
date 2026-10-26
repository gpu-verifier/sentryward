# Saved outcomes

Read the exact top-level `result`, with `checks`, `diagnostics` and `scope`. Older artifacts use `status`. A present null/unknown `result` does not fall back to `status`; missing checks are unknown. The renderer preserves unfamiliar results instead of treating them as success.

Public outputs keep witnesses under `evidence` and analysis details under `internal`; older outputs keep every field at the top level. The scripts read both layouts and cite fields at their saved location. Public outputs have two checks: `synchronization`, which includes the memory analysis, and `equivalence`. Older outputs also have a separate `memory` check.

In older outputs, integrated `equivalence` and `value_analysis` have their own outcomes. Public outputs keep these sections under `internal` without an outcome; `checks.equivalence` is the equivalence outcome. A numerical stop must not hide passed memory checks or memory warnings; nested success does not override the top-level result. `verified` applies to the saved operation and scope, which may be a sub-operation.

| Result | Explain | Preserve |
| --- | --- | --- |
| `verified`, `race_free` | Saved verification or race result | Checks, scope, material assumptions and warnings |
| `potential_race` / older `race_candidate` | A retained potentially conflicting pair | Affected accesses, overlap and unresolved bindings; no inferred cause or fix |
| `not_wellsynced` / older `synchronization_candidate` | The rejected synchronization requirement | Its direction, involved events and witness truncation |
| `invalid_lifecycle` / older `unsafe_lifecycle` | The failed lifecycle or protocol requirement | Each diagnostic category and its own supporting relations |
| `equivalence_not_proven` | The unestablished numerical comparison | Independent synchronization/memory outcomes |
| `unsupported` | The unavailable analysis and direct diagnostic | Completed checks, stopped stage and useful retained evidence |
| `invalid_input`, `missing_input` | Rejected or unavailable input | Exact diagnostic and available identity information |
| `timeout`, `memory_budget_exceeded`, `analysis_budget_exceeded`, `allocation_failed`, `tool_failure` | The specific stop/failure | Covered portion and incomplete outcomes; no clean verdict implied |

Historical stop categories remain distinct: `frontend_unsupported`, `native_frontend_error`, `memory_model_unsupported`, `invalid_source`, `unresolved_translation`, `native_resource_limit`, `native_timeout`. A frontend stop is not a failure of a completed memory analysis. Empty records do not imply no races.

Recovery is recorded by `analysis_target: recovered_program` and `recovery_policy`, under `internal` in public outputs, with original violations retained. Keep that qualification visible. Do not strip an unfamiliar `recovered_*` label into a recognized success.

## Coverage and finding selection

`memory_analysis_complete`, `search_complete` and `memory_analysis.exhaustive` are separate properties; public outputs keep them under `internal`. Total counts, retained counts and list truncation are also separate. State their saved meaning rather than inferring it from a similarly named input flag.

Diagnostic stage/code groups aid navigation; they do not prove common causes. Read both `failed_required_orders` and `diagnostics[].details`: a sampled order list may cover only one category. A lifecycle diagnostic without a saved order list does not justify inventing a completion/deallocation pair.

For `issue_not_ordered_before_next_phase`, the required edge is **wait issue → next-phase completion effect**. `required_events` are not predecessors of the wait. This direction follows the corresponding `FindDescendant(issue_id, next_info->effects)` check in `cpp/src/analysis/wait_checker.cpp`; another rejection needs its own interpretation.

For clean results, keep the visible account brief and scoped. For warnings, explain the classification rule without upgrading them to bugs or inventing value-equality assumptions. An identity mismatch prevents combining incompatible artifacts. See [evidence.md](evidence.md) for assumptions and source attribution.
