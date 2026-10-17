# Outcome-specific decisions

Use the applicable row; these are evidence rules, not a fixed report template.

Current saved results use `result`, `checks`, `diagnostics` and `scope`. Read the exact top-level `result` first; keep synchronization, memory and equivalence check outcomes beside it. A present but null or unknown `result` does not fall back to an older `status`. Missing checks are unknown, not passed. `verified` describes the saved operation and checks: a verified value-analysis sub-operation alone is not whole-kernel verification.

The current result vocabulary is `verified`, `race_free`, `not_wellsynced`, `potential_race`, `invalid_lifecycle`, `equivalence_not_proven`, `unsupported`, `invalid_input`, `missing_input`, `timeout`, `memory_budget_exceeded`, `analysis_budget_exceeded`, `allocation_failed` and `tool_failure`. The renderer also accepts the historical statuses in the table below. Current `not_wellsynced`, `potential_race` and `invalid_lifecycle` use the corresponding historical synchronization, access-pair and lifecycle views without changing their saved result labels.

Integrated `equivalence` and `value_analysis` have their own `result`, `checks`, `diagnostics` and `scope`. Read them independently. Passed memory checks can coexist with unsupported numerical analysis; a nested success cannot override a failed top-level result. Preserve disagreements explicitly. Diagnostic explanations now live in `diagnostics[].message` and evidence in `diagnostics[].details`; do not depend on a legacy top-level `message` or `reason`.

Recovery is recorded through `analysis_target: recovered_program` and `recovery_policy`, with original violations retained separately. Keep that qualification visible. `recovered_*` is not a supported public result category and must not be stripped to manufacture a recognized success.

| Evidence | Lead with | Preserve visibly | Avoid |
| --- | --- | --- | --- |
| `verified` | Verification result for the saved operation | Exact checks, scope and material assumptions | Treating a sub-operation or missing checks as full verification |
| `equivalence_not_proven` or numerical `unsupported` | Numerical comparison and its diagnostic | Passed/failed synchronization and memory checks separately | Reclassifying a numerical blocker as a memory failure |
| Completed `race_free`, no findings | No races found in this modeled run | Configuration and material modeling conditions; separate completion/exhaustiveness | Hardware safety, numerical correctness, a fabricated issue or mandatory follow-up |
| `race_free` with warnings | Recorded result plus warning count | The rule making the reported accesses warnings; unproved facts affecting safety | Reclassifying warnings as confirmed races, or hiding them behind a clean badge |
| `race_candidate` | Candidate and affected operations | Known overlap, unresolved cause when applicable, search completeness | Claiming a missing wait, runtime failure or a validated fix from a pair alone |
| `synchronization_candidate` | The rejected requirement and involved events | Saved rejection, source mapping, model witness, retained/total sequence lengths | Calling a model witness a GPU trace or filling a truncated sequence |
| `unsafe_lifecycle` | The required lifecycle order that was not established | Operation pair, failed-order kind, retained/total counts | Treating instruction IDs as source lines, or inferring the missing edge's cause from its absence |
| Frontend unsupported | Analysis stopped; no race verdict | Stop location/reason and what was not analyzed | Treating absent race fields or empty warnings as zero races |
| Failure before meaningful analysis | Failure and available diagnostic | Missing stage/location/input and usable evidence | Invented issue snippets, launch values or a speculative mechanism |
| Incomplete search or truncated findings | Saved result and covered portion | Total vs retained counts; incomplete enumeration | Treating a full retained list as a complete search, or `exhaustive=false` alone as incomplete search |
| Multiple mechanisms | Result followed by a short finding index | Each distinct retained group and its own supporting sites; partial coverage | One enormous flowchart, repeating shared conditions per finding, omitting lesser findings |
| Mismatched input and result | Evidence mismatch | Which identity differs and which claims remain attributable | A combined conclusion across incompatible runs |

For a frontend stop, say analysis stopped in the frontend and memory analysis was not completed; do not suggest the memory-analysis stage itself failed.

Keep references tied to claims even after an early stop. Stop/completion fields, absent verdicts and rule applicability matter; an inventory of zero-valued compression, pool or bookkeeping counters usually does not. Preserve a counter when it establishes a necessary scope fact, name that fact, and do not transcribe the surrounding ledger merely because it was supplied. Zero execution counters still do not resolve a rule's unknown frontend applicability.

For a clean run without findings, use one short source-supported scope explanation. Do not walk through the load/MMA/epilogue pipeline unless a saved diagnostic makes it relevant. No problem was reported, so an inventory of untraced guarantees creates work rather than explaining the result. Include a diagram only when it clarifies coverage; describe it as static structure unless execution is evidenced. For a failure without code, say code is unavailable. Visualize a supported stage boundary only if it explains more than the diagnostic itself; otherwise show the exact diagnostic concisely. Never manufacture code or structure to satisfy a formatting convention.

For `issue_not_ordered_before_next_phase`, preserve the direction shown by the base report: **wait issue → next-phase completion effect** is required and not established. `required_events` does not mean predecessors of the wait. This interpretation follows `cpp/src/analysis/wait_checker.cpp` (`FindDescendant(issue_id, next_info->effects)`, including the initial-parity branch); it describes the check, not a hardware schedule. For another rejection code, do not reuse this direction without establishing its meaning.

Analysis stops keep their saved status: `frontend_unsupported`, `native_frontend_error`, `memory_model_unsupported`, `invalid_source`, `unresolved_translation`, `native_resource_limit`, and `native_timeout` are distinct. An unsupported memory model is not a frontend failure; a resource limit or timeout is not a clean result. A stopped run may still retain useful diagnostics without having a completed verdict.

For multiple mechanisms, give a compact index such as “Output copy — warning; shared tile reuse — candidate”, followed by locally named sections. Preserve source record identities in the evidence disclosure. Keep distinct mechanisms distinct even if they share an instruction or error code; group only with evidence. Reuse shared run conditions once.

The base report groups diagnostics by saved stage and code for navigation. These are categories, not proven root causes: retain different source sites, messages and counts within each group. A sampled `failed_required_orders` list may cover only one category while `diagnostics[].details` records another; read both. A lifecycle result without a saved failed-order list may describe protocol diagnostics during event construction, so do not manufacture a completion/deallocation pair. For each category, explain what source analysis establishes or state the specific unresolved relation; do not silently omit it from a single-focus story.

`memory_analysis_complete`, `search_complete` and `exhaustive` describe separate properties. An input option and a result property may use similar names; identify their source. Do not infer an explanation for a false flag from its spelling. Assumption-enabled and assumption-triggered are also separate: unknown applicability is not false.

A duplicate-store classification rule can match footprints and site visits without asserting equal copied data. Do not say the run assumed value equality unless the saved rule explicitly says so; equality may instead be a separate next-inspection gap.
