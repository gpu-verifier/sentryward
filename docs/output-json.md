# Verification output (`output.json`)

The CLI emits one JSON object and Python verify returns the same structure.
result is one of 14 categories; checks and diagnostics explain the conclusion.

## JSON structure

The CLI writes one raw JSON object to stdout; Python returns the same structure.
Save it as `output.json` if a file is needed. The following is an abbreviated
successful result, showing the fields used to read the conclusion:

```json
{
  "result": "verified",
  "checks": {
    "synchronization": "passed",
    "equivalence": "passed"
  },
  "diagnostics": [],
  "warnings": [],
  "warning_count": 0,
  "warning_samples_truncated": false,
  "evidence": {"races": [], "race_count": 0, "race_samples_truncated": false},
  "internal": {}
}
```

The verifier has two public checks. `synchronization` covers synchronization,
memory race freedom and resource lifecycles; `equivalence` compares the kernel
with its reference. `result` names the cause of the conclusion, and `checks`
shows how far each check got.

`diagnostics` is an array. Each diagnostic's `code` identifies a concrete cause;
`stage` is the public stage it belongs to: `input`, `synchronization`,
`equivalence` or `pipeline`. Building the value view belongs to the
`equivalence` stage, although it can run before the synchronization check
finishes: a timeout there reports stage `equivalence`, with
`details.internal_stage` `value_view_construction`, while
`checks.synchronization` is `not_run`. A producer may include a readable
`message`; `details` retains operands, long expressions, source sites, resource
limits and other structured evidence. `details.internal_stage` names the
internal processing step when it differs from the public stage. Do not read only
the first diagnostic. Diagnostics can be informational, so a nonempty array does
not itself mean failure.

`warnings` contains structured warning samples, not another result category.
`warning_count` counts recorded warnings and can exceed the sample array length;
`warning_samples_truncated` identifies omitted samples. Completed runs omit
`warnings_complete`; some runs that stop early report it.
`warnings_complete: false` means warning collection did not complete, so zero
samples is not evidence of no warnings; `true` means it completed before the
run stopped. Early failures may omit count/sample fields; absence is not zero.

`evidence` holds witnesses and their counts. Once the memory analysis has run,
it contains `races`, `race_count` and `race_samples_truncated`, so it also
appears on passing runs. A failed synchronization check adds the witnesses
behind it, such as required orders that were not established
(`failed_required_orders`, with its count and truncation fields) or a
synchronization counterexample (`counterexample`, `rejection` and lists of the
events involved). An access pair that the memory analysis could not decide
appears as `unsupported_memory_pair`. See [race witnesses](#race-witnesses).

`internal` holds analysis details: the value analysis, scalar fold, frontend
and address-model statistics, equivalence receipts, counters and timing
breakdowns. It is not a stable interface. Its sections keep their own
diagnostics, and some keep intermediate results, such as
`internal.verification_result` and `internal.equivalence.numerical_result`.
Read the conclusion only from `result`, and the progress of each check from
`checks`. See [configuration formats](../README.md#documentation) for the input
files.

## Categories

| result | Meaning |
| --- | --- |
| verified | Equal domains and equal values on that domain proved for every requested output; whole-kernel scope additionally requires the synchronization check to pass. |
| race_free | The synchronization check passed; numerical equivalence was not requested. |
| not_wellsynced | Required synchronization/order relation is violated. |
| potential_race | Candidate conflicting memory accesses lack a proved ordering. |
| invalid_lifecycle | An allocation, barrier or other resource lifecycle constraint is violated. |
| equivalence_not_proven | Equal domains and equal values on that domain could not be proved; this is not a counterexample. |
| unsupported | A required operation, layout, address model, expression or representation is unsupported. |
| invalid_input | Malformed PTX/configuration/mapping or invalid invocation. |
| missing_input | A required file, reference or mapping is absent. |
| timeout | The configured time budget expired. |
| memory_budget_exceeded | The configured process memory budget was exceeded. |
| analysis_budget_exceeded | An algorithm work/node/iteration/output-family budget was reached. |
| allocation_failed | An allocator reported failure; physical OOM is not inferred. |
| tool_failure | Process crash, invalid process output, I/O failure or unexpected implementation failure. |

## Checks

Each check is `passed`, `failed`, `incomplete`, `not_run` or `not_requested`.

| Check | Meaning |
| --- | --- |
| `synchronization` | `passed`: the kernel is well synchronized, race free and its lifecycles are valid. `failed`: not_wellsynced, potential_race or invalid_lifecycle. `incomplete`: the analysis started but did not finish, for example because an operation was unsupported or the memory analysis could not decide an access pair. `not_run`: verification stopped before the check, for example at invalid input, unsupported PTX or a timeout. |
| `equivalence` | `passed`: the reference equivalence was proved. `incomplete`: the check ran without a proof, for example because the value analysis stopped, a budget was reached or no proof was found. `not_run`: a prerequisite failed or an input is missing. `not_requested`: no reference was given. |

Example: after the synchronization check passes, an equivalence timeout
reports result=timeout, checks.synchronization=passed and
checks.equivalence=incomplete. A potential race remains potential_race when
the reference is also missing; equivalence is not_run.

Branch choices from `branch_decisions` in input.json restrict the analyzed
executions. Results for such an input hold only for the supplied choices. No
stable output field repeats the choices: they appear only under
`internal.frontend.user_branch_decisions`, which is not a stable interface, so
keep the input with its result. `frontend_assumptions.data_branch_decisions`
lists decisions made under Representative branches or Bounded computation
loops, not the supplied choices.

Nested `definedness.status` and per-family proof receipts describe mathematical
properties, not alternative top-level result categories.

## Precedence

An observed synchronization, lifecycle or race failure remains the overall result. Only after the synchronization check passes may the equivalence result determine the overall category. Resource interruption retains completed checkpoints in checks.

A failed value analysis takes precedence over a missing or failed reference, because supplying the reference would not change the result: the result is then the value-analysis category (usually unsupported), and both causes appear in diagnostics. missing_input therefore means that the missing input is the only obstacle found.

verified requires all requested output families and same-domain equivalence; total definedness is reported separately; a family cap is analysis_budget_exceeded. equivalence_not_proven is not a counterexample. Configured process memory and algorithm budgets are distinct. Allocation failure never implies physical OOM without evidence.

## Additional fields

Fields depend on how far verification progressed. Missing fields are not proof
that a check passed, a count was zero or a resource was unused.

| Field | Meaning |
| --- | --- |
| `schema` | Report format identifier, such as `spverify.native-result/v1`; runs that stop early may report another identifier or none |
| `scope` | The operation performed, in `scope.operation`, such as `kernel_verification` |
| `assumptions` | Effective analysis assumptions with readable labels, stable keys, enabled flags, and triggered (null means unmeasured); do not substitute today's defaults for saved results |
| `assumption_reporting` | How to read `triggered` in `assumptions` |
| `assumption_hints` | Present when the analysis stopped at a construct that an assumption could cover; each hint gives the assumption's `key` and `label`, `suggested_value` and a `message` |
| `assumption_dependent` | True when any assumption is enabled, when `address_invariants` are given or when `tensor_properties` are given. The defaults enable some assumptions, so it is true under the defaults; `assumptions[].triggered` shows which assumptions a run used. Branch choices, pointer tables and shape arrays do not set it |
| `frontend_assumptions` | Input-side premises the analysis applied and their use: `pointer_contract`; the read-only input contracts `shape_input_contract` and `pointer_array_contract`; `tensor_properties` and the source lines they decided (`tensor_property_decisions`); decisions made under Representative branches or Bounded computation loops (`data_branch_decisions`); and usage counts such as `atomic_uses` |
| `verification_limitations` | Limitations that qualify a passed synchronization check |
| `pipeline_elapsed_seconds` | Time of the whole verification pipeline, in seconds |
| `elapsed_seconds` | Wall time, in seconds, of the whole call (Python verify and the CLI) or of a run stopped by its time or memory budget |
| `wellsync_timeout_seconds`, `whole_pipeline_timeout_seconds` | Time budgets, in seconds, of the synchronization check and of the whole verification pipeline. With Python verify and the CLI, both are the part of the total budget left after reference capture |
| `timeout_scope` | Which time budget expired: `whole_pipeline`, or `wellsync` for the synchronization check's budget |
| `resource` | The budget or resource that stopped the run: `kind` (such as `time` or `memory`), `origin` (such as `configured_budget`, `algorithm_budget` or `allocator`) and, where known, `limit` and `unit` |
| `peak_rss_mib` | Reported peak resident memory, in MiB |
| `rss_limit_mib` | Configured memory limit; null means no Sentryward memory budget |

Timing and memory fields are producer-specific measurements, not a guarantee
that every nested subprocess is included in a single figure.

## Equivalence and definedness

Equivalence requires the same domain and equal output values on that domain.
It does not require both functions to be defined for every possible input.
An informational `total_definedness_not_proven` diagnostic can coexist with
`verified`. Numerical proof receipts are under `internal.equivalence` in this
same object; no separate `equivalence.json` is produced. See the
[binding contract](reference-binding-json.md#equivalence-contract).

Synchronization findings take precedence over missing references or capture
failures. Read all diagnostics for every cause: a top-level category does not
erase the incomplete or blocked checks.

## Diagnostic details

`code` names the concrete cause and `stage` names the public stage.
Locations, operands, expressions, budgets and witnesses belong in the diagnostic
payload, including `details` where provided. Repeated occurrences may be grouped
with counts and ranges; samples may be truncated. Preserve these fields when
reading or summarizing a result.

A missing alignment proof is not a proven misalignment. A potential race is an
unordered candidate access pair. A configured memory budget, an algorithm budget
and a failed allocation are distinct causes; allocation failure alone does not
establish a physical out-of-memory condition.

## Race witnesses

Each entry of `evidence.races` is one pair of conflicting accesses without a
proved order. `first` and `second` describe the two accesses: the PTX
`location` (`file`, `line`, `column`), `mode` (`read` or `write`), `bytes`, an
`atomic_scope` for atomics, and ids such as `instruction`, `actor` and `event`
that refer to internal analysis records. `overlap` states why the accesses
meet. `region` gives the overlapping bytes (`address_space`, `allocation`,
`offset`, `bytes`) when they form a constant range. The search stops at the
first race, so `race_count` is not the number of racing pairs in the kernel.
`race_samples_truncated` is true when `races` lists fewer races than
`race_count`.

Clusters that behave alike, or CTAs when the launch has no clusters, may be
analyzed once for all of them. A race between two such units carries
`cross_unit`. `first_unit` and `second_unit` are the coordinates of the two
clusters (or CTAs); `first_cta` and `second_cta` are the coordinates of the two
CTAs. The accesses in `first` and `second` are those of the analyzed CTA,
translated to the listed CTAs. `allocation` names the global allocation where
they meet. `first_delta` and `second_delta` are the byte shifts of that
allocation in each listed unit relative to the analyzed one.

## Repeated entries

A warning or diagnostic that repeats identically in several CTAs or clusters is
grouped. The units are clusters when the launch has clusters, otherwise CTAs,
numbered in launch order. A grouped entry carries `multiplicity` (the number of
occurrences it stands for) and `members` (those units as `[first, last]` runs);
for diagnostics both fields are in `details`. The entry shown is the occurrence
in the lowest unit. The first entry of a list is written alone, without these
fields; its other occurrences follow as the next entry, whose `multiplicity`
and `members` count only them. Later grouped entries count every occurrence,
including the one shown. An entry that occurs once has neither field. Counts
such as `warning_count` remain totals.

## Resource limits

The CLI accepts `--timeout` and `--memory-limit-mib`; Python accepts
`timeout_seconds` and `memory_limit_mib`. Omitting the memory option or using
Python None imposes no Sentryward memory budget. Explicit memory limits must be
positive integer MiB values within the supported range; operating-system limits
still apply. See [invocation options](../README.md#verify-a-kernel).

The public result categories are listed above. They are maintained in
[config/result-categories.json](../config/result-categories.json).
