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
  "warnings_complete": true,
  "internal": {}
}
```

The verifier has two public checks. `synchronization` covers synchronization,
memory race freedom and resource lifecycles; `equivalence` compares the kernel
with its reference. `result` names the cause of the conclusion, and `checks`
shows how far each check got.

`diagnostics` is an array. Each diagnostic's `code` identifies a concrete cause;
`stage` is the public stage it belongs to: `input`, `synchronization`,
`equivalence` or `pipeline`. A producer may include a readable `message`;
`details` retains operands, long expressions, source sites, resource limits and
other structured evidence, and `details.internal_stage` names the internal
processing step. Do not read only the first diagnostic. Diagnostics can be
informational, so a nonempty array does not itself mean failure.

`warnings` contains structured warning samples, not another result category.
`warning_count` counts recorded warnings and can exceed the sample array length;
`warning_samples_truncated` identifies omitted samples. `warnings_complete: false`
means warning collection did not complete, so zero samples is not evidence of
no warnings. Early failures may omit count/sample fields; absence is not zero.

`evidence` holds the witnesses behind a failed synchronization check: race
samples (`races`, `race_count`), required orders that were not established
(`failed_required_orders`) and synchronization counterexamples.

`internal` holds analysis details: the value analysis, scalar fold, frontend
and address-model statistics, equivalence receipts, counters and timing
breakdowns. It is not a stable interface. Its sections keep their own
diagnostics but have no result of their own; read the conclusion from
`result` and `checks`. See [configuration formats](../README.md#documentation)
for the input files.

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
| `synchronization` | `passed`: the kernel is well synchronized, race free and its lifecycles are valid. `failed`: not_wellsynced, potential_race or invalid_lifecycle. `incomplete`: the analysis started but did not finish, for example because the memory analysis could not decide an access pair. |
| `equivalence` | `passed`: the reference equivalence was proved. `incomplete`: the comparison ran without a proof. `not_run`: a prerequisite failed or an input is missing. `not_requested`: no reference was given. |

Example: after the synchronization check passes, an equivalence timeout
reports result=timeout, checks.synchronization=passed and
checks.equivalence=incomplete. A potential race remains potential_race when
the reference is also missing; equivalence is not_run.

scope identifies the operation; conditional branch assumptions also remain in
analysis_scope. Nested definedness.status and per-family proof receipts describe
mathematical properties, not alternative top-level result categories.

## Precedence

An observed synchronization, lifecycle or race failure remains the overall result. Only after the synchronization check passes may the equivalence result determine the overall category. Resource interruption retains completed checkpoints in checks.

A failed value analysis takes precedence over a missing or failed reference, because supplying the reference would not change the result: the result is then the value-analysis category (usually unsupported), and both causes appear in diagnostics. missing_input therefore means that the missing input is the only obstacle found.

verified requires all requested output families and same-domain equivalence; total definedness is reported separately; a family cap is analysis_budget_exceeded. equivalence_not_proven is not a counterexample. Configured process memory and algorithm budgets are distinct. Allocation failure never implies physical OOM without evidence.

## Additional fields

Fields depend on how far verification progressed. Missing fields are not proof
that a check passed, a count was zero or a resource was unused.

| Field | Meaning |
| --- | --- |
| `assumptions` | Effective analysis assumptions with readable labels, stable keys, enabled flags, and triggered (null means unmeasured); do not substitute today's defaults for saved results |
| `assumption_dependent` | Whether the result depends on an enabled assumption or on trusted input properties |
| `frontend_assumptions` | Input-side assumptions the analysis applied, such as the pointer contract and `tensor_properties` |
| `verification_limitations` | Limitations that qualify a passed synchronization check |
| `pipeline_elapsed_seconds` | Time of the whole verification pipeline, in seconds |
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

## Repeated entries

A warning or diagnostic that repeats identically in several CTAs or clusters is
written once, with `multiplicity` (the number of occurrences) and `members`
(the CTAs, or clusters when the launch has clusters, it occurs in, numbered in
launch order as `[first, last]` runs; for diagnostics both fields are in
`details`). Counts such as `warning_count` remain totals.

## Resource limits

The CLI accepts `--timeout` and `--memory-limit-mib`; Python accepts
`timeout_seconds` and `memory_limit_mib`. Omitting the memory option or using
Python None imposes no Sentryward memory budget. Explicit memory limits must be
positive integer MiB values within the supported range; operating-system limits
still apply. See [invocation options](../README.md#verify-a-kernel).

The public result categories are listed above. They are maintained in
[config/result-categories.json](../config/result-categories.json).
