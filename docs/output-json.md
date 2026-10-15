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
    "memory": "passed",
    "equivalence": "passed"
  },
  "diagnostics": [],
  "warnings": [],
  "warning_count": 0,
  "warning_samples_truncated": false,
  "warnings_complete": true
}
```

`diagnostics` is an array. Each diagnostic's `code` identifies a concrete cause;
`stage` identifies where it occurred. A producer may include a readable `message`;
`details` retains operands, long expressions, source sites, resource limits and
other structured evidence. Nested stage diagnostics and proof receipts remain
available in the full output; do not read only the first diagnostic. Diagnostics
can be informational, so a nonempty array does not itself mean failure.

`warnings` contains structured warning samples, not another result category.
`warning_count` counts recorded warnings and can exceed the sample array length;
`warning_samples_truncated` identifies omitted samples. `warnings_complete: false`
means warning collection did not complete, so zero samples is not evidence of
no warnings. Early failures may omit count/sample fields; absence is not zero.

The full output also contains scope, timings, memory measurements and
stage-specific evidence. Numerical proof receipts are under `equivalence` in
this same object; no separate `equivalence.json` is produced. See
[configuration formats](../README.md#documentation) for the input files.

## Categories

| result | Meaning |
| --- | --- |
| verified | Equal domains and equal values on that domain proved for every requested output; whole-kernel scope additionally requires synchronization and memory checks to pass. |
| race_free | Synchronization and memory verification passed; numerical equivalence was not requested. |
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

Each diagnostic has a code and, where supplied by its producer, message, stage,
location/path and resource evidence. Preserve original structured witnesses,
warnings and expression receipts. checks records synchronization, memory and
equivalence separately with passed, failed, incomplete, not_run or not_requested.
scope identifies the operation; conditional branch assumptions also remain in
analysis_scope. Nested definedness.status and per-family proof receipts describe
mathematical properties, not alternative top-level result categories.

Example: after successful safety checking, an equivalence timeout reports
result=timeout, checks.synchronization=passed, checks.memory=passed and
checks.equivalence=incomplete. A potential race remains potential_race when
the reference is also missing; equivalence is not_run.

## Precedence

An observed synchronization/lifecycle/race failure remains the overall result. Only after synchronization and memory prerequisites pass may the numerical result determine the overall category. Resource interruption retains completed checkpoints in checks.

verified requires all requested output families and same-domain equivalence; total definedness is reported separately; a family cap is analysis_budget_exceeded. equivalence_not_proven is not a counterexample. Configured process memory and algorithm budgets are distinct. Allocation failure never implies physical OOM without evidence.

## Additional fields

Fields depend on how far verification progressed. Missing fields are not proof
that a check passed, a count was zero or a resource was unused.

| Field | Meaning |
| --- | --- |
| `memory_analysis_complete` | Whether memory analysis completed; read alongside the memory check and findings |
| `equivalence` | Nested numerical result, diagnostics, definedness and output-family proof receipts |
| `semantic_assumptions`, `assumptions` | Effective choices keyed by configuration name; metadata uses readable labels, stable keys, enabled flags, and triggered (null means unmeasured); do not substitute today's defaults for saved results |
| `verification_elapsed_seconds` | Time reported for the verification stage, in seconds |
| `peak_rss_mib` | Reported peak resident memory, in MiB |
| `rss_limit_mib` | Configured memory limit; null means no Sentryward memory budget |

Timing and memory fields are producer-specific measurements, not a guarantee
that every nested subprocess is included in a single figure.

## Equivalence and definedness

Equivalence requires the same domain and equal output values on that domain.
It does not require both functions to be defined for every possible input.
An informational `total_definedness_not_proven` diagnostic can coexist with
`verified`. See the [binding contract](reference-binding-json.md#equivalence-contract).

Synchronization or memory findings take precedence over missing references or
capture failures. Read nested diagnostics for all causes: a top-level category
does not erase the incomplete or blocked checks.

## Diagnostic details

`code` names the concrete cause and `stage` names the processing step.
Locations, operands, expressions, budgets and witnesses belong in the diagnostic
payload, including `details` where provided. Repeated occurrences may be grouped
with counts and ranges; samples may be truncated. Preserve these fields when
reading or summarizing a result.

A missing alignment proof is not a proven misalignment. A potential race is an
unordered candidate access pair. A configured memory budget, an algorithm budget
and a failed allocation are distinct causes; allocation failure alone does not
establish a physical out-of-memory condition.

## Resource limits

The CLI accepts `--timeout` and `--memory-limit-mib`; Python accepts
`timeout_seconds` and `memory_limit_mib`. Omitting the memory option or using
Python None imposes no Sentryward memory budget. Explicit memory limits must be
positive integer MiB values within the supported range; operating-system limits
still apply. See [invocation options](../README.md#verify-a-kernel).

The public result categories are listed above. They are maintained in
[config/result-categories.json](../config/result-categories.json).
