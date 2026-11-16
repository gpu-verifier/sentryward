# Verification output (`output.json`)

The CLI emits one JSON object and Python verify returns the same structure.
result is one of 14 categories; checks and diagnostics explain the conclusion.

## JSON structure

The CLI writes one JSON object to stdout by default; Python `verify` returns the
same structure. `--output PATH` writes that object to a file. `--format summary`
prints a compact text listing of `result`, `checks`, diagnostics, and warning
counts on stdout; the JSON object is unchanged, and `--output` still stores it.
`summarize(result)` returns the same listing from Python.

The following is an abbreviated successful result, showing the fields used to
read the conclusion:

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

A warning with code `same-value-writes` names unordered writes of exactly the
same bytes whose stored values the value analysis proved equal; such writes do
not race ([Same-value writes](assumptions.md#same-value-writes)). There is one
warning per pair of write sites and CTA; writes in two CTAs count under the CTA
of the write whose (site, CTA) is the lesser. When the values are not proven equal,
or the value analysis does not decide them, the pair is a `potential_race` with
diagnostic `same_value_writes_unproven` or `same_value_writes_unchecked`.

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

Arrivals on a barrier that no operation of the launch waits on order nothing.
The synchronization check removes them first and lists them in
`internal.eliminated_synchronization` by PTX line and barrier, with their count.
Checks that do not depend on barrier phases still cover them: an eliminated
mbarrier arrival must follow the barrier's initialization
(`barrier_initialization_before_use_not_proven`) and precede its invalidation.

## Categories

| result | Meaning |
| --- | --- |
| verified | Equal domains and equal values on that domain proved for every requested output; whole-kernel scope additionally requires the synchronization check to pass. |
| race_free | The synchronization check passed and no reference was supplied, so equivalence checking was skipped (diagnostic `reference_expression_not_supplied`; the example runner, `index.md` and the report skill print `race_free (equivalence checking skipped)`). Without a reference the value view and memory elimination still run, so unordered writes of the same bytes are still proved equal (see [assumptions.md](assumptions.md)); the scalar fold and the comparison are skipped, and with the fold its checks, among them sparse-metadata validity (`sparse_metadata_invalid`, see [Undefined behavior](#undefined-behavior)). |
| not_wellsynced | Required synchronization/order relation is violated. |
| potential_race | Candidate conflicting memory accesses lack a proved ordering. |
| invalid_lifecycle | An allocation, barrier or other resource lifecycle constraint is violated. |
| undefined_behavior | An executed instruction can reach behavior the PTX ISA leaves undefined, for a value the input allows; the diagnostic gives a witness (see [Undefined behavior](#undefined-behavior)). No equivalence is checked. |
| equivalence_not_proven | Equal domains and equal values on that domain could not be proved; this is not a counterexample. |
| unsupported | A required operation, layout, address model, expression or representation is unsupported. |
| invalid_input | Malformed PTX/configuration/mapping or invalid invocation. |
| missing_input | A required file is absent, or a reference is incomplete (its function or its binding is missing). A run without any reference skips equivalence checking instead (race_free when the synchronization check passes). |
| timeout | The configured time budget expired. |
| memory_budget_exceeded | The process memory budget (configured, or the default) was exceeded. |
| analysis_budget_exceeded | An algorithm work/node/iteration/output-family budget was reached. |
| allocation_failed | An allocator reported failure; physical OOM is not inferred. |
| tool_failure | Process crash, invalid process output, I/O failure or unexpected implementation failure. |

## Checks

Each check is `passed`, `failed`, `incomplete`, `not_run` or `not_requested`.

| Check | Meaning |
| --- | --- |
| `synchronization` | `passed`: the kernel is well synchronized, race free (unordered writes of the same bytes are allowed when their values are proven equal) and its lifecycles are valid. `failed`: not_wellsynced, potential_race or invalid_lifecycle. `incomplete`: the analysis started but did not finish, for example because an operation was unsupported or the memory analysis could not decide an access pair. `not_run`: verification stopped before the check, for example at invalid input, unsupported PTX or a timeout. |
| `equivalence` | `passed`: the reference equivalence was proved. `incomplete`: the check ran without a proof, for example because the value analysis stopped, a budget was reached or no proof was found. `not_run`: a prerequisite failed, an input is missing or the value analysis found undefined behavior. `not_requested`: no reference was given, so equivalence checking was skipped. |

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
`frontend_assumptions.branch_choices_with_static_visits`, present only when
nonempty, lists the PTX lines of supplied choices that a statically known
predicate overrode on some visit.

Nested `definedness.status` and per-family proof receipts describe mathematical
properties, not alternative top-level result categories.

## Precedence

An observed synchronization, lifecycle or race failure remains the overall result. Only after the synchronization check passes may the equivalence result determine the overall category. Undefined behavior that the value analysis finds is reported after the synchronization check passes, before any other value-analysis or equivalence outcome, with two limits: the scalar fold, which holds the sparse-metadata check, runs only with a reference or a reported reference failure (or the native option `--analyze-scalar-fold`), so without one equivalence checking is skipped and the result stays race_free; and the sparse-metadata check runs after memory elimination completed, so a memory-elimination stop is reported instead. Undefined behavior that a value stage which did run finds is reported with or without a reference. Undefined behavior that the frontend decides while executing the kernel stops the analysis there, with or without a reference. Resource interruption retains completed checkpoints in checks.

A failed value analysis takes precedence over an incomplete or failed reference, because supplying the reference would not change the result: the result is then the value-analysis category (usually unsupported), and both causes appear in diagnostics. missing_input therefore means that the missing input is the only obstacle found. Without any reference, memory elimination runs (it proves same-value writes for the synchronization check) and the scalar fold does not. The native option `--require-equivalence` without a reference reports missing_input, after any value-analysis failure.

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
| `resource` | The budget or resource that stopped the run: `kind` (such as `time` or `memory`), `origin` (such as `configured_budget`, `default_budget`, `algorithm_budget` or `allocator`) and, where known, `limit` and `unit` |
| `peak_rss_mib` | Reported peak resident memory, in MiB |
| `rss_limit_mib` | Configured memory limit; null means none was given, and the default budget applies (see [Resource limits](#resource-limits)) |

Timing and memory fields are producer-specific measurements, not a guarantee
that every nested subprocess is included in a single figure.

## Equivalence and definedness

Equivalence requires the same domain and equal output values on that domain.
It does not require both functions to be defined for every possible input.
An informational `total_definedness_not_proven` diagnostic can coexist with
`verified`. Numerical proof receipts are under `internal.equivalence` in this
same object; no separate `equivalence.json` is produced. See the
[binding contract](reference-binding-json.md#equivalence-contract).

Reads of a verified `shape_arrays` table (a read-only input that the race-free,
memory-complete run of the same input proved the kernel never writes) are
replaced by its declared values on both sides, the kernel side included;
`internal.equivalence.fixed_input_context.applied` lists each use with its `side`.

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

## Undefined behavior

`undefined_behavior` is reported only for a reachable violation: an executed
instruction instance and a value its operand can hold under the input contract
that the PTX ISA leaves undefined. Where the verifier cannot decide whether a
value can violate the condition, the result stays `unsupported` with the
reason. An undeclared input value can hold any bit pattern, so an input that
says too little about an object reaches the violation; declare the values the
producer of the object can generate (for example `tensor_properties`
`field_values`, see [input-json.md](input-json.md#tensor-properties)) and the
remaining violations are reported as found.

The value-analysis rows (`sparse_metadata_invalid`) are checked by the scalar
fold, which runs with a reference or a reported reference failure (and with the
native option `--analyze-scalar-fold`). Without one, equivalence checking is
skipped together with these checks, and a kernel whose synchronization check
passes is `race_free`. The other rows are decided by the frontend in every run.

| Diagnostic code | Condition | `details` |
| --- | --- | --- |
| `sparse_metadata_invalid` | `tcgen05.mma.sp`: each 4-bit field of the sparsity metadata must be one of the values PTX ISA 9.7.18.10.9 lists (`0b0100`, `0b1000`, `0b1001`, `0b1100`, `0b1101`, `0b0110`, `0b1110` for the 2:4 and 4:8-pair kinds). 9.7.18.10.9.2 (`.kind::f16`, `.kind::f8f6f4`, `.kind::mxf8f6f4`, `.kind::i8`, `.kind::ti16`) says "The only meaningful values of the index are" these and gives no meaning to the rest (`isa_wording` `only_meaningful`); 9.7.18.10.9.3 (`.kind::mxf4`, `.kind::mxf4nvf4`) adds "Rest of the values result in undefined behavior" (`isa_wording` `undefined`) | `condition` (the quoted sentence), `isa_section`, `isa_wording`, `isa_reference`, `instruction`, `line`; `source` (`input_object`, `constant` — a TMA out-of-bounds fill or a value the kernel writes — or `unwritten_tensor_memory`); `object`, `property` (`field_values`, or `none`: every value) and `declared_values` for an input object; `operand_byte` and `field` (`low` or `high`) locate the field in the metadata operand, whose bytes are the Tensor Memory footprint in lane order (lanes 128p to 128p+127 for CTA part p, 4 bytes per 32-bit column), not A rows; `object_byte_offset` when constant; `note` for `unwritten_tensor_memory`; `witness_value` and `witness_bits`; `defined_values` |
| `cp_async_source_size` | `cp.async` with a known `src-size` larger than `cp-size` (PTX ISA cp.async: undefined behavior) | `source_size`, `copy_size` and the instruction location |
| `invalid_bulk_transfer_size` | `cp.async.bulk` with a known `size` that is not a multiple of 16 (PTX ISA 9.7.10.28.4.1: undefined behavior) | `size`, `isa_section` |
| `bulk_reduction_size` | `cp.reduce.async.bulk` with a known `size` that is not a multiple of 16 (PTX ISA 9.7.10.28.4.2: undefined behavior) | `size`, `isa_section` |
| `bulk_reduction_alignment` | `cp.reduce.async.bulk` with a shared-memory address that is not 16-byte aligned under some legal placement of its shared object (PTX ISA 9.7.10.28.4.2: the addresses "must be aligned to 16 bytes"; PTX ISA 5.4.5 places a variable only at a multiple of its `.align`, or of its base-type size, unless the input fixes its base) (a global address is relative to an input object whose base alignment is not given: `bulk_reduction_base_alignment_unknown`, unsupported) | `address_space`, `offset` (the analysed placement), `alignment`, `isa_section`; for an address inside a shared object `shared_object`, `declared_alignment`, `offset_in_object`, and the witness placement `witness_object_base` and `witness_address` |
| `tmem_address_not_uniform` | `tcgen05.ld` or `tcgen05.st` whose lanes use different `taddr` values (PTX ISA 9.7.18.8.3/9.7.18.8.4: undefined behavior) | `lane` and its `taddr`/`resource`, `lane0_taddr`/`lane0_resource`, `line`, `warp`, `occurrence`, `instruction`, `isa_section` |
| `tmem_aligned_lanes_missing` | `tcgen05.ld` or `tcgen05.st` that a thread of its full warp does not execute at that occurrence (it exited or diverged): PTX ISA 9.7.18.8.3/9.7.18.8.4 make both undefined (`.aligned` and the exited-thread clause) | `missing_lanes`, `line`, `warp`, `occurrence`, `instruction`, `isa_section` |
| `tmem_lifetime_lanes_missing` | `tcgen05.alloc`, `tcgen05.dealloc` or `tcgen05.relinquish_alloc_permit` that a thread of its full warp does not execute (the same rules in PTX ISA 9.7.18.7.1) | the missing lanes in the message; `violation_ptx_line`, `violation_warp`, `violation_occurrence` |
| `tmem_invalid_alloc_size` | `tcgen05.alloc` nCols outside Table 58 (non-exclusive: a power of two in [32, 512]; exclusive: a multiple of 32 in [32, 512] on sm_100/sm_103/sm_110, [32, 576] on sm_107) (PTX ISA 9.7.18.7.1: undefined behavior) | nCols and target in the message; `violation_ptx_line`, `violation_warp`, `violation_occurrence` |
| `tmem_nonuniform_alloc_size`, `tmem_nonuniform_dealloc_size` | lanes use different nCols (PTX ISA 9.7.18.7.1: undefined behavior) | the values in the message; `violation_ptx_line`, `violation_warp`, `violation_occurrence` |
| `tmem_increasing_alloc_size` | a later allocation of the CTA, ordered after an earlier one, has more columns (PTX ISA 9.7.18.7.1: undefined behavior) | both sizes in the message; `violation_ptx_line`, `violation_warp`, `violation_occurrence` |
| `bulk_copy_alignment` | `cp.async.bulk` with a shared-memory `srcMem` or `dstMem` that is not 16-byte aligned under some legal placement of its shared object (PTX ISA 9.7.10.28.4.1: the addresses "must be aligned to 16 bytes"; placements as for `bulk_reduction_alignment`; a global address is not decided) | `address_space`, `offset` (the analysed placement), `alignment`, `operand`, `isa_section`; the shared-object and witness-placement fields of `bulk_reduction_alignment` |
| `tmem_alloc_destination_outside_shared_window` | A generic-addressed `tcgen05.alloc` whose `dst` every lane holds as a pointer into a global or parameter object (PTX ISA 9.7.18.7.1: a `dst` outside the `.shared::cta` window is undefined behavior) | the object in the message; `violation_ptx_line`, `violation_warp`, `violation_occurrence` |
| `tmem_alloc_after_relinquish` | `tcgen05.alloc` ordered after a `tcgen05.relinquish_alloc_permit` of its CTA (PTX ISA 9.7.18.7.1 calls it "illegal"; reported as undefined behavior under the strict reading) | `violation_ptx_line`, `violation_warp`, `violation_occurrence` |
| `tcgen05_cta_group_mixed` | The analysed kernel's tcgen05 instructions use both `.cta_group::1` and `.cta_group::2` (PTX ISA 9.7.18.7.1: they "must specify the same value") | both PTX lines in the message; `violation_ptx_line` |
| `aligned_lanes_missing` | Another `.aligned` instruction (tcgen05.wait, barrier.cluster, tensormap.cp_fenceproxy, ldmatrix, stmatrix, tcgen05 lifetime operations) that a thread of the warp does not execute at that occurrence ("all threads in the warp must execute the same" instruction) | `missing_lanes`, `line`, `warp`, `occurrence`, `instruction`, `isa_section`; for ldmatrix and stmatrix `missing_lane`, `warp`, `isa_section`, with the location as `ptx_line` |
| `tmem_lane_alignment_mismatch` | A layout C or F `tcgen05.mma` whose A, D and sparsity metadata do not share one Tensor Memory lane alignment ("must use the same Tensor Memory lane alignment") | `d_lane`, `a_lane`, `metadata_lane` and the instruction location |
| `mma_sparsity_version` | `.kind::mxf4` or `.kind::mxf4nvf4` instruction descriptor bit 12 against the target: v0 on sm_100a, sm_103a and sm_110a, v1 on sm_107a (PTX ISA 9.7.18.4.2.1 and 9.7.18.10.9.3: undefined behavior otherwise) | `target`, `sparsity_version`, `required`, `isa_section` |

Every row is decided from the values of an executed instruction instance, except
`tcgen05_cta_group_mixed`: PTX ISA 9.7.18.7.1 requires that "All tcgen05 instructions
within a kernel must specify the same value", so it is a static scan of every
tcgen05 instruction of the entry function, executed or not. Following
PTX strictly, a "must" requirement and a list of the only meaningful values are
binding like an explicit "undefined": a violation is reported, and `isa_wording`
or `isa_section` says which form the ISA uses.

The witness value is one the input contract permits; it is not an observed
hardware failure.

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
`race_count`. A race reported with diagnostic `same_value_writes_unproven` or
`same_value_writes_unchecked` is a pair of unordered writes of the same bytes
whose values were not proven equal; its `region` is their common footprint.

Clusters that behave alike, or CTAs when the launch has no clusters, may be
analyzed once for all of them. A race between two such units carries
`cross_unit`. `first_unit` and `second_unit` are the coordinates of the two
clusters (or CTAs); `first_cta` and `second_cta` are the coordinates of the two
CTAs. The accesses in `first` and `second` are those of the analyzed CTA,
translated to the listed CTAs. `allocation` names the global allocation where
they meet. `first_delta` and `second_delta` are the byte shifts of the first
and second access in their listed units relative to the analyzed one: the
allocation's shift, or the shift at the access's instruction site when the
allocation moves by different amounts at different sites.

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
`timeout_seconds` and `memory_limit_mib`. Explicit memory limits must be
positive integer MiB values within the supported range; operating-system limits
still apply. `--fail-on` does not change the result object; it only selects when
a finding or other requested category becomes exit status 1. See
[invocation options](../README.md#verify-a-kernel).

Omitting the memory option or using Python None applies the default budget:
90% of the memory the verifier process may use, enforced on it like a
configured limit. That memory is the machine's physical memory, bounded on
Linux by the memory limit of the process's cgroup (`memory.max`, or
`memory.limit_in_bytes` with cgroup v1). A run that reaches it ends with
`memory_budget_exceeded` instead of being stopped by the operating system; its
`resource` names the `default_budget` origin and the effective `limit` in MiB,
while `rss_limit_mib` stays null. The default therefore differs between
machines, and it does not count other processes. The default only stops a run:
budgets that scale with a configured limit (the look-ahead below, and class
mode's internal logs) stay as without a limit.

Verifiers that run at the same time can share the default budget. The native
option `--default-budget-jobs N` (a positive integer, default 1; not combined
with `--rss-limit-mib`) enforces one Nth of the default on each; `resource`
then also gives `default_budget_jobs`. It changes only the enforced limit: the
origin stays `default_budget`, `rss_limit_mib` stays null and the budgets above
stay as without a limit. `tools/rerun_examples.py --jobs N` passes it when N > 1
and `--rss-limit` is not given.

The memory limit also bounds work that runs ahead on worker threads. When CTAs
form classes, later CTAs are executed while earlier ones are analyzed. At most
the worker count of them are held in flight and, where the allocator reports
its usage, only as many (at least one) as fit in a quarter of the memory limit
at the first CTA's allocation (without a configured limit, a quarter of the
memory the process may use). The threads of one analyzed CTA are also lowered on the
workers. Neither changes a result; only time and peak memory differ.

When the analysis of CTA classes cannot finish exactly, the verifier analyzes
the launch again without classes, in the same process.
`internal.frontend.unit_restart` records the `reason` and whether the class
analysis's synchronization verdict was kept (`kept_synchronization_verdict`;
see [same-value writes](assumptions.md#same-value-writes)). Without a kept
verdict, the result is that of the analysis without classes. A verdict is kept
only when the classes failed because the values of the analyzed CTAs do not
stand for the others' (reason `value_lifting`) and the synchronization check
passed without needing those values. The second analysis then repeats the
well-synchronization check, whose order the value analysis uses, but not the
race search or the lifecycle order checks, and reports the class analysis's
synchronization result as is. That result has no races and no memory warnings,
so what comes from the class analysis is `checks.synchronization`,
`verification_elapsed_seconds`, the warnings of a recovered program
(`duplicate-barrier-initialization`) and the synchronization statistics in
`internal`, which count only the analyzed CTAs. The value analysis and the
equivalence check are the second analysis's.
Elapsed time and peak memory include both attempts and the release of the
first attempt's memory between them.

The second analysis is not started for `value_lifting` when its value view,
estimated from the class analysis's by scaling to every CTA of the launch,
exceeds 2^26 value instances. The synchronization result is then the class
analysis's, and the stage that refused ends with `analysis_budget_exceeded`:
diagnostic `unit_restart_limit_exceeded`, with `resource` `origin`
`algorithm_budget`, `kind` `per_unit_value_instances`, the `limit` and the
`estimate`. Two kinds of stage refuse:

- The value analysis, before memory elimination finishes: it ends so, and
  unordered writes of the same bytes stay unchecked
  (`same_value_writes_unchecked`).
- The scalar fold, after memory elimination finished: memory elimination has
  already decided those writes, and the scalar fold
  (`internal.value_analysis.scalar_fold`) ends so.

The public result categories are listed above. They are maintained in
[config/result-categories.json](../config/result-categories.json).
