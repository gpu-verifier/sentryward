# Evidence and ISA boundaries

Use this reference when interpreting access records, ambiguous identifiers or instruction guarantees. Consult the official ISA matching the supplied PTX version and exact instruction form.

## Access records and source locations

- A recorded byte count describes an effect, not necessarily the full transfer, tile or allocation. An overlap need not expose either access's complete interval. Matching footprints do not establish equal data; different registers do not establish different addresses.
- Preserve endpoint identity: actor, event, instruction, `access_index`, `effect_index`, internal address ID and saved source location. Allocation-relative offsets and physical PTX lines are different coordinates. Decode actor IDs only with an explicit mapping.
- When one saved source location matches multiple unrolled sites, label an excerpt as representative beside it. A source operand's role does not establish that a retained effect belongs to that operand.
- Retained samples describe retained records. Keep original counts, category coverage and truncation; do not infer unsaved distributions or independent-bug counts.

## Instruction ordering

Distinguish **issue, completion, visibility and observation**. A reached wait establishes its instruction-specific milestone and participant scope. Check implicit pipeline guarantees before explaining a candidate as a missing wait. A local program path, happens-before relation and execution witness are different evidence.

For `tcgen05`, check pipeline pairings, group qualifiers and same-thread versus cross-thread conditions. Equal recorded actor IDs alone do not establish the ISA's issuing-thread relationship. Preserve the saved candidate when reconciling it requires an unavailable access/effect mapping.

Selected official references:

| Version | Instruction or rule | Scope to check |
| --- | --- | --- |
| PTX 9.2 | [Bulk commit](https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-commit-group) | Groups the issuing thread's prior uncommitted bulk operations; commit is not completion. |
| PTX 9.2 | [Bulk wait](https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group) | `.read N` permits the newest N groups to remain pending at the read-side milestone; it does not establish destination-write completion or order other threads' stores. |
| PTX 9.2 | [Pipelined pairings](https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html#tcgen05-memory-consistency-model-pipelined-instructions) and [same-thread pattern](https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html#tcgen05-memory-consistency-model-canonical-sync-patterns-pipelined-same-thread) | Pairing, CTA-group qualifier and issuing-thread conditions. |
| PTX 9.0 | [Async-copy waits](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group) | `wait_group N` covers the issuing thread's groups. `wait_all` includes an implicit commit, which may be empty. |
| PTX 9.0 | [Shuffle](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl-sync) | Butterfly XOR offset, clamp/segment validity and member mask have separate roles. |

## Source deductions

Follow definitions that reach the operation under the supplied configuration. Preserve predicate polarity, signedness, overflow, guards and joins when reducing arithmetic. A later load of the same parameter may be bypassed; a declared buffer or tensor-map parameter does not supply its contents.

Keep a local guard's truth table separate from full-path reachability. Distinct predicate names do not prove distinct conditions: compare definitions and intervening assignments. For loops and unrolled code, distinguish prologue, repeated body, final pass and drain. Issued operations and values contributing to the output may have different counts.

TMEM handles loaded from shared memory still name tensor memory. Register-offset arithmetic does not establish a recorded byte intersection without the corresponding address mapping. Descriptor-axis positions do not establish logical matrix axes without a separate derivation.

## Model conditions

Read assumptions from the saved run, with their exact fields and quantifiers. Enabled, triggered and unknown applicability are separate states. Zero execution counters cannot determine a rule's role in an earlier frontend stop. Surface assumptions that change interpretation beside the conclusion they qualify.

A rule that classifies matching accesses as warnings need not prove equal copied values. Do not convert the saved status into a conditional hardware-safety theorem or predict the result of disabling that rule. Unknown causal explanations remain unknown; suggested fixes need their own support.
