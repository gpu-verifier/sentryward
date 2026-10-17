# PTX and evidence boundaries

Read for semantic explanations or ambiguous identifiers. Consult the matching official ISA; this file is not a substitute for instruction-specific rules.

## Memory records

An access's byte count describes the recorded effect, not necessarily an entire instruction transfer, tile or allocation. An overlap may expose neither access's complete interval. Distinguish allocation-relative byte offsets, internal endpoint IDs, event/instruction IDs and physical file lines. Without an explicit mapping, call actors “recorded actors”; do not decode thread coordinates from their numbers or bind an effect to a convenient operand.

When one recorded source location matches several unrolled physical sites, label the chosen excerpt as representative beside the excerpt itself. Say there that the retained event is not uniquely assigned to this occurrence; a mapping caveat only in a later fold leaves the visible code connection ambiguous.

Matching footprints, site visits or registers do not prove equal data. Different registers do not prove different addresses. A truncated list cannot establish the unsaved distribution, independent-bug count or complete enumeration. Preserve all retained groups and explain repeated fields once.

Retained access identity includes `access_index`, `effect_index` and the endpoint's internal address ID as well as actor, event, instruction and source location. Preserve those exact values in the named record group; grouping shared fields is allowed, dropping them is not. The checker only detects some missing labels, so review the actual record-to-table mapping too.

## Ordering

A static guard shows which threads may issue, not which did issue. Do not turn a uniform operand or matching saved footprint into a proven duplicate-copy cause. Untraced reassignments or scheduler branches remain a gap even for small fixed workloads; a cancellation assumption alone does not prove a unique dynamic path.

Keep issue, completion, visibility and observation distinct. Issuing an async instruction is not the same milestone as executing its memory effect; a model's immediate visibility of an executed effect does not make issue imply completion. State which operations and participants a wait covers, and what may remain pending. Check implicit instruction pipelines before concluding that the absence of a nearby wait explains a candidate. A listing, a static control-flow path and a happens-before graph are different evidence.

For bulk-group copies, distinguish a per-issuing-thread commit from completion. A `.read` wait concerns the specified read-side milestone, not destination-write completion; its operand governs allowed pending groups. Verify the exact instruction form and ISA version rather than copying this description mechanically.

For `tcgen05`, check pipeline pairings and same-thread versus cross-thread conditions. A record naming one actor does not, by itself, establish the issuing-thread relation needed by an ISA rule. If the saved candidate and the rule cannot be reconciled without that mapping, preserve the candidate and identify the missing evidence; do not silently override the verifier or manufacture a repair. Do not present a short list of possible explanations as exhaustive: an unrecorded cause stays unrecorded.

Useful PTX ISA 9.2 sections, only when this is the supplied version:

- [Bulk tensor copy/store](https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor)
- [TMEM copy](https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html#tcgen05-instructions-tcgen05-cp)
- [Dense tcgen05 matrix multiply](https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html#tcgen05-mma-instructions-mma)
- [Bulk commit](https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-commit-group)
- [Bulk wait](https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group)
- [tcgen05 pipeline pairings](https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html#tcgen05-memory-consistency-model-pipelined-instructions)
- [Same-thread pattern](https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html#tcgen05-memory-consistency-model-canonical-sync-patterns-pipelined-same-thread)

## Lowering and model conditions

Preserve predicate polarity, signedness, overflow behavior, guards and joins when simplifying arithmetic. State the range over which a reduced expression is valid. A declared buffer without contents is not known data; fixed dimensions may leave a data-dependent predicate unresolved.

Trace an operand to the definition that reaches its use under the supplied configuration. A later load of the same parameter may belong to a skipped reload path; citing it does not establish the value used by the current path. Check intervening assignments and branch conditions, or preserve that origin as a gap.

When separate predicate names guard a copy and its wait, compare their definitions and intervening assignments before implying that the issuer can skip the wait. Distinct predicate names alone do not establish distinct conditions.

Read assumptions from the saved run, not current defaults. Preserve quantifiers: “every cancellation request fails” is an assumed rule, not a list of observed failures. A run-level trigger does not show that the assumption caused one finding. Keep aliasing, extent, launch-isolation and assertion contracts with the evidence, and surface those that materially qualify the reported conclusion. Unbounded modeled allocations are not actual bounds checks.

Attribute contract statements to the exact structured fields that establish them. Do not use a similarly named standalone flag to invent a different effective contract. If the saved artifacts do not explain how apparently conflicting fields relate, retain both in reference and identify that interpretation gap there.

## Classification is not a conditional safety theorem

Report the verifier's status under its saved model. When a rule turns matching accesses into warnings without proving value equality, say exactly that. Do not rewrite it as “race_free holds if and only if the values are equal”, infer what a run without the rule would return, or treat equal values as sufficient hardware safety. Those require separate evidence. Preserve non-exhaustive coverage beside the visible result.

A null applicability flag means unknown. Zero recorded execution effects do not prove that every configured model rule was irrelevant to an earlier frontend stop. State the recorded direct stop reason and the unmeasured applicability separately. In particular, do not write that model rules "did not cause the stop" when their applicability is unknown. Neither positive nor negative causal attribution follows from zero counters or null trigger flags.

## Previously checked instruction forms

These version-scoped notes preserve already checked official rules, so the author need not rediscover them when the supplied PTX matches. The linked sections above are the authoritative references. Check the official text for a different version or instruction form.

- **PTX 9.2 `cp.async.bulk.commit_group`:** creates a bulk async group for the issuing thread's previously issued, uncommitted bulk operations. A commit is not completion and does not group all lanes' operations together.
- **PTX 9.2 `cp.async.bulk.wait_group.read N`:** permits at most N newest groups of that issuing thread to remain pending at the read-side milestone. For older groups, descriptor/source reads must finish; destination writes need not finish at this wait. With N=1 the newest group may still have pending reads. This does not by itself order stores from different issuing threads. State these as instruction guarantees, not recorded completion states.
- **PTX 9.0 `cp.async.wait_group N`:** waits for the issuing thread's committed async-copy groups until at most the N newest groups remain pending. It concerns copy completion for that thread; it is not a general cross-thread fence. A pipeline explanation must identify which slot/group is older and which is newest. The final drain and any barriers still need their own code evidence.
- **PTX 9.0 [`cp.async.wait_all`](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group):** is equivalent to `cp.async.commit_group` followed by `cp.async.wait_group 0`. That implicit commit can create an empty group when prior copies were already committed. Distinguish explicit source commits from implicit operations before claiming that every commit contains a copy; empty groups are trivially complete.
- **PTX 9.0 [`shfl.sync.bfly`](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl-sync):** the source lane is `lane XOR b[4:0]`, with a validity check against `maxLane` derived from the packed clamp/segment operand. Check the actual XOR offsets as well as that operand when deriving a subgroup; the segment mask alone does not establish butterfly isolation. The member mask separately identifies participating threads.

Keep a local guard's truth table separate from the set of threads able to reach that guard. Retain relevant outer branches when deriving a full-path eligibility set; otherwise explicitly say the local predicate alone has been evaluated.

An untraced path does not justify “every lane is eligible”, “only these threads reach” or a title promising reachability. Say “the evaluated local guards accept these lanes” and name the untraced part. Two evaluated guards are still not a complete path proof. A run-wide opcode count cannot close that gap.

For unrolled or looped code, separate the repeated body from prologue, final pass and drain. Audit universal words such as “each”, “every” and “always” against those boundaries and implicit operations in both prose and captions. If the final copy is committed without a read, say “the first seven copy/read passes, then a final unread copy”, rather than “each pass copies then reads”. Define which values actually contribute to the output; issued copies and consumed values need not have the same count.
