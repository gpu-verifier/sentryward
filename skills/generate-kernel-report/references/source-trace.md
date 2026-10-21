# Source tracing

Use the supplied input and PTX to explain the current finding. `source_context.py` locates entry-local definitions, role-path leads and barrier groups. Read the original instructions to resolve control flow; the index is not a reaching-definition or feasibility proof. Optional `report_facts.py --bindings` provides parameter-load and bounded guard inspection with the same input arguments.

## Follow the affected operation

1. **Object and milestone:** connect the operation to its computational purpose; distinguish issue, memory effect, completion and notification.
2. **Participant:** trace special registers through role arithmetic, dispatch and election. A warp reaching a wait and its elected lane issuing a copy are different participant sets.
3. **Admitting path:** substitute supplied scalar values into guards, outer exits and relevant loop iterations. Keep data-dependent conditions explicit.
4. **Required relationship:** connect both endpoints through source order, synchronization and the applicable ISA guarantee. Keep any missing model-to-source mapping separate.

For each scalar, follow **parameter offset → initial load or pointer selection → reload guards → live use**. Inspect intervening assignments. Zero-work configurations can still execute startup and teardown. A null pointer selecting a by-value parameter and a zero stride bypassing a reload are different paths.

## Branches, repeated accesses and final waits

Evaluate the exact branch polarity and target; retain the affected operation and rejoin. Local guard eligibility does not establish full-path reachability. For a wait or drain, inspect all relevant incoming paths and distinguish direct entry from a later bypass.

For repeated operations, compare copy, commit, wait and coordinate changes across the prologue, body and final pass. Trace accumulator-control flags through initialization and updates when they affect the explanation. An unavailable runtime value is a real gap; an available but uninspected definition is unfinished analysis.

For a retained access pair, identify each instruction family, address space, milestone and participant binding. Check the exact version's same-thread or collective guarantees before attributing the candidate to absent synchronization. A skipped wait alone does not prove a race.

## Reused barriers and release

Before claiming that a barrier orders asynchronous work before reuse or deallocation, establish:

| Relationship | Source support |
| --- | --- |
| Barrier identity | Address derivation, initialization count and initial phase |
| Participants | Role guards, election and admitted thread count |
| First and later rounds | Matching arrivals, count/transaction updates, waits and literal parities |
| Admitting path | Configured work, exits and iteration conditions |
| Completion boundary | Relevant instruction guarantee before reuse or free |

Count admitted threads rather than source sites or warps. A second arrival is not automatically phase 1: establish completion of the prior phase. An opposite-parity wait can succeed at startup without a new arrival. A count match explains the barrier protocol, not arbitrary asynchronous completion.

Compare all relevant load, wait and release sites before claiming an absent wait. Include a polling predicate and retry branch when explaining successful wait → release. Preserve bypasses around conditional stores or copies instead of drawing an unconditional chain. `guarded-path` in [draft-format.md](draft-format.md) renders supported local skip/rejoin corridors.

## Carry the answer into the report

State the supported relationship and decisive condition first. Use one compact view to connect the operations; place exact derivations in a named disclosure. Keep operation names consistent with the snippets and explain each edge's meaning. Shared context belongs once, not in every finding.

Keep unresolved values or mappings beside the relation they qualify. A static participant protocol can be explained without pretending it is a saved runtime schedule. For longer derivations, group support by purpose: **live value → incoming paths → final gate**, or a concise copy/commit/wait comparison. Exact ranges and verbatim instructions support the explanation; a list of registers alone does not explain it.
