# Bounded source tracing before composition

Use the supplied PTX and input, not a previous report. Trace the claims needed for the current finding; keep a concise exact-source reference that lets a reviewer check the operation, local guard and relevant completion boundary. A kernel overview is not a full reconstruction of every accumulator and output path.

## Resolve the mechanism before drafting

Use `source_context.py` to navigate the matched entry, then read the original lines. Work through these dependencies in scratch; carry their short answers into the report rather than copying the inspection packet:

1. **Operation and milestone.** Name the failed requirement's endpoints. An event mapped to a source instruction may represent its issue, completion or notification; that mapping does not make those milestones interchangeable.
2. **Participant.** Follow the relevant special register through role arithmetic and dispatch to the operation. Count admitted threads, including any election. A wait reached by a whole warp and a copy issued by its elected lane have different participants.
3. **Local path.** Resolve the guard, incoming paths and relevant iteration from the supplied scalar values. Mark where actual data or unknown control stops the derivation.
4. **Completion.** Follow the matching address, initialization and waits. For reused barriers fill the ledger below. Distinguish what the local source establishes from the model event bindings it cannot establish.
5. **Reader's answer.** State the supported relationship and its decisive condition first; show the smallest useful relation or participant view; put the derivation directly beneath it in a named fold.

A source index lists definition candidates, not reaching definitions. Resolve candidates using control flow. “Needs tracing” is not a missing input; finish that inspection before returning. A dynamic mapping can remain unresolved while the source roles and protocol are fully explained.

## Computation and scalar sources

Map named input/output descriptors and argument declarations to the relevant operands. Identify matrix multiplication from its MMA operations, not merely matrix-shaped descriptors. Trace accumulator initialization or auxiliary contribution when the explanation makes a claim about them; otherwise leave the exact supplied configuration in the saved-evidence component.

For a scalar affecting the mechanism, follow **declared parameter offset → first load or pointer selection → reload guards → live use**. Read all assignments to the live register in the supplied entry. A zero stride can skip a later reload; the earlier value then remains live. A null pointer can select a by-value parameter. Record that path, not merely the nearest load.

For each reload guard, name the actual field tested by its predicate. The scalar-predicate index includes named inputs beside exact expressions. A pointer and a stride may both be zero but have different roles: a predicate comparing the stride is not a null-pointer test. Preserve that distinction in the source explanation.

An accumulator-control register is not inherently an unknown runtime value. The preflight MMA flag index shows adjacent comparisons and all input assignments, including later loop updates. Use them to distinguish initialization from subsequent iterations; do not describe a supplied 0/1 definition as a missing value. Trace its initialization and the first/later updates before describing the product. Keep the visible operation sentence in data roles (multiply, accumulate, scale, stage, copy); register values and compiler instruction counts belong here. A short reference should include the special-register source of a decoded thread mask, scalar pointer/stride selection, and the decisive initialization, with exact ranges. Do not inflate the visible computation sentence with counts of compiler instructions.

## Local access versus full reachability

Follow a thread register back to its special-register source before decoding a mask. Derive the predicate polarity and show which local conditions admit or skip the access. Preserve an outer exit when it narrows this interpretation. Distinguish the evaluated local guards from a complete scheduler path and from recorded issuer identities. An unpredicated branch is unconditional when its block is reached; absence of a predicate definition is not missing evidence. Its block may still have conditional predecessors.

For repeated/unrolled accesses, make a compact phase-to-source lookup: relevant copy, commit and wait sites; address or coordinate changes; any materially different final predicate. Compare distinct copy/wait predicates through their definitions and intervening assignments. Different names alone do not imply different conditions.

## Configured completion and exit

Find relevant final waits/drains in the entry. For each one you mention, inspect **all incoming branches**, not just the branch immediately above it. Reduce their predicates using the configured scalar sources and reload conditions. A direct jump into the wait and a later branch around the wait are different paths; do not join their conditions with “or” unless the actual control flow supports that relation.

Use the local branch corridor in the preflight index to distinguish reached definitions from bypassed blocks. It follows only indexed constant decisions and stops at an unknown predicate. Explain that predicate using its latest definition on this corridor, not the nearest definition in listing order. The corridor assumes its start was entered; it does not prove kernel-wide reachability.

Say which configured path skips or reaches the wait, and give the physical ranges that establish it. Qualify this as static configuration analysis, not a saved runtime trace. If an actual data-dependent value or missing source prevents a reduction, name that specific missing value. “Not traced” is not an explanation when the needed constants, assignments and branches are supplied.

A reached wait establishes only its ISA milestone and participant scope. A skipped final wait does not by itself prove a race; retained classifications and instruction guarantees remain separate evidence.

## Keep the reference readable

### Barrier reuse and deallocation

When a barrier is the proposed link between asynchronous work and deallocation, trace its address on both sides, initialization count, later count/transaction updates, arrive sites and wait parity. Count the participants admitted by the relevant guards; a warp-level role does not imply one arrival per warp. Separate each participant's program order from the barrier completion condition and from the asynchronous instruction's completion guarantee. An arrival count of one alone does not prove that an arbitrary arrival releases a later wait.

Compare every relevant load, wait and deallocation site in the supplied entry before saying a completion wait is absent. A different wait kind may order a different operation. A guarded store between a load and an arrival needs its bypass shown or explicitly scoped; drawing a straight chain must not silently erase that branch. A source relation can be explained without binding it to a specific saved instruction ID; make that remaining mapping explicit.

If a constant, initialization or guard is present, follow it now. “Not traced here” identifies unfinished analysis, not missing input. A genuinely unresolved next step names the unavailable mapping, value or model rule and keeps the static relations already established.

Build a small **barrier ledger** in scratch before writing a release claim:

| Source fact | Derive from the supplied entry |
| --- | --- |
| Object and initialization | Address expression, initial count and initial phase |
| Participants | Role guards, thread count and any election |
| Initial source round | Every relevant arrival and wait; literal parity |
| Later source round | Every relevant arrival and wait; literal parity |
| Admitting path | Configured work, exits and loop conditions |
| Async completion | The instruction/milestone needed before reuse or free |

Count threads admitted by a role, not merely warps or instruction sites. An opposite-parity wait can pass during startup because the initialized barrier already has a different phase; do not automatically describe it as waiting for the next arrival. A complete arrival count explains the barrier protocol; it does not alone prove that an earlier asynchronous load completed. Show the derived protocol as a short comparison when it helps the reader; do not copy this authoring checklist into the report.

Separate a source's first/later arrival from a proven barrier phase. A thread's second arrival is not automatically in phase 1: establish completion of the previous phase before assigning it. Otherwise label rows as source rounds, give the literal wait parities, and leave the unestablished transition explicit. Never turn a count match into a release edge.

If a relation remains unknown, name that gap beside it; do not draw an established release edge. Missing event-to-thread bindings do not prevent tracing static warp roles. For a forward conditional operation followed by a shared continuation, use the `guarded-path` component in [draft-format.md](draft-format.md). It derives polarity and draws the bypass from the selected PTX. For other graphs, put guards on the edges themselves: a paragraph elsewhere does not qualify an unconditional arrow. Never merge a conditional store and an unconditional arrival into one mandatory stage.

Use short semantic groups: operation, local guard, repeated sites and exit. Each begins with its derived relation and exact physical ranges. Include only decisive exact instructions; supporting setup can be accounted for by expressions and ranges. Use the excerpt helper for copied instructions. Preserve unresolved dynamic actor/event mappings separately from the static deductions. The reader's question fold states the answer; this reference shows how to check it.

Separate the conclusion from its derivation with a sentence break and a meaningful lead-in. For a long exit derivation, use a short sequence: **live value → incoming paths → final gate**. Each step keeps the relevant expressions and physical lines beside its role. Do not pack all predicate assignments into one paragraph. When several passes share operation types, a small pass/copy/commit/wait table exposes the repetition; explain the exceptional final predicate immediately below it. These forms organize the evidence, not replace any source task above.

For TMEM, a base handle loaded from shared memory is still a tensor-memory address value, not a shared-memory base. Without a supplied mapping, base+256 versus base+258 does not prove an address lies inside a saved 16-byte write. The retained pair supplies the byte intersection; source arithmetic supplies operand relations only.

Quote the exact branch instruction and its predicate, not the next instruction's line. An elect.sync output flag and a setp comparison of that flag are different values: if the skip predicate is (flag == 0), the branch on that predicate is taken when the predicate is true, not when the predicate is zero.

The instruction can name a metadata address while the retained effect is not mapped to an operand. Explain the source operand's role but keep the saved read-to-operand attribution open until a supplied effect mapping establishes it. Also, four unrolled store sites and a descriptor box do not by themselves establish full tile coverage; that needs coordinate and path support. Leave descriptor-axis extents to the exact configured-geometry view.

For tensor-store coordinates, use descriptor-axis position (first, second, third). Do not label a varying tuple component a logical row or column in the report. A source-to-logical-axis mapping is a separate derivation, not something tuple order alone supplies.
