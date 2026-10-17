---
name: generate-kernel-report
description: Generate one readable English report.md from saved SentryWard input, verifier result and matching PTX. Present tool findings separately from evidence-supported code and synchronization explanations, for GPU developers unfamiliar with the kernel.
---

# Generate a kernel report

Deliver **one English `report.md` per kernel**, for a GPU programmer who has not read this kernel. Inputs: saved configuration, verifier output and matching PTX; existing source/identity files are optional. Keep scratch outside the delivery directory; default to an unused `build/skill-output/<kernel>/`. Do not rerun verification or kernels, install dependencies, change inputs or publish.

The report has **Verifier result → Code explanation → Evidence appendix**. The first part faithfully presents the tool; the second explains the relevant code. A reader should understand what happened, to which object/operations, why the source matters, and what to check next. These are reading boundaries, not a fixed amount of content: a clean result needs no explanation, one local finding needs one focused account, and interacting findings need connected accounts with shared context.

Within an explanation, introduce the kernel's relevant purpose, then connect the reported operation to it. Give the decisive condition its own short paragraph when it changes the conclusion. Diagrams must name the actual comparison, object or operation; a generic “Test” or “Step 1” should not force a return to the prose. Use the operand guide for the few registers needed to read an excerpt. Keep a branch's guard, affected operation and rejoin point together; move unrelated setup or loop bookkeeping to its named derivation. Keep decisive conditions in the main account; do not turn every sentence into a card or bullet.

Name each disclosure for the evidence it contains, such as “Derivation: tile ownership and guards”; avoid a long list of tasks beginning with “Check” or “Follow”. Keep one outer derivation and, if needed, one inner source-reference disclosure. The first expanded paragraph should answer the named question before listing records. Let the renderer provide the spacing and hierarchy; do not add empty headings, manual spacer lines or repeated separators.

Keep the presentation light: a short check-status line does not need another table inside the result panel. Use tables for comparisons with meaningful rows and columns. Around a code excerpt, prefer one short action caption and the necessary operand meanings; avoid stacking several introductions. Keep instructions verbatim and put explanatory annotations outside them. Reference-only headings must identify themselves as references, rather than appearing to restart the explanation.

## Preserve the result first

Use literal paths in separate commands; no shell-variable setup or chained commands:

```sh
python3 SKILL_DIR/scripts/report_facts.py --input INPUT --result RESULT --ptx PTX --report OUTPUT/report.md --result-panel
```

The composer owns saved outcomes, diagnostics, counts, scope and assumptions. Do not rewrite these in the explanation. Input identity conflicts are errors. Missing/null/unknown is not zero or success. Read [outcomes.md](references/outcomes.md) for the supplied status: current outputs use `result`/`checks`/`diagnostics`; legacy outputs use `status`. Numerical verification has its own outcome, usually in embedded `equivalence`.

Inventory **every diagnostic category, retained access pair and per-check outcome** before choosing the story. Repeated records are not independent bugs. A numerical stop must not hide memory warnings; a model limitation is not a protocol violation. For a completed clean run with no warnings or unresolved diagnostics, run the checker without `--draft` and finish. Do not invent a mechanism investigation to fill a template.

## Resolve the explanation before writing

```sh
python3 SKILL_DIR/scripts/source_context.py --input INPUT --result RESULT --ptx PTX > SCRATCH/source-context.md
```

Read **Supplied parameter entrances** and **Focus evidence** completely. Start with the saved scalar values, then the different diagnostic categories, operand definitions, role-path leads and nearby barrier address groups. Use `--at LINE,LINE` and original PTX for clipped or unresolved slices. These are source-reading leads, not verifier conclusions, reaching definitions or proof of feasibility.

For each relationship you intend to explain, record in scratch:

- **Object and participants:** what the operation does in this computation; its input/output role; selector definitions and actual taken/fallthrough edges to the operation. A nearby block does not establish warp ownership. Follow thread registers to their special-register sources before decoding masks.
- **Admitting conditions:** substitute supplied scalar values into the outer work/exit condition before describing the kernel's normal workload. Zero-work launches can still execute setup, dispatch and teardown. Trace first/later iterations and local predicates, decisive copy sizes, guards and descriptors to their input origins. A field declared `tensor_map` does not establish captured descriptor contents.
- **Required milestone:** issue, data use, completion, notification or release. For async/lifecycle findings, read [source-trace.md](references/source-trace.md) and complete the relevant barrier ledger: initialization, all matching arrivals/waits, participant counts, literal parity and established phase transitions. A count match alone does not establish async completion. Compare all relevant load/wait/free sites before identifying an exceptional one.

An access candidate can involve two operations issued by the same participant. Check their version-matched ISA ordering guarantees before proposing a missing wait; preserve the saved candidate while distinguishing source guarantees from the model's access/effect bindings. For synchronization failures, resolve startup parity separately from later reuse before drawing a phase sequence.

These are working answers, not another report section. **Finish source analysis that the supplied material allows.** Missing dynamic bindings can remain missing; an uninspected available definition or barrier round is unfinished work. If an upstream path is untraced, scope the claim to threads reaching the inspected block instead of saying every thread executes it. Record counts alone do not establish participation. A real gap names the unavailable value, mapping or model rule. Do not turn it into an unsupported remedy such as adding a wait or supplying data that may already exist.

Use saved `counterexample`, `rejection`, `source_events`, access pairs and failed orders first. Source lines, physical PTX lines, instruction IDs, event IDs and actor IDs are different coordinates. Only associate them with evidence. Model execution is a **verifier witness**, not a hardware trace; mark omitted intervals. Source-derived schedules are **possible interleavings** and must respect participants, branches, loops and completion. Absence of a runtime trace does not prevent source reasoning.

## Turn the reasoning into a readable report

Read [draft-format.md](references/draft-format.md) to write a version-2 scratch plan and [presentation.md](references/presentation.md) for the shared visual language.

- Orient the reader with the affected object's purpose, then follow the handoff, reuse, release or blocked analysis. Introduce only the computation needed for that account. Keep decisive guards, configured scope and assumptions beside the conclusion they qualify; fold their derivation, not the condition itself.
- When several concerns coexist, give them short names and separate local sections sharing one context. Use the same names in the overview and headings. A map earns its place when it locates the concerns in the code or relates their objects; a short list can introduce categories without repeating them as boxes. Do not invent causal arrows between concerns.
- **Show the relationship the prose asks the reader to reconstruct.** Use a dependency graph for a value feeding an unsupported operation; a branch view for conditional access; a phase comparison for barrier reuse; a participant view for supported concurrency. A second copy of the result diagram is not source explanation. Use `guarded-path` for supported skip-and-rejoin corridors so the bypass stays visible. Explain each arrow's meaning; listing order is not completion.
- For competing accesses, a compact participant-by-step view can show the two operations and the next synchronization without asserting either completed first. Derive a possible interleaving independently when saved actor bindings are absent, and state whether it identifies the retained pair. Do not leave the reader to reconstruct supported parallel behavior from a site list.
- Pair that view with the exact instructions establishing it, including both sides of the relation. Use short semantic captions, decisive opcode/operand emphasis, instruction-manual links and PTX source jumps. Name operand purposes before registers; use `code.operands` where useful. Registers, predicates, fields and rule IDs use code spans. Keep longer setup with its derivation.
- Each section gives its answer before its proof. Fold derivations by reader question, at most two levels; opening one should resolve that question before offering exact records. Tables compare parallel facts; explanations belong outside cells. Dimensions appear only when useful. Reread adjacent answers without headings: the next section should follow naturally, not switch to a new field inventory.

## Compose, inspect and finish in this invocation

```sh
python3 SKILL_DIR/scripts/report_facts.py --input INPUT --result RESULT --ptx PTX --draft SCRATCH/draft.json --report OUTPUT/report.md --result-panel
python3 SKILL_DIR/scripts/check_report.py --input INPUT --result RESULT --ptx PTX --draft SCRATCH/draft.json --report OUTPUT/report.md --result-panel
```

Use the same optional `--identity IDENTITY`, `--source FILE` and `--equivalence FILE` on both commands, only for supplied files. Correct the plan and rerender; never hand-edit the Markdown or substitute a handwritten report when execution fails. A bad plan must leave the base report intact.

Read the **whole generated report**, including expanded derivations. Verify the actual story, not just the checker's PASS: role identity, predicate polarity, configured loops, barrier rounds, load/store bypasses, issue versus completion and source-to-event attribution. Calculate numerical conversions and byte layouts rather than guessing. Remove speculative causes/remedies; preserve unresolved facts beside the affected claim. A truncated sample supports statements about retained records, not about all findings. Missing PTX-to-model binding does not erase the static explanation. For one local warning, combine context and mechanism; move unrelated sites and setup into supporting evidence. A short report should still answer why this warning matters under the saved rule.

Opening links identify PTX, configuration and verifier output. Displayed JSON pointers and matched PTX locations must be clickable: the composer links them to original-file lines, not unsupported GitHub `#/field` fragments or a pretty-printed copy's line numbers. Preserve relative layout when sharing. For GitHub review, optional `--artifact-base HTTPS_BLOB_URL_AT_COMMIT` on both commands gives direct links; first verify that commit contains the same input bytes. Native GitHub preview is preferred for checking diagrams, wrapping and disclosure hierarchy.

The commands above use `--result-panel` on both composer and checker. This groups saved findings and scope in a bordered panel, leaves the source explanation below it, and retains raw status identifiers in a disclosure. Result-specific relationships, instruction links and the full evidence remain. The layout works with or without an explanation; do not manually restyle the generated Markdown. Judge whether the finding is visible before the raw identifiers, and whether the explanation actually follows that finding.

Only `report.md` belongs in the output directory. Return its path and checks performed. If source interpretation remains unsupported, retain the basic report and identify the unresolved relation; do not count it as a successful explanation. Formatting checks alone prove neither the reasoning nor human readability.
