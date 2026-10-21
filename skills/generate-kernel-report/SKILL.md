---
name: generate-kernel-report
description: Generate one readable English report.md from saved SentryWard input, verifier output and matching PTX. Present tool findings separately from source-supported code explanations for GPU developers unfamiliar with the kernel.
---

# Generate a kernel report

Use the supplied `input.json`, `output.json` and matching PTX to create **one English `report.md`**. Existing source, identity and numerical-check files are optional. The reader knows GPU programming but has not read this kernel.

Scripts present saved findings; the agent explains relevant code. Keep those parts visibly separate. A reader should understand the finding, affected object and operations, supporting relationship, and any unresolved next check. Keep clean results short; explain one local finding directly; connect interacting findings through shared context.

Create only `report.md` at the requested destination, or default to an unused `build/skill-output/<kernel>/`. Generate on demand and leave generated reports uncommitted unless requested. Keep scratch elsewhere. In an existing example directory, leave other files unchanged. Do not rerun verification, change inputs, install dependencies or publish.

## 1. Preserve the saved result

Run from the repository root, substituting literal paths:

```sh
python3 SKILL_DIR/scripts/report_facts.py --input INPUT --result RESULT --ptx PTX --report OUTPUT/report.md --result-panel
```

The composer reads outcomes, diagnostics, counts, scope and assumptions directly from the artifacts. Identity conflicts are errors; missing values are not zero or success. See [outcomes.md](references/outcomes.md) to interpret the supplied result and check outcomes.

Inventory every diagnostic category, retained access pair and per-check outcome. Categories help navigation; repeated records are not independent bugs. Numerical analysis has a separate outcome and must not hide memory warnings. For a completed clean run with no unresolved diagnostics, check the base report without `--draft` and finish.

## 2. Trace the relevant code

```sh
python3 SKILL_DIR/scripts/source_context.py --input INPUT --result RESULT --ptx PTX > SCRATCH/source-context.md
```

Read **Supplied parameter entrances** and **Focus evidence**, then inspect the original PTX. Use `--at LINE,LINE` for additional slices. The index supplies definition candidates and local source deductions, not reaching definitions, execution records or feasibility proofs.

Establish the affected object's purpose, participants, admitting guards and relevant completion boundary. Substitute supplied scalar values before describing a path; inspect available definitions and branch/loop conditions. For asynchronous work, barrier reuse or release, use [source-trace.md](references/source-trace.md). Consult [evidence.md](references/evidence.md) for access identities, source mappings and ISA guarantees.

Use saved access pairs, `counterexample`, `rejection`, `source_events` and failed orders first. Physical PTX lines, source lines, instruction IDs, event IDs and actor IDs are distinct coordinates. A **verifier witness** is a saved model execution; a source-derived schedule is a **possible interleaving**. Preserve truncated intervals. Missing runtime traces do not prevent source reasoning, but an unestablished mapping must stay explicit.

## 3. Explain the finding

Write a scratch explanation plan using [draft-format.md](references/draft-format.md). It contains interpretation, visual relationships and checked evidence references; it cannot replace tool results. Follow [presentation.md](references/presentation.md) for code, diagrams and disclosure hierarchy.

Choose the reading path before the views: **finding → object's role in the computation → required relationship → what the source establishes → implication for the kernel**. Follow the relevant value, buffer or notification across operations; stop where its meaning for the finding is clear. Introduce that role before registers. A local finding can do this in one section; a clean result needs no added story.

For a larger case, give shared context once, then divide by questions that change the conclusion. Connect sections when one establishes the premise for the next; keep independent concerns separate rather than suggesting causation. End a section with a short `conclusion` when the reader needs its consequence or a bridge to the next relationship. Keep decisive uncertainty beside the claim it limits.

Name operations consistently across prose, diagrams and exact code. Choose a view that answers the current question: overlap, dependency, guarded path, phase comparison or supported participant sequence. Show the required relationship beside the established one when that makes the gap clear. Do not invent temporal order from listing order or repeat the entire kernel in every view.

Use source-supported action names in diagrams. Branches should show what each condition permits or skips, rather than requiring the reader to decode bare true/false labels. For multiple concerns, reuse section topic names in the overview; unique bold topic names link to their sections automatically.

Keep conditions that change the conclusion visible; put their derivations in named disclosures. Each expanded derivation begins with its answer, then supporting code and references. Use at most two disclosure levels. Code remains verbatim, with instruction-manual and physical-source links; annotations belong outside it. Use code spans for registers, predicates, fields and rule IDs.

## 4. Compose and review

```sh
python3 SKILL_DIR/scripts/report_facts.py --input INPUT --result RESULT --ptx PTX --draft SCRATCH/draft.json --report OUTPUT/report.md --result-panel
python3 SKILL_DIR/scripts/check_report.py --input INPUT --result RESULT --ptx PTX --draft SCRATCH/draft.json --report OUTPUT/report.md --result-panel
```

Use the same optional `--identity IDENTITY`, `--source FILE` and `--equivalence FILE` on both commands, only for supplied files. For GitHub links, `--artifact-base HTTPS_BLOB_URL_AT_COMMIT` must identify the same artifact bytes; paths are relative to the working directory. Otherwise preserve relative file layout when sharing.

Read the default view as a developer new to the kernel: can you locate the finding, follow the affected object's role, and say what each section adds to the conclusion? Then expand the derivations and check participant identity, predicate polarity, configured paths, barrier rounds, issue versus completion, and source-to-event attribution. Calculate conversions rather than guessing. Check that code supports its caption and that each view answers its neighboring text. Inspect GitHub rendering when available, including wrapping, links and disclosures.

Correct the plan and recompose; do not hand-edit generated Markdown. A failed plan leaves the existing report intact. The checker verifies composition, identities, citations and disclosure structure; semantic correctness and readability require review. If source interpretation remains unsupported, retain the useful base report and name the unresolved relationship. Return the report path and checks performed.
