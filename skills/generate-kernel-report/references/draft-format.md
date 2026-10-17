# Source explanation plan, version 2

The plan is scratch, not an output users must supply or read. It holds source explanations, presentation choices and evidence references; it is not a verifier intermediate result or additional proof. Keep it outside the delivery directory. Tool facts are read directly from the original artifacts. No `status`, counts or replacement fact block is accepted in this plan. Avoid copying raw records into prose: reference them and explain the relationship the reader needs. A clean base report needs no plan.

A minimal plan (fictitious names and lines; obtain yours from evidence):

```json
{
  "version": 2,
  "title": "Tile copy",
  "sections": [{
    "title": "The wait releases the source buffer",
    "answer": "Explain the scoped relationship established by these instructions.",
    "evidence": [
      {"kind":"ptx","lines":"12-14","contains":["cp.async.bulk.wait_group.read"]},
      {"kind":"result","pointer":"/status","equals":"race_candidate"}
    ],
    "code": {"caption":"Copy and source-read wait","lines":"12-14","focus":["cp.async.bulk.wait_group.read"]},
    "steps": [{
      "label":"Completion boundary",
      "text":"Give the source derivation and its scope.",
      "evidence":[{"kind":"ptx","lines":"14","contains":["wait_group.read"]}]
    }]
  }]
}
```

Each section requires `title`, `answer` and nonempty `evidence`. `code`, `steps`, `diagram`, `comparison` and `evidence_title` are optional. Steps require `label`, `text` and their own evidence; they become headings inside the section's evidence fold. A step may also carry `code` in the same format as a section: use this for supporting setup, dispatch or longer excerpts. Keep only the decisive short excerpt in the default view. Use `evidence_title` to name a reader's check, e.g. `Check the arrival count and phase`. A `comparison` has `headers` and equal-width `rows` of short cells, supported by the section's evidence. Use it for parallel sites or phases, not long explanations. Titles name the operation or relation; avoid a general architecture lesson. Keep answers short enough to read together. Length is guidance, not permission to drop a decisive condition.

Attach supporting code to the step that explains it. A step may also carry a `comparison` in the same format as a section; use it when parallel phase or binding details support a derivation rather than the default reading path. Do not add a generic final step repeating all section references. The renderer groups exact references by step in an `Exact source references` disclosure after the derivation, retaining additional section references there once. This is the second disclosure level; use the first level to explain the reasoning, not to repeat token lists.

A step can use a `diagram` in the Mermaid/math format below when a calculation or dependency is easier to follow visually. Its own evidence supports the diagram. Preserve branch conditions and remainder cases in the explanation; a neat arithmetic picture must not silently generalize beyond the configured inputs.

Code excerpts support optional `operands`: e.g. `[{"token":"%r7","meaning":"Release barrier address"}]`. Use a small guide only when it helps decode the decisive operands. The renderer checks that each token occurs in the selected instructions; the section evidence must support its meaning. Do not rename source registers. Known PTX 9.2 and 9.3 instruction families link automatically to their versioned NVIDIA manuals, both in snippets and inline code. For other versions or unmapped instructions, use `code.links` only with a verified, version-matched HTTPS destination. Source-range links come from the supplied PTX path and `--artifact-base`, not authored model IDs.

## Evidence

- `{"kind":"result","pointer":"/rejection/code"}` or `{"kind":"input","pointer":"/launch/block_dim"}` resolves the exact JSON pointer. Optional `equals` checks its exact value and type. Null is valid evidence; an absent pointer is an error. Array indices are zero based.
- `{"kind":"ptx","lines":"12-14,20","contains":["%p1","bra"]}` checks physical PTX lines and literal tokens. Tokens must occur in those lines. This does not prove the written interpretation.
- `{"kind":"source","file":"kernel.cu","lines":"40-46","contains":["consumer_wait"]}` reads only an explicitly supplied `--source kernel.cu`. Do not substitute a saved source line for a PTX line.

## Optional context and next inspection

A top-level `context` has `summary`, `evidence`, optional `title` (default `Code context`), optional `caption`, optional `diagram` (the Mermaid/math format below), optional `steps` and `evidence_title` (the same derivation format as sections), and optional `shapes` with `operators`. The context is a headed subsection, not an unmarked introduction. Use a small pipeline or lifetime map when the finding depends on several roles, stages or intermediate objects: locate the affected object and show where the detailed explanation starts. Its edges need the same evidence and meaning as section diagrams. When understanding this overview needs a derivation, use `steps` to explain the object’s journey before the nested exact references; do not make the first expanded view a list of tokens. Do not reconstruct unrelated computation or imply an execution witness. Use shapes only when data dimensions locate the finding. A shape is `{"name":"A","rows":{"label":"M","value":64},"columns":{"label":"K","value":128}}`; operators between shapes are `×`, `→` or `+`. Dimensions describe logical data, not pointer widths or descriptor-axis packing. Explain irregular shapes in text instead of forcing rectangles. Launch geometry is already in the evidence.

A top-level `next` has `text` and `evidence`, rendered under the final `Next check` subsection. The composer numbers context, explanation sections and this next check in reading order. Author titles without numbers. Omit the next check for a clean result without a meaningful follow-up. Do not invent a bug or speculative fix to fill a section.

## Diagrams

A section may include `{"kind":"mermaid","source":"flowchart LR\n A[\"Copy issued\"] --> B[\"Local wait\"]","meaning":"Arrows show local source order, not observed execution."}`. `math` accepts the same fields. Supply no code fences. Use `context`, `operation` and `focus` classes for the shared theme. Explain the exact meaning of arrows. Evidence belongs to the containing section.

For a simple forward branch, use `{"kind":"branch","comparison_line":12,"branch_line":13,"effect_line":18,"inputs":["Lane slot","Owner slot"]}`. The renderer derives comparison polarity and the skipped local/shared/global operation from PTX. Use a general diagram for more complex control flow.

When a conditional operation rejoins an unconditional operation, prefer `{"kind":"guarded-path","comparison_line":12,"branch_line":13,"effect_line":18,"join_line":25,"inputs":["Lane bit 16","0"]}`. This renders **both paths** from the comparison to the common operation. It reads the opcode, comparison, branch polarity and target from PTX; `inputs` only names the operands in their original order. Cite the operand derivation in the section. Selected operations must be unpredicated, with no intervening control transfer between the branch and its target or between the target and `join_line`. The component rejects unsupported corridors instead of flattening their control flow. It establishes local source flow, never async completion or a cross-thread dependency.

The base report already draws the verifier's failed ordering requirement. Explain what its endpoints mean; do not redraw it from a guess about the words `required_events`. A source graph beside it should answer a different question, such as which lanes skip a store or which phase a wait tests.

A `guarded-path` can include `before_line` to start at the affected operation before the comparison. The renderer checks the selected local corridor for control transfers; this does not prove that every incoming path executes the operation. Optional `path_labels` maps `"true"` and `"false"` comparison outcomes to short, source-supported names such as lane groups. These names are authored interpretation, like `inputs`; cite their derivation. With both options, named edges show the branch directly from the preceding operation, keeping the exact predicate in the caption. Branch inversion still determines which outcome skips the effect.

## Participant-by-step view

Use `kind:"timeline"` when you can support the participants and order. Required fields:

- `basis`: exactly `possible interleaving` or `verifier witness`.
- `participants`: one to four objects with distinct `id`, readable `label`, and `evidence` establishing the mapping.
- `steps`: ordered objects with `participant`, `action`, `evidence`. A witness step also supplies its exact integer `event`.
- `conditions`: `participants`, `branches`, `iterations`, `completion`, each an object with `text` and `evidence`. Explain the actual scope or why an item is inapplicable; do not fill these with generic caveats.
- For `verifier witness`, `sequence` is the exact saved `/counterexample/.../events` JSON pointer. The renderer rejects reversed/unlisted events and marks skipped or truncated intervals.

Rows are logical steps, not elapsed time. Empty cells mean no step shown, not that a participant is idle. A possible interleaving must be consistent with all relevant source conditions, not merely listing order. Missing participant mappings prevent a **verifier witness** participant table; they do not prevent a **possible interleaving** derived independently from thread guards and source ordering. For that view, establish concrete participants, the admitting branch/iteration, and the relevant completion constraints; say whether it is bound to the retained pair. If those source relations cannot be established, use the supported relation diagram. Do not silently turn static thread groups into observed issuers.

## Compose

Pass the plan through `report_facts.py --draft SCRATCH/draft.json` with the original inputs. Run `check_report.py` with the same arguments. A failed plan does not overwrite an existing good report. Only `report.md` belongs in the output directory.
