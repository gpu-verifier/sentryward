---
name: generate-kernel-report
description: Explain existing sentryward output to a kernel user in a concise Markdown report, with relevant PTX and local synchronization diagrams. Use after verifier output is available, not to run verification or debug the verifier implementation.
---

# Generate a kernel report

## Audience and inputs

Explain the verifier's findings to the kernel user. Take the recorded verifier
output, the corresponding PTX, and the input configuration when available.
Upstream source and artifacts from that run (events, HB graph, trace) are optional.
Use the verifier's reported conclusion as the basis of the explanation. Preserve
its status, severity, conditional scope, and distinction between candidates and
established results. Do not turn the report into a review of verifier internals.

Existing `examples/` files are read-only examples. Generate a new English
Markdown file at the requested new path; otherwise use a fresh directory under
`build/skill-output/`. Do not replace existing examples, configs, or reports.

## Read the evidence

- Identify the kernel entry and repository-relative PTX path. Check that supplied
  artifacts refer to the same input/run; compare recorded hashes when available.
  State mismatches rather than combining evidence from different runs.
- Read the actual output, including errors, warnings, assumption labels and keys,
  branch restrictions, and whether memory analysis ran or completed.
- Use the output's `semantic_assumptions` and usage records for the run's effective
  choices. `config/default-assumptions.json` describes new runs, not historical
  outputs. Consult `docs/assumptions.md` for meanings; never infer a missing recorded
  choice from today's defaults or reclassify a saved error using them. Explain
  assumption-related warnings and applicable enable-option hints from the output.
- Inspect the relevant PTX instructions and sufficient surrounding control flow.
  If upstream source is available, locate the corresponding block and link its
  public repository file, preferably at a fixed revision. PTX alone is sufficient
  when source is unavailable; do not invent a source mapping.
- Read existing event/HB/trace artifacts when needed. If evidence needed for an
  explanation is absent, name the missing information and the resulting limit.
  Preserve the reported result; do not invent HB edges or a witness schedule.
  Do not rerun the tool, change inputs, or modify implementation in this workflow.

## Write findings first

Keep the result status separate from warnings. In current outputs, `race_free`
may coexist with warnings under the recorded assumptions; do not rename it to a
warning-only result. Preserve the status stored in historical outputs.

Begin with the reported result and the minimum input scope needed to interpret it.
Explain errors/findings before warnings. Group repeated findings with the same
cause and code location; give a representative instance and the reported affected
range. Explain distinct locations or causes separately. Cover the supplied output,
not findings that would require additional searches. Mark truncated output as such.

For each group, use this compact structure:

1. **What happens:** a concrete one-sentence explanation.
2. **Code:** a short PTX excerpt with file and line locations; source excerpt if
   available and useful. Keep instructions exact and omissions explicit.
3. **Why this is reported:** explain the particular accesses, counters, or waits
   that lead to the finding. Relate addresses to their buffers where evidence allows.
4. **Local diagram:** for synchronization or race findings, show the relevant HB
   relations or a supported execution sequence, not the entire kernel.

Use descriptive node names such as “Write final statistics” and “Consumer wait
returns”; put instruction names and source locations beside them. In an HB graph,
`A -> B` means A happens before B. Keep address-overlap annotations visually
distinct from directed HB edges. Distinguish an instruction's issue from its async
execution when relevant. Show unordered conflicting accesses explicitly. Label a
trace sequence as a schedule rather than implying every step is an HB relation.
Mermaid in Markdown is suitable. Do not force a graph into an unsupported-feature
or missing-input report.

## Keep the report focused

- Use short paragraphs and specific subjects; explain necessary numeric identifiers.
- State assumption IDs and their recorded meaning only where they qualify a finding.
- If no finding exists, briefly state the result and scope; an interrupted,
  unsupported, or first-witness run does not establish exhaustive race freedom.
- Omit implementation architecture, speculative model diagnoses, generic background,
  and repair advice unless the user separately asks for them.
- Keep reproduction provenance compact: PTX/input links and available run/version
  identifiers. Do not replace the explanation with counters or hash inventories.
