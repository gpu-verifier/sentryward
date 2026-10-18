# Report presentation

The composer separates the **Verifier result**, **Code explanation** and **Evidence appendix**. `--result-panel` gives saved findings and scope a bordered GitHub-native frame; source interpretation follows below it. Keep long explanations left aligned.

## Reading order

Lead with what the tool found or where it stopped. Keep links to PTX, configuration and verifier output at the opening. For source interpretation, introduce the affected object's purpose, then follow **operation → relationship → decisive condition → evidence**. Several findings share context but retain their own short names and local explanations.

A title names the relationship; its paragraph answers it. Use the same semantic names in the overview, diagrams and code captions. Explain why the next section follows when the connection is not obvious. A clean result needs no invented investigation; a complex result needs connected explanations rather than one oversized diagram.

## Views and code

| Reader's question | Useful view |
| --- | --- |
| Which accesses overlap? | Region/access comparison without temporal arrows |
| Which value feeds this operation? | Data-dependency graph |
| Which path reaches or skips it? | Guarded branch with a common continuation |
| Which barrier round is required? | Phase comparison |
| What can each participant do? | Participant columns with supported steps downward |

Give every arrow a meaning: control flow, dependency, saved order or required order. A missing requirement is not a proven bad execution. Distinguish a **verifier witness** from a **possible interleaving**; show omitted intervals and participant conditions. Do not construct a timeline from an overlap alone.

Use neutral context, blue operations and purple focus sparingly; labels carry meaning without color. Draw dimensions only when they locate the affected region, with labels attached to the corresponding axes. Avoid duplicating diagram values in prose.

Choose the shortest exact snippet that establishes its caption. Include decisive operands, guards, retry branches or both sides of a relationship. Highlight opcodes/operands without altering instructions; put annotations outside the code. Preserve both instruction-manual links and physical PTX source jumps. Name operand purposes before registers and use a small operand guide when useful. Registers, predicates, fields and rule IDs use code spans.

Tables compare parallel properties. Put long reasoning outside cells. Prefer short semantic labels and keep exact long identifiers in adjacent references.

## Disclosure hierarchy

Keep results, decisive relationships and conclusion-changing conditions visible. Fold their derivations, then exact supporting references, at most two levels deep. Each summary names the evidence or question; expanded content starts with an answer. Let the renderer handle spacing and heading levels.

Supporting code belongs with the derivation that uses it. Avoid repeating a generic code/reference recap. Preserve retained diagnostics, counts, truncation and important rules; large execution or equivalence details remain in original files with precise links and hashes.

## GitHub review

Inspect default and expanded views, light/dark appearance and narrow widths when available. Check that links open original files and physical lines, diagrams remain legible, and disclosures retain a visible hierarchy. JSON pointers alone are not GitHub navigation.

GitHub rendering can fail for math in large disclosures. Split long derivations by purpose while preserving evidence, then inspect the native preview. Do not hide a decisive qualification to shorten a block. Screenshots help diagnose crowding, clipping and disconnected reading order; they do not replace the full report.
