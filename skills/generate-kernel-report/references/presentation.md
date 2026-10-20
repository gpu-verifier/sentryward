# Report presentation

The composer separates the **Verifier result**, **Code explanation** and **Evidence appendix**. `--result-panel` gives saved findings and scope a bordered GitHub-native frame; source interpretation follows below it. Keep long explanations left aligned.

## Reading order

Lead with what the tool found or where it stopped. Keep links to PTX, configuration and verifier output at the opening. Source interpretation follows the affected object through the relevant part of the computation: **what it is for → what the operations require → what is established → what this means for the finding**. The object may be a value, memory region, descriptor or synchronization notification. Follow only the part that explains this result.

A title names the relationship; its opening answers it. The view and short code excerpt establish that answer. An optional section `conclusion`, rendered after them and before the derivation, states the consequence or hands the established premise to the next section. Move reasoning there rather than duplicating the opening. A decisive limitation stays with the claim it limits, never only in the fold or final check.

Use the same semantic names in the overview, diagrams and code captions. Shared context locates several concerns; it does not make them one causal chain. A release question may follow a buffer from final use to notification to reclamation. A data-dependent analysis stop may only need to follow the value into its branch. Independent startup, gather and descriptor concerns need separate paths from the same kernel context. These are choices about the evidence, not mandatory report shapes.

Read neighboring sections without opening details: what has the reader learned, and why does the next question follow? Keep the answer explicit where the transition would otherwise require kernel knowledge. A clean result needs no invented investigation; complexity comes from interacting roles, paths or guarantees, not report length or diagnostic count alone.

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
