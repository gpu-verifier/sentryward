# Report evaluation

## Current public examples

The nine reports use the saved inputs, outputs and PTX at public commit
[`94a6b26`](https://github.com/gpu-verifier/sentryward/commit/94a6b26b1990a4b9df7ded3727acc4b4cc36f23c).
No verifier or kernel was rerun. All artifact links point at that commit after
byte comparison; source excerpts remain exact.

Five explanations originated in independent Codex invocations on the private
predecessor. The public inputs now have different assumption records, and some
saved diagnostics have changed. The plans were migrated and their reports
recomposed, rather than treating the old outputs as current results:

- Assumption references follow the saved rule's key and current record position.
- Named assumptions display their saved labels, without inventing old `A001/A002` IDs.
- GQA no longer describes the removed `warp_tmem_load_completion` setting as disabled.
- GQA's retained model orders and all result-layer fields come from the current output.

The source/code relationships were reviewed again. These are **migrated examples,
not fresh one-shot outputs on the public inputs**. The internal plans and exact
recomposition hashes are in the [manifest](../../tests/report_skill/fixtures/explanations/manifest.json).

| Example | What it exercises |
| --- | --- |
| [Gluon](gluon/report.md) | Race candidate, peer writers and a zero-work configuration that still runs setup/exit |
| [FlashMLA](flashmla/report.md) | Several distinct concerns; supported possible interleaving versus missing model inputs |
| [GQA](gqa/report.md) | Roles, two TMEM objects, reused barrier phases and the release-completion boundary |
| [TMA](tma/report.md) | Short warning account; recorded overlap versus copy-box size; assumption and completion scope |
| [MoE](moe/report.md) | Input-dependent selection branch; analysis stop without inventing a kernel bug |

FMHA, paired-TMEM, paged attention and clean B200 additionally exercise the
script-only result layer. Complexity means interacting roles, objects, phases or
concerns; it is not a warning count or a long file. The corpus retains one pair
per `potential_race` result, so several independent races in one run remain untested.

## Earlier independent generation

In the last pre-migration round, **six native Codex invocations across five cases
passed source/content review**. **Eight Sonnet 5.5 low attempts composed and
checked, but none met all source and presentation criteria**; reported cost was
**$3.7801**. These do not establish stable generation across models.

Each invocation used a frozen skill, raw input/result and matching PTX, with no
previous report, reference answer or follow-up correction. Tools and self-correction
within the invocation were allowed. The six accepted calls used three skill
snapshots; they were not six repetitions of one final version. Native host cost
and exact latency were unavailable, rather than zero.

<details>
<summary><strong>What the eight unsuccessful Sonnet attempts missed</strong></summary>

| Attempt | Content or reading gap |
| --- | --- |
| TMA · first | Truncated samples described as the only warned site; coordinate derivation left unfinished |
| Gluon · first | Configured zero-work path omitted; no participant view |
| Dense FlashMLA · first | Participation inferred from counts; upstream paths untraced |
| TMA · revised skill | Unsupported actor/thread identity; available coordinate definition deferred |
| Gluon · revised skill | Scope appeared too late; concurrency still left as prose |
| Dense FlashMLA · revised skill | Better scope, but long setup code obscured the decisive relation |
| MoE | Excess setup; supplying scores presented as sufficient without establishing frontend support |
| Paged attention | Alignment blamed on the base pointer without resolving the offset |

Three runner setup failures sent no prompt and produced empty streams; they were
not model calls. A separate earlier eight-call round cost $3.3525 and accepted
only the clean B200 case. All failures, reviews, hashes and original archive paths
remain in the [historical ledger](evaluation/history.json). Original reports and
sessions remain unchanged in the private predecessor archive; that material is
not presented as publicly reproducible evidence for these new inputs.

</details>

## Checks and reading review

**247 tests pass; 169/169 current saved results generate and check.**
[Check receipt](evaluation/current.json) records the base commit and outcome counts.

Current component tests cover exact excerpts, artifact identity, saved-result
rendering, reference rejection, truncation, outcome boundaries and complete
report recomposition. The current corpus check generates the result layer from
each saved output, then checks it against that output. It does not call a model
or establish the correctness of an explanation.

The predecessor's GitHub reading review covered five report defaults, GQA's
expanded barrier derivation, FlashMLA in light/dark and narrow layouts, and MoE's
compact branch view. The migration review checks the new public links and changed
assumption/result content. Narrow Mermaid views may still require zooming.
No human comprehension study or general one-shot reliability is claimed.
