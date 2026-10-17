# Kernel reports: finding → code → evidence

One English `report.md` helps a GPU developer understand a kernel finding without
first learning the verifier's internal representation. **Verifier result**
presents the saved outcome. **Code explanation** follows the relevant operations
and participants. **Evidence** keeps their supporting records available to check.

## Five different findings

Scale: **〈PTX file lines · output size · saved analysis time〉**. Time is verifier
analysis time, not kernel runtime; a PTX file may contain several kernel entries.

| Report | Scale | Saved finding |
| --- | --- | --- |
| [MoE · analysis stop](moe/report.md) | 〈523 · 9.2 KB · 0.020 s〉 | One unsupported control predicate |
| [Sparse GEMM · read/write candidate](sparse-gemm/report.md) | 〈7,362 · 26.4 KB · 27.23 s〉 | One retained TMEM write/read pair |
| [TMA GEMM · warnings](tma/report.md) | 〈4,035 · 1.04 MB · 7.91 s〉 | Verified; 341,248 warnings, sampled |
| [FMHA · synchronization failure](fmha/report.md) | 〈360,487 · 30.4 KB · 7.44 s〉 | Rejected barrier-phase requirement; truncated model replay |
| [FlashMLA · several concerns](flashmla/report.md) | 〈5,602 · 168 KB · 1.58 s〉 | 87 diagnostics across 4 categories |

These span a local analysis stop, an access conflict, warning classification,
barrier reuse and several coexisting concerns. Each follows **finding → relevant
code → relationship → evidence**; its depth follows the relationships involved.
The sparse GEMM pair remains a candidate. FMHA's saved model replay and FlashMLA's
source-derived possible interleaving are distinct from an observed GPU trace.

<details>
<summary><strong>Other outcomes and regression cases</strong></summary>

- [Gluon](gluon/report.md): peer writes to a shared dispatch word, followed by a barrier.
- [GQA](gqa/report.md): two TMEM objects, reader completion, barrier reuse and release.
- [Two-CTA backward attention](paired-tmem/report.md): local and paired-allocation lifetime requirements.
- [Paged attention](paged-attention/report.md): memory warnings alongside an incomplete numerical check.
- [B200 GEMM](b200/report.md): short clean-result report, with no forced code investigation.

Gluon and GQA include source explanations. Two-CTA backward attention, paged
attention and B200 demonstrate the deterministic result layer; B200 needs no
forced mechanism investigation. Current saved `potential_race` results retain
one pair each; this collection does not test several independent races in one run.

</details>

## Read or generate one report

The opening links go to the original PTX, configuration and verifier output.
Instruction names open the PTX manual; snippet captions open exact source lines.
GitHub cannot display the full FMHA PTX file; its **raw** link opens the original
file, and the report retains exact excerpts with physical line numbers.
Diagrams distinguish address overlap, dependencies, control flow and possible
interleavings. Conditions that change the conclusion stay visible. Expand a named
question for its derivation and exact evidence.

Give the [skill](../../skills/generate-kernel-report/SKILL.md), saved `input.json`,
`output.json` and matching PTX to an agent. Only `report.md` belongs in the delivery
directory. To generate the result layer without a model:

```sh
python3 skills/generate-kernel-report/scripts/report_facts.py \
  --input examples/KERNEL/input.json \
  --result examples/KERNEL/output.json \
  --ptx corpus/KERNEL.ptx \
  --report build/skill-output/KERNEL/report.md --result-panel
```

An agent adds an internal explanation plan through `--draft`. The user does not
supply or read that plan. The script still reads outcomes, counts and diagnostics
directly from the saved artifacts; the plan cannot replace them. References are
checked, but a valid reference does not prove the interpretation.

The skill and published examples use `--result-panel` to place saved findings and
scope in a GitHub panel above the source explanation. Use the option on both
composer and checker; it works for result-only reports too.

## What has been checked

All examples use public artifacts pinned to `94a6b26`. Five explanations
originate in independent trials on the predecessor repository and were adapted
for these artifacts. FMHA and sparse GEMM are local development additions.
**Recomposition and source review are not new one-shot trials.**
[Evaluation](evaluation.md) separates the earlier generation results from current
recomposition and corpus checks, including unsuccessful attempts.

[Reproduction manifests](../../tests/report_skill/fixtures/explanations/manifest.json)
identify every input, plan and report hash. Run the checks with:

```sh
python3 -m unittest discover -s tests/report_skill -v
```

For reading feedback, pick one finding: what happened, which object and participants
are involved, what condition matters, and which evidence would you open to check it?
Point to the passage where that connection becomes unclear.
