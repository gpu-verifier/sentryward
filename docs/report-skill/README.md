# Kernel reports: finding → code → evidence

One English `report.md` helps a GPU developer understand a kernel finding without
first learning the verifier's internal representation. **Verifier result**
presents the saved outcome. **Code explanation** follows the relevant operations
and participants. **Evidence** keeps their supporting records available to check.

## Start with a finding

| Report | What to follow |
| --- | --- |
| [Gluon · race candidate](gluon/report.md) | Peer stores to a shared dispatch word, and why the following barrier does not order those stores |
| [FlashMLA · several concerns](flashmla/report.md) | A possible cluster-barrier protocol violation, missing tensor-map captures and input-dependent copy control |
| [GQA · pipeline lifetime](gqa/report.md) | Two TMEM objects, reader completion, a half-warp bypass, two barrier rounds and release |

FlashMLA's participant view is a **source-derived possible interleaving**, not a
recorded GPU failure. Gluon's retained pair remains a **race candidate**. GQA
separates a missing model order from the source path worth investigating.

Compare the shorter [TMA warning](tma/report.md) and [MoE analysis stop](moe/report.md).
They use the same reading boundaries, with a local explanation instead of a full
pipeline account. More warnings alone do not make a case complex.

<details>
<summary><strong>Other outcomes and regression cases</strong></summary>

- [FMHA](fmha/report.md): a rejected synchronization wait and a truncated verifier witness.
- [Two-CTA backward attention](paired-tmem/report.md): local and paired-allocation lifetime requirements.
- [Paged attention](paged-attention/report.md): memory warnings alongside an incomplete numerical check.
- [B200 GEMM](b200/report.md): short clean-result report, with no forced code investigation.

These four demonstrate the deterministic result layer. They do not have a
model-written explanation. Current saved `potential_race` results retain one
pair each; this collection does not test several independent races in one run.

</details>

## Read or generate one report

The opening links go to the original PTX, configuration and verifier output.
Instruction names open the PTX manual; snippet captions open exact source lines.
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
  --report build/skill-output/KERNEL/report.md
```

An agent adds an internal explanation plan through `--draft`. The user does not
supply or read that plan. The script still reads outcomes, counts and diagnostics
directly from the saved artifacts; the plan cannot replace them. References are
checked, but a valid reference does not prove the interpretation.

## What has been checked

All examples use public artifacts pinned to `94a6b26`. The five explanations
originate in independent trials on the predecessor repository and were adapted
and reviewed for these artifacts. **This migration is not a new one-shot trial.**
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
