# Frozen report examples

These fixtures preserve historical input, verifier results and version-1 explanation
plans for TMA, MoE and TMEM. They are regression inputs, not per-kernel deliverables.

Each case contains `input.json`, `output.json`, `identity.json` and `draft.json`.
The [manifest](manifest.json) identifies the corresponding corpus PTX.
PTX is reused from `corpus/`, not duplicated here. The regression test checks
its SHA-256 and byte count, along with the frozen input and result, before running
the composer. Artifact hashes identify supplied files; they do not establish
which verifier revision executed the run.

The saved results are historical review snapshots,
not the newer `examples/*/output.json` files on main. Do not substitute a newer
result without reviewing the explanation plan and updating the report together.
The old `draft.json` files are historical evidence; the current version-2 composer
does not consume them. The current tests generate a base report from each frozen
input/result pair and verify preserved diagnostics, artifact identities and the
single-file output contract. They do not claim fresh explanation generation or
byte-identical reproduction of the historical reports.

Run `python3 -m unittest discover -s tests/report_skill -v` from the repository
root. It verifies the base reports and relevant component invariants.
