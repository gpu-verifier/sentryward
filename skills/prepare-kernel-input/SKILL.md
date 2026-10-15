---
name: prepare-kernel-input
description: Generate a public sentryward input JSON for a PTX kernel, deriving launch parameters, ABI bindings, and TMA descriptors from a chosen scenario and available source. Use for user input preparation, not verifier execution or frontend repair.
---

# Prepare kernel input

## Deliverable and scope

Generate a new `spverify.kernel-input/v1` JSON file for the selected PTX entry.
Support both “use defaults” and partial user choices such as a particular shape.
The user may supply just a PTX file, optionally a source location, entry, shape,
or launch choices. Source absence may limit derivation; do not require a GPU.
Existing `examples/` configs are read-only examples, not files to update.
Write to the requested new destination, or a fresh `build/skill-output/` directory.
Do not overwrite an existing config or edit repository examples.

Completion requires checked parameter bindings and evidence for derived values,
including launch-time descriptor fields. A complete configuration supplies the
permitted launch inputs for its scenario; it need not determine values later
loaded from device arrays. Parser success alone is not completion. Distinguish
unfinished derivation, unavailable launch information, and an input-interface
limitation from a fully configured kernel with dynamic device behavior.
Do not run specialization, frontend extraction, or verification. Config generation
does not promise that the current development frontend supports this kernel.

Typical requests:

- "Use prepare-kernel-input for `kernel.ptx`; inspect its source links, choose a
  legal default scenario, and write a new `input.json`. Do not choose branches."
- "Prepare input for this entry with M=256, N=128, K=512. Source is at this
  revision/path. Keep other legal defaults and write to this new destination."

## Determine the input contract

Inspect the selected PTX entry, parameter declarations, and launch requirements.
If multiple entries are present without a user selection, ask which entry to use.
Start with the public source links in the PTX header. Read the pinned kernel and
host sources, following their layout helpers and descriptor constructors, to
recover parameter-pack types, offsets, launch and descriptor construction.

Use the current public parser as the field reference:

- `cpp/src/frontend/ptx/public_input_parse.cpp` and
  `cpp/include/sentryward/frontend/ptx/public_input.h`;
- `skills/prepare-kernel-input/scripts/input_audit/input/parse.py` and `tma.py` for the static audit's
  Python representation, not a production frontend.

The public input has `schema`, optional `entry`, `launch`, typed `arguments`,
optional `declarations`, `shape_arrays`, `tma_descriptors`, `branch_decisions`, and `assumptions`.
Do not confuse it with internal frontend configs or native source-program JSON.
Confirm accepted fields and types against this checkout rather than importing
historical internal fields from an example.

## Select a scenario

Use this precedence:

1. The user's explicit choices.
2. A matching existing kernel example's default values, after checking the PTX
   entry/version and compatibility with the user's choices.
3. A legal, moderate-sized scenario supported by the source/host example and
   kernel constraints.

Explain selected defaults briefly when handing over the file. A default is one
analysis scenario, not a statement that all shapes or paths are covered. Derive
dependent strides, launch dimensions, scalar parameters, and descriptor fields
consistently; do not change only the visible shape while retaining incompatible
derived values. If choices contradict a compile-time constraint, explain the
conflict and ask for a valid choice instead of silently changing the user's input.

## Preserve the public input boundary

- Supply typed scalar values and a distinct symbolic object for each non-null
  input pointer parameter, including pointer fields in parameter packs. The
  project scope assumes those inputs do not overlap; do not reconstruct host
  aliases between them or require an object-offset extension. Pointer arithmetic
  derived from one input inside PTX remains the frontend's responsibility.
  Do not invent physical GPU addresses.
- Optional ABI declarations describe parameter types, field locations, and object
  identities. Recover offsets from evidence; unknown necessary fields require a
  question, not guessed bytes or a fabricated all-zero parameter pack.
- Numerical tensor contents, page-table entries, register results, and per-
  instruction execution contracts are not user input. Shape arrays describe
  shapes only; supplying `shape_arrays` opts into fixed, read-only configured
  shape bytes throughout the analyzed execution. This is part of the existing
  input contract, not a new configurable assumption. Algorithm choices such as
  elected lane and analysis step limits do not belong in this config.
- TMA descriptors are optional structured input. Include supported fields when
  supplied or reliably derived from host construction and the selected scenario.
  Do not invent a layout to make analysis succeed. If optional information is
  unavailable, omit it and state that descriptor-dependent analysis may be blocked.
- Omit branch overrides by default, including overrides found in reference
  examples. Include them only when explicitly authorized for this kernel/scenario,
  either as exact choices or through the optional pattern-based mode below.
- JSON has no comments. Use only supported note fields; do not introduce arbitrary
  metadata keys to explain defaults.

## Check parameter coverage

Trace source argument construction and PTX parameter accesses. Read
[Parameter coverage](references/parameter-coverage.md) for packed ABIs and
completeness checking. Bind consumed fields, including shape/stride scalars,
symbolic pointers, and descriptor references. A scalar declaration supplies a
type, not a value; an opaque map declaration does not supply its geometry.
Unused opaque bytes need no fabricated values. Indirect parameter accesses and
helper calls require review beyond direct-load coverage.
Before supplying shape arrays, or for grouped kernels, pointer arrays, workspace
suballocations, or device-updated tensor maps, read
[Grouped inputs](references/grouped-inputs.md).

## Check TMA descriptor evidence

When source is available, deriving descriptor values is part of this task.
Do not omit a descriptor merely because it has not yet been reviewed, because
its construction is in a helper, or because template/layout evaluation requires
more work. Do not replace a batch derivation with removal of historical values.

Follow the selected entry's concrete instantiation from its type arguments or
compilation recipe to the host tensor-map constructor. Substitute the selected
shape, strides, dtype, CTA/cluster tile, schedule and stage count. Follow layout
composition, views, swizzles, packing and partitioning to the actual copy atom.
Derive values from source, template arguments and layout formulas first. Input
preparation must not require a GPU or execution on the target machine. For complex
layouts, read [Source-based descriptor derivation](references/descriptor-derivation.md).
Any auxiliary evaluator belongs to this skill's workflow: generate it, validate
its correspondence to the source, and retain its inputs and derivation evidence.

For each supplied descriptor, establish its parameter/object binding, element
type, global dimensions and strides, coordinate order, transfer box, element
strides, and shared-memory swizzle from user-supplied information or source/PTX
evidence for the selected specialization. An existing config is a reference,
not evidence that its derived fields are correct.

Inspect device-side descriptor updates. A provisional host map does not establish
the final accessed layout. Preserve proven static fields; do not fill fields
replaced from device pointer/stride arrays with guessed constants.

Derive the box from the actual copy tile after CTA, warp-group, and pipeline-stage
partitioning. Check its byte count and swizzle against that copy's shared layout
and buffer extent. Input and output descriptors may use different tiles; derive
them independently. A full CTA tile must not replace a per-group or per-stage tile.
Do not substitute the smallest legal tile for an unknown actual tile: this can
underestimate memory footprints and transaction bytes. Use a minimum only when
the selected specialization's construction proves that it is the actual tile.
An explicitly requested hypothetical tile is a conditional scenario, not a
source-derived descriptor; disclose that distinction and any source inconsistency.

Only after tracing the construction, if a field depends on unavailable external
information, a missing source/dependency, or a genuinely unresolved derivation,
identify that specific dependency and the steps attempted. Unfinished review is
unfinished work, not missing kernel information. If a field cannot be established,
omit it when the public schema permits;
otherwise omit the incomplete descriptor. Missing-information diagnostics belong
to the verifier frontend when analysis needs those fields; the skill does not
emit a verifier status or simulate that diagnostic. Do not fill
gaps from historical configs, guess defaults, or tune footprints to remove races.
Keep this explanation outside the JSON unless a supported note field applies.

At handoff, give a compact source reference and derivation for each supplied
transfer box and swizzle, or identify them as explicitly user-supplied. For batch
generation, perform this check for every descriptor in every output; checking a
representative kernel does not validate the others. Disclose omitted optional
fields at handoff without claiming source validation or a verifier result.

## Optional pattern-based branch choices

Branch selection is opt-in. Without authorization, omit `branch_decisions`.
An explicit request to select recognized attention correction paths enables that
mode for the requested kernels. Read [Branch choices](references/branch-choices.md)
when enabled, and establish each branch's function and polarity in this PTX.
Disclose every supplied choice as analysis scope.

## Validate and hand over

Read `config/assumption-pool.json` for accepted keys, `config/default-assumptions.json`
for defaults, and `docs/assumptions.md` for meanings. Include effective Boolean switches
in the generated JSON's `assumptions` object, applying user overrides. The public
interface also permits omitted keys, which use repository defaults.
CLI `--assume` / `--no-assume` overrides JSON. Do not change repository defaults,
copy historical choices, or automatically enable assumptions to bypass unsupported
results. Briefly identify effective choices at handoff; do not run analysis.

Run the input-only audit from the repository root using Python 3.10 or newer.
Use the project environment created by `make venv`, or an equivalent supported
interpreter; the system `python3` may be older:

```sh
build/venv/bin/python skills/prepare-kernel-input/scripts/audit_input.py \
  --ptx path/to/kernel.ptx --input path/to/input.json \
  --output path/to/new-input-audit.json
```

Review gaps and manual-review items using source evidence. The helper does not
prove source correctness, reachability, or complete descriptor dataflow. A covered
read is not evidence that its supplied value is right. Do not invoke the frontend
or verifier to tune the input until it passes.

Keep the scenario, source revision/locations, derived bindings/layouts, and any
CPU evaluators beside the new artifact. Apply these checks to every batch member.

| Preparation outcome | Required handoff |
|---|---|
| Checked input for the selected scenario | JSON, derivation evidence, resolved audit items, assumptions and branch scope |
| Checked input with dynamic device dependencies | The same checked launch inputs, plus the device loads/updates whose values remain unknown; frontend support is a separate question |
| Derivation unfinished | Identify the remaining expression or ABI field; continue working rather than label it unavailable |
| Information unavailable or not representable | Exact field, dependency or interface restriction; legal partial JSON only when omissions are permitted |

Do not describe partial inputs as fully bound or report verifier success from this
audit. Ask only for missing information the user may legitimately supply; never
request prohibited device data as a workaround. Hand over concise results with
file links, selected defaults, user overrides, and explicit limits. When supplying
shape arrays, disclose the fixed, read-only shape scope and that no verifier run
was performed.

The audit implementation is bundled in scripts/input_audit/ and uses only the
Python standard library. It reports configured_assumptions exactly as supplied;
it checks their syntax but does not validate semantic names or resolve defaults.
Those checks remain with the verifier. The audit does not import sentryward or
load its configuration; the configuration references above guide input preparation.
