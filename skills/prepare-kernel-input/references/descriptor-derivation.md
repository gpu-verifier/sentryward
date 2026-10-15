# Source-based descriptor derivation

Use this procedure when descriptor construction involves templates or composed
layouts. The user supplies a PTX kernel, optionally source locations and scenario
choices; the skill performs the derivation without requiring a GPU.

## Recover the construction

1. Identify the pinned source and concrete specialization corresponding to the
   selected PTX entry. Recover template arguments from the entry and compilation
   recipe; do not substitute a nearby upstream example.
2. Follow host parameter construction to the tensor-map constructor, including
   layout helpers. Substitute the chosen shapes, strides, element types, cluster
   dimensions and compile-time tiles.
3. Evaluate views, packing, swizzles and copy partitioning. Derive each descriptor
   separately, including output descriptors. Record the actual per-copy box,
   rather than using the overall CTA tile.
4. Bind each descriptor to its public parameter or parameter-pack field and
   symbolic memory object. Follow the project's distinct-input-pointer scope;
   retain address transformations derived from a single input within the kernel.
5. Inspect device-side tensor-map replacement and update operations. Separate
   fields fixed by the constructor from fields later replaced using device
   pointer/shape/stride arrays. A provisional host map does not establish the
   final accessed layout. Do not supply device-array elements as launch inputs.

For each descriptor, retain the source expressions and their evaluated values:
element type, global dimensions, byte strides, coordinate order, box dimensions,
element strides and swizzle. Record additional constructor options if they affect
the represented footprint. Keep evidence outside the public input JSON.

## Evaluate complex formulas

Prefer direct algebra or a small local CPU evaluator of the relevant layout
formulas. Generating, running and checking that evaluator are part of this skill,
not extra tasks for the user. Save its source and inputs in the fresh output
directory. Check it against the source expressions, not historical descriptor
values or verifier results.

If faithful evaluation requires the original C++ template machinery, a CPU-only
helper may instantiate the exact specialization and capture the public arguments
before tensor-map encoding. It must not require device allocation, a CUDA context,
GPU execution or target-device queries. Capture arguments rather than decoding an
opaque descriptor. Respect machine-specific execution restrictions; compilation
permission does not authorize running a helper on that machine.

If neither approach is available, record the exact unresolved expression and
dependency. Do not require the user to provide a GPU to finish input preparation.

## Check and deliver

- Check coordinate order, stride units and packed element sizes explicitly.
- Compare each copy's byte footprint with its shared-memory layout and buffer
  extent. Barrier transaction counts can cross-check a derivation, but do not
  uniquely determine a multidimensional transfer box.
- For a multicast copy, `tma_partition` may split the tile across the multicast
  CTAs: each issuer loads one slice and multicasts it, so the box is that slice,
  not the per-CTA shared tile. Check the PTX for a rank-dependent coordinate and
  shared offset. Operands of one kernel can split differently (CUTLASS 2SM
  block-scaled SFB uses a 1SM layout and splits along M; SFA does not).
- A smallest legal box is not evidence of the actual box. Underestimating it can
  hide address overlaps and change transaction accounting. If source construction
  contains a minimum operation, evaluate that expression with the selected inputs.
- Retain a compact per-descriptor derivation and any evaluator artifacts. State
  which fields remain unresolved; omit them only as permitted by the public schema.
  Missing-TMA diagnostics are produced by the verifier frontend, not this skill.
- Check the public schema against actual derived values. Retain a rejected
  source value (for example, a zero outer stride) in the evidence; do not silently
  replace it with an accepted but different layout. Schema support and source
  derivation are separate questions.
- Do not run the verifier or adjust values to obtain a desired verification result.

For batch generation, apply these checks to every descriptor. Unreviewed
descriptors remain unfinished work, rather than evidence that information is absent.
