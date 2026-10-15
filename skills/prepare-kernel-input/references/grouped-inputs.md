# Grouped kernels and input completeness

Use this for kernels that select tensors through pointer arrays, update tensor
maps on device, or receive multiple pointers into one workspace allocation.

## Separate launch inputs from device contents

Inventory the host-created parameter pack and its constructor. Bind ordinary
scalars, pointer identities, allowed shape arrays, and initial tensor-map fields.
Recover nested epilogue/scheduler fields as carefully as the mainloop fields.
An unknown device dependency does not excuse leaving an unrelated launch scalar
unreviewed.

For nested epilogues, follow `Arguments -> to_underlying_arguments -> Params`.
The public alpha/beta defaults may become nested callback scalars, optional
scalar pointers and broadcast strides. Evaluate that conversion for the chosen
scenario. Default-constructing `Params` only to obtain field offsets does not
establish its runtime values. Inspect each pointer's role: an optional scalar
override can legitimately be null; a required workspace pointer needs an object.

For persistent schedulers, distinguish the selected scheduling budget from a
hardware measurement. When the source permits a positive host `sm_count`, a
CPU-only derivation may choose it consistently with the launch and evaluate the
scheduler formulas. Do not execute a fallback device query. Record that budget
as a scenario choice rather than claiming it describes the user's GPU.

For `p = ptrs[group]`, declaring the object `ptrs` supplies the array's identity,
not its pointer elements. Do not request or invent those elements. Distinct
pointer-array objects do not imply distinct pointees. If all permitted launch
inputs are configured, unknown array elements are a later analysis limitation,
not an incomplete user configuration.

For device-updated tensor maps, distinguish the initial host map from its later
states. Follow each replacement to a constant, launch parameter, or device load.
Supply the proven initial fields; retain the replacement dependency as evidence.
A placeholder initial address is not the final accessed tensor. Check how the
current public schema represents an address-free initial map before claiming
that map is fully configured.

If the source initializes a map with a null data address and replaces that address
on device before use, preserve the known geometry and omit its optional `object`.
Record the initial null and the replacement load in the evidence. Do not assign
a fictional final tensor object. This is a checked, address-free initial map;
analysis of its later device-derived address may remain unsupported.

## Bind fixed shape metadata

Supplying `shape_arrays` declares that the configured shape bytes remain fixed
and read-only throughout the analyzed execution. The existing public schema is
unchanged; do not add an assumption switch or a separate execution contract.
A known write to those bytes is an input-contract conflict. An unknown-address
write alone does not block preservation of the configured shape values under
this contract. Shape loads and other memory accesses still participate in normal
race reasoning; fixed values do not remove their memory-access events.

Map the logical rows through the source/ABI parameter selector, the pointer
field's byte `offset`, and proven element layout. The offset locates the pointer
in the parameter pack, not bytes within its pointee. The supported representation
is contiguous, equal-rank rows of nonnegative `s32`, `u32`, `s64`, or `u64`
dimensions. Establish row and dimension order, element width, and absence of
inter-element or inter-row padding from source/ABI evidence; do not guess them.
Distinguish an unfinished layout derivation from a proven but unsupported
representation. Neither permits PTX register bindings or loop-trip-count
overrides as a substitute.

This contract supplies only the configured shape bytes. It does not supply
pointer-array elements or imply that output pointees are mutually disjoint.
Disclose this fixed-shape scope with the binding/layout evidence at handoff;
input preparation does not run the verifier or prove the kernel race-free.

## Apply the input-pointer scope

Follow the host workspace partitioning, including alignment and callback storage:

```text
epilogue_base = workspace + epilogue_offset
mainloop_base = workspace + mainloop_offset
scheduler_base = workspace + scheduler_offset
```

When these pointers are passed as separate input parameters, assign each a
distinct symbolic object. The project's scope assumes different input pointer
parameters do not overlap, including fields in a parameter pack. The host
relationship above is outside that scope and is not a preparation blocker.
Do not add a configurable assumption or an object-offset interface for it.

The current public pointer declaration's `offset` locates the pointer field
inside the parameter pack. It does not locate the pointee inside an allocation.
Likewise a TMA descriptor's `base_byte_offset` describes its data view, not the
storage address of a generic workspace pointer.

If PTX itself derives several addresses from one input pointer, retain that one
input object. The frontend handles those address computations. The independent-
input scope does not make addresses derived from one pointer disjoint, nor does
it determine elements loaded from a pointer array.

## Finish the preparation

Resolve every parameter-audit item with a binding or specific source evidence.
For example, an unused vector component needs def-use evidence, not a padding
value. A device dependency needs its source load/update, not a blanket claim
that grouped kernels are dynamic.

A `selp` may select between normal and fallback parameter-resident maps. Check
both possible map ranges against their supplied descriptors; no branch choice
is needed to establish coverage. For wide loads crossing padding, check each
unbound destination register's uses within the selected entry, not across other
entries that reuse register names. Keep these resolutions separate from the raw
audit, which may still report `binding_gaps` or `needs_review`.

Report separately: checked launch inputs; device values intentionally unknown;
unavailable external launch information; unsupported input representations; and
derivations still unfinished. Only the last category is preparation work that
the agent should continue on its own. Do not claim completion while a required
launch field remains unbound or unrepresentable.
