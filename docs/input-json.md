# Kernel input (`input.json`)

The kernel input specializes one PTX entry at a launch configuration. It supplies
concrete control values and memory identities, not a tensor-data test vector.
The required schema is `spverify.kernel-input/v1`. Unknown top-level fields are rejected.

## Minimal example

This example describes a kernel with a pointer at parameter 0 and a 32-bit
size at parameter 1. Change the ABI and launch to match the actual PTX.

```json
{
  "schema": "spverify.kernel-input/v1",
  "launch": {"grid_dim": [1, 1, 1], "block_dim": [128, 1, 1]},
  "declarations": [
    {"parameter": 0, "type": "pointer", "space": "global", "object": "x"}
  ],
  "arguments": [
    {"parameter": 1, "type": "u32", "value": 128}
  ]
}
```

## Top-level fields

| Field | Format and meaning |
| --- | --- |
| `schema` | Required format identifier above |
| `entry` | Optional PTX entry name; specify it when selection would be ambiguous |
| `launch` | Required object: `grid_dim` and `block_dim` are three positive integers; optional `cluster_dim` has the same form; optional `dynamic_shared_bytes` is a nonnegative byte count |
| `declarations` | Optional array describing ABI fields, pointer identities and tensor-map identities |
| `arguments` | Optional array assigning concrete scalar values to ABI fields |
| `shape_arrays` | Optional array of concrete, read-only shape tables |
| `pointer_arrays` | Optional array of read-only pointer tables whose entries name input objects |
| `tma_descriptors` | Optional array describing tensor-map metadata |
| `branch_decisions` | Optional explicit branch choices; conclusions are conditional on them |
| `assumptions` | Optional Boolean overrides of the [named analysis assumptions](input-json.md#assumptions) |
| `address_invariants` | Optional trusted address constraints; see [forms and semantics](assumptions.md#address-invariants) |
| `tensor_properties` | Optional trusted element properties of input objects; see [Tensor properties](#tensor-properties) |
| `data_branch_loop_iterations` | Optional positive iteration bound for loops under `bounded_computation_loops`; default 4; does not itself enable that assumption |

Omitted record arrays are empty. Missing values needed during analysis produce
diagnostics; omission does not give a parameter an arbitrary concrete value.

## ABI fields and scalar values

A `parameter` selector is a zero-based PTX parameter index or its nonempty name.
`offset` is a byte offset inside that parameter, default 0, for packed structures.

A declaration accepts `parameter`, `offset`, `type`, `space`, `object`, and `name`.
`type` is required: a scalar type, `pointer`, or `tensor_map`. `space` is for
pointers and must be `global`; `object` names the storage identity for a pointer
or tensor map. `name` gives the field an optional alias.

A direct argument accepts `parameter`, `offset`, `type`, and required `value`.
Scalar types are `u8/u16/u32/u64`, `s8/s16/s32/s64`, `b8/b16/b32/b64`,
`f16/f32/f64`, and `bool`. Integers must fit their width; floating values must be
finite; Boolean values must be JSON booleans. A pointer argument only accepts
`null`, not a host/device address or tensor contents.

Named arguments have exactly `field` and `value`, and refer to a declaration's
`name`. A named field may bind several declarations. Example fragments:

```json
{
  "declarations": [{"parameter": 1, "offset": 8, "type": "u32", "name": "n"}],
  "arguments": [{"field": "n", "value": 128}]
}
```

For a reference scalar derived using `parameter`, use a direct argument record.
The reference resolver does not expand these named aliases; alternatively use
an explicit JSON pointer to the argument's value in the [binding](reference-binding-json.md).

## Shape arrays

Each record has `parameter`, optional `offset` (0), optional `element_type`
(`s32` by default; also `u32`, `s64`, `u64`), and `shapes`: a nonempty rectangular
array of nonnegative integer dimensions. Values must fit the element type.
These bytes are a read-only input contract, not just a hint for the reference.

```json
{
  "shape_arrays": [
    {"parameter": 2, "element_type": "s32", "shapes": [[16, 32], [16, 32]]}
  ]
}
```

## Pointer arrays

Each record has `parameter`, optional `offset` (0), and `entries`: a nonempty array
describing the global table that the 8-byte pointer field at that location points to.
Entry k is the pointer stored at table byte 8k. An entry is `null` (a null pointer)
or has `object` (the named global storage), optional `byte_offset` (0) into it, and
optional `extent_bytes` (a positive bound on the bytes reachable through the entry).
The table bytes are a read-only input contract, like shape arrays: the kernel
must not write them. Loading a whole entry yields a pointer to the named object;
other loads from the table are unknown.

```json
{
  "pointer_arrays": [
    {"parameter": 0, "offset": 1192,
     "entries": [{"object": "A.group0"}, {"object": "A.group1"}]}
  ]
}
```

Distinct objects are distinct allocations. An entry that names an object
already declared in the input, by a pointer declaration or a `tma_descriptors`
record, aliases that object. Entries that share one buffer use one object with
different `byte_offset` values; conclusions then rely on the stated offsets and
extents, which describe one launch.

The entries are trusted, not checked: the verifier does not see the table's
runtime contents. Results record the tables under
`frontend_assumptions.pointer_array_contract`.

## Tensor properties

Each record states one property of the elements of a declared input object,
read as `element_type` (`s32`, `u32`, `s64` or `u64`):

| `property` | Meaning |
| --- | --- |
| `injective` | The elements are pairwise distinct. |
| `lower_bound` | Every element is at least `value`. |
| `upper_bound` | Every element is at most `value`. |

```json
{
  "tensor_properties": [
    {"object": "kv_indices", "element_type": "s32", "property": "lower_bound", "value": 0},
    {"object": "kv_indices", "element_type": "s32", "property": "upper_bound", "value": 1023}
  ]
}
```

`object` names a declared pointer object, a `tma_descriptors` object or a
pointer-array entry object. A bound requires `value`, which must fit
`element_type`; `injective` takes none.
An object's properties use one element type; repeated bounds combine to the
tightest, and contradictory bounds make the input invalid.

Properties are trusted, not checked. An object with properties is read-only
input memory for the analyzed launch: a write to it makes the input invalid.
The element values themselves stay unknown. A load of one whole, aligned
element yields an unknown value within the bounds, and an integer comparison
that the bounds decide (for example a `-1` sentinel test against a nonnegative
index) takes its decided outcome. Loads of other widths are plain unknown
values. Results list the properties under
`frontend_assumptions.tensor_properties`, the source lines decided by them
under `frontend_assumptions.tensor_property_decisions`, and set
`assumption_dependent`.

The address model uses such a loaded value as an offset term. A global pointer
plus an affine offset in loaded elements keeps its allocation while the
offset's range stays within the allocation. The memory checker then treats
each loaded element as a symbol with the stated bounds, and `injective` lets it
conclude that loads of different elements give different values. For example,
CTAs that read-modify-write the state slot `state[idx[cta]]` are race-free when
`idx` is injective and the slot arithmetic cannot overflow. Arithmetic in 32
bits without an upper bound may wrap, so the offset stays unknown.

## Tensor-map descriptors

A descriptor identifies its ABI location with `parameter` and optional `offset`.
It may supply the following metadata; partial metadata can leave a transfer
unsupported if required information is missing.

| Field | Meaning |
| --- | --- |
| `object` | Named global storage |
| `element_type` | Tensor element type, such as `f16`, `bf16`, `f32`, `u8`, `s32` |
| `rank` | Tensor rank, 1 through 5 |
| `base_byte_offset` | Nonnegative byte offset in the storage |
| `global_shape`, `box_shape` | Global dimensions and transfer-box dimensions |
| `coordinate_strides_bytes` | Byte strides for all coordinates |
| `global_strides_bytes` | Alternative outer-dimension strides (rank minus one entries); mutually exclusive with `coordinate_strides_bytes` |
| `element_strides` | Per-dimension element steps |
| `swizzle` | `none`, `32B`, `64B`, `128B`, `128B_atom32B`, or `128B_atom64B` |
| `interleave` | `none`, `16B`, or `32B` |
| `oob_fill` | `zero` or `nan` |

Vector lengths must agree with rank. Packed element formats have additional
stride constraints. Accepted element types also include `s8`, `e4m3`, `e5m2`,
`u16`, `s16`, `u32`, `tf32`, `u64`, `s64`, `f64`, `16u4_align16b`, and `16u4_align8b`.
A syntactically accepted descriptor does not imply every associated PTX operation
or layout is supported.

## Branch choices

Each record has a unique one-based PTX `line` and `taken` (`true` or `false`).
`taken: "random"` instead requires a nonnegative integer `seed`; a Boolean choice
must not include a seed. An optional nonempty `note` explains the choice.

These choices restrict the analyzed executions; they are not proofs of branch
conditions. Representative branches and Bounded computation loops are independent opt-in assumptions, both disabled by default. Neither forces polling waits to complete. See [assumptions](assumptions.md#representative-branches).

For complete real inputs, see [the kernel examples](../examples/results.md).
The production parser is [public_input_parse.cpp](../cpp/src/frontend/ptx/public_input_parse.cpp).

## Assumptions


`input.json` accepts an `assumptions` object with named Boolean overrides.
Unspecified keys use [repository defaults](../config/default-assumptions.json).
Unknown names and non-Boolean values are rejected with distinct diagnostics.

| Key | Default |
| --- | --- |
| `duplicate_tma_store_benign` | `true` |
| `representative_election` | `true` |
| `clc_always_fails` | `true` |
| `representative_branches` | `false` |
| `bounded_computation_loops` | `false` |
| `order_independent_atomics` | `false` |

For example, this fragment opts into representative data-dependent control:

```json
{
  "assumptions": {"representative_branches": true, "bounded_computation_loops": true},
  "data_branch_loop_iterations": 4
}
```

See [assumptions.md](assumptions.md) for each assumption's exact scope and
limitations. `address_invariants` are separate trusted constraints, not Boolean
assumptions; their [forms and examples](assumptions.md#address-invariants)
are documented there. The verifier does not establish assumptions, address
invariants, `tensor_properties`, `pointer_arrays` or `shape_arrays` merely
because they appear in the configuration; it trusts them as stated.


## Related files

Reference arguments derive from this file and [reference_binding.json](reference-binding-json.md);
there is no separate reference_input.json. Verification writes [output.json](output-json.md).
Time and memory budgets are invocation options, not input fields; see the
[CLI and Python API](../README.md#verify-a-kernel).

The schema identifies the persisted format and version. Its spverify prefix is
retained for compatibility; use the exact schema string above.
