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
| `tma_descriptors` | Optional array describing tensor-map metadata |
| `branch_decisions` | Optional explicit branch choices; conclusions are conditional on them |
| `assumptions` | Optional Boolean overrides of the [named analysis assumptions](input-json.md#assumptions) |
| `address_invariants` | Optional trusted address constraints; see [forms and semantics](assumptions.md#address-invariants) |
| `data_branch_loop_iterations` | Optional positive iteration bound for loops under `bounded_computation_loops`; default 4; does not itself enable that assumption |

Omitted record arrays are empty. Missing values needed during analysis produce
diagnostics; omission does not give a parameter an arbitrary concrete value.

## Partial-warp blocks

The verifier models a CTA as complete 32-lane warps. A `block_dim` whose thread count is not a
multiple of 32 is verified only when the kernel cannot observe the difference: no barriers
(`bar`, `barrier`, `mbarrier`), warp collectives (`shfl`, `vote`, `match`, `redux`, `activemask`,
`elect`), lane or warp identities (`%laneid`, `%warpid`, `%lanemask_*`), asynchronous or
tensor-core operations, or calls. Such a kernel is verified at a block padded in `z` to whole
warps, in which the padded threads (`%tid.z` at or above the launched `z`) exit at entry and
`%ntid.z` reads the launched `z`; `x` and `y`, and so every thread's indices, are unchanged. The
result carries a `partial_warp_padded` warning with both block shapes. Witness locations are
those of the original PTX. Any other partial-warp kernel is refused with `partial_warp_launch`.

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

Immediate visibility remains a fixed premise of the current model; removing its
configuration switch did not add weak-memory analysis. Do not put
`immediate_visibility` in the `assumptions` object. Existing
`immediate_visibility: true` configurations can omit the field without changing
this premise; configurations requesting `false` are not supported.
The removed `warp_tmem_load_completion: false` override can also be omitted.
A former `true` override must not be silently dropped: its TMEM completion
semantics require review.



| Key | Default |
| --- | --- |
| `duplicate_tma_store_benign` | `true` |
| `representative_election` | `true` |
| `clc_always_fails` | `true` |
| `representative_branches` | `false` |
| `bounded_computation_loops` | `false` |

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
are documented there. Neither assumptions nor address invariants are established
by the verifier merely because they appear in the configuration.


## Related files

Reference arguments derive from this file and [reference_binding.json](reference-binding-json.md);
there is no separate reference_input.json. Verification writes [output.json](output-json.md).
Time and memory budgets are invocation options, not input fields; see the
[CLI and Python API](../README.md#verify-a-kernel).

The schema identifies the persisted format and version. Its spverify prefix is
retained for compatibility; use the exact schema string above.
