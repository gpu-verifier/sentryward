# Reference binding (`reference_binding.json`)

The binding identifies corresponding symbolic inputs and outputs and supplies
metadata for capturing `reference(...)`. Its schema remains
`spverify.equivalence-mapping.v1`. In the public pipeline, **left is the Python
reference and right is the PTX kernel**.

Pass the file with `--reference-binding` or `reference_binding=`. If omitted,
Sentryward looks beside the reference file, then in its parent directory.

## Mapping fields

| Field | Meaning |
| --- | --- |
| `schema` | `spverify.equivalence-mapping.v1` |
| `left_inputs`, `right_inputs` | Input identities and layouts for the two sides |
| `outputs` | Output correspondence to prove |
| `output_value_semantics` | Existing reference bindings use `ideal_real` |
| `reference` | Capture argument and optional output metadata, described below |
| `provenance` | Optional source metadata and recorded fixed-input scope, described below |
| `checker_context` | Optional explicit premises for the equivalence engine; advanced checker input, not inferred from a kernel name or previous proof |

Each input side has `allocations`, mapping side-specific allocation IDs to shared
logical names; `parameters`, mapping symbolic parameter names; and `layouts`,
indexed by side-specific allocation ID. A layout describes `source_offset`,
`target_offset`, `element_bytes`, `shape`, `source_strides_bytes`, and
`target_strides_bytes`. Offsets and strides are bytes. The target is the common
logical layout used for comparison. The two sides must agree on that layout.
Source strides need not be dense: sorted by size, each stride must be a positive
multiple of `element_bytes` and at least the byte extent of all smaller-stride
axes, so that every byte offset denotes at most one element. For example, shape
`[7, 4, 128]` with 2-byte elements and strides `[1040, 256, 2]` pads each
1024-byte block by 16 bytes. Other layouts are refused.
Allocation IDs identify the expression producer's storage; they are not CUDA
pointer values and should not be invented independently of that producer.

A single dense output uses `outputs.kind: "dense_tensor"`, with
`left_allocation`, `right_allocation`, `left_offset`, `right_offset`,
`element_bytes`, `shape`, `left_strides_bytes`, and `right_strides_bytes`.
Multiple outputs use `kind: "dense_tensors"` and a `tensors` array containing
these records. Output coverage must be complete for an overall `verified` result.
Only the bytes a record compares matter: stores into a mapped allocation are
analyzed at those bytes only, so padding, other fields and bytes outside the
tensor may hold any value, including a never-written one. Each compared byte
must be written by the kernel.

See the complete [WeightNorm binding](../examples/pytorch__aten_WeightNorm__sm_100a/reference_binding.json)
for two tensor inputs and two outputs. It pairs with this [kernel input](../examples/pytorch__aten_WeightNorm__sm_100a/input.json)
and [reference function](../examples/pytorch__aten_WeightNorm__sm_100a/reference.py).

## Reference arguments

`reference.arguments` is a required array in Python function parameter order.
Each entry has a unique nonempty `name` and one of these forms:

| Form | Fields and derivation |
| --- | --- |
| Tensor | `name`, `allocation`, `dtype`; optional `shape` and byte `offset`. Shape defaults to the matching `left_inputs.layouts` shape; offset defaults to its `source_offset` or 0. Supply metadata that cannot be derived explicitly. Dtype is never guessed from byte width. |
| Kernel scalar | `name`, `parameter`, optional `parameter_offset` (0). Resolves exactly one concrete kernel `arguments` record at that parameter and byte offset. |
| Constant scalar | `name`, `value`. The value is passed to the reference. |

This fragment illustrates all three forms; the surrounding mapping must describe
its allocations and outputs:

```json
{
  "reference": {
    "arguments": [
      {"name": "x", "allocation": "6:global1:x", "dtype": "float32"},
      {"name": "n", "parameter": 1},
      {"name": "epsilon", "value": 0.00001}
    ]
  }
}
```

The scalar resolver reads direct `arguments` records; it does not match an
integer selector to a differently spelled PTX parameter name, or expand
`field` aliases. Missing or ambiguous values are errors. Use a direct argument
with the same selector, or a JSON pointer to its concrete value.

Within reference argument/output metadata, an object with exactly one `input`
key is a JSON pointer into the kernel input. Resolution is recursive. For example:

```json
{"name": "grid_x", "value": {"input": "/launch/grid_dim/0"}}
```

Array indices are zero-based. Pointer tokens escape `~` as `~0` and `/` as `~1`.
A missing path is an error. This mechanism is not a general substitution language
for the entire native mapping: use concrete layout and output-mapping values.

## Reference outputs and guarantees

Capture output metadata normally derives from the dense output mapping's
`left_allocation`, `shape`, and `left_offset`. If it cannot be derived, provide
`reference.outputs`, an array of `allocation`, `shape`, and optional byte `offset`
records in reference return order. There is no separately maintained reference
input file.

Layouts describe correspondence; they do not assert that tensor elements are
nonzero, finite, positive, or in a particular range. Explicit checker premises
must come from a justified input contract. Equivalence means equal domains and
equal values on that domain, so matching undefined cases do not by themselves
invalidate a proof. See [equivalence checking](reference-binding-json.md#equivalence-contract).

The capture derivation is implemented in
[_reference_binding.py](../sentryward/_reference_binding.py).

## Equivalence contract

With a Python reference, Sentryward runs the synchronization check, including
race freedom, before establishing numerical equivalence. The reference defines a partial
function: a proof requires the PTX and reference to have the same domain and
equal outputs on that domain. Total definedness on every possible tensor input
is a separate property. An informational `total_definedness_not_proven`
diagnostic can therefore coexist with `verified`.

The current numerical model uses ideal real arithmetic with storage and index
interpretation. It does not prove bit-for-bit floating-point equality. Input
layouts do not imply nonzero values or other mathematical premises. A failed
proof is `equivalence_not_proven`, not a counterexample.

Operations on exact constants (literals, fixed launch parameters and values
computed only from them) are evaluated exactly as the instruction rounds, not as
ideal reals: binary32/binary64 `add`, `sub`, `mul`, `div` and `fma` with
round-to-nearest-even (explicit `.rn` or PTX's default rounding; `.ftz`
flushes binary32 subnormals, and a `.ftz` result that may round up to the
smallest normal magnitude from below stays ideal, since its flush depends on
when the hardware detects tininess), and `cvt` to `f16`, `bf16`, `f32` or
`f64` with `.rn`, `.rz`, `.rm` or `.rp`. A default-rounding `add`/`sub` with a
default-rounding `mul` operand stays ideal: PTX may contract the pair into an
`fma`. Saturating, flushing or other conversions, approximate instructions and
non-finite results stay ideal, as do all operations with a non-constant operand.
Reference literals of type `f32` and `f64` denote their nearest binary32/binary64
value. The numerical model also uses one range premise, configurable as the
assumption `floating_values_in_range` (enabled by default;
[assumptions](assumptions.md#floating-values-stay-in-range)).

Reduction stores combine into memory instead of overwriting it.
`cp.reduce.async.bulk` combines each element of its shared-memory payload into
the global element at the same offset. The compared value of such an element is
its prior contents combined with every contribution. The prior contents are the
last plain store, or else the initial input value. A floating-point `add` is an
ideal real sum, so the order in which the contributions arrive does not matter.
An integer `add` is modulo 2^w. Integer `min` and `max` are exact. The value
analysis models no other combination. Building the value view accepts `add` on
`f32`, `u32`, `s32` and `u64`, and `min` and `max` on `u32`, `s32`, `u64` and
`s64`; it refuses any other operator or type (`reduction_operation_unsupported`).
A floating `min` or `max`, or another operator, that reaches expression
construction is refused there (`scalar_fold_operation_unsupported`). The value
analysis also refuses:

- a read of an element that a reduction combined into (`reduced_read_unsupported`);
- different operators on one element (`mixed_reduction_operations`);
- a reduction or store that overlaps a reduced element only in part
  (`reduction_partial_overlap_unsupported`);
- a reduction at a symbolic position (`symbolic_reduction_unsupported`);
- a reduction into an allocation without initial contents before any store
  covers the element (`uninitialized_memory_read`).

See [output.json](output-json.md) for check statuses and diagnostics.

## Reference provenance


The optional `provenance` object inside this binding supplies source and fixed-input
metadata. There is no separate provenance file or CLI argument. It is not needed to describe ordinary reference tensors
or scalar arguments; those belong in [reference_binding.json](reference-binding-json.md).

Example bindings use the following provenance fields:

| Field | Meaning |
| --- | --- |
| `source`, `upstream` | Source description and upstream source URLs |
| `sha256`, `optimized_input_sha256` | Recorded input digest (existing corpus files retain both names) |
| `ptx_sha256` | Recorded PTX file digest |
| `scope` | Human-readable description of the recorded specialization |
| `scope_branch_choices` | Recorded branch choices |
| `scope_shape_arrays` | Recorded shape arrays |

See a [complete binding example](../examples/pytorch__aten_WeightNorm__sm_100a/reference_binding.json#provenance).
These are provenance records, not an authorization to assume arbitrary tensor
values. The optional fixed-input context adapter checks source hashes and scope
before using its supported concrete input facts. Its context receipt records the
actual source identities. A source URL or prose `scope` alone is not a checked
numerical premise. Changing an input requires reviewing the associated
provenance rather than copying a successful historical receipt.


## Reference function requirements

Reference capture needs the optional PyTorch dependency, but no GPU or concrete
tensor contents. Pass parameters and buffers explicitly as function arguments.
Input mutation, implicit module state, data-dependent control and unsupported
PyTorch operations can prevent capture. Outputs must be tensors with at least
one dimension. The reference function executes as Python code during capture.
