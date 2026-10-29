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
Allocation IDs identify the expression producer's storage; they are not CUDA
pointer values and should not be invented independently of that producer.

A single dense output uses `outputs.kind: "dense_tensor"`, with
`left_allocation`, `right_allocation`, `left_offset`, `right_offset`,
`element_bytes`, `shape`, `left_strides_bytes`, and `right_strides_bytes`.
Multiple outputs use `kind: "dense_tensors"` and a `tensors` array containing
these records. Output coverage must be complete for an overall `verified` result.

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
