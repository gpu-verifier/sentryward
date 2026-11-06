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

Reference literals of type `f32` and `f64` denote their nearest binary32/binary64
value. The numerical model also uses one range premise, configurable as the
assumption `floating_values_in_range` (enabled by default;
[assumptions](assumptions.md#floating-values-stay-in-range)).

**PTX values: constant propagation in the instruction's own type.** Where PTX
semantics fix a floating value, the comparison uses that value, computed as the
instruction computes it in its own type, not as an ideal real. Every other
floating value is an ideal real. The propagation has two parts.

- **Exact constants, at lowering.** An operation whose operands are all exact
  constants (literals, fixed launch parameters and values computed only from
  them) has its rounded result: binary32/binary64 `add`, `sub` and `mul` with
  round-to-nearest-even (explicit `.rn` or PTX's default rounding), `div` and
  `fma` with explicit `.rn` (`.ftz` flushes binary32 subnormals, and a `.ftz`
  result that may round up to the smallest normal magnitude from below stays
  ideal, since its flush depends on when the hardware detects tininess), and
  `cvt` to `f16`, `bf16`, `f32` or `f64` with `.rn`, `.rz`, `.rm` or `.rp`. A
  non-finite result is left to the import. PTX may fuse a default-rounding
  `mul` into a default-rounding `add`/`sub` that reads it (contraction below),
  so such an `add`/`sub` is not folded even when its operands are constants,
  and when the `mul` itself was folded the `add`/`sub` is no instruction image
  for the import: the fold hides the product the hardware may fuse. Finite
  constant folding is otherwise unchanged. Saturating, flushing or other
  conversions and approximate instructions stay ideal.
- **Value sets, at import.** Each floating node that is the image of one PTX
  instruction is evaluated again when the comparison imports it, from its direct
  operands. This extends constant propagation in three ways: a `selp` whose
  condition the current scope decided is its chosen operand; an operand that is
  not one constant is a set of values (an interval of the extended reals plus a
  may-be-NaN flag); and the result is the set of every value the hardware can
  produce under the evaluations PTX permits (rounding modes, `.ftz`,
  contraction). The outcome:
  - **One infinity.** The node imports as that infinity (counter
    `hardware_infinities`). An `ex2.approx` of -Inf is +0.0 (PTX ISA, `ex2`
    special cases); `max` of +Inf and `min` of -Inf without `.NaN` are that
    infinity, also for a NaN operand (`max.NaN` and `min.NaN` return NaN for one).
  - **Finite.** The node imports as an ideal real.
  - **Otherwise** (NaN possible, or finite and infinite values possible). When an
    operand carries an infinity that this rule introduced (below), the family is
    not proven (`hardware_value_not_determined`); else the node imports as an
    ideal real.

An infinity written as a literal in the PTX has the same value in the model and
on the hardware. It keeps the model's extended-real arithmetic, in which every
other operand is an ideal real: `-inf - m` is `-inf`. So does an infinity the
rule concludes from literal infinities alone (for example `-inf * log2(e)` with
a literal -inf, or `max(-inf, -inf)`), which is the model's own extended-real
value. Any other concluded infinity is introduced: it replaces a value the model
would compute as finite (for example `-FLT_MAX * log2(e)`, whose ideal value is
about -4.9e38). Every value that may be such an infinity, or hold its bits, is
tracked (per import of a node in its scope), and its uses are checked against
the value sets, so that the infinity holds under every permitted evaluation,
contraction included:

- A `selp` passes it on from its chosen operand, or from either operand when its
  condition is not decided; never from its condition. A negation, an absolute
  value, an integer or bit operation and a conversion (`mov.b32` round trips
  included) pass it on.
- A `max` (`min`) passes it on from an operand that may decide it: -inf never
  decides a `max` (+inf a `min`) and is dropped, unless every operand is that
  infinity. With `.NaN`, a NaN operand decides instead: an introduced infinity
  that may decide a `max.NaN` or `min.NaN` whose set may be NaN stops the
  family.
- A comparison is exact on it and passes nothing on. An instruction image whose
  set is finite (for example `ex2.approx` of -Inf) passes nothing on.
- Anything else stops the family: an `add`, `sub`, `mul`, `fma` or `ex2.approx`
  whose set is not finite, and every operation that is no instruction image (a
  division, a reassociated or accumulated sum, an MMA, a software
  function, a floating-point idiom such as a bit-level exponent or rounding), a
  Sum Fold and an unrolled sum.

The value sets follow PTX semantics:

- **Rounding.** "Instructions that support rounding modifiers are IEEE-754
  compliant" (PTX ISA, floating-point instructions); a `mul`, `add` or `sub`
  without rounding modifier rounds to nearest even. IEEE 754 round to nearest
  carries a result at or beyond 2^(emax+1) - 2^(emax-p) (2^128 (1 - 2^-25) for
  `f32`) to the infinity of its sign; `.rz` gives ±MAX, `.rm` and `.rp` give an
  infinity in one direction and ±MAX in the other. The scalar lowering records
  the rounding modes and `.ftz` of the instruction each node is the image of
  (provenance only, never part of a node's identity; a `max` or `min` records
  only `.ftz`). A node made by several instructions takes the union of their
  modes. A node that a rewrite builds in an instruction's place (a reassociated
  sum, the sum that an `fma` without rounding modifier becomes, a rewritten
  exponential, an MMA or reduction-store accumulation) is no image of any
  instruction, even where it coincides with one; its consumers then read every
  value from it. The
  template of a Fold is an image only when both prototype instances are, and
  each further instance must be an image whose modes the template's include, or
  no Fold is formed.
- **Contraction.** PTX may optimize "mul/add and mul/sub sequences with no
  rounding modifiers" into `fma` (PTX ISA, `mul`). The set of such an `add` or
  `sub` that reads a `mul` without rounding modifier also contains the exact
  product plus the other operand, rounded once.
- **Flush to zero.** `.ftz` flushes subnormal operands and results to a zero of
  their sign.
- **Operands.** A literal is its exact value; an `f16`/`bf16` decimal that is no
  value of its type admits every value. A constant read through Bitcasts of one
  width ending in an IEEE type is the value of those bits, and one floating
  conversion may follow, rounded in its mode (`.sat` limits it to [0.0, 1.0]);
  any other conversion admits every value. An operand imported as an infinity is
  that infinity; a `selp` whose condition this scope decided is its chosen
  operand; a negation negates its operand's set in this scope; an instruction
  image imported as a constant is evaluated again from its own operands in this
  scope, at most 16 such evaluations deep. Other operands have scope-independent
  sets computed the same way through constants, `selp`, `neg`, `max`/`min` (one
  operand that is surely a number bounds a maximum from below; `.NaN` propagates
  NaN), running maxima and minima (the initial value bounds a running maximum
  from below) and instruction images; any other node, an input element included,
  admits every value.
- **Conversions.** An edge conversion of an imported value follows the same
  rules; a `cvt.sat` of -inf is +0.0 ("`.sat` limits the result to the range
  [0.0, 1.0]", PTX ISA, `cvt`). A `cvt.sat` of +inf is not handled
  (`saturation_input_finiteness_not_proven`).
- **Never 0 from a finite input.** A finite argument of `ex2.approx.f32` gives a
  result within 2 ulp of the correctly rounded one, which admits the smallest
  subnormals, so it is never taken to be 0.
- **Maxima.** A -MAX_T operand of a `max` (+MAX_T of a `min`) whose other
  operand's set lies at or above -MAX_T (at or below MAX_T) and excludes NaN
  never decides it and is dropped (counter `range_bounds_proved`), without the
  range premise.

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
before using its supported concrete input facts. Kernel reads of a verified
`shape_arrays` table are replaced by the declared values too, not only reference
reads: the race-free, memory-complete native run of the same input proved that
the kernel never writes the table. Its context receipt records the
actual source identities. A source URL or prose `scope` alone is not a checked
numerical premise. Changing an input requires reviewing the associated
provenance rather than copying a successful historical receipt.


## Reference function requirements

Reference capture needs the optional PyTorch dependency, but no GPU or concrete
tensor contents. Pass parameters and buffers explicitly as function arguments.
Input mutation, implicit module state, data-dependent control and unsupported
PyTorch operations can prevent capture. Outputs must be tensors with at least
one dimension. The reference function executes as Python code during capture.
