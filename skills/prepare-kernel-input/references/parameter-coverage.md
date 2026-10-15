# Parameter coverage and evidence

Use this when recovering a packed ABI or checking a prepared input. This is
input-only inspection; it does not specialize PTX or run verification.

## Inventory the selected entry

Inspect its parameter declarations, direct `ld.param` accesses, parameter address
uses, calls, and tensor-map references. A vector load covers all its bytes:
`ld.param.v2.b32 ... [params+16]` reads `[16,24)`, not just offset 16.

Match ordinal or named selectors to the same parameter. For each consumed field
record its byte offset, type/width, role, chosen value or object, and source
expression. Example:

| PTX use | Input binding | Evidence to establish |
|---|---|---|
| Load a length from `params+8` | A typed scalar argument | Shape and host field offset |
| Load a pointer from `params+16` | Symbolic pointer declaration | Pointer type and object identity |
| Load a stride from `params+24` | A typed scalar argument | Source layout and whether units are elements or bytes |
| Tensor TMA uses a map at `params+128` | Descriptor binding | Map field location and constructor/layout derivation |

Declaring a scalar's type does not bind its value. Declaring a tensor map does
not provide its geometry. A pointer declaration provides an identity, not the
contents of the pointed-to tensor. An explicit null pointer must follow from the
scenario; do not use null to fill an unknown non-null pointer.
For default-constructed structs and layout tuples, inspect their initializers
and constructors. Bind zero only when the source establishes it; neither an
absent initializer nor an unused field justifies zero-filling a parameter pack.
Different input pointer parameters, including packed pointer fields, receive
distinct symbolic objects under the project's non-overlap scope. This also
applies when host code passes workspace suballocations as separate parameters.
No cross-parameter alias recovery or pointee-offset configuration is required.
Repeated loads of the same parameter field use the same object; arithmetic
derived from that pointer inside PTX is still analyzed normally.

## Inspect without inventing missing bytes

The audit helper inventories syntactic accesses. Some loads may be unreachable
for the selected scenario; establish that from source/control-flow evidence
before exempting them. Do not add a branch override to silence the audit.
An unused parameter or padding byte requires no invented value.

Common review cases include a wide load with unused vector components or struct
padding, and a scalar fallback used only when a bound non-null pointer is null.
Compiler `used_bytes_mask` pragmas can help locate unused bytes; confirm their
use/def or source layout rather than treating a compiler hint as a new execution
rule. For a dead fallback, record the guard and binding that excludes it. Keep
these resolutions beside the raw audit instead of supplying arbitrary padding
or forcing a branch choice.

The helper resolves unambiguous, dominating `mov`/`cvta` parameter-base aliases
and constant `add`/`sub` offsets. It does not execute branches or loop bodies.
Explicit PTX `.ptr` parameters already provide symbolic pointer bindings; do not
require redundant declarations. For untyped packed fields, source may establish
a pointer the frontend can infer, or a scalar that really needs a value. Resolve
that distinction before asking the user for anything.

Indirect parameter addressing or a helper call can hide accesses. Leave these
as manual-review items until their ranges and bindings are established. Direct
read coverage must not be reported as whole-program completeness in their
presence. Device loads are not parameter loads: their unknown contents can be a
legitimate limitation of constant-address analysis even with a complete launch
configuration.

## Read the audit result

| Audit status | Meaning and action |
|---|---|
| `coverage_complete` (exit 0) | Inventoried reads have bindings and no automatic review flags remain. Still check the values against source. |
| `binding_gaps` (exit 1) | Some syntactic reads lack explicit value/pointer coverage. Review their roles and reachability, then derive legitimate bindings or record why no binding is needed. |
| `needs_review` (exit 1) | Indirect reads, helper calls, opaque descriptor bytes, or map updates need source reasoning beyond this helper. |
| `invalid_input` (exit 2) | Correct the reported schema, selector, bounds, or conflicting-binding error. |

These are preparation diagnostics, not verifier outcomes. Preserve the raw audit
and record resolutions in adjacent evidence rather than editing its status.
For example, a kernel that constructs every tensor map on device may need no
launch descriptors: demonstrate the construction and scalar inputs, and resolve
the missing-launch-map review item. Conversely, a map updated from unknown
device pointer-array elements still has a genuine dynamic dependency.

## Retain reusable evidence

For nested parameter packs, follow the instantiated member types through their
host constructors. Missing bytes may hold ordinary epilogue controls (alpha,
beta, bias mode), not tensor elements. Choose a legal documented scenario and
derive its nested fields consistently. A caller-overridable scheduling parameter
may likewise have a selected default; disclose that choice and derive the launch
from it instead of presenting it as a measured hardware property.
Check runtime dtype enum values and widths against the selected source type.
An unknown leading extent does not prevent deriving a contiguous stride that
depends only on known trailing dimensions.

Keep source revision and locations, the scenario, and the evaluated expressions
beside the generated input. For a template or DSL ABI, retain a CPU-only evaluator
when needed, its inputs, dependency versions, and output. In particular, matching
CUDA/CUTLASS ABI alignment matters for packed tensor-map fields. Neither old JSON
nor a successful parser/verifier run substitutes for derivation evidence.
If a small CPU ABI projection replaces an expensive CUDA instantiation, preserve
every member's order, size, and alignment and the actual layout/template types.
Document any pointer-pointee erasure; it must preserve the relevant pointer ABI.

If the public schema cannot express a proven value, retain the exact value and
identify the restriction. Do not change zero to a positive stride, shrink a tile,
or supply arbitrary register or device-array values. A schema limitation needs
an implementation change; a skill should not hide it in a different scenario.
