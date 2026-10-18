# Configurable assumptions

These assumptions can be enabled or disabled to select the behavior analyzed by
the verifier. The pool is defined in [config/assumption-pool.json](../config/assumption-pool.json);
defaults are maintained in [config/default-assumptions.json](../config/default-assumptions.json).


Immediate visibility remains a fixed premise of the current model; removing its
configuration switch did not add weak-memory analysis. Do not put
`immediate_visibility` in the `assumptions` object. Existing
`immediate_visibility: true` configurations can omit the field without changing
this premise; configurations requesting `false` are not supported.
The removed `warp_tmem_load_completion: false` override can also be omitted.
A former `true` override must not be silently dropped: its TMEM completion
semantics require review.

## Duplicate TMA stores

Configuration key: `duplicate_tma_store_benign`.

Many kernels issue duplicate stores to avoid coordination overhead and improve
performance. These unordered, overlapping writes are races in our model.
This exception assumes that matching duplicate TMA stores issued by different lanes of the same
warp are benign. Matching requires the same PTX instruction occurrence and copy
footprints; equality of the stored values is assumed rather than proved.

- **Enabled:** report matching unordered stores as warnings and continue checking
  other accesses. A complete check with no remaining race or unresolved query
  returns `race_free`, with warnings reported separately.
- **Disabled:** report these unordered, overlapping writes as race candidates.


## Representative election

Configuration key: `representative_election`.

Assume that analyzing one representative election outcome is sufficient. Choose
lane 0 when it participates; otherwise choose the lowest-numbered participating
lane.

- **Enabled:** analyze that elected lane's execution.
- **Disabled:** report election instructions as unsupported, with a suggestion
  to enable Representative election.

## Failed launch cancellation

Configuration key: `clc_always_fails`.

Assume that every cluster launch cancellation (CLC) request fails. A CTA therefore
does not acquire additional work from a canceled launch.

- **Enabled:** analyze the failed-request path.
- **Disabled:** report CLC instructions as unsupported, with a suggestion to
  enable Failed launch cancellation.

## Representative branches

Configuration key: `representative_branches` (default `false`).

When exact execution cannot determine an ordinary structured if/else, select the
PTX branch's fallthrough edge every time. This is not necessarily the source
language's else arm. Unknown guards on ordinary non-control instructions execute
the instruction. Predicated ret, exit and trap use the same CFG classification
as conditional branches, including polling protection and loop bounds. Known predicates and explicit user choices take precedence.
Assertion-failing paths are not silently avoided.

The CFG must identify a reconverging branch, separate from loop continuation,
exit, break and continue control. Merely being inside a loop does not make an
if/else a loop test. This assumption cannot bound loops or complete polling waits.

## Bounded computation loops

Configuration key: `bounded_computation_loops` (default `false`).
The existing `data_branch_loop_iterations` input field is an optional positive
integer, default `4`; it does not enable the assumption by itself.

For a recognized computation loop whose continuation test remains unknown, execute
the specified number of body iterations. Counts are per lane and per loop
invocation, reset on re-entry, and independent for nested loops. Known exits,
breaks and returns still take effect. Ordinary if/else requires the separate
Representative branches assumption.

Classification uses CFG natural loops, their nesting, actual exit edges and the
register dependencies of every loop-exit test. An induction counter alone does
not make a loop computational if another exit polls memory. Recognized polling
waits keep their existing wait normalization; unmodeled polling remains
unsupported. Irreducible control, multiple latches, unknown break/continue and
unclassified loops remain unsupported rather than receiving a guessed bound.
The initial supported forms are single-latch counter-based pre-test and post-test
loops. Array scanning with data-dependent termination is not classified as a
computation loop.

Both assumptions are last resorts after exact execution. Existing exact handling
of pure computation regions remains independent. Results record the assumption
key, PTX line, decision rule and taken/not-taken counts, with separate usage
counters. A result covers the selected execution paths; these assumptions do not
establish all-path correctness.

The former combined `representative_data_branches` key is retired. Choose either
or both new keys explicitly; the old visit-count branch heuristic is not retained.

## Order-independent atomics

Configuration key: `order_independent_atomics` (default `false`).

Assume that the kernel is correct for every order in which concurrent atomics
(`atom`, `red`) on the same element take effect. Each atomic is then one
indivisible read-modify-write of its 2-, 4- or 8-byte element, complete before
the thread's next instruction; a packed half pair (`.f16x2`, `.bf16x2`) is one
4-byte element. Two atomics on the same element do not race when their scopes
include both threads: `.gpu` and `.sys` include every thread, `.cta` only threads
of the same CTA. `.cluster` is treated as `.cta`. An atomic and an ordinary
access to the same bytes race unless ordered. Results count executed atomics
under `frontend_assumptions.atomic_uses`.

Concurrent atomics are not data races, so results that depend on their order are
not reported: a floating-point sum that differs between runs, the value a
concurrent `exch` or `cas` leaves behind, or the order of tickets. Some
simplifications can only report more races: memory-order qualifiers
(`.acquire`, `.release`) add no ordering, `ld.acquire`/`ld.relaxed` count as
ordinary accesses, and returned values are unknown, so a branch on one needs the
data or Representative branches. Vector, 128-bit and `.async` forms, `multimem`
and surface reductions remain unsupported.

- **Enabled:** analyze `atom`/`red` as atomic accesses.
- **Disabled:** report the first `atom` or `red` as unsupported, with a
  suggestion to enable Order-independent atomics.

## Address invariants

Input key: `address_invariants` (optional array; default `[]`).

Unlike the switches above, address invariants are not on/off flags. They are trusted
facts about runtime address values that the symbolic address model may use to
prove two memory accesses disjoint. The verifier does not check them. If an
invariant is false for the analyzed execution, a real race may be reported as
safe. Any nonempty set sets the output flag `assumption_dependent` to true, and results
repeat the set under `address_model.address_invariants`.

### How the address model uses them

Each address is an allocation plus an offset `c0 + sum(ck * atom_k)`. An atom is
one runtime value identified by where it came from, so repeated uses of the
same value share one atom. The model first reuses the exact constant-offset
analysis. Pairs it cannot decide go through two layers:

- **Allocation separation (no invariants needed).** Accesses in different known
  memory spaces, or in different named allocations of one space, are disjoint.
  Allocations sharing an alias scope, `generic` addresses, and unresolved
  allocations are not separated.
- **SMT layer.** For two accesses in the same allocation, the model asks Z3
  whether the byte ranges can intersect under the invariants. They are disjoint
  only if Z3 proves they cannot. A satisfiable result, a timeout, or an
  inconsistent invariant set leaves the pair unresolved; the reason includes a
  candidate valuation when one exists. Offsets differing by a constant are
  decided without the solver, and an intersection is then reported as an
  overlap for every valuation.

Without invariants, and in builds configured with `-DSENTRYWARD_WITH_Z3=OFF`,
same-allocation pairs with symbolic offsets stay unresolved.

### Invariant forms

Each array entry is an object of one of these forms:

| Form | Meaning |
| --- | --- |
| `{"term": T, "lower_bound": L, "upper_bound": U, "multiple_of": M}` | `L <= T <= U` and `T` is divisible by `M` (a positive integer). At least one of the three fields is required; bounds are inclusive. |
| `{"gather": "g", "injective": true, "lower_bound": L, "upper_bound": U}` | Every lookup in table `g` lies in `[L, U]`. If injective, different ground arguments give different results. |
| `{"joint_injective": ["r", "s", ...]}` | For different ground arguments, at least one of the named tables gives different results. |
| `{"distinct_allocations": ["A", "B", ...]}` | The named allocations are pairwise distinct storage. |

Unknown fields, empty names, empty ranges and contradictory term bounds make the
input invalid.

`T` is a scalar expression:

- an integer;
- `{"operator": "param", "symbol": "PAGE"}`: an immutable runtime scalar;
- `{"operator": "gather", "symbol": "pages", "args": [3]}`: a table lookup with
  integer arguments;
- `add`, `sub`, `mul` or `neg` with an `operands` array.

A product of two non-constant operands becomes its own atom with no assumed
value. A term that names that product can still bound it.

Allocation names in `distinct_allocations` are binding allocation names (for
example `epilogue.tensormaps`), or symbolic targets of runtime tensor maps. A
TMA transfer whose tensor map's global address is unknown reads
`tensor_map_target(<location>+<offset>)`, where the location and offset identify
where the tensor map is stored. Unresolved-pair diagnostics print these names.
Two tensor-map targets are never assumed equal or distinct without an
invariant.

Example: page table `pages` maps logical blocks to distinct physical pages, and
the tensor read through one tensor map does not alias a workspace:

```json
"address_invariants": [
  {"gather": "pages", "injective": true, "lower_bound": 0},
  {"distinct_allocations": ["tensor_map_target(input.arg0.1184+0)", "epilogue.tensormaps"]}
]
```

The PTX frontend does not yet emit `param` or `gather` atoms. Today the SMT
layer applies to operand-derived offsets and to source JSON produced by other
tools, while `distinct_allocations` applies directly to frontend output.
