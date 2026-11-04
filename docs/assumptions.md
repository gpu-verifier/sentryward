# Configurable assumptions

These assumptions can be enabled or disabled to select the behavior analyzed by
the verifier. The pool is defined in [config/assumption-pool.json](../config/assumption-pool.json);
defaults are maintained in [config/default-assumptions.json](../config/default-assumptions.json).
Some premises of the model are fixed; they are described first.

## Invocation overrides

Persisted choices belong in the kernel input `assumptions` object. For one run,
`--assume KEY` / `--no-assume KEY` and Python `verify(..., assumptions={...})`
override those keys. Defaults apply first, then the input file, then these
overrides. `sentryward --list-assumptions` prints the accepted keys and the
repository defaults as JSON and does not require a PTX file.

## Fixed premises

These premises are not configurable.

Immediate visibility is a fixed premise of the model. Weak-memory behavior is
not analyzed.

TMEM completion follows our reading of the PTX documentation (tcgen05 memory
consistency). A `tcgen05.ld` delivers its registers to true register
dependencies, but its TMEM read completes only at `tcgen05.wait::ld`; using the
loaded registers does not order the read before later memory operations,
releases or deallocation. From the same warp, `tcgen05.ld`/`tcgen05.st` pairs
(ld then st, st then ld, st then st) access TMEM in issue order.

CTA barriers without a thread count follow the PTX ISA (`bar`, `barrier`;
`exit`). All threads of the CTA participate. `bar.sync`, `barrier.sync` and
`bar.red` wait only for the non-exited threads, and a barrier that waits only on
exited threads is released. When every use of a barrier in a CTA has this form,
thread u arrives n_u times and its k-th arrival belongs to phase k. Phase k
waits for the threads with n_u > k. Each other thread must end at `ret` or
`exit`, and the synchronization check proves that it can exit before phase n_u
completes (an exit gate). A gate is a scheduling obligation, not an ordering:
an exiting thread releases nothing, so its accesses after its last arrival and
its pending asynchronous copies stay unordered with the accesses after the
barrier. If the completion of phase k is ordered before the exit of a thread
gated at phase k or earlier on the same barrier, the kernel deadlocks
(`not_wellsynced`, diagnostic `exit_gate_unsatisfiable`). If it is ordered only
before the exit of a thread gated on another barrier, and the reference
execution does not let every gated thread exit first, the result is
`unsupported` (diagnostic `exit_gate_order_unproven`). A barrier also used with
a thread count or by `bar.arrive` keeps its counts: there, an exited thread
never counts as arrived. The shared-memory control executor also keeps the full
CTA count.

The value analysis models `cvt.{rz,rp}{.satfinite}.ue8m0x2` from `.f32` or
`.bf16x2` by the reference conversion in CUDA 13.2 `cuda_fp8.hpp`
(`__nv_cvt_float_to_e8m0`, `__nv_cvt_bfloat16raw_to_e8m0`), because the PTX ISA
does not state the bit rule. The sign is ignored. `.rz` keeps the biased
exponent. `.rp` first rounds |x| up to bf16, then up to a power of two: the
exponent is incremented when the bf16 mantissa is nonzero and
2^-127 < |x| < infinity. NaN gives 0xFF, and `.satfinite` turns a non-NaN 0xFF
into 0xFE.

The value analysis models an MMA whose A operand is in Tensor Memory and whose
data path has several lane groups (`.cta_group::2` M=128 with dense A, and
`.ws` M=64 and M=32; PTX layouts B, E and G) as one A copy per lane group: lane
group s reads every A row from its own lanes and multiplies it into D columns
`[s·N/g, (s+1)·N/g)`, where g is the number of lane groups. The copies need not
be equal. The PTX ISA does not state where a lane group reads A; this follows
CUTLASS's duplicated TMEM-A allocation (`UMMA::TmemAllocMode::Duplicated`).
Likewise, for `tcgen05.cp.64x128b`, TMEM lane L receives source row L mod 64
with `.warpx2::02_13` and row 32·⌊L/64⌋ + L mod 32 with `.warpx2::01_23`; the
ISA names the warp pairs but not which half each pair receives, so this follows
CUTLASS `SM100_UTCCP_2x64dp128bitlw0213` / `…lw0123`.

### Merged forward branches

A conditional forward branch whose predicate the input does not determine, for
example because it depends on tensor data, is merged when it skips a
straight-line block with these properties:

- The block contains only register computation, reads of parameter, local or
  constant memory, and `tcgen05.ld`, `tcgen05.st` and `tcgen05.wait`.
- No register that the block writes appears outside the block.

The block is then analyzed as executed on both paths. This over-approximates
the path that skips it.

- When the block contains tcgen05 operations, the predicate must also be
  provably warp-uniform, for example computed in the branch's basic block from
  constants and a full-warp `vote.sync`. A block without tcgen05 operations
  needs no warp-uniform predicate.
- If a tcgen05 operation that the block could order is pending at the branch,
  the result is `unsupported` (diagnostic `two_arm_merge_pending_tcgen05`).
- The value analysis selects between the two paths after the join: a register
  the block writes becomes the value with the block where the branch is not
  taken and its earlier value where it is. A `tcgen05.st` in the block stores,
  where the branch is taken, the values the cells already hold; these must be
  the registers of a `tcgen05.ld` of the same cells (same address register
  value and shape) earlier in the block, with no `tcgen05.st` in between.
  Otherwise the result is `unsupported` (diagnostic `two_arm_merge_values`).

A forward region of register computation and parameter, local or constant
reads is skipped when the input does not determine its branch. If ordinary
execution stops at such a branch, the region may also have several arms
(if-else, nested ifs) up to one join, and inline-asm blocks inside it that
declare only their own temporaries. Warp-level operations (`shfl`, `vote`,
`elect`, `match`, `activemask`, `redux`, `mma`) are never skipped. The value
analysis if-converts the region exactly: each register it writes becomes a
selection of its per-path values. It reports `unsupported` for local or
constant reads in the region, carry arithmetic, and a register without an
earlier value that is read on a path where the region does not assign it
(diagnostics `lifted_region_*`, or `missing_definition` after the join).

The region may also store to one of the thread's own `.local` arrays
(`st.local`, one element width) when each store's byte offset is proved to be
an aligned access inside the array. The address must be the array symbol plus
a constant plus integer registers, through registers with one definition each,
and the array may have at most 64 element slots. The proof evaluates the
offset over the registers' single definitions, each preceding its uses on
every path. Thread coordinates and the lane are bounded by the launch. A
`selp` with an unknown predicate and a `shfl` give any value of their
operands. The offset must take only exact values (a table over one bounded
register, or a single value), each an aligned access inside the array.
Synchronization treats the region as one read and one write of the whole
array. The executor forgets what it knew about the array's contents. The
value analysis updates every slot `j` with a selection for each store in
program order: the stored value on the store's path when its offset addresses
slot `j`, the slot's value otherwise. Without the proof, the result is
`unsupported` (diagnostic `lifted_region_local_store`).

A data-dependent branch that is neither merged nor skipped is unsupported unless a branch
choice or a representative-control assumption, such as
[Representative branches](#representative-branches), decides it.

### Same-value writes

Two unordered writes of the same bytes are a data race in the PTX memory model
(PTX ISA 8.7.1). This is a fixed rule, not an assumption: such a pair is not
reported as a race when the value analysis proves that both writes store the
same bytes. It applies when both writes:

- are plain (non-atomic) writes through the generic proxy: `st` (not
  `st.async`, `st.bulk` or `st.mmio`), `stmatrix`, or a non-bulk `cp.async`;
- cover exactly the same bytes with the same width. PTX calls a race between
  such writes uniform-size (8.7.2).

Read/write pairs are always checked. Partially overlapping writes form a
mixed-size race, for which the memory model gives no axioms; they stay races.
Asynchronous-proxy writes are not covered; duplicate TMA stores have their own
assumption below.

`race_free` therefore means: every pair of conflicting accesses is ordered,
except unordered write pairs of this kind whose stored bytes are proven equal.

How it is checked:

- The race check records each such pair as an obligation and continues. It
  works on access families: an obligation covers a pair of families and a
  translation between their members. Writes connected by obligations form a
  group; all writes of a group cover the same bytes.
- The value analysis runs under the assumption that the obligations hold: a
  later read may observe any write of a group. It processes the memory
  accesses in an order consistent with happens-before, and compares the bytes
  of each write of a group with those of the group's first write.
- Downstream checks use the same assumption. In the shared-constant and
  tensor-map validations, writes of one group do not interfere with each
  other: a read may take any of them as its source.
- Bytes are equal when they are copies of the same bytes, or when the stored
  values are equal. Values are equal when they are the same value, the same
  constant or launch parameter (same literal or symbol and type), loads of the
  same initial input bytes, loads of the same stored value, loads of unwritten
  storage at equal addresses, or the same operation (kind, attributes, types)
  on equal operands. Approximate instructions such as `ex2.approx` count as
  the same operation: their results are implementation-specific but taken to
  be a function of the operands. Thread exchanges (`shfl`) are resolved to the
  source thread's value.
- The only algebraic rewrite is swapping the two operands of a single `add`,
  `mul`, `min` or `max` on integers or on f16, bf16 or f32 values, and of
  `and`, `or` and `xor`. Integer forms are exactly commutative. For the
  floating-point forms the correctly rounded result of a finite or infinite
  result is symmetric, `min` and `max` order -0.0 below +0.0, and a NaN result
  is an unspecified NaN for either order (PTX ISA 9.7.3: "Single-precision
  instructions return an unspecified NaN"; 9.7.4 for half precision), so either
  write is a permitted result of either instruction. f64 forms are not swapped,
  since PTX propagates double-precision NaN payloads. No reassociation or other
  real-number identity is used.

Results:

- All obligations proven: `race_free`, or the equivalence result when a
  reference is given. Each pair of write sites (per CTA) is a warning with
  code `same-value-writes`; `warning_count` includes them. A pair whose writes
  lie in two CTAs counts under the CTA of the write whose (site, CTA) is the
  lesser.
- A write whose bytes are not proven equal: `potential_race` with diagnostic
  `same_value_writes_unproven`. The witness pairs that write with a write that
  the race check accepted with it: the two are unordered and cover the same
  bytes.
- Obligations not decided, because the value analysis was not requested,
  stopped before reaching them, or reached the equality proof's bounds on one
  of them (reason `stored_value_equality_limit_exceeded`): `potential_race`
  with diagnostic
  `same_value_writes_unchecked`, the first obligation pair as witness, and the
  value analysis's own failure in the message. Without the value analysis
  (synchronization-only runs) this applies even to writes of equal constants.
  A time or memory budget that stops the value analysis first reports the
  synchronization check as incomplete.

Why this is sound:

- Read/write races are still reported, so every read is ordered after every
  write it can observe. A read of a group's bytes is ordered after all writes
  of the group.
- Happens-before is acyclic. By induction over the order in which the value
  analysis processes accesses: when it compares a write, the stored value
  depends only on earlier reads, and each of those reads observes writes that
  were all processed before it. If those writes belong to groups, every write
  of those groups was processed and proven equal before the read. So the read
  values, and hence the compared values, do not depend on which write of a
  group the analysis chose. Proving each obligation under the assumption
  proves it outright.
- The first failed comparison is therefore about actual values. Its writes
  depend only on proven obligations.
- With equal bytes the coherence order of the writes does not matter: after
  both, the location holds those bytes, and every later read observes them.

Cross-CTA classes (class mode): a write pair inside an analyzed CTA or cluster
stands for that pair in every unit of its class, and its warning is counted
once per unit. If a member unit shifts one global allocation by different
amounts at different instruction sites (a piecewise shift, used only when the
run lifts no values), the cross-unit pass also checks each write at one such
site against the accesses at another, moved apart by the member's shift
difference; if they may overlap, the launch is re-run per unit. A value proof
that reads the CTA index numerically re-runs the launch per unit. A global
write pair between two different units, found by the cross-unit pass, is not
instantiated per member. If the rule above could accept it, the launch is
re-run per unit, whose analysis checks such pairs as obligations; otherwise it
is a race. A re-run caused only by values that do not lift keeps the class-mode
synchronization verdict when that verdict is race free without warnings or
same-value obligations. The re-run then redoes only the value analysis. It
still builds the per-unit happens-before order, which orders the value
analysis's memory actions.

## Floating values stay in range

Configuration key: `floating_values_in_range` (default `true`).

This premise belongs to the numerical (equivalence) model, which treats floating
values as ideal reals ([reference binding](reference-binding-json.md#equivalence-contract)).
Where that model treats a value of an IEEE type T (`f16`, `bf16`, `f32`, `f64`)
as a finite real, it assumes the value lies in T's finite range:
|x| <= MAX_T, the largest finite value of T (65504 for `f16`, (2 - 2^-7) 2^127
for `bf16`, (2 - 2^-23) 2^127 for `f32`, (2 - 2^-52) 2^1023 for `f64`). On
hardware a finite result of T always satisfies this; only the unrounded ideal
value can leave the range, where IEEE arithmetic would have overflowed to an
infinity.

The comparison uses the premise in one place: in a `Maximum` (`Minimum`) of
type T with at least one operand known to be finite, an operand that is the
constant -MAX_T (+MAX_T) cannot decide the result and is dropped (counter
`range_premise_bounds_dropped`). Example: a running maximum clamped from below
by -FLT_MAX before it is used as an exponent pivot (`max.f32 m, x, 0fFF7FFFFF`).
It adds no bound on inputs, no finiteness of any value and no other algebra.

- **Enabled:** drop such operands. In the result's `assumptions` entry,
  `triggered` is `true` when the comparison dropped one, `false` when it
  imported every output family without a drop, and `null` when it did not run
  or stopped early.
- **Disabled:** keep them. An equality that depends on the premise is then not
  proven.

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
as conditional branches, including polling protection and loop bounds. Known predicates, explicit user choices and [merged forward branches](#merged-forward-branches) take precedence.
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
data or Representative branches. Vector, 128-bit and `.async` forms of `atom`
and `red`, `multimem` and surface reductions remain unsupported. The bulk
reduction `cp.reduce.async.bulk` (to global from `shared::cta`) is analyzed
without this switch: each element is one atomic read-modify-write at the
instruction's scope, completed through bulk async-groups. Its value model is
described under [equivalence checking](reference-binding-json.md#equivalence-contract).

- **Enabled:** analyze `atom`/`red` as atomic accesses.
- **Disabled:** report the first `atom` or `red` as unsupported, with a
  suggestion to enable Order-independent atomics.

## Address invariants

Input key: `address_invariants` (optional array; default `[]`).

Unlike the switches above, address invariants are not on/off flags. They are trusted
facts about runtime address values that the symbolic address model may use to
prove two memory accesses disjoint. The verifier does not check them. If an
invariant is false for the analyzed execution, a real race may be reported as
safe. Any nonempty set sets the output flag `assumption_dependent` to true. Results
repeat the set under `internal.address_model.address_invariants`, which is not a
stable interface.

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

Example: the tensor read through one tensor map does not alias a workspace:

```json
"address_invariants": [
  {"distinct_allocations": ["tensor_map_target(input.arg0.1184+0)", "epilogue.tensormaps"]}
]
```

Bounds and injectivity of the elements of an input object, such as a page table,
are stated as [tensor properties](input-json.md#tensor-properties). An entry of
any other form, such as the earlier `{"gather": ...}` bound, is rejected.

The PTX frontend emits `gather` atoms for elements loaded from an input object
with tensor properties. When an address or TMA coordinate is affine in such
elements, each element appears as the atom
`{"operator": "gather", "symbol": "tensor_property:<object>:<element_type>", "args": [B]}`,
where `B` is the element's byte offset in the object. A non-affine operation of
one such element with constants (for example `x & 0xffffff`, `x >> 24` or
`min(x, n)`) is an element function: the atom
`{"operator": "gather", "symbol": "tensor_property_function:...", "args": [B], "object": ..., "lower": L, "upper": U}`
names that exact function of element `B`, whose values lie in `[L, U]`. Equal
atoms are equal values; distinct atoms are independent within their bounds, and
an element function is never injective. Term and `joint_injective` invariants
can name the element symbols, for example to bound one element or to state that
two index tables are jointly injective:

```json
"address_invariants": [
  {"term": {"operator": "gather", "symbol": "tensor_property:kv_indices:s32", "args": [0]}, "upper_bound": 15},
  {"joint_injective": ["tensor_property:row_idx:s32", "tensor_property:col_idx:s32"]}
]
```

The frontend does not emit `param` atoms. The SMT layer also applies to
operand-derived offsets and to source JSON produced by other tools, while
`distinct_allocations` applies directly to frontend output.
