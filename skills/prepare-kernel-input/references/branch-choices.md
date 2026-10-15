# Optional branch choices

Use only when the user explicitly authorizes branch choices for the requested
kernel/scenario. Ordinary default-input generation does not enable this mode.
An explicit request to select recognized attention correction paths is sufficient
for those patterns; individual line-by-line approval is unnecessary.

Inspect the PTX conditional branch, target, and fall-through, with source when
available. Match behavior, not kernel names or historical line numbers.

| Recognized pattern | Selected scenario |
|---|---|
| Skip rescaling accumulated attention output when the scale needs no update | Execute the correction/rescale path |
| A nested guard skips per-lane rescaling in that correction | Execute the per-lane update when its role is established |

These occurred in CUTLASS 77 and FlashAttention forward paths. A jump over the
update usually requires `taken: false`; determine polarity for each instruction,
including `@!p`. Selecting this path is not proof that it follows from the input
shape or covers every execution.

Do not automatically select numerical-formula branches such as CAKE softplus
thresholds, synchronization polling, bounds checks, or role selection. Ask for a
specific choice if needed. Historical exploratory overrides are not authority.

Encode supported fields only:

- `line`: conditional branch line in the supplied PTX.
- `taken`: `true` to jump or `false` to fall through.
- `taken: "random"` and nonnegative `seed`: a reproducible choice fixed per branch
  across threads and iterations, only when the user requests random scenarios.
- `note`: operation, selected path, authorization, and conditional scope.

Example note: “Always execute softmax correction; do not take the branch that
skips rescaling. User-enabled pattern-based scenario; revisit when dynamic
branch analysis is supported.”

List each configured branch's line, outcome, and functional meaning at handoff.
