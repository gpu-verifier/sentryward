<!-- report-layout:v2 -->

<!-- verifier-result:start -->
<p align="center">flash_attention__bwd_hd256_bf16_2cta__sm_100a</p>
<h1 align="center">LIFECYCLE ORDERING MISSING</h1>

**Inputs:** [PTX](https://github.com/gpu-verifier/sentryward/blob/94a6b26b1990a4b9df7ded3727acc4b4cc36f23c/corpus/flash_attention__bwd_hd256_bf16_2cta__sm_100a.ptx) · analyzed code; [input.json](https://github.com/gpu-verifier/sentryward/blob/94a6b26b1990a4b9df7ded3727acc4b4cc36f23c/examples/flash_attention__bwd_hd256_bf16_2cta__sm_100a/input.json) · launch, arguments and assumptions.

**Result:** [output.json](https://github.com/gpu-verifier/sentryward/blob/94a6b26b1990a4b9df7ded3727acc4b4cc36f23c/examples/flash_attention__bwd_hd256_bf16_2cta__sm_100a/output.json) · saved verifier findings.

<details>
<summary><strong>Analysis records — open the saved evidence</strong></summary>

Each record link opens its field in the original `output.json`. Model summaries are not full execution traces.

**Check the finding**

- [Diagnostic messages and details · `diagnostics`](https://github.com/gpu-verifier/sentryward/blob/94a6b26b1990a4b9df7ded3727acc4b4cc36f23c/examples/flash_attention__bwd_hd256_bf16_2cta__sm_100a/output.json#L67)
- [Individual check outcomes · `checks`](https://github.com/gpu-verifier/sentryward/blob/94a6b26b1990a4b9df7ded3727acc4b4cc36f23c/examples/flash_attention__bwd_hd256_bf16_2cta__sm_100a/output.json#L56)
- [Failed lifecycle orders · `failed_required_orders`](https://github.com/gpu-verifier/sentryward/blob/94a6b26b1990a4b9df7ded3727acc4b4cc36f23c/examples/flash_attention__bwd_hd256_bf16_2cta__sm_100a/output.json#L5652)

**Check how the model was built**

- [PTX translation · `frontend`](https://github.com/gpu-verifier/sentryward/blob/94a6b26b1990a4b9df7ded3727acc4b4cc36f23c/examples/flash_attention__bwd_hd256_bf16_2cta__sm_100a/output.json#L7033)
- [Model and source bindings · `semantic_program`](https://github.com/gpu-verifier/sentryward/blob/94a6b26b1990a4b9df7ded3727acc4b4cc36f23c/examples/flash_attention__bwd_hd256_bf16_2cta__sm_100a/output.json#L7340)
- [Synchronization-check setup · `synchronization_input`](https://github.com/gpu-verifier/sentryward/blob/94a6b26b1990a4b9df7ded3727acc4b4cc36f23c/examples/flash_attention__bwd_hd256_bf16_2cta__sm_100a/output.json#L12479)

**Check values separately**

- [Value-analysis status · `value_analysis`](https://github.com/gpu-verifier/sentryward/blob/94a6b26b1990a4b9df7ded3727acc4b4cc36f23c/examples/flash_attention__bwd_hd256_bf16_2cta__sm_100a/output.json#L12506)
- [Numerical comparison · `equivalence`](https://github.com/gpu-verifier/sentryward/blob/94a6b26b1990a4b9df7ded3727acc4b4cc36f23c/examples/flash_attention__bwd_hd256_bf16_2cta__sm_100a/output.json#L2845)

For the records explained in this report, see the [Evidence appendix](#evidence-appendix).

</details>

## 1. Verifier result

Saved findings and analysis scope, formatted directly from the verifier output.

### 1.1 Required lifecycle order

A required lifecycle happens-before relation is absent in the analyzed model.

**Saved result:** <code>invalid_lifecycle</code> · <code>output.json#/result</code>.

**Checks:** Synchronization <code>incomplete</code> · Memory <code>not_run</code> · Equivalence <code>not_run</code>.

**Operation:** <code>kernel_verification</code>.

**2 diagnostic categories · 2 retained records.**

Grouped by saved stage and code, not by inferred root cause or severity. Record counts are not counts of independent bugs.

| Saved diagnostic | Records | Reported extent |
| --- | ---: | --- |
| tcgen05.load must complete before the allocation is deallocated | 1 | 3,145,728 (`details.count`) |
| all accesses to both paired TMEM allocations must complete before either warp attempts deallocation | 1 | 6,291,456 (`details.count`) |

Category codes, locations and array indices: Evidence appendix → Diagnostic records.

**Different lifecycle requirements are reported.** Each relation below is a requirement the model did not establish, not an observed execution order.

#### One allocation

tcgen05.load must complete before the allocation is deallocated

```mermaid
flowchart LR
  F["Access completes"]
  T["Deallocate that allocation"]
  F -. "required" .-> T
  classDef context fill:#f6f8fa,stroke:#8c959f,color:#24292f
  classDef operation fill:#ddf4ff,stroke:#54aeff,color:#0550ae
  classDef focus fill:#fbefff,stroke:#c297ff,color:#6639ba
  class F operation
  class T focus
```

Dashed arrow: required ordering, not established by this run.

**This category:** 3,145,728 failures reported; 32 witnesses retained; truncated: <code>true</code>.

#### Paired TMEM allocations

all accesses to both paired TMEM allocations must complete before either warp attempts deallocation

```mermaid
flowchart LR
  F["All accesses to both<br/>allocations complete"]
  T["Either warp attempts<br/>deallocation"]
  F -. "required" .-> T
  classDef context fill:#f6f8fa,stroke:#8c959f,color:#24292f
  classDef operation fill:#ddf4ff,stroke:#54aeff,color:#0550ae
  classDef focus fill:#fbefff,stroke:#c297ff,color:#6639ba
  class F operation
  class T focus
```

Dashed arrow: required ordering, not established by this run.

**This category:** 6,291,456 failures reported; 32 witnesses retained; truncated: <code>true</code>.

**Top-level sample:** 32 orders retained out of 9,437,184 reported; truncated: <code>true</code>. It need not include every category; the per-category witnesses above come from `diagnostics[].details`.

<details>
<summary><strong>Check the separate requirements and their witnesses</strong></summary>

#### One allocation

**Requirement:** <code>access_completion_before_dealloc</code> · <code>output.json#/diagnostics/0/details</code>.

First saved witness: model instruction <code>85888</code> → <code>86362</code>. The arrow denotes the required order.

#### Paired TMEM allocations

**Requirement:** <code>tmem_group2_access_before_dealloc</code> · <code>output.json#/diagnostics/1/details</code>.

First saved witness: model instruction <code>85888</code> → <code>86362</code>. The arrow denotes the required order.

Instruction IDs identify model records, not physical PTX lines or CUDA issuers. All retained witnesses and effect lists remain in Evidence appendix → Diagnostic records.

</details>

### 1.2 Analysis scope

**Coverage:** Memory analysis complete: **no** · Search complete: <code>not saved</code> · Exhaustive search: <code>not saved</code>.

**Assumption-dependent:** <code>true</code>. Triggered rules: Representative election.

This result applies to the supplied launch, arguments and enabled assumptions.
<!-- verifier-result:end -->

<!-- verifier-evidence:start -->
## Evidence appendix

Use these records to check the verifier result and source explanation above. JSON pointers refer to the original saved files. Identical prerequisite copies use `same_records_as` to refer to the diagnostic evidence shown once; their original locations and content hashes remain recorded.

<details>
<summary><strong>Diagnostic records — verify the finding</strong></summary>

These are saved verifier records. Truncation flags describe missing records in the original result, not merely collapsed presentation.

**Saved error:** tcgen05.load must complete before the allocation is deallocated

#### Lifecycle requirements

9,437,184 failures reported; 32 retained; truncated: <code>true</code>.

| Requirement / from instruction | To instruction IDs | Retained records |
| --- | --- | --- |
| <code>access_completion_before_dealloc</code><br/>from <code>85888</code> | <code>86362</code> | <code>0</code> |
| <code>access_completion_before_dealloc</code><br/>from <code>86532</code> | <code>86362</code> | <code>1</code> |
| <code>access_completion_before_dealloc</code><br/>from <code>87176</code> | <code>86362</code> | <code>2</code> |
| <code>access_completion_before_dealloc</code><br/>from <code>87820</code> | <code>86362</code> | <code>3</code> |
| <code>access_completion_before_dealloc</code><br/>from <code>88464</code> | <code>86362</code> | <code>4</code> |
| <code>access_completion_before_dealloc</code><br/>from <code>89108</code> | <code>86362</code> | <code>5</code> |
| <code>access_completion_before_dealloc</code><br/>from <code>89752</code> | <code>86362</code> | <code>6</code> |
| <code>access_completion_before_dealloc</code><br/>from <code>90396</code> | <code>86362</code> | <code>7</code> |
| <code>access_completion_before_dealloc</code><br/>from <code>91040</code> | <code>86362</code> | <code>8</code> |
| <code>access_completion_before_dealloc</code><br/>from <code>91684</code> | <code>86362</code> | <code>9</code> |
| <code>access_completion_before_dealloc</code><br/>from <code>92328</code> | <code>86362</code> | <code>10</code> |
| <code>access_completion_before_dealloc</code><br/>from <code>92972</code> | <code>86362</code> | <code>11</code> |
| <code>access_completion_before_dealloc</code><br/>from <code>93616</code> | <code>86362</code> | <code>12</code> |
| <code>access_completion_before_dealloc</code><br/>from <code>94260</code> | <code>86362</code> | <code>13</code> |
| <code>access_completion_before_dealloc</code><br/>from <code>94904</code> | <code>86362</code> | <code>14</code> |
| <code>access_completion_before_dealloc</code><br/>from <code>95548</code> | <code>86362</code> | <code>15</code> |
| <code>access_completion_before_dealloc</code><br/>from <code>96192</code> | <code>86362</code> | <code>16</code> |
| <code>access_completion_before_dealloc</code><br/>from <code>96836</code> | <code>86362</code> | <code>17</code> |
| <code>access_completion_before_dealloc</code><br/>from <code>97480</code> | <code>86362</code> | <code>18</code> |
| <code>access_completion_before_dealloc</code><br/>from <code>98124</code> | <code>86362</code> | <code>19</code> |
| <code>access_completion_before_dealloc</code><br/>from <code>98768</code> | <code>86362</code> | <code>20</code> |
| <code>access_completion_before_dealloc</code><br/>from <code>99412</code> | <code>86362</code> | <code>21</code> |
| <code>access_completion_before_dealloc</code><br/>from <code>100056</code> | <code>86362</code> | <code>22</code> |
| <code>access_completion_before_dealloc</code><br/>from <code>100700</code> | <code>86362</code> | <code>23</code> |
| <code>access_completion_before_dealloc</code><br/>from <code>101344</code> | <code>86362</code> | <code>24</code> |
| <code>access_completion_before_dealloc</code><br/>from <code>101988</code> | <code>86362</code> | <code>25</code> |
| <code>access_completion_before_dealloc</code><br/>from <code>102632</code> | <code>86362</code> | <code>26</code> |
| <code>access_completion_before_dealloc</code><br/>from <code>103276</code> | <code>86362</code> | <code>27</code> |
| <code>access_completion_before_dealloc</code><br/>from <code>103920</code> | <code>86362</code> | <code>28</code> |
| <code>access_completion_before_dealloc</code><br/>from <code>104564</code> | <code>86362</code> | <code>29</code> |
| <code>access_completion_before_dealloc</code><br/>from <code>105208</code> | <code>86362</code> | <code>30</code> |
| <code>access_completion_before_dealloc</code><br/>from <code>105852</code> | <code>86362</code> | <code>31</code> |

tcgen05.load must complete before the allocation is deallocated

Effect lists and per-record flags: <code>output.json#/failed_required_orders</code>.

#### Diagnostics

##### access completion before deallocation not proven

**Saved code:** <code>access_completion_before_deallocation_not_proven</code> · **Stage:** <code>lifecycle_check</code>.

tcgen05.load must complete before the allocation is deallocated

**Diagnostic array indices:** <code>0</code>.

**Saved `details.count` values:** <code>3145728</code>.

##### paired access completion before deallocation not proven

**Saved code:** <code>paired_access_completion_before_deallocation_not_proven</code> · **Stage:** <code>lifecycle_check</code>.

all accesses to both paired TMEM allocations must complete before either warp attempts deallocation

**Diagnostic array indices:** <code>1</code>.

**Saved `details.count` values:** <code>6291456</code>.


<details>
<summary><strong>Exact diagnostic records and metadata</strong></summary>

##### Flags and empty collections

These retain their original top-level keys in `output.json`.

```json
{
  "warnings": [],
  "failed_required_orders_count": 9437184,
  "failed_required_orders_sample_limit": 32,
  "failed_required_orders_truncated": true,
  "source_violations": [],
  "verification_limitations": []
}
```

##### Diagnostics

<code>output.json#/diagnostics</code>

**Retained records:** 2.

```json
[
  {
    "code": "access_completion_before_deallocation_not_proven",
    "details": {
      "constraint": "access_completion_before_dealloc",
      "count": 3145728,
      "witnesses": [
        {
          "failed_target_count": 32,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 85888,
          "kind": "access_completion_before_dealloc",
          "message": "tcgen05.load must complete before the allocation is deallocated",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 32,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 86532,
          "kind": "access_completion_before_dealloc",
          "message": "tcgen05.load must complete before the allocation is deallocated",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 32,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 87176,
          "kind": "access_completion_before_dealloc",
          "message": "tcgen05.load must complete before the allocation is deallocated",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 32,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 87820,
          "kind": "access_completion_before_dealloc",
          "message": "tcgen05.load must complete before the allocation is deallocated",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 32,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 88464,
          "kind": "access_completion_before_dealloc",
          "message": "tcgen05.load must complete before the allocation is deallocated",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 32,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 89108,
          "kind": "access_completion_before_dealloc",
          "message": "tcgen05.load must complete before the allocation is deallocated",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 32,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 89752,
          "kind": "access_completion_before_dealloc",
          "message": "tcgen05.load must complete before the allocation is deallocated",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 32,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 90396,
          "kind": "access_completion_before_dealloc",
          "message": "tcgen05.load must complete before the allocation is deallocated",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 32,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 91040,
          "kind": "access_completion_before_dealloc",
          "message": "tcgen05.load must complete before the allocation is deallocated",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 32,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 91684,
          "kind": "access_completion_before_dealloc",
          "message": "tcgen05.load must complete before the allocation is deallocated",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 32,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 92328,
          "kind": "access_completion_before_dealloc",
          "message": "tcgen05.load must complete before the allocation is deallocated",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 32,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 92972,
          "kind": "access_completion_before_dealloc",
          "message": "tcgen05.load must complete before the allocation is deallocated",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 32,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 93616,
          "kind": "access_completion_before_dealloc",
          "message": "tcgen05.load must complete before the allocation is deallocated",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 32,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 94260,
          "kind": "access_completion_before_dealloc",
          "message": "tcgen05.load must complete before the allocation is deallocated",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 32,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 94904,
          "kind": "access_completion_before_dealloc",
          "message": "tcgen05.load must complete before the allocation is deallocated",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 32,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 95548,
          "kind": "access_completion_before_dealloc",
          "message": "tcgen05.load must complete before the allocation is deallocated",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 32,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 96192,
          "kind": "access_completion_before_dealloc",
          "message": "tcgen05.load must complete before the allocation is deallocated",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 32,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 96836,
          "kind": "access_completion_before_dealloc",
          "message": "tcgen05.load must complete before the allocation is deallocated",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 32,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 97480,
          "kind": "access_completion_before_dealloc",
          "message": "tcgen05.load must complete before the allocation is deallocated",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 32,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 98124,
          "kind": "access_completion_before_dealloc",
          "message": "tcgen05.load must complete before the allocation is deallocated",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 32,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 98768,
          "kind": "access_completion_before_dealloc",
          "message": "tcgen05.load must complete before the allocation is deallocated",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 32,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 99412,
          "kind": "access_completion_before_dealloc",
          "message": "tcgen05.load must complete before the allocation is deallocated",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 32,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 100056,
          "kind": "access_completion_before_dealloc",
          "message": "tcgen05.load must complete before the allocation is deallocated",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 32,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 100700,
          "kind": "access_completion_before_dealloc",
          "message": "tcgen05.load must complete before the allocation is deallocated",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 32,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 101344,
          "kind": "access_completion_before_dealloc",
          "message": "tcgen05.load must complete before the allocation is deallocated",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 32,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 101988,
          "kind": "access_completion_before_dealloc",
          "message": "tcgen05.load must complete before the allocation is deallocated",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 32,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 102632,
          "kind": "access_completion_before_dealloc",
          "message": "tcgen05.load must complete before the allocation is deallocated",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 32,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 103276,
          "kind": "access_completion_before_dealloc",
          "message": "tcgen05.load must complete before the allocation is deallocated",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 32,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 103920,
          "kind": "access_completion_before_dealloc",
          "message": "tcgen05.load must complete before the allocation is deallocated",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 32,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 104564,
          "kind": "access_completion_before_dealloc",
          "message": "tcgen05.load must complete before the allocation is deallocated",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 32,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 105208,
          "kind": "access_completion_before_dealloc",
          "message": "tcgen05.load must complete before the allocation is deallocated",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 32,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 105852,
          "kind": "access_completion_before_dealloc",
          "message": "tcgen05.load must complete before the allocation is deallocated",
          "to_instruction": 86362
        }
      ],
      "witnesses_truncated": true
    },
    "message": "tcgen05.load must complete before the allocation is deallocated",
    "stage": "lifecycle_check"
  },
  {
    "code": "paired_access_completion_before_deallocation_not_proven",
    "details": {
      "constraint": "tmem_group2_access_before_dealloc",
      "count": 6291456,
      "witnesses": [
        {
          "failed_target_count": 64,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 85888,
          "kind": "tmem_group2_access_before_dealloc",
          "message": "all accesses to both paired TMEM allocations must complete before either warp attempts deallocation",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 64,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 85890,
          "kind": "tmem_group2_access_before_dealloc",
          "message": "all accesses to both paired TMEM allocations must complete before either warp attempts deallocation",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 64,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 85892,
          "kind": "tmem_group2_access_before_dealloc",
          "message": "all accesses to both paired TMEM allocations must complete before either warp attempts deallocation",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 64,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 85894,
          "kind": "tmem_group2_access_before_dealloc",
          "message": "all accesses to both paired TMEM allocations must complete before either warp attempts deallocation",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 64,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 85896,
          "kind": "tmem_group2_access_before_dealloc",
          "message": "all accesses to both paired TMEM allocations must complete before either warp attempts deallocation",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 64,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 85898,
          "kind": "tmem_group2_access_before_dealloc",
          "message": "all accesses to both paired TMEM allocations must complete before either warp attempts deallocation",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 64,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 85906,
          "kind": "tmem_group2_access_before_dealloc",
          "message": "all accesses to both paired TMEM allocations must complete before either warp attempts deallocation",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 64,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 85908,
          "kind": "tmem_group2_access_before_dealloc",
          "message": "all accesses to both paired TMEM allocations must complete before either warp attempts deallocation",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 64,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 86042,
          "kind": "tmem_group2_access_before_dealloc",
          "message": "all accesses to both paired TMEM allocations must complete before either warp attempts deallocation",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 64,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 86044,
          "kind": "tmem_group2_access_before_dealloc",
          "message": "all accesses to both paired TMEM allocations must complete before either warp attempts deallocation",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 64,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 86046,
          "kind": "tmem_group2_access_before_dealloc",
          "message": "all accesses to both paired TMEM allocations must complete before either warp attempts deallocation",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 64,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 86048,
          "kind": "tmem_group2_access_before_dealloc",
          "message": "all accesses to both paired TMEM allocations must complete before either warp attempts deallocation",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 64,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 86056,
          "kind": "tmem_group2_access_before_dealloc",
          "message": "all accesses to both paired TMEM allocations must complete before either warp attempts deallocation",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 64,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 86058,
          "kind": "tmem_group2_access_before_dealloc",
          "message": "all accesses to both paired TMEM allocations must complete before either warp attempts deallocation",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 64,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 86060,
          "kind": "tmem_group2_access_before_dealloc",
          "message": "all accesses to both paired TMEM allocations must complete before either warp attempts deallocation",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 64,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 86062,
          "kind": "tmem_group2_access_before_dealloc",
          "message": "all accesses to both paired TMEM allocations must complete before either warp attempts deallocation",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 64,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 86196,
          "kind": "tmem_group2_access_before_dealloc",
          "message": "all accesses to both paired TMEM allocations must complete before either warp attempts deallocation",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 64,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 86198,
          "kind": "tmem_group2_access_before_dealloc",
          "message": "all accesses to both paired TMEM allocations must complete before either warp attempts deallocation",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 64,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 86206,
          "kind": "tmem_group2_access_before_dealloc",
          "message": "all accesses to both paired TMEM allocations must complete before either warp attempts deallocation",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 64,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 86208,
          "kind": "tmem_group2_access_before_dealloc",
          "message": "all accesses to both paired TMEM allocations must complete before either warp attempts deallocation",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 64,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 86210,
          "kind": "tmem_group2_access_before_dealloc",
          "message": "all accesses to both paired TMEM allocations must complete before either warp attempts deallocation",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 64,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 86212,
          "kind": "tmem_group2_access_before_dealloc",
          "message": "all accesses to both paired TMEM allocations must complete before either warp attempts deallocation",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 64,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 86214,
          "kind": "tmem_group2_access_before_dealloc",
          "message": "all accesses to both paired TMEM allocations must complete before either warp attempts deallocation",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 64,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 86216,
          "kind": "tmem_group2_access_before_dealloc",
          "message": "all accesses to both paired TMEM allocations must complete before either warp attempts deallocation",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 64,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 86532,
          "kind": "tmem_group2_access_before_dealloc",
          "message": "all accesses to both paired TMEM allocations must complete before either warp attempts deallocation",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 64,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 86534,
          "kind": "tmem_group2_access_before_dealloc",
          "message": "all accesses to both paired TMEM allocations must complete before either warp attempts deallocation",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 64,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 86536,
          "kind": "tmem_group2_access_before_dealloc",
          "message": "all accesses to both paired TMEM allocations must complete before either warp attempts deallocation",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 64,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 86538,
          "kind": "tmem_group2_access_before_dealloc",
          "message": "all accesses to both paired TMEM allocations must complete before either warp attempts deallocation",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 64,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 86540,
          "kind": "tmem_group2_access_before_dealloc",
          "message": "all accesses to both paired TMEM allocations must complete before either warp attempts deallocation",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 64,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 86542,
          "kind": "tmem_group2_access_before_dealloc",
          "message": "all accesses to both paired TMEM allocations must complete before either warp attempts deallocation",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 64,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 86550,
          "kind": "tmem_group2_access_before_dealloc",
          "message": "all accesses to both paired TMEM allocations must complete before either warp attempts deallocation",
          "to_instruction": 86362
        },
        {
          "failed_target_count": 64,
          "from_effects": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62
          ],
          "from_effects_count": 32,
          "from_effects_truncated": false,
          "from_instruction": 86552,
          "kind": "tmem_group2_access_before_dealloc",
          "message": "all accesses to both paired TMEM allocations must complete before either warp attempts deallocation",
          "to_instruction": 86362
        }
      ],
      "witnesses_truncated": true
    },
    "message": "all accesses to both paired TMEM allocations must complete before either warp attempts deallocation",
    "stage": "lifecycle_check"
  }
]
```

##### Error

<code>output.json#/error</code>

```json
{
  "code": "access_completion_before_deallocation_not_proven",
  "message": "tcgen05.load must complete before the allocation is deallocated"
}
```

##### Failed required orders

<code>output.json#/failed_required_orders</code>

**Retained records:** 32.

```json
{
  "same_records_as": "output.json#/diagnostics/0/details/witnesses",
  "stored_at": "output.json#/failed_required_orders",
  "size": "32 items",
  "canonical_json_bytes": 11403,
  "canonical_json_sha256": "771d60159f566c3e12f2b6d9e6e73e2a558e67efd8dcc5e0d2ae3dc62f352fd5"
}
```

</details>

</details>

<details>
<summary><strong>Assumptions — what qualifies the result</strong></summary>

Enabled and triggered answer different questions: whether a rule was selected, and whether its applicability was recorded. `triggered: null` means unknown or unmeasured, not inactive.

#### Duplicate TMA stores

**Key:** <code>duplicate_tma_store_benign</code>

**Enabled:** <code>true</code> · **Triggered:** <code>null</code>

Matching same-warp duplicate TMA stores are races in the model, but are reported as warnings under the assumption that their stored values agree.

#### Representative election

**Key:** <code>representative_election</code>

**Enabled:** <code>true</code> · **Triggered:** <code>true</code>

Election selects lane zero when participating, otherwise the lowest participating lane.

#### Failed launch cancellation

**Key:** <code>clc_always_fails</code>

**Enabled:** <code>true</code> · **Triggered:** <code>false</code>

Every cluster launch cancellation request fails.

#### Representative branches

**Key:** <code>representative_branches</code>

**Enabled:** <code>false</code> · **Triggered:** <code>false</code>

Choose fallthrough for an unresolved structured if/else; unknown ordinary instruction guards execute. Terminal control uses CFG classification. Known control and explicit choices take precedence.

#### Bounded computation loops

**Key:** <code>bounded_computation_loops</code>

**Enabled:** <code>false</code> · **Triggered:** <code>false</code>

Execute a configured number of body iterations for unresolved recognized computation loops, counted separately per invocation. Polling and unclassified loops are never forced to complete.

<details>
<summary><strong>Exact assumption settings</strong></summary>

```json
{
  "assumptions": [
    {
      "category": "analysis assumption",
      "enabled": true,
      "key": "duplicate_tma_store_benign",
      "label": "Duplicate TMA stores",
      "text": "Matching same-warp duplicate TMA stores are races in the model, but are reported as warnings under the assumption that their stored values agree.",
      "triggered": null
    },
    {
      "category": "analysis assumption",
      "enabled": true,
      "key": "representative_election",
      "label": "Representative election",
      "text": "Election selects lane zero when participating, otherwise the lowest participating lane.",
      "triggered": true
    },
    {
      "category": "analysis assumption",
      "enabled": true,
      "key": "clc_always_fails",
      "label": "Failed launch cancellation",
      "text": "Every cluster launch cancellation request fails.",
      "triggered": false
    },
    {
      "category": "analysis assumption",
      "enabled": false,
      "key": "representative_branches",
      "label": "Representative branches",
      "text": "Choose fallthrough for an unresolved structured if/else; unknown ordinary instruction guards execute. Terminal control uses CFG classification. Known control and explicit choices take precedence.",
      "triggered": false
    },
    {
      "category": "analysis assumption",
      "enabled": false,
      "key": "bounded_computation_loops",
      "label": "Bounded computation loops",
      "text": "Execute a configured number of body iterations for unresolved recognized computation loops, counted separately per invocation. Polling and unclassified loops are never forced to complete.",
      "triggered": false
    }
  ],
  "semantic_assumptions": {
    "bounded_computation_loops": false,
    "clc_always_fails": true,
    "duplicate_tma_store_benign": true,
    "representative_branches": false,
    "representative_election": true
  },
  "frontend_assumptions": {
    "assertion_failure_forms": [
      "trap",
      "__assertfail"
    ],
    "assertion_uses": 0,
    "bounded_loop_uses": 0,
    "clc_failure": true,
    "clc_uses": 0,
    "data_branch_decisions": [],
    "election_policy": {
      "fallback": "lowest_member",
      "preferred_lane": 0,
      "strategy": "preferred_member"
    },
    "election_uses": 104960,
    "fixed_election": true,
    "isolated_launch": true,
    "noalias_unbounded": false,
    "opaque_compute_regions": [],
    "pointer_contract": {
      "aliasing": "distinct_objects_noalias",
      "extent_bytes": "unbounded"
    },
    "representative_branch_uses": 0,
    "trap_only_equality_assertions_hold": true
  },
  "assumption_reporting": "triggered=null means applicability was not measured or analysis was not reached",
  "assumption_dependent": true
}
```

</details>

</details>

<details>
<summary><strong>Value analysis — independent status</strong></summary>

Memory safety and numerical equivalence are separate results.

**Value analysis:** <code>unsupported</code> · <code>output.json#/value_analysis</code>

**Checks:** Synchronization <code>not_requested</code> · Memory <code>not_requested</code> · Equivalence <code>not_requested</code>.

- **Prerequisite status:** <code>unsafe_lifecycle</code>
- **Scope:** {&quot;operation&quot;:&quot;<code>value_analysis</code>&quot;}
- **Diagnostic:** <code>prerequisite_check_not_passed</code> · <code>pipeline</code> · <code>output.json#/value_analysis/diagnostics/0</code>

**Equivalence:** <code>unsupported</code> · <code>output.json#/equivalence</code>

**Checks:** Synchronization <code>not_requested</code> · Memory <code>not_requested</code> · Equivalence <code>not_run</code>.

- **Scope:** {&quot;operation&quot;:&quot;equivalence&quot;}
- **Diagnostic:** <code>prerequisite_check_not_passed</code> · <code>pipeline</code> · <code>output.json#/equivalence/diagnostics/0</code>

<details>
<summary><strong>Exact value-analysis fields and large-data references</strong></summary>

##### Value analysis

```json
{
  "equivalence_check": "blocked",
  "prerequisite_status": "unsafe_lifecycle",
  "result": "unsupported",
  "checks": {
    "equivalence": "not_requested",
    "memory": "not_requested",
    "synchronization": "not_requested"
  },
  "diagnostics": {
    "artifact_reference": {
      "stored_at": "output.json#/value_analysis/diagnostics",
      "size": "1 items",
      "canonical_json_bytes": 24863,
      "canonical_json_sha256": "e868d1eec72ca56598d713dbf81a4dfdeedc7a46acec1ea3b1f14d3748bdc7c5"
    }
  },
  "scalar_fold": {
    "checks": {
      "equivalence": "not_requested",
      "memory": "not_requested",
      "synchronization": "not_requested"
    },
    "diagnostics": [
      {
        "code": "memory_elimination_not_completed",
        "stage": "pipeline"
      }
    ],
    "result": "unsupported",
    "scope": {
      "operation": "scalar_fold"
    }
  },
  "scope": {
    "operation": "value_analysis"
  }
}
```

##### Equivalence

```json
{
  "result": "unsupported",
  "checks": {
    "equivalence": "not_run",
    "memory": "not_requested",
    "synchronization": "not_requested"
  },
  "diagnostics": {
    "artifact_reference": {
      "stored_at": "output.json#/equivalence/diagnostics",
      "size": "1 items",
      "canonical_json_bytes": 24822,
      "canonical_json_sha256": "73e5d6669d1ee72aea58be0fba1df6cb9f75cd3551f67b2ee0f407fa7411a81e"
    }
  },
  "scope": {
    "operation": "equivalence"
  }
}
```

</details>

</details>

<details>
<summary><strong>Input configuration — reproduce this run</strong></summary>

**Launch and arguments define this run.** Tuple coordinates use `(x, y, z)`; grid and cluster dimensions count blocks, while block dimensions count threads.

| Launch quantity | Saved value |
| --- | --- |
| Grid · blocks | <code>[2,8,2]</code> |
| Block · threads | <code>[384,1,1]</code> |
| Cluster · blocks | <code>[2,1,1]</code> |
| Dynamic shared memory · bytes | <code>232448</code> |

**Arguments:** 13 retained entries. Exact parameter bindings, descriptors and entry name: <code>input.json#</code>.

<details>
<summary><strong>Exact input configuration</strong></summary>

```json
{
  "schema": "spverify.kernel-input/v1",
  "entry": "kernel_cutlass_kernel_flash_attncutesm100_hd256_2cta_fmha_backward_dqkernelBlackwellFusedMultiHeadAttentionBackwardDQKernel_object_at__TiledMMA_ThrLayoutVMNK21111000_PermutationMNK____MMA_0",
  "launch": {
    "grid_dim": [
      2,
      8,
      2
    ],
    "block_dim": [
      384,
      1,
      1
    ],
    "cluster_dim": [
      2,
      1,
      1
    ],
    "dynamic_shared_bytes": 232448
  },
  "arguments": [
    {
      "field": "shape.Q",
      "value": 128
    },
    {
      "field": "shape.K",
      "value": 256
    },
    {
      "field": "shape.query_heads_per_kv",
      "value": 1
    },
    {
      "field": "LSE.stride_head_local",
      "value": 128
    },
    {
      "field": "LSE.stride_head_kv",
      "value": 128
    },
    {
      "field": "LSE.stride_batch",
      "value": 1024
    },
    {
      "field": "sum_odo.stride_head_local",
      "value": 128
    },
    {
      "field": "sum_odo.stride_head_kv",
      "value": 128
    },
    {
      "field": "sum_odo.stride_batch",
      "value": 1024
    },
    {
      "field": "dQ.stride_sequence",
      "value": 2048
    },
    {
      "field": "dQ.stride_head_local",
      "value": 256
    },
    {
      "field": "dQ.stride_head_kv",
      "value": 256
    },
    {
      "field": "dQ.stride_batch",
      "value": 262144
    }
  ],
  "declarations": [
    {
      "name": "shape.Q",
      "parameter": 4,
      "offset": 0,
      "type": "u64"
    },
    {
      "name": "shape.Q",
      "parameter": 10,
      "offset": 0,
      "type": "u64"
    },
    {
      "name": "shape.Q",
      "parameter": 14,
      "offset": 0,
      "type": "u64"
    },
    {
      "name": "shape.K",
      "parameter": 6,
      "offset": 0,
      "type": "u64"
    },
    {
      "name": "shape.K",
      "parameter": 8,
      "offset": 0,
      "type": "u64"
    },
    {
      "name": "shape.query_heads_per_kv",
      "parameter": 4,
      "offset": 12,
      "type": "s32"
    },
    {
      "name": "shape.query_heads_per_kv",
      "parameter": 6,
      "offset": 12,
      "type": "s32"
    },
    {
      "name": "shape.query_heads_per_kv",
      "parameter": 8,
      "offset": 12,
      "type": "s32"
    },
    {
      "name": "shape.query_heads_per_kv",
      "parameter": 10,
      "offset": 12,
      "type": "s32"
    },
    {
      "name": "shape.query_heads_per_kv",
      "parameter": 14,
      "offset": 12,
      "type": "s32"
    },
    {
      "name": "shape.query_heads_per_kv",
      "parameter": 12,
      "offset": 16,
      "type": "s32"
    },
    {
      "name": "shape.query_heads_per_kv",
      "parameter": 15,
      "offset": 12,
      "type": "s32"
    },
    {
      "name": "shape.query_heads_per_kv",
      "parameter": 16,
      "offset": 16,
      "type": "s32"
    },
    {
      "name": "shape.query_heads_per_kv",
      "parameter": 17,
      "offset": 20,
      "type": "s32"
    },
    {
      "name": "LSE.stride_head_local",
      "parameter": 15,
      "offset": 24,
      "type": "s64"
    },
    {
      "name": "LSE.stride_head_kv",
      "parameter": 15,
      "offset": 32,
      "type": "s64"
    },
    {
      "name": "LSE.stride_batch",
      "parameter": 15,
      "offset": 40,
      "type": "s64"
    },
    {
      "name": "sum_odo.stride_head_local",
      "parameter": 16,
      "offset": 32,
      "type": "s64"
    },
    {
      "name": "sum_odo.stride_head_kv",
      "parameter": 16,
      "offset": 40,
      "type": "s64"
    },
    {
      "name": "sum_odo.stride_batch",
      "parameter": 16,
      "offset": 48,
      "type": "s64"
    },
    {
      "name": "dQ.stride_sequence",
      "parameter": 17,
      "offset": 32,
      "type": "s64"
    },
    {
      "name": "dQ.stride_head_local",
      "parameter": 17,
      "offset": 40,
      "type": "s64"
    },
    {
      "name": "dQ.stride_head_kv",
      "parameter": 17,
      "offset": 48,
      "type": "s64"
    },
    {
      "name": "dQ.stride_batch",
      "parameter": 17,
      "offset": 56,
      "type": "s64"
    },
    {
      "parameter": 3,
      "type": "tensor_map"
    },
    {
      "parameter": 5,
      "type": "tensor_map"
    },
    {
      "parameter": 7,
      "type": "tensor_map"
    },
    {
      "parameter": 9,
      "type": "tensor_map"
    },
    {
      "parameter": 11,
      "type": "tensor_map"
    },
    {
      "parameter": 13,
      "type": "tensor_map"
    },
    {
      "parameter": 15,
      "offset": 0,
      "type": "pointer",
      "object": "LSE"
    },
    {
      "parameter": 16,
      "offset": 0,
      "type": "pointer",
      "object": "sum_odo"
    },
    {
      "parameter": 17,
      "offset": 0,
      "type": "pointer",
      "object": "dQ"
    }
  ],
  "tma_descriptors": [
    {
      "parameter": 3,
      "offset": 0,
      "object": "Q",
      "element_type": "bf16",
      "global_shape": [
        256,
        128,
        1,
        8,
        2
      ],
      "coordinate_strides_bytes": [
        2,
        4096,
        512,
        512,
        524288
      ],
      "box_shape": [
        64,
        128,
        1,
        1,
        1
      ],
      "element_strides": [
        1,
        1,
        1,
        1,
        1
      ],
      "swizzle": "128B",
      "interleave": "none",
      "oob_fill": "zero"
    },
    {
      "parameter": 9,
      "offset": 0,
      "object": "dO",
      "element_type": "bf16",
      "global_shape": [
        256,
        128,
        1,
        8,
        2
      ],
      "coordinate_strides_bytes": [
        2,
        4096,
        512,
        512,
        524288
      ],
      "box_shape": [
        64,
        128,
        1,
        1,
        1
      ],
      "element_strides": [
        1,
        1,
        1,
        1,
        1
      ],
      "swizzle": "128B",
      "interleave": "none",
      "oob_fill": "zero"
    },
    {
      "parameter": 13,
      "offset": 0,
      "object": "dQ",
      "element_type": "bf16",
      "global_shape": [
        256,
        128,
        1,
        8,
        2
      ],
      "coordinate_strides_bytes": [
        2,
        4096,
        512,
        512,
        524288
      ],
      "box_shape": [
        64,
        128,
        1,
        1,
        1
      ],
      "element_strides": [
        1,
        1,
        1,
        1,
        1
      ],
      "swizzle": "128B",
      "interleave": "none",
      "oob_fill": "zero"
    },
    {
      "parameter": 5,
      "offset": 0,
      "object": "K",
      "element_type": "bf16",
      "global_shape": [
        256,
        256,
        8,
        2
      ],
      "coordinate_strides_bytes": [
        2,
        4096,
        512,
        1048576
      ],
      "box_shape": [
        64,
        64,
        1,
        1
      ],
      "element_strides": [
        1,
        1,
        1,
        1
      ],
      "swizzle": "128B",
      "interleave": "none",
      "oob_fill": "zero"
    },
    {
      "parameter": 7,
      "offset": 0,
      "object": "V",
      "element_type": "bf16",
      "global_shape": [
        256,
        256,
        8,
        2
      ],
      "coordinate_strides_bytes": [
        2,
        4096,
        512,
        1048576
      ],
      "box_shape": [
        64,
        64,
        1,
        1
      ],
      "element_strides": [
        1,
        1,
        1,
        1
      ],
      "swizzle": "128B",
      "interleave": "none",
      "oob_fill": "zero"
    },
    {
      "parameter": 11,
      "offset": 0,
      "object": "K",
      "element_type": "bf16",
      "global_shape": [
        256,
        256,
        8,
        2
      ],
      "coordinate_strides_bytes": [
        2,
        4096,
        512,
        1048576
      ],
      "box_shape": [
        64,
        128,
        1,
        1
      ],
      "element_strides": [
        1,
        1,
        1,
        1
      ],
      "swizzle": "128B",
      "interleave": "none",
      "oob_fill": "zero"
    }
  ],
  "assumptions": {
    "duplicate_tma_store_benign": true,
    "representative_election": true,
    "clc_always_fails": true
  }
}
```

</details>

</details>

<details>
<summary><strong>Provenance and coverage — check the saved run</strong></summary>

Use the original-file hashes to identify the run. Coverage flags describe what this result establishes; implementation statistics remain separate below.

**Coverage:** Memory analysis complete: **no** · Search complete: <code>not saved</code> · Exhaustive search: <code>not saved</code>.

**Assumption-dependent:** <code>true</code>. Triggered rules: Representative election.


#### Original artifacts

- **Input configuration:** [input.json](https://github.com/gpu-verifier/sentryward/blob/94a6b26b1990a4b9df7ded3727acc4b4cc36f23c/examples/flash_attention__bwd_hd256_bf16_2cta__sm_100a/input.json) · <code>examples/flash_attention__bwd_hd256_bf16_2cta__sm_100a/input.json</code> · SHA-256 <code>28e23ba704cc4cbf7dfaf4ae2c5806bed375f08823bbd195124c5905c766b47e</code>
- **Verifier output:** [output.json](https://github.com/gpu-verifier/sentryward/blob/94a6b26b1990a4b9df7ded3727acc4b4cc36f23c/examples/flash_attention__bwd_hd256_bf16_2cta__sm_100a/output.json) · <code>examples/flash_attention__bwd_hd256_bf16_2cta__sm_100a/output.json</code> · SHA-256 <code>d545f1b1e07cb4a8a22c4eab7a64d6a8346b4ef152a428a7aa186f8bb8d1aba4</code>
- **PTX:** [flash_attention__bwd_hd256_bf16_2cta__sm_100a.ptx](https://github.com/gpu-verifier/sentryward/blob/94a6b26b1990a4b9df7ded3727acc4b4cc36f23c/corpus/flash_attention__bwd_hd256_bf16_2cta__sm_100a.ptx) · <code>corpus/flash_attention__bwd_hd256_bf16_2cta__sm_100a.ptx</code> · SHA-256 <code>61e013adbe6bfa860cf68975481de3fbc3f640723f139616e11d2a4575f458ab</code>

Large structures are referenced by exact JSON pointer, item count and SHA-256 of their canonical JSON (UTF-8, sorted keys, compact separators). Original-artifact hashes above are hashes of the file bytes.

<details>
<summary><strong>Exact coverage, implementation and provenance fields</strong></summary>

```json
{
  "address_model": {
    "address_invariants": [],
    "kind": "symbolic",
    "layers": [
      0,
      2
    ],
    "solver": "z3",
    "solver_timeout_ms": 100
  },
  "checks": {
    "equivalence": "not_run",
    "memory": "not_run",
    "synchronization": "incomplete"
  },
  "counts": {
    "edges": 100933968,
    "events": 34995408,
    "instructions": 5745936,
    "threads": 12288
  },
  "frontend": {
    "address_value_bridge": {
      "diagnostics": [],
      "first_unavailable_reason": "",
      "queries": 0,
      "resolved": 0,
      "source": "value_lowering",
      "unavailable": 0,
      "value_view_complete": true
    },
    "cluster_completion_joins": 32,
    "cta_count": 32,
    "dependency_compression": {
      "bulk_read_wait": {
        "consumers": 4096,
        "emitted_references": 4096,
        "original_references": 10240,
        "prefix_effects": 0
      },
      "cp_async_mbarrier": {
        "consumers": 2048,
        "emitted_references": 7168,
        "original_references": 6144,
        "prefix_effects": 2048
      },
      "tcgen_commit": {
        "consumers": 240,
        "emitted_references": 1984,
        "original_references": 25856,
        "prefix_effects": 240
      },
      "tcgen_load_wait": {
        "consumers": 16384,
        "emitted_references": 131072,
        "original_references": 327680,
        "prefix_effects": 0
      },
      "tcgen_store_wait": {
        "consumers": 9216,
        "emitted_references": 16384,
        "original_references": 24576,
        "prefix_effects": 0
      }
    },
    "diagnostic_sample_count": 0,
    "discharged_codes": [
      "mma_memory_effect_refinement",
      "tcgen_pipeline_effect_refinement",
      "tcgen_commit_notification_order_refinement"
    ],
    "executed_records": 18230336,
    "frontend_assumptions": {
      "assertion_failure_forms": [
        "trap",
        "__assertfail"
      ],
      "assertion_uses": 0,
      "bounded_loop_uses": 0,
      "clc_failure": true,
      "clc_uses": 0,
      "data_branch_decisions": [],
      "election_policy": {
        "fallback": "lowest_member",
        "preferred_lane": 0,
        "strategy": "preferred_member"
      },
      "election_uses": 104960,
      "fixed_election": true,
      "isolated_launch": true,
      "noalias_unbounded": false,
      "opaque_compute_regions": [],
      "pointer_contract": {
        "aliasing": "distinct_objects_noalias",
        "extent_bytes": "unbounded"
      },
      "representative_branch_uses": 0,
      "trap_only_equality_assertions_hold": true
    },
    "group2_progress": {
      "blocked": [],
      "checked_pairs": 48,
      "premises": [
        "Baseline synchronization and asynchronous work must independently make progress.",
        "Resident peer CTAs must be scheduled fairly and have admitted TMEM capacity.",
        "Allocation placement and retirement RequiredOrders are separate obligations.",
        "Source references, actor ordering, and collective participation must be validated."
      ]
    },
    "implementation": "cpp",
    "input_objects": [
      {
        "bytes": 8,
        "object": "LSE",
        "offset": 0,
        "parameter": "kernel_cutlass_kernel_flash_attncutesm100_hd256_2cta_fmha_backward_dqkernelBlackwellFusedMultiHeadAttentionBackwardDQKernel_object_at__TiledMMA_ThrLayoutVMNK21111000_PermutationMNK____MMA_0_param_15",
        "source": "declaration"
      },
      {
        "bytes": 8,
        "object": "sum_odo",
        "offset": 0,
        "parameter": "kernel_cutlass_kernel_flash_attncutesm100_hd256_2cta_fmha_backward_dqkernelBlackwellFusedMultiHeadAttentionBackwardDQKernel_object_at__TiledMMA_ThrLayoutVMNK21111000_PermutationMNK____MMA_0_param_16",
        "source": "declaration"
      },
      {
        "bytes": 8,
        "object": "dQ",
        "offset": 0,
        "parameter": "kernel_cutlass_kernel_flash_attncutesm100_hd256_2cta_fmha_backward_dqkernelBlackwellFusedMultiHeadAttentionBackwardDQKernel_object_at__TiledMMA_ThrLayoutVMNK21111000_PermutationMNK____MMA_0_param_17",
        "source": "declaration"
      }
    ],
    "lowered_instruction_count": 5745936,
    "mma_refinement": {
      "blocked": [],
      "discharged": 1280
    },
    "reached_opcodes": {
      "add.rn.f32x2": 524288,
      "add.s32": 1363376,
      "add.s64": 4096,
      "and.b16": 73728,
      "and.b32": 396048,
      "and.pred": 26624,
      "bar.sync": 69632,
      "barrier.cluster.arrive": 12288,
      "barrier.cluster.arrive.relaxed": 12288,
      "barrier.cluster.wait": 24576,
      "bra": 411440,
      "bra.uni": 9232,
      "cp.async.bulk.commit_group": 4096,
      "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2": 640,
      "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint": 4096,
      "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2": 256,
      "cp.async.bulk.wait_group.read": 4096,
      "cp.async.ca.shared.global": 4096,
      "cp.async.mbarrier.arrive.noinc.shared.b64": 2048,
      "cvt.rn.bf16.f32": 1048576,
      "cvt.rn.bf16x2.f32": 524288,
      "cvt.s64.s32": 12288,
      "cvt.u16.u32": 592,
      "cvt.u32.u64": 20352,
      "cvt.u64.u32": 16272,
      "cvta.param.u64": 11136,
      "cvta.shared.u64": 896,
      "cvta.to.shared::cluster.u64": 896,
      "div.s32": 17408,
      "elect.sync": 104960,
      "ex2.approx.ftz.f32": 1048576,
      "fence.mbarrier_init.release.cluster": 49152,
      "fence.proxy.async.shared::cta": 20480,
      "fma.rn.f32x2": 524288,
      "ld.param.b32": 4096,
      "ld.param.b64": 184320,
      "ld.param.b8": 36864,
      "ld.param.v2.b32": 110592,
      "ld.param.v2.b8": 36864,
      "ld.shared.b32": 28672,
      "mad.lo.s32": 24576,
      "mad.lo.s64": 12288,
      "mapa.shared::cluster.u32": 17408,
      "mbarrier.arrive.expect_tx.shared.b64": 128,
      "mbarrier.arrive.shared.b64": 8192,
      "mbarrier.arrive.shared::cluster.b64": 29696,
      "mbarrier.init.shared.b64": 736,
      "mbarrier.try_wait.parity.shared.b64": 68608,
      "mov.b16": 1008,
      "mov.b32": 108976,
      "mov.b64": 5347744,
      "mov.pred": 23152,
      "mov.u32": 156800,
      "mul.f32": 4096,
      "mul.lo.s32": 6144,
      "mul.lo.s64": 4096,
      "mul.rn.f32x2": 1048576,
      "neg.f32": 16384,
      "neg.s32": 5120,
      "not.pred": 113152,
      "or.b32": 122960,
      "or.b64": 2704,
      "or.pred": 4096,
      "prefetch.tensormap": 6144,
      "rem.s32": 4096,
      "ret": 12288,
      "selp.b32": 25600,
      "selp.b64": 1040384,
      "setmaxnreg.dec.sync.aligned.u32": 10240,
      "setmaxnreg.inc.sync.aligned.u32": 4096,
      "setp.eq.b32": 12800,
      "setp.ge.s32": 532480,
      "setp.gt.s32": 10240,
      "setp.gt.u32": 12288,
      "setp.lt.s32": 43520,
      "setp.ne.b16": 73728,
      "setp.ne.b32": 349184,
      "shfl.sync.idx.b32": 58368,
      "shl.b32": 93600,
      "shl.b64": 4096,
      "shr.s32": 20480,
      "shr.u32": 344720,
      "st.shared.b16": 1048576,
      "st.shared.b32": 4096,
      "sub.s32": 35840,
      "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32": 1024,
      "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64": 240,
      "tcgen05.dealloc.cta_group::2.sync.aligned.b32": 1024,
      "tcgen05.ld.sync.aligned.32x32b.x16.b32": 131072,
      "tcgen05.ld.sync.aligned.32x32b.x32.b32": 131072,
      "tcgen05.mma.cta_group::2.kind::f16": 1280,
      "tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned": 1024,
      "tcgen05.st.sync.aligned.32x32b.x32.b32": 16384,
      "tcgen05.wait::ld.sync.aligned": 16384,
      "tcgen05.wait::st.sync.aligned": 9216,
      "xor.b32": 299600
    },
    "shared_constant_propagation": {
      "read_from_dependencies": 0,
      "representation": "sparse_known_bytes",
      "validation": "pending_backend"
    },
    "source_binding_storage": {
      "binding_templates": 400,
      "generated_address_rows": 0,
      "generated_body_rows": 0,
      "generated_memory_instances": 0,
      "logical_instructions": 5745936,
      "stored_bindings": 683344
    },
    "source_edge_storage": {
      "binding_templates": 27841,
      "logical_edges": 5733648,
      "stored_bindings": 27904
    },
    "tcgen_history_refinement": {
      "blocked": [],
      "notifications": 240,
      "operations": 1280
    },
    "thread_count": 12288,
    "unsupported_occurrence_counts": {},
    "user_branch_decisions": []
  },
  "hb_queries": {
    "required_orders": {
      "algorithm": "member-relations",
      "batches": 1659,
      "edge_visits": 75218312,
      "member_enumerations": 0,
      "request_construction": {
        "concrete_pair_materializations": 0,
        "endpoint_reads": 9536352,
        "family_requests": 434992,
        "logical_orders": 27119872,
        "owner_bindings": 26855216,
        "stored_orders": 568928
      }
    }
  },
  "implementation": "cpp",
  "memory_analysis_complete": false,
  "mma_operand_ordering": {},
  "mode": "verify",
  "peak_footprint_mib": 18557.578125,
  "peak_rss_mib": 18566.9453125,
  "pipeline_elapsed_seconds": 302.799368699,
  "required_order_domains": 568928,
  "required_orders": 27119872,
  "result": "invalid_lifecycle",
  "rss_limit_mib": null,
  "schema": "spverify.native-result/v1",
  "scope": {
    "operation": "kernel_verification"
  },
  "semantic_program": {
    "binding_diagnostic_count": 1280,
    "binding_diagnostics": {
      "artifact_reference": {
        "stored_at": "output.json#/semantic_program/binding_diagnostics",
        "size": "1280 items",
        "canonical_json_bytes": 142081,
        "canonical_json_sha256": "11587fcaf0bfb7ba09b30fe69dfa737a650e5ea3c8e47cde2a8d0a81e32016ed"
      }
    },
    "execution_cohorts": 704,
    "mapping_complete": false,
    "memory_binding_stored_rows": 8930704,
    "memory_binding_templates": 133616,
    "memory_bindings": 12924960,
    "memory_linking_templates_used": true,
    "synchronization_groups": 145760,
    "synchronization_groups_materialized": true,
    "synthetic_synchronization_nodes": 8768,
    "thread_prefix_groups": 4060688,
    "value_diagnostic_count": 0,
    "value_diagnostics": []
  },
  "stage": "lifecycle_check",
  "synchronization_input": {
    "compute_access_binding": "after_happens_before",
    "kind": "ordering_endpoints",
    "memory_address_payloads": false
  },
  "template_execution": {
    "completed_ctas": 32,
    "coordinate_bindings": 0,
    "logical_thread_steps": 18319120,
    "path_entries": 496624,
    "record_bindings": 773552,
    "record_nodes": 446720,
    "splits": 336,
    "template_steps": 453136,
    "thread_memberships": 12288,
    "value_nodes": 547618
  },
  "thread_record_compression": {
    "completed_ctas": 32,
    "enabled": true,
    "estimated_pool_bytes_sum": 0,
    "intern_calls": 0,
    "logical_records": 18230336,
    "record_handle_bytes": 32,
    "templates": 0,
    "unique_bindings": 0
  },
  "value_ir_instance_count": 63826496,
  "value_ir_node_count": 3563696,
  "value_view_complete": true,
  "wellsync_timeout_seconds": 500.0,
  "whole_pipeline_timeout_seconds": 500.0
}
```

</details>

</details>

<!-- verifier-evidence:end -->
