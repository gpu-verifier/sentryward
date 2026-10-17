<!-- report-layout:v2 -->

<!-- verifier-result:start -->
<p align="center">thunderkittens__thunderkittens_bf16_b200_gemm__sm_100a</p>
<h1 align="center">VERIFIED</h1>

**Inputs:** [PTX](https://github.com/gpu-verifier/sentryward/blob/94a6b26b1990a4b9df7ded3727acc4b4cc36f23c/corpus/thunderkittens__thunderkittens_bf16_b200_gemm__sm_100a.ptx) ([raw](https://github.com/gpu-verifier/sentryward/raw/94a6b26b1990a4b9df7ded3727acc4b4cc36f23c/corpus/thunderkittens__thunderkittens_bf16_b200_gemm__sm_100a.ptx)) · analyzed code; [input.json](https://github.com/gpu-verifier/sentryward/blob/94a6b26b1990a4b9df7ded3727acc4b4cc36f23c/examples/thunderkittens__thunderkittens_bf16_b200_gemm__sm_100a/input.json) · launch, arguments and assumptions.

**Result:** [output.json](https://github.com/gpu-verifier/sentryward/blob/94a6b26b1990a4b9df7ded3727acc4b4cc36f23c/examples/thunderkittens__thunderkittens_bf16_b200_gemm__sm_100a/output.json) · saved verifier findings.

<table>
<thead><tr><th align="left">VERIFIER RESULT · Recorded outcome</th></tr></thead>
<tbody><tr><td>

The verifier reports verification complete for the saved scope and checks below.

**Synchronization:** Passed · **Memory:** Passed · **Equivalence:** Passed

**Recorded totals:** 0 races · 0 warnings · 0 unresolved.

<details>
<summary><strong>Saved status and diagnostic codes</strong></summary>

**Saved result:** <code>verified</code> · <a href="https://github.com/gpu-verifier/sentryward/blob/94a6b26b1990a4b9df7ded3727acc4b4cc36f23c/examples/thunderkittens__thunderkittens_bf16_b200_gemm__sm_100a/output.json#L11081"><code>output.json#/result</code></a>.

**Operation:** <code>kernel_verification</code>.

**Checks:** Synchronization <code>passed</code> · Memory <code>passed</code> · Equivalence <code>passed</code>.

</details>

</td></tr>
<tr><td>

### Analysis scope

**Coverage:** Memory analysis complete: yes · Search complete: yes · Exhaustive search: **no**.

**Assumption-dependent:** <code>true</code>. Triggered rules: Representative election, Failed launch cancellation.

This result applies to the supplied launch, arguments and enabled assumptions.

<details>
<summary><strong>Analysis records — open the saved evidence</strong></summary>

Each record link opens its field in the original `output.json`. Model summaries are not full execution traces.

**Check the finding**

- [Diagnostic messages and details · `diagnostics`](https://github.com/gpu-verifier/sentryward/blob/94a6b26b1990a4b9df7ded3727acc4b4cc36f23c/examples/thunderkittens__thunderkittens_bf16_b200_gemm__sm_100a/output.json#L68)
- [Individual check outcomes · `checks`](https://github.com/gpu-verifier/sentryward/blob/94a6b26b1990a4b9df7ded3727acc4b4cc36f23c/examples/thunderkittens__thunderkittens_bf16_b200_gemm__sm_100a/output.json#L56)

**Check how the model was built**

- [PTX translation · `frontend`](https://github.com/gpu-verifier/sentryward/blob/94a6b26b1990a4b9df7ded3727acc4b4cc36f23c/examples/thunderkittens__thunderkittens_bf16_b200_gemm__sm_100a/output.json#L10765)
- [Model and source bindings · `semantic_program`](https://github.com/gpu-verifier/sentryward/blob/94a6b26b1990a4b9df7ded3727acc4b4cc36f23c/examples/thunderkittens__thunderkittens_bf16_b200_gemm__sm_100a/output.json#L11095)
- [Synchronization-check setup · `synchronization_input`](https://github.com/gpu-verifier/sentryward/blob/94a6b26b1990a4b9df7ded3727acc4b4cc36f23c/examples/thunderkittens__thunderkittens_bf16_b200_gemm__sm_100a/output.json#L11112)

**Check values separately**

- [Value-analysis status · `value_analysis`](https://github.com/gpu-verifier/sentryward/blob/94a6b26b1990a4b9df7ded3727acc4b4cc36f23c/examples/thunderkittens__thunderkittens_bf16_b200_gemm__sm_100a/output.json#L11140)
- [Numerical comparison · `equivalence`](https://github.com/gpu-verifier/sentryward/blob/94a6b26b1990a4b9df7ded3727acc4b4cc36f23c/examples/thunderkittens__thunderkittens_bf16_b200_gemm__sm_100a/output.json#L69)

For the records explained in this report, see the [Evidence appendix](#evidence-appendix).

</details>

</td></tr></tbody>
</table>
<!-- verifier-result:end -->

<!-- verifier-evidence:start -->
## Evidence appendix

Use these records to check the verifier result and source explanation above. JSON pointers refer to the original saved files. Identical prerequisite copies use `same_records_as` to refer to the diagnostic evidence shown once; their original locations and content hashes remain recorded.

<details>
<summary><strong>Diagnostic records — verify the finding</strong></summary>

These are saved verifier records. Truncation flags describe missing records in the original result, not merely collapsed presentation.

Inspect the exact records below for explicit empty, null and other saved diagnostic fields.

<details>
<summary><strong>Exact diagnostic records and metadata</strong></summary>

##### Flags and empty collections

These retain their original top-level keys in `output.json`.

```json
{
  "races": [],
  "warnings": [],
  "diagnostics": [],
  "source_violations": [],
  "verification_limitations": []
}
```

</details>

</details>

<details>
<summary><strong>Assumptions — what qualifies the result</strong></summary>

Enabled and triggered answer different questions: whether a rule was selected, and whether its applicability was recorded. `triggered: null` means unknown or unmeasured, not inactive.

#### Duplicate TMA stores

**Key:** <code>duplicate_tma_store_benign</code>

**Enabled:** <code>true</code> · **Triggered:** <code>false</code>

Matching same-warp duplicate TMA stores are races in the model, but are reported as warnings under the assumption that their stored values agree.

#### Representative election

**Key:** <code>representative_election</code>

**Enabled:** <code>true</code> · **Triggered:** <code>true</code>

Election selects lane zero when participating, otherwise the lowest participating lane.

#### Failed launch cancellation

**Key:** <code>clc_always_fails</code>

**Enabled:** <code>true</code> · **Triggered:** <code>true</code>

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
      "triggered": false
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
      "triggered": true
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
    "clc_uses": 262,
    "data_branch_decisions": [],
    "election_policy": {
      "fallback": "lowest_member",
      "preferred_lane": 0,
      "strategy": "preferred_member"
    },
    "election_uses": 224,
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

**Value analysis:** <code>verified</code> · <a href="https://github.com/gpu-verifier/sentryward/blob/94a6b26b1990a4b9df7ded3727acc4b4cc36f23c/examples/thunderkittens__thunderkittens_bf16_b200_gemm__sm_100a/output.json#L11140"><code>output.json#/value_analysis</code></a>

**Checks:** Synchronization <code>not_requested</code> · Memory <code>not_requested</code> · Equivalence <code>not_requested</code>.

- **Scope:** {&quot;operation&quot;:&quot;<code>value_analysis</code>&quot;}

**Equivalence:** <code>verified</code> · <a href="https://github.com/gpu-verifier/sentryward/blob/94a6b26b1990a4b9df7ded3727acc4b4cc36f23c/examples/thunderkittens__thunderkittens_bf16_b200_gemm__sm_100a/output.json#L69"><code>output.json#/equivalence</code></a>

**Checks:** Synchronization <code>not_requested</code> · Memory <code>not_requested</code> · Equivalence <code>passed</code>.

- **Numerical result:** verified
- **Scope:** {&quot;kind&quot;:&quot;<code>mapped_expression_equivalence</code>&quot;,&quot;operation&quot;:&quot;equivalence&quot;}
- **Semantics:** <code>ideal_real</code> numerical contract; exact word/index import; conditional on reported native premises, fixed input values and any <code>input_contract</code>
- **Equivalence contract:** <code>same_domain_and_equal_values</code>; total definedness reported separately
- **Prerequisites:** caller must combine numerical proof with synchronization, memory and scope results
- **Definedness:** <code>proved</code>

**Verification result:** <code>race_free</code> · <a href="https://github.com/gpu-verifier/sentryward/blob/94a6b26b1990a4b9df7ded3727acc4b4cc36f23c/examples/thunderkittens__thunderkittens_bf16_b200_gemm__sm_100a/output.json#L11270"><code>output.json#/verification_result</code></a>

<details>
<summary><strong>Exact value-analysis fields and large-data references</strong></summary>

##### Value analysis

```json
{
  "duplicate_write_value_checks": 0,
  "equivalence_check": "equivalent",
  "initial_memory_regions": 3440,
  "initial_read_families": 3440,
  "output_region_families": 0,
  "output_regions": 1408,
  "processed_memory_actions": 43245,
  "processed_memory_families": 0,
  "read_substitutions": 6670,
  "result": "verified",
  "shared_read_families": 8,
  "shared_read_instances": 496,
  "stored_read_substitutions": 6174,
  "checks": {
    "equivalence": "not_requested",
    "memory": "not_requested",
    "synchronization": "not_requested"
  },
  "diagnostics": [],
  "memory_action_storage": {
    "logical_actions": 43245,
    "stored_runs": 38733
  },
  "passes": [
    {
      "name": "packing",
      "rewritten_instances": 16384,
      "rewritten_nodes": 1248,
      "semantics": "exact_fixed_width"
    },
    {
      "abstraction": "matched approximation and clamp replaced by ideal Exp2",
      "name": "software_exp2",
      "rewritten_instances": 0,
      "rewritten_nodes": 0,
      "semantics": "ideal_exp2"
    },
    {
      "converted_instances": 88950,
      "name": "type_edges",
      "removed_nodes": 5722
    },
    {
      "name": "memory_elimination"
    },
    {
      "elements": 0,
      "identities": 286,
      "layouts": 1424,
      "mapping_blocks": 5520,
      "name": "view_normalization",
      "original_pieces": 30208,
      "reused_initial_reads": 3440,
      "reused_views": 0,
      "segmented": 1024,
      "slices": 0,
      "stored_mapping_blocks": 5520,
      "typed_output_pieces": 0,
      "typed_output_regions": 0
    }
  ],
  "scalar_fold": {
    "checks": {
      "equivalence": "not_requested",
      "memory": "not_requested",
      "synchronization": "not_requested"
    },
    "construction_collections": 0,
    "diagnostics": [],
    "output_count": 16384,
    "output_families_merged": 0,
    "output_family_count": 2,
    "output_root_count": 2,
    "peak_construction_nodes": 140,
    "result": "verified",
    "scope": {
      "operation": "scalar_fold"
    },
    "shared_view_projection_hits": 8,
    "source_outputs": 1408,
    "spatial_families_projected": 2,
    "spatial_outputs_projected": 16384,
    "spatial_template_operations": 0,
    "statistics": {
      "constants": 7,
      "edges": 68,
      "folds": 6,
      "indices": 3,
      "initial_cells": 6,
      "logical_iterations": 78,
      "logical_iterations_overflow": false,
      "max_fold_nesting": 2,
      "nodes": 47,
      "nodes_before_final_collection": 140,
      "nodes_reclaimed": 93,
      "operations": 25,
      "parameters": 0,
      "range_descriptions": 6,
      "storage_nodes": 47,
      "unique_fold_bodies": 6
    }
  },
  "scope": {
    "operation": "value_analysis"
  },
  "serialization": {
    "action_patterns": 38733,
    "binding_patterns": 38733,
    "domain_refinements": 119480,
    "elapsed_seconds": 0.322333257,
    "event_families": 65562,
    "expanded_event_records": 0,
    "largest_batch_members": 95,
    "logical_actions": 43245,
    "logical_events": 416474,
    "peak_rss_mib_at_completion": 314.5625,
    "reachability_queries": 0,
    "relations": 196583,
    "rss_mib_at_completion": 282.125,
    "scheduled_batches": 65562,
    "singleton_batches": 52144,
    "value_families": 34434
  }
}
```

##### Equivalence

```json
{
  "access_validity": "no_unresolved_symbolic_byte_reads",
  "arena_nodes": 11256,
  "comparison_elapsed_seconds": 0.081841123,
  "context_policy": "explicit native caller premises; no implied provenance certification",
  "elapsed_s": 0.076471227,
  "engine": "reference_guided_cpp",
  "equivalence_contract": "same_domain_and_equal_values; total definedness reported separately",
  "equivalent_families": 256,
  "finished_families": 256,
  "legacy_scalar_fallback": false,
  "native_context": null,
  "normalized_reachable_nodes": 513,
  "numerical_result": "verified",
  "pattern_nodes": 778,
  "peak_rss_kib": 322112,
  "peak_rss_scope": "process_lifetime_not_comparison_delta",
  "piece_nodes": 1792,
  "prerequisites": "caller must combine numerical proof with synchronization, memory and scope results",
  "reference_term_patterns": 2,
  "result": "verified",
  "scalar_engine": "mandatory_exponential_polynomial",
  "semantics": "ideal_real numerical contract; exact word/index import; conditional on reported native premises, fixed input values and any input_contract",
  "total_output_families": 256,
  "transport": "in_memory_cpp_objects",
  "checks": {
    "equivalence": "passed",
    "memory": "not_requested",
    "synchronization": "not_requested"
  },
  "context_declaration": {
    "assumptions": [],
    "fixed_inputs": [],
    "inputs": [],
    "predicate_nodes": []
  },
  "counters": {
    "add_term_visits": 25312,
    "affine_output_decline_singleton_projection_ambiguous": 256,
    "affine_state_banks": 1,
    "call_definition_node_visits": 2304,
    "canonical_factor_visits": 3072,
    "canonical_node_visits": 5121,
    "canonical_sum_initials": 256,
    "certified_state_pattern_cache_hits": 255,
    "certified_state_pattern_decline_no_certified_state": 1,
    "certified_state_pattern_work": 12,
    "coefficient_normalizations": 1,
    "completed_reductions": 256,
    "coordinate_digit_node_visits": 898,
    "coverage_polynomial_peak": 4,
    "coverage_polynomial_visits": 2304,
    "coverage_stored_piece_visits": 1024,
    "ep_body_anchor_edges": 1794,
    "ep_body_anchor_nodes": 2564,
    "ep_body_call_rebuilds": 256,
    "ep_body_deferred_call_anchors": 768,
    "ep_body_deferred_unmatched_anchors": 768,
    "ep_body_instance_cache_entries": 512,
    "ep_body_obligations": 768,
    "ep_body_pattern_anchor_cache_hits": 1022,
    "ep_body_proofs": 768,
    "ep_body_read_anchor_rejections": 256,
    "ep_body_target_anchor_cache_hits": 256,
    "ep_identity_proofs": 1024,
    "ep_scalar_attempts": 1024,
    "ep_scalar_proofs": 1024,
    "finite_bilinear_banks": 1,
    "finite_bilinear_fusions": 0,
    "finite_bilinear_masks": 0,
    "finite_bilinear_singleton_sums": 0,
    "finite_bilinear_swaps": 0,
    "finite_bilinear_work": 20480,
    "finite_state_reuse_banks": 1,
    "fp_idiom_power_constructions": 0,
    "fp_idiom_rm_producers": 0,
    "ideal_additive_compiler_exp_sequences": 0,
    "ideal_compiler_exp_sequences": 0,
    "ideal_compiler_log2_polynomials": 0,
    "ideal_compiler_lse_sequences": 0,
    "ideal_single_constant_exp_sequences": 0,
    "imported_context_nodes": 9996,
    "importer_scope_banks": 2,
    "importer_scope_binder_hits": 896,
    "importer_scope_binders_created": 640,
    "importer_scope_leaf_lookups": 1536,
    "importer_scope_nested_body_fallbacks": 512,
    "input_index_radix_quotients": 127,
    "intern_requests": 90689,
    "local_pattern_candidates": 1792,
    "masked_sum_prefix_queries": 1024,
    "normalization_cache_context_changes": 1,
    "normalization_node_visits": 5632,
    "packed_radix_probes": 127,
    "packed_radix_rewrites": 127,
    "parametric_state_attempts": 0,
    "pattern_build_peak_nodes": 3,
    "pattern_build_unique_nodes": 768,
    "pattern_node_visits": 5376,
    "select_max_probe_ancestor_nodes": 0,
    "select_max_probe_temporary_nodes": 0,
    "summary_bind_nodes": 768,
    "summary_concat_nodes": 256,
    "symbolic_byte_reads_without_extent_proof": 0
  },
  "definedness": {
    "partial_equivalence_families": 0,
    "proved_families": 256,
    "status": "proved"
  },
  "diagnostics": [],
  "ep_profile": {
    "body_instantiation_s": 0.0,
    "body_scalar_s": 0.0,
    "candidate_discovery_and_bookkeeping_s": 0.0,
    "cross_product_s": 0.0,
    "enabled": false,
    "equal_s": 0.0,
    "fraction": {
      "computations": 0,
      "previously_completed": 0,
      "repeated_across_queries": 0,
      "requests": 0,
      "shared_hits": 0,
      "unique_query_nodes": 0
    },
    "fraction_witness_s": 0.0,
    "match_total_s": 0.0,
    "normal": {
      "computations": 0,
      "previously_completed": 0,
      "repeated_across_queries": 0,
      "requests": 0,
      "shared_hits": 0,
      "unique_query_nodes": 0
    },
    "ordinary_polynomial_s": 0.0,
    "polynomial": {
      "computations": 0,
      "previously_completed": 0,
      "repeated_across_queries": 0,
      "requests": 0,
      "shared_hits": 0,
      "unique_query_nodes": 0
    },
    "query_count": 1024,
    "reset_s": 0.0,
    "reuse_cache": false,
    "shared_evictions": 0
  },
  "equivalence_context": {
    "applied_predicates": 0,
    "element_assumptions": 0,
    "fixed_input_elements": 0,
    "scoped_predicates": 0
  },
  "idealizations_applied": [],
  "input_contract_applied_reads": [],
  "mapping_input_memory_constraints": {},
  "piece_union": {
    "abstract_optimized_replacements": 0,
    "abstract_reference_replacements": 0,
    "abstraction": "schema-defined functions from sealed equalities; no additional input assumptions",
    "collected_pieces": 0,
    "late_certificates": "accepted_until_each_schema_is_complete",
    "method": "complete_output_equalities_with_finite_piece_union",
    "replacements": 0,
    "schemas": [],
    "stored_dag_work": 0,
    "work_limit": 2000000
  },
  "raw_nodes": [
    531,
    299
  ],
  "rows": {
    "stored_at": "output.json#/equivalence/rows",
    "size": "256 items",
    "canonical_json_bytes": 306649,
    "canonical_json_sha256": "2ac7e095bf42aec8df1b6aaee4ef376d62be3687f22a0878d948072075e8f60b"
  },
  "scope": {
    "kind": "mapped_expression_equivalence",
    "operation": "equivalence"
  },
  "stages_s": {
    "affine_state_pattern_s": 0.0027474630000000394,
    "exponential_polynomial_s": 0.0,
    "left_import_s": 0.016061456000000005,
    "left_normalize_s": 0.002880259999999994,
    "parametric_state_s": 2.3049999999953533e-05,
    "piece_union_record_s": 1.0260000000064293e-05,
    "piece_union_reuse_s": 0.0,
    "piece_union_seal_s": 6.450000000018182e-06,
    "proof_reuse_s": 0.0,
    "right_import_s": 0.039203014000000036,
    "right_normalize_s": 0.013904029999999939,
    "state_pattern_s": 0.0
  },
  "timings": {
    "load_and_scope_analysis": 0.000242371
  },
  "unmatched_reasons": {}
}
```

</details>

</details>

<details>
<summary><strong>Input configuration — reproduce this run</strong></summary>

**Launch and arguments define this run.** Tuple coordinates use `(x, y, z)`; grid and cluster dimensions count blocks, while block dimensions count threads.

| Launch quantity | Saved value |
| --- | --- |
| Grid · blocks | <code>[2,1,1]</code> |
| Block · threads | <code>[256,1,1]</code> |
| Cluster · blocks | <code>[2,1,1]</code> |
| Dynamic shared memory · bytes | <code>222208</code> |

**Arguments:** 6 retained entries. Exact parameter bindings, descriptors and entry name: <a href="https://github.com/gpu-verifier/sentryward/blob/94a6b26b1990a4b9df7ded3727acc4b4cc36f23c/examples/thunderkittens__thunderkittens_bf16_b200_gemm__sm_100a/input.json#L1"><code>input.json#</code></a>.

<details>
<summary><strong>Exact input configuration</strong></summary>

```json
{
  "schema": "spverify.kernel-input/v1",
  "entry": "_Z6kernelI6configILi256ELi64ELi128ELi4ELb1ELi5ELi2EEEv7globalsIT_E",
  "launch": {
    "grid_dim": [
      2,
      1,
      1
    ],
    "block_dim": [
      256,
      1,
      1
    ],
    "cluster_dim": [
      2,
      1,
      1
    ],
    "dynamic_shared_bytes": 222208
  },
  "arguments": [
    {
      "parameter": 0,
      "offset": 16,
      "type": "u64",
      "value": 256
    },
    {
      "parameter": 0,
      "offset": 24,
      "type": "u64",
      "value": 128
    },
    {
      "parameter": 0,
      "offset": 400,
      "type": "u64",
      "value": 64
    },
    {
      "parameter": 0,
      "offset": 408,
      "type": "u64",
      "value": 128
    },
    {
      "parameter": 0,
      "offset": 784,
      "type": "u64",
      "value": 256
    },
    {
      "parameter": 0,
      "offset": 792,
      "type": "u64",
      "value": 64
    }
  ],
  "declarations": [
    {
      "parameter": 0,
      "offset": 0,
      "type": "pointer",
      "object": "A"
    },
    {
      "parameter": 0,
      "offset": 384,
      "type": "pointer",
      "object": "B"
    },
    {
      "parameter": 0,
      "offset": 768,
      "type": "pointer",
      "object": "D"
    }
  ],
  "tma_descriptors": [
    {
      "parameter": 0,
      "offset": 128,
      "object": "A",
      "element_type": "bf16",
      "global_shape": [
        64,
        256,
        2,
        1,
        1
      ],
      "coordinate_strides_bytes": [
        2,
        256,
        128,
        65536,
        65536
      ],
      "box_shape": [
        64,
        128,
        2,
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
      "parameter": 0,
      "offset": 512,
      "object": "B",
      "element_type": "bf16",
      "global_shape": [
        64,
        64,
        2,
        1,
        1
      ],
      "coordinate_strides_bytes": [
        2,
        256,
        128,
        16384,
        16384
      ],
      "box_shape": [
        64,
        32,
        2,
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
      "parameter": 0,
      "offset": 896,
      "object": "D",
      "element_type": "bf16",
      "global_shape": [
        32,
        256,
        2,
        1,
        1
      ],
      "coordinate_strides_bytes": [
        2,
        128,
        64,
        32768,
        32768
      ],
      "box_shape": [
        32,
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
      "swizzle": "64B",
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

**Coverage:** Memory analysis complete: yes · Search complete: yes · Exhaustive search: **no**.

**Assumption-dependent:** <code>true</code>. Triggered rules: Representative election, Failed launch cancellation.


#### Original artifacts

- **Input configuration:** [input.json](https://github.com/gpu-verifier/sentryward/blob/94a6b26b1990a4b9df7ded3727acc4b4cc36f23c/examples/thunderkittens__thunderkittens_bf16_b200_gemm__sm_100a/input.json) · <code>examples/thunderkittens__thunderkittens_bf16_b200_gemm__sm_100a/input.json</code> · SHA-256 <code>4ef7a2faa91c72dc26c390e3dc38364d3bdde8ade73c5d1b78edb41d26261489</code>
- **Verifier output:** [output.json](https://github.com/gpu-verifier/sentryward/blob/94a6b26b1990a4b9df7ded3727acc4b4cc36f23c/examples/thunderkittens__thunderkittens_bf16_b200_gemm__sm_100a/output.json) · <code>examples/thunderkittens__thunderkittens_bf16_b200_gemm__sm_100a/output.json</code> · SHA-256 <code>fba85f59a0c556164f57c7215654f455ca30aa729be877b9b4bebeb656b9f790</code>
- **PTX:** [thunderkittens__thunderkittens_bf16_b200_gemm__sm_100a.ptx](https://github.com/gpu-verifier/sentryward/blob/94a6b26b1990a4b9df7ded3727acc4b4cc36f23c/corpus/thunderkittens__thunderkittens_bf16_b200_gemm__sm_100a.ptx) · <code>corpus/thunderkittens__thunderkittens_bf16_b200_gemm__sm_100a.ptx</code> · SHA-256 <code>2cd3cdc84dd51a8f58bf7b4a7d912107a8992cc11582102c4f4c9978e0210ad9</code>

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
    "equivalence": "passed",
    "memory": "passed",
    "synchronization": "passed"
  },
  "counts": {
    "edges": 414570,
    "events": 142602,
    "instructions": 29444,
    "memory_accesses": 91156,
    "threads": 512
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
    "cluster_completion_joins": 1,
    "cta_count": 2,
    "dependency_compression": {
      "bulk_read_wait": {
        "consumers": 512,
        "emitted_references": 0,
        "original_references": 0,
        "prefix_effects": 0
      },
      "tcgen_commit": {
        "consumers": 2,
        "emitted_references": 13,
        "original_references": 32,
        "prefix_effects": 2
      },
      "tcgen_load_wait": {
        "consumers": 256,
        "emitted_references": 1024,
        "original_references": 1024,
        "prefix_effects": 0
      }
    },
    "diagnostic_sample_count": 0,
    "discharged_codes": [
      "mma_memory_effect_refinement",
      "tcgen_pipeline_effect_refinement",
      "tcgen_commit_notification_order_refinement"
    ],
    "executed_records": 73725,
    "frontend_assumptions": {
      "assertion_failure_forms": [
        "trap",
        "__assertfail"
      ],
      "assertion_uses": 0,
      "bounded_loop_uses": 0,
      "clc_failure": true,
      "clc_uses": 262,
      "data_branch_decisions": [],
      "election_policy": {
        "fallback": "lowest_member",
        "preferred_lane": 0,
        "strategy": "preferred_member"
      },
      "election_uses": 224,
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
      "checked_pairs": 3,
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
        "object": "A",
        "offset": 0,
        "parameter": "_Z6kernelI6configILi256ELi64ELi128ELi4ELb1ELi5ELi2EEEv7globalsIT_E_param_0",
        "source": "declaration"
      },
      {
        "bytes": 8,
        "object": "B",
        "offset": 384,
        "parameter": "_Z6kernelI6configILi256ELi64ELi128ELi4ELb1ELi5ELi2EEEv7globalsIT_E_param_0",
        "source": "declaration"
      },
      {
        "bytes": 8,
        "object": "D",
        "offset": 768,
        "parameter": "_Z6kernelI6configILi256ELi64ELi128ELi4ELb1ELi5ELi2EEEv7globalsIT_E_param_0",
        "source": "declaration"
      }
    ],
    "lowered_instruction_count": 29444,
    "mma_refinement": {
      "blocked": [],
      "discharged": 8
    },
    "reached_opcodes": {
      "add.s32": 7737,
      "add.s64": 2075,
      "add.u64": 512,
      "and.b32": 3409,
      "and.b64": 2050,
      "bar.sync": 1792,
      "barrier.cluster.arrive.release.aligned": 512,
      "barrier.cluster.wait.acquire": 5,
      "barrier.cluster.wait.acquire.aligned": 256,
      "bra": 5171,
      "bra.uni": 262,
      "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128": 261,
      "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128": 1,
      "cp.async.bulk.commit_group": 4,
      "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint": 4,
      "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::cluster": 4,
      "cp.async.bulk.wait_group.read": 512,
      "createpolicy.fractional.L2::evict_first.b64": 4,
      "cvt.rn.bf16x2.f32": 8192,
      "cvt.u16.u32": 2,
      "cvt.u32.u64": 260,
      "cvt.u64.u32": 1090,
      "cvta.param.u64": 260,
      "cvta.shared.u64": 1088,
      "cvta.to.shared.u64": 260,
      "div.s32": 516,
      "elect.sync": 224,
      "fence.proxy.async.shared::cta": 5,
      "griddepcontrol.launch_dependents": 2,
      "griddepcontrol.wait": 2,
      "ld.local.b32": 257,
      "ld.param.b32": 512,
      "ld.param.v4.b32": 512,
      "ld.shared.b128": 261,
      "ld.shared.b32": 257,
      "mapa.shared::cluster.u32": 15,
      "mbarrier.arrive.expect_tx.shared::cta.b64": 3,
      "mbarrier.arrive.release.cta.shared::cta.b64": 2,
      "mbarrier.arrive.shared::cluster.b64": 11,
      "mbarrier.init.shared::cta.b64": 34,
      "mbarrier.try_wait.parity.shared::cta.b64": 843,
      "mov.b16": 2,
      "mov.b32": 4591,
      "mov.b64": 1024,
      "mov.u32": 1282,
      "mul.hi.s32": 1,
      "mul.hi.u32": 2,
      "mul.lo.s32": 519,
      "mul.wide.s32": 2,
      "mul.wide.u32": 257,
      "neg.s32": 1,
      "not.b32": 260,
      "or.b32": 764,
      "or.b64": 2,
      "prefetch.tensormap": 6,
      "ret": 512,
      "selp.b32": 258,
      "selp.u32": 485,
      "setmaxnreg.dec.sync.aligned.u32": 256,
      "setp.eq.b32": 999,
      "setp.eq.b64": 1536,
      "setp.eq.u32": 8,
      "setp.ge.s32": 258,
      "setp.lt.s32": 3,
      "setp.lt.u32": 2,
      "setp.ne.b32": 4098,
      "shl.b32": 2845,
      "shr.s32": 3335,
      "shr.u32": 4619,
      "shr.u64": 2,
      "st.local.v2.b32": 257,
      "stmatrix.sync.aligned.m8n8.x4.shared::cta.b16": 2048,
      "sub.s32": 779,
      "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32": 64,
      "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64": 2,
      "tcgen05.dealloc.cta_group::2.sync.aligned.b32": 64,
      "tcgen05.ld.sync.aligned.16x256b.x4.b32": 1024,
      "tcgen05.mma.cta_group::2.kind::f16": 8,
      "tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned": 64,
      "tcgen05.wait::ld.sync.aligned": 256,
      "xor.b32": 2051
    },
    "shared_constant_propagation": {
      "read_from_dependencies": 0,
      "representation": "sparse_known_bytes",
      "validation": "pending_backend"
    },
    "source_binding_storage": {
      "binding_templates": 22,
      "generated_address_rows": 0,
      "generated_body_rows": 0,
      "generated_memory_instances": 0,
      "logical_instructions": 29444,
      "stored_bindings": 2897
    },
    "source_edge_storage": {
      "binding_templates": 586,
      "logical_edges": 28932,
      "stored_bindings": 587
    },
    "tcgen_history_refinement": {
      "blocked": [],
      "notifications": 2,
      "operations": 8
    },
    "thread_count": 512,
    "unsupported_occurrence_counts": {},
    "user_branch_decisions": []
  },
  "hb_queries": {
    "memory": {
      "algorithm": "not-built",
      "batches": 0,
      "cache_hits": 0,
      "dfs_queries": 0,
      "edge_visits": 0,
      "filtered_queries": 0,
      "queries": 0,
      "shared_peak_index_bytes": 0
    },
    "required_orders": {
      "algorithm": "member-relations",
      "batches": 1,
      "edge_visits": 60053,
      "member_enumerations": 0,
      "request_construction": {
        "concrete_pair_materializations": 0,
        "endpoint_reads": 3904,
        "family_requests": 209,
        "logical_orders": 91849,
        "owner_bindings": 5729,
        "stored_orders": 1847
      }
    }
  },
  "implementation": "cpp",
  "memory_analysis": {
    "address_candidates": 203227,
    "address_model": {
      "cache_hits": 0,
      "ground_disjoint": 0,
      "ground_overlap": 0,
      "layer0_disjoint": 0,
      "refined_pairs": 0,
      "solver_queries": 0,
      "solver_sat": 0,
      "solver_unknown": 0,
      "solver_unsat": 0
    },
    "candidate_generation_fallback": false,
    "exhaustive": false,
    "family_analysis": {
      "access_families": 30412,
      "affine_member_rows": 0,
      "candidate_relations": 105905,
      "hb_frontier_pieces": 2757,
      "hb_member_enumerations": 0,
      "logical_accesses": 91156
    },
    "hb_algorithm": "member-relations",
    "hb_batches": 103,
    "hb_cache_hits": 0,
    "hb_dfs_queries": 0,
    "hb_edge_visits": 365989,
    "hb_filtered_queries": 0,
    "hb_peak_index_bytes": 0,
    "hb_queries": 203227,
    "hb_query_seconds": 0.07356076400000003,
    "overlap_queries": 0,
    "strategy": "conflict-family",
    "unordered_candidates_examined": 0
  },
  "memory_analysis_complete": true,
  "mma_operand_ordering": {},
  "mode": "verify",
  "peak_footprint_mib": 307.640625,
  "peak_rss_mib": 314.5625,
  "pipeline_elapsed_seconds": 1.917052914,
  "race_count": 0,
  "race_samples_truncated": false,
  "required_order_domains": 1847,
  "required_orders": 91849,
  "result": "verified",
  "rss_limit_mib": null,
  "schema": "spverify.native-result/v1",
  "scope": {
    "operation": "kernel_verification"
  },
  "search_complete": true,
  "semantic_program": {
    "binding_diagnostic_count": 0,
    "binding_diagnostics": [],
    "execution_cohorts": 38,
    "mapping_complete": true,
    "memory_binding_stored_rows": 38733,
    "memory_binding_templates": 3977,
    "memory_bindings": 43245,
    "memory_linking_templates_used": true,
    "synchronization_groups": 1024,
    "synchronization_groups_materialized": true,
    "synthetic_synchronization_nodes": 99,
    "thread_prefix_groups": 39064,
    "value_diagnostic_count": 0,
    "value_diagnostics": []
  },
  "synchronization_input": {
    "compute_access_binding": "after_happens_before",
    "kind": "ordering_endpoints",
    "memory_address_payloads": false
  },
  "template_execution": {
    "completed_ctas": 2,
    "coordinate_bindings": 0,
    "logical_thread_steps": 75192,
    "path_entries": 3039,
    "record_bindings": 2014,
    "record_nodes": 2244,
    "splits": 18,
    "template_steps": 2298,
    "thread_memberships": 512,
    "value_nodes": 517
  },
  "thread_record_compression": {
    "completed_ctas": 2,
    "enabled": true,
    "estimated_pool_bytes_sum": 0,
    "intern_calls": 0,
    "logical_records": 73725,
    "record_handle_bytes": 32,
    "templates": 0,
    "unique_bindings": 0
  },
  "unresolved_count": 0,
  "value_ir_instance_count": 173878,
  "value_ir_node_count": 29638,
  "value_view_complete": true,
  "verification_elapsed_seconds": 1.012114476,
  "warning_count": 0,
  "warning_samples_truncated": false,
  "wellsync_timeout_seconds": 500.0,
  "whole_pipeline_timeout_seconds": 498.58
}
```

</details>

</details>

<!-- verifier-evidence:end -->
