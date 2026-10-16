"""Saved-result rendering tests; no model or verifier invocation."""
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'skills/generate-kernel-report/scripts/result_view.py'
SPEC = importlib.util.spec_from_file_location('result_view', SCRIPT)
view = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(view)


def saved_case(name):
    case = ROOT / 'examples' / name
    paths = [case / 'input.json', case / 'output.json', ROOT / 'corpus' / (name + '.ptx')]
    return json.loads(paths[0].read_text()), json.loads(paths[1].read_text()), paths[2].read_text() if paths[2].is_file() else '', paths


def reconstruct_records(packed):
    def sequence(encoded):
        if 'values' in encoded:
            return encoded['values']
        start, step, count = encoded['range']
        return [start + index * step for index in range(count)]
    result = [None] * packed['record_count']
    used = set()
    for group in packed['groups']:
        positions = sequence(group['indices'])
        fields = {key: sequence(value) for key, value in group['fields'].items()}
        assert all(len(values) == len(positions) for values in fields.values())
        for offset, position in enumerate(positions):
            assert position not in used
            used.add(position)
            result[position] = dict(packed['shared'], **group['shared'], **{key: values[offset] for key, values in fields.items()})
            if packed['representation'] == 'indexed-record-groups/v2':
                nested = {}
                for pointer, value in result[position].items():
                    keys = [key.replace('~1', '/').replace('~0', '~') for key in pointer.split('/')[1:]]
                    target = nested
                    for key in keys[:-1]:
                        target = target.setdefault(key, {})
                    target[keys[-1]] = value
                result[position] = nested
    assert used == set(range(len(result)))
    return result


def collapsed_view(text):
    """Keep the initial reading path, excluding all closed disclosure bodies."""
    visible, depth = [], 0
    for part in re.split(r'(</?details>)', text):
        if part == '<details>':
            depth += 1
        elif part == '</details>':
            depth -= 1
        elif depth == 0:
            visible.append(part)
    return ''.join(visible)


class ResultViewTests(unittest.TestCase):
    def test_named_assumptions_do_not_require_legacy_ids(self):
        inp, result, ptx, paths = saved_case('cutlass__dense-gemm-f16-tma-1sm__sm_100a')
        rule = next(item for item in result['assumptions'] if item['key'] == 'duplicate_tma_store_benign')
        self.assertNotIn('id', rule)
        evidence = view.render_evidence(inp, result, ptx, paths)
        self.assertIn('#### Duplicate TMA stores', evidence)
        self.assertNotIn('#### <code>not saved</code>', evidence)
        self.assertIn(rule['text'], evidence)
        visible = view._coverage(result)
        self.assertIn('Duplicate TMA stores', visible)
        self.assertNotIn('duplicate tma store benign', visible)

    def test_repeated_prerequisite_evidence_is_exactly_referenced(self):
        records = [{'instruction': i, 'message': 'Required completion before release'} for i in range(8)]
        result = {'diagnostics': [{'details': {'witnesses': records}}]}
        original = json.dumps(result)
        rendered = view._reference_repeated_diagnostics(
            {'same': records, 'different': records + [{'instruction': 9}]}, '/equivalence', result)
        self.assertEqual(rendered['same']['same_records_as'], 'output.json#/diagnostics/0/details/witnesses')
        self.assertEqual(rendered['same']['stored_at'], 'output.json#/equivalence/same')
        self.assertEqual(rendered['same']['canonical_json_sha256'], hashlib.sha256(view._encoded(records).encode()).hexdigest())
        self.assertEqual(rendered['different'], records + [{'instruction': 9}])
        self.assertEqual(json.dumps(result), original)

    def test_gqa_witnesses_remain_once_with_duplicate_locations(self):
        inp, result, ptx, paths = saved_case('cutlass93_blackwell_low_latency_gqa.compute_100a')
        text = view.render_evidence(inp, result, ptx, paths)
        blocks = [json.loads(raw) for raw in re.findall(r'```json\n(.*?)\n```', text, re.S)]
        self.assertIn(result['diagnostics'], blocks)
        self.assertNotIn(result['failed_required_orders'], blocks)
        self.assertIn('output.json#/failed_required_orders', text)
        self.assertIn('output.json#/diagnostics/0/details/witnesses', text)
        self.assertIn('output.json#/equivalence/diagnostics', text)

    def test_completed_analysis_does_not_imply_complete_search(self):
        text = self.render({'result': 'potential_race', 'checks': {'synchronization': 'passed', 'memory': 'failed'},
                            'memory_analysis_complete': True, 'search_complete': False})
        self.assertIn('separate memory check still found a conflict', text)
        self.assertIn('do not establish that all possible conflicts were examined', text)

    def test_copy_model_limit_is_not_presented_as_hardware_requirement(self):
        result = {'diagnostics': [
            {'stage': 'event_view_construction', 'code': 'async_copy_source_control_not_static', 'message': 'Operand must have a statically determined integer value'},
            {'stage': 'event_view_construction', 'code': 'missing_tma_descriptor', 'message': 'tensor operand has no matching captured map'}]}
        text = view._primary_diagnostic(result, '', [])
        self.assertIn('Model needs a fixed copy-control value', text)
        self.assertIn('Operand must have a statically determined integer value', text)
        self.assertIn('Tensor-map capture missing', text)

    def render(self, result, ptx='', artifacts=()):
        return view.render_result({}, result, ptx, artifacts, 'Example')

    def test_base_title_uses_kernel_artifact_not_staging_directory(self):
        text = view.render_result({}, {'status':'race_free'}, '',
                                  ['raw/input.json', 'raw/output.json', 'corpus/matmul.ptx'])
        self.assertIn('<p align="center">matmul</p>', text)

    def test_file_links_resolve_from_report_and_encode_filename(self):
        from urllib.parse import unquote
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / 'raw inputs' / 'kernel #1.ptx'
            original.parent.mkdir()
            original.write_text('.entry test(){}')
            report = root / 'docs/sample/report.md'
            link = view._file_link('PTX', original, report)
            self.assertEqual(link, '[PTX](../../raw%20inputs/kernel%20%231.ptx)')
            target = link.partition('](')[2][:-1]
            self.assertEqual((report.parent / unquote(target)).resolve(), original.resolve())
            missing = view._file_link('Input', root / 'absent.json', report)
            self.assertIn('file unavailable', missing)
            self.assertNotIn('](', missing)

    def test_intermediate_index_only_names_saved_records(self):
        text = view.artifact_entry({'semantic_program':{}, 'source_events':[]},
                                  ['input.json', 'output.json', 'kernel.ptx'])
        self.assertIn('output.json#/semantic_program', text)
        self.assertIn('output.json#/source_events', text)
        self.assertNotIn('output.json#/counterexample', text)
        self.assertNotIn('equivalence.json', text)
        self.assertIn('source line unavailable', text)
        self.assertNotIn('#L', text)

    def test_record_links_point_to_original_top_level_lines(self):
        from urllib.parse import unquote
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / 'saved result.json'
            raw = '{\r\n  "nested": {"frontend": "quoted } and \\\""},\r\n  "frontend": {},\r\n  "semantic_program": {"value_analysis": []},\r\n  "value_analysis": null\r\n}'
            out.write_bytes(raw.encode())
            report = root / 'docs/report.md'
            text = view.artifact_entry(json.loads(raw), [None, out, None], report)
            links = re.findall(r'\]\(([^)]+)#L(\d+)\)', text)
            self.assertEqual(len(links), 3)
            for target, line in links:
                self.assertEqual((report.parent / unquote(target)).resolve(), out.resolve())
                self.assertRegex(raw.splitlines()[int(line) - 1], r'^  "(?:frontend|semantic_program|value_analysis)":')
            self.assertEqual(view._json_field_lines(out),
                             {'nested': 2, 'frontend': 3, 'semantic_program': 4, 'value_analysis': 5})

    def test_compact_and_escaped_keys_use_original_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'output.json'
            out.write_text('{"front\\u0065nd": {}, "semantic_program": {"frontend": 1}}')
            self.assertEqual(view._json_field_lines(out), {'frontend': 1, 'semantic_program': 1})

    def test_record_links_use_pinned_blob_not_json_pointer_fragments(self):
        inp, result, ptx, paths = saved_case('cutlass93_blackwell_low_latency_gqa.compute_100a')
        base = 'https://github.com/gpu-verifier/sentryward/blob/frozen'
        text = view.artifact_entry(result, paths, ROOT / 'docs/report.md', artifact_base=base)
        target = base + '/examples/' + paths[1].parent.name + '/output.json'
        self.assertIn('[Failed lifecycle orders · `failed_required_orders`](' + target + '#L', text)
        self.assertNotIn(target + '#/', text)
        for fragment in re.findall(re.escape(target) + r'#L(\d+)', text):
            line = paths[1].read_text().splitlines()[int(fragment) - 1]
            self.assertRegex(line, r'^  "[a-z_]+":')

    def test_pinned_artifact_links_preserve_repository_path(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); file=root/'raw #1.json'; file.write_text('{}')
            with patch.object(view.Path, 'cwd', return_value=root):
                base='https://github.com/owner/repo/blob/0123456'
                self.assertEqual(view._file_link('Result',file,root/'out/report.md',base),
                                 '[Result](https://github.com/owner/repo/blob/0123456/raw%20%231.json)')
                with self.assertRaises(ValueError):
                    view._file_link('Result',file,artifact_base='javascript:alert(1)')

    def test_warning_is_not_substituted_for_missing_race_pair(self):
        warning = {'first':{'event':7},'second':{'event':8}}
        text = self.render({'status':'race_candidate', 'races':[], 'warnings':[warning]})
        self.assertIn('no usable retained pair', text)
        self.assertNotIn('First access', text)
        warning_text = self.render({'status':'race_free', 'races':[warning], 'warnings':[warning]})
        self.assertIn('Warning to inspect', warning_text)
        self.assertIn('output.json#/warnings/0', warning_text)
        self.assertNotIn('output.json#/races/0', warning_text)

    def test_every_saved_result_has_a_non_default_outcome_and_exact_checks(self):
        seen = set()
        for path in sorted((ROOT / 'examples').glob('*/output.json')):
            result = json.loads(path.read_text())
            seen.add(result['result'])
            text = self.render(result)
            self.assertNotIn('RESULT NOT RECOGNIZED', text, path.parent.name)
            expected = 'LIFECYCLE CHECK FAILED' if result['result'] == 'invalid_lifecycle' and not any(k in result for k in ('failed_required_orders', 'failed_required_orders_count')) else view.RESULTS[result['result']][0]
            self.assertIn(expected, text)
            self.assertIn('**Saved result:** <code>' + result['result'] + '</code>', text)
            for key, label in [('synchronization', 'Synchronization'), ('memory', 'Memory'), ('equivalence', 'Equivalence')]:
                self.assertIn(label + ' <code>' + result['checks'][key] + '</code>', text)
            self.assertIn(view.FACTS_START, text)
            self.assertIn(view.FACTS_END, text)
        self.assertEqual(len(seen), 8)
        contract = json.loads((ROOT / 'config/result-categories.json').read_text())
        self.assertEqual(set(view.RESULTS), set(contract['results']))
        for status in set(view.RESULTS) - seen:
            text = self.render({'result': status})
            self.assertIn(view.RESULTS[status][0], text)
        for status in view.STATUSES:
            self.assertIn(view.STATUSES[status][0], self.render({'status': status}))

    def test_unknown_null_and_missing_status_never_become_success(self):
        for result in [{}, {'status': None}, {'status': 'new_backend_error'}, {'status': False}, {'status': ['race_free']}]:
            with self.subTest(result=result):
                text = self.render(result)
                self.assertIn('RESULT NOT RECOGNIZED', text)
                self.assertNotIn('NO RACES REPORTED', text)
                self.assertIn('No success verdict is inferred', text)

    def test_unknown_unified_result_never_falls_back_to_legacy_success(self):
        for value in (None, False, ['verified'], 'future_result', 'recovered_race_free'):
            with self.subTest(value=value):
                text = self.render({'result': value, 'status': 'race_free'})
                self.assertIn('RESULT NOT RECOGNIZED', text)
                self.assertNotIn('<h1 align="center">NO RACES REPORTED', text)

    def test_missing_null_and_unknown_checks_are_not_successes(self):
        for value in (None, {}, {'memory': None, 'synchronization': 'future_check', 'equivalence': False}):
            text = self.render({'result': 'verified', 'checks': value})
            self.assertIn('**Scope:** not saved', text)
            self.assertNotIn('Memory <code>passed</code>', text)
            self.assertNotIn('no races for this', text)
        text = self.render({'result': 'verified', 'checks': {'memory': 'passed'}, 'scope': {'operation': 'value_analysis'}})
        self.assertIn('Synchronization <code>not saved</code>', text)
        self.assertIn('**Operation:** <code>value_analysis</code>', text)

    def test_unified_diagnostic_location_and_evidence_are_preserved(self):
        result = {'result': 'unsupported', 'checks': {'memory': 'not_run'}, 'diagnostics': [
            {'code': 'opcode_unsupported', 'stage': 'frontend', 'message': 'Unsupported instruction',
             'details': {'location': {'file': 'kernel.ptx', 'line': 2}, 'operand': '%r1'}}]}
        text = self.render(result, '.entry test(){\nret;\n}', ['input.json', 'output.json', 'kernel.ptx'])
        for token in ('opcode_unsupported', 'frontend', 'Unsupported instruction', 'physical PTX 2', 'output.json#/diagnostics/0'):
            self.assertIn(token, text)
        evidence = view.render_evidence({}, result, '', [])
        blocks = [json.loads(raw) for raw in re.findall(r'```json\n(.*?)\n```', evidence, re.S)]
        self.assertIn(result['diagnostics'], blocks)

    def test_sparse_mla_exposes_all_categories_without_invented_failed_orders(self):
        inp, result, ptx, paths = saved_case('flash_attention__fwd_mla_sparse_bf16__sm_100a')
        text = view.render_result(inp, result, ptx, paths)
        visible = collapsed_view(text)
        self.assertIn('4 diagnostic categories · 87 retained records', visible)
        for record in result['diagnostics']:
            self.assertIn(view._text(record['message']), visible)
        self.assertIn('17 reported PTX sites', visible)
        self.assertIn('68 reported PTX sites', visible)
        self.assertNotIn('First diagnostic:', visible)
        self.assertNotIn('**Failed orders:**', visible)
        self.assertNotIn('happens-before relation is absent', visible)
        evidence = view.render_evidence(inp, result, ptx, paths)
        self.assertEqual(evidence.count('##### missing tma descriptor'), 1)
        blocks = [json.loads(raw) for raw in re.findall(r'```json\n(.*?)\n```', evidence, re.S)]
        self.assertIn(result['diagnostics'], blocks)

    def test_paired_tmem_does_not_hide_second_requirement_behind_first_sample(self):
        inp, result, ptx, paths = saved_case('flash_attention__bwd_hd256_bf16_2cta__sm_100a')
        full = view.render_result(inp, result, ptx, paths)
        text = collapsed_view(full)
        self.assertIn('all accesses to both paired TMEM allocations', text)
        self.assertIn('3,145,728', text)
        self.assertIn('6,291,456', text)
        self.assertIn('32 orders retained out of 9,437,184 reported', text)
        self.assertEqual(text.count('32 witnesses retained'), 2)
        self.assertIn('All accesses to both<br/>allocations complete', text)
        self.assertIn('Either warp attempts<br/>deallocation', text)
        self.assertIn('output.json#/diagnostics/1/details', full)
        self.assertIn('tmem_group2_access_before_dealloc', full)

    def test_lifecycle_categories_do_not_invent_missing_witnesses_or_totals(self):
        result = {'result': 'invalid_lifecycle', 'diagnostics': [
            {'stage': 'lifecycle_check', 'message': 'first', 'details': {'constraint': 'access_completion_before_dealloc'}},
            {'stage': 'lifecycle_check', 'message': 'second', 'details': {'constraint': 'tmem_group2_access_before_dealloc', 'witnesses': None}}]}
        text = self.render(result)
        self.assertNotIn('0 witnesses retained', text)
        self.assertNotIn('Top-level sample', text)
        self.assertIn('<code>not saved</code> witnesses retained', text)
        self.assertIn('<code>null</code> witnesses retained', text)

    def test_diagnostic_categories_keep_stages_variants_and_record_indices(self):
        records = [{'code': 'same_code', 'stage': 'first', 'message': 'a | b', 'details': {'ptx_line': 4}},
                   {'code': 'same_code', 'stage': 'first', 'message': 'different message', 'details': {'ptx_line': 7}},
                   {'code': 'same_code', 'stage': 'second', 'message': 'another stage'}]
        groups = view._diagnostic_groups(records)
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[('first', 'same_code')]['indices'], [0, 1])
        self.assertEqual(len(groups[('first', 'same_code')]['messages']), 2)
        self.assertIn('a &#124; b', view._diagnostic_overview(groups))
        self.assertIn('2 message variants', view._diagnostic_overview(groups))

    def test_numerical_stop_keeps_memory_warning_as_a_separate_finding(self):
        inp, result, ptx, paths = saved_case('flash_attention__fwd_paged_bf16_h128__sm_100a')
        full = view.render_result(inp, result, ptx, paths)
        text = collapsed_view(full)
        for token in ('input_alignment_not_proven', 'Memory <code>passed</code>', 'Equivalence <code>incomplete</code>',
                      'Memory warning alongside the stopped check', '10,919,936', 'PTX 1421'):
            self.assertIn(token, text)
        self.assertIn('output.json#/warnings/0', full)
        self.assertNotIn('race candidate to inspect', text)

    def test_integrated_equivalence_failure_does_not_hide_passed_memory(self):
        inp, result, ptx, paths = saved_case('tirx-flashinfer-quantization-nvfp4')
        text = self.render(result)
        self.assertIn('ANALYSIS UNSUPPORTED', text)
        self.assertIn('Memory <code>passed</code>', text)
        self.assertIn('Equivalence <code>incomplete</code>', text)
        self.assertIn('saturation_input_finiteness_not_proven', text)
        self.assertNotIn('NO RACES REPORTED</h1>', text)
        evidence = view.render_evidence(inp, result, ptx, paths)
        self.assertIn('**Value analysis:** <code>verified</code>', evidence)
        self.assertIn('**Equivalence:** <code>unsupported</code>', evidence)

    def test_contradictory_nested_success_is_visible_without_replacing_result(self):
        result = {'result': 'potential_race', 'checks': {'synchronization': 'passed', 'memory': 'failed', 'equivalence': 'not_run'},
                  'equivalence': {'result': 'verified', 'checks': {'equivalence': 'passed'}}}
        text = self.render(result)
        self.assertIn('RACE CANDIDATE', text)
        self.assertIn('**Saved fields disagree:**', text)
        self.assertIn('output.json#/equivalence/checks/equivalence', text)
        self.assertNotIn('<h1 align="center">VERIFIED', text)
        result['result'] = 'verified'
        text = self.render(result)
        self.assertIn('output.json#/checks/memory', text)
        self.assertIn('does not resolve the disagreement', text)

    def test_recovery_scope_is_visible_and_original_violations_retained(self):
        result = {'result': 'race_free', 'analysis_target': 'recovered_program', 'recovery_policy': 'deferred_tmem_deallocation',
                  'recovered_source_violations': [{'code': 'tmem_multiple_dealloc', 'message': 'Multiple deallocations'}]}
        text = self.render(result)
        self.assertIn('verdict describes the recovered model', text)
        self.assertIn('deferred_tmem_deallocation', text)
        evidence = view.render_evidence({}, result, '', [])
        blocks = [json.loads(raw) for raw in re.findall(r'```json\n(.*?)\n```', evidence, re.S)]
        self.assertIn(result['recovered_source_violations'], blocks)

    def test_nested_diagnostic_compression_round_trips_escaped_field_names(self):
        records = [{'code': 'test', 'details': {'source/line~id': i * i, 'empty': {}, 'message': 'same'}} for i in range(128)]
        packed = view._lossless_groups(records, '/diagnostics')
        self.assertEqual(packed['representation'], 'indexed-record-groups/v2')
        self.assertEqual(reconstruct_records(packed), records)
        self.assertEqual(packed['canonical_json_sha256'], hashlib.sha256(view._encoded(records).encode()).hexdigest())

    def test_missing_counts_are_not_zero_and_null_is_not_false(self):
        text = self.render({'status': 'race_free', 'memory_analysis_complete': False, 'search_complete': None})
        self.assertIn('not saved races', text)
        self.assertIn('Search complete: **unknown** (`null`)', text)
        self.assertIn('Memory analysis complete: **no**', text)
        self.assertIn('Exhaustive search: <code>not saved</code>', text)
        self.assertNotIn('0 races', text)

    def test_cutlass77_shows_rejection_and_gap_without_invented_participants(self):
        inp, result, ptx, paths = saved_case('cutlass77_blackwell_fmha_fp16.compute_100a')
        text = view.render_result(inp, result, ptx, paths)
        for required in ('issue_not_ordered_before_next_phase', '1290359', '1239648', '1290360', 'barrier 1', 'generation 0', '1,318,804 events not listed', 'not an observed GPU trace', '314797', '310187'):
            self.assertIn(required, text)
        self.assertNotIn('Thread 0', text)
        self.assertNotIn('sequenceDiagram', text)
        self.assertIn('no matching physical PTX', self.render(result))
        self.assertIn('do not join the retained prefix directly to the stop point', text)

    def test_next_phase_requirement_does_not_reverse_the_edge(self):
        result = {'status': 'synchronization_candidate', 'rejection': {
            'code': 'issue_not_ordered_before_next_phase', 'issue': 8,
            'required_events': {'events': [3, 5], 'total': 2, 'truncated': False}}}
        text = self.render(result)
        self.assertIn('I -. "required before: not established" .-> R', text)
        self.assertIn('before at least one listed phase effect', text)
        self.assertIn('not predecessors that the wait must follow', text)
        self.assertNotIn('R -. "required before: not established" .-> I', text)

    def test_unknown_rejection_does_not_acquire_a_direction(self):
        text = self.render({'status': 'synchronization_candidate', 'rejection': {
            'code': 'future_check', 'issue': 8, 'required_events': {'events': [3]}}})
        self.assertNotIn('required before: not established', text)
        self.assertIn('I --- W', text)

    def test_untruncated_witness_has_no_fabricated_gap(self):
        result = {'status': 'synchronization_candidate', 'counterexample': {'executed_prefix': {'events': [1, 2], 'total': 2, 'truncated': False}, 'first_unexecuted': 7}}
        text = self.render(result)
        self.assertNotIn('omitted prefix', text)
        self.assertIn('2 event IDs retained', text)

    def test_cutlass93_preserves_missing_order_and_retained_count(self):
        inp, result, ptx, paths = saved_case('cutlass93_blackwell_low_latency_gqa.compute_100a')
        text = view.render_result(inp, result, ptx, paths)
        self.assertIn('tcgen05.load must complete', text)
        self.assertIn('15736', text)
        self.assertIn('3546', text)
        self.assertIn('131,072 reported; 32 records retained', text)
        self.assertIn('not a recorded execution sequence', text)
        self.assertNotIn('physical PTX 15736', text)

    def test_same_actor_does_not_erase_distinct_accesses(self):
        folder = ROOT / 'tests/report_skill/fixtures/tmem'
        result = json.loads((folder / 'output.json').read_text())
        text = self.render(result)
        for value in ('First access', 'Second access', '258', '3072', 'bytes 1032–1039'):
            self.assertIn(value, text)
        self.assertIn('do not show execution order', text)
        self.assertIn('does not itself establish CUDA thread coordinates', text)

    def test_race_free_warnings_and_classification_remain_visible(self):
        folder = ROOT / 'tests/report_skill/fixtures/tma'
        result = json.loads((folder / 'output.json').read_text())
        # An enabled rule and an unmeasured trigger flag are separate facts.
        rule = next(item for item in result['assumptions'] if item['id'] == 'A002')
        rule['triggered'] = None
        text = self.render(result)
        self.assertIn('NO RACES REPORTED<br/>341,248 WARNINGS', text)
        self.assertIn('Matching same-warp duplicate TMA stores are reported as warnings instead of races.', text)
        self.assertIn('enabled <code>true</code> · triggered <code>null</code>', text)
        self.assertIn('Value equality is not proven', text)

    def test_graph_focus_is_shared_and_long_names_stay_outside_graph(self):
        folder = ROOT / 'tests/report_skill/fixtures/tmem'
        result = json.loads((folder / 'output.json').read_text())
        text = self.render(result)
        graph = text.split('```mermaid\n')[1].split('```')[0]
        self.assertIn('classDef context', graph)
        self.assertIn('classDef operation', graph)
        self.assertIn('classDef focus', graph)
        allocation = result['races'][0]['region']['allocation']
        self.assertNotIn(allocation, graph)
        self.assertIn(allocation, text)

    def test_source_locations_do_not_map_instruction_ids_or_other_files(self):
        ptx = '.entry test(){\nret;\n}'
        paths = ['input.json', 'output.json', '/repo/corpus/kernel.ptx']
        cases = [({'file': 'kernel.cu', 'line': 2}, False),
                 ({'file': 'other/kernel.ptx', 'line': 2}, False),
                 ({'file': 'corpus/kernel.ptx', 'line': 2}, True),
                 ({'file': 'corpus/kernel.ptx', 'line': 99}, False),
                 ({'instruction': 2}, False)]
        for location, matched in cases:
            text = self.render({'status': 'frontend_unsupported', 'location': location}, ptx, paths)
            with self.subTest(location=location):
                self.assertEqual('<pre><code>' in text, matched)
                if not matched:
                    self.assertIn('no matching physical PTX', text)

    def test_saved_ptx_excerpt_bolds_opcode_without_changing_text(self):
        ptx = '.entry test(){\n@%p1 bra done;\ndone:\nret;\n}'
        text = self.render({'status': 'frontend_unsupported', 'location': {'file': 'kernel.ptx', 'line': 2}}, ptx, ['input.json', 'output.json', 'kernel.ptx'])
        self.assertIn('<strong>bra</strong>', text)
        self.assertIn('@%p1', text)
        self.assertIn('done;', text)

    def test_model_facts_do_not_add_deterministic_source_reasoning(self):
        inp = {'entry': 'test', 'launch': {'block_dim': [32, 1, 1]}}
        ptx = '.entry test(){\nmov.u32 %r1, %tid.x;\nsetp.eq.u32 %p1, %r1, 0;\n@%p1 bra done;\ndone:\nret;\n}'
        text = view.render_result(inp, {'status': 'race_free'}, ptx, [], 'Test')
        self.assertNotIn('lane zero', text)
        self.assertNotIn('Predicate', text)
        self.assertNotIn('participant timeline', text)

    def test_retained_diagnostics_and_all_witness_sequences_survive(self):
        inp, result, ptx, paths = saved_case('cutlass77_blackwell_fmha_fp16.compute_100a')
        text = view.render_evidence(inp, result, ptx, paths)
        blocks = [json.loads(raw) for raw in re.findall(r'```json\n(.*?)\n```', text, re.S)]
        self.assertIn(result['counterexample'], blocks)
        self.assertIn(result['source_events'], blocks)
        self.assertIn(result['rejection'], blocks)
        self.assertEqual(text.count('<details>'), text.count('</details>'))
        depth = maximum = 0
        for token in re.findall(r'</?details>', text):
            depth += 1 if token == '<details>' else -1
            maximum = max(maximum, depth)
        self.assertLessEqual(maximum, 2)

    def test_sync_default_avoids_duplicate_error_and_long_source_paths(self):
        inp, result, ptx, paths = saved_case('cutlass77_blackwell_fmha_fp16.compute_100a')
        text = view.render_result(inp, result, ptx, paths)
        self.assertEqual(text.count('issue_not_ordered_before_next_phase'), 1)
        self.assertIn('could not establish that the wait is issued before at least one completion effect', text)
        self.assertIn('PTX line <code>314797</code>', text)
        visible_labels = re.sub(r'\]\([^)]*\)', ']', text)
        self.assertNotIn('corpus/cutlass77_blackwell_fmha_fp16.compute_100a.ptx', visible_labels)

    def test_sync_default_identifies_operations_before_coverage_and_keeps_gap_visible(self):
        inp, result, ptx, paths = saved_case('cutlass77_blackwell_fmha_fp16.compute_100a')
        text = view.render_result(inp, result, ptx, paths)
        visible = collapsed_view(text)
        graph = visible.split('```mermaid\n')[1].split('```')[0]
        for opcode in ('mbarrier.try_wait', 'tcgen05.commit'):
            self.assertIn(opcode, graph)
        for event in (result['rejection']['issue'], *result['rejection']['required_events']['events']):
            self.assertNotIn(str(event), graph)
        self.assertLess(visible.index('mbarrier.try_wait'), visible.index('**Coverage:**'))
        self.assertIn('1,318,804 events not listed', visible)
        self.assertNotIn('| Role |', visible)
        # Full opcodes and source line are joined to the correct model IDs,
        # including the distinct issue and stop events at one source operation.
        table = [line for line in text.splitlines() if line.startswith('| Wait issue |')]
        self.assertEqual(len(table), 1)
        for item in ('mbarrier.try_wait.parity.shared::cta.b64', '314797', '1290359', '330050'):
            self.assertIn(item, table[0])

    def test_lifecycle_default_has_semantic_roles_without_invented_source_mapping(self):
        inp, result, ptx, paths = saved_case('cutlass93_blackwell_low_latency_gqa.compute_100a')
        text = view.render_result(inp, result, ptx, paths)
        visible = collapsed_view(text)
        graph = visible.split('```mermaid\n')[1].split('```')[0]
        self.assertIn('TMEM read completes', graph)
        self.assertIn('Free TMEM allocation', graph)
        self.assertNotIn('15736', visible)
        self.assertNotIn('3546', visible)
        self.assertNotIn('<pre><code>', text)
        self.assertIn('do not map', visible)
        self.assertIn('131,072 reported; 32 records retained', visible)
        self.assertIn('15736', text)
        self.assertIn('3546', text)

    def test_matched_access_opcode_stays_beside_its_own_ids(self):
        ptx = 'st.shared.b32 [%r1], %r2;\nld.shared.b32 %r3, [%r1];'
        paths = ['input.json', 'output.json', '/repo/kernel.ptx']
        pair = {'first': {'event': 41, 'instruction': 73, 'mode': 'write', 'location': {'file': 'kernel.ptx', 'line': 1}},
                'second': {'event': 59, 'instruction': 82, 'mode': 'read', 'location': {'file': 'kernel.ptx', 'line': 2}}}
        text = self.render({'status': 'race_candidate', 'races': [pair]}, ptx, paths)
        rows = [line for line in text.splitlines() if line.startswith('| First access |') or line.startswith('| Second access |')]
        self.assertEqual(len(rows), 2)
        for row, opcode, event, instruction in zip(rows, ('st.shared.b32', 'ld.shared.b32'), (41, 59), (73, 82)):
            for value in (opcode, str(event), str(instruction)):
                self.assertIn(value, row)
        pair['first']['location']['file'] = 'other.ptx'
        text = self.render({'status': 'race_candidate', 'races': [pair]}, ptx, paths)
        first = next(line for line in text.splitlines() if line.startswith('| First access |'))
        self.assertNotIn('st.shared.b32', first)
        self.assertIn('unconfirmed', first)

    def test_source_operation_lookup_masks_comments_and_does_not_search_neighbor_lines(self):
        ptx = '/*\nst.shared.b32 [%r1], %r2;\n*/\nlabel:\n@%p1 ld.shared.b32 %r2, [%r1];'
        paths = ['input.json', 'output.json', 'kernel.ptx']
        locations = [{'file': 'kernel.ptx', 'line': line} for line in (2, 4, 5)]
        operations = view._source_operations(locations, ptx, paths)
        self.assertEqual(operations, {2: '', 4: '', 5: 'ld.shared.b32'})

    def test_clean_result_has_no_diagnostic_scaffolding_or_empty_details(self):
        result = {'status': 'race_free', 'race_count': 0, 'warning_count': 0, 'unresolved_count': 0,
                  'memory_analysis_complete': True, 'search_complete': True, 'memory_analysis': {'exhaustive': False},
                  'assumption_dependent': True, 'assumptions': [{'id': 'A003', 'enabled': True, 'triggered': True}]}
        text = self.render(result)
        self.assertNotIn('<details>', text)
        self.assertNotIn('Synchronization requirement', text)
        self.assertNotIn('Memory access relation', text)
        self.assertNotIn('Saved status:', text)
        self.assertIn('Exhaustive search: **no**', text)
        self.assertIn('A003', text)
        self.assertLess(len(text.split()), 115)

    def test_non_prefix_sequence_truncation_remains_visible(self):
        result = {'status': 'synchronization_candidate', 'counterexample': {
            'executed_prefix': {'events': [1, 2], 'total': 2, 'truncated': False},
            'candidate_order': {'events': [1, 2], 'total': 8, 'truncated': True}, 'first_unexecuted': 3}}
        text = collapsed_view(self.render(result))
        self.assertIn('candidate order', text)
        self.assertIn('truncated', text)
        self.assertNotIn('Replay gap:', text)

    def test_open_diagnostic_group_has_readable_structure_before_exact_json(self):
        inp, result, ptx, paths = saved_case('cutlass77_blackwell_fmha_fp16.compute_100a')
        text = view.render_evidence(inp, result, ptx, paths)
        visible = text.split('<summary><strong>Exact diagnostic records and metadata</strong></summary>')[0]
        self.assertIn('Model replay coverage', visible)
        self.assertIn('| Sequence | Retained / total events | Truncated |', visible)
        self.assertIn('Event-to-source lookup', visible)
        self.assertNotIn('```json', visible)

    def test_missing_source_identifiers_and_arguments_stay_unknown_in_evidence(self):
        text = view.render_evidence({}, {'source_events': [{'line': 7}]}, '', [])
        self.assertIn('<code>not saved</code>', text)
        self.assertIn('**Arguments:** not saved retained entries', text)
        self.assertNotIn('**Arguments:** 0', text)

    def test_large_retained_diagnostic_list_is_not_replaced_by_count(self):
        records = [{'code': 'distinct-failure-' + str(i), 'message': 'x' * 500} for i in range(50)]
        text = view.render_evidence({}, {'diagnostics': records}, '', [])
        for record in records:
            self.assertIn(record['code'], text)
        self.assertNotIn('output.json#/diagnostics"', text)

    def test_actual_large_diagnostic_list_is_lossless_and_report_below_one_megabyte(self):
        folder = ROOT / 'examples/vllm__dsv3_fused_a_gemm__sm_100f'
        result = json.loads((folder / 'output.json').read_text())
        inp = json.loads((folder / 'input.json').read_text())
        records = result['diagnostics']
        self.assertEqual(len(records), 95232)
        packed = view._lossless_groups(records, '/diagnostics')
        self.assertEqual(len(packed['groups']), 3072)
        self.assertEqual(packed['groups'][0]['indices'], {'range': [0, 32, 31]})
        self.assertEqual(packed['groups'][0]['fields']['/details/instruction'], {'range': [26313, 452, 31]})
        self.assertEqual(reconstruct_records(packed), records)
        self.assertEqual(packed['canonical_json_sha256'], hashlib.sha256(view._encoded(records).encode()).hexdigest())
        text = view.render_result(inp, result, '', [], 'Example') + view.render_evidence(inp, result, '', [folder / 'input.json', folder / 'output.json'])
        blocks = [json.loads(raw) for raw in re.findall(r'```json\n(.*?)\n```', text, re.S)]
        embedded = next(block for block in blocks if isinstance(block, dict) and block.get('representation') == 'indexed-record-groups/v2')
        self.assertEqual(reconstruct_records(embedded), records)
        self.assertLess(len(text.encode()), 1_000_000)
        self.assertIn('| <code>barrier_initialized_multiple_times</code> | 95,232 | 3,072 |', text)

    def test_lossless_grouping_retains_irregular_integer_values_and_null_fields(self):
        records = [{'code': 'example', 'message': None, 'instruction': i * i, 'metadata': {'optional': None}} for i in range(128)]
        packed = view._lossless_groups(records, '/diagnostics')
        self.assertIn('values', packed['groups'][0]['fields']['instruction'])
        self.assertEqual(reconstruct_records(packed), records)
        descending = [{'message': 'same', 'instruction': -i} for i in range(128)]
        self.assertEqual(reconstruct_records(view._lossless_groups(descending, '/diagnostics')), descending)

    def test_irregular_shapes_and_null_records_use_exact_raw_fallback(self):
        records = [{'code': 'example', 'instruction': i, 'message': 'same'} for i in range(128)]
        for replacement in (None, {'code': 'missing instruction'}, {'code': 'example', 'instruction': None, 'message': 'same'}):
            changed = list(records)
            changed[17] = replacement
            self.assertIsNone(view._lossless_groups(changed, '/diagnostics'))
            text = view.render_evidence({}, {'diagnostics': changed}, '', [])
            blocks = [json.loads(raw) for raw in re.findall(r'```json\n(.*?)\n```', text, re.S)]
            self.assertIn(changed, blocks)
        long_record = [{'code': 'large', 'message': 'x' * 25000}]
        text = view.render_evidence({}, {'diagnostics': long_record}, '', [])
        blocks = [json.loads(raw) for raw in re.findall(r'```json\n(.*?)\n```', text, re.S)]
        self.assertIn(long_record, blocks)

    def test_huge_equivalence_keeps_separate_status_contract_and_exact_pointer(self):
        rows = [{'family': i, 'status': 'equivalent', 'proof': 'x' * 100} for i in range(1000)]
        result = {'status': 'race_free', 'value_analysis': {'status': 'expressions_ready'}, 'equivalence': {'status': 'equivalent', 'definedness': {'status': 'not_proven'}, 'equivalence_contract': 'same_domain_and_equal_values', 'rows': rows}}
        text = view.render_evidence({}, result, '', [])
        self.assertIn('Memory safety and numerical equivalence are separate', text)
        self.assertIn('not_proven', text)
        self.assertIn('same_domain_and_equal_values', text)
        self.assertIn('output.json#/equivalence/rows', text)
        self.assertIn('1000 items', text)
        digest = hashlib.sha256(view._encoded(rows).encode()).hexdigest()
        self.assertIn(digest, text)
        self.assertLess(len(text), 15000)

    def test_original_artifact_hashes_hash_file_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ('input.json', 'output.json', 'kernel.ptx')]
            contents = ['{ "entry": "test" }\n', '{"status":"race_free"}\n', '.entry test(){ret;}\n']
            for path, content in zip(paths, contents):
                path.write_text(content)
            text = view.render_evidence(json.loads(contents[0]), json.loads(contents[1]), contents[2], paths)
            for content in contents:
                self.assertIn(hashlib.sha256(content.encode()).hexdigest(), text)
            self.assertIn('hashes of the file bytes', text)

    def test_backticks_html_and_table_pipes_cannot_break_saved_prose(self):
        result = {'status': 'frontend_unsupported', 'message': '<details> | %r48 A002', 'diagnostics': [{'message': '```\n</details>\n```'}]}
        text = self.render(result)
        self.assertNotIn('<details>', text)
        self.assertIn('&lt;details&gt;', text)
        self.assertIn('<code>%r48</code>', text)
        evidence = view.render_evidence({}, result, '', [])
        self.assertIn('````json', evidence)
        self.assertNotIn('\n</details>\n```', evidence)

    def test_saved_snake_case_identifiers_are_marked_without_changing_prose(self):
        original = 'Wait check failed: issue_not_ordered_before_next_phase. <field_name> & %r_value A002 | enabled'
        text = view._text(original)
        self.assertIn('<code>issue_not_ordered_before_next_phase</code>', text)
        self.assertIn('&lt;<code>field_name</code>&gt;', text)
        self.assertIn('<code>%r_value</code>', text)
        self.assertNotIn('<code><code>', text)
        self.assertEqual(html.unescape(text.replace('<code>', '').replace('</code>', '')), original)

    def test_numerical_blocked_remains_distinct_from_memory_result(self):
        inp, result, ptx, paths = saved_case('tirx-flashinfer-quantization-nvfp4')
        text = view.render_evidence(inp, result, ptx, paths)
        self.assertIn('translation_or_supported_fragment_blocked', text)
        self.assertIn('Could not prove that the saturation input is finite', text)
        self.assertIn('saturation_input_finiteness_not_proven', text)
        self.assertIn('not_proven', text)


if __name__ == '__main__':
    unittest.main()
