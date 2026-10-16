import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / 'skills/generate-kernel-report/scripts/check_report.py'
spec = importlib.util.spec_from_file_location('check_report', SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class ReportBoundaryChecks(unittest.TestCase):
    def test_representative_actor_pair_cannot_become_whole_retained_set(self):
        data={'status':'race_free','warnings':[
            {'first':{'actor':128},'second':{'actor':129}},
            {'first':{'actor':128},'second':{'actor':130}}]}
        self.result.write_text(json.dumps(data))
        self.report.write_text(self.text + 'All 2 retained warnings are between actors 128 and 129.')
        self.assertTrue(any('other actor pairs' in x for x in self.run_check()))
        self.report.write_text(self.text + 'One representative retained warning is between actors 128 and 129.')
        self.assertFalse(any('other actor pairs' in x for x in self.run_check()))
        data['warnings']=data['warnings'][:1];self.result.write_text(json.dumps(data))
        self.report.write_text(self.text + 'All retained warnings are between actors 128 and 129.')
        self.assertFalse(any('other actor pairs' in x for x in self.run_check()))

    def test_expanded_local_guard_does_not_prove_actual_issuers(self):
        self.report.write_text(self.text + '<details><summary>Exact source</summary>Each of the 32 lanes issues the copy.</details>')
        self.assertTrue(any('execution claim' in x for x in self.run_check()))
        self.report.write_text(self.text + '<details><summary>Exact source</summary>The evaluated guard admits 32 lanes; they may issue if reached.</details>')
        self.assertFalse(any('execution claim' in x for x in self.run_check()))

    def test_retained_warning_coordinate_is_not_a_dynamic_first_site(self):
        self.report.write_text(self.text + 'The retained warnings cover only the first site at 22.')
        self.assertTrue(any('unrolled occurrence' in x for x in self.run_check()))
        self.report.write_text(self.text + 'Only 16 warning records are retained, all at the first of four store sites.')
        self.assertTrue(any('unrolled occurrence' in x for x in self.run_check()))
        self.report.write_text(self.text + 'Records cite line 22; the first site is representative, not a unique dynamic binding.')
        self.assertFalse(any('unrolled occurrence' in x for x in self.run_check()))

    def test_unsaved_warning_distribution_cannot_inherit_sample_code(self):
        self.result.write_text(json.dumps({'status':'race_free','warning_count':100,'warnings':[{'code':'duplicate-tma-store'}]}))
        self.report.write_text(self.text + 'There were 100 duplicate-tma-store warnings.')
        self.assertTrue(any('type distribution' in x for x in self.run_check()))
        self.report.write_text(self.text + 'There were 100 warnings. The retained sample is duplicate-tma-store.')
        self.assertFalse(any('type distribution' in x for x in self.run_check()))

    def test_classification_rule_is_not_value_equality_assumption(self):
        self.result.write_text(json.dumps({'status':'race_free','assumptions':[
            {'key':'duplicate_tma_store_benign','enabled':True}]}))
        for claim in ('The run assumed equal values.', 'Inspect equality; the saved run assumed this.'):
            self.report.write_text(self.text + claim)
            self.assertTrue(any('does not assume copied-value' in x for x in self.run_check()))
        self.report.write_text(self.text + 'The rule classifies matching footprints; copied-data equality is not proven.')
        self.assertFalse(any('does not assume copied-value' in x for x in self.run_check()))

    def test_tmem_register_offsets_do_not_prove_saved_byte_footprint(self):
        self.ptx.write_text('tcgen05.cp.cta_group::1.128x128b [%r1], %rd1;')
        self.report.write_text(self.text + "MMA 2 reads base+258 inside the first copy's 16-byte write.")
        self.assertTrue(any('infer a byte footprint' in x for x in self.run_check()))
        self.report.write_text(self.text + 'No supplied mapping establishes that base+258 is inside the copy footprint.')
        self.assertFalse(any('infer a byte footprint' in x for x in self.run_check()))

    def test_read_one_cannot_leave_an_older_group_pending(self):
        self.report.write_text(self.text + 'read 1 allows one earlier group to remain pending.')
        self.assertTrue(any('newest group pending' in x for x in self.run_check()))

    def test_authored_branch_reference_names_an_actual_branch(self):
        self.ptx.write_text('@%p1 bra done;\nshl.b32 %r1, %r2, 8;\ndone:\n')
        self.report.write_text(self.text + 'The copy is skipped by the branch at 2.')
        self.assertTrue(any('not a branch instruction' in x for x in self.run_check()))
        self.report.write_text(self.text + 'The copy is skipped by 2 (%p1).')
        self.assertTrue(any('not a branch instruction' in x for x in self.run_check()))
        self.report.write_text(self.text + 'The copy is skipped by the branch at 1.')
        self.assertFalse(any('not a branch instruction' in x for x in self.run_check()))

    def test_tensor_coordinate_order_is_not_a_logical_axis_mapping(self):
        self.ptx.write_text('cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%rd1, {%r1,%r2,%r3}], [%r4];')
        self.report.write_text(self.text + 'The stores differ in the D row coordinate.')
        self.assertTrue(any('descriptor-axis position' in x for x in self.run_check()))
        self.report.write_text(self.text + 'The stores differ in the second descriptor-axis coordinate.')
        self.assertFalse(any('descriptor-axis position' in x for x in self.run_check()))

    def test_source_citation_must_contain_the_named_predicate(self):
        self.ptx.write_text('setp.ge.s32 %p67, %r1, %r2;\nshr.s32 %r3, %r4, 31;\n')
        self.report.write_text(self.text + 'The loop is bounded by %p67 at line 2.')
        self.assertTrue(any('does not contain' in x for x in self.run_check()))
        self.report.write_text(self.text + 'The loop is bounded by %p67 at line 1.')
        self.assertFalse(any('does not contain' in x for x in self.run_check()))

    def test_local_initialization_citation_must_be_a_store(self):
        self.ptx.write_text('st.local.f32 [%rd1], %f1;\nentry_label:\n')
        self.report.write_text(self.text + 'The local array is initially written at line 2.')
        self.assertTrue(any('not a local store' in x for x in self.run_check()))
        self.report.write_text(self.text + 'The local array is initially written at line 1.')
        self.assertFalse(any('not a local store' in x for x in self.run_check()))

    def test_composed_header_rejects_second_authored_result(self):
        header = '<!-- saved-result:start --><h1 align="center">Result</h1><!-- saved-result:end -->'
        for duplicate in ['<h1>Other result</h1>', '# Another result\n', '<p align="center">Status</p>']:
            self.assertTrue(any('Duplicate result' in x for x in checker.reading_checks(header + '\n' + duplicate)))
        self.assertFalse(any('Duplicate result' in x for x in checker.reading_checks(header + '\n## Computation\nContent.')))

    def test_composed_facts_do_not_consume_the_authored_word_budget(self):
        source = '<!-- saved-result:start -->' + 'saved ' * 150 + '<!-- saved-result:end -->'
        source += 'computation ' * 100 + '\n```math\nA=B\n```\n'
        source += '<!-- recorded-relation:start -->' + 'overlap ' * 100 + '<!-- recorded-relation:end -->'
        source += 'account ' * 80
        self.assertEqual(checker.reading_checks(source), [])

    def test_long_authored_account_is_still_measured_with_composed_header(self):
        source = '<!-- saved-result:start -->status<!-- saved-result:end -->' + 'account ' * 320
        self.assertTrue(any('Closed report' in issue for issue in checker.reading_checks(source)))

    def test_saved_evidence_is_not_a_source_derivation_for_async_claim(self):
        self.ptx.write_text('cp.async.bulk.wait_group.read 1;\n')
        source = 'race_free · non-exhaustive\n```math\na=b\n```\n<pre><code>cp.async.bulk.wait_group.read 1;</code></pre>'
        self.report.write_text(source + '<!-- saved-evidence:start --><details><summary>Exact source</summary>Saved field.</details><!-- saved-evidence:end -->')
        self.assertTrue(any('authored exact-source' in issue for issue in checker.check(self.report, self.ptx, self.result)))
        self.report.write_text(source + '<details><summary>Exact source · configured exit</summary>Derived path and physical ranges.</details>')
        self.assertFalse(any('authored exact-source' in issue for issue in checker.check(self.report, self.ptx, self.result)))

    def test_branch_taken_condition_matches_literal_predicate_polarity(self):
        self.ptx.write_text('@%p166 bra $L;\n@!%p2 bra $R;\n')
        self.report.write_text('race_free\nThe jump at line 1 is taken only if %p166 is false.\nThe branch at line 2 is taken if %p2 is true.\n')
        issues = checker.check(self.report, self.ptx, self.result)
        self.assertEqual(sum('Branch polarity' in issue for issue in issues), 2)
        self.report.write_text('race_free\nThe jump at line 1 is taken only if %p166 is true.\nThe branch at line 2 is taken if %p2 is false.\n')
        self.assertFalse(any('Branch polarity' in issue for issue in checker.check(self.report, self.ptx, self.result)))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ptx = self.root / 'kernel.ptx'
        self.ptx.write_text('mov.u32 %r1, %tid.x;\n')
        self.result = self.root / 'output.json'
        self.result.write_text(json.dumps({'status': 'race_free', 'memory_analysis': {'exhaustive': False}}))
        self.digest = hashlib.sha256(self.ptx.read_bytes()).hexdigest()
        self.identity = self.root / 'identity.json'
        self.identity.write_text(json.dumps({'files': {'kernel.ptx': {'sha256': self.digest}}}))
        self.report = self.root / 'report.md'
        self.text = ('race_free · non-exhaustive\n\n```math\na \\longrightarrow b\n```\n'
                     '<pre><code>mov.u32 <b>%r1</b>, %tid.x;</code></pre>\n'
                     '<details><summary>Sources</summary>\n' + self.digest + '\n</details>\n')
        self.report.write_text(self.text)

    def tearDown(self):
        self.temp.cleanup()

    def run_check(self):
        return checker.check(self.report, self.ptx, self.result, self.identity)

    def test_nested_reference_does_not_count_as_explanation(self):
        source = '<details><summary>Why this branch?</summary>Short answer.<details><summary>Exact derivation</summary>' + 'value ' * 400 + '</details></details>'
        self.assertEqual(checker.reading_checks(source), [])

    def test_math_budget_counts_nested_reference_content(self):
        source = '<details><summary>Owner comparison</summary>\n```math\nx=1\n```\n<details><summary>Exact derivation</summary>' + 'value ' * 700 + '</details></details>'
        self.assertTrue(any('math-bearing disclosure' in issue for issue in checker.reading_checks(source)))

    def test_adjacent_reference_preserves_math_and_evidence(self):
        source = '<details><summary>Owner comparison</summary>\n```math\nx=1\n```\n</details><details><summary>Exact derivation</summary>' + 'value ' * 700 + '</details>'
        self.assertEqual(checker.reading_checks(source), [])

    def test_long_nonmath_reference_has_no_render_budget(self):
        source = '<details><summary>Exact settings</summary>' + 'value ' * 700 + '</details>'
        self.assertEqual(checker.reading_checks(source), [])

    def test_quoted_math_disclosure_does_not_trigger_budget(self):
        source = '````text\n<details><summary>Exact settings</summary>\n```math\nx=1\n```\n' + 'value ' * 700 + '\n</details>\n````'
        self.assertEqual(checker.math_disclosure_checks(source), [])

    def test_multiple_formulas_do_not_mask_disclosures_between_them(self):
        block = '<details><summary>Exact derivation</summary>\n```math\nx=1\n```\n' + 'value ' * 700 + '</details>\n'
        source = '```math\ny=2\n```\n' + block + block
        self.assertEqual(len(checker.math_disclosure_checks(source)), 2)

    def test_math_budget_measures_utf8_not_character_count(self):
        source = '<details><summary>Exact reference</summary>\n```math\nx=1\n```\n' + '证据' * 700 + '</details>'
        self.assertEqual(len(checker.math_disclosure_checks(source)), 1)

    def test_closed_and_open_explanation_are_checked_separately(self):
        source = 'value ' * 320 + '<details><summary>Why this branch?</summary>' + 'value ' * 320 + '</details>'
        self.assertEqual(len(checker.reading_checks(source)), 2)

    def test_long_listing_needs_a_named_reference(self):
        listing = '<pre><code>' + '\n'.join(['mov.u32 %r1, %tid.x;'] * 20) + '</code></pre>'
        self.assertTrue(checker.reading_checks('<details><summary>Why this branch?</summary>' + listing + '</details>'))
        self.assertEqual(checker.reading_checks('<details><summary>Exact local PTX</summary>' + listing + '</details>'), [])

    def test_splitting_long_evidence_into_blocks_does_not_bypass_layer_limit(self):
        block = '<pre><code>' + '\n'.join(['mov.u32 %r1, %tid.x;'] * 6) + '</code></pre>'
        self.assertTrue(checker.reading_checks('<details><summary>Why this branch?</summary>' + block * 2 + '</details>'))

    def test_diagram_labels_are_not_prose(self):
        source = 'Short account.\n```mermaid\nflowchart LR\n' + 'x ' * 400 + '\n```'
        self.assertEqual(checker.reading_checks(source), [])

    def test_wide_schema_table_is_flagged_inside_reference(self):
        source = '<details><summary>Reference</summary>\n| Rule | ID | Enabled | Triggered | Meaning |\n| --- | --- | --- | --- | --- |\n| `duplicate_tma_store_benign` | A002 | true | true | Classification |\n</details>'
        self.assertTrue(any('indivisible identifier' in issue for issue in checker.reading_checks(source)))

    def test_exact_key_mapping_can_remain_in_two_columns(self):
        source = '| Role | Exact key |\n| --- | --- |\n| Matching stores | `duplicate_tma_store_benign` |\n'
        self.assertEqual(checker.reading_checks(source), [])

    def test_blank_line_after_table_header_is_a_render_failure(self):
        source = '| Actor | Event |\n| --- | --- |\n\n| 128 | 107459 |\n'
        self.assertTrue(any('split Markdown table rows' in issue for issue in checker.reading_checks(source)))

    def test_blank_line_between_pair_rows_is_a_render_failure(self):
        source = '| Actor | Pair |\n| --- | --- |\n| 128 | 1 |\n\n| 129 | 3 |\n'
        self.assertTrue(any('split Markdown table rows' in issue for issue in checker.reading_checks(source)))

    def test_two_separately_headed_tables_can_have_a_blank_between_them(self):
        table = '| Actor | Event |\n| --- | --- |\n| 128 | 107459 |\n'
        self.assertEqual(checker.reading_checks(table + '\n' + table), [])

    def test_quoted_pipe_rows_do_not_trigger_table_render_check(self):
        source = '```text\n| Actor | Event |\n| --- | --- |\n\n| 128 | 107459 |\n```'
        self.assertEqual(checker.reading_checks(source), [])

    def test_schema_text_in_fenced_example_is_not_a_table(self):
        source = '```text\n| Rule | ID | Enabled | Triggered | Meaning |\n| --- | --- | --- | --- | --- |\n| `duplicate_tma_store_benign` | A002 | true | true | Classification |\n```'
        self.assertEqual(checker.reading_checks(source), [])

    def test_explanation_relation_must_precede_code(self):
        code = '<pre><code>mov.u32 %r1, %tid.x;</code></pre>'
        relation = '\n```math\na = b\n```\n'
        opening = '<details><summary>Why this branch?</summary>Short answer.'
        self.assertTrue(any('precedes' in issue for issue in checker.reading_checks(opening + code + relation + '</details>')))
        self.assertEqual(checker.reading_checks(opening + relation + code + '</details>'), [])

    def test_relation_in_sibling_or_reference_does_not_supply_prerequisite(self):
        code = '<pre><code>mov.u32 %r1, %tid.x;</code></pre>'
        reference = '<details><summary>Exact path</summary>\n```math\na = b\n```\n</details>'
        source = '<details><summary>Why this branch?</summary>Short answer.' + reference + code + '</details>'
        self.assertTrue(any('precedes' in issue for issue in checker.reading_checks(source)))

    def test_comparison_table_can_supply_relation(self):
        source = '<details><summary>What does this wait complete?</summary>Short answer.\n\n| Older | Newest |\n| --- | --- |\n| Complete | May be pending |\n\n<pre><code>cp.async.wait_group 1;</code></pre></details>'
        self.assertEqual(checker.reading_checks(source), [])

    def test_unwrapped_operands_need_reformatting(self):
        source = '<pre><code>cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [' + ', '.join(['%rd123'] * 12) + '];</code></pre>'
        self.assertTrue(checker.reading_checks(source))

    def test_unknown_applicability_does_not_support_negative_causation(self):
        self.result.write_text(json.dumps({'status': 'frontend_unsupported', 'warnings': [], 'assumptions': [{'enabled': True, 'triggered': None}]}))
        self.report.write_text('frontend_unsupported · no race verdict\nThe model rules recorded for this run did not cause the stop.')
        self.assertTrue(any('Unknown rule applicability' in issue for issue in self.run_check()))

    def test_formatted_exact_code_passes_without_writing(self):
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        self.assertEqual(self.run_check(), [])
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})

    def test_rejected_math_macro_is_detected_inside_a_closed_fold(self):
        self.report.write_text(self.text + '<details><summary>Computation</summary>\n```math\n\\operatorname{Out}[t]=a+b\n```\n</details>')
        self.assertTrue(any('GitHub rejected' in issue for issue in self.run_check()))

    def test_supported_math_label_and_quoted_macro_remain_valid(self):
        self.report.write_text(self.text + '\n```math\n\\text{Out}[t]=a+b\n```\nUse text rather than `\\operatorname` for labels.\n')
        self.assertEqual(self.run_check(), [])

    def test_hidden_coverage_is_detected(self):
        self.report.write_text(self.text.replace(' · non-exhaustive', '').replace('Sources</summary>', 'Sources</summary>non-exhaustive'))
        self.assertTrue(any('coverage' in issue for issue in self.run_check()))

    def test_changed_operand_is_detected(self):
        self.report.write_text(self.text.replace('<b>%r1</b>', '<b>%r9</b>'))
        self.assertTrue(any('Instruction text differs' in issue for issue in self.run_check()))

    def test_input_identity_mismatch_is_detected(self):
        self.ptx.write_text(self.ptx.read_text() + '// changed input\n')
        self.assertTrue(any('identity does not match' in issue for issue in self.run_check()))

    def test_empty_warning_list_is_not_absence(self):
        data = json.loads(self.result.read_text())
        data['warnings'] = []
        self.result.write_text(json.dumps(data))
        self.report.write_text(self.text + '\nThere are no race or warning fields.\n')
        self.assertTrue(any('warnings is present' in issue for issue in self.run_check()))

    def test_retained_access_indices_must_not_disappear(self):
        data = json.loads(self.result.read_text())
        data['warnings'] = [{'first': {'access_index': 0, 'effect_index': 1}, 'second': {'access_index': 0, 'effect_index': 1}}]
        self.result.write_text(json.dumps(data))
        issues = self.run_check()
        self.assertTrue(any('access_index' in issue for issue in issues))
        self.assertTrue(any('effect_index' in issue for issue in issues))
        self.report.write_text(self.text + '\nReference: both accesses have access_index 0 and effect_index 1.\n')
        self.assertEqual(self.run_check(), [])

    def test_execution_claim_needs_its_scope(self):
        self.report.write_text(self.text + '\nThe store is issued by a whole warp.\n')
        self.assertTrue(any('execution claim' in issue for issue in self.run_check()))
        self.report.write_text(self.text + '\nAll 32 lanes may issue at this local guard.\n')
        self.assertEqual(self.run_check(), [])


if __name__ == '__main__':
    unittest.main()
