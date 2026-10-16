import importlib.util
from pathlib import Path
import subprocess
import tempfile
import json
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / 'skills/generate-kernel-report/scripts/report_facts.py'
spec = importlib.util.spec_from_file_location('report_facts', SCRIPT)
facts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(facts)


class SavedFactsTests(unittest.TestCase):
    def test_mma_flag_index_keeps_initialization_and_later_loop_update(self):
        inp={'entry':'test'}
        ptx='.entry test(){\nmov.b32 %r1, 0;\nsetp.ne.b32 p, %r1, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%r2], %rd3, %rd4, %r5, {%r6}, p;\nmov.b32 %r1, 1;\n}\n.entry other(){mov.b32 %r1, 9;}'
        text=facts.mma_flag_index(inp,ptx)
        self.assertIn('2: ',text)
        self.assertIn('5: ',text)
        self.assertNotIn('%r1, 9',text)
        self.assertIn('not treat the nearest earlier assignment',text)

    def test_actual_mma_index_includes_nonmissing_flag_origins(self):
        root=Path(__file__).parent / 'fixtures/tma'
        text=facts.mma_flag_index(json.loads((root/'input.json').read_text()),(SCRIPT.parents[3]/'corpus/cutlass__dense-gemm-f16-tma-1sm__sm_100a.ptx').read_text())
        self.assertIn('| 2168 | 2167 /',text)
        self.assertIn('2095: ',text)
        self.assertIn('2297: ',text)

    def test_unconditional_return_is_not_an_unknown_predicate(self):
        run_input = {'entry':'test', 'declarations':[{'name':'stride','parameter':0,'offset':8}]}
        ptx = '.entry test(){\nld.param.b64 %rd4, [test_param_0+8];\nbra.uni done;\ncp.async.bulk.wait_group.read 0;\ndone:\nret;\n}'
        result = {'warnings':[{'first':{'location':{'line':4}}}]}
        text = facts.control_index(run_input, result, ptx)
        self.assertIn('Unconditional branch; taken whenever its block is reached', text)
        self.assertNotIn('no indexed definition', text)

    def test_duplicate_result_rejection_preserves_authored_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'input.json').write_text('{"entry":"test"}')
            (root/'output.json').write_text('{"status":"race_free"}')
            (root/'kernel.ptx').write_text('.entry test(){ret;}')
            report = root/'report.md'
            for duplicate in ['<h1 align="center">Result</h1>', '# Result', '<p align="center">Status</p>']:
                report.write_text(duplicate + '\nThe kernel computes a value.\n')
                original = report.read_bytes()
                result = subprocess.run(['python3', str(SCRIPT), '--input', str(root/'input.json'), '--result', str(root/'output.json'), '--ptx', str(root/'kernel.ptx'), '--report', str(report)], capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(b'unrelated report', result.stderr)
                self.assertEqual(report.read_bytes(), original)

    def guard_fixture(self, definitions='', predicate='%p1', block=256):
        inp = {'entry':'test', 'launch':{'block_dim':[block,1,1]}}
        ptx = '.entry test(){\nmov.u32 %r1, %tid.x;\nand.b32 %r2, %r1, 96;\nsetp.ne.b32 %p1, %r2, 0;\n' + definitions + '\n@' + predicate + ' bra skip;\ncp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%rd1, {%r1, %r1, %r1}], [%r1];\nskip:\nret;\n}'
        return inp, ptx

    def test_guard_mask_alone_does_not_invent_outer_narrowing(self):
        inp, ptx = self.guard_fixture()
        result = facts.local_copy_guards(inp, ptx)
        self.assertEqual(result[0]['accepted_tid_x'], list(range(32)) + list(range(128,160)))
        changed = facts.local_copy_guards(inp, ptx.replace(', 96;', ', 32;'))
        self.assertEqual(len(changed[0]['accepted_tid_x']), 128)

    def test_negated_local_guard_reverses_accepted_coordinates(self):
        inp, ptx = self.guard_fixture(predicate='!%p1')
        values = facts.local_copy_guards(inp, ptx)[0]['accepted_tid_x']
        self.assertEqual(values, [v for v in range(256) if v & 96])

    def test_reused_or_predicated_definition_does_not_produce_coordinate_claim(self):
        for extra in ['mov.u32 %r1, 0;', '@%other mov.u32 %r1, 0;', 'mov.u32 %r2, 7; mov.u32 %r2, 8;']:
            inp, ptx = self.guard_fixture(extra)
            result = facts.local_copy_guards(inp, ptx)[0]
            self.assertIsNone(result['accepted_tid_x'])
            self.assertIsNone(result['skip_condition'])

    def test_guard_scope_excludes_other_entries(self):
        inp, ptx = self.guard_fixture()
        ptx += '\n.entry elsewhere(){\nmov.u32 %r1, 19;\nret;\n}'
        self.assertEqual(len(facts.local_copy_guards(inp, ptx)[0]['accepted_tid_x']), 64)

    def test_intervening_label_prevents_claiming_guarded_copy(self):
        inp, ptx = self.guard_fixture()
        ptx = ptx.replace('cp.async', 'other_entry:\ncp.async')
        self.assertEqual(facts.local_copy_guards(inp, ptx), [])

    def test_missing_launch_preserves_branch_without_inventing_count(self):
        inp, ptx = self.guard_fixture()
        inp.pop('launch')
        item = facts.local_copy_guards(inp, ptx)[0]
        self.assertIsNone(item['accepted_tid_x'])
        self.assertEqual(item['skip_condition'], '((tid.x & 96) != 0)')

    def test_actual_tma_preserves_distinct_predicates_and_last_wait(self):
        root = Path(__file__).parent / 'fixtures/tma'
        inp = json.loads((root/'input.json').read_text())
        items = facts.local_copy_guards(inp, (SCRIPT.parents[3]/'corpus/cutlass__dense-gemm-f16-tma-1sm__sm_100a.ptx').read_text())
        by_site = {i['site']:i for i in items}
        for site, predicate in [(3339,'%p176'),(3541,'%p182'),(3733,'%p186'),(3937,'%p190'),(3946,'%p191')]:
            self.assertEqual(by_site[site]['predicate'], predicate)
            self.assertEqual(by_site[site]['accepted_tid_x'], list(range(32)) + list(range(128,160)))

    def test_composed_tma_reference_is_accepted_as_lookup_not_long_prose(self):
        directory = Path(__file__).parent / 'fixtures/tma'
        artifacts = [directory / 'input.json', directory / 'output.json', SCRIPT.parents[3] / 'corpus/cutlass__dense-gemm-f16-tma-1sm__sm_100a.ptx']
        unit = facts.render(json.loads(artifacts[0].read_text()), json.loads(artifacts[1].read_text()), artifacts)
        spec = importlib.util.spec_from_file_location('reference_checker', SCRIPT.with_name('check_report.py'))
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        self.assertEqual(checker.reading_checks(unit), [])

    def test_repeated_pair_keeps_every_record_number(self):
        text = facts.retained([{'first': {'actor': 4}, 'second': {'actor': 9}}] * 6)
        self.assertIn('| **4** | 1, 2, 3, 4, 5, 6 |', text)
        self.assertIn('no retained record', text)

    def test_same_actor_with_distinct_accesses_uses_access_axes(self):
        text = facts.retained([{'first': {'actor': 4, 'event': 1}, 'second': {'actor': 4, 'event': 2}}] * 6)
        self.assertIn('First access', text)
        self.assertNotIn('First actor', text)
        self.assertIn('| **E1** | 1, 2, 3, 4, 5, 6 |', text)
        self.assertIn('event', text)

    def test_long_exact_value_uses_full_width_without_losing_markup(self):
        value = '<saved> | ' + 'message ' * 30
        text = facts.fields({'message': value, 'count': 0})
        self.assertIn(facts.cell(value), text)
        self.assertNotIn('| ' + facts.key_cell('message'), text)
        self.assertIn('| <code>count</code> | <code>0</code> |', text)

    def test_coverage_summary_distinguishes_false_null_and_absent(self):
        text = facts.run_scope({}, {'memory_analysis_complete': False, 'search_complete': None})
        self.assertIn('| **No** | <code>null</code> | <code>not supplied</code> |', text)
        self.assertNotIn('| **No** | **No** | **No** |', text)

    def test_descriptor_summary_does_not_merge_different_strides(self):
        base = {'object': 'D', 'global_shape': [4, 8], 'box_shape': [4, 2], 'offset': 1}
        first = dict(base, coordinate_strides_bytes=[2, 8])
        second = dict(base, coordinate_strides_bytes=[2, 16], offset=2)
        text = facts.run_scope({'tma_descriptors': [first, second]}, {})
        self.assertIn('<code>[2, 8]</code>', text)
        self.assertIn('<code>[2, 16]</code>', text)
        self.assertIn('<code>[1]</code>', text)
        self.assertIn('<code>[2]</code>', text)

    def test_unrecognized_contract_and_launch_fields_are_retained(self):
        text = facts.run_scope({'launch': {'grid_dim': [1,1,1], 'new_flag': None}}, {})
        self.assertIn('| <code>new_flag</code> | <code>null</code> |', text)
        rules = facts.rules({'frontend_assumptions': {'novel_contract': False}}, {})
        self.assertIn('| <code>novel_contract</code> | <code>false</code> |', rules)

    def test_control_index_distinguishes_direct_jump_from_fallthrough_gate(self):
        run_input = {'entry':'test', 'declarations':[{'name':'stride','parameter':0,'offset':8}]}
        ptx = '.entry test(){\nld.param.b64 %rd4, [test_param_0+8];\nsetp.ne.b64 %p2, %rd4, 0;\n@%p2 bra wait;\n@%p1 bra done;\nwait:\ncp.async.bulk.wait_group.read 0;\ndone:\nret;\n}'
        result = {'warnings':[{'first':{'location':{'line':7}}}]}
        text = facts.control_index(run_input, result, ptx)
        self.assertIn('Direct jump / 4', text)
        self.assertIn('Fallthrough predecessor / 5', text)
        self.assertIn('Earlier branches to return', text)
        self.assertIn('3: ', text)
        self.assertIn('not a reaching-definition proof', text)
    def skipped_loads(self, tail, stride=0):
        run_input = {'entry':'test', 'arguments':[{'field':'stride','value':stride}],
                     'declarations':[{'name':'stride','type':'s64','parameter':0,'offset':8},
                                     {'name':'beta','type':'u32','parameter':0,'offset':0}]}
        ptx = '.entry test(){\nld.param.b64 %rd4, [test_param_0+8];\nsetp.eq.b64 %p1, %rd4, 0;\n' + tail + '\n}\n'
        bindings = facts.parameter_bindings(run_input, ptx)
        return facts.skipped_parameter_loads(run_input, ptx, bindings,
                                            facts.constant_predicates(run_input, ptx, bindings))

    def test_known_forward_branch_skips_reload_even_with_later_target(self):
        loads = self.skipped_loads('@%p1 bra $done;\nld.param.b32 %r1, [test_param_0];\nadd.u32 %r2, %r2, 1;\n$done:\nret;')
        self.assertEqual([(l['line'], l['field'], l['branch'], l['target']) for l in loads], [(5, 'beta', 4, 7)])

    def test_later_outside_branch_into_gap_prevents_skipped_claim(self):
        tail = '@%p1 bra $done;\n$reload:\nld.param.b32 %r1, [test_param_0];\n$done:\n@%p2 bra $reload;\nret;'
        self.assertEqual(self.skipped_loads(tail), [])

    def test_false_predicate_does_not_skip_reload(self):
        self.assertEqual(self.skipped_loads('@%p1 bra done;\nld.param.b32 %r1, [test_param_0];\ndone:\nret;', stride=1), [])

    def test_negated_false_predicate_skips_reload(self):
        loads = self.skipped_loads('@!%p1 bra done;\nld.param.b32 %r1, [test_param_0];\ndone:\nret;', stride=1)
        self.assertEqual([l['field'] for l in loads], ['beta'])

    def test_wait_comparison_preserves_allowed_pending_and_issuer_scope(self):
        text = facts.wait_guarantee('.version 9.2\ncp.async.bulk.wait_group.read 1;')
        self.assertIn('permits at most the newest', text)
        self.assertIn('source reads to remain pending', text.lower())
        self.assertIn('Older groups: **finished**', text)
        self.assertIn('Newest group: may remain pending', text)
        self.assertIn('not a recorded runtime schedule', text)
        self.assertIn('one issuing thread', text.lower())
        self.assertIn('not a saved completion state', text)
        self.assertEqual(facts.wait_guarantee('.version 8.0\ncp.async.bulk.wait_group.read 1;'), '')

    def test_constant_or_ignores_unknown_other_operand_and_preserves_polarity(self):
        run_input = {'entry':'test','arguments':[{'field':'stride','value':0}], 'declarations':[{'name':'stride','type':'s64','parameter':0,'offset':8}]}
        ptx = '.entry test(){\nld.param.b64 %rd4, [test_param_0+8];\nsetp.eq.b64 %p165, %rd4, 0;\nor.pred %p6, %p165, %p1;\nnot.pred %p166, %p6;\n}\n'
        bindings = facts.parameter_bindings(run_input, ptx)
        values = {p['predicate']:p['value'] for p in facts.constant_predicates(run_input, ptx, bindings)}
        self.assertEqual(values, {'%p165':True,'%p6':True,'%p166':False})
        self.assertNotIn('%p1', values)

    def test_packed_load_is_not_treated_as_one_scalar_constant(self):
        run_input = {'entry':'test','arguments':[{'field':'n','value':0}], 'declarations':[{'name':'n','type':'u32','parameter':0,'offset':0}]}
        ptx = '.entry test(){\nld.param.b64 %rd1, [test_param_0];\nsetp.eq.b64 %p1, %rd1, 0;\n}'
        self.assertEqual(facts.constant_predicates(run_input, ptx, facts.parameter_bindings(run_input, ptx)), [])

    def test_predicate_names_stride_without_substituting_pointer(self):
        run_input = {'entry':'test', 'arguments':[{'field':'alpha.stride','value':0}, {'field':'alpha.pointer','value':None}],
                     'declarations':[{'name':'alpha.stride','type':'s64','parameter':0,'offset':8},
                                     {'name':'alpha.pointer','type':'pointer','parameter':0,'offset':16}]}
        ptx = '.entry test(){\nld.param.b64 %rd6, [test_param_0+8];\nld.param.b64 %rd7, [test_param_0+16];\nsetp.eq.b64 %p163, %rd6, 0;\n}'
        proofs = facts.constant_predicates(run_input, ptx, facts.parameter_bindings(run_input, ptx))
        self.assertEqual(proofs[0]['inputs'], {'alpha.stride':0})
        rendered = facts.predicate_table(proofs)
        self.assertIn('alpha.stride', rendered)
        self.assertNotIn('alpha.pointer', rendered)

    def test_null_pointer_predicate_requires_supplied_null_value(self):
        data={'entry':'test','arguments':[{'field':'ptr','value':None}],
              'declarations':[{'name':'ptr','type':'pointer','parameter':0,'offset':0}]}
        ptx='.entry test(){\nld.param.b64 %rd1, [test_param_0];\nsetp.eq.b64 %p1, %rd1, 0;\n}'
        bindings=facts.parameter_bindings(data,ptx)
        proofs=facts.constant_predicates(data,ptx,bindings)
        self.assertTrue(proofs[0]['value'])
        self.assertEqual(proofs[0]['inputs'],{'ptr':0})
        data['arguments']=[]
        self.assertEqual(facts.constant_predicates(data,ptx,bindings),[])

    def test_individual_parameters_use_zero_offset_and_named_value(self):
        data={'entry':'test','arguments':[{'parameter':1,'type':'pointer','value':None}],
              'declarations':[{'parameter':1,'name':'finished','type':'pointer'}]}
        ptx='.entry test(){\nld.param.u64 %rd7, [test_param_1];\nsetp.eq.s64 %p15, %rd7, 0;\n}'
        bindings=facts.parameter_bindings(data,ptx)
        self.assertEqual(bindings[0]['field'],'finished')
        proofs=facts.constant_predicates(data,ptx,bindings)
        self.assertEqual(proofs[0]['inputs'],{'finished':0})
        self.assertTrue(proofs[0]['value'])

    def test_tcgen_pairing_note_is_version_and_family_scoped(self):
        code='tcgen05.cp.cta_group::1.128x128b;\ntcgen05.mma.sp.cta_group::1.kind::f16;'
        self.assertIn('same-thread',facts.instruction_notes('.version 9.2\n'+code))
        self.assertIn('not the full instruction transfer',facts.instruction_notes('.version 9.2\n'+code))
        self.assertEqual(facts.instruction_notes('.version 8.0\n'+code),'')
        self.assertEqual(facts.instruction_notes('.version 9.2\ncp.async.bulk.commit_group;'),'')

    def test_branch_corridor_excludes_bypassed_redefinition(self):
        code={1:'@%pKnown bra other;',2:'setp.eq.b16 %pUnknown, %rs1, 0;',3:'bra.uni join;',
              4:'other:',5:'setp.eq.b16 %pUnknown, %rs2, 0;',6:'join:',7:'@%pUnknown bra finish;',8:'wait;'}
        trace=facts.branch_corridor(code,{'other':4,'join':6,'finish':9},
                                   [{'predicate':'%pKnown','value':False,'line':0}],1,8)
        self.assertIn('Not taken',trace)
        self.assertIn('Taken',trace)
        self.assertIn('Unknown; stop here',trace)
        self.assertIn('2:',trace)
        self.assertNotIn('%rs2',trace)
    def test_branch_corridor_does_not_claim_entry_reachability(self):
        trace=facts.branch_corridor({1:'@%p bra finish;'}, {'finish':2}, [],1,2)
        self.assertIn('Assuming entry',trace)
        self.assertIn('not a full kernel trace',trace)

    def test_recorded_region_does_not_invent_full_access_bounds(self):
        result = {'races':[{'region':{'allocation':'A<unsafe>','offset':16,'bytes':8}, 'first':{'actor':3,'bytes':64},'second':{'actor':9,'bytes':32}}]}
        text = facts.recorded_relation(result)
        self.assertIn('16–23', text)
        self.assertIn('reported intersecting bytes', text)
        self.assertIn('A&lt;unsafe&gt;', text)
        self.assertNotIn('byte 79', text)
        self.assertIn('actor 3', text)
        self.assertIn('actor 9', text)

    def test_same_actor_overlap_retains_distinct_modes(self):
        result = {'races':[{'region':{'allocation':'tensor','offset':1032,'bytes':8},
                  'first':{'actor':0,'mode':'write'},'second':{'actor':0,'mode':'read'}}]}
        rendered = facts.recorded_relation(result)
        self.assertIn('write · actor 0', rendered)
        self.assertIn('read · actor 0', rendered)
        self.assertIn('1032–1039', rendered)

    def test_candidate_search_and_counter_do_not_conflate_cause(self):
        rendered = facts.result_header('Test', {'status':'race_candidate','race_count':1,
                  'unresolved_count':0,'search_complete':False})
        self.assertIn('1 candidate pair', rendered)
        self.assertIn('Search is **incomplete**', rendered)
        self.assertNotIn('0 unresolved', rendered)

    def test_parameter_map_keeps_pointer_stride_and_entry_scope(self):
        run_input = {'entry': 'test', 'declarations': [{'name':'beta.stride','parameter':0,'offset':8}, {'name':'beta.pointer','parameter':0,'offset':16}]}
        ptx = '.visible .entry test(.param .align 8 .b8 test_param_0[32])\n{\nmov.b64 %rd1, test_param_0;\nld.param.b64 %rd4, [test_param_0+8];\nld.param.b64 %rd5, [%rd1+16];\n}\n.visible .entry other(){\nld.param.b64 %rd4, [test_param_0+16];\n}'
        bindings = facts.parameter_bindings(run_input, ptx)
        self.assertEqual([(b['register'], b['field']) for b in bindings], [('%rd4', 'beta.stride'), ('%rd5','beta.pointer')])

    def test_parameter_alias_redefinition_is_not_silently_used(self):
        run_input = {'entry':'test','declarations':[{'name':'n','parameter':0,'offset':4}]}
        ptx = '.entry test(){\nmov.b64 %rd1, test_param_0;\nadd.s64 %rd1, %rd1, 8;\nld.param.b32 %r1, [%rd1+4];\n}'
        self.assertEqual(facts.parameter_bindings(run_input, ptx), [])

    def test_header_scopes_warning_pattern_to_retained_records(self):
        result = {'status': 'race_free', 'race_count': 0, 'warning_count': 341248,
                  'warnings': [{'code': 'duplicate-tma-store'}] * 16,
                  'warning_samples_truncated': True, 'memory_analysis': {'exhaustive': False},
                  'assumption_dependent': True, 'assumptions': [{'id': 'A002', 'key': 'duplicate_tma_store_benign', 'enabled': True, 'text': 'Matching stores are warnings.'}]}
        header = facts.result_header('TMA <output>', result)
        self.assertIn('341,248 warnings', header)
        self.assertIn('16 warnings retained', header)
        self.assertIn('non-exhaustive', header)
        self.assertIn('assumption-dependent', header)
        self.assertNotIn('Each warning', header)
        self.assertNotIn('All warnings', header)
        self.assertIn('TMA &lt;output&gt;', header)

    def test_frontend_header_has_no_race_verdict_and_null_is_not_zero(self):
        header = facts.result_header('Stopped', {'status': 'frontend_unsupported', 'race_count': None, 'reason': 'unsupported op'})
        self.assertIn('No race verdict available', header)
        self.assertIn('null races', header)
        self.assertNotIn('0 races', header)
        self.assertIn('unsupported op', header)

    def test_common_does_not_merge_missing_null_false_zero(self):
        self.assertEqual(facts.common_fields([{'a': None}, {}]), {})
        self.assertEqual(facts.common_fields([{'a': False}, {'a': 0}]), {})
        self.assertEqual(facts.common_fields([{'a': None}, {'a': False}]), {})
        self.assertEqual(facts.common_fields([{'a': None}, {'a': None}]), {'a': None})

    def test_values_escape_markup_table_pipes_and_keep_unicode(self):
        value = 'a|<b>&“quoted”'
        cell = facts.cell(value)
        self.assertIn('&#124;', cell)
        self.assertIn('&lt;b&gt;', cell)
        self.assertIn('&amp;', cell)
        self.assertIn('“quoted”', cell)
        self.assertNotIn('<b>', cell)

    def test_endpoint_identity_includes_effect_and_not_only_actor(self):
        records = [{'code': 'x', 'first': {'actor': 7, 'event': 1, 'effect_index': 0}, 'second': {'actor': 7, 'event': 1, 'effect_index': 1}},
                   {'code': 'x', 'first': {'actor': 7, 'event': 1, 'effect_index': 0}, 'second': {'actor': 8, 'event': 2, 'effect_index': 0}}]
        text = facts.retained(records)
        self.assertIn('| 1 | E1 | E2 |', text)
        self.assertIn('| 2 | E1 | E3 |', text)
        self.assertIn('effect_index', text)
        self.assertIn('<code>1</code>', text)

    def test_record_specific_fields_and_repeat_pairs_are_not_dropped(self):
        records = [{'first': {'actor': 1}, 'second': {'actor': 2}, 'region': {'bytes': 2}},
                   {'first': {'actor': 1}, 'second': {'actor': 2}, 'region': {'bytes': 4}}]
        text = facts.retained(records)
        self.assertIn('| 1 | E1 | E2 |', text)
        self.assertIn('| 2 | E1 | E2 |', text)
        self.assertIn('Exact record 1', text)
        self.assertIn('Exact record 2', text)
        self.assertIn('region.bytes', text)

    def test_contract_dedup_only_when_identical_and_unknown_trigger_preserved(self):
        result = {'assumptions': [{'id': 'A1', 'key': 'test', 'text': 'Test meaning.', 'enabled': False, 'triggered': None}],
                  'frontend_assumptions': {'pointer_contract': {'aliasing': 'x'}, 'noalias_unbounded': False},
                  'frontend': {'frontend_assumptions': {'pointer_contract': {'aliasing': 'x'}, 'noalias_unbounded': False}}}
        text = facts.rules(result, {})
        self.assertIn('Identical copies', text)
        self.assertIn('Triggered <code>null</code>', text)
        self.assertIn('Enabled <code>false</code>', text)
        result['frontend']['frontend_assumptions']['noalias_unbounded'] = True
        text = facts.rules(result, {})
        self.assertNotIn('Identical copies', text)
        self.assertIn('<code>true</code>', text)
        self.assertIn('<code>false</code>', text)

    def test_duplicate_parameter_names_are_retained(self):
        text = facts.run_scope({'arguments': [{'field': 'n', 'value': 2}, {'field': 'n', 'value': 4}]}, {})
        self.assertIn('&quot;value&quot;: 2', text)
        self.assertIn('&quot;value&quot;: 4', text)

    def test_composition_has_one_output_and_checker_rejects_modified_facts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            src = root / 'kernel.ptx'
            src.write_text('.entry test(){\nmov.u32 %r1, %tid.x;\nret;\n}\n')
            result = root / 'result.json'
            result.write_text('{"status":"race_free"}')
            run_input = root / 'input.json'
            run_input.write_text('{"entry":"test", "arguments":[]}')
            output = root / 'output'
            output.mkdir()
            report = output / 'report.md'
            args = ['python3', str(SCRIPT), '--input', str(run_input), '--result', str(result), '--ptx', str(src), '--report', str(report)]
            subprocess.run(args, check=True, capture_output=True)
            appended = report.read_bytes()
            self.assertEqual(subprocess.run(args, capture_output=True).returncode, 0)
            self.assertEqual(appended, report.read_bytes())
            self.assertEqual(list(output.iterdir()), [report])
            check = ['python3', str(SCRIPT.with_name('check_report.py')), '--report', str(report), '--input', str(run_input), '--result', str(result), '--ptx', str(src)]
            self.assertEqual(subprocess.run(check, capture_output=True).returncode, 0)
            report.write_text(report.read_text().replace('race_free', 'changed', 1).replace('&quot;race_free&quot;', '&quot;different&quot;'))
            self.assertNotEqual(subprocess.run(check, capture_output=True).returncode, 0)

    def test_composer_preserves_unrelated_existing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'input.json').write_text('{"entry":"test"}')
            (root / 'output.json').write_text('{"status":"race_free"}')
            (root / 'kernel.ptx').write_text('.entry test(){ret;}')
            report = root / 'report.md'
            report.write_text('<!-- recorded-relation -->\nComputation follows too late.\n')
            original = report.read_bytes()
            result = subprocess.run(['python3', str(SCRIPT), '--input', str(root / 'input.json'), '--result', str(root / 'output.json'), '--ptx', str(root / 'kernel.ptx'), '--report', str(report)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b'unrelated report', result.stderr)
            self.assertEqual(report.read_bytes(), original)

    def test_real_tma_records_and_full_rule_text_survive(self):
        directory = Path(__file__).parent / 'fixtures/tma'
        source = json.loads((directory / 'output.json').read_text())
        text = facts.retained(source['warnings'])
        # Recover the original endpoint pair for every record from the rendered matrix.
        lines = text.splitlines()
        start = next(i for i,l in enumerate(lines) if l.startswith('| First actor'))
        columns = [int(x.strip()) for x in lines[start].strip('|').split('|')[1:]]
        recovered = {}
        for row in lines[start + 2:]:
            if not row.startswith('|'): break
            cells = [x.strip() for x in row.strip('|').split('|')]
            first = int(cells[0].strip('*'))
            for second, numbers in zip(columns, cells[1:]):
                for number in filter(None, numbers.split(', ')):
                    recovered[int(number)] = (first, second)
        self.assertEqual(recovered, {i:(r['first']['actor'],r['second']['actor']) for i,r in enumerate(source['warnings'],1)})
        self.assertIn('access_index', text)
        self.assertIn('effect_index', text)
        self.assertIn('12498', text)
        model = facts.rules(source, json.loads((directory / 'input.json').read_text()))
        self.assertIn('Async completion and synchronization remain modeled.', model)
        self.assertIn('noalias_unbounded', model)
        self.assertIn('pointer_contract.extent_bytes', model)


if __name__ == '__main__':
    unittest.main()
