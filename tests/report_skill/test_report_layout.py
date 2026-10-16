import copy
import importlib
import json
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[2] / 'skills/generate-kernel-report/scripts'
sys.path.insert(0, str(SCRIPTS))
layout = importlib.import_module('report_layout')

class ReportLayoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ptx = '.version 9.2\n.entry test(){\ncp.async.bulk.commit_group;\ncp.async.bulk.wait_group.read 1;\nret;\n}\n'
        self.input = {'entry':'test','launch':{'block_dim':[32,1,1]}}
        self.result = {'status':'race_free','race_count':0,'warning_count':0}
        self.artifacts = [self.root/name for name in ('input.json','output.json','kernel.ptx')]
        for p, body in zip(self.artifacts, [json.dumps(self.input),json.dumps(self.result),self.ptx]):
            p.write_text(body)
        self.plan = {'version':2,'title':'Tile copy','sections':[{
            'title':'Commit and wait', 'answer':'The code commits the copy group before the read wait.',
            'evidence':[{'kind':'ptx','lines':'3-4','contains':['commit_group','wait_group.read']},
                        {'kind':'result','pointer':'/status','equals':'race_free'}],
            'code':{'caption':'Commit and wait','lines':'3-4','focus':['1']}}]}

    def tearDown(self): self.tmp.cleanup()
    def render(self, plan=None):
        return layout.compose(plan or self.plan,self.input,self.result,self.ptx,self.artifacts)

    def test_base_report_needs_no_plan(self):
        s=layout.compose(None,self.input,self.result,self.ptx,self.artifacts)
        self.assertIn('race_free',s)
        self.assertNotIn('## 2. Code explanation',s)
        self.assertNotIn('Computation',s)

    def test_explanation_is_separate_from_unchanged_facts(self):
        import result_view
        base=layout.compose(None,self.input,self.result,self.ptx,self.artifacts)
        explained=self.render()
        self.assertIn('## 2. Code explanation',explained)
        self.assertIn(result_view.render_evidence(self.input,self.result,self.ptx,self.artifacts).strip(),explained)
        self.assertLess(explained.index('## 1. Verifier result'),explained.index('## 2. Code explanation'))
        self.assertIn('<strong>1</strong>',explained)

    def test_bad_source_token_is_rejected(self):
        self.plan['sections'][0]['evidence'][0]['contains']=['nonexistent_opcode']
        with self.assertRaisesRegex(ValueError,'exact token'): self.render()

    def test_instruction_links_follow_declared_isa_and_keep_explicit_override(self):
        ptx = '.version 9.2\ntcgen05.ld.sync.aligned.32x32b.x8.b32{%r1},[%r2];\n'
        code = {'caption':'Read the accumulator','lines':'2'}
        rendered = layout.code_view(code, ptx)
        self.assertIn('href="' + layout.ISA + 'tcgen05-instructions-tcgen05-ld"', rendered)
        self.assertNotIn('<a ', layout.code_view(code, ptx.replace('9.2','9.1')))
        opcode = 'tcgen05.ld.sync.aligned.32x32b.x8.b32'
        code['links'] = {opcode:'https://example.org/checked-instruction'}
        rendered = layout.code_view(code, ptx)
        self.assertIn('href="https://example.org/checked-instruction"', rendered)
        self.assertNotIn('docs.nvidia.com', rendered)
        self.assertIsNone(layout.instruction_url('tcgen05.ld_extra', layout.LINKS_92))

    def test_instruction_terms_link_without_touching_registers_code_or_existing_links(self):
        original = ('Read with `tcgen05.ld`; wait with tcgen05.wait::ld, using `%r1`.\n'
                    '[`tcgen05.ld`](https://example.org/explanation)\n'
                    '```ptx\ntcgen05.ld;\n```\n'
                    '<pre><code>tcgen05.ld;</code></pre>\n')
        rendered = layout.technical_prose(original, layout.LINKS_92)
        self.assertEqual(rendered.count('https://docs.nvidia.com/'), 2)
        self.assertIn('using `%r1`.', rendered)
        self.assertIn('[`tcgen05.ld`](https://example.org/explanation)', rendered)
        self.assertIn('```ptx\ntcgen05.ld;\n```', rendered)
        self.assertIn('<pre><code>tcgen05.ld;</code></pre>', rendered)

    def test_ptx93_cluster_and_copy_links_use_the_matching_manual(self):
        ptx = '.version 9.3\nbarrier.cluster.arrive.relaxed;\ncp.async.cg.shared.global [%r1], [%rd1], 16, %r2;\n'
        rendered = layout.code_view({'caption': 'Cluster and copy sites', 'lines': '2-3'}, ptx)
        self.assertIn(layout.ISA_93 + 'parallel-synchronization-and-communication-instructions-barrier-cluster', rendered)
        self.assertIn(layout.ISA_93 + 'data-movement-and-conversion-instructions-cp-async', rendered)
        self.assertNotIn('/13.2.0/', rendered)
        self.assertEqual(layout.instruction_docs(ptx.replace('9.3', '9.4')), {})

    def test_operand_guide_names_real_tokens_without_renaming_instructions(self):
        ptx = 'ld.shared.b32 %r12, [%r7];'
        code = {'caption':'Read buffer handle', 'lines':'1',
                'operands':[{'token':'%r7','meaning':'Handle storage address'}]}
        rendered = layout.code_view(code, ptx)
        self.assertIn('Handle storage address (<code>%r7</code>)', rendered)
        instructions = rendered.split('<pre><code>',1)[1].split('</code></pre>',1)[0]
        self.assertEqual(re.sub('<[^>]+>','',instructions),ptx)
        code['operands'][0]['token'] = '%r1'
        with self.assertRaisesRegex(ValueError, 'absent'): layout.code_view(code, ptx)

    def test_excerpt_source_jump_uses_supplied_ptx_and_selected_physical_range(self):
        rendered = layout.compose(self.plan,self.input,self.result,self.ptx,self.artifacts,
                                  report_path=self.root/'report.md')
        self.assertIn('href="kernel.ptx#L3-L4"', rendered)
        self.assertNotIn('output.json#L3', rendered)

    def test_reference_names_the_supplied_output_artifact(self):
        rendered=self.render()
        self.assertIn('output.json#/status',rendered)
        self.assertNotIn('result.json/status',rendered)

    def test_modified_tool_result_is_rejected_by_current_checker(self):
        from check_report import check_current
        output=self.root/'report.md'
        original=layout.compose(None,self.input,self.result,self.ptx,self.artifacts)
        output.write_text(original.replace('NO RACES REPORTED','CONFIRMED SAFE'))
        self.assertTrue(check_current(output,self.artifacts[2],self.artifacts[0],self.artifacts[1]))

    def test_identity_mismatch_preserves_previous_report(self):
        output=self.root/'out'/'report.md'
        common=[sys.executable,str(SCRIPTS/'report_facts.py'),'--input',str(self.artifacts[0]),'--result',str(self.artifacts[1]),'--ptx',str(self.artifacts[2]),'--report',str(output)]
        first=subprocess.run(common,capture_output=True,text=True)
        self.assertEqual(first.returncode,0,first.stderr)
        original=output.read_bytes()
        identity=self.root/'identity.json'
        identity.write_text(json.dumps({'files':{'input.json':{'sha256':'0'*64}}}))
        failed=subprocess.run(common+['--identity',str(identity)],capture_output=True,text=True)
        self.assertNotEqual(failed.returncode,0)
        self.assertIn('Identity mismatch',failed.stderr)
        self.assertEqual(output.read_bytes(),original)

    def test_entry_links_and_separate_equivalence_survive_cli_recomposition(self):
        from urllib.parse import unquote
        output = self.root / 'delivery/kernel/report.md'
        numerical = self.root / 'equivalence.json'
        numerical.write_text('{"status":"not_run"}')
        args = ['--input',str(self.artifacts[0]),'--result',str(self.artifacts[1]),
                '--ptx',str(self.artifacts[2]),'--equivalence',str(numerical),'--report',str(output)]
        for script in ('report_facts.py','check_report.py'):
            run = subprocess.run([sys.executable,str(SCRIPTS/script),*args],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
        content=output.read_text()
        links = re.findall(r'\]\(([^)#]+)\)',content)
        for link in links:
            self.assertTrue((output.parent/unquote(link)).is_file(),link)
        self.assertIn('[equivalence.json](../../equivalence.json)',content)
        self.assertIn('does not replace the memory/synchronization verdict',content)
        self.assertEqual(list(output.parent.iterdir()),[output])

    def test_absent_pointer_is_not_null(self):
        self.plan['sections'][0]['evidence']=[{'kind':'result','pointer':'/absent'}]
        with self.assertRaisesRegex(ValueError,'absent'):self.render()
        self.result['absent']=None
        self.assertIn('null',self.render())

    def test_exact_value_checks_type_as_well_as_equality(self):
        self.plan['sections'][0]['evidence']=[{'kind':'result','pointer':'/race_count','equals':False}]
        with self.assertRaisesRegex(ValueError,'value mismatch'):self.render()

    def test_json_pointer_escaping_and_array_index(self):
        self.assertIsNone(layout.json_pointer({'a/b':{'~x':[None]}},'/a~1b/~0x/0'))
        with self.assertRaises(ValueError):layout.json_pointer([1],'/-1')

    def test_draft_cannot_override_status_or_fact_block(self):
        for key in ('status','race_count','verifier_result'):
            plan=copy.deepcopy(self.plan);plan[key]='success'
            with self.assertRaisesRegex(ValueError,'unknown'):self.render(plan)

    def test_plain_context_and_longer_explanation_do_not_force_shapes(self):
        self.plan['context']={'summary':'A related operation.', 'evidence':[{'kind':'input','pointer':'/entry'}]}
        self.plan['sections'][0]['answer']='A necessary explanation. '*120
        self.assertIn('A necessary explanation.',self.render())

    def test_pipeline_context_has_evidence_and_does_not_replace_tool_result(self):
        self.plan['context'] = {'summary':'A buffer is produced, read and released.',
            'evidence':[{'kind':'input','pointer':'/entry'}],
            'diagram':{'kind':'mermaid','source':'flowchart LR\n A[Produce] --> B[Read]',
                       'meaning':'Edges summarize source stages.'}}
        rendered = self.render()
        self.assertLess(rendered.index('flowchart LR\n A[Produce]'), rendered.index('### 2.2 Commit and wait'))
        self.assertIn('Context evidence', rendered)
        self.plan['context']['evidence'][0]['pointer'] = '/absent'
        with self.assertRaises(ValueError):self.render()

    def test_context_sections_and_next_check_share_an_explicit_heading_hierarchy(self):
        self.plan['context'] = {'title':'Buffer lifetime', 'summary':'A buffer is produced and released.',
                                'evidence':[{'kind':'input','pointer':'/entry'}]}
        self.plan['next'] = {'text':'Check the release boundary.',
                             'evidence':[{'kind':'input','pointer':'/entry'}]}
        self.plan['sections'][0]['steps'] = [{'label':'Source derivation','text':'Check the original sequence.',
                                             'evidence':self.plan['sections'][0]['evidence']}]
        rendered = self.render()
        self.assertEqual(re.findall(r'^## (.+)$', rendered, re.M),
                         ['1. Verifier result','2. Code explanation','Evidence appendix'])
        self.assertEqual(re.findall(r'^### (2\..+)$', rendered, re.M),
                         ['2.1 Buffer lifetime','2.2 Commit and wait','2.3 Next check'])
        self.assertIn('#### Source derivation', rendered)
        self.plan.pop('context')
        rendered = self.render()
        self.assertEqual(re.findall(r'^### (2\..+)$', rendered, re.M),
                         ['2.1 Commit and wait','2.2 Next check'])

    def test_context_derivation_keeps_explanation_ahead_of_checked_references(self):
        self.plan['context'] = {
            'summary':'The copy group is committed before its wait.',
            'evidence':[{'kind':'input','pointer':'/entry'}],
            'evidence_title':'Follow the copy group',
            'steps':[{'label':'Commit, then wait', 'text':'The wait follows the committed group.',
                      'evidence':[{'kind':'ptx','lines':'3-4','contains':['commit_group','wait_group.read']}],
                      'code':self.plan['sections'][0]['code']}]}
        rendered = self.render()
        context = rendered.split('### 2.1 Code context', 1)[1].split('### 2.2', 1)[0]
        self.assertIn('<summary><strong>Follow the copy group</strong></summary>', context)
        self.assertLess(context.index('#### Commit, then wait'), context.index('<pre>'))
        self.assertLess(context.index('<pre>'), context.index('Exact source references'))
        self.assertIn('input.json#/entry', context)
        self.assertIn('physical PTX 3-4:', context)
        self.assertEqual(context.count('<details>'), 2)
        self.plan['context']['steps'][0]['evidence'][0]['contains'] = ['missing_wait']
        with self.assertRaisesRegex(ValueError, 'exact token'): self.render()

    def test_invalid_context_parent_reference_is_not_hidden_by_valid_steps(self):
        self.plan['context'] = {
            'summary':'The kernel input identifies the entry.',
            'evidence':[{'kind':'input','pointer':'/absent'}],
            'steps':[{'label':'Entry', 'text':'The entry name is supplied.',
                      'evidence':[{'kind':'input','pointer':'/entry'}]}]}
        with self.assertRaisesRegex(ValueError, 'absent'): self.render()

    def test_comparison_and_evidence_have_distinct_reading_levels(self):
        section = self.plan['sections'][0]
        section['comparison'] = {'headers':['Phase','Wait'], 'rows':[['0','Startup'],['1','Reuse']]}
        section['evidence_title'] = 'Check both phases'
        section['steps'] = [{'label':'Initial phase', 'text':'Initialization fixes the count.',
                             'evidence':[{'kind':'input','pointer':'/entry'}]}]
        rendered = self.render()
        self.assertIn('| 0 | Startup |', rendered)
        self.assertIn('<summary><strong>Check both phases</strong></summary>', rendered)
        self.assertIn('#### Initial phase\n\nInitialization fixes the count.', rendered)
        section['comparison']['rows'][0].append('extra')
        with self.assertRaisesRegex(ValueError, 'comparison headers'): self.render()

    def test_derivation_remains_readable_before_exact_references(self):
        section = self.plan['sections'][0]
        section['steps'] = [
            {'label':'Commit', 'text':'Commit closes the group.',
             'evidence':[{'kind':'ptx','lines':'3','contains':['commit_group']}]},
            {'label':'Wait', 'text':'The next instruction waits for reads.',
             'evidence':[{'kind':'ptx','lines':'4','contains':['wait_group.read']}]}]
        rendered = self.render()
        start = rendered.index('#### Commit')
        refs = rendered.index('<summary><strong>Exact source references')
        self.assertLess(rendered.index('#### Wait'), refs)
        self.assertNotIn('physical PTX', rendered[start:refs])
        self.assertIn('##### Commit', rendered[refs:])
        self.assertIn('##### Wait', rendered[refs:])
        self.assertIn('physical PTX 3:', rendered[refs:])
        # Folding citations changes visibility, not validation.
        section['steps'][1]['evidence'][0]['contains'] = ['missing_wait']
        with self.assertRaisesRegex(ValueError, 'exact token'): self.render()

    def test_supporting_diagram_stays_with_its_checked_derivation(self):
        section = self.plan['sections'][0]
        section['steps'] = [{'label':'Group count', 'text':'The configured count is two.',
            'diagram':{'kind':'math','source':'16 / 8 = 2',
                       'meaning':'Configured arithmetic, not observed execution.'},
            'evidence':[{'kind':'input','pointer':'/launch/block_dim'}]}]
        rendered = self.render()
        self.assertLess(rendered.index('#### Group count'),rendered.index('16 / 8 = 2'))
        self.assertLess(rendered.index('16 / 8 = 2'),rendered.index('Exact source references'))
        section['steps'][0]['evidence'][0]['pointer'] = '/missing'
        with self.assertRaisesRegex(ValueError,'absent'): self.render()

    def test_entry_mismatch_is_rejected(self):
        self.result['entry']='another'
        with self.assertRaisesRegex(ValueError,'identity mismatch'):self.render()
        self.result.pop('entry');self.input['entry']='missing'
        with self.assertRaisesRegex(ValueError,'absent'):self.render()

    def test_supporting_code_is_checked_inside_its_derivation(self):
        section = self.plan['sections'][0]
        section['steps'] = [{'label':'Check the instructions',
                             'text':'The original sequence is retained here.',
                             'evidence':section['evidence'],
                             'code':section.pop('code')}]
        rendered = self.render()
        start = rendered.index('<summary><strong>Derivation and source references')
        self.assertNotIn('<pre>', rendered[rendered.index('## 2. Code explanation'):start])
        self.assertIn('<pre>', rendered[start:])
        section['steps'][0]['code']['lines'] = '999'
        with self.assertRaises(ValueError):self.render()

    def test_supporting_comparison_stays_with_its_derivation(self):
        section = self.plan['sections'][0]
        section['steps'] = [{'label':'Configured input', 'text':'This input identifies the entry.',
                             'evidence':[{'kind':'input','pointer':'/entry'}],
                             'comparison':{'headers':['Binding','Value'], 'rows':[['Entry','test']]}}]
        rendered = self.render()
        boundary = rendered.index('<summary><strong>Derivation and source references')
        self.assertNotIn('| Entry | test |', rendered[:boundary])
        self.assertIn('| Entry | test |', rendered[boundary:])
        section['steps'][0]['comparison']['rows'][0].append('extra')
        with self.assertRaisesRegex(ValueError, 'comparison headers'):self.render()

    def test_derivation_avoids_duplicate_section_references_without_dropping_others(self):
        section = self.plan['sections'][0]
        shared = section['evidence'][0]
        section['steps'] = [{'label':'Check the sequence', 'text':'Both operations appear here.',
                             'evidence':[shared]}]
        rendered = layout.explanation(self.plan, self.input, self.result, self.ptx)
        reference = layout.evidence_view([shared], self.input, self.result, self.ptx)
        self.assertEqual(rendered.count(reference), 1)
        self.assertIn('output.json#/status', rendered)
        section['steps'][0]['evidence'] = section['evidence']
        self.assertNotIn('Additional section references',
                         layout.explanation(self.plan, self.input, self.result, self.ptx))
        section['evidence'][0] = {'kind':'result', 'pointer':'/absent'}
        with self.assertRaisesRegex(ValueError, 'absent'):self.render()

    def test_optional_source_requires_explicit_input(self):
        self.plan['sections'][0]['evidence']=[{'kind':'source','file':'kernel.cu','lines':'1','contains':['wait']}]
        with self.assertRaisesRegex(ValueError,'not supplied'):self.render()
        s=layout.compose(self.plan,self.input,self.result,self.ptx,self.artifacts,{'kernel.cu':'wait();'})
        self.assertIn('kernel.cu',s)

    def test_bad_plan_does_not_replace_good_report(self):
        output=self.root/'out'/'report.md'
        common=[sys.executable,str(SCRIPTS/'report_facts.py'),'--input',str(self.artifacts[0]),'--result',str(self.artifacts[1]),'--ptx',str(self.artifacts[2]),'--report',str(output)]
        first=subprocess.run(common,capture_output=True,text=True)
        self.assertEqual(first.returncode,0,first.stderr)
        original=output.read_bytes()
        self.plan['sections'][0]['evidence'][0]['lines']='500'
        draft=self.root/'draft.json';draft.write_text(json.dumps(self.plan))
        failed=subprocess.run(common+['--draft',str(draft)],capture_output=True,text=True)
        self.assertNotEqual(failed.returncode,0)
        self.assertEqual(original,output.read_bytes())
        self.assertEqual(list(output.parent.iterdir()),[output])

    def test_witness_timeline_rejects_reversed_events_and_marks_gap(self):
        ref=[{'kind':'result','pointer':'/status'}]
        diagram={'kind':'timeline','basis':'verifier witness', 'sequence':'/counterexample/executed_prefix/events',
                 'participants':[{'id':'p','label':'Model participant','evidence':ref}],
                 'conditions':{k:{'text':'Scoped to the saved model.','evidence':ref} for k in ('participants','branches','iterations','completion')},
                 'steps':[{'participant':'p','action':'First','event':1,'evidence':ref},
                          {'participant':'p','action':'Later','event':3,'evidence':ref}]}
        self.result['counterexample']={'executed_prefix':{'events':[1,2,3],'truncated':True,'total':90}}
        s=layout.timeline_view(diagram,self.input,self.result,self.ptx)
        self.assertIn('Saved events omitted',s)
        self.assertIn('Further events omitted',s)
        diagram['steps'].reverse()
        with self.assertRaisesRegex(ValueError,'reverses'):layout.timeline_view(diagram,self.input,self.result,self.ptx)

    def test_possible_interleaving_requires_conditions_and_evidence(self):
        diagram={'kind':'timeline','basis':'possible interleaving','participants':[{'id':'p','label':'Producer','evidence':[]}],'steps':[], 'conditions':{}}
        with self.assertRaises(ValueError):layout.timeline_view(diagram,self.input,self.result,self.ptx)

    def test_branch_picture_derives_comparison_and_polarity(self):
        ptx = 'setp.ne.s32 %p1, %r1, %r2;\n@%p1 bra end;\nst.local.u32 [%rd1], %r3;\nend:\n'
        diagram = {'kind':'branch','comparison_line':1,'branch_line':2,'effect_line':3,'inputs':['Lane slot','Owner slot']}
        rendered = layout.branch_diagram(diagram, ptx)
        self.assertIn('Lane slot ≠ Owner slot', rendered)
        self.assertIn('%r1 and %r2 (≠)', rendered)
        self.assertIn('G -->|"true: skip"| J', rendered)
        self.assertIn('G -->|"false"| E', rendered)
        self.assertIn('E --> J', rendered)
        self.assertIn('Continue<br/>PTX 4', rendered)
        self.assertIn('when this comparison is true', rendered)
        inverted = layout.branch_diagram(diagram, ptx.replace('@%p1','@!%p1'))
        self.assertIn('G -->|"false: skip"| J', inverted)
        self.assertIn('when this comparison is false', inverted)

    def test_branch_picture_rejects_an_operation_not_bypassed(self):
        diagram = {'kind':'branch','comparison_line':1,'branch_line':2,'effect_line':4,'inputs':['Left','Right']}
        with self.assertRaisesRegex(ValueError, 'bypass'):
            layout.branch_diagram(diagram, 'setp.eq.s32 %p1, %r1, %r2;\n@%p1 bra end;\nend:\nst.global.u32 [%rd1], %r3;\n')

    def test_branch_picture_cannot_skip_an_intervening_control_transfer(self):
        diagram = {'kind':'branch','comparison_line':1,'branch_line':2,'effect_line':4,'inputs':['Left','Right']}
        with self.assertRaisesRegex(ValueError, 'intervening'):
            layout.branch_diagram(diagram, 'setp.eq.s32 %p1, %r1, %r2;\n@%p1 bra end;\nbra other;\nst.global.u32 [%rd1], %r3;\nend:\n')

    def test_branch_picture_rejoin_rejects_a_transfer_after_effect(self):
        diagram = {'kind':'branch','comparison_line':1,'branch_line':2,'effect_line':3,'inputs':['Left','Right']}
        with self.assertRaisesRegex(ValueError, 'intervening'):
            layout.branch_diagram(diagram, 'setp.eq.s32 %p1, %r1, %r2;\n@%p1 bra end;\nst.local.u32 [%rd1], %r3;\nbra other;\nend:\n')

    def test_branch_picture_checks_predicate_redefinitions(self):
        diagram = {'kind':'branch','comparison_line':1,'branch_line':3,'effect_line':4,'inputs':['Left','Right']}
        for rewrite in ('not.pred %p1, %p1;', '@%p2 not.pred %p1, %p1;',
                        'setp.lt.s32 %p2|%p1, %r3, %r4;'):
            with self.subTest(rewrite=rewrite), self.assertRaisesRegex(ValueError, 'redefined'):
                layout.branch_diagram(diagram, 'setp.eq.s32 %p1, %r1, %r2;\n' + rewrite + '\n@%p1 bra end;\nst.global.u32 [%rd1], %r3;\nend:\n')

    def test_guarded_path_keeps_bypass_to_unconditional_arrival(self):
        ptx = ('setp.ne.b32 %p1, %r1, 0;\n@%p1 bra join;\n'
               'st.async.shared::cluster.b32 [%r2], %r3, [%r4];\njoin:\n'
               'mbarrier.arrive.shared::cta.b64 _, [%r5];\n')
        diagram = {'kind':'guarded-path', 'comparison_line':1, 'branch_line':2,
                   'effect_line':3, 'join_line':5, 'inputs':['Lane bit 16','0']}
        rendered = layout.guarded_path(diagram, ptx)
        self.assertIn('G -->|"true · skip"| J', rendered)
        self.assertIn('G -->|"false"| E', rendered)
        self.assertIn('E --> J', rendered)
        self.assertIn('st.async<br/>PTX 3', rendered)
        self.assertIn('mbarrier.arrive<br/>PTX 5', rendered)
        self.assertIn('do not establish asynchronous completion', rendered)
        inverted = layout.guarded_path(diagram, ptx.replace('@%p1', '@!%p1'))
        self.assertIn('G -->|"false · skip"| J', inverted)
        for bad in (ptx.replace('mbarrier.arrive', '@%p2 mbarrier.arrive'),
                    ptx.replace('join:', 'join: bra other;')):
            with self.assertRaises(ValueError): layout.guarded_path(diagram, bad)

    def test_guarded_path_rejects_extra_control_before_common_operation(self):
        ptx = ('setp.ne.b32 %p1, %r1, 0;\n@%p1 bra join;\n'
               'st.shared.b32 [%r2], %r3;\njoin:\nbra other;\n'
               'mbarrier.arrive.shared::cta.b64 _, [%r5];\n')
        diagram = {'kind':'guarded-path', 'comparison_line':1, 'branch_line':2,
                   'effect_line':3, 'join_line':6, 'inputs':['Left','Right']}
        with self.assertRaisesRegex(ValueError, 'intervening'):
            layout.guarded_path(diagram, ptx)

    def test_guarded_path_starts_at_the_affected_load_and_names_lane_paths(self):
        ptx = ('ld.shared.b32 %r7, [%r8];\nsetp.ne.b32 %p1, %r1, 0;\n'
               '@%p1 bra join;\nst.shared.b32 [%r2], %r3;\njoin:\n'
               'mbarrier.arrive.shared::cta.b64 _, [%r5];\n')
        diagram = {'kind':'guarded-path','before_line':1,'comparison_line':2,'branch_line':3,
                   'effect_line':4,'join_line':6,'inputs':['Lane bit 16','0'],
                   'path_labels':{'true':'Lanes 16–31','false':'Lanes 0–15'}}
        rendered = layout.guarded_path(diagram, ptx)
        self.assertIn('P["ld.shared<br/>PTX 1"]', rendered)
        self.assertNotIn('G{', rendered)
        self.assertIn('P -->|"Lanes 16–31 · skip"| J', rendered)
        self.assertIn('P -->|"Lanes 0–15"| E', rendered)
        self.assertIn('`%r1 ≠ 0`', rendered)
        inverted = layout.guarded_path(diagram, ptx.replace('@%p1','@!%p1'))
        self.assertIn('P -->|"Lanes 0–15 · skip"| J', inverted)
        unnamed = dict(diagram)
        unnamed.pop('path_labels')
        self.assertIn('P["ld.shared<br/>PTX 1"] --> G', layout.guarded_path(unnamed, ptx))
        with self.assertRaises(ValueError):
            layout.guarded_path(diagram, ptx.replace('ld.shared', '@%p2 ld.shared'))
        diagram['before_line'] = 2
        with self.assertRaisesRegex(ValueError, 'before the selected comparison'):
            layout.guarded_path(diagram, ptx)

    def test_guarded_path_rejects_a_control_transfer_after_the_preceding_operation(self):
        ptx = ('ld.shared.b32 %r7, [%r8];\nbra alternate;\n'
               'setp.ne.b32 %p1, %r1, 0;\n@%p1 bra join;\n'
               'st.shared.b32 [%r2], %r3;\njoin:\n'
               'mbarrier.arrive.shared::cta.b64 _, [%r5];\n')
        diagram = {'kind':'guarded-path','before_line':1,'comparison_line':3,'branch_line':4,
                   'effect_line':5,'join_line':7,'inputs':['Left','Right']}
        with self.assertRaisesRegex(ValueError, 'without a control transfer'):
            layout.guarded_path(diagram, ptx)

    def test_guarded_path_rejects_ambiguous_scoped_target(self):
        ptx = ('setp.ne.b32 %p1, %r1, 0;\n@%p1 bra join;\n'
               'st.shared.b32 [%r2], %r3;\njoin:\n'
               'mbarrier.arrive.shared::cta.b64 _, [%r5];\njoin:\n')
        diagram = {'kind':'guarded-path', 'comparison_line':1, 'branch_line':2,
                   'effect_line':3, 'join_line':5, 'inputs':['Left','Right']}
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            layout.guarded_path(diagram, ptx)

    def test_guarded_path_rejects_predicated_effect(self):
        ptx = ('setp.ne.b32 %p1, %r1, 0;\n@%p1 bra join;\n'
               '@%p2 st.shared.b32 [%r2], %r3;\njoin:\n'
               'mbarrier.arrive.shared::cta.b64 _, [%r5];\n')
        diagram = {'kind':'guarded-path', 'comparison_line':1, 'branch_line':2,
                   'effect_line':3, 'join_line':5, 'inputs':['Left','Right']}
        with self.assertRaisesRegex(ValueError, 'additional predicate'):
            layout.guarded_path(diagram, ptx)

    def test_real_gqa_guard_is_not_a_mandatory_store(self):
        ptx = (SCRIPTS.parents[2] / 'corpus/cutlass93_blackwell_low_latency_gqa.compute_100a.ptx').read_text()
        diagram = {'kind':'guarded-path','comparison_line':7377,'branch_line':7378,
                   'effect_line':7380,'join_line':7409,'inputs':['Lane bit 16','0']}
        rendered = layout.guarded_path(diagram, ptx)
        self.assertIn('st.async<br/>PTX 7380', rendered)
        self.assertIn('Both paths reach PTX 7409', rendered)
        self.assertIn('%r960 ≠ 0', rendered)
