import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'skills/generate-kernel-report/scripts/source_context.py'
spec = importlib.util.spec_from_file_location('source_context', SCRIPT)
context = importlib.util.module_from_spec(spec)
spec.loader.exec_module(context)


class SourceContextTests(unittest.TestCase):
    def index(self, body):
        return context.SourceIndex('.entry selected() {\n' + body + '\n}\n', 'selected')

    def test_entry_boundary_ignores_comments_strings_and_similar_name(self):
        ptx = '''// .entry selected(){bad;}
.entry selected_extra(){tcgen05.dealloc.fake %r0, 9;}
.entry selected() {
// } .entry intruder() {
.file 1 "file with } brace"
/* } */ mov.u32 %r1, %tid.x;
{ .reg .pred P1; setp.eq.u32 P1, %r1, 0; }
ret;
}
.entry other(){tcgen05.dealloc.fake %r0, 42;}
'''
        index = context.SourceIndex(ptx, 'selected')
        self.assertEqual((index.first, index.last), (3, 9))
        self.assertEqual(len(index.instructions), 3)
        self.assertNotIn('tcgen05.dealloc', context.render({'entry':'selected'}, {}, ptx))

    def test_invalid_or_ambiguous_entry_is_error(self):
        for ptx in ['.entry other(){ret;}', '.entry selected(){ret;',
                    '.entry selected();\n.entry other(){ret;}',
                    '.entry selected(){ret;}\n.entry selected(){ret;}']:
            with self.assertRaises(ValueError):
                context.SourceIndex(ptx, 'selected')

    def test_scoped_inline_predicates_and_labels_do_not_merge(self):
        index = self.index('''{
.reg .pred P1;
again: mbarrier.try_wait.parity.shared.b64 P1, [%r1], 0;
@!P1 bra again;
}
{
.reg .pred P1;
again: mbarrier.try_wait.parity.shared.b64 P1, [%r2], 1;
@!P1 bra again;
}''')
        keys = [key for key in index.definitions if key[1]=='P1']
        self.assertEqual(len(keys), 2)
        self.assertTrue(all(len(index.definitions[key])==1 for key in keys))
        self.assertEqual([i['target'][0] for i in index.branches], [4, 9])
        self.assertNotEqual(index.branches[0]['uses'], index.branches[1]['uses'])

    def test_same_line_redefinitions_are_counted_and_have_columns(self):
        index = self.index('mov.u32 %r1, 0; mov.u32 %r1, %tid.x;\nsetp.eq.u32 %p1, %r1, 0;\n@%p1 bra done;\ndone: ret;')
        defs = index.definitions[(0,'%r1')]
        self.assertEqual([i['column'] for i in defs], [1,17])
        text = context.render({'entry':'selected'}, {}, index.text)
        self.assertIn('2 REPEATED definitions; none selected as reaching', text)
        self.assertIn('2:1', text)
        self.assertIn('2:17', text)

    def test_role_chain_includes_shfl_and_definition_mutations(self):
        index = self.index('''mov.u32 %r1, %tid.x;
shr.u32 %r2, %r1, 5;
shfl.sync.idx.b32 %r3|%p0, %r2, 0, 31, -1;
setp.eq.u32 %p1, %r3, 2;
@%p1 bra consumer;
mov.u32 %r3, 9;
consumer: tcgen05.dealloc.cta_group::1.sync.aligned.b32 %r8, %r9;''')
        self.assertEqual([i['line'] for i in index.role_guards()], [6])
        self.assertEqual(len(index.definitions[(0,'%r3')]), 2)
        # Side-effect source operands must never become register definitions.
        self.assertNotIn((0,'%r8'), index.definitions)
        text = context.render({'entry':'selected'}, {}, index.text)
        self.assertIn('mov.u32 %r1, %tid.x', text)
        self.assertIn('shr.u32 %r2, %r1, 5', text)
        self.assertIn('setp.eq.u32 %p1, %r3, 2', text)

    def test_saved_source_priority_and_foreign_or_outside_endpoints(self):
        index = self.index('mov.u32 %r1, %tid.x;\nret;')
        result = {'source_events':[{'event':1,'line':2,'file':'corpus/a.ptx'},
                                   {'event':2,'line':3,'file':'corpus/a.ptx'},
                                   {'event':3,'line':99,'file':'corpus/a.ptx'},
                                   {'event':4,'line':2,'file':'elsewhere/a.ptx'}],
                  'counterexample':{'first_unexecuted':2}}
        sites, ordered, rejected = context.saved_sites(result, index, '/repo/corpus/a.ptx')
        self.assertEqual(ordered, [3,2])
        self.assertEqual(len(rejected), 2)
        self.assertEqual(len(sites[2]), 1)

    def test_dynamic_ids_are_never_used_as_physical_lines(self):
        index = self.index('tcgen05.dealloc.cta_group::1.sync.aligned.b32 %r1, 32;')
        text = context.render({'entry':'selected'}, {'failed_required_orders':[{'from_instruction':15736,'to_instruction':3546}]}, index.text)
        self.assertIn('Dynamic instruction IDs are not PTX line numbers', text)
        self.assertIn('tcgen05.dealloc: 1 sites; 2:1', text)
        self.assertNotIn('Anchor L15736', text)

    def test_tab_and_space_directives_never_consume_following_instruction(self):
        for whitespace in [' ', '\t', ' \t ']:
            with self.subTest(whitespace=repr(whitespace)):
                index = self.index(f'''.file{whitespace}1 "kernel.py"
.loc{whitespace}1 541 1
st.shared.b32 [shared+8], 7;
.pragma{whitespace}"nounroll"
mov.u32 %r1, %tid.x;
.loc{whitespace}1 542 1
stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 [%r1], {{%r2, %r3, %r4, %r5}};''')
                self.assertEqual([i['line'] for i in index.instructions], [4,6,8])
                matrix = index.instructions[-1]
                self.assertEqual(matrix['defs'], [])
                self.assertEqual(set(matrix['uses']), {(0,f'%r{n}') for n in range(1,6)})
                self.assertEqual([i['line'] for i in index.definitions[(0,'%r1')]], [6])

    def test_saved_sites_cover_result_families_without_guessing_dynamic_ids(self):
        index = self.index('\n'.join(['mov.b32 %r1, 0;']*7))
        result = {'races':[{'first':{'location':{'file':'kernel.ptx','line':2}}}],
                  'unsupported_memory_pair':{'second':{'location':{'file':'kernel.ptx','line':3}}},
                  'location':{'file':'kernel.ptx','line':4},
                  'diagnostics':[{'location':{'line':5}}, {'ptx_line':6},
                                 {'file':'kernel.ptx','line':7}, {'instruction':8}]}
        sites, ordered, rejected = context.saved_sites(result, index, 'kernel.ptx')
        self.assertEqual(ordered, [2,3,4,6,7])
        self.assertEqual(len(rejected), 1)
        self.assertIn('no source-file identity', rejected[0])
        self.assertIn('races[0].first.location', sites[2][0])
        self.assertIn('diagnostics[1].ptx_line', sites[6][0])

    def test_unqualified_source_event_is_not_a_ptx_coordinate(self):
        index = self.index('ret;')
        sites, ordered, rejected = context.saved_sites(
            {'source_events':[{'event':9,'line':2}]}, index, 'kernel.ptx')
        self.assertFalse(sites)
        self.assertFalse(ordered)
        self.assertIn('no source-file identity', rejected[0])

    def test_named_and_unnamed_supplied_arguments_keep_values_and_load_sites(self):
        ptx = '''.entry selected(){
ld.param.u32 %r1, [selected_param_12];
mov.b64 %rd1, selected_param_3;
ld.param.u32 %r2, [%rd1+8];
ret;
}'''
        inp = {'entry':'selected','declarations':[{'parameter':3,'offset':8,'name':'count','type':'u32'}],
               'arguments':[{'parameter':12,'value':0,'type':'u32'}, {'field':'count','value':4}]}
        text = context.render(inp, {}, ptx)
        self.assertIn('parameter 12 + 0 (u32) = 0; %r1@2', text)
        self.assertIn('count (u32) = 4; %r2@4', text)

    def test_reverse_edges_distinguish_taken_fallthrough_and_jump_cutoff(self):
        index = self.index('''mov.u32 %r1, %tid.x;
setp.eq.u32 %p1, %r1, 0;
@%p1 bra first;
setp.eq.u32 %p2, %r1, 64;
@%p2 bra second;
bra done;
first: st.shared.b32 [%r2], 7;
bra done;
second: tcgen05.dealloc.cta_group::1.sync.aligned.b32 %r8, 32;
done: ret;''')
        leads, truncated = index.predecessor_branches(10, index.role_guards())
        self.assertFalse(truncated)
        self.assertEqual({(i['line'],i['edge'],i['edge_to'][0]) for i in leads},
                         {(6,'taken',10),(4,'fallthrough',5)})
        # A lexical predecessor that jumps elsewhere cannot enter this block.
        self.assertNotIn((4,'taken'), {(i['line'],i['edge']) for i in leads})
        text = context.render({'entry':'selected'}, {}, index.text, at=[10])
        self.assertIn('taken edge -> block 10:9; requires %p2=true', text)
        self.assertIn('fallthrough edge -> block 5:1; requires %p1=false', text)

    def test_address_cross_reference_never_selects_repeated_definition(self):
        index = self.index('''mov.b32 %r1, shared_buffer;
add.s32 %r2, %r1, 80;
mov.b32 %r3, 6;
mbarrier.init.shared.b64 [%r2], %r3;
add.s32 %r4, %r1, 80;
mbarrier.arrive.shared.b64 _, [%r4];
mov.b32 %r5, 0;
mov.b32 %r5, %r2;
mbarrier.try_wait.parity.shared.b64 %p1, [%r5], 1;''')
        self.assertEqual(index.address_key('%r4', 0), ('shared_buffer', 80))
        self.assertIsNone(index.address_key('%r5', 0))
        text = context.render({'entry':'selected'}, {}, index.text)
        self.assertIn('shared_buffer + 80: mbarrier.init.shared.b64@5, mbarrier.arrive.shared.b64@7', text)
        self.assertIn('mov.b32 %r3, 6', text)

    def test_reverse_path_search_finds_nonlocal_role_gate(self):
        index = self.index('''mov.u32 %r1, %tid.x;
setp.lt.u32 %p1, %r1, 32;
@%p1 bra consumer;
ret;
consumer:
mov.u32 %r8, 4;
loop: @%p2 bra done;
bra loop;
done: tcgen05.dealloc.cta_group::1.sync.aligned.b32 %r8, 32;''')
        leads, truncated = index.predecessor_branches(11, index.role_guards())
        self.assertFalse(truncated)
        self.assertEqual([i['line'] for i in leads], [4])

    def test_diagnostic_repetitions_do_not_starve_other_categories(self):
        index = self.index('\n'.join(['mov.b32 %r1, 0;'] * 12))
        diagnostics = [{'code': 'repeated', 'ptx_line': n} for n in range(2, 11)]
        diagnostics += [{'code': 'other', 'ptx_line': 12}]
        result = {'diagnostics': diagnostics}
        sites, saved, _ = context.saved_sites(result, index, 'kernel.ptx')
        self.assertEqual(context.representative_sites(result, sites, saved), [2, 12])
        text = context.render({'entry': 'selected'}, result, index.text)
        self.assertIn('### Focus L12', text)
        self.assertNotIn('### Focus L3', text)

    def test_focus_retains_control_operand_chain_and_role_gate(self):
        index = self.index("""mov.u32 %tid, %tid.x;
shr.u32 %warp, %tid, 5;
setp.eq.u32 %owner, %warp, 2;
@!%owner bra done;
ld.global.b32 %index, [%ptr];
setp.ge.s32 %valid, %index, 0;
selp.b32 %size, 16, 0, %valid;
cp.async.cg.shared.global [%dst], [%src], 16, %size;
done: ret;""")
        text = context.render({'entry': 'selected'}, {}, index.text, at=[9])
        focus = text.split('## Focus evidence')[1].split('## Matching-entry operation families')[0]
        self.assertIn('ld.global.b32 %index', focus)
        self.assertIn('setp.ge.s32 %valid', focus)
        self.assertIn('requires %owner=true', focus)
        self.assertIn('setp.eq.u32 %owner, %warp, 2', focus)
        self.assertIn('mov.u32 %tid, %tid.x', focus)

    def test_focus_keeps_all_repeated_predicated_definitions(self):
        index = self.index("""mov.b32 %size, 0;
@%p mov.b32 %size, 16;
cp.async.cg.shared.global [%dst], [%src], 16, %size;""")
        text = context.render({'entry': 'selected'}, {}, index.text, at=[4])
        focus = text.split('## Focus evidence')[1].split('## Matching-entry operation families')[0]
        self.assertIn('2 REPEATED definitions; none selected as reaching', focus)
        self.assertIn('@%p mov.b32 %size, 16', focus)
        self.assertIn('mov.b32 %size, 0', focus)

    def test_focus_groups_release_barrier_initialization_arrivals_and_wait(self):
        index = self.index("""mov.b32 %base, shared_memory;
add.s32 %init, %base, 80;
mov.u32 %count, 160;
mbarrier.init.shared.b64 [%init], %count;
add.s32 %arrival, %base, 80;
mbarrier.arrive.shared.b64 _, [%arrival];
add.s32 %wait, %base, 80;
mov.u32 %phase, 1;
mbarrier.try_wait.parity.shared.b64 %p, [%wait], %phase;
tcgen05.dealloc.cta_group::1.sync.aligned.b32 %tmem, 32;""")
        text = context.render({'entry': 'selected'}, {}, index.text, at=[11])
        focus = text.split('## Focus evidence')[1].split('## Matching-entry operation families')[0]
        self.assertIn('shared_memory + 80', focus)
        self.assertIn('mbarrier.init.shared.b64 [%init], %count', focus)
        self.assertIn('mov.u32 %count, 160', focus)
        self.assertIn('mbarrier.arrive.shared.b64 _, [%arrival]', focus)
        self.assertIn('mov.u32 %phase, 1', focus)
        self.assertIn('establish participants and completed phases', focus)

    def test_predicate_substitution_preserves_not_and_branch_polarity(self):
        index = self.index("""mov.u32 %r1, %tid.x;
setp.ne.u32 %p1, %r1, 9;
not.pred %skip, %p1;
@%skip bra done;
cp.async.cg.shared.global [%dst], [%src], 16;
done: ret;""")
        text = context.render({'entry': 'selected'}, {}, index.text, at=[6])
        self.assertIn('requires %skip=false', text)
        self.assertIn('%p1=true via PTX 4', text)
        # Multiple or predicated definitions cannot be substituted as a fact.
        for body in ['not.pred %p, %q; mov.pred %p, %z;', '@%guard not.pred %p, %q;']:
            ambiguous = self.index(body)
            self.assertEqual(context.predicate_requirement(ambiguous, (0, '%p'), False), [])

    def test_stored_literal_bytes_are_computed_without_guessing_registers(self):
        index = self.index("""st.shared.b32 [buffer], 16777730;
st.shared.s32 [buffer], -1;
st.shared.f32 [buffer], 0f3f800000;
st.shared.b32 [buffer], %r1;""")
        self.assertEqual(context.stored_integer_bits(index.instructions[0]),
                         'literal bit pattern 0x01000202; little-endian bytes 02 02 00 01')
        self.assertIn('ff ff ff ff', context.stored_integer_bits(index.instructions[1]))
        self.assertIsNone(context.stored_integer_bits(index.instructions[2]))
        self.assertIsNone(context.stored_integer_bits(index.instructions[3]))
        self.assertIn('0x01000202; little-endian bytes 02 02 00 01',
                      context.render({'entry': 'selected'}, {}, index.text))

    def test_at_validation_and_cli_error_are_explicit(self):
        with self.assertRaises(ValueError):
            context.render({'entry':'selected'}, {}, '.entry selected(){ret;}', at=[99])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path/'input.json').write_text('{"entry":"missing"}')
            (path/'result.json').write_text('{}')
            (path/'kernel.ptx').write_text('.entry selected(){ret;}')
            result = subprocess.run(['python3',str(SCRIPT),'--input',str(path/'input.json'),'--result',str(path/'result.json'),'--ptx',str(path/'kernel.ptx')],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('matching .entry',result.stderr)
            self.assertFalse(result.stdout)

    def test_packet_and_source_lines_are_bounded(self):
        body = '\n'.join(f'mov.u32 %r{n}, {n}; mbarrier.init.shared.b64 [%r{n}], %r{n};' for n in range(1000))
        text = context.render({'entry':'selected'}, {}, '.entry selected(){\n'+body+'\n}')
        self.assertLess(len(text), context.MAX_CHARS+200)
        self.assertIn('omitted', text)
        self.assertNotIn('Anchor L1000', text)

    def test_actual_fmha_and_gqa_packets_remain_entry_local(self):
        cases = [('cutlass77_blackwell_fmha_fp16.compute_100a',(307789,315527)),
                 ('cutlass93_blackwell_low_latency_gqa.compute_100a',(4096,7515))]
        for case, bounds in cases:
            with self.subTest(case=case):
                inp=json.loads((ROOT/'examples'/case/'input.json').read_text())
                result=json.loads((ROOT/'examples'/case/'output.json').read_text())
                path=ROOT/'corpus'/(case+'.ptx')
                source=path.read_text()
                index=context.SourceIndex(source,inp['entry'])
                self.assertEqual((index.first,index.last),bounds)
                text=context.render(inp,result,source,str(path))
                self.assertLessEqual(len(text),context.MAX_CHARS+200)
                self.assertIn('tcgen05.dealloc',text)
                self.assertIn('tcgen05.wait::ld.sync.aligned',text)
                self.assertIn('mov.u32',text)
                self.assertIn('%tid.x',text)
                if 'fmha' in case:
                    self.assertIn('event 1290360',text)
                    self.assertIn('REPEATED definitions',text)
                else:
                    self.assertIn('No accepted saved source-line endpoints',text)
                    self.assertIn('108080',text)
                    self.assertIn('mov.b32 %r476, 1',text)
                    self.assertIn('mbarrier.arrive.shared::cta.b64@5536',text)
                    self.assertIn('mbarrier.arrive.shared::cta.b64@6589',text)
                    self.assertIn('mov.b32 %r756, 0',text)
                    self.assertIn('parameter 12 + 0 (u32) = 0; %r130@4132',text)
                    leads, _ = index.predecessor_branches(6542, index.role_guards())
                    self.assertIn((4487,'taken',5496), {(i['line'],i['edge'],i['edge_to'][0]) for i in leads})
                    self.assertNotIn(4491, {i['line'] for i in leads})

    def test_actual_gluon_stores_survive_debug_locations_without_python_mapping(self):
        case = 'tritongluon-08-ws-matmul-512'
        inp = json.loads((ROOT/'examples'/case/'input.json').read_text())
        result = json.loads((ROOT/'examples'/case/'output.json').read_text())
        path = ROOT/'corpus'/(case+'.ptx')
        source = path.read_text()
        index = context.SourceIndex(source,inp['entry'])
        instructions = {i['line']:i for i in index.instructions}
        self.assertEqual(instructions[170]['opcode'], 'st.shared.b32')
        self.assertEqual(instructions[1216]['opcode'], 'st.shared.b32')
        sites, _, rejected = context.saved_sites(result, index, str(path))
        self.assertFalse(sites)
        self.assertEqual(len(rejected), 2)
        self.assertTrue(all('08-warp-specialization.py:541' in item for item in rejected))
        text = context.render(inp,result,source,str(path))
        self.assertNotIn('Anchor L541',text)
        stores = next(line for line in text.splitlines() if line.startswith('- st.shared:'))
        self.assertIn('170:2', stores)
        self.assertIn('1216:2', stores)
        self.assertIn('parameter 11 + 0 (u32) = 0; %r20@43',text)
        self.assertIn('parameter 12 + 0 (u32) = 0; %r21@42',text)

    def test_actual_b200_stmatrix_has_only_register_uses(self):
        case = 'thunderkittens__thunderkittens_bf16_b200_gemm__sm_100a'
        inp = json.loads((ROOT/'examples'/case/'input.json').read_text())
        source = (ROOT/'corpus'/(case+'.ptx')).read_text()
        index = context.SourceIndex(source,inp['entry'])
        matrix = next(i for i in index.instructions if i['line']==2679)
        self.assertTrue(matrix['opcode'].startswith('stmatrix.'))
        self.assertEqual(matrix['defs'], [])
        self.assertIn((0,'%r191'), matrix['uses'])
        self.assertNotIn(2679,[i['line'] for i in index.definitions[(0,'%r191')]])


if __name__ == '__main__':
    unittest.main()
