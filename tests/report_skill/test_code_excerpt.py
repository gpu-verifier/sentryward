"""Meaningful invariants for the read-only PTX display helper."""

from html.parser import HTMLParser
import importlib.util
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
SCRIPT = Path(__file__).resolve().parents[2] / 'skills/generate-kernel-report/scripts/code_excerpt.py'
SPEC = importlib.util.spec_from_file_location('code_excerpt', SCRIPT)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


class CodeText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_code = False
        self.parts = []
        self.links = 0
        self.nested_link = False

    def handle_starttag(self, tag, attrs):
        if tag == 'code':
            self.in_code = True
        if tag == 'a':
            self.nested_link |= self.links > 0
            self.links += 1

    def handle_endtag(self, tag):
        if tag == 'code':
            self.in_code = False
        if tag == 'a':
            self.links -= 1

    def handle_data(self, data):
        if self.in_code:
            self.parts.append(data)


def parse_code(rendered):
    parser = CodeText()
    parser.feed(rendered)
    return ''.join(parser.parts), parser


def compact(value):
    return re.sub(r'\s+', '', value)


class CodeExcerptTests(unittest.TestCase):
    def test_compact_vector_operand_and_named_predicate_keep_opcode_emphasis(self):
        source=('LAB_WAIT:\n@P1 bra DONE;\n'
                'tcgen05.ld.sync.aligned.32x32b.x8.b32{%r1,%r2},[%r3];')
        rendered=helper.excerpt(source,'1-3')
        self.assertNotIn('<strong>LAB_WAIT',rendered)
        self.assertIn('@P1 <strong>bra</strong>',rendered)
        self.assertIn('<strong>tcgen05.ld.sync.aligned.32x32b.x8.b32</strong>{',rendered)
        code,_=parse_code(rendered)
        self.assertEqual(compact(code),compact(source))

    def test_wait_group_counts_are_landmarks_without_explicit_focus(self):
        source = ('cp.async.wait_group 0;\n'
                  'cp.async.bulk.wait_group.read 1;\n'
                  'wgmma.wait_group.sync.aligned 0x2;')
        rendered = helper.excerpt(source, '1-3')
        for count in ('0', '1', '0x2'):
            self.assertIn('<strong>' + count + '</strong>;', rendered)
        code, _ = parse_code(rendered)
        self.assertEqual(compact(code), compact(source))

    def test_wait_count_emphasis_ignores_comments_strings_and_other_operands(self):
        source = ('// cp.async.wait_group 1;\n'
                  '.file 1 "cp.async.wait_group 1;"\n'
                  '/* cp.async.bulk.wait_group.read 1; */\n'
                  'mov.u32 %r1, 1; cp.async.wait_group 1; // 1\n'
                  'tcgen05.wait::ld.sync.aligned;')
        rendered = helper.excerpt(source, '1-5')
        self.assertEqual(rendered.count('<strong>1</strong>'), 1)
        self.assertIn('mov.u32</strong> %r1, 1;', rendered)
        self.assertIn('// 1', rendered)
        code, _ = parse_code(rendered)
        self.assertEqual(compact(code), compact(source))

    def test_linked_wait_count_remains_visible_after_wrapping(self):
        source = '@%p1 cp.async.bulk.wait_group.read 1;'
        rendered = helper.excerpt(source, '1', width=25,
                                  links={'cp.async.bulk.wait_group.read': 'https://docs.example.org/wait'})
        self.assertIn('wait_group.read</a>', rendered)
        self.assertIn('<strong>1</strong>', rendered)
        code, parser = parse_code(rendered)
        self.assertEqual(compact(code), compact(source))
        self.assertFalse(parser.nested_link)

    def test_instruction_opcodes_are_emphasized_without_focus(self):
        source = ('mov.u32 %r1, %tid.x;\n'
                  '@%p1 bra L_done;\n'
                  '@!%p2 st.shared.u32 [%r2], %r1;\n'
                  'ret;')
        rendered = helper.excerpt(source, '1-4')
        for opcode in ('mov.u32', 'bra', 'st.shared.u32', 'ret'):
            self.assertIn('<strong>' + opcode + '</strong>', rendered)
        self.assertNotIn('<strong>@%p1</strong>', rendered)
        self.assertNotIn('<strong>@!%p2</strong>', rendered)
        code, _ = parse_code(rendered)
        self.assertEqual(compact(code), compact(source))

    def test_labels_directives_and_compiler_comments_are_not_opcodes(self):
        source = ('L_done:\n'
                  '.reg .u32 %r1;\n'
                  '// mov.u32 %r1, 0;\n'
                  '/* compiler comment\n'
                  'bra L_done;\n'
                  '*/\n'
                  'L_store: @%p1 st.shared.u32 [%r2], %r1; // st.shared.u32\n')
        rendered = helper.excerpt(source, '1-3,5,7')
        self.assertNotIn('<strong>L_done</strong>', rendered)
        self.assertNotIn('<strong>.reg</strong>', rendered)
        self.assertNotIn('<strong>mov.u32</strong>', rendered)
        self.assertNotIn('<strong>bra</strong>', rendered)
        self.assertEqual(rendered.count('<strong>st.shared.u32</strong>'), 1)
        self.assertIn('// st.shared.u32', rendered)

    def test_supplied_link_substitutes_for_automatic_opcode_emphasis(self):
        source = '@!%p1 st.shared.u32 [%r2], %r1;'
        links = {'st.shared.u32': 'https://docs.example.org/store'}
        rendered = helper.excerpt(source, '1', focus=('@!%p1',), links=links)
        self.assertIn('<a href="https://docs.example.org/store">st.shared.u32</a>', rendered)
        self.assertNotIn('<strong>st.shared.u32</strong>', rendered)
        self.assertIn('<strong>@!%p1</strong>', rendered)
        explicit = helper.excerpt(source, '1', focus=('st.shared.u32',), links=links)
        self.assertIn('<a href="https://docs.example.org/store"><strong>st.shared.u32</strong></a>', explicit)

    def test_statements_preserved_across_wrapping_and_omissions(self):
        source = ('  mov.u32 %r1, %tid.x;\n'
                  '  cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%rd5], [%rd6, {%r1, %r2, %r3}], 0;\n'
                  '  add.u32 %r4, %r1, 1;\n'
                  '  cp.async.bulk.wait_group.read 1;\n')
        rendered = helper.excerpt(source, '1-2,4', width=45, caption='Copy and wait')
        code, _ = parse_code(rendered)
        expected = '\n'.join(source.splitlines()[index - 1] for index in (1, 2, 4))
        self.assertEqual(compact(code.replace('\u2026', '')), compact(expected))
        self.assertEqual(code.count('\u2026'), 1)
        self.assertIn('physical PTX 1\u20132, 4', rendered)
        self.assertIn(' \u00b7 display wrapping added', rendered)
        self.assertIn('Copy and wait', rendered)

    def test_short_excerpt_caption_has_no_wrapping_metadata(self):
        rendered = helper.excerpt('ret;', '1', caption='Return')
        self.assertTrue(rendered.startswith('<p><strong>Return</strong> \u00b7 physical PTX 1</p>'))
        self.assertNotIn('wrapping', rendered)
        self.assertNotIn('target', rendered)

    def test_source_range_jump_preserves_gaps_and_exact_instructions(self):
        source = 'mov.u32 %r1, 0;\nadd.u32 %r1, %r1, 1;\nret;'
        rendered = helper.excerpt(source,'1,3',source_url='../corpus/kernel%20name.ptx')
        self.assertIn('href="../corpus/kernel%20name.ptx#L1-L3"', rendered)
        self.assertIn('physical PTX 1, 3</a>', rendered)
        code, _ = parse_code(rendered)
        self.assertEqual(compact(code.replace('…','')),compact('mov.u32 %r1, 0;ret;'))
        for invalid in ('javascript:alert(1)', '//example.org/code', 'https://example.org/a#L99'):
            with self.assertRaises(ValueError): helper.excerpt(source,'1',source_url=invalid)

    def test_fragmented_excerpt_keeps_exact_code_with_bounded_source_caption(self):
        source = '\n'.join(f'mov.u32 %r{number}, {number};' for number in range(1, 10))
        rendered = helper.excerpt(source, '1,3,5,7,9', caption='Selected initialization')
        code, _ = parse_code(rendered)
        expected = '\n'.join(source.splitlines()[index - 1] for index in (1, 3, 5, 7, 9))
        self.assertEqual(compact(code.replace('\u2026', '')), compact(expected))
        self.assertEqual(code.count('\u2026'), 4)
        self.assertIn('physical PTX 1\u20139 (selected lines; gaps marked)', rendered)

    def test_escaping_happens_before_markup(self):
        source = '  // <script> & "quote"\n  setp.lt.u32 %p1, %r1, 2;\n'
        rendered = helper.excerpt(source, '1-2', focus=('<script>', '%p1'), caption='Guard < limit & owner')
        self.assertNotIn('<script>', rendered)
        self.assertIn('<strong>&lt;script&gt;</strong>', rendered)
        self.assertIn('Guard &lt; limit &amp; owner', rendered)
        code, _ = parse_code(rendered)
        self.assertEqual(compact(code), compact(source))

    def test_no_break_inside_tokens_strings_or_line_comment(self):
        opcode = 'cp.async.bulk.tensor.3d.global.shared::cta.bulk_group'
        source = opcode + ' [%rd123456789], {%r1,%r2,%r3,%r4}; // keep the exact trailing comment\n'
        rendered = helper.excerpt(source, '1', width=18, focus=(opcode,))
        code, _ = parse_code(rendered)
        self.assertIn(opcode, code)
        self.assertIn('%rd123456789', code)
        self.assertIn('// keep the exact trailing comment', code)
        self.assertEqual(compact(code), compact(source))
        quoted = '.file 1 "a long name with spaces and , commas.ptx";'
        quoted_code, _ = parse_code(helper.excerpt(quoted, '1', width=15))
        self.assertIn('"a long name with spaces and , commas.ptx"', quoted_code)

    def test_operand_boundary_wrap_is_deterministic(self):
        source = 'st.shared.v4.u32 [%r1],{%r2,%r3,%r4,%r5};'
        wrapped = helper.wrap_line(source, 24)
        self.assertEqual(wrapped, helper.wrap_line(source, 24))
        self.assertEqual(compact('\n'.join(wrapped)), compact(source))
        self.assertTrue(all(len(line) <= 24 for line in wrapped))
        for register in ('%r1', '%r2', '%r3', '%r4', '%r5'):
            self.assertTrue(any(register in line for line in wrapped))

    def test_tabs_are_display_separators_but_string_whitespace_is_exact(self):
        source = 'st.shared.v4.u32\t[%r1],\t{%r2,%r3,%r4,%r5};'
        wrapped = helper.wrap_line(source, 26)
        self.assertEqual(compact('\n'.join(wrapped)), compact(source))
        self.assertNotIn('\t', '\n'.join(wrapped))
        self.assertTrue(all(len(line) <= 26 for line in wrapped))
        literal = '.file\t1\t"with\ttab and  double space.ptx";'
        self.assertIn('"with\ttab and  double space.ptx"', '\n'.join(helper.wrap_line(literal, 20)))

    def test_link_and_emphasis_overlap_without_nested_links(self):
        source = 'st.shared.u32 [%r1], %r10;'
        links = {'st.shared.u32': 'https://docs.example.org/ptx?a=1&b=2',
                 '[%r1]': 'https://docs.example.org/address',
                 '%r1': 'https://docs.example.org/register'}
        rendered = helper.excerpt(source, '1', focus=('st.shared.u32', '%r1'), links=links)
        self.assertIn('<a href="https://docs.example.org/ptx?a=1&amp;b=2"><strong>st.shared.u32</strong></a>', rendered)
        self.assertIn('<a href="https://docs.example.org/address">[<strong>%r1</strong>]</a>', rendered)
        self.assertNotIn('href="https://docs.example.org/register"', rendered)
        self.assertNotIn('<strong>%r10</strong>', rendered)
        code, parser = parse_code(rendered)
        self.assertFalse(parser.nested_link)
        self.assertEqual(code, source)

    def test_invalid_lines_and_links_fail_clearly(self):
        for selection in ('0', '2', '1-3', '2-1', '1,', 'one', ''):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                helper.excerpt('ret;\n', selection)
        for mapping in ('ret', 'ret=http://example.org', 'ret=javascript:alert(1)',
                        'ret=https://', 'ret=https://user:secret@example.org',
                        'ret=https://example.org/\nnext', 'ret=https://example.org:bad',
                        'ret=https://..', 'ret=https://-invalid.example'):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                helper.link_mapping(mapping)
        with self.assertRaises(ValueError):
            helper.excerpt('ret;', '1', width=0)

    def test_cli_only_writes_stdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ptx = root / 'input.ptx'
            ptx.write_text('  ret;\n', encoding='utf-8')
            before = {path.name: path.read_bytes() for path in root.iterdir()}
            result = subprocess.run([sys.executable, str(SCRIPT), '--ptx', str(ptx),
                                     '--lines', '1', '--focus', 'ret', '--caption', 'Return'],
                                    cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('<strong>ret</strong>;', result.stdout)
            self.assertEqual(result.stderr, '')
            self.assertEqual(before, {path.name: path.read_bytes() for path in root.iterdir()})
            bad = subprocess.run([sys.executable, str(SCRIPT), '--ptx', str(ptx), '--lines', '2'],
                                 cwd=root, capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertEqual(bad.stdout, '')
            self.assertIn('outside the supplied PTX', bad.stderr)
            self.assertEqual(before, {path.name: path.read_bytes() for path in root.iterdir()})


if __name__ == '__main__':
    unittest.main()
