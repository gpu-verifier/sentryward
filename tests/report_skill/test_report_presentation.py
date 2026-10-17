import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'skills/generate-kernel-report/scripts'))
import report_layout as layout
from report_layout import technical_prose
from report_facts import fold


class PresentationTests(unittest.TestCase):
    def test_technical_tokens_and_ordinary_quantities(self):
        rendered = technical_prose('Compare %r116 and %r3 using setp.ne.s32; A002 changes source_rows classification. tid.x spans 0–31, beta.stride is 128.')
        for token in ('%r116', '%r3', 'setp.ne.s32', 'A002', 'source_rows', 'tid.x', 'beta.stride'):
            self.assertIn('<code>' + token + '</code>', rendered)
        self.assertIn('spans 0–31', rendered)
        self.assertIn('is 128.', rendered)
        self.assertEqual(rendered, technical_prose(rendered))

    def test_code_math_links_and_html_are_preserved(self):
        protected = ['```ptx\nsetp.ne.s32 %p1, %r1, %r2;\n```\n',
                     '> ```math\n> %r1 source_rows\n> ```\n',
                     '<pre>\n%r1 st.local.u32\n</pre>\n',
                     '<code>%r1</code> `%r2` ``%r3``\n',
                     '[source_rows](https://example.com/source_rows) https://example.com/st.local.u32\n',
                     '<a href="source_rows">cp.async.bulk.commit_group</a>\n']
        for part in protected:
            with self.subTest(part=part):
                self.assertEqual(part, technical_prose(part))

    def test_child_indentation_preserves_nested_structure(self):
        inner = fold('Raw', '| key | value |\n| --- | --- |\n| site | 3 |')
        middle = fold('Records', 'Record answer.\n\n' + inner)
        outer = fold('Evidence', 'Evidence answer.\n\n' + middle)
        self.assertIn('Evidence answer.\n\n> <details>', outer)
        self.assertIn('> Record answer.', outer)
        self.assertIn('> > <summary>Raw</summary>', outer)
        self.assertIn('> > | site | 3 |', outer)
        self.assertEqual(outer.count('<details>'), outer.count('</details>'))

class SequenceDiagramTests(unittest.TestCase):
    def test_sequence_grammar_is_not_polluted_by_flowchart_classes(self):
        source = 'sequenceDiagram\nparticipant P as Producer\nparticipant C as Consumer\nP->>C: Release'
        rendered = layout.diagram_view({'kind':'mermaid','source':source,'meaning':'A possible interleaving, not an observed trace.'})
        self.assertIn(source, rendered)
        self.assertNotIn('classDef', rendered)
