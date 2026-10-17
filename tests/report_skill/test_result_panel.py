import importlib
import json
from collections import Counter
from pathlib import Path
import re
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[2] / 'skills/generate-kernel-report/scripts'
sys.path.insert(0, str(SCRIPTS))
layout = importlib.import_module('report_layout')
checker = importlib.import_module('check_report')


class ResultPanelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.run_input = {'entry': 'test', 'launch': {'block_dim': [32, 1, 1]}}
        self.ptx = '.version 9.2\n.entry test(){\nret;\n}\n'
        self.result = {'result': 'verified', 'checks': {'synchronization': 'passed', 'memory': 'passed', 'equivalence': 'not_requested'},
                       'diagnostics': [], 'scope': {'operation': 'kernel_verification'}}
        self.artifacts = [self.root / name for name in ('input.json', 'output.json', 'kernel.ptx')]

    def tearDown(self):
        self.tmp.cleanup()

    def compose(self, panel=True, draft=None):
        for path, value in zip(self.artifacts, [json.dumps(self.run_input), json.dumps(self.result), self.ptx]):
            path.write_text(value)
        return layout.compose(draft, self.run_input, self.result, self.ptx, self.artifacts,
                              report_path=self.root / 'report.md', result_panel=panel)

    def test_panel_preserves_evidence_and_all_link_destinations(self):
        flat, panel = self.compose(False), self.compose()
        self.assertEqual(flat.split('<!-- verifier-evidence:start -->')[1], panel.split('<!-- verifier-evidence:start -->')[1])
        for pattern in (r'href="([^"]+)"', r'\]\(([^)]+)\)'):
            self.assertEqual(Counter(re.findall(pattern, flat)), Counter(re.findall(pattern, panel)))
        self.assertIn('VERIFIER RESULT · Recorded outcome', panel)
        self.assertNotIn('## Code explanation', panel)
        self.assertLess(panel.index('VERIFIER RESULT'), panel.index('Analysis records'))

    def test_readable_checks_preserve_raw_values(self):
        panel = self.compose()
        self.assertIn('**Equivalence:** Not requested', panel)
        self.assertIn('Equivalence <code>not_requested</code>', panel)

    def test_unknown_and_missing_check_states_are_not_success(self):
        self.result['checks'] = {'synchronization': None, 'memory': 'new_status'}
        panel = self.compose()
        self.assertIn('**Synchronization:** <code>null</code>', panel)
        self.assertIn('**Memory:** <code>new_status</code>', panel)
        self.assertIn('**Equivalence:** <code>not saved</code>', panel)
        self.assertNotIn(':** Passed', panel)

    def test_additional_check_is_not_discarded(self):
        self.result['checks']['new_check'] = 'failed'
        panel = self.compose()
        self.assertIn('new_check <code>failed</code>', panel)

    def test_stop_diagnostic_remains_visible(self):
        self.result['result'] = 'unsupported'
        self.result['diagnostics'] = [{'code': 'unsupported_predicate', 'message': 'Cannot translate the predicate.'}]
        panel = self.compose()
        self.assertLess(panel.index('Cannot translate the predicate.'), panel.index('<summary><strong>Saved status'))

    def test_explanation_body_stays_intact(self):
        draft = {'version': 2, 'title': 'Test', 'sections': [{'title': 'Return', 'answer': 'The entry returns.',
                 'evidence': [{'kind': 'ptx', 'lines': '3', 'contains': ['ret;']}]}]}
        flat = self.compose(False, draft).split('<!-- code-explanation:start -->')[1].split('<!-- code-explanation:end -->')[0]
        panel = self.compose(True, draft).split('<!-- code-explanation:start -->')[1].split('<!-- code-explanation:end -->')[0]
        expected = re.sub(r'^### 2\.(\d+) ', r'### \1. ', flat.replace('## 2. Code explanation', '## Code explanation'), flags=re.M)
        expected = expected.replace('\n\n---\n\n', '\n\n', 1)
        self.assertEqual(expected, panel)

    def test_checker_accepts_panel_and_detects_tampering(self):
        report = self.root / 'report.md'
        text = self.compose()
        report.write_text(text)
        args = [report, self.artifacts[2], self.artifacts[0], self.artifacts[1]]
        self.assertEqual(checker.check_current(*args, result_panel=True), [])
        report.write_text(text.replace(':** Passed', ':** Failed', 1))
        self.assertTrue(checker.check_current(*args, result_panel=True))

    def test_long_table_opcodes_wrap_without_changing_source(self):
        from result_panel import render
        opcode = 'tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64'
        flat = self.compose(False)
        row = '| Phase effect | <code>' + opcode + '</code> | <code>1239648</code> |\n\n'
        excerpt = '<pre><code>' + opcode + ' [%r7];</code></pre>\n\n'
        flat = flat.replace('### 1.2 Analysis scope', row + excerpt + '### 1.2 Analysis scope')
        panel = render(flat)
        self.assertIn('tcgen05.commit.cta_group::1.</code><br/><code>', panel)
        self.assertIn(row, panel.replace('</code><br/><code>', ''))
        self.assertIn(excerpt, panel)


if __name__ == '__main__':
    unittest.main()
