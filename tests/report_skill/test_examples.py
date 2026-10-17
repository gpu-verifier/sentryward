"""Check the three frozen result families in the current base renderer; no model or verifier run."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / 'fixtures'
SCRIPTS = ROOT / 'skills/generate-kernel-report/scripts'


class PublishedExamplesTests(unittest.TestCase):
    def test_frozen_inputs_render_with_preserved_diagnostics(self):
        cases = json.loads((FIXTURES / 'manifest.json').read_text())
        for name, case in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                scratch = Path(directory)
                fixture = FIXTURES / name
                for filename in ('input.json', 'output.json', 'identity.json'):
                    shutil.copyfile(fixture / filename, scratch / filename)
                shutil.copyfile(ROOT / case['ptx'], scratch / 'kernel.ptx')
                identities = json.loads((fixture / 'identity.json').read_text())['files']
                for filename, expected in identities.items():
                    data = (scratch / filename).read_bytes()
                    self.assertEqual(len(data), expected['bytes'], filename)
                    self.assertEqual(hashlib.sha256(data).hexdigest(), expected['sha256'], filename)

                report = scratch / 'output/report.md'
                args = [
                    '--input', str(scratch / 'input.json'),
                    '--result', str(scratch / 'output.json'),
                    '--ptx', str(scratch / 'kernel.ptx'),
                    '--report', str(report),
                ]
                for script, extra in (
                    ('report_facts.py', []),
                    ('check_report.py', ['--identity', str(scratch / 'identity.json')]),
                ):
                    result = subprocess.run(
                        [sys.executable, str(SCRIPTS / script), *args, *extra],
                        capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                rendered = report.read_text()
                saved = json.loads((fixture / 'output.json').read_text())
                self.assertIn(saved['status'], rendered)
                for field in ('races', 'warnings'):
                    for record in saved.get(field, []):
                        for side in ('first', 'second'):
                            if side in record:
                                self.assertIn(str(record[side]['event']), rendered)
                self.assertNotIn('## 2. Code explanation', rendered)
                self.assertEqual(list(report.parent.iterdir()), [report])


if __name__ == '__main__':
    unittest.main()
