"""Recompose the public review samples from identified inputs and plans.

Recomposition is a provenance check, not semantic acceptance.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'tests/report_skill/fixtures/explanations/manifest.json'
SCRIPTS = ROOT / 'skills/generate-kernel-report/scripts'


class PublishedExplanationTests(unittest.TestCase):
    def test_review_samples_recompose_from_identified_inputs(self):
        for name, case in json.loads(MANIFEST.read_text()).items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                if case.get('historical_snapshot'):
                    # These trials used historical inputs and a frozen renderer.
                    # Current examples may have a newer result schema. Preserve
                    # the archived output rather than relabeling it as a rerun.
                    self.assertEqual(hashlib.sha256((ROOT / case['report']).read_bytes()).hexdigest(),
                                     case['report_sha256'])
                    continue
                # Preserve the report-to-input layout: links are report-relative.
                report = Path(directory) / case['report']
                args = []
                for key, relative in case['files'].items():
                    path = ROOT / relative
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                                     case['sha256'][key], (name, key))
                    display_path = case.get('render_paths', case['files'])[key]
                    copied = Path(directory) / display_path
                    copied.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, copied)
                    args += ['--' + key, display_path]
                for key, value in case.get('render_options', {}).items():
                    args += ['--' + key.replace('_', '-'), value]
                args += ['--report', str(report)]
                for script in ('report_facts.py', 'check_report.py'):
                    result = subprocess.run([sys.executable, str(SCRIPTS / script), *args],
                                            cwd=directory, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(report.read_bytes(), (ROOT / case['report']).read_bytes())
                self.assertEqual(hashlib.sha256(report.read_bytes()).hexdigest(),
                                 case['report_sha256'])
                self.assertEqual(list(report.parent.iterdir()), [report])


if __name__ == '__main__':
    unittest.main()
