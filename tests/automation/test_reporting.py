"""Exercise the real composer and checker on frozen, portable case bundles."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from tools.automation.reporting import generate_report, report_arguments, run_report_script


ROOT = Path(__file__).resolve().parents[2]


class ReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name).resolve()
        self.case = self.folder / "case"
        self.case.mkdir()
        (self.case / "kernel.ptx").write_text(
            ".version 9.2\n.target sm_100a\n.address_size 64\n.visible .entry test() {\n ret;\n}\n")
        (self.case / "input.json").write_text(json.dumps({"entry": "test", "launch": {
            "grid_dim": [1, 1, 1], "block_dim": [32, 1, 1]}}))
        (self.case / "reference.py").write_text("def reference(x):\n    return x\n")
        self.result = {"result": "unsupported", "entry": "test", "checks": {
            "synchronization": "incomplete", "memory": "not_run", "equivalence": "not_run"},
            "diagnostics": [{"code": "unsupported_instruction", "stage": "frontend"}],
            "warnings": [], "warning_count": 0, "warnings_complete": True,
            "future_verifier_field": {"must_preserve": True}}
        self.identity = self.case / "identity.json"
        self.save_result()

    def save_result(self):
        (self.case / "output.json").write_text(json.dumps(self.result))
        prefix = "https://evidence.example.test/sentryward"
        files = {}
        for name in ("input.json", "output.json", "kernel.ptx", "reference.py"):
            sha = hashlib.sha256((self.case / name).read_bytes()).hexdigest()
            files[name] = {"sha256": sha, "uri": f"{prefix}/objects/{sha}/{name}"}
        self.identity.write_text(json.dumps({"evidence_base_url": prefix, "files": files}))

    def check(self, case=None):
        case = case or self.case
        run_report_script(ROOT, "check_report.py", report_arguments(case, case / "identity.json"))

    def test_verified_and_incomplete_results_have_checked_base_reports(self):
        for clean in (False, True):
            with self.subTest(clean=clean):
                if clean:
                    self.result.update(result="verified", diagnostics=[], checks={
                        "synchronization": "passed", "memory": "passed", "equivalence": "passed"})
                    self.save_result()
                original = {path: path.read_bytes() for path in self.case.iterdir() if path.is_file()}
                report = generate_report(ROOT, self.case, self.identity)
                self.assertEqual(report, self.case / "report/report.md")
                self.assertIn(self.result["result"], report.read_text())
                self.assertIn("reference.py", report.read_text())
                self.assertNotIn("AI explanation status", report.read_text())
                self.check()
                for path, saved in original.items():
                    self.assertEqual(path.read_bytes(), saved)
                self.assertEqual(json.loads((self.case / "output.json").read_text())["future_verifier_field"],
                                 {"must_preserve": True})

    def test_checker_rejects_modified_facts_or_artifact_links(self):
        report = generate_report(ROOT, self.case, self.identity)
        original = report.read_text()
        variants = (original + "\nThis kernel is unconditionally verified.\n",
                    original.replace("https://evidence.example.test/", "https://wrong.example.test/", 1))
        for variant in variants:
            with self.subTest(modification=variant[-80:]):
                report.write_text(variant)
                with self.assertRaisesRegex(ValueError, "Report differs"):
                    self.check()

    def test_changed_input_is_rejected_without_replacing_existing_report(self):
        report = generate_report(ROOT, self.case, self.identity)
        original = report.read_bytes()
        (self.case / "input.json").write_text('{"entry":"different"}')
        with self.assertRaises(ValueError):
            generate_report(ROOT, self.case, self.identity)
        self.assertEqual(report.read_bytes(), original)
        with self.assertRaises(ValueError):
            self.check()

    def test_frozen_bundle_can_be_checked_and_recomposed_after_transfer(self):
        report = generate_report(ROOT, self.case, self.identity)
        original = report.read_bytes()
        transferred = self.folder / "transferred"
        shutil.copytree(self.case, transferred)
        self.check(transferred)
        regenerated = generate_report(ROOT, transferred, transferred / "identity.json")
        self.assertEqual(regenerated.read_bytes(), original)
        self.assertNotIn(str(self.case), original.decode())

    def test_manual_skill_drafts_remain_supported(self):
        draft = self.case / "draft.json"
        draft.write_text(json.dumps({"version": 2, "title": "Saved verification finding", "sections": [{
            "title": "Incomplete verification", "answer": "The saved result does not establish correctness.",
            "evidence": [{"kind": "result", "pointer": "/result", "equals": "unsupported"}],
        }]}))
        arguments = report_arguments(self.case, self.identity) + ["--draft", str(draft)]
        for script in ("report_facts.py", "check_report.py"):
            run_report_script(ROOT, script, arguments)
        report = self.case / "report/report.md"
        self.assertIn("The saved result does not establish correctness.", report.read_text())
        # The automated base path never picks up a draft merely because it exists.
        generate_report(ROOT, self.case, self.identity)
        self.assertNotIn("The saved result does not establish correctness.", report.read_text())


if __name__ == "__main__":
    unittest.main()
