"""A kernel run without a reference (docs/output-json.md, Categories and Precedence).

When no reference is supplied and the synchronization check passes, the result stays race_free:
the native report's equivalence section is skipped (reason reference_expression_not_supplied),
checks.equivalence is not_requested, and the printed verdict says that equivalence checking was
skipped. These tests need no native executable.
"""
import json
import tempfile
import unittest
from pathlib import Path

from sentryward.results import category, normalize, publish, verdict

SKIPPED = {"status": "skipped", "reason": "reference_expression_not_supplied",
           "message": "Equivalence checking was skipped: no reference expression was supplied."}


def native_report(status="race_free"):
    return {"status": status, "verification_status": status, "mode": "verify",
            "memory_analysis_complete": True, "equivalence": dict(SKIPPED)}


class EquivalenceSkippedTest(unittest.TestCase):
    def test_skipped_keeps_the_synchronization_verdict(self):
        out = normalize(native_report(), operation="kernel_verification")
        self.assertEqual(out["result"], "race_free")
        self.assertEqual(out["checks"]["synchronization"], "passed")
        self.assertEqual(out["checks"]["equivalence"], "not_requested")
        diagnostic = next(d for d in out["diagnostics"] if d["code"] == "reference_expression_not_supplied")
        self.assertIn("skipped", diagnostic["message"])

    def test_only_the_equivalence_section_is_skipped(self):
        self.assertEqual(category("skipped"), "tool_failure")
        self.assertEqual(normalize({"status": "skipped"}, operation="kernel_verification")["result"], "tool_failure")
        self.assertEqual(normalize(dict(SKIPPED), operation="equivalence")["result"], "race_free")

    def test_required_check_without_reference_is_missing_input(self):
        report = native_report()
        report["equivalence"] = {"status": "missing_input", "reason": "reference_expression_not_supplied"}
        out = normalize(report, operation="kernel_verification")
        self.assertEqual(out["result"], "missing_input")
        self.assertEqual(out["checks"]["equivalence"], "not_run")

    def test_runner_reference_rule(self):
        from tools.rerun_examples import missing_reference_files
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, config, mapping = root / "reference.py", root / "input.json", root / "reference_binding.json"
            config.write_text("{}")
            self.assertIsNone(missing_reference_files(source, config, mapping), "no reference at all: skipped")
            source.write_text("")
            self.assertEqual(missing_reference_files(source, config, mapping), [mapping], "a partial reference is missing input")
            mapping.write_text("{}")
            self.assertEqual(missing_reference_files(source, config, mapping), [], "a complete reference")
            source.unlink()
            self.assertEqual(missing_reference_files(source, config, mapping), [source], "a binding without its function")

    def test_published_report_says_skipped(self):
        out = publish(normalize(native_report(), operation="kernel_verification"))
        self.assertEqual(out["result"], "race_free")
        self.assertEqual(out["checks"], {"synchronization": "passed", "equivalence": "not_requested"})
        self.assertEqual(verdict(out["result"], out["checks"]), "race_free (equivalence checking skipped)")

    def test_other_verdicts_print_their_name(self):
        self.assertEqual(verdict("verified", {"equivalence": "passed"}), "verified")
        self.assertEqual(verdict("race_free", {"equivalence": "not_run"}), "race_free")
        self.assertEqual(verdict("potential_race", {"equivalence": "not_requested"}), "potential_race")

    def test_synchronization_failure_keeps_precedence(self):
        report = native_report("potential_race")
        out = normalize(report, operation="kernel_verification")
        self.assertEqual(out["result"], "potential_race")

    def test_index_prints_the_skipped_verdict(self):
        from tools.rerun_examples import write_index
        with tempfile.TemporaryDirectory() as directory:
            examples = Path(directory)
            record = {"name": "k", "result": "race_free", "rerun": True,
                      "checks": {"synchronization": "passed", "equivalence": "not_requested"},
                      "equivalence": {"result": "race_free", "check": "not_requested", "diagnostics": []}}
            write_index(examples, [record])
            text = (examples / "index.md").read_text()
            self.assertIn("`race_free` (equivalence checking skipped)", text)
            self.assertIn("skipped (no reference)", text)


if __name__ == "__main__":
    unittest.main()
