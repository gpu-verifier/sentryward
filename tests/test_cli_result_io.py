"""CLI result file, compact summary, and --fail-on exit status."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from sentryward import summarize
from sentryward.__main__ import main
from sentryward.results import (
    FINDINGS,
    apply_fail_on,
    encode_result,
    expand_fail_on,
    write_result,
)


RACE = {
    "result": "potential_race",
    "checks": {
        "synchronization": "passed",
        "memory": "failed",
        "equivalence": "not_requested",
    },
    "diagnostics": [
        {
            "code": "conflicting_accesses_unordered",
            "stage": "memory_check",
            "message": "two stores overlap",
        }
    ],
    "warnings": [],
    "warning_count": 0,
    "warnings_complete": True,
    "elapsed_seconds": 0.25,
}

VERIFIED = {
    "result": "verified",
    "checks": {
        "synchronization": "passed",
        "memory": "passed",
        "equivalence": "passed",
    },
    "diagnostics": [],
    "warnings": [],
    "warning_count": 0,
    "warnings_complete": True,
}


def _run(argv, result=RACE):
    stdout = io.StringIO()
    with patch("sentryward.__main__.verify", return_value=dict(result)):
        with redirect_stdout(stdout):
            code = main(argv)
    return code, stdout.getvalue()


class ExpandFailOnTests(unittest.TestCase):
    def test_findings_alias(self):
        self.assertEqual(expand_fail_on(["findings"]), set(FINDINGS))

    def test_any_excludes_clean_passes(self):
        selected = expand_fail_on(["any"])
        self.assertNotIn("verified", selected)
        self.assertNotIn("race_free", selected)
        self.assertIn("potential_race", selected)
        self.assertIn("timeout", selected)

    def test_repeatable_union(self):
        self.assertEqual(
            expand_fail_on(["potential_race", "timeout"]),
            {"potential_race", "timeout"},
        )

    def test_unknown_value(self):
        with self.assertRaisesRegex(ValueError, "not_a_category"):
            expand_fail_on(["not_a_category"])


class ApplyFailOnTests(unittest.TestCase):
    def test_default_zero_stays_zero(self):
        self.assertEqual(apply_fail_on(RACE, [], 0), 0)

    def test_finding_becomes_one(self):
        self.assertEqual(apply_fail_on(RACE, ["findings"], 0), 1)

    def test_clean_pass_stays_zero(self):
        self.assertEqual(apply_fail_on(VERIFIED, ["any"], 0), 0)

    def test_existing_nonzero_is_unchanged(self):
        timeout = {"result": "timeout"}
        self.assertEqual(apply_fail_on(timeout, ["timeout"], 2), 2)


class SummarizeTests(unittest.TestCase):
    def test_lists_result_checks_and_diagnostics(self):
        text = summarize(RACE)
        self.assertTrue(text.endswith("\n"))
        self.assertIn("result: potential_race\n", text)
        self.assertIn("  synchronization: passed\n", text)
        self.assertIn("  memory: failed\n", text)
        self.assertIn("diagnostics: 1\n", text)
        self.assertIn("conflicting_accesses_unordered [memory_check]: two stores overlap", text)
        self.assertIn("warnings: 0 (complete)\n", text)
        self.assertIn("elapsed_seconds: 0.25\n", text)

    def test_empty_diagnostics(self):
        text = summarize(VERIFIED)
        self.assertIn("diagnostics: none\n", text)

    def test_does_not_mutate(self):
        original = json.loads(json.dumps(RACE))
        summarize(RACE)
        self.assertEqual(RACE, original)

    def test_rejects_non_object(self):
        with self.assertRaises(TypeError):
            summarize(["potential_race"])

    def test_clips_long_messages(self):
        report = {
            "result": "unsupported",
            "checks": {},
            "diagnostics": [{"code": "long", "message": "x" * 400}],
        }
        text = summarize(report)
        self.assertIn("...", text)
        self.assertLess(len(text), 400)

    def test_truncates_long_diagnostic_lists(self):
        report = {
            "result": "unsupported",
            "diagnostics": [{"code": "d%d" % i} for i in range(25)],
        }
        text = summarize(report)
        self.assertIn("diagnostics: 25\n", text)
        self.assertIn("  - ... 5 more\n", text)

    def test_incomplete_warnings(self):
        text = summarize({"result": "timeout", "warnings": [], "warnings_complete": False})
        self.assertIn("warnings: 0 (incomplete)\n", text)


class WriteResultTests(unittest.TestCase):
    def test_writes_json_object(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "nested" / "output.json"
            write_result(RACE, path)
            self.assertEqual(json.loads(path.read_text()), RACE)

    def test_rejects_directory(self):
        with self.assertRaisesRegex(ValueError, "directory"):
            write_result(RACE, ".")


class CliTests(unittest.TestCase):
    def test_default_json_and_zero_exit_for_finding(self):
        code, stdout = _run(["kernel.ptx", "--input", "input.json"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout), RACE)

    def test_output_writes_file_and_keeps_stdout_json(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "output.json"
            code, stdout = _run(["kernel.ptx", "--input", "input.json", "--output", str(path)])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stdout), RACE)
            self.assertEqual(json.loads(path.read_text()), RACE)

    def test_summary_format_with_json_file(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        path = Path(scratch.name) / "summary.json"
        code, stdout = _run([
            "kernel.ptx", "--input", "input.json",
            "--format", "summary", "--output", str(path),
        ])
        self.assertEqual(code, 0)
        self.assertTrue(stdout.startswith("result: potential_race\n"))
        self.assertEqual(json.loads(path.read_text()), RACE)
        with self.assertRaises(json.JSONDecodeError):
            json.loads(stdout)

    def test_fail_on_findings(self):
        code, stdout = _run(["kernel.ptx", "--input", "input.json", "--fail-on", "findings"])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(stdout), RACE)

    def test_fail_on_does_not_fail_clean_result(self):
        code, stdout = _run(
            ["kernel.ptx", "--input", "input.json", "--fail-on", "any"],
            result=VERIFIED,
        )
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout)["result"], "verified")

    def test_unknown_fail_on_is_invalid_input(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as caught:
                main(["kernel.ptx", "--input", "input.json", "--fail-on", "nope"])
        self.assertEqual(caught.exception.code, 2)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["result"], "invalid_input")
        self.assertEqual(payload["diagnostics"][0]["code"], "invalid_arguments")

    def test_output_directory_is_invalid_input(self):
        stdout = io.StringIO()
        with patch("sentryward.__main__.verify", return_value=dict(RACE)):
            with redirect_stdout(stdout):
                code = main(["kernel.ptx", "--input", "input.json", "--output", "."])
        self.assertEqual(code, 2)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["result"], "invalid_input")
        self.assertEqual(payload["diagnostics"][0]["code"], "invalid_output_path")

    def test_encode_matches_cli_json(self):
        _, stdout = _run(["kernel.ptx", "--input", "input.json"])
        self.assertEqual(stdout, encode_result(RACE) + "\n")


if __name__ == "__main__":
    unittest.main()
