"""Public assumption overrides on verify() and the CLI."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from sentryward.api import verify
from sentryward.assumptions import (
    ASSUMPTIONS,
    assumption_listing,
    resolve_assumptions,
)
from sentryward.__main__ import main
from sentryward.results import ResultArgumentParser
from sentryward.assumptions import add_assumption_arguments, assumption_arguments


NATIVE = {
    "result": "unsupported",
    "checks": {
        "synchronization": "not_run",
        "memory": "not_run",
        "equivalence": "not_requested",
    },
    "diagnostics": [],
}


def _kernel(directory):
    ptx = Path(directory) / "kernel.ptx"
    config = Path(directory) / "input.json"
    ptx.write_text(".version 8.0\n.target sm_80\n.address_size 64\n")
    config.write_text(json.dumps({
        "schema": "spverify.kernel-input/v1",
        "launch": {"grid_dim": [1, 1, 1], "block_dim": [32, 1, 1]},
    }))
    return ptx, config


def _cli(argv, result=NATIVE):
    stdout = io.StringIO()
    with patch("sentryward.__main__.verify", return_value=dict(result)) as mocked:
        with redirect_stdout(stdout):
            code = main(argv)
    return code, stdout.getvalue(), mocked


class ListingTests(unittest.TestCase):
    def test_listing_covers_the_pool(self):
        listing = assumption_listing()
        self.assertEqual(set(listing["defaults"]), set(ASSUMPTIONS))
        self.assertEqual([item["key"] for item in listing["assumptions"]], list(ASSUMPTIONS))
        for item in listing["assumptions"]:
            self.assertEqual(item["enabled"], listing["defaults"][item["key"]])
            self.assertIn("label", item)
            self.assertIn("text", item)

    def test_overrides_win_over_input_file_flags(self):
        resolved = resolve_assumptions(
            {"order_independent_atomics": False},
            {"order_independent_atomics": True},
        )
        self.assertTrue(resolved["order_independent_atomics"])
        self.assertFalse(resolved["representative_branches"])


class AssumptionArgumentsTests(unittest.TestCase):
    def test_assume_and_no_assume(self):
        parser = ResultArgumentParser(allow_abbrev=False)
        add_assumption_arguments(parser)
        args = parser.parse_args([
            "--assume", "order_independent_atomics",
            "--no-assume", "duplicate_tma_store_benign",
        ])
        self.assertEqual(
            assumption_arguments(parser, args),
            {
                "order_independent_atomics": True,
                "duplicate_tma_store_benign": False,
            },
        )

    def test_conflicting_flags(self):
        parser = ResultArgumentParser(allow_abbrev=False)
        add_assumption_arguments(parser)
        args = parser.parse_args([
            "--assume", "representative_election",
            "--no-assume", "representative_election",
        ])
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as caught:
                assumption_arguments(parser, args)
        self.assertEqual(caught.exception.code, 2)
        self.assertEqual(json.loads(stdout.getvalue())["result"], "invalid_input")


class VerifyAssumptionsTests(unittest.TestCase):
    def test_passes_overrides_to_native(self):
        with tempfile.TemporaryDirectory() as scratch:
            ptx, config = _kernel(scratch)
            with patch("sentryward.api._run_ptx", return_value=dict(NATIVE)) as run:
                verify(
                    ptx, input=config, timeout_seconds=5,
                    assumptions={"order_independent_atomics": True},
                )
            self.assertEqual(
                run.call_args.kwargs["semantic_assumptions"],
                {"order_independent_atomics": True},
            )

    def test_omitted_assumptions_are_not_sent(self):
        with tempfile.TemporaryDirectory() as scratch:
            ptx, config = _kernel(scratch)
            with patch("sentryward.api._run_ptx", return_value=dict(NATIVE)) as run:
                verify(ptx, input=config, timeout_seconds=5)
            self.assertIsNone(run.call_args.kwargs["semantic_assumptions"])

    def test_empty_mapping_is_omitted(self):
        with tempfile.TemporaryDirectory() as scratch:
            ptx, config = _kernel(scratch)
            with patch("sentryward.api._run_ptx", return_value=dict(NATIVE)) as run:
                verify(ptx, input=config, timeout_seconds=5, assumptions={})
            self.assertIsNone(run.call_args.kwargs["semantic_assumptions"])

    def test_unknown_key_is_invalid_input(self):
        with tempfile.TemporaryDirectory() as scratch:
            ptx, config = _kernel(scratch)
            result = verify(ptx, input=config, timeout_seconds=5, assumptions={"nope": True})
        self.assertEqual(result["result"], "invalid_input")
        self.assertEqual(result["diagnostics"][0]["code"], "invalid_configuration")

    def test_non_boolean_flag_is_invalid_input(self):
        with tempfile.TemporaryDirectory() as scratch:
            ptx, config = _kernel(scratch)
            result = verify(
                ptx, input=config, timeout_seconds=5,
                assumptions={"order_independent_atomics": "true"},
            )
        self.assertEqual(result["result"], "invalid_input")


class CliTests(unittest.TestCase):
    def test_assume_reaches_verify(self):
        code, stdout, mocked = _cli([
            "kernel.ptx", "--input", "input.json",
            "--assume", "order_independent_atomics",
            "--no-assume", "representative_branches",
        ])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout), NATIVE)
        self.assertEqual(
            mocked.call_args.kwargs["assumptions"],
            {
                "order_independent_atomics": True,
                "representative_branches": False,
            },
        )

    def test_default_invocation_sends_no_overrides(self):
        _, _, mocked = _cli(["kernel.ptx", "--input", "input.json"])
        self.assertIsNone(mocked.call_args.kwargs["assumptions"])

    def test_list_assumptions(self):
        code, stdout, mocked = _cli(["--list-assumptions"])
        self.assertEqual(code, 0)
        self.assertEqual(mocked.call_count, 0)
        payload = json.loads(stdout)
        self.assertEqual(payload, assumption_listing())

    def test_list_assumptions_rejects_a_kernel_path(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as caught:
                main(["kernel.ptx", "--list-assumptions"])
        self.assertEqual(caught.exception.code, 2)
        self.assertEqual(json.loads(stdout.getvalue())["result"], "invalid_input")

    def test_unknown_assume_is_invalid_input(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as caught:
                main(["kernel.ptx", "--input", "input.json", "--assume", "nope"])
        self.assertEqual(caught.exception.code, 2)
        self.assertEqual(json.loads(stdout.getvalue())["result"], "invalid_input")

    def test_missing_ptx_without_list(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as caught:
                main(["--input", "input.json"])
        self.assertEqual(caught.exception.code, 2)
        message = json.loads(stdout.getvalue())["diagnostics"][0]["message"]
        self.assertIn("ptx", message)


if __name__ == "__main__":
    unittest.main()
