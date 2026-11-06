"""Per-kernel container boundaries; no Docker daemon or network is required."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tools.automation import ci_shard
from tools.automation.common import read_json, write_json


class CIShardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.plan = self.root / "plan.json"
        self.output = self.root / "work"
        self.export = self.root / "artifacts"
        self.data = {
            "run_id": "gh-1234", "evidence_base_url": "https://evidence.example.test/sentryward",
            "config": {"verification": {"shards": 4, "timeout_seconds": 500}},
            "tasks": [
                {"name": "alpha", "shard": 0, "action": "verify"},
                {"name": "beta", "shard": 0, "action": "verify"},
                {"name": "other-shard", "shard": 1, "action": "verify"},
            ],
        }
        self.calls = []
        self.events = []
        self.failures = {}
        self.cleanup_failures = {}
        self.container_cases = {}
        self.oom_killed = set()
        self.worker_outputs = {}

    def process(self, command, **options):
        self.calls.append((list(command), dict(options)))
        if command[:2] == ["docker", "run"]:
            name = command[-1]
            self.container_cases[command[command.index("--name") + 1]] = name
            self.events.append(("run", name))
            outcome = self.failures.get(name, 0)
            if isinstance(outcome, Exception):
                raise outcome
            if outcome == 0:
                write_json(self.output / "cases" / name / "record.json", {"name": name})
                if name in self.worker_outputs:
                    self.worker_outputs[name](self.output / "cases" / name)
            return subprocess.CompletedProcess(command, outcome)
        if command[:2] == ["docker", "inspect"]:
            name = self.container_cases[command[-1]]
            self.events.append(("inspect", name))
            return subprocess.CompletedProcess(command, 0, json.dumps({"OOMKilled": name in self.oom_killed}))
        self.assertEqual(command[:3], ["docker", "rm", "-f"])
        container = command[-1]
        self.events.append(("cleanup", container))
        if self.container_cases.get(container) in self.cleanup_failures:
            raise self.cleanup_failures[self.container_cases[container]]
        return subprocess.CompletedProcess(command, 0)

    def run_shard(self, shard=0, **options):
        write_json(self.plan, self.data)
        with patch.object(ci_shard.subprocess, "run", side_effect=self.process):
            return ci_shard.run_shard(self.plan, self.output, shard, export=self.export, **options)

    def test_each_case_has_no_network_credentials_or_shared_writable_mount(self):
        secrets = {"AWS_ACCESS_KEY_ID": "sentinel-storage", "AWS_SECRET_ACCESS_KEY": "sentinel-secret",
                   "ANTHROPIC_API_KEY": "sentinel-model", "GITHUB_TOKEN": "sentinel-github"}
        with patch.dict(os.environ, secrets), patch.object(ci_shard.os, "getuid", return_value=1001), \
                patch.object(ci_shard.os, "getgid", return_value=1002):
            result = self.run_shard(image="pinned-worker-image")
        self.assertFalse(result)
        runs = [call for call in self.calls if call[0][:2] == ["docker", "run"]]
        self.assertEqual([command[-1] for command, _ in runs], ["alpha", "beta"])
        self.assertEqual([event[0] for event in self.events], ["run", "cleanup", "run", "cleanup"])
        for command, options in runs:
            name = command[-1]
            for flag in ("--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                         "--memory=10g", "--memory-swap=10g", "--pids-limit=256"):
                self.assertIn(flag, command)
            self.assertEqual(command[command.index("--user") + 1], "1001:1002")
            envs = [command[i + 1] for i, item in enumerate(command) if item in {"--env", "-e"}]
            self.assertEqual(envs, ["PYTHONDONTWRITEBYTECODE=1", "HOME=/tmp", "USER=sentryward"])
            self.assertNotIn("--env-file", command)
            self.assertTrue(all(secret not in " ".join(command) for secret in secrets.values()))
            mounts = [command[i + 1] for i, item in enumerate(command) if item == "--mount"]
            self.assertEqual(mounts, [f"type=bind,src={self.plan},dst=/plan.json,readonly",
                                      f"type=bind,src={self.output / 'cases' / name},dst=/out/cases/{name}"])
            self.assertNotIn("-v", command)
            self.assertNotIn("--rm", command)
            self.assertIn("pinned-worker-image", command)
            self.assertEqual(command[-4:], ["--shard", "0", "--name", name])
            self.assertEqual(options["timeout"], 590)
        self.assertEqual(read_json(self.export / "shard-0-errors.json"), [])
        self.assertFalse((self.output / "cases/other-shard").exists())
        self.assertEqual(read_json(self.export / "cases/alpha/record.json"), {"name": "alpha"})

    def test_timeout_and_nonzero_exit_do_not_drop_other_case_records(self):
        self.data["tasks"].insert(1, {"name": "exit-failure", "shard": 0, "action": "verify"})
        self.failures = {"alpha": subprocess.TimeoutExpired(["docker", "run"], 590), "exit-failure": 137}
        self.assertTrue(self.run_shard())
        self.assertEqual([name for event, name in self.events if event == "run"], ["alpha", "exit-failure", "beta"])
        self.assertEqual(len([event for event in self.events if event[0] == "cleanup"]), 3)
        errors = read_json(self.export / "shard-0-errors.json")
        self.assertEqual([error["name"] for error in errors], ["alpha", "exit-failure"])
        self.assertIn("137", errors[1]["error"])
        self.assertNotIn("OOMKilled", errors[1]["error"])
        self.assertEqual(read_json(self.output / "cases/beta/record.json"), {"name": "beta"})

    def test_container_launch_error_still_cleans_up_and_continues(self):
        self.failures["alpha"] = OSError("Docker launch failed")
        self.assertTrue(self.run_shard())
        self.assertEqual([event[0] for event in self.events], ["run", "cleanup", "run", "cleanup"])
        self.assertTrue((self.output / "cases/beta/record.json").is_file())
        self.assertEqual(read_json(self.export / "shard-0-errors.json")[0]["name"], "alpha")

    def test_cleanup_error_is_bounded_recorded_and_does_not_stop_next_case(self):
        self.cleanup_failures["alpha"] = subprocess.TimeoutExpired(["docker", "rm"], 30)
        self.assertTrue(self.run_shard())
        self.assertTrue((self.output / "cases/beta/record.json").is_file())
        self.assertFalse((self.export / "cases/alpha").exists())
        self.assertTrue((self.export / "cases/beta/record.json").is_file())
        cleanup_calls = [options for command, options in self.calls if command[:2] == ["docker", "rm"]]
        self.assertEqual(len(cleanup_calls), 2)
        for options in cleanup_calls:
            self.assertGreater(options.get("timeout", 0), 0)
            self.assertLessEqual(options["timeout"], 60)
            self.assertTrue(options["check"])
        errors = read_json(self.export / "shard-0-errors.json")
        self.assertEqual(errors[0]["name"], "alpha")

    def test_reused_output_is_downloaded_and_checked_before_networkless_execution(self):
        raw = b'{"result":"potential_race"}\n'
        sha = hashlib.sha256(raw).hexdigest()
        entry = {"uri": self.data["evidence_base_url"] + f"/objects/{sha}/output.json", "sha256": sha}
        self.data["tasks"][0].update(action="reuse", previous={"bundle_files": {"output.json": entry}})

        def download(uri, destination, digest, *, base):
            self.events.append(("download", "alpha"))
            self.assertEqual((uri, digest, base), (entry["uri"], sha, self.data["evidence_base_url"]))
            self.assertEqual(destination, self.output / "cases/alpha/output.json")
            destination.write_bytes(raw)

        with patch.object(ci_shard, "download_verified", side_effect=download) as retrieve:
            self.assertFalse(self.run_shard())
        retrieve.assert_called_once()
        self.assertEqual(self.events[:2], [("download", "alpha"), ("run", "alpha")])
        self.assertEqual((self.output / "cases/alpha/output.json").read_bytes(), raw)

    def test_bad_reuse_evidence_does_not_execute_case_or_stop_next_case(self):
        self.data["tasks"][0].update(action="reuse", previous={"bundle_files": {
            "output.json": {"uri": self.data["evidence_base_url"] + "/invalid", "sha256": "f" * 64}}})
        with patch.object(ci_shard, "download_verified", side_effect=ValueError("Downloaded evidence checksum mismatch")):
            self.assertTrue(self.run_shard())
        self.assertEqual([name for event, name in self.events if event == "run"], ["beta"])
        self.assertEqual(read_json(self.export / "shard-0-errors.json")[0]["name"], "alpha")
        self.assertTrue((self.output / "cases/beta/record.json").is_file())

    def test_empty_shard_still_produces_success_receipt_for_artifact_collection(self):
        self.assertFalse(self.run_shard(shard=3))
        self.assertEqual(self.calls, [])
        self.assertEqual(read_json(self.export / "shard-3-errors.json"), [])

    def test_confirmed_oom_is_recorded_before_container_removal(self):
        self.failures["alpha"] = 137
        self.oom_killed.add("alpha")
        self.assertTrue(self.run_shard())
        errors = read_json(self.export / "shard-0-errors.json")
        self.assertIn("10 GiB process-tree memory limit", errors[0]["error"])
        self.assertIn("OOMKilled", errors[0]["error"])
        self.assertEqual([event[0] for event in self.events[:3]], ["run", "inspect", "cleanup"])

    def test_nonzero_cleanup_prevents_export_of_a_potentially_running_worker(self):
        self.cleanup_failures["alpha"] = subprocess.CalledProcessError(1, ["docker", "rm"])
        self.assertTrue(self.run_shard())
        self.assertFalse((self.export / "cases/alpha").exists())
        self.assertTrue((self.export / "cases/beta/record.json").is_file())

    def test_unsafe_case_is_not_exported_and_other_cases_survive(self):
        sentinel = self.root / "host-only-sentinel"
        sentinel.write_text("HOST_FILE_OUTSIDE_CONTAINER_MOUNT")
        self.worker_outputs["alpha"] = lambda case: (case / "leak.txt").symlink_to(sentinel)
        original_export = ci_shard.export_case
        def checked_export(case, destination):
            self.assertEqual(self.events[-1][0], "cleanup")
            original_export(case, destination)
        with patch.object(ci_shard, "export_case", side_effect=checked_export):
            self.assertTrue(self.run_shard())
        self.assertFalse((self.export / "cases/alpha").exists())
        self.assertTrue((self.export / "cases/beta/record.json").is_file())
        errors = read_json(self.export / "shard-0-errors.json")
        self.assertIn("Artifact export rejected", errors[0]["error"])
        self.assertEqual([path.name for path in self.export.rglob("*") if path.is_symlink()], [])

    def test_export_directory_cannot_include_or_sit_inside_untrusted_work(self):
        write_json(self.plan, self.data)
        for exported in (self.output, self.output / "nested", self.root):
            with self.subTest(exported=exported), self.assertRaisesRegex(ValueError, "separate"):
                ci_shard.run_shard(self.plan, self.output, 0, export=exported)

    def test_valid_long_identifiers_have_distinct_bounded_container_names(self):
        self.data["run_id"] = "r" * 221
        self.data["tasks"] = [{"name": prefix + "x" * 220, "shard": 0, "action": "verify"}
                              for prefix in ("a", "b")]
        self.assertFalse(self.run_shard())
        names = [command[command.index("--name") + 1] for command, _ in self.calls
                 if command[:2] == ["docker", "run"]]
        self.assertEqual(len(set(names)), 2)
        self.assertTrue(all(len(name) <= 63 for name in names))

    def test_artifact_export_copies_only_regular_files(self):
        case, exported = self.root / "case", self.root / "exported"
        write_json(case / "nested" / "record.json", {"name": "alpha"})
        (case / "stderr.log").write_text("worker log")
        ci_shard.export_case(case, exported)
        self.assertEqual(read_json(exported / "nested" / "record.json"), {"name": "alpha"})
        self.assertEqual((exported / "stderr.log").read_text(), "worker log")

    def test_artifact_export_rejects_host_and_directory_symlinks_before_copying(self):
        sentinel = self.root / "host-only-sentinel"
        sentinel.write_text("HOST_FILE_OUTSIDE_CONTAINER_MOUNT")
        for name, target in (("file", sentinel), ("directory", self.root),
                             ("broken", self.root / "missing")):
            with self.subTest(name=name):
                case, exported = self.root / name, self.root / (name + "-exported")
                write_json(case / "record.json", {"name": "alpha"})
                (case / "leak.txt").symlink_to(target)
                with self.assertRaisesRegex(ValueError, "regular files and directories"):
                    ci_shard.export_case(case, exported)
                self.assertFalse(exported.exists())
        self.assertEqual(sentinel.read_text(), "HOST_FILE_OUTSIDE_CONTAINER_MOUNT")

    def test_artifact_export_rejects_fifo_without_opening_it(self):
        case, exported = self.root / "case", self.root / "exported"
        case.mkdir()
        os.mkfifo(case / "blocked.txt")
        with self.assertRaisesRegex(ValueError, "regular files and directories"):
            ci_shard.export_case(case, exported)
        self.assertFalse(exported.exists())


if __name__ == "__main__":
    unittest.main()
