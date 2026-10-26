"""Public automation contracts, using tiny tracked cases and a fake verifier."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tools.automation import pipeline
from tools.automation.common import file_hash, read_json, write_json


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.root = self.folder / "source"
        self.root.mkdir()
        self.output = self.folder / "output"
        self.runtime = {"python": "3.12.7", "native_sha256": "a" * 64,
                        "packages": {"torch": "test"}}
        self.config = {"verification": {"shards": 4, "memory_limit_mib": 8192,
                                        "timeout_seconds": 500}}
        self.add_case("alpha")
        self.add_case("beta")
        self.write("cpp/core.cpp", "// verifier revision one\n")
        self.write("skills/generate-kernel-report/SKILL.md", "Report skill revision one\n")
        self.write("tools/automation/reporting.py", "# report implementation\n")
        self.write("config/automation.json", json.dumps(self.config))
        self.write("README.md", "A tiny public fixture.\n")
        self.git("init", "-q")
        self.commit()
        self.raw = {"result": "potential_race", "checks": {
            "synchronization": "passed", "memory": "failed", "equivalence": "not_run"},
            "diagnostics": [{"code": "unordered_access", "stage": "memory_check"}],
            "warning_count": 0, "warnings_complete": True,
            "memory_analysis_complete": True, "search_complete": True,
            "peak_rss_mib": 128, "elapsed_seconds": 0.5,
            "future_evidence": {"unknown_fields_must_survive": True}}
        self.calls = []

    def write(self, relative, content):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def add_case(self, name):
        self.write(f"examples/{name}/input.json", json.dumps({"entry": name}))
        self.write(f"examples/{name}/reference.py", "def reference(): return 1\n")
        self.write(f"examples/{name}/reference_binding.json", "{}\n")
        self.write(f"corpus/{name}.ptx", f".visible .entry {name}() {{ ret; }}\n")

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.root, check=True,
                              text=True, capture_output=True).stdout.strip()

    def commit(self):
        self.git("add", ".")
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                 "-c", "commit.gpgsign=false", "commit", "-qm", "Fixture revision")

    def plan(self, **options):
        return pipeline.plan(self.root, self.runtime, run_id=options.pop("run_id", "run-1"),
                             source_repository="gpu-verifier/sentryward",
                             config=options.pop("config", self.config),
                             evidence_base_url="https://evidence.example.org/sentryward/",
                             **options)

    def task(self, plan, name="alpha"):
        return next(task for task in plan["tasks"] if task["name"] == name)

    def fake_process(self, command, case, timeout):
        self.calls.append((command, case, timeout))
        write_json(case / "stdout.json", self.raw)
        (case / "stderr.log").write_text("")
        return 0, False

    def verify(self, plan, name="alpha", runner=None, output=None):
        return pipeline.verify_task(self.root, plan, self.task(plan, name),
                                    output or self.output,
                                    process_runner=runner or self.fake_process)

    @staticmethod
    def keys(plan):
        return {task["name"]: task["verification_key"] for task in plan["tasks"]}

    def test_one_case_input_change_invalidates_only_its_verification(self):
        before = self.plan()
        self.write("examples/alpha/input.json", '{"entry":"alpha","parameter":2}\n')
        self.commit()
        after = self.plan()
        self.assertNotEqual(before["source_revision"], after["source_revision"])
        self.assertNotEqual(self.keys(before)["alpha"], self.keys(after)["alpha"])
        self.assertEqual(self.keys(before)["beta"], self.keys(after)["beta"])

    def test_ptx_reference_and_binding_changes_are_verification_inputs(self):
        before = self.plan()
        for relative in ("corpus/alpha.ptx", "examples/alpha/reference.py",
                         "examples/alpha/reference_binding.json"):
            with self.subTest(relative=relative):
                path = self.root / relative
                original = path.read_text()
                path.write_text(original + "\n")
                changed = self.plan()
                self.assertNotEqual(self.keys(before)["alpha"], self.keys(changed)["alpha"])
                self.assertEqual(self.keys(before)["beta"], self.keys(changed)["beta"])
                path.write_text(original)

    def test_verifier_change_invalidates_every_case(self):
        before = self.plan()
        self.write("cpp/core.cpp", "// revised verifier semantics\n")
        self.commit()
        after = self.plan()
        for name in self.keys(before):
            self.assertNotEqual(self.keys(before)[name], self.keys(after)[name])

    def test_runtime_and_verification_options_are_cache_identity(self):
        before = self.plan()
        self.runtime["native_sha256"] = "b" * 64
        changed = self.plan()
        self.assertTrue(all(self.keys(before)[name] != self.keys(changed)[name]
                            for name in self.keys(before)))
        self.config["verification"]["memory_limit_mib"] = 4096
        changed_options = self.plan()
        self.assertNotEqual(self.keys(changed), self.keys(changed_options))

    def test_shared_reference_helper_change_invalidates_all_kernels(self):
        before = self.plan()
        self.write("examples/_reference/helpers.py", "# shared mathematical reference\n")
        self.commit()
        after = self.plan()
        self.assertTrue(all(self.keys(before)[name] != self.keys(after)[name]
                            for name in self.keys(before)))

    def test_uncommitted_new_cases_do_not_get_nonexistent_git_links(self):
        self.add_case("untracked")
        self.assertNotIn("untracked", self.keys(self.plan()))

    def test_untracked_reference_does_not_get_a_committed_source_link(self):
        self.git("rm", "examples/alpha/reference.py")
        self.commit()
        self.write("examples/alpha/reference.py", "def reference(): return 2\n")
        self.assertNotIn("reference.py", self.task(self.plan())["files"])

    def test_ptx_must_be_committed_with_its_input(self):
        self.git("rm", "corpus/alpha.ptx")
        self.commit()
        self.write("corpus/alpha.ptx", ".visible .entry alpha() { ret; }\n")
        with self.assertRaisesRegex(ValueError, "Missing committed PTX"):
            self.plan()

    def test_skill_and_report_implementation_do_not_invalidate_verification(self):
        before = self.plan()
        for relative in ("skills/generate-kernel-report/SKILL.md",
                         "tools/automation/reporting.py"):
            with self.subTest(relative=relative):
                path = self.root / relative
                original = path.read_text()
                path.write_text(original + "\nRevision two\n")
                after = self.plan()
                self.assertEqual(self.keys(before), self.keys(after))
                self.assertNotEqual(self.task(before)["report_version"],
                                    self.task(after)["report_version"])
                path.write_text(original)

    def test_documentation_revision_keeps_both_cache_keys(self):
        before = self.plan()
        self.write("README.md", "Updated documentation.\n")
        self.commit()
        after = self.plan()
        self.assertNotEqual(before["source_revision"], after["source_revision"])
        self.assertEqual(self.keys(before), self.keys(after))
        self.assertEqual(self.task(before)["report_version"], self.task(after)["report_version"])

    def test_publication_and_sharding_changes_reuse_completed_verification(self):
        paths = ("tools/automation/publication.py", "tools/automation/storage.py")
        for relative in paths:
            self.write(relative, "# publication implementation\n")
        self.commit()
        before = self.plan()
        rows = [self.verify(before, name) for name in ("alpha", "beta")]
        baseline = pipeline.catalog(before, rows)
        for relative in paths:
            with self.subTest(relative=relative):
                path = self.root / relative
                original = path.read_text()
                path.write_text(original + "# updated\n")
                after = self.plan(baseline=baseline)
                path.write_text(original)
                self.assertEqual(self.keys(before), self.keys(after))
                self.assertTrue(all(t["action"] == "reuse" for t in after["tasks"]))
        self.config["verification"]["shards"] = 1
        after = self.plan(baseline=baseline)
        self.assertEqual(self.keys(before), self.keys(after))
        self.assertTrue(all(t["action"] == "reuse" and t["shard"] == 0 for t in after["tasks"]))

    def test_new_kernel_is_discovered_without_workflow_changes(self):
        before = self.plan()
        self.add_case("gamma")
        self.commit()
        after = self.plan()
        self.assertEqual([task["name"] for task in after["tasks"]], ["alpha", "beta", "gamma"])
        self.assertEqual(self.task(after, "gamma")["action"], "verify")
        for name in self.keys(before):
            self.assertEqual(self.keys(before)[name], self.keys(after)[name])

    def test_historical_timeout_is_scheduled(self):
        self.write("examples/alpha/output.json", '{"result":"timeout","peak_rss_mib":32}\n')
        planned = self.plan()
        self.assertEqual(self.task(planned)["action"], "verify")
        self.verify(planned)
        self.assertEqual(len(self.calls), 1)

    def test_high_memory_case_is_deferred_without_running_process(self):
        self.write("examples/alpha/output.json", '{"result":"timeout","peak_rss_mib":88064}\n')
        planned = self.plan()
        row = self.verify(planned)
        self.assertEqual(row["automation"]["state"], "deferred_resource")
        self.assertEqual(row["automation"]["reason"], "historical_memory_exceeds_runner_budget")
        self.assertIsNone(row["result"])
        self.assertEqual(row["automation"]["report_status"], "not_run")
        self.assertEqual(self.calls, [])

    def test_selected_case_leaves_other_cases_explicitly_pending(self):
        planned = self.plan(names=["alpha"])
        row = self.verify(planned, "beta")
        self.assertEqual(row["automation"]["state"], "pending")
        self.assertEqual(row["automation"]["reason"], "outside_requested_selection")
        self.assertIsNone(row["result"])
        self.assertEqual(self.calls, [])
        with self.assertRaisesRegex(ValueError, "Unknown examples"):
            self.plan(names=["missing"])

    def test_exit_zero_with_finding_preserves_result_and_unknown_evidence(self):
        planned = self.plan()
        original = (self.root / "examples/alpha/input.json").read_bytes()
        row = self.verify(planned)
        self.assertEqual(row["result"], "potential_race")
        self.assertEqual(row["checks"]["memory"], "failed")
        self.assertEqual(row["automation"]["exit_code"], 0)
        self.assertEqual(row["automation"]["state"], "current")
        saved = read_json(self.output / "cases/alpha/output.json")
        self.assertEqual(saved, self.raw)
        command, case, timeout = self.calls[0]
        self.assertEqual(command[command.index("--memory-limit-mib") + 1], "8192")
        self.assertEqual(command[command.index("--timeout") + 1], "500")
        self.assertEqual(timeout, 515)
        self.assertEqual((self.root / "examples/alpha/input.json").read_bytes(), original)
        self.assertFalse((self.root / "examples/alpha/output.json").exists())
        self.assertEqual(row["provenance"]["source_repository"], planned["source_repository"])
        self.assertEqual(row["budgets"], self.config["verification"])
        self.assertFalse((case / "run.json").exists())

    def test_unknown_result_category_is_not_rewritten(self):
        self.raw["result"] = "future_verifier_category"
        row = self.verify(self.plan())
        self.assertEqual(row["result"], "future_verifier_category")
        self.assertEqual(row["automation"]["state"], "current")

    def test_invalid_stdout_is_tool_failure(self):
        for number, payload in enumerate(("not JSON", "[]", '{"result":null}', '{"result":""}', "{}")):
            with self.subTest(payload=payload):
                def invalid(command, case, timeout):
                    (case / "stdout.json").write_text(payload)
                    return 0, False
                row = self.verify(self.plan(), runner=invalid, output=self.folder / f"invalid-{number}")
                self.assertEqual(row["result"], "tool_failure")
                self.assertEqual(row["automation"]["state"], "failed")
                self.assertEqual(row["checks"]["memory"], "not_run")

    def test_process_deadline_is_timeout_even_if_stdout_claims_success(self):
        def expired(command, case, timeout):
            write_json(case / "stdout.json", {"result": "verified"})
            return -9, True
        row = self.verify(self.plan(), runner=expired)
        self.assertEqual(row["result"], "timeout")
        self.assertEqual(row["checks"]["memory"], "not_run")
        self.assertEqual(row["automation"]["exit_code"], -9)

    def test_memory_limit_result_and_nonzero_exit_are_preserved(self):
        self.raw["result"] = "out_of_memory"
        def memory_limited(command, case, timeout):
            self.fake_process(command, case, timeout)
            return 2, False
        row = self.verify(self.plan(), runner=memory_limited)
        self.assertEqual(row["result"], "out_of_memory")
        self.assertEqual(row["automation"]["exit_code"], 2)

    def test_source_changes_after_plan_are_rejected_before_execution(self):
        planned = self.plan()
        self.write("examples/alpha/input.json", '{"entry":"different"}\n')
        with self.assertRaisesRegex(ValueError, "Source changed after task planning"):
            self.verify(planned)
        self.assertEqual(self.calls, [])

    def test_process_cannot_change_frozen_input(self):
        def mutation(command, case, timeout):
            self.fake_process(command, case, timeout)
            (case / "input.json").write_text('{}\n')
            return 0, False
        with self.assertRaisesRegex(ValueError, "Frozen input changed"):
            self.verify(self.plan(), runner=mutation)

    def test_matching_result_reuses_raw_evidence_without_verifier(self):
        original_plan = self.plan()
        previous = self.verify(original_plan)
        raw_bytes = (self.output / "cases/alpha/output.json").read_bytes()
        next_plan = self.plan(run_id="run-2", baseline={"results": [previous]})
        self.assertEqual(self.task(next_plan)["action"], "reuse")
        self.calls.clear()
        def retrieve(uri, destination, sha, *, base):
            self.assertEqual(uri, previous["bundle_files"]["output.json"]["uri"])
            destination.write_bytes(raw_bytes)
            self.assertEqual(file_hash(destination), sha)
        with patch.object(pipeline, "download_verified", side_effect=retrieve) as download:
            row = self.verify(next_plan, output=self.folder / "reused")
        download.assert_called_once()
        self.assertEqual(self.calls, [])
        self.assertEqual(row["automation"]["state"], "reused")
        self.assertEqual(row["result"], previous["result"])
        self.assertEqual(row["provenance"]["run_id"], "run-2")

    def test_force_runs_selected_case_again(self):
        original = self.plan()
        previous = [self.verify(original, name) for name in ("alpha", "beta")]
        planned = self.plan(baseline={"results": previous}, names=["alpha"], force=True)
        self.assertEqual(self.task(planned)["action"], "verify")
        self.assertEqual(self.task(planned, "beta")["action"], "reuse")

    def test_missing_raw_output_cannot_be_selected_for_reuse(self):
        previous = self.verify(self.plan())
        del previous["bundle_files"]["output.json"]
        planned = self.plan(baseline={"results": [previous]})
        self.assertEqual(self.task(planned)["action"], "verify")

    def test_collect_rejects_another_run_even_with_matching_inputs(self):
        planned = self.plan()
        row = self.verify(planned)
        row["provenance"]["run_id"] = "another-run"
        write_json(self.output / "cases/alpha/record.json", row)
        with self.assertRaises(ValueError):
            pipeline.collect_records(planned, self.output)

    def test_collect_rejects_attempted_revision_and_raw_summary_conflicts(self):
        planned = self.plan()
        row = self.verify(planned)
        path = self.output / "cases/alpha/record.json"
        original = copy.deepcopy(row)
        row["automation"]["attempted_revision"] = "0" * 40
        write_json(path, row)
        with self.assertRaises(ValueError):
            pipeline.collect_records(planned, self.output)
        original["result"] = "verified"
        write_json(path, original)
        with self.assertRaisesRegex(ValueError, "summary differs"):
            pipeline.collect_records(planned, self.output)

    def test_missing_shard_record_is_explicit_failure(self):
        planned = self.plan()
        self.verify(planned)
        rows = pipeline.collect_records(planned, self.output)
        missing = next(row for row in rows if row["name"] == "beta")
        self.assertIsNone(missing["result"])
        self.assertEqual(missing["automation"]["state"], "failed")
        self.assertEqual(missing["automation"]["reason"], "worker_record_missing")
        self.assertEqual(missing["automation"]["attempted_revision"], planned["source_revision"])

    def test_catalog_counts_only_current_and_reused_results(self):
        planned = self.plan()
        current = self.verify(planned)
        reused = copy.deepcopy(current)
        reused.update(name="reused", result="verified")
        reused["automation"]["state"] = "reused"
        rows = [current, reused]
        for state in ("pending", "deferred_resource", "failed", "historical"):
            row = copy.deepcopy(current)
            row.update(name=state, result="verified")
            row["automation"]["state"] = state
            rows.append(row)
        snapshot = pipeline.catalog(planned, rows)
        self.assertEqual(snapshot["counts"], {"potential_race": 1, "verified": 1})
        self.assertEqual(snapshot["publication"]["run_id"], planned["run_id"])
        self.assertEqual(snapshot["publication"]["source_revision"], planned["source_revision"])

    def test_partial_failure_retains_previous_evidence_and_worker_error(self):
        original = self.plan()
        previous = self.verify(original)
        planned = self.plan(baseline={"results": [previous]}, force=True)
        fresh = self.folder / "interrupted"
        write_json(fresh / "shard-0-errors.json", [{"name": "alpha", "error": "Worker container exited 137"}])
        row = pipeline.collect_records(planned, fresh)[0]
        self.assertEqual(row["automation"]["state"], "failed")
        self.assertIsNone(row["result"])
        self.assertEqual(row["historical"]["result"], "potential_race")
        self.assertIn("137", row["automation"]["execution_errors"][0])
        retried = self.plan(baseline={"results": [row]})
        self.assertEqual(self.task(retried)["historical"]["result"], "potential_race")

    def test_deferred_catalog_keeps_history_and_pins_current_sources(self):
        study = {"source_revision": "1" * 40, "description": "Original B200 study"}
        historical = {"name": "alpha", "result": "verified", "checks": {"memory": "passed"},
                      "synchronization_removal": {"input_sha256": "2" * 64}}
        self.write("examples/results.json", json.dumps({"results": [historical],
                   "synchronization_removal_study": study}))
        self.write("examples/alpha/output.json", '{"result":"verified","peak_rss_mib":88064}\n')
        self.commit()
        planned = self.plan()
        row = self.verify(planned)
        snapshot = pipeline.catalog(planned, [row])
        self.assertIsNone(row["result"])
        self.assertEqual(row["historical"]["result"], "verified")
        self.assertEqual(row["automation"]["state"], "deferred_resource")
        self.assertEqual(snapshot["counts"], {})
        self.assertEqual(row["historical"]["checks"], {"memory": "passed"})
        self.assertNotIn("checks", row)
        self.assertEqual(snapshot["synchronization_removal_study"], study)
        self.assertEqual(row["synchronization_removal"], historical["synchronization_removal"])
        self.assertEqual(set(row["files"]), {"input", "ptx", "reference", "reference_binding"})
        for descriptor in row["files"].values():
            self.assertEqual(descriptor["revision"], planned["source_revision"])
            self.assertEqual(descriptor["sha256"], file_hash(self.root / descriptor["path"]))
        self.assertEqual(row["automation"]["attempted_revision"], snapshot["publication"]["source_revision"])
        pending = self.verify(self.plan(names=["beta"]), output=self.folder / "pending")
        self.assertEqual(pending["automation"]["state"], "pending")
        self.assertIsNone(pending["result"])
        self.assertEqual(pending["historical"]["result"], "verified")
        row["result"] = "verified"
        write_json(self.output / "cases/alpha/record.json", row)
        with self.assertRaisesRegex(ValueError, "Unexecuted task"):
            pipeline.collect_records(planned, self.output)


if __name__ == "__main__":
    unittest.main()
