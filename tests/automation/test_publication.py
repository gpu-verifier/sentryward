"""Evidence-first publication and Git reference races, with an in-memory API."""
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from tools.automation.common import encode, file_hash, object_url, write_json
from tools.automation.publication import GitHub, fetch_baseline, publish
from tools.automation.__main__ import report_tasks


ROOT = Path(__file__).resolve().parents[2]


def missing():
    return api_error(404, "missing")


def api_error(code, message):
    error = HTTPError("https://api.github.com/repos/example/sentryward/contents", code, message, {}, io.BytesIO())
    error.close()
    return error


class EvidenceStore:
    def __init__(self, events):
        self.events = events
        self.objects = {}
        self.fail_at = None

    def upload(self, path):
        path = Path(path)
        self.events.append(("upload", path.name))
        if self.fail_at == path.name:
            raise OSError("Evidence upload interrupted")
        digest = file_hash(path)
        uri = object_url("https://evidence.example.test/sentryward", digest, path.name)
        self.objects[uri] = path.read_bytes()
        return {"sha256": digest, "uri": uri}


class FakeGitHub:
    """Preserves Git's fast-forward requirement instead of merely counting calls."""
    repo = "example/sentryward"

    def __init__(self, source_revision, events):
        self.events = events
        self.refs = {"main": source_revision}
        self.trees = {}
        self.commits = {}
        self.after_commit = None

    def ref(self, branch):
        self.events.append(("ref", branch))
        return self.refs.get(branch)

    def file(self, path, ref):
        tree = self.trees[self.commits[ref]["tree"]]
        if path not in tree:
            raise missing()
        return tree[path]

    def request(self, method, path, body=None):
        self.events.append((method, path))
        if method == "GET" and path.startswith("git/commits/"):
            return {"tree": {"sha": self.commits[path.rsplit("/", 1)[1]]["tree"]}}
        if (method, path) == ("POST", "git/trees"):
            files = dict(self.trees.get(body.get("base_tree"), {}))
            files.update({entry["path"]: entry["content"].encode() for entry in body["tree"]})
            sha = hashlib.sha1(encode({name: data.decode() for name, data in files.items()})).hexdigest()
            self.trees[sha] = files
            return {"sha": sha}
        if (method, path) == ("POST", "git/commits"):
            sha = hashlib.sha1(encode(body)).hexdigest()
            self.commits[sha] = dict(body)
            if self.after_commit:
                self.after_commit(sha)
            return {"sha": sha}
        if (method, path) == ("POST", "git/refs"):
            if "results" in self.refs:
                raise api_error(422, "already exists")
            if body["ref"] != "refs/heads/results":
                raise AssertionError(body)
            self.refs["results"] = body["sha"]
            return {}
        if (method, path) == ("PATCH", "git/refs/heads/results"):
            if body.get("force") is not False:
                raise AssertionError("Publication must never force-push")
            if self.refs["results"] not in self.commits[body["sha"]]["parents"]:
                raise api_error(422, "not fast forward")
            self.refs["results"] = body["sha"]
            return {}
        raise AssertionError((method, path, body))


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name).resolve()
        self.case = self.output / "cases/kernel"
        self.case.mkdir(parents=True)
        self.source = "a" * 40
        self.plan = {
            "run_id": "run-1", "source_revision": self.source,
            "source_repository": "example/sentryward", "runtime": {"native_sha256": "b" * 64},
            "config": {"verification": {}}, "created_at": "2026-10-21T00:00:00Z",
            "evidence_base_url": "https://evidence.example.test/sentryward",
            "tasks": [{"name": "kernel", "verification_key": "v1", "report_version": "report-v1", "hashes": {}, "files": {}}],
        }
        inputs = {"input.json": b'{"entry":"kernel"}\n', "kernel.ptx": b".visible .entry kernel() { ret; }\n"}
        for name, data in inputs.items():
            (self.case / name).write_bytes(data)
            self.plan["tasks"][0]["hashes"][name] = hashlib.sha256(data).hexdigest()
            self.plan["tasks"][0]["files"][name] = "corpus/kernel.ptx" if name == "kernel.ptx" else "examples/kernel/input.json"
        write_json(self.case / "output.json", {"result": "potential_race", "checks": {"memory": "failed"}, "diagnostics": [], "future_field": {"retain": 7}})
        self.row = {
            "name": "kernel", "result": "potential_race", "checks": {"memory": "failed"}, "diagnostics": [],
            "automation": {"state": "current", "verification_key": "v1", "report_status": "base",
                           "attempted_revision": self.source},
            "provenance": {"source_revision": self.source, "runtime": self.plan["runtime"], "run_id": "run-1"},
            "files": {}, "bundle_files": {},
        }
        for filename, kind in (("input.json", "input"), ("kernel.ptx", "ptx"), ("output.json", "result")):
            sha = file_hash(self.case / filename)
            uri = object_url(self.plan["evidence_base_url"], sha, filename)
            self.row["bundle_files"][filename] = {"sha256": sha, "uri": uri}
            self.row["files"][kind] = {"path": self.plan["tasks"][0]["files"].get(filename, "examples/kernel/output.json"), "sha256": sha}
            if kind == "result":
                self.row["files"][kind].update(url=uri, raw_url=uri)
            else:
                self.row["files"][kind]["revision"] = self.source
        self.save_row()
        with patch("builtins.print"):
            report_tasks(ROOT, self.plan, self.output)
        self.row = json.loads((self.case / "record.json").read_text())
        self.events = []
        self.store = EvidenceStore(self.events)
        self.api = FakeGitHub(self.source, self.events)

    def save_row(self):
        write_json(self.case / "record.json", self.row)

    def mutations(self):
        return [event for event in self.events if event[0] in {"POST", "PATCH"}]

    def test_one_report_failure_preserves_results_and_other_reports(self):
        from tools.automation.reporting import generate_report
        task = copy.deepcopy(self.plan["tasks"][0])
        task["name"] = "other"
        self.plan["tasks"].append(task)
        other = self.output / "cases/other"
        shutil.copytree(self.case, other)
        row = copy.deepcopy(self.row)
        row["name"] = "other"
        row["files"]["result"]["path"] = "examples/other/output.json"
        write_json(other / "record.json", row)

        def compose(root, case, identity):
            if case == self.case:
                raise ValueError("Report check failed")
            return generate_report(root, case, identity)

        with patch("tools.automation.reporting.generate_report", side_effect=compose):
            report_tasks(ROOT, self.plan, self.output)
        preview = json.loads((self.output / "preview-results.json").read_text())
        self.assertEqual([r["automation"]["report_status"] for r in preview["results"]], ["failed", "base"])
        self.assertEqual(preview["counts"], {"potential_race": 2})
        result = publish(self.plan, self.output, self.store, self.api)
        catalog = json.loads(self.api.file(result["pointer"]["catalog_path"], result["commit"]))
        failed, passed = catalog["results"]
        self.assertEqual(failed["automation"]["report_status"], "failed")
        self.assertEqual(failed["result"], "potential_race")
        self.assertNotIn("report", failed["files"])
        self.assertEqual(passed["automation"]["report_status"], "base")
        self.assertIn("report", passed["files"])

    def test_all_evidence_precedes_atomic_ref_update(self):
        result = publish(self.plan, self.output, self.store, self.api)
        self.assertEqual(result["status"], "published")
        first_mutation = next(i for i, event in enumerate(self.events) if event[0] in {"POST", "PATCH"})
        self.assertTrue(all(i < first_mutation for i, event in enumerate(self.events) if event[0] == "upload"))
        head = self.api.refs["results"]
        pointer = json.loads(self.api.file("latest.json", head))
        raw_catalog = self.api.file(pointer["catalog_path"], head)
        self.assertEqual(pointer["catalog_sha256"], hashlib.sha256(raw_catalog).hexdigest())
        row = json.loads(raw_catalog)["results"][0]
        self.assertEqual(row["result"], "potential_race")
        self.assertEqual(json.loads(self.store.objects[row["bundle_files"]["output.json"]["uri"]])["future_field"], {"retain": 7})
        self.assertEqual(row["files"]["input"]["revision"], self.source)
        self.assertEqual(self.api.file(row["files"]["report"]["path"], head), (self.case / "report/report.md").read_bytes())
        self.assertIn(row["evidence"]["uri"], self.store.objects)
        self.assertEqual(self.mutations()[-1], ("POST", "git/refs"))

    def test_github_reads_large_catalog_as_raw_bytes_at_an_exact_commit(self):
        payload = b'{"results":[]}' + b" " * (1024 * 1024)
        api = GitHub("example/sentryward", token="example-not-a-real-token")
        with patch("tools.automation.publication.urlopen", return_value=io.BytesIO(payload)) as request:
            self.assertEqual(api.file("runs/run-1/results.json", self.source), payload)
        sent = request.call_args.args[0]
        headers = {key.lower(): value for key, value in sent.header_items()}
        self.assertEqual(headers["accept"], "application/vnd.github.raw+json")
        self.assertTrue(sent.full_url.endswith("/contents/runs/run-1/results.json?ref=" + self.source))
        self.assertEqual(headers["authorization"], "Bearer example-not-a-real-token")
        with self.assertRaises(ValueError):
            api.file("runs/run-1/results.json", "main")

    def test_upload_failure_leaves_git_untouched(self):
        self.store.fail_at = "output.json"
        with self.assertRaisesRegex(OSError, "interrupted"):
            publish(self.plan, self.output, self.store, self.api)
        self.assertNotIn("results", self.api.refs)
        self.assertEqual(self.mutations(), [])

    def test_catalog_drops_unmaterialized_worker_evidence_references(self):
        self.row["bundle_files"]["ghost.json"] = {
            "sha256": "f" * 64,
            "uri": object_url(self.plan["evidence_base_url"], "f" * 64, "ghost.json"),
        }
        self.save_row()
        result = publish(self.plan, self.output, self.store, self.api)
        catalog = json.loads(self.api.file(result["pointer"]["catalog_path"], self.api.refs["results"]))
        bundle = catalog["results"][0]["bundle_files"]
        self.assertNotIn("ghost.json", bundle)
        for descriptor in bundle.values():
            self.assertIn(descriptor["uri"], self.store.objects)
            self.assertEqual(descriptor["sha256"], hashlib.sha256(self.store.objects[descriptor["uri"]]).hexdigest())

    def test_same_published_run_is_idempotent(self):
        first = publish(self.plan, self.output, self.store, self.api)
        self.events.clear()
        repeated = publish(self.plan, self.output, self.store, self.api)
        self.assertEqual(repeated, {"status": "already_published", "commit": first["commit"]})
        self.assertEqual(self.mutations(), [])
        self.assertFalse(any(event[0] == "upload" for event in self.events))

    def test_successful_retry_publishes_a_new_immutable_snapshot(self):
        saved = (self.case / "record.json").read_bytes()
        (self.case / "record.json").unlink()
        first = publish(self.plan, self.output, self.store, self.api)
        old_path = first["pointer"]["catalog_path"]
        original = self.api.file(old_path, first["commit"])
        (self.case / "record.json").write_bytes(saved)
        repaired = publish(self.plan, self.output, self.store, self.api)
        self.assertEqual(repaired["status"], "published")
        self.assertNotEqual(repaired["pointer"]["catalog_path"], old_path)
        self.assertEqual(self.api.file(old_path, repaired["commit"]), original)
        current = json.loads(self.api.file(repaired["pointer"]["catalog_path"], repaired["commit"]))
        self.assertEqual(current["results"][0]["automation"]["state"], "current")

    def test_stale_partial_plan_cannot_discard_completed_results(self):
        first = publish(self.plan, self.output, self.store, self.api)
        newer = copy.deepcopy(self.plan)
        newer.update(run_id="run-2", created_at="2026-10-21T00:01:00Z")
        (self.case / "record.json").unlink()
        self.events.clear()
        with self.assertRaisesRegex(ValueError, "Results changed since planning"):
            publish(newer, self.output, self.store, self.api)
        self.assertEqual(self.api.refs["results"], first["commit"])
        self.assertEqual(self.mutations(), [])

    def test_retry_cannot_discard_an_already_completed_case(self):
        first = publish(self.plan, self.output, self.store, self.api)
        (self.case / "record.json").unlink()
        with self.assertRaisesRegex(ValueError, "completed result"):
            publish(self.plan, self.output, self.store, self.api)
        self.assertEqual(self.api.refs["results"], first["commit"])

    def test_publication_retries_same_snapshot_after_an_interrupted_commit(self):
        def interrupt(commit):
            raise OSError("Git commit interrupted")

        self.api.after_commit = interrupt
        with self.assertRaisesRegex(OSError, "interrupted"):
            publish(self.plan, self.output, self.store, self.api)
        self.assertNotIn("results", self.api.refs)
        prepared = copy.deepcopy(self.api.trees)
        self.api.after_commit = None
        retried = publish(self.plan, self.output, self.store, self.api)
        self.assertEqual(retried["status"], "published")
        self.assertEqual(self.api.trees, prepared)

    def test_old_source_is_not_published(self):
        self.api.refs["main"] = "c" * 40
        result = publish(self.plan, self.output, self.store, self.api)
        self.assertEqual(result["status"], "superseded")
        self.assertEqual(self.mutations(), [])
        self.assertEqual(self.store.objects, {})

    def test_main_advancing_during_upload_cannot_change_latest(self):
        self.api.after_commit = lambda commit: self.api.refs.update(main="c" * 40)
        result = publish(self.plan, self.output, self.store, self.api)
        self.assertEqual(result["status"], "superseded")
        self.assertNotIn("results", self.api.refs)
        self.assertFalse(any(path in {"git/refs", "git/refs/heads/results"} for _, path in self.mutations()))

    def test_concurrent_results_writer_is_not_overwritten(self):
        publish(self.plan, self.output, self.store, self.api)
        previous = self.api.refs["results"]
        self.plan["baseline_revision"] = previous
        self.plan["run_id"] = "run-2"
        self.row["provenance"]["run_id"] = "run-2"
        self.save_row()
        concurrent = "f" * 40

        def advance_results(commit):
            self.api.commits[concurrent] = dict(self.api.commits[previous], parents=[previous])
            self.api.refs["results"] = concurrent

        self.api.after_commit = advance_results
        with self.assertRaises(HTTPError) as caught:
            publish(self.plan, self.output, self.store, self.api)
        self.assertEqual(caught.exception.code, 422)
        self.assertEqual(self.api.refs["results"], concurrent)
        self.assertEqual(self.api.file("latest.json", concurrent), self.api.file("latest.json", previous))

    def test_new_dispatch_keeps_old_snapshot_immutable(self):
        first = publish(self.plan, self.output, self.store, self.api)
        old_path = first["pointer"]["catalog_path"]
        original = self.api.file(old_path, first["commit"])
        self.plan.update(run_id="run-2", baseline_revision=first["commit"])
        self.row["provenance"]["run_id"] = "run-2"
        self.save_row()
        second = publish(self.plan, self.output, self.store, self.api)
        self.assertEqual(second["status"], "published")
        self.assertEqual(self.api.file(old_path, second["commit"]), original)
        self.assertNotEqual(first["pointer"]["snapshot_id"], second["pointer"]["snapshot_id"])

    def test_record_identity_mismatch_stops_before_upload(self):
        self.row["automation"]["verification_key"] = "wrong-task"
        self.save_row()
        with self.assertRaisesRegex(ValueError, "does not match"):
            publish(self.plan, self.output, self.store, self.api)
        self.assertEqual(self.store.objects, {})
        self.assertEqual(self.mutations(), [])

    def test_tampered_current_record_cannot_publish_different_facts_or_references(self):
        original = copy.deepcopy(self.row)
        changes = (
            lambda row: row["automation"].update(attempted_revision="f" * 40),
            lambda row: row["provenance"].update(source_revision="f" * 40),
            lambda row: row["provenance"].update(run_id="another-run"),
            lambda row: row["provenance"].update(runtime={"native_sha256": "f" * 64}),
            lambda row: row.update(result="verified"),
            lambda row: row["checks"].update(memory="passed"),
            lambda row: row["files"]["input"].update(revision="f" * 40),
            lambda row: row["bundle_files"]["output.json"].update(sha256="f" * 64),
        )
        for number, change in enumerate(changes):
            with self.subTest(change=number):
                self.events.clear()
                self.store = EvidenceStore(self.events)
                self.api = FakeGitHub(self.source, self.events)
                self.row = copy.deepcopy(original)
                change(self.row)
                self.save_row()
                with self.assertRaises(ValueError):
                    publish(self.plan, self.output, self.store, self.api)
                self.assertEqual(self.store.objects, {})
                self.assertEqual(self.mutations(), [])

    def test_changed_or_missing_current_input_blocks_publication(self):
        path = self.case / "input.json"
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "input identity"):
            publish(self.plan, self.output, self.store, self.api)
        path.unlink()
        with self.assertRaises(OSError):
            publish(self.plan, self.output, self.store, self.api)
        self.assertEqual(self.store.objects, {})
        self.assertEqual(self.mutations(), [])

    def test_late_run_for_same_source_cannot_replace_a_newer_run(self):
        newer = copy.deepcopy(self.plan)
        newer.update(run_id="run-2", created_at="2026-10-21T00:01:00Z")
        self.row["provenance"]["run_id"] = "run-2"
        self.save_row()
        publish(newer, self.output, self.store, self.api)
        latest = self.api.refs["results"]
        self.row["provenance"]["run_id"] = "run-1"
        self.save_row()
        self.events.clear()
        result = publish(self.plan, self.output, self.store, self.api)
        self.assertEqual(result["status"], "superseded")
        self.assertEqual(self.api.refs["results"], latest)
        self.assertEqual(self.mutations(), [])

    def test_tampered_report_or_wrong_immutable_reference_blocks_publication(self):
        report = self.case / "report/report.md"
        original = report.read_text()
        self.assertIn(f"/blob/{self.source}/", original)
        for text in (original + "\nThis kernel is unconditionally verified.\n",
                     original.replace(f"/blob/{self.source}/", "/blob/" + "f" * 40 + "/", 1)):
            with self.subTest(modification=text[-80:]):
                self.events.clear()
                self.store = EvidenceStore(self.events)
                self.api = FakeGitHub(self.source, self.events)
                report.write_text(text)
                with self.assertRaises(ValueError):
                    publish(self.plan, self.output, self.store, self.api)
                self.assertEqual(self.mutations(), [])
                self.assertEqual(self.store.objects, {})


    def test_missing_worker_is_explicitly_failed_not_current(self):
        missing_plan = copy.deepcopy(self.plan)
        missing_plan["tasks"].append({"name": "lost-worker", "verification_key": "v2"})
        result = publish(missing_plan, self.output, self.store, self.api)
        catalog = json.loads(self.api.file(result["pointer"]["catalog_path"], self.api.refs["results"]))
        missing_row = catalog["results"][1]
        self.assertEqual(missing_row["automation"]["state"], "failed")
        self.assertEqual(missing_row["automation"]["reason"], "worker_record_missing")
        self.assertIsNone(missing_row["result"])
        self.assertEqual(catalog["counts"], {"potential_race": 1})

    def test_baseline_checks_catalog_digest_and_pointer_identity(self):
        publish(self.plan, self.output, self.store, self.api)
        with patch("tools.automation.publication.GitHub", return_value=self.api):
            baseline = fetch_baseline(self.api.repo, self.output / "baseline.json")
            self.assertEqual(baseline["publication"]["run_id"], "run-1")
            self.assertEqual(baseline["_results_revision"], self.api.refs["results"])
            self.assertEqual(json.loads((self.output / "baseline.json").read_text()), baseline)
            tree = self.api.trees[self.api.commits[self.api.refs["results"]]["tree"]]
            pointer = json.loads(tree["latest.json"])
            pointer["source_revision"] = "d" * 40
            tree["latest.json"] = encode(pointer)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                fetch_baseline(self.api.repo, self.output / "baseline.json")
            pointer["source_revision"] = self.source
            pointer["catalog_sha256"] = "0" * 64
            tree["latest.json"] = encode(pointer)
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                fetch_baseline(self.api.repo, self.output / "baseline.json")


if __name__ == "__main__":
    unittest.main()
