"""Publish evidence first, then atomically advance a Git results snapshot."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import re
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .common import encode, identifier, repository, revision, file_hash, write_json, confined, digest
from .pipeline import catalog, collect_records, identity_manifest


def check_publication_report(plan_data, task, case, row):
    """Recheck transferred artifacts using this trusted checkout's composer."""
    from .reporting import report_arguments, run_report_script
    if row["automation"].get("report_status") != "base":
        return False
    identity = case / "identity.json"
    saved_identity = identity.read_bytes()
    identity_manifest(plan_data, task, case)
    if identity.read_bytes() != saved_identity:
        raise ValueError("Report identity differs from the planned evidence")
    run_report_script(Path(__file__).resolve().parents[2], "check_report.py", report_arguments(case, identity))
    return True


class GitHub:
    def __init__(self, repo, token=None):
        self.repo = repository(repo)
        self.token = token

    def request(self, method, path, body=None):
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "sentryward-automation"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        data = encode(body) if body is not None else None
        with urlopen(Request(f"https://api.github.com/repos/{self.repo}/{path}", data=data, headers=headers, method=method), timeout=60) as response:
            return json.load(response)

    def ref(self, branch):
        try:
            return self.request("GET", f"git/ref/heads/{branch}")["object"]["sha"]
        except HTTPError as error:
            if error.code != 404:
                raise
            return None

    def file(self, path, ref):
        # Raw media type supports catalogs above the Contents JSON API's 1 MiB limit.
        headers = {"Accept": "application/vnd.github.raw+json", "User-Agent": "sentryward-automation"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        with urlopen(Request(f"https://api.github.com/repos/{self.repo}/contents/{path}?ref={revision(ref)}",
                             headers=headers), timeout=60) as response:
            return response.read()


def fetch_baseline(repo, output):
    api = GitHub(repo)
    head = api.ref("results")
    if head is None:
        write_json(output, {})
        return None
    pointer = json.loads(api.file("latest.json", head))
    if pointer.get("schema") != "sentryward.latest.v1":
        raise ValueError("Unsupported publication pointer")
    run_id = identifier(pointer["run_id"])
    snapshot = pointer.get("snapshot_id", "")
    if not re.fullmatch(r"[a-f0-9]{64}", snapshot):
        raise ValueError("Invalid snapshot identity")
    if pointer["catalog_path"] != f"runs/{run_id}/{snapshot}/results.json":
        raise ValueError("Invalid snapshot path")
    raw = api.file(pointer["catalog_path"], head)
    if hashlib.sha256(raw).hexdigest() != pointer["catalog_sha256"]:
        raise ValueError("Published catalog checksum mismatch")
    result = json.loads(raw)
    if result.get("schema") != "sentryward.kernel-results.v1" or result.get("publication", {}).get("source_repository") != repo:
        raise ValueError("Published catalog belongs to another repository or schema")
    for key in ("run_id", "snapshot_id", "source_revision", "evidence_base_url"):
        if result["publication"][key] != pointer[key]:
            raise ValueError("Snapshot/pointer identity mismatch")
    result["_results_revision"] = head
    write_json(output, result)
    return result


# Only supervisor-owned files belong to a published verification bundle.
EVIDENCE_FILES = ("input.json", "kernel.ptx", "reference.py", "reference_binding.json",
                  "stdout.json", "stderr.log", "output.json", "identity.json", "report/report.md")


def snapshot_identity(plan_data, output, rows):
    files = {}
    for task in plan_data["tasks"]:
        case = confined(output, "cases/" + identifier(task["name"]))
        for name in (*EVIDENCE_FILES, "record.json"):
            path = confined(case, name)
            if path.exists():
                files[f"{task['name']}/{name}"] = file_hash(path)
    return digest({"plan": plan_data, "records": rows, "files": files})


def prepare_snapshot(plan_data, output, store, rows, snapshot):
    """No Git mutation until every published evidence object has been verified."""
    run_id = identifier(plan_data["run_id"])
    prefix = f"runs/{run_id}/{snapshot}"
    tasks = {task["name"]: task for task in plan_data["tasks"]}
    tree = []
    for row in rows:
        case = Path(output) / "cases" / row["name"]
        checked_report = check_publication_report(plan_data, tasks[row["name"]], case, row)
        bundle = {}
        for name in EVIDENCE_FILES:
            path = confined(case, name)
            if path.is_file():
                bundle[name] = store.upload(path)
        row["bundle_files"] = bundle
        report = case / "report" / "report.md"
        if checked_report:
            path = f"{prefix}/reports/{row['name']}/report.md"
            row["files"]["report"] = {"path": path, "sha256": file_hash(report),
                "url": f"https://github.com/{plan_data['source_repository']}/blob/results/{path}"}
            tree.append({"path": path, "mode": "100644", "type": "blob", "content": report.read_text()})
        row.setdefault("provenance", {})["snapshot_id"] = snapshot
        record_path = case / "published-record.json"
        write_json(record_path, row)
        row["evidence"] = store.upload(record_path)
    result = catalog(plan_data, rows)
    result["publication"]["snapshot_id"] = snapshot
    raw = encode(result)
    path = f"{prefix}/results.json"
    tree.append({"path": path, "mode": "100644", "type": "blob", "content": raw.decode()})
    manifest = {"schema": "sentryward.run.v1", "run_id": run_id, "snapshot_id": snapshot,
                "source_revision": plan_data["source_revision"], "source_repository": plan_data["source_repository"],
                "runtime": plan_data["runtime"], "config": plan_data["config"],
                "records": [{"name": r["name"], "evidence": r["evidence"], "automation": r["automation"]} for r in rows]}
    manifest_path = Path(output) / "run-manifest.json"
    write_json(manifest_path, manifest)
    manifest_evidence = store.upload(manifest_path)
    tree.append({"path": f"{prefix}/manifest.json", "mode": "100644", "type": "blob", "content": encode(manifest).decode()})
    pointer = {"schema": "sentryward.latest.v1", "run_id": run_id, "snapshot_id": snapshot,
               "source_revision": plan_data["source_revision"], "catalog_path": path,
               "catalog_sha256": hashlib.sha256(raw).hexdigest(), "evidence_base_url": plan_data["evidence_base_url"],
               "manifest_evidence": manifest_evidence}
    tree.append({"path": "latest.json", "mode": "100644", "type": "blob", "content": encode(pointer).decode()})
    return tree, pointer


def publish(plan_data, output, store, api):
    if api.repo != plan_data["source_repository"]:
        raise ValueError("Publisher repository differs from planned repository")
    expected_source = revision(plan_data["source_revision"])
    if api.ref("main") != expected_source:
        return {"status": "superseded", "source_revision": expected_source}
    rows = collect_records(plan_data, output)
    snapshot = snapshot_identity(plan_data, output, rows)
    parent = api.ref("results")
    if parent:
        current = json.loads(api.file("latest.json", parent))
        if current["snapshot_id"] == snapshot:
            return {"status": "already_published", "commit": parent}
        current_catalog = json.loads(api.file(current["catalog_path"], parent))
        current_time = datetime.fromisoformat(current_catalog["publication"]["created_at"].replace("Z", "+00:00"))
        planned_time = datetime.fromisoformat(plan_data["created_at"].replace("Z", "+00:00"))
        if current["source_revision"] == expected_source and current_time > planned_time:
            return {"status": "superseded", "source_revision": expected_source}
        if parent != plan_data.get("baseline_revision") and current["run_id"] != plan_data["run_id"]:
            raise ValueError("Results changed since planning; start a new dispatch")
        completed = {r["name"]: r for r in current_catalog["results"]
                     if r["automation"]["state"] in {"current", "reused"}}
        tasks = {t["name"]: t for t in plan_data["tasks"]}
        for row in rows:
            prior = completed.get(row["name"])
            if (prior and prior["automation"].get("verification_key") == tasks[row["name"]]["verification_key"]
                    and row["automation"]["state"] not in {"current", "reused"}):
                raise ValueError("Publication would discard a completed result: " + row["name"])
    entries, pointer = prepare_snapshot(plan_data, output, store, rows, snapshot)
    # Retrying an older snapshot must not rewrite immutable files or roll back latest.
    if parent:
        try:
            api.file(pointer["catalog_path"], parent)
        except HTTPError as error:
            if error.code != 404:
                raise
        else:
            return {"status": "already_published", "commit": parent}
    tree_body = {"tree": entries}
    if parent:
        tree_body["base_tree"] = api.request("GET", f"git/commits/{parent}")["tree"]["sha"]
    tree = api.request("POST", "git/trees", tree_body)["sha"]
    commit = api.request("POST", "git/commits", {"message": f"Record verification for {expected_source[:12]}",
                         "tree": tree, "parents": [parent] if parent else []})["sha"]
    if api.ref("main") != expected_source:
        return {"status": "superseded", "source_revision": expected_source}
    # Non-fast-forward updates fail if any other publisher changed the branch.
    if parent:
        api.request("PATCH", "git/refs/heads/results", {"sha": commit, "force": False})
    else:
        api.request("POST", "git/refs", {"ref": "refs/heads/results", "sha": commit})
    if api.ref("results") != commit:
        raise RuntimeError("Publication ref verification failed")
    return {"status": "published", "commit": commit, "pointer": pointer}
