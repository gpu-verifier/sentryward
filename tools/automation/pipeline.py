"""Plan content-addressed tasks and run the public verifier in isolated bundles."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time

from .common import (confined, digest, file_hash, identifier,
                     object_url, read_json, repository, revision, write_json,
                     download_verified)

VERIFIER_PATHS = ("cpp/", "sentryward/", "vendor/", "CMakeLists.txt", "pyproject.toml",
                  "tools/automation/", "config/", "examples/_reference/",
                  ".dockerignore", ".github/workflows/continuous-verification.yml")
CASE_FILES = ("input.json", "reference.py", "reference_binding.json")
SUMMARY_FIELDS = ("result", "checks", "diagnostics", "scope", "analysis_scope",
                  "warning_count", "warnings_complete", "warning_samples_truncated",
                  "memory_analysis_complete", "search_complete", "peak_rss_mib",
                  "elapsed_seconds", "verification_elapsed_seconds")


def now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def runtime_identity():
    from sentryward.frontend.ptx_native import native_executable
    binary = Path(native_executable())
    return {"python": platform.python_version(), "platform": platform.system(),
            "os_release": Path("/etc/os-release").read_text() if Path("/etc/os-release").exists() else platform.mac_ver()[0],
            "machine": platform.machine(), "native_sha256": file_hash(binary),
            "packages": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())),
            "system_packages": subprocess.check_output(["dpkg-query", "-W", "-f=${Package}=${Version}\\n"], text=True)
                if shutil.which("dpkg-query") else None}


def tracked_files(root):
    result = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    return [p.decode() for p in result.split(b"\0") if p]


def fingerprint(root, paths):
    return digest({p: file_hash(confined(root, p)) for p in sorted(paths)})


def plan(root, runtime, *, run_id, source_repository, config, baseline=None,
         names=None, force=False, evidence_base_url="https://evidence.invalid/sentryward"):
    root = Path(root).resolve()
    sha = revision(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip())
    paths = set(tracked_files(root))
    # Post-processing changes do not invalidate expensive verification.
    verifier_paths = [p for p in paths if (p.startswith(VERIFIER_PATHS) or p == "Makefile") and
                      p not in {"config/automation.json", "tools/automation/reporting.py",
                                "tools/automation/publication.py", "tools/automation/storage.py"}]
    verifier_key = digest({"source": fingerprint(root, verifier_paths),
                           "runtime": runtime,
                           "options": {k: v for k, v in config["verification"].items() if k != "shards"}})
    report_paths = [p for p in paths if p.startswith("skills/generate-kernel-report/") or
                    p == "tools/automation/reporting.py"]
    report_version = fingerprint(root, report_paths)
    existing = {r["name"]: r for r in (baseline or {}).get("results", [])}
    historical = read_json(root / "examples/results.json") if (root / "examples/results.json").exists() else {}
    saved = {r["name"]: r for r in historical.get("results", [])}
    cases = sorted(Path(p).parent.name for p in paths if len(Path(p).parts) == 3
                   and p.startswith("examples/") and p.endswith("/input.json"))
    if names and set(names) - set(cases):
        raise ValueError("Unknown examples: " + ", ".join(sorted(set(names) - set(cases))))
    tasks = []
    for index, name in enumerate(cases):
        identifier(name)
        files = {filename: f"examples/{name}/{filename}" for filename in CASE_FILES
                 if f"examples/{name}/{filename}" in paths}
        files["kernel.ptx"] = f"corpus/{name}.ptx"
        if files["kernel.ptx"] not in paths:
            raise ValueError("Missing committed PTX: " + files["kernel.ptx"])
        hashes = {k: file_hash(confined(root, v)) for k, v in files.items()}
        key = digest({"verifier": verifier_key, "files": hashes})
        previous = existing.get(name)
        history = previous or saved.get(name)
        if previous and previous.get("automation", {}).get("state") not in {"current", "reused"}:
            history = previous.get("historical") or saved.get(name)
        action, reason = "verify", None
        previous_auto = (previous or {}).get("automation", {})
        force_case = force and (not names or name in names)
        if (not force_case and previous_auto.get("verification_key") == key and
                previous_auto.get("state") in {"current", "reused"} and
                previous.get("result") != "tool_failure" and
                previous.get("bundle_files", {}).get("output.json")):
            action = "reuse"
        peak = (previous or {}).get("peak_rss_mib")
        if peak is None:
            output = root / "examples" / name / "output.json"
            if output.is_file():
                peak = read_json(output).get("peak_rss_mib")
        if action == "verify" and isinstance(peak, (int, float)) and peak > config["verification"]["memory_limit_mib"]:
            action, reason = "deferred_resource", "historical_memory_exceeds_runner_budget"
        if names and name not in names and action != "reuse":
            action, reason = "pending", "outside_requested_selection"
        tasks.append({"name": name, "shard": index % config["verification"]["shards"],
                      "action": action, "reason": reason, "files": files, "hashes": hashes,
                      "verification_key": key, "report_version": report_version,
                      "previous": previous, "historical": history,
                      "historical_peak_rss_mib": peak})
    return {"schema": "sentryward.plan.v1", "run_id": identifier(run_id), "source_revision": sha,
            "source_repository": repository(source_repository), "created_at": now(),
            "runtime": runtime, "config": config, "evidence_base_url": evidence_base_url,
            "verifier_key": verifier_key, "tasks": tasks,
            "baseline_revision": (baseline or {}).get("_results_revision"),
            "historical_study": historical.get("synchronization_removal_study")}


def copy_inputs(root, task, case):
    case.mkdir(parents=True, exist_ok=True)
    for filename, relative in task["files"].items():
        source = confined(root, relative)
        if file_hash(source) != task["hashes"][filename]:
            raise ValueError("Source changed after task planning: " + relative)
        shutil.copyfile(source, case / filename)


def run_process(command, case, timeout):
    """One process group per kernel; the workflow also bounds the whole container."""
    with (case / "stdout.json").open("wb") as out, (case / "stderr.log").open("wb") as err:
        proc = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
        try:
            return proc.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return proc.returncode, True


def raw_failure(code, message):
    return {"result": code, "checks": {"synchronization": "not_run", "memory": "not_run", "equivalence": "not_run"},
            "diagnostics": [{"code": code, "stage": "automation_worker", "message": message}]}


def artifact_descriptors(plan_data, task, case):
    files, bundle = {}, {}
    kinds = {"input.json": "input", "kernel.ptx": "ptx", "output.json": "result",
             "reference.py": "reference", "reference_binding.json": "reference_binding"}
    for filename, kind in kinds.items():
        path = case / filename
        if not path.exists():
            continue
        sha = file_hash(path)
        url = object_url(plan_data["evidence_base_url"], sha, filename)
        bundle[filename] = {"sha256": sha, "uri": url}
        descriptor = {"path": task["files"].get(filename, f"examples/{task['name']}/{filename}"),
                      "sha256": sha}
        if filename in task["files"]:
            descriptor["revision"] = plan_data["source_revision"]
        else:
            descriptor.update(url=url, raw_url=url)
        files[kind] = descriptor
    return files, bundle


def verify_task(root, plan_data, task, output, *, process_runner=run_process):
    root, output = Path(root), Path(output)
    case = output / "cases" / identifier(task["name"])
    copy_inputs(root, task, case)
    auto = {"state": task["action"], "verification_key": task["verification_key"],
            "report_version": task["report_version"], "attempted_revision": plan_data["source_revision"],
            "report_status": "pending", "reason": task["reason"]}
    record = {"name": task["name"], "result": None, "automation": auto,
              "files": {}, "provenance": {"source_revision": plan_data["source_revision"],
              "runtime": plan_data["runtime"], "run_id": plan_data["run_id"]}}
    history = task.get("historical")
    if history and history.get("synchronization_removal"):
        record["synchronization_removal"] = history["synchronization_removal"]
    record["files"], record["bundle_files"] = artifact_descriptors(plan_data, task, case)
    if task["action"] in {"pending", "deferred_resource"}:
        if history:
            record["historical"] = {k: v for k, v in history.items()
                                    if k not in {"historical", "bundle_files", "synchronization_removal"}}
            record["result"] = history.get("result")
        auto["report_status"] = "not_run"
    else:
        started = time.monotonic()
        if task["action"] == "reuse":
            previous = task["previous"]
            record["verification_origin"] = previous.get("verification_origin") or {
                "run_id": previous.get("provenance", {}).get("run_id"), "evidence": previous.get("evidence")}
            # Source inputs come from this checkout; only the raw result is reused.
            entry = previous["bundle_files"]["output.json"]
            cached = case / "output.json"
            if not cached.exists() or file_hash(cached) != entry["sha256"]:
                download_verified(entry["uri"], cached, entry["sha256"], base=plan_data["evidence_base_url"])
            auto["state"] = "reused"
        else:
            options = plan_data["config"]["verification"]
            command = [sys.executable, "-m", "sentryward", str(case / "kernel.ptx"),
                       "--input", str(case / "input.json"), "--timeout", str(options["timeout_seconds"]),
                       "--memory-limit-mib", str(options["memory_limit_mib"])]
            # Missing reference remains an explicit missing-input result for equivalence.
            command += ["--reference", str(case / "reference.py"), "--reference-binding", str(case / "reference_binding.json")]
            returncode, expired = process_runner(command, case, options["timeout_seconds"] + 15)
            if expired:
                write_json(case / "output.json", raw_failure("timeout", "Worker process-group deadline exceeded"))
            else:
                try:
                    raw = read_json(case / "stdout.json")
                    if not isinstance(raw, dict) or not isinstance(raw.get("result"), str) or not raw["result"].strip():
                        raise ValueError("Verifier did not produce a result object")
                except (ValueError, OSError) as error:
                    raw = raw_failure("tool_failure", f"Invalid verifier output (exit {returncode}): {error}")
                # Preserve the complete output, including unknown fields and result categories.
                write_json(case / "output.json", raw)
            auto.update(state="current", exit_code=returncode)
        raw = read_json(case / "output.json")
        for filename, sha in task["hashes"].items():
            if file_hash(case / filename) != sha:
                raise ValueError("Frozen input changed during verification: " + filename)
        record.update({k: raw[k] for k in SUMMARY_FIELDS if k in raw})
        if raw.get("result") == "tool_failure":
            auto["state"] = "failed"
            if history:
                record["historical"] = {k: v for k, v in history.items()
                                        if k not in {"historical", "bundle_files"}}
        record["analysis_time_seconds"] = raw.get("verification_elapsed_seconds", raw.get("elapsed_seconds"))
        record["worker_elapsed_seconds"] = time.monotonic() - started
        record["files"], record["bundle_files"] = artifact_descriptors(plan_data, task, case)
        write_json(case / "run.json", {"source_revision": plan_data["source_revision"],
                   "runtime": plan_data["runtime"], "options": plan_data["config"]["verification"],
                   "verification_key": task["verification_key"], "automation": auto,
                   "input_hashes": task["hashes"], "source_repository": plan_data["source_repository"]})
    write_json(case / "record.json", record)
    return record


def identity_manifest(plan_data, task, case):
    files = {}
    for filename in (*CASE_FILES, "kernel.ptx", "output.json"):
        path = case / filename
        if not path.exists():
            continue
        sha = file_hash(path)
        uri = object_url(plan_data["evidence_base_url"], sha, filename)
        raw_uri = uri
        if filename in task["files"]:
            relative = task["files"][filename]
            uri = f"https://github.com/{plan_data['source_repository']}/blob/{plan_data['source_revision']}/{relative}"
            raw_uri = f"https://raw.githubusercontent.com/{plan_data['source_repository']}/{plan_data['source_revision']}/{relative}"
        files[filename] = {"sha256": sha, "uri": uri, "raw_uri": raw_uri}
    value = {"files": files, "evidence_base_url": plan_data["evidence_base_url"]}
    write_json(case / "identity.json", value)
    return case / "identity.json"


def collect_records(plan_data, output):
    errors = {}
    for receipt in sorted(Path(output).glob("shard-*-errors.json")):
        for error in read_json(receipt):
            errors.setdefault(error["name"], []).append(error["error"])
    rows = []
    for task in plan_data["tasks"]:
        record = Path(output) / "cases" / task["name"] / "record.json"
        if record.exists():
            row = read_json(record)
            if row["name"] != task["name"] or row["automation"]["verification_key"] != task["verification_key"]:
                raise ValueError("Worker record does not match its planned task")
            if row["automation"].get("attempted_revision") != plan_data["source_revision"]:
                raise ValueError("Worker record belongs to another source revision")
            if row["automation"]["state"] not in {"current", "reused", "pending", "deferred_resource", "failed"}:
                raise ValueError("Unknown worker execution state")
            case = record.parent
            if row.get("provenance", {}).get("source_revision") != plan_data["source_revision"] or row["provenance"].get("runtime") != plan_data["runtime"]:
                raise ValueError("Worker provenance differs from planned runtime")
            if row["provenance"].get("run_id") != plan_data["run_id"]:
                raise ValueError("Worker provenance belongs to another run")
            for filename, expected in task["hashes"].items():
                if file_hash(confined(case, filename)) != expected:
                    raise ValueError("Worker input identity mismatch")
            if row["automation"]["state"] in {"current", "reused", "failed"}:
                raw = read_json(case / "output.json")
                for key in SUMMARY_FIELDS:
                    if (key in raw) != (key in row) or key in raw and raw[key] != row[key]:
                        raise ValueError("Worker summary differs from raw result: " + key)
            files, bundle = artifact_descriptors(plan_data, task, case)
            if row.get("files", {}) != files:
                raise ValueError("Worker file references differ from actual evidence")
            if any(row.get("bundle_files", {}).get(k) != v for k, v in bundle.items()):
                raise ValueError("Worker bundle hashes differ from actual evidence")
        else:
            row = {"name": task["name"], "result": None, "diagnostics": [], "files": {},
                   "automation": {"state": "failed", "reason": "worker_record_missing",
                                  "attempted_revision": plan_data["source_revision"], "report_status": "not_run"}}
            if errors.get(task["name"]):
                row["automation"]["execution_errors"] = errors[task["name"]]
            if task.get("historical"):
                row["historical"] = {k: v for k, v in task["historical"].items()
                                     if k not in {"historical", "bundle_files"}}
        rows.append(row)
    return rows


def catalog(plan_data, rows):
    counts = Counter(r["result"] for r in rows if r["automation"]["state"] in {"current", "reused"} and r["result"] is not None)
    result = {"schema": "sentryward.kernel-results.v1", "publication": {
        "run_id": plan_data["run_id"], "source_revision": plan_data["source_revision"],
        "source_repository": plan_data["source_repository"], "created_at": plan_data["created_at"],
        "evidence_base_url": plan_data["evidence_base_url"]}, "counts": dict(counts), "results": rows}
    if plan_data.get("historical_study"):
        result["synchronization_removal_study"] = plan_data["historical_study"]
    return result
