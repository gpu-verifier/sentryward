"""Run one shard in per-case containers; no reference code runs on the host."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess

from .common import digest, download_verified, identifier, read_json, write_json


def export_case(case, destination):
    """Copy a stopped worker's ordinary files; never let an uploader follow links."""
    if not stat.S_ISDIR(case.lstat().st_mode):
        raise ValueError("Worker artifacts must contain only regular files and directories: " + str(case))
    paths = list(case.rglob("*"))
    for path in [case, *paths]:
        mode = path.lstat().st_mode
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError("Worker artifacts must contain only regular files and directories: " + str(path))
    destination.mkdir(parents=True)
    try:
        for path in paths:
            if path.is_file():
                target = destination / path.relative_to(case)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
    except OSError:
        shutil.rmtree(destination)
        raise


def run_shard(plan_path, output, shard, *, export, image="sentryward-worker"):
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    export = Path(export).resolve()
    if export.is_relative_to(output) or output.is_relative_to(export):
        raise ValueError("Artifact export must be separate from the worker directory")
    export.mkdir(parents=True)
    data = read_json(plan_path)
    errors = []
    write_json(export / f"shard-{shard}-errors.json", errors)
    for task in data["tasks"]:
        if task["shard"] != shard:
            continue
        name = identifier(task["name"])
        case = output / "cases" / name
        case.mkdir(parents=True, exist_ok=True)
        container = "sentryward-" + digest([data["run_id"], shard, name])[:32]
        launched, stopped = False, False
        try:
            # Network access is limited to checked object downloads on the host.
            if task["action"] == "reuse":
                entry = task["previous"]["bundle_files"]["output.json"]
                download_verified(entry["uri"], case / "output.json", entry["sha256"], base=data["evidence_base_url"])
            command = ["docker", "run", "--name", container,
                       "--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                       "--memory=10g", "--memory-swap=10g", "--cpus=2", "--pids-limit=256",
                       "--user", f"{os.getuid()}:{os.getgid()}", "--env", "PYTHONDONTWRITEBYTECODE=1",
                       "--env", "HOME=/tmp", "--env", "USER=sentryward",
                       "--tmpfs", "/tmp:rw,nosuid,nodev,size=1g,mode=1777", "--tmpfs", "/out:rw,nosuid,nodev,size=16m,mode=1777",
                       "--mount", f"type=bind,src={plan_path},dst=/plan.json,readonly",
                       "--mount", f"type=bind,src={case},dst=/out/cases/{name}", image,
                       "python", "-m", "tools.automation", "verify", "--plan", "/plan.json",
                       "--work", "/out", "--shard", str(shard), "--name", name]
            launched = True
            result = subprocess.run(command, timeout=data["config"]["verification"]["timeout_seconds"] + 90)
            if result.returncode:
                errors.append({"name": name, "error": f"Worker container exited {result.returncode}"})
                inspected = subprocess.run(["docker", "inspect", "--format", "{{json .State}}", container],
                                           check=True, capture_output=True, text=True, timeout=30)
                if json.loads(inspected.stdout).get("OOMKilled") is True:
                    errors[-1]["error"] = ("Worker container exceeded the 10 GiB process-tree memory limit "
                                           f"(OOMKilled; exit {result.returncode})")
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            errors.append({"name": name, "error": str(error)})
        finally:
            if launched:
                try:
                    subprocess.run(["docker", "rm", "-f", container], check=True, timeout=30,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    stopped = True
                except (OSError, subprocess.SubprocessError) as error:
                    errors.append({"name": name, "error": "Container cleanup failed: " + str(error)})
        if stopped:
            try:
                export_case(case, export / "cases" / name)
            except (ValueError, OSError) as error:
                errors.append({"name": name, "error": "Artifact export rejected: " + str(error)})
        write_json(export / f"shard-{shard}-errors.json", errors)
    return bool(errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True, help="Fresh, separate directory safe for artifact upload")
    parser.add_argument("--shard", type=int, required=True)
    args = parser.parse_args()
    raise SystemExit(run_shard(args.plan, args.work, args.shard, export=args.export))
