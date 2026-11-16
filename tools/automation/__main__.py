"""Stage CLI: python -m tools.automation --help."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from .common import read_json, write_json, https_base
from .pipeline import plan, runtime_identity, verify_task, identity_manifest, collect_records, catalog


def report_tasks(root, plan_data, work):
    from .reporting import generate_report
    rows = collect_records(plan_data, work)
    for task, record in zip(plan_data["tasks"], rows):
        case = Path(work) / "cases" / task["name"]
        if not (case / "record.json").exists() or not (case / "output.json").exists():
            continue
        identity = identity_manifest(plan_data, task, case)
        try:
            generate_report(Path(root), case, identity)
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            record["automation"].update(report_status="failed", report_error=str(error))
        else:
            record["automation"]["report_status"] = "base"
            record["automation"].pop("report_error", None)
        write_json(case / "record.json", record)
    write_json(Path(work) / "preview-results.json", catalog(plan_data, rows))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", type=Path, default=Path.cwd())
    commands = result.add_subparsers(dest="stage", required=True)
    runtime = commands.add_parser("runtime", help="Record the built verifier and dependency identity")
    runtime.add_argument("--out", type=Path, required=True)
    baseline = commands.add_parser("baseline", help="Read and validate the currently published snapshot")
    baseline.add_argument("--repository", default="gpu-verifier/sentryward")
    baseline.add_argument("--out", type=Path, required=True)
    planner = commands.add_parser("plan")
    planner.add_argument("--runtime", type=Path, required=True)
    planner.add_argument("--config", type=Path, default=Path("config/automation.json"))
    planner.add_argument("--baseline", type=Path)
    planner.add_argument("--run-id", required=True)
    planner.add_argument("--repository", default="gpu-verifier/sentryward")
    planner.add_argument("--evidence-url", default=os.environ.get("SENTRYWARD_EVIDENCE_URL", "https://evidence.invalid/sentryward"))
    planner.add_argument("--name", action="append")
    planner.add_argument("--force", action="store_true")
    planner.add_argument("--out", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--plan", type=Path, required=True)
    verify.add_argument("--work", type=Path, required=True)
    verify.add_argument("--shard", type=int, required=True)
    verify.add_argument("--name", help="Execute one task, for per-kernel containers")
    report = commands.add_parser("report")
    report.add_argument("--plan", type=Path, required=True)
    report.add_argument("--work", type=Path, required=True)
    publish = commands.add_parser("publish")
    publish.add_argument("--plan", type=Path, required=True)
    publish.add_argument("--work", type=Path, required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    args.root = args.root.resolve()
    if args.stage == "runtime":
        write_json(args.out, runtime_identity())
    elif args.stage == "baseline":
        from .publication import fetch_baseline
        fetch_baseline(args.repository, args.out)
    elif args.stage == "plan":
        if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=args.root):
            raise ValueError("Plan from a clean committed checkout so Git links match the evidence")
        config = read_json(args.config)
        if config.get("schema") != "sentryward.automation.v1":
            raise ValueError("Unsupported automation configuration")
        for key in ("timeout_seconds", "memory_limit_mib", "shards"):
            if type(config["verification"].get(key)) is not int or config["verification"][key] <= 0:
                raise ValueError("Invalid verification budget: " + key)
        write_json(args.out, plan(args.root, read_json(args.runtime), run_id=args.run_id,
                   source_repository=args.repository, config=config,
                   baseline=read_json(args.baseline) if args.baseline else None,
                   names=args.name, force=args.force, evidence_base_url=https_base(args.evidence_url)))
    elif args.stage == "verify":
        data = read_json(args.plan)
        if runtime_identity() != data["runtime"]:
            raise ValueError("Worker runtime differs from the planned build")
        if not 0 <= args.shard < data["config"]["verification"]["shards"]:
            raise ValueError("Invalid shard index")
        failures = []
        for task in data["tasks"]:
            if task["shard"] != args.shard or args.name and task["name"] != args.name:
                continue
            try:
                record = verify_task(args.root, data, task, args.work)
                print(task["name"], record["automation"]["state"], record["result"], flush=True)
            except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
                failures.append({"name": task["name"], "error": str(error)})
        write_json(args.work / f"shard-{args.shard}-errors.json", failures)
        return bool(failures)
    elif args.stage == "report":
        data = read_json(args.plan)
        report_tasks(args.root, data, args.work.resolve())
    elif args.stage == "publish":
        from .storage import S3Store
        from .publication import GitHub, publish
        store = S3Store.from_env()
        data = read_json(args.plan)
        if store.public_base != data["evidence_base_url"]:
            raise ValueError("Storage and planned evidence URL differ")
        receipt = publish(data, args.work, store, GitHub(data["source_repository"], os.environ["GITHUB_TOKEN"]))
        write_json(args.work / "publication.json", receipt)
        print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
