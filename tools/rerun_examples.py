"""Rerun saved examples from PTX and refresh their raw results and index."""
from __future__ import annotations


from sentryward.results import ResultArgumentParser, normalize, failure, RESULTS, diagnostic_code, run_cli, REPORT_FIELDS

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path
import shutil
import subprocess
import time

import ijson

from sentryward.frontend.ptx_native import ROOT, native_executable


TIMEOUTS = {"timeout", "native_timeout", "runner_timeout", "wellsync_timeout"}
SUMMARY_FIELDS = {"result", "status", "verification_status", "mode", "stage", "warning_count", "warnings_complete", "search_complete",
                  "memory_analysis_complete", "completed_memory_check", "unresolved_count",
                  "reason", "timeout_scope", "active_stage", "failure_stage",
                  "value_view_complete", "value_ir_node_count", "value_ir_instance_count",
                  "message", "peak_rss_mib", "rss_limit_mib", "verification_result", "verification_elapsed_seconds"}
VALUE_SUMMARY_FIELDS = {"value_analysis", "value_view_complete", "value_ir_node_count",
                        "value_ir_instance_count", "semantic_program"}
NESTED_SUMMARY_FIELDS = {
    "value_analysis": {"result", "status", "reason", "stage", "line", "prerequisite_status",
                       "message", "instruction_index",
                       "processed_memory_actions", "read_substitutions",
                       "initial_memory_regions", "output_regions",
                       "equivalence_check", "equivalence_reason"},
    "semantic_program": {"thread_prefix_groups", "execution_cohorts",
                         "synchronization_groups", "synthetic_synchronization_nodes",
                         "memory_bindings", "mapping_complete",
                         "value_diagnostic_count", "binding_diagnostic_count", "unsupported_occurrence_counts", "diagnostic_sample_count"},
    "rejection": {"code"},
    "error": {"code"},
}


def read_report_summary(path: Path) -> dict:
    """Validate a complete report without retaining its potentially huge arrays.

    ijson owns JSON parsing, including truncation/trailing-data checks. Only root
    summary fields, compact value-analysis fields, memory-analysis presence, and
    a warning sample count survive the event stream.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    result, keys = {}, set()
    depth, key, warning_samples = 0, None, 0
    nested_key = None
    started = finished = False
    with opener(path, "rb") as stream:
        for event, value in ijson.basic_parse(stream):
            if depth == 0:
                if not started and event == "start_map":
                    started, depth = True, 1
                    continue
                raise ValueError("native report is not one status object")
            if depth == 1 and event == "map_key":
                if value in keys:
                    raise ValueError("duplicate top-level report field: " + value)
                keys.add(value)
                key = value
                nested_key = None
            elif depth == 1 and event not in {"start_map", "start_array", "end_map"}:
                if key in SUMMARY_FIELDS:
                    result[key] = float(value) if key in {"peak_rss_mib", "verification_elapsed_seconds"} and event == "number" else value
            if depth == 1 and key == "memory_analysis" and event == "start_map":
                result["memory_analysis"] = {}  # Presence only, not the object.
            if key in NESTED_SUMMARY_FIELDS:
                if depth == 1 and event == "start_map":
                    result[key] = {}
                elif depth == 2 and event == "map_key":
                    nested_key = value
                elif depth == 2 and event in {"string", "number", "boolean", "null"}:
                    if nested_key in NESTED_SUMMARY_FIELDS[key]:
                        result[key][nested_key] = value
            if depth == 2 and key == "warnings" and event in {
                    "start_map", "start_array", "string", "number", "boolean", "null"}:
                warning_samples += 1
            if event in {"start_map", "start_array"}:
                depth += 1
            elif event in {"end_map", "end_array"}:
                depth -= 1
                finished = depth == 0
    if not finished or not isinstance(result.get("result", result.get("status")), str):
        raise ValueError("native report is not a status object")
    for field in ("warning_count", "unresolved_count", "value_ir_node_count", "value_ir_instance_count"):
        if field in result and (type(result[field]) is not int or result[field] < 0):
            raise ValueError(field + " must be a nonnegative integer")
    for field in ("warnings_complete", "search_complete", "memory_analysis_complete", "completed_memory_check", "value_view_complete"):
        if field in result and type(result[field]) is not bool:
            raise ValueError(field + " must be Boolean")
    if "warnings" in keys:
        result["warning_sample_count"] = warning_samples
    if "equivalence" in keys:
        with opener(path, "rb") as stream:
            result["equivalence"] = next(ijson.items(stream, "equivalence", use_float=True))
    for field in ("checks","scope","resource","error","rejection","diagnostic"):
        if field in keys:
            with opener(path, "rb") as stream:
                result[field] = next(ijson.items(stream, field, use_float=True))
    # Retain complete diagnostics at every report depth, including arbitrary
    # nested evidence. Do not retain unrelated expression/warning arrays.
    with opener(path, "rb") as stream:
        builder = None
        capture_prefix = None
        balance = 0
        for prefix, event, value in ijson.parse(stream, use_float=True):
            if builder is None and (prefix == "diagnostics" or prefix.endswith(".diagnostics") or prefix.endswith("_diagnostics")) and event == "start_array" and ".item." not in prefix:
                builder = ijson.common.ObjectBuilder()
                capture_prefix = prefix
                balance = 0
            if builder is not None:
                builder.event(event, value)
                if event in {"start_map","start_array"}: balance += 1
                elif event in {"end_map","end_array"}: balance -= 1
                if balance == 0:
                    target = result
                    parts = capture_prefix.split(".")
                    for part in parts[:-1]: target = target.setdefault(part, {})
                    target[parts[-1]] = builder.value
                    builder = None
    result.setdefault("diagnostics", [])
    return normalize(result)


def verification_status(report: dict) -> str:
    """A later numerical deadline must not erase completed verification.

    Only a native checkpoint with completed memory analysis can separate these
    outcomes. Transport failures must never inherit a successful partial report.
    """
    checks = report.get("checks", {})
    if checks.get("synchronization") == "passed" and checks.get("memory") == "passed":
        return "race_free"
    return report.get("verification_result", report["result"])


def read_result(case: Path) -> dict:
    for name in ("output.json", "output.json.gz"):
        path = case / name
        if path.exists():
            try:
                return read_report_summary(path)
            except (OSError, EOFError, ValueError, ijson.JSONError) as error:
                return {"result": "tool_failure", "message": str(error),
                        "memory_analysis_complete": False, "warnings_complete": False}
    return {}


def _stderr_tail(path: Path) -> str:
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - 8192))
        return stream.read(8192).decode("utf-8", errors="replace")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _native_command(case: Path, ptx: Path, destination: Path, *, timeout: int,
                wellsync_timeout: int, rss_limit: int | None,
                analyze_values: bool = False, scalar_fold: bool = False,
                reference: bytes | None = None, mapping: Path | None = None,
                reference_provenance: Path | None = None,
                export_expressions: bool = False,
                equivalence_timeout: int = 100, require_equivalence: bool = False,
                reference_failure: dict | None = None) -> list[str]:
    """Keep native bytes on disk; publish a small wrapper only on transport failure."""
    raw = destination / "native-output.json"
    stdout_path, stderr_path = destination / "stdout.log", destination / "stderr.log"
    command = [str(native_executable()), "--ptx", str(ptx),
               "--config", str(destination / "input.json"),
               "--wellsync-timeout-seconds", format(wellsync_timeout, "g"),
               "--sync-threads", "8",
               "--timeout-seconds", format(timeout, "g"), "--report", str(raw)]
    if rss_limit is not None:
        command += ["--rss-limit-mib", str(rss_limit)]
    if analyze_values:
        command.append("--analyze-values")
    if scalar_fold:
        command += ["--analyze-scalar-fold"]
        if export_expressions:
            command += ["--dump-scalar-fold", str(destination / "expression.json")]
    if require_equivalence:
        command += ["--require-equivalence"]
    if reference_failure is not None:
        command += ["--reference-failure-json", json.dumps(normalize(reference_failure, operation="equivalence"))]
    if reference is not None:
        command += ["--reference-stdin", "--equivalence-mapping", str(mapping),
                    "--equivalence-timeout-seconds", str(equivalence_timeout)]
        if reference_provenance is not None:
            command += ["--reference-provenance", str(reference_provenance)]
    return command


def _run_native(case: Path, ptx: Path, destination: Path, **options):
    command = _native_command(case, ptx, destination, **options)
    return _execute_native(command, case, destination,
                           options.get("reference"), options["wellsync_timeout"])


def _execute_native(command, case, destination, reference, wellsync_timeout):
    """Shared process transport: retain full output and report failures faithfully."""
    raw = destination / "native-output.json"
    stdout_path, stderr_path = destination / "stdout.log", destination / "stderr.log"
    started = time.monotonic()
    interrupted, launch_error, returncode = False, None, None
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        try:
            returncode = subprocess.run(command, input=reference, stdout=stdout, stderr=stderr, check=False).returncode
        except KeyboardInterrupt:
            interrupted, returncode = True, -2
        except OSError as error:
            launch_error = str(error)
    process = {"command": command, "returncode": returncode,
               "elapsed_seconds": time.monotonic() - started}
    error_output = _stderr_tail(stderr_path)
    result, report_error = None, None
    if raw.exists() and not interrupted and launch_error is None:
        try:
            result = read_report_summary(raw)
        except (OSError, EOFError, ValueError, ijson.JSONError) as error:
            report_error = str(error)
    failure, message = None, None
    if interrupted:
        failure, message = "tool_failure", "Native run interrupted by user"
    elif launch_error:
        failure, message = "tool_failure", launch_error
    elif result is None:
        if returncode == 124 and "SENTRYWARD_TIMEOUT wellsync" in error_output.splitlines():
            failure = "timeout"
            message = f"WellSync exceeded its {wellsync_timeout:g}-second budget"
        elif report_error is not None:
            failure, message = "tool_failure", report_error
        else:
            failure = "tool_failure" if returncode else "tool_failure"
            message = error_output or "Native process produced no JSON report"
    elif returncode:
        # These exits are part of the native status protocol, not transport errors.
        expected = {1: {"invalid_input", "invalid_input", "missing_input", "tool_failure"}, 2: {"tool_failure", "allocation_failed", "unsupported"},
                    124: TIMEOUTS, 137: {"memory_budget_exceeded"}}
        if result["result"] not in expected.get(returncode, set()):
            failure, message = "tool_failure", "Process failed after emitting a report"
    if failure is not None:
        wrapper = {"schema": "spverify.native-ptx-run/v1", "implementation": "cpp",
                   "result": failure, "message": message, "native_exit_code": returncode,
                   "memory_analysis_complete": False, "warnings_complete": False}
        if raw.exists():
            wrapper["raw_native_report"] = os.path.relpath(raw, case)
        if result is not None:
            wrapper["native_status"] = result["result"]
            wrapper["native_result"] = result
        if failure == "timeout":
            wrapper["timeout_scope"] = "wellsync"
        published = destination / "new-output.json"
        wrapper = normalize(wrapper)
        published.write_text(json.dumps(wrapper, indent=2) + "\n")
        return wrapper, process, published
    return result, process, raw


def publish_report(source: Path, destination: Path, root: Path) -> None:
    """Copy a report with repository-relative paths and bounded buffering."""
    prefix = json.dumps(str(root.resolve()) + "/")[1:-1].encode()
    tail = b""
    with source.open("rb") as incoming, destination.open("wb") as outgoing:
        for chunk in iter(lambda: incoming.read(1024 * 1024), b""):
            data = tail + chunk
            start = 0
            while (position := data.find(prefix, start)) >= 0:
                outgoing.write(data[start:position])
                start = position + len(prefix)
            remaining = data[start:]
            keep = min(len(remaining), len(prefix) - 1)
            outgoing.write(remaining[:len(remaining) - keep])
            tail = remaining[len(remaining) - keep:]
        outgoing.write(tail)


def run_case(case: Path, corpus: Path, archive: Path, *, include_timeouts=False,
             timeout=300, wellsync_timeout=120, rss_limit=None,
             analyze_values: bool = False, check_equivalence: bool = True,
             equivalence_timeout: int = 1200,
             export_expressions: bool = False) -> dict:
    previous = read_result(case)
    if not include_timeouts and previous.get("result") in TIMEOUTS:
        return {"name": case.name, "result": verification_status(previous),
                "pipeline_status": previous["result"], "rerun": False,
                "skip_reason": "previous_timeout"}
    destination = archive / case.name
    destination.mkdir()
    for name in ("input.json", "output.json", "output.json.gz", "report.md", "equivalence.json"):
        if (case / name).exists():
            shutil.copyfile(case / name, destination / name)
    original_input = (case / "input.json").read_bytes()
    ptx = corpus / (case.name + ".ptx")
    ptx_sha256 = _file_sha256(ptx)
    reference, mapping, reference_failure, reference_provenance = None, None, None, None
    capture_started = time.monotonic()
    if check_equivalence:
        from tools._example_files import reference_files
        source, config, mapping = reference_files(ROOT, case.name)
        missing = [str(p.relative_to(ROOT)) for p in (source, config, mapping) if not p.is_file()]
        if missing:
            reference_failure = dict(result="missing_input", reason="missing_reference_config_or_mapping", missing=missing)
        else:
            shutil.copyfile(mapping, destination / "reference_binding.json")
            mapping = destination / "reference_binding.json"
            with (destination / "reference-capture.log").open("wb") as log:
                try:
                    captured = subprocess.run([sys.executable, "-m", "sentryward._capture_reference",
                        str(source), str(config), str(mapping)] + (["--rss-limit-mib", str(rss_limit)] if rss_limit is not None else []),
                        stdout=subprocess.PIPE, stderr=log, timeout=timeout, check=False)
                    if captured.returncode:
                        try:
                            reference_failure = json.loads(captured.stdout)["capture_failure"]
                        except (ValueError, KeyError, TypeError):
                            reference_failure = failure("tool_failure", "reference_capture_failed",
                                "Reference worker failed", stage="reference_capture",
                                details={"exit_code": captured.returncode})
                    else:
                        reference = captured.stdout
                        if export_expressions:
                            (destination / "reference-expression.json").write_bytes(reference)
                except subprocess.TimeoutExpired:
                    reference_failure = dict(result="timeout", reason="reference_capture_timeout")
    capture_seconds = time.monotonic() - capture_started
    remaining = max(0.001, timeout - capture_seconds)
    result, process, report = _run_native(case, ptx, destination, timeout=remaining,
                                        wellsync_timeout=wellsync_timeout, rss_limit=rss_limit,
                                        analyze_values=analyze_values or check_equivalence,
                                        scalar_fold=check_equivalence,
                                        reference=reference, mapping=mapping,
                                        reference_provenance=reference_provenance,
                                        export_expressions=export_expressions,
                                        equivalence_timeout=equivalence_timeout,
                                        require_equivalence=check_equivalence, reference_failure=reference_failure)
    assert (case / "input.json").read_bytes() == original_input, "Input changed during run"
    temporary = case / "output.json.tmp"
    publish_report(report, temporary, ROOT)
    temporary.replace(case / "output.json")
    if (case / "output.json.gz").exists():
        (case / "output.json.gz").unlink()  # Archived above.
    record = {"name": case.name, "result": result["result"], "verification_result": verification_status(result), "rerun": True,
              "diagnostics": result.get("diagnostics", []), "checks": result.get("checks", {}),
              "scope": result.get("scope", {}),
              "pipeline_status": result["result"],
              "previous_status": verification_status(previous) if previous else None,
              "input_sha256": hashlib.sha256(original_input).hexdigest(),
              "ptx_sha256": ptx_sha256,
              "warning_count": result.get("warning_count", result.get("warning_sample_count", 0)),
              "warnings_complete": result.get("warnings_complete", "warning_count" in result),
              "memory_analysis_entered": "memory_analysis" in result,
              "memory_search_complete": result.get("search_complete", False),
              "memory_analysis_complete": result.get("memory_analysis_complete",
                  "memory_analysis" in result and result.get("search_complete", False)
                  and result.get("unresolved_count", 0) == 0),
              "elapsed_seconds": process.get("elapsed_seconds", 0) + capture_seconds,
              "reference_capture_seconds": capture_seconds,
              "native_peak_rss_mib": result.get("peak_rss_mib"),
              "rss_limit_mib": rss_limit,
              "exit_code": process.get("returncode")}
    if "verification_elapsed_seconds" in result:
        record["verification_elapsed_seconds"] = result["verification_elapsed_seconds"]
    supplied_choices = json.loads(original_input).get("branch_decisions", [])
    if supplied_choices:
        record["analysis_scope"] = {"kind": "configured_branch_scenario",
                                    "branch_decisions": supplied_choices}
    for field in REPORT_FIELDS:
        if field != "equivalence" and field in result: record[field] = result[field]
    if analyze_values or check_equivalence:
        record.update((field, result[field]) for field in VALUE_SUMMARY_FIELDS if field in result)
    if check_equivalence:
        correctness = result.get("equivalence") or (reference_failure if record["verification_result"] == "race_free" else None) or dict(
            result="timeout" if result["result"] in TIMEOUTS else
                   result["result"] if result["result"] in {"memory_budget_exceeded","analysis_budget_exceeded","allocation_failed","tool_failure","invalid_input","missing_input"} else "unsupported",
            reason="prerequisite_" + result["result"], stage=result.get("active_stage"))
        correctness = normalize(correctness, operation="equivalence")
        if not result.get("equivalence"):
            correctness["checks"]["equivalence"] = "not_run"
        if supplied_choices:
            correctness["analysis_scope"] = record["analysis_scope"]
        if not result.get("equivalence"):
            # A process failure may leave only a wrapper report. Keep its check
            # diagnostics in the same output, without loading large IR arrays.
            output = case / "output.json"
            temporary = case / "output.json.tmp"
            shutil.copyfile(output, temporary)
            with temporary.open("r+b") as stream:
                stream.seek(0, 2)
                end = stream.tell()
                while end:
                    end -= 1
                    stream.seek(end)
                    byte = stream.read(1)
                    if not byte.isspace():
                        break
                if byte != b"}":
                    raise ValueError("output report must be a JSON object")
                stream.seek(end)
                stream.truncate()
                stream.write(b', "equivalence": ' + json.dumps(correctness).encode() + b"}\n")
            temporary.replace(output)
        # Detailed per-family receipts live in output.json under equivalence.
        # Batch summaries must not repeatedly copy all those rows for every case.
        record["equivalence"] = {k: v for k, v in correctness.items()
                                 if k not in {"rows", "counters", "unmatched_reasons"}}
    else:
        record["equivalence"] = {"result": "unsupported", "reason": "verification_only"}
    if check_equivalence:
        record["checks"]["equivalence"] = record["equivalence"].get("checks", {}).get("equivalence", "not_run")
        if record["verification_result"] == "race_free" and record["result"] in {"race_free","verified"}:
            record["result"] = record["equivalence"]["result"]
            if record["result"] != "verified":
                record["diagnostics"] = result.get("diagnostics", []) + [
                    d for d in record["equivalence"]["diagnostics"] if d not in result.get("diagnostics", [])]
    if (case / "equivalence.json").exists():
        (case / "equivalence.json").unlink()  # Legacy result archived above.
    record["raw_report"] = str(destination / "native-output.json")
    record = normalize(record)
    (destination / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def write_index(examples: Path, records: list[dict]) -> None:
    """Refresh result columns without discarding issue/report metadata."""
    index = examples / "index.md"
    text = index.read_text() if index.exists() else "# Example Summary\n\n"
    lines = text.splitlines()
    header_at = next((i for i, line in enumerate(lines)
                      if line.startswith("| Example |")), None)
    header = ([c.strip() for c in lines[header_at].split("|")[1:-1]]
              if header_at is not None else ["Example", "PTX", "Verify Result", "Report"])
    if "Correctness Check" not in header:
        header.append("Correctness Check")
    previous = {}
    if header_at is not None:
        for line in lines[header_at+2:]:
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if cells:
                previous[cells[0]] = dict(zip(header, cells))
    output = lines[:header_at] if header_at is not None else lines
    output += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for record in records:
        name = record["name"]
        row = previous.get(name, {})
        row.update({"Example": name, "PTX": f"[PTX](../corpus/{name}.ptx)",
                    "Verify Result": f"`{record['result']}`"})
        row.setdefault("Report", f"[Report — previous run]({name}/report.md)"
                       if (examples / name / "report.md").exists() else "—")
        row.setdefault("Correctness Check", "not run")
        if record.get("rerun"):
            row["Report"] = (f"[Report — previous run]({name}/report.md)"
                             if (examples / name / "report.md").exists() else "—")
        check = record.get("equivalence")
        if check:
            row["Correctness Check"] = check["result"]
            if check.get("analysis_scope", {}).get("kind") == "configured_branch_scenario":
                row["Correctness Check"] += " (conditional branch scenario)"
            if diagnostic_code(check):
                row["Correctness Check"] += ": " + str(diagnostic_code(check)).replace("|", "/").replace("\n", " ")
            if check.get("definedness"):
                defined = check["definedness"]
                row["Correctness Check"] += " (total definedness: " + str(defined.get("status", "not_proven")) + ")"
                if defined.get("partial_equivalence_families", 0):
                    row["Correctness Check"] += " [same-domain partial-function equivalence]"
        output.append("| " + " | ".join(row.get(c, "—") for c in header) + " |")
    index.write_text("\n".join(output) + "\n")


def main(argv=None) -> int:
    parser = ResultArgumentParser(description=__doc__)
    parser.add_argument("--include-timeouts", action="store_true")
    parser.add_argument("--analyze-values", action="store_true",
                        help="Also extract symbolic values and run checked memory elimination")
    parser.add_argument("--verification-only", "--skip-correctness-check", action="store_true",
                        help="Diagnostic mode: omit the default race-free equivalence stage")
    parser.add_argument("--equivalence-timeout", type=int, default=1200)
    parser.add_argument("--status", action="append", help="Select by current saved verifier status")
    parser.add_argument("--name", action="append", help="Select exact example names")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--wellsync-timeout", type=int, default=120)
    parser.add_argument("--rss-limit", type=int, help="optional positive memory budget in MiB; default: unlimited")
    args = parser.parse_args(argv)
    if min(args.jobs, args.timeout, args.wellsync_timeout, args.equivalence_timeout) <= 0 or (args.rss_limit is not None and args.rss_limit <= 0):
        parser.error("limits and jobs must be positive")
    examples = ROOT / "examples"
    cases = sorted(path for path in examples.iterdir()
                   if path.is_dir() and path.name not in {"_reference", "__pycache__"})
    all_cases = cases
    if args.name:
        missing = set(args.name) - {p.name for p in cases}
        if missing:
            parser.error("Unknown examples: " + ", ".join(sorted(missing)))
        cases = [p for p in cases if p.name in args.name]
    if args.status:
        cases = [p for p in cases if verification_status(read_result(p) or {"result": "unsupported"}) in args.status]
    for case in cases:
        if not (case / "input.json").is_file() or not (ROOT / "corpus" / (case.name + ".ptx")).is_file():
            parser.error(f"Missing input or PTX: {case.name}")
    if args.dry_run:
        skipped = [p.name for p in cases if not args.include_timeouts and read_result(p).get("result") in TIMEOUTS]
        print(json.dumps({"run_count": len(cases)-len(skipped), "skipped_timeouts": skipped}, indent=2))
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = ROOT / "build/artifacts" / ("example-regression-" + stamp)
    archive.mkdir(parents=True)
    (archive / "selection.json").write_text(json.dumps({"cases": [p.name for p in cases],
        "status_filter": args.status, "timeout_seconds": args.timeout,
        "rss_limit_mib": args.rss_limit, "equivalence_timeout_seconds": args.equivalence_timeout}, indent=2)+"\n")
    records = []
    def run(case):
        return run_case(case, ROOT / "corpus", archive, include_timeouts=args.include_timeouts,
                        timeout=args.timeout, wellsync_timeout=args.wellsync_timeout, rss_limit=args.rss_limit,
                        analyze_values=args.analyze_values, check_equivalence=not args.verification_only,
                        equivalence_timeout=args.equivalence_timeout,
                        export_expressions=False)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for record in pool.map(run, cases):
            records.append(record)
            (archive / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
            print(record["name"], record["result"], "" if record["rerun"] else "(skipped)", flush=True)
    by_name = {r["name"]: r for r in records}
    write_index(examples, [by_name.get(p.name) or {"name": p.name,
        "result": verification_status(read_result(p) or {"result": "unsupported"}), "rerun": False} for p in all_cases])
    print(f"Summary: {archive / 'summary.json'}")
    return int(any(r["result"] in {"tool_failure", "tool_failure", "tool_failure"} for r in records))


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
