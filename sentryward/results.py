"""Shared public result contract. Legacy names are accepted only on input."""
from __future__ import annotations
import json
from pathlib import Path

_package = Path(__file__).resolve().parent
_rules_path = _package / "_config/result-categories.json"
if not _rules_path.is_file():
    _rules_path = _package.parent / "config/result-categories.json"
RULES = json.loads(_rules_path.read_text())
RESULTS = frozenset(RULES["results"])
MAP = RULES["legacy"]
REPORT_FIELDS = ("equivalence", "value_analysis", "scalar_fold", "expression_export",
                 "partial_report", "native_result", "partial_native_result", "expression_diagnostic")
SUCCESS = {"race_free", "equivalent", "verified", "valid", "extraction_complete",
           "semantic_complete", "value_view_complete", "expressions_ready",
           "reference_expressions_ready", "ready", "complete"}

def category(status, reason="", resource=None):
    resource = resource or {}
    if reason in {"time_budget", "reference_capture_timeout"} or status in {"native_timeout","runner_timeout","wellsync_timeout"}:
        return "timeout"
    if reason == "rss_limit" or reason == "native_pipeline_memory_limit_exceeded":
        return "memory_budget_exceeded"
    if status == "representation_limit_exceeded" or reason == "representation_limit_exceeded":
        return "unsupported"
    if reason == "allocation_failed":
        return "allocation_failed"
    if reason in {"node_budget","iteration_budget","work_budget","alpha_normalization_context_budget",
                  "alpha_normalization_node_budget","projection_memo_budget"} or reason.endswith("_limit_exceeded"):
        return "analysis_budget_exceeded"
    if reason in {"inputs_changed_during_run","extraction_inputs_changed_during_run","unexpected_process_exit","invalid_native_result"} or reason.startswith("process_error:"):
        return "tool_failure"
    if status in RESULTS:
        return status
    if status == "resource_limit":
        return {"configured_budget": "memory_budget_exceeded", "algorithm_budget":"analysis_budget_exceeded",
                "representation":"unsupported", "allocator":"allocation_failed"}.get(resource.get("origin"), "tool_failure")
    return MAP.get(status, "tool_failure")

def _diagnostics(out, diagnostics, stage, result):
    """Preserve every diagnostic and its evidence; never deduplicate by code."""
    converted = []
    for original in diagnostics:
        d = dict(original)
        code = str(d.get("code", "diagnostic_code_missing"))
        d["stage"] = d.get("stage") or stage or "pipeline"
        evidence = d.get("details", {})
        details = dict(evidence) if isinstance(evidence,dict) else {"evidence":evidence}
        for key in list(d):
            if key not in {"code", "stage", "message", "details"}:
                details[key] = d.pop(key)
        code = RULES["diagnostic_codes"].get(code, code)
        if code == d["stage"] or code in {"memory_budget_exceeded","memory_limit_exceeded"} or code.startswith("semantic_ir.cta") or code.startswith("source_ir.cta") or code.startswith("value_ir.cta") or code == "verification.memory":
            if result == "memory_budget_exceeded":
                if code.startswith(("semantic_ir.cta","source_ir.cta","value_ir.cta")) or code=="verification.memory": d["stage"]=code
                code = "process_memory_budget_exceeded"
        if code in {"rss_limit", "native_pipeline_memory_limit_exceeded", "memory_limit_exceeded"}:
            code = "process_memory_budget_exceeded"
        if d["stage"].startswith(("semantic_ir.cta","source_ir.cta","value_ir.cta")):
            prefix, _, cta = d["stage"].partition(".cta")
            details.setdefault("cta_progress", int(cta) if cta.isdigit() else cta)
            d["stage"] = {"semantic_ir":"value_view_construction","source_ir":"event_view_construction","value_ir":"value_view_construction"}[prefix]
        elif d["stage"] == "verification.memory":
            d["stage"] = "memory_check"
        if code == "process_memory_budget_exceeded":
            for key in ("resource","peak_rss_mib","rss_limit_mib","memory_limit_mib"):
                if key in out: details.setdefault(key,out[key])
        if code in {"total_definedness_not_proven","input_access_validity_not_proven"}:
            for key in ("definedness","access_validity","equivalent_families"):
                if key in out: details.setdefault(key,out[key])
        if code == "prerequisite_check_not_passed":
            for key in ("prerequisite_status","prerequisite_result","prerequisite_diagnostics","blocked_stage"):
                if key in out: details.setdefault(key,out[key])
        d["code"] = code
        if details: d["details"] = details
        converted.append(d)
    for key in ("equivalence","value_analysis"):
        child=out.get(key)
        if isinstance(child,dict):
            for d in child.get("diagnostics",[]):
                if d.get("code")=="prerequisite_check_not_passed":
                    d.setdefault("details",{}).update(prerequisite_result=result,
                        prerequisite_checks=out.get("checks",{}),prerequisite_diagnostics=converted)
    return converted

def normalize(report, *, operation=None):
    """Normalize only report metadata; do not traverse expression nodes or witnesses."""
    if not isinstance(report, dict):
        return failure("tool_failure", "invalid_result_object", "Expected a JSON object")
    out = dict(report)
    if "result" in out and out["result"] in RESULTS and "status" not in out and "diagnostics" in out and "checks" in out and "scope" in out and "reason" not in out and "message" not in out:
        if out["result"] not in RESULTS or not isinstance(out.get("diagnostics"), list):
            return failure("tool_failure", "invalid_result_contract", "Unknown result or missing diagnostics")
        for key in REPORT_FIELDS:
            if isinstance(out.get(key), dict) and ("status" in out[key] or "result" in out[key]):
                out[key] = normalize(out[key], operation="equivalence" if key=="equivalence" else key)
        out["diagnostics"] = _diagnostics(out, out["diagnostics"], out.get("stage", "pipeline"), out["result"])
        return out
    status = str(out.pop("status", out.pop("result", "internal_error")))
    reason = str(out.pop("reason", "") or "")
    if not reason:
        for key in ("rejection", "error", "diagnostic"):
            if isinstance(out.get(key), dict) and out[key].get("code"):
                reason = str(out[key]["code"]); break
    message = out.pop("message", None)
    if "numerical_status" in out:
        out["numerical_result"] = category(str(out.pop("numerical_status")))
    stage = out.get("stage", out.get("blocked_stage", out.get("active_stage", "pipeline")))
    result = category(status, reason, out.get("resource"))
    raw_diagnostics = out.get("diagnostics", [])
    diagnostics = [dict(d) for d in raw_diagnostics if isinstance(d, dict)] if isinstance(raw_diagnostics,list) else []
    if status in {"invalid_source_ir","invalid_event_view","invalid_source"}:
        if any(d.get("code") in {"duplicate-barrier-initialization","barrier_initialized_multiple_times"} for d in diagnostics):
            result = "invalid_lifecycle"
    if status == "blocked" and reason == "verification_prerequisite_not_satisfied":
        result = category(str(out.get("prerequisite_status","unsupported")))
    if status == "not_proven" and not reason:
        result = "equivalence_not_proven"
    if status == "partial_scope_equivalent":
        result = "analysis_budget_exceeded"
        reason = "output_family_budget_exceeded"
    if status == "resource_limit" and result == "tool_failure":
        reason = reason or "unclassified_resource_limit"
    checks = dict(out.get("checks", {}))
    kernel = operation in {"kernel_verification", "verification"} or out.get("mode") == "verify"
    for key in ("synchronization","memory","equivalence"):
        checks.setdefault(key,"not_run" if kernel and key != "equivalence" else "not_requested")
    verified = out.pop("verification_status", "")
    if verified == "race_free" or status in {"race_free","potential_race","memory_analysis_inconclusive"}:
        checks["synchronization"] = "passed"
        checks["memory"] = "failed" if status == "potential_race" else "passed" if out.get("memory_analysis_complete") else "incomplete"
    elif status == "not_wellsynced":
        checks.update(synchronization="failed",memory="not_run",equivalence="not_run")
    elif out.get("mode") == "verify" or "verification_performed" in out:
        checks.update(synchronization="incomplete",memory="not_run")
    if verified:
        out["verification_result"] = category(str(verified))
    for key in REPORT_FIELDS:
        if isinstance(out.get(key),dict) and ("status" in out[key] or "result" in out[key]):
            out[key] = normalize(out[key], operation="equivalence" if key=="equivalence" else key)
    if operation=="equivalence" or status in {"equivalent","not_proven","partial_scope_equivalent","translation_or_supported_fragment_blocked"}:
        checks["equivalence"] = "passed" if result=="verified" else "not_run" if status in {"blocked","missing_reference","missing_input"} or reason=="verification_prerequisite_not_satisfied" else "incomplete"
        # Equivalence requires equal domains and equal values on that domain;
        # total definedness is a separate property, not a success prerequisite.
        if result=="verified" and out.get("definedness",{}).get("status")=="not_proven":
            diagnostics.append({"code":"total_definedness_not_proven", "stage":"equivalence_check"})
        if result=="verified" and out.get("access_validity")=="not_proven":
            result="equivalence_not_proven";reason="input_access_validity_not_proven"
            checks["equivalence"]="incomplete"
    eq=out.get("equivalence")
    if isinstance(eq,dict) and "result" in eq:
        checks["equivalence"]=eq.get("checks",{}).get("equivalence","not_run")
        if checks["synchronization"]=="passed" and checks["memory"]=="passed" and result in {"race_free","verified"}:
            result=eq["result"]
            diagnostics.extend(eq["diagnostics"])
    if reason or message or (status not in SUCCESS and not diagnostics):
        diagnostic={"code":reason or ("process_memory_budget_exceeded" if result=="memory_budget_exceeded" else status), "stage":stage}
        for key in ("rejection","error","diagnostic"):
            if isinstance(out.get(key),dict) and out[key].get("code")==diagnostic["code"]:
                diagnostic.update(out[key])
        if message:diagnostic["message"]=message
        for key in ("location","path","resource","details"):
            if key in out:diagnostic[key]=out[key]
        if not diagnostics or (reason and reason != "unsafe-lifecycle" and not any(d.get("code")==diagnostic["code"] for d in diagnostics)):
            diagnostics.insert(0,diagnostic)
    for d in diagnostics:
        d.setdefault("stage",stage)
    if any(d.get("code") in {"unexpected_exception","active_thread_context_missing"} for d in diagnostics): result="tool_failure"
    out["result"]=result
    out["checks"]=checks
    out["diagnostics"]=_diagnostics(out, diagnostics, stage, result)
    scope=dict(out.get("scope",{})) if isinstance(out.get("scope",{}),dict) else {"kind":out["scope"]}
    scope.setdefault("operation",operation or ("verification" if checks["synchronization"]!="not_requested" else "analysis"))
    out["scope"]=scope
    return out

def failure(result, code, message, **details):
    if result not in RESULTS:
        raise ValueError("Unknown result category")
    return {"result":result,"diagnostics":[{"code":code,"message":str(message),**details}],
            "checks":{"synchronization":"not_run","memory":"not_run","equivalence":"not_run"}}

# Categories that report a kernel finding rather than a clean pass or a tool stop.
FINDINGS = frozenset({
    "not_wellsynced", "potential_race", "invalid_lifecycle", "equivalence_not_proven",
})
FAIL_ON_ALIASES = {
    "findings": FINDINGS,
    "any": RESULTS - {"verified", "race_free"},
}


def expand_fail_on(names):
    """Resolve --fail-on values to public result categories."""
    selected = set()
    unknown = []
    for name in names or ():
        if name in FAIL_ON_ALIASES:
            selected.update(FAIL_ON_ALIASES[name])
        elif name in RESULTS:
            selected.add(name)
        else:
            unknown.append(name)
    if unknown:
        raise ValueError("unknown --fail-on value: " + ", ".join(unknown))
    return selected


def apply_fail_on(result, fail_on, default_code):
    """Exit 1 when a passing default code matches a requested result category."""
    selected = fail_on if isinstance(fail_on, (set, frozenset)) else expand_fail_on(fail_on)
    if result.get("result") in selected and default_code == 0:
        return 1
    return default_code


def _plain(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _clip(text, limit=160):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def summarize(report):
    """Return a compact text view of a public result dictionary.

    The JSON object is unchanged. This listing does not infer a pass from a
    missing field, and it does not collapse diagnostics by code.
    """
    if not isinstance(report, dict):
        raise TypeError("summarize expects a result dictionary")
    lines = ["result: " + _plain(report.get("result"))]
    checks = report.get("checks")
    if isinstance(checks, dict):
        lines.append("checks:")
        seen = set()
        for key in ("synchronization", "memory", "equivalence"):
            if key in checks:
                lines.append("  " + key + ": " + _plain(checks[key]))
                seen.add(key)
        for key, value in checks.items():
            if key not in seen:
                lines.append("  " + key + ": " + _plain(value))
    elif "checks" in report:
        lines.append("checks: " + _plain(checks))
    diagnostics = report.get("diagnostics")
    if not isinstance(diagnostics, list) or not diagnostics:
        if "diagnostics" in report:
            lines.append("diagnostics: none" if diagnostics == [] else "diagnostics: " + _plain(diagnostics))
        else:
            lines.append("diagnostics: none")
    else:
        lines.append("diagnostics: " + str(len(diagnostics)))
        for item in diagnostics[:20]:
            if not isinstance(item, dict):
                lines.append("  - " + _clip(_plain(item)))
                continue
            code = item.get("code", "diagnostic_code_missing")
            stage = item.get("stage")
            line = "  - " + _plain(code)
            if stage:
                line += " [" + _plain(stage) + "]"
            if item.get("message"):
                line += ": " + _clip(item["message"])
            lines.append(line)
        extra = len(diagnostics) - 20
        if extra > 0:
            lines.append("  - ... " + str(extra) + " more")
    if "warning_count" in report or "warnings" in report:
        count = report.get("warning_count")
        if count is None and isinstance(report.get("warnings"), list):
            count = len(report["warnings"])
        complete = report.get("warnings_complete")
        suffix = ""
        if complete is False:
            suffix = " (incomplete)"
        elif complete is True:
            suffix = " (complete)"
        lines.append("warnings: " + _plain(count) + suffix)
    if "elapsed_seconds" in report:
        lines.append("elapsed_seconds: " + _plain(report["elapsed_seconds"]))
    assumptions = report.get("semantic_assumptions")
    if isinstance(assumptions, dict) and assumptions:
        parts = [key + "=" + _plain(assumptions[key]) for key in assumptions]
        lines.append("semantic_assumptions: " + ", ".join(parts))
    return "\n".join(lines) + "\n"


def encode_result(report):
    return json.dumps(report, indent=2)


def write_result(report, path):
    """Write the JSON result object. Does not change the in-memory report."""
    destination = Path(path)
    if destination.exists() and destination.is_dir():
        raise ValueError("output path is a directory: " + str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(encode_result(report) + "\n", encoding="utf-8")
    return destination


import argparse
class ResultArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        print(json.dumps(failure("invalid_input", "invalid_arguments", message)))
        raise SystemExit(2)


def diagnostic_code(report):
    items=report.get("diagnostics", [])
    return items[0].get("code", "") if items else report.get("reason", "")

def reports(value):
    if isinstance(value, list):
        return [reports(x) for x in value]
    if isinstance(value, dict):
        value=dict(value)
        for key in ("results","records"):
            if key in value:value[key]=reports(value[key])
        if "status" in value or "result" in value:
            return normalize(value)
    return value


def run_cli(main):
    """Keep command invocation failures inside the public JSON contract."""
    try:
        return main() or 0
    except FileNotFoundError as error:
        result = failure("missing_input", "required_file_missing", error)
    except json.JSONDecodeError as error:
        result = failure("invalid_input", "invalid_json", error)
    except MemoryError as error:
        result = failure("allocation_failed", "memory_allocation_failed", error)
    except ValueError as error:
        result = failure("invalid_input", "invalid_configuration", error)
    except OSError as error:
        result = failure("tool_failure", "io_failure", error)
    except Exception as error:
        result = failure("tool_failure", "unexpected_exception", error,
                         exception_type=type(error).__name__)
    print(json.dumps(result))
    return 2
