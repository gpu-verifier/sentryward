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
    if is_public(report):
        return dict(report)
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
    stage = out.get("stage", out.get("blocked_stage", out.get("active_stage",
                    "equivalence_check" if operation == "equivalence" else "pipeline")))
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

PUBLIC_FIELDS = frozenset({"schema", "scope", "result", "checks", "diagnostics", "warnings", "warning_count",
    "warning_samples_truncated", "warnings_complete", "assumptions", "assumption_dependent", "assumption_reporting",
    "assumption_hints", "frontend_assumptions", "verification_limitations", "pipeline_elapsed_seconds",
    "elapsed_seconds", "peak_rss_mib", "rss_limit_mib", "wellsync_timeout_seconds",
    "whole_pipeline_timeout_seconds", "resource", "timeout_scope"})
EVIDENCE_FIELDS = frozenset({"races", "race_count", "race_samples_truncated", "failed_required_orders",
    "failed_required_orders_count", "failed_required_orders_sample_limit", "failed_required_orders_truncated",
    "counterexample", "rejection", "rejection_kind", "source_events", "synthetic_events", "source_event_scope",
    "provenance_events_total", "unmapped_witness_events", "witness_event_list_limit",
    "witness_event_lists_truncated", "unsupported_memory_pair"})

def public_stage(stage):
    """Public stage of an internal stage name (mirrors result.h PublicStage)."""
    if stage in {"input_validation", "input_configuration", "cli_arguments", "file_io", "ptx_parser",
                 "launch_binding"} or stage.startswith("ptx_input"):
        return "input"
    if stage == "memory_effect_binding" or stage.startswith(("value_", "semantic_ir", "scalar_", "expression",
            "equivalence", "reference_", "left_", "right_")) or stage.endswith("_s"):
        return "equivalence"
    if stage.startswith(("ptx_execution", "source_ir", "event_view", "memory_", "synchronization", "lifecycle",
                         "cluster_", "shared_memory", "verification", "frontend")):
        return "synchronization"
    return "pipeline"

def is_public(report):
    if not isinstance(report, dict):
        return False
    checks = report.get("checks")
    return "internal" in report or (isinstance(checks, dict) and "synchronization" in checks
                                    and "memory" not in checks)

def _strip_sections(section):
    for key in REPORT_FIELDS:
        child = section.get(key)
        if isinstance(child, dict):
            child = section[key] = {k: v for k, v in child.items() if k not in {"result", "checks", "scope"}}
            _strip_sections(child)

def publish(report):
    """Project a normalized report onto the public contract (mirrors result.h Publish).

    Two checks, diagnostics with public stages, failure evidence, assumptions and
    resource figures stay at the top level; everything else moves under internal,
    whose sections keep their diagnostics but no conclusion of their own.
    """
    if not isinstance(report, dict) or is_public(report):
        return report
    checks = report.get("checks", {})
    sync = checks.get("synchronization", "not_run")
    memory = checks.get("memory", sync)
    merged = ("failed" if report.get("result") == "invalid_lifecycle" or "failed" in (sync, memory) else
              ("passed" if memory == "passed" else "incomplete") if sync == "passed" else sync)
    diagnostics = []
    for original in report.get("diagnostics", []):
        d = dict(original) if isinstance(original, dict) else original
        if isinstance(d, dict):
            internal = d.get("stage") if isinstance(d.get("stage"), str) else "pipeline"
            d["stage"] = public_stage(internal)
            if d["stage"] != internal:
                details = d.get("details", {})
                d["details"] = {**(details if isinstance(details, dict) else {"evidence": details}),
                                "internal_stage": internal}
        diagnostics.append(d)
    published, evidence, internal = {}, {}, {}
    for key, value in report.items():
        if key in PUBLIC_FIELDS or key.startswith("_"):
            published[key] = value
        elif key in EVIDENCE_FIELDS:
            evidence[key] = value
        else:
            internal[key] = value
    if "diagnostics" in report:
        published["diagnostics"] = diagnostics
    published["checks"] = {"synchronization": merged, "equivalence": checks.get("equivalence", "not_requested")}
    if evidence:
        published["evidence"] = evidence
    _strip_sections(internal)
    if internal:
        published["internal"] = internal
    return published

def failure(result, code, message, **details):
    if result not in RESULTS:
        raise ValueError("Unknown result category")
    return {"result":result,"diagnostics":[{"code":code,"message":str(message),**details}],
            "checks":{"synchronization":"not_run","memory":"not_run","equivalence":"not_run"}}

import argparse
class ResultArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        print(json.dumps(publish(failure("invalid_input", "invalid_arguments", message))))
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
    print(json.dumps(publish(result)))
    return 2
