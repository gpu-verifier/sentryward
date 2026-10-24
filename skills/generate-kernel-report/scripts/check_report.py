#!/usr/bin/env python3
"""Check report composition, artifact identities, citations and disclosure structure."""

import argparse
import json
from pathlib import Path
import re
import sys

from report_facts import verify_identity
from report_layout import MARKER, compose, saved_result, validate_inputs


def check_current(
    report,
    ptx_path,
    input_path,
    result_path,
    draft_path=None,
    identity=None,
    source_paths=(),
    equivalence_path=None,
    artifact_base=None,
    result_panel=False,
):
    """Check current composition and citation validity, not semantic correctness."""
    artifacts = [input_path, result_path, ptx_path]
    verify_identity(identity, artifacts)
    source = report.read_text()
    if not source.startswith(MARKER):
        return ["Expected a version-2 report. Recompose through the current skill."]
    draft = json.loads(draft_path.read_text()) if draft_path else None
    expected = compose(
        draft,
        json.loads(input_path.read_text()),
        saved_result(json.loads(result_path.read_text())),
        ptx_path.read_text(),
        artifacts,
        {p.name: p.read_text() for p in source_paths},
        report_path=report,
        source_paths=source_paths,
        equivalence_path=equivalence_path,
        artifact_base=artifact_base,
        result_panel=result_panel,
    )
    issues = []
    if source != expected:
        issues.append(
            "Report differs from saved facts or its explanation plan. Recompose; do not edit Markdown."
        )
    depth = 0
    for tag in re.findall(r"</?details\b[^>]*>", source):
        depth += -1 if tag.startswith("</") else 1
        if depth < 0 or depth > 2:
            issues.append("Disclosures must be balanced and no deeper than two levels.")
            break
    if depth != 0:
        issues.append("Unbalanced disclosures.")
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("ptx", "result"):
        parser.add_argument("--" + flag, required=True, type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--draft", type=Path)
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--source", type=Path, action="append", default=[])
    parser.add_argument("--equivalence", type=Path)
    parser.add_argument("--artifact-base")
    parser.add_argument("--result-panel", action="store_true")
    args = parser.parse_args()
    try:
        if not args.report:
            verify_identity(
                args.identity,
                [p for p in (args.input, args.result, args.ptx) if p is not None],
            )
            raw = saved_result(json.loads(args.result.read_text()))
            if args.input:
                validate_inputs(
                    json.loads(args.input.read_text()), raw, args.ptx.read_text()
                )
            print(
                "INPUT CHECK: identities and entry checked where supplied; no explanation checked."
            )
            return 0
        if not args.input:
            parser.error("--report requires --input for fact preservation checks.")
        issues = check_current(
            args.report,
            args.ptx,
            args.input,
            args.result,
            args.draft,
            args.identity,
            args.source,
            args.equivalence,
            args.artifact_base,
            args.result_panel,
        )
        for issue in issues:
            print("FAIL:", issue)
        if not issues:
            print(
                "PASS: artifact agreement, composition, exact citations and disclosure structure. Semantic correctness and readability require review."
            )
        return bool(issues)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print("FAIL:", str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
