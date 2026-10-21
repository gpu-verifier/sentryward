#!/usr/bin/env python3
"""Compose report.md from saved verifier artifacts and an optional source explanation."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from report_layout import MARKER, compose, validate_inputs


def verify_identity(identity, artifacts):
    """Validate the optional supplied manifest before writing a report."""
    if identity is None:
        return
    manifest = json.loads(identity.read_text())
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Identity must identify the supplied files.")
    digests = {}
    for name, item in files.items():
        path = (identity.parent / name).resolve()
        if not path.is_relative_to(identity.parent.resolve()):
            raise ValueError("Identity path escapes its directory: " + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != item.get("sha256"):
            raise ValueError("Identity mismatch: " + name)
        digests.setdefault(path.name, set()).add(item["sha256"])
    known = set().union(*digests.values())
    for actual in artifacts:
        expected = digests.get(actual.name, known)
        if hashlib.sha256(actual.read_bytes()).hexdigest() not in expected:
            raise ValueError(
                "Supplied artifact is not covered by identity: " + str(actual)
            )


def main():
    parser = argparse.ArgumentParser(
        description="Compose one report from saved results, optionally with a source explanation."
    )
    for key in ("input", "result", "ptx"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument(
        "--report", type=Path, help="Write one report.md; otherwise print to stdout."
    )
    parser.add_argument(
        "--draft",
        type=Path,
        help="Optional version-2 source explanation, outside the output directory.",
    )
    parser.add_argument("--identity", type=Path)
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        default=[],
        help="Optional source file available to evidence references by basename.",
    )
    parser.add_argument(
        "--equivalence",
        type=Path,
        help="Link an existing numerical-check artifact from the same run; does not replace the verifier result.",
    )
    parser.add_argument(
        "--artifact-base",
        help="HTTPS repository blob URL, preferably at a fixed commit; artifact paths are relative to the current directory.",
    )
    parser.add_argument(
        "--bindings",
        action="store_true",
        help="Print source-analysis aids, not verifier findings.",
    )
    parser.add_argument(
        "--result-panel",
        action="store_true",
        help="Group saved findings and scope in a GitHub-native result panel.",
    )
    args = parser.parse_args()
    try:
        artifacts = [args.input, args.result, args.ptx]
        verify_identity(args.identity, artifacts)
        run_input, result = (
            json.loads(args.input.read_text()),
            json.loads(args.result.read_text()),
        )
        ptx = args.ptx.read_text()
        if args.bindings:
            validate_inputs(run_input, result, ptx)
            if args.report or args.draft:
                parser.error(
                    "--bindings is an inspection command; omit --report and --draft."
                )
            from parameter_analysis import render

            print(render(run_input, result, ptx))
            return
        if len({p.name for p in args.source}) != len(args.source):
            raise ValueError("Optional source basenames must be distinct.")
        sources = {p.name: p.read_text() for p in args.source}
        draft = json.loads(args.draft.read_text()) if args.draft else None
        if args.equivalence and not args.equivalence.is_file():
            raise ValueError(
                "Numerical-check artifact does not exist: " + str(args.equivalence)
            )
        composed = compose(
            draft,
            run_input,
            result,
            ptx,
            artifacts,
            sources,
            report_path=args.report,
            source_paths=args.source,
            equivalence_path=args.equivalence,
            artifact_base=args.artifact_base,
            result_panel=args.result_panel,
        )
        if args.report is None:
            print(composed, end="")
            return
        if args.report.name != "report.md":
            raise ValueError("The output must be named report.md.")
        if args.draft and args.draft.resolve().is_relative_to(
            args.report.parent.resolve()
        ):
            raise ValueError("Keep draft.json outside the report delivery directory.")
        if args.report.exists() and not args.report.read_text().startswith(
            (MARKER, "<!-- report-layout:v1 -->")
        ):
            raise ValueError(
                "Refusing to replace an unrelated report. Choose an unused output directory."
            )
        # Compose and validate completely before touching the previous good report.
        args.report.parent.mkdir(parents=True, exist_ok=True)
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=args.report.parent, delete=False
            ) as stream:
                tmp = Path(stream.name)
                stream.write(composed)
            os.replace(tmp, args.report)
        finally:
            if tmp is not None and tmp.exists():
                tmp.unlink()
        print("COMPOSED: " + str(args.report))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
