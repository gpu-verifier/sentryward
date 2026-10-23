#!/usr/bin/env python3
"""Compose report.md from saved verifier artifacts and an optional source explanation."""

import argparse
import json
import os
from pathlib import Path
import tempfile

from report_layout import MARKER, compose, validate_inputs
from artifact_links import verify_identity


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
        run_input, result = (
            json.loads(args.input.read_text()),
            json.loads(args.result.read_text()),
        )
        ptx = args.ptx.read_text()
        if args.bindings:
            verify_identity(args.identity, artifacts)
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
            identity=args.identity,
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
