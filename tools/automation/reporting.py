"""Compose and check deterministic reports from frozen verification evidence."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def report_arguments(case: Path, identity: Path) -> list[str]:
    case = Path(case).resolve()
    arguments = ["--input", str(case / "input.json"), "--result", str(case / "output.json"),
                 "--ptx", str(case / "kernel.ptx"), "--identity", str(Path(identity).resolve()),
                 "--report", str(case / "report/report.md"), "--result-panel"]
    if (case / "reference.py").is_file():
        arguments += ["--source", str(case / "reference.py")]
    return arguments


def run_report_script(root: Path, name: str, arguments: list[str]) -> None:
    """Use the trusted report skill without forwarding publication credentials."""
    command = [sys.executable, str(Path(root).resolve() / "skills/generate-kernel-report/scripts" / name),
               *arguments]
    environment = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL", "SYSTEMROOT")
                   if key in os.environ}
    completed = subprocess.run(command, cwd=root, env=environment, capture_output=True,
                               text=True, timeout=120)
    if completed.returncode:
        raise ValueError(f"{name} failed: " + (completed.stderr or completed.stdout)[-4000:])


def generate_report(root: Path, case: Path, identity: Path) -> Path:
    """Compose the facts and verify the exact artifact before returning its path."""
    arguments = report_arguments(case, identity)
    for name in ("report_facts.py", "check_report.py"):
        run_report_script(root, name, arguments)
    return Path(case).resolve() / "report/report.md"
