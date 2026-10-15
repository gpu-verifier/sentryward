"""Locate native executables and scratch space.

An installed wheel carries its executables in ``sentryward/_native``. A
repository checkout keeps them in ``build/cmake`` and its scratch files under
``build/tmp``. ``assumptions.py`` resolves the configuration the same way.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parent
_PACKAGED_NATIVE = PACKAGE_DIR / "_native"
_SUFFIX = ".exe" if sys.platform == "win32" else ""


def native_binary(name: str, environment_variable: str | None = None) -> Path:
    """Resolve an executable; never build implicitly."""
    configured = os.environ.get(environment_variable) if environment_variable else None
    if configured:
        return Path(configured)
    packaged = _PACKAGED_NATIVE / (name + _SUFFIX)
    return packaged if packaged.is_file() else ROOT / "build/cmake" / name


def temporary_root() -> Path:
    if not (ROOT / "CMakeLists.txt").is_file():
        return Path(tempfile.gettempdir())
    root = ROOT / "build/tmp"
    root.mkdir(parents=True, exist_ok=True)
    return root
