"""Locate the co-located example input, reference and binding."""
from pathlib import Path

def reference_files(root: Path, name: str):
    folder = root / "examples" / name
    return folder / "reference.py", folder / "input.json", folder / "reference_binding.json"
