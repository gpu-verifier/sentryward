"""Dynamic references must be importable by the capture backend's introspection."""
import contextlib
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from sentryward._capture_reference import _main


class ReferenceCaptureTests(unittest.TestCase):
    def test_backend_can_resolve_reference_defining_module(self):
        backend = types.ModuleType("sentryward._reference")
        def extract(function, config):
            module = importlib.import_module(function.__module__)
            self.assertIs(module.reference, function)
            return {"captured": function()}, None
        backend.extract = extract
        binding = types.ModuleType("sentryward._reference_binding")
        binding.derive_reference_input = lambda config, mapping: config
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "reference.py").write_text("def reference(): return 7\n")
            for name in ("input.json", "binding.json"):
                (root / name).write_text("{}")
            stdout = io.StringIO()
            original_path = list(sys.path)
            try:
                with patch.dict(sys.modules, {"sentryward._reference": backend,
                                             "sentryward._reference_binding": binding}), \
                     patch.object(sys, "argv", ["capture", str(root / "reference.py"),
                                                str(root / "input.json"), str(root / "binding.json")]), \
                     contextlib.redirect_stdout(stdout):
                    _main()
            finally:
                sys.path[:] = original_path
            self.assertEqual(json.loads(stdout.getvalue()), {"captured": 7})


if __name__ == "__main__":
    unittest.main()
