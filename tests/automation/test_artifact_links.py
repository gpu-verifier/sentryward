import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[2] / "skills/generate-kernel-report/scripts"
sys.path.insert(0, str(SCRIPTS))
from artifact_links import validate_uri, verify_identity
from check_report import check_current
from report_layout import compose


class ArtifactLinkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "input.json"
        self.output = self.root / "output.json"
        self.ptx = self.root / "kernel.ptx"
        self.identity = self.root / "identity.json"
        self.report = self.root / "delivery/report.md"
        self.report.parent.mkdir()
        self.input.write_text(json.dumps({"entry": "kernel", "launch": {"block_dim": [32, 1, 1]}}))
        self.output.write_text(json.dumps({"entry": "kernel", "result": "race_free", "warnings": [],
            "warning_count": 0, "warnings_complete": True,
            "checks": {"synchronization": "passed", "memory": "passed", "equivalence": "not_requested"}}))
        self.ptx.write_text(".version 9.2\n.target sm_100a\n.visible .entry kernel() {\n    ret;\n}\n")
        self.artifacts = [self.input, self.output, self.ptx]
        sha = "a" * 40
        self.github = f"https://github.com/gpu-verifier/sentryward/blob/{sha}"
        self.prefix = "https://evidence.example/sentryward"
        self.manifest = {"evidence_base_url": self.prefix, "files": {
            p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in self.artifacts}}
        self.manifest["files"]["input.json"]["uri"] = self.github + "/examples/kernel/input.json"
        self.manifest["files"]["kernel.ptx"].update(uri=self.github + "/corpus/kernel.ptx",
            raw_uri=f"https://raw.githubusercontent.com/gpu-verifier/sentryward/{sha}/corpus/kernel.ptx")
        self.manifest["files"]["output.json"]["uri"] = self.prefix + "/objects/" + "b" * 64 + "/output.json"
        self.save_manifest()

    def save_manifest(self):
        self.identity.write_text(json.dumps(self.manifest))

    def compose(self, identity=True, **kwargs):
        return compose(None, json.loads(self.input.read_text()), json.loads(self.output.read_text()),
            self.ptx.read_text(), self.artifacts, report_path=self.report,
            identity=self.identity if identity else None, **kwargs)

    def test_composer_and_checker_share_mixed_immutable_locations(self):
        document = self.compose(artifact_base="https://github.com/unused/repo/blob/" + "c" * 40)
        self.report.write_text(document)
        self.assertIn(self.github + "/corpus/kernel.ptx", document)
        self.assertIn(self.manifest["files"]["kernel.ptx"]["raw_uri"], document)
        self.assertIn(self.manifest["files"]["output.json"]["uri"], document)
        self.assertNotIn("unused/repo", document)
        self.assertNotIn(str(self.root), document)
        self.assertEqual(check_current(self.report, self.ptx, self.input, self.output,
                                       identity=self.identity), [])
        self.report.write_text(document.replace("RACE", "ALTERED", 1))
        self.assertTrue(check_current(self.report, self.ptx, self.input, self.output,
                                     identity=self.identity))

    def test_cli_identity_links_and_checker_agree(self):
        args = ["--input", str(self.input), "--result", str(self.output), "--ptx", str(self.ptx),
                "--identity", str(self.identity), "--report", str(self.report), "--result-panel"]
        for script in ("report_facts.py", "check_report.py"):
            completed = subprocess.run([sys.executable, str(SCRIPTS / script), *args],
                                       capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
        self.assertIn(self.prefix, self.report.read_text())

    def test_legacy_identity_and_relative_calls_remain_compatible(self):
        self.manifest = {"files": {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                                   for p in self.artifacts}}
        self.save_manifest()
        self.assertEqual(self.compose(), self.compose(identity=False))
        self.assertIn("../kernel.ptx", self.compose())
        renamed = self.root / "saved-ptx.txt"
        renamed.write_bytes(self.ptx.read_bytes())
        self.assertIn(renamed.resolve(), verify_identity(self.identity, [renamed]))

    def test_byte_mismatch_and_ambiguous_copies_are_rejected(self):
        self.output.write_text('{"result":"verified"}')
        with self.assertRaisesRegex(ValueError, "Identity mismatch"):
            self.compose()
        self.manifest["files"]["output.json"]["sha256"] = hashlib.sha256(self.output.read_bytes()).hexdigest()
        for name in ("one", "two"):
            path = self.root / name / "copy.json"
            path.parent.mkdir()
            path.write_text("{}")
            self.manifest["files"][name + "/copy.json"] = {
                "sha256": hashlib.sha256(b"{}").hexdigest(), "uri": self.prefix + "/" + name + "/copy.json"}
        third = self.root / "copy.json"
        third.write_text("{}")
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            verify_identity(self.identity, [third])

    def test_unsafe_and_floating_uris_are_rejected(self):
        uris = ["http://evidence.example/sentryward/result.json", "https://u:p@evidence.example/sentryward/x",
                self.prefix + "/../secret", self.prefix + "/%2e%2e/secret", self.prefix + "/%252e%252e/secret",
                self.prefix + "/x?token=secret", self.prefix + "/x#fragment",
                "https://evidence.example/sentryward-other/x", "https://outside.example/sentryward/x",
                "https://github.com/gpu-verifier/sentryward/blob/main/x", "https://raw.githubusercontent.com/gpu-verifier/sentryward/main/x"]
        for uri in uris:
            with self.subTest(uri=uri), self.assertRaises(ValueError):
                validate_uri(uri, self.prefix)
        for uri in (self.github + "/corpus/k.ptx",
                    "https://github.com/gpu-verifier/sentryward/blob/results/runs/run-12/reports/k/report.md"):
            self.assertEqual(validate_uri(uri), uri)

    def test_manifest_paths_cannot_escape_or_follow_external_symlink(self):
        for name in ("../input.json", "/input.json", "a/../input.json", "./input.json"):
            self.manifest["files"] = {name: {"sha256": "a" * 64}}
            self.save_manifest()
            with self.subTest(name=name), self.assertRaises(ValueError):
                verify_identity(self.identity, [])
        with tempfile.TemporaryDirectory() as outside:
            external = Path(outside) / "source.ptx"
            external.write_text("outside")
            (self.root / "link.ptx").symlink_to(external)
            self.manifest["files"] = {"link.ptx": {"sha256": hashlib.sha256(b"outside").hexdigest()}}
            self.save_manifest()
            with self.assertRaisesRegex(ValueError, "escapes"):
                verify_identity(self.identity, [])


if __name__ == "__main__":
    unittest.main()
