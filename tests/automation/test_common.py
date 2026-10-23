"""Public evidence downloads fail closed before replacing saved bytes."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tools.automation.common import download_verified, https_base, NoEvidenceRedirects


class EvidenceDownloadTests(unittest.TestCase):
    def test_store_prefix_rejects_ambiguous_paths_and_credentials(self):
        for url in ("http://store.test/root", "https://user:password@store.test/root",
                    "https://store.test/a/../b", "https://store.test/a/./b",
                    "https://store.test/%2e%2e/b", "https://store.test/a\\b",
                    "https://store.test/a b", "https://store.test/root?key=secret"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                https_base(url)

    def test_bad_hash_preserves_existing_file_and_cleans_temporary_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "output.json"
            path.write_bytes(b"previous")
            response = io.BytesIO(b"wrong bytes")
            response.url = "https://store.test/root/output.json"
            opener = Mock()
            opener.open.return_value = response
            with patch("tools.automation.common.build_opener", return_value=opener):
                with self.assertRaisesRegex(ValueError, "checksum"):
                    download_verified(response.url, path, hashlib.sha256(b"expected").hexdigest(),
                                      base="https://store.test/root")
            self.assertEqual(path.read_bytes(), b"previous")
            self.assertEqual(list(Path(temporary).iterdir()), [path])

    def test_sibling_prefix_is_rejected_before_network_access(self):
        with patch("tools.automation.common.build_opener") as opener:
            with self.assertRaises(ValueError):
                download_verified("https://store.test/root-other/data", Path("unused"), "a" * 64,
                                  base="https://store.test/root")
            opener.assert_not_called()

    def test_redirect_is_rejected_before_requesting_new_origin(self):
        with self.assertRaisesRegex(ValueError, "redirect"):
            NoEvidenceRedirects().redirect_request(None, None, 302, "moved", {}, "https://elsewhere.test")
