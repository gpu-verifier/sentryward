"""Public storage contracts, exercised without credentials or network calls."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from urllib.error import HTTPError

from tools.automation.common import object_key
from tools.automation.storage import NoRedirects, S3Store, sign


def http_error(code):
    error = HTTPError("https://s3.example.test/bucket/key", code, "simulated", {}, io.BytesIO())
    error.close()
    return error


class Response(io.BytesIO):
    pass


class MemoryStore(S3Store):
    """Implements S3's ETag preconditions, including interleaved writers."""
    def __init__(self):
        super().__init__("https://s3.example.test", "bucket", "https://evidence.example.test",
                         access_key="example", secret_key="example")
        self.objects = {}
        self.calls = []
        self.before_put = None
        self.public_override = None
        self.opener = self

    def request(self, method, key, body=b"", headers=None):
        headers = headers or {}
        self.calls.append((method, key, dict(headers)))
        if method == "GET":
            if key not in self.objects:
                raise http_error(404)
            data = self.objects[key]
            return data
        if method != "PUT":
            raise AssertionError(method)
        if self.before_put:
            self.before_put(key, body, headers)
        previous = self.objects.get(key)
        if headers.get("If-None-Match") == "*" and previous is not None:
            raise http_error(412)
        self.objects[key] = body
        return b""

    def open(self, uri, timeout):
        self.calls.append(("PUBLIC", uri, {}))
        if not uri.startswith(self.public_base + "/"):
            raise AssertionError(uri)
        body = self.objects[uri[len(self.public_base) + 1:]]
        return Response(body if self.public_override is None else self.public_override)


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "output.json"
        self.path.write_bytes(b'{"result":"potential_race","future_field":[1,2]}\n')
        self.store = MemoryStore()

    def test_aws_published_signature_vector(self):
        # AWS S3 GET Object example, sig-v4-header-based-auth.html, 2013-05-24.
        signed = sign("GET", "https://examplebucket.s3.amazonaws.com/test.txt", b"",
                      access_key="AKIAIOSFODNN7EXAMPLE",
                      secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", region="us-east-1",
                      headers={"Range": "bytes=0-9"}, timestamp="20130524T000000Z")
        self.assertEqual(signed["authorization"],
                         "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request, "
                         "SignedHeaders=host;range;x-amz-content-sha256;x-amz-date, "
                         "Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41")

    def test_signing_binds_body_precondition_and_session_token(self):
        options = dict(access_key="example", secret_key="example", region="us-east-1",
                       timestamp="20261021T000000Z", token="session")
        first = sign("PUT", "https://s3.example.test/bucket/key", b"one",
                     headers={"If-Match": '"one"'}, **options)
        changed_body = sign("PUT", "https://s3.example.test/bucket/key", b"two",
                            headers={"If-Match": '"one"'}, **options)
        changed_condition = sign("PUT", "https://s3.example.test/bucket/key", b"one",
                                 headers={"If-Match": '"two"'}, **options)
        self.assertEqual(first["x-amz-security-token"], "session")
        self.assertIn("if-match", first["authorization"])
        self.assertIn("x-amz-security-token", first["authorization"])
        self.assertNotEqual(first["authorization"], changed_body["authorization"])
        self.assertNotEqual(first["authorization"], changed_condition["authorization"])
        with self.assertRaises(ValueError):
            sign("GET", "https://s3.example.test/bucket/key?version=1", b"", **options)

    def test_request_uses_quoted_path_and_authenticated_conditional_put(self):
        captured = []

        class Capture:
            def open(self, request, timeout):
                captured.append(request)
                return Response(b"stored")

        self.store.opener = Capture()
        data = S3Store.request(self.store, "PUT", "objects/a/file name.json", b"payload",
                                    {"If-None-Match": "*"})
        request = captured[0]
        self.assertEqual(request.full_url, "https://s3.example.test/bucket/objects/a/file%20name.json")
        self.assertEqual(request.data, b"payload")
        self.assertEqual(request.get_method(), "PUT")
        headers = {name.lower(): value for name, value in request.header_items()}
        self.assertEqual(headers["if-none-match"], "*")
        self.assertIn("if-none-match", headers["authorization"])
        self.assertEqual(data, b"stored")
        for key in ("../private", "/objects/x", "objects//x", "objects/./x"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                S3Store.request(self.store, "PUT", key)

    def test_upload_checks_private_and_public_bytes_and_is_idempotent(self):
        first = self.store.upload(self.path)
        repeated = self.store.upload(self.path)
        sha = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.assertEqual(first, {"sha256": sha, "uri": self.store.public_base + "/" + object_key(sha, self.path.name)})
        self.assertEqual(first, repeated)
        self.assertEqual(len(self.store.objects), 1)
        self.assertEqual([call[0] for call in self.store.calls], ["PUT", "GET", "PUBLIC"] * 2)
        self.assertEqual(self.store.calls[0][2]["If-None-Match"], "*")

    def test_corrupt_existing_object_or_public_mirror_cannot_be_published(self):
        sha = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.store.objects[object_key(sha, self.path.name)] = b"corrupt"
        with self.assertRaisesRegex(ValueError, "Stored evidence checksum"):
            self.store.upload(self.path)
        self.store.objects.clear()
        self.store.public_override = b"different public bytes"
        with self.assertRaisesRegex(ValueError, "Public evidence URL"):
            self.store.upload(self.path)

    def test_unexpected_upload_errors_are_not_treated_as_existing_objects(self):
        self.store.before_put = lambda *args: (_ for _ in ()).throw(http_error(503))
        with self.assertRaises(HTTPError) as caught:
            self.store.upload(self.path)
        self.assertEqual(caught.exception.code, 503)
        self.assertEqual([call[0] for call in self.store.calls], ["PUT"])

    def test_authenticated_requests_must_not_follow_redirects(self):
        with self.assertRaisesRegex(ValueError, "must not redirect"):
            NoRedirects().redirect_request(None, None, 307, "redirect", {}, "https://other.example.test")


if __name__ == "__main__":
    unittest.main()
