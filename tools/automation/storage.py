"""S3-compatible evidence storage, using HTTPS/SigV4 and conditional writes."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler

from .common import https_base, object_key, file_hash


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Authenticated storage requests must not redirect")


def sign(method, url, body, *, access_key, secret_key, region, token=None, headers=None, timestamp=None):
    """AWS Signature Version 4, path-style bucket addressing, no query parameters."""
    timestamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parts = urlsplit(url)
    if parts.query or parts.fragment:
        raise ValueError("Signed storage URL must not have query or fragment")
    payload = hashlib.sha256(body).hexdigest()
    signed = {k.lower(): " ".join(str(v).split()) for k, v in (headers or {}).items()}
    signed.update(host=parts.netloc, **{"x-amz-date": timestamp, "x-amz-content-sha256": payload})
    if token:
        signed["x-amz-security-token"] = token
    names = ";".join(sorted(signed))
    canonical = "\n".join([method, parts.path or "/", "", "".join(f"{k}:{signed[k]}\n" for k in sorted(signed)), names, payload])
    scope = f"{timestamp[:8]}/{region}/s3/aws4_request"
    to_sign = "\n".join(["AWS4-HMAC-SHA256", timestamp, scope, hashlib.sha256(canonical.encode()).hexdigest()])
    key = ("AWS4" + secret_key).encode()
    for item in (timestamp[:8], region, "s3", "aws4_request"):
        key = hmac.new(key, item.encode(), hashlib.sha256).digest()
    signature = hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest()
    signed["authorization"] = f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, SignedHeaders={names}, Signature={signature}"
    return signed


class S3Store:
    def __init__(self, endpoint, bucket, public_base, *, region="us-east-1", access_key=None, secret_key=None, token=None):
        self.endpoint, self.public_base = https_base(endpoint), https_base(public_base)
        if not bucket or "/" in bucket:
            raise ValueError("Invalid S3 bucket")
        self.bucket, self.region = bucket, region
        self.access_key = access_key or os.environ["AWS_ACCESS_KEY_ID"]
        self.secret_key = secret_key or os.environ["AWS_SECRET_ACCESS_KEY"]
        self.token = token or os.environ.get("AWS_SESSION_TOKEN")
        self.opener = build_opener(NoRedirects())

    @classmethod
    def from_env(cls):
        return cls(os.environ["SENTRYWARD_S3_ENDPOINT"], os.environ["SENTRYWARD_S3_BUCKET"],
                   os.environ["SENTRYWARD_EVIDENCE_URL"], region=os.environ.get("AWS_REGION") or "us-east-1")

    def request(self, method, key, body=b"", headers=None):
        if any(p in {"", ".", ".."} for p in key.split("/")):
            raise ValueError("Invalid storage key")
        url = self.endpoint + "/" + quote(self.bucket, safe="") + "/" + quote(key, safe="/-_.~")
        signed = sign(method, url, body, access_key=self.access_key, secret_key=self.secret_key,
                      region=self.region, token=self.token, headers=headers)
        with self.opener.open(Request(url, data=body if method == "PUT" else None, method=method, headers=signed), timeout=120) as response:
            return response.read()

    def upload(self, path):
        path = Path(path)
        sha = file_hash(path)
        key = object_key(sha, path.name)
        try:
            self.request("PUT", key, path.read_bytes(), {"If-None-Match": "*", "x-amz-meta-sha256": sha})
        except HTTPError as error:
            if error.code not in {409, 412}:
                raise
        body = self.request("GET", key)
        if hashlib.sha256(body).hexdigest() != sha:
            raise ValueError("Stored evidence checksum mismatch")
        uri = self.public_base + "/" + key
        with self.opener.open(uri, timeout=120) as response:
            h = hashlib.sha256()
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                h.update(chunk)
        if h.hexdigest() != sha:
            raise ValueError("Public evidence URL does not serve uploaded bytes")
        return {"sha256": sha, "uri": uri}
