"""Small serialization and artifact-boundary helpers; no third-party runtime."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import urlsplit
from urllib.request import build_opener, HTTPRedirectHandler


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Evidence requests must not redirect")


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    atomic_write(path, encode(value))


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".writing-")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,220}", value) or value in {".", ".."}:
        raise ValueError("Invalid case or run identifier")
    return value


def revision(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{40}", value):
        raise ValueError("Expected a full Git commit SHA")
    return value


def repository(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value):
        raise ValueError("Expected owner/repository")
    return value


def confined(root, path):
    root = Path(root).resolve()
    path = Path(path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Artifact path escapes its root")
    target = root / path
    if target.is_symlink() or not target.resolve().is_relative_to(root):
        raise ValueError("Artifact symlink escapes its root")
    return target


def https_base(value):
    parts = urlsplit(value)
    if (parts.scheme != "https" or not parts.netloc or parts.username or parts.password or
            parts.query or parts.fragment or any(p in {".", ".."} for p in parts.path.split("/")) or
            re.search(r"[\\\s%\x00-\x1f]", value)):
        raise ValueError("Evidence base must be a credential-free HTTPS prefix")
    return value.rstrip("/")


def object_key(sha256, name):
    if not re.fullmatch(r"[a-f0-9]{64}", sha256):
        raise ValueError("Expected SHA-256")
    return f"objects/{sha256}/{identifier(name)}"


def object_url(base, sha256, name):
    return https_base(base) + "/" + object_key(sha256, name)


def download_verified(url, path, sha256, *, base):
    if not https_base(url).startswith(https_base(base) + "/"):
        raise ValueError("Evidence URL outside configured store")
    if not re.fullmatch(r"[a-f0-9]{64}", sha256):
        raise ValueError("Missing artifact SHA-256")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        h = hashlib.sha256()
        with os.fdopen(fd, "wb") as out, build_opener(NoRedirects()).open(url, timeout=60) as response:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                h.update(block)
                out.write(block)
        if h.hexdigest() != sha256:
            raise ValueError("Downloaded evidence checksum mismatch")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
