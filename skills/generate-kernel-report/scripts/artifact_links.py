"""Validate saved file identities and their optional immutable public locations."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from urllib.parse import unquote, urlsplit


def _https_url(value):
    if not isinstance(value, str):
        raise ValueError("Artifact URI must be text.")
    parsed = urlsplit(value)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment
            or any(c.isspace() or c in "<>\\\"" for c in value)):
        raise ValueError("Artifact URI must be HTTPS without credentials, query or fragment.")
    decoded = unquote(parsed.path)
    if (any(part in (".", "..") for part in decoded.split("/"))
            or "\\" in decoded or any(ord(c) < 32 for c in decoded)
            or re.search(r"%[0-9a-fA-F]{2}", decoded)):
        raise ValueError("Artifact URI contains an unsafe encoded path.")
    return parsed


def validate_uri(value, evidence_base_url=None):
    """Allow pinned GitHub files or the producer's declared evidence prefix."""
    parsed = _https_url(value)
    parts = unquote(parsed.path).strip("/").split("/")
    if parsed.hostname == "github.com":
        if len(parts) < 5 or parts[2] not in {"blob", "raw"}:
            raise ValueError("GitHub artifact URI must identify a file.")
        ref, suffix = parts[3], parts[4:]
    elif parsed.hostname == "raw.githubusercontent.com":
        if len(parts) < 4:
            raise ValueError("Raw GitHub artifact URI must identify a file.")
        ref, suffix = parts[2], parts[3:]
    else:
        if evidence_base_url is None:
            raise ValueError("Object artifact URI requires evidence_base_url.")
        base = _https_url(evidence_base_url)
        prefix = base.path.rstrip("/") + "/"
        if parsed.netloc != base.netloc or not parsed.path.startswith(prefix):
            raise ValueError("Artifact URI is outside evidence_base_url.")
        return value
    if re.fullmatch(r"[0-9a-fA-F]{40}", ref):
        return value
    if (ref == "results" and len(suffix) >= 3 and suffix[0] == "runs"
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", suffix[1])):
        return value
    raise ValueError("GitHub artifact URI must pin a commit or an immutable results run.")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_identity(identity, artifacts):
    """Return descriptors keyed by the exact supplied path after validating bytes.

    Legacy manifests containing only ``sha256`` remain valid. A copied artifact
    may match by basename and digest; different URI identities for identical
    copies are ambiguous and are rejected instead of choosing one silently.
    """
    if identity is None:
        return {}
    identity = Path(identity)
    manifest = json.loads(identity.read_text())
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Identity must identify the supplied files.")
    prefix = manifest.get("evidence_base_url")
    if prefix is not None:
        _https_url(prefix)
    descriptors = {}
    for name, item in files.items():
        relative = PurePosixPath(name)
        if (not name or relative.is_absolute() or "\\" in name
                or any(part in ("", ".", "..") for part in name.split("/"))):
            raise ValueError("Identity path must be a confined relative path: " + name)
        path = (identity.parent / name).resolve()
        if not path.is_relative_to(identity.parent.resolve()):
            raise ValueError("Identity path escapes its directory: " + name)
        if not isinstance(item, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))):
            raise ValueError("Identity requires a SHA-256 for " + name)
        if _sha256(path) != item["sha256"]:
            raise ValueError("Identity mismatch: " + name)
        for key in ("uri", "raw_uri"):
            if key in item:
                validate_uri(item[key], prefix)
        if "raw_uri" in item and "uri" not in item:
            raise ValueError("raw_uri requires uri: " + name)
        descriptors[path] = dict(item, name=name)
    resolved = dict(descriptors)
    for actual in artifacts:
        if actual is None:
            continue
        path = Path(actual).resolve()
        digest = _sha256(path)
        if path in descriptors:
            matches = [descriptors[path]]
        else:
            matches = [item for saved, item in descriptors.items()
                       if saved.name == path.name and item["sha256"] == digest]
            if not matches and not any(item.get("uri") for item in descriptors.values()):
                # Old hash-only manifests also identified renamed copies by bytes.
                matches = [item for item in descriptors.values() if item["sha256"] == digest]
        if not matches or any(item["sha256"] != digest for item in matches):
            raise ValueError("Supplied artifact is not covered by identity: " + str(actual))
        if len({(item.get("uri"), item.get("raw_uri")) for item in matches}) > 1:
            raise ValueError("Supplied artifact has ambiguous identity links: " + str(actual))
        resolved[path] = matches[0]
    return resolved
