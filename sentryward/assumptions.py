"""User-selected PTX semantic assumptions."""

from collections.abc import Mapping
import json
from pathlib import Path

# Installed wheels carry the configuration inside the package.
_PACKAGED = Path(__file__).resolve().parent / "_config"
CONFIG_DIR = _PACKAGED if _PACKAGED.is_dir() else Path(__file__).resolve().parents[1] / "config"
ASSUMPTION_POOL = json.loads((CONFIG_DIR / "assumption-pool.json").read_text())
ASSUMPTIONS = tuple(item["key"] for item in ASSUMPTION_POOL)


def validate_assumptions(value):
    if not isinstance(value, Mapping):
        raise ValueError("semantic_assumptions must be an object")
    if any(name not in ASSUMPTIONS for name in value):
        raise ValueError("unknown PTX semantic assumption")
    if any(type(enabled) is not bool for enabled in value.values()):
        raise ValueError("semantic assumption flags must be Boolean")
    return dict(value)


def resolve_assumptions(config=None, overrides=None):
    path = CONFIG_DIR / "default-assumptions.json"
    result = validate_assumptions(json.loads(path.read_text()))
    if set(result) != set(ASSUMPTIONS):
        raise ValueError("default configuration must specify every assumption")
    result.update(validate_assumptions({} if config is None else config))
    result.update(validate_assumptions({} if overrides is None else overrides))
    return result


def add_assumption_arguments(parser):
    parser.add_argument("--assume", action="append", choices=ASSUMPTIONS, default=[],
                        help="enable a named analysis assumption for this invocation")
    parser.add_argument("--no-assume", action="append", choices=ASSUMPTIONS, default=[],
                        help="disable a named analysis assumption for this invocation")


def assumption_arguments(parser, args):
    if len(set(args.assume + args.no_assume)) != len(args.assume + args.no_assume):
        parser.error("each semantic assumption may be specified only once")
    return {**dict.fromkeys(args.assume, True), **dict.fromkeys(args.no_assume, False)}


def assumption_listing():
    """Accepted keys and repository defaults, independent of a kernel input."""
    defaults = resolve_assumptions()
    return {
        "defaults": defaults,
        "assumptions": [
            {
                "label": item["label"],
                "key": item["key"],
                "text": item["text"],
                "enabled": defaults[item["key"]],
            }
            for item in ASSUMPTION_POOL
        ],
    }
