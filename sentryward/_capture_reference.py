"""Capture a PyTorch reference to a pipe; no expression files or PTX reruns."""
import argparse
import contextlib
import importlib.util
import json
import resource
import sys
from pathlib import Path


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("input", type=Path)
    parser.add_argument("binding", type=Path)
    parser.add_argument("--rss-limit-mib", type=int, help="optional positive memory budget in MiB; default: unlimited")
    args = parser.parse_args()
    if args.rss_limit_mib is not None:
        if args.rss_limit_mib <= 0:
            parser.error("memory limit must be positive when specified")
        limit = args.rss_limit_mib * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    # Resolve sibling helpers and imports from the reference's own project,
    # independently of the caller's working directory.
    sys.path.insert(0, str(args.source.resolve().parent))
    for parent in args.source.resolve().parents:
        if (parent / 'pyproject.toml').is_file() or (parent / 'setup.py').is_file():
            sys.path.insert(1, str(parent))
            break
    # Reference/library diagnostics must not corrupt the expression stream.
    with contextlib.redirect_stdout(sys.stderr):
        from sentryward._reference import extract
        spec = importlib.util.spec_from_file_location("example_reference", args.source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from sentryward._reference_binding import derive_reference_input
        config = derive_reference_input(json.loads(args.input.read_text()), json.loads(args.binding.read_text()))
        expression, _ = extract(module.reference, config)
    json.dump(expression, sys.stdout, separators=(",", ":"))


def main():
    from sentryward.results import failure
    try:
        _main()
    except Exception as error:
        if isinstance(error, ModuleNotFoundError):
            result, code = 'missing_input', 'reference_dependency_missing'
        elif isinstance(error, MemoryError):
            result, code = 'allocation_failed', 'memory_allocation_failed'
        elif type(error).__name__ == 'BindingError':
            result, code = 'invalid_input', 'reference_binding_invalid'
        elif type(error).__name__ == 'Unsupported':
            result, code = 'unsupported', 'reference_operation_unsupported'
        else:
            result, code = 'invalid_input', 'reference_execution_failed'
        json.dump({'capture_failure': failure(result, code, error, stage='reference_capture')}, sys.stdout)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
