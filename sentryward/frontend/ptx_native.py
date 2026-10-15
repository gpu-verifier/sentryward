"""Thin process interface to the standalone C++ PTX pipeline.

This module uses the standard library only. PTX parsing, specialization, model
construction, and verification all execute inside ``sentryward-ptx``.
"""
from __future__ import annotations

from sentryward.results import ResultArgumentParser, failure

from sentryward.results import normalize, failure, RESULTS

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Mapping, Sequence

from .._paths import ROOT, native_binary


DEFAULT_WELLSYNC_TIMEOUT_SECONDS = 600
DEFAULT_RSS_LIMIT_MIB = None


def native_executable() -> Path:
    """Locate the independently runnable executable; never build implicitly."""
    return native_binary('sentryward-ptx', 'SENTRYWARD_PTX_NATIVE')


def _positive(value, name, *, zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or not zero and value == 0):
        raise ValueError(f'{name} must be finite and {"nonnegative" if zero else "positive"}')


def _run_ptx(ptx: str | Path, config: str | Path, *,
            executable: str | Path | None = None,
            report_path: str | Path | None = None,
            stdout_path: str | Path | None = None,
            stderr_path: str | Path | None = None,
                    wellsync_timeout_seconds: float = DEFAULT_WELLSYNC_TIMEOUT_SECONDS,
            pipeline_timeout_seconds: float | None = None,
            rss_limit_mib: int | None = DEFAULT_RSS_LIMIT_MIB,
            sync_threads: int = 8,
            exhaustive_memory_search: bool = False,
            continue_memory_search: bool = False,
            semantic_assumptions: Mapping[str, bool] | None = None,
            reference: bytes | None = None, reference_binding: Path | None = None,
            reference_provenance: Path | None = None,
            deadline: float | None = None, reference_failure: dict | None = None) -> dict:
    """Run native PTX processing and return its JSON report.

    ``_process`` contains interface bookkeeping: exit code, wall time and retained
    log paths. The native process owns both deadline scopes and RSS enforcement;
    An omitted RSS limit (None) imposes no tool memory budget.
    Python adds no outer timeout. An interrupted native result retains completed
    timing snapshots, without turning partial work into a verification success.
    """
    _positive(wellsync_timeout_seconds, 'WellSync timeout')
    if pipeline_timeout_seconds is not None:
        _positive(pipeline_timeout_seconds, 'pipeline timeout', zero=True)
    if rss_limit_mib is not None and (type(rss_limit_mib) is not int or not 0 < rss_limit_mib <= (2**64 - 1) // (1024 * 1024)):
        raise ValueError('RSS limit must be None or a positive integer in MiB within the native byte range')
    if type(sync_threads) is not int or not 1 <= sync_threads <= 64:
        raise ValueError('sync threads must be an integer in [1, 64]')
    if type(exhaustive_memory_search) is not bool:
        raise ValueError('exhaustive memory search must be a boolean')
    if type(continue_memory_search) is not bool:
        raise ValueError('continue memory search must be a boolean')
    if continue_memory_search and exhaustive_memory_search:
        raise ValueError('continue memory search requires verification and excludes exhaustive search')
    command = [str(executable or native_executable()), '--ptx', str(ptx), '--config', str(config),
               '--wellsync-timeout-seconds', format(wellsync_timeout_seconds, 'g'),
               '--sync-threads', str(sync_threads)]
    if rss_limit_mib is not None:
        command += ['--rss-limit-mib', str(rss_limit_mib)]
    if pipeline_timeout_seconds is not None:
        command += ['--timeout-seconds', format(pipeline_timeout_seconds, 'g')]
    if exhaustive_memory_search:
        command.append('--exhaustive-memory-search')
    if continue_memory_search:
        command.append('--continue-memory-search')
    for option, path in (('--report', report_path),):
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            command += [option, str(path)]
    for name, enabled in (semantic_assumptions or {}).items():
        if not isinstance(name, str) or not name or type(enabled) is not bool:
            raise ValueError('semantic assumptions must map nonempty names to booleans')
        command += ['--assume' if enabled else '--no-assume', name]
    if reference_failure is not None:
        command += ['--analyze-scalar-fold', '--require-equivalence',
                    '--reference-failure-json', json.dumps(reference_failure)]
    if reference is not None:
        command += ['--analyze-scalar-fold', '--require-equivalence', '--reference-stdin',
                    '--equivalence-mapping', str(reference_binding),
                    '--equivalence-timeout-seconds', format(pipeline_timeout_seconds or 1e9, 'g')]
        if reference_provenance is not None:
            command += ['--reference-provenance', str(reference_provenance)]
    started = time.monotonic()
    with ExitStack() as stack:
        def stream(path):
            if path is None:
                return stack.enter_context(tempfile.TemporaryFile(mode='w+', encoding='utf-8'))
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            return stack.enter_context(destination.open('w+', encoding='utf-8'))
        stdout, stderr = stream(stdout_path), stream(stderr_path)
        interrupted = False
        try:
            options = {}
            if reference is not None:
                options['input'] = reference
            if deadline is not None:
                options['timeout'] = max(0.001, deadline - time.monotonic())
            process = subprocess.run(command, stdout=stdout, stderr=stderr, check=False, **options)
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            return normalize({'result': 'timeout', 'reason': 'time_budget',
                'stage': 'kernel_verification', 'memory_analysis_complete': False,
                'warnings_complete': False,
                'resource': {'kind': 'time', 'origin': 'configured_budget',
                             'limit': pipeline_timeout_seconds, 'unit': 'seconds'}})
        except KeyboardInterrupt:
            interrupted = True
            returncode = -2
        elapsed = time.monotonic() - started
        stdout.seek(0)
        output = stdout.read()
        stderr.seek(0)
        error_output = stderr.read()
        try:
            result = json.loads(output)
            if not isinstance(result, dict) or not isinstance(result.get('result'), str):
                raise ValueError('native report is not a status object')
        except (ValueError, json.JSONDecodeError):
            if interrupted:
                result = {'result': 'tool_failure', 'message': 'Native run interrupted by user'}
            elif returncode == 124 and 'SENTRYWARD_TIMEOUT wellsync' in error_output.splitlines():
                result = {'result': 'timeout', 'timeout_scope': 'wellsync',
                          'resource': {'kind': 'time', 'origin': 'configured_budget',
                                       'limit': wellsync_timeout_seconds, 'unit': 'seconds'},
                          'message': f'WellSync exceeded its {wellsync_timeout_seconds:g}-second budget'}
            else:
                result = {'result': 'tool_failure' if returncode else 'tool_failure',
                          'message': error_output[-8192:] or 'Native process produced no valid JSON report'}
            result.update(schema='spverify.native-ptx-run/v1', implementation='cpp',
                          memory_analysis_complete=False, warnings_complete=False)
        if returncode and result.get('result', '').removeprefix('recovered_') in {'race_free', 'potential_race',
                                                  'verified', 'verified'}:
            result = {'schema': 'spverify.native-ptx-run/v1', 'implementation': 'cpp',
                      'result': 'tool_failure', 'message': 'Process failed after emitting a successful report',
                      'partial_report': result, 'memory_analysis_complete': False, 'warnings_complete': False}
        if interrupted:
            result = {'schema': 'spverify.native-ptx-run/v1', 'implementation': 'cpp',
                      'result': 'tool_failure', 'message': 'Native run interrupted by user',
                      'memory_analysis_complete': False, 'warnings_complete': False}
    result['_process'] = {'command': command, 'returncode': returncode, 'elapsed_seconds': elapsed}
    for field, path in (('stdout', stdout_path), ('stderr', stderr_path)):
        if path is not None:
            result['_process'][field] = str(Path(path).resolve())
    return normalize(result, operation="kernel_verification")


def run_ptx(ptx, config, *, executable=None, report_path=None,
            stdout_path=None, stderr_path=None,
            wellsync_timeout_seconds=DEFAULT_WELLSYNC_TIMEOUT_SECONDS,
            pipeline_timeout_seconds=None, rss_limit_mib=DEFAULT_RSS_LIMIT_MIB,
            sync_threads=8, exhaustive_memory_search=False, continue_memory_search=False,
            semantic_assumptions=None):
    """Verify PTX and return the common result/diagnostics contract."""
    arguments=locals()
    try:
        return _run_ptx(**arguments)
    except FileNotFoundError as error:
        return failure("tool_failure", "executable_missing", error)
    except ValueError as error:
        return failure("invalid_input", "invalid_configuration", error)
    except MemoryError as error:
        return failure("allocation_failed", "memory_allocation_failed", error)
    except OSError as error:
        return failure("tool_failure", "process_or_io_failure", error)


def _exit_code(result):
    code = result.get('_process', {}).get('returncode', 0)
    if code:
        return code if code > 0 else 128 - code
    return 0 if result.get('result') in {'verified','race_free','potential_race','not_wellsynced','invalid_lifecycle','unsupported','equivalence_not_proven'} else 2


def main(argv: Sequence[str] | None = None) -> int:
    from sentryward.__main__ import main as public_main
    return public_main(argv)

if __name__ == "__main__":
    raise SystemExit(main())
