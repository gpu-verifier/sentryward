"""Public verification API. Implementation and diagnostic transports are private."""
from pathlib import Path
import json
import subprocess
import sys
import time
from .frontend.ptx_native import _positive, _run_ptx
from .results import failure, normalize, publish


def verify(ptx, *, input, reference=None,
           reference_binding=None, timeout_seconds=600, memory_limit_mib=None):
    """Return a complete result/diagnostics dictionary.

    Paths identify PTX, kernel input JSON and an optional Python reference.
    Reference arguments are derived from kernel input and reference_binding.json.
    timeout_seconds covers this entire call, including reference capture.
    The default total time budget is 600 seconds; memory is unlimited by default.
    """
    return publish(_verify(ptx, input=input, reference=reference, reference_binding=reference_binding,
                           timeout_seconds=timeout_seconds, memory_limit_mib=memory_limit_mib))


def _verify(ptx, *, input, reference, reference_binding, timeout_seconds, memory_limit_mib):
    started = time.monotonic()
    stage = 'input_validation'
    try:
        _positive(timeout_seconds, 'timeout_seconds')
        if memory_limit_mib is not None and (type(memory_limit_mib) is not int or
                not 0 < memory_limit_mib <= (2**64 - 1) // (1024 * 1024)):
            raise ValueError('memory_limit_mib must be None or a positive integer MiB budget')
        if reference is None and (reference_binding is not None):
            raise ValueError('reference_binding requires reference')
        deadline = started + timeout_seconds
        ptx, config = Path(ptx), Path(input)
        for path in (ptx, config):
            if not path.is_file():
                return failure('missing_input', 'input_file_missing', str(path), stage=stage)
        # Validate here so malformed user JSON is not mistaken for a worker crash.
        if not isinstance(json.loads(config.read_text()), dict):
            raise ValueError('input must contain a JSON object')
        expression, binding, reference_failure = None, None, None
        if reference is not None:
            stage = 'reference_capture'
            source = Path(reference)
            binding = Path(reference_binding) if reference_binding is not None else source.parent / 'reference_binding.json'
            if reference_binding is None and not binding.is_file():
                binding = source.parent.parent / 'reference_binding.json'
            missing = [str(p) for p in (source, binding) if not p.is_file()]
            if missing:
                reference_failure = failure('missing_input', 'reference_file_missing',
                                            'Required reference files are missing',
                                            stage=stage, details={'paths': missing})
            else:
                for path in (binding,):
                    if not isinstance(json.loads(path.read_text()), dict):
                        raise ValueError(f'{path} must contain a JSON object')
                command = [sys.executable, '-m', 'sentryward._capture_reference', str(source), str(config), str(binding)]
                if memory_limit_mib is not None:
                    command += ['--rss-limit-mib', str(memory_limit_mib)]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout_seconds)
                captured = subprocess.run(command, capture_output=True, timeout=remaining, check=False)
                if captured.returncode:
                    try:
                        reference_failure = json.loads(captured.stdout)['capture_failure']
                    except (ValueError, KeyError, TypeError):
                        reference_failure = failure('tool_failure', 'reference_capture_failed',
                            'Reference worker failed', stage=stage,
                            details={'exit_code': captured.returncode,
                                     'stderr': captured.stderr.decode(errors='replace')})
                else:
                    expression = captured.stdout
        if reference_failure is not None:
            reference_failure = normalize(reference_failure, operation='equivalence')
            reference_failure['checks']['equivalence'] = 'not_run'
        stage = 'kernel_verification'
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired([], timeout_seconds)
        result = _run_ptx(ptx, config, reference=expression, reference_binding=binding,
                          deadline=deadline,
                          reference_failure=reference_failure,
                          pipeline_timeout_seconds=remaining,
                          wellsync_timeout_seconds=remaining,
                          rss_limit_mib=memory_limit_mib)
        result.pop('_process', None)
        result['elapsed_seconds'] = time.monotonic() - started
        if result['elapsed_seconds'] > timeout_seconds and result.get('result') != 'timeout':
            # The completed report is public and keeps its two checks: normalize
            # reads its synchronization check as including the memory check.
            checks = result.get('checks', {})
            result = normalize({'result': 'timeout', 'stage': 'result_processing',
                'diagnostics': [{'code': 'invocation_time_budget_exceeded',
                    'stage': 'result_processing', 'details': {'limit_seconds': timeout_seconds}}],
                'checks': {'synchronization': checks.get('synchronization', 'not_run'),
                           'equivalence': checks.get('equivalence', 'not_run')},
                'partial_report': result})
        return result
    except subprocess.TimeoutExpired:
        return normalize({'result': 'timeout', 'stage': stage,
                          'diagnostics': [{'code': 'invocation_time_budget_exceeded', 'stage': stage}],
                          'resource': {'kind': 'time', 'origin': 'configured_budget',
                                       'limit': timeout_seconds, 'unit': 'seconds'}})
    except (ValueError, TypeError) as error:
        return failure('invalid_input', 'invalid_configuration', error, stage=stage)
    except FileNotFoundError as error:
        return failure('tool_failure', 'executable_missing', error, stage=stage)
    except MemoryError as error:
        return failure('allocation_failed', 'memory_allocation_failed', error, stage=stage)
    except OSError as error:
        return failure('tool_failure', 'process_or_io_failure', error, stage=stage)
