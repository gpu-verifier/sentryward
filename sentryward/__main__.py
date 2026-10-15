"""Public PTX verification command."""
import json
import sys
from .api import verify
from .frontend.ptx_native import _exit_code
from .results import ResultArgumentParser


def main(argv=None):
    parser = ResultArgumentParser(prog='sentryward', description='Verify a PTX kernel and optionally its reference equivalence.', allow_abbrev=False)
    parser.add_argument('ptx')
    parser.add_argument('--input', required=True, help='kernel input configuration JSON')
    parser.add_argument('--reference', help='Python reference defining reference(...)')
    parser.add_argument('--reference-binding', help='reference correspondence JSON; default: adjacent reference_binding.json or parent')
    parser.add_argument('--timeout', type=float, dest='timeout_seconds', default=600, help='total call budget in seconds; default: 600')
    parser.add_argument('--memory-limit-mib', type=int, help='memory budget in MiB; default: unlimited')
    args = parser.parse_args(argv)
    result = verify(**vars(args))
    print(json.dumps(result, indent=2))
    return _exit_code(result)

if __name__ == '__main__':
    raise SystemExit(main())
