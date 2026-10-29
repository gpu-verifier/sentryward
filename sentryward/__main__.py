"""Public PTX verification command."""
import json
from .api import verify
from .assumptions import add_assumption_arguments, assumption_arguments, assumption_listing
from .frontend.ptx_native import _exit_code
from .results import ResultArgumentParser


def main(argv=None):
    parser = ResultArgumentParser(prog='sentryward', description='Verify a PTX kernel and optionally its reference equivalence.', allow_abbrev=False)
    parser.add_argument('ptx', nargs='?', help='PTX module to verify')
    parser.add_argument('--input', help='kernel input configuration JSON')
    parser.add_argument('--reference', help='Python reference defining reference(...)')
    parser.add_argument('--reference-binding', help='reference correspondence JSON; default: adjacent reference_binding.json or parent')
    parser.add_argument('--timeout', type=float, dest='timeout_seconds', default=600, help='total call budget in seconds; default: 600')
    parser.add_argument('--memory-limit-mib', type=int, help='memory budget in MiB; default: unlimited')
    parser.add_argument('--list-assumptions', action='store_true',
                        help='print accepted analysis assumptions and their defaults as JSON')
    add_assumption_arguments(parser)
    args = parser.parse_args(argv)
    if args.list_assumptions:
        if args.ptx or args.input or args.reference or args.reference_binding or args.assume or args.no_assume:
            parser.error('--list-assumptions cannot be combined with a kernel invocation')
        print(json.dumps(assumption_listing(), indent=2))
        return 0
    if args.ptx is None:
        parser.error('the following arguments are required: ptx')
    if args.input is None:
        parser.error('the following arguments are required: --input')
    assumptions = assumption_arguments(parser, args)
    result = verify(args.ptx, input=args.input, reference=args.reference,
                    reference_binding=args.reference_binding,
                    timeout_seconds=args.timeout_seconds,
                    memory_limit_mib=args.memory_limit_mib,
                    assumptions=assumptions or None)
    print(json.dumps(result, indent=2))
    return _exit_code(result)

if __name__ == '__main__':
    raise SystemExit(main())
