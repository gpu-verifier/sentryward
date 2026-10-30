"""Public PTX verification command."""
import json
from .api import verify
from .assumptions import add_assumption_arguments, assumption_arguments, assumption_listing
from .frontend.ptx_native import _exit_code
from .results import (
    ResultArgumentParser,
    apply_fail_on,
    encode_result,
    expand_fail_on,
    failure,
    summarize,
    write_result,
)


def main(argv=None):
    parser = ResultArgumentParser(prog='sentryward', description='Verify a PTX kernel and optionally its reference equivalence.', allow_abbrev=False)
    parser.add_argument('ptx', nargs='?', help='PTX module to verify')
    parser.add_argument('--input', help='kernel input configuration JSON')
    parser.add_argument('--reference', help='Python reference defining reference(...)')
    parser.add_argument('--reference-binding', help='reference correspondence JSON; default: adjacent reference_binding.json or parent')
    parser.add_argument('--timeout', type=float, dest='timeout_seconds', default=600, help='total call budget in seconds; default: 600')
    parser.add_argument('--memory-limit-mib', type=int, help='memory budget in MiB; default: unlimited')
    parser.add_argument('--output', help='write the JSON result object to this path')
    parser.add_argument('--format', dest='result_format', choices=('json', 'summary'), default='json',
                        help='stdout representation; default: json. summary is compact text; JSON is unchanged')
    parser.add_argument('--fail-on', action='append', default=[], metavar='CATEGORY',
                        help='exit 1 when result matches a category. Repeatable. Aliases: findings, any')
    parser.add_argument('--list-assumptions', action='store_true',
                        help='print accepted analysis assumptions and their defaults as JSON')
    add_assumption_arguments(parser)
    args = parser.parse_args(argv)
    if args.list_assumptions:
        if (args.ptx or args.input or args.reference or args.reference_binding or args.assume or args.no_assume
                or args.output or args.fail_on or args.result_format != 'json'):
            parser.error('--list-assumptions cannot be combined with a kernel invocation')
        print(json.dumps(assumption_listing(), indent=2))
        return 0
    if args.ptx is None:
        parser.error('the following arguments are required: ptx')
    if args.input is None:
        parser.error('the following arguments are required: --input')
    try:
        fail_on = expand_fail_on(args.fail_on)
    except ValueError as error:
        parser.error(str(error))
    assumptions = assumption_arguments(parser, args)
    result = verify(args.ptx, input=args.input, reference=args.reference,
                    reference_binding=args.reference_binding,
                    timeout_seconds=args.timeout_seconds,
                    memory_limit_mib=args.memory_limit_mib,
                    assumptions=assumptions or None)
    if args.output:
        try:
            write_result(result, args.output)
        except ValueError as error:
            print(json.dumps(failure('invalid_input', 'invalid_output_path', error)))
            return 2
        except OSError as error:
            print(json.dumps(failure('tool_failure', 'io_failure', error)))
            return 2
    text = encode_result(result) if args.result_format == 'json' else summarize(result).rstrip('\n')
    print(text)
    return apply_fail_on(result, fail_on, _exit_code(result))

if __name__ == '__main__':
    raise SystemExit(main())
