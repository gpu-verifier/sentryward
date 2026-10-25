#!/usr/bin/env python3
"""Readable views of saved verifier facts; no kernel-semantic inference.

This module never changes an analysis verdict or interprets a PTX instruction.
Its two public renderers return GFM fragments for the single-file composer.
"""
import hashlib
import html
import json
import os
from pathlib import Path
import re
from urllib.parse import quote, urlsplit

from code_excerpt import excerpt, instruction_opcodes, mask_noncode
from report_layout import saved_pointer

FACTS_START = '<!-- verifier-result:start -->'
FACTS_END = '<!-- verifier-result:end -->'
EVIDENCE_START = '<!-- verifier-evidence:start -->'
EVIDENCE_END = '<!-- verifier-evidence:end -->'
MISSING = object()
STATUSES = {
    'race_free': ('NO RACES REPORTED', 'The verifier reported no races for this saved configuration.'),
    'race_candidate': ('RACE CANDIDATE', 'The verifier retained potentially conflicting memory accesses.'),
    'synchronization_candidate': ('SYNCHRONIZATION CANDIDATE', 'A synchronization check rejected the candidate model execution.'),
    'unsafe_lifecycle': ('LIFECYCLE ORDERING MISSING', 'A required lifecycle happens-before relation is absent in the analyzed model.'),
    'frontend_unsupported': ('FRONTEND UNSUPPORTED', 'The frontend stopped before a complete memory-analysis verdict.'),
    'native_frontend_error': ('FRONTEND ERROR', 'The frontend failed; this is not a successful verification result.'),
    'memory_model_unsupported': ('MEMORY MODEL UNSUPPORTED', 'The memory model could not decide a required access relation.'),
    'invalid_source': ('SOURCE REJECTED', 'The verifier rejected the source under its modeled requirements.'),
    'unresolved_translation': ('TRANSLATION UNRESOLVED', 'The frontend could not resolve the translation needed for verification.'),
    'native_resource_limit': ('RESOURCE LIMIT REACHED', 'Analysis stopped at a resource limit; no complete verification verdict is available.'),
    'native_timeout': ('ANALYSIS TIMED OUT', 'Analysis timed out; no complete verification verdict is available.'),
}
RESULTS = {
    'verified': ('VERIFIED', 'The verifier reports verification complete for the saved scope and checks below.'),
    'race_free': STATUSES['race_free'],
    'not_wellsynced': STATUSES['synchronization_candidate'],
    'potential_race': STATUSES['race_candidate'],
    'invalid_lifecycle': STATUSES['unsafe_lifecycle'],
    'equivalence_not_proven': ('EQUIVALENCE NOT PROVEN', 'The numerical comparison did not establish equivalence; memory and synchronization checks remain separate below.'),
    'unsupported': ('ANALYSIS UNSUPPORTED', 'A required operation is unsupported. The saved checks identify which analyses completed.'),
    'invalid_input': ('INVALID INPUT', 'The verifier rejected the input; no successful verification is inferred.'),
    'missing_input': ('REQUIRED INPUT MISSING', 'A required input was unavailable; no successful verification is inferred.'),
    'timeout': STATUSES['native_timeout'],
    'memory_budget_exceeded': ('MEMORY BUDGET EXCEEDED', 'Analysis stopped at its memory budget; no complete verification verdict is available.'),
    'analysis_budget_exceeded': ('ANALYSIS BUDGET EXCEEDED', 'Analysis stopped at an analysis budget; only the saved checks and covered scope are established.'),
    'allocation_failed': ('ALLOCATION FAILED', 'Analysis could not allocate a required resource; this is not a successful verification result.'),
    'tool_failure': ('TOOL FAILURE', 'The verification tool failed; this is not a successful verification result.'),
}
_VIEWS = {'verified': 'race_free', 'not_wellsynced': 'synchronization_candidate',
          'potential_race': 'race_candidate', 'invalid_lifecycle': 'unsafe_lifecycle',
          'unsupported': 'memory_model_unsupported'}


def _encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _plain(value):
    if value is MISSING:
        return 'not saved'
    if isinstance(value, str):
        return value
    return _encoded(value)


def _inline(value):
    return '<code>' + html.escape(_plain(value)).replace('|', '&#124;').replace('\n', ' ') + '</code>'


def _text(value):
    """Quote saved prose without letting it introduce HTML or table columns."""
    text = html.escape(_plain(value)).replace('|', '&#124;')
    return re.sub(r'(?<![\w>])(%[\w.$]+|A\d{3}|[A-Za-z]\w*_\w+)(?![\w<])', r'<code>\1</code>', text)


def _number(value):
    return f'{value:,}' if type(value) is int else _plain(value)


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _records(value):
    return value if isinstance(value, list) else []


def _outcome(result):
    # A present but unknown/null public result must not fall back to a legacy
    # success status. Do not normalize or overwrite any saved verifier fields.
    return result.get('result', result.get('status', MISSING))


def _view_kind(result):
    value = _outcome(result)
    return _VIEWS.get(value, value) if isinstance(value, str) else None


def _check_summary(result):
    checks = result.get('checks', MISSING)
    if not isinstance(checks, dict):
        return '**Checks:** ' + _inline(checks) + '. No check outcomes are inferred.\n\n'
    labels = {'synchronization': 'Synchronization', 'memory': 'Memory', 'equivalence': 'Equivalence'}
    # Two-check outputs include the memory analysis in the synchronization check.
    fields = [(label, checks.get(key, MISSING)) for key, label in labels.items()
              if key != 'memory' or key in checks]
    fields.extend((key, value) for key, value in checks.items() if key not in labels)
    body = '**Checks:** ' + ' · '.join(html.escape(label) + ' ' + _inline(value) for label, value in fields) + '.\n\n'
    if _view_kind(result) == 'race_candidate':
        if checks.get('synchronization') == 'passed' and checks.get('memory') == 'failed':
            body += 'The synchronization check passed; the separate memory check still found a conflict. Passing one check does not clear the other.\n\n'
        elif 'memory' not in checks and checks.get('synchronization') == 'failed':
            body += 'This output has no separate memory check: the synchronization check includes the memory analysis, which reported the potential race.\n\n'
    return body


def _result_consistency(result):
    """Expose contradictory saved metadata without choosing a replacement verdict."""
    if 'result' not in result:
        return ''
    conflicts = []
    checks = _mapping(result.get('checks'))
    if _outcome(result) == 'verified':
        for key, value in checks.items():
            if value not in ('passed', 'not_requested'):
                conflicts.append(_cite(result, '/checks/' + key) + ' is ' + _inline(value))
    equivalence = _mapping(result.get('equivalence'))
    if equivalence:
        nested = _mapping(equivalence.get('checks')).get('equivalence', MISSING)
        top = checks.get('equivalence', MISSING)
        if nested is not MISSING and top is not MISSING and nested != top:
            conflicts.append('equivalence check is ' + _inline(top) + ' at ' + _cite(result, '/checks/equivalence') + ' but ' + _inline(nested) + ' at ' + _cite(result, '/equivalence/checks/equivalence'))
        # Public outputs keep the conclusion at the top level only; a section
        # without an outcome of its own cannot contradict it.
        if top == 'passed' and ('result' in equivalence or 'status' in equivalence):
            eq_result = _outcome(equivalence)
            if eq_result not in ('verified', 'equivalent'):
                conflicts.append(_cite(result, '/equivalence/result' if 'result' in equivalence else '/equivalence/status') + ' is ' + _inline(eq_result) + ' despite the passed equivalence check')
    if 'status' in result and _view_kind({'status': result['status']}) != _view_kind({'result': result['result']}):
        conflicts.append('legacy ' + _cite(result, '/status') + ' is ' + _inline(result['status']))
    if not conflicts:
        return ''
    return '**Saved fields disagree:** ' + '; '.join(conflicts) + '. The report preserves these fields; it does not resolve the disagreement into a successful verdict.\n\n'


def _ptr(base, key):
    return base + '/' + str(key).replace('~', '~0').replace('/', '~1')


def _ref(pointer, artifact='output.json'):
    return _inline(artifact + '#' + pointer)


def _cite(result, pointer):
    """Cite a pointer into the result view at its saved output.json location."""
    return _ref(saved_pointer(result, pointer))


def _section(result, key):
    """Saved section (`internal` or `evidence`) holding a public output's field."""
    return getattr(result, 'origins', {}).get(key)


def _fold(label, body):
    return '<details>\n<summary><strong>' + html.escape(label) + '</strong></summary>\n\n' + body.strip() + '\n\n</details>\n\n'


def artifact_url(path, report_path=None, artifact_base=None, line=None):
    """Resolve from the report, not the command's working directory."""
    if path is None or not Path(path).is_file():
        return None
    path = Path(path)
    base = Path(report_path).resolve().parent if report_path else Path.cwd().resolve()
    target = quote(os.path.relpath(path.resolve(), base), safe='/.-_')
    if artifact_base is not None:
        url = urlsplit(artifact_base)
        if url.scheme != 'https' or not url.hostname or url.username or url.password or url.query or url.fragment or any(c.isspace() or c in '<>\\' for c in artifact_base):
            raise ValueError('Artifact base must be an HTTPS URL without credentials, query or fragment.')
        relative = path.resolve().relative_to(Path.cwd().resolve())
        target = artifact_base.rstrip('/') + '/' + quote(relative.as_posix(), safe='/.-_')
    if line is not None:
        if type(line) is not int or line < 1:
            raise ValueError('A source link requires a positive physical line number.')
        target += '#L' + str(line)
    return target


def _file_link(label, path, report_path=None, artifact_base=None, line=None):
    if path is None:
        return label + ' (not provided)'
    target = artifact_url(path, report_path, artifact_base, line)
    if target is None:
        return label + ' (file unavailable: ' + _inline(str(path)) + ')'
    label = re.sub(r'([\\\[\]])', r'\\\1', html.escape(label))
    return '[' + label + '](' + target + ')'


def _json_pointer_lines(path, pointers):
    """Find requested nested values in original JSON, without reformatting it.

    Descend only along requested paths; do not build an index of every event
    in large results. Object references point at their key, array items at
    the start of the item. Unavailable paths have no invented line number.
    """
    if path is None or not Path(path).is_file():
        return {}
    tree = {}
    for pointer in pointers:
        if pointer and (not pointer.startswith('/') or re.search(r'~(?![01])', pointer)):
            continue
        node = tree
        for token in pointer.split('/')[1:]:
            node = node.setdefault(token.replace('~1', '/').replace('~0', '~'), {})
        node[None] = pointer
    if not tree:
        return {}
    source = Path(path).read_text(encoding='utf-8')
    decoder = json.JSONDecoder()
    skip = lambda pos: json.decoder.WHITESPACE.match(source, pos).end()
    positions = {}

    def visit(pos, node, anchor=None):
        pos = skip(pos)
        if None in node:
            positions[node[None]] = pos if anchor is None else anchor
        if not any(key is not None for key in node) or source[pos:pos + 1] not in ('{', '['):
            return decoder.raw_decode(source, pos)[1]
        is_object = source[pos] == '{'
        close = '}' if is_object else ']'
        pos, index = skip(pos + 1), 0
        while source[pos:pos + 1] != close:
            start = pos
            if is_object:
                key, pos = decoder.raw_decode(source, pos)
                pos = skip(pos)
                if not isinstance(key, str) or source[pos:pos + 1] != ':':
                    raise ValueError('Invalid JSON object member.')
                pos = skip(pos + 1)
            else:
                key = str(index)
                index += 1
            child = node.get(key)
            if child is None:
                pos = decoder.raw_decode(source, pos)[1]
            else:
                # JSON decoding keeps the last duplicate key, including its
                # entire subtree; do not cite descendants of an older value.
                pending = [child]
                while pending:
                    item = pending.pop()
                    positions.pop(item.get(None), None)
                    pending.extend(value for key, value in item.items() if key is not None)
                pos = visit(pos, child, start)
            pos = skip(pos)
            if source[pos:pos + 1] == close:
                break
            if source[pos:pos + 1] != ',':
                raise ValueError('Invalid JSON separator.')
            pos = skip(pos + 1)
        return pos + 1

    end = visit(0, tree)
    if skip(end) != len(source):
        raise ValueError('Unexpected content after JSON.')
    lines, previous, line = {}, 0, 1
    for pointer, position in sorted(positions.items(), key=lambda item: item[1]):
        line += source.count('\n', previous, position)
        lines[pointer] = line
        previous = position
    return lines


def link_references(document, ptx, artifacts, report_path=None, artifact_base=None):
    """Link displayed citations after composition; never alter quoted evidence."""
    parts = re.split(r'(<pre\b[^>]*>.*?</pre>|<a\b[^>]*>.*?</a>|^```[^\n]*\n.*?^```[ \t]*$)',
                     document, flags=re.S | re.M)
    references = re.compile(r'<code>((?:input|output)\.json#(?:/[^<]*)?)</code>|`((?:input|output)\.json#(?:/[^`]*)?)`')
    files = dict(zip(('input.json', 'output.json'), artifacts[:2]))
    targets = {}
    for name, path in files.items():
        pointers = {html.unescape(match[1] or match[2]).split('#', 1)[1]
                    for part in parts[::2] for match in references.finditer(part)
                    if html.unescape(match[1] or match[2]).startswith(name + '#')}
        for pointer, line in _json_pointer_lines(path, pointers).items():
            targets[name + '#' + pointer] = artifact_url(path, report_path, artifact_base, line)

    def anchor(label, target):
        return '<a href="' + html.escape(target, quote=True) + '">' + label + '</a>'

    def reference(match):
        label = html.unescape(match[1] or match[2])
        target = targets.get(label)
        return anchor(_inline(label), target) if target else match[0]

    def location(match):
        line = _matching_line({'file': html.unescape(match[1]), 'line': int(match[2])}, ptx, artifacts)
        target = artifact_url(artifacts[2], report_path, artifact_base, line) if line else None
        return anchor(match[0], target) if target else match[0]

    def caption(match):
        line = int(match[2])
        if not 1 <= line <= len(ptx.splitlines()) or len(artifacts) < 3:
            return match[0]
        target = artifact_url(artifacts[2], report_path, artifact_base, line)
        return match[1] + anchor('physical PTX ' + match[2], target) + (match[3] or '') + '</p>' if target else match[0]

    for index in range(0, len(parts), 2):
        part = references.sub(reference, parts[index])
        part = re.sub(r'<code>([^<]+)</code> · source line <code>(\d+)</code>', location, part)
        parts[index] = re.sub(r'(<p><strong>[^\n]*</strong> · )physical PTX (\d+)( · display wrapping added)?</p>', caption, part)
    return ''.join(parts)


def artifact_entry(result, artifacts, report_path=None, source_paths=(), equivalence_path=None, artifact_base=None):
    if not artifacts:
        return ''
    inp, out, ptx = (list(artifacts) + [None] * 3)[:3]
    ptx_link = _file_link('PTX', ptx, report_path, artifact_base)
    target = artifact_url(ptx, report_path, artifact_base)
    if target:
        url = urlsplit(target)
        pieces = url.path.split('/')
        if url.hostname == 'github.com' and len(pieces) > 5 and pieces[3] == 'blob':
            pieces[3] = 'raw'
            ptx_link += ' ([raw](https://github.com' + '/'.join(pieces) + '))'
    body = '**Inputs:** ' + ptx_link + ' · analyzed code; '
    body += _file_link('input.json', inp, report_path, artifact_base) + ' · launch, arguments and assumptions.\n\n'
    body += '**Result:** ' + _file_link('output.json', out, report_path, artifact_base) + ' · saved verifier findings.\n\n'
    directory = (
        ('Check the finding', (
            ('diagnostics', 'Diagnostic messages and details'),
            ('checks', 'Individual check outcomes'),
            ('failed_required_orders', 'Failed lifecycle orders'),
            ('rejection', 'Rejected synchronization requirement'),
            ('counterexample', 'Saved model replay'),
            ('source_events', 'Event-to-source mappings'))),
        ('Check how the model was built', (
            ('frontend', 'PTX translation'),
            ('semantic_program', 'Model and source bindings'),
            ('synchronization_input', 'Synchronization-check setup'))),
        ('Check values separately', (
            ('value_analysis', 'Value-analysis status'),
            ('equivalence', 'Numerical comparison'))))
    # Public outputs save some fields under `internal` or `evidence`.
    locations = {key: saved_pointer(result, '/' + key) for _, fields in directory for key, _ in fields if key in result}
    field_lines = _json_pointer_lines(out, set(locations.values()))
    groups = []
    for heading, fields in directory:
        records = []
        for key, purpose in fields:
            if key not in result:
                continue
            if locations[key] in field_lines:
                records.append('- ' + _file_link(purpose + ' · `' + key + '`', out, report_path, artifact_base, field_lines[locations[key]]))
            else:
                records.append('- ' + purpose + ': ' + _ref(locations[key]) + ' (source line unavailable)')
        if records:
            groups.append('**' + heading + '**\n\n' + '\n'.join(records))
    extra = []
    for path in source_paths:
        extra.append('- Source: ' + _file_link(Path(path).name, path, report_path, artifact_base))
    if equivalence_path is not None:
        extra.append('- Separate numerical-check artifact: ' + _file_link('equivalence.json', equivalence_path, report_path, artifact_base) + '. It does not replace the memory/synchronization verdict above.')
    if groups or extra:
        detail = 'Each record link opens its field in the original `output.json`. Model summaries are not full execution traces.\n\n'
        detail += '\n\n'.join(groups)
        if extra:
            detail += '\n\n' + '\n'.join(extra)
        detail += '\n\nFor the records explained in this report, see the [Evidence appendix](#evidence-appendix).'
        body += _fold('Analysis records — open the saved evidence', detail)
    return body


def _diagram(nodes, edges, caption, focus=None, direction='LR'):
    # Labels are escaped to Mermaid entities; diagram text cannot supply syntax.
    def label(value):
        return html.escape(str(value), quote=True).replace('`', '&#96;').replace('\n', '<br/>')
    body = ['```mermaid', 'flowchart ' + direction]
    body.extend(f'  {key}["{label(value)}"]' for key, value in nodes.items())
    body.extend('  ' + edge for edge in edges)
    body += ['  classDef context fill:#f6f8fa,stroke:#8c959f,color:#24292f',
             '  classDef operation fill:#ddf4ff,stroke:#54aeff,color:#0550ae',
             '  classDef focus fill:#fbefff,stroke:#c297ff,color:#6639ba']
    for key in nodes:
        kind = 'focus' if key == focus else 'context' if key == 'R' else 'operation'
        body.append(f'  class {key} {kind}')
    body += ['```', '', caption, '', '']
    return '\n'.join(body)


def _location(location):
    item = _mapping(location)
    if not item:
        return 'source location not saved'
    file = item.get('file', MISSING)
    line = item.get('line', MISSING)
    column = item.get('column', MISSING)
    return _inline(file) + ' · source line ' + _inline(line) + (', column ' + _inline(column) if column is not MISSING else '')


def _brief_location(location, ptx, artifacts):
    line = _matching_line(location, ptx, artifacts)
    if line is not None:
        return 'PTX line ' + _inline(line)
    item = _mapping(location)
    return 'source line ' + _inline(item['line']) + ' · PTX mapping unconfirmed' if 'line' in item else 'source location not saved'


def _matching_line(location, ptx, artifacts):
    """Never interpret an instruction/event ID or another source file as PTX."""
    loc = _mapping(location)
    saved, line = loc.get('file'), loc.get('line')
    if not isinstance(saved, str) or not saved.endswith('.ptx') or type(line) is not int:
        return None
    target = Path(artifacts[2]) if len(artifacts) > 2 and artifacts[2] else None
    if target is None or not 1 <= line <= len(ptx.splitlines()):
        return None
    source = Path(saved)
    same = source == target or source.is_absolute() and source.resolve() == target.resolve()
    if not source.is_absolute() and len(source.parts) > 1:
        same = same or target.parts[-len(source.parts):] == source.parts
    if source.name == saved:
        same = same or target.name == saved
    return line if same else None


def _snippets(locations, ptx, artifacts):
    selected, missing = [], []
    for label, location in locations:
        line = _matching_line(location, ptx, artifacts)
        if line is None:
            missing.append(label)
        elif line not in [number for _, number in selected]:
            selected.append((label, line))
    body = ('**Recorded source anchors.** These lines locate the saved events. Their guards and surrounding control flow belong to the code explanation.\n\n' if selected else '')
    for label, line in selected[:3]:
        body += excerpt(ptx, str(line), caption=label + ' · saved source location') + '\n\n'
    if missing:
        body += '**PTX mapping:** ' + ', '.join(html.escape(label) for label in missing) + ': no matching physical PTX location is established by the saved record.\n\n'
    return body


def _source_operations(locations, ptx, artifacts):
    """Read opcodes only at file-matched physical lines; never guess an ID mapping."""
    selected = {_matching_line(location, ptx, artifacts) for location in locations}
    selected.discard(None)
    if not selected:
        return {}
    lines = mask_noncode(ptx).splitlines()
    return {line: ', '.join(lines[line - 1][first:last]
                           for first, last in instruction_opcodes(lines[line - 1]))
            for line in selected}


def _operation(location, operations, ptx, artifacts):
    line = _matching_line(location, ptx, artifacts)
    opcode = operations.get(line)
    return (_inline(opcode) + '<br/>' if opcode else '') + _brief_location(location, ptx, artifacts)


def _operation_node(role, location, operations, ptx, artifacts):
    line = _matching_line(location, ptx, artifacts)
    opcode = operations.get(line)
    # The complete opcode stays beside its model IDs in the lookup table.
    if opcode and len(opcode) > 48:
        opcode = '.'.join(opcode.split('.')[:2]) + '…'
    return role + ('\n' + opcode if opcode else '') + ('\nPTX ' + str(line) if line else '\nsource not mapped')


def _coverage(result):
    memory = _mapping(result.get('memory_analysis'))
    items = []
    for label, value in [('Memory analysis complete', result.get('memory_analysis_complete', MISSING)),
                         ('Search complete', result.get('search_complete', MISSING)),
                         ('Exhaustive search', memory.get('exhaustive', MISSING))]:
        if value is True:
            wording = 'yes'
        elif value is False:
            wording = '**no**'
        elif value is None:
            wording = '**unknown** (`null`)'
        else:
            wording = _inline(value)
        items.append(label + ': ' + wording)
    body = '**Coverage:** ' + ' · '.join(items) + '.\n\n'
    if result.get('memory_analysis_complete') is True and result.get('search_complete') is False:
        body += 'The memory analysis returned a result, but its search is not complete. This finding is available to inspect; the flags do not establish that all possible conflicts were examined.\n\n'
    if 'assumption_dependent' in result:
        body += '**Assumption-dependent:** ' + _inline(result['assumption_dependent']) + '.'
        active = [a for a in _records(result.get('assumptions')) if isinstance(a, dict) and a.get('enabled') is True and a.get('triggered') is True]
        labels = []
        for rule in active:
            label = _text(rule['label']) if isinstance(rule.get('label'), str) else _inline(rule.get('id', rule.get('key', MISSING)))
            if 'id' in rule and isinstance(rule.get('key'), str):
                label += ' (' + _text(rule['key'].replace('_', ' ')) + ')'
            labels.append(label)
        body += (' Triggered rules: ' + ', '.join(labels) + '.' if labels else '')
        body += '\n\n'
    return body


def _pair(result):
    # A warning must never masquerade as the pair supporting a race verdict.
    keys = {'race_candidate': ('races',), 'race_free': ('warnings',),
            'memory_model_unsupported': ('unsupported_memory_pair',)}
    status = _view_kind(result)
    for key in keys.get(status if isinstance(status, str) else '', ()):
        value = result.get(key)
        if isinstance(value, dict) and 'first' in value:
            return key, value
        for index, record in enumerate(_records(value)):
            if isinstance(record, dict) and ('first' in record or 'second' in record):
                return key + '/' + str(index), record
    return None, None


def _access_view(pair, location, ptx, artifacts):
    """`location` is the pair's saved output.json pointer."""
    first, second, region = (_mapping(pair.get(key)) for key in ('first', 'second', 'region'))
    body = ''
    if region:
        body += '**' + html.escape(_plain(first.get('mode', MISSING))) + ' / ' + html.escape(_plain(second.get('mode', MISSING))) + ' overlap:** ' + _inline(region.get('bytes', MISSING)) + ' bytes in ' + _inline(region.get('address_space', MISSING)) + ' memory, at byte offset ' + _inline(region.get('offset', MISSING)) + '.\n\n'
    if 'message' in pair:
        body += _text(pair['message']) + '\n\n'
    if 'reason' in pair:
        body += '**Reason:** ' + _text(pair['reason']) + '\n\n'
    locations = [('First access', first.get('location')), ('Second access', second.get('location'))]
    operations = _source_operations([location for _, location in locations], ptx, artifacts)
    nodes = {}
    for key, access, name in [('A', first, 'First access'), ('B', second, 'Second access')]:
        nodes[key] = _operation_node(name + '\n' + _plain(access.get('mode', MISSING)) + ' · ' + _plain(access.get('bytes', MISSING)) + ' bytes', access.get('location'), operations, ptx, artifacts)
    if region:
        offset, count = region.get('offset'), region.get('bytes')
        span = f'bytes {offset}–{offset + count - 1}' if type(offset) is int and type(count) is int and count > 0 else 'offset ' + _plain(region.get('offset', MISSING)) + ' · bytes ' + _plain(region.get('bytes', MISSING))
        allocation = _plain(region.get('allocation', MISSING))
        nodes['R'] = (allocation if len(allocation) <= 30 else 'Recorded overlap') + '\n' + span
        body += _diagram(nodes, ['A --- R', 'B --- R'], 'Lines connect the two recorded accesses to their shared byte region; they do not show execution order.', focus='R')
    else:
        body += _diagram(nodes, ['A --- B'], 'The line identifies the pair selected by the verifier; no overlap range or execution order is inferred.')
    if any(_matching_line(location, ptx, artifacts) is None for _, location in locations):
        body += '\n**Source attribution:** this pair is not fully mapped to physical PTX; model IDs do not identify CUDA threads.\n\n'
    detail = '| Access | Source operation | Model event / instruction | Actor ID |\n| --- | --- | --- | --- |\n'
    for label, access in [('First access', first), ('Second access', second)]:
        detail += '| ' + label + ' | ' + _operation(access.get('location'), operations, ptx, artifacts) + ' | ' + _inline(access.get('event', MISSING)) + ' / ' + _inline(access.get('instruction', MISSING)) + ' | ' + _inline(access.get('actor', MISSING)) + ' |\n'
    detail += '\nActor IDs name verifier participants; this record does not itself establish CUDA thread coordinates. Event and instruction IDs identify model records, not source-line numbers.\n\n'
    detail += _snippets(locations, ptx, artifacts)
    detail += '**Saved record:** ' + _ref(location) + '. '
    if 'overlap' in pair:
        detail += '**Overlap check:** ' + _text(pair['overlap']) + '. '
    if region:
        detail += '**Allocation:** ' + _inline(region.get('allocation', MISSING)) + '. **Address space:** ' + _inline(region.get('address_space', MISSING)) + '.'
    return body + '\n' + _fold('Locate the accesses and check their recorded overlap', detail)


def _source_map(result):
    return {item['event']: item for item in _records(result.get('source_events')) if isinstance(item, dict) and type(item.get('event')) is int}


def _sync_view(result, ptx, artifacts):
    rejection = _mapping(result.get('rejection'))
    witness = _mapping(result.get('counterexample'))
    phase = _mapping(rejection.get('phase'))
    required = _mapping(rejection.get('required_events'))
    events = _records(required.get('events'))
    issue, stopped = rejection.get('issue', MISSING), witness.get('first_unexecuted', MISSING)
    mapping = _source_map(result)
    known_check = rejection.get('code', result.get('rejection_kind')) == 'issue_not_ordered_before_next_phase'
    focus = [('Wait issue' if known_check else 'Issue', issue)] + [('Phase effect' if known_check else 'Required event', event) for event in events[:3]] + [('First unexecuted', stopped)]
    locations = [(label + ' event ' + _plain(event), mapping.get(event, {}) if type(event) is int else {}) for label, event in focus]
    operations = _source_operations([location for _, location in locations], ptx, artifacts)
    body = ''
    check = rejection.get('code', result.get('rejection_kind', MISSING))
    if check == 'issue_not_ordered_before_next_phase':
        body += 'This finding concerns **barrier reuse**. The model could not establish that the wait is issued before at least one completion effect of the next barrier phase. The failed requirement is shown below.\n\n'
    else:
        body += _text(rejection.get('message', result.get('message', MISSING))) + '\n\n'
    issue_source = mapping.get(issue, {}) if type(issue) is int else {}
    effect_source = mapping.get(events[0], {}) if events and type(events[0]) is int else {}
    nodes = {'I': _operation_node('Issue', issue_source, operations, ptx, artifacts),
             'W': 'Rejected check\nbarrier ' + _plain(phase.get('barrier', MISSING)) + ' · generation ' + _plain(phase.get('generation', MISSING)),
             'R': _operation_node('Required event', effect_source, operations, ptx, artifacts) if len(events) == 1 else str(len(events)) + ' retained required events'}
    if check == 'issue_not_ordered_before_next_phase':
        # wait_checker.cpp uses FindDescendant(issue_id, next_info->effects).
        # This named requirement points from issue to a next-phase effect.
        nodes['I'] = nodes['I'].replace('Issue', 'Wait issue', 1)
        nodes['R'] = nodes['R'].replace('Required event', 'Next-phase completion effect').replace('required events', 'next-phase effects')
        body += '**Checked phase:** barrier ' + _text(phase.get('barrier', MISSING)) + ' · generation ' + _text(phase.get('generation', MISSING)) + '.\n\n'
        body += _diagram({key: nodes[key] for key in ('I', 'R')}, ['I -. "required before: not established" .-> R'], 'The dashed arrow names the failed requirement: issue before at least one listed phase effect. It does not show an observed order.', focus='R', direction='TB')
    else:
        body += _diagram(nodes, ['I --- W', 'R --- W'], 'Links associate the issue and required events with the saved rejection. They are not happens-before edges or execution order.', focus='W')
    if required.get('truncated') is True:
        body += '**Required-event list truncated:** ' + str(len(events)) + ' retained of ' + _number(required.get('total', MISSING)) + '.\n\n'
    if any(_matching_line(location, ptx, artifacts) is None for _, location in locations):
        body += '**Source attribution:** some model events have no confirmed physical PTX mapping.\n\n'
    primary = _mapping((_records(result.get('diagnostics')) or [{}])[0])
    check_label = 'requirement details' if 'result' in result and primary.get('code') == check else _inline(check)
    detail = '**Saved check:** ' + check_label + ' · ' + _cite(result, '/rejection') + '.\n\n'
    if check == 'issue_not_ordered_before_next_phase':
        detail += 'The listed required events are next-phase completion effects, not predecessors that the wait must follow.\n\n'
    detail += '| Role | Source operation | Model event / instruction |\n| --- | --- | --- |\n'
    for label, event in focus:
        source = mapping.get(event, {}) if type(event) is int else {}
        detail += '| ' + label + ' | ' + _operation(source, operations, ptx, artifacts) + ' | ' + _inline(event) + ' / ' + _inline(source.get('instruction', MISSING)) + ' |\n'
    if len(events) > 3:
        detail += '\n' + str(len(events) - 3) + ' additional retained required events are in Evidence appendix → Diagnostic records.\n'
    detail += '\nEvent and instruction IDs are model identifiers; only the separately saved source location identifies a file and line.\n\n'
    detail += _snippets(locations, ptx, artifacts)
    if witness:
        prefix = _mapping(witness.get('executed_prefix'))
        retained = _records(prefix.get('events'))
        total = prefix.get('total', MISSING)
        body += '**Model replay — not an observed GPU trace:** ' + str(len(retained)) + ' event IDs retained from an executed prefix of ' + _number(total) + '. First unexecuted event: ' + _inline(stopped) + '.\n\n'
        if prefix.get('truncated') is True:
            omitted = total - len(retained) if type(total) is int and total >= len(retained) else MISSING
            body += '**Replay gap: ' + _number(omitted) + ' events not listed**; do not join the retained prefix directly to the stop point.\n\n'
        other_truncated = [key for key, sequence in witness.items() if key != 'executed_prefix' and isinstance(sequence, dict) and sequence.get('truncated') is True]
        if other_truncated:
            body += '**Other truncated sequences:** ' + ', '.join(html.escape(key.replace('_', ' ')) for key in other_truncated) + '.\n\n'
        detail += 'The saved witness does not map its events to CUDA participant columns. Exact retained sequences, candidate order and source mappings: Evidence appendix → Diagnostic records.\n\n'
    return body + _fold('Locate the wait, phase effect and replay stop', detail)


def _multiple_lifecycle_view(result):
    categories = []
    for index, raw in enumerate(_records(result.get('diagnostics'))):
        item = _mapping(raw)
        details = _mapping(item.get('details'))
        if item.get('stage') == 'lifecycle_check' and isinstance(details.get('constraint'), str):
            categories.append((index, item, details))
    if len({details['constraint'] for _, _, details in categories}) < 2:
        return ''
    body = '**Different lifecycle requirements are reported.** Each relation below is a requirement the model did not establish, not an observed execution order.\n\n'
    roles = {
        'access_completion_before_dealloc': ('One allocation', 'Access completes', 'Deallocate that allocation'),
        'tmem_group2_access_before_dealloc': ('Paired TMEM allocations', 'All accesses to both\nallocations complete', 'Either warp attempts\ndeallocation'),
    }
    detail = ''
    for index, item, saved in categories:
        kind = saved['constraint']
        heading, source, target = roles.get(kind, (kind.replace('_', ' '), 'Required predecessor', 'Required successor'))
        witnesses = _records(saved.get('witnesses'))
        first = _mapping(witnesses[0]) if witnesses else {}
        body += '#### ' + _text(heading) + '\n\n' + _text(item.get('message', MISSING)) + '\n\n'
        body += _diagram({'F': source, 'T': target}, ['F -. "required" .-> T'],
                         'Dashed arrow: required ordering, not established by this run.', focus='T', direction='LR')
        retained = str(len(witnesses)) if isinstance(saved.get('witnesses'), list) else _inline(saved.get('witnesses', MISSING))
        body += '**This category:** ' + _number(saved.get('count', MISSING)) + ' failures reported; ' + retained + ' witnesses retained; truncated: ' + _inline(saved.get('witnesses_truncated', MISSING)) + '.\n\n'
        detail += '#### ' + _text(heading) + '\n\n**Requirement:** ' + _inline(kind) + ' · ' + _cite(result, '/diagnostics/' + str(index) + '/details') + '.\n\n'
        if first:
            detail += 'First saved witness: model instruction ' + _inline(first.get('from_instruction', MISSING)) + ' → ' + _inline(first.get('to_instruction', MISSING)) + '. The arrow denotes the required order.\n\n'
        else:
            detail += 'No usable witness is saved for this category.\n\n'
    if 'failed_required_orders' in result:
        records = result['failed_required_orders']
        retained = str(len(records)) if isinstance(records, list) else _inline(records)
        body += '**Top-level sample:** ' + retained + ' orders retained out of ' + _number(result.get('failed_required_orders_count', MISSING)) + ' reported; truncated: ' + _inline(result.get('failed_required_orders_truncated', MISSING)) + '. It need not include every category; the per-category witnesses above come from `diagnostics[].details`.\n\n'
    detail += 'Instruction IDs identify model records, not physical PTX lines or CUDA issuers. All retained witnesses and effect lists remain in Evidence appendix → Diagnostic records.\n\n'
    return body + _fold('Check the separate requirements and their witnesses', detail)


def _lifecycle_view(result):
    multiple = _multiple_lifecycle_view(result)
    if multiple:
        return multiple
    if not any(key in result for key in ('failed_required_orders', 'failed_required_orders_count')):
        return ''
    records = _records(result.get('failed_required_orders'))
    first = _mapping(records[0]) if records else {}
    body = ''
    body += _text(first.get('message', _mapping(result.get('error')).get('message', MISSING))) + '\n\n'
    if first:
        kind = first.get('kind')
        roles = {'access_completion_before_dealloc': ('Access completion', 'Allocation deallocation')}
        source, target = roles.get(kind, ('Required predecessor', 'Required successor'))
        if kind == 'access_completion_before_dealloc' and str(first.get('message', '')).startswith('tcgen05.load must complete'):
            source, target = 'TMEM read completes', 'Free TMEM allocation'
        nodes = {'F': source, 'T': target}
        body += _diagram(nodes, ['F -. "required before: not established" .-> T'], 'The dashed arrow is a failed requirement in the analyzed model, not a recorded execution sequence.', focus='T', direction='TB')
    body += '**Failed orders:** ' + _number(result.get('failed_required_orders_count', MISSING)) + ' reported; ' + str(len(records)) + ' records retained; truncated: ' + _inline(result.get('failed_required_orders_truncated', MISSING)) + '.\n\n'
    body += '**Source attribution:** the retained orders do not map these model instructions to physical PTX sites or CUDA issuers.\n\n'
    if first:
        detail = '**Requirement:** ' + _inline(first.get('kind', MISSING)) + ' · ' + _cite(result, '/failed_required_orders/0') + '.\n\n'
        detail += '| Required role | Model instruction |\n| --- | --- |\n| ' + source + ' | ' + _inline(first.get('from_instruction', MISSING)) + ' |\n| ' + target + ' | ' + _inline(first.get('to_instruction', MISSING)) + ' |\n\n'
        detail += 'Instruction IDs refer to model instructions, not physical PTX lines. The operation roles above come from the saved requirement and message. All retained orders and effect lists are in Evidence appendix → Diagnostic records.\n'
        body += _fold('Check the required order and retained instruction IDs', detail)
    return body


def _stopped_view(result, ptx, artifacts):
    body = ''
    for key, label in [('active_stage', 'Stage'), ('failure_stage', 'Stage'), ('reason', 'Reason'), ('message', 'Message'), ('error', 'Error')]:
        if key in result:
            value = result[key]
            body += '**' + label + ':** ' + (_text(value.get('message', value)) if isinstance(value, dict) else _text(value)) + '\n\n'
    if 'location' in result:
        body += '**Saved source location:** ' + _location(result['location']) + '.\n\n'
        body += _snippets([('Stop location', result['location'])], ptx, artifacts)
    if 'result' not in result:
        body += _primary_diagnostic(result, ptx, artifacts)
    return body


def _diagnostic_groups(records):
    """Group saved categories for navigation, never as inferred root causes."""
    groups = {}
    for index, raw in enumerate(records):
        item = _mapping(raw)
        key = (_plain(item.get('stage', MISSING)), _plain(item.get('code', MISSING)))
        group = groups.setdefault(key, {'indices': [], 'messages': {}, 'lines': set(), 'counts': []})
        group['indices'].append(index)
        message = _plain(item.get('message', item.get('reason', raw)))
        group['messages'][message] = group['messages'].get(message, 0) + 1
        details = _mapping(item.get('details'))
        line = item.get('ptx_line', details.get('ptx_line'))
        if type(line) is int:
            group['lines'].add(line)
        if 'count' in details:
            group['counts'].append(details['count'])
    return groups


def _diagnostic_overview(groups, *, detailed=False):
    body = ''
    if not detailed:
        body += '**' + str(len(groups)) + ' diagnostic categories · ' + _number(sum(len(g['indices']) for g in groups.values())) + ' retained records.**\n\n'
        body += 'Grouped by saved stage and code, not by inferred root cause or severity. Record counts are not counts of independent bugs.\n\n'
        body += '| Saved diagnostic | Records | Reported extent |\n| --- | ---: | --- |\n'
    for (stage, code), group in groups.items():
        message = next(iter(group['messages']))
        if detailed:
            body += '##### ' + _text(code.replace('_', ' ')) + '\n\n'
            body += '**Saved code:** ' + _inline(code) + ' · **Stage:** ' + _inline(stage) + '.\n\n'
            body += _text(message) + '\n\n'
            body += '**Diagnostic array indices:** ' + _id_set(group['indices']) + '.\n\n'
            if group['lines']:
                body += '**Reported PTX lines:** ' + _id_set(group['lines']) + '. These diagnostic coordinates do not by themselves establish a source-file binding.\n\n'
            if group['counts']:
                body += '**Saved `details.count` values:** ' + ', '.join(_inline(x) for x in group['counts']) + '.\n\n'
            if len(group['messages']) > 1:
                body += '**Message variants:** ' + _number(len(group['messages'])) + '; all variants remain in the exact records below.\n\n'
        else:
            if len(group['lines']) == 1:
                extent = 'PTX ' + _inline(next(iter(group['lines'])))
            elif group['lines']:
                extent = str(len(group['lines'])) + ' reported PTX sites'
            elif len(group['counts']) == 1:
                extent = _number(group['counts'][0]) + ' (`details.count`)'
            else:
                extent = 'See saved records'
            if len(group['messages']) > 1:
                message += ' (' + str(len(group['messages'])) + ' message variants)'
            meanings = {
                ('event_view_construction', 'cluster_barrier_arrive_repeated_without_wait'): 'Cluster protocol',
                ('event_view_construction', 'cluster_wait_without_arrive'): 'Cluster protocol',
                ('event_view_construction', 'missing_tma_descriptor'): 'Tensor-map capture missing',
                ('event_view_construction', 'async_copy_source_control_not_static'): 'Model needs a fixed copy-control value',
            }
            meaning = meanings.get((stage, code))
            label = ('**' + meaning + '**<br/>' if meaning else '') + _text(message).replace('\n', ' ')
            body += '| ' + label + ' | ' + str(len(group['indices'])) + ' | ' + extent + ' |\n'
    return body + '\n'


def _primary_diagnostic(result, ptx, artifacts):
    diagnostics = result.get('diagnostics', MISSING)
    if not isinstance(diagnostics, list):
        return '**Diagnostics:** ' + _inline(diagnostics) + '.\n\n' if 'result' in result else ''
    if not diagnostics:
        return ''
    groups = _diagnostic_groups(diagnostics)
    if len(groups) > 1:
        return _diagnostic_overview(groups) + 'Category codes, locations and array indices: Evidence appendix → Diagnostic records.\n\n'
    first = _mapping(diagnostics[0])
    code = first.get('code', MISSING)
    body = '**First diagnostic:** ' + _inline(code)
    if 'stage' in first:
        body += ' · ' + _inline(first['stage'])
    message = first.get('message', MISSING)
    # The synchronization view explains this named check, and the lifecycle
    # view already quotes its exact requirement. Keep the code once, below.
    explained = code == _mapping(result.get('rejection')).get('code') or message == _mapping((_records(result.get('failed_required_orders')) or [{}])[0]).get('message')
    if message is not MISSING and not explained:
        body += ' — ' + _text(message)
    body += ' · ' + _cite(result, '/diagnostics/0') + '.\n\n'
    details = _mapping(first.get('details'))
    location = first.get('location', details.get('location', MISSING))
    if location is not MISSING and 'location' not in result:
        body += '**Saved diagnostic location:** ' + _location(location) + '.\n\n'
        body += _snippets([('Diagnostic location', location)], ptx, artifacts)
    line = first.get('ptx_line', details.get('ptx_line', MISSING))
    if line is not MISSING:
        body += '**Reported PTX line:** ' + _inline(line) + '; the diagnostic does not save a source-file mapping.\n\n'
    if len(diagnostics) > 1:
        body += '**Retained diagnostics:** ' + _number(len(diagnostics)) + '; all records are in the [Evidence appendix](#evidence-appendix).\n\n'
    return body


def render_result(run_input, result, ptx, artifacts, title=None, *, report_path=None, source_paths=(), equivalence_path=None, artifact_base=None):
    """Render only saved outcomes and relationships; source explanation is separate."""
    saved = _outcome(result)
    status = _view_kind(result)
    vocabulary = RESULTS if 'result' in result else STATUSES
    headline, description = vocabulary.get(saved if isinstance(saved, str) else '', ('RESULT NOT RECOGNIZED', 'No success verdict is inferred from a missing, null or unrecognized status.'))
    lifecycle_diagnostics_only = saved == 'invalid_lifecycle' and not any(key in result for key in ('failed_required_orders', 'failed_required_orders_count'))
    if lifecycle_diagnostics_only:
        headline = 'LIFECYCLE CHECK FAILED'
        description = 'Lifecycle checking did not complete successfully. The saved diagnostics identify the reported problems.'
    warnings = _records(result.get('warnings'))
    count = result.get('warning_count')
    if status == 'race_free' and (warnings or type(count) is int and count > 0):
        headline += '<br/>' + (_number(count) + ' WARNINGS' if type(count) is int and count > 0 else 'WARNINGS RETAINED')
    kernel = title or (Path(artifacts[2]).stem if len(artifacts) > 2 and artifacts[2] else 'Kernel report')
    pointer, pair = _pair(result)
    if status == 'synchronization_candidate':
        finding_title = 'Synchronization requirement'
    elif status == 'unsafe_lifecycle':
        finding_title = 'Lifecycle diagnostics' if lifecycle_diagnostics_only else 'Required lifecycle order'
    elif pair:
        finding_title = 'Warning to inspect' if status == 'race_free' else 'Potentially conflicting accesses' if status == 'race_candidate' else 'Unresolved access relation'
    elif saved == 'equivalence_not_proven':
        finding_title = 'Numerical comparison'
    elif status not in ('race_free', 'race_candidate'):
        finding_title = 'Where analysis stopped'
    else:
        finding_title = 'Recorded outcome'
    body = FACTS_START + '\n<p align="center">' + html.escape(kernel) + '</p>\n<h1 align="center">' + headline + '</h1>\n\n'
    body += artifact_entry(result, artifacts, report_path, source_paths, equivalence_path, artifact_base)
    body += '## 1. Verifier result\n\nSaved findings and analysis scope, formatted directly from the verifier output.\n\n'
    body += '### 1.1 ' + finding_title + '\n\n'
    body += description + (' Saved status: ' + _inline(saved) + '.' if 'result' not in result and (not isinstance(saved, str) or saved not in STATUSES) else '') + '\n\n'
    if 'result' in result:
        body += '**Saved result:** ' + _inline(saved) + ' · ' + _cite(result, '/result') + '.\n\n' + _check_summary(result)
        operation = _mapping(result.get('scope')).get('operation', MISSING)
        if operation is not MISSING:
            body += '**Operation:** ' + _inline(operation) + '.\n\n'
        elif saved == 'verified':
            body += '**Scope:** not saved. The label alone does not identify which operation was verified.\n\n'
        body += _result_consistency(result)
        body += _primary_diagnostic(result, ptx, artifacts)
    if 'analysis_target' in result or 'recovery_policy' in result:
        body += '**Analyzed target:** ' + _inline(result.get('analysis_target', MISSING)) + ' · **Recovery policy:** ' + _inline(result.get('recovery_policy', MISSING)) + '.\n\n'
        if result.get('analysis_target') == 'recovered_program':
            body += 'This verdict describes the recovered model, not the original program without recovery. Original violations remain in Evidence.\n\n'
    qualifications = ''
    if any(isinstance(item, dict) and item.get('code') == 'duplicate-tma-store' for item in warnings):
        rule = next((item for item in _records(result.get('assumptions')) if isinstance(item, dict) and (item.get('key') == 'duplicate_tma_store_benign' or item.get('id') == 'A002' and 'key' not in item)), None)
        if rule:
            label = _text(rule['label']) if isinstance(rule.get('label'), str) else _inline(rule.get('id', rule.get('key', MISSING)))
            qualifications += '**Warning classification rule:** ' + label + ' — ' + _text(rule.get('text', MISSING)) + '\n\n'
            qualifications += 'Saved rule flags: enabled ' + _inline(rule.get('enabled', MISSING)) + ' · triggered ' + _inline(rule.get('triggered', MISSING)) + '.\n\n'
    summary = ''
    if status in ('race_free', 'race_candidate', 'memory_model_unsupported') and (saved != 'unsupported' or any(key in result for key in ('race_count', 'warning_count', 'unresolved_count'))):
        summary += '**Recorded totals:** ' + _number(result.get('race_count', MISSING)) + ' races · ' + _number(result.get('warning_count', MISSING)) + ' warnings · ' + _number(result.get('unresolved_count', MISSING)) + ' unresolved.\n\n'
        for key in ('races', 'warnings'):
            retained = len(_records(result.get(key)))
            truncated = result.get(key[:-1] + '_samples_truncated', MISSING)
            if truncated is True or retained and result.get(key[:-1] + '_count') != retained:
                summary += '**Retained ' + key + ':** ' + str(retained) + '; samples truncated: ' + _inline(truncated) + '.\n\n'
    body += summary + qualifications
    if pair:
        category = pointer.split('/')[0]
        retained = len(_records(result.get(category)))
        if category == 'warnings':
            body += '**Why inspect this warning?** The saved classification is a warning, not a race. It remains relevant to interpreting the no-race result.\n\n'
        elif category == 'races':
            body += 'The pair below is a race candidate to inspect, not an observed GPU failure.\n\n'
        if retained > 1:
            body += 'Showing the first of ' + _number(retained) + ' retained ' + category + '; this is file order, not a severity ranking. All retained records are in the [Evidence appendix](#evidence-appendix).\n\n'
    if status == 'synchronization_candidate':
        body += _sync_view(result, ptx, artifacts)
    elif status == 'unsafe_lifecycle':
        body += _lifecycle_view(result)
    elif pair:
        body += _access_view(pair, saved_pointer(result, '/' + pointer), ptx, artifacts)
    elif status not in ('race_free', 'race_candidate'):
        body += _stopped_view(result, ptx, artifacts)
    elif status == 'race_candidate':
        body += '**Access records:** no usable retained pair is saved; the candidate status is preserved.\n\n'
    checks = _mapping(result.get('checks'))
    # Two-check outputs include the memory analysis in the synchronization check.
    memory_check = 'memory' if 'memory' in checks else 'synchronization'
    if not pair and warnings and checks.get(memory_check) == 'passed':
        retained = [(index, record) for index, record in enumerate(warnings) if isinstance(record, dict) and ('first' in record or 'second' in record)]
        if retained:
            index, warning = retained[0]
            body += '#### Memory warning alongside the stopped check\n\n'
            body += ('The memory check passed with warnings' if memory_check == 'memory' else
                     'The synchronization check, which includes the memory analysis, passed with warnings')
            body += '; the overall result above also includes other checks. This warning does not replace or explain the separate analysis stop.\n\n'
            body += 'First of ' + str(len(warnings)) + ' retained warnings, in file order. All retained warnings remain in Evidence.\n\n'
            body += _access_view(warning, saved_pointer(result, '/warnings/' + str(index)), ptx, artifacts)
    body += '### 1.2 Analysis scope\n\n' + _coverage(result)
    for key in ('source_violations', 'frontend_source_violations', 'verification_limitations'):
        if result.get(key):
            body += '**' + key.replace('_', ' ').capitalize() + ':** ' + str(len(result[key]) if isinstance(result[key], list) else 1) + ' saved; inspect Evidence before generalizing this result.\n\n'
    body += 'This result applies to the supplied launch, arguments and enabled assumptions.\n' + FACTS_END + '\n'
    return body


def _summary(value, pointer, artifact):
    raw = _encoded(value).encode('utf-8')
    count = str(len(value)) + (' entries' if isinstance(value, dict) else ' items') if isinstance(value, (dict, list)) else str(len(raw)) + ' UTF-8 bytes'
    return {'stored_at': artifact + '#' + pointer, 'size': count, 'canonical_json_bytes': len(raw), 'canonical_json_sha256': hashlib.sha256(raw).hexdigest()}


BOUND = 20000


def _bounded(value, pointer, artifact, limit=BOUND):
    """Keep diagnostics exact where small; huge collections remain addressable."""
    raw = _encoded(value)
    if len(raw) <= limit:
        return value
    if isinstance(value, dict):
        return {key: _bounded(item, _ptr(pointer, key), artifact, limit) for key, item in value.items()}
    return {'artifact_reference': _summary(value, pointer, artifact)}


def _bounded_fields(result, keys):
    """Bound several saved fields together; cite large ones at their saved location."""
    present = {key: result[key] for key in keys if key in result}
    if len(_encoded(present)) <= BOUND:
        return present
    return {key: _bounded(value, saved_pointer(result, _ptr('', key)), 'output.json') for key, value in present.items()}


def _json_block(value):
    raw = json.dumps(value, indent=2, ensure_ascii=False)
    fence = '`' * max(3, max((len(match.group()) + 1 for match in re.finditer(r'`+', raw)), default=3))
    return fence + 'json\n' + raw + '\n' + fence + '\n'


def _sequence_encoding(values):
    """A range is exact only when every consecutive difference agrees."""
    step = values[1] - values[0] if len(values) > 1 else 0
    if all(right - left == step for left, right in zip(values, values[1:])):
        return {'range': [values[0], step, len(values)]}
    return {'values': values}


def _lossless_groups(records, pointer):
    """Compress repeated, same-shaped records without losing array positions.

    Nonuniform shapes stay in their original form. Integer fields can use exact
    arithmetic ranges; irregular integer sequences retain every value.
    """
    if len(records) < 128 or not all(isinstance(record, dict) for record in records):
        return None
    keys = set(records[0])
    if not all(set(record) == keys for record in records):
        return None
    common = {key: value for key, value in records[0].items()
              if all(_encoded(record[key]) == _encoded(value) for record in records[1:])}
    integers = [key for key in sorted(keys - set(common)) if all(type(record[key]) is int for record in records)]
    if not integers:
        # Unified diagnostics put evidence under details. Reuse the same
        # grouping algorithm on JSON-pointer leaves, retaining a reversible
        # encoding instead of expanding tens of thousands of repeated records.
        if any(isinstance(value, dict) and value for record in records for value in record.values()):
            def leaves(record, base=''):
                flat = {}
                for key, value in record.items():
                    path = _ptr(base, key)
                    if isinstance(value, dict) and value:
                        flat.update(leaves(value, path))
                    else:
                        flat[path] = value
                return flat
            packed = _lossless_groups([leaves(record) for record in records], pointer)
            if packed:
                packed['representation'] = 'indexed-record-groups/v2'
                packed['field_encoding'] = 'JSON-pointer leaves; unescape ~1 and ~0, then rebuild nested objects. Empty objects and arrays remain literal values.'
                packed['canonical_json_sha256'] = hashlib.sha256(_encoded(records).encode('utf-8')).hexdigest()
                packed['reconstruction'] += ' Rebuild each object from its JSON-pointer leaf keys.'
            return packed
        return None
    shared_keys = sorted(keys - set(common) - set(integers))
    grouped = {}
    for index, record in enumerate(records):
        shared = {key: record[key] for key in shared_keys}
        group = grouped.setdefault(_encoded(shared), {'shared': shared, 'indices': [], 'fields': {key: [] for key in integers}})
        group['indices'].append(index)
        for key in integers:
            group['fields'][key].append(record[key])
    if len(grouped) > len(records) // 2:
        return None
    for group in grouped.values():
        group['indices'] = _sequence_encoding(group['indices'])
        group['fields'] = {key: _sequence_encoding(values) for key, values in group['fields'].items()}
    return {'representation': 'indexed-record-groups/v1',
            'source': 'output.json#' + pointer, 'record_count': len(records),
            'canonical_json_sha256': hashlib.sha256(_encoded(records).encode('utf-8')).hexdigest(),
            'range_rule': '[start, step, count]: value[k] = start + k * step for 0 <= k < count; values arrays are literal',
            'reconstruction': 'For each group, merge shared fields from the document and group, then zip each field sequence with the original array indices. Place every record at its original index.',
            'shared': common, 'groups': list(grouped.values())}


def _grouped_json_block(value):
    # One compact JSON row per group avoids megabytes of indentation while
    # preserving a normal, parseable JSON document within the single report.
    metadata = {key: item for key, item in value.items() if key != 'groups'}
    raw = _encoded(metadata)[:-1] + ',"groups":[\n' + ',\n'.join(_encoded(group) for group in value['groups']) + '\n]}'
    fence = '`' * max(3, max((len(match.group()) + 1 for match in re.finditer(r'`+', raw)), default=3))
    return fence + 'json\n' + raw + '\n' + fence + '\n'


def _id_set(values):
    """Compress identifier sets, explicitly not execution sequences."""
    if not all(type(item) is int for item in values):
        return ', '.join(_inline(item) for item in values)
    groups = []
    for item in sorted(set(values)):
        if groups and groups[-1][1] + 1 == item:
            groups[-1][1] = item
        else:
            groups.append([item, item])
    return ', '.join(_inline(first if first == last else str(first) + '–' + str(last)) for first, last in groups) or 'none saved'


def _diagnostic_summary(result):
    body = ''
    if isinstance(result.get('unsupported_memory_pair'), dict):
        pair = result['unsupported_memory_pair']
        body += '#### Unresolved access relation\n\n' + _text(pair.get('reason', MISSING)) + '\n\n'
        body += 'First event ' + _inline(_mapping(pair.get('first')).get('event', MISSING)) + ' · second event ' + _inline(_mapping(pair.get('second')).get('event', MISSING)) + ' · ' + _cite(result, '/unsupported_memory_pair') + '.\n\n'
    if isinstance(result.get('error'), dict):
        body += '**Saved error:** ' + _text(result['error'].get('message', result['error'])) + '\n\n'
    rejection = _mapping(result.get('rejection'))
    if rejection:
        body += '#### Rejected check\n\n' + _text(rejection.get('message', MISSING)) + '\n\n'
        body += '**Issue event:** ' + _inline(rejection.get('issue', MISSING)) + ' · **Barrier phase:** ' + _inline(rejection.get('phase', MISSING)) + ' · ' + _cite(result, '/rejection') + '\n\n'
    witness = _mapping(result.get('counterexample'))
    if witness:
        body += '#### Model replay coverage\n\nEach row describes one saved sequence. Truncation means events are missing from the original result; the sequences cannot be joined into an uninterrupted timeline.\n\n'
        body += '| Sequence | Retained / total events | Truncated |\n| --- | --- | --- |\n'
        for key, sequence in witness.items():
            if not isinstance(sequence, dict) or 'events' not in sequence:
                continue
            body += '| ' + _inline(key) + ' | ' + str(len(_records(sequence['events']))) + ' / ' + _inline(sequence.get('total', MISSING)) + ' | ' + _inline(sequence.get('truncated', MISSING)) + ' |\n'
        body += '\n**Replay stop:** first unexecuted event ' + _inline(witness.get('first_unexecuted', MISSING)) + '. Exact arrays and flags: ' + _cite(result, '/counterexample') + '.\n\n'
    sources = [record for record in _records(result.get('source_events')) if isinstance(record, dict)]
    if sources:
        files = list(dict.fromkeys(_plain(record.get('file', MISSING)) for record in sources))
        body += '#### Event-to-source lookup\n\nEvent IDs below are lookup sets, **not execution order**. Each file is named once.\n\n'
        for index, file in enumerate(files, 1):
            body += '- **File ' + str(index) + ':** ' + _inline(file) + '\n'
        grouped = {}
        for record in sources:
            key = (_plain(record.get('file', MISSING)), _plain(record.get('line', MISSING)), _plain(record.get('column', MISSING)))
            grouped.setdefault(key, []).append(record)
        body += '\n| File · source line / column | Event IDs | Instruction IDs |\n| --- | --- | --- |\n'
        for (file, line, column), records in grouped.items():
            body += '| ' + str(files.index(file) + 1) + ' · ' + _inline(line) + ' / ' + _inline(column) + ' | ' + _id_set([r.get('event', MISSING) for r in records]) + ' | ' + _id_set([r.get('instruction', MISSING) for r in records]) + ' |\n'
        body += '\nExact per-event associations: ' + _cite(result, '/source_events') + '.\n\n'
    failed = _records(result.get('failed_required_orders'))
    if failed:
        body += '#### Lifecycle requirements\n\n' + _number(result.get('failed_required_orders_count', MISSING)) + ' failures reported; ' + str(len(failed)) + ' retained; truncated: ' + _inline(result.get('failed_required_orders_truncated', MISSING)) + '.\n\n'
        grouped = {}
        for index, record in enumerate(failed):
            record = _mapping(record)
            signature = (_plain(record.get('kind', MISSING)), _plain(record.get('from_instruction', MISSING)), _plain(record.get('message', MISSING)))
            grouped.setdefault(signature, []).append((index, record.get('to_instruction', MISSING)))
        body += '| Requirement / from instruction | To instruction IDs | Retained records |\n| --- | --- | --- |\n'
        for (kind, source, message), records in grouped.items():
            targets = _id_set([target for _, target in records[:4]])
            if len(records) > 4:
                targets += '; ' + str(len(records) - 4) + ' more in exact records'
            body += '| ' + _inline(kind) + '<br/>from ' + _inline(source) + ' | ' + targets + ' | ' + _id_set([index for index, _ in records]) + ' |\n'
        messages = list(dict.fromkeys(key[2] for key in grouped))
        body += '\n' + '\n\n'.join(_text(message) for message in messages) + '\n\nEffect lists and per-record flags: ' + _cite(result, '/failed_required_orders') + '.\n\n'
    for key in ('races', 'warnings'):
        records = _records(result.get(key))
        if records:
            body += '#### Retained ' + key + '\n\n' + str(len(records)) + ' saved records. These are the retained sample, not necessarily all findings.\n\n'
            body += '| Record | First / second event | Modes | Saved code |\n| --- | --- | --- | --- |\n'
            for index, record in enumerate(records):
                record = _mapping(record)
                first, second = _mapping(record.get('first')), _mapping(record.get('second'))
                body += '| ' + str(index) + ' | ' + _inline(first.get('event', MISSING)) + ' / ' + _inline(second.get('event', MISSING)) + ' | ' + _inline(first.get('mode', MISSING)) + ' / ' + _inline(second.get('mode', MISSING)) + ' | ' + _inline(record.get('code', MISSING)) + ' |\n'
            messages = list(dict.fromkeys(_plain(r['message']) for r in records if isinstance(r, dict) and 'message' in r))
            body += '\n' + '\n\n'.join(_text(message) for message in messages) + '\n\nByte regions, endpoints and exact locations: ' + _cite(result, '/' + key) + '.\n\n'
    for key in ('diagnostics', 'source_violations', 'frontend_source_violations', 'verification_limitations'):
        records = _records(result.get(key))
        if records:
            body += '#### ' + key.replace('_', ' ').capitalize() + '\n\n'
            if key == 'diagnostics' and 1 < len(records) <= 128:
                body += _diagnostic_overview(_diagnostic_groups(records), detailed=True)
                continue
            if len(records) > 128:
                groups = {}
                for record in records:
                    item = _mapping(record)
                    code = _plain(item.get('code', MISSING))
                    message = _plain(item.get('message', item.get('reason', record)))
                    group = groups.setdefault(code, {'count': 0, 'messages': {}})
                    group['count'] += 1
                    group['messages'][message] = group['messages'].get(message, 0) + 1
                body += '| Saved code | Records | Distinct messages |\n| --- | --- | --- |\n'
                for code, group in groups.items():
                    body += '| ' + _inline(code) + ' | ' + _number(group['count']) + ' | ' + _number(len(group['messages'])) + ' |\n'
                body += '\n**First retained message:** ' + _text(next(iter(next(iter(groups.values()))['messages']))) + '\n\n'
                body += 'Every message and record remains in Exact diagnostic records below. Repeated records may use a lossless indexed representation; it preserves original array positions.\n\n'
                continue
            for index, record in enumerate(records):
                if isinstance(record, dict):
                    body += '- ' + _inline(record.get('code', index)) + ': ' + _text(record.get('message', record.get('reason', record))) + '\n'
                else:
                    body += '- ' + _text(record) + '\n'
            body += '\n'
    return body or 'Inspect the exact records below for explicit empty, null and other saved diagnostic fields.\n\n'


def _group_records(result, keys, title, explanation):
    present = {key: result[key] for key in keys if key in result}
    if not present:
        return ''
    body = explanation + '\n\n' + _diagnostic_summary(result)
    metadata = {key: value for key, value in present.items() if not isinstance(value, (dict, list)) or not value}
    where = ('keys in `output.json`, at the top level or in its `evidence` or `internal` section'
             if any(_section(result, key) for key in metadata) else 'top-level keys in `output.json`')
    exact = ('##### Flags and empty collections\n\nThese retain their original ' + where + '.\n\n' + _json_block(metadata) + '\n') if metadata else ''
    for key, value in present.items():
        if key in metadata:
            continue
        location = saved_pointer(result, '/' + key)
        exact += '##### ' + key.replace('_', ' ').capitalize() + '\n\n' + _ref(location) + '\n\n'
        if isinstance(value, list):
            exact += '**Retained records:** ' + str(len(value)) + '.\n\n'
            packed = _lossless_groups(value, location)
            if packed:
                exact += '**Lossless grouped records:** ' + str(len(packed['groups'])) + ' groups. Shared fields apply to every record; range triples are `[start, step, count]`, verified against every member. Original array indices preserve ordering.\n\n'
                exact += _grouped_json_block(packed) + '\n'
                continue
            # Irregular retained records stay exact; a pointer is not a lossless
            # replacement for diagnostic evidence in the report.
            shown = value
        else:
            shown = _bounded(value, location, 'output.json')
        if key != 'diagnostics':
            shown = _reference_repeated_diagnostics(shown, location, result)
        exact += _json_block(shown) + '\n'
    return _fold(title, body + _fold('Exact diagnostic records and metadata', exact))


def _reference_repeated_diagnostics(value, pointer, result):
    """Keep the primary diagnostics exact; link identical prerequisite copies.

    `pointer` is the saved location of `value`.
    """
    originals = {}

    def index(item, path):
        if not isinstance(item, (dict, list)):
            return
        raw = _encoded(item)
        if len(raw) >= 256:
            originals.setdefault(raw, path)
        for key, child in (item.items() if isinstance(item, dict) else enumerate(item)):
            index(child, _ptr(path, key))

    index(result.get('diagnostics'), saved_pointer(result, '/diagnostics'))

    def replace(item, path):
        if not isinstance(item, (dict, list)):
            return item
        raw = _encoded(item)
        if raw in originals:
            return {'same_records_as': 'output.json#' + originals[raw],
                    **_summary(item, path, 'output.json')}
        if isinstance(item, dict):
            return {key: replace(child, _ptr(path, key)) for key, child in item.items()}
        return [replace(child, _ptr(path, key)) for key, child in enumerate(item)]

    return replace(value, pointer)


def _numerical(result):
    body = ''
    exact = ''
    for key in ('value_analysis', 'equivalence', 'verification_status', 'verification_result'):
        if key not in result:
            continue
        value = result[key]
        location = saved_pointer(result, '/' + key)
        label = '**' + key.replace('_', ' ').capitalize() + ':** '
        if isinstance(value, dict) and _section(result, key) and 'result' not in value and 'status' not in value:
            # A public output keeps conclusions in its top-level checks; this
            # section saves details without an outcome of its own.
            if key == 'equivalence':
                body += label + 'check ' + _inline(_mapping(result.get('checks')).get('equivalence', MISSING)) + ' · ' + _cite(result, '/checks/equivalence') + '; details without a separate outcome · ' + _ref(location) + '\n\n'
            else:
                body += label + 'details without a separate outcome; the top-level checks hold the conclusions · ' + _ref(location) + '\n\n'
        else:
            body += label + _inline(_outcome(value) if isinstance(value, dict) else value) + ' · ' + _ref(location) + '\n\n'
        if isinstance(value, dict):
            if 'checks' in value or 'result' in value:
                body += _check_summary(value)
            for name in ('reason', 'message', 'prerequisite_status', 'prerequisite_result', 'numerical_status', 'numerical_result', 'scope', 'semantics', 'equivalence_contract', 'prerequisites'):
                if name in value:
                    body += '- **' + name.replace('_', ' ').capitalize() + ':** ' + _text(value[name]) + '\n'
            for index, diagnostic in enumerate(_records(value.get('diagnostics'))):
                diagnostic = _mapping(diagnostic)
                body += '- **Diagnostic:** ' + _inline(diagnostic.get('code', MISSING)) + ' · ' + _inline(diagnostic.get('stage', MISSING))
                if 'message' in diagnostic:
                    body += ' — ' + _text(diagnostic['message'])
                body += ' · ' + _ref(location + '/diagnostics/' + str(index)) + '\n'
            if isinstance(value.get('definedness'), dict):
                body += '- **Definedness:** ' + _inline(value['definedness'].get('status', MISSING)) + '\n'
            body += '\n'
            summary = {name: item for name, item in value.items() if not isinstance(item, (dict, list))}
            # Keep independent statuses and diagnostics; avoid repeating giant expression DAGs.
            for name, item in value.items():
                if isinstance(item, (dict, list)):
                    if len(_encoded(item)) <= 4000 or any(word in name for word in ('diagnostic', 'error', 'reason', 'limitation', 'scalar_fold', 'definedness', 'contract', 'context', 'assumption', 'idealization')):
                        summary[name] = _bounded(item, location + '/' + name, 'output.json')
                    else:
                        summary[name] = _summary(item, location + '/' + name, 'output.json')
            summary = _reference_repeated_diagnostics(summary, location, result)
            exact += '##### ' + key.replace('_', ' ').capitalize() + '\n\n' + _json_block(summary) + '\n'
    if not body:
        body = 'No independent value-analysis or equivalence status is saved.\n'
    return _fold('Value analysis — independent status', 'Memory safety and numerical equivalence are separate results.\n\n' + body + (_fold('Exact value-analysis fields and large-data references', exact) if exact else ''))


def render_evidence(run_input, result, ptx, artifacts, *, report_path=None, source_paths=(), equivalence_path=None, artifact_base=None):
    """Group exact diagnostics, assumptions, configuration and artifact references."""
    body = EVIDENCE_START + '\n## Evidence appendix\n\nUse these records to check the verifier result and source explanation above. JSON pointers refer to the original saved files. Identical prerequisite copies use `same_records_as` to refer to the diagnostic evidence shown once; their original locations and content hashes remain recorded.\n\n'
    diagnostics = ('rejection', 'rejection_kind', 'counterexample', 'witness_event_lists_truncated', 'witness_event_list_limit', 'source_event_scope', 'source_events', 'unmapped_witness_events', 'synthetic_events', 'races', 'warnings', 'unsupported_memory_pair', 'diagnostics', 'error', 'failed_required_orders', 'failed_required_orders_count', 'failed_required_orders_sample_limit', 'failed_required_orders_truncated', 'source_violations', 'frontend_source_violations', 'verification_limitations', 'recovered_source_violations')
    body += _group_records(result, diagnostics, 'Diagnostic records — verify the finding', 'These are saved verifier records. Truncation flags describe missing records in the original result, not merely collapsed presentation.')
    assumption_keys = ('assumptions', 'semantic_assumptions', 'frontend_assumptions', 'assumption_effects', 'assumption_reporting', 'assumption_dependent')
    rules = 'Enabled and triggered answer different questions: whether a rule was selected, and whether its applicability was recorded. `triggered: null` means unknown or unmeasured, not inactive.\n\n'
    for item in _records(result.get('assumptions')):
        if isinstance(item, dict):
            label = item.get('key')
            label = label.replace('_', ' ') if isinstance(label, str) else 'Saved rule'
            label = item.get('label', label)
            prefix = _inline(item['id']) + ' · ' if 'id' in item else ''
            rules += '#### ' + prefix + _text(label) + '\n\n'
            if 'key' in item:
                rules += '**Key:** ' + _inline(item['key']) + '\n\n'
            rules += '**Enabled:** ' + _inline(item.get('enabled', MISSING)) + ' · **Triggered:** ' + _inline(item.get('triggered', MISSING)) + '\n\n'
            rules += _text(item.get('text', MISSING)) + '\n\n'
    rules += _fold('Exact assumption settings', _json_block(_bounded_fields(result, assumption_keys)))
    body += _fold('Assumptions — what qualifies the result', rules)
    body += _numerical(result)
    scope = '**Launch and arguments define this run.** Tuple coordinates use `(x, y, z)`; grid and cluster dimensions count blocks, while block dimensions count threads.\n\n'
    launch = _mapping(run_input.get('launch'))
    scope += '| Launch quantity | Saved value |\n| --- | --- |\n'
    for key, label in [('grid_dim', 'Grid · blocks'), ('block_dim', 'Block · threads'), ('cluster_dim', 'Cluster · blocks'), ('dynamic_shared_bytes', 'Dynamic shared memory · bytes')]:
        scope += '| ' + label + ' | ' + _inline(launch.get(key, MISSING)) + ' |\n'
    argument_count = len(run_input['arguments']) if isinstance(run_input.get('arguments'), list) else run_input.get('arguments', MISSING)
    scope += '\n**Arguments:** ' + _number(argument_count) + ' retained entries. Exact parameter bindings, descriptors and entry name: ' + _ref('', 'input.json') + '.\n\n'
    scope += _fold('Exact input configuration', _json_block(_bounded(run_input, '', 'input.json')))
    body += _fold('Input configuration — reproduce this run', scope)
    handled = set(diagnostics + assumption_keys + ('value_analysis', 'equivalence', 'verification_status', 'verification_result'))
    remaining = {key: _bounded(value, saved_pointer(result, '/' + key), 'output.json') for key, value in result.items() if key not in handled}
    provenance = 'Use the original-file hashes to identify the run. Coverage flags describe what this result establishes; implementation statistics remain separate below.\n\n' + _coverage(result)
    provenance += '\n#### Original artifacts\n\n'
    originals = list(zip(('Input configuration', 'Verifier output', 'PTX'), artifacts))
    originals += [('Source ' + Path(path).name, path) for path in source_paths]
    if equivalence_path is not None:
        originals.append(('Separate numerical check', equivalence_path))
    for role, path in originals:
        if path is None:
            provenance += '- **' + role + ':** not provided.\n'
            continue
        path = Path(path)
        provenance += '- **' + role + ':** ' + _file_link(path.name, path, report_path, artifact_base) + ' · ' + _inline(str(path))
        if path.is_file():
            provenance += ' · SHA-256 ' + _inline(hashlib.sha256(path.read_bytes()).hexdigest())
        else:
            provenance += ' · file bytes unavailable; SHA-256 not computed'
        provenance += '\n'
    provenance += '\nLarge structures are referenced by exact JSON pointer, item count and SHA-256 of their canonical JSON (UTF-8, sorted keys, compact separators). Original-artifact hashes above are hashes of the file bytes.\n'
    provenance += '\n' + _fold('Exact coverage, implementation and provenance fields', _json_block(remaining))
    body += _fold('Provenance and coverage — check the saved run', provenance)
    return body + EVIDENCE_END + '\n'
