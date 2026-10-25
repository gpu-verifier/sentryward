"""Render an authored evidence plan with the skill's native GitHub components.

The agent supplies source conclusions. Saved fields, exact instructions and
bounded comparison/branch relations come directly from the supplied artifacts.
"""
import html
import json
from pathlib import Path
import re
from code_excerpt import excerpt, select_lines, instruction_opcodes, mask_noncode, occurrences, literal_token

MARKER = '<!-- report-layout:v2 -->'
ISA = 'https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html#'
LINKS_92 = {
    'cp.async.bulk.tensor': ISA + 'data-movement-and-conversion-instructions-cp-async-bulk-tensor',
    'cp.async.bulk.commit_group': ISA + 'data-movement-and-conversion-instructions-cp-async-bulk-commit-group',
    'cp.async.bulk.wait_group': ISA + 'data-movement-and-conversion-instructions-cp-async-bulk-wait-group',
    'tcgen05.ld': ISA + 'tcgen05-instructions-tcgen05-ld',
    'tcgen05.st': ISA + 'tcgen05-instructions-tcgen05-st',
    'tcgen05.wait': ISA + 'tcgen05-instructions-tcgen05-wait',
    'tcgen05.alloc': ISA + 'tcgen05-instructions-tcgen05-alloc-dealloc-relinquish-alloc-permit',
    'tcgen05.dealloc': ISA + 'tcgen05-instructions-tcgen05-alloc-dealloc-relinquish-alloc-permit',
    'tcgen05.relinquish_alloc_permit': ISA + 'tcgen05-instructions-tcgen05-alloc-dealloc-relinquish-alloc-permit',
    'tcgen05.commit': ISA + 'tcgen-async-sync-operations-commit',
    'tcgen05.mma': ISA + 'tcgen05-mma-instructions',
    'tcgen05.cp': ISA + 'tcgen05-instructions-tcgen05-cp',
    'mbarrier.try_wait': ISA + 'parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait',
    'mbarrier.test_wait': ISA + 'parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait',
    'mbarrier.arrive': ISA + 'parallel-synchronization-and-communication-instructions-mbarrier-arrive',
    'mbarrier.init': ISA + 'parallel-synchronization-and-communication-instructions-mbarrier-init',
    'st.async': ISA + 'data-movement-and-conversion-instructions-st-async',
}
ISA_93 = ISA.replace('/13.2.0/', '/13.3.0/')
LINKS_93 = {family: url.replace('/13.2.0/', '/13.3.0/') for family, url in LINKS_92.items()}
LINKS_93.update({
    'barrier.cluster': ISA_93 + 'parallel-synchronization-and-communication-instructions-barrier-cluster',
    'cp.async.cg': ISA_93 + 'data-movement-and-conversion-instructions-cp-async',
    'cp.async.ca': ISA_93 + 'data-movement-and-conversion-instructions-cp-async',
})


def instruction_docs(ptx):
    version = re.search(r'^\s*\.version\s+(\d+\.\d+)\b', ptx, re.M)
    return {'9.2': LINKS_92, '9.3': LINKS_93}.get(version.group(1), {}) if version else {}


def instruction_url(opcode, docs):
    for family in sorted(docs, key=len, reverse=True):
        if opcode == family or opcode.startswith((family + '.', family + '::')):
            return docs[family]
    return None

def fields(value, required, optional=()):
    if not isinstance(value, dict):
        raise ValueError('Expected an object with fields: ' + ', '.join(required))
    missing = set(required) - value.keys()
    extra = value.keys() - set(required) - set(optional)
    if missing or extra:
        raise ValueError('Draft fields: missing=' + str(sorted(missing)) + ', unknown=' + str(sorted(extra)))

def text(value, name, limit=None):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + ' must be nonempty text.')
    if re.search(r'<(?:details|summary|h[1-6]|pre|img)\b|^#\s|align\s*=|<!--', value, re.I | re.M):
        raise ValueError(name + ': leave page structure to the composer.')
    return value.strip()

def label(value):
    value = text(value, 'Label')
    if not re.fullmatch(r'[A-Za-z0-9 ._()/−–-]{1,32}', value):
        raise ValueError('Use a short plain label (up to 32 characters): ' + value)
    return value

def tex_label(value):
    return label(value).replace('_', r'\_')

def quantity(value):
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return str(value)
    if isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9 ?.+*/()≤≥=−–-]{1,20}', value):
        return value
    raise ValueError('Use a positive integer or short exact symbolic/range quantity.')

def shape_view(shapes, operators):
    if not isinstance(shapes, list) or not 1 <= len(shapes) <= 3:
        raise ValueError('Show one to three logical objects; name same-shaped auxiliary outputs in the caption.')
    if len(operators) != len(shapes) - 1 or any(x not in ('×', '→', '+') for x in operators):
        raise ValueError('Use one ×, → or + operator between neighboring logical objects.')
    for index, operator in enumerate(operators):
        if operator == '×':
            inner_left = shapes[index]['columns']['value']
            inner_right = shapes[index + 1]['rows']['value']
            if type(inner_left) is int and type(inner_right) is int and inner_left != inner_right:
                raise ValueError('Matrix multiplication must share the logical reduction extent. Do not multiply a packed-storage extent by a logical operand; draw logical shapes and explain packing separately.')
    tops, bottoms = [], []
    for i, item in enumerate(shapes):
        fields(item, ('name', 'rows', 'columns'))
        fields(item['rows'], ('label', 'value'))
        fields(item['columns'], ('label', 'value'))
        name = tex_label(item['name'])
        if len(item['name']) > 16:
            raise ValueError('Shape names must fit beside their axes (at most 16 characters). Use a short role label and name every separate allocation in the caption.')
        row, col = item['rows'], item['columns']
        tops += ['', r'\begin{matrix}\text{' + tex_label(col['label']) + r'}\\' + quantity(col['value']) + r'\end{matrix}']
        bottoms += [
            r'\begin{matrix}\text{' + tex_label(row['label']) + r'}\\' + quantity(row['value']) + r'\end{matrix}\left\{\rule[-0.5em]{0pt}{1.8em}\right.',
            r'\overbrace{\left[\rule[-0.5em]{0pt}{1.8em}\hspace{.4em}\text{' + name + r'}\hspace{.4em}\right]}'
        ]
        if i < len(operators):
            tops.append('')
            bottoms.append({'×':r'\times','→':r'\longrightarrow','+':'+'}[operators[i]])
    alignment = 'rc' + 'crc' * (len(shapes)-1)
    # Long role/axis labels need a compact font in GitHub's narrow report pane.
    label_width = sum(max(len(item['name']), len(item['columns']['label'])) + len(item['rows']['label']) for item in shapes)
    font = r'\normalsize' if label_width > 36 else r'\large'
    return '\n'.join(['```math', font + r'\begin{array}{' + alignment + '}', '&'.join(tops) + r'\\[-4pt]', '&'.join(bottoms), r'\end{array}', '```'])

def table_view(table):
    fields(table, ('headers', 'rows'))
    headers, rows = table['headers'], table['rows']
    if not isinstance(headers, list) or not 2 <= len(headers) <= 6:
        raise ValueError('A comparison needs two to six short column headers.')
    if not isinstance(rows, list) or not rows:
        raise ValueError('A comparison needs rows.')
    def cell(value):
        return text(str(value), 'Table cell').replace('|', '&#124;').replace('\n','<br/>')
    rendered = ['| ' + ' | '.join(cell(h) for h in headers) + ' |',
                '| ' + ' | '.join(['---']*len(headers)) + ' |']
    for row in rows:
        if not isinstance(row, list) or len(row) != len(headers):
            raise ValueError('Every row must match the comparison headers.')
        rendered.append('| ' + ' | '.join(cell(x) for x in row) + ' |')
    return '\n'.join(rendered)

def code_view(code, ptx, width=72, *, source_url=None):
    fields(code, ('caption','lines'), ('focus','links','operands'))
    docs, links = instruction_docs(ptx), {}
    if docs:
        masked = mask_noncode(ptx).splitlines()
        for number in select_lines(code['lines'], len(masked)):
            for first, last in instruction_opcodes(masked[number - 1]):
                opcode = masked[number - 1][first:last]
                url = instruction_url(opcode, docs)
                if url:
                    links[opcode] = url
    links.update(code.get('links', {}))
    rendered = excerpt(ptx, code['lines'], code.get('focus',[]), links, width=width, caption=text(code['caption'],'Code caption',16), source_url=source_url)
    if 'operands' in code:
        if not isinstance(code['operands'], list) or not code['operands']:
            raise ValueError('An operand guide needs named operands.')
        lines = ptx.splitlines()
        selected = '\n'.join(lines[n - 1] for n in select_lines(code['lines'], len(lines)))
        names = []
        for item in code['operands']:
            fields(item, ('token', 'meaning'))
            token = literal_token(item['token'])
            if not list(occurrences(mask_noncode(selected), token)):
                raise ValueError('Named operand is absent from the selected instructions: ' + token)
            names.append('<li><code>' + html.escape(token) + '</code> — ' + html.escape(text(item['meaning'], 'Operand meaning')) + '</li>')
        guide = '<ul>\n' + '\n'.join(names) + '\n</ul>\n\n'
        rendered = rendered.replace('<pre><code>', guide + '<pre><code>', 1)
    return rendered


def forward_branch(diagram, ptx):
    """Resolve one straight, forward skip; this is not a kernel CFG proof."""
    lines = ptx.splitlines()
    numbers = [diagram[k] for k in ('comparison_line','branch_line','effect_line')]
    if not all(type(n) is int and 1 <= n <= len(lines) for n in numbers):
        raise ValueError('Branch diagram needs valid physical source lines.')
    compare_line, branch_line, effect_line = numbers
    compare = re.fullmatch(r'\s*setp\.(eq|ne|lt|le|gt|ge)\.[\w.]+\s+(%p\d+),\s*([^,;]+),\s*([^,;]+);\s*', lines[compare_line-1])
    branch = re.fullmatch(r'\s*@(!?)(%p\d+)\s+bra(?:\.uni)?\s+([^; ]+);\s*', lines[branch_line-1])
    if not compare or not branch or compare[2] != branch[2] or compare_line >= branch_line:
        raise ValueError('Branch diagram needs the literal comparison feeding its matching conditional branch.')
    if any(re.search(r'\b(?:bra(?:\.uni)?|ret|exit|trap)\b', l) for l in lines[compare_line:branch_line-1]):
        raise ValueError('Branch comparison must reach the branch without another control transfer.')
    if any(re.search(r'^\s*(?:@!?[\w%.$]+\s+)?[\w.]+\s+[^,;]*' + re.escape(compare[2]) + r'(?=[\s,|])', l)
           for l in lines[compare_line:branch_line-1]):
        raise ValueError('Branch predicate is redefined after the selected comparison.')
    targets = [i for i,l in enumerate(lines,1) if l.strip() == branch[3] + ':']
    if len(targets) > 1:
        raise ValueError('The branch target is ambiguous across PTX scopes.')
    target = targets[0] if targets else None
    if target is None or not branch_line < effect_line < target:
        raise ValueError('The forward branch must bypass the selected guarded operation.')
    if any(re.search(r'\b(?:bra(?:\.uni)?|ret|exit|trap)\b', l) for l in lines[branch_line:target-1]):
        raise ValueError('This bounded branch view cannot skip an intervening control transfer.')
    if lines[effect_line-1].lstrip().startswith('@'):
        raise ValueError('The selected operation has an additional predicate; use an explicitly scoped diagram.')
    return lines, compare, branch, target


def branch_labels(diagram, comparison, comparison_source):
    """Name paths by their conditions without changing predicate semantics."""
    inputs = diagram['inputs']
    if not isinstance(inputs, list) or len(inputs) != 2:
        raise ValueError('Name the two comparison operands in order.')
    left, right = [label(value) for value in inputs]
    operators = {'eq':'=', 'ne':'≠', 'lt':'<', 'le':'≤', 'gt':'>', 'ge':'≥'}
    inverse = {'eq':'ne', 'ne':'eq', 'lt':'ge', 'le':'gt', 'gt':'le', 'ge':'lt'}
    operator = operators[comparison[1]]
    condition = left + ' ' + operator + ' ' + right
    # Ordered floating-point comparisons can be false for NaNs. Their false
    # branch is not generally the opposite ordered comparison.
    integer = re.match(r'\s*setp\.[\w.]+\.[bsu]\d+\s', comparison_source)
    complement = (left + ' ' + operators[inverse[comparison[1]]] + ' ' + right
                  if integer else 'not (' + condition + ')')
    paths = {'true': condition, 'false': complement}
    if 'path_labels' in diagram:
        fields(diagram['path_labels'], ('true', 'false'))
        paths = {key: label(value) for key, value in diagram['path_labels'].items()}
    return operator, {
        key: html.escape(value, quote=True) for key, value in paths.items()}


def operation_label(diagram, role, fallback, allowed):
    labels = diagram.get('operation_labels', {})
    fields(labels, (), allowed)
    return html.escape(label(labels[role]), quote=True) if role in labels else fallback


def branch_diagram(diagram, ptx):
    fields(diagram, ('kind', 'comparison_line', 'branch_line', 'effect_line', 'inputs'),
           ('path_labels', 'operation_labels'))
    lines, compare, branch, target = forward_branch(diagram, ptx)
    compare_line, branch_line, effect_line = (diagram[k] for k in ('comparison_line','branch_line','effect_line'))
    operator, paths = branch_labels(diagram, compare, lines[compare_line-1])
    effect = lines[effect_line-1].strip()
    operation = re.match(r'(st|ld)\.(local|shared|global)\.', effect)
    if operation:
        action = operation[2].capitalize() + (' store' if operation[1] == 'st' else ' load')
    elif effect.startswith(('cp.async', 'tcgen05.cp')):
        action = 'Copy'
    else:
        raise ValueError('The bounded branch diagram supports an exact memory load, store or copy.')
    taken, fallthrough = ('false','true') if branch[1] else ('true','false')
    effect_name = operation_label(diagram, 'effect', action, ('effect', 'join'))
    join_name = operation_label(diagram, 'join', 'Continue', ('effect', 'join'))
    source = ('flowchart LR\n'
              f'  G{{"Compare operands<br/>PTX {compare_line}"}}\n'
              f'  E["{effect_name}<br/>PTX {effect_line}"]\n'
              f'  J["{join_name}<br/>PTX {target}"]\n'
              f'  G -->|"{paths[fallthrough]}"| E\n'
              f'  G -->|"{paths[taken]} · skip"| J\n'
              '  E --> J\n  class G context\n  class E focus\n  class J operation')
    meaning = ('Arrows show local control flow, not observed execution. The comparison at PTX ' + str(compare_line) +
               ' uses ' + compare[3].strip() + ' and ' + compare[4].strip() +
               ' (' + operator + '); the branch at ' + str(branch_line) + ' bypasses the ' + action.lower() +
               ' at ' + str(effect_line) + ' when this comparison is ' + taken +
               '. Both paths rejoin at PTX ' + str(target) + '.')
    return diagram_view({'kind':'mermaid','source':source,'meaning':meaning}, compact=True)


def guarded_path(diagram, ptx):
    """Show both sides of a source branch, including their common operation."""
    fields(diagram, ('kind', 'comparison_line', 'branch_line', 'effect_line', 'join_line', 'inputs'),
           ('before_line', 'path_labels', 'operation_labels'))
    lines, compare, branch, target = forward_branch(diagram, ptx)
    join = diagram['join_line']
    if type(join) is not int or not target < join <= len(lines):
        raise ValueError('The common operation must follow the branch target.')
    corridor = lines[target:join-1]
    if any(re.search(r'\b(?:bra(?:\.uni)?|ret|exit|trap)\b|^\s*\.entry\b|^\s*\.func\b', l) for l in corridor):
        raise ValueError('The common operation must be reached without an intervening control transfer.')
    if lines[join-1].lstrip().startswith('@'):
        raise ValueError('The common operation is predicated, not shared by both paths.')
    operator, paths = branch_labels(diagram, compare, lines[diagram['comparison_line']-1])

    roles = ('before', 'effect', 'join') if 'before_line' in diagram else ('effect', 'join')

    def operation(number, role):
        # Labels identify an issue site. They deliberately do not imply completion.
        match = re.match(r'\s*((?:st|ld)\.async|tcgen05\.(?:ld|st|cp|commit|dealloc)|'
                         r'cp\.async(?:\.bulk(?:\.tensor)?)?|mbarrier\.(?:arrive|try_wait)|'
                         r'(?:st|ld)\.(?:local|shared|global))\.', lines[number-1])
        if not match:
            raise ValueError('Guarded paths need an exact memory, async or barrier operation at each selected site.')
        return operation_label(diagram, role, match[1], roles) + '<br/>PTX ' + str(number)

    selected = diagram['effect_line']
    taken, fallthrough = ('false', 'true') if branch[1] else ('true', 'false')
    origin, origin_class = 'G', 'context'
    start = f'  G{{"Compare operands<br/>PTX {diagram["comparison_line"]}"}}\n'
    if 'before_line' in diagram:
        before = diagram['before_line']
        if type(before) is not int or not 1 <= before < diagram['comparison_line']:
            raise ValueError('The preceding operation must be before the selected comparison.')
        if any(re.search(r'\b(?:bra(?:\.uni)?|ret|exit|trap)\b|^\s*\.(?:entry|func)\b', line)
               for line in lines[before:diagram['comparison_line']-1]):
            raise ValueError('The preceding operation must reach the comparison without a control transfer.')
        preceding = f'  P["{operation(before, "before")}"]'
        if 'path_labels' in diagram:
            # Named outcomes carry the split; retain its exact predicate in the caption.
            origin, origin_class = 'P', 'operation'
            start = preceding + '\n'
        else:
            start = preceding + ' --> G\n' + start + '  class P operation\n'
    source = ('flowchart LR\n' + start +
              f'  E["{operation(selected, "effect")}"]\n'
              f'  J["{operation(join, "join")}"]\n'
              f'  {origin} -->|"{paths[fallthrough]}"| E\n'
              f'  {origin} -->|"{paths[taken]} · skip"| J\n'
              f'  E --> J\n  class {origin} {origin_class}\n  class E focus\n  class J operation')
    meaning = (f'Local source paths: the branch at PTX {diagram["branch_line"]} tests '
               f'`{compare[3].strip()} {operator} {compare[4].strip()}`. Both paths reach PTX {join}. '
               'Intervening instructions are omitted; arrows do not establish asynchronous completion or cross-thread order.')
    return diagram_view({'kind':'mermaid', 'source':source, 'meaning':meaning}, compact=True)

def diagram_view(diagram, *, compact=False):
    # A diagram describes a relation, never a substitute for its evidence.
    fields(diagram, ('kind', 'source', 'meaning'))
    if diagram['kind'] not in ('mermaid','math'):
        raise ValueError('Diagram kind must be mermaid or math.')
    source = text(diagram['source'], 'Diagram')
    if '```' in source:
        raise ValueError('Supply diagram content without fences.')
    if diagram['kind'] == 'mermaid' and not source.startswith('---') and re.match(r'(?:flowchart|graph)\b', source):
        spacing = '    nodeSpacing: 16\n    rankSpacing: 16\n    padding: 8\n' if compact else '    nodeSpacing: 24\n    rankSpacing: 24\n'
        font = '16px' if compact else '17px'
        source = '---\nconfig:\n  theme: base\n  flowchart:\n    curve: linear\n' + spacing + '  themeVariables:\n    lineColor: "#8c959f"\n    edgeLabelBackground: "#f6f8fa"\n    fontFamily: Arial\n    fontSize: ' + font + '\n---\n' + source
        source += '\nclassDef focus fill:#f0ebfa,stroke:#bba6d9,color:#513477\nclassDef context fill:#f6f8fa,stroke:#abb5c0,color:#344251\nclassDef operation fill:#e8f0ff,stroke:#8ca7d2,color:#254b83'
    return '```' + diagram['kind'] + '\n' + source + '\n```\n\n' + text(diagram['meaning'],'Arrow/axis meaning',45)

def technical_prose(document, docs=None):
    """Mark unambiguous identifiers without rewriting code, math or links."""
    token = re.compile(
        r'(?<![\w%])(?:%[A-Za-z][\w]*(?:\.[xyz])?|\$[A-Za-z_][\w]*|A\d{3}\b|'
        r'(?:tid|ntid|ctaid|nctaid|threadIdx|blockIdx|blockDim|gridDim)\.[xyz]\b|'
        r'[A-Za-z][A-Za-z0-9]*(?:\.[A-Za-z0-9]+)*(?:_[A-Za-z0-9]+)+(?:\.[A-Za-z0-9_]+)*|'
        r'(?:st|ld|setp|mov|add|sub|mul|mad|and|or|xor|cvt|cp|tcgen05|bar|mbarrier|bra|wait|beta)'
        r'(?:\.[A-Za-z0-9_]+(?:::[A-Za-z0-9_]+)?)+)(?![\w])')
    protected = re.compile(
        r'(<code\b[^>]*>.*?</code>|<a\b[^>]*>.*?</a>|<[^>]+>|'
        r'`+[^`]*`+|!?\[[^\]]*\]\([^\n]*?\)|https?://[^\s<>]+)', re.I)
    lines, fence, pre = [], None, False
    for line in document.splitlines(keepends=True):
        content = re.sub(r'^(?:> ?)*', '', line).lstrip()
        delimiter = re.match(r'(`{3,}|~{3,})', content)
        if fence:
            lines.append(line)
            if delimiter and delimiter[1][0] == fence[0] and len(delimiter[1]) >= len(fence):
                fence = None
            continue
        if delimiter:
            fence = delimiter[1]
            lines.append(line)
            continue
        if pre or re.search(r'<pre\b', line, re.I):
            pre = not bool(re.search(r'</pre>', line, re.I))
            lines.append(line)
            continue
        pieces = protected.split(line)
        for i in range(0, len(pieces), 2):
            pieces[i] = token.sub(lambda m: '<code>' + m[0] + '</code>', pieces[i])
        def linked_term(match):
            opcode = match[1] or match[2]
            url = instruction_url(opcode, docs or {})
            if url is None:
                return match[0]
            return '<a href="' + html.escape(url, quote=True) + '"><code>' + html.escape(opcode) + '</code></a>'
        for i, piece in enumerate(pieces):
            # Existing links are already an intentional destination; never nest them.
            if re.match(r'<a\b|!?\[|https?://', piece, re.I):
                continue
            pieces[i] = re.sub(r'<code>([^<>]+)</code>|(?<!`)`([^`]+)`(?!`)', linked_term, piece)
        lines.append(''.join(pieces))
    return ''.join(lines)


EXPLANATION_START = '<!-- code-explanation:start -->'
EXPLANATION_END = '<!-- code-explanation:end -->'


def json_pointer(document, pointer):
    """Resolve an RFC 6901 pointer, preserving missing versus null."""
    if not isinstance(pointer, str) or (pointer and not pointer.startswith('/')):
        raise ValueError('Evidence needs a JSON pointer, e.g. /rejection/code.')
    value = document
    for part in pointer.split('/')[1:]:
        key = part.replace('~1', '/').replace('~0', '~')
        try:
            if isinstance(value, list):
                if not re.fullmatch(r'0|[1-9][0-9]*', key):
                    raise ValueError('Invalid array index in evidence: ' + pointer)
                value = value[int(key)]
            elif isinstance(value, dict):
                value = value[key]
            else:
                raise KeyError(key)
        except (KeyError, IndexError) as exc:
            raise ValueError('Evidence pointer is absent: ' + pointer) from exc
    return value


# Sections of a public (two-stage) verifier output. The report reads their
# fields as root fields. On a name collision the root field wins, then
# `evidence`, then `internal`.
SAVED_SECTIONS = ('evidence', 'internal')


class SavedResult(dict):
    """Flat view of a public verifier output, built by saved_result.

    `origins` maps each field read from a section to that section's name.
    `saved` is the original document; citations refer to its locations.
    """

    def __init__(self, fields, origins, saved):
        super().__init__(fields)
        self.origins = origins
        self.saved = saved


def saved_result(result):
    """Read a saved verifier output in the flat layout the report expects.

    A public output keeps analysis details under `internal` and witnesses
    under `evidence`. The view reads their fields as root fields under the
    SAVED_SECTIONS precedence rule, in saved order, and omits the section
    objects themselves. saved_pointer translates a pointer into this view to
    the saved location. Older flat outputs are returned unchanged.
    """
    if isinstance(result, SavedResult) or not isinstance(result, dict):
        return result
    sections = {name: result[name] for name in SAVED_SECTIONS if isinstance(result.get(name), dict)}
    if not sections:
        return result
    owner = {key: None for key in result if key not in sections}
    for name, fields in sections.items():
        for key in fields:
            owner.setdefault(key, name)
    view = {}
    for key, value in result.items():
        if key in sections:
            view.update((field, item) for field, item in value.items() if owner[field] == key)
        else:
            view[key] = value
    return SavedResult(view, {key: name for key, name in owner.items() if name}, result)


def load_saved_result(path):
    """Read one saved verifier output file through saved_result."""
    return saved_result(json.loads(Path(path).read_text()))


def _pointer_key(token):
    return token.replace('~1', '/').replace('~0', '~')


def saved_pointer(result, pointer):
    """Locate a pointer into the saved_result view in the original saved file.

    A pointer to a field read from a section gains that section's prefix.
    Pointers that already name a saved location, and every pointer into an
    older flat output, are returned unchanged.
    """
    origins = getattr(result, 'origins', None)
    if not origins or not isinstance(pointer, str) or not pointer.startswith('/'):
        return pointer
    section = origins.get(_pointer_key(pointer[1:].split('/', 1)[0]))
    return '/' + section + pointer if section else pointer


def view_pointer(result, pointer):
    """Inverse of saved_pointer: the view pointer of a saved location."""
    origins = getattr(result, 'origins', None)
    if origins and isinstance(pointer, str) and pointer.startswith('/'):
        section, _, rest = pointer[1:].partition('/')
        if section in SAVED_SECTIONS and origins.get(_pointer_key(rest.split('/', 1)[0])) == section:
            return '/' + rest
    return pointer


def saved_value(result, pointer):
    """Resolve a view or saved pointer; return its saved location and value."""
    location = saved_pointer(result, pointer)
    return location, json_pointer(getattr(result, 'saved', result), location)


def validate_inputs(run_input, result, ptx):
    if not isinstance(run_input, dict) or not isinstance(result, dict):
        raise ValueError('Input and result must be JSON objects.')
    entry = run_input.get('entry')
    if not isinstance(entry, str) or not entry:
        raise ValueError('Input must identify a PTX entry.')
    if not re.search(r'\.entry\s+' + re.escape(entry) + r'(?=\s|\()', ptx):
        raise ValueError('Input entry is absent from the supplied PTX: ' + entry)
    frontend = result.get('frontend')
    entries = [result.get('entry')]
    if isinstance(frontend, dict):
        entries.append(frontend.get('entry'))
    if any(e is not None and e != entry for e in entries):
        raise ValueError('Input/result entry identity mismatch.')


def evidence_view(refs, run_input, result, ptx, sources=None):
    if not isinstance(refs, list) or not refs:
        raise ValueError('Each explanation requires evidence references.')
    items = []
    for ref in refs:
        if not isinstance(ref, dict):
            raise ValueError('Evidence references must be objects.')
        kind = ref.get('kind')
        if kind in ('input', 'result'):
            fields(ref, ('kind', 'pointer'), ('equals',))
            if kind == 'input':
                pointer, value = ref['pointer'], json_pointer(run_input, ref['pointer'])
            else:
                # Cite the saved location, also for a pointer into the flat view.
                pointer, value = saved_value(result, ref['pointer'])
            if 'equals' in ref and (type(value) != type(ref['equals']) or value != ref['equals']):
                raise ValueError('Evidence value mismatch: ' + kind + ref['pointer'])
            shown = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
            # The pointer remains exact even when the referenced object is large.
            detail = shown if len(shown) <= 180 else f'{type(value).__name__}, {len(value)} entries; see original artifact'
            artifact = 'input.json' if kind == 'input' else 'output.json'
            items.append(f'- `{artifact}#{pointer}`: <code>{html.escape(detail)}</code>')
        elif kind in ('ptx', 'source'):
            fields(ref, ('kind', 'lines', 'contains') if kind == 'ptx' else ('kind', 'file', 'lines', 'contains'))
            body = ptx if kind == 'ptx' else (sources or {}).get(ref['file'])
            if body is None:
                raise ValueError('Source was not supplied through --source: ' + ref['file'])
            lines = body.splitlines()
            selected = select_lines(ref['lines'], len(lines))
            quote = '\n'.join(lines[n - 1] for n in selected)
            tokens = ref['contains']
            if not isinstance(tokens, list) or not tokens or not all(isinstance(t, str) and t and t in quote for t in tokens):
                raise ValueError('Cited source must contain every exact token: ' + str(ref))
            name = 'physical PTX' if kind == 'ptx' else ref['file']
            items.append('- ' + html.escape(name) + ' ' + html.escape(ref['lines']) + ': ' + ', '.join('<code>' + html.escape(t) + '</code>' for t in tokens))
        else:
            raise ValueError('Evidence kind must be input, result, ptx or source.')
    return '\n'.join(items)


def timeline_view(diagram, run_input, result, ptx, sources=None, code=''):
    fields(diagram, ('kind', 'basis', 'participants', 'steps', 'conditions'), ('sequence',))
    basis = diagram['basis']
    if basis not in ('possible interleaving', 'verifier witness'):
        raise ValueError('Timeline basis must be possible interleaving or verifier witness.')
    participants = diagram['participants']
    if not isinstance(participants, list) or not 1 <= len(participants) <= 4:
        raise ValueError('A timeline needs one to four named participants.')
    ids, labels, support = [], [], []
    for participant in participants:
        fields(participant, ('id', 'label', 'evidence'))
        if not isinstance(participant['id'], str) or participant['id'] in ids:
            raise ValueError('Participant IDs must be distinct strings.')
        ids.append(participant['id'])
        labels.append(text(participant['label'], 'Participant label'))
        support.append(evidence_view(participant['evidence'], run_input, result, ptx, sources))
    fields(diagram['conditions'], ('participants', 'branches', 'iterations', 'completion'))
    conditions = []
    for name, condition in diagram['conditions'].items():
        fields(condition, ('text', 'evidence'))
        conditions.append('- **' + name.capitalize() + ':** ' + text(condition['text'], 'Timeline condition'))
        support.append(evidence_view(condition['evidence'], run_input, result, ptx, sources))
    sequence = None
    if basis == 'verifier witness':
        # Public outputs save the witness at /evidence/counterexample/...;
        # both that location and the older /counterexample/... are accepted.
        pointer = saved_pointer(result, diagram.get('sequence'))
        if (not isinstance(pointer, str) or not view_pointer(result, pointer).startswith('/counterexample/')
                or not pointer.endswith('/events')):
            raise ValueError('Verifier witness requires an exact /counterexample/.../events pointer.')
        pointer, sequence = saved_value(result, pointer)
        if not isinstance(sequence, list):
            raise ValueError('Witness sequence must be a saved list.')
    rows, last = [], -1
    if not isinstance(diagram['steps'], list) or not diagram['steps']:
        raise ValueError('A timeline needs steps.')
    for index, step in enumerate(diagram['steps'], 1):
        fields(step, ('participant', 'action', 'evidence'), ('event',))
        if step['participant'] not in ids:
            raise ValueError('Timeline step uses an unknown participant.')
        support.append(evidence_view(step['evidence'], run_input, result, ptx, sources))
        if sequence is not None:
            event = step.get('event')
            positions = [i for i, e in enumerate(sequence) if type(event) is int and e == event and i > last]
            if not positions:
                raise ValueError('Timeline event is absent or reverses the saved witness order.')
            position = positions[0]
            if position > last + 1:
                rows.append(['…'] + ['Saved events omitted from this view'] + ['—'] * (len(ids) - 1))
            last = position
        row = [str(index)] + ['—'] * len(ids)
        row[1 + ids.index(step['participant'])] = text(step['action'], 'Timeline action')
        rows.append(row)
    if sequence is not None:
        _, parent = saved_value(result, pointer.rsplit('/', 1)[0])
        if not isinstance(parent, dict):
            raise ValueError('Witness sequence must belong to a saved sequence object.')
        if last < len(sequence) - 1 or parent.get('truncated') is True:
            rows.append(['…'] + ['Further events omitted / not retained'] + ['—'] * (len(ids) - 1))
    caption = '**' + basis.capitalize() + '.** Steps run downward; they are not a measured GPU clock or a hardware trace.'
    references = '\n'.join(dict.fromkeys(line for block in support for line in block.splitlines()))
    reference_title = 'Sources for this interleaving' if basis == 'possible interleaving' else 'Saved sequence and source mappings'
    return caption + '\n\n' + table_view({'headers':['Step'] + labels, 'rows':rows}) + '\n\n' + code + '\n\n' + '\n'.join(conditions) + '\n\n' + disclosure(reference_title, references)


def disclosure(title, body, *, inset=False):
    content = body.strip()
    if inset:
        content = '<blockquote>\n\n' + content + '\n\n</blockquote>'
    return '<details>\n<summary><strong>' + html.escape(title) + '</strong></summary>\n\n' + content + '\n\n</details>'


def derivation_view(section, run_input, result, ptx, sources=None, *, default_title='Derivation and source references', source_url=None):
    refs = evidence_view(section['evidence'], run_input, result, ptx, sources)
    title = text(section.get('evidence_title', default_title), 'Evidence title')
    steps, citations, step_refs = [], [], set()
    for step in section.get('steps', []):
        fields(step, ('label', 'text', 'evidence'), ('code', 'comparison', 'diagram'))
        steps.append('#### ' + text(step['label'], 'Derivation step') + '\n\n' + text(step['text'], 'Derivation'))
        if 'diagram' in step:
            steps.append(diagram_view(step['diagram'], compact=True))
        if 'comparison' in step:
            steps.append(table_view(step['comparison']))
        if 'code' in step:
            steps.append(code_view(step['code'], ptx, source_url=source_url))
        supporting_refs = evidence_view(step['evidence'], run_input, result, ptx, sources)
        citations.append('##### Source references: ' + text(step['label'], 'Derivation step') + '\n\n' + supporting_refs)
        step_refs.update(supporting_refs.splitlines())
    if steps:
        # Parent references can be broader than individual derivation steps.
        extra_refs = '\n'.join(line for line in refs.splitlines() if line not in step_refs)
        if extra_refs:
            citations.append('##### Additional section references\n\n' + extra_refs)
        source_title = 'Sources: ' + title.removeprefix('Derivation: ') if 'evidence_title' in section else 'Sources: derivation references'
        steps.append(disclosure(source_title, '\n\n'.join(citations), inset=True))
    else:
        steps.append(refs)
    return disclosure(title, '\n\n'.join(steps))


def link_section_topics(summary, sections):
    """Link explicit topic names to unique sections; never infer a relationship."""
    topics = {}
    for index, section in enumerate(sections, 1):
        title = text(section['title'], 'Explanation title')
        topic = title.split(':', 1)[0].strip().casefold()
        topics.setdefault(topic, []).append(index)

    def replace(match):
        targets = topics.get(match[1].strip().removesuffix(':').strip().casefold(), [])
        if len(targets) != 1:
            return match[0]
        return '[' + match[0] + '](#user-content-code-section-' + str(targets[0]) + ')'

    return re.sub(r'(?<!\[)\*\*([^*\n]+)\*\*(?!\])', replace, summary)


def explanation(draft, run_input, result, ptx, sources=None, *, source_url=None):
    fields(draft, ('version', 'title', 'sections'), ('context', 'next'))
    if draft['version'] != 2:
        raise ValueError('Use evidence plan version 2; see references/draft-format.md.')
    sections = draft['sections']
    if not isinstance(sections, list) or not sections:
        raise ValueError('An explanation needs at least one evidence-supported section.')
    parts = [EXPLANATION_START, '---', '## 2. Code explanation', 'Interpretation from the PTX and saved inputs: how the reported operations fit together and what remains to check.']
    number = 1
    if 'context' in draft:
        context = draft['context']
        fields(context, ('summary', 'evidence'), ('title', 'shapes', 'operators', 'caption', 'diagram', 'steps', 'evidence_title'))
        parts.append('### 2.' + str(number) + ' ' + text(context.get('title', 'Code context'), 'Context title'))
        number += 1
        parts.append(link_section_topics(text(context['summary'], 'Relevant computation'), sections))
        if 'shapes' in context:
            parts.append(shape_view(context['shapes'], context.get('operators', [])))
        if 'diagram' in context:
            parts.append(diagram_view(context['diagram'], compact=True))
        if 'caption' in context:
            parts.append(text(context['caption'], 'Context caption'))
        parts.append(derivation_view(context, run_input, result, ptx, sources, default_title='Context evidence', source_url=source_url))
    for section_index, section in enumerate(sections, 1):
        fields(section, ('title', 'answer', 'evidence'), ('conclusion', 'diagram', 'comparison', 'code', 'steps', 'evidence_title'))
        # GitHub prefixes HTML IDs; the name also supports ordinary HTML views.
        parts += ['<a id="code-section-' + str(section_index) + '" name="user-content-code-section-' + str(section_index) + '"></a>',
                  '### 2.' + str(number) + ' ' + text(section['title'], 'Explanation title'),
                  text(section['answer'], 'Explanation answer')]
        number += 1
        if 'diagram' in section:
            diagram = section['diagram']
            if diagram.get('kind') == 'timeline':
                code = code_view(section['code'], ptx, source_url=source_url) if 'code' in section else ''
                parts.append(timeline_view(diagram, run_input, result, ptx, sources, code))
            elif diagram.get('kind') == 'branch':
                parts.append(branch_diagram(diagram, ptx))
            elif diagram.get('kind') == 'guarded-path':
                parts.append(guarded_path(diagram, ptx))
            else:
                parts.append(diagram_view(diagram))
        if 'comparison' in section:
            parts.append(table_view(section['comparison']))
        if 'code' in section and section.get('diagram', {}).get('kind') != 'timeline':
            parts.append(code_view(section['code'], ptx, source_url=source_url))
        if 'conclusion' in section:
            parts.append('**Implication.** ' + link_section_topics(
                text(section['conclusion'], 'Section conclusion'), sections))
        parts.append(derivation_view(section, run_input, result, ptx, sources, source_url=source_url))
    if 'next' in draft:
        fields(draft['next'], ('text', 'evidence'))
        refs = evidence_view(draft['next']['evidence'], run_input, result, ptx, sources)
        parts += ['### 2.' + str(number) + ' Next check', text(draft['next']['text'], 'Next inspection')]
        parts.append(disclosure('Evidence for the next check', refs))
    parts.append(EXPLANATION_END)
    return technical_prose('\n\n'.join(parts), instruction_docs(ptx))


def compose(draft, run_input, result, ptx, artifacts, sources=None, *, report_path=None, source_paths=(), equivalence_path=None, artifact_base=None, result_panel=False):
    from result_view import render_result, render_evidence, artifact_url, link_references
    result = saved_result(result)
    validate_inputs(run_input, result, ptx)
    # Validate the entire authored layer before rendering or replacing an artifact.
    source_url = artifact_url(artifacts[2], report_path, artifact_base) if len(artifacts) > 2 else None
    authored = explanation(draft, run_input, result, ptx, sources, source_url=source_url) if draft is not None else ''
    title = text(draft['title'], 'Kernel name') if draft is not None else None
    links = dict(report_path=report_path, source_paths=source_paths, equivalence_path=equivalence_path, artifact_base=artifact_base)
    parts = [MARKER, render_result(run_input, result, ptx, artifacts, title=title, **links)]
    if authored:
        parts.append(authored)
    parts.append(render_evidence(run_input, result, ptx, artifacts, **links))
    document = '\n\n'.join(part.strip() for part in parts if part) + '\n'
    document = link_references(document, ptx, artifacts, report_path, artifact_base)
    if result_panel:
        from result_panel import render
        document = render(document)
    return document
