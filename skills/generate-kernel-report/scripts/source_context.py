#!/usr/bin/env python3
"""Bounded, entry-local PTX evidence index; never executes PTX or the verifier."""
import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict, deque
import json
from pathlib import Path
import re

from parameter_analysis import parameter_bindings

# Limits bound displayed evidence, not the source range that is indexed.
MAX_CHARS = 48000
MAX_LINE = 260
TOKEN = re.compile(r'%[A-Za-z_$][\w$]*(?:\.[xyz])?|[A-Za-z_$][\w$]*')
SPECIAL = re.compile(r'%(?:tid\.[xyz]|laneid|warpid|nwarpid|lanemask_\w+)\b')
ROLE_OPS = {'mov', 'cvt', 'shr', 'shl', 'bfe', 'and', 'or', 'xor', 'not',
            'add', 'sub', 'mul', 'mad', 'div', 'rem', 'setp', 'selp', 'shfl'}
NO_DEST = {'st', 'stmatrix', 'sust', 'sured', 'red', 'bra', 'brx', 'call', 'ret', 'exit', 'trap', 'fence',
           'membar', 'bar', 'barrier', 'prefetch', 'discard', 'cp'}


def masked_source(text):
    """Blank comments/strings while retaining physical offsets and newlines."""
    return re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"',
                  lambda m: re.sub(r'[^\n]', ' ', m[0]), text)


def split_operands(text):
    parts, start, depth = [], 0, 0
    for index, char in enumerate(text):
        depth += (char in '[{(') - (char in ']})')
        if char == ',' and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return parts


class SourceIndex:
    """Lexical scope + conservative def/use and direct branch location index."""
    def __init__(self, text, entry):
        if not isinstance(entry, str) or not entry:
            raise ValueError('input.entry must name a PTX entry')
        self.text, self.lines, self.entry = text, text.splitlines(), entry
        self.starts = [0] + [m.end() for m in re.finditer('\n', text)]
        clean = masked_source(text)
        matches = list(re.finditer(r'\.entry\s+' + re.escape(entry) + r'(?=\s|\()', clean))
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one matching .entry; found {len(matches)}')
        match = matches[0]
        opening = clean.find('{', match.end())
        if opening < 0 or ';' in clean[match.end():opening]:
            raise ValueError('Matching entry has no body')
        depth, closing = 1, opening + 1
        while closing < len(clean) and depth:
            depth += (clean[closing] == '{') - (clean[closing] == '}')
            closing += 1
        if depth:
            raise ValueError('Matching entry has an unbalanced body')
        self.first, self.last = self.position(match.start())[0], self.position(closing - 1)[0]
        self.scopes = {0: None}
        self.scope_lines = {0: self.first}
        self.declarations = defaultdict(set)
        self.instructions, self.labels = [], {}
        scope, stack, start, vector_depth = 0, [0], opening + 1, 0

        def statement(end):
            nonlocal start
            raw = clean[start:end]
            offset = start + len(raw) - len(raw.lstrip())
            raw = raw.strip()
            label = re.match(r'^([A-Za-z_$][\w$]*):(?!=|:)', raw)
            if label:
                self.labels[(scope, label[1])] = self.position(offset)
                offset += label.end()
                rest = raw[label.end():]
                offset += len(rest) - len(rest.lstrip())
                raw = rest.strip()
            if not raw:
                return
            if raw.startswith('.reg'):
                for name, count in re.findall(r'(%?[A-Za-z_$][\w$]*)(?:<(\d+)>)?', raw.split(None, 2)[-1]):
                    if count:
                        # A range is resolved lazily, without expanding huge declarations.
                        self.declarations[scope].add((name, int(count)))
                    else:
                        self.declarations[scope].add((name, None))
                return
            if raw.startswith('.'):
                return
            instruction = re.match(r'(?:(@!?[%\w$]+)\s+)?([A-Za-z][\w.:]*)\s*(.*)', raw, re.S)
            if not instruction:
                return
            guard, opcode, operands = instruction.groups()
            line, column = self.position(offset)
            self.instructions.append({'line': line, 'column': column, 'scope': scope,
                                      'guard': guard, 'opcode': opcode,
                                      'operands': split_operands(operands),
                                      'text': ' '.join(raw.split()) + ';'})

        for token in re.finditer(r'[{};\n]', clean[opening + 1:closing - 1]):
            pos = opening + 1 + token.start()
            char, pending = token[0], clean[start:pos].strip()
            if char == '{':
                if pending or vector_depth:
                    vector_depth += 1
                else:
                    child = len(self.scopes)
                    self.scopes[child] = scope
                    self.scope_lines[child] = self.position(pos)[0]
                    stack.append(child)
                    scope, start = child, pos + 1
            elif char == '}':
                if vector_depth:
                    vector_depth -= 1
                else:
                    statement(pos)
                    stack.pop()
                    scope, start = stack[-1], pos + 1
            elif char == ';' and not vector_depth:
                statement(pos)
                start = pos + 1
            elif char == '\n' and not vector_depth:
                if re.fullmatch(r'[A-Za-z_$][\w$]*:', pending) or re.match(r'\.(?:loc|file|pragma)(?:\s|$)', pending):
                    statement(pos)
                    start = pos + 1
        statement(closing - 1)
        self.definitions = defaultdict(list)
        self.branches = []
        for inst in self.instructions:
            operands, opcode = inst['operands'], inst['opcode']
            first = opcode.split('.')[0]
            no_dest = first in NO_DEST or opcode.startswith(('tcgen05.', 'mbarrier.init', 'mbarrier.inval', 'mbarrier.expect_tx', 'mbarrier.complete_tx'))
            if opcode.startswith('tcgen05.ld.'):
                no_dest = False
            destinations = [] if no_dest else self.registers(operands[0], inst['scope'])
            inst['defs'] = destinations
            use_text = ','.join(operands if no_dest else operands[1:])
            inst['uses'] = self.registers(use_text + ' ' + (inst['guard'] or ''), inst['scope'])
            for register in destinations:
                self.definitions[register].append(inst)
            if first == 'bra':
                target = self.resolve_label(inst['scope'], operands[0])
                inst['target'] = target
                self.branches.append(inst)

    def position(self, offset):
        index = bisect_right(self.starts, offset) - 1
        return index + 1, offset - self.starts[index] + 1

    def resolve_register(self, name, scope):
        current = scope
        while current is not None:
            for declared, count in self.declarations[current]:
                if name == declared or (count is not None and name.startswith(declared)
                                        and name[len(declared):].isdigit()
                                        and int(name[len(declared):]) < count):
                    return current, name
            current = self.scopes[current]
        if name.startswith('%'):
            return 0, name
        return None

    def registers(self, text, scope):
        return list(dict.fromkeys(register for token in TOKEN.findall(text)
                                  if (register := self.resolve_register(token, scope))))

    def resolve_label(self, scope, name):
        while scope is not None:
            if (scope, name) in self.labels:
                return self.labels[(scope, name)]
            scope = self.scopes[scope]
        return None

    def register_name(self, key):
        scope, name = key
        return name if scope == 0 else f'{name}@scope-L{self.scope_lines[scope]}'

    def role_guards(self):
        # Syntactic dependency propagation includes all definitions and loops.
        # It establishes possible dependence, not a value or participating set.
        role = set()
        producers = []
        for inst in self.instructions:
            if SPECIAL.search(inst['text']):
                role.update(inst['defs'])
        for _ in range(16):
            previous = len(role)
            for inst in self.instructions:
                if inst['opcode'].split('.')[0] in ROLE_OPS and any(reg in role for reg in inst['uses']):
                    role.update(inst['defs'])
            if len(role) == previous:
                break
        for inst in self.instructions:
            if inst['guard'] and any(reg in role for reg in self.registers(inst['guard'], inst['scope'])):
                producers.append(inst)
        return producers

    def predecessor_branches(self, line, role_branches):
        """Bounded reverse reading of direct edges; no feasibility/dominance claim."""
        if not self.instructions:
            return [], False
        positions = [(i['line'], i['column']) for i in self.instructions]
        def instruction_at(position):
            return bisect_left(positions, position)
        leaders = {0}
        for position in self.labels.values():
            leaders.add(instruction_at(position))
        for number, inst in enumerate(self.instructions[:-1]):
            if inst['opcode'].split('.')[0] in {'bra', 'brx', 'ret', 'exit', 'trap'}:
                leaders.add(number+1)
        leaders = sorted(n for n in leaders if n<len(positions))
        predecessors = defaultdict(list)
        for block, start in enumerate(leaders):
            stop = leaders[block+1] if block+1<len(leaders) else len(positions)
            terminal = self.instructions[stop-1]
            opcode = terminal['opcode'].split('.')[0]
            destinations = []
            if opcode == 'bra' and terminal['target']:
                target = instruction_at(terminal['target'])
                if target<len(positions):
                    destinations.append((bisect_right(leaders, target)-1, 'taken'))
            if opcode not in {'bra','brx','ret','exit','trap'} or terminal['guard']:
                if block+1<len(leaders):
                    destinations.append((block+1, 'fallthrough'))
            for destination, edge in destinations:
                predecessors[destination].append((block, edge))
        point = min(instruction_at((line, 1)), len(positions)-1)
        queue, seen, found = deque([bisect_right(leaders, point)-1]), set(), []
        role_sites = {(i['line'],i['column']) for i in role_branches}
        while queue and len(seen)<96:
            block = queue.popleft()
            if block in seen:
                continue
            seen.add(block)
            for previous, edge in predecessors[block]:
                stop = leaders[previous+1] if previous+1<len(leaders) else len(positions)
                terminal = self.instructions[stop-1]
                if (terminal['line'],terminal['column']) in role_sites:
                    found.append({**terminal, 'edge': edge, 'edge_to': positions[leaders[block]]})
                queue.append(previous)
        return list({(i['line'],i['column'],i['edge'],i['edge_to']):i for i in found}.values()), any(block not in seen for block in queue)

    def address_key(self, operand, scope, depth=0):
        """Small syntactic base+offset cross-reference, only unique plain defs."""
        if depth>8:
            return None
        operand = operand.strip('[] ')
        try:
            return '', int(operand, 0)
        except ValueError:
            pass
        if re.fullmatch(r'[A-Za-z_$][\w$]*', operand) and not self.resolve_register(operand, scope):
            return operand, 0
        register = self.resolve_register(operand, scope)
        definitions = self.definitions.get(register, [])
        if len(definitions)!=1 or definitions[0]['guard']:
            return None
        inst = definitions[0]
        operands = inst['operands']
        base = inst['opcode'].split('.')[0]
        if base in {'mov', 'cvta'} and len(operands)==2:
            return self.address_key(operands[1], inst['scope'], depth+1)
        if base in {'add','sub'} and len(operands)==3:
            left = self.address_key(operands[1], inst['scope'], depth+1)
            right = self.address_key(operands[2], inst['scope'], depth+1)
            if left and right and not right[0]:
                return left[0], left[1] + right[1] * (-1 if base=='sub' else 1)
        return None


def family(opcode):
    for prefix in ('mbarrier.init', 'mbarrier.arrive', 'mbarrier.try_wait', 'mbarrier.test_wait',
                   'mbarrier.inval', 'tcgen05.alloc', 'tcgen05.dealloc', 'tcgen05.ld',
                   'tcgen05.st', 'tcgen05.wait', 'tcgen05.commit', 'tcgen05.mma',
                   'tcgen05.cp', 'cp.async', 'st.shared', 'ld.shared', 'stmatrix',
                   'barrier', 'bar.', 'fence'):
        if opcode.startswith(prefix):
            return prefix.rstrip('.')
    return None


def saved_sites(result, index, ptx_path):
    """Preserve source coordinates, prioritizing explicit blocked-event endpoints."""
    sites, rejected = defaultdict(list), []
    counterexample = result.get('counterexample') or {}
    priority = {counterexample.get('first_unexecuted')}
    priority.update((counterexample.get('unexecuted_candidate') or {}).get('events', []))
    first = []

    def add(location, origin, explicit_ptx=False):
        line = location.get('line')
        if type(line) is not int:
            return
        source = location.get('file')
        if not source and not explicit_ptx:
            rejected.append(f'{origin}: line {line} has no source-file identity; not a PTX mapping')
        elif source and not (str(source) == str(ptx_path) or Path(source).as_posix() == Path(ptx_path).as_posix()
                           or (not Path(source).is_absolute() and str(Path(ptx_path)).endswith('/' + str(source)))):
            rejected.append(f'{origin}: foreign file {source}:{line}')
        elif not index.first <= line <= index.last:
            rejected.append(f'{origin}: line {line} outside matching entry')
        else:
            sites[line].append(origin + f" (saved column {location.get('column', 'unspecified')})")
    for item in result.get('source_events', []):
        if not isinstance(item, dict):
            continue
        add(item, f"event {item.get('event')} / dynamic instruction {item.get('instruction')}")
        if item.get('event') in priority and item.get('line') in sites:
            first.append(item['line'])

    def visit(value, path):
        if isinstance(value, dict):
            if 'line' in value and ('file' in value or path.endswith('.location') or path=='location'):
                add(value, path)
            if type(value.get('ptx_line')) is int:
                add({'line': value['ptx_line']}, path + '.ptx_line', explicit_ptx=True)
            for key, child in value.items():
                visit(child, path + '.' + key)
        elif isinstance(value, list):
            for number, child in enumerate(value):
                visit(child, f'{path}[{number}]')
    for key in ('source_violations', 'warnings', 'counterexample', 'races',
                'unsupported_memory_pair', 'location', 'diagnostics'):
        visit(result.get(key), key)
    return sites, list(dict.fromkeys(first + sorted(sites))), rejected


def representative_sites(result, sites, saved):
    """One accepted coordinate per diagnostic kind before repeated sites."""
    representatives, kinds = [], set()
    for line in saved:
        for origin in sites[line]:
            match = re.match(r'(diagnostics|source_violations|warnings)\[(\d+)\]', origin)
            if not match:
                continue
            collection, number = match[1], int(match[2])
            record = result[collection][number]
            kind = (collection, record.get('stage'), record.get('code', str(number)))
            if kind not in kinds:
                kinds.add(kind)
                representatives.append(line)
    # Keep an explicit blocked-event endpoint first, even when it has no diagnostic.
    witness = result.get('counterexample') or {}
    priority = {witness.get('first_unexecuted')}
    priority.update((witness.get('unexecuted_candidate') or {}).get('events', []))
    blocked = [line for line in saved if any(
        origin.startswith(f'event {event} /') for event in priority if event is not None
        for origin in sites[line])]
    return list(dict.fromkeys(blocked + representatives)) if blocked or representatives else saved


def stored_integer_bits(inst):
    """Decode a literal scalar integer store; never infer register values."""
    match = re.fullmatch(r'st(?:\.[a-z_:]+)*\.[bsu](8|16|32|64)', inst['opcode'])
    if not match or len(inst['operands']) != 2:
        return None
    literal = inst['operands'][1]
    if not re.fullmatch(r'-?(?:0x[0-9a-fA-F]+|[0-9]+)', literal):
        return None
    bits = int(match[1])
    value = int(literal, 16 if '0x' in literal else 10)
    if not -(1 << (bits - 1)) <= value < (1 << bits):
        return None
    unsigned = value & ((1 << bits) - 1)
    octets = ' '.join(f'{byte:02x}' for byte in unsigned.to_bytes(bits // 8, 'little'))
    return f'literal bit pattern 0x{unsigned:0{bits // 4}x}; little-endian bytes {octets}'


def predicate_requirement(index, register, value):
    """Expand only unique unconditional predicate copies/inversions, not paths."""
    result, seen = [], set()
    while register not in seen and len(seen) < 8:
        seen.add(register)
        definitions = index.definitions.get(register, [])
        if len(definitions) != 1 or definitions[0]['guard']:
            break
        inst = definitions[0]
        if inst['opcode'] not in {'not.pred', 'mov.pred'} or len(inst['uses']) != 1:
            break
        if inst['opcode'] == 'not.pred':
            value = not value
        register = inst['uses'][0]
        result.append(f"{index.register_name(register)}={str(value).lower()} via PTX {inst['line']}")
    return result


def dependency_slice(index, roots, limit=24, levels=8):
    """Bounded candidates, retaining repeated writes rather than selecting one."""
    queue = deque((root, 0) for root in roots)
    seen, rows = set(), []
    while queue and len(seen) < limit:
        register, depth = queue.popleft()
        if register in seen:
            continue
        seen.add(register)
        definitions = index.definitions.get(register, [])
        choices = definitions if len(definitions) <= 6 else definitions[:3] + definitions[-3:]
        rows.append((register, definitions, choices))
        if depth < levels:
            dependencies = [reg for inst in choices for reg in inst['uses']]
            queue.extendleft((reg, depth + 1) for reg in reversed(dependencies))
        elif any(inst['uses'] for inst in choices):
            rows.append((None, [], []))
    return rows, bool(queue) or any(row[0] is None for row in rows)


def short(text, limit=MAX_LINE):
    return text if len(text) <= limit else text[:limit] + ' … [text clipped; read source at this coordinate]'


def coords(items, limit=24):
    values = [f"{i['line']}:{i['column']}" for i in items]
    if len(values) > limit:
        half = limit // 2
        return ', '.join(values[:half]) + f' … {len(values)-limit} omitted … ' + ', '.join(values[-half:])
    return ', '.join(values)


def supplied_parameter_bindings(run_input, ptx):
    """Reuse the report's parameter-load index for supplied argument entrances."""
    declarations = list(run_input.get('declarations', []))
    by_name = {d.get('name'):d for d in declarations if isinstance(d, dict) and d.get('name')}
    values = {}
    for argument in run_input.get('arguments', []):
        if not isinstance(argument, dict) or 'value' not in argument:
            continue
        field = argument.get('field')
        declaration = by_name.get(field, argument)
        if 'parameter' not in declaration:
            continue
        parameter, offset = declaration['parameter'], declaration.get('offset', 0)
        label = field or f'parameter {parameter} + {offset}'
        declarations.append({'parameter': parameter, 'offset': offset, 'name': label})
        values[label] = {'parameter': parameter, 'offset': offset, 'value': argument['value'],
                         'type': argument.get('type', declaration.get('type', 'unspecified'))}
    enriched = {**run_input, 'declarations': declarations}
    return values, parameter_bindings(enriched, ptx)


def render(run_input, result, ptx, ptx_path='kernel.ptx', at=()):
    index = SourceIndex(ptx, run_input.get('entry'))
    if len(at) > 12:
        raise ValueError('--at accepts at most 12 physical source lines per packet')
    if any(type(line) is not int or not index.first <= line <= index.last for line in at):
        raise ValueError('--at line is outside the matching entry')
    sites, saved, rejected = saved_sites(result, index, ptx_path)
    groups = defaultdict(list)
    for inst in index.instructions:
        if group := family(inst['opcode']):
            groups[group].append(inst)
    sections = []
    def emit(text):
        sections.append(text)
    def instruction(inst):
        text = f"{inst['line']}:{inst['column']}  {short(inst['text'])}"
        bits = stored_integer_bits(inst)
        return text + (f"  [{bits}; not a runtime observation]" if bits else '')
    emit('# PTX source inspection packet\n')
    emit(f'Source: `{ptx_path}`; physical entry lines **{index.first}–{index.last}**; {len(index.instructions)} indexed statements.\nEntry: `{index.entry}`\n')
    emit('This is a bounded static evidence index, not a complete CFG, reaching-definition proof, dynamic event mapping, or hardware ordering proof. Direct edges and syntactic dependencies are reading leads. Repeated definitions include later/loop/predicated writes; lexical proximity does not select the reaching definition. Missing dynamic mapping does not erase static evidence.\n')
    emit('## Saved source endpoints\n')
    if sites:
        for line in saved[:24]:
            origins = sites[line]
            emit(f'- L{line}: ' + '; '.join(origins[:3]) + (f'; {len(origins)-3} more saved references' if len(origins)>3 else ''))
        if len(saved)>24:
            emit(f'{len(saved)-24} additional distinct saved lines omitted.')
    else:
        emit('No accepted saved source-line endpoints. Dynamic instruction IDs are not PTX line numbers. The operation-family index below is static candidate evidence, not a mapping of lifecycle IDs.')
    if rejected:
        emit('Rejected source coordinates: ' + '; '.join(rejected[:5]) + (f'; {len(rejected)-5} more' if len(rejected)>5 else ''))
    values, bindings = supplied_parameter_bindings(run_input, '\n'.join(index.lines[index.first-1:index.last]))
    emit('\n## Supplied parameter entrances\nSaved values and parameter-load locations only; no value propagation or branch feasibility is inferred. Missing declaration names do not suppress supplied scalar arguments.\n')
    for name, value in list(values.items())[:32]:
        parameter_loads = [b for b in bindings if b['field']==name]
        locations = ', '.join(f"{b['register']}@{b['line']+index.first-1}" for b in parameter_loads[:4]) or 'no indexed load'
        emit(f"- {name} ({value['type']}) = {short(json.dumps(value['value']), 100)}; {locations}"
             + (f'; {len(parameter_loads)-4} more loads' if len(parameter_loads)>4 else ''))
    if len(values)>32:
        emit(f'{len(values)-32} additional supplied arguments omitted.')
    literal_stores = [inst for inst in index.instructions if stored_integer_bits(inst)]
    if literal_stores:
        emit('\n## Scalar literal stores\nComputed integer bit patterns, useful for packed control words. These do not establish execution or ordering.\n')
        for inst in (literal_stores if len(literal_stores) <= 12 else literal_stores[:6] + literal_stores[-6:]):
            emit(instruction(inst))
        if len(literal_stores) > 12:
            emit(f'{len(literal_stores) - 12} additional literal store sites omitted.')
    barrier_addresses = defaultdict(list)
    for inst in index.instructions:
        if inst['opcode'].startswith('mbarrier.'):
            operand = next((part for part in inst['operands'] if part.startswith('[')), None)
            if operand and (key := index.address_key(operand, inst['scope'])) is not None:
                barrier_addresses[key].append(inst)
    displayed_barriers = set()
    role = index.role_guards()
    role_branches = [i for i in role if i['opcode'].startswith('bra')]
    ranked = representative_sites(result, sites, saved)
    loads = groups.get('tcgen05.ld', [])
    load_waits = [i for i in groups.get('tcgen05.wait', []) if '::ld' in i['opcode']]
    anchors = list(dict.fromkeys(at or ranked[:12] or (
        [i['line'] for i in groups.get('tcgen05.dealloc', [])][:2]
        + [i['line'] for i in (loads[:1] + loads[-1:] + load_waits[:1])]
        + [i['line'] for i in groups.get('mbarrier.init', [])][:1])))
    emit('\n## Focus evidence — read before assigning roles or causes\n'
         'Representative diagnostic categories precede repeated sites. Each slice shows '
         'candidate definitions, not selected reaching definitions. A one-sided reverse '
         'edge is a path lead, not proof of participation; both-sided branches are grouped '
         'separately because either edge can reach this operation. Read original source '
         'when a slice is clipped.\n')
    for line in anchors[:8]:
        operations = [i for i in index.instructions if i['line'] == line]
        emit(f'### Focus L{line}\n')
        for inst in operations:
            emit(instruction(inst))
        roots = []
        for inst in operations:
            # Async copy sizes/guards are often the stopping condition; address
            # calculations are still indexed, after those scalar roots.
            operands = list(reversed(inst['operands'])) if inst['opcode'].startswith('cp.async') else inst['operands']
            roots.extend(index.registers(inst['guard'] or '', inst['scope']))
            roots.extend(reg for operand in operands for reg in index.registers(operand, inst['scope'])
                         if reg in inst['uses'])
        leads, bounded = index.predecessor_branches(line, role_branches)
        edges = defaultdict(set)
        for lead in leads:
            edges[(lead['line'], lead['column'])].add(lead['edge'])
        distinct = [lead for lead in leads if len(edges[(lead['line'], lead['column'])]) == 1]
        selected = distinct[:6]
        for lead in selected:
            predicate = lead['guard'].lstrip('@!')
            value = (lead['edge'] == 'taken') != lead['guard'].startswith('@!')
            emit('Role path candidate: ' + instruction(lead)
                 + f" {lead['edge']} edge -> block {lead['edge_to'][0]}:{lead['edge_to'][1]}; requires {predicate}={str(value).lower()}")
            register = index.resolve_register(predicate, lead['scope'])
            conditions = predicate_requirement(index, register, value)
            if conditions:
                emit('  Predicate substitution: ' + ' -> '.join(conditions)
                     + '; reaching writes and path feasibility remain to be checked.')

        emit(f'Reverse search: {len(distinct)} one-sided leads; '
             f'{sum(len(v) == 2 for v in edges.values())} branches with both edges reaching the focus; '
             f'search truncated={str(bounded).lower()}. No dominance/feasibility claim.')
        rows, clipped = dependency_slice(index, roots)
        role_rows, role_clipped = dependency_slice(index, [reg for lead in selected for reg in lead['uses']], limit=16)
        rows.extend(role_rows)
        printed_registers = set()
        printed_definitions = set()
        for register, definitions, choices in rows:
            if register is None or register in printed_registers:
                continue
            printed_registers.add(register)
            name = index.register_name(register)
            if not definitions:
                emit(f'  {name}: ' + ('special-register input' if SPECIAL.fullmatch(register[1]) else 'no indexed definition'))
                continue
            label = 'REPEATED definitions; none selected as reaching' if len(definitions) > 1 else 'indexed definition'
            emit(f'  {name}: {len(definitions)} {label}; {coords(definitions, 6)}')
            for inst in choices:
                key = inst['line'], inst['column']
                if key not in printed_definitions:
                    emit('    ' + instruction(inst))
                    printed_definitions.add(key)
        if clipped or role_clipped:
            emit(f'  Slice truncated at 24 operand / 16 role registers / eight levels. Read source definitions above or use --at {line}.')
        if any(inst['opcode'].startswith(('mbarrier.', 'tcgen05.dealloc', 'tcgen05.ld')) for inst in operations):
            nearby = sorted((inst for items in barrier_addresses.values() for inst in items),
                            key=lambda inst: abs(inst['line'] - line))[:3]
            related = [(key, items) for key, items in barrier_addresses.items()
                       if key not in displayed_barriers and any(inst in nearby for inst in items)]
            for key, items in related[:2]:
                displayed_barriers.add(key)
                emit(f'Nearby barrier address group: {short(key[0], 100) or "constant"} + {key[1]}')
                emit('  Same syntactic address only; establish participants and completed phases before treating this as a release protocol.')
                for inst in items[:16]:
                    emit('  ' + instruction(inst))
                    for reg in inst['uses']:
                        definitions = index.definitions.get(reg, [])
                        if len(definitions) == 1 and definitions[0]['opcode'].startswith('mov.') and not definitions[0]['guard']:
                            emit('    Operand source: ' + instruction(definitions[0]))
                if len(items) > 16:
                    emit(f'  {len(items) - 16} additional address-group sites omitted; inspect original source.')
    if len(anchors) > 8:
        emit('Additional focus lines not expanded here: ' + ', '.join(map(str, anchors[8:])) + '; use --at.')
    emit('\n## Matching-entry operation families\nAll counts cover this entry; coordinate lists are bounded (first/last samples). Read omitted sites with --at.\n')
    for group, items in groups.items():
        emit(f'- {group}: {len(items)} sites; {coords(items)}')

    emit('\n## Thread-role / guard dependency leads\nGuards with syntactic dependence on tid/lane/warp registers; no participant-count or branch-feasibility claim. Election and memory-derived control may need separate reading.\n')
    special_reads = [i for i in index.instructions if SPECIAL.search(i['text'])]
    for inst in special_reads[:12]:
        emit('Special-register source: ' + instruction(inst))
    selected_roles = role_branches[:32]
    if at:
        selected_roles = sorted(role_branches, key=lambda i: min(abs(i['line']-n) for n in at))[:16]
        selected_roles.sort(key=lambda i:i['line'])
    for inst in selected_roles:
        emit(instruction(inst) + f"  target={inst.get('target') or 'unresolved'}")
    emit(f'Shown {len(selected_roles)}/{len(role_branches)} role-dependent direct branches; {len(role)-len(role_branches)} other predicated statements indexed.')

    # Always retain counts/addresses for initialization and terminal deallocation;
    # saved endpoints (or --at) choose detailed local neighborhoods.
    sync = [i for i in index.instructions if family(i['opcode']) in
            {'mbarrier.init','mbarrier.arrive','mbarrier.try_wait','mbarrier.test_wait','tcgen05.wait','tcgen05.dealloc'}]
    focus = [i for i in index.instructions if i['line'] in anchors]
    for line in anchors:
        neighbors = sorted(sync, key=lambda i:abs(i['line']-line))[:3]
        focus.extend(neighbors)
    if not at:
        focus.extend(groups.get('mbarrier.init', [])[:32])
        focus.extend(groups.get('tcgen05.dealloc', [])[:2] + loads[:1] + load_waits[:1])
    focus = list({(i['line'],i['column']):i for i in focus}.values())
    emit('\n## Synchronization / lifecycle operands\nEach instruction retains its actual address, count, state/parity and predicate operands. Equal register spellings or similar addresses alone do not prove a shared barrier generation.\n')
    for inst in sorted(focus, key=lambda i:(i['line'],i['column']))[:64]:
        emit(instruction(inst))

    emit('\n## Barrier address-expression cross-references\nUnique, unpredicated mov/cvta/add/sub definitions only; shared symbolic base+offset is a reading lead, not an alias, generation or ordering proof. Other expressions remain unresolved.\n')
    addresses = barrier_addresses
    matches = [(key, items) for key,items in addresses.items() if len(items)>1]
    for (base, offset), items in matches[:24]:
        emit(f'- {short(base, 100) or "constant"} + {offset}: ' + ', '.join(f"{i['opcode']}@{i['line']}" for i in items[:12])
             + (f'; {len(items)-12} additional sites omitted' if len(items)>12 else ''))
        if any(inst in focus and not inst['opcode'].startswith('mbarrier.init') for inst in items):
            # An address match can be distant and in a different role path.
            # Expand its arrivals/init/waits rather than stopping at a local window.
            for inst in items[:12]:
                if inst not in focus:
                    focus.append(inst)
                    emit('  Same-expression candidate: ' + instruction(inst))
    if len(matches)>24:
        emit(f'{len(matches)-24} additional shared expression groups omitted.')

    emit('\n## Definition candidates to read\nAll indexed definitions of each displayed register are counted; up to six candidates are followed per register. Scoped inline registers stay separate. The slice is bounded at 150 registers / six dependency levels.\n')
    queue = deque()
    # Role roots precede operand slices, to retain the dispatch chain in big kernels.
    for inst in selected_roles + focus:
        queue.extend((reg, 0) for reg in inst['uses'])
    seen, definition_lines = set(), set()
    while queue and len(seen) < 150:
        register, depth = queue.popleft()
        if register in seen:
            continue
        seen.add(register)
        definitions = index.definitions.get(register, [])
        if not definitions:
            # Special registers are inputs, not missing source evidence.
            if SPECIAL.fullmatch(register[1]):
                emit(f'- {index.register_name(register)}: special-register input')
            else:
                emit(f'- {index.register_name(register)}: no indexed definition; inspect parameter/special-register/call or unsupported syntax')
            continue
        label = 'REPEATED definitions; none selected as reaching' if len(definitions)>1 else 'indexed definition'
        emit(f'- {index.register_name(register)}: {len(definitions)} {label}; {coords(definitions, 6)}')
        choices = definitions if len(definitions)<=6 else definitions[:3]+definitions[-3:]
        for inst in choices:
            key = inst['line'], inst['column']
            if key not in definition_lines:
                emit('  ' + instruction(inst))
                definition_lines.add(key)
            if depth < 6:
                queue.extendleft((reg, depth+1) for reg in reversed(inst['uses']))
    if queue:
        emit(f'Definition traversal stopped at budget; {len(queue)} queued references not expanded. Use --at for a focused packet.')

    emit('\n## Local neighborhoods and direct path locations\nPhysical source order is not execution order. Incoming branches below target the nearest preceding label; this does not establish dominance.\n')
    printed = set()
    for line in anchors[:12]:
        emit(f'\nAnchor L{line}:')
        for number in range(max(index.first, line-3), min(index.last, line+3)+1):
            if number not in printed:
                emit(f'{number}: {short(index.lines[number-1])}')
                printed.add(number)
        nearby = next((i for i in index.instructions if i['line']>=line), None)
        scope, enclosing = nearby['scope'] if nearby else 0, set()
        while scope is not None:
            enclosing.add(scope)
            scope = index.scopes[scope]
        labels = [(key, pos) for key,pos in index.labels.items() if pos[0]<=line and key[0] in enclosing]
        if labels:
            key, position = max(labels, key=lambda item:item[1])
            emit(f'Nearest preceding label: {key[1]} at {position[0]}:{position[1]} (lexical scope L{index.scope_lines[key[0]]}).')
            incoming = [i for i in index.branches if i['target']==position]
            for branch in incoming[:4]:
                emit('Incoming direct branch: ' + instruction(branch))
        previous = [i for i in index.branches if i['line']<line]
        for branch in previous[-2:]:
            emit('Nearby earlier branch (not necessarily a guard): ' + instruction(branch) + f" target={branch['target'] or 'unresolved'}")
        leads, bounded = index.predecessor_branches(line, role_branches)
        for branch in leads[:6]:
            predicate = branch['guard'].lstrip('@!')
            value = (branch['edge']=='taken') != branch['guard'].startswith('@!')
            emit('Reverse direct-edge role lead: ' + instruction(branch)
                 + f" {branch['edge']} edge -> block {branch['edge_to'][0]}:{branch['edge_to'][1]}; requires {predicate}={str(value).lower()}")
        if bounded or len(leads)>6:
            emit('Reverse path search truncated (96 blocks / six displayed role branches); no exclusion or dominance claim.')
    indirect = [i for i in index.instructions if i['opcode'].startswith(('brx.', 'call'))]
    if indirect:
        emit(f'Indirect branches/calls require additional inspection: {coords(indirect)}')
    emit('\nContinue with --at line,... using physical PTX lines above. Compare initialization counts, arrivals, wait parity/state, role branches and alternate entries/backedges before explaining participants or ordering. Unlisted source is not evidence of absence.')
    packet = '\n'.join(sections) + '\n'
    if len(packet)>MAX_CHARS:
        packet = packet[:MAX_CHARS].rsplit('\n',1)[0] + '\n\n[Packet character budget reached; remaining sections omitted. Use --at line,... to focus. No completeness claim.]\n'
    return packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--result', required=True, type=Path)
    parser.add_argument('--ptx', required=True, type=Path)
    parser.add_argument('--at', default='', help='Up to 12 comma-separated physical PTX lines')
    args = parser.parse_args()
    try:
        at = tuple(int(v) for v in args.at.split(',')) if args.at else ()
        run_input = json.loads(args.input.read_text())
        result = json.loads(args.result.read_text())
        print(render(run_input, result, args.ptx.read_text(), str(args.ptx), at), end='')
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
