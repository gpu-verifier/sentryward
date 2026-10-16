#!/usr/bin/env python3
"""Render saved facts into one report, without interpreting kernel semantics.

Default: print a complete base report. --report writes report.md; --draft adds
an evidence-linked explanation. No sidecars, dependencies, or verifier execution.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re

START = '<!-- saved-evidence:start -->'
END = '<!-- saved-evidence:end -->'
RESULT_START = '<!-- saved-result:start -->'
RESULT_END = '<!-- saved-result:end -->'
RELATION_START = '<!-- recorded-relation:start -->'
RELATION_END = '<!-- recorded-relation:end -->'
WAIT_START = '<!-- wait-guarantee:start -->'
WAIT_END = '<!-- wait-guarantee:end -->'
MISSING = object()


def parameter_bindings(run_input, ptx):
    """Index parameter loads in the selected entry; no branch execution inferred."""
    entry = run_input.get('entry')
    if not isinstance(entry, str):
        return []
    match = re.search(r'\.entry\s+' + re.escape(entry) + r'(?=\s|\()', ptx)
    if not match:
        return []
    opening = ptx.find('{', match.end())
    if opening == -1:
        return []
    depth = 0
    end = None
    for index in range(opening, len(ptx)):
        if ptx[index] == '{':
            depth += 1
        elif ptx[index] == '}':
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end is None:
        raise ValueError('Selected PTX entry has an unbalanced body.')
    declarations = {(str(d.get('parameter')), d.get('offset', 0)): d.get('name')
                    for d in run_input.get('declarations', []) if isinstance(d, dict)}
    aliases = {}
    result = []
    for number, line in enumerate(ptx[:end].splitlines(), 1):
        if number <= ptx[:opening].count('\n'):
            continue
        code = line.split('//', 1)[0]
        assignment = re.search(r'\b[\w.]+\s+(%\w+)\s*,', code)
        alias = re.search(r'\bmov\.\w+\s+(%\w+),\s*' + re.escape(entry) + r'_param_(\d+)\s*;', code)
        if assignment:
            aliases.pop(assignment.group(1), None)
        if alias:
            aliases[alias.group(1)] = alias.group(2)
        load = re.search(r'\bld\.param\.\w+\s+(%\w+),\s*\[([^\]]+)\]', code)
        if not load:
            continue
        address = load.group(2).replace(' ', '')
        direct = re.fullmatch(re.escape(entry) + r'_param_(\d+)(?:\+(\d+))?', address)
        indirect = re.fullmatch(r'(%\w+)(?:\+(\d+))?', address)
        parameter = direct.group(1) if direct else aliases.get(indirect.group(1)) if indirect else None
        offset = int((direct or indirect).group(2) or 0) if direct or indirect else None
        name = declarations.get((parameter, offset))
        if name is not None:
            result.append({'field': name, 'register': load.group(1), 'line': number, 'offset': offset})
    return result


def constant_predicates(run_input, ptx, bindings):
    """Fold integer-scalar predicates with unique definitions; no CFG inference."""
    entry = run_input.get('entry', '')
    start = re.search(r'\.entry\s+' + re.escape(entry) + r'(?=\s|\()', ptx)
    if not start:
        return []
    next_entry = re.search(r'\.(?:entry|func)\b', ptx[start.end():])
    end = start.end() + next_entry.start() if next_entry else len(ptx)
    lines = ptx[:end].splitlines()
    first_line = ptx[:start.start()].count('\n') + 1
    definitions = {}
    for number, line in enumerate(lines, 1):
        if number < first_line:
            continue
        code = line.split('//', 1)[0].strip()
        match = re.match(r'(?:@!?%\w+\s+)?([\w.]+)\s+(%\w+),\s*(.*?)\s*;', code)
        if match:
            definitions.setdefault(match.group(2), []).append((number, match.group(1), match.group(3)))
    names = {(str(d.get('parameter')), d.get('offset', 0)): d.get('name')
             for d in run_input.get('declarations', []) if isinstance(d, dict)}
    arguments = {}
    for argument in run_input.get('arguments', []):
        if not isinstance(argument, dict) or 'value' not in argument:
            continue
        name = argument.get('field') or names.get((str(argument.get('parameter')), argument.get('offset', 0)))
        if name is not None:
            arguments[name] = argument['value']
    types = {d.get('name'): d.get('type') for d in run_input.get('declarations', []) if isinstance(d, dict)}
    known = {}
    named_inputs = {}
    for binding in bindings:
        register = binding['register']
        value = arguments.get(binding['field'])
        width = re.fullmatch(r'[us](\d+)', str(types.get(binding['field'])))
        load = definitions.get(register, [])
        load_width = re.fullmatch(r'ld\.param\.[bus](\d+)', load[0][1]) if len(load) == 1 else None
        integer = width and load_width and width.group(1) == load_width.group(1) and type(value) is int and 0 <= value < 2 ** int(width.group(1))
        null_pointer = types.get(binding['field']) == 'pointer' and binding['field'] in arguments and value is None and load_width and load_width.group(1) in ('32','64')
        if integer or null_pointer:
            known[register] = 0 if null_pointer else value
            named_inputs[register] = binding['field']
    proofs = []
    def value(token):
        token = token.strip()
        if token in known:
            return known[token]
        try:
            return int(token, 0)
        except ValueError:
            return None
    for register, instructions in sorted(definitions.items(), key=lambda item: item[1][0][0]):
        if len(instructions) != 1:
            continue
        number, opcode, raw = instructions[0]
        operands = [part.strip() for part in raw.split(',')]
        values = [value(part) for part in operands]
        folded = None
        if re.fullmatch(r'setp\.(?:eq|ne)\.(?:b|s|u)\d+', opcode) and len(values) == 2 and all(v is not None and v >= 0 for v in values):
            folded = values[0] == values[1] if '.eq.' in opcode else values[0] != values[1]
        elif opcode == 'or.pred' and len(values) == 2:
            folded = True if any(v is True for v in values) else False if all(v is False for v in values) else None
        elif opcode == 'and.pred' and len(values) == 2:
            folded = False if any(v is False for v in values) else True if all(v is True for v in values) else None
        elif opcode == 'not.pred' and len(values) == 1 and type(values[0]) is bool:
            folded = not values[0]
        if folded is not None:
            known[register] = folded
            proofs.append({'predicate': register, 'value': folded, 'line': number, 'expression': opcode + ' ' + raw,
                           'inputs': {named_inputs[token]: known[token] for token in operands if token in named_inputs}})
    return proofs


def predicate_table(proofs):
    if not proofs:
        return ''
    body = 'Unique-definition scalar and null-pointer simplifications. Values apply **when these definitions are reached**; they do not prove path reachability. A true OR operand fixes the result even if the other operand is unknown.\n\n'
    body += '| Predicate | Value / physical PTX | Definition |\n| --- | --- | --- |\n'
    for proof in proofs:
        definition = key_cell(proof['expression'])
        if proof.get('inputs'):
            definition += '<br/>' + ', '.join(key_cell(field) + ' = ' + cell(value) for field, value in proof['inputs'].items())
        body += '| ' + key_cell(proof['predicate']) + ' | ' + cell(proof['value']) + ' / ' + str(proof['line']) + ' | ' + definition + ' |\n'
    return body


def skipped_parameter_loads(run_input, ptx, bindings, proofs):
    """Forward known-taken branches, excluding outside entries into their gap."""
    entry = run_input.get('entry')
    if not isinstance(entry, str):
        return []
    match = re.search(r'\.entry\s+' + re.escape(entry) + r'(?=\s|\()', ptx)
    if not match:
        return []
    opening = ptx.find('{', match.end())
    if opening == -1:
        return []
    depth = 0
    closing = None
    for index in range(opening, len(ptx)):
        if ptx[index] == '{':
            depth += 1
        elif ptx[index] == '}':
            depth -= 1
            if depth == 0:
                closing = index
                break
    if closing is None:
        return []
    lines = ptx.splitlines()
    first = ptx[:opening].count('\n') + 1
    last = ptx[:closing].count('\n') + 1
    body = [(n, line.split('//', 1)[0]) for n, line in enumerate(lines, 1)
            if first <= n <= last]
    if any(re.search(r'\bbrx\.', line) for _, line in body):
        return []
    values = {p['predicate']: (p['value'], p['line']) for p in proofs}
    # Include every direct branch in the selected entry, including branches
    # after the final parameter load that could re-enter a skipped gap.
    labels = {m.group(1): n for n, line in body
              if (m := re.match(r'\s*([A-Za-z_$][\w$]*):', line))}
    branches = []
    for n, line in body:
        match = re.match(r'\s*(?:@(!?)(%\w+)\s+)?bra(?:\.uni)?\s+([A-Za-z_$][\w$]*)\s*;', line)
        if match:
            branches.append((n, match.group(1), match.group(2), match.group(3)))
    skipped = []
    for n, negative, predicate, target in branches:
        if predicate not in values or target not in labels:
            continue
        value, definition = values[predicate]
        if definition >= n:
            continue
        taken = not value if negative else value
        end = labels[target]
        if not taken or end <= n:
            continue
        inner_labels = {label for label, where in labels.items() if n < where < end}
        if any((where <= n or where >= end) and destination in inner_labels
               for where, _, _, destination in branches):
            continue
        for binding in bindings:
            if n < binding['line'] < end:
                proof = next(p for p in proofs if p['predicate'] == predicate)
                skipped.append({**binding, 'branch': n, 'target': end, 'predicate': predicate,
                                'condition': proof['expression'], 'inputs': proof.get('inputs', {})})
    return skipped


def skipped_table(items):
    if not items:
        return ''
    body = 'Forward branches fixed by supplied integer values bypass these parameter loads. No outside direct branch enters the skipped gap. This is static source support, not a saved runtime trace.\n\n'
    body += '| Load / field | Known-taken branch | Target |\n| --- | --- | --- |\n'
    for item in items:
        guard = key_cell(item['predicate'])
        if item.get('inputs'):
            guard += '<br/>' + ', '.join(key_cell(field) + ' = ' + cell(value) for field, value in item['inputs'].items())
        body += f'| {item["line"]} / ' + key_cell(item['field']) + f' | {item["branch"]} / ' + guard + f' | {item["target"]} |\n'
    return body


def wait_guarantee(ptx):
    if not re.search(r'\.version\s+9\.2\b', ptx):
        return ''
    if 'cp.async.bulk.wait_group.read 1;' not in ptx:
        return ''
    visible = '**The thread can continue while its copy is unfinished.**\n\n'
    visible += 'Local instruction order ↓ · one issuing thread. Cells show guarantees, not a recorded runtime schedule.\n\n'
    visible += '| Issuing thread ↓ | Source reads (shared) | Destination writes (global) |\n| --- | --- | --- |\n'
    visible += '| Issue store → commit group | Asynchronous work | Asynchronous work |\n'
    visible += '| `.read 1` returns | Older groups: **finished**<br/>Newest group: may remain pending | May remain pending |\n\n'
    visible += 'Returning from this wait does **not order another thread’s stores**.\n'
    body = 'A group collects one issuing thread’s copies at commit; commit itself does not finish them.\n\n'
    body += 'The operand **1** permits at most the newest group’s source reads to remain pending. This is an instruction guarantee, not a saved completion state.\n\n'
    body += 'A reached `.read 0` would finish all of that thread’s source reads. It still would not guarantee destination-write completion or order different issuing threads.\n\n'
    body += '[PTX 9.2 bulk wait](https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group)\n'
    return WAIT_START + '\n' + visible + '\n' + fold('<b>Read-1 versus read-0</b> · exact completion guarantee', body) + WAIT_END + '\n'


def instruction_notes(ptx):
    if not re.search(r'^\s*\.version\s+9\.2\b', ptx, re.M):
        return ''
    if not ('tcgen05.cp.' in ptx and re.search(r'tcgen05\.mma(?:\.sp)?\.', ptx)):
        return ''
    return ('**PTX 9.2 source-check boundary.** `tcgen05.cp.cta_group::N` → `tcgen05.mma.cta_group::N` is a pipelined pairing for the same N. '
            'The same-thread pattern needs no explicit ordering mechanism. Before attributing a candidate to a missing wait, check the issuing-thread relation and applicability of this pairing. '
            'A saved effect byte count is not the full instruction transfer; do not convert TMEM register offsets into saved byte offsets without a supplied mapping.\n\n'
            '[Pipelined pairings](https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html#tcgen05-memory-consistency-model-pipelined-instructions) · '
            '[Same-thread pattern](https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html#tcgen05-memory-consistency-model-canonical-sync-patterns-pipelined-same-thread)\n')


def binding_table(bindings):
    body = 'Parameter loads in the selected PTX entry. **Listing a load does not prove its branch is reached.** Pointer and stride are different fields. For a packed load, the name identifies the field at its starting offset, not the entire loaded register; inspect the instruction width.\n\n'
    body += '| Start field / byte offset | Destination | Physical PTX |\n| --- | --- | --- |\n'
    for binding in bindings:
        body += '| ' + key_cell(binding['field']) + ' / ' + str(binding['offset']) + ' | ' + key_cell(binding['register']) + ' | ' + str(binding['line']) + ' |\n'
    return body


def control_index(run_input, result, ptx):
    """List source branches and definitions; do not infer their reachability."""
    bindings = parameter_bindings(run_input, ptx)
    if not bindings:
        return ''
    lines = ptx.splitlines()
    first = min(b['line'] for b in bindings)
    end_match = re.search(r'\.(?:entry|func)\b', '\n'.join(lines[first:]))
    last = first + '\n'.join(lines[first:])[:end_match.start()].count('\n') if end_match else len(lines)
    code = {n: line.split('//', 1)[0].strip() for n, line in enumerate(lines, 1) if first <= n <= last}
    labels = {m.group(1): n for n, line in code.items() if (m := re.fullmatch(r'([A-Za-z_$][\w$]*):', line))}
    branches = [(n, m.group(1), m.group(2)) for n, line in code.items()
                if (m := re.match(r'(?:@(!?%\w+)\s+)?bra(?:\.uni)?\s+([A-Za-z_$][\w$]*)\s*;', line))]
    definitions = {}
    for n, line in code.items():
        match = re.match(r'(?:@!?%\w+\s+)?[\w.]+\s+(%\w+)(?:\|%\w+)?,', line)
        if match:
            definitions.setdefault(match.group(1), []).append((n, line))
    def support(predicate, before):
        if predicate is None:
            return 'Unconditional branch; taken whenever its block is reached. Block reachability is separate.'
        pending = [predicate.lstrip('!')]
        seen = set()
        found = []
        for _ in range(4):
            next_pending = []
            for register in pending:
                if register in seen:
                    continue
                seen.add(register)
                candidates = [(n, text) for n, text in definitions.get(register, []) if n < before]
                if candidates:
                    n, text = candidates[-1]
                    found.append(f'{n}: ' + key_cell(text))
                    next_pending.extend(re.findall(r'%\w+', text)[1:])
            pending = next_pending
        return '<br/>'.join(found) or 'no indexed definition'
    body = 'Exact direct-branch index. Listed definitions are nearest earlier definitions, not a reaching-definition proof. Check the configured path separately.\n\n'
    proofs = constant_predicates(run_input, ptx, bindings)
    waits = [(n, line) for n, line in code.items() if re.search(r'cp\.async\.bulk\.wait_group\.read\s+0\s*;', line)]
    for n, line in waits:
        previous = [(ln, text) for ln, text in code.items() if ln < n and text and text not in ('{','}')]
        if not previous or not previous[-1][1].endswith(':'):
            continue
        label = previous[-1][1][:-1]
        body += f'**Final read wait {n}, block ' + key_cell(label) + '.**\n\n'
        entrances = [branch for branch, _, target in branches if target == label and branch < n]
        if entrances:
            body += branch_corridor(code, labels, proofs, min(entrances), n) + '\n'
        body += '| Entrance | Exact source | Earlier definitions |\n| --- | --- | --- |\n'
        for branch, predicate, target in branches:
            if target == label:
                body += f'| Direct jump / {branch} | ' + key_cell(code[branch]) + ' | ' + support(predicate, branch) + ' |\n'
        predecessor = previous[:-1]
        if predecessor:
            at, text = predecessor[-1]
            predicate = next((p for b,p,_ in branches if b == at), None)
            evidence = support(predicate, at) if any(b == at for b,_,_ in branches) else 'Not a direct branch; inspect the preceding block.'
            body += f'| Fallthrough predecessor / {at} | ' + key_cell(text) + ' | ' + evidence + ' |\n'
    locations = [record.get(side, {}).get('location', {}).get('line')
                 for kind in ('races','warnings') for record in result.get(kind, []) if isinstance(record, dict)
                 for side in ('first','second')]
    locations = [n for n in locations if type(n) is int]
    return_labels = set()
    for label, number in labels.items():
        following = [(n, text) for n, text in code.items() if n > number and text and text not in ('{','}')]
        if following and re.match(r'ret\s*;', following[0][1]):
            return_labels.add(label)
    exits = [(n,p,t) for n,p,t in branches if locations and n < min(locations) and t in return_labels]
    if exits:
        body += '\n**Earlier branches to return before the recorded source location.** These may narrow local eligibility; barrier counts alone do not identify participating threads.\n\n'
        body += '| Physical PTX | Exact branch | Earlier definitions |\n| --- | --- | --- |\n'
        for n,predicate,target in exits[-3:]:
            body += f'| {n} | ' + key_cell(code[n]) + ' | ' + support(predicate, n) + ' |\n'
    return body


def branch_corridor(code, labels, proofs, start, end):
    """Follow a bounded local corridor using only indexed constant predicates.

    Entry to start and the indexed definitions' reachability are assumptions,
    not proved here. Stop at an unknown decision rather than guessing it.
    """
    known = {p['predicate']:(p['value'],p['line']) for p in proofs}
    branches = []
    definitions = {}
    seen = set()
    at = start
    while start <= at <= end and at not in seen:
        seen.add(at)
        line = code.get(at, '')
        definition = re.match(r'(?:@!?%\w+\s+)?[\w.]+\s+(%\w+)(?:\|%\w+)?,',line)
        if definition:
            definitions[definition.group(1)] = (at,line)
        branch = re.match(r'(?:@(!?)(%\w+)\s+)?bra(?:\.uni)?\s+([A-Za-z_$][\w$]*)\s*;',line)
        if branch:
            negative,predicate,target = branch.groups()
            folded = known.get(predicate)
            if predicate is not None and (folded is None or folded[1] >= at):
                origin = definitions.get(predicate)
                detail = ('Latest definition **on this corridor**: '+str(origin[0])+': '+key_cell(origin[1])) if origin else 'Value not evaluated by this bounded index.'
                branches.append(f'| {at} | '+key_cell(line)+' | **Unknown; stop here.** '+detail+' |')
                break
            taken = True if predicate is None else (not folded[0] if negative else folded[0])
            branches.append(f'| {at} | '+key_cell(line)+' | '+('Taken' if taken else 'Not taken')+' |')
            if taken:
                target_line=labels.get(target)
                if target_line is None or not start <= target_line <= end:
                    break
                at=target_line
                continue
        at += 1
    if not branches:
        return ''
    return ('**Local branch corridor.** Assuming entry to '+str(start)+' and the indexed constant definitions, this path follows branches until an unknown decision. It is not a full kernel trace. Definitions in bypassed blocks are not used.\n\n'
            '| Physical PTX | Branch | Decision / source |\n| --- | --- | --- |\n'+'\n'.join(branches)+'\n')



def mma_flag_index(run_input, ptx):
    """Index adjacent MMA flag comparisons and all their source definitions.

    This is listing evidence, not control-flow or loop reaching-value analysis.
    """
    from code_excerpt import mask_noncode
    clean = mask_noncode(ptx)
    entry = run_input.get('entry')
    if not isinstance(entry, str):
        return ''
    start = re.search(r'\.entry\s+' + re.escape(entry) + r'(?=\s|\()', clean)
    if not start:
        return ''
    opening = clean.find('{', start.end())
    if opening < 0:
        return ''
    depth, closing = 0, None
    for i in range(opening, len(clean)):
        if clean[i] == '{': depth += 1
        if clean[i] == '}':
            depth -= 1
            if depth == 0:
                closing = i
                break
    if closing is None:
        raise ValueError('Selected PTX entry has an unbalanced body.')
    first = clean[:opening].count('\n') + 1
    last = clean[:closing].count('\n') + 1
    lines = clean.splitlines()
    originals = ptx.splitlines()
    definitions = {}
    for n in range(first, last+1):
        line = lines[n-1].strip()
        m = re.match(r'(?:@!?[\w%]+\s+)?[\w.]+\s+(%\w+)\s*,', line)
        if m:
            definitions.setdefault(m[1], []).append((n, originals[n-1].strip()))
    rows = []
    for n in range(first+1,last+1):
        if not re.search(r'\btcgen05\.mma\.', lines[n-1]):
            continue
        flag = re.search(r'setp\.ne\.b32\s+([\w%]+)\s*,\s*(%\w+|\d+)\s*,\s*0\s*;', lines[n-2])
        if not flag or not re.search(r',\s*' + re.escape(flag[1]) + r'\s*;', lines[n-1]):
            continue
        operand = flag[2]
        support = definitions.get(operand, [])
        shown = '<br/>'.join(str(at) + ': ' + key_cell(text) for at,text in support) if support else key_cell(operand) + ' (literal)' if operand.isdigit() else 'No indexed definition; inspect source.'
        rows.append(f'| {n} | {n-1} / ' + key_cell(operand) + ' | ' + shown + ' |\n')
    if not rows:
        return ''
    return ('\n**MMA accumulate-flag source index.** Comparisons are adjacent to the MMA in the listing. '
            'All indexed assignments to their input are shown, including later loop updates. '
            '**Do not treat the nearest earlier assignment as a loop reaching-value proof.** '
            'Trace initialization versus subsequent iterations.\n\n'
            '| MMA / physical PTX | Flag comparison / input | Input assignments in this entry |\n| --- | --- | --- |\n'
            + ''.join(rows))


def local_copy_guards(run_input, ptx):
    """Evaluate simple local copy guards, never full-path reachability.

    Only unique, unpredicated mov/and/setp definitions over tid.x and unsigned
    32-bit literals are understood. Unknown or reused definitions stay unknown.
    """
    entry = run_input.get('entry')
    if not isinstance(entry, str):
        return []
    clean = re.sub(r'/\*.*?\*/', lambda m: '\n' * m.group().count('\n'), ptx, flags=re.S)
    start = re.search(r'\.entry\s+' + re.escape(entry) + r'(?=\s|\()', clean)
    if not start:
        return []
    opening = clean.find('{', start.end())
    if opening < 0:
        return []
    depth, closing = 0, None
    for index in range(opening, len(clean)):
        depth += (clean[index] == '{') - (clean[index] == '}')
        if depth == 0:
            closing = index
            break
    if closing is None:
        return []
    first = clean[:opening].count('\n') + 1
    last = clean[:closing].count('\n') + 1
    body = [(n, line.split('//', 1)[0].strip())
            for n, line in enumerate(clean.splitlines(), 1) if first <= n <= last]
    definitions, labels = {}, {}
    for n, code in body:
        label = re.match(r'([A-Za-z_$][\w$]*):', code)
        if label:
            labels[label.group(1)] = n
        for definition in re.finditer(r'(?:^|(?<=[;{}]))\s*(@!?%\w+\s+)?([\w.]+)\s+(%\w+)(?:\|%\w+)?,\s*([^;]+);', code):
            predicated, opcode, dest, args = definition.groups()
            definitions.setdefault(dest, []).append((n, bool(predicated), opcode, args))

    def expression(token, before, seen=frozenset()):
        if token == '%tid.x':
            return ('tid.x',), []
        try:
            number = int(token, 0)
            return (('literal', number), []) if 0 <= number < 2**32 else (None, [])
        except ValueError:
            pass
        found = definitions.get(token, [])
        if token in seen or len(found) != 1:
            return None, []
        n, predicated, op, args = found[0]
        if predicated or n >= before:
            return None, []
        operands = [v.strip() for v in args.split(',')]
        allowed = {'mov.u32':1, 'mov.b32':1, 'mov.s32':1,
                   'and.b32':2, 'setp.eq.b32':2, 'setp.ne.b32':2,
                   'setp.eq.u32':2, 'setp.ne.u32':2,
                   'setp.eq.s32':2, 'setp.ne.s32':2}
        if op not in allowed or len(operands) != allowed[op]:
            return None, []
        children, support = [], []
        for operand in operands:
            node, proof = expression(operand, n, seen | {token})
            if node is None:
                return None, []
            children.append(node)
            support += proof
        support += [(n, token, op + ' ' + args)]
        return (op, *children), support

    def evaluate(node, tid):
        op = node[0]
        if op == 'literal': return node[1]
        if op == 'tid.x': return tid
        values = [evaluate(child, tid) for child in node[1:]]
        if op.startswith('mov.'): return values[0]
        if op == 'and.b32': return values[0] & values[1]
        return values[0] == values[1] if '.eq.' in op else values[0] != values[1]

    def describe(node):
        op = node[0]
        if op == 'literal': return str(node[1])
        if op == 'tid.x': return 'tid.x'
        if op.startswith('mov.'): return describe(node[1])
        symbol = '&' if op == 'and.b32' else '==' if '.eq.' in op else '!='
        return '(' + describe(node[1]) + ' ' + symbol + ' ' + describe(node[2]) + ')'

    block = run_input.get('launch', {}).get('block_dim', [])
    limit = block[0] if isinstance(block, list) and block and type(block[0]) is int and 0 < block[0] <= 1024 else None
    answer = []
    previous_branch = None
    for n, code in body:
        if re.match(r'[A-Za-z_$][\w$]*:', code):
            previous_branch = None
        branch = re.fullmatch(r'(?:@(!?)(%\w+)\s+)?bra(?:\.uni)?\s+([A-Za-z_$][\w$]*)\s*;', code)
        if branch:
            previous_branch = (n, *branch.groups())
            continue
        if not re.match(r'cp\.async\.bulk\.(?:tensor\.\dd\.global\.shared|wait_group\.read)', code):
            continue
        if not previous_branch:
            continue
        at, negated, predicate, target = previous_branch
        if predicate is None or labels.get(target, -1) <= n:
            continue
        node, support = expression(predicate, at)
        # A branch operand must resolve to a predicate, not an integer mask.
        if node is not None and not node[0].startswith('setp.'):
            node = None
        admitted = None
        if node is not None and limit is not None:
            admitted = [tid for tid in range(limit) if not (not evaluate(node, tid) if negated else evaluate(node, tid))]
        answer.append({'site':n, 'operation':code.split()[0], 'branch':at,
                       'predicate':predicate, 'negated':bool(negated), 'skip_target':labels[target],
                       'skip_condition':('NOT ' if negated else '') + describe(node) if node is not None else None,
                       'accepted_tid_x':admitted, 'support':support})
    return answer


def local_guard_table(items):
    if not items:
        return ''
    text = '**Isolated copy/wait guards — outer exits NOT applied.** These are literal branch sites and bounded predicate evaluations, assuming their indexed definitions are reached; dominance is not proved. They are not recorded issuers or a complete path. `tid.x` values are x coordinates; do not count them as full thread identities. Combine any required outer condition before stating a narrowed range.\n\n'
    for item in items:
        text += f'- Operation {item["site"]}; skip branch {item["branch"]}: ' + key_cell(('!' if item['negated'] else '') + item['predicate']) + f' → {item["skip_target"]}. '
        text += 'Condition: ' + key_cell(item['skip_condition'] or 'not evaluated') + '.\n'
        values = item['accepted_tid_x']
        if values is not None:
            runs = []
            for value in values:
                if runs and value == runs[-1][1] + 1:
                    runs[-1][1] = value
                else:
                    runs.append([value, value])
            ranges = ', '.join(str(a) if a == b else f'{a}–{b}' for a,b in runs) or 'none'
            text += f'  Passing this guard alone: **{ranges}** ({len(values)} distinct x values).\n'
        for line, register, expression in item['support']:
            text += f'  {line}: ' + key_cell(register + ' ← ' + expression) + '\n'
    return text


def encoded(value):
    if value is MISSING:
        return 'not supplied'
    return json.dumps(value, ensure_ascii=False, separators=(', ', ': '))


def same(first, second):
    return json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def cell(value):
    # HTML code preserves exact data, escapes source markup and table pipes.
    return '<code>' + html.escape(encoded(value)).replace('|', '&#124;') + '</code>'


def key_cell(value):
    return '<code>' + html.escape(str(value)).replace('|', '&#124;') + '</code>'


def result_header(title, result):
    """Result-changing facts, with no inference about unsaved record types."""
    status = result.get('status', 'status not supplied')
    if status == 'frontend_unsupported':
        headline = 'Analysis stopped'
    elif isinstance(result.get('race_count'), int) and result['race_count'] > 0:
        headline = 'Race candidate' if result['race_count'] == 1 else 'Race candidates'
    elif isinstance(result.get('warning_count'), int) and result['warning_count'] > 0:
        headline = 'Warnings in the saved run'
    else:
        headline = 'No races reported' if status == 'race_free' else 'Saved analysis result'
    counts = []
    for key, label in [('race_count', 'races'), ('warning_count', 'warnings'), ('unresolved_count', 'unresolved')]:
        if key in result:
            value = result[key]
            if key == 'unresolved_count' and value == 0:
                continue  # This counter is not a verdict about the mechanism's cause.
            if key == 'race_count' and status == 'race_candidate':
                label = 'candidate pair' if value == 1 else 'candidate pairs'
            elif value == 1 and label.endswith('s'):
                label = label[:-1]
            counts.append((f'{value:,}' if type(value) is int else encoded(value)) + ' ' + label)
    text = RESULT_START + '\n<p align="center">' + html.escape(title) + '</p>\n\n'
    text += '<h1 align="center">' + headline + '</h1>\n\n'
    text += '<p align="center"><strong>' + html.escape(str(status)) + '</strong>'
    if counts:
        text += ' · ' + html.escape(' · '.join(counts))
    text += '</p>\n\n'
    qualifiers = []
    if status == 'frontend_unsupported':
        qualifiers.append('No race verdict available.')
    for kind in ('races', 'warnings'):
        if result.get(kind + '_truncated', result.get(kind[:-1] + '_samples_truncated')) is True and isinstance(result.get(kind), list):
            qualifiers.append(f'{len(result[kind])} {kind} retained; the unsaved records are not available.')
    if result.get('search_complete') is False:
        qualifiers.append('Search is **incomplete**.')
    if result.get('memory_analysis', {}).get('exhaustive') is False:
        qualifiers.append('Coverage is **non-exhaustive**.')
    if result.get('assumption_dependent') is True:
        qualifiers.append('The result is **assumption-dependent**.')
    if qualifiers:
        text += ' '.join(qualifiers) + '\n\n'
    # The rule text is quoted as a saved condition, not inferred counterfactual behavior.
    rules_list = result.get('assumptions', [])
    if isinstance(rules_list, list) and any(isinstance(w, dict) and w.get('code') == 'duplicate-tma-store' for w in result.get('warnings', [])):
        rule = next((r for r in rules_list if isinstance(r, dict) and r.get('key') == 'duplicate_tma_store_benign' and r.get('enabled') is True), None)
        if rule:
            text += '**Classification rule ' + html.escape(str(rule.get('id', ''))) + '.** ' + html.escape(str(rule.get('text', 'Meaning not supplied.'))) + ' The retained messages do not prove equal data.\n\n'
    if status == 'frontend_unsupported' and 'reason' in result:
        text += '**Stopped at:** ' + html.escape(str(result['reason'])) + '\n\n'
    text += RESULT_END + '\n'
    return text


def recorded_relation(result):
    records = result.get('races', []) or result.get('warnings', [])
    if not isinstance(records, list) or not records or not isinstance(records[0], dict):
        return ''
    record = records[0]
    region = record.get('region', {})
    start, size = region.get('offset'), region.get('bytes')
    if type(start) is not int or type(size) is not int or size <= 0:
        return ''
    allocation = str(region.get('allocation', 'unnamed allocation'))
    sides = [record.get(side, {}) for side in ('first', 'second')]
    if not all(isinstance(side, dict) and type(side.get('actor')) is int for side in sides):
        return ''
    # Allocation names stay in prose; only integer identifiers enter TeX.
    body = '**Recorded overlap.** One representative retained pair in ' + key_cell(allocation) + '.\n\n'
    # Group the two saved endpoints without asserting event order or full bounds.
    # Allocation names remain escaped prose; only known modes and integer IDs enter TeX.
    endpoints = []
    for index, side in enumerate(sides):
        mode = side.get('mode')
        access = mode if mode in ('read', 'write', 'readwrite') else ('first access' if index == 0 else 'second access')
        endpoints.append(r'\begin{matrix}\textsf{' + access + ' · actor ' + str(side['actor']) + r'}\end{matrix}')
    body += '```math\n' + r'\begin{gathered}\underbrace{' + r'\qquad\qquad'.join(endpoints) + r'}\\[2pt]\textsf{Shared bytes ' + str(start) + '–' + str(start + size - 1) + ' · ' + str(size) + r' bytes}\end{gathered}' + '\n```\n\n'
    body += 'The brace groups two accesses to the reported intersecting bytes; it does not show execution order. Actor IDs are recorded identities, not thread coordinates; equal footprints do not prove equal values.\n'
    return RELATION_START + '\n' + body + RELATION_END + '\n'


def leaves(value, prefix=''):
    if isinstance(value, dict) and value:
        for key, child in value.items():
            yield from leaves(child, prefix + ('.' if prefix else '') + key)
    else:
        yield prefix, value


def fields(value):
    # Long diagnostics and identities need the page width, not a narrow table cell.
    sections, rows = [], []
    def flush():
        if rows:
            sections.append('\n'.join(['| Saved field | Value |', '| --- | --- |'] + rows))
            rows.clear()
    for key, child in leaves(value):
        if len(encoded(child)) > 80 or len(key) > 28 or len(key) + len(encoded(child)) > 58:
            flush()
            sections.append('**' + key_cell(key or 'Value') + '**\n\n' + cell(child))
        else:
            rows.append('| ' + key_cell(key or 'Value') + ' | ' + cell(child) + ' |')
    flush()
    return '\n\n'.join(sections) + '\n'


def display(value):
    """Readable summary values; exact typed data remains in the reference."""
    return key_cell(value) if isinstance(value, str) else cell(value)


def state(value):
    return {True: '**Yes**', False: '**No**'}.get(value, display(value)) if type(value) is bool else display(value)


def fold(label, body):
    # A child disclosure is a supporting layer, not another peer-level heading.
    # Quote its whole block so GitHub renders a left rule and consistent indent.
    lines = []
    child = False
    for line in body.strip().splitlines():
        if line == '<details>':
            child = True
        lines.append('> ' + line if child else line)
        if line == '</details>':
            child = False
    return '<details>\n<summary>' + label + '</summary>\n\n' + '\n'.join(lines) + '\n\n</details>\n'



def common_fields(items):
    """Only merge keys present and identically valued in every record."""
    if not items:
        return {}
    return {key: value for key, value in items[0].items()
            if all(key in item and same(item[key], value) for item in items)}


def retained(records):
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        return fields({'records': records})
    if not records:
        return 'No retained records (`[]`).\n'
    shared = {k: v for k, v in common_fields(records).items() if k not in ('first', 'second')}
    endpoints, links, extras = [], [], []
    for record in records:
        ids = []
        for side in ('first', 'second'):
            value = record.get(side, MISSING)
            if value is MISSING:
                ids.append('not supplied')
            else:
                matches = [i for i, endpoint in enumerate(endpoints) if same(value, endpoint)]
                if not matches:
                    endpoints.append(value)
                ids.append('E' + str(matches[0] + 1 if matches else len(endpoints)))
        links.append(ids)
        extras.append({k: v for k, v in record.items()
                       if k not in shared and k not in ('first', 'second')})
    labels = ['E' + str(i + 1) for i in range(len(endpoints))]
    actors = [e.get('actor', MISSING) if isinstance(e, dict) else MISSING for e in endpoints]
    # Actor axes are unambiguous only when each actor has one retained access identity.
    actor_axes = all(type(a) is int for a in actors) and len(set(actors)) == len(actors)
    names = dict(zip(labels, map(str, actors))) if actor_axes else dict(zip(labels, labels))
    record_word = 'record' if len(records) == 1 else 'records'
    identity_word = 'identity' if len(endpoints) == 1 else 'identities'
    body = f'**{len(records)} retained {record_word} · {len(endpoints)} distinct access {identity_word}.**\n\n'
    region = shared.get('region', {})
    if isinstance(region, dict) and type(region.get('offset')) is int and type(region.get('bytes')) is int and region['bytes'] > 0:
        body += 'Every retained record reports the same intersection: ' + display(region.get('allocation', MISSING))
        body += f", **bytes {region['offset']}–{region['offset'] + region['bytes'] - 1}** ({region['bytes']} bytes).\n\n"
    body += 'Cells contain record numbers, starting at 1. Axes identify the saved first and second endpoints, not execution order.\n\n'
    matrix = 6 <= len(links) and 2 <= len(endpoints) <= 8 and not any(extras) and all('not supplied' not in pair for pair in links)
    if matrix:
        rows = [e for e in labels if any(pair[0] == e for pair in links)]
        columns = [e for e in labels if any(pair[1] == e for pair in links)]
        axis = 'First actor ↓ / second →' if actor_axes else 'First access ↓ / second →'
        body += '| ' + axis + ' | ' + ' | '.join(names[e] for e in columns) + ' |\n'
        body += '| --- | ' + ' | '.join([':---:'] * len(columns)) + ' |\n'
        for first in rows:
            cells = [', '.join(str(i + 1) for i, pair in enumerate(links) if pair == [first, second]) for second in columns]
            body += '| **' + names[first] + '** | ' + ' | '.join(cells) + ' |\n'
        body += '\nBlank cells mean **no retained record**, not proven absence. '
        body += 'Actor IDs are saved identities, not CUDA thread coordinates.\n' if actor_axes else 'E labels distinguish saved accesses, including different accesses by the same actor.\n'
    else:
        body += '| Record | First access | Second access |\n| --- | --- | --- |\n'
        for index, (first, second) in enumerate(links, 1):
            body += f'| {index} | {first} | {second} |\n'
        body += '\nE labels identify saved endpoint objects, not threads. Missing pairs are not proven absent.\n'
    if endpoints:
        endpoint_common = common_fields(endpoints) if all(isinstance(e, dict) for e in endpoints) else {}
        varying = [{k: v for k, v in e.items() if k not in endpoint_common}
                   if isinstance(e, dict) else {'value': e} for e in endpoints]
        lookup = 'Use this lookup to distinguish the accesses behind the pairs. Event and instruction IDs are different coordinate systems.\n\n'
        if all(set(e) <= {'actor', 'event', 'instruction'} for e in varying):
            lookup += '| Access | Actor | Event | Instruction |\n| --- | --- | --- | --- |\n'
            for index, e in enumerate(varying, 1):
                lookup += f'| E{index} | {cell(e.get("actor", MISSING))} | {cell(e.get("event", MISSING))} | {cell(e.get("instruction", MISSING))} |\n'
        else:
            for index, e in enumerate(varying, 1):
                lookup += '\n' + fold('Exact access E' + str(index), fields(e))
        if endpoint_common:
            lookup += '\n**Shared by all accesses:**\n\n' + fields(endpoint_common)
        body += '\n' + fold('Access identities · events, instructions and shared location', lookup)
    if shared:
        body += '\n' + fold('Shared record fields · saved message and intersection', fields(shared))
    for index, extra in enumerate(extras, 1):
        if extra:
            body += '\n' + fold('Exact record ' + str(index) + ' · distinct fields', fields(extra))
    return body


RULE_NAMES = {
    'warp_tmem_load_completion': 'TMEM read completion',
    'duplicate_tma_store_benign': 'Duplicate-store classification',
    'immediate_visibility': 'Visibility of executed effects',
    'representative_election': 'Representative election',
    'clc_always_fails': 'Cluster cancellation',
}


def rules(result, run_input):
    body = '**Enabled** is a configuration choice. **Triggered** records whether the rule applied in this run; it does not attribute a particular finding to that rule. `null` means unknown.\n'
    assumptions = result.get('assumptions', MISSING)
    metadata = ''
    if isinstance(assumptions, list):
        for rule in assumptions:
            if not isinstance(rule, dict):
                metadata += '\n' + fields({'rule': rule})
                continue
            label = RULE_NAMES.get(rule.get('key'), rule.get('key', 'Unnamed rule'))
            identity = html.escape(str(rule.get('id', 'ID not supplied')))
            body += '\n**' + identity + ' · ' + html.escape(label) + '**  \n'
            body += 'Enabled ' + cell(rule.get('enabled', MISSING)) + ' · Triggered ' + cell(rule.get('triggered', MISSING)) + '\n\n'
            body += html.escape(str(rule.get('text', 'Meaning not supplied.'))) + '\n'
            metadata += '\n**' + identity + '**\n\n' + fields({k: v for k, v in rule.items() if k not in ('text', 'enabled', 'triggered', 'id')})
    else:
        metadata += '\n' + fields({'assumptions': assumptions})
    sources = [('input.assumptions', run_input.get('assumptions', MISSING)),
               ('result.semantic_assumptions', result.get('semantic_assumptions', MISSING))]
    if all(v is not MISSING for _, v in sources) and same(sources[0][1], sources[1][1]):
        metadata += '\nIdentical in `input.assumptions` and `result.semantic_assumptions`:\n\n' + fields(sources[0][1])
    else:
        for name, value in sources:
            metadata += '\n**' + name + '**\n\n' + fields({name: value})
    for key in ('assumption_effects', 'assumption_reporting'):
        if key in result:
            metadata += '\n' + fields({key: result[key]})
    body += '\n' + fold('Exact rule settings · keys, input agreement and usage', metadata)
    contracts = [(name, value) for name, value in (
        ('result.frontend_assumptions', result.get('frontend_assumptions', MISSING)),
        ('result.frontend.frontend_assumptions', result.get('frontend', {}).get('frontend_assumptions', MISSING))) if value is not MISSING]
    contract_body = 'Structured pointer contracts and standalone flags are separate saved fields. Neither is silently substituted for the other.\n\n'
    if not contracts:
        contract_body += 'Frontend contracts not supplied.\n'
    elif len(contracts) == 2 and same(contracts[0][1], contracts[1][1]):
        contract_body += 'Identical copies in `' + contracts[0][0] + '` and `' + contracts[1][0] + '`, shown once:\n\n' + fields(contracts[0][1])
    else:
        for name, value in contracts:
            contract_body += '\n**' + name + '**\n\n' + fields(value)
    body += '\n' + fold('Exact frontend contracts · pointer scope, election and assertions', contract_body)
    return body


def run_scope(run_input, result):
    body = '**Analysis completion, search completion and exhaustive coverage are separate questions.**\n\n'
    analysis = result.get('memory_analysis', {})
    body += '| Memory analysis complete? | Search complete? | Exhaustive? |\n| --- | --- | --- |\n'
    body += '| ' + ' | '.join(state(v) for v in (result.get('memory_analysis_complete', MISSING), result.get('search_complete', MISSING), analysis.get('exhaustive', MISSING))) + ' |\n'
    result_keys = ('schema', 'mode', 'status', 'reason', 'race_count', 'warning_count', 'unresolved_count', 'race_samples_truncated', 'warning_samples_truncated', 'memory_analysis_complete', 'search_complete', 'assumption_dependent', 'source_violations', 'verification_limitations')
    exact = fields({k: result[k] for k in result_keys if k in result})
    exact += '\nRetained: ' + ', '.join(f"{len(result[k])} {k[:-1] if len(result[k]) == 1 else k}" for k in ('races', 'warnings') if isinstance(result.get(k), list)) + '.\n'
    analysis_keys = ('exhaustive', 'strategy', 'candidate_generation_fallback', 'address_candidates', 'overlap_queries', 'unordered_candidates_examined')
    exact += '\n' + fields({'memory_analysis': {k: analysis[k] for k in analysis_keys if k in analysis}})
    body += '\n' + fold('Exact result and search fields', exact)
    launch = run_input.get('launch', MISSING)
    body += '\n**Launch** · tuples use `(x, y, z)`.\n\n'
    if isinstance(launch, dict):
        body += '| Grid · blocks | Block · threads | Cluster · blocks |\n| --- | --- | --- |\n'
        body += '| ' + ' | '.join(cell(launch.get(k, MISSING)) for k in ('grid_dim', 'block_dim', 'cluster_dim')) + ' |\n'
        body += '\nDynamic shared memory: ' + cell(launch.get('dynamic_shared_bytes', MISSING)) + ' bytes.\n'
        other_launch = {k:v for k,v in launch.items() if k not in ('grid_dim', 'block_dim', 'cluster_dim', 'dynamic_shared_bytes')}
        if other_launch:
            body += '\n' + fold('Other launch fields', fields(other_launch))
    else:
        body += fields({'launch': launch})
    arguments = run_input.get('arguments', MISSING)
    declarations = run_input.get('declarations', [])
    regular = isinstance(arguments, list) and all(isinstance(a, dict) and set(a) == {'field', 'value'} for a in arguments)
    unique = regular and len({a['field'] for a in arguments}) == len(arguments)
    values = {a['field']: a['value'] for a in arguments} if unique else {}
    if any(name in values for name in ('alpha', 'beta')):
        body += '\n**Scalar inputs** · values, pointer overrides and strides are distinct.\n\n'
        body += '| Scalar | Value | Pointer | Stride |\n| --- | --- | --- | --- |\n'
        for name in ('alpha', 'beta'):
            if name in values:
                body += '| ' + name + ' | ' + ' | '.join(cell(values.get(name + suffix, MISSING)) for suffix in ('', '.pointer', '.stride')) + ' |\n'
    exact = 'Values and parameter offsets are separate. The authored source explanation establishes which loads reach a use.\n\n'
    exact += fields(values) if unique else fields({'arguments': arguments})
    if declarations:
        if isinstance(declarations, list) and all(isinstance(d, dict) and set(d) == {'name', 'parameter', 'offset', 'type'} for d in declarations):
            lookup = '| Field | Parameter / byte offset | Type |\n| --- | --- | --- |\n'
            for d in declarations:
                lookup += '| ' + key_cell(d['name']) + ' | ' + cell(d['parameter']) + ' / ' + cell(d['offset']) + ' | ' + display(d['type']) + ' |\n'
        else:
            lookup = fields({'declarations': declarations})
        exact += '\n' + fold('Exact parameter declarations · byte offsets and types', lookup)
    body += '\n' + fold('Exact parameters · dimensions, scalar values and scheduler settings', exact)
    descriptors = run_input.get('tma_descriptors', [])
    if isinstance(descriptors, list):
        unique_descriptors = []
        for descriptor in descriptors:
            if not isinstance(descriptor, dict):
                unique_descriptors.append((descriptor, []))
                continue
            shape = {k:v for k,v in descriptor.items() if k != 'offset'}
            for existing, offsets in unique_descriptors:
                if same(existing, shape):
                    offsets.append(descriptor.get('offset', MISSING))
                    break
            else:
                unique_descriptors.append((shape, [descriptor.get('offset', MISSING)]))
        if unique_descriptors:
            body += '\n**Transfer geometry** · both shapes count elements in descriptor-axis order. The reported byte intersection is a separate quantity.\n\n'
            body += '| Object | Global shape | Copy box |\n| --- | --- | --- |\n'
            exact = ''
            for descriptor, offsets in unique_descriptors:
                data = dict(descriptor) if isinstance(descriptor, dict) else {'descriptor': descriptor}
                if isinstance(descriptor, dict):
                    body += '| ' + ' | '.join(display(descriptor.get(k, MISSING)) for k in ('object', 'global_shape', 'box_shape')) + ' |\n'
                if offsets:
                    data['offsets'] = [x if x is not MISSING else 'not supplied' for x in offsets]
                exact += '\n**' + html.escape(str(data.get('object', 'Unnamed descriptor'))) + '**\n\n' + fields(data)
            body += '\n' + fold('Exact descriptors · byte strides, storage and parameter offsets', exact)
    else:
        body += fields({'tma_descriptors': descriptors})
    covered = {'schema', 'entry', 'launch', 'arguments', 'declarations', 'assumptions', 'tma_descriptors'}
    other = {k:v for k,v in run_input.items() if k not in covered}
    if other:
        body += '\n' + fold('Other configured inputs', fields(other))
    return body


def render(run_input, result, artifacts):
    body = '**Pairs** locate the finding; **rules** explain the model; **run setup** fixes its scope. Source lookups and artifact identities provide exact support.\n\n'
    findings = ''
    for kind in ('races', 'warnings'):
        if kind in result:
            if result[kind] == []:
                findings += '\n**' + kind.capitalize() + ':** no retained records (`[]`).\n'
            else:
                findings += '\n### ' + kind.capitalize() + '\n\n' + retained(result[kind])
        else:
            findings += '\n' + kind.capitalize() + ' not supplied.\n'
    body += fold('Retained records · effects, identities and pairs', findings)
    body += '\n' + fold('Model rules · what was assumed', rules(result, run_input))
    body += '\n' + fold('Run setup and coverage · what was checked', run_scope(run_input, result))
    bindings = parameter_bindings(run_input, artifacts[-1].read_text())
    if bindings:
        lookup = fold('Exact parameter loads · source names and physical lines', binding_table(bindings))
        proofs = constant_predicates(run_input, artifacts[-1].read_text(), bindings)
        if proofs:
            lookup += '\n' + fold('Exact scalar predicate simplifications · configured values', predicate_table(proofs))
            skipped = skipped_parameter_loads(run_input, artifacts[-1].read_text(), bindings, proofs)
            if skipped:
                lookup += '\n' + fold('Exact skipped parameter loads · configured branches', skipped_table(skipped))
        body += '\n' + fold('Source lookup · parameter loads and configured branches', lookup)
    identity = 'These SHA-256 hashes identify the supplied files, not an executing verifier revision.\n\n'
    identity += '**Input schema:** ' + cell(run_input.get('schema', MISSING)) + '\n\n'
    identity += '**Entry:**\n\n' + cell(run_input.get('entry', MISSING)) + '\n'
    if 'implementation' in result:
        identity += '\n**Saved implementation:** ' + display(result['implementation']) + '\n'
    for path in artifacts:
        identity += '\n**' + html.escape(path.name) + '**\n\n' + cell(hashlib.sha256(path.read_bytes()).hexdigest()) + '\n'
    body += '\n' + fold('Provenance · supplied artifact identity', identity)
    return START + '\n' + fold('<b>Complete saved evidence</b> · records, model and run', body) + END + '\n'


def verify_identity(identity, artifacts):
    """Validate the optional supplied manifest before writing a report."""
    if identity is None:
        return
    manifest = json.loads(identity.read_text())
    for name, item in manifest.get('files', {}).items():
        path = (identity.parent / name).resolve()
        if not path.is_relative_to(identity.parent.resolve()):
            raise ValueError('Identity path escapes its directory: ' + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != item.get('sha256'):
            raise ValueError('Identity mismatch: ' + name)
        supplied = [p for p in artifacts if p.name == path.name]
        for actual in supplied:
            if hashlib.sha256(actual.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('Supplied artifact differs from identity: ' + name)


def main():
    parser = argparse.ArgumentParser(description='Compose one report from saved results, optionally with a source explanation.')
    for key in ('input', 'result', 'ptx'):
        parser.add_argument('--' + key, required=True, type=Path)
    parser.add_argument('--report', type=Path, help='Write one report.md; otherwise print to stdout.')
    parser.add_argument('--draft', type=Path, help='Optional version-2 source explanation, outside the output directory.')
    parser.add_argument('--identity', type=Path)
    parser.add_argument('--source', type=Path, action='append', default=[], help='Optional source file available to evidence references by basename.')
    parser.add_argument('--equivalence', type=Path, help='Link an existing numerical-check artifact from the same run; does not replace the verifier result.')
    parser.add_argument('--artifact-base', help='HTTPS repository blob URL, preferably at a fixed commit; artifact paths are relative to the current directory.')
    parser.add_argument('--bindings', action='store_true', help='Print source-analysis aids, not verifier findings.')
    args = parser.parse_args()
    try:
        artifacts = [args.input, args.result, args.ptx]
        verify_identity(args.identity, artifacts)
        run_input, result = json.loads(args.input.read_text()), json.loads(args.result.read_text())
        ptx = args.ptx.read_text()
        from report_layout import compose, validate_inputs, MARKER
        validate_inputs(run_input, result, ptx)
        if args.bindings:
            if args.report or args.draft:
                parser.error('--bindings is an inspection command; omit --report and --draft.')
            bindings = parameter_bindings(run_input, ptx)
            proofs = constant_predicates(run_input, ptx, bindings)
            print('SOURCE ANALYSIS AIDS: local deductions, not verifier findings or reachability proofs.')
            print(binding_table(bindings))
            print(predicate_table(proofs))
            print(skipped_table(skipped_parameter_loads(run_input, ptx, bindings, proofs)))
            print(control_index(run_input, result, ptx))
            print(local_guard_table(local_copy_guards(run_input, ptx)))
            print(mma_flag_index(run_input, ptx))
            print(instruction_notes(ptx))
            return
        if len({p.name for p in args.source}) != len(args.source):
            raise ValueError('Optional source basenames must be distinct.')
        sources = {p.name: p.read_text() for p in args.source}
        draft = json.loads(args.draft.read_text()) if args.draft else None
        if args.equivalence and not args.equivalence.is_file():
            raise ValueError('Numerical-check artifact does not exist: ' + str(args.equivalence))
        composed = compose(draft, run_input, result, ptx, artifacts, sources,
                           report_path=args.report, source_paths=args.source, equivalence_path=args.equivalence, artifact_base=args.artifact_base)
        if args.report is None:
            print(composed, end='')
            return
        if args.report.name != 'report.md':
            raise ValueError('The output must be named report.md.')
        if args.draft and args.draft.resolve().is_relative_to(args.report.parent.resolve()):
            raise ValueError('Keep draft.json outside the report delivery directory.')
        if args.report.exists() and not args.report.read_text().startswith((MARKER, '<!-- report-layout:v1 -->')):
            raise ValueError('Refusing to replace an unrelated report. Choose an unused output directory.')
        # Compose and validate completely before touching the previous good report.
        import os
        import tempfile
        args.report.parent.mkdir(parents=True, exist_ok=True)
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=args.report.parent, delete=False) as stream:
                tmp = Path(stream.name)
                stream.write(composed)
            os.replace(tmp, args.report)
        finally:
            if tmp is not None and tmp.exists():
                tmp.unlink()
        print('COMPOSED: ' + str(args.report))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
