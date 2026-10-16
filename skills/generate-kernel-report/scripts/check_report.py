#!/usr/bin/env python3
"""Read-only report checks. Prints diagnostics; never writes a sidecar file."""
import argparse
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys


class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == 'details':
            self.depth += 1

    def handle_endtag(self, tag):
        if tag == 'details':
            self.depth -= 1

    def handle_data(self, data):
        if self.depth == 0:
            self.parts.append(data)


def plain(value):
    return html.unescape(re.sub(r'<[^>]*>', '', value))


def compact(value):
    return re.sub(r'\s+', '', value)


def math_disclosure_checks(source):
    """Guard the tested native-rendering budget, including nested references."""
    masked = re.sub(r'^(`{3,}|~{3,})([^\n]*)\n.*?^\1[ \t]*(?:\n|$)',
                    lambda match: match.group() if match.group(2).strip() == 'math'
                    else ' ' * len(match.group()), source, flags=re.S | re.M)
    stack = []
    issues = []
    for tag in re.finditer(r'<details\b[^>]*>|</details\s*>', masked, re.I):
        if tag.group().lower().startswith('</'):
            if not stack:
                continue  # Disclosure balance is checked separately.
            start = stack.pop()
            block = masked[start:tag.end()]
            if re.search(r'```math\b', block) and len(source[start:tag.end()].encode('utf-8')) >= 4000:
                summary = re.search(r'<summary\b[^>]*>(.*?)</summary>', block, re.S | re.I)
                label = plain(summary.group(1)).strip() if summary else 'Disclosure'
                issues.append(f'{label}: math-bearing disclosure exceeds the tested 4,000-byte authoring budget, including nested content. Keep its explanation short and move extensive exact support to an adjacent reference; preserve evidence. Inspect native rendering.')
        else:
            stack.append(tag.start())
    return issues


class ReadingScopes(HTMLParser):
    """Measure each independently visible layer without its nested references."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.scopes = [{'label': 'Closed report', 'parts': [], 'blocks': [], 'relation': False, 'code_before_relation': False}]
        self.stack = [self.scopes[0]]
        self.in_summary = False
        self.in_pre = False
        self.block = []

    def handle_starttag(self, tag, attrs):
        if tag == 'details':
            scope = {'label': '', 'parts': [], 'blocks': [], 'relation': False, 'code_before_relation': False}
            self.scopes.append(scope)
            self.stack.append(scope)
        elif tag == 'summary':
            self.in_summary = True
        elif tag == 'pre':
            self.in_pre = True
            self.block = []
            if not self.stack[-1]['relation']:
                self.stack[-1]['code_before_relation'] = True
        elif tag in ('reading-relation', 'table'):
            self.stack[-1]['relation'] = True

    def handle_endtag(self, tag):
        if tag == 'details' and len(self.stack) > 1:
            self.stack.pop()
        elif tag == 'summary':
            self.in_summary = False
        elif tag == 'pre':
            self.stack[-1]['blocks'].append(''.join(self.block))
            self.in_pre = False

    def handle_data(self, data):
        if self.in_pre:
            self.block.append(data)
        elif self.in_summary:
            self.stack[-1]['label'] += data
        else:
            self.stack[-1]['parts'].append(data)
            if re.search(r'^\s*\|?\s*:?-{3,}:?\s*\|', data, re.M):
                self.stack[-1]['relation'] = True


def reading_checks(source):
    # Count the authored account, not the immutable result/evidence supplied by
    # the compositor. Counting both encouraged deleting the computation context
    # or moving the overlap before it merely to satisfy the word budget.
    author_source = source
    for marker in ('saved-result', 'recorded-relation', 'wait-guarantee'):
        author_source = re.sub(r'<!-- ' + marker + r':start -->.*?<!-- ' + marker + r':end -->', '', author_source, flags=re.S)
    duplicate_header = re.sub(r'^(`{3,}|~{3,})[^\n]*\n.*?^\1[ \t]*(?:\n|$)', '', author_source, flags=re.S | re.M)
    header_issues = []
    if '<!-- saved-result:start -->' in source and re.search(r"<h1\b|^#\s|align\s*=\s*[\"']?center\b", duplicate_header, re.I | re.M):
        header_issues.append('Duplicate result presentation: retain the helper-owned centered header only; start the authored account with the computation.')
    # Formula and diagram labels are assessed by rendering, not prose limits.
    prose_source = re.sub(r'```(?:math|mermaid)\b[^\n]*\n.*?```', '<reading-relation/>', source, flags=re.S)
    prose_source = re.sub(r'```[^\n]*\n.*?```', '', prose_source, flags=re.S)
    tree = ReadingScopes()
    tree.feed(prose_source)
    issues = header_issues + math_disclosure_checks(source)
    for index, scope in enumerate(tree.scopes):
        reference = index > 0 and re.match(r'\s*(?:reference|exact|raw|trace|retained records|provenance)\b', scope['label'], re.I)
        if reference:
            continue
        if index > 0 and scope['code_before_relation']:
            issues.append(f'{scope["label"]}: instruction evidence precedes a relationship view. Show a compact formula, diagram or comparison before code; put bare derivations in a named reference.')
        words = len(' '.join(scope['parts']).split())
        if index == 0 and author_source != source:
            account = ReadingScopes()
            account.feed(re.sub(r'```[^\n]*\n.*?```', '', author_source, flags=re.S))
            words = len(' '.join(account.scopes[0]['parts']).split())
        if words > 300:
            issues.append(f'{scope["label"]}: {words} words before subordinate references. Recompose repeated scope and move derivations, preserving the explanatory bridge.')
        displayed = sum(sum(bool(line.strip()) and line.strip() not in ('...', '…') for line in block.splitlines()) for block in scope['blocks'])
        if displayed > 8:
            issues.append(f'{scope["label"]}: {displayed} displayed code lines. Keep the decisive excerpt here and place the full listing in a named reference fold.')
        for block in scope['blocks']:
            lines = [line for line in block.splitlines() if line.strip() and line.strip() not in ('...', '…')]
            for line in lines:
                if len(line) > 84 and len(line.split()) > 1:
                    issues.append(f'{scope["label"]}: long code line; use code_excerpt.py operand wrapping.')
                    break
    prefix = re.split(r'```(?:math|mermaid)\b', author_source, maxsplit=1)[0]
    before = ReadingScopes()
    before.feed(re.sub(r'```[^\n]*\n.*?```', '', prefix, flags=re.S))
    if re.search(r'```(?:math|mermaid)\b', source) and len(' '.join(before.scopes[0]['parts']).split()) > 140:
        issues.append('The first relationship comes after more than 140 prose words. Introduce the object briefly, then show the relation; fold the full algorithm and raw counters.')
    table_source = re.sub(r'```[^\n]*\n.*?```', '', source, flags=re.S)
    for gap in re.finditer(r'^\|[^\n]*\|[ \t]*\n(?:[ \t]*\n)+(?=\|[^\n]*\|)', table_source, flags=re.M):
        following = table_source[gap.end():].splitlines()
        separator = following[1].strip().strip('|').split('|') if len(following) > 1 else []
        if separator and all(re.fullmatch(r'\s*:?-{3,}:?\s*', cell) for cell in separator):
            continue  # A separately headed table may follow another table.
        issues.append('Blank lines split Markdown table rows into pipe-text paragraphs. Keep each table header, separator and body contiguous; preserve every record.')
        break
    for table in re.findall(r'^\|[^\n]*\|[ \t]*\n(?:\|[^\n]*\|[ \t]*(?:\n|$))+', table_source, flags=re.M):
        columns = len(table.splitlines()[0].strip().strip('|').split('|'))
        key = re.search(r'\b[A-Za-z_][A-Za-z0-9_.]{23,}\b', table)
        if columns >= 4 and key:
            issues.append('Wide table contains an indivisible identifier (' + key.group() + '). Native GitHub clipped these layouts; use short semantic labels and preserve exact keys in an adjacent mapping or two-column reference.')
    return issues


def check(report, ptx, result, identity=None):
    # Markdown blockquote prefixes are structure, not literal PTX or table text.
    source = re.sub(r'^(?:> ?)+', '', report.read_text(), flags=re.M)
    raw = json.loads(result.read_text())
    tree = VisibleText()
    tree.feed(source)
    visible = '\n'.join(tree.parts)
    all_text = plain(source)
    issues = reading_checks(source)
    for formula in re.findall(r'```math\b[^\n]*\n(.*?)```', source, flags=re.S):
        if re.search(r'\\operatorname\b', formula):
            issues.append('GitHub rejected operatorname in native rendering. Use a short text label in this formula, including inside disclosures.')
    status = raw.get('status')
    if status and status not in visible:
        issues.append('Keep the exact saved status in the visible result area.')
    if raw.get('memory_analysis', {}).get('exhaustive') is False:
        if not re.search(r'non[ -]exhaustive|not exhaustive|exhaustive\s*[:=]\s*false', visible, re.I):
            issues.append('Non-exhaustive coverage is hidden or absent; state it beside the result.')
    if raw.get('assumption_dependent') is True:
        if not re.search(r'assumption|conditional|conditioned', visible, re.I):
            issues.append('Keep assumption dependence visible beside the recorded result.')
    if status == 'frontend_unsupported' and not re.search(r'no race verdict|race verdict.*unavailable|no.*race.*conclusion', visible, re.I):
        issues.append('State visibly that this frontend stop has no race verdict.')
    if 'warnings' in raw and re.search(r'no (?:race or )?warning fields|no race or warning fields', all_text, re.I):
        issues.append('warnings is present in the source; do not describe the field as absent.')
    assumptions = raw.get('assumptions', [])
    if isinstance(assumptions, list) and any(isinstance(a, dict) and a.get('triggered') is None for a in assumptions):
        if re.search(r'\b(?:model|semantic) rules\b[^.\n]{0,80}\bdid not cause\b', all_text, re.I):
            issues.append('Unknown rule applicability cannot establish that model rules did not cause or affect the stop. State the recorded stop reason and unknown relevance separately.')
    findings = [record for kind in ('races', 'warnings') for record in raw.get(kind, []) if isinstance(record, dict)]
    for key in ('access_index', 'effect_index'):
        present = any(key in record.get(side, {}) for record in findings for side in ('first', 'second'))
        readable_key = key.replace('_', r'[ _-]')
        grouped_label = re.search(r'access\s*/\s*effect\s+indices', all_text, re.I)
        if present and not (re.search(readable_key, all_text, re.I) or grouped_label):
            issues.append(f'Retained access records include {key}; preserve its exact values in the reference, rather than silently dropping it.')
    claim_source = re.sub(r'<!-- (saved-result|saved-evidence|recorded-relation|wait-guarantee):start -->.*?<!-- \1:end -->', '', source, flags=re.S)
    claim_source = re.sub(r'<pre>.*?</pre>|```[^\n]*\n.*?```', '', claim_source, flags=re.S)
    claim_text = plain(claim_source)
    for sentence in re.split(r'(?<=[.!?])\s+|\n', claim_text):
        quantified = re.search(r'\b(?:all|every|each|only|\d+)\b|whole[- ]warp', sentence, re.I)
        participants = re.search(r'\b(?:lanes?|threads?|warps?)\b', sentence, re.I)
        execution = re.search(r'\b(?:issues?|issued|executes?|executed|runs?)\b', sentence, re.I)
        bounded = re.search(r'\bmay\b|\bcan\b|\bcould\b|\bwould\b|eligible|eligibility|if reached|if they reach|not establish|does not prove|not proven|not known|unknown|not observed', sentence, re.I)
        if quantified and participants and execution and not bounded:
            issues.append('A quantified lane/thread execution claim exceeds local guard evidence: ' + sentence[:170] + '. Describe permission under the evaluated conditions; saved actor IDs alone do not prove CUDA issuer coordinates.')
        if re.search(r'\b(?:retained|recorded|warned)\b.{0,100}\b(?:first|last)\b.{0,40}\b(?:sites?|occurrences?|passes?)\b', sentence, re.I) and not re.search(r'not uniquely|not a unique|no unique|representative|does not (?:bind|assign)|do not (?:bind|assign)', sentence, re.I):
            issues.append('Saved location is being promoted to an unrolled occurrence: ' + sentence[:170] + '. Report the coordinate and distinguish representative source from dynamic-event binding.')
    duplicate_rule = any(isinstance(rule, dict) and rule.get('key') == 'duplicate_tma_store_benign'
                         and rule.get('enabled') is True for rule in assumptions) if isinstance(assumptions, list) else False
    if duplicate_rule and re.search(r'\b(?:assumed? (?:this|equal|identical)|assum(?:ption|es?|ed).{0,45}(?:value|data) equality)\b', claim_text, re.I):
        issues.append('The duplicate-store rule classifies matching footprints/site visits; it does not assume copied-value equality. State the unproved equality separately from the classification premise.')
    if 'cp.async.bulk.tensor' in ptx.read_text() and re.search(r'\b(?:row|column)[- ]coordinates?\b', claim_text, re.I):
        issues.append('Describe tensor-store coordinates by descriptor-axis position, not logical row/column. Their mapping needs separate source support; the report should not infer it from tuple order.')
    has_tmem_copy = 'tcgen05.cp' in ptx.read_text()
    for sentence in re.split(r'(?<=[.!?])\s+|\n', claim_text):
        raw_offset = re.search(r'%r\d+|base\s*\+\s*\d+', sentence, re.I)
        byte_relation = re.search(r'\binside\b|\bwithin\b|\blies? in\b', sentence, re.I)
        access_range = re.search(r'\b(?:bytes?|copy|cp|write|written|footprint|range)\b', sentence, re.I)
        denied = re.search(r'not (?:prove|show|establish|mapped)|no (?:supplied )?mapping|cannot (?:prove|map|infer)', sentence, re.I)
        if has_tmem_copy and raw_offset and byte_relation and access_range and not denied:
            issues.append('A tensor-memory register offset is being used to infer a byte footprint: ' + sentence[:180] + '. Without a supplied address/effect mapping, the saved pair alone establishes byte intersection.')
        if re.search(r'\bread[ ._-]*1\b.{0,55}\b(?:allows?|permits?|lets?)\b.{0,35}\bearlier group\b', sentence, re.I):
            issues.append('read 1 may leave the newest group pending; older groups source reads must be finished. Use the renderer-owned guarantee instead of another interpretation.')
    if re.search(r"\bI (?:did not|haven't|have not|didn't)\b|\bnot resolved here\b|\bnot traced\b|\buntraced\b", claim_text, re.I):
        issues.append('Authoring-process uncertainty is not an evidence gap. Inspect the indexed definitions and state a supported configured relation or the specific unavailable input. Do not leave supplied scalar or accumulator initialization unexamined.')
    if re.search(r'\bfirst[- ]site warnings\b', claim_text, re.I):
        issues.append('Do not assign retained warnings to a first dynamic occurrence. State their saved source coordinate and the representative physical site separately.')
    pairs = {frozenset((str(record.get('first',{}).get('actor')),str(record.get('second',{}).get('actor'))))
             for record in findings if 'first' in record and 'second' in record}
    for sentence in re.split(r'(?<=[.!?])\s+|\n', claim_text):
        pair_claim = re.search(r'\bbetween\s+actors?\s+(\d+)\s+and\s+(\d+)\b',sentence,re.I)
        if pair_claim and re.search(r'\ball\b.{0,50}\bretained\b',sentence,re.I):
            claimed = frozenset(pair_claim.groups())
            if pairs and pairs != {claimed}:
                issues.append('The retained set contains other actor pairs. Identify this as one representative pair; the complete saved-evidence matrix supplies all records.')
    retained_count = len(raw.get('warnings',[])) if isinstance(raw.get('warnings'),list) else 0
    total_warnings = raw.get('warning_count')
    if isinstance(total_warnings,int) and total_warnings > retained_count:
        digits = str(total_warnings)
        normalized_claims = re.sub(r'(?<=\d)[,_](?=\d)', '', claim_text)
        if re.search(r'\b' + digits + r'\s+(?:duplicate[- ](?:tma[- ])?store|same[- ]warp|output[- ]copy)', normalized_claims, re.I):
            issues.append('The unsaved warnings have no recorded type distribution. Attach the mechanism to retained warnings; the total count belongs to all warnings.')
    for key in ('race_count', 'warning_count', 'unresolved_count'):
        count = raw.get(key)
        if isinstance(count, int) and count > 0:
            if str(count) not in re.sub(r'(?<=\d)[,_](?=\d)', '', visible):
                issues.append(f'Keep the recorded {key}={count} visible.')
    if source.count('<details>') != source.count('</details>') or tree.depth:
        issues.append('Disclosure tags are not balanced.')
    if re.search(r'!\[[^\]]*\]\(|<img\b', source, re.I):
        issues.append('Replace image dependencies with inline Markdown visuals.')
    code_blocks = re.findall(r'<pre>\s*<code[^>]*>(.*?)</code>\s*</pre>', source, re.S)
    code_blocks += re.findall(r'```ptx\s*\n(.*?)```', source, re.S)
    authored = re.sub(r'<!-- saved-evidence:start -->.*?<!-- saved-evidence:end -->', '', source, flags=re.S)
    if code_blocks and 'cp.async.bulk' in plain(authored):
        summaries = re.findall(r'<summary\b[^>]*>(.*?)</summary>', authored, re.S | re.I)
        if not any(re.match(r'\s*(?:reference|exact)\b', plain(label), re.I)
                   and re.search(r'\b(?:source|PTX|derivation)\b', plain(label), re.I)
                   for label in summaries):
            issues.append('Async instruction account needs an authored exact-source reference, distinct from saved facts. Preserve computation/scalar origins, local and outer guards, repeated sites and all configured exit branches; move supporting code here rather than deleting derivations during length repair.')
    if code_blocks and not re.search(r'```(?:mermaid|math)\s*\n', visible):
        issues.append('A report with relevant code needs a visible relationship visual; metadata tables do not substitute.')
    if code_blocks and not any(compact(plain(b)) in compact(visible) for b in code_blocks):
        issues.append('Keep a short decisive exact-code excerpt in the visible reading path.')
    for key in raw.get('semantic_assumptions', {}):
        if key not in all_text:
            issues.append(f'Missing exact assumption key in reference detail: {key}.')
    source_ptx = compact(ptx.read_text())
    physical_lines = ptx.read_text().splitlines()
    for reference in re.finditer(r'(%[prdf]+\d+)\s+at\s+(?:PTX\s+)?(?:line\s+)?(\d+)', claim_text):
        token, number = reference[1], int(reference[2])
        if not 1 <= number <= len(physical_lines) or not re.search(re.escape(token) + r'(?!\d)', physical_lines[number-1]):
            issues.append('Source citation ' + token + ' at physical PTX ' + str(number) + ' does not contain that register/predicate. Read the exact supplied line rather than estimating its coordinate.')
    for reference in re.finditer(r'\blocal (?:scratch |memory )?(?:array|scratch)\b[^.]{0,50}\b(?:written|stored|initialized)\s+at\s+(?:line\s+)?(\d+)', claim_text, re.I):
        number = int(reference[1])
        if not 1 <= number <= len(physical_lines) or not re.search(r'\bst\.local\.', physical_lines[number-1]):
            issues.append('The claimed local initialization at physical PTX ' + str(number) + ' is not a local store. Use the exact write coordinate.')
    for reference in re.finditer(r'\b(?:branch|jump)\s+at\s+(?:PTX\s+)?(?:line\s+)?(\d+)|\bskipped by\s+(\d+)', claim_text, re.I):
        number = int(reference[1] or reference[2])
        if not 1 <= number <= len(physical_lines) or not re.search(r'\bbra(?:\.uni)?\b', physical_lines[number-1]):
            issues.append('The claimed branch at physical PTX ' + str(number) + ' is not a branch instruction. Use the exact source coordinate and literal predicate; an election output flag and its skip predicate are different values.')
    for claim in re.finditer(r'\b(?:branch|jump)\b[^.\n]{0,140}?(\d+)[^.\n]{0,180}?\btaken(?:\s+only)?\s+(?:if|when)\s+(%p\d+)\s+(?:is\s+)?(true|false)\b', all_text, re.I):
        number = int(claim.group(1))
        if 1 <= number <= len(physical_lines):
            branch = re.search(r'@(!?)' + re.escape(claim.group(2)) + r'\s+bra\b', physical_lines[number - 1])
            expected = 'false' if branch and branch.group(1) else 'true'
            if branch and claim.group(3).lower() != expected:
                issues.append(f'Branch polarity contradicts PTX {number}: {claim.group(2)} must be {expected} to take this branch. Recheck its configured value and successor separately.')
    for block in code_blocks:
        statement = ''
        for line in plain(block).splitlines():
            line = line.strip()
            if not line or line in ('...', '…'):
                continue
            if re.match(r'^\d+[:\t ]', line):
                issues.append('Move physical line prefixes out of copied instructions into captions.')
                continue
            statement += line
            if line.endswith(';') or line.endswith(':'):
                if compact(statement) not in source_ptx:
                    issues.append('Instruction text differs from the supplied PTX: ' + statement[:130])
                statement = ''
        if statement and compact(statement) not in source_ptx:
            issues.append('Unmatched trailing code fragment: ' + statement[:130])
    if identity:
        manifest = json.loads(identity.read_text())
        for name, entry in manifest.get('files', {}).items():
            digest = entry.get('sha256')
            artifact = identity.parent / name
            if digest:
                if hashlib.sha256(artifact.read_bytes()).hexdigest() != digest:
                    issues.append(f'Supplied identity does not match artifact: {name}.')
                if digest not in source:
                    issues.append(f'Keep the full supplied SHA-256 for {name}; do not abbreviate it.')
    return issues


def check_current(report, ptx_path, input_path, result_path, draft_path=None, identity=None, source_paths=(), equivalence_path=None, artifact_base=None):
    """Check current composition and citation validity, not semantic correctness."""
    from report_layout import compose, MARKER
    from report_facts import verify_identity
    artifacts = [input_path, result_path, ptx_path]
    verify_identity(identity, artifacts)
    source = report.read_text()
    if not source.startswith(MARKER):
        return ['Expected a version-2 report. Recompose through the current skill.']
    draft = json.loads(draft_path.read_text()) if draft_path else None
    expected = compose(draft, json.loads(input_path.read_text()), json.loads(result_path.read_text()),
                       ptx_path.read_text(), artifacts, {p.name:p.read_text() for p in source_paths},
                       report_path=report, source_paths=source_paths, equivalence_path=equivalence_path, artifact_base=artifact_base)
    issues = []
    if source != expected:
        issues.append('Report differs from saved facts or its explanation plan. Recompose; do not edit Markdown.')
    depth = 0
    for tag in re.findall(r'</?details\b[^>]*>', source):
        depth += -1 if tag.startswith('</') else 1
        if depth < 0 or depth > 2:
            issues.append('Disclosures must be balanced and no deeper than two levels.')
            break
    if depth != 0:
        issues.append('Unbalanced disclosures.')
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('ptx', 'result'):
        parser.add_argument('--' + flag, required=True, type=Path)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--draft', type=Path)
    parser.add_argument('--identity', type=Path)
    parser.add_argument('--source', type=Path, action='append', default=[])
    parser.add_argument('--equivalence', type=Path)
    parser.add_argument('--artifact-base')
    args = parser.parse_args()
    try:
        from report_layout import validate_inputs
        from report_facts import verify_identity
        verify_identity(args.identity, [p for p in (args.input,args.result,args.ptx) if p is not None])
        raw = json.loads(args.result.read_text())
        if args.input:
            validate_inputs(json.loads(args.input.read_text()), raw, args.ptx.read_text())
        if not args.report:
            print('INPUT CHECK: identities and entry checked where supplied; no explanation checked.')
            return 0
        if not args.input:
            parser.error('--report requires --input for fact preservation checks.')
        issues = check_current(args.report,args.ptx,args.input,args.result,args.draft,args.identity,args.source,args.equivalence,args.artifact_base)
        for issue in issues:
            print('FAIL:',issue)
        if not issues:
            print('PASS: artifact agreement, composition, exact citations and disclosure structure. Semantic correctness and readability require review.')
        return bool(issues)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print('FAIL:', str(exc))
        return 1


if __name__ == '__main__':
    sys.exit(main())
