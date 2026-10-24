"""Optional GitHub-native layout of the existing checked report composition."""
import re


def render(document):
    opening, rest = document.split('## 1. Verifier result\n\n', 1)
    facts, after = rest.split('<!-- verifier-result:end -->', 1)
    intro = 'Saved findings and analysis scope, formatted directly from the verifier output.\n\n'
    if not facts.startswith(intro):
        raise ValueError('Expected the renderer-owned result introduction.')
    facts = facts[len(intro):]
    finding, scope = facts.split('### 1.2 Analysis scope\n\n', 1)
    heading, finding = finding.split('\n\n', 1)
    title = heading.removeprefix('### 1.1 ')

    # Metadata follows the relationship it identifies. Stops and unfamiliar
    # diagnostics retain their primary message in the default visible view.
    metadata = []
    labels = ['Saved result', 'Operation']
    if title in ('Synchronization requirement', 'Potentially conflicting accesses', 'Warning to inspect', 'Unresolved access relation'):
        labels.append('First diagnostic')
    def retain_metadata(match):
        metadata.append(match[0].strip())
        return ''
    finding = re.sub(r'^\*\*(?:' + '|'.join(labels) + r'):\*\* .*?\n\n', retain_metadata, finding, flags=re.M)

    def checks(match):
        # Original field values remain available; these are display labels.
        metadata.append(match[0].strip())
        outcomes = []
        names = ('Synchronization', 'Memory', 'Equivalence')
        values = {'passed': 'Passed', 'failed': 'Failed', 'not_run': 'Not run', 'not_requested': 'Not requested', 'incomplete': 'Incomplete'}
        for name, value in zip(names, match.groups()):
            if value is None:
                continue
            raw = value[len('<code>'):-len('</code>')]
            outcomes.append('**' + name + ':** ' + values.get(raw, value))
        return ' · '.join(outcomes) + '\n\n'
    finding = re.sub(r'\*\*Checks:\*\* Synchronization (<code>[^<]+</code>)(?: · Memory (<code>[^<]+</code>))? · Equivalence (<code>[^<]+</code>)\.\n\n', checks, finding)

    # Plots show operation families and physical locations. Full opcode
    # suffixes remain in the source excerpts and original records.
    def compact_graph(match):
        graph = match[0]
        graph = re.sub(r'\bmbarrier\.try_wait\.parity\.shared::cta\.b64', 'mbarrier.try_wait', graph)
        graph = re.sub(r'\btcgen05\.cp\.cta_group::1\.128x128b', 'tcgen05.cp', graph)
        graph = re.sub(r'\btcgen05\.mma\.sp\.cta_group::1\.kind::f16', 'tcgen05.mma.sp', graph)
        if len(re.findall(r'^  \w+\["', graph, re.M)) == 2:
            graph = graph.replace('flowchart TB', 'flowchart LR', 1)
        return graph
    finding = re.sub(r'```mermaid\nflowchart (?:TB|LR)\n.*?```', compact_graph, finding, flags=re.S)

    # Long PTX opcode suffixes must not push model IDs off the visible table.
    # GitHub hides line breaks inside inline code. Break between code spans;
    # the full name stays legible and copied source excerpts remain untouched.
    def wrap_token(match):
        lines, line = [], ''
        for segment in re.split(r'(?<=[._/])|(?<=::)', match[1]):
            if line and len(line + segment) > 34:
                lines.append(line)
                line = ''
            line += segment
        lines.append(line)
        return '<code>' + '</code><br/><code>'.join(lines) + '</code>'
    def wrap_row(match):
        return re.sub(r'<code>([^<\n]{35,})</code>', wrap_token, match[0])
    finding = re.sub(r'^\|.*\|$', wrap_row, finding, flags=re.M)

    # Preserve opening input/output links. The long record directory belongs
    # beside scope and evidence, after the reader has seen the finding.
    records = ''
    record_match = re.search(r'<details>\n<summary><strong>Analysis records .*?</details>\n\n', opening, re.S)
    if record_match:
        records = record_match[0]
        opening = opening[:record_match.start()] + opening[record_match.end():]
    metadata_view = ''
    if metadata:
        metadata_view = ('<details>\n<summary><strong>Saved status and diagnostic codes</strong></summary>\n\n'
                         + '\n\n'.join(metadata) + '\n\n</details>\n\n')
    panel = ('<table>\n<thead><tr><th align="left">VERIFIER RESULT · ' + title + '</th></tr></thead>\n'
             '<tbody><tr><td>\n\n' + finding.strip() + '\n\n' + metadata_view
             + '</td></tr>\n<tr><td>\n\n### Analysis scope\n\n' + scope.strip()
             + '\n\n' + records + '</td></tr></tbody>\n</table>\n')
    after = after.replace('## 2. Code explanation', '## Code explanation', 1)
    after = after.replace('<!-- code-explanation:start -->\n\n---\n\n', '<!-- code-explanation:start -->\n\n', 1)
    after = re.sub(r'^### 2\.(\d+) ', r'### \1. ', after, flags=re.M)
    return opening + panel + '<!-- verifier-result:end -->' + after
