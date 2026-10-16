#!/usr/bin/env python3
"""Print a GitHub-safe PTX excerpt; only display whitespace may change."""

import argparse
import html
import ipaddress
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit


def literal_token(value):
    if not value or any(char.isspace() or ord(char) < 32 for char in value):
        raise ValueError('A focus or link token must be nonempty and contain no whitespace.')
    return value


def link_mapping(value):
    token, separator, url = value.partition('=')
    if not separator:
        raise ValueError('A link must use TOKEN=HTTPS_URL.')
    literal_token(token)
    if not url or any(char.isspace() or ord(char) < 32 or char in '\\<>"' for char in url):
        raise ValueError('A link must be a valid absolute HTTPS URL.')
    try:
        parts = urlsplit(url)
        host = parts.hostname
        port = parts.port
    except ValueError as exc:
        raise ValueError('A link must be a valid absolute HTTPS URL.') from exc
    if parts.scheme != 'https' or not host or parts.username is not None or parts.password is not None:
        raise ValueError('A link must be an absolute HTTPS URL without credentials.')
    if port is not None and not 1 <= port <= 65535:
        raise ValueError('A link port must be between 1 and 65535.')
    if ':' in host:
        try:
            ipaddress.IPv6Address(host)
        except ValueError as exc:
            raise ValueError('A link must use a valid hostname or IP address.') from exc
    elif not all(re.fullmatch(r'[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?', label)
                 for label in host.rstrip('.').split('.')):
        raise ValueError('A link must use a valid ASCII hostname or IP address.')
    return token, url


def select_lines(value, count):
    selected = set()
    for part in value.split(','):
        part = part.strip()
        match = re.fullmatch(r'(\d+)(?:-(\d+))?', part)
        if not match:
            raise ValueError('Select physical lines with comma-separated numbers or ranges, such as 12-14,20.')
        first = int(match.group(1))
        last = int(match.group(2) or first)
        if first > last:
            raise ValueError('A physical-line range must be ascending: ' + part)
        if first < 1 or last > count:
            raise ValueError(f'Physical-line selection {part} is outside the supplied PTX (1-{count}).')
        selected.update(range(first, last + 1))
    return sorted(selected)


def line_ranges(selected):
    ranges = []
    for number in selected:
        if ranges and number == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], number)
        else:
            ranges.append((number, number))
    if len(ranges) > 3:
        return f'{selected[0]}\u2013{selected[-1]} (selected lines; gaps marked)'
    return ', '.join(str(first) if first == last else f'{first}\u2013{last}' for first, last in ranges)


def token_character(char):
    return char.isalnum() or char in '_.$%:'


def occurrences(source, token):
    """Match complete PTX tokens, including a register after a predicate @."""
    start = 0
    while True:
        start = source.find(token, start)
        if start == -1:
            return
        end = start + len(token)
        before_ok = start == 0 or not token_character(source[start - 1]) or not token_character(token[0])
        after_ok = end == len(source) or not token_character(source[end]) or not token_character(token[-1])
        if before_ok and after_ok:
            yield start, end
        start += 1


def mask_noncode(source):
    """Keep offsets/physical lines while masking comments and quoted literals."""
    masked = list(source)
    index = 0
    while index < len(source):
        if source.startswith('//', index):
            end = source.find('\n', index + 2)
            end = len(source) if end == -1 else end
        elif source.startswith('/*', index):
            end = source.find('*/', index + 2)
            end = len(source) if end == -1 else end + 2
        elif source[index] in ('"', "'"):
            quote = source[index]
            end = index + 1
            while end < len(source):
                if source[end] == '\\':
                    end += 2
                elif source[end] == quote:
                    end += 1
                    break
                else:
                    end += 1
        else:
            index += 1
            continue
        for position in range(index, min(end, len(source))):
            if source[position] not in '\r\n':
                masked[position] = ' '
        index = end
    return ''.join(masked)


def instruction_opcodes(masked_line):
    """Locate instruction opcodes after optional labels and a PTX predicate."""
    pattern = re.compile(
        r'\s*(?:[{}]\s*)*'
        r'(?:(?:[A-Za-z_$][A-Za-z0-9_.$]*):(?!:)\s*)*'
        r'(?:@!?%?[A-Za-z_.$][A-Za-z0-9_.$]*\s+)?'
        r'(?P<opcode>[A-Za-z][A-Za-z0-9_.]*(?:::[A-Za-z0-9_.]+)*)(?=\s|[{};]|$)')
    spans = []
    for statement in re.finditer(r'(?:^|;)([^;]*)', masked_line):
        match = pattern.match(statement.group(1))
        if match:
            offset = statement.start(1)
            spans.append((offset + match.start('opcode'), offset + match.end('opcode')))
    return spans


def wait_counts(masked_line, opcodes):
    """Locate the single immediate operand of supported wait-group forms."""
    spans = []
    for first, last in opcodes:
        opcode = masked_line[first:last]
        if not re.match(r'(?:cp\.async(?:\.bulk)?|wgmma)\.wait_group(?:\.|$)', opcode):
            continue
        count = re.match(r'\s+(0[xX][0-9a-fA-F]+|\d+)\s*(?=;|$)', masked_line[last:])
        if count:
            spans.append((last + count.start(1), last + count.end(1)))
    return spans


def automatic_emphasis(source_line, displayed, landmarks, links):
    """Map exact instruction landmarks through whitespace-only changes."""
    spans = []
    for first, last in landmarks:
        token = source_line[first:last]
        if token in links:
            continue
        original_matches = list(occurrences(source_line, token))
        ordinal = original_matches.index((first, last))
        display_matches = list(occurrences(displayed, token))
        if ordinal >= len(display_matches):
            raise ValueError('Could not preserve the instruction landmark during display wrapping: ' + token)
        spans.append(display_matches[ordinal])
    return spans


def breakpoints(source):
    """Yield safe whitespace/operand breaks outside quoted strings and comments."""
    index = 0
    while index < len(source):
        char = source[index]
        if source.startswith('//', index):
            return  # Wrapping inside // would change the meaning of the source.
        if source.startswith('/*', index):
            end = source.find('*/', index + 2)
            if end == -1:
                return
            index = end + 2
            continue
        if char in ('"', "'"):
            quote = char
            index += 1
            while index < len(source):
                if source[index] == '\\':
                    index += 2
                elif source[index] == quote:
                    index += 1
                    break
                else:
                    index += 1
            continue
        if char.isspace():
            end = index + 1
            while end < len(source) and source[end].isspace():
                end += 1
            yield index, end
            index = end
            continue
        if char == ',':
            end = index + 1
            while end < len(source) and source[end].isspace():
                end += 1
            yield index + 1, end
            index = end
            continue
        index += 1


def wrap_line(source, width, protected_tokens=()):
    source = source.strip()
    # PTX listings use tabs. Normalize only safe display separators so their
    # rendered width is predictable; spaces inside strings/comments stay exact.
    pieces = []
    cursor = 0
    for left, right in breakpoints(source):
        if right > left:
            pieces.extend((source[cursor:left], ' '))
            cursor = right
    pieces.append(source[cursor:])
    source = ''.join(pieces)
    protected = [span for token in protected_tokens for span in occurrences(source, token)]
    breaks = [(left, right) for left, right in breakpoints(source)
              if not any(first < left < last or first < right < last for first, last in protected)]
    output = []
    cursor = 0
    while cursor < len(source):
        indent = '' if not output else '    '
        available = max(1, width - len(indent))
        if len(source) - cursor <= available:
            output.append(indent + source[cursor:])
            break
        following = [(left, right) for left, right in breaks if left > cursor]
        within = [(left, right) for left, right in following if left - cursor <= available]
        if within:
            left, right = within[-1]
        elif following:
            left, right = following[0]  # An indivisible source token may exceed the target width.
        else:
            output.append(indent + source[cursor:])
            break
        output.append(indent + source[cursor:left])
        cursor = right
    return output or ['']


def render_line(source, focus, links, automatic_spans=()):
    bold_spans = [span for token in focus for span in occurrences(source, token)] + list(automatic_spans)
    link_spans = [(first, last, url) for token, url in links.items()
                  for first, last in occurrences(source, token)]
    # Longest match at the same position wins. Never nest overlapping links.
    chosen = []
    for span in sorted(link_spans, key=lambda item: (item[0], -(item[1] - item[0]), item[2])):
        if not chosen or span[0] >= chosen[-1][1]:
            chosen.append(span)

    def render_bold(first, last):
        boundaries = {first, last}
        for start, end in bold_spans:
            if start < last and end > first:
                boundaries.update((max(first, start), min(last, end)))
        positions = sorted(boundaries)
        chunks = []
        for start, end in zip(positions, positions[1:]):
            escaped = html.escape(source[start:end], quote=False)
            if any(left <= start and end <= right for left, right in bold_spans):
                escaped = '<strong>' + escaped + '</strong>'
            chunks.append(escaped)
        return ''.join(chunks)

    chunks = []
    cursor = 0
    for first, last, url in chosen:
        chunks.append(render_bold(cursor, first))
        chunks.append('<a href="' + html.escape(url, quote=True) + '">' + render_bold(first, last) + '</a>')
        cursor = last
    chunks.append(render_bold(cursor, len(source)))
    return ''.join(chunks)


def excerpt(source, selection, focus=(), links=None, width=72, caption='PTX excerpt', *, source_url=None):
    if width < 1:
        raise ValueError('Display width must be a positive integer.')
    if not caption.strip() or '\n' in caption or '\r' in caption:
        raise ValueError('The semantic caption must be a nonempty single line.')
    for token in focus:
        literal_token(token)
    links = dict(links or {})
    for token, url in links.items():
        link_mapping(token + '=' + url)
    physical = source.splitlines()
    masked = mask_noncode(source).splitlines()
    selected = select_lines(selection, len(physical))
    display = []
    wrapped = []
    previous = None
    for number in selected:
        if previous is not None and number > previous + 1:
            display.append('\u2026')
        source_line = physical[number - 1]
        lines = wrap_line(source_line, width, tuple(focus) + tuple(links))
        if len(lines) > 1:
            wrapped.append(number)
        opcodes = instruction_opcodes(masked[number - 1])
        landmarks = opcodes + wait_counts(masked[number - 1], opcodes)
        automatic = automatic_emphasis(source_line, '\n'.join(lines), landmarks, links)
        offset = 0
        for line in lines:
            local = [(first - offset, last - offset) for first, last in automatic
                     if offset <= first and last <= offset + len(line)]
            display.append(render_line(line, focus, links, local))
            offset += len(line) + 1
        previous = number
    title = html.escape(caption.strip(), quote=False)
    note = ' \u00b7 display wrapping added' if wrapped else ''
    location = 'physical PTX ' + line_ranges(selected)
    if source_url is not None:
        parts = urlsplit(source_url)
        if parts.scheme not in ('', 'https') or parts.fragment or parts.username or parts.password or (parts.netloc and parts.scheme != 'https'):
            raise ValueError('Source links require a relative path or HTTPS URL without credentials or fragment.')
        fragment = '#L' + str(selected[0])
        if selected[-1] != selected[0]:
            fragment += '-L' + str(selected[-1])
        location = '<a href="' + html.escape(source_url + fragment, quote=True) + '" title="Open the original PTX">' + location + '</a>'
    return ('<p><strong>' + title + '</strong> \u00b7 ' + location +
            note + '</p>\n\n<pre><code>' + '\n'.join(display) + '</code></pre>\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ptx', required=True, type=Path)
    parser.add_argument('--lines', required=True, help='Physical lines: 12-14,20. Nonadjacent ranges have an omission marker.')
    parser.add_argument('--focus', action='append', default=[],
                        help='Exact decisive predicate/operand to emphasize; repeatable. Unlinked opcodes and wait-group counts are emphasized automatically.')
    parser.add_argument('--link', action='append', default=[], help='Exact TOKEN=HTTPS_URL mapping; repeatable.')
    parser.add_argument('--width', type=int, default=72)
    parser.add_argument('--caption', default='PTX excerpt', help='Short semantic purpose, such as Test ownership.')
    args = parser.parse_args()
    try:
        links = {}
        for value in args.link:
            token, url = link_mapping(value)
            if token in links and links[token] != url:
                raise ValueError('Conflicting URLs for token: ' + token)
            links[token] = url
        rendered = excerpt(args.ptx.read_text(encoding='utf-8'), args.lines, args.focus,
                           links, args.width, args.caption)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    sys.stdout.write(rendered)
    return 0


if __name__ == '__main__':
    sys.exit(main())
