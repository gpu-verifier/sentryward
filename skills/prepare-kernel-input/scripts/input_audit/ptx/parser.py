"""Selected-entry PTX syntax and CFG helpers for the static input audit.

There is no module execution or lowering API here. Instruction contents remain
opaque except for the control-flow syntax needed to establish dominating aliases.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import re
from typing import Iterable, Sequence

from .model import (
    BasicBlock,
    CfgEdge,
    ControlKind,
    DebugLocation,
    EdgeKind,
    FunctionItem,
    FunctionKind,
    Predicate,
    PtxDirective,
    PtxFunction,
    PtxInstruction,
    PtxLabel,
    PtxOperand,
    PtxParameter,
    PtxParseError,
    SourcePosition,
    SourceSpan,
)


_IDENTIFIER_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_LABEL_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_PREDICATE_RE = re.compile(
    r"@(?P<negated>!)?"
    r"(?P<register>%?[A-Za-z_$][A-Za-z0-9_$]*)\s+"
)
_OPCODE_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+(?:::[A-Za-z0-9_]+)*)*"
)
_FUNCTION_START_RE = re.compile(
    r"(?m)^[ \t]*"
    r"(?P<qualifiers>(?:\.(?:visible|extern|weak)\s+)*)"
    r"\.(?P<kind>entry|func)\b"
)
_DIRECTIVE_NAME_RE = re.compile(r"\.[A-Za-z_][A-Za-z0-9_]*")

# Compiler-emitted line directives terminate at the physical newline.
_LINE_DIRECTIVES = {
    ".address_size",
    ".blocksareclusters",
    ".explicitcluster",
    ".file",
    ".loc",
    ".maxclusterrank",
    ".maxnctapersm",
    ".maxnreg",
    ".maxntid",
    ".minnctapersm",
    ".noreturn",
    ".reqnctapercluster",
    ".reqntid",
    ".section",
    ".target",
    ".version",
}


@dataclass(frozen=True)
class _FunctionRegion:
    start: int
    kind_start: int
    kind_end: int
    header_end: int
    body_start: int | None
    body_end: int | None
    end: int
    kind: FunctionKind
    qualifiers: tuple[str, ...]


class _SourceMap:
    def __init__(self, text: str, source_name: str) -> None:
        self.text = text
        self.source_name = source_name
        self.line_starts = [0]
        self.line_starts.extend(
            index + 1 for index, char in enumerate(text) if char == "\n"
        )

    def position(self, offset: int) -> SourcePosition:
        if not 0 <= offset <= len(self.text):
            raise ValueError(f"source offset out of range: {offset}")
        line_index = bisect_right(self.line_starts, offset) - 1
        return SourcePosition(
            line=line_index + 1,
            column=offset - self.line_starts[line_index] + 1,
            offset=offset,
        )

    def span(self, start: int, end: int) -> SourceSpan:
        return SourceSpan(self.position(start), self.position(end))

    def error(self, message: str, offset: int) -> PtxParseError:
        position = self.position(min(max(offset, 0), len(self.text)))
        line_start = self.line_starts[position.line - 1]
        line_end = self.text.find("\n", line_start)
        if line_end < 0:
            line_end = len(self.text)
        context = self.text[line_start:line_end].rstrip()
        return PtxParseError(
            message,
            source_name=self.source_name,
            position=position,
            context=context,
        )


def _mask_comments(text: str, source_map: _SourceMap) -> str:
    """Replace comment bytes with spaces while preserving offsets/newlines."""

    chars = list(text)
    index = 0
    in_string = False
    escaped = False
    while index < len(text):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            index += 1
            continue
        if text.startswith("//", index):
            end = text.find("\n", index + 2)
            if end < 0:
                end = len(text)
            for cursor in range(index, end):
                chars[cursor] = " "
            index = end
            continue
        if text.startswith("/*", index):
            end_marker = text.find("*/", index + 2)
            if end_marker < 0:
                raise source_map.error("unterminated block comment", index)
            end = end_marker + 2
            for cursor in range(index, end):
                if chars[cursor] != "\n":
                    chars[cursor] = " "
            index = end
            continue
        index += 1
    if in_string:
        raise source_map.error("unterminated string literal", len(text))
    return "".join(chars)


def _find_function_regions(
    masked: str, source_map: _SourceMap
) -> tuple[_FunctionRegion, ...]:
    regions: list[_FunctionRegion] = []
    search_at = 0
    while True:
        match = _FUNCTION_START_RE.search(masked, search_at)
        if not match:
            break
        start = match.start()
        # Function regions must not overlap preceding initializers.
        if regions and start < regions[-1].end:
            search_at = regions[-1].end
            continue
        kind = FunctionKind(match.group("kind"))
        qualifiers = tuple(
            token[1:]
            for token in re.findall(r"\.[A-Za-z_][A-Za-z0-9_]*",
                                    match.group("qualifiers"))
        )
        terminator, terminator_offset = _scan_header_terminator(
            masked, match.end(), source_map
        )
        if terminator == ";":
            region = _FunctionRegion(
                start=start,
                kind_start=match.start("kind") - 1,
                kind_end=match.end("kind"),
                header_end=terminator_offset,
                body_start=None,
                body_end=None,
                end=terminator_offset + 1,
                kind=kind,
                qualifiers=qualifiers,
            )
        else:
            close = _matching_delimiter(
                masked, terminator_offset, "{", "}", source_map
            )
            region = _FunctionRegion(
                start=start,
                kind_start=match.start("kind") - 1,
                kind_end=match.end("kind"),
                header_end=terminator_offset,
                body_start=terminator_offset + 1,
                body_end=close,
                end=close + 1,
                kind=kind,
                qualifiers=qualifiers,
            )
        regions.append(region)
        search_at = region.end
    return tuple(regions)


def _scan_header_terminator(
    text: str, start: int, source_map: _SourceMap
) -> tuple[str, int]:
    paren = bracket = 0
    in_string = escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "(":
            paren += 1
        elif char == ")":
            paren -= 1
            if paren < 0:
                raise source_map.error(
                    "unmatched ')' in function header", index
                )
        elif char == "[":
            bracket += 1
        elif char == "]":
            bracket -= 1
            if bracket < 0:
                raise source_map.error(
                    "unmatched ']' in function header", index
                )
        elif paren == 0 and bracket == 0 and char in "{;":
            return char, index
    raise source_map.error("unterminated PTX function header", start)


def _matching_delimiter(
    text: str,
    opening: int,
    open_char: str,
    close_char: str,
    source_map: _SourceMap,
) -> int:
    depth = 0
    in_string = escaped = False
    for index in range(opening, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return index
            if depth < 0:
                break
    raise source_map.error(
        f"unmatched {open_char!r} delimiter", opening
    )


def _parse_function(
    original: str,
    masked: str,
    source_map: _SourceMap,
    region: _FunctionRegion,
) -> PtxFunction:
    name, parameters, returns, attributes = _parse_function_header(
        original, masked, source_map, region
    )
    if region.body_start is None or region.body_end is None:
        return PtxFunction(
            name=name,
            kind=region.kind,
            qualifiers=region.qualifiers,
            parameters=parameters,
            return_parameters=returns,
            attributes=attributes,
            items=(),
            instructions=(),
            blocks=(),
            span=source_map.span(region.start, region.end),
            header_span=source_map.span(region.start, region.header_end),
            has_body=False,
        )

    items = _parse_body(
        original,
        masked,
        source_map,
        region.body_start,
        region.body_end,
    )
    instructions = tuple(
        item for item in items if isinstance(item, PtxInstruction)
    )
    blocks = _build_cfg(items, source_map, region.body_start, region.body_end)
    return PtxFunction(
        name=name,
        kind=region.kind,
        qualifiers=region.qualifiers,
        parameters=parameters,
        return_parameters=returns,
        attributes=attributes,
        items=tuple(items),
        instructions=instructions,
        blocks=blocks,
        span=source_map.span(region.start, region.end),
        header_span=source_map.span(region.start, region.header_end),
        has_body=True,
    )


def _parse_function_header(
    original: str,
    masked: str,
    source_map: _SourceMap,
    region: _FunctionRegion,
) -> tuple[
    str,
    tuple[PtxParameter, ...],
    tuple[PtxParameter, ...],
    tuple[PtxDirective, ...],
]:
    cursor = _skip_ws(masked, region.kind_end, region.header_end)
    returns: tuple[PtxParameter, ...] = ()
    if region.kind is FunctionKind.FUNC and cursor < region.header_end:
        if masked[cursor] == "(":
            close = _matching_delimiter(
                masked, cursor, "(", ")", source_map
            )
            if close >= region.header_end:
                raise source_map.error(
                    "return parameter list escapes function header", cursor
                )
            returns = _parse_parameter_list(
                original, masked, source_map, cursor + 1, close
            )
            cursor = _skip_ws(masked, close + 1, region.header_end)

    name_match = _IDENTIFIER_RE.match(masked, cursor, region.header_end)
    if not name_match:
        raise source_map.error("expected PTX function name", cursor)
    name = name_match.group()
    cursor = _skip_ws(masked, name_match.end(), region.header_end)

    parameters: tuple[PtxParameter, ...] = ()
    if cursor < region.header_end and masked[cursor] == "(":
        close = _matching_delimiter(
            masked, cursor, "(", ")", source_map
        )
        if close >= region.header_end:
            raise source_map.error(
                "parameter list escapes function header", cursor
            )
        parameters = _parse_parameter_list(
            original, masked, source_map, cursor + 1, close
        )
        cursor = close + 1

    attributes: list[PtxDirective] = []
    while True:
        cursor = _skip_ws(masked, cursor, region.header_end)
        if cursor >= region.header_end:
            break
        if masked[cursor] != ".":
            raise source_map.error(
                "unexpected text after function parameter list", cursor
            )
        directive, cursor = _parse_directive(
            original,
            masked,
            source_map,
            cursor,
            region.header_end,
            None,
            module_scope=False,
        )
        attributes.append(directive)
    return name, parameters, returns, tuple(attributes)


def _parse_parameter_list(
    original: str,
    masked: str,
    source_map: _SourceMap,
    start: int,
    end: int,
) -> tuple[PtxParameter, ...]:
    parameters: list[PtxParameter] = []
    for part_start, part_end in _split_top_level_ranges(
        masked, start, end, ",", source_map
    ):
        trimmed_start, trimmed_end = _trim_range(
            masked, part_start, part_end
        )
        if trimmed_start == trimmed_end:
            continue
        raw = original[trimmed_start:trimmed_end]
        tokens = re.findall(
            r"\.[A-Za-z_][A-Za-z0-9_]*|"
            r"[A-Za-z_$][A-Za-z0-9_$]*(?:\[[^\]]*\])?",
            masked[trimmed_start:trimmed_end],
        )
        storage = tokens[0] if tokens and tokens[0] in {
            ".param", ".reg"
        } else None
        align_match = re.search(
            r"\.align\s+(\d+)", masked[trimmed_start:trimmed_end]
        )
        alignment = int(align_match.group(1)) if align_match else None
        dot_tokens = tuple(
            token
            for token in tokens
            if token.startswith(".")
            and token not in {storage, ".align"}
        )
        name_match = re.search(
            r"([A-Za-z_$][A-Za-z0-9_$]*)(\[[^\]]*\])?\s*$",
            masked[trimmed_start:trimmed_end],
        )
        name = name_match.group(1) if name_match else None
        array_spec = name_match.group(2) if name_match else None
        parameters.append(
            PtxParameter(
                name=name,
                storage=storage,
                type_specifiers=dot_tokens,
                alignment=alignment,
                array_specifier=array_spec,
                raw=raw,
                span=source_map.span(trimmed_start, trimmed_end),
            )
        )
    return tuple(parameters)


def _parse_body(
    original: str,
    masked: str,
    source_map: _SourceMap,
    start: int,
    end: int,
) -> list[FunctionItem]:
    items: list[FunctionItem] = []
    cursor = start
    current_location: DebugLocation | None = None
    instruction_index = 0
    scope_stack: list[int] = []
    next_scope_id = 0
    while cursor < end:
        while cursor < end and (
            masked[cursor].isspace() or masked[cursor] == ";"
        ):
            cursor += 1
        if cursor >= end:
            break
        if masked[cursor] == "{":
            next_scope_id += 1
            scope_stack.append(next_scope_id)
            cursor += 1
            continue
        if masked[cursor] == "}":
            if not scope_stack:
                raise source_map.error(
                    "unmatched '}' in PTX function body", cursor
                )
            scope_stack.pop()
            cursor += 1
            continue
        scope = tuple(scope_stack)

        label_match = _LABEL_RE.match(masked, cursor, end)
        if label_match:
            after = _skip_horizontal_ws(masked, label_match.end(), end)
            if after < end and masked[after] == ":":
                label_end = after + 1
                items.append(
                    PtxLabel(
                        name=label_match.group(),
                        raw=original[cursor:label_end],
                        span=source_map.span(cursor, label_end),
                        scope=scope,
                    )
                )
                cursor = label_end
                continue

        if masked[cursor] == ".":
            directive, cursor = _parse_directive(
                original,
                masked,
                source_map,
                cursor,
                end,
                current_location,
                module_scope=False,
                scope=scope,
            )
            items.append(directive)
            if directive.name == ".loc":
                current_location = _parse_debug_location(
                    directive, source_map
                )
            continue

        statement_end = _find_statement_semicolon(
            masked, cursor, end, source_map
        )
        instruction = _parse_instruction(
            original,
            masked,
            source_map,
            cursor,
            statement_end + 1,
            instruction_index,
            current_location,
            scope,
        )
        items.append(instruction)
        instruction_index += 1
        cursor = statement_end + 1
    if scope_stack:
        raise source_map.error(
            "unterminated nested scope in PTX function body", end
        )
    return items


def _parse_directive(
    original: str,
    masked: str,
    source_map: _SourceMap,
    start: int,
    limit: int,
    current_location: DebugLocation | None,
    *,
    module_scope: bool,
    scope: tuple[int, ...] = (),
    force_line: bool = False,
) -> tuple[PtxDirective, int]:
    match = _DIRECTIVE_NAME_RE.match(masked, start, limit)
    if not match:
        raise source_map.error("malformed PTX directive", start)
    name = match.group()
    if force_line or name in _LINE_DIRECTIVES:
        newline = masked.find("\n", match.end(), limit)
        end = limit if newline < 0 else newline
        # Exclude an optional semicolon from line-directive arguments.
        trimmed_end = end
        while trimmed_end > match.end() and masked[trimmed_end - 1].isspace():
            trimmed_end -= 1
        if trimmed_end > match.end() and masked[trimmed_end - 1] == ";":
            trimmed_end -= 1
        next_cursor = end + 1 if newline >= 0 else end
    else:
        try:
            semicolon = _find_statement_semicolon(
                masked, start, limit, source_map
            )
        except PtxParseError:
            if not module_scope:
                raise
            # Preserve newline-terminated module directives.
            newline = masked.find("\n", match.end(), limit)
            if newline < 0:
                newline = limit
            trimmed_end = newline
            next_cursor = newline + 1 if newline < limit else newline
        else:
            trimmed_end = semicolon
            next_cursor = semicolon + 1
    args_start, args_end = _trim_range(
        masked, match.end(), trimmed_end
    )
    raw_end = next_cursor
    while raw_end > start and original[raw_end - 1] in "\r\n":
        raw_end -= 1
    directive = PtxDirective(
        name=name,
        arguments=original[args_start:args_end],
        raw=original[start:raw_end],
        span=source_map.span(start, raw_end),
        debug_location=current_location,
        scope=scope,
    )
    return directive, next_cursor


def _parse_debug_location(
    directive: PtxDirective, source_map: _SourceMap
) -> DebugLocation:
    match = re.match(
        r"\s*(?P<file>\d+)\s+(?P<line>\d+)\s+(?P<column>\d+)"
        r"(?P<attributes>.*)$",
        directive.arguments,
        flags=re.DOTALL,
    )
    if not match:
        raise source_map.error(
            "malformed .loc directive", directive.span.start.offset
        )
    return DebugLocation(
        file_id=int(match.group("file")),
        line=int(match.group("line")),
        column=int(match.group("column")),
        attributes=match.group("attributes").strip(),
        raw=directive.raw,
    )


def _find_statement_semicolon(
    text: str, start: int, end: int, source_map: _SourceMap
) -> int:
    paren = bracket = brace = 0
    in_string = escaped = False
    for index in range(start, end):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "(":
            paren += 1
        elif char == ")":
            paren -= 1
        elif char == "[":
            bracket += 1
        elif char == "]":
            bracket -= 1
        elif char == "{":
            brace += 1
        elif char == "}":
            brace -= 1
        elif char == ";" and paren == bracket == brace == 0:
            return index
        if min(paren, bracket, brace) < 0:
            raise source_map.error(
                "unbalanced delimiter in PTX statement", index
            )
    raise source_map.error(
        "PTX statement is missing its terminating ';'", start
    )


def _parse_instruction(
    original: str,
    masked: str,
    source_map: _SourceMap,
    start: int,
    end: int,
    index: int,
    debug_location: DebugLocation | None,
    scope: tuple[int, ...],
) -> PtxInstruction:
    statement_start, statement_end = _trim_range(
        masked, start, end - 1
    )
    if statement_start == statement_end:
        raise source_map.error("empty PTX instruction", start)
    cursor = statement_start
    predicate: Predicate | None = None
    pred_match = _PREDICATE_RE.match(masked, cursor, statement_end)
    if pred_match:
        predicate = Predicate(
            register=pred_match.group("register"),
            negated=bool(pred_match.group("negated")),
        )
        cursor = pred_match.end()

    opcode_match = _OPCODE_RE.match(masked, cursor, statement_end)
    if not opcode_match:
        raise source_map.error("expected PTX opcode", cursor)
    opcode_text = opcode_match.group()
    opcode_parts = opcode_text.split(".")
    cursor = opcode_match.end()
    # tcgen05 operands may start immediately with ``{`` or ``[``.
    if (
        cursor < statement_end
        and not masked[cursor].isspace()
        and masked[cursor] not in "{["
    ):
        raise source_map.error(
            "unexpected character after PTX opcode", cursor
        )
    operands: list[PtxOperand] = []
    operand_start = _skip_ws(masked, cursor, statement_end)
    if operand_start < statement_end:
        for part_start, part_end in _split_top_level_ranges(
            masked, operand_start, statement_end, ",", source_map
        ):
            trimmed_start, trimmed_end = _trim_range(
                masked, part_start, part_end
            )
            if trimmed_start == trimmed_end:
                raise source_map.error("empty PTX operand", part_start)
            operands.append(
                PtxOperand(
                    text=original[trimmed_start:trimmed_end],
                    span=source_map.span(trimmed_start, trimmed_end),
                )
            )

    control_kind, targets = _classify_control(
        opcode_parts[0], operands, source_map, statement_start
    )
    return PtxInstruction(
        index=index,
        predicate=predicate,
        opcode=opcode_parts[0],
        modifiers=tuple(opcode_parts[1:]),
        operands=tuple(operands),
        raw=original[statement_start:end],
        span=source_map.span(statement_start, end),
        debug_location=debug_location,
        control_kind=control_kind,
        branch_targets=targets,
        scope=scope,
    )


def _classify_control(
    opcode: str,
    operands: Sequence[PtxOperand],
    source_map: _SourceMap,
    offset: int,
) -> tuple[ControlKind, tuple[str, ...]]:
    if opcode == "bra":
        if len(operands) != 1:
            raise source_map.error(
                "direct bra must have exactly one target operand", offset
            )
        target = operands[0].text.strip()
        if not _LABEL_RE.fullmatch(target):
            raise source_map.error(
                f"invalid direct branch target {target!r}", offset
            )
        return ControlKind.DIRECT_BRANCH, (target,)
    if opcode == "brx":
        if len(operands) < 2:
            raise source_map.error(
                "brx must name an index and a branch-target table", offset
            )
        table = operands[-1].text.strip()
        if not _LABEL_RE.fullmatch(table):
            raise source_map.error(
                f"invalid indirect branch table {table!r}", offset
            )
        return ControlKind.INDIRECT_BRANCH, (table,)
    if opcode == "call":
        return ControlKind.CALL, ()
    if opcode == "ret":
        return ControlKind.RETURN, ()
    if opcode == "exit":
        return ControlKind.EXIT, ()
    if opcode == "trap":
        return ControlKind.TRAP, ()
    return ControlKind.NONE, ()


@dataclass
class _MutableBlock:
    id: str
    index: int
    label_nodes: list[PtxLabel]
    instructions: list[PtxInstruction]
    directives: list[PtxDirective]
    start: int
    end: int
    successors: list[CfgEdge]
    predecessors: list[CfgEdge]


def _build_cfg(
    items: Sequence[FunctionItem],
    source_map: _SourceMap,
    body_start: int,
    body_end: int,
) -> tuple[BasicBlock, ...]:
    blocks: list[_MutableBlock] = []
    pending_labels: list[PtxLabel] = []
    pending_directives: list[PtxDirective] = []
    current: _MutableBlock | None = None

    def begin_block(offset: int) -> _MutableBlock:
        nonlocal pending_labels, pending_directives
        block = _MutableBlock(
            id=f"bb{len(blocks)}",
            index=len(blocks),
            label_nodes=list(pending_labels),
            instructions=[],
            directives=list(pending_directives),
            start=(
                pending_labels[0].span.start.offset
                if pending_labels
                else pending_directives[0].span.start.offset
                if pending_directives
                else offset
            ),
            end=offset,
            successors=[],
            predecessors=[],
        )
        pending_labels = []
        pending_directives = []
        blocks.append(block)
        return block

    for item in items:
        if isinstance(item, PtxLabel):
            if current is not None and (
                current.instructions or current.directives
            ):
                current = None
            pending_labels.append(item)
            continue
        if isinstance(item, PtxDirective):
            if current is None:
                pending_directives.append(item)
            else:
                current.directives.append(item)
                current.end = item.span.end.offset
            continue
        if current is None:
            current = begin_block(item.span.start.offset)
        current.instructions.append(item)
        current.end = item.span.end.offset
        if item.is_terminator:
            current = None

    if pending_labels or pending_directives:
        begin_block(
            pending_labels[0].span.start.offset
            if pending_labels
            else pending_directives[0].span.start.offset
        )
    if not blocks and items:
        begin_block(items[0].span.start.offset)
    if not blocks:
        return ()

    label_to_block: dict[str, _MutableBlock] = {}
    scoped_label_to_block: dict[
        tuple[tuple[int, ...], str], _MutableBlock
    ] = {}
    for block in blocks:
        for label in block.label_nodes:
            key = (label.scope, label.name)
            if key in scoped_label_to_block:
                raise source_map.error(
                    f"duplicate PTX label {label.name!r} in one scope",
                    label.span.start.offset,
                )
            scoped_label_to_block[key] = block
            label_to_block[label.qualified_name] = block

    def resolve_label(
        scope: tuple[int, ...],
        name: str,
        offset: int,
        description: str,
    ) -> _MutableBlock:
        for depth in range(len(scope), -1, -1):
            target = scoped_label_to_block.get((scope[:depth], name))
            if target is not None:
                return target
        raise source_map.error(
            f"undefined {description} {name!r}", offset
        )

    branch_tables: dict[
        tuple[tuple[int, ...], str],
        tuple[tuple[str, ...], tuple[int, ...]],
    ] = {}
    for block in blocks:
        for directive in block.directives:
            if directive.name != ".branchtargets":
                continue
            if not block.label_nodes:
                raise source_map.error(
                    ".branchtargets directive must follow a table label",
                    directive.span.start.offset,
                )
            targets = tuple(
                original.strip()
                for original in directive.arguments.split(",")
                if original.strip()
            )
            if not targets or any(
                not _LABEL_RE.fullmatch(target) for target in targets
            ):
                raise source_map.error(
                    "malformed .branchtargets directive",
                    directive.span.start.offset,
                )
            table_label = block.label_nodes[-1]
            branch_tables[(table_label.scope, table_label.name)] = (
                targets,
                directive.scope,
            )

    def add_edge(
        source: _MutableBlock,
        target: _MutableBlock,
        kind: EdgeKind,
        predicate: Predicate | None = None,
    ) -> None:
        edge = CfgEdge(source.id, target.id, kind, predicate)
        if edge not in source.successors:
            source.successors.append(edge)
            target.predecessors.append(edge)

    for block_index, block in enumerate(blocks):
        next_block = (
            blocks[block_index + 1]
            if block_index + 1 < len(blocks)
            else None
        )
        terminator = (
            block.instructions[-1] if block.instructions else None
        )
        if terminator is None or not terminator.is_terminator:
            if next_block is not None:
                add_edge(block, next_block, EdgeKind.FALLTHROUGH)
            continue

        if terminator.control_kind is ControlKind.DIRECT_BRANCH:
            target_name = terminator.branch_targets[0]
            target = resolve_label(
                terminator.scope,
                target_name,
                terminator.span.start.offset,
                "direct branch target",
            )
            add_edge(
                block, target, EdgeKind.BRANCH, terminator.predicate
            )
            if terminator.predicate is not None and next_block is not None:
                add_edge(block, next_block, EdgeKind.FALLTHROUGH)
            continue

        if terminator.control_kind is ControlKind.INDIRECT_BRANCH:
            table_name = terminator.branch_targets[0]
            table_entry = None
            for depth in range(len(terminator.scope), -1, -1):
                table_entry = branch_tables.get(
                    (terminator.scope[:depth], table_name)
                )
                if table_entry is not None:
                    break
            if table_entry is None:
                raise source_map.error(
                    f"undefined .branchtargets table {table_name!r}",
                    terminator.span.start.offset,
                )
            targets, table_scope = table_entry
            for target_name in targets:
                target = resolve_label(
                    table_scope,
                    target_name,
                    terminator.span.start.offset,
                    "indirect branch target",
                )
                add_edge(
                    block,
                    target,
                    EdgeKind.INDIRECT_BRANCH,
                    terminator.predicate,
                )
            if terminator.predicate is not None and next_block is not None:
                add_edge(block, next_block, EdgeKind.FALLTHROUGH)
            continue

        if terminator.predicate is not None and next_block is not None:
            add_edge(block, next_block, EdgeKind.FALLTHROUGH)

    return tuple(
        BasicBlock(
            id=block.id,
            index=block.index,
            labels=tuple(label.name for label in block.label_nodes),
            instructions=tuple(block.instructions),
            directives=tuple(block.directives),
            successors=tuple(block.successors),
            predecessors=tuple(block.predecessors),
            span=source_map.span(
                block.start,
                max(block.start, block.end),
            ),
            qualified_labels=tuple(
                label.qualified_name for label in block.label_nodes
            ),
        )
        for block in blocks
    )


def _split_top_level_ranges(
    text: str,
    start: int,
    end: int,
    separator: str,
    source_map: _SourceMap,
) -> Iterable[tuple[int, int]]:
    paren = bracket = brace = 0
    in_string = escaped = False
    part_start = start
    for index in range(start, end):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "(":
            paren += 1
        elif char == ")":
            paren -= 1
        elif char == "[":
            bracket += 1
        elif char == "]":
            bracket -= 1
        elif char == "{":
            brace += 1
        elif char == "}":
            brace -= 1
        elif char == separator and paren == bracket == brace == 0:
            yield part_start, index
            part_start = index + 1
        if min(paren, bracket, brace) < 0:
            raise source_map.error(
                "unbalanced delimiter while splitting PTX syntax", index
            )
    if paren or bracket or brace or in_string:
        raise source_map.error(
            "unbalanced delimiter while splitting PTX syntax", start
        )
    yield part_start, end


def _skip_ws(text: str, start: int, end: int) -> int:
    while start < end and text[start].isspace():
        start += 1
    return start


def _skip_horizontal_ws(text: str, start: int, end: int) -> int:
    while start < end and text[start] in " \t\r":
        start += 1
    return start


def _trim_range(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end
