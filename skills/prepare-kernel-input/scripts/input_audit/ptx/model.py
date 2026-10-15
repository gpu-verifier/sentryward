"""Syntax-only PTX nodes used by the non-production input audit."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Union


@dataclass(frozen=True, order=True)
class SourcePosition:
    """A source position with one-based line and column."""

    line: int
    column: int
    offset: int


@dataclass(frozen=True)
class SourceSpan:
    """Half-open source range ``[start, end)``."""

    start: SourcePosition
    end: SourcePosition


@dataclass(frozen=True)
class DebugLocation:
    """The active PTX ``.loc`` source mapping for an instruction."""

    file_id: int
    line: int
    column: int
    attributes: str = ""
    raw: str = ""


@dataclass(frozen=True)
class Predicate:
    """PTX instruction guard, e.g. ``@%p1`` or ``@!%p1``."""

    register: str
    negated: bool = False

    @property
    def text(self) -> str:
        return f"@{'!' if self.negated else ''}{self.register}"


@dataclass(frozen=True)
class PtxOperand:
    """One top-level comma-separated operand, preserved as PTX text."""

    text: str
    span: SourceSpan


@dataclass(frozen=True)
class PtxDirective:
    """A PTX directive with its raw argument text."""

    name: str
    arguments: str
    raw: str
    span: SourceSpan
    debug_location: DebugLocation | None = None
    scope: tuple[int, ...] = ()


@dataclass(frozen=True)
class PtxLabel:
    name: str
    raw: str
    span: SourceSpan
    scope: tuple[int, ...] = ()

    @property
    def qualified_name(self) -> str:
        if not self.scope:
            return self.name
        return f"{'/'.join(map(str, self.scope))}::{self.name}"


class FunctionKind(str, Enum):
    ENTRY = "entry"
    FUNC = "func"


class ControlKind(str, Enum):
    """Syntactic control classification used to construct a CFG."""

    NONE = "none"
    DIRECT_BRANCH = "direct_branch"
    INDIRECT_BRANCH = "indirect_branch"
    CALL = "call"
    RETURN = "return"
    EXIT = "exit"
    TRAP = "trap"


class EdgeKind(str, Enum):
    FALLTHROUGH = "fallthrough"
    BRANCH = "branch"
    INDIRECT_BRANCH = "indirect_branch"


@dataclass(frozen=True)
class PtxInstruction:
    """A parsed PTX instruction without an attached execution semantics."""

    index: int
    predicate: Predicate | None
    opcode: str
    modifiers: tuple[str, ...]
    operands: tuple[PtxOperand, ...]
    raw: str
    span: SourceSpan
    debug_location: DebugLocation | None
    control_kind: ControlKind = ControlKind.NONE
    branch_targets: tuple[str, ...] = ()
    scope: tuple[int, ...] = ()

    @property
    def opcode_text(self) -> str:
        return ".".join((self.opcode, *self.modifiers))

    @property
    def is_predicated(self) -> bool:
        return self.predicate is not None

    @property
    def is_opaque(self) -> bool:
        """Whether this instruction has no parser-level control semantics."""

        return self.control_kind is ControlKind.NONE

    @property
    def is_terminator(self) -> bool:
        return self.control_kind in {
            ControlKind.DIRECT_BRANCH,
            ControlKind.INDIRECT_BRANCH,
            ControlKind.RETURN,
            ControlKind.EXIT,
            ControlKind.TRAP,
        }


@dataclass(frozen=True)
class CfgEdge:
    """An edge between basic-block IDs."""

    source: str
    target: str
    kind: EdgeKind
    predicate: Predicate | None = None


@dataclass(frozen=True)
class BasicBlock:
    id: str
    index: int
    labels: tuple[str, ...]
    instructions: tuple[PtxInstruction, ...]
    directives: tuple[PtxDirective, ...]
    successors: tuple[CfgEdge, ...]
    predecessors: tuple[CfgEdge, ...]
    span: SourceSpan
    qualified_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class PtxParameter:
    """A function parameter or return parameter declaration."""

    name: str | None
    storage: str | None
    type_specifiers: tuple[str, ...]
    alignment: int | None
    array_specifier: str | None
    raw: str
    span: SourceSpan


FunctionItem = Union[PtxDirective, PtxLabel, PtxInstruction]


@dataclass(frozen=True)
class PtxFunction:
    name: str
    kind: FunctionKind
    qualifiers: tuple[str, ...]
    parameters: tuple[PtxParameter, ...]
    return_parameters: tuple[PtxParameter, ...]
    attributes: tuple[PtxDirective, ...]
    items: tuple[FunctionItem, ...]
    instructions: tuple[PtxInstruction, ...]
    blocks: tuple[BasicBlock, ...]
    span: SourceSpan
    header_span: SourceSpan
    has_body: bool

    @property
    def block_by_id(self) -> dict[str, BasicBlock]:
        return {block.id: block for block in self.blocks}

    @property
    def label_to_block(self) -> dict[str, BasicBlock]:
        return {
            label: block
            for block in self.blocks
            for label in block.qualified_labels
        }


class PtxParseError(ValueError):
    """An unrecoverable syntax or provenance error."""

    def __init__(
        self,
        message: str,
        *,
        source_name: str,
        position: SourcePosition,
        context: str | None = None,
    ) -> None:
        self.message = message
        self.source_name = source_name
        self.position = position
        self.context = context
        rendered = (
            f"{source_name}:{position.line}:{position.column}: {message}"
        )
        if context:
            rendered += f"\n    {context}"
        super().__init__(rendered)
