"""Public kernel-input records for the static, non-production audit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


SCHEMA = "spverify.kernel-input/v1"
SCALAR_TYPES = frozenset(
    [f"{kind}{width}" for kind in ("u", "s", "b") for width in (8, 16, 32, 64)]
    + ["f16", "f32", "f64", "bool"]
)


class InputContractError(ValueError):
    def __init__(self, code: str, path: str, message: str, *,
                 status: str = "invalid_configuration", location: Mapping[str, Any] | None = None):
        super().__init__(f"{path}: {message}")
        self.status = status
        self.code = code
        self.path = path
        self.location = dict(location) if location is not None else None

    def to_dict(self) -> dict[str, Any]:
        result = {"status": self.status, "reason": self.code, "path": self.path}
        if self.location is not None:
            result["location"] = self.location
        return result


@dataclass(frozen=True)
class Launch:
    grid_dim: tuple[int, int, int]
    block_dim: tuple[int, int, int]
    cluster_dim: tuple[int, int, int] | None = None
    dynamic_shared_bytes: int = 0


@dataclass(frozen=True)
class ScalarArgument:
    parameter: int | str
    type: str
    value: int | float | bool | None
    offset: int = 0


@dataclass(frozen=True)
class ParameterDeclaration:
    parameter: int | str
    type: str
    offset: int = 0
    space: str = "global"
    object: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class ShapeArray:
    parameter: int | str
    shapes: tuple[tuple[int, ...], ...]
    element_type: str = "s32"
    offset: int = 0


@dataclass(frozen=True)
class TmaDescriptorInput:
    parameter: int | str
    fields: Mapping[str, Any] = field(default_factory=dict)
    offset: int = 0


@dataclass(frozen=True)
class BranchDecision:
    line: int
    taken: bool | str
    note: str | None = None
    seed: int | None = None


@dataclass(frozen=True)
class KernelInput:
    launch: Launch
    entry: str | None = None
    arguments: tuple[ScalarArgument, ...] = ()
    declarations: tuple[ParameterDeclaration, ...] = ()
    shape_arrays: tuple[ShapeArray, ...] = ()
    tma_descriptors: tuple[TmaDescriptorInput, ...] = ()
    branch_decisions: tuple[BranchDecision, ...] = ()
    assumptions: Mapping[str, bool] = field(default_factory=dict)
    # Address-invariant environment for the symbolic address model; the native
    # verifier validates each constraint.
    address_invariants: tuple[Mapping[str, Any], ...] = ()
    # Representative control flow loop bound; None leaves the frontend default.
    data_branch_loop_iterations: int | None = None
