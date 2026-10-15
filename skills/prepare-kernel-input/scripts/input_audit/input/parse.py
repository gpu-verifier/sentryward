"""Validate the public JSON input whitelist."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from .model import (SCHEMA, SCALAR_TYPES, BranchDecision, InputContractError, KernelInput, Launch,
                    ParameterDeclaration, ScalarArgument, ShapeArray, TmaDescriptorInput)


def validate_assumptions(value):
    """Check syntax only; semantic names and defaults belong to the verifier."""
    if not isinstance(value, Mapping):
        raise ValueError("assumptions must be an object")
    if any(not isinstance(name, str) or not name for name in value):
        raise ValueError("assumption names must be nonempty strings")
    if any(type(enabled) is not bool for enabled in value.values()):
        raise ValueError("assumption flags must be Boolean")
    return dict(value)


def _object(value: Any, keys: set[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InputContractError("invalid_type", path, "expected an object")
    for key in value:
        if key not in keys:
            raise InputContractError("unknown_field", f"{path}.{key}", "field is not accepted")
    return value


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise InputContractError("invalid_value", path, f"expected an integer >= {minimum}")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise InputContractError("invalid_type", path, "expected a nonempty string")
    return value


def _selector(value: Any, path: str) -> int | str:
    return _integer(value, path) if type(value) is int else _string(value, path)


def _dimensions(value: Any, path: str) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise InputContractError("invalid_value", path, "expected [x, y, z]")
    return tuple(_integer(item, f"{path}[{index}]", minimum=1)
                 for index, item in enumerate(value))


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise InputContractError("invalid_type", path, "expected an array")
    return value


def _scalar(value: Any, type_name: str, path: str) -> int | float | bool | None:
    if type_name == "pointer":
        if value is not None:
            raise InputContractError("physical_pointer_value", path, "only null is accepted as a pointer value")
    elif type_name == "bool":
        if type(value) is not bool:
            raise InputContractError("invalid_type", path, "expected a Boolean")
    elif type_name.startswith("f"):
        try:
            finite = type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise InputContractError("invalid_value", path, "expected a finite number")
    else:
        width = int(type_name[1:])
        lower = -(1 << (width - 1)) if type_name.startswith("s") else 0
        upper = (1 << (width - 1)) if type_name.startswith("s") else (1 << width)
        if type(value) is not int or not lower <= value < upper:
            raise InputContractError("invalid_value", path, f"value is outside {type_name}")
    return value


def parse_kernel_input(value: Mapping[str, Any]) -> KernelInput:
    root = _object(value, {"schema", "entry", "launch", "arguments", "declarations", "shape_arrays", "tma_descriptors", "branch_decisions", "assumptions", "address_invariants", "data_branch_loop_iterations"}, "$")
    try:
        assumptions = validate_assumptions(root.get("assumptions", {}))
    except ValueError as error:
        raise InputContractError("invalid_assumptions", "$.assumptions", str(error)) from error
    address_invariants = _address_invariants(root.get("address_invariants", []))
    loop_iterations = (_integer(root["data_branch_loop_iterations"], "$.data_branch_loop_iterations", minimum=1)
                       if "data_branch_loop_iterations" in root else None)
    if root.get("schema") != SCHEMA:
        raise InputContractError("invalid_schema", "$.schema", f"expected {SCHEMA}")
    launch = _object(root.get("launch"),
                     {"grid_dim", "block_dim", "cluster_dim", "dynamic_shared_bytes"}, "$.launch")
    for required in ("grid_dim", "block_dim"):
        if required not in launch:
            raise InputContractError("missing_launch_dimension", f"$.launch.{required}",
                                     "launch dimensions are required")
    parsed_launch = Launch(
        _dimensions(launch["grid_dim"], "$.launch.grid_dim"),
        _dimensions(launch["block_dim"], "$.launch.block_dim"),
        _dimensions(launch["cluster_dim"], "$.launch.cluster_dim") if "cluster_dim" in launch else None,
        _integer(launch.get("dynamic_shared_bytes", 0), "$.launch.dynamic_shared_bytes"),
    )
    entry = _string(root["entry"], "$.entry") if "entry" in root else None
    declarations = []
    for index, item in enumerate(_list(root.get("declarations", []), "$.declarations")):
        path = f"$.declarations[{index}]"
        item = _object(item, {"parameter", "offset", "type", "space", "object", "name"}, path)
        type_name = item.get("type")
        if not isinstance(type_name, str) or type_name not in SCALAR_TYPES | {"pointer", "tensor_map"}:
            raise InputContractError("invalid_declaration_type", path + ".type", "unknown parameter type")
        if "space" in item and (type_name != "pointer" or item["space"] != "global"):
            raise InputContractError("invalid_value", path + ".space", "only global input pointers are supported")
        if "object" in item and type_name not in {"pointer", "tensor_map"}:
            raise InputContractError("invalid_value", path + ".object", "scalar declarations do not name memory objects")
        declarations.append(ParameterDeclaration(
            _selector(item.get("parameter"), path + ".parameter"), type_name,
            _integer(item.get("offset", 0), path + ".offset"),
            item.get("space", "global"),
            _string(item["object"], path + ".object") if "object" in item else None,
            _string(item["name"], path + ".name") if "name" in item else None,
        ))
    from .bindings import expand_named_arguments
    arguments = []
    for index, item in enumerate(expand_named_arguments(
            _list(root.get("arguments", []), "$.arguments"), declarations)):
        path = f"$.arguments[{index}]"
        item = _object(item, {"parameter", "offset", "type", "value"}, path)
        type_name = item.get("type")
        if not isinstance(type_name, str) or type_name not in SCALAR_TYPES | {"pointer"}:
            raise InputContractError("invalid_scalar_type", path + ".type", "expected a scalar type or null pointer")
        if "value" not in item:
            raise InputContractError("missing_argument_value", path + ".value", "value is required")
        arguments.append(ScalarArgument(
            _selector(item.get("parameter"), path + ".parameter"), type_name,
            _scalar(item["value"], type_name, path + ".value"),
            _integer(item.get("offset", 0), path + ".offset"),
        ))
    shape_arrays = []
    for index, item in enumerate(_list(root.get("shape_arrays", []), "$.shape_arrays")):
        path = f"$.shape_arrays[{index}]"
        item = _object(item, {"parameter", "offset", "element_type", "shapes"}, path)
        element_type = item.get("element_type", "s32")
        if element_type not in ("s32", "u32", "s64", "u64"):
            raise InputContractError("invalid_shape_type", path + ".element_type", "expected a 32- or 64-bit integer type")
        shapes = []
        for row, shape in enumerate(_list(item.get("shapes"), path + ".shapes")):
            row_path = f"{path}.shapes[{row}]"
            dimensions = tuple(_scalar(_integer(dim, row_path), element_type, row_path)
                               for dim in _list(shape, row_path))
            if not dimensions or (shapes and len(dimensions) != len(shapes[0])):
                raise InputContractError("invalid_shape_array", row_path, "expected nonempty shapes of equal rank")
            shapes.append(dimensions)
        if not shapes:
            raise InputContractError("invalid_shape_array", path + ".shapes", "expected at least one shape")
        shape_arrays.append(ShapeArray(
            _selector(item.get("parameter"), path + ".parameter"), tuple(shapes), element_type,
            _integer(item.get("offset", 0), path + ".offset")))
    from .tma import parse_descriptors
    decisions = []
    lines = set()
    for index, item in enumerate(_list(root.get("branch_decisions", []), "$.branch_decisions")):
        path = f"$.branch_decisions[{index}]"
        item = _object(item, {"line", "taken", "note", "seed"}, path)
        line = _integer(item.get("line"), path + ".line", minimum=1)
        taken = item.get("taken")
        seed = None
        if taken == "random":
            seed = _integer(item.get("seed"), path + ".seed")
        else:
            taken = _scalar(taken, "bool", path + ".taken")
            if "seed" in item:
                raise InputContractError("invalid_branch_seed", path + ".seed", "seed requires taken=random")
        if line in lines:
            raise InputContractError("duplicate_branch_decision", path, "branch line is already configured")
        lines.add(line)
        note = _string(item["note"], path + ".note") if "note" in item else None
        decisions.append(BranchDecision(line, taken, note, seed))
    return KernelInput(parsed_launch, entry, tuple(arguments), tuple(declarations), tuple(shape_arrays),
                       parse_descriptors(root.get("tma_descriptors", [])), tuple(decisions), assumptions,
                       address_invariants, loop_iterations)


def _address_invariants(value: Any) -> tuple[Mapping[str, Any], ...]:
    items = _list(value, "$.address_invariants")
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise InputContractError("invalid_address_invariant", f"$.address_invariants[{index}]",
                                     "expected an object")
    return tuple(dict(item) for item in items)


def load_kernel_input(path: str | Path) -> KernelInput:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise InputContractError("duplicate_field", "$", f"duplicate JSON field {key!r}")
            result[key] = value
        return result
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs)
    except (OSError, json.JSONDecodeError) as error:
        raise InputContractError("invalid_input_file", "$", str(error)) from error
    return parse_kernel_input(value)


def validate_kernel_input(value: KernelInput) -> KernelInput:
    """Apply the same contract to directly constructed Python inputs."""
    if not isinstance(value, KernelInput) or not isinstance(value.launch, Launch):
        raise InputContractError("invalid_type", "$", "expected a KernelInput with a Launch")
    def dims(value):
        return list(value) if isinstance(value, tuple) else value
    raw = {"schema": SCHEMA, "launch": {
        "grid_dim": dims(value.launch.grid_dim), "block_dim": dims(value.launch.block_dim),
        "dynamic_shared_bytes": value.launch.dynamic_shared_bytes,
    }, "arguments": [], "declarations": []}
    raw["assumptions"] = validate_assumptions(value.assumptions)
    if value.address_invariants:
        raw["address_invariants"] = [dict(item) if isinstance(item, Mapping) else item
                                     for item in value.address_invariants]
    if value.data_branch_loop_iterations is not None:
        raw["data_branch_loop_iterations"] = value.data_branch_loop_iterations
    if value.entry is not None:
        raw["entry"] = value.entry
    if value.launch.cluster_dim is not None:
        raw["launch"]["cluster_dim"] = dims(value.launch.cluster_dim)
    for field_name, expected in (("arguments", ScalarArgument), ("declarations", ParameterDeclaration)):
        items = getattr(value, field_name)
        if not isinstance(items, (tuple, list)):
            raise InputContractError("invalid_type", f"$.{field_name}", "expected parameter records")
        for index, item in enumerate(items):
            if not isinstance(item, expected):
                raise InputContractError("invalid_type", f"$.{field_name}[{index}]", "unexpected parameter record")
            record = {"parameter": item.parameter, "offset": item.offset, "type": item.type}
            if field_name == "arguments":
                record["value"] = item.value
            else:
                if item.type == "pointer" or item.space != "global":
                    record["space"] = item.space
                if item.object is not None:
                    record["object"] = item.object
                if item.name is not None:
                    record["name"] = item.name
            raw[field_name].append(record)
    raw["shape_arrays"] = []
    for index, item in enumerate(value.shape_arrays):
        if not isinstance(item, ShapeArray):
            raise InputContractError("invalid_type", f"$.shape_arrays[{index}]", "expected a ShapeArray")
        raw["shape_arrays"].append({"parameter": item.parameter, "offset": item.offset,
                                    "element_type": item.element_type,
                                    "shapes": [list(shape) for shape in item.shapes]})
    raw["tma_descriptors"] = []
    for index, item in enumerate(value.tma_descriptors):
        if not isinstance(item, TmaDescriptorInput) or not isinstance(item.fields, Mapping):
            raise InputContractError("invalid_type", f"$.tma_descriptors[{index}]", "expected a TmaDescriptorInput")
        if {"parameter", "offset"} & item.fields.keys():
            raise InputContractError("invalid_tma_descriptor", f"$.tma_descriptors[{index}]", "binding keys cannot be layout fields")
        raw["tma_descriptors"].append({"parameter": item.parameter, "offset": item.offset, **item.fields})
    raw["branch_decisions"] = []
    if not isinstance(value.branch_decisions, (tuple, list)):
        raise InputContractError("invalid_type", "$.branch_decisions", "expected branch decisions")
    for index, item in enumerate(value.branch_decisions):
        if not isinstance(item, BranchDecision):
            raise InputContractError("invalid_type", f"$.branch_decisions[{index}]", "expected a BranchDecision")
        raw["branch_decisions"].append({"line": item.line, "taken": item.taken})
        if item.seed is not None:
            raw["branch_decisions"][-1]["seed"] = item.seed
        if item.note is not None:
            raw["branch_decisions"][-1]["note"] = item.note
    return parse_kernel_input(raw)
