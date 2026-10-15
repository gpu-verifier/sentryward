#!/usr/bin/env python3
"""Audit public input bindings without specializing or executing a PTX kernel."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import re
import struct
import sys

if sys.version_info < (3, 10):
    print("Input audit requires Python 3.10+; use build/venv/bin/python in this repository.", file=sys.stderr)
    raise SystemExit(2)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from input_audit.input.model import InputContractError, KernelInput
from input_audit.input.parse import load_kernel_input, parse_kernel_input, validate_kernel_input
from input_audit.ptx import parser as ptx_parser
from input_audit.ptx.model import FunctionKind, PtxParseError

SCHEMA = "spverify.input-binding-audit/v1"
_TYPE = re.compile(r"[busf](8|16|32|64|128)$")
_ADDRESS = re.compile(r"\[\s*(%?[A-Za-z_$][A-Za-z0-9_$]*)\s*(?:([+-])\s*(0[xX][0-9a-fA-F]+|[0-9]+))?\s*\]")


class ParameterAliases:
    """Resolve uniquely defined, dominating parameter-base expressions."""

    def __init__(self, function, by_name):
        self.by_name = by_name
        self.local_parameters = [item for item in function.items
                                 if getattr(item, "name", None) == ".param"]
        self.definitions = {}
        self.blocks = {instruction.index: block.index for block in function.blocks
                       for instruction in block.instructions}
        self.dominators = self._dominators(function)
        self.indirect_control = any(instruction.opcode in {"brx", "brx.idx"}
                                    or instruction.control_kind.value == "indirect_branch"
                                    for instruction in function.instructions)
        for instruction in function.instructions:
            if not instruction.operands:
                continue
            destination = instruction.operands[0].text.strip()
            if destination.startswith("[") or instruction.opcode in {"st", "red", "bra", "brx", "ret", "exit"}:
                continue
            for register in re.findall(r"%[A-Za-z_$][A-Za-z0-9_$]*", destination):
                self.definitions.setdefault(register, []).append(instruction)

    @staticmethod
    def _dominators(function):
        if not function.blocks:
            return []
        blocks = function.blocks
        indices = {block.id: block.index for block in blocks}
        reachable, pending = set(), [0]
        while pending:
            index = pending.pop()
            if index in reachable:
                continue
            reachable.add(index)
            pending.extend(indices[edge.target] for edge in blocks[index].successors if edge.target in indices)
        all_bits = sum(1 << index for index in reachable)
        result = [all_bits if index in reachable else 0 for index in range(len(blocks))]
        result[0] = 1
        changed = True
        while changed:
            changed = False
            for index in sorted(reachable - {0}):
                predecessors = [indices[edge.source] for edge in blocks[index].predecessors
                                if edge.source in indices and indices[edge.source] in reachable]
                bits = all_bits
                for predecessor in predecessors:
                    bits &= result[predecessor]
                bits |= 1 << index
                if bits != result[index]:
                    result[index] = bits
                    changed = True
        return result

    def resolve(self, operand, use, visiting=frozenset()):
        if operand in self.by_name:
            for declaration in self.local_parameters:
                if (use.scope[:len(declaration.scope)] == declaration.scope
                        and re.search(r"(?<![\w.$%])" + re.escape(operand) + r"(?![\w$])",
                                      declaration.arguments)):
                    return None
            return self.by_name[operand], 0, []
        if operand in visiting or not operand.startswith("%"):
            return None
        definitions = self.definitions.get(operand, [])
        if len(definitions) != 1:
            return None
        definition = definitions[0]
        if definition.predicate is not None or definition.scope or use.scope:
            return None
        defining_block, using_block = self.blocks[definition.index], self.blocks[use.index]
        if defining_block == using_block:
            if definition.index >= use.index:
                return None
        elif self.indirect_control or not self.dominators[using_block] & (1 << defining_block):
            return None
        operands = [item.text.strip() for item in definition.operands]
        if not any(item in {"u64", "s64", "b64"} for item in definition.modifiers):
            return None
        if definition.opcode == "cvta" and "param" not in definition.modifiers:
            return None
        if definition.opcode in {"mov", "cvta"} and len(operands) == 2:
            result = self.resolve(operands[1], definition, visiting | {operand})
        elif definition.opcode in {"add", "sub"} and len(operands) == 3:
            if "cc" in definition.modifiers or "sat" in definition.modifiers:
                return None
            try:
                displacement = int(operands[2], 0)
            except ValueError:
                return None
            result = self.resolve(operands[1], definition, visiting | {operand})
            if result is not None:
                index, offset, lines = result
                result = index, offset + (-displacement if definition.opcode == "sub" else displacement), lines
        else:
            return None
        if result is None:
            return None
        index, offset, lines = result
        if not -(1 << 63) <= offset < (1 << 63):
            return None
        return index, offset, lines + [definition.span.start.line]


def _width(type_name):
    if type_name == "pointer":
        return 8
    if type_name in {"bool", "pred"}:
        return 1
    match = _TYPE.fullmatch(type_name.lstrip("."))
    return int(match[1]) // 8 if match else None


def _parameter_size(parameter):
    widths = [_width(item) for item in parameter.type_specifiers]
    widths = [item for item in widths if item is not None]
    if len(widths) != 1:
        return None
    if parameter.array_specifier is None:
        return widths[0]
    extent = parameter.array_specifier[1:-1].strip()
    return widths[0] * int(extent) if extent.isdecimal() else None


def _selected_function(text, entry, source_name):
    """Parse only the selected body, retaining module-relative source positions."""
    source_map = ptx_parser._SourceMap(text, source_name)
    masked = ptx_parser._mask_comments(text, source_map)
    regions = ptx_parser._find_function_regions(masked, source_map)
    entries = []
    for region in regions:
        if region.kind is FunctionKind.ENTRY:
            name = ptx_parser._parse_function_header(text, masked, source_map, region)[0]
            entries.append((name, region))
    if entry is None:
        if len(entries) != 1:
            raise InputContractError("entry_required", "$.entry", "select exactly one PTX entry")
        selected = entries[0][1]
    else:
        matches = [region for name, region in entries if name == entry]
        if len(matches) != 1:
            raise InputContractError("unknown_entry", "$.entry", "entry must identify exactly one PTX kernel")
        selected = matches[0]
    function = ptx_parser._parse_function(text, masked, source_map, selected)
    if not function.has_body:
        raise InputContractError("missing_entry_body", "$.entry", "selected entry has no body")
    return function


def _uncovered(start, end, intervals):
    cursor, gaps = start, []
    for low, high in sorted(intervals):
        if high <= cursor or low >= end:
            continue
        if low > cursor:
            gaps.append([cursor, min(low, end)])
        cursor = max(cursor, min(high, end))
    if cursor < end:
        gaps.append([cursor, end])
    return gaps


def audit_input(ptx_text, config, *, source_name="<memory>"):
    """Return a JSON-ready inventory; coverage is syntactic, not reachability proof."""
    report = {"schema": SCHEMA, "status": "invalid_configuration", "entry": None,
              "parameters": [], "reads": [], "descriptors": [], "issues": [],
              "branch_scope": [], "configured_assumptions": {}, "address_invariants": [],
              "limitations": ["All syntactic ld.param reads are inventoried, including unreachable paths; parameter-base aliases use only unique dominating mov/cvta and constant add/sub definitions.",
                              "Coverage does not validate supplied values against source construction.",
                              "Assumption names and effective defaults are validated by the verifier; this audit only checks object/key/Boolean types and records explicitly configured values.",
                              "Indirect reads, helper calls, and runtime descriptor updates require review.",
                              "Pointer inference from instruction use is not performed; explicit PTX .ptr inputs are recognized."]}
    issues = report["issues"]

    def issue(code, message, *, severity="needs_review", **details):
        issues.append({"code": code, "severity": severity, "message": message, **details})

    try:
        value = validate_kernel_input(config) if isinstance(config, KernelInput) else parse_kernel_input(config)
        report["configured_assumptions"] = dict(value.assumptions)
        report["branch_scope"] = [asdict(item) for item in value.branch_decisions]
        report["address_invariants"] = [dict(item) for item in value.address_invariants]
        function = _selected_function(ptx_text, value.entry, source_name)
    except (InputContractError, PtxParseError, ValueError) as error:
        issue(getattr(error, "code", "invalid_ptx"), str(error), severity="invalid_configuration")
        return report
    report["entry"] = function.name
    parameters = function.parameters
    by_name = {item.name: index for index, item in enumerate(parameters)}
    sizes = [_parameter_size(item) for item in parameters]
    coverage = [[] for _ in parameters]
    bindings = [[] for _ in parameters]
    opaque_maps = [[] for _ in parameters]
    abi_fields = [[] for _ in parameters]
    for index, parameter in enumerate(parameters):
        report["parameters"].append({"index": index, "name": parameter.name,
                                     "size_bytes": sizes[index], "line": parameter.span.start.line})
        if sizes[index] is None:
            issue("unknown_parameter_size", "Cannot determine parameter extent.", parameter=index)
        if ".ptr" in parameter.type_specifiers:
            spaces = {item for item in parameter.type_specifiers
                      if item in {".global", ".shared", ".local", ".const"}}
            if spaces - {".global"} or sizes[index] != 8:
                issue("pointer_declaration_needs_review", "PTX pointer declaration is outside the public global-pointer representation.",
                      parameter=index)
            else:
                coverage[index].append((0, 8))
                bindings[index].append({"kind": "ptx_symbolic_pointer", "path": "PTX parameter declaration",
                                        "byte_range": [0, 8]})

    def resolve(selector, path):
        index = selector if type(selector) is int else by_name.get(selector)
        if index is None or not 0 <= index < len(parameters):
            issue("unknown_parameter", "Binding does not identify a selected-entry parameter.",
                  severity="invalid_configuration", path=path, selector=selector)
            return None
        return index

    def record(item, path, kind, width, *, covers=True):
        index = resolve(item.parameter, path)
        if index is None:
            return None
        end = item.offset + width
        if sizes[index] is not None and end > sizes[index]:
            issue("binding_out_of_bounds", "Binding exceeds the parameter extent.",
                  severity="invalid_configuration", path=path, parameter=index,
                  byte_range=[item.offset, end], parameter_size=sizes[index])
            return None
        type_name = getattr(item, "type", "tensor_map" if kind == "descriptor_pointer" else "pointer")
        parameter = parameters[index]
        if parameter.array_specifier is None and sizes[index] is not None:
            base = next((part[1:] for part in parameter.type_specifiers if _TYPE.fullmatch(part[1:])), "")
            compatible = (base in {"b64", "u64", "s64"} if type_name in {"pointer", "tensor_map"}
                          else base.startswith("b") or ("u8" if type_name == "bool" else type_name) == base)
            if item.offset != 0 or width != sizes[index] or not compatible:
                issue("inconsistent_parameter_type", "Binding type or width contradicts the PTX scalar parameter.",
                      severity="invalid_configuration", path=path, parameter=index)
        if ".ptr" in parameter.type_specifiers and type_name not in {"pointer", "tensor_map"}:
            issue("physical_pointer_value", "PTX pointer inputs require symbolic objects or null, not scalar addresses.",
                  severity="invalid_configuration", path=path, parameter=index)
        for low, high, previous_type, previous_path in abi_fields[index]:
            if low < end and item.offset < high:
                same = (low, high, previous_type) == (item.offset, end, type_name)
                duplicate_declaration = previous_path.startswith("declarations[") and path.startswith("declarations[")
                if not same or duplicate_declaration:
                    issue("conflicting_binding_types", "Overlapping ABI fields have incompatible declarations.",
                          severity="invalid_configuration", path=path, previous_path=previous_path, parameter=index)
        abi_fields[index].append((item.offset, end, type_name, path))
        if covers:
            coverage[index].append((item.offset, end))
            bindings[index].append({"kind": kind, "path": path, "byte_range": [item.offset, end]})
        return index

    scalar_values = {}
    scalar_ranges = {}
    for number, argument in enumerate(value.arguments):
        path = f"arguments[{number}]"
        if argument.type.startswith("f"):
            try:
                encoding = {"f16": "e", "f32": "f", "f64": "d"}[argument.type]
                encoded = struct.pack("<" + encoding, argument.value)
                representable = math.isfinite(struct.unpack("<" + encoding, encoded)[0])
            except (OverflowError, struct.error):
                representable = False
            if not representable:
                issue("scalar_value_out_of_range", "Floating value is outside its declared scalar type.",
                      severity="invalid_configuration", path=path)
        index = record(argument, path, "scalar_value", _width(argument.type))
        key = (index, argument.offset)
        signature = (argument.type, argument.value)
        if index is not None and key in scalar_values:
            issue("conflicting_arguments", "Multiple values bind the same parameter offset.",
                  severity="invalid_configuration", path=path, parameter=index, offset=argument.offset)
        if index is not None:
            end = argument.offset + _width(argument.type)
            for low, high in scalar_ranges.get(index, []):
                if low < end and argument.offset < high and low != argument.offset:
                    issue("conflicting_arguments", "Scalar value fields overlap.", severity="invalid_configuration",
                          path=path, parameter=index, byte_range=[argument.offset, end])
            scalar_ranges.setdefault(index, []).append((argument.offset, end))
        scalar_values[key] = signature
    for number, declaration in enumerate(value.declarations):
        path = f"declarations[{number}]"
        if declaration.type == "tensor_map":
            selected_index = resolve(declaration.parameter, path)
            width = 8 if selected_index is not None and parameters[selected_index].array_specifier is None and sizes[selected_index] == 8 else 128
            index = record(declaration, path, "descriptor_pointer", width, covers=width == 8)
            if index is not None:
                if width == 128:
                    opaque_maps[index].append((declaration.offset, declaration.offset + width))
        else:
            index = record(declaration, path, "symbolic_pointer", _width(declaration.type),
                           covers=declaration.type == "pointer")
            if index is not None and declaration.type == "pointer":
                for argument in value.arguments:
                    arg_index = argument.parameter if type(argument.parameter) is int else by_name.get(argument.parameter)
                    if arg_index != index:
                        continue
                    end = argument.offset + _width(argument.type)
                    if argument.offset < declaration.offset + 8 and declaration.offset < end:
                        if argument.type != "pointer" or declaration.object is not None:
                            issue("conflicting_pointer_binding", "Scalar/null value conflicts with a symbolic pointer declaration.",
                                  severity="invalid_configuration", path=path, parameter=index)
    for number, array in enumerate(value.shape_arrays):
        index = record(array, f"shape_arrays[{number}]", "shape_array_pointer", 8)
        if (index, array.offset) in scalar_values:
            issue("conflicting_shape_array_binding", "A supplied shape array cannot use a null or scalar pointer value.",
                  severity="invalid_configuration", parameter=index, offset=array.offset)

    declarations = {(resolve(item.parameter, "declarations"), item.offset): item
                    for item in value.declarations if item.type in {"pointer", "tensor_map"}}
    for number, descriptor in enumerate(value.tma_descriptors):
        path = f"tma_descriptors[{number}]"
        index = resolve(descriptor.parameter, path)
        if index is None:
            continue
        width = sizes[index] if parameters[index].array_specifier is None else 128
        if width is None:
            issue("descriptor_extent_needs_review", "Cannot determine descriptor parameter extent.",
                  parameter=index, offset=descriptor.offset)
            continue
        record(descriptor, path, "descriptor_pointer", width, covers=width == 8)
        if width == 128:
            opaque_maps[index].append((descriptor.offset, descriptor.offset + width))
        fields = descriptor.fields
        required = {"object", "element_type", "global_shape", "box_shape", "element_strides",
                    "swizzle", "interleave", "oob_fill"}
        missing = sorted(required - fields.keys())
        if not ({"coordinate_strides_bytes", "global_strides_bytes"} & fields.keys()):
            missing.append("coordinate_strides_bytes_or_global_strides_bytes")
        report["descriptors"].append({"parameter": index, "offset": descriptor.offset,
                                      "storage": "pointer" if width == 8 else "inline",
                                      "fields_present": sorted(fields), "missing_fields": missing})
        if missing:
            issue("partial_descriptor", "Descriptor layout fields need source review.",
                  parameter=index, offset=descriptor.offset, fields=missing)

    descriptor_keys = {(resolve(item.parameter, "tma_descriptors"), item.offset)
                       for item in value.tma_descriptors}
    for (index, offset), declaration in declarations.items():
        if declaration.type == "tensor_map" and (index, offset) not in descriptor_keys:
            issue("descriptor_not_supplied", "Declared tensor map has no structured descriptor.",
                  parameter=index, offset=offset)

    aliases = ParameterAliases(function, by_name)
    tma_lines, updates = [], []
    for instruction in function.instructions:
        line = instruction.span.start.line
        if instruction.opcode == "call":
            issue("helper_call", "Helper parameter flow is outside the direct-read inventory.",
                  line=line, instruction=instruction.raw)
        if "tensor" in instruction.modifiers:
            tma_lines.append(line)
        if instruction.opcode == "tensormap":
            updates.append(line)
        if instruction.opcode != "ld" or "param" not in instruction.modifiers:
            continue
        widths = [_width(modifier) for modifier in instruction.modifiers]
        widths = [width for width in widths if width is not None]
        vectors = [int(item[1:]) for item in instruction.modifiers if re.fullmatch(r"v[248]", item)]
        width = widths[0] * (vectors[0] if vectors else 1) if len(widths) == 1 else None
        operand = instruction.operands[-1].text if instruction.operands else ""
        match = _ADDRESS.fullmatch(operand)
        read = {"line": line, "instruction": instruction.raw,
                "width_bytes": width, "status": "needs_review"}
        report["reads"].append(read)
        resolved = aliases.resolve(match[1], instruction) if match else None
        if width is None or match is None or resolved is None:
            issue("indirect_or_unresolved_parameter_read", "Read needs parameter-address or helper analysis.",
                  line=line, instruction=instruction.raw)
            continue
        index, base_offset, alias_lines = resolved
        offset = int(match[3], 16 if match[3].lower().startswith("0x") else 10) if match[3] else 0
        offset *= -1 if match[2] == "-" else 1
        offset += base_offset
        end = offset + width
        read.update(parameter=index, parameter_name=parameters[index].name, byte_range=[offset, end],
                    address_kind="parameter_alias" if alias_lines else "direct_parameter",
                    alias_definition_lines=alias_lines)
        if offset < 0 or (sizes[index] is not None and end > sizes[index]):
            read["status"] = "invalid_configuration"
            issue("read_out_of_bounds", "Direct read exceeds the parameter extent.",
                  severity="invalid_configuration", line=line, parameter=index, byte_range=[offset, end])
            continue
        read["bindings"] = [item for item in bindings[index]
                            if item["byte_range"][0] < end and offset < item["byte_range"][1]]
        read["uncovered_ranges"] = _uncovered(offset, end, coverage[index])
        if any(low < end and offset < high for low, high in opaque_maps[index]):
            issue("opaque_descriptor_bytes", "Structured descriptor fields do not supply opaque descriptor bytes.",
                  line=line, parameter=index, byte_range=[offset, end])
        elif read["uncovered_ranges"]:
            read["status"] = "unbound"
            issue("unbound_parameter_bytes", "No value or symbolic pointer supplies these directly read bytes.",
                  severity="binding_gaps", line=line, parameter=index, ranges=read["uncovered_ranges"])
        else:
            read["status"] = "covered"
    report["tma_instruction_lines"] = tma_lines
    report["descriptor_update_lines"] = updates
    if tma_lines and not value.tma_descriptors:
        issue("tma_binding_needs_review", "TMA instructions are present without structured launch descriptors.",
              lines=tma_lines)
    if updates:
        issue("runtime_descriptor_updates", "Descriptor contents can change in device code; review the updated fields.",
              lines=updates)
    branch_lines = {item.span.start.line for item in function.instructions
                    if item.opcode == "bra" and item.predicate is not None}
    for decision in value.branch_decisions:
        if decision.line not in branch_lines:
            issue("invalid_branch_location", "Branch choice does not identify a conditional branch in this entry.",
                  severity="invalid_configuration", line=decision.line)
    severities = {item["severity"] for item in issues}
    report["status"] = next((name for name in ("invalid_configuration", "binding_gaps", "needs_review")
                             if name in severities), "coverage_complete")
    report["summary"] = {"direct_read_count": len(report["reads"]),
                         "covered_read_count": sum(item["status"] == "covered" for item in report["reads"]),
                         "unbound_read_count": sum(item["status"] == "unbound" for item in report["reads"]),
                         "review_read_count": sum(item["status"] == "needs_review" for item in report["reads"]),
                         "descriptor_count": len(report["descriptors"])}
    return report


def audit_files(ptx_path, input_path):
    """Load files, rejecting duplicate JSON keys, and return the audit report."""
    try:
        config = load_kernel_input(input_path)
        text = Path(ptx_path).read_text(encoding="utf-8")
    except (InputContractError, OSError) as error:
        return {"schema": SCHEMA, "status": "invalid_configuration", "issues": [
            {"code": getattr(error, "code", "file_error"), "severity": "invalid_configuration", "message": str(error)}]}
    return audit_input(text, config, source_name=str(ptx_path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ptx", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; choose a new audit destination")
    report = audit_files(args.ptx, args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(report["status"])
    return {"coverage_complete": 0, "invalid_configuration": 2}.get(report["status"], 1)


if __name__ == "__main__":
    raise SystemExit(main())
