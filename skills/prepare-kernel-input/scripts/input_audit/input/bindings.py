"""Expand named host scalar values through declared parameter ABI fields."""

from collections.abc import Mapping

from .model import InputContractError


def expand_named_arguments(arguments, declarations):
    bindings = {}
    for declaration in declarations:
        if declaration.name is not None:
            bindings.setdefault(declaration.name, []).append(declaration)
    result = []
    seen = set()
    for index, argument in enumerate(arguments):
        path = f"$.arguments[{index}]"
        if not isinstance(argument, Mapping) or "field" not in argument:
            result.append(argument)
            continue
        if set(argument) != {"field", "value"}:
            raise InputContractError("invalid_named_argument", path, "expected field and value")
        name = argument["field"]
        if not isinstance(name, str) or not name or name not in bindings:
            raise InputContractError("unknown_argument_field", path + ".field", "no matching ABI declaration")
        if name in seen:
            raise InputContractError("duplicate_argument", path, "named field already has a value")
        seen.add(name)
        for declaration in bindings[name]:
            if declaration.type == "tensor_map":
                raise InputContractError("invalid_named_argument", path, "descriptor contents are not accepted")
            result.append({"parameter": declaration.parameter, "offset": declaration.offset,
                           "type": declaration.type, "value": argument["value"]})
    return result
