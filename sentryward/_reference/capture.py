"""Capture fixed-shape, functional PyTorch references on the meta device."""

import inspect
import torch

from .graph import strides
from .lowering import Lowering, Unsupported, tensor_type


class FunctionModule(torch.nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, *args):
        return self.function(*args)


def extract(reference, config):
    if config.get("schema") != "spverify.pytorch-reference.v1":
        raise ValueError("unsupported PyTorch reference config schema")
    arguments = config["arguments"]
    if set(config) - {"schema", "arguments", "outputs"}:
        raise ValueError("unknown reference config field")
    function = reference.forward if isinstance(reference, torch.nn.Module) else reference
    names = list(inspect.signature(function).parameters)
    if [a["name"] for a in arguments] != names:
        raise ValueError("arguments must match the reference signature in order")
    values = []
    for argument in arguments:
        allowed = {"name", "value"} if "value" in argument else {"name", "shape", "dtype", "allocation", "offset"}
        if set(argument) - allowed:
            raise ValueError("unknown argument binding field")
        if "value" in argument:
            if not isinstance(argument["value"], (int, float, bool)):
                raise ValueError("only scalar specialized parameters are supported")
            values.append(argument["value"])
            continue
        shape = argument["shape"]
        if any(type(x) is not int or x <= 0 for x in shape):
            raise ValueError("tensor shapes must contain positive integers")
        if not isinstance(argument.get("allocation"), str) or not argument["allocation"]:
            raise ValueError("every tensor parameter requires an allocation binding")
        value = torch.empty(shape, dtype=getattr(torch, argument["dtype"]), device="meta")
        tensor_type(value)
        values.append(value)
    module = reference if isinstance(reference, torch.nn.Module) else FunctionModule(reference)
    if list(module.parameters()) or list(module.buffers()):
        raise Unsupported("pass module parameters and buffers as explicit reference arguments")
    try:
        exported = torch.export.export(module, tuple(values), strict=True)
    except Exception as error:
        raise Unsupported(f"PyTorch capture failed: {error}") from error
    signature = exported.graph_signature
    if any(s.kind.name != "USER_INPUT" for s in signature.input_specs):
        raise Unsupported("lifted constants or implicit parameters are unsupported")
    if any(s.kind.name != "USER_OUTPUT" for s in signature.output_specs):
        raise Unsupported("reference must not mutate inputs or module state")
    for node in exported.graph.nodes:
        if node.op == "call_function":
            schema = getattr(node.target, "_schema", None)
            if schema is None or schema.is_mutable:
                raise Unsupported("reference must contain functional ATen operators")
    placeholders = [n for n in exported.graph.nodes if n.op == "placeholder"]
    if len(placeholders) != len(arguments):
        raise Unsupported("unsupported structured input signature")
    bindings = {}
    for node, argument, value in zip(placeholders, arguments, values):
        if not isinstance(value, torch.Tensor):
            continue
        binding = dict(argument)
        if "strides_bytes" in binding:
            raise ValueError("reference tensors use logical contiguous layout; put PTX layouts in equivalence mapping")
        if type(binding.get("offset", 0)) is not int or binding.get("offset", 0) < 0:
            raise ValueError("input offset must be nonnegative")
        binding["strides_bytes"] = strides(value.shape, value.element_size())
        bindings[node.name] = binding
    output = next(n for n in exported.graph.nodes if n.op == "output")
    artifact = Lowering(bindings).outputs(list(output.args[0]), config["outputs"])
    return artifact, dict(status="expressions_ready", torch_version=torch.__version__,
                          numerical_semantics="ideal_real", node_count=len(artifact["nodes"]),
                          output_family_count=len(artifact["output_families"]),
                          specialized_parameters={a["name"]: a["value"] for a in arguments if "value" in a},
                          exported_graph=str(exported.graph))
