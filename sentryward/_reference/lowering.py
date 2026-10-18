"""Index-based ATen lowering without enumerating tensor elements."""

import math
import torch

from .graph import Graph, strides


class Unsupported(ValueError):
    pass


TYPES = {torch.float16: "f16", torch.bfloat16: "bf16", torch.float32: "f32",
         torch.float64: "f64", torch.float8_e4m3fn: "e4m3", torch.float8_e5m2: "e5m2",
         torch.uint8: "u8", torch.int8: "s8", torch.int16: "s16", torch.int32: "s32", torch.int64: "s64",
         torch.bool: "pred"}


def tensor_type(value):
    if not isinstance(value, torch.Tensor) or value.dtype not in TYPES:
        raise Unsupported("tensor numeric type is unsupported")
    return TYPES[value.dtype]


class Lowering:
    def __init__(self, bindings):
        self.bindings = bindings
        self.graph = Graph()
        self.cache = {}

    def shape(self, node):
        return tuple(int(x) for x in node.meta["val"].shape)

    def value(self, node, indices, depth):
        def key_of(request):
            n, coordinates, level = request
            return n, tuple(tuple(sorted(x.items())) for x in coordinates), level

        request = node, indices, depth
        root = key_of(request)
        if root in self.cache:
            return self.cache[root]
        # Suspended generators retain their local work. A dependency miss does
        # not restart an operator (especially stack/cat), nor recurse through
        # Python frames. Each distinct (node, coordinates, depth) is built once;
        # work is O(U+E), excluding coordinate-key and graph-interning costs.
        pending = [(root, self._value(*request))]
        active = {root}
        result = None
        while pending:
            key, continuation = pending[-1]
            try:
                request = continuation.send(result)
            except StopIteration as done:
                result = done.value
                self.cache[key] = result
                active.remove(key)
                pending.pop()
                continue
            child = key_of(request)
            if child in self.cache:
                result = self.cache[child]
            else:
                if child in active:
                    raise Unsupported("cyclic reference expression dependency")
                active.add(child)
                pending.append((child, self._value(*request)))
                result = None
        return self.cache[root]

    def _value(self, node, indices, depth):
        g = self.graph
        dtype = tensor_type(node.meta["val"])
        shape = self.shape(node)
        if len(indices) != len(shape):
            raise Unsupported("index rank mismatch")

        def operand(arg, target=indices):
            if isinstance(arg, (int, float, bool)):
                return g.constant(arg, dtype)
            if not isinstance(arg, torch.fx.Node):
                raise Unsupported(f"unsupported operand {arg!r}")
            source_shape = self.shape(arg)
            if len(source_shape) > len(target):
                raise Unsupported("invalid broadcast rank")
            selected = target[len(target)-len(source_shape):] if source_shape else []
            selected = [g.constant(0) if n == 1 else i for n, i in zip(source_shape, selected)]
            return (yield (arg, selected, depth))

        if node.op == "placeholder":
            binding = self.bindings[node.name]
            offset = g.constant(binding.get("offset", 0))
            for index, stride in zip(indices, binding["strides_bytes"]):
                offset = g.op("Add", offset, g.op("Multiply", index, g.constant(stride)))
            return g.node("InitialCell", dtype, binding["allocation"], (offset,))
        if node.op != "call_function":
            raise Unsupported(f"unsupported graph node {node.op}")
        op = str(node.target)
        args = node.args
        if op == "aten.view.dtype":
            old = tensor_type(args[0].meta["val"])
            widths = {"f16": 16, "bf16": 16, "f32": 32, "f64": 64,
                      "e4m3": 8, "e5m2": 8, "u8": 8, "s32": 32, "s64": 64}
            if old not in widths or dtype not in widths or widths[old] != widths[dtype]:
                raise Unsupported("dtype view with changed element width requires explicit byte indexing")
            if self.shape(args[0]) != self.shape(node):
                raise Unsupported("equal-width dtype view changed logical shape")
            value = dict((yield from operand(args[0])))
            value["conversion"] = "; ".join(filter(None, [value.get("conversion"),
                                                        f"Bitcast({old} -> {dtype})"]))
            value["converted_type"] = dtype
            return value
        if op == "aten.signbit.default":
            old = tensor_type(args[0].meta["val"])
            widths = {"f16": 16, "bf16": 16, "f32": 32, "f64": 64,
                      "e4m3": 8, "e5m2": 8}
            value = yield from operand(args[0])
            if old in widths:
                width = widths[old]
                storage = f"u{width}"
                bits = dict(value)
                bits["conversion"] = "; ".join(filter(None, [value.get("conversion"),
                                                            f"Bitcast({old} -> {storage})"]))
                bits["converted_type"] = storage
                sign = g.op("BitAnd", bits, g.constant(1 << (width-1), storage), dtype=storage)
                return g.op("NotEqual", sign, g.constant(0, storage), dtype="pred")
            if old in {"s32", "s64"}:
                return g.op("Less", value, g.constant(0, old), dtype="pred")
            if old in {"u8", "pred"}:
                return g.constant(False, "pred")
            raise Unsupported("signbit input type unsupported")
        if op in {"aten.zeros.default", "aten.ones.default",
                  "aten.zeros_like.default", "aten.ones_like.default",
                  "aten.new_zeros.default", "aten.new_ones.default"}:
            # The exported shape/type already specializes this factory. Every
            # coordinate has the same value: no tensor-sized materialization.
            fill = int("ones" in op)
            return g.constant(bool(fill) if dtype == "pred" else fill, dtype)
        binary = {"aten.add.Tensor": "Add", "aten.sub.Tensor": "Subtract",
                  "aten.mul.Tensor": "Multiply", "aten.div.Tensor": "Divide",
                  "aten.floor_divide.default": "FloorDivide", "aten.remainder.Scalar": "Modulo"}
        unary = {"aten.neg.default": "Negate", "aten.rsqrt.default": "Rsqrt",
                 "aten.sqrt.default": "Sqrt", "aten.exp.default": "Exp",
                 "aten.exp2.default": "Exp2", "aten.log.default": "Log",
                 "aten.cos.default": "Cos", "aten.sin.default": "Sin",
                 "aten.abs.default": "Absolute"}
        comparisons = {"eq": "Equal", "ne": "NotEqual", "lt": "Less", "le": "LessEqual",
                       "gt": "Greater", "ge": "GreaterEqual"}
        if op in {f"aten.{name}.{overload}" for name in comparisons for overload in ("Tensor", "Scalar")}:
            source_type = tensor_type(args[0].meta["val"])
            right = (g.constant(args[1], source_type) if not isinstance(args[1], torch.fx.Node)
                     else (yield from operand(args[1])))
            return g.op(comparisons[op.split(".")[1]], (yield from operand(args[0])), right, dtype="pred")
        if op in {"aten.where.self", "aten.where.ScalarOther", "aten.where.ScalarSelf", "aten.where.Scalar"}:
            return g.op("Select", (yield from operand(args[0])), (yield from operand(args[1])),
                        (yield from operand(args[2])), dtype=dtype)
        if op in {"aten.bitwise_and.Tensor", "aten.bitwise_or.Tensor", "aten.__and__.Tensor", "aten.__or__.Tensor", "aten.logical_and.default", "aten.logical_or.default"}:
            name = "BitAnd" if "and" in op else "BitOr"
            return g.op(name, (yield from operand(args[0])), (yield from operand(args[1])), dtype=dtype)
        if op in {"aten.arange.default", "aten.arange.start", "aten.arange.start_step"}:
            start = args[0] if len(args) > 1 else 0
            step = args[2] if len(args) > 2 else 1
            if not isinstance(start, (int, float)) or not isinstance(step, (int, float)):
                raise Unsupported("arange bounds must be specialized scalars")
            index = dict(indices[0], conversion=f"Convert(u64 -> {dtype})", converted_type=dtype)
            return g.op("Add", g.constant(start, dtype),
                        g.op("Multiply", index, g.constant(step, dtype), dtype=dtype), dtype=dtype)
        if op in {"aten.pow.Scalar", "aten.pow.Tensor_Scalar", "aten.pow.Tensor_Tensor"} and args[1] != 2:
            return g.op("Power", (yield from operand(args[0])), (yield from operand(args[1])), dtype=dtype)
        if op in binary:
            right = yield from operand(args[1])
            alpha = node.kwargs.get("alpha", 1)
            if op in {"aten.add.Tensor", "aten.sub.Tensor"} and alpha != 1:
                right = g.op("Multiply", right, g.constant(alpha, dtype), dtype=dtype)
            if set(node.kwargs) - {"alpha"}:
                raise Unsupported(f"unsupported attributes for {op}")
            return g.op(binary[op], (yield from operand(args[0])), right, dtype=dtype)
        if op in unary:
            return g.op(unary[op], (yield from operand(args[0])), dtype=dtype)
        if op == "aten.reciprocal.default":
            return g.op("Divide", g.constant(1, dtype), (yield from operand(args[0])), dtype=dtype)
        if op == "aten.rsub.Scalar":
            alpha = node.kwargs.get("alpha", 1)
            left = g.op("Multiply", (yield from operand(args[0])), g.constant(alpha, dtype), dtype=dtype)
            return g.op("Subtract", (yield from operand(args[1])), left, dtype=dtype)
        if op in {"aten.clamp.default", "aten.clamp_min.default", "aten.clamp_max.default"}:
            result = yield from operand(args[0])
            limits = {"min": args[1] if len(args) > 1 else node.kwargs.get("min"),
                      "max": args[2] if len(args) > 2 else node.kwargs.get("max")}
            if op == "aten.clamp_max.default":
                limits = {"max": args[1] if len(args) > 1 else node.kwargs["max"]}
            for key, value in limits.items():
                if value is not None:
                    # PyTorch clamp preserves NaN, unlike default PTX min/max.
                    # Match the existing ValueIR ArithmeticAttributes spelling;
                    # do not expose this head to NaN-ignoring rewrites.
                    attributes = "propagate_nan " if dtype in {"f16", "bf16", "f32", "f64", "e4m3", "e5m2"} else ""
                    result = g.node("Operation", dtype,
                                    "Maximum" if key == "min" else "Minimum",
                                    (result, (yield from operand(value))), attributes)
            return result
        if op == "aten.square.default" or (op == "aten.pow.Tensor_Scalar" and args[1] == 2):
            v = yield from operand(args[0])
            return g.op("Multiply", v, v, dtype=dtype)
        if op in {"aten.permute.default", "aten.transpose.int", "aten.t.default", "aten.numpy_T.default"}:
            rank = len(self.shape(args[0]))
            perm = list(range(rank))
            if op == "aten.permute.default":
                perm = [int(x) % rank for x in args[1]]
            elif op == "aten.transpose.int":
                a, b = int(args[1]) % rank, int(args[2]) % rank
                perm[a], perm[b] = perm[b], perm[a]
            else:
                perm.reverse()
            inverse = [None] * rank
            for j, axis in enumerate(perm):
                inverse[axis] = indices[j]
            return (yield (args[0], inverse, depth))
        if op in {"aten.matmul.default", "aten.mm.default", "aten.bmm.default"}:
            a, b = args[:2]
            sa, sb = self.shape(a), self.shape(b)
            if len(sa) < 2 or len(sb) < 2 or sa[-1] != sb[-2]:
                raise Unsupported("matmul requires matrix operands")
            k = g.index(depth)
            def batch_operand(n, sh, tail):
                batch = indices[:len(indices)-2]
                batch = batch[len(batch)-(len(sh)-2):] if len(sh) > 2 else []
                batch = [g.constant(0) if extent == 1 else index
                         for extent, index in zip(sh[:-2], batch)]
                return (yield (n, batch + tail, depth+1))
            av = yield from batch_operand(a, sa, [indices[-2], k])
            bv = yield from batch_operand(b, sb, [k, indices[-1]])
            return g.fold(g.op("Multiply", av, bv, dtype=dtype), depth, sa[-1], dtype)
        if op in {"aten.sum.dim_IntList", "aten.mean.dim", "aten.sum.default", "aten.mean.default", "aten.amax.default", "aten.amin.default"}:
            source_shape = self.shape(args[0])
            dims = args[1] if len(args) > 1 else node.kwargs.get("dim")
            if dims is None or dims == []:
                axes = list(range(len(source_shape)))
            else:
                dims = [dims] if isinstance(dims, int) else dims
                axes = sorted(int(x) % len(source_shape) for x in dims)
            keep = args[2] if len(args) > 2 else node.kwargs.get("keepdim", False)
            source_indices = []
            cursor = 0
            for axis in range(len(source_shape)):
                if axis in axes:
                    source_indices.append(g.index(depth+axes.index(axis)))
                    if keep:
                        cursor += 1
                else:
                    source_indices.append(indices[cursor])
                    cursor += 1
            body = yield (args[0], source_indices, depth+len(axes))
            for pos in reversed(range(len(axes))):
                reducer = "Max" if op == "aten.amax.default" else "Min" if op == "aten.amin.default" else "Sum"
                body = g.fold(body, depth+pos, source_shape[axes[pos]], dtype, reducer)
            if op.startswith("aten.mean"):
                body = g.op("Divide", body, g.constant(math.prod(source_shape[x] for x in axes), dtype), dtype=dtype)
            return body
        if op in {"aten.clone.default", "aten.contiguous.default", "aten.alias.default", "aten.detach.default"}:
            return (yield (args[0], indices, depth))
        if op == "aten.slice.Tensor":
            source_shape = self.shape(args[0])
            axis = int(args[1]) % len(source_shape)
            start, stop, step = slice(args[2] if len(args) > 2 else None,
                                     args[3] if len(args) > 3 else None,
                                     args[4] if len(args) > 4 else 1).indices(source_shape[axis])
            if step <= 0:
                raise Unsupported("nonpositive slice step")
            source_indices = list(indices)
            source_indices[axis] = g.op("Add", g.constant(start), g.op("Multiply", indices[axis], g.constant(step)))
            return (yield (args[0], source_indices, depth))
        if op == "aten.select.int":
            source_shape = self.shape(args[0])
            axis = int(args[1]) % len(source_shape)
            index = int(args[2]) % source_shape[axis]
            source_indices = list(indices)
            source_indices.insert(axis, g.constant(index))
            return (yield (args[0], source_indices, depth))
        if op == "aten.unsqueeze.default":
            axis = int(args[1]) % len(shape)
            return (yield (args[0], indices[:axis]+indices[axis+1:], depth))
        if op == "aten.gather.default":
            axis = int(args[1]) % len(shape)
            coordinates = list(indices)
            coordinates[axis] = yield (args[2], indices, depth)
            return (yield (args[0], coordinates, depth))
        if op == "aten.index_select.default":
            axis = int(args[1]) % len(shape)
            coordinates = list(indices)
            coordinates[axis] = yield (args[2], [indices[axis]], depth)
            return (yield (args[0], coordinates, depth))
        if op == "aten.index.Tensor":
            source_shape = self.shape(args[0])
            index_args = list(args[1]) + [None] * (len(source_shape) - len(args[1]))
            axes = [i for i, arg in enumerate(index_args) if arg is not None]
            if not axes or any(tensor_type(index_args[i].meta["val"]) not in {"s32", "s64"} for i in axes):
                raise Unsupported("advanced indexing requires fixed-rank integer indices")
            broadcast = torch.broadcast_shapes(*(self.shape(index_args[i]) for i in axes))
            contiguous = axes == list(range(axes[0], axes[-1]+1))
            prefix = axes[0] if contiguous else 0
            selected = indices[prefix:prefix+len(broadcast)]
            basic = iter(indices[:prefix] + indices[prefix+len(broadcast):])
            coordinates = []
            for axis, arg in enumerate(index_args):
                if arg is None:
                    coordinates.append(next(basic))
                    continue
                coordinate = yield from operand(arg, selected)
                index_type = tensor_type(arg.meta["val"])
                negative = g.op("Less", coordinate, g.constant(0, index_type), dtype="pred")
                wrapped = g.op("Add", coordinate, g.constant(source_shape[axis], index_type), dtype=index_type)
                coordinate = g.op("Select", negative, wrapped, coordinate, dtype=index_type)
                coordinates.append(dict(coordinate, conversion=f"Convert({index_type} -> u64)", converted_type="u64"))
            return (yield (args[0], coordinates, depth))
        if op == "aten.cat.default":
            axis = int(args[1]) % len(shape) if len(args) > 1 else 0
            choices = []
            start = 0
            for source in args[0]:
                size = self.shape(source)[axis]
                if size == 0:
                    continue
                coordinates = list(indices)
                # Bounded in inactive branches; equals i-start in the selected segment.
                local = g.op("Remainder", indices[axis], g.constant(size))
                local = g.op("Add", local, g.constant((-start) % size))
                coordinates[axis] = g.op("Remainder", local, g.constant(size))
                choices.append((start+size, (yield (source, coordinates, depth))))
                start += size
            if not choices:
                raise Unsupported("empty concatenation")
            result = choices[-1][1]
            for end, value in reversed(choices[:-1]):
                result = g.op("Select", g.op("Less", indices[axis], g.constant(end), dtype="pred"), value, result, dtype=dtype)
            return result
        if op in {"aten.view.default", "aten.reshape.default"}:
            flat = g.constant(0)
            for index, stride in zip(indices, strides(shape, 1)):
                flat = g.op("Add", flat, g.op("Multiply", index, g.constant(stride)))
            oldshape = self.shape(args[0])
            oldindices = [g.op("Remainder", g.op("Divide", flat, g.constant(stride)), g.constant(size))
                          for size, stride in zip(oldshape, strides(oldshape, 1))]
            return (yield (args[0], oldindices, depth))
        if op == "aten.__xor__.Scalar":
            op = "aten.bitwise_xor.Scalar"
        if op == "aten.__and__.Scalar":
            op = "aten.bitwise_and.Scalar"
        if op == "aten.__rshift__.Scalar":
            op = "aten.bitwise_right_shift.Tensor_Scalar"
        if op in {"aten.bitwise_and.Scalar", "aten.bitwise_xor.Scalar", "aten.bitwise_right_shift.Tensor_Scalar"}:
            name = {"aten.bitwise_and.Scalar": "BitAnd", "aten.bitwise_xor.Scalar": "BitXor",
                    "aten.bitwise_right_shift.Tensor_Scalar": "ShiftRight"}[op]
            return g.op(name, (yield from operand(args[0])), (yield from operand(args[1])), dtype=dtype)
        if op == "aten.stack.default":
            axis = int(args[1]) % len(shape) if len(args) > 1 else 0
            coordinates = indices[:axis]+indices[axis+1:]
            choices = []
            for x in args[0]:
                choices.append((yield (x, coordinates, depth)))
            result = choices[-1]
            for i in reversed(range(len(choices)-1)):
                condition = g.op("Equal", indices[axis], g.constant(i), dtype="pred")
                result = g.op("Select", condition, choices[i], result, dtype=dtype)
            return result
        if op in {"aten.to.dtype", "aten._to_copy.default"}:
            old = tensor_type(args[0].meta["val"])
            value = yield (args[0], indices, depth)
            if old == dtype:
                return value
            value = dict(value)
            conversion = f"Convert({old} -> {dtype})"
            if old in {"u8", "s32", "s64"} and dtype in {"f32", "f64"}:
                # Record the ordinary ATen cast's nearest-even semantics on
                # this use. A bare native Convert is NOT a default-RN alias.
                # Small/quantized float formats can have intermediate casts
                # or saturation semantics; do not invent a direct RN edge.
                conversion = f"Convert rounding=nearest-even({old} -> {dtype})"
            if old in {"f16", "bf16", "f32", "f64"} and dtype in {"e4m3", "e5m2"}:
                # Ordinary ATen float8 cast is RN, not finite saturation.
                # A preceding clamp remains a separate mathematical operation.
                conversion = f"Convert rounding=nearest-even({old} -> {dtype})"
            value["conversion"] = "; ".join(filter(None, [value.get("conversion"), conversion]))
            value["converted_type"] = dtype
            return value
        raise Unsupported(f"operator not supported: {op} ({node.name})")

    def outputs(self, nodes, bindings):
        if len(nodes) != len(bindings):
            raise ValueError("provide one binding per tensor output")
        families = []
        for node, binding in zip(nodes, bindings):
            if set(binding) - {"allocation", "shape", "offset"}:
                raise ValueError("unknown output binding field")
            if not isinstance(binding.get("allocation"), str) or not binding["allocation"]:
                raise ValueError("every output requires an allocation binding")
            if type(binding.get("offset", 0)) is not int or binding.get("offset", 0) < 0:
                raise ValueError("output offset must be nonnegative")
            if not isinstance(node, torch.fx.Node):
                raise Unsupported("only tensor outputs are supported")
            shape = self.shape(node)
            if not shape or any(x <= 0 for x in shape):
                raise Unsupported("output must have nonempty fixed dimensions")
            element_bytes = node.meta["val"].element_size()
            if "shape" in binding and list(shape) != binding["shape"]:
                raise ValueError("output binding shape mismatch")
            body = self.value(node, [self.graph.index(i) for i in range(len(shape))], len(shape))
            families.append(dict(allocation=binding["allocation"], offset=binding.get("offset", 0),
                                 element_bytes=element_bytes, shape=list(shape),
                                 strides_bytes=strides(shape, element_bytes), body=body))
        return self.graph.artifact(families)
