"""Hash-consed construction of scalar-fold v2 artifacts."""

import json
import math


class Graph:
    def __init__(self):
        self.nodes = []
        self.keys = {}

    def node(self, kind, dtype, name="", operands=(), attributes="", **extra):
        item = dict(kind=kind, type=dtype, name=name, attributes=attributes,
                    operands=[dict(x) for x in operands], **extra)
        key = json.dumps(item, sort_keys=True)
        if key not in self.keys:
            self.keys[key] = len(self.nodes)
            self.nodes.append(item)
        return {"node": self.keys[key]}

    def constant(self, value, dtype="u64"):
        return self.node("Constant", dtype, str(value).lower() if isinstance(value, bool) else str(value))

    def index(self, level):
        return self.node("Index", "u64", index_level=level)

    def op(self, name, *operands, dtype="u64"):
        return self.node("Operation", dtype, name, operands)

    def fold(self, body, level, count, dtype, reducer="Sum"):
        initial = {"Sum": 0, "Max": -math.inf, "Min": math.inf}[reducer]
        return self.node("Fold", dtype, operands=(self.constant(initial, dtype), body),
                         fold=dict(reducer=reducer, binder_level=level, count=count))

    def artifact(self, families):
        return dict(format="spverify.scalar-fold.v2", numerical_semantics="ideal_real",
                    nodes=self.nodes, outputs=[], output_families=families)


def strides(shape, element_bytes):
    result = []
    for size in reversed(shape):
        result.append(element_bytes)
        element_bytes *= size
    return list(reversed(result))
