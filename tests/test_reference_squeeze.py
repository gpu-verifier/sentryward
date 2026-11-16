"""Reference capture of aten.squeeze (sentryward/_reference/lowering.py).

squeeze is a pure shape operation: a named dimension of size 1 is removed, a named dimension of
another size is kept, and squeeze() names every dimension. Each case checks that the exported graph
uses the intended squeeze overload and that its scalar-fold artifact is identical to the one of an
equivalent reference written with select/indexing or without the squeeze. These tests need torch
but no native executable.
"""
import unittest

try:
    import torch
    from sentryward._reference.capture import extract
    from sentryward._reference.graph import strides
    from sentryward._reference.lowering import Lowering, Unsupported
except ImportError:  # pragma: no cover - environments without torch
    torch = None


def capture(function, shapes, dtype="float32", output_shape=None):
    names = [f"x{i}" for i in range(len(shapes))]
    arguments = [dict(name=n, shape=list(s), dtype=dtype, allocation=n) for n, s in zip(names, shapes)]
    output = dict(allocation="out")
    if output_shape is not None:
        output["shape"] = list(output_shape)
    namespace = {}
    exec(f"def reference({', '.join(names)}):\n    return (function({', '.join(names)}),)", dict(function=function), namespace)
    artifact, meta = extract(namespace["reference"], dict(schema="spverify.pytorch-reference.v1",
                                                          arguments=arguments, outputs=[output]))
    return artifact, meta["exported_graph"]


def lowered(target, shape, out_shape, args=(), kwargs=None):
    """Lower one call_function node of a hand-built FX graph over a float32 placeholder x."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty(shape, device="meta")
    node = graph.call_function(target, (x, *args), kwargs or {})
    node.meta["val"] = torch.empty(out_shape, device="meta")
    bindings = {"x": dict(allocation="x", strides_bytes=strides(shape, 4))}
    return Lowering(bindings).outputs([node], [dict(allocation="out")])


def rendered(artifact):
    """Every output family with its body as an expression tree: node numbering (which depends on
    construction order, including nodes no output reaches) does not matter."""
    nodes = artifact["nodes"]
    memo = {}

    def ref(r):
        extra = tuple((k, r[k]) for k in ("conversion", "converted_type", "pointer_provenance") if k in r)
        return (node(r["node"]),) + extra

    def node(i):
        if i not in memo:
            n = nodes[i]
            memo[i] = (n["kind"], n["type"], n["name"], n["attributes"], n.get("index_level"),
                       tuple(sorted(n.get("fold", {}).items())), tuple(ref(r) for r in n["operands"]))
        return memo[i]

    return [({k: v for k, v in f.items() if k != "body"}, ref(f["body"])) for f in artifact["output_families"]]


@unittest.skipIf(torch is None, "torch is not installed")
class ReferenceSqueezeTest(unittest.TestCase):
    def same(self, squeezed, plain, shapes, overload, output_shape=None):
        left, graph = capture(squeezed, shapes, output_shape=output_shape)
        self.assertIn(f"aten.squeeze.{overload}", graph)
        right, other = capture(plain, shapes, output_shape=output_shape)
        self.assertNotIn("aten.squeeze", other)
        self.assertEqual(rendered(left), rendered(right))
        return left

    def test_dim_removes_a_size_one_dimension(self):
        artifact = self.same(lambda x: x.squeeze(1), lambda x: x.select(1, 0), [(3, 1, 4)], "dim", (3, 4))
        self.assertEqual(artifact["output_families"][0]["shape"], [3, 4])

    def test_negative_dim(self):
        self.same(lambda x: x.squeeze(-2), lambda x: x.select(1, 0), [(3, 1, 4)], "dim", (3, 4))

    def test_dim_of_another_size_is_a_no_op(self):
        artifact = self.same(lambda x: x.squeeze(1) * 2, lambda x: x * 2, [(3, 2, 4)], "dim", (3, 2, 4))
        self.assertEqual(artifact["output_families"][0]["shape"], [3, 2, 4])

    def test_dims_removes_only_size_one_dimensions(self):
        self.same(lambda x: x.squeeze((0, 2)), lambda x: x[0, :, 0], [(1, 5, 1, 2)], "dims", (5, 2))
        self.same(lambda x: x.squeeze((0, 1)), lambda x: x.select(0, 0), [(1, 5, 2)], "dims", (5, 2))
        self.same(lambda x: x.squeeze((-1, -3)), lambda x: x[0, :, 0], [(1, 5, 1)], "dims", (5,))

    def test_every_dimension(self):
        self.same(lambda x: x.squeeze(), lambda x: x[0, :, 0, :], [(1, 3, 1, 2)], "default", (3, 2))
        self.same(lambda x: x.squeeze() + 1, lambda x: x + 1, [(3, 2)], "default", (3, 2))

    def test_kept_reduction_axis(self):
        # The target's LSE form: log(sum(w, -1, keepdim).squeeze(-1)) = log(sum(w, -1)).
        self.same(lambda w: torch.log(w.sum(dim=-1, keepdim=True).squeeze(-1)),
                  lambda w: torch.log(w.sum(dim=-1)), [(2, 3, 8)], "dim", (2, 3))

    def test_squeeze_then_broadcast(self):
        self.same(lambda x, y: x.squeeze(0) + y, lambda x, y: x.select(0, 0) + y, [(1, 4), (3, 4)], "dim", (3, 4))

    def test_keyword_dimension(self):
        squeeze = lowered(torch.ops.aten.squeeze.dim, (3, 1, 4), (3, 4), kwargs=dict(dim=1))
        select = lowered(torch.ops.aten.select.int, (3, 1, 4), (3, 4), args=(1, 0))
        self.assertEqual(rendered(squeeze), rendered(select))
        squeeze = lowered(torch.ops.aten.squeeze.dims, (1, 5, 1), (5,), kwargs=dict(dim=[0, -1]))
        self.assertEqual(squeeze["output_families"][0]["shape"], [5])

    def test_inconsistent_graph_is_refused(self):
        # A result shape that does not follow from the removed size-1 dimensions, a dimension out
        # of range and a missing dimension are refused, never lowered.
        with self.assertRaises(Unsupported):
            lowered(torch.ops.aten.squeeze.dim, (3, 1, 4), (3, 1, 4), args=(1,))
        with self.assertRaises(Unsupported):
            lowered(torch.ops.aten.squeeze.dim, (3, 2, 4), (3, 4), args=(1,))
        with self.assertRaises(Unsupported):
            lowered(torch.ops.aten.squeeze.dim, (3, 1, 4), (3, 4), args=(3,))
        with self.assertRaises(Unsupported):
            lowered(torch.ops.aten.squeeze.dims, (3, 1, 4), (3, 4))

    def test_scalar(self):
        self.same(lambda x: x.sum().squeeze(0).unsqueeze(0), lambda x: x.sum().unsqueeze(0), [(4,)], "dim", (1,))


if __name__ == "__main__":
    unittest.main()
