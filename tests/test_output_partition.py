"""The comparison adapter's output partition (cpp/src/reference_guided/production.cpp).

Scatter: thread t < 16 stores input[t] at output + 4 x indices[t], with indices injective in [0, 31]. The output is
[4, 8] f32, so a write's slot spans both dense axes; the partition fuses them into one slot of extent 32. A slot range
beyond the fused extent, or a padded row stride, still refuses with symbolic_write_outside_output.

Row scale: thread t < 32 stores input[t] x scale[t / 8] at output + 4 x t. The scale changes at every row of the
[4, 8] output. When the kernel's output families group the last word of a row with the first word of the next, the
partition cuts such a family into boxes instead of refusing.

These runs need the native executable; they are skipped when it is not built.
"""
import json
import tempfile
import unittest
from pathlib import Path

from sentryward.api import verify
from sentryward.frontend.ptx_native import native_executable

SCATTER = """.version 8.0
.target sm_100a
.address_size 64

.visible .entry scatter(
	.param .u64 scatter_param_0,
	.param .u64 scatter_param_1,
	.param .u64 scatter_param_2
)
.reqntid 32
{
	.reg .pred 	%p<2>;
	.reg .b32 	%r<3>;
	.reg .f32 	%f<2>;
	.reg .b64 	%rd<12>;

	ld.param.u64 	%rd1, [scatter_param_0];
	ld.param.u64 	%rd2, [scatter_param_1];
	ld.param.u64 	%rd3, [scatter_param_2];
	cvta.to.global.u64 	%rd4, %rd1;
	cvta.to.global.u64 	%rd5, %rd2;
	cvta.to.global.u64 	%rd6, %rd3;
	mov.u32 	%r1, %tid.x;
	setp.ge.u32 	%p1, %r1, 16;
	@%p1 bra 	$DONE;
	mul.wide.u32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd4, %rd7;
	ld.global.f32 	%f1, [%rd8];
	add.s64 	%rd9, %rd5, %rd7;
	ld.global.s32 	%r2, [%rd9];
	mul.wide.s32 	%rd10, %r2, 4;
	add.s64 	%rd11, %rd6, %rd10;
	st.global.f32 	[%rd11], %f1;
$DONE:
	ret;
}
"""

# Sequential scatter (last write outermost), defined where the indices are distinct.
SCATTER_REFERENCE = """import torch


def reference(input, indices, output):
    positions = torch.arange(32, device=input.device).reshape(4, 8)
    result = output
    for i in range(16):
        result = torch.where(positions == indices[i], input[i], result)
    return result
"""

ROW_SCALE = """.version 8.0
.target sm_100a
.address_size 64

.visible .entry row_scale(
	.param .u64 row_scale_param_0,
	.param .u64 row_scale_param_1,
	.param .u64 row_scale_param_2
)
.reqntid 32
{
	.reg .b32 	%r<3>;
	.reg .f32 	%f<4>;
	.reg .b64 	%rd<14>;

	ld.param.u64 	%rd1, [row_scale_param_0];
	ld.param.u64 	%rd2, [row_scale_param_1];
	ld.param.u64 	%rd3, [row_scale_param_2];
	cvta.to.global.u64 	%rd4, %rd1;
	cvta.to.global.u64 	%rd5, %rd2;
	cvta.to.global.u64 	%rd6, %rd3;
	mov.u32 	%r1, %tid.x;
	shr.u32 	%r2, %r1, 3;
	mul.wide.u32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd4, %rd7;
	ld.global.f32 	%f1, [%rd8];
	mul.wide.u32 	%rd9, %r2, 4;
	add.s64 	%rd10, %rd5, %rd9;
	ld.global.f32 	%f2, [%rd10];
	mul.rn.f32 	%f3, %f1, %f2;
	add.s64 	%rd11, %rd6, %rd7;
	st.global.f32 	[%rd11], %f3;
	ret;
}
"""

ROW_SCALE_REFERENCE = """import torch


def reference(input, scale):
    return input * scale.reshape(4, 1)
"""


def _identity(name):
    return f"6:global{len(name)}:{name}"


def _layout(shape, element_bytes, strides=None):
    if strides is None:
        strides, step = [], element_bytes
        for extent in reversed(shape):
            strides.insert(0, step)
            step *= extent
    return {"source_offset": 0, "target_offset": 0, "element_bytes": element_bytes, "shape": shape,
            "source_strides_bytes": strides, "target_strides_bytes": strides}


def _binding(inputs, output, arguments, right_strides=None):
    """`right_strides` places the kernel's output (and its initial contents) at other strides than the reference's."""
    name, layout = output
    right = dict(inputs)
    if right_strides is not None:
        right[name] = {**layout, "source_strides_bytes": right_strides}
    def side(layouts):
        return {"allocations": {_identity(n): n for n in layouts}, "parameters": {},
                "layouts": {_identity(n): value for n, value in layouts.items()}}
    mapping = {"kind": "dense_tensor", "left_allocation": _identity(name), "right_allocation": _identity(name),
               "left_offset": 0, "right_offset": 0, "element_bytes": layout["element_bytes"],
               "shape": layout["shape"], "left_strides_bytes": layout["source_strides_bytes"],
               "right_strides_bytes": right_strides or layout["source_strides_bytes"]}
    return {"schema": "spverify.equivalence-mapping.v1", "output_value_semantics": "ideal_real",
            "left_inputs": side(inputs), "right_inputs": side(right), "outputs": mapping,
            "reference": {"arguments": [{"name": n, "allocation": _identity(n), "dtype": d} for n, d in arguments]}}


def _run(directory, entry, ptx, objects, reference, binding, tensor_properties=()):
    directory = Path(directory)
    (directory / "kernel.ptx").write_text(ptx)
    (directory / "input.json").write_text(json.dumps({
        "schema": "spverify.kernel-input/v1",
        "entry": entry,
        "launch": {"grid_dim": [1, 1, 1], "block_dim": [32, 1, 1]},
        "declarations": [{"parameter": i, "type": "pointer", "object": name, "name": name}
                         for i, name in enumerate(objects)],
        "tensor_properties": list(tensor_properties),
    }))
    (directory / "reference.py").write_text(reference)
    (directory / "reference_binding.json").write_text(json.dumps(binding))
    return verify(directory / "kernel.ptx", input=directory / "input.json", reference=directory / "reference.py",
                  reference_binding=directory / "reference_binding.json", timeout_seconds=120)


def _scatter(directory, upper=31, output_strides=None):
    properties = [{"object": "indices", "element_type": "s32", "property": "lower_bound", "value": 0},
                  {"object": "indices", "element_type": "s32", "property": "upper_bound", "value": upper},
                  {"object": "indices", "element_type": "s32", "property": "injective"}]
    output = _layout([4, 8], 4)
    binding = _binding({"input": _layout([16], 4), "indices": _layout([16], 4), "output": output},
                       ("output", output), [("input", "float32"), ("indices", "int32"), ("output", "float32")],
                       right_strides=output_strides)
    return _run(directory, "scatter", SCATTER, ["input", "indices", "output"], SCATTER_REFERENCE, binding,
                properties)


def _codes(result):
    return [d.get("code") for d in result.get("diagnostics", [])]


@unittest.skipUnless(native_executable().is_file(), "native executable not built")
class OutputPartitionTests(unittest.TestCase):
    def test_scatter_slot_over_two_dense_axes(self):
        with tempfile.TemporaryDirectory() as scratch:
            result = _scatter(scratch)
        self.assertEqual(result["checks"]["synchronization"], "passed", result)
        self.assertNotIn("symbolic_write_outside_output", _codes(result), result)
        self.assertEqual(result["result"], "verified", result)

    def test_scatter_slot_range_beyond_the_output_refuses(self):
        # Index 32 would write past the 32-element output.
        with tempfile.TemporaryDirectory() as scratch:
            result = _scatter(scratch, upper=32)
        self.assertIn("symbolic_write_outside_output", _codes(result), result)

    def test_scatter_padded_rows_refuse(self):
        # Row stride 36 bytes: the axes are not dense, so no fused slot; a slot on the column axis alone
        # cannot hold [0, 31].
        with tempfile.TemporaryDirectory() as scratch:
            result = _scatter(scratch, output_strides=[36, 4])
        self.assertIn("symbolic_write_outside_output", _codes(result), result)

    def test_row_scale_families_cut_at_row_boundaries(self):
        output = _layout([4, 8], 4)
        binding = _binding({"input": output, "scale": _layout([4], 4)}, ("output", output),
                           [("input", "float32"), ("scale", "float32")])
        with tempfile.TemporaryDirectory() as scratch:
            result = _run(scratch, "row_scale", ROW_SCALE, ["input", "scale", "output"], ROW_SCALE_REFERENCE,
                          binding)
        self.assertEqual(result["result"], "verified", result)
        # More compared pieces than kernel families: some family was cut at a row boundary.
        internal = result["internal"]
        self.assertGreater(internal["equivalence"]["total_output_families"],
                           internal["value_analysis"]["scalar_fold"]["output_family_count"], result)


if __name__ == "__main__":
    unittest.main()
