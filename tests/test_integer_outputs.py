"""Reference equivalence for integer kernels: signed outputs, signed bytes and wrapping integer sums.

Each kernel in data/integer_outputs.cu computes exactly what its PyTorch reference computes, with the same
two's-complement storage, so each must verify; a wrong reference must not.
"""
import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

DATA = Path(__file__).parent / "data"
NATIVE = os.environ.get("SENTRYWARD_PTX_NATIVE")
pytestmark = pytest.mark.skipif(not NATIVE or not Path(NATIVE).is_file(), reason="needs the native verifier (SENTRYWARD_PTX_NATIVE)")

BYTES = {"int8": 1, "int32": 4}


def allocation(name):
    return f"6:global{len(name)}:{name}"


def check(tmp_path, entry, threads, inputs, output, body):
    """inputs: [(name, dtype, n)]; output: (name, dtype, n); body: the reference's return expression."""
    from sentryward import verify
    names = [i[0] for i in inputs] + [output[0]]
    (tmp_path / "input.json").write_text(json.dumps({
        "schema": "spverify.kernel-input/v1", "entry": entry, "launch": {"grid_dim": [1, 1, 1], "block_dim": [threads, 1, 1]},
        "declarations": [{"parameter": i, "type": "pointer", "space": "global", "object": n, "name": n} for i, n in enumerate(names)]}))

    def layout(dtype, n):
        e = BYTES[dtype]
        return {"source_offset": 0, "target_offset": 0, "element_bytes": e, "shape": [n], "source_strides_bytes": [e], "target_strides_bytes": [e]}

    side = {"allocations": {allocation(n): n for n, _, _ in inputs}, "parameters": {}, "layouts": {allocation(n): layout(d, k) for n, d, k in inputs}}
    e = BYTES[output[1]]
    (tmp_path / "reference_binding.json").write_text(json.dumps({
        "schema": "spverify.equivalence-mapping.v1", "left_inputs": side, "right_inputs": side, "output_value_semantics": "ideal_real",
        "outputs": {"kind": "dense_tensor", "left_allocation": allocation(output[0]), "right_allocation": allocation(output[0]), "left_offset": 0, "right_offset": 0,
                    "element_bytes": e, "shape": [output[2]], "left_strides_bytes": [e], "right_strides_bytes": [e]},
        "reference": {"arguments": [{"name": n, "allocation": allocation(n), "dtype": d} for n, d, _ in inputs]}}))
    (tmp_path / "reference.py").write_text(f"import torch\n\n\ndef reference({', '.join(n for n, _, _ in inputs)}):\n    return {body}\n")
    return verify(DATA / "integer_outputs.ptx", input=tmp_path / "input.json", reference=tmp_path / "reference.py",
                  reference_binding=tmp_path / "reference_binding.json", timeout_seconds=300)


def test_int32_output_is_read_as_signed(tmp_path):
    result = check(tmp_path, "add_i32", 64, [("a", "int32", 64), ("b", "int32", 64)], ("c", "int32", 64), "a + b")
    assert result["result"] == "verified", result.get("diagnostics")


def test_int8_input_widened_to_int32(tmp_path):
    result = check(tmp_path, "widen_i8", 64, [("x", "int8", 64)], ("y", "int32", 64), "x.to(torch.int32)")
    assert result["result"] == "verified", result.get("diagnostics")


def test_int32_sum_wraps_like_the_kernel(tmp_path):
    result = check(tmp_path, "sum_i32", 32, [("a", "int32", 8)], ("s", "int32", 1), "a.sum(0, dtype=torch.int32).reshape(1)")
    assert result["result"] == "verified", result.get("diagnostics")


def test_wrong_reference_is_not_proven(tmp_path):
    result = check(tmp_path, "add_i32", 64, [("a", "int32", 64), ("b", "int32", 64)], ("c", "int32", 64), "a - b")
    assert result["result"] == "equivalence_not_proven", result.get("diagnostics")
