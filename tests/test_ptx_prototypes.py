"""A .func forward declaration followed by its definition is valid PTX (nvcc emits both for its internal math
helpers, e.g. __internal_accurate_pow); two definitions of one function are not."""
import json
import os
from pathlib import Path

import pytest

PTX = r"""
.version 8.5
.target sm_89
.address_size 64

.func  (.param .b32 func_retval0) plus_one
(
	.param .b32 plus_one_param_0
)
;

.visible .entry k(.param .u64 k_param_0)
{
	.reg .b32 %r<4>;
	.reg .b64 %rd<5>;
	ld.param.u64 %rd1, [k_param_0];
	cvta.to.global.u64 %rd2, %rd1;
	mov.u32 %r1, %tid.x;
	{
	.param .b32 param0;
	st.param.b32 [param0], %r1;
	.param .b32 retval0;
	call.uni (retval0), plus_one, (param0);
	ld.param.b32 %r2, [retval0];
	}
	mul.wide.u32 %rd3, %r1, 4;
	add.s64 %rd4, %rd2, %rd3;
	st.global.u32 [%rd4], %r2;
	ret;
}

.func  (.param .b32 func_retval0) plus_one(
	.param .b32 plus_one_param_0
)
{
	.reg .b32 %r<3>;
	ld.param.b32 %r1, [plus_one_param_0];
	add.s32 %r2, %r1, 1;
	st.param.b32 [func_retval0], %r2;
	ret;
}
"""

INPUT = {"schema": "spverify.kernel-input/v1", "entry": "k",
         "launch": {"grid_dim": [1, 1, 1], "block_dim": [32, 1, 1]},
         "declarations": [{"parameter": 0, "type": "pointer", "space": "global", "object": "out"}]}

NATIVE = os.environ.get("SENTRYWARD_PTX_NATIVE")
needs_native = pytest.mark.skipif(not NATIVE or not Path(NATIVE).is_file(),
                                  reason="needs the native verifier (SENTRYWARD_PTX_NATIVE)")


def _verify(tmp_path, ptx):
    from sentryward import verify
    (tmp_path / "k.ptx").write_text(ptx)
    (tmp_path / "in.json").write_text(json.dumps(INPUT))
    return verify(tmp_path / "k.ptx", input=tmp_path / "in.json", timeout_seconds=120)


@needs_native
def test_prototype_then_definition_is_accepted(tmp_path):
    result = _verify(tmp_path, PTX)
    assert result["result"] != "invalid_input", result
    assert not any(d.get("code") == "invalid_ptx_syntax" for d in result.get("diagnostics", []))


@needs_native
def test_two_definitions_are_refused(tmp_path):
    definition = PTX[PTX.rindex(".func"):]
    result = _verify(tmp_path, PTX + definition)
    assert result["result"] == "invalid_input", result
    assert "duplicate function" in json.dumps(result)
