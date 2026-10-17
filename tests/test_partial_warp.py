"""Partial-warp launches of kernels without barriers or warp collectives (sentryward.frontend.partial_warp)."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from sentryward.frontend.partial_warp import pad

# nvcc -ptx -arch=compute_89 of: b[i] = a[i] * 2 (one element a thread), b[0] = threadIdx.x (every thread
# writes b[0]), and a shared-memory exchange behind __syncthreads().
PTX = r"""
.version 8.5
.target sm_89
.address_size 64

.visible .entry own(.param .u64 own_param_0, .param .u64 own_param_1)
{
	.reg .b32 %r<5>;
	.reg .f32 %f<3>;
	.reg .b64 %rd<8>;
	ld.param.u64 %rd1, [own_param_0];
	ld.param.u64 %rd2, [own_param_1];
	cvta.to.global.u64 %rd3, %rd2;
	cvta.to.global.u64 %rd4, %rd1;
	mov.u32 %r1, %ctaid.x;
	mov.u32 %r2, %ntid.x;
	mov.u32 %r3, %tid.x;
	mad.lo.s32 %r4, %r1, %r2, %r3;
	mul.wide.s32 %rd5, %r4, 4;
	add.s64 %rd6, %rd4, %rd5;
	ld.global.f32 %f1, [%rd6];
	add.f32 %f2, %f1, %f1;
	add.s64 %rd7, %rd3, %rd5;
	st.global.f32 [%rd7], %f2;
	ret;
}

.visible .entry same(.param .u64 same_param_0)
{
	.reg .b32 %r<2>;
	.reg .f32 %f<2>;
	.reg .b64 %rd<3>;
	ld.param.u64 %rd1, [same_param_0];
	cvta.to.global.u64 %rd2, %rd1;
	mov.u32 %r1, %tid.x;
	cvt.rn.f32.u32 %f1, %r1;
	st.global.f32 [%rd2], %f1;
	ret;
}

.visible .entry synced(.param .u64 synced_param_0)
{
	.reg .b32 %r<2>;
	.reg .b64 %rd<3>;
	ld.param.u64 %rd1, [synced_param_0];
	mov.u32 %r1, %tid.x;
	bar.sync 0;
	ret;
}
"""


def _input(entry, grid, block, pointers):
    return {"schema": "spverify.kernel-input/v1", "entry": entry,
            "launch": {"grid_dim": grid, "block_dim": block},
            "declarations": [{"parameter": i, "type": "pointer", "space": "global", "object": f"p{i}"}
                             for i in range(pointers)]}


def test_whole_warps_are_left_alone():
    assert pad(PTX, _input("own", [1, 1, 1], [64, 1, 1], 2)) is None


def test_partial_warp_is_padded_in_z_with_a_guard_on_the_brace_line():
    ptx, padded, warning = pad(PTX, _input("own", [3, 1, 1], [12, 1, 1], 2))
    assert padded["launch"]["block_dim"] == [12, 1, 8]  # 12 * 8 = 96 threads, three whole warps
    assert warning["code"] == "partial_warp_padded"
    # The guard shares the line of the entry's '{': every original line keeps its number.
    assert ptx.count("\n") == PTX.count("\n")
    assert "setp.ge.u32 %sw_partial_p, %sw_partial_t, 1; @%sw_partial_p ret;" in ptx


def test_kernels_with_barriers_keep_the_refusal():
    assert pad(PTX, _input("synced", [1, 1, 1], [12, 1, 1], 1)) is None


NATIVE = os.environ.get("SENTRYWARD_PTX_NATIVE")


@pytest.mark.skipif(not NATIVE or not shutil.which(NATIVE) and not Path(NATIVE).is_file(),
                    reason="needs the native verifier (SENTRYWARD_PTX_NATIVE)")
@pytest.mark.parametrize("entry,grid,block,pointers,expected", [
    ("own", [3, 1, 1], [12, 1, 1], 2, "race_free"),
    ("same", [1, 1, 1], [12, 1, 1], 1, "potential_race"),
    ("same", [1, 1, 1], [1, 1, 1], 1, "race_free"),        # one thread: the padded threads never write
    ("same", [4, 1, 1], [1, 1, 1], 1, "potential_race"),   # one thread a block, four blocks
])
def test_padded_launch_verdicts(tmp_path, entry, grid, block, pointers, expected):
    from sentryward import verify
    (tmp_path / "k.ptx").write_text(PTX)
    (tmp_path / "in.json").write_text(json.dumps(_input(entry, grid, block, pointers)))
    result = verify(tmp_path / "k.ptx", input=tmp_path / "in.json", timeout_seconds=120)
    assert result["result"] == expected, result
    assert any(w.get("code") == "partial_warp_padded" for w in result.get("warnings", []))
