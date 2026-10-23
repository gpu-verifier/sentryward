"""Mathematical reference; sources (pinned):
https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/include/flashinfer/gemm/group_gemm_mxfp4_groupwise_sm100.cuh
https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/group_gemm_mxfp4_groupwise_sm100_kernel_inst.jinja
https://github.com/NVIDIA/cutlass/blob/b46b16d003484063bca4ed365e44095c4c6ed633/include/cutlass/detail/sm100_blockscaled_layout.hpp

CutlassMXFP4GroupwiseScaledGroupGEMMSM100 (SwapAB = false): CUTLASS ptr-array block-scaled grouped
GEMM, A = e4m3 (RowMajor, K-major), B = e2m1 packed two per byte, low nibble first (ColumnMajor,
i.e. (N, K) K-major), ue8m0 scale factors per 32 K elements (ScaleGranularity == 32), C void,
D bf16 RowMajor, alpha = 1, beta = 0 (fusion_args). For group g with problem (M, N, K):
  D_g(m, n) = alpha * sum_k SFA_g(m, k // 32) A_g(m, k) * SFB_g(n, k // 32) B_g(n, k)
SFA_g / SFB_g are the original physical scale storage in the canonical
tile_atom_to_shape_SFA/SFB layout written by compute_sm100_cutlass_group_gemm_args.
"""
import torch
from examples._reference.pytorch_blockscaled import block_scales, fp4_values


def _scaled(values, storage, block):
    rows, k = values.shape
    scales = block_scales(storage, rows, k // block, True)
    return (values.reshape(rows, k // block, block) * scales.unsqueeze(-1)).reshape(rows, k)


def reference(A0, A1, A2, A3, B0, B1, B2, B3, SFA0, SFA1, SFA2, SFA3, SFB0, SFB1, SFB2, SFB3, alpha, table_problem_shapes, table_scheduler_problem_shapes, table_mainloop_dA_array, table_mainloop_dB_array, table_epilogue_dC_array, table_epilogue_dD_array, table_mainloop_layout_SFA_array, table_mainloop_layout_SFB_array):
    # table_problem_shapes, table_scheduler_problem_shapes, table_mainloop_dA_array, table_mainloop_dB_array, table_epilogue_dC_array, table_epilogue_dD_array, table_mainloop_layout_SFA_array, table_mainloop_layout_SFB_array: fixed shape-array tables, bound only so the native fixed-input
    # context finds a common identity on both binding sides (fixed_input_context.hpp decode_verified);
    # their values enter through the binding shapes and scalars, not through this computation.
    outputs = []
    for a, b, sa, sb in zip((A0, A1, A2, A3), (B0, B1, B2, B3), (SFA0, SFA1, SFA2, SFA3), (SFB0, SFB1, SFB2, SFB3)):
        acc = _scaled(a.float(), sa, 32) @ _scaled(fp4_values(b), sb, 32).T
        outputs.append((alpha * acc).to(torch.bfloat16))
    return tuple(outputs)
