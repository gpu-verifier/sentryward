"""Mathematical reference; sources (pinned):
https://github.com/vllm-project/vllm/blob/605c3ddcbaaee06a2687a961ae7be51aec5dddc0/csrc/libtorch_stable/quantization/fp4/mxfp4_blockwise_moe_kernel.cu
https://github.com/NVIDIA/cutlass/blob/cb4247394dd82148787aed73e5dc7cef33cbf862/include/cutlass/detail/sm100_blockscaled_layout.hpp

CUTLASS ptr-array block-scaled grouped GEMM (SM100 1SM, 128x128x128, mx_float4_t operands, ue8m0 scale factors per 32 K elements; e2m1 packed two per
byte, low nibble first). A_g is (M, K) RowMajor, B_g the ColumnMajor K x N operand stored as (N, K)
K-major, D_g bf16 RowMajor. fusion::LinearCombination: D = alpha * acc + beta * C with
alpha_ptr_array, alpha_ptr, beta_ptr_array and beta_ptr null and beta = 0 in the input, so
  D_g(m, n) = alpha * sum_k SFA_g(m, k // 32) A_g(m, k) * SFB_g(n, k // 32) B_g(n, k)
SFA_g / SFB_g are the original physical scale storage in the canonical tile_atom_to_shape_SFA/SFB
layout written by the get_group_gemm_starts kernel.
"""
import torch
from examples._reference.pytorch_blockscaled import product


def reference(A0, A1, A2, A3, B0, B1, B2, B3, SFA0, SFA1, SFA2, SFA3, SFB0, SFB1, SFB2, SFB3, alpha, table_problem_shapes, table_scheduler_problem_shapes, table_mainloop_dA_array, table_mainloop_dB_array, table_epilogue_dC_array, table_epilogue_dD_array, table_mainloop_layout_SFA_array, table_mainloop_layout_SFB_array):
    # table_problem_shapes, table_scheduler_problem_shapes, table_mainloop_dA_array, table_mainloop_dB_array, table_epilogue_dC_array, table_epilogue_dD_array, table_mainloop_layout_SFA_array, table_mainloop_layout_SFB_array: fixed shape-array tables, bound only so the native fixed-input
    # context finds a common identity on both binding sides (fixed_input_context.hpp decode_verified);
    # their values enter through the binding shapes and scalars, not through this computation.
    outputs = []
    for a, b, sa, sb in zip((A0, A1, A2, A3), (B0, B1, B2, B3), (SFA0, SFA1, SFA2, SFA3), (SFB0, SFB1, SFB2, SFB3)):
        acc = product(a, b, sa, sb, packed=True, block=32, e8m0=True)
        outputs.append((alpha * acc).to(torch.bfloat16))
    return tuple(outputs)
