"""Mathematical reference; sources (pinned):
https://github.com/pytorch/pytorch/blob/2b3ec34829036a65cd9d1398ea72a0167dc37470/aten/src/ATen/native/cuda/GroupMM.cu
https://github.com/pytorch/pytorch/blob/2b3ec34829036a65cd9d1398ea72a0167dc37470/aten/src/ATen/native/cuda/GroupMMCommon.cuh

bf16bf16_grouped_gemm_impl_sm90_sm100<Sm100, a_row_major = true, b_row_major = false, 1SM,
128x256x64>: CUTLASS ptr-array grouped GEMM with fusion::LinearCombination<bf16, float> and void C
(alpha = 1 from the launcher). For group g with problem (M, N, K):
  out_g(m, n) = alpha * sum_k A_g(m, k) * B_g(n, k)
A_g is (M, K) row-major; B_g is the column-major K x N operand, stored as (N, K) K-major
(mat_b.stride(-1) != 1); out_g is (M, N) row-major (LayoutOutput = RowMajor).
"""
import torch


def reference(A0, A1, B0, B1, alpha, table_problem_sizes, table_scheduler_problem_sizes, table_stride_output, table_stride_A, table_stride_B):
    # table_problem_sizes, table_scheduler_problem_sizes, table_stride_output, table_stride_A, table_stride_B: fixed shape-array tables, bound only so the native fixed-input
    # context finds a common identity on both binding sides (fixed_input_context.hpp decode_verified);
    # their values enter through the binding shapes and scalars, not through this computation.
    return tuple((alpha * (a.float() @ b.float().T)).to(torch.bfloat16) for a, b in ((A0, B0), (A1, B1)))
