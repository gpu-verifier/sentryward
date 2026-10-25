"""Mathematical reference; sources (pinned):
https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl
https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh
https://github.com/NVIDIA/cutlass/blob/b46b16d003484063bca4ed365e44095c4c6ed633/include/cutlass/detail/sm100_blockscaled_layout.hpp

CUTLASS ptr-array block-scaled grouped GEMM (SM100, MXFP8: e4m3 x e4m3 with ue8m0 scale factors,
32-element K blocks, swap_ab = true, fusion::LinearCombination<half, float, void, float>; C is void).
For group g with problem (M, N, K):
  D_g(m, n) = alpha * sum_k SFA_g(m, k // 32) A_g(m, k) * SFB_g(n, k // 32) B_g(n, k)
A_g (M, K) and B_g (N, K) are K-major e4m3. SFA_g / SFB_g are the original physical scale storage in
the canonical Sm1xxBlkScaledConfig layout (tile_atom_to_shape_SFA/SFB, which the upstream host code
writes for every expert). D_g is (M, N) column-major (LayoutD_T); the binding maps it.
"""
import torch
from examples._reference.pytorch_blockscaled import product


def reference(A0, A1, A2, A3, B0, B1, B2, B3, SFA0, SFA1, SFA2, SFA3, SFB0, SFB1, SFB2, SFB3, alpha, table_problem_shapes, table_scheduler_problem_shapes, table_mainloop_dA_array, table_mainloop_dB_array, table_epilogue_dC_array, table_epilogue_dD_array, table_mainloop_layout_SFA_array, table_mainloop_layout_SFB_array):
    # table_problem_shapes, table_scheduler_problem_shapes, table_mainloop_dA_array, table_mainloop_dB_array, table_epilogue_dC_array, table_epilogue_dD_array, table_mainloop_layout_SFA_array, table_mainloop_layout_SFB_array: fixed shape-array tables, passed for the binding and
    # not used by the computation; their values enter through the binding shapes and scalars.
    outputs = []
    for a, b, sa, sb in zip((A0, A1, A2, A3), (B0, B1, B2, B3), (SFA0, SFA1, SFA2, SFA3), (SFB0, SFB1, SFB2, SFB3)):
        acc = product(a, b, sa, sb, packed=False, block=32, e8m0=True)
        outputs.append((alpha * acc).to(torch.float16))
    return tuple(outputs)
