"""Mathematical reference; sources (pinned):
https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl
https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/moe_gemm_kernels.h
https://github.com/NVIDIA/cutlass/blob/b46b16d003484063bca4ed365e44095c4c6ed633/include/cutlass/epilogue/collective/sm100_epilogue_array_nosmem.hpp

CUTLASS ptr-array grouped GEMM (SM100, half x half -> half, swap_ab = true, NoSmem epilogue,
thread::LinearCombination with ScaleType::OnlyAlphaScaling). For group g with problem (M, N, K):
  D_g(m, n) = alpha * sum_k A_g(m, k) * B_g(n, k)
A_g is (M, K) K-major (the expert weights after swap_ab), B_g is (N, K) K-major (the routed
activations), D_g is (M, N) column-major (LayoutD_T). The reference returns D_g as an (M, N) tensor;
the binding maps it onto the column-major kernel storage.
"""
import torch


def reference(A0, A1, A2, A3, B0, B1, B2, B3, alpha, table_problem_shapes, table_scheduler_problem_shapes, table_mainloop_dA_array, table_mainloop_dB_array, table_epilogue_dC_array, table_epilogue_dD_array):
    # table_problem_shapes, table_scheduler_problem_shapes, table_mainloop_dA_array, table_mainloop_dB_array, table_epilogue_dC_array, table_epilogue_dD_array: fixed shape-array tables, passed for the binding and
    # not used by the computation; their values enter through the binding shapes and scalars.
    outputs = []
    for a, b in zip((A0, A1, A2, A3), (B0, B1, B2, B3)):
        outputs.append((alpha * (a.float() @ b.float().T)).to(torch.float16))
    return tuple(outputs)
