"""CUTLASS SM100 grouped (ptr-array) MXFP4 block-scaled GEMM, tnt, 256x128x128 2SM, in the ideal-real model.

Source (CUTLASS 0efeafbafc16d5b8b17fd74bbac79674325b38a2):
  https://github.com/NVIDIA/cutlass/blob/0efeafbafc16d5b8b17fd74bbac79674325b38a2/include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized.hpp
  https://github.com/NVIDIA/cutlass/blob/0efeafbafc16d5b8b17fd74bbac79674325b38a2/include/cutlass/gemm/collective/sm100_blockscaled_mma_array_warpspecialized.hpp
  https://github.com/NVIDIA/cutlass/blob/0efeafbafc16d5b8b17fd74bbac79674325b38a2/include/cutlass/detail/sm100_blockscaled_layout.hpp
  https://github.com/NVIDIA/cutlass/blob/0efeafbafc16d5b8b17fd74bbac79674325b38a2/python/cutlass_library/generator.py

Per group g: D_g = alpha * (dec(A_g) * SFA_g) @ (dec(B_g) * SFB_g)^T, D f32 (C is void).
A_g/B_g are packed e2m1 bytes (rows, K_g/2); SFA_g/SFB_g are the raw UE8M0 scale storage of
Sm1xxBlkScaledConfig<32> (one 512-byte atom per 128 rows x 128 K), decoded by the shared helper
(same helper as the dense sibling cutlass__block-scaled-mxfp4-gemm-2sm__sm_100a).
The 2SM tiling does not change the mathematical result.
"""
import torch
from examples._reference.pytorch_blockscaled import product


def _group(a, b, scale_a, scale_b, alpha):
    result = product(a, b, scale_a, scale_b, packed=True, block=32, e8m0=True)
    return (alpha * result).to(torch.float32)


def reference(A0, A1, A2, A3, B0, B1, B2, B3, SFA0, SFA1, SFA2, SFA3, SFB0, SFB1, SFB2, SFB3,
              fixed_problem_shape_problem_shapes, fixed_scheduler_problem_shapes, fixed_mainloop_dA_array,
              fixed_mainloop_dB_array, fixed_epilogue_dC_array, fixed_epilogue_dD_array,
              fixed_mainloop_layout_SFA_array, fixed_mainloop_layout_SFB_array, alpha):
    # fixed_*: fixed shape-array tables (read-only kernel inputs fixed by input.json), passed for the
    # binding; not used by the computation.
    return (_group(A0, B0, SFA0, SFB0, alpha), _group(A1, B1, SFA1, SFB1, alpha),
            _group(A2, B2, SFA2, SFB2, alpha), _group(A3, B3, SFA3, SFB3, alpha))
