"""CUTLASS SM100 grouped (ptr-array) f16 GEMM, nnn, 64x128x64 1SM, in the ideal-real comparison model.

Source (CUTLASS 0efeafbafc16d5b8b17fd74bbac79674325b38a2):
  https://github.com/NVIDIA/cutlass/blob/0efeafbafc16d5b8b17fd74bbac79674325b38a2/include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized.hpp
  https://github.com/NVIDIA/cutlass/blob/0efeafbafc16d5b8b17fd74bbac79674325b38a2/include/cutlass/gemm/collective/sm100_mma_array_warpspecialized.hpp
  https://github.com/NVIDIA/cutlass/blob/0efeafbafc16d5b8b17fd74bbac79674325b38a2/include/cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp
  https://github.com/NVIDIA/cutlass/blob/0efeafbafc16d5b8b17fd74bbac79674325b38a2/python/cutlass_library/generator.py

Per group g: D_g = alpha * A_g @ B_g^T + beta * C_g (LinearCombination<half_t x4>).
A_g is the logical (M_g, K_g) operand and B_g the logical (N_g, K_g) operand (CUTLASS B is N x K).
beta = 0 in the input, so the epilogue does not read C_g and the term is omitted.
"""
import torch


def _group(a, b, alpha):
    return (alpha * (a.float() @ b.float().T)).to(torch.float16)


def reference(A0, A1, A2, A3, B0, B1, B2, B3, fixed_problem_shape_problem_shapes,
              fixed_scheduler_problem_shapes, fixed_mainloop_dA_array, fixed_mainloop_dB_array,
              fixed_epilogue_dC_array, fixed_epilogue_dD_array, alpha):
    # fixed_*: fixed shape-array tables (read-only kernel inputs fixed by input.json), passed for the
    # binding; not used by the computation.
    return (_group(A0, B0, alpha), _group(A1, B1, alpha),
            _group(A2, B2, alpha), _group(A3, B3, alpha))
