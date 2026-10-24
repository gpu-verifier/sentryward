"""CUTLASS SM100 grouped (ptr-array) blockwise-scaled FP8 GEMM,
nnn, 64x32x128 1SM, Sm1xxBlockwiseScaleConfig<64, 32, 128>, in the ideal-real comparison model.

Source (CUTLASS 0efeafbafc16d5b8b17fd74bbac79674325b38a2):
  https://github.com/NVIDIA/cutlass/blob/0efeafbafc16d5b8b17fd74bbac79674325b38a2/include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized_mma_transform.hpp
  https://github.com/NVIDIA/cutlass/blob/0efeafbafc16d5b8b17fd74bbac79674325b38a2/include/cutlass/gemm/collective/sm100_mma_array_warpspecialized_blockwise_scaling.hpp
  https://github.com/NVIDIA/cutlass/blob/0efeafbafc16d5b8b17fd74bbac79674325b38a2/include/cutlass/detail/blockwise_scale_layout.hpp
  https://github.com/NVIDIA/cutlass/blob/0efeafbafc16d5b8b17fd74bbac79674325b38a2/python/cutlass_library/generator.py

Per group g, with 128-wide K blocks kb:
  D_g[m, n] = alpha * sum_kb (A_g[m, kb] . B_g[n, kb]) * SFA_g[m // 64, kb] * SFB_g[n // 32, kb]  (+ beta * C_g)
A_g: logical (M_g, K_g) e4m3; B_g: logical (N_g, K_g) e4m3 (CUTLASS B is N x K);
SFA_g: logical (ceil(M_g/64), K_g/128) f32; SFB_g: logical (ceil(N_g/32), K_g/128) f32; D_g f16.
beta = 0 in the input (`+1280`, null beta pointers), so the epilogue does not read C_g and the term is omitted.
"""
import torch


def _group(a, b, scale_a, scale_b, alpha):
    m, k = a.shape
    n = b.shape[0]
    result = 0.0
    for block in range(k // 128):
        lo, hi = block * 128, (block + 1) * 128
        part = (a[:, lo:hi].float() @ b[:, lo:hi].float().T).reshape(m // 64, 64, n // 32, 32)
        part = part * scale_a[:, block].reshape(m // 64, 1, 1, 1) * scale_b[:, block].reshape(1, 1, n // 32, 1)
        result = result + part.reshape(m, n)
    return (alpha * result).to(torch.float16)


def reference(A0, A1, A2, A3, B0, B1, B2, B3, SFA0, SFA1, SFA2, SFA3, SFB0, SFB1, SFB2, SFB3,
              fixed_problem_shape_problem_shapes, fixed_scheduler_problem_shapes,
              fixed_mainloop_layout_SFA_array, fixed_mainloop_layout_SFB_array, fixed_mainloop_dA_array,
              fixed_mainloop_dB_array, fixed_epilogue_dC_array, fixed_epilogue_dD_array, alpha):
    # fixed_*: shape-array tables (read-only kernel inputs fixed by input.json), bound only so the native
    # fixed-input context finds a common identity (fixed_input_context.hpp decode_verified); not used here.
    return (_group(A0, B0, SFA0, SFB0, alpha), _group(A1, B1, SFA1, SFB1, alpha),
            _group(A2, B2, SFA2, SFB2, alpha), _group(A3, B3, SFA3, SFB3, alpha))
