"""CUTLASS SM100 MoE GEMM (MoEProblemShape, mixed TMA A / cp.async B), f16, tnn, 64x128x64 1SM,
in the ideal-real comparison model.

Source (CUTLASS 0efeafbafc16d5b8b17fd74bbac79674325b38a2):
  https://github.com/NVIDIA/cutlass/blob/0efeafbafc16d5b8b17fd74bbac79674325b38a2/include/cutlass/gemm/kernel/sm100_gemm_mixed_tma_cpasync_warpspecialized.hpp
  https://github.com/NVIDIA/cutlass/blob/0efeafbafc16d5b8b17fd74bbac79674325b38a2/include/cutlass/gemm/collective/sm100_mma_mixed_tma_cpasync_warpspecialized.hpp
  https://github.com/NVIDIA/cutlass/blob/0efeafbafc16d5b8b17fd74bbac79674325b38a2/include/cutlass/gemm/group_array_problem_shape.hpp
  https://github.com/NVIDIA/cutlass/blob/0efeafbafc16d5b8b17fd74bbac79674325b38a2/python/cutlass_library/generator.py

Expert g has problem (max_m, tokens_per_expert[g], max_k). A is the packed (L, max_m, K) RowMajor
weight tensor, B the packed (L, max_n, K) token tensor (ColumnMajor B = K contiguous; the layout uses
the max shape), D the (L, max_m, max_n) ColumnMajor output. D[g] = alpha * A[g] @ B[g]^T (+ beta * C,
beta = 0 in the input so C is not read).
Rows n >= tokens_per_expert[g] are a padding region (cp.async zero-fill); that case is not modeled.
"""
import torch


def reference(A, B, fixed_tokens_per_expert_problem_shape, fixed_tokens_per_expert, alpha, tokens_0,
              tokens_1, tokens_2, tokens_3):
    # fixed_*: fixed shape-array tables (read-only kernel inputs fixed by input.json), passed for the
    # binding; not used by the computation.
    max_n = B.shape[1]
    if any(t != max_n for t in (tokens_0, tokens_1, tokens_2, tokens_3)):
        raise NotImplementedError("reference covers tokens_per_expert[g] == max_n only")
    # Per-expert 2-D products.
    experts = [alpha * (A[g].float() @ B[g].float().T) for g in range(A.shape[0])]
    return torch.stack(experts).to(torch.float16)
