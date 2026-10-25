"""CUTLASS SM100 block-scaled MoE GEMM (MoEProblemShape, mixed TMA A / cp.async B), MXFP4
(kind::mxf4, UE8M0 scales per 32 elements), bf16 output, tnn, 128x128x128 1SM, in the ideal-real model.

Source (CUTLASS 0efeafbafc16d5b8b17fd74bbac79674325b38a2):
  https://github.com/NVIDIA/cutlass/blob/0efeafbafc16d5b8b17fd74bbac79674325b38a2/include/cutlass/gemm/kernel/sm100_gemm_mixed_tma_cpasync_warpspecialized.hpp
  https://github.com/NVIDIA/cutlass/blob/0efeafbafc16d5b8b17fd74bbac79674325b38a2/include/cutlass/gemm/collective/sm100_blockscaled_mma_mixed_tma_cpasync_warpspecialized.hpp
  https://github.com/NVIDIA/cutlass/blob/0efeafbafc16d5b8b17fd74bbac79674325b38a2/include/cutlass/gemm/group_array_problem_shape.hpp
  https://github.com/NVIDIA/cutlass/blob/0efeafbafc16d5b8b17fd74bbac79674325b38a2/include/cutlass/detail/sm100_blockscaled_layout.hpp
  https://github.com/NVIDIA/cutlass/blob/0efeafbafc16d5b8b17fd74bbac79674325b38a2/include/cute/arch/mma_sm100_desc.hpp

Expert g has problem (max_m, tokens_per_expert[g], max_k). A: packed e2m1 bytes (L, max_m, K/2);
B: packed e2m1 bytes (L, max_n, K/2) (max-shape packed layout); scale_a/scale_b: per-expert raw UE8M0
storage of Sm1xxBlkScaledConfig<32> (L, 512 * ceil(rows/128) * ceil(K/128)).
D[g] = alpha * (dec(A[g]) * SFA[g]) @ (dec(B[g]) * SFB[g])^T, bf16, ColumnMajor (C is void).
The A/B element type is runtime-typed (type_erased_dynamic_float4_t), but kind::mxf4 admits only
E2M1 (MXF4Format has the single value E2M1 = 1), so the decode is e2m1.
Rows n >= tokens_per_expert[g] are a padding region (cp.async zero-fill); that case is not modeled.
"""
import torch
from examples._reference.pytorch_blockscaled import product


def reference(A, B, scale_a, scale_b, fixed_tokens_per_expert_problem_shape, fixed_tokens_per_expert, alpha,
              tokens_0, tokens_1, tokens_2, tokens_3):
    # fixed_*: fixed shape-array tables (read-only kernel inputs fixed by input.json), passed for the
    # binding; not used by the computation.
    max_n = B.shape[1]
    if any(t != max_n for t in (tokens_0, tokens_1, tokens_2, tokens_3)):
        raise NotImplementedError("reference covers tokens_per_expert[g] == max_n only")
    experts = [alpha * product(A[g], B[g], scale_a[g], scale_b[g], packed=True, block=32, e8m0=True)
               for g in range(A.shape[0])]
    return torch.stack(experts).to(torch.bfloat16)
