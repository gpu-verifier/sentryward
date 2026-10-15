"""Mathematical reference; source: https://github.com/NVIDIA/cutlass/blob/0efeafbafc16d5b8b17fd74bbac79674325b38a2/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized_mma_transform.hpp"""
import torch


def reference(A, B, scale_a, scale_b):
    result = 0.0
    for block in range(A.shape[1] // 128):
        lo, hi = block * 128, (block + 1) * 128
        product = A[:, lo:hi].float() @ B[:, lo:hi].float().T
        result = result + product * scale_a[:, block, None] * scale_b[:, block][None, :]
    return result.to(torch.float16)
