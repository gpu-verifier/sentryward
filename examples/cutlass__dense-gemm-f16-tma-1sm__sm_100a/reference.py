"""Mathematical reference; source: https://github.com/NVIDIA/cutlass/blob/0efeafbafc16d5b8b17fd74bbac79674325b38a2/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp"""
import torch


def reference(A, B):
    return (A.float() @ B.float().T).to(torch.float16)
