"""Mathematical reference; source: https://github.com/deepseek-ai/DeepGEMM/blob/559d79fb6994a58b8a15b4b93bf13ccc16edf247/deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm.cuh"""
import torch


def reference(A, B):
    return (A.float() @ B.float().T).to(torch.bfloat16)
