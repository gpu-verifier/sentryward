"""Mathematical reference; source: https://github.com/HazyResearch/ThunderKittens/blob/1c3920d993404dd49a6d4c7267ea11d583bd5c68/kernels/gemm/bf16_b200/bf16_b200_gemm.cu"""
import torch


def reference(A, B):
    return (A.float() @ B.float().T).to(torch.bfloat16)
