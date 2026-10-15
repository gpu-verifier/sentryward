"""CUTLASS 86 scale-only GEMM, in the ideal-real comparison model.

Source: https://github.com/NVIDIA/cutlass/blob/e64a9136dd929639e5f7c969fe5af3bf7415cd4f/examples/86_blackwell_mixed_dtype_gemm/86_blackwell_mixed_dtype.cu
"""
import torch


def reference(A, B, scale):
    packed = B.to(torch.int32)
    low = ((packed & 15) ^ 8) - 8
    high = (((packed >> 4) & 15) ^ 8) - 8
    weights = torch.stack((low, high), dim=-1).reshape(B.shape[0], B.shape[1] * 2)
    return (A.float() @ (weights.float() * scale.float()).T).to(torch.bfloat16)
