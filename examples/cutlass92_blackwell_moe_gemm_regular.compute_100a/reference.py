"""Mathematical reference; source: https://github.com/NVIDIA/cutlass/blob/e64a9136dd929639e5f7c969fe5af3bf7415cd4f/examples/92_blackwell_moe_gemm/92_blackwell_moe_gemm_regular.cu"""
import torch


def reference(A, B):
    return (A.float() @ B.float().T).to(torch.float16)
