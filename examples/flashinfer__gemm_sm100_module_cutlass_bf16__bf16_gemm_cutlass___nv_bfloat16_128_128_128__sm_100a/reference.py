"""Mathematical reference; source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/bf16_gemm_cutlass.jinja"""
import torch


def reference(A, B):
    return (A.float() @ B.float().T).to(torch.bfloat16)
