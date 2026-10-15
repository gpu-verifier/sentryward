"""Mathematical reference; source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/fp8_gemm_cutlass.jinja"""
import torch


def reference(A, B, scale_a, scale_b):
    return ((A.float() @ B.float().T) * scale_a * scale_b).to(torch.bfloat16)
