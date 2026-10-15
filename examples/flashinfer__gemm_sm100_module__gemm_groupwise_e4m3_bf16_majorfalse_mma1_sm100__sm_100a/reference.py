"""Mathematical reference; source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/gemm_groupwise_sm100_kernel_inst.jinja"""
import torch


def reference(A, B, scale_a, scale_b):
    result = 0.0
    for block in range(A.shape[1] // 128):
        lo, hi = block * 128, (block + 1) * 128
        product = A[:, lo:hi].float() @ B[:, lo:hi].float().T
        result = result + product * scale_a[:, block, None] * scale_b[:, block][None, :]
    return result.to(torch.bfloat16)
