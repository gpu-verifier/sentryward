"""Mathematical reference; source: https://github.com/vllm-project/vllm/blob/605c3ddcbaaee06a2687a961ae7be51aec5dddc0/csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm100_fp8.cu"""
import torch


def reference(A, B, scale_a, scale_b):
    result = 0.0
    for block in range(A.shape[1] // 128):
        lo, hi = block * 128, (block + 1) * 128
        product = A[:, lo:hi].float() @ B[:, lo:hi].float().T
        result = result + product * scale_a[:, block, None] * scale_b[:, block][None, :]
    return result.to(torch.bfloat16)
