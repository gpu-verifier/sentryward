"""Mathematical reference; source: https://github.com/vllm-project/vllm/blob/605c3ddcbaaee06a2687a961ae7be51aec5dddc0/csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm100_fp8.cu"""
import torch


def reference(A, B, scale_a, scale_b, bias):
    return ((A.float() @ B.float().T) * scale_a[:, None] * scale_b[None, :] + bias.float()).to(torch.bfloat16)
