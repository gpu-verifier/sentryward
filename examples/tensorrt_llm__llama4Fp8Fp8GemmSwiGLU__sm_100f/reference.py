"""Reference for TensorRT-LLM llama4Fp8Fp8GemmSwiGLUPerBlockTemplate.cuh, revision ef567dd9b8919f6ee8b2da1f6f6d0c90206a78f8."""
import torch


def reference(A, B, in_scale, out_scale_inv):
    n = B.shape[0] // 2
    gate = (A.float() @ B[:n].float().T) * in_scale
    linear = A.float() @ B[n:].float().T
    return (gate / (1.0 + torch.exp(-gate)) * linear * in_scale * out_scale_inv).to(torch.float8_e4m3fn)
