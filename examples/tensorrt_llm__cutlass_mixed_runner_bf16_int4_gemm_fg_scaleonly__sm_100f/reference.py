"""Mathematical reference; source: https://github.com/NVIDIA/TensorRT-LLM/blob/ef567dd9b8919f6ee8b2da1f6f6d0c90206a78f8/cpp/tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/launchers/fpA_intB_launcher_sm100.inl"""
import torch


def reference(A, B, scale):
    packed = B.to(torch.int32)
    low = ((packed & 15) ^ 8) - 8
    high = (((packed >> 4) & 15) ^ 8) - 8
    weights = torch.stack((low, high), dim=-1).reshape(B.shape[0], B.shape[1] * 2).float()
    return (A.float() @ (weights * scale.float()).T).to(torch.bfloat16)
