# Source: https://github.com/NVIDIA/TensorRT-LLM/blob/ef567dd9b8919f6ee8b2da1f6f6d0c90206a78f8/cpp/tensorrt_llm/kernels/llama4MinLatencyKernels/llama4Bf16Bf16Gemm.cu
# Source: https://github.com/NVIDIA/TensorRT-LLM/blob/ef567dd9b8919f6ee8b2da1f6f6d0c90206a78f8/cpp/tensorrt_llm/kernels/llama4MinLatencyKernels/llama4Utils.cuh
import torch
from examples._reference.pytorch_models import gemm, attention, mamba, kda


def reference(A, B):
    return gemm(A, B)
