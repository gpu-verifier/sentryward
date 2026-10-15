# Source: https://github.com/NVIDIA/TensorRT-LLM/blob/ef567dd9b8919f6ee8b2da1f6f6d0c90206a78f8/cpp/tensorrt_llm/kernels/kdaDecode/kdaDecode.cu
# Source: https://github.com/NVIDIA/TensorRT-LLM/blob/ef567dd9b8919f6ee8b2da1f6f6d0c90206a78f8/cpp/tensorrt_llm/kernels/kdaDecode/kdaDecode.h
import torch
from examples._reference.pytorch_models import gemm, attention, mamba, kda


def reference(x_q, x_k, x_v, w_q, w_k, w_v, bias_q, bias_k, bias_v, cs_q, cs_k, cs_v, a_log, g, dt_bias, beta, onorm_g, onorm_weight, state, lower_bound, scale, epsilon):
    return kda(x_q, x_k, x_v, w_q, w_k, w_v, bias_q, bias_k, bias_v, cs_q, cs_k, cs_v, a_log, g, dt_bias, beta, onorm_g, onorm_weight, state, lower_bound, scale, epsilon)
