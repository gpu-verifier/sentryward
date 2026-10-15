# Source: https://github.com/NVIDIA/cutlass/blob/0efeafbafc16d5b8b17fd74bbac79674325b38a2/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp
# Source: https://github.com/NVIDIA/cutlass/blob/0efeafbafc16d5b8b17fd74bbac79674325b38a2/python/cutlass_library/generator.py
import torch
from examples._reference.pytorch_models import gemm, attention, mamba, kda


def reference(A, B):
    return gemm(A, B)
