# Source: https://github.com/NVIDIA/cutlass/blob/0efeafbafc16d5b8b17fd74bbac79674325b38a2/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp
# Source: https://github.com/NVIDIA/cutlass/blob/0efeafbafc16d5b8b17fd74bbac79674325b38a2/include/cutlass/gemm/kernel/sm100_tile_scheduler_stream_k.hpp
# Source: https://github.com/NVIDIA/cutlass/blob/0efeafbafc16d5b8b17fd74bbac79674325b38a2/python/cutlass_library/generator.py
# Kernel cutlass3x_sm100_tensorop_gemm_f16_f16_f16_f16_f16_64x128x64_0x0x1_0_nnn_align8_stream_k_1sm:
# D = alpha * A @ B^T + beta * C with alpha = 1, beta = 0 (input.json); the split-K fixup only reorders the K sum.
# The reduction workspace is scratch and is not a reference output.
import torch
from examples._reference.pytorch_models import gemm


def reference(A, B):
    return gemm(A, B)
