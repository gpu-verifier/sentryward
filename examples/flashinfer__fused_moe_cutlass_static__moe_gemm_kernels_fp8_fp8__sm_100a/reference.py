"""Mathematical reference; sources (pinned):
https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp8_fp8.cu
https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch.h
https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/nv_internal/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/gemm/kernel/moe_cutlass_kernel.h
https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/nv_internal/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/epilogue_helpers.h
https://github.com/NVIDIA/cutlass/blob/b46b16d003484063bca4ed365e44095c4c6ed633/include/cutlass/epilogue/thread/linear_combination_relu.h

MoeFCGemm<MmaPipelined 16x256x128, e4m3 A RowMajor, e4m3 B ColumnMajor, Sm89, LinearCombinationRelu<half, 8, float,
float, ScaleType::Default>> (EpilogueOpDefaultReLU). Expert e owns activation rows
[last_row[e-1], last_row[e]) (last_row_for_problem, fixed by the input's shape array); B[e] is the ColumnMajor K x N
operand stored as [N, K]; D is RowMajor [total_rows, N]. LinearCombinationRelu computes
max(alpha * acc + beta * C, threshold) (NaN-propagating maximum); the input fixes beta = 0 with a null beta_ptr, so the
source term is not read. For the FP8 ReLU tag the launcher builds Params(alpha = 1, beta = biases ? 1 : 0) and does not
attach alpha_scale_ptr_array (only EpilogueOpDefault does), so alpha_ptr is null and threshold is 0.
"""
import torch


def _rows(row_end_0, row_end_1, row_end_2, row_end_3):
    ends = (row_end_0, row_end_1, row_end_2, row_end_3)
    return list(zip((0,) + ends[:-1], ends))


def reference(activations, expert_weights, alpha, threshold, row_end_0, row_end_1, row_end_2, row_end_3, table_last_row_for_problem):
    # table_last_row_for_problem: the fixed shape-array table, bound only so the native fixed-input context finds a
    # common identity (fixed_input_context.hpp decode_verified); its values enter through row_end_0..3.
    parts = []
    for e, (start, end) in enumerate(_rows(row_end_0, row_end_1, row_end_2, row_end_3)):
        acc = activations[start:end].float() @ expert_weights[e].float().T
        parts.append(torch.clamp(alpha * acc, min=threshold))
    return torch.cat(parts).to(torch.float16)
