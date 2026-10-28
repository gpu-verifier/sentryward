"""Mathematical reference; source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_bf16_bf16.cu ; https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/nv_internal/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/gemm/kernel/moe_cutlass_kernel.h"""
import torch


def _rows(row_end_0, row_end_1, row_end_2, row_end_3):
    # Expert e owns rows [last_row[e-1], last_row[e]) (last_row_for_problem, fixed by the input's shape array).
    ends = (row_end_0, row_end_1, row_end_2, row_end_3)
    return list(zip((0,) + ends[:-1], ends))


def reference(activations, expert_weights, row_end_0, row_end_1, row_end_2, row_end_3, table_last_row_for_problem):
    # table_last_row_for_problem: the fixed shape-array table, passed for the binding and not used
    # by the computation; its values enter through row_end_0..3.
    # expert_weights[e] is the ColumnMajor K x N operand stored as [N, K].
    parts = []
    for e, (start, end) in enumerate(_rows(row_end_0, row_end_1, row_end_2, row_end_3)):
        acc = activations[start:end].float() @ expert_weights[e].float().T
        parts.append(torch.clamp(acc, min=0.0))
    return torch.cat(parts).to(torch.bfloat16)
