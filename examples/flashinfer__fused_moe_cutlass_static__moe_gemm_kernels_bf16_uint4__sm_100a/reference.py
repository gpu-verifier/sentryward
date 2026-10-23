"""Mathematical reference; source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_bf16_uint4.cu ; https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/nv_internal/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/gemm/kernel/moe_cutlass_kernel.h"""
import torch


def _rows(row_end_0, row_end_1, row_end_2, row_end_3):
    # Expert e owns rows [last_row[e-1], last_row[e]) (last_row_for_problem, fixed by the input's shape array).
    ends = (row_end_0, row_end_1, row_end_2, row_end_3)
    return list(zip((0,) + ends[:-1], ends))


def _weights(expert_weights, experts, k, n):
    # ColumnMajorTileInterleave<64, 4> with the W4_A16 row permutation and nibble interleave: the nibble of
    # W[k][n] is t = (n//4)*4K + (k//64)*256 + (n%4)*64 + 32*((k//32)%2) + 8*((k%8)//2) + 4*(k%2) + (k%32)//8
    # (low nibble of byte t//2 if t is even), stored as w + 8.
    packed = expert_weights.to(torch.int32).reshape(experts, n // 4, k // 64, 4, 2, 4, 2, 2)
    nibbles = torch.stack(((packed & 15) - 8, (packed >> 4) - 8), dim=-1)
    return nibbles.permute(0, 2, 4, 7, 8, 5, 6, 1, 3).reshape(experts, k, n).float()


def reference(activations, expert_weights, weight_scales, row_end_0, row_end_1, row_end_2, row_end_3, table_last_row_for_problem):
    # table_last_row_for_problem: the fixed shape-array table, bound only so the native fixed-input context finds a
    # common identity (fixed_input_context.hpp decode_verified); its values enter through row_end_0..3.
    experts, n = weight_scales.shape
    weights = _weights(expert_weights, experts, activations.shape[1], n)
    parts = []
    for e, (start, end) in enumerate(_rows(row_end_0, row_end_1, row_end_2, row_end_3)):
        acc = activations[start:end].float() @ (weights[e] * weight_scales[e].float()[None, :])
        parts.append(torch.clamp(acc, min=0.0))
    return torch.cat(parts).to(torch.bfloat16)
